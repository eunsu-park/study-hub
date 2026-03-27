"""
CME Propagation: Drag-Based Model with Ensemble Forecasting.

Demonstrates:
- Drag-based equation of motion for interplanetary CME propagation
- Dependence on initial speed: fast CMEs decelerate, slow CMEs accelerate
- Monte Carlo ensemble to quantify arrival time uncertainty
- Statistical arrival time distributions at 1 AU

Physics:
    After eruption, a CME propagates through the heliosphere and interacts
    with the ambient solar wind. The aerodynamic drag force decelerates fast
    CMEs and accelerates slow CMEs toward the solar wind speed:
        dv/dt = -gamma * (v - v_sw) * |v - v_sw|
    where gamma = C_d * A * rho_sw / M_cme depends on the CME cross-section,
    mass, and the solar wind density rho_sw ~ r^{-2}. This drag-based model
    (Vrsnak et al. 2013) is widely used for space weather forecasting.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Physical Constants and Parameters ===
R_sun = 6.957e8    # solar radius (m)
AU = 1.496e11      # astronomical unit (m)

# Solar wind parameters
v_sw_default = 400e3  # ambient solar wind speed (m/s)
rho_0 = 6e-21         # proton density at r_0 ~ 20 R_sun (kg/m^3)
r_0 = 20 * R_sun      # initial heliocentric distance

# CME parameters
M_cme = 1e13          # CME mass (kg), typical ~10^12-10^13 kg
omega_half = np.radians(30)  # CME half-angular width
C_d = 1.0              # drag coefficient


def drag_ode(t, y, gamma_0, v_sw, r_ref):
    """
    Drag-based ODE system.
    y = [r, v] where r is heliocentric distance, v is radial speed.

    The drag parameter gamma varies with distance because rho_sw ~ r^{-2}:
        gamma(r) = gamma_0 * (r_ref / r)^2
    """
    r, v = y
    # Drag parameter decreases with distance (density drops as r^-2)
    gamma = gamma_0 * (r_ref / r) ** 2
    dv = v  # dr/dt = v
    da = -gamma * (v - v_sw) * abs(v - v_sw)  # dv/dt
    return [dv, da]


def propagate_cme(v_0, v_sw, gamma_0, r_start, r_end, t_max=5 * 86400):
    """
    Propagate CME from r_start to r_end using drag-based model.
    Returns time of arrival and velocity at r_end.
    """
    def event_arrival(t, y, *args):
        return y[0] - r_end
    event_arrival.terminal = True
    event_arrival.direction = 1

    sol = solve_ivp(
        drag_ode, [0, t_max], [r_start, v_0],
        args=(gamma_0, v_sw, r_start),
        events=event_arrival,
        max_step=3600,  # 1 hour max step
        rtol=1e-8, atol=1e-6,
        dense_output=True
    )
    return sol


# === Compute gamma_0 from physical parameters ===
# Cross-sectional area at r_0: A = pi * (r_0 * tan(omega))^2
A_cme = np.pi * (r_0 * np.tan(omega_half)) ** 2
gamma_0 = C_d * A_cme * rho_0 / M_cme

print("=" * 60)
print("CME DRAG-BASED PROPAGATION MODEL")
print("=" * 60)
print(f"CME mass: {M_cme:.1e} kg")
print(f"Angular half-width: {np.degrees(omega_half):.0f} deg")
print(f"Drag parameter gamma_0: {gamma_0:.2e} /km")
print(f"Solar wind speed: {v_sw_default / 1e3:.0f} km/s")
print(f"Initial distance: {r_0 / R_sun:.0f} R_sun")

# === Three test cases: fast, medium, slow CME ===
cases = {
    'Fast (2000 km/s)':   2000e3,
    'Medium (1000 km/s)': 1000e3,
    'Slow (300 km/s)':     300e3,
}

print("\n--- Single-run Results ---")
solutions = {}
for label, v_0 in cases.items():
    sol = propagate_cme(v_0, v_sw_default, gamma_0, r_0, AU)
    if sol.t_events[0].size > 0:
        t_arr = sol.t_events[0][0] / 3600  # hours
        v_arr = sol.y_events[0][0][1] / 1e3  # km/s
        print(f"  {label}: arrival = {t_arr:.1f} h ({t_arr / 24:.1f} days), "
              f"v_1AU = {v_arr:.0f} km/s")
    else:
        t_arr = np.nan
        v_arr = np.nan
        print(f"  {label}: did not reach 1 AU within time limit")
    solutions[label] = sol

# === Monte Carlo Ensemble Forecasting ===
print("\n--- Ensemble Forecasting (100 runs each, +/-20% variation) ---")
np.random.seed(42)
N_ensemble = 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {'Fast (2000 km/s)': 'red', 'Medium (1000 km/s)': 'blue',
          'Slow (300 km/s)': 'green'}

arrival_times = {}

for label, v_0_nom in cases.items():
    t_arrivals = []
    v_arrivals = []

    for _ in range(N_ensemble):
        # Perturb parameters by +/-20%
        v_0 = v_0_nom * np.random.uniform(0.8, 1.2)
        v_sw = v_sw_default * np.random.uniform(0.8, 1.2)
        gamma_pert = gamma_0 * np.random.uniform(0.8, 1.2)

        sol = propagate_cme(v_0, v_sw, gamma_pert, r_0, AU)
        if sol.t_events[0].size > 0:
            t_arrivals.append(sol.t_events[0][0] / 3600)  # hours
            v_arrivals.append(sol.y_events[0][0][1] / 1e3)

    t_arr_arr = np.array(t_arrivals)
    arrival_times[label] = t_arr_arr
    print(f"  {label}: {np.mean(t_arr_arr):.1f} +/- {np.std(t_arr_arr):.1f} h "
          f"({np.mean(t_arr_arr) / 24:.1f} +/- {np.std(t_arr_arr) / 24:.1f} days)")

# === Plotting ===

# 1. v(r) profiles
ax = axes[0, 0]
r_plot = np.linspace(r_0, AU, 500)
for label, sol in solutions.items():
    t_dense = np.linspace(sol.t[0], sol.t[-1], 500)
    y_dense = sol.sol(t_dense)
    r_dense = y_dense[0] / AU  # in AU
    v_dense = y_dense[1] / 1e3  # km/s
    ax.plot(r_dense, v_dense, color=colors[label], linewidth=2, label=label)
ax.axhline(v_sw_default / 1e3, color='gray', linestyle='--', alpha=0.7,
           label=f'Solar wind ({v_sw_default / 1e3:.0f} km/s)')
ax.set_xlabel('Heliocentric Distance (AU)')
ax.set_ylabel('CME Speed (km/s)')
ax.set_title('CME Speed vs Distance')
ax.legend(fontsize=8)
ax.set_xlim(0, 1.1)

# 2. r(t) profiles
ax = axes[0, 1]
for label, sol in solutions.items():
    t_hr = sol.t / 3600
    r_au = sol.y[0] / AU
    ax.plot(t_hr, r_au, color=colors[label], linewidth=2, label=label)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='1 AU')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Distance (AU)')
ax.set_title('CME Distance vs Time')
ax.legend(fontsize=8)

# 3. Arrival time distributions (histogram)
ax = axes[1, 0]
for label in cases:
    if label in arrival_times:
        ax.hist(arrival_times[label] / 24, bins=20, alpha=0.5,
                color=colors[label], label=label, edgecolor='black',
                linewidth=0.5)
ax.set_xlabel('Arrival Time (days)')
ax.set_ylabel('Count')
ax.set_title('Ensemble Arrival Time Distribution (N=100)')
ax.legend(fontsize=8)

# 4. Fast vs slow comparison: acceleration/deceleration
ax = axes[1, 1]
for label, sol in solutions.items():
    t_dense = np.linspace(sol.t[0], sol.t[-1], 500)
    y_dense = sol.sol(t_dense)
    v_dense = y_dense[1] / 1e3
    # Compute acceleration
    accel = np.gradient(v_dense * 1e3, t_dense) / 1e3  # km/s^2
    r_dense = y_dense[0] / AU
    ax.plot(r_dense, accel * 3600, color=colors[label], linewidth=2, label=label)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Heliocentric Distance (AU)')
ax.set_ylabel('Acceleration (km/s/h)')
ax.set_title('CME Acceleration (>0: speeding up, <0: slowing down)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Solar_Physics/09_cme_propagation.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to 09_cme_propagation.png")
print("\nKey insight: Fast CMEs decelerate toward v_sw, slow CMEs accelerate toward v_sw.")

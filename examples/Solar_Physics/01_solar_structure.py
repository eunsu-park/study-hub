"""
Standard Solar Model: Radial Profiles of Temperature, Density, Pressure.

Demonstrates the internal structure of the Sun using a polytropic model
with index n=3 (Eddington's standard model for radiative interiors).
The Lane-Emden equation is solved numerically and scaled to solar values.

Key physics:
  - Hydrostatic equilibrium: dP/dr = -G*M(r)*rho(r)/r^2
  - Polytropic relation: P = K * rho^(1 + 1/n)
  - Lane-Emden equation: (1/xi^2) d/dxi (xi^2 dtheta/dxi) = -theta^n
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical constants ---
G = 6.674e-11        # gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30     # solar mass [kg]
R_sun = 6.957e8      # solar radius [m]
L_sun = 3.828e26     # solar luminosity [W]
k_B = 1.381e-23      # Boltzmann constant [J/K]
m_p = 1.673e-27      # proton mass [kg]
mu = 0.62            # mean molecular weight (fully ionized solar composition)

# --- Solve the Lane-Emden equation for n=3 ---
# Dimensionless form: d^2(theta)/d(xi)^2 + (2/xi)*d(theta)/d(xi) + theta^n = 0
# Let y1 = theta, y2 = d(theta)/d(xi)
# BCs: theta(0)=1, theta'(0)=0

n_poly = 3  # polytropic index for radiative interior

def lane_emden(xi, y):
    """RHS of Lane-Emden equation as a first-order system."""
    theta, dtheta = y
    if xi < 1e-10:
        # L'Hopital near origin: d^2theta/dxi^2 = -1/3 for theta=1
        return [dtheta, -1.0 / 3.0]
    # Prevent negative theta (unphysical)
    theta_safe = max(theta, 0.0)
    ddtheta = -2.0 * dtheta / xi - theta_safe ** n_poly
    return [dtheta, ddtheta]

# Integrate outward until theta crosses zero
xi_span = (1e-6, 20.0)
y0 = [1.0 - (1e-6)**2 / 6.0, -1e-6 / 3.0]  # Taylor expansion near origin

def theta_zero(xi, y):
    """Event: theta = 0 (stellar surface)."""
    return y[0]
theta_zero.terminal = True
theta_zero.direction = -1

sol = solve_ivp(lane_emden, xi_span, y0, events=theta_zero,
                max_step=0.01, dense_output=True)

xi_1 = sol.t_events[0][0]       # first zero of theta
dtheta_xi1 = sol.y_events[0][0][1]  # theta'(xi_1)

print("=" * 60)
print("Lane-Emden Solution for n = 3")
print(f"  xi_1 (surface)        = {xi_1:.4f}")
print(f"  -xi_1^2 * theta'(xi_1)= {-xi_1**2 * dtheta_xi1:.4f}")
print("  (Exact: xi_1 = 6.8969, -xi^2 theta' = 2.0182)")
print("=" * 60)

# --- Scale to solar values ---
# Central density and pressure from polytropic relations
rho_c = -M_sun * xi_1 / (4 * np.pi * R_sun**3 * dtheta_xi1)
alpha = R_sun / xi_1  # length scale: r = alpha * xi
P_c = (4 * np.pi * G * alpha**2 * rho_c**2) / (n_poly + 1)
T_c = P_c * mu * m_p / (rho_c * k_B)

print(f"\nCentral values (scaled to Sun):")
print(f"  rho_c = {rho_c:.3e} kg/m^3  (observed: ~1.5e5)")
print(f"  P_c   = {P_c:.3e} Pa       (observed: ~2.5e16)")
print(f"  T_c   = {T_c:.3e} K        (observed: ~1.57e7)")

# --- Build radial profiles ---
xi_arr = np.linspace(1e-4, xi_1 * 0.999, 500)
theta_arr = sol.sol(xi_arr)[0]
theta_arr = np.maximum(theta_arr, 0)

r_frac = xi_arr / xi_1                          # r / R_sun
rho_arr = rho_c * theta_arr ** n_poly            # density profile
P_arr = P_c * theta_arr ** (n_poly + 1)          # pressure profile
T_arr = T_c * theta_arr                          # temperature profile

# Luminosity profile: approximate L(r) using mass fraction
# M(r)/M_sun ~ -( xi^2 theta'(xi) ) / ( xi_1^2 |theta'(xi_1)| )
dtheta_arr = sol.sol(xi_arr)[1]
M_frac = -(xi_arr**2 * dtheta_arr) / (xi_1**2 * abs(dtheta_xi1))
# Simple approximation: energy generation concentrated in core
# L(r)/L_sun ~ (M(r)/M_sun)^5 is a rough fit
L_frac = np.clip(M_frac, 0, 1) ** 1.5  # gentler for display

# --- Gravitational and thermal timescale ---
E_grav = 3 * G * M_sun**2 / (5 * R_sun)  # Kelvin-Helmholtz estimate
tau_KH = E_grav / L_sun / (3.156e7)       # in years
print(f"\nGravitational energy:    E_grav = {E_grav:.3e} J")
print(f"Kelvin-Helmholtz time:   tau_KH = {tau_KH:.3e} years (~1.57e7)")

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Standard Solar Model (Polytropic n=3)", fontsize=14, y=0.98)

# Boundary markers
r_core = 0.25   # core boundary
r_rcb = 0.71    # radiative-convective boundary

for ax in axes.flat:
    ax.axvline(r_core, color='orange', ls='--', alpha=0.5, label='Core (0.25)')
    ax.axvline(r_rcb, color='red', ls='--', alpha=0.5, label='RCB (0.71)')

# Temperature
axes[0, 0].plot(r_frac, T_arr / 1e6, 'r-', lw=2)
axes[0, 0].set_ylabel("Temperature [MK]")
axes[0, 0].set_title("Temperature T(r)")
axes[0, 0].set_xlim(0, 1)

# Density (log scale)
axes[0, 1].semilogy(r_frac, rho_arr, 'b-', lw=2)
axes[0, 1].set_ylabel(r"Density [kg/m$^3$]")
axes[0, 1].set_title(r"Density $\rho(r)$")
axes[0, 1].set_xlim(0, 1)

# Pressure (log scale)
axes[1, 0].semilogy(r_frac, P_arr, 'g-', lw=2)
axes[1, 0].set_ylabel("Pressure [Pa]")
axes[1, 0].set_title("Pressure P(r)")
axes[1, 0].set_xlim(0, 1)

# Luminosity and mass fraction
axes[1, 1].plot(r_frac, L_frac, 'm-', lw=2, label='L(r)/L_sun')
axes[1, 1].plot(r_frac, M_frac, 'k--', lw=2, label='M(r)/M_sun')
axes[1, 1].set_ylabel("Fraction")
axes[1, 1].set_title("Luminosity & Mass Fraction")
axes[1, 1].legend(loc='lower right')
axes[1, 1].set_xlim(0, 1)

for ax in axes.flat:
    ax.set_xlabel(r"$r / R_\odot$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Solar_Physics/01_solar_structure.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: 01_solar_structure.png")

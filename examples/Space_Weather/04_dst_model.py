"""
Geomagnetic Storm Modeling: Burton Equation and O'Brien-McPherron Extension.

Solves the Burton et al. (1975) ordinary differential equation for the Dst
index, modeling the ring current injection and decay during a geomagnetic
storm. The O'Brien-McPherron (2000) extension introduces Dst-dependent
decay time and electric-field-dependent injection.

Key physics:
  - Burton equation: dDst*/dt = Q(t) - Dst*/tau
    where Dst* = pressure-corrected Dst = Dst - b*sqrt(P_dyn) + c
  - Injection function Q:
    Q = -4.4*(VBs - 0.5) [nT/h] for VBs > 0.5, else 0
    where VBs = V_sw * max(-Bz, 0) / 1000 [mV/m]
  - Decay time tau ~ 7.7 hours (Burton) or tau(Dst) (O'Brien-McPherron)
  - O'Brien-McPherron: tau = 2.4*exp(9.74/(4.69 + E_y)) hours
    Q = -4.4*(E_y - 0.49) for E_y > 0.49
  - Storm phases: sudden commencement (SC), main phase, recovery phase
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical constants ---
R_E = 6.371e6  # Earth radius [m]

print("=" * 65)
print("Geomagnetic Storm Model: Burton & O'Brien-McPherron Equations")
print("=" * 65)


# =========================================================================
# 1. GENERATE SYNTHETIC CME-DRIVEN STORM
# =========================================================================
# Timeline (hours):
#   0-10:  Quiet pre-storm (V~400, Bz~+2, n~5)
#   10-11: Sudden commencement (shock: V jumps, n spikes, P_dyn up)
#   11-20: Main phase (strong southward Bz, elevated V and n)
#   20-24: Late main phase (Bz starts rotating northward)
#   24-72: Recovery phase (Bz north, V decreasing)

np.random.seed(123)
dt = 0.05  # time step [hours] (3 minutes)
t = np.arange(0, 72, dt)
N = len(t)

# Solar wind speed [km/s]
V_sw = np.full(N, 400.0)
V_sw[t >= 10] = 400 + 350 * (1 - np.exp(-(t[t >= 10] - 10) / 1.0))
V_sw[t >= 24] = V_sw[int(24/dt)] * np.exp(-(t[t >= 24] - 24) / 40)
V_sw = np.clip(V_sw + 10 * np.random.randn(N), 300, 850)

# IMF Bz [nT]
Bz = np.full(N, 2.0)
# Main phase: strong southward turning
main_mask = (t >= 11) & (t < 20)
Bz[main_mask] = -20 * np.sin(np.pi * (t[main_mask] - 11) / 18)
# Late main phase: rotation toward north
late_mask = (t >= 20) & (t < 24)
Bz[late_mask] = -20 * np.sin(np.pi * (20 - 11) / 18) * np.exp(-(t[late_mask] - 20) / 3)
# Recovery: weakly northward
Bz[t >= 24] = 2.0 * (1 - np.exp(-(t[t >= 24] - 24) / 5))
Bz += 1.5 * np.random.randn(N)

# Proton density [cm^-3]
n_sw = np.full(N, 5.0)
n_sw[(t >= 10) & (t < 12)] = 30.0  # shock compression
n_sw[(t >= 12) & (t < 20)] = 12.0  # elevated sheath
n_sw[t >= 20] = 5.0 + 7.0 * np.exp(-(t[t >= 20] - 20) / 8)
n_sw += 1.0 * np.abs(np.random.randn(N))
n_sw = np.clip(n_sw, 1.0, 50.0)

# Dynamic pressure [nPa] = 1.67e-6 * n[cm^-3] * V[km/s]^2
P_dyn = 1.67e-6 * n_sw * V_sw**2

# Derived quantities
VBs = V_sw * np.maximum(-Bz, 0) / 1000.0  # dawn-dusk electric field proxy [mV/m]
E_y = V_sw * np.maximum(-Bz, 0) / 1000.0  # same as VBs for this context

print(f"\nSynthetic storm parameters:")
print(f"  Peak V_sw  = {V_sw.max():.0f} km/s")
print(f"  Min Bz     = {Bz.min():.1f} nT")
print(f"  Max P_dyn  = {P_dyn.max():.1f} nPa")
print(f"  Max E_y    = {E_y.max():.2f} mV/m")


# =========================================================================
# 2. BURTON MODEL (1975)
# =========================================================================
# dDst*/dt = Q(t) - Dst*/tau
# Dst* = Dst - b*sqrt(P_dyn) + c (pressure correction)
# b = 7.26 nT/sqrt(nPa), c = 11 nT (typical values)
# Q = -4.4*(VBs - 0.5) for VBs > 0.5, else 0  [nT/h]
# tau = 7.7 hours

b_press = 7.26     # pressure correction coefficient [nT/sqrt(nPa)]
c_press = 11.0     # pressure correction offset [nT]
tau_burton = 7.7   # decay time [hours]


def burton_Q(VBs_val):
    """Burton injection function Q [nT/h]."""
    if VBs_val > 0.5:
        return -4.4 * (VBs_val - 0.5)
    return 0.0


def burton_rhs(t_val, Dst_star, VBs_interp):
    """RHS of Burton equation for solve_ivp.

    We interpolate VBs from the synthetic data at the current time.
    """
    idx = int(t_val / dt)
    idx = min(idx, N - 1)
    Q = burton_Q(VBs_interp[idx])
    return [Q - Dst_star[0] / tau_burton]


# Solve with scipy
sol_burton = solve_ivp(
    burton_rhs, [0, 72], [0.0], args=(VBs,),
    t_eval=t, method='RK45', max_step=0.1
)
Dst_star_burton = sol_burton.y[0]

# Convert Dst* to observed Dst: Dst = Dst* + b*sqrt(P_dyn) - c
Dst_burton = Dst_star_burton + b_press * np.sqrt(P_dyn) - c_press

print(f"\nBurton Model Results:")
print(f"  Min Dst*   = {Dst_star_burton.min():.1f} nT")
print(f"  Min Dst    = {Dst_burton.min():.1f} nT")
print(f"  Time of min Dst = {t[np.argmin(Dst_burton)]:.1f} h")


# =========================================================================
# 3. O'BRIEN-McPHERRON MODEL (2000)
# =========================================================================
# Extensions:
#   tau(E_y) = 2.4 * exp(9.74 / (4.69 + E_y)) hours
#   Q = -4.4 * (E_y - 0.49) for E_y > 0.49, else 0  [nT/h]
# The decay time decreases during strong driving (faster ring current loss).

def obrien_tau(E_y_val):
    """O'Brien-McPherron Dst-dependent decay time [hours]."""
    return 2.4 * np.exp(9.74 / (4.69 + E_y_val))


def obrien_Q(E_y_val):
    """O'Brien-McPherron injection function [nT/h]."""
    if E_y_val > 0.49:
        return -4.4 * (E_y_val - 0.49)
    return 0.0


def obrien_rhs(t_val, Dst_star, E_y_interp):
    """RHS of O'Brien-McPherron equation."""
    idx = int(t_val / dt)
    idx = min(idx, N - 1)
    ey = E_y_interp[idx]
    Q = obrien_Q(ey)
    tau = obrien_tau(ey)
    return [Q - Dst_star[0] / tau]


sol_obrien = solve_ivp(
    obrien_rhs, [0, 72], [0.0], args=(E_y,),
    t_eval=t, method='RK45', max_step=0.1
)
Dst_star_obrien = sol_obrien.y[0]
Dst_obrien = Dst_star_obrien + b_press * np.sqrt(P_dyn) - c_press

print(f"\nO'Brien-McPherron Model Results:")
print(f"  Min Dst*   = {Dst_star_obrien.min():.1f} nT")
print(f"  Min Dst    = {Dst_obrien.min():.1f} nT")
print(f"  Time of min Dst = {t[np.argmin(Dst_obrien)]:.1f} h")

# Compute tau variation during the storm
tau_profile = np.array([obrien_tau(ey) for ey in E_y])
print(f"  Min tau    = {tau_profile.min():.1f} h (during strong driving)")
print(f"  Max tau    = {tau_profile[t < 10].mean():.1f} h (quiet time)")


# =========================================================================
# 4. GENERATE SYNTHETIC "OBSERVED" Dst
# =========================================================================
# Add noise and slight offset to O'Brien model as "observed" reference
Dst_observed = Dst_obrien + 5 * np.random.randn(N) + 3
# Smooth slightly to look realistic
from scipy.ndimage import uniform_filter1d
Dst_observed = uniform_filter1d(Dst_observed, size=10)


# =========================================================================
# 5. PLOTTING
# =========================================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle("Geomagnetic Storm Model: Burton & O'Brien-McPherron Equations",
             fontsize=14, y=0.98)

# Storm phase shading
phase_colors = [
    (0, 10, 'white', 'Quiet'),
    (10, 11, 'yellow', 'SC'),
    (11, 20, 'salmon', 'Main Phase'),
    (20, 24, 'lightsalmon', 'Late Main'),
    (24, 72, 'lightgreen', 'Recovery')
]
for ax in axes:
    for t0, t1, color, label in phase_colors:
        ax.axvspan(t0, t1, color=color, alpha=0.15)

# --- Panel 1: Solar wind inputs ---
ax = axes[0]
ax2 = ax.twinx()
l1, = ax.plot(t, V_sw, 'b-', lw=1, label='V [km/s]')
l2, = ax2.plot(t, Bz, 'r-', lw=1, label='$B_z$ [nT]')
ax2.axhline(0, color='gray', ls=':', lw=0.5)
ax.set_ylabel("V [km/s]", color='b', fontsize=11)
ax2.set_ylabel("$B_z$ [nT]", color='r', fontsize=11)
ax.set_title("Solar Wind Drivers", fontsize=12)
ax.legend(handles=[l1, l2], fontsize=8, loc='upper right')

# Phase labels
for t0, t1, color, label in phase_colors:
    mid = (t0 + t1) / 2
    if label != 'white':
        axes[0].text(mid, V_sw.max() * 0.98, label, fontsize=8,
                     ha='center', va='top', style='italic')

# --- Panel 2: VBs and dynamic pressure ---
ax = axes[1]
ax2 = ax.twinx()
l1, = ax.plot(t, E_y, 'darkorange', lw=1, label='$E_y$ = $VB_s$ [mV/m]')
l2, = ax2.plot(t, P_dyn, 'green', lw=1, label='$P_{dyn}$ [nPa]')
ax.set_ylabel("$E_y$ [mV/m]", color='darkorange', fontsize=11)
ax2.set_ylabel("$P_{dyn}$ [nPa]", color='green', fontsize=11)
ax.set_title("Coupling Parameters", fontsize=12)
ax.legend(handles=[l1, l2], fontsize=8, loc='upper right')

# --- Panel 3: Decay time comparison ---
ax = axes[2]
ax.plot(t, np.full(N, tau_burton), 'b--', lw=1.5, label=f'Burton $\\tau$ = {tau_burton} h (constant)')
ax.plot(t, tau_profile, 'r-', lw=1.5, label="O'Brien-McPherron $\\tau(E_y)$")
ax.set_ylabel(r"$\tau$ [hours]", fontsize=11)
ax.set_title("Ring Current Decay Time", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 20)

# --- Panel 4: Dst comparison ---
ax = axes[3]
ax.plot(t, Dst_observed, 'k-', lw=1.5, alpha=0.5, label='Synthetic "observed"')
ax.plot(t, Dst_burton, 'b-', lw=1.5, label='Burton (1975)')
ax.plot(t, Dst_obrien, 'r-', lw=1.5, label="O'Brien-McPherron (2000)")

# Storm severity thresholds
ax.axhline(-30, color='gold', ls=':', alpha=0.5, label='Weak storm (-30 nT)')
ax.axhline(-50, color='orange', ls=':', alpha=0.5, label='Moderate storm (-50 nT)')
ax.axhline(-100, color='red', ls=':', alpha=0.5, label='Intense storm (-100 nT)')

ax.set_ylabel("Dst [nT]", fontsize=11)
ax.set_xlabel("Time [hours]", fontsize=11)
ax.set_title("Dst Index: Model Comparison", fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Space_Weather/04_dst_model.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: 04_dst_model.png")

"""
Helioseismology: p-mode Frequencies and l-nu Diagram.

Demonstrates solar oscillation analysis using the asymptotic theory of
acoustic (p-mode) oscillations. The Sun rings like a bell with millions
of resonant modes characterized by (n, l, m) quantum numbers.

Key physics:
  - Sound speed profile c(r) determines mode frequencies
  - Asymptotic formula: nu_nl ~ Delta_nu * (n + l/2 + epsilon)
  - Large separation Delta_nu ~ 135 microHz (sound travel time)
  - Small separation delta_nu ~ 9 microHz (core sensitivity)
  - Acoustic ray turning point: c(r_t)/r_t = omega/sqrt(l(l+1))
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --- Solar parameters ---
R_sun = 6.957e8   # solar radius [m]
M_sun = 1.989e30  # solar mass [kg]
G = 6.674e-11     # gravitational constant

# --- Sound speed profile (simplified model) ---
# c(r) based on polytropic n=3 model, roughly matching helioseismic inversions
# c_center ~ 500 km/s, c_surface ~ 10 km/s
def sound_speed(r_frac):
    """Sound speed as a function of r/R_sun (simplified profile).
    Returns c in m/s. Based on fit to standard solar model."""
    x = r_frac
    # Polynomial fit to solar sound speed profile
    c = 5.0e5 * (1 - 0.6 * x**2 - 0.35 * x**4 + 0.05 * x**6)
    c = np.maximum(c, 1e4)  # floor near surface
    return c

r_grid = np.linspace(0.001, 0.999, 1000)
c_grid = sound_speed(r_grid)

# --- Acoustic travel time and large separation ---
# Delta_nu = 1 / (2 * integral dr/c from 0 to R)
dr = (r_grid[1] - r_grid[0]) * R_sun
tau_acoustic = np.sum(dr / c_grid)  # one-way travel time [s]
Delta_nu = 1.0 / (2.0 * tau_acoustic) * 1e6  # microHz

print("=" * 60)
print("Helioseismology: Asymptotic p-mode Analysis")
print(f"  Acoustic travel time (one-way): {tau_acoustic:.0f} s")
print(f"  Large separation Delta_nu: {Delta_nu:.1f} microHz (observed: ~135)")
print("=" * 60)

# --- Generate p-mode frequencies using asymptotic formula ---
# nu_nl = Delta_nu_obs * (n + l/2 + epsilon_0) - D_0 * l*(l+1) / nu_nl
# Simplified: nu_nl ~ Delta_nu * (n + l/2 + epsilon_0)
Delta_nu_obs = 135.0  # observed large separation [microHz]
epsilon_0 = 1.5       # phase offset
D_0 = 1.5             # small separation coefficient

l_max = 200
n_max = 30

l_values = np.arange(0, l_max + 1)
n_values = np.arange(1, n_max + 1)

# Create mode frequency grid
freqs = {}
for l in l_values:
    for n in n_values:
        nu = Delta_nu_obs * (n + l / 2.0 + epsilon_0)
        # Second-order correction (small separation)
        if nu > 0:
            nu -= D_0 * l * (l + 1) / nu * Delta_nu_obs
        if 500 < nu < 5500:  # observable range
            freqs[(n, l)] = nu

print(f"\nGenerated {len(freqs)} p-modes in range 500-5500 microHz")

# --- Large and small separations for low-degree modes ---
print("\nFrequency separations for low-l modes:")
print(f"{'n':>3} {'nu_0':>10} {'nu_1':>10} {'nu_2':>10} {'Delta_nu':>10} {'delta_02':>10}")
for n in range(10, 26):
    if (n, 0) in freqs and (n, 1) in freqs and (n, 2) in freqs:
        nu0 = freqs[(n, 0)]
        nu1 = freqs[(n, 1)]
        nu2 = freqs[(n, 2)]
        Dnu = freqs[(n + 1, 0)] - nu0 if (n + 1, 0) in freqs else 0
        dnu02 = nu0 - freqs[(n - 1, 2)] if (n - 1, 2) in freqs else 0
        print(f"{n:3d} {nu0:10.1f} {nu1:10.1f} {nu2:10.1f} {Dnu:10.1f} {dnu02:10.1f}")

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Helioseismology: Solar Oscillation Analysis", fontsize=14, y=0.98)

# Panel 1: l-nu diagram (power spectrum ridges)
ax = axes[0, 0]
l_arr = []
nu_arr = []
for (n, l), nu in freqs.items():
    l_arr.append(l)
    nu_arr.append(nu)
ax.scatter(l_arr, nu_arr, s=0.3, c='navy', alpha=0.5)
ax.set_xlabel("Spherical degree l")
ax.set_ylabel(r"Frequency [$\mu$Hz]")
ax.set_title(r"$\ell$-$\nu$ Diagram (p-mode Ridges)")
ax.set_xlim(0, l_max)
ax.set_ylim(500, 5500)
ax.grid(True, alpha=0.3)

# Panel 2: Sound speed profile
ax = axes[0, 1]
ax.plot(r_grid, c_grid / 1e3, 'b-', lw=2)
ax.set_xlabel(r"$r / R_\odot$")
ax.set_ylabel("Sound speed [km/s]")
ax.set_title("Sound Speed Profile c(r)")
ax.axvline(0.25, color='orange', ls='--', alpha=0.5, label='Core boundary')
ax.axvline(0.71, color='red', ls='--', alpha=0.5, label='Conv. zone base')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Acoustic ray paths
ax = axes[1, 0]
# Turning point: c(r_t)/r_t = c(R)/R * sqrt(l(l+1))/omega ~ horizontal phase speed
# For display, trace rays for several l values
theta_grid = np.linspace(0, 2 * np.pi, 500)

for l_ray in [1, 5, 20, 75, 150]:
    # Find turning point where c(r)/r = c(R_sun)/(R_sun) * (some function of l)
    # Simplified: r_t/R_sun ~ (l / l_max)^0.5 for display
    w = c_grid / (r_grid * R_sun)  # omega/sqrt(l(l+1)) threshold
    w_surface = c_grid[-1] / (r_grid[-1] * R_sun)
    # Turning point where c/r = w_target
    w_target = w_surface * np.sqrt(l_ray * (l_ray + 1)) / (2 * np.pi * 3000e-6)
    valid = w > w_target
    if np.any(valid):
        r_turn = r_grid[np.argmin(np.abs(w - w_target))]
    else:
        r_turn = 0.01

    # Draw schematic ray path (simplified as elliptical arcs)
    n_bounces = min(l_ray, 8)
    if n_bounces == 0:
        n_bounces = 1
    arc_angle = np.pi / n_bounces
    for i in range(n_bounces):
        theta_start = i * 2 * arc_angle
        theta_arc = np.linspace(theta_start, theta_start + 2 * arc_angle, 100)
        # Ray dips to r_turn at midpoint
        r_ray = 1.0 - (1.0 - r_turn) * np.sin(
            np.pi * (theta_arc - theta_start) / (2 * arc_angle)
        )**2
        x = r_ray * np.cos(theta_arc)
        y = r_ray * np.sin(theta_arc)
        ax.plot(x, y, lw=1.0, alpha=0.7, label=f'l={l_ray}' if i == 0 else '')

# Draw solar disk
circle = plt.Circle((0, 0), 1, fill=False, color='black', lw=2)
ax.add_patch(circle)
circle_core = plt.Circle((0, 0), 0.25, fill=True, color='yellow', alpha=0.2)
ax.add_patch(circle_core)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title("Acoustic Ray Paths (schematic)")
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

# Panel 4: Echelle diagram (nu mod Delta_nu vs nu)
ax = axes[1, 1]
for l_plot in [0, 1, 2, 3]:
    l_freqs = [(n, nu) for (n, l), nu in freqs.items() if l == l_plot and nu < 4500]
    if l_freqs:
        ns, nus = zip(*l_freqs)
        nu_mod = np.array(nus) % Delta_nu_obs
        ax.scatter(nu_mod, nus, s=15, label=f'l = {l_plot}', zorder=3)

ax.set_xlabel(r"$\nu$ mod $\Delta\nu$ [$\mu$Hz]")
ax.set_ylabel(r"Frequency [$\mu$Hz]")
ax.set_title(fr"Echelle Diagram ($\Delta\nu$ = {Delta_nu_obs} $\mu$Hz)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Solar_Physics/03_helioseismology.png",
            dpi=150, bbox_inches='tight')
plt.show()

# --- Derived sound speed at key depths ---
print("\nDerived sound speed at key depths:")
for r_val in [0.0, 0.1, 0.25, 0.5, 0.71, 0.9, 0.99]:
    print(f"  c(r={r_val:.2f} R_sun) = {sound_speed(r_val)/1e3:.1f} km/s")

print("\nPlot saved: 03_helioseismology.png")

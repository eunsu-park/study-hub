"""
Focused Transport Equation: 1D SEP Propagation Along Parker Spiral.

Demonstrates:
- Focused transport of solar energetic particles (SEPs) along the Parker spiral
- Adiabatic focusing in diverging magnetic field (focusing length L)
- Pitch-angle scattering (diffusion in mu-space)
- Velocity dispersion: faster particles arrive first at 1 AU
- Scatter-free vs diffusive transport regimes
- Onset time analysis to determine path length

Physics:
    The focused transport equation (Roelof 1969, Ruffolo 1995) describes SEP
    propagation along the interplanetary magnetic field:

        df/dt + v*mu*df/ds + (1-mu^2)/(2L)*v*df/dmu = D_mumu * d^2f/dmu^2

    where:
        f(s, mu, t) = particle distribution function
        s = distance along field line (Parker spiral)
        mu = cos(pitch angle) = v_parallel / v
        L(s) = -B / (dB/ds) = focusing length
        D_mumu = scattering coefficient ~ v / (3*lambda_mfp) * (1-mu^2)

    In scatter-free transport (lambda >> 1 AU), particles stream freely with
    velocity dispersion. In diffusive transport (lambda << 1 AU), spatial
    diffusion dominates. The onset time analysis exploits velocity dispersion:
    plotting 1/v vs t_onset yields the path length as the slope.
"""

import numpy as np
import matplotlib.pyplot as plt

# === Physical Constants ===
AU = 1.496e11       # astronomical unit (m)
R_sun = 6.957e8     # solar radius (m)
c_light = 2.998e8   # speed of light (m/s)
MeV_to_J = 1.602e-13
m_p = 1.673e-27     # proton mass (kg)

# Solar wind and Parker spiral parameters
v_sw = 400e3          # solar wind speed (m/s)
Omega_sun = 2.87e-6   # solar rotation rate (rad/s)
r_0_mag = 5 * R_sun   # source surface for B
B_0 = 5e-5            # B at r_0 (0.5 Gauss)


def parker_spiral_B(r):
    """
    Magnetic field magnitude along Parker spiral in equatorial plane.
    B(r) = B_0 * (r_0/r)^2 * sqrt(1 + (Omega*r/v_sw)^2)
    """
    return B_0 * (r_0_mag / r) ** 2 * np.sqrt(1 + (Omega_sun * r / v_sw) ** 2)


def focusing_length(r, dr=1e8):
    """
    Focusing length L = -B / (dB/ds) where s is along the spiral.
    ds/dr = sqrt(1 + (Omega*r/v_sw)^2) for the spiral arc length.
    """
    B_here = parker_spiral_B(r)
    B_ahead = parker_spiral_B(r + dr)
    dBdr = (B_ahead - B_here) / dr
    # ds/dr = spiral factor
    spiral_factor = np.sqrt(1 + (Omega_sun * r / v_sw) ** 2)
    dBds = dBdr / spiral_factor
    L = np.where(np.abs(dBds) > 0, -B_here / dBds, 1e15)
    return L


def proton_speed(E_MeV):
    """Non-relativistic proton speed for energy E in MeV."""
    E_J = E_MeV * MeV_to_J
    return np.sqrt(2 * E_J / m_p)


# === Grid Setup ===
# Spatial grid along spiral field line
s_max = 1.8 * AU  # field line length to ~1 AU (spiral is longer than radial)
Ns = 200
ds = s_max / Ns
s = np.linspace(0, s_max, Ns)

# Pitch-angle cosine grid
Nmu = 41
mu = np.linspace(-1, 1, Nmu)
dmu = mu[1] - mu[0]

# Time grid
dt = 2.0  # seconds
Nt = 4000
t = np.arange(Nt) * dt

# Map s -> r (approximate: r ~ s for small winding angles near Sun)
# More accurate: ds = sqrt(1+(Omega*r/v_sw)^2) dr -> invert numerically
r_of_s = np.linspace(r_0_mag, r_0_mag + s_max * 0.85, Ns)  # approximate
L_s = focusing_length(r_of_s)

# Observer location: index closest to 1 AU along field line
# Parker spiral path length to 1 AU ~ 1.2 AU for v_sw = 400 km/s
s_1AU = 1.2 * AU
obs_idx = np.argmin(np.abs(s - s_1AU))

print("=" * 60)
print("FOCUSED TRANSPORT: SEP PROPAGATION")
print("=" * 60)
print(f"Spatial grid: {Ns} points, ds = {ds / AU:.4f} AU")
print(f"Pitch-angle grid: {Nmu} points")
print(f"Time step: {dt:.1f} s, total time: {Nt * dt / 3600:.1f} hours")
print(f"Observer at s = {s_1AU / AU:.2f} AU (index {obs_idx})")
print(f"Parker spiral path to 1 AU: ~{s_1AU / AU:.2f} AU")


def solve_focused_transport(E_MeV, lambda_mfp_AU, label=""):
    """
    Solve the focused transport equation for given energy and mean free path.
    Returns f(t) at the observer location (omnidirectional flux).
    """
    v = proton_speed(E_MeV)
    lambda_mfp = lambda_mfp_AU * AU

    # Scattering coefficient: D_mumu = v/(3*lambda) * (1-mu^2)
    # This is the quasi-linear theory (QLT) form
    D_coeff = v / (3 * lambda_mfp)  # base scattering rate

    # Initialize distribution: delta injection at s=0, t=0, isotropic in mu>0
    f = np.zeros((Ns, Nmu))
    # Inject at s ~ a few grid points from origin, forward hemisphere only
    inject_idx = 2
    for j in range(Nmu):
        if mu[j] > 0:
            f[inject_idx, j] = 1.0 / (dmu * ds)  # normalized

    # Storage for observer time profile
    flux_at_obs = np.zeros(Nt)

    for n in range(Nt):
        # Record omnidirectional flux at observer: J = integral(f * v * |mu| * dmu)
        flux_at_obs[n] = np.sum(f[obs_idx, :] * np.abs(mu)) * dmu * v

        # --- Advection in s: upwind scheme ---
        f_new = f.copy()
        for j in range(Nmu):
            v_s = v * mu[j]  # streaming speed along s
            for i in range(1, Ns - 1):
                if v_s > 0:
                    # Upwind: use left neighbor
                    f_new[i, j] -= dt * v_s * (f[i, j] - f[i - 1, j]) / ds
                else:
                    # Upwind: use right neighbor
                    f_new[i, j] -= dt * v_s * (f[i + 1, j] - f[i, j]) / ds

        # --- Focusing term: (1-mu^2)/(2L) * v * df/dmu ---
        f = f_new.copy()
        for i in range(1, Ns - 1):
            L_here = L_s[min(i, len(L_s) - 1)]
            for j in range(1, Nmu - 1):
                focusing = (1 - mu[j] ** 2) / (2 * L_here) * v
                # Central difference in mu
                dfdu = (f[i, j + 1] - f[i, j - 1]) / (2 * dmu)
                f_new[i, j] -= dt * focusing * dfdu

        # --- Pitch-angle diffusion: D_mumu * d^2f/dmu^2 ---
        f = f_new.copy()
        for i in range(1, Ns - 1):
            for j in range(1, Nmu - 1):
                D_here = D_coeff * (1 - mu[j] ** 2)
                d2f = (f[i, j + 1] - 2 * f[i, j] + f[i, j - 1]) / dmu ** 2
                f_new[i, j] += dt * D_here * d2f

        # Apply boundary conditions: f = 0 at boundaries
        f_new[0, :] = 0
        f_new[-1, :] = 0
        f_new[:, 0] = f_new[:, 1]    # reflecting at mu = -1
        f_new[:, -1] = f_new[:, -2]  # reflecting at mu = +1

        # Ensure non-negative
        f = np.maximum(f_new, 0)

    return flux_at_obs, f


# === Run simulations ===
# Test multiple energies for velocity dispersion analysis
energies_MeV = [10, 30, 100]  # proton energies
lambda_diffusive = 0.1   # AU, diffusive regime
lambda_scatter_free = 5.0  # AU, scatter-free regime

print("\n--- Proton energies and speeds ---")
for E in energies_MeV:
    v = proton_speed(E)
    print(f"  E = {E} MeV: v = {v / 1e6:.2f} Mm/s = {v / c_light:.3f} c, "
          f"free-streaming time to 1.2 AU = {s_1AU / v / 3600:.2f} h")

# Run scatter-free case
print("\n--- Solving scatter-free transport (lambda = 5 AU) ---")
results_sf = {}
for E in energies_MeV:
    print(f"  E = {E} MeV ...", end=" ", flush=True)
    flux, _ = solve_focused_transport(E, lambda_scatter_free)
    results_sf[E] = flux
    print("done")

# Run diffusive case
print("\n--- Solving diffusive transport (lambda = 0.1 AU) ---")
results_diff = {}
for E in energies_MeV:
    print(f"  E = {E} MeV ...", end=" ", flush=True)
    flux, _ = solve_focused_transport(E, lambda_diffusive)
    results_diff[E] = flux
    print("done")

# === Onset Analysis ===
# Onset time: first time flux exceeds threshold
threshold_frac = 0.01  # 1% of peak
onset_times_sf = {}
peak_times_sf = {}

print("\n--- Onset Time Analysis (scatter-free) ---")
for E in energies_MeV:
    flux = results_sf[E]
    peak = np.max(flux)
    if peak > 0:
        onset_idx = np.argmax(flux > threshold_frac * peak)
        peak_idx = np.argmax(flux)
        onset_times_sf[E] = t[onset_idx]
        peak_times_sf[E] = t[peak_idx]
        v = proton_speed(E)
        print(f"  E = {E} MeV: onset = {t[onset_idx] / 60:.1f} min, "
              f"peak = {t[peak_idx] / 60:.1f} min, "
              f"duration = {(t[peak_idx] - t[onset_idx]) / 60:.1f} min")
    else:
        onset_times_sf[E] = np.nan
        peak_times_sf[E] = np.nan
        print(f"  E = {E} MeV: no significant flux detected")

# === Plotting ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_E = {10: 'red', 30: 'blue', 100: 'green'}
t_hours = t / 3600

# 1. Scatter-free flux profiles
ax = axes[0, 0]
for E in energies_MeV:
    flux = results_sf[E]
    if np.max(flux) > 0:
        ax.plot(t_hours, flux / np.max(flux), color=colors_E[E],
                linewidth=1.5, label=f'{E} MeV')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Normalized Flux')
ax.set_title(f'Scatter-free Transport ($\\lambda$ = {lambda_scatter_free} AU)')
ax.legend()
ax.set_xlim(0, t_hours[-1])

# 2. Diffusive flux profiles
ax = axes[0, 1]
for E in energies_MeV:
    flux = results_diff[E]
    if np.max(flux) > 0:
        ax.plot(t_hours, flux / np.max(flux), color=colors_E[E],
                linewidth=1.5, label=f'{E} MeV')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Normalized Flux')
ax.set_title(f'Diffusive Transport ($\\lambda$ = {lambda_diffusive} AU)')
ax.legend()
ax.set_xlim(0, t_hours[-1])

# 3. Velocity dispersion: 1/v vs onset time
ax = axes[1, 0]
inv_v = []
t_onsets = []
for E in energies_MeV:
    if not np.isnan(onset_times_sf.get(E, np.nan)):
        v = proton_speed(E)
        inv_v.append(1 / v)
        t_onsets.append(onset_times_sf[E])
if len(inv_v) > 1:
    inv_v = np.array(inv_v)
    t_onsets = np.array(t_onsets)
    ax.scatter(t_onsets / 60, inv_v * 1e6, c=[colors_E[E] for E in energies_MeV
               if not np.isnan(onset_times_sf.get(E, np.nan))],
               s=100, zorder=5, edgecolors='black')
    # Linear fit: 1/v = (1/path_length) * t_onset
    if len(inv_v) >= 2:
        coeffs = np.polyfit(t_onsets, inv_v, 1)
        path_length = 1 / coeffs[0] if coeffs[0] != 0 else np.inf
        t_fit = np.linspace(0, max(t_onsets) * 1.2, 50)
        ax.plot(t_fit / 60, np.polyval(coeffs, t_fit) * 1e6, 'k--',
                label=f'Path = {path_length / AU:.2f} AU')
    for E in energies_MeV:
        if not np.isnan(onset_times_sf.get(E, np.nan)):
            v = proton_speed(E)
            ax.annotate(f'{E} MeV', xy=(onset_times_sf[E] / 60, 1e6 / v),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel('Onset Time (minutes)')
ax.set_ylabel('1/v ($\\times 10^{-6}$ s/m)')
ax.set_title('Velocity Dispersion Analysis (Onset Method)')
ax.legend()

# 4. Comparison: scatter-free vs diffusive for 30 MeV
ax = axes[1, 1]
E_compare = 30
flux_sf = results_sf[E_compare]
flux_df = results_diff[E_compare]
if np.max(flux_sf) > 0:
    ax.plot(t_hours, flux_sf / np.max(flux_sf), 'b-', linewidth=2,
            label=f'Scatter-free ($\\lambda$ = {lambda_scatter_free} AU)')
if np.max(flux_df) > 0:
    ax.plot(t_hours, flux_df / np.max(flux_df), 'r-', linewidth=2,
            label=f'Diffusive ($\\lambda$ = {lambda_diffusive} AU)')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Normalized Flux')
ax.set_title(f'{E_compare} MeV Protons: Scatter-free vs Diffusive')
ax.legend()
ax.set_xlim(0, t_hours[-1])

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Solar_Physics/12_sep_transport.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to 12_sep_transport.png")
print("\nKey insights:")
print("  - Scatter-free: sharp onset, velocity dispersion clearly visible")
print("  - Diffusive: gradual rise, extended duration, less velocity dispersion")
print("  - Onset analysis yields magnetic field line path length from Sun to observer")

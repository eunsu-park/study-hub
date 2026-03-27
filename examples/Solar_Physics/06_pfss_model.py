"""
Potential Field Source Surface (PFSS) Extrapolation.

Demonstrates the standard model for coronal magnetic field reconstruction
from photospheric magnetograms. Assumes the corona is current-free
(j = 0, so curl B = 0) between the photosphere and a source surface
where field lines are forced radial.

Key physics:
  - Potential field: B = -grad(Phi), with Laplace equation nabla^2 Phi = 0
  - Spherical shell: R_sun <= r <= R_ss (source surface at 2.5 R_sun)
  - Inner BC: B_r(R_sun) = magnetogram
  - Outer BC: B_theta(R_ss) = 0 (radial field at source surface)
  - Axisymmetric (m=0): expand in Legendre polynomials
  - Open vs closed field lines determine coronal holes and streamers
"""

import numpy as np
from scipy.special import lpmv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Parameters ---
R_sun = 1.0          # normalize to solar radius
R_ss = 2.5           # source surface radius [R_sun]
l_max = 15           # maximum spherical harmonic degree

# --- Synthetic magnetogram: dipole + quadrupole + octupole ---
# B_r(R_sun, theta) = sum_l g_l * P_l(cos theta)
# g_l are Gauss coefficients in Gauss (1e-4 T)
g_coeffs = np.zeros(l_max + 1)
g_coeffs[1] = 5.0    # dipole: 5 Gauss (dominant, like Sun near minimum)
g_coeffs[2] = -1.5   # quadrupole: -1.5 Gauss (asymmetry)
g_coeffs[3] = 0.8    # octupole: higher-order structure

print("=" * 60)
print("PFSS Model: Potential Field Source Surface Extrapolation")
print(f"  Source surface: R_ss = {R_ss} R_sun")
print(f"  Maximum l: {l_max}")
print(f"  Gauss coefficients: g1={g_coeffs[1]}, g2={g_coeffs[2]}, g3={g_coeffs[3]}")
print("=" * 60)

# --- PFSS solution in spherical coordinates (axisymmetric, m=0) ---
# For each l, the radial function is:
#   Phi_l(r) = A_l * r^l + B_l * r^(-(l+1))
#
# Boundary conditions:
#   B_r(R_sun) = -dPhi/dr|_R = g_l * P_l(cos theta)
#   B_theta(R_ss) = 0 => (1/r) dPhi/dtheta|_R_ss has specific constraint
#
# This gives:
#   A_l = g_l * R^(l+2) * (l+1) / [l * R_ss^(2l+1) + (l+1) * R^(2l+1)]
#     ... actually, we solve the standard PFSS coefficient equations:
#
# B_r(r, theta) = sum_l [c_l(r)] * P_l(cos theta)
# where c_l(r) = g_l * [ (l+1)*(r/R)^(-l-2) + l*(r/R)^(l-1)*(R/R_ss)^(2l+1) ]
#                       / [ l+1 + l*(R/R_ss)^(2l+1) ]

def B_r_pfss(r, theta, g, R=R_sun, Rss=R_ss, lmax=l_max):
    """Compute radial magnetic field B_r at (r, theta) using PFSS."""
    cos_theta = np.cos(theta)
    Br = np.zeros_like(r * theta, dtype=float) if np.ndim(r * theta) > 0 else 0.0
    for l in range(1, lmax + 1):
        if abs(g[l]) < 1e-12:
            continue
        Pl = lpmv(0, l, cos_theta)  # P_l^0(cos theta)
        ratio = (R / Rss) ** (2 * l + 1)
        # Radial coefficient for PFSS
        cl = g[l] * ((l + 1) * (R / r) ** (l + 2) + l * (r / R) ** (l - 1) * ratio) \
             / (l + 1 + l * ratio)
        Br = Br + cl * Pl
    return Br

def B_theta_pfss(r, theta, g, R=R_sun, Rss=R_ss, lmax=l_max):
    """Compute theta-component of B at (r, theta) using PFSS.
    B_theta = -(1/r) * dPhi/dtheta, computed via derivative of Legendre polynomials."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta) + 1e-12  # avoid division by zero
    Bt = np.zeros_like(r * theta, dtype=float) if np.ndim(r * theta) > 0 else 0.0
    for l in range(1, lmax + 1):
        if abs(g[l]) < 1e-12:
            continue
        # dP_l/dtheta = l*cos(theta)*P_l(cos theta)/(sin^2 theta) - l*P_{l-1}/(sin theta)
        # Use recurrence: dP_l/dtheta = -P_l^1(cos theta)
        dPl = -lpmv(1, l, cos_theta)  # = -P_l^1(cos theta)
        ratio = (R / Rss) ** (2 * l + 1)
        # Theta coefficient: from -(1/r)*d(Phi)/d(theta)
        # The sign ensures B_theta = 0 at R_ss
        cl = g[l] * ((R / r) ** (l + 2) - (r / R) ** (l - 1) * ratio) \
             / (l + 1 + l * ratio)
        Bt = Bt + cl * dPl / r
    return Bt

# --- Field line tracing using RK4 ---
def trace_field_line(r0, theta0, g, ds=0.005, max_steps=5000, direction=1):
    """Trace a field line from (r0, theta0) in the meridional plane.
    direction: +1 for outward from positive B_r, -1 for inward."""
    r_path = [r0]
    theta_path = [theta0]
    r, theta = r0, theta0

    for _ in range(max_steps):
        Br = B_r_pfss(r, theta, g)
        Bt = B_theta_pfss(r, theta, g)
        B_mag = np.sqrt(Br**2 + Bt**2) + 1e-15

        # Step in field line direction (ds along B)
        dr = direction * ds * Br / B_mag
        dtheta = direction * ds * Bt / (B_mag * r)

        r_new = r + dr
        theta_new = theta + dtheta

        # Check boundaries
        if r_new < R_sun or r_new > R_ss + 0.1:
            r_path.append(np.clip(r_new, R_sun, R_ss))
            theta_path.append(theta_new)
            break
        if theta_new < 0 or theta_new > np.pi:
            break

        r = r_new
        theta = theta_new
        r_path.append(r)
        theta_path.append(theta)

    return np.array(r_path), np.array(theta_path)

# --- Classify field lines as open or closed ---
print("\nTracing field lines and classifying open/closed...")

# Start points along the photosphere
n_start = 60
theta_starts = np.linspace(0.05, np.pi - 0.05, n_start)
field_lines = []
classifications = []  # 'open_pos', 'open_neg', 'closed'

for th0 in theta_starts:
    Br_surface = B_r_pfss(R_sun, th0, g_coeffs)
    direction = 1 if Br_surface > 0 else -1

    r_path, theta_path = trace_field_line(R_sun, th0, g_coeffs, direction=direction)
    field_lines.append((r_path, theta_path))

    # Classify
    r_end = r_path[-1]
    if r_end >= R_ss - 0.05:
        classifications.append('open_pos' if Br_surface > 0 else 'open_neg')
    else:
        # Closed: trace in opposite direction too
        r_path2, theta_path2 = trace_field_line(R_sun, th0, g_coeffs, direction=-direction)
        r_full = np.concatenate([r_path2[::-1], r_path])
        theta_full = np.concatenate([theta_path2[::-1], theta_path])
        field_lines[-1] = (r_full, theta_full)
        classifications.append('closed')

n_open = sum(1 for c in classifications if 'open' in c)
n_closed = sum(1 for c in classifications if c == 'closed')
print(f"  Open field lines: {n_open}")
print(f"  Closed field lines: {n_closed}")

# --- Find heliospheric current sheet (HCS) ---
# HCS is where B_r = 0 at the source surface
theta_hcs = np.linspace(0.01, np.pi - 0.01, 500)
Br_ss = np.array([B_r_pfss(R_ss, th, g_coeffs) for th in theta_hcs])
# Find zero crossings
hcs_crossings = []
for i in range(len(Br_ss) - 1):
    if Br_ss[i] * Br_ss[i + 1] < 0:
        # Linear interpolation
        theta_cross = theta_hcs[i] - Br_ss[i] * (theta_hcs[i + 1] - theta_hcs[i]) / (Br_ss[i + 1] - Br_ss[i])
        hcs_crossings.append(theta_cross)
        print(f"  HCS crossing at theta = {np.degrees(theta_cross):.1f} deg (colatitude)")

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Potential Field Source Surface (PFSS) Model", fontsize=14, y=0.98)

# Panel 1: Magnetogram (input B_r at photosphere)
ax = axes[0, 0]
theta_plot = np.linspace(0, np.pi, 200)
Br_photo = np.array([B_r_pfss(R_sun, th, g_coeffs) for th in theta_plot])
lat_plot = 90 - np.degrees(theta_plot)  # convert colatitude to latitude
ax.plot(lat_plot, Br_photo, 'k-', lw=2)
ax.fill_between(lat_plot, 0, Br_photo, where=Br_photo > 0, alpha=0.3, color='red', label='Positive')
ax.fill_between(lat_plot, 0, Br_photo, where=Br_photo < 0, alpha=0.3, color='blue', label='Negative')
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.set_xlabel("Latitude [deg]")
ax.set_ylabel("B_r [Gauss]")
ax.set_title("Synthetic Magnetogram (Photospheric $B_r$)")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Field lines in meridional plane
ax = axes[0, 1]
# Draw solar surface
theta_c = np.linspace(0, 2 * np.pi, 200)
ax.fill(R_sun * np.sin(theta_c), R_sun * np.cos(theta_c), color='yellow', alpha=0.4)
ax.plot(R_sun * np.sin(theta_c), R_sun * np.cos(theta_c), 'k-', lw=2)

# Draw source surface
ax.plot(R_ss * np.sin(theta_c), R_ss * np.cos(theta_c), 'k--', lw=1, alpha=0.5)

# Plot field lines
for (r_path, theta_path), cls in zip(field_lines, classifications):
    x = r_path * np.sin(theta_path)
    y = r_path * np.cos(theta_path)
    if cls == 'open_pos':
        ax.plot(x, y, 'r-', lw=0.8, alpha=0.7)
    elif cls == 'open_neg':
        ax.plot(x, y, 'b-', lw=0.8, alpha=0.7)
    else:
        ax.plot(x, y, 'gray', lw=0.8, alpha=0.5)

# Mark HCS at source surface
for theta_cross in hcs_crossings:
    ax.plot(R_ss * np.sin(theta_cross), R_ss * np.cos(theta_cross),
            'g*', markersize=12, zorder=5, label='HCS' if theta_cross == hcs_crossings[0] else '')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_xlabel(r"$x / R_\odot$")
ax.set_ylabel(r"$z / R_\odot$ (rotation axis)")
ax.set_title("PFSS Field Lines (meridional plane)")
ax.legend(fontsize=8)

# Panel 3: B_r at source surface (determines open flux)
ax = axes[1, 0]
Br_ss_plot = np.array([B_r_pfss(R_ss, th, g_coeffs) for th in theta_plot])
lat = 90 - np.degrees(theta_plot)
ax.plot(lat, Br_ss_plot, 'k-', lw=2)
ax.fill_between(lat, 0, Br_ss_plot, where=Br_ss_plot > 0, alpha=0.3, color='red')
ax.fill_between(lat, 0, Br_ss_plot, where=Br_ss_plot < 0, alpha=0.3, color='blue')
ax.axhline(0, color='gray', ls='-', lw=0.5)
for theta_cross in hcs_crossings:
    ax.axvline(90 - np.degrees(theta_cross), color='green', ls=':', lw=2, label='HCS')
ax.set_xlabel("Latitude [deg]")
ax.set_ylabel("B_r [Gauss]")
ax.set_title("$B_r$ at Source Surface (r = 2.5 $R_\\odot$)")
ax.grid(True, alpha=0.3)

# Panel 4: B_r contour map in r-theta plane
ax = axes[1, 1]
r_grid = np.linspace(R_sun, R_ss, 100)
theta_grid = np.linspace(0.01, np.pi - 0.01, 100)
R_GRID, TH_GRID = np.meshgrid(r_grid, theta_grid)

# Compute B_r on the grid
Br_grid = np.zeros_like(R_GRID)
for i in range(R_GRID.shape[0]):
    for j in range(R_GRID.shape[1]):
        Br_grid[i, j] = B_r_pfss(R_GRID[i, j], TH_GRID[i, j], g_coeffs)

# Convert to Cartesian for display
X_GRID = R_GRID * np.sin(TH_GRID)
Y_GRID = R_GRID * np.cos(TH_GRID)

levels = np.linspace(-5, 5, 21)
cf = ax.contourf(X_GRID, Y_GRID, Br_grid, levels=levels, cmap='RdBu_r', extend='both')
ax.contour(X_GRID, Y_GRID, Br_grid, levels=[0], colors='green', linewidths=2)
plt.colorbar(cf, ax=ax, label='$B_r$ [Gauss]', shrink=0.8)

# Draw boundaries
ax.plot(R_sun * np.sin(theta_c), R_sun * np.cos(theta_c), 'k-', lw=2)
ax.plot(R_ss * np.sin(theta_c), R_ss * np.cos(theta_c), 'k--', lw=1)
ax.set_xlim(0, 2.8)
ax.set_ylim(-2.8, 2.8)
ax.set_aspect('equal')
ax.set_xlabel(r"$x / R_\odot$")
ax.set_ylabel(r"$z / R_\odot$")
ax.set_title("$B_r$ Contours (green = neutral line)")

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Solar_Physics/06_pfss_model.png",
            dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved: 06_pfss_model.png")

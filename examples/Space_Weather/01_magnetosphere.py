"""
Earth's Magnetosphere Structure: Dipole Field, Magnetopause, and Key Regions.

Models the geomagnetic field as a magnetic dipole and adds a simplified
magnetopause boundary using the Shue et al. (1998) empirical model. The
magnetopause standoff distance depends on solar wind dynamic pressure and
IMF Bz, making it a fundamental parameter of magnetospheric physics.

Key physics:
  - Dipole field: B_r = -2*M*cos(theta)/(4*pi*r^3)
                  B_theta = -M*sin(theta)/(4*pi*r^3)
  - Shue magnetopause: r = r0 * (2 / (1 + cos(theta)))^alpha
    where r0 = standoff distance, alpha = tail flaring exponent
  - r0 depends on solar wind dynamic pressure P_dyn and IMF Bz
  - Key regions: bow shock (~1.3 * r_mp), plasmasphere (L < 4),
    radiation belts (inner ~1.5-2.5 R_E, outer ~3-7 R_E)
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
R_E = 6.371e6          # Earth radius [m]
mu_0 = 4 * np.pi * 1e-7  # permeability of free space [H/m]
M_E = 8.0e22           # Earth's magnetic dipole moment [A*m^2] (= 8e15 T*m^3)
B_eq = mu_0 * M_E / (4 * np.pi * R_E**3)  # equatorial surface field ~31000 nT

print("=" * 65)
print("Earth's Magnetosphere: Dipole Model + Shue Magnetopause")
print("=" * 65)
print(f"  Earth radius R_E        = {R_E:.3e} m")
print(f"  Dipole moment M_E       = {M_E:.2e} A*m^2")
print(f"  Equatorial surface |B|  = {B_eq*1e9:.1f} nT  (observed: ~31000 nT)")

# =========================================================================
# 1. DIPOLE MAGNETIC FIELD
# =========================================================================
# In spherical coordinates (r, theta) with theta from magnetic north pole:
#   B_r     = -2 * mu_0 * M * cos(theta) / (4*pi*r^3)
#   B_theta = -mu_0 * M * sin(theta) / (4*pi*r^3)
#   |B|     = (mu_0*M)/(4*pi*r^3) * sqrt(1 + 3*cos^2(theta))

def dipole_field(r, theta):
    """Compute dipole B_r and B_theta in Tesla.

    Parameters
    ----------
    r : array-like
        Radial distance [m].
    theta : array-like
        Colatitude from magnetic north pole [rad].

    Returns
    -------
    B_r, B_theta : arrays in Tesla
    """
    factor = mu_0 * M_E / (4 * np.pi * r**3)
    B_r = -2.0 * factor * np.cos(theta)
    B_theta = -factor * np.sin(theta)
    return B_r, B_theta


# =========================================================================
# 2. DIPOLE FIELD LINES IN THE NOON-MIDNIGHT MERIDIAN
# =========================================================================
# A dipole field line satisfies r = L * R_E * sin^2(theta)
# where L is the McIlwain L-shell parameter.

def dipole_field_line_cartesian(L, n_points=500):
    """Generate (x, z) coordinates of a dipole field line for given L-shell.

    The field line equation in the magnetic meridian plane:
        r = L * R_E * sin^2(theta)
    Converting to Cartesian: x = r*sin(theta), z = r*cos(theta)

    Returns both northern and southern halves.
    """
    theta = np.linspace(0.01, np.pi - 0.01, n_points)
    r = L * np.sin(theta)**2  # in units of R_E
    x = r * np.sin(theta)
    z = r * np.cos(theta)
    return x, z


# =========================================================================
# 3. SHUE MAGNETOPAUSE MODEL
# =========================================================================
# Shue et al. (1998) empirical model:
#   r_mp = r0 * (2 / (1 + cos(theta)))^alpha
#
# Standoff distance r0 and flaring alpha depend on solar wind:
#   r0 = (10.22 + 1.29 * tanh(0.184*(Bz + 8.14))) * P_dyn^(-1/6.6)
#   alpha = (0.58 - 0.007*Bz) * (1 + 0.024 * ln(P_dyn))
#
# P_dyn = dynamic pressure [nPa], Bz = IMF Bz [nT]

def shue_magnetopause(P_dyn_nPa, Bz_nT, theta_arr):
    """Compute Shue et al. (1998) magnetopause shape.

    Parameters
    ----------
    P_dyn_nPa : float
        Solar wind dynamic pressure [nPa].
    Bz_nT : float
        IMF Bz component [nT].
    theta_arr : array
        Angle from Sun-Earth line [rad], 0 = subsolar.

    Returns
    -------
    r_mp : array
        Magnetopause distance in Earth radii.
    r0 : float
        Standoff distance in Earth radii.
    """
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz_nT + 8.14))) * P_dyn_nPa**(-1.0/6.6)
    alpha = (0.58 - 0.007 * Bz_nT) * (1.0 + 0.024 * np.log(P_dyn_nPa))
    r_mp = r0 * (2.0 / (1.0 + np.cos(theta_arr)))**alpha
    return r_mp, r0


# --- Typical quiet-time solar wind conditions ---
P_dyn_quiet = 2.0    # nPa (typical)
Bz_quiet = 0.0       # nT (no strong IMF)

theta_mp = np.linspace(0, np.pi * 0.99, 300)
r_mp_quiet, r0_quiet = shue_magnetopause(P_dyn_quiet, Bz_quiet, theta_mp)

print(f"\nShue Magnetopause (quiet: P_dyn={P_dyn_quiet} nPa, Bz={Bz_quiet} nT):")
print(f"  Standoff distance r0   = {r0_quiet:.2f} R_E  (typical: ~10 R_E)")

# Bow shock is roughly 1.3 times the magnetopause standoff
r_bs = 1.3 * r0_quiet
print(f"  Bow shock (approx)     = {r_bs:.2f} R_E")


# =========================================================================
# 4. MAGNETOPAUSE STANDOFF vs SOLAR WIND PRESSURE
# =========================================================================
P_dyn_range = np.linspace(0.5, 20.0, 100)  # nPa
r0_vs_P = np.array([shue_magnetopause(P, 0.0, np.array([0.0]))[1] for P in P_dyn_range])
r0_vs_P_south = np.array([shue_magnetopause(P, -10.0, np.array([0.0]))[1] for P in P_dyn_range])

print(f"\nStandoff at P_dyn=10 nPa (Bz=0):   r0 = {shue_magnetopause(10, 0, np.array([0.0]))[1]:.2f} R_E")
print(f"Standoff at P_dyn=10 nPa (Bz=-10): r0 = {shue_magnetopause(10, -10, np.array([0.0]))[1]:.2f} R_E")


# =========================================================================
# 5. PLOTTING
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("Earth's Magnetosphere Structure", fontsize=15, y=1.02)

# --- Panel 1: Field lines + magnetopause in noon-midnight meridian ---
ax = axes[0]

# Draw Earth
earth_theta = np.linspace(0, 2 * np.pi, 100)
ax.fill(np.cos(earth_theta), np.sin(earth_theta), color='steelblue', alpha=0.7, zorder=5)
ax.plot(np.cos(earth_theta), np.sin(earth_theta), 'k-', lw=1.5, zorder=5)

# Dipole field lines for various L-shells
L_values = [2, 3, 4, 5, 6, 7, 8, 10]
colors_L = plt.cm.viridis(np.linspace(0.2, 0.9, len(L_values)))
for i, L in enumerate(L_values):
    x_fl, z_fl = dipole_field_line_cartesian(L)
    # Dayside (positive x = sunward)
    ax.plot(x_fl, z_fl, color=colors_L[i], lw=0.8, alpha=0.7)
    # Nightside (negative x)
    ax.plot(-x_fl, z_fl, color=colors_L[i], lw=0.8, alpha=0.7)

# Magnetopause boundary (Shue model)
x_mp = r_mp_quiet * np.cos(theta_mp)  # theta=0 is sunward
z_mp = r_mp_quiet * np.sin(theta_mp)
ax.plot(x_mp, z_mp, 'r-', lw=2.5, label='Magnetopause')
ax.plot(x_mp, -z_mp, 'r-', lw=2.5)

# Bow shock (approximate)
r_bs_shape = 1.3 * r_mp_quiet
x_bs = r_bs_shape * np.cos(theta_mp)
z_bs = r_bs_shape * np.sin(theta_mp)
ax.plot(x_bs, z_bs, 'r--', lw=1.5, alpha=0.7, label=f'Bow shock (~1.3 r_mp)')
ax.plot(x_bs, -z_bs, 'r--', lw=1.5, alpha=0.7)

# Mark plasmasphere (L < 4)
theta_ps = np.linspace(0, 2 * np.pi, 200)
r_ps = 4.0  # L = 4 in equatorial plane
ax.fill(r_ps * np.cos(theta_ps), r_ps * np.sin(theta_ps),
        color='yellow', alpha=0.15, label='Plasmasphere (L<4)')

# Mark radiation belts (inner ~1.5-2.5 R_E, outer ~3-7 R_E)
for r_inner, r_outer, label, color in [
    (1.5, 2.5, 'Inner belt', 'orange'),
    (3.0, 6.0, 'Outer belt', 'salmon')
]:
    ring_theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(
        np.concatenate([r_outer * np.cos(ring_theta), r_inner * np.cos(ring_theta[::-1])]),
        np.concatenate([r_outer * np.sin(ring_theta), r_inner * np.sin(ring_theta[::-1])]),
        color=color, alpha=0.15, label=label
    )

ax.set_xlabel("X (sunward) [$R_E$]", fontsize=11)
ax.set_ylabel("Z (north) [$R_E$]", fontsize=11)
ax.set_title("Noon-Midnight Meridian", fontsize=12)
ax.set_xlim(-20, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='lower left')
ax.grid(True, alpha=0.3)

# Sun direction arrow
ax.annotate("Sun", xy=(14, 0), fontsize=10, ha='center', color='darkorange',
            fontweight='bold')
ax.annotate("", xy=(14.5, 0), xytext=(12, 0),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

# --- Panel 2: |B| along equator (theta = pi/2) ---
ax = axes[1]
r_eq = np.linspace(1.0, 12.0, 500)  # in R_E
B_r_eq, B_th_eq = dipole_field(r_eq * R_E, np.pi / 2)
B_mag_eq = np.sqrt(B_r_eq**2 + B_th_eq**2) * 1e9  # convert to nT

ax.semilogy(r_eq, B_mag_eq, 'b-', lw=2, label=r'Dipole $|B| = B_0/L^3$')
ax.axvline(r0_quiet, color='red', ls='--', lw=1.5, label=f'Magnetopause ({r0_quiet:.1f} $R_E$)')
ax.axvspan(1.5, 2.5, color='orange', alpha=0.2, label='Inner belt')
ax.axvspan(3.0, 6.0, color='salmon', alpha=0.2, label='Outer belt')

ax.set_xlabel("Equatorial distance [$R_E$]", fontsize=11)
ax.set_ylabel("|B| [nT]", fontsize=11)
ax.set_title("Equatorial Magnetic Field Strength", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 12)

# --- Panel 3: Standoff distance vs solar wind pressure ---
ax = axes[2]
ax.plot(P_dyn_range, r0_vs_P, 'b-', lw=2, label=r'$B_z = 0$ nT')
ax.plot(P_dyn_range, r0_vs_P_south, 'r-', lw=2, label=r'$B_z = -10$ nT (southward)')

# Mark typical conditions
ax.axvline(2.0, color='green', ls=':', alpha=0.7, label='Quiet (2 nPa)')
ax.axvline(10.0, color='orange', ls=':', alpha=0.7, label='Storm (10 nPa)')

ax.set_xlabel("Solar wind dynamic pressure $P_{dyn}$ [nPa]", fontsize=11)
ax.set_ylabel("Standoff distance $r_0$ [$R_E$]", fontsize=11)
ax.set_title("Magnetopause Standoff vs $P_{dyn}$", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Space_Weather/01_magnetosphere.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: 01_magnetosphere.png")

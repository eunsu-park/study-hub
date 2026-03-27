"""
Parker Spiral: Interplanetary Magnetic Field Geometry.

Demonstrates:
- Derivation of the Parker spiral from frozen-in flux in the rotating solar frame
- Radial and azimuthal components of the heliospheric magnetic field
- Garden-hose angle as a function of distance and solar wind speed
- |B|(r) showing transition from r^{-2} (radial) to r^{-1} (azimuthal) dominance
- Polar plot of field lines from multiple source longitudes

Physics:
    The solar corona is magnetically connected to the Sun's surface, which rotates
    with angular velocity Omega_sun ~ 2.87e-6 rad/s (sidereal period ~25.4 days).
    The solar wind carries the magnetic field radially outward (frozen-in condition),
    but the footpoint rotation wraps the field into an Archimedean spiral (Parker 1958).

    B_r(r) = B_0 * (r_0/r)^2             (radial flux conservation)
    B_phi(r) = -B_r * Omega * r * sin(theta) / v_sw  (frozen-in winding)

    The garden-hose angle psi = arctan(|B_phi|/|B_r|) = arctan(Omega*r*sin(theta)/v_sw)
    reaches ~45 deg at 1 AU for v_sw ~ 400 km/s.
"""

import numpy as np
import matplotlib.pyplot as plt

# === Physical Constants ===
AU = 1.496e11       # astronomical unit (m)
R_sun = 6.957e8     # solar radius (m)

# Solar rotation (sidereal)
Omega_sun = 2.87e-6  # rad/s (25.4-day sidereal rotation period)

# Reference magnetic field at r_0 = 10 R_sun
r_0 = 10 * R_sun
B_0 = 1e-4  # 1 Gauss = 1e-4 T at r_0

# Co-latitude (equatorial plane)
theta = np.pi / 2  # sin(theta) = 1

print("=" * 60)
print("PARKER SPIRAL: INTERPLANETARY MAGNETIC FIELD")
print("=" * 60)

# === Magnetic Field Components ===
def parker_B(r, v_sw, B_ref=B_0, r_ref=r_0):
    """
    Compute Parker spiral magnetic field components.
    Returns B_r, B_phi, |B|, garden-hose angle psi.
    """
    B_r = B_ref * (r_ref / r) ** 2
    B_phi = -B_r * Omega_sun * r * np.sin(theta) / v_sw
    B_mag = np.sqrt(B_r ** 2 + B_phi ** 2)
    psi = np.arctan2(np.abs(B_phi), np.abs(B_r))
    return B_r, B_phi, B_mag, psi


# === Evaluate for different solar wind speeds ===
v_sw_values = {
    'Slow (300 km/s)': 300e3,
    'Medium (400 km/s)': 400e3,
    'Fast (600 km/s)': 600e3,
}

r_range = np.linspace(r_0, 2 * AU, 1000)

print(f"\nReference: B_0 = {B_0 * 1e4:.1f} G at r_0 = {r_0 / R_sun:.0f} R_sun")
print(f"Solar rotation: Omega = {Omega_sun:.3e} rad/s "
      f"(P = {2 * np.pi / Omega_sun / 86400:.1f} days)")

print("\n--- Values at 1 AU ---")
for label, v_sw in v_sw_values.items():
    B_r, B_phi, B_mag, psi = parker_B(AU, v_sw)
    print(f"  {label}:")
    print(f"    B_r     = {B_r * 1e9:.2f} nT")
    print(f"    B_phi   = {B_phi * 1e9:.2f} nT")
    print(f"    |B|     = {B_mag * 1e9:.2f} nT")
    print(f"    psi     = {np.degrees(psi):.1f} deg (garden-hose angle)")

# === Parker Spiral Field Lines (equatorial plane) ===
# In polar coordinates (r, phi):
#   r(phi) = r_0 - v_sw/Omega * (phi - phi_0)
# or equivalently: phi(r) = phi_0 - Omega/v_sw * (r - r_0)
def spiral_phi(r, v_sw, phi_0=0):
    """Parker spiral angle as function of distance."""
    return phi_0 - (Omega_sun / v_sw) * (r - r_0)


# === Plotting ===
fig = plt.figure(figsize=(16, 12))

# 1. Polar plot of Parker spiral field lines
ax1 = fig.add_subplot(221, polar=True)
r_spiral = np.linspace(r_0, 2.5 * AU, 500)
source_longitudes = np.linspace(0, 2 * np.pi, 8, endpoint=False)

colors_sw = {'Slow (300 km/s)': 'blue', 'Medium (400 km/s)': 'green',
             'Fast (600 km/s)': 'red'}

for label, v_sw in v_sw_values.items():
    for i, phi_0 in enumerate(source_longitudes):
        phi = spiral_phi(r_spiral, v_sw, phi_0)
        alpha = 0.8 if i == 0 else 0.3
        lw = 2 if i == 0 else 0.8
        lbl = label if i == 0 else None
        ax1.plot(phi, r_spiral / AU, color=colors_sw[label],
                 linewidth=lw, alpha=alpha, label=lbl)

ax1.set_rmax(2.5)
ax1.set_title('Parker Spiral Field Lines\n(equatorial plane)', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=8)

# Mark 1 AU circle
theta_circle = np.linspace(0, 2 * np.pi, 100)
ax1.plot(theta_circle, np.ones_like(theta_circle), 'k--', alpha=0.3, linewidth=1)
ax1.annotate('1 AU', xy=(0.3, 1.05), fontsize=8, color='gray')

# 2. |B|(r) showing r^-2 to r^-1 transition
ax2 = fig.add_subplot(222)
for label, v_sw in v_sw_values.items():
    B_r, B_phi, B_mag, psi = parker_B(r_range, v_sw)
    ax2.loglog(r_range / AU, B_mag * 1e9, color=colors_sw[label],
               linewidth=2, label=label)

# Reference slopes
r_ref = r_range / AU
B_r2 = B_0 * (r_0 / r_range) ** 2 * 1e9  # pure r^-2
ax2.loglog(r_ref, B_r2, 'k--', alpha=0.3, linewidth=1, label='$\\propto r^{-2}$')
# Approximate r^-1 region
B_r1 = B_r2[0] * (r_ref[0] / r_ref) ** 1 * 0.5
ax2.loglog(r_ref, B_r1, 'k:', alpha=0.3, linewidth=1, label='$\\propto r^{-1}$')

ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Heliocentric Distance (AU)')
ax2.set_ylabel('|B| (nT)')
ax2.set_title('Magnetic Field Magnitude vs Distance')
ax2.legend(fontsize=8)
ax2.set_xlim(r_0 / AU, 2.0)

# 3. Garden-hose angle vs distance
ax3 = fig.add_subplot(223)
for label, v_sw in v_sw_values.items():
    _, _, _, psi = parker_B(r_range, v_sw)
    ax3.plot(r_range / AU, np.degrees(psi), color=colors_sw[label],
             linewidth=2, label=label)
ax3.axhline(45, color='gray', linestyle=':', alpha=0.5, label='45 deg')
ax3.axvline(1.0, color='gray', linestyle='--', alpha=0.3)
ax3.set_xlabel('Heliocentric Distance (AU)')
ax3.set_ylabel('Garden-Hose Angle (degrees)')
ax3.set_title('Spiral Angle vs Distance')
ax3.legend(fontsize=8)
ax3.set_ylim(0, 90)

# 4. B_r and B_phi components at 1 AU vs v_sw
ax4 = fig.add_subplot(224)
v_sw_scan = np.linspace(200e3, 800e3, 100)
B_r_1au, B_phi_1au, B_mag_1au, psi_1au = parker_B(AU, v_sw_scan)

ax4.plot(v_sw_scan / 1e3, np.full_like(v_sw_scan, np.abs(B_r_1au) * 1e9), 'b-',
         linewidth=2, label='$|B_r|$')
ax4.plot(v_sw_scan / 1e3, np.abs(B_phi_1au) * 1e9, 'r-', linewidth=2,
         label='$|B_\\phi|$')
ax4.plot(v_sw_scan / 1e3, B_mag_1au * 1e9, 'k--', linewidth=2,
         label='$|B|$')

# Mark typical values
for label, v_sw in v_sw_values.items():
    _, _, bm, _ = parker_B(AU, v_sw)
    ax4.axvline(v_sw / 1e3, color=colors_sw[label], linestyle=':', alpha=0.5)
    ax4.plot(v_sw / 1e3, bm * 1e9, 'o', color=colors_sw[label], markersize=8)

ax4.set_xlabel('Solar Wind Speed (km/s)')
ax4.set_ylabel('B Component (nT)')
ax4.set_title('Field Components at 1 AU vs Solar Wind Speed')
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Solar_Physics/10_parker_spiral.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nNote: B_r is independent of v_sw (radial flux conservation).")
print("B_phi ~ 1/v_sw: slower wind -> tighter spiral, stronger B_phi.")
print("\nPlot saved to 10_parker_spiral.png")

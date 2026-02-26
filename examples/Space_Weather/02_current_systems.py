"""
Magnetospheric Current Systems: Ring Current, Tail Current, Birkeland Currents.

Models the three major current systems that shape the magnetosphere and
drive geomagnetic disturbances. The ring current encircles Earth at ~3-7 R_E,
the tail current sheet flows across the magnetotail, and Birkeland (field-
aligned) currents connect the magnetosphere to the ionosphere.

Key physics:
  - Ring current: Westward current of trapped ions at L ~ 3-7
    Magnetic perturbation at Earth's center from a circular loop:
    Delta_B = -mu_0 * I / (2 * a)
  - Tail current sheet: Harris model B_x = B0 * tanh(z / D)
    Current density J_y = (B0 / (mu_0 * D)) * sech^2(z / D)
  - Birkeland currents: Field-aligned currents J_|| = J0 * cos(phi)
    Region 1 (poleward) and Region 2 (equatorward) systems
  - Dessler-Parker-Sckopke relation:
    Dst ~ -mu_0 * E_RC / (4*pi * R_E^3 * B_0)
    connecting ring current energy to the Dst index
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
R_E = 6.371e6          # Earth radius [m]
mu_0 = 4 * np.pi * 1e-7  # permeability of free space [H/m]
B_0 = 3.12e-5          # equatorial surface field [T] (~31200 nT)

print("=" * 65)
print("Magnetospheric Current Systems")
print("=" * 65)


# =========================================================================
# 1. RING CURRENT MODEL
# =========================================================================
# The ring current is modeled as a set of concentric circular current loops
# at the magnetic equator. For a circular loop of radius a carrying current I,
# the on-axis magnetic field at the center is:
#   Delta_B_z = -mu_0 * I / (2 * a)  (opposes dipole field -> negative Dst)

# Off-axis field from a circular current loop (Biot-Savart, exact for axis):
# For full spatial variation, we use the field of a finite-width ring.

def ring_current_field_on_axis(I_total, a, z_points):
    """Magnetic field on the axis of a circular current loop.

    Parameters
    ----------
    I_total : float
        Total current [A].
    a : float
        Loop radius [m].
    z_points : array
        Distance along axis from loop center [m].

    Returns
    -------
    B_z : array in Tesla
    """
    return mu_0 * I_total * a**2 / (2.0 * (a**2 + z_points**2)**1.5)


# Typical ring current parameters
L_ring = 4.5                    # average ring current L-shell
a_ring = L_ring * R_E           # ring radius [m]
I_ring = 3.0e6                  # total ring current ~3 MA for moderate storm

# Field perturbation at Earth's center (z=0)
Delta_B_center = -mu_0 * I_ring / (2.0 * a_ring)
print(f"\nRing Current (L={L_ring}, I={I_ring/1e6:.1f} MA):")
print(f"  Delta B at center = {Delta_B_center*1e9:.1f} nT")
print(f"  This corresponds to Dst ~ {Delta_B_center*1e9:.0f} nT (moderate storm)")

# On-axis field variation
z_axis = np.linspace(-10 * R_E, 10 * R_E, 500)
B_ring_axis = ring_current_field_on_axis(I_ring, a_ring, z_axis)

# Dessler-Parker-Sckopke relation
# E_RC = total energy of ring current particles
# Dst* = -mu_0 * E_RC / (4*pi * R_E^3 * B_0)
# For a rough estimate: E_RC ~ 3/2 * n * k_T * V where V is ring volume
E_RC_values = np.linspace(0, 8e15, 100)  # Joules (typical: 1e14 to 8e15)
Dst_DPS = -mu_0 * E_RC_values / (4 * np.pi * R_E**3 * B_0) * 1e9  # nT

print(f"\nDessler-Parker-Sckopke relation:")
print(f"  E_RC = 1e15 J -> Dst ~ {-mu_0*1e15/(4*np.pi*R_E**3*B_0)*1e9:.1f} nT")
print(f"  E_RC = 5e15 J -> Dst ~ {-mu_0*5e15/(4*np.pi*R_E**3*B_0)*1e9:.1f} nT")


# =========================================================================
# 2. TAIL CURRENT SHEET (HARRIS MODEL)
# =========================================================================
# The magnetotail current sheet is well-described by the Harris (1962) model:
#   B_x(z) = B0 * tanh(z / D)
#   J_y(z) = (B0 / (mu_0 * D)) * sech^2(z / D)
# where D is the half-thickness of the current sheet.

B0_tail = 20e-9              # lobe field strength [T] (~20 nT)
D_tail = 1.0 * R_E           # current sheet half-thickness (~ 1 R_E typical)

z_tail = np.linspace(-5 * R_E, 5 * R_E, 500)
B_x_harris = B0_tail * np.tanh(z_tail / D_tail)
J_y_harris = (B0_tail / (mu_0 * D_tail)) / np.cosh(z_tail / D_tail)**2

J_peak = B0_tail / (mu_0 * D_tail)
print(f"\nTail Current Sheet (Harris model):")
print(f"  Lobe field B0       = {B0_tail*1e9:.1f} nT")
print(f"  Half-thickness D    = {D_tail/R_E:.1f} R_E")
print(f"  Peak current J_peak = {J_peak*1e9:.2f} nA/m^2")


# =========================================================================
# 3. BIRKELAND (FIELD-ALIGNED) CURRENTS
# =========================================================================
# Birkeland currents connect the magnetosphere to the ionosphere along
# magnetic field lines. They are organized in two concentric rings:
#   Region 1 (poleward, ~70-75 deg MLAT): connects to magnetopause
#   Region 2 (equatorward, ~65-70 deg MLAT): connects to ring current
#
# Simplified model: J_parallel = J0 * cos(phi) at ionospheric altitude
# where phi is magnetic local time (MLT) angle.
# Region 1: upward (away from ionosphere) on dusk side
# Region 2: upward on dawn side (opposite sense)

phi_mlt = np.linspace(0, 2 * np.pi, 360)  # MLT angle, 0 = midnight

# Current densities (positive = upward/away from Earth)
J0_R1 = 1.0   # microA/m^2 (typical quiet time)
J0_R2 = 0.7   # microA/m^2

J_R1 = J0_R1 * np.sin(phi_mlt)   # Region 1: sin pattern (upward on dusk)
J_R2 = -J0_R2 * np.sin(phi_mlt)  # Region 2: opposite sense

# Ionospheric latitudes for the two regions
mlat_R1 = 72.5  # degrees (average Region 1 latitude)
mlat_R2 = 67.5  # degrees (average Region 2 latitude)

# Total current: integrate over the ring
# I_total ~ J0 * 2 * pi * R_E * cos(mlat) * delta_mlat * R_E
delta_mlat = 2.5 * np.pi / 180  # width of each region
I_R1_total = J0_R1 * 1e-6 * 2 * R_E * np.cos(np.radians(mlat_R1)) * delta_mlat * R_E
print(f"\nBirkeland Currents (FAC):")
print(f"  Region 1: J0 = {J0_R1:.1f} uA/m^2, MLAT ~ {mlat_R1} deg")
print(f"  Region 2: J0 = {J0_R2:.1f} uA/m^2, MLAT ~ {mlat_R2} deg")
print(f"  Estimated total R1 current ~ {I_R1_total/1e6:.2f} MA (each hemisphere)")


# =========================================================================
# 4. PLOTTING
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Magnetospheric Current Systems", fontsize=15, y=1.01)

# --- Panel 1: Ring current on-axis field ---
ax = axes[0, 0]
ax.plot(z_axis / R_E, B_ring_axis * 1e9, 'b-', lw=2, label='Ring current $B_z$ (on-axis)')
ax.axhline(Delta_B_center * 1e9, color='r', ls='--', lw=1,
           label=f'$\\Delta B$ at center = {Delta_B_center*1e9:.1f} nT')
ax.axvspan(-L_ring, -L_ring * 0.8, color='orange', alpha=0.3)
ax.axvspan(L_ring * 0.8, L_ring, color='orange', alpha=0.3, label='Ring current location')

ax.set_xlabel("Distance along dipole axis [$R_E$]", fontsize=11)
ax.set_ylabel("$\\Delta B_z$ [nT]", fontsize=11)
ax.set_title(f"Ring Current Field (I = {I_ring/1e6:.0f} MA, L = {L_ring})", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: Harris current sheet ---
ax = axes[0, 1]
ax2 = ax.twinx()

line1, = ax.plot(z_tail / R_E, B_x_harris * 1e9, 'b-', lw=2, label='$B_x$ (Harris)')
line2, = ax2.plot(z_tail / R_E, J_y_harris * 1e9, 'r-', lw=2, label='$J_y$ (current density)')

ax.set_xlabel("$z$ (north-south) [$R_E$]", fontsize=11)
ax.set_ylabel("$B_x$ [nT]", fontsize=11, color='b')
ax2.set_ylabel("$J_y$ [nA/m$^2$]", fontsize=11, color='r')
ax.set_title("Tail Current Sheet (Harris Model)", fontsize=12)
ax.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# --- Panel 3: Birkeland currents (polar view) ---
ax = axes[1, 0]
ax = plt.subplot(223, projection='polar')

# Region 1 (poleward) - shown at inner ring
r_R1 = np.full_like(phi_mlt, 90 - mlat_R1)  # convert MLAT to colatitude for polar plot
r_R2 = np.full_like(phi_mlt, 90 - mlat_R2)

# Color-code by current direction
norm_R1 = J_R1 / J0_R1
norm_R2 = J_R2 / J0_R2

# Plot as colored scatter
sc1 = ax.scatter(phi_mlt, r_R1, c=norm_R1, cmap='bwr', s=15,
                 vmin=-1, vmax=1, label='Region 1', zorder=3)
sc2 = ax.scatter(phi_mlt, r_R2, c=norm_R2, cmap='bwr', s=15,
                 vmin=-1, vmax=1, label='Region 2', zorder=3)

ax.set_theta_zero_location('S')  # midnight at bottom
ax.set_theta_direction(-1)       # clockwise (MLT convention)
ax.set_ylim(0, 30)
ax.set_yticks([10, 20, 30])
ax.set_yticklabels(['80', '70', '60'])
ax.set_title("Birkeland Currents\n(Red=upward, Blue=downward)", fontsize=11, pad=15)

# MLT labels
ax.set_xticks(np.array([0, 6, 12, 18]) * np.pi / 12)
ax.set_xticklabels(['00 MLT\n(midnight)', '06 MLT\n(dawn)',
                    '12 MLT\n(noon)', '18 MLT\n(dusk)'], fontsize=8)

# --- Panel 4: Dessler-Parker-Sckopke relation ---
ax = axes[1, 1]
ax.plot(E_RC_values / 1e15, Dst_DPS, 'b-', lw=2)
ax.axhline(-50, color='orange', ls='--', alpha=0.7, label='Moderate storm (-50 nT)')
ax.axhline(-100, color='red', ls='--', alpha=0.7, label='Intense storm (-100 nT)')
ax.axhline(-250, color='darkred', ls='--', alpha=0.7, label='Superstorm (-250 nT)')

ax.set_xlabel("Ring current energy $E_{RC}$ [$\\times 10^{15}$ J]", fontsize=11)
ax.set_ylabel("Dst [nT]", fontsize=11)
ax.set_title("Dessler-Parker-Sckopke Relation", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Space_Weather/02_current_systems.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: 02_current_systems.png")

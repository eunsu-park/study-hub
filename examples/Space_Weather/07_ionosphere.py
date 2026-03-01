"""
Ionospheric Electron Density and GPS Signal Propagation.

Demonstrates:
- Chapman production function for ionospheric layer formation
- Electron density profiles for D, E, F1, F2 layers at various solar zenith angles
- Total Electron Content (TEC) computation by numerical integration
- GPS signal delay from TEC (dual-frequency L1/L2)
- Simplified scintillation index S4 model
- Dependence of ionospheric structure on solar illumination

Physics:
    The ionosphere is created by solar EUV/X-ray photoionization of the neutral
    atmosphere. The Chapman production function describes the ion production rate
    at altitude z for a plane-stratified atmosphere:

        q(z) = q_max * exp(1 - z' - sec(chi) * exp(-z'))

    where z' = (z - z_max) / H is the reduced height, chi is the solar zenith
    angle, and H is the neutral scale height. The resulting electron density
    profile (in photochemical equilibrium where production = loss) is:

        N_e(z) ~ sqrt(q(z) / alpha_r)

    where alpha_r is the recombination coefficient.

    The ionosphere is conventionally divided into layers:
        D layer (~60-90 km):   z_max~80 km,  H~5 km,  weak, daytime only
        E layer (~90-150 km):  z_max~110 km, H~10 km, moderate
        F1 layer (~150-220 km): z_max~180 km, H~25 km, daytime
        F2 layer (~220-600 km): z_max~300 km, H~50 km, strongest, persists at night

    GPS signals are delayed by the ionosphere:
        dt = 40.3 * TEC / f^2
    where TEC = integral of N_e along the signal path (TECU = 1e16 el/m^2)
    and f is the signal frequency (Hz).

    Scintillation (rapid amplitude fluctuations) is quantified by S4:
        S4 = sqrt(<I^2> - <I>^2) / <I>
    where I is signal intensity. S4 > 0.5 indicates strong scintillation.

References:
    - Chapman, S. (1931). "The absorption and dissociative or ionizing effect
      of monochromatic radiation in an atmosphere on a rotating earth."
    - Klobuchar, J.A. (1987). "Ionospheric time-delay algorithm for
      single-frequency GPS users."
"""

import numpy as np
import matplotlib.pyplot as plt

# === Physical Constants ===
R_E = 6.371e6    # Earth radius [m]
k_B = 1.381e-23  # Boltzmann constant [J/K]

# GPS frequencies
f_L1 = 1575.42e6  # GPS L1 frequency [Hz]
f_L2 = 1227.60e6  # GPS L2 frequency [Hz]

print("=" * 65)
print("IONOSPHERIC ELECTRON DENSITY AND GPS SIGNAL PROPAGATION")
print("=" * 65)


# =========================================================================
# 1. CHAPMAN PRODUCTION FUNCTION
# =========================================================================
def chapman_production(z, z_max, H, chi_deg, q_max=1.0):
    """
    Chapman production function for ionization rate.

    Parameters:
        z       : altitude array [km]
        z_max   : altitude of maximum production [km]
        H       : neutral scale height [km]
        chi_deg : solar zenith angle [degrees]
        q_max   : peak production rate [arbitrary units]

    Returns:
        q(z) : production rate at each altitude
    """
    chi = np.radians(chi_deg)
    # Limit sec(chi) to avoid divergence near chi=90
    sec_chi = np.clip(1.0 / np.cos(chi), 1.0, 50.0)
    z_prime = (z - z_max) / H
    q = q_max * np.exp(1.0 - z_prime - sec_chi * np.exp(-z_prime))
    return q


def electron_density(z, z_max, H, chi_deg, N_max):
    """
    Electron density from Chapman profile (photochemical equilibrium).

    In the photochemical equilibrium regime (below ~200 km),
    N_e ~ sqrt(q / alpha_r). We normalize so that N_e(z_max, chi=0) = N_max.

    Parameters:
        z       : altitude [km]
        z_max   : peak altitude [km]
        H       : scale height [km]
        chi_deg : solar zenith angle [degrees]
        N_max   : peak electron density at chi=0 [el/m^3]

    Returns:
        N_e(z) : electron density [el/m^3]
    """
    q = chapman_production(z, z_max, H, chi_deg)
    q0 = chapman_production(z_max, z_max, H, 0.0)  # normalization
    # N_e ~ sqrt(q), normalized to N_max
    N_e = N_max * np.sqrt(q / q0)
    return N_e


# === Ionospheric Layer Parameters ===
# Each layer: (name, z_max [km], H [km], N_max at chi=0 [el/m^3])
layers = {
    'D':  {'z_max': 80,  'H': 5,   'N_max': 1e9,   'color': 'purple'},
    'E':  {'z_max': 110, 'H': 10,  'N_max': 1e11,  'color': 'blue'},
    'F1': {'z_max': 180, 'H': 25,  'N_max': 3e11,  'color': 'green'},
    'F2': {'z_max': 300, 'H': 50,  'N_max': 1e12,  'color': 'red'},
}

# Altitude grid
z = np.linspace(50, 800, 2000)  # [km]

print("\n--- Ionospheric Layer Parameters ---")
print(f"{'Layer':<6} {'z_max [km]':<12} {'H [km]':<10} {'N_max [el/m³]':<15}")
for name, params in layers.items():
    print(f"{name:<6} {params['z_max']:<12} {params['H']:<10} {params['N_max']:<15.1e}")


# =========================================================================
# 2. ELECTRON DENSITY PROFILES FOR DIFFERENT ZENITH ANGLES
# =========================================================================
chi_values = [0, 30, 60, 75, 85]  # solar zenith angles [deg]

print("\n--- Peak Electron Densities at Different Zenith Angles ---")
print(f"{'chi [deg]':<12}", end="")
for name in layers:
    print(f"{name + ' [el/m³]':<16}", end="")
print()

for chi in chi_values:
    print(f"{chi:<12}", end="")
    for name, params in layers.items():
        Ne = electron_density(params['z_max'], params['z_max'],
                              params['H'], chi, params['N_max'])
        print(f"{Ne:<16.2e}", end="")
    print()


# =========================================================================
# 3. TOTAL ELECTRON CONTENT (TEC)
# =========================================================================
def compute_tec(z_km, chi_deg):
    """
    Compute vertical TEC by integrating N_e over all layers.

    TEC = integral of N_e dz from bottom to top [el/m^2]
    1 TECU = 1e16 el/m^2
    """
    N_total = np.zeros_like(z_km, dtype=float)
    for name, params in layers.items():
        N_total += electron_density(z_km, params['z_max'], params['H'],
                                    chi_deg, params['N_max'])
    # Integrate (z in km, convert to m)
    tec = np.trapezoid(N_total, z_km * 1e3)  # [el/m^2]
    return tec, N_total


chi_scan = np.linspace(0, 85, 50)
tec_values = np.array([compute_tec(z, chi)[0] for chi in chi_scan])

print("\n--- Total Electron Content vs Solar Zenith Angle ---")
for chi in [0, 30, 60, 75, 85]:
    tec, _ = compute_tec(z, chi)
    print(f"  chi = {chi:>2d} deg: TEC = {tec:.2e} el/m^2 "
          f"= {tec / 1e16:.1f} TECU")


# =========================================================================
# 4. GPS SIGNAL DELAY
# =========================================================================
def gps_range_delay(tec, freq):
    """
    Ionospheric range delay for GPS signal.

    The group delay (pseudorange error) is:
        delta_rho = 40.3 * TEC / f^2  [meters]

    This is the extra apparent path length caused by the ionosphere.
    The corresponding time delay is delta_rho / c.

    Parameters:
        tec  : Total Electron Content [el/m^2]
        freq : signal frequency [Hz]

    Returns:
        delta_rho : range delay [meters]
    """
    return 40.3 * tec / freq**2


print("\n--- GPS Ionospheric Delay (vertical, chi=30 deg) ---")
tec_30, _ = compute_tec(z, 30)
range_L1 = gps_range_delay(tec_30, f_L1)  # [m]
range_L2 = gps_range_delay(tec_30, f_L2)
dt_L1 = range_L1 / 3e8  # time delay [s]
dt_L2 = range_L2 / 3e8

print(f"  TEC = {tec_30 / 1e16:.1f} TECU")
print(f"  L1 ({f_L1 / 1e6:.2f} MHz): range error = {range_L1:.2f} m, "
      f"dt = {dt_L1 * 1e9:.1f} ns")
print(f"  L2 ({f_L2 / 1e6:.2f} MHz): range error = {range_L2:.2f} m, "
      f"dt = {dt_L2 * 1e9:.1f} ns")
print(f"  Differential range (L2-L1): {range_L2 - range_L1:.2f} m")

# GPS range delay vs TEC
tec_range = np.linspace(1e16, 100e16, 200)  # 1-100 TECU
delay_L1 = gps_range_delay(tec_range, f_L1)  # [m]
delay_L2 = gps_range_delay(tec_range, f_L2)  # [m]


# =========================================================================
# 5. SCINTILLATION INDEX S4
# =========================================================================
def scintillation_s4(chi_deg, f10_7=150, latitude_deg=10):
    """
    Simplified scintillation index S4 model.

    S4 depends on local time (proxy: chi), solar activity (F10.7),
    and magnetic latitude. Strongest at equatorial/low latitudes
    after sunset (large chi).

    This is a simplified empirical model for demonstration.
    """
    # Post-sunset enhancement (chi > 90 means nightside, but we model
    # enhancement near twilight chi ~ 80-90)
    chi_factor = np.exp(-((chi_deg - 85)**2) / (2 * 15**2))

    # Solar activity dependence
    f107_factor = (f10_7 / 150.0)**0.8

    # Latitude dependence (peak near magnetic equator)
    lat_factor = np.exp(-(latitude_deg**2) / (2 * 20**2))

    # Base S4 (can exceed 1 in severe cases)
    s4 = 0.8 * chi_factor * f107_factor * lat_factor

    # Add some randomness for realism
    s4 = s4 * (1 + 0.2 * np.sin(chi_deg * 0.1))
    return np.clip(s4, 0, 1.5)


chi_s4 = np.linspace(0, 90, 200)
s4_low = np.array([scintillation_s4(c, f10_7=70) for c in chi_s4])
s4_high = np.array([scintillation_s4(c, f10_7=200) for c in chi_s4])

print("\n--- Scintillation Index S4 (equatorial, different solar activity) ---")
print(f"  Solar minimum (F10.7=70):  max S4 = {s4_low.max():.2f}")
print(f"  Solar maximum (F10.7=200): max S4 = {s4_high.max():.2f}")
print(f"  S4 > 0.5 threshold indicates strong scintillation")


# =========================================================================
# 6. PLOTTING
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# --- Panel 1: N_e profiles for different chi ---
ax = axes[0, 0]
for chi in chi_values:
    _, N_total = compute_tec(z, chi)
    ax.semilogx(N_total, z, linewidth=2, label=f'$\\chi = {chi}°$')

# Mark layer peaks
for name, params in layers.items():
    ax.axhline(params['z_max'], color=params['color'], linestyle=':',
               alpha=0.4)
    ax.text(5e7, params['z_max'] + 5, f'{name} layer',
            fontsize=8, color=params['color'])

ax.set_xlabel('Electron Density [el/m³]')
ax.set_ylabel('Altitude [km]')
ax.set_title('Ionospheric Electron Density Profiles')
ax.set_xlim(1e7, 2e12)
ax.set_ylim(50, 600)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 2: Individual layer profiles (chi=30) ---
ax = axes[0, 1]
chi_show = 30
for name, params in layers.items():
    Ne = electron_density(z, params['z_max'], params['H'],
                          chi_show, params['N_max'])
    ax.semilogx(Ne, z, color=params['color'], linewidth=2, label=f'{name} layer')

# Show total
_, N_total = compute_tec(z, chi_show)
ax.semilogx(N_total, z, 'k--', linewidth=2.5, label='Total', alpha=0.7)

ax.set_xlabel('Electron Density [el/m³]')
ax.set_ylabel('Altitude [km]')
ax.set_title(f'Layer Decomposition ($\\chi = {chi_show}°$)')
ax.set_xlim(1e6, 2e12)
ax.set_ylim(50, 600)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 3: TEC vs solar zenith angle ---
ax = axes[0, 2]
ax.plot(chi_scan, tec_values / 1e16, 'b-', linewidth=2.5)
ax.set_xlabel('Solar Zenith Angle [degrees]')
ax.set_ylabel('TEC [TECU]')
ax.set_title('Total Electron Content vs Zenith Angle')
ax.grid(True, alpha=0.3)
# Mark typical values
for chi_mark in [0, 30, 60]:
    idx = np.argmin(np.abs(chi_scan - chi_mark))
    ax.plot(chi_scan[idx], tec_values[idx] / 1e16, 'ro', markersize=8)
    ax.annotate(f'{tec_values[idx] / 1e16:.1f} TECU',
                xy=(chi_scan[idx], tec_values[idx] / 1e16),
                xytext=(chi_scan[idx] + 3, tec_values[idx] / 1e16 + 1),
                fontsize=8)

# --- Panel 4: GPS range delay vs TEC ---
ax = axes[1, 0]
ax.plot(tec_range / 1e16, delay_L1, 'b-', linewidth=2, label='L1 (1575.42 MHz)')
ax.plot(tec_range / 1e16, delay_L2, 'r-', linewidth=2, label='L2 (1227.60 MHz)')
ax.set_xlabel('TEC [TECU]')
ax.set_ylabel('Range Delay [m]')
ax.set_title('GPS Ionospheric Range Error vs TEC')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add secondary y-axis for time delay
ax2 = ax.twinx()
ax2.set_ylabel('Time Delay [ns]')
y1_min, y1_max = ax.get_ylim()
ax2.set_ylim(y1_min / 0.3, y1_max / 0.3)  # range [m] / c [m/ns]

# --- Panel 5: Scintillation S4 ---
ax = axes[1, 1]
ax.plot(chi_s4, s4_low, 'b-', linewidth=2, label='Solar min (F10.7=70)')
ax.plot(chi_s4, s4_high, 'r-', linewidth=2, label='Solar max (F10.7=200)')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5,
           label='Strong scintillation threshold')
ax.fill_between(chi_s4, 0.5, 1.5, alpha=0.1, color='orange')
ax.set_xlabel('Solar Zenith Angle [degrees]')
ax.set_ylabel('Scintillation Index S4')
ax.set_title('Scintillation Index (equatorial)')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 6: Chapman production function shape ---
ax = axes[1, 2]
z_norm = np.linspace(-5, 5, 500)  # normalized altitude z'
for chi in [0, 30, 60, 75]:
    sec_chi = 1.0 / np.cos(np.radians(chi))
    q = np.exp(1.0 - z_norm - sec_chi * np.exp(-z_norm))
    ax.plot(q, z_norm, linewidth=2, label=f'$\\chi = {chi}°$')

ax.set_xlabel('Normalized Production Rate q/q_max')
ax.set_ylabel("Reduced Height z' = (z - z_max) / H")
ax.set_title('Chapman Production Function')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Space_Weather/07_ionosphere.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nKey insights:")
print("  - F2 layer dominates TEC due to highest density and largest scale height")
print("  - TEC decreases with increasing zenith angle (less solar illumination)")
print("  - GPS L2 is more affected than L1 (delay ~ 1/f^2)")
print("  - Dual-frequency receivers can correct for ionospheric delay")
print("  - Scintillation peaks near sunset at equatorial latitudes")
print("\nPlot saved to 07_ionosphere.png")

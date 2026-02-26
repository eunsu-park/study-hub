"""
Thermospheric Density and Satellite Orbital Drag.

Demonstrates:
- Exponential atmospheric density model with altitude-dependent scale height
- Solar activity (F10.7) influence on thermospheric density
- Aerodynamic drag acceleration on LEO satellites
- King-Hele orbital decay equation (semi-analytical)
- Orbit lifetime estimation under quiet, active, and storm conditions
- Dramatic difference in satellite lifetimes between solar minimum and maximum

Physics:
    Above ~100 km, the thermosphere is heated primarily by solar EUV radiation.
    The neutral density decreases approximately exponentially with altitude:

        rho(h) = rho_0 * exp(-(h - h_0) / H(h))

    The scale height H depends on temperature (and thus solar activity):
        H = k_B * T / (m_avg * g)

    where T increases with F10.7 solar flux index. During geomagnetic storms,
    Joule heating and particle precipitation further inflate the thermosphere,
    increasing density at satellite altitudes by factors of 2-10.

    Atmospheric drag decelerates the satellite:
        a_drag = -0.5 * C_D * (A/m) * rho * v^2

    where C_D ~ 2.2 (free molecular flow), A/m is area-to-mass ratio.

    For a nearly circular orbit, the King-Hele decay rate gives:
        da/dt ~ -2*pi * C_D * (A/m) * rho * a^2 / P

    where a is semi-major axis and P is orbital period. The orbit spirals
    inward slowly until atmospheric density increases enough for rapid reentry.

References:
    - King-Hele, D. (1987). "Satellite Orbits in an Atmosphere."
    - Jacchia, L. (1977). "Thermospheric Temperature, Density, and Composition."
    - Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications."
"""

import numpy as np
import matplotlib.pyplot as plt

# === Physical Constants ===
R_E = 6.371e6      # Earth radius [m]
g_0 = 9.80665      # surface gravity [m/s^2]
mu_E = 3.986e14     # Earth gravitational parameter [m^3/s^2]
k_B = 1.381e-23     # Boltzmann constant [J/K]
m_O = 16 * 1.661e-27  # atomic oxygen mass [kg] (dominant species ~300 km)

print("=" * 65)
print("THERMOSPHERIC DENSITY AND SATELLITE ORBITAL DRAG")
print("=" * 65)


# =========================================================================
# 1. ATMOSPHERIC DENSITY MODEL
# =========================================================================
def exospheric_temperature(f107):
    """
    Simplified exospheric temperature from F10.7 solar flux.

    Based on Jacchia-type empirical relation. The exospheric temperature
    T_inf sets the thermospheric density profile above ~200 km.

    Parameters:
        f107 : F10.7 solar radio flux [SFU] (10.7 cm wavelength)

    Returns:
        T_inf : exospheric temperature [K]
    """
    # Empirical: T_inf ~ 500 + 3.5 * F10.7 (simplified from Jacchia-71)
    return 500.0 + 3.5 * f107


def scale_height(h_km, f107, storm_factor=1.0):
    """
    Altitude-dependent atmospheric scale height.

    H = k_B * T(h) / (m_avg * g(h))

    Temperature increases from ~200 K at 120 km to T_inf above ~400 km.
    Storm heating increases the effective temperature by storm_factor.

    Parameters:
        h_km          : altitude [km]
        f107          : F10.7 index [SFU]
        storm_factor  : multiplicative factor for storm heating (1.0 = quiet)

    Returns:
        H : scale height [km]
    """
    T_inf = exospheric_temperature(f107) * storm_factor
    T_120 = 200.0  # temperature at 120 km [K]

    # Bates temperature profile: T(h) = T_inf - (T_inf - T_120) * exp(-s*(h-120))
    s = 0.025  # shape parameter [1/km]
    h = np.maximum(h_km, 120.0)
    T = T_inf - (T_inf - T_120) * np.exp(-s * (h - 120.0))

    # Gravity decreases with altitude
    g = g_0 * (R_E / (R_E + h * 1e3))**2

    # Mean molecular mass transitions from ~28 amu (N2) below 200 km
    # to ~16 amu (O) above 300 km (atomic oxygen dominance)
    m_avg_amu = 28.0 - 12.0 * np.clip((h - 150.0) / 200.0, 0, 1)
    m_avg = m_avg_amu * 1.661e-27  # [kg]

    H = k_B * T / (m_avg * g) / 1e3  # [km]
    return H


def atmospheric_density(h_km, f107, storm_factor=1.0):
    """
    Thermospheric mass density using piecewise exponential model.

    Starting from a reference density at 120 km, integrate upward
    using altitude-dependent scale height.

    Parameters:
        h_km          : altitude [km] (scalar or array)
        f107          : F10.7 index [SFU]
        storm_factor  : storm heating multiplier (1.0 = quiet)

    Returns:
        rho : mass density [kg/m^3]
    """
    h = np.atleast_1d(np.float64(h_km))
    rho_120 = 2.0e-8  # reference density at 120 km [kg/m^3]

    # Numerical integration of dh/H from 120 km to h
    rho = np.zeros_like(h)
    for i, hi in enumerate(h):
        if hi <= 120:
            # Below thermosphere: use simple exponential with H=7 km
            rho[i] = rho_120 * np.exp(-(120 - hi) / 7.0)
        else:
            # Integrate from 120 to hi
            n_steps = max(int((hi - 120) / 0.5), 10)
            h_arr = np.linspace(120, hi, n_steps)
            H_arr = scale_height(h_arr, f107, storm_factor)
            # Cumulative integral of 1/H
            integrand = 1.0 / H_arr
            integral = np.trapezoid(integrand, h_arr)
            rho[i] = rho_120 * np.exp(-integral)

    return rho.squeeze()


# === Display Density at Key Altitudes ===
altitudes_check = [200, 300, 400, 500, 600, 800]
conditions = {
    'Quiet (F10.7=70)':  {'f107': 70,  'storm': 1.0},
    'Active (F10.7=200)': {'f107': 200, 'storm': 1.0},
    'Storm (F10.7=200+heating)': {'f107': 200, 'storm': 1.5},
}

print("\n--- Thermospheric Density [kg/m³] ---")
print(f"{'Alt [km]':<12}", end="")
for label in conditions:
    print(f"{label:<28}", end="")
print()

for h in altitudes_check:
    print(f"{h:<12}", end="")
    for label, cond in conditions.items():
        rho = atmospheric_density(h, cond['f107'], cond['storm'])
        print(f"{rho:<28.3e}", end="")
    print()


# =========================================================================
# 2. DRAG ACCELERATION
# =========================================================================
def orbital_velocity(h_km):
    """Circular orbital velocity [m/s] at altitude h."""
    r = R_E + h_km * 1e3
    return np.sqrt(mu_E / r)


def orbital_period(h_km):
    """Orbital period [s] at altitude h."""
    r = R_E + h_km * 1e3
    return 2 * np.pi * np.sqrt(r**3 / mu_E)


def drag_acceleration(h_km, f107, storm_factor, C_D=2.2, A_over_m=0.01):
    """
    Aerodynamic drag acceleration magnitude.

    a_drag = 0.5 * C_D * (A/m) * rho * v^2

    Parameters:
        h_km      : altitude [km]
        f107      : F10.7 index
        storm_factor : storm heating multiplier
        C_D       : drag coefficient (~2.2 for free molecular flow)
        A_over_m  : area-to-mass ratio [m^2/kg] (typical: 0.005-0.02)

    Returns:
        a_drag : drag acceleration magnitude [m/s^2]
    """
    rho = atmospheric_density(h_km, f107, storm_factor)
    v = orbital_velocity(h_km)
    return 0.5 * C_D * A_over_m * rho * v**2


print("\n--- Drag Acceleration at 400 km (A/m = 0.01 m²/kg) ---")
for label, cond in conditions.items():
    a_d = drag_acceleration(400, cond['f107'], cond['storm'])
    print(f"  {label}: a_drag = {a_d:.3e} m/s²")


# =========================================================================
# 3. ORBITAL DECAY SIMULATION (KING-HELE)
# =========================================================================
def simulate_decay(h_init_km, f107, storm_factor, C_D=2.2, A_over_m=0.01,
                   max_days=3650, dt_orbits=1):
    """
    Simulate orbital decay using King-Hele's formula.

    da/rev ~ -2*pi * C_D * (A/m) * rho * a^2

    We step one orbit at a time, updating density each step.

    Parameters:
        h_init_km   : initial altitude [km]
        f107        : F10.7 index (constant)
        storm_factor: storm heating multiplier
        C_D         : drag coefficient
        A_over_m    : area-to-mass ratio [m^2/kg]
        max_days    : maximum simulation time [days]
        dt_orbits   : orbits per step

    Returns:
        times : time array [days]
        alts  : altitude array [km]
    """
    a = R_E + h_init_km * 1e3  # semi-major axis [m]
    times = [0.0]
    alts = [h_init_km]
    t = 0.0
    reentry_alt = 120.0  # [km]

    while t < max_days * 86400 and (a - R_E) / 1e3 > reentry_alt:
        h_km = (a - R_E) / 1e3
        rho = atmospheric_density(h_km, f107, storm_factor)
        P = 2 * np.pi * np.sqrt(a**3 / mu_E)  # period [s]

        # King-Hele: change in semi-major axis per orbit
        da_per_orbit = -2 * np.pi * C_D * A_over_m * rho * a**2

        # Advance by dt_orbits orbits
        a += da_per_orbit * dt_orbits
        t += P * dt_orbits

        times.append(t / 86400)  # convert to days
        alts.append((a - R_E) / 1e3)

        # Adaptive step: smaller steps at lower altitudes (faster decay)
        if h_km < 200:
            dt_orbits = 1
        elif h_km < 300:
            dt_orbits = 5
        else:
            dt_orbits = 20

    return np.array(times), np.array(alts)


# === Run Simulations ===
print("\n--- Orbital Decay Simulation (400 km initial, A/m = 0.01 m²/kg) ---")
decay_results = {}
for label, cond in conditions.items():
    t_arr, h_arr = simulate_decay(400, cond['f107'], cond['storm'])
    decay_results[label] = (t_arr, h_arr)
    lifetime_days = t_arr[-1]
    print(f"  {label}: lifetime ~ {lifetime_days:.0f} days "
          f"({lifetime_days / 365.25:.1f} years)")


# =========================================================================
# 4. ORBITAL LIFETIME VS INITIAL ALTITUDE
# =========================================================================
h_init_range = np.arange(200, 801, 25)
lifetimes = {label: [] for label in conditions}

print("\n--- Computing lifetimes for various initial altitudes ---")
for h_init in h_init_range:
    for label, cond in conditions.items():
        t_arr, _ = simulate_decay(h_init, cond['f107'], cond['storm'],
                                  max_days=36500)  # 100 years max
        lifetimes[label].append(t_arr[-1] / 365.25)  # years

print("  Done. Range: 200-800 km.")


# =========================================================================
# 5. PLOTTING
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

colors = {'Quiet (F10.7=70)': 'blue',
          'Active (F10.7=200)': 'red',
          'Storm (F10.7=200+heating)': 'darkred'}

# --- Panel 1: Density vs altitude ---
ax = axes[0, 0]
h_plot = np.arange(150, 801, 5)
for label, cond in conditions.items():
    rho_arr = np.array([atmospheric_density(h, cond['f107'], cond['storm'])
                        for h in h_plot])
    ax.semilogy(rho_arr, h_plot, color=colors[label], linewidth=2, label=label)

# Mark ISS and typical LEO altitudes
for h_mark, name in [(400, 'ISS (~400 km)'), (550, 'LEO (~550 km)')]:
    ax.axhline(h_mark, color='gray', linestyle=':', alpha=0.5)
    ax.text(1e-14, h_mark + 10, name, fontsize=8, color='gray')

ax.set_xlabel('Mass Density [kg/m³]')
ax.set_ylabel('Altitude [km]')
ax.set_title('Thermospheric Density Profile')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1e-16, 1e-8)
ax.set_ylim(150, 800)

# --- Panel 2: Altitude vs time (orbital decay) ---
ax = axes[0, 1]
for label, (t_arr, h_arr) in decay_results.items():
    ax.plot(t_arr, h_arr, color=colors[label], linewidth=2, label=label)

ax.axhline(120, color='black', linestyle='--', alpha=0.5, label='Reentry (~120 km)')
ax.set_xlabel('Time [days]')
ax.set_ylabel('Altitude [km]')
ax.set_title('Orbital Decay (initial alt = 400 km)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(100, 420)

# --- Panel 3: Lifetime vs initial altitude ---
ax = axes[1, 0]
for label in conditions:
    ax.semilogy(h_init_range, lifetimes[label], color=colors[label],
                linewidth=2, label=label, marker='o', markersize=3)

ax.axhline(25, color='gray', linestyle=':', alpha=0.5)
ax.text(210, 30, '25-year guideline', fontsize=8, color='gray')
ax.set_xlabel('Initial Altitude [km]')
ax.set_ylabel('Orbital Lifetime [years]')
ax.set_title('Orbital Lifetime vs Initial Altitude')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.001, 200)
ax.set_xlim(200, 800)

# --- Panel 4: Scale height and temperature vs altitude ---
ax = axes[1, 1]
h_sh = np.linspace(120, 800, 200)
ax_twin = ax.twiny()

for label, cond in conditions.items():
    H_arr = scale_height(h_sh, cond['f107'], cond['storm'])
    ax.plot(H_arr, h_sh, color=colors[label], linewidth=2, label=label)

ax.set_xlabel('Scale Height [km]')
ax.set_ylabel('Altitude [km]')
ax.set_title('Scale Height & Exospheric Temperature')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

# Show exospheric temperatures
for label, cond in conditions.items():
    T_inf = exospheric_temperature(cond['f107']) * cond['storm']
    ax.text(0.98, 0.95 - list(conditions.keys()).index(label) * 0.06,
            f'{label.split("(")[0].strip()}: T$_\\infty$ = {T_inf:.0f} K',
            transform=ax.transAxes, fontsize=8, ha='right',
            color=colors[label])

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Space_Weather/08_satellite_drag.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nKey insights:")
print("  - Thermospheric density varies by 1-2 orders of magnitude with solar activity")
print("  - Geomagnetic storms can temporarily increase density by factor ~2-5")
print("  - A 400 km satellite decays in ~1 year (solar max) vs ~5+ years (solar min)")
print("  - The 25-year deorbit guideline requires careful altitude selection")
print("  - Starlink (~550 km) relies on active drag management for deorbit compliance")
print("\nPlot saved to 08_satellite_drag.png")

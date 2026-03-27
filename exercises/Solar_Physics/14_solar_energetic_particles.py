"""
Exercises for Lesson 14: Solar Energetic Particles
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
c = 3.0e8              # speed of light [m/s]
m_p = 1.673e-27        # proton mass [kg]
e = 1.602e-19          # elementary charge [C]
AU = 1.496e11          # astronomical unit [m]


def exercise_1():
    """
    Problem 1: DSA Spectral Index

    CME shock with compression ratio r = 3.5.
    (a) DSA spectral index gamma for f(E) ~ E^(-gamma).
    (b) Convert to differential intensity: dJ/dE ~ E^(-delta), delta = gamma + 1.
    (c) Compare with strong-shock limit.
    (d) Conditions for r < 4.
    """
    r = 3.5  # compression ratio

    # DSA theory: spectral index of momentum distribution f(p) ~ p^(-q)
    # where q = 3*r / (r - 1)
    q = 3.0 * r / (r - 1.0)

    # For non-relativistic particles, E = p^2/(2m):
    # f(E) ~ E^(-gamma) where gamma = (q-1)/2 = (r+2)/(2*(r-1))
    gamma = (r + 2.0) / (2.0 * (r - 1.0))

    # (b) Differential intensity: dJ/dE ~ E^(-delta) where delta = gamma + 1
    # Actually, the relationship depends on the convention.
    # The problem states delta = gamma + 1
    delta = gamma + 1.0

    print(f"  (a) Compression ratio: r = {r}")
    print(f"      DSA momentum index: q = 3r/(r-1) = {q:.2f}")
    print(f"      Energy spectral index: gamma = (r+2)/(2(r-1)) = {gamma:.2f}")
    print(f"      f(E) ~ E^(-{gamma:.2f})")

    print(f"\n  (b) Differential intensity: dJ/dE ~ E^(-delta)")
    print(f"      delta = gamma + 1 = {delta:.2f}")
    print(f"      dJ/dE ~ E^(-{delta:.2f})")

    # (c) Strong-shock limit: r -> 4 (maximum for non-relativistic gas, gamma=5/3)
    r_strong = 4.0
    gamma_strong = (r_strong + 2.0) / (2.0 * (r_strong - 1.0))
    delta_strong = gamma_strong + 1.0

    print(f"\n  (c) Strong shock limit (r = 4):")
    print(f"      gamma = (4+2)/(2*3) = {gamma_strong:.2f}")
    print(f"      delta = {delta_strong:.2f}")
    print(f"      Our shock (r={r}): delta = {delta:.2f} (steeper than strong-shock limit)")
    print(f"      Weaker shocks (r < 4) produce steeper spectra (larger delta)")

    # (d) Conditions for r < 4
    print(f"\n  (d) Conditions producing r < 4:")
    print(f"      - The Rankine-Hugoniot limit r = (gamma_ad+1)/(gamma_ad-1) = 4")
    print(f"        applies for gamma_ad = 5/3 (monatomic gas) at infinite Mach number.")
    print(f"      - r < 4 when:")
    print(f"        * Mach number is finite (weak-to-moderate shock)")
    print(f"        * Upstream plasma has non-zero beta (thermal pressure)")
    print(f"        * Oblique shocks (magnetic field reduces compression)")
    print(f"        * Cosmic ray pressure modifies the shock structure")
    print(f"        * Relativistic particles (adiabatic index -> 4/3, r -> 7)")


def exercise_2():
    """
    Problem 2: 3He Enrichment

    Observed 3He/4He = 0.5 (solar value 5e-4).
    Gyrofrequency ratios and ICR wave heating explanation.
    """
    He3_He4_obs = 0.5
    He3_He4_solar = 5.0e-4

    # Enrichment factor
    enrichment = He3_He4_obs / He3_He4_solar

    print(f"  Observed 3He/4He = {He3_He4_obs}")
    print(f"  Solar value 3He/4He = {He3_He4_solar:.0e}")
    print(f"  Enrichment factor: {enrichment:.0f}")

    # Gyrofrequencies
    # 3He2+: q = 2e, m = 3*m_p => Omega = 2eB/(3m_p)
    # 4He2+: q = 2e, m = 4*m_p => Omega = 2eB/(4m_p) = eB/(2m_p)
    # Ratio: Omega(3He2+) / Omega(4He2+) = (2/(3)) / (1/2) = 4/3
    ratio_gyro = (2.0 / 3.0) / (1.0 / 2.0)

    print(f"\n  Gyrofrequencies:")
    print(f"    3He2+: Omega = 2eB/(3m_p)")
    print(f"    4He2+: Omega = 2eB/(4m_p) = eB/(2m_p)")
    print(f"    Ratio: Omega(3He2+)/Omega(4He2+) = (2/3)/(1/2) = {ratio_gyro:.3f}")

    print(f"\n  Ion cyclotron resonance explanation:")
    print(f"    - Electromagnetic ion cyclotron (EMIC) waves can resonate with")
    print(f"      ions at their gyrofrequency (or harmonics)")
    print(f"    - The ratio {ratio_gyro:.3f} means 3He2+ gyrates {ratio_gyro:.3f}x faster than 4He2+")
    print(f"    - If EMIC waves are generated near the 4He2+ gyrofrequency,")
    print(f"      they do NOT resonate with 4He2+ (since Omega(4He2+) is below")
    print(f"      the wave frequency for that wave mode)")
    print(f"    - However, 3He2+ with its higher gyrofrequency CAN resonate")
    print(f"      with these waves")
    print(f"    - This preferential resonance selectively accelerates 3He2+")
    print(f"      while leaving 4He2+ largely unaffected")
    print(f"    - The acceleration occurs in impulsive flare events where")
    print(f"      magnetic reconnection generates the necessary wave spectrum")
    print(f"    - This produces the observed extreme enrichment of 3He")


def exercise_3():
    """
    Problem 3: Velocity Dispersion Analysis

    100 MeV protons (v = 0.43c) arrive at t1 = 08:20 UT.
    10 MeV protons (v = 0.14c) arrive at t2 = 09:05 UT.
    (a) Calculate path length L.
    (b) Solar release time t0.
    (c) Compare with Parker spiral length (1.15 AU).
    """
    v1 = 0.43 * c        # m/s (100 MeV protons)
    v2 = 0.14 * c        # m/s (10 MeV protons)

    # Times in seconds from midnight
    t1_hr, t1_min = 8, 20
    t2_hr, t2_min = 9, 5
    t1 = t1_hr * 3600 + t1_min * 60  # s
    t2 = t2_hr * 3600 + t2_min * 60  # s
    dt = t2 - t1  # s

    print(f"  100 MeV protons: v1 = {v1/c:.2f}c, arrival = {t1_hr:02d}:{t1_min:02d} UT")
    print(f"  10 MeV protons:  v2 = {v2/c:.2f}c, arrival = {t2_hr:02d}:{t2_min:02d} UT")
    print(f"  Time difference: dt = {dt:.0f} s = {dt/60:.0f} min")

    # Scatter-free propagation: t_arrival = t_0 + L/v
    # t1 = t0 + L/v1
    # t2 = t0 + L/v2
    # dt = t2 - t1 = L * (1/v2 - 1/v1)
    # L = dt / (1/v2 - 1/v1)
    L = dt / (1.0/v2 - 1.0/v1)
    L_AU = L / AU

    print(f"\n  (a) Path length calculation:")
    print(f"      t_arrival = t_0 + L/v (scatter-free)")
    print(f"      L = dt / (1/v2 - 1/v1)")
    print(f"        = {dt:.0f} / ({1.0/v2:.4e} - {1.0/v1:.4e})")
    print(f"        = {L:.3e} m")
    print(f"        = {L_AU:.2f} AU")

    # (b) Solar release time
    t0 = t1 - L / v1  # seconds from midnight
    t0_hr = int(t0 // 3600)
    t0_min = int((t0 % 3600) // 60)
    t0_sec = int(t0 % 60)

    print(f"\n  (b) Solar release time:")
    print(f"      t_0 = t_1 - L/v_1 = {t1:.0f} - {L/v1:.0f}")
    print(f"          = {t0:.0f} s from midnight")
    print(f"          = {t0_hr:02d}:{t0_min:02d}:{t0_sec:02d} UT")

    # (c) Compare with Parker spiral
    L_Parker = 1.15 * AU
    ratio = L_AU / 1.15
    print(f"\n  (c) Parker spiral length: {1.15:.2f} AU")
    print(f"      Derived path length: {L_AU:.2f} AU")
    print(f"      Ratio: L / L_Parker = {ratio:.2f}")

    if abs(ratio - 1.0) < 0.2:
        print(f"      The path length is consistent with the Parker spiral field line,")
        print(f"      suggesting near scatter-free propagation along the IMF.")
    elif ratio > 1.2:
        print(f"      The path length exceeds the Parker spiral ({ratio:.1f}x),")
        print(f"      suggesting significant scattering or a non-standard field geometry.")
    else:
        print(f"      The path length is shorter than the Parker spiral,")
        print(f"      which may indicate a more direct magnetic connection.")


def exercise_4():
    """
    Problem 4: Astronaut Radiation Dose

    Proton fluence: 1e10 protons/cm^2 above 10 MeV.
    Average LET = 2 keV/um in tissue (rho = 1 g/cm^3).
    Quality factor Q = 2.
    Path length through tissue = 10 cm.
    """
    fluence = 1.0e10     # protons/cm^2
    LET = 2.0            # keV/um = 2e3 keV/mm = 2e4 keV/cm
    rho_tissue = 1.0     # g/cm^3
    Q = 2.0              # quality factor
    path = 10.0          # cm path through tissue

    # Convert LET to SI-like units
    # LET = 2 keV/um = 2e3 eV / um = 2e9 eV/m = 2e9 * 1.602e-19 J/m
    LET_J_per_m = 2.0e3 * 1.602e-19 / 1.0e-6  # J/m = keV/um in J/m
    # Simpler: 2 keV/um = 2e3 * 1.602e-19 J / 1e-6 m = 3.204e-10 J/m

    # Energy deposited per proton over 10 cm path
    # E_per_proton = LET * path
    E_per_proton_keV = LET * path * 1e4  # keV/um * 10 cm * 1e4 um/cm = keV
    E_per_proton_J = E_per_proton_keV * 1e3 * 1.602e-19  # keV -> eV -> J

    print(f"  SEP event parameters:")
    print(f"    Proton fluence: {fluence:.0e} protons/cm^2 above 10 MeV")
    print(f"    Average LET: {LET} keV/um")
    print(f"    Tissue density: {rho_tissue} g/cm^3")
    print(f"    Quality factor: Q = {Q}")
    print(f"    Path length: {path:.0f} cm")

    # Energy deposited per unit mass (dose)
    # Dose = fluence * LET * path / (rho * path) ? No...
    # Actually: each proton deposits LET * path length of energy
    # Total energy deposited per unit area = fluence * LET * path
    # Dose = energy per unit mass = (fluence * LET * path) / (rho * path)
    #       = fluence * LET / rho

    # Wait, let me think more carefully.
    # LET = energy loss per unit path length = dE/dx
    # Energy deposited in tissue of thickness d: E = LET * d (per proton)
    # Mass of tissue per unit area: m_area = rho * d
    # Dose = Energy / mass = fluence * LET * d / (rho * d) = fluence * LET / rho

    # In CGS: LET in keV/um = keV/(1e-4 cm) = 1e4 keV/cm
    LET_cgs = LET * 1.0e4  # keV/cm

    # Energy per proton per cm = LET_cgs keV/cm
    # Over path cm: E_per_proton = LET_cgs * path  [keV]
    E_per_proton = LET_cgs * path  # keV

    # Total energy per unit area = fluence * E_per_proton [keV/cm^2]
    E_per_area = fluence * E_per_proton  # keV/cm^2

    # Mass per unit area = rho * path [g/cm^2]
    mass_per_area = rho_tissue * path  # g/cm^2

    # Dose = E_per_area / mass_per_area [keV/g]
    dose_keV_g = E_per_area / mass_per_area  # keV/g

    # Convert to Gray: 1 Gy = 1 J/kg = 6.242e12 keV / 1000 g = 6.242e9 keV/g
    keV_per_g_to_Gy = 1.0 / (6.242e9)  # Gy per (keV/g)
    dose_Gy = dose_keV_g * keV_per_g_to_Gy

    # Dose equivalent in Sv
    dose_Sv = dose_Gy * Q

    print(f"\n  Calculation:")
    print(f"    LET = {LET} keV/um = {LET_cgs:.0e} keV/cm")
    print(f"    Energy deposited per proton over {path:.0f} cm:")
    print(f"      E = LET * path = {E_per_proton:.0e} keV = {E_per_proton/1e3:.0f} MeV")
    print(f"    Total energy per unit area:")
    print(f"      E/A = fluence * E = {E_per_area:.2e} keV/cm^2")
    print(f"    Mass per unit area: {mass_per_area:.0f} g/cm^2")
    print(f"    Absorbed dose:")
    print(f"      D = E/(A*m) = {dose_keV_g:.2e} keV/g = {dose_Gy:.3f} Gy")
    print(f"    Dose equivalent:")
    print(f"      H = D * Q = {dose_Gy:.3f} * {Q:.0f} = {dose_Sv:.3f} Sv = {dose_Sv*1e3:.0f} mSv")

    # NASA limits
    NASA_30day = 0.25  # Sv (250 mSv)
    print(f"\n  NASA 30-day limit: {NASA_30day*1e3:.0f} mSv")
    print(f"  Our dose: {dose_Sv*1e3:.0f} mSv")
    if dose_Sv > NASA_30day:
        print(f"  EXCEEDS the 30-day limit by {dose_Sv/NASA_30day:.1f}x!")
    else:
        print(f"  Within the 30-day limit (ratio: {dose_Sv/NASA_30day:.2f})")

    print(f"\n  Shielding strategies:")
    print(f"    - Passive shielding: polyethylene (hydrogen-rich) 5-20 g/cm^2")
    print(f"    - Storm shelter: thicker shielding in a small volume")
    print(f"    - Water walls: dual-use (drinking water + radiation shield)")
    print(f"    - Active magnetic shielding (conceptual, high mass)")
    print(f"    - Warning systems: 30-60 min forecast for crew to reach shelter")


def exercise_5():
    """
    Problem 5: Type II Burst Shock Speed

    200 MHz -> 50 MHz over 8 minutes.
    f_p = 9000 * sqrt(n_e) kHz (n_e in cm^-3).
    n_e(r) ~ r^(-2). Estimate densities, distance ratio, shock speed.
    """
    f_start = 200.0e6    # Hz
    f_end = 50.0e6       # Hz
    dt = 8.0 * 60.0      # seconds

    # Plasma frequency: f_p = 9 * sqrt(n_e) kHz with n_e in cm^-3
    # Actually: f_p = 9000 * sqrt(n_e) Hz  [n_e in cm^-3]
    # => n_e = (f_p / 9000)^2 cm^-3

    # Assuming fundamental emission (f = f_p)
    n_start = (f_start / 9000.0)**2  # cm^-3
    n_end = (f_end / 9000.0)**2

    print(f"  Type II burst: {f_start/1e6:.0f} MHz -> {f_end/1e6:.0f} MHz in {dt/60:.0f} min")
    print(f"  Using f_p = 9000 sqrt(n_e) Hz (fundamental emission)")
    print(f"  ")
    print(f"  At {f_start/1e6:.0f} MHz: n_e = ({f_start/9000:.0f})^2 = {n_start:.2e} cm^-3")
    print(f"  At {f_end/1e6:.0f} MHz:  n_e = ({f_end/9000:.0f})^2 = {n_end:.2e} cm^-3")

    # Density ratio and distance ratio
    n_ratio = n_start / n_end
    print(f"\n  Density ratio: n_start / n_end = {n_ratio:.0f}")

    # If n_e ~ r^(-2):
    # n_start / n_end = (r_end / r_start)^2
    r_ratio = np.sqrt(n_ratio)
    print(f"  For n_e ~ r^(-2): r_end / r_start = sqrt({n_ratio:.0f}) = {r_ratio:.1f}")

    # Estimate absolute distances
    # At f = 200 MHz, typical coronal height ~ 1.2-1.5 R_sun
    # Using Newkirk-like model for reference: n_e ~ 5e8 at ~1.5 R_sun
    # Our n_start ~ 5e8, consistent with r_start ~ 1.5 R_sun
    r_start = 1.5  # R_sun (estimate)
    r_end = r_start * r_ratio  # R_sun

    dr = (r_end - r_start) * R_sun  # meters
    v_shock = dr / dt
    v_shock_kms = v_shock / 1e3

    print(f"\n  Estimated distances:")
    print(f"    r_start ~ {r_start:.1f} R_sun (where n_e ~ {n_start:.0e} cm^-3)")
    print(f"    r_end ~ {r_end:.1f} R_sun")
    print(f"    dr = {(r_end-r_start):.1f} R_sun = {dr/1e6:.0f} Mm")
    print(f"  Average shock speed: v = dr/dt = {v_shock_kms:.0f} km/s")

    print(f"\n  Why this shock efficiently accelerates particles:")
    print(f"    - Speed ({v_shock_kms:.0f} km/s) >> fast magnetosonic speed (~few 100 km/s)")
    print(f"      => Strong shock (high Mach number)")
    print(f"    - In the low corona, magnetic field and density are high")
    print(f"      => Large compression ratio and strong turbulence behind shock")
    print(f"    - Long duration (gradual event) allows DSA to operate over")
    print(f"      extended time as shock propagates through corona into solar wind")
    print(f"    - Seed particles from suprathermal pool are readily available")
    print(f"    - Quasi-parallel geometry (common for CME-driven shocks)")
    print(f"      provides efficient injection into the DSA process")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: DSA Spectral Index ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: 3He Enrichment ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Velocity Dispersion Analysis ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Astronaut Radiation Dose ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Type II Burst Shock Speed ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)

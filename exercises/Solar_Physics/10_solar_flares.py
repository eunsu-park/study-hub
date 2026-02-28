"""
Exercises for Lesson 10: Solar Flares
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
m_p = 1.673e-27        # proton mass [kg]
mu_0 = 4.0e-7 * np.pi  # vacuum permeability [T m/A]
AU = 1.496e11           # astronomical unit [m]
erg_to_J = 1.0e-7       # erg to Joules
megaton_erg = 4.2e22    # 1 megaton TNT in erg


def exercise_1():
    """
    Problem 1: GOES Classification

    Peak flux = 3.7e-5 W/m^2.
    (a) GOES class.
    (b) Ratio to C1 flare.
    (c) Total SXR luminosity.
    """
    F_peak = 3.7e-5  # W/m^2

    # GOES classification:
    # A: < 1e-7, B: 1e-7 to 1e-6, C: 1e-6 to 1e-5,
    # M: 1e-5 to 1e-4, X: > 1e-4
    print(f"  (a) Peak GOES 1-8 A flux: F = {F_peak:.1e} W/m^2")

    if F_peak >= 1e-4:
        goes_class = "X"
        goes_number = F_peak / 1e-4
    elif F_peak >= 1e-5:
        goes_class = "M"
        goes_number = F_peak / 1e-5
    elif F_peak >= 1e-6:
        goes_class = "C"
        goes_number = F_peak / 1e-6
    elif F_peak >= 1e-7:
        goes_class = "B"
        goes_number = F_peak / 1e-7
    else:
        goes_class = "A"
        goes_number = F_peak / 1e-8

    print(f"      GOES class: {goes_class}{goes_number:.1f}")

    # (b) Ratio to C1 flare (1e-6 W/m^2)
    F_C1 = 1.0e-6
    ratio = F_peak / F_C1
    print(f"\n  (b) C1 flux: {F_C1:.0e} W/m^2")
    print(f"      Ratio: {F_peak:.1e} / {F_C1:.0e} = {ratio:.0f}")
    print(f"      This flare is {ratio:.0f} times more intense than a C1 flare.")

    # (c) Total SXR luminosity (assuming isotropic)
    L_SXR = F_peak * 4.0 * np.pi * AU**2
    print(f"\n  (c) Assuming isotropic emission:")
    print(f"      L_SXR = F * 4 pi d^2")
    print(f"            = {F_peak:.1e} * 4 pi * ({AU:.3e})^2")
    print(f"            = {L_SXR:.2e} W")
    print(f"      Note: This is the SXR (1-8 A) luminosity only.")
    print(f"      The total radiated energy across all wavelengths is much larger.")


def exercise_2():
    """
    Problem 2: Reconnection Rate

    CSHKP model: v_in = M_A * v_A.
    B = 100 G, n = 1e9 cm^-3, M_A = 0.05.
    (a) Alfven speed.
    (b) Inflow speed.
    (c) Energy release rate per unit area.
    """
    B_G = 100.0          # Gauss
    B = B_G * 1.0e-4     # Tesla
    n = 1.0e9 * 1.0e6    # cm^-3 to m^-3
    M_A = 0.05           # reconnection rate (Alfven Mach number)

    rho = n * m_p

    # (a) Alfven speed
    v_A = B / np.sqrt(mu_0 * rho)

    print(f"  Parameters:")
    print(f"    B = {B_G:.0f} G = {B:.4f} T")
    print(f"    n = {n/1e6:.0e} cm^-3 = {n:.1e} m^-3")
    print(f"    rho = n m_p = {rho:.2e} kg/m^3")
    print(f"    M_A = {M_A}")

    print(f"\n  (a) Alfven speed: v_A = B / sqrt(mu_0 rho)")
    print(f"                       = {B} / sqrt({mu_0:.2e} * {rho:.2e})")
    print(f"                       = {v_A:.2e} m/s = {v_A/1e3:.0f} km/s")

    # (b) Inflow speed
    v_in = M_A * v_A
    print(f"\n  (b) Reconnection inflow speed: v_in = M_A * v_A")
    print(f"                                      = {M_A} * {v_A/1e3:.0f}")
    print(f"                                      = {v_in/1e3:.0f} km/s")

    # (c) Energy release rate per unit area of current sheet
    # In CGS: dE/dA/dt = B^2 v_in / (4 pi)
    # In SI: dE/dA/dt = B^2 v_in / mu_0
    E_dot_A = B**2 * v_in / mu_0  # W/m^2

    # Also in CGS for comparison
    B_cgs = B_G  # Gauss
    v_in_cgs = v_in * 100.0  # cm/s
    E_dot_A_cgs = B_cgs**2 * v_in_cgs / (4.0 * np.pi)  # erg/cm^2/s

    print(f"\n  (c) Energy release rate per unit area:")
    print(f"      dE/dA/dt = B^2 * v_in / mu_0  (SI)")
    print(f"               = {E_dot_A:.2e} W/m^2")
    print(f"      In CGS: B^2 v_in / (4 pi) = {E_dot_A_cgs:.2e} erg/cm^2/s")

    # Estimate total for a typical current sheet
    L_cs = 1.0e7  # m (10 Mm length)
    W_cs = 1.0e5  # m (100 km width) -- this is the sheet area dimension
    A_cs = L_cs * W_cs
    E_dot_total = E_dot_A * A_cs
    print(f"\n      For a current sheet {L_cs/1e6:.0f} Mm x {W_cs/1e3:.0f} km:")
    print(f"      Total rate: {E_dot_total:.2e} W = {E_dot_total*1e7:.2e} erg/s")


def exercise_3():
    """
    Problem 3: Thick-Target Problem

    HXR photon spectral index gamma = 4 above 20 keV.
    (a) Injected electron index delta.
    (b) Total power in non-thermal electrons.
    """
    gamma_photon = 4.0     # HXR photon spectral index
    E_c = 20.0             # cutoff energy [keV]
    N_dot = 5.0e35         # electron flux above E_c [electrons/s]

    # (a) Thick-target relation: gamma = delta - 1
    # => delta = gamma + 1
    delta = gamma_photon + 1.0
    print(f"  (a) HXR photon spectral index: gamma = {gamma_photon:.0f}")
    print(f"      Thick-target relation: gamma = delta - 1")
    print(f"      => Injected electron index: delta = {delta:.0f}")
    print(f"      (electron distribution: F(E) ~ E^(-{delta:.0f}) above E_c)")

    # (b) Total power in non-thermal electrons
    # For a power-law distribution F(E) = F_0 * (E/E_c)^(-delta) above E_c:
    # N_dot = integral_Ec^inf F(E) dE = F_0 * E_c / (delta - 1)
    # Mean energy: <E> = E_c * delta / (delta - 2) for delta > 2
    # Total power: P = N_dot * <E>
    E_mean = E_c * delta / (delta - 2.0)  # keV
    P_keV = N_dot * E_mean  # keV/s
    P_erg = P_keV * 1.602e-9  # keV to erg: 1 keV = 1.602e-9 erg
    P_W = P_erg * 1.0e-7  # erg/s to W

    print(f"\n  (b) Low-energy cutoff: E_c = {E_c:.0f} keV")
    print(f"      Electron flux above E_c: N_dot = {N_dot:.1e} electrons/s")
    print(f"      Mean energy: <E> = E_c * delta / (delta - 2)")
    print(f"                       = {E_c} * {delta:.0f} / ({delta:.0f} - 2)")
    print(f"                       = {E_mean:.1f} keV")
    print(f"      Total power: P = N_dot * <E>")
    print(f"                     = {N_dot:.1e} * {E_mean:.1f} keV")
    print(f"                     = {P_erg:.2e} erg/s")
    print(f"                     = {P_W:.2e} W")

    # Put in context
    print(f"\n      For a 30-second impulsive phase:")
    E_total = P_erg * 30.0
    print(f"      Total non-thermal energy: {E_total:.2e} erg")
    print(f"      This is typical of an M-class flare.")


def exercise_4():
    """
    Problem 4: Flare Energy Budget

    X5.0 flare, total energy = 5e32 erg.
    Emslie et al. partition:
    (a) CME, (b) thermal, (c) non-thermal electrons, (d) total radiation.
    Express in erg and megatons TNT.
    """
    E_total = 5.0e32  # erg

    # Emslie et al. (2012) approximate energy partition for large flares:
    # CME kinetic energy: ~30-40% of total
    # Thermal plasma: ~20-30%
    # Non-thermal electrons: ~20-40%
    # Total radiation: ~10-20%
    # (These overlap since non-thermal -> thermal -> radiation)

    # Using representative fractions:
    partitions = {
        "CME (kinetic + potential)": 0.35,
        "Thermal plasma": 0.25,
        "Non-thermal electrons": 0.30,
        "Total radiation": 0.15,
    }

    print(f"  X5.0 flare total energy: E = {E_total:.1e} erg")
    print(f"  Emslie et al. approximate energy partition:\n")

    for component, frac in partitions.items():
        E_comp = frac * E_total
        E_megatons = E_comp / megaton_erg
        print(f"  ({list(partitions.keys()).index(component)+1}) {component}:")
        print(f"      = {frac*100:.0f}% of total = {E_comp:.2e} erg")
        print(f"      = {E_megatons:.1e} megatons TNT")

    # Context
    E_total_MT = E_total / megaton_erg
    print(f"\n  Total energy: {E_total_MT:.1e} megatons TNT")
    print(f"  (Tsar Bomba was ~50 MT; this flare is {E_total_MT/50:.0e}x larger)")
    print(f"\n  Note: The partition percentages are approximate and overlap,")
    print(f"  since non-thermal electron energy is partly converted to thermal")
    print(f"  and radiated energy. The CME kinetic energy is often the largest")
    print(f"  single component for eruptive flares.")


def exercise_5():
    """
    Problem 5: QPP Diagnostics

    Flare loop: L = 5e9 cm, QPP period P = 30 s, loop radius a = 1e8 cm.
    Sausage mode: P ~ 2a / v_A.
    (a) Alfven speed.
    (b) Magnetic field for n_e = 1e11 cm^-3.
    """
    L = 5.0e9           # loop length [cm]
    P = 30.0            # QPP period [s]
    a = 1.0e8           # loop radius [cm]
    n_e = 1.0e11        # electron density [cm^-3]

    # Convert to SI
    L_m = L * 0.01      # m
    a_m = a * 0.01      # m
    n_e_m3 = n_e * 1e6  # m^-3

    # (a) Sausage mode: P ~ 2a / v_A
    v_A = 2.0 * a_m / P
    v_A_kms = v_A / 1e3

    print(f"  Loop parameters:")
    print(f"    Length: L = {L:.0e} cm = {L_m/1e6:.0f} Mm")
    print(f"    Radius: a = {a:.0e} cm = {a_m/1e3:.0f} km")
    print(f"    QPP period: P = {P:.0f} s")
    print(f"    n_e = {n_e:.0e} cm^-3")

    print(f"\n  (a) Sausage mode: P = 2a / v_A")
    print(f"      v_A = 2a / P = 2 * {a_m/1e3:.0f} km / {P:.0f} s")
    print(f"          = {v_A_kms:.0f} km/s")

    # (b) Magnetic field: v_A = B / sqrt(mu_0 * n_i * m_p)
    # Assume n_i ~ n_e for hydrogen plasma
    rho = n_e_m3 * m_p
    B = v_A * np.sqrt(mu_0 * rho)
    B_G = B * 1e4  # Gauss

    print(f"\n  (b) v_A = B / sqrt(mu_0 * rho)")
    print(f"      rho = n_e * m_p = {rho:.2e} kg/m^3")
    print(f"      B = v_A * sqrt(mu_0 * rho)")
    print(f"        = {v_A:.0f} * sqrt({mu_0:.2e} * {rho:.2e})")
    print(f"        = {B:.3e} T = {B_G:.0f} G")
    print(f"\n      This magnetic field strength ({B_G:.0f} G) is reasonable for a")
    print(f"      flaring coronal loop, consistent with typical active region")
    print(f"      coronal fields of 100-500 G.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: GOES Classification ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Reconnection Rate ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Thick-Target Problem ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Flare Energy Budget ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: QPP Diagnostics ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)

"""
Exercises for Lesson 02: Nuclear Energy Generation
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
k_B = 1.381e-23        # Boltzmann constant [J/K]
k_B_eV = 8.617e-5      # Boltzmann constant [eV/K]
m_p = 1.673e-27        # proton mass [kg]
m_p_MeV = 938.272      # proton mass energy [MeV]
c = 3.0e8              # speed of light [m/s]
e = 1.602e-19          # elementary charge [C]
hbar = 1.055e-34       # reduced Planck constant [J s]
R_sun = 6.957e8        # solar radius [m]
L_sun = 3.828e26       # solar luminosity [W]
MeV_to_J = 1.602e-13   # MeV to Joules
AU = 1.496e11          # astronomical unit [m]


def exercise_1():
    """
    Problem 1: Gamow Peak for C(p,gamma)N

    Calculate the Gamow peak energy E_0 and width Delta_E_0 for
    12C(p,gamma)13N at T = 1.5e7 K.
    Z_1 = 1, Z_2 = 6, m_r = m_p * 12/13.
    Compare with pp Gamow peak.
    """
    T = 1.5e7  # K
    Z_1 = 1
    Z_2 = 6
    m_r_ratio = 12.0 / 13.0  # reduced mass in units of m_p

    # Gamow energy: E_G = (2 * m_r * c^2) * (pi * alpha * Z_1 * Z_2)^2
    # where alpha = e^2 / (4 pi epsilon_0 hbar c) ~ 1/137
    alpha_fine = 1.0 / 137.036  # fine structure constant

    # Reduced mass in kg
    m_r = m_r_ratio * m_p

    # Gamow energy in Joules
    E_G = 2.0 * m_r * c**2 * (np.pi * alpha_fine * Z_1 * Z_2)**2
    E_G_keV = E_G / (1.0e3 * e)

    print(f"  Reduced mass: m_r = {m_r_ratio:.4f} m_p = {m_r:.4e} kg")
    print(f"  Gamow energy: E_G = {E_G_keV:.1f} keV")

    # Gamow peak: E_0 = (E_G * (k_B T)^2 / 4)^(1/3)
    kBT = k_B * T
    kBT_keV = k_B_eV * T * 1e3  # convert eV to keV
    E_0 = (E_G * kBT**2 / 4.0)**(1.0 / 3.0)
    E_0_keV = E_0 / (1.0e3 * e)

    print(f"  k_B T = {kBT_keV:.3f} keV")
    print(f"  Gamow peak energy: E_0 = {E_0_keV:.1f} keV")

    # Width: Delta_E_0 = 4 * sqrt(E_0 * k_B T / 3)
    Delta_E_0 = 4.0 * np.sqrt(E_0 * kBT / 3.0)
    Delta_E_0_keV = Delta_E_0 / (1.0e3 * e)
    print(f"  Gamow peak width: Delta_E_0 = {Delta_E_0_keV:.1f} keV")

    # Compare with pp Gamow peak (Z_1 = Z_2 = 1, m_r = m_p/2)
    print(f"\n  --- Comparison with pp reaction ---")
    Z_pp = 1
    m_r_pp = m_p / 2.0
    E_G_pp = 2.0 * m_r_pp * c**2 * (np.pi * alpha_fine * Z_pp * Z_pp)**2
    E_G_pp_keV = E_G_pp / (1.0e3 * e)
    kBT_pp = kBT
    E_0_pp = (E_G_pp * kBT_pp**2 / 4.0)**(1.0 / 3.0)
    E_0_pp_keV = E_0_pp / (1.0e3 * e)

    print(f"  pp Gamow energy: E_G_pp = {E_G_pp_keV:.1f} keV")
    print(f"  pp Gamow peak: E_0_pp = {E_0_pp_keV:.1f} keV")
    print(f"  CNO Gamow peak is ~{E_0_keV/E_0_pp_keV:.1f}x higher than pp.")
    print(f"  The CNO peak is higher because Z_2=6 increases the Coulomb barrier,")
    print(f"  requiring higher kinetic energy to achieve appreciable tunneling.")


def exercise_2():
    """
    Problem 2: pp Chain Energy Bookkeeping

    (a) Net energy release for pp-I: 26.73 MeV
    (b) Energy carried by neutrinos vs heating the Sun.
    (c) Fraction of proton rest mass converted to energy.
    """
    # pp-I chain reactions (need 2 of first two reactions for each He-3+He-3):
    # 1) p + p -> d + e+ + nu_e          Q = 1.442 MeV (incl. e+ annihilation)
    #    neutrino energy: average 0.267 MeV (pp neutrino, max 0.423 MeV)
    # 2) d + p -> He-3 + gamma            Q = 5.493 MeV
    # 3) He-3 + He-3 -> He-4 + 2p         Q = 12.860 MeV

    Q_pp = 1.442       # MeV per pp reaction (including e+ annihilation)
    Q_dp = 5.493       # MeV per d+p reaction
    Q_33 = 12.860      # MeV per He-3 + He-3 reaction
    nu_pp_avg = 0.267   # average pp neutrino energy [MeV]

    # (a) Total energy: 2 * Q_pp + 2 * Q_dp + Q_33
    Q_total = 2 * Q_pp + 2 * Q_dp + Q_33
    print(f"  (a) pp-I chain energy release:")
    print(f"      2 x (p+p):       2 x {Q_pp} = {2*Q_pp:.3f} MeV")
    print(f"      2 x (d+p):       2 x {Q_dp} = {2*Q_dp:.3f} MeV")
    print(f"      1 x (He3+He3):   1 x {Q_33} = {Q_33:.3f} MeV")
    print(f"      Total:           {Q_total:.3f} MeV")

    # Cross-check: 4p -> He-4 + 2e+ + 2nu_e
    # Mass deficit: 4*m_p - m_He4 = 4*938.272 - 3727.379 = 25.709 MeV
    # Plus 2 * e+ annihilation (2 * 2 * 0.511 = 2.044 MeV)
    mass_deficit = 4 * 938.272 - 3727.379
    e_annihilation = 2 * 2 * 0.511
    print(f"      Cross-check: mass deficit = {mass_deficit:.3f} MeV")
    print(f"      Plus 2 e+ annihilation = {e_annihilation:.3f} MeV")
    print(f"      Total = {mass_deficit + e_annihilation:.3f} MeV")

    # (b) Energy carried by neutrinos
    E_neutrino = 2 * nu_pp_avg  # two pp neutrinos per complete chain
    E_heating = Q_total - E_neutrino
    print(f"\n  (b) Neutrino energy: 2 x {nu_pp_avg} = {E_neutrino:.3f} MeV")
    print(f"      Heating energy: {Q_total:.3f} - {E_neutrino:.3f} = {E_heating:.3f} MeV")
    print(f"      Fraction to neutrinos: {E_neutrino/Q_total*100:.1f}%")
    print(f"      Fraction to heating:   {E_heating/Q_total*100:.1f}%")

    # (c) Fraction of rest mass energy converted
    E_rest = 4 * m_p_MeV  # MeV
    fraction = Q_total / E_rest
    print(f"\n  (c) Rest mass energy of 4 protons: {E_rest:.3f} MeV")
    print(f"      Energy released: {Q_total:.3f} MeV")
    print(f"      Fraction converted: {fraction:.5f} = {fraction*100:.3f}%")
    print(f"      About 0.7% of the mass is converted to energy.")


def exercise_3():
    """
    Problem 3: Solar Luminosity from First Principles

    pp reaction rate formula and estimate luminosity from core volume.
    """
    # Given parameters
    T = 1.5e7       # K
    rho = 100.0     # g/cm^3
    X = 0.5         # hydrogen mass fraction in core
    T6 = T / 1.0e6  # T in units of 10^6 K

    # pp reaction rate: r_pp = 3.09e-37 * rho^2 * X^2 * T6^(-2/3) * exp(-33.80/T6^(1/3))
    # reactions / cm^3 / s
    r_pp = 3.09e-37 * rho**2 * X**2 * T6**(-2.0/3.0) * np.exp(-33.80 / T6**(1.0/3.0))
    print(f"  T = {T:.2e} K, T6 = {T6:.1f}")
    print(f"  rho = {rho} g/cm^3, X = {X}")
    print(f"  pp reaction rate: r_pp = {r_pp:.3e} reactions/cm^3/s")

    # Each pp-I chain requires 2 pp reactions and releases ~26 MeV
    # So energy per pp reaction = 26/2 = 13 MeV (approximate)
    E_per_pp = 13.0 * MeV_to_J  # J per pp reaction

    # Volume of sphere of radius 0.1 R_sun
    r_core = 0.1 * R_sun * 100.0  # convert to cm
    V_core = (4.0 / 3.0) * np.pi * r_core**3  # cm^3
    print(f"  Core radius: 0.1 R_sun = {r_core:.3e} cm")
    print(f"  Core volume: {V_core:.3e} cm^3")

    # Total reaction rate in the volume
    total_rate = r_pp * V_core  # reactions/s
    print(f"  Total pp reactions/s: {total_rate:.3e}")

    # Luminosity
    L_estimate = total_rate * E_per_pp  # W
    print(f"  Energy per pp reaction: ~{E_per_pp/MeV_to_J:.0f} MeV")
    print(f"  Estimated luminosity: L = {L_estimate:.3e} W")
    print(f"  Solar luminosity:     L_sun = {L_sun:.3e} W")
    print(f"  Ratio L/L_sun = {L_estimate/L_sun:.2f}")
    print(f"\n  Note: The estimate is rough because of uniform core approximation.")
    print(f"  In reality, T and rho vary significantly within the core.")


def exercise_4():
    """
    Problem 4: CNO vs pp Crossover Temperature

    epsilon_CNO / epsilon_pp ~ T^13 * (Z_CNO / X)
    At T_c = 1.57e7 K, this ratio is ~0.01.
    Find T where the ratio = 1.
    """
    T_solar = 1.57e7    # solar core temperature [K]
    ratio_solar = 0.01   # epsilon_CNO / epsilon_pp at solar T_c

    # epsilon_CNO/epsilon_pp = ratio_solar * (T/T_solar)^13
    # (The Z_CNO/X factor is absorbed into the ratio at solar conditions)
    # Set ratio = 1: 1 = 0.01 * (T_cross / T_solar)^13
    # T_cross = T_solar * (1/0.01)^(1/13) = T_solar * 100^(1/13)

    T_cross = T_solar * (1.0 / ratio_solar)**(1.0 / 13.0)
    T_cross_MK = T_cross / 1.0e6

    print(f"  At solar center (T = {T_solar/1e6:.2f} MK):")
    print(f"    epsilon_CNO / epsilon_pp = {ratio_solar}")
    print(f"  Crossover condition: ratio = 1")
    print(f"  T_cross = T_solar * (1/{ratio_solar})^(1/13)")
    print(f"          = {T_solar/1e6:.2f} MK * {(1.0/ratio_solar)**(1.0/13.0):.3f}")
    print(f"          = {T_cross_MK:.1f} MK")
    print(f"\n  Stars with core T > ~{T_cross_MK:.0f} MK are dominated by the CNO cycle.")
    print(f"  This corresponds to stars slightly more massive than the Sun (~1.3 M_sun).")
    print(f"  CNO-dominant stars have convective cores (steep T dependence -> steep L gradient).")


def exercise_5():
    """
    Problem 5: Neutrino Oscillation Survival Probability

    P(nu_e -> nu_e) = 1 - sin^2(2 theta_12) * sin^2(Dm^2 L / (4 E_nu))
    theta_12 ~ 33.4 deg, Dm^2 = 7.53e-5 eV^2
    (a) Calculate the oscillation argument for L=1 AU, E_nu=0.3 MeV.
    (b) Average survival probability.
    (c) Compare with Borexino.
    (d) MSW effect for 8B vs pp neutrinos.
    """
    theta_12 = np.radians(33.4)    # mixing angle
    Dm2 = 7.53e-5                  # eV^2
    L_m = AU                       # 1 AU in meters
    E_nu_MeV = 0.3                 # pp neutrino energy in MeV

    # (a) Calculate Dm^2 * L / (4 * E_nu)
    # Need consistent units. Using natural units where hbar*c = 197.3 MeV*fm
    # Dm^2 * L / (4 * E_nu) where Dm^2 in eV^2, L in m, E_nu in MeV
    # = Dm^2 [eV^2] * L [m] / (4 * E_nu [eV]) * (1 / (hbar c))
    # hbar*c = 197.3 MeV*fm = 197.3e-15 m * 1e6 eV = 197.3e-9 eV*m
    hbar_c = 197.3e-15  # MeV * m
    hbar_c_eV_m = 197.3e-9  # eV * m  (= 197.3 MeV*fm)

    # Phase = Dm^2 * L / (4 * E_nu * hbar_c)  [all in eV and m]
    E_nu_eV = E_nu_MeV * 1.0e6  # eV
    phase = Dm2 * L_m / (4.0 * E_nu_eV * hbar_c_eV_m)

    print(f"  (a) theta_12 = {np.degrees(theta_12):.1f} deg")
    print(f"      Dm^2_21 = {Dm2:.2e} eV^2")
    print(f"      L = 1 AU = {L_m:.3e} m")
    print(f"      E_nu = {E_nu_MeV} MeV = {E_nu_eV:.0e} eV")
    print(f"      Phase = Dm^2 * L / (4 * E_nu) = {phase:.3e} (in natural units)")
    print(f"      This is an enormous number >> 1.")

    # Oscillation length: L_osc = 4 pi E_nu / Dm^2 (in natural units)
    L_osc = 4.0 * np.pi * E_nu_eV * hbar_c_eV_m / Dm2
    print(f"      Oscillation length: L_osc = {L_osc:.1f} m = {L_osc/1e3:.1f} km")
    print(f"      L / L_osc = {L_m / L_osc:.2e} >> 1")

    # (b) Since L >> L_osc, sin^2 averages to 1/2
    sin2_2theta = np.sin(2.0 * theta_12)**2
    P_survival = 1.0 - 0.5 * sin2_2theta

    print(f"\n  (b) sin^2(2 theta_12) = {sin2_2theta:.4f}")
    print(f"      Averaged survival probability:")
    print(f"      P(nu_e -> nu_e) = 1 - (1/2) sin^2(2 theta_12) = {P_survival:.3f}")

    # (c) Borexino measurement
    P_borexino = 0.51  # approximate Borexino measurement for pp neutrinos
    print(f"\n  (c) Borexino measurement: P ~ {P_borexino}")
    print(f"      Our vacuum oscillation result: P = {P_survival:.3f}")
    print(f"      Reasonably consistent (vacuum approximation works for pp neutrinos).")

    # (d) MSW effect
    print(f"\n  (d) The MSW (matter) effect is important when the neutrino energy")
    print(f"      is high enough that the matter potential V ~ sqrt(2) G_F n_e")
    print(f"      is comparable to Dm^2/(2E). For pp neutrinos (E ~ 0.3 MeV),")
    print(f"      Dm^2/(2E) >> V, so matter effects are negligible (vacuum regime).")
    print(f"      For 8B neutrinos (E ~ 6-15 MeV), Dm^2/(2E) becomes comparable")
    print(f"      to V in the solar core, so the MSW resonance significantly")
    print(f"      modifies the survival probability (P ~ 0.3 vs vacuum ~0.55).")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Gamow Peak ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: pp Chain Energy Bookkeeping ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Solar Luminosity from First Principles ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: CNO vs pp Crossover Temperature ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Neutrino Oscillation Survival Probability ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)

"""
Exercises for Lesson 05: Chromosphere and Transition Region
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
k_B = 1.381e-23        # Boltzmann constant [J/K]
m_H = 1.673e-27        # hydrogen mass [kg]
m_p = 1.673e-27        # proton mass [kg]
c = 3.0e8              # speed of light [m/s]
mu_0 = 4.0e-7 * np.pi  # vacuum permeability [T m/A]
R_sun = 6.957e8        # solar radius [m]
g_sun = 274.0          # solar surface gravity [m/s^2]


def exercise_1():
    """
    Problem 1: Temperature Gradient in the Transition Region

    TR spans T = 2.5e4 K to 1e6 K over ~100 km.
    (a) Average dT/dh in K/km.
    (b) Conductive heat flux at T = 5e5 K using kappa = kappa_0 * T^(5/2),
        kappa_0 = 1e-6 erg/(cm s K^(7/2)).
    (c) Compare with chromospheric radiative loss rate of 4 kW/m^2.
    """
    T_low = 2.5e4      # K
    T_high = 1.0e6     # K
    delta_h = 100.0     # km

    # (a) Average temperature gradient
    dT_dh = (T_high - T_low) / delta_h  # K/km
    print(f"  (a) TR temperature range: {T_low:.1e} K to {T_high:.1e} K")
    print(f"      Height span: {delta_h:.0f} km")
    print(f"      Average dT/dh = {dT_dh:.1e} K/km = {dT_dh:.0f} K/km")

    # (b) Conductive heat flux at T = 5e5 K
    # Spitzer conductivity: kappa = kappa_0 * T^(5/2) [erg cm^-1 s^-1 K^-1]
    # Heat flux: F = -kappa * dT/dh
    kappa_0_cgs = 1.0e-6    # erg cm^-1 s^-1 K^(-7/2)
    T_eval = 5.0e5           # K

    kappa_T = kappa_0_cgs * T_eval**2.5    # erg cm^-1 s^-1 K^-1

    # Convert dT/dh to cgs: K/km -> K/cm
    dT_dh_cgs = dT_dh / 1.0e5   # K/cm (1 km = 1e5 cm)

    F_cond_cgs = kappa_T * dT_dh_cgs  # erg cm^-2 s^-1
    # Convert to SI: 1 erg/cm^2/s = 1e-3 W/m^2
    F_cond_SI = F_cond_cgs * 1.0e-3   # W/m^2

    print(f"\n  (b) At T = {T_eval:.1e} K:")
    print(f"      kappa(T) = kappa_0 * T^(5/2) = {kappa_T:.2e} erg/(cm s K)")
    print(f"      dT/dh = {dT_dh_cgs:.2e} K/cm")
    print(f"      Conductive flux F = kappa * |dT/dh|")
    print(f"                       = {F_cond_cgs:.2e} erg/(cm^2 s)")
    print(f"                       = {F_cond_SI:.1f} W/m^2")
    # Also convert to kW/m^2
    print(f"                       = {F_cond_SI/1e3:.2f} kW/m^2")

    # (c) Compare with chromospheric radiative losses
    F_chrom = 4.0e3   # W/m^2 = 4 kW/m^2
    print(f"\n  (c) Chromospheric radiative loss rate: {F_chrom/1e3:.0f} kW/m^2")
    print(f"      Conductive flux from corona: {F_cond_SI/1e3:.2f} kW/m^2")
    ratio = F_cond_SI / F_chrom
    print(f"      Ratio F_cond / F_chrom = {ratio:.2f}")
    print(f"      The downward conductive flux from the corona is comparable to")
    print(f"      the chromospheric radiative losses, consistent with the picture")
    print(f"      that coronal thermal conduction is a major energy source for")
    print(f"      the upper chromosphere and transition region.")


def exercise_2():
    """
    Problem 2: Acoustic Cutoff and Spicule Driving

    (a) Acoustic cutoff period for T = 6000 K, mu = 1.3 m_H, gamma = 5/3.
    (b) Effective cutoff in inclined flux tube (theta = 50 deg).
    (c) Relevance for Type I spicule formation.
    """
    T = 6000.0          # K
    mu_mol = 1.3        # mean molecular weight (in units of m_H)
    gamma = 5.0 / 3.0   # adiabatic index

    # (a) Acoustic cutoff frequency: omega_ac = gamma * g / (2 c_s)
    #     where c_s = sqrt(gamma * k_B * T / (mu * m_H))
    #     Cutoff period: P_ac = 2 pi / omega_ac = 4 pi c_s / (gamma g)
    c_s = np.sqrt(gamma * k_B * T / (mu_mol * m_H))
    omega_ac = gamma * g_sun / (2.0 * c_s)
    P_ac = 2.0 * np.pi / omega_ac

    print(f"  (a) Sound speed: c_s = sqrt(gamma k_B T / (mu m_H))")
    print(f"                       = {c_s:.0f} m/s = {c_s/1e3:.2f} km/s")
    print(f"      Acoustic cutoff frequency: omega_ac = gamma g / (2 c_s)")
    print(f"                                         = {omega_ac:.4f} rad/s")
    print(f"      Acoustic cutoff period: P_ac = 2 pi / omega_ac = {P_ac:.0f} s")
    print(f"                                   = {P_ac/60:.1f} min")

    # (b) Inclined flux tube with theta = 50 degrees
    theta = np.radians(50.0)
    # Effective gravity: g_eff = g * cos(theta)
    # Effective cutoff period: P_eff = P_ac / cos(theta)
    P_eff = P_ac / np.cos(theta)
    print(f"\n  (b) Flux tube inclination: theta = 50 deg")
    print(f"      Effective gravity: g_eff = g cos(theta) = {g_sun * np.cos(theta):.1f} m/s^2")
    print(f"      Effective cutoff period: P_eff = P_ac / cos(theta)")
    print(f"                             = {P_ac:.0f} / {np.cos(theta):.3f}")
    print(f"                             = {P_eff:.0f} s = {P_eff/60:.1f} min")

    # (c) Spicule relevance
    print(f"\n  (c) Relevance for Type I spicules:")
    print(f"      The 5-minute photospheric oscillation period (~300 s) is")
    print(f"      BELOW the vertical cutoff period (~{P_ac:.0f} s), so these waves")
    print(f"      are evanescent in vertical magnetic fields.")
    print(f"      ")
    print(f"      However, in inclined flux tubes (like at network boundaries),")
    print(f"      the effective cutoff period increases to ~{P_eff:.0f} s (~{P_eff/60:.1f} min),")
    print(f"      which is ABOVE the 5-minute period. This means the p-mode")
    print(f"      oscillations can now propagate upward along the inclined field.")
    print(f"      ")
    print(f"      These propagating acoustic waves steepen into shocks in the")
    print(f"      chromosphere and drive Type I spicules at network boundaries.")
    print(f"      This explains why spicules are preferentially found at network")
    print(f"      boundaries where flux tubes are inclined.")


def exercise_3():
    """
    Problem 3: Spicule Mass Flux

    N = 1e6 spicules, diameter 500 km, v_up = 25 km/s,
    n_e = 2e10 cm^-3.
    (a) Total mass flux.
    (b) Compare with solar wind mass loss rate 2e9 kg/s.
    (c) Fate of spicule material.
    """
    N_spicules = 1.0e6
    diameter = 500.0e3    # m
    v_up = 25.0e3         # m/s
    n_e = 2.0e10 * 1.0e6  # convert cm^-3 to m^-3

    # (a) Mass flux per spicule and total
    # Cross-section area of one spicule
    A_spicule = np.pi * (diameter / 2.0)**2
    print(f"  (a) Spicule parameters:")
    print(f"      Number: N = {N_spicules:.0e}")
    print(f"      Diameter: {diameter/1e3:.0f} km")
    print(f"      Upflow speed: {v_up/1e3:.0f} km/s")
    print(f"      Electron density: n_e = {n_e:.2e} m^-3")

    # Mass density: rho = n_e * m_p (for pure hydrogen plasma, n_e = n_p)
    rho = n_e * m_p
    print(f"      Mass density: rho = n_e * m_p = {rho:.2e} kg/m^3")

    # Mass flux per spicule: dm/dt = rho * v * A
    dm_dt_one = rho * v_up * A_spicule
    dm_dt_total = N_spicules * dm_dt_one

    print(f"      Cross-section per spicule: A = {A_spicule:.2e} m^2")
    print(f"      Mass flux per spicule: {dm_dt_one:.2e} kg/s")
    print(f"      Total spicule mass flux: {dm_dt_total:.2e} kg/s")

    # (b) Compare with solar wind
    dm_dt_sw = 2.0e9  # kg/s
    ratio = dm_dt_total / dm_dt_sw
    print(f"\n  (b) Solar wind mass loss rate: {dm_dt_sw:.1e} kg/s")
    print(f"      Spicule mass flux / Solar wind = {ratio:.0f}")
    print(f"      Spicules carry ~{ratio:.0f} times more mass than the solar wind!")

    # (c) Fate of material
    print(f"\n  (c) Since the spicule mass flux is ~{ratio:.0f}x the solar wind rate,")
    print(f"      the vast majority of spicule material must FALL BACK to the")
    print(f"      chromosphere. Only a tiny fraction (~{1/ratio*100:.1f}%) could potentially")
    print(f"      contribute to the solar wind mass flux.")
    print(f"      Spicules are primarily a circulation pattern, not a mass loss")
    print(f"      mechanism. However, some heated spicule material may contribute")
    print(f"      to the coronal plasma supply, particularly Type II spicules")
    print(f"      that appear to heat to coronal temperatures.")


def exercise_4():
    """
    Problem 4: Differential Emission Measure

    Constant-pressure TR slab, T varies linearly from T1=1e5 K to T2=1e6 K
    over Delta_h = 100 km. p = n_e * k_B * T (pure hydrogen).
    (a) Express n_e(T) in terms of p and T.
    (b) Derive DEM(T) = n_e^2 * dh/dT.
    (c) At what T is DEM largest?
    """
    T1 = 1.0e5       # K
    T2 = 1.0e6       # K
    delta_h = 100.0e3  # m (100 km)

    # Assume a pressure (for illustration)
    # Typical TR pressure: p ~ 0.01-0.1 dyn/cm^2 ~ 1e-3 to 1e-2 Pa
    p = 0.05 * 0.1  # dyn/cm^2 -> Pa: 1 dyn/cm^2 = 0.1 Pa => p = 0.005 Pa
    # Actually let's use a more typical value
    p = 0.02  # Pa (typical TR pressure)

    # (a) For a pure hydrogen plasma at constant pressure:
    # p = 2 n_e k_B T  (electrons + protons contribute equally)
    # n_e = p / (2 k_B T)
    print(f"  (a) At constant pressure p = n_e k_B T (the problem states n_e = n_p,")
    print(f"      so the gas pressure for H plasma is p = 2 n_e k_B T:")
    print(f"      => n_e(T) = p / (2 k_B T)")
    print(f"")
    print(f"      However, using the problem's notation p = n_e k_B T,")
    print(f"      which defines p as the electron partial pressure:")
    print(f"      => n_e(T) = p / (k_B T)")

    # Using p = n_e * k_B * T as stated in the problem
    n_e_T1 = p / (k_B * T1)
    n_e_T2 = p / (k_B * T2)
    print(f"\n      For p = {p:.3f} Pa:")
    print(f"      n_e(T1={T1:.0e} K) = {n_e_T1:.2e} m^-3 = {n_e_T1*1e-6:.2e} cm^-3")
    print(f"      n_e(T2={T2:.0e} K) = {n_e_T2:.2e} m^-3 = {n_e_T2*1e-6:.2e} cm^-3")

    # (b) DEM(T) = n_e^2 * dh/dT
    # T varies linearly with h: T(h) = T1 + (T2 - T1) * h / delta_h
    # dT/dh = (T2 - T1) / delta_h = const
    # dh/dT = delta_h / (T2 - T1) = const
    dh_dT = delta_h / (T2 - T1)  # m/K

    print(f"\n  (b) Linear temperature profile: dT/dh = (T2-T1)/Delta_h = const")
    print(f"      dh/dT = Delta_h / (T2-T1) = {dh_dT:.3e} m/K")
    print(f"      n_e(T) = p / (k_B T)")
    print(f"      DEM(T) = n_e^2 * dh/dT = [p/(k_B T)]^2 * dh/dT")
    print(f"             = p^2 / (k_B^2 * T^2) * dh/dT")
    print(f"             proportional to T^(-2)")

    # Evaluate at a few temperatures
    T_sample = np.array([1e5, 2e5, 5e5, 1e6])
    print(f"\n      DEM values (in m^-5 K^-1 = cm^-5 K^-1 * 1e-2):")
    for T in T_sample:
        n_e = p / (k_B * T)
        dem = n_e**2 * dh_dT
        dem_cgs = dem * 1.0e-2  # m^-5 K^-1 to cm^-5 K^-1 (1 m^-5 = 1e-10 cm^-5, then *dh in cm)
        # Actually DEM has units of n_e^2 * dh/dT : [m^-3]^2 * [m/K] = m^-5 K^-1
        print(f"      T = {T:.0e} K: DEM = {dem:.2e} m^-5 K^-1")

    # (c) DEM is largest at lowest T
    print(f"\n  (c) Since DEM(T) ~ T^(-2), the DEM is LARGEST at the LOWEST")
    print(f"      temperature (T = {T1:.0e} K).")
    print(f"      This is because at constant pressure, lower temperature means")
    print(f"      higher density (n_e ~ 1/T), and DEM ~ n_e^2 ~ 1/T^2.")
    print(f"      The cool, dense base of the TR dominates the emission measure.")


def exercise_5():
    """
    Problem 5: Moreton Wave Speed

    Moreton wave: 400 Mm in 5.5 minutes.
    (a) Propagation speed.
    (b) Required Alfven speed for fast-mode speed = Moreton speed,
        given c_s = 180 km/s.
    (c) Magnetic field strength for n_e = 5e8 cm^-3.
    """
    distance = 400.0e6   # m (400 Mm)
    time = 5.5 * 60.0    # s (5.5 min)

    # (a) Propagation speed
    v_moreton = distance / time
    v_moreton_kms = v_moreton / 1e3
    print(f"  (a) Distance: {distance/1e6:.0f} Mm")
    print(f"      Time: {time:.0f} s = {time/60:.1f} min")
    print(f"      Moreton wave speed: v = {v_moreton_kms:.0f} km/s")

    # (b) Fast-mode speed: v_f = sqrt(c_s^2 + v_A^2)
    # Set v_f = v_moreton => v_A = sqrt(v_moreton^2 - c_s^2)
    c_s = 180.0e3       # coronal sound speed [m/s]
    v_A = np.sqrt(v_moreton**2 - c_s**2)
    v_A_kms = v_A / 1e3

    print(f"\n  (b) Coronal sound speed: c_s = {c_s/1e3:.0f} km/s (at T = 1.5 MK)")
    print(f"      Fast-mode speed: v_f = sqrt(c_s^2 + v_A^2)")
    print(f"      Setting v_f = v_moreton:")
    print(f"      v_A = sqrt(v_moreton^2 - c_s^2)")
    print(f"          = sqrt({v_moreton_kms:.0f}^2 - {c_s/1e3:.0f}^2) km/s")
    print(f"          = {v_A_kms:.0f} km/s")

    # (c) Magnetic field from Alfven speed
    # v_A = B / sqrt(mu_0 * rho)
    # rho = n_e * m_p (assuming n_e ~ n_i for hydrogen plasma)
    n_e_cgs = 5.0e8     # cm^-3
    n_e = n_e_cgs * 1.0e6  # m^-3
    rho = n_e * m_p

    B = v_A * np.sqrt(mu_0 * rho)
    B_G = B * 1.0e4  # T to Gauss

    print(f"\n  (c) Electron density: n_e = {n_e_cgs:.0e} cm^-3 = {n_e:.1e} m^-3")
    print(f"      Mass density: rho = n_e m_p = {rho:.2e} kg/m^3")
    print(f"      B = v_A * sqrt(mu_0 * rho)")
    print(f"        = {v_A_kms:.0f} km/s * sqrt({mu_0:.2e} * {rho:.2e})")
    print(f"        = {B:.3e} T = {B_G:.1f} G")
    print(f"      This is a reasonable coronal magnetic field strength (~a few G)")
    print(f"      for the quiet corona where Moreton waves propagate.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Temperature Gradient in the TR ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Acoustic Cutoff and Spicule Driving ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Spicule Mass Flux ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Differential Emission Measure ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Moreton Wave Speed ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)

"""
Exercises for Lesson 06: Corona
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
k_B = 1.381e-23        # Boltzmann constant [J/K]
m_p = 1.673e-27        # proton mass [kg]
R_sun = 6.957e8        # solar radius [m]
L_sun = 3.828e26       # solar luminosity [W]
mu_0 = 4.0e-7 * np.pi  # vacuum permeability [T m/A]


def exercise_1():
    """
    Problem 1: Coronal Energy Budget

    Quiet corona requires ~300 W/m^2.
    (a) Fraction of total solar luminous flux.
    (b) Required photospheric Alfven wave velocity amplitude delta_v,
        given rho = 2e-4 kg/m^3, v_A = 1 km/s, 1% transmission.
    (c) Consistency with observed motions.
    """
    F_corona = 300.0    # W/m^2
    R = R_sun

    # (a) Total solar radiative flux at the surface
    A_sun = 4.0 * np.pi * R**2
    F_total = L_sun / A_sun   # W/m^2

    fraction = F_corona / F_total
    print(f"  (a) Solar surface area: A = 4 pi R_sun^2 = {A_sun:.3e} m^2")
    print(f"      Total radiative flux: F_rad = L_sun / A = {F_total:.0f} W/m^2")
    print(f"      Required coronal heating: {F_corona:.0f} W/m^2")
    print(f"      Fraction: F_corona / F_rad = {fraction:.2e}")
    print(f"      Only ~{fraction*100:.3f}% of the total luminosity is needed to heat the corona!")

    # (b) Alfven wave energy flux
    # F_wave = (1/2) rho delta_v^2 * v_A  (for Alfven waves)
    # Only 1% reaches the corona: F_corona = 0.01 * F_wave
    # => F_wave = F_corona / 0.01 = 100 * F_corona
    rho_photo = 2.0e-4   # kg/m^3
    v_A = 1.0e3           # m/s
    transmission = 0.01

    F_wave_needed = F_corona / transmission
    # F_wave = 0.5 * rho * delta_v^2 * v_A
    # delta_v = sqrt(2 * F_wave / (rho * v_A))
    delta_v = np.sqrt(2.0 * F_wave_needed / (rho_photo * v_A))
    delta_v_kms = delta_v / 1e3

    print(f"\n  (b) Alfven wave energy flux: F_wave = (1/2) rho dv^2 v_A")
    print(f"      rho_photosphere = {rho_photo:.1e} kg/m^3")
    print(f"      v_A (photosphere) = {v_A/1e3:.0f} km/s")
    print(f"      Transmission efficiency: {transmission*100:.0f}%")
    print(f"      Required photospheric wave flux: F_wave = {F_wave_needed:.0f} W/m^2")
    print(f"      Required delta_v = sqrt(2 F_wave / (rho v_A))")
    print(f"                       = {delta_v_kms:.2f} km/s = {delta_v:.0f} m/s")

    # (c) Observed motions
    print(f"\n  (c) Observed photospheric velocities:")
    print(f"      - Granulation: 1-2 km/s")
    print(f"      - p-mode oscillations: ~0.3 km/s")
    print(f"      - Required delta_v = {delta_v_kms:.2f} km/s")
    print(f"      The required velocity is comparable to observed photospheric")
    print(f"      motions, making Alfven wave heating plausible in principle.")
    print(f"      The main challenge is the 1% transmission -- most wave energy")
    print(f"      is reflected at the TR due to the steep density gradient.")


def exercise_2():
    """
    Problem 2: RTV Scaling Laws

    Coronal loop: half-length L = 50 Mm, base pressure p0 = 0.5 dyn/cm^2.
    (a) Predict apex temperature using RTV scaling.
    (b) Check gravitational stratification.
    (c) Temperature change if L doubles.
    """
    L = 50.0e6          # half-length [m] = 50 Mm
    p0_cgs = 0.5        # base pressure [dyn/cm^2]
    p0 = p0_cgs * 0.1   # convert to Pa (1 dyn/cm^2 = 0.1 Pa)

    # (a) RTV scaling law: T_max = 1400 * (p0 * L)^(1/3)
    # where T in K, p0 in dyn/cm^2, L in cm
    L_cm = L * 100.0    # convert m to cm
    T_max = 1400.0 * (p0_cgs * L_cm)**(1.0 / 3.0)
    T_max_MK = T_max / 1.0e6

    print(f"  (a) RTV scaling law: T_max = 1400 * (p0 * L)^(1/3)")
    print(f"      p0 = {p0_cgs} dyn/cm^2")
    print(f"      L = {L/1e6:.0f} Mm = {L_cm:.2e} cm")
    print(f"      p0 * L = {p0_cgs * L_cm:.2e} dyn cm^-1")
    print(f"      T_max = 1400 * ({p0_cgs * L_cm:.2e})^(1/3)")
    print(f"            = {T_max:.0f} K = {T_max_MK:.2f} MK")

    # (b) Gravitational stratification
    # Pressure scale height: H = k_B T / (mu m_p g)
    # For corona: mu ~ 0.6, g ~ g_sun = 274 m/s^2
    mu = 0.6
    g = 274.0  # m/s^2
    H = k_B * T_max / (mu * m_p * g)
    H_Mm = H / 1.0e6

    # Effective scale height along semicircular loop: H_eff = pi * H / 2
    H_eff = np.pi * H / 2.0
    H_eff_Mm = H_eff / 1.0e6

    print(f"\n  (b) Pressure scale height: H = k_B T / (mu m_p g)")
    print(f"      H = {H_Mm:.0f} Mm")
    print(f"      Effective scale height for semicircular loop: pi*H/2 = {H_eff_Mm:.0f} Mm")
    print(f"      Loop half-length: L = {L/1e6:.0f} Mm")

    if L < H_eff:
        print(f"      L < pi*H/2 => Loop is NOT significantly stratified")
        print(f"      (pressure approximately uniform along the loop)")
    else:
        print(f"      L > pi*H/2 => Loop IS gravitationally stratified")
        print(f"      (significant pressure drop from footpoint to apex)")

    # (c) If L doubles, what happens to T?
    # T ~ (p0 * L)^(1/3) => T ~ L^(1/3) at constant pressure
    factor_T = 2.0**(1.0 / 3.0)
    T_new = T_max * factor_T
    print(f"\n  (c) If L doubles (at same p0): T ~ L^(1/3)")
    print(f"      T changes by factor 2^(1/3) = {factor_T:.3f}")
    print(f"      New T_max = {T_new:.0f} K = {T_new/1e6:.2f} MK")
    print(f"      Longer loops at the same pressure are hotter.")


def exercise_3():
    """
    Problem 3: Nanoflare Frequency Distribution

    dN/dE = A * E^(-alpha) between E_min=1e24 erg and E_max=1e28 erg.
    (a) alpha = 1.8: total energy rate, show largest events dominate.
    (b) alpha = 2.3: show smallest events dominate.
    (c) Critical alpha value.
    """
    E_min = 1.0e24   # erg
    E_max = 1.0e28   # erg
    A = 1.0          # normalization (arbitrary for relative comparison)

    # Total energy release rate: E_dot = integral E * dN/dE dE
    #                                   = A * integral E * E^(-alpha) dE
    #                                   = A * integral E^(1-alpha) dE

    for case, alpha in [("a", 1.8), ("b", 2.3)]:
        print(f"  ({case}) alpha = {alpha}:")
        exponent = 2.0 - alpha  # exponent of E in the integrand

        if abs(exponent) > 1e-10:
            # integral E^(1-alpha) dE = [E^(2-alpha) / (2-alpha)] from E_min to E_max
            E_dot = A * (E_max**(2.0 - alpha) - E_min**(2.0 - alpha)) / (2.0 - alpha)

            # Contribution from "small" events (lower half in log)
            E_mid = np.sqrt(E_min * E_max)  # geometric mean
            E_dot_small = A * (E_mid**(2.0 - alpha) - E_min**(2.0 - alpha)) / (2.0 - alpha)
            E_dot_large = A * (E_max**(2.0 - alpha) - E_mid**(2.0 - alpha)) / (2.0 - alpha)

            frac_small = E_dot_small / E_dot
            frac_large = E_dot_large / E_dot
        else:
            # alpha = 2: integral is ln(E_max/E_min)
            E_dot = A * np.log(E_max / E_min)
            E_mid = np.sqrt(E_min * E_max)
            E_dot_small = A * np.log(E_mid / E_min)
            E_dot_large = A * np.log(E_max / E_mid)
            frac_small = 0.5
            frac_large = 0.5

        print(f"      Integrand: E^(2-alpha) = E^({2-alpha:.1f})")

        if alpha < 2:
            print(f"      Since 2-alpha = {2-alpha:.1f} > 0, the integrand increases with E.")
            print(f"      => LARGEST events dominate the total energy budget.")
        elif alpha > 2:
            print(f"      Since 2-alpha = {2-alpha:.1f} < 0, the integrand decreases with E.")
            print(f"      => SMALLEST events dominate the total energy budget.")

        print(f"      Fraction from lower half (E < {E_mid:.0e}): {frac_small*100:.1f}%")
        print(f"      Fraction from upper half (E > {E_mid:.0e}): {frac_large*100:.1f}%")
        print()

    # (c) Critical alpha
    print(f"  (c) The critical value is alpha = 2.")
    print(f"      - For alpha < 2: largest events dominate (microflares, not nanoflares)")
    print(f"      - For alpha > 2: smallest events dominate (nanoflares can heat corona)")
    print(f"      - For alpha = 2: equal energy contribution per decade of energy")
    print(f"      ")
    print(f"      Physical significance: if the observed distribution has alpha > 2,")
    print(f"      then the cumulative energy in unresolved small events exceeds that")
    print(f"      of observable larger events, supporting the nanoflare heating hypothesis.")
    print(f"      Observations find alpha ~ 1.5-2.6 depending on the diagnostic,")
    print(f"      making this a topic of active debate.")


def exercise_4():
    """
    Problem 4: Thermal Non-Equilibrium

    Loop half-length L = 80 Mm, footpoint heating below h0 = 10 Mm.
    (a) RTV equilibrium temperature for uniform heating.
    (b) Why footpoint heating leads to TNE.
    (c) Coronal rain fall time.
    """
    L = 80.0e6        # half-length [m]
    h0 = 10.0e6       # heating scale height [m]

    # (a) RTV equilibrium with uniform heating
    # Need to assume a pressure. For a typical AR loop, p ~ 1 dyn/cm^2
    p0_cgs = 1.0  # dyn/cm^2
    L_cm = L * 100.0

    T_max = 1400.0 * (p0_cgs * L_cm)**(1.0 / 3.0)
    T_max_MK = T_max / 1.0e6

    print(f"  (a) Loop half-length: L = {L/1e6:.0f} Mm")
    print(f"      Assuming p0 = {p0_cgs} dyn/cm^2 (typical active region)")
    print(f"      RTV temperature: T_max = 1400 * (p0 * L)^(1/3)")
    print(f"                     = {T_max:.0f} K = {T_max_MK:.2f} MK")

    # (b) TNE explanation
    print(f"\n  (b) With footpoint-concentrated heating (below h0 = {h0/1e6:.0f} Mm):")
    print(f"      - The apex region receives very little direct heating")
    print(f"      - It relies on thermal conduction from the heated footpoints")
    print(f"      - For long loops (L >> h0), the conductive flux cannot maintain")
    print(f"        the apex at coronal temperatures")
    print(f"      - The apex cools radiatively, increasing density through")
    print(f"        chromospheric evaporation, leading to more radiation")
    print(f"      - This positive feedback creates a thermal instability")
    print(f"      - The result is periodic cycling: heating -> evaporation ->")
    print(f"        condensation -> catastrophic cooling -> coronal rain -> repeat")
    print(f"      - No static equilibrium exists: 'thermal non-equilibrium' (TNE)")

    # (c) Coronal rain fall time
    v_rain = 100.0e3   # fall speed [m/s]
    # For a semicircular loop, the apex is at height ~ L * 2/pi above footpoint
    # Path from apex to footpoint along the loop = L (half-length)
    path_length = L  # along the loop

    t_fall = path_length / v_rain
    t_fall_min = t_fall / 60.0
    t_fall_hr = t_fall / 3600.0

    print(f"\n  (c) Coronal rain fall speed: {v_rain/1e3:.0f} km/s")
    print(f"      Path from apex to footpoint: L = {L/1e6:.0f} Mm")
    print(f"      Fall time: t = L/v = {t_fall:.0f} s = {t_fall_min:.0f} min = {t_fall_hr:.1f} hr")
    print(f"      Typical TNE cycle period: 2-10 hours")
    print(f"      The fall time ({t_fall_min:.0f} min) is shorter than the full cycle")
    print(f"      because the cycle includes heating, evaporation, and condensation")
    print(f"      phases before the rain falls.")


def exercise_5():
    """
    Problem 5: DEM Inversion

    (a) Integral equation relating AIA intensities to DEM.
    (b) Why the inversion is ill-posed.
    (c) Two physical scenarios for DEM with 1.5 MK peak + 8 MK tail.
    (d) Observational test to distinguish them.
    """
    # This problem is largely conceptual/analytical
    print(f"  (a) The observed intensity in AIA channel i is:")
    print(f"      I_i = integral_T R_i(T) * DEM(T) dT")
    print(f"      where R_i(T) is the temperature response function of channel i,")
    print(f"      and DEM(T) = n_e^2 * ds/dT [cm^-5 K^-1] is the differential")
    print(f"      emission measure.")

    print(f"\n  (b) The inversion is ill-posed because:")
    print(f"      - We have only 6 measurements (I_1 through I_6) but want to")
    print(f"        recover a continuous function DEM(T)")
    print(f"      - The response functions R_i(T) overlap significantly, so the")
    print(f"        6 channels do not provide 6 independent constraints")
    print(f"      - Small errors in I_i can produce large changes in DEM(T)")
    print(f"      - Multiple very different DEM solutions can produce nearly")
    print(f"        identical predicted intensities (non-uniqueness)")
    print(f"      - Regularization or prior assumptions are needed (e.g., smoothness)")

    print(f"\n  (c) Two scenarios for a DEM with 1.5 MK peak + 8 MK tail:")
    print(f"      ")
    print(f"      Scenario 1 (Steady heating):")
    print(f"      - Multi-thermal loop system with different-temperature strands")
    print(f"      - Most loops are at ~1.5 MK (peak)")
    print(f"      - Some high-temperature loops heated to ~8 MK (flare-productive AR)")
    print(f"      - Each strand is in quasi-steady equilibrium")
    print(f"      ")
    print(f"      Scenario 2 (Impulsive nanoflares):")
    print(f"      - Individual strands are heated impulsively by nanoflares")
    print(f"      - After each nanoflare, plasma heats to ~10 MK then cools")
    print(f"      - The 8 MK tail represents recently heated plasma (cooling phase)")
    print(f"      - The 1.5 MK peak represents fully cooled plasma (majority)")
    print(f"      - The DEM is a superposition of many strands at different cooling stages")

    print(f"\n  (d) Distinguishing observational tests:")
    print(f"      - Time variability: Nanoflares produce flickering in hot channels")
    print(f"        (94 A, 131 A); steady heating produces constant emission")
    print(f"      - DEM slope at high T: Nanoflare models predict a specific")
    print(f"        power-law slope for the hot tail related to the cooling timescale")
    print(f"      - Emission measure time lag analysis: In the nanoflare scenario,")
    print(f"        hot channels peak BEFORE cool channels as plasma cools")
    print(f"      - Non-equilibrium ionization: Rapid heating may produce Fe charge")
    print(f"        states out of ionization equilibrium (detectable in spectroscopy)")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Coronal Energy Budget ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: RTV Scaling Laws ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Nanoflare Frequency Distribution ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Thermal Non-Equilibrium ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: DEM Inversion ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)

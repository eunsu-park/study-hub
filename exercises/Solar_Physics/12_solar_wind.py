"""
Exercises for Lesson 12: Solar Wind
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import integrate


# --- Physical constants ---
k_B = 1.381e-23        # Boltzmann constant [J/K]
m_p = 1.673e-27        # proton mass [kg]
G = 6.674e-11          # gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30       # solar mass [kg]
R_sun = 6.957e8        # solar radius [m]
AU = 1.496e11          # astronomical unit [m]
mu_0 = 4.0e-7 * np.pi  # vacuum permeability [T m/A]
yr_s = 3.156e7          # year in seconds


def exercise_1():
    """
    Problem 1: Solar Wind Mass Loss

    n = 5 cm^-3, v = 400 km/s at 1 AU.
    (a) Mass loss rate in g/s and M_sun/yr.
    (b) Time to lose 1% of solar mass.
    (c) Compare with radiative mass loss rate.
    """
    n = 5.0 * 1e6        # cm^-3 to m^-3
    v = 400.0e3           # m/s
    r = AU

    # (a) Mass loss rate: M_dot = 4 pi r^2 * n * m_p * v
    M_dot = 4.0 * np.pi * r**2 * n * m_p * v  # kg/s
    M_dot_gs = M_dot * 1e3   # g/s
    M_dot_Msun_yr = M_dot * yr_s / M_sun

    print(f"  (a) At 1 AU: n = {n/1e6:.0f} cm^-3, v = {v/1e3:.0f} km/s")
    print(f"      M_dot = 4 pi r^2 n m_p v")
    print(f"            = {M_dot:.2e} kg/s")
    print(f"            = {M_dot_gs:.2e} g/s")
    print(f"            = {M_dot_Msun_yr:.2e} M_sun/yr")

    # (b) Time to lose 1% of solar mass
    M_01 = 0.01 * M_sun
    t_01 = M_01 / M_dot
    t_01_yr = t_01 / yr_s

    print(f"\n  (b) Time to lose 1% of M_sun:")
    print(f"      t = 0.01 * M_sun / M_dot")
    print(f"        = {t_01_yr:.2e} years")
    print(f"      This is much longer than the Sun's age (4.57e9 years).")
    print(f"      The solar wind mass loss is negligible over the Sun's lifetime.")

    # (c) Radiative mass loss: L_sun / c^2
    L_sun = 3.828e26  # W
    c = 3.0e8          # m/s
    M_dot_rad = L_sun / c**2  # kg/s
    ratio = M_dot / M_dot_rad

    print(f"\n  (c) Radiative mass loss: M_dot_rad = L_sun/c^2 = {M_dot_rad:.2e} kg/s")
    print(f"      Solar wind mass loss: {M_dot:.2e} kg/s")
    print(f"      Ratio (wind/radiation): {ratio:.1f}")
    print(f"      The solar wind loses mass ~{ratio:.0f}x faster than radiation!")
    print(f"      (But both are negligible compared to M_sun.)")


def exercise_2():
    """
    Problem 2: Parker Spiral Angle

    (a) Derive garden-hose angle.
    (b) Calculate at 1 AU for slow (350 km/s) and fast (750 km/s) wind.
    (c) Distance where psi = 80 deg for 400 km/s wind.
    (d) Sketch (text description).
    """
    Omega = 2.0 * np.pi / (25.4 * 86400.0)  # sidereal rotation rate [rad/s]
    # ~27 days synodic, ~25.4 days sidereal

    # (a) Garden-hose angle derivation
    print(f"  (a) Parker spiral angle (garden-hose angle):")
    print(f"      In the rotating frame, the field line connects the Sun to the wind parcel.")
    print(f"      A radial wind at speed v_sw carries the field outward.")
    print(f"      Solar rotation at Omega wraps the field line azimuthally.")
    print(f"      ")
    print(f"      At distance r: B_r ~ 1/r^2, B_phi ~ -Omega*(r-R_sun)*sin(theta)/(v_sw*r)*B_r*r^2")
    print(f"      tan(psi) = -B_phi/B_r = Omega * (r - R_sun) * sin(theta) / v_sw")
    print(f"      For r >> R_sun: tan(psi) ~ Omega * r / v_sw (at equator, sin(theta)=1)")
    print(f"      Omega = 2pi/P_rot = {Omega:.3e} rad/s (sidereal)")

    # (b) At 1 AU
    r = AU
    for label, v_sw in [("Slow wind", 350.0e3), ("Fast wind", 750.0e3)]:
        tan_psi = Omega * r / v_sw
        psi = np.degrees(np.arctan(tan_psi))
        print(f"\n  (b) {label} (v_sw = {v_sw/1e3:.0f} km/s):")
        print(f"      tan(psi) = Omega * r / v_sw = {tan_psi:.2f}")
        print(f"      psi = {psi:.1f} deg")

    # (c) Distance where psi = 80 deg for v_sw = 400 km/s
    v_sw = 400.0e3
    psi_target = 80.0
    tan_target = np.tan(np.radians(psi_target))
    r_80 = tan_target * v_sw / Omega
    r_80_AU = r_80 / AU

    print(f"\n  (c) For psi = {psi_target} deg with v_sw = {v_sw/1e3:.0f} km/s:")
    print(f"      r = v_sw * tan(psi) / Omega")
    print(f"        = {r_80_AU:.1f} AU")

    # (d) Description
    print(f"\n  (d) Field line sketch (ecliptic plane):")
    print(f"      - Slow wind (350 km/s): tightly wound spiral, ~{np.degrees(np.arctan(Omega*AU/350e3)):.0f} deg at 1 AU")
    print(f"      - Fast wind (750 km/s): loosely wound spiral, ~{np.degrees(np.arctan(Omega*AU/750e3)):.0f} deg at 1 AU")
    print(f"      Both spirals originate from the same rotating source.")
    print(f"      The slow wind sector has more tightly wound field lines")
    print(f"      (more azimuthal) than the fast wind sector (more radial).")


def exercise_3():
    """
    Problem 3: CIR Compression

    Fast: v_f=700 km/s, n_f=3 cm^-3, B_f=4 nT
    Slow: v_s=350 km/s, n_s=10 cm^-3, B_s=5 nT
    (a) Inflow speeds in stream interface frame.
    (b) Compression ratio estimates.
    """
    v_f = 700.0     # km/s
    n_f = 3.0       # cm^-3
    B_f = 4.0       # nT
    v_s = 350.0     # km/s
    n_s = 10.0      # cm^-3
    B_s = 5.0       # nT

    # (a) Stream interface frame
    # The interface moves at some intermediate speed
    # Approximate: v_interface ~ (v_f + v_s) / 2 = 525 km/s
    # More accurately for mass-balanced interface:
    # Momentum balance: n_f * m_p * (v_f - v_i) ~ n_s * m_p * (v_i - v_s)
    # v_i = (n_f * v_f + n_s * v_s) / (n_f + n_s)
    v_i = (n_f * v_f + n_s * v_s) / (n_f + n_s)

    v_f_inflow = v_f - v_i  # fast wind inflow (from behind)
    v_s_inflow = v_i - v_s  # slow wind inflow (from ahead)

    print(f"  (a) Stream interface frame:")
    print(f"      Interface speed (density-weighted): v_i = {v_i:.0f} km/s")
    print(f"      Fast wind inflow: v_f - v_i = {v_f_inflow:.0f} km/s")
    print(f"      Slow wind inflow: v_i - v_s = {v_s_inflow:.0f} km/s")

    # (b) Compression ratios
    # Fast magnetosonic speed: c_f = sqrt(c_s^2 + v_A^2)
    # Approximate: c_s ~ 50 km/s for coronal wind at 1 AU (T ~ 1e5 K)
    T_sw = 1.0e5  # K (typical)
    c_s = np.sqrt(k_B * T_sw / m_p) / 1e3  # km/s

    # Alfven speed
    # v_A = B / sqrt(mu_0 * n * m_p)
    for label, n, B, v_in in [("Fast wind shock", n_f, B_f, v_f_inflow),
                                ("Slow wind shock", n_s, B_s, v_s_inflow)]:
        n_m3 = n * 1e6  # cm^-3 to m^-3
        B_T = B * 1e-9  # nT to T
        v_A = B_T / np.sqrt(mu_0 * n_m3 * m_p) / 1e3  # km/s
        c_fast = np.sqrt(c_s**2 + v_A**2)
        M = v_in / c_fast  # Mach number

        # Rankine-Hugoniot for perpendicular shock (high Mach limit):
        # r = (gamma+1)*M^2 / ((gamma-1)*M^2 + 2) for gas dynamic
        gamma = 5.0 / 3.0
        if M > 1:
            r_compression = (gamma + 1) * M**2 / ((gamma - 1) * M**2 + 2.0)
        else:
            r_compression = 1.0  # no shock for subsonic flow

        print(f"\n  (b) {label}:")
        print(f"      n = {n:.0f} cm^-3, B = {B:.0f} nT")
        print(f"      v_A = {v_A:.0f} km/s, c_s = {c_s:.0f} km/s")
        print(f"      c_fast = sqrt(c_s^2 + v_A^2) = {c_fast:.0f} km/s")
        print(f"      Inflow speed: {v_in:.0f} km/s")
        print(f"      Fast Mach number: M = {M:.1f}")
        if M > 1:
            print(f"      Compression ratio: r = {r_compression:.2f}")
        else:
            print(f"      M < 1: no shock forms (subsonic)")


def exercise_4():
    """
    Problem 4: Turbulence Spectrum

    P(f) = P_0 * f^(-5/3), inertial range f1=1e-3 to f2=0.5 Hz.
    P_0 = 10 nT^2/Hz at f = 1e-2 Hz.
    (a) Total fluctuation energy.
    (b) Ratio to mean field energy.
    (c) Turbulent heating rate.
    """
    alpha = 5.0 / 3.0  # spectral index
    f1 = 1.0e-3         # Hz (inertial range start)
    f2 = 0.5            # Hz (inertial range end)
    P_ref = 10.0        # nT^2/Hz at f_ref
    f_ref = 1.0e-2      # Hz
    n = 5.0 * 1e6       # m^-3
    B_0 = 5.0           # nT

    # P(f) = P_ref * (f / f_ref)^(-5/3)
    # => P_0 = P_ref * f_ref^(5/3)  (so P(f) = P_0 * f^(-5/3) where P_0 = P_ref * f_ref^alpha)

    # (a) Total fluctuation energy: <dB^2> = integral P(f) df from f1 to f2
    # = P_ref * f_ref^alpha * integral f^(-alpha) df from f1 to f2
    # = P_ref * f_ref^alpha * [f^(1-alpha)/(1-alpha)] from f1 to f2

    exponent = 1.0 - alpha  # = -2/3
    coeff = P_ref * f_ref**alpha  # nT^2/Hz * Hz^(5/3) = nT^2 * Hz^(2/3)

    dB2 = coeff * (f2**exponent - f1**exponent) / exponent  # nT^2
    dB_rms = np.sqrt(dB2)

    print(f"  (a) Power spectrum: P(f) = {P_ref} nT^2/Hz * (f/{f_ref})^(-5/3)")
    print(f"      Inertial range: {f1:.0e} to {f2} Hz")
    print(f"      <delta B^2> = integral P(f) df = {dB2:.1f} nT^2")
    print(f"      delta B_rms = {dB_rms:.1f} nT")

    # (b) Ratio to mean field energy
    B_0_nT = B_0
    ratio = dB2 / B_0_nT**2
    print(f"\n  (b) Mean field: B_0 = {B_0_nT:.0f} nT")
    print(f"      <delta B^2> / B_0^2 = {dB2:.1f} / {B_0_nT**2:.0f} = {ratio:.2f}")
    if ratio > 1:
        print(f"      The fluctuation energy EXCEEDS the mean field energy!")
        print(f"      The turbulence is strong (delta B ~ B_0).")
    else:
        print(f"      The fluctuation energy is {ratio*100:.0f}% of mean field energy.")

    # (c) Turbulent heating rate
    # Kolmogorov cascade: epsilon ~ delta_v^3 / l
    # Energy equipartition: delta_v ~ delta_B / sqrt(mu_0 * rho)
    rho = n * m_p  # kg/m^3
    dB_SI = dB_rms * 1.0e-9  # T
    delta_v = dB_SI / np.sqrt(mu_0 * rho)

    # Outer scale: correlation length ~ v_sw / f1 or 1/f1 * v_sw
    v_sw = 400.0e3  # m/s
    l_outer = v_sw / f1  # m (using Taylor hypothesis)

    epsilon = delta_v**3 / l_outer  # W/kg (specific heating rate)
    # Volumetric heating rate
    Q_vol = rho * epsilon  # W/m^3

    print(f"\n  (c) Turbulent heating rate estimate:")
    print(f"      delta_v ~ delta_B / sqrt(mu_0 rho) = {delta_v/1e3:.1f} km/s")
    print(f"      Outer scale: l ~ v_sw/f_min = {l_outer:.2e} m = {l_outer/AU:.3f} AU")
    print(f"      Specific rate: epsilon ~ delta_v^3 / l = {epsilon:.2e} W/kg")
    print(f"      Volumetric rate: Q = rho * epsilon = {Q_vol:.2e} W/m^3")
    print(f"      This rate should be compared with the empirical heating rate")
    print(f"      needed to explain the non-adiabatic temperature profile of")
    print(f"      the solar wind at 1 AU.")


def exercise_5():
    """
    Problem 5: Critical Point

    Parker isothermal model, T = 1.5e6 K.
    (a) Isothermal sound speed.
    (b) Critical radius.
    (c) Asymptotic wind speed using Bernoulli equation.
    """
    T = 1.5e6          # K

    # (a) Isothermal sound speed
    c_s = np.sqrt(k_B * T / m_p)
    c_s_kms = c_s / 1e3

    print(f"  (a) Coronal temperature: T = {T:.1e} K")
    print(f"      Isothermal sound speed: c_s = sqrt(k_B T / m_p)")
    print(f"                             = {c_s:.0f} m/s = {c_s_kms:.0f} km/s")

    # (b) Critical radius: r_c = G M_sun / (2 c_s^2)
    r_c = G * M_sun / (2.0 * c_s**2)
    r_c_Rsun = r_c / R_sun

    print(f"\n  (b) Critical radius: r_c = G M_sun / (2 c_s^2)")
    print(f"                          = {r_c:.3e} m")
    print(f"                          = {r_c_Rsun:.1f} R_sun")

    # (c) Asymptotic wind speed (Bernoulli equation)
    # (1/2)v_inf^2 ~ 2*c_s^2*ln(r_c/r_0) - G*M_sun/r_0 + (1/2)*c_s^2 + (5/2)*c_s^2
    # Using r_0 = R_sun
    r_0 = R_sun
    term1 = 2.0 * c_s**2 * np.log(r_c / r_0)
    term2 = -G * M_sun / r_0
    term3 = 0.5 * c_s**2
    term4 = 2.5 * c_s**2

    v_inf_sq = 2.0 * (term1 + term2 + term3 + term4)

    print(f"\n  (c) Bernoulli equation for asymptotic speed:")
    print(f"      (1/2)v_inf^2 = 2*c_s^2*ln(r_c/r_0) - GM/r_0 + (1/2)c_s^2 + (5/2)c_s^2")
    print(f"      Terms:")
    print(f"        2*c_s^2*ln(r_c/R_sun) = {term1:.3e} m^2/s^2")
    print(f"        -GM/R_sun = {term2:.3e} m^2/s^2")
    print(f"        (1/2)*c_s^2 = {term3:.3e} m^2/s^2")
    print(f"        (5/2)*c_s^2 = {term4:.3e} m^2/s^2")
    print(f"      Sum = {term1 + term2 + term3 + term4:.3e} m^2/s^2")

    if v_inf_sq > 0:
        v_inf = np.sqrt(v_inf_sq)
        v_inf_kms = v_inf / 1e3
        print(f"      v_inf = {v_inf_kms:.0f} km/s")
    else:
        print(f"      v_inf^2 < 0 => no wind solution at this temperature!")
        print(f"      The isothermal Parker model at T = {T/1e6:.1f} MK does not produce")
        print(f"      a supersonic wind. (The gravitational potential is too deep.)")
        v_inf_kms = 0.0

    print(f"\n      Discussion:")
    print(f"      The isothermal Parker model at 1.5 MK gives v_inf ~ {v_inf_kms:.0f} km/s,")
    print(f"      which is comparable to the slow solar wind (~350-450 km/s)")
    if v_inf_kms > 0 and v_inf_kms < 700:
        print(f"      but insufficient for the fast wind (700-800 km/s).")
        print(f"      The fast wind requires additional acceleration, likely from:")
        print(f"      - Alfven wave pressure (wave-driven wind)")
        print(f"      - Non-isothermal temperature profiles (extended heating)")
        print(f"      - Coronal hole geometry (flux tube expansion)")
    else:
        print(f"      Additional heating/acceleration mechanisms are needed.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Solar Wind Mass Loss ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Parker Spiral Angle ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: CIR Compression ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Turbulence Spectrum ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Critical Point ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)

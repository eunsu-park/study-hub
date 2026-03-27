"""
Plasma Physics - Lesson 11: Electromagnetic Waves
Exercise Solutions

Topics covered:
- Ionospheric reflection of radio waves
- ECRH system design for tokamak heating
- Whistler wave propagation and dispersion
- CMA diagram region identification
- Faraday rotation measurement
"""

import numpy as np

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
mu_0 = 4 * np.pi * 1e-7
c = 3e8
eV_to_J = e


def exercise_1():
    """
    Exercise 1: Ionospheric Reflection
    Peak ionospheric density: n_e = 10^12 m^-3.
    """
    print("--- Exercise 1: Ionospheric Reflection ---")

    n_peak = 1e12  # Peak electron density [m^-3]

    # (a) Plasma frequency at peak
    omega_pe = np.sqrt(n_peak * e**2 / (epsilon_0 * m_e))
    f_pe = omega_pe / (2 * np.pi)

    print(f"(a) Peak ionospheric density: n_e = {n_peak:.0e} m^-3")
    print(f"    Plasma frequency: f_pe = {f_pe/1e6:.2f} MHz")
    print(f"    omega_pe = {omega_pe:.4e} rad/s")

    # (b) AM radio at 1 MHz
    f_AM = 1e6  # Hz
    print(f"\n(b) AM radio at f = {f_AM/1e6} MHz:")
    if f_AM < f_pe:
        print(f"    f_AM < f_pe -> signal REFLECTS off ionosphere")
        print(f"    This enables long-distance AM radio (skywave propagation)")
    else:
        print(f"    f_AM > f_pe -> signal passes through")

    # (c) Minimum frequency for transmission
    print(f"\n(c) Minimum frequency for ionospheric transmission:")
    print(f"    f_min = f_pe = {f_pe/1e6:.2f} MHz")
    print(f"    All frequencies above {f_pe/1e6:.1f} MHz pass through the ionosphere")
    print(f"    FM radio (88-108 MHz) >> f_pe -> always passes through")
    print(f"    This is why FM has limited range (line-of-sight only)")

    # (d) Daytime density increase
    n_day = n_peak * 10
    f_pe_day = np.sqrt(n_day * e**2 / (epsilon_0 * m_e)) / (2 * np.pi)

    print(f"\n(d) Daytime (n increases 10x):")
    print(f"    n_day = {n_day:.0e} m^-3")
    print(f"    f_pe_day = {f_pe_day/1e6:.1f} MHz")
    print(f"    AM radio at 1 MHz:")
    if f_AM < f_pe_day:
        print(f"    Still reflects (f_AM < f_pe_day)")
    print(f"    Higher frequencies (up to {f_pe_day/1e6:.0f} MHz) now also reflect")
    print(f"    D-layer absorption increases during day -> AM signals weaker")
    print(f"    At night: D-layer disappears, AM skywave propagation improves")
    print()


def exercise_2():
    """
    Exercise 2: ECRH System Design
    Tokamak: B_0 = 3.5 T on axis.
    """
    print("--- Exercise 2: ECRH System Design ---")

    B_0 = 3.5  # T (on axis)
    n = 5e19   # m^-3

    # (a) Electron cyclotron frequency
    omega_ce = e * B_0 / m_e
    f_ce = omega_ce / (2 * np.pi)
    lambda_ce = c / f_ce

    print(f"(a) B_0 = {B_0} T on axis")
    print(f"    f_ce = eB/(2*pi*m_e) = {f_ce/1e9:.1f} GHz")
    print(f"    Wavelength: lambda = {lambda_ce*1e3:.2f} mm")

    # (b) 2nd harmonic
    f_2ce = 2 * f_ce
    print(f"\n(b) 2nd harmonic ECRH:")
    print(f"    f_gyrotron = 2*f_ce = {f_2ce/1e9:.1f} GHz")
    print(f"    Wavelength: {c/f_2ce*1e3:.2f} mm")

    # (c) O-mode cutoff density at 2*f_ce
    # O-mode cutoff: omega = omega_pe -> n_cutoff at omega = 2*omega_ce
    omega_2ce = 2 * omega_ce
    n_cutoff = epsilon_0 * m_e * omega_2ce**2 / e**2

    print(f"\n(c) O-mode cutoff density at 2*f_ce:")
    print(f"    n_cutoff = epsilon_0*m_e*(2*omega_ce)^2/e^2 = {n_cutoff:.2e} m^-3")
    print(f"    Actual density: n = {n:.0e} m^-3")
    if n < n_cutoff:
        print(f"    n < n_cutoff: O-mode CAN reach the center!")
    else:
        print(f"    n > n_cutoff: O-mode CANNOT reach the center")
        print(f"    Need alternative approach")

    # (d) Alternative approaches
    print(f"\n(d) If O-mode blocked:")
    print(f"    Option 1: X-mode from high-field side")
    print(f"      X-mode has different cutoff/resonance structure")
    print(f"      Can access region where O-mode is cutoff")
    print(f"    Option 2: O-X-B mode conversion")
    print(f"      O-mode -> slow X-mode at O-cutoff")
    print(f"      X-mode -> EBW at UH resonance")
    print(f"      EBW absorbed at cyclotron harmonic")
    print(f"    Option 3: Higher harmonic (3rd: f = {3*f_ce/1e9:.0f} GHz)")
    print(f"      Higher cutoff density, but weaker absorption")
    print()


def exercise_3():
    """
    Exercise 3: Whistler Propagation
    Lightning stroke at magnetic equator, propagation to opposite hemisphere.
    B = 5e-5 T, n = 10^7 m^-3.
    """
    print("--- Exercise 3: Whistler Propagation ---")

    B = 5e-5     # T (Earth's field)
    n = 1e7      # m^-3 (magnetosphere)
    m_i = m_p    # Hydrogen (protons in magnetosphere)

    # (a) Cyclotron and plasma frequencies
    omega_ce = e * B / m_e
    omega_ci = e * B / m_i
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    f_ce = omega_ce / (2 * np.pi)
    f_ci = omega_ci / (2 * np.pi)
    f_pe = omega_pe / (2 * np.pi)

    print(f"(a) B = {B*1e6:.0f} uT, n = {n:.0e} m^-3")
    print(f"    f_ce = {f_ce/1e3:.2f} kHz")
    print(f"    f_ci = {f_ci:.2f} Hz")
    print(f"    f_pe = {f_pe/1e3:.2f} kHz")

    # (b) Verify whistler frequency range
    f_test = np.array([1e3, 5e3, 10e3])  # Hz
    print(f"\n(b) Whistler range: f_ci << f << f_ce")
    print(f"    f_ci = {f_ci:.2f} Hz, f_ce = {f_ce/1e3:.2f} kHz")
    for f in f_test:
        satisfies = f > 10 * f_ci and f < 0.1 * f_ce
        print(f"    f = {f/1e3:.0f} kHz: f_ci << f << f_ce? {satisfies}")

    # (c) Group velocity at 5 kHz
    # Whistler dispersion: omega = k^2*c^2 * omega_ce / (omega_pe^2 + k^2*c^2)
    # For omega_pe >> omega_ce (typical): omega ~ k^2*c^2*omega_ce / omega_pe^2
    # Group velocity: v_g = d(omega)/dk = 2*c*sqrt(omega*omega_ce) / omega_pe
    f_5k = 5e3
    omega_5k = 2 * np.pi * f_5k
    v_g_5k = 2 * c * np.sqrt(omega_5k * omega_ce) / omega_pe

    print(f"\n(c) Group velocity at f = 5 kHz:")
    print(f"    v_g = 2c*sqrt(omega*omega_ce)/omega_pe")
    print(f"    v_g(5 kHz) = {v_g_5k:.4e} m/s = {v_g_5k/1e3:.0f} km/s")
    print(f"    v_g / c = {v_g_5k/c:.4f}")

    # (d) Travel times
    L = 10e6  # Path length 10,000 km [m]

    frequencies = [1e3, 2e3, 5e3, 10e3]
    print(f"\n(d) Travel time for L = {L/1e6:.0f} Mm:")
    print(f"    {'f [kHz]':>10} {'v_g [km/s]':>12} {'t [s]':>10}")
    print("    " + "-" * 36)

    for f in frequencies:
        omega_f = 2 * np.pi * f
        v_g = 2 * c * np.sqrt(omega_f * omega_ce) / omega_pe
        t = L / v_g
        print(f"    {f/1e3:>10.0f} {v_g/1e3:>12.0f} {t:>10.3f}")

    # Time difference between high and low frequencies
    v_g_1k = 2 * c * np.sqrt(2 * np.pi * 1e3 * omega_ce) / omega_pe
    v_g_10k = 2 * c * np.sqrt(2 * np.pi * 10e3 * omega_ce) / omega_pe
    t_1k = L / v_g_1k
    t_10k = L / v_g_10k
    delta_t = t_1k - t_10k

    print(f"\n    Time delay (1 kHz vs 10 kHz): Delta_t = {delta_t:.2f} s")
    print(f"    High frequency arrives first (faster v_g)")
    print(f"    This creates the descending 'whistling' tone")
    print(f"    (higher pitch first, then descending as slower low-f arrives)")
    print()


def exercise_4():
    """
    Exercise 4: CMA Diagram Regions
    f_pe = 50 GHz, f_ce = 70 GHz.
    Determine propagating modes at different frequencies.
    """
    print("--- Exercise 4: CMA Diagram Regions ---")

    f_pe = 50e9   # Hz
    f_ce = 70e9   # Hz
    omega_pe = 2 * np.pi * f_pe
    omega_ce = 2 * np.pi * f_ce

    print(f"Plasma parameters: f_pe = {f_pe/1e9:.0f} GHz, f_ce = {f_ce/1e9:.0f} GHz")
    print(f"  omega_pe/omega_ce = {omega_pe/omega_ce:.3f}")
    print()

    # Stix parameters: S, D, P
    # For each frequency, calculate X = omega_pe^2/omega^2, Y = omega_ce/omega
    # P = 1 - X
    # R = 1 - X/(1 - Y)
    # L = 1 - X/(1 + Y)
    # S = (R + L)/2
    # D = (R - L)/2

    test_frequencies = [
        ("(a) 40 GHz (X-band)", 40e9),
        ("(b) 75 GHz (W-band)", 75e9),
        ("(c) 120 GHz (ECRH 2nd harm)", 120e9),
    ]

    for label, f in test_frequencies:
        omega = 2 * np.pi * f
        X = (omega_pe / omega)**2
        Y = omega_ce / omega

        P = 1 - X
        R = 1 - X / (1 - Y)
        L = 1 - X / (1 + Y)
        S = (R + L) / 2
        D = (R - L) / 2

        # Refractive index for parallel propagation (theta = 0)
        n_R_sq = 1 - X / (1 - Y)  # R-wave (RCP)
        n_L_sq = 1 - X / (1 + Y)  # L-wave (LCP)

        # Refractive index for perpendicular propagation (theta = pi/2)
        n_O_sq = P   # O-mode: n^2 = P = 1 - X
        n_X_sq = (S**2 - D**2) / S if S != 0 else float('inf')  # X-mode: n^2 = (RL)/S

        print(f"\n{label}:")
        print(f"    f = {f/1e9:.0f} GHz, X = {X:.4f}, Y = {Y:.4f}")
        print(f"    Stix parameters: S = {S:.4f}, D = {D:.4f}, P = {P:.4f}")
        print(f"    R = {R:.4f}, L = {L:.4f}")
        print(f"    Parallel: n_R^2 = {n_R_sq:.4f} ({'propagates' if n_R_sq > 0 else 'evanescent'})")
        print(f"              n_L^2 = {n_L_sq:.4f} ({'propagates' if n_L_sq > 0 else 'evanescent'})")
        print(f"    Perp:     n_O^2 = {n_O_sq:.4f} ({'propagates' if n_O_sq > 0 else 'evanescent'})")
        print(f"              n_X^2 = {n_X_sq:.4f} ({'propagates' if n_X_sq > 0 else 'evanescent'})")

        modes = []
        if n_R_sq > 0:
            modes.append("R-wave")
        if n_L_sq > 0:
            modes.append("L-wave")
        if n_O_sq > 0:
            modes.append("O-mode")
        if n_X_sq > 0:
            modes.append("X-mode")
        print(f"    Accessible modes: {', '.join(modes) if modes else 'None (evanescent)'}")

    print()


def exercise_5():
    """
    Exercise 5: Faraday Rotation Measurement
    Polarized wave at 5 GHz through plasma slab.
    n_e = 10^18 m^-3, B_par = 0.1 T, L = 1 m.
    """
    print("--- Exercise 5: Faraday Rotation ---")

    n_e = 1e18
    B_par = 0.1  # T (parallel component)
    L = 1.0       # m (path length)
    f1 = 5e9      # 5 GHz
    f2 = 10e9     # 10 GHz
    lambda1 = c / f1  # 6 cm
    lambda2 = c / f2  # 3 cm

    omega_pe = np.sqrt(n_e * e**2 / (epsilon_0 * m_e))
    omega_ce = e * B_par / m_e

    # (a) Faraday rotation angle
    # theta = (e^3 / (2*m_e^2*c)) * n_e * B_par * L * lambda^2 / (2*pi*c)
    # More standard: theta = (omega_pe^2 * omega_ce * L) / (2*c*omega^2)
    # = (e^3 * n_e * B_par * L) / (2 * m_e^2 * epsilon_0 * c * omega^2)

    omega1 = 2 * np.pi * f1
    theta1 = omega_pe**2 * omega_ce * L / (2 * c * omega1**2)

    print(f"(a) Faraday rotation at f = {f1/1e9} GHz:")
    print(f"    n_e = {n_e:.0e} m^-3, B_par = {B_par} T, L = {L} m")
    print(f"    omega_pe = {omega_pe:.4e} rad/s")
    print(f"    omega_ce = {omega_ce:.4e} rad/s")
    print(f"    theta = {theta1:.4f} rad = {np.degrees(theta1):.2f} degrees")

    # (b) Two-wavelength measurement
    omega2 = 2 * np.pi * f2
    theta2 = omega_pe**2 * omega_ce * L / (2 * c * omega2**2)
    delta_theta = theta1 - theta2

    print(f"\n(b) Two-wavelength measurement:")
    print(f"    theta(lambda1 = {lambda1*100:.0f} cm) = {np.degrees(theta1):.4f} deg")
    print(f"    theta(lambda2 = {lambda2*100:.0f} cm) = {np.degrees(theta2):.4f} deg")
    print(f"    Delta_theta = {np.degrees(delta_theta):.4f} deg = {delta_theta:.4f} rad")

    # (c) Rotation measure RM
    # RM = theta / lambda^2 (in CGS-like convention)
    # More precisely: theta = RM * lambda^2
    # RM = (e^3 / (8*pi^2*m_e^2*c^3*epsilon_0)) * integral n_e * B_par * dl
    RM = theta1 / lambda1**2

    print(f"\n(c) Rotation Measure:")
    print(f"    RM = theta / lambda^2 = {RM:.4e} rad/m^2")
    print(f"    Verify: RM * lambda1^2 = {RM * lambda1**2:.4f} rad (= theta1)")
    print(f"    RM * lambda2^2 = {RM * lambda2**2:.4f} rad (= theta2)")

    # (d) Separating n_e and B
    print(f"\n(d) Determining B_par from RM:")
    print(f"    RM = const * integral n_e * B_par * dl")
    print(f"    RM alone gives the product n_e * B_par * L")
    print(f"    Need independent n_e measurement to get B_par:")
    print(f"    Option 1: Interferometry -> measures integral n_e * dl")
    print(f"    Option 2: Thomson scattering -> local n_e")
    print(f"    Then: B_par = RM / (const * integral n_e dl)")
    print(f"    This is commonly used in: astrophysics (pulsars, galaxy clusters),")
    print(f"    and tokamak diagnostics (polarimetry)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")

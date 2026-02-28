"""
Exercises for Lesson 17: Spectroscopy
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.special import voigt_profile


def exercise_1():
    """
    Exercise 1: Line Profile Analysis
    Calculate and compare Doppler, Lorentzian, and Voigt profiles
    for the hydrogen Balmer-alpha line at given conditions.
    """
    c = 3.0e8
    k_B = 1.381e-23
    m_H = 1.673e-27   # Hydrogen atom mass (kg)

    lam_0 = 656.28e-9  # H-alpha wavelength (m)
    nu_0 = c / lam_0
    T = 10000           # Temperature (K)
    n_e = 1e21          # Electron density (/m^3)
    delta_lam_P = 0.02e-9  # Pressure broadening width (m)

    print("Line Profile Analysis: H-alpha")
    print(f"Wavelength: {lam_0*1e9:.2f} nm")
    print(f"Temperature: T = {T} K")
    print(f"Electron density: n_e = {n_e:.0e} /m^3")

    # (a) Doppler width
    # delta_nu_D = (nu_0/c) * sqrt(2*k_B*T/m)
    v_th = np.sqrt(2 * k_B * T / m_H)
    delta_nu_D = nu_0 * v_th / c  # FWHM in frequency
    delta_lam_D = lam_0 * v_th / c  # FWHM in wavelength

    print(f"\n(a) Doppler broadening:")
    print(f"  Thermal velocity: v_th = {v_th:.0f} m/s")
    print(f"  FWHM (frequency): delta_nu_D = {delta_nu_D/1e9:.4f} GHz")
    print(f"  FWHM (wavelength): delta_lam_D = {delta_lam_D*1e12:.2f} pm")

    # (b) Lorentzian (pressure broadening)
    delta_nu_P = c * delta_lam_P / lam_0**2
    print(f"\n(b) Pressure (Lorentzian) broadening:")
    print(f"  FWHM (wavelength): delta_lam_P = {delta_lam_P*1e12:.1f} pm")
    print(f"  FWHM (frequency): delta_nu_P = {delta_nu_P/1e9:.4f} GHz")

    # (c) Voigt profile comparison
    # Voigt parameter: a = delta_nu_L / delta_nu_D (ratio of Lorentzian to Gaussian)
    # Gaussian sigma (not FWHM): sigma_G = delta_nu_D / (2*sqrt(2*ln2))
    sigma_G = delta_nu_D / (2 * np.sqrt(2 * np.log(2)))
    gamma_L = delta_nu_P / 2  # Half-width at half-maximum

    print(f"\n(c) Voigt profile comparison:")
    print(f"  Gaussian sigma: {sigma_G/1e9:.4f} GHz")
    print(f"  Lorentzian HWHM: {gamma_L/1e9:.4f} GHz")

    # Evaluate profiles at selected detunings
    delta_nu_range = np.linspace(-50e9, 50e9, 1000)
    delta_lam_range = delta_nu_range * lam_0**2 / c

    # Gaussian profile
    G = np.exp(-delta_nu_range**2 / (2 * sigma_G**2)) / (sigma_G * np.sqrt(2*np.pi))

    # Lorentzian profile
    L = (gamma_L / np.pi) / (delta_nu_range**2 + gamma_L**2)

    # Voigt profile using scipy
    V = voigt_profile(delta_nu_range, sigma_G, gamma_L)

    # Normalize all to peak = 1 for comparison
    G_norm = G / G.max()
    L_norm = L / L.max()
    V_norm = V / V.max()

    print(f"\n  Profile comparison (normalized to peak):")
    print(f"{'Detuning (GHz)':>16} {'Gaussian':>10} {'Lorentzian':>12} {'Voigt':>10}")
    print("-" * 50)

    for delta_nu_ghz in [0, 5, 10, 15, 20, 30, 40, 50]:
        idx = np.argmin(np.abs(delta_nu_range - delta_nu_ghz * 1e9))
        print(f"{delta_nu_ghz:>16} {G_norm[idx]:>10.4f} {L_norm[idx]:>12.4f} {V_norm[idx]:>10.4f}")

    # (d) Where Lorentzian wing dominates
    # Crossover: Gaussian falls faster than Lorentzian
    # At large detuning: G ~ exp(-x^2), L ~ 1/x^2
    # Crossover at roughly delta_nu ~ 2-3 * delta_nu_D
    print(f"\n(d) Lorentzian wings dominate at:")
    for delta_nu_ghz in np.arange(5, 51, 5):
        idx = np.argmin(np.abs(delta_nu_range - delta_nu_ghz * 1e9))
        ratio = L_norm[idx] / G_norm[idx] if G_norm[idx] > 1e-10 else np.inf
        if ratio > 1:
            print(f"  delta_nu > ~{delta_nu_ghz:.0f} GHz: L/G ratio = {ratio:.1f}")
            break

    print(f"  At {delta_nu_ghz} GHz detuning: Lorentzian dominates")
    print(f"  This is approximately {delta_nu_ghz*1e9/delta_nu_D:.1f} Doppler widths")


def exercise_2():
    """
    Exercise 2: Czerny-Turner Grating Spectrometer Design
    Design a spectrometer to resolve the sodium D doublet.
    """
    # Sodium D doublet
    lam1 = 589.0e-9  # D1 line (m)
    lam2 = 589.6e-9  # D2 line (m)
    delta_lam = abs(lam2 - lam1)
    lam_avg = (lam1 + lam2) / 2

    print("Czerny-Turner Spectrometer Design:")
    print(f"Na D1: {lam1*1e9:.1f} nm")
    print(f"Na D2: {lam2*1e9:.1f} nm")
    print(f"Separation: {delta_lam*1e9:.1f} nm")

    # (a) Required resolving power
    R_needed = lam_avg / delta_lam
    print(f"\n(a) Required resolving power: R = lambda/delta_lambda = {R_needed:.0f}")

    # (b) Grating parameters
    N_grooves_mm = 1200  # grooves/mm
    m = 1                # First diffraction order
    d = 1e-3 / N_grooves_mm  # Grating period (m)

    # R = m*N, where N is total illuminated grooves
    N_grooves_needed = R_needed / m
    W_needed = N_grooves_needed * d

    print(f"\n(b) Grating: {N_grooves_mm} grooves/mm, order m = {m}")
    print(f"  Grating period: d = {d*1e6:.3f} um")
    print(f"  Minimum grooves needed: N = R/m = {N_grooves_needed:.0f}")
    print(f"  Minimum illuminated width: W = {W_needed*1e3:.1f} mm")

    # Diffraction angle at sodium wavelength
    theta_i = np.radians(30)  # Incidence angle (typical Littrow-like)
    sin_theta_m = m * lam_avg / d - np.sin(theta_i)  # No, wrong sign

    # Grating equation: d*(sin(alpha) + sin(beta)) = m*lambda
    # For Czerny-Turner near-Littrow: alpha ~ beta
    # sin(alpha) + sin(beta) = m*lambda/d
    alpha = np.radians(20)  # Incidence angle
    sin_beta = m * lam_avg / d - np.sin(alpha)
    if abs(sin_beta) <= 1:
        beta = np.arcsin(sin_beta)
        print(f"\n  Incidence angle: alpha = {np.degrees(alpha):.1f} deg")
        print(f"  Diffraction angle: beta = {np.degrees(beta):.2f} deg")
    else:
        # Near-Littrow
        theta_lit = np.arcsin(m * lam_avg / (2 * d))
        print(f"  Littrow angle: theta = {np.degrees(theta_lit):.2f} deg")
        alpha = theta_lit
        beta = theta_lit

    # (c) Linear separation on detector
    f_cam = 0.500  # Camera focal length (m)

    # Angular dispersion: d(beta)/d(lambda) = m / (d * cos(beta))
    ang_disp = m / (d * np.cos(beta))  # rad/m
    linear_disp = f_cam * ang_disp      # m/m
    separation = linear_disp * delta_lam  # m

    print(f"\n(c) Camera focal length: f = {f_cam*100:.0f} cm")
    print(f"  Angular dispersion: {ang_disp:.2f} rad/m = {ang_disp/1e6*1e9:.4f} deg/nm")
    print(f"  Linear dispersion: {linear_disp*1e3:.2f} mm/nm")
    print(f"  D1-D2 separation on detector: {separation*1e3:.3f} mm")

    # Check with typical pixel size
    pixel = 13e-6  # 13 um pixels
    print(f"  With {pixel*1e6:.0f} um pixels: {separation/pixel:.1f} pixels between D lines")

    # (d) Free spectral range
    FSR_1 = lam_avg / m  # In first order
    FSR_2 = lam_avg / (m + 1)  # In second order

    print(f"\n(d) Free spectral range:")
    print(f"  First order (m=1):  FSR = {FSR_1*1e9:.1f} nm")
    print(f"  Second order (m=2): FSR = {FSR_2*1e9:.1f} nm")
    print(f"  Note: In m=2, resolving power doubles but FSR halves")
    print(f"  Must use order-sorting filter to avoid overlap")


def exercise_3():
    """
    Exercise 3: Fabry-Perot Spectroscopy
    Analyze a Fabry-Perot etalon for high-resolution spectroscopy,
    including scanning by piezoelectric tuning.
    """
    c = 3.0e8
    d = 10e-3       # Mirror separation (m)
    R = 0.97        # Mirror reflectance
    n = 1.0         # Air gap
    lam_0 = 500e-9  # Design wavelength (m)
    nu_0 = c / lam_0

    print("Fabry-Perot Etalon Analysis:")
    print(f"Mirror separation: d = {d*1e3:.0f} mm")
    print(f"Reflectance: R = {R}")
    print(f"Wavelength: {lam_0*1e9:.0f} nm")

    # (a) Finesse, FSR, and resolving power
    F = np.pi * np.sqrt(R) / (1 - R)
    F_coeff = 4 * R / (1 - R)**2

    FSR_nu = c / (2 * n * d)  # Hz
    FSR_lam = lam_0**2 / (2 * n * d)  # m

    m = int(2 * n * d / lam_0)  # Order number
    R_power = m * F  # Resolving power
    delta_lam = lam_0 / R_power

    print(f"\n(a) Key parameters:")
    print(f"  Finesse: F = {F:.1f}")
    print(f"  FSR (frequency): {FSR_nu/1e9:.4f} GHz")
    print(f"  FSR (wavelength): {FSR_lam*1e12:.4f} pm")
    print(f"  Order number: m = {m}")
    print(f"  Resolving power: R = m*F = {R_power:.0f}")
    print(f"  Resolution: delta_lambda = {delta_lam*1e15:.2f} fm")

    # FWHM of transmission peak
    FWHM_nu = FSR_nu / F
    print(f"  Transmission peak FWHM: {FWHM_nu/1e6:.2f} MHz")

    # (b) Transmission function
    print(f"\n(b) Transmission around 500 nm:")
    print(f"{'Detuning (pm)':>16} {'T':>10}")
    print("-" * 28)

    lam_scan = np.linspace(lam_0 - 0.5e-9, lam_0 + 0.5e-9, 1000)
    for lam in np.linspace(lam_0 - 0.5e-9, lam_0 + 0.5e-9, 11):
        delta = 2 * np.pi * n * d / lam
        T = 1.0 / (1 + F_coeff * np.sin(delta)**2)
        detuning_pm = (lam - lam_0) * 1e12
        print(f"{detuning_pm:>16.2f} {T:>10.6f}")

    # (c) Number of peaks in 1 nm range
    n_peaks = int(1e-9 / FSR_lam) + 1
    print(f"\n(c) Number of transmission peaks in 1 nm range:")
    print(f"  FSR = {FSR_lam*1e12:.4f} pm")
    print(f"  Range = 1000 pm")
    print(f"  Number of peaks: {n_peaks}")

    # (d) Piezo scanning
    delta_d = 250e-9  # Gap change (m)

    # Wavelength shift: delta_lambda = lambda * delta_d / d
    delta_lam_scan = lam_0 * delta_d / d

    # Equivalently: delta_nu = nu * delta_d / d = c * delta_d / (lambda * d)
    delta_nu_scan = c * delta_d / (lam_0 * d)

    print(f"\n(d) Piezo scanning (delta_d = {delta_d*1e9:.0f} nm):")
    print(f"  Wavelength scan range: {delta_lam_scan*1e12:.4f} pm")
    print(f"  Frequency scan range: {delta_nu_scan/1e9:.4f} GHz")
    print(f"  Scan range / FSR: {delta_lam_scan / FSR_lam:.2f}")
    print(f"  (Need delta_d = lambda/2 = {lam_0/2*1e9:.0f} nm to scan one FSR)")

    # Pre-filter requirement
    print(f"\n  Practical note: Fabry-Perot must be used with a pre-filter")
    print(f"  (grating or interference filter) to select a single FSR")
    print(f"  within which the etalon provides ultra-high resolution")


def exercise_4():
    """
    Exercise 4: Quantitative Absorption (Beer-Lambert Law)
    Apply Beer-Lambert law to analyze KMnO4 solutions and discuss
    the limits of absorption measurements.
    """
    # KMnO4 at 525 nm
    epsilon = 2455    # Molar absorption coefficient (L/(mol*cm))
    lam = 525e-9      # Peak wavelength (m)
    path = 1.0        # Cuvette path length (cm)

    print("Beer-Lambert Law: KMnO4 Analysis")
    print(f"Molar absorption coefficient: epsilon = {epsilon} L/(mol*cm)")
    print(f"Path length: l = {path} cm")
    print(f"Wavelength: {lam*1e9:.0f} nm")

    # (a) Absorbance and transmittance vs concentration
    concentrations_mmol = np.array([0.01, 0.02, 0.05, 0.1, 0.2])  # mmol/L
    concentrations_mol = concentrations_mmol * 1e-3  # mol/L

    print(f"\n(a) Absorbance and Transmittance:")
    print(f"{'c (mmol/L)':>12} {'A':>10} {'T':>10} {'T (%)':>10}")
    print("-" * 44)

    for c_mmol, c_mol in zip(concentrations_mmol, concentrations_mol):
        A = epsilon * c_mol * path
        T = 10**(-A)
        print(f"{c_mmol:>12.3f} {A:>10.4f} {T:>10.4f} {T*100:>10.2f}")

    # (b) Unknown concentration from transmittance
    T_measured = 0.35
    A_measured = -np.log10(T_measured)
    c_unknown = A_measured / (epsilon * path)  # mol/L

    print(f"\n(b) Unknown sample:")
    print(f"  Measured transmittance: T = {T_measured}")
    print(f"  Absorbance: A = -log10(T) = {A_measured:.4f}")
    print(f"  Concentration: c = A/(epsilon*l) = {c_unknown*1e3:.4f} mmol/L")
    print(f"  = {c_unknown*1e6:.2f} umol/L")

    # (c) Concentration at A = 2
    A_target = 2.0
    c_A2 = A_target / (epsilon * path)
    T_A2 = 10**(-A_target)

    print(f"\n(c) At A = {A_target}:")
    print(f"  Concentration: c = {c_A2*1e3:.4f} mmol/L")
    print(f"  Transmittance: T = 10^(-{A_target}) = {T_A2:.4f} = {T_A2*100:.2f}%")

    # (d) Accuracy limits
    print(f"\n(d) Accuracy of absorbance measurements:")
    print(f"  The uncertainty in concentration depends on the absorbance value.")
    print(f"  For a photometric accuracy of delta_T = 0.005:")

    delta_T = 0.005  # Typical photometric accuracy
    print(f"\n{'A':>8} {'T':>8} {'delta_A':>10} {'delta_c/c (%)':>14}")
    print("-" * 42)

    for A_test in [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        T_test = 10**(-A_test)
        # delta_A = delta_T / (T * ln(10))
        if T_test > delta_T:
            delta_A = delta_T / (T_test * np.log(10))
            relative_error = delta_A / A_test * 100
            print(f"{A_test:>8.2f} {T_test:>8.4f} {delta_A:>10.4f} {relative_error:>14.2f}")
        else:
            print(f"{A_test:>8.2f} {T_test:>8.4f} {'---':>10} {'too high':>14}")

    # Optimal absorbance range
    # Minimum error when d(delta_c/c)/dA = 0
    # This gives A_optimal = 1/ln(10) ~ 0.434
    A_optimal = 1 / np.log(10)
    print(f"\n  Optimal absorbance: A = 1/ln(10) = {A_optimal:.3f}")
    print(f"  (Minimum relative error in concentration)")
    print(f"  Practical range: A = 0.1 to 1.5 (good accuracy)")
    print(f"  A < 0.01: noise dominates (low signal)")
    print(f"  A > 2: stray light causes deviations from Beer's law")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Line Profile Analysis", exercise_1),
        ("Exercise 2: Grating Spectrometer Design", exercise_2),
        ("Exercise 3: Fabry-Perot Spectroscopy", exercise_3),
        ("Exercise 4: Quantitative Absorption Analysis", exercise_4),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

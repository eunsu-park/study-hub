"""
Exercises for Lesson 05: Wave Optics - Interference
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Young's Double-Slit Experiment
    Calculate fringe spacing, angular positions of maxima, and
    the effect of changing slit parameters.
    """
    lam = 632.8e-9   # He-Ne laser wavelength (m)
    d = 0.25e-3      # Slit separation (m)
    L = 1.5           # Screen distance (m)

    # Fringe spacing
    delta_y = lam * L / d
    print("Young's Double-Slit Experiment:")
    print(f"Wavelength: {lam*1e9:.1f} nm")
    print(f"Slit separation: d = {d*1e3:.2f} mm")
    print(f"Screen distance: L = {L:.1f} m")
    print(f"Fringe spacing: dy = {delta_y*1e3:.3f} mm")

    # Angular positions of first 5 maxima
    print(f"\nMaxima positions:")
    print(f"{'m':>4} {'theta (mrad)':>14} {'y (mm)':>10}")
    print("-" * 30)
    for m in range(6):
        theta = np.arcsin(m * lam / d) if m * lam / d <= 1 else None
        if theta is not None:
            y = L * np.tan(theta)
            print(f"{m:>4} {theta*1e3:>14.4f} {y*1e3:>10.3f}")

    # Effect of slit width on envelope
    a = 0.05e-3  # Slit width (m)
    print(f"\nSlit width: a = {a*1e3:.2f} mm")
    # First zero of single-slit envelope
    theta_zero = lam / a
    m_missing = int(d / a)
    print(f"First single-slit minimum at theta = {theta_zero*1e3:.4f} mrad")
    print(f"Missing order: m = d/a = {d/a:.0f}")

    # Intensity pattern
    y = np.linspace(-10e-3, 10e-3, 1000)
    beta = np.pi * d * y / (lam * L)
    alpha = np.pi * a * y / (lam * L)
    # Avoid division by zero
    sinc_alpha = np.where(np.abs(alpha) < 1e-10, 1.0, np.sin(alpha)/alpha)
    I = (np.cos(beta))**2 * sinc_alpha**2

    # Count visible fringes within central diffraction envelope
    n_fringes_central = 2 * m_missing - 1
    print(f"Number of bright fringes in central envelope: {n_fringes_central}")


def exercise_2():
    """
    Exercise 2: Anti-Reflection Coating Design
    Design a single-layer AR coating for glass and calculate the
    reflectance spectrum.
    """
    # Quarter-wave AR coating: n_coating = sqrt(n_substrate * n_air)
    n_air = 1.0
    n_glass = 1.52  # BK7 glass

    # Ideal coating index
    n_ideal = np.sqrt(n_air * n_glass)
    print("Single-Layer AR Coating Design:")
    print(f"Substrate: n = {n_glass}")
    print(f"Ideal coating index: n_c = sqrt({n_air}*{n_glass}) = {n_ideal:.4f}")

    # Use MgF2 (closest common material)
    n_MgF2 = 1.38
    print(f"MgF2 index: n = {n_MgF2}")

    # Design wavelength
    lam_0 = 550e-9  # nm (center of visible)
    # Quarter-wave thickness
    t = lam_0 / (4 * n_MgF2)
    print(f"\nDesign wavelength: {lam_0*1e9:.0f} nm")
    print(f"Coating thickness: t = lambda/(4n) = {t*1e9:.2f} nm")

    # Reflectance at design wavelength
    # R = ((n_air*n_glass - n_c^2) / (n_air*n_glass + n_c^2))^2
    R_coated = ((n_air * n_glass - n_MgF2**2) / (n_air * n_glass + n_MgF2**2))**2
    R_bare = ((n_glass - n_air) / (n_glass + n_air))**2

    print(f"\nBare glass reflectance: R = {R_bare*100:.2f}%")
    print(f"Coated reflectance at {lam_0*1e9:.0f} nm: R = {R_coated*100:.4f}%")
    print(f"Reduction factor: {R_bare/R_coated:.0f}x")

    # Reflectance spectrum
    wavelengths = np.linspace(400e-9, 800e-9, 200)
    print(f"\nReflectance spectrum:")
    print(f"{'Lambda (nm)':>12} {'R (%)':>10}")
    print("-" * 24)

    for lam in [400e-9, 450e-9, 500e-9, 550e-9, 600e-9, 650e-9, 700e-9, 750e-9, 800e-9]:
        # Phase thickness
        delta = 2 * np.pi * n_MgF2 * t / lam
        # Transfer matrix method for single layer
        r12 = (n_air - n_MgF2) / (n_air + n_MgF2)
        r23 = (n_MgF2 - n_glass) / (n_MgF2 + n_glass)
        # Total reflectance (Airy formula)
        r_total = (r12 + r23 * np.exp(-2j * delta)) / (1 + r12 * r23 * np.exp(-2j * delta))
        R = np.abs(r_total)**2
        print(f"{lam*1e9:>12.0f} {R*100:>10.4f}")

    # With ideal coating
    print(f"\nWith ideal coating (n={n_ideal:.4f}):")
    R_ideal = ((n_air * n_glass - n_ideal**2) / (n_air * n_glass + n_ideal**2))**2
    print(f"Reflectance at design wavelength: {R_ideal:.2e} (essentially zero)")


def exercise_3():
    """
    Exercise 3: Michelson Interferometer
    Analyze a Michelson interferometer observing the sodium doublet,
    including fringe visibility as a function of path difference.
    """
    # Sodium doublet
    lam1 = 589.0e-9   # D1 line (m)
    lam2 = 589.6e-9   # D2 line (m)
    lam_avg = (lam1 + lam2) / 2
    delta_lam = abs(lam2 - lam1)

    print("Michelson Interferometer with Sodium Doublet:")
    print(f"D1 line: {lam1*1e9:.1f} nm")
    print(f"D2 line: {lam2*1e9:.1f} nm")
    print(f"Average: {lam_avg*1e9:.1f} nm")
    print(f"Separation: {delta_lam*1e9:.1f} nm")

    # Beat length: path difference where fringes disappear
    # Visibility = |cos(pi * delta_d * delta_nu / c)|
    # Fringes disappear when delta_d = lambda^2 / (2 * delta_lambda)
    L_beat = lam_avg**2 / (2 * delta_lam)
    print(f"\nBeat half-period (fringe disappearance): {L_beat*1e3:.4f} mm")
    print(f"  = {L_beat/lam_avg:.0f} wavelengths")

    # Fringe visibility vs path difference
    print(f"\nFringe visibility vs mirror displacement:")
    print(f"{'Displacement (mm)':>20} {'Visibility':>12}")
    print("-" * 34)

    displacements = np.linspace(0, 1.0e-3, 11)  # 0 to 1 mm
    for d in displacements:
        # Each line produces: I_i = I_0 * (1 + cos(2*pi*d/lam_i * 2))
        # OPD = 2*d (round trip)
        opd = 2 * d
        delta_phi = 2 * np.pi * opd * delta_lam / lam_avg**2
        V = abs(np.cos(delta_phi / 2))
        print(f"{d*1e3:>20.4f} {V:>12.4f}")

    # Coherence length
    L_coh = lam_avg**2 / delta_lam
    print(f"\nCoherence length: {L_coh*1e3:.4f} mm")
    print(f"  (Visibility drops significantly beyond this)")


def exercise_4():
    """
    Exercise 4: Fabry-Perot Etalon
    Calculate the finesse, FSR, and resolving power of a Fabry-Perot
    etalon, and determine its spectral analysis capabilities.
    """
    # Etalon parameters
    d = 10.0e-3    # Spacing (m)
    R = 0.95       # Mirror reflectance
    n = 1.0        # Air gap
    lam = 550e-9   # Wavelength (m)
    c = 3.0e8      # Speed of light (m/s)

    # Finesse
    F = np.pi * np.sqrt(R) / (1 - R)
    print("Fabry-Perot Etalon:")
    print(f"Mirror spacing: d = {d*1e3:.1f} mm")
    print(f"Reflectance: R = {R}")
    print(f"Finesse: F = {F:.1f}")

    # Coefficient of finesse
    F_coeff = 4 * R / (1 - R)**2
    print(f"Coefficient of finesse: F_c = {F_coeff:.0f}")

    # Free Spectral Range
    FSR_freq = c / (2 * n * d)         # Hz
    FSR_lam = lam**2 / (2 * n * d)     # m
    print(f"\nFree Spectral Range:")
    print(f"  FSR (frequency) = {FSR_freq/1e9:.3f} GHz")
    print(f"  FSR (wavelength) = {FSR_lam*1e12:.4f} pm")

    # Resolving power
    m = int(2 * n * d / lam)  # Order number
    resolving_power = m * F
    delta_lam_min = lam / resolving_power

    print(f"\nOrder number: m = {m}")
    print(f"Resolving power: R = m*F = {resolving_power:.0f}")
    print(f"Minimum resolvable wavelength: {delta_lam_min*1e15:.2f} fm")

    # Transmission function (Airy function)
    print(f"\nTransmission at selected detunings:")
    print(f"{'Phase offset (rad)':>20} {'Transmission':>14}")
    print("-" * 36)

    for delta_phase_deg in [0, 1, 2, 5, 10, 30, 45, 90, 180]:
        delta = np.radians(delta_phase_deg)
        T = 1.0 / (1 + F_coeff * np.sin(delta/2)**2)
        print(f"{delta_phase_deg:>20} deg {T:>12.6f}")

    # FWHM of transmission peak
    FWHM_phase = 2 * np.arcsin(1 / np.sqrt(F_coeff))
    FWHM_freq = FSR_freq / F
    print(f"\nFWHM of transmission peak: {FWHM_freq/1e6:.2f} MHz")


def exercise_5():
    """
    Exercise 5: Coherence and Fringe Visibility
    Calculate temporal and spatial coherence for different light sources
    and predict fringe visibility.
    """
    c = 3.0e8

    print("Coherence Properties of Light Sources:")
    print(f"{'Source':>25} {'dlam (nm)':>10} {'L_coh (mm)':>12} {'tau_coh (ps)':>14}")
    print("-" * 63)

    sources = [
        ("White light", 550e-9, 300e-9),
        ("LED (green)", 525e-9, 30e-9),
        ("Na lamp (filtered)", 589e-9, 0.6e-9),
        ("He-Ne laser", 632.8e-9, 0.002e-9),
        ("Single-freq laser", 1064e-9, 1e-15),
    ]

    for name, lam, dlam in sources:
        L_coh = lam**2 / dlam
        tau_coh = L_coh / c
        print(f"{name:>25} {dlam*1e9:>10.4f} {L_coh*1e3:>12.4f} {tau_coh*1e12:>14.4f}")

    # Spatial coherence
    print("\n\nSpatial Coherence (van Cittert-Zernike theorem):")
    print("For a circular incoherent source of angular diameter alpha,")
    print("the coherence radius is: r_coh = 0.61 * lambda / alpha")

    alpha_sun = 0.53 * np.pi / 180  # Sun's angular diameter (rad)
    lam_vis = 550e-9
    r_coh_sun = 0.61 * lam_vis / (alpha_sun / 2)
    print(f"\nSun (alpha = 0.53 deg):")
    print(f"  Spatial coherence radius: {r_coh_sun*1e6:.1f} um")
    print(f"  (This is why sunlight can produce fringes in a double-slit")
    print(f"   experiment only when the slits are closer than ~{r_coh_sun*1e6:.0f} um)")

    # Fringe visibility with partial coherence
    print("\n\nFringe Visibility vs. Path Difference (He-Ne laser):")
    lam_HeNe = 632.8e-9
    dlam_HeNe = 0.002e-9  # ~1 GHz linewidth
    L_coh_HeNe = lam_HeNe**2 / dlam_HeNe

    print(f"Coherence length: {L_coh_HeNe:.2f} m")
    print(f"{'OPD (cm)':>12} {'Visibility':>12}")
    print("-" * 26)

    for opd_cm in [0, 1, 5, 10, 20, 50, 100, 200]:
        opd = opd_cm * 0.01  # m
        # Visibility envelope: V = exp(-pi*(opd/L_coh)^2) for Gaussian linewidth
        V = np.exp(-np.pi * (opd / L_coh_HeNe)**2)
        print(f"{opd_cm:>12} {V:>12.6f}")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Young's Double-Slit Experiment", exercise_1),
        ("Exercise 2: Anti-Reflection Coating Design", exercise_2),
        ("Exercise 3: Michelson Interferometer", exercise_3),
        ("Exercise 4: Fabry-Perot Etalon", exercise_4),
        ("Exercise 5: Coherence and Fringe Visibility", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

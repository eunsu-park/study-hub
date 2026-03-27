"""
Exercises for Lesson 06: Diffraction
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Single-Slit Diffraction
    Calculate the diffraction pattern from a single slit, including
    positions of minima and relative intensities of secondary maxima.
    """
    lam = 600e-9    # Wavelength (m)
    a = 0.10e-3     # Slit width (m)
    L = 2.0          # Screen distance (m)

    print("Single-Slit Diffraction:")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"Slit width: a = {a*1e3:.2f} mm")
    print(f"Screen distance: L = {L:.1f} m")

    # Positions of minima: a*sin(theta) = m*lambda
    print(f"\nMinima positions:")
    print(f"{'m':>4} {'theta (mrad)':>14} {'y (mm)':>10}")
    print("-" * 30)
    for m in range(1, 6):
        sin_theta = m * lam / a
        if sin_theta > 1:
            break
        theta = np.arcsin(sin_theta)
        y = L * np.tan(theta)
        print(f"{m:>4} {theta*1e3:>14.4f} {y*1e3:>10.3f}")

    # Relative intensities of secondary maxima
    # Maxima occur approximately at beta = (m + 0.5)*pi for m >= 1
    print(f"\nSecondary maxima (relative to central peak):")
    print(f"{'m':>4} {'beta/pi':>10} {'I/I_0':>12} {'I/I_0 (dB)':>12}")
    print("-" * 40)
    for m in range(1, 6):
        beta = (m + 0.5) * np.pi
        I_rel = (np.sin(beta) / beta)**2
        I_dB = 10 * np.log10(I_rel)
        print(f"{m:>4} {beta/np.pi:>10.1f} {I_rel:>12.6f} {I_dB:>12.2f}")

    # Central maximum width
    central_width = 2 * lam * L / a
    print(f"\nCentral maximum full width: {central_width*1e3:.3f} mm")
    print(f"Secondary maxima width: {lam*L/a*1e3:.3f} mm (half of central)")


def exercise_2():
    """
    Exercise 2: Airy Disk and HST Resolution
    Calculate the Airy disk size and angular resolution for the
    Hubble Space Telescope.
    """
    D_HST = 2.4       # HST primary mirror diameter (m)
    lam = 550e-9       # Visible wavelength (m)
    lam_NIR = 1.6e-6   # Near-IR wavelength (m)

    print("Hubble Space Telescope Resolution:")
    print(f"Primary mirror diameter: D = {D_HST:.1f} m")

    # Rayleigh criterion: theta = 1.22 * lambda / D
    theta_vis = 1.22 * lam / D_HST
    theta_NIR = 1.22 * lam_NIR / D_HST

    print(f"\nAt lambda = {lam*1e9:.0f} nm (visible):")
    print(f"  Angular resolution: {theta_vis*1e6:.4f} urad = {np.degrees(theta_vis)*3600:.4f} arcsec")
    print(f"  = {np.degrees(theta_vis)*3600*1000:.2f} milliarcsec")

    print(f"\nAt lambda = {lam_NIR*1e6:.1f} um (near-IR):")
    print(f"  Angular resolution: {theta_NIR*1e6:.4f} urad = {np.degrees(theta_NIR)*3600:.4f} arcsec")

    # Airy disk on focal plane
    f_HST = 57.6  # HST effective focal length (m) f/24 system
    r_airy_vis = 1.22 * lam * f_HST / D_HST
    print(f"\nHST effective focal length: f = {f_HST} m (f/{f_HST/D_HST:.0f})")
    print(f"Airy disk radius on focal plane: {r_airy_vis*1e6:.2f} um")
    print(f"  (HST WFC3 pixel size: 40 um, so Airy disk ~ {r_airy_vis*1e6/40:.2f} pixels)")

    # Encircled energy
    print(f"\nEncircled energy for Airy pattern:")
    print(f"  Within first dark ring (r = 1.22 lam/D): ~84%")
    print(f"  Within second dark ring: ~91%")
    print(f"  Within third dark ring: ~94%")

    # Compare with ground-based telescope
    seeing = 1.0  # arcsec typical seeing
    D_ground = 8.0  # 8m telescope
    theta_ground_diff = 1.22 * lam / D_ground
    print(f"\nComparison with {D_ground:.0f}m ground telescope:")
    print(f"  Diffraction limit: {np.degrees(theta_ground_diff)*3600:.4f} arcsec")
    print(f"  Atmospheric seeing: {seeing} arcsec")
    print(f"  Seeing/diffraction = {seeing / (np.degrees(theta_ground_diff)*3600):.0f}x")
    print(f"  => Ground resolution limited by atmosphere, not optics")


def exercise_3():
    """
    Exercise 3: Diffraction Grating Spectroscopy
    Analyze a diffraction grating for spectroscopic applications,
    including resolving power and angular dispersion.
    """
    N = 600         # Grooves per mm
    W = 50e-3       # Grating width (m)
    lam = 550e-9    # Wavelength (m)

    d = 1.0 / (N * 1000)  # Grating period (m)
    N_total = int(W / d)    # Total number of grooves

    print("Diffraction Grating Analysis:")
    print(f"Groove density: {N} grooves/mm")
    print(f"Grating width: {W*1e3:.0f} mm")
    print(f"Grating period: d = {d*1e6:.4f} um")
    print(f"Total grooves: N = {N_total}")

    # Diffraction maxima: d*sin(theta) = m*lambda
    print(f"\nDiffraction orders at lambda = {lam*1e9:.0f} nm:")
    print(f"{'m':>4} {'sin(theta)':>12} {'theta (deg)':>14}")
    print("-" * 32)
    m_max = int(d / lam)
    for m in range(-m_max, m_max + 1):
        sin_theta = m * lam / d
        if abs(sin_theta) <= 1:
            theta = np.degrees(np.arcsin(sin_theta))
            print(f"{m:>4} {sin_theta:>12.6f} {theta:>14.2f}")

    # Resolving power
    print(f"\nResolving power:")
    for m in range(1, 4):
        R = m * N_total
        delta_lam = lam / R
        print(f"  Order {m}: R = {R} => delta_lambda = {delta_lam*1e12:.3f} pm")

    # Angular dispersion
    m_order = 1
    theta_m = np.arcsin(m_order * lam / d)
    angular_disp = m_order / (d * np.cos(theta_m))  # rad/m
    print(f"\nAngular dispersion (m={m_order}):")
    print(f"  dtheta/dlambda = {angular_disp:.2f} rad/m = {angular_disp/1e6:.4f} deg/nm")

    # Linear dispersion on focal plane
    f_camera = 0.5  # Camera focal length (m)
    linear_disp = f_camera * angular_disp  # m/m
    print(f"  Linear dispersion (f={f_camera*100:.0f} cm): {linear_disp*1e3:.2f} mm/nm")

    # Resolving the Na doublet
    dlam_Na = 0.6e-9  # Na D doublet separation (m)
    R_needed = 589e-9 / dlam_Na
    N_needed = R_needed / m_order
    W_needed = N_needed * d
    print(f"\nTo resolve Na doublet ({dlam_Na*1e9:.1f} nm separation):")
    print(f"  Required R = {R_needed:.0f}")
    print(f"  Need N = {N_needed:.0f} grooves in order {m_order}")
    print(f"  Grating width >= {W_needed*1e3:.1f} mm")


def exercise_4():
    """
    Exercise 4: Fresnel Zone Plate Design
    Design a Fresnel zone plate as a diffractive lens and calculate
    its focal length and efficiency.
    """
    lam = 550e-9    # Wavelength (m)
    f = 0.50        # Desired focal length (m)

    print("Fresnel Zone Plate Design:")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"Primary focal length: f = {f*100:.0f} cm")

    # Zone radii: r_n = sqrt(n * lambda * f)
    print(f"\nZone radii (first 20 zones):")
    print(f"{'n':>4} {'r_n (mm)':>10} {'width (um)':>12}")
    print("-" * 28)

    r_prev = 0
    for n_zone in range(1, 21):
        r_n = np.sqrt(n_zone * lam * f)
        width = r_n - r_prev
        print(f"{n_zone:>4} {r_n*1e3:>10.4f} {width*1e6:>12.2f}")
        r_prev = r_n

    # Total number of zones for given aperture
    R_plate = 5e-3  # 5 mm radius plate
    N_zones = int(R_plate**2 / (lam * f))
    print(f"\nFor plate radius = {R_plate*1e3:.0f} mm:")
    print(f"  Number of zones: N = {N_zones}")
    print(f"  Outermost zone width: {np.sqrt(N_zones*lam*f) - np.sqrt((N_zones-1)*lam*f):.2e} m")
    print(f"    = {(np.sqrt(N_zones*lam*f) - np.sqrt((N_zones-1)*lam*f))*1e6:.2f} um")

    # Numerical aperture and resolution
    NA = R_plate / np.sqrt(R_plate**2 + f**2)
    resolution = 0.61 * lam / NA
    print(f"\nNumerical aperture: NA = {NA:.4f}")
    print(f"Resolution limit: {resolution*1e6:.2f} um")

    # Multiple focal orders
    print(f"\nFocal orders of zone plate:")
    print(f"{'m':>4} {'f_m (cm)':>10} {'Efficiency':>12}")
    print("-" * 28)
    # Amplitude ZP: f_m = f/m (odd m only), efficiency ~ 1/(m*pi)^2
    for m in [1, -1, 3, -3, 5, -5]:
        f_m = f / m
        eff = 1.0 / (m * np.pi)**2 if m != 0 else 0
        if m > 0:
            print(f"{m:>4} {f_m*100:>10.2f} {eff*100:>11.2f}%")
        else:
            print(f"{m:>4} {f_m*100:>10.2f} {eff*100:>11.2f}% (virtual)")

    # Binary phase zone plate efficiency
    print(f"\nBinary amplitude ZP: max efficiency = {(1/np.pi**2)*100:.1f}% (m=+1)")
    print(f"Binary phase ZP: max efficiency = {(4/np.pi**2)*100:.1f}% (m=+1)")
    print(f"Blazed (multi-level): approaches 100%")


def exercise_5():
    """
    Exercise 5: Diffraction-Limited Photography
    Determine the optimal f-number for a camera before diffraction
    limits the resolution, accounting for pixel size.
    """
    pixel_sizes = [1.4e-6, 3.7e-6, 6.5e-6]  # Different sensor types
    labels = ["Smartphone (1.4 um)", "APS-C (3.7 um)", "Full-frame (6.5 um)"]
    lam = 550e-9  # Green light

    print("Diffraction-Limited Photography:")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"\nAiry disk diameter = 2.44 * lambda * N (N = f-number)")

    print(f"\n{'Sensor':>25} {'Pixel':>8} {'Optimal N':>10} {'Airy (um)':>10}")
    print("-" * 55)

    for pixel, label in zip(pixel_sizes, labels):
        # Airy disk diameter = 2.44 * lambda * N
        # Diffraction limit when Airy disk = 2 * pixel (Nyquist)
        N_optimal = 2 * pixel / (2.44 * lam)
        airy_at_optimal = 2.44 * lam * N_optimal

        print(f"{label:>25} {pixel*1e6:>7.1f} {N_optimal:>10.1f} "
              f"{airy_at_optimal*1e6:>10.2f}")

    # Detailed analysis for full-frame sensor
    pixel = 6.5e-6
    print(f"\nDetailed analysis: Full-frame sensor (pixel = {pixel*1e6:.1f} um)")
    print(f"{'f-number':>10} {'Airy (um)':>10} {'Airy/pixel':>12} {'Status':>20}")
    print("-" * 54)

    for N in [2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0]:
        airy = 2.44 * lam * N
        ratio = airy / pixel
        if ratio < 1:
            status = "Pixel-limited"
        elif ratio < 2:
            status = "Near-optimal"
        else:
            status = "Diffraction-limited"
        print(f"f/{N:<7.1f} {airy*1e6:>10.2f} {ratio:>12.2f} {status:>20}")

    # MTF analysis
    print(f"\nMTF at Nyquist frequency for full-frame sensor:")
    f_nyquist = 1 / (2 * pixel)  # cycles/m
    print(f"Nyquist frequency: {f_nyquist:.2e} cy/m = {f_nyquist/1e3:.0f} cy/mm")

    for N in [4.0, 8.0, 16.0]:
        f_cutoff = 1 / (lam * N)  # Diffraction cutoff
        rho = f_nyquist / f_cutoff
        if rho < 1:
            MTF = (2/np.pi) * (np.arccos(rho) - rho * np.sqrt(1 - rho**2))
        else:
            MTF = 0
        print(f"  f/{N:.0f}: MTF = {MTF:.3f} (f_c = {f_cutoff/1e3:.0f} cy/mm, rho = {rho:.3f})")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Single-Slit Diffraction", exercise_1),
        ("Exercise 2: Airy Disk and HST Resolution", exercise_2),
        ("Exercise 3: Diffraction Grating Spectroscopy", exercise_3),
        ("Exercise 4: Fresnel Zone Plate Design", exercise_4),
        ("Exercise 5: Diffraction-Limited Photography", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

"""
Exercises for Lesson 10: Fourier Optics
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Angular Spectrum Propagation
    Propagate a field using the angular spectrum method and compare
    the near-field and far-field diffraction patterns.
    """
    # Grid parameters
    N = 512
    L = 2.0e-3        # Physical size (m)
    dx = L / N
    lam = 633e-9       # He-Ne wavelength (m)
    k = 2 * np.pi / lam

    print("Angular Spectrum Propagation:")
    print(f"Grid: {N}x{N}, size = {L*1e3:.1f} mm, dx = {dx*1e6:.2f} um")
    print(f"Wavelength: {lam*1e9:.1f} nm")

    # Create coordinate arrays
    x = np.linspace(-L/2, L/2 - dx, N)
    fx = np.fft.fftfreq(N, dx)

    # Input field: rectangular aperture
    a = 100e-6  # Aperture half-width (m)
    U0 = np.zeros(N, dtype=complex)
    U0[np.abs(x) <= a] = 1.0

    print(f"\nInput: Rectangular aperture, width = {2*a*1e6:.0f} um")

    # Angular spectrum
    A0 = np.fft.fft(U0)

    # Propagation distances
    z_values = [0.5e-3, 1.0e-3, 5.0e-3, 10.0e-3, 50.0e-3]

    # Fresnel number: N_F = a^2 / (lambda * z)
    print(f"\nPropagation results:")
    print(f"{'z (mm)':>10} {'N_F':>8} {'Regime':>15} {'Central I/I0':>14}")
    print("-" * 49)

    for z in z_values:
        # Transfer function
        H = np.exp(1j * k * z * np.sqrt(1 - (lam * fx)**2 + 0j))
        # Evanescent waves (|fx| > 1/lambda): already handled by complex sqrt

        # Propagated field
        U_z = np.fft.ifft(A0 * H)
        I_z = np.abs(U_z)**2

        N_F = a**2 / (lam * z)
        if N_F > 1:
            regime = "Fresnel (near)"
        else:
            regime = "Fraunhofer (far)"

        central_I = I_z[N//2]
        print(f"{z*1e3:>10.2f} {N_F:>8.2f} {regime:>15} {central_I:>14.4f}")

    # Fraunhofer limit: the pattern approaches the Fourier transform
    z_far = 0.1  # 100 mm
    H_far = np.exp(1j * k * z_far * np.sqrt(1 - (lam * fx)**2 + 0j))
    U_far = np.fft.ifft(A0 * H_far)
    I_far = np.abs(U_far)**2

    # Analytical Fraunhofer: sinc^2(a*x/(lam*z))
    x_theory = x
    I_theory = np.sinc(a * x_theory / (lam * z_far))**2

    # Compare central lobe width
    # First zero at x = lam*z/a
    x_zero = lam * z_far / a
    print(f"\nFraunhofer limit (z = {z_far*1e3:.0f} mm):")
    print(f"  First zero (theory): x = {x_zero*1e6:.1f} um")
    print(f"  Central lobe width: {2*x_zero*1e6:.1f} um")


def exercise_2():
    """
    Exercise 2: 4f Optical System Design
    Design a 4f system for spatial filtering and compute the
    point spread function and optical transfer function.
    """
    f1 = 100e-3  # First lens focal length (m)
    f2 = 200e-3  # Second lens focal length (m)
    lam = 550e-9  # Wavelength (m)
    D = 25e-3     # Lens diameter (m)

    print("4f Optical System:")
    print(f"f1 = {f1*1e3:.0f} mm, f2 = {f2*1e3:.0f} mm")
    print(f"Magnification: M = -f2/f1 = {-f2/f1:.1f}")
    print(f"Total length: {2*(f1+f2)*1e3:.0f} mm")

    # Spatial frequency cutoff
    f_cutoff = D / (2 * lam * f1)  # cycles/m
    print(f"\nSpatial frequency cutoff: {f_cutoff:.0f} cy/m = {f_cutoff/1e3:.1f} cy/mm")

    # Resolution (Rayleigh criterion)
    resolution = 1.22 * lam * f1 / D
    print(f"Rayleigh resolution: {resolution*1e6:.2f} um")

    # PSF (Airy pattern)
    print(f"\nPoint Spread Function (Airy pattern):")
    print(f"  Airy disk radius: {1.22*lam*f1/D*1e6:.2f} um (at input plane)")
    print(f"  At output plane: {1.22*lam*f2/D*1e6:.2f} um (magnified by |M|)")

    # Spatial filtering examples
    print(f"\n--- Spatial Filtering Examples ---")

    # Low-pass filter (circular aperture in Fourier plane)
    print(f"\n1. Low-Pass Filter:")
    r_filter_values = [0.5e-3, 1.0e-3, 2.5e-3, 5.0e-3]
    for r_f in r_filter_values:
        f_pass = r_f / (lam * f1)
        feature_size = 1 / f_pass if f_pass > 0 else np.inf
        print(f"  Pinhole r = {r_f*1e3:.1f} mm: passes < {f_pass:.0f} cy/m "
              f"(features > {feature_size*1e6:.1f} um)")

    # High-pass filter (opaque dot in center)
    print(f"\n2. High-Pass Filter (edge enhancement):")
    r_block = 1.0e-3
    f_block = r_block / (lam * f1)
    print(f"  Block r = {r_block*1e3:.1f} mm: blocks < {f_block:.0f} cy/m")
    print(f"  Result: edges and fine details enhanced")

    # Band-pass filter (annular aperture)
    print(f"\n3. Band-Pass Filter (annular aperture):")
    r_inner, r_outer = 1.0e-3, 3.0e-3
    f_inner = r_inner / (lam * f1)
    f_outer = r_outer / (lam * f1)
    print(f"  Annulus {r_inner*1e3:.1f}-{r_outer*1e3:.1f} mm: "
          f"passes {f_inner:.0f}-{f_outer:.0f} cy/m")

    # Phase contrast (pi phase shift of zero-order)
    print(f"\n4. Phase Contrast (Zernike method):")
    print(f"  Small phase dot at DC: shifts zero-order by pi/2")
    print(f"  Converts phase variations to intensity contrast")
    print(f"  Critical for transparent biological specimens")


def exercise_3():
    """
    Exercise 3: Modulation Transfer Function (MTF) Analysis
    Calculate the MTF of a diffraction-limited optical system and
    compare with aberrated systems.
    """
    lam = 550e-9     # Wavelength (m)
    D = 50e-3        # Aperture diameter (m)
    f = 200e-3       # Focal length (m)

    print("MTF Analysis:")
    print(f"Aperture: D = {D*1e3:.0f} mm")
    print(f"Focal length: f = {f*1e3:.0f} mm")
    print(f"f-number: f/{f/D:.1f}")
    print(f"Wavelength: {lam*1e9:.0f} nm")

    # Diffraction cutoff frequency
    f_cutoff = D / (lam * f)  # cycles/m on focal plane
    # or equivalently: 1/(lambda*N) where N is f-number
    N_num = f / D
    f_cutoff_alt = 1 / (lam * N_num)

    print(f"\nCutoff frequency: {f_cutoff:.0f} cy/m = {f_cutoff/1e3:.0f} cy/mm")
    print(f"  = {f_cutoff*lam*f/D:.4f} (normalized)")

    # Diffraction-limited MTF for circular aperture
    print(f"\nDiffraction-Limited MTF:")
    print(f"{'f (cy/mm)':>12} {'f/f_c':>8} {'MTF':>10}")
    print("-" * 32)

    f_values_normalized = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for rho in f_values_normalized:
        if rho == 0:
            mtf = 1.0
        elif rho >= 1:
            mtf = 0.0
        else:
            # MTF for circular aperture: (2/pi)[arccos(rho) - rho*sqrt(1-rho^2)]
            mtf = (2/np.pi) * (np.arccos(rho) - rho * np.sqrt(1 - rho**2))
        f_real = rho * f_cutoff / 1e3  # cy/mm
        print(f"{f_real:>12.1f} {rho:>8.2f} {mtf:>10.4f}")

    # Effect of defocus on MTF
    print(f"\n--- Effect of Defocus (W_20 in waves) ---")
    print(f"{'W_20 (waves)':>14} {'MTF at f_c/2':>14} {'Strehl':>10}")
    print("-" * 40)

    rho_eval = 0.5  # Evaluate at half-cutoff
    for W20 in [0, 0.1, 0.25, 0.5, 0.7, 1.0]:
        # Approximate MTF reduction due to defocus
        # Strehl ~ exp(-(2*pi*W20/sqrt(12))^2) for small aberrations
        # More precisely: sigma_W = W20 / sqrt(3.5) for defocus (Zernike)
        sigma_W = W20 / np.sqrt(3.5)  # RMS for Zernike defocus
        strehl = np.exp(-(2*np.pi*sigma_W)**2)

        # MTF_aberrated ~ MTF_diff * strehl (approximate for low frequencies)
        mtf_diff = (2/np.pi) * (np.arccos(rho_eval) - rho_eval * np.sqrt(1 - rho_eval**2))
        mtf_aberr = mtf_diff * strehl
        print(f"{W20:>14.2f} {mtf_aberr:>14.4f} {strehl:>10.4f}")

    # Resolution criteria
    print(f"\n--- Resolution Criteria ---")
    print(f"Rayleigh criterion: {1.22*lam*f/D*1e6:.2f} um")
    print(f"Sparrow criterion: {0.95*lam*f/D*1e6:.2f} um")
    print(f"Dawes criterion: {1.02*lam*f/D*1e6:.2f} um")
    print(f"Nyquist pixel size: {1/(2*f_cutoff)*1e6:.2f} um")


def exercise_4():
    """
    Exercise 4: Phase Contrast Microscopy Simulation
    Simulate Zernike phase contrast imaging of a transparent
    biological specimen.
    """
    N = 256
    L = 200e-6   # Field of view (m)
    dx = L / N
    lam = 550e-9
    f = 10e-3    # Objective focal length (m)

    print("Phase Contrast Microscopy Simulation:")
    print(f"Grid: {N}x{N}, FOV = {L*1e6:.0f} um")
    print(f"Wavelength: {lam*1e9:.0f} nm")

    # Create a simple phase object (cell-like)
    x = np.linspace(-L/2, L/2 - dx, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    # Phase object: circular cell with organelles
    phi = np.zeros((N, N))
    # Cell body
    phi[R < 20e-6] = 0.3   # radians
    # Nucleus
    phi[R < 8e-6] = 0.6
    # Small organelle
    phi[np.sqrt((X-10e-6)**2 + (Y-5e-6)**2) < 3e-6] = 0.5

    # Field after object
    U_obj = np.exp(1j * phi)
    I_brightfield = np.abs(U_obj)**2

    print(f"\nPhase object statistics:")
    print(f"  Max phase: {phi.max():.2f} rad")
    print(f"  Min phase: {phi.min():.2f} rad")

    # Bright-field imaging (no filter)
    print(f"\nBright-field intensity:")
    print(f"  Max: {I_brightfield.max():.6f}")
    print(f"  Min: {I_brightfield.min():.6f}")
    print(f"  Contrast: {(I_brightfield.max()-I_brightfield.min())/(I_brightfield.max()+I_brightfield.min()):.6f}")
    print(f"  (Nearly zero contrast - phase objects are invisible!)")

    # Phase contrast: add pi/2 phase shift to zero-order
    A_obj = np.fft.fft2(U_obj)

    # Create phase plate: pi/2 shift at DC and nearby frequencies
    # Phase ring in Fourier plane
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    FR = np.sqrt(FX**2 + FY**2)

    # Phase ring parameters
    r_ring = 5e3  # Ring radius in freq space (cy/m)
    ring_width = 2e3
    phase_plate = np.ones((N, N), dtype=complex)
    ring_mask = np.abs(FR - r_ring) < ring_width/2
    # For simplicity, apply to DC region
    dc_mask = FR < 2e3  # Near DC
    phase_plate[dc_mask] = np.exp(1j * np.pi/2)  # Positive phase contrast

    # Apply phase plate
    A_pc = A_obj * phase_plate
    U_pc = np.fft.ifft2(A_pc)
    I_pc = np.abs(U_pc)**2

    print(f"\nPhase contrast imaging:")
    print(f"  Max intensity: {I_pc.max():.4f}")
    print(f"  Min intensity: {I_pc.min():.4f}")
    print(f"  Contrast: {(I_pc.max()-I_pc.min())/(I_pc.max()+I_pc.min()):.4f}")

    # Analytical estimate for weak phase objects
    # I_pc ~ 1 + 2*phi (for positive phase contrast, phi << 1)
    print(f"\n--- Analytical Approximation (weak phase) ---")
    print(f"For positive phase contrast: I ~ 1 + 2*phi")
    I_approx_max = 1 + 2 * phi.max()
    I_approx_min = 1 + 2 * phi.min()
    print(f"  Expected max: {I_approx_max:.4f}")
    print(f"  Expected min (background): {I_approx_min:.4f}")

    # Negative phase contrast
    print(f"\nNegative phase contrast: I ~ 1 - 2*phi")
    print(f"  Phase-dense objects appear dark on bright background")
    print(f"  (Standard in biological microscopy)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Angular Spectrum Propagation", exercise_1),
        ("Exercise 2: 4f Optical System Design", exercise_2),
        ("Exercise 3: MTF Analysis", exercise_3),
        ("Exercise 4: Phase Contrast Simulation", exercise_4),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

"""
Exercises for Lesson 11: Holography
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Hologram Resolution Requirements
    Calculate the spatial frequency requirements for recording a
    hologram and determine the required film resolution.
    """
    lam = 632.8e-9  # He-Ne laser wavelength (m)
    c = 3.0e8

    print("Hologram Recording Requirements:")
    print(f"Wavelength: {lam*1e9:.1f} nm")

    # Off-axis holography: maximum spatial frequency
    # f_max = 2 * sin(theta_max) / lambda
    # where theta_max is the maximum angle between object and reference beams

    print(f"\nMaximum fringe frequency vs. beam angle:")
    print(f"{'Angle (deg)':>12} {'f_max (cy/mm)':>16} {'Period (um)':>12}")
    print("-" * 42)

    for theta_deg in [5, 10, 15, 20, 30, 45, 60, 90]:
        theta = np.radians(theta_deg)
        f_max = 2 * np.sin(theta / 2) / lam  # For two beams at symmetric angles
        # More precisely for reference at theta, object at 0:
        f_max_asym = np.sin(theta) / lam
        period = 1 / f_max_asym if f_max_asym > 0 else np.inf
        print(f"{theta_deg:>12} {f_max_asym/1e3:>16.1f} {period*1e6:>12.3f}")

    # Recording materials
    print(f"\nRecording Material Resolution:")
    print(f"{'Material':>25} {'Resolution (cy/mm)':>20} {'Max angle (deg)':>16}")
    print("-" * 63)

    materials = [
        ("Silver halide film", 5000),
        ("Dichromated gelatin", 10000),
        ("Photopolymer", 6000),
        ("Photorefractive crystal", 10000),
        ("CCD sensor (3.5 um)", int(1/(2*3.5e-3))),  # Nyquist
    ]

    for name, resolution in materials:
        # Maximum recordable angle: sin(theta) = f * lambda
        sin_theta = resolution * 1e3 * lam  # resolution in cy/m
        if sin_theta <= 1:
            theta_max = np.degrees(np.arcsin(sin_theta))
        else:
            theta_max = 90  # Beyond limit
        print(f"{name:>25} {resolution:>20} {theta_max:>16.1f}")

    # Object size and depth requirements
    print(f"\n--- Object Depth and Coherence ---")
    # Coherence length must exceed max path difference
    delta_nu = 1e6  # 1 MHz linewidth (typical single-mode He-Ne)
    L_coh = c / delta_nu
    print(f"Laser linewidth: {delta_nu/1e6:.0f} MHz")
    print(f"Coherence length: {L_coh:.2f} m")
    print(f"Maximum object depth: ~{L_coh/2:.2f} m (path difference < L_coh)")

    # Stability requirement: fringe must not move > lambda/10 during exposure
    print(f"\nStability requirement: motion < lambda/10 = {lam/10*1e9:.1f} nm")
    print(f"For 1 s exposure, vibration velocity < {lam/10*1e9:.1f} nm/s")


def exercise_2():
    """
    Exercise 2: Volume Hologram (Bragg Condition)
    Design a volume hologram and calculate the angular and wavelength
    selectivity using Bragg diffraction theory.
    """
    lam = 532e-9       # Recording wavelength (m)
    n = 1.5            # Material refractive index
    d = 100e-6         # Hologram thickness (m)
    theta_ref = 30     # Reference beam angle (degrees, in air)

    print("Volume Hologram Design:")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"Material index: n = {n}")
    print(f"Thickness: d = {d*1e6:.0f} um")
    print(f"Reference beam angle (in air): {theta_ref} deg")

    # Angle inside medium (Snell's law)
    theta_in = np.degrees(np.arcsin(np.sin(np.radians(theta_ref)) / n))
    print(f"Reference beam angle (in medium): {theta_in:.2f} deg")

    # Grating period (for plane wave reference and object at normal incidence)
    # Lambda_g = lambda / (2*n*sin(theta/2))
    theta_half = np.radians(theta_ref) / 2  # Half-angle between beams
    Lambda_g = lam / (2 * n * np.sin(np.radians(theta_in / 2 + theta_in / 2)))
    # For reference at theta_in and object at 0:
    Lambda_g = lam / (n * np.sin(np.radians(theta_in)))
    print(f"\nGrating period: Lambda = {Lambda_g*1e9:.1f} nm")
    print(f"Number of grating planes: N = {d/Lambda_g:.0f}")

    # Bragg selectivity
    # Angular selectivity: delta_theta ~ Lambda / d (in medium)
    delta_theta_medium = Lambda_g / d  # radians
    delta_theta_air = delta_theta_medium * n  # approximate
    print(f"\nBragg Selectivity:")
    print(f"  Angular selectivity (FWHM): {np.degrees(delta_theta_medium)*60:.2f} arcmin "
          f"({np.degrees(delta_theta_medium):.4f} deg) in medium")
    print(f"  In air: ~{np.degrees(delta_theta_air)*60:.2f} arcmin")

    # Wavelength selectivity: delta_lambda / lambda ~ Lambda / (d * tan(theta))
    delta_lam = lam * Lambda_g / (d * np.tan(np.radians(theta_in)))
    print(f"  Wavelength selectivity: {delta_lam*1e9:.3f} nm")

    # Diffraction efficiency
    # For transmission hologram (Kogelnik theory):
    # eta = sin^2(nu), where nu = pi*delta_n*d/(lambda*cos(theta))
    print(f"\nDiffraction Efficiency (Kogelnik theory):")
    print(f"{'delta_n':>10} {'nu (rad)':>10} {'eta':>10}")
    print("-" * 32)

    for delta_n in [0.001, 0.005, 0.010, 0.020, 0.050]:
        nu = np.pi * delta_n * d / (lam * np.cos(np.radians(theta_in)))
        eta = np.sin(nu)**2
        print(f"{delta_n:>10.3f} {nu:>10.3f} {eta*100:>9.1f}%")

    # For 100% efficiency (nu = pi/2):
    delta_n_100 = lam * np.cos(np.radians(theta_in)) / (2 * d)
    print(f"\nFor 100% efficiency: delta_n = {delta_n_100:.4f}")

    # Multiplexing capacity
    print(f"\n--- Angular Multiplexing ---")
    angular_range = 40  # degrees total range available
    n_holograms = angular_range / np.degrees(delta_theta_air)
    print(f"Angular range: {angular_range} deg")
    print(f"Angular selectivity: {np.degrees(delta_theta_air):.4f} deg")
    print(f"Maximum number of multiplexed holograms: ~{int(n_holograms)}")


def exercise_3():
    """
    Exercise 3: Digital Holography Simulation
    Simulate the recording and reconstruction of a digital hologram
    using the angular spectrum method.
    """
    # Simulation parameters
    N = 512
    lam = 633e-9       # Wavelength (m)
    pixel = 3.45e-6    # Camera pixel size (m)
    L = N * pixel      # Sensor size (m)
    k = 2 * np.pi / lam

    print("Digital Holography Simulation:")
    print(f"Grid: {N}x{N}")
    print(f"Pixel size: {pixel*1e6:.2f} um")
    print(f"Sensor size: {L*1e3:.2f} mm x {L*1e3:.2f} mm")
    print(f"Wavelength: {lam*1e9:.1f} nm")

    # Create coordinate arrays
    x = np.arange(N) * pixel - L/2
    X, Y = np.meshgrid(x, x)

    # Object: three point sources at different depths
    z_obj = 50e-3  # Object distance (m)
    print(f"\nObject distance: z = {z_obj*1e3:.0f} mm")

    # Point source at origin
    r1 = np.sqrt(X**2 + Y**2 + z_obj**2)
    U_obj = np.exp(1j * k * r1) / r1

    # Maximum recordable angle
    theta_max = np.arcsin(lam / (2 * pixel))  # Nyquist
    print(f"Maximum off-axis angle: {np.degrees(theta_max):.2f} deg")
    print(f"  (Set by pixel Nyquist: lambda/(2*pixel) = {lam/(2*pixel):.6f})")

    # Reference beam (plane wave at angle)
    theta_ref = np.radians(1.0)  # Small off-axis angle
    kx_ref = k * np.sin(theta_ref)
    U_ref = np.exp(1j * kx_ref * X)

    # Record hologram (interference pattern)
    U_total = U_obj + U_ref
    I_hologram = np.abs(U_total)**2

    print(f"\nHologram statistics:")
    print(f"  Mean intensity: {np.mean(I_hologram):.4e}")
    print(f"  Max intensity: {np.max(I_hologram):.4e}")
    print(f"  Min intensity: {np.min(I_hologram):.4e}")
    print(f"  Fringe contrast: {(np.max(I_hologram)-np.min(I_hologram))/(np.max(I_hologram)+np.min(I_hologram)):.4f}")

    # Reconstruction: illuminate hologram with reference beam
    # U_recon = I_hologram * U_ref
    U_recon = I_hologram * U_ref

    # Angular spectrum propagation back to object plane
    fx = np.fft.fftfreq(N, pixel)
    FX, FY = np.meshgrid(fx, fx)

    # Transfer function for back-propagation (negative z)
    H = np.exp(-1j * k * z_obj * np.sqrt(1 - (lam*FX)**2 - (lam*FY)**2 + 0j))

    A_recon = np.fft.fft2(U_recon)
    U_image = np.fft.ifft2(A_recon * H)
    I_image = np.abs(U_image)**2

    # Find the reconstructed point source
    idx_max = np.unravel_index(np.argmax(I_image), I_image.shape)
    x_max = x[idx_max[1]]
    y_max = x[idx_max[0]]

    print(f"\nReconstruction (back-propagation to z = {z_obj*1e3:.0f} mm):")
    print(f"  Peak location: ({x_max*1e6:.1f}, {y_max*1e6:.1f}) um")
    print(f"  Peak intensity: {I_image[idx_max]:.4e}")
    print(f"  Background RMS: {np.std(I_image):.4e}")
    print(f"  SNR: {I_image[idx_max]/np.std(I_image):.1f}")

    # Resolution estimate
    # Lateral resolution: delta_x ~ lambda * z / L
    delta_x = lam * z_obj / L
    # Axial resolution: delta_z ~ lambda / (NA)^2 where NA ~ L/(2*z)
    NA = L / (2 * z_obj)
    delta_z = lam / NA**2
    print(f"\nResolution estimates:")
    print(f"  Lateral: {delta_x*1e6:.2f} um")
    print(f"  Axial: {delta_z*1e3:.2f} mm")
    print(f"  Effective NA: {NA:.4f}")


def exercise_4():
    """
    Exercise 4: Gabor vs Off-Axis Holography Comparison
    Compare in-line (Gabor) and off-axis (Leith-Upatnieks) holography,
    discussing the twin-image problem and its solutions.
    """
    lam = 633e-9
    pixel = 3.45e-6
    N = 512

    print("Gabor vs Off-Axis Holography Comparison:")
    print("=" * 55)

    print(f"\n{'Property':>25} {'Gabor (In-line)':>20} {'Off-Axis (L-U)':>20}")
    print("-" * 67)

    comparisons = [
        ("Reference beam", "Collinear", "Angled"),
        ("Setup complexity", "Simple", "Complex"),
        ("Twin image", "Overlapping", "Separated"),
        ("Bandwidth usage", "1/4 of sensor", "1/4 of sensor"),
        ("Object type", "Sparse/weak", "Any"),
        ("Phase retrieval", "Iterative needed", "Direct"),
        ("Max object NA", "~lambda/pixel", "~lambda/(2*pixel)"),
    ]

    for prop, gabor, offaxis in comparisons:
        print(f"{prop:>25} {gabor:>20} {offaxis:>20}")

    # Gabor hologram: twin image problem
    print(f"\n--- Gabor Hologram: Twin Image Problem ---")
    print(f"Hologram intensity: I = |R + O|^2 = |R|^2 + |O|^2 + R*O* + R*O")
    print(f"Reconstruction terms:")
    print(f"  |R|^2: DC background (zero-order)")
    print(f"  |O|^2: Autocorrelation (halo)")
    print(f"  R*O*:  Real image (desired)")
    print(f"  R*O:   Twin image (conjugate, overlapping)")

    # Off-axis: spatial separation
    print(f"\n--- Off-Axis: Spatial Frequency Separation ---")
    theta_ref = 3.0  # Reference beam angle (deg)
    f_carrier = np.sin(np.radians(theta_ref)) / lam  # carrier frequency
    f_nyquist = 1 / (2 * pixel)

    print(f"Reference angle: {theta_ref} deg")
    print(f"Carrier frequency: {f_carrier/1e3:.1f} cy/mm")
    print(f"Nyquist frequency: {f_nyquist/1e3:.1f} cy/mm")
    print(f"Bandwidth available: {f_nyquist/1e3:.1f} cy/mm")

    # Object bandwidth must be < carrier/2 for no overlap
    f_obj_max = f_carrier / 2
    print(f"\nMax object bandwidth: {f_obj_max/1e3:.1f} cy/mm")
    print(f"  (To avoid overlap between zero-order and image terms)")

    # Condition for separation: theta_ref > 3*theta_obj
    theta_obj_max = theta_ref / 3
    print(f"Max object angular extent: {theta_obj_max:.2f} deg")

    # Solutions for twin-image in Gabor holography
    print(f"\n--- Solutions for Twin-Image Problem ---")
    solutions = [
        "1. Phase-shifting: Record 3+ holograms with shifted reference",
        "2. Iterative algorithms (Fienup, GS) to isolate real image",
        "3. Multi-distance recording (different z, then propagate)",
        "4. Off-axis recording (the standard solution)",
        "5. Compressive holography (sparsity-based reconstruction)",
    ]
    for sol in solutions:
        print(f"  {sol}")

    # Space-bandwidth product
    print(f"\n--- Space-Bandwidth Product ---")
    SBP_gabor = (N/2)**2  # Half bandwidth usable
    SBP_offaxis = (N/4)**2  # Quarter bandwidth per dimension
    print(f"Gabor SBP: {SBP_gabor:.0f} pixels (effective)")
    print(f"Off-axis SBP: {SBP_offaxis:.0f} pixels (effective)")
    print(f"Off-axis uses {SBP_offaxis/SBP_gabor*100:.0f}% of Gabor efficiency")
    print(f"  (Trade-off: SBP for direct phase retrieval)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Hologram Resolution Requirements", exercise_1),
        ("Exercise 2: Volume Hologram (Bragg Condition)", exercise_2),
        ("Exercise 3: Digital Holography Simulation", exercise_3),
        ("Exercise 4: Gabor vs Off-Axis Comparison", exercise_4),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

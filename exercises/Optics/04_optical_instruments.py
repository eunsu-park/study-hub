"""
Exercises for Lesson 04: Optical Instruments
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Eye Correction
    Calculate the corrective lens power needed for myopia and hyperopia,
    and determine the range of clear vision with correction.
    """
    print("--- Part (a): Myopia (nearsightedness) ---")
    # Far point at 2.0 m instead of infinity
    far_point = 2.0  # meters
    # Corrective lens: bring infinity to far point
    # 1/f = 1/v - 1/u, u = infinity => 1/f = 1/v
    # Image should be at -far_point (virtual, on same side as object)
    P_myopia = -1.0 / far_point  # Diopters (diverging lens)
    print(f"Far point: {far_point} m")
    print(f"Corrective lens power: P = {P_myopia:.2f} D")
    print(f"Focal length: f = {1/P_myopia*100:.0f} cm")

    print("\n--- Part (b): Hyperopia (farsightedness) ---")
    # Near point at 1.0 m instead of 25 cm
    near_point_uncorrected = 1.0  # m
    near_point_normal = 0.25      # m
    # Lens should image object at 25 cm to virtual image at 1.0 m
    # 1/f = 1/v - 1/u = 1/(-1.0) - 1/(-0.25) = -1 + 4 = 3
    P_hyperopia = 1.0/(-near_point_uncorrected) - 1.0/(-near_point_normal)
    print(f"Near point (uncorrected): {near_point_uncorrected} m")
    print(f"Desired near point: {near_point_normal*100:.0f} cm")
    print(f"Corrective lens power: P = {P_hyperopia:.2f} D")

    print("\n--- Part (c): Presbyopia (age-related) ---")
    # Reading glasses for person with near point at 50 cm
    near_actual = 0.50    # m
    near_desired = 0.25   # m
    P_reading = 1.0/(-near_actual) - 1.0/(-near_desired)
    print(f"Near point: {near_actual*100:.0f} cm")
    print(f"Reading glasses power: P = {P_reading:.2f} D")
    print(f"  (Standard reading glasses: +1.0 to +3.0 D)")


def exercise_2():
    """
    Exercise 2: Microscope Design
    Design a compound microscope with specified magnification,
    calculate NA, and determine the resolution limit.
    """
    # Target specifications
    M_total = 400       # Total magnification
    f_obj = 0.004       # Objective focal length (4 mm)
    f_eye = 0.025       # Eyepiece focal length (25 mm)
    L = 0.160           # Tube length (160 mm standard)
    d_near = 0.25       # Near point distance (25 cm)

    # Magnification of objective
    M_obj = -L / f_obj
    # Magnification of eyepiece
    M_eye = d_near / f_eye
    # Total magnification
    M_total_calc = abs(M_obj) * M_eye

    print("Compound Microscope Design:")
    print(f"Objective f = {f_obj*1000:.0f} mm")
    print(f"Eyepiece f = {f_eye*1000:.0f} mm")
    print(f"Tube length: {L*1000:.0f} mm")
    print(f"\nObjective magnification: M_obj = {abs(M_obj):.0f}x")
    print(f"Eyepiece magnification: M_eye = {M_eye:.0f}x")
    print(f"Total magnification: {M_total_calc:.0f}x")

    # Numerical aperture and resolution
    n_medium = 1.0   # Air
    theta_max = np.radians(30)  # Half-angle of objective
    NA = n_medium * np.sin(theta_max)
    print(f"\nDry objective:")
    print(f"  Half-angle: {np.degrees(theta_max):.0f} deg")
    print(f"  NA = {NA:.3f}")

    # Rayleigh criterion: d_min = 0.61 * lambda / NA
    lam = 550e-9  # Green light
    d_min = 0.61 * lam / NA
    print(f"  Resolution (Rayleigh): {d_min*1e6:.3f} um = {d_min*1e9:.0f} nm")

    # Oil immersion
    n_oil = 1.515
    theta_oil = np.radians(64)  # Typical oil immersion angle
    NA_oil = n_oil * np.sin(theta_oil)
    d_min_oil = 0.61 * lam / NA_oil
    print(f"\nOil immersion objective:")
    print(f"  NA = {NA_oil:.3f}")
    print(f"  Resolution: {d_min_oil*1e6:.3f} um = {d_min_oil*1e9:.0f} nm")

    # Useful magnification range
    M_useful_min = 500 * NA_oil
    M_useful_max = 1000 * NA_oil
    print(f"\nUseful magnification range: {M_useful_min:.0f}x to {M_useful_max:.0f}x")
    print(f"  (Below = resolution wasted, Above = empty magnification)")


def exercise_3():
    """
    Exercise 3: Telescope Comparison
    Compare refracting and reflecting telescope designs and calculate
    their angular resolution and light-gathering power.
    """
    print("--- Refracting Telescope (Keplerian) ---")
    f_obj_refract = 1.0    # Objective focal length (m)
    f_eye_refract = 0.025  # Eyepiece focal length (m)
    D_refract = 0.10       # Aperture diameter (m)

    M_refract = f_obj_refract / f_eye_refract
    theta_rayleigh_refract = 1.22 * 550e-9 / D_refract

    print(f"Objective: f = {f_obj_refract*100:.0f} cm, D = {D_refract*100:.0f} cm")
    print(f"Eyepiece: f = {f_eye_refract*1000:.0f} mm")
    print(f"Angular magnification: M = {M_refract:.0f}x")
    print(f"Angular resolution: {np.degrees(theta_rayleigh_refract)*3600:.2f} arcsec")
    print(f"Light-gathering power (vs eye 7mm): "
          f"{(D_refract/0.007)**2:.0f}x")

    print("\n--- Reflecting Telescope (Newtonian) ---")
    D_reflect = 0.20       # 200 mm aperture
    f_reflect = 1.2        # Primary focal length (m)
    f_eye_reflect = 0.020  # Eyepiece (m)

    M_reflect = f_reflect / f_eye_reflect
    theta_rayleigh_reflect = 1.22 * 550e-9 / D_reflect

    print(f"Primary: f = {f_reflect*100:.0f} cm, D = {D_reflect*100:.0f} cm")
    print(f"f-ratio: f/{f_reflect/D_reflect:.0f}")
    print(f"Angular magnification: M = {M_reflect:.0f}x")
    print(f"Angular resolution: {np.degrees(theta_rayleigh_reflect)*3600:.2f} arcsec")
    print(f"Light-gathering power: {(D_reflect/0.007)**2:.0f}x")

    print("\n--- Comparison ---")
    print(f"{'Property':>25} {'Refractor':>12} {'Reflector':>12}")
    print("-" * 51)
    print(f"{'Aperture (cm)':>25} {D_refract*100:>12.0f} {D_reflect*100:>12.0f}")
    print(f"{'Magnification':>25} {M_refract:>12.0f} {M_reflect:>12.0f}")
    print(f"{'Resolution (arcsec)':>25} "
          f"{np.degrees(theta_rayleigh_refract)*3600:>12.2f} "
          f"{np.degrees(theta_rayleigh_reflect)*3600:>12.2f}")
    print(f"{'Light-gathering':>25} "
          f"{(D_refract/0.007)**2:>12.0f}x "
          f"{(D_reflect/0.007)**2:>12.0f}x")
    print(f"{'Chromatic aberration':>25} {'Yes':>12} {'No':>12}")
    print(f"{'Central obstruction':>25} {'No':>12} {'Yes':>12}")


def exercise_4():
    """
    Exercise 4: Camera Settings and Depth of Field
    Calculate depth of field for a camera lens at different f-numbers
    and focus distances.
    """
    f = 0.050   # 50 mm lens
    c = 0.03e-3  # Circle of confusion (30 um for full-frame)

    print("Camera Depth of Field Analysis")
    print(f"Focal length: {f*1000:.0f} mm")
    print(f"Circle of confusion: {c*1e6:.0f} um")

    # Hyperfocal distance: H = f^2/(N*c) + f
    print(f"\n{'f-number':>10} {'Hyperfocal (m)':>16}")
    print("-" * 28)
    for N in [1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0]:
        H = f**2 / (N * c) + f
        print(f"f/{N:<6.1f} {H:>16.2f}")

    # Depth of field for a portrait (subject at 3 m)
    s = 3.0  # Subject distance (m)
    print(f"\nDepth of field at s = {s} m:")
    print(f"{'f-number':>10} {'Near (m)':>10} {'Far (m)':>10} {'DOF (m)':>10}")
    print("-" * 42)

    for N in [1.4, 2.8, 5.6, 8.0, 16.0]:
        H = f**2 / (N * c)
        # Near focus limit
        D_near = s * (H - f) / (H + s - 2*f)
        # Far focus limit
        if H - s + 2*f > 0:
            D_far_val = s * (H - f) / (H - s)
        else:
            D_far_val = np.inf
        DOF = D_far_val - D_near if np.isfinite(D_far_val) else np.inf

        far_str = f"{D_far_val:.2f}" if np.isfinite(D_far_val) else "inf"
        dof_str = f"{DOF:.2f}" if np.isfinite(DOF) else "inf"
        print(f"f/{N:<6.1f} {D_near:>10.2f} {far_str:>10} {dof_str:>10}")

    # Exposure equivalence
    print("\nExposure Equivalence (same exposure):")
    print(f"{'Setting':>30} {'Relative light':>16}")
    print("-" * 48)
    settings = [
        ("f/2.8, 1/500s, ISO 100", 2.8, 1/500, 100),
        ("f/4.0, 1/250s, ISO 100", 4.0, 1/250, 100),
        ("f/5.6, 1/125s, ISO 100", 5.6, 1/125, 100),
        ("f/5.6, 1/250s, ISO 200", 5.6, 1/250, 200),
        ("f/8.0, 1/125s, ISO 200", 8.0, 1/125, 200),
    ]
    ref_ev = np.log2(2.8**2 / (1/500)) - np.log2(100/100)
    for name, N, t, iso in settings:
        ev = np.log2(N**2 / t) - np.log2(iso/100)
        print(f"{name:>30} {2**(ref_ev - ev):>16.2f}x")


def exercise_5():
    """
    Exercise 5: Satellite Imaging Resolution
    Calculate the ground resolution of an Earth-observation satellite
    camera system.
    """
    # Satellite parameters
    h = 700e3           # Orbital altitude (m)
    D = 0.30            # Aperture diameter (m)
    f = 3.6             # Focal length (m)
    pixel_size = 6.5e-6  # Pixel size (m)
    lam = 550e-9        # Wavelength (m)

    print("Satellite Imaging System:")
    print(f"Orbital altitude: {h/1000:.0f} km")
    print(f"Aperture: D = {D*100:.0f} cm")
    print(f"Focal length: f = {f:.1f} m")
    print(f"Pixel size: {pixel_size*1e6:.1f} um")

    # Diffraction-limited angular resolution (Rayleigh)
    theta_diff = 1.22 * lam / D
    GSD_diff = h * theta_diff  # Ground Sample Distance (diffraction limit)

    print(f"\nDiffraction-limited resolution:")
    print(f"  Angular: {theta_diff*1e6:.2f} urad = {np.degrees(theta_diff)*3600:.3f} arcsec")
    print(f"  Ground: {GSD_diff:.2f} m")

    # Pixel-limited resolution (IFOV)
    IFOV = pixel_size / f  # Instantaneous Field of View
    GSD_pixel = h * IFOV

    print(f"\nPixel-limited resolution:")
    print(f"  IFOV: {IFOV*1e6:.2f} urad")
    print(f"  Ground: {GSD_pixel:.2f} m")

    # Nyquist sampling check
    Q = theta_diff / IFOV  # Q factor (should be ~2 for Nyquist)
    print(f"\nNyquist Q factor: {Q:.2f}")
    if Q > 2:
        print(f"  System is oversampled (pixel-limited)")
    elif Q < 1:
        print(f"  System is undersampled (aliasing possible)")
    else:
        print(f"  System is well-sampled")

    # Actual GSD (determined by the larger of the two)
    GSD_actual = max(GSD_diff, GSD_pixel)
    print(f"\nActual GSD: {GSD_actual:.2f} m")

    # Swath width
    n_pixels = 12000  # Number of pixels across
    swath = n_pixels * GSD_pixel
    print(f"\nSwath width ({n_pixels} pixels): {swath/1000:.1f} km")

    # MTF considerations
    print("\nMTF at Nyquist frequency:")
    # MTF_diff at Nyquist: for circular aperture, MTF(f) = (2/pi)[arccos(f/fc) - (f/fc)sqrt(1-(f/fc)^2)]
    f_nyquist = 1 / (2 * pixel_size)  # cycles/mm on focal plane
    f_cutoff = D / (lam * f)  # diffraction cutoff frequency
    rho = f_nyquist / f_cutoff  # normalized spatial frequency

    if rho < 1:
        MTF_diff = (2/np.pi) * (np.arccos(rho) - rho * np.sqrt(1 - rho**2))
    else:
        MTF_diff = 0
    print(f"  Diffraction MTF: {MTF_diff:.3f}")
    print(f"  (f_nyquist/f_cutoff = {rho:.3f})")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Eye Correction", exercise_1),
        ("Exercise 2: Microscope Design", exercise_2),
        ("Exercise 3: Telescope Comparison", exercise_3),
        ("Exercise 4: Camera Settings and Depth of Field", exercise_4),
        ("Exercise 5: Satellite Imaging Resolution", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

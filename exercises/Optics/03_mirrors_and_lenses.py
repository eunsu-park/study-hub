"""
Exercises for Lesson 03: Mirrors and Lenses
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Mirror Problems
    Solve for image position, magnification, and characteristics
    for concave and convex mirrors.
    """
    print("--- Part (a): Concave mirror ---")
    R_concave = -0.40   # Radius of curvature (m), negative for concave (convention)
    f_concave = R_concave / 2  # Focal length
    # Actually, for concave mirror: f = R/2 > 0
    f = 0.20  # focal length of concave mirror = R/2 = 0.20 m

    # Object at different distances
    object_distances = [0.50, 0.30, 0.20, 0.15, 0.10]  # meters

    print(f"Concave mirror: f = {f*100:.0f} cm")
    print(f"{'s (cm)':>8} {'s_prime (cm)':>14} {'M':>8} {'Image type':>20}")
    print("-" * 54)

    for s in object_distances:
        # Mirror equation: 1/s + 1/s' = 1/f
        if abs(s - f) < 1e-10:
            print(f"{s*100:>8.0f} {'infinity':>14} {'---':>8} {'---':>20}")
            continue
        s_prime = 1.0 / (1.0/f - 1.0/s)
        M = -s_prime / s
        if s_prime > 0:
            img_type = "real, " + ("inverted" if M < 0 else "upright")
        else:
            img_type = "virtual, " + ("inverted" if M < 0 else "upright")
        print(f"{s*100:>8.0f} {s_prime*100:>14.2f} {M:>8.2f} {img_type:>20}")

    print("\n--- Part (b): Convex mirror ---")
    f_convex = -0.15  # Negative focal length for convex mirror (m)

    s = 0.30  # Object at 30 cm
    s_prime = 1.0 / (1.0/f_convex - 1.0/s)
    M = -s_prime / s
    print(f"Convex mirror: f = {f_convex*100:.0f} cm")
    print(f"Object distance: s = {s*100:.0f} cm")
    print(f"Image distance: s' = {s_prime*100:.2f} cm")
    print(f"Magnification: M = {M:.4f}")
    print(f"Image is {'virtual' if s_prime < 0 else 'real'}, "
          f"{'upright' if M > 0 else 'inverted'}, "
          f"{'diminished' if abs(M) < 1 else 'magnified'}")


def exercise_2():
    """
    Exercise 2: Lensmaker's Equation
    Calculate the focal length of a lens given its radii of curvature
    and refractive index using the lensmaker's equation.
    """
    # Lensmaker's equation: 1/f = (n-1)[1/R1 - 1/R2 + (n-1)*d/(n*R1*R2)]
    # For thin lens: 1/f = (n-1)[1/R1 - 1/R2]

    n = 1.52        # Crown glass
    R1 = 0.10       # First radius of curvature (m), convex
    R2 = -0.15      # Second radius of curvature (m), concave

    # Thin lens approximation
    inv_f_thin = (n - 1) * (1/R1 - 1/R2)
    f_thin = 1 / inv_f_thin

    print("Lensmaker's Equation:")
    print(f"n = {n}, R1 = {R1*100:.0f} cm, R2 = {R2*100:.0f} cm")
    print(f"\nThin lens focal length: f = {f_thin*100:.2f} cm")
    print(f"Optical power: P = {inv_f_thin:.2f} diopters")

    # Thick lens version
    d = 0.005  # Lens thickness = 5 mm
    inv_f_thick = (n-1) * (1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))
    f_thick = 1 / inv_f_thick

    print(f"\nThick lens (d = {d*1000:.0f} mm):")
    print(f"Focal length: f = {f_thick*100:.2f} cm")
    print(f"Difference: {abs(f_thick-f_thin)*100:.4f} cm ({abs(f_thick-f_thin)/f_thin*100:.2f}%)")

    # Different lens types
    print("\n--- Common Lens Types ---")
    lens_types = [
        ("Biconvex (symmetric)", 1.52, 0.10, -0.10),
        ("Plano-convex", 1.52, 0.10, np.inf),
        ("Positive meniscus", 1.52, 0.08, 0.15),
        ("Biconcave (symmetric)", 1.52, -0.10, 0.10),
        ("Plano-concave", 1.52, np.inf, 0.10),
    ]
    print(f"{'Type':>25} {'R1 (cm)':>9} {'R2 (cm)':>9} {'f (cm)':>9}")
    print("-" * 56)
    for name, n_l, r1, r2 in lens_types:
        inv_f = (n_l - 1) * (1/r1 - 1/r2)
        fl = 1/inv_f if inv_f != 0 else np.inf
        r1_str = f"{r1*100:.0f}" if np.isfinite(r1) else "inf"
        r2_str = f"{r2*100:.0f}" if np.isfinite(r2) else "inf"
        print(f"{name:>25} {r1_str:>9} {r2_str:>9} {fl*100:>9.2f}")


def exercise_3():
    """
    Exercise 3: Compound Optical System (Thin Lenses + ABCD Matrices)
    Analyze a two-lens system using both the thin lens equation
    iteratively and ABCD ray transfer matrices.
    """
    f1 = 0.10   # First lens focal length (m)
    f2 = 0.05   # Second lens focal length (m)
    d = 0.15    # Separation between lenses (m)
    s1 = 0.20   # Object distance from first lens (m)

    print("--- Method 1: Sequential thin lens equation ---")
    # Image from first lens
    s1_prime = 1.0 / (1.0/f1 - 1.0/s1)
    M1 = -s1_prime / s1
    print(f"Lens 1 (f={f1*100:.0f} cm): object at {s1*100:.0f} cm")
    print(f"  Image at s1' = {s1_prime*100:.2f} cm, M1 = {M1:.4f}")

    # Object for second lens
    s2 = d - s1_prime  # Distance from second lens to the intermediate image
    s2_prime = 1.0 / (1.0/f2 - 1.0/s2)
    M2 = -s2_prime / s2
    print(f"Lens 2 (f={f2*100:.0f} cm): object at {s2*100:.2f} cm")
    print(f"  Image at s2' = {s2_prime*100:.2f} cm, M2 = {M2:.4f}")

    M_total = M1 * M2
    print(f"Total magnification: M = M1*M2 = {M_total:.4f}")

    print("\n--- Method 2: ABCD Matrix ---")
    # ABCD matrices:
    # Free space: [[1, d], [0, 1]]
    # Thin lens: [[1, 0], [-1/f, 1]]

    # System matrix = Lens2 @ FreeSpace @ Lens1
    L1 = np.array([[1, 0], [-1/f1, 1]])
    Space = np.array([[1, d], [0, 1]])
    L2 = np.array([[1, 0], [-1/f2, 1]])

    M_sys = L2 @ Space @ L1
    A, B, C, D = M_sys[0, 0], M_sys[0, 1], M_sys[1, 0], M_sys[1, 1]

    print(f"System ABCD matrix:")
    print(f"  A = {A:.4f}, B = {B:.4f}")
    print(f"  C = {C:.4f}, D = {D:.4f}")

    # Effective focal length: f_eff = -1/C
    f_eff = -1.0 / C
    print(f"Effective focal length: f_eff = {f_eff*100:.2f} cm")

    # Image location: for object at s1, image at s2'
    # s' = (A*s + B) / (C*s + D) -- but careful with sign conventions
    # Using the input ray [y, u]: y_out = A*y + B*u, u_out = C*y + D*u
    # For axial object at s1: input ray [h, -h/s1], solve for s' where y_out = 0
    # Alternatively, use the sequential result already computed.
    print(f"\nBoth methods give same total magnification: M = {M_total:.4f}")

    # Trace a ray
    h_obj = 0.01  # 1 cm object height
    ray_in = np.array([h_obj, -h_obj/s1])  # [height, angle] at lens 1

    ray_after_L1 = L1 @ ray_in
    ray_after_space = Space @ ray_after_L1
    ray_after_L2 = L2 @ ray_after_space

    print(f"\nRay trace (h_obj = {h_obj*100:.1f} cm):")
    print(f"  At Lens 1: y = {ray_in[0]*100:.4f} cm, u = {ray_in[1]:.4f} rad")
    print(f"  After L1:  y = {ray_after_L1[0]*100:.4f} cm, u = {ray_after_L1[1]:.4f} rad")
    print(f"  After gap: y = {ray_after_space[0]*100:.4f} cm, u = {ray_after_space[1]:.4f} rad")
    print(f"  After L2:  y = {ray_after_L2[0]*100:.4f} cm, u = {ray_after_L2[1]:.4f} rad")


def exercise_4():
    """
    Exercise 4: Achromatic Doublet Design
    Design an achromatic doublet lens that corrects chromatic aberration
    by combining crown and flint glass elements.
    """
    # Achromatic condition: phi1/V1 + phi2/V2 = 0
    # Total power: phi = phi1 + phi2 - (d/n)*phi1*phi2
    # For thin cemented doublet: phi = phi1 + phi2

    # Target focal length
    f_target = 0.200  # 200 mm

    # Glass parameters (from Schott catalog)
    # Crown glass (BK7): n_d = 1.5168, V_d = 64.17
    # Flint glass (SF2):  n_d = 1.6477, V_d = 33.85
    n_crown, V_crown = 1.5168, 64.17
    n_flint, V_flint = 1.6477, 33.85

    phi_total = 1.0 / f_target  # Total power (diopters)

    # Solve achromatic condition:
    # phi1/V1 + phi2/V2 = 0 => phi1 = -phi2 * V1/V2
    # phi1 + phi2 = phi_total
    # => phi2 * (1 - V1/V2) = phi_total
    phi2 = phi_total / (1 - V_crown / V_flint)
    phi1 = phi_total - phi2

    f1 = 1.0 / phi1
    f2 = 1.0 / phi2

    print("Achromatic Doublet Design:")
    print(f"Target focal length: {f_target*1000:.0f} mm")
    print(f"\nCrown glass (BK7): n_d = {n_crown}, V_d = {V_crown}")
    print(f"Flint glass (SF2):  n_d = {n_flint}, V_d = {V_flint}")
    print(f"\nPower of crown element: phi1 = {phi1:.2f} D (f1 = {f1*1000:.1f} mm)")
    print(f"Power of flint element: phi2 = {phi2:.2f} D (f2 = {f2*1000:.1f} mm)")
    print(f"Total power: phi = {phi1 + phi2:.2f} D (f = {1/(phi1+phi2)*1000:.1f} mm)")

    # Verify achromatism
    chromo_sum = phi1/V_crown + phi2/V_flint
    print(f"\nAchromatic check: phi1/V1 + phi2/V2 = {chromo_sum:.2e} (should be ~0)")

    # Surface curvatures (cemented doublet, equi-convex crown)
    # Crown: biconvex, R1 = R, R2 = -R
    # phi1 = (n_crown - 1) * (1/R1 - 1/R2) = (n_crown - 1) * 2/R
    R_crown = (n_crown - 1) * 2 / phi1
    print(f"\nCrown element (biconvex): R = {R_crown*1000:.2f} mm")

    # Flint: cemented surface R2 matches, outer surface R3
    R2 = -R_crown  # Cemented surface
    # phi2 = (n_flint - 1) * (1/R2 - 1/R3)
    R3 = 1.0 / (1.0/R2 - phi2/(n_flint - 1))
    print(f"Cemented surface: R2 = {R2*1000:.2f} mm")
    print(f"Flint outer surface: R3 = {R3*1000:.2f} mm")


def exercise_5():
    """
    Exercise 5: Aberration Identification
    Compute Seidel aberration coefficients for simple lens configurations
    and identify the dominant aberrations.
    """
    # For a thin lens with object at infinity, the third-order
    # aberrations depend on the shape factor q and conjugate factor p.

    # Shape factor: q = (R2 + R1) / (R2 - R1)
    # For object at infinity: p = -1

    n = 1.5168  # BK7 glass
    f = 0.100   # 100 mm focal length
    D = 0.025   # 25 mm aperture diameter
    h = D / 2   # Semi-aperture

    print("Seidel Aberrations for a Single Thin Lens")
    print(f"n = {n}, f = {f*1000:.0f} mm, D = {D*1000:.0f} mm")

    # Object at infinity: p = -1
    p = -1.0

    # Scan over shape factor q
    print(f"\n{'Shape (q)':>10} {'Type':>20} {'S_I (SA)':>12} {'S_II (Coma)':>14}")
    print("-" * 58)

    q_values = [-2.0, -1.0, 0.0, 0.714, 1.0, 2.0]
    labels = ["concave-convex", "plano-concave", "symmetric", "best-form",
              "plano-convex", "convex-concave"]

    for q, label in zip(q_values, labels):
        # Seidel coefficients (normalized)
        # S_I (spherical aberration):
        n2 = n**2
        S_I = (h**4 / (8 * f**3)) * (
            (n/(n-1))**2 * q**2
            + (4*(n+1)/(n*(n-1))) * p * q
            + ((3*n+2)*(n-1)/(n**2)) * p**2
            + (n**3 / (n-1)**2)
        )
        # S_II (coma) - simplified for object at infinity
        S_II = (h**2 * 0.01 / (2 * f**2)) * (
            (n/(n-1)) * q + (2*n+1)/n * p
        )
        print(f"{q:>10.3f} {label:>20} {S_I*1e6:>12.3f} {S_II*1e6:>14.3f}")

    # Best shape for minimum spherical aberration
    # dS_I/dq = 0 => q_best = -2*p*(n+1)*(n-1)/(n*(3*n+2)*(n-1))
    # Simplified for p = -1:
    q_best = -2 * p * (n**2 - 1) / (n**2 + 2)
    print(f"\nBest shape factor for min SA: q = {q_best:.4f}")
    print(f"(This is the 'best-form' lens)")

    # Corresponding radii for f = 100 mm
    # q = (R2+R1)/(R2-R1), phi = (n-1)(1/R1 - 1/R2) = 1/f
    # R1 = 2f(n-1)/(q+1), R2 = 2f(n-1)/(q-1)
    if abs(q_best + 1) > 1e-10 and abs(q_best - 1) > 1e-10:
        R1 = 2*f*(n-1) / (q_best + 1)
        R2 = 2*f*(n-1) / (q_best - 1)
        print(f"R1 = {R1*1000:.2f} mm, R2 = {R2*1000:.2f} mm")

    print("\nNote: A single thin lens cannot correct all Seidel aberrations.")
    print("Aberration types: SA (spherical), Coma, Astigmatism, Field curvature, Distortion")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Mirror Problems", exercise_1),
        ("Exercise 2: Lensmaker's Equation", exercise_2),
        ("Exercise 3: Compound System (ABCD Matrices)", exercise_3),
        ("Exercise 4: Achromatic Doublet Design", exercise_4),
        ("Exercise 5: Aberration Identification", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

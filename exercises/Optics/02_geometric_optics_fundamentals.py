"""
Exercises for Lesson 02: Geometric Optics Fundamentals
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Fermat's Principle and Snell's Law
    Derive Snell's law from Fermat's principle by minimizing the optical
    path length for refraction at a flat interface.
    """
    # Demonstrate Fermat's principle numerically:
    # Light travels from point A in medium n1 to point B in medium n2
    # through a flat interface. Find the point of incidence that minimizes OPL.

    n1 = 1.0   # Air
    n2 = 1.5   # Glass

    # Point A: (x_a, y_a) above interface (y > 0)
    x_a, y_a = 0.0, 10.0   # mm
    # Point B: (x_b, y_b) below interface (y < 0)
    x_b, y_b = 8.0, -6.0   # mm

    # Parameterize by x-position on interface
    x_interface = np.linspace(-2, 12, 1000)

    # Optical path length
    d1 = np.sqrt((x_interface - x_a)**2 + y_a**2)
    d2 = np.sqrt((x_interface - x_b)**2 + y_b**2)
    opl = n1 * d1 + n2 * d2

    # Find minimum
    idx_min = np.argmin(opl)
    x_min = x_interface[idx_min]
    opl_min = opl[idx_min]

    # Compute angles at the minimum
    theta1 = np.arctan(abs(x_min - x_a) / abs(y_a))
    theta2 = np.arctan(abs(x_min - x_b) / abs(y_b))

    print("Fermat's Principle -> Snell's Law:")
    print(f"Point A: ({x_a}, {y_a}) mm in medium n1 = {n1}")
    print(f"Point B: ({x_b}, {y_b}) mm in medium n2 = {n2}")
    print(f"Optimal interface point: x = {x_min:.4f} mm")
    print(f"Minimum OPL: {opl_min:.4f} mm")
    print(f"\nAngle of incidence:  theta1 = {np.degrees(theta1):.2f} deg")
    print(f"Angle of refraction: theta2 = {np.degrees(theta2):.2f} deg")
    print(f"\nn1 * sin(theta1) = {n1 * np.sin(theta1):.6f}")
    print(f"n2 * sin(theta2) = {n2 * np.sin(theta2):.6f}")
    print(f"Snell's law verified: {np.isclose(n1*np.sin(theta1), n2*np.sin(theta2), atol=0.01)}")


def exercise_2():
    """
    Exercise 2: Total Internal Reflection and Fiber NA
    Calculate the critical angle and numerical aperture for an
    optical fiber with given core and cladding indices.
    """
    n_core = 1.48      # Core refractive index
    n_clad = 1.46      # Cladding refractive index
    n_air = 1.0

    # Critical angle at core-cladding interface
    theta_c = np.arcsin(n_clad / n_core)
    print(f"Core index: n_core = {n_core}")
    print(f"Cladding index: n_clad = {n_clad}")
    print(f"Critical angle: theta_c = {np.degrees(theta_c):.2f} deg")

    # Numerical aperture
    NA = np.sqrt(n_core**2 - n_clad**2)
    print(f"Numerical aperture: NA = {NA:.4f}")

    # Maximum acceptance half-angle in air
    theta_max = np.arcsin(NA / n_air)
    print(f"Maximum acceptance half-angle: {np.degrees(theta_max):.2f} deg")

    # Fraction of isotropic light coupled (for Lambertian source)
    coupling_fraction = NA**2
    print(f"Coupling fraction (Lambertian): {coupling_fraction:.4f} = {coupling_fraction*100:.2f}%")

    # V-number for single-mode condition
    # V = (pi * d / lambda) * NA < 2.405 for single mode
    lam = 1.55e-6  # m (telecom wavelength)
    d_single = 2.405 * lam / (np.pi * NA)
    print(f"\nFor single-mode at lambda = {lam*1e6:.2f} um:")
    print(f"Maximum core diameter: {d_single*1e6:.2f} um")


def exercise_3():
    """
    Exercise 3: Prism Analysis
    Calculate the deviation angle and minimum deviation for a prism,
    then use minimum deviation to measure the refractive index.
    """
    # Prism parameters
    A = 60.0  # Apex angle (degrees)
    n = 1.5   # Refractive index

    A_rad = np.radians(A)

    # Minimum deviation: occurs when ray passes symmetrically
    # sin((A + D_min)/2) = n * sin(A/2)
    D_min_rad = 2 * np.arcsin(n * np.sin(A_rad / 2)) - A_rad
    D_min = np.degrees(D_min_rad)

    print(f"Prism apex angle: A = {A:.1f} deg")
    print(f"Refractive index: n = {n}")
    print(f"Minimum deviation: D_min = {D_min:.4f} deg")

    # Verify: n = sin((A + D_min)/2) / sin(A/2)
    n_check = np.sin((A_rad + D_min_rad) / 2) / np.sin(A_rad / 2)
    print(f"Verification: n = {n_check:.6f}")

    # Deviation as a function of incidence angle
    print(f"\nDeviation vs incidence angle:")
    print(f"{'theta_i (deg)':>14} {'theta_r (deg)':>14} {'Deviation (deg)':>16}")
    print("-" * 46)

    for theta_i_deg in range(30, 80, 5):
        theta_i = np.radians(theta_i_deg)
        # Refraction at first surface
        sin_r1 = np.sin(theta_i) / n
        if abs(sin_r1) > 1:
            continue
        r1 = np.arcsin(sin_r1)
        # Angle at second surface
        r2 = A_rad - r1
        # Refraction at second surface
        sin_theta_out = n * np.sin(r2)
        if abs(sin_theta_out) > 1:
            print(f"{theta_i_deg:>14} {'TIR':>14} {'---':>16}")
            continue
        theta_out = np.arcsin(sin_theta_out)
        # Total deviation
        D = theta_i + theta_out - A_rad
        print(f"{theta_i_deg:>14} {np.degrees(theta_out):>14.2f} {np.degrees(D):>16.4f}")

    # Angle at minimum deviation
    theta_i_min = np.arcsin(n * np.sin(A_rad / 2))
    print(f"\nAt minimum deviation: theta_i = {np.degrees(theta_i_min):.2f} deg")
    print(f"  (symmetric: input angle = output angle)")


def exercise_4():
    """
    Exercise 4: Evanescent Wave
    Calculate the penetration depth of the evanescent wave during
    total internal reflection at a glass-air interface.
    """
    n1 = 1.5     # Glass
    n2 = 1.0     # Air
    lam = 633e-9  # He-Ne laser wavelength (m)

    # Critical angle
    theta_c = np.arcsin(n2 / n1)
    print(f"Critical angle: {np.degrees(theta_c):.2f} deg")

    # Evanescent wave penetration depth for several angles > theta_c
    print(f"\nPenetration depth of evanescent wave (lambda = {lam*1e9:.0f} nm):")
    print(f"{'theta_i (deg)':>14} {'delta (nm)':>12} {'delta/lambda':>14}")
    print("-" * 42)

    for theta_deg in [42.5, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0]:
        theta = np.radians(theta_deg)
        if theta <= theta_c:
            continue
        # Penetration depth: delta = lambda / (2*pi*sqrt(n1^2*sin^2(theta) - n2^2))
        denom = np.sqrt(n1**2 * np.sin(theta)**2 - n2**2)
        delta = lam / (2 * np.pi * denom)
        print(f"{theta_deg:>14.1f} {delta*1e9:>12.2f} {delta/lam:>14.4f}")

    # At 45 degrees (just above critical angle)
    theta_45 = np.radians(45.0)
    denom_45 = np.sqrt(n1**2 * np.sin(theta_45)**2 - n2**2)
    delta_45 = lam / (2 * np.pi * denom_45)
    print(f"\nAt theta = 45 deg:")
    print(f"Penetration depth = {delta_45*1e9:.2f} nm")
    print(f"  = {delta_45/lam:.4f} wavelengths")

    # Intensity decay: I(z) = I_0 * exp(-2z/delta)
    print(f"\nIntensity decay at z = delta: I/I_0 = {np.exp(-2):.4f} = {np.exp(-2)*100:.2f}%")
    print(f"At z = 3*delta: I/I_0 = {np.exp(-6):.6f} = {np.exp(-6)*100:.4f}%")


def exercise_5():
    """
    Exercise 5: Mirage Estimation
    Model a mirage using a linear temperature gradient near a hot road
    surface and compute the ray curvature.
    """
    # Temperature gradient model:
    # T(h) = T_ground + dT/dh * h, where h is height above road
    T_ground = 70 + 273.15  # Ground temperature (K) = 343 K
    T_air = 30 + 273.15     # Air at height (K) = 303 K
    h_gradient = 2.0         # Height over which gradient exists (m)
    dT_dh = (T_air - T_ground) / h_gradient  # K/m (negative: hot at bottom)

    print(f"Ground temperature: {T_ground - 273.15:.0f} C = {T_ground:.0f} K")
    print(f"Air temperature at {h_gradient} m: {T_air - 273.15:.0f} C")
    print(f"Temperature gradient: dT/dh = {dT_dh:.1f} K/m")

    # Refractive index of air: n ~ 1 + 7.86e-4 * P / T  (P in atm, T in K)
    P = 1.0  # atm
    n_ground = 1 + 7.86e-4 * P / T_ground
    n_air = 1 + 7.86e-4 * P / T_air

    print(f"\nn(ground): {n_ground:.8f}")
    print(f"n({h_gradient}m):    {n_air:.8f}")
    print(f"Delta_n = {n_air - n_ground:.2e}")

    # Refractive index gradient
    dn_dh = (n_air - n_ground) / h_gradient
    print(f"dn/dh = {dn_dh:.2e} /m")

    # Radius of curvature of a ray: R = n / (dn/dh * cos(theta))
    # For near-horizontal ray, cos(theta) ~ 1
    n_avg = (n_ground + n_air) / 2
    R = n_avg / abs(dn_dh)
    print(f"\nRadius of ray curvature: R = {R:.0f} m = {R/1000:.1f} km")

    # Critical grazing angle for total internal reflection (mirage condition)
    # sin(theta_c) = n_ground / n_air
    # For small angle: theta_c ~ sqrt(2 * delta_n / n)
    delta_n = n_air - n_ground
    theta_c_approx = np.sqrt(2 * abs(delta_n) / n_avg)
    print(f"\nCritical grazing angle: {np.degrees(theta_c_approx)*60:.2f} arcminutes")
    print(f"  = {np.degrees(theta_c_approx):.4f} degrees")

    # At what distance does the mirage appear?
    # For an observer at height h_obs looking at grazing angle:
    h_obs = 1.5  # eye height (m)
    # Distance ~ sqrt(2 * R * h_obs)
    d_mirage = np.sqrt(2 * R * h_obs)
    print(f"\nFor observer at height {h_obs} m:")
    print(f"Mirage appears at distance: {d_mirage:.0f} m")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Fermat's Principle and Snell's Law", exercise_1),
        ("Exercise 2: Total Internal Reflection and Fiber NA", exercise_2),
        ("Exercise 3: Prism Analysis", exercise_3),
        ("Exercise 4: Evanescent Wave", exercise_4),
        ("Exercise 5: Mirage Estimation", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")

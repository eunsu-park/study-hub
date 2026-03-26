"""
Exercises for Lesson 11: Reflection and Refraction
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: Diamond Brilliance
    n = 2.42. Compute critical angle, Brewster angle, plot Rs and Rp.
    """
    n_diamond = 2.42
    n_air = 1.0

    # (a) Critical angle for diamond-air (internal reflection)
    theta_c = np.degrees(np.arcsin(n_air / n_diamond))

    # (b) Brewster angle from air into diamond
    theta_B = np.degrees(np.arctan(n_diamond / n_air))

    print(f"  Diamond refractive index: n = {n_diamond}")
    print(f"  (a) Critical angle (diamond -> air): {theta_c:.2f} degrees")
    print(f"  (b) Brewster angle (air -> diamond): {theta_B:.2f} degrees")

    # (c) Plot Rs and Rp for air -> diamond
    theta_i = np.linspace(0, 89.9, 1000)
    theta_rad = np.radians(theta_i)

    # Snell's law: n1*sin(theta_i) = n2*sin(theta_t)
    sin_theta_t = (n_air / n_diamond) * np.sin(theta_rad)
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    cos_theta_i = np.cos(theta_rad)

    # Fresnel coefficients
    rs = (n_air * cos_theta_i - n_diamond * cos_theta_t) / \
         (n_air * cos_theta_i + n_diamond * cos_theta_t)
    rp = (n_diamond * cos_theta_i - n_air * cos_theta_t) / \
         (n_diamond * cos_theta_i + n_air * cos_theta_t)

    Rs = np.abs(rs)**2
    Rp = np.abs(rp)**2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(theta_i, Rs * 100, 'b-', linewidth=2, label='Rs (s-polarized)')
    ax.plot(theta_i, Rp * 100, 'r-', linewidth=2, label='Rp (p-polarized)')
    ax.axvline(x=theta_B, color='green', linestyle='--', alpha=0.7,
               label=f'Brewster angle = {theta_B:.1f} deg')
    ax.set_xlabel('Angle of incidence (degrees)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('Fresnel Reflectance: Air to Diamond')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex11_diamond_fresnel.png', dpi=150)
    plt.close()

    print(f"  (d) Diamond appears brilliant because:")
    print(f"      Small critical angle ({theta_c:.1f} deg) means most light entering")
    print(f"      the top facet is totally internally reflected by the pavilion facets,")
    print(f"      bouncing back out the top rather than leaking through the bottom.")
    print("  Plot saved: ex11_diamond_fresnel.png")


def exercise_2():
    """
    Exercise 2: Multilayer Dielectric Mirror
    5-layer (alternating nH/nL) quarter-wave mirror at 1064 nm.
    Transfer matrix method.
    """
    n_H = 2.3    # high-index material
    n_L = 1.38   # low-index material
    n_s = 1.52   # glass substrate
    n_0 = 1.0    # air (incident medium)
    lambda_design = 1064e-9  # design wavelength

    # Quarter-wave thicknesses
    d_H = lambda_design / (4 * n_H)
    d_L = lambda_design / (4 * n_L)

    # 5-layer stack: H L H L H (on substrate)
    layers = [(n_H, d_H), (n_L, d_L), (n_H, d_H), (n_L, d_L), (n_H, d_H)]

    lambda_range = np.linspace(800e-9, 1300e-9, 1000)
    R_spectrum = np.zeros(len(lambda_range))

    for idx, lam in enumerate(lambda_range):
        k0 = 2 * np.pi / lam

        # Transfer matrix at normal incidence
        M = np.eye(2, dtype=complex)
        for n_layer, d_layer in layers:
            phi = k0 * n_layer * d_layer
            m = np.array([
                [np.cos(phi), -1j * np.sin(phi) / n_layer],
                [-1j * n_layer * np.sin(phi), np.cos(phi)]
            ])
            M = M @ m

        # Reflection coefficient
        # r = (M[0,0]*n_s + M[0,1]*n_0*n_s - M[1,0] - M[1,1]*n_0) /
        #     (M[0,0]*n_s + M[0,1]*n_0*n_s + M[1,0] + M[1,1]*n_0)
        num = (M[0, 0] + M[0, 1] * n_s) * n_0 - (M[1, 0] + M[1, 1] * n_s)
        den = (M[0, 0] + M[0, 1] * n_s) * n_0 + (M[1, 0] + M[1, 1] * n_s)
        r = num / den
        R_spectrum[idx] = np.abs(r)**2

    R_at_design = R_spectrum[np.argmin(np.abs(lambda_range - lambda_design))]

    print(f"  5-layer dielectric mirror: H-L-H-L-H")
    print(f"  nH = {n_H}, nL = {n_L}, substrate ns = {n_s}")
    print(f"  Design wavelength: {lambda_design*1e9:.0f} nm")
    print(f"  Layer thicknesses: dH = {d_H*1e9:.1f} nm, dL = {d_L*1e9:.1f} nm")
    print(f"  R at {lambda_design*1e9:.0f} nm: {R_at_design*100:.2f}%")
    print(f"  {'R > 99%' if R_at_design > 0.99 else 'R < 99% -- need more layers'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambda_range * 1e9, R_spectrum * 100, 'b-', linewidth=2)
    ax.axvline(x=lambda_design * 1e9, color='red', linestyle='--', alpha=0.5,
               label=f'{lambda_design*1e9:.0f} nm')
    ax.axhline(y=99, color='green', linestyle=':', alpha=0.5, label='99%')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('5-Layer Dielectric Mirror')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex11_dielectric_mirror.png', dpi=150)
    plt.close()
    print("  Plot saved: ex11_dielectric_mirror.png")


def exercise_3():
    """
    Exercise 3: Frustrated Total Internal Reflection
    Two glass prisms separated by air gap. T vs d/lambda.
    """
    n_glass = 1.5
    n_air = 1.0
    theta_i = 45.0  # degrees (above critical angle ~41.8 for n=1.5)

    theta_rad = np.radians(theta_i)
    theta_c = np.degrees(np.arcsin(n_air / n_glass))

    print(f"  Glass prisms: n = {n_glass}")
    print(f"  Critical angle: {theta_c:.2f} degrees")
    print(f"  Incidence angle: {theta_i} degrees (above critical angle)")

    # For TIR: the evanescent wave decays as exp(-kappa*d) in the gap
    # kappa = (2*pi/lambda) * sqrt(n_glass^2 * sin^2(theta) - n_air^2)
    d_over_lambda = np.linspace(0.001, 3.0, 500)

    # Transfer matrix for the air gap layer with evanescent field
    sin_theta = n_glass * np.sin(theta_rad)  # conserved
    kappa = np.sqrt(sin_theta**2 - n_air**2)  # evanescent decay constant (normalized)

    T_vals = np.zeros(len(d_over_lambda))

    for idx, d_lam in enumerate(d_over_lambda):
        d = d_lam  # d in units of lambda (we work in normalized units)
        k0d = 2 * np.pi * d

        # Phase in the air gap (imaginary for evanescent):
        # phi_air = k0 * n_air * cos(theta_t) * d = k0 * d * i * kappa (for TIR)
        phi_air = 1j * 2 * np.pi * kappa * d_lam

        # Transfer matrix for air gap (s-polarization for simplicity)
        cos_theta_air = 1j * kappa / n_air  # imaginary for evanescent
        n_eff_air = n_air * cos_theta_air

        cos_theta_glass = np.cos(theta_rad)
        n_eff_glass = n_glass * cos_theta_glass

        # 3-layer system: glass | air(d) | glass
        # Using Airy formula for a thin film:
        # r12 = (n_eff_glass - n_eff_air) / (n_eff_glass + n_eff_air)
        r12 = (n_eff_glass - n_eff_air) / (n_eff_glass + n_eff_air)
        t12 = 2 * n_eff_glass / (n_eff_glass + n_eff_air)
        r21 = -r12
        t21 = 2 * n_eff_air / (n_eff_glass + n_eff_air)

        # Total transmission through the gap
        exp_phi = np.exp(2j * phi_air)  # round trip phase (imaginary)
        t_total = t12 * t21 * np.exp(1j * phi_air) / (1 - r21**2 * exp_phi)
        T_vals[idx] = np.abs(t_total)**2 * np.real(n_eff_glass / n_eff_glass)

    # Normalize so T -> 1 as d -> 0
    T_vals = T_vals / T_vals[0] if T_vals[0] > 0 else T_vals

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(d_over_lambda, T_vals, 'b-', linewidth=2)
    ax.set_xlabel(r'd / $\lambda$')
    ax.set_ylabel('Transmittance T')
    ax.set_title(f'Frustrated TIR (n={n_glass}, theta={theta_i} deg)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex11_frustrated_tir.png', dpi=150)
    plt.close()
    print(f"  T at d=0: ~1 (prisms touching)")
    print(f"  T at d=lambda: {T_vals[np.argmin(np.abs(d_over_lambda-1.0))]:.4e}")
    print("  Plot saved: ex11_frustrated_tir.png")


def exercise_4():
    """
    Exercise 4: Polarization by Reflection
    Unpolarized light on glass. Plot degree of polarization vs angle.
    """
    n = 1.5
    theta_i = np.linspace(0.1, 89.9, 1000)
    theta_rad = np.radians(theta_i)

    sin_theta_t = np.sin(theta_rad) / n
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    cos_theta_i = np.cos(theta_rad)

    # Fresnel reflectances
    rs = (cos_theta_i - n * cos_theta_t) / (cos_theta_i + n * cos_theta_t)
    rp = (n * cos_theta_i - cos_theta_t) / (n * cos_theta_i + cos_theta_t)
    Rs = rs**2
    Rp = rp**2

    # Degree of polarization
    P = (Rs - Rp) / (Rs + Rp)

    # Brewster angle
    theta_B = np.degrees(np.arctan(n))

    # P at 60 degrees
    idx_60 = np.argmin(np.abs(theta_i - 60))
    P_60 = P[idx_60]

    print(f"  Glass: n = {n}")
    print(f"  Brewster angle: {theta_B:.2f} degrees")
    print(f"  (a) Max polarization = 1.0 at Brewster angle (Rp = 0)")
    print(f"  (b) P is maximized at theta_B = {theta_B:.2f} degrees")
    print(f"  (c) P at 60 degrees: {P_60:.4f}")
    print("      Polarized sunglasses block s-polarized reflected light,")
    print("      reducing glare from horizontal surfaces.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(theta_i, Rs * 100, 'b-', linewidth=2, label='Rs')
    axes[0].plot(theta_i, Rp * 100, 'r-', linewidth=2, label='Rp')
    axes[0].axvline(x=theta_B, color='green', linestyle='--', label=f'Brewster ({theta_B:.1f} deg)')
    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('Reflectance (%)')
    axes[0].set_title('Fresnel Reflectance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(theta_i, P, 'purple', linewidth=2)
    axes[1].axvline(x=theta_B, color='green', linestyle='--', label=f'Brewster')
    axes[1].axvline(x=60, color='orange', linestyle=':', label=f'60 deg (P={P_60:.3f})')
    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('Degree of Polarization P')
    axes[1].set_title('Polarization by Reflection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Glass (n={n}): Reflection Polarization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex11_polarization_reflection.png', dpi=150)
    plt.close()
    print("  Plot saved: ex11_polarization_reflection.png")


if __name__ == "__main__":
    print("=== Exercise 1: Diamond Brilliance ===")
    exercise_1()
    print("\n=== Exercise 2: Multilayer Dielectric Mirror ===")
    exercise_2()
    print("\n=== Exercise 3: Frustrated Total Internal Reflection ===")
    exercise_3()
    print("\n=== Exercise 4: Polarization by Reflection ===")
    exercise_4()
    print("\nAll exercises completed!")

"""
Exercise Solutions: Lesson 02 - Complex Numbers
Mathematical Methods for Physical Sciences

Covers: polar form, De Moivre's theorem, nth roots, complex logarithm,
        AC circuit analysis, conformal mapping
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1_polar_conversion():
    """
    Problem 1: Express in polar form r*e^{i*theta} (-pi < theta <= pi):
    (a) z = -1 + i
    (b) z = -3 - 3*sqrt(3)*i
    (c) z = 5i
    """
    print("=" * 60)
    print("Problem 1: Polar Coordinate Conversion")
    print("=" * 60)

    problems = {
        '(a) z = -1 + i': -1 + 1j,
        '(b) z = -3 - 3*sqrt(3)*i': -3 - 3 * np.sqrt(3) * 1j,
        '(c) z = 5i': 5j,
    }

    for label, z in problems.items():
        r = abs(z)
        theta = np.angle(z)  # returns angle in (-pi, pi]
        print(f"\n{label}")
        print(f"  |z| = {r:.6f}")
        print(f"  theta = {theta:.6f} rad = {np.degrees(theta):.2f} deg")
        print(f"  Polar form: {r:.4f} * exp(i * {theta:.6f})")
        # Verify
        z_reconstructed = r * np.exp(1j * theta)
        print(f"  Verification: {z_reconstructed:.6f} (original: {z:.6f})")

    # (a) -1+i: r = sqrt(2), theta = 3pi/4 = 135 deg
    # (b) -3 - 3*sqrt(3)*i: r = sqrt(9+27) = 6, theta = -2pi/3 = -120 deg
    # (c) 5i: r = 5, theta = pi/2 = 90 deg


def exercise_2_de_moivre():
    """
    Problem 2: Use De Moivre's theorem to express sin(4*theta)
    in terms of sin(theta) and cos(theta).
    """
    print("\n" + "=" * 60)
    print("Problem 2: De Moivre's Theorem - sin(4*theta)")
    print("=" * 60)

    # (cos(theta) + i*sin(theta))^4 = cos(4*theta) + i*sin(4*theta)
    # Expand LHS using binomial theorem:
    # = C(4,0)*cos^4 + C(4,1)*cos^3*(i*sin) + C(4,2)*cos^2*(i*sin)^2
    #   + C(4,3)*cos*(i*sin)^3 + C(4,4)*(i*sin)^4
    # = cos^4 + 4i*cos^3*sin - 6*cos^2*sin^2 - 4i*cos*sin^3 + sin^4

    # Imaginary part = sin(4*theta):
    # sin(4*theta) = 4*cos^3*sin - 4*cos*sin^3
    #              = 4*cos*sin*(cos^2 - sin^2)
    #              = 4*cos*sin*cos(2*theta)

    print("\nDerivation:")
    print("  (cos t + i sin t)^4 = cos 4t + i sin 4t")
    print("\n  Expanding with binomial theorem:")
    print("  = cos^4(t) + 4i cos^3(t) sin(t) - 6 cos^2(t) sin^2(t)")
    print("    - 4i cos(t) sin^3(t) + sin^4(t)")
    print("\n  Imaginary part:")
    print("  sin(4t) = 4 cos^3(t) sin(t) - 4 cos(t) sin^3(t)")
    print("          = 4 cos(t) sin(t) [cos^2(t) - sin^2(t)]")

    # Numerical verification
    print("\nNumerical verification:")
    thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for t in thetas:
        lhs = np.sin(4 * t)
        rhs = 4 * np.cos(t)**3 * np.sin(t) - 4 * np.cos(t) * np.sin(t)**3
        print(f"  t = {t:.4f}: sin(4t) = {lhs:+.8f}, formula = {rhs:+.8f}, "
              f"match = {np.isclose(lhs, rhs)}")


def exercise_3_nth_roots():
    """
    Problem 3: Find all roots of z^4 = -16.
    Plot them in the complex plane.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Fourth Roots of -16")
    print("=" * 60)

    # -16 = 16 * e^{i*pi}
    # z_k = 16^{1/4} * exp(i*(pi + 2*pi*k)/4), k = 0, 1, 2, 3
    # 16^{1/4} = 2

    w = -16 + 0j
    n = 4
    R = abs(w) ** (1.0 / n)
    Phi = np.angle(w)

    print(f"\n-16 = 16 * exp(i*pi)")
    print(f"Modulus of roots: 16^(1/4) = {R:.1f}")
    print(f"\nRoots z_k = 2 * exp(i*(pi + 2*pi*k)/4), k = 0,1,2,3:\n")

    roots = []
    for k in range(n):
        angle = (Phi + 2 * np.pi * k) / n
        z_k = R * np.exp(1j * angle)
        roots.append(z_k)
        print(f"  z_{k} = {z_k.real:+.6f} {z_k.imag:+.6f}i")
        print(f"       = 2 * exp(i * {np.degrees(angle):.1f} deg)")
        verify = z_k ** 4
        print(f"       Verify: z^4 = {verify.real:.4f} + {verify.imag:.4f}i")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    circle = plt.Circle((0, 0), R, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)

    for k, z_k in enumerate(roots):
        ax.plot(z_k.real, z_k.imag, 'ro', markersize=12)
        ax.annotate(f'z_{k} = {z_k.real:.2f}{z_k.imag:+.2f}i',
                    (z_k.real, z_k.imag), textcoords="offset points",
                    xytext=(10, 10), fontsize=10)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Fourth roots of -16', fontsize=14)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex02_fourth_roots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex02_fourth_roots.png")


def exercise_4_complex_logarithm():
    """
    Problem 4: Compute principal values:
    (a) ln(-e)     (b) ln(1+i)     (c) i^i
    """
    print("\n" + "=" * 60)
    print("Problem 4: Complex Logarithm")
    print("=" * 60)

    # (a) ln(-e) = ln(e) + i*arg(-e) = 1 + i*pi
    print("\n(a) ln(-e):")
    z_a = -np.e
    result_a = np.log(complex(z_a))
    print(f"  -e = {z_a:.6f}")
    print(f"  ln|-e| = ln(e) = 1")
    print(f"  arg(-e) = pi")
    print(f"  ln(-e) = 1 + i*pi = {result_a}")

    # (b) ln(1+i) = ln(sqrt(2)) + i*pi/4
    print("\n(b) ln(1+i):")
    z_b = 1 + 1j
    result_b = np.log(z_b)
    print(f"  |1+i| = sqrt(2) = {abs(z_b):.6f}")
    print(f"  arg(1+i) = pi/4 = {np.pi/4:.6f}")
    print(f"  ln(1+i) = ln(sqrt(2)) + i*pi/4")
    print(f"          = {np.log(np.sqrt(2)):.6f} + {np.pi/4:.6f}i")
    print(f"  Computed: {result_b}")

    # (c) i^i = exp(i*ln(i)) = exp(i * (ln|i| + i*pi/2)) = exp(i * i*pi/2) = exp(-pi/2)
    print("\n(c) i^i:")
    ln_i = np.log(1j)
    print(f"  ln(i) = ln|i| + i*arg(i) = 0 + i*pi/2 = {ln_i}")
    result_c = np.exp(1j * ln_i)
    print(f"  i^i = exp(i * ln(i)) = exp(i * i*pi/2) = exp(-pi/2)")
    print(f"      = {np.exp(-np.pi/2):.10f}")
    print(f"  Computed: {result_c:.10f}")
    print(f"  Note: i^i is a REAL number!")


def exercise_5_rlc_circuit():
    """
    Problem 5: RLC circuit analysis
    R = 50 Ohm, L = 20 mH, C = 10 uF, V(t) = 10*cos(omega*t)
    (a) Find resonance frequency f_0
    (b) |Z| and phase at f = 500 Hz
    (c) Maximum current at resonance
    """
    print("\n" + "=" * 60)
    print("Problem 5: AC Circuit Analysis (Series RLC)")
    print("=" * 60)

    R = 50       # Ohm
    L = 20e-3    # Henry
    C = 10e-6    # Farad
    V0 = 10      # Volts

    # (a) Resonance frequency
    omega_0 = 1 / np.sqrt(L * C)
    f_0 = omega_0 / (2 * np.pi)
    print(f"\n(a) Resonance frequency:")
    print(f"  omega_0 = 1/sqrt(LC) = 1/sqrt({L}*{C})")
    print(f"          = {omega_0:.2f} rad/s")
    print(f"  f_0 = omega_0/(2*pi) = {f_0:.2f} Hz")

    # (b) Impedance at f = 500 Hz
    f = 500
    omega = 2 * np.pi * f
    Z = R + 1j * (omega * L - 1 / (omega * C))
    print(f"\n(b) Impedance at f = {f} Hz:")
    print(f"  omega = {omega:.2f} rad/s")
    print(f"  X_L = omega*L = {omega * L:.4f} Ohm")
    print(f"  X_C = 1/(omega*C) = {1/(omega*C):.4f} Ohm")
    print(f"  Z = R + i*(X_L - X_C) = {Z:.4f}")
    print(f"  |Z| = {abs(Z):.4f} Ohm")
    print(f"  phase = {np.degrees(np.angle(Z)):.2f} deg")

    # (c) Maximum current at resonance
    I_max = V0 / R  # At resonance, Z = R (imaginary part cancels)
    print(f"\n(c) Maximum current at resonance:")
    print(f"  At resonance: X_L = X_C, so Z = R = {R} Ohm")
    print(f"  I_max = V0/R = {V0}/{R} = {I_max:.4f} A = {I_max*1000:.1f} mA")

    # Quality factor
    Q = omega_0 * L / R
    print(f"\n  Quality factor Q = omega_0*L/R = {Q:.2f}")
    print(f"  Bandwidth = f_0/Q = {f_0/Q:.2f} Hz")


def exercise_6_conformal_mapping():
    """
    Problem 6: Joukowski transform w = z + 1/z
    (a) Parametric representation when |z| = 2
    (b) Show it's an ellipse and find axes
    """
    print("\n" + "=" * 60)
    print("Problem 6: Conformal Mapping (Joukowski Transform)")
    print("=" * 60)

    # For |z| = R (circle of radius R):
    # z = R*exp(i*theta) = R*cos(theta) + i*R*sin(theta)
    # w = z + 1/z = (R + 1/R)*cos(theta) + i*(R - 1/R)*sin(theta)
    # This is an ellipse with semi-major a = R + 1/R, semi-minor b = |R - 1/R|

    R = 2
    print(f"\n(a) For |z| = {R}:")
    print(f"  z = {R}*exp(i*theta)")
    print(f"  w = z + 1/z = ({R} + 1/{R})*cos(theta) + i*({R} - 1/{R})*sin(theta)")
    print(f"    = {R + 1/R:.4f}*cos(theta) + i*{R - 1/R:.4f}*sin(theta)")

    a = R + 1 / R
    b = abs(R - 1 / R)
    print(f"\n(b) This is an ellipse with:")
    print(f"  u = {a:.4f}*cos(theta)")
    print(f"  v = {b:.4f}*sin(theta)")
    print(f"  => (u/{a:.4f})^2 + (v/{b:.4f})^2 = 1")
    print(f"\n  Semi-major axis a = R + 1/R = {a:.4f}")
    print(f"  Semi-minor axis b = R - 1/R = {b:.4f}")

    # Verification plot
    theta = np.linspace(0, 2 * np.pi, 500)
    z = R * np.exp(1j * theta)
    w = z + 1.0 / z

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # z-plane
    axes[0].plot(z.real, z.imag, 'b-', linewidth=2)
    axes[0].set_title(f'z-plane: |z| = {R}', fontsize=13)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Re(z)')
    axes[0].set_ylabel('Im(z)')

    # w-plane
    axes[1].plot(w.real, w.imag, 'r-', linewidth=2, label='Joukowski image')
    # Overlay exact ellipse
    u_ell = a * np.cos(theta)
    v_ell = b * np.sin(theta)
    axes[1].plot(u_ell, v_ell, 'g--', linewidth=1.5, alpha=0.7, label='Ellipse verification')
    axes[1].set_title(f'w-plane: w = z + 1/z (Ellipse)', fontsize=13)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Re(w)')
    axes[1].set_ylabel('Im(w)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('ex02_joukowski_ellipse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex02_joukowski_ellipse.png")


if __name__ == "__main__":
    exercise_1_polar_conversion()
    exercise_2_de_moivre()
    exercise_3_nth_roots()
    exercise_4_complex_logarithm()
    exercise_5_rlc_circuit()
    exercise_6_conformal_mapping()

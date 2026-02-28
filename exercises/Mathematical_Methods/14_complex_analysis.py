"""
Exercises for Lesson 14: Complex Analysis
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Cauchy-Riemann analyticity check.
    (a) z^3, (b) |z|^2, (c) z_bar, (d) e^{-z} sin(z)
    """
    print("=" * 60)
    print("Problem 1: Cauchy-Riemann Conditions")
    print("=" * 60)

    x, y = sp.symbols('x y', real=True)

    cases = [
        ("(a) f(z) = z^3",
         x**3 - 3 * x * y**2,
         3 * x**2 * y - y**3),
        ("(b) f(z) = |z|^2 = x^2 + y^2",
         x**2 + y**2,
         sp.Integer(0)),
        ("(c) f(z) = z_bar = x - iy",
         x,
         -y),
        ("(d) f(z) = e^{-z} sin(z)",
         sp.re(sp.exp(-(x + sp.I * y)) * sp.sin(x + sp.I * y)),
         sp.im(sp.exp(-(x + sp.I * y)) * sp.sin(x + sp.I * y))),
    ]

    for desc, u_expr, v_expr in cases:
        print(f"\n{desc}")
        u_expr = sp.expand(u_expr)
        v_expr = sp.expand(v_expr)

        ux = sp.diff(u_expr, x)
        uy = sp.diff(u_expr, y)
        vx = sp.diff(v_expr, x)
        vy = sp.diff(v_expr, y)

        cr1 = sp.simplify(ux - vy)
        cr2 = sp.simplify(uy + vx)

        analytic = (cr1 == 0) and (cr2 == 0)
        print(f"  u = {u_expr}, v = {v_expr}")
        print(f"  u_x - v_y = {cr1}")
        print(f"  u_y + v_x = {cr2}")
        print(f"  Analytic: {analytic}")


def exercise_2():
    """
    Problem 2: Find harmonic conjugate v of u = x^3 - 3xy^2 + 2x.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Harmonic Conjugate")
    print("=" * 60)

    x, y = sp.symbols('x y', real=True)
    u = x**3 - 3 * x * y**2 + 2 * x

    print(f"\nu(x,y) = {u}")

    # Check u is harmonic
    lap_u = sp.diff(u, x, 2) + sp.diff(u, y, 2)
    print(f"Laplacian of u: {sp.simplify(lap_u)} (should be 0)")

    # From C-R: v_y = u_x, v_x = -u_y
    ux = sp.diff(u, x)
    uy = sp.diff(u, y)
    print(f"u_x = {ux}")
    print(f"u_y = {uy}")

    # v_y = u_x = 3x^2 - 3y^2 + 2
    # Integrate w.r.t. y: v = 3x^2 y - y^3 + 2y + g(x)
    v_partial = sp.integrate(ux, y)
    print(f"\nIntegrate v_y = u_x w.r.t. y: v = {v_partial} + g(x)")

    # Check: v_x = -u_y => 6xy + g'(x) = -(-6xy) = 6xy
    # g'(x) = 0 => g(x) = C
    v = v_partial
    vx_check = sp.diff(v, x)
    neg_uy = -uy
    g_prime = sp.simplify(neg_uy - vx_check)
    print(f"v_x = {vx_check}, -u_y = {neg_uy}")
    print(f"g'(x) = {g_prime}")
    print(f"\nv(x,y) = {v} + C")
    print(f"f(z) = u + iv = (x^3 - 3xy^2 + 2x) + i(3x^2 y - y^3 + 2y)")
    print(f"     = z^3 + 2z  (analytic function)")


def exercise_3():
    """
    Problem 3: Contour integrals over |z|=2.
    (a) e^z/z^2, (b) cos(z)/z^3, (c) z^2/((z-1)(z+2))
    """
    print("\n" + "=" * 60)
    print("Problem 3: Contour Integrals")
    print("=" * 60)

    z = sp.Symbol('z')

    print("\nC: circle |z| = 2 (counterclockwise)")

    # (a) e^z/z^2: pole at z=0 (order 2), inside C
    print("\n(a) oint e^z/z^2 dz")
    f_a = sp.exp(z) / z**2
    res_a = sp.residue(f_a, z, 0)
    result_a = 2 * sp.pi * sp.I * res_a
    print(f"  Residue at z=0: {res_a}")
    print(f"  Integral = 2*pi*i * {res_a} = {result_a}")
    print(f"  (Using Cauchy formula: f'(0) where f=e^z, f'(0)=1, so 2*pi*i*1)")

    # (b) cos(z)/z^3: pole at z=0 (order 3)
    print("\n(b) oint cos(z)/z^3 dz")
    f_b = sp.cos(z) / z**3
    res_b = sp.residue(f_b, z, 0)
    result_b = 2 * sp.pi * sp.I * res_b
    print(f"  Residue at z=0: {res_b}")
    print(f"  Integral = 2*pi*i * ({res_b}) = {result_b}")
    print(f"  (Using Cauchy formula: f''(0)/2! where f=cos(z), f''=-cos(0)=-1)")

    # (c) z^2/((z-1)(z+2)): poles at z=1 and z=-2, both inside |z|=2
    print("\n(c) oint z^2/((z-1)(z+2)) dz")
    f_c = z**2 / ((z - 1) * (z + 2))
    res_c1 = sp.residue(f_c, z, 1)
    res_c2 = sp.residue(f_c, z, -2)
    result_c = 2 * sp.pi * sp.I * (res_c1 + res_c2)
    print(f"  Residue at z=1: {res_c1}")
    print(f"  Residue at z=-2: {res_c2}")
    print(f"  Sum of residues: {res_c1 + res_c2}")
    print(f"  Integral = 2*pi*i * ({res_c1 + res_c2}) = {sp.simplify(result_c)}")


def exercise_4():
    """
    Problem 4: Residue theorem integrals.
    (a) int_0^{2pi} dtheta/(5+4cos(theta))
    (b) int_0^inf x^2/((x^2+1)(x^2+4)) dx
    (c) int_0^inf cos(3x)/(x^2+1) dx
    """
    print("\n" + "=" * 60)
    print("Problem 4: Residue Theorem Applications")
    print("=" * 60)

    z = sp.Symbol('z')

    # (a) Type 1: trigonometric rational
    print("\n(a) int_0^{2pi} dtheta/(5+4cos(theta))")
    # z = e^{itheta}, cos = (z+1/z)/2, dtheta = dz/(iz)
    integrand_a = 1 / (sp.I * (2 * z**2 + 5 * z + 2))
    poles_a = sp.solve(2 * z**2 + 5 * z + 2, z)
    print(f"  Poles: {poles_a}")
    inner_pole = sp.Rational(-1, 2)  # |z| < 1
    res_a = sp.residue(integrand_a, z, inner_pole)
    result_a = sp.simplify(2 * sp.pi * sp.I * res_a)
    print(f"  Interior pole: z = {inner_pole}")
    print(f"  Residue: {res_a}")
    print(f"  Result: {result_a} = {float(result_a):.6f}")

    # Numerical check
    num_a, _ = quad(lambda t: 1 / (5 + 4 * np.cos(t)), 0, 2 * np.pi)
    print(f"  Numerical: {num_a:.6f}")

    # (b) Type 2: rational function
    print("\n(b) int_0^inf x^2/((x^2+1)(x^2+4)) dx")
    f_b = z**2 / ((z**2 + 1) * (z**2 + 4))
    # Poles in upper half: z=i, z=2i
    res_bi = sp.residue(f_b, z, sp.I)
    res_b2i = sp.residue(f_b, z, 2 * sp.I)
    # Full real line integral = 2*pi*i*sum(residues)
    full_line = sp.simplify(2 * sp.pi * sp.I * (res_bi + res_b2i))
    half = sp.simplify(full_line / 2)  # 0 to inf = half (even function)
    print(f"  Residue at z=i: {res_bi}")
    print(f"  Residue at z=2i: {res_b2i}")
    print(f"  int_{-inf}^{inf} = {full_line}")
    print(f"  int_0^inf = {half} = {float(half):.6f}")

    num_b, _ = quad(lambda x: x**2 / ((x**2 + 1) * (x**2 + 4)), 0, np.inf)
    print(f"  Numerical: {num_b:.6f}")

    # (c) Type 3: Fourier-type
    print("\n(c) int_0^inf cos(3x)/(x^2+1) dx")
    f_c = sp.exp(3 * sp.I * z) / (z**2 + 1)
    res_ci = sp.residue(f_c, z, sp.I)
    full_c = sp.simplify(2 * sp.pi * sp.I * res_ci)
    half_c = sp.simplify(sp.re(full_c) / 2)  # cos integral, half line
    print(f"  Residue at z=i: {res_ci}")
    print(f"  int_{-inf}^{inf} cos(3x)/(x^2+1) dx = Re[{full_c}]")
    print(f"  int_0^inf = {half_c} = {float(half_c):.8f}")
    print(f"  = pi/(2*e^3) = {np.pi / (2 * np.exp(3)):.8f}")

    num_c, _ = quad(lambda x: np.cos(3 * x) / (x**2 + 1), 0, np.inf)
    print(f"  Numerical: {num_c:.8f}")


def exercise_5():
    """
    Problem 5: Residues of z/((z-1)^2(z+2)) and integral over |z|=3.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Multiple Singularities")
    print("=" * 60)

    z = sp.Symbol('z')
    f = z / ((z - 1)**2 * (z + 2))

    print(f"\nf(z) = z/((z-1)^2(z+2))")
    print("Singularities: z=1 (order 2), z=-2 (simple)")

    res1 = sp.residue(f, z, 1)
    res2 = sp.residue(f, z, -2)
    total = res1 + res2
    integral = 2 * sp.pi * sp.I * total

    print(f"\nResidue at z=1: {res1}")
    print(f"  Method: lim d/dz[(z-1)^2 f(z)] = lim d/dz[z/(z+2)]")
    print(f"        = lim [1*(z+2) - z*1]/(z+2)^2 = 2/9")

    print(f"\nResidue at z=-2: {res2}")
    print(f"  Method: lim (z+2)*f(z) = lim z/(z-1)^2 = -2/9")

    print(f"\nSum of residues: {total}")
    print(f"Integral over |z|=3: {sp.simplify(integral)}")


def exercise_6():
    """
    Problem 6: Fluid mechanics - source at z=a, sink at z=-a,
    x-axis as wall. Find streamline function using method of images.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Method of Images (Fluid Mechanics)")
    print("=" * 60)

    print("\nSource of strength Q at z=a (above x-axis)")
    print("Sink of strength -Q at z=-a (above x-axis)")
    print("x-axis is a solid wall")
    print("\nImage system: source +Q at z=a_bar (below x-axis)")
    print("              sink -Q at z=-a_bar (below x-axis)")
    print("\nComplex potential for upper half-plane:")
    print("  W(z) = (Q/2pi)*[ln(z-a) - ln(z+a) + ln(z-a_bar) - ln(z+a_bar)]")
    print("\nFor a = id (purely imaginary, source on y-axis):")
    print("  W(z) = (Q/2pi)*ln[(z-id)(z+id) / ((z+id)(z-id))]")
    print("  Streamlines: psi = Im(W) = const")

    a = 1.0  # source at z=i, sink at z=-i
    Q = 1.0

    x = np.linspace(-3, 3, 400)
    y = np.linspace(0.01, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Source at z=ia, sink at z=-ia, images at z=-ia, z=ia
    z_src = 1j * a
    z_snk = -1j * a
    z_img_src = -1j * a  # image of source (reflected)
    z_img_snk = 1j * a   # image of sink (reflected)

    with np.errstate(divide='ignore', invalid='ignore'):
        W = (Q / (2 * np.pi)) * (np.log(Z - z_src) - np.log(Z - z_snk) +
                                   np.log(Z - z_img_src) - np.log(Z - z_img_snk))

    psi = np.imag(W)
    psi = np.clip(psi, -2, 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.contour(X, Y, psi, levels=30, colors='blue', linewidths=0.8)
    ax.axhline(0, color='k', linewidth=2)
    ax.plot(0, a, 'r+', markersize=15, markeredgewidth=3, label='source')
    ax.plot(0, -a, 'b_', markersize=15, markeredgewidth=3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamlines: Source + Sink with Wall (Method of Images)')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig('ex14_method_of_images.png', dpi=150)
    plt.close()
    print("Plot saved to ex14_method_of_images.png")


def exercise_7():
    """
    Problem 7: QM density of states from Green's function.
    """
    print("\n" + "=" * 60)
    print("Problem 7: Density of States from Green's Function")
    print("=" * 60)

    print("\nGreen's function: G(E) = 1/(E - H + i*epsilon)")
    print("For discrete spectrum H|n> = E_n|n>:")
    print("  G(E) = sum_n |n><n| / (E - E_n + i*epsilon)")
    print("\nDensity of states:")
    print("  rho(E) = -(1/pi) Im[Tr G(E)]")
    print("         = -(1/pi) sum_n Im[1/(E - E_n + i*eps)]")
    print("         = (1/pi) sum_n eps/((E-E_n)^2 + eps^2)")
    print("         -> sum_n delta(E - E_n)  as eps -> 0")
    print("\nThis is the spectral representation: each eigenvalue")
    print("contributes a delta function to the density of states.")

    # Numerical demonstration: harmonic oscillator
    E_n = np.arange(0.5, 10.5, 1.0)  # E_n = (n+1/2)*hbar*omega
    E = np.linspace(0, 11, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for eps, ax in zip([0.5, 0.05], axes):
        rho = np.zeros_like(E)
        for en in E_n:
            rho += (1 / np.pi) * eps / ((E - en)**2 + eps**2)
        ax.plot(E, rho, 'b-', linewidth=2)
        for en in E_n:
            ax.axvline(en, color='r', linestyle='--', alpha=0.3)
        ax.set_xlabel('E')
        ax.set_ylabel('rho(E)')
        ax.set_title(f'Density of States (eps={eps})')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Spectral Function from Green\'s Function', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex14_density_of_states.png', dpi=150)
    plt.close()
    print("Plot saved to ex14_density_of_states.png")


if __name__ == "__main__":
    print("=== Exercise 1 ===")
    exercise_1()
    print("\n=== Exercise 2 ===")
    exercise_2()
    print("\n=== Exercise 3 ===")
    exercise_3()
    print("\n=== Exercise 4 ===")
    exercise_4()
    print("\n=== Exercise 5 ===")
    exercise_5()
    print("\n=== Exercise 6 ===")
    exercise_6()
    print("\n=== Exercise 7 ===")
    exercise_7()
    print("\nAll exercises completed!")

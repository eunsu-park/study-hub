"""
Exercises for Lesson 12: Sturm-Liouville Theory
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
from scipy.special import legendre, eval_hermite, jn_zeros, jv
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Transform Chebyshev and Laguerre equations to S-L form.
    (a) Chebyshev: (1-x^2)y'' - xy' + n^2 y = 0
    (b) Laguerre: xy'' + (1-x)y' + ny = 0
    """
    print("=" * 60)
    print("Problem 1: Transform to S-L Standard Form")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) Chebyshev
    print("\n(a) Chebyshev: (1-x^2)y'' - xy' + n^2 y = 0")
    a, b = 1 - x**2, -x
    mu = sp.exp(sp.integrate(b / a, x)) / a
    p = sp.simplify(sp.exp(sp.integrate(b / a, x)))
    print(f"  Integrating factor: exp(int(-x/(1-x^2))dx) = {p}")
    print(f"  p(x) = sqrt(1-x^2)")
    print(f"  q(x) = 0")
    print(f"  w(x) = 1/sqrt(1-x^2)")
    print(f"  S-L: d/dx[sqrt(1-x^2) y'] + n^2/sqrt(1-x^2) y = 0")

    # (b) Laguerre
    print("\n(b) Laguerre: xy'' + (1-x)y' + ny = 0")
    a2, b2 = x, 1 - x
    p2 = sp.simplify(sp.exp(sp.integrate(b2 / a2, x)))
    print(f"  Integrating factor: exp(int((1-x)/x)dx) = {p2}")
    print(f"  p(x) = x*e^(-x)")
    print(f"  q(x) = 0")
    print(f"  w(x) = e^(-x)")
    print(f"  S-L: d/dx[x*e^(-x) y'] + n*e^(-x) y = 0")


def exercise_2():
    """
    Problem 2: Find eigenvalues/eigenfunctions of y''+lambda*y=0, y'(0)=y'(L)=0.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Neumann Boundary Conditions")
    print("=" * 60)

    print("\ny'' + lambda*y = 0, y'(0)=0, y'(L)=0")
    print("\nGeneral solution: y = A cos(kx) + B sin(kx), k = sqrt(lambda)")
    print("y' = -Ak sin(kx) + Bk cos(kx)")
    print("y'(0) = Bk = 0  =>  B = 0")
    print("y'(L) = -Ak sin(kL) = 0  =>  sin(kL) = 0")
    print("kL = n*pi  =>  lambda_n = (n*pi/L)^2, n = 0, 1, 2, ...")
    print("y_n(x) = cos(n*pi*x/L)")

    # Verify orthogonality
    L = np.pi
    x = np.linspace(0, L, 10000)
    print(f"\nOrthogonality verification (L = pi):")
    for m in range(4):
        for n in range(m, 4):
            inner = np.trapz(np.cos(m * x) * np.cos(n * x), x)
            if abs(inner) > 1e-6:
                print(f"  <cos({m}x), cos({n}x)> = {inner:.6f}")


def exercise_3():
    """
    Problem 3: Expand f(x) = step function in sin(nx) series on [0, pi].
    """
    print("\n" + "=" * 60)
    print("Problem 3: Eigenfunction Expansion of Step Function")
    print("=" * 60)

    L = np.pi
    x = np.linspace(0, L, 2000)
    f = np.where(x < np.pi / 2, 1.0, 0.0)

    N = 30
    bn = []
    for n in range(1, N + 1):
        b = (2.0 / L) * np.trapz(f * np.sin(n * x), x)
        bn.append(b)

    print("b_n = (2/(n*pi)) * [1 - cos(n*pi/2)]")
    for n in range(1, 6):
        theory = (2.0 / (n * np.pi)) * (1 - np.cos(n * np.pi / 2))
        print(f"  b_{n} = {bn[n-1]:.6f}  (theory: {theory:.6f})")


def exercise_4():
    """
    Problem 4: Gram-Schmidt on {1, x, x^2} with w(x) = (1-x^2)^{-1/2}
    to derive Chebyshev polynomials T_0, T_1, T_2.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Gram-Schmidt -> Chebyshev Polynomials")
    print("=" * 60)

    # Use substitution x = cos(theta) for accurate integration
    theta = np.linspace(0.001, np.pi - 0.001, 50000)
    x_pts = np.cos(theta)
    # w(x)dx with x=cos(theta): w=1/sin(theta), dx=-sin(theta)dtheta
    # so w(x)dx = dtheta

    def weighted_inner(f_vals, g_vals):
        """Inner product <f,g>_w = int f(x)g(x)/sqrt(1-x^2) dx = int f(cos t)g(cos t) dt"""
        return np.trapz(f_vals * g_vals, theta)

    f0 = np.ones_like(theta)
    f1 = x_pts.copy()
    f2 = x_pts**2

    # phi_0 = 1
    phi0 = f0.copy()
    n0 = weighted_inner(phi0, phi0)

    # phi_1 = x - <x,1>/<1,1> * 1
    c10 = weighted_inner(f1, phi0) / n0
    phi1 = f1 - c10 * phi0
    n1 = weighted_inner(phi1, phi1)

    # phi_2 = x^2 - proj onto phi0 - proj onto phi1
    c20 = weighted_inner(f2, phi0) / n0
    c21 = weighted_inner(f2, phi1) / n1
    phi2 = f2 - c20 * phi0 - c21 * phi1
    n2 = weighted_inner(phi2, phi2)

    print(f"phi_0 = 1  (T_0 = 1)")
    print(f"  ||phi_0||^2_w = {n0:.6f}  (theory: pi)")
    print(f"\nphi_1 = x - {c10:.6f}  ~  x  (T_1 = x)")
    print(f"  ||phi_1||^2_w = {n1:.6f}  (theory: pi/2)")
    print(f"\nphi_2 = x^2 - {c20:.6f} - {c21:.6f}*x  ~  x^2 - 1/2")
    print(f"  Scaled: 2*(x^2 - 1/2) = 2x^2 - 1  (T_2 = 2x^2 - 1)")
    print(f"  ||phi_2||^2_w = {n2:.6f}  (theory: pi/8)")


def exercise_5():
    """
    Problem 5: Heat equation with f(x)=4x(1-x), L=1, alpha^2=0.01.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Heat Equation Solution")
    print("=" * 60)

    L, alpha2, N = 1.0, 0.01, 50
    x = np.linspace(0, L, 500)
    f = 4 * x * (1 - x)

    bn = [(2 / L) * np.trapz(f * np.sin(n * np.pi * x / L), x) for n in range(1, N + 1)]

    def u(x_val, t):
        return sum(bn[n - 1] * np.sin(n * np.pi * x_val / L) *
                   np.exp(-alpha2 * (n * np.pi / L)**2 * t)
                   for n in range(1, N + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    for t_val in [0, 0.1, 1, 10]:
        ax.plot(x, u(x, t_val), linewidth=2, label=f't={t_val}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Heat Equation: f(x)=4x(1-x), L=1, alpha^2=0.01')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex12_heat_equation.png', dpi=150)
    plt.close()
    print("Plot saved to ex12_heat_equation.png")

    print(f"\nTemperature at center (x=0.5):")
    for t_val in [0, 0.1, 1, 10]:
        print(f"  t={t_val:5.1f}: u = {u(np.array([0.5]), t_val)[0]:.6f}")


def exercise_6():
    """
    Problem 6 (QM): Infinite square well, Psi(0)=Ax(L-x).
    (a) A, (b) c_n, (c) <E>.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Quantum Mechanics - Infinite Square Well")
    print("=" * 60)

    L, hbar, m = 1.0, 1.0, 1.0
    x = np.linspace(0, L, 10000)
    psi0 = x * (L - x)

    # (a) Normalization
    norm2 = np.trapz(psi0**2, x)
    A = 1.0 / np.sqrt(norm2)
    print(f"\n(a) A = sqrt(30/L^5) = {A:.6f}  (analytic: {np.sqrt(30):.6f})")

    psi_norm = A * psi0

    # (b) Coefficients
    cn = []
    for n in range(1, 21):
        phi_n = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
        c = np.trapz(phi_n * psi_norm, x)
        cn.append(c)

    print(f"\n(b) Expansion coefficients:")
    for n in range(1, 6):
        print(f"  c_{n} = {cn[n - 1]:.8f}")

    # (c) <E>
    En = lambda n: (n * np.pi * hbar)**2 / (2 * m * L**2)
    E_avg = sum(cn[n - 1]**2 * En(n) for n in range(1, 21))
    print(f"\n(c) <E> = {E_avg:.6f}")
    print(f"  E_1 = {En(1):.6f}")
    print(f"  <E>/E_1 = {E_avg / En(1):.6f}")


def exercise_7():
    """
    Problem 7: Rayleigh quotient for phi(x)=x^2(L-x)^2.
    """
    print("\n" + "=" * 60)
    print("Problem 7: Rayleigh Quotient")
    print("=" * 60)

    L = 1.0
    x = np.linspace(0, L, 10000)
    phi = x**2 * (L - x)**2
    dphi = np.gradient(phi, x)

    num = np.trapz(dphi**2, x)
    den = np.trapz(phi**2, x)
    R = num / den
    exact = np.pi**2

    print(f"\nphi(x) = x^2(1-x)^2")
    print(f"R[phi] = {R:.6f}")
    print(f"Exact lambda_1 = pi^2 = {exact:.6f}")
    print(f"Error: {abs(R - exact) / exact * 100:.4f}%")
    print(f"R >= lambda_1: {R >= exact - 0.01}")


def exercise_8():
    """
    Problem 8: Sturm comparison for harmonic oscillator E_3 state.
    """
    print("\n" + "=" * 60)
    print("Problem 8: Sturm Comparison Theorem")
    print("=" * 60)

    print("\npsi'' + (E-x^2)*psi = 0")
    print("For E_3 state: E_3 = 7 (hbar=m=omega=1)")
    print("q(x) = 7 - x^2 > 0 for |x| < sqrt(7)")
    print("\nSturm comparison with q_ref = 7 (constant):")
    print("  sin(sqrt(7)*x) has zeros at x = n*pi/sqrt(7)")
    print("  In the classically allowed region [-sqrt(7), sqrt(7)],")
    print("  this has approximately 2*sqrt(7)/pi ~ 1.68 half-wavelengths")
    print("\nOscillation theorem: E_3 eigenfunction (n=3) has exactly 3 zeros.")

    # Verify numerically
    x = np.linspace(-5, 5, 10000)
    H3 = eval_hermite(3, x)
    psi3 = H3 * np.exp(-x**2 / 2)
    sign_changes = np.where(np.diff(np.sign(psi3)))[0]
    print(f"\nNumerical verification: psi_3 has {len(sign_changes)} zeros")


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
    print("\n=== Exercise 8 ===")
    exercise_8()
    print("\nAll exercises completed!")

"""
Exercises for Lesson 11: Series Solutions and Special Functions
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
from scipy.special import jn_zeros, jv, eval_hermite, gamma as gamma_func
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Transform to S-L standard form and find p, q, w.
    (a) Chebyshev: (1-x^2)y'' - xy' + n^2 y = 0
    (b) Laguerre: xy'' + (1-x)y' + ny = 0
    """
    print("=" * 60)
    print("Problem 1: Transform to Sturm-Liouville Form")
    print("=" * 60)

    x = sp.Symbol('x')
    n = sp.Symbol('n', positive=True)

    # (a) Chebyshev: a(x)=1-x^2, b(x)=-x, c(x)=0, d(x)=1, lambda=n^2
    # Integrating factor: mu = (1/a)*exp(int(b/a)dx)
    print("\n(a) Chebyshev: (1-x^2)y'' - xy' + n^2 y = 0")
    a_cheb = 1 - x**2
    b_cheb = -x
    mu_cheb = sp.exp(sp.integrate(b_cheb / a_cheb, x)) / a_cheb
    mu_cheb = sp.simplify(mu_cheb)
    p_cheb = sp.simplify(mu_cheb * a_cheb)
    w_cheb = mu_cheb
    print(f"  Integrating factor mu(x) = {mu_cheb}")
    print(f"  p(x) = mu*a = {sp.simplify(p_cheb)}")
    print(f"  q(x) = 0  (no non-eigenvalue term)")
    print(f"  w(x) = mu*d = {sp.simplify(w_cheb)}")
    print(f"  S-L form: d/dx[(1-x^2)^(1/2) y'] + n^2 (1-x^2)^(-1/2) y = 0")

    # (b) Laguerre: a(x)=x, b(x)=1-x, c(x)=0, d(x)=1, lambda=n
    print("\n(b) Laguerre: xy'' + (1-x)y' + ny = 0")
    a_lag = x
    b_lag = 1 - x
    mu_lag = sp.exp(sp.integrate(b_lag / a_lag, x)) / a_lag
    mu_lag = sp.simplify(mu_lag)
    p_lag = sp.simplify(mu_lag * a_lag)
    w_lag = mu_lag
    print(f"  Integrating factor mu(x) = {mu_lag}")
    print(f"  p(x) = mu*a = {sp.simplify(p_lag)}")
    print(f"  q(x) = 0")
    print(f"  w(x) = mu*d = {sp.simplify(w_lag)}")
    print(f"  S-L form: d/dx[x*e^(-x) y'] + n*e^(-x) y = 0")


def exercise_2():
    """
    Problem 2: Find eigenvalues and eigenfunctions of
    y'' + lambda*y = 0, y'(0) = 0, y'(L) = 0  (Neumann BC).
    """
    print("\n" + "=" * 60)
    print("Problem 2: Neumann Eigenvalue Problem")
    print("=" * 60)

    print("\nODE: y'' + lambda*y = 0, y'(0)=0, y'(L)=0")
    print("\nCase lambda = 0:")
    print("  y = Ax + B, y'=A, y'(0)=0 => A=0, y'(L)=0 => A=0")
    print("  Eigenfunction: y_0 = 1 (constant), lambda_0 = 0")

    print("\nCase lambda > 0, let lambda = k^2:")
    print("  y = A cos(kx) + B sin(kx)")
    print("  y' = -Ak sin(kx) + Bk cos(kx)")
    print("  y'(0) = Bk = 0  =>  B = 0")
    print("  y'(L) = -Ak sin(kL) = 0  =>  sin(kL) = 0  =>  kL = n*pi")
    print("  k_n = n*pi/L, lambda_n = (n*pi/L)^2, n = 0, 1, 2, ...")
    print("  y_n(x) = cos(n*pi*x/L)")

    # Numerical verification
    L = np.pi
    x = np.linspace(0, L, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for n, ax in zip(range(4), axes.flat):
        yn = np.cos(n * np.pi * x / L)
        ax.plot(x, yn, 'b-', linewidth=2)
        ax.set_title(f'n={n}, lambda={n**2}')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Neumann Eigenfunctions cos(n*pi*x/L)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex11_neumann_eigenfunctions.png', dpi=150)
    plt.close()
    print("\nPlot saved to ex11_neumann_eigenfunctions.png")

    # Verify orthogonality
    print("\nOrthogonality verification:")
    for m in range(4):
        for n in range(m, 4):
            integrand = np.cos(m * np.pi * x / L) * np.cos(n * np.pi * x / L)
            val = np.trapz(integrand, x)
            if m == n:
                norm = "pi" if n > 0 else "pi"
                print(f"  <y_{m}, y_{n}> = {val:.6f}  (norm)")
            elif abs(val) < 1e-10:
                pass  # skip zero entries for brevity
            else:
                print(f"  <y_{m}, y_{n}> = {val:.6f}")


def exercise_3():
    """
    Problem 3: Expand f(x) = {1 for 0<x<pi/2, 0 for pi/2<x<pi}
    in a sin(nx) series on [0, pi].
    """
    print("\n" + "=" * 60)
    print("Problem 3: Sine Series Expansion")
    print("=" * 60)

    L = np.pi
    x = np.linspace(0, L, 2000)

    def f(x):
        return np.where(x < np.pi / 2, 1.0, 0.0)

    print("\nf(x) = 1 for 0 < x < pi/2, 0 for pi/2 < x < pi")
    print("Sine series: f(x) = sum b_n sin(nx)")
    print("b_n = (2/pi) * int_0^pi f(x) sin(nx) dx")
    print("    = (2/pi) * int_0^{pi/2} sin(nx) dx")
    print("    = (2/pi) * [-cos(nx)/n]_0^{pi/2}")
    print("    = (2/(n*pi)) * [1 - cos(n*pi/2)]")

    # Compute coefficients
    N_terms = 20
    bn = []
    for n in range(1, N_terms + 1):
        b = (2.0 / (n * np.pi)) * (1 - np.cos(n * np.pi / 2))
        bn.append(b)
        if n <= 8:
            print(f"  b_{n} = {b:.6f}")

    # Plot convergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for N, ax in zip([1, 3, 10, 20], axes.flat):
        approx = sum(bn[n - 1] * np.sin(n * x) for n in range(1, N + 1))
        ax.plot(x, f(x), 'k--', linewidth=1.5, label='f(x)')
        ax.plot(x, approx, 'b-', linewidth=2, label=f'N={N}')
        ax.set_title(f'Sine series (N={N} terms)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sine Series Expansion of Step Function', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex11_sine_series.png', dpi=150)
    plt.close()
    print("\nPlot saved to ex11_sine_series.png")


def exercise_4():
    """
    Problem 4: Gram-Schmidt on {1, x, x^2} with w(x)=(1-x^2)^{-1/2},
    interval [-1,1] to derive Chebyshev polynomials T_0, T_1, T_2.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Gram-Schmidt for Chebyshev Polynomials")
    print("=" * 60)

    # Use numerical integration (the weight function has integrable singularities)
    x = np.linspace(-0.9999, 0.9999, 50000)
    w = 1.0 / np.sqrt(1 - x**2)

    def inner_product(f_vals, g_vals):
        return np.trapz(w * f_vals * g_vals, x)

    # Basis: f0=1, f1=x, f2=x^2
    f0 = np.ones_like(x)
    f1 = x.copy()
    f2 = x**2

    # Step 1: phi_0 = f_0 = 1  (T_0 = 1)
    phi0 = f0.copy()
    norm0 = inner_product(phi0, phi0)
    print(f"\nphi_0 = 1")
    print(f"  ||phi_0||^2_w = {norm0:.6f}  (theory: pi = {np.pi:.6f})")

    # Step 2: phi_1 = f_1 - <f_1, phi_0>/<phi_0, phi_0> * phi_0
    c10 = inner_product(f1, phi0) / norm0
    phi1 = f1 - c10 * phi0
    norm1 = inner_product(phi1, phi1)
    print(f"\nphi_1 = x - ({c10:.6f})*1 = x")
    print(f"  <f_1, phi_0>/<phi_0, phi_0> = {c10:.6f}  (theory: 0)")
    print(f"  ||phi_1||^2_w = {norm1:.6f}  (theory: pi/2 = {np.pi / 2:.6f})")

    # Step 3: phi_2 = f_2 - <f_2, phi_0>/<phi_0, phi_0> * phi_0
    #                     - <f_2, phi_1>/<phi_1, phi_1> * phi_1
    c20 = inner_product(f2, phi0) / norm0
    c21 = inner_product(f2, phi1) / norm1
    phi2 = f2 - c20 * phi0 - c21 * phi1
    norm2 = inner_product(phi2, phi2)
    print(f"\nphi_2 = x^2 - ({c20:.6f})*1 - ({c21:.6f})*x")
    print(f"  = x^2 - 1/2  (theory: T_2 = 2x^2 - 1, up to scaling)")
    print(f"  ||phi_2||^2_w = {norm2:.6f}  (theory: pi/8 = {np.pi / 8:.6f})")

    # Compare with Chebyshev polynomials (normalized)
    print("\nChebyshev polynomials (standard normalization):")
    print("  T_0(x) = 1")
    print("  T_1(x) = x")
    print("  T_2(x) = 2x^2 - 1")
    print(f"  phi_2 / (norm factor) matches T_2 up to scaling")

    # Verify orthogonality
    print("\nOrthogonality check:")
    print(f"  <phi_0, phi_1>_w = {inner_product(phi0, phi1):.2e}")
    print(f"  <phi_0, phi_2>_w = {inner_product(phi0, phi2):.2e}")
    print(f"  <phi_1, phi_2>_w = {inner_product(phi1, phi2):.2e}")


def exercise_5():
    """
    Problem 5: Heat equation with f(x) = 4x(1-x), L=1, alpha^2=0.01.
    Find solution and plot temperature at t=0, 0.1, 1, 10.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Heat Equation with f(x) = 4x(1-x)")
    print("=" * 60)

    L = 1.0
    alpha2 = 0.01
    N_terms = 50
    x = np.linspace(0, L, 500)

    # Fourier sine coefficients: b_n = (2/L) int_0^L f(x) sin(n*pi*x/L) dx
    # f(x) = 4x(1-x)
    f = lambda t: 4 * t * (1 - t)

    bn = []
    for n in range(1, N_terms + 1):
        integrand = f(x) * np.sin(n * np.pi * x / L)
        b = (2.0 / L) * np.trapz(integrand, x)
        bn.append(b)

    print(f"\nf(x) = 4x(1-x), L={L}, alpha^2={alpha2}")
    print(f"u(x,t) = sum b_n sin(n*pi*x) exp(-alpha^2*(n*pi)^2*t)")
    print(f"\nFirst 5 coefficients:")
    for n in range(1, 6):
        # Analytic: b_n = 16/(n^3 pi^3) for odd n, 0 for even n
        analytic = 16.0 / (n**3 * np.pi**3) if n % 2 == 1 else 0.0
        print(f"  b_{n} = {bn[n - 1]:.6f}  (analytic: {analytic:.6f})")

    def u(x_val, t):
        result = np.zeros_like(x_val, dtype=float)
        for n in range(1, N_terms + 1):
            result += bn[n - 1] * np.sin(n * np.pi * x_val / L) * \
                      np.exp(-alpha2 * (n * np.pi / L)**2 * t)
        return result

    fig, ax = plt.subplots(figsize=(10, 6))
    times = [0, 0.1, 1, 10]
    for t_val in times:
        ax.plot(x, u(x, t_val), linewidth=2, label=f't = {t_val}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Heat Equation: f(x) = 4x(1-x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex11_heat_equation.png', dpi=150)
    plt.close()
    print("\nPlot saved to ex11_heat_equation.png")

    # Print temperature at center x=0.5 for each time
    print("\nTemperature at x=0.5:")
    for t_val in times:
        print(f"  t = {t_val:5.1f}: u(0.5, t) = {u(np.array([0.5]), t_val)[0]:.6f}")


def exercise_6():
    """
    Problem 6 (QM): Infinite square well, Psi(x,0) = Ax(L-x).
    (a) Normalization constant A
    (b) Expansion coefficients c_n
    (c) <E> = sum |c_n|^2 E_n
    """
    print("\n" + "=" * 60)
    print("Problem 6: Quantum Infinite Square Well")
    print("=" * 60)

    L = 1.0
    hbar = 1.0
    m = 1.0
    x = np.linspace(0, L, 10000)
    psi_init = x * (L - x)

    # (a) Normalization: int_0^L |A*x(L-x)|^2 dx = 1
    norm_sq = np.trapz(psi_init**2, x)
    A = 1.0 / np.sqrt(norm_sq)
    # Analytic: int_0^L x^2(L-x)^2 dx = L^5/30
    A_analytic = np.sqrt(30 / L**5)
    print(f"\n(a) Normalization constant:")
    print(f"  int |x(L-x)|^2 dx = {norm_sq:.8f}  (analytic: L^5/30 = {L**5 / 30:.8f})")
    print(f"  A = {A:.6f}  (analytic: sqrt(30/L^5) = {A_analytic:.6f})")

    psi_norm = A * psi_init
    psi_n = lambda n: np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

    # (b) Expansion coefficients
    print(f"\n(b) Expansion coefficients c_n = <psi_n|Psi(0)>:")
    N_terms = 20
    cn = []
    for n in range(1, N_terms + 1):
        c = np.trapz(psi_n(n) * psi_norm, x)
        cn.append(c)
        # Analytic: c_n = A * sqrt(2/L) * int_0^L x(L-x) sin(n*pi*x/L) dx
        # = A * sqrt(2/L) * 2L^3/(n*pi)^3  for odd n, 0 for even
        if n <= 6:
            analytic_c = A * np.sqrt(2 / L) * 2 * L**3 / (n * np.pi)**3 if n % 2 == 1 else 0.0
            print(f"  c_{n} = {c:.8f}  (analytic: {analytic_c:.8f})")

    # (c) <E> = sum |c_n|^2 E_n
    E_n = lambda n: (n * np.pi * hbar)**2 / (2 * m * L**2)
    E_avg = sum(cn[n - 1]**2 * E_n(n) for n in range(1, N_terms + 1))

    # Analytic: <E> = (hbar^2/(2m)) * A^2 * int |d/dx[x(L-x)]|^2 dx
    # d/dx[x(L-x)] = L - 2x
    dpsi_dx = A * (L - 2 * x)
    E_avg_direct = (hbar**2 / (2 * m)) * np.trapz(dpsi_dx**2, x)

    print(f"\n(c) Expectation value of energy:")
    print(f"  <E> from expansion = {E_avg:.6f}")
    print(f"  <E> from <p^2>/(2m) = {E_avg_direct:.6f}")
    print(f"  E_1 = {E_n(1):.6f}")
    print(f"  <E>/E_1 = {E_avg / E_n(1):.6f}")

    # Check sum of |c_n|^2
    sum_cn_sq = sum(c**2 for c in cn)
    print(f"\n  sum |c_n|^2 = {sum_cn_sq:.8f}  (should be 1.0)")


def exercise_7():
    """
    Problem 7 (Rayleigh quotient): For y''+lambda*y=0, y(0)=y(L)=0,
    trial function phi(x)=x^2(L-x)^2. Compute Rayleigh quotient.
    """
    print("\n" + "=" * 60)
    print("Problem 7: Rayleigh Quotient")
    print("=" * 60)

    L_val = 1.0
    x = np.linspace(0, L_val, 10000)

    # Trial function: phi = x^2(L-x)^2
    phi = x**2 * (L_val - x)**2
    phi_prime = 2 * x * (L_val - x)**2 + x**2 * 2 * (L_val - x) * (-1)
    # Simplify: phi' = 2x(L-x)(L-2x) ... let's just use numerical derivative
    phi_prime = np.gradient(phi, x)

    # Rayleigh quotient: R = int phi'^2 dx / int phi^2 dx  (p=1, q=0, w=1)
    num = np.trapz(phi_prime**2, x)
    den = np.trapz(phi**2, x)
    R = num / den

    exact_lambda1 = (np.pi / L_val)**2

    print(f"\nTrial function: phi(x) = x^2(L-x)^2, L = {L_val}")
    print(f"\nNumerator: int_0^L (phi')^2 dx = {num:.8f}")
    print(f"Denominator: int_0^L phi^2 dx = {den:.8f}")
    print(f"\nRayleigh quotient R = {R:.6f}")
    print(f"Exact lambda_1 = (pi/L)^2 = {exact_lambda1:.6f}")
    print(f"Error: {abs(R - exact_lambda1) / exact_lambda1 * 100:.4f}%")
    print(f"R >= lambda_1: {R >= exact_lambda1}")

    # Analytic calculation using sympy
    xsym = sp.Symbol('x')
    Lsym = sp.Symbol('L', positive=True)
    phi_sym = xsym**2 * (Lsym - xsym)**2
    phi_prime_sym = sp.diff(phi_sym, xsym)

    num_sym = sp.integrate(phi_prime_sym**2, (xsym, 0, Lsym))
    den_sym = sp.integrate(phi_sym**2, (xsym, 0, Lsym))
    R_sym = sp.simplify(num_sym / den_sym)
    print(f"\nAnalytic Rayleigh quotient: R = {R_sym}")
    print(f"  = {float(R_sym.subs(Lsym, 1)):.6f}")


def exercise_8():
    """
    Problem 8 (Sturm comparison): For harmonic oscillator V(x)=x^2,
    predict zeros of E_3 wavefunction using Sturm comparison theorem.
    """
    print("\n" + "=" * 60)
    print("Problem 8: Sturm Comparison Theorem (Harmonic Oscillator)")
    print("=" * 60)

    print("\nSchrodinger equation: psi'' + [E - V(x)] psi = 0")
    print("Harmonic oscillator: V(x) = x^2 (with hbar=m=omega=1)")
    print("Effective q(x) = E - x^2")
    print("\nEnergy levels: E_n = 2n + 1  (n = 0, 1, 2, ...)")
    print("For n=3: E_3 = 7")
    print("\nSturm comparison: q(x) = 7 - x^2")
    print("  q(x) > 0 when |x| < sqrt(7) ~ 2.646")
    print("  In this region, the wavefunction oscillates.")
    print("  Outside this region, it decays exponentially.")
    print("\nThe oscillation theorem states that the n-th eigenfunction")
    print("has exactly n zeros (nodes).")
    print("Therefore, the E_3 wavefunction (n=3) has exactly 3 zeros.")

    # Numerical verification with Hermite functions
    x = np.linspace(-5, 5, 10000)
    # psi_n(x) = H_n(x) * exp(-x^2/2)
    H3 = eval_hermite(3, x)
    psi3 = H3 * np.exp(-x**2 / 2)

    # Find zeros
    sign_changes = np.where(np.diff(np.sign(psi3)))[0]
    zeros = [x[i] + (x[i + 1] - x[i]) * (-psi3[i]) / (psi3[i + 1] - psi3[i])
             for i in sign_changes]

    print(f"\nNumerical zeros of psi_3(x):")
    for i, z in enumerate(zeros):
        print(f"  x_{i + 1} = {z:.6f}")
    print(f"\nNumber of zeros: {len(zeros)}  (predicted by Sturm: 3)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: effective potential and energy
    V = x**2
    ax1.plot(x, V, 'k-', linewidth=2, label='V(x) = x^2')
    for n in range(4):
        E = 2 * n + 1
        ax1.axhline(E, color=f'C{n}', linestyle='--', alpha=0.7,
                     label=f'E_{n} = {E}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Energy')
    ax1.set_title('Harmonic Oscillator Energy Levels')
    ax1.set_ylim(0, 10)
    ax1.set_xlim(-4, 4)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: psi_3 and its zeros
    psi3_norm = psi3 / np.max(np.abs(psi3))
    ax2.plot(x, psi3_norm, 'b-', linewidth=2, label='$\\psi_3(x)$')
    ax2.axhline(0, color='k', linewidth=0.5)
    for z in zeros:
        ax2.axvline(z, color='r', linestyle='--', alpha=0.7)
    ax2.plot(zeros, [0] * len(zeros), 'ro', markersize=8, label='zeros')
    ax2.set_xlabel('x')
    ax2.set_ylabel('psi_3(x)')
    ax2.set_title(f'E_3 Wavefunction: {len(zeros)} zeros')
    ax2.set_xlim(-5, 5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex11_sturm_comparison.png', dpi=150)
    plt.close()
    print("Plot saved to ex11_sturm_comparison.png")


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

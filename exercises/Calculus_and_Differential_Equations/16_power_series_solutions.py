"""
Exercise Solutions: Lesson 16 - Power Series Solutions
Calculus and Differential Equations

Topics covered:
- Singular point classification
- Power series solution with recurrence relation
- Frobenius method for Euler equation
- Bessel function J_1 computation
- Legendre polynomial derivation
"""

import numpy as np
import sympy as sp
from scipy import special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Singular Point Classification
# ============================================================
def exercise_1():
    """
    (1+x^2)y'' + 2xy' + 4y = 0.
    Identify singular points, classify, find radius of convergence at x=0.
    """
    print("=" * 60)
    print("Problem 1: Singular Point Classification")
    print("=" * 60)

    x = sp.Symbol('x')

    # Standard form: y'' + P(x)*y' + Q(x)*y = 0
    # P(x) = 2x/(1+x^2), Q(x) = 4/(1+x^2)
    print(f"\n  (1+x^2)y'' + 2xy' + 4y = 0")
    print(f"  Standard form: y'' + [2x/(1+x^2)]y' + [4/(1+x^2)]y = 0")
    print(f"  P(x) = 2x/(1+x^2), Q(x) = 4/(1+x^2)")

    # Singular points: where 1+x^2 = 0 => x = +/- i
    print(f"\n  Singular points: 1+x^2 = 0 => x = +i, x = -i")
    print(f"  (Complex singular points -- no real singular points)")

    # Classification at x = i:
    # (x-i)*P(x) = (x-i)*2x/((x-i)(x+i)) = 2x/(x+i) -- analytic at x=i
    # (x-i)^2*Q(x) = (x-i)^2*4/((x-i)(x+i)) = 4(x-i)/(x+i) -- analytic at x=i
    # => x = i is a REGULAR singular point (same for x = -i)
    print(f"\n  At x = i:")
    print(f"  (x-i)*P(x) = 2x/(x+i) -- analytic at x=i => regular")
    print(f"  (x-i)^2*Q(x) = 4(x-i)/(x+i) -- analytic at x=i => regular")
    print(f"  x = +/-i are REGULAR singular points")

    # Radius of convergence at x_0 = 0
    # = distance to nearest singular point = |i - 0| = 1
    print(f"\n  Radius of convergence at x_0 = 0:")
    print(f"  = distance to nearest singular point = |i| = 1")
    print(f"  The power series solution is guaranteed to converge for |x| < 1")


# ============================================================
# Problem 2: Series Solution
# ============================================================
def exercise_2():
    """
    y'' + x*y' + y = 0, y(0)=1, y'(0)=0.
    Find first 6 nonzero terms, write recurrence.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Series Solution")
    print("=" * 60)

    print(f"\n  y'' + x*y' + y = 0, y(0)=1, y'(0)=0")
    print(f"\n  Assume y = sum_{{n=0}}^inf a_n * x^n")
    print(f"  y' = sum a_n*n*x^(n-1)")
    print(f"  y'' = sum a_n*n*(n-1)*x^(n-2)")
    print(f"\n  Substituting and collecting powers of x^n:")
    print(f"  a_(n+2)*(n+2)*(n+1) + a_n*n + a_n = 0")
    print(f"  => a_(n+2) = -a_n*(n+1) / ((n+2)*(n+1)) = -a_n / (n+2)")
    print(f"\n  Recurrence relation: a_(n+2) = -a_n / (n+2)")

    # Initial conditions: a_0 = y(0) = 1, a_1 = y'(0) = 0
    # Since a_1 = 0, all odd coefficients are 0
    a = [0] * 14
    a[0] = 1  # y(0) = 1
    a[1] = 0  # y'(0) = 0

    print(f"\n  a_0 = 1, a_1 = 0")
    for n in range(0, 12):
        a[n + 2] = -a[n] / (n + 2) if n + 2 < len(a) else 0

    print(f"\n  Computing coefficients:")
    for n in range(14):
        if a[n] != 0:
            print(f"    a_{n} = {sp.Rational(a[n]).limit_denominator(10000)}")

    # First 6 nonzero terms (only even terms are nonzero)
    x = sp.Symbol('x')
    y_approx = sum(sp.Rational(a[n]).limit_denominator(100000) * x**n for n in range(14))
    print(f"\n  y(x) = {y_approx} + ...")

    # Numerical verification with SymPy
    x_sym = sp.Symbol('x')
    y_func = sp.Function('y')
    ode = y_func(x_sym).diff(x_sym, 2) + x_sym*y_func(x_sym).diff(x_sym) + y_func(x_sym)
    sol = sp.dsolve(ode, y_func(x_sym), ics={y_func(0): 1, y_func(x_sym).diff(x_sym).subs(x_sym, 0): 0})
    series_check = sp.series(sol.rhs, x_sym, 0, 13)
    print(f"\n  SymPy series check: {series_check}")


# ============================================================
# Problem 3: Frobenius Method (Euler Equation)
# ============================================================
def exercise_3():
    """
    x^2*y'' + 3x*y' + y = 0. Frobenius method.
    Indicial equation. Verify with exact x^r solutions.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Frobenius Method (Euler Equation)")
    print("=" * 60)

    r, x = sp.symbols('r x')

    print(f"\n  x^2*y'' + 3x*y' + y = 0")
    print(f"  This is a Cauchy-Euler (equidimensional) equation.")
    print(f"  x = 0 is a regular singular point.")

    # Frobenius: y = x^r * sum a_n * x^n
    # For Euler equation, try y = x^r:
    # x^2 * r(r-1)x^(r-2) + 3x * r*x^(r-1) + x^r = 0
    # r(r-1) + 3r + 1 = 0
    # r^2 + 2r + 1 = 0
    # (r + 1)^2 = 0 => r = -1 (repeated root)

    indicial = r*(r-1) + 3*r + 1
    indicial_simplified = sp.expand(indicial)
    roots = sp.solve(indicial_simplified, r)

    print(f"\n  Indicial equation (substitute y = x^r):")
    print(f"  r(r-1) + 3r + 1 = 0")
    print(f"  {indicial_simplified} = 0")
    print(f"  (r+1)^2 = 0")
    print(f"  Roots: r = {roots} (repeated)")

    # Solutions:
    # y1 = x^(-1) = 1/x
    # y2 = x^(-1) * ln(x) = ln(x)/x (from the repeated root Frobenius theory)
    print(f"\n  For repeated root r = -1:")
    print(f"    y1 = x^(-1) = 1/x")
    print(f"    y2 = x^(-1) * ln(x) = ln(x)/x")
    print(f"  General solution: y = C1/x + C2*ln(x)/x")

    # Verification
    x_sym = sp.Symbol('x', positive=True)
    y1 = 1/x_sym
    y2 = sp.ln(x_sym)/x_sym

    lhs1 = x_sym**2*sp.diff(y1, x_sym, 2) + 3*x_sym*sp.diff(y1, x_sym) + y1
    lhs2 = x_sym**2*sp.diff(y2, x_sym, 2) + 3*x_sym*sp.diff(y2, x_sym) + y2

    print(f"\n  Verification:")
    print(f"    Substituting y1 = 1/x: LHS = {sp.simplify(lhs1)}")
    print(f"    Substituting y2 = ln(x)/x: LHS = {sp.simplify(lhs2)}")

    # SymPy dsolve
    y_func = sp.Function('y')
    ode = x_sym**2*y_func(x_sym).diff(x_sym, 2) + 3*x_sym*y_func(x_sym).diff(x_sym) + y_func(x_sym)
    sol = sp.dsolve(ode, y_func(x_sym))
    print(f"\n  SymPy dsolve: {sol}")


# ============================================================
# Problem 4: Bessel Function J_1
# ============================================================
def exercise_4():
    """
    Compute J_1(x) using series. Compare with scipy.special.jv.
    Find number of terms for 8-digit accuracy at x=10.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Bessel Function J_1 Computation")
    print("=" * 60)

    # J_1(x) = sum_{k=0}^inf (-1)^k / (k! * (k+1)!) * (x/2)^(2k+1)
    def bessel_j1_series(x, n_terms=50):
        """Compute J_1(x) using the series representation."""
        result = 0.0
        for k in range(n_terms):
            term = ((-1)**k / (np.math.factorial(k) * np.math.factorial(k + 1))) * (x/2)**(2*k + 1)
            result += term
        return result

    print(f"\n  J_1(x) = sum_{{k=0}}^inf (-1)^k / (k! * (k+1)!) * (x/2)^(2k+1)")

    # Compare with scipy
    x_test = np.linspace(0, 15, 300)
    j1_scipy = special.jv(1, x_test)

    # Test at x = 10: find terms needed for 8-digit accuracy
    print(f"\n  Finding terms needed for 8-digit accuracy at x = 10:")
    x_val = 10.0
    j1_exact = special.jv(1, x_val)
    print(f"  J_1(10) exact = {j1_exact:.15f}")

    n_required = None
    for n in range(5, 60):
        approx = bessel_j1_series(x_val, n)
        err = abs(approx - j1_exact)
        if err < 5e-9 and n_required is None:
            n_required = n
        if n <= 15 or n == n_required or n % 10 == 0:
            print(f"    N = {n:3d}: J_1(10) = {approx:>20.15f}, error = {err:.2e}" +
                  (" <--" if n == n_required else ""))

    print(f"\n  Need {n_required} terms for 8-digit accuracy at x = 10")

    # Plot comparison
    j1_series_vals = np.array([bessel_j1_series(xv, 40) for xv in x_test])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(x_test, j1_scipy, 'b-', linewidth=2, label='scipy.special.jv(1, x)')
    ax1.plot(x_test, j1_series_vals, 'r--', linewidth=1.5, label='Series (40 terms)')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('$J_1(x)$', fontsize=12)
    ax1.set_title('Bessel Function $J_1(x)$', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(x_test[1:], np.abs(j1_scipy[1:] - j1_series_vals[1:]) + 1e-16, 'b-', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('|Error|', fontsize=12)
    ax2.set_title('Series Approximation Error (40 terms)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_bessel_j1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex16_bessel_j1.png]")


# ============================================================
# Problem 5: Legendre Polynomial P_3
# ============================================================
def exercise_5():
    """
    Derive series solution of Legendre's equation for l=3.
    Show it terminates. Verify P_3(x) = (5x^3 - 3x)/2.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Legendre Polynomial P_3")
    print("=" * 60)

    x = sp.Symbol('x')
    n = sp.Symbol('n', integer=True, nonneg=True)

    # Legendre's equation: (1-x^2)y'' - 2xy' + l(l+1)y = 0, l=3
    # => y'' - 2x/(1-x^2) y' + 12/(1-x^2) y = 0
    ell = 3
    print(f"\n  Legendre's equation for l = {ell}:")
    print(f"  (1-x^2)y'' - 2xy' + {ell*(ell+1)}y = 0")

    # Series solution: y = sum a_n x^n
    # Recurrence: a_{n+2} = [n(n+1) - l(l+1)] / [(n+1)(n+2)] * a_n
    # = [n(n+1) - 12] / [(n+1)(n+2)] * a_n
    print(f"\n  Recurrence relation:")
    print(f"  a_(n+2) = [n(n+1) - l(l+1)] / [(n+1)(n+2)] * a_n")
    print(f"          = [n(n+1) - 12] / [(n+1)(n+2)] * a_n")

    # For l = 3: a_{n+2} = 0 when n(n+1) = 12 => n = 3
    # So the series TERMINATES at n = 3 (for the odd series)
    print(f"\n  Termination: a_(n+2) = 0 when n(n+1) = 12 => n = 3")
    print(f"  The odd series terminates!")

    # Compute coefficients (odd series since l=3 is odd):
    # a_1 = 1 (arbitrary), a_0 = 0
    # a_3 = [1*2 - 12] / [2*3] * a_1 = -10/6 * a_1 = -5/3
    # a_5 = [3*4 - 12] / [4*5] * a_3 = 0 (terminates!)
    a = {0: 0, 1: 1}  # odd series
    for k in range(1, 6):
        nk = 2*k - 1  # odd index
        a[nk + 2] = sp.Rational(nk*(nk+1) - ell*(ell+1), (nk+1)*(nk+2)) * a.get(nk, 0)

    print(f"\n  Odd series (a_0 = 0, a_1 = 1):")
    for k in sorted(a.keys()):
        if a[k] != 0:
            print(f"    a_{k} = {a[k]}")

    # y = a_1*x + a_3*x^3 = x - (5/3)*x^3 (unnormalized)
    y_unnorm = x - sp.Rational(5, 3) * x**3
    print(f"\n  Unnormalized: y = {y_unnorm}")

    # Normalize: P_3(1) = 1
    y_at_1 = y_unnorm.subs(x, 1)
    P3 = y_unnorm / y_at_1
    P3_simplified = sp.simplify(P3)
    print(f"  y(1) = {y_at_1}")
    print(f"  P_3(x) = y/y(1) = {P3_simplified}")
    print(f"         = (5x^3 - 3x)/2")

    # Verify
    P3_standard = sp.Rational(1, 2) * (5*x**3 - 3*x)
    print(f"\n  Standard P_3(x) = (5x^3 - 3x)/2")
    print(f"  Match: {sp.simplify(P3_simplified - P3_standard) == 0}")

    # Verify P_3(1) = 1
    print(f"  P_3(1) = {P3_standard.subs(x, 1)}")

    # Also verify via SymPy's legendre
    P3_sympy = sp.legendre(3, x)
    print(f"  SymPy legendre(3, x) = {sp.expand(P3_sympy)}")
    print(f"  Agreement: {sp.simplify(P3_simplified - P3_sympy) == 0}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("\n" + "=" * 60)
    print("All exercises for Lesson 16 completed.")
    print("=" * 60)

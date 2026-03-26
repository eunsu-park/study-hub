"""
Exercise Solutions: Lesson 04 - Integration Fundamentals
Calculus and Differential Equations

Topics covered:
- Riemann sum computation (left, right, midpoint)
- Fundamental Theorem of Calculus applications
- Signed area interpretation
- Numerical integration comparison
- FTC proof (symbolic demonstration)
"""

import numpy as np
import sympy as sp
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Riemann Sum Computation
# ============================================================
def exercise_1():
    """
    Compute left, right, and midpoint Riemann sums for
    integral_0^2 x^3 dx with n=4 subintervals.
    Compare with the exact value.
    """
    print("=" * 60)
    print("Problem 1: Riemann Sum Computation")
    print("=" * 60)

    f = lambda x: x**3
    a, b = 0, 2
    n = 4
    dx = (b - a) / n

    # Exact value: integral of x^3 from 0 to 2 = [x^4/4]_0^2 = 16/4 = 4
    exact = 4.0

    # Left Riemann sum: sum f(x_i) * dx, x_i = a + i*dx for i = 0..n-1
    x_left = np.array([a + i*dx for i in range(n)])
    left_sum = np.sum(f(x_left)) * dx

    # Right Riemann sum: sum f(x_i) * dx, x_i = a + i*dx for i = 1..n
    x_right = np.array([a + i*dx for i in range(1, n+1)])
    right_sum = np.sum(f(x_right)) * dx

    # Midpoint Riemann sum: sum f(x_i) * dx, x_i = a + (i+0.5)*dx
    x_mid = np.array([a + (i + 0.5)*dx for i in range(n)])
    mid_sum = np.sum(f(x_mid)) * dx

    print(f"\n  integral_0^2 x^3 dx, n = {n}, dx = {dx}")
    print(f"  Exact value = [x^4/4]_0^2 = 16/4 = {exact}")
    print(f"\n  Left endpoints:  {x_left.tolist()}")
    print(f"  f(left):         {f(x_left).tolist()}")
    print(f"  Left sum  = {left_sum:.4f}  (error = {abs(left_sum - exact):.4f})")
    print(f"\n  Right endpoints: {x_right.tolist()}")
    print(f"  f(right):        {f(x_right).tolist()}")
    print(f"  Right sum = {right_sum:.4f}  (error = {abs(right_sum - exact):.4f})")
    print(f"\n  Mid endpoints:   {x_mid.tolist()}")
    print(f"  f(mid):          {f(x_mid).tolist()}")
    print(f"  Mid sum   = {mid_sum:.4f}  (error = {abs(mid_sum - exact):.4f})")
    print(f"\n  Midpoint is closest to exact value (error O(dx^2) vs O(dx) for left/right)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x_plot = np.linspace(a, b, 200)

    for ax, title, x_pts, s_val in [
        (axes[0], f'Left Sum = {left_sum:.2f}', x_left, left_sum),
        (axes[1], f'Right Sum = {right_sum:.2f}', x_right, right_sum),
        (axes[2], f'Midpoint Sum = {mid_sum:.2f}', x_mid, mid_sum),
    ]:
        ax.plot(x_plot, f(x_plot), 'b-', linewidth=2)
        for xi in x_pts:
            ax.bar(xi if 'Mid' not in title else xi - dx/2,
                   f(xi), width=dx, alpha=0.3, color='orange',
                   edgecolor='orange', linewidth=1,
                   align='edge' if 'Mid' not in title else 'edge')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('$x^3$')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Riemann Sums for $\\int_0^2 x^3 \\, dx$ (exact = {exact})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('ex04_riemann_sums.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex04_riemann_sums.png]")


# ============================================================
# Problem 2: FTC Application
# ============================================================
def exercise_2():
    """
    Use FTC to evaluate:
    (a) integral_1^4 (3*sqrt(x) - 1/x) dx
    (b) integral_0^{pi/4} sec^2(theta) d(theta)
    (c) d/dx integral_0^{x^2} sin(t^2) dt  (FTC Part 1 + chain rule)
    """
    print("\n" + "=" * 60)
    print("Problem 2: FTC Application")
    print("=" * 60)

    x, t, theta = sp.symbols('x t theta')

    # (a) integral_1^4 (3*sqrt(x) - 1/x) dx
    integrand_a = 3*sp.sqrt(x) - 1/x
    antideriv_a = sp.integrate(integrand_a, x)
    result_a = sp.integrate(integrand_a, (x, 1, 4))
    print(f"\n(a) integral_1^4 (3*sqrt(x) - 1/x) dx")
    print(f"    Antiderivative: F(x) = {antideriv_a}")
    print(f"    F(4) - F(1) = {antideriv_a.subs(x, 4)} - ({antideriv_a.subs(x, 1)})")
    print(f"    = {result_a}")
    print(f"    = {float(result_a):.6f}")

    # (b) integral_0^{pi/4} sec^2(theta) d(theta)
    integrand_b = sp.sec(theta)**2
    result_b = sp.integrate(integrand_b, (theta, 0, sp.pi/4))
    print(f"\n(b) integral_0^{{pi/4}} sec^2(theta) d(theta)")
    print(f"    Antiderivative of sec^2(theta) = tan(theta)")
    print(f"    tan(pi/4) - tan(0) = 1 - 0 = {result_b}")

    # (c) d/dx integral_0^{x^2} sin(t^2) dt
    # By FTC Part 1 + chain rule:
    # Let G(u) = integral_0^u sin(t^2) dt, then G'(u) = sin(u^2)
    # d/dx G(x^2) = G'(x^2) * 2x = sin(x^4) * 2x
    print(f"\n(c) d/dx integral_0^{{x^2}} sin(t^2) dt")
    print(f"    Let G(u) = integral_0^u sin(t^2) dt")
    print(f"    By FTC Part 1: G'(u) = sin(u^2)")
    print(f"    By chain rule: d/dx G(x^2) = G'(x^2) * d/dx(x^2)")
    print(f"                                = sin((x^2)^2) * 2x")
    print(f"                                = 2x * sin(x^4)")

    # SymPy verification
    G = sp.integrate(sp.sin(t**2), (t, 0, x**2))
    dG_dx = sp.diff(G, x)
    print(f"    SymPy verification: {sp.simplify(dG_dx)}")


# ============================================================
# Problem 3: Signed Area Interpretation
# ============================================================
def exercise_3():
    """
    Compute integral_0^{2pi} sin(x) dx.
    Explain why = 0. Compute total unsigned area.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Signed Area Interpretation")
    print("=" * 60)

    x = sp.Symbol('x')

    # Signed integral
    signed_area = sp.integrate(sp.sin(x), (x, 0, 2*sp.pi))
    print(f"\n  Signed area: integral_0^{{2pi}} sin(x) dx = {signed_area}")
    print(f"\n  Explanation:")
    print(f"    sin(x) > 0 on (0, pi): positive area = integral_0^pi sin(x) dx = 2")
    print(f"    sin(x) < 0 on (pi, 2pi): negative area = integral_pi^{{2pi}} sin(x) dx = -2")
    print(f"    The positive and negative areas cancel exactly, giving 0.")

    # Unsigned (absolute) area
    unsigned_area = sp.integrate(sp.Abs(sp.sin(x)), (x, 0, 2*sp.pi))
    print(f"\n  Unsigned area: integral_0^{{2pi}} |sin(x)| dx = {unsigned_area}")
    print(f"    = integral_0^pi sin(x) dx + integral_pi^{{2pi}} (-sin(x)) dx")
    print(f"    = 2 + 2 = 4")

    # Numerical verification
    x_num = np.linspace(0, 2*np.pi, 10000)
    signed_num = np.trapz(np.sin(x_num), x_num)
    unsigned_num = np.trapz(np.abs(np.sin(x_num)), x_num)
    print(f"\n  Numerical verification:")
    print(f"    Signed area   ~ {signed_num:.6f}")
    print(f"    Unsigned area ~ {unsigned_num:.6f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_vals = np.linspace(0, 2*np.pi, 500)
    y_vals = np.sin(x_vals)

    # Signed area
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2)
    ax1.fill_between(x_vals, y_vals, 0, where=(y_vals >= 0), alpha=0.3, color='green', label='Positive area (+2)')
    ax1.fill_between(x_vals, y_vals, 0, where=(y_vals < 0), alpha=0.3, color='red', label='Negative area (-2)')
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.set_title('Signed Area = 0 (areas cancel)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Unsigned area
    ax2.plot(x_vals, np.abs(y_vals), 'b-', linewidth=2)
    ax2.fill_between(x_vals, np.abs(y_vals), 0, alpha=0.3, color='blue', label='Total area = 4')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.set_title('Unsigned (Absolute) Area = 4', fontsize=12)
    ax2.set_xlabel('x')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex04_signed_area.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex04_signed_area.png]")


# ============================================================
# Problem 4: Numerical Integration Comparison
# ============================================================
def exercise_4():
    """
    Approximate integral_0^1 e^(-x^2) dx using:
    - Left Riemann sum (n=1000)
    - Trapezoidal rule (n=1000)
    - Simpson's rule (n=1000)
    - scipy.integrate.quad
    """
    print("\n" + "=" * 60)
    print("Problem 4: Numerical Integration Comparison")
    print("=" * 60)

    f = lambda x: np.exp(-x**2)
    a, b = 0.0, 1.0
    n = 1000

    # Reference value from scipy quad
    exact, exact_err = integrate.quad(f, a, b)
    print(f"\n  Integral: integral_0^1 e^(-x^2) dx")
    print(f"  Reference (scipy.quad): {exact:.15f} (est. error: {exact_err:.2e})")

    # Left Riemann sum
    dx = (b - a) / n
    x_left = np.linspace(a, b - dx, n)
    left_sum = np.sum(f(x_left)) * dx
    left_err = abs(left_sum - exact)

    # Trapezoidal rule
    x_trap = np.linspace(a, b, n + 1)
    trap_sum = np.trapz(f(x_trap), x_trap)
    trap_err = abs(trap_sum - exact)

    # Simpson's rule
    x_simp = np.linspace(a, b, n + 1)
    simp_sum = integrate.simpson(f(x_simp), x=x_simp)
    simp_err = abs(simp_sum - exact)

    print(f"\n  {'Method':<25s}  {'Result':>18s}  {'Error':>14s}  {'Func evals':>12s}")
    print(f"  {'-'*25}  {'-'*18}  {'-'*14}  {'-'*12}")
    print(f"  {'Left Riemann (n=1000)':<25s}  {left_sum:>18.15f}  {left_err:>14.2e}  {n:>12d}")
    print(f"  {'Trapezoidal (n=1000)':<25s}  {trap_sum:>18.15f}  {trap_err:>14.2e}  {n+1:>12d}")
    print(f"  {'Simpsons (n=1000)':<25s}  {simp_sum:>18.15f}  {simp_err:>14.2e}  {n+1:>12d}")
    print(f"  {'scipy.integrate.quad':<25s}  {exact:>18.15f}  {'~1e-14':>14s}  {'adaptive':>12s}")

    print(f"\n  Analysis:")
    print(f"    Left Riemann: O(1/n) error, simplest but least accurate")
    print(f"    Trapezoidal:  O(1/n^2) error, significant improvement for same n")
    print(f"    Simpson's:    O(1/n^4) error, best accuracy for polynomial-like integrands")
    print(f"    Adaptive quad: automatically adjusts step size, best overall")
    print(f"    Simpson's gives the most accuracy per function evaluation among the fixed methods.")


# ============================================================
# Problem 5: Proving the FTC
# ============================================================
def exercise_5():
    """
    Given FTC Part 1: d/dx integral_a^x f(t) dt = f(x),
    prove FTC Part 2: integral_a^b f(x) dx = F(b) - F(a) where F' = f.

    We demonstrate the proof symbolically and verify with examples.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Proving the FTC")
    print("=" * 60)

    print("\n  Proof of FTC Part 2 from Part 1:")
    print("  ================================")
    print("")
    print("  Define G(x) = integral_a^x f(t) dt.")
    print("  By FTC Part 1: G'(x) = f(x).")
    print("")
    print("  Given: F'(x) = f(x) (F is any antiderivative of f).")
    print("")
    print("  Since G'(x) = F'(x) for all x in [a, b],")
    print("  G(x) - F(x) = C (constant) for some C.  [functions with equal derivatives differ by a constant]")
    print("")
    print("  Evaluate at x = a:")
    print("    G(a) = integral_a^a f(t) dt = 0")
    print("    So C = G(a) - F(a) = 0 - F(a) = -F(a)")
    print("    Therefore: G(x) = F(x) - F(a)")
    print("")
    print("  Evaluate at x = b:")
    print("    G(b) = integral_a^b f(t) dt = F(b) - F(a)")
    print("")
    print("  This completes the proof: integral_a^b f(x) dx = F(b) - F(a).  QED")

    # Verification with examples
    x, t = sp.symbols('x t')

    examples = [
        (x**2, 0, 3, "x^2"),
        (sp.sin(x), 0, sp.pi, "sin(x)"),
        (sp.exp(-x), 0, 1, "e^(-x)"),
    ]

    print(f"\n  Numerical verifications:")
    for f_expr, a, b, name in examples:
        F = sp.integrate(f_expr, x)
        ftc_result = F.subs(x, b) - F.subs(x, a)
        direct_result = sp.integrate(f_expr, (x, a, b))
        print(f"\n    f(x) = {name}")
        print(f"    F(x) = {F}")
        print(f"    F({b}) - F({a}) = {sp.simplify(ftc_result)}")
        print(f"    Direct integral  = {sp.simplify(direct_result)}")
        print(f"    Match: {sp.simplify(ftc_result - direct_result) == 0}")


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
    print("All exercises for Lesson 04 completed.")
    print("=" * 60)

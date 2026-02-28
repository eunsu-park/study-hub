"""
Exercise Solutions: Lesson 01 - Limits and Continuity
Calculus and Differential Equations

Topics covered:
- Algebraic limit evaluation
- Epsilon-delta proofs (numerical verification)
- Discontinuity classification
- Intermediate Value Theorem and bisection method
- Squeeze Theorem visualization
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Algebraic Limit Evaluation
# ============================================================
def exercise_1():
    """
    Evaluate the following limits algebraically:
    (a) lim_{x->3} (x^2 - 9)/(x - 3)
    (b) lim_{x->0} (sqrt(1+x) - 1)/x  (rationalize the numerator)
    (c) lim_{x->inf} (2x^3 - x + 5)/(4x^3 + 3x^2)
    """
    print("=" * 60)
    print("Problem 1: Algebraic Limit Evaluation")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) lim_{x->3} (x^2 - 9)/(x - 3)
    # Factor: (x^2 - 9) = (x-3)(x+3), so the limit is lim_{x->3} (x+3) = 6
    expr_a = (x**2 - 9) / (x - 3)
    limit_a = sp.limit(expr_a, x, 3)
    print(f"\n(a) lim_{{x->3}} (x^2 - 9)/(x - 3)")
    print(f"    Factor: (x^2-9)/(x-3) = (x-3)(x+3)/(x-3) = x+3")
    print(f"    Substituting x=3: 3+3 = 6")
    print(f"    SymPy verification: {limit_a}")

    # (b) lim_{x->0} (sqrt(1+x) - 1)/x
    # Rationalize: multiply by (sqrt(1+x)+1)/(sqrt(1+x)+1)
    # = (1+x-1) / (x*(sqrt(1+x)+1)) = 1/(sqrt(1+x)+1) -> 1/2
    expr_b = (sp.sqrt(1 + x) - 1) / x
    limit_b = sp.limit(expr_b, x, 0)
    print(f"\n(b) lim_{{x->0}} (sqrt(1+x) - 1)/x")
    print(f"    Rationalize: multiply by (sqrt(1+x)+1)/(sqrt(1+x)+1)")
    print(f"    = (1+x-1) / (x*(sqrt(1+x)+1)) = 1/(sqrt(1+x)+1)")
    print(f"    As x->0: 1/(1+1) = 1/2")
    print(f"    SymPy verification: {limit_b}")

    # (c) lim_{x->inf} (2x^3 - x + 5)/(4x^3 + 3x^2)
    # Divide numerator and denominator by x^3:
    # = (2 - 1/x^2 + 5/x^3)/(4 + 3/x) -> 2/4 = 1/2
    expr_c = (2*x**3 - x + 5) / (4*x**3 + 3*x**2)
    limit_c = sp.limit(expr_c, x, sp.oo)
    print(f"\n(c) lim_{{x->inf}} (2x^3 - x + 5)/(4x^3 + 3x^2)")
    print(f"    Divide by x^3: (2 - 1/x^2 + 5/x^3)/(4 + 3/x)")
    print(f"    As x->inf: 2/4 = 1/2")
    print(f"    SymPy verification: {limit_c}")


# ============================================================
# Problem 2: Epsilon-Delta Proof
# ============================================================
def exercise_2():
    """
    Using the epsilon-delta definition, prove that lim_{x->2} (3x+1) = 7.
    Numerical verification: for various epsilon, find delta and confirm.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Epsilon-Delta Proof")
    print("=" * 60)

    # Proof:
    # We need: for every eps > 0, there exists delta > 0 such that
    #   0 < |x - 2| < delta  =>  |(3x+1) - 7| < eps
    #
    # Work backward:
    #   |(3x+1) - 7| = |3x - 6| = 3|x - 2| < eps
    #   => |x - 2| < eps/3
    #
    # So choose delta = eps/3.

    print("\nProof:")
    print("  We need: |f(x) - L| < eps  whenever  0 < |x - 2| < delta")
    print("  |(3x+1) - 7| = |3x - 6| = 3|x - 2|")
    print("  So 3|x - 2| < eps  =>  |x - 2| < eps/3")
    print("  Choose delta = eps/3.")
    print("  Then 0 < |x-2| < delta => |(3x+1)-7| = 3|x-2| < 3*delta = eps. QED")

    # Numerical verification
    print("\nNumerical verification:")
    print(f"  {'epsilon':>12s}  {'delta=eps/3':>12s}  {'max |f(x)-7| in (2-d,2+d)':>30s}  {'< eps?':>8s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*30}  {'-'*8}")

    for eps in [1.0, 0.1, 0.01, 0.001, 1e-6]:
        delta = eps / 3.0
        # Sample many x in (2 - delta, 2 + delta), excluding x = 2
        x_vals = np.linspace(2 - delta + 1e-15, 2 + delta - 1e-15, 10000)
        max_deviation = np.max(np.abs((3*x_vals + 1) - 7))
        satisfied = max_deviation < eps
        print(f"  {eps:>12.1e}  {delta:>12.6e}  {max_deviation:>30.10e}  {'Yes' if satisfied else 'No':>8s}")


# ============================================================
# Problem 3: Discontinuity Classification
# ============================================================
def exercise_3():
    """
    Classify discontinuities:
    (a) f(x) = sin(x)/x at x = 0  (removable)
    (b) f(x) = floor(x) at x = 2  (jump)
    (c) f(x) = 1/(x-1)^2 at x = 1 (infinite)
    """
    print("\n" + "=" * 60)
    print("Problem 3: Discontinuity Classification")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) sin(x)/x at x=0
    # The limit exists: lim_{x->0} sin(x)/x = 1
    # But f(0) is undefined. This is a removable discontinuity.
    lim_a = sp.limit(sp.sin(x)/x, x, 0)
    print(f"\n(a) f(x) = sin(x)/x at x = 0")
    print(f"    lim_{{x->0}} sin(x)/x = {lim_a}")
    print(f"    f(0) is undefined (0/0)")
    print(f"    The limit exists and is finite => REMOVABLE discontinuity")
    print(f"    (Can be removed by defining f(0) = 1)")

    # (b) floor(x) at x=2
    # lim_{x->2^-} floor(x) = 1, lim_{x->2^+} floor(x) = 2
    # Left and right limits differ => jump discontinuity
    lim_b_left = sp.limit(sp.floor(x), x, 2, '-')
    lim_b_right = sp.limit(sp.floor(x), x, 2, '+')
    print(f"\n(b) f(x) = floor(x) at x = 2")
    print(f"    lim_{{x->2^-}} floor(x) = {lim_b_left}")
    print(f"    lim_{{x->2^+}} floor(x) = {lim_b_right}")
    print(f"    Left limit != Right limit => JUMP discontinuity")

    # (c) 1/(x-1)^2 at x=1
    # lim_{x->1} 1/(x-1)^2 = +inf
    # The function blows up => infinite discontinuity
    lim_c = sp.limit(1/(x-1)**2, x, 1)
    print(f"\n(c) f(x) = 1/(x-1)^2 at x = 1")
    print(f"    lim_{{x->1}} 1/(x-1)^2 = {lim_c}")
    print(f"    The function grows without bound => INFINITE discontinuity")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) sin(x)/x
    x_vals = np.linspace(-4, 4, 1000)
    x_vals_nz = x_vals[x_vals != 0]
    y_a = np.sin(x_vals_nz) / x_vals_nz
    axes[0].plot(x_vals_nz, y_a, 'b-', linewidth=2)
    axes[0].plot(0, 1, 'ro', markersize=10, markerfacecolor='white', markeredgewidth=2)
    axes[0].set_title('(a) Removable: sin(x)/x')
    axes[0].set_xlabel('x')
    axes[0].grid(True, alpha=0.3)

    # (b) floor(x)
    x_vals2 = np.linspace(-0.5, 4.5, 1000)
    y_b = np.floor(x_vals2)
    axes[1].step(x_vals2, y_b, 'b-', linewidth=2, where='post')
    axes[1].plot(2, 1, 'o', color='blue', markersize=10,
                 markerfacecolor='white', markeredgewidth=2)
    axes[1].plot(2, 2, 'o', color='blue', markersize=10, markeredgewidth=2)
    axes[1].axvline(x=2, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('(b) Jump: floor(x) at x=2')
    axes[1].set_xlabel('x')
    axes[1].grid(True, alpha=0.3)

    # (c) 1/(x-1)^2
    x_vals3 = np.linspace(-1, 3, 1000)
    x_vals3 = x_vals3[np.abs(x_vals3 - 1) > 0.05]
    y_c = 1.0 / (x_vals3 - 1)**2
    axes[2].plot(x_vals3, y_c, 'b-', linewidth=2)
    axes[2].set_ylim(-1, 20)
    axes[2].axvline(x=1, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title('(c) Infinite: 1/(x-1)^2')
    axes[2].set_xlabel('x')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex01_discontinuity_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n    [Plot saved: ex01_discontinuity_types.png]")


# ============================================================
# Problem 4: IVT Application
# ============================================================
def exercise_4():
    """
    Show that x^5 - 3x + 1 = 0 has at least one root in [0, 1].
    Use bisection to find the root to 8 decimal places.
    """
    print("\n" + "=" * 60)
    print("Problem 4: IVT Application")
    print("=" * 60)

    def f(x):
        return x**5 - 3*x + 1

    # Check IVT conditions
    fa = f(0)
    fb = f(1)
    print(f"\n  f(x) = x^5 - 3x + 1")
    print(f"  f(0) = {fa}  (positive)")
    print(f"  f(1) = {fb}  (negative)")
    print(f"  f is continuous (polynomial) and f(0)*f(1) = {fa*fb} < 0")
    print(f"  By the IVT, there exists c in (0,1) such that f(c) = 0.")

    # Bisection method
    def bisection(f, a, b, tol=1e-10, max_iter=200):
        """Find root of f in [a, b] using bisection."""
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        iterations = 0
        for i in range(max_iter):
            c = (a + b) / 2.0
            iterations = i + 1
            if abs(f(c)) < tol or (b - a) / 2.0 < tol:
                return c, iterations
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return c, iterations

    root, iters = bisection(f, 0, 1, tol=5e-9)
    print(f"\n  Bisection method result:")
    print(f"    Root = {root:.8f}")
    print(f"    f(root) = {f(root):.2e}")
    print(f"    Iterations = {iters}")

    # Verify with SymPy
    x = sp.Symbol('x')
    roots_sym = sp.nsolve(x**5 - 3*x + 1, x, 0.5)
    print(f"\n  SymPy nsolve verification:")
    print(f"    Root = {float(roots_sym):.8f}")
    print(f"    Agreement: {abs(root - float(roots_sym)):.2e}")


# ============================================================
# Problem 5: Squeeze Theorem
# ============================================================
def exercise_5():
    """
    Use the Squeeze Theorem to evaluate lim_{x->0} x^2 * cos(1/x^2).
    Visualize the function and its bounding functions on [-0.5, 0.5].
    """
    print("\n" + "=" * 60)
    print("Problem 5: Squeeze Theorem")
    print("=" * 60)

    # Analytical solution:
    # Since -1 <= cos(1/x^2) <= 1 for all x != 0,
    # we have -x^2 <= x^2 cos(1/x^2) <= x^2.
    # Since lim_{x->0} (-x^2) = 0 and lim_{x->0} x^2 = 0,
    # by the Squeeze Theorem, lim_{x->0} x^2 cos(1/x^2) = 0.

    print("\n  Proof using the Squeeze Theorem:")
    print("  Since -1 <= cos(1/x^2) <= 1 for all x != 0,")
    print("  we have: -x^2 <= x^2 * cos(1/x^2) <= x^2")
    print("  lim_{x->0} (-x^2) = 0")
    print("  lim_{x->0}  x^2   = 0")
    print("  By the Squeeze Theorem: lim_{x->0} x^2*cos(1/x^2) = 0")

    # SymPy verification
    x = sp.Symbol('x')
    lim_result = sp.limit(x**2 * sp.cos(1/x**2), x, 0)
    print(f"\n  SymPy verification: {lim_result}")

    # Visualization
    x_vals = np.linspace(-0.5, 0.5, 100000)
    x_nz = x_vals[np.abs(x_vals) > 1e-10]  # avoid division by zero
    y_func = x_nz**2 * np.cos(1.0 / x_nz**2)
    y_upper = x_nz**2
    y_lower = -x_nz**2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_nz, y_func, 'b-', linewidth=0.5, alpha=0.8, label=r'$x^2 \cos(1/x^2)$')
    ax.plot(x_nz, y_upper, 'r--', linewidth=2, label=r'$x^2$ (upper bound)')
    ax.plot(x_nz, y_lower, 'g--', linewidth=2, label=r'$-x^2$ (lower bound)')
    ax.plot(0, 0, 'ko', markersize=8, label='Limit = 0')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(r'Squeeze Theorem: $\lim_{x \to 0} x^2 \cos(1/x^2) = 0$', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig('ex01_squeeze_theorem.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex01_squeeze_theorem.png]")


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
    print("All exercises for Lesson 01 completed.")
    print("=" * 60)

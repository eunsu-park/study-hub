"""
Numerical and Symbolic Integration

Demonstrates:
  - Riemann sums: left, right, midpoint, and Simpson's rule
  - Fundamental Theorem of Calculus (FTC) verification
  - Numerical vs symbolic comparison
  - Convergence rate analysis for each quadrature rule

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Riemann Sum Methods
# ---------------------------------------------------------------------------
def left_riemann(f, a, b, n):
    """Left Riemann sum: sum f(x_i) * dx  with x_i = left edge.

    Error is O(1/n) for smooth functions — first-order accurate.
    """
    x = np.linspace(a, b, n + 1)
    dx = (b - a) / n
    return np.sum(f(x[:-1])) * dx


def right_riemann(f, a, b, n):
    """Right Riemann sum: uses x_i = right edge of each sub-interval."""
    x = np.linspace(a, b, n + 1)
    dx = (b - a) / n
    return np.sum(f(x[1:])) * dx


def midpoint_rule(f, a, b, n):
    """Midpoint rule: evaluate f at the center of each sub-interval.

    Surprisingly, this is O(1/n^2) — same order as the trapezoidal rule
    — because the symmetric placement cancels the first-order error term.
    """
    dx = (b - a) / n
    midpoints = np.linspace(a + dx / 2, b - dx / 2, n)
    return np.sum(f(midpoints)) * dx


def simpson_rule(f, a, b, n):
    """Composite Simpson's 1/3 rule: O(1/n^4) accuracy.

    Simpson's rule fits a parabola through every three consecutive points.
    It exactly integrates polynomials up to degree 3, giving fourth-order
    convergence.  Requires n to be even.
    """
    if n % 2 != 0:
        n += 1  # Simpson requires even n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = (b - a) / n
    # Weights: 1, 4, 2, 4, 2, ..., 4, 1
    return dx / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) +
                     2 * np.sum(y[2:-2:2]))


# ---------------------------------------------------------------------------
# 2. Visualize Riemann Sums
# ---------------------------------------------------------------------------
def plot_riemann(f, a, b, n=10):
    """Draw left Riemann rectangles alongside the true curve."""
    x_fine = np.linspace(a, b, 500)
    y_fine = f(x_fine)
    dx = (b - a) / n
    x_left = np.linspace(a, b - dx, n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_fine, y_fine, "b-", lw=2, label="f(x)")

    # Draw rectangles
    for xi in x_left:
        rect = plt.Rectangle((xi, 0), dx, f(xi), alpha=0.3,
                              edgecolor="navy", facecolor="skyblue")
        ax.add_patch(rect)

    approx = left_riemann(f, a, b, n)
    ax.set_title(f"Left Riemann Sum (n={n}): approx = {approx:.6f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_riemann_sum.png", dpi=100)
    plt.close()
    print("[Saved] 04_riemann_sum.png")


# ---------------------------------------------------------------------------
# 3. FTC Demonstration
# ---------------------------------------------------------------------------
def ftc_demo():
    """Verify the Fundamental Theorem of Calculus numerically.

    FTC Part 1: d/dx integral_{a}^{x} f(t) dt = f(x).
    We compute F(x) = integral_0^x sin(t) dt numerically and differentiate F
    numerically, then compare with sin(x).
    """
    from scipy.integrate import cumulative_trapezoid

    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x)

    # F(x) = integral_0^x sin(t) dt = 1 - cos(x)
    F_numerical = cumulative_trapezoid(y, x, initial=0)
    F_exact = 1 - np.cos(x)

    # d/dx F(x) should equal sin(x)
    dF = np.gradient(F_numerical, x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x, F_numerical, "b-", lw=2, label="F(x) numerical")
    axes[0].plot(x, F_exact, "r--", lw=1.5, label="1 - cos(x) exact")
    axes[0].set_title("FTC Part 1: F(x) = integral sin(t) dt")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, dF, "b-", lw=2, label="dF/dx (numerical)")
    axes[1].plot(x, y, "r--", lw=1.5, label="sin(x)")
    axes[1].set_title("FTC Verification: dF/dx = f(x)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_ftc.png", dpi=100)
    plt.close()
    print("[Saved] 04_ftc.png")


# ---------------------------------------------------------------------------
# 4. Convergence Rate Analysis
# ---------------------------------------------------------------------------
def convergence_analysis(f, a, b, exact_val):
    """Measure how quickly each quadrature rule converges.

    Expected rates:
      Left/Right Riemann : O(1/n)   -> slope -1 on log-log
      Midpoint           : O(1/n^2) -> slope -2
      Simpson            : O(1/n^4) -> slope -4
    """
    n_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    methods = {
        "Left Riemann": left_riemann,
        "Right Riemann": right_riemann,
        "Midpoint": midpoint_rule,
        "Simpson": simpson_rule,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]

    for (name, method), marker in zip(methods.items(), markers):
        errors = []
        for n in n_values:
            approx = method(f, a, b, n)
            errors.append(abs(approx - exact_val))
        errors = [max(e, 1e-16) for e in errors]
        ax.loglog(n_values, errors, f"{marker}-", label=name, ms=5)

    # Reference slopes
    n_arr = np.array(n_values, dtype=float)
    ax.loglog(n_arr, 5 / n_arr, "k--", alpha=0.3, label="O(1/n)")
    ax.loglog(n_arr, 5 / n_arr ** 2, "k:", alpha=0.3, label="O(1/n^2)")
    ax.loglog(n_arr, 50 / n_arr ** 4, "k-.", alpha=0.3, label="O(1/n^4)")

    ax.set_xlabel("Number of sub-intervals (n)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Quadrature Convergence Rates")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_convergence.png", dpi=100)
    plt.close()
    print("[Saved] 04_convergence.png")


# ---------------------------------------------------------------------------
# 5. Symbolic vs Numerical Comparison
# ---------------------------------------------------------------------------
def symbolic_vs_numerical():
    """Compare SymPy exact integral with numerical quadrature."""
    x = sp.Symbol("x")
    # A non-trivial integral: int_0^1 x^2 * exp(-x) dx
    f_sym = x ** 2 * sp.exp(-x)
    exact = sp.integrate(f_sym, (x, 0, 1))
    exact_float = float(exact)

    f_np = lambda t: t ** 2 * np.exp(-t)
    simp = simpson_rule(f_np, 0, 1, 100)

    print("Symbolic vs Numerical Integration")
    print(f"  Integral : int_0^1 x^2 * exp(-x) dx")
    print(f"  Symbolic : {exact} = {exact_float:.15f}")
    print(f"  Simpson  : {simp:.15f} (n=100)")
    print(f"  Error    : {abs(simp - exact_float):.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    f = lambda x: np.sin(x)
    a, b = 0, np.pi
    exact = 2.0  # integral of sin from 0 to pi

    # --- Demo 1: Riemann sums ---
    print("=" * 60)
    print(f"Integrating sin(x) from 0 to pi  (exact = {exact})")
    print("=" * 60)
    for n in [10, 100, 1000]:
        print(f"\n  n = {n}")
        print(f"    Left    : {left_riemann(f, a, b, n):.10f}")
        print(f"    Right   : {right_riemann(f, a, b, n):.10f}")
        print(f"    Midpoint: {midpoint_rule(f, a, b, n):.10f}")
        print(f"    Simpson : {simpson_rule(f, a, b, n):.10f}")

    # --- Demo 2: Visualization ---
    print("\nDemo 2: Riemann sum visualization")
    plot_riemann(f, a, b, n=15)

    # --- Demo 3: FTC ---
    print("\nDemo 3: Fundamental Theorem of Calculus")
    ftc_demo()

    # --- Demo 4: Convergence rates ---
    print("\nDemo 4: Convergence rate analysis")
    convergence_analysis(f, a, b, exact)

    # --- Demo 5: Symbolic vs numerical ---
    print("\nDemo 5: Symbolic vs numerical")
    symbolic_vs_numerical()

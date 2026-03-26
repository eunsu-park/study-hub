"""
Numerical and Symbolic Differentiation

Demonstrates:
  - Forward, central, and complex-step differentiation
  - Accuracy comparison across methods and step sizes
  - Symbolic differentiation with SymPy
  - Tangent line visualization

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Numerical Differentiation Methods
# ---------------------------------------------------------------------------
def forward_diff(f, x, h=1e-8):
    """Forward difference: O(h) accuracy.

    Simplest finite difference but only first-order accurate because the
    Taylor expansion  f(x+h) = f(x) + h f'(x) + h^2/2 f''(x) + ...
    leaves an O(h) truncation error after dividing by h.
    """
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h=1e-5):
    """Central difference: O(h^2) accuracy.

    By using symmetric points, the leading error term (proportional to h)
    cancels out, leaving O(h^2).  This is the standard workhorse for
    numerical derivatives.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def complex_step_diff(f, x, h=1e-30):
    """Complex-step derivative: machine-precision accuracy.

    The key insight: f(x + ih) = f(x) + ih f'(x) - h^2/2 f''(x) - ...
    Taking Im(f(x+ih))/h gives f'(x) with NO subtractive cancellation,
    so we can use extremely small h (e.g., 1e-30) without round-off error.
    Requires f to be analytic and accept complex inputs.
    """
    return np.imag(f(x + 1j * h)) / h


# ---------------------------------------------------------------------------
# 2. Accuracy Comparison
# ---------------------------------------------------------------------------
def compare_accuracy(f, f_exact, x0, h_values):
    """Compare errors of the three differentiation methods.

    We evaluate the absolute error |approx - exact| for each method
    at a range of step sizes h.  The complex-step method should reach
    machine epsilon regardless of h (down to ~1e-300).
    """
    exact = f_exact(x0)
    results = {"forward": [], "central": [], "complex_step": []}

    for h in h_values:
        results["forward"].append(abs(forward_diff(f, x0, h) - exact))
        results["central"].append(abs(central_diff(f, x0, h) - exact))
        results["complex_step"].append(abs(complex_step_diff(f, x0, h) - exact))

    return results


def plot_accuracy(h_values, results):
    """Log-log plot showing error vs step size for each method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {"forward": ("o-", "C0"), "central": ("s-", "C1"),
              "complex_step": ("^-", "C2")}

    for name, errors in results.items():
        marker, color = styles[name]
        # Replace exact zeros with a tiny number so log scale works
        errors_safe = [max(e, 1e-17) for e in errors]
        ax.loglog(h_values, errors_safe, marker, color=color,
                  label=name, ms=5, lw=1.5)

    # Reference slopes
    ax.loglog(h_values, h_values, "k--", alpha=0.3, label="O(h)")
    ax.loglog(h_values, np.array(h_values) ** 2, "k:", alpha=0.3,
              label="O(h^2)")

    ax.set_xlabel("Step size h")
    ax.set_ylabel("Absolute error")
    ax.set_title("Differentiation Accuracy Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("02_derivative_accuracy.png", dpi=100)
    plt.close()
    print("[Saved] 02_derivative_accuracy.png")


# ---------------------------------------------------------------------------
# 3. Symbolic Differentiation with SymPy
# ---------------------------------------------------------------------------
def symbolic_derivatives():
    """Demonstrate symbolic differentiation for several functions."""
    x = sp.Symbol("x")
    functions = [
        sp.sin(x) * sp.exp(x),
        sp.log(1 + x ** 2),
        x ** x,                       # requires logarithmic differentiation
        sp.atan(sp.sqrt(x)),
    ]

    print("Symbolic Derivatives (SymPy)")
    print("-" * 60)
    for f_sym in functions:
        df = sp.diff(f_sym, x)
        df_simplified = sp.simplify(df)
        print(f"  f(x)  = {f_sym}")
        print(f"  f'(x) = {df_simplified}")
        print()

    # Higher-order derivative
    f_higher = sp.sin(x)
    print("Higher-order derivatives of sin(x):")
    for n in range(1, 6):
        dn = sp.diff(f_higher, x, n)
        print(f"  d^{n}/dx^{n} sin(x) = {dn}")


# ---------------------------------------------------------------------------
# 4. Tangent Line Visualization
# ---------------------------------------------------------------------------
def plot_tangent_lines(f, f_sym, x0_list, x_range=(-2, 4)):
    """Plot a function and tangent lines at several points.

    The tangent line at x0 is:  y = f(x0) + f'(x0)(x - x0).
    """
    x_sym = sp.Symbol("x")
    df_sym = sp.diff(f_sym, x_sym)
    df_func = sp.lambdify(x_sym, df_sym, "numpy")

    x = np.linspace(*x_range, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "b-", lw=2, label="f(x)")

    colors = plt.cm.Set1(np.linspace(0, 1, len(x0_list)))
    for x0, color in zip(x0_list, colors):
        y0 = f(x0)
        slope = float(df_func(x0))
        tangent = y0 + slope * (x - x0)
        ax.plot(x, tangent, "--", color=color, lw=1.5,
                label=f"Tangent at x={x0} (slope={slope:.2f})")
        ax.plot(x0, y0, "o", color=color, ms=8, zorder=5)

    ax.set_ylim(min(y) - 1, max(y) + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Function and Tangent Lines")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("02_tangent_lines.png", dpi=100)
    plt.close()
    print("[Saved] 02_tangent_lines.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test function: f(x) = sin(x)*exp(x), f'(x) = exp(x)*(sin(x)+cos(x))
    f = lambda x: np.sin(x) * np.exp(x)
    f_exact = lambda x: np.exp(x) * (np.sin(x) + np.cos(x))
    x0 = 1.0

    # --- Demo 1: Single-point comparison ---
    print("=" * 60)
    print(f"Derivatives at x = {x0}")
    print("=" * 60)
    print(f"  Exact     : {f_exact(x0):.15f}")
    print(f"  Forward   : {forward_diff(f, x0):.15f}")
    print(f"  Central   : {central_diff(f, x0):.15f}")
    print(f"  Complex   : {complex_step_diff(f, x0):.15f}")

    # --- Demo 2: Accuracy vs step size ---
    print("\nDemo 2: Accuracy comparison plot")
    h_vals = np.logspace(-1, -15, 30)
    results = compare_accuracy(f, f_exact, x0, h_vals)
    plot_accuracy(h_vals, results)

    # --- Demo 3: Symbolic derivatives ---
    print("\nDemo 3: Symbolic differentiation")
    symbolic_derivatives()

    # --- Demo 4: Tangent lines for x*sin(x) ---
    print("\nDemo 4: Tangent line visualization")
    x_sym = sp.Symbol("x")
    g_sym = x_sym * sp.sin(x_sym)
    g = lambda x: x * np.sin(x)
    plot_tangent_lines(g, g_sym, [0.5, 1.5, 3.0], x_range=(-1, 5))

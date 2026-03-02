"""
Root Finding
Numerical Root Finding Methods

Methods for numerically finding x that satisfies f(x) = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional


# =============================================================================
# 1. Bisection Method
# =============================================================================
def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    Find a root of f(x) = 0 using the bisection method

    Condition: f(a) and f(b) must have opposite signs (Intermediate Value Theorem)
    Convergence rate: Linear (interval halved each iteration)

    Args:
        f: Target function
        a, b: Initial interval
        tol: Tolerance
        max_iter: Maximum number of iterations

    Returns:
        (root, number of iterations, midpoint history)
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    history = []

    for i in range(max_iter):
        c = (a + b) / 2
        history.append(c)

        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, i + 1, history

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2, max_iter, history


# =============================================================================
# 2. Newton-Raphson Method
# =============================================================================
def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    Find a root of f(x) = 0 using the Newton-Raphson method

    x_{n+1} = x_n - f(x_n) / f'(x_n)

    Convergence rate: Quadratic
    Drawbacks: Requires derivative, sensitive to initial value

    Args:
        f: Target function
        df: Derivative
        x0: Initial value
        tol: Tolerance
        max_iter: Maximum number of iterations

    Returns:
        (root, number of iterations, history)
    """
    x = x0
    history = [x]

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-15:
            raise ValueError("Derivative near zero: risk of divergence")

        x_new = x - fx / dfx
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, i + 1, history

        x = x_new

    return x, max_iter, history


# =============================================================================
# 3. Secant Method
# =============================================================================
def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    Find a root of f(x) = 0 using the secant method

    Approximates Newton's method derivative with finite differences:
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Convergence rate: ~1.618 (golden ratio)
    Advantage: No derivative required

    Args:
        f: Target function
        x0, x1: Two initial values
        tol: Tolerance
        max_iter: Maximum number of iterations

    Returns:
        (root, number of iterations, history)
    """
    history = [x0, x1]

    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)

        if abs(f1 - f0) < 1e-15:
            raise ValueError("Denominator near zero")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)

        if abs(x2 - x1) < tol:
            return x2, i + 1, history

        x0, x1 = x1, x2

    return x1, max_iter, history


# =============================================================================
# 4. Fixed-Point Iteration
# =============================================================================
def fixed_point(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    Fixed-point iteration: find x satisfying x = g(x)

    Transform f(x) = 0 into x = g(x)
    Example: x^2 - 2 = 0  ->  x = 2/x or x = (x + 2/x)/2

    Convergence condition: |g'(x*)| < 1

    Args:
        g: Iteration function
        x0: Initial value
        tol: Tolerance
        max_iter: Maximum number of iterations

    Returns:
        (fixed point, number of iterations, history)
    """
    x = x0
    history = [x]

    for i in range(max_iter):
        x_new = g(x)
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, i + 1, history

        x = x_new

    return x, max_iter, history


# =============================================================================
# 5. Brent's Method (comparison with scipy)
# =============================================================================
def brents_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int]:
    """
    Brent's Method (simplified version)
    Combines bisection, secant, and inverse quadratic interpolation

    For production use, scipy.optimize.brentq is recommended
    """
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = c  # Initialize d (used in first iteration)
    mflag = True

    for i in range(max_iter):
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Check conditions, fallback to bisection
        conditions = [
            not ((3 * a + b) / 4 <= s <= b or b <= s <= (3 * a + b) / 4),
            mflag and abs(s - b) >= abs(b - c) / 2,
            not mflag and abs(s - b) >= abs(c - d) / 2,
        ]

        if any(conditions):
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d, c, fc = c, b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(b - a) < tol or abs(fb) < tol:
            return b, i + 1

    return b, max_iter


# =============================================================================
# Visualization
# =============================================================================
def plot_convergence(f, methods_data, x_range, title="Root Finding Convergence Comparison"):
    """Visualize convergence process"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Function and roots
    x = np.linspace(x_range[0], x_range[1], 500)
    y = [f(xi) for xi in x]
    axes[0].plot(x, y, 'b-', label='f(x)')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    colors = plt.cm.tab10.colors
    for i, (name, root, _, history) in enumerate(methods_data):
        axes[0].scatter([root], [0], s=100, color=colors[i], zorder=5, label=f'{name}: x={root:.6f}')

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Function and Roots')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Convergence rate
    for i, (name, root, iters, history) in enumerate(methods_data):
        if history:
            errors = [abs(h - root) for h in history]
            errors = [e if e > 1e-16 else 1e-16 for e in errors]
            axes[1].semilogy(errors, 'o-', color=colors[i], label=f'{name} ({iters} iters)')

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Error (log scale)')
    axes[1].set_title('Convergence Rate Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/root_finding.png', dpi=150)
    plt.close()
    print("    Graph saved: root_finding.png")


# =============================================================================
# Test
# =============================================================================
def main():
    print("=" * 60)
    print("Root Finding Examples")
    print("=" * 60)

    # Example 1: f(x) = x^3 - x - 2 = 0 (root ~ 1.5214)
    print("\n[Example 1] f(x) = x^3 - x - 2 = 0")
    print("-" * 40)

    f = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1

    methods_data = []

    # Bisection
    root, iters, hist = bisection(f, 1, 2)
    methods_data.append(("Bisection", root, iters, hist))
    print(f"Bisection:      root = {root:.10f}, iterations = {iters}")

    # Newton-Raphson
    root, iters, hist = newton_raphson(f, df, 1.5)
    methods_data.append(("Newton", root, iters, hist))
    print(f"Newton-Raphson: root = {root:.10f}, iterations = {iters}")

    # Secant
    root, iters, hist = secant(f, 1, 2)
    methods_data.append(("Secant", root, iters, hist))
    print(f"Secant:         root = {root:.10f}, iterations = {iters}")

    # Visualization
    try:
        plot_convergence(f, methods_data, (0, 3), "f(x) = x^3 - x - 2")
    except Exception as e:
        print(f"    Graph generation failed: {e}")

    # Example 2: Finding sqrt(2) (x^2 - 2 = 0)
    print("\n[Example 2] Finding sqrt(2) (x^2 - 2 = 0)")
    print("-" * 40)

    f2 = lambda x: x**2 - 2
    df2 = lambda x: 2*x
    g = lambda x: (x + 2/x) / 2  # Babylonian method

    root, iters, _ = newton_raphson(f2, df2, 1.0)
    print(f"Newton-Raphson:    sqrt(2) = {root:.15f}, iterations = {iters}")

    root, iters, _ = fixed_point(g, 1.0)
    print(f"Fixed-point iter.: sqrt(2) = {root:.15f}, iterations = {iters}")

    print(f"Actual sqrt(2):             {np.sqrt(2):.15f}")

    # Example 3: cos(x) = x fixed point
    print("\n[Example 3] cos(x) = x (Dottie Number)")
    print("-" * 40)

    g_cos = lambda x: np.cos(x)
    root, iters, _ = fixed_point(g_cos, 0.5)
    print(f"Fixed point x = cos(x): {root:.10f}, iterations = {iters}")

    print("\n" + "=" * 60)
    print("Root Finding Methods Comparison")
    print("=" * 60)
    print("""
    | Method          | Convergence | Advantages              | Disadvantages           |
    |-----------------|-------------|-------------------------|-------------------------|
    | Bisection       | Linear      | Always converges, stable| Slow, interval required |
    | Newton-Raphson  | Quadratic   | Very fast               | Derivative needed, may diverge|
    | Secant          | ~1.618      | No derivative needed    | Slower than Newton      |
    | Fixed-point     | Linear~Quad | Simple                  | Convergence check needed|
    | Brent           | Combined    | Stable + fast           | Complex implementation  |

    Practical recommendations:
    - scipy.optimize.brentq: Robust root finding
    - scipy.optimize.newton: Newton-Raphson/secant
    - scipy.optimize.fsolve: Multivariable equations
    """)


if __name__ == "__main__":
    main()

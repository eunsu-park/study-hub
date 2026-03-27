"""
Numerical Integration
Numerical Integration Methods

Methods for numerically computing the definite integral integral[a,b] f(x)dx.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# =============================================================================
# 1. Rectangular Rule
# =============================================================================
def rectangular_left(f: Callable, a: float, b: float, n: int) -> float:
    """
    Left rectangular rule
    Uses the function value at the left endpoint of each subinterval as the rectangle height
    """
    h = (b - a) / n
    result = 0
    for i in range(n):
        x = a + i * h
        result += f(x)
    return h * result


def rectangular_right(f: Callable, a: float, b: float, n: int) -> float:
    """Right rectangular rule"""
    h = (b - a) / n
    result = 0
    for i in range(1, n + 1):
        x = a + i * h
        result += f(x)
    return h * result


def rectangular_midpoint(f: Callable, a: float, b: float, n: int) -> float:
    """
    Midpoint rectangular rule (Midpoint Rule)
    Error: O(h^2)
    """
    h = (b - a) / n
    result = 0
    for i in range(n):
        x = a + (i + 0.5) * h
        result += f(x)
    return h * result


# =============================================================================
# 2. Trapezoidal Rule
# =============================================================================
def trapezoidal(f: Callable, a: float, b: float, n: int) -> float:
    """
    Trapezoidal rule
    Approximates each subinterval as a trapezoid
    Error: O(h^2)

    Formula: h/2 * [f(x_0) + 2*sum(f(x_i)) + f(x_n)]
    """
    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        result += 2 * f(x)

    return h * result / 2


def trapezoidal_vectorized(f: Callable, a: float, b: float, n: int) -> float:
    """Trapezoidal rule (NumPy vectorized)"""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return np.trapz(y, x)


# =============================================================================
# 3. Simpson's Rule
# =============================================================================
def simpsons_rule(f: Callable, a: float, b: float, n: int) -> float:
    """
    Simpson's 1/3 rule
    Approximates each subinterval with a quadratic polynomial
    Error: O(h^4) - very accurate

    Condition: n must be even
    Formula: h/3 * [f(x_0) + 4*sum(f(odd)) + 2*sum(f(even)) + f(x_n)]
    """
    if n % 2 != 0:
        n += 1  # Adjust to even

    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)

    return h * result / 3


def simpsons_38(f: Callable, a: float, b: float, n: int) -> float:
    """
    Simpson's 3/8 rule
    Cubic polynomial approximation
    Condition: n must be a multiple of 3
    """
    if n % 3 != 0:
        n = (n // 3 + 1) * 3

    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 3 == 0:
            result += 2 * f(x)
        else:
            result += 3 * f(x)

    return 3 * h * result / 8


# =============================================================================
# 4. Romberg Integration
# =============================================================================
def romberg(f: Callable, a: float, b: float, max_order: int = 5) -> float:
    """
    Romberg integration
    Repeatedly applies Richardson extrapolation to the trapezoidal rule
    Can achieve very high accuracy
    """
    R = np.zeros((max_order, max_order))

    # First column: trapezoidal rule
    h = b - a
    R[0, 0] = h * (f(a) + f(b)) / 2

    for i in range(1, max_order):
        h = h / 2
        # Sum of new points
        sum_new = sum(f(a + (2*k - 1) * h) for k in range(1, 2**(i-1) + 1))
        R[i, 0] = R[i-1, 0] / 2 + h * sum_new

        # Richardson extrapolation
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    return R[max_order-1, max_order-1]


# =============================================================================
# 5. Gaussian Quadrature
# =============================================================================
def gauss_legendre(f: Callable, a: float, b: float, n: int = 5) -> float:
    """
    Gauss-Legendre quadrature
    Exactly integrates polynomials up to degree (2n-1) with n points
    Very efficient

    Applies transformation from [-1, 1] to [a, b]
    """
    # n nodes and weights (precomputed values)
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # Interval transformation: [-1, 1] -> [a, b]
    # x = (b-a)/2 * t + (a+b)/2
    # dx = (b-a)/2 * dt

    transformed_nodes = (b - a) / 2 * nodes + (a + b) / 2
    result = sum(w * f(x) for x, w in zip(transformed_nodes, weights))

    return (b - a) / 2 * result


# =============================================================================
# 6. Adaptive Quadrature
# =============================================================================
def adaptive_simpsons(
    f: Callable,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 50
) -> Tuple[float, int]:
    """
    Adaptive Simpson's integration
    Recursively subdivides intervals where the error is large
    """
    call_count = [0]

    def _adaptive(a, b, fa, fb, fc, S, tol, depth):
        call_count[0] += 1
        c = (a + b) / 2
        d = (a + c) / 2
        e = (c + b) / 2

        fd = f(d)
        fe = f(e)

        S_left = (c - a) / 6 * (fa + 4*fd + fc)
        S_right = (b - c) / 6 * (fc + 4*fe + fb)
        S_new = S_left + S_right

        # Error estimate
        error = (S_new - S) / 15

        if depth >= max_depth or abs(error) <= tol:
            return S_new + error  # Richardson extrapolation
        else:
            left = _adaptive(a, c, fa, fc, fd, S_left, tol/2, depth+1)
            right = _adaptive(c, b, fc, fb, fe, S_right, tol/2, depth+1)
            return left + right

    fa, fb = f(a), f(b)
    fc = f((a + b) / 2)
    S = (b - a) / 6 * (fa + 4*fc + fb)

    result = _adaptive(a, b, fa, fb, fc, S, tol, 0)
    return result, call_count[0]


# =============================================================================
# Error Analysis and Visualization
# =============================================================================
def analyze_convergence(f: Callable, a: float, b: float, exact: float):
    """Convergence rate analysis"""
    ns = [4, 8, 16, 32, 64, 128, 256, 512]
    methods = {
        'Midpoint': rectangular_midpoint,
        'Trapezoidal': trapezoidal,
        'Simpson': simpsons_rule,
    }

    print("\nConvergence Analysis:")
    print("-" * 70)
    print(f"{'n':>6} | {'Midpoint':>14} | {'Trapezoidal':>14} | {'Simpson':>14}")
    print("-" * 70)

    errors = {name: [] for name in methods}

    for n in ns:
        row = f"{n:>6} |"
        for name, method in methods.items():
            result = method(f, a, b, n)
            error = abs(result - exact)
            errors[name].append(error)
            row += f" {error:>14.2e} |"
        print(row)

    return ns, errors


def plot_methods_comparison(f, a, b, n=10):
    """Integration method visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_dense = np.linspace(a, b, 500)
    y_dense = f(x_dense)

    methods = [
        ('Midpoint', rectangular_midpoint),
        ('Trapezoidal', trapezoidal),
        ('Simpson', simpsons_rule),
    ]

    # Function and integration area
    for ax, (name, method) in zip(axes.flat[:3], methods):
        ax.plot(x_dense, y_dense, 'b-', linewidth=2, label='f(x)')
        ax.fill_between(x_dense, y_dense, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

        h = (b - a) / n
        x_pts = np.linspace(a, b, n + 1)

        if 'mid' in name.lower():
            for i in range(n):
                xm = a + (i + 0.5) * h
                ax.bar(xm, f(xm), width=h, alpha=0.5, edgecolor='r', fill=False)
        elif 'trap' in name.lower():
            for i in range(n):
                x0, x1 = x_pts[i], x_pts[i+1]
                ax.fill([x0, x1, x1, x0], [0, 0, f(x1), f(x0)], alpha=0.5, edgecolor='r', fill=False)

        result = method(f, a, b, n)
        ax.set_title(f'{name}: {result:.6f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)

    # Convergence graph
    ax = axes[1, 1]
    ns, errors = [4, 8, 16, 32, 64, 128], {name: [] for name, _ in methods}
    exact = 2.0  # integral[0,pi] sin(x)dx = 2

    for n in ns:
        for name, method in methods:
            errors[name].append(abs(method(f, a, b, n) - exact))

    for name, errs in errors.items():
        ax.loglog(ns, errs, 'o-', label=name)

    ax.set_xlabel('n (number of subintervals)')
    ax.set_ylabel('Error')
    ax.set_title('Convergence Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/numerical_integration.png', dpi=150)
    plt.close()
    print("Graph saved: numerical_integration.png")


# =============================================================================
# Test
# =============================================================================
def main():
    print("=" * 60)
    print("Numerical Integration Examples")
    print("=" * 60)

    # Example 1: integral[0,pi] sin(x)dx = 2
    print("\n[Example 1] integral[0,pi] sin(x)dx = 2")
    print("-" * 50)

    f = np.sin
    a, b = 0, np.pi
    exact = 2.0

    n = 100
    results = {
        'Midpoint': rectangular_midpoint(f, a, b, n),
        'Trapezoidal': trapezoidal(f, a, b, n),
        'Simpson 1/3': simpsons_rule(f, a, b, n),
        'Romberg': romberg(f, a, b, 6),
        'Gauss-Legendre (5pt)': gauss_legendre(f, a, b, 5),
    }

    print(f"Exact value: {exact}")
    print(f"{'Method':<20} {'Result':<15} {'Error':<15}")
    print("-" * 50)
    for name, result in results.items():
        error = abs(result - exact)
        print(f"{name:<20} {result:<15.10f} {error:<15.2e}")

    # Example 2: integral[0,1] e^(-x^2)dx ~ 0.7468...
    print("\n[Example 2] integral[0,1] e^(-x^2)dx (Gaussian integral)")
    print("-" * 50)

    f2 = lambda x: np.exp(-x**2)
    exact2 = 0.746824132812427  # True value (using erf)

    for n in [10, 50, 100]:
        trap = trapezoidal(f2, 0, 1, n)
        simp = simpsons_rule(f2, 0, 1, n)
        print(f"n={n:3d}: Trapezoidal={trap:.10f}, Simpson={simp:.10f}")

    gauss = gauss_legendre(f2, 0, 1, 10)
    print(f"Gauss-Legendre (10pt): {gauss:.10f}")
    print(f"Exact value: {exact2:.10f}")

    # Example 3: Adaptive integration
    print("\n[Example 3] Adaptive integration (rapidly varying function)")
    print("-" * 50)

    f3 = lambda x: 1 / (1 + 100 * (x - 0.5)**2)  # Narrow peak
    exact3 = 0.3141277802329  # ~ pi/10

    trap = trapezoidal(f3, 0, 1, 100)
    result, calls = adaptive_simpsons(f3, 0, 1, tol=1e-8)

    print(f"Trapezoidal (n=100):  {trap:.10f}, error: {abs(trap - exact3):.2e}")
    print(f"Adaptive Simpson:     {result:.10f}, error: {abs(result - exact3):.2e}, calls: {calls}")

    # Visualization
    try:
        plot_methods_comparison(np.sin, 0, np.pi, 10)
    except Exception as e:
        print(f"Graph generation failed: {e}")

    # Convergence analysis
    print("\n" + "=" * 60)
    print("Convergence Rate Analysis (integral[0,pi] sin(x)dx)")
    analyze_convergence(np.sin, 0, np.pi, 2.0)

    print("\n" + "=" * 60)
    print("Numerical Integration Methods Comparison")
    print("=" * 60)
    print("""
    | Method              | Error Order | Characteristics                    |
    |---------------------|-------------|-------------------------------------|
    | Rectangular (L/R)   | O(h)        | Simplest, inaccurate               |
    | Midpoint            | O(h^2)      | Most accurate among rectangular    |
    | Trapezoidal         | O(h^2)      | Simple and efficient               |
    | Simpson 1/3         | O(h^4)      | Very accurate, good for smooth fns |
    | Romberg             | ~O(h^2k)    | Richardson extrapolation, high acc.|
    | Gaussian quadrature | ~O(h^2n)    | Maximum accuracy with minimum pts  |
    | Adaptive            | Variable    | Auto-subdivides rapidly varying    |

    Practical recommendations:
    - scipy.integrate.quad: Adaptive Gaussian quadrature (most general)
    - scipy.integrate.romberg: Romberg integration
    - scipy.integrate.simps: Simpson's rule (evenly spaced data)
    - scipy.integrate.trapz: Trapezoidal (evenly spaced data)
    """)


if __name__ == "__main__":
    main()

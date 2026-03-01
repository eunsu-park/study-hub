"""
Optimization and Root Finding

Demonstrates:
  - Finding extrema using first and second derivative tests
  - Newton's method for root finding with convergence analysis
  - Constrained optimization via Lagrange multipliers (SymPy)
  - Convergence visualization comparing bisection and Newton

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Finding Extrema with Derivative Tests
# ---------------------------------------------------------------------------
def find_extrema_symbolic(f_sym, x_sym):
    """Find and classify critical points using calculus.

    Strategy:
      1. Solve f'(x) = 0 to find critical points.
      2. Evaluate f''(x) at each critical point:
         - f'' > 0 => local minimum
         - f'' < 0 => local maximum
         - f'' = 0 => inconclusive (need higher-order test)
    """
    df = sp.diff(f_sym, x_sym)
    d2f = sp.diff(f_sym, x_sym, 2)

    critical_pts = sp.solve(df, x_sym)
    results = []

    for cp in critical_pts:
        # Only consider real critical points
        if not cp.is_real:
            continue
        d2f_val = d2f.subs(x_sym, cp)
        if d2f_val > 0:
            kind = "local minimum"
        elif d2f_val < 0:
            kind = "local maximum"
        else:
            kind = "inconclusive (inflection?)"
        results.append((float(cp), float(f_sym.subs(x_sym, cp)), kind))

    return results


def plot_extrema(f_func, extrema, x_range=(-3, 5)):
    """Plot the function and mark its extrema."""
    x = np.linspace(*x_range, 500)
    y = f_func(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "b-", lw=2, label="f(x)")

    for xc, yc, kind in extrema:
        marker = "v" if "min" in kind else "^"
        color = "green" if "min" in kind else "red"
        ax.plot(xc, yc, marker, color=color, ms=12, zorder=5,
                label=f"{kind} at x={xc:.2f}")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Extrema Detection via Derivative Tests")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("03_extrema.png", dpi=100)
    plt.close()
    print("[Saved] 03_extrema.png")


# ---------------------------------------------------------------------------
# 2. Newton's Method for Root Finding
# ---------------------------------------------------------------------------
def newtons_method(f, df, x0, tol=1e-14, max_iter=50):
    """Newton-Raphson iteration: x_{n+1} = x_n - f(x_n)/f'(x_n).

    Converges quadratically near simple roots — each iteration roughly
    doubles the number of correct digits.  However, it can diverge if
    the initial guess is far from the root or f'(x) ~ 0 near the iterate.
    """
    history = [(0, x0, f(x0))]
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            print("  Warning: derivative near zero — stopping.")
            break
        x_new = x - fx / dfx
        history.append((i, x_new, f(x_new)))

        if abs(x_new - x) < tol:
            break
        x = x_new

    return x, history


# ---------------------------------------------------------------------------
# 3. Constrained Optimization (Lagrange Multipliers)
# ---------------------------------------------------------------------------
def lagrange_multiplier_demo():
    """Minimize f(x,y) = x^2 + y^2 subject to g(x,y) = x + y - 1 = 0.

    The method of Lagrange multipliers converts a constrained problem into
    solving the system:  grad f = lambda * grad g,  g = 0.
    For this simple example the answer is (1/2, 1/2) with f = 1/2.
    """
    x, y, lam = sp.symbols("x y lambda")
    f = x ** 2 + y ** 2
    g = x + y - 1

    # Lagrangian: L = f - lambda * g
    L = f - lam * g
    grad_L = [sp.diff(L, v) for v in (x, y, lam)]

    solution = sp.solve(grad_L, (x, y, lam))
    print("Constrained Optimization (Lagrange Multipliers)")
    print(f"  Minimize  : f(x,y) = x^2 + y^2")
    print(f"  Subject to: x + y = 1")
    print(f"  Solution  : {solution}")
    f_val = f.subs([(x, solution[x]), (y, solution[y])])
    print(f"  f(x*,y*)  : {f_val}")
    return solution


# ---------------------------------------------------------------------------
# 4. Convergence Visualization
# ---------------------------------------------------------------------------
def plot_convergence(hist_newton, hist_bisect, true_root):
    """Compare convergence rates of Newton vs bisection.

    Newton converges quadratically (error ~ C * error_prev^2),
    bisection converges linearly (interval halves each step).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Newton errors
    newton_iters = [h[0] for h in hist_newton]
    newton_errors = [abs(h[1] - true_root) for h in hist_newton]
    newton_errors = [max(e, 1e-17) for e in newton_errors]  # for log scale
    ax.semilogy(newton_iters, newton_errors, "ro-", label="Newton", ms=6)

    # Bisection errors
    bisect_iters = [h[0] for h in hist_bisect]
    bisect_errors = [abs(h[1] - true_root) for h in hist_bisect]
    bisect_errors = [max(e, 1e-17) for e in bisect_errors]
    ax.semilogy(bisect_iters, bisect_errors, "bs-", label="Bisection", ms=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Absolute error")
    ax.set_title("Newton vs Bisection Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("03_convergence.png", dpi=100)
    plt.close()
    print("[Saved] 03_convergence.png")


def bisection(f, a, b, tol=1e-14, max_iter=60):
    """Standard bisection for comparison with Newton's method."""
    history = []
    for i in range(max_iter):
        mid = (a + b) / 2.0
        history.append((i, mid, f(mid)))
        if abs(f(mid)) < tol or (b - a) / 2 < tol:
            break
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return mid, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Demo 1: Extrema of f(x) = x^3 - 6x^2 + 9x + 1 ---
    print("=" * 60)
    print("Demo 1: Extrema via derivative tests")
    print("=" * 60)
    x_sym = sp.Symbol("x")
    f_sym = x_sym ** 3 - 6 * x_sym ** 2 + 9 * x_sym + 1
    extrema = find_extrema_symbolic(f_sym, x_sym)
    for xc, yc, kind in extrema:
        print(f"  x = {xc:.4f}, f(x) = {yc:.4f} — {kind}")

    f_np = lambda x: x ** 3 - 6 * x ** 2 + 9 * x + 1
    plot_extrema(f_np, extrema, x_range=(-0.5, 5))

    # --- Demo 2: Newton's method for x^3 - 2 = 0 (find cube root of 2) ---
    print("\nDemo 2: Newton's method for x^3 - 2 = 0")
    f_root = lambda x: x ** 3 - 2
    df_root = lambda x: 3 * x ** 2
    root, hist_n = newtons_method(f_root, df_root, x0=1.5)
    true_root = 2 ** (1 / 3)
    print(f"  Root found  : {root:.15f}")
    print(f"  True value  : {true_root:.15f}")
    print(f"  Iterations  : {len(hist_n) - 1}")

    # --- Demo 3: Lagrange multipliers ---
    print()
    lagrange_multiplier_demo()

    # --- Demo 4: Convergence comparison ---
    print("\nDemo 4: Convergence comparison (Newton vs Bisection)")
    _, hist_b = bisection(f_root, 1.0, 2.0)
    plot_convergence(hist_n, hist_b, true_root)
    print(f"  Newton iterations  : {len(hist_n) - 1}")
    print(f"  Bisection iterations: {len(hist_b)}")

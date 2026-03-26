"""
Taylor Series and Convergence Analysis

Demonstrates:
  - Taylor series partial sums visualization
  - Convergence tests: ratio test, root test, integral test
  - Radius of convergence computation
  - Lagrange error bound analysis

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Taylor Series Partial Sums
# ---------------------------------------------------------------------------
def taylor_coefficients(f_sym, x_sym, a, n_terms):
    """Compute Taylor coefficients symbolically using SymPy.

    The n-th Taylor coefficient about x=a is f^(n)(a) / n!.
    We use SymPy's diff to compute derivatives exactly.
    """
    coeffs = []
    for n in range(n_terms):
        dn = sp.diff(f_sym, x_sym, n)
        cn = dn.subs(x_sym, a) / sp.factorial(n)
        coeffs.append(float(cn))
    return coeffs


def taylor_partial_sum(coeffs, x, a):
    """Evaluate the Taylor polynomial: sum c_n * (x-a)^n."""
    result = np.zeros_like(x, dtype=float)
    for n, cn in enumerate(coeffs):
        result += cn * (x - a) ** n
    return result


def plot_taylor_approximations(f_sym, x_sym, f_np, a=0, orders=None,
                               x_range=(-4, 4)):
    """Visualize how increasing Taylor polynomial order improves accuracy.

    Each added term extends the region where the polynomial closely
    approximates the true function.
    """
    if orders is None:
        orders = [1, 3, 5, 9, 15]

    x = np.linspace(*x_range, 500)
    y_exact = f_np(x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y_exact, "k-", lw=2.5, label="f(x) exact")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(orders)))
    for order, color in zip(orders, colors):
        coeffs = taylor_coefficients(f_sym, x_sym, a, order + 1)
        y_taylor = taylor_partial_sum(coeffs, x, a)
        ax.plot(x, y_taylor, "--", color=color, lw=1.5,
                label=f"Order {order}")

    ax.set_ylim(min(y_exact) - 2, max(y_exact) + 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Taylor Series Approximations about x = {a}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_taylor_series.png", dpi=100)
    plt.close()
    print("[Saved] 05_taylor_series.png")


# ---------------------------------------------------------------------------
# 2. Convergence Tests
# ---------------------------------------------------------------------------
def ratio_test(a_func, n_start=1, n_terms=20):
    """Apply the ratio test: L = lim |a_{n+1}/a_n|.

    L < 1 => series converges absolutely
    L > 1 => series diverges
    L = 1 => test is inconclusive
    """
    ratios = []
    for n in range(n_start, n_start + n_terms):
        an = abs(a_func(n))
        an1 = abs(a_func(n + 1))
        if an > 0:
            ratios.append(an1 / an)
        else:
            ratios.append(0)

    L = ratios[-1] if ratios else None
    return L, ratios


def root_test(a_func, n_start=1, n_terms=20):
    """Apply the root test: L = lim |a_n|^{1/n}.

    Same convergence criteria as the ratio test.  Sometimes easier
    to evaluate when the general term involves n-th powers.
    """
    roots = []
    for n in range(n_start, n_start + n_terms):
        an = abs(a_func(n))
        roots.append(an ** (1.0 / n) if an > 0 else 0)
    L = roots[-1] if roots else None
    return L, roots


def integral_test_demo():
    """Demonstrate the integral test for sum 1/n^p (p-series).

    The series sum_{n=1}^inf 1/n^p converges iff integral_1^inf 1/x^p dx
    converges, which happens when p > 1.
    """
    print("Integral Test: p-series sum 1/n^p")
    print("-" * 50)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        # Compute partial sum for reference
        N = 10000
        partial_sum = sum(1.0 / n ** p for n in range(1, N + 1))

        # Integral from 1 to inf of 1/x^p
        if p == 1:
            converges = False
            integral_val = "diverges (ln(x))"
        elif p > 1:
            converges = True
            integral_val = f"{1.0 / (p - 1):.4f}"
        else:
            converges = False
            integral_val = "diverges"

        status = "CONVERGES" if converges else "DIVERGES"
        print(f"  p = {p:.1f}: S_{N} = {partial_sum:12.4f}  "
              f"integral = {integral_val:>14s}  => {status}")


# ---------------------------------------------------------------------------
# 3. Radius of Convergence
# ---------------------------------------------------------------------------
def compute_radius_of_convergence(coeffs_func, n_max=50):
    """Estimate the radius of convergence via the ratio test.

    For a power series sum a_n x^n, the radius R = lim |a_n / a_{n+1}|.
    We compute successive ratios and watch for convergence.
    """
    ratios = []
    for n in range(1, n_max):
        an = abs(coeffs_func(n))
        an1 = abs(coeffs_func(n + 1))
        if an1 > 1e-300:
            ratios.append(an / an1)

    # The radius of convergence is the limit of these ratios
    return ratios


def plot_radius_convergence():
    """Show convergence of R estimates for several series."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Series 1: e^x -> a_n = 1/n!, R = infinity
    coeffs_exp = lambda n: 1.0 / np.math.factorial(n) if n < 170 else 0
    r_exp = compute_radius_of_convergence(coeffs_exp, 50)
    ax.plot(range(len(r_exp)), r_exp, "o-", ms=3, label="e^x (R = inf)")

    # Series 2: 1/(1-x) -> a_n = 1, R = 1
    coeffs_geom = lambda n: 1.0
    r_geom = compute_radius_of_convergence(coeffs_geom, 50)
    ax.plot(range(len(r_geom)), r_geom, "s-", ms=3, label="1/(1-x) (R = 1)")

    # Series 3: ln(1+x) -> a_n = (-1)^{n+1}/n, R = 1
    coeffs_log = lambda n: 1.0 / n if n > 0 else 0
    r_log = compute_radius_of_convergence(coeffs_log, 50)
    ax.plot(range(len(r_log)), r_log, "^-", ms=3, label="ln(1+x) (R = 1)")

    ax.set_xlabel("n (term index)")
    ax.set_ylabel("Estimated R = |a_n / a_{n+1}|")
    ax.set_title("Radius of Convergence Estimation")
    ax.set_ylim(0, 60)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_radius_convergence.png", dpi=100)
    plt.close()
    print("[Saved] 05_radius_convergence.png")


# ---------------------------------------------------------------------------
# 4. Lagrange Error Bound
# ---------------------------------------------------------------------------
def lagrange_error_bound(f_sym, x_sym, a, n, x_eval):
    """Compute the Lagrange remainder bound for a Taylor polynomial.

    |R_n(x)| <= M * |x - a|^{n+1} / (n+1)!
    where M = max |f^{(n+1)}(c)| for c between a and x.
    """
    # Compute (n+1)-th derivative
    dn1 = sp.diff(f_sym, x_sym, n + 1)
    dn1_func = sp.lambdify(x_sym, sp.Abs(dn1), "numpy")

    # Estimate M on the interval [a, x_eval] (or [x_eval, a])
    lo, hi = min(a, x_eval), max(a, x_eval)
    c_vals = np.linspace(lo, hi, 1000)
    M = np.max(dn1_func(c_vals))

    bound = M * abs(x_eval - a) ** (n + 1) / np.math.factorial(n + 1)
    return bound


def error_bound_analysis():
    """Show how the Taylor error bound decreases with polynomial order."""
    x_sym = sp.Symbol("x")
    f_sym = sp.sin(x_sym)
    a = 0
    x_eval = 1.0

    print("Lagrange Error Bound for sin(x) Taylor series at x=1")
    print(f"{'Order':>6s} | {'Bound':>14s} | {'Actual Error':>14s}")
    print("-" * 42)

    f_np = np.sin
    for n in range(1, 16, 2):  # odd orders for sin
        bound = lagrange_error_bound(f_sym, x_sym, a, n, x_eval)
        coeffs = taylor_coefficients(f_sym, x_sym, a, n + 1)
        approx = taylor_partial_sum(coeffs, np.array([x_eval]), a)[0]
        actual = abs(approx - f_np(x_eval))
        print(f"{n:6d} | {bound:14.2e} | {actual:14.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x_sym = sp.Symbol("x")

    # --- Demo 1: Taylor series for sin(x) ---
    print("=" * 60)
    print("Demo 1: Taylor series partial sums for sin(x)")
    print("=" * 60)
    plot_taylor_approximations(
        sp.sin(x_sym), x_sym, np.sin, a=0,
        orders=[1, 3, 5, 9, 15], x_range=(-6, 6)
    )

    # --- Demo 2: Convergence tests ---
    print("\nDemo 2: Convergence tests")
    print("-" * 60)
    # Series: sum n^2 / 3^n  (converges by ratio test, L = 1/3)
    a_func = lambda n: n ** 2 / 3 ** n
    L_ratio, ratios = ratio_test(a_func)
    L_root, roots = root_test(a_func)
    print(f"  Series: sum n^2 / 3^n")
    print(f"  Ratio test limit : {L_ratio:.6f} (< 1 => converges)")
    print(f"  Root test limit  : {L_root:.6f}")

    print()
    integral_test_demo()

    # --- Demo 3: Radius of convergence ---
    print("\nDemo 3: Radius of convergence estimation")
    plot_radius_convergence()

    # --- Demo 4: Error bound analysis ---
    print("\nDemo 4: Lagrange error bound")
    error_bound_analysis()

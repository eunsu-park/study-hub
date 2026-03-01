"""
Limits and Continuity — Numerical Exploration

Demonstrates:
  - Numerical limit evaluation (approaching from both sides)
  - Epsilon-delta visualization of the limit definition
  - Discontinuity detection via sampling
  - Intermediate Value Theorem (IVT) bisection root finder

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Numerical Limit Evaluation
# ---------------------------------------------------------------------------
def numerical_limit(f, a, n_points=12):
    """Evaluate lim_{x->a} f(x) by approaching from both sides.

    We use geometrically-shrinking step sizes so that each row is 10x closer
    to `a` than the previous one — this reveals convergence (or divergence)
    clearly in a table.
    """
    offsets = [10 ** (-k) for k in range(1, n_points + 1)]
    left_vals = [f(a - h) for h in offsets]
    right_vals = [f(a + h) for h in offsets]

    print(f"{'h':>14s} | {'f(a-h)':>22s} | {'f(a+h)':>22s}")
    print("-" * 64)
    for h, lv, rv in zip(offsets, left_vals, right_vals):
        print(f"{h:14.1e} | {lv:22.15f} | {rv:22.15f}")

    return left_vals[-1], right_vals[-1]


# ---------------------------------------------------------------------------
# 2. Epsilon-Delta Visualization
# ---------------------------------------------------------------------------
def plot_epsilon_delta(f, a, L, epsilon=0.3, delta=None):
    """Visualize the epsilon-delta definition of a limit.

    For every epsilon > 0 there exists delta > 0 such that
    0 < |x - a| < delta  =>  |f(x) - L| < epsilon.
    If delta is not provided we estimate it by sampling.
    """
    if delta is None:
        # Find a suitable delta by bisection on |f(a+h) - L| < epsilon
        delta = epsilon  # initial guess
        for _ in range(60):
            test_x = np.linspace(a - delta, a + delta, 400)
            # Avoid the point a itself (limit need not equal f(a))
            mask = np.abs(test_x - a) > 1e-14
            if np.all(np.abs(f(test_x[mask]) - L) < epsilon):
                break
            delta *= 0.5

    x = np.linspace(a - 1.5, a + 1.5, 800)
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "b-", lw=2, label="f(x)")

    # Epsilon band (horizontal)
    ax.axhspan(L - epsilon, L + epsilon, color="green", alpha=0.15,
               label=f"L +/- eps  ({L - epsilon:.2f}, {L + epsilon:.2f})")
    # Delta band (vertical)
    ax.axvspan(a - delta, a + delta, color="orange", alpha=0.15,
               label=f"a +/- delta ({a - delta:.3f}, {a + delta:.3f})")

    ax.axhline(L, color="green", ls="--", lw=0.8)
    ax.axvline(a, color="orange", ls="--", lw=0.8)
    ax.plot(a, L, "ro", ms=8, zorder=5, label=f"Limit = {L:.4f}")

    ax.set_title(f"Epsilon-Delta: eps={epsilon}, delta={delta:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("01_epsilon_delta.png", dpi=100)
    plt.close()
    print("[Saved] 01_epsilon_delta.png")


# ---------------------------------------------------------------------------
# 3. Discontinuity Detection
# ---------------------------------------------------------------------------
def detect_discontinuities(f, x_range, n_samples=2000, threshold=0.5):
    """Detect likely discontinuities by looking for large jumps in f(x).

    The idea is simple: sample densely, compute |f(x_{i+1}) - f(x_i)|,
    and flag locations where this jump exceeds a threshold relative to
    the step size.  This catches jump discontinuities but not removable ones.
    """
    x = np.linspace(*x_range, n_samples)
    y = f(x)
    jumps = np.abs(np.diff(y))
    dx = x[1] - x[0]

    # Normalize by step size to get an approximate derivative magnitude
    candidates = np.where(jumps / dx > threshold * np.max(jumps / dx))[0]

    # Cluster nearby indices (within 3 samples) into single discontinuity
    disc_points = []
    if len(candidates) > 0:
        cluster = [candidates[0]]
        for idx in candidates[1:]:
            if idx - cluster[-1] <= 3:
                cluster.append(idx)
            else:
                disc_points.append(x[int(np.mean(cluster))])
                cluster = [idx]
        disc_points.append(x[int(np.mean(cluster))])

    return disc_points


# ---------------------------------------------------------------------------
# 4. IVT Bisection Root Finder
# ---------------------------------------------------------------------------
def ivt_bisection(f, a, b, tol=1e-12, max_iter=100):
    """Find a root of f on [a, b] using the Intermediate Value Theorem.

    The IVT guarantees a root exists when f(a) and f(b) have opposite signs
    (assuming f is continuous).  Bisection halves the interval each step,
    giving linear convergence with factor 1/2.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    history = []
    for i in range(max_iter):
        mid = (a + b) / 2.0
        fm = f(mid)
        history.append((i, mid, fm, b - a))

        if abs(fm) < tol or (b - a) / 2 < tol:
            break

        # Narrow the interval to the half that contains the sign change
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    return mid, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Demo 1: lim_{x->0} sin(x)/x = 1 ---
    print("=" * 64)
    print("Demo 1: Numerical limit of sin(x)/x as x -> 0")
    print("=" * 64)
    sinc = lambda x: np.sin(x) / x
    numerical_limit(sinc, 0.0)

    # --- Demo 2: Epsilon-delta for f(x) = x^2 at x=2, L=4 ---
    print("\nDemo 2: Epsilon-delta visualization for x^2 at x=2")
    plot_epsilon_delta(lambda x: x ** 2, a=2, L=4, epsilon=0.5)

    # --- Demo 3: Detect discontinuities in floor(x) ---
    print("\nDemo 3: Discontinuity detection in floor(x)")
    floor_disc = detect_discontinuities(np.floor, (-3, 3))
    print(f"  Detected discontinuities near: {[f'{d:.2f}' for d in floor_disc]}")

    # --- Demo 4: IVT bisection for cos(x) = x  (find root of cos(x)-x) ---
    print("\nDemo 4: IVT bisection root finder for cos(x) - x = 0")
    root, hist = ivt_bisection(lambda x: np.cos(x) - x, 0, np.pi / 2)
    print(f"  Root found: {root:.15f}")
    print(f"  Iterations: {len(hist)}")
    print(f"  Residual  : {np.cos(root) - root:.2e}")

    # Show convergence table (last 6 iterations)
    print(f"\n  {'Iter':>4s} | {'Midpoint':>20s} | {'f(mid)':>14s} | {'Interval':>12s}")
    for it, mid, fm, width in hist[-6:]:
        print(f"  {it:4d} | {mid:20.15f} | {fm:14.2e} | {width:12.2e}")

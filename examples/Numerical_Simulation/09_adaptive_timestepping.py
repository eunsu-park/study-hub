#!/usr/bin/env python3
"""
Adaptive Time-Stepping with Embedded Runge-Kutta Methods
=========================================================

Implements adaptive step-size control using embedded RK methods,
specifically the Dormand-Prince RK4(5) pair (the method behind
MATLAB's ode45 and SciPy's solve_ivp default).

Key Idea:
    Embedded RK methods compute two solutions of different orders
    (4th and 5th) using the SAME function evaluations. The difference
    between them estimates the local truncation error, which drives
    step-size adaptation.

Why adaptive stepping?
    - Fixed step wastes compute on smooth regions and fails on stiff regions
    - Adaptive steps are small where the solution changes rapidly and
      large where it is smooth — optimal use of function evaluations
    - Error is controlled to user-specified tolerance automatically

Key Concepts:
    - Embedded Runge-Kutta pairs (Butcher tableau)
    - Local error estimation
    - PI controller for step size
    - Stiff ODE detection heuristic

Author: Educational example for Numerical Simulation
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional


# =============================================================================
# Dormand-Prince RK4(5) Coefficients (Butcher Tableau)
# =============================================================================
# Why Dormand-Prince over Fehlberg?
#   DP uses the 5th-order solution as the propagated value (local extrapolation),
#   giving better accuracy per step. Fehlberg uses the 4th-order solution.

# Nodes (c_i): time offsets within each step
DP_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

# Runge-Kutta matrix (a_ij): how stages depend on previous stages
DP_A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
])

# 5th-order weights (b_i): used for the propagated solution
DP_B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

# 4th-order weights: used only for error estimation
DP_B4 = np.array([5179/57600, 0, 7571/16695, 393/640,
                   -92097/339200, 187/2100, 1/40])

# Error coefficients: difference between 5th and 4th order
DP_E = DP_B5 - DP_B4


def rkdp45_step(f: Callable, t: float, y: np.ndarray, h: float
                ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform one Dormand-Prince RK4(5) step.

    Returns the 5th-order solution and the estimated local error.

    Why 7 stages for a 5th-order method?
        The 7th stage of step n equals the 1st stage of step n+1 (FSAL property),
        so effectively only 6 new evaluations per step.
    """
    n = len(y)
    K = np.zeros((7, n))

    # Compute all 7 stages
    K[0] = f(t, y)
    for i in range(1, 7):
        y_stage = y + h * np.dot(DP_A[i, :i], K[:i])
        K[i] = f(t + DP_C[i] * h, y_stage)

    # 5th-order solution (propagated value)
    y_new = y + h * np.dot(DP_B5, K)

    # Error estimate: difference between 5th and 4th order
    # Why: This costs zero extra function evaluations — just a different
    # linear combination of the same K values.
    error_vec = h * np.dot(DP_E, K)
    error_norm = np.sqrt(np.mean((error_vec / np.maximum(
        np.abs(y_new), 1e-10)) ** 2))

    return y_new, K[6], error_norm


def solve_adaptive(f: Callable, t_span: Tuple[float, float], y0: np.ndarray,
                   atol: float = 1e-8, rtol: float = 1e-6,
                   h_init: Optional[float] = None,
                   h_min: float = 1e-12, h_max: float = 1.0,
                   max_steps: int = 100000
                   ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Solve ODE y' = f(t, y) with adaptive RK4(5) stepping.

    Uses a PI controller for step-size selection (more stable than
    the standard I-controller).

    Args:
        f: Right-hand side function f(t, y)
        t_span: (t_start, t_end)
        y0: Initial condition
        atol: Absolute error tolerance
        rtol: Relative error tolerance
        h_init: Initial step size (auto-selected if None)
        h_min, h_max: Step size bounds
        max_steps: Maximum number of steps

    Returns:
        t_array, y_array, info_dict
    """
    t0, tf = t_span
    y = np.array(y0, dtype=float)

    # Auto-select initial step size
    # Why: h ~ (atol / |f(t0, y0)|)^(1/5) balances error vs step count.
    if h_init is None:
        f0_norm = np.linalg.norm(f(t0, y))
        h_init = min(h_max, max(h_min, 0.01 * (atol / max(f0_norm, 1e-10)) ** 0.2))

    h = h_init
    t = t0

    t_list = [t]
    y_list = [y.copy()]
    h_list = []
    err_list = []

    # PI controller parameters
    # Why: A pure I-controller (h_new = h * (tol/err)^(1/5)) oscillates.
    # The PI controller adds a damping term using the previous error ratio.
    beta1 = 0.7 / 5   # Integral gain (divided by method order)
    beta2 = 0.4 / 5   # Proportional gain
    prev_err = 1.0
    n_reject = 0
    n_steps = 0

    while t < tf and n_steps < max_steps:
        h = min(h, tf - t)  # Don't overshoot the end

        y_new, _, err = rkdp45_step(f, t, y, h)

        # Compute tolerance: mixed absolute + relative
        # Why: atol handles near-zero values, rtol handles large values.
        tol = atol + rtol * max(np.max(np.abs(y)), np.max(np.abs(y_new)))
        err_ratio = err / max(tol, 1e-30)

        if err_ratio <= 1.0:
            # Step accepted
            t += h
            y = y_new
            t_list.append(t)
            y_list.append(y.copy())
            h_list.append(h)
            err_list.append(err)
            n_steps += 1

            # PI controller: compute new step size
            # Why: Using both current and previous error ratios smooths
            # the step-size trajectory, avoiding the "sawtooth" pattern.
            factor = (1.0 / max(err_ratio, 1e-10)) ** beta1 * \
                     prev_err ** beta2
            prev_err = err_ratio
        else:
            # Step rejected — reduce step size and retry
            n_reject += 1
            factor = (1.0 / max(err_ratio, 1e-10)) ** (1.0 / 5)

        # Safety factor prevents overly aggressive step changes
        factor = min(5.0, max(0.2, 0.9 * factor))
        h = np.clip(h * factor, h_min, h_max)

    info = {
        "n_steps": n_steps,
        "n_reject": n_reject,
        "h_history": np.array(h_list),
        "err_history": np.array(err_list),
    }
    return np.array(t_list), np.array(y_list), info


def detect_stiffness(h_history: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Heuristic stiffness detection.

    If the step size is forced to be very small despite smooth-looking
    solutions, the problem may be stiff and an implicit method is needed.

    Why this matters:
        Explicit RK methods (like RK4/5) become unstable for stiff ODEs,
        requiring impractically small steps. Implicit methods (BDF, SDIRK)
        remain stable with much larger steps.
    """
    if len(h_history) < 20:
        return False
    # Check if step size ratio (max/min over recent window) is extreme
    recent = h_history[-20:]
    ratio = recent.max() / max(recent.min(), 1e-30)
    return ratio > threshold


def demo_problems():
    """Demonstrate adaptive stepping on various ODE problems."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Problem 1: Smooth oscillator (SHO)
    # y'' + y = 0 → y1' = y2, y2' = -y1
    def sho(t, y):
        return np.array([y[1], -y[0]])

    t, y, info = solve_adaptive(sho, (0, 20 * np.pi), np.array([1.0, 0.0]))
    axes[0, 0].plot(t, y[:, 0], 'b-', linewidth=0.8)
    axes[0, 0].set_title(f"Simple Harmonic Oscillator\n"
                         f"{info['n_steps']} steps, {info['n_reject']} rejected")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("y(t)")

    axes[0, 1].semilogy(info['h_history'], 'g-', linewidth=0.5)
    axes[0, 1].set_title("Step Size History (SHO)")
    axes[0, 1].set_ylabel("h")

    # Problem 2: Van der Pol oscillator (stiff for large mu)
    # Why: Van der Pol with large mu has thin boundary layers where the
    # solution changes rapidly, requiring small steps. Between layers,
    # the solution is nearly constant and large steps suffice.
    mu = 10.0
    def vanderpol(t, y):
        return np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])

    t, y, info = solve_adaptive(vanderpol, (0, 50), np.array([2.0, 0.0]),
                                rtol=1e-6, atol=1e-9)
    axes[1, 0].plot(t, y[:, 0], 'r-', linewidth=0.8)
    axes[1, 0].set_title(f"Van der Pol (mu={mu})\n"
                         f"{info['n_steps']} steps, {info['n_reject']} rejected")
    axes[1, 0].set_xlabel("t")

    axes[1, 1].semilogy(info['h_history'], 'g-', linewidth=0.5)
    axes[1, 1].set_title("Step Size History (Van der Pol)")
    axes[1, 1].set_ylabel("h")
    is_stiff = detect_stiffness(info['h_history'])
    axes[1, 1].text(0.02, 0.95, f"Stiff detected: {is_stiff}",
                    transform=axes[1, 1].transAxes, fontsize=9,
                    verticalalignment='top')

    # Problem 3: Lorenz system (chaotic)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    def lorenz(t, y):
        return np.array([
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2]
        ])

    t, y, info = solve_adaptive(lorenz, (0, 50), np.array([1.0, 1.0, 1.0]),
                                rtol=1e-8, atol=1e-10)
    axes[2, 0].plot(y[:, 0], y[:, 2], 'purple', linewidth=0.3, alpha=0.7)
    axes[2, 0].set_title(f"Lorenz Attractor (x-z phase)\n"
                         f"{info['n_steps']} steps, {info['n_reject']} rejected")
    axes[2, 0].set_xlabel("x")
    axes[2, 0].set_ylabel("z")

    axes[2, 1].semilogy(info['h_history'], 'g-', linewidth=0.5)
    axes[2, 1].set_title("Step Size History (Lorenz)")
    axes[2, 1].set_ylabel("h")
    axes[2, 1].set_xlabel("Step number")

    plt.suptitle("Adaptive RK4(5) Time-Stepping", fontsize=14)
    plt.tight_layout()
    plt.savefig("adaptive_timestepping.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: adaptive_timestepping.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive Time-Stepping with Dormand-Prince RK4(5)")
    print("=" * 60)

    # Accuracy test: exponential decay y' = -y, y(0) = 1
    print("\n--- Accuracy Test: y' = -y ---")
    def exp_decay(t, y):
        return -y

    for tol in [1e-4, 1e-6, 1e-8, 1e-10]:
        t, y, info = solve_adaptive(exp_decay, (0, 10), np.array([1.0]),
                                    rtol=tol, atol=tol * 0.1)
        exact = np.exp(-t[-1])
        error = abs(y[-1, 0] - exact)
        print(f"  tol={tol:.0e}: error={error:.2e}, "
              f"steps={info['n_steps']}, rejected={info['n_reject']}")

    # Demonstrate on various problems
    print("\nGenerating visualization...")
    demo_problems()

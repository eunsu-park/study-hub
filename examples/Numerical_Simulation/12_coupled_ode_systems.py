#!/usr/bin/env python3
"""
Coupled ODE Systems and Phase Space Visualization
===================================================

Implements and visualizes classic coupled ODE systems:
1. Lotka-Volterra (predator-prey dynamics)
2. Double Pendulum (chaotic mechanical system)
3. Lorenz Attractor (chaotic flow, weather model)

Key Idea:
    Many physical, biological, and engineering systems are described by
    coupled ODEs: dx/dt = f(x, y, ...). Phase space visualization reveals
    the qualitative behavior (fixed points, limit cycles, chaos) that
    time series alone cannot show.

Why these three systems?
    - Lotka-Volterra: closed orbits (conservative), ecological modeling
    - Double Pendulum: transition from periodic to chaotic motion
    - Lorenz: strange attractor, sensitive dependence on initial conditions

Key Concepts:
    - Phase portraits and state space trajectories
    - Fixed point analysis and stability
    - Lyapunov exponents (chaos quantification)
    - Sensitivity to initial conditions (butterfly effect)

Author: Educational example for Numerical Simulation
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Callable, Tuple


# =============================================================================
# 1. Lotka-Volterra (Predator-Prey)
# =============================================================================

def lotka_volterra(t, state, alpha=1.0, beta=0.5, delta=0.2, gamma=0.8):
    """
    Lotka-Volterra predator-prey equations.

    dx/dt = alpha*x - beta*x*y    (prey growth minus predation)
    dy/dt = delta*x*y - gamma*y   (predator growth from prey minus death)

    Why this model matters:
        It's the simplest model of two-species interaction, yet captures
        the key phenomenon: population oscillations where predator peaks
        lag behind prey peaks.

    Fixed points:
        (0, 0) — extinction (unstable saddle)
        (gamma/delta, alpha/beta) — coexistence (neutrally stable center)
    """
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


def plot_lotka_volterra():
    """Solve and visualize Lotka-Volterra system."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Solve for multiple initial conditions
    t_span = (0, 40)
    t_eval = np.linspace(*t_span, 2000)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))

    for i, (x0, y0) in enumerate([(2, 1), (4, 1), (2, 3), (5, 2)]):
        sol = solve_ivp(lotka_volterra, t_span, [x0, y0], t_eval=t_eval,
                        rtol=1e-10, atol=1e-12)
        axes[0].plot(sol.t, sol.y[0], color=colors[i], linewidth=0.8,
                     label=f'Prey (x0={x0})')
        axes[0].plot(sol.t, sol.y[1], '--', color=colors[i], linewidth=0.8,
                     label=f'Predator (y0={y0})')
        axes[1].plot(sol.y[0], sol.y[1], color=colors[i], linewidth=0.8)

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Population")
    axes[0].set_title("Time Series")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Prey (x)")
    axes[1].set_ylabel("Predator (y)")
    axes[1].set_title("Phase Portrait")
    # Mark the fixed point
    # Why: The fixed point (gamma/delta, alpha/beta) is a center — orbits
    # close around it but never converge to it (no energy dissipation).
    fp_x, fp_y = 0.8 / 0.2, 1.0 / 0.5
    axes[1].plot(fp_x, fp_y, 'r*', markersize=12, label='Fixed point')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Vector field
    x_range = np.linspace(0.5, 8, 15)
    y_range = np.linspace(0.5, 6, 12)
    X, Y = np.meshgrid(x_range, y_range)
    DX = 1.0 * X - 0.5 * X * Y
    DY = 0.2 * X * Y - 0.8 * Y
    speed = np.sqrt(DX ** 2 + DY ** 2)
    axes[2].streamplot(X, Y, DX, DY, color=speed, cmap='coolwarm',
                       density=1.5, linewidth=0.8)
    axes[2].plot(fp_x, fp_y, 'r*', markersize=12)
    axes[2].set_xlabel("Prey (x)")
    axes[2].set_ylabel("Predator (y)")
    axes[2].set_title("Vector Field + Streamlines")

    plt.suptitle("Lotka-Volterra Predator-Prey", fontsize=13)
    plt.tight_layout()
    return fig


# =============================================================================
# 2. Double Pendulum
# =============================================================================

def double_pendulum(t, state, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """
    Double pendulum equations of motion (Lagrangian mechanics).

    State: [theta1, theta2, omega1, omega2]

    Why the double pendulum?
        It's the simplest mechanical system that exhibits chaos. Small
        differences in initial angle lead to completely different trajectories
        after a few swings — the "butterfly effect" made visible.

    The equations are derived from the Lagrangian L = T - V and are
    quite involved due to coupling between the two pendulum arms.
    """
    th1, th2, w1, w2 = state
    dth = th1 - th2

    # Why: The denominator approaches zero when the arms are nearly
    # aligned (dth ≈ 0), making the system numerically sensitive.
    den = 2 * m1 + m2 - m2 * np.cos(2 * dth)

    # Angular acceleration of pendulum 1
    dw1 = (-g * (2 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2 * th2)
            - 2 * np.sin(dth) * m2 * (w2 ** 2 * L2 + w1 ** 2 * L1 * np.cos(dth))
           ) / (L1 * den)

    # Angular acceleration of pendulum 2
    dw2 = (2 * np.sin(dth) *
           (w1 ** 2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1)
            + w2 ** 2 * L2 * m2 * np.cos(dth))
           ) / (L2 * den)

    return [w1, w2, dw1, dw2]


def plot_double_pendulum():
    """Solve and visualize the double pendulum, showing chaos."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 5000)

    # Two nearby initial conditions (demonstrating chaos)
    th1_0 = np.pi / 2
    eps_list = [0, 1e-6, 1e-3]
    colors = ['blue', 'red', 'green']
    labels = ['th2=2.0', 'th2=2.0+1e-6', 'th2=2.0+1e-3']

    solutions = []
    for eps, color, label in zip(eps_list, colors, labels):
        y0 = [th1_0, 2.0 + eps, 0, 0]
        sol = solve_ivp(double_pendulum, t_span, y0, t_eval=t_eval,
                        rtol=1e-10, atol=1e-12)
        solutions.append(sol)

        # Time series of theta1
        axes[0].plot(sol.t, sol.y[0], color=color, linewidth=0.5,
                     alpha=0.8, label=label)

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("theta_1 (rad)")
    axes[0].set_title("Chaos: Sensitivity to Initial Conditions")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Phase portrait (theta1 vs omega1)
    for sol, color in zip(solutions, colors):
        axes[1].plot(sol.y[0], sol.y[2], color=color, linewidth=0.3, alpha=0.6)
    axes[1].set_xlabel("theta_1")
    axes[1].set_ylabel("omega_1")
    axes[1].set_title("Phase Portrait (arm 1)")
    axes[1].grid(True, alpha=0.3)

    # Trajectory of pendulum tip
    # Why: Converting to Cartesian shows the actual motion path,
    # which helps build intuition for the chaotic behavior.
    L1, L2 = 1.0, 1.0
    sol = solutions[0]
    x1 = L1 * np.sin(sol.y[0])
    y1 = -L1 * np.cos(sol.y[0])
    x2 = x1 + L2 * np.sin(sol.y[1])
    y2 = y1 - L2 * np.cos(sol.y[1])

    axes[2].plot(x2, y2, 'purple', linewidth=0.2, alpha=0.5)
    axes[2].plot(x2[0], y2[0], 'go', markersize=8, label='Start')
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Tip Trajectory (Cartesian)")
    axes[2].set_aspect('equal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Double Pendulum — Chaotic Dynamics", fontsize=13)
    plt.tight_layout()
    return fig


# =============================================================================
# 3. Lorenz Attractor
# =============================================================================

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Lorenz system — the canonical example of deterministic chaos.

    dx/dt = sigma * (y - x)         (fluid convection rate)
    dy/dt = x * (rho - z) - y       (temperature difference)
    dz/dt = x * y - beta * z        (nonlinear heat transport)

    Why these parameter values?
        sigma=10, rho=28, beta=8/3 are Lorenz's original parameters.
        For rho > 24.74, the system exhibits chaos: trajectories orbit
        two unstable fixed points, switching between them unpredictably.
    """
    x, y, z = state
    return [sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z]


def estimate_lyapunov(f: Callable, t_span: Tuple, y0: np.ndarray,
                      dt: float = 0.01, n_renorm: int = 1000) -> float:
    """
    Estimate the largest Lyapunov exponent using the tangent linear method.

    The Lyapunov exponent lambda measures the average exponential rate of
    divergence of nearby trajectories:
        |delta(t)| ~ |delta(0)| * exp(lambda * t)

    lambda > 0 → chaos (nearby trajectories diverge exponentially)
    lambda = 0 → marginally stable
    lambda < 0 → stable (trajectories converge)

    Why this matters:
        A positive Lyapunov exponent is the mathematical definition of chaos.
        It quantifies how fast prediction accuracy degrades over time.
    """
    eps = 1e-8
    y = np.array(y0, dtype=float)
    lam_sum = 0.0

    for _ in range(n_renorm):
        # Evolve reference trajectory
        sol = solve_ivp(f, (0, dt), y, rtol=1e-10, atol=1e-12)
        y_new = sol.y[:, -1]

        # Evolve perturbed trajectory
        y_pert = y + eps * np.array([1, 0, 0])
        sol_p = solve_ivp(f, (0, dt), y_pert, rtol=1e-10, atol=1e-12)
        y_pert_new = sol_p.y[:, -1]

        # Measure divergence
        delta = np.linalg.norm(y_pert_new - y_new)
        lam_sum += np.log(delta / eps)

        y = y_new

    return lam_sum / (n_renorm * dt)


def plot_lorenz():
    """Solve and visualize the Lorenz attractor."""
    fig = plt.figure(figsize=(16, 10))

    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 20000)

    # Solve from two nearby initial conditions
    y0_a = [1.0, 1.0, 1.0]
    y0_b = [1.0 + 1e-10, 1.0, 1.0]

    sol_a = solve_ivp(lorenz, t_span, y0_a, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    sol_b = solve_ivp(lorenz, t_span, y0_b, t_eval=t_eval, rtol=1e-10, atol=1e-12)

    # 3D attractor
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(sol_a.y[0], sol_a.y[1], sol_a.y[2], 'b-', linewidth=0.3, alpha=0.7)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Lorenz Attractor (3D)")
    ax1.view_init(elev=25, azim=120)

    # x-z projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(sol_a.y[0], sol_a.y[2], 'b-', linewidth=0.2, alpha=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title("x-z Projection")
    ax2.grid(True, alpha=0.3)

    # Butterfly effect: divergence of nearby trajectories
    ax3 = fig.add_subplot(2, 2, 3)
    diff = np.sqrt(np.sum((sol_a.y - sol_b.y) ** 2, axis=0))
    ax3.semilogy(sol_a.t, diff, 'r-', linewidth=0.8)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("|delta(t)|")
    ax3.set_title("Butterfly Effect\n(delta_0 = 1e-10, exponential divergence)")
    ax3.grid(True, alpha=0.3)

    # Time series comparison
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(sol_a.t, sol_a.y[0], 'b-', linewidth=0.5, alpha=0.7, label='Trajectory A')
    ax4.plot(sol_b.t, sol_b.y[0], 'r--', linewidth=0.5, alpha=0.7, label='Trajectory B')
    ax4.set_xlabel("Time")
    ax4.set_ylabel("x(t)")
    ax4.set_title("x(t) — Divergence After t~20")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Lorenz System — Deterministic Chaos", fontsize=14)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Coupled ODE Systems and Phase Space Visualization")
    print("=" * 60)

    # 1. Lotka-Volterra
    print("\n--- 1. Lotka-Volterra Predator-Prey ---")
    fig1 = plot_lotka_volterra()
    plt.savefig("coupled_lotka_volterra.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: coupled_lotka_volterra.png")

    # 2. Double Pendulum
    print("\n--- 2. Double Pendulum ---")
    fig2 = plot_double_pendulum()
    plt.savefig("coupled_double_pendulum.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: coupled_double_pendulum.png")

    # 3. Lorenz Attractor
    print("\n--- 3. Lorenz Attractor ---")
    fig3 = plot_lorenz()
    plt.savefig("coupled_lorenz.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: coupled_lorenz.png")

    # 4. Lyapunov exponent estimation
    print("\n--- 4. Lyapunov Exponent (Lorenz) ---")
    lam = estimate_lyapunov(lorenz, (0, 100), [1.0, 1.0, 1.0])
    print(f"  Estimated largest Lyapunov exponent: {lam:.3f}")
    print(f"  (Expected ≈ 0.905 for sigma=10, rho=28, beta=8/3)")
    print(f"  Positive => system is chaotic")

    print("\nDone.")

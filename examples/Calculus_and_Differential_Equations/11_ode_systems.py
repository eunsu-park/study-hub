"""
Systems of ODEs — Phase Portraits and Stability

Demonstrates:
  - Phase portraits for linear 2D systems (node, saddle, spiral, center)
  - Eigenvalue classification and stability
  - Lotka-Volterra predator-prey model
  - Stability analysis of equilibrium points

Dependencies: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# 1. Linear 2D System Phase Portraits
# ---------------------------------------------------------------------------
def plot_phase_portrait(A, title="Phase Portrait", x_range=(-3, 3),
                        n_trajectories=20, filename="11_phase.png"):
    """Plot the phase portrait of dx/dt = A @ x for a 2x2 matrix A.

    We generate trajectories from initial conditions distributed around
    the origin.  The eigenvalues of A determine the topology:
      - Real, same sign      => stable/unstable node
      - Real, opposite signs => saddle point
      - Complex, Re != 0     => stable/unstable spiral
      - Pure imaginary        => center (periodic orbits)
    """
    eigvals = np.linalg.eigvals(A)

    def rhs(t, state):
        return A @ state

    fig, ax = plt.subplots(figsize=(7, 7))

    # Stream plot for background flow
    x = np.linspace(*x_range, 30)
    y = np.linspace(*x_range, 30)
    X, Y = np.meshgrid(x, y)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    speed = np.sqrt(U ** 2 + V ** 2)
    ax.streamplot(X, Y, U, V, color=speed, cmap="coolwarm",
                  linewidth=0.8, density=1.5, arrowsize=1.2)

    # Individual trajectories from a ring of initial conditions
    angles = np.linspace(0, 2 * np.pi, n_trajectories, endpoint=False)
    r0 = 2.5
    for theta in angles:
        ic = [r0 * np.cos(theta), r0 * np.sin(theta)]
        sol = solve_ivp(rhs, [0, 10], ic, max_step=0.05,
                        t_eval=np.linspace(0, 10, 500))
        ax.plot(sol.y[0], sol.y[1], "k-", lw=0.5, alpha=0.3)

    ax.plot(0, 0, "ko", ms=8)
    ax.set_xlim(*x_range)
    ax.set_ylim(*x_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}\neigenvalues: {eigvals[0]:.2f}, {eigvals[1]:.2f}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"[Saved] {filename}")


# ---------------------------------------------------------------------------
# 2. Eigenvalue Classification
# ---------------------------------------------------------------------------
def classify_equilibrium(A):
    """Classify the equilibrium at the origin based on eigenvalues.

    The trace (tr) and determinant (det) of A determine the type:
      det < 0             => saddle
      det > 0, tr < 0     => stable (node or spiral depending on tr^2 - 4*det)
      det > 0, tr > 0     => unstable
      det > 0, tr = 0     => center
    """
    eigvals = np.linalg.eigvals(A)
    tr = np.trace(A)
    det = np.linalg.det(A)
    disc = tr ** 2 - 4 * det

    if det < 0:
        kind = "Saddle point"
    elif abs(tr) < 1e-10:
        kind = "Center"
    elif disc > 0:
        stability = "Stable" if tr < 0 else "Unstable"
        kind = f"{stability} node"
    else:
        stability = "Stable" if tr < 0 else "Unstable"
        kind = f"{stability} spiral"

    return kind, eigvals, tr, det


# ---------------------------------------------------------------------------
# 3. Lotka-Volterra Predator-Prey Model
# ---------------------------------------------------------------------------
def lotka_volterra(alpha=1.0, beta=0.5, gamma=0.5, delta=0.2,
                   x0=4.0, y0=2.0, t_end=50.0):
    """Simulate the Lotka-Volterra predator-prey model.

    dx/dt = alpha*x - beta*x*y    (prey: grows, eaten by predator)
    dy/dt = delta*x*y - gamma*y   (predator: grows from prey, dies naturally)

    This system has a non-trivial equilibrium at:
      x* = gamma/delta,  y* = alpha/beta
    The solutions are periodic orbits around this equilibrium.
    """
    def rhs(t, state):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]

    t_eval = np.linspace(0, t_end, 2000)
    sol = solve_ivp(rhs, [0, t_end], [x0, y0], t_eval=t_eval,
                    method="RK45", rtol=1e-10)

    # Equilibrium
    x_eq = gamma / delta
    y_eq = alpha / beta

    return sol.t, sol.y[0], sol.y[1], x_eq, y_eq


def plot_lotka_volterra(t, prey, pred, x_eq, y_eq):
    """Plot time series and phase portrait for Lotka-Volterra."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Time series
    axes[0].plot(t, prey, "b-", lw=1.5, label="Prey (x)")
    axes[0].plot(t, pred, "r-", lw=1.5, label="Predator (y)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Population")
    axes[0].set_title("Lotka-Volterra: Time Series")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase portrait
    axes[1].plot(prey, pred, "b-", lw=1.5)
    axes[1].plot(prey[0], pred[0], "go", ms=10, label="Start")
    axes[1].plot(x_eq, y_eq, "r*", ms=15, label=f"Equilibrium ({x_eq:.1f}, {y_eq:.1f})")
    axes[1].set_xlabel("Prey (x)")
    axes[1].set_ylabel("Predator (y)")
    axes[1].set_title("Lotka-Volterra: Phase Portrait")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("11_lotka_volterra.png", dpi=100)
    plt.close()
    print("[Saved] 11_lotka_volterra.png")


# ---------------------------------------------------------------------------
# 4. Stability Analysis with Jacobian
# ---------------------------------------------------------------------------
def stability_analysis_lotka_volterra(alpha, beta, gamma, delta):
    """Linearize the Lotka-Volterra system around its equilibria and classify.

    The Jacobian at (x*, y*) determines local stability:
      J = [[alpha - beta*y,  -beta*x],
           [delta*y,          delta*x - gamma]]

    At the non-trivial equilibrium (gamma/delta, alpha/beta):
      J = [[0,                   -beta*gamma/delta],
           [delta*alpha/beta,     0                ]]

    The eigenvalues are purely imaginary => center (neutrally stable).
    """
    x_eq = gamma / delta
    y_eq = alpha / beta

    J = np.array([
        [alpha - beta * y_eq, -beta * x_eq],
        [delta * y_eq, delta * x_eq - gamma]
    ])

    kind, eigvals, tr, det_ = classify_equilibrium(J)

    print(f"\nStability Analysis at ({x_eq:.2f}, {y_eq:.2f}):")
    print(f"  Jacobian: {J.tolist()}")
    print(f"  Eigenvalues: {eigvals}")
    print(f"  Trace: {tr:.4f}, Determinant: {det_:.4f}")
    print(f"  Classification: {kind}")

    return J, kind


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Systems of ODEs — Phase Portraits and Stability")
    print("=" * 60)

    # --- Demo 1: Linear phase portraits for different matrix types ---
    print("\nDemo 1: Phase portraits for 2D linear systems")

    systems = [
        (np.array([[-1, 0], [0, -2]]), "Stable Node", "11_stable_node.png"),
        (np.array([[1, 0], [0, -1]]), "Saddle Point", "11_saddle.png"),
        (np.array([[-0.5, 2], [-2, -0.5]]), "Stable Spiral", "11_stable_spiral.png"),
        (np.array([[0, 1], [-1, 0]]), "Center", "11_center.png"),
    ]

    for A, title, fname in systems:
        kind, eigvals, tr, det_ = classify_equilibrium(A)
        print(f"  {title}: {kind} (eigenvalues: {eigvals})")
        plot_phase_portrait(A, title=title, filename=fname)

    # --- Demo 2: Eigenvalue classification summary ---
    print("\nDemo 2: Classification summary")
    print(f"  {'Type':<20s} | {'Trace':>8s} | {'Det':>8s} | {'Eigenvalues'}")
    print("-" * 70)
    for A, title, _ in systems:
        kind, eigvals, tr, det_ = classify_equilibrium(A)
        print(f"  {kind:<20s} | {tr:8.2f} | {det_:8.2f} | {eigvals}")

    # --- Demo 3: Lotka-Volterra ---
    print("\nDemo 3: Lotka-Volterra predator-prey")
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 0.2
    t, prey, pred, x_eq, y_eq = lotka_volterra(alpha, beta, gamma, delta)
    print(f"  Equilibrium: prey = {x_eq:.1f}, predator = {y_eq:.1f}")
    plot_lotka_volterra(t, prey, pred, x_eq, y_eq)

    # --- Demo 4: Stability analysis ---
    print("\nDemo 4: Stability analysis")
    stability_analysis_lotka_volterra(alpha, beta, gamma, delta)

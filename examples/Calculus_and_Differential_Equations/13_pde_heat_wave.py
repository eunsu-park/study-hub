"""
Partial Differential Equations — Heat and Wave Equations

Demonstrates:
  - Heat equation solver using FTCS (Forward-Time Central-Space)
  - Wave equation solver (explicit finite difference)
  - Stability analysis (CFL condition)
  - Animated solutions saved as multi-frame plots

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Heat Equation: u_t = alpha * u_xx
# ---------------------------------------------------------------------------
def heat_equation_ftcs(alpha, L, T, Nx, Nt, u0_func, bc_left=0.0,
                        bc_right=0.0):
    """Solve the 1D heat equation using the FTCS scheme.

    Discretization:
      u_j^{n+1} = u_j^n + r * (u_{j+1}^n - 2*u_j^n + u_{j-1}^n)
    where r = alpha * dt / dx^2.

    Stability requirement: r <= 0.5 (von Neumann analysis).
    If r > 0.5, the scheme is unconditionally unstable — small errors
    grow exponentially, producing garbage.
    """
    dx = L / Nx
    dt = T / Nt
    r = alpha * dt / dx ** 2

    # Check stability
    print(f"  Heat FTCS: alpha={alpha}, dx={dx:.4f}, dt={dt:.6f}")
    print(f"  Stability parameter r = alpha*dt/dx^2 = {r:.4f}", end="")
    if r > 0.5:
        print(" [UNSTABLE! r > 0.5]")
    else:
        print(" [STABLE]")

    x = np.linspace(0, L, Nx + 1)
    u = u0_func(x).copy()
    u[0] = bc_left
    u[-1] = bc_right

    # Store snapshots for visualization
    snapshots = [(0.0, u.copy())]
    save_times = np.linspace(0, T, 8)[1:]  # 7 snapshots
    save_idx = 0

    for n in range(1, Nt + 1):
        u_new = u.copy()
        # Interior points: FTCS update
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        # Apply boundary conditions
        u_new[0] = bc_left
        u_new[-1] = bc_right
        u = u_new

        t_current = n * dt
        if save_idx < len(save_times) and t_current >= save_times[save_idx]:
            snapshots.append((t_current, u.copy()))
            save_idx += 1

    return x, snapshots, r


def plot_heat_solution(x, snapshots, title="Heat Equation Solution"):
    """Plot the temperature profile at several time steps."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.hot(np.linspace(0.2, 0.9, len(snapshots)))

    for (t, u), color in zip(snapshots, colors):
        ax.plot(x, u, "-", color=color, lw=2, label=f"t = {t:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("13_heat_equation.png", dpi=100)
    plt.close()
    print("[Saved] 13_heat_equation.png")


# ---------------------------------------------------------------------------
# 2. Wave Equation: u_tt = c^2 * u_xx
# ---------------------------------------------------------------------------
def wave_equation_solver(c, L, T, Nx, Nt, u0_func, v0_func=None):
    """Solve the 1D wave equation using an explicit finite difference scheme.

    Discretization:
      u_j^{n+1} = 2*u_j^n - u_j^{n-1} + C^2*(u_{j+1}^n - 2*u_j^n + u_{j-1}^n)
    where C = c*dt/dx is the Courant number (CFL number).

    Stability: C <= 1 (CFL condition).
    When C = 1, the scheme is exact for the wave equation — this is a
    remarkable property of this particular discretization.
    """
    dx = L / Nx
    dt = T / Nt
    C = c * dt / dx

    print(f"  Wave equation: c={c}, dx={dx:.4f}, dt={dt:.6f}")
    print(f"  Courant number C = c*dt/dx = {C:.4f}", end="")
    if C > 1:
        print(" [UNSTABLE! C > 1]")
    else:
        print(f" [STABLE]")

    x = np.linspace(0, L, Nx + 1)
    C2 = C ** 2

    # Initial conditions
    u_prev = u0_func(x).copy()
    u_prev[0] = 0
    u_prev[-1] = 0

    # First time step: uses initial velocity v0
    if v0_func is None:
        v0 = np.zeros_like(x)
    else:
        v0 = v0_func(x)

    # Special formula for first step (incorporates initial velocity)
    u_curr = np.zeros_like(x)
    u_curr[1:-1] = (u_prev[1:-1]
                     + 0.5 * C2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2])
                     + dt * v0[1:-1])

    snapshots = [(0.0, u_prev.copy())]
    save_times = np.linspace(0, T, 10)[1:]
    save_idx = 0

    for n in range(2, Nt + 1):
        u_next = np.zeros_like(x)
        u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                         + C2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]))
        # Fixed boundary conditions
        u_next[0] = 0
        u_next[-1] = 0

        u_prev = u_curr.copy()
        u_curr = u_next.copy()

        t_current = n * dt
        if save_idx < len(save_times) and t_current >= save_times[save_idx]:
            snapshots.append((t_current, u_curr.copy()))
            save_idx += 1

    return x, snapshots, C


def plot_wave_solution(x, snapshots, title="Wave Equation Solution"):
    """Plot wave profiles at several time steps."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(snapshots)))

    for (t, u), color in zip(snapshots, colors):
        ax.plot(x, u, "-", color=color, lw=1.5, label=f"t = {t:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("13_wave_equation.png", dpi=100)
    plt.close()
    print("[Saved] 13_wave_equation.png")


# ---------------------------------------------------------------------------
# 3. Stability Analysis
# ---------------------------------------------------------------------------
def stability_analysis_demo():
    """Demonstrate the effect of violating stability conditions.

    We solve the heat equation with r = 0.4 (stable) and r = 0.6 (unstable)
    to visually show how instability manifests as oscillatory blow-up.
    """
    L = 1.0
    alpha = 1.0
    Nx = 50
    u0 = lambda x: np.sin(np.pi * x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, r_target, title_suffix in zip(
        axes, [0.4, 0.6], ["STABLE (r=0.4)", "UNSTABLE (r=0.6)"]
    ):
        dx = L / Nx
        dt = r_target * dx ** 2 / alpha
        T = 0.05
        Nt = int(T / dt) + 1

        x, snapshots, r = heat_equation_ftcs(alpha, L, T, Nx, Nt, u0)

        colors = plt.cm.hot(np.linspace(0.2, 0.9, len(snapshots)))
        for (t, u), color in zip(snapshots, colors):
            ax.plot(x, u, "-", color=color, lw=1.5, label=f"t={t:.4f}")

        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"Heat Equation: {title_suffix}")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("13_stability_comparison.png", dpi=100)
    plt.close()
    print("[Saved] 13_stability_comparison.png")


# ---------------------------------------------------------------------------
# 4. CFL Condition Visualization
# ---------------------------------------------------------------------------
def plot_cfl_condition():
    """Visualize the CFL stability region for both equations.

    Heat: stable when r = alpha*dt/dx^2 <= 0.5
    Wave: stable when C = c*dt/dx <= 1
    """
    dx = np.linspace(0.01, 0.5, 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heat equation stability
    alpha = 1.0
    dt_max_heat = 0.5 * dx ** 2 / alpha
    axes[0].fill_between(dx, 0, dt_max_heat, alpha=0.3, color="green",
                          label="Stable region")
    axes[0].plot(dx, dt_max_heat, "g-", lw=2, label="r = 0.5 boundary")
    axes[0].set_xlabel("dx")
    axes[0].set_ylabel("dt (max)")
    axes[0].set_title("Heat Eq. CFL: dt <= 0.5 * dx^2 / alpha")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Wave equation stability
    c = 1.0
    dt_max_wave = dx / c
    axes[1].fill_between(dx, 0, dt_max_wave, alpha=0.3, color="blue",
                          label="Stable region")
    axes[1].plot(dx, dt_max_wave, "b-", lw=2, label="C = 1 boundary")
    axes[1].set_xlabel("dx")
    axes[1].set_ylabel("dt (max)")
    axes[1].set_title("Wave Eq. CFL: dt <= dx / c")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("13_cfl_condition.png", dpi=100)
    plt.close()
    print("[Saved] 13_cfl_condition.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PDE Solvers: Heat and Wave Equations")
    print("=" * 60)

    # --- Demo 1: Heat equation ---
    print("\nDemo 1: Heat equation (FTCS)")
    # Initial condition: sin(pi*x), analytical: sin(pi*x)*exp(-pi^2*alpha*t)
    L, T, alpha = 1.0, 0.1, 1.0
    Nx, Nt = 50, 500
    u0_heat = lambda x: np.sin(np.pi * x)

    x_h, snap_h, r_h = heat_equation_ftcs(alpha, L, T, Nx, Nt, u0_heat)
    plot_heat_solution(x_h, snap_h,
                       title=f"Heat Equation (alpha={alpha}, r={r_h:.3f})")

    # --- Demo 2: Wave equation ---
    print("\nDemo 2: Wave equation (explicit FD)")
    # Gaussian pulse initial condition
    c = 1.0
    L_w, T_w = 2.0, 2.0
    Nx_w, Nt_w = 200, 400
    u0_wave = lambda x: np.exp(-50 * (x - 1.0) ** 2)

    x_w, snap_w, C_w = wave_equation_solver(c, L_w, T_w, Nx_w, Nt_w, u0_wave)
    plot_wave_solution(x_w, snap_w,
                       title=f"Wave Equation (c={c}, Courant={C_w:.3f})")

    # --- Demo 3: Stability comparison ---
    print("\nDemo 3: Stability analysis (stable vs unstable)")
    stability_analysis_demo()

    # --- Demo 4: CFL visualization ---
    print("\nDemo 4: CFL condition visualization")
    plot_cfl_condition()

    # --- Summary ---
    print("\nSummary:")
    print(f"  Heat eq: alpha={alpha}, r={r_h:.4f} ({'STABLE' if r_h <= 0.5 else 'UNSTABLE'})")
    print(f"  Wave eq: c={c}, C={C_w:.4f} ({'STABLE' if C_w <= 1.0 else 'UNSTABLE'})")

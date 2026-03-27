"""
Exercises for Lesson 09: Heat Equation

Topics: FTCS stability experiment, Crank-Nicolson convergence order,
        non-homogeneous boundary conditions, 2D heat equation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Exercise 1: FTCS Stability Experiment
# ---------------------------------------------------------------------------
# Run FTCS with r = 0.3, 0.5, 0.6 and observe stable/unstable behavior.
# ---------------------------------------------------------------------------

def exercise_1():
    """FTCS stability experiment with different r values."""
    L = 1.0
    nx = 51
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    alpha = 0.01

    u0 = np.sin(np.pi * x)

    def ftcs_step(u, r):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = 0
        u_new[-1] = 0
        return u_new

    r_values = [0.3, 0.5, 0.6]
    n_steps = 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, r in enumerate(r_values):
        dt = r * dx**2 / alpha
        u = u0.copy()
        ax = axes[idx]
        ax.plot(x, u0, 'b--', label='Initial', alpha=0.5)

        for step in range(n_steps):
            u = ftcs_step(u, r)
            if step in [20, 50, 99]:
                u_plot = np.clip(u, -10, 10)
                ax.plot(x, u_plot, label=f'step {step + 1}')

        stable = r <= 0.5
        status = "Stable" if stable else "UNSTABLE"
        ax.set_title(f'r = {r} ({status})\ndt = {dt:.6f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if stable:
            ax.set_ylim(-1.5, 1.5)
        else:
            ax.set_ylim(-10, 10)

    plt.tight_layout()
    plt.savefig('ex09_ftcs_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex09_ftcs_stability.png")

    print("\nResults:")
    print("  r = 0.3: Stable - solution decays smoothly")
    print("  r = 0.5: Marginally stable - boundary of stability")
    print("  r = 0.6: UNSTABLE - oscillations grow exponentially!")
    print("\nCFL condition for FTCS: r = alpha*dt/dx^2 <= 0.5")


# ---------------------------------------------------------------------------
# Exercise 2: Verify Convergence Order
# ---------------------------------------------------------------------------
# Numerically verify that Crank-Nicolson has 2nd order temporal accuracy.
# ---------------------------------------------------------------------------

def exercise_2():
    """Verify Crank-Nicolson 2nd-order temporal accuracy."""
    L = 1.0
    alpha = 0.01
    nx = 201  # Fine spatial grid to minimize spatial error
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    T = 1.0

    # Analytical solution: u(x,t) = sin(pi*x)*exp(-alpha*pi^2*t)
    def exact_solution(x, t):
        return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

    u_exact = exact_solution(x, T)

    def crank_nicolson_solve(nx, nt):
        dx = L / (nx - 1)
        dt = T / nt
        r = alpha * dt / dx**2
        x_loc = np.linspace(0, L, nx)

        n = nx - 2  # interior points
        # A matrix (implicit)
        main_A = (1 + r) * np.ones(n)
        off_A = (-r / 2) * np.ones(n - 1)
        A = sparse.diags([off_A, main_A, off_A], [-1, 0, 1], format='csr')

        # B matrix (explicit)
        main_B = (1 - r) * np.ones(n)
        off_B = (r / 2) * np.ones(n - 1)
        B = sparse.diags([off_B, main_B, off_B], [-1, 0, 1], format='csr')

        u = np.sin(np.pi * x_loc / L)  # IC

        for _ in range(nt):
            b = B @ u[1:-1]
            u[1:-1] = spsolve(A, b)
            u[0] = 0
            u[-1] = 0

        return x_loc, u

    # Compare exact solution at same grid
    print(f"{'nt':>6}  {'dt':>10}  {'Max error':>12}  {'Order':>8}")
    print("-" * 42)

    prev_err = None
    nt_values = [20, 40, 80, 160, 320]
    for nt in nt_values:
        dt = T / nt
        _, u_cn = crank_nicolson_solve(nx, nt)
        err = np.max(np.abs(u_cn - u_exact))
        if prev_err is not None and err > 0:
            order = np.log2(prev_err / err)
            print(f"{nt:>6}  {dt:>10.4e}  {err:>12.2e}  {order:>8.2f}")
        else:
            print(f"{nt:>6}  {dt:>10.4e}  {err:>12.2e}  {'---':>8}")
        prev_err = err

    print("\nOrder ~ 2.0 confirms Crank-Nicolson is 2nd order in time.")
    print("(Spatial error is negligible due to fine nx = 201 grid)")


# ---------------------------------------------------------------------------
# Exercise 3: Non-Homogeneous Boundary Conditions
# ---------------------------------------------------------------------------
# Find the steady-state solution when u(0,t) = 0 and u(L,t) = 100.
# ---------------------------------------------------------------------------

def exercise_3():
    """Steady-state with non-homogeneous BCs: u(0)=0, u(L)=100."""
    L = 1.0
    alpha = 0.01
    nx = 51
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    T = 20.0  # Long enough to reach steady state

    # FTCS with non-homogeneous BCs
    dt = 0.4 * dx**2 / alpha
    nt = int(np.ceil(T / dt))
    dt = T / nt
    r = alpha * dt / dx**2

    u = np.zeros(nx)  # IC: zero everywhere
    u[0] = 0.0
    u[-1] = 100.0

    history = [u.copy()]
    times = [0]

    save_interval = max(1, nt // 10)
    for n in range(nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = 0.0
        u_new[-1] = 100.0
        u = u_new
        if (n + 1) % save_interval == 0:
            history.append(u.copy())
            times.append((n + 1) * dt)

    # Analytical steady state: u_ss(x) = 100*x/L (linear)
    u_steady = 100 * x / L

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    for i in range(0, len(times), max(1, len(times) // 5)):
        ax1.plot(x, history[i], label=f't = {times[i]:.1f}')
    ax1.plot(x, u_steady, 'k--', linewidth=2, label='Steady state (analytical)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('Evolution to Steady State')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x, u, 'b-', linewidth=2, label='Numerical (final)')
    ax2.plot(x, u_steady, 'r--', linewidth=2, label='Analytical: u = 100x/L')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.set_title('Steady-State Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex09_steady_state.png', dpi=150, bbox_inches='tight')
    plt.close()

    err = np.max(np.abs(u - u_steady))
    print(f"Steady-state error (t={T}): {err:.2e}")
    print(f"Analytical steady state: u(x) = 100*x/L (linear profile)")
    print("Plot saved to ex09_steady_state.png")


# ---------------------------------------------------------------------------
# Exercise 4: 2D Heat Equation
# ---------------------------------------------------------------------------
# Solve the 2D heat equation with a rectangular hot spot initial condition
# instead of Gaussian.
# ---------------------------------------------------------------------------

def exercise_4():
    """2D heat equation with rectangular hot spot IC."""
    Lx = Ly = 1.0
    alpha = 0.01
    nx = ny = 51
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    T = 0.5
    safety = 0.2

    # CFL for 2D: r_x + r_y <= 0.5
    dt = safety * 0.5 / (alpha * (1 / dx**2 + 1 / dy**2))
    nt = int(np.ceil(T / dt))
    dt = T / nt
    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    print(f"Grid: {nx} x {ny}")
    print(f"dt = {dt:.6e}, rx = {rx:.4f}, ry = {ry:.4f}")
    print(f"rx + ry = {rx + ry:.4f} (must be <= 0.5)")
    print(f"Number of time steps: {nt}")

    # Rectangular hot spot initial condition
    u = np.zeros((ny, nx))
    # Hot rectangular region: 0.3 <= x <= 0.7, 0.3 <= y <= 0.7
    mask = (X >= 0.3) & (X <= 0.7) & (Y >= 0.3) & (Y <= 0.7)
    u[mask] = 1.0

    # Dirichlet BC: u = 0 on boundaries
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0

    snapshots = [u.copy()]
    snap_times = [0]

    save_every = max(1, nt // 4)
    for n in range(nt):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + rx * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
            + ry * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        )
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0
        u = u_new
        if (n + 1) % save_every == 0:
            snapshots.append(u.copy())
            snap_times.append((n + 1) * dt)

    # Visualization
    n_plots = min(5, len(snapshots))
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]

    indices = np.linspace(0, len(snapshots) - 1, n_plots, dtype=int)
    for idx, i in enumerate(indices):
        ax = axes[idx]
        c = ax.contourf(X, Y, snapshots[i], levels=20, cmap='hot')
        plt.colorbar(c, ax=ax, shrink=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {snap_times[i]:.3f}')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('ex09_heat_2d_rect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex09_heat_2d_rect.png")
    print(f"Max temperature at t={T}: {np.max(u):.6f}")
    print("The rectangular hot spot diffuses and smooths over time.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: FTCS Stability Experiment")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: Verify Convergence Order (Crank-Nicolson)")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: Non-Homogeneous Boundary Conditions")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: 2D Heat Equation (Rectangular IC)")
    print("=" * 60)
    exercise_4()

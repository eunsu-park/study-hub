"""
Exercises for Lesson 10: Wave Equation

Topics: Courant number stability, standing wave modes, absorbing boundary
        conditions, circular membrane normal modes.
"""

import numpy as np
from scipy.special import jn_zeros, jn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Exercise 1: Courant Number Experiment
# ---------------------------------------------------------------------------
# Verify numerical solution stability for C = 0.5, 0.8, 1.0, 1.1.
# Use the CTCS scheme for the 1D wave equation.
# ---------------------------------------------------------------------------

def exercise_1():
    """Courant number experiment for 1D wave equation."""
    L = 1.0
    c = 1.0
    T = 2.0
    nx = 101

    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # Initial condition: sin(pi*x/L), zero velocity
    u0 = np.sin(np.pi * x / L)

    # Analytical solution: u(x,t) = sin(pi*x/L)*cos(pi*c*t/L)
    def exact(x, t):
        return np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)

    courant_values = [0.5, 0.8, 1.0, 1.1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, C in enumerate(courant_values):
        dt = C * dx / c
        nt = int(np.ceil(T / dt))
        dt = T / nt

        # Initialize: u^0 and u^1
        u_prev = u0.copy()
        # First step using Taylor expansion: u^1 = u^0 + dt*v0 + 0.5*dt^2*c^2*u_xx
        C_actual = c * dt / dx
        u_curr = u0.copy()
        u_curr[1:-1] = (u_prev[1:-1]
                        + 0.5 * C_actual**2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]))
        u_curr[0] = 0
        u_curr[-1] = 0

        # Time stepping (CTCS)
        for n in range(1, nt):
            u_next = np.zeros(nx)
            u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                            + C_actual**2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]))
            u_next[0] = 0
            u_next[-1] = 0
            u_prev = u_curr.copy()
            u_curr = u_next.copy()

        u_exact = exact(x, T)
        error = np.max(np.abs(u_curr - u_exact))

        ax = axes[idx]
        ax.plot(x, u_exact, 'b-', linewidth=2, label='Exact')
        u_plot = np.clip(u_curr, -5, 5)
        ax.plot(x, u_plot, 'r--', linewidth=1.5, label='Numerical')
        stable = C <= 1.0
        status = "Stable" if stable else "UNSTABLE"
        ax.set_title(f'C = {C} ({status})\nerror = {error:.2e}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if stable:
            ax.set_ylim(-1.5, 1.5)
        else:
            ax.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig('ex10_courant_experiment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex10_courant_experiment.png")
    print("\nC <= 1.0: Stable. C = 1.0 is actually exact for this scheme.")
    print("C > 1.0: UNSTABLE. Solution blows up.")


# ---------------------------------------------------------------------------
# Exercise 2: Standing Wave Mode
# ---------------------------------------------------------------------------
# Starting from u(x,0) = sin(2*pi*x/L), verify the second standing
# wave mode: u(x,t) = sin(2*pi*x/L)*cos(2*pi*c*t/L).
# ---------------------------------------------------------------------------

def exercise_2():
    """Verify second standing wave mode."""
    L = 1.0
    c = 1.0
    nx = 101
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # Second mode: n=2
    n_mode = 2
    u0 = np.sin(n_mode * np.pi * x / L)

    def exact(x, t):
        return np.sin(n_mode * np.pi * x / L) * np.cos(n_mode * np.pi * c * t / L)

    # Use C = 1.0 for exact CTCS solution
    C = 0.9
    dt = C * dx / c
    T = 2.0  # One full period for n=2 is L/(n*c) * 2 = 1.0
    nt = int(np.ceil(T / dt))
    dt = T / nt
    C_actual = c * dt / dx

    # Initialize
    u_prev = u0.copy()
    u_curr = u0.copy()
    u_curr[1:-1] = (u_prev[1:-1]
                    + 0.5 * C_actual**2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]))
    u_curr[0] = 0
    u_curr[-1] = 0

    # Store snapshots
    snapshots = [(0, u0.copy())]
    snap_times = [0, 0.25, 0.5, 0.75, 1.0]
    snap_idx = 1

    for n in range(1, nt):
        u_next = np.zeros(nx)
        u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                        + C_actual**2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]))
        u_next[0] = 0
        u_next[-1] = 0

        current_t = (n + 1) * dt
        if snap_idx < len(snap_times) and current_t >= snap_times[snap_idx] - dt / 2:
            snapshots.append((current_t, u_next.copy()))
            snap_idx += 1

        u_prev = u_curr.copy()
        u_curr = u_next.copy()

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 3.5))
    for idx, (t, u) in enumerate(snapshots):
        ax = axes[idx]
        u_ex = exact(x, t)
        ax.plot(x, u_ex, 'b-', linewidth=2, label='Exact')
        ax.plot(x, u, 'r--', linewidth=1.5, label='Numerical')
        # Show node positions (n_mode + 1 nodes at x = k*L/n_mode)
        for k in range(n_mode + 1):
            ax.axvline(x=k * L / n_mode, color='green', linestyle=':', alpha=0.5)
        ax.set_title(f't = {t:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_ylim(-1.5, 1.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Second Standing Wave Mode (n={n_mode})', fontsize=12)
    plt.tight_layout()
    plt.savefig('ex10_standing_wave.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex10_standing_wave.png")
    print(f"Mode n={n_mode}: frequency = {n_mode}*c/(2L) = {n_mode * c / (2 * L):.2f} Hz")
    print(f"Period = {2 * L / (n_mode * c):.2f} s")
    print(f"Node at x = L/{n_mode} = {L / n_mode:.2f}")


# ---------------------------------------------------------------------------
# Exercise 3: Improved Absorbing Boundary
# ---------------------------------------------------------------------------
# Implement 2nd order Sommerfeld absorbing boundary condition and compare
# with 1st order.
# ---------------------------------------------------------------------------

def exercise_3():
    """Compare 1st and 2nd order absorbing boundary conditions."""
    L = 2.0
    c = 1.0
    nx = 201
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    C = 0.9
    dt = C * dx / c
    T = 3.0
    nt = int(np.ceil(T / dt))
    dt = T / nt
    C_actual = c * dt / dx

    # Gaussian pulse initial condition, moving right
    x0 = 0.5
    sigma = 0.1
    u0 = np.exp(-(x - x0)**2 / (2 * sigma**2))

    def solve_wave_abc(abc_order):
        """Solve 1D wave equation with absorbing BCs."""
        u_prev = u0.copy()
        u_curr = u0.copy()
        # First step (zero initial velocity)
        u_curr[1:-1] = (u_prev[1:-1]
                        + 0.5 * C_actual**2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]))

        # Store u for 2nd order ABC
        u_pprev = None

        for n in range(1, nt):
            u_next = np.zeros(nx)
            u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                            + C_actual**2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]))

            if abc_order == 1:
                # 1st order Sommerfeld at right: du/dt + c*du/dx = 0
                # u_N^{n+1} = u_N^n + C*(u_{N-1}^n - u_N^n)
                #            = u_{N-1}^n  when C=1
                u_next[-1] = u_curr[-1] + C_actual * (u_curr[-2] - u_curr[-1])
                u_next[0] = u_curr[0] - C_actual * (u_curr[0] - u_curr[1])
            elif abc_order == 2:
                # 2nd order ABC (Engquist-Majda)
                # Improved absorbing boundary using higher-order approximation
                if u_pprev is not None:
                    # Right boundary
                    u_next[-1] = (-u_pprev[-2]
                                  + C_actual * (u_next[-2] + u_pprev[-1])
                                  + (2 - 2 * C_actual) * u_curr[-1]
                                  + C_actual * (C_actual - 1) * (u_curr[-2] + u_curr[-1])
                                  ) / (1 + C_actual)
                    # Left boundary
                    u_next[0] = (-u_pprev[1]
                                 + C_actual * (u_next[1] + u_pprev[0])
                                 + (2 - 2 * C_actual) * u_curr[0]
                                 + C_actual * (C_actual - 1) * (u_curr[1] + u_curr[0])
                                 ) / (1 + C_actual)
                else:
                    u_next[-1] = u_curr[-1] + C_actual * (u_curr[-2] - u_curr[-1])
                    u_next[0] = u_curr[0] - C_actual * (u_curr[0] - u_curr[1])

            u_pprev = u_prev.copy()
            u_prev = u_curr.copy()
            u_curr = u_next.copy()

        return u_curr

    # Reference: fixed boundaries (strong reflection)
    u_prev = u0.copy()
    u_curr = u0.copy()
    u_curr[1:-1] = (u_prev[1:-1]
                    + 0.5 * C_actual**2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]))
    u_curr[0] = 0
    u_curr[-1] = 0
    for n in range(1, nt):
        u_next = np.zeros(nx)
        u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                        + C_actual**2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]))
        u_next[0] = 0
        u_next[-1] = 0
        u_prev = u_curr.copy()
        u_curr = u_next.copy()
    u_fixed = u_curr

    u_abc1 = solve_wave_abc(1)
    u_abc2 = solve_wave_abc(2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, u_sol, title in [(axes[0], u_fixed, 'Fixed BC (full reflection)'),
                              (axes[1], u_abc1, '1st order Sommerfeld ABC'),
                              (axes[2], u_abc2, '2nd order ABC')]:
        ax.plot(x, u_sol, 'b-', linewidth=1.5)
        ax.plot(x, u0, 'r--', alpha=0.3, label='Initial')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'{title}\nmax|u| = {np.max(np.abs(u_sol)):.4f}')
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex10_absorbing_bc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex10_absorbing_bc.png")
    print(f"\nResidual energy after t={T}:")
    print(f"  Fixed BC:    max|u| = {np.max(np.abs(u_fixed)):.4e}")
    print(f"  1st order:   max|u| = {np.max(np.abs(u_abc1)):.4e}")
    print(f"  2nd order:   max|u| = {np.max(np.abs(u_abc2)):.4e}")
    print("\n2nd order ABC absorbs outgoing waves more effectively.")


# ---------------------------------------------------------------------------
# Exercise 4: Circular Membrane Normal Mode
# ---------------------------------------------------------------------------
# Compare the first normal mode of a circular membrane (Bessel function)
# with numerical solution.
# ---------------------------------------------------------------------------

def exercise_4():
    """Circular membrane first normal mode (Bessel function)."""
    # The first normal mode of a circular membrane with fixed edge:
    #   u(r, theta, t) = J_0(alpha_{01} * r/R) * cos(alpha_{01} * c * t / R)
    # where alpha_{01} is the first zero of J_0.

    R = 1.0
    c = 1.0
    nr = 51
    ntheta = 1  # axisymmetric mode (m=0)

    # First zero of J_0
    alpha_01 = jn_zeros(0, 1)[0]  # ~ 2.4048
    print(f"First zero of J_0: alpha_01 = {alpha_01:.6f}")
    print(f"Frequency: omega = alpha_01 * c / R = {alpha_01 * c / R:.6f}")
    print(f"Period: T = 2*pi/omega = {2 * np.pi / (alpha_01 * c / R):.6f}")

    # Radial grid
    dr = R / (nr - 1)
    r = np.linspace(0, R, nr)

    # Analytical mode shape
    u_mode = jn(0, alpha_01 * r / R)

    # 1D wave equation in cylindrical coordinates (axisymmetric):
    # u_tt = c^2 * (u_rr + (1/r)*u_r)
    # CTCS scheme with special treatment at r=0

    T_sim = 2 * np.pi / (alpha_01 * c / R)  # one period
    C = 0.5
    dt = C * dr / c
    nt = int(np.ceil(T_sim / dt))
    dt = T_sim / nt

    # Initialize with mode shape
    u_prev = u_mode.copy()
    u_curr = u_mode.copy()

    # First step (zero initial velocity => stays at mode shape for first step)
    # u^1 = u^0 + 0.5*dt^2*c^2*(u_rr + (1/r)*u_r)
    for i in range(1, nr - 1):
        u_rr = (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]) / dr**2
        u_r = (u_prev[i + 1] - u_prev[i - 1]) / (2 * dr)
        u_curr[i] = u_prev[i] + 0.5 * dt**2 * c**2 * (u_rr + u_r / r[i])
    # At r=0: use L'Hopital, u_rr + (1/r)*u_r -> 2*u_rr at r=0
    u_curr[0] = u_prev[0] + 0.5 * dt**2 * c**2 * 2 * (u_prev[1] - u_prev[0]) / dr**2
    u_curr[-1] = 0  # Fixed boundary

    # Time stepping
    for n in range(1, nt):
        u_next = np.zeros(nr)
        for i in range(1, nr - 1):
            u_rr = (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]) / dr**2
            u_r = (u_curr[i + 1] - u_curr[i - 1]) / (2 * dr)
            u_next[i] = 2 * u_curr[i] - u_prev[i] + dt**2 * c**2 * (u_rr + u_r / r[i])
        # r=0
        u_next[0] = 2 * u_curr[0] - u_prev[0] + dt**2 * c**2 * 2 * (u_curr[1] - u_curr[0]) / dr**2
        u_next[-1] = 0
        u_prev = u_curr.copy()
        u_curr = u_next.copy()

    # After one period, solution should return to initial condition
    u_analytical_after_period = u_mode  # cos(omega*T) = cos(2*pi) = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(r, u_mode, 'b-', linewidth=2, label='J_0(alpha_01 * r/R)')
    ax1.plot(r, u_curr, 'ro', markersize=4, label='Numerical (after 1 period)')
    ax1.set_xlabel('r')
    ax1.set_ylabel('u(r)')
    ax1.set_title(f'Circular Membrane 1st Mode\n(after 1 period T={T_sim:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(r, np.abs(u_curr - u_analytical_after_period), 'g-', linewidth=2)
    ax2.set_xlabel('r')
    ax2.set_ylabel('|Error|')
    ax2.set_title(f'Error: max = {np.max(np.abs(u_curr - u_mode)):.2e}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex10_circular_membrane.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nMax error after one period: {np.max(np.abs(u_curr - u_mode)):.2e}")
    print("Plot saved to ex10_circular_membrane.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Courant Number Experiment")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: Standing Wave Mode")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: Improved Absorbing Boundary")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: Circular Membrane Normal Mode")
    print("=" * 60)
    exercise_4()

"""
Exercises for Lesson 12: Advection Equation
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Confirm FTCS Instability ===
# Problem: Run FTCS at various Courant numbers and confirm instability.

def exercise_1():
    """Confirm FTCS is unconditionally unstable for the advection equation."""
    L = 4.0
    c = 1.0
    nx = 101
    T = 0.5
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    courant_values = [0.2, 0.5, 0.8, 0.95]
    print("FTCS Advection: Testing various Courant numbers")
    print("-" * 55)

    for C_target in courant_values:
        dt = C_target * dx / abs(c)
        nt = int(np.ceil(T / dt))
        dt = T / nt
        C = c * dt / dx

        u = u0(x).copy()
        max_vals = [np.max(np.abs(u))]

        for n in range(nt):
            u_new = u.copy()
            # FTCS: central difference in space, forward in time
            u_new[1:-1] = u[1:-1] - (C / 2) * (u[2:] - u[:-2])
            # Periodic BCs
            u_new[0] = u[0] - (C / 2) * (u[1] - u[-2])
            u_new[-1] = u_new[0]
            u = u_new
            max_vals.append(np.max(np.abs(u)))

        # Check amplification factor: |G|^2 = 1 + C^2 sin^2(k dx) > 1
        amplification = max_vals[-1] / max_vals[0]
        status = "UNSTABLE" if amplification > 1.5 else "growing"
        print(f"  C = {C_target:.2f}: max|u| grew from {max_vals[0]:.4f} to "
              f"{max_vals[-1]:.4e} (ratio={amplification:.2e}) -> {status}")

    print("\nConclusion: FTCS is unconditionally unstable for advection.")
    print("  |G|^2 = 1 + C^2 sin^2(k dx) > 1 for ALL C > 0")


# === Exercise 2: Reverse Advection ===
# Problem: Modify the Upwind scheme for c < 0 and test.

def exercise_2():
    """Upwind scheme for negative advection velocity (c < 0)."""
    L = 4.0
    c = -1.0  # Leftward advection
    nx = 201
    T = 1.5
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    C_target = 0.8

    dt = C_target * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt
    C = c * dt / dx  # Negative for c < 0

    def u0(x):
        return np.exp(-(x - 3.0)**2 / 0.08)

    u = u0(x).copy()

    for n in range(nt):
        u_new = u.copy()
        # For c < 0: use forward difference (information comes from right)
        u_new[:-1] = u[:-1] - C * (u[1:] - u[:-1])
        # Right boundary: inflow
        u_new[-1] = u0(x[-1] - c * (n + 1) * dt)
        u = u_new

    # Analytical solution
    u_exact = u0(x - c * T)

    error = np.max(np.abs(u - u_exact))
    print(f"Reverse advection (c = {c}):")
    print(f"  Courant number C = {C:.4f}")
    print(f"  Max error at t = {T}: {error:.6f}")
    print(f"  Peak location: numerical = {x[np.argmax(u)]:.2f}, "
          f"exact = {x[np.argmax(u_exact)]:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, u0(x), 'k--', alpha=0.5, label='Initial')
    ax.plot(x, u, 'b-', linewidth=2, label='Numerical (Upwind)')
    ax.plot(x, u_exact, 'r--', linewidth=2, label='Exact')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'Reverse Advection (c = {c}), Max Error = {error:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex12_reverse_advection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex12_reverse_advection.png")


# === Exercise 3: Beam-Warming Scheme ===
# Problem: Implement second-order upwind (Beam-Warming) and compare with Lax-Wendroff.

def exercise_3():
    """Beam-Warming second-order upwind scheme vs Lax-Wendroff."""
    L = 4.0
    c = 1.0
    nx = 201
    T = 2.0
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    C_target = 0.8

    dt = C_target * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt
    C = c * dt / dx

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    # --- Beam-Warming (second-order upwind, c > 0) ---
    # u_i^{n+1} = u_i - C/2 (3u_i - 4u_{i-1} + u_{i-2})
    #           + C^2/2 (u_i - 2u_{i-1} + u_{i-2})
    u_bw = u0(x).copy()
    for n in range(nt):
        u_new = u_bw.copy()
        for i in range(2, nx):
            u_new[i] = (u_bw[i]
                        - (C / 2) * (3 * u_bw[i] - 4 * u_bw[i-1] + u_bw[i-2])
                        + (C**2 / 2) * (u_bw[i] - 2 * u_bw[i-1] + u_bw[i-2]))
        u_bw = u_new

    # --- Lax-Wendroff ---
    u_lw = u0(x).copy()
    C2 = C**2
    for n in range(nt):
        u_new = u_lw.copy()
        u_new[1:-1] = (u_lw[1:-1]
                       - (C / 2) * (u_lw[2:] - u_lw[:-2])
                       + (C2 / 2) * (u_lw[2:] - 2 * u_lw[1:-1] + u_lw[:-2]))
        u_lw = u_new

    u_exact = u0(x - c * T)

    err_bw = np.max(np.abs(u_bw - u_exact))
    err_lw = np.max(np.abs(u_lw - u_exact))

    print(f"Second-order scheme comparison (C = {C:.4f}, T = {T}):")
    print(f"  Beam-Warming max error: {err_bw:.6f}")
    print(f"  Lax-Wendroff max error: {err_lw:.6f}")
    print(f"  Both are second-order accurate (O(dt^2, dx^2))")
    print(f"  Beam-Warming: trailing oscillations (dispersive, behind wavefront)")
    print(f"  Lax-Wendroff: leading oscillations (dispersive, ahead of wavefront)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x, u0(x), 'k--', alpha=0.5, label='Initial')
    ax.plot(x, u_bw, 'b-', linewidth=2, label=f'Beam-Warming (err={err_bw:.4f})')
    ax.plot(x, u_exact, 'r--', linewidth=2, label='Exact')
    ax.set_title('Beam-Warming (2nd-order upwind)')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    ax = axes[1]
    ax.plot(x, u0(x), 'k--', alpha=0.5, label='Initial')
    ax.plot(x, u_lw, 'g-', linewidth=2, label=f'Lax-Wendroff (err={err_lw:.4f})')
    ax.plot(x, u_exact, 'r--', linewidth=2, label='Exact')
    ax.set_title('Lax-Wendroff')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    plt.suptitle('Beam-Warming vs Lax-Wendroff Comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig('ex12_beam_warming.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex12_beam_warming.png")


# === Exercise 4: 2D Advection ===
# Problem: Solve du/dt + c_x du/dx + c_y du/dy = 0.

def exercise_4():
    """2D advection equation using dimensional splitting with upwind."""
    nx = ny = 101
    Lx = Ly = 4.0
    cx, cy = 1.0, 0.5  # Advection velocities
    T = 2.0

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # CFL condition: cx dt/dx + cy dt/dy <= 1
    dt = 0.8 / (abs(cx) / dx + abs(cy) / dy)
    nt = int(np.ceil(T / dt))
    dt = T / nt
    Cx = cx * dt / dx
    Cy = cy * dt / dy

    print(f"2D Advection: cx={cx}, cy={cy}")
    print(f"  Cx = {Cx:.4f}, Cy = {Cy:.4f}, Cx+Cy = {Cx+Cy:.4f}")

    # Initial condition: 2D Gaussian
    def u0(X, Y):
        return np.exp(-((X - 1.0)**2 + (Y - 1.0)**2) / 0.1)

    u = u0(X, Y)

    # Dimensional splitting: Upwind in x, then upwind in y
    for n in range(nt):
        u_new = u.copy()
        # x-sweep (cx > 0: backward difference)
        if cx > 0:
            u_new[:, 1:] = u[:, 1:] - Cx * (u[:, 1:] - u[:, :-1])
        else:
            u_new[:, :-1] = u[:, :-1] - Cx * (u[:, 1:] - u[:, :-1])

        # y-sweep (cy > 0: backward difference)
        u_temp = u_new.copy()
        if cy > 0:
            u_new[1:, :] = u_temp[1:, :] - Cy * (u_temp[1:, :] - u_temp[:-1, :])
        else:
            u_new[:-1, :] = u_temp[:-1, :] - Cy * (u_temp[1:, :] - u_temp[:-1, :])

        u = u_new

    # Exact solution
    u_exact = u0(X - cx * T, Y - cy * T)

    error = np.max(np.abs(u - u_exact))
    print(f"  Max error at t = {T}: {error:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    im = ax.contourf(X, Y, u0(X, Y), levels=20, cmap='viridis')
    ax.set_title('Initial (t=0)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.contourf(X, Y, u, levels=20, cmap='viridis')
    ax.set_title(f'Numerical (t={T})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.contourf(X, Y, u_exact, levels=20, cmap='viridis')
    ax.set_title(f'Exact (t={T})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    plt.suptitle(f'2D Advection (cx={cx}, cy={cy}), Error={error:.4f}', fontsize=13)
    plt.tight_layout()
    plt.savefig('ex12_2d_advection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex12_2d_advection.png")


if __name__ == "__main__":
    print("=== Exercise 1: Confirm FTCS Instability ===")
    exercise_1()
    print("\n=== Exercise 2: Reverse Advection ===")
    exercise_2()
    print("\n=== Exercise 3: Beam-Warming Scheme ===")
    exercise_3()
    print("\n=== Exercise 4: 2D Advection ===")
    exercise_4()
    print("\nAll exercises completed!")

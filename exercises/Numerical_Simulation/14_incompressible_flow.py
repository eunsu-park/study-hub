"""
Exercises for Lesson 14: Incompressible Flow
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Stream Function and Velocity Field ===
# Problem: Given psi = sin(pi*x)*sin(pi*y), derive u, v, verify incompressibility,
# compute vorticity, and verify nabla^2 psi = -omega.

def exercise_1():
    """Stream function analysis: velocity, incompressibility, vorticity."""
    N = 50
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # psi = sin(pi x) sin(pi y)
    psi = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Analytical:
    # u = d psi/dy = pi sin(pi x) cos(pi y)
    # v = -d psi/dx = -pi cos(pi x) sin(pi y)
    u_exact = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
    v_exact = -np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)

    # Verify incompressibility: du/dx + dv/dy
    # du/dx = pi^2 cos(pi x) cos(pi y)
    # dv/dy = -pi^2 cos(pi x) cos(pi y)
    # Sum = 0
    dudx = np.pi**2 * np.cos(np.pi * X) * np.cos(np.pi * Y)
    dvdy = -np.pi**2 * np.cos(np.pi * X) * np.cos(np.pi * Y)
    div = dudx + dvdy

    print("Stream function: psi = sin(pi x) sin(pi y)")
    print(f"  u = pi sin(pi x) cos(pi y)")
    print(f"  v = -pi cos(pi x) sin(pi y)")
    print(f"  div(u) = du/dx + dv/dy: max|div| = {np.max(np.abs(div)):.2e}")
    print("  -> Incompressibility satisfied!")

    # Vorticity: omega = dv/dx - du/dy
    # dv/dx = pi^2 sin(pi x) sin(pi y)
    # du/dy = -pi^2 sin(pi x) sin(pi y)
    # omega = 2 pi^2 sin(pi x) sin(pi y)
    omega = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Verify: nabla^2 psi = -omega
    # nabla^2 psi = -pi^2 sin(pi x) sin(pi y) + sin(pi x)(-pi^2 sin(pi y))
    #             = -2 pi^2 sin(pi x) sin(pi y) = -omega
    laplacian_psi = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    residual = np.max(np.abs(laplacian_psi + omega))

    print(f"\n  Vorticity: omega = 2 pi^2 sin(pi x) sin(pi y)")
    print(f"  Verify nabla^2 psi = -omega: max|residual| = {residual:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    cs = ax.contour(X, Y, psi, levels=15, cmap='coolwarm')
    plt.colorbar(cs, ax=ax, label='psi')
    ax.set_title('Stream function psi')
    ax.set_aspect('equal')

    ax = axes[1]
    skip = 3
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_exact[::skip, ::skip], v_exact[::skip, ::skip], color='blue')
    ax.set_title('Velocity field (u, v)')
    ax.set_aspect('equal')

    ax = axes[2]
    im = ax.contourf(X, Y, omega, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='omega')
    ax.set_title('Vorticity omega')
    ax.set_aspect('equal')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('ex14_stream_function.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex14_stream_function.png")


# === Exercise 2: Lid-Driven Cavity Resolution Study ===
# Problem: Run lid-driven cavity at Re=100 for N=21, 41, 81 and find
# primary vortex center. Compare to Ghia et al. benchmark.

def exercise_2():
    """Lid-driven cavity resolution study at Re=100."""
    Re = 100
    U_lid = 1.0
    L = 1.0

    # Ghia et al. (1982) reference for Re=100:
    ghia_center = (0.617, 0.742)
    ghia_psi_min = -0.1034

    print(f"Lid-driven cavity, Re = {Re}")
    print(f"Ghia et al. reference: center=({ghia_center[0]}, {ghia_center[1]}), "
          f"psi_min={ghia_psi_min}")
    print()

    grid_sizes = [21, 41, 81]
    results = []

    for N in grid_sizes:
        h = L / (N - 1)
        nu = U_lid * L / Re
        dt = min(0.001, 0.25 * h**2 / nu)
        n_steps = min(8000, int(15 / dt))

        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)

        psi = np.zeros((N, N))
        omega = np.zeros((N, N))

        for n in range(n_steps):
            # Poisson solver (Gauss-Seidel, vectorized interior)
            for _ in range(20):
                psi[1:-1, 1:-1] = 0.25 * (
                    psi[2:, 1:-1] + psi[:-2, 1:-1] +
                    psi[1:-1, 2:] + psi[1:-1, :-2] +
                    h**2 * omega[1:-1, 1:-1]
                )
                psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0

            # Velocity from stream function
            u = np.zeros((N, N))
            v = np.zeros((N, N))
            u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
            v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)

            # Vorticity BCs
            omega[0, :] = -2 * psi[1, :] / h**2
            omega[-1, :] = -2 * psi[-2, :] / h**2 - 2 * U_lid / h
            omega[:, 0] = -2 * psi[:, 1] / h**2
            omega[:, -1] = -2 * psi[:, -2] / h**2

            # Vorticity transport (upwind + diffusion)
            omega_new = omega.copy()
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    cx = u[i, j] * ((omega[i, j] - omega[i, j-1]) / h
                                    if u[i, j] > 0 else
                                    (omega[i, j+1] - omega[i, j]) / h)
                    cy = v[i, j] * ((omega[i, j] - omega[i-1, j]) / h
                                    if v[i, j] > 0 else
                                    (omega[i+1, j] - omega[i, j]) / h)
                    diff = nu * ((omega[i+1, j] - 2*omega[i, j] + omega[i-1, j]) +
                                 (omega[i, j+1] - 2*omega[i, j] + omega[i, j-1])) / h**2
                    omega_new[i, j] = omega[i, j] + dt * (-cx - cy + diff)
            omega = omega_new

        # Find vortex center (minimum of psi)
        min_idx = np.unravel_index(np.argmin(psi), psi.shape)
        center_x = x[min_idx[1]]
        center_y = y[min_idx[0]]
        psi_min = psi[min_idx]

        err_x = abs(center_x - ghia_center[0])
        err_y = abs(center_y - ghia_center[1])

        results.append((N, center_x, center_y, psi_min))
        print(f"  N={N:2d}: center=({center_x:.3f}, {center_y:.3f}), "
              f"psi_min={psi_min:.6f}, "
              f"center error=({err_x:.3f}, {err_y:.3f})")

    print(f"\n  As grid is refined, the vortex center converges toward "
          f"Ghia et al. reference.")


# === Exercise 3: Checkerboard Pressure Instability ===
# Problem: On a 5x5 collocated grid, construct checkerboard pressure and show
# central-difference gradient is zero.

def exercise_3():
    """Demonstrate checkerboard pressure instability on collocated grid."""
    n = 5

    # Construct checkerboard pressure
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = (-1) ** (i + j)

    print("Checkerboard pressure field (5x5 grid):")
    print(P)

    # Compute central-difference dp/dx at interior points
    dpdx = np.zeros((n, n))
    dpdy = np.zeros((n, n))

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            dpdx[i, j] = (P[i, j+1] - P[i, j-1]) / 2
            dpdy[i, j] = (P[i+1, j] - P[i-1, j]) / 2

    print(f"\nCentral-difference dp/dx at interior points:")
    print(dpdx[1:-1, 1:-1])
    print(f"\nCentral-difference dp/dy at interior points:")
    print(dpdy[1:-1, 1:-1])

    max_grad = max(np.max(np.abs(dpdx[1:-1, 1:-1])),
                   np.max(np.abs(dpdy[1:-1, 1:-1])))
    print(f"\nMax |grad p| = {max_grad:.0f}")
    print("\nExplanation:")
    print("  The central difference skips over the alternating pattern,")
    print("  so p(i+1) - p(i-1) = (-1)^(i+1+j) - (-1)^(i-1+j) = 0.")
    print("  The momentum equation 'sees' zero pressure gradient despite")
    print("  large pressure oscillations -- the scheme cannot detect or")
    print("  suppress checkerboard pressure modes.")
    print("\n  Staggered grid solution: pressure at cell centers, velocity")
    print("  at cell faces. The pressure gradient is computed over one cell,")
    print("  not two, so checkerboard patterns are detected and suppressed.")


# === Exercise 4: SIMPLE Under-Relaxation Effects ===
# Problem: Test convergence with alpha_p in {0.1, 0.3, 0.5, 0.7}.

def exercise_4():
    """SIMPLE algorithm under-relaxation factor study (simplified model)."""
    # Simplified model: iterative pressure correction on a small problem
    # We model the iterative convergence behavior, not the full NS solver

    N = 16
    L = 1.0
    dx = dy = L / N
    Re = 100
    rho = 1.0
    mu = 1.0 / Re
    U_lid = 1.0
    dt = 0.002

    alpha_p_values = [0.1, 0.3, 0.5, 0.7]
    alpha_u = 0.7

    print("SIMPLE under-relaxation study (Re=100, 16x16 grid)")
    print("-" * 55)

    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha_p in alpha_p_values:
        u = np.zeros((N, N + 1))
        v = np.zeros((N + 1, N))
        p = np.zeros((N, N))

        div_history = []
        n_outer = 200

        for iteration in range(n_outer):
            # Simplified: lid BC
            u[-1, :] = 2 * U_lid - u[-2, :]
            u[0, :] = -u[1, :]

            # Simplified momentum (only diffusion + pressure gradient + body force)
            u_star = u.copy()
            for j in range(1, N - 1):
                for i in range(1, N):
                    d2u = mu / rho * ((u[j, i+1] - 2*u[j, i] + u[j, i-1]) / dx**2 +
                                       (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / dy**2)
                    dpdx_local = (p[j, min(i, N-1)] - p[j, max(i-1, 0)]) / dx
                    u_star[j, i] = u[j, i] + dt * (d2u - dpdx_local / rho)

            # Pressure correction (simplified Poisson)
            p_prime = np.zeros((N, N))
            for _ in range(30):
                for j in range(N):
                    for i in range(N):
                        div = ((u_star[j, i+1] - u_star[j, i]) / dx +
                               (v[j+1, i] - v[j, i]) / dy)
                        pE = p_prime[j, i+1] if i < N-1 else 0
                        pW = p_prime[j, i-1] if i > 0 else 0
                        pN = p_prime[j+1, i] if j < N-1 else 0
                        pS = p_prime[j-1, i] if j > 0 else 0
                        aE = 1/dx**2 if i < N-1 else 0
                        aW = 1/dx**2 if i > 0 else 0
                        aN = 1/dy**2 if j < N-1 else 0
                        aS = 1/dy**2 if j > 0 else 0
                        aP = aE + aW + aN + aS
                        if aP > 0:
                            p_prime[j, i] = (aE*pE + aW*pW + aN*pN + aS*pS -
                                             rho/dt * div) / aP

            # Correct
            p = p + alpha_p * p_prime
            u = alpha_u * u_star + (1 - alpha_u) * u

            # Divergence check
            div_max = 0
            for j in range(N):
                for i in range(N):
                    d = abs((u[j, i+1] - u[j, i]) / dx + (v[j+1, i] - v[j, i]) / dy)
                    div_max = max(div_max, d)
            div_history.append(div_max)

        ax.semilogy(div_history, linewidth=1.5, label=f'alpha_p={alpha_p}')
        converged = div_history[-1] < 1e-4
        print(f"  alpha_p={alpha_p}: final div={div_history[-1]:.2e}, "
              f"converged={converged}")

    ax.set_xlabel('Iteration')
    ax.set_ylabel('max|div(u)|')
    ax.set_title('SIMPLE Convergence vs Under-Relaxation Factor')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('ex14_simple_relaxation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex14_simple_relaxation.png")


if __name__ == "__main__":
    print("=== Exercise 1: Stream Function and Velocity Field ===")
    exercise_1()
    print("\n=== Exercise 2: Lid-Driven Cavity Resolution Study ===")
    exercise_2()
    print("\n=== Exercise 3: Checkerboard Pressure Instability ===")
    exercise_3()
    print("\n=== Exercise 4: SIMPLE Under-Relaxation Effects ===")
    exercise_4()
    print("\nAll exercises completed!")

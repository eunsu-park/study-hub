"""
Exercises for Lesson 11: Laplace and Poisson Equations
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, lil_matrix
from scipy.sparse.linalg import spsolve


# === Exercise 1: SOR Optimal omega Search ===
# Problem: Experimentally find optimal omega for various grid sizes
# and compare with theoretical values.

def exercise_1():
    """SOR optimal omega search: experimental vs theoretical."""

    def sor_solve(n, f, u_bc, omega, tol=1e-6, max_iter=10000):
        """Run SOR iteration and return iteration count."""
        h = 1.0 / (n - 1)
        u = u_bc.copy()
        h2 = h ** 2

        for k in range(max_iter):
            max_change = 0.0
            for j in range(1, n - 1):
                for i in range(1, n - 1):
                    u_old = u[j, i]
                    u_gs = 0.25 * (
                        u[j, i+1] + u[j, i-1] +
                        u[j+1, i] + u[j-1, i] -
                        h2 * f[j, i]
                    )
                    u[j, i] = (1 - omega) * u_old + omega * u_gs
                    max_change = max(max_change, abs(u[j, i] - u_old))

            if max_change < tol:
                return k + 1
        return max_iter

    grid_sizes = [11, 21, 31, 41]

    print("SOR Optimal omega Search")
    print("=" * 70)
    print(f"{'Grid n':<10}{'omega_opt (theory)':<22}{'omega_opt (expt)':<22}{'Iterations':<12}")
    print("-" * 70)

    for n in grid_sizes:
        h = 1.0 / (n - 1)
        omega_theory = 2.0 / (1 + np.sin(np.pi * h))

        f = np.zeros((n, n))
        u_bc = np.zeros((n, n))
        u_bc[-1, :] = 100  # Top boundary

        # Search over omega values near theoretical optimum
        omega_candidates = np.linspace(max(1.0, omega_theory - 0.15),
                                        min(1.99, omega_theory + 0.15), 15)
        best_omega = 1.0
        best_iters = float('inf')

        for omega in omega_candidates:
            iters = sor_solve(n, f, u_bc.copy(), omega, tol=1e-6, max_iter=5000)
            if iters < best_iters:
                best_iters = iters
                best_omega = omega

        print(f"{n:<10}{omega_theory:<22.6f}{best_omega:<22.6f}{best_iters:<12}")

    print("\nConclusion: Experimental optimal omega closely matches the")
    print("theoretical value omega_opt = 2 / (1 + sin(pi*h)).")


# === Exercise 2: L-shaped Domain ===
# Problem: Solve the Laplace equation on an L-shaped domain.

def exercise_2():
    """Solve Laplace equation on an L-shaped domain using Jacobi iteration."""

    n = 61  # Grid points per side
    h = 1.0 / (n - 1)

    # Create L-shaped domain mask: unit square minus upper-right quadrant
    # Domain: [0,1]x[0,1] minus [0.5,1]x[0.5,1]
    u = np.zeros((n, n))
    mask = np.ones((n, n), dtype=bool)  # True = interior/active point

    # Remove upper-right quadrant from domain
    mid = n // 2
    mask[mid:, mid:] = False

    # Set boundary conditions
    # Bottom edge (y=0): u = 100
    u[0, :] = 100
    # Left edge (x=0): u = 100
    u[:, 0] = 100
    # All other boundaries: u = 0 (re-entrant corner edges, top of lower half, right of lower half)

    # Jacobi iteration on L-shaped domain
    tol = 1e-6
    max_iter = 20000

    for iteration in range(max_iter):
        u_old = u.copy()

        for j in range(1, n - 1):
            for i in range(1, n - 1):
                if not mask[j, i]:
                    continue
                # Skip points adjacent to removed region (they use boundary = 0)
                u[j, i] = 0.25 * (u[j, i+1] + u[j, i-1] + u[j+1, i] + u[j-1, i])

        residual = np.max(np.abs(u - u_old))
        if residual < tol:
            print(f"L-shaped domain Jacobi converged in {iteration + 1} iterations")
            break
    else:
        print(f"Did not converge after {max_iter} iterations, residual = {residual:.2e}")

    # Mask the removed region for display
    u_display = np.ma.masked_where(~mask, u)

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.contourf(X, Y, u_display, levels=30, cmap='hot')
    plt.colorbar(c, ax=ax, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Laplace Equation on L-shaped Domain')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('exercise_11_2_l_shaped.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Solution range: [{np.min(u[mask]):.2f}, {np.max(u[mask]):.2f}]")
    print("Plot saved to exercise_11_2_l_shaped.png")


# === Exercise 3: Non-homogeneous Source ===
# Problem: Solve the Poisson equation with f(x,y) = sin(2*pi*x)*sin(2*pi*y)
# and compare with the analytical solution.

def exercise_3():
    """Solve Poisson equation with sinusoidal source and compare with analytical solution."""

    nx = ny = 51
    Lx = Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Source term: f(x,y) = sin(2*pi*x) * sin(2*pi*y)
    # Analytical solution: u(x,y) = -sin(2*pi*x)*sin(2*pi*y) / (8*pi^2)
    # since nabla^2 u = -(2*pi)^2 u - (2*pi)^2 u = -8*pi^2 * u = f
    # => u = -f / (8*pi^2)

    def f_source(X, Y):
        return np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    def u_analytical(X, Y):
        return -np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) / (8 * np.pi**2)

    # Build 2D Laplacian using Kronecker product
    mx = nx - 2
    my = ny - 2

    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(mx, mx)) / dx**2
    Dyy = diags([1, -2, 1], [-1, 0, 1], shape=(my, my)) / dy**2
    Ix = eye(mx)
    Iy = eye(my)
    L = kron(Iy, Dxx) + kron(Dyy, Ix)

    # Source term at interior points
    F = f_source(X[1:-1, 1:-1], Y[1:-1, 1:-1]).flatten()

    # Boundary conditions are all zero (Dirichlet), so no BC contribution needed
    u_inner = spsolve(L.tocsr(), F)

    # Reconstruct full solution
    u_numerical = np.zeros((ny, nx))
    u_numerical[1:-1, 1:-1] = u_inner.reshape((my, mx))

    # Analytical solution
    u_exact = u_analytical(X, Y)

    # Error
    error = np.max(np.abs(u_numerical - u_exact))
    print(f"Poisson equation with f(x,y) = sin(2*pi*x)*sin(2*pi*y)")
    print(f"Grid: {nx} x {ny}")
    print(f"Maximum error: {error:.6e}")
    print(f"Analytical solution max: {np.max(np.abs(u_exact)):.6f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    c1 = axes[0].contourf(X, Y, u_numerical, levels=30, cmap='RdBu_r')
    plt.colorbar(c1, ax=axes[0])
    axes[0].set_title('Numerical Solution')
    axes[0].set_aspect('equal')

    c2 = axes[1].contourf(X, Y, u_exact, levels=30, cmap='RdBu_r')
    plt.colorbar(c2, ax=axes[1])
    axes[1].set_title('Analytical Solution')
    axes[1].set_aspect('equal')

    c3 = axes[2].contourf(X, Y, np.abs(u_numerical - u_exact), levels=30, cmap='hot')
    plt.colorbar(c3, ax=axes[2])
    axes[2].set_title(f'Error (max: {error:.2e})')
    axes[2].set_aspect('equal')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('exercise_11_3_poisson_source.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_11_3_poisson_source.png")


# === Exercise 4: Red-Black Gauss-Seidel ===
# Problem: Implement Red-Black Gauss-Seidel that updates in a checkerboard pattern.

def exercise_4():
    """Red-Black Gauss-Seidel: checkerboard-pattern update for Laplace equation."""

    nx = ny = 51
    h = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    # Problem setup: Laplace equation, top boundary = 100, others = 0
    f = np.zeros((ny, nx))
    u = np.zeros((ny, nx))
    u[-1, :] = 100  # Top boundary

    tol = 1e-6
    max_iter = 10000
    h2 = h ** 2

    residuals = []

    for k in range(max_iter):
        u_old = u.copy()

        # Red sweep: update points where (i + j) is even
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if (i + j) % 2 == 0:  # Red points
                    u[j, i] = 0.25 * (
                        u[j, i+1] + u[j, i-1] +
                        u[j+1, i] + u[j-1, i] -
                        h2 * f[j, i]
                    )

        # Black sweep: update points where (i + j) is odd
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if (i + j) % 2 == 1:  # Black points
                    u[j, i] = 0.25 * (
                        u[j, i+1] + u[j, i-1] +
                        u[j+1, i] + u[j-1, i] -
                        h2 * f[j, i]
                    )

        residual = np.max(np.abs(u - u_old))
        residuals.append(residual)

        if residual < tol:
            print(f"Red-Black Gauss-Seidel converged in {k + 1} iterations")
            break
    else:
        print(f"Did not converge, final residual = {residuals[-1]:.2e}")

    # Compare with standard Gauss-Seidel
    u_gs = np.zeros((ny, nx))
    u_gs[-1, :] = 100
    residuals_gs = []

    for k in range(max_iter):
        u_gs_old = u_gs.copy()
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_gs[j, i] = 0.25 * (
                    u_gs[j, i+1] + u_gs[j, i-1] +
                    u_gs[j+1, i] + u_gs[j-1, i] -
                    h2 * f[j, i]
                )
        residual_gs = np.max(np.abs(u_gs - u_gs_old))
        residuals_gs.append(residual_gs)
        if residual_gs < tol:
            break

    print(f"Standard Gauss-Seidel converged in {len(residuals_gs)} iterations")
    print(f"Solution difference (max): {np.max(np.abs(u - u_gs)):.2e}")
    print("\nNote: Red-Black G-S has the same convergence rate as standard G-S,")
    print("but its sweeps are fully parallelizable (all red points are independent).")

    # Plot convergence comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(residuals, 'r-', label=f'Red-Black GS ({len(residuals)} iter)')
    ax.semilogy(residuals_gs, 'b-', label=f'Standard GS ({len(residuals_gs)} iter)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Red-Black vs Standard Gauss-Seidel')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_11_4_red_black_gs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_11_4_red_black_gs.png")


if __name__ == "__main__":
    print("=== Exercise 1: SOR Optimal omega Search ===")
    exercise_1()

    print("\n=== Exercise 2: L-shaped Domain ===")
    exercise_2()

    print("\n=== Exercise 3: Non-homogeneous Source ===")
    exercise_3()

    print("\n=== Exercise 4: Red-Black Gauss-Seidel ===")
    exercise_4()

    print("\nAll exercises completed!")

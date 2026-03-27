"""
Finite Difference Method
Finite Difference Methods for PDEs

Numerical methods for solving partial differential equations (PDEs).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# =============================================================================
# 1. 1D Heat Equation
# =============================================================================
def heat_equation_explicit(
    L: float,
    T: float,
    nx: int,
    nt: int,
    alpha: float,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_left: float = 0,
    boundary_right: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D Heat Equation (explicit method)

    du/dt = alpha * d^2u/dx^2

    FTCS (Forward Time, Central Space):
    u(i,n+1) = u(i,n) + r[u(i+1,n) - 2u(i,n) + u(i-1,n)]
    where r = alpha*dt/dx^2

    Stability condition: r <= 0.5

    Args:
        L: Spatial domain [0, L]
        T: Time domain [0, T]
        nx: Number of spatial grid points
        nt: Number of time steps
        alpha: Thermal diffusivity
        initial_condition: Initial condition function u(x, 0)
        boundary_left, boundary_right: Boundary conditions

    Returns:
        (x array, t array, u array)
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2

    print(f"dx={dx:.4f}, dt={dt:.6f}, r={r:.4f}")
    if r > 0.5:
        print(f"Warning: stability condition violated (r={r:.4f} > 0.5)")

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))

    # Initial condition
    u[0, :] = initial_condition(x)

    # Boundary conditions
    u[:, 0] = boundary_left
    u[:, -1] = boundary_right

    # Time stepping
    for n in range(nt):
        for i in range(1, nx - 1):
            u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])

    return x, t, u


def heat_equation_implicit(
    L: float,
    T: float,
    nx: int,
    nt: int,
    alpha: float,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_left: float = 0,
    boundary_right: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D Heat Equation (implicit method - Crank-Nicolson)

    Unconditionally stable, O(dt^2, dx^2) accuracy
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / (2 * dx**2)

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))

    # Initial condition
    u[0, :] = initial_condition(x)
    u[:, 0] = boundary_left
    u[:, -1] = boundary_right

    # Tridiagonal matrix setup
    n_inner = nx - 2
    A = np.zeros((n_inner, n_inner))
    B = np.zeros((n_inner, n_inner))

    for i in range(n_inner):
        A[i, i] = 1 + 2 * r
        B[i, i] = 1 - 2 * r
        if i > 0:
            A[i, i - 1] = -r
            B[i, i - 1] = r
        if i < n_inner - 1:
            A[i, i + 1] = -r
            B[i, i + 1] = r

    # Time stepping
    for n in range(nt):
        # Compute right-hand side
        b = B @ u[n, 1:-1]
        b[0] += r * (u[n + 1, 0] + u[n, 0])
        b[-1] += r * (u[n + 1, -1] + u[n, -1])

        # Solve linear system
        u[n + 1, 1:-1] = np.linalg.solve(A, b)

    return x, t, u


# =============================================================================
# 2. 1D Wave Equation
# =============================================================================
def wave_equation(
    L: float,
    T: float,
    nx: int,
    nt: int,
    c: float,
    initial_displacement: Callable[[np.ndarray], np.ndarray],
    initial_velocity: Callable[[np.ndarray], np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D Wave Equation

    d^2u/dt^2 = c^2 * d^2u/dx^2

    Finite difference:
    u(i,n+1) = 2u(i,n) - u(i,n-1) + s^2[u(i+1,n) - 2u(i,n) + u(i-1,n)]
    where s = c*dt/dx (Courant number)

    Stability condition: s <= 1 (CFL condition)
    """
    dx = L / (nx - 1)
    dt = T / nt
    s = c * dt / dx

    print(f"Courant number s={s:.4f}")
    if s > 1:
        print(f"Warning: CFL condition violated (s={s:.4f} > 1)")

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))

    # Initial condition
    u[0, :] = initial_displacement(x)

    # First time step (using initial velocity)
    if initial_velocity is None:
        initial_velocity = lambda x: np.zeros_like(x)

    v0 = initial_velocity(x)
    for i in range(1, nx - 1):
        u[1, i] = (u[0, i] + dt * v0[i] +
                   0.5 * s**2 * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1]))

    # Boundary conditions (fixed)
    u[:, 0] = 0
    u[:, -1] = 0

    # Time stepping
    for n in range(1, nt):
        for i in range(1, nx - 1):
            u[n + 1, i] = (2 * u[n, i] - u[n - 1, i] +
                          s**2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]))

    return x, t, u


# =============================================================================
# 3. 2D Laplace/Poisson Equation
# =============================================================================
def laplace_2d(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    boundary_conditions: dict,
    max_iter: int = 10000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D Laplace Equation (Jacobi iteration)

    nabla^2 u = 0  or  d^2u/dx^2 + d^2u/dy^2 = 0

    Jacobi iteration:
    u(i,j)_new = 0.25 * [u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1)]

    Args:
        boundary_conditions: {'top': val, 'bottom': val, 'left': val, 'right': val}
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    u = np.zeros((ny, nx))

    # Set boundary conditions
    bc = boundary_conditions
    if callable(bc.get('top')):
        u[-1, :] = bc['top'](x)
    else:
        u[-1, :] = bc.get('top', 0)

    if callable(bc.get('bottom')):
        u[0, :] = bc['bottom'](x)
    else:
        u[0, :] = bc.get('bottom', 0)

    if callable(bc.get('left')):
        u[:, 0] = bc['left'](y)
    else:
        u[:, 0] = bc.get('left', 0)

    if callable(bc.get('right')):
        u[:, -1] = bc['right'](y)
    else:
        u[:, -1] = bc.get('right', 0)

    # Jacobi iteration
    for iteration in range(max_iter):
        u_old = u.copy()

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = 0.25 * (u_old[i + 1, j] + u_old[i - 1, j] +
                                  u_old[i, j + 1] + u_old[i, j - 1])

        # Convergence check
        error = np.max(np.abs(u - u_old))
        if error < tol:
            print(f"Converged: {iteration + 1} iterations, error={error:.2e}")
            break

    return x, y, u


def laplace_2d_sor(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    boundary_conditions: dict,
    omega: float = 1.5,
    max_iter: int = 10000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D Laplace Equation (SOR - Successive Over-Relaxation)

    Faster convergence than Jacobi

    Args:
        omega: Relaxation factor (1 < omega < 2 for acceleration)
    """
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    u = np.zeros((ny, nx))

    bc = boundary_conditions
    u[-1, :] = bc.get('top', 0) if not callable(bc.get('top')) else bc['top'](x)
    u[0, :] = bc.get('bottom', 0) if not callable(bc.get('bottom')) else bc['bottom'](x)
    u[:, 0] = bc.get('left', 0) if not callable(bc.get('left')) else bc['left'](y)
    u[:, -1] = bc.get('right', 0) if not callable(bc.get('right')) else bc['right'](y)

    for iteration in range(max_iter):
        max_diff = 0

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u_old = u[i, j]
                u_gs = 0.25 * (u[i + 1, j] + u[i - 1, j] +
                              u[i, j + 1] + u[i, j - 1])
                u[i, j] = (1 - omega) * u_old + omega * u_gs
                max_diff = max(max_diff, abs(u[i, j] - u_old))

        if max_diff < tol:
            print(f"SOR converged: {iteration + 1} iterations")
            break

    return x, y, u


# =============================================================================
# Visualization
# =============================================================================
def plot_heat_equation():
    """Heat equation visualization"""
    # Initial condition: hot region in the center
    initial = lambda x: np.sin(np.pi * x)

    x, t, u = heat_equation_explicit(
        L=1.0, T=0.5, nx=51, nt=500, alpha=0.01,
        initial_condition=initial
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Spatiotemporal contour
    ax = axes[0]
    X, T_mesh = np.meshgrid(x, t)
    contour = ax.contourf(X, T_mesh, u, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax, label='Temperature')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Time t')
    ax.set_title('Heat Equation Solution')

    # Temporal profiles
    ax = axes[1]
    times_to_plot = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
    for idx in times_to_plot:
        ax.plot(x, u[idx, :], label=f't={t[idx]:.3f}')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Temperature u')
    ax.set_title('Temperature Distribution Over Time')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/heat_equation.png', dpi=150)
    plt.close()
    print("Graph saved: heat_equation.png")


def plot_wave_equation():
    """Wave equation visualization"""
    # Initial displacement: Gaussian pulse
    initial = lambda x: np.exp(-100 * (x - 0.5)**2)

    x, t, u = wave_equation(
        L=1.0, T=2.0, nx=101, nt=400, c=1.0,
        initial_displacement=initial
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Spatiotemporal contour
    ax = axes[0]
    X, T_mesh = np.meshgrid(x, t)
    contour = ax.contourf(X, T_mesh, u, levels=20, cmap='RdBu')
    plt.colorbar(contour, ax=ax, label='Displacement')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Time t')
    ax.set_title('Wave Equation Solution')

    # Snapshots
    ax = axes[1]
    for idx in [0, 50, 100, 150, 200]:
        ax.plot(x, u[idx, :], label=f't={t[idx]:.2f}')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Displacement u')
    ax.set_title('Wave Shape Over Time')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/wave_equation.png', dpi=150)
    plt.close()
    print("Graph saved: wave_equation.png")


def plot_laplace_2d():
    """2D Laplace equation visualization"""
    bc = {
        'top': lambda x: np.sin(np.pi * x),
        'bottom': 0,
        'left': 0,
        'right': 0
    }

    x, y, u = laplace_2d(
        Lx=1.0, Ly=1.0, nx=51, ny=51,
        boundary_conditions=bc,
        tol=1e-5
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    X, Y = np.meshgrid(x, y)

    # Contour
    ax = axes[0]
    contour = ax.contourf(X, Y, u, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Laplace Equation Solution')
    ax.set_aspect('equal')

    # 3D surface
    ax = axes[1]
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, u, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title('3D Surface')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/laplace_2d.png', dpi=150)
    plt.close()
    print("Graph saved: laplace_2d.png")


# =============================================================================
# Test
# =============================================================================
def main():
    print("=" * 60)
    print("Finite Difference Method")
    print("=" * 60)

    # 1. Heat equation
    print("\n[1] 1D Heat Equation")
    print("-" * 40)

    initial = lambda x: np.sin(np.pi * x)

    print("Explicit method (FTCS):")
    x, t, u_explicit = heat_equation_explicit(
        L=1.0, T=0.1, nx=21, nt=100, alpha=0.1,
        initial_condition=initial
    )
    print(f"  Center temperature at t=0.1: {u_explicit[-1, 10]:.6f}")

    print("\nImplicit method (Crank-Nicolson):")
    x, t, u_implicit = heat_equation_implicit(
        L=1.0, T=0.1, nx=21, nt=100, alpha=0.1,
        initial_condition=initial
    )
    print(f"  Center temperature at t=0.1: {u_implicit[-1, 10]:.6f}")

    # Analytical solution: u = exp(-pi^2*alpha*t) sin(pi*x)
    exact = np.exp(-np.pi**2 * 0.1 * 0.1) * np.sin(np.pi * 0.5)
    print(f"  Analytical solution: {exact:.6f}")

    # 2. Wave equation
    print("\n[2] 1D Wave Equation")
    print("-" * 40)

    initial_wave = lambda x: np.sin(np.pi * x)

    x, t, u_wave = wave_equation(
        L=1.0, T=2.0, nx=51, nt=200, c=1.0,
        initial_displacement=initial_wave
    )
    print(f"Period T = 2L/c = 2.0")
    print(f"Center displacement at t=2.0: {u_wave[-1, 25]:.6f}")
    print(f"(Should equal initial value: {initial_wave(0.5):.6f})")

    # 3. 2D Laplace equation
    print("\n[3] 2D Laplace Equation")
    print("-" * 40)

    bc = {'top': 100, 'bottom': 0, 'left': 0, 'right': 0}

    print("Jacobi iteration:")
    x, y, u_jacobi = laplace_2d(1.0, 1.0, 31, 31, bc, tol=1e-4)

    print("\nSOR (omega=1.5):")
    x, y, u_sor = laplace_2d_sor(1.0, 1.0, 31, 31, bc, omega=1.5, tol=1e-4)

    print(f"\nCenter temperature: {u_jacobi[15, 15]:.4f}")

    # Visualization
    try:
        plot_heat_equation()
        plot_wave_equation()
        plot_laplace_2d()
    except Exception as e:
        print(f"Graph generation failed: {e}")

    print("\n" + "=" * 60)
    print("Finite Difference Method Summary")
    print("=" * 60)
    print("""
    PDE types and methods:

    | PDE Type    | Example         | Recommended Method         |
    |-------------|-----------------|----------------------------|
    | Parabolic   | Heat equation   | FTCS, Crank-Nicolson       |
    | Hyperbolic  | Wave equation   | Central difference, Lax-Wendroff|
    | Elliptic    | Laplace         | Jacobi, GS, SOR            |

    Stability conditions:
    - Heat equation (explicit): r = alpha*dt/dx^2 <= 0.5
    - Wave equation: CFL = c*dt/dx <= 1

    Finite difference approximations:
    - Forward difference: du/dt ~ [u(t+dt) - u(t)] / dt
    - Backward difference: du/dt ~ [u(t) - u(t-dt)] / dt
    - Central difference: d^2u/dx^2 ~ [u(x+dx) - 2u(x) + u(x-dx)] / dx^2

    Production use:
    - scipy.ndimage: Simple filtering
    - FEniCS, FiPy: Python PDE frameworks
    - OpenFOAM: CFD (finite volume method)
    """)


if __name__ == "__main__":
    main()

# 11. Laplace and Poisson Equations

## Learning Objectives
- Understand the characteristics of elliptic PDEs
- Implement the 5-point stencil finite difference method
- Learn iterative methods (Jacobi, Gauss-Seidel, SOR)
- Convergence analysis and optimization
- Efficient implementation using scipy.sparse

---

**Why This Lesson Matters:** The Laplace and Poisson equations appear whenever a system reaches steady state. Unlike parabolic and hyperbolic PDEs where time marches forward, elliptic equations describe equilibrium configurations where every point in the domain simultaneously influences every other point. This global coupling means the solution cannot be obtained by time-stepping; instead, we must solve the entire domain at once, which leads naturally to large sparse linear systems and iterative solvers.

## 1. Laplace/Poisson Equation Theory

### 1.1 Definition

The Laplace equation describes steady-state phenomena with no sources, while the Poisson equation includes a source term $f(x,y)$:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 \quad \text{(Laplace)}$$

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x,y) \quad \text{(Poisson)}$$

A key property of the Laplace equation is the **mean value property**: the value at any interior point equals the average of its neighbors. This is exactly what the Jacobi iterative method exploits numerically.

```
Laplace Equation (homogeneous):
nabla²u = d²u/dx² + d²u/dy² = 0

Poisson Equation (non-homogeneous):
nabla²u = d²u/dx² + d²u/dy² = f(x, y)
```

### 1.2 Physical Applications

| Field | Equation | Meaning of u |
|-------|----------|--------------|
| Steady heat conduction | nabla²T = 0 | Temperature |
| Electrostatics | nabla²phi = -rho/epsilon | Electric potential |
| Fluid mechanics | nabla²psi = -omega | Stream function |
| Gravitational field | nabla²phi = 4*pi*G*rho | Gravitational potential |
| Membrane deformation | nabla²w = p/T | Displacement |

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def elliptic_pde_examples():
    """Elliptic PDE examples"""
    print("="*60)
    print("Elliptic PDE (Laplace/Poisson) Applications")
    print("="*60)

    print("\n[1] Steady Heat Conduction")
    print("    -k*nabla²T = Q (heat source)")
    print("    Determines interior temperature distribution when boundary temperature is fixed")

    print("\n[2] Electrostatics")
    print("    nabla²phi = -rho/epsilon_0")
    print("    Determines potential phi given charge distribution")

    print("\n[3] Elastic Membrane")
    print("    T*nabla²w = -p (w: displacement, T: tension, p: pressure)")
    print("    Uniform pressure applied to membrane with fixed boundary")

elliptic_pde_examples()
```

### 1.3 Maximum Principle

Solutions to the Laplace equation do not have extrema inside the domain.
(Maximum/minimum occur only on the boundary)

```python
def maximum_principle_demo():
    """Maximum principle demonstration"""
    # Analytical solution: u(x,y) = x² - y² (harmonic function)
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)

    u = X**2 - Y**2  # Laplacian = 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Contour plot
    ax1 = axes[0]
    c = ax1.contourf(X, Y, u, levels=20, cmap='RdBu_r')
    plt.colorbar(c, ax=ax1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('u(x,y) = x² - y² (harmonic function)')
    ax1.set_aspect('equal')

    # Compare boundary and interior values
    ax2 = axes[1]
    # Boundary values
    boundary_values = []
    # Top/bottom boundary
    boundary_values.extend(u[0, :].tolist())
    boundary_values.extend(u[-1, :].tolist())
    # Left/right boundary
    boundary_values.extend(u[:, 0].tolist())
    boundary_values.extend(u[:, -1].tolist())

    # Interior values
    interior_values = u[1:-1, 1:-1].flatten()

    ax2.hist(interior_values, bins=30, alpha=0.7, label='Interior', density=True)
    ax2.axvline(x=np.min(boundary_values), color='r', linestyle='--',
               label=f'Boundary min: {np.min(boundary_values):.2f}')
    ax2.axvline(x=np.max(boundary_values), color='g', linestyle='--',
               label=f'Boundary max: {np.max(boundary_values):.2f}')
    ax2.set_xlabel('u value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Maximum Principle: Interior values are between boundary values')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('maximum_principle.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Boundary value range: [{np.min(boundary_values):.2f}, {np.max(boundary_values):.2f}]")
    print(f"Interior value range: [{np.min(interior_values):.2f}, {np.max(interior_values):.2f}]")

# maximum_principle_demo()
```

---

## 2. Five-Point Stencil

### 2.1 Discretization

Central difference for 2D Laplacian:

```
nabla²u ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/dx²
        + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/dy²

Uniform grid (dx = dy = h):
nabla²u ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
```

### 2.2 Stencil Visualization

```
           [i, j+1]
              |
              |
  [i-1, j]---[i, j]---[i+1, j]
              |
              |
           [i, j-1]

Coefficients: 4 neighbors = 1, center = -4
```

### 2.3 Matrix Form

Discretizing Poisson equation nabla²u = f yields linear system Au = b.

```python
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

def create_laplacian_2d(nx, ny, dx, dy):
    """
    Create 2D Laplacian matrix

    Parameters:
    -----------
    nx, ny : int - number of grid points in each direction (including boundary)
    dx, dy : float - grid spacing

    Returns:
    --------
    A : sparse matrix - Laplacian matrix (for interior points)
    """
    # Number of interior points
    mx = nx - 2
    my = ny - 2
    n = mx * my

    # 1D Laplacian operators
    # d²/dx² ≈ (1, -2, 1) / dx²
    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(mx, mx)) / dx**2
    Dyy = diags([1, -2, 1], [-1, 0, 1], shape=(my, my)) / dy**2

    # 2D Laplacian: Kronecker product
    # nabla² = d²/dx² x I_y + I_x x d²/dy²
    Ix = eye(mx)
    Iy = eye(my)

    L = kron(Iy, Dxx) + kron(Dyy, Ix)

    return L.tocsr()


def create_laplacian_2d_explicit(nx, ny, h):
    """
    2D Laplacian matrix (explicit construction, uniform grid)

    Interior point indexing: k = (j-1)*mx + (i-1)
    """
    mx = nx - 2
    my = ny - 2
    n = mx * my

    # Build in COO format
    rows = []
    cols = []
    data = []

    for j in range(my):
        for i in range(mx):
            k = j * mx + i  # 1D index of current point

            # Center point: -4
            rows.append(k)
            cols.append(k)
            data.append(-4.0 / h**2)

            # Left neighbor (i-1, j)
            if i > 0:
                rows.append(k)
                cols.append(k - 1)
                data.append(1.0 / h**2)

            # Right neighbor (i+1, j)
            if i < mx - 1:
                rows.append(k)
                cols.append(k + 1)
                data.append(1.0 / h**2)

            # Bottom neighbor (i, j-1)
            if j > 0:
                rows.append(k)
                cols.append(k - mx)
                data.append(1.0 / h**2)

            # Top neighbor (i, j+1)
            if j < my - 1:
                rows.append(k)
                cols.append(k + mx)
                data.append(1.0 / h**2)

    from scipy.sparse import coo_matrix
    A = coo_matrix((data, (rows, cols)), shape=(n, n))

    return A.tocsr()


# Test
nx = ny = 5
h = 1.0 / (nx - 1)

A1 = create_laplacian_2d(nx, ny, h, h)
A2 = create_laplacian_2d_explicit(nx, ny, h)

print(f"Laplacian matrix size: {A1.shape}")
print(f"Number of non-zero elements: {A1.nnz}")
print(f"\nMatrix comparison error: {np.max(np.abs(A1 - A2))}")
```

---

## 3. Direct Solver

### 3.1 Sparse Matrix Direct Solve

```python
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

class PoissonSolverDirect:
    """
    2D Poisson equation direct solver

    nabla²u = f(x, y)
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        """
        Parameters:
        -----------
        Lx, Ly : float - domain size
        nx, ny : int - number of grid points
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        # Grid generation
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Laplacian matrix
        self.A = create_laplacian_2d(nx, ny, self.dx, self.dy)

        print(f"Poisson direct solver setup")
        print(f"  Grid: {nx} x {ny}")
        print(f"  Number of interior points: {(nx-2)*(ny-2)}")

    def solve(self, f_func, bc_func):
        """
        Solve Poisson equation

        Parameters:
        -----------
        f_func : callable - source term f(x, y)
        bc_func : callable - boundary condition u(x, y) at boundary

        Returns:
        --------
        u : 2D array - solution
        """
        mx = self.nx - 2
        my = self.ny - 2

        # Source term (interior points)
        X_inner = self.X[1:-1, 1:-1]
        Y_inner = self.Y[1:-1, 1:-1]
        f = f_func(X_inner, Y_inner).flatten()

        # Boundary condition contribution
        b = f.copy()

        # Apply boundary conditions
        u_bc = bc_func(self.X, self.Y)

        # Bottom boundary (j=0)
        b[:mx] -= u_bc[0, 1:-1] / self.dy**2

        # Top boundary (j=ny-1)
        b[-mx:] -= u_bc[-1, 1:-1] / self.dy**2

        # Left boundary (i=0)
        for j in range(my):
            b[j*mx] -= u_bc[j+1, 0] / self.dx**2

        # Right boundary (i=nx-1)
        for j in range(my):
            b[j*mx + mx - 1] -= u_bc[j+1, -1] / self.dx**2

        # Solve linear system
        u_inner = spsolve(self.A, b)

        # Reconstruct full solution
        u = u_bc.copy()
        u[1:-1, 1:-1] = u_inner.reshape((my, mx))

        return u


def demo_poisson_direct():
    """Poisson direct solver demo"""
    # Problem: nabla²u = -2*pi²*sin(pi*x)*sin(pi*y)
    # Boundary condition: u = 0
    # Analytical solution: u(x,y) = sin(pi*x)*sin(pi*y)

    solver = PoissonSolverDirect(Lx=1.0, Ly=1.0, nx=51, ny=51)

    def f_func(X, Y):
        return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    def bc_func(X, Y):
        return np.zeros_like(X)

    u = solver.solve(f_func, bc_func)

    # Analytical solution
    u_exact = np.sin(np.pi * solver.X) * np.sin(np.pi * solver.Y)

    # Error
    error = np.max(np.abs(u - u_exact))
    print(f"\nMaximum error: {error:.2e}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    c1 = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='viridis')
    plt.colorbar(c1, ax=axes[0])
    axes[0].set_title('Numerical Solution')
    axes[0].set_aspect('equal')

    c2 = axes[1].contourf(solver.X, solver.Y, u_exact, levels=30, cmap='viridis')
    plt.colorbar(c2, ax=axes[1])
    axes[1].set_title('Analytical Solution')
    axes[1].set_aspect('equal')

    c3 = axes[2].contourf(solver.X, solver.Y, np.abs(u - u_exact), levels=30, cmap='hot')
    plt.colorbar(c3, ax=axes[2])
    axes[2].set_title(f'Error (max: {error:.2e})')
    axes[2].set_aspect('equal')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('poisson_direct.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u, u_exact

# solver, u, u_exact = demo_poisson_direct()
```

---

## 4. Iterative Methods

### 4.1 Overview of Iterative Methods

For large-scale systems, iterative methods can be more efficient than direct solvers.

```
Transform Au = b into u^(k+1) = M*u^(k) + c form

Convergence condition: spectral radius of iteration matrix M, rho(M) < 1
```

### 4.2 Jacobi Iteration Method

Calculate new value at each point from previous values of neighbors:

```
u_{i,j}^(k+1) = (1/4) · (u_{i+1,j}^(k) + u_{i-1,j}^(k) + u_{i,j+1}^(k) + u_{i,j-1}^(k) - h²f_{i,j})
```

```python
class JacobiSolver:
    """
    Jacobi iteration method for Poisson/Laplace equation
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Assume uniform grid
        assert abs(self.dx - self.dy) < 1e-10, "Uniform grid required"
        self.h = self.dx

    def solve(self, f, u_bc, tol=1e-6, max_iter=10000, verbose=True):
        """
        Execute Jacobi iteration

        Parameters:
        -----------
        f : 2D array - source term
        u_bc : 2D array - initial value with boundary conditions set
        tol : float - convergence tolerance
        max_iter : int - maximum number of iterations

        Returns:
        --------
        u : 2D array - solution
        residuals : list - residual at each iteration
        """
        u = u_bc.copy()
        u_new = u.copy()
        h2 = self.h ** 2

        residuals = []

        for k in range(max_iter):
            # Jacobi update (interior points only)
            u_new[1:-1, 1:-1] = 0.25 * (
                u[1:-1, 2:] + u[1:-1, :-2] +  # left-right neighbors
                u[2:, 1:-1] + u[:-2, 1:-1] -  # top-bottom neighbors
                h2 * f[1:-1, 1:-1]            # source term
            )

            # Calculate residual
            residual = np.max(np.abs(u_new - u))
            residuals.append(residual)

            # Check convergence
            if residual < tol:
                if verbose:
                    print(f"Jacobi converged: {k+1} iterations, residual = {residual:.2e}")
                return u_new, residuals

            u = u_new.copy()

        if verbose:
            print(f"Jacobi: max iterations reached, residual = {residuals[-1]:.2e}")

        return u_new, residuals


def demo_jacobi():
    """Jacobi iteration demo"""
    solver = JacobiSolver(Lx=1.0, Ly=1.0, nx=51, ny=51)

    # Laplace equation (f = 0)
    # Boundary conditions: top temperature 100, others 0
    f = np.zeros((solver.ny, solver.nx))

    u_bc = np.zeros((solver.ny, solver.nx))
    u_bc[-1, :] = 100  # Top boundary

    u, residuals = solver.solve(f, u_bc, tol=1e-6, max_iter=10000)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Solution
    c = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='hot')
    plt.colorbar(c, ax=axes[0], label='Temperature')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Laplace Equation Solution (Jacobi)')
    axes[0].set_aspect('equal')

    # Convergence history
    axes[1].semilogy(residuals, 'b-')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Residual')
    axes[1].set_title(f'Jacobi Convergence ({len(residuals)} iterations)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobi_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u, residuals

# solver, u, residuals = demo_jacobi()
```

### 4.3 Gauss-Seidel Iteration Method

Use newly computed values immediately:

```
u_{i,j}^(k+1) = (1/4) · (u_{i+1,j}^(k) + u_{i-1,j}^(k+1) + u_{i,j+1}^(k) + u_{i,j-1}^(k+1) - h²f_{i,j})
```

```python
class GaussSeidelSolver:
    """
    Gauss-Seidel iteration method
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.h = self.dx

    def solve(self, f, u_bc, tol=1e-6, max_iter=10000, verbose=True):
        """
        Execute Gauss-Seidel iteration
        """
        u = u_bc.copy()
        h2 = self.h ** 2

        residuals = []

        for k in range(max_iter):
            max_change = 0.0

            # Gauss-Seidel update (update immediately in order)
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    u_old = u[j, i]
                    u[j, i] = 0.25 * (
                        u[j, i+1] + u[j, i-1] +  # left-right
                        u[j+1, i] + u[j-1, i] -  # top-bottom
                        h2 * f[j, i]
                    )
                    max_change = max(max_change, abs(u[j, i] - u_old))

            residuals.append(max_change)

            if max_change < tol:
                if verbose:
                    print(f"Gauss-Seidel converged: {k+1} iterations, residual = {max_change:.2e}")
                return u, residuals

        if verbose:
            print(f"Gauss-Seidel: max iterations reached, residual = {residuals[-1]:.2e}")

        return u, residuals


def compare_jacobi_gs():
    """Jacobi vs Gauss-Seidel comparison"""
    nx = ny = 51

    # Same problem setup
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    f = np.zeros((ny, nx))
    u_bc = np.zeros((ny, nx))
    u_bc[-1, :] = 100

    # Jacobi
    jacobi = JacobiSolver(nx=nx, ny=ny)
    u_jacobi, res_jacobi = jacobi.solve(f, u_bc, tol=1e-6, verbose=False)

    # Gauss-Seidel
    gs = GaussSeidelSolver(nx=nx, ny=ny)
    u_gs, res_gs = gs.solve(f, u_bc, tol=1e-6, verbose=False)

    print(f"Jacobi: {len(res_jacobi)} iterations")
    print(f"Gauss-Seidel: {len(res_gs)} iterations")
    print(f"Speedup: {len(res_jacobi) / len(res_gs):.2f}x")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(res_jacobi, 'b-', label=f'Jacobi ({len(res_jacobi)} iter)')
    ax.semilogy(res_gs, 'r-', label=f'Gauss-Seidel ({len(res_gs)} iter)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Jacobi vs Gauss-Seidel Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobi_vs_gs.png', dpi=150, bbox_inches='tight')
    plt.show()

    return res_jacobi, res_gs

# res_jacobi, res_gs = compare_jacobi_gs()
```

### 4.4 SOR (Successive Over-Relaxation)

Accelerate convergence by introducing over-relaxation parameter omega:

```
u_{i,j}^(k+1) = (1-omega)·u_{i,j}^(k) + omega·(Gauss-Seidel value)

Optimal omega ≈ 2 / (1 + sin(pi*h))  (h: grid spacing, square domain)
```

```python
class SORSolver:
    """
    SOR (Successive Over-Relaxation) method
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.h = self.dx

        # Optimal relaxation parameter (square domain)
        self.omega_opt = 2 / (1 + np.sin(np.pi * self.h))
        print(f"SOR optimal omega = {self.omega_opt:.4f}")

    def solve(self, f, u_bc, omega=None, tol=1e-6, max_iter=10000, verbose=True):
        """
        Execute SOR iteration

        Parameters:
        -----------
        omega : float - relaxation parameter (use optimal if None)
        """
        if omega is None:
            omega = self.omega_opt

        u = u_bc.copy()
        h2 = self.h ** 2

        residuals = []

        for k in range(max_iter):
            max_change = 0.0

            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    u_old = u[j, i]

                    # Gauss-Seidel value
                    u_gs = 0.25 * (
                        u[j, i+1] + u[j, i-1] +
                        u[j+1, i] + u[j-1, i] -
                        h2 * f[j, i]
                    )

                    # SOR update
                    u[j, i] = (1 - omega) * u_old + omega * u_gs

                    max_change = max(max_change, abs(u[j, i] - u_old))

            residuals.append(max_change)

            if max_change < tol:
                if verbose:
                    print(f"SOR (omega={omega:.3f}) converged: {k+1} iterations, residual = {max_change:.2e}")
                return u, residuals

        if verbose:
            print(f"SOR: max iterations reached, residual = {residuals[-1]:.2e}")

        return u, residuals


def demo_sor():
    """SOR demo: compare various omega values"""
    nx = ny = 51

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    f = np.zeros((ny, nx))
    u_bc = np.zeros((ny, nx))
    u_bc[-1, :] = 100

    sor_solver = SORSolver(nx=nx, ny=ny)

    omega_values = [1.0, 1.2, 1.5, 1.7, sor_solver.omega_opt, 1.95]
    results = {}

    for omega in omega_values:
        u, res = sor_solver.solve(f, u_bc.copy(), omega=omega, tol=1e-6, verbose=False)
        results[omega] = (u, res)
        print(f"omega = {omega:.3f}: {len(res)} iterations")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(omega_values)))

    for idx, omega in enumerate(omega_values):
        u, res = results[omega]
        label = f'omega = {omega:.3f}' + (' (optimal)' if abs(omega - sor_solver.omega_opt) < 0.01 else '')
        ax.semilogy(res, color=colors[idx], label=f'{label}: {len(res)} iter', linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('SOR: Convergence vs Relaxation Parameter omega')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sor_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

# results = demo_sor()
```

---

## 5. Convergence Analysis

### 5.1 Theoretical Convergence Rates

| Method | Iteration Matrix Spectral Radius | Convergence Rate |
|--------|----------------------------------|------------------|
| Jacobi | cos(pi*h) | Slow |
| Gauss-Seidel | cos²(pi*h) | 2x faster |
| SOR (optimal) | 1 - 2*pi*h | Much faster |

```python
def convergence_rate_analysis():
    """Convergence rate theoretical analysis"""
    h_values = np.array([1/10, 1/20, 1/40, 1/80, 1/160])

    # Theoretical spectral radii
    rho_jacobi = np.cos(np.pi * h_values)
    rho_gs = np.cos(np.pi * h_values)**2
    omega_opt = 2 / (1 + np.sin(np.pi * h_values))
    rho_sor = omega_opt - 1  # Optimal SOR

    # Number of iterations needed for convergence (reduce residual to 1e-6)
    target_reduction = -np.log(1e-6)  # ln(10^6)

    iter_jacobi = target_reduction / (-np.log(rho_jacobi))
    iter_gs = target_reduction / (-np.log(rho_gs))
    iter_sor = target_reduction / (-np.log(rho_sor))

    print("Convergence Theory Analysis")
    print("=" * 70)
    print(f"{'h':<10} {'n=1/h':<10} {'Jacobi':<12} {'G-S':<12} {'SOR':<12}")
    print("-" * 70)

    for i, h in enumerate(h_values):
        n = int(1/h)
        print(f"{h:<10.4f} {n:<10} {iter_jacobi[i]:<12.0f} {iter_gs[i]:<12.0f} {iter_sor[i]:<12.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_values = 1 / h_values

    ax1 = axes[0]
    ax1.semilogy(n_values, 1 - rho_jacobi, 'o-', label='Jacobi')
    ax1.semilogy(n_values, 1 - rho_gs, 's-', label='Gauss-Seidel')
    ax1.semilogy(n_values, 1 - rho_sor, '^-', label='SOR (optimal)')
    ax1.set_xlabel('n = 1/h')
    ax1.set_ylabel('1 - rho (convergence factor)')
    ax1.set_title('Spectral Radius')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.loglog(n_values, iter_jacobi, 'o-', label='Jacobi')
    ax2.loglog(n_values, iter_gs, 's-', label='Gauss-Seidel')
    ax2.loglog(n_values, iter_sor, '^-', label='SOR (optimal)')
    ax2.loglog(n_values, n_values**2, 'k--', alpha=0.5, label='O(n²)')
    ax2.loglog(n_values, n_values, 'k:', alpha=0.5, label='O(n)')
    ax2.set_xlabel('n = 1/h')
    ax2.set_ylabel('Iterations required')
    ax2.set_title('Iterations Needed for Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# convergence_rate_analysis()
```

### 5.2 Grid Size Comparison

```python
def grid_size_comparison():
    """Compare iterative methods by grid size"""
    grid_sizes = [21, 41, 61, 81]

    results = {'Jacobi': [], 'GS': [], 'SOR': []}

    for n in grid_sizes:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)

        f = np.zeros((n, n))
        u_bc = np.zeros((n, n))
        u_bc[-1, :] = 100

        # Jacobi
        solver = JacobiSolver(nx=n, ny=n)
        _, res = solver.solve(f, u_bc.copy(), tol=1e-6, verbose=False)
        results['Jacobi'].append(len(res))

        # Gauss-Seidel
        solver = GaussSeidelSolver(nx=n, ny=n)
        _, res = solver.solve(f, u_bc.copy(), tol=1e-6, verbose=False)
        results['GS'].append(len(res))

        # SOR
        solver = SORSolver(nx=n, ny=n)
        _, res = solver.solve(f, u_bc.copy(), tol=1e-6, verbose=False)
        results['SOR'].append(len(res))

        print(f"n={n}: Jacobi={results['Jacobi'][-1]}, GS={results['GS'][-1]}, SOR={results['SOR'][-1]}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(grid_sizes, results['Jacobi'], 'o-', label='Jacobi', linewidth=2, markersize=8)
    ax.loglog(grid_sizes, results['GS'], 's-', label='Gauss-Seidel', linewidth=2, markersize=8)
    ax.loglog(grid_sizes, results['SOR'], '^-', label='SOR (optimal)', linewidth=2, markersize=8)

    # Reference lines
    n_ref = np.array(grid_sizes)
    ax.loglog(n_ref, 0.5 * n_ref**2, 'k--', alpha=0.5, label='O(n²)')
    ax.loglog(n_ref, 2 * n_ref, 'k:', alpha=0.5, label='O(n)')

    ax.set_xlabel('Grid size n')
    ax.set_ylabel('Number of iterations')
    ax.set_title('Convergence Comparison by Grid Size')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('grid_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

# results = grid_size_comparison()
```

---

## 6. Practical Application Examples

### 6.1 Heat Conduction Problem

```python
def heat_conduction_example():
    """2D steady heat conduction problem"""
    # Square plate with different temperatures on each side
    nx = ny = 51

    solver = SORSolver(Lx=1.0, Ly=1.0, nx=nx, ny=ny)

    # Source term: no internal heat source
    f = np.zeros((ny, nx))

    # Boundary conditions
    u_bc = np.zeros((ny, nx))
    u_bc[0, :] = 0      # Bottom: 0 deg C
    u_bc[-1, :] = 100   # Top: 100 deg C
    u_bc[:, 0] = 50     # Left: 50 deg C
    u_bc[:, -1] = 50    # Right: 50 deg C

    # Corner treatment (average)
    u_bc[0, 0] = 25
    u_bc[0, -1] = 25
    u_bc[-1, 0] = 75
    u_bc[-1, -1] = 75

    u, _ = solver.solve(f, u_bc, tol=1e-6)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Contours
    c1 = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='hot')
    plt.colorbar(c1, ax=axes[0], label='Temperature (deg C)')
    axes[0].contour(solver.X, solver.Y, u, levels=10, colors='white', alpha=0.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Steady Heat Conduction (Temperature Distribution)')
    axes[0].set_aspect('equal')

    # Heat flux vectors
    # q = -k*nabla*T
    qx, qy = np.gradient(u, solver.dx, solver.dy)
    qx = -qx
    qy = -qy

    skip = 3
    axes[1].contourf(solver.X, solver.Y, u, levels=30, cmap='hot', alpha=0.5)
    axes[1].quiver(solver.X[::skip, ::skip], solver.Y[::skip, ::skip],
                   qx[::skip, ::skip], qy[::skip, ::skip],
                   scale=500, color='blue')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Heat Flux Vectors')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('heat_conduction.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u

# solver, u = heat_conduction_example()
```

### 6.2 Electrostatic Problem

```python
def electrostatic_example():
    """Electrostatic problem: potential between two conductors"""
    nx = ny = 81

    solver = SORSolver(Lx=1.0, Ly=1.0, nx=nx, ny=ny)

    # Boundary conditions: all external boundaries grounded (0V)
    u_bc = np.zeros((ny, nx))

    # Two circular conductors inside
    # Conductor 1: center (0.3, 0.5), radius 0.1, potential +100V
    # Conductor 2: center (0.7, 0.5), radius 0.1, potential -100V

    def is_inside_conductor(X, Y, cx, cy, r):
        return (X - cx)**2 + (Y - cy)**2 <= r**2

    conductor1_mask = is_inside_conductor(solver.X, solver.Y, 0.3, 0.5, 0.08)
    conductor2_mask = is_inside_conductor(solver.X, solver.Y, 0.7, 0.5, 0.08)

    u_bc[conductor1_mask] = 100
    u_bc[conductor2_mask] = -100

    # Source term: no charge (Laplace)
    f = np.zeros((ny, nx))

    # Solve (conductor interior fixed as boundary condition)
    u = u_bc.copy()

    for _ in range(5000):
        u_new = u.copy()

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # Skip conductor interior
                if conductor1_mask[j, i] or conductor2_mask[j, i]:
                    continue

                u_new[j, i] = 0.25 * (u[j, i+1] + u[j, i-1] + u[j+1, i] + u[j-1, i])

        if np.max(np.abs(u_new - u)) < 1e-6:
            break

        u = u_new.copy()

    # Calculate electric field: E = -nabla*phi
    Ey, Ex = np.gradient(u, solver.dy, solver.dx)
    Ex = -Ex
    Ey = -Ey
    E_mag = np.sqrt(Ex**2 + Ey**2)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Potential
    c1 = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='RdBu_r')
    plt.colorbar(c1, ax=axes[0], label='Potential (V)')
    axes[0].contour(solver.X, solver.Y, u, levels=20, colors='k', alpha=0.3)

    # Show conductors
    theta = np.linspace(0, 2*np.pi, 50)
    axes[0].plot(0.3 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'r-', linewidth=2)
    axes[0].plot(0.7 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'b-', linewidth=2)

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Potential Distribution')
    axes[0].set_aspect('equal')

    # Electric field
    axes[1].contourf(solver.X, solver.Y, E_mag, levels=30, cmap='hot')
    skip = 4
    axes[1].streamplot(solver.X, solver.Y, Ex, Ey, color='white', density=1.5, linewidth=0.5)

    axes[1].plot(0.3 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'r-', linewidth=2)
    axes[1].plot(0.7 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'b-', linewidth=2)

    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Electric Field (Streamlines)')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('electrostatic.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u

# solver, u = electrostatic_example()
```

---

## 7. Summary

### Iterative Method Comparison

| Method | Characteristics | Convergence Speed | Memory |
|--------|-----------------|-------------------|--------|
| Jacobi | Easy parallelization | O(n²) iterations | 2x storage |
| Gauss-Seidel | Sequential update | 2x faster | In-place |
| SOR | Over-relaxation acceleration | O(n) iterations | In-place |
| Direct method | Single solve | - | O(n²) storage |

### Optimal Relaxation Parameter

```
omega_opt = 2 / (1 + sin(pi*h))

Example: h = 1/50 -> omega_opt ≈ 1.937
```

### Next Steps

1. **Chapter 12**: Advection Equation - first-order hyperbolic PDE
2. Multigrid method - faster convergence
3. Conjugate Gradient (CG) - large-scale systems

---

## Exercises

### Exercise 1: SOR Optimal omega Search
Experimentally find optimal omega for various grid sizes and compare with theoretical values.

### Exercise 2: L-shaped Domain
Solve the Laplace equation on an L-shaped domain.

### Exercise 3: Non-homogeneous Source
Solve the Poisson equation with f(x,y) = sin(2*pi*x)·sin(2*pi*y) and compare with analytical solution.

### Exercise 4: Red-Black Gauss-Seidel
Implement Red-Black G-S that updates in a checkerboard pattern.

---

## References

1. **Textbook**: LeVeque, "Finite Difference Methods"
2. **Iterative methods**: Saad, "Iterative Methods for Sparse Linear Systems"
3. **Python**: scipy.sparse, scipy.sparse.linalg

# 09. Heat Equation

## Learning Objectives
- Understand the physical meaning of 1D/2D heat equations
- Implement FTCS (Forward Time Central Space) explicit method
- Implement BTCS (Backward Time Central Space) implicit method
- Understand and implement the Crank-Nicolson method
- Learn how to handle various boundary conditions

---

## 1. Heat Equation Theory

### 1.1 Physical Background

The heat equation is a parabolic PDE that describes heat conduction phenomena.

```
1D Heat Equation:
∂u/∂t = α · ∂²u/∂x²

Where:
- u(x,t): temperature
- α: thermal diffusivity
- x: spatial coordinate
- t: time
```

### 1.2 Thermal Diffusivity

```python
"""
Thermal diffusivity α = k / (ρ·c)

Where:
- k: thermal conductivity (W/m·K)
- ρ: density (kg/m³)
- c: specific heat (J/kg·K)
"""

# Thermal diffusivity by material (m²/s)
thermal_diffusivity = {
    'Copper': 1.11e-4,
    'Aluminum': 9.7e-5,
    'Iron': 2.3e-5,
    'Concrete': 7.5e-7,
    'Water': 1.43e-7,
    'Air': 2.2e-5,
}

for material, alpha in thermal_diffusivity.items():
    print(f"{material}: α = {alpha:.2e} m²/s")
```

### 1.3 Analytical Solution (Separation of Variables)

For boundary conditions u(0,t) = u(L,t) = 0 and initial condition u(x,0) = f(x):

```
u(x,t) = Σ Bₙ · sin(nπx/L) · exp(-α(nπ/L)²t)

Bₙ = (2/L) ∫₀^L f(x)·sin(nπx/L) dx
```

```python
import numpy as np
import matplotlib.pyplot as plt

def exact_solution_heat(x, t, alpha, L, n_terms=50):
    """
    Analytical solution of heat equation (Fourier series)

    Initial condition: u(x,0) = sin(πx/L) (first mode only)
    Boundary conditions: u(0,t) = u(L,t) = 0
    """
    # For simple initial condition
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

# Visualization
L = 1.0
alpha = 0.01
x = np.linspace(0, L, 101)

fig, ax = plt.subplots(figsize=(10, 6))

times = [0, 0.5, 1.0, 2.0, 5.0]
for t in times:
    u = exact_solution_heat(x, t, alpha, L)
    ax.plot(x, u, label=f't = {t}')

ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('1D Heat Equation Analytical Solution (Evolution Over Time)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('heat_exact.png', dpi=150)
# plt.show()
```

---

## 2. FTCS Explicit Method

### 2.1 Discretization

FTCS = Forward Time, Central Space

```
Time: Forward difference
∂u/∂t ≈ (u_i^{n+1} - u_i^n) / Δt

Space: Central difference
∂²u/∂x² ≈ (u_{i+1}^n - 2u_i^n + u_{i-1}^n) / Δx²

Combined:
u_i^{n+1} = u_i^n + r·(u_{i+1}^n - 2u_i^n + u_{i-1}^n)

Where r = α·Δt/Δx² (stability condition: r ≤ 0.5)
```

### Why r ≤ 0.5? Von Neumann Stability Analysis

The FTCS scheme for the heat equation has the stability parameter r = α·Δt/Δx². To determine the stability bound, we substitute a Fourier mode u_j^n = G^n · e^{ikjΔx} into the discrete equation and solve for the amplification factor:

```
G = 1 - 4r·sin²(kΔx/2)
```

For stability we need |G| ≤ 1, which requires 0 ≤ 4r·sin²(kΔx/2) ≤ 2. The worst case is sin² = 1 (the highest frequency mode), giving **r ≤ 0.5**. Physically, this means the time step must be small enough that heat doesn't "skip over" grid cells — the numerical diffusion rate must match or exceed the physical diffusion rate.

### 2.2 FTCS Stencil Visualization

```
Time n+1:         [i]
                   ↑
Time n:    [i-1]--[i]--[i+1]
```

### 2.3 FTCS Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HeatEquation1D_FTCS:
    """
    1D Heat Equation FTCS Explicit Method

    ∂u/∂t = α · ∂²u/∂x²
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, safety=0.4):
        """
        Parameters:
        -----------
        L : float - domain length
        alpha : float - thermal diffusivity
        nx : int - number of spatial grid points
        T : float - final time
        safety : float - CFL safety factor (0 < safety ≤ 0.5)
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T

        # Generate grid
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

        # Determine time step based on stability condition
        self.dt = safety * self.dx**2 / alpha
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt  # Adjust to reach T exactly

        self.r = alpha * self.dt / self.dx**2

        print(f"FTCS Heat Equation Setup")
        print(f"  dx = {self.dx:.4f}, dt = {self.dt:.6f}")
        print(f"  r = α·dt/dx² = {self.r:.4f}")
        print(f"  Number of time steps: {self.nt}")
        print(f"  Stability: {'OK' if self.r <= 0.5 else 'WARNING!'}")

    def set_initial_condition(self, func):
        """Set initial condition"""
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, left_type='dirichlet', left_value=0,
                                  right_type='dirichlet', right_value=0):
        """Set boundary conditions"""
        self.bc = {
            'left': {'type': left_type, 'value': left_value},
            'right': {'type': right_type, 'value': right_value}
        }

    def apply_bc(self, u):
        """Apply boundary conditions"""
        # Left boundary
        if self.bc['left']['type'] == 'dirichlet':
            u[0] = self.bc['left']['value']
        elif self.bc['left']['type'] == 'neumann':
            # ∂u/∂x = flux => u[0] = u[1] - flux * dx
            u[0] = u[1] - self.bc['left']['value'] * self.dx

        # Right boundary
        if self.bc['right']['type'] == 'dirichlet':
            u[-1] = self.bc['right']['value']
        elif self.bc['right']['type'] == 'neumann':
            # ∂u/∂x = flux => u[-1] = u[-2] + flux * dx
            u[-1] = u[-2] + self.bc['right']['value'] * self.dx

        return u

    def step(self):
        """Advance one time step (FTCS)"""
        u_new = self.u.copy()

        # Update interior points
        u_new[1:-1] = self.u[1:-1] + self.r * (
            self.u[2:] - 2*self.u[1:-1] + self.u[:-2]
        )

        # Apply boundary conditions
        u_new = self.apply_bc(u_new)

        self.u = u_new

    def solve(self, save_interval=None):
        """Solve over entire time domain"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_ftcs():
    """FTCS demo"""
    # Problem setup
    solver = HeatEquation1D_FTCS(L=1.0, alpha=0.01, nx=51, T=2.0, safety=0.4)

    # Initial condition: sine wave
    solver.set_initial_condition(lambda x: np.sin(np.pi * x))

    # Boundary conditions: fixed at both ends
    solver.set_boundary_conditions('dirichlet', 0, 'dirichlet', 0)

    # Solve
    times, history = solver.solve(save_interval=20)

    # Compare with analytical solution
    u_exact = exact_solution_heat(solver.x, times[-1], solver.alpha, solver.L)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Solution over time
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    for i, (t, u) in enumerate(zip(times[::10], history[::10])):
        ax1.plot(solver.x, u, color=colors[i*10] if i*10 < len(colors) else colors[-1],
                label=f't={t:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('FTCS Heat Equation Solution')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Comparison at final time
    ax2 = axes[1]
    ax2.plot(solver.x, history[-1], 'b-', label='FTCS numerical', linewidth=2)
    ax2.plot(solver.x, u_exact, 'r--', label='Analytical', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.set_title(f'Comparison with Analytical Solution at t = {times[-1]:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    error = np.max(np.abs(history[-1] - u_exact))
    print(f"\nMaximum error at final time: {error:.2e}")

    plt.tight_layout()
    plt.savefig('heat_ftcs.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, times, history

# solver, times, history = demo_ftcs()
```

---

## 3. BTCS Implicit Method

### 3.1 Discretization

BTCS = Backward Time, Central Space

```
Time: Backward difference (evaluated at n+1)
∂u/∂t ≈ (u_i^{n+1} - u_i^n) / Δt

Space: Central difference (at n+1)
∂²u/∂x² ≈ (u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}) / Δx²

Rearranged:
-r·u_{i-1}^{n+1} + (1+2r)·u_i^{n+1} - r·u_{i+1}^{n+1} = u_i^n
```

### 3.2 Matrix Form

```
A · u^{n+1} = u^n

Where A is a tridiagonal matrix:
    | 1+2r  -r    0   ...  |
    | -r   1+2r  -r   ...  |
A = |  0    -r  1+2r  ...  |
    | ...               -r |
    |             -r  1+2r |
```

### 3.3 BTCS Implementation

```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class HeatEquation1D_BTCS:
    """
    1D Heat Equation BTCS Implicit Method

    Unconditionally stable
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, nt=100):
        """
        Parameters:
        -----------
        L : float - domain length
        alpha : float - thermal diffusivity
        nx : int - number of spatial grid points
        T : float - final time
        nt : int - number of time steps
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T
        self.nt = nt

        # Generate grid
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)

        self.r = alpha * self.dt / self.dx**2

        print(f"BTCS Heat Equation Setup")
        print(f"  dx = {self.dx:.4f}, dt = {self.dt:.6f}")
        print(f"  r = α·dt/dx² = {self.r:.4f}")
        print(f"  BTCS is unconditionally stable (no restriction on r)")

        # Build matrix A (interior points only)
        self._build_matrix()

    def _build_matrix(self):
        """Build BTCS matrix"""
        n = self.nx - 2  # Number of interior points

        main_diag = (1 + 2*self.r) * np.ones(n)
        off_diag = -self.r * np.ones(n - 1)

        self.A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

    def set_initial_condition(self, func):
        """Set initial condition"""
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, left_value=0, right_value=0):
        """Set Dirichlet boundary conditions"""
        self.u_left = left_value
        self.u_right = right_value

    def step(self):
        """Advance one time step (BTCS)"""
        # Right-hand side vector (interior points)
        b = self.u[1:-1].copy()

        # Boundary condition contribution
        b[0] += self.r * self.u_left
        b[-1] += self.r * self.u_right

        # Solve linear system
        u_inner = spsolve(self.A, b)

        # Update full solution
        self.u[1:-1] = u_inner
        self.u[0] = self.u_left
        self.u[-1] = self.u_right

    def solve(self, save_interval=None):
        """Solve over entire time domain"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def compare_ftcs_btcs():
    """Compare FTCS vs BTCS"""
    L = 1.0
    alpha = 0.01
    nx = 51
    T = 2.0

    # FTCS (CFL restricted)
    ftcs = HeatEquation1D_FTCS(L, alpha, nx, T, safety=0.4)
    ftcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    ftcs.set_boundary_conditions('dirichlet', 0, 'dirichlet', 0)
    times_ftcs, history_ftcs = ftcs.solve()

    # BTCS (can use larger time steps)
    btcs = HeatEquation1D_BTCS(L, alpha, nx, T, nt=50)  # Much fewer time steps
    btcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    btcs.set_boundary_conditions(0, 0)
    times_btcs, history_btcs = btcs.solve()

    # Analytical solution
    u_exact = exact_solution_heat(ftcs.x, T, alpha, L)

    # Comparison
    print(f"\nComparison Results:")
    print(f"  FTCS time steps: {ftcs.nt}")
    print(f"  BTCS time steps: {btcs.nt}")
    print(f"  FTCS maximum error: {np.max(np.abs(history_ftcs[-1] - u_exact)):.2e}")
    print(f"  BTCS maximum error: {np.max(np.abs(history_btcs[-1] - u_exact)):.2e}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(ftcs.x, history_ftcs[-1], 'b-', label=f'FTCS (dt={ftcs.dt:.5f})', linewidth=2)
    ax1.plot(btcs.x, history_btcs[-1], 'g--', label=f'BTCS (dt={btcs.dt:.4f})', linewidth=2)
    ax1.plot(ftcs.x, u_exact, 'r:', label='Analytical', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title(f'Comparison at t = {T}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(ftcs.x, np.abs(history_ftcs[-1] - u_exact) + 1e-16, 'b-', label='FTCS error')
    ax2.semilogy(btcs.x, np.abs(history_btcs[-1] - u_exact) + 1e-16, 'g--', label='BTCS error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Numerical Error Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heat_ftcs_vs_btcs.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_ftcs_btcs()
```

---

## 4. Crank-Nicolson Method

### 4.1 Theory

Crank-Nicolson = Average of FTCS and BTCS (2nd order accuracy)

```
(u_i^{n+1} - u_i^n) / Δt = (α/2) · [(∂²u/∂x²)^n + (∂²u/∂x²)^{n+1}]

Rearranged:
-r/2·u_{i-1}^{n+1} + (1+r)·u_i^{n+1} - r/2·u_{i+1}^{n+1}
    = r/2·u_{i-1}^n + (1-r)·u_i^n + r/2·u_{i+1}^n
```

### 4.2 Matrix Form

```
A · u^{n+1} = B · u^n

A: (1+r) diagonal, -r/2 off-diagonal
B: (1-r) diagonal, r/2 off-diagonal
```

### 4.3 Crank-Nicolson Implementation

```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class HeatEquation1D_CrankNicolson:
    """
    1D Heat Equation Crank-Nicolson Method

    - Unconditionally stable
    - 2nd order accuracy (both time and space)
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, nt=100):
        """
        Parameters:
        -----------
        L : float - domain length
        alpha : float - thermal diffusivity
        nx : int - number of spatial grid points
        T : float - final time
        nt : int - number of time steps
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T
        self.nt = nt

        # Generate grid
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)

        self.r = alpha * self.dt / self.dx**2

        print(f"Crank-Nicolson Heat Equation Setup")
        print(f"  dx = {self.dx:.4f}, dt = {self.dt:.6f}")
        print(f"  r = α·dt/dx² = {self.r:.4f}")
        print(f"  2nd order accuracy & unconditionally stable")

        # Build matrices
        self._build_matrices()

    def _build_matrices(self):
        """Build Crank-Nicolson matrices A, B"""
        n = self.nx - 2  # Number of interior points
        r = self.r

        # A matrix: left side (implicit part)
        main_A = (1 + r) * np.ones(n)
        off_A = (-r/2) * np.ones(n - 1)
        self.A = diags([off_A, main_A, off_A], [-1, 0, 1], format='csr')

        # B matrix: right side (explicit part)
        main_B = (1 - r) * np.ones(n)
        off_B = (r/2) * np.ones(n - 1)
        self.B = diags([off_B, main_B, off_B], [-1, 0, 1], format='csr')

    def set_initial_condition(self, func):
        """Set initial condition"""
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, left_value=0, right_value=0):
        """Set Dirichlet boundary conditions"""
        self.u_left = left_value
        self.u_right = right_value

    def step(self):
        """Advance one time step (Crank-Nicolson)"""
        r = self.r

        # Right-hand side: B·u^n + boundary condition contribution
        b = self.B @ self.u[1:-1]

        # Boundary condition contribution (both left and right sides)
        b[0] += (r/2) * (self.u_left + self.u_left)  # BC at n and n+1
        b[-1] += (r/2) * (self.u_right + self.u_right)

        # Solve linear system
        u_inner = spsolve(self.A, b)

        # Update full solution
        self.u[1:-1] = u_inner
        self.u[0] = self.u_left
        self.u[-1] = self.u_right

    def solve(self, save_interval=None):
        """Solve over entire time domain"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def compare_all_schemes():
    """Compare FTCS, BTCS, Crank-Nicolson"""
    L = 1.0
    alpha = 0.01
    nx = 51
    T = 2.0

    # Same (large) time step for comparison
    nt = 40  # FTCS would be unstable

    # Crank-Nicolson
    cn = HeatEquation1D_CrankNicolson(L, alpha, nx, T, nt)
    cn.set_initial_condition(lambda x: np.sin(np.pi * x))
    cn.set_boundary_conditions(0, 0)
    times_cn, history_cn = cn.solve()

    # BTCS
    btcs = HeatEquation1D_BTCS(L, alpha, nx, T, nt)
    btcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    btcs.set_boundary_conditions(0, 0)
    times_btcs, history_btcs = btcs.solve()

    # FTCS (stable settings)
    ftcs = HeatEquation1D_FTCS(L, alpha, nx, T, safety=0.4)
    ftcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    ftcs.set_boundary_conditions('dirichlet', 0, 'dirichlet', 0)
    times_ftcs, history_ftcs = ftcs.solve()

    # Analytical solution
    u_exact = exact_solution_heat(cn.x, T, alpha, L)

    # Error comparison
    print(f"\nAccuracy Comparison (t = {T}):")
    print(f"  FTCS (dt={ftcs.dt:.5f}, {ftcs.nt} steps): {np.max(np.abs(history_ftcs[-1] - u_exact)):.2e}")
    print(f"  BTCS (dt={btcs.dt:.4f}, {btcs.nt} steps): {np.max(np.abs(history_btcs[-1] - u_exact)):.2e}")
    print(f"  C-N  (dt={cn.dt:.4f}, {cn.nt} steps): {np.max(np.abs(history_cn[-1] - u_exact)):.2e}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(cn.x, history_ftcs[-1], 'b-', label='FTCS', linewidth=2)
    ax1.plot(cn.x, history_btcs[-1], 'g--', label='BTCS', linewidth=2)
    ax1.plot(cn.x, history_cn[-1], 'm:', label='Crank-Nicolson', linewidth=2)
    ax1.plot(cn.x, u_exact, 'r-.', label='Analytical', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title(f'Three Schemes Comparison at t = {T}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy convergence test
    ax2 = axes[1]
    dt_values = []
    errors_btcs = []
    errors_cn = []

    for nt_test in [20, 40, 80, 160, 320]:
        dt_test = T / nt_test
        dt_values.append(dt_test)

        # BTCS
        solver = HeatEquation1D_BTCS(L, alpha, nx, T, nt_test)
        solver.set_initial_condition(lambda x: np.sin(np.pi * x))
        solver.set_boundary_conditions(0, 0)
        _, hist = solver.solve()
        errors_btcs.append(np.max(np.abs(hist[-1] - u_exact)))

        # Crank-Nicolson
        solver = HeatEquation1D_CrankNicolson(L, alpha, nx, T, nt_test)
        solver.set_initial_condition(lambda x: np.sin(np.pi * x))
        solver.set_boundary_conditions(0, 0)
        _, hist = solver.solve()
        errors_cn.append(np.max(np.abs(hist[-1] - u_exact)))

    ax2.loglog(dt_values, errors_btcs, 'gs-', label='BTCS (1st order)', linewidth=2)
    ax2.loglog(dt_values, errors_cn, 'mo-', label='Crank-Nicolson (2nd order)', linewidth=2)

    # Reference lines
    dt_ref = np.array(dt_values)
    ax2.loglog(dt_ref, 0.5*dt_ref, 'k--', alpha=0.5, label='O(Δt)')
    ax2.loglog(dt_ref, 0.5*dt_ref**2, 'k:', alpha=0.5, label='O(Δt²)')

    ax2.set_xlabel('Δt')
    ax2.set_ylabel('Maximum Error')
    ax2.set_title('Temporal Accuracy Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('heat_scheme_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_all_schemes()
```

---

## 5. 2D Heat Equation

### 5.1 2D Heat Equation

```
∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²) = α · ∇²u
```

### 5.2 FTCS 2D Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HeatEquation2D_FTCS:
    """
    2D Heat Equation FTCS Explicit Method

    ∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²)
    """

    def __init__(self, Lx=1.0, Ly=1.0, alpha=0.01, nx=51, ny=51, T=0.5, safety=0.2):
        """
        Parameters:
        -----------
        Lx, Ly : float - domain size
        alpha : float - thermal diffusivity
        nx, ny : int - number of grid points
        T : float - final time
        safety : float - CFL safety factor
        """
        self.Lx = Lx
        self.Ly = Ly
        self.alpha = alpha
        self.nx = nx
        self.ny = ny
        self.T = T

        # Generate grid
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 2D CFL condition: r_x + r_y ≤ 0.5
        dt_cfl = safety * 0.5 / (alpha * (1/self.dx**2 + 1/self.dy**2))
        self.dt = dt_cfl
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.rx = alpha * self.dt / self.dx**2
        self.ry = alpha * self.dt / self.dy**2

        print(f"2D FTCS Heat Equation Setup")
        print(f"  Grid: {nx} x {ny}")
        print(f"  dx = {self.dx:.4f}, dy = {self.dy:.4f}, dt = {self.dt:.6f}")
        print(f"  r_x = {self.rx:.4f}, r_y = {self.ry:.4f}")
        print(f"  r_x + r_y = {self.rx + self.ry:.4f} (must be ≤ 0.5 for stability)")

    def set_initial_condition(self, func):
        """Set initial condition: u(x,y,0) = func(X, Y)"""
        self.u = func(self.X, self.Y)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, bc_value=0):
        """Dirichlet boundary conditions (same value on all boundaries)"""
        self.bc_value = bc_value

    def apply_bc(self, u):
        """Apply boundary conditions"""
        u[0, :] = self.bc_value   # Bottom
        u[-1, :] = self.bc_value  # Top
        u[:, 0] = self.bc_value   # Left
        u[:, -1] = self.bc_value  # Right
        return u

    def step(self):
        """Advance one time step (2D FTCS)"""
        u_new = self.u.copy()

        # Update interior points
        u_new[1:-1, 1:-1] = self.u[1:-1, 1:-1] + \
            self.rx * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) + \
            self.ry * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1])

        # Boundary conditions
        u_new = self.apply_bc(u_new)

        self.u = u_new

    def solve(self, save_interval=None):
        """Solve over entire time domain"""
        if save_interval is None:
            save_interval = max(1, self.nt // 50)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), history


def demo_heat_2d():
    """2D heat equation demo"""
    # Problem setup
    solver = HeatEquation2D_FTCS(Lx=1.0, Ly=1.0, alpha=0.01, nx=51, ny=51, T=0.5)

    # Initial condition: Gaussian hot spot
    def ic(X, Y):
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return np.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2))

    solver.set_initial_condition(ic)
    solver.set_boundary_conditions(0)

    # Solve
    times, history = solver.solve(save_interval=20)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Solution at selected times
    plot_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    for idx, i in enumerate(plot_indices[:5]):
        if idx < 5:
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            c = ax.contourf(solver.X, solver.Y, history[i], levels=30, cmap='hot')
            plt.colorbar(c, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f't = {times[i]:.3f}')
            ax.set_aspect('equal')

    # Handle empty subplot
    if len(plot_indices) < 6:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('heat_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Temperature at center point over time
    fig2, ax = plt.subplots(figsize=(10, 5))
    center_values = [h[solver.ny//2, solver.nx//2] for h in history]
    ax.plot(times, center_values, 'b-', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel('u(0.5, 0.5, t)')
    ax.set_title('Temperature at Center Point')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heat_2d_center.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, times, history

# solver, times, history = demo_heat_2d()
```

### 5.3 2D Crank-Nicolson (ADI Method)

For large 2D problems, the ADI (Alternating Direction Implicit) method is efficient.

```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class HeatEquation2D_ADI:
    """
    2D Heat Equation ADI (Alternating Direction Implicit) Method

    Each time step is split into two half-steps:
    1. x-direction implicit, y-direction explicit
    2. y-direction implicit, x-direction explicit

    Unconditionally stable + 2nd order accuracy
    """

    def __init__(self, Lx=1.0, Ly=1.0, alpha=0.01, nx=51, ny=51, T=0.5, nt=100):
        self.Lx = Lx
        self.Ly = Ly
        self.alpha = alpha
        self.nx = nx
        self.ny = ny
        self.T = T
        self.nt = nt

        # Generate grid
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dt = T / nt
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.rx = alpha * self.dt / (2 * self.dx**2)
        self.ry = alpha * self.dt / (2 * self.dy**2)

        print(f"2D ADI Heat Equation Setup")
        print(f"  Grid: {nx} x {ny}")
        print(f"  r_x = {self.rx:.4f}, r_y = {self.ry:.4f}")

        # Build matrices
        self._build_matrices()

    def _build_matrices(self):
        """Build ADI tridiagonal matrices"""
        # x-direction (for each y)
        mx = self.nx - 2
        main_x = (1 + 2*self.rx) * np.ones(mx)
        off_x = -self.rx * np.ones(mx - 1)
        self.Ax = diags([off_x, main_x, off_x], [-1, 0, 1], format='csr')

        # y-direction (for each x)
        my = self.ny - 2
        main_y = (1 + 2*self.ry) * np.ones(my)
        off_y = -self.ry * np.ones(my - 1)
        self.Ay = diags([off_y, main_y, off_y], [-1, 0, 1], format='csr')

    def set_initial_condition(self, func):
        """Set initial condition"""
        self.u = func(self.X, self.Y)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, bc_value=0):
        """Dirichlet boundary conditions"""
        self.bc_value = bc_value

    def step(self):
        """Advance one time step (ADI two half-steps)"""
        u = self.u
        bc = self.bc_value

        # Intermediate solution array
        u_half = np.zeros_like(u)
        u_new = np.zeros_like(u)

        # Half-step 1: x-implicit, y-explicit
        for j in range(1, self.ny - 1):
            # y-explicit part (right-hand side)
            b = u[j, 1:-1] + self.ry * (u[j+1, 1:-1] - 2*u[j, 1:-1] + u[j-1, 1:-1])
            # Boundary conditions
            b[0] += self.rx * bc
            b[-1] += self.rx * bc
            # x-implicit solve
            u_half[j, 1:-1] = spsolve(self.Ax, b)

        # Apply boundary conditions
        u_half[0, :] = bc
        u_half[-1, :] = bc
        u_half[:, 0] = bc
        u_half[:, -1] = bc

        # Half-step 2: y-implicit, x-explicit
        for i in range(1, self.nx - 1):
            # x-explicit part (right-hand side)
            b = u_half[1:-1, i] + self.rx * (u_half[1:-1, i+1] - 2*u_half[1:-1, i] + u_half[1:-1, i-1])
            # Boundary conditions
            b[0] += self.ry * bc
            b[-1] += self.ry * bc
            # y-implicit solve
            u_new[1:-1, i] = spsolve(self.Ay, b)

        # Apply boundary conditions
        u_new[0, :] = bc
        u_new[-1, :] = bc
        u_new[:, 0] = bc
        u_new[:, -1] = bc

        self.u = u_new

    def solve(self, save_interval=None):
        """Solve over entire time domain"""
        if save_interval is None:
            save_interval = max(1, self.nt // 50)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), history


def compare_2d_methods():
    """Compare 2D FTCS vs ADI"""
    Lx = Ly = 1.0
    alpha = 0.01
    nx = ny = 41
    T = 0.3

    # Initial condition
    def ic(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    # FTCS
    ftcs = HeatEquation2D_FTCS(Lx, Ly, alpha, nx, ny, T, safety=0.2)
    ftcs.set_initial_condition(ic)
    ftcs.set_boundary_conditions(0)
    times_ftcs, history_ftcs = ftcs.solve()

    # ADI
    adi = HeatEquation2D_ADI(Lx, Ly, alpha, nx, ny, T, nt=30)
    adi.set_initial_condition(ic)
    adi.set_boundary_conditions(0)
    times_adi, history_adi = adi.solve()

    # Analytical solution: u = sin(πx)sin(πy)exp(-2απ²t)
    u_exact = np.sin(np.pi * adi.X) * np.sin(np.pi * adi.Y) * \
              np.exp(-2 * alpha * np.pi**2 * T)

    print(f"\nComparison Results (t = {T}):")
    print(f"  FTCS time steps: {ftcs.nt}")
    print(f"  ADI time steps: {adi.nt}")
    print(f"  FTCS maximum error: {np.max(np.abs(history_ftcs[-1] - u_exact)):.2e}")
    print(f"  ADI maximum error: {np.max(np.abs(history_adi[-1] - u_exact)):.2e}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    c1 = axes[0].contourf(adi.X, adi.Y, history_ftcs[-1], levels=30, cmap='hot')
    plt.colorbar(c1, ax=axes[0])
    axes[0].set_title(f'FTCS (dt={ftcs.dt:.5f})')
    axes[0].set_aspect('equal')

    c2 = axes[1].contourf(adi.X, adi.Y, history_adi[-1], levels=30, cmap='hot')
    plt.colorbar(c2, ax=axes[1])
    axes[1].set_title(f'ADI (dt={adi.dt:.4f})')
    axes[1].set_aspect('equal')

    c3 = axes[2].contourf(adi.X, adi.Y, u_exact, levels=30, cmap='hot')
    plt.colorbar(c3, ax=axes[2])
    axes[2].set_title('Analytical')
    axes[2].set_aspect('equal')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('heat_2d_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_2d_methods()
```

---

## 6. Handling Various Boundary Conditions

### 6.1 Neumann Boundary Conditions

```python
class HeatEquation1D_Neumann:
    """
    1D Heat Equation with Neumann Boundary Conditions

    ∂u/∂x|_{x=0} = flux_left
    ∂u/∂x|_{x=L} = flux_right
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, safety=0.4):
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = safety * self.dx**2 / alpha
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.r = alpha * self.dt / self.dx**2

        print(f"Neumann BC Heat Equation: r = {self.r:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)

    def set_boundary_conditions(self, flux_left=0, flux_right=0):
        """Set Neumann boundary conditions"""
        self.flux_left = flux_left
        self.flux_right = flux_right

    def step(self):
        """Advance one time step (with Neumann BC)"""
        u_new = self.u.copy()

        # Interior points
        u_new[1:-1] = self.u[1:-1] + self.r * (
            self.u[2:] - 2*self.u[1:-1] + self.u[:-2]
        )

        # Left Neumann BC: ∂u/∂x = flux_left
        # Ghost node method: u[-1] = u[1] - 2*dx*flux_left
        # u_new[0] = u[0] + r*(u[1] - 2*u[0] + u[-1])
        #          = u[0] + r*(u[1] - 2*u[0] + u[1] - 2*dx*flux_left)
        #          = u[0] + r*(2*u[1] - 2*u[0] - 2*dx*flux_left)
        u_new[0] = self.u[0] + self.r * (
            2*self.u[1] - 2*self.u[0] - 2*self.dx*self.flux_left
        )

        # Right Neumann BC: ∂u/∂x = flux_right
        u_new[-1] = self.u[-1] + self.r * (
            2*self.u[-2] - 2*self.u[-1] + 2*self.dx*self.flux_right
        )

        self.u = u_new

    def solve(self):
        history = [self.u.copy()]
        times = [0]

        save_interval = max(1, self.nt // 100)

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_neumann_bc():
    """Neumann boundary condition demo: insulated ends"""
    solver = HeatEquation1D_Neumann(L=1.0, alpha=0.01, nx=51, T=5.0)

    # Initial condition: left half hot, right half cold
    solver.set_initial_condition(lambda x: np.where(x < 0.5, 1.0, 0.0))

    # Insulated ends (flux = 0)
    solver.set_boundary_conditions(flux_left=0, flux_right=0)

    times, history = solver.solve()

    # Check energy conservation
    energies = [np.trapz(h, solver.x) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Temperature distribution
    ax1 = axes[0]
    for i in range(0, len(times), len(times)//5):
        ax1.plot(solver.x, history[i], label=f't = {times[i]:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('Insulated Boundary Conditions (∂u/∂x = 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Total energy
    ax2 = axes[1]
    ax2.plot(times, energies, 'b-', linewidth=2)
    ax2.axhline(y=energies[0], color='r', linestyle='--', label='Initial energy')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Energy (∫u dx)')
    ax2.set_title('Energy Conservation Check')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heat_neumann.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Initial energy: {energies[0]:.6f}")
    print(f"Final energy: {energies[-1]:.6f}")
    print(f"Energy change: {(energies[-1] - energies[0]) / energies[0] * 100:.4f}%")

# demo_neumann_bc()
```

### 6.2 Robin Boundary Conditions

```python
def demo_robin_bc():
    """Robin boundary condition: convective heat transfer"""
    L = 1.0
    alpha = 0.01
    nx = 51
    T = 2.0

    dx = L / (nx - 1)
    dt = 0.4 * dx**2 / alpha
    nt = int(np.ceil(T / dt))
    dt = T / nt
    r = alpha * dt / dx**2

    x = np.linspace(0, L, nx)

    # Initial condition: uniform temperature
    u = np.ones(nx)

    # Robin BC parameters
    # -k·∂u/∂x = h·(u - T_inf) at x = 0
    # where k=thermal conductivity, h=heat transfer coefficient, T_inf=ambient temperature
    k = 1.0
    h = 10.0  # Heat transfer coefficient
    T_inf = 0.0  # Ambient temperature
    Bi = h * dx / k  # Biot number

    history = [u.copy()]
    times = [0]

    for n in range(nt):
        u_new = u.copy()

        # Interior points
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])

        # Left Robin BC: h(u - T_inf) + k·∂u/∂x = 0
        # (u[1] - u[0])/dx = (h/k)(u[0] - T_inf)
        # Ghost node: u[-1] = u[1] - 2*dx*(h/k)*(u[0] - T_inf)
        u_new[0] = u[0] + r * (2*u[1] - 2*u[0] - 2*dx*(h/k)*(u[0] - T_inf))

        # Right: Dirichlet (fixed temperature)
        u_new[-1] = 1.0

        u = u_new

        if (n + 1) % (nt // 50) == 0:
            history.append(u.copy())
            times.append((n + 1) * dt)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(0, len(times), len(times)//5):
        ax.plot(x, history[i], label=f't = {times[i]:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'Robin Boundary Condition (Convective Heat Transfer)\nBiot Number = {Bi:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heat_robin.png', dpi=150, bbox_inches='tight')
    plt.show()

# demo_robin_bc()
```

---

## 7. Summary

### Scheme Comparison Table

| Scheme | Accuracy | Stability | Computational Cost | Features |
|------|--------|--------|-----------|------|
| FTCS | O(Δt, Δx²) | Conditional (r≤0.5) | Low | Simple, explicit |
| BTCS | O(Δt, Δx²) | Unconditional | Medium | Implicit, matrix solve |
| Crank-Nicolson | O(Δt², Δx²) | Unconditional | Medium | 2nd order accuracy |
| ADI (2D) | O(Δt², Δx²) | Unconditional | Medium | Efficient for 2D |

### CFL Conditions

```
1D FTCS: r = α·Δt/Δx² ≤ 0.5
2D FTCS: r_x + r_y ≤ 0.5
```

### Next Steps

1. **Chapter 10**: Wave Equation - hyperbolic PDE
2. **Chapter 11**: Laplace/Poisson - elliptic PDE
3. **Chapter 12**: Advection Equation - first-order hyperbolic PDE

---

## Exercises

### Exercise 1: FTCS Stability Experiment
Run FTCS with r = 0.3, 0.5, 0.6 and observe stable/unstable behavior.

### Exercise 2: Verify Convergence Order
Numerically verify that Crank-Nicolson has 2nd order temporal accuracy.

### Exercise 3: Non-Homogeneous Boundary Conditions
Find the steady-state solution when u(0,t) = 0 and u(L,t) = 100.

### Exercise 4: 2D Heat Equation
Solve the 2D heat equation with a rectangular hot spot initial condition instead of Gaussian.

---

## References

1. **Textbook**: LeVeque, "Finite Difference Methods for Ordinary and Partial Differential Equations"
2. **Python**: scipy.sparse, numpy
3. **Visualization**: matplotlib.animation

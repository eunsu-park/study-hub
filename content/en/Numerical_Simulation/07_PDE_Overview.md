# 07. Partial Differential Equations Overview

## Learning Objectives
- Understand the basic concepts and classification of partial differential equations (PDEs)
- Grasp the characteristics of parabolic, hyperbolic, and elliptic PDEs
- Understand the role of boundary conditions and initial conditions
- Learn the concept of well-posed problems

---

**Why This Lesson Matters:** While ODEs describe how quantities change in time alone, most physical phenomena -- heat conduction, wave propagation, fluid flow, electromagnetic fields -- involve variation in both space and time. PDEs are the mathematical language for these multi-dimensional phenomena. Understanding PDE classification is essential because each type (elliptic, parabolic, hyperbolic) demands fundamentally different numerical strategies and boundary condition treatments.

## 1. What is a Partial Differential Equation?

### 1.1 Definition

A Partial Differential Equation (PDE) is an equation that contains partial derivatives with respect to multiple independent variables.

```
General form of a second-order PDE:
A·∂²u/∂x² + B·∂²u/∂x∂y + C·∂²u/∂y² + D·∂u/∂x + E·∂u/∂y + F·u = G
```

Here, A, B, C, D, E, F, G can be functions of x and y.

### 1.2 ODE vs PDE Comparison

| Property | ODE | PDE |
|------|-----|-----|
| Independent Variables | 1 (usually t) | 2 or more (usually x, y, z, t) |
| Type of Derivative | Ordinary derivative | Partial derivative |
| Form of Solution | Function y(t) | Function u(x, y, ...) |
| Boundary Conditions | Initial conditions | Boundary conditions + Initial conditions |
| Solution Difficulty | Relatively easy | Complex |

### 1.3 Physical Application Examples

```python
"""
Major physical phenomena and corresponding PDEs
"""

# Heat Conduction
# ∂u/∂t = α · ∂²u/∂x²
# Temperature distribution change over time

# Wave Propagation
# ∂²u/∂t² = c² · ∂²u/∂x²
# Propagation of sound, light, vibration

# Steady-State Heat Distribution
# ∂²u/∂x² + ∂²u/∂y² = 0
# Temperature distribution without time change

# Advection
# ∂u/∂t + v · ∂u/∂x = 0
# Transport of matter

# Diffusion
# ∂u/∂t = D · ∇²u
# Concentration diffusion phenomenon
```

---

## 2. PDE Classification

### 2.1 Classification of Second-Order Linear PDEs

The classification of second-order linear PDEs parallels the classification of conic sections ($Ax^2 + Bxy + Cy^2 = \ldots$) -- hence the names elliptic, parabolic, and hyperbolic. The discriminant $\Delta = B^2 - 4AC$ determines the nature of the characteristic curves, which in turn dictate how information propagates through the domain and what types of boundary/initial conditions are needed.

General form of second-order linear PDE:
```
A·∂²u/∂x² + B·∂²u/∂x∂y + C·∂²u/∂y² + (lower order terms) = 0
```

**Classification by discriminant Δ = B² - 4AC**:

| Classification | Condition | Representative Equation | Physical Phenomenon |
|------|------|-------------|-----------|
| **Elliptic** | Δ < 0 | Laplace, Poisson | Steady-state |
| **Parabolic** | Δ = 0 | Heat equation | Diffusion |
| **Hyperbolic** | Δ > 0 | Wave equation | Wave propagation |

```python
import numpy as np

def classify_pde(A, B, C):
    """
    Classify second-order linear PDE

    Parameters:
    -----------
    A : float - coefficient of ∂²u/∂x²
    B : float - coefficient of ∂²u/∂x∂y
    C : float - coefficient of ∂²u/∂y²

    Returns:
    --------
    str : PDE classification
    """
    delta = B**2 - 4*A*C

    if delta < 0:
        return "Elliptic"
    elif delta == 0:
        return "Parabolic"
    else:
        return "Hyperbolic"

# Examples
print("Laplace equation (A=1, B=0, C=1):", classify_pde(1, 0, 1))
print("Heat equation (A=1, B=0, C=0):", classify_pde(1, 0, 0))
print("Wave equation (A=1, B=0, C=-1):", classify_pde(1, 0, -1))
```

### 2.2 Canonical Forms

#### Elliptic Canonical Form (Laplace equation)
```
∂²u/∂x² + ∂²u/∂y² = 0
```

#### Parabolic Canonical Form (Heat equation)
```
∂u/∂t = α · ∂²u/∂x²
```

#### Hyperbolic Canonical Form (Wave equation)
```
∂²u/∂t² = c² · ∂²u/∂x²
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pde_types():
    """Visualize characteristics by PDE type"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)

    # Elliptic: Steady-state - time independent
    # u = x(1-x) (solution satisfying boundary conditions)
    U_elliptic = X * (1 - X)
    ax1 = axes[0]
    c1 = ax1.contourf(X, T, U_elliptic, levels=20, cmap='coolwarm')
    ax1.set_title('Elliptic\nSteady-state - Time Independent', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y (or t)')
    plt.colorbar(c1, ax=ax1)

    # Parabolic: Diffusion - smoothing over time
    # Initial triangular wave becomes flat over time
    U_parabolic = np.exp(-np.pi**2 * T * 0.1) * np.sin(np.pi * X)
    ax2 = axes[1]
    c2 = ax2.contourf(X, T, U_parabolic, levels=20, cmap='coolwarm')
    ax2.set_title('Parabolic\nDiffusion - Smoothing Over Time', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    plt.colorbar(c2, ax=ax2)

    # Hyperbolic: Wave - oscillatory propagation
    c_speed = 1.0
    U_hyperbolic = np.sin(2*np.pi*X) * np.cos(2*np.pi*c_speed*T)
    ax3 = axes[2]
    c3 = ax3.contourf(X, T, U_hyperbolic, levels=20, cmap='coolwarm')
    ax3.set_title('Hyperbolic\nWave - Oscillatory Propagation', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    plt.colorbar(c3, ax=ax3)

    plt.tight_layout()
    plt.savefig('pde_types.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_pde_types()
```

---

## 3. Boundary Conditions

### 3.1 Types of Boundary Conditions

Appropriate boundary conditions are required to solve PDEs. Think of boundary conditions as the "constraints" that select a unique solution from the infinite family of functions satisfying the PDE. Choosing the wrong type or number of boundary conditions leads to an ill-posed problem: no solution, non-unique solutions, or solutions that blow up from tiny perturbations.

```python
"""
Three main types of boundary conditions
"""
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_boundary_conditions():
    """Visualize boundary conditions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(0, 1, 100)

    # 1. Dirichlet Boundary Condition (Dirichlet BC)
    # u(0) = a, u(L) = b - specify function value at boundary
    ax1 = axes[0]
    u_dirichlet = 0 + (1 - 0) * x  # Linear interpolation example
    ax1.plot(x, u_dirichlet, 'b-', linewidth=2)
    ax1.scatter([0, 1], [0, 1], color='red', s=100, zorder=5, label='Boundary values')
    ax1.set_title('Dirichlet Boundary Condition\nu(0) = 0, u(L) = 1', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Neumann Boundary Condition (Neumann BC)
    # du/dx|₀ = a, du/dx|_L = b - specify derivative (slope) at boundary
    ax2 = axes[1]
    # du/dx = 0 at both ends (insulated condition)
    u_neumann = np.cos(np.pi * x)  # Zero slope at both ends
    ax2.plot(x, u_neumann, 'b-', linewidth=2)
    ax2.annotate('', xy=(0.05, u_neumann[5]), xytext=(0, u_neumann[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.annotate('', xy=(0.95, u_neumann[-6]), xytext=(1, u_neumann[-1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.set_title('Neumann Boundary Condition\n∂u/∂x|₀ = 0, ∂u/∂x|_L = 0', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.grid(True, alpha=0.3)

    # 3. Robin Boundary Condition (Robin/Mixed BC)
    # a·u + b·du/dx = c - linear combination of function value and derivative
    ax3 = axes[2]
    u_robin = np.exp(-x) * np.cos(2*np.pi*x)
    ax3.plot(x, u_robin, 'b-', linewidth=2)
    ax3.scatter([0], [u_robin[0]], color='green', s=100, zorder=5)
    ax3.set_title('Robin Boundary Condition\nα·u + β·∂u/∂n = γ', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('u(x)')
    ax3.grid(True, alpha=0.3)
    ax3.annotate('Mixed condition', xy=(0, u_robin[0]), xytext=(0.2, 0.8),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('boundary_conditions.png', dpi=150, bbox_inches='tight')
    plt.show()

# demonstrate_boundary_conditions()
```

### 3.2 Boundary Conditions in Detail

#### Dirichlet Boundary Condition (Dirichlet BC)
- **Definition**: Specify the function value at the boundary
- **Formula**: u(boundary) = g
- **Physical meaning**: Fixed temperature, fixed displacement

```python
def apply_dirichlet_bc(u, left_value, right_value):
    """Apply Dirichlet boundary condition"""
    u[0] = left_value    # Left boundary
    u[-1] = right_value  # Right boundary
    return u

# Example: Fix temperatures at both ends of a rod
u = np.zeros(100)
u = apply_dirichlet_bc(u, left_value=100.0, right_value=0.0)
print(f"Left boundary: {u[0]}°C, Right boundary: {u[-1]}°C")
```

#### Neumann Boundary Condition (Neumann BC)
- **Definition**: Specify the normal derivative at the boundary
- **Formula**: ∂u/∂n|boundary = h
- **Physical meaning**: Heat flux specification, insulated condition (h=0)

```python
def apply_neumann_bc(u, dx, left_flux, right_flux):
    """
    Apply Neumann boundary condition (1st order accuracy)

    Parameters:
    -----------
    u : array - solution array
    dx : float - grid spacing
    left_flux : float - left boundary flux (∂u/∂x)
    right_flux : float - right boundary flux (∂u/∂x)
    """
    # Left: ∂u/∂x|₀ = left_flux
    # (u[1] - u[0])/dx = left_flux
    u[0] = u[1] - dx * left_flux

    # Right: ∂u/∂x|_L = right_flux
    # (u[-1] - u[-2])/dx = right_flux
    u[-1] = u[-2] + dx * right_flux

    return u

# Example: Insulated boundary condition (heat flux = 0)
u = np.linspace(100, 0, 100)
dx = 0.01
u = apply_neumann_bc(u, dx, left_flux=0.0, right_flux=0.0)
print(f"Insulated boundary applied")
```

#### Robin Boundary Condition (Robin/Mixed BC)
- **Definition**: Linear combination of function value and derivative
- **Formula**: α·u + β·∂u/∂n = γ
- **Physical meaning**: Convective heat transfer

```python
def apply_robin_bc(u, dx, alpha, beta, gamma):
    """
    Apply Robin boundary condition (left boundary)
    α·u + β·∂u/∂x = γ
    """
    # α·u[0] + β·(u[1] - u[0])/dx = γ
    # (α - β/dx)·u[0] + (β/dx)·u[1] = γ
    # u[0] = (γ - (β/dx)·u[1]) / (α - β/dx)

    if abs(alpha - beta/dx) > 1e-10:
        u[0] = (gamma - (beta/dx) * u[1]) / (alpha - beta/dx)

    return u

# Example: Convective heat transfer
# h·(u - T_inf) + k·∂u/∂x = 0
# where h is heat transfer coefficient, k is thermal conductivity, T_inf is ambient temperature
```

---

## 4. Initial Conditions

### 4.1 Role of Initial Conditions

For time-dependent PDEs (parabolic, hyperbolic), the initial state must be specified.

```python
def demonstrate_initial_conditions():
    """Various initial condition examples"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.linspace(0, 1, 200)
    L = 1.0

    # 1. Sine function initial condition
    u1 = np.sin(np.pi * x / L)
    axes[0, 0].plot(x, u1, 'b-', linewidth=2)
    axes[0, 0].set_title('Sine Initial Condition\nu(x,0) = sin(πx/L)', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.2, 1.2)

    # 2. Gaussian pulse initial condition
    x0 = 0.5  # center
    sigma = 0.1  # width
    u2 = np.exp(-(x - x0)**2 / (2 * sigma**2))
    axes[0, 1].plot(x, u2, 'b-', linewidth=2)
    axes[0, 1].set_title('Gaussian Pulse\nu(x,0) = exp(-(x-x₀)²/2σ²)', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.2, 1.2)

    # 3. Step function (discontinuous)
    u3 = np.where(x < 0.5, 1.0, 0.0)
    axes[1, 0].plot(x, u3, 'b-', linewidth=2)
    axes[1, 0].set_title('Step Function\nu(x,0) = H(0.5-x)', fontsize=12)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.2, 1.2)

    # 4. Triangular wave
    u4 = np.where(x < 0.5, 2*x, 2*(1-x))
    axes[1, 1].plot(x, u4, 'b-', linewidth=2)
    axes[1, 1].set_title('Triangular Wave\nu(x,0) = 2min(x, 1-x)', fontsize=12)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig('initial_conditions.png', dpi=150, bbox_inches='tight')
    plt.show()

# demonstrate_initial_conditions()
```

### 4.2 Initial Conditions for Parabolic vs Hyperbolic PDEs

| PDE Type | Required Initial Conditions | Example |
|----------|-----------------|------|
| Parabolic (Heat equation) | u(x, 0) | Initial temperature distribution |
| Hyperbolic (Wave equation) | u(x, 0) and ∂u/∂t(x, 0) | Initial displacement + initial velocity |

```python
class PDEProblem:
    """PDE problem definition class"""

    def __init__(self, pde_type, domain, nx, nt=None):
        """
        Parameters:
        -----------
        pde_type : str - 'parabolic', 'hyperbolic', 'elliptic'
        domain : tuple - (x_min, x_max) or ((x_min, x_max), (y_min, y_max))
        nx : int - number of grid points in x direction
        nt : int - number of time steps (for time-dependent problems)
        """
        self.pde_type = pde_type
        self.domain = domain
        self.nx = nx
        self.nt = nt

        # Create grid
        self.x = np.linspace(domain[0], domain[1], nx)
        self.dx = self.x[1] - self.x[0]

        # Store initial and boundary conditions
        self.initial_condition = None
        self.initial_velocity = None  # For hyperbolic type
        self.bc_left = {'type': 'dirichlet', 'value': 0}
        self.bc_right = {'type': 'dirichlet', 'value': 0}

    def set_initial_condition(self, func):
        """Set initial condition"""
        self.initial_condition = func(self.x)
        return self

    def set_initial_velocity(self, func):
        """Set initial velocity (for wave equation)"""
        if self.pde_type != 'hyperbolic':
            print("Warning: Initial velocity is only needed for hyperbolic PDEs.")
        self.initial_velocity = func(self.x)
        return self

    def set_boundary_condition(self, side, bc_type, value=0, flux=0, alpha=1, beta=0, gamma=0):
        """
        Set boundary condition

        Parameters:
        -----------
        side : str - 'left' or 'right'
        bc_type : str - 'dirichlet', 'neumann', 'robin'
        """
        bc = {'type': bc_type}

        if bc_type == 'dirichlet':
            bc['value'] = value
        elif bc_type == 'neumann':
            bc['flux'] = flux
        elif bc_type == 'robin':
            bc['alpha'] = alpha
            bc['beta'] = beta
            bc['gamma'] = gamma

        if side == 'left':
            self.bc_left = bc
        else:
            self.bc_right = bc

        return self

    def summary(self):
        """Print problem summary"""
        print(f"\n{'='*50}")
        print(f"PDE Problem Summary")
        print(f"{'='*50}")
        print(f"Type: {self.pde_type}")
        print(f"Domain: [{self.domain[0]}, {self.domain[1]}]")
        print(f"Grid points: {self.nx}")
        print(f"Grid spacing (dx): {self.dx:.6f}")
        print(f"\nLeft boundary condition: {self.bc_left}")
        print(f"Right boundary condition: {self.bc_right}")

        if self.initial_condition is not None:
            print(f"\nInitial condition: Set")
        if self.initial_velocity is not None:
            print(f"Initial velocity: Set")
        print(f"{'='*50}\n")

# Usage example
problem = PDEProblem('parabolic', (0, 1), 101)
problem.set_initial_condition(lambda x: np.sin(np.pi * x))
problem.set_boundary_condition('left', 'dirichlet', value=0)
problem.set_boundary_condition('right', 'dirichlet', value=0)
problem.summary()
```

---

## 5. Well-Posed Problems

### 5.1 Hadamard's Well-Posedness Criteria

Well-posedness is not just an abstract mathematical requirement -- it directly affects whether a numerical simulation will produce meaningful results. If a problem is ill-posed, numerical errors (which are always present) will be amplified catastrophically, and no amount of grid refinement can fix it. Verifying well-posedness before writing a single line of code is therefore essential.

For a PDE problem to be "well-posed", it must satisfy three conditions:

1. **Existence**: A solution must exist
2. **Uniqueness**: The solution must be unique
3. **Stability (Continuous Dependence)**: The solution must depend continuously on initial/boundary conditions

```python
def demonstrate_well_posedness():
    """Demonstrate well-posedness"""

    print("="*60)
    print("Well-Posed Problem Examples")
    print("="*60)

    # Heat equation: Well-posed
    print("\n[1] Heat Equation (Well-Posed)")
    print("    ∂u/∂t = α·∂²u/∂x², 0 < x < L, t > 0")
    print("    Boundary: u(0,t) = u(L,t) = 0")
    print("    Initial: u(x,0) = f(x)")
    print("    → Existence: O, Uniqueness: O, Stability: O")

    # Backward heat equation: Ill-posed
    print("\n[2] Backward Heat Equation (Ill-Posed)")
    print("    ∂u/∂t = -α·∂²u/∂x² (time reversed)")
    print("    → Small errors in initial conditions grow exponentially")
    print("    → Stability condition violated!")

    # Laplace equation + proper boundary conditions: Well-Posed
    print("\n[3] Laplace Equation (Well-Posed with Dirichlet BC)")
    print("    ∂²u/∂x² + ∂²u/∂y² = 0 in Ω")
    print("    Boundary: u = g on ∂Ω")
    print("    → Existence: O, Uniqueness: O, Stability: O")

    # Cauchy problem for Laplace: Ill-Posed
    print("\n[4] Laplace Cauchy Problem (Ill-Posed)")
    print("    ∂²u/∂x² + ∂²u/∂y² = 0")
    print("    u(x,0) = 0, ∂u/∂y(x,0) = (1/n)sin(nx)")
    print("    → Solution: u = (1/n²)sin(nx)sinh(ny)")
    print("    → As n→∞, initial condition→0 but solution explodes!")

demonstrate_well_posedness()
```

### 5.2 Appropriate Boundary Conditions for Each PDE Type

```python
def required_conditions_table():
    """Required conditions for each PDE type"""

    conditions = """
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │     PDE Type    │   Boundary Conditions │   Initial Conditions  │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │   Elliptic      │ On entire boundary:  │ Not required         │
    │                 │ Dirichlet/Neumann/   │                      │
    │                 │ Robin conditions     │                      │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │   Parabolic     │ On spatial boundary: │ At t=0:              │
    │                 │ Dirichlet/Neumann    │ u(x,0) = f(x)        │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │   Hyperbolic    │ On spatial boundary: │ At t=0:              │
    │                 │ Dirichlet/Neumann    │ u(x,0) = f(x)        │
    │                 │                      │ ∂u/∂t(x,0) = g(x)    │
    └─────────────────┴──────────────────────┴──────────────────────┘
    """
    print(conditions)

required_conditions_table()
```

### 5.3 Numerical Importance of Stability

```python
import numpy as np
import matplotlib.pyplot as plt

def stability_demonstration():
    """Demonstrate the importance of numerical stability"""

    # Parameters
    L = 1.0
    nx = 50
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    alpha = 0.01  # Thermal diffusivity

    # CFL stability condition: dt <= dx² / (2*alpha)
    # Physical intuition: diffusion spreads information by one grid cell in time dx²/(2α).
    # Exceeding this means the numerical scheme "outpaces" the physics, causing instability.
    dt_stable = 0.4 * dx**2 / (2 * alpha)  # Stable: safety factor of 0.4
    dt_unstable = 1.5 * dx**2 / (2 * alpha)  # Unstable: exceeds limit by 50%

    print(f"Grid spacing dx = {dx:.4f}")
    print(f"Thermal diffusivity α = {alpha}")
    print(f"Stability condition: dt ≤ {dx**2 / (2*alpha):.6f}")
    print(f"Stable dt = {dt_stable:.6f}")
    print(f"Unstable dt = {dt_unstable:.6f}")

    # Initial condition
    u0 = np.sin(np.pi * x)

    # FTCS (Forward Time Central Space) scheme
    def ftcs_step(u, alpha, dt, dx):
        u_new = u.copy()
        r = alpha * dt / dx**2
        for i in range(1, len(u)-1):
            u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        # Boundary conditions (Dirichlet)
        u_new[0] = 0
        u_new[-1] = 0
        return u_new

    # Simulation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stable case
    u_stable = u0.copy()
    ax1 = axes[0]
    ax1.plot(x, u_stable, 'b-', label='t=0', alpha=0.8)

    for step in range(100):
        u_stable = ftcs_step(u_stable, alpha, dt_stable, dx)
        if step in [10, 30, 60, 99]:
            ax1.plot(x, u_stable, label=f't={step*dt_stable:.4f}', alpha=0.8)

    ax1.set_title(f'Stable Case\ndt = {dt_stable:.6f} (CFL condition satisfied)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 1.5)

    # Unstable case
    u_unstable = u0.copy()
    ax2 = axes[1]
    ax2.plot(x, u_unstable, 'b-', label='t=0', alpha=0.8)

    for step in range(10):  # Explodes in just a few steps
        u_unstable = ftcs_step(u_unstable, alpha, dt_unstable, dx)
        if step in [1, 3, 5, 9]:
            u_clipped = np.clip(u_unstable, -10, 10)  # Clip for visualization
            ax2.plot(x, u_clipped, label=f't={step*dt_unstable:.4f}', alpha=0.8)

    ax2.set_title(f'Unstable Case\ndt = {dt_unstable:.6f} (CFL condition violated)', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-10, 10)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('stability_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nMaximum value after unstable simulation: {np.max(np.abs(u_unstable)):.2e}")
    print("→ Violating CFL condition causes solution to explode!")

# stability_demonstration()
```

---

## 6. Comprehensive Example: 1D Heat Equation Problem Definition

```python
import numpy as np
import matplotlib.pyplot as plt

class HeatEquation1D:
    """
    1D Heat equation problem definition and analytical solution comparison

    ∂u/∂t = α · ∂²u/∂x²

    Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet)
    Initial condition: u(x,0) = sin(πx/L)

    Analytical solution: u(x,t) = sin(πx/L) · exp(-α(π/L)²t)
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, nt=1000):
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

        # Create grid
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt + 1)

        # Check CFL condition
        self.r = alpha * self.dt / self.dx**2
        self.cfl_satisfied = self.r <= 0.5

        print(f"1D Heat Equation Problem Setup")
        print(f"  Domain: [0, {L}]")
        print(f"  Thermal diffusivity α = {alpha}")
        print(f"  Spatial grid: nx = {nx}, dx = {self.dx:.4f}")
        print(f"  Time grid: nt = {nt}, dt = {self.dt:.6f}")
        print(f"  CFL number: r = α·dt/dx² = {self.r:.4f}")
        print(f"  CFL condition satisfied: {self.cfl_satisfied}")

    def initial_condition(self, x):
        """Initial condition: u(x,0) = sin(πx/L)"""
        return np.sin(np.pi * x / self.L)

    def exact_solution(self, x, t):
        """Analytical solution: u(x,t) = sin(πx/L) · exp(-α(π/L)²t)"""
        return np.sin(np.pi * x / self.L) * np.exp(-self.alpha * (np.pi / self.L)**2 * t)

    def boundary_conditions(self):
        """Return boundary conditions"""
        return {
            'left': {'type': 'dirichlet', 'value': 0.0},
            'right': {'type': 'dirichlet', 'value': 0.0}
        }

    def plot_exact_solution(self):
        """Visualize analytical solution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Solution at various times
        times = [0, 0.1, 0.3, 0.5, 1.0]
        for t in times:
            u = self.exact_solution(self.x, t)
            ax1.plot(self.x, u, label=f't = {t}')

        ax1.set_title('1D Heat Equation Analytical Solution (Time Evolution)', fontsize=12)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x,t)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Spatiotemporal contour plot
        X, T = np.meshgrid(self.x, self.t)
        U = self.exact_solution(X, T)

        c = ax2.contourf(X, T, U, levels=20, cmap='coolwarm')
        plt.colorbar(c, ax=ax2, label='u(x,t)')
        ax2.set_title('1D Heat Equation Spatiotemporal Distribution', fontsize=12)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')

        plt.tight_layout()
        plt.savefig('heat_exact.png', dpi=150, bbox_inches='tight')
        plt.show()

# Define problem and check analytical solution
problem = HeatEquation1D(L=1.0, alpha=0.01, nx=51, T=1.0, nt=1000)
# problem.plot_exact_solution()
```

---

## 7. Summary

### Key Concepts Review

| Concept | Description |
|------|------|
| PDE | Equation containing partial derivatives with respect to multiple independent variables |
| Elliptic | Δ < 0, steady-state (Laplace, Poisson) |
| Parabolic | Δ = 0, diffusion/heat conduction (Heat equation) |
| Hyperbolic | Δ > 0, wave propagation (Wave equation) |
| Dirichlet BC | Specify function value at boundary |
| Neumann BC | Specify derivative (flux) at boundary |
| Robin BC | Combination of function value and derivative |
| Well-posed | Existence, uniqueness, continuous dependence |

### Next Steps

1. **Chapter 08**: Finite Difference Basics - spatial/temporal discretization
2. **Chapter 09**: Heat Equation Numerical Methods - FTCS, BTCS, Crank-Nicolson
3. **Chapter 10**: Wave Equation Numerical Methods - CTCS, boundary condition handling
4. **Chapter 11**: Laplace/Poisson Equations - iterative methods
5. **Chapter 12**: Advection Equation - upwind method, numerical diffusion

---

## Exercises

### Exercise 1: PDE Classification
Classify the following PDEs (elliptic/parabolic/hyperbolic):

1. ∂²u/∂x² + 2∂²u/∂y² = 0
2. ∂u/∂t = 4∂²u/∂x²
3. ∂²u/∂t² = 9∂²u/∂x²
4. ∂²u/∂x² - ∂²u/∂y² = 0

### Exercise 2: Setting Boundary Conditions
Choose the appropriate boundary condition type for the following physical situations in a rod heat conduction problem:

1. The left end is immersed in ice water (0°C)
2. The right end is perfectly insulated
3. Heat exchange with air occurs at the left end

### Exercise 3: Deriving Analytical Solution
For the 1D heat equation u_t = α·u_xx with boundary conditions u(0,t) = u(L,t) = 0 and initial condition u(x,0) = sin(2πx/L), derive the analytical solution.

### Exercise 4: Checking Well-Posedness
Determine whether the following problems are well-posed:

1. Laplace equation + Dirichlet boundary conditions
2. Heat equation + initial condition + Dirichlet boundary conditions
3. Heat equation (time reversed) + final condition

---

## References

1. **Textbooks**:
   - "Numerical Methods for Engineers" - Chapra & Canale
   - "Numerical Solution of Partial Differential Equations" - Morton & Mayers

2. **Online**:
   - MIT OCW 18.303: Linear Partial Differential Equations
   - Stanford CME 306: Numerical Solution of PDEs

3. **Software**:
   - NumPy, SciPy: Python numerical computing
   - FEniCS, FiPy: PDE-specific libraries

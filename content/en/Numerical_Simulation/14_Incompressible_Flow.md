# 14. Incompressible Flow

## Learning Objectives
- Understand stream function-vorticity formulation
- Implement the Lid-Driven Cavity problem
- Understand pressure-velocity coupling problem
- Learn SIMPLE algorithm basics
- Grasp the concept of staggered grid

**Why This Lesson Matters:** Incompressible flow is the most common regime in everyday fluid mechanics -- water in pipes, air in buildings (at low speeds), blood in vessels. The incompressible Navier-Stokes equations present a unique numerical challenge: pressure has no evolution equation of its own. Instead, it acts as a Lagrange multiplier that enforces the divergence-free constraint ($\nabla \cdot \mathbf{u} = 0$). This pressure-velocity coupling is the central difficulty of incompressible CFD and has inspired ingenious algorithms like SIMPLE, fractional-step, and stream function-vorticity methods.

---

## 1. Incompressible Navier-Stokes Equations

### 1.1 Primitive Variable Formulation

The incompressible Navier-Stokes equations in primitive variables (velocity $\mathbf{u}$ and pressure $p$) consist of a divergence-free constraint and momentum equations:

```
Incompressible NS equations (primitive variables: u, v, p):

Continuity equation:
∂u/∂x + ∂v/∂y = 0

Momentum equations:
∂u/∂t + u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
∂v/∂t + u∂v/∂x + v∂v/∂y = -1/ρ ∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)

Problem:
- 3 equations, 3 unknowns (u, v, p)
- No independent equation for pressure
- Pressure-velocity coupling problem
```

### 1.2 Pressure Poisson Equation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

"""
Taking the divergence of the continuity equation:
∇²p = -ρ∇·(u·∇u)

For steady state:
∇²p = ρ[(∂u/∂x)² + 2(∂u/∂y)(∂v/∂x) + (∂v/∂y)²]

This equation can be used to determine pressure
"""

def pressure_poisson_concept():
    """Visualization of pressure Poisson equation concept"""

    print("=" * 60)
    print("Pressure-Velocity Coupling Problem")
    print("=" * 60)

    explanation = """
    Key problem in incompressible flow:

    1. Role of pressure:
       - Pressure adjusts to satisfy continuity equation (∇·u = 0)
       - Pressure waves propagate infinitely fast (incompressible)
       - No independent governing equation for pressure

    2. Solution methods:
       a) Stream function-vorticity formulation:
          - Automatically satisfies continuity equation
          - Applicable to 2D flows only

       b) Pressure Poisson equation:
          - Derive pressure equation from continuity
          - Projection/Fractional Step Method

       c) SIMPLE family algorithms:
          - Iterative pressure-velocity correction
          - Industry standard
    """
    print(explanation)

pressure_poisson_concept()
```

---

## 2. Stream Function-Vorticity Formulation

### 2.1 Definition

```
2D incompressible flow:

Stream function ψ:
u = ∂ψ/∂y,  v = -∂ψ/∂x
→ Automatically satisfies continuity: ∂u/∂x + ∂v/∂y = 0

Vorticity ω:
ω = ∂v/∂x - ∂u/∂y = -∇²ψ

Vorticity transport equation:
∂ω/∂t + u∂ω/∂x + v∂ω/∂y = ν∇²ω

Poisson equation:
∇²ψ = -ω

Advantages: Eliminates pressure term, automatically satisfies continuity
Disadvantages: Only for 2D, complex boundary conditions
```

```python
def stream_function_vorticity_derivation():
    """Derivation of stream function-vorticity formulation"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Stream function concept
    ax1 = axes[0]

    # Streamlines (iso-ψ lines)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Example: uniform flow + doublet
    U_inf = 1.0
    kappa = 1.0  # doublet strength
    r2 = X**2 + Y**2 + 0.01
    psi = U_inf * Y - kappa * Y / r2

    # Velocity field
    u = U_inf + kappa * (X**2 - Y**2) / r2**2 * 2 * kappa
    v = -2 * kappa * X * Y / r2**2 * 2 * kappa

    # Simplified velocity field
    u = np.gradient(psi, y, axis=0)
    v = -np.gradient(psi, x, axis=1)

    levels = np.linspace(-3, 3, 21)
    cs = ax1.contour(X, Y, psi, levels=levels, colors='blue', linewidths=0.5)
    ax1.streamplot(X, Y, u, v, color='red', density=1, linewidth=0.5)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(r'Stream function ψ (iso-ψ lines = streamlines)')
    ax1.set_aspect('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    # (2) Vorticity concept
    ax2 = axes[1]

    # Vortex example
    Gamma = 2 * np.pi  # circulation
    r = np.sqrt(X**2 + Y**2) + 0.1
    theta = np.arctan2(Y, X)

    # Rankine vortex (core radius = 0.5)
    r_core = 0.5
    omega = np.where(r < r_core,
                    Gamma / (np.pi * r_core**2),  # solid body rotation
                    0)  # potential flow

    im = ax2.pcolormesh(X, Y, omega, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax2, label=r'$\omega$ [1/s]')

    # Velocity field (tangential)
    u_theta = np.where(r < r_core,
                      Gamma * r / (2 * np.pi * r_core**2),
                      Gamma / (2 * np.pi * r))
    u_vortex = -u_theta * np.sin(theta)
    v_vortex = u_theta * np.cos(theta)

    skip = 5
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_vortex[::skip, ::skip], v_vortex[::skip, ::skip],
              color='black', alpha=0.7)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(r'Vorticity ω (Rankine vortex)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig('stream_vorticity.png', dpi=150, bbox_inches='tight')
    plt.show()

# stream_function_vorticity_derivation()
```

---

## 3. Lid-Driven Cavity Problem

### 3.1 Problem Definition

```
Lid-Driven Cavity:
- Square cavity
- Top wall moves at constant velocity
- Other walls are stationary

Boundary conditions:
- Top (y=H): u = U_lid, v = 0
- Bottom (y=0): u = v = 0
- Left (x=0): u = v = 0
- Right (x=L): u = v = 0

Characteristics:
- Flow pattern changes with Reynolds number
- Low Re: single primary vortex
- High Re: corner vortices appear
- Standard problem for CFD code verification
```

```python
def lid_driven_cavity_stream_vorticity():
    """
    Lid-Driven Cavity simulation
    Stream function-vorticity formulation
    """

    # Grid setup
    N = 41  # number of grid points (odd number)
    L = 1.0  # cavity size
    h = L / (N - 1)

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    # Properties and settings
    Re = 100  # Reynolds number
    U_lid = 1.0
    nu = U_lid * L / Re

    # Time settings
    dt = 0.001
    n_steps = 10000

    # Initialize
    psi = np.zeros((N, N))  # stream function
    omega = np.zeros((N, N))  # vorticity

    # Check CFL
    CFL = U_lid * dt / h
    print(f"Re = {Re}, N = {N}, h = {h:.4f}")
    print(f"dt = {dt}, CFL = {CFL:.4f}")

    def apply_bc_psi(psi):
        """Stream function boundary condition: ψ = 0 on walls"""
        psi[0, :] = 0   # bottom
        psi[-1, :] = 0  # top
        psi[:, 0] = 0   # left
        psi[:, -1] = 0  # right
        return psi

    def apply_bc_omega(omega, psi, h, U_lid):
        """
        Vorticity boundary condition (Thom's formula):
        At wall: ω = -2(ψ_neighbor - ψ_wall)/h²
        """
        # Bottom (no-slip)
        omega[0, :] = -2 * psi[1, :] / h**2

        # Top (moving lid)
        omega[-1, :] = -2 * psi[-2, :] / h**2 - 2 * U_lid / h

        # Left
        omega[:, 0] = -2 * psi[:, 1] / h**2

        # Right
        omega[:, -1] = -2 * psi[:, -2] / h**2

        return omega

    def solve_poisson(psi, omega, h, n_iter=50, tol=1e-6):
        """
        Solve Poisson equation: ∇²ψ = -ω
        Gauss-Seidel iteration
        """
        for _ in range(n_iter):
            psi_old = psi.copy()

            for i in range(1, N-1):
                for j in range(1, N-1):
                    psi[i, j] = 0.25 * (psi[i+1, j] + psi[i-1, j] +
                                       psi[i, j+1] + psi[i, j-1] +
                                       h**2 * omega[i, j])

            # Boundary conditions
            psi = apply_bc_psi(psi)

            # Check convergence
            if np.max(np.abs(psi - psi_old)) < tol:
                break

        return psi

    def compute_velocity(psi, h):
        """Compute velocity from stream function"""
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)

        # Interior points (central difference)
        for i in range(1, N-1):
            for j in range(1, N-1):
                u[i, j] = (psi[i+1, j] - psi[i-1, j]) / (2 * h)
                v[i, j] = -(psi[i, j+1] - psi[i, j-1]) / (2 * h)

        return u, v

    def advect_diffuse_omega(omega, u, v, nu, h, dt):
        """Time advance for vorticity transport equation"""
        omega_new = omega.copy()

        for i in range(1, N-1):
            for j in range(1, N-1):
                # Convection term (upwind)
                if u[i, j] > 0:
                    domega_dx = (omega[i, j] - omega[i, j-1]) / h
                else:
                    domega_dx = (omega[i, j+1] - omega[i, j]) / h

                if v[i, j] > 0:
                    domega_dy = (omega[i, j] - omega[i-1, j]) / h
                else:
                    domega_dy = (omega[i+1, j] - omega[i, j]) / h

                convection = u[i, j] * domega_dx + v[i, j] * domega_dy

                # Diffusion term (central difference)
                diffusion = nu * ((omega[i+1, j] - 2*omega[i, j] + omega[i-1, j]) / h**2 +
                                 (omega[i, j+1] - 2*omega[i, j] + omega[i, j-1]) / h**2)

                omega_new[i, j] = omega[i, j] + dt * (-convection + diffusion)

        return omega_new

    # Time integration
    print("\nStarting simulation...")

    for n in range(n_steps):
        # 1. Solve Poisson equation
        psi = solve_poisson(psi, omega, h)

        # 2. Compute velocity
        u, v = compute_velocity(psi, h)

        # 3. Vorticity boundary conditions
        omega = apply_bc_omega(omega, psi, h, U_lid)

        # 4. Transport vorticity
        omega = advect_diffuse_omega(omega, u, v, nu, h, dt)

        # Progress report
        if n % 2000 == 0:
            print(f"Step {n}: max|ω| = {np.max(np.abs(omega)):.4f}, "
                  f"max|ψ| = {np.max(np.abs(psi)):.6f}")

    print("Simulation complete!")

    # Final velocity field
    u, v = compute_velocity(psi, h)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # (1) Streamlines
    ax1 = axes[0, 0]
    levels = np.linspace(psi.min(), psi.max(), 30)
    cs = ax1.contour(X, Y, psi, levels=levels, colors='blue', linewidths=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Streamlines, Re = {Re}')
    ax1.set_aspect('equal')

    # (2) Vorticity distribution
    ax2 = axes[0, 1]
    vmax = np.max(np.abs(omega)) * 0.8
    im = ax2.pcolormesh(X, Y, omega, cmap='RdBu_r', shading='auto',
                       vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax2, label=r'$\omega$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Vorticity distribution')
    ax2.set_aspect('equal')

    # (3) Velocity vectors
    ax3 = axes[1, 0]
    skip = 2
    speed = np.sqrt(u**2 + v**2)
    ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              speed[::skip, ::skip], cmap='jet', scale=20)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Velocity vector field')
    ax3.set_aspect('equal')

    # (4) Centerline velocity profile
    ax4 = axes[1, 1]

    # u distribution on vertical centerline (x = 0.5)
    j_center = N // 2
    u_centerline = u[:, j_center]

    # Ghia et al. (1982) reference values (Re=100)
    y_ghia = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                      0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 1.0000])
    u_ghia = np.array([0.0000, -0.0372, -0.0419, -0.0477, -0.0643, -0.1015,
                      -0.1566, -0.2109, -0.2058, -0.1364, 0.0033, 0.2315, 0.6872, 1.0000])

    ax4.plot(u_centerline, y, 'b-', linewidth=2, label='Present')
    ax4.plot(u_ghia, y_ghia, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax4.set_xlabel('u')
    ax4.set_ylabel('y')
    ax4.set_title('Vertical centerline velocity profile (x = 0.5)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lid_driven_cavity.png', dpi=150, bbox_inches='tight')
    plt.show()

    return psi, omega, u, v, X, Y

# psi, omega, u, v, X, Y = lid_driven_cavity_stream_vorticity()
```

### 3.2 Reynolds Number Effects

```python
def reynolds_effect_cavity():
    """Compare Lid-Driven Cavity at various Reynolds numbers"""

    reynolds_numbers = [100, 400, 1000]
    results = []

    N = 51
    L = 1.0
    h = L / (N - 1)

    for Re in reynolds_numbers:
        print(f"\n=== Re = {Re} ===")

        # Settings
        U_lid = 1.0
        nu = U_lid * L / Re
        dt = min(0.001, 0.25 * h**2 / nu)  # diffusion stability
        n_steps = int(20 / dt)  # sufficient time

        # Initialize
        psi = np.zeros((N, N))
        omega = np.zeros((N, N))

        # Simplified simulation (fast execution)
        for n in range(min(n_steps, 5000)):
            # Poisson
            for _ in range(20):
                psi_old = psi.copy()
                psi[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] +
                                         psi[1:-1, 2:] + psi[1:-1, :-2] +
                                         h**2 * omega[1:-1, 1:-1])
                psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0

            # Velocity
            u = np.zeros((N, N))
            v = np.zeros((N, N))
            u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*h)
            v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*h)

            # Vorticity boundary conditions
            omega[0, :] = -2 * psi[1, :] / h**2
            omega[-1, :] = -2 * psi[-2, :] / h**2 - 2 * U_lid / h
            omega[:, 0] = -2 * psi[:, 1] / h**2
            omega[:, -1] = -2 * psi[:, -2] / h**2

            # Vorticity advance (FTCS + upwind)
            omega_new = omega.copy()
            for i in range(1, N-1):
                for j in range(1, N-1):
                    conv_x = u[i,j] * (omega[i,j] - omega[i,j-1])/h if u[i,j]>0 else u[i,j] * (omega[i,j+1] - omega[i,j])/h
                    conv_y = v[i,j] * (omega[i,j] - omega[i-1,j])/h if v[i,j]>0 else v[i,j] * (omega[i+1,j] - omega[i,j])/h
                    diff = nu * ((omega[i+1,j] - 2*omega[i,j] + omega[i-1,j]) +
                                (omega[i,j+1] - 2*omega[i,j] + omega[i,j-1])) / h**2
                    omega_new[i,j] = omega[i,j] + dt * (-conv_x - conv_y + diff)
            omega = omega_new

        results.append((Re, psi.copy(), omega.copy(), u.copy(), v.copy()))
        print(f"  Complete: max|ψ| = {np.max(np.abs(psi)):.6f}")

    # Comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    for idx, (Re, psi, omega, u, v) in enumerate(results):
        # Streamlines
        ax = axes[0, idx]
        levels = np.linspace(psi.min(), psi.max(), 25)
        ax.contour(X, Y, psi, levels=levels, colors='blue', linewidths=0.5)
        ax.set_title(f'Re = {Re}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # Vorticity
        ax = axes[1, idx]
        vmax = min(5, np.max(np.abs(omega)))
        ax.pcolormesh(X, Y, omega, cmap='RdBu_r', shading='auto',
                     vmin=-vmax, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    axes[0, 0].set_ylabel('Streamlines')
    axes[1, 0].set_ylabel('Vorticity')

    plt.suptitle('Lid-Driven Cavity Flow Patterns vs Reynolds Number', fontsize=14)
    plt.tight_layout()
    plt.savefig('cavity_reynolds_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# reynolds_effect_cavity()
```

---

## 4. Staggered Grid

### 4.1 Problems with Collocated Grid

```
Collocated Grid:
- All variables (u, v, p) stored at same location
- Central difference for pressure: (p_{i+1} - p_{i-1})/(2Δx)
- Problem: checkerboard instability

Checkerboard phenomenon:
- Pressure oscillations have no effect on numerical solution
- Oscillations in form p(i) = p0 + (-1)^i * ε
- Oscillations cancel out in central differencing
```

```python
def checkerboard_problem():
    """Visualization of checkerboard instability"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) Collocated grid
    ax1 = axes[0]
    n = 6
    for i in range(n):
        for j in range(n):
            ax1.plot(i, j, 'ko', markersize=10)
            ax1.text(i, j+0.2, 'u,v,p', fontsize=6, ha='center')

    ax1.set_xlim(-0.5, n-0.5)
    ax1.set_ylim(-0.5, n)
    ax1.set_title('Collocated Grid')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # (2) Checkerboard pressure
    ax2 = axes[1]
    n = 8
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    P = (-1) ** (X + Y)  # checkerboard pattern

    im = ax2.pcolormesh(X-0.5, Y-0.5, P, cmap='RdBu_r', shading='auto')
    ax2.set_title('Checkerboard Pressure Distribution')
    ax2.set_aspect('equal')
    plt.colorbar(im, ax=ax2)

    # Pressure gradient (central difference = 0!)
    dpdx = np.zeros_like(P)
    for i in range(1, n-1):
        for j in range(1, n-1):
            dpdx[i, j] = (P[i, j+1] - P[i, j-1]) / 2  # always 0!

    ax2.set_xlabel('Central difference ∂p/∂x = 0 (incorrect!)')

    # (3) Staggered grid
    ax3 = axes[2]
    n = 5

    # Pressure points (cell center)
    for i in range(n):
        for j in range(n):
            ax3.plot(i+0.5, j+0.5, 'ro', markersize=10, label='p' if i==0 and j==0 else '')

    # u-velocity points (vertical faces)
    for i in range(n+1):
        for j in range(n):
            ax3.plot(i, j+0.5, 'b>', markersize=8, label='u' if i==0 and j==0 else '')

    # v-velocity points (horizontal faces)
    for i in range(n):
        for j in range(n+1):
            ax3.plot(i+0.5, j, 'g^', markersize=8, label='v' if i==0 and j==0 else '')

    # Grid lines
    for i in range(n+1):
        ax3.axvline(x=i, color='gray', linestyle='-', linewidth=0.5)
        ax3.axhline(y=i, color='gray', linestyle='-', linewidth=0.5)

    ax3.set_xlim(-0.3, n+0.3)
    ax3.set_ylim(-0.3, n+0.3)
    ax3.set_title('Staggered Grid')
    ax3.legend(loc='upper right')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('staggered_grid.png', dpi=150, bbox_inches='tight')
    plt.show()

# checkerboard_problem()
```

### 4.2 Staggered Grid Structure

```
Staggered Grid (MAC Grid):

     v(i,j+1)
        │
  ──────┼──────
  │     │     │
  │  p(i,j)   │
u(i,j)─┼──u(i+1,j)
  │     │     │
  │     │     │
  ──────┼──────
     v(i,j)

Storage locations:
- p: cell center (i, j)
- u: cell east face (i+1/2, j)
- v: cell north face (i, j+1/2)

Advantages:
- Prevents checkerboard
- Accurate mass conservation
- Natural pressure-velocity coupling
```

---

## 5. SIMPLE Algorithm

### 5.1 Concept

```
SIMPLE (Semi-Implicit Method for Pressure-Linked Equations):

Basic idea:
1. Use guessed pressure field p*
2. Compute guessed velocity field u*, v* (momentum equations)
3. Compute pressure correction p' (continuity equation)
4. Update pressure/velocity: p = p* + p', u = u* + u'
5. Iterate until convergence

Key equation:
∇²p' = ρ/Δt · ∇·u*

where p' = p - p* (pressure correction)
```

```python
def simple_algorithm_concept():
    """Explanation of SIMPLE algorithm concept"""

    print("=" * 60)
    print("SIMPLE Algorithm Flowchart")
    print("=" * 60)

    flowchart = """
    ┌─────────────────────────────────────────┐
    │      Initial guess: p*, u*, v*           │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Step 1: Solve momentum equations        │
    │  → Compute guessed velocity u*, v*       │
    │  (using pressure p*)                     │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Step 2: Solve pressure correction eq.   │
    │  ∇²p' = (ρ/Δt) ∇·u*                      │
    │  (divergence of u* = mass imbalance)     │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Step 3: Correct pressure/velocity       │
    │  p = p* + αp·p'                          │
    │  u = u* - (Δt/ρ)·∂p'/∂x                  │
    │  v = v* - (Δt/ρ)·∂p'/∂y                  │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Check convergence: |∇·u| < ε ?          │
    │                                          │
    │  No → p* = p, u* = u, v* = v (repeat)    │
    │  Yes → Exit                              │
    └─────────────────────────────────────────┘

    Under-relaxation factors:
    - αp ~ 0.3 (pressure)
    - αu ~ 0.7 (velocity)
    """
    print(flowchart)

simple_algorithm_concept()
```

### 5.2 SIMPLE Algorithm Implementation

```python
def simple_lid_driven_cavity():
    """
    Solve Lid-Driven Cavity with SIMPLE algorithm
    Using staggered grid
    """

    # Grid setup
    Nx = 32  # number of cells (x direction)
    Ny = 32  # number of cells (y direction)
    L = 1.0

    dx = L / Nx
    dy = L / Ny

    # Properties
    rho = 1.0
    mu = 0.01
    Re = rho * 1.0 * L / mu
    print(f"Reynolds number: Re = {Re}")

    # Under-relaxation
    alpha_u = 0.7
    alpha_p = 0.3

    # Initialize arrays (staggered grid)
    # u: (Ny, Nx+1) - vertical faces
    # v: (Ny+1, Nx) - horizontal faces
    # p: (Ny, Nx) - cell centers
    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))
    p = np.zeros((Ny, Nx))
    p_prime = np.zeros((Ny, Nx))

    # Boundary conditions
    U_lid = 1.0

    def apply_boundary_conditions():
        """Apply boundary conditions"""
        nonlocal u, v

        # Top (lid): u = U_lid
        u[-1, :] = 2 * U_lid - u[-2, :]  # linear extrapolation (U_lid at wall)

        # Bottom: u = 0
        u[0, :] = -u[1, :]  # linear extrapolation

        # Left/right: u = 0 (already 0)
        u[:, 0] = 0
        u[:, -1] = 0

        # v boundary conditions
        v[0, :] = 0    # bottom
        v[-1, :] = 0   # top
        v[:, 0] = -v[:, 1]   # left
        v[:, -1] = -v[:, -2]  # right

    def solve_momentum(u, v, p, mu, rho, dx, dy, dt):
        """Solve momentum equations (guessed velocity)"""
        u_star = u.copy()
        v_star = v.copy()

        # u-momentum
        for j in range(1, Ny - 1):
            for i in range(1, Nx):
                # Convection term (upwind)
                u_face = 0.5 * (u[j, i] + u[j, i-1]) if i > 0 else u[j, i]

                if u_face > 0:
                    dudx = (u[j, i] - u[j, i-1]) / dx
                else:
                    dudx = (u[j, i+1] - u[j, i]) / dx if i < Nx else (u[j, i] - u[j, i-1]) / dx

                v_face = 0.25 * (v[j, min(i, Nx-1)] + v[j+1, min(i, Nx-1)] +
                                v[j, max(i-1, 0)] + v[j+1, max(i-1, 0)])

                if v_face > 0:
                    dudy = (u[j, i] - u[j-1, i]) / dy
                else:
                    dudy = (u[j+1, i] - u[j, i]) / dy if j < Ny-1 else (u[j, i] - u[j-1, i]) / dy

                # Diffusion term
                d2udx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / dx**2 if 0 < i < Nx else 0
                d2udy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / dy**2

                # Pressure gradient
                dpdx = (p[j, min(i, Nx-1)] - p[j, max(i-1, 0)]) / dx

                # Time advance
                conv = u_face * dudx + v_face * dudy
                diff = mu / rho * (d2udx2 + d2udy2)
                u_star[j, i] = u[j, i] + dt * (-conv - dpdx / rho + diff)

        # v-momentum (similarly)
        for j in range(1, Ny):
            for i in range(1, Nx - 1):
                u_face = 0.25 * (u[min(j, Ny-1), i] + u[min(j, Ny-1), i+1] +
                                u[max(j-1, 0), i] + u[max(j-1, 0), i+1])

                if u_face > 0:
                    dvdx = (v[j, i] - v[j, i-1]) / dx
                else:
                    dvdx = (v[j, i+1] - v[j, i]) / dx if i < Nx-1 else (v[j, i] - v[j, i-1]) / dx

                v_face = 0.5 * (v[j, i] + v[j-1, i]) if j > 0 else v[j, i]

                if v_face > 0:
                    dvdy = (v[j, i] - v[j-1, i]) / dy
                else:
                    dvdy = (v[j+1, i] - v[j, i]) / dy if j < Ny else (v[j, i] - v[j-1, i]) / dy

                d2vdx2 = (v[j, i+1] - 2*v[j, i] + v[j, i-1]) / dx**2
                d2vdy2 = (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / dy**2 if 0 < j < Ny else 0

                dpdy = (p[min(j, Ny-1), i] - p[max(j-1, 0), i]) / dy

                conv = u_face * dvdx + v_face * dvdy
                diff = mu / rho * (d2vdx2 + d2vdy2)
                v_star[j, i] = v[j, i] + dt * (-conv - dpdy / rho + diff)

        return u_star, v_star

    def solve_pressure_correction(u_star, v_star, dx, dy, dt, rho, n_iter=50):
        """Solve pressure correction equation"""
        p_prime = np.zeros((Ny, Nx))

        for _ in range(n_iter):
            p_old = p_prime.copy()

            for j in range(Ny):
                for i in range(Nx):
                    # Mass imbalance (divergence)
                    div = ((u_star[j, i+1] - u_star[j, i]) / dx +
                          (v_star[j+1, i] - v_star[j, i]) / dy)

                    # Neighbor pressures
                    p_E = p_prime[j, i+1] if i < Nx-1 else 0
                    p_W = p_prime[j, i-1] if i > 0 else 0
                    p_N = p_prime[j+1, i] if j < Ny-1 else 0
                    p_S = p_prime[j-1, i] if j > 0 else 0

                    # Poisson equation coefficients
                    aE = 1/dx**2 if i < Nx-1 else 0
                    aW = 1/dx**2 if i > 0 else 0
                    aN = 1/dy**2 if j < Ny-1 else 0
                    aS = 1/dy**2 if j > 0 else 0
                    aP = aE + aW + aN + aS

                    if aP > 0:
                        p_prime[j, i] = (aE*p_E + aW*p_W + aN*p_N + aS*p_S -
                                        rho/dt * div) / aP

        return p_prime

    def correct_velocity(u_star, v_star, p_prime, dx, dy, dt, rho):
        """Velocity correction"""
        u_new = u_star.copy()
        v_new = v_star.copy()

        # u correction
        for j in range(Ny):
            for i in range(1, Nx):
                dpdx = (p_prime[j, i] - p_prime[j, i-1]) / dx
                u_new[j, i] = u_star[j, i] - dt / rho * dpdx

        # v correction
        for j in range(1, Ny):
            for i in range(Nx):
                dpdy = (p_prime[j, i] - p_prime[j-1, i]) / dy
                v_new[j, i] = v_star[j, i] - dt / rho * dpdy

        return u_new, v_new

    # Simulation
    dt = 0.001
    n_outer = 500

    print("\nStarting SIMPLE algorithm...")

    for n in range(n_outer):
        apply_boundary_conditions()

        # 1. Momentum equations
        u_star, v_star = solve_momentum(u, v, p, mu, rho, dx, dy, dt)

        # 2. Pressure correction
        p_prime = solve_pressure_correction(u_star, v_star, dx, dy, dt, rho)

        # 3. Velocity/pressure correction
        u_new, v_new = correct_velocity(u_star, v_star, p_prime, dx, dy, dt, rho)
        p_new = p + alpha_p * p_prime

        # Under-relaxation
        u = alpha_u * u_new + (1 - alpha_u) * u
        v = alpha_u * v_new + (1 - alpha_u) * v
        p = p_new

        # Convergence check
        if n % 100 == 0:
            div_max = 0
            for j in range(Ny):
                for i in range(Nx):
                    div = abs((u[j, i+1] - u[j, i]) / dx +
                             (v[j+1, i] - v[j, i]) / dy)
                    div_max = max(div_max, div)
            print(f"Iteration {n}: max|div(u)| = {div_max:.2e}")

    print("Complete!")

    # Convert to cell-centered velocity
    u_center = 0.5 * (u[:, :-1] + u[:, 1:])
    v_center = 0.5 * (v[:-1, :] + v[1:, :])

    # Visualize results
    x = np.linspace(dx/2, L - dx/2, Nx)
    y = np.linspace(dy/2, L - dy/2, Ny)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Velocity vectors
    ax1 = axes[0]
    speed = np.sqrt(u_center**2 + v_center**2)
    ax1.streamplot(X, Y, u_center, v_center, color=speed, cmap='jet',
                  density=2, linewidth=1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Streamlines (SIMPLE), Re = {Re:.0f}')
    ax1.set_aspect('equal')

    # Pressure distribution
    ax2 = axes[1]
    im = ax2.pcolormesh(X, Y, p, cmap='coolwarm', shading='auto')
    plt.colorbar(im, ax=ax2, label='p')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Pressure distribution')
    ax2.set_aspect('equal')

    # Centerline velocity
    ax3 = axes[2]
    j_center = Nx // 2
    ax3.plot(u_center[:, j_center], y, 'b-', linewidth=2)
    ax3.set_xlabel('u')
    ax3.set_ylabel('y')
    ax3.set_title('Vertical centerline u-velocity')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_cavity.png', dpi=150, bbox_inches='tight')
    plt.show()

    return u, v, p

# u, v, p = simple_lid_driven_cavity()
```

---

## 6. SIMPLE Variant Algorithms

### 6.1 SIMPLER, SIMPLEC, PISO

```
SIMPLE variants:

1. SIMPLER (SIMPLE Revised):
   - Directly calculate pressure (no guessing needed)
   - Faster convergence
   - Increased computational cost

2. SIMPLEC (SIMPLE Consistent):
   - Consider neighbor velocity correction terms
   - Less need for under-relaxation
   - Improved convergence

3. PISO (Pressure-Implicit with Splitting of Operators):
   - Suitable for unsteady problems
   - Additional pressure/velocity corrections
   - Improved time accuracy

Selection guide:
- Steady problems: SIMPLE or SIMPLEC
- Unsteady problems: PISO
- Fast convergence needed: SIMPLER
```

```python
def algorithm_comparison():
    """Algorithm comparison concept diagram"""

    print("=" * 60)
    print("SIMPLE Family Algorithm Comparison")
    print("=" * 60)

    comparison = """
    ┌─────────────┬────────────┬────────────┬────────────┐
    │  Algorithm  │   SIMPLE   │  SIMPLEC   │    PISO    │
    ├─────────────┼────────────┼────────────┼────────────┤
    │ Pressure    │     1      │     1      │    2+      │
    │ correction  │            │            │            │
    │ Iter./time  │   Many     │   Few      │   Few      │
    │ αp          │  0.3~0.5   │  0.7~1.0   │    1.0     │
    │ Application │  Steady    │  Steady    │  Unsteady  │
    │ Comp. cost  │   Low      │  Medium    │   High     │
    └─────────────┴────────────┴────────────┴────────────┘

    Key differences of each algorithm:

    SIMPLE:
    - Standard method, simple and robust
    - Under-relaxation essential (αp ~ 0.3)

    SIMPLEC:
    - Consider neighbor contributions in velocity correction
    - Can use αp close to 1

    PISO:
    - Predictor-corrector for unsteady problems
    - 2+ corrections per time step
    - Requires Courant number ≤ 1
    """
    print(comparison)

algorithm_comparison()
```

---

## 7. Practice Problems

### Exercise 1: Stream Function-Vorticity
Find the velocity field and vorticity corresponding to stream function ψ = xy, and verify that it satisfies the incompressibility condition.

### Exercise 2: Lid-Driven Cavity
Perform Lid-Driven Cavity simulation at Re = 400 and compare with Re = 100 results. Observe the development of corner vortices.

### Exercise 3: SIMPLE Under-relaxation
Compare convergence speed in SIMPLE algorithm while varying under-relaxation factor αp as 0.1, 0.3, 0.5.

### Exercise 4: Grid Convergence
Analyze numerical solution convergence for Lid-Driven Cavity problem while varying grid size as 16x16, 32x32, 64x64.

---

## 8. References

### Key Papers
- Ghia et al. (1982) "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method" - Lid-Driven Cavity benchmark
- Patankar & Spalding (1972) "A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows" - SIMPLE algorithm

### Textbooks
- Patankar, "Numerical Heat Transfer and Fluid Flow" (SIMPLE details)
- Ferziger & Peric, "Computational Methods for Fluid Dynamics"
- Moukalled et al., "The Finite Volume Method in Computational Fluid Dynamics"

### CFD Codes
- OpenFOAM: icoFoam (incompressible laminar)
- SIMPLE implementation tutorials: CFD-Online, PyFR

---

## Summary

```
Incompressible flow essentials:

1. Governing equations:
   - Continuity: ∇·u = 0
   - Momentum: Du/Dt = -∇p/ρ + ν∇²u

2. Formulation methods:
   a) Stream function-vorticity (2D):
      - ∇²ψ = -ω
      - ∂ω/∂t + (u·∇)ω = ν∇²ω
   b) Primitive variables + SIMPLE:
      - Handle pressure-velocity coupling

3. Staggered grid:
   - Prevents checkerboard
   - u, v on cell faces, p at cell center

4. SIMPLE algorithm:
   ① Momentum equation with guessed pressure
   ② Pressure correction Poisson equation
   ③ Update velocity/pressure
   ④ Iterate until convergence

5. Numerical considerations:
   - Under-relaxation: αp ~ 0.3, αu ~ 0.7
   - Grid convergence check essential
   - Comply with CFL condition
```

---

## Exercises

### Exercise 1: Stream Function and Velocity Field
Given the stream function ψ = sin(πx)sin(πy) on the unit square [0,1]×[0,1], derive the velocity components u = ∂ψ/∂y and v = -∂ψ/∂x analytically. Verify that the incompressibility condition ∂u/∂x + ∂v/∂y = 0 is satisfied. Then compute the vorticity ω = ∂v/∂x - ∂u/∂y and show that ∇²ψ = -ω holds.

### Exercise 2: Lid-Driven Cavity Resolution Study
Run the stream function-vorticity lid-driven cavity simulation at Re = 100 using grid sizes N = 21, 41, and 81. For each grid, record the location and value of the primary vortex center (minimum of ψ). Compare your results to the Ghia et al. (1982) benchmark: vortex center at approximately (0.617, 0.742) with ψ_min ≈ -0.1034. How does the error depend on grid size?

### Exercise 3: Checkerboard Pressure Instability
On a 5×5 collocated grid, manually construct a checkerboard pressure field p(i,j) = (-1)^(i+j). Compute the central-difference pressure gradient ∂p/∂x at each interior point and show it is identically zero. Explain why this causes problems for the momentum equation and how the staggered grid resolves it.

### Exercise 4: SIMPLE Under-Relaxation Effects
Implement the SIMPLE algorithm for the lid-driven cavity at Re = 100 and experiment with pressure under-relaxation factors αp ∈ {0.1, 0.3, 0.5, 0.7}. For each value, record: (a) the number of outer iterations to reach convergence (max divergence < 10⁻⁴), and (b) whether the simulation diverges. Plot convergence history and identify the optimal αp for fastest stable convergence.

### Exercise 5: Reynolds Number Transition Study
Using the stream function-vorticity solver, simulate the lid-driven cavity at Re = 100, 400, and 1000. For each case, identify: (a) the primary vortex center coordinates, (b) whether secondary corner vortices appear, and (c) the maximum vorticity magnitude. Describe qualitatively how increasing Re changes the flow structure and explain the physical mechanism responsible for corner vortex formation.

---

Next lesson covers numerical analysis of electromagnetics and discretization of Maxwell equations.

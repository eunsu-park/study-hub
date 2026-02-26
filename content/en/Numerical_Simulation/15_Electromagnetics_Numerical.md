# 15. Computational Electromagnetics

## Learning Objectives
- Review the physical meaning of Maxwell's equations
- Understand numerical discretization of electromagnetic fields
- Introduction to the FDTD (Finite-Difference Time-Domain) method
- Understand the Yee lattice structure
- Learn the Courant condition for electromagnetic waves

**Why This Lesson Matters:** Maxwell's equations unify electricity, magnetism, and light into a single framework. Unlike fluids, electromagnetic waves travel at the speed of light and can propagate through vacuum. Numerical methods for electromagnetics (particularly FDTD) are used to design antennas, waveguides, photonic crystals, and to analyze electromagnetic compatibility. The staggered Yee lattice -- where E and H fields are offset by half a grid cell -- is an elegant discretization that automatically satisfies Gauss's laws and has become the industry standard.

---

## 1. Review of Maxwell's Equations

### 1.1 Differential Form

```
Maxwell's Equations (vacuum, SI units):

1. Gauss's Law (Electric):
   ∇·E = ρ/ε₀

2. Gauss's Law (Magnetic):
   ∇·B = 0

3. Faraday's Law:
   ∇×E = -∂B/∂t

4. Ampère-Maxwell Law:
   ∇×B = μ₀J + μ₀ε₀ ∂E/∂t

Where:
- E: Electric field [V/m]
- B: Magnetic field (magnetic flux density) [T]
- ρ: Charge density [C/m³]
- J: Current density [A/m²]
- ε₀ = 8.854×10⁻¹² F/m (vacuum permittivity)
- μ₀ = 4π×10⁻⁷ H/m (vacuum permeability)
- c = 1/√(μ₀ε₀) ≈ 3×10⁸ m/s (speed of light)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
eps0 = 8.854e-12  # Vacuum permittivity [F/m]
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]
c0 = 1 / np.sqrt(mu0 * eps0)  # Speed of light [m/s]

print(f"Vacuum permittivity ε₀ = {eps0:.3e} F/m")
print(f"Vacuum permeability μ₀ = {mu0:.3e} H/m")
print(f"Speed of light c₀ = {c0:.3e} m/s")

def maxwell_equations_overview():
    """Maxwell's equations overview"""

    print("=" * 60)
    print("Maxwell's Equations and Physical Meaning")
    print("=" * 60)

    overview = """
    ┌─────────────────────────────────────────────────────────┐
    │              Maxwell's Equations System                  │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  Gauss (Electric): ∇·E = ρ/ε₀                          │
    │  → Charges are sources of electric field divergence    │
    │                                                         │
    │  Gauss (Magnetic): ∇·B = 0                              │
    │  → No magnetic monopoles (always N-S pairs)            │
    │                                                         │
    │  Faraday: ∇×E = -∂B/∂t                                 │
    │  → Time-varying magnetic field induces electric curl   │
    │                                                         │
    │  Ampère-Maxwell: ∇×B = μ₀J + μ₀ε₀∂E/∂t                 │
    │  → Current and time-varying E induce magnetic curl     │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    Derivation of wave equation:
    Combining Faraday and Ampère laws:

    ∇×(∇×E) = -∂(∇×B)/∂t = -μ₀∂J/∂t - μ₀ε₀∂²E/∂t²

    Applying vector identity ∇×(∇×E) = ∇(∇·E) - ∇²E:

    In vacuum without charges/currents:
    ∇²E = μ₀ε₀ ∂²E/∂t² = (1/c²) ∂²E/∂t²

    → Wave equation with velocity c = 1/√(μ₀ε₀)!
    """
    print(overview)

maxwell_equations_overview()
```

### 1.2 1D Wave Equation

```python
def electromagnetic_wave_1d():
    """Visualization of 1D electromagnetic wave analytical solution"""

    # 1D TEM wave (propagating in x, y polarization)
    # Only Ey and Hz components exist

    # Spatial/temporal setup
    L = 10.0  # Domain length [m]
    T = 3 * L / c0  # Simulation time

    x = np.linspace(0, L, 500)
    times = [0, L/(3*c0), 2*L/(3*c0), L/c0]

    # Initial Gaussian pulse
    x0 = L / 4
    sigma = L / 20
    wavelength = L / 5
    k = 2 * np.pi / wavelength
    omega = c0 * k

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, t in enumerate(times):
        ax = axes[idx // 2, idx % 2]

        # Electric field Ey (Gaussian-modulated sine wave)
        envelope = np.exp(-((x - x0 - c0*t) / sigma)**2)
        Ey = envelope * np.sin(k*(x - x0) - omega*t)

        # Magnetic field Hz (proportional to Ey, same phase)
        Hz = Ey / (c0 * mu0)  # Hz = Ey/(c*μ₀) for plane wave

        ax.plot(x, Ey, 'b-', linewidth=2, label=r'$E_y$ (Electric field)')
        ax.plot(x, Hz * c0 * mu0, 'r--', linewidth=2, label=r'$\mu_0 c H_z$ (Magnetic field)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('Field amplitude')
        ax.set_title(f't = {t*1e9:.2f} ns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)
        ax.set_ylim(-1.5, 1.5)

    plt.suptitle('1D Electromagnetic Wave Propagation (TEM Mode)', fontsize=14)
    plt.tight_layout()
    plt.savefig('em_wave_1d.png', dpi=150, bbox_inches='tight')
    plt.show()

# electromagnetic_wave_1d()
```

---

## 2. Discretization of Maxwell's Equations

### 2.1 Finite Difference Approach

```
1D TEM wave (Ey, Hz components):

Faraday's Law (z component):
∂Ey/∂x = -∂Bz/∂t = -μ₀ ∂Hz/∂t

Ampère's Law (y component):
∂Hz/∂x = -ε₀ ∂Ey/∂t

Discretization (central difference):
∂Ey/∂x ≈ (Ey[i+1] - Ey[i]) / Δx
∂Hz/∂t ≈ (Hz[n+1/2] - Hz[n-1/2]) / Δt

Temporal staggering (Leapfrog):
- E at integer time steps: t = nΔt
- H at half-integer time steps: t = (n+1/2)Δt
```

```python
def fdtd_discretization_concept():
    """Visualization of FDTD discretization concept"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Spatial discretization
    ax1 = axes[0]

    # Staggered positions of E and H
    n_points = 6
    for i in range(n_points):
        # E points (integer positions)
        ax1.plot(i, 0.5, 'bo', markersize=15)
        ax1.text(i, 0.7, f'$E_y^n[{i}]$', ha='center', fontsize=10, color='blue')

        # H points (half-integer positions)
        if i < n_points - 1:
            ax1.plot(i + 0.5, 0.5, 'r^', markersize=12)
            ax1.text(i + 0.5, 0.3, f'$H_z^{{n+1/2}}[{i}]$', ha='center', fontsize=9, color='red')

    # Grid lines
    for i in range(n_points):
        ax1.axvline(x=i, color='gray', linestyle=':', alpha=0.5)

    ax1.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlim(-0.5, n_points - 0.5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Position index i')
    ax1.set_title('Spatial Staggering')
    ax1.set_yticks([])

    # Legend
    ax1.plot([], [], 'bo', markersize=10, label='E field')
    ax1.plot([], [], 'r^', markersize=10, label='H field')
    ax1.legend(loc='upper right')

    # (2) Temporal discretization (Leapfrog)
    ax2 = axes[1]

    n_steps = 5
    for n in range(n_steps):
        # E points (integer time)
        ax2.plot(0.3, n, 'bo', markersize=15)
        ax2.text(0.5, n, f'$E^{n}$', ha='left', fontsize=12, color='blue')

        # H points (half-integer time)
        ax2.plot(0.3, n + 0.5, 'r^', markersize=12)
        ax2.text(0.5, n + 0.5, f'$H^{{{n}+1/2}}$', ha='left', fontsize=11, color='red')

    # Arrows (update sequence)
    for n in range(n_steps - 1):
        # E -> H
        ax2.annotate('', xy=(0.3, n + 0.5), xytext=(0.3, n),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        # H -> E
        ax2.annotate('', xy=(0.3, n + 1), xytext=(0.3, n + 0.5),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, n_steps)
    ax2.set_ylabel('Time step n')
    ax2.set_title('Temporal Staggering (Leapfrog)')
    ax2.set_xticks([])

    plt.tight_layout()
    plt.savefig('fdtd_discretization.png', dpi=150, bbox_inches='tight')
    plt.show()

# fdtd_discretization_concept()
```

### 2.2 Update Equations

```
1D FDTD update equations:

1. H update (n → n+1/2):
   Hz^(n+1/2)[i] = Hz^(n-1/2)[i] - (Δt/μ₀Δx)(Ey^n[i+1] - Ey^n[i])

2. E update (n+1/2 → n+1):
   Ey^(n+1)[i] = Ey^n[i] - (Δt/ε₀Δx)(Hz^(n+1/2)[i] - Hz^(n+1/2)[i-1])

Parameterization:
C_a = Δt/(μ₀Δx)  (H coefficient)
C_b = Δt/(ε₀Δx)  (E coefficient)

Hz^(n+1/2)[i] = Hz^(n-1/2)[i] - C_a (Ey^n[i+1] - Ey^n[i])
Ey^(n+1)[i] = Ey^n[i] - C_b (Hz^(n+1/2)[i] - Hz^(n+1/2)[i-1])
```

```python
def simple_1d_fdtd():
    """Simple 1D FDTD simulation"""

    # Grid setup
    Nx = 200
    dx = 1e-3  # 1 mm

    # Time setup
    dt = dx / (2 * c0)  # Satisfy Courant condition
    n_steps = 500

    # Initialize arrays
    Ey = np.zeros(Nx)
    Hz = np.zeros(Nx)

    # Coefficients
    Ca = dt / (mu0 * dx)
    Cb = dt / (eps0 * dx)

    # Courant number
    S = c0 * dt / dx
    print(f"Courant number S = {S:.4f}")

    # Source position
    source_pos = Nx // 4

    # Recording
    Ey_history = []
    times_to_record = [0, 100, 200, 300, 400]

    # Main loop
    for n in range(n_steps):
        # H update
        Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

        # Source (Gaussian pulse)
        t = n * dt
        t0 = 30 * dt
        tau = 10 * dt
        source = np.exp(-((t - t0) / tau) ** 2)
        Ey[source_pos] += source

        # E update
        Ey[1:] = Ey[1:] - Cb * (Hz[1:] - Hz[:-1])

        # Boundary conditions (simple absorption)
        Ey[0] = 0
        Ey[-1] = 0

        # Recording
        if n in times_to_record:
            Ey_history.append((n, Ey.copy()))

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    x = np.arange(Nx) * dx * 1000  # mm

    for idx, (n, Ey_snap) in enumerate(Ey_history):
        ax = axes[idx // 3, idx % 3]
        ax.plot(x, Ey_snap, 'b-', linewidth=1.5)
        ax.axvline(x=source_pos * dx * 1000, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('Ey')
        ax.set_title(f'Step n = {n}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)

    # Last subplot with description
    ax = axes[1, 2]
    ax.text(0.5, 0.8, 'FDTD Parameters:', fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.6, f'Nx = {Nx}', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, f'dx = {dx*1000:.2f} mm', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'dt = {dt*1e12:.2f} ps', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, f'Courant S = {S:.2f}', fontsize=10, ha='center', transform=ax.transAxes)
    ax.axis('off')

    plt.suptitle('1D FDTD Simulation - Gaussian Pulse Propagation', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_1d_simple.png', dpi=150, bbox_inches='tight')
    plt.show()

    return Ey, Hz

# Ey, Hz = simple_1d_fdtd()
```

---

## 3. Yee Lattice

### 3.1 3D Yee Cell

```
Yee Lattice (1966):
- E and H components are spatially staggered
- Also staggered by half a step in time
- Natural discretization of Maxwell's equations

3D Yee cell structure:
- Ex: Edge centers of y-z faces
- Ey: Edge centers of x-z faces
- Ez: Edge centers of x-y faces
- Hx: Centers of x-perpendicular faces
- Hy: Centers of y-perpendicular faces
- Hz: Centers of z-perpendicular faces

Each E component is surrounded by 4 H components
Each H component is surrounded by 4 E components
```

```python
def yee_cell_visualization():
    """3D Yee cell visualization"""

    fig = plt.figure(figsize=(14, 6))

    # (1) 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    # Cube vertices
    a = 1.0  # Cell size

    # E field positions (edge centers)
    # Ex: (a, a/2, 0), (a, a/2, a), (0, a/2, 0), (0, a/2, a)
    Ex_pos = [(a, a/2, 0), (a, a/2, a), (0, a/2, 0), (0, a/2, a)]
    # Ey positions
    Ey_pos = [(a/2, a, 0), (a/2, a, a), (a/2, 0, 0), (a/2, 0, a)]
    # Ez positions
    Ez_pos = [(0, 0, a/2), (a, 0, a/2), (0, a, a/2), (a, a, a/2)]

    # H field positions (face centers)
    Hx_pos = [(a/2, a/2, 0), (a/2, a/2, a)]
    Hy_pos = [(a/2, 0, a/2), (a/2, a, a/2)]
    Hz_pos = [(0, a/2, a/2), (a, a/2, a/2)]

    # Draw cube
    vertices = [
        [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
        [0, 0, a], [a, 0, a], [a, a, a], [0, a, a]
    ]

    # Edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]

    for edge in edges:
        points = [vertices[edge[0]], vertices[edge[1]]]
        ax1.plot3D(*zip(*points), 'k-', alpha=0.3)

    # E field (arrows)
    for pos in Ex_pos:
        ax1.quiver(pos[0]-0.15, pos[1], pos[2], 0.3, 0, 0, color='blue', arrow_length_ratio=0.3)
    for pos in Ey_pos[:2]:
        ax1.quiver(pos[0], pos[1]-0.15, pos[2], 0, 0.3, 0, color='blue', arrow_length_ratio=0.3)
    for pos in Ez_pos[:2]:
        ax1.quiver(pos[0], pos[1], pos[2]-0.15, 0, 0, 0.3, color='blue', arrow_length_ratio=0.3)

    # H field (arrows)
    for pos in Hx_pos:
        ax1.quiver(pos[0]-0.15, pos[1], pos[2], 0.3, 0, 0, color='red', arrow_length_ratio=0.3)
    for pos in Hy_pos:
        ax1.quiver(pos[0], pos[1]-0.15, pos[2], 0, 0.3, 0, color='red', arrow_length_ratio=0.3)
    for pos in Hz_pos:
        ax1.quiver(pos[0], pos[1], pos[2]-0.15, 0, 0, 0.3, color='red', arrow_length_ratio=0.3)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('3D Yee Cell')

    # Legend
    ax1.plot([], [], 'b-', linewidth=2, label='E field')
    ax1.plot([], [], 'r-', linewidth=2, label='H field')
    ax1.legend()

    # (2) 2D view (x-y plane)
    ax2 = fig.add_subplot(122)

    # Grid
    for i in range(3):
        ax2.axhline(y=i, color='gray', linestyle='-', linewidth=0.5)
        ax2.axvline(x=i, color='gray', linestyle='-', linewidth=0.5)

    # Ez (cell vertices)
    for i in range(3):
        for j in range(3):
            ax2.plot(i, j, 'bo', markersize=12)
            ax2.text(i+0.1, j+0.1, f'Ez({i},{j})', fontsize=8, color='blue')

    # Hx (horizontal edge centers)
    for i in range(3):
        for j in range(2):
            ax2.plot(i, j+0.5, 'r>', markersize=10)

    # Hy (vertical edge centers)
    for i in range(2):
        for j in range(3):
            ax2.plot(i+0.5, j, 'r^', markersize=10)

    # Hz (cell centers)
    for i in range(2):
        for j in range(2):
            ax2.plot(i+0.5, j+0.5, 'rs', markersize=10)
            ax2.text(i+0.6, j+0.5, f'Hz', fontsize=8, color='red')

    ax2.set_xlabel('i')
    ax2.set_ylabel('j')
    ax2.set_title('2D Yee Grid (TM Mode)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.3, 2.5)
    ax2.set_ylim(-0.3, 2.5)

    # Legend
    ax2.plot([], [], 'bo', markersize=10, label='Ez')
    ax2.plot([], [], 'r>', markersize=10, label='Hx')
    ax2.plot([], [], 'r^', markersize=10, label='Hy')
    ax2.plot([], [], 'rs', markersize=10, label='Hz')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('yee_cell.png', dpi=150, bbox_inches='tight')
    plt.show()

# yee_cell_visualization()
```

### 3.2 2D FDTD Modes

```
2D FDTD decomposition:

TM Mode (Transverse Magnetic):
- Components: Ez, Hx, Hy
- Ez in z direction, H in x-y plane

Equations:
∂Hx/∂t = -(1/μ) ∂Ez/∂y
∂Hy/∂t = (1/μ) ∂Ez/∂x
∂Ez/∂t = (1/ε)(∂Hy/∂x - ∂Hx/∂y) - σ/ε Ez

TE Mode (Transverse Electric):
- Components: Hz, Ex, Ey
- Hz in z direction, E in x-y plane

Equations:
∂Ex/∂t = (1/ε) ∂Hz/∂y - σ/ε Ex
∂Ey/∂t = -(1/ε) ∂Hz/∂x - σ/ε Ey
∂Hz/∂t = (1/μ)(∂Ex/∂y - ∂Ey/∂x)
```

```python
def fdtd_2d_tm_mode():
    """2D FDTD TM mode simulation"""

    # Grid setup
    Nx, Ny = 100, 100
    dx = dy = 1e-3  # 1 mm

    # Time setup
    dt = 1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2)) * 0.99  # Courant
    n_steps = 300

    # Material properties
    eps = eps0 * np.ones((Ny, Nx))  # Permittivity
    mu = mu0 * np.ones((Ny, Nx))    # Permeability
    sigma = np.zeros((Ny, Nx))      # Conductivity

    # Initialize arrays
    Ez = np.zeros((Ny, Nx))
    Hx = np.zeros((Ny, Nx))
    Hy = np.zeros((Ny, Nx))

    # Coefficients
    Ca = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
    Cb = (dt / eps) / (1 + sigma * dt / (2 * eps))

    # Source position
    source_x, source_y = Nx // 2, Ny // 2

    # Courant number
    S = c0 * dt * np.sqrt(1/dx**2 + 1/dy**2)
    print(f"2D Courant number S = {S:.4f}")

    # Snapshot recording
    snapshots = []
    record_steps = [50, 100, 150, 200, 250]

    # Main loop
    for n in range(n_steps):
        # H update
        Hx[:, :-1] = Hx[:, :-1] - dt / (mu[:, :-1] * dy) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] = Hy[:-1, :] + dt / (mu[:-1, :] * dx) * (Ez[1:, :] - Ez[:-1, :])

        # Source (Gaussian pulse)
        t = n * dt
        t0 = 50 * dt
        tau = 20 * dt
        source = np.exp(-((t - t0) / tau) ** 2)

        # E update
        Ez[1:-1, 1:-1] = (Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1] +
                         Cb[1:-1, 1:-1] * (
                             (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
                             (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
                         ))

        # Source injection
        Ez[source_y, source_x] += source

        # Snapshot recording
        if n in record_steps:
            snapshots.append((n, Ez.copy()))

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x = np.arange(Nx) * dx * 1000
    y = np.arange(Ny) * dy * 1000
    X, Y = np.meshgrid(x, y)

    for idx, (step, Ez_snap) in enumerate(snapshots):
        ax = axes[idx // 3, idx % 3]
        vmax = np.max(np.abs(Ez_snap)) * 0.8
        if vmax == 0:
            vmax = 1

        im = ax.pcolormesh(X, Y, Ez_snap, cmap='RdBu_r', shading='auto',
                          vmin=-vmax, vmax=vmax)
        ax.plot(source_x * dx * 1000, source_y * dy * 1000, 'k*', markersize=10)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step n = {step}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Ez')

    # Last subplot
    ax = axes[1, 2]
    ax.text(0.5, 0.7, '2D FDTD TM Mode', fontsize=14, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, f'Grid: {Nx} x {Ny}', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'dx = dy = {dx*1000:.1f} mm', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, f'Courant S = {S:.2f}', fontsize=11, ha='center', transform=ax.transAxes)
    ax.axis('off')

    plt.suptitle('2D FDTD Simulation - Circular Wave Propagation from Point Source', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_2d_tm.png', dpi=150, bbox_inches='tight')
    plt.show()

    return Ez, Hx, Hy

# Ez, Hx, Hy = fdtd_2d_tm_mode()
```

---

## 4. Courant Condition

### 4.1 Stability Condition for Electromagnetic Waves

```
Courant-Friedrichs-Lewy (CFL) Condition:

1D:
c·Δt/Δx ≤ 1

2D:
c·Δt·√(1/Δx² + 1/Δy²) ≤ 1

3D:
c·Δt·√(1/Δx² + 1/Δy² + 1/Δz²) ≤ 1

Physical meaning:
- Wave cannot travel more than one cell in one time step
- Numerical information propagation speed ≥ Physical wave speed

Isotropic grid (Δx = Δy = Δz = Δ):
1D: Δt ≤ Δ/c
2D: Δt ≤ Δ/(c√2)
3D: Δt ≤ Δ/(c√3)
```

```python
def courant_condition_analysis():
    """Courant condition analysis"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Stability vs Courant number
    ax1 = axes[0]

    # 1D FDTD simulation (various Courant numbers)
    def run_1d_fdtd(S, n_steps=200):
        Nx = 100
        dx = 1e-3
        dt = S * dx / c0

        Ey = np.zeros(Nx)
        Hz = np.zeros(Nx)

        Ca = dt / (mu0 * dx)
        Cb = dt / (eps0 * dx)

        max_values = []

        for n in range(n_steps):
            Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

            if n == 0:
                Ey[Nx//4] = 1.0  # Initial pulse

            Ey[1:] = Ey[1:] - Cb * (Hz[1:] - Hz[:-1])
            Ey[0] = Ey[-1] = 0

            max_values.append(np.max(np.abs(Ey)))

        return max_values

    courant_numbers = [0.5, 0.9, 1.0, 1.01, 1.1]
    colors = ['green', 'blue', 'orange', 'red', 'darkred']

    for S, color in zip(courant_numbers, colors):
        max_vals = run_1d_fdtd(S)
        label = f'S = {S}' + (' (stable)' if S <= 1 else ' (unstable)')
        ax1.semilogy(max_vals, color=color, linewidth=1.5, label=label)

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('max|Ey|')
    ax1.set_title('1D FDTD: Stability vs Courant Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-3, 1e10)

    # (2) Dispersion relation
    ax2 = axes[1]

    # FDTD numerical dispersion relation
    # ω_num = (2/Δt) arcsin(S sin(kΔx/2))
    # Analytical: ω = ck

    k_norm = np.linspace(0, np.pi, 100)  # kΔx

    for S in [0.5, 0.8, 1.0]:
        omega_exact = k_norm / S  # Normalized ωΔt
        omega_fdtd = 2 * np.arcsin(S * np.sin(k_norm / 2))

        # Phase velocity ratio
        vp_ratio = omega_fdtd / omega_exact

        ax2.plot(k_norm / np.pi, vp_ratio, linewidth=2, label=f'S = {S}')

    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Exact')
    ax2.set_xlabel(r'$k\Delta x / \pi$')
    ax2.set_ylabel(r'$v_p^{num} / c$')
    ax2.set_title('FDTD Numerical Dispersion (Phase Velocity Ratio)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.8, 1.05)

    plt.tight_layout()
    plt.savefig('courant_condition.png', dpi=150, bbox_inches='tight')
    plt.show()

# courant_condition_analysis()
```

### 4.2 Numerical Dispersion

```
FDTD numerical dispersion:

1D dispersion relation:
sin(ωΔt/2) = S·sin(kΔx/2)

Where S = cΔt/Δx (Courant number)

Analytical solution: ω = ck (no dispersion)
FDTD: ω ≠ ck (numerical dispersion occurs)

Problems:
- Severe dispersion at short wavelengths (high frequency)
- Waveform distortion, phase error

Solutions:
- Use minimum 10-20 cells per wavelength
- Use higher-order difference schemes
- Apply dispersion correction techniques
```

---

## 5. Material Modeling

### 5.1 Dielectrics and Conductors

```
Considering material properties:

Dielectrics (ε > ε₀):
- Wave velocity decreases: v = c/√(εᵣ)
- Wavelength decreases: λ = λ₀/√(εᵣ)
- Grid resolution adjustment needed

Conductors (σ > 0):
- Current induction: J = σE
- Wave attenuation
- Skin depth: δ = √(2/(ωμσ))

Lossy media:
Add -σE/ε to ∂E/∂t term
```

```python
def material_modeling_fdtd():
    """2D FDTD with material properties"""

    # Grid setup
    Nx, Ny = 150, 100
    dx = dy = 1e-3

    # Time setup
    dt = 1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2)) * 0.99
    n_steps = 400

    # Initialize material property arrays
    eps_r = np.ones((Ny, Nx))  # Relative permittivity
    sigma = np.zeros((Ny, Nx))  # Conductivity

    # Add dielectric block (εᵣ = 4)
    eps_r[30:70, 80:100] = 4.0

    # Add conductor block (PEC approximation)
    sigma[40:60, 40:55] = 1e7  # High conductivity

    # Actual permittivity
    eps = eps0 * eps_r

    # Initialize arrays
    Ez = np.zeros((Ny, Nx))
    Hx = np.zeros((Ny, Nx))
    Hy = np.zeros((Ny, Nx))

    # Coefficients (including loss)
    Ca = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
    Cb = (dt / eps) / (1 + sigma * dt / (2 * eps))

    # Source position
    source_x, source_y = 20, Ny // 2

    # Snapshots
    snapshots = []
    record_steps = [50, 100, 200, 300]

    for n in range(n_steps):
        # H update
        Hx[:, :-1] = Hx[:, :-1] - dt / (mu0 * dy) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] = Hy[:-1, :] + dt / (mu0 * dx) * (Ez[1:, :] - Ez[:-1, :])

        # Source
        t = n * dt
        t0 = 50 * dt
        tau = 15 * dt
        freq = 5e9  # 5 GHz
        source = np.exp(-((t - t0) / tau) ** 2) * np.sin(2 * np.pi * freq * t)

        # E update
        Ez[1:-1, 1:-1] = (Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1] +
                         Cb[1:-1, 1:-1] * (
                             (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
                             (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
                         ))

        Ez[source_y, source_x] += source

        if n in record_steps:
            snapshots.append((n, Ez.copy()))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(Nx) * dx * 1000
    y = np.arange(Ny) * dy * 1000
    X, Y = np.meshgrid(x, y)

    for idx, (step, Ez_snap) in enumerate(snapshots):
        ax = axes[idx // 2, idx % 2]
        vmax = np.max(np.abs(Ez_snap)) * 0.5
        if vmax == 0:
            vmax = 1

        im = ax.pcolormesh(X, Y, Ez_snap, cmap='RdBu_r', shading='auto',
                          vmin=-vmax, vmax=vmax)

        # Display material regions
        ax.contour(X, Y, eps_r, levels=[2], colors='green', linewidths=2)
        ax.contour(X, Y, sigma, levels=[1e6], colors='black', linewidths=2)

        ax.plot(source_x * dx * 1000, source_y * dy * 1000, 'r*', markersize=10)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step n = {step}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Ez')

    # Legend
    axes[0, 0].plot([], [], 'g-', linewidth=2, label=r'Dielectric ($\epsilon_r=4$)')
    axes[0, 0].plot([], [], 'k-', linewidth=2, label='Conductor (PEC)')
    axes[0, 0].legend(loc='upper right')

    plt.suptitle('2D FDTD with Material Properties', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_materials.png', dpi=150, bbox_inches='tight')
    plt.show()

    return Ez

# Ez = material_modeling_fdtd()
```

---

## 6. CEM Method Comparison

### 6.1 Major Numerical Techniques

```
Computational Electromagnetics (CEM) methods:

1. FDTD (Finite-Difference Time-Domain):
   - Direct time-domain analysis
   - Broadband characteristics in one simulation
   - Simple implementation, easy parallelization

2. FEM (Finite Element Method):
   - Advantageous for complex geometries
   - Uses unstructured grids
   - Higher accuracy with higher-order elements

3. MoM (Method of Moments):
   - Based on integral equations
   - Suitable for open-region problems
   - Antenna, scattering problems

4. FIT (Finite Integration Technique):
   - Integral form Maxwell's equations
   - Excellent energy conservation
   - Commercial software (CST)

5. FDFD (Finite-Difference Frequency-Domain):
   - Frequency-domain steady state
   - Efficient for specific frequency analysis
```

```python
def cem_methods_comparison():
    """CEM methods comparison"""

    print("=" * 70)
    print("Computational Electromagnetics (CEM) Method Comparison")
    print("=" * 70)

    comparison = """
    ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │   Method    │    FDTD     │     FEM     │     MoM     │    FDFD     │
    ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
    │ Domain      │  Time       │ Time/Freq   │  Frequency  │  Frequency  │
    │ Grid        │  Structured │ Unstructured│  Surface    │ Structured  │
    │ Matrix      │  None       │  Sparse     │  Dense      │  Sparse     │
    │ Broadband   │  Efficient  │ Multi-calc  │ Multi-calc  │ Multi-calc  │
    │ Inhomog.    │  Easy       │  Easy       │  Difficult  │  Easy       │
    │ Open region │  ABC/PML    │ Inf. elem.  │  Automatic  │  ABC        │
    │ Parallel    │  Very easy  │  Easy       │  Difficult  │  Easy       │
    │ Nonlinear   │  Possible   │  Possible   │  Difficult  │  Difficult  │
    └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

    Application areas:

    FDTD:
    - Optical devices, antennas, EMC/EMI
    - Bioelectromagnetics
    - Radar cross section (RCS)

    FEM:
    - Waveguides with complex geometries
    - Microwave circuits
    - Antenna feed structures

    MoM:
    - Wire antennas
    - Planar microstrip
    - Scattering problems

    Commercial software:
    - FDTD: Lumerical, XFdtd
    - FEM: ANSYS HFSS, COMSOL
    - MoM: FEKO, NEC
    - FIT: CST Studio
    """
    print(comparison)

cem_methods_comparison()
```

---

## 7. Exercises

### Exercise 1: Maxwell's Equations
Derive the wave equation for the magnetic field B by combining Faraday's law and Ampère's law.

### Exercise 2: 1D FDTD
Simulate reflection and transmission at a material interface (ε₁ → ε₂) in 1D FDTD code. Compare the reflection and transmission coefficients with Fresnel formulas.

### Exercise 3: Courant Condition
Compare the numerical dispersion for Courant numbers S = 0.5 and S = 1.0 in a 2D isotropic grid. Which case is more accurate?

### Exercise 4: Yee Lattice
Draw the positions of H components needed for updating Ex in a 3D Yee lattice and write the update equation.

---

## 8. References

### Key Papers
- Yee (1966) "Numerical Solution of Initial Boundary Value Problems Involving Maxwell's Equations in Isotropic Media" - Original FDTD paper
- Taflove & Brodwin (1975) - Absorbing boundary conditions

### Textbooks
- Taflove & Hagness, "Computational Electrodynamics: The Finite-Difference Time-Domain Method"
- Sullivan, "Electromagnetic Simulation Using the FDTD Method"
- Jin, "The Finite Element Method in Electromagnetics"

### Open Source Tools
- MEEP (MIT, FDTD)
- gprMax (Ground Penetrating Radar)
- OpenEMS (FDTD + circuits)

---

## Summary

```
Computational Electromagnetics Essentials:

1. Maxwell's Equations:
   - ∇×E = -∂B/∂t (Faraday)
   - ∇×H = J + ∂D/∂t (Ampère)
   - ∇·D = ρ (Gauss electric)
   - ∇·B = 0 (Gauss magnetic)

2. FDTD Essentials:
   - E and H spatially/temporally staggered
   - Yee lattice structure
   - Leapfrog time advancement

3. Courant Condition:
   1D: cΔt/Δx ≤ 1
   2D: cΔt√(1/Δx² + 1/Δy²) ≤ 1
   3D: cΔt√(1/Δx² + 1/Δy² + 1/Δz²) ≤ 1

4. Numerical Dispersion:
   - Severe at short wavelengths
   - Recommended 10-20 cells per wavelength

5. Material Modeling:
   - Dielectrics: ε = ε₀εᵣ
   - Conductors: σ > 0
   - Lossy media: Modified Ca, Cb coefficients
```

---

## Exercises

### Exercise 1: Wave Equation Derivation
Starting from Faraday's law (∇×E = -∂B/∂t) and the Ampère-Maxwell law (∇×B = μ₀ε₀ ∂E/∂t) in vacuum with no sources, derive the wave equation for E: ∇²E = (1/c²) ∂²E/∂t². Show each algebraic step clearly and identify where the vector identity ∇×(∇×F) = ∇(∇·F) - ∇²F is used.

### Exercise 2: 1D FDTD Dielectric Interface
Implement the 1D FDTD code from this lesson and add a dielectric half-space at x = L/2 with relative permittivity εr = 4 (simulating glass). Launch a Gaussian pulse from x = L/4. Record the incident, reflected, and transmitted electric field amplitudes at suitable probe points. Compare the reflection coefficient |r| = (√εr1 - √εr2)/(√εr1 + √εr2) and transmission coefficient |t| = 2√εr1/(√εr1 + √εr2) from your simulation against the Fresnel formulas for normal incidence.

### Exercise 3: Courant Number and Stability
Using the 1D FDTD code, run three simulations with Courant numbers S = 0.5, 0.99, and 1.01. For each, plot the maximum field amplitude as a function of time step up to n = 500. Quantify: (a) the maximum amplitude reached for the stable cases, and (b) how many time steps elapse before the S = 1.01 case becomes visibly unstable. Explain the physical meaning of the Courant condition.

### Exercise 4: Numerical Dispersion Measurement
Run a long 1D FDTD simulation (Nx = 1000, n_steps = 5000) with a sinusoidal source at frequency f = 5 GHz. After the simulation reaches steady state, measure the wavelength of the standing wave pattern in the domain. Compare the measured wavelength λ_FDTD = c/f_num to the analytical wavelength λ = c/f to quantify the dispersion error. Repeat for grid spacings dx = λ/10 and dx = λ/20 and observe how dispersion error changes with resolution.

### Exercise 5: 2D TM Mode Scattering by a Cylinder
Extend the 2D FDTD TM mode code to simulate plane wave scattering by a perfectly conducting (PEC) circular cylinder of radius r = 10 mm in a 200×200 grid with dx = dy = 1 mm. Place the cylinder at the center and launch a line source (uniform in y) from one side. Plot the scattered field pattern at t = 300 steps and identify the shadow region, specular reflection, and forward scattering. How does the scattering pattern change if the cylinder radius is doubled?

---

In the next lesson, we will cover detailed FDTD implementation and absorbing boundary conditions.

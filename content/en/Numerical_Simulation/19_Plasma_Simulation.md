# 19. Plasma Simulation

## Learning Objectives
- Understand the basic principles of the Particle-In-Cell (PIC) method
- Implement particle pusher (Boris algorithm)
- Solve field equations (Poisson equation)
- Perform particle-grid interpolation
- Implement 1D electrostatic PIC simulation
- Simulate two-stream instability

**Why This Lesson Matters:** The Particle-In-Cell (PIC) method bridges the gap between fluid models (which average over velocity distributions) and full kinetic theory (which is analytically intractable). By tracking individual particles on a spatial grid, PIC captures non-equilibrium phenomena like wave-particle interactions, Landau damping, and plasma instabilities that fluid models miss. PIC is the primary tool for modeling laser-plasma interactions, particle accelerators, spacecraft charging, and kinetic turbulence in space plasmas.

---

## 1. Introduction to PIC Method

### 1.1 Concepts and Principles

```
Particle-In-Cell (PIC) Method:

Core Idea:
- Represent plasma as discrete "super-particles"
- Electromagnetic fields are calculated on a grid
- Couple particles and grid through interpolation

Advantages:
- Capture kinetic effects (non-equilibrium, wave-particle interactions)
- Easily include relativistic effects
- Handle complex geometries and boundary conditions

Disadvantages:
- Computational cost (many particles required)
- Statistical noise (finite particle number)
- Time step limitations (plasma frequency)

History:
- Buneman, Dawson (1960s)
- Birdsall & Langdon (1991): standard textbook
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# Physical constants (normalized units)
# Length: Debye length λD
# Time: plasma period ωpe^-1
# Velocity: thermal velocity vth

def pic_introduction():
    """Introduction to PIC method"""

    print("=" * 60)
    print("Particle-In-Cell (PIC) Method")
    print("=" * 60)

    intro = """
    PIC Algorithm Cycle:

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │     ┌─────────┐                    ┌─────────┐         │
    │     │Particles│ ──── Interp ────→  │  Grid   │         │
    │     │ (x, v)  │ ← (charge/current) │ (ρ, J)  │         │
    │     └────┬────┘                    └────┬────┘         │
    │          │                              │               │
    │          │                              │ Field         │
    │  Particle│                              │ Solve         │
    │   Push   │                              │ (Poisson/    │
    │          │                              │  Maxwell)    │
    │          │                              │               │
    │     ┌────┴────┐                    ┌────┴────┐         │
    │     │Particles│ ←─── Interp ────   │  Grid   │         │
    │     │ (x, v)  │   (E, B)→accel.    │ (E, B)  │         │
    │     └─────────┘                    └─────────┘         │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    Time Advancement (Leapfrog):
    - Position: t^n → t^{n+1}  (x^{n+1} = x^n + v^{n+1/2}Δt)
    - Velocity: t^{n-1/2} → t^{n+1/2}  (Boris algorithm)
    - Fields: t^n (synchronized with position)
    """
    print(intro)

pic_introduction()
```

### 1.2 PIC Time/Space Scales

```python
def pic_scales():
    """PIC simulation scales"""

    print("=" * 60)
    print("PIC Simulation Scale Conditions")
    print("=" * 60)

    scales = """
    Spatial Resolution:
    Δx < λD (Debye length)
    - λD = √(ε₀kT/(n e²)) = vth/ωpe
    - If Δx ~ λD, unphysical heating occurs

    Temporal Resolution:
    Δt < ωpe^{-1} (plasma period)
    - ωpe = √(ne²/(ε₀m))
    - Δt ~ 0.1-0.2 × ωpe^{-1} recommended

    CFL Condition:
    Δt × vmax < Δx
    - Electrostatic: vmax ~ thermal velocity
    - Electromagnetic: vmax = c (speed of light)

    Number of Particles:
    N_ppc (particles per cell) > 50-100
    - Noise ~ 1/√N_ppc
    - More particles reduce noise

    Simulation Box:
    L > several wavelengths (of phenomenon of interest)
    - Periodic boundaries: long wavelength limitation
    - Open boundaries: need reflection handling
    """
    print(scales)

    # Visualize scale relationships
    fig, ax = plt.subplots(figsize=(10, 6))

    # In normalized units
    # Length unit: λD, Time unit: ωpe^-1

    n_cells = np.arange(1, 100)
    n_ppc = np.array([10, 50, 100, 500])

    for ppc in n_ppc:
        noise = 1 / np.sqrt(ppc * n_cells)
        ax.loglog(n_cells, noise * 100, linewidth=2, label=f'N_ppc = {ppc}')

    ax.axhline(y=1, color='red', linestyle='--', label='1% noise level')
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('Relative noise [%]')
    ax.set_title('PIC Statistical Noise vs Cell Count and Particle Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    plt.tight_layout()
    plt.savefig('pic_noise.png', dpi=150, bbox_inches='tight')
    plt.show()

# pic_scales()
```

---

## 2. Particle Pusher: Boris Algorithm

The Boris algorithm is the most widely used particle integrator in plasma physics, and for good reason: it is second-order accurate, time-reversible, and exactly conserves energy in a pure magnetic field (no electric field). The key insight is splitting the Lorentz force into electric acceleration and magnetic rotation, handling each with an appropriate scheme.

### 2.1 Equations of Motion

The charged particle equation of motion under the Lorentz force is:

$$m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

The magnetic force $q\mathbf{v} \times \mathbf{B}$ is always perpendicular to the velocity, so it changes direction but not speed -- this is why the Boris rotation step is designed to preserve $|\mathbf{v}|$ exactly. The electric force $q\mathbf{E}$ changes both speed and direction, and is handled by simple half-step acceleration.

```
Equations of motion for charged particles:

dx/dt = v

m dv/dt = q(E + v × B)

Boris Algorithm (1970):
- Time-centered scheme
- Energy conservation in magnetic field
- Second-order accuracy

Steps:
1. Half-acceleration (by E): v⁻ = v^{n-1/2} + (qE/m)(Δt/2)
2. Rotation (by B): v' → v⁺ (energy-conserving rotation)
3. Half-acceleration: v^{n+1/2} = v⁺ + (qE/m)(Δt/2)
4. Position update: x^{n+1} = x^n + v^{n+1/2}Δt
```

```python
def boris_pusher(x, v, E, B, q, m, dt):
    """
    Boris particle pusher

    Parameters:
    - x, v: particle position, velocity (arrays)
    - E, B: electric/magnetic fields at particle position
    - q, m: charge, mass
    - dt: time step

    Returns:
    - x_new, v_new: updated position, velocity
    """
    # Convenience coefficient
    qmdt2 = q * dt / (2 * m)

    # 1. Half-acceleration (E)
    v_minus = v + qmdt2 * E

    # 2. Rotation (B)
    # t = (q/m)(B)(Δt/2)
    t = qmdt2 * B
    s = 2 * t / (1 + np.dot(t, t))

    # v' = v⁻ + v⁻ × t
    v_prime = v_minus + np.cross(v_minus, t)

    # v⁺ = v⁻ + v' × s
    v_plus = v_minus + np.cross(v_prime, s)

    # 3. Half-acceleration (E)
    v_new = v_plus + qmdt2 * E

    # 4. Position update
    x_new = x + v_new * dt

    return x_new, v_new


def boris_demo():
    """Boris algorithm demonstration: particle motion in uniform magnetic field"""

    # Setup
    q = 1.0   # charge
    m = 1.0   # mass
    B0 = 1.0  # magnetic field strength (z-direction)

    # Cyclotron frequency and period
    omega_c = q * B0 / m
    T_c = 2 * np.pi / omega_c

    # Time setup
    dt = 0.1  # about 1/63 of T_c
    n_steps = int(3 * T_c / dt)

    # Initial conditions
    x = np.array([0.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])  # initial velocity in x direction
    E = np.array([0.0, 0.0, 0.0])  # no electric field
    B = np.array([0.0, 0.0, B0])   # magnetic field in z direction

    # Record trajectory
    trajectory = [x.copy()]
    velocity = [v.copy()]

    for _ in range(n_steps):
        x, v = boris_pusher(x, v, E, B, q, m, dt)
        trajectory.append(x.copy())
        velocity.append(v.copy())

    trajectory = np.array(trajectory)
    velocity = np.array(velocity)

    # Visualization
    fig = plt.figure(figsize=(14, 5))

    # (1) x-y plane trajectory
    ax1 = fig.add_subplot(131)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=10, label='End')

    # Analytical solution (circle)
    r_L = m * v[0] / (q * B0)  # Larmor radius
    theta = np.linspace(0, 2*np.pi, 100)
    x_exact = r_L * np.sin(theta)
    y_exact = r_L * (1 - np.cos(theta))
    ax1.plot(x_exact, y_exact, 'k--', alpha=0.5, label='Analytical')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('x-y Plane Trajectory (Cyclotron Motion)')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # (2) Energy conservation
    ax2 = fig.add_subplot(132)
    KE = 0.5 * m * np.sum(velocity**2, axis=1)
    t = np.arange(len(KE)) * dt

    ax2.plot(t / T_c, KE / KE[0], 'b-', linewidth=1.5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r't / $T_c$')
    ax2.set_ylabel(r'KE / KE$_0$')
    ax2.set_title('Kinetic Energy Conservation')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.99, 1.01)

    # (3) Velocity components
    ax3 = fig.add_subplot(133)
    ax3.plot(t / T_c, velocity[:, 0], 'r-', linewidth=1.5, label=r'$v_x$')
    ax3.plot(t / T_c, velocity[:, 1], 'b-', linewidth=1.5, label=r'$v_y$')
    ax3.set_xlabel(r't / $T_c$')
    ax3.set_ylabel('v')
    ax3.set_title('Velocity Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('boris_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Accuracy analysis
    print(f"\nBoris Algorithm Analysis:")
    print(f"  dt = {dt:.3f}, ωc·dt = {omega_c * dt:.3f}")
    print(f"  Energy change: {(KE[-1] / KE[0] - 1) * 100:.6f}%")
    print(f"  Larmor radius: Theory {r_L:.3f}, Simulation {np.max(trajectory[:, 0]):.3f}")

# boris_demo()
```

---

## 3. Field Solve

### 3.1 Poisson Equation

```
Field solve in electrostatic PIC:

Poisson equation:
∇²φ = -ρ/ε₀

1D:
d²φ/dx² = -ρ/ε₀

Discretization (central difference):
(φ_{i+1} - 2φ_i + φ_{i-1})/Δx² = -ρ_i/ε₀

Matrix form:
A·φ = b

Boundary conditions:
- Periodic: φ(0) = φ(L)
- Dirichlet: φ = specified value
- Neumann: dφ/dn = specified value

Electric field:
E = -∇φ
E_i = -(φ_{i+1} - φ_{i-1})/(2Δx)
```

```python
def solve_poisson_1d_periodic(rho, dx, eps0=1.0):
    """
    Solve 1D Poisson equation (periodic boundary condition)
    Using FFT

    ∇²φ = -ρ/ε₀
    """
    Nx = len(rho)
    L = Nx * dx

    # FFT
    rho_k = fft(rho)

    # Wavenumber
    k = fftfreq(Nx, dx) * 2 * np.pi

    # Poisson solve (excluding k=0 mode)
    phi_k = np.zeros_like(rho_k, dtype=complex)
    phi_k[1:] = rho_k[1:] / (eps0 * k[1:]**2)
    phi_k[0] = 0  # k=0 mode (average potential = 0)

    # Inverse FFT
    phi = np.real(ifft(phi_k))

    return phi

def solve_poisson_1d_dirichlet(rho, dx, phi_left=0, phi_right=0, eps0=1.0):
    """
    Solve 1D Poisson equation (Dirichlet boundary condition)
    Tridiagonal matrix
    """
    Nx = len(rho)

    # Tridiagonal matrix solver (Thomas algorithm)
    a = np.ones(Nx - 1)         # lower diagonal
    b = -2 * np.ones(Nx)        # diagonal
    c = np.ones(Nx - 1)         # upper diagonal
    d = -rho * dx**2 / eps0     # right-hand side

    # Boundary conditions
    d[0] -= phi_left
    d[-1] -= phi_right

    # Forward sweep
    c_star = np.zeros(Nx - 1)
    d_star = np.zeros(Nx)

    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, Nx - 1):
        c_star[i] = c[i] / (b[i] - a[i-1] * c_star[i-1])

    for i in range(1, Nx):
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / (b[i] - a[i-1] * c_star[i-1] if i < Nx-1 else b[i] - a[i-1] * c_star[i-1])

    # Back substitution
    phi = np.zeros(Nx)
    phi[-1] = d_star[-1]

    for i in range(Nx - 2, -1, -1):
        phi[i] = d_star[i] - c_star[i] * phi[i+1]

    return phi

def electric_field_from_potential(phi, dx):
    """Calculate electric field from potential"""
    Nx = len(phi)
    E = np.zeros(Nx)

    # Central difference (periodic boundary condition)
    E[1:-1] = -(phi[2:] - phi[:-2]) / (2 * dx)
    E[0] = -(phi[1] - phi[-1]) / (2 * dx)
    E[-1] = -(phi[0] - phi[-2]) / (2 * dx)

    return E
```

---

## 4. Particle-Grid Interpolation

### 4.1 Charge Assignment

```
Charge assignment from particles to grid:

1. NGP (Nearest Grid Point):
   - Assign entire charge to nearest grid point
   - First-order accuracy, discontinuous, high noise

2. CIC (Cloud-In-Cell) / Linear:
   - Distribute to two adjacent grid points with linear weights
   - Weight: W_i = 1 - |x - x_i|/Δx
   - Second-order accuracy, continuous, reduced noise

3. TSC (Triangular-Shaped Cloud):
   - Quadratic polynomial weights
   - Distribute to 3 grid points
   - Smoother

4. Spline interpolation:
   - Higher-order B-splines
   - Involve more grid points
```

```python
def charge_to_grid_cic(x_particles, q_particles, Nx, dx, L):
    """
    CIC (Cloud-In-Cell) charge assignment

    Parameters:
    - x_particles: particle position array
    - q_particles: particle charge array
    - Nx: number of grid points
    - dx: grid spacing
    - L: domain size

    Returns:
    - rho: charge density on grid
    """
    rho = np.zeros(Nx)

    for x, q in zip(x_particles, q_particles):
        # Periodic boundary
        x = x % L

        # Left grid point index
        i = int(x / dx)
        i_next = (i + 1) % Nx

        # CIC weights
        frac = (x / dx) - i
        w_left = 1 - frac
        w_right = frac

        # Charge assignment
        rho[i] += q * w_left / dx
        rho[i_next] += q * w_right / dx

    return rho

def field_to_particle_cic(E_grid, x_particle, dx, L):
    """
    CIC field interpolation (grid -> particle)

    Parameters:
    - E_grid: electric field on grid
    - x_particle: particle position
    - dx: grid spacing
    - L: domain size

    Returns:
    - E_particle: electric field at particle position
    """
    Nx = len(E_grid)
    x = x_particle % L

    i = int(x / dx)
    i_next = (i + 1) % Nx

    frac = (x / dx) - i
    w_left = 1 - frac
    w_right = frac

    E_particle = w_left * E_grid[i] + w_right * E_grid[i_next]

    return E_particle


def interpolation_demo():
    """Demonstration of interpolation methods"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Charge assignment comparison
    ax1 = axes[0]

    Nx = 20
    dx = 1.0
    L = Nx * dx

    # Single particle
    x_particle = 5.3 * dx
    q = 1.0

    # NGP
    rho_ngp = np.zeros(Nx)
    i_ngp = int(round(x_particle / dx)) % Nx
    rho_ngp[i_ngp] = q / dx

    # CIC
    rho_cic = charge_to_grid_cic([x_particle], [q], Nx, dx, L)

    x_grid = np.arange(Nx) * dx

    ax1.bar(x_grid - 0.15, rho_ngp, width=0.3, label='NGP', alpha=0.7)
    ax1.bar(x_grid + 0.15, rho_cic, width=0.3, label='CIC', alpha=0.7)
    ax1.axvline(x=x_particle, color='red', linestyle='--', label='Particle Position')

    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$\rho$')
    ax1.set_title('Charge Assignment: NGP vs CIC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Weight functions
    ax2 = axes[1]

    x = np.linspace(-2, 2, 200)

    # NGP
    w_ngp = np.where(np.abs(x) < 0.5, 1, 0)

    # CIC
    w_cic = np.where(np.abs(x) < 1, 1 - np.abs(x), 0)

    # TSC
    w_tsc = np.where(np.abs(x) < 0.5, 0.75 - x**2,
                    np.where(np.abs(x) < 1.5, 0.5 * (1.5 - np.abs(x))**2, 0))

    ax2.plot(x, w_ngp, 'b-', linewidth=2, label='NGP')
    ax2.plot(x, w_cic, 'r-', linewidth=2, label='CIC')
    ax2.plot(x, w_tsc, 'g-', linewidth=2, label='TSC')

    ax2.set_xlabel(r'$(x - x_i) / \Delta x$')
    ax2.set_ylabel('Weight W')
    ax2.set_title('Interpolation Weight Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pic_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()

# interpolation_demo()
```

---

## 5. 1D Electrostatic PIC Simulator

### 5.1 Complete Implementation

```python
class PIC_1D_Electrostatic:
    """1D electrostatic PIC simulator"""

    def __init__(self, Nx, L, dt, n_particles, species_params):
        """
        Parameters:
        - Nx: number of grid points
        - L: domain size (normalized: in λD units)
        - dt: time step (normalized: in ωpe^-1 units)
        - n_particles: number of particles
        - species_params: list of species parameters
          [{'q': charge, 'm': mass, 'n': particle count, 'vth': thermal velocity}]
        """
        self.Nx = Nx
        self.L = L
        self.dx = L / Nx
        self.dt = dt

        # Grid
        self.x_grid = np.linspace(0, L - self.dx, Nx)
        self.rho = np.zeros(Nx)
        self.phi = np.zeros(Nx)
        self.E = np.zeros(Nx)

        # Particle arrays
        self.x = []      # position
        self.v = []      # velocity
        self.q = []      # charge
        self.m = []      # mass
        self.species = []  # species index

        # Initialize particles by species
        for sp_idx, params in enumerate(species_params):
            q_sp = params['q']
            m_sp = params['m']
            n_sp = params['n']
            vth_sp = params.get('vth', 1.0)
            v_drift = params.get('v_drift', 0.0)

            # Uniform position distribution
            x_sp = np.random.uniform(0, L, n_sp)

            # Maxwell velocity distribution (with drift)
            v_sp = np.random.normal(v_drift, vth_sp, n_sp)

            self.x.extend(x_sp)
            self.v.extend(v_sp)
            self.q.extend([q_sp] * n_sp)
            self.m.extend([m_sp] * n_sp)
            self.species.extend([sp_idx] * n_sp)

        self.x = np.array(self.x)
        self.v = np.array(self.v)
        self.q = np.array(self.q)
        self.m = np.array(self.m)
        self.species = np.array(self.species)

        self.n_particles = len(self.x)

        # Super-particle weight (charge density normalization)
        self.weight = L / self.n_particles

        print(f"PIC 1D Initialization:")
        print(f"  Grid: Nx = {Nx}, dx = {self.dx:.4f}")
        print(f"  Time: dt = {dt}")
        print(f"  Particles: N = {self.n_particles}")

    def deposit_charge(self):
        """Charge deposition (CIC)"""
        self.rho = np.zeros(self.Nx)

        for i in range(self.n_particles):
            x = self.x[i] % self.L
            q = self.q[i]

            j = int(x / self.dx)
            j_next = (j + 1) % self.Nx

            frac = (x / self.dx) - j
            w_left = 1 - frac
            w_right = frac

            self.rho[j] += q * w_left * self.weight / self.dx
            self.rho[j_next] += q * w_right * self.weight / self.dx

        # Background charge (neutralization)
        self.rho -= np.mean(self.rho)

    def solve_field(self):
        """Solve Poisson equation (FFT)"""
        self.phi = solve_poisson_1d_periodic(self.rho, self.dx)
        self.E = electric_field_from_potential(self.phi, self.dx)

    def interpolate_field(self):
        """Field interpolation (grid -> particle)"""
        E_particles = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            x = self.x[i] % self.L
            j = int(x / self.dx)
            j_next = (j + 1) % self.Nx

            frac = (x / self.dx) - j
            w_left = 1 - frac
            w_right = frac

            E_particles[i] = w_left * self.E[j] + w_right * self.E[j_next]

        return E_particles

    def push_particles(self):
        """Particle pusher (Leapfrog)"""
        E_p = self.interpolate_field()

        # Velocity update: v^{n-1/2} -> v^{n+1/2}
        self.v += (self.q / self.m) * E_p * self.dt

        # Position update: x^n -> x^{n+1}
        self.x += self.v * self.dt

        # Periodic boundary
        self.x = self.x % self.L

    def compute_diagnostics(self):
        """Compute diagnostics"""
        # Kinetic energy
        KE = 0.5 * np.sum(self.m * self.v**2) * self.weight

        # Field energy (electrostatic)
        FE = 0.5 * np.sum(self.E**2) * self.dx

        # Total energy
        TE = KE + FE

        return {'KE': KE, 'FE': FE, 'TE': TE}

    def step(self):
        """One time step"""
        self.deposit_charge()
        self.solve_field()
        self.push_particles()

    def run(self, n_steps, diag_interval=10):
        """Run simulation"""
        diagnostics = {'t': [], 'KE': [], 'FE': [], 'TE': []}

        for n in range(n_steps):
            if n % diag_interval == 0:
                diag = self.compute_diagnostics()
                diagnostics['t'].append(n * self.dt)
                diagnostics['KE'].append(diag['KE'])
                diagnostics['FE'].append(diag['FE'])
                diagnostics['TE'].append(diag['TE'])

            self.step()

        return {k: np.array(v) for k, v in diagnostics.items()}
```

### 5.2 Langmuir Wave Test

```python
def langmuir_wave_test():
    """Langmuir wave (electron plasma wave) test"""

    # Setup (normalized units)
    Nx = 64
    L = 2 * np.pi * 4  # 4 wavelengths
    dt = 0.1           # in ωpe^-1 units

    # Electrons (ions are fixed background)
    n_electrons = 10000
    vth = 1.0  # thermal velocity (normalized)

    species = [
        {'q': -1.0, 'm': 1.0, 'n': n_electrons, 'vth': vth, 'v_drift': 0.0}
    ]

    # Create simulator
    pic = PIC_1D_Electrostatic(Nx, L, dt, n_electrons, species)

    # Initial perturbation (density wave)
    k = 2 * np.pi / (L / 4)  # wavenumber (4 wavelengths)
    amplitude = 0.01

    # Position perturbation
    pic.x += amplitude * np.sin(k * pic.x) / k

    # Run simulation
    n_steps = 500
    diagnostics = pic.run(n_steps, diag_interval=5)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) Energy time evolution
    ax1 = axes[0, 0]
    t = diagnostics['t']
    ax1.plot(t, diagnostics['KE'], 'b-', label='Kinetic Energy')
    ax1.plot(t, diagnostics['FE'], 'r-', label='Field Energy')
    ax1.plot(t, diagnostics['TE'], 'k--', label='Total Energy')
    ax1.set_xlabel(r't [$\omega_{pe}^{-1}$]')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Time Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Field energy (log scale, verify oscillation)
    ax2 = axes[0, 1]
    ax2.semilogy(t, diagnostics['FE'], 'r-')
    ax2.set_xlabel(r't [$\omega_{pe}^{-1}$]')
    ax2.set_ylabel('Field Energy')
    ax2.set_title('Field Energy (Langmuir Oscillation)')
    ax2.grid(True, alpha=0.3)

    # (3) Phase space
    ax3 = axes[1, 0]
    ax3.scatter(pic.x, pic.v, s=0.5, alpha=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('v')
    ax3.set_title(f'Phase Space (t = {n_steps * dt:.1f})')
    ax3.grid(True, alpha=0.3)

    # (4) Charge density
    ax4 = axes[1, 1]
    pic.deposit_charge()
    ax4.plot(pic.x_grid, pic.rho, 'b-')
    ax4.set_xlabel('x')
    ax4.set_ylabel(r'$\rho$')
    ax4.set_title('Charge Density')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Langmuir Wave Test', fontsize=14)
    plt.tight_layout()
    plt.savefig('langmuir_wave.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Frequency analysis
    from scipy.signal import find_peaks

    FE = diagnostics['FE']
    peaks, _ = find_peaks(FE)

    if len(peaks) > 1:
        T_measured = np.mean(np.diff(t[peaks]))
        omega_measured = 2 * np.pi / T_measured
        print(f"\nMeasured frequency: ω = {omega_measured:.3f} ωpe")
        print(f"Theoretical value: ω ≈ ωpe = 1.0 (k→0 limit)")

# langmuir_wave_test()
```

---

## 6. Two-Stream Instability

### 6.1 Physical Background

```
Two-Stream Instability:

Setup:
- Two electron beams moving in opposite directions
- Ions are fixed background

Dispersion relation:
1 = ωpe²/2 × [1/(ω - kv₀)² + 1/(ω + kv₀)²]

Instability condition:
k < kc = ωpe/v₀

Growth rate (maximum):
γmax ≈ √3/2 × ωpe (at k = ωpe/v₀)

Physical outcome:
- Exponential field growth
- Particle trapping (phase space vortices)
- Thermalization
```

```python
def two_stream_instability():
    """Two-stream instability simulation"""

    # Setup
    Nx = 128
    L = 2 * np.pi * 8  # 8 wavelengths
    dt = 0.1

    # Two electron beams
    n_per_beam = 20000
    vth = 0.1  # small thermal velocity
    v_drift = 3.0  # drift velocity

    species = [
        # Beam 1 (moving right)
        {'q': -1.0, 'm': 1.0, 'n': n_per_beam, 'vth': vth, 'v_drift': v_drift},
        # Beam 2 (moving left)
        {'q': -1.0, 'm': 1.0, 'n': n_per_beam, 'vth': vth, 'v_drift': -v_drift}
    ]

    # Create simulator
    pic = PIC_1D_Electrostatic(Nx, L, dt, n_per_beam * 2, species)

    # Small initial perturbation
    np.random.seed(42)
    pic.x += 0.001 * np.random.randn(pic.n_particles)
    pic.x = pic.x % L

    # Simulation
    n_steps = 600
    diag_interval = 5

    # Save phase space snapshots
    snapshots = []
    snapshot_times = [0, 100, 200, 300, 400, 500]

    diagnostics = {'t': [], 'FE': []}

    for n in range(n_steps):
        if n % diag_interval == 0:
            pic.deposit_charge()
            pic.solve_field()
            FE = 0.5 * np.sum(pic.E**2) * pic.dx
            diagnostics['t'].append(n * dt)
            diagnostics['FE'].append(FE)

        if n in snapshot_times:
            snapshots.append({
                't': n * dt,
                'x': pic.x.copy(),
                'v': pic.v.copy()
            })

        pic.step()

    diagnostics = {k: np.array(v) for k, v in diagnostics.items()}

    # Visualization
    fig = plt.figure(figsize=(16, 12))

    # Phase space snapshots
    for i, snap in enumerate(snapshots):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.scatter(snap['x'], snap['v'], s=0.2, alpha=0.3, c='blue')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title(f"t = {snap['t']:.0f}")
        ax.set_xlim(0, L)
        ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Two-Stream Instability: Phase Space Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig('two_stream_phase_space.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Energy and growth rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Field energy (log scale)
    ax1 = axes[0]
    t = diagnostics['t']
    FE = diagnostics['FE']

    ax1.semilogy(t, FE, 'b-', linewidth=1.5)

    # Fit linear growth region
    linear_region = (t > 10) & (t < 35)
    if np.any(linear_region) and np.any(FE[linear_region] > 0):
        log_FE = np.log(FE[linear_region])
        t_fit = t[linear_region]
        coeffs = np.polyfit(t_fit, log_FE, 1)
        gamma_measured = coeffs[0] / 2  # FE ∝ exp(2γt)

        ax1.semilogy(t_fit, np.exp(np.polyval(coeffs, t_fit)), 'r--',
                    linewidth=2, label=f'Fit: γ = {gamma_measured:.3f}')
        ax1.legend()

    ax1.set_xlabel(r't [$\omega_{pe}^{-1}$]')
    ax1.set_ylabel('Field Energy')
    ax1.set_title('Field Energy Growth')
    ax1.grid(True, alpha=0.3)

    # Theoretical growth rate
    ax2 = axes[1]

    k = np.linspace(0.01, 2, 100)
    # Growth rate from approximate dispersion relation (maximum growth)
    # γ ≈ ωpe × √(3)/2 × (k v₀/ωpe)^(1/3) for small k
    gamma_theory = 0.866  # √3/2

    ax2.text(0.5, 0.7, f'Theoretical Maximum Growth Rate:\nγmax ≈ (√3/2)ωpe ≈ {gamma_theory:.3f}',
            transform=ax2.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat'))

    ax2.text(0.5, 0.4, f'Measured Growth Rate:\nγ ≈ {gamma_measured:.3f}' if 'gamma_measured' in dir() else '',
            transform=ax2.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax2.text(0.1, 0.1, """
Two-Stream Instability Characteristics:
1. Linear phase: exponential growth
2. Nonlinear phase: particle trapping
3. Saturation phase: thermalization
    """, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom')

    ax2.axis('off')
    ax2.set_title('Instability Analysis')

    plt.tight_layout()
    plt.savefig('two_stream_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return pic, diagnostics

# pic, diagnostics = two_stream_instability()
```

---

## 7. Advanced Topics

### 7.1 Electromagnetic PIC

```
Electromagnetic PIC Extension:

Additional Physics:
- Include magnetic field (B)
- Solve full Maxwell equations
- Relativistic particle motion

Maxwell Equations:
∂B/∂t = -∇×E
∂E/∂t = c²∇×B - J/ε₀

Current Assignment:
J = Σᵢ qᵢvᵢ × (interpolation weight)

Time Advancement:
- E, B: FDTD-like
- Particles: relativistic Boris algorithm

Applications:
- Laser-plasma interactions
- Relativistic shocks
- Particle acceleration
```

```python
def em_pic_overview():
    """Electromagnetic PIC overview"""

    print("=" * 60)
    print("Electromagnetic PIC")
    print("=" * 60)

    overview = """
    Electromagnetic PIC Algorithm:

    ┌─────────────────────────────────────────────────────────┐
    │ Time step n → n+1                                       │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ 1. B^n → B^{n+1/2} (half step)                          │
    │    B^{n+1/2} = B^n - (Δt/2)∇×E^n                        │
    │                                                         │
    │ 2. Particle push: (x^n, v^{n-1/2}) → (x^{n+1}, v^{n+1/2})│
    │    - Interpolate: E^n, B^{n+1/2} → particles            │
    │    - Boris: acceleration and rotation                   │
    │    - Position update                                    │
    │                                                         │
    │ 3. Current assignment: J^{n+1/2}                        │
    │    Esirkepov (charge conserving) or Villasenor-Buneman │
    │                                                         │
    │ 4. E update: E^n → E^{n+1}                              │
    │    E^{n+1} = E^n + Δt(c²∇×B^{n+1/2} - J^{n+1/2}/ε₀)    │
    │                                                         │
    │ 5. B^{n+1/2} → B^{n+1} (half step)                      │
    │    B^{n+1} = B^{n+1/2} - (Δt/2)∇×E^{n+1}               │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    Code Examples:
    - EPOCH (UK)
    - OSIRIS (UCLA/IST)
    - SMILEI (France)
    - WarpX (LBNL)
    """
    print(overview)

em_pic_overview()
```

---

## 8. Exercises

### Exercise 1: Boris Algorithm
Simulate particle motion in a uniform electric field E = E₀x and magnetic field B = B₀z. Verify E×B drift.

### Exercise 2: CIC vs NGP
Compare CIC and NGP charge assignment for the same particle distribution. Which gives smoother charge density?

### Exercise 3: Thermal Equilibrium
Verify that a Maxwell distribution is maintained in a single-species plasma PIC simulation. Does numerical heating occur?

### Exercise 4: Two-Beam Instability
Vary the drift velocity v₀ in the two-stream simulation and measure the growth rate γ. Compare with theoretical values.

---

## 9. References

### Core Textbooks
- Birdsall & Langdon, "Plasma Physics via Computer Simulation" (standard textbook)
- Hockney & Eastwood, "Computer Simulation Using Particles"
- Arber et al., "Contemporary Particle-In-Cell Approach to Laser-Plasma Modelling"

### PIC Codes
- EPOCH: laser-plasma
- OSIRIS: high-performance, relativistic
- SMILEI: modular, open-source
- WarpX: GPU-accelerated, Exascale

### Online Resources
- Plasma Theory Group resources
- UCLA PICKSC tutorials
- LBNL WarpX documentation

---

## Summary

```
PIC Simulation Essentials:

1. Algorithm Cycle:
   Particles → charge/current → field solve → interpolation → particle push

2. Particle Push (Boris):
   - Half-acceleration → rotation → half-acceleration
   - Energy conservation, second-order accuracy

3. Field Solve:
   - Electrostatic: Poisson (∇²φ = -ρ/ε₀)
   - Electromagnetic: Maxwell (FDTD-like)

4. Interpolation:
   - NGP: 0th order, discontinuous
   - CIC: 1st order, linear
   - TSC: 2nd order, smooth

5. Scale Conditions:
   - Δx < λD
   - Δt < ωpe^-1
   - N_ppc > 50-100

6. Validation Tests:
   - Langmuir wave (plasma oscillation)
   - Two-stream instability
   - Particle drift

7. Diagnostics:
   - Energy conservation
   - Phase space distribution
   - Dispersion relation
```

---

## Exercises

### Exercise 1: E×B Drift with Boris Algorithm
Using the boris_pusher function, simulate a proton (charge q = +e, mass mp = 1.673×10⁻²⁷ kg) in crossed uniform fields: electric field E = 1000 V/m in the x-direction and magnetic field B = 0.01 T in the z-direction. The theoretical E×B drift velocity is vd = E/B in the y-direction, independent of charge and mass. Run the simulation for 10 cyclotron periods, record the guiding center trajectory, and verify that the drift speed matches the theoretical value vd = E×B / B² = E/B. Also verify that the Boris algorithm conserves kinetic energy in a pure magnetic field (no electric field).

### Exercise 2: CIC vs NGP Charge Density Comparison
Set up a 1D domain with 100 grid cells and place 10,000 electrons initialized with a Maxwell-Boltzmann velocity distribution at thermal velocity vth = 1 (normalized units). Compute the charge density on the grid using both NGP and CIC interpolation schemes. Plot both density profiles on the same axis and compute the high-frequency noise level (standard deviation relative to mean density) for each method. Show that CIC reduces noise by approximately a factor of √(N_cell) compared to NGP for the same number of particles.

### Exercise 3: Langmuir Wave Frequency Measurement
Initialize a 1D electrostatic PIC simulation with Nx = 64 cells, L = 4λD (Debye lengths), N = 10,000 electrons, and a sinusoidal density perturbation of amplitude δn/n = 0.01 at wavenumber k = 2π/L. Record the electrostatic field energy as a function of time. Use the FFT of the field energy time series to measure the oscillation frequency ωpe. Compare your measured frequency to the theoretical plasma frequency ωpe = √(ne²/(ε₀me)) in normalized units (ωpe = 1). At what wavenumber k does the Bohm-Gross dispersion ω² = ωpe²(1 + 3k²λD²) predict a 10% correction over ωpe?

### Exercise 4: Two-Stream Instability Growth Rate Measurement
Run the two-stream instability simulation with drift velocities v₀ ∈ {1.0, 2.0, 3.0, 5.0} (normalized by the thermal velocity). For each v₀, measure the linear growth rate γ by fitting an exponential to the field energy during the early exponential growth phase (E_field ∝ exp(2γt)). Plot the measured γ versus v₀ and compare to the theoretical maximum growth rate γmax ≈ (√3/2)ωpe obtained from the two-beam dispersion relation. At what v₀ does the growth rate saturate, and what physical effect causes this?

### Exercise 5: Particle Number and Numerical Heating
Run three PIC simulations of a single-species Maxwellian plasma (no instability driver) for 500 plasma periods using N_ppc ∈ {10, 50, 200} particles per cell. In an ideal simulation with no numerical heating, the total kinetic energy should remain constant. Plot the total kinetic energy as a function of time for all three cases. Measure the numerical heating rate (energy increase per plasma period) and verify that it decreases approximately as 1/N_ppc. What is the minimum N_ppc needed to keep numerical heating below 1% of the initial energy over 500 periods?

---

This concludes the numerical simulation series on CFD, electromagnetism, MHD, and plasma topics.

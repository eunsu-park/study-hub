# 3. Plasma Description Hierarchy

## Learning Objectives

- Understand the hierarchy of plasma descriptions from N-body to kinetic to fluid models
- Derive the Klimontovich equation and its relationship to statistical descriptions
- Explain the closure problem in moment equations and common closure schemes
- Identify when to use particle, kinetic, or fluid descriptions based on physical regime
- Implement simple numerical comparisons of different description levels
- Appreciate the trade-offs between accuracy and computational efficiency across the hierarchy

## 1. Overview of the Hierarchy

Plasmas can be described at multiple levels of detail, forming a **hierarchy of models** that trade accuracy for computational tractability:

```
Hierarchy of Plasma Descriptions:

┌─────────────────────────────────────────────────────────────┐
│  Level 1: N-Body (Microscopic)                              │
│  Track all particles individually: {x_i(t), v_i(t)}         │
│  Phase space: 6N dimensions                                 │
│  Exact but intractable for N ~ 10^20                        │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼ Ensemble averaging
┌─────────────────────────────────────────────────────────────┐
│  Level 2: Kinetic (Statistical)                             │
│  Distribution function: f(x, v, t)                          │
│  Phase space: 6 dimensions + time                           │
│  Vlasov (collisionless) or Boltzmann (collisional)         │
│  Retains velocity space information                         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼ Velocity moments
┌─────────────────────────────────────────────────────────────┐
│  Level 3: Fluid (Macroscopic)                               │
│  Moments: n(x,t), u(x,t), T(x,t), ...                       │
│  Configuration space: 3 dimensions + time                   │
│  MHD (magnetohydrodynamics)                                 │
│  Requires closure approximation                             │
└─────────────────────────────────────────────────────────────┘
```

Each level is obtained from the one above by **coarse-graining** or **averaging**, sacrificing information but gaining computational efficiency.

### 1.1 When to Use Each Level

| Description | Best For | Examples |
|-------------|----------|----------|
| **N-body** | Few particles, strong correlations | Dusty plasmas, molecular dynamics, validation |
| **Kinetic** | Wave-particle interactions, non-Maxwellian distributions | Landau damping, beam instabilities, magnetic reconnection |
| **Fluid** | Large scales, low frequency, high collisionality | MHD equilibria, macroscopic stability, turbulence |

The choice depends on:
- **Scale separation**: Are there well-separated fast/slow timescales?
- **Collisionality**: Is the distribution Maxwellian (fluid) or non-Maxwellian (kinetic)?
- **Computational resources**: Kinetic simulations are much more expensive than fluid

## 2. The N-Body Description

### 2.1 Exact Equations of Motion

For $N$ particles with charges $q_\alpha$ and masses $m_\alpha$, the exact dynamics are governed by:

$$m_\alpha \frac{d\mathbf{v}_\alpha}{dt} = q_\alpha \left(\mathbf{E}(\mathbf{x}_\alpha, t) + \mathbf{v}_\alpha \times \mathbf{B}(\mathbf{x}_\alpha, t)\right)$$

$$\frac{d\mathbf{x}_\alpha}{dt} = \mathbf{v}_\alpha$$

where the fields are determined by all charges:

$$\mathbf{E}(\mathbf{x}, t) = \sum_{\beta=1}^{N} \frac{q_\beta}{4\pi\epsilon_0} \frac{\mathbf{x} - \mathbf{x}_\beta}{|\mathbf{x} - \mathbf{x}_\beta|^3} + \mathbf{E}_{ext}$$

$$\mathbf{B}(\mathbf{x}, t) = \mathbf{B}_{ext} + \text{(relativistic corrections)}$$

This is a **6N-dimensional** dynamical system. For a typical plasma with $N \sim 10^{20}$ particles, this is completely intractable.

### 2.2 Liouville's Theorem

The N-particle distribution function $F_N(\mathbf{x}_1, \mathbf{v}_1, \ldots, \mathbf{x}_N, \mathbf{v}_N, t)$ satisfies **Liouville's equation**:

$$\frac{dF_N}{dt} = \frac{\partial F_N}{\partial t} + \sum_{\alpha=1}^{N} \left(\mathbf{v}_\alpha \cdot \frac{\partial F_N}{\partial \mathbf{x}_\alpha} + \frac{\mathbf{F}_\alpha}{m_\alpha} \cdot \frac{\partial F_N}{\partial \mathbf{v}_\alpha}\right) = 0$$

**Interpretation:** The probability density is conserved along trajectories in phase space (incompressible flow in phase space).

This is exact but still involves 6N variables.

## 3. The Klimontovich Equation

### 3.1 Microscopic Distribution Function

To bridge from N-body to kinetic, we introduce the **Klimontovich microscopic density**:

$$f^{micro}(\mathbf{x}, \mathbf{v}, t) = \sum_{\alpha=1}^{N} \delta(\mathbf{x} - \mathbf{x}_\alpha(t)) \delta(\mathbf{v} - \mathbf{v}_\alpha(t))$$

This is a sum of delta functions—a "grainy" distribution function representing the exact positions and velocities of all particles.

### 3.2 Klimontovich Equation

The microscopic distribution satisfies:

$$\frac{\partial f^{micro}}{\partial t} + \mathbf{v} \cdot \nabla f^{micro} + \frac{q}{m}(\mathbf{E}^{micro} + \mathbf{v} \times \mathbf{B}^{micro}) \cdot \nabla_v f^{micro} = 0$$

where $\mathbf{E}^{micro}$ and $\mathbf{B}^{micro}$ are the microscopic fields generated by all particles (including the particle under consideration).

This is the **Klimontovich equation**—still exact, but now in $(x, v, t)$ space rather than 6N-dimensional phase space.

**Key point:** $f^{micro}$ is extremely singular (sum of delta functions), so it's not directly useful. We need to **smooth** it via ensemble averaging.

## 4. From Klimontovich to Vlasov/Boltzmann

### 4.1 Ensemble Averaging

We perform a **statistical average** over an ensemble of realizations:

$$f(\mathbf{x}, \mathbf{v}, t) = \langle f^{micro}(\mathbf{x}, \mathbf{v}, t) \rangle$$

This replaces the grainy microscopic distribution with a **smooth average distribution**.

The fields are similarly decomposed:

$$\mathbf{E}^{micro} = \mathbf{E} + \delta\mathbf{E}$$
$$\mathbf{B}^{micro} = \mathbf{B} + \delta\mathbf{B}$$

where $\mathbf{E}, \mathbf{B}$ are the smoothed (mean) fields and $\delta\mathbf{E}, \delta\mathbf{B}$ are fluctuations.

### 4.2 Vlasov Equation (Collisionless Limit)

If we average the Klimontovich equation and **neglect correlations** (mean-field approximation), we obtain the **Vlasov equation**:

$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{q}{m}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f = 0$$

coupled to Maxwell's equations:

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}, \quad \rho = \sum_s q_s \int f_s d^3v$$

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \epsilon_0 \mu_0 \frac{\partial \mathbf{E}}{\partial t}, \quad \mathbf{J} = \sum_s q_s \int \mathbf{v} f_s d^3v$$

This is the **Vlasov-Maxwell system**, describing collisionless plasma dynamics.

**Assumptions:**
- Mean-field: each particle responds to the smoothed fields, not individual particle fields
- No collisions: $f$ evolves reversibly
- Valid when: $n\lambda_D^3 \gg 1$ (weak coupling)

### 4.3 Boltzmann Equation (With Collisions)

When collisions are important, the ensemble averaging introduces a **collision term**:

$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{q}{m}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f = C[f]$$

where $C[f]$ is the **collision operator**, typically the Boltzmann collision integral:

$$C[f_a] = \sum_b \int d^3v' \int d\Omega \, \sigma(\Omega) \, |\mathbf{v} - \mathbf{v}'| \left(f_a' f_b'^* - f_a f_b\right)$$

Here, $f_a'$ denotes $f_a(\mathbf{v}')$, and $f_b'^*$ denotes $f_b(\mathbf{v}'^*)$ after a collision.

For Coulomb collisions, the collision integral is more complex (Landau or Fokker-Planck form).

## 5. The BBGKY Hierarchy

### 5.1 Reduced Distribution Functions

An alternative systematic approach uses **reduced distribution functions**:

- **One-particle distribution** $f_1(\mathbf{x}_1, \mathbf{v}_1, t)$: probability to find any particle at $(\mathbf{x}_1, \mathbf{v}_1)$
- **Two-particle distribution** $f_2(\mathbf{x}_1, \mathbf{v}_1, \mathbf{x}_2, \mathbf{v}_2, t)$: joint probability
- And so on...

These are obtained by integrating the N-particle distribution:

$$f_1(\mathbf{x}_1, \mathbf{v}_1, t) = \int d^3x_2 \cdots d^3x_N \, d^3v_2 \cdots d^3v_N \, F_N$$

### 5.2 BBGKY Equations

The equation for $f_1$ involves $f_2$; the equation for $f_2$ involves $f_3$; and so on:

$$\frac{\partial f_1}{\partial t} + \mathbf{v}_1 \cdot \nabla_1 f_1 + \frac{\mathbf{F}_1^{ext}}{m} \cdot \nabla_{v_1} f_1 = \int d^3x_2 d^3v_2 \, \frac{\mathbf{F}_{12}}{m} \cdot \nabla_{v_1} f_2$$

This is the **BBGKY hierarchy** (Bogoliubov-Born-Green-Kirkwood-Yvon).

**Closure:** To solve for $f_1$, we need an approximation for $f_2$ in terms of $f_1$. Common approximations:
- **Mean field**: $f_2(\mathbf{x}_1, \mathbf{v}_1, \mathbf{x}_2, \mathbf{v}_2) \approx f_1(\mathbf{x}_1, \mathbf{v}_1) f_1(\mathbf{x}_2, \mathbf{v}_2)$ (no correlations)
- **Boltzmann**: Include two-body correlations but neglect higher-order

The mean-field closure recovers the Vlasov equation.

## 6. Fluid Moments and Closure

### 6.1 Moment Definitions

From the kinetic distribution $f(\mathbf{x}, \mathbf{v}, t)$, we define **fluid moments** by integrating over velocity:

**Density (0th moment):**

$$n(\mathbf{x}, t) = \int f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

**Mean velocity (1st moment):**

$$\mathbf{u}(\mathbf{x}, t) = \frac{1}{n} \int \mathbf{v} \, f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

**Pressure tensor (2nd moment):**

$$\mathsf{P}(\mathbf{x}, t) = m \int (\mathbf{v} - \mathbf{u})(\mathbf{v} - \mathbf{u}) \, f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

For an isotropic distribution, $\mathsf{P} = p \mathsf{I}$ with scalar pressure $p = nk_B T$.

**Heat flux (3rd moment):**

$$\mathbf{q}(\mathbf{x}, t) = \frac{m}{2} \int |\mathbf{v} - \mathbf{u}|^2 (\mathbf{v} - \mathbf{u}) \, f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

### 6.2 Moment Equations

Taking moments of the Vlasov/Boltzmann equation yields a hierarchy of fluid equations:

**0th moment (Continuity):**

$$\frac{\partial n}{\partial t} + \nabla \cdot (n\mathbf{u}) = 0$$

**1st moment (Momentum):**

$$mn\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}\right) = qn(\mathbf{E} + \mathbf{u} \times \mathbf{B}) - \nabla \cdot \mathsf{P} + \mathbf{R}$$

where $\mathbf{R}$ is the collision-induced momentum transfer.

**2nd moment (Energy/Pressure):**

$$\frac{\partial}{\partial t}\left(\frac{3}{2}p\right) + \nabla \cdot \left(\frac{3}{2}p\mathbf{u}\right) + \mathsf{P}:\nabla\mathbf{u} + \nabla \cdot \mathbf{q} = Q$$

where $Q$ represents heating/cooling.

### 6.3 The Closure Problem

**Problem:** Each moment equation introduces the next higher moment:
- Continuity involves $\mathbf{u}$
- Momentum involves $\mathsf{P}$
- Energy involves $\mathbf{q}$
- And so on...

This is an **infinite hierarchy**—we have more unknowns than equations.

**Solution:** **Closure**—approximate higher moments in terms of lower ones.

### 6.4 Common Closures

**1. Adiabatic closure (MHD):**

Assume $\mathbf{q} = 0$ (no heat conduction) and isotropic pressure with an adiabatic equation of state:

$$p \propto n^\gamma$$

where $\gamma$ is the adiabatic index (5/3 for a monatomic gas).

**2. Isothermal closure:**

Assume $T = \text{const}$, so $p = nk_B T$.

**3. Braginskii closure:**

For magnetized plasmas, use parallel and perpendicular heat fluxes and viscosity tensors derived from kinetic theory.

**4. Moment closures (e.g., 13-moment):**

Carry more moments (e.g., $\mathsf{P}$, $\mathbf{q}$, and fourth moment) with phenomenological closures.

## 7. Comparison of Descriptions

### 7.1 Computational Cost

| Method | Variables | Dimensionality | Typical Grid Size | Scalability |
|--------|-----------|----------------|-------------------|-------------|
| N-body | $6N$ | 6N-dimensional | — | $\mathcal{O}(N^2)$ or $\mathcal{O}(N\log N)$ |
| PIC (particle-in-cell) | $N$ particles + fields | 3D + particles | $10^6$ cells, $10^9$ particles | $\mathcal{O}(N)$ |
| Vlasov (continuum) | $f(x,v,t)$ | 6D + time | $10^6$ grid points (3D-3V) | Grid-dependent |
| Gyrokinetic | $f(x,v_\parallel,\mu,t)$ | 5D + time | $10^5$ grid points | Grid-dependent |
| Fluid (MHD) | $n, \mathbf{u}, T, \mathbf{B}$ | 3D + time | $10^5$ cells | Grid-dependent |

**PIC** (Particle-In-Cell) is a hybrid: it uses particles (kinetic) but computes fields on a grid, reducing cost to $\mathcal{O}(N)$.

### 7.2 Physical Fidelity

```
Physical Fidelity:

High ┌────────────────────────────────────────────────┐
  ↑  │ N-body                                         │
  │  │  • Exact (within classical EM)                 │
  │  │  • All correlations included                   │
  │  │  • Intractable for large N                     │
  │  └────────────────────────────────────────────────┘
  │                   ↓
  │  ┌────────────────────────────────────────────────┐
  │  │ Kinetic (Vlasov/Boltzmann)                     │
  │  │  • Retains velocity distribution               │
  │  │  • Captures wave-particle interactions         │
  │  │  • Non-Maxwellian effects                      │
  │  │  • Expensive: 6D phase space                   │
  │  └────────────────────────────────────────────────┘
  │                   ↓
  │  ┌────────────────────────────────────────────────┐
  │  │ Fluid (MHD)                                    │
  │  │  • Only moments (n, u, T)                      │
  │  │  • Assumes local thermodynamic equilibrium     │
  │  │  • Closure approximation required              │
  │  │  • Fast: 3D only                               │
Low └────────────────────────────────────────────────┘
```

### 7.3 Validity Regimes

**Fluid (MHD) valid when:**
- High collisionality: $\nu \gg \omega$ (collision frequency exceeds wave frequency)
- Near-Maxwellian distribution
- Length scales $\gg r_{L,i}$ (ion Larmor radius)
- Time scales $\gg \omega_{ci}^{-1}$ (ion gyroperiod)

**Kinetic required when:**
- Wave-particle resonances (Landau damping, cyclotron resonance)
- Non-Maxwellian features (beams, loss cones)
- Collisionless shocks
- Magnetic reconnection (electron scale)

## 8. Numerical Examples

### 8.1 One-Dimensional Plasma Oscillation

We'll compare N-body, Vlasov, and fluid descriptions for a simple 1D plasma oscillation.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Physical constants
e = 1.602176634e-19
m_e = 9.1093837015e-31
epsilon_0 = 8.8541878128e-12
k_B = 1.380649e-23

# Plasma parameters
n0 = 1e18  # m^-3
T = 1e5    # K (~ 10 eV)
L = 1.0    # Domain length [m]
omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))

print(f"Plasma frequency: ω_pe = {omega_pe:.3e} rad/s")
print(f"Plasma period: τ_pe = {2*np.pi/omega_pe:.3e} s")

# Perturbation
k = 2 * np.pi / L  # Wavenumber
amplitude = 0.01   # Perturbation amplitude

class NBodyPlasma1D:
    """1D N-body electrostatic plasma simulation."""

    def __init__(self, N, L, n0, T, amplitude, k):
        self.N = N
        self.L = L
        self.n0 = n0
        self.T = T
        self.v_th = np.sqrt(k_B * T / m_e)

        # Initialize positions (perturbed uniform distribution)
        self.x = np.linspace(0, L, N, endpoint=False)
        self.x += amplitude * L / k * np.sin(k * self.x)
        self.x = self.x % L  # Periodic

        # Initialize velocities (Maxwellian)
        self.v = np.random.normal(0, self.v_th, N)

        # Charge per particle
        self.q = -e
        self.m = m_e

    def compute_field(self, Ng=128):
        """Compute electric field using Poisson solver on grid."""
        # Deposit charge to grid
        rho_grid, edges = np.histogram(self.x, bins=Ng, range=(0, self.L))
        dx = self.L / Ng
        rho_grid = self.q * rho_grid / dx  # Charge density

        # Background neutralizing charge
        rho_grid -= self.q * self.N / self.L

        # Solve Poisson equation: d^2 phi / dx^2 = -rho / epsilon_0
        # Using FFT
        rho_k = np.fft.rfft(rho_grid)
        k_modes = 2 * np.pi * np.fft.rfftfreq(Ng, d=dx)
        k_modes[0] = 1  # Avoid division by zero (set DC to zero)

        phi_k = -rho_k / (epsilon_0 * k_modes**2)
        phi_k[0] = 0  # No DC potential

        # Electric field: E = -d phi / dx
        E_k = 1j * k_modes * phi_k
        E_grid = np.fft.irfft(E_k, n=Ng)

        return E_grid, edges

    def step(self, dt):
        """Advance system by dt using leapfrog."""
        # Compute field
        E_grid, edges = self.compute_field()

        # Interpolate field to particle positions
        Ng = len(E_grid)
        dx = self.L / Ng
        indices = np.floor(self.x / dx).astype(int) % Ng
        E_particles = E_grid[indices]

        # Push velocities (half step)
        self.v += (self.q / self.m) * E_particles * (dt / 2)

        # Push positions
        self.x += self.v * dt
        self.x = self.x % self.L  # Periodic BC

        # Push velocities (half step)
        E_grid, _ = self.compute_field()
        indices = np.floor(self.x / dx).astype(int) % Ng
        E_particles = E_grid[indices]
        self.v += (self.q / self.m) * E_particles * (dt / 2)


class VlasovPlasma1D:
    """1D Vlasov-Poisson simulation on phase space grid."""

    def __init__(self, Nx, Nv, L, v_max, n0, T, amplitude, k):
        self.Nx = Nx
        self.Nv = Nv
        self.L = L
        self.v_max = v_max

        self.x = np.linspace(0, L, Nx, endpoint=False)
        self.v = np.linspace(-v_max, v_max, Nv)
        self.dx = L / Nx
        self.dv = 2 * v_max / Nv

        X, V = np.meshgrid(self.x, self.v, indexing='ij')

        # Initial distribution: perturbed Maxwellian
        v_th = np.sqrt(k_B * T / m_e)
        f_max = (1 / (np.sqrt(2*np.pi) * v_th)) * np.exp(-V**2 / (2*v_th**2))
        density_pert = n0 * (1 + amplitude * np.sin(k * X))

        self.f = density_pert[:, np.newaxis] * f_max

    def compute_field(self):
        """Compute electric field from Poisson equation."""
        # Density
        n = np.trapz(self.f, self.v, axis=1)

        # Charge density (with neutralizing background)
        rho = -e * (n - n0)

        # Solve Poisson via FFT
        rho_k = np.fft.rfft(rho)
        k_modes = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.dx)
        k_modes[0] = 1

        phi_k = -rho_k / (epsilon_0 * k_modes**2)
        phi_k[0] = 0

        E_k = 1j * k_modes * phi_k
        E = np.fft.irfft(E_k, n=self.Nx)

        return E

    def step(self, dt):
        """Advance Vlasov equation using splitting."""
        E = self.compute_field()

        # Step 1: Advection in x (v * df/dx)
        for j in range(self.Nv):
            v_val = self.v[j]
            shift = int(np.round(v_val * dt / self.dx))
            self.f[:, j] = np.roll(self.f[:, j], -shift)

        # Step 2: Acceleration in v ((qE/m) * df/dv)
        for i in range(self.Nx):
            accel = -e * E[i] / m_e
            shift = int(np.round(accel * dt / self.dv))
            self.f[i, :] = np.roll(self.f[i, :], -shift)


class FluidPlasma1D:
    """1D cold fluid (two-fluid) model."""

    def __init__(self, Nx, L, n0, amplitude, k):
        self.Nx = Nx
        self.L = L
        self.dx = L / Nx

        self.x = np.linspace(0, L, Nx, endpoint=False)

        # Initialize density and velocity
        self.n = n0 * (1 + amplitude * np.sin(k * self.x))
        self.u = np.zeros(Nx)  # Initially at rest

    def compute_field(self):
        """Compute electric field."""
        rho = -e * (self.n - n0)

        rho_k = np.fft.rfft(rho)
        k_modes = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.dx)
        k_modes[0] = 1

        phi_k = -rho_k / (epsilon_0 * k_modes**2)
        phi_k[0] = 0

        E_k = 1j * k_modes * phi_k
        E = np.fft.irfft(E_k, n=self.Nx)

        return E

    def step(self, dt):
        """Advance using simple Euler (for demonstration)."""
        E = self.compute_field()

        # Continuity: dn/dt = -d(nu)/dx
        nu = self.n * self.u
        nu_k = np.fft.rfft(nu)
        k_modes = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.dx)
        d_nu_dx = np.fft.irfft(1j * k_modes * nu_k, n=self.Nx)

        self.n -= d_nu_dx * dt

        # Momentum (cold, neglecting pressure): du/dt = qE/m - u * du/dx
        u_k = np.fft.rfft(self.u)
        du_dx = np.fft.irfft(1j * k_modes * u_k, n=self.Nx)

        self.u += (-e * E / m_e - self.u * du_dx) * dt


# Run simulations
print("\nRunning simulations...")

# N-body
N_particles = 10000
nbody = NBodyPlasma1D(N_particles, L, n0, T, amplitude, k)

# Vlasov
Nx_vlasov = 64
Nv_vlasov = 64
v_max = 5 * np.sqrt(k_B * T / m_e)
vlasov = VlasovPlasma1D(Nx_vlasov, Nv_vlasov, L, v_max, n0, T, amplitude, k)

# Fluid
Nx_fluid = 128
fluid = FluidPlasma1D(Nx_fluid, L, n0, amplitude, k)

# Time stepping
dt = 0.01 / omega_pe
Nt = 200
times = np.arange(Nt) * dt

# Storage for diagnostics
nbody_density_history = []
vlasov_density_history = []
fluid_density_history = []

for step in range(Nt):
    # N-body
    nbody.step(dt)
    rho_nb, _ = np.histogram(nbody.x, bins=64, range=(0, L))
    rho_nb = rho_nb / (L/64) * N_particles / n0  # Normalize
    nbody_density_history.append(rho_nb)

    # Vlasov
    vlasov.step(dt)
    n_vl = np.trapz(vlasov.f, vlasov.v, axis=1) / n0
    vlasov_density_history.append(n_vl)

    # Fluid
    fluid.step(dt)
    fluid_density_history.append(fluid.n / n0)

print("Simulations complete.")

# Plot comparison at t = π/ω_pe (quarter period)
idx = int(np.pi / (omega_pe * dt) / 2)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

x_plot = np.linspace(0, L, 64)

ax = axes[0]
ax.plot(x_plot, nbody_density_history[idx], 'o-', label='N-body', alpha=0.7)
ax.plot(x_plot, nbody_density_history[0], '--', label='Initial', alpha=0.5)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('N-body Simulation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
x_vlasov = vlasov.x
ax.plot(x_vlasov, vlasov_density_history[idx], 's-', label='Vlasov', alpha=0.7, markersize=4)
ax.plot(x_vlasov, vlasov_density_history[0], '--', label='Initial', alpha=0.5)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Vlasov Simulation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
x_fluid = fluid.x
ax.plot(x_fluid, fluid_density_history[idx], '^-', label='Fluid', alpha=0.7, markersize=3)
ax.plot(x_fluid, fluid_density_history[0], '--', label='Initial', alpha=0.5)
ax.set_xlabel('Position [m]', fontsize=11)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Fluid Simulation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plasma_oscillation_comparison.png', dpi=150)
plt.show()

print(f"\nPlot shows density at t = {times[idx]:.2e} s ≈ π/(2ω_pe)")
```

### 8.2 Phase Space Evolution (Vlasov)

```python
def plot_phase_space_evolution():
    """Visualize phase space evolution for Vlasov simulation."""

    # Reinitialize
    vlasov2 = VlasovPlasma1D(64, 128, L, v_max, n0, T, amplitude, k)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times_plot = [0, int(0.25*Nt), int(0.5*Nt), int(0.75*Nt)]
    titles = ['t = 0', 't = π/(2ω_pe)', 't = π/ω_pe', 't = 3π/(2ω_pe)']

    for ax, t_idx, title in zip(axes.flat, times_plot, titles):
        # Advance to desired time
        for _ in range(t_idx):
            vlasov2.step(dt)

        X, V = np.meshgrid(vlasov2.x, vlasov2.v, indexing='ij')

        contour = ax.contourf(X, V / np.sqrt(k_B * T / m_e),
                              vlasov2.f.T, levels=20, cmap='viridis')
        ax.set_xlabel('Position x [m]', fontsize=11)
        ax.set_ylabel(r'Velocity $v/v_{th}$', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.colorbar(contour, ax=ax, label=r'$f(x,v)$')

    plt.tight_layout()
    plt.savefig('vlasov_phase_space.png', dpi=150)
    plt.show()

plot_phase_space_evolution()
```

### 8.3 Energy Conservation Check

```python
def compare_energy_conservation():
    """Compare energy conservation across methods."""

    # Reinitialize
    nbody3 = NBodyPlasma1D(10000, L, n0, T, amplitude, k)
    vlasov3 = VlasovPlasma1D(64, 64, L, v_max, n0, T, amplitude, k)
    fluid3 = FluidPlasma1D(128, L, n0, amplitude, k)

    nbody_KE = []
    vlasov_KE = []
    fluid_KE = []

    for step in range(Nt):
        # N-body kinetic energy
        KE_nb = 0.5 * m_e * np.sum(nbody3.v**2)
        nbody_KE.append(KE_nb)
        nbody3.step(dt)

        # Vlasov kinetic energy
        V_grid = vlasov3.v[np.newaxis, :]
        KE_vl = 0.5 * m_e * np.sum(vlasov3.f * V_grid**2) * vlasov3.dx * vlasov3.dv
        vlasov_KE.append(KE_vl)
        vlasov3.step(dt)

        # Fluid kinetic energy
        KE_fl = 0.5 * m_e * np.sum(fluid3.n * fluid3.u**2) * fluid3.dx
        fluid_KE.append(KE_fl)
        fluid3.step(dt)

    # Normalize to initial value
    nbody_KE = np.array(nbody_KE) / nbody_KE[0]
    vlasov_KE = np.array(vlasov_KE) / vlasov_KE[0]
    fluid_KE = np.array(fluid_KE) / fluid_KE[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(times * omega_pe, nbody_KE, label='N-body', linewidth=2, alpha=0.8)
    ax.plot(times * omega_pe, vlasov_KE, label='Vlasov', linewidth=2, alpha=0.8)
    ax.plot(times * omega_pe, fluid_KE, label='Fluid', linewidth=2, alpha=0.8)

    ax.set_xlabel(r'Time $\omega_{pe} t$', fontsize=12)
    ax.set_ylabel('Normalized Kinetic Energy', fontsize=12)
    ax.set_title('Energy Conservation Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('energy_conservation_comparison.png', dpi=150)
    plt.show()

compare_energy_conservation()
```

## Summary

The hierarchy of plasma descriptions provides a systematic framework for modeling plasmas at different levels of detail:

1. **N-body**: Exact classical dynamics of all particles, intractable for real plasmas ($N \sim 10^{20}$).

2. **Klimontovich equation**: Exact microscopic distribution function (sum of delta functions), bridging N-body and statistical descriptions.

3. **Kinetic (Vlasov/Boltzmann)**: Smooth distribution function $f(x, v, t)$ in 6D phase space, obtained by ensemble averaging. Retains velocity-space information essential for wave-particle interactions.

4. **Fluid (MHD)**: Velocity moments $n, \mathbf{u}, T$ in 3D configuration space, obtained by integrating over velocity. Requires closure to truncate the moment hierarchy.

5. **Closure problem**: Moment equations form an infinite hierarchy; approximations (adiabatic, isothermal, Braginskii) are needed to close the system.

6. **Regime dependence**: Particle methods for strong correlations, kinetic for non-Maxwellian distributions and wave-particle resonance, fluid for large-scale MHD phenomena.

Understanding when each level is appropriate—and how to navigate between them—is essential for efficient and accurate plasma modeling.

## Practice Problems

### Problem 1: Klimontovich to Vlasov

Starting from the Klimontovich equation:

$$\frac{\partial f^{micro}}{\partial t} + \mathbf{v} \cdot \nabla f^{micro} + \frac{q}{m}(\mathbf{E}^{micro} + \mathbf{v} \times \mathbf{B}^{micro}) \cdot \nabla_v f^{micro} = 0$$

(a) Write $f^{micro} = \langle f^{micro} \rangle + \delta f$ and $\mathbf{E}^{micro} = \langle \mathbf{E}^{micro} \rangle + \delta \mathbf{E}$.

(b) Ensemble-average the Klimontovich equation. What assumption leads to the Vlasov equation (hint: neglect $\langle \delta f \delta \mathbf{E} \rangle$)?

(c) Physically, what does the term $\langle \delta f \delta \mathbf{E} \rangle$ represent? Why can it be neglected for weakly coupled plasmas?

### Problem 2: Moment Closure

For a 1D electrostatic plasma with electron distribution $f_e(x, v, t)$:

(a) Derive the continuity equation (0th moment):
   $$\frac{\partial n_e}{\partial t} + \frac{\partial (n_e u_e)}{\partial x} = 0$$

(b) Derive the momentum equation (1st moment), showing that it involves the pressure $p_e$.

(c) Assume an isothermal closure $p_e = n_e k_B T_e$ with constant $T_e$. Combined with Poisson's equation, show that small perturbations satisfy:
   $$\frac{\partial^2 n_e}{\partial t^2} = \frac{k_B T_e}{m_e} \frac{\partial^2 n_e}{\partial x^2} + \omega_{pe}^2 (n_e - n_0)$$

(d) For plane waves $n_e \propto e^{i(kx - \omega t)}$, find the dispersion relation. Compare to Langmuir waves from kinetic theory.

### Problem 3: Collisionless vs Collisional Regimes

Consider a plasma oscillation with frequency $\omega \approx \omega_{pe}$.

(a) Using the Knudsen number $Kn = \lambda_{mfp}/L$ and collision frequency $\nu_{ei}$, determine the criterion for the oscillation to be collisionless.

(b) For a tokamak with $n_e = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $L = 1$ m, compute $\nu_{ei}$ and $\omega_{pe}$. Is the plasma oscillation collisionless?

(c) Repeat for a glow discharge with $n_e = 10^{16}$ m$^{-3}$, $T_e = 2$ eV, $L = 0.1$ m.

(d) Based on these results, which system requires a kinetic description for plasma oscillations?

### Problem 4: Phase Space Density Conservation

(a) Show that the Vlasov equation can be written as:
   $$\frac{df}{dt} = \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \mathbf{a} \cdot \nabla_v f = 0$$
   where $\mathbf{a} = (q/m)(\mathbf{E} + \mathbf{v} \times \mathbf{B})$.

(b) Interpret this as $df/dt = 0$ along particle trajectories in $(x, v)$ phase space.

(c) Use this to argue that the volume in phase space is conserved (**Liouville's theorem**).

(d) Explain why the Boltzmann equation (with collisions) violates phase space volume conservation.

### Problem 5: Fluid vs Kinetic Landau Damping

The damping of Langmuir waves differs between fluid and kinetic treatments.

(a) From Problem 2, the fluid dispersion relation for Langmuir waves is:
   $$\omega^2 = \omega_{pe}^2 + 3k^2 v_{te}^2$$
   Show that this predicts **no damping** (real $\omega$).

(b) Landau's kinetic theory gives:
   $$\omega^2 \approx \omega_{pe}^2 + 3k^2 v_{te}^2 - i\sqrt{\frac{\pi}{8}} \frac{\omega_{pe}}{(kv_{te})^3} e^{-1/(2k^2\lambda_D^2)}$$
   for $k\lambda_D \ll 1$. Identify the imaginary part responsible for damping.

(c) Physically, why does the fluid model miss Landau damping? (Hint: resonant particles at $v \approx \omega/k$.)

(d) For $n_e = 10^{19}$ m$^{-3}$, $T_e = 100$ eV, $k = 100$ m$^{-1}$, estimate the Landau damping rate $\gamma = \text{Im}(\omega)$.

---

**Previous:** [Coulomb Collisions](./02_Coulomb_Collisions.md) | **Next:** [Single Particle Motion I](./04_Single_Particle_Motion_I.md)

# 17. Spectral and Advanced Methods

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand the principles of pseudo-spectral methods for MHD
- Implement Fourier-based spectral solvers for periodic MHD problems
- Apply dealiasing techniques to handle nonlinear terms correctly
- Use Chebyshev spectral methods for non-periodic boundary value problems
- Understand Adaptive Mesh Refinement (AMR) for MHD
- Describe hybrid MHD-PIC methods for kinetic effects
- Compare SPH-MHD with grid-based methods
- Survey production MHD codes: Athena++, PLUTO, FLASH, Pencil Code, Dedalus
- Implement a simplified 2D pseudo-spectral MHD solver in Python

---

## 1. Introduction to Spectral Methods

### 1.1 Why Spectral Methods?

Finite difference and finite volume methods offer:
- **Accuracy:** $O(\Delta x^2)$ to $O(\Delta x^5)$ (WENO)
- **Flexibility:** Complex geometries, AMR, shocks

Spectral methods offer:
- **Accuracy:** Exponential convergence for smooth solutions
- **Efficiency:** Exact derivatives via FFT ($O(N \log N)$)
- **Simplicity:** Natural for incompressible turbulence, dynamos

**Trade-offs:**
| Method | Convergence | Shocks | BC | Geometries |
|--------|-------------|--------|-----|------------|
| Finite Volume | Algebraic | Excellent | Flexible | Any |
| Spectral (Fourier) | Exponential | Poor (Gibbs) | Periodic only | Simple |
| Spectral (Chebyshev) | Exponential | Poor | Flexible | 1D/2D slab |

**Best use cases:**
- MHD turbulence (periodic box)
- Dynamo simulations (high Reynolds numbers)
- Linear stability analysis (eigenmodes)

### 1.2 Fourier vs Chebyshev Basis

**Fourier:**
$$
f(x) = \sum_{k=-N/2}^{N/2-1} \hat{f}_k e^{i k x}
$$
- Periodic BC: $f(0) = f(2\pi)$
- Derivatives: $\partial_x f \leftrightarrow i k \hat{f}_k$
- Orthogonality: $\int_0^{2\pi} e^{i k x} e^{-i k' x} dx = 2\pi \delta_{kk'}$

**Chebyshev:**
$$
f(x) = \sum_{n=0}^{N} a_n T_n(x), \quad T_n(x) = \cos(n \arccos x), \quad x \in [-1, 1]
$$
- Non-periodic BC: Dirichlet, Neumann, mixed
- Derivatives: recurrence relations (more complex than Fourier)
- Clustering: Chebyshev points cluster near boundaries (resolves boundary layers)

---

## 2. Pseudo-Spectral Methods for MHD

### 2.1 Basic Algorithm

Consider incompressible MHD in a periodic box:
$$
\frac{\partial \mathbf{v}}{\partial t} = -(\mathbf{v} \cdot \nabla)\mathbf{v} + (\mathbf{B} \cdot \nabla)\mathbf{B} - \nabla p + \nu \nabla^2 \mathbf{v}
$$
$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}
$$
$$
\nabla \cdot \mathbf{v} = 0, \quad \nabla \cdot \mathbf{B} = 0
$$

**Pseudo-spectral scheme:**

1. **Initialization:** FFT of $\mathbf{v}(\mathbf{x}, 0)$ and $\mathbf{B}(\mathbf{x}, 0)$ → $\hat{\mathbf{v}}(\mathbf{k}, 0)$, $\hat{\mathbf{B}}(\mathbf{k}, 0)$

2. **Time loop:**
   - **Nonlinear terms:** Compute $(\mathbf{v} \cdot \nabla)\mathbf{v}$ and $\mathbf{v} \times \mathbf{B}$ in **physical space**
     - IFFT: $\hat{\mathbf{v}}(\mathbf{k}) \to \mathbf{v}(\mathbf{x})$, $\hat{\mathbf{B}}(\mathbf{k}) \to \mathbf{B}(\mathbf{x})$
     - Compute: $\mathbf{NL}_v = -(\mathbf{v} \cdot \nabla)\mathbf{v} + (\mathbf{B} \cdot \nabla)\mathbf{B}$, $\mathbf{NL}_B = \nabla \times (\mathbf{v} \times \mathbf{B})$
     - FFT: $\mathbf{NL}_v(\mathbf{x}) \to \widehat{\mathbf{NL}}_v(\mathbf{k})$, $\mathbf{NL}_B(\mathbf{x}) \to \widehat{\mathbf{NL}}_B(\mathbf{k})$

   - **Linear terms:** Compute in **spectral space**
     - Diffusion: $\widehat{\nabla^2 \mathbf{v}} = -k^2 \hat{\mathbf{v}}$
     - Pressure: Project $\widehat{\mathbf{NL}}_v$ to divergence-free subspace
       $$
       \hat{\mathbf{v}}_{\perp}(\mathbf{k}) = \hat{\mathbf{v}}(\mathbf{k}) - \frac{\mathbf{k} \cdot \hat{\mathbf{v}}(\mathbf{k})}{k^2} \mathbf{k}
       $$

   - **Time integration:** RK4 or exponential integrator
     $$
     \frac{d \hat{\mathbf{v}}}{dt} = \widehat{\mathbf{NL}}_v - \nu k^2 \hat{\mathbf{v}}
     $$

3. **Divergence cleaning:** Enforce $\nabla \cdot \mathbf{B} = 0$ at each step:
   $$
   \hat{\mathbf{B}}_{\perp}(\mathbf{k}) = \hat{\mathbf{B}}(\mathbf{k}) - \frac{\mathbf{k} \cdot \hat{\mathbf{B}}(\mathbf{k})}{k^2} \mathbf{k}
   $$

### 2.2 Dealiasing (2/3 Rule)

**Problem:** Aliasing errors from nonlinear terms.

Product of two functions:
$$
f(x) g(x) \quad \text{with max wavenumber } k_{\max}
$$
has Fourier modes up to $2 k_{\max}$ (convolution theorem).

If grid has $N$ points, Nyquist wavenumber $k_N = N/2$. Modes $k > k_N$ **alias** to lower $k$.

**Solution:** 2/3 dealiasing (Orszag 1971)
- Keep only modes $|k| \leq k_{\max} = 2N/3$
- Zero out modes $|k| > 2N/3$ before IFFT
- After computing nonlinear term, zero high-$k$ modes again

**Cost:** Lose 1/3 of modes, but **exact** representation of quadratic nonlinearity.

### 2.3 Time Integration

**Explicit RK4:**
$$
\frac{d \hat{u}}{dt} = L \hat{u} + N(\hat{u})
$$
where $L = -\nu k^2$ (linear), $N$ (nonlinear).

Standard RK4:
$$
\begin{aligned}
k_1 &= L \hat{u}^n + N(\hat{u}^n) \\
k_2 &= L(\hat{u}^n + \frac{\Delta t}{2} k_1) + N(\hat{u}^n + \frac{\Delta t}{2} k_1) \\
k_3 &= L(\hat{u}^n + \frac{\Delta t}{2} k_2) + N(\hat{u}^n + \frac{\Delta t}{2} k_2) \\
k_4 &= L(\hat{u}^n + \Delta t k_3) + N(\hat{u}^n + \Delta t k_3) \\
\hat{u}^{n+1} &= \hat{u}^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

**Exponential integrator (ETDRK4):**

For stiff linear terms ($\nu k^2$ large at high $k$):
$$
\hat{u}^{n+1} = e^{L \Delta t} \hat{u}^n + \Delta t \phi_1(L \Delta t) N(\hat{u}^n) + \ldots
$$
where $\phi_1(z) = (e^z - 1)/z$.

**Advantage:** Unconditionally stable for diffusion (allows larger $\Delta t$).

---

## 3. Incompressible MHD Turbulence

### 3.1 Governing Equations (Dimensionless)

$$
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\nabla p + (\mathbf{B} \cdot \nabla)\mathbf{B} + \frac{1}{Re} \nabla^2 \mathbf{v} + \mathbf{f}
$$
$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \frac{1}{Rm} \nabla^2 \mathbf{B}
$$
$$
\nabla \cdot \mathbf{v} = 0, \quad \nabla \cdot \mathbf{B} = 0
$$

Parameters:
- $Re = U L / \nu$ (Reynolds number)
- $Rm = U L / \eta$ (magnetic Reynolds number)
- $\mathbf{f}$ = large-scale forcing (maintain turbulence)

### 3.2 Energy Spectra

**Kinetic energy spectrum:**
$$
E_K(k) = \frac{1}{2} \sum_{|\mathbf{k}'| \in [k, k+dk]} |\hat{\mathbf{v}}(\mathbf{k}')|^2
$$

**Magnetic energy spectrum:**
$$
E_M(k) = \frac{1}{2} \sum_{|\mathbf{k}'| \in [k, k+dk]} |\hat{\mathbf{B}}(\mathbf{k}')|^2
$$

**Total energy:**
$$
E_{\text{total}} = \int E_K(k) dk + \int E_M(k) dk
$$

**Kolmogorov spectrum (hydrodynamic turbulence):**
$$
E(k) \propto k^{-5/3}
$$

**Iroshnikov-Kraichnan spectrum (MHD turbulence):**
$$
E(k) \propto k^{-3/2}
$$
(Debated; modern DNS shows $k^{-5/3}$ for strong $B_0$, $k^{-3/2}$ for weak $B_0$)

### 3.3 Invariants

**Ideal MHD conserves:**
- Total energy: $E = \int (\mathbf{v}^2 + \mathbf{B}^2)/2 \, d^3x$
- Magnetic helicity: $H_M = \int \mathbf{A} \cdot \mathbf{B} \, d^3x$ (for $\eta = 0$)
- Cross-helicity: $H_C = \int \mathbf{v} \cdot \mathbf{B} \, d^3x$

**Spectral method:** Monitor these to verify energy conservation (check for numerical dissipation).

---

## 4. Chebyshev Spectral Methods

### 4.1 Chebyshev Polynomials

**Definition:**
$$
T_n(x) = \cos(n \arccos x), \quad x \in [-1, 1]
$$

**Recurrence:**
$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
$$

**Orthogonality:**
$$
\int_{-1}^{1} \frac{T_m(x) T_n(x)}{\sqrt{1-x^2}} dx = \begin{cases} 0 & m \neq n \\ \pi/2 & m = n \neq 0 \\ \pi & n = 0 \end{cases}
$$

**Collocation points (Gauss-Lobatto):**
$$
x_j = \cos\left(\frac{\pi j}{N}\right), \quad j = 0, 1, \ldots, N
$$

### 4.2 Differentiation Matrices

Approximate:
$$
f(x) \approx \sum_{n=0}^N a_n T_n(x)
$$

**Derivative:**
$$
\frac{df}{dx}(x_i) = \sum_{j=0}^N D_{ij} f(x_j)
$$

where $D_{ij}$ is the Chebyshev differentiation matrix (computed once, dense matrix).

**Second derivative:**
$$
\frac{d^2 f}{dx^2}(x_i) = \sum_{j=0}^N D^{(2)}_{ij} f(x_j)
$$

### 4.3 Application: Grad-Shafranov Solver

**Axisymmetric MHD equilibrium:**
$$
\nabla^2 \psi = -\mu_0 R^2 \frac{dp}{d\psi} - \frac{1}{2} \frac{dF^2}{d\psi}
$$
where $\psi(R, Z)$ is the poloidal flux function.

**Chebyshev discretization:**
- Map $(R, Z) \in [R_{\min}, R_{\max}] \times [Z_{\min}, Z_{\max}]$ to $[-1, 1]^2$
- Expand $\psi$ on Chebyshev grid
- Compute $\nabla^2 \psi$ via differentiation matrices
- Solve nonlinear BVP (Newton iteration)

**Boundary conditions:**
- Dirichlet: $\psi = 0$ on wall
- Modify first/last rows of $D^{(2)}$

---

## 5. Adaptive Mesh Refinement (AMR)

### 5.1 Block-Structured AMR

**Idea:** Refine grid where gradients large (shocks, current sheets), coarsen elsewhere.

**Hierarchy:**
```
Level 0: Coarse grid (base)
Level 1: Refined patches (2x resolution)
Level 2: Doubly refined (4x resolution)
...
```

**Data structure:**
- Tree of nested grids
- Each level covers subset of domain
- Ghost cells for inter-level communication

### 5.2 Refinement Criteria

**Gradient-based:**
$$
\text{Refine if } \frac{|\nabla \rho|}{\rho} > \epsilon_{\text{refine}}
$$

**Current density:**
$$
\text{Refine if } |\mathbf{J}| > J_{\text{thresh}}
$$

**Löhner estimator:**
$$
\mathcal{E} = \frac{\sum_i |\Delta u_i|}{\sum_i |u_i| + \delta}
$$
where $\Delta u_i = u_{i+1} - 2u_i + u_{i-1}$ (2nd difference).

### 5.3 Prolongation and Restriction

**Prolongation (coarse → fine):**
- Interpolate coarse cell values to fine cells
- Piecewise linear or cubic interpolation

**Restriction (fine → coarse):**
- Average fine cells:
$$
U_{\text{coarse}} = \frac{1}{4} \sum_{\text{fine cells}} U_{\text{fine}}
$$
(2D; factor 8 in 3D)

**Conservative restriction:**
$$
U_{\text{coarse}} A_{\text{coarse}} = \sum_{\text{fine}} U_{\text{fine}} A_{\text{fine}}
$$
ensures mass/energy conservation.

### 5.4 Flux Correction

**Problem:** Fine/coarse interface sees flux mismatch.

**Flux-Correction Algorithm:**
1. Compute fluxes at fine level
2. Sum fine fluxes at coarse boundary
3. Replace coarse flux with summed fine flux
4. Correct conserved quantities

**Software:** Paramesh, CHOMBO, AMReX.

---

## 6. Hybrid MHD-PIC Methods

### 6.1 Motivation

Pure MHD: Fluid description, misses kinetic effects (Landau damping, wave-particle interaction).

Pure PIC: Fully kinetic, prohibitively expensive for large-scale astrophysical systems.

**Hybrid approach:**
- **Bulk plasma:** MHD (ions + electrons as single fluid)
- **Energetic particles:** PIC (cosmic rays, SEPs, suprathermal populations)

### 6.2 Coupling

**MHD → Particles:**
- Particles experience Lorentz force:
$$
\frac{d\mathbf{p}_i}{dt} = q_i (\mathbf{E} + \mathbf{v}_i \times \mathbf{B})
$$
where $\mathbf{E}$, $\mathbf{B}$ from MHD solution.

**Particles → MHD:**
- Particles contribute pressure/momentum:
$$
p_{\text{CR}} = \sum_i \frac{p_i^2}{3m_i}, \quad \mathbf{f}_{\text{CR}} = -\nabla p_{\text{CR}}
$$
- Add $\mathbf{f}_{\text{CR}}$ to momentum equation:
$$
\frac{\partial (\rho \mathbf{v})}{\partial t} + \ldots = \ldots + \mathbf{f}_{\text{CR}}
$$

### 6.3 Applications

- **Cosmic ray transport:** Diffusion, streaming instability
- **Solar energetic particles:** Acceleration at shocks
- **Planetary magnetospheres:** Ring current particles

---

## 7. SPH-MHD

### 7.1 Smoothed Particle Hydrodynamics (SPH)

**Lagrangian method:**
- Fluid represented by particles (mass elements)
- Properties smoothed over kernel:
$$
f(\mathbf{x}) = \sum_j m_j \frac{f_j}{\rho_j} W(|\mathbf{x} - \mathbf{x}_j|, h)
$$
where $W$ is smoothing kernel (e.g., cubic spline), $h$ = smoothing length.

**Equations of motion:**
$$
\frac{d\mathbf{v}_i}{dt} = -\sum_j m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right) \nabla_i W_{ij}
$$

### 7.2 MHD in SPH

**Challenge:** Magnetic field is Eulerian (grid-based), but SPH is Lagrangian.

**Approaches:**
1. **Conservative formulation:** Evolve $\mathbf{B}$ on particles, use $\nabla \cdot \mathbf{B} = 0$ cleaning
2. **Vector potential:** Evolve $\mathbf{A}$, compute $\mathbf{B} = \nabla \times \mathbf{A}$ (guarantees $\nabla \cdot \mathbf{B} = 0$)

**Tensile instability:**
- Particle clumping in magnetic pressure gradient regions
- Mitigation: artificial viscosity, improved kernels

### 7.3 Applications

- **Star formation:** Collapse of magnetized molecular clouds
- **Protoplanetary disks:** MRI, disk-planet interaction
- **Galaxy evolution:** Cosmological simulations with B-fields

**Pros:**
- Naturally Lagrangian (follows fluid)
- No advection errors
- Good for large density contrasts

**Cons:**
- Low-order accuracy (SPH kernels $\sim O(h^2)$)
- $\nabla \cdot \mathbf{B}$ errors accumulate
- Expensive neighbor search

---

## 8. Production MHD Codes

### 8.1 Athena++

**Type:** Grid-based, finite volume

**Features:**
- MHD, SRMHD, radiation MHD
- Static mesh refinement (SMR) and AMR
- Curvilinear coordinates (spherical, cylindrical)
- Constrained transport (CT) for $\nabla \cdot \mathbf{B} = 0$
- Multi-physics: dust, cosmic rays, chemistry

**Riemann solvers:** HLLC, HLLD, Roe

**Use cases:**
- Accretion disks (MRI turbulence)
- Protostellar jets
- Galaxy cluster simulations

**Website:** https://www.athena-astro.app/

### 8.2 PLUTO

**Type:** Grid-based, finite volume

**Features:**
- MHD, RMHD, RHD, HD
- Modular physics: cooling, thermal conduction, particles
- Multiple Riemann solvers
- Adaptive grids (CHOMBO AMR)
- User-friendly setup (pluto.ini config file)

**Coordinates:** Cartesian, cylindrical, spherical, curvilinear

**Use cases:**
- Astrophysical jets
- Stellar winds
- Planet-disk interaction

**Website:** http://plutocode.ph.unito.it/

### 8.3 FLASH

**Type:** Grid-based, AMR (Paramesh/CHOMBO)

**Features:**
- Compressible MHD, radiation hydro
- Nuclear burning (stellar evolution)
- Gravity (Poisson solver, tree code)
- Laser energy deposition (ICF)

**Best for:** Multi-physics simulations (SNe, ICF, stellar interiors)

**Website:** https://flash.rochester.edu/

### 8.4 Pencil Code

**Type:** Finite difference, high-order (6th order spatial)

**Features:**
- **Spectral-like accuracy** on Cartesian grids
- MHD turbulence, dynamos
- Non-ideal effects: ambipolar diffusion, Hall effect
- Dust, chemistry, radiation

**Best for:**
- Direct numerical simulation (DNS) of turbulence
- Large $Re$, $Rm$ studies
- Dynamo theory

**Website:** https://github.com/pencil-code/

### 8.5 Dedalus

**Type:** Spectral (Fourier + Chebyshev)

**Features:**
- Flexible PDE solver (user specifies equations symbolically)
- Parallelized (MPI + efficient eigensolvers)
- Time-stepping: SBDF, RK443
- Linear stability analysis (eigenvalue problems)

**Best for:**
- Convection, dynamos in spherical shells
- Stability analysis (MRI, MHD instabilities)
- Custom PDEs

**Website:** https://dedalus-project.org/

---

## 9. Python Implementation: 2D Pseudo-Spectral MHD

### 9.1 Problem Setup

Simulate 2D incompressible MHD turbulence with forcing:
$$
\frac{\partial \mathbf{v}}{\partial t} = -(\mathbf{v} \cdot \nabla)\mathbf{v} + (\mathbf{B} \cdot \nabla)\mathbf{B} - \nabla p + \nu \nabla^2 \mathbf{v} + \mathbf{f}
$$
$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}
$$

Domain: $[0, 2\pi]^2$, periodic BC, $N_x = N_y = 128$.

### 9.2 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq

# Parameters
N = 128          # Grid points
L = 2.0 * np.pi  # Domain size
nu = 1e-3        # Viscosity
eta = 1e-3       # Resistivity
dt = 0.001       # Timestep
T_end = 1.0      # End time
forcing_amp = 0.1

# Grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Wavenumbers
kx = fftfreq(N, L/(2*np.pi*N)) * 2*np.pi
ky = fftfreq(N, L/(2*np.pi*N)) * 2*np.pi
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
# K2[0,0] = 1 avoids division by zero in the pressure projection P⊥ = k/k²;
# the k=0 (mean) mode has zero divergence by definition, so setting K2=1 there
# gives P⊥ = 0, which correctly leaves the mean field unchanged
K2[0, 0] = 1.0  # Avoid division by zero

# Dealiasing mask (2/3 rule): keep only |k| ≤ N/3 (not N/2) because the product
# of two functions with max wavenumber N/3 produces modes up to 2N/3 ≤ N/2 — still
# within the Nyquist limit — so no aliasing occurs; modes beyond N/3 are zeroed out
# before IFFT to prevent aliased high-k content from contaminating lower modes
kmax = N // 3
dealias = (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)

# Initial conditions (random vorticity + magnetic field)
np.random.seed(42)
vx = np.random.randn(N, N) * 0.1
vy = np.random.randn(N, N) * 0.1
Bx = np.sin(2*X) * np.cos(Y)
By = -np.cos(X) * np.sin(2*Y)

# Project initial v and B onto their divergence-free subspaces before the first
# time step — random initial fields will generically have nonzero divergence, and
# even a small ∇·v ≠ 0 at t=0 would create a spurious pressure that grows each step
vx_hat = fftn(vx)
vy_hat = fftn(vy)
# div_v in spectral space = ik_x * vx_hat + ik_y * vy_hat: in spectral space
# differentiation is exact (no truncation error), so this divergence is the
# machine-precision representative of the continuous divergence
div_v = 1j*KX*vx_hat + 1j*KY*vy_hat
# Subtract the potential part (k * (k·v)/k²): this is the Helmholtz decomposition —
# any vector field splits uniquely into curl-free (potential) + divergence-free parts;
# subtracting the potential part leaves only the solenoidal (divergence-free) remainder
vx_hat -= 1j*KX*div_v/K2
vy_hat -= 1j*KY*div_v/K2
vx = np.real(ifftn(vx_hat))
vy = np.real(ifftn(vy_hat))

# Same divergence cleaning for B: ∇·B = 0 is a constraint required by Maxwell's
# equations; in the spectral method it is maintained at every step by the projection
# in project_divergence_free(), but ensuring it holds at t=0 prevents accumulation
# of machine-precision errors from amplifying over many time steps
Bx_hat = fftn(Bx)
By_hat = fftn(By)
div_B = 1j*KX*Bx_hat + 1j*KY*By_hat
Bx_hat -= 1j*KX*div_B/K2
By_hat -= 1j*KY*div_B/K2
Bx = np.real(ifftn(Bx_hat))
By = np.real(ifftn(By_hat))

def compute_nonlinear(vx, vy, Bx, By):
    """Compute nonlinear terms in physical space."""
    # Velocity advection
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)
    # Derivatives computed in spectral space (exact) then transformed back:
    # this is the pseudo-spectral strategy — ik*f_hat gives the exact spectral
    # derivative, which is then multiplied with the field in physical space;
    # doing the multiplication in physical space avoids the expensive convolution
    # sum that would result from doing it entirely in spectral space
    dvx_dx = np.real(ifftn(1j*KX*vx_hat))
    dvx_dy = np.real(ifftn(1j*KY*vx_hat))
    dvy_dx = np.real(ifftn(1j*KX*vy_hat))
    dvy_dy = np.real(ifftn(1j*KY*vy_hat))

    NL_vx = -(vx*dvx_dx + vy*dvx_dy)
    NL_vy = -(vx*dvy_dx + vy*dvy_dy)

    # Magnetic tension (B·∇)B: this term is what makes MHD different from
    # pure hydrodynamics — tension along field lines resists bending, which
    # is the physical mechanism behind Alfvén wave propagation and magnetic braking
    Bx_hat = fftn(Bx)
    By_hat = fftn(By)
    dBx_dx = np.real(ifftn(1j*KX*Bx_hat))
    dBx_dy = np.real(ifftn(1j*KY*Bx_hat))
    dBy_dx = np.real(ifftn(1j*KX*By_hat))
    dBy_dy = np.real(ifftn(1j*KY*By_hat))

    NL_vx += Bx*dBx_dx + By*dBx_dy
    NL_vy += Bx*dBy_dx + By*dBy_dy

    # Induction equation RHS: ∇×(v×B) written out component-wise as vx*∂B - B*∂v;
    # the anti-symmetric structure preserves magnetic helicity under ideal evolution —
    # any deviation from this form would introduce spurious helicity injection
    NL_Bx = vx*dBx_dx + vy*dBx_dy - Bx*dvx_dx - By*dvx_dy
    NL_By = vx*dBy_dx + vy*dBy_dy - Bx*dvy_dx - By*dvy_dy

    return NL_vx, NL_vy, NL_Bx, NL_By

def project_divergence_free(fx_hat, fy_hat):
    """Project to divergence-free subspace."""
    div_f = 1j*KX*fx_hat + 1j*KY*fy_hat
    fx_hat -= 1j*KX*div_f/K2
    fy_hat -= 1j*KY*div_f/K2
    return fx_hat, fy_hat

def rhs(vx, vy, Bx, By, t):
    """Compute RHS of equations."""
    NL_vx, NL_vy, NL_Bx, NL_By = compute_nonlinear(vx, vy, Bx, By)

    # Add forcing (low wavenumber)
    forcing_vx = forcing_amp * np.sin(2*X + t) * np.cos(Y)
    forcing_vy = forcing_amp * np.cos(X) * np.sin(2*Y + t)
    NL_vx += forcing_vx
    NL_vy += forcing_vy

    # Apply dealiasing mask after FFT-ing the nonlinear products: the physical-space
    # multiplication created high-k modes up to 2*kmax that would alias back into
    # the simulation wavenumbers without this mask — the dealias array zeros those
    # modes so they cannot contaminate the energy cascade toward lower wavenumbers
    NL_vx_hat = fftn(NL_vx) * dealias
    NL_vy_hat = fftn(NL_vy) * dealias
    NL_Bx_hat = fftn(NL_Bx) * dealias
    NL_By_hat = fftn(NL_By) * dealias

    # Project nonlinear velocity forcing to divergence-free subspace: even though
    # v is already divergence-free, the nonlinear term (v·∇)v may generate a
    # compressible component; projecting removes it and implicitly enforces the
    # pressure equation without needing to solve a Poisson equation explicitly
    NL_vx_hat, NL_vy_hat = project_divergence_free(NL_vx_hat, NL_vy_hat)
    NL_Bx_hat, NL_By_hat = project_divergence_free(NL_Bx_hat, NL_By_hat)

    # Add diffusion
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)
    Bx_hat = fftn(Bx)
    By_hat = fftn(By)

    # -ν*k²*v_hat is exact spectral diffusion: no spatial truncation error, unlike
    # finite differences where ∇²v ≈ (v_{i+1}-2v_i+v_{i-1})/Δx² has O(Δx²) error;
    # spectral diffusion is exact because Fourier modes are eigenfunctions of ∇²,
    # which is why spectral methods achieve exponential convergence for smooth flows
    dvx_dt_hat = NL_vx_hat - nu*K2*vx_hat
    dvy_dt_hat = NL_vy_hat - nu*K2*vy_hat
    dBx_dt_hat = NL_Bx_hat - eta*K2*Bx_hat
    dBy_dt_hat = NL_By_hat - eta*K2*By_hat

    return (np.real(ifftn(dvx_dt_hat)), np.real(ifftn(dvy_dt_hat)),
            np.real(ifftn(dBx_dt_hat)), np.real(ifftn(dBy_dt_hat)))

# Time integration (RK4)
def rk4_step(vx, vy, Bx, By, dt, t):
    k1 = rhs(vx, vy, Bx, By, t)
    k2 = rhs(vx + 0.5*dt*k1[0], vy + 0.5*dt*k1[1],
             Bx + 0.5*dt*k1[2], By + 0.5*dt*k1[3], t + 0.5*dt)
    k3 = rhs(vx + 0.5*dt*k2[0], vy + 0.5*dt*k2[1],
             Bx + 0.5*dt*k2[2], By + 0.5*dt*k2[3], t + 0.5*dt)
    k4 = rhs(vx + dt*k3[0], vy + dt*k3[1],
             Bx + dt*k3[2], By + dt*k3[3], t + dt)

    vx_new = vx + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    vy_new = vy + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    Bx_new = Bx + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    By_new = By + (dt/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    return vx_new, vy_new, Bx_new, By_new

# Energy spectrum
def compute_spectrum(vx, vy, Bx, By):
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)
    Bx_hat = fftn(Bx)
    By_hat = fftn(By)

    E_kin = 0.5*(np.abs(vx_hat)**2 + np.abs(vy_hat)**2)
    E_mag = 0.5*(np.abs(Bx_hat)**2 + np.abs(By_hat)**2)

    k_bins = np.arange(1, N//2)
    E_kin_spec = np.zeros(len(k_bins))
    E_mag_spec = np.zeros(len(k_bins))

    K_mag = np.sqrt(KX**2 + KY**2)
    for i, k in enumerate(k_bins):
        mask = (K_mag >= k) & (K_mag < k+1)
        E_kin_spec[i] = np.sum(E_kin[mask])
        E_mag_spec[i] = np.sum(E_mag[mask])

    return k_bins, E_kin_spec, E_mag_spec

# Main loop
t = 0.0
step = 0
snapshots = []

print("Running 2D spectral MHD simulation...")
while t < T_end:
    if step % 100 == 0:
        E_kin_total = 0.5*np.sum(vx**2 + vy**2)
        E_mag_total = 0.5*np.sum(Bx**2 + By**2)
        print(f"Step {step:4d}, t={t:.3f}, E_kin={E_kin_total:.4f}, E_mag={E_mag_total:.4f}")

        if len(snapshots) < 4:
            snapshots.append((t, vx.copy(), vy.copy(), Bx.copy(), By.copy()))

    vx, vy, Bx, By = rk4_step(vx, vy, Bx, By, dt, t)
    t += dt
    step += 1

# Final spectrum
k_bins, E_kin_spec, E_mag_spec = compute_spectrum(vx, vy, Bx, By)

# Plot spectra
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(k_bins, E_kin_spec, 'b-', label='Kinetic', linewidth=2)
ax.loglog(k_bins, E_mag_spec, 'r--', label='Magnetic', linewidth=2)

# Reference slopes
k_ref = np.array([2, 20])
ax.loglog(k_ref, 1e2*k_ref**(-5.0/3.0), 'k:', label=r'$k^{-5/3}$', alpha=0.5)
ax.loglog(k_ref, 1e2*k_ref**(-3.0/2.0), 'g:', label=r'$k^{-3/2}$', alpha=0.5)

ax.set_xlabel('Wavenumber k')
ax.set_ylabel('Energy E(k)')
ax.set_title('MHD Turbulence Energy Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('mhd_spectrum.png', dpi=150, bbox_inches='tight')
print("Spectrum saved: mhd_spectrum.png")
plt.close()

# Plot snapshots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, (t_snap, vx_snap, vy_snap, Bx_snap, By_snap) in enumerate(snapshots):
    ax = axes[i // 2, i % 2]
    vorticity = np.real(ifftn(-K2 * fftn(vx_snap)))  # Simplified vorticity
    im = ax.contourf(X, Y, vorticity, levels=20, cmap='RdBu_r')
    ax.streamplot(X, Y, Bx_snap, By_snap, color='k', density=0.8, linewidth=0.5)
    ax.set_title(f't = {t_snap:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='Vorticity')

plt.tight_layout()
plt.savefig('mhd_snapshots.png', dpi=150, bbox_inches='tight')
print("Snapshots saved: mhd_snapshots.png")
plt.close()
```

### 9.3 Expected Output

- **Energy spectrum:** Power-law decay (check if $k^{-5/3}$ or $k^{-3/2}$)
- **Snapshots:** Vorticity contours + magnetic field lines
- **Energy conservation:** Total energy should stay approximately constant (with slow dissipation from $\nu$, $\eta$)

---

## 10. Chebyshev Equilibrium Solver

### 10.1 1D Force Balance

Solve:
$$
\frac{d}{dx}\left(p + \frac{B^2}{2}\right) = 0
$$
for $p(x)$, $B(x)$ in $x \in [-1, 1]$ with BC: $p(-1) = 1$, $B(-1) = 0.5$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Chebyshev differentiation matrix
def cheb_diff_matrix(N):
    """Compute Chebyshev differentiation matrix."""
    x = np.cos(np.pi * np.arange(N+1) / N)
    c = np.ones(N+1)
    c[0] = c[-1] = 2.0
    c[1:-1] = 1.0
    c *= (-1)**np.arange(N+1)

    X = np.tile(x, (N+1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0/c) / (dX + np.eye(N+1))
    D -= np.diag(np.sum(D, axis=1))

    return D, x

N = 32
D, x = cheb_diff_matrix(N)

# Initial guess
p = 1.0 - 0.5*(x + 1.0)
B = 0.5 + 0.3*(x + 1.0)

# Solve force balance: d/dx(p + B^2/2) = 0
# Integrate: p + B^2/2 = const
# BC: p(-1) = 1, B(-1) = 0.5 => const = 1 + 0.5^2/2 = 1.125

const = 1.0 + 0.5**2 / 2.0
p_sol = const - B**2 / 2.0

# Verify
ptot = p_sol + B**2 / 2.0
dptot_dx = D @ ptot

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, p_sol, 'b-', linewidth=2, label='Pressure p')
axes[0].set_xlabel('x')
axes[0].set_ylabel('p')
axes[0].set_title('Pressure Profile')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(x, B, 'r-', linewidth=2, label='Magnetic Field B')
axes[1].set_xlabel('x')
axes[1].set_ylabel('B')
axes[1].set_title('Magnetic Field Profile')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(x, dptot_dx, 'g-', linewidth=2, label=r'$d(p+B^2/2)/dx$')
axes[2].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[2].set_xlabel('x')
axes[2].set_ylabel('Residual')
axes[2].set_title('Force Balance Residual')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig('chebyshev_equilibrium.png', dpi=150, bbox_inches='tight')
print("Plot saved: chebyshev_equilibrium.png")
plt.close()
```

---

## 11. AMR Concept Demonstration

### 11.1 1D Adaptive Grid

```python
import numpy as np
import matplotlib.pyplot as plt

# Function with sharp gradient
def f(x):
    return np.tanh(20*(x - 0.5))

# Initial coarse grid
x_coarse = np.linspace(0, 1, 11)
f_coarse = f(x_coarse)

# Refinement criterion: large gradient
df_dx = np.gradient(f_coarse, x_coarse)
refine_indices = np.where(np.abs(df_dx) > 5.0)[0]

# Refined grid (insert midpoints)
x_fine = []
f_fine = []
for i in range(len(x_coarse)):
    x_fine.append(x_coarse[i])
    f_fine.append(f_coarse[i])
    if i in refine_indices and i < len(x_coarse) - 1:
        x_mid = 0.5*(x_coarse[i] + x_coarse[i+1])
        x_fine.append(x_mid)
        f_fine.append(f(x_mid))

x_fine = np.array(x_fine)
f_fine = np.array(f_fine)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_exact = np.linspace(0, 1, 500)
f_exact = f(x_exact)

axes[0].plot(x_exact, f_exact, 'k-', linewidth=1, label='Exact', alpha=0.5)
axes[0].plot(x_coarse, f_coarse, 'bo-', linewidth=2, label='Coarse Grid', markersize=8)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Coarse Grid (11 points)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_exact, f_exact, 'k-', linewidth=1, label='Exact', alpha=0.5)
axes[1].plot(x_fine, f_fine, 'ro-', linewidth=2, label='AMR Grid', markersize=6)
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title(f'AMR Grid ({len(x_fine)} points)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('amr_concept.png', dpi=150, bbox_inches='tight')
print("Plot saved: amr_concept.png")
print(f"Coarse grid: {len(x_coarse)} points")
print(f"AMR grid: {len(x_fine)} points")
plt.close()
```

**Output:** AMR concentrates points in high-gradient region (around $x = 0.5$).

---

## Summary

Advanced MHD methods extend basic finite-volume approaches:

1. **Pseudo-spectral:** Exponential accuracy for smooth, periodic problems (MHD turbulence, dynamos)
2. **Dealiasing:** Essential for correct nonlinear term evaluation (2/3 rule)
3. **Chebyshev:** Non-periodic BC, ideal for boundary value problems (equilibria, stability)
4. **AMR:** Adaptive resolution for multi-scale problems (shocks, current sheets)
5. **Hybrid MHD-PIC:** Combine fluid bulk + kinetic particles (cosmic rays)
6. **SPH-MHD:** Lagrangian method for star formation, astrophysical flows
7. **Production codes:** Athena++, PLUTO, FLASH (grid), Pencil Code (high-order FD), Dedalus (spectral)

**When to use:**
- **Spectral:** Turbulence DNS, dynamos (high $Re$, $Rm$)
- **AMR:** Shocks, jets, accretion (multi-scale)
- **Chebyshev:** Equilibria, linear stability
- **Hybrid:** Cosmic ray acceleration, energetic particles

Each method has trade-offs; choose based on physics and computational resources.

---

## Practice Problems

1. **Fourier Derivative:**
   For $f(x) = \sin(3x)$ on $[0, 2\pi]$ with $N = 16$ points, compute $f'(x)$ using FFT. Compare to exact $f'(x) = 3\cos(3x)$. What is the maximum error?

2. **2/3 Dealiasing:**
   Consider $f(x) = \sin(5x)$ and $g(x) = \sin(7x)$. Their product has max wavenumber 12. If $N = 16$ (Nyquist $k_N = 8$), will aliasing occur without dealiasing? What is the aliased wavenumber?

3. **Chebyshev Points:**
   Compute the $N = 8$ Gauss-Lobatto Chebyshev points $x_j = \cos(\pi j / N)$. Plot their distribution. Why do they cluster at endpoints?

4. **Energy Spectrum Slope:**
   From the 2D spectral MHD code, vary $\nu$ and $\eta$ by factors of 2. Does the inertial range slope change? At what $Re$ do you observe a clear $k^{-5/3}$ range?

5. **AMR Efficiency:**
   For a 1D shock (Sod test), estimate the number of grid points needed for uniform grid vs 2-level AMR (refinement factor 4) to resolve the shock to 1% accuracy. Assume shock width $\sim 10$ cells.

6. **Hybrid MHD-PIC:**
   In a hybrid code, cosmic ray particles diffuse with $D = 10^{28}$ cm²/s. If the MHD grid has $\Delta x = 10^{10}$ cm, estimate the timestep from the diffusion CFL condition $\Delta t < (\Delta x)^2 / D$.

7. **SPH Kernel:**
   For the cubic spline kernel $W(r, h) \propto (1 - r/h)^3$ for $r < h$, verify that $\int W(\mathbf{r}, h) d^3r = 1$. Compute the normalization constant in 2D.

8. **Athena++ vs PLUTO:**
   Research: What Riemann solver does Athena++ use by default for MHD? Compare to PLUTO's default. Which is more diffusive?

---

**Previous:** [Relativistic MHD](./16_Relativistic_MHD.md) | **Next:** [Projects](./18_Projects.md)

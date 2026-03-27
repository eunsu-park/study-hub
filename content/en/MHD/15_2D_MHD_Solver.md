# 15. 2D MHD Solver

## Learning Objectives

- Extend 1D MHD methods to 2D using dimensional splitting and unsplit schemes
- Implement finite volume methods on 2D Cartesian grids
- Apply Constrained Transport (CT) to preserve $\nabla \cdot B = 0$ exactly
- Use staggered grids (Yee mesh) for magnetic field components
- Implement higher-order reconstruction: PLM, WENO
- Simulate benchmark problems: Orszag-Tang vortex, Kelvin-Helmholtz instability
- Understand corner transport upwind (CTU) methods

## 1. Introduction to 2D MHD

Extending MHD simulations from 1D to 2D introduces new challenges: multidimensional wave propagation, geometric source terms, and the critical requirement of preserving $\nabla \cdot B = 0$ in multiple dimensions.

### 1.1 2D MHD Equations

In 2D Cartesian coordinates $(x, y)$, the ideal MHD equations are:

```
∂U/∂t + ∂F/∂x + ∂G/∂y = 0
```

where the conserved variables are:

```
U = [ρ, ρv_x, ρv_y, ρv_z, B_x, B_y, B_z, E]ᵀ
```

The fluxes in the $x$ direction:

```
F = [ρv_x, ρv_x² + p_T - B_x²/μ₀, ρv_x v_y - B_x B_y/μ₀, ρv_x v_z - B_x B_z/μ₀,
     0, v_x B_y - v_y B_x, v_x B_z - v_z B_x,
     v_x(E + p_T) - B_x(v·B)/μ₀]ᵀ
```

and similarly for $G$ (fluxes in $y$ direction), with total pressure:

```
p_T = p + B²/(2μ₀)
```

### 1.2 Divergence Constraint

The magnetic field must satisfy:

```
∇ · B = ∂B_x/∂x + ∂B_y/∂y = 0
```

In 1D, this reduces to $\partial B_x / \partial x = 0$, which is automatically satisfied if $B_x$ is constant initially. In 2D, preserving $\nabla \cdot B = 0$ requires special numerical treatment.

**Consequences of violating $\nabla \cdot B = 0$:**
- Unphysical monopole forces
- Numerical instabilities
- Incorrect wave speeds and shock structure

### 1.3 Challenges in 2D

1. **Computational cost**: $N_x \times N_y$ cells, $\mathcal{O}(N^2)$ operations per timestep
2. **Multidimensional effects**: Corner coupling, transverse waves
3. **Divergence preservation**: Requires specialized discretization (CT, divergence cleaning, etc.)
4. **CFL condition**: Timestep limited by 2D wave propagation

## 2. Finite Volume Method in 2D

### 2.1 Cell-Centered Discretization

Divide the domain into rectangular cells $[x_{i-1/2}, x_{i+1/2}] \times [y_{j-1/2}, y_{j+1/2}]$.

Cell-averaged conserved variables:

```
U_{i,j} = (1/ΔxΔy) ∫∫ U(x,y,t) dx dy
```

### 2.2 Semi-Discrete Form

The finite volume discretization:

```
dU_{i,j}/dt = -(F_{i+1/2,j} - F_{i-1/2,j})/Δx - (G_{i,j+1/2} - G_{i,j-1/2})/Δy
```

Fluxes at cell faces are computed from Riemann solvers (HLL, HLLD, Roe, etc.).

### 2.3 Dimensional Splitting (Strang Splitting)

**Idea**: Split 2D evolution into alternating 1D sweeps.

For one timestep $\Delta t$:

1. **Half step in $x$**: Evolve using $\partial U / \partial t + \partial F / \partial x = 0$ for $\Delta t / 2$
2. **Full step in $y$**: Evolve using $\partial U / \partial t + \partial G / \partial y = 0$ for $\Delta t$
3. **Half step in $x$**: Evolve again in $x$ for $\Delta t / 2$

This is **Strang splitting** (second-order accurate in time if each 1D step is second-order).

**Advantages:**
- Reuse 1D Riemann solvers
- Simple to implement

**Disadvantages:**
- Anisotropic errors (directional bias)
- Difficult to preserve $\nabla \cdot B = 0$ across dimension splits

### 2.4 Unsplit Methods

**Corner Transport Upwind (CTU)** methods update all directions simultaneously, including transverse flux corrections.

**Algorithm** (simplified CTU):

1. Reconstruct states at cell faces in both $x$ and $y$ directions
2. Solve Riemann problems at all faces
3. Compute transverse flux corrections (e.g., upwind corner states)
4. Update conserved variables including corner coupling

CTU methods are fully multidimensional and reduce directional bias.

## 3. Constrained Transport (CT)

Constrained Transport is a numerical method that preserves $\nabla \cdot B = 0$ to machine precision by using a staggered grid for magnetic fields and evolving the magnetic field via Faraday's law in integral form.

### 3.1 Yee Mesh (Staggered Grid)

**Cell-centered quantities** (at $(x_i, y_j)$):
- $\rho, p, v_x, v_y, v_z, E$

**Face-centered magnetic field** (staggered):
- $B_x$ at $(x_{i-1/2}, y_j)$ (faces perpendicular to $x$)
- $B_y$ at $(x_i, y_{j-1/2})$ (faces perpendicular to $y$)
- $B_z$ at cell centers $(x_i, y_j)$ (if $B_z$ is present but does not affect divergence in 2D)

**Edge-centered electric field**:
- $E_z$ at $(x_i, y_j)$ (cell corners in 2D $xy$ plane)

### 3.2 Faraday's Law in Integral Form

Faraday's law:

```
∂B/∂t = -∇ × E
```

In 2D (with $B = (B_x, B_y, B_z)$ and $E_z$ the only relevant electric field component):

```
∂B_x/∂t = -∂E_z/∂y
∂B_y/∂t = ∂E_z/∂x
```

Integrate over cell faces:

```
d/dt ∫ B_x dy = -[E_z(top) - E_z(bottom)]
d/dt ∫ B_y dx = [E_z(right) - E_z(left)]
```

This naturally preserves $\nabla \cdot B = 0$ if it holds initially.

### 3.3 Electric Field Calculation

The electric field in ideal MHD:

```
E = -v × B
```

In 2D:

```
E_z = v_x B_y - v_y B_x
```

**CT algorithm**:

1. **Reconstruct primitive variables** at cell faces
2. **Solve Riemann problems** to get face-centered velocities and magnetic fields
3. **Compute electric field at cell edges** using upwinded $v$ and $B$ from adjacent faces:
   ```
   E_z(i,j) = [v_x B_y - v_y B_x]_{i,j}
   ```
   Averaging strategies: arithmetic mean, upwind, Riemann solver-based

4. **Update magnetic field** using discrete Faraday's law:
   ```
   B_x(i-1/2, j)^{n+1} = B_x(i-1/2, j)^n - Δt/Δy [E_z(i,j+1/2) - E_z(i,j-1/2)]
   B_y(i, j-1/2)^{n+1} = B_y(i, j-1/2)^n + Δt/Δx [E_z(i+1/2,j) - E_z(i-1/2,j)]
   ```

### 3.4 Divergence-Free Guarantee

Since the discrete update is derived from the integral form of Faraday's law, the discrete divergence:

```
(∇ · B)_{i,j} = [B_x(i+1/2,j) - B_x(i-1/2,j)]/Δx + [B_y(i,j+1/2) - B_y(i,j-1/2)]/Δy
```

is preserved to machine precision (if zero initially).

## 4. Higher-Order Reconstruction

### 4.1 Piecewise Linear Method (PLM)

Second-order spatial accuracy requires reconstructing the solution within each cell as linear:

```
U(x) = U_i + σ_i (x - x_i)
```

where $\sigma_i$ is the slope, estimated from neighboring cells:

```
σ_i ≈ (U_{i+1} - U_{i-1}) / (2Δx)  (centered difference)
```

**Slope limiting**: To prevent spurious oscillations near discontinuities, apply a limiter:

```
σ_i = minmod(σ_L, σ_C, σ_R)
```

where:
- $\sigma_L = (U_i - U_{i-1}) / \Delta x$
- $\sigma_C = (U_{i+1} - U_{i-1}) / (2 \Delta x)$
- $\sigma_R = (U_{i+1} - U_i) / \Delta x$

**minmod limiter**:

```
minmod(a, b, c) =
    min(|a|, |b|, |c|) * sign(a)  if sign(a) = sign(b) = sign(c)
    0                              otherwise
```

Other limiters: MC (monotonized central), van Leer, superbee.

### 4.2 WENO (Weighted Essentially Non-Oscillatory)

WENO schemes achieve high-order accuracy (5th order or higher) by using weighted combinations of multiple stencils.

**WENO5 reconstruction** (simplified):

Use 5-point stencil $\{U_{i-2}, U_{i-1}, U_i, U_{i+1}, U_{i+2}\}$ to construct three 3-point candidate polynomials, then blend them with smoothness-based weights to obtain the reconstructed value at $x_{i+1/2}$.

**Advantages**:
- High-order accuracy in smooth regions
- Non-oscillatory near discontinuities

**Disadvantages**:
- Computationally expensive (large stencil, nonlinear weights)
- Complex implementation

### 4.3 Characteristic vs. Primitive Variable Reconstruction

Reconstruction can be done in:
- **Primitive variables** $(\\rho, v_x, v_y, v_z, B_x, B_y, B_z, p)$: Simpler, but may produce unphysical states
- **Conserved variables** $U$: Guarantees conservation, but can generate oscillations
- **Characteristic variables** $W = L \cdot U$: Decouple waves, best for capturing discontinuities

For MHD, characteristic reconstruction is preferred but requires solving for eigenvectors of the Jacobian (expensive in 2D/3D).

## 5. Time Integration

### 5.1 CFL Condition in 2D

The timestep is limited by:

```
Δt ≤ CFL * min(Δx, Δy) / max(|λ|)
```

where $\lambda$ are the wave speeds (fast magnetosonic, Alfvén, slow magnetosonic, entropy).

For safety, typically $CFL \approx 0.4-0.8$.

### 5.2 Time Stepping Schemes

**Second-order Runge-Kutta (RK2)**:

```
U* = U^n + Δt L(U^n)
U^{n+1} = 0.5 U^n + 0.5 U* + 0.5 Δt L(U*)
```

where $L(U) = -(∂F/∂x + ∂G/∂y)$ is the spatial operator.

**Third-order Runge-Kutta (RK3)** (TVD-RK3):

```
U^{(1)} = U^n + Δt L(U^n)
U^{(2)} = 3/4 U^n + 1/4 U^{(1)} + 1/4 Δt L(U^{(1)})
U^{n+1} = 1/3 U^n + 2/3 U^{(2)} + 2/3 Δt L(U^{(2)})
```

RK3 is commonly used with WENO schemes.

## 6. Benchmark Problem: Orszag-Tang Vortex

The **Orszag-Tang vortex** is a standard 2D MHD test problem featuring the formation of shocks, current sheets, and complex vortex structures.

### 6.1 Initial Conditions

Domain: $[0, 1] \times [0, 1]$ with periodic boundary conditions.

```
ρ = γ²
p = γ
v_x = -sin(2πy)
v_y = sin(2πx)
v_z = 0
B_x = -sin(2πy) / sqrt(4π)
B_y = sin(4πx) / sqrt(4π)
B_z = 0
```

where $\gamma = 5/3$ (adiabatic index).

This setup has $\beta \sim 1$ (comparable plasma and magnetic pressure) and creates a turbulent cascade.

### 6.2 Evolution

As the vortex evolves:
- Shocks form and interact
- Current sheets develop at boundaries between oppositely directed fields
- Magnetic reconnection occurs
- Vorticity cascades to smaller scales

**Key diagnostics**:
- Density contours: Show shock structures
- Magnetic field lines: Illustrate reconnection and topology changes
- Current density $|j_z| = |\nabla \times B|_z$: Locates current sheets

### 6.3 Expected Results

At $t \approx 0.5$, a complex pattern of shocks and vortices emerges. High-resolution simulations (512² or higher) are required to resolve fine structures.

## 7. Kelvin-Helmholtz Instability in MHD

The **Kelvin-Helmholtz (KH) instability** arises at the interface between two fluids in relative shear motion.

### 7.1 Setup

Domain: $[0, 1] \times [-1, 1]$ with periodic boundary in $x$, reflecting or periodic in $y$.

**Velocity shear layer**:

```
v_x(y) = -V₀ tanh(y / a)
v_y = δv₀ sin(2πx)  (perturbation)
```

where $V_0$ is the shear velocity, $a$ is the shear layer thickness, and $\delta v_0 \ll V_0$ is the perturbation amplitude.

**Magnetic field** (uniform in $x$ direction):

```
B_x = B₀
B_y = 0
```

### 7.2 Linear Stability Analysis

In the absence of magnetic field, the KH growth rate for mode $k$ is:

```
γ_KH ~ k V₀ / 2  (for thin shear layer)
```

With magnetic field, magnetic tension stabilizes short wavelengths. The dispersion relation:

```
γ² = k² V₀² - k² v_A²
```

where $v_A = B_0 / \sqrt{\mu_0 \rho}$ is the Alfvén speed.

**Stabilization condition**:

```
B₀ > sqrt(μ₀ ρ) V₀  (v_A > V₀)
```

If $v_A > V_0$, the KH mode is suppressed.

### 7.3 Numerical Simulation

Initial conditions:

```
ρ = 1
p = 1
v_x = -V₀ tanh(y / a)
v_y = δv₀ sin(2πx)
B_x = B₀
B_y = 0
```

Typical parameters: $V_0 = 1$, $a = 0.1$, $\delta v_0 = 0.01$, $B_0 = 0$ to $2$.

**Observations**:
- $B_0 = 0$: Classic KH rolls develop
- $B_0 = 0.5$: KH growth slowed, but still unstable
- $B_0 = 2$: KH suppressed, magnetic tension stabilizes

## 8. Python Implementation: 2D MHD Solver with CT

### 8.1 Data Structures

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class MHD2D:
    def __init__(self, Nx, Ny, Lx, Ly, gamma=5/3):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / Nx, Ly / Ny
        self.gamma = gamma

        # Cell centers
        self.x = np.linspace(0.5*self.dx, Lx - 0.5*self.dx, Nx)
        self.y = np.linspace(0.5*self.dy, Ly - 0.5*self.dy, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Staggered grid for magnetic field (CT):
        # Bx lives on x-faces (i±1/2, j) and By on y-faces (i, j±1/2) so that
        # the discrete divergence ∂_x Bx + ∂_y By uses values from opposite faces
        # of the same cell — this is the geometric core of CT: the update from
        # Faraday's law on these staggered positions preserves ∇·B = 0 exactly
        self.x_Bx = np.linspace(0, Lx, Nx+1)
        self.y_By = np.linspace(0, Ly, Ny+1)

        # Conserved variables (cell-centered)
        self.rho = np.ones((Nx, Ny))
        self.mx = np.zeros((Nx, Ny))
        self.my = np.zeros((Nx, Ny))
        self.mz = np.zeros((Nx, Ny))
        self.E = np.ones((Nx, Ny))

        # Bx has Nx+1 values in x because it lives on cell faces, not centers;
        # there are always one more face than cell in each direction — this extra
        # index is what allows the CT update to touch both sides of every cell
        self.Bx = np.zeros((Nx+1, Ny))  # Face-centered in x
        self.By = np.zeros((Nx, Ny+1))  # Face-centered in y
        self.Bz = np.zeros((Nx, Ny))    # Cell-centered (if needed)

        # Ez at cell corners (Nx+1, Ny+1): the discrete curl of Ez via differences
        # at the 4 surrounding corners is what updates Bx and By in Faraday's law —
        # placing Ez at corners ensures each face flux is updated consistently
        self.Ez = np.zeros((Nx+1, Ny+1))

        self.t = 0.0

    def primitive_variables(self):
        """Compute primitive variables from conserved."""
        vx = self.mx / self.rho
        vy = self.my / self.rho
        vz = self.mz / self.rho

        # Average magnetic field to cell centers
        Bx_cc = 0.5 * (self.Bx[:-1, :] + self.Bx[1:, :])
        By_cc = 0.5 * (self.By[:, :-1] + self.By[:, 1:])
        Bz_cc = self.Bz

        B2 = Bx_cc**2 + By_cc**2 + Bz_cc**2
        p = (self.gamma - 1) * (self.E - 0.5 * self.rho * (vx**2 + vy**2 + vz**2) - 0.5 * B2)

        return self.rho, vx, vy, vz, p, Bx_cc, By_cc, Bz_cc

    def compute_dt(self, CFL=0.4):
        """Compute timestep based on CFL condition."""
        rho, vx, vy, vz, p, Bx, By, Bz = self.primitive_variables()

        # Fast magnetosonic speed cf = √(cs² + vA²) is used — not just cs or vA
        # alone — because cf is the fastest signal speed in MHD; the CFL condition
        # requires that no information propagates more than one cell per timestep,
        # and the fast magnetosonic wave is the envelope of all wave modes
        cs = np.sqrt(self.gamma * p / rho)
        va = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
        cf = np.sqrt(cs**2 + va**2)

        # Separate dt_x and dt_y constraints: in 2D both must be satisfied
        # simultaneously — taking min() enforces the strictest of both directions
        # so that waves in neither direction can outrun the update
        dt_x = self.dx / np.max(np.abs(vx) + cf)
        dt_y = self.dy / np.max(np.abs(vy) + cf)

        return CFL * min(dt_x, dt_y)

    def check_divergence(self):
        """Check divergence of B (should be ~0)."""
        div_B = (self.Bx[1:, :] - self.Bx[:-1, :]) / self.dx + \
                (self.By[:, 1:] - self.By[:, :-1]) / self.dy
        return np.max(np.abs(div_B))
```

### 8.2 CT Update for Magnetic Field

```python
def update_magnetic_field_CT(self):
    """Update magnetic field using Constrained Transport."""
    # Compute electric field Ez at cell corners (edges in 2D)
    # Ez = vx * By - vy * Bx

    # Need velocities and B fields at corners
    # Simple averaging (can be improved with Riemann solver values)

    # Average vx to y-edges: we need vx at corners to compute Ez = vx*By - vy*Bx;
    # averaging adjacent cell values to the shared face is the simplest consistent
    # interpolation that does not introduce spurious directional bias
    vx_avg_y = 0.5 * (self.mx[:-1, :] / self.rho[:-1, :] + self.mx[1:, :] / self.rho[1:, :])
    vx_avg_y = np.pad(vx_avg_y, ((1,0), (0,0)), mode='wrap')  # Periodic

    # Average vy to x-edges
    vy_avg_x = 0.5 * (self.my[:, :-1] / self.rho[:, :-1] + self.my[:, 1:] / self.rho[:, 1:])
    vy_avg_x = np.pad(vy_avg_x, ((0,0), (1,0)), mode='wrap')

    # Bx lives on x-faces so must be averaged to corners; the ±1/2 face neighbor
    # in the y-direction gives the corner value — this average is simple but
    # production codes replace it with upwind estimates from Riemann solver results
    # to reduce numerical dissipation at current sheets
    Bx_corner = 0.5 * (self.Bx[:, :-1] + self.Bx[:, 1:])
    Bx_corner = np.pad(Bx_corner, ((0,0), (0,1)), mode='wrap')

    # Average By to corners (from faces)
    By_corner = 0.5 * (self.By[:-1, :] + self.By[1:, :])
    By_corner = np.pad(By_corner, ((0,1), (0,0)), mode='wrap')

    # Ez = vx*By - vy*Bx at corners: this is the z-component of -v×B (Ohm's law
    # in ideal MHD), and placing it at corners is the key to CT — each face flux
    # is updated by the curl of Ez at the two corners bounding that face, so the
    # discrete ∇·B = 0 is preserved as a telescoping sum of corner Ez values
    self.Ez = vx_avg_y * By_corner - vy_avg_x * Bx_corner

    # ∂Bx/∂t = -∂Ez/∂y: differencing Ez at adjacent corners gives a consistent
    # discrete Faraday's law — the sign ensures that a positive Ez at the upper
    # corner reduces Bx (consistent with the right-hand rule of ∂B/∂t = -∇×E)
    dt = self.compute_dt()
    self.Bx[:, :] -= dt / self.dy * (self.Ez[:, 1:] - self.Ez[:, :-1])

    # ∂By/∂t = +∂Ez/∂x (opposite sign to Bx update, from the anti-symmetric curl)
    self.By[:, :] += dt / self.dx * (self.Ez[1:, :] - self.Ez[:-1, :])
```

### 8.3 Full 2D MHD Solver (Simplified)

Due to the complexity of a full 2D MHD solver with Riemann solvers, CT, and higher-order reconstruction, here we provide a conceptual skeleton. Production codes like Athena, FLASH, and Pluto implement these in thousands of lines.

**Simplified algorithm**:

```python
def step(self, dt):
    """Single timestep using operator splitting."""
    # Step 1: Half step in x direction
    self.step_x(dt / 2)

    # Step 2: Full step in y direction
    self.step_y(dt)

    # Step 3: Half step in x direction
    self.step_x(dt / 2)

    # Update magnetic field using CT
    self.update_magnetic_field_CT()

    self.t += dt

def step_x(self, dt):
    """1D sweep in x direction (simplified)."""
    # For each j, solve 1D Riemann problems along x
    for j in range(self.Ny):
        # Extract 1D slice
        rho_1d = self.rho[:, j]
        mx_1d = self.mx[:, j]
        # ... (other variables)

        # Reconstruct, solve Riemann problem, update
        # (Reuse 1D MHD solver)

        # Update conserved variables
        # self.rho[:, j] = ...
        pass

def step_y(self, dt):
    """1D sweep in y direction."""
    # Similar to step_x but along y
    pass
```

**Note**: Implementing a robust 2D MHD solver requires:
- 1D Riemann solver (HLL, HLLD, etc.)
- Reconstruction (PLM, WENO)
- CT electric field calculation from Riemann solver face states
- Boundary conditions
- Source terms (gravity, etc., if applicable)

For educational purposes, we demonstrate the Orszag-Tang vortex setup and a simple forward-Euler update.

### 8.4 Orszag-Tang Vortex Setup

```python
def init_orszag_tang(mhd, gamma=5/3):
    """Initialize Orszag-Tang vortex."""
    X, Y = mhd.X, mhd.Y

    # ρ = γ² and p = γ come from the Orszag-Tang normalization: at these values
    # the Alfvén Mach number is O(1), placing the problem in the interesting
    # regime where kinetic and magnetic energies are comparable — this β~1 choice
    # ensures both shocks and current sheets form during the evolution
    mhd.rho[:, :] = gamma**2
    p = gamma * np.ones_like(mhd.rho)

    # Velocity
    vx = -np.sin(2 * np.pi * Y)
    vy = np.sin(2 * np.pi * X)
    vz = np.zeros_like(vx)

    mhd.mx = mhd.rho * vx
    mhd.my = mhd.rho * vy
    mhd.mz = mhd.rho * vz

    # Magnetic field initialized on the staggered grid so that CT immediately
    # has a divergence-free starting point; evaluating Bx at face positions
    # x_{i-1/2} rather than cell centers ensures the initial ∇·B is identically
    # zero in the discrete CT sense — not just approximately zero
    X_Bx, Y_Bx = np.meshgrid(mhd.x_Bx, mhd.y, indexing='ij')
    # The 1/√(4π) normalization in Gaussian units makes the Alfvén speed vA = 1
    # at the reference density, which simplifies comparison between runs
    mhd.Bx[:, :] = -np.sin(2 * np.pi * Y_Bx) / np.sqrt(4 * np.pi)

    # By uses sin(4πX) (double frequency compared to Bx) to break the symmetry
    # between x and y directions — a purely symmetric initial field would remain
    # symmetric forever and miss the off-axis reconnection that makes the test interesting
    X_By, Y_By = np.meshgrid(mhd.x, mhd.y_By, indexing='ij')
    mhd.By[:, :] = np.sin(4 * np.pi * X_By) / np.sqrt(4 * np.pi)

    # Bz (cell-centered, zero)
    mhd.Bz[:, :] = 0.0

    # Average staggered Bx,By to cell centers before computing total energy:
    # the energy must be consistent with the staggered B representation to avoid
    # an unphysical energy mismatch at t=0 that would drive spurious flows
    Bx_cc = 0.5 * (mhd.Bx[:-1, :] + mhd.Bx[1:, :])
    By_cc = 0.5 * (mhd.By[:, :-1] + mhd.By[:, 1:])
    B2 = Bx_cc**2 + By_cc**2

    mhd.E = p / (gamma - 1) + 0.5 * mhd.rho * (vx**2 + vy**2 + vz**2) + 0.5 * B2

# Initialize
Nx, Ny = 128, 128
mhd = MHD2D(Nx, Ny, Lx=1.0, Ly=1.0, gamma=5/3)
init_orszag_tang(mhd)

print(f"Orszag-Tang vortex initialized on {Nx}×{Ny} grid")
print(f"Initial div(B) max: {mhd.check_divergence():.3e}")
```

### 8.5 Simple Forward Euler Update (for demonstration)

```python
def simple_update(mhd, dt):
    """
    Simplified forward Euler update (NOT recommended for production).
    For demonstration only.
    """
    rho, vx, vy, vz, p, Bx, By, Bz = mhd.primitive_variables()

    # Compute fluxes (very simplified, ignoring Riemann solver)
    # This will NOT capture shocks correctly!

    # Flux in x direction (at i+1/2, j)
    Fx_rho = mhd.mx  # rho * vx
    # ... (other flux components)

    # Update conserved variables (forward Euler, very crude)
    # drho/dt = -(dFx/dx + dFy/dy)

    # For proper implementation, use Riemann solver at each face
    # This is just a placeholder

    # Update magnetic field via CT
    mhd.update_magnetic_field_CT()

    mhd.t += dt

# Note: This is NOT a working solver, just a skeleton
# For actual simulation, use established codes or implement full Riemann solver
```

### 8.6 Visualization

```python
def plot_orszag_tang(mhd):
    """Plot density and magnetic field."""
    rho, vx, vy, vz, p, Bx, By, Bz = mhd.primitive_variables()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Density
    im1 = ax1.contourf(mhd.X, mhd.Y, rho, levels=50, cmap='viridis')
    ax1.set_title(f'Density at t = {mhd.t:.3f}', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=ax1)

    # Magnetic field lines
    ax2.contourf(mhd.X, mhd.Y, np.sqrt(Bx**2 + By**2), levels=50, cmap='plasma')
    ax2.streamplot(mhd.X.T, mhd.Y.T, Bx.T, By.T, color='white', linewidth=0.5, density=1.5)
    ax2.set_title(f'Magnetic Field at t = {mhd.t:.3f}', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'orszag_tang_t{mhd.t:.3f}.png', dpi=150)
    plt.show()

plot_orszag_tang(mhd)
```

### 8.7 Kelvin-Helmholtz Instability Setup

```python
def init_kelvin_helmholtz(mhd, V0=1.0, a=0.1, dv=0.01, B0=0.0):
    """Initialize Kelvin-Helmholtz instability."""
    X, Y = mhd.X, mhd.Y

    # Density (uniform)
    mhd.rho[:, :] = 1.0

    # Pressure (uniform)
    p = 1.0 * np.ones_like(mhd.rho)

    # tanh profile gives smooth shear with characteristic thickness a;
    # using a=0.1 keeps the shear layer resolved on the grid while being thin
    # enough to have a fast KH growth rate γ_KH ~ k*V0/2 with large k available
    vx = -V0 * np.tanh(Y / a)
    # Small sinusoidal perturbation seeds the fastest-growing KH mode (k = 2π);
    # dv << V0 ensures the system starts in the linear regime so we can observe
    # exponential growth before nonlinear saturation sets in
    vy = dv * np.sin(2 * np.pi * X)
    vz = np.zeros_like(vx)

    mhd.mx = mhd.rho * vx
    mhd.my = mhd.rho * vy
    mhd.mz = mhd.rho * vz

    # Uniform Bx field aligned with the flow: magnetic tension opposes bending
    # of field lines by KH rolls — stabilization occurs when vA = B0/√(μ₀ρ) > V0,
    # so varying B0 from 0 to 2 spans the transition from unstable to stable regimes
    mhd.Bx[:, :] = B0
    mhd.By[:, :] = 0.0
    mhd.Bz[:, :] = 0.0

    # Total energy
    Bx_cc = 0.5 * (mhd.Bx[:-1, :] + mhd.Bx[1:, :])
    By_cc = 0.5 * (mhd.By[:, :-1] + mhd.By[:, 1:])
    B2 = Bx_cc**2 + By_cc**2

    mhd.E = p / (mhd.gamma - 1) + 0.5 * mhd.rho * (vx**2 + vy**2) + 0.5 * B2

# Initialize KH instability
mhd_kh = MHD2D(Nx=256, Ny=256, Lx=1.0, Ly=2.0, gamma=5/3)
init_kelvin_helmholtz(mhd_kh, V0=1.0, a=0.1, dv=0.01, B0=0.5)

print(f"Kelvin-Helmholtz initialized: V0=1.0, B0=0.5")
print(f"Alfvén speed: vA = {0.5 / np.sqrt(1.0):.2f} (should stabilize if vA > V0)")
```

### 8.8 Production-Quality Codes

For actual research, use established MHD codes:

**Athena++**: https://github.com/PrincetonUniversity/athena
- C++, modern AMR (adaptive mesh refinement)
- MHD, radiation, GR options
- CT for divergence-free B

**PLUTO**: http://plutocode.ph.unito.it/
- Modular, supports various physics modules
- MHD, relativistic MHD, radiation
- Multiple Riemann solvers, CT

**FLASH**: https://flash.rochester.edu/
- Large-scale astrophysical simulations
- AMR, MHD, hydrodynamics
- Widely used for supernova, star formation

**Example**: Running Athena for Orszag-Tang vortex:

```bash
# Athena input file (athinput.orszag_tang)
<problem>
problem_id = OrszagTang
gamma = 1.666667

<mesh>
nx1 = 256
nx2 = 256
x1min = 0.0
x1max = 1.0
x2min = 0.0
x2max = 1.0
ix1_bc = periodic
ox1_bc = periodic
ix2_bc = periodic
ox2_bc = periodic

<hydro>
evolution = mhd

<time>
tlim = 0.5
```

Run:

```bash
./athena -i athinput.orszag_tang
```

## 9. Advanced Topics

### 9.1 Adaptive Mesh Refinement (AMR)

AMR dynamically refines the grid in regions of interest (shocks, current sheets) and coarsens elsewhere, saving computational cost.

**Challenges in MHD AMR**:
- Preserving $\nabla \cdot B = 0$ across refinement boundaries
- Prolongation and restriction operators for staggered fields

**Solutions**:
- Flux correction at coarse-fine boundaries
- Divergence-preserving prolongation (e.g., Balsara's method)

### 9.2 Divergence Cleaning

Alternative to CT: Allow non-zero $\nabla \cdot B$ but add a damping mechanism.

**Hyperbolic divergence cleaning** (Dedner et al. 2002):

Add an auxiliary scalar field $\psi$ and evolve:

```
∂B/∂t + ∇ψ = ... (usual MHD terms)
∂ψ/∂t + c_h² ∇·B = -c_h² ψ / τ
```

where $c_h$ is a cleaning speed (typically $c_h \sim c_{fast}$) and $\tau$ is a damping timescale.

This propagates divergence errors out of the domain as waves.

**Pros**: Works on unstructured grids, easier to implement than CT
**Cons**: Does not guarantee $\nabla \cdot B = 0$ exactly, requires tuning parameters

### 9.3 Positivity-Preserving Methods

Ensuring $\rho > 0$ and $p > 0$ after each update is critical for stability.

**Techniques**:
- Limiting reconstructed states to physical range
- Adjusting fluxes to prevent negative density/pressure
- Positivity-preserving Riemann solvers

### 9.4 High-Order Methods

Beyond second-order:
- **WENO5**: 5th-order reconstruction
- **Discontinuous Galerkin (DG)**: High-order within cells, coupled via fluxes
- **Spectral methods**: For smooth problems (not ideal for shocks)

Trade-off: Higher order reduces numerical dissipation but increases cost and complexity.

## 10. Summary

This lesson covered advanced numerical techniques for 2D MHD:

1. **2D MHD equations**: Extension from 1D, 8 conserved variables in 2D
2. **Finite volume method**: Cell-centered discretization, semi-discrete form
3. **Dimensional splitting**: Strang splitting (second-order), reuses 1D solvers
4. **Unsplit methods**: CTU for multidimensional coupling
5. **Constrained Transport**: Staggered grid (Yee mesh), preserves $\nabla \cdot B = 0$ exactly
6. **Higher-order reconstruction**: PLM (limiters), WENO (5th order)
7. **Orszag-Tang vortex**: Benchmark problem, turbulent MHD
8. **Kelvin-Helmholtz instability**: Magnetic field stabilizes shear layer
9. **Python implementation**: Skeleton code for 2D MHD with CT

For production simulations, use established codes (Athena, PLUTO, FLASH) that have been extensively tested and optimized.

## Practice Problems

1. **CFL condition**: For a 2D MHD simulation with $\Delta x = \Delta y = 0.01$, fast magnetosonic speed $c_f = 2$, and maximum flow velocity $|v| = 1$, calculate the maximum timestep for $CFL = 0.5$.

2. **Divergence preservation**: Explain why the standard finite volume method for updating $B$ does not preserve $\nabla \cdot B = 0$, but Constrained Transport does. Sketch the Yee mesh and show where $B_x$, $B_y$, and $E_z$ are located.

3. **Orszag-Tang vortex**: Why is the Orszag-Tang vortex a good test problem for 2D MHD codes? What physical processes does it test (list at least 3)?

4. **Kelvin-Helmholtz stabilization**: For a shear layer with $V_0 = 2$ m/s, $\rho = 1$ kg/m³, calculate the minimum magnetic field $B_0$ required to suppress the KH instability (i.e., $v_A \geq V_0$). Express in Tesla (assume vacuum permeability $\mu_0 = 4\pi \times 10^{-7}$ H/m).

5. **PLM reconstruction**: Given cell-centered values $U_{i-1} = 1.0$, $U_i = 1.5$, $U_{i+1} = 2.5$, compute the slope $\sigma_i$ using (a) centered difference, (b) minmod limiter. What are the left and right states at the cell interface $i+1/2$?

6. **CT electric field**: In Constrained Transport, the electric field $E_z$ at a cell corner is computed from velocities and magnetic fields at adjacent faces. Write the formula for $E_z(i, j)$ in terms of $v_x$, $v_y$, $B_x$, $B_y$ at the four surrounding face centers. (Assume simple averaging.)

7. **Dimensional splitting error**: Strang splitting ($L_x^{1/2} L_y L_x^{1/2}$) is second-order accurate in time. What is the order of accuracy if you use simple splitting ($L_x L_y$)? Why is Strang splitting preferred?

8. **WENO advantage**: WENO schemes are 5th-order accurate in smooth regions but reduce to lower order near discontinuities. Why is this beneficial compared to always using 2nd-order PLM?

9. **AMR refinement criterion**: In an MHD simulation, you want to refine the grid near current sheets. Propose a criterion based on $|\\nabla \\times B|$ to trigger refinement. Write the condition mathematically.

10. **Computational cost**: Compare the computational cost (operations per timestep) of a 2D MHD simulation on a $256 \times 256$ grid versus a 1D simulation with 256 cells, assuming the same physics and Riemann solver. Estimate the ratio (ignoring constants).

---

**Previous**: [Space Weather MHD](./14_Space_Weather.md) | **Next**: [Relativistic MHD](./16_Relativistic_MHD.md)

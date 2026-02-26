# 9. Dynamo Theory

## Learning Objectives

By the end of this lesson, you should be able to:

- Explain the fundamental dynamo problem and why planets and stars need a dynamo mechanism to maintain magnetic fields
- Derive and interpret the magnetic induction equation in the MHD approximation
- Understand anti-dynamo theorems (Cowling, Zeldovich) and their implications for dynamo requirements
- Analyze kinematic dynamo models including the stretch-twist-fold mechanism
- Apply mean-field theory to understand large-scale dynamo action (α-effect, β-effect, α-Ω dynamos)
- Distinguish between kinematic and dynamical dynamo regimes and understand saturation mechanisms
- Implement numerical models of simple dynamos and analyze their growth rates

## 1. The Dynamo Problem

### 1.1 Why Do We Need Dynamos?

The Earth, Sun, and many other astrophysical objects possess large-scale magnetic fields that have persisted for billions of years. This presents a fundamental problem:

**The Free Decay Problem:**

In the absence of any generation mechanism, magnetic fields in conducting fluids decay on a resistive timescale:

```
τ_η = L²/η
```

where:
- `L` is the characteristic length scale
- `η = 1/(μ₀σ)` is the magnetic diffusivity
- `σ` is the electrical conductivity

For the Earth's core:
- `L ~ 10⁶ m`
- `η ~ 1-2 m²/s`
- `τ_η ~ 10⁴ - 10⁵ years`

This is much shorter than the Earth's age (~4.5 billion years), yet the geomagnetic field has existed for at least 3.5 billion years (from paleomagnetic evidence). Therefore, **there must be an active generation mechanism**.

**Definition:** A **dynamo** is a mechanism that converts kinetic energy of a conducting fluid into magnetic energy, maintaining a magnetic field against resistive dissipation.

### 1.2 The Magnetic Induction Equation

The evolution of the magnetic field in a moving conducting fluid is governed by the **magnetic induction equation**, derived from Maxwell's equations and Ohm's law in the MHD approximation:

```
∂B/∂t = ∇ × (v × B) + η∇²B
```

In component form:

```
∂B_i/∂t + v_j ∂B_i/∂x_j = B_j ∂v_i/∂x_j + η ∂²B_i/∂x_j∂x_j
```

**Physical interpretation:**

1. **Advection term** `v·∇B`: magnetic field is frozen into and carried by the fluid
2. **Stretching term** `B·∇v`: magnetic field lines are stretched by velocity gradients (shear, strain)
3. **Diffusion term** `η∇²B`: resistive dissipation

The relative importance of advection/stretching vs. diffusion is measured by the **magnetic Reynolds number**:

```
Rm = UL/η
```

where:
- `U` is characteristic velocity
- `L` is characteristic length scale

**Dynamo regimes:**
- `Rm ≪ 1`: diffusion-dominated, magnetic field decays
- `Rm ≫ 1`: advection-dominated, dynamo action possible
- Typically, `Rm_critical ~ O(10)` for dynamo onset

### 1.3 Energy Considerations

The magnetic energy evolution is:

```
dE_B/dt = ∫ B·(∇×(v×B)) dV - ∫ η J² dV
```

where:
- First term: work done by fluid motion (can be positive → amplification)
- Second term: Ohmic dissipation (always negative)

For a sustained dynamo:

```
∫ B·(∇×(v×B)) dV ≥ ∫ η J² dV
```

The dynamo converts kinetic energy to magnetic energy at a rate sufficient to overcome resistive losses.

## 2. Anti-Dynamo Theorems

Before understanding how dynamos work, it's crucial to know what **cannot** work. Anti-dynamo theorems place fundamental constraints on dynamo mechanisms.

### 2.1 Cowling's Theorem (1934)

**Statement:** An axisymmetric magnetic field (independent of azimuthal angle φ in cylindrical or spherical coordinates) cannot be maintained by dynamo action.

**Proof sketch (cylindrical coordinates):**

For an axisymmetric field:

```
B = B_r(r,z,t) e_r + B_φ(r,z,t) e_φ + B_z(r,z,t) e_z
```

The toroidal component B_φ satisfies:

```
∂B_φ/∂t = r(B·∇)(v_φ/r) + (1/r)∂(rB_r)/∂r v_φ + ∂B_z/∂z v_φ + η(∇²B_φ - B_φ/r²)
```

For a neutral line where `B_φ = 0` at some surface, the right-hand side must also vanish there. Along this neutral line, `∂B_φ/∂t = 0`, so the field cannot grow through it. By continuity, if `B_φ = 0` initially on a surface, it remains zero → no dynamo.

**Implication:** The magnetic field **must** have non-axisymmetric components, even if the mean field is axisymmetric (e.g., Earth's dipole-dominated field arises from time-averaged non-axisymmetric fluctuations).

### 2.2 Zeldovich's Theorem (1956)

**Statement:** A purely two-dimensional flow (velocity and fields independent of one coordinate, say z) cannot sustain a dynamo.

**Proof sketch:**

In 2D, all field lines and streamlines lie in parallel planes. Consider magnetic field lines in the x-y plane. The induction equation becomes:

```
∂B/∂t = ∇ × (v × B) + η∇²B
```

with `v = v(x,y,t)` and `B = B(x,y,t)`. The field can be written as:

```
B = ∇ × (ψ(x,y,t) e_z)  (for B_z = 0)
```

or with a z-component:

```
B = B_z(x,y,t) e_z + ∇ × (ψ(x,y,t) e_z)
```

For the poloidal part (ψ), the induction equation gives:

```
∂ψ/∂t = v·∇ψ + η∇²ψ
```

This is a pure advection-diffusion equation with no source term → ψ decays. The z-component can be stretched but not regenerated.

**Implication:** Three-dimensional flows are **necessary** for dynamo action.

### 2.3 Summary of Constraints

From anti-dynamo theorems:

1. **Need 3D flow**: at least one component must vary in all three directions
2. **Need non-axisymmetric components**: even if mean field is axisymmetric
3. **Need sufficient complexity**: simple flows (e.g., uniform rotation) cannot dynamo

These theorems guide the search for dynamo mechanisms: we need flows with helicity, differential rotation, or convective turbulence.

## 3. Kinematic Dynamo Theory

### 3.1 The Kinematic Approximation

In **kinematic dynamo theory**, the velocity field `v(x,t)` is prescribed (given), and we solve the induction equation for the magnetic field evolution, **ignoring the Lorentz force back-reaction** on the flow.

```
∂B/∂t = ∇ × (v × B) + η∇²B    (v prescribed)
```

This is valid when:

```
B² / (μ₀ρv²) ≪ 1
```

i.e., magnetic energy ≪ kinetic energy.

**Eigenvalue problem:**

Assume solutions of the form:

```
B(x,t) = b(x) exp(γt)
```

where `γ` is the growth rate (complex in general). Substituting:

```
γ b = ∇ × (v × b) + η∇²b
```

This is an eigenvalue problem:
- If `Re(γ) > 0` for any eigenmode: **dynamo action** (field grows)
- If `Re(γ) < 0` for all modes: field decays

### 3.2 Stretch-Twist-Fold Mechanism

The generic mechanism for kinematic dynamo action:

**1. Stretching:** Velocity shear stretches magnetic field lines, increasing field strength (frozen-in theorem: `B/ρ` increases with stretching).

**2. Twisting:** Helical or rotational flows twist field lines, converting poloidal ↔ toroidal components.

**3. Folding:** Reconnnection or topological rearrangement prevents indefinite stretching, creating new field topology.

**Cycle:**

```
Poloidal field B_p
    ↓ (differential rotation → stretching)
Toroidal field B_t
    ↓ (helical motion → twisting)
New poloidal field B_p'
    ↓ (reconnection/folding)
Enhanced poloidal field
```

If the net amplification per cycle exceeds diffusive losses, the field grows exponentially → dynamo.

### 3.3 Ponomarenko Dynamo (1973)

An analytically tractable example: helical flow in a cylindrical conductor.

**Setup:**
- Infinite cylinder of radius `a`, conductor inside
- Velocity: `v = (0, rΩ, U)` in cylindrical coordinates (r, φ, z)
  - Rotation: `v_φ = rΩ`
  - Translation: `v_z = U`
- Boundary: `B` continuous at r=a, decays outside

**Induction equation:**

```
∂B/∂t = ∇ × (v × B) + η∇²B
```

Seek normal modes:

```
B ~ exp(γt + imφ + ikz)
```

where:
- `m` is azimuthal wavenumber
- `k` is axial wavenumber

**Dispersion relation (simplified for |m|=1, small k):**

```
γ ≈ kU - (k² + 1/a²)η  for small Rm
```

At sufficiently large `Rm = Ua/η`, the first term (advection) overcomes diffusion:

```
Rm_critical ~ O(10)  (depends on k,m)
```

**Physical picture:**
- Helical flow twists field lines into helices
- Axial advection (U) reinforces the twisting faster than diffusion can smooth it out
- Growth occurs for modes with wavelength ~ a

### 3.4 Roberts Flow Dynamo

A 2D periodic cellular flow (in x-y plane) with a vertical (z) component can also dynamo.

**Roberts flow (Roberts, 1972):**

```
v_x = V₀ sin(ky) cos(kx)
v_y = -V₀ cos(ky) sin(kx)
v_z = √2 V₀ sin(kx) sin(ky)
```

This flow has:
- Cellular structure with vortices
- Helicity: `⟨v·(∇×v)⟩ ≠ 0`
- Kinetic helicity drives α-effect (see mean-field theory)

**Dynamo properties:**
- Critical `Rm_c ~ 5` (very efficient)
- Fast growth rates
- Used as a test case for numerical codes

## 4. Mean-Field Dynamo Theory

### 4.1 Reynolds Decomposition

For turbulent flows (e.g., stellar convection zones), the flow and fields have both mean and fluctuating components:

```
v = ⟨v⟩ + u    (⟨u⟩ = 0)
B = ⟨B⟩ + b    (⟨b⟩ = 0)
```

where `⟨·⟩` denotes an ensemble or spatial average.

**Goal:** Derive an equation for the **mean field** `⟨B⟩` alone, parameterizing the effect of small-scale turbulence.

### 4.2 Mean-Field Induction Equation

Averaging the induction equation:

```
∂⟨B⟩/∂t = ∇ × (⟨v⟩ × ⟨B⟩) + ∇ × ℰ + η∇²⟨B⟩
```

where the **mean electromotive force (EMF)** is:

```
ℰ = ⟨u × b⟩
```

This is the mean electric field induced by small-scale turbulent motions. The challenge is to relate `ℰ` to `⟨B⟩`.

### 4.3 The α-Effect

**Assumption:** For homogeneous, isotropic turbulence with small Rm (fluctuations), linear closure:

```
ℰ ≈ α⟨B⟩ - β∇×⟨B⟩
```

where:
- `α`: alpha coefficient (pseudo-scalar, changes sign under parity)
- `β`: turbulent diffusivity (scalar)

**α-effect derivation (Steenbeck, Krause, Rädler, 1966):**

For helical turbulence with correlation time `τ_c` and velocity `u_rms`:

```
α ≈ -(1/3) τ_c ⟨u·(∇×u)⟩
   = -(1/3) τ_c ⟨h⟩
```

where `⟨h⟩ = ⟨u·(∇×u)⟩` is the **kinetic helicity**.

**Physical interpretation:**
- Helical turbulence has a preferred handedness (cyclonic convection in rotating systems)
- Twisting of field lines by helical eddies converts toroidal → poloidal (or vice versa)
- α > 0: right-handed helicity
- α < 0: left-handed helicity

**β-effect:**

```
β ≈ (1/3) τ_c u_rms²
```

This is an **enhanced diffusivity** due to turbulent mixing. The effective diffusivity is:

```
η_eff = η + β
```

In stellar convection zones, `β ≫ η`, so turbulent diffusion dominates.

### 4.4 α-Ω Dynamos

In rotating, differentially rotating systems (e.g., Sun, planets), the mean flow has:
- Differential rotation: `⟨v⟩ = rΩ(r,θ) e_φ` (toroidal)
- α-effect from helical turbulence

**α-Ω dynamo cycle (in spherical coordinates):**

1. **Ω-effect:** Differential rotation shears poloidal field `⟨B_p⟩` into toroidal field `⟨B_t⟩`:

```
∂⟨B_φ⟩/∂t ≈ r sin(θ) (⟨B_r⟩ ∂Ω/∂r + ⟨B_θ⟩/r ∂Ω/∂θ)
```

2. **α-effect:** Helical turbulence regenerates poloidal from toroidal:

```
∂⟨B_p⟩/∂t ≈ ∇ × (α⟨B_t⟩ e_φ)
```

**Feedback loop:**

```
⟨B_p⟩ → (Ω-effect) → ⟨B_t⟩ → (α-effect) → ⟨B_p⟩'
```

If the net amplification per cycle exceeds diffusion, the field grows → dynamo.

### 4.5 α² Dynamos

If differential rotation is weak or absent, but helicity is strong, an **α² dynamo** can operate:

```
∂⟨B⟩/∂t = ∇ × (α⟨B⟩) + η_eff ∇²⟨B⟩
```

Both poloidal → toroidal and toroidal → poloidal conversions are driven by α.

**Dynamo number:**

For α² dynamos in a sphere of radius R:

```
D_α = α R / η_eff
```

Dynamo onset typically at `|D_α| ~ 10`.

For α-Ω dynamos:

```
D_αΩ = (α ΔΩ R³) / η_eff²
```

where `ΔΩ` is the differential rotation rate. Onset at `|D_αΩ| ~ O(1)`.

### 4.6 Solar α-Ω Dynamo

**Application to the Sun:**

- **Differential rotation (Ω):** Measured by helioseismology:
  - Equator rotates faster than poles: `Ω(θ)`
  - Radial shear near tachocline (base of convection zone): `∂Ω/∂r`

- **α-effect:** Cyclonic convection in rotating frame → helicity
  - Northern hemisphere: α < 0 (predominant)
  - Southern hemisphere: α > 0

**Solar cycle:**
- Period: ~11 years (sunspot cycle), ~22 years (magnetic polarity cycle)
- Toroidal field generated at tachocline by Ω-effect
- Poloidal field regenerated by α-effect (or Babcock-Leighton mechanism: tilted sunspot pairs)
- Equatorward propagation of toroidal field (butterfly diagram)
- Poleward migration of poloidal field

**Challenges:**
- α-quenching at high Rm (see dynamical effects)
- Magnetic buoyancy: strong toroidal fields become unstable and rise → sunspots
- Flux transport: meridional circulation modulates cycle

## 5. Dynamical Dynamo Theory

### 5.1 The Lorentz Force Back-Reaction

In the kinematic regime, the magnetic field grows exponentially. But as `B² / (μ₀ρv²) → O(1)`, the **Lorentz force** becomes important:

```
ρ(∂v/∂t + v·∇v) = -∇p + J×B + ρν∇²v
```

The term `J×B` modifies the flow, which in turn affects `∂B/∂t`. This is the **dynamical regime**.

**Saturation:** The field growth slows and eventually saturates at a level where:

```
Input power (from flow) = Ohmic dissipation
```

Typically:

```
B_sat² / (2μ₀) ~ ε_B × ρv²/2
```

where `ε_B` is the efficiency of energy conversion (often `ε_B ~ 0.01 - 0.1`).

### 5.2 α-Quenching

In mean-field theory, the α-effect is reduced as the magnetic field grows:

**Simple quenching formula:**

```
α(B) = α₀ / (1 + (B_eq/B_*)²)
```

where:
- `α₀`: kinematic alpha
- `B_eq = √(μ₀ρ) u_rms`: equipartition field
- `B_*`: quenching field strength

**Catastrophic quenching:**

At very high magnetic Reynolds number `Rm → ∞`, α-quenching becomes severe:

```
α(B) ~ α₀ / Rm
```

This implies the dynamo would shut off in the `Rm → ∞` limit, which is unphysical for astrophysical objects. This led to intense debate and exploration of solutions:
- Magnetic helicity fluxes (boundary effects)
- Shear-driven dynamos (less reliant on α)
- Large-scale dynamo from inverse cascade

**Current understanding:**
- For closed boundaries: catastrophic quenching is a real issue
- For open boundaries (stellar surfaces, disk coronae): helicity fluxes alleviate quenching
- In highly turbulent systems: dynamo may be predominantly small-scale

### 5.3 Geodynamo

**Earth's dynamo:**

- Location: Liquid outer core (r = 3480 - 6371 km from center)
- Composition: Iron-nickel alloy, σ ~ 10⁶ S/m
- Convection driven by: cooling and solidification of inner core (compositional + thermal buoyancy)
- Rotation: Ω = 7.3 × 10⁻⁵ rad/s (Coriolis-dominated)

**Regime:**
- `Rm ~ 10² - 10³` (turbulent)
- Ekman number `E = ν/(ΩL²) ~ 10⁻¹⁵` (rapid rotation)
- Magnetic Prandtl number `Pm = ν/η ~ 10⁻⁵` (small viscosity)

**Dynamo mechanism:**
- Convection in rapidly rotating sphere → helical flow
- α-effect from cyclonic vortices
- Differential rotation from thermal wind balance
- α-Ω or α² dynamo, depending on strength of differential rotation

**Numerical simulations:**
- Glatzmaier-Roberts (1995): first 3D dynamo simulation reproducing geomagnetic reversals
- Modern codes: MagIC, Rayleigh, Parody
- Challenges: cannot reach true geophysical parameters (E too small)

### 5.4 Solar Dynamo Revisited

With back-reaction included:

- **Tachocline Ω-effect:** Generates strong toroidal field `B_φ ~ 10⁴ G`
- **Magnetic buoyancy:** Toroidal flux tubes rise due to magnetic buoyancy:

```
ρg = (B²/2μ₀) / H_p
```

where `H_p` is pressure scale height. Instability when `B² / (2μ₀) ~ ρc_s²`.

- **Flux emergence:** Rising flux tubes form sunspots at surface
- **Babcock-Leighton mechanism:** Tilted bipolar sunspots (Joy's law) →  poloidal field via surface diffusion and flux transport
- **Meridional circulation:** Equatorward at surface, poleward at base → flux transport dynamo

**Interface dynamo vs. distributed dynamo:**
- **Interface:** Ω-effect at tachocline, α-effect in convection zone (separate layers)
- **Distributed:** Dynamo throughout convection zone
- Current consensus: likely interface or flux-transport dynamo

## 6. Numerical Methods for Dynamo Simulations

### 6.1 Spectral Methods

For periodic domains or spherical geometries, **spectral methods** are highly efficient.

**Fourier representation:**

```
B(x,t) = Σ_k B̂_k(t) exp(ik·x)
```

The induction equation in Fourier space:

```
∂B̂_k/∂t = ik × (v̂×B̂)_k - ηk² B̂_k
```

The convolution `(v̂×B̂)_k` can be computed via FFT:
1. Transform `v̂_k, B̂_k → v(x), B(x)` (inverse FFT)
2. Compute `v×B` in real space
3. Transform back: `v×B → (v̂×B̂)_k` (forward FFT)

**Spherical harmonics:**

For spherical geometry (e.g., stars, planets):

```
B(r,θ,φ,t) = Σ_lm B̂_lm(r,t) Y_lm(θ,φ)
```

where `Y_lm` are spherical harmonics. Coupled with radial discretization (finite differences or Chebyshev polynomials).

### 6.2 Time Stepping

**Explicit schemes (e.g., Runge-Kutta):**

```
B^(n+1) = B^n + Δt × RHS(B^n, v^n)
```

Stability constraint (CFL):

```
Δt ≤ min(Δx / |v|, Δx² / η)
```

**Implicit schemes (e.g., Crank-Nicolson):**

Treat diffusion implicitly to avoid restrictive `Δt ~ Δx²` constraint:

```
(B^(n+1) - B^n)/Δt = (1/2)[RHS(B^(n+1)) + RHS(B^n)]
```

Requires solving a linear system each timestep, but allows larger Δt.

### 6.3 Incompressibility Constraint

The condition `∇·B = 0` must be maintained numerically. Methods:

**1. Vector potential:**

```
B = ∇ × A
```

Divergence-free by construction. Evolve `A` via:

```
∂A/∂t = v × B - ∇ψ + η∇²A
```

where `ψ` is a gauge.

**2. Projection method:**

After each timestep, project `B` onto divergence-free space:

```
B ← B - ∇(∇⁻²(∇·B))
```

In Fourier space: `B̂_k ← B̂_k - k(k·B̂_k)/k²`.

**3. Constrained transport (CT):**

Discretize B on cell faces, ensuring ∇·B = 0 to machine precision.

## 7. Python Implementations

### 7.1 α-Ω Mean-Field Dynamo (1D)

Simplified 1D model in radius `r`, assuming axisymmetry and mean fields:

```python
import numpy as np
import matplotlib.pyplot as plt

def alpha_omega_dynamo_1d():
    """
    1D α-Ω mean-field dynamo model.

    Equations (in cylindrical r-z, suppress z for 1D):
      ∂B_φ/∂t = r ∂Ω/∂r B_r + η ∂²B_φ/∂r²
      ∂B_r/∂t = ∂/∂r(α B_φ) + η ∂²B_r/∂r²

    Simplified to 1D in radius with periodic or no-flux boundaries.
    """
    # Parameters
    Nr = 100
    r_max = 1.0
    r = np.linspace(0, r_max, Nr)
    dr = r[1] - r[0]

    # Differential rotation profile: Ω(r) = Ω0(1 - r²)
    # The r² shape mimics solar-like differential rotation: fastest at the
    # center (analogous to the equator), slower at the edge (analogous to
    # the poles).  The shear dΩ/dr = -2Ω0 r is the driver of the Ω-effect.
    Omega0 = 1.0
    Omega = Omega0 * (1 - r**2)
    dOmega_dr = -2 * Omega0 * r

    # Alpha profile: α(r) = α0 sin(πr)
    # The sin(πr) profile vanishes at both boundaries so that the α-effect
    # is confined to the interior — physically, helicity is generated by
    # rotating convection in the bulk, not at the walls where boundary
    # conditions would contaminate the physics.
    alpha0 = 0.1
    alpha = alpha0 * np.sin(np.pi * r)

    # Magnetic diffusivity
    eta = 0.01

    # Time stepping
    dt = 0.001
    Nt = 5000

    # Initialize fields
    B_phi = np.zeros(Nr)
    B_r = np.zeros(Nr)

    # Initial perturbation
    # A single-point seed at the midpoint avoids introducing a preferred
    # large-scale mode by hand; the dynamo must amplify whatever normal
    # mode has the highest growth rate from this localized initial condition.
    B_r[Nr//2] = 0.01

    # Storage for plotting
    B_phi_hist = []
    B_r_hist = []
    times = []

    # Time evolution
    for n in range(Nt):
        # Compute second derivatives (finite differences)
        d2B_phi = np.zeros(Nr)
        d2B_r = np.zeros(Nr)

        d2B_phi[1:-1] = (B_phi[2:] - 2*B_phi[1:-1] + B_phi[:-2]) / dr**2
        d2B_r[1:-1] = (B_r[2:] - 2*B_r[1:-1] + B_r[:-2]) / dr**2

        # Boundary conditions: no-flux (∂B/∂r = 0)
        # Neumann conditions mimic the physical constraint that no current
        # crosses the boundary; they are weaker than Dirichlet (B=0) and
        # allow the field profile to settle naturally near the wall.
        d2B_phi[0] = d2B_phi[1]
        d2B_phi[-1] = d2B_phi[-2]
        d2B_r[0] = d2B_r[1]
        d2B_r[-1] = d2B_r[-2]

        # Ω-effect: generates B_φ from B_r
        # This is the key source term: differential rotation (dΩ/dr) stretches
        # radial field lines azimuthally, converting poloidal flux (B_r)
        # into toroidal flux (B_φ) — the first half of the α-Ω cycle.
        omega_term = r * dOmega_dr * B_r

        # α-effect: generates B_r from B_φ
        # The α-effect is computed as a spatial gradient ∂(αB_φ)/∂r so that
        # it correctly drives a curl of the mean EMF (ε = αB̄), which is
        # what the mean-field induction equation requires to regenerate
        # poloidal field from toroidal — closing the dynamo loop.
        alpha_term = np.zeros(Nr)
        alpha_term[1:-1] = (alpha[2:] * B_phi[2:] - alpha[:-2] * B_phi[:-2]) / (2*dr)

        # Update equations
        dB_phi_dt = omega_term + eta * d2B_phi
        dB_r_dt = alpha_term + eta * d2B_r

        B_phi += dt * dB_phi_dt
        B_r += dt * dB_r_dt

        # Store snapshots
        if n % 100 == 0:
            B_phi_hist.append(B_phi.copy())
            B_r_hist.append(B_r.copy())
            times.append(n * dt)

    # Plot evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for i in range(0, len(times), len(times)//10):
        ax1.plot(r, B_phi_hist[i], label=f't={times[i]:.2f}')
        ax2.plot(r, B_r_hist[i], label=f't={times[i]:.2f}')

    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Toroidal field $B_\\phi$')
    ax1.set_title('α-Ω Dynamo: Toroidal Field Evolution')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Radial field $B_r$')
    ax2.set_title('α-Ω Dynamo: Radial Field Evolution')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('alpha_omega_dynamo_1d.png', dpi=150)
    plt.show()

    # Growth rate analysis
    B_total = [np.sqrt(np.mean(Bp**2 + Br**2)) for Bp, Br in zip(B_phi_hist, B_r_hist)]

    plt.figure(figsize=(10, 6))
    plt.semilogy(times, B_total, 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Total field energy (RMS)')
    plt.title('α-Ω Dynamo: Exponential Growth')
    plt.grid(True)
    plt.savefig('alpha_omega_growth.png', dpi=150)
    plt.show()

    # Estimate growth rate
    if len(times) > 10:
        log_B = np.log(np.array(B_total[5:]))  # Exclude initial transient
        t_fit = np.array(times[5:])
        coeffs = np.polyfit(t_fit, log_B, 1)
        growth_rate = coeffs[0]
        print(f"Estimated growth rate γ: {growth_rate:.4f}")

    return r, B_phi_hist, B_r_hist, times

# Run simulation
alpha_omega_dynamo_1d()
```

### 7.2 Ponomarenko Dynamo Dispersion Relation

Calculate the growth rate vs. wavenumber for the Ponomarenko dynamo:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def ponomarenko_dispersion():
    """
    Solve the Ponomarenko dynamo dispersion relation.

    For helical flow in cylinder:
      v_φ = rΩ, v_z = U

    Simplified dispersion (for small k, |m|=1):
      γ ≈ kU - (k² + π²/a²)η

    More accurate: solve eigenvalue problem numerically.
    """
    # Parameters
    a = 1.0  # Cylinder radius
    Omega = 1.0  # Rotation rate
    U = 1.0  # Axial velocity
    eta_vals = np.array([0.01, 0.02, 0.05, 0.1])  # Magnetic diffusivities

    k_vals = np.linspace(0.1, 5.0, 100)  # Axial wavenumber

    plt.figure(figsize=(10, 6))

    for eta in eta_vals:
        Rm = U * a / eta
        gamma = np.zeros_like(k_vals)

        for i, k in enumerate(k_vals):
            # Simplified growth rate
            gamma[i] = k * U - (k**2 + (np.pi/a)**2) * eta

        plt.plot(k_vals, gamma, label=f'Rm = {Rm:.1f} (η={eta})')

    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Axial wavenumber k')
    plt.ylabel('Growth rate γ')
    plt.title('Ponomarenko Dynamo Dispersion Relation')
    plt.legend()
    plt.grid(True)
    plt.savefig('ponomarenko_dispersion.png', dpi=150)
    plt.show()

    # Find critical Rm
    print("\nCritical Magnetic Reynolds Numbers:")
    for k in [1.0, 2.0, 3.0]:
        # At marginal stability: γ = 0
        # 0 = kU - (k² + π²/a²)η
        # η_c = kU / (k² + π²/a²)
        eta_c = k * U / (k**2 + (np.pi/a)**2)
        Rm_c = U * a / eta_c
        print(f"  k = {k:.1f}: Rm_c = {Rm_c:.2f}")

ponomarenko_dispersion()
```

### 7.3 Solar Butterfly Diagram Simulation

Simulate the latitudinal migration of toroidal field in an α-Ω dynamo:

```python
import numpy as np
import matplotlib.pyplot as plt

def solar_butterfly_diagram():
    """
    Simplified solar butterfly diagram from α-Ω dynamo.

    2D model in (θ, t) where θ is latitude.

    Equations:
      ∂B_φ/∂t = C_Ω ∂²Ω/∂θ² B_θ + η ∂²B_φ/∂θ²
      ∂B_θ/∂t = C_α α(θ) B_φ + η ∂²B_θ/∂θ²

    Use profiles:
      Ω(θ) ~ 1 + δΩ cos²(θ)  (equator faster)
      α(θ) ~ cos(θ)  (sign changes across equator)
    """
    # Parameters
    Ntheta = 100
    theta = np.linspace(-np.pi/2, np.pi/2, Ntheta)  # Latitude
    dtheta = theta[1] - theta[0]

    # Differential rotation: Ω(θ) = Ω0(1 + δΩ cos²θ)
    # The cos²θ profile matches helioseismology: equator rotates ~20% faster
    # than the poles.  The second derivative d²Ω/dθ² drives the Ω-effect
    # (latitudinal shear ≡ the dominant toroidal field source in the Sun).
    Omega0 = 1.0
    delta_Omega = 0.2
    Omega = Omega0 * (1 + delta_Omega * np.cos(theta)**2)
    d2Omega_dtheta2 = -2 * delta_Omega * Omega0 * (np.cos(theta)**2 - np.sin(theta)**2)

    # Alpha effect: α(θ) = α0 cos(θ)
    # The cos(θ) dependence encodes the physics of rotating convection:
    # Coriolis-induced helicity is maximum at the equator and vanishes at
    # the poles, and it changes sign across the equator — which is why the
    # butterfly diagram is antisymmetric about the equatorial plane.
    alpha0 = 0.5
    alpha = alpha0 * np.cos(theta)

    # Coefficients
    C_Omega = 10.0  # Ω-effect strength
    C_alpha = 1.0   # α-effect strength
    eta = 0.1       # Diffusivity

    # Time stepping
    dt = 0.01
    Nt = 2000

    # Initialize fields
    B_phi = np.zeros(Ntheta)
    B_theta = np.zeros(Ntheta)

    # Initial perturbation at mid-latitudes
    # A Gaussian seed at mid-latitude (θ ~ π/4) mimics the observed onset
    # of each new solar cycle at ~30° latitude; starting at the equator
    # would suppress the simulation because α(0) ≠ 0 but d²Ω/dθ²|₀ ≈ 0.
    B_theta += 0.01 * np.exp(-((theta - np.pi/4)**2) / 0.1)

    # Storage
    B_phi_hist = np.zeros((Nt//10, Ntheta))
    times = np.zeros(Nt//10)

    # Time evolution
    for n in range(Nt):
        # Second derivatives
        d2B_phi = np.zeros(Ntheta)
        d2B_theta = np.zeros(Ntheta)

        d2B_phi[1:-1] = (B_phi[2:] - 2*B_phi[1:-1] + B_phi[:-2]) / dtheta**2
        d2B_theta[1:-1] = (B_theta[2:] - 2*B_theta[1:-1] + B_theta[:-2]) / dtheta**2

        # Boundary: zero at poles
        # B = 0 at the poles is the physical condition that no toroidal flux
        # threads the rotation axis; it forces the butterfly wings to end
        # rather than accumulate indefinitely at high latitudes.
        d2B_phi[0] = 0
        d2B_phi[-1] = 0
        d2B_theta[0] = 0
        d2B_theta[-1] = 0

        # Ω-effect
        omega_term = C_Omega * d2Omega_dtheta2 * B_theta

        # α-effect
        alpha_term = C_alpha * alpha * B_phi

        # Update
        dB_phi_dt = omega_term + eta * d2B_phi
        dB_theta_dt = alpha_term + eta * d2B_theta

        B_phi += dt * dB_phi_dt
        B_theta += dt * dB_theta_dt

        # Store
        if n % 10 == 0:
            B_phi_hist[n//10, :] = B_phi
            times[n//10] = n * dt

    # Plot butterfly diagram
    theta_deg = np.degrees(theta)

    plt.figure(figsize=(12, 6))
    plt.contourf(times, theta_deg, B_phi_hist.T, levels=50, cmap='RdBu_r')
    plt.colorbar(label='Toroidal field $B_\\phi$')
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Latitude (degrees)')
    plt.title('Solar Butterfly Diagram (α-Ω Dynamo Simulation)')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.savefig('butterfly_diagram.png', dpi=150)
    plt.show()

solar_butterfly_diagram()
```

### 7.4 Kinematic Dynamo Growth Rate Calculation

General framework for computing growth rates in kinematic dynamos:

```python
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

def kinematic_dynamo_eigenvalue():
    """
    Compute eigenvalues of the kinematic dynamo operator.

    Discretize the induction equation:
      ∂B/∂t = ∇×(v×B) + η∇²B

    in Fourier space for periodic domain.

    Eigenvalue problem: γ b = L b
    where L is the linear operator.
    """
    # Simplified 1D model for illustration
    # Consider B(x,t) in periodic domain [0, 2π]

    N = 32  # Number of Fourier modes
    k = np.fft.fftfreq(N, d=2*np.pi/N) * 2 * np.pi  # Wavenumbers

    # Prescribed velocity: v(x) = V0 sin(x)
    V0 = 1.0
    eta = 0.01

    # In Fourier space, multiplication by v becomes convolution
    # For simplicity, use a simple shear flow: v = V0 x̂
    # Then (v×B) has components involving derivatives

    # Construct operator matrix (simplified for 1D scalar case)
    # This is a toy model; real dynamos need full 3D vector treatment

    L = np.zeros((N, N), dtype=complex)

    for i in range(N):
        # Diagonal: diffusion term
        L[i, i] = -eta * k[i]**2

        # Off-diagonal: advection/stretching (coupling between modes)
        if i > 0:
            L[i, i-1] = 1j * V0 * k[i]  # Simplified coupling

    # Compute eigenvalues
    eigenvalues, eigenvectors = eig(L)

    # Growth rates are real parts
    growth_rates = np.real(eigenvalues)
    frequencies = np.imag(eigenvalues)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=growth_rates, cmap='RdYlGn')
    plt.colorbar(label='Growth rate Re(γ)')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xlabel('Re(γ)')
    plt.ylabel('Im(γ)')
    plt.title('Eigenvalue Spectrum')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.stem(np.arange(N), growth_rates)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Mode number')
    plt.ylabel('Growth rate Re(γ)')
    plt.title('Growth Rates by Mode')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dynamo_eigenvalues.png', dpi=150)
    plt.show()

    max_growth_idx = np.argmax(growth_rates)
    print(f"Maximum growth rate: {growth_rates[max_growth_idx]:.4f}")
    print(f"Corresponding frequency: {frequencies[max_growth_idx]:.4f}")

    if growth_rates[max_growth_idx] > 0:
        print("Dynamo action detected!")
    else:
        print("No dynamo action (all modes decay).")

kinematic_dynamo_eigenvalue()
```

## 8. Summary

**Dynamo theory** provides the framework for understanding how astrophysical magnetic fields are generated and maintained:

1. **The dynamo problem:** Magnetic fields decay on resistive timescales much shorter than astrophysical ages → active generation needed.

2. **Induction equation:** `∂B/∂t = ∇×(v×B) + η∇²B` governs field evolution, with competition between advection/stretching and diffusion.

3. **Anti-dynamo theorems:**
   - **Cowling:** No axisymmetric dynamo
   - **Zeldovich:** No 2D dynamo
   - Implication: need 3D, non-axisymmetric flows

4. **Kinematic dynamos:** Prescribed velocity, solve for B growth.
   - **Stretch-twist-fold** mechanism
   - **Ponomarenko dynamo:** helical flow in cylinder
   - **Roberts flow:** cellular flow with helicity

5. **Mean-field theory:**
   - **α-effect:** helical turbulence regenerates poloidal from toroidal (and vice versa)
   - **β-effect:** turbulent diffusion
   - **α-Ω dynamos:** differential rotation + α-effect (Sun, planets)
   - **α² dynamos:** α-effect alone

6. **Dynamical dynamos:**
   - Lorentz force back-reaction saturates field growth
   - **α-quenching:** reduces α as B increases
   - **Catastrophic quenching:** severe reduction at high Rm (resolved by helicity fluxes)

7. **Applications:**
   - **Geodynamo:** convection in Earth's outer core, α-Ω or α² mechanism
   - **Solar dynamo:** α-Ω at tachocline, 11/22-year cycle, butterfly diagram
   - Stellar and galactic dynamos

8. **Numerical methods:** spectral, finite-difference, vector potential, constrained transport.

Understanding dynamos is crucial for explaining planetary magnetism, stellar activity cycles, and the magnetization of galaxies and the early universe.

## Practice Problems

1. **Free Decay Timescale:** Calculate the magnetic diffusion timescale for:
   - Earth's core: `L = 10⁶ m`, `η = 2 m²/s`
   - Sun's convection zone: `L = 2×10⁸ m`, `η = 10⁴ m²/s` (turbulent)
   - Compare to their ages.

2. **Magnetic Reynolds Number:** For the solar convection zone with `v ~ 100 m/s`, `L ~ 10⁸ m`, `η ~ 10⁴ m²/s`, compute `Rm`. Is dynamo action possible?

3. **Cowling's Theorem:** Consider a purely toroidal field `B = B_φ(r,z,t) e_φ`. Show that the induction equation requires sources from poloidal field to sustain `B_φ`.

4. **Ponomarenko Growth Rate:** For a cylinder with `a = 1 m`, `U = 1 m/s`, `Ω = 1 rad/s`, `η = 0.05 m²/s`, estimate the growth rate for `k = 1 m⁻¹` using the simplified formula.

5. **α-Effect Estimate:** For convective turbulence with `u_rms = 10 m/s`, correlation time `τ_c = 10⁴ s`, and helicity `⟨h⟩ = 10⁻³ m/s²`, estimate α.

6. **α-Ω Dynamo Number:** In a spherical shell of radius `R = 10⁸ m`, with `α = 1 m/s`, `ΔΩ = 10⁻⁶ rad/s`, `η_eff = 10⁴ m²/s`, compute the dynamo number `D_αΩ`. Is dynamo expected?

7. **Equipartition Field:** For a flow with `ρ = 10³ kg/m³`, `v = 100 m/s`, estimate the equipartition magnetic field strength.

8. **Python Exercise:** Modify the α-Ω 1D code to include α-quenching: `α(B) = α₀/(1 + B²/B_eq²)`. Observe the transition from exponential growth to saturation.

9. **Butterfly Diagram Analysis:** From the simulation, measure the period of oscillations and the equatorward propagation speed. How do they depend on `C_Ω` and `C_α`?

10. **Advanced:** Implement a simple 2D kinematic dynamo using Fourier spectral methods. Prescribe a Roberts flow and solve for magnetic field evolution. Compute the growth rate and compare to literature values.

---

**Previous:** [MHD Turbulence](./08_MHD_Turbulence.md) | **Next:** [Turbulent Dynamo](./10_Turbulent_Dynamo.md)

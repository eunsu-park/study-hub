# 10. Turbulent Dynamo

## Learning Objectives

By the end of this lesson, you should be able to:

- Distinguish between small-scale (fluctuation) and large-scale (mean-field) turbulent dynamos
- Understand Kazantsev theory and the kinematic growth of magnetic fields in turbulence
- Explain the role of magnetic Prandtl number (Pm) in dynamo action
- Analyze magnetic helicity conservation and its constraints on large-scale dynamo growth
- Describe saturation mechanisms and the transition from kinematic to dynamic regimes
- Understand numerical simulation approaches (DNS, LES) for MHD turbulence
- Implement models of small-scale dynamo growth and helicity evolution

## 1. Introduction to Turbulent Dynamos

### 1.1 Turbulence-Driven Magnetic Field Generation

In many astrophysical environments—stellar interiors, accretion disks, the interstellar medium, galaxy clusters—the flows are highly **turbulent**. Turbulent dynamos differ from laminar dynamos in several key ways:

1. **Broad spectrum of scales:** Turbulence involves motion across a wide range of length scales, from the energy injection scale `L` down to the dissipation scale (Kolmogorov scale `η_K` or resistive scale `η_R`).

2. **Stochastic nature:** Turbulent flows are chaotic and time-dependent, requiring statistical description.

3. **Multiple dynamo mechanisms:** Both **small-scale dynamo** (fluctuation fields) and **large-scale dynamo** (mean fields) can operate simultaneously.

**Key questions:**
- What is the critical magnetic Reynolds number `Rm_c` for dynamo onset?
- How does the magnetic field saturate?
- What is the structure of the magnetic field (intermittent, filamentary, smooth)?
- How does the magnetic energy spectrum `E_B(k)` compare to the kinetic energy spectrum `E_K(k)`?

### 1.2 Small-Scale vs. Large-Scale Dynamos

**Small-scale (fluctuation) dynamo:**
- Amplifies magnetic field at scales **comparable to or smaller than** the turbulent forcing scale
- Driven by turbulent stretching and folding
- Does not require helicity or large-scale shear
- Produces tangled, intermittent magnetic structures
- Relevant for: ISM, galaxy clusters, early universe

**Large-scale (mean-field) dynamo:**
- Generates magnetic field at scales **larger than** the turbulent forcing scale
- Requires helicity (e.g., from rotation and stratification) or large-scale shear
- Produces coherent, organized fields (e.g., galactic spirals, solar dipole)
- Constrained by magnetic helicity conservation
- Relevant for: galaxies, stars, planets

Both can operate simultaneously, but the large-scale dynamo is more constrained and slower.

## 2. Small-Scale Dynamo Theory

### 2.1 Kazantsev Theory (1968)

**Kazantsev** developed a statistical theory for the kinematic growth of magnetic fields in a random, short-correlated (in time) velocity field.

**Assumptions:**
- Velocity field `v(x,t)` is a Gaussian random field
- Correlation time `τ_c ≪ τ_η = ℓ²/η` (short-correlated, or delta-correlated in time)
- Velocity correlation function:

```
⟨v_i(x,t) v_j(x',t')⟩ = δ(t - t') K_{ij}(x - x')
```

- Isotropic, homogeneous turbulence

**Induction equation:**

```
∂B/∂t = ∇×(v×B) + η∇²B
```

In the kinematic regime, `v` is prescribed.

**Magnetic field correlator:**

Define the two-point magnetic correlation tensor:

```
M_{ij}(r,t) = ⟨B_i(x,t) B_j(x+r,t)⟩
```

In isotropic turbulence, `M_{ij}` depends only on `|r|` and can be decomposed into:

```
M_{ij}(r,t) = M_N(r,t) (δ_{ij} - r_i r_j / r²) + M_L(r,t) r_i r_j / r²
```

where `M_N` and `M_L` are the transverse and longitudinal correlation functions.

**Kazantsev equation:**

For the scalar correlation function `M(r,t) = ⟨B(x,t)·B(x+r,t)⟩`, Kazantsev derived a diffusion-like equation in `r`-space (for delta-correlated velocity):

```
∂M/∂t = (1/r^{d-1}) ∂/∂r [r^{d-1} (D(r) ∂M/∂r - v(r) M)]
```

where:
- `d` is spatial dimension (usually `d=3`)
- `D(r)` is the diffusion coefficient in `r`-space, related to velocity correlations
- `v(r)` is a drift term

For short-correlation time and in the limit `r → 0`:

```
D(r) ≈ D₀ r²
v(r) ≈ v₀ r
```

where `D₀` and `v₀` are constants depending on the velocity spectrum.

**Exponential growth:**

Seek solutions `M(r,t) ~ exp(γt) m(r)`. For `r ≪ η_K` (small scales), the solution has:

```
γ ~ (u_rms / ℓ) × (Rm / Rm_c)^{1/2}   for Rm > Rm_c
```

where:
- `ℓ` is the turbulent correlation scale
- `u_rms` is the RMS velocity
- `Rm = u_rms ℓ / η`
- `Rm_c` is the critical magnetic Reynolds number (typically `Rm_c ~ 50-200`)

**Magnetic energy spectrum:**

In the kinematic growth phase, the magnetic energy spectrum at small scales is:

```
E_B(k) ∝ k^{3/2}   (Kazantsev spectrum)
```

This is **steeper** than the Kolmogorov kinetic spectrum `E_K(k) ∝ k^{-5/3}`, indicating that magnetic energy concentrates at small scales (intermittent structures).

### 2.2 Critical Magnetic Reynolds Number

The onset of small-scale dynamo requires:

```
Rm > Rm_c
```

**Dependence on Pm:**

The critical `Rm_c` depends on the **magnetic Prandtl number**:

```
Pm = ν / η
```

where:
- `ν` is kinematic viscosity
- `η` is magnetic diffusivity

Numerical simulations (e.g., Schekochihin et al., 2004; Brandenburg & Subramanian, 2005) find:

- **High Pm regime (`Pm ≫ 1`):** `Rm_c ~ 100` (weakly dependent on Pm)
  - Viscous cutoff is below resistive cutoff: `η_K ≪ η_R`
  - Dynamo operates at scales between `η_K` and `η_R`

- **Low Pm regime (`Pm ≪ 1`):** `Rm_c` increases as Pm decreases
  - Resistive cutoff is below viscous cutoff: `η_R ≪ η_K`
  - Dynamo is suppressed by resistive diffusion at small scales
  - More power in velocity field needed to overcome diffusion

- **Pm ~ 1:** `Rm_c ~ 50-100`

**Astrophysical relevance:**
- **Stars:** `Pm ~ 10^{-5} - 10^{-7}` (very small, hard to simulate)
- **ISM, galaxy clusters:** `Pm ≫ 1` (easier to dynamo)
- **Laboratory plasmas:** `Pm ~ 10^{-6}` (challenging)

### 2.3 Stretching Mechanism

The fundamental driver of small-scale dynamo is **stretching of magnetic field lines** by turbulent strain.

**Strain rate tensor:**

```
S_{ij} = (1/2)(∂v_i/∂x_j + ∂v_j/∂x_i)
```

**Magnetic field stretching:**

The evolution of a magnetic field line element `δℓ` aligned with `B` follows:

```
d(ln|δℓ|)/dt = S_{ij} (δℓ_i δℓ_j) / |δℓ|²
```

For a chaotic flow with positive **Lyapunov exponent** `λ > 0`, line elements stretch exponentially:

```
|δℓ(t)| ~ exp(λt)
```

Since magnetic field strength scales as `B ~ B₀ (|δℓ| / |δℓ₀|)` (flux freezing), we have:

```
B(t) ~ B₀ exp(λt)
```

This is the kinematic growth of the small-scale dynamo.

**Anti-dynamo constraint:**

However, stretching alone is not sufficient. Field lines can also align with the flow (along principal eigenvector of `S_{ij}`), leading to **saturation** or **suppression**. The dynamo requires:

1. **Persistent stretching:** flow must continually create new field orientation
2. **Folding:** reconnection or topological rearrangement prevents indefinite stretching in one direction

### 2.4 Saturation and Nonlinear Regime

In the **kinematic regime**, the magnetic field grows exponentially:

```
B²(t) ~ B₀² exp(2γt)
```

Eventually, the Lorentz force becomes important:

```
J × B / (ρv·∇v) ~ B² / (μ₀ρv²) ~ 1
```

This marks the transition to the **nonlinear (dynamic) regime**.

**Saturation level:**

Dimensional analysis suggests:

```
B_sat² / (2μ₀) ~ ε_B × (1/2) ρ v²
```

where `ε_B` is the magnetic-to-kinetic energy ratio at saturation.

Simulations find:
- **High Pm:** `ε_B ~ 0.1 - 1` (near-equipartition)
- **Low Pm:** `ε_B ≪ 1` (sub-equipartition, because small scales are suppressed by viscosity)

**Field structure:**

In saturation:
- Magnetic field is highly **intermittent** (concentrated in sheets, filaments)
- **Magnetic Reynolds stress** `B_iB_j / μ₀` back-reacts on velocity
- Effective reduction of turbulent kinetic energy at small scales
- Magnetic energy spectrum flattens: `E_B(k) ~ k^{-1}` to `k^{-3/2}` (less steep than Kazantsev)

## 3. Large-Scale Dynamo in Turbulence

### 3.1 Inverse Cascade of Magnetic Helicity

While small-scale dynamo amplifies fields at small scales, the **large-scale dynamo** generates coherent fields at scales larger than the forcing.

**Key concept:** **Magnetic helicity** acts as a conserved quantity (in ideal MHD) and provides a constraint.

**Magnetic helicity:**

```
H_B = ∫ A·B dV
```

where `B = ∇×A`.

**Helicity conservation:**

In ideal MHD (`η → 0`), helicity is conserved:

```
dH_B/dt = 0   (ideal MHD)
```

With finite resistivity:

```
dH_B/dt = -2η ∫ J·B dV ≈ -2η/ℓ² H_B
```

So helicity decays on a resistive timescale `τ_η = ℓ²/η`.

**Inverse cascade:**

In 3D MHD turbulence, magnetic helicity tends to **cascade to large scales** (inverse cascade), while magnetic energy cascades to small scales (forward cascade).

**Implications for dynamo:**

- Small-scale dynamo generates small-scale magnetic fields with **small-scale helicity**
- Magnetic helicity inverse cascade → builds up **large-scale helicity**
- Large-scale helicity → coherent large-scale magnetic field

This mechanism is sometimes called the **α²-dynamo** in mean-field language.

### 3.2 Helicity Constraint on Large-Scale Dynamo

**Problem:** In a closed (periodic or confined) system, the total magnetic helicity is conserved. As the large-scale field grows, it accumulates large-scale helicity. To conserve total helicity, small-scale helicity must grow with **opposite sign**. This small-scale helicity suppresses the α-effect via catastrophic quenching.

**Catastrophic α-quenching revisited:**

In mean-field theory, the α-effect is quenched by the large-scale field:

```
α(B) = α₀ / (1 + Rm (B/B_eq)²)
```

At high `Rm`, this leads to `α ~ α₀/Rm → 0`, shutting down the dynamo.

**Resolution: Helicity fluxes**

If the boundaries are **open** (e.g., stellar surface, galactic halo), magnetic helicity can **escape** through boundaries:

```
dH_B/dt = -2η ∫ J·B dV - ∫ (E × A)·dS
```

where the surface integral represents helicity flux out of the volume.

With helicity fluxes, the constraint is alleviated:
- Small-scale helicity is expelled
- Large-scale field can grow without catastrophic quenching
- Saturation occurs when helicity production ~ helicity flux + resistive dissipation

**Astrophysical applications:**
- **Solar dynamo:** Helicity carried away by solar wind and coronal mass ejections
- **Galactic dynamo:** Helicity escape to intergalactic medium via galactic winds
- **Accretion disk dynamo:** Helicity advected inward or ejected in outflows

### 3.3 Mean-Field Turbulent Dynamo

**Large-scale field evolution:**

Recall from mean-field theory:

```
∂⟨B⟩/∂t = ∇×(⟨v⟩×⟨B⟩) + ∇×(α⟨B⟩) + (η + β)∇²⟨B⟩
```

where:
- `α ~ -(1/3)τ_c⟨u·(∇×u)⟩` (helicity effect)
- `β ~ (1/3)τ_c u²` (turbulent diffusivity)

**Dynamo number:**

For α² dynamo in a domain of size `L`:

```
D_α = α L / (η + β)
```

For dynamo onset: `|D_α| ≳ 10`.

**Helicity injection:**

In rotating, stratified turbulence (e.g., stellar convection zones):
- **Coriolis force** + **density stratification** → cyclonic eddies
- Cyclonic eddies have net helicity: `⟨u·(∇×u)⟩ ≠ 0`
- Sign of helicity depends on hemisphere (opposite in N and S)

**Growth rate:**

In the kinematic regime:

```
γ ~ α² / (η_eff L)
```

Much **slower** than small-scale dynamo growth rate `γ ~ u/ℓ`.

**Saturation:**

Large-scale dynamo saturates when:
- Lorentz force modifies flow (reduces α)
- Helicity balance: production ≈ flux + dissipation

## 4. Magnetic Prandtl Number Effects

### 4.1 Definition and Regimes

**Magnetic Prandtl number:**

```
Pm = ν / η = (molecular diffusion of momentum) / (magnetic diffusion)
```

**Reynolds numbers:**

```
Re = UL / ν    (Reynolds number for flow)
Rm = UL / η    (magnetic Reynolds number)

Pm = Rm / Re
```

**Astrophysical values:**

- **Stellar interiors:** `Pm ~ 10^{-7} - 10^{-5}`
  - High conductivity (low `η`), low viscosity (high `ν` in molecular sense, but turbulent `ν_t` can be large)
- **Liquid metals (experiments):** `Pm ~ 10^{-6} - 10^{-5}`
- **Interstellar medium:** `Pm ≫ 1` (collisionless plasma, magnetic diffusion dominates viscosity)
- **Galaxy clusters:** `Pm ≫ 1`

**Scale separation:**

- **Kolmogorov scale:** `η_K = (ν³/ε)^{1/4}` (smallest scale where kinetic energy dissipates)
- **Resistive scale:** `η_R = (η³/ε)^{1/4}` (smallest scale where magnetic energy dissipates)

Ratio:

```
η_R / η_K = Pm^{-3/4}
```

- `Pm ≫ 1`: `η_R ≪ η_K` (magnetic dissipation at smaller scales)
- `Pm ≪ 1`: `η_R ≫ η_K` (viscous dissipation at smaller scales)

### 4.2 Dynamo in High Pm Regime

**Characteristics:**
- Resistive scale is below viscous scale: `η_R ≪ η_K`
- Magnetic field can be excited at scales between `η_K` and `η_R`
- **Wide inertial range** for magnetic field

**Dynamo mechanism:**
- Small-scale dynamo operates efficiently
- Critical `Rm_c ~ 100` (relatively low)
- Saturation near equipartition: `B² / (2μ₀) ~ ρv²/2`

**Applications:**
- **Interstellar medium:** Magnetic field amplification in turbulent clouds
- **Galaxy clusters:** ICM turbulence generates `μG` fields

### 4.3 Dynamo in Low Pm Regime

**Characteristics:**
- Viscous scale is below resistive scale: `η_K ≪ η_R`
- Magnetic field dissipates before reaching smallest velocity scales
- **Narrow inertial range** for magnetic field

**Dynamo mechanism:**
- Small-scale dynamo is suppressed (higher `Rm_c`)
- Saturation is **sub-equipartition**: `B²/(2μ₀) ≪ ρv²/2`
- Ratio scales as `B²/(μ₀ρv²) ~ Pm^{1/2}` (Schekochihin et al.)

**Challenges:**
- Numerical simulations at low Pm require resolving both `η_K` and `η_R` → computationally expensive
- Most astrophysical systems have `Pm ≪ 1`, but simulations often use `Pm ~ 1` or higher

**Applications:**
- **Stellar dynamos:** True `Pm ~ 10^{-6}`, but effective turbulent `Pm_t` may be closer to 1
- **Liquid metal experiments:** VKS (von Kármán Sodium) experiment, Riga dynamo

## 5. Numerical Simulations of Turbulent Dynamos

### 5.1 Direct Numerical Simulation (DNS)

**DNS** resolves all scales from the energy injection scale `L` down to the dissipation scales (`η_K` and `η_R`).

**MHD equations (incompressible):**

```
∂v/∂t + v·∇v = -∇p + J×B + ν∇²v + f
∂B/∂t = ∇×(v×B) + η∇²B
∇·v = 0
∇·B = 0
```

where `f` is a forcing term (to drive turbulence at large scales).

**Spatial resolution requirements:**

To resolve dissipation scales:

```
N_x ≥ (L / η_K)  for velocity
N_x ≥ (L / η_R)  for magnetic field
```

For `Re = 10⁴` and `Pm = 1`:

```
η_K ~ L / Re^{3/4} ~ L / 100
η_R ~ L / Rm^{3/4} ~ L / 100

N_x ≥ 100  →  N_total = 100³ = 10^6 grid points (3D)
```

For higher `Re` or low `Pm`, resolution requirements explode.

**Spectral methods:**

Typically use **Fourier pseudospectral** methods:

1. Represent fields in Fourier space: `v(x) ↔ v̂(k)`
2. Compute nonlinear terms `v·∇v`, `v×B` in real space (via FFT)
3. Compute derivatives in Fourier space: `∇ → ik`
4. Enforce `∇·v = 0`: project onto solenoidal subspace

**Time stepping:**

- **Explicit (RK3, RK4):** Simple, but CFL constraint: `Δt ≤ C Δx / |v|_max`
- **Implicit (Crank-Nicolson):** For diffusion terms, allows larger `Δt`
- **IMEX (Implicit-Explicit):** Treat advection explicitly, diffusion implicitly

### 5.2 Large Eddy Simulation (LES)

For very high Reynolds numbers (beyond DNS reach), use **LES**:

**Concept:**
- Resolve only large scales (up to some cutoff `k_c`)
- Model the effect of unresolved small scales (subgrid scales, SGS)

**Filtering:**

Apply spatial filter with width `Δ`:

```
⟨v⟩(x) = ∫ G(x - x', Δ) v(x') dx'
```

where `G` is a filter kernel (e.g., Gaussian, box, spectral cutoff).

**Filtered MHD equations:**

```
∂⟨v⟩/∂t + ⟨v⟩·∇⟨v⟩ = -∇⟨p⟩ + ⟨J⟩×⟨B⟩ + ν∇²⟨v⟩ - ∇·τ_SGS
∂⟨B⟩/∂t = ∇×(⟨v⟩×⟨B⟩) + η∇²⟨B⟩ + ∇×ε_SGS
```

where:
- `τ_SGS = ⟨vv⟩ - ⟨v⟩⟨v⟩` (SGS stress)
- `ε_SGS = ⟨v×B⟩ - ⟨v⟩×⟨B⟩` (SGS EMF)

**Subgrid models:**

1. **Eddy viscosity/resistivity:**

```
τ_SGS ≈ -ν_t(∇⟨v⟩ + (∇⟨v⟩)^T)
ε_SGS ≈ -η_t ∇×⟨B⟩
```

where `ν_t`, `η_t` are turbulent viscosity/resistivity (e.g., Smagorinsky model).

2. **Gradient model:**

```
τ_SGS ≈ C Δ² ∇⟨v⟩·∇⟨v⟩
```

3. **Dynamic models:** Compute `C` dynamically from resolved scales (Germano identity).

**Challenges for MHD LES:**
- SGS magnetic field can have strong back-reaction (dynamo at small scales)
- Standard eddy viscosity models may not capture inverse cascade of helicity
- Active research area

### 5.3 Forcing and Boundary Conditions

**Forcing:**

To maintain statistically steady turbulence, inject energy at large scales:

```
f(x,t) = F(k, t)  for k in band [k_min, k_max]
```

Common schemes:
- **Stochastic forcing:** Random phases, Gaussian statistics
- **ABC forcing:** Arnold-Beltrami-Childress flow (helical)
- **Velocity forcing:** Fix `|v̂(k)|` for certain modes, randomize phases

**Boundary conditions:**

- **Periodic:** Simplest, used in many studies
  - Problem: helicity is conserved (no flux), catastrophic quenching
- **Open (outflow):** Allow helicity flux
  - Implementation: extrapolation or zero-gradient at boundaries
- **Conducting walls:** `B_n` continuous, `E_t = 0` (or `v×B = 0` tangentially)

### 5.4 Analysis Tools

**Energy spectra:**

```
E_K(k) = (1/2) Σ_{|k'| ≈ k} |v̂(k')|²
E_B(k) = (1/2μ₀) Σ_{|k'| ≈ k} |B̂(k')|²
```

**Helicity spectra:**

```
H_K(k) = Σ_{|k'| ≈ k} Re(v̂*(k')·(ik' × v̂(k')))
H_B(k) = Σ_{|k'| ≈ k} Re(Â*(k')·B̂(k'))
```

**Structure functions:**

```
S_p(r) = ⟨|v(x+r) - v(x)|^p⟩
```

Measure intermittency: for Kolmogorov, `S_p(r) ~ r^{ζ_p}` with `ζ_p = p/3`; deviations indicate intermittency.

**Magnetic field PDFs:**

```
P(B) = probability distribution of field strength
```

Typically **non-Gaussian**, with exponential or stretched-exponential tails (intermittency).

## 6. Applications of Turbulent Dynamo

### 6.1 Interstellar Medium (ISM)

**Context:**
- ISM is highly turbulent: supernova explosions, stellar winds, thermal instabilities
- Magnetic field observed: `B ~ μG`
- `Pm ≫ 1` (collisionless plasma)

**Dynamo mechanism:**
- **Small-scale dynamo:** Amplifies seed fields to `μG` levels
- Saturation at near-equipartition with turbulent kinetic energy
- Magnetic field structure: filamentary, intermittent

**Observational tests:**
- Faraday rotation measures (RM): probe magnetic field along line of sight
- Synchrotron emission: total intensity and polarization
- Zeeman splitting: direct B measurement (limited to dense regions)

**Numerical findings:**
- Small-scale dynamo saturates at `B ~ 3-10 μG` for typical ISM turbulence
- Agrees with observations in spiral arms, star-forming regions

### 6.2 Galaxy Clusters

**Context:**
- Intracluster medium (ICM): hot, dilute plasma
- Turbulence driven by mergers, AGN feedback
- Observed magnetic fields: `B ~ μG` (from RM, radio halos)

**Dynamo mechanism:**
- Small-scale dynamo amplifies seed fields during cluster formation
- `Pm ≫ 1` (collisionless)
- Fast growth: `τ_dyn ~ Gyr`

**Challenges:**
- Conduction can suppress small-scale fluctuations (Braginski viscosity)
- Cosmic rays may affect dynamo

**Simulations:**
- Vazza et al., Miniati, Ryu: cluster formation simulations with MHD
- Find `B ~ 0.1 - 1 μG` from dynamo

### 6.3 Accretion Disks

**Context:**
- Magnetorotational instability (MRI) generates turbulence (see Lesson 12)
- Turbulent dynamo amplifies and sustains magnetic field

**Dynamo mechanism:**
- Both small-scale (MRI turbulence) and large-scale (dynamo from MRI-driven α-effect)
- Vertical field threading disk can be amplified
- `Pm ≪ 1` in protoplanetary disks (dead zones), `Pm ~ 1` in hot disks

**Saturation:**
- Magnetic stress: `⟨B_rB_φ⟩/μ₀ ~ α ⟨p⟩` with `α ~ 0.01 - 0.1`
- Corresponds to `B ~ √(αp)` → sub-thermal pressure

**Observational implications:**
- Jet launching: requires large-scale poloidal field (dynamo + advection)
- Disk winds: pressure from toroidal field

### 6.4 Early Universe

**Context:**
- Seed magnetic fields in primordial plasma
- Turbulence from phase transitions, primordial density fluctuations

**Dynamo:**
- Small-scale dynamo during radiation era (before recombination)
- Amplification factor: can reach `10^{30}` from weak seed fields
- Magnetic field coherence length: limited by horizon or damping scale

**Relevance:**
- Explain observed `nG` fields in voids and high-redshift galaxies
- Affects structure formation (magnetic pressure support)

## 7. Python Implementations

### 7.1 Kazantsev Spectrum Model

```python
import numpy as np
import matplotlib.pyplot as plt

def kazantsev_spectrum():
    """
    Model magnetic energy spectrum in small-scale dynamo.

    Kazantsev prediction: E_B(k) ∝ k^{3/2} in kinematic regime.
    """
    # Wavenumber range
    k = np.logspace(-1, 2, 100)

    # Kinetic energy spectrum (Kolmogorov)
    E_K = k**(-5/3)

    # Magnetic energy spectrum (Kazantsev kinematic)
    E_B_kinematic = k**(3/2)

    # Magnetic energy spectrum (saturated, example: k^{-3/2})
    E_B_saturated = k**(-3/2)

    # Normalize
    E_K /= E_K[len(E_K)//2]
    E_B_kinematic /= E_B_kinematic[len(E_B_kinematic)//2]
    E_B_saturated /= E_B_saturated[len(E_B_saturated)//2]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(k, E_K, 'b-', linewidth=2, label='$E_K(k) \propto k^{-5/3}$ (Kolmogorov)')
    plt.loglog(k, E_B_kinematic, 'r--', linewidth=2, label='$E_B(k) \propto k^{3/2}$ (Kazantsev kinematic)')
    plt.loglog(k, E_B_saturated, 'g-.', linewidth=2, label='$E_B(k) \propto k^{-3/2}$ (Saturated)')

    plt.xlabel('Wavenumber $k$', fontsize=14)
    plt.ylabel('Energy spectrum $E(k)$', fontsize=14)
    plt.title('Kazantsev Spectrum: Small-Scale Dynamo', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('kazantsev_spectrum.png', dpi=150)
    plt.show()

kazantsev_spectrum()
```

### 7.2 Small-Scale Dynamo Growth Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def small_scale_dynamo_growth():
    """
    Simulate kinematic growth of magnetic energy in small-scale dynamo.

    Model:
      dE_B/dt = 2γ E_B - (E_B/τ_η)

    where:
      γ = growth rate from turbulent stretching
      τ_η = resistive dissipation timescale
    """
    # Parameters
    u_rms = 1.0       # RMS velocity
    ell = 1.0         # Correlation scale
    eta_vals = [0.001, 0.005, 0.01, 0.02]  # Magnetic diffusivity

    # Time array
    t = np.linspace(0, 10, 1000)

    plt.figure(figsize=(12, 6))

    for eta in eta_vals:
        Rm = u_rms * ell / eta
        Rm_c = 60  # Critical magnetic Reynolds number

        if Rm > Rm_c:
            # Growth rate (simplified Kazantsev)
            gamma = (u_rms / ell) * np.sqrt((Rm - Rm_c) / Rm_c) * 0.1
        else:
            gamma = 0  # No dynamo

        # Resistive timescale
        tau_eta = ell**2 / eta

        # Differential equation: dE_B/dt = 2*gamma*E_B - E_B/tau_eta
        def dE_dt(E, t):
            return 2 * gamma * E - E / tau_eta

        # Initial condition
        E0 = 1e-6

        # Solve ODE
        E_B = odeint(dE_dt, E0, t)

        # Plot
        plt.semilogy(t, E_B, linewidth=2, label=f'Rm={Rm:.1f}, γ={gamma:.3f}')

    plt.xlabel('Time $t$ (in $\ell/u_{rms}$)', fontsize=14)
    plt.ylabel('Magnetic Energy $E_B$', fontsize=14)
    plt.title('Small-Scale Dynamo: Kinematic Growth', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('small_scale_dynamo_growth.png', dpi=150)
    plt.show()

small_scale_dynamo_growth()
```

### 7.3 Magnetic Helicity Evolution

```python
import numpy as np
import matplotlib.pyplot as plt

def magnetic_helicity_evolution():
    """
    Simulate evolution of magnetic helicity in a turbulent dynamo.

    Model helicity production, dissipation, and flux:
      dH_B/dt = Production - Dissipation - Flux
    """
    # Parameters
    L = 1.0           # Domain size
    eta = 0.01        # Magnetic diffusivity
    alpha0 = 0.1      # Alpha effect (helicity production rate coefficient)
    flux_rate = 0.05  # Helicity flux rate (if boundaries are open)

    # Time array
    t = np.linspace(0, 100, 1000)
    dt = t[1] - t[0]

    # Two scenarios: closed vs open boundaries
    scenarios = {
        'Closed (no flux)': 0.0,
        'Open (with flux)': flux_rate
    }

    plt.figure(figsize=(12, 8))

    for i, (label, flux) in enumerate(scenarios.items()):
        # Initialize
        H_B = np.zeros(len(t))
        B_rms = np.zeros(len(t))

        H_B[0] = 0.0
        B_rms[0] = 0.01

        # Time evolution
        for n in range(len(t) - 1):
            # Helicity production (from alpha effect and field growth)
            production = alpha0 * B_rms[n]**2

            # Resistive dissipation
            dissipation = (2 * eta / L**2) * H_B[n]

            # Helicity flux (for open boundaries)
            flux_term = flux * H_B[n]

            # Update helicity
            dH_dt = production - dissipation - flux_term
            H_B[n+1] = H_B[n] + dt * dH_dt

            # Simple model for field growth with helicity constraint
            # α-quenching: α_eff = α0 / (1 + |H_B| / H_sat)
            H_sat = 0.1
            alpha_eff = alpha0 / (1 + np.abs(H_B[n]) / H_sat)

            # Field growth (simplified)
            gamma = alpha_eff - eta / L**2
            dB_dt = gamma * B_rms[n]
            B_rms[n+1] = B_rms[n] + dt * dB_dt

        # Plot
        plt.subplot(2, 1, 1)
        plt.plot(t, H_B, linewidth=2, label=label)

        plt.subplot(2, 1, 2)
        plt.plot(t, B_rms, linewidth=2, label=label)

    plt.subplot(2, 1, 1)
    plt.xlabel('Time $t$', fontsize=14)
    plt.ylabel('Magnetic Helicity $H_B$', fontsize=14)
    plt.title('Magnetic Helicity Evolution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.xlabel('Time $t$', fontsize=14)
    plt.ylabel('RMS Magnetic Field $B_{rms}$', fontsize=14)
    plt.title('Magnetic Field Evolution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('helicity_evolution.png', dpi=150)
    plt.show()

magnetic_helicity_evolution()
```

### 7.4 Turbulent Cascade with Dynamo

```python
import numpy as np
import matplotlib.pyplot as plt

def turbulent_cascade_with_dynamo():
    """
    Simulate energy cascade in MHD turbulence with dynamo.

    Model shell-averaged energy equations:
      dE_K(k)/dt = T_K(k) + F_K(k) - ν k² E_K(k) - M(k)
      dE_B(k)/dt = T_B(k) + Dynamo(k) - η k² E_B(k) + M(k)

    where:
      T_K, T_B: nonlinear transfer (cascade)
      F_K: forcing
      M: magnetic-kinetic energy exchange
      Dynamo: energy input from stretching
    """
    # Wavenumber bins (logarithmic)
    N_bins = 20
    k = np.logspace(0, 2, N_bins)
    dk = np.diff(np.log(k))
    dk = np.append(dk, dk[-1])

    # Parameters
    nu = 0.01      # Viscosity
    eta = 0.005    # Magnetic diffusivity
    forcing_k = 2  # Forcing wavenumber index

    # Time stepping
    dt = 0.001
    Nt = 5000

    # Initialize
    E_K = np.zeros(N_bins)
    E_B = np.zeros(N_bins)

    # Initial kinetic energy (inject at large scales)
    E_K[forcing_k] = 1.0

    # Storage
    E_K_hist = []
    E_B_hist = []

    for n in range(Nt):
        # Forcing
        F_K = np.zeros(N_bins)
        F_K[forcing_k] = 0.1  # Constant energy injection

        # Nonlinear transfer (simplified cascade model)
        # T_K(k) ~ -d/dk(k² E_K)  (dimensional, forward cascade)
        T_K = np.zeros(N_bins)
        T_B = np.zeros(N_bins)

        for i in range(1, N_bins - 1):
            # Forward cascade for kinetic
            T_K[i] = -0.5 * (E_K[i] - E_K[i-1]) / dk[i]

            # Forward cascade for magnetic (Iroshnikov-Kraichnan)
            T_B[i] = -0.3 * (E_B[i] - E_B[i-1]) / dk[i]

        # Dynamo effect: kinetic energy → magnetic energy at small scales
        Dynamo = np.zeros(N_bins)
        for i in range(N_bins):
            if k[i] > k[forcing_k]:
                # Stretching proportional to strain rate ~ k E_K^{1/2}
                Dynamo[i] = 0.1 * k[i] * np.sqrt(E_K[i]) * (1 - E_B[i] / (E_K[i] + 1e-10))

        # Magnetic-kinetic coupling (Lorentz force back-reaction)
        M = 0.05 * E_B * np.sqrt(E_K + 1e-10)

        # Dissipation
        D_K = nu * k**2 * E_K
        D_B = eta * k**2 * E_B

        # Update
        dE_K_dt = T_K + F_K - D_K - M
        dE_B_dt = T_B + Dynamo - D_B + M

        E_K += dt * dE_K_dt
        E_B += dt * dE_B_dt

        # Prevent negative energies
        E_K = np.maximum(E_K, 0)
        E_B = np.maximum(E_B, 0)

        # Store snapshots
        if n % 100 == 0:
            E_K_hist.append(E_K.copy())
            E_B_hist.append(E_B.copy())

    # Plot final spectra
    plt.figure(figsize=(10, 6))
    plt.loglog(k, E_K, 'b-o', linewidth=2, markersize=5, label='Kinetic $E_K(k)$')
    plt.loglog(k, E_B, 'r-s', linewidth=2, markersize=5, label='Magnetic $E_B(k)$')

    # Reference slopes
    k_ref = k[5:15]
    plt.loglog(k_ref, 0.1 * k_ref**(-5/3), 'k--', linewidth=1, label='$k^{-5/3}$ (Kolmogorov)')
    plt.loglog(k_ref, 0.01 * k_ref**(-3/2), 'g--', linewidth=1, label='$k^{-3/2}$ (IK or saturated dynamo)')

    plt.xlabel('Wavenumber $k$', fontsize=14)
    plt.ylabel('Energy $E(k)$', fontsize=14)
    plt.title('Energy Spectra in MHD Turbulence with Dynamo', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('turbulent_cascade_dynamo.png', dpi=150)
    plt.show()

    # Animate evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(0, len(E_K_hist), max(1, len(E_K_hist)//10)):
        ax.clear()
        ax.loglog(k, E_K_hist[i], 'b-o', linewidth=2, markersize=5, label='Kinetic')
        ax.loglog(k, E_B_hist[i], 'r-s', linewidth=2, markersize=5, label='Magnetic')
        ax.set_xlabel('Wavenumber $k$', fontsize=14)
        ax.set_ylabel('Energy $E(k)$', fontsize=14)
        ax.set_title(f'Energy Spectra (t = {i*100*dt:.2f})', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        plt.pause(0.1)

    plt.show()

turbulent_cascade_with_dynamo()
```

### 7.5 Pm Dependence of Dynamo Onset

```python
import numpy as np
import matplotlib.pyplot as plt

def Pm_dependence_dynamo():
    """
    Plot critical Rm vs Pm for small-scale dynamo onset.

    Empirical fits from simulations:
      - High Pm: Rm_c ~ 100 (const)
      - Low Pm: Rm_c ~ C * Pm^{-α} (increases as Pm decreases)
    """
    Pm = np.logspace(-3, 2, 100)

    # Empirical model (Schekochihin et al.)
    Rm_c = np.zeros_like(Pm)

    for i, pm in enumerate(Pm):
        if pm >= 1:
            # High Pm regime
            Rm_c[i] = 100
        else:
            # Low Pm regime (example: Rm_c ~ 100 * Pm^{-1/2})
            Rm_c[i] = 100 * pm**(-0.5)

    plt.figure(figsize=(10, 6))
    plt.loglog(Pm, Rm_c, 'b-', linewidth=2.5, label='Critical $Rm_c(Pm)$')

    # Reference lines
    plt.axhline(100, color='k', linestyle='--', linewidth=1, label='$Rm_c \\approx 100$ (High Pm)')
    plt.loglog(Pm[Pm < 1], 100 * Pm[Pm < 1]**(-0.5), 'r--', linewidth=1, label='$Rm_c \propto Pm^{-1/2}$ (Low Pm)')

    plt.xlabel('Magnetic Prandtl Number $Pm = \\nu/\\eta$', fontsize=14)
    plt.ylabel('Critical Magnetic Reynolds Number $Rm_c$', fontsize=14)
    plt.title('Dynamo Onset: Dependence on Magnetic Prandtl Number', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('Pm_dependence_dynamo.png', dpi=150)
    plt.show()

Pm_dependence_dynamo()
```

## 8. Summary

**Turbulent dynamos** are essential for understanding magnetic field generation in a wide range of astrophysical systems:

1. **Small-scale dynamo:**
   - Amplifies magnetic field at scales ≤ turbulent forcing scale
   - Driven by turbulent stretching (Lyapunov exponent)
   - **Kazantsev theory:** kinematic growth rate `γ ~ (u/ℓ) (Rm/Rm_c)^{1/2}` for `Rm > Rm_c`
   - **Kazantsev spectrum:** `E_B(k) ∝ k^{3/2}` (kinematic), flattens upon saturation
   - Critical `Rm_c ~ 50-200`, depends on **Pm**

2. **Magnetic Prandtl number (Pm = ν/η):**
   - **High Pm (`Pm ≫ 1`):** Efficient dynamo, near-equipartition saturation
   - **Low Pm (`Pm ≪ 1`):** Higher `Rm_c`, sub-equipartition saturation
   - Most astrophysical plasmas have `Pm ≪ 1`, but effective turbulent Pm may be `~ 1`

3. **Large-scale dynamo:**
   - Requires **helicity** (kinetic or magnetic) to generate fields at scales > forcing
   - **Inverse cascade** of magnetic helicity builds coherent large-scale field
   - **Helicity constraint:** In closed systems, helicity conservation leads to **catastrophic α-quenching**
   - **Resolution:** Open boundaries → helicity fluxes alleviate quenching

4. **Saturation mechanisms:**
   - Lorentz force back-reaction reduces turbulent stretching
   - **α-quenching** in mean-field picture
   - Balance: dynamo production ≈ resistive dissipation + helicity flux

5. **Numerical simulations:**
   - **DNS:** Resolve all scales, limited to `Re, Rm ≲ 10^4`
   - **LES:** Model subgrid scales, reach higher `Re, Rm`
   - Challenges: low Pm requires resolving both `η_K` and `η_R`

6. **Applications:**
   - **ISM:** Small-scale dynamo → `μG` fields (observed)
   - **Galaxy clusters:** Small-scale dynamo during mergers
   - **Accretion disks:** MRI-driven turbulent dynamo
   - **Early universe:** Seed field amplification

Turbulent dynamos are a vibrant research area, with ongoing work on understanding saturation, helicity fluxes, and the transition from small-scale to large-scale dynamos.

## Practice Problems

1. **Kazantsev Growth Rate:** For turbulence with `u_rms = 10 m/s`, `ℓ = 10⁶ m`, `η = 10⁴ m²/s`, compute `Rm` and estimate the growth rate assuming `Rm_c = 100`.

2. **Critical Rm for Low Pm:** If `Rm_c ~ 100 Pm^{-1/2}` for `Pm < 1`, what is `Rm_c` for liquid sodium with `Pm = 10^{-5}`?

3. **Resistive Scale:** For `Re = 10⁴` and `Pm = 0.01`, compute the ratio `η_R / η_K`. Which dissipates at smaller scales?

4. **Equipartition Field:** In the ISM with `ρ = 10^{-21} kg/m³`, `v = 10 km/s`, compute the equipartition magnetic field in Gauss.

5. **Helicity Dissipation:** For a domain of size `L = 1 kpc` with `η = 10^{26} cm²/s` (ISM), estimate the resistive decay timescale of magnetic helicity.

6. **Kazantsev Spectrum:** Plot the expected `E_B(k)` for a kinematic small-scale dynamo and compare to a saturated state with `E_B(k) ∝ k^{-3/2}`. At what wavenumber do they cross?

7. **Pm Scaling:** If saturation field strength scales as `B_sat ∝ Pm^{1/2}` for low Pm, by what factor does `B_sat` decrease when going from `Pm = 1` to `Pm = 10^{-6}`?

8. **Python Exercise:** Modify the small-scale dynamo growth code to include saturation via `γ(B) = γ₀(1 - B²/B_eq²)`. Observe the exponential growth → saturation transition.

9. **Helicity Flux:** In the magnetic helicity evolution code, increase the flux rate and observe how it affects the saturation level of the magnetic field.

10. **Advanced:** Implement a simple shell-model for MHD turbulence with dynamo. Use logarithmically spaced wavenumber shells and model the nonlinear transfer between shells. Study the energy cascade and dynamo onset as `Rm` is varied.

---

**Previous:** [Dynamo Theory](./09_Dynamo_Theory.md) | **Next:** [Solar MHD](./11_Solar_MHD.md)

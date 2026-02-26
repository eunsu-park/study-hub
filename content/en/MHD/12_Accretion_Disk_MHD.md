# 12. Accretion Disk MHD

## Learning Objectives

By the end of this lesson, you should be able to:

- Understand the fundamental problem of angular momentum transport in accretion disks
- Derive and analyze the magnetorotational instability (MRI) dispersion relation
- Explain why MRI is essential for astrophysical disk accretion
- Calculate MRI growth rates and characteristic wavelengths
- Understand angular momentum transport via Maxwell and Reynolds stresses
- Relate MRI turbulence to the α-disk model
- Describe disk winds and jet formation mechanisms (Blandford-Payne, magnetic tower)
- Implement numerical models of MRI and disk physics

## 1. Accretion Disk Basics

### 1.1 The Angular Momentum Problem

**Accretion disks** form around compact objects (black holes, neutron stars, white dwarfs) and protostars when infalling material has significant angular momentum. The material settles into a rotating disk configuration.

**Keplerian rotation:**

For a test particle in circular orbit at radius `r` around a central mass `M`:

```
Centrifugal force = Gravitational force
v²/r = GM/r²
v = √(GM/r)
```

Angular velocity:

```
Ω(r) = v/r = √(GM/r³) ∝ r^{-3/2}
```

This is **Keplerian rotation**: angular velocity decreases with radius.

**The problem:**

For material to accrete (move inward), it must **lose angular momentum**. But how?

**Specific angular momentum:**

```
ℓ = r² Ω = √(GM r)
```

At the innermost stable circular orbit (ISCO) for a Schwarzschild black hole:

```
r_ISCO = 6 GM/c²
ℓ_ISCO = √(6 GM² / c²)
```

Material at large `r` has much larger `ℓ`. To reach the ISCO, it must shed angular momentum.

**Possible mechanisms:**

1. **Molecular viscosity:** Far too small (Re ~ 10¹⁴ in astrophysical disks)
2. **Gravitational torques:** In stellar binaries or self-gravitating disks (marginal)
3. **Magnetic fields + turbulence:** This is the key!

### 1.2 The α-Disk Model (Shakura & Sunyaev 1973)

In the absence of a detailed understanding of angular momentum transport, Shakura & Sunyaev parameterized it:

**Viscous stress:**

```
τ_rφ = -ρ ν_eff r dΩ/dr
```

where `ν_eff` is an **effective viscosity** (not molecular!).

**Parameterization:**

```
ν_eff = α c_s H
```

where:
- `α` is a dimensionless parameter (0 < α < 1)
- `c_s` is the sound speed
- `H` is the disk scale height

**Physical interpretation:**

- Turbulent eddies of size `~ H` move with velocity `~ c_s`
- Mixing length theory: `ν_eff ~ v_turb × ℓ_mix ~ c_s H`
- `α` measures efficiency of angular momentum transport

**Typical values:**

From observations (fits to X-ray binaries, AGN):

```
α ~ 0.01 - 0.1
```

**Key question:** What physical process sets `α`? For decades, this was unknown. The answer: **MRI-driven turbulence**.

### 1.3 Disk Structure Equations

**Mass conservation:**

```
∂Σ/∂t + (1/r) ∂(r Σ v_r)/∂r = 0
```

where `Σ` is surface density, `v_r` is radial velocity.

**Angular momentum conservation:**

```
∂(Σ r² Ω)/∂t + (1/r) ∂(r² Σ v_r r² Ω)/∂r = (1/r) ∂(r² τ_rφ)/∂r
```

**Energy equation:**

Viscous dissipation heats the disk:

```
Q_vis = τ_rφ r dΩ/dr
```

This energy is radiated away:

```
Q_rad = σ T_eff⁴
```

where `T_eff` is the effective surface temperature.

**Steady-state accretion:**

In steady state, `∂/∂t = 0`, and there's a constant accretion rate `Ṁ`:

```
Σ v_r × 2πr = -Ṁ
```

(Negative because inflow.)

**Temperature profile:**

For a geometrically thin disk (`H ≪ r`):

```
T_eff ∝ (Ṁ / r³)^{1/4}
```

For a black hole accretion disk:

```
T_eff ~ 10⁶ (Ṁ / Ṁ_Edd)^{1/4} (M / 10 M_☉)^{-1/4} (r / 10 R_s)^{-3/4} K
```

This is hot enough to emit X-rays in the inner regions!

## 2. Magnetorotational Instability (MRI)

### 2.1 Discovery and Importance

**Balbus & Hawley (1991)** rediscovered (after Velikhov 1959, Chandrasekhar 1960) that a weak magnetic field in a differentially rotating disk is **linearly unstable** to the **magnetorotational instability (MRI)**.

**Significance:**

- MRI is the **most important instability** in accretion disk theory
- Generates turbulence → effective viscosity → accretion
- Sets `α ~ 0.01` in MRI-turbulent disks
- Operates in protoplanetary disks, X-ray binaries, AGN, tidal disruption events

### 2.2 Physical Mechanism: Spring Analogy

**Setup:**

- Two fluid elements at radii `r` and `r + δr`
- Connected by a magnetic field line (acts like a spring)
- Disk has Keplerian rotation: `Ω(r) ∝ r^{-3/2}`

**Unperturbed state:**

Both elements rotate at their local `Ω(r)`.

**Perturbation:**

Displace the inner element outward and the outer element inward (slight radial perturbation).

**Evolution:**

1. **Without magnetic field:**
   - Inner element moves outward → conserves angular momentum → rotates slower than local Keplerian → lags behind
   - Outer element moves inward → rotates faster → moves ahead
   - They drift apart azimuthally → **stable** (Rayleigh criterion)

2. **With magnetic field:**
   - Magnetic tension acts like a spring connecting the two elements
   - Inner element (now outward, slower) is **pulled forward** by the field → gains angular momentum → moves further outward
   - Outer element (now inward, faster) is **pulled backward** → loses angular momentum → moves further inward
   - **Positive feedback** → instability!

**Key insight:**

The magnetic field **transports angular momentum outward** (from inner to outer element), allowing the inner element to move outward and the outer to move inward → **instability** despite Rayleigh stability.

### 2.3 Local Linear Analysis: Shearing Box

To analyze MRI, we use the **shearing box** approximation (Goldreich & Lynden-Bell 1965):

**Coordinates:**

- Local Cartesian frame at radius `r_0`
- `x = r - r_0` (radial, outward)
- `y = r_0 (φ - Ω_0 t)` (azimuthal, in rotating frame)
- `z` (vertical)

**Shear flow:**

In the rotating frame, the background velocity is:

```
v_y = -q Ω_0 x
```

where `q = -d ln Ω / d ln r`. For Keplerian: `q = 3/2`.

**Linearized MHD equations:**

Perturb around a uniform vertical field `B_0 = B_z ẑ`:

```
∂v'/∂t + (v·∇)v' = -(1/ρ)∇p' + (1/ρμ_0)(∇×B')×B_0 + 2q Ω_0² x x̂
∂B'/∂t = ∇×(v'×B_0)
∇·v' = 0, ∇·B' = 0
```

The term `2q Ω_0² x x̂` is the **tidal force** (from local expansion of gravity).

**Seek plane wave solutions:**

```
v' ~ exp(i k·x + γ t)
B' ~ exp(i k·x + γ t)
```

### 2.4 MRI Dispersion Relation

For simplicity, consider **axisymmetric modes** (`k_y = 0` initially) with vertical field `B_0 = B_z ẑ` and wavenumber `k = k_z`.

**Incompressibility:**

Assume `∇·v' = 0` (valid for subsonic perturbations).

**Dispersion relation:**

After some algebra (see Balbus & Hawley 1991, 1998 for derivation), the growth rate `γ` satisfies:

```
γ⁴ + γ² [2κ² - (2 - q) Ω_0²] + κ² [κ² - (k v_A)²] = 0
```

where:
- `κ² = 2 Ω_0 (Ω_0 + r dΩ/dr) = (2 - q) Ω_0²` is the **epicyclic frequency** squared
- `v_A = B_0 / √(μ_0 ρ)` is the Alfvén speed

For Keplerian rotation (`q = 3/2`):

```
κ² = Ω_0²
```

The dispersion relation becomes:

```
γ⁴ + γ² (2 Ω_0² - Ω_0²/2) + Ω_0² [Ω_0² - (k v_A)²] = 0
γ⁴ + (3/2) Ω_0² γ² + Ω_0² [Ω_0² - (k v_A)²] = 0
```

**Instability condition:**

For `γ²` to be positive (exponential growth):

The discriminant of the quadratic (in `γ²`) must be positive, and at least one root `γ² > 0`.

Solving:

```
γ² = -(3/4) Ω_0² ± √[(9/16) Ω_0⁴ - Ω_0² (Ω_0² - (k v_A)²)]
    = -(3/4) Ω_0² ± Ω_0² √[(9/16) - 1 + (k v_A / Ω_0)²]
    = -(3/4) Ω_0² ± Ω_0² √[(k v_A / Ω_0)² - 7/16]
```

For instability, we need:

```
(k v_A / Ω_0)² < 7/16  (long wavelengths)
```

Then the `+` root gives `γ² > 0` for some range of `k v_A / Ω_0`.

**Maximum growth rate (Balbus & Hawley 1998):**

For Keplerian disk with weak vertical field:

```
γ² = (1/2) [(k v_A)² - Ω_0²] + (1/2) √[(k v_A)² + Ω_0²)² - 16 q Ω_0² (k v_A)²]
```

For `q = 3/2` (Keplerian):

Maximum growth rate:

```
γ_max = (√(3)/4) Ω_0 ≈ 0.433 Ω_0
```

This occurs at `k v_A → 0` (long wavelengths).

**Key results:**

1. **Instability criterion:**
   - For Keplerian rotation: **ANY** magnetic field (no matter how weak!) is unstable
   - Condition: `dΩ²/d ln r < 0` (angular velocity decreasing outward)

2. **Growth rate:**
   - Comparable to orbital frequency: `γ ~ Ω_0` (very fast!)
   - E-folding time: `τ = 1/γ ~ 1/(Ω_0) ~ orbital period`

3. **Fastest growing mode:**
   - Wavelength: `λ_MRI ~ 2π v_A / Ω_0`
   - For weak field, `λ_MRI` can be much smaller than disk height

### 2.5 Non-Axisymmetric Modes

For modes with `k_y ≠ 0` (azimuthal structure), the instability persists. Including radial and azimuthal wavenumbers:

**Full dispersion relation (general):**

```
γ⁴ + γ² [κ² + (k·v_A)² - 2 Ω_0 k_y v_{Ay}]
    + κ² [(k·v_A)² - 4 Ω_0 k_y v_{Ay}] = 0
```

where `v_A = B_0 / √(μ_0 ρ)` is the Alfvén velocity vector.

**Key points:**

- Instability exists for all orientations of weak field (vertical, toroidal, etc.)
- Growth rate is always `~ Ω_0` for Keplerian disks
- MRI is **ubiquitous** in magnetized disks

### 2.6 Why Keplerian Disks Are Rayleigh Stable but MRI Unstable

**Rayleigh criterion:**

A rotating fluid is stable to hydrodynamic (non-magnetic) perturbations if:

```
d(r² Ω)² / dr > 0
```

For Keplerian: `Ω ∝ r^{-3/2}` → `r² Ω ∝ r^{1/2}` → derivative > 0 → **stable**.

This is why Keplerian disks cannot have **hydrodynamic turbulence** from shear alone.

**MRI changes the game:**

Magnetic field provides a **channel for angular momentum transport** that bypasses the Rayleigh criterion. The magnetic tension creates a **negative effective viscosity** or **destabilizing torque** that overcomes the stabilizing centrifugal effect.

## 3. Angular Momentum Transport in MRI Turbulence

### 3.1 Maxwell Stress

In MHD, the **stress tensor** has contributions from both fluid motions (Reynolds stress) and magnetic fields (Maxwell stress).

**Total stress tensor:**

```
T_{ij} = ρ v_i v_j + p δ_{ij} + (B_i B_j / μ_0 - B² δ_{ij} / (2μ_0))
```

**Angular momentum transport:**

The radial transport of angular momentum (per unit area) is:

```
τ_rφ = ρ v_r v_φ - B_r B_φ / μ_0
```

where:
- `ρ v_r v_φ` is the **Reynolds stress** (hydrodynamic)
- `-B_r B_φ / μ_0` is the **Maxwell stress** (magnetic)

**Sign convention:**

Positive `τ_rφ` → outward transport of angular momentum.

### 3.2 MRI Turbulence Simulations

Numerical simulations (Hawley, Balbus, Stone, et al.) using shearing-box MHD codes find:

**Maxwell stress dominates:**

```
⟨B_r B_φ⟩ / μ_0  ≫  ⟨ρ v_r v_φ⟩
```

Typically, Maxwell stress is ~10 times larger than Reynolds stress.

**α parameter:**

Define:

```
α = ⟨τ_rφ⟩ / ⟨p⟩
```

where `⟨·⟩` denotes time and volume average.

From simulations (e.g., Hawley, Gammie, Balbus 1995):

- **Zero net vertical flux:** `α ~ 0.01-0.02`
- **With net vertical flux:** `α ~ 0.05-0.5` (stronger transport)

**Field configuration matters:**

- **Net flux (ordered field):** Sustained channel modes, higher α
- **Zero net flux (turbulent field):** Sustained turbulence, but lower α

### 3.3 Effective Viscosity

From the α parameter:

```
ν_eff = α c_s H
```

For a typical disk with `c_s ~ 10 km/s`, `H ~ 0.1 R ~ 10⁹ cm`, `α ~ 0.01`:

```
ν_eff ~ 10¹⁵ cm²/s
```

This is **enormous** compared to molecular viscosity (`ν_mol ~ 1 cm²/s`), but arises naturally from MRI turbulence.

**Accretion timescale:**

```
τ_acc ~ R² / ν_eff ~ (10¹⁰ cm)² / (10¹⁵ cm²/s) ~ 10⁵ s ~ 1 day
```

This allows rapid accretion onto compact objects, explaining observed X-ray variability.

## 4. Nonlinear MRI and Turbulent Saturation

### 4.1 Channel Solution

**Channel modes** are exact nonlinear solutions of the incompressible MHD equations in a shearing box (Goodman & Xu 1994).

**Structure:**

- Traveling waves in the azimuthal direction
- Exponentially growing in the linear regime
- Sustained in the nonlinear regime (for disks with net vertical flux)

**Energy:**

Channel modes carry most of the magnetic energy and stress in saturated state (for net flux cases).

**Parasitic instabilities:**

Channel modes themselves are unstable to **parasitic modes** (secondary instabilities) that break them up, leading to turbulence.

### 4.2 Saturation Mechanism

**Lorentz force back-reaction:**

As MRI grows, magnetic field strength increases. The Lorentz force modifies the flow:

```
J × B ~ B² / L
```

When magnetic pressure becomes comparable to thermal pressure:

```
B² / (2μ_0) ~ p
```

or

```
β = 2μ_0 p / B² ~ 1
```

the growth saturates.

**Typical saturation:**

Simulations find:

```
⟨B²⟩ / (2μ_0 ⟨p⟩) ~ 1-10  (order unity)
```

Magnetic energy is **sub-thermal** to **super-thermal**, depending on field configuration.

### 4.3 MRI in Stratified Disks

In **stratified shearing-box** simulations (with gravity in `z`-direction):

**Butterfly effect:**

Magnetic field develops a **butterfly pattern**: alternating toroidal field polarity above and below midplane, propagating in `z`.

**Zonal flows:**

Large-scale **azimuthal flows** (independent of `φ`) develop due to Reynolds stress.

**Accretion stress:**

Averaged over the vertical extent, the effective `α` is similar to unstratified case (~0.01-0.1), but with significant vertical variation.

## 5. Dead Zones and Non-Ideal MHD Effects

### 5.1 MRI in Protoplanetary Disks

**Problem:**

MRI requires ionization (for coupling between gas and magnetic field). In protoplanetary disks:

- **Inner regions (< 0.1 AU):** Hot, thermally ionized → MRI active
- **Outer regions (> 1 AU) at midplane:** Cold, dense → **poorly ionized** → MRI suppressed

**Dead zone:**

Region where ionization is too low for MRI. Criteria:

```
Magnetic Reynolds number: Rm = v L / η_Ohm > 10⁴ (for MRI)
```

where `η_Ohm = c² / (4π σ)` is Ohmic resistivity.

At the midplane of a disk at 1 AU around a young star:
- Temperature: `T ~ 100-300 K`
- Ionization fraction: `x_e ~ 10^{-13}` (from cosmic rays, radioactive decay)
- `Rm ~ 10-100` → **MRI suppressed**

**Layered accretion:**

- **Active zone:** Near surface (ionized by UV, X-rays, cosmic rays)
- **Dead zone:** Midplane (neutral, no MRI)
- Accretion proceeds primarily in the active layers

### 5.2 Non-Ideal MHD Effects

In weakly ionized plasmas, three non-ideal effects become important:

**1. Ohmic diffusion:**

```
∂B/∂t = ∇×(v×B) + η_Ohm ∇²B
```

Resistivity `η_Ohm` damps small-scale field fluctuations.

**2. Ambipolar diffusion:**

Ions and neutrals are not perfectly coupled. Magnetic field slips through neutrals:

```
∂B/∂t = ∇×(v×B) + ∇×(η_AD (∇×B)×B / B²)
```

where `η_AD ~ B / (ρ_n γ_in)`, `γ_in` is ion-neutral collision rate.

**3. Hall effect:**

Electrons drift relative to ions in the presence of current:

```
∂B/∂t = ∇×(v×B) - ∇×(η_H (∇×B)×B / |B|)
```

where `η_H ~ B / (e n_e)`.

**Impact on MRI:**

- **Ohmic diffusion:** Suppresses MRI at small scales → raises critical wavelength
- **Ambipolar diffusion:** Suppresses MRI in dense, weakly ionized regions
- **Hall effect:** Can modify MRI (enhance or suppress depending on field direction and sign of Hall term)

**Consequence:**

In protoplanetary disks, MRI may be active only in:
- Surface layers
- Inner hot regions
- Regions with enhanced ionization (near star, wind-swept zones)

Alternative mechanisms (e.g., vertical shear instability, gravitational instability) may operate in dead zones.

## 6. Disk Winds and Jets

### 6.1 Blandford-Payne Mechanism (1982)

Magnetic fields threading the disk can launch **centrifugally driven winds**.

**Setup:**

- Poloidal magnetic field `B_p` threading disk
- Field lines anchored in disk, bend outward
- Gas flows along field lines

**Condition for centrifugal launch:**

The field line must be inclined at angle `θ` from the disk normal (vertical):

```
θ > 30°  (critical angle)
```

**Physical picture:**

1. Gas at radius `r` has angular velocity `Ω(r)`
2. If it moves along a field line inclined outward, it conserves angular momentum: `L = r v_φ = const`
3. At larger cylindrical radius `R > r`, centrifugal force `L²/(R³)` can exceed gravity + magnetic tension
4. Gas is **flung outward** (centrifugal acceleration)

**Acceleration:**

The wind accelerates to:

```
v_∞² ~ 2 G M / r_0
```

where `r_0` is the launch radius. This is comparable to the **escape speed**.

**Mass loss rate:**

Fraction of accretion flow ejected as wind:

```
Ṁ_wind / Ṁ_acc ~ (B_p / √(4π ρ v_K²)) ~ β_p^{-1/2}
```

where `β_p = 8π p / B_p²` is the plasma beta.

### 6.2 Magnetic Tower

In the case of strong toroidal field (`B_φ`), the **magnetic pressure gradient** can drive collimated outflows.

**Mechanism:**

1. Differential rotation winds up poloidal field → strong toroidal field `B_φ`
2. Toroidal field has pressure `B_φ² / (2μ_0)`
3. Pressure gradient in `z` (vertical) → upward force
4. Gas is pushed upward along the axis

**Hoop stress:**

The toroidal field also exerts a **pinch force** (hoop stress):

```
F_pinch ~ -∂(B_φ² / (2μ_0)) / ∂R
```

This **collimates** the flow toward the axis → jet formation.

**Simulations:**

MHD simulations (e.g., Lynden-Bell, Kato, Kudoh) show that a combination of:
- Toroidal field buildup
- Magnetic tower formation
- Collimation by hoop stress

can produce **bipolar jets** similar to those observed in young stellar objects and AGN.

### 6.3 Application to AGN Jets

**Active Galactic Nuclei (AGN):**

Supermassive black holes (10⁶-10⁹ M☉) at galaxy centers accrete gas and launch powerful **relativistic jets**:
- Length: up to Mpc scales
- Speed: `v ~ 0.1-0.99 c`
- Power: `L_jet ~ 10⁴²-10⁴⁷ erg/s`

**Magnetic launching:**

Leading model: **Blandford-Znajek mechanism** (1977)
- Magnetic field threading the **black hole horizon** (not just disk)
- Black hole rotation twists field lines → electromagnetic energy extraction
- Power: `P ~ B² a² c` where `a` is black hole spin

Alternatively, **Blandford-Payne from disk** can contribute.

**Collimation:**

Toroidal field (from rotation) + external pressure (disk wind, cocoon) → collimated jet.

**Observational evidence:**

- VLBI imaging: jets resolved down to ~10 Schwarzschild radii
- Event Horizon Telescope (EHT): M87* jet base at horizon scale
- Polarization: consistent with ordered poloidal+toroidal field

## 7. Python Implementations

### 7.1 MRI Dispersion Relation

```python
import numpy as np
import matplotlib.pyplot as plt

def mri_dispersion():
    """
    Solve MRI dispersion relation for Keplerian disk.

    Dispersion relation (vertical field, incompressible):
      γ⁴ + γ² [κ² + (k v_A)²] - 3 Ω² (k v_A)² = 0

    For Keplerian: κ² = Ω²
    """
    Omega = 1.0  # Orbital frequency (normalized)

    # Range of k v_A / Omega
    k_vA_over_Omega = np.linspace(0, 2, 200)

    # Dispersion relation: γ⁴ + γ² [Ω² + (k v_A)²] - 3 Ω² (k v_A)² = 0
    # Let X = γ² / Ω², Y = (k v_A / Ω)²
    # X² + X [1 + Y] - 3 Y = 0
    # This normalisation reveals that the instability is controlled entirely
    # by the ratio k v_A / Ω — the ratio of the "spring constant" (Alfvén
    # restoring force) to the "centrifugal restoring force" (orbital shear).
    # When this ratio is small (weak field), the spring is too compliant and
    # allows runaway angular-momentum transfer; that is the spring analogy.

    Y = k_vA_over_Omega**2

    # Solve quadratic for X = γ² / Ω²
    # X = -(1+Y)/2 ± √[(1+Y)²/4 + 3Y]
    # The +√ root can be positive (X > 0 → γ² > 0 → exponential growth);
    # the -√ root is always negative, representing a damped oscillation
    # (restoring torque wins) — these are the two physically distinct branches.
    discriminant = (1 + Y)**2 / 4 + 3 * Y
    X_plus = -(1 + Y)/2 + np.sqrt(discriminant)
    X_minus = -(1 + Y)/2 - np.sqrt(discriminant)

    # Growth rate γ / Ω (take positive root)
    # We mask X_plus < 0 with zeros because negative X_plus means γ² < 0,
    # i.e. the mode is oscillatory (not growing) — there is an upper cutoff
    # wavelength above which the magnetic spring is too stiff to be stretched
    # by the orbital shear and MRI is stabilised.
    gamma_over_Omega = np.where(X_plus > 0, np.sqrt(X_plus), 0)

    # Maximum growth rate (at k v_A → 0)
    # As k v_A → 0 the field becomes infinitely flexible (zero tension) and
    # the growth rate approaches its maximum ~0.75 Ω; this is why even an
    # infinitesimally weak field is sufficient to trigger MRI in a Keplerian disk.
    gamma_max = np.sqrt(X_plus[0]) * Omega
    print(f"Maximum growth rate: γ_max / Ω = {gamma_max:.4f}")
    print(f"                     γ_max = {gamma_max:.4f} Ω")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_vA_over_Omega, gamma_over_Omega, 'b-', linewidth=2.5)
    plt.axhline(gamma_max, color='r', linestyle='--', linewidth=1.5, label=f'Max: γ/Ω = {gamma_max:.3f}')
    plt.axvline(0, color='k', linestyle='-', linewidth=0.5)
    plt.axhline(0, color='k', linestyle='-', linewidth=0.5)

    plt.xlabel('$k v_A / \\Omega$', fontsize=14)
    plt.ylabel('$\\gamma / \\Omega$', fontsize=14)
    plt.title('MRI Dispersion Relation (Keplerian Disk, Vertical Field)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)
    plt.ylim(0, 0.5)
    plt.savefig('mri_dispersion.png', dpi=150)
    plt.show()

    # Most unstable wavelength
    idx_max = np.argmax(gamma_over_Omega)
    k_vA_max = k_vA_over_Omega[idx_max] * Omega
    print(f"\nMost unstable mode: k v_A = {k_vA_max:.4f} Ω")
    print(f"Wavelength: λ = 2π / k = {2*np.pi / k_vA_max:.2f} (v_A / Ω)")

mri_dispersion()
```

### 7.2 MRI Growth Rate vs. Field Strength

```python
import numpy as np
import matplotlib.pyplot as plt

def mri_growth_vs_field():
    """
    Plot MRI growth rate as a function of magnetic field strength.
    """
    Omega = 1.0  # Orbital frequency
    rho = 1.0    # Density (normalized)
    mu_0 = 1.0   # Permeability (normalized)

    # Range of magnetic field strengths
    B = np.logspace(-3, 1, 100)

    # Alfvén speed
    v_A = B / np.sqrt(mu_0 * rho)

    # For maximum growth (k v_A → 0), γ_max ≈ 0.75 Ω (Keplerian)
    # But growth rate depends on k v_A / Ω

    # Choose a fixed wavelength: k = Ω / (H) where H ~ c_s / Ω
    # Then k v_A / Ω = v_A / c_s
    # This choice is physically motivated: the disk scale height H is the
    # largest coherent length scale in the disk, so k ~ 1/H is the smallest
    # wavenumber (longest wave) that fits inside the disk.  At this k the
    # MRI wavelength λ_MRI ~ 2π v_A / Ω must resolve within H.
    c_s = 1.0  # Sound speed (normalized)

    # k_vA / Ω = v_A / c_s: when v_A ≪ c_s (weak field, sub-Alfvénic disk)
    # we are in the long-wavelength MRI regime where growth is near-maximum;
    # when v_A ≳ c_s the field becomes too stiff and MRI is suppressed.
    k_vA_over_Omega = v_A / c_s

    # Compute growth rate from dispersion relation
    Y = k_vA_over_Omega**2
    discriminant = (1 + Y)**2 / 4 + 3 * Y
    X_plus = -(1 + Y)/2 + np.sqrt(discriminant)
    gamma_over_Omega = np.where(X_plus > 0, np.sqrt(X_plus), 0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Growth rate vs. B
    ax1.semilogx(B, gamma_over_Omega, 'b-', linewidth=2.5)
    ax1.axhline(0.75, color='r', linestyle='--', linewidth=1, label='Max (k→0): γ/Ω ≈ 0.75')
    ax1.set_xlabel('Magnetic field $B$ (normalized)', fontsize=14)
    ax1.set_ylabel('Growth rate $\\gamma / \\Omega$', fontsize=14)
    ax1.set_title('MRI Growth Rate vs. Magnetic Field Strength', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Wavelength of fastest growing mode
    # λ_MRI = 2π v_A / (Ω γ/Ω) = 2π v_A / γ; this scales linearly with B
    # because stronger field → faster Alfvén speed → longer optimal wavelength.
    # The NaN masking removes the stabilised (γ = 0) field-strength regime so
    # that the log-log plot shows only the window where MRI actually operates.
    lambda_MRI = 2 * np.pi * v_A / (Omega * gamma_over_Omega)
    lambda_MRI = np.where(gamma_over_Omega > 0, lambda_MRI, np.nan)

    ax2.loglog(B, lambda_MRI, 'b-', linewidth=2.5)
    ax2.set_xlabel('Magnetic field $B$ (normalized)', fontsize=14)
    ax2.set_ylabel('MRI wavelength $\\lambda_{MRI} / H$', fontsize=14)
    ax2.set_title('Fastest Growing MRI Wavelength', fontsize=16)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mri_growth_vs_field.png', dpi=150)
    plt.show()

mri_growth_vs_field()
```

### 7.3 Shearing Box Local Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def shearing_box_trajectories():
    """
    Visualize fluid element trajectories in a shearing box.

    Shear flow: v_y = -q Ω x  (Keplerian: q = 3/2)
    """
    Omega = 1.0
    # q = 3/2 for Keplerian: this is the local shear rate dΩ/d ln r = -3/2.
    # It is exactly 3/2 in a Keplerian potential (Ω ∝ r^{-3/2}) and is the
    # single parameter that controls MRI in the shearing-box frame — a
    # sub-Keplerian disk (q < 3/2) grows MRI more slowly.
    q = 1.5  # Keplerian

    # Initial positions
    N_particles = 20
    x0 = np.random.uniform(-2, 2, N_particles)
    y0 = np.random.uniform(-2, 2, N_particles)

    # Time evolution
    t_max = 10.0
    dt = 0.1
    Nt = int(t_max / dt)

    # Trajectories
    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(N_particles):
        x = np.zeros(Nt)
        y = np.zeros(Nt)

        x[0] = x0[i]
        y[0] = y0[i]

        for n in range(Nt - 1):
            # Shear flow: pure azimuthal drift proportional to radial offset.
            # In the co-rotating frame the background velocity is v_y = -q Ω x
            # (inner annuli move forward, outer annuli lag behind).  There is
            # NO radial velocity in pure shear, so x stays constant — this is
            # why all trajectories are vertical lines: each fluid element simply
            # drifts in y at a rate set by its fixed x position.
            v_y = -q * Omega * x[n]

            # Update (Euler)
            x[n+1] = x[n]
            y[n+1] = y[n] + dt * v_y

        ax.plot(x, y, alpha=0.7, linewidth=1.5)
        ax.plot(x[0], y[0], 'go', markersize=5)
        ax.plot(x[-1], y[-1], 'ro', markersize=5)

    ax.set_xlabel('Radial $x$', fontsize=14)
    ax.set_ylabel('Azimuthal $y$', fontsize=14)
    ax.set_title('Fluid Element Trajectories in Shearing Box (Keplerian)', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(['Start', 'End'], fontsize=12)
    plt.savefig('shearing_box_trajectories.png', dpi=150)
    plt.show()

shearing_box_trajectories()
```

### 7.4 Blandford-Payne Wind Solution (Simplified)

```python
import numpy as np
import matplotlib.pyplot as plt

def blandford_payne_wind():
    """
    Simplified model of Blandford-Payne centrifugal wind.

    Assume:
      - Field line shape: z = r tan(θ)
      - Angular momentum conservation: r v_φ = r₀² Ω₀
      - Centrifugal vs. gravitational balance
    """
    # Parameters
    GM = 1.0       # Gravitational parameter (normalized)
    r_0 = 1.0      # Launch radius
    # Keplerian Ω₀ is fixed at r_0 so that the centrifugal potential L²/(2R²)
    # at the launch point exactly balances gravity — any gas anchored to the
    # disk surface is in circular equilibrium before the wind is launched.
    Omega_0 = np.sqrt(GM / r_0**3)  # Keplerian angular velocity at r_0

    # Field line inclination
    theta_deg = np.array([30, 45, 60, 75])  # degrees
    theta = np.radians(theta_deg)

    # Cylindrical radius along field line
    R = np.linspace(r_0, 5*r_0, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for th, th_d in zip(theta, theta_deg):
        # Height z = (R - r_0) tan(θ): a straight field line inclined at θ
        # from the disk normal.  A shallower θ (field nearly vertical) means
        # the gas must climb steeply before R increases significantly, so the
        # centrifugal "slingshot" is weak; a steeper θ (field nearly horizontal)
        # gives a larger lever arm but requires the field to be strongly bent.
        z = (R - r_0) * np.tan(th)

        # Spherical radius
        r_sph = np.sqrt(R**2 + z**2)

        # Angular momentum: L = r₀² Ω₀ is conserved once gas leaves the disk
        # surface and slides along the frozen-in field line.  This constraint
        # is the core of the Blandford-Payne mechanism — the disk rotation is
        # communicated outward along the field line, making the wind carry away
        # angular momentum and allowing the underlying disk to accrete inward.
        L = r_0**2 * Omega_0

        # Azimuthal velocity: v_φ = L/R decreases as R grows, so the centrifugal
        # force (L²/R³) drops faster than gravity (GM/r_sph²), eventually turning
        # the effective potential into a barrier for steep (θ < 30°) field lines.
        v_phi = L / R

        # Centrifugal force (per unit mass)
        F_cent = v_phi**2 / R

        # Gravitational force (radial component)
        F_grav = GM / r_sph**2

        # Effective potential Φ_eff = -GM/r_sph + L²/(2R²): its slope along the
        # field line tells whether gas accelerates (dΦ_eff/dl < 0) or decelerates.
        # For θ > 30° the potential is monotonically decreasing along the field
        # line → gas is always accelerated and escapes to infinity (centrifugal wind).
        # For θ < 30° a local maximum appears → gas is trapped unless thermally
        # driven; this is the Blandford-Payne critical-angle theorem.
        Phi_eff = -GM / r_sph + L**2 / (2 * R**2)

        # Plot field lines
        ax1.plot(R, z, label=f'θ = {th_d}°', linewidth=2)

        # Plot effective potential
        ax2.plot(r_sph, Phi_eff, label=f'θ = {th_d}°', linewidth=2)

    ax1.set_xlabel('Cylindrical radius $R$ (in $r_0$)', fontsize=14)
    ax1.set_ylabel('Height $z$ (in $r_0$)', fontsize=14)
    ax1.set_title('Blandford-Payne Field Line Geometry', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Spherical radius $r$ (in $r_0$)', fontsize=14)
    ax2.set_ylabel('Effective potential $\\Phi_{eff}$', fontsize=14)
    ax2.set_title('Effective Potential for Wind Acceleration', fontsize=16)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('blandford_payne_wind.png', dpi=150)
    plt.show()

    print("Critical angle for centrifugal launch: θ > 30°")
    print("For θ < 30°: effective potential has no barrier → wind cannot be centrifugally driven")

blandford_payne_wind()
```

### 7.5 MRI Turbulence Energy Evolution (Toy Model)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def mri_turbulence_energy():
    """
    Toy model for MRI turbulence energy evolution.

    Equations:
      dE_K/dt = -α Ω E_K + β E_M - ε_K
      dE_M/dt = γ_MRI E_M - β E_M - ε_M

    where:
      E_K: kinetic energy
      E_M: magnetic energy
      γ_MRI: MRI growth rate
      β: energy exchange (Lorentz force)
      ε: dissipation
    """
    # Parameters
    Omega = 1.0
    # gamma_MRI = 0.75 Ω is the maximum MRI growth rate for a Keplerian disk
    # in the limit k v_A → 0 (from the dispersion relation solved earlier).
    # The magnetic energy E_M grows exponentially at this rate in the linear
    # phase until Lorentz-force back-reaction (the β and ε_M terms) saturates it.
    gamma_MRI = 0.75 * Omega  # MRI growth rate
    alpha_visc = 0.01         # Effective viscosity parameter
    beta_exchange = 0.1       # Energy exchange rate
    epsilon_K = 0.05          # Kinetic dissipation
    epsilon_M = 0.05          # Magnetic dissipation

    def dE_dt(E, t):
        E_K, E_M = E

        # E_K equation: kinetic energy is sourced by Lorentz work (β E_M)
        # and drained by viscous dissipation (-α Ω E_K) and ohmic-like losses
        # (-ε_K E_K).  In MRI turbulence simulations the Maxwell stress
        # ⟨-B_r B_φ⟩/μ₀ is the dominant angular-momentum flux, so the β E_M
        # coupling encapsulates how the growing magnetic field stirs the fluid.
        dE_K_dt = -alpha_visc * Omega * E_K + beta_exchange * E_M - epsilon_K * E_K

        # E_M equation: net growth rate is (γ_MRI - β - ε_M).  The β term
        # represents energy leaving the magnetic reservoir to the kinetic
        # reservoir via Lorentz-force work (J × B · v); the ε_M term is Ohmic
        # dissipation.  When γ_MRI > β + ε_M the field grows; saturation
        # occurs when the back-reaction on the flow reduces the effective γ_MRI.
        dE_M_dt = gamma_MRI * E_M - beta_exchange * E_M - epsilon_M * E_M

        return [dE_K_dt, dE_M_dt]

    # Initial conditions: E_M ≪ E_K mimics a disk that already has turbulent
    # kinetic energy (from, e.g., convection or accretion shocks) but only a
    # tiny seed magnetic field — the MRI then amplifies that seed exponentially
    # until the field reaches equipartition with the turbulent pressure.
    E0 = [1.0, 0.01]  # Initial kinetic energy, small magnetic seed

    # Time
    t = np.linspace(0, 100, 1000)

    # Solve
    E = odeint(dE_dt, E0, t)

    E_K = E[:, 0]
    E_M = E[:, 1]
    E_total = E_K + E_M

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(t, E_K, 'b-', linewidth=2, label='Kinetic $E_K$')
    ax1.plot(t, E_M, 'r-', linewidth=2, label='Magnetic $E_M$')
    ax1.plot(t, E_total, 'k--', linewidth=2, label='Total $E_K + E_M$')
    ax1.set_xlabel('Time (in $\\Omega^{-1}$)', fontsize=14)
    ax1.set_ylabel('Energy', fontsize=14)
    ax1.set_title('MRI Turbulence: Energy Evolution', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Semi-log: the exponential (linear MRI) phase appears as a straight line
    # on this scale, making it easy to read off the growth rate directly.
    # The break from linear to nonlinear behaviour is visible as the curve
    # bends over — this is when magnetic energy approaches equipartition and
    # the Lorentz back-reaction term becomes comparable to the MRI drive.
    ax2.semilogy(t, E_K, 'b-', linewidth=2, label='Kinetic $E_K$')
    ax2.semilogy(t, E_M, 'r-', linewidth=2, label='Magnetic $E_M$')
    ax2.set_xlabel('Time (in $\\Omega^{-1}$)', fontsize=14)
    ax2.set_ylabel('Energy (log scale)', fontsize=14)
    ax2.set_title('MRI Turbulence: Energy Evolution (Log Scale)', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mri_turbulence_energy.png', dpi=150)
    plt.show()

    # Saturation values
    print(f"Saturation:")
    print(f"  Kinetic energy: {E_K[-1]:.4f}")
    print(f"  Magnetic energy: {E_M[-1]:.4f}")
    print(f"  Ratio E_M / E_K: {E_M[-1] / E_K[-1]:.4f}")

mri_turbulence_energy()
```

## 8. Summary

**Accretion disk MHD** is governed by the interplay of rotation, gravity, and magnetic fields:

1. **Angular momentum problem:**
   - Keplerian disks are Rayleigh-stable → no hydrodynamic turbulence
   - Material must shed angular momentum to accrete → need magnetic fields

2. **Magnetorotational instability (MRI):**
   - **Balbus & Hawley (1991):** weak magnetic field in differentially rotating disk is unstable
   - **Growth rate:** `γ ~ Ω` (very fast!)
   - **Fastest growing wavelength:** `λ_MRI ~ v_A / Ω`
   - **Instability criterion:** `dΩ²/d ln r < 0` (Keplerian satisfied)
   - **Mechanism:** Magnetic tension acts like a spring, enabling angular momentum transport outward

3. **Angular momentum transport:**
   - **Maxwell stress:** `-B_r B_φ / μ_0` (dominant)
   - **Reynolds stress:** `ρ v_r v_φ` (subdominant)
   - **α-parameter:** `α ~ 0.01-0.1` from MRI turbulence
   - Explains observed accretion rates in X-ray binaries, AGN

4. **Nonlinear saturation:**
   - MRI grows until `B² / (2μ_0) ~ p`
   - Lorentz force modifies flow → turbulent saturation
   - Channel modes, parasitic instabilities, sustained turbulence

5. **Dead zones:**
   - In weakly ionized regions (protoplanetary disks), Ohmic, ambipolar, Hall effects suppress MRI
   - **Layered accretion:** Active near surface, dead at midplane
   - Alternative instabilities may operate

6. **Disk winds and jets:**
   - **Blandford-Payne:** Centrifugal wind launched along inclined magnetic field lines (θ > 30°)
   - **Magnetic tower:** Toroidal field pressure drives and collimates outflow
   - **Applications:** YSO jets, AGN jets, pulsar winds

MRI is one of the most important discoveries in astrophysics, providing the long-sought mechanism for accretion disk turbulence and making possible rapid accretion onto compact objects.

## Practice Problems

1. **MRI Growth Rate:** For a disk at radius `r = 10¹⁰ cm` around a `M = 10 M☉` black hole, compute the orbital frequency `Ω` and the maximum MRI growth rate `γ_max`.

2. **MRI Wavelength:** For `v_A = 10 km/s` and `Ω = 10⁻³ s⁻¹`, estimate the wavelength of the fastest growing MRI mode.

3. **Maxwell Stress:** If `⟨B_r B_φ⟩ = 10⁻² × ⟨p⟩`, what is the effective α parameter?

4. **Accretion Timescale:** For `α = 0.01`, `c_s = 10 km/s`, `H = 10⁹ cm`, `R = 10¹¹ cm`, estimate the accretion timescale `τ_acc ~ R² / ν_eff`.

5. **Equipartition Field:** In a disk with `ρ = 10⁻⁹ g/cm³`, `c_s = 10⁷ cm/s`, compute the magnetic field strength at `β = B² / (8π p) = 1`.

6. **Dead Zone Criterion:** For a protoplanetary disk at 1 AU with ionization fraction `x_e = 10⁻¹³`, temperature `T = 200 K`, density `ρ = 10⁻⁹ g/cm³`, estimate the Ohmic resistivity and magnetic Reynolds number. Is MRI active?

7. **Blandford-Payne Angle:** Why is θ = 30° the critical angle for centrifugal wind launch? (Hint: consider balance of centrifugal and gravitational forces along the field line.)

8. **Jet Power:** For a black hole of mass `M = 10⁹ M☉`, accretion rate `Ṁ = 0.1 Ṁ_Edd`, and jet efficiency `η_jet = 0.1`, estimate the jet power in erg/s.

9. **Python Exercise:** Modify the MRI dispersion relation code to include toroidal field `B_φ` in addition to vertical field. How does the growth rate change?

10. **Advanced:** Implement a 1D vertically-integrated disk evolution model with MRI-driven α-viscosity. Start with a ring of material and watch it spread and accrete over time.

---

**Previous:** [Solar MHD](./11_Solar_MHD.md) | **Next:** [Fusion MHD](./13_Fusion_MHD.md)

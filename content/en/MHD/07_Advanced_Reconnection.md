# 7. Advanced Reconnection

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain the plasmoid instability and its role in fast reconnection at high Lundquist number
2. Understand turbulent reconnection models (Lazarian-Vishniac)
3. Analyze the effects of guide fields on reconnection dynamics
4. Describe relativistic reconnection and its applications
5. Understand three-dimensional reconnection and quasi-separatrix layers
6. Implement numerical models of plasmoid instability and 3D reconnection structures

## 1. The Plasmoid Instability

### 1.1 The Problem with Sweet-Parker at High S

As we saw in Lesson 5, Sweet-Parker reconnection predicts an extremely slow rate at large Lundquist numbers:

$$M_A = S^{-1/2}$$

For the solar corona with $S \sim 10^{14}$:

$$M_A \sim 10^{-7}$$

This is far too slow. However, there's a fundamental issue: **the Sweet-Parker current sheet itself becomes unstable** at high $S$.

### 1.2 Onset of the Plasmoid Instability

The plasmoid instability (also called tearing instability of the current sheet) was first identified by Biskamp (1986) and systematically studied by Loureiro et al. (2007), Bhattacharjee et al. (2009), and others.

**Physical mechanism:**

The Sweet-Parker current sheet has length $L$ and width $\delta \sim L/S$. For large $S$, the sheet becomes very long and thin. Such a configuration is unstable to the **tearing mode**, which breaks the sheet into multiple magnetic islands (plasmoids).

**Critical Lundquist number:**

Linear stability analysis gives a critical Lundquist number:

$$S_c \sim 10^4$$

For $S > S_c$, the Sweet-Parker sheet is unstable and fragments into a chain of plasmoids.

**Growth rate:**

The fastest-growing mode has wavenumber:

$$k_{max} L \sim S^{1/4}$$

and growth rate:

$$\gamma \tau_A \sim S^{1/4}$$

where $\tau_A = L/v_A$ is the Alfvén time. This is much faster than resistive diffusion ($\gamma_{resistive} \sim S^{-1}$).

**Physical picture:**

```
Initial: Sweet-Parker sheet
    ════════════════════════  Current sheet (length L, width δ)


Unstable: Plasmoid formation
    ════O════X════O════X════O════  X-points and O-points
```

The instability creates a **plasmoid chain**: multiple X-points (reconnection sites) and O-points (magnetic islands).

### 1.3 Plasmoid-Dominated Reconnection

Once the plasmoid instability sets in, the reconnection dynamics change fundamentally.

**Cascade to smaller scales:**

Each plasmoid can itself become unstable (recursive instability), creating smaller plasmoids. This leads to a **hierarchy of scales**:

- Primary X-points (original)
- Secondary plasmoids (size $\sim \delta$)
- Tertiary plasmoids (smaller), etc.

The cascade continues until resistivity becomes important at small scales.

**Effective reconnection rate:**

Although each individual X-point may still reconnect at the local Sweet-Parker rate, the **total** reconnection rate is much faster because:

1. **Multiple X-points**: Many reconnection sites work in parallel
2. **Shorter current sheets**: Each segment has length $\ell \sim L/N$ where $N \sim S^{1/4}$ is the number of plasmoids

The effective Lundquist number for each segment is:

$$S_{eff} = \frac{\ell v_A}{\eta} \sim \frac{L v_A}{N \eta} \sim \frac{S}{S^{1/4}} = S^{3/4}$$

The local reconnection rate at each X-point is:

$$M_{A,local} \sim S_{eff}^{-1/2} \sim S^{-3/8}$$

But there are $N \sim S^{1/4}$ X-points, so the total flux reconnected per unit time is:

$$M_{A,total} \sim N \cdot M_{A,local} \sim S^{1/4} \cdot S^{-3/8} = S^{-1/8}$$

This is **much faster** than Sweet-Parker ($S^{-1/2}$)!

**Asymptotic behavior:**

In the limit $S \to \infty$, if the cascade continues to kinetic scales, the reconnection rate becomes **independent of $S$**:

$$M_A \sim \text{const} \sim 0.01\text{–}0.1$$

This resolves the reconnection rate problem for high-$S$ plasmas.

### 1.4 Numerical Simulations

**2D resistive MHD simulations:**

- Loureiro et al. (2007): Demonstrated plasmoid instability in 2D MHD, confirmed $\gamma \propto S^{1/4}$
- Bhattacharjee et al. (2009): Developed scaling theory
- Huang & Bhattacharjee (2010, 2012): Showed transition from Sweet-Parker to plasmoid-dominated regime
- Uzdensky et al. (2010): Plasmoid instability in relativistic reconnection

**Key findings:**

1. For $S < 10^4$: Sweet-Parker reconnection is stable
2. For $S > 10^4$: Plasmoid instability sets in
3. For $S \gg 10^6$: Fully developed plasmoid-dominated regime, $M_A \sim 0.01$

**Observational evidence:**

- **Solar flares**: Supra-arcade downflows (SADs) in SOHO and SDO images are interpreted as plasmoids
- **Magnetotail**: Earthward and tailward-moving flux ropes (plasmoids) observed by spacecraft
- **Tokamaks**: Sawtooth crashes show bursts of small-scale structures

### 1.5 Python Example: Plasmoid Instability Growth Rate

```python
import numpy as np
import matplotlib.pyplot as plt

# Lundquist number range
S = np.logspace(2, 10, 200)

# Critical Lundquist number
S_c = 1e4

# Growth rate scaling
# Below S_c: resistive growth (very slow)
# γ ∝ S^{-1} because resistive tearing requires the resistive diffusion
# time to act across the current sheet; at low S the sheet is diffusively
# stable and any perturbation decays on the Alfvén time.
gamma_resistive = 0.01 * S**(-1)

# Above S_c: plasmoid instability
# γ ∝ S^{1/4} — this S-scaling encodes the key physics: as the Lundquist
# number rises, the Sweet-Parker sheet becomes thinner (δ ~ L/S), making
# it more susceptible to tearing.  The S^{1/4} law comes from linear
# stability analysis of a Sweet-Parker-width current sheet.
gamma_plasmoid = np.where(S > S_c, 0.1 * S**(1/4), gamma_resistive)

# Alfven time (normalized)
tau_A = 1.0

# Growth rate in units of 1/tau_A
gamma_norm = gamma_plasmoid / tau_A

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Growth rate
ax = axes[0]
ax.loglog(S, gamma_resistive, label='Resistive (no plasmoids): $\\gamma \\propto S^{-1}$',
          linewidth=2, linestyle='--', color='blue')
ax.loglog(S, gamma_plasmoid, label='Plasmoid instability: $\\gamma \\propto S^{1/4}$',
          linewidth=2.5, color='red')
ax.axvline(S_c, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.text(S_c, 1e-4, f'$S_c \\sim {S_c:.0e}$', fontsize=12, rotation=90, va='bottom',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

ax.set_xlabel('Lundquist number $S$', fontsize=14)
ax.set_ylabel('Growth rate $\\gamma \\tau_A$', fontsize=14)
ax.set_title('Plasmoid Instability Growth Rate', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Panel 2: Reconnection rate scaling
ax = axes[1]

# Sweet-Parker
# M_A ~ S^{-1/2} because the reconnection rate is limited by resistive
# diffusion across the entire current sheet — the bottleneck that makes
# Sweet-Parker hopelessly slow for solar and astrophysical parameters.
M_SP = S**(-0.5)

# Plasmoid-mediated
# M_A ~ S^{-1/8} — plasmoids break the sheet into N ~ S^{1/4} segments,
# each shorter and thus faster.  The exponent -1/8 = 1/4 - 3/8 reflects
# the gain from having many parallel X-points overcoming each local rate.
M_plasmoid = np.where(S > S_c, 0.01 * S**(-1/8), M_SP)

# Hall reconnection (constant)
# In collisionless (Hall) reconnection the rate saturates at ~0.1 because
# the Hall term decouples ions from electrons at the ion inertial length,
# enabling a much thicker effective diffusion region independent of Rm.
M_Hall = 0.1 * np.ones_like(S)

ax.loglog(S, M_SP, label='Sweet-Parker: $M_A \\propto S^{-1/2}$',
          linewidth=2, linestyle='--', color='blue')
ax.loglog(S, M_plasmoid, label='Plasmoid-mediated: $M_A \\propto S^{-1/8}$',
          linewidth=2.5, color='red')
ax.loglog(S, M_Hall, label='Hall (collisionless): $M_A \\sim 0.1$',
          linewidth=2, linestyle='-.', color='green')

ax.axvline(S_c, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5)
ax.text(1e9, 0.015, 'Typical observed rate', fontsize=11, color='gray')

ax.set_xlabel('Lundquist number $S$', fontsize=14)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=14)
ax.set_title('Reconnection Rate: Sweet-Parker vs Plasmoid-Mediated', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-8, 1)

plt.tight_layout()
plt.savefig('plasmoid_instability_scaling.png', dpi=150)
plt.show()

# Print transition properties
print("Plasmoid Instability Transition")
print("=" * 50)
print(f"Critical Lundquist number: S_c ~ {S_c:.0e}")
print(f"\nAt S = {S_c:.0e}:")
print(f"  Growth rate: γ τ_A ~ {gamma_plasmoid[np.argmin(np.abs(S - S_c))]:.3f}")
print(f"  Reconnection rate: M_A ~ {M_plasmoid[np.argmin(np.abs(S - S_c))]:.2e}")

S_high = 1e8
idx_high = np.argmin(np.abs(S - S_high))
print(f"\nAt S = {S_high:.0e} (solar corona):")
print(f"  Number of plasmoids: N ~ S^(1/4) ~ {S_high**(1/4):.0f}")
print(f"  Growth rate: γ τ_A ~ {gamma_plasmoid[idx_high]:.1f}")
print(f"  Reconnection rate: M_A ~ {M_plasmoid[idx_high]:.3f}")
print(f"  (Compare Sweet-Parker: M_A ~ {M_SP[idx_high]:.2e})")
```

### 1.6 Plasmoid Size Distribution

In fully developed plasmoid-mediated reconnection, there is a distribution of plasmoid sizes, from large to small.

**Power-law distribution:**

Simulations and theoretical models suggest a power-law size distribution:

$$N(w) \propto w^{-\alpha}$$

where $N(w)$ is the number of plasmoids with width $w$, and $\alpha \sim 1$–2 depending on the regime.

**Largest plasmoids:**

The largest plasmoids have size:

$$w_{max} \sim \delta \sim L / S$$

(the original current sheet width).

**Smallest plasmoids:**

The cascade terminates at the scale where resistivity (or kinetic effects) become important:

$$w_{min} \sim \eta / v_A \sim L / S$$

Wait, this seems the same as $\delta$! The key is that in the plasmoid-dominated regime, the effective resistivity or kinetic scale is different from the naive estimate.

Actually, the minimum scale is set by:
- In resistive MHD: diffusion scale $\sim \sqrt{\eta t}$
- In kinetic plasma: ion skin depth $d_i$ or electron skin depth $d_e$

## 2. Turbulent Reconnection

### 2.1 Lazarian-Vishniac Model

Lazarian & Vishniac (1999, LV99) proposed a radical idea: **turbulence** in the plasma can enable fast reconnection independent of resistivity.

**Key idea:**

In a turbulent plasma, magnetic field lines undergo **random walk** (stochastic wandering). This effectively broadens the reconnection region, allowing field lines to diffuse out faster.

**Model setup:**

Consider reconnection in a turbulent medium with:
- Turbulent velocity $\delta v_l$ at scale $l$
- Turbulent magnetic field perturbation $\delta B_l$
- Background reconnecting field $B_0$

**Field line wandering:**

A field line traced over distance $L$ undergoes random displacements. The r.m.s. displacement is:

$$\delta x \sim \frac{\delta B_l}{B_0} l \left( \frac{L}{l} \right)^{1/2}$$

This is larger than the Sweet-Parker width $\delta \sim L/S$ if turbulence is strong enough.

**Effective diffusion region:**

The turbulent wandering makes the effective width of the diffusion region:

$$\delta_{eff} \sim \delta x \gg \delta_{SP}$$

**Reconnection rate:**

The reconnection rate becomes:

$$M_A \sim \frac{\delta_{eff}}{L} \sim \frac{\delta B}{B_0} \left( \frac{l}{L} \right)^{1/2}$$

If the turbulence is trans-Alfvénic ($\delta v_l \sim v_A$, implying $\delta B_l \sim B_0$), then:

$$M_A \sim \left( \frac{l}{L} \right)^{1/2}$$

For turbulence on scales $l \sim 0.01 L$:

$$M_A \sim 0.1$$

**Key result:** The reconnection rate is **independent of resistivity**, depending only on the turbulence properties.

### 2.2 Conditions for Turbulent Reconnection

For the LV99 model to apply:

1. **Turbulence must exist**: Pre-existing turbulence or self-generated (reconnection itself drives turbulence)
2. **Strong turbulence**: $\delta v_l / v_A \sim 1$ (trans-Alfvénic)
3. **Large scales**: Turbulence injection scale $l$ comparable to $L$

**Applications:**

- **Molecular clouds**: Star-forming regions have strong supersonic turbulence
- **Galaxy clusters**: Turbulent ICM (intracluster medium)
- **Solar wind**: Alfvénic turbulence is ubiquitous
- **Accretion disks**: MRI-driven turbulence

**Debate:**

The LV99 model is controversial. Critics argue:
- Turbulence may be damped near the X-point
- Simulations show different behavior
- The model assumes pre-existing turbulence, but how does it arise?

However, the idea that turbulence can facilitate reconnection has gained traction, especially in astrophysical contexts.

### 2.3 Reconnection-Driven Turbulence

Reconnection itself can generate turbulence through:

- **Plasmoid instability**: Creates fluctuations and flows
- **Kelvin-Helmholtz instability**: In the outflow jets
- **Streaming instabilities**: From energetic particles

This **self-generated turbulence** can then feed back and enhance reconnection, creating a positive feedback loop.

## 3. Guide Field Reconnection

### 3.1 What is a Guide Field?

So far, we've considered **anti-parallel reconnection**: the reconnecting field components are opposite, with no component along the current direction.

A **guide field** $B_g$ is a magnetic field component **parallel to the reconnection current** (out-of-plane in 2D):

```
Reconnecting field:    B_x (reverses across sheet)
Current direction:     J_z (out of plane)
Guide field:           B_g = B_z (uniform, out of plane)
```

**Normalized guide field:**

$$B_g / B_0$$

where $B_0$ is the reconnecting field strength.

- $B_g = 0$: anti-parallel (or null-guide) reconnection
- $B_g / B_0 \ll 1$: weak guide field
- $B_g / B_0 \sim 1$: moderate guide field
- $B_g / B_0 \gg 1$: strong guide field (component reconnection)

### 3.2 Effects of Guide Field

The guide field profoundly affects reconnection dynamics:

**1. Breaks symmetry:**

Anti-parallel reconnection is symmetric (up-down). A guide field breaks this symmetry.

**2. Modifies outflow:**

The outflow velocity direction is tilted. In anti-parallel reconnection, outflows are perpendicular to the inflow. With a guide field, outflows are oblique.

**3. Suppresses plasmoid instability:**

A strong guide field stabilizes the current sheet against the plasmoid instability. The critical Lundquist number increases:

$$S_c(B_g) \sim S_c(0) \cdot \left( 1 + \frac{B_g^2}{B_0^2} \right)^{3/2}$$

**4. Affects particle acceleration:**

- **Anti-parallel**: Particles can be accelerated by the reconnection electric field (direct field-aligned acceleration)
- **Guide field**: Particles gyrate around $B_g$, changing the acceleration mechanism (Fermi reflection, curvature drift)

**5. Changes Hall field structure:**

In collisionless reconnection, the Hall quadrupolar field is modified by the guide field. For strong $B_g$, the Hall field is suppressed.

### 3.3 Reconnection Rate vs Guide Field

Simulations show that the reconnection rate depends on the guide field strength:

$$M_A(B_g) \approx \frac{M_A(0)}{1 + B_g^2 / B_0^2}$$

For $B_g \gg B_0$, reconnection becomes very slow ("component reconnection").

**Physical interpretation:**

The guide field increases the magnetic tension, making it harder to stretch and break field lines.

### 3.4 Applications

Guide field reconnection is relevant in:

- **Solar corona**: Coronal loops often have a strong axial field
- **Magnetopause**: The magnetosheath field has a component parallel to the current
- **Tokamaks**: Reconnection in 3D can involve guide fields
- **Magnetotail**: During northward IMF, there can be a guide field at the magnetopause

## 4. Relativistic Reconnection

### 4.1 When is Reconnection Relativistic?

Reconnection becomes **relativistic** when:

1. **Magnetically dominated plasma**: Magnetization $\sigma \gg 1$, where:

$$\sigma = \frac{B^2}{\mu_0 \rho c^2} = \frac{v_A^2}{c^2} \cdot \gamma_{th}$$

   For $\sigma \gg 1$, the magnetic energy density exceeds the rest-mass energy density.

2. **Relativistic flows**: Outflow speeds $v \sim c$, Lorentz factors $\Gamma > 1$

3. **Relativistic particles**: Particle energies $\gamma m c^2 \gg m c^2$

**Where does this occur?**

- **Pulsar magnetospheres**: $\sigma \sim 10^4$–$10^7$
- **Magnetars**: $\sigma \sim 10^{10}$
- **AGN jets**: $\sigma \sim 1$–$10$ (moderate)
- **Gamma-ray bursts**: $\sigma \sim 10$–$100$ (or higher)
- **Black hole magnetospheres**: Near the event horizon

### 4.2 Relativistic MHD Equations

The relativistic MHD equations involve the **stress-energy tensor** $T^{\mu\nu}$ and **electromagnetic tensor** $F^{\mu\nu}$.

**Energy-momentum conservation:**

$$\partial_\mu T^{\mu\nu} = 0$$

where:

$$T^{\mu\nu} = (\rho c^2 + u + p + b^2) \frac{u^\mu u^\nu}{c^2} + (p + b^2/2) g^{\mu\nu} - b^\mu b^\nu$$

and $b^\mu$ is the four-vector magnetic field, $u^\mu$ is the four-velocity.

**Ideal Ohm's law:**

In the plasma rest frame:

$$E^{\mu} + (u \times B)^\mu = 0$$

**Reconnection electric field:**

In relativistic reconnection, the electric field in the plasma frame is:

$$E' \sim \Gamma v B \sim v_A B$$

where $\Gamma$ is the bulk Lorentz factor of the inflow.

### 4.3 Reconnection Rate in Relativistic Regime

Surprisingly, **relativistic reconnection is also fast**, with:

$$M_A \equiv \frac{v_{in}}{c} \sim 0.1$$

This is similar to non-relativistic collisionless reconnection!

**Why?**

The key physics is similar:
- **Two-scale structure**: Ions (or pairs) decouple on scales $\sim$ skin depth
- **Electron-scale dissipation**: Electrons control reconnection at small scales
- **Fast rate**: Independent of resistivity for large $S$

**Differences:**

- Outflow can reach $v_{out} \sim c$ (Lorentz factor $\Gamma_{out} \sim$ few to 10)
- Magnetic field compression in outflow is limited by relativistic effects
- Particle acceleration is more efficient (power-law tails)

### 4.4 Applications: Pulsar Winds and GRBs

**Pulsar wind nebulae:**

The Crab Nebula is powered by the Crab pulsar. The pulsar wind has:

$$\sigma_{wind} \sim 10^4 \text{ (near pulsar)} \to 0.01\text{–}0.1 \text{ (at termination shock)}$$

**The sigma problem:** How is magnetic energy converted to particle energy?

**Answer:** Reconnection in the striped wind (alternating polarity) dissipates magnetic energy.

- Reconnection rate: $M_A \sim 0.1$
- Particle acceleration: Non-thermal $\gamma$-ray emission
- Flares: Crab flares (2011) attributed to reconnection events

**Gamma-ray bursts:**

In GRB jets, relativistic reconnection can:

- Dissipate magnetic energy → prompt gamma-ray emission
- Accelerate electrons → synchrotron radiation
- Produce rapid variability → plasmoid ejections

Recent simulations (Sironi, Spitkovsky, Werner, Uzdensky) show that relativistic reconnection efficiently produces power-law particle distributions, consistent with GRB spectra.

### 4.5 Python Example: Relativistic Reconnection Outflow

```python
import numpy as np
import matplotlib.pyplot as plt

# Magnetization parameter
sigma = np.logspace(-1, 4, 100)

# Alfven speed (non-relativistic)
v_A_nonrel = 1  # Normalized to c

# Relativistic Alfven speed
# v_A = c√(σ/(1+σ)) — the +1 in the denominator accounts for the rest-mass
# energy density ρc², which resists acceleration even in a magnetically
# dominated plasma; as σ→∞, v_A→c but never exceeds it (causality).
v_A_rel = np.sqrt(sigma / (1 + sigma))  # In units of c

# Outflow speed (approximate, from simulations)
# Non-relativistic: v_out ~ v_A
v_out_nonrel = v_A_nonrel * np.ones_like(sigma)

# Relativistic: v_out ~ c for large sigma
# Factor 0.9 accounts for the fact that some magnetic energy is converted
# to enthalpy (thermal energy) rather than bulk kinetic energy during
# reconnection — the outflow cannot reach exactly v_A because of this loss.
v_out_rel = 0.9 * v_A_rel  # Slightly less than v_A due to compression

# Lorentz factor of outflow
# Γ = (1 - v²/c²)^{-1/2} diverges as v→c, so even a small increase in
# v near c produces a large jump in Γ — the key reason that relativistic
# reconnection is so effective at energizing particles in pulsar winds.
gamma_out_rel = 1 / np.sqrt(1 - v_out_rel**2)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Outflow velocity
ax = axes[0]
ax.semilogx(sigma, v_A_rel, label='Relativistic Alfvén speed $v_A/c$', linewidth=2.5, color='blue')
ax.semilogx(sigma, v_out_rel, label='Relativistic outflow $v_{out}/c$', linewidth=2.5, linestyle='--', color='red')
ax.axhline(1, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.text(1e3, 1.05, 'Speed of light', fontsize=12, color='black')

ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.text(1, 0.1, '$\\sigma = 1$', fontsize=12, rotation=90, va='bottom', color='gray')

ax.set_xlabel('Magnetization $\\sigma = B^2/(\\mu_0 \\rho c^2)$', fontsize=14)
ax.set_ylabel('Velocity (units of $c$)', fontsize=14)
ax.set_title('Relativistic Reconnection Outflow Velocity', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.2)

# Panel 2: Lorentz factor
ax = axes[1]
ax.loglog(sigma, gamma_out_rel, linewidth=2.5, color='purple')
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.axhline(1, color='black', linestyle=':', alpha=0.7)

# Mark example regimes
ax.axvline(10, color='orange', linestyle=':', alpha=0.7)
ax.text(10, 0.5, 'GRB jets\n$\\sigma \\sim 10$', fontsize=11, ha='center', color='orange',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.axvline(1e6, color='green', linestyle=':', alpha=0.7)
ax.text(1e6, 0.5, 'Pulsar\nwind', fontsize=11, ha='center', color='green',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlabel('Magnetization $\\sigma$', fontsize=14)
ax.set_ylabel('Outflow Lorentz factor $\\Gamma_{out}$', fontsize=14)
ax.set_title('Outflow Lorentz Factor vs Magnetization', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1e2)

plt.tight_layout()
plt.savefig('relativistic_reconnection_outflow.png', dpi=150)
plt.show()

# Print example values
print("Relativistic Reconnection Outflow Properties")
print("=" * 60)
sigma_examples = [0.1, 1, 10, 100, 1000, 1e6]
for sig in sigma_examples:
    v_a = np.sqrt(sig / (1 + sig))
    v_out = 0.9 * v_a
    gamma = 1 / np.sqrt(1 - v_out**2)
    print(f"σ = {sig:>8.1e}:  v_A/c = {v_a:.4f},  v_out/c = {v_out:.4f},  Γ_out = {gamma:>6.2f}")
```

## 5. Three-Dimensional Reconnection

### 5.1 Limitations of 2D Models

All the models discussed so far assume **2D geometry**: variation in $x$ and $y$, but invariance in $z$.

In reality, reconnection is **three-dimensional**:

- Magnetic field has all three components
- Current sheets are not infinite in extent
- Reconnection regions are localized

**Key 3D effects:**

1. **Finite extent**: Reconnection regions have finite length in the third dimension
2. **Oblique fields**: Magnetic field can be oblique to the current sheet
3. **Spine-fan topology**: 3D nulls have complex structure (not simple X-points)
4. **Flux tube interactions**: Individual flux tubes reconnect, not entire sheets
5. **Quasi-separatrix layers (QSLs)**: Generalization of separatrices

### 5.2 Magnetic Nulls in 3D

In 3D, a magnetic null is a point where $\mathbf{B} = 0$. Near a null, the field can be linearized:

$$\mathbf{B} = \mathbf{M} \cdot \mathbf{r}$$

where $\mathbf{M}$ is the Jacobian matrix. The eigenvalues of $\mathbf{M}$ determine the null type.

**Types of 3D nulls:**

1. **Radial null**: All eigenvalues have the same sign (source or sink) — **unstable**, not observed in force-free fields

2. **Spiral null**: One real eigenvalue, two complex conjugate — field spirals around the null

3. **Proper null**: Three real eigenvalues, two with one sign (fan plane), one opposite (spine) — **most common**

**Spine-fan structure:**

```
           Spine (1D)
               |
               |
        Fan plane (2D)
```

The **spine** is a field line through the null (1D). The **fan** is a surface of field lines through the null (2D).

**Reconnection at a 3D null:**

Reconnection occurs in the fan plane (separator reconnection) or along the spine (spine reconnection).

### 5.3 Quasi-Separatrix Layers (QSLs)

In 2D, **separatrices** are field lines that separate regions of different topology. In 3D, exact separatrices are rare.

Instead, **quasi-separatrix layers (QSLs)** are thin layers where field line connectivity changes rapidly.

**Definition:**

The **squashing factor** $Q$ measures how much a bundle of field lines is squashed:

$$Q = \frac{|\nabla_\perp \phi|^2}{|\sin \theta|}$$

where $\phi$ is the field line mapping and $\theta$ is the angle between field line and surface.

High $Q$ (e.g., $Q > 2$) indicates a QSL.

**Properties:**

- QSLs are surfaces (2D in 3D space)
- Field lines within a QSL undergo strong shear
- Currents concentrate in QSLs
- Reconnection preferentially occurs in QSLs

**Applications:**

- **Solar corona**: Observed flare ribbons often trace QSLs
- **Tokamaks**: Edge localized modes (ELMs) involve QSL dynamics
- **Magnetosphere**: Magnetopause reconnection is inherently 3D

### 5.4 Slip-Running Reconnection

In 3D, reconnection need not occur at a single X-point. Instead, reconnection can **slip** along field lines.

**Slip-running reconnection:**

Imagine two flux tubes intersecting at an angle. Reconnection starts at one point and then **propagates** along the intersection line (the separator).

This is called **slip-running** or **zipper reconnection**.

**Observational evidence:**

- **Solar eruptions**: Flare ribbons often show propagating brightenings (slipping motion)
- **Magnetic clouds**: CME flux ropes show signatures of progressive reconnection

### 5.5 Python Example: 3D Null Point Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D magnetic field with a null point
# Example: Spine-fan null

def magnetic_field_null_3d(x, y, z):
    """
    Create a spine-fan null:
    Eigenvalues: (+a, +a, -2a) to satisfy div B = 0
    """
    # The eigenvalue ratio (+a, +a, -2a) is chosen so that ∇·B = a+a-2a = 0
    # — the divergence-free constraint is a fundamental law, not an option.
    # Two positive eigenvalues define the fan plane (field converges toward
    # the null in x-y), while the one negative eigenvalue defines the spine
    # (field diverges along z), giving the classic spine-fan topology.
    a = 1.0
    Bx = a * x
    By = a * y
    Bz = -2 * a * z
    return Bx, By, Bz

# Grid
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
z = np.linspace(-2, 2, 15)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Magnetic field
Bx, By, Bz = magnetic_field_null_3d(X, Y, Z)

# Magnitude
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

# Plot
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Quiver plot (subsample for clarity)
# We subsample by skip=2 because the full 15³ grid produces ~3375 arrows,
# which overwhelms the plot and hides the topological structure; a coarser
# sampling still reveals the global field pattern without clutter.
skip = 2
ax.quiver(X[::skip, ::skip, ::skip], Y[::skip, ::skip, ::skip], Z[::skip, ::skip, ::skip],
          Bx[::skip, ::skip, ::skip], By[::skip, ::skip, ::skip], Bz[::skip, ::skip, ::skip],
          length=0.3, normalize=True, color='blue', alpha=0.6)

# Mark the null point
ax.scatter([0], [0], [0], color='red', s=200, marker='o', label='Null point')

# Spine (along z-axis, Bz direction)
# The spine is the unique field line that passes through the null; all other
# field lines asymptote to the fan plane, making the spine the axis about
# which reconnection at a 3D null is topologically organized.
z_spine = np.linspace(-2, 2, 50)
x_spine = np.zeros_like(z_spine)
y_spine = np.zeros_like(z_spine)
ax.plot(x_spine, y_spine, z_spine, 'r-', linewidth=3, label='Spine (field line through null)')

# Fan plane (z=0 plane)
# The fan is a 2D separatrix surface: field lines that cross the fan change
# their topological connectivity, making the fan the preferred site for
# current buildup and reconnection in 3D configurations.
theta_fan = np.linspace(0, 2*np.pi, 100)
r_fan = 1.5
x_fan = r_fan * np.cos(theta_fan)
y_fan = r_fan * np.sin(theta_fan)
z_fan = np.zeros_like(theta_fan)
ax.plot(x_fan, y_fan, z_fan, 'g-', linewidth=3, label='Fan (field lines in z=0 plane)')

# Field lines in fan
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    r_line = np.linspace(0.1, 1.8, 20)
    x_line = r_line * np.cos(angle)
    y_line = r_line * np.sin(angle)
    z_line = np.zeros_like(r_line)
    ax.plot(x_line, y_line, z_line, 'g--', linewidth=1, alpha=0.5)

ax.set_xlabel('X', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_zlabel('Z', fontsize=13)
ax.set_title('3D Magnetic Null: Spine-Fan Structure', fontsize=16, weight='bold')
ax.legend(fontsize=11)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plt.tight_layout()
plt.savefig('3d_null_spine_fan.png', dpi=150)
plt.show()

# Plot field magnitude
fig = plt.figure(figsize=(12, 5))

# XY plane
ax = fig.add_subplot(121)
z_idx = len(z) // 2
contour = ax.contourf(X[:, :, z_idx], Y[:, :, z_idx], B_mag[:, :, z_idx], levels=20, cmap='viridis')
ax.streamplot(X[:, :, z_idx], Y[:, :, z_idx], Bx[:, :, z_idx], By[:, :, z_idx],
              color='white', linewidth=1, density=1.5)
ax.plot(0, 0, 'ro', markersize=12)
plt.colorbar(contour, ax=ax, label='$|\\mathbf{B}|$')
ax.set_xlabel('X', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Fan Plane (z=0): Field Magnitude', fontsize=14)
ax.set_aspect('equal')

# XZ plane
ax = fig.add_subplot(122)
y_idx = len(y) // 2
contour = ax.contourf(X[:, y_idx, :], Z[:, y_idx, :], B_mag[:, y_idx, :], levels=20, cmap='plasma')
ax.streamplot(X[:, y_idx, :], Z[:, y_idx, :], Bx[:, y_idx, :], Bz[:, y_idx, :],
              color='white', linewidth=1, density=1.5)
ax.plot(0, 0, 'ro', markersize=12)
plt.colorbar(contour, ax=ax, label='$|\\mathbf{B}|$')
ax.set_xlabel('X', fontsize=13)
ax.set_ylabel('Z', fontsize=13)
ax.set_title('Spine Direction (y=0): Field Magnitude', fontsize=14)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('3d_null_field_magnitude.png', dpi=150)
plt.show()
```

### 5.6 Observational Signatures of 3D Reconnection

**Solar observations:**

- **Slipping magnetic reconnection**: Flare ribbons show slipping motion along the polarity inversion line (PIL), indicating reconnection propagating along a separator
- **Circular ribbon flares**: Trace the fan of a 3D null
- **Spine-related jets**: Confined to the spine field line

**Magnetospheric observations:**

- **Patchy reconnection**: Reconnection at the magnetopause is localized in 3D, not uniform along the magnetopause
- **FTEs (Flux Transfer Events)**: 3D flux ropes generated by reconnection

## Summary

We explored advanced topics in magnetic reconnection:

1. **Plasmoid instability**: At high Lundquist number ($S > 10^4$), the Sweet-Parker current sheet becomes unstable, fragmenting into a chain of plasmoids and X-points. The growth rate scales as $\gamma \propto S^{1/4}$. Plasmoid-mediated reconnection gives a much faster rate, $M_A \propto S^{-1/8}$, approaching a constant $\sim 0.01$–$0.1$ for very large $S$. This resolves the reconnection rate problem for astrophysical plasmas.

2. **Turbulent reconnection**: The Lazarian-Vishniac model posits that turbulence causes stochastic field line wandering, effectively broadening the diffusion region and enabling fast reconnection independent of resistivity. The reconnection rate depends on turbulence properties, $M_A \sim (l/L)^{1/2}$. While debated, this model is relevant in turbulent astrophysical environments.

3. **Guide field effects**: A magnetic field component parallel to the reconnection current (guide field) breaks symmetry, suppresses the plasmoid instability, and reduces the reconnection rate. Strong guide fields lead to slow "component reconnection." Guide fields also modify particle acceleration mechanisms and the Hall magnetic field structure.

4. **Relativistic reconnection**: In magnetically dominated plasmas ($\sigma \gg 1$), reconnection is relativistic. Despite the extreme conditions, the reconnection rate remains fast, $M_A \sim 0.1$, similar to non-relativistic collisionless reconnection. Outflows can reach $v \sim c$ with Lorentz factors $\Gamma \sim$ few to 10. Applications include pulsar winds, magnetars, GRB jets, and AGN. Relativistic reconnection efficiently accelerates particles to non-thermal distributions.

5. **Three-dimensional reconnection**: Real reconnection is 3D. 3D magnetic nulls have spine-fan structure (not simple X-points). Reconnection can occur at separators (intersection of separatrices) or in quasi-separatrix layers (QSLs), where field line connectivity changes rapidly. Slip-running (zipper) reconnection propagates along separators. Solar flare ribbons and magnetospheric flux transfer events show 3D signatures.

These advanced topics show that reconnection is a rich, multi-scale, often turbulent phenomenon. The transition from laminar Sweet-Parker to plasmoid-dominated, turbulent, or kinetic reconnection explains the universally observed fast rates.

## Practice Problems

1. **Plasmoid instability onset**:
   a) For a solar flare current sheet with $L = 10^9$ m, $v_A = 10^6$ m/s, $\eta = 10^{-4}$ Ω·m, calculate $S$.
   b) Is this above or below the critical $S_c \sim 10^4$?
   c) Estimate the number of plasmoids: $N \sim S^{1/4}$.

2. **Plasmoid growth rate**:
   a) Using the scaling $\gamma \tau_A \sim S^{1/4}$, calculate the growth rate for $S = 10^{12}$ and $\tau_A = 1000$ s.
   b) How does this compare to the resistive diffusion time $\tau_{diff} \sim L^2 / \eta$?

3. **Plasmoid-mediated reconnection rate**:
   a) Plot $M_A$ vs $S$ for both Sweet-Parker ($S^{-1/2}$) and plasmoid-mediated ($S^{-1/8}$) scaling, for $S = 10^4$ to $10^{16}$.
   b) At what $S$ is the plasmoid rate 10 times faster than Sweet-Parker?

4. **Turbulent reconnection (LV99)**:
   a) In a molecular cloud with turbulence injection scale $l = 0.1 L$, estimate the reconnection rate $M_A \sim (l/L)^{1/2}$.
   b) If $L = 1$ pc and $v_A = 1$ km/s, what is the reconnection time?
   c) Compare to the star formation time scale (~Myr).

5. **Guide field suppression**:
   a) The reconnection rate with guide field is $M_A(B_g) \approx M_A(0) / (1 + B_g^2/B_0^2)$. If $M_A(0) = 0.1$ and $B_g = B_0$, what is $M_A(B_g)$?
   b) For $B_g = 3 B_0$, what is $M_A(B_g)$?
   c) Plot $M_A$ vs $B_g/B_0$ for $0 \le B_g/B_0 \le 5$.

6. **Relativistic Alfvén speed**:
   a) Show that the relativistic Alfvén speed is $v_A = c \sqrt{\sigma/(1+\sigma)}$.
   b) For $\sigma = 0.1, 1, 10, 100$, calculate $v_A/c$.
   c) At what $\sigma$ is $v_A = 0.9c$?

7. **Relativistic outflow Lorentz factor**:
   a) If the reconnection outflow is $v_{out} = 0.95c$, calculate the Lorentz factor $\Gamma = 1/\sqrt{1 - v^2/c^2}$.
   b) For a pulsar wind with $\sigma = 10^4$, estimate the outflow speed and Lorentz factor.

8. **Sigma problem in pulsars**:
   a) A pulsar wind has $\sigma = 10^6$ near the light cylinder. If 99% of the magnetic energy is converted to particle energy via reconnection, what is the final $\sigma$?
   b) Is this sufficient to explain the observed $\sigma \sim 0.01$ at the termination shock?
   c) What additional dissipation might be needed?

9. **3D null eigenvalues**:
   a) For a spine-fan null with eigenvalues $(a, a, -2a)$, verify that $\nabla \cdot \mathbf{B} = 0$.
   b) Write the field components $\mathbf{B} = (ax, ay, -2az)$ and sketch the field lines.
   c) Identify the spine (1D) and fan (2D) structures.

10. **QSL squashing factor**:
    a) Research the definition of the squashing factor $Q$. Why is high $Q$ associated with strong currents?
    b) In solar flare observations, flare ribbons often trace regions with $Q > 2$. Why?
    c) How would you compute $Q$ numerically from a 3D magnetic field?

## Navigation

Previous: [Reconnection Applications](./06_Reconnection_Applications.md) | Next: [MHD Turbulence](./08_MHD_Turbulence.md)

# 8. MHD Turbulence

## Learning Objectives

By the end of this lesson, you should be able to:

1. Review hydrodynamic turbulence and the Kolmogorov K41 theory
2. Understand the Iroshnikov-Kraichnan (IK) theory of MHD turbulence
3. Explain the Goldreich-Sridhar critical balance theory and anisotropic cascade
4. Work with Elsässer variables and their role in MHD turbulence
5. Describe energy cascade, intermittency, and structure functions
6. Analyze solar wind turbulence observations
7. Implement numerical models of MHD turbulence spectra

## 1. Review of Hydrodynamic Turbulence

### 1.1 The Turbulence Problem

Turbulence is ubiquitous in nature, from coffee stirring to galactic dynamics. It is characterized by:

- **Chaotic, irregular motion**: Unpredictable, sensitive to initial conditions
- **Multi-scale structure**: Eddies within eddies (Richardson cascade)
- **Enhanced mixing**: Transport far exceeding molecular diffusion
- **Energy dissipation**: Conversion of kinetic energy to heat at small scales

The fundamental difficulty: the **Navier-Stokes equations** are nonlinear, making turbulence analytically intractable.

$$\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla) \mathbf{v} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{v} + \mathbf{f}$$

Turbulence involves a huge range of scales, from the **energy injection scale** $L$ (largest eddies) to the **dissipation scale** $\eta$ (Kolmogorov scale), where viscosity dominates.

### 1.2 Kolmogorov 1941 (K41) Theory

Kolmogorov (1941) developed a statistical theory of turbulence based on dimensional analysis and universality.

**Key assumptions:**

1. **Statistical isotropy and homogeneity**: No preferred direction or location (locally)
2. **Scale separation**: $L \gg \eta$ (high Reynolds number $Re \gg 1$)
3. **Inertial range**: Scales $\eta \ll \ell \ll L$ where energy is transferred without dissipation
4. **Local energy transfer**: Energy cascades from large to small scales

**Energy cascade:**

Energy is injected at large scales (e.g., by stirring) at rate $\epsilon$ (energy per unit time per unit mass). This energy **cascades** to smaller scales through eddy breakup, eventually dissipating at the Kolmogorov scale.

**Dimensional analysis:**

In the inertial range, the only relevant parameters are the energy cascade rate $\epsilon$ and the scale $\ell$. The velocity fluctuation at scale $\ell$ is:

$$v_\ell \sim (\epsilon \ell)^{1/3}$$

The eddy turnover time at scale $\ell$ is:

$$\tau_\ell \sim \ell / v_\ell \sim \ell^{2/3} / \epsilon^{1/3}$$

**Energy spectrum:**

The energy per unit wavenumber is:

$$E(k) \sim \epsilon^{2/3} k^{-5/3}$$

where $k \sim 1/\ell$ is the wavenumber. This is the famous **Kolmogorov $-5/3$ spectrum**.

**Velocity structure function:**

The $p$-th order structure function is:

$$S_p(\ell) = \langle |\delta v(\ell)|^p \rangle$$

where $\delta v(\ell) = v(\mathbf{x} + \boldsymbol{\ell}) - v(\mathbf{x})$ is the velocity increment over distance $\ell$.

For K41:

$$S_p(\ell) \sim (\epsilon \ell)^{p/3}$$

In particular, $S_2(\ell) \sim \epsilon^{2/3} \ell^{2/3}$ is consistent with the $k^{-5/3}$ spectrum (Fourier transform).

### 1.3 Limitations of K41

K41 is remarkably successful, but it has limitations:

- **Assumes isotropy**: Real turbulence often has anisotropy (shear, rotation, stratification)
- **Ignores intermittency**: Turbulence is not self-similar; extreme events are more common than Gaussian statistics predict
- **Local cascade**: Non-local interactions can occur
- **Neglects coherent structures**: Vortices, shocks, etc.

Despite these limitations, K41 provides a baseline for comparison.

### 1.4 Reynolds Number and Kolmogorov Scale

The **Reynolds number** measures the ratio of inertial to viscous forces:

$$Re = \frac{v L}{\nu}$$

For turbulence, $Re \gg 1$.

The **Kolmogorov scale** $\eta$ is where viscosity becomes important:

$$\eta = \left( \frac{\nu^3}{\epsilon} \right)^{1/4}$$

The ratio of scales is:

$$\frac{L}{\eta} \sim Re^{3/4}$$

For atmospheric turbulence ($Re \sim 10^6$), this gives $L/\eta \sim 10^{4.5} \sim 30,000$ — a vast range!

## 2. MHD Turbulence: Early Theories

### 2.1 Why is MHD Turbulence Different?

In magnetohydrodynamics, the magnetic field introduces:

1. **Anisotropy**: The field direction is a preferred direction
2. **Alfvén waves**: Propagating disturbances (absent in hydrodynamics)
3. **Reduced nonlinearity**: Alfvén wave interactions are weaker than hydrodynamic eddy interactions
4. **Magnetic tension**: Suppresses perpendicular motions

These effects fundamentally alter the turbulent cascade.

### 2.2 Iroshnikov-Kraichnan (IK) Theory

Iroshnikov (1963) and Kraichnan (1965) independently proposed the first theory of MHD turbulence.

**Key idea:**

Turbulent eddies are made of colliding **Alfvén wave packets**. Alfvén waves propagate along the mean field $\mathbf{B}_0$ at the Alfvén speed $v_A$. Wave packets traveling in opposite directions collide and interact weakly.

**Collision time:**

An eddy at scale $\ell_\perp$ (perpendicular to $\mathbf{B}_0$) interacts over a time:

$$\tau_{coll} \sim \frac{\ell_\parallel}{v_A}$$

where $\ell_\parallel$ is the parallel scale. If we assume **isotropy** ($\ell_\parallel \sim \ell_\perp \sim \ell$):

$$\tau_{coll} \sim \frac{\ell}{v_A}$$

**Cascade time:**

The energy cascades when an eddy undergoes many collisions. The number of collisions needed is:

$$N_{coll} \sim \frac{\tau_{eddy}}{\tau_{coll}}$$

where $\tau_{eddy} \sim \ell / v_\ell$ is the eddy turnover time.

Energy cascades when $N_{coll} \sim 1$ collision has occurred, but in MHD, the interaction is weak, so many collisions are needed:

$$N_{coll} \sim \left( \frac{v_A}{v_\ell} \right)^2$$

(The square comes from the weak interaction strength.)

The cascade time is then:

$$\tau_{cascade} \sim N_{coll} \cdot \tau_{coll} \sim \frac{v_A}{v_\ell^2} \cdot \ell$$

**Dimensional analysis:**

Setting the cascade time equal to the eddy turnover time (energy transfer):

$$\frac{\ell}{v_\ell} \sim \frac{v_A \ell}{v_\ell^2}$$

Solving:

$$v_\ell \sim v_A$$

This just says eddies move at the Alfvén speed, which is not very informative!

**Correct IK scaling:**

The energy cascade rate is:

$$\epsilon \sim \frac{v_\ell^2}{\tau_{cascade}} \sim \frac{v_\ell^4}{v_A \ell}$$

Solving for $v_\ell$:

$$v_\ell \sim (\epsilon v_A \ell)^{1/4}$$

The energy spectrum is:

$$E(k) \sim (\epsilon v_A)^{1/2} k^{-3/2}$$

This is the **Iroshnikov-Kraichnan $-3/2$ spectrum**, shallower than Kolmogorov's $-5/3$.

### 2.3 Problems with IK Theory

Numerical simulations and observations showed that:

1. **IK assumes isotropy**: But MHD turbulence is strongly **anisotropic** (elongated along $\mathbf{B}_0$)
2. **Observed spectra**: Often closer to $-5/3$ than $-3/2$
3. **Solar wind**: Shows $k^{-5/3}$ in the inertial range

The IK theory was a good first step but failed to capture the essential anisotropy of MHD turbulence.

## 3. Goldreich-Sridhar Critical Balance Theory

### 3.1 Anisotropy in MHD Turbulence

Observations and simulations showed that MHD turbulence is **anisotropic**:

- **Perpendicular cascade**: Eddies cascade to smaller scales $\perp$ to $\mathbf{B}_0$
- **Parallel elongation**: Eddies are elongated along $\mathbf{B}_0$

Goldreich & Sridhar (1995, GS95) proposed a theory incorporating this anisotropy.

**Key idea: Critical balance**

At each scale $\ell_\perp$ (perpendicular size), the **nonlinear cascade time** is comparable to the **Alfvén wave period**:

$$\tau_{nl} \sim \tau_A$$

where:

$$\tau_{nl} \sim \frac{\ell_\perp}{v_{\ell_\perp}}$$

is the eddy turnover time, and:

$$\tau_A \sim \frac{\ell_\parallel}{v_A}$$

is the Alfvén wave crossing time along the parallel direction.

### 3.2 Derivation of GS95 Scaling

**Critical balance condition:**

$$\frac{\ell_\perp}{v_{\ell_\perp}} \sim \frac{\ell_\parallel}{v_A}$$

**Kolmogorov-like cascade in $\perp$ direction:**

Assume a Kolmogorov cascade in the perpendicular direction:

$$v_{\ell_\perp} \sim (\epsilon \ell_\perp)^{1/3}$$

**Relating $\ell_\parallel$ and $\ell_\perp$:**

From critical balance:

$$\ell_\parallel \sim \frac{v_A \ell_\perp}{v_{\ell_\perp}} \sim \frac{v_A \ell_\perp}{(\epsilon \ell_\perp)^{1/3}}$$

Simplify:

$$\ell_\parallel \sim v_A \ell_\perp^{2/3} / \epsilon^{1/3}$$

Or, normalizing by the outer scale $L$:

$$\frac{\ell_\parallel}{L} \sim \left( \frac{\ell_\perp}{L} \right)^{2/3}$$

(assuming $v_A \sim (epsilon L)^{1/3}$ at the outer scale, which is consistent).

In terms of wavenumbers $k_\parallel \sim 1/\ell_\parallel$, $k_\perp \sim 1/\ell_\perp$:

$$k_\parallel \propto k_\perp^{2/3}$$

**Anisotropic cascade:**

Eddies become increasingly elongated along $\mathbf{B}_0$ as they cascade to smaller $\ell_\perp$:

$$\frac{\ell_\parallel}{\ell_\perp} \propto \ell_\perp^{-1/3} \to \infty \quad \text{as } \ell_\perp \to 0$$

At small scales, eddies are ribbon-like, with $\ell_\parallel \gg \ell_\perp$.

**Perpendicular energy spectrum:**

The energy spectrum in the perpendicular direction is:

$$E(k_\perp) \propto k_\perp^{-5/3}$$

the same as Kolmogorov! The cascade is Kolmogorov-like in the $\perp$ direction, but highly anisotropic.

**Parallel spectrum:**

Due to the anisotropy relation $k_\parallel \propto k_\perp^{2/3}$, the parallel spectrum is steeper.

### 3.3 Physical Interpretation

**Why critical balance?**

If $\tau_{nl} \ll \tau_A$, eddies cascade quickly before Alfvén waves have time to propagate — the cascade would be nearly hydrodynamic (Kolmogorov).

If $\tau_{nl} \gg \tau_A$, Alfvén waves propagate many times before eddies evolve — energy is trapped in waves, not cascading effectively.

**Critical balance** is the marginally unstable state where both processes are equally important, allowing efficient energy transfer.

**Alfvénic turbulence:**

GS95 assumes turbulence is made of Alfvén waves (Elsässer modes), which we'll discuss shortly.

### 3.4 Observational Support

**Solar wind:**

- Perpendicular spectrum: $E(k_\perp) \propto k_\perp^{-5/3}$ (consistent with GS95)
- Anisotropy: Fluctuations are elongated along $\mathbf{B}_0$
- Critical balance: Observations suggest $\tau_{nl} \sim \tau_A$

**Simulations:**

Numerical MHD simulations confirm:
- Anisotropic cascade with $k_\parallel \propto k_\perp^{2/3}$
- Perpendicular $k^{-5/3}$ spectrum
- Critical balance maintained across scales

GS95 is now the **standard model** of strong MHD turbulence.

## 4. Elsässer Variables

### 4.1 Definition

Elsässer (1950) introduced variables that symmetrize the MHD equations for incompressible, constant-density MHD.

Define:

$$\mathbf{z}^+ = \mathbf{v} + \frac{\mathbf{B}}{\sqrt{\mu_0 \rho}}$$

$$\mathbf{z}^- = \mathbf{v} - \frac{\mathbf{B}}{\sqrt{\mu_0 \rho}}$$

These represent **counter-propagating Alfvén waves**:

- $\mathbf{z}^+$: Alfvén wave propagating in the $+\mathbf{B}_0$ direction
- $\mathbf{z}^-$: Alfvén wave propagating in the $-\mathbf{B}_0$ direction

**Velocity and magnetic field in terms of Elsässer variables:**

$$\mathbf{v} = \frac{\mathbf{z}^+ + \mathbf{z}^-}{2}$$

$$\frac{\mathbf{B}}{\sqrt{\mu_0 \rho}} = \frac{\mathbf{z}^+ - \mathbf{z}^-}{2}$$

### 4.2 MHD Equations in Elsässer Form

For incompressible MHD with uniform density, the equations become:

$$\frac{\partial \mathbf{z}^+}{\partial t} + (\mathbf{z}^- \cdot \nabla) \mathbf{z}^+ = -\nabla P^+ + \nu \nabla^2 \mathbf{z}^+ + \eta \nabla^2 \mathbf{z}^+$$

$$\frac{\partial \mathbf{z}^-}{\partial t} + (\mathbf{z}^+ \cdot \nabla) \mathbf{z}^- = -\nabla P^- + \nu \nabla^2 \mathbf{z}^- + \eta \nabla^2 \mathbf{z}^-$$

$$\nabla \cdot \mathbf{z}^+ = 0, \quad \nabla \cdot \mathbf{z}^- = 0$$

where $P^\pm$ are generalized pressures.

**Key observation:**

The nonlinear term in the $\mathbf{z}^+$ equation involves $\mathbf{z}^-$, and vice versa. This shows that **$\mathbf{z}^+$ and $\mathbf{z}^-$ interact with each other**, not with themselves.

Physically: Counter-propagating Alfvén waves collide and interact; co-propagating waves do not.

### 4.3 Balanced vs Imbalanced Turbulence

**Balanced turbulence:**

If $|\mathbf{z}^+| \approx |\mathbf{z}^-|$, the turbulence is **balanced**. This is the case assumed in GS95.

**Imbalanced turbulence:**

If $|\mathbf{z}^+| \neq |\mathbf{z}^-|$, the turbulence is **imbalanced**. For example, if $|\mathbf{z}^+| \gg |\mathbf{z}^-|$:

- $\mathbf{z}^+$ dominates the energy
- $\mathbf{z}^-$ is a weak minority population
- Interaction rate is reduced (fewer collisions)

**Solar wind:**

The solar wind is typically imbalanced:

$$\frac{E(z^-)}{E(z^+)} \sim 0.2\text{–}0.5$$

This imbalance affects the cascade rate and may lead to different scaling.

### 4.4 Energy in Elsässer Variables

The total energy density is:

$$E = \frac{1}{2} \rho v^2 + \frac{B^2}{2\mu_0} = \frac{\rho}{4} \left( |\mathbf{z}^+|^2 + |\mathbf{z}^-|^2 \right)$$

The energy in each Elsässer component:

$$E^+ = \frac{\rho}{4} |\mathbf{z}^+|^2, \quad E^- = \frac{\rho}{4} |\mathbf{z}^-|^2$$

In balanced turbulence, $E^+ \approx E^-$. In imbalanced turbulence, one dominates.

## 5. Energy Cascade and Intermittency

### 5.1 Direct vs Inverse Cascade

In 3D hydrodynamics, energy cascades **directly** from large to small scales (forward cascade).

In 2D hydrodynamics, energy cascades **inversely** from small to large scales, while enstrophy cascades forward. This is due to conservation of both energy and enstrophy in 2D.

**MHD:**

In 3D MHD, there are conserved quantities:
- **Total energy**: $E = E_{kin} + E_{mag}$
- **Cross-helicity**: $H_c = \int \mathbf{v} \cdot \mathbf{B} \, dV$
- **Magnetic helicity**: $H_m = \int \mathbf{A} \cdot \mathbf{B} \, dV$ (in certain cases)

**Direct cascade:**

In most cases, energy cascades **forward** (large to small scales) in 3D MHD, similar to hydrodynamics.

**Inverse cascade:**

If magnetic helicity is present and conserved, there can be an **inverse cascade of magnetic helicity** to large scales, while energy still cascades forward. This is relevant in dynamos (Lesson 9).

### 5.2 Intermittency

**What is intermittency?**

Intermittency refers to the departure from self-similar scaling. In real turbulence:
- Intense, localized structures (current sheets, vortex filaments) exist
- Dissipation is concentrated in small regions
- Structure functions show anomalous scaling: $S_p(\ell) \propto \ell^{\zeta_p}$ with $\zeta_p \neq p/3$

**Multifractal model:**

The dissipation field is a multifractal, characterized by a spectrum of singularities. Different regions have different local scaling exponents.

**Consequences:**

- **Non-Gaussian statistics**: PDFs of velocity increments have extended tails
- **Anomalous scaling**: Deviations from K41 predictions
- **Coherent structures**: Current sheets, magnetic flux tubes, shocks

Intermittency is more pronounced in MHD than in hydrodynamic turbulence, due to the anisotropy and current sheet formation.

### 5.3 Structure Functions

The $p$-th order structure function is:

$$S_p(\ell) = \langle |\delta z(\ell)|^p \rangle$$

where $\delta z(\ell) = z(\mathbf{x} + \boldsymbol{\ell}) - z(\mathbf{x})$ is the Elsässer variable increment.

**K41 prediction:**

$$S_p(\ell) \propto \ell^{p/3}$$

**Intermittent turbulence:**

$$S_p(\ell) \propto \ell^{\zeta_p}$$

where $\zeta_p$ deviates from $p/3$, especially for large $p$ (rare, intense events).

**Measurement:**

Structure functions are computed from spacecraft data (solar wind) or simulation outputs. They provide insight into the cascade and intermittency.

### 5.4 Python Example: Structure Function Scaling

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic turbulent velocity field
# (Simplified: assume a power-law spectrum)

np.random.seed(42)

# Spatial grid
N = 512
L = 1.0
x = np.linspace(0, L, N, endpoint=False)

# Wavenumber
k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
k[0] = 1e-10  # Avoid division by zero

# Power spectrum: E(k) ~ k^{-5/3}
# We prescribe the K41 spectrum rather than evolving the Navier-Stokes
# equations because this gives us a clean test case for structure functions;
# the goal here is to measure ζ_p, not to simulate the dynamics.
P_k = k**(-5/3)
P_k[0] = 0  # Zero mean

# Random phases
# Assigning uniformly random phases makes the synthetic field statistically
# homogeneous and isotropic — the same assumption underpinning K41 theory —
# so deviations of ζ_p from p/3 in our measurement are due to finite-sample
# noise, not physical intermittency.
phase = np.exp(2j * np.pi * np.random.rand(N))

# Velocity in Fourier space
v_k = np.sqrt(P_k) * phase

# Velocity in real space
v = np.fft.ifft(v_k).real

# Normalize
v = v / np.std(v)

# Compute structure functions
# Logarithmic spacing of lags captures both the inertial range (small ℓ)
# and the energy-containing scales (large ℓ) — linear spacing would waste
# most samples in the inertial range where the physics is self-similar.
lags = np.logspace(np.log10(L/N), np.log10(L/4), 30)
orders = [1, 2, 3, 4, 5, 6]
S_p = {p: [] for p in orders}

for lag in lags:
    lag_idx = int(lag / (L/N))
    if lag_idx == 0:
        lag_idx = 1
    delta_v = v[lag_idx:] - v[:-lag_idx]

    for p in orders:
        # Taking absolute value before the p-th power is essential: without
        # it, odd-order S_p would be zero by symmetry (the field has zero
        # mean), giving no information about the velocity increment PDF.
        S_p[p].append(np.mean(np.abs(delta_v)**p))

# Convert to arrays
for p in orders:
    S_p[p] = np.array(S_p[p])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Structure functions
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
for i, p in enumerate(orders):
    ax.loglog(lags, S_p[p], 'o-', label=f'$S_{p}$', color=colors[i], markersize=5)

# K41 predictions
for i, p in enumerate(orders):
    K41_slope = p / 3
    S_K41 = 0.1 * lags**K41_slope  # Arbitrary normalization
    ax.loglog(lags, S_K41, '--', color=colors[i], alpha=0.5)

ax.set_xlabel('Lag $\\ell$', fontsize=13)
ax.set_ylabel('Structure function $S_p(\\ell)$', fontsize=13)
ax.set_title('Structure Functions (K41 Scaling)', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Scaling exponents
ax = axes[1]

# Fit power-law to extract exponents
# A linear fit in log-log space directly gives the scaling exponent ζ_p;
# deviations of ζ_p from the K41 line p/3 at high p diagnose intermittency
# because rare intense events (current sheets) contribute disproportionately
# to high-order moments even when they occupy little volume.
zeta_p = []
for p in orders:
    # Fit log(S_p) vs log(ell)
    coeffs = np.polyfit(np.log10(lags), np.log10(S_p[p]), 1)
    zeta_p.append(coeffs[0])

zeta_K41 = np.array(orders) / 3

ax.plot(orders, zeta_p, 'o-', label='Measured $\\zeta_p$', markersize=8, linewidth=2, color='blue')
ax.plot(orders, zeta_K41, '--', label='K41: $\\zeta_p = p/3$', linewidth=2, color='red')

ax.set_xlabel('Order $p$', fontsize=13)
ax.set_ylabel('Scaling exponent $\\zeta_p$', fontsize=13)
ax.set_title('Scaling Exponents: K41 vs Measured', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('structure_functions_K41.png', dpi=150)
plt.show()

print("Scaling Exponents:")
print(f"{'p':>5} {'ζ_p (measured)':>20} {'ζ_p (K41 = p/3)':>20}")
print("-" * 50)
for p, zeta, zeta_k41 in zip(orders, zeta_p, zeta_K41):
    print(f"{p:>5} {zeta:>20.3f} {zeta_k41:>20.3f}")
```

## 6. Solar Wind Turbulence

### 6.1 The Solar Wind as a Turbulence Laboratory

The **solar wind** is a supersonic, super-Alfvénic stream of plasma flowing from the Sun. It provides an ideal laboratory for studying MHD turbulence:

- **In situ measurements**: Spacecraft (ACE, Wind, Ulysses, PSP, Solar Orbiter) measure $\mathbf{v}$, $\mathbf{B}$, $n$, $T$ at high cadence
- **Large Reynolds numbers**: $Re \sim 10^6$, $R_m \sim 10^6$
- **Extended inertial range**: Decades in scale
- **Imbalanced turbulence**: Outward-propagating waves dominate

### 6.2 Observed Spectral Regimes

Solar wind turbulence exhibits several spectral ranges:

**1. Energy-containing range** ($f < 10^{-4}$ Hz, $\ell > 10^6$ km):

Large-scale structures: coronal mass ejections, stream interaction regions, corotating interaction regions. Not universal.

**2. Inertial range** ($10^{-4} \text{ Hz} < f < f_{ion}$):

Power-law spectrum:

$$E(f) \propto f^{-\alpha}$$

with $\alpha \approx 5/3$ (consistent with GS95 or K41).

This range spans 2–3 decades in frequency.

**3. Dissipation range** ($f > f_{ion}$):

At the **ion gyrofrequency** $f_{ion} \sim 0.1\text{–}1$ Hz (at 1 AU), the spectrum steepens:

$$E(f) \propto f^{-\beta}$$

with $\beta \approx 2.5\text{–}3$. This is where ion-scale kinetic physics (gyro-resonances, Landau damping) becomes important.

**4. Electron dissipation range** ($f > f_{electron}$):

At even higher frequencies ($f \sim 100$ Hz), electron-scale physics dominates. Recent high-resolution data (MMS) are exploring this regime.

### 6.3 Spectral Break at Ion Scales

The **spectral break** at ion scales is a key feature. The break frequency corresponds to the ion gyroradius or ion inertial length:

$$f_{break} \sim \frac{v_{sw}}{2\pi d_i}$$

where $v_{sw}$ is the solar wind speed and $d_i = c/\omega_{pi}$ is the ion inertial length.

**Physical interpretation:**

- **Below $f_{break}$**: MHD turbulence (fluid description valid)
- **Above $f_{break}$**: Kinetic turbulence (kinetic effects: cyclotron resonance, Landau damping)

**Heating:**

The dissipation range is where turbulent energy is converted to heat. The solar wind is observed to be much hotter than adiabatic expansion would predict, suggesting turbulent heating.

### 6.4 Anisotropy in the Solar Wind

Measurements using the **Taylor frozen-in hypothesis** (converting time to space via $\mathbf{k} \cdot \mathbf{v}_{sw} = \omega$) show:

- **Perpendicular spectrum**: $E(k_\perp) \propto k_\perp^{-5/3}$
- **Parallel spectrum**: Steeper (less power at small $\ell_\parallel$)
- **Anisotropy relation**: Approximately $k_\parallel \propto k_\perp^{2/3}$, consistent with GS95

However, precise measurements are challenging due to:
- Single-point measurements (most spacecraft)
- Ambiguity in separating spatial and temporal variations
- Multi-spacecraft missions (Cluster, MMS, PSP-Solar Orbiter) help resolve this

### 6.5 Python Example: Solar Wind Spectrum

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic solar wind spectrum
# Frequency range
f = np.logspace(-5, 2, 500)  # Hz

# Define spectral regimes
f_inertial = 1e-4  # Start of inertial range
f_ion = 0.5        # Ion gyrofrequency (spectral break)
f_electron = 50    # Electron scales

# Energy-containing range: flat or slightly rising
# The slight positive slope captures the large-scale energy reservoir
# (solar wind streams, CME-driven structures) that inject energy into
# the inertial range; the turbulence itself lives at higher frequencies.
E_energy = np.where(f < f_inertial, 1e2 * (f / f_inertial)**0.5, 0)

# Inertial range: -5/3 slope
# The Kolmogorov/GS95 -5/3 slope persists over roughly two decades because
# in this range energy is transferred without dissipation — the "pipeline"
# between the source at large scales and the sink at ion scales.
E_inertial = np.where((f >= f_inertial) & (f < f_ion),
                      1e2 * (f / f_inertial)**(-5/3), 0)

# Dissipation range (ion scales): -2.8 slope
# The steepening to ~-2.8 at ion scales reflects the onset of kinetic
# damping (ion Landau damping, cyclotron resonance): waves of wavelength
# λ ~ ρ_i interact resonantly with ions and deposit energy as heat, breaking
# the self-similar cascade that produced the -5/3 inertial range.
E_dissipation = np.where((f >= f_ion) & (f < f_electron),
                         1e2 * (f_ion / f_inertial)**(-5/3) * (f / f_ion)**(-2.8), 0)

# Electron dissipation: steeper
# At electron scales the slope steepens further to ~-4 because electrons
# also begin to damp the fluctuations; the remaining energy is dissipated
# as electron heating, which is why the solar wind is observed to heat
# electrons differently from ions.
E_electron = np.where(f >= f_electron,
                      1e2 * (f_ion / f_inertial)**(-5/3) * (f_electron / f_ion)**(-2.8) * (f / f_electron)**(-4), 0)

# Total spectrum
E_total = E_energy + E_inertial + E_dissipation + E_electron

# Add noise to make it realistic
# Multiplicative log-normal noise mimics the variance of a real single-point
# spacecraft measurement: the spectrum is a noisy sample from an ensemble,
# and the scatter is proportional to the signal (not additive).
np.random.seed(42)
E_total *= 10**(np.random.normal(0, 0.1, len(f)))

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

ax.loglog(f, E_total, linewidth=2, color='blue', label='Solar wind spectrum (synthetic)')

# Mark regimes
ax.axvline(f_inertial, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.text(f_inertial, 1e-2, 'Inertial range\nstart', fontsize=11, color='green', rotation=90, va='bottom')

ax.axvline(f_ion, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(f_ion, 1e-2, 'Ion gyrofrequency\n(spectral break)', fontsize=11, color='red', rotation=90, va='bottom')

ax.axvline(f_electron, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax.text(f_electron, 1e-2, 'Electron\nscales', fontsize=11, color='purple', rotation=90, va='bottom')

# Reference slopes
f_ref = np.array([2e-4, 2e-1])
E_53 = 1e1 * (f_ref / f_ref[0])**(-5/3)
E_28 = 1e-1 * (f_ref / f_ref[0])**(-2.8)

ax.loglog(f_ref, E_53, 'k--', linewidth=2, alpha=0.6, label='$f^{-5/3}$ (inertial)')
ax.loglog(f_ref, E_28, 'k:', linewidth=2, alpha=0.6, label='$f^{-2.8}$ (dissipation)')

# Annotations
ax.text(1e-4, 5e1, 'Energy-containing\nrange', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(1e-2, 1e-1, 'Inertial range\n(MHD turbulence)', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(5, 1e-5, 'Dissipation range\n(kinetic)', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

ax.set_xlabel('Frequency $f$ (Hz)', fontsize=14)
ax.set_ylabel('Power Spectral Density $E(f)$ (arbitrary units)', fontsize=14)
ax.set_title('Solar Wind Magnetic Field Spectrum', fontsize=16, weight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1e-5, 1e2)
ax.set_ylim(1e-6, 1e3)

plt.tight_layout()
plt.savefig('solar_wind_spectrum.png', dpi=150)
plt.show()
```

### 6.6 Heating and Dissipation

The solar wind temperature decreases slower than adiabatic expansion predicts:

$$T \propto r^{-\gamma}$$

with observed $\gamma \sim 1$ (adiabatic would give $\gamma = 4/3$ for protons).

**Turbulent heating mechanisms:**

1. **Ion cyclotron resonance**: Ions resonate with Alfvén/ion-cyclotron waves, gaining perpendicular energy
2. **Landau damping**: Wave-particle interaction transfers wave energy to parallel particle motion
3. **Stochastic heating**: Particles gain energy from time-varying fields in turbulence
4. **Reconnection**: Dissipation in current sheets formed by turbulence

Determining which mechanism dominates is an active area of research.

## 7. Python Examples: MHD Turbulence Spectra

### 7.1 Comparison of Spectral Models

```python
import numpy as np
import matplotlib.pyplot as plt

# Wavenumber range (perpendicular)
k = np.logspace(-1, 2, 200)

# Kolmogorov (hydrodynamic)
# k^{-5/3} arises from dimensional analysis: in the inertial range the only
# relevant quantities are ε (energy flux) and k, giving E(k) ~ ε^{2/3} k^{-5/3}.
E_K41 = k**(-5/3)

# Iroshnikov-Kraichnan (MHD, isotropic)
# IK assumes isotropic Alfvén wave collisions; each interaction is weakened
# by the ratio (v_ℓ/v_A)², making the cascade slower and the spectrum
# shallower (k^{-3/2}).  IK gets the physics partially right but ignores
# the critical role of anisotropy.
E_IK = k**(-3/2)

# Goldreich-Sridhar (MHD, anisotropic, perpendicular)
# GS95 recovers k^{-5/3} in k_⊥ because perpendicular eddies cascade
# like Kolmogorov while the parallel dynamics are constrained by Alfvén
# wave propagation (critical balance) — distinguishing it from K41 only
# in the anisotropy (k_∥ ∝ k_⊥^{2/3}), not the perpendicular slope.
E_GS = k**(-5/3)

# Normalize at k=1
# Normalizing at k=1 sets a common reference so the plot reveals the slope
# differences rather than arbitrary amplitude offsets between theories.
E_K41 = E_K41 / E_K41[np.argmin(np.abs(k - 1))]
E_IK = E_IK / E_IK[np.argmin(np.abs(k - 1))]
E_GS = E_GS / E_GS[np.argmin(np.abs(k - 1))]

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.loglog(k, E_K41, linewidth=2.5, label='Kolmogorov (K41): $k^{-5/3}$', color='blue')
ax.loglog(k, E_IK, linewidth=2.5, label='Iroshnikov-Kraichnan (IK): $k^{-3/2}$', color='red')
ax.loglog(k, E_GS, linewidth=2.5, linestyle='--', label='Goldreich-Sridhar (GS95): $k_\\perp^{-5/3}$', color='green')

# Reference lines
k_ref = np.array([1, 10])
ax.loglog(k_ref, 1 * k_ref**(-5/3), 'k:', linewidth=2, alpha=0.5, label='$k^{-5/3}$ reference')
ax.loglog(k_ref, 1.5 * k_ref**(-3/2), 'k--', linewidth=2, alpha=0.5, label='$k^{-3/2}$ reference')

ax.set_xlabel('Wavenumber $k$ (or $k_\\perp$)', fontsize=14)
ax.set_ylabel('Energy spectrum $E(k)$ (normalized)', fontsize=14)
ax.set_title('Comparison of Turbulence Spectral Models', fontsize=16, weight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.1, 100)
ax.set_ylim(1e-4, 10)

plt.tight_layout()
plt.savefig('turbulence_spectral_models.png', dpi=150)
plt.show()

# Print spectral indices
print("Spectral Indices:")
print(f"Kolmogorov (K41):           α = -5/3 = {-5/3:.4f}")
print(f"Iroshnikov-Kraichnan (IK):  α = -3/2 = {-3/2:.4f}")
print(f"Goldreich-Sridhar (GS95):   α = -5/3 = {-5/3:.4f} (in k_perp)")
```

### 7.2 Anisotropy Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Perpendicular wavenumber
k_perp = np.logspace(-1, 2, 100)

# Goldreich-Sridhar anisotropy relation
# k_∥ ∝ k_⊥^{2/3} is the signature of critical balance: at each perpendicular
# scale ℓ_⊥ the Alfvén crossing time τ_A = ℓ_∥/v_A equals the eddy turnover
# time τ_nl = ℓ_⊥/δv.  Eddies that violate this balance either cascade
# immediately (τ_nl < τ_A) or become wave-like (τ_nl > τ_A), so the
# turbulence self-organizes to stay exactly on this anisotropy curve.
k_para_GS = k_perp**(2/3)

# Isotropic (IK)
# IK assumes k_∥ = k_⊥ (spherically symmetric energy distribution), which
# neglects the fact that Alfvén waves carry energy preferentially along B_0;
# this is the fundamental flaw that causes IK to predict the wrong spectrum.
k_para_iso = k_perp

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: k_parallel vs k_perp
ax = axes[0]
ax.loglog(k_perp, k_para_GS, linewidth=2.5, label='GS95: $k_\\parallel \\propto k_\\perp^{2/3}$', color='green')
ax.loglog(k_perp, k_para_iso, linewidth=2.5, linestyle='--', label='Isotropic: $k_\\parallel = k_\\perp$', color='blue')

# Shaded region
ax.fill_between(k_perp, k_para_GS, k_para_iso, alpha=0.3, color='yellow', label='Anisotropic regime')

ax.set_xlabel('$k_\\perp$ (perpendicular wavenumber)', fontsize=13)
ax.set_ylabel('$k_\\parallel$ (parallel wavenumber)', fontsize=13)
ax.set_title('Anisotropy in MHD Turbulence', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')

# Panel 2: Aspect ratio
ax = axes[1]
# k_∥/k_⊥ = k_⊥^{-1/3} → 0 as k_⊥ → ∞, meaning small-scale eddies are
# highly elongated along B_0 (ℓ_∥ ≫ ℓ_⊥ in real space); this anisotropy
# makes MHD turbulence fundamentally different from isotropic Navier-Stokes.
aspect_GS = k_para_GS / k_perp  # = k_perp^{-1/3}
aspect_iso = np.ones_like(k_perp)

ax.loglog(k_perp, aspect_GS, linewidth=2.5, label='GS95: $k_\\parallel / k_\\perp \\propto k_\\perp^{-1/3}$', color='green')
ax.loglog(k_perp, aspect_iso, linewidth=2.5, linestyle='--', label='Isotropic: $k_\\parallel / k_\\perp = 1$', color='blue')

ax.set_xlabel('$k_\\perp$ (perpendicular wavenumber)', fontsize=13)
ax.set_ylabel('Aspect ratio $k_\\parallel / k_\\perp$', fontsize=13)
ax.set_title('Eddy Aspect Ratio vs Scale', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')

# Annotation
ax.text(5, 0.05, 'Eddies become elongated\nalong $\\mathbf{B}_0$ at small scales', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('mhd_turbulence_anisotropy.png', dpi=150)
plt.show()

# Print aspect ratios at selected scales
print("Aspect Ratio (k_parallel / k_perp) for GS95:")
print(f"{'k_perp':>10} {'k_para':>10} {'Aspect':>10} {'l_para/l_perp':>15}")
print("-" * 50)
for kp in [0.1, 1, 10, 100]:
    kpa = kp**(2/3)
    aspect = kpa / kp
    ell_aspect = kp / kpa  # Invert for real-space aspect ratio
    print(f"{kp:>10.1f} {kpa:>10.3f} {aspect:>10.3f} {ell_aspect:>15.3f}")
```

## Summary

MHD turbulence is a rich and complex phenomenon:

1. **Kolmogorov K41 theory**: The foundation of turbulence theory, predicting a $k^{-5/3}$ energy spectrum in the inertial range of isotropic hydrodynamic turbulence. Energy cascades from large to small scales, dissipating at the Kolmogorov scale.

2. **Iroshnikov-Kraichnan theory**: The first MHD turbulence theory, assuming isotropic Alfvén wave collisions. Predicts a $k^{-3/2}$ spectrum. However, it fails to capture the strong anisotropy of MHD turbulence and is not supported by observations.

3. **Goldreich-Sridhar (GS95) theory**: The standard model of strong MHD turbulence. Incorporates anisotropy via **critical balance**: the nonlinear cascade time equals the Alfvén wave period at each scale. Predicts $k_\parallel \propto k_\perp^{2/3}$ (eddies elongate along $\mathbf{B}_0$) and $E(k_\perp) \propto k_\perp^{-5/3}$ (Kolmogorov-like in the perpendicular direction). Widely supported by simulations and solar wind observations.

4. **Elsässer variables**: $\mathbf{z}^+ = \mathbf{v} + \mathbf{B}/\sqrt{\mu_0\rho}$ and $\mathbf{z}^- = \mathbf{v} - \mathbf{B}/\sqrt{\mu_0\rho}$ represent counter-propagating Alfvén waves. The MHD equations become symmetric in Elsässer form, clarifying that $\mathbf{z}^+$ and $\mathbf{z}^-$ interact with each other. Balanced turbulence has $E^+ \approx E^-$; imbalanced turbulence (e.g., solar wind) has unequal energies.

5. **Energy cascade and intermittency**: Energy cascades from large to small scales (direct cascade). Intermittency (non-self-similar, multifractal structure) leads to anomalous scaling of structure functions, with intense, localized structures (current sheets, vortices). MHD turbulence is more intermittent than hydrodynamic turbulence.

6. **Solar wind turbulence**: The solar wind is a natural laboratory for MHD turbulence. Observed spectra show a $k^{-5/3}$ inertial range, a spectral break at ion scales, and a steeper dissipation range. Anisotropy and critical balance are confirmed. Turbulent heating explains the observed slow temperature decline. Recent missions (PSP, Solar Orbiter, MMS) are providing unprecedented high-resolution data.

Understanding MHD turbulence is essential for interpreting astrophysical and space plasma observations, modeling turbulent heating and transport, and advancing theories of dynamos, reconnection, and particle acceleration.

## Practice Problems

1. **Kolmogorov scaling**:
   a) For a turbulent flow with energy injection rate $\epsilon = 10^{-3}$ m²/s³ and largest eddy size $L = 1$ m, estimate the velocity at scale $\ell = 0.01$ m.
   b) Calculate the eddy turnover time at this scale.
   c) If the kinematic viscosity is $\nu = 10^{-5}$ m²/s, estimate the Kolmogorov scale $\eta = (\nu^3/\epsilon)^{1/4}$.

2. **Reynolds number**:
   a) For Earth's atmosphere with $L = 1000$ km, $v = 10$ m/s, $\nu = 1.5 \times 10^{-5}$ m²/s, calculate the Reynolds number.
   b) Estimate the ratio $L/\eta$.
   c) How many decades in scale does the inertial range span?

3. **IK vs K41 spectra**:
   a) Plot $E(k)$ vs $k$ for both IK ($k^{-3/2}$) and K41 ($k^{-5/3}$) on a log-log plot for $k = 0.1$ to $100$.
   b) At what wavenumber $k$ do the two spectra differ by a factor of 2 (assuming they are equal at $k=1$)?
   c) Over two decades ($k = 1$ to $100$), which spectrum has more energy?

4. **Goldreich-Sridhar anisotropy**:
   a) If $k_\perp = 100$ m⁻¹, what is $k_\parallel$ according to GS95 ($k_\parallel \propto k_\perp^{2/3}$)? Assume $k_\parallel = k_\perp = 1$ at the outer scale.
   b) What is the aspect ratio $\ell_\parallel / \ell_\perp$?
   c) Sketch the shape of an eddy at this scale.

5. **Elsässer variables**:
   a) Given $\mathbf{v} = (1, 0, 0)$ m/s and $\mathbf{B} = (0, 0.01, 0)$ T in a plasma with $\rho = 10^{-12}$ kg/m³, calculate $\mathbf{z}^+$ and $\mathbf{z}^-$.
   b) Compute the kinetic energy $E_{kin} = \frac{1}{2}\rho v^2$ and magnetic energy $E_{mag} = B^2/(2\mu_0)$.
   c) Verify that $E_{kin} + E_{mag} = \frac{\rho}{4}(|\mathbf{z}^+|^2 + |\mathbf{z}^-|^2)$.

6. **Critical balance**:
   a) In a solar wind with $v_A = 50$ km/s and turbulence injection scale $L = 10^6$ km, estimate the velocity fluctuation at $L$: $v_L \sim v_A$ (by critical balance).
   b) At scale $\ell_\perp = 100$ km, estimate $v_{\ell_\perp}$ using Kolmogorov scaling.
   c) Calculate the parallel scale $\ell_\parallel$ from critical balance.

7. **Structure function**:
   a) Generate a synthetic velocity field with power-law spectrum $E(k) \propto k^{-5/3}$.
   b) Compute the second-order structure function $S_2(\ell) = \langle |\delta v(\ell)|^2 \rangle$ for various lags $\ell$.
   c) Fit a power law $S_2 \propto \ell^{\zeta_2}$ and compare $\zeta_2$ to the K41 prediction $2/3$.

8. **Solar wind spectral break**:
   a) At 1 AU, the solar wind has $n = 10^7$ m⁻³, $B = 5$ nT. Calculate the ion inertial length $d_i = c/\omega_{pi}$.
   b) If the solar wind speed is $v_{sw} = 400$ km/s, estimate the break frequency $f_{break} = v_{sw}/(2\pi d_i)$ using Taylor's hypothesis.
   c) Compare to the observed break frequency ~0.5 Hz.

9. **Turbulent heating rate**:
   a) If the turbulent energy cascade rate is $\epsilon = 10^{-16}$ erg/g/s in the solar wind, how much energy is dissipated per proton per second?
   b) If this heats the protons, estimate the temperature increase over 1 day.
   c) Is this sufficient to explain the slow temperature decline in the solar wind?

10. **Anisotropic energy spectrum**:
    a) In a 2D $k_\perp$-$k_\parallel$ plane, sketch contours of constant energy density $E(k_\perp, k_\parallel)$ for GS95 turbulence.
    b) The energy is concentrated near $k_\parallel \propto k_\perp^{2/3}$. Sketch this "critical balance surface" in $k$-space.
    c) How does this differ from isotropic turbulence (where energy would be on spheres $k_\perp^2 + k_\parallel^2 = \text{const}$)?

## Navigation

Previous: [Advanced Reconnection](./07_Advanced_Reconnection.md) | Next: [Dynamo Theory](./09_Dynamo_Theory.md)

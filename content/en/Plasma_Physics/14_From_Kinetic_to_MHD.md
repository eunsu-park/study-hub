# 14. From Kinetic to MHD

## Learning Objectives

- Understand the systematic reduction from 6D kinetic theory to 3D single-fluid MHD
- Derive single-fluid MHD equations from two-fluid theory by combining species
- Identify the validity conditions and limitations of MHD approximations
- Explain the CGL (Chew-Goldberger-Low) double adiabatic model for collisionless plasmas
- Understand drift-kinetic and gyrokinetic theories as intermediate reductions
- Compare different plasma models and know when to apply each

## 1. The Hierarchy of Plasma Models

### 1.1 Overview: From Full Kinetic to MHD

Plasma physics has a rich hierarchy of models, each with different levels of approximation and computational cost:

```
Full Kinetic (Vlasov-Maxwell)
    ↓  [average over gyration]
Drift-Kinetic (5D)
    ↓  [average over bounce motion / perturbative expansion]
Gyrokinetic (5D, with FLR)
    ↓  [take moments]
Two-Fluid (3D × 2 species)
    ↓  [combine species]
Extended MHD (Hall, FLR, etc.)
    ↓  [drop small terms]
Single-Fluid MHD (3D)
    ↓  [equilibrium, linearize]
MHD Waves, Instabilities
```

Each step down the hierarchy:
- **Reduces** dimensionality or number of variables
- **Simplifies** the equations
- **Loses** some physics
- **Increases** computational efficiency

The art of plasma physics is choosing the right model for the problem at hand.

### 1.2 What Does Each Model Capture?

| Model | Dimensions | Captures | Misses |
|-------|------------|----------|---------|
| **Vlasov-Maxwell** | 6D (r,v,t) | Everything: wave-particle, anisotropy, kinetic instabilities | Computationally prohibitive |
| **Drift-Kinetic** | 5D (R,v∥,μ,t) | Parallel dynamics, trapped particles, collisionless damping | Cyclotron resonance, gyrophase |
| **Gyrokinetic** | 5D (R,v∥,μ,t) | FLR, turbulence, microinstabilities | Fast magnetosonic, compressibility |
| **Two-Fluid** | 3D × 2 species | Hall effect, electron pressure, separate species | Kinetic effects (damping, instabilities) |
| **Hall MHD** | 3D | Whistler, fast reconnection, dispersive waves | Kinetic damping, pressure anisotropy |
| **Resistive MHD** | 3D | Reconnection, resistive instabilities | Fast processes at small scales |
| **Ideal MHD** | 3D | Alfvén/magnetosonic waves, gross equilibria | Reconnection, kinetic physics, small scales |

### 1.3 When to Use Which Model?

**Use Ideal MHD when**:
- Large-scale equilibria and stability (tokamak, stellar atmosphere)
- Low-frequency waves ($\omega \ll \omega_{ci}$)
- Collisional plasmas with isotropic pressure
- Magnetic Reynolds number $R_m \gg 1$

**Use Resistive MHD when**:
- Magnetic reconnection (solar flares, substorms)
- Resistive instabilities (tearing modes)
- Current-driven dynamics

**Use Hall MHD when**:
- Scales approaching $d_i$ (magnetopause, reconnection)
- Fast reconnection with whistler outflow
- Magnetic field generation (dynamo)

**Use Two-Fluid when**:
- Separate electron and ion dynamics are important
- Pressure anisotropy within each species
- Kinetic effects are secondary

**Use Gyrokinetic when**:
- Tokamak turbulence (ion-temperature-gradient modes, trapped-electron modes)
- Microinstabilities with FLR effects
- Collisionless plasmas with weak perturbations

**Use Full Kinetic when**:
- Wave-particle resonances are crucial (Landau damping, cyclotron heating)
- Strongly non-Maxwellian distributions (beam-plasma, runaway electrons)
- Velocity-space instabilities (two-stream, bump-on-tail)

## 2. From Two-Fluid to Single-Fluid MHD

### 2.1 Defining Single-Fluid Variables

Recall the two-fluid equations for species $s$ (electrons $e$, ions $i$):

**Continuity**:
$$\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{u}_s) = 0$$

**Momentum**:
$$m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla p_s + \mathbf{R}_s$$

**Energy** (adiabatic closure):
$$\frac{d}{dt}\left( \frac{p_s}{n_s^\gamma} \right) = 0$$

To derive single-fluid MHD, we define **center-of-mass (fluid) variables**:

**Mass density**:
$$\rho = m_i n_i + m_e n_e \approx m_i n$$

(using quasi-neutrality $n_i \approx n_e \equiv n$ and $m_i \gg m_e$)

**Fluid velocity** (center-of-mass velocity):
$$\mathbf{v} = \frac{m_i n_i \mathbf{u}_i + m_e n_e \mathbf{u}_e}{\rho} \approx \mathbf{u}_i$$

**Total pressure**:
$$p = p_i + p_e$$

**Current density**:
$$\mathbf{J} = e(n_i \mathbf{u}_i - n_e \mathbf{u}_e) \approx en(\mathbf{u}_i - \mathbf{u}_e)$$

**Charge density** (quasi-neutrality):
$$\rho_c = e(n_i - n_e) \approx 0$$

### 2.2 Combining Continuity Equations

Add the electron and ion continuity equations:

$$\frac{\partial n_e}{\partial t} + \nabla \cdot (n_e \mathbf{u}_e) = 0$$
$$\frac{\partial n_i}{\partial t} + \nabla \cdot (n_i \mathbf{u}_i) = 0$$

Multiply the electron equation by $m_e$ and the ion equation by $m_i$, then add:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (m_e n_e \mathbf{u}_e + m_i n_i \mathbf{u}_i) = 0$$

Using $\rho \mathbf{v} = m_i n_i \mathbf{u}_i + m_e n_e \mathbf{u}_e \approx m_i n \mathbf{u}_i$:

$$\boxed{\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0}$$

This is the **mass continuity equation** for single-fluid MHD.

### 2.3 Combining Momentum Equations

Add the electron and ion momentum equations:

$$m_e n_e \frac{d \mathbf{u}_e}{dt} = -e n_e (\mathbf{E} + \mathbf{u}_e \times \mathbf{B}) - \nabla p_e + \mathbf{R}_e$$
$$m_i n_i \frac{d \mathbf{u}_i}{dt} = +e n_i (\mathbf{E} + \mathbf{u}_i \times \mathbf{B}) - \nabla p_i + \mathbf{R}_i$$

The collision terms cancel: $\mathbf{R}_e + \mathbf{R}_i = 0$ (momentum conservation).

The electric field terms cancel (using quasi-neutrality):
$$-e n_e \mathbf{E} + e n_i \mathbf{E} = e(n_i - n_e) \mathbf{E} \approx 0$$

The Lorentz force terms give:
$$-e n_e \mathbf{u}_e \times \mathbf{B} + e n_i \mathbf{u}_i \times \mathbf{B} = e n (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B} = \mathbf{J} \times \mathbf{B}$$

The inertial terms:
$$m_e n_e \frac{d \mathbf{u}_e}{dt} + m_i n_i \frac{d \mathbf{u}_i}{dt} \approx m_i n \frac{d \mathbf{u}_i}{dt} = \rho \frac{d \mathbf{v}}{dt}$$

(neglecting the electron inertia term $m_e n_e d\mathbf{u}_e/dt \ll m_i n_i d\mathbf{u}_i/dt$).

Putting it together:

$$\boxed{\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla p}$$

This is the **momentum equation** for single-fluid MHD.

### 2.4 Ideal Ohm's Law

The key step to ideal MHD is to derive **Ohm's law**. In Lesson 13, we derived the generalized Ohm's law:

$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{en} \mathbf{J} \times \mathbf{B} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$$

In **ideal MHD**, we make the following approximations:

1. **High conductivity** ($\eta \to 0$): neglect resistive term
2. **Large scales** ($L \gg d_i$): neglect Hall term
3. **Slow dynamics**: neglect electron inertia
4. **Negligible electron pressure gradient** (or isotropic electron pressure): neglect pressure term

This gives the **ideal Ohm's law**:

$$\boxed{\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0}$$

This is the **frozen-in condition**: the magnetic field is frozen into the fluid and moves with it.

### 2.5 Faraday's Law and Induction Equation

From Maxwell's equations:
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

Substituting the ideal Ohm's law $\mathbf{E} = -\mathbf{v} \times \mathbf{B}$:

$$\nabla \times (-\mathbf{v} \times \mathbf{B}) = -\frac{\partial \mathbf{B}}{\partial t}$$

Using the vector identity $\nabla \times (\mathbf{A} \times \mathbf{B}) = \mathbf{A}(\nabla \cdot \mathbf{B}) - \mathbf{B}(\nabla \cdot \mathbf{A}) + (\mathbf{B} \cdot \nabla)\mathbf{A} - (\mathbf{A} \cdot \nabla)\mathbf{B}$:

$$\nabla \times (\mathbf{v} \times \mathbf{B}) = \mathbf{v}(\nabla \cdot \mathbf{B}) - \mathbf{B}(\nabla \cdot \mathbf{v}) + (\mathbf{B} \cdot \nabla)\mathbf{v} - (\mathbf{v} \cdot \nabla)\mathbf{B}$$

Since $\nabla \cdot \mathbf{B} = 0$:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) = (\mathbf{B} \cdot \nabla)\mathbf{v} - \mathbf{B}(\nabla \cdot \mathbf{v}) - (\mathbf{v} \cdot \nabla)\mathbf{B}$$

Rearranging:

$$\boxed{\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B})}$$

Or equivalently:

$$\boxed{\frac{d \mathbf{B}}{dt} = (\mathbf{B} \cdot \nabla)\mathbf{v} - \mathbf{B}(\nabla \cdot \mathbf{v})}$$

where $d/dt = \partial/\partial t + \mathbf{v} \cdot \nabla$ is the convective derivative.

This is the **induction equation** (or **magnetic evolution equation**). It describes how the magnetic field evolves as the plasma flows.

### 2.6 Summary: Ideal MHD Equations

The **ideal MHD equations** are:

**Mass continuity**:
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

**Momentum**:
$$\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla p$$

**Induction**:
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B})$$

**Energy** (adiabatic):
$$\frac{d}{dt}\left( \frac{p}{\rho^\gamma} \right) = 0$$

**Ampère's law** (neglecting displacement current):
$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

**No magnetic monopoles**:
$$\nabla \cdot \mathbf{B} = 0$$

These are **8 equations** for **8 unknowns**: $\rho$, $\mathbf{v}$ (3 components), $p$, $\mathbf{B}$ (3 components), given the constraint $\nabla \cdot \mathbf{B} = 0$.

(The electric field $\mathbf{E}$ is determined by Ohm's law: $\mathbf{E} = -\mathbf{v} \times \mathbf{B}$.)

## 3. Validity Conditions for MHD

### 3.1 Low Frequency: $\omega \ll \omega_{ci}$

MHD is a **low-frequency approximation**. The time scale of phenomena must be much longer than the ion cyclotron period:

$$\omega \ll \omega_{ci} = \frac{eB}{m_i}$$

This ensures that ions have time to gyrate and respond to the fields in a fluid-like manner, rather than exhibiting individual particle behavior.

**Example**: For $B = 1$ T, $\omega_{ci} \approx 10^8$ rad/s ($f \approx 16$ MHz). MHD is valid for phenomena slower than ~10 MHz.

### 3.2 Large Scale: $L \gg \rho_i$

The spatial scale must be much larger than the **ion gyroradius**:

$$L \gg \rho_i = \frac{v_{th,i}}{\omega_{ci}}$$

At scales $\lesssim \rho_i$, finite Larmor radius (FLR) effects become important, and MHD breaks down.

**Example**: For $T_i = 10$ keV and $B = 1$ T, $\rho_i \approx 0.5$ cm. MHD is valid for scales $\gg 1$ cm.

### 3.3 Collisional: $\lambda_{mfp} \ll L$

For isotropic pressure (as assumed in ideal MHD), collisions must be frequent enough to isotropize the distribution function:

$$\lambda_{mfp} = v_{th} \tau \ll L$$

where $\tau$ is the collision time.

In collisionless plasmas, the pressure tensor is **anisotropic** ($p_\parallel \neq p_\perp$), requiring a more general closure (e.g., CGL, discussed below).

**Example**: In the solar wind, $\lambda_{mfp} \sim 1$ AU $\gg L$ for any reasonable structure. Standard MHD is not valid—CGL or kinetic models are needed.

### 3.4 Non-Relativistic: $v \ll c$

Plasma flows and thermal velocities must be non-relativistic:

$$v, v_{th} \ll c$$

This allows us to neglect displacement current in Ampère's law and use non-relativistic momentum equations.

**Example**: For $T = 10$ keV, $v_{th,e} \approx 0.04c$ (relativistic corrections ~few percent). For higher temperatures, relativistic MHD is needed.

### 3.5 Quasi-Neutrality: $n_e \approx n_i$

The plasma must be quasi-neutral on the scales of interest:

$$L \gg \lambda_D = \sqrt{\frac{\epsilon_0 k_B T}{n e^2}}$$

This allows us to neglect charge separation and drop the displacement current.

**Example**: For $n = 10^{19}$ m$^{-3}$ and $T = 10$ eV, $\lambda_D \approx 10$ μm. MHD is valid for $L \gg 10$ μm.

### 3.6 High Magnetic Reynolds Number: $R_m \gg 1$

For **ideal MHD** (frozen-in), the magnetic Reynolds number must be large:

$$R_m = \frac{\mu_0 V L}{\eta} \gg 1$$

where $V$ is a characteristic flow velocity, $L$ is a length scale, and $\eta$ is the resistivity.

When $R_m \sim 1$, resistivity becomes important → **resistive MHD**.

**Example**: In a tokamak, $V \sim 100$ m/s, $L \sim 1$ m, $\eta \sim 10^{-8}$ Ω·m → $R_m \sim 10^{10}$. Ideal MHD is excellent.

### 3.7 Validity Regime Summary

```
Ideal MHD is valid when ALL of the following hold:

1. ω << ω_ci           (low frequency)
2. L >> ρ_i            (large scale)
3. λ_mfp << L          (collisional, for isotropic p)
4. v << c              (non-relativistic)
5. L >> λ_D            (quasi-neutral)
6. R_m >> 1            (frozen-in)

Violations → need extended MHD or kinetic models.
```

## 4. CGL (Double Adiabatic) Model

### 4.1 Motivation: Collisionless Magnetized Plasmas

In many astrophysical plasmas (solar wind, magnetosphere, galaxy clusters), the collision mean free path is **enormous**:

$$\lambda_{mfp} \gg L$$

In such plasmas, particles can travel long distances without colliding. The pressure tensor becomes **anisotropic**:

$$\overleftrightarrow{P} = p_\perp \overleftrightarrow{I} + (p_\parallel - p_\perp) \hat{\mathbf{b}} \hat{\mathbf{b}}$$

where $\hat{\mathbf{b}} = \mathbf{B}/B$ and:
- $p_\parallel$: pressure parallel to $\mathbf{B}$
- $p_\perp$: pressure perpendicular to $\mathbf{B}$

Standard MHD assumes $p_\parallel = p_\perp = p$ (isotropic), which is invalid in collisionless plasmas.

### 4.2 Chew-Goldberger-Low (1956) Model

Chew, Goldberger, and Low (CGL) derived a closure for collisionless, strongly magnetized plasmas by assuming **conservation of adiabatic invariants**:

**First adiabatic invariant** (magnetic moment):
$$\mu = \frac{m v_\perp^2}{2B} = \text{const}$$

This implies:
$$\frac{d}{dt}\left( \frac{p_\perp}{n B} \right) = 0$$

**Second adiabatic invariant** (longitudinal action):
$$J = \oint v_\parallel ds = \text{const}$$

For a local fluid element (not bouncing between mirrors), this becomes:
$$\frac{d}{dt}\left( \frac{p_\parallel B^2}{n^3} \right) = 0$$

These are the **CGL equations** (also called **double adiabatic** equations).

### 4.3 CGL Closure Relations

The CGL equations are:

$$\boxed{\frac{d}{dt}\left( \frac{p_\perp}{nB} \right) = 0}$$

$$\boxed{\frac{d}{dt}\left( \frac{p_\parallel B^2}{n^3} \right) = 0}$$

These can be rewritten as:

$$\frac{1}{p_\perp} \frac{dp_\perp}{dt} = \frac{1}{n} \frac{dn}{dt} + \frac{1}{B} \frac{dB}{dt}$$

$$\frac{1}{p_\parallel} \frac{dp_\parallel}{dt} = 3 \frac{1}{n} \frac{dn}{dt} - 2 \frac{1}{B} \frac{dB}{dt}$$

**Physical interpretation**:

- When the field increases ($dB/dt > 0$), $p_\perp$ increases (betatron heating), but $p_\parallel$ decreases (magnetic mirror effect).
- Compression ($dn/dt > 0$) increases both $p_\perp$ and $p_\parallel$.

### 4.4 CGL Pressure Tensor

The momentum equation with the CGL pressure tensor becomes:

$$\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla \cdot \overleftrightarrow{P}$$

where:
$$\nabla \cdot \overleftrightarrow{P} = \nabla p_\perp + (p_\parallel - p_\perp) \left[ \frac{\nabla \cdot \mathbf{B}}{B} \hat{\mathbf{b}} + \frac{(\mathbf{B} \cdot \nabla) \mathbf{B}}{B^2} \right]$$

Using $\nabla \cdot \mathbf{B} = 0$ and $(\mathbf{B} \cdot \nabla)\mathbf{B} = B^2 \boldsymbol{\kappa}$ (where $\boldsymbol{\kappa}$ is the curvature):

$$\nabla \cdot \overleftrightarrow{P} = \nabla p_\perp + (p_\parallel - p_\perp) \boldsymbol{\kappa}$$

So the momentum equation is:

$$\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla p_\perp - (p_\parallel - p_\perp) \boldsymbol{\kappa}$$

The anisotropy creates an extra force $-(p_\parallel - p_\perp) \boldsymbol{\kappa}$ along the field curvature.

### 4.5 CGL Instabilities

The CGL model predicts **pressure-anisotropy-driven instabilities** when:

1. **Mirror instability**: If $p_\perp / p_\parallel$ is too large
   $$\frac{p_\perp}{p_\parallel} > 1 + \frac{1}{\beta_\perp}$$
   where $\beta_\perp = 2\mu_0 p_\perp / B^2$.

   The plasma creates local **magnetic mirrors** (enhanced $B$ regions) to reduce $p_\perp$.

2. **Firehose instability**: If $p_\parallel / p_\perp$ is too large
   $$\frac{p_\parallel}{p_\perp} > 1 + \frac{2}{\beta_\parallel}$$
   where $\beta_\parallel = 2\mu_0 p_\parallel / B^2$.

   The magnetic field line "kinks" like a firehose under pressure.

These instabilities are observed in the solar wind and Earth's magnetosheath.

### 4.6 Limitations of CGL

1. **No heat flux**: CGL assumes no parallel heat conduction. In reality, heat flux is important on long parallel scales.

2. **No collisions**: CGL is for collisionless plasmas. Adding even weak collisions modifies the evolution.

3. **Local approximation**: CGL assumes the second adiabatic invariant holds locally, which breaks down for trapped particles bouncing on long scales.

4. **Slow dynamics**: CGL assumes slow evolution compared to the gyro-period and bounce period.

Despite these limitations, CGL captures essential physics of anisotropic pressure in collisionless plasmas and is widely used in space physics.

## 5. Beyond MHD: Drift-Kinetic and Gyrokinetic Theory

### 5.1 Drift-Kinetic Theory

**Drift-kinetic theory** reduces the dimensionality from 6D to **5D** by **averaging over the gyrophase**.

The idea: in a magnetized plasma, particles rapidly gyrate around field lines. If we only care about slow dynamics ($\omega \ll \omega_c$), we can average over the fast gyration.

**Variables**:
- $\mathbf{R}$: guiding center position (3D)
- $v_\parallel$: parallel velocity (1D)
- $\mu$: magnetic moment (adiabatic invariant, parameter)
- Time $t$

**Distribution function**: $F(\mathbf{R}, v_\parallel, \mu, t)$ (5D instead of 6D)

**Drift-kinetic equation** (simplified):
$$\frac{\partial F}{\partial t} + \mathbf{v}_d \cdot \nabla_\mathbf{R} F + \frac{d v_\parallel}{dt} \frac{\partial F}{\partial v_\parallel} = C[F]$$

where $\mathbf{v}_d$ includes the parallel motion and perpendicular drifts:
$$\mathbf{v}_d = v_\parallel \hat{\mathbf{b}} + \mathbf{v}_E + \mathbf{v}_{\nabla B} + \mathbf{v}_\kappa + \cdots$$

**What it captures**:
- Parallel motion and bounce dynamics (trapped particles)
- All guiding-center drifts
- Collisionless (Landau) damping

**What it misses**:
- Cyclotron resonance (averaged out with gyrophase)
- Finite Larmor radius (FLR) effects

**Applications**:
- Neoclassical transport (tokamak collisional diffusion)
- Bounce-averaged kinetic theory (trapped-particle instabilities)
- Radiation belt dynamics (drift-loss-cone)

### 5.2 Gyrokinetic Theory

**Gyrokinetic theory** is the most sophisticated reduced model, capturing **finite Larmor radius (FLR)** effects while still averaging over gyrophase.

**Key innovation**: Expand in small parameters:
$$\delta = \frac{\rho_i}{L} \sim \frac{\omega}{\omega_{ci}} \sim \frac{\delta f}{f_0} \ll 1$$

This is the **gyrokinetic ordering**: slow, small-amplitude, long-wavelength fluctuations.

**Variables** (same as drift-kinetic):
- $\mathbf{R}$: gyrocenter position
- $v_\parallel$: parallel velocity
- $\mu$: magnetic moment

**Gyrokinetic distribution**: $g(\mathbf{R}, v_\parallel, \mu, t)$ (perturbed part)

**Gyrokinetic equation** (schematic):
$$\frac{\partial g}{\partial t} + \mathbf{v}_d \cdot \nabla g + \frac{dv_\parallel}{dt} \frac{\partial g}{\partial v_\parallel} = \text{(source terms with FLR)}$$

The key difference from drift-kinetic: **FLR effects** are retained through:
- Gyroaveraged electric field: $\langle \phi \rangle_\alpha$ (averaged over gyro-orbit)
- Gyroaveraged magnetic perturbation

**What it captures**:
- FLR effects (ion Landau damping, wave-particle resonances with FLR)
- Microinstabilities: ITG (ion temperature gradient), TEM (trapped electron mode), ETG (electron temperature gradient)
- Turbulence cascades with FLR

**What it misses**:
- Compressible Alfvén waves (fast magnetosonic)
- Low-frequency approximation: $\omega \ll \omega_{ci}$

**Applications**:
- **Tokamak turbulence**: gyrokinetic simulations (GENE, GS2, GYRO) predict turbulent transport, which limits confinement
- **Microinstability analysis**: determine growth rates of ITG, TEM, ETG modes
- **Zonal flows**: self-generated sheared flows that regulate turbulence

Gyrokinetic simulations are the state-of-the-art for tokamak physics and run on the world's largest supercomputers.

### 5.3 Comparison: Drift-Kinetic vs. Gyrokinetic

| Feature | Drift-Kinetic | Gyrokinetic |
|---------|---------------|-------------|
| Dimensions | 5D | 5D |
| FLR effects | No | Yes |
| Gyrophase-averaged | Yes | Yes |
| Ordering | None (exact gyroaverage) | $\delta \ll 1$ (perturbative) |
| What it solves | Bounce motion, drifts | Turbulence, microinstabilities |
| Typical application | Neoclassical, radiation belts | Tokamak turbulence, ITG/TEM |
| Computational cost | Moderate | Very high |

## 6. Extended MHD Models

### 6.1 Hall MHD

**Hall MHD** includes the Hall term in Ohm's law:

$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \frac{1}{en} \mathbf{J} \times \mathbf{B}$$

This allows ions and electrons to decouple at scales $\sim d_i$ (ion skin depth).

**Key features**:
- Whistler waves at high $k$
- Fast magnetic reconnection (Petschek rate)
- Dispersive Alfvén waves

**Applications**:
- Magnetic reconnection (magnetopause, magnetotail, solar corona)
- Dynamo theory (magnetic field generation)
- Space plasma turbulence

### 6.2 Two-Temperature MHD

Separate electron and ion temperatures:

$$\frac{d p_e}{dt} + \gamma p_e \nabla \cdot \mathbf{v} = Q_{ei} + Q_e$$
$$\frac{d p_i}{dt} + \gamma p_i \nabla \cdot \mathbf{v} = -Q_{ei} + Q_i$$

where $Q_{ei}$ is electron-ion energy exchange, and $Q_{e,i}$ are external heating.

**Applications**:
- Heating and energy partition (e.g., shocks heat ions more than electrons initially)
- Radiative cooling (electrons radiate more efficiently)

### 6.3 FLR-MHD

Include finite Larmor radius corrections to the pressure tensor:

$$\overleftrightarrow{P} = p \overleftrightarrow{I} + \overleftrightarrow{\Pi}^{\text{FLR}}$$

where $\overleftrightarrow{\Pi}^{\text{FLR}}$ includes gyroviscosity and other FLR effects.

**Applications**:
- Kinetic Alfvén waves
- FLR stabilization of MHD instabilities

### 6.4 Inertial MHD (Electron MHD)

At very small scales ($d_e$), electron inertia becomes important:

$$\mathbf{E} + \mathbf{v}_e \times \mathbf{B} = \frac{m_e}{e^2 n} \frac{d \mathbf{J}}{dt}$$

This is **electron MHD** (EMHD), where ions are stationary and only electrons move.

**Dispersion relation** (whistler in EMHD):
$$\omega = k^2 V_A d_e$$

**Applications**:
- Magnetic reconnection diffusion region
- Electron-scale turbulence

## 7. Python Code Examples

### 7.1 Validity Regime Diagram

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameter space: length scale vs. frequency
L = np.logspace(-4, 6, 200)  # 0.1 mm to 1000 km
omega = np.logspace(2, 10, 200)  # 100 rad/s to 10 GHz

L_grid, omega_grid = np.meshgrid(L, omega)

# Plasma parameters (typical tokamak)
n = 1e20  # m^-3
B = 2.0   # T
T = 5e3   # eV (5 keV)

e = 1.6e-19
m_i = 1.67e-27
m_e = 9.11e-31
k_B = 1.38e-23

# Characteristic scales and frequencies
omega_ci = e * B / m_i
omega_ce = e * B / m_e
omega_pi = np.sqrt(n * e**2 / (m_i * 8.85e-12))
omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))

v_th_i = np.sqrt(2 * k_B * T * e / m_i)
v_th_e = np.sqrt(2 * k_B * T * e / m_e)

rho_i = v_th_i / omega_ci
rho_e = v_th_e / omega_ce
d_i = 3e8 / omega_pi
d_e = 3e8 / omega_pe
lambda_D = np.sqrt(8.85e-12 * k_B * T * e / (n * e**2))

print("Characteristic scales and frequencies:")
print(f"  Ion gyrofrequency ω_ci = {omega_ci:.2e} rad/s ({omega_ci/(2*np.pi):.2e} Hz)")
print(f"  Ion gyroradius ρ_i = {rho_i*100:.2f} cm")
print(f"  Ion skin depth d_i = {d_i:.2f} m")
print(f"  Electron skin depth d_e = {d_e*100:.2f} cm")
print(f"  Debye length λ_D = {lambda_D*1e6:.2f} μm")
print()

# Define validity regions
# 1. MHD: ω << ω_ci, L >> ρ_i
# Factor 0.1 is a conservative safety margin: MHD ordering assumes ω/ω_ci → 0,
# so we require at least an order-of-magnitude separation to keep higher-order
# corrections (ε = ω/ω_ci) below ~10%. Similarly, L > 10ρ_i ensures FLR
# corrections (ε = ρ_i/L) are small enough for the fluid description to hold.
MHD = (omega_grid < 0.1 * omega_ci) & (L_grid > 10 * rho_i)

# 2. Hall MHD: ω << ω_ci, L ~ d_i
# Hall effects become O(1) when L ~ d_i; the upper bound L < 100 d_i marks
# where the Hall term (d_i/L) first becomes appreciable (>1%), defining the
# transition zone between MHD and Hall MHD regimes.
Hall_MHD = (omega_grid < 0.1 * omega_ci) & (L_grid > 10 * rho_i) & (L_grid < 100 * d_i)

# 3. Two-fluid: ω << ω_ce, L > d_e
# Two-fluid theory breaks down at the electron skin depth d_e (electron inertia
# becomes O(1)) and at ω ~ ω_ce (electron cyclotron resonance is not included).
# The factor 0.1 provides the same order-of-magnitude safety margin as for MHD.
Two_Fluid = (omega_grid < 0.1 * omega_ce) & (L_grid > 10 * d_e)

# 4. Gyrokinetic: ω ~ ω_ci, L ~ ρ_i
# Gyrokinetics is a perturbative theory valid near ω_ci and L ~ ρ_i; it
# becomes inaccurate outside a decade around these characteristic values.
Gyrokinetic = (omega_grid > 0.01 * omega_ci) & (omega_grid < omega_ci) & \
              (L_grid > rho_i) & (L_grid < 100 * rho_i)

# 5. Full kinetic: always valid (but expensive)
Full_Kinetic = np.ones_like(L_grid, dtype=bool)

# Plot
fig, ax = plt.subplots(figsize=(11, 8))

# Color regions
ax.contourf(L_grid, omega_grid, MHD.astype(int), levels=[0.5, 1.5],
            colors=['lightblue'], alpha=0.6)
ax.contourf(L_grid, omega_grid, Hall_MHD.astype(int), levels=[0.5, 1.5],
            colors=['lightcoral'], alpha=0.6)
ax.contourf(L_grid, omega_grid, Gyrokinetic.astype(int), levels=[0.5, 1.5],
            colors=['lightgreen'], alpha=0.6)

# Boundary lines
ax.axhline(omega_ci, color='r', linestyle='--', linewidth=2, label=f'$\omega_{{ci}}$ = {omega_ci:.2e} rad/s')
ax.axhline(omega_ce, color='m', linestyle='--', linewidth=1.5, label=f'$\omega_{{ce}}$ = {omega_ce:.2e} rad/s')

ax.axvline(rho_i, color='b', linestyle='--', linewidth=2, label=f'$\\rho_i$ = {rho_i*100:.1f} cm')
ax.axvline(d_i, color='g', linestyle='--', linewidth=2, label=f'$d_i$ = {d_i:.1f} m')
ax.axvline(d_e, color='orange', linestyle='--', linewidth=1.5, label=f'$d_e$ = {d_e*100:.1f} cm')

# Labels for regions
ax.text(1e0, 1e3, 'MHD', fontsize=16, weight='bold', color='blue')
ax.text(1e-1, 1e4, 'Hall MHD', fontsize=14, weight='bold', color='red')
ax.text(1e-2, 1e7, 'Gyrokinetic', fontsize=14, weight='bold', color='green')
ax.text(1e-3, 1e9, 'Full Kinetic', fontsize=14, weight='bold', color='black')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Length scale L (m)', fontsize=13)
ax.set_ylabel('Frequency ω (rad/s)', fontsize=13)
ax.set_title('Plasma Model Validity Regimes (n=$10^{20}$ m$^{-3}$, B=2 T, T=5 keV)', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-4, 1e6)
ax.set_ylim(1e2, 1e10)

plt.tight_layout()
plt.savefig('validity_regimes.png', dpi=150)
plt.show()
```

### 7.2 CGL vs. Isotropic MHD: Mirror Instability

```python
import numpy as np
import matplotlib.pyplot as plt

def mirror_instability_threshold(beta_perp):
    """
    Mirror instability threshold: p_perp/p_parallel > 1 + 1/beta_perp
    """
    return 1 + 1/beta_perp

# Beta range
beta_perp = np.logspace(-2, 2, 200)

# Threshold
threshold = mirror_instability_threshold(beta_perp)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Threshold curve
ax1.plot(beta_perp, threshold, 'r-', linewidth=3, label='Mirror instability threshold')
ax1.fill_between(beta_perp, 1, threshold, alpha=0.3, color='red', label='Unstable')
ax1.fill_between(beta_perp, threshold, 10, alpha=0.3, color='green', label='Stable')

ax1.set_xscale('log')
ax1.set_xlabel(r'$\beta_\perp = 2\mu_0 p_\perp / B^2$', fontsize=12)
ax1.set_ylabel(r'$p_\perp / p_\parallel$', fontsize=12)
ax1.set_title('Mirror Instability Threshold', fontsize=13)
ax1.set_ylim(1, 10)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Growth rate (simplified)
# γ/Ω_i ~ sqrt(β_perp) * (p_perp/p_parallel - 1 - 1/β_perp) for unstable
beta_example = 1.0
anisotropy = np.linspace(1, 5, 100)
threshold_value = mirror_instability_threshold(beta_example)

# np.where enforces causality: growth rate is exactly zero below threshold,
# because the mirror mode is linearly stable there (no energy source for the
# instability). The sqrt(β_perp) prefactor reflects that higher plasma pressure
# relative to magnetic pressure provides more free energy for the instability.
gamma_normalized = np.where(anisotropy > threshold_value,
                             np.sqrt(beta_example) * (anisotropy - threshold_value),
                             0)

ax2.plot(anisotropy, gamma_normalized, 'b-', linewidth=3)
ax2.axvline(threshold_value, color='r', linestyle='--', linewidth=2,
            label=f'Threshold at $\\beta_\\perp$ = {beta_example}')
ax2.fill_between(anisotropy, 0, gamma_normalized, alpha=0.3, color='blue')

ax2.set_xlabel(r'$p_\perp / p_\parallel$', fontsize=12)
ax2.set_ylabel(r'Growth rate $\gamma / \Omega_i$', fontsize=12)
ax2.set_title(f'Mirror Instability Growth Rate ($\\beta_\\perp$ = {beta_example})', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mirror_instability.png', dpi=150)
plt.show()

print(f"Mirror instability:")
print(f"  At β_perp = 0.1: threshold p_perp/p_parallel > {mirror_instability_threshold(0.1):.2f}")
print(f"  At β_perp = 1.0: threshold p_perp/p_parallel > {mirror_instability_threshold(1.0):.2f}")
print(f"  At β_perp = 10:  threshold p_perp/p_parallel > {mirror_instability_threshold(10):.2f}")
print()
print("Physical interpretation:")
print("  High β_perp (strong pressure): easier to go unstable (lower threshold)")
print("  Low β_perp (weak pressure): harder to go unstable (higher threshold)")
```

### 7.3 Dispersion Comparison: MHD vs. Hall MHD vs. Kinetic

```python
import numpy as np
import matplotlib.pyplot as plt

# Plasma parameters
n = 1e19
B = 0.1
T_e = 10  # eV
T_i = 10

e = 1.6e-19
m_i = 1.67e-27
m_e = 9.11e-31
mu_0 = 4e-7 * np.pi
k_B = 1.38e-23

# Derived quantities
omega_ci = e * B / m_i
omega_ce = e * B / m_e
omega_pi = np.sqrt(n * e**2 / (m_i * 8.85e-12))

v_A = B / np.sqrt(mu_0 * n * m_i)
c_s = np.sqrt(k_B * (T_e + T_i) * e / m_i)
d_i = 3e8 / omega_pi

v_th_e = np.sqrt(2 * k_B * T_e * e / m_e)
v_th_i = np.sqrt(2 * k_B * T_i * e / m_i)

print("Plasma parameters:")
print(f"  V_A = {v_A:.2e} m/s")
print(f"  c_s = {c_s:.2e} m/s")
print(f"  d_i = {d_i:.2e} m")
print(f"  ω_ci = {omega_ci:.2e} rad/s")
print()

# Wavenumber range
k = np.logspace(-3, 3, 500) / d_i  # normalized to d_i

# 1. MHD Alfvén wave
omega_MHD = k * v_A / omega_ci * (k * d_i)  # normalized to omega_ci

# 2. Hall MHD (whistler)
omega_Hall = k * v_A / omega_ci * (k * d_i) * np.sqrt(1 + (k * d_i)**2)

# 3. Kinetic Alfvén wave (warm plasma, with electron Landau damping)
# Approximate dispersion (electrostatic limit)
k_perp = k / 2  # assume oblique
rho_s = c_s / omega_ci
omega_KAW = k * v_A / omega_ci * (k * d_i) * np.sqrt(1 + (k_perp * d_i * rho_s / d_i)**2)

# 4. Ion acoustic wave
omega_ion_acoustic = k * c_s / omega_ci * (k * d_i)

# Plot
fig, ax = plt.subplots(figsize=(11, 7))

ax.loglog(k * d_i, omega_MHD, 'b-', linewidth=3, label='MHD Alfvén: $\omega = k_\parallel V_A$')
ax.loglog(k * d_i, omega_Hall, 'r--', linewidth=3, label='Hall MHD (whistler): $\omega = k_\parallel V_A \sqrt{1+(kd_i)^2}$')
ax.loglog(k * d_i, omega_KAW, 'g-.', linewidth=3, label='Kinetic Alfvén (warm)')
ax.loglog(k * d_i, omega_ion_acoustic, 'm:', linewidth=3, label='Ion acoustic: $\omega = k c_s$')

# Reference lines
ax.axvline(1, color='k', linestyle=':', alpha=0.5, linewidth=2, label='$k d_i = 1$')
ax.axhline(1, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='$\omega = \omega_{ci}$')

# Asymptotic slopes
k_ref = np.logspace(-2, 0, 50)
ax.loglog(k_ref * d_i, (k_ref * d_i)**1 * 0.01, 'k--', alpha=0.4, label='slope = 1')
ax.loglog(k_ref * d_i * 10, (k_ref * d_i * 10)**2 * 0.001, 'k-.', alpha=0.4, label='slope = 2')

ax.set_xlabel(r'$k d_i$ (normalized wavenumber)', fontsize=13)
ax.set_ylabel(r'$\omega / \omega_{ci}$ (normalized frequency)', fontsize=13)
ax.set_title('Dispersion Relations: MHD vs. Hall MHD vs. Kinetic', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(1e-4, 1e2)

plt.tight_layout()
plt.savefig('dispersion_comparison.png', dpi=150)
plt.show()

print("Dispersion relations:")
print("  MHD: ω ∝ k (linear, non-dispersive)")
print("  Hall MHD: ω ∝ k² at k d_i >> 1 (whistler, dispersive)")
print("  Kinetic: includes Landau damping (not shown, requires complex ω)")
```

## Summary

In this lesson, we traced the systematic reduction from kinetic theory to MHD:

1. **Two-fluid to single-fluid**: By combining electron and ion equations, we obtain the MHD momentum and continuity equations. The key step is deriving the ideal Ohm's law $\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0$ from the generalized Ohm's law by dropping resistive, Hall, pressure, and inertia terms.

2. **Validity conditions**: MHD is valid for low-frequency ($\omega \ll \omega_{ci}$), large-scale ($L \gg \rho_i$), collisional ($\lambda_{mfp} \ll L$), non-relativistic ($v \ll c$), quasi-neutral ($L \gg \lambda_D$), high-$R_m$ plasmas. Violations require extended MHD or kinetic models.

3. **CGL model**: For collisionless plasmas, the pressure is anisotropic ($p_\parallel \neq p_\perp$). The CGL (double adiabatic) closure uses conservation of adiabatic invariants: $p_\perp/(nB) = \text{const}$ and $p_\parallel B^2 / n^3 = \text{const}$. This predicts mirror and firehose instabilities.

4. **Drift-kinetic and gyrokinetic**: These 5D models average over gyrophase while retaining kinetic effects. Drift-kinetic captures bounce dynamics; gyrokinetic includes FLR effects and is used for tokamak turbulence simulations.

5. **Extended MHD**: Hall MHD, two-temperature MHD, FLR-MHD, and electron MHD extend standard MHD to capture additional physics at the cost of increased complexity.

6. **Model comparison**: Each model has a regime of validity. The choice depends on the scales, frequencies, and physics of interest. MHD is simple and captures large-scale dynamics; kinetic theory is comprehensive but computationally expensive.

Understanding the hierarchy of plasma models is essential for choosing the appropriate level of description for a given problem.

## Practice Problems

### Problem 1: Ideal MHD from Generalized Ohm's Law
Starting from the generalized Ohm's law:
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{en} \mathbf{J} \times \mathbf{B} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$$
For a tokamak plasma with $n = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $B = 5$ T, $L = 1$ m, $V = 100$ m/s:
(a) Calculate the characteristic time scale $\tau = L/V$.
(b) Estimate the magnitude of each term on the RHS relative to the LHS.
(c) Which terms can be neglected to obtain ideal MHD? Justify your answer.

### Problem 2: CGL Pressure Evolution
A collisionless plasma is compressed adiabatically by increasing the magnetic field from $B_0$ to $2B_0$ while keeping the density constant ($n = n_0$).
(a) Using the CGL equations, find the final values of $p_\perp$ and $p_\parallel$ in terms of the initial values.
(b) If initially $p_{\perp 0} = p_{\parallel 0} = p_0$, what is the anisotropy ratio $p_\perp / p_\parallel$ after compression?
(c) For $\beta_{\perp 0} = 0.5$, does the compressed plasma exceed the mirror instability threshold?

### Problem 3: Frozen-In Flux
In ideal MHD, the magnetic flux through any closed loop moving with the fluid is conserved:
$$\frac{d\Phi}{dt} = 0, \quad \text{where } \Phi = \int_S \mathbf{B} \cdot d\mathbf{A}$$
(a) Prove this **frozen-in theorem** using the ideal Ohm's law and the induction equation.
(b) A circular flux tube of initial radius $r_0 = 10$ cm has magnetic field $B_0 = 0.1$ T. The plasma is compressed radially to $r = 5$ cm. What is the final magnetic field (assuming incompressible flow)?
(c) What is the physical meaning of "frozen-in"? Can field lines reconnect in ideal MHD?

### Problem 4: Gyrokinetic Ordering
In gyrokinetic theory, the ordering is:
$$\frac{\rho_i}{L} \sim \frac{\omega}{\omega_{ci}} \sim \frac{\delta f}{f_0} \sim \delta \ll 1$$
(a) For a tokamak with $L = 1$ m, $\rho_i = 5$ mm, what is $\delta$?
(b) If $\omega_{ci} = 10^8$ rad/s, what is the maximum frequency resolved by gyrokinetics?
(c) The fast magnetosonic wave has $\omega \sim k V_A$ with no upper frequency limit. Why can't gyrokinetics capture this wave?

### Problem 5: Hall MHD Reconnection
In resistive MHD, the Sweet-Parker reconnection rate is:
$$V_{in} \sim \frac{\eta}{L} \sim \frac{V_A}{S^{1/2}}$$
where $S = L V_A / \eta$ is the Lundquist number.

In Hall MHD, the reconnection rate becomes (Petschek):
$$V_{in} \sim 0.1 V_A$$
independent of resistivity!

(a) For a solar flare with $B = 0.01$ T, $n = 10^{16}$ m$^{-3}$, $L = 10^4$ km, $T_e = 10^6$ K, calculate the Alfvén speed and the ion skin depth.
(b) Estimate the Sweet-Parker reconnection time $\tau_{SP} \sim L / V_{in}$ (use Spitzer resistivity).
(c) Estimate the Hall MHD reconnection time $\tau_{Hall}$.
(d) Solar flares release energy on time scales of minutes. Which model is consistent with observations?

---

**Previous**: [Two-Fluid Model](./13_Two_Fluid_Model.md) | **Next**: [Plasma Diagnostics](./15_Plasma_Diagnostics.md)

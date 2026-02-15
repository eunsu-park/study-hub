# 13. Fusion MHD

## Learning Objectives

- Understand magnetic confinement concepts: tokamak, stellarator, reversed field pinch (RFP)
- Analyze tokamak equilibrium: aspect ratio, elongation, triangularity, Shafranov shift
- Derive beta limits and understand the Troyon limit
- Identify major MHD instabilities in tokamaks: sawteeth, ELMs, disruptions, NTMs, RWMs
- Apply disruption mitigation strategies and understand their physical basis
- Compare stellarator advantages for steady-state fusion
- Implement Python models for beta limits, sawtooth periods, and disruption forces

## 1. Introduction to Magnetic Confinement Fusion

Magnetic confinement fusion aims to achieve controlled thermonuclear fusion by confining a hot plasma (T ~ 10-20 keV) using magnetic fields. The primary challenge is achieving sufficient confinement time and pressure to satisfy the Lawson criterion while maintaining plasma stability against MHD instabilities.

### 1.1 Fusion Reactions and Requirements

The deuterium-tritium (D-T) fusion reaction is the most accessible:

```
D + T → He-4 (3.5 MeV) + n (14.1 MeV)
```

The fusion triple product requirement:

```
n T τ_E ≥ 3 × 10²¹ m⁻³ keV s
```

where:
- $n$ is the plasma density
- $T$ is the temperature
- $\tau_E$ is the energy confinement time

### 1.2 Magnetic Confinement Principles

Charged particles gyrate around magnetic field lines with Larmor radius:

```
r_L = (m v_⊥)/(q B)
```

For thermal particles at T = 10 keV in B = 5 T:
- Electrons: $r_{Le} \sim 0.1$ mm
- Ions: $r_{Li} \sim 4$ mm

The small Larmor radius compared to device size enables confinement. However, toroidal geometry introduces drift motions and requires careful magnetic field configuration.

## 2. Tokamak Configuration

The tokamak is the leading magnetic confinement concept, employing a combination of toroidal and poloidal magnetic fields to confine the plasma in a torus.

### 2.1 Tokamak Magnetic Field Structure

The total magnetic field in a tokamak consists of:

1. **Toroidal field** $B_φ$: Strong field produced by external toroidal field coils
2. **Poloidal field** $B_θ$: Weaker field produced by plasma current $I_p$ and external poloidal field coils
3. **Vertical field** $B_z$: Controls plasma position and shape

The total field:

```
B = B_φ e_φ + B_θ e_θ
```

The field lines wind around the torus on nested flux surfaces (magnetic surfaces).

### 2.2 Safety Factor

The safety factor $q$ measures the pitch of field lines:

```
q = (r B_φ)/(R B_θ)
```

where $R$ is the major radius and $r$ is the minor radius.

For a large aspect ratio tokamak with circular cross-section:

```
q(r) = (r B_0)/(R B_θ(r)) ≈ (2π r² B_0)/(μ₀ R I_p(r))
```

where $I_p(r)$ is the plasma current inside radius $r$.

The safety factor profile $q(r)$ is crucial for MHD stability:
- $q < 1$ on axis allows sawtooth oscillations
- $q_{edge} < 2$ leads to disruptions
- Rational surfaces where $q = m/n$ are susceptible to tearing modes

### 2.3 Plasma Current Profile

The plasma current density follows from Ampère's law:

```
∇ × B = μ₀ j
```

In a tokamak, the toroidal current density:

```
j_φ = (1/μ₀ r) ∂(r B_θ)/∂r
```

Common current profiles:
- **Peaked**: $j(r) = j_0 (1 - r²/a²)^ν$, $ν > 0$
- **Flat**: $ν \approx 1$
- **Hollow**: current density maximum off-axis

The current profile determines the $q$-profile and affects stability.

### 2.4 Aspect Ratio and Plasma Shape

Key geometric parameters:

- **Aspect ratio**: $A = R/a$ (typically 2.5-4)
- **Elongation**: $κ = b/a$ (vertical/horizontal minor radius, typically 1.5-2)
- **Triangularity**: $δ$ (characterizes D-shaped cross-section)

High elongation increases plasma volume and improves confinement but increases susceptibility to vertical displacement events (VDEs).

## 3. Tokamak Equilibrium

### 3.1 Grad-Shafranov Equation

The tokamak equilibrium is governed by the Grad-Shafranov (GS) equation, derived from the force balance $j × B = ∇p$ in axisymmetric geometry.

Introducing the poloidal flux function $\psi(R, Z)$:

```
B_R = -(1/R) ∂ψ/∂Z
B_Z = (1/R) ∂ψ/∂R
```

The GS equation is:

```
Δ* ψ = -μ₀ R² dp/dψ - F dF/dψ
```

where the elliptic operator:

```
Δ* ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z²
```

and $F(ψ) = R B_φ$ is the toroidal field function.

### 3.2 Shafranov Shift

In a finite-pressure plasma, the magnetic axis shifts outward from the geometric center due to toroidal effects. This **Shafranov shift** $\Delta$ is approximately:

```
Δ/a ≈ β_p + l_i/2
```

where:
- $\beta_p = 2 μ₀ \langle p \rangle / \langle B_θ² \rangle$ is the poloidal beta
- $l_i$ is the internal inductance (depends on current profile)

For typical tokamak parameters ($\beta_p \sim 0.5$, $l_i \sim 1$), $\Delta/a \sim 0.5-1$.

The Shafranov shift increases with pressure and affects equilibrium limits.

### 3.3 Beta Limits

The plasma beta is the ratio of plasma pressure to magnetic pressure:

```
β = 2 μ₀ p / B²
```

Several definitions:
- **Total beta**: $\beta = 2 μ₀ \langle p \rangle / B_0²$
- **Poloidal beta**: $\beta_p = 2 μ₀ \langle p \rangle / \langle B_θ² \rangle$
- **Toroidal beta**: $\beta_t = 2 μ₀ \langle p \rangle / \langle B_φ² \rangle$

The **Troyon limit** is an empirical scaling for the maximum achievable beta:

```
β_N = β (%·T·m/MA) = β a B_0 / I_p ≤ β_N^max
```

where:
- $a$ is the minor radius (m)
- $B_0$ is the toroidal field (T)
- $I_p$ is the plasma current (MA)
- $\beta_N^{max} \approx 2.8-4$ for standard scenarios

High beta is desirable for fusion power density, but MHD instabilities (pressure-driven modes, external kinks) impose limits.

### 3.4 Equilibrium Beta Limit

For a large aspect ratio tokamak, balancing pressure gradient and magnetic tension:

```
β_t ≤ a/(q R) = 1/(A q)
```

This gives a crude estimate. More refined calculations using the GS equation yield the Troyon limit.

## 4. Major MHD Instabilities in Tokamaks

### 4.1 Sawtooth Oscillations

Sawteeth are periodic relaxations of the core temperature and density in tokamaks with $q_0 < 1$.

**Mechanism:**
1. Ohmic heating creates a peaked temperature profile
2. Internal kink mode ($m=1, n=1$) becomes unstable when $q_0 < 1$
3. Magnetic reconnection flattens the core temperature profile (sawtooth crash)
4. Cycle repeats as ohmic heating re-establishes peaked profile

**Kadomtsev reconnection model:**

The internal kink mode reconnects field lines at the $q=1$ surface, flattening the temperature profile inside the mixing radius $r_{mix}$.

The sawtooth period scales as:

```
τ_sawtooth ∝ a² / (η S^α)
```

where $S$ is the Lundquist number and $α \approx 0.6$ (from simulations).

**Impact:**
- Periodic heat pulses to edge
- Can trigger neoclassical tearing modes (NTMs)
- Beneficial: prevents excessive peaking, ejecting impurities

**Control methods:**
- Electron cyclotron current drive (ECCD) near $q=1$ surface
- Pellet injection to trigger controlled crashes

### 4.2 Edge Localized Modes (ELMs)

ELMs are periodic instabilities at the plasma edge in high-confinement mode (H-mode). H-mode features a steep pressure gradient (pedestal) near the edge, which can become unstable.

**Peeling-ballooning instability:**

Two driving mechanisms:
1. **Peeling**: edge current density drives external kink modes
2. **Ballooning**: steep pressure gradient drives interchange-like modes

The stability boundary in $(j_{edge}, \nabla p_{edge})$ space forms a "peeling-ballooning" diagram.

**Types of ELMs:**

- **Type I (giant ELMs)**: Large, periodic crashes expelling 5-15% of pedestal energy
  - Can cause significant heat flux to divertor ($> 10$ MW/m²)
  - Frequency: 10-100 Hz

- **Type III (small ELMs)**: Smaller, more frequent
  - Lower pedestal pressure
  - Less divertor concern

- **QH-mode (ELM-free)**: Quiescent H-mode with edge harmonic oscillation (EHO)
  - Continuous edge particle/energy exhaust without large ELMs
  - Requires rotational shear, observed in DIII-D

**Divertor heat flux from ELMs:**

```
q_peak ≈ W_ELM / (A_wet τ_ELM)
```

where:
- $W_{ELM}$ is the energy expelled per ELM
- $A_{wet}$ is the wetted area on divertor
- $\tau_{ELM}$ is the ELM energy deposition time (0.1-1 ms)

For ITER, unmitigated Type I ELMs could deliver $q_{peak} > 20$ MW/m², exceeding material limits.

**ELM mitigation strategies:**

1. **Resonant Magnetic Perturbations (RMPs)**: External 3D fields create stochastic edge layer
   - Demonstrated on DIII-D, ASDEX-U, KSTAR
   - Reduces or eliminates ELMs at cost of some confinement

2. **Pellet pacing**: Injecting small pellets triggers ELMs at higher frequency, reducing size

3. **QH-mode or I-mode**: Achieving ELM-free regimes

### 4.3 Disruptions

A disruption is a catastrophic loss of plasma confinement occurring on a timescale of milliseconds. Disruptions pose major challenges for large tokamaks like ITER.

**Causes:**

1. **Density limit**: Approaching the Greenwald density
   ```
   n_G = I_p / (π a²) (10²⁰ m⁻³ MA⁻¹ m⁻²)
   ```
   Exceeding $n_G$ leads to radiative collapse and thermal instability.

2. **Current limit**: Edge safety factor $q_{edge} < 2$ leads to external kink modes

3. **Locked modes**: Tearing modes that lock to the wall due to error fields or low rotation

4. **Beta limit**: Exceeding MHD beta limit triggers ideal modes

**Disruption phases:**

1. **Thermal quench (TQ)**: Loss of thermal energy (0.1-1 ms)
   - Temperature collapse: $T \rightarrow 0$
   - Heat flux to wall: can exceed material limits
   - Causes: MHD mode growth, stochastization

2. **Current quench (CQ)**: Loss of plasma current (1-100 ms)
   - Plasma current decays: $I_p \rightarrow 0$
   - Induced voltages and forces on conducting structures
   - Runaway electron (RE) generation risk

3. **Runaway electron beam**: Highly relativistic electrons
   - Accelerated by inductive electric field during CQ
   - Can carry significant current (MA-level)
   - Highly localized heat deposition if beam impacts wall

**Forces on tokamak structures:**

During the CQ, changing plasma current induces eddy currents in the vacuum vessel and structures, leading to large electromagnetic forces.

**Vertical force:**

```
F_z ~ (dI_p/dt) * (mutual inductance)
```

For ITER disruption: $F_z$ can reach several MN.

**Halo currents:**

Currents flowing through plasma scrape-off layer into first wall, then through structures back to plasma. These create toroidal asymmetric forces.

**Disruption mitigation:**

1. **Massive Gas Injection (MGI)**: Inject large quantities of noble gas (Ne, Ar)
   - Radiates thermal energy more uniformly
   - Increases electron density to suppress runaway generation
   - Slows current quench to reduce forces

2. **Shattered Pellet Injection (SPI)**: Inject frozen pellet that shatters into fragments
   - Deeper penetration and faster assimilation than MGI
   - More effective radiation distribution
   - Baseline mitigation system for ITER

3. **Disruption prediction and avoidance**: Machine learning models predict disruptions tens to hundreds of ms in advance
   - Real-time control to avoid disruption region
   - Trigger mitigation if avoidance fails

### 4.4 Neoclassical Tearing Modes (NTMs)

NTMs are resistive tearing modes driven by the perturbation to the bootstrap current inside magnetic islands.

**Bootstrap current:**

In a toroidal plasma with pressure gradient, trapped particles contribute to a net toroidal current:

```
j_bs = C(ν*, ε) d p/dr
```

where $\nu^*$ is the collisionality and $\varepsilon = r/R$ is the inverse aspect ratio.

**Island dynamics:**

When a tearing mode creates a magnetic island at a rational surface $q = m/n$, the pressure flattens inside the island, reducing the local bootstrap current. This missing current drives island growth.

The modified Rutherford equation for NTM island width $w$:

```
τ_R dw/dt = r_s Δ'_{classical} + r_s Δ'_{bs}(w)
```

where:

```
Δ'_{bs} = L_{q,p} / w²
```

is the bootstrap drive term (positive, destabilizing) and $L_{q,p}$ depends on pressure and safety factor profiles.

**Threshold for NTM excitation:**

NTMs require a seed island (typically from sawteeth or ELMs) to exceed a critical width:

```
w_crit ~ sqrt(L_{q,p} / |Δ'_{classical}|)
```

**Control:**

Electron Cyclotron Current Drive (ECCD) localized at the rational surface can replace the missing bootstrap current, suppressing or preventing NTM growth.

### 4.5 Resistive Wall Modes (RWMs)

RWMs are external kink modes partially stabilized by a resistive conducting wall.

**Ideal kink with conducting wall:**

An ideal external kink mode can be stabilized by a perfectly conducting wall close to the plasma. With a resistive wall, stabilization is temporary: the mode grows on the wall resistive timescale $\tau_{wall}$.

**Growth rate:**

```
γ ≈ τ_wall^{-1}
```

where $\tau_{wall} \sim μ₀ \sigma d_{wall} b_{wall}$ ($\sigma$ is wall conductivity, $d_{wall}$ thickness, $b_{wall}$ radius).

Typical timescales: $\tau_{wall} \sim 10-100$ ms (much slower than ideal MHD).

**Stabilization:**

- **Plasma rotation**: If plasma rotates faster than the RWM growth rate, mode is stabilized
  ```
  ω_rot > γ_RWM
  ```

- **Active feedback control**: External coils detect mode and apply correcting field

- **Kinetic effects**: Resonance with precession drifts of energetic particles can provide damping

RWMs limit the achievable beta in advanced tokamak scenarios without rotation or feedback.

## 5. Stellarator Configuration

The stellarator is an alternative to the tokamak that achieves confinement using external 3D magnetic fields without relying on plasma current.

### 5.1 Stellarator Magnetic Field

In a stellarator, twisted magnetic field lines are produced entirely by external coils, which create a rotational transform (equivalent to 1/q in a tokamak).

**Advantages:**
- **Steady-state**: No need for current drive
- **No disruptions**: No large plasma current, no current-driven instabilities
- **Flexible optimization**: Field shape can be optimized for stability and confinement

**Challenges:**
- **Complex 3D geometry**: Difficult to design, construct, and analyze
- **Neoclassical transport**: Drift orbits in 3D field can lead to enhanced transport
- **Limited experimental database**: Fewer large devices than tokamaks

### 5.2 Quasi-Symmetry

Modern stellarators aim for quasi-symmetry: the field strength $|B|$ is approximately symmetric in a particular direction in magnetic coordinates (e.g., quasi-helical, quasi-toroidal, quasi-axisymmetric).

Quasi-symmetry reduces neoclassical transport by ensuring that particle drift surfaces coincide with flux surfaces.

**Examples:**
- W7-X (Germany): Quasi-isodynamic, modular coils
- HSX (USA): Quasi-helically symmetric
- NCSX (USA, canceled): Quasi-axisymmetric

### 5.3 MHD Stability in Stellarators

Stellarators can be designed to avoid low-order rational surfaces, reducing susceptibility to tearing modes. However, they face other MHD stability challenges:

- **Interchange modes**: Unfavorable curvature regions can drive interchange instabilities
- **Ballooning modes**: Pressure-driven instabilities similar to tokamaks
- **External kinks**: If equilibrium is not optimal

Numerical optimization codes (e.g., VMEC for equilibrium, TERPSICHORE for stability) are essential for stellarator design.

### 5.4 W7-X Results

Wendelstein 7-X (W7-X) achieved first plasma in 2015 and has demonstrated:
- Long pulses (up to 101 s)
- Good energy confinement comparable to tokamaks
- Low neoclassical transport in line with predictions
- Control of islands and error fields

Stellarators remain a strong candidate for fusion reactors, especially for steady-state operation.

## 6. Reversed Field Pinch (RFP)

The RFP is a toroidal confinement concept where the toroidal magnetic field reverses direction in the outer region of the plasma.

### 6.1 RFP Magnetic Field Structure

The RFP has comparable toroidal and poloidal fields:

```
B_φ(r) changes sign at r = r_reversal
B_θ(r) ~ constant
```

The field configuration is sustained by plasma current and MHD dynamo action.

### 6.2 Taylor Relaxation

Taylor's hypothesis: A turbulent plasma relaxes to a minimum energy state subject to the constraint of constant global magnetic helicity.

The relaxed state satisfies:

```
∇ × B = μ B
```

where $\mu$ is a constant (eigenvalue of the force-free equation).

In a cylinder, this yields a Bessel function profile:

```
B_z(r) = B_0 J_0(μ r)
B_θ(r) = B_0 J_1(μ r)
```

If $\mu a$ is chosen such that $J_0(\mu a) < 0$, the field reverses at the edge.

### 6.3 RFP MHD Activity

RFPs exhibit strong MHD fluctuations (tearing modes) that relax the current profile and sustain the reversed field. This "MHD dynamo" is essential to RFP operation but degrades confinement.

**Recent improvements:**
- **Pulsed Poloidal Current Drive (PPCD)**: Reduces MHD fluctuations, improves confinement
- **Quasi-single-helicity (QSH) states**: One dominant mode, reduced chaos

RFPs achieve $\beta \sim 10-20\%$, higher than tokamaks, but confinement time is shorter.

## 7. Beta Limits and Stability Boundaries

### 7.1 Troyon Beta Limit Derivation (Heuristic)

Consider a large aspect ratio tokamak. The external kink mode is driven by plasma pressure and current. Balancing the destabilizing pressure term and stabilizing field line bending:

```
β ~ 1 / (q a/R)
```

Expressing in terms of plasma current using $q \sim a² B_φ / (μ₀ R I_p)$:

```
β ~ μ₀ I_p / (a B_φ)
```

Rearranging:

```
β a B_φ / I_p ~ constant
```

This is the normalized beta $\beta_N$. More detailed calculations give:

```
β_N^{max} ≈ C_T l_i / (A q_cyl)
```

where $C_T \approx 2.8$ (Troyon coefficient), $l_i$ is internal inductance, $A$ is aspect ratio, and $q_{cyl}$ is cylindrical safety factor.

### 7.2 Ballooning Mode Limit

Ballooning modes are high toroidal mode number ($n \rightarrow \infty$) pressure-driven instabilities localized to regions of unfavorable curvature.

The Mercier criterion gives a local stability condition:

```
D_I > 0
```

where $D_I$ involves pressure gradient, shear, and magnetic well depth.

For a tokamak, ballooning stability roughly requires:

```
dp/dr < (critical gradient)
```

The ballooning limit on beta:

```
β_crit ~ (ε/q²) (shear factor)
```

High shear ($s = r q'/q$) and large aspect ratio improve ballooning stability.

### 7.3 Global Beta Limits in Advanced Scenarios

Advanced tokamak scenarios aim for high beta, high bootstrap fraction, and steady-state operation. These scenarios operate near or above the no-wall ideal kink limit but below the with-wall limit.

**Operating space:**
- $\beta_N \sim 3-4$ (above no-wall limit ~2.5)
- Requires resistive wall mode control (rotation, feedback)
- Elevated $q$-profile (e.g., $q_{min} > 2$) to avoid sawteeth and reduce NTM drive

## 8. Python Implementations

### 8.1 Troyon Beta Limit

```python
import numpy as np
import matplotlib.pyplot as plt

def troyon_beta_limit(I_p, a, B_0, C_Troyon=2.8):
    """
    Calculate Troyon beta limit.

    Parameters:
    I_p : plasma current (MA)
    a : minor radius (m)
    B_0 : toroidal magnetic field on axis (T)
    C_Troyon : Troyon coefficient (dimensionless, typically 2.8)

    Returns:
    beta_N : normalized beta limit (%)
    beta_percent : absolute beta limit (%)
    """
    beta_N = C_Troyon  # Troyon limit (% T m / MA)
    beta_percent = beta_N * I_p / (a * B_0)
    return beta_N, beta_percent

# Example: ITER-like parameters
I_p_ITER = 15.0  # MA
a_ITER = 2.0     # m
B_0_ITER = 5.3   # T

beta_N_limit, beta_limit = troyon_beta_limit(I_p_ITER, a_ITER, B_0_ITER)
print(f"ITER parameters: I_p = {I_p_ITER} MA, a = {a_ITER} m, B_0 = {B_0_ITER} T")
print(f"Troyon limit: β_N = {beta_N_limit:.2f} % T m / MA")
print(f"Absolute beta limit: β = {beta_limit:.2f} %")

# Scan over plasma current
I_p_scan = np.linspace(5, 20, 50)
beta_scan = [troyon_beta_limit(I_p, a_ITER, B_0_ITER)[1] for I_p in I_p_scan]

plt.figure(figsize=(8, 5))
plt.plot(I_p_scan, beta_scan, 'b-', linewidth=2)
plt.xlabel('Plasma Current (MA)', fontsize=12)
plt.ylabel('Beta Limit (%)', fontsize=12)
plt.title('Troyon Beta Limit vs Plasma Current', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('troyon_beta_limit.png', dpi=150)
plt.show()
```

### 8.2 Sawtooth Period Model

```python
def sawtooth_period(a, T_e, n_e, B, S_exp=0.6):
    """
    Estimate sawtooth period using scaling law.

    Parameters:
    a : minor radius (m)
    T_e : electron temperature (eV)
    n_e : electron density (m^-3)
    B : magnetic field (T)
    S_exp : Lundquist number exponent (typically 0.6)

    Returns:
    tau_sawtooth : sawtooth period (s)
    """
    # Physical constants
    e = 1.602e-19  # C
    m_e = 9.109e-31  # kg
    epsilon_0 = 8.854e-12  # F/m
    mu_0 = 4 * np.pi * 1e-7  # H/m

    # Spitzer resistivity
    ln_Lambda = 15.0  # Coulomb logarithm (approximate)
    eta = (e**2 * ln_Lambda * m_e**0.5) / (12 * np.pi**1.5 * epsilon_0**2 * (e * T_e)**1.5)

    # Lundquist number
    tau_R = mu_0 * a**2 / eta
    tau_A = a / (B / np.sqrt(mu_0 * n_e * m_e * 1836))  # Alfven time (approximation)
    S = tau_R / tau_A

    # Sawtooth period scaling
    tau_sawtooth = tau_R / S**S_exp * 50  # Empirical factor

    return tau_sawtooth, S, eta

# Example: JET-like parameters
a_JET = 1.0  # m
T_e_JET = 2000  # eV (core temperature)
n_e_JET = 5e19  # m^-3
B_JET = 3.0  # T

tau_saw, S_JET, eta_JET = sawtooth_period(a_JET, T_e_JET, n_e_JET, B_JET)
print(f"\nJET parameters: a = {a_JET} m, T_e = {T_e_JET} eV, n_e = {n_e_JET:.1e} m^-3, B = {B_JET} T")
print(f"Spitzer resistivity: η = {eta_JET:.3e} Ω m")
print(f"Lundquist number: S = {S_JET:.2e}")
print(f"Estimated sawtooth period: τ = {tau_saw:.3f} s")

# Scan over temperature
T_e_scan = np.linspace(500, 5000, 50)
tau_scan = [sawtooth_period(a_JET, T_e, n_e_JET, B_JET)[0] for T_e in T_e_scan]

plt.figure(figsize=(8, 5))
plt.plot(T_e_scan, tau_scan, 'r-', linewidth=2)
plt.xlabel('Electron Temperature (eV)', fontsize=12)
plt.ylabel('Sawtooth Period (s)', fontsize=12)
plt.title('Sawtooth Period vs Electron Temperature', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sawtooth_period.png', dpi=150)
plt.show()
```

### 8.3 Disruption Force Estimation

```python
def disruption_forces(I_p, dI_dt, R, a, b_wall):
    """
    Estimate electromagnetic forces during disruption.

    Parameters:
    I_p : initial plasma current (MA)
    dI_dt : current quench rate (MA/s)
    R : major radius (m)
    a : minor radius (m)
    b_wall : wall minor radius (m)

    Returns:
    F_z : vertical force (MN)
    V_loop : loop voltage (V)
    """
    mu_0 = 4 * np.pi * 1e-7

    # Mutual inductance (simple model)
    M = mu_0 * R * (np.log(8 * R / a) - 2 + 0.5)  # H

    # Vertical force (simplified)
    F_z = abs(I_p * 1e6 * dI_dt * 1e6 * M / (2 * np.pi * R)) / 1e6  # MN

    # Loop voltage
    V_loop = abs(M * dI_dt * 1e6)  # V

    return F_z, V_loop

# Example: ITER disruption
I_p_ITER_disr = 15.0  # MA
dI_dt_ITER = -15.0 / 0.15  # MA/s (15 MA in 150 ms)
R_ITER = 6.2  # m
a_ITER_disr = 2.0  # m
b_wall_ITER = 2.3  # m

F_z_ITER, V_loop_ITER = disruption_forces(I_p_ITER_disr, dI_dt_ITER, R_ITER, a_ITER_disr, b_wall_ITER)
print(f"\nITER disruption: I_p = {I_p_ITER_disr} MA, dI/dt = {dI_dt_ITER:.1f} MA/s")
print(f"Estimated vertical force: F_z ~ {F_z_ITER:.2f} MN")
print(f"Estimated loop voltage: V_loop ~ {V_loop_ITER:.1f} V")

# Current quench timescale scan
tau_CQ_scan = np.linspace(0.01, 0.5, 50)  # s
dI_dt_scan = -I_p_ITER_disr / tau_CQ_scan
F_z_scan = [disruption_forces(I_p_ITER_disr, dI_dt, R_ITER, a_ITER_disr, b_wall_ITER)[0] for dI_dt in dI_dt_scan]

plt.figure(figsize=(8, 5))
plt.plot(tau_CQ_scan * 1000, F_z_scan, 'g-', linewidth=2)
plt.xlabel('Current Quench Time (ms)', fontsize=12)
plt.ylabel('Vertical Force (MN)', fontsize=12)
plt.title('Disruption Vertical Force vs Current Quench Time', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('disruption_forces.png', dpi=150)
plt.show()
```

### 8.4 Safety Factor Profile

```python
def safety_factor_profile(r, a, R, B_0, I_p, profile='parabolic', nu=1.0):
    """
    Calculate safety factor profile.

    Parameters:
    r : radial coordinate (m) or array
    a : minor radius (m)
    R : major radius (m)
    B_0 : toroidal field on axis (T)
    I_p : plasma current (MA)
    profile : 'parabolic' or 'flat'
    nu : profile parameter (for parabolic)

    Returns:
    q : safety factor
    """
    r = np.atleast_1d(r)
    mu_0 = 4 * np.pi * 1e-7

    if profile == 'parabolic':
        # j(r) = j_0 (1 - (r/a)^2)^nu
        # I(r) = 2π ∫ j(r') r' dr'
        # For simplicity, approximate q(r)
        q_edge = (a**2 * B_0) / (mu_0 * R * I_p * 1e6) * 2 * np.pi
        q_0 = q_edge / (nu + 1)
        q = q_0 + (q_edge - q_0) * (r / a)**2
    elif profile == 'flat':
        # Flat current profile
        q = (r**2 * B_0) / (mu_0 * R * I_p * 1e6 / (np.pi * a**2)) / (2 * np.pi)
        q[r == 0] = 0  # Avoid singularity
    else:
        raise ValueError("Profile must be 'parabolic' or 'flat'")

    return q

# Plot q-profile for different current profiles
r_array = np.linspace(0, a_ITER, 100)

q_parabolic_1 = safety_factor_profile(r_array, a_ITER, 6.2, B_0_ITER, I_p_ITER, 'parabolic', nu=1.0)
q_parabolic_2 = safety_factor_profile(r_array, a_ITER, 6.2, B_0_ITER, I_p_ITER, 'parabolic', nu=2.0)

plt.figure(figsize=(10, 6))
plt.plot(r_array, q_parabolic_1, 'b-', linewidth=2, label='Parabolic (ν=1)')
plt.plot(r_array, q_parabolic_2, 'r-', linewidth=2, label='Parabolic (ν=2)')
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='q=1 (sawtooth)')
plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='q=2 (disruption)')
plt.xlabel('Minor Radius r (m)', fontsize=12)
plt.ylabel('Safety Factor q', fontsize=12)
plt.title('Safety Factor Profile', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('safety_factor_profile.png', dpi=150)
plt.show()
```

### 8.5 Greenwald Density Limit

```python
def greenwald_density(I_p, a):
    """
    Calculate Greenwald density limit.

    Parameters:
    I_p : plasma current (MA)
    a : minor radius (m)

    Returns:
    n_G : Greenwald density (10^20 m^-3)
    """
    n_G = I_p / (np.pi * a**2)  # 10^20 m^-3
    return n_G

# ITER Greenwald density
n_G_ITER = greenwald_density(I_p_ITER, a_ITER)
print(f"\nITER Greenwald density limit: n_G = {n_G_ITER:.2f} × 10^20 m^-3")

# Scan over current
I_p_scan_greenwald = np.linspace(5, 20, 50)
n_G_scan = [greenwald_density(I_p, a_ITER) for I_p in I_p_scan_greenwald]

plt.figure(figsize=(8, 5))
plt.plot(I_p_scan_greenwald, n_G_scan, 'm-', linewidth=2)
plt.xlabel('Plasma Current (MA)', fontsize=12)
plt.ylabel('Greenwald Density Limit (10²⁰ m⁻³)', fontsize=12)
plt.title('Greenwald Density Limit vs Plasma Current', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('greenwald_density.png', dpi=150)
plt.show()
```

### 8.6 ELM Energy Loss and Divertor Heat Flux

```python
def elm_heat_flux(W_ELM, A_wet, tau_ELM):
    """
    Estimate peak divertor heat flux from ELM.

    Parameters:
    W_ELM : energy expelled per ELM (MJ)
    A_wet : wetted area on divertor (m^2)
    tau_ELM : energy deposition timescale (ms)

    Returns:
    q_peak : peak heat flux (MW/m^2)
    """
    q_peak = W_ELM / (A_wet * tau_ELM * 1e-3)  # MW/m^2
    return q_peak

# ITER Type I ELM (unmitigated)
W_ELM_ITER = 1.0  # MJ (10% of pedestal energy ~ 10 MJ)
A_wet_ITER = 0.5  # m^2 (narrow wetted area)
tau_ELM_ITER = 0.5  # ms

q_peak_ITER = elm_heat_flux(W_ELM_ITER, A_wet_ITER, tau_ELM_ITER)
print(f"\nITER Type I ELM (unmitigated):")
print(f"W_ELM = {W_ELM_ITER} MJ, A_wet = {A_wet_ITER} m^2, τ_ELM = {tau_ELM_ITER} ms")
print(f"Peak heat flux: q_peak ~ {q_peak_ITER:.1f} MW/m^2")

# Mitigation: smaller, more frequent ELMs
W_ELM_mitigated = 0.1  # MJ
n_ELMs = 10  # 10x more frequent

q_peak_mitigated = elm_heat_flux(W_ELM_mitigated, A_wet_ITER, tau_ELM_ITER)
print(f"\nMitigated ELMs:")
print(f"W_ELM = {W_ELM_mitigated} MJ (10x smaller), frequency 10x higher")
print(f"Peak heat flux: q_peak ~ {q_peak_mitigated:.1f} MW/m^2")

# Scan over ELM size
W_ELM_scan = np.linspace(0.05, 2.0, 50)
q_peak_scan = [elm_heat_flux(W, A_wet_ITER, tau_ELM_ITER) for W in W_ELM_scan]

plt.figure(figsize=(8, 5))
plt.plot(W_ELM_scan, q_peak_scan, 'orange', linewidth=2)
plt.axhline(y=10, color='r', linestyle='--', linewidth=2, label='Material limit (~10 MW/m²)')
plt.xlabel('ELM Energy (MJ)', fontsize=12)
plt.ylabel('Peak Heat Flux (MW/m²)', fontsize=12)
plt.title('ELM Divertor Heat Flux vs ELM Energy', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elm_heat_flux.png', dpi=150)
plt.show()
```

### 8.7 Neoclassical Tearing Mode Island Width Evolution

```python
def ntm_island_evolution(w0, Delta_prime_bs, Delta_prime_class, r_s, tau_R, t_max, dt):
    """
    Evolve NTM island width using modified Rutherford equation.

    Parameters:
    w0 : initial island width (m)
    Delta_prime_bs : bootstrap drive (m^-1)
    Delta_prime_class : classical tearing stability parameter (m^-1)
    r_s : radius of rational surface (m)
    tau_R : resistive timescale (s)
    t_max : maximum time (s)
    dt : timestep (s)

    Returns:
    t_array : time array
    w_array : island width evolution
    """
    N_steps = int(t_max / dt)
    t_array = np.zeros(N_steps)
    w_array = np.zeros(N_steps)

    w = w0
    t = 0.0

    for i in range(N_steps):
        t_array[i] = t
        w_array[i] = w

        # Modified Rutherford equation: dw/dt = (r_s/τ_R) * (Δ'_class + L_qp/w^2)
        # Simplified: Δ'_bs ~ L_qp / w^2
        if w > 1e-6:  # Avoid singularity
            dw_dt = (r_s / tau_R) * (Delta_prime_class * w + Delta_prime_bs / w)
        else:
            dw_dt = 0.0

        w += dw_dt * dt
        t += dt

        # Stop if island saturates or decays
        if w < 0:
            w = 0
            break
        if w > 0.5:  # Cap at half minor radius
            break

    return t_array[:i+1], w_array[:i+1]

# Example: NTM at q=3/2 surface
r_s_ntm = 0.6  # m (60% of minor radius)
tau_R_ntm = 1.0  # s
Delta_prime_class_ntm = -0.5  # m^-1 (classically stable)
Delta_prime_bs_ntm = 0.001  # m (bootstrap drive parameter)

# Case 1: Small seed island (below threshold)
w0_small = 0.01  # m
t_small, w_small = ntm_island_evolution(w0_small, Delta_prime_bs_ntm, Delta_prime_class_ntm,
                                         r_s_ntm, tau_R_ntm, 10.0, 0.01)

# Case 2: Large seed island (above threshold)
w0_large = 0.05  # m
t_large, w_large = ntm_island_evolution(w0_large, Delta_prime_bs_ntm, Delta_prime_class_ntm,
                                         r_s_ntm, tau_R_ntm, 10.0, 0.01)

plt.figure(figsize=(10, 6))
plt.plot(t_small, w_small * 100, 'b-', linewidth=2, label=f'Small seed (w₀={w0_small*100:.1f} cm)')
plt.plot(t_large, w_large * 100, 'r-', linewidth=2, label=f'Large seed (w₀={w0_large*100:.1f} cm)')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Island Width (cm)', fontsize=12)
plt.title('NTM Island Width Evolution', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ntm_island_evolution.png', dpi=150)
plt.show()

print(f"\nNTM evolution:")
print(f"Small seed: final width = {w_small[-1]*100:.2f} cm (decays)")
print(f"Large seed: final width = {w_large[-1]*100:.2f} cm (grows)")
```

### 8.8 RFP Taylor State

```python
def rfp_taylor_state(r, a, mu_a):
    """
    Calculate RFP Taylor state magnetic field profiles.

    Parameters:
    r : radial coordinate (array)
    a : minor radius (m)
    mu_a : Taylor eigenvalue * a (dimensionless)

    Returns:
    B_z : toroidal field (normalized)
    B_theta : poloidal field (normalized)
    """
    from scipy.special import jv  # Bessel function

    x = mu_a * r / a
    B_z = jv(0, x)  # J_0
    B_theta = jv(1, x)  # J_1

    return B_z, B_theta

# RFP Taylor state
a_RFP = 0.5  # m
mu_a_RFP = 3.8  # First zero of J_0 is ~2.4, choose higher for reversal

r_RFP = np.linspace(0, a_RFP, 200)
B_z_RFP, B_theta_RFP = rfp_taylor_state(r_RFP, a_RFP, mu_a_RFP)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(r_RFP, B_z_RFP, 'b-', linewidth=2, label='$B_z$ (toroidal)')
ax1.plot(r_RFP, B_theta_RFP, 'r-', linewidth=2, label='$B_θ$ (poloidal)')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Radius r (m)', fontsize=12)
ax1.set_ylabel('Magnetic Field (normalized)', fontsize=12)
ax1.set_title('RFP Taylor State: Magnetic Field Profiles', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Field line pitch
q_RFP = np.where(np.abs(B_theta_RFP) > 0.01, B_z_RFP / B_theta_RFP * a_RFP / 6.0, np.nan)
ax2.plot(r_RFP, q_RFP, 'g-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Radius r (m)', fontsize=12)
ax2.set_ylabel('Safety Factor q', fontsize=12)
ax2.set_title('RFP Safety Factor (approximate)', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-2, 2])

plt.tight_layout()
plt.savefig('rfp_taylor_state.png', dpi=150)
plt.show()

print(f"\nRFP Taylor state: μa = {mu_a_RFP}")
print(f"Field reversal at r/a ~ {r_RFP[B_z_RFP < 0][0] / a_RFP:.2f}")
```

## 9. Summary

This lesson covered the major MHD aspects of magnetic confinement fusion:

1. **Tokamak configuration**: Toroidal + poloidal fields, safety factor, plasma current
2. **Tokamak equilibrium**: Grad-Shafranov equation, Shafranov shift, beta limits (Troyon limit)
3. **Major instabilities**:
   - **Sawteeth**: $q_0 < 1$, internal kink, Kadomtsev reconnection
   - **ELMs**: Peeling-ballooning modes, Type I/III, mitigation (RMP, pellet pacing, QH-mode)
   - **Disruptions**: Thermal quench, current quench, runaway electrons, mitigation (MGI, SPI)
   - **NTMs**: Bootstrap-driven island growth, ECCD stabilization
   - **RWMs**: Resistive wall modes, requires rotation or feedback control
4. **Stellarator**: 3D external coils, no plasma current, quasi-symmetry, no disruptions
5. **RFP**: Reversed toroidal field, Taylor relaxation, MHD dynamo

Understanding and controlling these MHD phenomena is essential for achieving practical fusion energy. ITER will test many of these concepts at reactor-relevant scales.

## Practice Problems

1. **Troyon limit**: For a tokamak with $I_p = 10$ MA, $a = 1.5$ m, $B_0 = 4$ T, calculate the maximum achievable beta using the Troyon limit ($\beta_N = 3$). What is the corresponding plasma pressure?

2. **Safety factor**: A tokamak has $R = 3$ m, $a = 1$ m, $B_0 = 5$ T, $I_p = 5$ MA. Calculate the edge safety factor $q_a$ assuming a flat current profile. Is this tokamak at risk of disruption ($q_a < 2$)?

3. **Sawtooth period**: Estimate the sawtooth period for a plasma with $a = 1$ m, $T_e = 3$ keV, $n_e = 5 \times 10^{19}$ m$^{-3}$, $B = 3$ T. Use the provided Python function. How does the period change if $T_e$ doubles?

4. **Greenwald density**: For ITER ($I_p = 15$ MA, $a = 2$ m), the Greenwald density limit is $n_G = 1.19 \times 10^{20}$ m$^{-3}$. If the average density is $n_e = 1.0 \times 10^{20}$ m$^{-3}$, what is the Greenwald fraction ($n_e / n_G$)? Is the plasma close to the density limit?

5. **ELM heat flux**: An ELM expels $W_{ELM} = 0.5$ MJ over a wetted area $A_{wet} = 1$ m$^2$ in $\tau_{ELM} = 1$ ms. Calculate the peak heat flux. Compare this to a typical material limit of 10 MW/m$^2$. Is mitigation required?

6. **Disruption forces**: During a disruption, the plasma current decays from $I_p = 5$ MA to zero in $\tau_{CQ} = 100$ ms. Estimate the current quench rate $dI_p/dt$ and the induced loop voltage using the Python function (assume $R = 3$ m, $a = 1$ m). What is the magnitude of the vertical force?

7. **NTM threshold**: An NTM is driven by bootstrap current with $\Delta'_{bs} = 0.001$ m and damped by classical tearing with $\Delta'_{class} = -1$ m$^{-1}$. Estimate the critical island width $w_{crit} \sim \sqrt{L_{qp}/|\Delta'_{class}|}$ where $L_{qp} = \Delta'_{bs} / r_s$ and $r_s = 0.5$ m. What seed island size is required to trigger the NTM?

8. **RFP field reversal**: For an RFP with $\mu a = 4.0$, find the radius where the toroidal field $B_z = 0$ (reversal surface). Use the Bessel function $J_0(x)$ and find the first zero. Express the result as $r/a$.

9. **Stellarator comparison**: List three advantages and three disadvantages of stellarators compared to tokamaks for fusion reactors. Under what circumstances might a stellarator be preferred?

10. **Beta optimization**: A tokamak operates at $\beta_N = 2.5$ (below the Troyon limit of 3.0). Propose two methods to increase the achievable beta without triggering MHD instabilities. Consider equilibrium shaping, current profile control, and kinetic stabilization.

---

**Previous**: [Accretion Disk MHD](./12_Accretion_Disk_MHD.md) | **Next**: [Space Weather MHD](./14_Space_Weather.md)

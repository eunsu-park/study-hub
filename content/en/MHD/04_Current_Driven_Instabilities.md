# 4. Current-Driven Instabilities

## Learning Objectives

- Understand kink instabilities (external and internal) driven by plasma current
- Analyze the sausage instability and its stabilization
- Derive tearing mode theory and magnetic island formation
- Study resistive wall modes and feedback stabilization
- Understand neoclassical tearing modes (NTMs) in tokamaks
- Compute Δ' parameter and growth rates for tearing modes
- Implement numerical solvers for current-driven instabilities
- Connect theory to experimental observations (sawteeth, disruptions)

## 1. Introduction to Current-Driven Instabilities

Current-driven instabilities are MHD modes where the **free energy source** is the plasma current (or equivalently, the magnetic field configuration). Unlike pressure-driven modes, these can exist even at **zero pressure**.

**Key features**:
- Driven by current/magnetic field configuration
- Can be ideal (no resistivity) or resistive (reconnection)
- Often limit maximum achievable plasma current
- Lead to major disruptions in tokamaks

```
Classification of Current-Driven Instabilities:
==============================================

IDEAL MHD (η = 0):
├─ External kink (m=1): Column bends, q(a) < 1
├─ Internal kink (m=1): Core kink, q(0) < 1
└─ Sausage (m=0): Pinching/expansion, stabilized by Bz

RESISTIVE MHD (η ≠ 0):
├─ Tearing mode: Magnetic reconnection at q = m/n
├─ Resistive wall mode: Wall-stabilized kink
└─ Neoclassical tearing mode (NTM): Bootstrap current
```

## 2. Kink Instability

### 2.1 External Kink Mode

The **external kink** (m=1, n=1 in toroidal systems) is a global displacement of the plasma column.

**Physical picture**:
```
External Kink (m=1):
===================

Before:          After:
   │                ╱
   │               ╱  ← Column bends
   │     →        │
   │               ╲
   │                ╲

Entire plasma column
displaces helically
```

**Energy balance**:
- **Destabilizing**: Magnetic pressure imbalance when column bends
- **Stabilizing**: Axial field $B_z$ provides line-bending energy

**Kruskal-Shafranov criterion** (from Lesson 2):

$$
q(a) > \frac{m}{n}
$$

For $(m,n) = (1,1)$:

$$
q(a) > 1
$$

### 2.2 Sharp-Boundary Model

For a sharp-boundary Z-pinch with:
- Radius $a$
- Current $I$
- Axial field $B_z$

The dispersion relation for $m=1$ kink:

$$
\omega^2 = -\frac{B_\theta^2(a)}{\mu_0\rho}\left(1 - q^2(a)\right)
$$

**Stability**:
- If $q(a) < 1$: $\omega^2 < 0$ → **unstable**
- If $q(a) > 1$: $\omega^2 > 0$ → **stable** (oscillatory)

**Growth rate** (when unstable):

$$
\gamma = \frac{B_\theta(a)}{\sqrt{\mu_0\rho}}\sqrt{1 - q^2(a)}
$$

This is on the **Alfvén timescale**: $\tau_A \sim a/v_A$.

### 2.3 Internal Kink Mode

The **internal kink** mode occurs when $q < 1$ somewhere in the plasma (typically on-axis).

**Characteristics**:
- Localized to region where $q < 1$
- Does not require wall or edge instability
- Causes **sawtooth oscillations** in tokamaks

**Sawtooth crash**:
1. Central temperature rises
2. Central current density rises
3. $q(0)$ drops below 1
4. Internal kink triggers
5. Rapid reconnection flattens $T$ and $q$ profiles
6. Cycle repeats

```
Sawtooth Oscillation:
====================

T₀ |     ╱|     ╱|     ╱
   |    ╱ |    ╱ |    ╱
   |   ╱  |   ╱  |   ╱
   |  ╱   |  ╱   |  ╱
   | ╱    | ╱    | ╱
   |╱_____|╱_____|╱_____ time
     ↑      ↑      ↑
   Crash  Crash  Crash
   (m=1,n=1 internal kink)
```

**Bussac criterion** for internal kink stability:

$$
\beta_p < \frac{0.3}{q_0^2}
$$

where $\beta_p$ is poloidal beta and $q_0 = q(0)$.

### 2.4 Sausage Instability (m=0)

The **sausage mode** ($m=0$) is a symmetric pinching/expansion:

```
Sausage Mode (m=0):
==================

Before:        After:
  ║║║║║         ║║║║║
  ║║║║║    →   ╱║║║║║╲
  ║║║║║         ║║║║║
  ║║║║║        ╱║║║║║╲
  ║║║║║         ║║║║║

Column pinches and expands
alternately along z
```

**Dispersion relation** (Z-pinch, no $B_z$):

$$
\omega^2 = -\frac{B_\theta^2(a)}{\mu_0\rho}k_z^2 a^2
$$

Always unstable ($\omega^2 < 0$) for $k_z \neq 0$.

**With axial field** $B_z$:

$$
\omega^2 = \frac{B_\theta^2(a)}{\mu_0\rho}k_z^2 a^2\left(\frac{B_z^2}{B_\theta^2(a)} - 1\right)
$$

**Stability**: Stable if $B_z > B_\theta(a)$, i.e., $q(a) > 1$.

### 2.5 Helical Perturbations

General helical perturbation:

$$
\boldsymbol{\xi} = \hat{\boldsymbol{\xi}}(r)e^{i(m\theta + k_z z - \omega t)}
$$

For a cylinder with toroidal identification: $k_z = n/R_0$.

**Resonant surface**: Where $\mathbf{k}\cdot\mathbf{B} = 0$:

$$
mB_\theta(r_s) + k_z B_z = 0 \quad \Rightarrow \quad q(r_s) = \frac{m}{n}
$$

At the resonant surface, field line bending vanishes → **ideal MHD becomes singular** → resistivity required.

## 3. Tearing Mode Theory

### 3.1 Resistive Instability Basics

In **ideal MHD**, magnetic topology is conserved (frozen-in theorem). In **resistive MHD**, finite resistivity $\eta$ allows **magnetic reconnection**.

**Tearing mode**: Resistive instability at rational surfaces where $q = m/n$.

**Physical mechanism**:
1. Perturbation creates current sheet at $q = m/n$ surface
2. Resistive diffusion breaks field lines
3. Reconnection forms magnetic islands
4. Islands grow, tearing apart flux surfaces

```
Tearing Mode and Magnetic Islands:
==================================

Initial:           Reconnected:
 ═══════            ═══╱═╲═══
 ═══════    →       ══╱   ╲══  (O-point)
 ═══════            ═╱  ⊗  ╲═
   ↑                 ╲     ╱
Rational              ═══════
surface              (X-point)

Island width w grows with time
```

### 3.2 Tearing Layer Analysis

Near the resonant surface $r = r_s$, there is a narrow **tearing layer** of width $\delta$.

**Layer structure**:
- Inner layer ($|r - r_s| < \delta$): resistivity important
- Outer region ($|r - r_s| > \delta$): ideal MHD

**Boundary layer equation** (constant-$\psi$ regime):

$$
\frac{d^4\psi}{dx^4} - k^2\frac{d^2\psi}{dx^2} + \frac{i\omega\mu_0}{\eta}\psi = 0
$$

where $x = r - r_s$ is distance from resonant surface.

**Matching condition**: Inner solution must match outer ideal solution.

### 3.3 Δ' Parameter

The **Δ' parameter** characterizes the jump in logarithmic derivative of $\psi$ across the resonant surface:

$$
\Delta' = \left[\frac{d(\ln\psi')}{dr}\right]_{r_s^+}^{r_s^-} = \left(\frac{1}{\psi'}\frac{d\psi'}{dr}\right)_{r_s^+} - \left(\frac{1}{\psi'}\frac{d\psi'}{dr}\right)_{r_s^-}
$$

where $\psi$ is the perturbed poloidal flux.

**Stability criterion**:
- $\Delta' > 0$: **unstable** (tearing)
- $\Delta' < 0$: **stable**
- $\Delta' = 0$: **marginal**

### 3.4 Growth Rate Scaling

The growth rate depends on regime:

**Constant-$\psi$ regime** (resistivity-dominated):

$$
\gamma \propto \eta^{3/5}(\Delta')^{4/5}
$$

**Non-constant-$\psi$ regime** (slower):

$$
\gamma \propto \eta^{1/3}(\Delta')
$$

**Typical values**:
- $\eta \sim 10^{-9}$ (tokamak conditions)
- $\gamma \sim 10^2 - 10^4$ s$^{-1}$ (much slower than Alfvén)

### 3.5 Magnetic Island Width

The saturated island width is related to the reconnected flux $\delta\psi$:

$$
w = 4\sqrt{\frac{\delta\psi r_s}{m B_\theta(r_s)}}
$$

**Island growth**: Described by the **Rutherford equation**:

$$
\frac{d(\delta\psi)}{dt} = \eta J_s \Delta'(w) w
$$

where $\Delta'(w)$ depends on island width.

**Nonlinear evolution**:
- Small islands: $\Delta'(w) \approx \Delta'(0)$ → exponential growth
- Large islands: $\Delta'(w)$ decreases → saturation or continued growth

## 4. Resistive Wall Mode

### 4.1 Wall Stabilization

A **conducting wall** near the plasma can stabilize ideal external kink modes by inducing **image currents** that cancel the perturbation.

**Ideal wall** (perfect conductor, $r = r_w$):
- Perturbation induces surface current
- Image current exactly cancels external field
- External kink stabilized even if $q(a) < 1$

**Resistive wall** (finite conductivity):
- Image current decays on resistive timescale $\tau_w = \mu_0\sigma d r_w$
- Mode grows slowly: $\gamma \sim 1/\tau_w$

```
Resistive Wall Mode:
===================

        Plasma
          ║
          ║  ← Perturbation
          ║
     ════════  Resistive wall
       ↓
    Induced current
    (decays on τw)

γ ~ 1/τw << γ_ideal
```

### 4.2 Growth Rate

For a resistive wall at radius $r_w$, the growth rate:

$$
\gamma_{RWM} \approx \frac{1}{\tau_w}\frac{r_w - a}{r_w}
$$

where $\tau_w = \mu_0\sigma d r_w$ is the wall time constant, $\sigma$ is conductivity, $d$ is wall thickness.

**Typical values**:
- $\tau_w \sim 0.01 - 1$ s (copper wall)
- $\gamma_{RWM} \sim 1 - 100$ s$^{-1}$ (intermediate timescale)

### 4.3 Feedback Stabilization

Since RWM grows slowly, it can be **actively stabilized** by feedback coils:

1. Detect mode (magnetic sensors)
2. Compute correction field
3. Drive coils to cancel perturbation
4. Feedback must be faster than $\gamma_{RWM}$

**Applications**:
- High-beta tokamak operation
- Advanced scenarios ($q(0) < 1$)
- Maintained equilibrium beyond no-wall beta limit

## 5. Neoclassical Tearing Mode (NTM)

### 5.1 Bootstrap Current Drive

In a tokamak, **bootstrap current** arises from pressure gradient due to collisional momentum conservation:

$$
J_{bs} \propto \frac{dp}{dr}
$$

When a magnetic island forms:
- Island flattens pressure profile locally
- Bootstrap current is reduced inside island
- Missing current creates helical current perturbation
- Perturbation drives further island growth

**Positive feedback loop** → **metastable NTM**

### 5.2 Modified Rutherford Equation

The island evolution including bootstrap drive:

$$
\tau_r\frac{dw}{dt} = r_s\Delta'(w) + r_s\frac{L_q}{L_p}\beta_p\left(\frac{w_d^2}{w^2} - \frac{w_{sat}^2}{w^2 + w_{sat}^2}\right)
$$

where:
- $\tau_r$: resistive diffusion time
- $\Delta'(w)$: stability index (can be negative)
- $L_q = q/(dq/dr)$: shear length
- $L_p = p/(dp/dr)$: pressure scale length
- $\beta_p$: poloidal beta
- $w_d$: threshold width for onset
- $w_{sat}$: saturated width

### 5.3 Threshold and Metastability

**Key feature**: NTM requires a **seed island** to grow.

- Small perturbation ($w < w_d$): island decays
- Above threshold ($w > w_d$): island grows to $w_{sat}$

**Triggering mechanisms**:
- Sawtooth crash
- ELM
- Other MHD events

```
NTM Metastability:
=================

dw/dt |        ╱
      |       ╱ ← Growth region
      |      ╱
      |_____╱_________ w
      |    ╱  ↑   ↑
      |   ╱  w_d w_sat
      |  ╱
      | ╱ ← Decay region

Two stable states:
1. w = 0 (no island)
2. w = w_sat (saturated)
```

### 5.4 Suppression Techniques

**Methods to suppress NTM**:
1. **Electron cyclotron current drive (ECCD)**: Drive current at island O-point to replace missing bootstrap current
2. **Reduce beta**: Lower pressure gradient → smaller bootstrap drive
3. **Increase rotation**: Stabilize through velocity shear
4. **Avoid triggers**: Reduce sawtooth amplitude, control ELMs

## 6. Numerical Implementation

### 6.1 Kink Mode Growth Rate

```python
import numpy as np
import matplotlib.pyplot as plt

def kink_growth_rate(q_edge, Btheta, rho):
    """
    Compute external kink growth rate for cylindrical plasma

    Parameters:
    -----------
    q_edge: safety factor at edge
    Btheta: poloidal field at edge [T]
    rho: plasma density [kg/m^3]

    Returns:
    --------
    gamma: growth rate [1/s] (0 if stable)
    stable: True if stable
    """
    mu0 = 4*np.pi*1e-7

    if q_edge < 1:
        # The Alfvénic prefactor Bθ/√(μ₀ρ) = v_A (poloidal Alfvén speed) sets
        # the growth rate scale: the kink instability is an ideal MHD mode
        # driven and stabilized by the magnetic field itself, so its timescale
        # is naturally the poloidal Alfvén transit time τ_A = a/v_A.
        # √(1 - q²) measures how far q is below the stability threshold q=1;
        # at q→0 the full poloidal Alfvén drive is realized, while at q→1
        # the mode is marginally stable and γ→0.
        gamma = (Btheta / np.sqrt(mu0 * rho)) * np.sqrt(1 - q_edge**2)
        stable = False
    else:
        # Stable
        gamma = 0.0
        stable = True

    return gamma, stable

def plot_kink_stability():
    """Plot kink growth rate vs q(a)"""

    # Plasma parameters
    Btheta = 0.5  # T
    rho = 1e-6    # kg/m^3
    mu0 = 4*np.pi*1e-7

    vA = Btheta / np.sqrt(mu0 * rho)  # Alfvén speed

    # Safety factor range
    q_vals = np.linspace(0.1, 2.0, 100)

    gamma_vals = []
    for q in q_vals:
        gamma, stable = kink_growth_rate(q, Btheta, rho)
        gamma_vals.append(gamma)

    gamma_vals = np.array(gamma_vals)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot growth rate
    ax.plot(q_vals, gamma_vals, 'b-', linewidth=2)

    # Mark stability boundary
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2,
               label='Stability boundary (q=1)')

    # Shade unstable region
    unstable_mask = q_vals < 1.0
    if np.any(unstable_mask):
        ax.fill_between(q_vals, 0, gamma_vals, where=unstable_mask,
                        alpha=0.3, color='red', label='Unstable')

    # Shade stable region
    stable_mask = q_vals >= 1.0
    ax.fill_between(q_vals[stable_mask], 0, gamma_vals[stable_mask],
                    alpha=0.3, color='green', label='Stable')

    ax.set_xlabel('Edge safety factor q(a)', fontsize=12)
    ax.set_ylabel('Growth rate γ [1/s]', fontsize=12)
    ax.set_title(f'External Kink Growth Rate (vₐ = {vA:.2e} m/s)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig

def example_kink_instability():
    """Analyze external kink for various configurations"""

    print("=== External Kink Instability Analysis ===\n")

    # Plasma parameters
    a = 0.5       # Minor radius [m]
    I = 1e6       # Plasma current [A]
    n = 1e20      # Density [m^-3]
    T = 1e7       # Temperature [K]
    mu0 = 4*np.pi*1e-7
    mp = 1.67e-27

    rho = n * mp

    # Poloidal field at edge
    Btheta = mu0 * I / (2*np.pi*a)

    print(f"Plasma parameters:")
    print(f"  Radius: a = {a} m")
    print(f"  Current: I = {I/1e6} MA")
    print(f"  Density: n = {n:.2e} m^-3")
    print(f"  Edge poloidal field: Bθ(a) = {Btheta:.3f} T")

    # Test various toroidal fields (varying q)
    R0 = 3.0  # Major radius
    Bt_values = [1.0, 2.0, 3.0, 5.0]  # Toroidal field [T]

    print(f"\nKink stability for various toroidal fields (R0 = {R0} m):")
    print("-" * 60)

    for Bt in Bt_values:
        # Safety factor
        q_edge = (a * Bt) / (R0 * Btheta)

        # Growth rate
        gamma, stable = kink_growth_rate(q_edge, Btheta, rho)

        # Alfvén time
        vA = Btheta / np.sqrt(mu0 * rho)
        tau_A = a / vA

        status = "STABLE" if stable else "UNSTABLE"

        print(f"\nBt = {Bt} T:")
        print(f"  q(a) = {q_edge:.2f}")
        print(f"  Status: {status}")

        if not stable:
            print(f"  Growth rate: γ = {gamma:.2e} s^-1")
            print(f"  Growth time: τ = {1/gamma:.2e} s")
            print(f"  Alfvén time: τA = {tau_A:.2e} s")
            print(f"  γ/ωA = {gamma * tau_A:.2f}")

    # Plot
    fig = plot_kink_stability()
    plt.savefig('/tmp/kink_stability.png', dpi=150)
    print("\n\nKink stability plot saved to /tmp/kink_stability.png")
    plt.close()

if __name__ == "__main__":
    example_kink_instability()
```

### 6.2 Δ' Calculation for Tearing Mode

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def solve_outer_region(r, rs, m, q_profile, psi_rs):
    """
    Solve ideal MHD equation in outer region

    d/dr(r d psi/dr) - m² psi/r = 0 at resonance

    Near resonance: psi ~ C+ (r-rs)  for r > rs
                          C- (r-rs)  for r < rs

    Returns derivative at rs
    """
    # Simplified: assume psi' = const near rs
    # In reality, solve ODE from boundaries

    # For a simple current profile, analytical Δ' exists
    # Here we use a model

    # Estimate from q-profile curvature
    dr = 0.001
    q_rs = q_profile(rs)
    q_p = (q_profile(rs + dr) - q_profile(rs - dr)) / (2*dr)
    q_pp = (q_profile(rs + dr) - 2*q_profile(rs) + q_profile(rs - dr)) / dr**2

    # Approximate Δ' from q''
    # Δ' ≈ (2/r_s) + (q''/q')
    Delta_prime = (2/rs) + (q_pp / q_p) if abs(q_p) > 1e-10 else 0

    return Delta_prime

def tearing_growth_rate(Delta_prime, eta, rs, Btheta):
    """
    Estimate tearing mode growth rate

    γ ~ η^(3/5) Δ'^(4/5)  (constant-psi regime)

    Parameters:
    -----------
    Delta_prime: stability index [1/m]
    eta: resistivity [Ohm*m]
    rs: resonant surface radius [m]
    Btheta: poloidal field [T]

    Returns:
    --------
    gamma: growth rate [1/s]
    """
    mu0 = 4*np.pi*1e-7

    if Delta_prime <= 0:
        return 0.0  # Stable

    # τ_R = μ₀rs²/η is the resistive diffusion time at the resonant surface:
    # it sets the "clock" for how quickly resistivity can diffuse through the
    # layer of thickness rs.  All tearing growth rates scale with 1/τ_R
    # because reconnection requires resistive diffusion to break field lines.
    tau_R = mu0 * rs**2 / eta

    # The constant-ψ (Furth-Killeen-Rosenbluth) scaling γ ∝ η^(3/5)Δ'^(4/5)
    # arises from matching the outer ideal-MHD solution (characterized by Δ')
    # to the inner resistive layer: the fractional powers reflect the fact that
    # both inertia and resistivity compete inside the thin tearing layer, and
    # neither is negligible.  η^(3/5) means growth slows as resistivity drops,
    # but not as fast as simple diffusion (which would give η^1).
    gamma = (eta / (mu0 * rs**2))**(3/5) * (Delta_prime * rs)**(4/5)

    return gamma

def plot_delta_prime():
    """Plot Δ' for various current profiles"""

    a = 0.5  # Plasma radius
    R0 = 3.0

    # Different current profiles
    profiles = {
        'Parabolic': lambda r: 1.0 + 2.0*(r/a)**2,
        'Peaked': lambda r: 1.0 + 3.0*(r/a)**4,
        'Hollow': lambda r: 0.8 + 0.5*(r/a) + 2.0*(r/a)**2,
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: q-profiles
    ax = axes[0]
    r_vals = np.linspace(0.01, a, 200)

    for name, q_func in profiles.items():
        q_vals = [q_func(r) for r in r_vals]
        ax.plot(r_vals/a, q_vals, linewidth=2, label=name)

    # Mark rational surfaces
    for q_rational in [1.5, 2.0, 2.5, 3.0]:
        ax.axhline(q_rational, color='gray', linestyle=':', alpha=0.5)
        ax.text(1.05, q_rational, f'q={q_rational}', fontsize=9)

    ax.set_xlabel('r/a', fontsize=12)
    ax.set_ylabel('q(r)', fontsize=12)
    ax.set_title('Safety Factor Profiles', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # Panel 2: Δ' at q=2 surface
    ax = axes[1]

    m = 2
    q_target = 2.0

    Delta_primes = []
    rs_vals = []

    for name, q_func in profiles.items():
        # Find resonant surface where q = 2
        for r in r_vals:
            if abs(q_func(r) - q_target) < 0.05:
                rs = r
                break
        else:
            rs = None

        if rs is not None:
            Delta_p = solve_outer_region(r_vals, rs, m, q_func, psi_rs=1.0)
            Delta_primes.append(Delta_p)
            rs_vals.append(rs)
        else:
            Delta_primes.append(0)
            rs_vals.append(0)

    colors = ['blue', 'orange', 'green']
    x_pos = np.arange(len(profiles))

    bars = ax.bar(x_pos, Delta_primes, color=colors, alpha=0.7)

    # Mark stability boundary
    ax.axhline(0, color='red', linestyle='--', linewidth=2,
               label='Δ\'=0 (marginal)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(profiles.keys())
    ax.set_ylabel('Δ\' [1/m]', fontsize=12)
    ax.set_title(f'Tearing Stability Index at q={q_target} (m={m})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate stability
    for i, (dp, bar) in enumerate(zip(Delta_primes, bars)):
        if dp > 0:
            status = "UNSTABLE"
            color = 'red'
        else:
            status = "STABLE"
            color = 'green'

        ax.text(i, dp + 0.1, status, ha='center', fontsize=10,
                color=color, fontweight='bold')

    plt.tight_layout()
    return fig

def example_tearing_mode():
    """Analyze tearing mode for tokamak"""

    print("=== Tearing Mode Analysis ===\n")

    # Parameters
    a = 0.5
    R0 = 3.0
    Bt = 5.0
    Ip = 1e6
    eta = 1e-7  # Resistivity [Ohm*m]

    mu0 = 4*np.pi*1e-7

    # Poloidal field
    Btheta = mu0 * Ip / (2*np.pi*a)

    # q-profile (parabolic)
    def q_profile(r):
        q0 = 1.0
        qa = 3.0
        return q0 + (qa - q0)*(r/a)**2

    # Find q=2 surface
    m = 2
    n = 1
    q_target = m / n

    r_vals = np.linspace(0.01, a, 1000)
    for r in r_vals:
        if abs(q_profile(r) - q_target) < 0.001:
            rs = r
            break

    print(f"Plasma parameters:")
    print(f"  Minor radius: a = {a} m")
    print(f"  Major radius: R0 = {R0} m")
    print(f"  Plasma current: Ip = {Ip/1e6} MA")
    print(f"  Resistivity: η = {eta:.2e} Ω·m")

    print(f"\nResonant surface (m={m}, n={n}):")
    print(f"  q = {q_target}")
    print(f"  Location: rs = {rs:.3f} m (rs/a = {rs/a:.2f})")

    # Compute Δ'
    Delta_p = solve_outer_region(r_vals, rs, m, q_profile, psi_rs=1.0)

    print(f"\nStability index:")
    print(f"  Δ' = {Delta_p:.2f} m^-1")

    if Delta_p > 0:
        print("  Status: UNSTABLE (tearing mode)")

        # Growth rate
        gamma = tearing_growth_rate(Delta_p, eta, rs, Btheta)

        print(f"\nGrowth rate:")
        print(f"  γ = {gamma:.2e} s^-1")
        print(f"  Growth time: τ = {1/gamma:.2e} s")

        # Compare to resistive diffusion time
        tau_R = mu0 * rs**2 / eta
        print(f"  Resistive time: τR = {tau_R:.2e} s")
        print(f"  γ τR = {gamma * tau_R:.3f}")

        # Estimate island width (nonlinear)
        # Simplified: w ~ sqrt(Δ' * rs)
        w_estimate = np.sqrt(Delta_p * rs) * 0.1  # Order of magnitude
        print(f"\nEstimated saturated island width: w ~ {w_estimate:.3f} m")
        print(f"  (w/rs = {w_estimate/rs:.2f})")

    else:
        print("  Status: STABLE")

    # Plot
    fig = plot_delta_prime()
    plt.savefig('/tmp/tearing_delta_prime.png', dpi=150)
    print("\nΔ' plot saved to /tmp/tearing_delta_prime.png")
    plt.close()

if __name__ == "__main__":
    example_tearing_mode()
```

### 6.3 Rutherford Island Evolution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rutherford_equation(t, w, Delta_prime_func, eta, Js, rs):
    """
    Rutherford equation for island width evolution

    τR dw/dt = r_s Δ'(w) w

    Parameters:
    -----------
    t: time [s]
    w: island width [m]
    Delta_prime_func: function Δ'(w)
    eta: resistivity [Ohm*m]
    Js: current density at resonant surface [A/m^2]
    rs: resonant surface radius [m]

    Returns:
    --------
    dw/dt: growth rate [m/s]
    """
    mu0 = 4*np.pi*1e-7

    # τ_R = μ₀rs²/η is the resistive diffusion time: it appears in the
    # denominator because resistivity is required to change the magnetic
    # topology — faster resistive diffusion (larger η, smaller τ_R) means
    # islands grow more quickly by allowing field lines to slip.
    tau_R = mu0 * rs**2 / eta

    # Δ'(w) now depends on island width because large islands flatten the
    # current density profile inside them, reducing the driving term;
    # this is the nonlinear feedback mechanism that leads to saturation.
    Delta_p = Delta_prime_func(w)

    # The Rutherford equation dw/dt ∝ Δ'(w)·w/τ_R has two key features:
    # 1. Linear dependence on w: unlike the linear phase (exponential growth),
    #    the nonlinear island grows only algebraically (w ∝ t), because the
    #    outer ideal region fully accommodates the island without additional
    #    free energy release as w increases.
    # 2. Saturation: when Δ'(w) → 0 at w = w_sat, dw/dt → 0 — the island
    #    reaches a self-consistent width at which no further reconnection is
    #    driven by the current gradient.
    dwdt = (rs * Delta_p * w) / tau_R

    return dwdt

def Delta_prime_saturating(w, Delta0, w_sat):
    """
    Δ'(w) model with saturation

    Δ'(w) = Δ0 * (1 - w²/w_sat²)

    As w → w_sat, Δ' → 0 → saturation
    """
    if w >= w_sat:
        return 0.0
    else:
        # The (1 - w²/w_sat²) factor is a simple model for how the island
        # flattens the current density profile: a wider island removes more
        # of the current gradient that drives the tearing mode, progressively
        # reducing Δ' until growth ceases at w = w_sat.  More sophisticated
        # models (e.g., polynomial in w) yield slightly different saturation
        # levels but the same qualitative nonlinear behaviour.
        return Delta0 * (1 - (w/w_sat)**2)

def plot_island_evolution():
    """Simulate and plot island evolution"""

    # Parameters
    rs = 0.3      # Resonant surface [m]
    eta = 1e-7    # Resistivity [Ohm*m]
    Js = 1e6      # Current density [A/m^2]
    Delta0 = 5.0  # Initial Δ' [1/m]
    w_sat = 0.1   # Saturation width [m]

    # Δ'(w) function
    Delta_prime_func = lambda w: Delta_prime_saturating(w, Delta0, w_sat)

    # Initial condition
    w0 = 0.001  # Initial seed island [m]

    # Time span
    mu0 = 4*np.pi*1e-7
    tau_R = mu0 * rs**2 / eta
    t_span = (0, 5*tau_R)
    t_eval = np.linspace(0, t_span[1], 500)

    # Solve
    sol = solve_ivp(
        lambda t, w: rutherford_equation(t, w, Delta_prime_func, eta, Js, rs),
        t_span,
        [w0],
        t_eval=t_eval,
        method='RK45'
    )

    t = sol.t
    w = sol.y[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Island width vs time
    ax = axes[0]
    ax.plot(t/tau_R, w*100, 'b-', linewidth=2)
    ax.axhline(w_sat*100, color='r', linestyle='--', linewidth=2,
               label=f'Saturation (w_sat = {w_sat*100} cm)')

    ax.set_xlabel('Time (τR)', fontsize=12)
    ax.set_ylabel('Island width [cm]', fontsize=12)
    ax.set_title('Magnetic Island Growth (Rutherford Equation)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Growth rate vs width
    ax = axes[1]

    # Compute dw/dt
    dwdt = np.array([rutherford_equation(ti, wi, Delta_prime_func, eta, Js, rs)
                     for ti, wi in zip(t, w)])

    ax.plot(w*100, dwdt*100, 'g-', linewidth=2)
    ax.axvline(w_sat*100, color='r', linestyle='--', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=1)

    ax.set_xlabel('Island width [cm]', fontsize=12)
    ax.set_ylabel('Growth rate dw/dt [cm/s]', fontsize=12)
    ax.set_title('Island Growth Rate vs Width', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def example_island_evolution():
    """Simulate magnetic island evolution"""

    print("=== Magnetic Island Evolution (Rutherford Equation) ===\n")

    # Parameters
    rs = 0.3
    eta = 1e-7
    Delta0 = 5.0
    w_sat = 0.1

    mu0 = 4*np.pi*1e-7
    tau_R = mu0 * rs**2 / eta

    print(f"Parameters:")
    print(f"  Resonant surface: rs = {rs} m")
    print(f"  Resistivity: η = {eta:.2e} Ω·m")
    print(f"  Initial Δ': Δ0 = {Delta0} m^-1")
    print(f"  Saturation width: w_sat = {w_sat*100} cm")
    print(f"  Resistive time: τR = {tau_R:.2f} s")

    # Initial island
    w0 = 0.001

    print(f"\nInitial seed island: w0 = {w0*100} cm")

    # Time to reach half saturation
    # Approximate: w(t) ~ w_sat tanh(t/τ_growth)
    # τ_growth ~ τR / (rs Δ0)
    tau_growth = tau_R / (rs * Delta0)

    print(f"Growth timescale: τ_growth ~ {tau_growth:.2f} s")
    print(f"  (τ_growth / τR = {tau_growth/tau_R:.3f})")

    # Plot
    fig = plot_island_evolution()
    plt.savefig('/tmp/island_evolution.png', dpi=150)
    print("\nIsland evolution plot saved to /tmp/island_evolution.png")
    plt.close()

if __name__ == "__main__":
    example_island_evolution()
```

## 7. Experimental Observations

### 7.1 Sawteeth in Tokamaks

**Sawtooth oscillations** are periodic crashes of central temperature caused by internal kink (m=1, n=1):

- **Ramp phase**: Central heating → $T_0$ rises, $q(0)$ drops
- **Crash**: When $q(0) < 1$, internal kink triggers → fast reconnection
- **Flattening**: Temperature and current profiles flatten → $q(0)$ rises above 1
- **Repeat**: Cycle period typically 10-100 ms

**Impact**:
- Beneficial: Prevents impurity accumulation, limits central pressure
- Detrimental: Triggers NTMs, limits fusion performance

### 7.2 Disruptions

**Major disruptions** are catastrophic events caused by large-scale MHD instabilities:

**Thermal quench** (ms timescale):
- Loss of confinement
- Temperature collapses
- Runaway electron generation

**Current quench** (ms timescale):
- Plasma current decays
- Large induced voltages in structures
- Mechanical forces on vessel

**Mitigation strategies** (ITER):
- Massive gas injection
- Shattered pellet injection
- Disruption prediction and avoidance

### 7.3 Neoclassical Tearing Modes

Observed in high-beta tokamak plasmas:
- Triggered by sawteeth or ELMs
- Grow to large islands (10-20% of minor radius)
- Degrade confinement
- Can lead to disruptions

**Control methods**:
- ECCD at island O-point
- Rotation control
- Beta reduction

## Summary

In this lesson, we have studied current-driven MHD instabilities:

1. **Kink Instability**: External kink ($q(a) < 1$) bends entire column; internal kink ($q(0) < 1$) causes sawteeth. Stabilized by sufficient axial field ($q > 1$).

2. **Sausage Mode**: m=0 pinching/expansion, always unstable without $B_z$. Stabilized when $B_z > B_\theta$.

3. **Tearing Mode**: Resistive reconnection at rational surfaces ($q = m/n$). Stability determined by Δ' parameter. Growth rate $\gamma \propto \eta^{3/5}(\Delta')^{4/5}$.

4. **Magnetic Islands**: Formed by tearing modes. Width evolution governed by Rutherford equation. Saturate when Δ'(w) → 0.

5. **Resistive Wall Mode**: Slow-growing kink mode on resistive wall timescale. Can be stabilized by feedback control.

6. **Neoclassical Tearing Mode**: Metastable islands driven by missing bootstrap current. Require seed island to grow. Major concern for high-beta operation.

7. **Numerical Tools**: Growth rate calculations, Δ' estimation, island evolution simulations.

These instabilities fundamentally limit tokamak performance and motivate advanced control strategies, profile optimization, and disruption mitigation systems for ITER and future reactors.

## Practice Problems

### Problem 1: External Kink Stability

A cylindrical Z-pinch has:
- Radius $a = 0.1$ m
- Current $I = 500$ kA
- Density $\rho = 10^{-6}$ kg/m³
- No axial field initially

**(a)** Compute the azimuthal field at the edge $B_\theta(a)$.

**(b)** Calculate the Alfvén speed $v_A = B_\theta/\sqrt{\mu_0\rho}$.

**(c)** Since $B_z = 0$, what is $q(a)$?

**(d)** Is the configuration stable or unstable to external kink?

**(e)** If unstable, compute the growth rate $\gamma$.

**(f)** What minimum $B_z$ is required to stabilize the kink ($q(a) > 1$)?

### Problem 2: Sawtooth Period

A tokamak has:
- Central temperature $T_0(t=0) = 5$ keV
- Heating power $P = 10$ MW
- Central volume $V_c \approx 1$ m³
- Particle density $n = 10^{20}$ m$^{-3}$

Assume central temperature rises linearly until $q(0) = 0.95$ triggers a sawtooth crash.

**(a)** Estimate the heating rate $dT_0/dt = P/(3nV_c k_B)$.

**(b)** If sawtooth crashes when $T_0 = 6$ keV, compute the time to crash.

**(c)** After crash, $T_0$ drops to 4 keV. What is the sawtooth period?

**(d)** How much energy is redistributed per crash?

**(e)** Compare the crash timescale (~ Alfvén time $\tau_A \sim$ 1 μs) to the ramp timescale.

### Problem 3: Tearing Mode Δ'

For a cylindrical plasma with current density:

$$
J_z(r) = J_0\left(1 - \frac{r^2}{a^2}\right)
$$

The safety factor is:

$$
q(r) = q_0\frac{1 + (r/a)^2}{1 - (r^2/a^2)} \quad (q_0 = q(0))
$$

**(a)** Find the location $r_s$ where $q(r_s) = 2$ (assume $q_0 = 1$).

**(b)** Compute $q'(r_s)$ and $q''(r_s)$ numerically or analytically.

**(c)** Estimate $\Delta' \approx \frac{2}{r_s} + \frac{q''}{q'}$ at $r_s$.

**(d)** Is the $m=2, n=1$ mode stable or unstable?

**(e)** If $\eta = 10^{-7}$ Ω·m and $a = 0.5$ m, estimate the growth rate.

### Problem 4: Magnetic Island Width

A tearing mode has reconnected flux $\delta\psi = 10^{-3}$ Wb at resonant surface $r_s = 0.3$ m.

The equilibrium poloidal field at $r_s$ is $B_\theta(r_s) = 0.4$ T, and mode number $m=2$.

**(a)** Compute the island width:
$$
w = 4\sqrt{\frac{\delta\psi r_s}{m B_\theta(r_s)}}
$$

**(b)** If the island grows to $w = 5$ cm, what is the new $\delta\psi$?

**(c)** Estimate the flattened region in the pressure profile (approximately $\pm w/2$ around $r_s$).

**(d)** If $dp/dr = -10^6$ Pa/m, how much pressure gradient is lost due to flattening?

**(e)** Discuss the impact on bootstrap current and NTM drive.

### Problem 5: Resistive Wall Mode

A tokamak has:
- Plasma minor radius $a = 1$ m
- Resistive wall at $r_w = 1.2$ m
- Wall thickness $d = 5$ cm
- Wall conductivity $\sigma = 5\times 10^7$ S/m (copper)

**(a)** Compute the wall time constant $\tau_w = \mu_0\sigma d r_w$.

**(b)** Estimate the RWM growth rate:
$$
\gamma_{RWM} \approx \frac{1}{\tau_w}\frac{r_w - a}{r_w}
$$

**(c)** If feedback control has response time $\tau_{fb} = 10$ ms, can it stabilize the RWM?

**(d)** What is the required bandwidth for the feedback system?

**(e)** If the wall were ideal (perfect conductor), what would happen to the external kink?

---

**Previous**: [Pressure-Driven Instabilities](./03_Pressure_Driven_Instabilities.md) | **Next**: [Reconnection Theory](./05_Reconnection_Theory.md)

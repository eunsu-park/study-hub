# 3. Pressure-Driven Instabilities

## Learning Objectives

- Understand the physical mechanism of interchange instabilities driven by unfavorable curvature
- Analyze the Rayleigh-Taylor instability in magnetized plasmas
- Study the Parker instability in stratified atmospheres
- Derive and apply ballooning mode theory for high-n modes in toroidal geometry
- Use the Mercier criterion for local interchange stability
- Implement numerical simulations of pressure-driven instabilities
- Understand the connection to experimental observations (ELMs in tokamaks)

## 1. Introduction to Pressure-Driven Instabilities

Pressure-driven instabilities arise when the **pressure gradient** provides free energy that can drive fluid motion against magnetic field line bending. These instabilities are particularly important in:

- **Fusion plasmas**: Limiting achievable pressure (beta limit)
- **Astrophysical plasmas**: Solar prominences, coronal mass ejections
- **Planetary magnetospheres**: Magnetic field configuration in magnetotail

The fundamental physics is the competition between:
- **Destabilizing**: Pressure gradient in unfavorable curvature
- **Stabilizing**: Magnetic field line bending (tension)

```
Pressure-Driven Instability Mechanism:
=====================================

Favorable curvature           Unfavorable curvature
(magnetic well):              (magnetic hill):

    ∇p                            ∇p
     ↓                             ↑
  ═══════  B                   ═══════  B
 (       )                      ‾‾‾‾‾‾‾
  ‾‾‾‾‾‾‾                      (       )

Pressure pushes              Pressure pushes
against field bending        in same direction
→ STABLE                     as curvature
                             → UNSTABLE
```

## 2. Interchange Instability

### 2.1 Physical Picture

The **interchange instability** occurs when adjacent flux tubes exchange positions. This is analogous to the Rayleigh-Taylor instability in hydrodynamics but with magnetic field replacing gravity.

**Energy consideration**:

Consider two flux tubes at different radii with:
- Tube 1 at $r_1$: pressure $p_1$, magnetic field $B_1$
- Tube 2 at $r_2 > r_1$: pressure $p_2 < p_1$, magnetic field $B_2$

If we interchange them, the change in potential energy is:

$$
\delta W \propto (p_1 - p_2)(B_2^2 - B_1^2)
$$

**Instability condition**: If $B$ decreases outward faster than $p$, then $\delta W < 0$ → unstable.

### 2.2 Curvature and Pressure Gradient

The stability depends on the sign of:

$$
\kappa \cdot \nabla p
$$

where $\boldsymbol{\kappa} = \mathbf{b}\cdot\nabla\mathbf{b}$ is the field line curvature, and $\mathbf{b} = \mathbf{B}/B$.

**Favorable curvature** ($\boldsymbol{\kappa} \cdot \nabla p < 0$):
- Curvature points away from high pressure
- Like a heavy fluid above a light fluid in gravity (stable)

**Unfavorable curvature** ($\boldsymbol{\kappa} \cdot \nabla p > 0$):
- Curvature points toward high pressure
- Like a light fluid below a heavy fluid in gravity (unstable)

### 2.3 Interchange Condition

From the energy principle, the interchange instability requires:

$$
\int \left[\frac{|\mathbf{B}_1|^2}{\mu_0} - 2(\boldsymbol{\xi}\cdot\boldsymbol{\kappa})(\boldsymbol{\xi}\cdot\nabla p)\right]dV < 0
$$

For incompressible perturbations perpendicular to $\mathbf{B}$:

$$
\delta W \approx -\int (\boldsymbol{\xi}_\perp\cdot\boldsymbol{\kappa})(\boldsymbol{\xi}_\perp\cdot\nabla p)\, dV
$$

If $\boldsymbol{\kappa}\cdot\nabla p > 0$, we can choose $\boldsymbol{\xi}_\perp$ parallel to both $\boldsymbol{\kappa}$ and $\nabla p$ to make $\delta W < 0$.

### 2.4 Good vs Bad Curvature in Tokamaks

In a tokamak, the magnetic field has toroidal and poloidal components. Going around poloidally:

**Outboard side** (low-field side, large $R$):
- Field lines curve inward (toward plasma)
- $\boldsymbol{\kappa}$ points inward
- $\nabla p$ points outward
- $\boldsymbol{\kappa}\cdot\nabla p < 0$ → **favorable** ("good curvature")

**Inboard side** (high-field side, small $R$):
- Field lines curve outward (away from plasma)
- $\boldsymbol{\kappa}$ points outward
- $\nabla p$ points outward
- $\boldsymbol{\kappa}\cdot\nabla p > 0$ → **unfavorable** ("bad curvature")

```
Tokamak Cross-Section:
=====================

        Bad curvature
             ↓
        ═══════════
       ║           ║
       ║  Plasma   ║  ← Good curvature
       ║           ║
        ═══════════

Instabilities tend to localize
on bad curvature (inboard) side
```

### 2.5 Average Curvature and Stability

For a closed field line, the **average curvature** determines stability:

$$
\bar{\kappa} = \frac{1}{L}\oint \boldsymbol{\kappa}\cdot d\mathbf{l}
$$

If $\bar{\kappa}\cdot\nabla p > 0$: potentially unstable (requires detailed calculation).

**Magnetic well**: Configuration where $\bar{\kappa}\cdot\nabla p < 0$ everywhere → stable.

## 3. Rayleigh-Taylor Instability in MHD

### 3.1 Hydrodynamic Rayleigh-Taylor

In hydrodynamics, a heavy fluid on top of a light fluid in a gravitational field is unstable.

**Dispersion relation** (no magnetic field):

$$
\omega^2 = -gk\frac{\rho_2 - \rho_1}{\rho_2 + \rho_1}
$$

If $\rho_2 > \rho_1$ (heavy on top): $\omega^2 < 0$ → unstable.

Growth rate: $\gamma = \sqrt{gk}$ (independent of viscosity).

### 3.2 MHD Rayleigh-Taylor with Transverse Field

Add a horizontal magnetic field $\mathbf{B}_0 = B_0\hat{\mathbf{x}}$ perpendicular to gravity $\mathbf{g} = -g\hat{\mathbf{z}}$.

**Modified dispersion relation**:

$$
\omega^2 = -gk\frac{\rho_2 - \rho_1}{\rho_2 + \rho_1} + \frac{B_0^2}{\mu_0(\rho_1 + \rho_2)}k_x^2
$$

where $k_x$ is the wavenumber along the field.

**Stability analysis**:

- Perturbations with $\mathbf{k} \parallel \mathbf{B}$ ($k_x = k$): stabilized if
  $$
  \frac{B_0^2}{\mu_0} > g(\rho_2 - \rho_1)/k
  $$

- Perturbations with $\mathbf{k} \perp \mathbf{B}$ ($k_x = 0$): **always unstable** (same as hydro).

**Critical wavenumber** for stabilization:

$$
k_c = \frac{g(\rho_2 - \rho_1)\mu_0}{B_0^2}
$$

Short wavelengths ($k > k_c$) are stable; long wavelengths are unstable.

### 3.3 Growth Rate

For unstable modes ($k < k_c$):

$$
\gamma = \sqrt{gk\frac{\rho_2-\rho_1}{\rho_2+\rho_1} - \frac{B_0^2}{\mu_0(\rho_1+\rho_2)}k_x^2}
$$

Maximum growth rate (at $k_x = 0$):

$$
\gamma_{max} = \sqrt{gk\frac{\rho_2-\rho_1}{\rho_2+\rho_1}}
$$

### 3.4 Analogy to Magnetic Curvature

The gravitational acceleration $\mathbf{g}$ can be replaced by effective gravity from magnetic curvature:

$$
\mathbf{g}_{eff} = \frac{B^2}{\mu_0\rho}\boldsymbol{\kappa}
$$

This connects RT instability to interchange instability.

## 4. Parker Instability

### 4.1 Magnetic Buoyancy

The **Parker instability** (Parker, 1966) occurs in a stratified atmosphere with a horizontal magnetic field. It is driven by **magnetic buoyancy**: the weight of plasma slides along bent field lines.

**Configuration**:
- Stratified atmosphere: $\rho(z)$, $p(z)$
- Horizontal field: $\mathbf{B}_0 = B_0(z)\hat{\mathbf{x}}$
- Gravity: $\mathbf{g} = -g\hat{\mathbf{z}}$

### 4.2 Physical Mechanism

When a field line is bent upward:
1. Plasma slides down along the field line (due to gravity component along $\mathbf{B}$)
2. Density at the apex decreases
3. Magnetic pressure pushes the apex further up
4. **Runaway instability**

```
Parker Instability:
==================

Initial:     Perturbed:
B ────────   B ╱‾‾‾‾╲
             Plasma slides
   ρgh         ╲    ╱
                ╲  ╱
                 ▼▼
             Less mass at apex
             → magnetic buoyancy
             → further uplift
```

### 4.3 Dispersion Relation

For isothermal atmosphere with scale height $H = kT/(mg)$:

$$
\omega^2 = -\frac{g}{H}\left(1 - \frac{B_0^2}{B_0^2 + \mu_0\rho_0 c_s^2}\right)
$$

where $c_s = \sqrt{\gamma p/\rho}$ is the sound speed.

**Instability condition**:

$$
\beta = \frac{2\mu_0 p}{B^2} > \frac{2}{\gamma} \approx 1.2 \quad (\text{for } \gamma=5/3)
$$

If plasma beta is too high, magnetic field cannot support the plasma → Parker unstable.

### 4.4 Applications

- **Interstellar medium**: Formation of molecular clouds
- **Solar atmosphere**: Prominence eruptions
- **Galactic dynamics**: Vertical structure of disk galaxies

## 5. Ballooning Modes

### 5.1 High-n Instabilities in Toroidal Geometry

**Ballooning modes** are high-$n$ (large toroidal mode number) pressure-driven instabilities that localize on the bad curvature side of a torus.

**Characteristics**:
- Large $n$ → short perpendicular wavelength
- Localized on outboard side (unfavorable curvature)
- Driven by pressure gradient
- Stabilized by magnetic shear and favorable average curvature

### 5.2 Physical Picture

The mode "balloons out" on the bad curvature side, like a balloon expanding where the field is weakest.

```
Ballooning Mode in Tokamak:
===========================

   Top view:

        n=10 perturbation
         ║ ║ ║ ║ ║
    ════╬═╬═╬═╬═╬════  Outboard (bad curvature)
         ║ ║ ║ ║ ║

         (localized)

    ═══════════════  Inboard (good curvature)
         (weak)
```

### 5.3 Field-Aligned Coordinates

To analyze ballooning modes, use **field-aligned coordinates** $(\psi, \theta, \phi)$ where:
- $\psi$: flux surface label
- $\theta$: poloidal angle (along field)
- $\phi$: toroidal angle

Magnetic field: $\mathbf{B} = \nabla\phi\times\nabla\psi + q(\psi)\nabla\psi\times\nabla\theta$.

### 5.4 Ballooning Equation

The ballooning mode eigenvalue equation in the limit $n \to \infty$:

$$
\frac{d}{d\theta}\left[g(\theta)\frac{d\hat{\Phi}}{d\theta}\right] + h(\theta)\hat{\Phi} = \lambda\hat{\Phi}
$$

where:
- $g(\theta) = |\nabla\psi|^2/B^2$: metric coefficient
- $h(\theta)$: includes pressure gradient and curvature
- $\lambda$: eigenvalue (related to growth rate)
- $\hat{\Phi}$: ballooning amplitude

**Boundary conditions**: $\hat{\Phi}(\theta \pm \infty) = 0$ (localization).

### 5.5 s-α Diagram

Stability is often represented in the **s-α diagram**:

- **Magnetic shear**: $s = (r/q)(dq/dr)$
- **Pressure gradient**: $\alpha = -(2\mu_0 R_0^2 q^2/B_0^2)(dp/dr)$

```
s-α Stability Diagram:
=====================

α |       UNSTABLE
  |      /
  |     /
  |    /
  |   /  ← Stability boundary
  |  /
  | /____STABLE____
  |_________________ s
  0

High shear (large s): stabilizing
High pressure gradient (large α): destabilizing
```

**Approximate stability boundary**:

$$
\alpha_c \approx 0.6s
$$

### 5.6 Connection to ELMs

In tokamaks, ballooning modes are believed to trigger **Edge Localized Modes (ELMs)**:

- High edge pressure gradient in H-mode
- Exceeds ballooning stability boundary
- Periodic eruptions (ELMs) relax pressure gradient
- Concern for ITER: large ELMs can damage first wall

**ELM mitigation strategies**:
- Resonant magnetic perturbations (RMPs)
- Pellet injection
- Vertical kicks

## 6. Mercier Criterion

### 6.1 Local Interchange Stability

The **Mercier criterion** (Mercier, 1960) provides a **necessary condition** for local interchange stability in toroidal geometry.

**Criterion**:

$$
D_I = D_S + D_W + D_G > \frac{1}{4}
$$

where:
- $D_S$: magnetic shear contribution
- $D_W$: magnetic well contribution
- $D_G$: geodesic curvature contribution

### 6.2 Explicit Form

For large-aspect-ratio tokamak:

$$
D_S = \frac{1}{4}\left(\frac{r}{q}\frac{dq}{dr}\right)^2
$$

$$
D_W = \frac{\mu_0 r}{B_p^2}\frac{dp}{dr}\left(1 + 2q^2\right)
$$

$$
D_G \approx \frac{r^2}{R_0 q^2}
$$

**Stability**: $D_I > 1/4$.

### 6.3 Physical Interpretation

- **$D_S$**: Shear stabilizes by decoupling flux surfaces
- **$D_W$**: Negative if pressure gradient is stabilizing (magnetic well)
- **$D_G$**: Geodesic curvature effect (generally stabilizing)

### 6.4 Relation to Suydam Criterion

In cylindrical geometry, Mercier criterion reduces to **Suydam criterion** (Lesson 2):

$$
\frac{r}{4}\left(\frac{q'}{q}\right)^2 + \frac{2\mu_0 p'}{B_z^2} > 0
$$

### 6.5 Limitation

Like Suydam, Mercier criterion is **necessary** but not **sufficient**. It only checks local interchange; global modes require full stability analysis.

## 7. Numerical Simulations

### 7.1 Rayleigh-Taylor Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=0):
    """
    Compute growth rate for MHD Rayleigh-Taylor instability

    Parameters:
    -----------
    k: total wavenumber [1/m]
    g: gravitational acceleration [m/s^2]
    rho1: lower fluid density [kg/m^3]
    rho2: upper fluid density [kg/m^3]
    B0: horizontal magnetic field [T]
    kx_frac: fraction of k along B direction (0 to 1)

    Returns:
    --------
    gamma: growth rate [1/s] (or 0 if stable)
    """
    mu0 = 4*np.pi*1e-7

    # kx is the component of the wavevector along the magnetic field;
    # only this component stretches and bends field lines, so only
    # perturbations with kx ≠ 0 feel the magnetic tension stabilization.
    kx = kx_frac * k

    # The Atwood number A = (ρ₂-ρ₁)/(ρ₂+ρ₁) normalizes the density contrast:
    # A=1 is the limiting case of heavy over vacuum (maximum drive), while
    # A→0 means nearly equal densities (vanishing RT drive).
    A = (rho2 - rho1) / (rho2 + rho1)

    # vA² = B₀²/(μ₀(ρ₁+ρ₂)) uses the sum of densities because both fluid
    # layers contribute to the inertia that the magnetic tension must
    # accelerate; this is the Alfvén speed of the combined system.
    vA2 = B0**2 / (mu0 * (rho1 + rho2))

    # The dispersion relation ω² = -gkA + vA²kx² shows a competition:
    # the -gkA (gravity) term drives instability, while vA²kx² (magnetic
    # tension along B) stabilizes modes with k∥B — but perturbations
    # with k⊥B (kx=0) are unaffected by the field and remain unstable.
    omega_sq = -g * k * A + vA2 * kx**2

    if omega_sq < 0:
        gamma = np.sqrt(-omega_sq)
    else:
        gamma = 0.0  # Stable (oscillatory)

    return gamma

def plot_rt_growth_rate():
    """Plot RT growth rate vs wavenumber and field strength"""

    # Physical parameters
    g = 10  # m/s^2
    rho1 = 1.0  # kg/m^3 (light fluid)
    rho2 = 2.0  # kg/m^3 (heavy fluid)

    # Wavenumber range
    k_vals = np.logspace(-2, 2, 100)  # [1/m]

    # Magnetic field strengths
    B_vals = [0, 0.1, 0.5, 1.0]  # [T]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Growth rate vs k for different B (k perpendicular to B)
    ax = axes[0]
    for B0 in B_vals:
        gamma_vals = [rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=0)
                      for k in k_vals]
        ax.loglog(k_vals, gamma_vals, linewidth=2, label=f'B = {B0} T')

    ax.set_xlabel('Wavenumber k [1/m]', fontsize=12)
    ax.set_ylabel('Growth rate γ [1/s]', fontsize=12)
    ax.set_title('RT Growth Rate vs Wavenumber (k ⊥ B)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    # Panel 2: Growth rate vs angle between k and B
    ax = axes[1]
    k_fixed = 1.0  # Fixed wavenumber
    kx_frac_vals = np.linspace(0, 1, 100)

    for B0 in [0.5, 1.0, 2.0]:
        gamma_vals = [rayleigh_taylor_growth_rate(k_fixed, g, rho1, rho2, B0, kx_frac)
                      for kx_frac in kx_frac_vals]
        ax.plot(kx_frac_vals, gamma_vals, linewidth=2, label=f'B = {B0} T')

    ax.set_xlabel('k_x / k (alignment with B)', fontsize=12)
    ax.set_ylabel('Growth rate γ [1/s]', fontsize=12)
    ax.set_title(f'RT Growth Rate vs Field Alignment (k = {k_fixed} m⁻¹)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig

def example_rt_instability():
    """Example: Rayleigh-Taylor instability analysis"""

    print("=== MHD Rayleigh-Taylor Instability ===\n")

    # Parameters
    g = 10.0
    rho1 = 1.0
    rho2 = 2.0
    k = 1.0

    print(f"Heavy fluid on top: ρ₂ = {rho2} kg/m³")
    print(f"Light fluid below:  ρ₁ = {rho1} kg/m³")
    print(f"Gravity: g = {g} m/s²")
    print(f"Wavenumber: k = {k} m⁻¹")

    # No magnetic field
    gamma_0 = rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0=0, kx_frac=0)
    print(f"\n--- No magnetic field ---")
    print(f"Growth rate: γ = {gamma_0:.3f} s⁻¹")
    print(f"Growth time: τ = {1/gamma_0:.3f} s")

    # With transverse magnetic field (k perpendicular to B)
    B0 = 1.0
    gamma_perp = rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=0)
    print(f"\n--- Magnetic field B = {B0} T (k ⊥ B) ---")
    print(f"Growth rate: γ = {gamma_perp:.3f} s⁻¹")
    print(f"Still unstable!")

    # With field-aligned perturbation
    gamma_par = rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=1.0)
    print(f"\n--- Magnetic field B = {B0} T (k ∥ B) ---")
    if gamma_par > 0:
        print(f"Growth rate: γ = {gamma_par:.3f} s⁻¹")
    else:
        print("STABLE (γ = 0)")

    # Critical wavenumber
    mu0 = 4*np.pi*1e-7
    k_c = g * (rho2 - rho1) * mu0 / B0**2
    print(f"\nCritical wavenumber: k_c = {k_c:.3f} m⁻¹")
    print(f"Modes with k > k_c are stable (if k ∥ B)")

    # Plot
    fig = plot_rt_growth_rate()
    plt.savefig('/tmp/rt_growth_rate.png', dpi=150)
    print("\nGrowth rate plot saved to /tmp/rt_growth_rate.png")
    plt.close()

if __name__ == "__main__":
    example_rt_instability()
```

### 7.2 Ballooning Stability Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def ballooning_stability_boundary(s_vals):
    """
    Approximate ballooning stability boundary in s-α space

    Parameters:
    -----------
    s_vals: array of magnetic shear values

    Returns:
    --------
    alpha_crit: critical alpha for marginal stability
    """
    # The linear approximation α_crit ≈ 0.6s comes from the leading-order
    # expansion of the full ballooning eigenvalue problem in the (s,α) plane:
    # higher shear (s) decouples neighboring flux surfaces and suppresses
    # the flute-like interchange drive, allowing a larger pressure gradient (α)
    # before the mode goes unstable.  The coefficient 0.6 is an empirical fit
    # to numerical solutions of the ballooning equation.
    alpha_crit = 0.6 * s_vals

    return alpha_crit

def compute_alpha_parameter(r, p, q, B0, R0):
    """
    Compute pressure gradient parameter α

    α = -(2μ₀R₀²q²/B₀²)(dp/dr)

    Parameters:
    -----------
    r: minor radius [m]
    p: pressure [Pa]
    q: safety factor
    B0: toroidal field [T]
    R0: major radius [m]

    Returns:
    --------
    alpha: normalized pressure gradient
    """
    mu0 = 4*np.pi*1e-7

    # Centered finite difference gives second-order accuracy in dr without
    # requiring the analytic form of dp/dr, which may not be available
    # for numerically-computed pressure profiles.
    dr = 0.001
    dpdx = (p(r + dr) - p(r - dr)) / (2*dr)

    # α = -(2μ₀R₀²q²/B₀²)(dp/dr) is the dimensionless pressure gradient that
    # appears in the ballooning equation.  The q² factor reflects that higher-q
    # flux surfaces have more toroidal length per poloidal circuit, amplifying
    # the destabilizing curvature-pressure interaction; R₀²/B₀² sets the
    # scale of the curvature drive relative to the restoring tension.
    alpha = -(2*mu0*R0**2*q**2/B0**2) * dpdx

    return alpha

def plot_s_alpha_diagram():
    """Plot s-α stability diagram"""

    # Shear range
    s_vals = np.linspace(0, 5, 100)

    # Stability boundary
    alpha_crit = ballooning_stability_boundary(s_vals)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Fill stable and unstable regions
    ax.fill_between(s_vals, 0, alpha_crit, alpha=0.3, color='green', label='STABLE')
    ax.fill_between(s_vals, alpha_crit, 10, alpha=0.3, color='red', label='UNSTABLE')

    # Boundary
    ax.plot(s_vals, alpha_crit, 'b-', linewidth=3, label='Marginal stability')

    # Example operating points
    examples = [
        (1.0, 0.3, 'Standard H-mode', 'blue'),
        (2.0, 1.5, 'ELMy H-mode', 'red'),
        (3.0, 1.0, 'Improved confinement', 'orange'),
    ]

    for s, alpha, label, color in examples:
        stable = alpha < ballooning_stability_boundary(np.array([s]))[0]
        marker = 'o' if stable else 'x'
        markersize = 10

        ax.plot(s, alpha, marker, color=color, markersize=markersize,
                label=label, markeredgewidth=2)

    ax.set_xlabel('Magnetic shear s = (r/q)(dq/dr)', fontsize=14)
    ax.set_ylabel('Pressure parameter α', fontsize=14)
    ax.set_title('Ballooning Stability Diagram (s-α)', fontsize=16)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    return fig

def analyze_tokamak_ballooning():
    """Analyze ballooning stability for a tokamak equilibrium"""

    # Tokamak parameters
    R0 = 3.0  # Major radius [m]
    a = 1.0   # Minor radius [m]
    B0 = 5.0  # Toroidal field [T]

    # Profiles
    def p_profile(r):
        p0 = 5e5  # Central pressure [Pa]
        return p0 * (1 - (r/a)**2)**2

    def q_profile(r):
        q0 = 1.0
        qa = 4.0
        return q0 + (qa - q0) * (r/a)**2

    # Compute s and α profiles
    r_vals = np.linspace(0.1*a, 0.9*a, 50)

    s_vals = []
    alpha_vals = []

    for r in r_vals:
        q = q_profile(r)

        # Magnetic shear
        dr = 0.001
        dqdx = (q_profile(r + dr) - q_profile(r - dr)) / (2*dr)
        s = (r / q) * dqdx

        # Alpha parameter
        alpha = compute_alpha_parameter(r, p_profile, q, B0, R0)

        s_vals.append(s)
        alpha_vals.append(alpha)

    s_vals = np.array(s_vals)
    alpha_vals = np.array(alpha_vals)

    # Check stability
    alpha_crit_vals = ballooning_stability_boundary(s_vals)
    stable = alpha_vals < alpha_crit_vals

    print("=== Tokamak Ballooning Stability Analysis ===\n")
    print(f"Major radius: R0 = {R0} m")
    print(f"Minor radius: a = {a} m")
    print(f"Toroidal field: B0 = {B0} T")

    # Find most unstable location
    margin = alpha_vals - alpha_crit_vals
    idx_worst = np.argmax(margin)

    print(f"\nMost unstable location:")
    print(f"  r/a = {r_vals[idx_worst]/a:.2f}")
    print(f"  s = {s_vals[idx_worst]:.2f}")
    print(f"  α = {alpha_vals[idx_worst]:.2f}")
    print(f"  α_crit = {alpha_crit_vals[idx_worst]:.2f}")
    print(f"  Margin: {margin[idx_worst]:+.2f}")

    if margin[idx_worst] > 0:
        print("  Status: UNSTABLE to ballooning modes")
    else:
        print("  Status: STABLE")

    # Plot profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pressure
    ax = axes[0, 0]
    r_plot = np.linspace(0, a, 100)
    p_plot = [p_profile(ri) for ri in r_plot]
    ax.plot(r_plot/a, np.array(p_plot)/1e3, 'b-', linewidth=2)
    ax.set_xlabel('r/a')
    ax.set_ylabel('Pressure [kPa]')
    ax.set_title('Pressure Profile')
    ax.grid(True, alpha=0.3)

    # Safety factor
    ax = axes[0, 1]
    q_plot = [q_profile(ri) for ri in r_plot]
    ax.plot(r_plot/a, q_plot, 'g-', linewidth=2)
    ax.set_xlabel('r/a')
    ax.set_ylabel('q')
    ax.set_title('Safety Factor Profile')
    ax.grid(True, alpha=0.3)

    # s-α trajectory
    ax = axes[1, 0]
    s_boundary = np.linspace(0, max(s_vals)*1.2, 100)
    alpha_boundary = ballooning_stability_boundary(s_boundary)

    ax.fill_between(s_boundary, 0, alpha_boundary, alpha=0.2, color='green')
    ax.fill_between(s_boundary, alpha_boundary, max(alpha_vals)*1.2,
                    alpha=0.2, color='red')

    ax.plot(s_vals, alpha_vals, 'bo-', linewidth=2, markersize=4,
            label='Equilibrium trajectory')
    ax.plot(s_boundary, alpha_boundary, 'k--', linewidth=2,
            label='Stability boundary')

    # Mark unstable region
    unstable_mask = ~stable
    if np.any(unstable_mask):
        ax.plot(s_vals[unstable_mask], alpha_vals[unstable_mask], 'ro',
                markersize=6, label='Unstable locations')

    ax.set_xlabel('s')
    ax.set_ylabel('α')
    ax.set_title('s-α Stability Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stability margin
    ax = axes[1, 1]
    ax.plot(r_vals/a, margin, 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.fill_between(r_vals/a, 0, margin, where=(margin>0), alpha=0.3,
                    color='red', label='Unstable')
    ax.fill_between(r_vals/a, margin, 0, where=(margin<=0), alpha=0.3,
                    color='green', label='Stable')

    ax.set_xlabel('r/a')
    ax.set_ylabel('α - α_crit')
    ax.set_title('Ballooning Stability Margin')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ballooning_analysis.png', dpi=150)
    print("\nBallooning analysis plot saved to /tmp/ballooning_analysis.png")
    plt.close()

    # Also create s-α diagram
    fig2 = plot_s_alpha_diagram()
    plt.savefig('/tmp/s_alpha_diagram.png', dpi=150)
    print("s-α diagram saved to /tmp/s_alpha_diagram.png")
    plt.close()

if __name__ == "__main__":
    analyze_tokamak_ballooning()
```

### 7.3 Mercier Criterion Evaluation

```python
import numpy as np
import matplotlib.pyplot as plt

def evaluate_mercier_criterion(r, q, p, Bp, R0):
    """
    Evaluate Mercier criterion: D_I > 1/4

    Parameters:
    -----------
    r: minor radius [m]
    q: safety factor
    p: pressure [Pa]
    Bp: poloidal field [T]
    R0: major radius [m]

    Returns:
    --------
    D_I: Mercier stability parameter
    stable: True if stable, False if unstable
    """
    mu0 = 4*np.pi*1e-7

    # Compute derivatives numerically
    dr = 0.001

    # D_S = (1/4)(r q'/q)² is always ≥ 0: magnetic shear (q'/q) stabilizes
    # interchange by decoupling adjacent flux surfaces, preventing them from
    # moving coherently — the 1/4 prefactor is the Mercier coefficient that
    # quantifies the minimum shear needed to overcome the pressure drive.
    dqdx = (q(r + dr) - q(r - dr)) / (2*dr)
    D_S = 0.25 * ((r / q(r)) * dqdx)**2

    # D_W captures the magnetic well: when dp/dr < 0 (pressure decreasing
    # outward) and B_p is finite, D_W is negative, indicating a destabilizing
    # interchange drive; the (1 + 2q²) factor accounts for both the poloidal
    # and toroidal parts of the curvature.
    dpdx = (p(r + dr) - p(r - dr)) / (2*dr)
    D_W = (mu0 * r / Bp(r)**2) * dpdx * (1 + 2*q(r)**2)

    # D_G = r²/(R₀²q²) is the geodesic curvature correction: the non-circular
    # poloidal trajectory of field lines in a torus adds an extra stabilizing
    # curvature component that is absent in a straight cylinder.
    D_G = r**2 / (R0**2 * q(r)**2)

    # Total
    D_I = D_S + D_W + D_G

    # The threshold 1/4 is the Mercier stability criterion: D_I > 1/4 means
    # the combined shear + geodesic stabilization exceeds the pressure drive.
    # It is a necessary (not sufficient) condition — passing D_I > 1/4 does
    # not rule out global modes that are not captured by local analysis.
    stable = D_I > 0.25

    return D_I, D_S, D_W, D_G, stable

def plot_mercier_stability():
    """Analyze Mercier stability for a tokamak"""

    # Parameters
    R0 = 3.0
    a = 1.0
    p0 = 5e5
    Bp0 = 0.5

    # Profiles
    def q_func(r):
        return 1.0 + 3.0*(r/a)**2

    def p_func(r):
        return p0 * (1 - (r/a)**2)**2

    def Bp_func(r):
        # Approximate: Bp ~ r for parabolic current
        return Bp0 * (r/a) if r > 0 else 1e-6

    # Evaluate over radius
    r_vals = np.linspace(0.1*a, 0.95*a, 100)

    D_I_vals = []
    D_S_vals = []
    D_W_vals = []
    D_G_vals = []
    stable_vals = []

    for r in r_vals:
        D_I, D_S, D_W, D_G, stable = evaluate_mercier_criterion(
            r, q_func, p_func, Bp_func, R0)

        D_I_vals.append(D_I)
        D_S_vals.append(D_S)
        D_W_vals.append(D_W)
        D_G_vals.append(D_G)
        stable_vals.append(stable)

    D_I_vals = np.array(D_I_vals)
    D_S_vals = np.array(D_S_vals)
    D_W_vals = np.array(D_W_vals)
    D_G_vals = np.array(D_G_vals)
    stable_vals = np.array(stable_vals)

    print("=== Mercier Criterion Analysis ===\n")
    print(f"Major radius: R0 = {R0} m")
    print(f"Minor radius: a = {a} m")

    # Check if any location is Mercier unstable
    if np.all(stable_vals):
        print("\n✓ STABLE: Mercier criterion satisfied at all radii")
    else:
        print("\n✗ UNSTABLE: Mercier criterion violated!")
        unstable_r = r_vals[~stable_vals]
        print(f"  Unstable region: r/a ∈ [{unstable_r[0]/a:.2f}, {unstable_r[-1]/a:.2f}]")

    # Find minimum D_I
    idx_min = np.argmin(D_I_vals)
    print(f"\nMost dangerous location:")
    print(f"  r/a = {r_vals[idx_min]/a:.2f}")
    print(f"  D_I = {D_I_vals[idx_min]:.4f} (critical: 0.25)")
    print(f"  Margin: {D_I_vals[idx_min] - 0.25:+.4f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Individual contributions
    ax = axes[0]
    ax.plot(r_vals/a, D_S_vals, 'b-', linewidth=2, label='D_S (shear)')
    ax.plot(r_vals/a, D_W_vals, 'r-', linewidth=2, label='D_W (well)')
    ax.plot(r_vals/a, D_G_vals, 'g-', linewidth=2, label='D_G (geodesic)')
    ax.plot(r_vals/a, D_I_vals, 'k-', linewidth=3, label='D_I (total)')

    ax.set_xlabel('r/a', fontsize=12)
    ax.set_ylabel('Mercier Contributions', fontsize=12)
    ax.set_title('Mercier Stability Parameter Components', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Stability margin
    ax = axes[1]
    margin = D_I_vals - 0.25

    ax.plot(r_vals/a, margin, 'b-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.fill_between(r_vals/a, 0, margin, where=(margin>0), alpha=0.3,
                    color='green', label='Stable (D_I > 0.25)')
    ax.fill_between(r_vals/a, margin, 0, where=(margin<=0), alpha=0.3,
                    color='red', label='Unstable (D_I < 0.25)')

    ax.set_xlabel('r/a', fontsize=12)
    ax.set_ylabel('D_I - 0.25', fontsize=12)
    ax.set_title('Mercier Stability Margin', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/mercier_stability.png', dpi=150)
    print("\nMercier stability plot saved to /tmp/mercier_stability.png")
    plt.close()

if __name__ == "__main__":
    plot_mercier_stability()
```

## Summary

In this lesson, we have explored pressure-driven MHD instabilities:

1. **Interchange Instability**: Driven by unfavorable curvature ($\boldsymbol{\kappa}\cdot\nabla p > 0$). Stabilized by magnetic shear and favorable average curvature.

2. **Rayleigh-Taylor Instability**: Heavy fluid on top of light fluid. Magnetic field stabilizes short wavelengths along field direction but not perpendicular.

3. **Parker Instability**: Magnetic buoyancy in stratified atmosphere. Unstable when $\beta > 2/\gamma \approx 1.2$. Important for astrophysical plasmas.

4. **Ballooning Modes**: High-$n$ modes localized on bad curvature side in toroidal geometry. Stability boundary in s-α space: $\alpha_c \approx 0.6s$. Connected to ELMs in tokamaks.

5. **Mercier Criterion**: Necessary condition for local interchange stability: $D_I = D_S + D_W + D_G > 1/4$. Combines shear, magnetic well, and geodesic curvature.

6. **Numerical Tools**: Implementations for RT growth rates, ballooning stability diagrams, and Mercier criterion evaluation.

These instabilities limit achievable plasma pressure (beta limit) and drive the need for careful equilibrium design, shaping, and active control in fusion devices.

## Practice Problems

### Problem 1: Interchange Stability in a Mirror

A magnetic mirror has field strength $B(z) = B_0(1 + z^2/L^2)$ and uniform pressure $p = p_0$.

**(a)** Compute the curvature $\boldsymbol{\kappa} = \mathbf{b}\cdot\nabla\mathbf{b}$ where $\mathbf{b} = \mathbf{B}/B$.

**(b)** Evaluate $\boldsymbol{\kappa}\cdot\nabla p$.

**(c)** Is this configuration stable or unstable to interchange?

**(d)** How would adding a pressure gradient $p(z) = p_0 e^{-z^2/L_p^2}$ affect stability?

### Problem 2: RT Critical Wavelength

Plasma with density $\rho_2 = 10^{-6}$ kg/m³ is supported above vacuum ($\rho_1 \approx 0$) by a horizontal magnetic field $B_0 = 1$ T in effective gravity $g_{eff} = 10^4$ m/s² (centrifugal acceleration).

**(a)** Compute the critical wavenumber $k_c$ for RT stability.

**(b)** Convert to critical wavelength $\lambda_c = 2\pi/k_c$.

**(c)** For $k = 0.5k_c$, compute the growth rate $\gamma$.

**(d)** Estimate the growth time $\tau = 1/\gamma$.

**(e)** Compare $\tau$ to the Alfvén time $\tau_A = \lambda_c/v_A$.

### Problem 3: Ballooning Stability Boundary

A tokamak has magnetic shear $s = 2.5$ at mid-radius.

**(a)** Using the approximate formula $\alpha_c \approx 0.6s$, compute the critical pressure parameter.

**(b)** If the actual $\alpha = 2.0$, is the plasma stable or unstable to ballooning?

**(c)** By what factor must the pressure gradient be reduced to achieve marginal stability?

**(d)** If instead $s$ is increased to 4.0 (while keeping $\alpha=2.0$), does the plasma become stable?

**(e)** Discuss two experimental methods to increase $s$ in a tokamak.

### Problem 4: Parker Instability in Galactic Disk

A galactic disk has:
- Gas density $\rho_0 = 10^{-24}$ kg/m³
- Temperature $T = 10^4$ K
- Magnetic field $B = 5\times 10^{-10}$ T
- Scale height $H = 100$ pc = $3\times 10^{18}$ m

**(a)** Compute the gas pressure $p = nkT$ (assume $n = \rho_0/m_p$).

**(b)** Calculate plasma beta $\beta = 2\mu_0 p/B^2$.

**(c)** Check the Parker instability condition $\beta > 2/\gamma$ (use $\gamma = 5/3$).

**(d)** If unstable, estimate the growth rate $\gamma \sim \sqrt{g/H}$ where $g \sim kT/(m_p H)$.

**(e)** Convert growth time to years. Is this consistent with observed timescales for molecular cloud formation?

### Problem 5: Mercier Criterion Components

A tokamak has at $r = 0.5a$:
- Safety factor $q = 1.5$
- Magnetic shear $(r/q)(dq/dr) = 1.0$
- Pressure $p = 2\times 10^5$ Pa
- Pressure gradient $dp/dr = -10^6$ Pa/m
- Poloidal field $B_p = 0.4$ T
- Major radius $R_0 = 3$ m

**(a)** Compute the shear contribution $D_S = \frac{1}{4}\left(\frac{r}{q}\frac{dq}{dr}\right)^2$.

**(b)** Compute the magnetic well contribution $D_W = \frac{\mu_0 r}{B_p^2}\frac{dp}{dr}(1 + 2q^2)$.

**(c)** Compute the geodesic curvature $D_G = \frac{r^2}{R_0^2 q^2}$.

**(d)** Evaluate $D_I = D_S + D_W + D_G$.

**(e)** Check if $D_I > 0.25$ (Mercier stable).

**(f)** Which term contributes most to stability/instability?

---

**Previous**: [Linear Stability](./02_Linear_Stability.md) | **Next**: [Current-Driven Instabilities](./04_Current_Driven_Instabilities.md)

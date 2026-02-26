# 1. MHD Equilibria

## Learning Objectives

- Derive the MHD force balance equation from the ideal MHD momentum equation
- Understand the consequences of force balance: pressure constant on flux surfaces
- Analyze 1D equilibria: θ-pinch, Z-pinch, and screw pinch configurations
- Formulate and solve the Grad-Shafranov equation for axisymmetric equilibria
- Calculate the safety factor q and understand its significance for stability
- Compute plasma beta and understand operational limits
- Implement numerical solutions for simple equilibrium configurations

## 1. Introduction to MHD Equilibria

Magnetohydrodynamic equilibria describe the stationary configurations of a magnetized plasma where all forces are balanced and no net acceleration occurs. Understanding equilibria is fundamental to fusion energy research, astrophysical plasma physics, and any application involving confined plasmas.

An MHD equilibrium satisfies:

```
∂/∂t = 0  (time-independent)
v = 0      (static plasma)
```

The equilibrium state is governed by a balance between:
- **Plasma pressure gradient force**: ∇p (pushes outward)
- **Magnetic tension**: (B·∇)B/μ₀ (pulls along field lines)
- **Magnetic pressure gradient**: -∇(B²/2μ₀) (pushes from high to low field)

The force balance equation is the foundation of all equilibrium calculations.

## 2. Derivation of Force Balance Equation

### 2.1 Starting from Ideal MHD Momentum Equation

The ideal MHD momentum equation is:

$$
\rho\frac{D\mathbf{v}}{Dt} = -\nabla p + \mathbf{J}\times\mathbf{B}
$$

For equilibrium, the left-hand side vanishes (no acceleration, static plasma):

$$
\nabla p = \mathbf{J}\times\mathbf{B}
$$

This is the **fundamental MHD equilibrium equation**.

### 2.2 Expressing in Terms of Magnetic Field

Using Ampère's law (neglecting displacement current):

$$
\mathbf{J} = \frac{1}{\mu_0}\nabla\times\mathbf{B}
$$

The force balance becomes:

$$
\nabla p = \frac{1}{\mu_0}(\nabla\times\mathbf{B})\times\mathbf{B}
$$

Using the vector identity:

$$
(\nabla\times\mathbf{B})\times\mathbf{B} = (\mathbf{B}\cdot\nabla)\mathbf{B} - \nabla\left(\frac{B^2}{2}\right)
$$

We obtain:

$$
\nabla p = \frac{1}{\mu_0}\left[(\mathbf{B}\cdot\nabla)\mathbf{B} - \nabla\left(\frac{B^2}{2}\right)\right]
$$

Rearranging:

$$
\nabla\left(p + \frac{B^2}{2\mu_0}\right) = \frac{1}{\mu_0}(\mathbf{B}\cdot\nabla)\mathbf{B}
$$

The left side is the gradient of **total pressure** (kinetic + magnetic):

$$
p_{total} = p + \frac{B^2}{2\mu_0}
$$

The right side represents **magnetic tension** along field lines.

### 2.3 Physical Interpretation

```
Force Balance Components:
=======================

1. Pressure gradient: ∇p
   - Pushes from high to low pressure
   - Isotropic force

2. Magnetic pressure: -∇(B²/2μ₀)
   - Pushes from high to low field
   - Acts perpendicular to B

3. Magnetic tension: (B·∇)B/μ₀
   - Pulls along curved field lines
   - Like a stretched rubber band

Net force: ∇p + ∇(B²/2μ₀) - (B·∇)B/μ₀ = 0
```

## 3. Consequences of Force Balance

### 3.1 Pressure Constant on Flux Surfaces

Taking the dot product of the force balance equation with **B**:

$$
\mathbf{B}\cdot\nabla p = \mathbf{B}\cdot(\mathbf{J}\times\mathbf{B}) = 0
$$

This implies:

$$
\mathbf{B}\cdot\nabla p = 0
$$

**Consequence**: Pressure is constant along magnetic field lines. In toroidal configurations, field lines lie on nested flux surfaces, so **pressure is constant on each flux surface**.

### 3.2 Current Perpendicular to Pressure Gradient

Taking the dot product of the force balance equation with **J**:

$$
\mathbf{J}\cdot\nabla p = \mathbf{J}\cdot(\mathbf{J}\times\mathbf{B}) = 0
$$

This implies:

$$
\mathbf{J}\cdot\nabla p = 0
$$

**Consequence**: Current flows perpendicular to the pressure gradient, hence current also lies on flux surfaces.

### 3.3 Flux Surface Coordinates

The above two results mean that both **B** and **J** lie on surfaces of constant pressure. These surfaces are called **magnetic flux surfaces**. This is the foundation of flux surface coordinates used in tokamak equilibrium calculations.

```
Flux Surface Structure:
======================

        ┌─────────────┐
        │             │   Outer flux surface (low p)
        │  ┌───────┐  │
        │  │       │  │   Middle flux surface
        │  │   ┌─┐ │  │
        │  │   │·│ │  │   Magnetic axis (peak p)
        │  │   └─┘ │  │
        │  │       │  │
        │  └───────┘  │
        │             │
        └─────────────┘

Properties:
- p = p(ψ) where ψ labels the flux surface
- B lies in flux surfaces
- J lies in flux surfaces
```

## 4. One-Dimensional Equilibria

### 4.1 θ-Pinch (Longitudinal Magnetic Field)

The θ-pinch uses a purely longitudinal (axial) magnetic field to confine plasma radially.

**Configuration**:
- Cylindrical geometry (r, θ, z)
- $B_z(r)$, $p(r)$
- $B_r = B_θ = 0$
- No current in plasma: $J_z = 0$

**Force Balance**:

In cylindrical coordinates:

$$
\frac{dp}{dr} = -\frac{1}{\mu_0}\frac{d}{dr}\left(\frac{B_z^2}{2}\right)
$$

Integrating from $r$ to the plasma edge at $r=a$ (where $p(a)=0$):

$$
p(r) = \frac{B_z^2(a) - B_z^2(r)}{2\mu_0}
$$

**Physical Picture**:
- Plasma pressure balanced by magnetic pressure
- No magnetic tension (straight field lines)
- Pure radial confinement

**Bennett Relation** (total pressure balance):

Integrating over the cross-section:

$$
\int_0^a 2\pi r\, p(r)\, dr = \frac{\pi a^2 B_{ext}^2}{2\mu_0}
$$

where $B_{ext}$ is the external field.

### 4.2 Z-Pinch (Azimuthal Magnetic Field)

The Z-pinch uses a self-generated azimuthal magnetic field from an axial current.

**Configuration**:
- $B_θ(r)$, $p(r)$
- $J_z(r)$ (axial current)
- $B_r = B_z = 0$

**Ampère's Law**:

$$
B_θ(r) = \frac{\mu_0}{2\pi r}\int_0^r J_z(r')2\pi r'\, dr' = \frac{\mu_0 I(r)}{2\pi r}
$$

where $I(r)$ is the current enclosed within radius $r$.

**Force Balance**:

$$
\frac{dp}{dr} = -\frac{1}{\mu_0}\frac{d}{dr}\left(\frac{B_θ^2}{2}\right) = -\frac{B_θ}{\mu_0 r}
$$

Using $B_θ = \mu_0 I/(2\pi r)$:

$$
\frac{dp}{dr} = -\frac{J_z B_θ}{\mu_0}
$$

**Bennett Relation** (dimensional analysis):

For a uniform current density $J_z = I/(\pi a^2)$:

$$
I^2 = \frac{8\pi}{\mu_0}NkT
$$

where $N$ is the total number of particles and $T$ is temperature.

**Physical Picture**:
- Current generates azimuthal field
- Magnetic pressure pinches plasma inward
- Plasma pressure pushes outward
- Highly unstable (kink, sausage instabilities)

### 4.3 Screw Pinch (Combined Fields)

The screw pinch combines axial and azimuthal fields for improved stability.

**Configuration**:
- $B_z(r)$, $B_θ(r)$, $p(r)$
- $J_z(r)$, $J_θ(r)$

**Force Balance**:

$$
\frac{dp}{dr} = -\frac{1}{\mu_0}\frac{d}{dr}\left(\frac{B_z^2 + B_θ^2}{2}\right)
$$

**Safety Factor** (pitch of field lines):

$$
q(r) = \frac{rB_z}{RB_θ}
$$

where $R$ is the major radius (for toroidal systems) or a characteristic length.

**Physical Picture**:
- $B_z$ provides stability against kink modes
- $B_θ$ provides confinement
- Shear in $q(r)$ stabilizes short-wavelength modes
- Basis for tokamak confinement

```
Screw Pinch Field Line:
======================

      z ^
        |    /
        |   / ← Field line spirals
        |  /
        | /
        |/________> θ

q = Δz / (2πR) per poloidal turn
```

## 5. Grad-Shafranov Equation

### 5.1 Axisymmetric Equilibria

For axisymmetric toroidal systems (tokamaks, stellarators in simplified form), the equilibrium can be described by a single scalar function: the **poloidal flux function** $\psi(R,Z)$.

**Cylindrical Coordinates**: $(R, \phi, Z)$ where $\phi$ is the toroidal angle.

**Magnetic Field Representation**:

$$
\mathbf{B} = F(R,Z)\nabla\phi + \nabla\phi\times\nabla\psi
$$

where:
- $F(R,Z) = RB_\phi$ (toroidal field function)
- $\psi$ is the poloidal flux function

In component form:

$$
B_R = \frac{1}{R}\frac{\partial\psi}{\partial Z}, \quad B_Z = -\frac{1}{R}\frac{\partial\psi}{\partial R}, \quad B_\phi = \frac{F}{R}
$$

### 5.2 Derivation of Grad-Shafranov Equation

Starting from the force balance:

$$
\nabla p = \mathbf{J}\times\mathbf{B}
$$

and using $\mathbf{J} = \nabla\times\mathbf{B}/\mu_0$, the toroidal component gives:

$$
\frac{dp}{d\psi} = -\frac{1}{\mu_0 R^2}\frac{dF}{d\psi}F
$$

The poloidal component yields the **Grad-Shafranov equation**:

$$
\Delta^*\psi \equiv R\frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right) + \frac{\partial^2\psi}{\partial Z^2} = -\mu_0 R^2\frac{dp}{d\psi} - F\frac{dF}{d\psi}
$$

This is an **elliptic partial differential equation** for $\psi(R,Z)$.

### 5.3 Free Functions

The Grad-Shafranov equation requires two **free functions** to be specified:

1. **Pressure profile**: $p(\psi)$
2. **Toroidal field function**: $F(\psi)$ (or equivalently $F^2(\psi)$)

These functions are determined by:
- Plasma heating and current drive
- Boundary conditions (conducting walls, external coils)
- Transport processes (beyond MHD)

Common choices for analytic solutions:
- $p(\psi) = p_0(1 - \psi/\psi_0)^\alpha$
- $F^2(\psi) = F_0^2 + \beta(\psi - \psi_0)$

### 5.4 Solovev Equilibrium (Analytical Solution)

For the free functions:

$$
p(\psi) = 0, \quad F^2(\psi) = F_0^2 + c\psi
$$

the Grad-Shafranov equation becomes linear:

$$
\Delta^*\psi = -\mu_0 c R^2
$$

**Solovev solution** (for circular cross-section):

$$
\psi(R,Z) = \frac{1}{8}c\mu_0\left[(R^2 - R_0^2)^2 + Z^2\right] + \psi_0
$$

This represents circular flux surfaces shifted outward by the Shafranov shift.

### 5.5 Numerical Solution Methods

For general $p(\psi)$ and $F(\psi)$, the Grad-Shafranov equation must be solved numerically:

**Iterative Scheme**:
1. Guess initial $\psi^{(0)}(R,Z)$
2. Compute $p(\psi^{(n)})$ and $F(\psi^{(n)})$
3. Solve linear elliptic equation:
   $$
   \Delta^*\psi^{(n+1)} = -\mu_0 R^2\frac{dp}{d\psi}\Big|_{\psi^{(n)}} - F\frac{dF}{d\psi}\Big|_{\psi^{(n)}}
   $$
4. Iterate until convergence: $|\psi^{(n+1)} - \psi^{(n)}| < \epsilon$

**Discretization**: Finite difference or finite element methods on $(R,Z)$ grid.

## 6. Safety Factor

### 6.1 Definition

The **safety factor** $q(\psi)$ measures the pitch of magnetic field lines on a flux surface:

$$
q = \frac{1}{2\pi}\oint\frac{d\ell_\parallel}{Rd\phi}
$$

where the integral is over one poloidal turn.

**Physical Interpretation**:
- $q$ is the number of toroidal turns a field line makes per poloidal turn
- Rational values ($q = m/n$) define **resonant surfaces** where perturbations can resonate

### 6.2 Cylindrical Approximation

For a large-aspect-ratio tokamak ($R \approx R_0 + r\cos\theta$):

$$
q(r) \approx \frac{rB_z}{R_0 B_θ}
$$

Using $B_θ = \mu_0 I(r)/(2\pi r)$:

$$
q(r) = \frac{2\pi r^2 B_z}{\mu_0 R_0 I(r)}
$$

### 6.3 Relation to Current Density

Differentiating:

$$
\frac{dq}{dr} = \frac{2\pi r B_z}{\mu_0 R_0}\left(\frac{2}{I} - \frac{r J_z}{I}\right)
$$

**Magnetic Shear**:

$$
s = \frac{r}{q}\frac{dq}{dr}
$$

Positive shear ($dq/dr > 0$) is typically stabilizing.

### 6.4 Significance for Stability

- **Kruskal-Shafranov criterion**: External kink modes require $q(a) > m/n$
- For $m=1, n=1$: $q(a) > 1$ is necessary (but not sufficient)
- **Internal kink** (sawtooth oscillations): $q(0) < 1$
- **Rational surfaces** $q = m/n$: sites of tearing modes

```
Typical q-profile in Tokamak:
============================

q |     ___________________  q(a) > 2
  |    /
  |   /                     ← monotonic (positive shear)
  |  /
  | /                       q(0) ~ 1
  |/________________________ r
  0                        a

Features:
- q(0) ~ 1 (on-axis)
- q(a) > 2-3 (edge, for stability)
- Reversed shear: dq/dr < 0 in core (advanced scenarios)
```

## 7. Flux Surfaces and Shafranov Shift

### 7.1 Nested Flux Surfaces

In a well-confined plasma, flux surfaces are nested topological tori. The magnetic axis (innermost surface) is the surface with:
- Highest pressure
- $q(\psi_{axis})$ minimum
- $|\nabla\psi| = 0$

### 7.2 Shafranov Shift

Due to toroidal effects, the magnetic axis is shifted **outward** (larger $R$) relative to the geometric center.

**Physical Origin**:
- Higher field on inboard side → higher magnetic pressure
- Plasma pressure gradient creates outward shift to balance

**Approximate Formula**:

$$
\Delta_{Shafranov} \approx \frac{a^2\beta_p}{2R_0}
$$

where $\beta_p = 2\mu_0\langle p\rangle/B_p^2$ is the poloidal beta.

### 7.3 Flux Surface Shaping

Modern tokamaks use non-circular cross-sections:
- **Elongation** $\kappa = b/a$ (vertical stretching): improves stability and confinement
- **Triangularity** $\delta$: indentation of inboard side, stabilizes ballooning modes

## 8. Plasma Beta

### 8.1 Definition

**Plasma beta** is the ratio of plasma pressure to magnetic pressure:

$$
\beta = \frac{2\mu_0 p}{B^2}
$$

**Volume-averaged beta**:

$$
\langle\beta\rangle = \frac{2\mu_0\langle p\rangle}{\langle B^2\rangle}
$$

**Poloidal beta**:

$$
\beta_p = \frac{2\mu_0\langle p\rangle}{B_p^2}
$$

**Toroidal beta**:

$$
\beta_t = \frac{2\mu_0\langle p\rangle}{B_t^2}
$$

### 8.2 Relation Between Betas

For large-aspect-ratio tokamak:

$$
\beta_t \approx \beta_p\left(\frac{B_p}{B_t}\right)^2 \approx \beta_p\left(\frac{a}{R_0}\right)^2\frac{1}{q^2}
$$

### 8.3 Beta Limits

High beta is desirable for fusion reactors (more fusion power), but MHD stability limits $\beta$:

**Troyon limit** (empirical scaling):

$$
\beta_N \equiv \frac{\beta_t(\%)}{I_p/(aB_t)} \lesssim 2.8 - 3.5
$$

where $I_p$ is plasma current (MA), $a$ in meters, $B_t$ in Tesla.

**Physical origin**:
- High $\beta$ → strong pressure gradient → pressure-driven instabilities
- External kink modes limit $\beta$ at fixed $q(a)$

### 8.4 Beta Optimization

Strategies to increase beta limit:
- High elongation $\kappa$
- High triangularity $\delta$
- Conducting wall stabilization
- Plasma rotation
- Advanced tokamak scenarios (reversed shear, transport barriers)

## 9. Python Implementation: Grad-Shafranov Solver

### 9.1 Simple Solovev Equilibrium

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Solovev equilibrium solver
class SolovevEquilibrium:
    def __init__(self, R0, a, kappa, delta, Bt0, Ip):
        """
        R0: major radius [m]
        a: minor radius [m]
        kappa: elongation
        delta: triangularity
        Bt0: toroidal field on axis [T]
        Ip: plasma current [A]
        """
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.delta = delta
        self.Bt0 = Bt0
        self.Ip = Ip

        # Physical constants
        self.mu0 = 4*np.pi*1e-7

    def compute_psi(self, R, Z):
        """Compute poloidal flux function (Solovev solution)"""
        # Solovev solution linearizes the GS equation by choosing p(ψ)=0 and
        # F²(ψ) = F₀² + cψ (linear in ψ), which makes the GS equation itself
        # linear and therefore admits an exact polynomial solution in (R,Z).
        # This is the simplest non-trivial analytic equilibrium: valid when
        # β ≪ 1 and the cross-section is nearly circular (hence "low beta").
        c = -2 * self.mu0 * self.Ip / (np.pi * self.a**2)

        # Dividing by κ in Z scales the elliptical cross-section back to
        # circular, so a single radial coordinate r_norm correctly labels
        # flux surfaces for elongated plasmas under the Solovev assumption.
        r_norm = np.sqrt((R - self.R0)**2 + (Z/self.kappa)**2) / self.a

        # The quadratic dependence on r_norm reflects that the Solovev ψ is
        # a second-order polynomial; higher-order terms would require a more
        # general (non-linear) free-function prescription.
        psi = -c * self.R0**2 * self.a**2 * r_norm**2 / 8

        return psi

    def compute_B(self, R, Z):
        """Compute magnetic field components"""
        # Centered finite differences are used for ∂ψ/∂R and ∂ψ/∂Z because
        # they achieve second-order accuracy with only two function evaluations,
        # which is sufficient given the smooth Solovev flux function.
        dR = 0.001
        dZ = 0.001

        dpsi_dR = (self.compute_psi(R+dR, Z) - self.compute_psi(R-dR, Z)) / (2*dR)
        dpsi_dZ = (self.compute_psi(R, Z+dZ) - self.compute_psi(R, Z-dZ)) / (2*dZ)

        # The 1/R factors arise from the cylindrical definition B_R = -(1/R)∂ψ/∂Z
        # and B_Z = (1/R)∂ψ/∂R — they ensure ∇·B = 0 in toroidal geometry.
        BR = -1/R * dpsi_dZ
        BZ = 1/R * dpsi_dR
        # B_φ ∝ 1/R follows from Ampère's law applied to the toroidal current-
        # free vacuum region; Bt0*R0 is the constant F = R*B_φ on axis.
        Bphi = self.Bt0 * self.R0 / R

        return BR, BZ, Bphi

    def compute_q(self, psi_vals):
        """Compute safety factor profile"""
        # ψ_edge gives the flux label of the Last Closed Flux Surface (LCFS);
        # normalizing by it maps ψ onto [0,1] so that ψ_norm = 0 at the axis
        # and ψ_norm = 1 at the plasma boundary, independent of current level.
        psi_edge = self.compute_psi(self.R0 + self.a, 0)
        psi_norm = psi_vals / psi_edge

        # A parabolic q(ψ) is the simplest profile consistent with a peaked
        # current density on axis; in more accurate equilibria q(ψ) is found
        # self-consistently from the current profile, but this suffices for
        # illustrating the safety-factor structure without a full GS solve.
        q0 = 1.0  # On-axis q
        qa = 3.0  # Edge q

        q = q0 + (qa - q0) * psi_norm**2

        return q

    def plot_flux_surfaces(self, nr=50, nz=50):
        """Plot flux surfaces"""
        R_grid = np.linspace(self.R0 - 1.2*self.a, self.R0 + 1.2*self.a, nr)
        Z_grid = np.linspace(-1.2*self.kappa*self.a, 1.2*self.kappa*self.a, nz)

        R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid)
        psi_mesh = self.compute_psi(R_mesh, Z_mesh)

        fig, ax = plt.subplots(figsize=(8, 10))

        # Contour plot of flux surfaces
        levels = 20
        CS = ax.contour(R_mesh, Z_mesh, psi_mesh, levels=levels, colors='blue')
        ax.clabel(CS, inline=True, fontsize=8)

        # Mark magnetic axis
        ax.plot(self.R0, 0, 'r*', markersize=15, label='Magnetic axis')

        # Mark last closed flux surface
        theta = np.linspace(0, 2*np.pi, 100)
        R_lcfs = self.R0 + self.a*np.cos(theta + self.delta*np.sin(theta))
        Z_lcfs = self.kappa*self.a*np.sin(theta)
        ax.plot(R_lcfs, Z_lcfs, 'r--', linewidth=2, label='LCFS')

        ax.set_xlabel('R [m]', fontsize=12)
        ax.set_ylabel('Z [m]', fontsize=12)
        ax.set_title('Flux Surfaces (Solovev Equilibrium)', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_q_profile(self):
        """Plot safety factor profile"""
        # Radial coordinate (minor radius)
        r = np.linspace(0, self.a, 100)
        R_vals = self.R0 + r
        Z_vals = np.zeros_like(r)

        # Compute psi along midplane
        psi_vals = np.array([self.compute_psi(R, 0) for R in R_vals])

        # Compute q
        q_vals = self.compute_q(psi_vals)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(r/self.a, q_vals, 'b-', linewidth=2)
        ax.axhline(y=1, color='r', linestyle='--', label='q = 1 (sawtooth)')
        ax.axhline(y=2, color='g', linestyle='--', label='q = 2 (m=2 resonance)')

        ax.set_xlabel('r/a (normalized radius)', fontsize=12)
        ax.set_ylabel('q (safety factor)', fontsize=12)
        ax.set_title('Safety Factor Profile', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_pressure_profile(self):
        """Plot pressure and current density profiles"""
        r = np.linspace(0, self.a, 100)

        # Parabolic pressure profile
        p0 = 1e5  # Central pressure [Pa]
        p = p0 * (1 - (r/self.a)**2)**2

        # Current density (from force balance)
        # J_phi ~ dp/dr (simplified)
        dp_dr = np.gradient(p, r)
        j_phi = -dp_dr / (self.Bt0 * self.R0) * 1e-6  # Normalized

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Pressure
        ax1.plot(r/self.a, p/1e3, 'b-', linewidth=2)
        ax1.set_xlabel('r/a', fontsize=12)
        ax1.set_ylabel('Pressure [kPa]', fontsize=12)
        ax1.set_title('Pressure Profile', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Current density
        ax2.plot(r/self.a, j_phi, 'r-', linewidth=2)
        ax2.set_xlabel('r/a', fontsize=12)
        ax2.set_ylabel('Current Density (normalized)', fontsize=12)
        ax2.set_title('Toroidal Current Density Profile', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example usage
def example_tokamak_equilibrium():
    """ITER-like parameters"""
    R0 = 6.2    # Major radius [m]
    a = 2.0     # Minor radius [m]
    kappa = 1.7 # Elongation
    delta = 0.33# Triangularity
    Bt0 = 5.3   # Toroidal field [T]
    Ip = 15e6   # Plasma current [A]

    eq = SolovevEquilibrium(R0, a, kappa, delta, Bt0, Ip)

    print("=== ITER-like Tokamak Equilibrium ===")
    print(f"Major radius R0 = {R0} m")
    print(f"Minor radius a = {a} m")
    print(f"Aspect ratio A = {R0/a:.2f}")
    print(f"Elongation κ = {kappa}")
    print(f"Triangularity δ = {delta}")
    print(f"Toroidal field Bt0 = {Bt0} T")
    print(f"Plasma current Ip = {Ip/1e6:.1f} MA")

    # Compute some equilibrium quantities
    psi_axis = eq.compute_psi(R0, 0)
    psi_edge = eq.compute_psi(R0 + a, 0)

    print(f"\nFlux at axis: {psi_axis:.3e} Wb")
    print(f"Flux at edge: {psi_edge:.3e} Wb")

    # Safety factor
    q_axis = eq.compute_q(np.array([psi_axis]))[0]
    q_edge = eq.compute_q(np.array([psi_edge]))[0]

    print(f"\nSafety factor q(0) = {q_axis:.2f}")
    print(f"Safety factor q(a) = {q_edge:.2f}")

    # Plot results
    fig1 = eq.plot_flux_surfaces()
    plt.savefig('/tmp/flux_surfaces.png', dpi=150)
    print("\nFlux surfaces plot saved to /tmp/flux_surfaces.png")

    fig2 = eq.plot_q_profile()
    plt.savefig('/tmp/q_profile.png', dpi=150)
    print("q-profile plot saved to /tmp/q_profile.png")

    fig3 = eq.plot_pressure_profile()
    plt.savefig('/tmp/pressure_profile.png', dpi=150)
    print("Pressure profile plot saved to /tmp/pressure_profile.png")

    plt.close('all')

if __name__ == "__main__":
    example_tokamak_equilibrium()
```

### 9.2 Numerical Grad-Shafranov Solver

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

class GradShafranovSolver:
    """Numerical solver for Grad-Shafranov equation"""

    def __init__(self, R_min, R_max, Z_min, Z_max, nR, nZ):
        """
        Set up computational domain
        """
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.nR = nR
        self.nZ = nZ

        # Grid spacing
        self.dR = (R_max - R_min) / (nR - 1)
        self.dZ = (Z_max - Z_min) / (nZ - 1)

        # Create grid
        self.R = np.linspace(R_min, R_max, nR)
        self.Z = np.linspace(Z_min, Z_max, nZ)
        self.R_grid, self.Z_grid = np.meshgrid(self.R, self.Z)

        # Initialize flux
        self.psi = np.zeros((nZ, nR))

        # Constants
        self.mu0 = 4*np.pi*1e-7

    def set_free_functions(self, p_func, F_func):
        """
        Set pressure and toroidal field functions
        p_func: p(psi) callable
        F_func: F(psi) callable where F = R*B_phi
        """
        self.p_func = p_func
        self.F_func = F_func

    def build_operator_matrix(self):
        """
        Build discrete Grad-Shafranov operator matrix
        Δ* ψ = R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²
        """
        nR = self.nR
        nZ = self.nZ
        N = nR * nZ

        # Flatten index: (i,j) -> i*nR + j
        def idx(i, j):
            return i * nR + j

        # Build sparse matrix
        row = []
        col = []
        data = []

        for i in range(nZ):
            for j in range(nR):
                n = idx(i, j)
                R = self.R[j]

                # Interior points
                if 0 < i < nZ-1 and 0 < j < nR-1:
                    # R derivatives: R ∂/∂R(1/R ∂ψ/∂R)
                    # = ∂²ψ/∂R² + (1/R)∂ψ/∂R - (ψ/R²)
                    # Discretize: centered differences

                    # ∂²ψ/∂R²
                    coef_R_pp = 1 / self.dR**2
                    coef_R_0 = -2 / self.dR**2
                    coef_R_mm = 1 / self.dR**2

                    # (1/R)∂ψ/∂R
                    coef_R_p_1st = 1 / (2*R*self.dR)
                    coef_R_m_1st = -1 / (2*R*self.dR)

                    # Z derivatives: ∂²ψ/∂Z²
                    coef_Z_pp = 1 / self.dZ**2
                    coef_Z_0 = -2 / self.dZ**2
                    coef_Z_mm = 1 / self.dZ**2

                    # Combine
                    # Center
                    row.append(n)
                    col.append(n)
                    data.append(coef_R_0 + coef_Z_0)

                    # R+1
                    row.append(n)
                    col.append(idx(i, j+1))
                    data.append(coef_R_pp + coef_R_p_1st)

                    # R-1
                    row.append(n)
                    col.append(idx(i, j-1))
                    data.append(coef_R_mm + coef_R_m_1st)

                    # Z+1
                    row.append(n)
                    col.append(idx(i+1, j))
                    data.append(coef_Z_pp)

                    # Z-1
                    row.append(n)
                    col.append(idx(i-1, j))
                    data.append(coef_Z_mm)

                else:
                    # Boundary: ψ = 0
                    row.append(n)
                    col.append(n)
                    data.append(1.0)

        matrix = csr_matrix((data, (row, col)), shape=(N, N))
        return matrix

    def solve_fixed_boundary(self, psi_boundary=0, max_iter=100, tol=1e-6):
        """
        Solve Grad-Shafranov with fixed boundary using Picard iteration
        """
        nR = self.nR
        nZ = self.nZ
        N = nR * nZ

        # The operator matrix A encodes Δ* (the Grad-Shafranov operator) and
        # does NOT depend on ψ, so it is built once and reused every iteration
        # — avoiding an expensive rebuild at each Picard step.
        A = self.build_operator_matrix()

        # Picard (fixed-point) iteration: the GS equation is non-linear because
        # p(ψ) and F(ψ) depend on the solution itself.  Each step freezes those
        # free functions at the current ψ^(n), solves the resulting LINEAR system
        # for ψ^(n+1), and repeats.  Convergence is guaranteed for well-posed
        # pressure/current profiles but may be slow for strongly non-linear cases.
        for iteration in range(max_iter):
            psi_old = self.psi.copy()

            # Compute RHS: -μ₀ R² dp/dψ - F dF/dψ
            rhs = np.zeros((nZ, nR))

            for i in range(nZ):
                for j in range(nR):
                    R = self.R[j]
                    psi_val = self.psi[i, j]

                    # Centered finite difference in ψ-space gives the dp/dψ and
                    # dF/dψ source terms at the current-iteration flux value;
                    # the small step dpsi=1e-6 avoids cancellation error while
                    # remaining within the smooth part of the free functions.
                    dpsi = 1e-6
                    dpdpsi = (self.p_func(psi_val + dpsi) - self.p_func(psi_val - dpsi)) / (2*dpsi)

                    F_val = self.F_func(psi_val)
                    dFdpsi = (self.F_func(psi_val + dpsi) - self.F_func(psi_val - dpsi)) / (2*dpsi)

                    # The R² weight in the pressure term comes directly from the
                    # GS equation: -μ₀R²(dp/dψ) drives the poloidal field
                    # curvature that balances plasma pressure in the torus.
                    rhs[i, j] = -self.mu0 * R**2 * dpdpsi - F_val * dFdpsi

            # Dirichlet boundary condition ψ=0 at the domain edges represents
            # a "vacuum" boundary where no flux crosses; this is the simplest
            # fixed-boundary assumption used in free-boundary GS codes before
            # the coil currents are iterated.
            rhs[0, :] = psi_boundary
            rhs[-1, :] = psi_boundary
            rhs[:, 0] = psi_boundary
            rhs[:, -1] = psi_boundary

            # Solve linear system
            rhs_flat = rhs.flatten()
            psi_flat = spsolve(A, rhs_flat)
            self.psi = psi_flat.reshape((nZ, nR))

            # L∞ norm of the update is used here because a single large
            # deviation anywhere (e.g., near the axis) would be missed by an
            # average norm — max-norm catches the worst-case residual.
            error = np.max(np.abs(self.psi - psi_old))

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: max error = {error:.3e}")

            if error < tol:
                print(f"Converged in {iteration+1} iterations")
                break

        return self.psi

    def plot_solution(self):
        """Plot the computed equilibrium"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Flux surfaces
        ax = axes[0]
        levels = 30
        CS = ax.contour(self.R_grid, self.Z_grid, self.psi, levels=levels, colors='blue')
        ax.clabel(CS, inline=True, fontsize=8)
        ax.set_xlabel('R [m]', fontsize=12)
        ax.set_ylabel('Z [m]', fontsize=12)
        ax.set_title('Poloidal Flux Surfaces', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Pressure profile
        ax = axes[1]
        psi_flat = self.psi.flatten()
        psi_min = np.min(psi_flat)
        psi_max = np.max(psi_flat)
        psi_range = np.linspace(psi_min, psi_max, 100)

        p_range = np.array([self.p_func(psi) for psi in psi_range])

        ax.plot(psi_range, p_range/1e3, 'r-', linewidth=2)
        ax.set_xlabel('ψ [Wb]', fontsize=12)
        ax.set_ylabel('Pressure [kPa]', fontsize=12)
        ax.set_title('Pressure Profile p(ψ)', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example: solve simple equilibrium
def example_gs_solver():
    """Example Grad-Shafranov solution"""

    # Domain: tokamak-like geometry
    R0 = 1.0  # Major radius
    a = 0.3   # Minor radius

    R_min = R0 - 1.5*a
    R_max = R0 + 1.5*a
    Z_min = -1.5*a
    Z_max = 1.5*a

    nR = 80
    nZ = 80

    solver = GradShafranovSolver(R_min, R_max, Z_min, Z_max, nR, nZ)

    # Define free functions
    # Simple parabolic pressure
    p0 = 1e5  # 100 kPa
    psi0 = -0.1

    def p_func(psi):
        if psi > psi0:
            return 0.0
        else:
            return p0 * (1 - psi/psi0)**2

    # Constant F (uniform toroidal field)
    Bt0 = 2.0  # Tesla
    F0 = Bt0 * R0

    def F_func(psi):
        return F0

    solver.set_free_functions(p_func, F_func)

    print("=== Grad-Shafranov Solver ===")
    print(f"Grid: {nR} x {nZ}")
    print(f"Domain: R ∈ [{R_min}, {R_max}], Z ∈ [{Z_min}, {Z_max}]")
    print(f"Central pressure: {p0/1e3} kPa")
    print(f"Toroidal field: {Bt0} T")

    # Solve
    psi = solver.solve_fixed_boundary(max_iter=100, tol=1e-6)

    # Plot
    fig = solver.plot_solution()
    plt.savefig('/tmp/gs_solution.png', dpi=150)
    print("\nSolution plot saved to /tmp/gs_solution.png")
    plt.close()

if __name__ == "__main__":
    example_gs_solver()
```

## 10. Beta Calculation

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_beta(R0, a, p_profile, B_profiles, nr=100):
    """
    Compute various beta values for a tokamak equilibrium

    Parameters:
    -----------
    R0: major radius [m]
    a: minor radius [m]
    p_profile: function p(r) giving pressure profile
    B_profiles: dict with 'Bt', 'Bp' functions of r
    nr: number of radial points

    Returns:
    --------
    dict with beta_p, beta_t, beta_N
    """
    r = np.linspace(0, a, nr)
    dr = r[1] - r[0]

    # dV = 2πR₀ · 2πr dr is the toroidal volume element in the large-aspect-
    # ratio limit: the 2πR₀ factor accounts for the full toroidal circuit while
    # 2πr dr sweeps a thin annular ring in the poloidal cross-section.
    def volume_element(r_val):
        return 4 * np.pi**2 * R0 * r_val * dr

    # Compute volume-averaged quantities
    p_vals = np.array([p_profile(r_val) for r_val in r])
    Bt_vals = np.array([B_profiles['Bt'](r_val) for r_val in r])
    Bp_vals = np.array([B_profiles['Bp'](r_val) for r_val in r])

    # Weighted sums approximate the volume integral ∫ f dV / ∫ dV;
    # using the explicit dV weight makes the code's discretization transparent
    # and avoids hidden normalization errors when the grid is non-uniform.
    V_total = np.sum([volume_element(r[i]) for i in range(nr)])

    p_avg = np.sum([p_vals[i] * volume_element(r[i]) for i in range(nr)]) / V_total

    Bt2_avg = np.sum([Bt_vals[i]**2 * volume_element(r[i]) for i in range(nr)]) / V_total
    Bp2_avg = np.sum([Bp_vals[i]**2 * volume_element(r[i]) for i in range(nr)]) / V_total

    mu0 = 4*np.pi*1e-7

    # β_p uses <B_p²> rather than B_p(0) because poloidal beta captures how
    # well the plasma pressure is balanced by its own poloidal field — the
    # figure of merit for current-driven stability limits.
    beta_p = 2 * mu0 * p_avg / Bp2_avg

    # β_t uses <B_t²> ≈ B_t0² because toroidal beta is the engineering limit
    # for the external vacuum field that the coils must produce; it directly
    # enters the Troyon empirical beta limit.
    beta_t = 2 * mu0 * p_avg / Bt2_avg

    # For beta_N, need plasma current
    # I_p = ∮ J·dl ~ B_p * circumference / μ₀
    Bp_edge = Bp_vals[-1]
    # Ampère's law on a circle of radius a: I_p = (2πa B_p(a)) / μ₀
    # This avoids integrating the current density profile directly, relying
    # instead on the boundary value of the poloidal field.
    Ip = 2 * np.pi * a * Bp_edge / mu0

    # β_N normalizes β_t by I_p/(aB_t) so that the Troyon limit β_N < 2.8–3.5
    # is machine-independent: the factor I_p/(aB_t) scales with the plasma's
    # ability to drive kink-stabilizing current relative to the applied field.
    Bt_axis = Bt_vals[0]
    beta_N = beta_t * 100 / (Ip / (a * Bt_axis))  # percentage

    results = {
        'beta_p': beta_p,
        'beta_t': beta_t,
        'beta_N': beta_N,
        'p_avg': p_avg,
        'Ip': Ip
    }

    return results

def example_beta_calculation():
    """Example beta calculation for tokamak"""

    # ITER-like parameters
    R0 = 6.2
    a = 2.0

    # Parabolic pressure profile
    p0 = 5e5  # 500 kPa
    def p_profile(r):
        return p0 * (1 - (r/a)**2)**2

    # Magnetic field profiles
    Bt0 = 5.3  # On-axis toroidal field
    Ip = 15e6  # Plasma current

    def Bt_profile(r):
        return Bt0 * R0 / (R0 + r)  # 1/R dependence

    def Bp_profile(r):
        mu0 = 4*np.pi*1e-7
        # From Ampere's law, current ~ r² profile
        I_enclosed = Ip * (r/a)**2
        return mu0 * I_enclosed / (2 * np.pi * r) if r > 0 else 0

    B_profiles = {
        'Bt': Bt_profile,
        'Bp': Bp_profile
    }

    # Compute betas
    results = compute_beta(R0, a, p_profile, B_profiles)

    print("=== Beta Calculation ===")
    print(f"Major radius R0 = {R0} m")
    print(f"Minor radius a = {a} m")
    print(f"Central pressure p0 = {p0/1e3} kPa")
    print(f"Toroidal field Bt0 = {Bt0} T")
    print(f"Plasma current Ip = {results['Ip']/1e6:.1f} MA")
    print(f"\nAverage pressure <p> = {results['p_avg']/1e3:.1f} kPa")
    print(f"Poloidal beta β_p = {results['beta_p']:.3f}")
    print(f"Toroidal beta β_t = {results['beta_t']*100:.2f} %")
    print(f"Normalized beta β_N = {results['beta_N']:.2f}")

    # Troyon limit check
    beta_N_limit = 3.5
    print(f"\nTroyon limit β_N < {beta_N_limit}")
    if results['beta_N'] < beta_N_limit:
        print("✓ Within stability limit")
    else:
        print("✗ Exceeds stability limit!")

    # Plot profiles
    r = np.linspace(0.01, a, 100)
    p = np.array([p_profile(ri) for ri in r])
    Bt = np.array([Bt_profile(ri) for ri in r])
    Bp = np.array([Bp_profile(ri) for ri in r])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pressure
    axes[0,0].plot(r/a, p/1e3, 'b-', linewidth=2)
    axes[0,0].set_xlabel('r/a')
    axes[0,0].set_ylabel('Pressure [kPa]')
    axes[0,0].set_title('Pressure Profile')
    axes[0,0].grid(True, alpha=0.3)

    # Toroidal field
    axes[0,1].plot(r/a, Bt, 'g-', linewidth=2)
    axes[0,1].set_xlabel('r/a')
    axes[0,1].set_ylabel('B_t [T]')
    axes[0,1].set_title('Toroidal Field')
    axes[0,1].grid(True, alpha=0.3)

    # Poloidal field
    axes[1,0].plot(r/a, Bp, 'r-', linewidth=2)
    axes[1,0].set_xlabel('r/a')
    axes[1,0].set_ylabel('B_p [T]')
    axes[1,0].set_title('Poloidal Field')
    axes[1,0].grid(True, alpha=0.3)

    # Local beta
    beta_local = 2 * (4*np.pi*1e-7) * p / (Bt**2 + Bp**2)
    axes[1,1].plot(r/a, beta_local*100, 'm-', linewidth=2)
    axes[1,1].set_xlabel('r/a')
    axes[1,1].set_ylabel('β [%]')
    axes[1,1].set_title('Local Beta Profile')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/beta_profiles.png', dpi=150)
    print("\nProfiles plot saved to /tmp/beta_profiles.png")
    plt.close()

if __name__ == "__main__":
    example_beta_calculation()
```

## Summary

In this lesson, we have covered the fundamentals of MHD equilibria:

1. **Force Balance**: The fundamental equation $\nabla p = \mathbf{J}\times\mathbf{B}$ balances plasma pressure gradient against magnetic forces (pressure + tension).

2. **Consequences**: Pressure and current lie on magnetic flux surfaces, enabling flux surface coordinates.

3. **1D Equilibria**: θ-pinch (pure axial field), Z-pinch (self-generated azimuthal field, Bennett relation), and screw pinch (combined fields with shear).

4. **Grad-Shafranov Equation**: The elliptic PDE governing axisymmetric toroidal equilibria, requiring specification of two free functions $p(\psi)$ and $F(\psi)$.

5. **Safety Factor**: The parameter $q$ measuring field line pitch, critical for stability analysis (Kruskal-Shafranov limit, rational surfaces).

6. **Flux Surfaces**: Nested toroidal surfaces on which pressure is constant, with Shafranov shift due to toroidal effects.

7. **Plasma Beta**: The ratio of plasma to magnetic pressure, with operational limits set by MHD stability (Troyon limit).

8. **Numerical Methods**: Implementation of equilibrium solvers using finite differences and iterative techniques.

These equilibrium concepts form the foundation for understanding plasma stability (next lessons), transport, and confinement in fusion devices.

## Practice Problems

### Problem 1: Force Balance in a Cylindrical Plasma

A cylindrical plasma column has the following profiles:
- Pressure: $p(r) = p_0(1 - r^2/a^2)$ for $r < a$, $p=0$ for $r \geq a$
- Axial field: $B_z = B_0 = \text{const}$
- Azimuthal field: $B_θ(r)$ to be determined

**(a)** Using the radial force balance equation, derive $B_θ(r)$.

**(b)** Compute the total plasma current $I_p$.

**(c)** Calculate the safety factor $q(r)$ assuming major radius $R_0 = 5a$.

**(d)** What is $q$ on the magnetic axis ($r=0$)?

**Hint**: Use $\nabla p = \mathbf{J}\times\mathbf{B}$ in cylindrical coordinates.

### Problem 2: Bennett Relation for Z-Pinch

A Z-pinch has uniform density $n = 10^{20}$ m$^{-3}$, temperature $T = 10$ keV, length $L = 1$ m, and radius $a = 1$ cm.

**(a)** Calculate the total number of particles $N = n \pi a^2 L$.

**(b)** Using the Bennett relation $I^2 = (8\pi/\mu_0)NkT$, compute the required current $I_p$.

**(c)** Estimate the magnetic field at the surface $B_θ(a) = \mu_0 I_p/(2\pi a)$.

**(d)** Compute the magnetic pressure $B_θ^2/(2\mu_0)$ and compare with plasma pressure $p = nkT$.

**(e)** Discuss the stability implications of this configuration.

### Problem 3: Grad-Shafranov with Constant Pressure

Consider the Grad-Shafranov equation with:
- $p(\psi) = p_0 = \text{const}$
- $F(\psi) = F_0 = \text{const}$

**(a)** Show that the equation reduces to:
$$
\Delta^*\psi = -\mu_0 p_0 R^2
$$

**(b)** For a large-aspect-ratio circular tokamak ($R \approx R_0$), approximate this as:
$$
\frac{1}{r}\frac{d}{dr}\left(r\frac{d\psi}{dr}\right) + \frac{d^2\psi}{dz^2} \approx -\mu_0 p_0 R_0^2
$$

**(c)** Propose a separable solution $\psi(r,z) = R_r(r)Z_z(z)$ and derive ODEs for $R_r$ and $Z_z$.

**(d)** Solve for circular flux surfaces $\psi \propto r^2 + \kappa^2 z^2$ and determine $\kappa$ (elongation).

### Problem 4: Safety Factor and Current Profile

A tokamak has major radius $R_0 = 3$ m, minor radius $a = 1$ m, and toroidal field $B_t = 5$ T (approximately constant). The current density profile is:

$$
J_z(r) = J_0\left(1 - \frac{r^2}{a^2}\right)
$$

**(a)** Compute the enclosed current $I(r) = \int_0^r J_z(r') 2\pi r' dr'$.

**(b)** Using Ampère's law, find $B_θ(r) = \mu_0 I(r)/(2\pi r)$.

**(c)** Calculate the safety factor profile $q(r) = rB_t/(R_0 B_θ(r))$.

**(d)** Determine $q(0)$ (on-axis) and $q(a)$ (at edge).

**(e)** Find the radius $r_s$ where $q(r_s) = 2$ (the $m=2$ rational surface).

**(f)** Plot $q(r)$ and identify stability-relevant features.

### Problem 5: Beta Limits

An experimental tokamak operates with:
- Minor radius $a = 0.5$ m
- Toroidal field $B_t = 3$ T
- Plasma current $I_p = 1$ MA
- Central pressure $p_0 = 10^5$ Pa
- Pressure profile $p(r) = p_0(1 - r^2/a^2)^2$

**(a)** Compute the volume-averaged pressure $\langle p\rangle$.

**(b)** Calculate the toroidal beta $\beta_t = 2\mu_0\langle p\rangle/B_t^2$.

**(c)** Compute the normalized beta $\beta_N = \beta_t(\%)/(I_p[MA]/(a[m]B_t[T]))$.

**(d)** Compare $\beta_N$ with the Troyon limit $\beta_N < 3.5$.

**(e)** If the experiment wants to double the pressure, what adjustments to $I_p$ or $B_t$ would maintain stability?

**Hint**: For volume average in a cylinder: $\langle p\rangle = \frac{\int_0^a p(r) 2\pi r dr}{\pi a^2}$.

---

**Previous**: [Overview](./00_Overview.md) | **Next**: [Linear Stability](./02_Linear_Stability.md)

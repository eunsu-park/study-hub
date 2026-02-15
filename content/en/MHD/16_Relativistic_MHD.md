# 16. Relativistic MHD

## Learning Objectives

By the end of this lesson, you will be able to:

- Formulate the equations of special relativistic MHD (SRMHD) using covariant notation
- Understand the stress-energy tensor and electromagnetic field tensor in SRMHD
- Derive the 3+1 decomposition of SRMHD equations for numerical implementation
- Solve the relativistic primitive variable recovery problem
- Analyze the wave structure of relativistic MHD (fast, slow, Alfvén)
- Apply SRMHD to relativistic jets, pulsar magnetospheres, and black hole accretion
- Implement a 1D SRMHD shock tube solver in Python
- Understand the basics of general relativistic MHD (GRMHD) and its applications

---

## 1. Introduction to Relativistic MHD

### 1.1 When is Relativistic MHD Necessary?

Non-relativistic MHD assumes $v \ll c$ and neglects:
- Lorentz contraction of the electromagnetic field
- Displacement current in Maxwell equations
- Relativistic mass-energy equivalence

Relativistic MHD (RMHD) is essential when:

```
• Flow velocities approach c: v/c ~ 0.1-1
  - Relativistic jets: AGN, GRBs (Γ ~ 10-100)
  - Pulsar winds (Γ ~ 10⁴-10⁶)
  - Accretion disk inner regions (v ~ 0.3c at ISCO)

• Magnetic pressure dominates: σ = B²/(4πρc²) ≫ 1
  - Magnetar magnetospheres: B ~ 10¹⁵ G
  - Pulsar polar caps

• Strong gravitational fields:
  - Black hole accretion (r ~ 2-10 GM/c²)
  - Neutron star mergers
```

### 1.2 Key Differences from Non-Relativistic MHD

| Aspect | Non-Relativistic | Relativistic |
|--------|------------------|--------------|
| Electric field | $\mathbf{E} = -\mathbf{v} \times \mathbf{B}/c$ | $\mathbf{E}$ is independent dynamical variable |
| Displacement current | Neglected | $\partial \mathbf{E}/\partial t$ included |
| Conservation laws | Mass, momentum, energy separate | Unified in $\partial_\mu T^{\mu\nu} = 0$ |
| Wave speeds | Independent of $v$ | Modified by Lorentz factor $W$ |
| Primitive recovery | Algebraic | Implicit rootfinding required |

---

## 2. Special Relativistic MHD (SRMHD)

### 2.1 Covariant Formulation

**Metric and 4-vectors:**

Minkowski metric (signature $-,+,+,+$):
$$
\eta_{\mu\nu} = \text{diag}(-1, 1, 1, 1)
$$

4-velocity:
$$
u^\mu = W(c, \mathbf{v}), \quad u^\mu u_\mu = -c^2
$$
where $W = 1/\sqrt{1 - v^2/c^2}$ is the Lorentz factor.

**Electromagnetic field tensor:**
$$
F^{\mu\nu} = \begin{pmatrix}
0 & -E_x/c & -E_y/c & -E_z/c \\
E_x/c & 0 & -B_z & B_y \\
E_y/c & B_z & 0 & -B_x \\
E_z/c & -B_y & B_x & 0
\end{pmatrix}
$$

Dual tensor:
$$
F^{*\mu\nu} = \frac{1}{2} \epsilon^{\mu\nu\alpha\beta} F_{\alpha\beta}
$$
where $\epsilon^{\mu\nu\alpha\beta}$ is the Levi-Civita tensor.

### 2.2 Maxwell Equations in Covariant Form

**Source-free Maxwell equations:**
$$
\partial_\mu F^{*\mu\nu} = 0 \quad \Rightarrow \quad \nabla \cdot \mathbf{B} = 0, \quad \nabla \times \mathbf{E} + \frac{\partial \mathbf{B}}{\partial t} = 0
$$

**With currents (for resistive RMHD):**
$$
\partial_\mu F^{\mu\nu} = \frac{4\pi}{c} J^\nu
$$
where $J^\mu = (c\rho_e, \mathbf{J})$ is the 4-current.

### 2.3 Stress-Energy Tensor

**Total stress-energy tensor:**
$$
T^{\mu\nu} = T^{\mu\nu}_{\text{fluid}} + T^{\mu\nu}_{\text{EM}}
$$

**Fluid contribution:**
$$
T^{\mu\nu}_{\text{fluid}} = (\rho h + u_m) u^\mu u^\nu + (p + p_m) \eta^{\mu\nu}
$$
where:
- $\rho$ = rest mass density
- $h = 1 + \epsilon + p/(\rho c^2)$ = specific enthalpy
- $\epsilon$ = specific internal energy
- $u_m = b^2/(8\pi)$ = magnetic energy density in comoving frame
- $p_m = b^2/(8\pi)$ = magnetic pressure (isotropic part)

**Electromagnetic contribution:**
$$
T^{\mu\nu}_{\text{EM}} = \frac{1}{4\pi} \left( F^{\mu\alpha} F^\nu_{\ \alpha} - \frac{1}{4} \eta^{\mu\nu} F^{\alpha\beta} F_{\alpha\beta} \right)
$$

**4-magnetic field (comoving frame):**
$$
b^\mu = \frac{1}{c} F^{*\mu\nu} u_\nu = W(\mathbf{v} \cdot \mathbf{B}/c, \mathbf{B}/W + W(\mathbf{v} \cdot \mathbf{B})\mathbf{v}/c^2)
$$

Satisfies $b^\mu u_\mu = 0$ (orthogonality) and $b^2 = b^\mu b_\mu = (B^2 + (v \times B)^2/c^2)/W^2$.

### 2.4 Conservation Laws

**Energy-momentum conservation:**
$$
\partial_\mu T^{\mu\nu} = 0
$$

Expands to:
- $\nu = 0$: energy conservation
- $\nu = i$: momentum conservation

**Mass conservation:**
$$
\partial_\mu (\rho u^\mu) = 0
$$

---

## 3. Ideal SRMHD

### 3.1 Ideal Condition

In ideal SRMHD, the electric field vanishes in the comoving frame:
$$
F^{\mu\nu} u_\nu = 0
$$

This gives:
$$
\mathbf{E} = -\frac{\mathbf{v} \times \mathbf{B}}{c} \frac{1}{1 - v^2/c^2}
$$

The magnetic field is **frozen into the plasma** (relativistic version).

### 3.2 3+1 Decomposition

For numerical implementation, we decompose into lab-frame (3+1) variables.

**Conserved variables:**
$$
\mathbf{U} = \begin{pmatrix} D \\ \mathbf{S} \\ \tau \\ \mathbf{B} \end{pmatrix}
$$
where:
- $D = \rho W$ (conserved density)
- $\mathbf{S} = (\rho h + b^2) W^2 \mathbf{v} - (\mathbf{v} \cdot \mathbf{B})\mathbf{B}/(4\pi)$ (momentum density)
- $\tau = (\rho h + b^2) W^2 - p - b^2/2 - D c^2$ (energy density)
- $\mathbf{B}$ (magnetic field)

**Flux functions:**
$$
\mathbf{F}(\mathbf{U}) = \begin{pmatrix}
D v_x \\
S_x v_x + p_{\text{tot}} - B_x^2/(4\pi) \\
S_y v_x - B_x B_y/(4\pi) \\
S_z v_x - B_x B_z/(4\pi) \\
\tau v_x + p_{\text{tot}} v_x - (\mathbf{v} \cdot \mathbf{B}) B_x/(4\pi) \\
0 \\
B_y v_x - B_x v_y \\
B_z v_x - B_x v_z
\end{pmatrix}
$$
where $p_{\text{tot}} = p + B^2/(8\pi)$.

**Conservation form:**
$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = 0
$$

### 3.3 Primitive Variable Recovery

**The challenge:** Given conserved variables $(\mathbf{U})$, recover primitive variables $(\rho, \mathbf{v}, p, \mathbf{B})$.

**Algebraic constraints:**
$$
\begin{aligned}
D &= \rho W \\
\mathbf{B} &= \text{(known)} \\
\mathbf{S} &= (\rho h + b^2) W^2 \mathbf{v} - (\mathbf{v} \cdot \mathbf{B})\mathbf{B}/(4\pi) \\
\tau &= (\rho h + b^2) W^2 - p - b^2/2 - D c^2
\end{aligned}
$$

**Problem:** Nonlinear system coupling $W$, $p$, $\rho$, $\mathbf{v}$.

**Standard approach (2D Newton-Raphson):**

1. Choose unknowns: $z = W$ and $w = p$
2. Invert:
   $$
   v^2 = 1 - \frac{1}{z^2}
   $$
3. From $\mathbf{S}$ and momentum equation, solve for $\rho$, $h$
4. Use EOS: $p = p(\rho, \epsilon)$
5. Iterate until convergence

**Alternative (1D rootfinding):**

Use pressure $p$ as independent variable, solve:
$$
f(p) = \tau + p - \frac{(\mathbf{S} \cdot \mathbf{B})^2}{(\tau + p + D c^2 + B^2/(4\pi))^2 - (\mathbf{S}^2 + (\mathbf{S} \cdot \mathbf{B})^2/(B^2))} - D c^2 = 0
$$

**Challenges:**
- Multiple roots possible
- Numerical stiffness for $W \gg 1$
- Breakdown near vacuum ($\rho \to 0$)

**Best practices:**
- Use robust rootfinder (Brent's method)
- Good initial guess (from previous timestep)
- Floor values for $\rho$, $p$

---

## 4. Wave Structure in SRMHD

### 4.1 Eigenstructure

The SRMHD system has **7 waves** (1D):
$$
\lambda_{1,7} = \alpha_{\pm}^{\text{fast}}, \quad \lambda_{2,6} = \alpha_{\pm}^{\text{slow}}, \quad \lambda_{3,5} = \alpha_{\pm}^{\text{Alf}}, \quad \lambda_4 = v_x
$$

**Relativistic wave speeds:**

Define:
$$
c_s^2 = \frac{\partial p}{\partial \rho h} \quad \text{(relativistic sound speed)}
$$
$$
v_A^2 = \frac{B^2/(4\pi)}{\rho h + B^2/(4\pi)} \quad \text{(relativistic Alfvén speed)}
$$

Fast magnetosonic speed:
$$
\alpha_{\pm}^{\text{fast}} = \frac{v_x \pm c_{\text{fast}}}{1 \pm v_x c_{\text{fast}}/c^2}
$$
where:
$$
c_{\text{fast}}^2 = \frac{c_s^2 + v_A^2 - c_s^2 v_A^2}{1 - c_s^2 v_A^2}
$$

Slow magnetosonic and Alfvén speeds defined similarly with modified dispersion relation.

**Key differences from non-relativistic:**
- All wave speeds < $c$ (causality)
- Lorentz factor $W$ enters dispersion relations
- For $v \to c$, waves cannot outrun flow (bunching)

### 4.2 Relativistic Riemann Problem

**HLLC solver for SRMHD:**

Estimate wave speeds:
$$
\lambda_L = \min(\lambda_L^{\text{fast}}, \lambda_R^{\text{fast}}), \quad \lambda_R = \max(\lambda_L^{\text{fast}}, \lambda_R^{\text{fast}})
$$

Contact wave speed $\lambda_*$ from momentum jump conditions.

**HLLD solver:**

Includes all 5 waves (fast, Alfvén, contact). More accurate but complex.

**Relativistic Brio-Wu shock tube:**

Initial conditions:
$$
(\rho, v_x, v_y, v_z, p, B_y) = \begin{cases}
(1, 0, 0, 0, 1, 1) & x < 0.5 \\
(0.125, 0, 0, 0, 0.1, -1) & x > 0.5
\end{cases}
$$
with $B_x = 0.5$ constant.

Expected structure: left fast → compound → contact → slow → right fast.

---

## 5. Applications of SRMHD

### 5.1 Relativistic Jets

**Astrophysical context:**
- Active Galactic Nuclei (AGN): jets from supermassive black holes (Lorentz factors $\Gamma \sim 10-30$)
- Gamma-Ray Bursts (GRBs): jets from stellar-mass black holes ($\Gamma \sim 100-1000$)
- Microquasars: jets from X-ray binaries ($\Gamma \sim 2-10$)

**Physics:**
- **Acceleration:** magnetic pressure converts to kinetic energy ($\sigma \to 0$)
- **Collimation:** magnetic hoop stress confines jet
- **Instabilities:** Kelvin-Helmholtz (jet-ambient interface), current-driven kink

**Light cylinder:**

For rotating magnetosphere with angular velocity $\Omega$:
$$
R_L = \frac{c}{\Omega}
$$

Inside $R_L$: corotation possible. Outside: magnetic field opens, wind launched.

**Numerical challenges:**
- Large Lorentz factors: $W \sim 100$ → stiff primitive recovery
- Magnetization parameter $\sigma = B^2/(4\pi \rho h W^2)$ ranges over decades
- Grid-scale instabilities if $\sigma \gg 1$

### 5.2 Pulsar Magnetospheres

**Oblique rotator:**
- Rotating magnetic dipole: $\mathbf{m}$ inclined at angle $\alpha$ to $\boldsymbol{\Omega}$
- Open field lines: $\theta < \theta_{\text{pc}}$ (polar cap angle)
- Closed field lines: corotating plasma

**Force-free electrodynamics (FFE):**

When $\sigma \gg 1$, plasma inertia negligible:
$$
\rho_e \mathbf{E} + \frac{\mathbf{J} \times \mathbf{B}}{c} = 0, \quad \mathbf{E} \cdot \mathbf{B} = 0
$$

Solve for $\mathbf{E}$, $\mathbf{B}$ given boundary conditions (neutron star surface, infinity).

**Pulsar wind:**
- Particles accelerated along open field lines
- Magnetization $\sigma \sim 10^4$ at light cylinder
- Dissipation at termination shock (pulsar wind nebula)

**SRMHD vs FFE:**
- FFE: valid for $\sigma \gg 1$, no dissipation
- SRMHD: includes inertia, reconnection, particle heating

### 5.3 Black Hole Accretion

**Innermost Stable Circular Orbit (ISCO):**
- Schwarzschild: $r_{\text{ISCO}} = 6 GM/c^2$
- Kerr (extreme): $r_{\text{ISCO}} = GM/c^2$ (prograde)

Orbital velocity at ISCO:
$$
v_{\text{ISCO}} \sim 0.5c \quad \text{(Schwarzschild)} \quad \text{to} \quad 0.7c \quad \text{(Kerr, prograde)}
$$

**Jet launching:**

Blandford-Znajek mechanism (for spinning black hole):
- Magnetic field threads horizon
- Frame dragging: $\Omega_H$ (horizon angular velocity)
- Poynting flux extracted: $L_{\text{BZ}} \sim \Omega_H^2 B_H^2 r_H^4 / c$

Requires **general relativistic MHD** (GRMHD).

**Magnetically Arrested Disk (MAD):**

When magnetic flux accumulates:
$$
\phi \sim \sqrt{\dot{M} r_g c} \quad \Rightarrow \quad \text{magnetically dominated}
$$

Leads to:
- Suppressed accretion (flux barrier)
- Enhanced jet power
- Time-variable flow

---

## 6. General Relativistic MHD (GRMHD)

### 6.1 Curved Spacetime Formulation

**Kerr metric (Boyer-Lindquist coordinates):**
$$
ds^2 = -\alpha^2 dt^2 + \gamma_{ij} (dx^i + \beta^i dt)(dx^j + \beta^j dt)
$$
where:
- $\alpha$ = lapse function
- $\beta^i$ = shift vector
- $\gamma_{ij}$ = spatial metric

**Stress-energy tensor:**
$$
\nabla_\mu T^{\mu\nu} = 0
$$
where $\nabla_\mu$ is the covariant derivative (includes Christoffel symbols).

**3+1 ADM form:**

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}^i}{\partial x^i} = \mathbf{S}
$$

where $\mathbf{S}$ includes metric source terms (spacetime curvature).

### 6.2 HARM Code Formulation

The **HARM** (High Accuracy Relativistic Magnetohydrodynamics) code uses:

- Conservative variables: $\sqrt{-g} (\rho u^t, T^t_{\ i}, \sqrt{-g} B^i)$
- Flux-conservative scheme
- Flux-CT for divergence-free $\mathbf{B}$
- Primitive recovery in curved spacetime

**Metric:**
Modified Kerr-Schild coordinates (horizon-penetrating).

**Applications:**
- Black hole accretion disks (Sgr A*, M87)
- Neutron star mergers
- Jet launching simulations

### 6.3 Horizon Boundary Conditions

**Excision:**
- Remove points inside horizon (causally disconnected)
- Outflow boundary condition

**Inflow equilibrium:**
- Force matter to fall inward at inflow rate
- Maintain stationary disk profile

---

## 7. Numerical Methods for SRMHD

### 7.1 Godunov-Type Schemes

**HLLC flux:**

$$
\mathbf{F}_{\text{HLLC}} = \begin{cases}
\mathbf{F}_L & \lambda_L \geq 0 \\
\mathbf{F}_L^* & \lambda_L < 0 \leq \lambda_* \\
\mathbf{F}_R^* & \lambda_* < 0 \leq \lambda_R \\
\mathbf{F}_R & \lambda_R < 0
\end{cases}
$$

Star states $\mathbf{U}_L^*$, $\mathbf{U}_R^*$ from Rankine-Hugoniot jump conditions.

**Time stepping:**

- RK2 or RK3 (TVD)
- CFL condition:
$$
\Delta t \leq C \min_i \frac{\Delta x_i}{c_{\text{fast}, i} + |v_i|}
$$
where $C \sim 0.4-0.5$ (more restrictive than non-relativistic).

### 7.2 Adaptive Timestep

For high Lorentz factors ($W \gg 1$), local timestep can be tiny. Use:

- Hierarchical timestepping (AMR codes)
- Implicit-explicit (IMEX) schemes (stiff terms implicit)

### 7.3 Floors and Ceilings

**Density floor:**
$$
\rho \geq \rho_{\min} = 10^{-6} \rho_{\text{max}}
$$

**Temperature ceiling:**
$$
T \leq T_{\max} = 10^{13} \, \text{K} \quad \text{(avoid pair production)}
$$

**Magnetization ceiling:**
$$
\sigma \leq \sigma_{\max} \sim 100 \quad \text{(avoid numerical instability)}
$$

---

## 8. Python Implementation: 1D SRMHD Shock Tube

### 8.1 Problem Setup

Relativistic Brio-Wu test:
$$
(\rho, v_x, p, B_y) = \begin{cases}
(1.0, 0.0, 1.0, 1.0) & x < 0.5 \\
(0.125, 0.0, 0.1, -1.0) & x \geq 0.5
\end{cases}
$$
with $B_x = 0.5$ (constant), $\Gamma = 5/3$ (ideal gas).

Domain: $x \in [0, 1]$, $t \in [0, 0.4]$.

### 8.2 Code Structure

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
C = 1.0  # Speed of light
GAMMA = 5.0/3.0  # Adiabatic index

# Grid
NX = 400
XL, XR = 0.0, 1.0
dx = (XR - XL) / NX
x = np.linspace(XL + dx/2, XR - dx/2, NX)

# Primitive variables: [rho, vx, vy, vz, p, Bx, By, Bz]
def initial_conditions():
    prim = np.zeros((NX, 8))
    Bx_const = 0.5

    for i in range(NX):
        if x[i] < 0.5:
            prim[i] = [1.0, 0.0, 0.0, 0.0, 1.0, Bx_const, 1.0, 0.0]
        else:
            prim[i] = [0.125, 0.0, 0.0, 0.0, 0.1, Bx_const, -1.0, 0.0]

    return prim

# Lorentz factor
def lorentz_factor(vx, vy, vz):
    v2 = vx**2 + vy**2 + vz**2
    return 1.0 / np.sqrt(1.0 - v2 / C**2)

# Primitive to conserved
def prim2cons(prim):
    rho, vx, vy, vz, p = prim[:5]
    Bx, By, Bz = prim[5:]

    W = lorentz_factor(vx, vy, vz)
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2
    vdotB = vx*Bx + vy*By + vz*Bz

    # Specific enthalpy
    h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho

    # Comoving magnetic field
    b0 = W * vdotB / C
    bx = Bx/W + W*vx*vdotB/C**2
    by = By/W + W*vy*vdotB/C**2
    bz = Bz/W + W*vz*vdotB/C**2
    b2 = (B2 + (vdotB)**2/C**2) / W**2

    # Conserved variables
    D = rho * W
    Sx = (rho*h + b2)*W**2*vx - (vdotB)*Bx/(4.0*np.pi)
    Sy = (rho*h + b2)*W**2*vy - (vdotB)*By/(4.0*np.pi)
    Sz = (rho*h + b2)*W**2*vz - (vdotB)*Bz/(4.0*np.pi)
    tau = (rho*h + b2)*W**2 - p - b2/2.0 - D*C**2

    return np.array([D, Sx, Sy, Sz, tau, Bx, By, Bz])

# Conserved to primitive (simplified 1D Newton-Raphson)
def cons2prim(cons, prim_guess):
    D, Sx, Sy, Sz, tau = cons[:5]
    Bx, By, Bz = cons[5:]

    # Initial guess
    rho, vx, vy, vz, p = prim_guess[:5]

    # Newton-Raphson iteration (simplified)
    max_iter = 50
    tol = 1e-10

    for iteration in range(max_iter):
        W = lorentz_factor(vx, vy, vz)
        h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho

        vdotB = vx*Bx + vy*By + vz*Bz
        b2 = (Bx**2 + By**2 + Bz**2 + (vdotB)**2/C**2) / W**2

        # Residuals
        f1 = D - rho*W
        f2 = Sx - ((rho*h + b2)*W**2*vx - (vdotB)*Bx/(4.0*np.pi))
        f3 = tau - ((rho*h + b2)*W**2 - p - b2/2.0 - D*C**2)

        if abs(f1) + abs(f2) + abs(f3) < tol:
            break

        # Simple update (damped)
        rho = D / W
        p_new = (tau + D*C**2 + p + b2/2.0) / (W**2) - rho*h - b2
        p = 0.5 * p + 0.5 * max(p_new, 1e-10)

        # Update velocity (simplified)
        S2 = Sx**2 + Sy**2 + Sz**2
        S = np.sqrt(S2)
        if S > 1e-12:
            vx = Sx / (rho*h*W**2)
            vy = Sy / (rho*h*W**2)
            vz = Sz / (rho*h*W**2)

        # Limit velocity
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        if v >= C:
            scale = 0.99 * C / v
            vx *= scale
            vy *= scale
            vz *= scale

    # Apply floors
    rho = max(rho, 1e-10)
    p = max(p, 1e-10)

    return np.array([rho, vx, vy, vz, p, Bx, By, Bz])

# Flux function
def flux(prim):
    rho, vx, vy, vz, p = prim[:5]
    Bx, By, Bz = prim[5:]

    W = lorentz_factor(vx, vy, vz)
    vdotB = vx*Bx + vy*By + vz*Bz
    B2 = Bx**2 + By**2 + Bz**2
    b2 = (B2 + (vdotB)**2/C**2) / W**2
    h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho

    ptot = p + B2/(8.0*np.pi)

    F = np.zeros(8)
    F[0] = rho * W * vx
    F[1] = (rho*h + b2)*W**2*vx*vx + ptot - Bx**2/(4.0*np.pi) - (vdotB)*Bx*vx/(4.0*np.pi)
    F[2] = (rho*h + b2)*W**2*vx*vy - Bx*By/(4.0*np.pi) - (vdotB)*Bx*vy/(4.0*np.pi)
    F[3] = (rho*h + b2)*W**2*vx*vz - Bx*Bz/(4.0*np.pi) - (vdotB)*Bx*vz/(4.0*np.pi)
    F[4] = ((rho*h + b2)*W**2 - p - b2/2.0)*vx + ptot*vx - (vdotB)*Bx/(4.0*np.pi)
    F[5] = 0.0  # Bx constant
    F[6] = By*vx - Bx*vy
    F[7] = Bz*vx - Bx*vz

    return F

# HLLC Riemann solver (simplified HLL for SRMHD)
def hll_flux(pL, pR):
    # Estimate wave speeds (very simplified)
    rhoL, vxL, pL_val = pL[0], pL[1], pL[4]
    rhoR, vxR, pR_val = pR[0], pR[1], pR[4]

    # Sound speed
    csL = np.sqrt(GAMMA * pL_val / (rhoL * (1.0 + GAMMA/(GAMMA-1.0)*pL_val/rhoL)))
    csR = np.sqrt(GAMMA * pR_val / (rhoR * (1.0 + GAMMA/(GAMMA-1.0)*pR_val/rhoR)))

    WL = lorentz_factor(pL[1], pL[2], pL[3])
    WR = lorentz_factor(pR[1], pR[2], pR[3])

    # Fast magnetosonic speed estimate (crude)
    BxL, ByL, BzL = pL[5:]
    BxR, ByR, BzR = pR[5:]
    B2L = BxL**2 + ByL**2 + BzL**2
    B2R = BxR**2 + ByR**2 + BzR**2

    hL = 1.0 + GAMMA/(GAMMA-1.0)*pL_val/rhoL
    hR = 1.0 + GAMMA/(GAMMA-1.0)*pR_val/rhoR

    vAL = np.sqrt(B2L/(4.0*np.pi*(rhoL*hL + B2L/(4.0*np.pi))))
    vAR = np.sqrt(B2R/(4.0*np.pi*(rhoR*hR + B2R/(4.0*np.pi))))

    cfL = np.sqrt((csL**2 + vAL**2 - csL**2*vAL**2)/(1.0 - csL**2*vAL**2))
    cfR = np.sqrt((csR**2 + vAR**2 - csR**2*vAR**2)/(1.0 - csR**2*vAR**2))

    # Wave speeds (relativistic addition)
    lamL = (vxL - cfL) / (1.0 - vxL*cfL/C**2)
    lamR = (vxR + cfR) / (1.0 + vxR*cfR/C**2)

    # HLL flux
    consL = prim2cons(pL)
    consR = prim2cons(pR)
    FL = flux(pL)
    FR = flux(pR)

    if lamL >= 0:
        return FL
    elif lamR <= 0:
        return FR
    else:
        F_hll = (lamR*FL - lamL*FR + lamL*lamR*(consR - consL)) / (lamR - lamL)
        return F_hll

# Main evolution
def evolve_srmhd():
    prim = initial_conditions()
    cons = np.array([prim2cons(p) for p in prim])

    t = 0.0
    t_end = 0.4
    CFL = 0.4

    snapshots = []

    while t < t_end:
        # Compute dt
        v_max = 0.0
        for i in range(NX):
            rho, vx, vy, vz, p = prim[i, :5]
            Bx, By, Bz = prim[i, 5:]

            cs = np.sqrt(GAMMA * p / (rho * (1.0 + GAMMA/(GAMMA-1.0)*p/rho)))
            B2 = Bx**2 + By**2 + Bz**2
            h = 1.0 + GAMMA/(GAMMA-1.0)*p/rho
            vA = np.sqrt(B2/(4.0*np.pi*(rho*h + B2/(4.0*np.pi))))
            cf = np.sqrt((cs**2 + vA**2)/(1.0 + cs**2*vA**2))

            W = lorentz_factor(vx, vy, vz)
            v_signal = max(abs((vx + cf)/(1.0 + vx*cf/C**2)),
                          abs((vx - cf)/(1.0 - vx*cf/C**2)))
            v_max = max(v_max, v_signal)

        dt = CFL * dx / v_max
        if t + dt > t_end:
            dt = t_end - t

        # RK2 time integration
        # Stage 1
        flux_arr = np.zeros((NX+1, 8))
        for i in range(NX+1):
            if i == 0:
                flux_arr[i] = flux(prim[0])
            elif i == NX:
                flux_arr[i] = flux(prim[-1])
            else:
                flux_arr[i] = hll_flux(prim[i-1], prim[i])

        cons_1 = cons.copy()
        for i in range(NX):
            cons_1[i] -= dt/dx * (flux_arr[i+1] - flux_arr[i])

        # Recover primitives
        prim_1 = np.array([cons2prim(cons_1[i], prim[i]) for i in range(NX)])

        # Stage 2
        for i in range(NX+1):
            if i == 0:
                flux_arr[i] = flux(prim_1[0])
            elif i == NX:
                flux_arr[i] = flux(prim_1[-1])
            else:
                flux_arr[i] = hll_flux(prim_1[i-1], prim_1[i])

        cons_2 = cons_1.copy()
        for i in range(NX):
            cons_2[i] -= dt/dx * (flux_arr[i+1] - flux_arr[i])

        cons = 0.5 * (cons + cons_2)
        prim = np.array([cons2prim(cons[i], prim_1[i]) for i in range(NX)])

        t += dt

        if len(snapshots) < 5:
            if t >= len(snapshots) * t_end / 4:
                snapshots.append((t, prim.copy()))

    return prim, snapshots

# Run simulation
print("Running SRMHD shock tube...")
prim_final, snapshots = evolve_srmhd()

# Plot results
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

vars_to_plot = [
    ('Density', 0),
    ('Velocity vx', 1),
    ('Pressure', 4),
    ('By', 6),
    ('Lorentz Factor', None),
    ('Energy Density', None)
]

for idx, (var_name, var_idx) in enumerate(vars_to_plot):
    ax = axes[idx // 2, idx % 2]

    if var_idx is not None:
        ax.plot(x, prim_final[:, var_idx], 'b-', linewidth=1.5, label='t=0.4')
    else:
        if 'Lorentz' in var_name:
            W = np.array([lorentz_factor(p[1], p[2], p[3]) for p in prim_final])
            ax.plot(x, W, 'b-', linewidth=1.5)
        elif 'Energy' in var_name:
            energy = prim_final[:, 4] / (GAMMA - 1.0)
            ax.plot(x, energy, 'b-', linewidth=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel(var_name)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{var_name} at t=0.4')

plt.tight_layout()
plt.savefig('srmhd_shock_tube.png', dpi=150, bbox_inches='tight')
print("Plot saved: srmhd_shock_tube.png")
plt.close()

# Plot wave structure
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['red', 'orange', 'green', 'blue']
for i, (t, prim_snap) in enumerate(snapshots):
    ax.plot(x, prim_snap[:, 0], color=colors[i], label=f't={t:.2f}', alpha=0.7)

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('SRMHD Shock Tube: Density Evolution')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('srmhd_evolution.png', dpi=150, bbox_inches='tight')
print("Plot saved: srmhd_evolution.png")
plt.close()
```

### 8.3 Expected Output

The code produces:
- Density jump at shock
- Rarefaction fan
- Contact discontinuity
- Magnetic field reversal
- Lorentz factor peaks in jet-like features

**Challenges:**
- Primitive recovery can fail if initial guess poor
- Damping needed in Newton iteration
- Floors prevent unphysical states

---

## 9. Relativistic Alfvén Speed

### 9.1 Comparison: Non-Relativistic vs Relativistic

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
rho = 1.0
p = 1.0
B_range = np.logspace(-2, 2, 100)
GAMMA = 5.0/3.0

# Non-relativistic Alfven speed
vA_nr = B_range / np.sqrt(4.0 * np.pi * rho)

# Relativistic Alfven speed
h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho
vA_r = B_range / np.sqrt(4.0*np.pi*(rho*h + B_range**2/(4.0*np.pi)))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(B_range, vA_nr, 'b-', linewidth=2, label='Non-relativistic')
ax.loglog(B_range, vA_r, 'r--', linewidth=2, label='Relativistic')
ax.axhline(1.0, color='k', linestyle=':', label='Speed of light c=1')
ax.set_xlabel('Magnetic Field B')
ax.set_ylabel('Alfvén Speed $v_A$')
ax.set_title('Alfvén Speed: Non-Relativistic vs Relativistic')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('alfven_speed_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved: alfven_speed_comparison.png")
plt.close()
```

**Key observation:**
- Non-relativistic $v_A$ can exceed $c$ for large $B$ (unphysical)
- Relativistic $v_A < c$ always (saturates as $B \to \infty$)

---

## 10. Light Cylinder Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Pulsar parameters
R_NS = 1.0  # Neutron star radius
Omega = 1.0  # Angular velocity
c = 1.0
R_L = c / Omega  # Light cylinder radius

# Grid
theta = np.linspace(0, 2*np.pi, 100)
r = np.linspace(0.5*R_NS, 3*R_L, 100)
R, Theta = np.meshgrid(r, theta)

# Corotation velocity
v_corot = Omega * R

# Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Velocity field (normalized)
v_norm = np.clip(v_corot / c, 0, 1.5)
cf = ax.contourf(Theta, R, v_norm, levels=20, cmap='RdYlBu_r')

# Light cylinder
ax.plot(theta, R_L * np.ones_like(theta), 'k--', linewidth=2, label='Light Cylinder')

# Neutron star
ax.fill_between(theta, 0, R_NS, color='gray', alpha=0.5, label='Neutron Star')

ax.set_ylim(0, 3*R_L)
ax.set_title('Pulsar Magnetosphere: Corotation Velocity\n(Dashed = Light Cylinder)', pad=20)
plt.colorbar(cf, ax=ax, label='$v_{corot}/c$')
ax.legend(loc='upper right')
plt.savefig('light_cylinder.png', dpi=150, bbox_inches='tight')
print("Plot saved: light_cylinder.png")
plt.close()
```

Inside $R_L$: corotation possible ($v < c$).
Outside $R_L$: field lines open, particles escape (pulsar wind).

---

## Summary

Relativistic MHD extends classical MHD to regimes where $v \sim c$:

1. **SRMHD formulation:** Covariant 4-tensor approach, stress-energy tensor, frozen-in condition
2. **3+1 decomposition:** Lab-frame conserved variables for numerical implementation
3. **Primitive recovery:** Nonlinear implicit solve (major numerical challenge)
4. **Wave structure:** 7 waves, relativistic dispersion (all speeds < $c$)
5. **Applications:** Relativistic jets, pulsar magnetospheres, black hole accretion
6. **GRMHD:** Curved spacetime, Kerr metric, horizon-penetrating coordinates
7. **Numerical methods:** HLLC/HLLD Riemann solvers, adaptive timestepping, floors

**Key challenges:**
- Primitive recovery robustness
- Handling high Lorentz factors ($W \gg 1$)
- Magnetization parameter $\sigma$ spanning many decades
- Coupling to radiation, pair production (beyond ideal MHD)

Relativistic MHD is essential for understanding the most energetic phenomena in the universe: jets, pulsars, black holes, and neutron star mergers.

---

## Practice Problems

1. **Lorentz Transformations:**
   Given $\mathbf{E} = (1, 0, 0)$ and $\mathbf{B} = (0, 1, 0)$ in the lab frame, compute $\mathbf{E}'$ and $\mathbf{B}'$ in a frame moving with $\mathbf{v} = (0.5c, 0, 0)$. Verify $\mathbf{E}' \cdot \mathbf{B}' = \mathbf{E} \cdot \mathbf{B}$ (Lorentz invariant).

2. **Primitive Recovery:**
   Implement a 1D Newton-Raphson primitive recovery routine. Test on: $D = 2.0$, $S_x = 1.0$, $\tau = 3.0$, $B_x = 0.5$, $B_y = 1.0$. Initial guess: $\rho = 1.0$, $v_x = 0.3$, $p = 1.0$. Does it converge? How many iterations?

3. **Relativistic Brio-Wu:**
   Modify the shock tube code to use $B_x = 0$ (purely transverse field). Compare wave speeds and structure to the $B_x = 0.5$ case. Why does the solution differ?

4. **Magnetization Parameter:**
   For a pulsar with $B = 10^{12}$ G, $\rho = 10^7$ g/cm³, $\Gamma = 10$, compute $\sigma = B^2/(4\pi \rho h W^2 c^2)$. Assume $h \approx 1$. Is this force-free ($\sigma \gg 1$) or fluid-dominated?

5. **Light Cylinder:**
   For the Crab pulsar ($P = 33$ ms), compute $R_L$. At what radius does the corotation velocity equal $c$? Compare to the pulsar radius $R_{NS} \sim 10$ km.

6. **ISCO Velocity:**
   For a Schwarzschild black hole, compute the orbital velocity at the ISCO ($r = 6GM/c^2$). What is the Lorentz factor $W$? Repeat for a Kerr black hole with $a = 0.998$ (nearly extremal) at the prograde ISCO.

7. **Alfvén Speed Saturation:**
   Plot the relativistic Alfvén speed $v_A = B/\sqrt{4\pi(\rho h + B^2/(4\pi))}$ vs $B$ for fixed $\rho = 1$, $p = 0.1$, $\Gamma = 4/3$. Show that $v_A \to c$ as $B \to \infty$ but never exceeds $c$.

8. **HARM Timestep:**
   In GRMHD near a black hole horizon, the lapse function $\alpha \to 0$. Why does this require smaller timesteps? Estimate the CFL timestep near $r = 2.01 GM/c^2$ for a radial grid with $\Delta r = 0.01 GM/c^2$.

---

**Previous:** [2D MHD Solver](./15_2D_MHD_Solver.md) | **Next:** [Spectral and Advanced Methods](./17_Spectral_Methods.md)

# 13. Two-Fluid Model

## Learning Objectives

- Derive fluid equations from kinetic theory by taking velocity-space moments of the Vlasov equation
- Understand the closure problem and various closure approximations (isothermal, adiabatic, CGL)
- Derive the generalized Ohm's law from the electron momentum equation and analyze each term's physical significance
- Explain the Hall effect and its role in decoupling ions from the magnetic field at small scales
- Distinguish between particle drifts and fluid drifts, particularly the diamagnetic drift
- Apply two-fluid theory to understand wave phenomena beyond single-fluid MHD

## 1. From Vlasov to Fluid Equations

### 1.1 The Moment Hierarchy

The Vlasov equation describes the evolution of the distribution function $f_s(\mathbf{r}, \mathbf{v}, t)$ for species $s$:

$$\frac{\partial f_s}{\partial t} + \mathbf{v} \cdot \nabla f_s + \frac{q_s}{m_s}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \frac{\partial f_s}{\partial \mathbf{v}} = \left(\frac{\partial f_s}{\partial t}\right)_{\text{coll}}$$

While the Vlasov equation contains complete information about the plasma, it is a 6D partial differential equation that is computationally expensive to solve. For many applications, we don't need the full distribution function—we only care about macroscopic quantities like density, flow velocity, and pressure.

The **method of moments** reduces the dimensionality by integrating the Vlasov equation over velocity space with different weights. The $n$-th moment is obtained by multiplying the Vlasov equation by $v^n$ and integrating:

$$\int (\text{Vlasov equation}) \times (\text{weight function}) \, d^3v$$

This generates a hierarchy of fluid equations, where each equation involves the next higher-order moment.

### 1.2 Zeroth Moment: Continuity Equation

The zeroth moment (weight = 1) gives the **continuity equation**:

$$\int \frac{\partial f_s}{\partial t} d^3v + \int \mathbf{v} \cdot \nabla f_s d^3v + \int \frac{q_s}{m_s}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \frac{\partial f_s}{\partial \mathbf{v}} d^3v = 0$$

The number density is:
$$n_s(\mathbf{r}, t) = \int f_s(\mathbf{r}, \mathbf{v}, t) d^3v$$

For the first term:
$$\int \frac{\partial f_s}{\partial t} d^3v = \frac{\partial}{\partial t} \int f_s d^3v = \frac{\partial n_s}{\partial t}$$

For the second term, using the divergence theorem in velocity space:
$$\int \mathbf{v} \cdot \nabla f_s d^3v = \nabla \cdot \int \mathbf{v} f_s d^3v = \nabla \cdot (n_s \mathbf{u}_s)$$

where the mean flow velocity is:
$$\mathbf{u}_s = \frac{1}{n_s} \int \mathbf{v} f_s d^3v$$

For the third term, the Lorentz force term vanishes because:
$$\int \frac{\partial f_s}{\partial \mathbf{v}} d^3v = [f_s]_{v=-\infty}^{v=+\infty} = 0$$

(assuming $f_s \to 0$ as $|\mathbf{v}| \to \infty$).

The result is the **continuity equation**:

$$\boxed{\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{u}_s) = 0}$$

This is conservation of particles. In the presence of ionization/recombination, a source term would appear on the right-hand side.

### 1.3 First Moment: Momentum Equation

The first moment (weight = $m_s \mathbf{v}$) gives the **momentum equation**. We multiply the Vlasov equation by $m_s \mathbf{v}$ and integrate:

Define the momentum density:
$$\mathbf{p}_s = m_s n_s \mathbf{u}_s = m_s \int \mathbf{v} f_s d^3v$$

The peculiar velocity (thermal velocity) is:
$$\mathbf{w} = \mathbf{v} - \mathbf{u}_s$$

The pressure tensor is:
$$\overleftrightarrow{P}_s = m_s \int \mathbf{w} \mathbf{w} f_s d^3v$$

After considerable algebra (using integration by parts and the divergence theorem), the momentum equation becomes:

$$\boxed{m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla \cdot \overleftrightarrow{P}_s + \mathbf{R}_s}$$

where $d/dt = \partial/\partial t + \mathbf{u}_s \cdot \nabla$ is the convective derivative, and $\mathbf{R}_s$ is the momentum transfer from collisions with other species.

This is Newton's second law for a fluid element:
- **LHS**: mass × acceleration
- **RHS**: Lorentz force + pressure gradient force + collision force

**Key point**: This equation introduces a new quantity, the pressure tensor $\overleftrightarrow{P}_s$, which is a second-order moment of the distribution function.

### 1.4 Second Moment: Energy Equation

The second moment (weight = $\frac{1}{2} m_s v^2$) gives the **energy equation**:

Define the thermal energy density:
$$\mathcal{E}_s = \frac{1}{2} m_s \int w^2 f_s d^3v$$

For an isotropic pressure ($\overleftrightarrow{P}_s = p_s \overleftrightarrow{I}$), we have:
$$p_s = \frac{1}{3} m_s \int w^2 f_s d^3v = \frac{2}{3} \mathcal{E}_s$$

The energy equation becomes:

$$\frac{\partial \mathcal{E}_s}{\partial t} + \nabla \cdot (\mathcal{E}_s \mathbf{u}_s) = -p_s \nabla \cdot \mathbf{u}_s - \nabla \cdot \mathbf{q}_s + Q_s$$

where:
- $\mathbf{q}_s = \frac{1}{2} m_s \int w^2 \mathbf{w} f_s d^3v$ is the heat flux vector (third-order moment)
- $Q_s$ is the collisional energy transfer

Using $p_s = \frac{2}{3} \mathcal{E}_s$, this can be rewritten as:

$$\frac{3}{2} \frac{d p_s}{dt} + \frac{5}{2} p_s \nabla \cdot \mathbf{u}_s = -\nabla \cdot \mathbf{q}_s + Q_s$$

**The closure problem**: The energy equation introduces the heat flux $\mathbf{q}_s$, a third-order moment. If we took the third moment, we'd get an equation involving a fourth-order moment, and so on. This infinite hierarchy must be **closed** by making an assumption about the highest-order moment.

### 1.5 The Closure Problem

```
Moment hierarchy:

0th moment:  ∂n/∂t + ∇·(nu) = 0              (introduces u)
1st moment:  mn(du/dt) = qn(E+u×B) - ∇·P + R  (introduces P)
2nd moment:  dp/dt = -p∇·u - ∇·q + Q          (introduces q)
3rd moment:  ...                              (introduces next moment)
...

Each equation introduces a new unknown from the next higher moment.
This is the CLOSURE PROBLEM.
```

We need to **truncate** the hierarchy by assuming a relationship between the highest moment and lower moments. Common closures:

**1. Isothermal closure**: Assume constant temperature
$$p_s = n_s k_B T_s, \quad T_s = \text{const}$$

This is valid when heat conduction is very efficient, so temperature equilibrates instantly.

**2. Adiabatic closure**: Assume no heat flux ($\mathbf{q}_s = 0$) and adiabatic evolution
$$\frac{d}{dt}\left( \frac{p_s}{n_s^\gamma} \right) = 0$$

where $\gamma$ is the adiabatic index ($\gamma = 5/3$ for a monatomic gas). This is valid for rapid processes where heat conduction is negligible.

**3. CGL closure** (Chew-Goldberger-Low): For collisionless magnetized plasmas, pressure is anisotropic:
$$\overleftrightarrow{P}_s = p_{\perp s} \overleftrightarrow{I} + (p_{\parallel s} - p_{\perp s}) \hat{\mathbf{b}} \hat{\mathbf{b}}$$

with double adiabatic equations:
$$\frac{d}{dt}\left( \frac{p_{\perp s}}{n_s B} \right) = 0, \quad \frac{d}{dt}\left( \frac{p_{\parallel s} B^2}{n_s^3} \right) = 0$$

We'll discuss CGL in Lesson 14.

### 1.6 Two-Fluid Equations Summary

For each species (electrons $e$, ions $i$), we have:

**Continuity**:
$$\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{u}_s) = 0$$

**Momentum**:
$$m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla p_s + \mathbf{R}_s$$

(assuming isotropic pressure)

**Energy** (with adiabatic closure):
$$\frac{d}{dt}\left( \frac{p_s}{n_s^\gamma} \right) = 0$$

These are coupled to **Maxwell's equations**:
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$
$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}, \quad \nabla \cdot \mathbf{B} = 0$$

where the charge and current densities are:
$$\rho = \sum_s q_s n_s, \quad \mathbf{J} = \sum_s q_s n_s \mathbf{u}_s$$

The collision terms $\mathbf{R}_s$ couple the species. For electron-ion collisions:
$$\mathbf{R}_e = -\mathbf{R}_i = -\frac{m_e n_e}{\tau_{ei}} (\mathbf{u}_e - \mathbf{u}_i)$$

where $\tau_{ei}$ is the electron-ion collision time.

## 2. Generalized Ohm's Law

### 2.1 Derivation from Electron Momentum Equation

One of the most important results from two-fluid theory is the **generalized Ohm's law**, which relates the electric field to the current. In ideal MHD, we have the simple form:
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0$$

But this is a severe approximation. Let's derive the full form from the electron momentum equation.

Starting with:
$$m_e n_e \frac{d \mathbf{u}_e}{dt} = -e n_e (\mathbf{E} + \mathbf{u}_e \times \mathbf{B}) - \nabla p_e + \mathbf{R}_e$$

The collision term can be written as:
$$\mathbf{R}_e = -\frac{m_e n_e}{\tau_{ei}} (\mathbf{u}_e - \mathbf{u}_i) \approx -\frac{m_e n_e \mathbf{u}_e}{\tau_{ei}}$$

(assuming $u_e \gg u_i$ for current-carrying electrons).

Rearranging:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = \frac{m_e}{e \tau_{ei}} \mathbf{u}_e - \frac{1}{e n_e} \nabla p_e + \frac{m_e}{e n_e} \frac{d \mathbf{u}_e}{dt}$$

Now, express everything in terms of the **current density** $\mathbf{J}$ and the **center-of-mass velocity** $\mathbf{v}$.

Define:
$$\mathbf{J} = -e n_e \mathbf{u}_e + e n_i \mathbf{u}_i \approx -e n_e (\mathbf{u}_e - \mathbf{u}_i)$$
$$\mathbf{v} = \frac{m_i n_i \mathbf{u}_i + m_e n_e \mathbf{u}_e}{m_i n_i + m_e n_e} \approx \mathbf{u}_i$$

(using $m_i \gg m_e$ and quasi-neutrality $n_e \approx n_i \equiv n$).

From the current definition:
$$\mathbf{u}_e = \mathbf{u}_i - \frac{\mathbf{J}}{e n} \approx \mathbf{v} - \frac{\mathbf{J}}{e n}$$

Substituting into the rearranged electron equation:

$$\mathbf{E} + \left( \mathbf{v} - \frac{\mathbf{J}}{en} \right) \times \mathbf{B} = \frac{m_e}{e^2 n \tau_{ei}} \mathbf{J} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n} \frac{d}{dt}\left( -\frac{\mathbf{J}}{e n} \right)$$

Simplifying the cross product:
$$\mathbf{u}_e \times \mathbf{B} = \mathbf{v} \times \mathbf{B} - \frac{\mathbf{J} \times \mathbf{B}}{en}$$

This gives the **generalized Ohm's law**:

$$\boxed{\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{en} \mathbf{J} \times \mathbf{B} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}}$$

where the **resistivity** is:
$$\eta = \frac{m_e}{e^2 n \tau_{ei}}$$

### 2.2 Physical Interpretation of Each Term

Let's identify each term on the RHS:

1. **Resistive term**: $\eta \mathbf{J}$
   - Ohmic dissipation due to electron-ion collisions
   - Causes magnetic diffusion (resistive MHD)
   - $\eta \sim T_e^{-3/2}$ (decreases with temperature)

2. **Hall term**: $\frac{1}{en} \mathbf{J} \times \mathbf{B}$
   - Decoupling of ions from magnetic field
   - Important at scales $\sim$ ion skin depth $d_i = c/\omega_{pi}$
   - Enables fast magnetic reconnection

3. **Electron pressure term**: $-\frac{1}{en} \nabla p_e$
   - Pressure gradient drives current even without E field
   - Important in steep gradient regions (e.g., current sheets)

4. **Electron inertia term**: $\frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$
   - Important at electron skin depth $d_e = c/\omega_{pe}$
   - Relevant for very fast phenomena (whistler waves, reconnection)

### 2.3 Scale Analysis: When Does Each Term Matter?

Let's perform an **order-of-magnitude analysis** to determine when each term is important.

Define characteristic scales:
- Length: $L$
- Velocity: $V$
- Magnetic field: $B_0$
- Density: $n_0$
- Current: $J_0 \sim B_0/(\mu_0 L)$ (from Ampère's law)

**Ideal MHD term** (LHS):
$$\mathbf{v} \times \mathbf{B} \sim V B_0$$

**Resistive term**:
$$\eta \mathbf{J} \sim \eta \frac{B_0}{\mu_0 L}$$

Ratio:
$$\frac{\eta J}{\mathbf{v} \times \mathbf{B}} \sim \frac{\eta}{\mu_0 V L} = \frac{1}{R_m}$$

where $R_m = \mu_0 V L / \eta$ is the **magnetic Reynolds number**. Resistivity is important when $R_m \lesssim 1$.

**Hall term**:
$$\frac{\mathbf{J} \times \mathbf{B}}{en} \sim \frac{B_0^2}{\mu_0 e n_0 L}$$

Ratio:
$$\frac{J \times B / en}{\mathbf{v} \times \mathbf{B}} \sim \frac{B_0}{\mu_0 e n_0 V L} = \frac{V_A}{V} \frac{d_i}{L}$$

where $d_i = c/\omega_{pi} = \sqrt{m_i / (\mu_0 e^2 n_0)}$ is the **ion skin depth** and $V_A = B_0/\sqrt{\mu_0 m_i n_0}$ is the Alfvén speed.

The Hall term is important when $L \lesssim d_i$ or when $V \lesssim V_A$ at ion scales.

**Electron pressure term**:
$$\frac{\nabla p_e}{en} \sim \frac{k_B T_e}{eL}$$

Ratio:
$$\frac{\nabla p_e / en}{\mathbf{v} \times \mathbf{B}} \sim \frac{k_B T_e}{e V B_0 L} = \frac{v_{te}^2}{V^2} \frac{\rho_e}{L}$$

where $v_{te} = \sqrt{k_B T_e / m_e}$ is the electron thermal speed and $\rho_e = v_{te}/\omega_{ce}$ is the electron gyroradius.

This term is important in regions of steep pressure gradients.

**Electron inertia term**:
$$\frac{m_e}{e^2 n^2} \frac{dJ}{dt} \sim \frac{m_e}{e^2 n_0^2} \frac{B_0}{\mu_0 L} \frac{V}{L} = \frac{m_e V B_0}{\mu_0 e^2 n_0 L^2}$$

Ratio:
$$\frac{m_e dJ/dt / (e^2 n^2)}{v \times B} \sim \frac{m_e}{\mu_0 e^2 n_0 L^2} = \frac{d_e^2}{L^2}$$

where $d_e = c/\omega_{pe}$ is the **electron skin depth**.

This term is important when $L \lesssim d_e$.

**Summary**:
```
Term                Scale               When important
----------------    -----------------   ------------------------
Resistive           1/R_m               R_m ~ 1 (low T, small L)
Hall                d_i/L               L ~ d_i (ion scales)
Electron pressure   β_e ρ_e/L           Steep gradients
Electron inertia    (d_e/L)^2           L ~ d_e (electron scales)

Typical ordering: d_e << ρ_e << d_i << L (MHD)
```

### 2.4 Limiting Cases

**Ideal MHD** ($R_m \to \infty$, $L \gg d_i$):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0$$

Magnetic field is frozen into the fluid.

**Resistive MHD** (keep resistive term, drop others):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J}$$

Allows magnetic reconnection, but slow (Sweet-Parker rate).

**Hall MHD** (keep Hall term, drop resistive/inertia):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \frac{1}{en} \mathbf{J} \times \mathbf{B}$$

Enables fast reconnection (Petschek rate), whistler waves.

**Electron MHD** (keep Hall + inertia, drop resistive):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \frac{1}{en} \mathbf{J} \times \mathbf{B} + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$$

Relevant at electron scales (e.g., reconnection diffusion region).

## 3. The Hall Effect

### 3.1 Physics of the Hall Term

The Hall term $\frac{1}{en} \mathbf{J} \times \mathbf{B}$ arises from the difference in electron and ion motions. When a current flows across a magnetic field, electrons and ions experience different Lorentz forces, creating a charge separation and thus an **electric field perpendicular to both $\mathbf{J}$ and $\mathbf{B}$**.

Consider a current $\mathbf{J} = J_x \hat{\mathbf{x}}$ in a magnetic field $\mathbf{B} = B_0 \hat{\mathbf{z}}$:

$$\mathbf{J} \times \mathbf{B} = J_x B_0 \hat{\mathbf{y}}$$

This creates an electric field:
$$E_y = \frac{J_x B_0}{en}$$

This is the **Hall electric field**.

### 3.2 Hall Parameter

The **Hall parameter** quantifies the importance of the magnetic field:

$$\Omega_s \tau_s = \omega_{cs} \tau_{cs}$$

where $\omega_{cs} = q_s B / m_s$ is the cyclotron frequency and $\tau_{cs}$ is the collision time.

- When $\Omega_s \tau_s \ll 1$: collisions dominate, particle orbits are interrupted before completing a gyration → **unmagnetized**
- When $\Omega_s \tau_s \gg 1$: particles complete many gyrations between collisions → **magnetized**

For electrons in typical plasmas, $\Omega_e \tau_e \gg 1$ (strongly magnetized).
For ions, $\Omega_i \tau_i$ can vary (weakly magnetized in collisional plasmas, strongly magnetized in hot fusion plasmas).

### 3.3 Decoupling of Ions from Magnetic Field

At scales larger than the ion skin depth $d_i$, both electrons and ions are frozen to the magnetic field (ideal MHD). But at scales $L \lesssim d_i$, the Hall term becomes important, and **ions decouple from the magnetic field**.

To see this, consider the ion and electron momentum equations:

**Ions**:
$$m_i n \frac{d \mathbf{u}_i}{dt} = e n (\mathbf{E} + \mathbf{u}_i \times \mathbf{B}) - \nabla p_i$$

**Electrons** (from generalized Ohm's law, keeping only Hall term):
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} \approx \frac{1}{en} \mathbf{J} \times \mathbf{B}$$

Using $\mathbf{J} = en(\mathbf{u}_i - \mathbf{u}_e)$:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = \frac{1}{en} en (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B}$$

Rearranging:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B}$$
$$\mathbf{E} + \mathbf{u}_i \times \mathbf{B} = 0$$

So **electrons** satisfy the frozen-in condition:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = 0$$

But **ions** do not! They experience an electric field:
$$\mathbf{E} = -\mathbf{u}_i \times \mathbf{B} + (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B} = \mathbf{u}_e \times \mathbf{B} \neq -\mathbf{u}_i \times \mathbf{B}$$

This means the magnetic field is **frozen into the electron fluid**, not the ion fluid, at scales $\sim d_i$.

### 3.4 Hall MHD Waves

Including the Hall term modifies MHD wave dispersion. The key change is the appearance of **whistler waves** at high frequencies.

The Hall MHD dispersion relation (in the low-frequency, small-amplitude limit) gives:

**Alfvén/whistler branch**:
$$\omega = k_\parallel V_A \sqrt{1 + k^2 d_i^2}$$

- At $k d_i \ll 1$ (large scales): $\omega \approx k_\parallel V_A$ (Alfvén wave)
- At $k d_i \gg 1$ (small scales): $\omega \approx k_\parallel V_A k d_i = k \sqrt{k_\parallel V_A d_i}$ (whistler)

Whistler waves have:
- **Right-hand circular polarization** (in ion frame)
- **Dispersive**: phase velocity increases with $k$
- **No ion motion**: only electrons respond

We'll compute this dispersion relation in the Python code below.

## 4. Diamagnetic Drift

### 4.1 Particle vs. Fluid Drifts

In Lesson 3, we derived **particle drifts** from single-particle orbit theory:

$$\mathbf{v}_E = \frac{\mathbf{E} \times \mathbf{B}}{B^2}, \quad \mathbf{v}_{\nabla B} = \frac{m v_\perp^2}{2 q B^3} \mathbf{B} \times \nabla B, \quad \text{etc.}$$

These are drifts of individual particles.

In fluid theory, we have **fluid drifts** that emerge from pressure gradients and other collective effects. The most important is the **diamagnetic drift**.

### 4.2 Derivation of Diamagnetic Drift

Consider the momentum equation in a magnetized plasma with a pressure gradient perpendicular to $\mathbf{B}$:

$$m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla p_s$$

In equilibrium ($d\mathbf{u}_s/dt = 0$) with no electric field ($\mathbf{E} = 0$):

$$0 = q_s n_s \mathbf{u}_s \times \mathbf{B} - \nabla p_s$$

Taking the cross product with $\mathbf{B}$:

$$q_s n_s (\mathbf{u}_s \times \mathbf{B}) \times \mathbf{B} = -\nabla p_s \times \mathbf{B}$$

Using the vector identity $(\mathbf{A} \times \mathbf{B}) \times \mathbf{C} = \mathbf{B}(\mathbf{A} \cdot \mathbf{C}) - \mathbf{A}(\mathbf{B} \cdot \mathbf{C})$:

$$q_s n_s [\mathbf{B} (\mathbf{u}_s \cdot \mathbf{B}) - \mathbf{u}_s B^2] = -\nabla p_s \times \mathbf{B}$$

If the flow is perpendicular to $\mathbf{B}$ (i.e., $\mathbf{u}_s \cdot \mathbf{B} = 0$):

$$\mathbf{u}_s = \frac{\nabla p_s \times \mathbf{B}}{q_s n_s B^2} = -\frac{\mathbf{B} \times \nabla p_s}{q_s n_s B^2}$$

This is the **diamagnetic drift velocity**:

$$\boxed{\mathbf{v}_{*s} = -\frac{\mathbf{B} \times \nabla p_s}{q_s n_s B^2}}$$

For electrons ($q_e = -e$):
$$\mathbf{v}_{*e} = \frac{\mathbf{B} \times \nabla p_e}{e n_e B^2}$$

For ions ($q_i = +e$):
$$\mathbf{v}_{*i} = -\frac{\mathbf{B} \times \nabla p_i}{e n_i B^2}$$

### 4.3 Diamagnetic Current

The **diamagnetic current** is:

$$\mathbf{J}_* = \sum_s q_s n_s \mathbf{v}_{*s} = -\frac{\mathbf{B} \times \nabla p_e}{B^2} - \frac{\mathbf{B} \times \nabla p_i}{B^2} = \frac{\mathbf{B} \times \nabla p}{B^2}$$

where $p = p_e + p_i$ is the total pressure.

This can also be written as:
$$\mathbf{J}_* = -\nabla p \times \frac{\mathbf{B}}{B^2}$$

**Key point**: The diamagnetic drift is **not a particle drift**! If you solve for individual particle orbits, you won't find this drift. It arises from the **spatial variation of the distribution function** due to pressure gradients.

To see this, note that the diamagnetic drift velocity depends on the gradient scale length $L_p = p / |\nabla p|$:

$$v_* \sim \frac{p}{q n B L_p} = \frac{k_B T}{q B L_p} \sim \frac{\rho}{L_p} v_{th}$$

where $\rho = v_{th}/\omega_c$ is the gyroradius.

In a collisionless plasma, particles on different gyro-orbits have different densities, creating a net drift when averaged over a distribution.

### 4.4 Physical Interpretation: Magnetization Current

The diamagnetic current can be understood as a **magnetization current** arising from the magnetic moments of gyrating particles.

The magnetization is:
$$\mathbf{M} = -n \mu \frac{\mathbf{B}}{B}$$

where $\mu = m v_\perp^2 / (2B)$ is the magnetic moment.

The magnetization current is:
$$\mathbf{J}_m = \nabla \times \mathbf{M}$$

For a pressure gradient perpendicular to $\mathbf{B}$, this gives:
$$\mathbf{J}_m = \frac{\mathbf{B} \times \nabla p_\perp}{B^2}$$

which is exactly the diamagnetic current.

### 4.5 Example: Cylindrical Plasma Column

Consider a cylindrical plasma column with:
- Axial magnetic field: $\mathbf{B} = B_0 \hat{\mathbf{z}}$
- Radial pressure profile: $p(r) = p_0 \left(1 - \frac{r^2}{a^2}\right)$

The pressure gradient is:
$$\nabla p = \frac{dp}{dr} \hat{\mathbf{r}} = -\frac{2 p_0 r}{a^2} \hat{\mathbf{r}}$$

The diamagnetic current is:
$$\mathbf{J}_* = \frac{\mathbf{B} \times \nabla p}{B^2} = \frac{B_0 \hat{\mathbf{z}} \times \left( -\frac{2 p_0 r}{a^2} \hat{\mathbf{r}} \right)}{B_0^2} = \frac{2 p_0 r}{B_0 a^2} \hat{\boldsymbol{\theta}}$$

This is an azimuthal current that opposes the applied field (diamagnetic).

The diamagnetic drift velocity for electrons is:
$$\mathbf{v}_{*e} = \frac{\mathbf{B} \times \nabla p_e}{e n_e B^2} = \frac{2 k_B T_e r}{e B_0 a^2} \hat{\boldsymbol{\theta}}$$

Electrons drift in the $+\hat{\boldsymbol{\theta}}$ direction (counterclockwise when viewed from above).

Ions drift in the opposite direction:
$$\mathbf{v}_{*i} = -\frac{2 k_B T_i r}{e B_0 a^2} \hat{\boldsymbol{\theta}}$$

The net current is the sum of electron and ion contributions.

## 5. Two-Fluid Waves

### 5.1 Kinetic Alfvén Wave

At scales approaching the ion gyroradius, the Alfvén wave is modified by kinetic effects. The **kinetic Alfvén wave (KAW)** has a dispersion relation:

$$\omega^2 = k_\parallel^2 V_A^2 \left( 1 + k_\perp^2 \rho_s^2 \right)$$

where $\rho_s = c_s / \omega_{ci}$ is the **ion sound gyroradius** (or hybrid gyroradius), with $c_s = \sqrt{k_B T_e / m_i}$ the ion sound speed.

Key features:
- Finite $k_\perp$ increases the wave frequency
- Electric field has a parallel component: $E_\parallel \neq 0$
- Electrons can be accelerated parallel to $\mathbf{B}$

The KAW is important in:
- Auroral acceleration
- Solar wind turbulence
- Tokamak edge turbulence

### 5.2 Whistler Wave from Two-Fluid Perspective

In Lesson 10, we derived whistler waves from kinetic theory. Here's the two-fluid perspective:

Starting from Hall MHD (electron inertia neglected), the dispersion relation for electromagnetic waves is:

$$\omega = \frac{k_\parallel^2 V_A^2}{\omega_{ci}} \equiv k_\parallel V_A k_\parallel d_i$$

This is the **whistler wave**:
- High-frequency ($\omega \ll \omega_{ce}$, but $\omega \gg \omega_{ci}$)
- Right-hand polarized (electrons gyrate, ions stationary)
- Phase velocity increases with $k$ (dispersive)

The whistler wave plays a key role in:
- Magnetic reconnection (enables fast inflow)
- Radiation belt dynamics (pitch-angle scattering of energetic electrons)
- Solar corona heating

### 5.3 Ion-Cyclotron Wave

At frequencies near the ion cyclotron frequency, the **ion-cyclotron wave** (or **ion Bernstein wave**) appears:

$$\omega \approx \omega_{ci} + k_\parallel^2 V_A^2 / \omega_{ci}$$

Features:
- Left-hand polarized (ions gyrate, electrons respond adiabatically)
- Resonant absorption at $\omega = \omega_{ci}$
- Used for plasma heating (ICRF heating in tokamaks)

### 5.4 Two-Stream Instability

When two fluids have relative streaming velocity $u_0$, the system can be unstable. Consider ions at rest and electrons streaming with velocity $u_0$:

The dispersion relation becomes:
$$\omega^2 - k^2 c_s^2 - \omega_{pe}^2 = 0, \quad \text{(ion acoustic)}$$
$$(\omega - k u_0)^2 - \omega_{pe}^2 = 0 \quad \text{(Langmuir shifted by Doppler)}$$

When these modes couple, we get the **two-stream instability** if $u_0 > v_{te}$ (electron thermal speed).

Growth rate:
$$\gamma \sim \frac{\omega_{pe}}{3^{1/3}} \left( \frac{u_0}{v_{te}} \right)^{2/3}$$

This is a **kinetic instability**, but it can be captured in two-fluid theory with appropriate closure.

## 6. Python Code Examples

### 6.1 Two-Fluid vs. Single-Fluid Dispersion Relations

```python
import numpy as np
import matplotlib.pyplot as plt

# Plasma parameters
m_i = 1.67e-27  # proton mass (kg)
m_e = 9.11e-31  # electron mass (kg)
e = 1.6e-19     # elementary charge (C)
c = 3e8         # speed of light (m/s)
mu_0 = 4e-7 * np.pi  # permeability

n = 1e19        # density (m^-3)
B = 0.1         # magnetic field (T)
T_e = 10        # electron temperature (eV)
T_i = 10        # ion temperature (eV)

# Convert temperature to Joules
k_B = 1.38e-23
T_e_J = T_e * e
T_i_J = T_i * e

# Derived quantities
omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))
omega_pi = np.sqrt(n * e**2 / (m_i * 8.85e-12))
omega_ce = e * B / m_e
omega_ci = e * B / m_i

v_A = B / np.sqrt(mu_0 * n * m_i)  # Alfvén speed
c_s = np.sqrt((T_e_J + T_i_J) / m_i)  # ion sound speed
d_i = c / omega_pi  # ion skin depth
d_e = c / omega_pe  # electron skin depth

print("Plasma parameters:")
print(f"  Alfvén speed V_A = {v_A:.2e} m/s = {v_A/c:.2e} c")
print(f"  Ion sound speed c_s = {c_s:.2e} m/s")
print(f"  Ion skin depth d_i = {d_i:.2e} m")
print(f"  Electron skin depth d_e = {d_e:.2e} m")
print(f"  Ion gyrofrequency ω_ci = {omega_ci:.2e} rad/s")
print(f"  Electron gyrofrequency ω_ce = {omega_ce:.2e} rad/s")
print()

# Wavenumber range (parallel to B)
k_min = 1 / (100 * d_i)
k_max = 1 / (0.1 * d_i)
k = np.logspace(np.log10(k_min), np.log10(k_max), 500)

# MHD Alfvén wave (single-fluid)
omega_MHD = k * v_A

# Hall MHD Alfvén/whistler wave (two-fluid)
omega_Hall = k * v_A * np.sqrt(1 + (k * d_i)**2)

# Kinetic Alfvén wave (with finite k_perp)
k_perp = k / 2  # assume oblique propagation
rho_s = c_s / omega_ci  # ion sound gyroradius
omega_KAW = k * v_A * np.sqrt(1 + (k_perp * rho_s)**2)

# Plot dispersion relations
plt.figure(figsize=(10, 6))
plt.loglog(k * d_i, omega_MHD / omega_ci, 'b-', label='MHD Alfvén', linewidth=2)
plt.loglog(k * d_i, omega_Hall / omega_ci, 'r--', label='Hall MHD (whistler)', linewidth=2)
plt.loglog(k * d_i, omega_KAW / omega_ci, 'g-.', label='Kinetic Alfvén', linewidth=2)

plt.axvline(1, color='k', linestyle=':', alpha=0.5, label='$k d_i = 1$')
plt.xlabel(r'$k d_i$ (normalized wavenumber)', fontsize=12)
plt.ylabel(r'$\omega / \omega_{ci}$ (normalized frequency)', fontsize=12)
plt.title('Two-Fluid Dispersion Relations: Alfvén to Whistler Transition', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('two_fluid_dispersion.png', dpi=150)
plt.show()

print("At k d_i = 1:")
idx = np.argmin(np.abs(k * d_i - 1))
print(f"  MHD: ω/ω_ci = {omega_MHD[idx]/omega_ci:.2f}")
print(f"  Hall MHD: ω/ω_ci = {omega_Hall[idx]/omega_ci:.2f}")
print(f"  Kinetic Alfvén: ω/ω_ci = {omega_KAW[idx]/omega_ci:.2f}")
```

### 6.2 Generalized Ohm's Law: Relative Term Magnitudes

```python
import numpy as np
import matplotlib.pyplot as plt

def ohm_law_terms(n, T_e, B, L, V, eta=None):
    """
    Calculate relative magnitudes of generalized Ohm's law terms.

    Parameters:
    n: density (m^-3)
    T_e: electron temperature (eV)
    B: magnetic field (T)
    L: length scale (m)
    V: flow velocity (m/s)
    eta: resistivity (Ω·m), if None calculate from Spitzer
    """
    e = 1.6e-19
    m_e = 9.11e-31
    m_i = 1.67e-27
    mu_0 = 4e-7 * np.pi
    k_B = 1.38e-23
    c = 3e8

    # Spitzer resistivity is the classical (collisional) baseline; anomalous resistivity
    # from turbulence is not included here because it requires a turbulence model.
    # The T_e^{-3/2} scaling means hot plasmas are nearly ideal (low resistivity).
    if eta is None:
        T_e_eV = T_e
        ln_Lambda = 15  # Coulomb logarithm (typical)
        eta = 5.2e-5 * ln_Lambda * T_e_eV**(-3/2)  # Ω·m

    # Ampere's law (∇×B = μ₀J) estimates J ~ B/(μ₀L): the curl of B over a
    # characteristic scale L sets the order-of-magnitude for the current density.
    # This is the standard dimensionless ordering used in scale analysis.
    J = B / (mu_0 * L)

    # The ideal MHD term v×B sets the reference scale: all other terms are
    # normalized to this to quantify how far from ideal MHD we are.
    E_ideal = V * B

    # Generalized Ohm's law terms
    E_resistive = eta * J
    E_Hall = J * B / (e * n)
    # Pressure gradient ∇p_e ~ nkT/L divided by ne gives kT/(eL), independent of n.
    # This is why the pressure term becomes important in steep gradient regions
    # regardless of density, unlike the Hall term which scales as B/(μ₀nen L).
    E_pressure = k_B * T_e * e / (e * L)  # ∇p_e ~ nkT/L

    omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))
    d_e = c / omega_pe
    # Electron inertia term scales as (d_e/L)^2: it enters only when the length
    # scale L approaches the electron skin depth, far smaller than d_i.
    E_inertia = (m_e / (e**2 * n**2)) * J * (V / L)

    # Normalize to ideal MHD term
    terms = {
        'Ideal (v×B)': E_ideal,
        'Resistive (ηJ)': E_resistive,
        'Hall (J×B/ne)': E_Hall,
        'Pressure (∇p_e/ne)': E_pressure,
        'Inertia (m_e dJ/dt)': E_inertia
    }

    # Normalize every term to E_ideal so the output directly shows which terms
    # are O(1) (important) vs. much smaller than 1 (negligible) at each scale.
    return {k: v/E_ideal for k, v in terms.items()}, eta

# Parameter scan: vary length scale
L_range = np.logspace(-3, 3, 100)  # 1 mm to 1 km
n = 1e19
T_e = 10
B = 0.1
V = 1e5  # 100 km/s

terms_vs_L = {k: [] for k in ['Ideal (v×B)', 'Resistive (ηJ)',
                               'Hall (J×B/ne)', 'Pressure (∇p_e/ne)',
                               'Inertia (m_e dJ/dt)']}

for L in L_range:
    terms, _ = ohm_law_terms(n, T_e, B, L, V)
    for k, v in terms.items():
        terms_vs_L[k].append(v)

# Plot
plt.figure(figsize=(10, 6))
for key, values in terms_vs_L.items():
    if key != 'Ideal (v×B)':
        plt.loglog(L_range, values, label=key, linewidth=2)

# Mark characteristic scales
d_e = 3e8 / np.sqrt(n * (1.6e-19)**2 / (9.11e-31 * 8.85e-12))
d_i = 3e8 / np.sqrt(n * (1.6e-19)**2 / (1.67e-27 * 8.85e-12))
plt.axvline(d_e, color='r', linestyle=':', alpha=0.7, label=f'$d_e$ = {d_e:.2e} m')
plt.axvline(d_i, color='b', linestyle=':', alpha=0.7, label=f'$d_i$ = {d_i:.2e} m')

plt.xlabel('Length scale L (m)', fontsize=12)
plt.ylabel('Relative magnitude (normalized to v×B)', fontsize=12)
plt.title('Generalized Ohm\'s Law: Term Magnitudes vs. Scale', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('ohm_law_terms.png', dpi=150)
plt.show()

# Print values at specific scales
print("\nRelative term magnitudes:")
print(f"\nAt L = {d_e:.2e} m (electron skin depth):")
terms, _ = ohm_law_terms(n, T_e, B, d_e, V)
for k, v in terms.items():
    print(f"  {k:25s}: {v:.2e}")

print(f"\nAt L = {d_i:.2e} m (ion skin depth):")
terms, _ = ohm_law_terms(n, T_e, B, d_i, V)
for k, v in terms.items():
    print(f"  {k:25s}: {v:.2e}")

print(f"\nAt L = 1 m (macroscopic scale):")
terms, _ = ohm_law_terms(n, T_e, B, 1.0, V)
for k, v in terms.items():
    print(f"  {k:25s}: {v:.2e}")
```

### 6.3 Diamagnetic Drift Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def diamagnetic_drift_cylinder():
    """
    Visualize diamagnetic drift in a cylindrical plasma column.
    """
    # Plasma parameters
    a = 0.1  # plasma radius (m)
    B_0 = 1.0  # axial magnetic field (T)
    p_0 = 1e5  # peak pressure (Pa)
    T_e = 10  # electron temperature (eV)
    T_i = 10  # ion temperature (eV)
    n_0 = 1e19  # peak density (m^-3)

    e = 1.6e-19
    k_B = 1.38e-23

    # Radial grid
    r = np.linspace(0, a, 100)

    # Pressure profile (parabolic)
    p = p_0 * (1 - (r/a)**2)
    p_e = p / 2
    p_i = p / 2
    n = n_0 * (1 - (r/a)**2)

    # Pressure gradient
    dp_dr = -2 * p_0 * r / a**2
    dp_e_dr = dp_dr / 2
    dp_i_dr = dp_dr / 2

    # Diamagnetic drift velocities
    v_star_e = -dp_e_dr / (e * n * B_0)  # azimuthal (θ) direction
    v_star_i = dp_i_dr / (e * n * B_0)

    # Diamagnetic current density
    J_theta = -dp_dr / B_0

    # Plot profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pressure profile
    axes[0, 0].plot(r*100, p/1e3, 'b-', linewidth=2, label='Total')
    axes[0, 0].plot(r*100, p_e/1e3, 'r--', linewidth=2, label='Electron')
    axes[0, 0].plot(r*100, p_i/1e3, 'g--', linewidth=2, label='Ion')
    axes[0, 0].set_xlabel('Radius (cm)', fontsize=11)
    axes[0, 0].set_ylabel('Pressure (kPa)', fontsize=11)
    axes[0, 0].set_title('Pressure Profile', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Diamagnetic drift velocities
    axes[0, 1].plot(r*100, v_star_e/1e3, 'r-', linewidth=2, label='Electron')
    axes[0, 1].plot(r*100, v_star_i/1e3, 'g-', linewidth=2, label='Ion')
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Radius (cm)', fontsize=11)
    axes[0, 1].set_ylabel('Drift velocity (km/s)', fontsize=11)
    axes[0, 1].set_title('Diamagnetic Drift Velocity (azimuthal)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Diamagnetic current
    axes[1, 0].plot(r*100, J_theta/1e3, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Radius (cm)', fontsize=11)
    axes[1, 0].set_ylabel('Current density (kA/m²)', fontsize=11)
    axes[1, 0].set_title('Diamagnetic Current Density (azimuthal)', fontsize=12)
    axes[1, 0].grid(alpha=0.3)

    # 2D visualization: top view
    ax = axes[1, 1]
    theta = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Pressure contour
    P_grid = np.outer(np.ones_like(theta), p)
    contour = ax.contourf(X*100, Y*100, P_grid/1e3, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax, label='Pressure (kPa)')

    # Velocity vectors (sample points)
    n_arrows = 8
    r_arrows = np.linspace(0.2*a, 0.9*a, 5)
    theta_arrows = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)

    for ri in r_arrows:
        for ti in theta_arrows:
            xi = ri * np.cos(ti)
            yi = ri * np.sin(ti)

            # Diamagnetic drift is in theta direction
            # In Cartesian: v_theta = -sin(θ) v_r_hat + cos(θ) v_θ_hat
            idx = np.argmin(np.abs(r - ri))
            v_mag = v_star_e[idx]

            vx = -v_mag * np.sin(ti)
            vy = v_mag * np.cos(ti)

            ax.arrow(xi*100, yi*100, vx*1e-3, vy*1e-3,
                    head_width=0.5, head_length=0.3, fc='cyan', ec='cyan', alpha=0.8)

    ax.set_xlabel('x (cm)', fontsize=11)
    ax.set_ylabel('y (cm)', fontsize=11)
    ax.set_title('Electron Diamagnetic Drift (top view)', fontsize=12)
    ax.set_aspect('equal')
    ax.add_patch(Circle((0, 0), a*100, fill=False, edgecolor='white', linewidth=2))

    plt.tight_layout()
    plt.savefig('diamagnetic_drift.png', dpi=150)
    plt.show()

    # Print values at r = a/2
    idx = np.argmin(np.abs(r - a/2))
    print(f"\nAt r = a/2 = {a/2*100:.1f} cm:")
    print(f"  Pressure: {p[idx]/1e3:.2f} kPa")
    print(f"  Electron drift: {v_star_e[idx]/1e3:.2f} km/s")
    print(f"  Ion drift: {v_star_i[idx]/1e3:.2f} km/s")
    print(f"  Current density: {J_theta[idx]/1e3:.2f} kA/m²")
    print(f"  Drift frequency: {v_star_e[idx]/(a/2):.2e} rad/s")
    print(f"  Compare to ω_ci = {e*B_0/(1.67e-27):.2e} rad/s")

diamagnetic_drift_cylinder()
```

### 6.4 Two-Fluid Closure Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_closures():
    """
    Compare different closure models: isothermal vs. adiabatic.
    Simulate compression of a plasma element.
    """
    # Initial conditions
    n_0 = 1e19  # m^-3
    T_0 = 10    # eV
    V_0 = 1.0   # m^3

    gamma = 5/3  # adiabatic index

    # Compression ratio
    V = np.linspace(V_0, 0.1*V_0, 100)
    n = n_0 * (V_0 / V)  # density increases as volume decreases

    # Isothermal: T = const
    T_isothermal = np.ones_like(V) * T_0
    p_isothermal = n * T_isothermal

    # Adiabatic: p V^γ = const
    p_adiabatic = n_0 * T_0 * (V_0 / V)**gamma
    T_adiabatic = p_adiabatic / n

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Temperature vs. compression
    axes[0].plot(V/V_0, T_isothermal, 'b-', linewidth=2, label='Isothermal')
    axes[0].plot(V/V_0, T_adiabatic, 'r--', linewidth=2, label='Adiabatic (γ=5/3)')
    axes[0].set_xlabel('V / V₀', fontsize=12)
    axes[0].set_ylabel('Temperature (eV)', fontsize=12)
    axes[0].set_title('Temperature Evolution under Compression', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Pressure vs. density
    axes[1].loglog(n/n_0, p_isothermal/(n_0*T_0), 'b-', linewidth=2, label='Isothermal (p ∝ n)')
    axes[1].loglog(n/n_0, p_adiabatic/(n_0*T_0), 'r--', linewidth=2, label='Adiabatic (p ∝ n^γ)')
    axes[1].set_xlabel('n / n₀', fontsize=12)
    axes[1].set_ylabel('p / (n₀ T₀)', fontsize=12)
    axes[1].set_title('Pressure vs. Density', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('closure_comparison.png', dpi=150)
    plt.show()

    # At 50% compression
    idx = np.argmin(np.abs(V/V_0 - 0.5))
    print("\nAt 50% compression (V = 0.5 V₀):")
    print(f"  Density: {n[idx]/n_0:.2f} n₀")
    print(f"  Isothermal:")
    print(f"    T = {T_isothermal[idx]:.2f} eV (unchanged)")
    print(f"    p = {p_isothermal[idx]/(n_0*T_0):.2f} (n₀ T₀)")
    print(f"  Adiabatic:")
    print(f"    T = {T_adiabatic[idx]:.2f} eV")
    print(f"    p = {p_adiabatic[idx]/(n_0*T_0):.2f} (n₀ T₀)")
    print(f"  Pressure ratio (adiabatic/isothermal): {p_adiabatic[idx]/p_isothermal[idx]:.2f}")

compare_closures()
```

## Summary

In this lesson, we derived the two-fluid model by taking velocity-space moments of the Vlasov equation. Key points:

1. **Moment hierarchy**: Each moment equation introduces the next higher-order moment, leading to a closure problem.

2. **Closure models**: Isothermal, adiabatic, and CGL closures truncate the hierarchy with different physical assumptions.

3. **Generalized Ohm's law**: The full form includes resistive, Hall, electron pressure, and electron inertia terms. Each term becomes important at different length scales:
   - Resistive: low $R_m$ (collisional plasmas)
   - Hall: $L \sim d_i$ (ion skin depth)
   - Electron pressure: steep gradients
   - Electron inertia: $L \sim d_e$ (electron skin depth)

4. **Hall effect**: At scales $\lesssim d_i$, ions decouple from the magnetic field while electrons remain frozen-in. This enables fast magnetic reconnection and whistler wave propagation.

5. **Diamagnetic drift**: A fluid drift arising from pressure gradients, not a single-particle drift. Creates a current $\mathbf{J}_* = \mathbf{B} \times \nabla p / B^2$.

6. **Two-fluid waves**: Hall MHD modifies Alfvén waves into whistler waves at small scales. Kinetic Alfvén waves include finite-$k_\perp$ effects.

The two-fluid model bridges the gap between single-particle kinetic theory and single-fluid MHD. It captures important physics at intermediate scales (ion gyroradius to ion skin depth) that are missed by ideal MHD but don't require the full complexity of kinetic theory.

## Practice Problems

### Problem 1: Moment Calculation
Starting from the Vlasov equation, derive the second moment (energy equation) explicitly. Show that the heat flux $\mathbf{q}_s = \frac{1}{2} m_s \int w^2 \mathbf{w} f_s d^3v$ appears. What physical process does the heat flux represent?

### Problem 2: Hall MHD Dispersion
Derive the dispersion relation for whistler waves in Hall MHD:
$$\omega = \frac{k_\parallel^2 V_A^2}{\omega_{ci}}$$
Start from the two-fluid equations with the Hall term, assume $\omega \ll \omega_{ce}$ and $\omega \gg \omega_{ci}$, and use the cold plasma approximation ($p = 0$).

### Problem 3: Diamagnetic Current in a Tokamak
In a tokamak with major radius $R_0 = 3$ m and minor radius $a = 1$ m, the electron pressure profile is:
$$p_e(r) = p_0 \left(1 - \frac{r^2}{a^2}\right)^2$$
with $p_0 = 5 \times 10^5$ Pa. The toroidal magnetic field is $B_\phi = 5$ T. Calculate:
(a) The diamagnetic current density at $r = a/2$.
(b) The total poloidal current from the diamagnetic effect (integrate $J_\theta$ over the cross-section).
(c) Compare this to the bootstrap current (which has a similar profile).

### Problem 4: Generalized Ohm's Law in a Current Sheet
In a magnetic reconnection current sheet, the length scale is $L = 10 d_i$, where $d_i = 100$ km is the ion skin depth. The plasma density is $n = 10^7$ m$^{-3}$ (solar wind), electron temperature $T_e = 100$ eV, and magnetic field $B = 10$ nT. Calculate the relative magnitudes of:
(a) The ideal MHD term $\mathbf{v} \times \mathbf{B}$
(b) The Hall term $\mathbf{J} \times \mathbf{B} / (en)$
(c) The electron pressure term $\nabla p_e / (en)$
(d) The electron inertia term
Which term(s) are important in this current sheet?

### Problem 5: Two-Fluid Instability
Consider a two-fluid plasma with $T_e = T_i$ and a density gradient $\nabla n = -n_0 / L_n \hat{\mathbf{x}}$ in a magnetic field $\mathbf{B} = B_0 \hat{\mathbf{z}}$.
(a) Calculate the electron and ion diamagnetic drift velocities.
(b) The drift-wave instability occurs when the phase difference between density and potential perturbations causes wave growth. Using the continuity equation and quasi-neutrality, show that electrostatic drift waves have the dispersion relation:
$$\omega = \frac{k_y k_B T_e}{e B_0 L_n}$$
where $k_y$ is the wavenumber perpendicular to both $\mathbf{B}$ and $\nabla n$.
(c) For $L_n = 1$ cm, $T_e = 1$ eV, $B_0 = 0.1$ T, and $k_y = 100$ m$^{-1}$, calculate the drift wave frequency.

---

**Previous**: [Wave Heating and Instabilities](./12_Wave_Heating_and_Instabilities.md) | **Next**: [From Kinetic to MHD](./14_From_Kinetic_to_MHD.md)

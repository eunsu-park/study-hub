# 9. Collisional Kinetics

## Learning Objectives

- Understand the Boltzmann collision operator and its physical meaning
- Derive the Fokker-Planck equation for Coulomb collisions in plasmas
- Master the Rosenbluth potential formulation for efficient collision calculations
- Learn Braginskii transport theory and transport coefficients in magnetized plasmas
- Understand neoclassical transport regimes in toroidal confinement devices
- Apply collision operators to practical problems like particle slowing down and resistivity

## Introduction

In previous lessons, we treated plasmas as collisionless systems governed by the Vlasov equation. However, real plasmas experience collisions, albeit at much lower rates than neutral gases. Collisions are crucial for:

- Thermalization and approach to equilibrium
- Electrical resistivity and energy dissipation
- Transport of particles, momentum, and energy across magnetic fields
- Neoclassical effects in toroidal confinement

The challenge in plasma collision theory is that electromagnetic (Coulomb) interactions are **long-range**. Unlike hard-sphere collisions in neutral gases, charged particles interact over large distances through their $1/r^2$ electric fields. This leads to a predominance of **small-angle deflections** rather than large-angle "hard" collisions.

In this lesson, we develop the kinetic theory of collisions, starting from the general Boltzmann operator, specializing to the Fokker-Planck equation for plasmas, and deriving transport coefficients for practical applications.

## 1. The Boltzmann Collision Operator

### 1.1 General Formulation

The full kinetic equation including collisions is:

$$\frac{\partial f}{\partial t} + \mathbf{v}\cdot\frac{\partial f}{\partial \mathbf{x}} + \mathbf{a}\cdot\frac{\partial f}{\partial \mathbf{v}} = \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}$$

where $\mathbf{a}$ is the acceleration from external fields. The collision term represents the net rate of change of $f$ due to binary collisions.

For binary collisions with cross-section $\sigma(\mathbf{v}, \mathbf{v}_1; \mathbf{v}', \mathbf{v}_1')$, the **Boltzmann collision operator** is:

$$\left(\frac{\partial f}{\partial t}\right)_{\text{coll}} = \int\int\int \left[f(\mathbf{v}')f(\mathbf{v}_1') - f(\mathbf{v})f(\mathbf{v}_1)\right] |\mathbf{v}-\mathbf{v}_1| \sigma(\Omega)\, d\Omega\, d^3v_1$$

Physical interpretation:
- **Gain term**: $f(\mathbf{v}')f(\mathbf{v}_1')$ - particles scattered INTO velocity $\mathbf{v}$ from $(\mathbf{v}', \mathbf{v}_1')$
- **Loss term**: $f(\mathbf{v})f(\mathbf{v}_1)$ - particles scattered OUT OF velocity $\mathbf{v}$
- $|\mathbf{v}-\mathbf{v}_1|$ - relative velocity determines collision rate
- Integration over all impact partners $\mathbf{v}_1$ and scattering angles $\Omega$

The primes denote post-collision velocities. Conservation laws constrain:
- Momentum: $\mathbf{v} + \mathbf{v}_1 = \mathbf{v}' + \mathbf{v}_1'$
- Energy: $v^2 + v_1^2 = v'^2 + v_1'^2$

### 1.2 Collision Invariants and Conservation Laws

A quantity $\psi(\mathbf{v})$ is a **collision invariant** if:

$$\int \psi(\mathbf{v}) C[f]\, d^3v = 0$$

for any distribution $f$, where $C[f]$ is the collision operator.

**Summational invariants** satisfy:
$$\psi(\mathbf{v}) + \psi(\mathbf{v}_1) = \psi(\mathbf{v}') + \psi(\mathbf{v}_1')$$

The five summational invariants are:
1. Mass: $\psi = m$ (particle conservation)
2. Momentum: $\psi = m\mathbf{v}$ (three components)
3. Energy: $\psi = \frac{1}{2}mv^2$

These lead to conservation laws:

```
Particle number:   ∫ C[f] d³v = 0
Momentum:          ∫ mv C[f] d³v = 0
Energy:            ∫ (½mv²) C[f] d³v = 0
```

### 1.3 Boltzmann's H-Theorem

Define the **H-function**:
$$H(t) = \int f(\mathbf{v},t) \ln f(\mathbf{v},t)\, d^3v$$

Boltzmann proved that $dH/dt \leq 0$ for any collision operator satisfying detailed balance. This is the **H-theorem**, showing that entropy $S = -k_B H$ always increases.

The proof relies on:
$$\frac{dH}{dt} = \int (\ln f + 1) C[f]\, d^3v$$

Using the symmetry of the collision integral and the inequality $\ln x \leq x - 1$, one shows $dH/dt \leq 0$ with equality only when:

$$f(\mathbf{v})f(\mathbf{v}_1) = f(\mathbf{v}')f(\mathbf{v}_1')$$

for all colliding pairs. This condition is satisfied by the **Maxwellian distribution**:

$$f_M(\mathbf{v}) = n\left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{m v^2}{2k_B T}\right)$$

Thus, collisions drive any distribution toward a Maxwellian equilibrium.

### 1.4 Collision Frequency Estimates

For a rough estimate, dimensional analysis gives:

$$\nu_{\text{coll}} \sim n \sigma v_{\text{th}}$$

For Coulomb collisions, the effective cross-section is:
$$\sigma \sim \pi b_{90}^2$$

where $b_{90}$ is the impact parameter for 90° deflection:
$$b_{90} = \frac{e^2}{4\pi\epsilon_0 m v_{\text{th}}^2}$$

This gives:
$$\nu_{ei} \sim \frac{n e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e v_{\text{th}}^3} \sim \frac{n e^4 \ln\Lambda}{4\pi\epsilon_0^2 (k_B T)^{3/2} m_e^{1/2}}$$

The **Coulomb logarithm** $\ln\Lambda$ arises from integrating over impact parameters:
$$\ln\Lambda = \ln\left(\frac{\lambda_D}{b_{90}}\right) \approx 10-20$$

for most plasmas. Typical values:
- Fusion plasma (ITER): $\ln\Lambda \approx 17$
- Solar corona: $\ln\Lambda \approx 20$
- Lab plasma: $\ln\Lambda \approx 10-15$

## 2. Fokker-Planck Equation for Plasmas

### 2.1 Derivation from Small-Angle Scattering

Coulomb collisions are dominated by **many small-angle deflections** rather than rare large-angle collisions. Consider the cumulative effect of many weak scatterings over time $\Delta t$:

$$\Delta \mathbf{v} = \sum_{i=1}^{N} \Delta \mathbf{v}_i$$

where $N \sim n \sigma v \Delta t$ is the number of collisions.

For small deflections:
- Mean deflection: $\langle \Delta \mathbf{v} \rangle \sim N \langle \Delta v_i \rangle \propto \Delta t$
- Variance: $\langle (\Delta \mathbf{v})^2 \rangle \sim N \langle (\Delta v_i)^2 \rangle \propto \Delta t$

Since many uncorrelated scatterings occur, the Central Limit Theorem applies. The change in the distribution function can be expanded:

$$f(\mathbf{v} + \Delta\mathbf{v}, t + \Delta t) - f(\mathbf{v}, t) = \Delta t \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}$$

Expanding the left side to second order (since variance is first order in $\Delta t$):

$$f(\mathbf{v},t) + \frac{\partial f}{\partial v_i}\langle\Delta v_i\rangle + \frac{1}{2}\frac{\partial^2 f}{\partial v_i \partial v_j}\langle\Delta v_i \Delta v_j\rangle - f(\mathbf{v},t) = \Delta t \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}$$

This yields the **Fokker-Planck collision operator**:

$$\boxed{\left(\frac{\partial f}{\partial t}\right)_{\text{coll}} = -\frac{\partial}{\partial v_i}\left[f \langle\Delta v_i\rangle\right] + \frac{1}{2}\frac{\partial^2}{\partial v_i \partial v_j}\left[f \langle\Delta v_i \Delta v_j\rangle\right]}$$

(Einstein summation over repeated indices implied.)

The two terms represent:
1. **Dynamical friction** (drag): $\langle\Delta v_i\rangle$ - systematic deceleration
2. **Velocity diffusion**: $\langle\Delta v_i \Delta v_j\rangle$ - random walk in velocity space

### 2.2 Rosenbluth Potentials

Computing $\langle\Delta v_i\rangle$ and $\langle\Delta v_i \Delta v_j\rangle$ directly from Coulomb scattering is cumbersome. **Rosenbluth (1957)** showed these can be expressed in terms of two potentials.

Define:
$$\boxed{h(\mathbf{v}) = \int \frac{f(\mathbf{v}')}{|\mathbf{v}-\mathbf{v}'|}\, d^3v'}$$

$$\boxed{g(\mathbf{v}) = \int |\mathbf{v}-\mathbf{v}'| f(\mathbf{v}')\, d^3v'}$$

These are the **Rosenbluth potentials** (also called $H$ and $G$ in some texts).

The friction and diffusion coefficients are:

$$\langle\Delta v_i\rangle = -\Gamma \frac{\partial g}{\partial v_i}$$

$$\langle\Delta v_i \Delta v_j\rangle = \Gamma \frac{\partial^2 h}{\partial v_i \partial v_j}$$

where $\Gamma = \frac{e^4 \ln\Lambda}{4\pi\epsilon_0^2 m^2}$ is a constant.

The Fokker-Planck operator becomes:

$$\boxed{C[f] = \Gamma \frac{\partial}{\partial v_i}\left[f \frac{\partial g}{\partial v_i} + \frac{\partial}{\partial v_j}\left(f \frac{\partial^2 h}{\partial v_i \partial v_j}\right)\right]}$$

This form is computationally efficient: compute $h$ and $g$ once via integrals, then evaluate derivatives.

### 2.3 Like-Particle and Unlike-Particle Collisions

For a plasma with multiple species (electrons, ions), we must consider:

**Electron-electron collisions**: $C_{ee}[f_e]$
**Ion-ion collisions**: $C_{ii}[f_i]$
**Electron-ion collisions**: $C_{ei}[f_e]$ and $C_{ie}[f_i]$

The mass ratio $m_i/m_e \approx 1836$ (for protons) simplifies electron-ion collisions:
- Electrons lose energy slowly to ions (many collisions to thermalize)
- Ions gain little energy from electrons but experience momentum drag

The **energy exchange rate** between species is:

$$\frac{dT_e}{dt} \bigg|_{\text{coll}} = -\nu_{eq}(T_e - T_i)$$

where the **equilibration frequency** is:

$$\nu_{eq} = \frac{m_e}{m_i} \nu_{ei} \sim \frac{1}{1836} \nu_{ei}$$

This explains why electron and ion temperatures can differ significantly in many plasmas.

### 2.4 Landau Form of Fokker-Planck Operator

An alternative form emphasizes the tensor structure. Define:

$$\mathbf{A}(\mathbf{v}) = \int \frac{(\mathbf{v}-\mathbf{v}')}{|\mathbf{v}-\mathbf{v}'|^3} f(\mathbf{v}')\, d^3v'$$

$$\overleftrightarrow{B}(\mathbf{v}) = \int \frac{\overleftrightarrow{I} - \hat{\mathbf{v}}\hat{\mathbf{v}}}{|\mathbf{v}-\mathbf{v}'|} f(\mathbf{v}')\, d^3v'$$

where $\overleftrightarrow{I}$ is the identity tensor and $\hat{\mathbf{v}} = (\mathbf{v}-\mathbf{v}')/|\mathbf{v}-\mathbf{v}'|$.

Then:
$$C[f] = \Gamma \nabla_v \cdot \left[f \mathbf{A} + \nabla_v \cdot (f \overleftrightarrow{B})\right]$$

This is the **Landau form**, useful for analytical calculations.

## 3. Test Particle Slowing Down

### 3.1 Electron Slowing in a Maxwellian Background

Consider a fast electron (from fusion alpha particles, runaway electrons, or NBI) slowing down in a thermal background plasma.

For a test particle with velocity $\mathbf{v}$ much larger than thermal velocity $v_{\text{th}}$, the friction force simplifies:

$$\frac{d\mathbf{v}}{dt} = -\nu_s \frac{\mathbf{v}}{v}$$

where the **slowing-down frequency** is:

$$\nu_s = \frac{n e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e^2 v^3} \cdot \Phi\left(\frac{v}{v_{\text{th}}}\right)$$

For $v \gg v_{\text{th}}$, $\Phi \approx 1$, giving:

$$\frac{dv}{dt} = -\nu_s \propto -\frac{1}{v^3}$$

Solving:
$$v^4 - v_0^4 = -4\nu_s' t$$

The slowing-down time from $v_0$ to $v_{\text{th}}$ is:

$$\tau_s = \frac{v_0^3}{4\nu_s(v_0)} \sim \frac{4\pi\epsilon_0^2 m_e^2 v_0^3}{n e^4 \ln\Lambda}$$

For a 3.5 MeV alpha particle in a fusion plasma:
- $n = 10^{20}$ m$^{-3}$, $T = 10$ keV
- $\tau_s \approx 1$ second

This is much longer than confinement time in many devices, so alphas can heat the plasma effectively.

### 3.2 Critical Velocity

When a fast particle slows down, it transfers energy to both electrons (via collisions) and ions. The **critical velocity** $v_c$ is where the drag on electrons equals the drag on ions:

$$\nu_e(v_c) = \nu_i(v_c)$$

For $v > v_c$: primarily electron heating
For $v < v_c$: primarily ion heating

The critical velocity is:

$$v_c \approx v_{\text{th},e} \left(\frac{m_i}{m_e}\right)^{1/3} Z^{2/3}$$

For deuterium plasma ($Z=1$):
$$v_c \approx 12.2 \, v_{\text{th},e}$$

This corresponds to an energy:
$$E_c = \frac{1}{2}m_e v_c^2 \approx 14.8 \, T_e$$

Particles with $E > E_c$ heat electrons; $E < E_c$ heat ions.

### 3.3 Runaway Electrons

In the presence of an electric field $E$, the force balance is:

$$eE = m_e \nu_s v$$

For high velocities, $\nu_s \propto v^{-3}$, so the friction decreases with speed. If $eE$ exceeds the maximum friction force, electrons **run away** to arbitrarily high energies.

The **Dreicer field** is:

$$E_D = \frac{n e^3 \ln\Lambda}{4\pi\epsilon_0^2 k_B T_e}$$

For $E > E_D$, a significant fraction of electrons run away. This is dangerous in tokamaks:
- During disruptions, $E$ spikes
- Runaway electrons accelerate to 10-100 MeV
- Can damage plasma-facing components

Mitigation strategies: massive gas injection, shattered pellet injection.

## 4. Braginskii Transport Theory

### 4.1 Moment Approach

Instead of solving the full Fokker-Planck equation, we can take **velocity moments** to derive fluid equations with collision terms.

Define moments:
- Density: $n = \int f\, d^3v$
- Flow velocity: $\mathbf{u} = \frac{1}{n}\int \mathbf{v} f\, d^3v$
- Pressure tensor: $\overleftrightarrow{P} = m \int (\mathbf{v}-\mathbf{u})(\mathbf{v}-\mathbf{u}) f\, d^3v$
- Heat flux: $\mathbf{q} = \frac{m}{2} \int (\mathbf{v}-\mathbf{u})^2 (\mathbf{v}-\mathbf{u}) f\, d^3v$

Taking moments of the kinetic equation yields a hierarchy:

```
Continuity:        ∂n/∂t + ∇·(nu) = 0
Momentum:          mn(∂u/∂t + u·∇u) = -∇·P + F_coll
Energy:            ∂/∂t(3nT/2) + ∇·q = Q_coll
```

The collision terms $F_{\text{coll}}$ and $Q_{\text{coll}}$ involve moments of $C[f]$.

**Braginskii (1965)** solved the Fokker-Planck equation perturbatively, assuming small deviations from Maxwellian, to derive **transport coefficients**.

### 4.2 Parallel Transport

Along magnetic field lines, particles stream freely (collisions are weak). The parallel transport coefficients are:

**Parallel viscosity**:
$$\eta_{\parallel} = 0.96 \, n T \tau$$

where $\tau = 1/\nu$ is the collision time.

**Parallel thermal conductivity**:
$$\kappa_{\parallel,e} = 3.16 \, \frac{n T \tau}{m_e}$$

$$\kappa_{\parallel,i} = 3.9 \, \frac{n T \tau}{m_i}$$

**Electrical conductivity** (inverse of resistivity):
$$\sigma_{\parallel} = \frac{n e^2 \tau}{m_e} = \frac{1.96 \, n e^2 \tau}{m_e}$$

The **classical resistivity** is:
$$\eta_{\text{classical}} = \frac{1}{\sigma_{\parallel}} = \frac{m_e}{1.96 \, n e^2 \tau} \propto T^{-3/2}$$

Numerical value:
$$\eta_{\text{classical}} \approx 5.2 \times 10^{-5} \frac{\ln\Lambda}{T_e^{3/2}} \quad (\Omega\cdot\text{m}, \, T_e \text{ in eV})$$

### 4.3 Perpendicular Transport

Across magnetic field lines, particles must diffuse via collisions (Larmor orbits trap them). The perpendicular transport is **much weaker**.

**Perpendicular thermal conductivity**:
$$\kappa_{\perp,e} = 4.66 \, \frac{n T}{m_e \omega_{ce}^2 \tau}$$

$$\kappa_{\perp,i} = 2.0 \, \frac{n T}{m_i \omega_{ci}^2 \tau}$$

The ratio of parallel to perpendicular is:

$$\frac{\kappa_{\parallel}}{\kappa_{\perp}} \sim (\omega_c \tau)^2$$

For a fusion plasma ($B = 5$ T, $T_e = 10$ keV, $n = 10^{20}$ m$^{-3}$):
- $\omega_{ce} \tau \approx 10^6$
- $\kappa_{\parallel}/\kappa_{\perp} \approx 10^{12}$

This enormous anisotropy means heat flows almost exclusively along field lines.

**Perpendicular viscosity**:
Involves multiple coefficients $\eta_0, \eta_1, \eta_2, \eta_3, \eta_4$ for different stress components. The key result is that perpendicular momentum transport is also suppressed by $(\omega_c \tau)^{-2}$.

### 4.4 Anomalous Transport

In real experiments, the observed transport is often **orders of magnitude larger** than classical Braginskii predictions. This is **anomalous transport**, caused by:

- Turbulence (drift waves, ITG, ETG modes)
- Magnetic field perturbations
- Non-Maxwellian distributions

Empirical scaling laws (e.g., ITER H-mode confinement):
$$\tau_E \sim I_p^{0.93} B^{0.15} P^{-0.69} n^{0.41} M^{0.19} R^{1.97} \epsilon^{0.58} \kappa^{0.78}$$

where $I_p$ is plasma current, $P$ is heating power, $M$ is mass, $R$ is major radius, $\epsilon$ is inverse aspect ratio, $\kappa$ is elongation.

Understanding and controlling anomalous transport is a central challenge in fusion research.

## 5. Neoclassical Transport

### 5.1 Toroidal Geometry Effects

In a **torus** (tokamak, stellarator), the magnetic field strength varies:
$$B(\theta) = B_0 \left(1 + \epsilon \cos\theta\right)$$

where $\epsilon = r/R$ is the inverse aspect ratio and $\theta$ is the poloidal angle.

Particles with small parallel velocity can be **trapped** in the low-field region on the outside of the torus. They bounce back and forth, never completing a full poloidal circuit.

The fraction of trapped particles is:
$$f_{\text{trapped}} \sim \sqrt{\epsilon}$$

For ITER ($\epsilon \sim 0.3$): $f_{\text{trapped}} \sim 0.5$ (50% trapped).

### 5.2 Banana Orbits

Trapped particles execute **banana orbits**: their drift orbits form banana-shaped regions in poloidal cross-section.

```
       |  Passing particles:
   ____|____   complete full poloidal circuit
  /    |    \
 /     |     \   Trapped particles:
|      |      |  bounce in "banana" orbit
|    __|__    |  on outer side
 \   /   \   /
  \_/     \_/

     ^
   Low B region
```

The banana width is:
$$\Delta_b \sim \rho_L \sqrt{\epsilon}$$

where $\rho_L$ is the Larmor radius.

### 5.3 Collisionality Regimes

Neoclassical transport depends on the **collisionality** $\nu_* = \nu / \omega_b$ where $\omega_b$ is the bounce frequency.

**Banana regime** ($\nu_* \ll 1$): particles complete many bounces before collision
- Transport: $D \sim D_{\text{classical}} / \epsilon^{3/2}$
- Enhancement over classical: $1/\epsilon^{3/2}$

**Plateau regime** ($\nu_* \sim 1$): collision frequency $\sim$ bounce frequency
- Transport: $D \sim D_{\text{classical}} / \epsilon$
- Intermediate enhancement

**Pfirsch-Schlüter regime** ($\nu_* \gg 1$): many collisions per bounce
- Transport: $D \sim D_{\text{classical}} \cdot q^2$
- Enhancement by safety factor squared

Typical tokamak parameters place electrons in plateau/banana regime, ions in banana regime.

### 5.4 Bootstrap Current

A remarkable neoclassical effect: a **self-generated current** flows without external drive.

Physical origin: trapped particles have unbalanced friction force
- On inward leg: friction slows particles
- On outward leg: different distribution → different friction
- Net momentum transfer → current

The **bootstrap current** is:

$$j_{\text{bs}} \sim \frac{n T}{eB_p} \left(\frac{d \ln p}{dr}\right) f_{\text{bs}}(\nu_*)$$

where $B_p$ is poloidal field and $f_{\text{bs}}$ is a numerical function.

For ITER parameters:
- Bootstrap fraction: $\sim 20-30\%$ of total current
- Reduces need for external current drive

Advanced scenarios aim for $\sim 100\%$ bootstrap current (steady-state operation).

## 6. Python Implementation

### 6.1 Fokker-Planck Solver for Slowing Down

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Physical constants
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m
m_e = 9.109e-31  # kg
k_B = 1.381e-23  # J/K

def slowing_down_frequency(v, n, T_e, ln_Lambda=15):
    """
    Slowing-down frequency for test particle in thermal background.

    Parameters:
    -----------
    v : float
        Test particle velocity (m/s)
    n : float
        Background density (m^-3)
    T_e : float
        Background temperature (eV)
    ln_Lambda : float
        Coulomb logarithm

    Returns:
    --------
    nu_s : float
        Slowing-down frequency (s^-1)
    """
    T_J = T_e * e  # Convert to Joules
    v_th = np.sqrt(2 * T_J / m_e)

    # Gamma encodes the fundamental Coulomb scattering cross-section:
    # e^4 comes from two charges interacting, ln_Lambda sums contributions
    # from all impact parameters between the 90° deflection radius and Debye length.
    Gamma = n * e**4 * ln_Lambda / (4 * np.pi * epsilon_0**2 * m_e**2)

    # The 1/v^3 dependence is the hallmark of Coulomb drag: fast particles
    # spend less time near each field particle, so the effective cross-section shrinks.
    # The v > 3*v_th threshold identifies the regime where the background
    # acts as a stationary scattering medium (test particle far outruns thermal particles).
    if v > 3 * v_th:
        nu_s = Gamma / v**3
    else:
        # Chandrasekhar's phi function accounts for the fraction of background
        # particles slower than the test particle — only those contribute drag
        # in the same direction. This correction is essential near v ~ v_th where
        # half the background moves faster and partly cancels the drag.
        x = v / v_th
        phi = (np.erf(x) - 2*x*np.exp(-x**2)/np.sqrt(np.pi)) / (2*x**2)
        nu_s = Gamma * phi / v**3

    return nu_s

def dv_dt(v, t, n, T_e, E_field=0):
    """
    Time derivative of velocity including slowing down and electric field.

    Parameters:
    -----------
    v : float
        Current velocity (m/s)
    t : float
        Time (s)
    n : float
        Density (m^-3)
    T_e : float
        Temperature (eV)
    E_field : float
        Electric field (V/m)

    Returns:
    --------
    dvdt : float
        Rate of change of velocity
    """
    if v <= 0:
        return 0.0

    nu_s = slowing_down_frequency(v, n, T_e)

    # The competition between drag (-nu_s * v) and electric acceleration (eE/m_e)
    # determines runaway: when E > E_Dreicer, friction can never balance the field
    # for high-v particles because nu_s * v ∝ 1/v^2 → 0 as v → ∞.
    dvdt = -nu_s * v + e * E_field / m_e

    return dvdt

# Parameters for fusion plasma
n = 1e20  # m^-3
T_e = 10e3  # eV (10 keV)
v_th = np.sqrt(2 * T_e * e / m_e)

# Initial velocity of 3.5 MeV alpha particle
E_alpha = 3.5e6 * e  # Joules
m_alpha = 4 * 1.673e-27  # kg (alpha particle)
v_0 = np.sqrt(2 * E_alpha / m_alpha)

print(f"Thermal velocity: {v_th/1e6:.2f} Mm/s")
print(f"Initial alpha velocity: {v_0/1e6:.2f} Mm/s")
print(f"Velocity ratio v_0/v_th: {v_0/v_th:.1f}")

# Time array
t = np.linspace(0, 2, 1000)  # seconds

# Solve ODE
v_solution = odeint(dv_dt, v_0, t, args=(n, T_e))
v_solution = v_solution.flatten()

# Convert to energy
E_solution = 0.5 * m_alpha * v_solution**2 / e / 1e6  # MeV

# Find slowing-down time (when E drops to thermal energy)
E_thermal = 1.5 * T_e / 1e6  # MeV
idx_thermal = np.argmax(E_solution < E_thermal)
if idx_thermal > 0:
    tau_s = t[idx_thermal]
    print(f"Slowing-down time to thermal energy: {tau_s:.3f} s")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t, v_solution/1e6, 'b-', linewidth=2, label='Test particle')
ax1.axhline(v_th/1e6, color='r', linestyle='--', label='$v_{th}$')
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Velocity (Mm/s)', fontsize=12)
ax1.set_title('Alpha Particle Slowing Down', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.semilogy(t, E_solution, 'b-', linewidth=2, label='Test particle')
ax2.axhline(E_thermal, color='r', linestyle='--', label='Thermal energy')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Energy (MeV)', fontsize=12)
ax2.set_title('Energy vs Time', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('slowing_down.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 Braginskii Transport Coefficients

```python
def braginskii_coefficients(n, T, B, Z=1, A=1, ln_Lambda=15):
    """
    Compute Braginskii transport coefficients.

    Parameters:
    -----------
    n : float or array
        Density (m^-3)
    T : float or array
        Temperature (eV)
    B : float or array
        Magnetic field (T)
    Z : int
        Ion charge number
    A : float
        Ion mass number (in proton masses)
    ln_Lambda : float
        Coulomb logarithm

    Returns:
    --------
    dict with transport coefficients
    """
    # Convert to SI
    T_J = T * e
    m_i = A * 1.673e-27  # kg

    # Braginskii showed tau ~ T^(3/2)/n from the Fokker-Planck collision operator;
    # the T^(3/2) scaling reflects that faster particles have smaller Coulomb cross-sections.
    tau_e = 12 * np.pi**(3/2) * epsilon_0**2 * m_e**(1/2) * T_J**(3/2) / \
            (n * e**4 * ln_Lambda * np.sqrt(2))
    # Ion collision time is longer by sqrt(m_i/m_e) because heavier ions move slower
    # and interact less frequently for the same temperature.
    tau_i = np.sqrt(m_i / m_e) * tau_e

    # Cyclotron frequencies
    omega_ce = e * B / m_e
    omega_ci = Z * e * B / m_i

    # Parallel transport is set by random-walk mean-free-path: κ ~ n*T*τ/m ~ v_th^2/ν.
    # The Braginskii numerical prefactors (3.16, 3.9) come from the full Fokker-Planck
    # solution accounting for the non-Maxwellian correction to the distribution function.
    kappa_par_e = 3.16 * n * T_J * tau_e / m_e
    kappa_par_i = 3.9 * n * T_J * tau_i / m_i
    eta_par = 0.96 * n * T_J * tau_i
    sigma_par = 1.96 * n * e**2 * tau_e / m_e

    # Perpendicular conductivity is suppressed by (ω_c τ)^2 because particles must
    # scatter (break their cyclotron orbit) to step across field lines; each step is
    # only one Larmor radius, giving κ_⊥ ~ nT / (m * ω_c^2 * τ).
    kappa_perp_e = 4.66 * n * T_J / (m_e * omega_ce**2 * tau_e)
    kappa_perp_i = 2.0 * n * T_J / (m_i * omega_ci**2 * tau_i)
    eta_perp_0 = 0.73 * n * T_J / (omega_ci**2 * tau_i)

    # The anisotropy ratio κ_∥/κ_⊥ ~ (ω_c τ)^2 quantifies how strongly
    # the magnetic field suppresses cross-field heat flow.
    chi_e = kappa_par_e / kappa_perp_e
    chi_i = kappa_par_i / kappa_perp_i

    return {
        'tau_e': tau_e,
        'tau_i': tau_i,
        'kappa_par_e': kappa_par_e,
        'kappa_par_i': kappa_par_i,
        'kappa_perp_e': kappa_perp_e,
        'kappa_perp_i': kappa_perp_i,
        'eta_par': eta_par,
        'eta_perp_0': eta_perp_0,
        'sigma_par': sigma_par,
        'chi_e': chi_e,
        'chi_i': chi_i,
        'omega_ce_tau': omega_ce * tau_e,
        'omega_ci_tau': omega_ci * tau_i
    }

# ITER-like parameters
n = 1e20  # m^-3
T = np.logspace(2, 4, 100)  # eV, 100 eV to 10 keV
B = 5.3  # T

results = braginskii_coefficients(n, T, B, Z=1, A=2)  # Deuterium

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Thermal conductivity
ax1.loglog(T, results['kappa_par_e'], 'b-', linewidth=2, label='$\\kappa_{\\parallel e}$')
ax1.loglog(T, results['kappa_perp_e'], 'b--', linewidth=2, label='$\\kappa_{\\perp e}$')
ax1.loglog(T, results['kappa_par_i'], 'r-', linewidth=2, label='$\\kappa_{\\parallel i}$')
ax1.loglog(T, results['kappa_perp_i'], 'r--', linewidth=2, label='$\\kappa_{\\perp i}$')
ax1.set_xlabel('Temperature (eV)', fontsize=12)
ax1.set_ylabel('Thermal Conductivity (W/m/K)', fontsize=12)
ax1.set_title('Thermal Conductivity', fontsize=14)
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

# Anisotropy
ax2.loglog(T, results['chi_e'], 'b-', linewidth=2, label='Electrons')
ax2.loglog(T, results['chi_i'], 'r-', linewidth=2, label='Ions')
ax2.set_xlabel('Temperature (eV)', fontsize=12)
ax2.set_ylabel('$\\kappa_{\\parallel} / \\kappa_{\\perp}$', fontsize=12)
ax2.set_title('Thermal Conductivity Anisotropy', fontsize=14)
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

# Resistivity
eta_classical = 1 / results['sigma_par']
ax3.loglog(T, eta_classical, 'b-', linewidth=2)
ax3.set_xlabel('Temperature (eV)', fontsize=12)
ax3.set_ylabel('Resistivity ($\\Omega\\cdot$m)', fontsize=12)
ax3.set_title('Classical Resistivity ($\\eta \\propto T^{-3/2}$)', fontsize=14)
ax3.grid(True, which='both', alpha=0.3)

# Collision time
ax4.loglog(T, results['tau_e']*1e6, 'b-', linewidth=2, label='Electrons')
ax4.loglog(T, results['tau_i']*1e6, 'r-', linewidth=2, label='Ions')
ax4.set_xlabel('Temperature (eV)', fontsize=12)
ax4.set_ylabel('Collision Time ($\\mu$s)', fontsize=12)
ax4.set_title('Collision Time', fontsize=14)
ax4.legend()
ax4.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('braginskii_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

# Print some typical values
T_typical = 1e4  # 10 keV
idx = np.argmin(np.abs(T - T_typical))
print(f"\nTypical values at T = {T_typical/1e3:.1f} keV, B = {B} T:")
print(f"  Electron collision time: {results['tau_e'][idx]*1e6:.2e} μs")
print(f"  ωce τ: {results['omega_ce_tau'][idx]:.2e}")
print(f"  κ_∥e / κ_⊥e: {results['chi_e'][idx]:.2e}")
print(f"  Classical resistivity: {eta_classical[idx]:.2e} Ω·m")
print(f"  Parallel thermal conductivity (e): {results['kappa_par_e'][idx]:.2e} W/m/K")
```

### 6.3 Neoclassical Transport Regimes

```python
def neoclassical_diffusion(r, R, B_0, n, T, Z=1, A=1):
    """
    Estimate neoclassical diffusion coefficient in different regimes.

    Parameters:
    -----------
    r : float
        Minor radius position (m)
    R : float
        Major radius (m)
    B_0 : float
        Magnetic field on axis (T)
    n : float
        Density (m^-3)
    T : float
        Temperature (eV)
    Z : int
        Ion charge
    A : float
        Ion mass number

    Returns:
    --------
    dict with diffusion coefficients and collisionality
    """
    epsilon = r / R  # Inverse aspect ratio
    m_i = A * 1.673e-27
    T_J = T * e

    # Larmor radius
    v_th = np.sqrt(2 * T_J / m_i)
    omega_ci = Z * e * B_0 / m_i
    rho_L = v_th / omega_ci

    # Collision frequency
    ln_Lambda = 15
    tau_i = 12 * np.pi**(3/2) * epsilon_0**2 * np.sqrt(m_i) * T_J**(3/2) / \
            (n * Z**4 * e**4 * ln_Lambda * np.sqrt(2))
    nu_ii = 1 / tau_i

    # Bounce frequency is the rate at which a trapped particle traverses its banana orbit;
    # v_th / (π R √ε) arises because the trapped particle covers a poloidal arc ~π R
    # at its parallel velocity v_th * √ε (only the small parallel component survives).
    omega_b = v_th / (np.pi * R * np.sqrt(epsilon))

    # ν* < 1 means particles complete many bounces before scattering — the banana regime.
    # ν* quantifies whether geometry (bouncing) or collisions dominate particle dynamics.
    nu_star = nu_ii / omega_b

    # Classical diffusion is a random walk with step size ρ_L and step rate ν_ii;
    # this is the Bohm-free-path estimate before any toroidal geometry corrections.
    D_classical = rho_L**2 * nu_ii

    # In the banana regime, the effective step size is the banana width ~ ρ_L / √ε,
    # much larger than the Larmor radius, giving D_nc ~ D_classical / ε^(3/2).
    if nu_star < 0.1:  # Banana regime
        D_nc = D_classical / epsilon**(3/2)
        regime = "Banana"
    elif nu_star < 10:  # Plateau regime
        # In the plateau regime the orbit-averaging partially recovers classical scaling;
        # the ε^(-1) enhancement comes from the fraction of trapped particles ~ √ε
        # combined with their widened step size.
        D_nc = D_classical / epsilon
        regime = "Plateau"
    else:  # Pfirsch-Schlüter
        # In the high-collisionality PS regime, the q^2 factor arises because the
        # poloidal circulation of particles generates a return current that enhances
        # the effective cross-field step by the safety factor q.
        q = r * B_0 / (R * (B_0 * r / R))  # Approximate safety factor
        D_nc = D_classical * q**2
        regime = "Pfirsch-Schlüter"

    return {
        'D_classical': D_classical,
        'D_neoclassical': D_nc,
        'nu_star': nu_star,
        'regime': regime,
        'epsilon': epsilon,
        'rho_L': rho_L,
        'omega_b': omega_b,
        'nu_ii': nu_ii
    }

# ITER parameters
R = 6.2  # m
a = 2.0  # m
B_0 = 5.3  # T
n = 1e20  # m^-3

r_array = np.linspace(0.1, a, 50)
T_array = np.array([1e3, 5e3, 10e3, 20e3])  # eV

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

colors = ['blue', 'green', 'orange', 'red']

for i, T in enumerate(T_array):
    D_class = []
    D_nc = []
    nu_star_array = []

    for r in r_array:
        result = neoclassical_diffusion(r, R, B_0, n, T, Z=1, A=2)
        D_class.append(result['D_classical'])
        D_nc.append(result['D_neoclassical'])
        nu_star_array.append(result['nu_star'])

    ax1.semilogy(r_array, D_nc, color=colors[i], linewidth=2,
                 label=f'T = {T/1e3:.0f} keV')
    ax2.loglog(r_array, nu_star_array, color=colors[i], linewidth=2,
               label=f'T = {T/1e3:.0f} keV')
    ax3.semilogy(r_array, np.array(D_nc) / np.array(D_class),
                 color=colors[i], linewidth=2, label=f'T = {T/1e3:.0f} keV')

ax1.set_xlabel('Minor Radius (m)', fontsize=12)
ax1.set_ylabel('Neoclassical Diffusion ($m^2/s$)', fontsize=12)
ax1.set_title('Neoclassical Diffusion Coefficient', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.axhline(1, color='k', linestyle='--', alpha=0.5, label='Regime boundaries')
ax2.axhline(10, color='k', linestyle='--', alpha=0.5)
ax2.text(0.5, 0.05, 'Banana', transform=ax2.transAxes, fontsize=11)
ax2.text(0.5, 0.3, 'Plateau', transform=ax2.transAxes, fontsize=11)
ax2.text(0.5, 0.7, 'Pfirsch-Schlüter', transform=ax2.transAxes, fontsize=11)
ax2.set_xlabel('Minor Radius (m)', fontsize=12)
ax2.set_ylabel('Collisionality $\\nu_*$', fontsize=12)
ax2.set_title('Collisionality Parameter', fontsize=14)
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

ax3.set_xlabel('Minor Radius (m)', fontsize=12)
ax3.set_ylabel('$D_{nc} / D_{classical}$', fontsize=12)
ax3.set_title('Neoclassical Enhancement Factor', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Bootstrap current coefficient
epsilon_array = r_array / R
L_p = 1.0  # Pressure scale length (m)
beta_p = 0.5  # Poloidal beta

# Bootstrap current arises because the banana-orbit drift is asymmetric: on the
# outer leg, trapped particles carry a net counter-current that the passing particles
# must compensate, producing a self-driven current proportional to √ε (trapped fraction).
# The (1 + ε^2) denominator regularises the formula at large ε where geometry saturates.
j_bs_coeff = epsilon_array**(1/2) / (1 + epsilon_array**2)

ax4.plot(r_array, j_bs_coeff, 'b-', linewidth=2)
ax4.set_xlabel('Minor Radius (m)', fontsize=12)
ax4.set_ylabel('Bootstrap Current Coefficient', fontsize=12)
ax4.set_title('Bootstrap Current Profile Shape', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neoclassical_transport.png', dpi=150, bbox_inches='tight')
plt.show()

# Print regime information at r = a/2
r_mid = a / 2
for T in T_array:
    result = neoclassical_diffusion(r_mid, R, B_0, n, T, Z=1, A=2)
    print(f"\nAt T = {T/1e3:.0f} keV, r = {r_mid:.1f} m:")
    print(f"  Regime: {result['regime']}")
    print(f"  ν* = {result['nu_star']:.2f}")
    print(f"  D_nc / D_classical = {result['D_neoclassical']/result['D_classical']:.1f}")
```

## Summary

Collisional kinetics bridges the gap between collisionless Vlasov theory and fluid descriptions:

**Boltzmann collision operator**:
- Describes binary collisions with arbitrary cross-sections
- Conservation of particles, momentum, and energy built in
- H-theorem: entropy increases, driving toward Maxwellian equilibrium

**Fokker-Planck equation**:
- Specialized to Coulomb collisions: many small-angle deflections
- Dynamical friction (drag) and velocity diffusion
- Rosenbluth potentials provide efficient computational approach

**Test particle dynamics**:
- Slowing down: $dv/dt \propto -v^{-3}$ for fast particles
- Critical velocity separates electron vs ion heating
- Runaway electrons when electric field exceeds Dreicer limit

**Braginskii transport**:
- Moment approach yields fluid equations with collision terms
- Parallel transport: efficient along field lines
- Perpendicular transport: suppressed by factor $(\omega_c \tau)^{-2} \sim 10^{-12}$
- Classical resistivity: $\eta \propto T^{-3/2}$

**Neoclassical effects**:
- Toroidal geometry creates trapped particles in banana orbits
- Three regimes: Pfirsch-Schlüter (high $\nu_*$), plateau, banana (low $\nu_*$)
- Enhancement over classical transport: factors of $q^2$ or $\epsilon^{-3/2}$
- Bootstrap current: self-generated current from pressure gradients

**Anomalous transport**:
- Real plasmas show transport 100× classical predictions
- Caused by turbulence, not collisions
- Empirical scaling laws guide fusion reactor design

These collision theories are essential for:
- Predicting confinement times in fusion devices
- Understanding current drive and heating mechanisms
- Analyzing astrophysical plasmas (solar corona, accretion disks)
- Designing diagnostics and control systems

## Practice Problems

### Problem 1: Collision Frequencies
A hydrogen plasma has $n = 10^{19}$ m$^{-3}$, $T_e = T_i = 1$ keV.

(a) Calculate the electron-electron collision frequency $\nu_{ee}$ using $\ln\Lambda = 15$.

(b) Calculate the electron-ion collision frequency $\nu_{ei}$.

(c) Calculate the energy equilibration time $\tau_{eq}$ between electrons and ions.

(d) Compare the collision time to the plasma period $2\pi/\omega_{pe}$ and the electron thermal transit time across 1 meter. What does this tell you about the plasma's collisionality?

### Problem 2: Alpha Particle Slowing Down
A 3.5 MeV alpha particle (from D-T fusion) is born in a plasma with $n = 5 \times 10^{19}$ m$^{-3}$, $T_e = 15$ keV, $T_i = 12$ keV.

(a) Calculate the critical energy $E_c$ separating electron and ion heating.

(b) What fraction of the alpha energy goes to electrons vs ions?

(c) Estimate the slowing-down time from birth energy to thermal energy.

(d) If the energy confinement time is $\tau_E = 3$ s, will the alphas thermalize before being lost? What is the implication for self-heating in a reactor?

### Problem 3: Classical vs Neoclassical Transport
Consider a tokamak with $R = 3$ m, $a = 1$ m, $B_0 = 2$ T, $n = 5 \times 10^{19}$ m$^{-3}$, $T_i = 5$ keV.

(a) At $r = 0.5$ m, calculate the collisionality $\nu_*$ for deuterium ions.

(b) Identify the neoclassical regime (banana, plateau, or Pfirsch-Schlüter).

(c) Calculate the ratio $D_{nc}/D_{classical}$.

(d) If anomalous transport gives $D_{anomalous} = 1$ m$^2$/s, how does this compare to classical and neoclassical predictions? What does this tell you about the dominant transport mechanism?

### Problem 4: Perpendicular vs Parallel Transport
A plasma has $n = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $B = 5$ T.

(a) Calculate the parallel thermal conductivity $\kappa_{\parallel,e}$.

(b) Calculate the perpendicular thermal conductivity $\kappa_{\perp,e}$.

(c) If a temperature gradient of $dT/dx = 10^6$ K/m exists perpendicular to the field, what is the cross-field heat flux?

(d) What temperature gradient parallel to the field would give the same heat flux? Comment on the implications for temperature profile control in magnetic confinement.

### Problem 5: Bootstrap Current
A tokamak has a pressure profile $p(r) = p_0(1 - r^2/a^2)^2$ with $p_0 = 5 \times 10^5$ Pa, $a = 2$ m.

(a) Calculate the pressure gradient at $r = 1$ m.

(b) Using the approximate formula $j_{bs} \sim \epsilon^{1/2}/(1+\epsilon^2) \cdot (n T/e B_p)(dp/dr)$, estimate the bootstrap current density at $r = 1$ m. Assume $R = 6$ m, $B_p = 0.5$ T.

(c) If the total plasma current is $I_p = 15$ MA, estimate the bootstrap current fraction.

(d) Why is a high bootstrap fraction desirable for a fusion reactor?

---

**Previous**: [8. Landau Damping](./08_Landau_Damping.md)
**Next**: [10. Electrostatic Waves](./10_Electrostatic_Waves.md)

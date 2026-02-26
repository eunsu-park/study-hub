# 7. Vlasov Equation

## Learning Objectives

- Understand phase space and the distribution function $f(\mathbf{x},\mathbf{v},t)$
- Derive the Vlasov equation from Liouville's theorem for collisionless plasmas
- Compute macroscopic quantities (density, bulk velocity, pressure) as moments of $f$
- Explore equilibrium distribution functions (Maxwellian, bi-Maxwellian, kappa)
- Analyze conservation laws (particles, momentum, energy, entropy) from the Vlasov equation
- Implement numerical solutions of the Vlasov equation using Python

## 1. Phase Space and Distribution Function

### 1.1 Phase Space

For a single particle, the **phase space** is the 6-dimensional space of positions and velocities:

$$
(\mathbf{x}, \mathbf{v}) = (x, y, z, v_x, v_y, v_z)
$$

For $N$ particles, the full phase space is $6N$-dimensional. But for large $N$ (plasmas have $\sim 10^{20}$ particles!), tracking individual particles is impractical.

Instead, we use a statistical description via the **distribution function**.

### 1.2 Distribution Function

The distribution function $f(\mathbf{x}, \mathbf{v}, t)$ gives the **number density of particles** in phase space:

$$
dN = f(\mathbf{x}, \mathbf{v}, t) \, d^3x \, d^3v
$$

**Interpretation**: $f\,d^3x\,d^3v$ is the number of particles in the infinitesimal volume $d^3x$ around $\mathbf{x}$ with velocities in $d^3v$ around $\mathbf{v}$ at time $t$.

```
    Phase space (6D)

    v_z ↑
        |       • particle
        |     /
        |    /  represented by
        |   /   density f(x,v,t)
        |  /
        | /________________→ v_x
       /
      / v_y
     ↓

    Position space x,y,z (3D)
```

### 1.3 Normalization

The total number of particles in a volume $V$ is:

$$
N(t) = \int_V d^3x \int_{-\infty}^{\infty} f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

For the entire plasma:

$$
N_{\text{total}} = \int_{\text{all space}} d^3x \int_{-\infty}^{\infty} f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

### 1.4 Moments: Macroscopic Quantities

Macroscopic quantities are obtained by integrating $f$ over velocity space (**moments**):

**Number density**:
$$
n(\mathbf{x}, t) = \int f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

**Bulk (fluid) velocity**:
$$
\mathbf{u}(\mathbf{x}, t) = \frac{1}{n(\mathbf{x}, t)} \int \mathbf{v} f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

**Pressure tensor**:
$$
\mathbf{P}(\mathbf{x}, t) = m \int (\mathbf{v} - \mathbf{u})(\mathbf{v} - \mathbf{u}) f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

where the outer product $(\mathbf{v} - \mathbf{u})(\mathbf{v} - \mathbf{u})$ gives a tensor.

**Scalar pressure** (for isotropic distribution):
$$
P = \frac{1}{3}\text{Tr}(\mathbf{P}) = \frac{m}{3}\int |\mathbf{v} - \mathbf{u}|^2 f \, d^3v
$$

**Temperature** (kinetic definition):
$$
T = \frac{P}{nk_B} = \frac{m}{3nk_B}\int |\mathbf{v} - \mathbf{u}|^2 f \, d^3v
$$

**Energy density**:
$$
\mathcal{E} = \frac{m}{2}\int v^2 f \, d^3v = \frac{1}{2}m n u^2 + \frac{3}{2}nk_BT
$$

(kinetic energy = bulk flow + thermal energy)

### 1.5 Example: 1D Velocity Distribution

Consider a 1D problem where $f = f(v_x)$ is Gaussian:

$$
f(v_x) = n_0 \sqrt{\frac{m}{2\pi k_B T}} \exp\left(-\frac{m(v_x - u)^2}{2k_BT}\right)
$$

Moments:
- $\int f \, dv_x = n_0$ (density)
- $\int v_x f \, dv_x = n_0 u$ (momentum density)
- $\int (v_x - u)^2 f \, dv_x = n_0 k_BT/m$ (variance)

## 2. Vlasov Equation

### 2.1 Derivation from Liouville's Theorem

In classical mechanics, **Liouville's theorem** states that phase space density is conserved along trajectories:

$$
\frac{df}{dt} = 0
$$

Expanding the total derivative:

$$
\frac{\partial f}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot\frac{\partial f}{\partial \mathbf{x}} + \frac{d\mathbf{v}}{dt}\cdot\frac{\partial f}{\partial \mathbf{v}} = 0
$$

For a charged particle:
- $\frac{d\mathbf{x}}{dt} = \mathbf{v}$
- $\frac{d\mathbf{v}}{dt} = \frac{q}{m}(\mathbf{E} + \mathbf{v}\times\mathbf{B})$

Substituting:

$$
\boxed{\frac{\partial f}{\partial t} + \mathbf{v}\cdot\nabla f + \frac{q}{m}(\mathbf{E} + \mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial \mathbf{v}} = 0}
$$

This is the **Vlasov equation** (also called **collisionless Boltzmann equation**).

### 2.2 Physical Interpretation

The Vlasov equation states that the distribution function is **convected** through phase space by the particle trajectories, with no sources or sinks (for collisionless plasma).

```
    Phase space flow

       f(x,v,t)   →   f(x+vδt, v+aδt, t+δt)

    Particles flow along trajectories in (x,v) space
    Distribution function is "painted" on phase space
    and advected by the flow
```

Each term:
- $\frac{\partial f}{\partial t}$: explicit time dependence
- $\mathbf{v}\cdot\nabla f$: advection in position space
- $\frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial \mathbf{v}}$: acceleration in velocity space

### 2.3 Self-Consistency: Vlasov-Maxwell System

The electric and magnetic fields $\mathbf{E}$ and $\mathbf{B}$ are produced by the plasma itself. They must satisfy **Maxwell's equations**:

$$
\nabla\cdot\mathbf{E} = \frac{\rho}{\epsilon_0} = \frac{1}{\epsilon_0}\sum_s q_s \int f_s \, d^3v
$$

$$
\nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}
$$

$$
\nabla\cdot\mathbf{B} = 0
$$

$$
\nabla\times\mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial\mathbf{E}}{\partial t} = \mu_0\sum_s q_s\int\mathbf{v}f_s\,d^3v + \mu_0\epsilon_0\frac{\partial\mathbf{E}}{\partial t}
$$

The coupled system (Vlasov + Maxwell) is the **Vlasov-Maxwell system**, a self-consistent kinetic description of plasmas.

For electrostatic problems (no $\mathbf{B}$ variation), we have the **Vlasov-Poisson system**:

$$
\frac{\partial f}{\partial t} + \mathbf{v}\cdot\nabla f + \frac{q}{m}\mathbf{E}\cdot\frac{\partial f}{\partial \mathbf{v}} = 0
$$

$$
\nabla\cdot\mathbf{E} = \frac{1}{\epsilon_0}\sum_s q_s \int f_s \, d^3v
$$

### 2.4 Multi-Species Plasma

For a plasma with multiple species (electrons, ions, impurities), we have a separate distribution function $f_s$ for each species $s$:

$$
\frac{\partial f_s}{\partial t} + \mathbf{v}\cdot\nabla f_s + \frac{q_s}{m_s}(\mathbf{E} + \mathbf{v}\times\mathbf{B})\cdot\frac{\partial f_s}{\partial \mathbf{v}} = 0
$$

with $\mathbf{E}$ and $\mathbf{B}$ determined by the sum over all species.

## 3. Conservation Laws

### 3.1 Particle Conservation

Integrate the Vlasov equation over all velocities:

$$
\int \frac{\partial f}{\partial t} d^3v + \int \mathbf{v}\cdot\nabla f \, d^3v + \int \frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial \mathbf{v}} d^3v = 0
$$

First term:
$$
\int \frac{\partial f}{\partial t} d^3v = \frac{\partial}{\partial t}\int f \, d^3v = \frac{\partial n}{\partial t}
$$

Second term:
$$
\int \mathbf{v}\cdot\nabla f \, d^3v = \nabla\cdot\int \mathbf{v}f \, d^3v = \nabla\cdot(n\mathbf{u})
$$

Third term (integration by parts, assuming $f\to 0$ as $|\mathbf{v}|\to\infty$):
$$
\int \frac{q}{m}\mathbf{F}\cdot\frac{\partial f}{\partial \mathbf{v}} d^3v = -\frac{q}{m}\int f\nabla_v\cdot\mathbf{F} \, d^3v = 0
$$

because $\nabla_v\cdot(\mathbf{E}+\mathbf{v}\times\mathbf{B}) = \nabla_v\cdot\mathbf{E} + \nabla_v\cdot(\mathbf{v}\times\mathbf{B}) = 0$ (E is independent of v, and the cross product has zero divergence in velocity space).

Result:
$$
\boxed{\frac{\partial n}{\partial t} + \nabla\cdot(n\mathbf{u}) = 0}
$$

**Continuity equation**: particle number is conserved.

### 3.2 Momentum Conservation

Multiply Vlasov equation by $m\mathbf{v}$ and integrate:

$$
\int m\mathbf{v}\frac{\partial f}{\partial t} d^3v + \int m\mathbf{v}(\mathbf{v}\cdot\nabla f) d^3v + \int q\mathbf{v}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial\mathbf{v}} d^3v = 0
$$

After some algebra (integration by parts, etc.):

$$
\frac{\partial}{\partial t}(mn\mathbf{u}) + \nabla\cdot\mathbf{P} = qn(\mathbf{E} + \mathbf{u}\times\mathbf{B})
$$

where $\mathbf{P}$ is the momentum flux tensor (includes pressure and flow).

**Momentum equation**: momentum changes due to electromagnetic force.

### 3.3 Energy Conservation

Multiply by $\frac{1}{2}mv^2$ and integrate:

$$
\frac{\partial}{\partial t}\left(\frac{1}{2}mn\langle v^2\rangle\right) + \nabla\cdot\mathbf{Q} = qn\mathbf{u}\cdot\mathbf{E}
$$

where $\mathbf{Q}$ is the energy flux.

**Energy equation**: kinetic energy changes due to work done by electric field.

(Magnetic field does no work: $\mathbf{v}\times\mathbf{B}\perp\mathbf{v}$)

### 3.4 Entropy and Casimir Invariants

Define the **entropy**:

$$
S = -k_B \int f \ln f \, d^3x \, d^3v
$$

From the Vlasov equation, we can show:

$$
\frac{dS}{dt} = 0
$$

**Entropy is conserved** in collisionless plasmas (reversible dynamics). This is very different from collisional systems where entropy increases (H-theorem).

More generally, any functional of the form:

$$
C = \int G(f) \, d^3x \, d^3v
$$

where $G$ is an arbitrary function, is conserved if $f$ satisfies the Vlasov equation. These are called **Casimir invariants**.

## 4. Equilibrium Distribution Functions

### 4.1 Maxwellian Distribution

The most common equilibrium is the **Maxwellian** (thermal equilibrium):

$$
\boxed{f_0(\mathbf{v}) = n_0 \left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{m|\mathbf{v} - \mathbf{u}|^2}{2k_BT}\right)}
$$

where:
- $n_0$: equilibrium density
- $\mathbf{u}$: bulk drift velocity
- $T$: temperature

**Properties**:
- Isotropic in the frame moving with $\mathbf{u}$
- Maximizes entropy for given density and energy
- Stationary solution of Vlasov equation if $\mathbf{E} = \mathbf{B} = 0$ (or uniform $\mathbf{u} = \mathbf{E}\times\mathbf{B}/B^2$)

### 4.2 Jeans Theorem

**Jeans Theorem**: Any function of constants of motion is a stationary solution of the Vlasov equation.

For a particle in fields $\mathbf{E}$, $\mathbf{B}$, if the energy $H = \frac{1}{2}mv^2 + q\Phi$ is conserved:

$$
f = f(H)
$$

is a stationary solution.

More generally:
$$
f = f(H, \mathbf{P}_{\text{canonical}}, \mu, J, \Phi, \ldots)
$$

where the arguments are any constants of motion (energy, canonical momentum, adiabatic invariants, etc.).

### 4.3 Bi-Maxwellian Distribution

In a magnetized plasma, anisotropy can develop. A common model is the **bi-Maxwellian**:

$$
\boxed{f_0(v_\perp, v_\parallel) = n_0 \frac{m}{2\pi k_B} \frac{1}{T_\perp}\frac{1}{\sqrt{2\pi k_B T_\parallel/m}} \exp\left(-\frac{mv_\perp^2}{2k_BT_\perp} - \frac{m v_\parallel^2}{2k_BT_\parallel}\right)}
$$

where $T_\perp$ and $T_\parallel$ are temperatures perpendicular and parallel to $\mathbf{B}$.

**Anisotropy parameter**:
$$
A = \frac{T_\perp}{T_\parallel} - 1
$$

- $A > 0$: perpendicular heating (e.g., cyclotron resonance heating)
- $A < 0$: parallel heating (e.g., magnetic compression)

Bi-Maxwellian distributions can drive instabilities (e.g., electromagnetic ion cyclotron waves for $A > 0$).

### 4.4 Kappa Distribution

In space plasmas (solar wind, magnetosphere), observations show **non-thermal tails** — more high-energy particles than a Maxwellian predicts. A common model is the **kappa distribution**:

$$
\boxed{f_\kappa(v) = n_0 \frac{1}{(\pi\kappa\theta^2)^{3/2}} \frac{\Gamma(\kappa+1)}{\Gamma(\kappa-1/2)} \left(1 + \frac{v^2}{\kappa\theta^2}\right)^{-(\kappa+1)}}
$$

where:
- $\kappa > 3/2$: spectral index (lower $\kappa$ = fatter tail)
- $\theta^2 = \frac{2k_BT}{m}\frac{\kappa - 3/2}{\kappa}$: thermal speed parameter
- $\Gamma$: Gamma function

**Limits**:
- $\kappa \to \infty$: recovers Maxwellian
- $\kappa \to 3/2$: power-law tail $f \propto v^{-2(\kappa+1)} = v^{-5}$

Kappa distributions arise from:
- Intermittent particle acceleration
- Collisionless relaxation to "quasi-equilibrium"
- Turbulent heating

### 4.5 Drifting Maxwellian (Beam)

A plasma with a beam has two populations:

$$
f = f_{\text{bulk}} + f_{\text{beam}}
$$

For example:
$$
f(v) = n_b\left(\frac{m}{2\pi k_B T_b}\right)^{3/2}\exp\left(-\frac{m(v - v_b)^2}{2k_BT_b}\right) + n_c\left(\frac{m}{2\pi k_B T_c}\right)^{3/2}\exp\left(-\frac{mv^2}{2k_BT_c}\right)
$$

where subscript $b$ = beam, $c$ = core.

**Bump-on-tail** (1D version):

```
    f(v)
      ↑
      |    Core
      |   /‾‾‾\
      |  /     \___
      | /          \___   Beam (bump)
      |/               \__/‾‾\_______
     ─┴───────────────────────────────→ v
                               v_b

    Positive slope df/dv > 0 at resonance
    → Unstable (two-stream instability)
```

This drives instabilities (two-stream, bump-on-tail), transferring energy from the beam to waves.

## 5. Linearization and Perturbation Theory

### 5.1 Equilibrium + Perturbation

For small-amplitude waves, decompose:

$$
f = f_0(\mathbf{v}) + f_1(\mathbf{x}, \mathbf{v}, t)
$$

$$
\mathbf{E} = \mathbf{E}_0 + \mathbf{E}_1(\mathbf{x}, t)
$$

where subscript $0$ = equilibrium, $1$ = perturbation with $|f_1| \ll f_0$.

### 5.2 Linearized Vlasov Equation

Substitute into Vlasov and keep only first-order terms:

$$
\frac{\partial f_1}{\partial t} + \mathbf{v}\cdot\nabla f_1 + \frac{q}{m}(\mathbf{E}_0 + \mathbf{v}\times\mathbf{B}_0)\cdot\frac{\partial f_1}{\partial\mathbf{v}} = -\frac{q}{m}\mathbf{E}_1\cdot\frac{\partial f_0}{\partial\mathbf{v}}
$$

Coupled with linearized Poisson:

$$
\nabla\cdot\mathbf{E}_1 = \frac{1}{\epsilon_0}\sum_s q_s \int f_1^{(s)} \, d^3v
$$

This is the basis for **linear kinetic theory** of plasma waves and instabilities (we'll solve this in Lesson 8 for Landau damping).

### 5.3 BGK Modes

**BGK (Bernstein-Greene-Kruskal) modes** are exact nonlinear solutions of the Vlasov-Poisson system, representing electrostatic wave packets with trapped particles.

For 1D:
$$
f(x, v, t) = f(v - u(x))
$$

where particles are trapped in the potential well of the wave. These are traveling wave solutions with phase velocity $v_{\text{ph}}$ and amplitude determined by the trapping width.

BGK modes represent a balance between:
- Particle trapping (nonlinear effect)
- Wave propagation

They are relevant for:
- Nonlinear Landau damping
- Electron and ion holes
- Double layers

## 6. Python Implementations

### 6.1 Plotting Distribution Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Constants
k_B = 1.380649e-23  # J/K
m_p = 1.67e-27      # kg
m_e = 9.11e-31      # kg
e = 1.6e-19         # C

def maxwellian_1d(v, n, T, m, u=0):
    """
    1D Maxwellian distribution
    """
    return n * np.sqrt(m / (2 * np.pi * k_B * T)) * np.exp(-m * (v - u)**2 / (2 * k_B * T))

def maxwellian_3d(v, n, T, m):
    """
    3D Maxwellian (isotropic, speed distribution)
    f(v) dv = 4π v^2 f(v_vec) dv
    """
    return 4 * np.pi * v**2 * n * (m / (2 * np.pi * k_B * T))**(3/2) * np.exp(-m * v**2 / (2 * k_B * T))

def kappa_1d(v, n, T, m, kappa, u=0):
    """
    1D kappa distribution
    """
    theta_sq = (2 * k_B * T / m) * (kappa - 3/2) / kappa
    norm = n / (np.sqrt(np.pi * kappa * theta_sq)) * gamma(kappa + 1) / gamma(kappa - 1/2)
    return norm * (1 + (v - u)**2 / (kappa * theta_sq))**(-kappa - 1)

def bi_maxwellian_vperp(v_perp, v_para, n, T_perp, T_para, m):
    """
    Bi-Maxwellian: f(v_perp, v_para)
    Here we fix v_para and plot vs v_perp
    """
    return n * (m / (2 * np.pi * k_B * T_perp)) * np.sqrt(m / (2 * np.pi * k_B * T_para)) * \
           np.exp(-m * v_perp**2 / (2 * k_B * T_perp) - m * v_para**2 / (2 * k_B * T_para))

# Parameters
n0 = 1e19  # m^-3
T_eV = 100  # eV
T = T_eV * e / k_B  # Kelvin
m = m_p

v = np.linspace(-5e5, 5e5, 1000)  # m/s

# Plot 1D distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Maxwellian
ax = axes[0, 0]
f_max = maxwellian_1d(v, n0, T, m)
ax.plot(v/1e3, f_max, 'b-', linewidth=2, label='Maxwellian')
ax.set_xlabel('v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('1D Maxwellian Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Drifting Maxwellian (beam)
ax = axes[0, 1]
f_core = maxwellian_1d(v, n0*0.9, T, m, u=0)
f_beam = maxwellian_1d(v, n0*0.1, T*0.5, m, u=2e5)
f_total = f_core + f_beam
ax.plot(v/1e3, f_core, 'b-', linewidth=1, label='Core')
ax.plot(v/1e3, f_beam, 'r-', linewidth=1, label='Beam')
ax.plot(v/1e3, f_total, 'k-', linewidth=2, label='Total')
ax.set_xlabel('v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('Bump-on-Tail Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Kappa distribution comparison
ax = axes[1, 0]
kappa_values = [3, 5, 10, 100]
colors = ['red', 'orange', 'green', 'blue']
for kappa_val, color in zip(kappa_values, colors):
    # kappa must be > 3/2 for the distribution to be normalisable (the integral over
    # all velocities converges only when the power-law exponent κ+1 > 5/2).
    if kappa_val > 3/2:
        f_kappa = kappa_1d(v, n0, T, m, kappa_val)
        # κ = 100 is numerically indistinguishable from Maxwellian on linear scale,
        # but labelling it "κ → ∞" shows students the theoretical limit explicitly.
        label = f'κ = {kappa_val}' if kappa_val < 100 else 'κ → ∞ (Maxwellian)'
        ax.plot(v/1e3, f_kappa, color=color, linewidth=2, label=label)

ax.set_xlabel('v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('Kappa Distributions (Non-thermal Tails)', fontsize=14, fontweight='bold')
# Log scale is essential here: the suprathermal enhancement is only visible in the
# far tail (v >> v_th), where f spans many decades. A linear scale would make all
# curves look identical near the peak, hiding the key physics of the power-law tail.
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend()

# 3D speed distribution
ax = axes[1, 1]
v_speed = np.linspace(0, 6e5, 1000)
f_3d = maxwellian_3d(v_speed, n0, T, m)
v_th = np.sqrt(2 * k_B * T / m)
ax.plot(v_speed/1e3, f_3d, 'b-', linewidth=2)
ax.axvline(x=v_th/1e3, color='r', linestyle='--', linewidth=2,
          label=f'v_th = {v_th/1e3:.1f} km/s')
ax.set_xlabel('Speed v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('3D Maxwellian Speed Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('distribution_functions.png', dpi=150)
print("Saved: distribution_functions.png")

# Bi-Maxwellian
fig, ax = plt.subplots(figsize=(10, 6))

v_perp_array = np.linspace(0, 5e5, 1000)
T_perp_eV = 200
T_para_eV = 50
T_perp = T_perp_eV * e / k_B
T_para = T_para_eV * e / k_B

# Different v_para slices
v_para_values = [0, 1e5, 2e5, 3e5]
colors = ['blue', 'green', 'orange', 'red']

for v_para, color in zip(v_para_values, colors):
    f_bi = bi_maxwellian_vperp(v_perp_array, v_para, n0, T_perp, T_para, m)
    ax.plot(v_perp_array/1e3, f_bi, color=color, linewidth=2,
           label=f'v_para = {v_para/1e3:.0f} km/s')

ax.set_xlabel('v_perp (km/s)', fontsize=12)
ax.set_ylabel('f(v_perp, v_para) (s/m⁴)', fontsize=12)
ax.set_title(f'Bi-Maxwellian: T_perp = {T_perp_eV} eV, T_para = {T_para_eV} eV',
            fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend()
plt.tight_layout()
plt.savefig('bi_maxwellian.png', dpi=150)
print("Saved: bi_maxwellian.png")
```

### 6.2 Moments Calculation

```python
from scipy.integrate import simps

def compute_moments(v_array, f_array):
    """
    Compute moments of 1D distribution function
    """
    # Use Simpson's rule (simps) rather than np.trapz because simps has O(h^4) error
    # vs O(h^2) for trapz — important when f has smooth but curved tails, which is
    # typical of Maxwellian and kappa distributions sampled on a coarse velocity grid.
    n = simps(f_array, v_array)

    # Divide by n (not n0) so the formula gives the true mean velocity even when the
    # distribution has been perturbed away from the reference density n0.
    u = simps(v_array * f_array, v_array) / n

    # Variance (temperature measure)
    # Compute variance relative to the calculated mean u (not 0) to get the thermal
    # variance correctly for a drifting distribution where u ≠ 0.
    var = simps((v_array - u)**2 * f_array, v_array) / n

    # Thermal velocity
    v_th = np.sqrt(var)

    return n, u, v_th, var

# Test with Maxwellian
v_test = np.linspace(-1e6, 1e6, 10000)
n_test = 1e19
T_test = 100 * e / k_B
u_test = 1e5  # drifting

f_test = maxwellian_1d(v_test, n_test, T_test, m_p, u=u_test)

n_calc, u_calc, v_th_calc, var_calc = compute_moments(v_test, f_test)

print("\n=== Moment Calculation ===")
print(f"Input:")
print(f"  n = {n_test:.2e} m^-3")
print(f"  u = {u_test:.2e} m/s")
print(f"  T = {T_test*k_B/e:.2f} eV")
print(f"  v_th (expected) = {np.sqrt(2*k_B*T_test/m_p):.2e} m/s")

print(f"\nCalculated from distribution:")
print(f"  n = {n_calc:.2e} m^-3")
print(f"  u = {u_calc:.2e} m/s")
print(f"  v_th = {v_th_calc:.2e} m/s")
print(f"  T = {m_p*var_calc/k_B/2:.2f} K = {m_p*var_calc/e/2:.2f} eV")

errors = [
    abs(n_calc - n_test) / n_test,
    abs(u_calc - u_test) / abs(u_test),
    abs(v_th_calc - np.sqrt(2*k_B*T_test/m_p)) / np.sqrt(2*k_B*T_test/m_p)
]
print(f"\nRelative errors: {errors}")
```

### 6.3 Simple 1D Vlasov Solver (Operator Splitting)

```python
def vlasov_1d_solver(x, v, f0, E_func, dt, num_steps, q, m):
    """
    Simple 1D Vlasov solver using operator splitting

    x: position grid
    v: velocity grid
    f0: initial distribution f(x,v,t=0)
    E_func: function E(x,t) giving electric field
    dt: timestep
    num_steps: number of steps
    q, m: charge and mass

    Returns: f(x,v,t) at final time
    """
    f = f0.copy()
    dx = x[1] - x[0]
    dv = v[1] - v[0]
    Nx, Nv = len(x), len(v)

    # Storage for snapshots
    snapshots = []
    snapshot_times = []

    for n in range(num_steps):
        t = n * dt

        # Step 1: Advection in x (∂f/∂t + v ∂f/∂x = 0)
        # Operator splitting separates the 6D Vlasov equation into two 1D advections
        # (in x and v), each of which is solved independently. This is valid when dt
        # is small (Strang splitting gives O(dt^2) accuracy overall).
        # Use upwind scheme: upwind differencing is chosen here because it is stable
        # (dissipative) for the advection equation. The stencil always takes the
        # derivative from the direction the information is traveling — upstream —
        # preventing non-physical oscillations from appearing in f.
        f_new = np.zeros_like(f)
        for j in range(Nv):
            for i in range(Nx):
                if v[j] > 0:
                    # Particle moves in +x; information comes from the left cell.
                    # Periodic BC (% Nx) enforces the assumption that f is the same
                    # at x=0 and x=L — appropriate for a spatially periodic plasma wave.
                    i_up = (i - 1) % Nx  # periodic BC
                    f_new[i, j] = f[i, j] - v[j] * dt / dx * (f[i, j] - f[i_up, j])
                else:
                    i_up = (i + 1) % Nx
                    f_new[i, j] = f[i, j] - v[j] * dt / dx * (f[i_up, j] - f[i, j])
        f = f_new.copy()

        # Step 2: Acceleration in v (∂f/∂t + (q/m)E ∂f/∂v = 0)
        # The electric field is re-evaluated at the current time t (not t+dt) to keep
        # the explicit time-stepping first-order. Using E at t makes this equivalent
        # to a forward Euler step in v-space — simple but requires small dt for accuracy.
        E = E_func(x, t)
        f_new = np.zeros_like(f)
        for i in range(Nx):
            a = q * E[i] / m  # acceleration = q*E/m (Newton's law in velocity space)
            for j in range(Nv):
                if a > 0:
                    # Positive acceleration shifts f toward higher v: use left-neighbor
                    # (lower v) as the upwind cell. Clamp at boundary (j_up = 0) to
                    # prevent particles from wrapping around in velocity space —
                    # unlike position, velocity has physical limits in this setup.
                    j_up = max(j - 1, 0)
                    f_new[i, j] = f[i, j] - a * dt / dv * (f[i, j] - f[i, j_up])
                else:
                    j_up = min(j + 1, Nv - 1)
                    f_new[i, j] = f[i, j] - a * dt / dv * (f[i, j_up] - f[i, j])
        f = f_new.copy()

        # Save snapshots
        if n % (num_steps // 10) == 0:
            snapshots.append(f.copy())
            snapshot_times.append(t)

    return f, snapshots, snapshot_times

# Setup 1D problem
Nx, Nv = 128, 128
# Lx = 2π/k sets the box to exactly one wavelength, ensuring the periodic BC is
# consistent with the wave: f(x=0) = f(x=Lx) by construction, avoiding artificial
# reflections that would arise if the box contained a fractional number of waves.
Lx = 2 * np.pi / 0.5  # wavelength
x = np.linspace(0, Lx, Nx)
# Velocity grid covers ±3×10^5 m/s ≈ ±6 v_th at 100 eV, capturing >99.9% of the
# Maxwellian. Truncating at too few v_th would lose particles and violate conservation;
# extending too far wastes grid points on the exponentially small tail.
v = np.linspace(-3e5, 3e5, Nv)

X, V = np.meshgrid(x, v, indexing='ij')

# Initial condition: perturbed Maxwellian
n0 = 1e19
T0 = 100 * e / k_B
k_wave = 0.5  # wavenumber (1/m)
# Small amplitude (1%) keeps the perturbation in the linear regime so the simulation
# result can be compared directly with linear wave theory (Landau damping, etc.).
# Larger amplitudes would drive particle trapping and nonlinear saturation.
amplitude = 0.01

f0 = np.zeros((Nx, Nv))
for i in range(Nx):
    n_pert = n0 * (1 + amplitude * np.cos(k_wave * x[i]))
    f0[i, :] = maxwellian_1d(v, n_pert, T0, m_e, u=0)

# Electric field (for this demo, use a static wave)
def E_field(x, t):
    E0 = 1e2  # V/m
    return E0 * np.sin(k_wave * x) * np.cos(1e5 * t)

# Solve
print("\nSolving 1D Vlasov equation...")
dt = 1e-8  # s
num_steps = 500

f_final, snapshots, times = vlasov_1d_solver(x, v, f0, E_field, dt, num_steps, -e, m_e)

print(f"Completed {num_steps} steps, final time = {times[-1]:.2e} s")

# Plot phase space evolution
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (snap, t) in enumerate(zip([f0] + snapshots[:5], [0] + times[:5])):
    ax = axes[idx]
    im = ax.contourf(x, v/1e3, snap.T, levels=30, cmap='viridis')
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('v (km/s)', fontsize=10)
    ax.set_title(f't = {t:.2e} s', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='f(x,v)')

plt.tight_layout()
plt.savefig('vlasov_1d_evolution.png', dpi=150)
print("Saved: vlasov_1d_evolution.png")

# Density evolution
fig, ax = plt.subplots(figsize=(10, 6))
for idx, (snap, t) in enumerate(zip([f0] + snapshots[::2], [0] + times[::2])):
    n_x = simps(snap, v, axis=1)
    ax.plot(x, n_x/n0, linewidth=1.5, label=f't = {t:.1e} s')

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('n(x,t) / n0', fontsize=12)
ax.set_title('Density Oscillation (1D Vlasov)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vlasov_density.png', dpi=150)
print("Saved: vlasov_density.png")
```

## Summary

In this lesson, we developed the kinetic theory of plasmas using the Vlasov equation:

1. **Phase space and distribution function**: $f(\mathbf{x}, \mathbf{v}, t)$ describes the statistical state of the plasma in 6D phase space.

2. **Moments**: Macroscopic quantities (density, velocity, pressure, temperature) are obtained by integrating $f$ over velocity space.

3. **Vlasov equation**:
   $$
   \frac{\partial f}{\partial t} + \mathbf{v}\cdot\nabla f + \frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial\mathbf{v}} = 0
   $$
   describes collisionless evolution of $f$, derived from Liouville's theorem.

4. **Self-consistency**: Vlasov equation coupled with Maxwell's equations forms the Vlasov-Maxwell system.

5. **Conservation laws**: Vlasov equation conserves particles, momentum, energy, and entropy (Casimir invariants).

6. **Equilibrium distributions**:
   - **Maxwellian**: thermal equilibrium
   - **Bi-Maxwellian**: anisotropic magnetized plasma
   - **Kappa**: non-thermal tails (space plasmas)
   - **Beam**: bump-on-tail (unstable)

7. **Linearization**: Perturbation theory for small-amplitude waves leads to linear kinetic theory (next lesson: Landau damping).

The Vlasov equation is the foundation of kinetic plasma physics, capturing phenomena that fluid models miss:
- Landau damping (collisionless wave damping)
- Kinetic instabilities
- Wave-particle resonances

## Practice Problems

### Problem 1: Moments of a Drifting Maxwellian

Consider a 1D Maxwellian distribution:

$$
f(v) = n_0\sqrt{\frac{m}{2\pi k_B T}}\exp\left(-\frac{m(v-u_0)^2}{2k_BT}\right)
$$

(a) Compute the zeroth moment (density): $n = \int f(v) \, dv$.

(b) Compute the first moment (mean velocity): $\langle v\rangle = \frac{1}{n}\int v f(v) \, dv$.

(c) Compute the second moment (temperature): $T = \frac{m}{k_B}\int (v - \langle v\rangle)^2 f(v) \, dv$.

(d) Verify your results numerically using the Python code provided.

---

### Problem 2: Bi-Maxwellian Anisotropy

A magnetized plasma has a bi-Maxwellian distribution with $T_\perp = 200$ eV and $T_\parallel = 50$ eV.

(a) Calculate the anisotropy parameter $A = T_\perp/T_\parallel - 1$.

(b) Compute the pressure tensor components $P_\perp = nk_BT_\perp$ and $P_\parallel = nk_BT_\parallel$ for $n = 10^{19}$ m$^{-3}$.

(c) Is this distribution isotropic? Stable? (Hint: Large $A > 0$ can drive cyclotron instabilities.)

(d) Estimate the free energy available in the anisotropy: $\Delta W = \frac{3}{2}nk_B(T_\perp - T_\parallel)$.

---

### Problem 3: Kappa Distribution vs Maxwellian

For a kappa distribution with $\kappa = 3$, compute the ratio of the number of particles with $v > 3v_{\text{th}}$ (suprathermal particles) compared to a Maxwellian with the same density and temperature.

(a) For a Maxwellian, the fraction of particles with $v > 3v_{\text{th}}$ is approximately $\exp(-9/2) \approx 0.011$ (1.1%). Calculate this numerically.

(b) For the kappa distribution, compute the same fraction numerically by integrating:
   $$
   f_{\text{super}} = \frac{\int_{3v_{th}}^\infty f_\kappa(v) \, dv}{\int_0^\infty f_\kappa(v) \, dv}
   $$

(c) What is the enhancement factor (kappa/Maxwellian)?

(d) Explain why kappa distributions are observed in the solar wind and magnetosphere.

---

### Problem 4: Conservation of Entropy

Show analytically that the entropy $S = -k_B\int f\ln f \, d^3x\,d^3v$ is conserved by the Vlasov equation.

(a) Start with $\frac{dS}{dt} = -k_B\int\frac{\partial}{\partial t}(f\ln f) \, d^3x\,d^3v$.

(b) Use $\frac{\partial}{\partial t}(f\ln f) = (1 + \ln f)\frac{\partial f}{\partial t}$ and substitute the Vlasov equation.

(c) Integrate by parts and use $\nabla_x\cdot\mathbf{v} = 0$ and $\nabla_v\cdot\mathbf{a} = 0$ for Lorentz acceleration.

(d) Show that all terms vanish, hence $\frac{dS}{dt} = 0$.

---

### Problem 5: Linearized Vlasov-Poisson for Langmuir Waves

Consider a 1D electrostatic perturbation in an unmagnetized plasma:

$$
f = f_0(v) + f_1(x, v, t)
$$

$$
E = E_1(x, t)
$$

where $f_0$ is a Maxwellian at rest.

(a) Write the linearized Vlasov equation for $f_1$.

(b) Write the linearized Poisson equation for $E_1$ in terms of $f_1$.

(c) Assume plane wave solutions: $f_1 \propto e^{i(kx - \omega t)}$, $E_1 \propto e^{i(kx - \omega t)}$. Derive the dispersion relation (you should get the Bohm-Gross relation for real $\omega$; we'll treat Landau damping in Lesson 8).

(d) For $k\lambda_D \ll 1$, show that $\omega^2 \approx \omega_{pe}^2(1 + 3k^2\lambda_D^2)$ where $\lambda_D = \sqrt{\epsilon_0 k_BT/(ne^2)}$.

**Hint**: This problem previews Lesson 8. The full solution requires handling the pole at $v = \omega/k$ (Landau prescription).

---

## Navigation

- **Previous**: [Magnetic Mirrors and Adiabatic Invariants](./06_Magnetic_Mirrors_Adiabatic_Invariants.md)
- **Next**: [Landau Damping](./08_Landau_Damping.md)

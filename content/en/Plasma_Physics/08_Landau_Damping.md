# 8. Landau Damping

## Learning Objectives

- Derive the dispersion relation for electrostatic waves in warm plasmas using linearized Vlasov-Poisson
- Understand the Landau contour and its role in handling the singularity at $v = \omega/k$
- Calculate Landau damping rates and analyze their dependence on plasma parameters
- Explore the physical mechanism of wave-particle energy exchange at resonance
- Study inverse Landau damping and the bump-on-tail instability
- Simulate Landau damping and particle trapping using Python

## 1. Electrostatic Waves in Warm Plasma

### 1.1 Linearized Vlasov-Poisson System

We consider a 1D, unmagnetized, electrostatic plasma. The equilibrium is:

$$
f = f_0(v), \quad \mathbf{E} = 0
$$

where $f_0(v)$ is the equilibrium distribution (typically Maxwellian).

For small perturbations:

$$
f = f_0(v) + f_1(x, v, t), \quad E = E_1(x, t)
$$

with $|f_1| \ll f_0$, $|E_1|$ small.

**Linearized Vlasov equation**:

$$
\frac{\partial f_1}{\partial t} + v\frac{\partial f_1}{\partial x} + \frac{q}{m}E_1\frac{\partial f_0}{\partial v} = 0
$$

**Linearized Poisson equation**:

$$
\frac{\partial E_1}{\partial x} = \frac{1}{\epsilon_0}\sum_s q_s \int f_1^{(s)} \, dv
$$

where the sum is over species $s$ (electrons, ions).

### 1.2 Fourier-Laplace Transform

Assume plane wave solutions:

$$
f_1(x, v, t) = \hat{f}_1(v) e^{i(kx - \omega t)}
$$

$$
E_1(x, t) = \hat{E}_1 e^{i(kx - \omega t)}
$$

where $k$ is the wavenumber and $\omega$ is the (complex) frequency.

Substituting into the linearized Vlasov equation:

$$
-i\omega\hat{f}_1 + ikv\hat{f}_1 + \frac{q}{m}\hat{E}_1\frac{df_0}{dv} = 0
$$

Solving for $\hat{f}_1$:

$$
\hat{f}_1(v) = \frac{iq}{m}\frac{\hat{E}_1}{kv - \omega}\frac{df_0}{dv}
$$

### 1.3 Poisson Equation and Charge Density

From Poisson:

$$
ik\hat{E}_1 = \frac{1}{\epsilon_0}\sum_s q_s \int \hat{f}_1^{(s)} dv
$$

Substituting $\hat{f}_1$:

$$
ik\hat{E}_1 = \frac{1}{\epsilon_0}\sum_s q_s \int \frac{iq_s}{m_s}\frac{\hat{E}_1}{kv - \omega}\frac{df_0^{(s)}}{dv} dv
$$

Canceling $\hat{E}_1$ (assuming $\hat{E}_1 \neq 0$ for non-trivial solutions):

$$
k = \frac{1}{\epsilon_0}\sum_s \frac{q_s^2}{m_s k} \int \frac{1}{v - \omega/k}\frac{df_0^{(s)}}{dv} dv
$$

Rearranging:

$$
1 = \frac{1}{\epsilon_0 k^2}\sum_s \frac{q_s^2}{m_s} \int \frac{1}{v - \omega/k}\frac{df_0^{(s)}}{dv} dv
$$

Or, defining the **dielectric function** $\epsilon(k, \omega)$:

$$
\boxed{\epsilon(k, \omega) = 1 - \sum_s \frac{\omega_{ps}^2}{k^2} \int \frac{\partial f_0^{(s)}/\partial v}{v - \omega/k} dv = 0}
$$

where $\omega_{ps}^2 = n_s q_s^2/(\epsilon_0 m_s)$ is the plasma frequency for species $s$.

**Dispersion relation**: $\epsilon(k, \omega) = 0$.

### 1.4 The Pole at $v = \omega/k$

The integrand has a **pole** at $v = v_{\text{ph}} = \omega/k$ (the phase velocity). This singularity requires careful treatment:

- For real $\omega$, the integral is undefined (principal value + residue).
- The correct prescription comes from causality (Laplace transform with initial conditions).

## 2. Landau Contour and Analytic Continuation

### 2.1 Causality and Laplace Transform

Properly, we should use a Laplace transform in time with $\text{Im}(\omega) > 0$ initially (exponential decay ensures convergence). The integral is then well-defined:

$$
\int \frac{1}{v - \omega/k} dv
$$

with $\text{Im}(\omega/k) < 0$ (the pole is **below** the real axis in velocity space).

After solving for $\omega(k)$, we analytically continue to the physical solution, which may have $\text{Im}(\omega) < 0$ (damping) or $\text{Im}(\omega) > 0$ (growth).

### 2.2 Landau Prescription

The result is the **Landau contour**: the integration path in velocity space goes **below** the pole at $v = \omega/k$.

```
    Complex v-plane

       Im(v)
         ↑
         |
    ─────┼─────────────→ Re(v)
         |        × pole at v = ω/k
         |      (contour goes below)
```

Using the **Plemelj formula**:

$$
\frac{1}{v - v_0 - i0^+} = \mathcal{P}\frac{1}{v - v_0} + i\pi\delta(v - v_0)
$$

where $\mathcal{P}$ denotes the principal value and $\delta$ is the Dirac delta function.

Thus:

$$
\int \frac{\partial f_0/\partial v}{v - \omega/k} dv = \mathcal{P}\int \frac{\partial f_0/\partial v}{v - \omega/k} dv + i\pi\frac{\partial f_0}{\partial v}\bigg|_{v = \omega/k}
$$

### 2.3 Dielectric Function with Landau Prescription

The dielectric function becomes:

$$
\epsilon(k, \omega) = 1 - \sum_s \frac{\omega_{ps}^2}{k^2}\left[\mathcal{P}\int \frac{\partial f_0^{(s)}/\partial v}{v - \omega/k} dv + i\pi\frac{\partial f_0^{(s)}}{\partial v}\bigg|_{v = \omega/k}\right]
$$

Setting $\epsilon = 0$ gives the dispersion relation. Since $\epsilon$ is complex, $\omega$ is generally complex:

$$
\omega = \omega_r + i\gamma
$$

where:
- $\omega_r$: real part (oscillation frequency)
- $\gamma$: imaginary part (growth rate if $\gamma > 0$, damping rate if $\gamma < 0$)

## 3. Landau Damping of Electron Plasma Waves

### 3.1 Electron Plasma Waves (Langmuir Waves)

Consider a plasma with mobile electrons and immobile ions ($m_i \to \infty$). The equilibrium electron distribution is Maxwellian:

$$
f_0(v) = n_0\sqrt{\frac{m_e}{2\pi k_B T_e}}\exp\left(-\frac{m_e v^2}{2k_BT_e}\right)
$$

The derivative is:

$$
\frac{df_0}{dv} = -\frac{m_e v}{k_BT_e}f_0(v)
$$

### 3.2 Dispersion Relation: Real Part

For $|\gamma| \ll \omega_r$, we can approximate $\omega/k \approx \omega_r/k$ in the principal value integral. The real part of $\epsilon = 0$ gives:

$$
1 - \frac{\omega_{pe}^2}{k^2}\mathcal{P}\int \frac{df_0/dv}{v - \omega_r/k} dv = 0
$$

Using integration by parts:

$$
\mathcal{P}\int \frac{df_0/dv}{v - \omega_r/k} dv = -\int f_0(v) \frac{\partial}{\partial v}\left[\mathcal{P}\frac{1}{v - \omega_r/k}\right] dv
$$

For a Maxwellian and $k\lambda_D \ll 1$ (where $\lambda_D = \sqrt{\epsilon_0 k_B T_e/(n_0 e^2)}$ is the Debye length), the result is:

$$
\boxed{\omega_r^2 \approx \omega_{pe}^2 + 3k^2v_{th,e}^2}
$$

where $v_{th,e} = \sqrt{k_BT_e/m_e}$ is the electron thermal velocity.

This is the **Bohm-Gross dispersion relation** for electron plasma waves (Langmuir waves).

### 3.3 Imaginary Part: Damping Rate

The imaginary part of $\epsilon = 0$ gives the damping rate. For small damping ($|\gamma| \ll \omega_r$):

$$
\gamma \approx -\frac{\pi\omega_{pe}^2}{2k^2}\frac{df_0}{dv}\bigg|_{v = \omega_r/k}
$$

For a Maxwellian:

$$
\frac{df_0}{dv}\bigg|_{v = \omega_r/k} = -\frac{m_e\omega_r}{k k_B T_e}f_0(\omega_r/k) = -\frac{m_e\omega_r}{k k_B T_e}n_0\sqrt{\frac{m_e}{2\pi k_B T_e}}\exp\left(-\frac{m_e\omega_r^2}{2k^2k_BT_e}\right)
$$

Simplifying:

$$
\gamma = \frac{\pi\omega_{pe}^2}{2k^2} \cdot \frac{m_e\omega_r}{k k_B T_e}n_0\sqrt{\frac{m_e}{2\pi k_B T_e}}\exp\left(-\frac{\omega_r^2}{2k^2v_{th,e}^2}\right)
$$

Using $\omega_r^2 \approx \omega_{pe}^2(1 + 3k^2\lambda_D^2)$ and $k\lambda_D \ll 1$:

$$
\frac{\omega_r^2}{2k^2v_{th,e}^2} \approx \frac{\omega_{pe}^2}{2k^2v_{th,e}^2} = \frac{1}{2k^2\lambda_D^2}
$$

Thus:

$$
\boxed{\gamma \approx -\sqrt{\frac{\pi}{8}}\frac{\omega_{pe}}{(k\lambda_D)^3}\exp\left(-\frac{1}{2k^2\lambda_D^2}\right)}
$$

**Key features**:
- $\gamma < 0$: damping (not growth)
- $|\gamma| \propto \exp(-1/(2k^2\lambda_D^2))$: exponentially weak for $k\lambda_D \ll 1$
- $|\gamma|/\omega_r \propto (k\lambda_D)^{-3}\exp(-1/(2k^2\lambda_D^2))$: very small for typical plasmas

### 3.4 Validity Condition

Landau damping is significant when:

$$
k\lambda_D \sim 1
$$

For $k\lambda_D \ll 1$, damping is exponentially weak. For $k\lambda_D \gg 1$, the wave is heavily damped (overdamped).

### 3.5 Numerical Example

**Example**: Laboratory plasma with $n_e = 10^{18}$ m$^{-3}$, $T_e = 10$ eV.

Calculate:
- $\omega_{pe} = \sqrt{n_e e^2/(\epsilon_0 m_e)} = 5.64\times 10^{10}$ rad/s
- $\lambda_D = \sqrt{\epsilon_0 k_B T_e/(n_e e^2)} = 2.35\times 10^{-5}$ m
- For $k = 10^5$ m$^{-1}$: $k\lambda_D = 2.35$

Then:
- $\omega_r \approx \omega_{pe}\sqrt{1 + 3(k\lambda_D)^2} \approx 1.23\omega_{pe} = 6.94\times 10^{10}$ rad/s
- $\gamma/\omega_{pe} \approx -0.09\exp(-0.09) \approx -0.082$
- $|\gamma|/\omega_r \approx 0.067$

The wave damps in $\sim 15$ oscillations.

## 4. Physical Mechanism: Wave-Particle Resonance

### 4.1 Resonant Particles

Landau damping arises from resonant particles: those moving with velocity $v \approx v_{\text{ph}} = \omega/k$ (the phase velocity of the wave).

These particles "surf" the wave, exchanging energy with it.

```
    Wave electric field

         E(x,t) = E0 sin(kx - ωt)

    Particle at x = x0, v = v_ph:
    - Sees stationary potential (in wave frame)
    - Can gain or lose energy

    Phase space:
         v
         ↑
         |    •   slow particles (v < v_ph)
         |   ••
         |  •••  ← bulk of distribution
         | •••
         |•••──────→ x
         |  ← v_ph (resonance)
         |
       ••|       fast particles (v > v_ph)
        •|

    For Maxwellian: more slow particles than fast
    → Net energy transfer: wave → particles → damping
```

### 4.2 Energy Exchange

In the wave frame (moving at $v_{\text{ph}}$), the electric field is static. A particle sees:

$$
E(x - v_{\text{ph}}t) = E_0\sin(kx - kv_{\text{ph}}t) = E_0\sin(kx - \omega t)
$$

If the particle velocity in the wave frame is $v' = v - v_{\text{ph}}$:

- $v' > 0$ (particle faster than wave): particle climbs potential hill, loses energy
- $v' < 0$ (particle slower than wave): particle slides down, gains energy

The net energy transfer depends on the **distribution function gradient** at $v = v_{\text{ph}}$:

$$
\frac{df_0}{dv}\bigg|_{v = v_{\text{ph}}}
$$

For a Maxwellian (monotonically decreasing from $v = 0$), $df_0/dv < 0$ at all $v > 0$. There are **more slow particles than fast particles** at resonance.

Result: More particles gain energy (slow) than lose energy (fast) → net energy transfer from wave to particles → **damping**.

### 4.3 Surfing Analogy

Think of surfers on ocean waves:

- **Slow surfers** (behind wave crest): accelerated by wave, gain energy
- **Fast surfers** (ahead of wave crest): decelerated, lose energy
- If there are more slow surfers, net energy transfer from wave to surfers → wave damps

### 4.4 Damping vs Growth

The sign of $df_0/dv$ at resonance determines damping or growth:

$$
\gamma \propto -\frac{df_0}{dv}\bigg|_{v = v_{\text{ph}}}
$$

- $df_0/dv < 0$ (decreasing distribution): $\gamma < 0$ → **damping**
- $df_0/dv > 0$ (increasing distribution): $\gamma > 0$ → **growth** (inverse Landau damping)

## 5. Inverse Landau Damping: Bump-on-Tail Instability

### 5.1 Non-Monotonic Distribution

If $f_0(v)$ has a region where $df_0/dv > 0$ (positive slope), waves with $v_{\text{ph}}$ in that region will grow.

A classic example is the **bump-on-tail** distribution:

$$
f_0(v) = f_{\text{core}}(v) + f_{\text{beam}}(v)
$$

where:
- Core: Maxwellian centered at $v = 0$
- Beam: Maxwellian centered at $v = v_b > 0$ (drifting beam)

```
    f(v)
      ↑
      |   Core
      |  /‾‾\___
      | /       \___   Beam
      |/            \_/‾\____
     ─┴──────────────────────→ v
                      v_b

    Between core and beam: df/dv > 0 → unstable
```

### 5.2 Growth Rate

For a dilute beam ($n_b \ll n_c$), the growth rate is:

$$
\gamma \approx \frac{\pi\omega_{pe}^2}{2k^2}\frac{df_0}{dv}\bigg|_{v = \omega/k}
$$

At the positive slope region:

$$
\gamma > 0 \quad \Rightarrow \quad \text{growth (instability)}
$$

The maximum growth occurs when $v_{\text{ph}}$ matches the steepest positive slope.

### 5.3 Quasilinear Relaxation

As the wave grows, particles near resonance are trapped (see next section) and the distribution flattens:

```
    Initial:  f(v) with bump
              /‾\  ← bump
             /   \_____

    After relaxation:  flattened
              /‾‾‾‾\____
```

The flattening of $df/dv$ at resonance reduces the growth rate. Eventually, the system reaches a **quasilinear plateau** where $df/dv \approx 0$ at resonance, and growth stops.

This is **quasilinear relaxation**: wave growth → particle trapping → distribution flattening → saturation.

### 5.4 Applications

Inverse Landau damping (bump-on-tail instability) occurs in:
- **Electron beams** in plasmas (laboratory, space)
- **Ion beams** in solar wind
- **Current-driven instabilities** (e.g., electron current in fusion plasmas)

## 6. Nonlinear Landau Damping and Particle Trapping

### 6.1 Particle Trapping

When the wave amplitude is large, particles near $v \approx v_{\text{ph}}$ can be **trapped** in the wave potential.

In the wave frame, the potential is:

$$
\Phi(x) = \frac{E_0}{k}\cos(kx)
$$

A particle with small velocity $v' = v - v_{\text{ph}}$ in the wave frame sees a potential well and can execute **bounce oscillations**.

The bounce frequency is:

$$
\omega_b = \sqrt{\frac{ekE_0}{m}} = \sqrt{\frac{eE_0 k}{m}}
$$

### 6.2 Phase Space Vortex

Trapped particles form a **phase space vortex** (cat's eye structure):

```
    Phase space (x, v)

         v
         ↑
         |        •••
         |      ••   ••   ← separatrix
         |     •  ⊗  •      (trapped particles)
         |      ••   ••
         |        •••
         |──────────────→ x
              λ = 2π/k

    ⊗ = wave fixed point (v = v_ph)
    Particles inside separatrix are trapped
    Particles outside are passing
```

The separatrix (boundary between trapped and passing) corresponds to energy:

$$
W_{\text{sep}} = e\Phi_0 = \frac{eE_0}{k}
$$

The velocity width of the trapped region is:

$$
\Delta v_{\text{trap}} \sim \frac{\omega_b}{k} = \frac{1}{k}\sqrt{\frac{ekE_0}{m}}
$$

### 6.3 BGK Modes and O'Neil's Theorem

**BGK (Bernstein-Greene-Kruskal) modes** are exact nonlinear solutions of Vlasov-Poisson with trapped particles. They represent steady-state electrostatic structures (electron holes, ion holes).

**O'Neil's theorem**: Landau damping can be viewed as a phase-mixing of linear modes (eigenmodes with different phase velocities). The electric field decays while the perturbed distribution $f_1$ persists (redistributed among particles).

This is fundamentally different from collisional damping:
- **Collisional**: energy dissipated as heat (irreversible)
- **Landau**: energy transferred to particles, stored in distribution (reversible, until nonlinearity or collisions act)

### 6.4 Recurrence and Echoes

Because Landau damping is reversible, the system can exhibit **recurrence**: the electric field reappears after many plasma periods. In practice, recurrence is destroyed by:
- Collisions
- Nonlinearity (trapping)
- Finite geometry

**Plasma echoes**: If two perturbations are applied at different times, a "echo" signal appears at a later time, demonstrating the reversible nature of Landau damping.

## 7. Ion Acoustic Waves and Landau Damping

### 7.1 Ion Acoustic Waves

Ion acoustic waves are low-frequency electrostatic waves with:
- Electrons provide restoring force (via pressure)
- Ions provide inertia
- Dispersion: $\omega/k \approx c_s = \sqrt{k_B T_e/m_i}$ (ion sound speed)

The dispersion relation (including Landau damping from ions and electrons) is:

$$
\epsilon(k, \omega) = 1 + \frac{1}{k^2\lambda_{De}^2} - \frac{\omega_{pi}^2}{k^2}\int \frac{df_i/dv}{v - \omega/k} dv = 0
$$

where the electron contribution is approximated by $1/k^2\lambda_{De}^2$ (assuming $\omega/k \ll v_{th,e}$).

### 7.2 Condition for Weak Damping

The ion Landau damping rate is:

$$
\gamma_i \propto -\frac{df_i}{dv}\bigg|_{v = c_s}
$$

For weak damping, we need $c_s \gg v_{th,i}$ (phase velocity much faster than ion thermal velocity):

$$
\sqrt{\frac{k_B T_e}{m_i}} \gg \sqrt{\frac{k_B T_i}{m_i}} \quad \Rightarrow \quad T_e \gg T_i
$$

Thus, ion acoustic waves propagate with low damping when **electrons are much hotter than ions**.

For $T_e \sim T_i$, ion Landau damping is strong, and the wave is heavily damped.

### 7.3 Applications

Ion acoustic waves are important in:
- **Laser-plasma interactions** (stimulated Brillouin scattering)
- **Inertial confinement fusion** (energy transport)
- **Space plasmas** (solar wind turbulence)

## 8. Python Implementations

### 8.1 Plasma Dispersion Function Z(ζ)

The plasma dispersion function is defined as:

$$
Z(\zeta) = \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty} \frac{e^{-t^2}}{t - \zeta} dt
$$

with the Landau contour (pole below real axis). This function appears frequently in plasma kinetic theory.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

def plasma_dispersion_function(zeta):
    """
    Plasma dispersion function Z(zeta)
    Uses Faddeeva function (wofz in scipy)

    Z(zeta) = i*sqrt(pi) * w(zeta)
    where w(z) is the Faddeeva function
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)

# Plot Z(ζ) for real ζ
zeta_real = np.linspace(-5, 5, 1000)
Z_real = plasma_dispersion_function(zeta_real)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(zeta_real, Z_real.real, 'b-', linewidth=2, label='Re[Z(ζ)]')
ax1.plot(zeta_real, Z_real.imag, 'r-', linewidth=2, label='Im[Z(ζ)]')
ax1.set_xlabel('ζ', fontsize=14)
ax1.set_ylabel('Z(ζ)', fontsize=14)
ax1.set_title('Plasma Dispersion Function', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# For small ζ: Z(ζ) ≈ i*sqrt(pi)*exp(-ζ^2) - 2ζ (asymptotic)
zeta_small = np.linspace(-2, 2, 100)
Z_approx = 1j*np.sqrt(np.pi)*np.exp(-zeta_small**2) - 2*zeta_small

ax2.plot(zeta_small, np.abs(Z_real[400:600]), 'b-', linewidth=2, label='|Z(ζ)| exact')
ax2.plot(zeta_small, np.abs(Z_approx), 'r--', linewidth=2, label='|Z(ζ)| approx')
ax2.set_xlabel('ζ', fontsize=14)
ax2.set_ylabel('|Z(ζ)|', fontsize=14)
ax2.set_title('Asymptotic Approximation', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plasma_dispersion_function.png', dpi=150)
print("Saved: plasma_dispersion_function.png")
```

### 8.2 Landau Damping Rate vs kλ_D

```python
# Constants
e = 1.6e-19
m_e = 9.11e-31
epsilon_0 = 8.85e-12
k_B = 1.38e-23

# Plasma parameters
n_e = 1e18  # m^-3
T_e_eV = 10  # eV
T_e = T_e_eV * e / k_B  # K

# Derived quantities
omega_pe = np.sqrt(n_e * e**2 / (epsilon_0 * m_e))
v_th = np.sqrt(k_B * T_e / m_e)
lambda_D = np.sqrt(epsilon_0 * k_B * T_e / (n_e * e**2))

print(f"Plasma parameters:")
print(f"  n_e = {n_e:.2e} m^-3")
print(f"  T_e = {T_e_eV} eV")
print(f"  ω_pe = {omega_pe:.2e} rad/s")
print(f"  v_th = {v_th:.2e} m/s")
print(f"  λ_D = {lambda_D:.2e} m")

# Range of k*lambda_D
k_lambda_D = np.linspace(0.1, 3, 100)
k_array = k_lambda_D / lambda_D

# Dispersion relation (Bohm-Gross)
omega_r = omega_pe * np.sqrt(1 + 3 * k_lambda_D**2)

# Landau damping rate
gamma = -np.sqrt(np.pi / 8) * (omega_pe / k_lambda_D**3) * np.exp(-1 / (2 * k_lambda_D**2))

# Damping decrement
damping_decrement = -gamma / omega_r

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Dispersion relation
ax = axes[0, 0]
ax.plot(k_lambda_D, omega_r / omega_pe, 'b-', linewidth=2)
ax.axhline(y=1, color='r', linestyle='--', linewidth=1, label='ω_pe (cold plasma)')
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('ω_r / ω_pe', fontsize=12)
ax.set_title('Dispersion Relation (Bohm-Gross)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Damping rate
ax = axes[0, 1]
ax.plot(k_lambda_D, np.abs(gamma) / omega_pe, 'r-', linewidth=2)
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('|γ| / ω_pe', fontsize=12)
ax.set_title('Landau Damping Rate', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# Damping decrement
ax = axes[1, 0]
ax.plot(k_lambda_D, damping_decrement, 'g-', linewidth=2)
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('|γ| / ω_r', fontsize=12)
ax.set_title('Damping Decrement (per radian)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# Number of oscillations before e-fold decay
ax = axes[1, 1]
N_osc = omega_r / (2 * np.pi * np.abs(gamma))
ax.plot(k_lambda_D, N_osc, 'm-', linewidth=2)
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('N (oscillations)', fontsize=12)
ax.set_title('Number of Oscillations Before e-Fold Decay', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('landau_damping_rate.png', dpi=150)
print("Saved: landau_damping_rate.png")

# Print specific values
print(f"\nLandau damping at kλ_D = 0.3:")
idx = np.argmin(np.abs(k_lambda_D - 0.3))
print(f"  ω_r/ω_pe = {omega_r[idx]/omega_pe:.3f}")
print(f"  |γ|/ω_pe = {np.abs(gamma[idx])/omega_pe:.3e}")
print(f"  |γ|/ω_r = {damping_decrement[idx]:.3e}")
print(f"  N_osc = {N_osc[idx]:.1f}")
```

### 8.3 Vlasov-Poisson Simulation with Landau Damping

```python
class VlasovPoisson1D:
    """
    1D Vlasov-Poisson solver for Landau damping
    """
    def __init__(self, Nx, Nv, Lx, v_max, n0, T_eV, m, q):
        self.Nx = Nx
        self.Nv = Nv
        self.Lx = Lx
        self.v_max = v_max

        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.v = np.linspace(-v_max, v_max, Nv)
        self.dx = Lx / Nx
        self.dv = 2 * v_max / Nv

        self.n0 = n0
        self.T = T_eV * e / k_B
        self.m = m
        self.q = q

        # Initialize distribution function
        self.f = self._initialize_maxwellian()

    def _initialize_maxwellian(self):
        """Maxwellian distribution"""
        v_th = np.sqrt(k_B * self.T / self.m)
        f = np.zeros((self.Nx, self.Nv))
        for i in range(self.Nx):
            f[i, :] = self.n0 * (self.m / (2 * np.pi * k_B * self.T))**0.5 * \
                     np.exp(-self.m * self.v**2 / (2 * k_B * self.T))
        return f

    def add_perturbation(self, k_mode, amplitude):
        """Add sinusoidal density perturbation"""
        for i in range(self.Nx):
            pert = 1 + amplitude * np.cos(k_mode * self.x[i])
            self.f[i, :] *= pert

    def compute_density(self):
        """Compute density from distribution function"""
        return np.trapz(self.f, self.v, axis=1)

    def compute_electric_field(self):
        """Solve Poisson equation for E-field (periodic BC)"""
        n = self.compute_density()
        rho = self.q * (n - self.n0)  # charge density (background neutrality)

        # Fourier transform
        rho_k = np.fft.fft(rho)
        k_modes = 2 * np.pi * np.fft.fftfreq(self.Nx, self.dx)

        # Poisson: -ε₀ d²φ/dx² = ρ → φ_k = -rho_k / (ε₀ k²)
        phi_k = np.zeros_like(rho_k, dtype=complex)
        phi_k[1:] = -rho_k[1:] / (epsilon_0 * k_modes[1:]**2)
        phi_k[0] = 0  # Set DC component to zero (neutrality)

        # E = -dφ/dx → E_k = i*k*φ_k
        E_k = 1j * k_modes * phi_k

        # Inverse FFT
        E = np.fft.ifft(E_k).real

        return E

    def step(self, dt):
        """Operator splitting: advection in x, then in v"""
        # Step 1: Advection in x (∂f/∂t + v ∂f/∂x = 0)
        f_new = np.zeros_like(self.f)
        for j in range(self.Nv):
            # Upwind scheme
            if self.v[j] > 0:
                for i in range(self.Nx):
                    i_up = (i - 1) % self.Nx
                    f_new[i, j] = self.f[i, j] - self.v[j] * dt / self.dx * \
                                 (self.f[i, j] - self.f[i_up, j])
            else:
                for i in range(self.Nx):
                    i_up = (i + 1) % self.Nx
                    f_new[i, j] = self.f[i, j] - self.v[j] * dt / self.dx * \
                                 (self.f[i_up, j] - self.f[i, j])
        self.f = f_new.copy()

        # Step 2: Acceleration in v (∂f/∂t + a ∂f/∂v = 0)
        E = self.compute_electric_field()
        f_new = np.zeros_like(self.f)
        for i in range(self.Nx):
            a = self.q * E[i] / self.m  # acceleration
            for j in range(self.Nv):
                if a > 0:
                    j_up = max(j - 1, 0)
                    f_new[i, j] = self.f[i, j] - a * dt / self.dv * \
                                 (self.f[i, j] - self.f[i, j_up])
                else:
                    j_up = min(j + 1, self.Nv - 1)
                    f_new[i, j] = self.f[i, j] - a * dt / self.dv * \
                                 (self.f[i, j_up] - self.f[i, j])
        self.f = f_new.copy()

    def run(self, dt, num_steps, save_interval=10):
        """Run simulation"""
        times = []
        E_history = []

        for n in range(num_steps):
            if n % save_interval == 0:
                E = self.compute_electric_field()
                E_max = np.max(np.abs(E))
                E_history.append(E_max)
                times.append(n * dt)
                if n % (num_steps // 10) == 0:
                    print(f"Step {n}/{num_steps}, t = {n*dt:.3e} s, E_max = {E_max:.3e} V/m")

            self.step(dt)

        return np.array(times), np.array(E_history)

# Simulation parameters
Nx = 64
Nv = 128
n0 = 1e18  # m^-3
T_eV = 10  # eV
m = m_e
q = -e

# Domain
lambda_D = np.sqrt(epsilon_0 * k_B * (T_eV * e / k_B) / (n0 * e**2))
k_mode = 0.3 / lambda_D  # kλ_D = 0.3
Lx = 2 * np.pi / k_mode
v_max = 5 * np.sqrt(k_B * (T_eV * e / k_B) / m)

# Initialize solver
print("\n=== Landau Damping Simulation ===")
print(f"Nx = {Nx}, Nv = {Nv}")
print(f"Lx = {Lx:.3e} m, v_max = {v_max:.3e} m/s")
print(f"kλ_D = 0.3")

solver = VlasovPoisson1D(Nx, Nv, Lx, v_max, n0, T_eV, m, q)

# Add perturbation
amplitude = 0.01
solver.add_perturbation(k_mode, amplitude)

# Run simulation
dt = 1e-11  # s (must satisfy CFL condition)
num_steps = 2000
save_interval = 5

times, E_max_history = solver.run(dt, num_steps, save_interval)

# Theoretical damping
omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
k_lambda_D_val = 0.3
omega_r = omega_pe * np.sqrt(1 + 3 * k_lambda_D_val**2)
gamma_theory = -np.sqrt(np.pi / 8) * (omega_pe / k_lambda_D_val**3) * \
               np.exp(-1 / (2 * k_lambda_D_val**2))

E_theory = E_max_history[0] * np.exp(gamma_theory * times)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear plot
ax1.plot(times * omega_pe, E_max_history, 'b-', linewidth=2, label='Simulation')
ax1.plot(times * omega_pe, E_theory, 'r--', linewidth=2, label='Theory')
ax1.set_xlabel('ω_pe t', fontsize=12)
ax1.set_ylabel('E_max (V/m)', fontsize=12)
ax1.set_title('Landau Damping of Electric Field', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Log plot
ax2.semilogy(times * omega_pe, E_max_history, 'b-', linewidth=2, label='Simulation')
ax2.semilogy(times * omega_pe, E_theory, 'r--', linewidth=2, label='Theory')
ax2.set_xlabel('ω_pe t', fontsize=12)
ax2.set_ylabel('E_max (V/m)', fontsize=12)
ax2.set_title('Landau Damping (Log Scale)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('landau_damping_simulation.png', dpi=150)
print("\nSaved: landau_damping_simulation.png")

# Fit damping rate
log_E = np.log(E_max_history)
fit = np.polyfit(times, log_E, 1)
gamma_fit = fit[0]

print(f"\nTheoretical damping rate: γ = {gamma_theory:.3e} rad/s")
print(f"Fitted damping rate: γ = {gamma_fit:.3e} rad/s")
print(f"Relative error: {abs(gamma_fit - gamma_theory)/abs(gamma_theory)*100:.1f}%")
```

### 8.4 Particle Trapping Visualization

```python
def particle_in_wave(E0, k, m, q, v_ph, num_particles=100, duration=1e-7, dt=1e-10):
    """
    Simulate particles in a static wave (wave frame)
    """
    # Particle initial conditions
    np.random.seed(42)
    x0 = np.random.uniform(0, 2*np.pi/k, num_particles)
    v0 = np.random.normal(v_ph, 1e4, num_particles)  # spread around v_ph

    # Storage
    num_steps = int(duration / dt)
    x_traj = np.zeros((num_particles, num_steps))
    v_traj = np.zeros((num_particles, num_steps))
    x_traj[:, 0] = x0
    v_traj[:, 0] = v0

    # Integrate equations of motion
    for n in range(1, num_steps):
        x = x_traj[:, n-1]
        v = v_traj[:, n-1]

        # Electric field (wave frame: static)
        E = E0 * np.sin(k * x)
        a = q * E / m

        # Velocity Verlet
        v_half = v + 0.5 * a * dt
        x_new = x + v_half * dt
        x_new = x_new % (2 * np.pi / k)  # periodic

        E_new = E0 * np.sin(k * x_new)
        a_new = q * E_new / m
        v_new = v_half + 0.5 * a_new * dt

        x_traj[:, n] = x_new
        v_traj[:, n] = v_new

    return x_traj, v_traj

# Parameters
E0 = 1e3  # V/m (large amplitude)
k = 1e5   # m^-1
v_ph = 1e5  # m/s
omega_b = np.sqrt(e * k * E0 / m_e)

print(f"\n=== Particle Trapping ===")
print(f"E0 = {E0} V/m, k = {k} m^-1")
print(f"v_ph = {v_ph:.2e} m/s")
print(f"Bounce frequency ω_b = {omega_b:.2e} rad/s")
print(f"Bounce period τ_b = {2*np.pi/omega_b:.2e} s")

x_traj, v_traj = particle_in_wave(E0, k, m_e, -e, v_ph, num_particles=50,
                                   duration=2e-7, dt=1e-10)

# Plot phase space
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Initial phase space
ax1.scatter(x_traj[:, 0] * k / (2*np.pi), (v_traj[:, 0] - v_ph) / 1e3,
           c='blue', s=10, alpha=0.6)
ax1.set_xlabel('kx / 2π', fontsize=12)
ax1.set_ylabel('v - v_ph (km/s)', fontsize=12)
ax1.set_title('Initial Phase Space', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Final phase space (with separatrix)
ax2.scatter(x_traj[:, -1] * k / (2*np.pi), (v_traj[:, -1] - v_ph) / 1e3,
           c='red', s=10, alpha=0.6, label='Particles')

# Separatrix
phi_0 = E0 / k
v_sep = np.sqrt(2 * e * phi_0 / m_e)
x_sep = np.linspace(0, 2*np.pi, 100)
v_upper = np.sqrt(2 * e * phi_0 / m_e * (1 + np.cos(x_sep)))
v_lower = -v_upper
ax2.plot(x_sep / (2*np.pi), v_upper / 1e3, 'k-', linewidth=2, label='Separatrix')
ax2.plot(x_sep / (2*np.pi), v_lower / 1e3, 'k-', linewidth=2)

ax2.set_xlabel('kx / 2π', fontsize=12)
ax2.set_ylabel('v - v_ph (km/s)', fontsize=12)
ax2.set_title('Final Phase Space (Trapped Particles)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('particle_trapping.png', dpi=150)
print("Saved: particle_trapping.png")
```

## Summary

Landau damping is one of the most profound results in plasma physics:

1. **Dispersion relation**: Linearized Vlasov-Poisson gives $\epsilon(k,\omega) = 0$ with a pole at $v = \omega/k$.

2. **Landau contour**: Causality requires the integration path to go below the pole, yielding:
   $$
   \epsilon = 1 - \sum_s \frac{\omega_{ps}^2}{k^2}\left[\mathcal{P}\int + i\pi\frac{df_0}{dv}\bigg|_{v=\omega/k}\right]
   $$

3. **Damping rate**: For a Maxwellian,
   $$
   \gamma \approx -\sqrt{\frac{\pi}{8}}\frac{\omega_{pe}}{(k\lambda_D)^3}\exp\left(-\frac{1}{2k^2\lambda_D^2}\right)
   $$
   Exponentially weak for $k\lambda_D \ll 1$.

4. **Physical mechanism**: Resonant particles ($v \approx v_{\text{ph}}$) exchange energy with the wave. For Maxwellian (more slow than fast), net energy flows wave → particles → damping.

5. **Inverse Landau damping**: $df_0/dv > 0$ at resonance → growth (bump-on-tail instability).

6. **Nonlinear effects**: Large amplitude → particle trapping → phase space vortex → quasilinear relaxation.

7. **Ion acoustic waves**: Low damping requires $T_e \gg T_i$.

Landau damping is:
- **Collisionless** (no entropy increase)
- **Reversible** (phase mixing, echoes)
- **Kinetic** (fluid models cannot capture it)

Understanding Landau damping is essential for:
- Plasma heating (e.g., wave absorption)
- Stability analysis
- Turbulence damping
- Astrophysical plasmas

## Practice Problems

### Problem 1: Bohm-Gross Dispersion

Derive the Bohm-Gross dispersion relation $\omega^2 = \omega_{pe}^2 + 3k^2v_{th}^2$ from the linearized Vlasov-Poisson system for a Maxwellian distribution, assuming $k\lambda_D \ll 1$ and neglecting damping.

**Hint**: Use the principal value integral and expand for small $k\lambda_D$.

---

### Problem 2: Landau Damping at Different kλ_D

For an electron plasma with $n_e = 10^{19}$ m$^{-3}$ and $T_e = 100$ eV:

(a) Calculate $\omega_{pe}$ and $\lambda_D$.

(b) For $k\lambda_D = 0.2$, 0.5, and 1.0, compute the Landau damping rate $\gamma$ and the damping decrement $|\gamma|/\omega_r$.

(c) How many oscillations does the wave undergo before the amplitude decays by a factor of $e$?

(d) At what $k\lambda_D$ is the damping strongest (as a fraction of $\omega_r$)?

---

### Problem 3: Bump-on-Tail Instability

Consider a distribution function:

$$
f_0(v) = n_c\sqrt{\frac{m}{2\pi k_BT_c}}\exp\left(-\frac{mv^2}{2k_BT_c}\right) + n_b\sqrt{\frac{m}{2\pi k_BT_b}}\exp\left(-\frac{m(v-v_b)^2}{2k_BT_b}\right)
$$

with $n_b = 0.1 n_c$, $T_b = T_c$, and $v_b = 3v_{th,c}$ where $v_{th,c} = \sqrt{k_BT_c/m}$.

(a) Plot $f_0(v)$ and identify the region where $df_0/dv > 0$.

(b) Estimate the phase velocity $v_{\text{ph}}$ where the growth rate is maximum.

(c) Using the Landau formula, estimate the growth rate at $v_{\text{ph}}$.

(d) Discuss how quasilinear relaxation would flatten the distribution over time.

---

### Problem 4: Ion Acoustic Wave Damping

In a hydrogen plasma with $n = 10^{18}$ m$^{-3}$, $T_e = 1$ keV, and $T_i = 100$ eV:

(a) Calculate the ion sound speed $c_s = \sqrt{k_BT_e/m_i}$.

(b) Estimate the ion thermal velocity $v_{th,i} = \sqrt{k_BT_i/m_i}$.

(c) Compare $c_s$ and $v_{th,i}$. Is the ion Landau damping weak or strong?

(d) What would be the condition on $T_i/T_e$ for the ion acoustic wave to propagate with weak damping ($|\gamma|/\omega \ll 1$)?

---

### Problem 5: Particle Trapping and Bounce Frequency

A wave with amplitude $E_0 = 10^4$ V/m and wavenumber $k = 10^5$ m$^{-1}$ propagates in an electron plasma.

(a) Calculate the bounce frequency $\omega_b = \sqrt{ekE_0/m_e}$.

(b) Estimate the width in velocity space of the trapped region: $\Delta v_{\text{trap}} \sim \omega_b/k$.

(c) For a Maxwellian with $T_e = 10$ eV, what fraction of electrons have velocities within $\Delta v_{\text{trap}}$ of the phase velocity $v_{\text{ph}} = 10^6$ m/s?

(d) If the bounce frequency is $\omega_b \sim 10^8$ rad/s and the plasma frequency is $\omega_{pe} \sim 10^{11}$ rad/s, is the trapping timescale fast or slow compared to the plasma oscillation?

---

## Navigation

- **Previous**: [Vlasov Equation](./07_Vlasov_Equation.md)
- **Next**: [Collisional Kinetics](./09_Collisional_Kinetics.md)

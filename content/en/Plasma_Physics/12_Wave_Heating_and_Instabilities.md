# 12. Wave Heating and Instabilities

## Learning Objectives

- Understand the physical mechanisms of wave heating in fusion plasmas
- Master the theory of velocity space instabilities (beam-plasma, bump-on-tail, Weibel)
- Analyze pressure-driven instabilities in magnetized plasmas (firehose, mirror)
- Learn the conditions for parametric instabilities in laser-plasma interactions
- Apply instability theory to practical problems in fusion and astrophysics
- Compute growth rates and stability boundaries for various instability mechanisms

## Introduction

Waves in plasmas serve two critical roles:
1. **Heating and current drive**: External waves transfer energy to plasma particles
2. **Instabilities**: Waves can grow spontaneously, extracting free energy from the plasma

This lesson covers both aspects, focusing on:
- **Wave heating**: How RF waves deposit energy in fusion plasmas
- **Velocity space instabilities**: Arising from non-Maxwellian distributions
- **Pressure-driven instabilities**: From temperature anisotropies
- **Parametric instabilities**: Wave-wave coupling in high-power laser systems

These phenomena are crucial for:
- Fusion reactor design (heating systems, current drive)
- Astrophysical plasmas (solar wind, pulsar magnetospheres, GRB afterglows)
- Laser-plasma interactions (inertial confinement fusion)
- Space weather (radiation belts, magnetospheric dynamics)

## 1. Wave Heating in Fusion Plasmas

### 1.1 Overview of Heating Methods

Fusion plasmas require temperatures $T \sim 10-20$ keV ($\sim 100-200$ million K). Three main heating methods:

**Ohmic heating**:
- $P = I^2 R$ where $R \propto T_e^{-3/2}$ (classical resistivity)
- Effective at low $T$, ineffective at high $T$
- Limited to $\sim 1-2$ keV in tokamaks

**Neutral Beam Injection (NBI)**:
- Fast neutrals (50-1000 keV) injected, ionized, transfer energy via collisions
- Not a wave method, but important for comparison
- Power: 10-50 MW per beamline in ITER

**Radio-Frequency (RF) heating**:
- Electromagnetic waves launched from antennas or waveguides
- Three frequency ranges: ECRH, ICRH, LHCD
- Advantages: localized deposition, current drive capability, no particle source

### 1.2 Electron Cyclotron Resonance Heating (ECRH)

**Frequency**: $\omega \approx n\omega_{ce}$ where $n = 1, 2$ typically

**Resonance condition**: At spatial location where $\omega = n\omega_{ce}(r)$, the wave phase matches electron gyration.

**Absorption mechanism**:
- Electrons in resonance with the wave ($\omega - k_\parallel v_\parallel = n\omega_{ce}$)
- Wave electric field perpendicular to $\mathbf{B}$ does work on gyrating electrons
- Power absorption: $P \propto \int d^3v \, \mathbf{j} \cdot \mathbf{E}$

**Dispersion**: Use X-mode or O-mode depending on density
- O-mode: cutoff at $n_c = \epsilon_0 m_e \omega^2/e^2$
- X-mode: higher cutoff, better for overdense plasmas

**Typical parameters** (ITER):
- Frequency: 170 GHz (for $B \sim 5.3$ T at $n=2$)
- Power: 20 MW total (gyrotrons)
- Beam width: $\sim 5$ cm (highly localized)

**Advantages**:
- Excellent localization ($\Delta r \sim$ cm)
- Current drive capability (ECCD)
- Real-time control of deposition location

**Challenges**:
- Requires high-frequency gyrotrons (expensive)
- Transmission losses in waveguides
- Mirror alignment critical

### 1.3 Ion Cyclotron Resonance Heating (ICRH)

**Frequency**: $\omega \approx n\omega_{ci}$ where $\omega_{ci} = ZeB/m_i$

**Resonance condition**: Ions gyrate at $\omega_{ci} \sim 2\pi \times (30-100)$ MHz for tokamaks.

**Heating schemes**:

**Fundamental majority heating**: $\omega = \omega_{ci}$ for main ion species
- Direct resonance, but weak single-pass absorption
- Requires multiple passes

**Second harmonic**: $\omega = 2\omega_{ci}$
- Stronger absorption than fundamental
- Used in low-field devices

**Minority heating**: $\omega = \omega_{ci,\text{minority}}$
- Introduce minority species (e.g., 5-10% $^3$He in D plasma)
- Resonance at $\omega_{ci}(^3\text{He})$ while bulk deuterium is off-resonance
- Minority ions heated to high energy, collisionally transfer to bulk

**Mode conversion**: Fast wave converts to ion Bernstein wave or ion cyclotron wave
- Occurs near hybrid resonance layer
- Can heat electrons efficiently

**Typical parameters**:
- Frequency: 40-80 MHz
- Power: 20 MW (ITER)
- Antenna: large coils at vessel wall

**Advantages**:
- Well-established technology
- Can heat both ions and electrons
- Central heating possible

**Challenges**:
- Antenna-plasma interaction (impurities, hot spots)
- Parasitic losses
- Less localized than ECRH

### 1.4 Lower Hybrid Current Drive (LHCD)

**Frequency**: $\omega_{ci} \ll \omega \ll \omega_{ce}$ (typically 1-8 GHz)

**Purpose**: Primarily for **current drive**, not heating
- Non-inductive current generation
- Steady-state operation in tokamaks

**Mechanism**:
- Lower hybrid wave propagates with high $n_\parallel = k_\parallel c/\omega$
- Strong Landau damping on tail electrons ($v_\parallel \sim \omega/k_\parallel$)
- Asymmetric damping creates net current

**Current drive efficiency**:
$$\eta_{CD} = \frac{n_{20} I_A R}{P_{\text{MW}}}$$

where $n_{20}$ is density in $10^{20}$ m$^{-3}$, $I_A$ is current in MA, $R$ is major radius in m, $P$ is power in MW.

Typical: $\eta_{CD} \sim 0.2-0.5$ for LHCD.

**Accessibility**: Lower hybrid wave must penetrate to desired location
- Density limit: $n < n_{\text{access}}$ where wave cutoff occurs
- In high-density core, may not penetrate

**Typical parameters**:
- Frequency: 3.7-5 GHz
- Power: 20 MW (ITER)
- Launcher: waveguide array (grill)

**Advantages**:
- High current drive efficiency
- Off-axis current profile control

**Challenges**:
- Density limit for accessibility
- Spectral gap (difficulty coupling power)
- Nonlinear effects at high power

### 1.5 Comparison of Heating Methods

| Method | Frequency | Main Target | Localization | Current Drive | Power (ITER) |
|--------|-----------|-------------|--------------|---------------|--------------|
| ECRH   | 140-170 GHz | Electrons | Excellent | Yes (ECCD) | 20 MW |
| ICRH   | 40-80 MHz | Ions | Moderate | Weak | 20 MW |
| LHCD   | 3-8 GHz | Electrons | Good | Excellent | 20 MW |
| NBI    | N/A | Ions | Poor | Yes | 33 MW |

**Synergy**: Combining methods is often optimal
- NBI + ICRH: NBI creates fast ion tail, ICRH heats tail further
- ECRH + LHCD: ECRH for MHD control, LHCD for current profile

## 2. Velocity Space Instabilities

### 2.1 Beam-Plasma Instability (Two-Stream)

Consider a **cold electron beam** with density $n_b$ and velocity $v_0$ streaming through a background plasma with density $n_0$.

**Setup**:
- Beam: $f_b(\mathbf{v}) = n_b \delta(v_x - v_0)\delta(v_y)\delta(v_z)$
- Background: $f_0(\mathbf{v}) = n_0 \delta(v_x)\delta(v_y)\delta(v_z)$
- Both cold ($T = 0$)

**Dispersion relation**: From linearized Vlasov + Poisson:

$$1 = \frac{\omega_{p0}^2}{\omega^2} + \frac{\omega_{pb}^2}{(\omega - kv_0)^2}$$

where $\omega_{p0}^2 = n_0 e^2/(\epsilon_0 m_e)$ and $\omega_{pb}^2 = n_b e^2/(\epsilon_0 m_e)$.

**Analysis**: Assume $\omega = \omega_r + i\gamma$ and look for unstable solutions ($\gamma > 0$).

For $n_b \ll n_0$, expand around the Langmuir wave $\omega \approx \omega_{p0} + \delta\omega$:

$$\delta\omega \approx -\frac{\omega_{pb}^2}{2\omega_{p0}} \frac{1}{1 - kv_0/\omega_{p0}}$$

When denominator is small, $\delta\omega$ becomes large. For $kv_0 \approx \omega_{p0}$, the correction becomes imaginary.

**Growth rate** (for $n_b/n_0 \ll 1$):

$$\boxed{\gamma \approx \omega_{p0} \left(\frac{n_b}{n_0}\right)^{1/3}}$$

The instability is **strongest** when:
$$kv_0 \approx \omega_{p0}$$

**Physical picture**:
```
Beam electrons:  ──→  ──→  ──→  ──→
Background:      ·    ·    ·    ·

Perturbation creates bunching:
  ──→ ──→    ──→ ──→     (density wave)

Bunches enhance electric field → feedback → growth
```

**Applications**:
- Electron beams in plasma (accelerators, space)
- Ionospheric instabilities
- Caused early issues in particle accelerators

### 2.2 Bump-on-Tail Instability

A more realistic scenario: a **small population of fast electrons** on a Maxwellian background.

**Distribution**:
$$f(v) = f_M(v) + f_{\text{bump}}(v)$$

where $f_{\text{bump}}$ is a small population at $v \sim v_{\text{bump}} > v_{th}$.

**Criterion for instability**: The distribution must have a **positive slope** in velocity space at the resonant velocity:

$$\frac{\partial f}{\partial v}\bigg|_{v = \omega/k} > 0$$

This is **inverse Landau damping**: particles at $v = v_\phi$ transfer energy TO the wave instead of FROM the wave.

**Growth rate**: For small bump with density $n_b$ and width $\Delta v$:

$$\gamma \sim \omega_{pe} \left(\frac{n_b}{n_0}\right)^{1/3} \frac{v_{\text{bump}}}{v_{th}}$$

**Quasilinear relaxation**: As the wave grows, it diffuses particles in velocity space via:

$$\frac{\partial f}{\partial t} = \frac{\partial}{\partial v}\left(D \frac{\partial f}{\partial v}\right)$$

where $D \propto |E_k|^2$ is the diffusion coefficient.

Result: The **bump flattens** into a plateau:
```
Initial:        After saturation:
f(v)            f(v)
  |\              |----\
  | \             |     \
  |  \___         |      \___
  |      \        |          \
  +-------v       +----------v
      v_bump           plateau
```

This **quasilinear plateau formation** is a fundamental nonlinear saturation mechanism.

**Applications**:
- Runaway electrons in tokamaks
- Solar wind electron beams
- Laser-plasma interactions

### 2.3 Weibel Instability

The **Weibel instability** grows from **temperature anisotropy**: $T_\perp > T_\parallel$ (or vice versa).

**Physical mechanism**:
- Anisotropic distribution creates current fluctuations
- Currents generate magnetic fields
- Magnetic fields enhance anisotropy → positive feedback

**Setup**: Consider a distribution:
$$f(v_\parallel, v_\perp) = n_0 \left(\frac{m}{2\pi k_B T_\parallel}\right)^{1/2}\left(\frac{m}{2\pi k_B T_\perp}\right) \exp\left(-\frac{mv_\parallel^2}{2k_BT_\parallel} - \frac{mv_\perp^2}{2k_BT_\perp}\right)$$

**Dispersion** (for $T_\perp > T_\parallel$): Purely growing mode (no real frequency):

$$\omega = i\gamma$$

**Growth rate**:

$$\boxed{\gamma_{\text{max}} \approx \omega_{pe} \sqrt{\frac{T_\perp - T_\parallel}{T_\parallel}}}$$

for wavenumber:
$$k_{\text{max}} \approx \frac{\omega_{pe}}{c}\sqrt{\frac{T_\perp}{T_\parallel} - 1}$$

**Generated magnetic field**: The instability creates **small-scale magnetic fields** with strength:

$$\frac{B^2}{8\pi} \sim n k_B (T_\perp - T_\parallel)$$

**Applications**:
- **Collisionless shocks**: In astrophysical shocks (e.g., supernova remnants, GRB afterglows), Weibel instability generates magnetic fields that mediate the shock
- **Laser-plasma interaction**: Intense laser creates anisotropic electron distribution → Weibel instability → magnetic field generation
- **Magnetospheric plasmas**: Anisotropic distributions in radiation belts
- **Magnetic field generation**: Weibel is a mechanism for seed fields in cosmology

This instability was theoretically predicted by Weibel (1959) and experimentally confirmed in laser-plasma experiments (2000s).

## 3. Pressure-Driven Instabilities

### 3.1 Firehose Instability

In a magnetized plasma with $p_\parallel > p_\perp$ (parallel pressure exceeds perpendicular), the **firehose instability** can occur.

**Analogy**: Like a pressurized garden hose writhing when pressure is too high.

**Physical mechanism**:
- Magnetic field line bends
- Parallel pressure pushes plasma along bent field
- Curvature increases → positive feedback

**Stability criterion**: From MHD with anisotropic pressure:

$$\boxed{p_\parallel - p_\perp < \frac{B^2}{\mu_0}}$$

or equivalently:

$$\beta_\parallel - \beta_\perp < 1$$

where $\beta_\parallel = 2\mu_0 p_\parallel/B^2$ and $\beta_\perp = 2\mu_0 p_\perp/B^2$.

**Growth rate** (for unstable case):

$$\gamma^2 \approx k^2 v_A^2 \left(\frac{p_\parallel - p_\perp}{p_\parallel + p_\perp/2} - \frac{1}{\beta_\parallel}\right)$$

where $v_A = B/\sqrt{\mu_0 \rho}$ is the Alfvén speed.

**Maximum growth** at:
$$k \sim \frac{1}{L}$$

where $L$ is the system size (low-$k$ instability).

**Observations**:
- **Solar wind**: Often marginally stable/unstable to firehose
- **Magnetosheath**: Compressed plasma can violate stability condition
- **Tokamak edge**: Fast ion populations can drive firehose

**Saturation**: Pitch-angle scattering relaxes anisotropy, quenching the instability.

### 3.2 Mirror Instability

The opposite anisotropy, $p_\perp > p_\parallel$, can drive the **mirror instability**.

**Physical mechanism**:
- Magnetic field strength fluctuates: $B = B_0 + B_1$
- Particles with high $\mu = mv_\perp^2/(2B)$ are trapped in low-$B$ regions (magnetic mirrors)
- Enhanced $p_\perp$ in low-$B$ regions → $B$ decreases further → feedback

**Stability criterion**:

$$\boxed{\frac{p_\perp}{p_\parallel} < 1 + \frac{1}{\beta_\perp}}$$

or:

$$\beta_\perp\left(\frac{p_\perp}{p_\parallel} - 1\right) < 1$$

**Growth rate**: For $p_\perp/p_\parallel - 1 = A$ (anisotropy):

$$\gamma \approx k_\parallel v_A \sqrt{A \beta_\perp}$$

for $k_\parallel L \sim 1$ where $L$ is scale length.

**Characteristics**:
- **Non-propagating**: $\omega_r = 0$ (purely growing)
- **Compressional**: creates $\delta B_\parallel$ and $\delta n$
- **Anisotropic structure**: elongated along $\mathbf{B}$

**Observations**:
- **Solar wind**: Mirror mode structures (slow-mode structures with anticorrelated $B$ and $n$)
- **Magnetosheath**: Very common, quasi-steady structures
- **Planetary magnetospheres**: Jupiter, Saturn

**Saturation**: Creates quasi-static magnetic bottles that trap particles, reducing anisotropy.

### 3.3 Comparison: Firehose vs Mirror

| Property | Firehose | Mirror |
|----------|----------|--------|
| Anisotropy | $p_\parallel > p_\perp$ | $p_\perp > p_\parallel$ |
| Criterion | $\beta_\parallel - \beta_\perp < 1$ | $\beta_\perp(p_\perp/p_\parallel - 1) < 1$ |
| $\omega_r$ | Finite (propagating) | Zero (non-propagating) |
| $\delta B$ | Transverse | Compressional |
| Saturation | Pitch-angle scattering | Magnetic trapping |

Both instabilities are ubiquitous in **collisionless plasmas** where pressure can remain anisotropic (collision time $\gg$ dynamical time).

## 4. Parametric Instabilities

### 4.1 Three-Wave Coupling

**Parametric instabilities** involve coupling of three waves:
$$\omega_0 = \omega_1 + \omega_2$$
$$\mathbf{k}_0 = \mathbf{k}_1 + \mathbf{k}_2$$

where wave 0 (pump) decays into waves 1 and 2 (daughter waves).

**Mechanism**:
- Pump wave creates density/velocity oscillations
- Oscillations modulate the plasma response
- Modulated plasma can amplify daughter waves if matching conditions met

**Growth rate**: Scales with pump amplitude:
$$\gamma \propto \sqrt{\frac{I}{I_c}}$$

where $I$ is pump intensity and $I_c$ is a threshold.

### 4.2 Stimulated Raman Scattering (SRS)

**Process**: Electromagnetic wave (pump) $\to$ EM wave (scattered) + Langmuir wave

**Matching conditions**:
$$\omega_0 = \omega_s + \omega_L$$
$$\mathbf{k}_0 = \mathbf{k}_s + \mathbf{k}_L$$

where $\omega_L \approx \omega_{pe}$ (Langmuir wave) and $\omega_s < \omega_0$ (scattered EM wave).

**Dispersion constraints**:
- Pump: $\omega_0^2 = \omega_{pe}^2 + k_0^2 c^2$
- Scattered: $\omega_s^2 = \omega_{pe}^2 + k_s^2 c^2$
- Langmuir: $\omega_L^2 \approx \omega_{pe}^2 + 3k_L^2 v_{th}^2$

**Growth rate**:

$$\gamma_{SRS} = \frac{k_L v_{osc}}{4} \left(\frac{\omega_0}{\omega_s}\right)^{1/2}$$

where $v_{osc} = eE_0/(m_e\omega_0)$ is the quiver velocity in the pump wave.

**Threshold**: Requires $\gamma > \nu_L$ where $\nu_L$ is Landau damping rate.

**Relevance**: Laser fusion (ICF)
- High-power lasers ($I \sim 10^{15}$ W/cm$^2$) can drive SRS
- Scattered light lost → reduced coupling efficiency
- Hot electrons from Langmuir wave heating → preheat target (bad for compression)

**Mitigation**:
- Bandwidth: broadband laser reduces coherence
- Beam smoothing: reduces local intensity spikes
- Wavelength: shorter wavelength (UV) has higher threshold

### 4.3 Stimulated Brillouin Scattering (SBS)

**Process**: EM wave $\to$ EM wave + ion acoustic wave

**Matching**:
$$\omega_0 = \omega_s + \omega_{ia}$$
$$\mathbf{k}_0 = \mathbf{k}_s + \mathbf{k}_{ia}$$

where $\omega_{ia} = k_{ia} c_s$ (ion acoustic wave).

**Growth rate**:

$$\gamma_{SBS} = \frac{k_{ia} v_{osc}}{4\sqrt{2}} \sqrt{\frac{\omega_0}{\omega_{ia}}}$$

**Characteristics**:
- Lower threshold than SRS (ion acoustic damping weaker than Landau damping)
- **Backscattering**: strongest for $\mathbf{k}_s \approx -\mathbf{k}_0$
- Can reflect significant fraction of laser energy

**Relevance**: SBS is often the **dominant** parametric instability in laser fusion.

**Mitigation**: Similar to SRS, plus:
- Gas-filled hohlraum reduces SBS
- Multi-ion species (increases ion acoustic damping)

### 4.4 Impact on Inertial Confinement Fusion (ICF)

In **National Ignition Facility (NIF)** and other ICF experiments:
- Laser power: $\sim 500$ TW
- Intensity: $10^{14}-10^{15}$ W/cm$^2$ in hohlraum
- SRS and SBS can reflect 10-50% of laser energy

**Consequences**:
- Reduced coupling to target → lower compression
- Hot electrons from SRS preheat fuel → reduced gain
- Asymmetry in drive

**Recent progress** (2022-2023): NIF achieved ignition ($Q > 1$) by:
- Improved hohlraum design
- Better beam smoothing
- Higher laser energy (2.05 MJ)
- Mitigation of SRS/SBS through wavelength detuning

## 5. Instability Classification

### 5.1 Free Energy Sources

Instabilities extract free energy from:

1. **Velocity space**: Non-Maxwellian distributions
   - Beam-plasma: relative drift
   - Bump-on-tail: positive $\partial f/\partial v$
   - Weibel: temperature anisotropy

2. **Configuration space**: Gradients in density, temperature, magnetic field
   - Drift waves (not covered here)
   - Interchange modes
   - Tearing modes

3. **Current**: Parallel or perpendicular currents
   - Current-driven instabilities
   - Kink modes

4. **External drive**: Pumped by external waves
   - Parametric instabilities (SRS, SBS)

### 5.2 Instability Summary Table

| Instability | Free Energy | Condition | Growth Rate | Application |
|-------------|-------------|-----------|-------------|-------------|
| Beam-plasma | Beam drift $v_0$ | $kv_0 \sim \omega_{pe}$ | $\omega_{pe}(n_b/n_0)^{1/3}$ | Accelerators, space |
| Bump-on-tail | $\partial f/\partial v > 0$ | Resonant particles | $\omega_{pe}(n_b/n_0)^{1/3}$ | Runaways, solar wind |
| Weibel | $T_\perp > T_\parallel$ | Anisotropy | $\omega_{pe}\sqrt{(T_\perp-T_\parallel)/T_\parallel}$ | Shocks, lasers |
| Firehose | $p_\parallel > p_\perp$ | $\beta_\parallel - \beta_\perp > 1$ | $k v_A \sqrt{\Delta p/p}$ | Solar wind |
| Mirror | $p_\perp > p_\parallel$ | $\beta_\perp(p_\perp/p_\parallel-1) > 1$ | $k_\parallel v_A \sqrt{A\beta_\perp}$ | Magnetosheath |
| SRS | Pump laser | $I > I_c$ | $(k_L v_{osc}/4)\sqrt{\omega_0/\omega_s}$ | Laser fusion |
| SBS | Pump laser | $I > I_c$ | $(k_{ia}v_{osc}/4\sqrt{2})\sqrt{\omega_0/\omega_{ia}}$ | Laser fusion |

## 6. Python Implementation

### 6.1 Two-Stream Instability Dispersion

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def two_stream_dispersion(k, omega_p0, omega_pb, v0):
    """
    Solve two-stream dispersion: 1 = ω_p0²/ω² + ω_pb²/(ω-kv0)²

    Returns complex frequency ω(k).
    """
    def dispersion_eq(omega_complex):
        omega = omega_complex[0] + 1j * omega_complex[1]
        eps = 1 - omega_p0**2/omega**2 - omega_pb**2/(omega - k*v0)**2
        return [np.real(eps), np.imag(eps)]

    # Initial guess
    omega_guess = [omega_p0, 0.1 * omega_p0]

    sol = fsolve(dispersion_eq, omega_guess)
    return sol[0] + 1j * sol[1]

# Parameters
n0 = 1e19  # m^-3
nb_frac = 0.01  # nb/n0 = 1%
m_e = 9.109e-31  # kg
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m

omega_p0 = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
omega_pb = np.sqrt(nb_frac * n0 * e**2 / (epsilon_0 * m_e))

# Beam velocity
v0 = 2 * omega_p0 * (1e8 / omega_p0)  # Choose v0 ~ ω_p0/k_typical

print(f"Background plasma frequency: ω_p0 = {omega_p0:.2e} rad/s")
print(f"Beam plasma frequency: ω_pb = {omega_pb:.2e} rad/s")
print(f"Beam velocity: v0 = {v0:.2e} m/s")

# Wavenumber scan
k_array = np.linspace(0.5, 3, 100) * omega_p0 / v0

omega_real = []
omega_imag = []

for k in k_array:
    omega = two_stream_dispersion(k, omega_p0, omega_pb, v0)
    omega_real.append(np.real(omega))
    omega_imag.append(np.imag(omega))

omega_real = np.array(omega_real)
omega_imag = np.array(omega_imag)

# Analytical approximation for small nb/n0
gamma_approx = omega_p0 * (nb_frac)**(1/3) * np.ones_like(k_array)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Real frequency
ax1.plot(k_array * v0 / omega_p0, omega_real / omega_p0, 'b-',
         linewidth=2, label='Numerical')
ax1.axhline(1, color='r', linestyle='--', label='$\\omega_{p0}$')
ax1.plot(k_array * v0 / omega_p0, k_array * v0 / omega_p0, 'g--',
         label='$kv_0$')
ax1.set_xlabel('$kv_0 / \\omega_{p0}$', fontsize=13)
ax1.set_ylabel('$\\omega_r / \\omega_{p0}$', fontsize=13)
ax1.set_title('Two-Stream: Real Frequency', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Growth rate
ax2.plot(k_array * v0 / omega_p0, omega_imag / omega_p0, 'b-',
         linewidth=2, label='Numerical')
ax2.plot(k_array * v0 / omega_p0, gamma_approx / omega_p0, 'r--',
         linewidth=1.5, label=f'Approx: $(n_b/n_0)^{{1/3}} = {nb_frac**(1/3):.3f}$')
ax2.set_xlabel('$kv_0 / \\omega_{p0}$', fontsize=13)
ax2.set_ylabel('$\\gamma / \\omega_{p0}$', fontsize=13)
ax2.set_title(f'Two-Stream: Growth Rate ($n_b/n_0 = {nb_frac}$)', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 0.5])

plt.tight_layout()
plt.savefig('two_stream_instability.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 Weibel Instability Growth Rate

```python
def weibel_growth_rate(T_perp, T_parallel, n, B0=0):
    """
    Weibel instability growth rate.

    γ_max ≈ ω_pe √[(T_⊥ - T_∥)/T_∥]
    """
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    anisotropy = (T_perp - T_parallel) / T_parallel

    if anisotropy > 0:
        gamma_max = omega_pe * np.sqrt(anisotropy)
    else:
        gamma_max = 0

    return gamma_max, omega_pe

# Parameters
n = 1e20  # m^-3
T_parallel = 1e3  # eV
T_perp_array = np.linspace(1e3, 10e3, 100)  # eV

gamma_array = []
for T_perp in T_perp_array:
    gamma, omega_pe = weibel_growth_rate(T_perp, T_parallel, n)
    gamma_array.append(gamma)

gamma_array = np.array(gamma_array)
anisotropy_array = (T_perp_array - T_parallel) / T_parallel

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Growth rate vs anisotropy
ax1.plot(anisotropy_array, gamma_array / omega_pe, 'b-', linewidth=2)
ax1.set_xlabel('Anisotropy $(T_\\perp - T_\\parallel)/T_\\parallel$', fontsize=13)
ax1.set_ylabel('$\\gamma / \\omega_{pe}$', fontsize=13)
ax1.set_title('Weibel Instability Growth Rate', fontsize=14)
ax1.grid(True, alpha=0.3)

# Growth rate vs T_perp
ax2.plot(T_perp_array / 1e3, gamma_array / omega_pe, 'r-', linewidth=2)
ax2.axvline(T_parallel / 1e3, color='k', linestyle='--',
            label=f'$T_\\parallel = {T_parallel/1e3:.0f}$ keV')
ax2.set_xlabel('$T_\\perp$ (keV)', fontsize=13)
ax2.set_ylabel('$\\gamma / \\omega_{pe}$', fontsize=13)
ax2.set_title(f'Growth Rate vs Perpendicular Temperature', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('weibel_instability.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nWeibel instability at T_⊥ = {T_perp_array[-1]/1e3:.0f} keV, T_∥ = {T_parallel/1e3:.0f} keV:")
print(f"  Anisotropy: {anisotropy_array[-1]:.1f}")
print(f"  γ/ω_pe: {gamma_array[-1]/omega_pe:.2f}")
```

### 6.3 Firehose and Mirror Stability Boundaries

```python
def firehose_criterion(beta_parallel, beta_perp):
    """
    Firehose stability: β_∥ - β_⊥ < 1
    Returns True if stable.
    """
    return (beta_parallel - beta_perp) < 1

def mirror_criterion(beta_perp, p_perp, p_parallel):
    """
    Mirror stability: β_⊥(p_⊥/p_∥ - 1) < 1
    Returns True if stable.
    """
    return beta_perp * (p_perp / p_parallel - 1) < 1

# Generate stability diagram
beta_perp_range = np.linspace(0, 5, 200)
beta_parallel_range = np.linspace(0, 5, 200)

Beta_perp, Beta_parallel = np.meshgrid(beta_perp_range, beta_parallel_range)

# Firehose boundary: β_∥ - β_⊥ = 1
firehose_stable = Beta_parallel - Beta_perp < 1

# Mirror boundary: β_⊥(p_⊥/p_∥ - 1) = 1
# → p_⊥/p_∥ = 1 + 1/β_⊥
# Assume isotropic for simplicity in demo (real case needs p_ratio)
# For demo, use β_⊥(β_∥/β_⊥ - 1) < 1 → β_∥ < β_⊥ + 1
mirror_stable = Beta_parallel < Beta_perp + 1

# Combined stability region
stable = firehose_stable & mirror_stable

fig, ax = plt.subplots(figsize=(10, 8))

# Plot stability regions
ax.contourf(Beta_perp, Beta_parallel, stable.astype(int),
            levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.3)

# Boundaries
beta_line = np.linspace(0, 5, 100)
ax.plot(beta_line, beta_line + 1, 'b-', linewidth=2,
        label='Firehose boundary: $\\beta_\\parallel - \\beta_\\perp = 1$')
ax.plot(beta_line, beta_line - 1, 'r-', linewidth=2,
        label='Mirror boundary (approx)')

# Diagonal
ax.plot(beta_line, beta_line, 'k--', alpha=0.5, label='$\\beta_\\parallel = \\beta_\\perp$')

ax.set_xlabel('$\\beta_\\perp$', fontsize=14)
ax.set_ylabel('$\\beta_\\parallel$', fontsize=14)
ax.set_title('Pressure Anisotropy Stability Diagram', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])

# Annotate regions
ax.text(1, 3.5, 'Firehose\nUnstable', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax.text(3.5, 1, 'Mirror\nUnstable', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
ax.text(2, 2, 'Stable', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

plt.tight_layout()
plt.savefig('anisotropy_stability.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.4 Parametric Instability Thresholds

```python
def srs_growth_rate(I_laser, n, T_e, lambda_laser=1.053e-6):
    """
    Stimulated Raman Scattering growth rate.

    Parameters:
    -----------
    I_laser : float
        Laser intensity (W/m²)
    n : float
        Density (m^-3)
    T_e : float
        Electron temperature (eV)
    lambda_laser : float
        Laser wavelength (m)

    Returns:
    --------
    gamma_SRS : float
        Growth rate (s^-1)
    """
    c = 3e8
    omega_0 = 2 * np.pi * c / lambda_laser
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))

    # Quiver velocity
    E_0 = np.sqrt(2 * I_laser / (c * epsilon_0))
    v_osc = e * E_0 / (m_e * omega_0)

    # Scattered wave frequency (backward scattering)
    omega_s = omega_0 - omega_pe  # Approximate

    # Langmuir wavenumber
    k_L = 2 * omega_0 / c  # Backscatter

    # Growth rate
    gamma_SRS = (k_L * v_osc / 4) * np.sqrt(omega_0 / omega_s)

    return gamma_SRS

# Laser parameters
lambda_laser = 351e-9  # m (UV, 3ω Nd:glass)
I_array = np.logspace(13, 16, 100)  # W/m²
n = 0.1 * 1.1e21  # m^-3 (nc/10 where nc is critical density)
T_e = 3e3  # eV

gamma_array = []
for I in I_array:
    gamma = srs_growth_rate(I, n, T_e, lambda_laser)
    gamma_array.append(gamma)

gamma_array = np.array(gamma_array)

# Landau damping (approximate)
v_th = np.sqrt(2 * T_e * e / m_e)
omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
k_L = 4 * np.pi / lambda_laser
zeta = omega_pe / (k_L * v_th)
gamma_Landau = omega_pe * np.sqrt(np.pi/8) * np.exp(-zeta**2/2) / (k_L**3 * (v_th/omega_pe)**3)

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(I_array / 1e15, gamma_array / omega_pe, 'b-',
          linewidth=2, label='SRS growth rate')
ax.axhline(gamma_Landau / omega_pe, color='r', linestyle='--',
           linewidth=2, label=f'Landau damping: $\\gamma_L/\\omega_{{pe}} = {gamma_Landau/omega_pe:.2e}$')

# Threshold
I_threshold_idx = np.argmin(np.abs(gamma_array - gamma_Landau))
I_threshold = I_array[I_threshold_idx]
ax.axvline(I_threshold / 1e15, color='g', linestyle=':',
           linewidth=2, label=f'Threshold: $I_{{th}} = {I_threshold/1e15:.2f}$ PW/cm²')

ax.set_xlabel('Laser Intensity (PW/cm²)', fontsize=13)
ax.set_ylabel('$\\gamma / \\omega_{pe}$', fontsize=13)
ax.set_title('Stimulated Raman Scattering Growth Rate', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('srs_threshold.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSRS parameters:")
print(f"  Density: n = {n:.2e} m^-3 (n/n_c = {n/1.1e21:.2f})")
print(f"  Temperature: T_e = {T_e/1e3:.0f} keV")
print(f"  Threshold intensity: I_th = {I_threshold:.2e} W/m² = {I_threshold/1e15:.2f} PW/cm²")
```

## Summary

Wave heating and instabilities are central to plasma physics:

**Wave heating in fusion**:
- **ECRH**: $\omega \approx n\omega_{ce}$, 140-170 GHz, excellent localization, current drive
- **ICRH**: $\omega \approx n\omega_{ci}$, 40-80 MHz, ion heating, minority schemes
- **LHCD**: $\omega_{ci} \ll \omega \ll \omega_{ce}$, 3-8 GHz, efficient current drive
- Synergistic use of multiple methods optimal for fusion reactors

**Velocity space instabilities**:
- **Beam-plasma**: Cold beam on cold background, $\gamma \sim \omega_{pe}(n_b/n_0)^{1/3}$
- **Bump-on-tail**: Positive $\partial f/\partial v$ drives inverse Landau damping, quasilinear plateau
- **Weibel**: Temperature anisotropy generates magnetic fields, $\gamma \sim \omega_{pe}\sqrt{\Delta T/T}$

**Pressure-driven instabilities**:
- **Firehose**: $p_\parallel > p_\perp$, criterion $\beta_\parallel - \beta_\perp < 1$, bends field lines
- **Mirror**: $p_\perp > p_\parallel$, criterion $\beta_\perp(p_\perp/p_\parallel - 1) < 1$, creates magnetic bottles
- Both ubiquitous in collisionless plasmas (solar wind, magnetosphere)

**Parametric instabilities**:
- **SRS**: EM $\to$ EM + Langmuir, hot electron generation, laser fusion issue
- **SBS**: EM $\to$ EM + ion acoustic, backscattering, energy loss
- Thresholds depend on pump intensity, damping rates
- Major challenge in ICF, mitigation through bandwidth, smoothing

Applications span fusion energy, astrophysics, space physics, and high-energy-density physics. Understanding instabilities is essential for controlling and optimizing plasma performance.

## Practice Problems

### Problem 1: ECRH System Design
A tokamak has $B_0 = 2.5$ T on axis and a density profile $n(r) = n_0(1 - r^2/a^2)^2$ with $n_0 = 8 \times 10^{19}$ m$^{-3}$, $a = 0.5$ m.

(a) Calculate the electron cyclotron frequency $f_{ce}$ at the magnetic axis.

(b) For 2nd harmonic ECRH ($\omega = 2\omega_{ce}$), what gyrotron frequency is needed?

(c) Calculate the O-mode cutoff density at this frequency. Can the wave reach the core?

(d) If X-mode is used instead, where is the upper hybrid resonance layer located?

### Problem 2: Two-Stream Instability
An electron beam with $n_b = 10^{17}$ m$^{-3}$, $v_0 = 10^7$ m/s propagates through a plasma with $n_0 = 10^{19}$ m$^{-3}$.

(a) Calculate the background plasma frequency $\omega_{p0}$.

(b) Estimate the growth rate $\gamma$ using $\gamma \approx \omega_{p0}(n_b/n_0)^{1/3}$.

(c) At what wavenumber $k$ is the instability resonant (i.e., $kv_0 \approx \omega_{p0}$)?

(d) How many $e$-folding times does it take for the wave amplitude to grow by a factor of 1000? If the beam transits the plasma in $L/v_0 = 1$ μs, is this sufficient for significant growth?

### Problem 3: Weibel Instability in Laser Plasmas
A laser-heated plasma has $T_\perp = 500$ eV (heated by laser), $T_\parallel = 50$ eV (cold in the laser direction), and $n = 10^{21}$ m$^{-3}$.

(a) Calculate the anisotropy parameter $(T_\perp - T_\parallel)/T_\parallel$.

(b) Estimate the maximum Weibel growth rate $\gamma_{\text{max}}$.

(c) The generated magnetic field scales as $B^2/(8\pi) \sim nk_B(T_\perp - T_\parallel)$. Estimate the field strength in Tesla.

(d) Compare this field to the field required for electron gyroradius $\rho_L \sim 1/k_{\text{max}}$ where $k_{\text{max}}$ is the wavenumber of maximum growth. Are electrons magnetized by the self-generated field?

### Problem 4: Solar Wind Anisotropy
Solar wind observations at 1 AU show $\beta_\parallel = 0.8$, $\beta_\perp = 1.5$.

(a) Check if the plasma is stable to the firehose instability.

(b) Check if the plasma is stable to the mirror instability.

(c) If unstable, estimate the growth rate for $v_A = 50$ km/s, $L = 10^6$ km.

(d) The observed anisotropy is quasi-steady, suggesting marginal stability. Propose a mechanism that maintains the plasma near the stability boundary.

### Problem 5: Laser-Plasma Instabilities in ICF
A laser with intensity $I = 3 \times 10^{15}$ W/cm$^2$ and wavelength $\lambda = 351$ nm illuminates a plasma with $n = 0.1 n_c$ (where $n_c$ is critical density) and $T_e = 3$ keV.

(a) Calculate the critical density $n_c$ for this wavelength.

(b) Estimate the quiver velocity $v_{osc}$ of electrons in the laser field.

(c) Calculate the SRS growth rate using $\gamma_{SRS} \approx (k_L v_{osc}/4)\sqrt{\omega_0/\omega_s}$ where $k_L \approx 2\omega_0/c$ (backscatter).

(d) Compare to the Landau damping rate for Langmuir waves. Is SRS above threshold? What strategies could reduce SRS?

---

**Previous**: [11. Electromagnetic Waves](./11_Electromagnetic_Waves.md)
**Next**: [13. Two-Fluid Model](./13_Two_Fluid_Model.md)

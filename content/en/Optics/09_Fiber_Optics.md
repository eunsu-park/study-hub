# 09. Fiber Optics

[← Previous: 08. Laser Fundamentals](08_Laser_Fundamentals.md) | [Next: 10. Fourier Optics →](10_Fourier_Optics.md)

---

The global internet — streaming video, cloud computing, financial trading — depends almost entirely on hair-thin glass fibers carrying pulses of light across oceans and continents. A single optical fiber can transmit terabits per second over thousands of kilometers, a feat that would require thousands of copper cables. Fiber optics is the backbone of modern telecommunications, and its principles extend to sensors, medical instruments, and laser delivery systems.

This lesson develops the physics of optical fibers from first principles. We analyze how light is guided by total internal reflection, classify fiber types, quantify the loss and dispersion mechanisms that limit performance, and survey the key technologies — amplifiers, wavelength multiplexing, and gratings — that make modern fiber networks possible.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

1. Derive the guiding condition for step-index fibers and calculate the numerical aperture
2. Distinguish step-index and graded-index fibers and explain how grading reduces modal dispersion
3. Calculate the V-number and determine the single-mode cutoff condition
4. Identify and quantify the major attenuation mechanisms (absorption, Rayleigh scattering, bending loss)
5. Analyze the three types of dispersion (modal, chromatic, PMD) and their effects on pulse broadening
6. Explain the operating principle of erbium-doped fiber amplifiers (EDFA) and their role in long-haul communication
7. Describe wavelength division multiplexing (WDM) and fiber Bragg gratings (FBG)

---

## Table of Contents

1. [Guiding Light in a Fiber](#1-guiding-light-in-a-fiber)
2. [Fiber Types and Structure](#2-fiber-types-and-structure)
3. [Numerical Aperture and Acceptance Angle](#3-numerical-aperture-and-acceptance-angle)
4. [Modes in Optical Fibers](#4-modes-in-optical-fibers)
5. [Attenuation](#5-attenuation)
6. [Dispersion](#6-dispersion)
7. [Fiber Amplifiers (EDFA)](#7-fiber-amplifiers-edfa)
8. [Wavelength Division Multiplexing (WDM)](#8-wavelength-division-multiplexing-wdm)
9. [Fiber Bragg Gratings (FBG)](#9-fiber-bragg-gratings-fbg)
10. [Python Examples](#10-python-examples)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Guiding Light in a Fiber

### 1.1 Total Internal Reflection

An optical fiber guides light through **total internal reflection (TIR)**. When light travels from a medium with higher refractive index $n_1$ (core) to one with lower refractive index $n_2$ (cladding), and the angle of incidence exceeds the critical angle:

$$\theta_c = \arcsin\!\left(\frac{n_2}{n_1}\right)$$

the light is completely reflected back into the core with zero transmission loss.

> **Analogy**: Imagine skipping a stone across a lake. If you throw the stone at a shallow angle (grazing incidence), it bounces off the surface and continues. If you throw it steeply, it plunges in. The fiber core-cladding interface works the same way for light rays — shallow angles bounce and stay guided; steep angles escape into the cladding. The critical angle is the threshold between bouncing and plunging.

### 1.2 Basic Fiber Structure

A typical fiber has three layers:

```
            Cross-section:
         ┌────────────────────┐
         │   Coating (250 µm) │  Polymer protective layer
         │  ┌──────────────┐  │
         │  │ Cladding      │  │  Pure silica (n₂ ≈ 1.444)
         │  │  (125 µm)    │  │
         │  │  ┌────────┐  │  │
         │  │  │  Core  │  │  │  Doped silica (n₁ ≈ 1.450)
         │  │  │(8-62µm)│  │  │
         │  │  └────────┘  │  │
         │  └──────────────┘  │
         └────────────────────┘
```

The refractive index difference $\Delta = (n_1 - n_2)/n_1$ is very small — typically 0.3% to 2%. Despite this tiny difference, it is enough to confine light over kilometers.

### 1.3 Historical Context

Charles Kao and George Hockham predicted in 1966 that glass fibers could carry information if losses were reduced below 20 dB/km. At the time, fiber losses were ~1000 dB/km. Corning Glass Works achieved 17 dB/km in 1970 using fused silica with controlled dopants. By the 1980s, losses reached the theoretical minimum of ~0.2 dB/km at 1550 nm. Kao received the 2009 Nobel Prize in Physics for this work.

---

## 2. Fiber Types and Structure

### 2.1 Step-Index Fiber

The refractive index profile is a simple step function:

$$n(r) = \begin{cases} n_1 & r < a \quad (\text{core}) \\ n_2 & r \geq a \quad (\text{cladding}) \end{cases}$$

where $a$ is the core radius.

**Advantages**: Simple to manufacture, low cost.
**Disadvantage**: In multimode operation, different ray paths have different lengths, causing **modal dispersion** — the dominant bandwidth limiter.

### 2.2 Graded-Index Fiber

The refractive index decreases gradually from the center:

$$n(r) = \begin{cases} n_1\sqrt{1 - 2\Delta(r/a)^\alpha} & r < a \\ n_1\sqrt{1 - 2\Delta} \approx n_2 & r \geq a \end{cases}$$

The optimal profile exponent is $\alpha \approx 2$ (parabolic). With this profile, rays taking longer paths travel through lower-index regions and therefore move faster, arriving at nearly the same time as axial rays. This dramatically reduces modal dispersion.

**Result**: Graded-index fibers reduce modal dispersion by a factor of $\sim \Delta / 2$ compared to step-index fibers — from nanoseconds/km to picoseconds/km.

### 2.3 Single-Mode vs. Multimode

| Parameter | Single-mode (SMF) | Multimode (MMF) |
|-----------|-------------------|-----------------|
| Core diameter | 8-10 µm | 50 or 62.5 µm |
| Cladding diameter | 125 µm | 125 µm |
| Index profile | Step | Graded (usually) |
| Modes | 1 (LP$_{01}$) | Hundreds to thousands |
| Wavelength | 1310 nm, 1550 nm | 850 nm, 1300 nm |
| Bandwidth | Very high (THz·km) | Moderate (MHz·km to GHz·km) |
| Coupling | Difficult (tight alignment) | Easier (larger core) |
| Distance | Up to ~100 km (unamplified) | Up to ~2 km |
| Cost per meter | Lower | Lower |
| Connector cost | Higher | Lower |
| Application | Telecom, long-haul | LAN, data centers |

---

## 3. Numerical Aperture and Acceptance Angle

### 3.1 Derivation

Consider a ray entering the fiber at angle $\theta_a$ from the axis. By Snell's law at the entrance face:

$$n_0 \sin\theta_a = n_1 \sin\theta_r$$

where $n_0$ is the index of the surrounding medium (usually air, $n_0 = 1$) and $\theta_r$ is the refraction angle. For TIR at the core-cladding interface, we need the ray to hit the interface at an angle $\geq \theta_c$.

Working through the geometry, the maximum acceptance angle gives the **numerical aperture**:

$$\boxed{\text{NA} = n_0\sin\theta_a = \sqrt{n_1^2 - n_2^2} \approx n_1\sqrt{2\Delta}}$$

### 3.2 Typical Values

For a standard single-mode fiber: $n_1 = 1.450$, $n_2 = 1.447$, so:

$$\text{NA} = \sqrt{1.450^2 - 1.447^2} = \sqrt{0.008691} \approx 0.093$$

This gives an acceptance half-angle of $\theta_a = \arcsin(0.093) \approx 5.3°$ — a narrow cone. For multimode fibers, NA is typically 0.2-0.3 (acceptance angle ~12-17°).

### 3.3 Physical Meaning

The NA determines:
- How easily light can be coupled into the fiber (larger NA → easier coupling)
- How many modes the fiber supports (larger NA → more modes)
- The sensitivity of the fiber to bending (larger NA → less bending loss)

---

## 4. Modes in Optical Fibers

### 4.1 The V-Number (Normalized Frequency)

The **V-number** is the single most important parameter for characterizing a fiber:

$$\boxed{V = \frac{2\pi a}{\lambda}\,\text{NA} = \frac{2\pi a}{\lambda}\sqrt{n_1^2 - n_2^2}}$$

### 4.2 Single-Mode Condition

For a step-index fiber, only the fundamental mode (LP$_{01}$) propagates when:

$$\boxed{V < 2.405}$$

The value 2.405 is the first zero of the Bessel function $J_0$. At exactly $V = 2.405$ (the **cutoff** wavelength $\lambda_c$), the second mode (LP$_{11}$) just begins to propagate.

$$\lambda_c = \frac{2\pi a\,\text{NA}}{2.405}$$

### 4.3 Number of Modes

For a step-index multimode fiber with large V:

$$M \approx \frac{V^2}{2}$$

For a graded-index fiber with parabolic profile:

$$M \approx \frac{V^2}{4}$$

**Example**: A multimode step-index fiber with $a = 25\,\mu\text{m}$, NA = 0.2, $\lambda = 850\,\text{nm}$:

$$V = \frac{2\pi \times 25\times10^{-6}}{850\times10^{-9}} \times 0.2 = 36.9$$

$$M \approx \frac{36.9^2}{2} \approx 681 \text{ modes}$$

### 4.4 Mode Field Diameter

For a single-mode fiber, the fundamental mode has a near-Gaussian intensity profile. The **mode field diameter (MFD)** is defined as the diameter at which the field amplitude drops to $1/e$ of its peak. The Marcuse approximation gives:

$$\frac{w}{a} \approx 0.65 + \frac{1.619}{V^{3/2}} + \frac{2.879}{V^6}$$

The MFD is larger than the core diameter (the field extends into the cladding), especially near cutoff.

---

## 5. Attenuation

### 5.1 Beer-Lambert Law for Fibers

The power at distance $L$ is:

$$P(L) = P_0 \cdot 10^{-\alpha L / 10}$$

where $\alpha$ is the attenuation coefficient in **dB/km**. Equivalently:

$$\alpha_{\text{dB/km}} = \frac{10}{L}\log_{10}\!\left(\frac{P_0}{P(L)}\right)$$

### 5.2 Attenuation Mechanisms

**Rayleigh Scattering**: Intrinsic scattering from microscopic density fluctuations in the glass. Scales as $\lambda^{-4}$, making it the dominant loss mechanism at short wavelengths. At 1550 nm: ~0.15 dB/km.

$$\alpha_R = \frac{C_R}{\lambda^4}$$

where $C_R \approx 0.7\text{-}0.9\;\text{dB}\cdot\mu\text{m}^4/\text{km}$ for silica.

**Material Absorption**:
- **UV absorption**: Electronic transitions in SiO$_2$ (Urbach tail). Exponentially decreasing with wavelength.
- **IR absorption**: Molecular vibrations (Si-O bonds). Exponentially increasing with wavelength beyond ~1600 nm.
- **OH absorption**: Water impurity ($\text{OH}^-$ ions) creates a strong peak at 1383 nm (and harmonics at 1240, 950 nm). Modern "low-water-peak" fibers (ITU-T G.652.D) virtually eliminate this peak.

**Bending Loss**:
- **Macrobending**: When the fiber is bent below a critical radius, the evanescent field in the cladding must travel faster than $c/n_2$, which is impossible — the light radiates away. Critical bend radius increases sharply near cutoff wavelength.
- **Microbending**: Small random deformations (from manufacturing or cable stress) couple guided modes to radiation modes. Minimized by proper cabling.

### 5.3 The Attenuation Spectrum

```
Attenuation (dB/km)
    5 │
      │╲
    2 │ ╲    Rayleigh                                IR
      │  ╲   scattering                             absorption
    1 │   ╲       ∧                                    ╱
      │    ╲     ╱ ╲ OH peak                         ╱
  0.5 │     ╲   ╱   ╲ (1383 nm)                    ╱
      │      ╲ ╱     ╲                             ╱
  0.2 │───────╳───────╲────╲──────────────────╱───
      │              ╲  ╲────────╲──────╱
 0.15 │               Minimum: ~0.17 dB/km at 1550 nm
      └──────────────────────────────────────────
      800   1000  1200  1400  1550  1600   1800
                    Wavelength (nm)

      |--O band--|--E--|S band|C band|L band|
```

The three primary telecom windows:
- **O-band** (1260-1360 nm): Zero dispersion for standard SMF
- **C-band** (1530-1565 nm): Minimum loss, EDFA gain window
- **L-band** (1565-1625 nm): Extended EDFA window

---

## 6. Dispersion

Dispersion causes optical pulses to broaden as they propagate, limiting the bit rate and distance of a fiber link. There are three types.

### 6.1 Modal Dispersion

In multimode fibers, different modes travel different path lengths. For a step-index fiber:

$$\Delta\tau_{\text{modal}} = \frac{n_1 L}{c}\Delta \approx \frac{n_1^2 L}{n_2 c}\Delta$$

**Example**: For $n_1 = 1.48$, $\Delta = 0.01$, $L = 1\,\text{km}$:

$$\Delta\tau = \frac{1.48 \times 1000}{3\times10^8} \times 0.01 \approx 49\,\text{ns/km}$$

This limits bandwidth to roughly $B \approx 1/(2\Delta\tau) \approx 10\,\text{MHz}\cdot\text{km}$.

For graded-index (parabolic) fibers: $\Delta\tau \approx \frac{n_1 L}{2c}\Delta^2$ — reduction by a factor of $\Delta/2$.

**Single-mode fibers eliminate modal dispersion entirely**, since there is only one mode.

### 6.2 Chromatic Dispersion

Even in single-mode fibers, different wavelength components of a pulse travel at different group velocities. Chromatic dispersion has two contributions:

**Material dispersion**: The refractive index of silica varies with wavelength ($dn/d\lambda \neq 0$).

**Waveguide dispersion**: The mode's effective index depends on how much of the field is in the core vs. cladding, which changes with wavelength.

The total chromatic dispersion is characterized by the **dispersion parameter**:

$$D = -\frac{\lambda}{c}\frac{d^2n_{\text{eff}}}{d\lambda^2} \quad [\text{ps/(nm·km)}]$$

For standard SMF-28 fiber: $D = 0$ at $\lambda_0 \approx 1310\,\text{nm}$ and $D \approx +17\,\text{ps/(nm·km)}$ at 1550 nm.

**Pulse broadening** due to chromatic dispersion:

$$\Delta\tau_{\text{chrom}} = |D| \cdot \Delta\lambda \cdot L$$

where $\Delta\lambda$ is the spectral width of the source.

**Example**: Laser diode with $\Delta\lambda = 0.1\,\text{nm}$ at 1550 nm, $L = 100\,\text{km}$:

$$\Delta\tau = 17 \times 0.1 \times 100 = 170\,\text{ps}$$

This limits the bit rate to roughly $B < 1/(4\Delta\tau) \approx 1.5\,\text{Gb/s}$.

### 6.3 Dispersion Management

**Dispersion-shifted fiber (DSF)**: Engineered waveguide dispersion shifts $\lambda_0$ to 1550 nm ($D = 0$ at 1550 nm). Problem: $D = 0$ enhances nonlinear effects (four-wave mixing) in WDM systems.

**Non-zero dispersion-shifted fiber (NZ-DSF)**: Small but nonzero $D$ at 1550 nm (e.g., $D \approx 4\,\text{ps/(nm·km)}$). Reduces dispersion while avoiding nonlinear penalties.

**Dispersion-compensating fiber (DCF)**: Fiber with large negative $D$ (e.g., $-80\,\text{ps/(nm·km)}$) used to cancel the accumulated dispersion of standard fiber. A short length of DCF compensates a long span of SMF.

### 6.4 Polarization Mode Dispersion (PMD)

In an ideal single-mode fiber, two orthogonal polarization modes (degenerate) travel at the same speed. In real fibers, imperfections (core ellipticity, stress) break this degeneracy, causing a differential group delay:

$$\Delta\tau_{\text{PMD}} = D_{\text{PMD}} \sqrt{L}$$

where $D_{\text{PMD}}$ is the PMD coefficient, typically 0.01-0.1 $\text{ps}/\sqrt{\text{km}}$ for modern fibers. Note the $\sqrt{L}$ dependence — PMD is a random process (mode coupling changes along the fiber).

At 40 Gb/s and above, PMD becomes a significant impairment requiring adaptive compensation.

---

## 7. Fiber Amplifiers (EDFA)

### 7.1 The Need for Amplification

Before optical amplifiers, long-haul fiber links required electronic regenerators every 40-80 km: the optical signal was detected, converted to electrical, re-amplified, re-timed, and re-transmitted as light. These regenerators were expensive, wavelength-specific, and bit-rate dependent.

The **erbium-doped fiber amplifier (EDFA)**, demonstrated by Desurvire and Mears in the late 1980s, revolutionized fiber communications. It amplifies light directly — all wavelengths in the C-band simultaneously, at any bit rate, with any modulation format.

### 7.2 Operating Principle

Erbium ions (Er$^{3+}$) doped into silica fiber have a three-level transition at ~1530-1565 nm (the C-band). Pumped at 980 nm or 1480 nm:

```
      ⁴I₁₁/₂ ─────── 980 nm pump band
                ↓ fast non-radiative
      ⁴I₁₃/₂ ─────── Metastable upper level (τ ≈ 10 ms)
                ↓ signal amplification (1530-1565 nm)
      ⁴I₁₅/₂ ─────── Ground state
```

- 980 nm pump: absorbed by $^4I_{11/2}$ band, fast decay to $^4I_{13/2}$. Low noise figure (~3 dB).
- 1480 nm pump: directly pumps to $^4I_{13/2}$. Higher efficiency but slightly higher noise.
- Signal photons stimulate emission from $^4I_{13/2}$ to $^4I_{15/2}$, producing gain.

### 7.3 Key EDFA Parameters

- **Gain**: 20-40 dB (100 to 10,000 times amplification)
- **Bandwidth**: ~35 nm in C-band (extended with L-band EDFA to ~70 nm)
- **Saturation output power**: 10-25 dBm (10-300 mW)
- **Noise figure**: 3-6 dB (quantum limit: 3 dB)
- **Fiber length**: 5-30 m of erbium-doped fiber

### 7.4 Amplified Spontaneous Emission (ASE)

Spontaneous emission from excited Er$^{3+}$ ions is also amplified, producing broadband **ASE noise**. This is the fundamental noise source in optically amplified systems. The noise power spectral density is:

$$S_{\text{ASE}} = n_{\text{sp}}(G-1)h\nu$$

where $n_{\text{sp}} \geq 1$ is the spontaneous emission factor and $G$ is the gain. The noise figure is $\text{NF} = 2n_{\text{sp}}(G-1)/G \approx 2n_{\text{sp}}$ for high gain.

> **Analogy**: An EDFA is like a relay station for light. Imagine a chain of people passing a whispered message across a field (the signal). At each relay, the message is amplified (shouted louder), but each relay also adds some background chatter (ASE noise). After many relays, the message is loud but so is the accumulated chatter. The art of system design is spacing relays so the signal stays louder than the noise.

---

## 8. Wavelength Division Multiplexing (WDM)

### 8.1 Concept

WDM is the optical equivalent of radio frequency multiplexing: multiple signals at different wavelengths share the same fiber, each carrying independent data. This multiplies the fiber's capacity by the number of channels.

```
   λ₁ ──┐                                     ┌── λ₁
   λ₂ ──┤ MUX ════════ Single fiber ════════ DEMUX ├── λ₂
   λ₃ ──┤      (all wavelengths together)      ├── λ₃
    ⋮   ┘                                      └──  ⋮
   λₙ ──┘                                     └── λₙ
```

### 8.2 WDM Standards

**CWDM (Coarse WDM)**: Channel spacing 20 nm, ~18 channels (1270-1610 nm). Low cost, no temperature control needed. Short-reach (metro, access networks).

**DWDM (Dense WDM)**: Channel spacing 100 GHz (~0.8 nm), 50 GHz (~0.4 nm), or even 12.5 GHz. 40-160+ channels in the C-band alone. Long-haul, submarine cables.

### 8.3 WDM System Capacity

A modern submarine cable (e.g., Google's Dunant, 2020) carries:
- ~250 DWDM channels per fiber pair
- Each channel: 200-400 Gb/s (coherent modulation)
- Multiple fiber pairs per cable
- Total capacity: ~250 Tb/s per cable

### 8.4 Key Components

- **Multiplexers/Demultiplexers**: Arrayed waveguide gratings (AWG), thin-film filters
- **Optical add/drop multiplexers (OADM)**: Insert/remove individual wavelengths at intermediate nodes
- **Reconfigurable OADM (ROADM)**: Remotely configurable wavelength routing using wavelength-selective switches (WSS)

---

## 9. Fiber Bragg Gratings (FBG)

### 9.1 Structure

A fiber Bragg grating is a periodic modulation of the refractive index in the fiber core:

$$n(z) = n_{\text{eff}} + \delta n \cos\!\left(\frac{2\pi z}{\Lambda}\right)$$

where $\Lambda$ is the grating period (typically ~500 nm) and $\delta n$ is the index modulation amplitude (~$10^{-4}$ to $10^{-3}$).

FBGs are written into the fiber by exposing the core to a UV interference pattern (using a phase mask or two-beam interferometry).

### 9.2 Bragg Condition

The grating reflects light at the **Bragg wavelength**:

$$\boxed{\lambda_B = 2n_{\text{eff}}\Lambda}$$

Light at other wavelengths passes through unaffected. The reflection bandwidth is:

$$\Delta\lambda \approx \lambda_B \frac{\delta n}{n_{\text{eff}}} \quad (\text{for weak gratings})$$

### 9.3 Applications

**Wavelength filters**: Ultra-narrow bandpass/bandstop filters for WDM systems.

**Sensors**: The Bragg wavelength shifts with temperature ($\sim 10\,\text{pm/°C}$) and strain ($\sim 1.2\,\text{pm/}\mu\varepsilon$). Multiple FBGs at different $\lambda_B$ can be multiplexed on a single fiber for distributed sensing in bridges, pipelines, aircraft wings, and more.

**Dispersion compensation**: Chirped FBGs (varying $\Lambda$ along the length) reflect different wavelengths at different positions, introducing a controlled time delay to compress dispersed pulses.

**Laser stabilization**: FBG reflectors in fiber lasers define the lasing wavelength with high precision and stability.

---

## 10. Python Examples

### 10.1 Fiber Mode Calculation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv  # Bessel functions

def compute_modes(n1, n2, a, wavelength):
    """
    Compute guided mode effective indices for a step-index fiber.

    We solve the characteristic equation for LP modes:
    J_l(u) / [u * J_{l-1}(u)] = -K_l(w) / [w * K_{l-1}(w)]
    where u = a*sqrt(k0^2*n1^2 - beta^2), w = a*sqrt(beta^2 - k0^2*n2^2)

    This eigenvalue equation arises from matching the field and its derivative
    at the core-cladding boundary — the same physics as quantum well states.
    """
    k0 = 2 * np.pi / wavelength
    V = k0 * a * np.sqrt(n1**2 - n2**2)

    print(f"V-number: {V:.3f}")
    print(f"NA: {np.sqrt(n1**2 - n2**2):.4f}")
    print(f"Acceptance angle: {np.degrees(np.arcsin(np.sqrt(n1**2 - n2**2))):.2f}°")

    if V < 2.405:
        print("Single-mode fiber (only LP01 propagates)")
    else:
        # Approximate number of modes
        M_step = V**2 / 2
        print(f"Multimode fiber: ~{int(M_step)} modes (step-index)")
        print(f"                 ~{int(V**2/4)} modes (graded-index parabolic)")

    return V

# Standard single-mode fiber (SMF-28)
print("=== SMF-28 at 1310 nm ===")
V_1310 = compute_modes(n1=1.4504, n2=1.4447, a=4.1e-6, wavelength=1310e-9)

print("\n=== SMF-28 at 1550 nm ===")
V_1550 = compute_modes(n1=1.4504, n2=1.4447, a=4.1e-6, wavelength=1550e-9)

print("\n=== Multimode fiber at 850 nm ===")
V_mm = compute_modes(n1=1.480, n2=1.460, a=25e-6, wavelength=850e-9)
```

### 10.2 Attenuation Spectrum Model

```python
import numpy as np
import matplotlib.pyplot as plt

def fiber_attenuation(wavelength_nm):
    """
    Model the attenuation spectrum of standard silica fiber.

    Three contributions are modeled:
    1. Rayleigh scattering (∝ λ^-4): dominant below ~1300 nm
    2. IR absorption (exponential rise): dominant above ~1600 nm
    3. OH peak at 1383 nm: from residual water in the glass

    The minimum loss occurs around 1550 nm — the sweet spot for telecom.
    This is not a coincidence: the entire long-haul fiber industry
    operates at 1550 nm because physics dictates it.
    """
    lam = wavelength_nm / 1000.0  # Convert to micrometers

    # Rayleigh scattering: C_R / lambda^4
    rayleigh = 0.8 / lam**4

    # Infrared absorption: exponential tail of molecular resonances
    ir_absorption = 6e11 * np.exp(-48.0 / lam)

    # UV absorption: Urbach tail
    uv_absorption = 1.5e-2 * np.exp(4.6 * (1.0/lam - 1.0/0.16))

    # OH impurity peak at 1383 nm (Gaussian approximation)
    oh_peak = 0.5 * np.exp(-0.5 * ((lam - 1.383) / 0.015)**2)

    total = rayleigh + ir_absorption + uv_absorption + oh_peak
    return total, rayleigh, ir_absorption, oh_peak

wavelengths = np.linspace(800, 1700, 1000)
total, rayleigh, ir, oh = fiber_attenuation(wavelengths)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(wavelengths, total, 'k-', linewidth=2, label='Total')
ax.semilogy(wavelengths, rayleigh, 'b--', linewidth=1, label='Rayleigh scattering')
ax.semilogy(wavelengths, ir, 'r--', linewidth=1, label='IR absorption')
ax.semilogy(wavelengths, oh, 'g--', linewidth=1, label='OH peak (1383 nm)')

# Mark telecom bands
bands = {'O': (1260, 1360), 'E': (1360, 1460), 'S': (1460, 1530),
         'C': (1530, 1565), 'L': (1565, 1625)}
colors = ['#FFE0B2', '#E0F7FA', '#F3E5F5', '#E8F5E9', '#FFF9C4']
for (name, (lo, hi)), color in zip(bands.items(), colors):
    ax.axvspan(lo, hi, alpha=0.3, color=color, label=f'{name}-band')

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Attenuation (dB/km)', fontsize=12)
ax.set_title('Silica Optical Fiber Attenuation Spectrum', fontsize=14)
ax.set_ylim(0.1, 10)
ax.set_xlim(800, 1700)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fiber_attenuation.png', dpi=150, bbox_inches='tight')
plt.show()

# Find and report the minimum
min_idx = np.argmin(total)
print(f"Minimum attenuation: {total[min_idx]:.3f} dB/km at {wavelengths[min_idx]:.0f} nm")
```

### 10.3 Dispersion Calculation

```python
import numpy as np
import matplotlib.pyplot as plt

def sellmeier_silica(wavelength_um):
    """
    Sellmeier equation for fused silica refractive index.

    The Sellmeier coefficients encode the resonance wavelengths
    of the UV and IR absorption bands of SiO2. Between these
    resonances, the material is transparent — the fiber window.
    """
    lam2 = wavelength_um**2
    # Standard Sellmeier coefficients for fused silica
    B = [0.6961663, 0.4079426, 0.8974794]
    C = [0.0684043**2, 0.1162414**2, 9.896161**2]  # C_i = lambda_i^2
    n2 = 1.0
    for bi, ci in zip(B, C):
        n2 += bi * lam2 / (lam2 - ci)
    return np.sqrt(n2)

def material_dispersion(wavelength_um):
    """
    Compute material dispersion D_mat = -(λ/c) * d²n/dλ².

    Material dispersion arises because the glass refractive index
    varies with wavelength. It is zero near 1270 nm and positive
    (anomalous) above that wavelength for silica.
    """
    dlam = 1e-4  # Small step for numerical derivative (in µm)
    lam = wavelength_um

    # Second derivative via central differences (3-point formula)
    n_plus = sellmeier_silica(lam + dlam)
    n_minus = sellmeier_silica(lam - dlam)
    n_center = sellmeier_silica(lam)
    d2n_dlam2 = (n_plus - 2*n_center + n_minus) / dlam**2

    # D = -(λ/c) * d²n/dλ² [ps/(nm·km)]
    c = 3e8  # m/s
    # Convert units: λ in µm → m, d²n/dλ² in 1/µm² → 1/m²
    D = -(lam * 1e-6 / c) * (d2n_dlam2 * 1e12)  # Result in ps/(nm·km)
    return D

wavelengths = np.linspace(1.0, 1.7, 500)  # micrometers
n_vals = sellmeier_silica(wavelengths)
D_vals = material_dispersion(wavelengths)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(wavelengths * 1000, n_vals, 'b-', linewidth=2)
ax1.set_ylabel('Refractive index n', fontsize=12)
ax1.set_title('Silica Refractive Index and Chromatic Dispersion', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(wavelengths * 1000, D_vals, 'r-', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle='--')
ax2.axvline(x=1310, color='green', linestyle=':', alpha=0.7,
            label='Zero dispersion (~1310 nm)')
ax2.axvline(x=1550, color='orange', linestyle=':', alpha=0.7,
            label='C-band center (1550 nm)')
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('D [ps/(nm·km)]', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fiber_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()

# Report D at key wavelengths
for lam in [1.31, 1.55]:
    print(f"D at {lam*1000:.0f} nm: {material_dispersion(lam):.2f} ps/(nm·km)")
```

---

## 11. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Total internal reflection | Critical angle: $\theta_c = \arcsin(n_2/n_1)$ |
| Numerical aperture | $\text{NA} = \sqrt{n_1^2 - n_2^2}$ |
| V-number | $V = (2\pi a/\lambda)\,\text{NA}$ |
| Single-mode condition | $V < 2.405$ |
| Number of modes (step) | $M \approx V^2/2$ |
| Attenuation | $P = P_0 \cdot 10^{-\alpha L/10}$; minimum ~0.17 dB/km at 1550 nm |
| Rayleigh scattering | $\propto \lambda^{-4}$ |
| Modal dispersion (step) | $\Delta\tau \approx n_1 L \Delta / c$ |
| Chromatic dispersion | $\Delta\tau = |D| \Delta\lambda\, L$; $D = 0$ at ~1310 nm for standard SMF |
| PMD | $\Delta\tau = D_{\text{PMD}}\sqrt{L}$ |
| EDFA | Amplifies C-band (1530-1565 nm); gain 20-40 dB; NF $\geq$ 3 dB |
| Bragg wavelength | $\lambda_B = 2n_{\text{eff}}\Lambda$ |
| WDM | Multiple wavelengths on one fiber; DWDM spacing ~0.4-0.8 nm |

---

## 12. Exercises

### Exercise 1: Fiber Design

Design a step-index single-mode fiber for operation at $\lambda = 1550\,\text{nm}$ with $n_1 = 1.450$ and $\Delta = 0.3\%$.

(a) Calculate $n_2$ and the NA.
(b) Find the maximum core radius $a$ for single-mode operation.
(c) Calculate the mode field diameter using the Marcuse approximation.
(d) What is the cutoff wavelength?

### Exercise 2: Link Budget

A fiber optic link uses standard SMF with $\alpha = 0.2\,\text{dB/km}$ at 1550 nm. The transmitter power is +3 dBm and the receiver sensitivity is -28 dBm. Each connector has 0.5 dB loss, and there are 2 connectors per splice and 5 splices.

(a) Calculate the total connector/splice loss.
(b) What is the maximum fiber length with a 3 dB system margin?
(c) If an EDFA with 25 dB gain is inserted at the midpoint, what is the new maximum length?

### Exercise 3: Dispersion-Limited Distance

A 10 Gb/s NRZ system uses a DFB laser ($\Delta\lambda = 0.1\,\text{nm}$) at 1550 nm on standard SMF ($D = 17\,\text{ps/(nm·km)}$).

(a) Calculate the pulse broadening per km.
(b) Using the criterion $\Delta\tau < T_{\text{bit}}/4$ (where $T_{\text{bit}} = 100\,\text{ps}$), find the maximum uncompensated distance.
(c) How much DCF ($D = -80\,\text{ps/(nm·km)}$) is needed to compensate 100 km of standard fiber?

### Exercise 4: FBG Sensor

An FBG with $\Lambda = 535\,\text{nm}$ is written in fiber with $n_{\text{eff}} = 1.447$.

(a) Calculate the Bragg wavelength.
(b) If the temperature sensitivity is $10\,\text{pm/°C}$, what Bragg wavelength shift corresponds to a 50°C temperature rise?
(c) If the strain sensitivity is $1.2\,\text{pm/}\mu\varepsilon$, what strain produces the same shift as in (b)?

---

## 13. References

1. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapter 10.
2. Agrawal, G. P. (2021). *Fiber-Optic Communication Systems* (5th ed.). Wiley.
3. Okamoto, K. (2022). *Fundamentals of Optical Waveguides* (3rd ed.). Academic Press.
4. Desurvire, E. (2002). *Erbium-Doped Fiber Amplifiers*. Wiley.
5. Kao, C. K. (2009). Nobel Prize Lecture: "Sand from centuries past: Send future voices fast."

---

[← Previous: 08. Laser Fundamentals](08_Laser_Fundamentals.md) | [Next: 10. Fourier Optics →](10_Fourier_Optics.md)

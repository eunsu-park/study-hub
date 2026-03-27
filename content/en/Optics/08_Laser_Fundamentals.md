# 08. Laser Fundamentals

[← Previous: 07. Polarization](07_Polarization.md) | [Next: 09. Fiber Optics →](09_Fiber_Optics.md)

---

The laser — Light Amplification by Stimulated Emission of Radiation — is arguably the most transformative optical invention of the 20th century. From reading barcodes at the supermarket to performing eye surgery, from telecommunications to gravitational-wave detection (LIGO), lasers are everywhere. What makes a laser special is not just that it produces light, but *how* it produces light: through a fundamentally quantum-mechanical process that yields extraordinary coherence, directionality, and spectral purity.

This lesson builds the physics of lasers from the ground up. We start with Einstein's insight about stimulated emission, work through the conditions needed for laser action, examine the optical cavity that provides feedback, and survey the major laser types. We then develop the Gaussian beam formalism that describes how laser light propagates through space and optical systems.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

1. Explain the three radiation processes (absorption, spontaneous emission, stimulated emission) and derive the Einstein relations
2. Define population inversion and analyze the conditions required to achieve it in three-level and four-level systems
3. Describe the role of the optical cavity and derive the threshold gain condition
4. Distinguish longitudinal and transverse modes and their physical origins
5. Compare the operating principles, characteristics, and applications of major laser types (gas, solid-state, semiconductor, fiber)
6. Quantify laser coherence (temporal and spatial) and relate it to linewidth and beam quality
7. Apply Gaussian beam propagation formulas and ABCD matrices to predict beam evolution through optical systems

---

## Table of Contents

1. [Light-Matter Interaction: The Three Processes](#1-light-matter-interaction-the-three-processes)
2. [Einstein Coefficients and Relations](#2-einstein-coefficients-and-relations)
3. [Population Inversion](#3-population-inversion)
4. [Optical Feedback: The Laser Cavity](#4-optical-feedback-the-laser-cavity)
5. [Laser Modes](#5-laser-modes)
6. [Major Laser Types](#6-major-laser-types)
7. [Coherence Properties](#7-coherence-properties)
8. [Gaussian Beam Propagation](#8-gaussian-beam-propagation)
9. [ABCD Matrix Formalism](#9-abcd-matrix-formalism)
10. [Python Examples](#10-python-examples)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Light-Matter Interaction: The Three Processes

When electromagnetic radiation interacts with atoms (or molecules, or ions), three fundamental processes can occur, all involving transitions between two energy levels $E_1$ (lower) and $E_2$ (upper) with energy difference $\Delta E = E_2 - E_1 = h\nu$.

### 1.1 Absorption

A photon with energy $h\nu = E_2 - E_1$ is absorbed, promoting the atom from $E_1$ to $E_2$. The photon is destroyed. The rate of absorption is:

$$\frac{dN_1}{dt}\bigg|_{\text{abs}} = -B_{12}\,\rho(\nu)\,N_1$$

where $N_1$ is the population of level 1, $\rho(\nu)$ is the spectral energy density of the radiation field, and $B_{12}$ is the Einstein B coefficient for absorption.

### 1.2 Spontaneous Emission

An atom in the excited state $E_2$ spontaneously drops to $E_1$, emitting a photon of energy $h\nu$. This process is random in both timing and direction — the emitted photon goes in a random direction with a random phase. The rate is:

$$\frac{dN_2}{dt}\bigg|_{\text{sp}} = -A_{21}\,N_2$$

where $A_{21}$ is the Einstein A coefficient. The spontaneous lifetime is $\tau_{\text{sp}} = 1/A_{21}$.

### 1.3 Stimulated Emission

This is Einstein's key insight (1917). An incoming photon with energy $h\nu$ can *stimulate* an atom in $E_2$ to drop to $E_1$, producing a second photon that is **identical** to the first — same frequency, same direction, same phase, same polarization. The rate is:

$$\frac{dN_2}{dt}\bigg|_{\text{st}} = -B_{21}\,\rho(\nu)\,N_2$$

> **Analogy**: Think of stimulated emission like a perfectly synchronized choir. When one singer (the incoming photon) hits a note, another singer (the excited atom) joins in at exactly the same pitch, timing, and key. The result is two perfectly synchronized voices (two identical photons) — and those two can stimulate more singers, leading to an ever-growing, perfectly coherent chorus. This is amplification by stimulated emission.

### 1.4 Why Stimulated Emission Matters

The stimulated photon is a **clone** of the stimulating photon. This means:
- **Same frequency** $\nu$ — spectral purity
- **Same direction** $\hat{k}$ — directionality
- **Same phase** $\phi$ — coherence
- **Same polarization** $\hat{e}$ — polarization purity

No other light source produces photons with all four of these properties simultaneously.

---

## 2. Einstein Coefficients and Relations

### 2.1 Thermal Equilibrium Argument

At thermal equilibrium, the rate of upward transitions equals the rate of downward transitions:

$$B_{12}\,\rho(\nu)\,N_1 = A_{21}\,N_2 + B_{21}\,\rho(\nu)\,N_2$$

The populations follow the Boltzmann distribution:

$$\frac{N_2}{N_1} = \frac{g_2}{g_1}\exp\!\left(-\frac{h\nu}{k_BT}\right)$$

where $g_1, g_2$ are the degeneracies of the two levels.

### 2.2 Deriving the Einstein Relations

Solving for $\rho(\nu)$ and comparing with Planck's blackbody formula:

$$\rho(\nu) = \frac{8\pi h\nu^3}{c^3}\frac{1}{e^{h\nu/(k_BT)}-1}$$

we obtain the two **Einstein relations**:

$$\boxed{g_1 B_{12} = g_2 B_{21}}$$

$$\boxed{A_{21} = \frac{8\pi h\nu^3}{c^3}B_{21}}$$

The first relation tells us that absorption and stimulated emission are fundamentally the same process (for equal degeneracies, $B_{12} = B_{21}$). The second shows that spontaneous emission grows rapidly with frequency ($\propto \nu^3$), which is why making X-ray lasers is extraordinarily difficult.

### 2.3 Physical Interpretation

The ratio of spontaneous to stimulated emission rates is:

$$\frac{A_{21}}{B_{21}\rho(\nu)} = e^{h\nu/(k_BT)} - 1$$

At room temperature ($T \approx 300\,\text{K}$) and optical frequencies ($\nu \sim 5 \times 10^{14}\,\text{Hz}$):

$$\frac{h\nu}{k_BT} \approx \frac{(6.63\times10^{-34})(5\times10^{14})}{(1.38\times10^{-23})(300)} \approx 80$$

So $A_{21}/(B_{21}\rho) \approx e^{80} \approx 10^{35}$. Spontaneous emission completely dominates at thermal equilibrium for visible light. This is why incandescent bulbs produce incoherent light, and why achieving laser action requires driving the system far from equilibrium.

---

## 3. Population Inversion

### 3.1 The Fundamental Requirement

For stimulated emission to dominate over absorption, we need more atoms in the upper level than the lower level (accounting for degeneracies):

$$\frac{N_2}{g_2} > \frac{N_1}{g_1}$$

This condition — **population inversion** — never occurs at thermal equilibrium (the Boltzmann factor $e^{-h\nu/(k_BT)} < 1$ always). It must be created by an external energy source, called the **pump**.

### 3.2 Why Two-Level Systems Cannot Lase

Consider pumping a two-level system hard. The maximum we can achieve is $N_2 = N_1$ (equal populations, called *saturation* or *bleaching*). We can never get $N_2 > N_1$ because the stronger we pump, the more stimulated emission also increases. The system approaches transparency but never achieves gain.

**Result**: A true two-level system cannot sustain population inversion. We need at least three levels.

### 3.3 Three-Level System

In a three-level laser (e.g., ruby laser, the first laser demonstrated by Maiman in 1960):

```
Level 3: --------- Pump band (short-lived)
           ↑ pump    ↓ fast non-radiative decay
Level 2: --------- Upper laser level (metastable, long τ)
           ↓ laser transition (slow)
Level 1: --------- Ground state (lower laser level)
```

- Atoms are pumped from level 1 to level 3
- Fast non-radiative relaxation brings them to level 2
- Level 2 is metastable (long lifetime $\tau_2$), so population accumulates
- Laser transition occurs from 2 → 1

**Challenge**: The lower laser level is the ground state. At room temperature, most atoms start in level 1, so we must pump more than half the atoms to level 2 before we even reach transparency. Three-level lasers have high threshold pump power.

### 3.4 Four-Level System

Most modern lasers use a four-level scheme (e.g., Nd:YAG):

```
Level 4: --------- Pump band (short-lived)
           ↑ pump    ↓ fast decay
Level 3: --------- Upper laser level (metastable)
           ↓ laser transition
Level 2: --------- Lower laser level (short-lived)
           ↓ fast decay
Level 1: --------- Ground state
```

- Lower laser level (2) empties rapidly by non-radiative decay to level 1
- Population inversion between levels 3 and 2 is achieved with minimal pumping
- Threshold is much lower than three-level systems

### 3.5 Gain Coefficient

With population inversion, an optical beam experiences exponential growth. The small-signal gain coefficient is:

$$g(\nu) = \frac{c^2}{8\pi\nu^2\tau_{\text{sp}}}\left(\frac{N_2}{g_2} - \frac{N_1}{g_1}\right)g_2\,\mathcal{L}(\nu)$$

where $\mathcal{L}(\nu)$ is the normalized lineshape function. The intensity grows as:

$$I(z) = I_0\,e^{g(\nu)\,z}$$

This is **optical amplification** — the "A" in LASER.

---

## 4. Optical Feedback: The Laser Cavity

### 4.1 From Amplifier to Oscillator

A gain medium alone is an optical amplifier (like an EDFA in fiber optics). To make an **oscillator** (a laser), we add **positive feedback** — mirrors that send the light back through the gain medium repeatedly.

The simplest cavity is the **Fabry-Perot resonator**: two mirrors separated by distance $L$.

```
     Mirror 1 (R₁)          Gain Medium          Mirror 2 (R₂)
     ┌──────┐         ┌─────────────────┐         ┌──────┐
  ←──│██████│←────────│  ← → ← → ← →  │────────→│██████│──→ Output
     │██████│  R₁≈1   │   amplification │         │██████│  R₂<1
     └──────┘         └─────────────────┘         └──────┘
         ←─────────────── L ──────────────────→
```

One mirror (the *output coupler*) has $R_2 < 1$, allowing a fraction of the light to escape as the laser beam.

### 4.2 Threshold Condition

For the laser to oscillate, the round-trip gain must equal (or exceed) the round-trip loss. After one round trip ($2L$):

$$R_1 R_2\,e^{2(g-\alpha)L} \geq 1$$

where $\alpha$ is the internal loss coefficient per unit length (scattering, absorption). Taking logarithms, the **threshold gain** is:

$$\boxed{g_{\text{th}} = \alpha + \frac{1}{2L}\ln\!\left(\frac{1}{R_1 R_2}\right)}$$

The second term represents the mirror loss. When $g > g_{\text{th}}$, light intensity builds up exponentially until gain saturation brings $g$ exactly to $g_{\text{th}}$ in steady state.

### 4.3 Cavity Stability

Not all mirror configurations form a stable cavity. Using the stability parameter $g_i = 1 - L/R_i$ (where $R_i$ is the radius of curvature of mirror $i$, not to be confused with reflectivity), the **stability condition** is:

$$\boxed{0 \leq g_1 g_2 \leq 1}$$

Common stable configurations:
- **Plane-parallel** ($R_1 = R_2 = \infty$): $g_1 g_2 = 1$ (marginally stable, hard to align)
- **Confocal** ($R_1 = R_2 = L$): $g_1 g_2 = 0$ (minimizes diffraction loss)
- **Hemispherical** ($R_1 = \infty, R_2 = L$): $g_1 g_2 = 0$ (easy to align)
- **Concentric** ($R_1 = R_2 = L/2$): $g_1 g_2 = 1$ (marginally stable, tight focus)

---

## 5. Laser Modes

### 5.1 Longitudinal Modes

Standing waves must fit within the cavity. The resonance condition is:

$$\nu_q = q\frac{c}{2nL}, \quad q = 1, 2, 3, \ldots$$

The **free spectral range (FSR)** — the frequency spacing between adjacent longitudinal modes — is:

$$\Delta\nu_{\text{FSR}} = \frac{c}{2nL}$$

For a 30 cm cavity with $n = 1$: $\Delta\nu_{\text{FSR}} = 500\,\text{MHz}$.

The laser oscillates on all longitudinal modes that fall within the gain bandwidth and exceed the threshold. For single-frequency operation, techniques like intracavity etalons or ring cavities are used.

### 5.2 Transverse Modes (TEM Modes)

The transverse intensity profile of the laser beam is described by **Hermite-Gaussian** (rectangular symmetry) or **Laguerre-Gaussian** (cylindrical symmetry) modes, denoted $\text{TEM}_{mn}$.

The fundamental mode $\text{TEM}_{00}$ has a Gaussian intensity profile:

$$I(r) = I_0 \exp\!\left(-\frac{2r^2}{w^2}\right)$$

where $w$ is the beam radius (at which intensity drops to $1/e^2$ of the peak). This mode has the lowest diffraction loss and the highest beam quality.

Higher-order modes ($\text{TEM}_{10}$, $\text{TEM}_{01}$, $\text{TEM}_{11}$, etc.) have nodes in the transverse profile and larger beam diameters. They are usually undesirable and are suppressed by using intracavity apertures.

### 5.3 The M² Beam Quality Factor

Real laser beams are characterized by the beam quality factor $M^2$:

$$M^2 = \frac{\theta_{\text{actual}} \cdot w_{0,\text{actual}}}{\theta_{\text{Gaussian}} \cdot w_{0,\text{Gaussian}}} \geq 1$$

- $M^2 = 1$: ideal $\text{TEM}_{00}$ Gaussian beam (diffraction-limited)
- $M^2 > 1$: beam diverges $M^2$ times faster than a diffraction-limited beam
- Typical values: HeNe $\approx 1.0$, Nd:YAG $\approx 1.1\text{-}1.3$, high-power diode bars $\approx 20\text{-}50$

---

## 6. Major Laser Types

### 6.1 Gas Lasers

**Helium-Neon (He-Ne)**:
- Wavelength: 632.8 nm (red), also 543.5 nm (green), 1152 nm (IR)
- Power: 0.5-50 mW (typical)
- Mechanism: Electrical discharge excites He atoms; energy transfer to Ne atoms via collisions (resonant energy levels)
- Properties: Excellent beam quality ($M^2 \approx 1$), long coherence length (~30 cm)
- Applications: alignment, interferometry, barcode scanners, teaching labs

**Carbon Dioxide (CO$_2$)**:
- Wavelength: 9.4 and 10.6 $\mu$m (mid-infrared)
- Power: up to tens of kW (industrial)
- Mechanism: Vibrational-rotational transitions of CO$_2$ molecules; N$_2$ used for resonant excitation
- Properties: Very high efficiency (~20%), high power
- Applications: industrial cutting/welding, surgery, LIDAR

**Excimer Lasers** (ArF 193 nm, KrF 248 nm, XeCl 308 nm):
- UV output, pulsed operation
- Applications: semiconductor lithography (ArF at 193 nm drives modern chip fabrication), eye surgery (LASIK)

### 6.2 Solid-State Lasers

**Nd:YAG (Neodymium-doped Yttrium Aluminum Garnet)**:
- Wavelength: 1064 nm (primary), frequency-doubled to 532 nm (green)
- Four-level system (low threshold)
- Pumped by flashlamps or diode lasers
- Applications: materials processing, medical surgery, range finding, scientific research

**Ti:Sapphire (Titanium-doped Sapphire)**:
- Wavelength: tunable from 650-1100 nm (broadest gain bandwidth of any laser)
- Pumped by green laser (usually frequency-doubled Nd:YAG)
- Enables ultrashort pulse generation (mode-locked, down to ~5 fs)
- Applications: ultrafast science, multiphoton microscopy, spectroscopy

### 6.3 Semiconductor Lasers (Laser Diodes)

- Based on stimulated emission in a p-n junction (direct bandgap semiconductors like GaAs, InGaAsP)
- Wavelength determined by bandgap: $\lambda = hc/E_g$
- Population inversion achieved by current injection (no external pump laser needed)
- Very compact (chip size), high efficiency (>50%), directly modulated at GHz rates
- Types: Fabry-Perot, DFB (distributed feedback), VCSEL (vertical-cavity surface-emitting)
- Applications: fiber optic communications, optical storage (CD/DVD/Blu-ray), laser printers, laser pointers

### 6.4 Fiber Lasers

- Gain medium is an optical fiber doped with rare-earth ions (Yb, Er, Tm)
- Very long gain path (meters), excellent thermal management (high surface-to-volume ratio)
- Inherently single-mode output, high beam quality
- Ytterbium fiber lasers: multi-kW CW output possible
- Applications: industrial cutting/welding (replacing CO$_2$ lasers), telecommunications, defense

### 6.5 Comparison Table

| Parameter | He-Ne | CO$_2$ | Nd:YAG | Ti:Sapph | Diode | Fiber |
|-----------|-------|--------|--------|----------|-------|-------|
| Wavelength | 632.8 nm | 10.6 $\mu$m | 1064 nm | 650-1100 nm | 0.4-2 $\mu$m | 1-2 $\mu$m |
| Power | mW | kW | W-kW | W | mW-W | W-kW |
| Efficiency | <0.1% | ~20% | ~3% | <0.1% | >50% | >30% |
| Beam quality $M^2$ | ~1 | 1-5 | 1-20 | ~1 | 1-50 | ~1 |
| Tunability | No | Limited | No | Broad | Limited | Limited |
| Key use | Metrology | Cutting | Material | Ultrafast | Telecom | Industrial |

---

## 7. Coherence Properties

### 7.1 Temporal Coherence

**Temporal coherence** quantifies how well a light wave maintains a predictable phase relationship over time. It is measured by the **coherence time** $\tau_c$ and **coherence length** $L_c$:

$$L_c = c\,\tau_c = \frac{c}{\Delta\nu}$$

where $\Delta\nu$ is the spectral linewidth. Lasers have very narrow linewidths, leading to long coherence lengths:

| Source | Linewidth $\Delta\nu$ | Coherence length $L_c$ |
|--------|----------------------|----------------------|
| White light | $\sim 3 \times 10^{14}$ Hz | $\sim 1$ $\mu$m |
| LED | $\sim 10^{13}$ Hz | $\sim 30$ $\mu$m |
| He-Ne (multimode) | $\sim 1.5$ GHz | $\sim 20$ cm |
| He-Ne (single-mode) | $\sim 1$ MHz | $\sim 300$ m |
| Stabilized laser | $\sim 1$ Hz | $\sim 3 \times 10^8$ m |

### 7.2 Spatial Coherence

**Spatial coherence** describes how well the phase is correlated across different transverse points in the beam at the same instant. A $\text{TEM}_{00}$ laser beam has essentially perfect spatial coherence across the entire beam cross-section.

Spatial coherence is what allows a laser beam to be focused to a diffraction-limited spot and to produce high-contrast interference patterns over large areas (essential for holography).

### 7.3 The van Cittert-Zernike Theorem

For partially coherent sources, the degree of spatial coherence is related to the source geometry through the **van Cittert-Zernike theorem**: the complex degree of coherence equals the normalized Fourier transform of the source intensity distribution. This theorem explains why even incoherent sources (like stars) can produce interference fringes at sufficient distance.

---

## 8. Gaussian Beam Propagation

### 8.1 The Paraxial Wave Equation

Starting from the Helmholtz equation and assuming slow variation of the amplitude envelope (paraxial approximation), we obtain:

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + 2ik\frac{\partial u}{\partial z} = 0$$

The fundamental solution is the **Gaussian beam**:

$$u(r,z) = \frac{w_0}{w(z)}\exp\!\left(-\frac{r^2}{w^2(z)}\right)\exp\!\left(-i\left[kz + \frac{kr^2}{2R(z)} - \psi(z)\right]\right)$$

### 8.2 Key Gaussian Beam Parameters

**Beam radius** $w(z)$:

$$w(z) = w_0\sqrt{1 + \left(\frac{z}{z_R}\right)^2}$$

**Rayleigh range** $z_R$ (distance from waist where the beam area doubles):

$$z_R = \frac{\pi w_0^2}{\lambda}$$

**Radius of curvature** $R(z)$:

$$R(z) = z\left[1 + \left(\frac{z_R}{z}\right)^2\right]$$

**Gouy phase** $\psi(z)$:

$$\psi(z) = \arctan\!\left(\frac{z}{z_R}\right)$$

**Far-field divergence half-angle**:

$$\theta = \frac{\lambda}{\pi w_0}$$

> **Analogy**: A Gaussian beam is like an hourglass made of light. The narrowest point (the waist $w_0$) is in the middle, and the beam expands in both directions, approaching a cone in the far field. The Rayleigh range $z_R$ is like the neck of the hourglass — the region where the beam stays roughly collimated. A tighter waist means a shorter neck but a wider opening angle, and vice versa. You can never have both a tiny waist and a tiny divergence — that is the uncertainty principle at work.

### 8.3 Important Properties

1. **Beam waist-divergence product** (invariant for Gaussian beams):

$$w_0 \cdot \theta = \frac{\lambda}{\pi}$$

This is the minimum possible value, achieved only by perfect $\text{TEM}_{00}$ beams. Real beams have $w_0 \theta = M^2 \lambda / \pi$.

2. **Depth of focus** (confocal parameter): $b = 2z_R = 2\pi w_0^2 / \lambda$

3. **Peak intensity at waist**: $I_0 = 2P/(\pi w_0^2)$ for total power $P$

4. **The beam stays collimated** (within a factor of $\sqrt{2}$ of the waist) over a distance $2z_R$.

---

## 9. ABCD Matrix Formalism

### 9.1 Ray Transfer Matrices

In the paraxial approximation, any optical element transforms a ray described by $(y, \theta)$ (height and angle) via:

$$\begin{pmatrix} y_{\text{out}} \\ \theta_{\text{out}} \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix}\begin{pmatrix} y_{\text{in}} \\ \theta_{\text{in}} \end{pmatrix}$$

Common ABCD matrices:

| Element | Matrix |
|---------|--------|
| Free space (length $d$) | $\begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}$ |
| Thin lens (focal length $f$) | $\begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$ |
| Curved mirror (radius $R$) | $\begin{pmatrix} 1 & 0 \\ -2/R & 1 \end{pmatrix}$ |
| Flat interface ($n_1 \to n_2$) | $\begin{pmatrix} 1 & 0 \\ 0 & n_1/n_2 \end{pmatrix}$ |

For a sequence of elements, multiply matrices from right to left: $M = M_N \cdots M_2 \cdot M_1$.

### 9.2 Gaussian Beam Transformation

The **complex beam parameter** $q(z)$ encodes both the radius of curvature and the beam radius:

$$\frac{1}{q(z)} = \frac{1}{R(z)} - i\frac{\lambda}{\pi w^2(z)}$$

At the beam waist ($z = 0$): $q_0 = iz_R$.

The transformation of $q$ through an ABCD system is:

$$\boxed{q_{\text{out}} = \frac{Aq_{\text{in}} + B}{Cq_{\text{in}} + D}}$$

This single formula tracks a Gaussian beam through any paraxial optical system. It is one of the most useful equations in laser optics.

### 9.3 Example: Focusing a Gaussian Beam with a Thin Lens

A Gaussian beam with waist $w_0$ (at the lens) passes through a thin lens of focal length $f$. What are the new waist size and location?

Input: $q_{\text{in}} = iz_R = i\pi w_0^2/\lambda$

Lens matrix: $A = 1, B = 0, C = -1/f, D = 1$

$$q_{\text{out}} = \frac{iz_R}{-iz_R/f + 1} = \frac{iz_R f}{f - iz_R}$$

After working through the algebra (multiplying numerator and denominator by the conjugate):

$$\frac{1}{q_{\text{out}}} = \frac{f - iz_R}{iz_R f} = \frac{1}{iz_R} - \frac{1}{f} + \cdots$$

The new waist radius (in the focal plane, for $f \gg z_R$):

$$w_0' \approx \frac{f\lambda}{\pi w_0}$$

A larger input beam ($w_0$) produces a smaller focused spot ($w_0'$) — this is why laser beams are typically expanded before focusing for applications like laser cutting or optical trapping.

---

## 10. Python Examples

### 10.1 Gaussian Beam Propagation

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_beam(z, w0, wavelength):
    """
    Calculate Gaussian beam parameters at position z.

    We compute all key quantities from just the waist size and wavelength.
    The Rayleigh range z_R sets the natural scale: for |z| < z_R the beam
    is quasi-collimated; for |z| >> z_R the beam diverges linearly.
    """
    z_R = np.pi * w0**2 / wavelength  # Rayleigh range: boundary between near and far field
    w = w0 * np.sqrt(1 + (z / z_R)**2)  # Beam radius expands hyperbolically
    R = np.where(
        np.abs(z) > 1e-15,
        z * (1 + (z_R / z)**2),  # Radius of curvature (infinite at waist, minimum at z_R)
        np.inf
    )
    gouy = np.arctan(z / z_R)  # Gouy phase: the pi phase shift through focus
    return w, R, gouy, z_R

# Parameters: a typical He-Ne laser beam
wavelength = 632.8e-9  # 632.8 nm
w0 = 0.5e-3  # 0.5 mm beam waist

z = np.linspace(-0.5, 0.5, 1000)  # propagation axis (meters)
w, R, gouy, z_R = gaussian_beam(z, w0, wavelength)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Beam envelope — the "hourglass" shape
axes[0].fill_between(z * 100, w * 1000, -w * 1000, alpha=0.3, color='red')
axes[0].plot(z * 100, w * 1000, 'r-', linewidth=2)
axes[0].plot(z * 100, -w * 1000, 'r-', linewidth=2)
axes[0].axhline(y=w0 * 1000, color='gray', linestyle='--', alpha=0.5,
                label=f'w₀ = {w0*1e3:.1f} mm')
axes[0].axvline(x=z_R * 100, color='blue', linestyle='--', alpha=0.5,
                label=f'z_R = {z_R*100:.1f} cm')
axes[0].axvline(x=-z_R * 100, color='blue', linestyle='--', alpha=0.5)
axes[0].set_ylabel('w(z) [mm]')
axes[0].set_title(f'Gaussian Beam Propagation (λ = {wavelength*1e9:.1f} nm, w₀ = {w0*1e3:.1f} mm)')
axes[0].legend()

# Radius of curvature
R_display = np.clip(R, -1e3, 1e3)  # Clip to avoid plotting infinity
axes[1].plot(z * 100, R_display, 'b-', linewidth=2)
axes[1].set_ylabel('R(z) [m]')
axes[1].set_ylim(-5, 5)
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Gouy phase
axes[2].plot(z * 100, np.degrees(gouy), 'g-', linewidth=2)
axes[2].set_ylabel('Gouy phase [deg]')
axes[2].set_xlabel('z [cm]')

plt.tight_layout()
plt.savefig('gaussian_beam_propagation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Rayleigh range: z_R = {z_R*100:.2f} cm")
print(f"Divergence half-angle: θ = {np.degrees(wavelength/(np.pi*w0)):.4f}°")
print(f"Depth of focus: b = 2z_R = {2*z_R*100:.2f} cm")
```

### 10.2 Cavity Stability Diagram

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_stability_diagram():
    """
    Plot the stability region for a two-mirror resonator.

    The stability parameters g1, g2 are defined as g_i = 1 - L/R_i.
    Stable cavities satisfy 0 <= g1*g2 <= 1, which defines the
    unshaded region on the plot. Common cavity configurations sit
    at special points on this diagram.
    """
    g1 = np.linspace(-2.5, 2.5, 500)
    g2 = np.linspace(-2.5, 2.5, 500)
    G1, G2 = np.meshgrid(g1, g2)

    # Stability condition: 0 <= g1*g2 <= 1
    stable = (G1 * G2 >= 0) & (G1 * G2 <= 1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Shade the UNSTABLE regions (so stable regions remain white)
    ax.contourf(G1, G2, ~stable, levels=[0.5, 1.5],
                colors=['lightcoral'], alpha=0.3)
    ax.contour(G1, G2, G1 * G2, levels=[0, 1], colors='black', linewidths=1.5)

    # Mark special configurations
    configs = {
        'Plane-parallel': (1, 1),
        'Confocal': (0, 0),
        'Concentric': (-1, -1),
        'Hemispherical': (0, 1),
    }
    for name, (x, y) in configs.items():
        ax.plot(x, y, 'ko', markersize=8)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('g₁ = 1 - L/R₁', fontsize=12)
    ax.set_ylabel('g₂ = 1 - L/R₂', fontsize=12)
    ax.set_title('Laser Cavity Stability Diagram', fontsize=14)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Add label in the stable region
    ax.text(0.5, 0.5, 'STABLE', fontsize=14, ha='center', va='center',
            fontweight='bold', color='green')
    ax.text(-1.5, 1.5, 'UNSTABLE', fontsize=14, ha='center', va='center',
            fontweight='bold', color='red', alpha=0.7)

    plt.tight_layout()
    plt.savefig('cavity_stability.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_stability_diagram()
```

### 10.3 ABCD Matrix Beam Propagation

```python
import numpy as np

def abcd_propagate(q_in, M):
    """
    Propagate a Gaussian beam through an ABCD matrix.

    The q-parameter transformation q_out = (A*q + B)/(C*q + D) is the
    master equation of Gaussian beam optics. It elegantly unifies
    geometric ray tracing with diffraction in a single formula.
    """
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    q_out = (A * q_in + B) / (C * q_in + D)
    return q_out

def q_to_params(q, wavelength):
    """Extract beam radius w and radius of curvature R from q-parameter."""
    inv_q = 1.0 / q
    R = 1.0 / np.real(inv_q) if np.abs(np.real(inv_q)) > 1e-15 else np.inf
    # The imaginary part of 1/q gives -lambda/(pi*w^2)
    w = np.sqrt(-wavelength / (np.pi * np.imag(inv_q)))
    return w, R

# --- Example: focus a collimated beam with a lens ---
wavelength = 1064e-9  # Nd:YAG laser
w0_input = 2e-3  # 2 mm input beam waist at the lens

z_R_input = np.pi * w0_input**2 / wavelength
q_input = 1j * z_R_input  # q at the beam waist (which is at the lens)

f = 0.1  # 100 mm focal length lens
M_lens = np.array([[1, 0], [-1/f, 1]])  # Thin lens matrix

q_after_lens = abcd_propagate(q_input, M_lens)

# Propagate through free space to find the focus
distances = np.linspace(0.09, 0.11, 1000)
beam_waists = []

for d in distances:
    M_space = np.array([[1, d], [0, 1]])
    q_final = abcd_propagate(q_after_lens, M_space)
    w, R = q_to_params(q_final, wavelength)
    beam_waists.append(w)

beam_waists = np.array(beam_waists)
min_idx = np.argmin(beam_waists)

print(f"Input beam waist: {w0_input*1e3:.2f} mm")
print(f"Focal length: {f*1e3:.1f} mm")
print(f"Focus position: {distances[min_idx]*1e3:.2f} mm from lens")
print(f"Focused waist: {beam_waists[min_idx]*1e6:.2f} µm")
print(f"Theoretical: {f*wavelength/(np.pi*w0_input)*1e6:.2f} µm")
```

---

## 11. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Stimulated emission | Incoming photon creates identical clone; basis of laser action |
| Einstein A/B relation | $A_{21} = (8\pi h\nu^3/c^3) B_{21}$ |
| Population inversion | $N_2/g_2 > N_1/g_1$; requires $\geq$ 3 levels |
| Threshold gain | $g_{\text{th}} = \alpha + \frac{1}{2L}\ln(1/R_1R_2)$ |
| Cavity stability | $0 \leq g_1g_2 \leq 1$ where $g_i = 1 - L/R_i$ |
| Longitudinal modes | $\nu_q = qc/(2nL)$; FSR $= c/(2nL)$ |
| Transverse modes | TEM$_{mn}$ (Hermite-Gaussian); TEM$_{00}$ is Gaussian |
| Coherence length | $L_c = c/\Delta\nu$ |
| Gaussian beam radius | $w(z) = w_0\sqrt{1+(z/z_R)^2}$ |
| Rayleigh range | $z_R = \pi w_0^2/\lambda$ |
| Divergence | $\theta = \lambda/(\pi w_0)$ |
| ABCD law | $q_{\text{out}} = (Aq_{\text{in}}+B)/(Cq_{\text{in}}+D)$ |
| $M^2$ factor | $w_0\theta = M^2\lambda/\pi$ ($M^2=1$ for ideal Gaussian) |

---

## 12. Exercises

### Exercise 1: Einstein Coefficients

A laser transition at $\lambda = 694.3\,\text{nm}$ (ruby laser) has a spontaneous lifetime of $\tau_{\text{sp}} = 3\,\text{ms}$.

(a) Calculate the Einstein $A$ coefficient.
(b) Calculate the Einstein $B$ coefficient.
(c) At what temperature would the rates of spontaneous and stimulated emission be equal for blackbody radiation? What does this tell you about building a thermal-equilibrium laser?

### Exercise 2: Cavity Design

You want to build a Nd:YAG laser with a cavity length $L = 20\,\text{cm}$. The gain medium has a small-signal gain coefficient $g_0 = 0.5\,\text{cm}^{-1}$ over a crystal length of 5 cm, and the internal loss is $\alpha = 0.01\,\text{cm}^{-1}$.

(a) If $R_1 = 1.0$ (100% reflector), what is the minimum $R_2$ for lasing?
(b) Calculate the FSR.
(c) Design a stable cavity using one flat mirror and one curved mirror. What radius of curvature should the curved mirror have?

### Exercise 3: Gaussian Beam

A He-Ne laser emits a TEM$_{00}$ beam with $w_0 = 0.4\,\text{mm}$ at $\lambda = 632.8\,\text{nm}$.

(a) Calculate the Rayleigh range and depth of focus.
(b) What is the beam diameter at 1 m from the waist? At 10 m?
(c) The beam is focused by a lens with $f = 50\,\text{mm}$, placed at the waist. Calculate the focused spot size.
(d) Write a Python script to plot the beam radius from $z = 0$ to $z = 5\,\text{m}$.

### Exercise 4: Laser Comparison

Create a table comparing He-Ne, CO$_2$, Nd:YAG, Ti:Sapphire, and semiconductor diode lasers across the following parameters: wavelength, pumping mechanism, efficiency, typical power, coherence length, and primary application. For each laser, explain the physical reason it emits at its particular wavelength.

### Exercise 5: ABCD Matrix Cascade

A Gaussian beam ($w_0 = 1\,\text{mm}$, $\lambda = 1064\,\text{nm}$) passes through the following sequence:
1. Free space propagation of 30 cm
2. Thin lens with $f = 20\,\text{cm}$
3. Free space propagation of 20 cm

Use Python to compute the final beam radius and radius of curvature. Is the beam converging, diverging, or at a waist?

---

## 13. References

1. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapters 15-17.
2. Siegman, A. E. (1986). *Lasers*. University Science Books. — The definitive laser physics textbook.
3. Svelto, O. (2010). *Principles of Lasers* (5th ed.). Springer.
4. Milonni, P. W., & Eberly, J. H. (2010). *Laser Physics*. Wiley.
5. Einstein, A. (1917). "Zur Quantentheorie der Strahlung." *Physikalische Zeitschrift*, 18, 121-128.

---

[← Previous: 07. Polarization](07_Polarization.md) | [Next: 09. Fiber Optics →](09_Fiber_Optics.md)

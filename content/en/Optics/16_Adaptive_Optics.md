# 16. Adaptive Optics

[← Previous: 15. Zernike Polynomials](15_Zernike_Polynomials.md) | [Next: 17. Spectroscopy →](17_Spectroscopy.md)

---

In 1990, the Hubble Space Telescope was launched with a primary mirror polished to the wrong shape — 2.2 micrometers of spherical aberration, imperceptible to the eye but devastating for astronomical imaging. The $1.5 billion instrument was nearly blind. Three years later, astronauts installed COSTAR, a set of corrective optics that restored Hubble's vision to diffraction-limited perfection. This dramatic fix was possible because the aberration was static and known. But what about ground-based telescopes, where the atmosphere scrambles the wavefront hundreds of times per second with random, ever-changing distortions?

**Adaptive optics** (AO) is the technology that answers this challenge. First proposed by Horace Babcock in 1953, AO systems measure atmospheric wavefront distortions in real time using a wavefront sensor, compute the correction using a fast reconstructor, and apply it using a deformable mirror — all within a few milliseconds. The result: ground-based telescopes with diameters of 8–10 meters can achieve resolution approaching the diffraction limit, rivaling or surpassing Hubble for certain observations.

This lesson covers the complete AO system from atmospheric turbulence characterization through wavefront sensing, reconstruction, and correction, building on the Zernike polynomial foundation from Lesson 15.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Explain why atmospheric turbulence limits ground-based telescope resolution and quantify the seeing limit using the Fried parameter
2. Describe the key atmospheric parameters (Fried parameter $r_0$, Greenwood frequency $f_G$, isoplanatic angle $\theta_0$) and their dependence on wavelength and zenith angle
3. Draw and explain the block diagram of a classical AO system: guide star, wavefront sensor, reconstructor, deformable mirror, science camera
4. Describe the operating principles of Shack-Hartmann, curvature, and pyramid wavefront sensors
5. Derive the least-squares wavefront reconstruction from slope measurements and explain zonal vs. modal approaches
6. Explain deformable mirror technologies, influence functions, and actuator-to-wavefront mapping
7. Analyze closed-loop AO control including integrator gain, temporal error, and the AO error budget
8. Describe advanced AO concepts: laser guide stars, multi-conjugate AO, and extreme AO for exoplanet detection

---

## Table of Contents

1. [Why Adaptive Optics?](#1-why-adaptive-optics)
2. [Atmospheric Turbulence Parameters](#2-atmospheric-turbulence-parameters)
3. [AO System Architecture](#3-ao-system-architecture)
4. [Wavefront Sensors](#4-wavefront-sensors)
5. [Wavefront Reconstruction](#5-wavefront-reconstruction)
6. [Deformable Mirrors](#6-deformable-mirrors)
7. [Closed-Loop Control](#7-closed-loop-control)
8. [Performance Metrics and Error Budget](#8-performance-metrics-and-error-budget)
9. [Advanced AO Concepts](#9-advanced-ao-concepts)
10. [Python Examples](#10-python-examples)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Why Adaptive Optics?

### 1.1 The Seeing Limit

A perfect telescope of diameter $D$ should resolve angles as small as:

$$\theta_{\text{diff}} = 1.22 \frac{\lambda}{D}$$

For an 8-meter telescope at $\lambda = 500$ nm, this gives $\theta_{\text{diff}} \approx 0.016$ arcseconds — sharp enough to read a newspaper headline from 30 km away. But atmospheric turbulence typically limits the resolution to:

$$\theta_{\text{seeing}} \approx 0.98 \frac{\lambda}{r_0} \approx 0.5\text{--}2\,\text{arcsec}$$

For $r_0 = 10$ cm at 500 nm, the seeing is about 1 arcsecond — **50 times worse** than the diffraction limit. The entire investment in a large primary mirror is wasted unless the atmospheric distortion is corrected.

### 1.2 Short Exposures vs. Long Exposures

In a single short exposure ($\sim 10$ ms), the atmospheric wavefront is roughly "frozen" and the image consists of a speckle pattern — a random collection of bright spots, each the size of the diffraction limit. Over a long exposure (seconds to minutes), these speckles average to produce the familiar seeing-limited blob.

> **Analogy**: Imagine looking at a coin at the bottom of a swimming pool. The ripples on the water surface distort the image — the coin appears to dance and shimmer. If you could freeze the water surface and then push it flat with a flexible mold, the coin would appear sharp and still. Adaptive optics does exactly this: it measures the "ripples" (atmospheric turbulence) and flattens them with a deformable mirror, hundreds of times per second.

### 1.3 Historical Development

| Year | Milestone |
|------|-----------|
| 1953 | Babcock proposes AO concept |
| 1970s | US military develops classified AO for satellite imaging |
| 1989 | ESO COME-ON: first astronomical AO system |
| 1994 | Laser guide star demonstrated |
| 2002 | Keck AO operational with laser guide star |
| 2010s | Extreme AO systems (GPI, SPHERE) image exoplanets directly |
| 2020s | ELT (39 m), TMT (30 m), GMT (25 m) all designed around MCAO |

---

## 2. Atmospheric Turbulence Parameters

### 2.1 Fried Parameter $r_0$

The Fried parameter (introduced in Lesson 15, §7.2) is the single most important number characterizing atmospheric turbulence. It represents the effective coherent aperture diameter:

$$r_0 = \left[0.423 k^2 \sec\gamma \int_0^\infty C_n^2(h)\,dh\right]^{-3/5}$$

Key scaling laws:

$$r_0 \propto \lambda^{6/5} \qquad r_0 \propto (\cos\gamma)^{3/5}$$

At a good site, $r_0 \approx 15$ cm at 500 nm. At infrared wavelengths ($K$-band, 2.2 $\mu$m), $r_0$ scales up to ~75 cm, making AO correction much easier.

### 2.2 Greenwood Frequency $f_G$

The atmospheric turbulence is not static — wind blows the turbulent layers across the telescope aperture. The **Greenwood frequency** characterizes the temporal bandwidth needed for AO correction:

$$f_G = 0.427 \frac{v_{\text{eff}}}{r_0}$$

where $v_{\text{eff}}$ is the effective wind speed (typically the wind at the strongest turbulence layer). Typical values are $f_G \approx 20$–$50$ Hz at visible wavelengths. The AO system must operate at a closed-loop bandwidth of at least $f_G$ to keep up with the atmosphere.

More precisely, the $-3\,\text{dB}$ bandwidth of the AO system should be:

$$f_{3\text{dB}} \gtrsim f_G$$

If the bandwidth falls below $f_G$, the temporal error dominates the error budget.

### 2.3 Isoplanatic Angle $\theta_0$

The AO correction is valid only within a limited field of view around the guide star. The **isoplanatic angle** is the angular radius over which the wavefront is correlated:

$$\theta_0 = 0.314 \frac{r_0}{\bar{h}}$$

where $\bar{h}$ is the effective height of the turbulence (weighted by $C_n^2$). Typical values: $\theta_0 \approx 2$–$5$ arcseconds at visible, $10$–$20$ arcseconds at $K$-band. The science target must lie within $\theta_0$ of the guide star for effective correction.

### 2.4 $C_n^2$ Profile

The refractive index structure constant $C_n^2(h)$ varies with altitude. A typical profile has:

- **Surface layer** (0–1 km): Strong turbulence from ground heating
- **Free atmosphere** (1–10 km): Weak, smooth turbulence
- **Tropopause** (~10–12 km): Strong shear layer (jet stream)

| Layer | Typical $C_n^2$ (m$^{-2/3}$) | Contribution to Seeing |
|-------|:---------------------------:|----------------------|
| Ground (0–500 m) | $10^{-14}$ – $10^{-13}$ | 50–80% |
| Free atmosphere | $10^{-17}$ – $10^{-16}$ | 10–30% |
| Tropopause | $10^{-16}$ – $10^{-15}$ | 10–30% |

The multi-layer structure of turbulence is crucial for advanced AO concepts like multi-conjugate AO (MCAO), which places deformable mirrors conjugate to the dominant turbulence layers.

---

## 3. AO System Architecture

### 3.1 Block Diagram

A classical single-conjugate AO system consists of four main components in a feedback loop:

```
                    ┌──────────────┐
                    │  Guide Star  │
                    └──────┬───────┘
                           │ turbulent wavefront
                    ┌──────▼───────┐
                    │  Deformable  │◄─── DM commands
                    │    Mirror    │
                    └──────┬───────┘
                           │ corrected wavefront
              ┌────────────┼────────────┐
              │                         │
       ┌──────▼───────┐         ┌──────▼───────┐
       │   Wavefront   │         │   Science    │
       │    Sensor     │         │   Camera     │
       └──────┬────────┘         └──────────────┘
              │ slope measurements
       ┌──────▼────────┐
       │ Reconstructor │
       │  & Controller │
       └──────┬────────┘
              │ DM commands
              └─────────────────────────────────►
```

The light from a guide star (natural or laser) passes through the atmosphere, is reflected off the deformable mirror, and is split between the wavefront sensor and the science camera. The wavefront sensor measures the residual aberration, the reconstructor computes the correction, and the deformable mirror applies it — closing the loop at hundreds of Hz.

### 3.2 Optical Path

The beam from the telescope is first reflected off the **deformable mirror** (DM), which applies the correction. A **dichroic beamsplitter** then separates the light:
- The guide star wavelength goes to the **wavefront sensor** (WFS)
- The science wavelength passes to the **science camera**

This arrangement ensures the WFS sees the residual wavefront error *after* the DM correction, enabling a closed-loop feedback system.

### 3.3 Timing Requirements

The closed-loop delay from measurement to correction must be shorter than the atmospheric coherence time:

$$\tau_0 = 0.314 \frac{r_0}{v_{\text{eff}}} = \frac{0.134}{f_G}$$

At $f_G = 30$ Hz, $\tau_0 \approx 4.5$ ms. The total loop delay (CCD readout + computation + DM settling) must be well below this. Modern AO systems achieve loop rates of 500–2000 Hz.

---

## 4. Wavefront Sensors

### 4.1 Shack-Hartmann Sensor

The most widely used WFS in astronomy. A **microlens array** (lenslet array) divides the pupil into subapertures (typically $d \sim r_0$ in size). Each lenslet forms an image of the guide star; the displacement of this spot from its reference position is proportional to the average wavefront slope across that subaperture:

$$s_x = \frac{\partial W}{\partial x}\bigg|_{\text{avg}}, \qquad s_y = \frac{\partial W}{\partial y}\bigg|_{\text{avg}}$$

**Centroiding algorithms** measure the spot displacement:

**Center of gravity (CoG)**:
$$s_x = \frac{\sum_i x_i I_i}{\sum_i I_i}, \qquad s_y = \frac{\sum_i y_i I_i}{\sum_i I_i}$$

where $I_i$ is the intensity at pixel $i$. This is simple and fast but sensitive to background noise.

**Thresholded CoG**: Set pixels below a threshold to zero before computing CoG. Reduces noise but introduces bias.

**Correlation**: Cross-correlate each subaperture image with a reference PSF. More robust to noise but computationally expensive.

| Parameter | Typical Value |
|-----------|:-------------:|
| Subaperture size | $r_0$ – $2r_0$ |
| Number of subapertures | 10×10 to 80×80 |
| Pixels per subaperture | 4×4 to 16×16 |
| Frame rate | 500–3000 Hz |

### 4.2 Curvature Sensor

Proposed by Roddier (1988), the curvature sensor measures the intensity difference between two out-of-focus planes:

$$\frac{I_1(\mathbf{r}) - I_2(\mathbf{r})}{I_1(\mathbf{r}) + I_2(\mathbf{r})} \propto \nabla^2 W(\mathbf{r}) + \frac{\partial W}{\partial n}\bigg|_{\text{edge}}$$

where $I_1$ and $I_2$ are the intra-focal and extra-focal intensities, and the edge term accounts for wavefront slopes at the pupil boundary.

**Advantages**: Directly measures the Laplacian of the wavefront (second derivative), which maps naturally to bimorph deformable mirrors. Simpler optics than Shack-Hartmann.

**Disadvantages**: Requires a vibrating membrane mirror to oscillate between focus planes; lower spatial resolution.

### 4.3 Pyramid Sensor

Developed by Ragazzoni (1996), the pyramid sensor uses a glass pyramid placed at the focal plane to split the beam into four copies of the pupil image:

$$s_x \propto \frac{(I_1 + I_2) - (I_3 + I_4)}{I_1 + I_2 + I_3 + I_4}$$

The pyramid can be modulated (oscillated around the focus) to adjust the sensitivity range — a unique advantage over other sensors.

**Advantages**: Higher sensitivity than Shack-Hartmann for bright guide stars; adjustable dynamic range via modulation.

**Disadvantages**: Nonlinear response at large aberrations; requires modulation for extended range.

### 4.4 Comparison

| Property | Shack-Hartmann | Curvature | Pyramid |
|----------|:--------------:|:---------:|:-------:|
| Measured quantity | Slope ($\nabla W$) | Laplacian ($\nabla^2 W$) | Slope ($\nabla W$) |
| Linearity | Linear to $\pm \lambda/2$ per subaperture | Linear | Linear with modulation |
| Sensitivity (bright) | Moderate | Moderate | High |
| Sensitivity (faint) | Good | Good | Moderate |
| Spatial sampling | Fixed by lenslet | Fixed by defocus distance | Adjustable (modulation) |
| Complexity | Simple, robust | Moderate | Complex optics |
| Use cases | Most AO systems | Early AO systems, ESO | ELT (MAORY), TMT |

---

## 5. Wavefront Reconstruction

### 5.1 The Reconstruction Problem

Given slope measurements $\mathbf{s} = [s_{x,1}, s_{y,1}, s_{x,2}, s_{y,2}, \ldots]^T$ from the WFS, we need to estimate the wavefront $W(\mathbf{r})$ or equivalently the DM commands $\mathbf{c}$ that will flatten it.

### 5.2 Zonal Reconstruction

In the **zonal** approach, the wavefront is represented as a grid of phase values at each subaperture center. The slope between adjacent grid points is:

$$s_x \approx \frac{W_{i+1,j} - W_{i,j}}{\Delta x}$$

This gives a sparse linear system $\mathbf{s} = \mathbf{G}\mathbf{w}$, where $\mathbf{G}$ is the **geometry matrix** (sparse, with entries $\pm 1/\Delta x$). The least-squares solution is:

$$\hat{\mathbf{w}} = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{s}$$

The matrix $\mathbf{G}^T\mathbf{G}$ is a discrete Laplacian, so reconstruction can be done efficiently using Fourier methods or iterative solvers (Gauss-Seidel, conjugate gradient).

### 5.3 Modal Reconstruction

In the **modal** approach, the wavefront is expanded in Zernike (or other) modes:

$$W(\rho, \theta) = \sum_{j=1}^{J} a_j Z_j(\rho, \theta)$$

The slopes at each subaperture are linear functions of the coefficients:

$$\mathbf{s} = \mathbf{D}\mathbf{a}$$

where $\mathbf{D}$ is the **interaction matrix** (derivative of each Zernike mode at each subaperture). The least-squares reconstructor is:

$$\hat{\mathbf{a}} = \mathbf{D}^+ \mathbf{s} = (\mathbf{D}^T\mathbf{D})^{-1}\mathbf{D}^T\mathbf{s}$$

**Advantages**: Modes can be filtered (e.g., exclude piston); natural connection to Kolmogorov statistics; noise on high-order modes can be suppressed. **Disadvantages**: Requires computing Zernike derivatives; limited by the number of modes chosen.

### 5.4 The Interaction Matrix (Calibration)

In practice, $\mathbf{D}$ is measured empirically:

1. Apply each DM actuator one at a time (or each mode)
2. Record the WFS response $\mathbf{s}_j$ for each command $\mathbf{c}_j$
3. The interaction matrix is $\mathbf{M} = [\mathbf{s}_1 | \mathbf{s}_2 | \ldots]$
4. The command matrix (reconstructor) is $\mathbf{R} = \mathbf{M}^+$ (pseudo-inverse via SVD)

This calibration process accounts for all real-world effects: WFS geometry, DM influence functions, optical misalignment.

### 5.5 Noise Propagation

The reconstructed wavefront amplifies WFS measurement noise. The **noise propagation coefficient** $\eta$ is defined as:

$$\sigma_{\text{recon}}^2 = \eta \, \sigma_{\text{meas}}^2$$

For zonal reconstructors, $\eta$ depends on the geometry and is typically 0.2–1.0. For modal reconstructors, noise propagation increases with mode number, motivating mode truncation at a level where the atmospheric signal drops below the noise floor.

---

## 6. Deformable Mirrors

### 6.1 Types of Deformable Mirrors

**Segmented DM**: An array of flat mirror segments, each with piston, tip, and tilt actuators. Used in segmented-mirror telescopes (Keck, ELT). Gap diffraction is a disadvantage.

**Continuous facesheet DM**: A thin reflective membrane supported by an array of actuators underneath. The most common type for astronomical AO. Produces a smooth continuous correction.

**Bimorph DM**: Two piezoelectric layers bonded together; applying voltage bends the mirror. Natural match for curvature sensors (both deal with the Laplacian of the wavefront).

**MEMS DM**: Micro-electro-mechanical systems with thousands of actuators on a chip-scale mirror. Used in extreme AO and laboratory systems.

### 6.2 Influence Functions

The **influence function** $\phi_k(\mathbf{r})$ describes the mirror surface shape when actuator $k$ is pushed by a unit command:

$$W_{\text{DM}}(\mathbf{r}) = \sum_{k=1}^{K} c_k \phi_k(\mathbf{r})$$

For continuous facesheet DMs, the influence function is approximately Gaussian:

$$\phi_k(\mathbf{r}) \approx \exp\left(-\ln 2 \frac{|\mathbf{r} - \mathbf{r}_k|^2}{w^2}\right)$$

where $w$ is the influence function width (related to the inter-actuator coupling). The **coupling** is the fraction of stroke felt by neighboring actuators — typically 10–15% for piezoelectric stack DMs.

### 6.3 Key DM Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|:--------------:|
| Number of actuators | Spatial DOF | 100–10,000 |
| Actuator pitch | Inter-actuator spacing | 0.3–10 mm |
| Stroke | Maximum surface displacement | 2–10 $\mu$m |
| Coupling | Neighbor response fraction | 10–15% |
| Bandwidth | Mechanical resonance limit | 1–10 kHz |
| Flatness | RMS residual with all actuators at zero | 5–30 nm |

### 6.4 Fitting Error

The DM cannot reproduce wavefront features smaller than the actuator spacing $d$. The **fitting error** is the dominant error term in many AO systems:

$$\sigma_{\text{fit}}^2 = \alpha \left(\frac{d}{r_0}\right)^{5/3}$$

where $\alpha \approx 0.23$ for continuous facesheet DMs and $\alpha \approx 0.28$ for segmented DMs. To achieve a Strehl ratio $S$ at wavelength $\lambda$, the actuator spacing must satisfy:

$$d \lesssim r_0 \left(\frac{-\ln S}{(2\pi \alpha)^{2}}\right)^{3/10}$$

---

## 7. Closed-Loop Control

### 7.1 The AO Temporal Loop

At each time step $t_k$:

1. **Measure**: WFS reads slopes $\mathbf{s}_k$ from the residual wavefront
2. **Reconstruct**: Compute residual $\hat{\mathbf{w}}_k = \mathbf{R} \mathbf{s}_k$ using the command matrix
3. **Control**: Update DM commands $\mathbf{c}_{k+1} = \mathbf{c}_k + g \hat{\mathbf{w}}_k$
4. **Apply**: Send commands to the DM

This is an **integrator** controller with gain $g \in (0, 1]$.

### 7.2 Integrator Control

The integrator accumulates corrections over time, which is appropriate because the wavefront error is slowly varying compared to the loop rate. The transfer function of the integrator is:

$$H_{\text{rej}}(f) = \frac{1}{1 + g \frac{f_s}{2\pi i f} e^{-i2\pi f \tau}}$$

where $f_s$ is the loop frequency and $\tau$ is the total delay (typically 1–2 frames). The rejection bandwidth (where $|H_{\text{rej}}| = 0.5$) is:

$$f_{-3\text{dB}} \approx \frac{g f_s}{2\pi}$$

**Gain selection**: Too low gain gives poor bandwidth (temporal error). Too high gain amplifies noise (noise error) and can drive the loop unstable. The optimal gain balances these:

$$g_{\text{opt}} = 1 - \exp(-2\pi f_G \tau)$$

### 7.3 Temporal Error

When the AO bandwidth is insufficient to track the atmosphere, the residual temporal error is:

$$\sigma_{\text{temp}}^2 = \left(\frac{f_G}{f_{-3\text{dB}}}\right)^{5/3}$$

This is the dominant error term for systems that are "bandwidth-starved" (e.g., visible-light AO with high Greenwood frequency).

### 7.4 Advanced Controllers

The simple integrator can be improved with more sophisticated control laws:

- **Proportional-Integral (PI)**: Adds a proportional term for faster response
- **Linear Quadratic Gaussian (LQG)**: Optimal controller using a Kalman filter to predict the wavefront based on temporal and spatial correlations
- **Predictive control**: Uses wind speed and direction to anticipate the wavefront evolution

---

## 8. Performance Metrics and Error Budget

### 8.1 Strehl Ratio

The Strehl ratio is the primary metric for AO performance:

$$S = \frac{I_{\text{peak}}}{I_{\text{diffraction limit}}} \approx \exp\left[-(2\pi\sigma_{\text{total}})^2\right]$$

where $\sigma_{\text{total}}$ is the total residual wavefront error in waves.

### 8.2 Error Budget

The total wavefront variance is the sum of independent error terms:

$$\sigma_{\text{total}}^2 = \sigma_{\text{fit}}^2 + \sigma_{\text{temp}}^2 + \sigma_{\text{noise}}^2 + \sigma_{\text{alias}}^2 + \sigma_{\text{aniso}}^2 + \sigma_{\text{other}}^2$$

| Error Term | Source | Formula |
|------------|--------|---------|
| Fitting | DM actuator density | $\alpha (d/r_0)^{5/3}$ |
| Temporal | Bandwidth vs. Greenwood freq | $(f_G / f_{-3\text{dB}})^{5/3}$ |
| Noise | WFS photon + readout noise | $\propto 1/\text{SNR}^2$ |
| Aliasing | High-order modes folded to low-order | $\approx 0.08 (d/r_0)^{5/3}$ |
| Anisoplanatism | Angular separation from guide star | $(\theta/\theta_0)^{5/3}$ |

> **Analogy**: Think of the AO error budget as a chain with multiple links. The overall performance (Strehl ratio) is limited by the weakest link. A system with an excellent DM (small fitting error) but a slow control loop (large temporal error) will still perform poorly. Good AO engineering means balancing all error terms so that no single source dominates.

### 8.3 Sky Coverage

Natural guide star (NGS) AO requires a sufficiently bright star within the isoplanatic angle. The probability of finding such a star is the **sky coverage**. For a limiting magnitude of $m_V = 14$ and $\theta_0 = 5''$:

$$\text{Sky coverage} \approx 1\%$$

This extremely low coverage motivates laser guide stars (see §9).

---

## 9. Advanced AO Concepts

### 9.1 Laser Guide Stars (LGS)

Since bright natural guide stars are rare, AO systems create artificial beacons using lasers:

**Rayleigh LGS**: A pulsed laser (typically 532 nm) scattered by molecules at 10–20 km altitude. Time-gated detection isolates the desired altitude. Limited by the **cone effect** — the laser beam samples a cone, not a cylinder, so high-altitude turbulence is under-sampled.

**Sodium LGS**: A CW or pulsed laser tuned to the sodium D2 line (589 nm) excites the mesospheric sodium layer at ~90 km altitude. Higher beacon altitude reduces the cone effect but requires more expensive laser technology.

**Limitations**:
- LGS cannot measure tip-tilt (the laser goes up and comes back down through the same atmosphere — tilt is reciprocal and cancels). A dim NGS is still needed for tip-tilt.
- Cone effect (focus anisoplanatism) limits the correctable field for single-LGS systems.

### 9.2 Multi-Conjugate AO (MCAO)

MCAO uses multiple guide stars and multiple deformable mirrors, each conjugate to a different turbulence layer, to correct a wider field of view:

```
Atmosphere:  Layer 1 (ground)    Layer 2 (high)
                │                    │
DM1 ◄──────────┘                    │
(conjugate to ground)                │
                                     │
DM2 ◄───────────────────────────────┘
(conjugate to 10 km)
```

By using 3–5 laser guide stars spread across the field and 2–3 DMs, MCAO achieves uniform correction over a field 1–2 arcminutes in diameter — much larger than the isoplanatic angle.

**Example**: Gemini South's GeMS system uses 5 sodium LGS and 2 DMs to correct a 85-arcsecond field.

### 9.3 Ground-Layer AO (GLAO)

GLAO corrects only the ground-layer turbulence (which contributes 50–80% of the total seeing) using a single DM conjugate to the ground. This provides a modest improvement (50% seeing reduction) but over a very wide field (arcminutes to degrees).

### 9.4 Extreme AO (ExAO)

Extreme AO systems are designed for high-contrast imaging of exoplanets and circumstellar disks. They feature:

- Very high actuator count (3000–10,000)
- Very fast loop rates (1–3 kHz)
- Coronagraphs to suppress starlight
- Post-processing speckle subtraction (ADI, SDI)
- Target Strehl > 90% in the near-infrared

**Examples**: Gemini Planet Imager (GPI), VLT/SPHERE, Subaru/SCExAO.

### 9.5 AO for the Extremely Large Telescopes

The next-generation ELTs (25–39 m diameter) are designed around AO from the start:

| Telescope | Diameter | AO System | DM Actuators | LGS |
|-----------|:--------:|-----------|:------------:|:---:|
| ELT (ESO) | 39 m | MAORY (MCAO) | ~5000 (M4) | 6 sodium |
| TMT | 30 m | NFIRAOS (MCAO) | ~5000 | 6 sodium |
| GMT | 25 m | LTAO/GLAO | ~3500 per segment | 6 sodium |

---

## 10. Python Examples

### 10.1 Shack-Hartmann Spot Simulation

```python
import numpy as np

def shack_hartmann_spots(wavefront: np.ndarray, n_sub: int,
                         pupil_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Shack-Hartmann spot positions from a wavefront.

    The wavefront is divided into n_sub × n_sub subapertures. For each
    subaperture, the average slope is computed from finite differences.

    Parameters
    ----------
    wavefront : 2D array  — Wavefront phase (radians) on a square grid
    n_sub : int  — Number of subapertures across the pupil diameter
    pupil_radius : float  — Physical radius of the pupil (same units as wavefront grid)

    Returns
    -------
    sx, sy : arrays of shape (n_sub, n_sub)  — Slope measurements (rad/m)
    """
    N = wavefront.shape[0]
    sub_size = N // n_sub
    dx = 2 * pupil_radius / N  # pixel size in physical units

    sx = np.zeros((n_sub, n_sub))
    sy = np.zeros((n_sub, n_sub))

    for i in range(n_sub):
        for j in range(n_sub):
            # Extract subaperture
            sub = wavefront[i*sub_size:(i+1)*sub_size,
                            j*sub_size:(j+1)*sub_size]
            # Average x-slope from finite differences
            dWdx = np.diff(sub, axis=1) / dx
            sx[i, j] = np.mean(dWdx)
            # Average y-slope
            dWdy = np.diff(sub, axis=0) / dx
            sy[i, j] = np.mean(dWdy)

    return sx, sy
```

### 10.2 Influence Matrix and Reconstructor

```python
def build_influence_matrix(n_act: int, n_sub: int,
                           coupling: float = 0.15) -> np.ndarray:
    """Build a simplified influence matrix for a DM with Gaussian
    influence functions, sensed by a Shack-Hartmann WFS.

    The influence matrix M maps DM commands to WFS slopes:
        s = M @ c

    The command matrix (reconstructor) is R = pinv(M).

    Parameters
    ----------
    n_act : int  — Number of actuators across the pupil
    n_sub : int  — Number of WFS subapertures across the pupil
    coupling : float  — Inter-actuator coupling (0 to 1)

    Returns
    -------
    M : 2D array of shape (2*n_sub^2, n_act^2)  — Influence matrix
    """
    # Actuator positions (normalized to [0, 1])
    act_pos = np.linspace(0, 1, n_act)
    ax, ay = np.meshgrid(act_pos, act_pos)
    act_x = ax.ravel()
    act_y = ay.ravel()

    # Subaperture center positions
    sub_pos = np.linspace(0.5/n_sub, 1 - 0.5/n_sub, n_sub)
    sx_grid, sy_grid = np.meshgrid(sub_pos, sub_pos)
    sub_x = sx_grid.ravel()
    sub_y = sy_grid.ravel()

    n_slopes = 2 * len(sub_x)
    n_actuators = len(act_x)
    M = np.zeros((n_slopes, n_actuators))

    # Width of influence function from coupling
    # coupling = exp(-ln2 * (pitch/w)^2), so w = pitch / sqrt(-ln(coupling)/ln2)
    pitch = 1.0 / (n_act - 1) if n_act > 1 else 1.0
    w = pitch / np.sqrt(-np.log(coupling) / np.log(2))

    for k in range(n_actuators):
        # Distance from subaperture centers to actuator k
        dx = sub_x - act_x[k]
        dy = sub_y - act_y[k]
        r2 = dx**2 + dy**2
        # Gaussian influence
        phi = np.exp(-np.log(2) * r2 / w**2)
        # Slopes are derivatives of the influence function
        # d(phi)/dx = phi * (-2*ln2*dx/w^2)
        dphi_dx = phi * (-2 * np.log(2) * dx / w**2)
        dphi_dy = phi * (-2 * np.log(2) * dy / w**2)
        M[:len(sub_x), k] = dphi_dx
        M[len(sub_x):, k] = dphi_dy

    return M
```

### 10.3 Closed-Loop AO Simulation

```python
def ao_closed_loop(phase_screens: list[np.ndarray],
                   n_sub: int, n_act: int,
                   gain: float = 0.5,
                   coupling: float = 0.15,
                   pupil_radius: float = 1.0) -> dict:
    """Run a closed-loop AO simulation over a sequence of phase screens.

    Each phase screen represents the atmospheric wavefront at one time step.
    The AO system measures slopes, reconstructs, and applies correction.

    Returns a dict with residual wavefronts and Strehl ratios per step.
    """
    N = phase_screens[0].shape[0]

    # Build interaction matrix and reconstructor
    M = build_influence_matrix(n_act, n_sub, coupling)
    # Pseudo-inverse via SVD (truncate small singular values)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_inv = np.where(s > 0.01 * s[0], 1.0 / s, 0.0)
    R = (Vt.T * s_inv) @ U.T  # command matrix

    # Initialize DM commands
    dm_commands = np.zeros(n_act * n_act)
    strehls = []
    rms_list = []

    for screen in phase_screens:
        # Apply current DM correction
        # Reconstruct DM surface from commands (simplified: nearest-neighbor)
        dm_surface = _commands_to_surface(dm_commands, n_act, N, coupling)
        residual = screen - dm_surface

        # Measure slopes
        sx, sy = shack_hartmann_spots(residual, n_sub, pupil_radius)
        slopes = np.concatenate([sx.ravel(), sy.ravel()])

        # Reconstruct and update commands
        correction = R @ slopes
        dm_commands += gain * correction

        # Compute metrics
        pupil_mask = _circular_mask(N)
        rms = np.std(residual[pupil_mask])
        strehl = np.exp(-(2 * np.pi * rms)**2) if rms < 0.5 else 0.0
        strehls.append(strehl)
        rms_list.append(rms)

    return {'strehls': np.array(strehls), 'rms': np.array(rms_list)}


def _commands_to_surface(commands: np.ndarray, n_act: int, N: int,
                         coupling: float) -> np.ndarray:
    """Convert DM actuator commands to a wavefront surface on an NxN grid."""
    surface = np.zeros((N, N))
    act_pos = np.linspace(0, 1, n_act)
    grid = np.linspace(0, 1, N)
    X, Y = np.meshgrid(grid, grid)

    pitch = 1.0 / (n_act - 1) if n_act > 1 else 1.0
    w = pitch / np.sqrt(-np.log(coupling) / np.log(2))

    for idx, cmd in enumerate(commands):
        if abs(cmd) < 1e-12:
            continue
        i, j = divmod(idx, n_act)
        r2 = (X - act_pos[j])**2 + (Y - act_pos[i])**2
        surface += cmd * np.exp(-np.log(2) * r2 / w**2)
    return surface


def _circular_mask(N: int) -> np.ndarray:
    """Create a circular boolean mask on an NxN grid."""
    y, x = np.mgrid[-1:1:complex(N), -1:1:complex(N)]
    return (x**2 + y**2) <= 1.0
```

---

## 11. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Seeing limit | $\theta_{\text{seeing}} \approx 0.98\lambda/r_0$ |
| Fried parameter | $r_0 \propto \lambda^{6/5}$, typically 10–20 cm at 500 nm |
| Greenwood frequency | $f_G = 0.427 v_{\text{eff}}/r_0$, typically 20–50 Hz |
| Isoplanatic angle | $\theta_0 = 0.314 r_0/\bar{h}$, typically 2–5 arcsec |
| Coherence time | $\tau_0 = 0.314 r_0/v_{\text{eff}}$ |
| Shack-Hartmann | Microlens array measures local slopes $\nabla W$ |
| Least-squares reconstruction | $\hat{\mathbf{a}} = \mathbf{D}^+\mathbf{s}$ (modal) or $\hat{\mathbf{w}} = \mathbf{G}^+\mathbf{s}$ (zonal) |
| Influence function | $W_{\text{DM}} = \sum_k c_k \phi_k(\mathbf{r})$ |
| Fitting error | $\sigma_{\text{fit}}^2 = \alpha(d/r_0)^{5/3}$ |
| Temporal error | $\sigma_{\text{temp}}^2 = (f_G/f_{-3\text{dB}})^{5/3}$ |
| Integrator gain | $\mathbf{c}_{k+1} = \mathbf{c}_k + g\hat{\mathbf{w}}_k$ |
| Strehl ratio | $S \approx e^{-(2\pi\sigma)^2}$ |
| Sky coverage (NGS) | ~1% at visible wavelengths |
| Laser guide star | Artificial beacon at 10–90 km; cannot sense tip-tilt |
| MCAO | Multiple DMs conjugate to turbulence layers; wide field |

---

## 12. Exercises

### Exercise 1: AO System Design

You are designing a Shack-Hartmann AO system for a 4-meter telescope at a site with $r_0 = 12$ cm at 500 nm and $v_{\text{eff}} = 15$ m/s. The science wavelength is $K$-band (2.2 $\mu$m). (a) Calculate $r_0$, $f_G$, $\theta_0$, and $\tau_0$ at $K$-band. (b) Choose the number of subapertures and DM actuators. (c) What loop rate is needed for the temporal error to be less than 50 nm RMS? (d) Estimate the total Strehl ratio assuming fitting, temporal, and 30 nm noise error terms.

### Exercise 2: Wavefront Reconstruction

Generate a random atmospheric wavefront with $D/r_0 = 10$ (use the Kolmogorov phase screen function from Lesson 15). (a) Simulate a 16×16 Shack-Hartmann measurement (compute average slopes in each subaperture). (b) Reconstruct the wavefront using both zonal and modal (Zernike, 36 modes) methods. (c) Compare the reconstruction error maps. (d) How does the reconstruction error change with measurement noise (add Gaussian noise to slopes with SNR = 10, 20, 50)?

### Exercise 3: Closed-Loop Simulation

Using the `ao_closed_loop` function provided in §10.3 (or your own implementation), simulate 100 time steps of a 12×12 actuator AO system with a 10×10 WFS on Kolmogorov phase screens with $D/r_0 = 8$. (a) Plot the Strehl ratio vs. time step for gains $g = 0.3, 0.5, 0.7, 0.9$. (b) Which gain gives the best average Strehl? (c) Show the uncorrected and corrected wavefront side-by-side for the best case. (d) Compute the PSF for the corrected case and compare with the diffraction-limited PSF.

### Exercise 4: Laser Guide Star Limitations

A sodium laser guide star is created at altitude $h = 90$ km above a 10-meter telescope with $r_0 = 15$ cm at 500 nm. (a) Calculate the angular extent of the cone effect (the difference between the LGS beam and a plane wave). (b) Estimate the focus anisoplanatism error using the formula $\sigma_{\text{cone}}^2 = (D/d_0)^{5/3}$ where $d_0 \approx 2.91 r_0 (h/\bar{h})^{6/5}$ and $\bar{h} = 5$ km. (c) At what telescope diameter does the cone effect error equal the fitting error for a 20×20 actuator system? (d) How does MCAO mitigate the cone effect?

---

## 13. References

1. Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press. — The foundational textbook on astronomical AO.
2. Roddier, F. (Ed.) (1999). *Adaptive Optics in Astronomy*. Cambridge University Press. — Comprehensive edited volume covering all aspects of AO.
3. Tyson, R. K. (2015). *Principles of Adaptive Optics* (4th ed.). CRC Press. — Accessible introduction with emphasis on system engineering.
4. Davies, R., & Kasper, M. (2012). "Adaptive optics for astronomy." *Annual Review of Astronomy and Astrophysics*, 50, 305–351. — Excellent modern review.
5. Roddier, F. (1988). "Curvature sensing and compensation: a new concept in adaptive optics." *Applied Optics*, 27(7), 1223–1225. — Original curvature sensor paper.
6. Ragazzoni, R. (1996). "Pupil plane wavefront sensing with an oscillating prism." *Journal of Modern Optics*, 43(2), 289–293. — The pyramid sensor.
7. Rigaut, F. (2015). "Astronomical adaptive optics." *Publications of the Astronomical Society of the Pacific*, 127(958), 1197–1203. — Overview of AO systems and performance.
8. Guyon, O. (2005). "Limits of adaptive optics for high-contrast imaging." *The Astrophysical Journal*, 629, 592–614. — Extreme AO theory for exoplanet imaging.

---

[← Previous: 15. Zernike Polynomials](15_Zernike_Polynomials.md) | [Next: 17. Spectroscopy →](17_Spectroscopy.md)

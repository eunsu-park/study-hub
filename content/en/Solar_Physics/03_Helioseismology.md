# Helioseismology

[← Previous: Nuclear Energy Generation](02_Nuclear_Energy_Generation.md) | [Next: Photosphere →](04_Photosphere.md)

## Learning Objectives

1. Classify solar oscillation modes into p-modes, g-modes, and f-modes, and explain the physical restoring force for each
2. Describe the spherical harmonic decomposition of solar oscillations and interpret the quantum numbers $(n, l, m)$
3. Read and interpret the $l$-$\nu$ (power spectrum) diagram, identifying ridges and mode spacings
4. Explain how acoustic ray paths probe different depths and derive the turning point condition
5. Outline helioseismic inversion techniques and the key discoveries about the solar interior
6. Distinguish global from local helioseismology and describe applications of time-distance and ring-diagram methods

---

## Why This Matters

Helioseismology is one of the most remarkable achievements in modern astrophysics. By observing tiny oscillations at the solar surface — velocity fluctuations of less than 1 m/s in a star with surface convective velocities of 1-2 km/s — scientists have mapped the Sun's internal sound speed, density, and rotation with astonishing precision. The sound speed profile has been determined to better than 0.1% accuracy throughout most of the interior, and the internal rotation has been measured from the core to the surface as a function of both radius and latitude.

Helioseismology validated the Standard Solar Model, discovered the tachocline, measured the depth of the convection zone, and constrained the helium abundance of the solar envelope. It transformed the Sun from a star with a hypothetical interior into one whose internal structure is known with the precision of a laboratory experiment. Today, local helioseismology can image active regions on the far side of the Sun, map subsurface flows, and probe the structure beneath sunspots.

> **Analogy**: Helioseismology is to solar physics what ultrasound imaging is to medicine. Just as a doctor can "see" inside a patient by analyzing how sound waves reflect and refract through tissue, helioseismologists can "see" inside the Sun by analyzing how sound waves propagate through its interior. The Sun rings like a bell struck by the constant hammering of convection, and we listen to its tones to deduce its internal structure.

---

## 1. Solar Oscillations

### Discovery

In 1960, Robert Leighton discovered that the solar surface oscillates with a characteristic period of about 5 minutes. Initially interpreted as a local atmospheric phenomenon, these oscillations were recognized in the 1970s (by Ulrich, Leibacher, and Stein; independently by Deubner) as global resonant modes — standing sound waves trapped inside the Sun.

The Sun vibrates simultaneously in approximately **10 million** distinct modes, each with a precisely defined frequency. The typical velocity amplitude of an individual mode is only $\sim 10$ cm/s, but the superposition of millions of modes produces a root-mean-square surface velocity of $\sim 0.5$ km/s.

### What Drives the Oscillations?

The oscillations are **stochastically excited** by turbulent convection near the surface. Convective eddies at the top of the convection zone continuously pump energy into the modes. The modes are also **damped** by convection (the same process that excites them also dissipates them). Each mode is continuously excited and damped, reaching a statistical equilibrium amplitude.

The mode lifetimes range from a few hours for high-$l$ modes to several months for low-$l$ modes. The longer the lifetime, the narrower the spectral peak, and the more precisely the frequency can be measured.

### Spherical Harmonic Decomposition

Because the Sun is (to an excellent approximation) spherically symmetric, its oscillation patterns are described by **spherical harmonics** $Y_l^m(\theta, \phi)$. Any displacement field on the solar surface can be decomposed as:

$$\xi(\theta, \phi, t) = \sum_{n,l,m} A_{nlm} \, Y_l^m(\theta, \phi) \, e^{-i\omega_{nlm} t}$$

where:
- $n$ = **radial order** — number of nodes in the radial direction. $n = 0$ means no radial nodes (fundamental mode); $n = 1, 2, 3, \ldots$ means 1, 2, 3, ... radial nodes
- $l$ = **angular degree** — total number of nodal lines on the surface. $l = 0$ means purely radial oscillation; $l = 1$ is a dipole; $l = 2$ is a quadrupole
- $m$ = **azimuthal order** — number of nodal lines passing through the poles. $-l \leq m \leq +l$

The angular degree $l$ determines the horizontal wavelength on the surface:

$$\lambda_h = \frac{2\pi R_\odot}{\sqrt{l(l+1)}} \approx \frac{2\pi R_\odot}{l} \quad \text{for large } l$$

For $l = 1$: $\lambda_h \sim 4.4 \times 10^9$ m (global). For $l = 1000$: $\lambda_h \sim 4.4 \times 10^6$ m (granular scale).

### Observation Techniques

Two complementary methods are used to observe solar oscillations:

**Doppler velocity**: Measure the line-of-sight velocity at each point on the solar disk using the Doppler shift of a spectral line (typically Ni I 6768 A). Ground-based networks like **GONG** (Global Oscillation Network Group, 6 stations worldwide) and space instruments like **MDI** (on SOHO) and **HMI** (on SDO) use this technique.

**Intensity (photometry)**: Oscillation modes also produce tiny brightness variations ($\delta I/I \sim 10^{-6}$) due to temperature fluctuations. The space missions **SOHO/VIRGO** and **SDO/HMI** measure these.

The advantage of full-disk networks and space-based instruments is continuous, long-duration coverage. Gaps in the time series create sidelobes in the power spectrum that can obscure closely spaced modes.

---

## 2. Mode Classification

### p-Modes (Pressure Modes)

**Restoring force**: Pressure (compression and rarefaction of the gas)

p-modes are **acoustic waves** — sound waves trapped in the solar interior by refraction (increasing sound speed with depth bends the waves back toward the surface) and reflection (the sharp density drop at the photosphere reflects the waves inward).

Properties:
- **Frequency range**: 1-5 mHz (periods of 3-17 minutes), with a peak near 3 mHz (5-minute period)
- **Radial order**: $n \geq 1$ (at least one radial node)
- **Propagation**: Throughout the interior, with deeper penetration for lower $l$
- **Detection**: Well-observed; they dominate the solar oscillation spectrum
- **Sensitivity**: Primarily to the sound speed profile $c(r)$ and density $\rho(r)$

p-modes are the workhorses of helioseismology. Their frequencies have been measured for modes spanning $l = 0$ to $l \sim 3500$ and $n$ up to $\sim 30$.

### g-Modes (Gravity Modes)

**Restoring force**: Buoyancy (gravity acting on density perturbations in a stably stratified region)

g-modes are **internal gravity waves** analogous to waves at the interface of oil and water (where gravity provides the restoring force).

Properties:
- **Frequency range**: Below $\sim 0.4$ mHz (periods > 40 minutes)
- **Propagation**: Only in the radiative zone (where the stratification is stable). They are **evanescent** in the convection zone
- **Surface amplitude**: Extremely small ($v \lesssim 1$ mm/s) because the evanescent barrier of the convection zone exponentially attenuates them
- **Detection**: Marginal — individual g-modes have not been unambiguously detected, though statistical evidence for their signature has been reported (Garcia et al., 2007)
- **Sensitivity**: Primarily to the core conditions (rotation, composition)

The detection of individual g-modes would be transformative for solar physics, providing direct information about the rotation and magnetic field in the deep core.

### f-Modes (Fundamental Modes)

**Restoring force**: Surface gravity (like ocean waves)

f-modes are **surface gravity waves** with no radial nodes ($n = 0$). They are analogous to deep-water ocean waves.

Properties:
- **Dispersion relation**: $\omega^2 \approx g_\text{surf} k_h = g_\text{surf} \sqrt{l(l+1)} / R_\odot$
- **Frequency range**: Intermediate between p-modes and g-modes
- **Penetration depth**: Shallow (roughly $1/k_h$)
- **Sensitivity**: To near-surface density and the solar radius

f-modes are useful for calibrating the solar radius and probing near-surface properties.

---

## 3. The $l$-$\nu$ Diagram

### Structure of the Power Spectrum

The oscillation power spectrum, when plotted as frequency $\nu$ versus angular degree $l$, reveals a striking **ridge structure**. Each ridge corresponds to a constant radial order $n$: the $n = 0$ ridge (f-modes), $n = 1$ ridge (first overtone p-modes), $n = 2$ ridge, and so on.

```
Frequency ν (mHz)
     5 │           ╱╱╱╱   n=20
       │         ╱╱╱╱
     4 │       ╱╱╱╱        n=10
       │     ╱╱╱╱
     3 │   ╱╱╱╱             n=5
       │ ╱╱╱╱
     2 │╱╱╱╱                n=2
       │╱╱                  n=1
     1 │╱                   n=0 (f-modes)
       │
     0 └──────────────────────────
       0    500   1000  1500  2000
                Angular degree l
```

Key features:
- Ridges curve upward (higher $l$ modes have higher frequency for given $n$)
- Ridges are approximately equally spaced in frequency at low $l$ (the large separation)
- High-$l$ modes are concentrated near 3 mHz (the acoustic cutoff limits their frequency)
- The power is strongest near 3 mHz (the 5-minute oscillation peak)

### Mode Frequencies and Spacings

**Large frequency separation** ($\Delta\nu$): The frequency difference between modes of consecutive radial order at fixed $l$:

$$\Delta\nu = \nu_{n+1,l} - \nu_{n,l} \approx \left(2\int_0^{R_\odot} \frac{dr}{c(r)}\right)^{-1} \approx 135 \; \mu\text{Hz}$$

This is the inverse of the sound travel time across a solar diameter. It depends on the mean density of the Sun:

$$\Delta\nu \propto \left(\frac{M}{R^3}\right)^{1/2} \propto \bar{\rho}^{1/2}$$

**Small frequency separation** ($\delta\nu$): The frequency difference between modes with $(n, l)$ and $(n-1, l+2)$:

$$\delta\nu = \nu_{n,l} - \nu_{n-1,l+2}$$

This spacing is sensitive to the sound speed gradient in the core, making it a probe of stellar age (as hydrogen is converted to helium, the core composition changes, altering the sound speed gradient).

> **Physical insight**: The large separation tells you "how big the Sun is" (mean density), while the small separation tells you "how old the Sun is" (core composition). Together, they form the basis of asteroseismology — applying helioseismic techniques to other stars.

### The Acoustic Cutoff Frequency

There is a maximum frequency for trapped acoustic modes, called the **acoustic cutoff frequency**:

$$\nu_\text{ac} = \frac{c}{4\pi H_P}$$

where $c$ is the sound speed and $H_P$ is the pressure scale height at the photosphere. For the Sun: $\nu_\text{ac} \approx 5.3$ mHz (period $\sim 3$ minutes). Waves with $\nu > \nu_\text{ac}$ propagate freely into the atmosphere and are not trapped.

The 5-minute oscillation peak occurs just below the acoustic cutoff — this is not a coincidence. Convection excites waves at all frequencies, but only those below $\nu_\text{ac}$ are trapped and can build up as resonant modes.

---

## 4. Asymptotic Theory

### The Duvall Law

For high-order ($n \gg l$) p-modes, the frequencies follow an asymptotic relation known as the **Duvall law**:

$$\int_{r_t}^{R_\odot} \left(\frac{1}{c^2(r)} - \frac{l(l+1)}{\omega^2 r^2}\right)^{1/2} dr = \frac{(n + \alpha)\pi}{\omega}$$

where:
- $r_t$ is the **inner turning point** (where the wave refracts back upward)
- $\omega = 2\pi\nu$ is the angular frequency
- $\alpha$ is a phase constant related to the surface reflection properties
- $c(r)$ is the sound speed profile

This integral equation connects the observed frequencies $(\nu_{nl})$ to the internal sound speed profile $c(r)$ — the fundamental relationship that makes helioseismic inversion possible.

### Acoustic Ray Paths

Sound waves in the Sun follow curved paths determined by **Snell's law** in a medium with a radially varying sound speed $c(r)$. A ray launched at angle $\theta$ to the radial direction at the surface obeys:

$$\frac{r \sin\theta}{c(r)} = \text{constant} = \frac{R_\odot \sin\theta_0}{c(R_\odot)}$$

This is analogous to geometric optics. Because the sound speed increases with depth (higher temperature), rays are continuously refracted (bent) away from the radial direction, eventually turning around and returning to the surface.

### The Turning Point

The inner turning point $r_t$ is where the ray path becomes horizontal (purely tangential). At this point:

$$\frac{\omega}{k_h} = c(r_t) \quad \Longrightarrow \quad \frac{\omega}{\sqrt{l(l+1)}} \cdot r_t = c(r_t)$$

or equivalently:

$$\frac{c(r_t)}{r_t} = \frac{\omega}{\sqrt{l(l+1)}}$$

> **Key insight**: The turning point depth depends on the ratio $\omega / l$. For a given frequency:
> - **High $l$** (many nodal lines): shallow turning point — the mode probes only the outer layers
> - **Low $l$** (few nodal lines): deep turning point — the mode probes deep into the interior
> - **$l = 0$** (radial modes): no turning point — the mode passes through the center

This is the fundamental reason why observing modes over a wide range of $l$ values provides information at all depths within the Sun.

Approximate turning point depths:

| Angular degree $l$ | Turning point $r_t / R_\odot$ | Probes |
|---------------------|-------------------------------|--------|
| 0-3 | 0 (reaches center) | Core |
| 20 | ~0.2 | Core/radiative zone |
| 100 | ~0.5 | Radiative zone |
| 500 | ~0.85 | Upper convection zone |
| 1000 | ~0.95 | Near-surface layers |

### Sound Speed Profile

The sound speed in an ideal gas is:

$$c = \sqrt{\frac{\gamma P}{\rho}} = \sqrt{\frac{\gamma k_B T}{\mu m_H}}$$

Since temperature increases monotonically inward (from $\sim 5800$ K at the surface to $\sim 1.57 \times 10^7$ K at the center), the sound speed also increases inward, from $\sim 7$ km/s at the photosphere to $\sim 500$ km/s at the center.

The sound travel time from center to surface is:

$$\tau_\text{travel} = \int_0^{R_\odot} \frac{dr}{c(r)} \approx 3700 \text{ s} \approx 62 \text{ min}$$

And the round-trip time $2\tau_\text{travel} \approx 7400$ s, giving $\Delta\nu \approx 1/7400 \text{ s} \approx 135 \; \mu$Hz, consistent with the observed large separation.

---

## 5. Inversion Techniques

### The Inverse Problem

Helioseismic inversion is the process of determining the Sun's internal structure from the observed oscillation frequencies. It is a classic **inverse problem**: we observe the frequencies (effects) and want to deduce the sound speed profile (cause).

The starting point is a linearized relation. If the actual Sun deviates slightly from a reference model (e.g., the SSM) in its sound speed $c(r)$, the frequency shifts are:

$$\frac{\delta\omega_{nl}}{\omega_{nl}} = \int_0^{R_\odot} K_{nl}^{c^2}(r) \frac{\delta c^2}{c^2}(r) \, dr + \int_0^{R_\odot} K_{nl}^{\rho}(r) \frac{\delta\rho}{\rho}(r) \, dr + \text{surface terms}$$

where:
- $\delta\omega_{nl} = \omega_{nl}^\text{obs} - \omega_{nl}^\text{model}$ is the frequency difference between observation and model
- $K_{nl}^{c^2}(r)$ and $K_{nl}^{\rho}(r)$ are **kernels** — known functions that describe how sensitive each mode frequency is to perturbations at each depth
- The surface terms account for inadequate modeling of the near-surface layers

The kernels are computed from the eigenfunctions of the reference model. Each mode has a kernel peaked near its turning point and inner cavity, so different modes provide information about different depths.

### Regularized Least Squares (RLS)

The RLS method parameterizes the unknown perturbation $\delta c^2 / c^2$ on a radial grid and solves the discretized integral equation by least squares, with a regularization (smoothing) term to prevent the solution from oscillating wildly:

$$\text{minimize} \sum_{nl} \left(\frac{\delta\omega_{nl}^\text{obs} - \delta\omega_{nl}^\text{predicted}}{\sigma_{nl}}\right)^2 + \lambda \int \left(\frac{d^2}{dr^2}\frac{\delta c^2}{c^2}\right)^2 dr$$

where $\sigma_{nl}$ are the frequency measurement uncertainties and $\lambda$ is the regularization parameter (balancing fit quality against smoothness).

### Optimally Localized Averages (OLA)

The OLA method (also called the Backus-Gilbert method) takes a different approach: for each target radius $r_0$, it finds a linear combination of modes whose combined kernel is as localized around $r_0$ as possible:

$$\frac{\hat{\delta c^2}}{c^2}(r_0) = \sum_{nl} a_{nl}(r_0) \frac{\delta\omega_{nl}}{\omega_{nl}}$$

The coefficients $a_{nl}$ are chosen so that the **averaging kernel** $\mathcal{K}(r_0, r) = \sum_{nl} a_{nl} K_{nl}(r)$ peaks sharply at $r = r_0$ and is small elsewhere.

The resolution at each depth is determined by how well the available modes can be combined to create a localized kernel. Resolution is best where many modes have turning points (roughly $0.1$-$0.9 R_\odot$) and worst near the center (few modes penetrate there) and very near the surface (surface term uncertainties).

---

## 6. Key Discoveries

Helioseismic inversions have produced some of the most important results in solar physics:

### 6.1 Sound Speed Profile

The inverted sound speed profile agrees with the Standard Solar Model to better than **0.2%** throughout most of the interior ($0.1$-$0.9 R_\odot$). This is a spectacular validation of the SSM. The largest deviations occur:
- Just below the convection zone base (related to the tachocline and possible overshoot)
- In the core (limited by the small number of low-$l$ modes)
- Near the surface (where the SSM's treatment of convection is crude)

The sound speed comparison is the most stringent test of the SSM and is the primary reason we are confident in our understanding of the solar interior.

### 6.2 Base of the Convection Zone

The depth of the convection zone base is determined by a change in the gradient of the sound speed (from nearly adiabatic above to radiative below). Helioseismology gives:

$$r_\text{BCZ} = (0.713 \pm 0.001) R_\odot$$

This extraordinarily precise measurement provides a strong constraint on the SSM, particularly on the opacity profile (which determines where convection begins).

### 6.3 Internal Rotation

Helioseismic rotation inversions revealed:

1. **Convection zone**: differential rotation persisting throughout, with equator rotating faster ($\sim 460$ nHz) than poles ($\sim 340$ nHz). The rotation is nearly constant on **radial lines** (constant along cones at fixed colatitude), not on cylinders — contradicting early theoretical expectations.

2. **Radiative zone**: rotates nearly rigidly at $\sim 430$ nHz, intermediate between the equatorial and polar rates of the convection zone.

3. **Tachocline**: thin ($\sim 0.04 R_\odot$) transition layer between the differential rotation above and rigid rotation below (discussed in Lesson 01).

4. **Near-surface shear layer**: a narrow region near the surface ($r > 0.95 R_\odot$) where the rotation rate increases inward.

5. **Core rotation**: less well constrained, but consistent with rigid rotation at $\sim 430$ nHz. There is ongoing investigation into whether the innermost core rotates faster (which would be evidence for angular momentum transport processes).

### 6.4 Helium Abundance

The ionization of helium in the convection zone produces a localized dip in the first adiabatic exponent $\Gamma_1$, which affects the sound speed. By analyzing the oscillatory signature this produces in the frequencies, helioseismology constrains the envelope helium mass fraction:

$$Y_\text{env} = 0.2485 \pm 0.0035$$

This is lower than the initial solar helium abundance ($Y_0 \approx 0.27$) because of gravitational settling of helium over the Sun's lifetime.

---

## 7. Local Helioseismology

While global helioseismology analyzes mode frequencies (a frequency-domain approach), **local helioseismology** analyzes the wave field in the time-space domain to probe three-dimensional structure and flows.

### Ring-Diagram Analysis

In small patches of the solar surface ($\sim 15° \times 15°$), the local power spectrum forms **rings** in the $(k_x, k_y, \omega)$ space (rather than ridges, because both horizontal wave vector components are resolved). Perturbations to the ring shapes reveal:

- **Subsurface flow velocities** (from Doppler shifts of the rings)
- **Sound speed perturbations** (from changes in ring radii)
- **Magnetic field effects** (from ring asymmetries)

Ring diagrams can map flows at different depths beneath active regions and detect subsurface jet streams.

### Time-Distance Helioseismology

This technique measures the **travel time** of acoustic waves between pairs of points on the solar surface. By cross-correlating the oscillation signals at two points separated by a known distance, the travel time can be extracted with a precision of a few seconds.

Differences between travel times in opposite directions reveal **flows** (the wave travels faster with the flow than against it):

$$\delta\tau = \tau_+ - \tau_- \propto \int_\text{ray path} \frac{\mathbf{v} \cdot d\mathbf{s}}{c^2}$$

where $\mathbf{v}$ is the flow velocity and $d\mathbf{s}$ is the ray path element.

Applications:
- Mapping **meridional circulation** (poleward flow at the surface, $\sim 15$ m/s; equatorward return flow at depth)
- Detecting **subsurface flows** around active regions
- Measuring **sound speed anomalies** beneath sunspots (they are found to be faster — hotter — below the visible spot)

### Acoustic Holography (Far-Side Imaging)

Perhaps the most dramatic application of local helioseismology: detecting active regions on the **far side** of the Sun (the hemisphere not visible from Earth).

The principle: acoustic waves generated on the near side propagate through the interior, reach the far side, interact with any structures there (active regions perturb the local acoustic properties), and some of the wave energy returns to the near side. By analyzing the returning waves — essentially "focusing" through the Sun using holographic techniques — the presence and location of far-side active regions can be determined.

This capability (implemented by Lindsey and Braun, 2000) provides several days of warning before an active region rotates onto the Earth-facing hemisphere — valuable for space weather forecasting.

### Sunspot Seismology

Sunspots absorb and scatter acoustic waves, creating a complex interaction that can be studied using local helioseismology. Key findings:

- Sunspots **absorb** up to 50% of incident p-mode power (the missing energy is likely converted to MHD waves)
- The sound speed beneath sunspots is **increased** at depths of 4-7 Mm (possibly due to the magnetic field suppressing convective heat transport, causing a thermal rearrangement)
- Sunspots have characteristic **downdrafts** and **inflows** at the surface, with return circulation at depth

---

## 8. Modern Helioseismology

### Space Missions

Three pivotal space missions have driven helioseismology forward:

**SOHO (1995-present)**: Solar and Heliospheric Observatory, at the L1 Lagrange point. Instruments: MDI (Michelson Doppler Imager, 1996-2011) and GOLF (Global Oscillations at Low Frequencies, still operating). SOHO provided the first continuous, uninterrupted observations of solar oscillations from space.

**SDO (2010-present)**: Solar Dynamics Observatory, in geosynchronous orbit. Instrument: HMI (Helioseismic and Magnetic Imager). HMI observes the full solar disk at 4096 $\times$ 4096 resolution, measuring Doppler velocity and magnetic field every 45 seconds. This is the current workhorse for helioseismology.

**Solar Orbiter (2020-present)**: Will provide out-of-ecliptic observations, enabling new perspectives on solar oscillations (especially near the poles, which are foreshortened as seen from Earth).

### Ground-Based Networks

**GONG** (Global Oscillation Network Group): Six stations around the world providing nearly continuous Doppler velocity observations since 1995. The network has recently been upgraded with higher-resolution cameras.

**BiSON** (Birmingham Solar Oscillations Network): Specializes in low-$l$ modes using Sun-as-a-star observations (unresolved disk). These modes penetrate to the core and are crucial for core rotation measurements.

### Asteroseismology: Exporting the Technique

The techniques developed for helioseismology have been exported to other stars (**asteroseismology**). Space missions like **Kepler** (2009-2018) and **TESS** (2018-present) measure brightness oscillations in thousands of stars. The large and small frequency separations ($\Delta\nu$ and $\delta\nu$) are used to estimate stellar masses, radii, and ages with remarkable precision — typically 2-5% for mass and radius, and 10-20% for age.

This is a direct legacy of helioseismology: the Sun served as the calibration standard against which all asteroseismic techniques are validated.

---

## Summary

- The Sun oscillates in $\sim 10$ million modes simultaneously, excited stochastically by convection
- **p-modes** (acoustic, pressure restoring force) dominate the observed spectrum at 1-5 mHz and probe the interior sound speed
- **g-modes** (buoyancy restoring force) are trapped in the radiative zone and have vanishingly small surface amplitudes
- The **$l$-$\nu$ diagram** reveals ridge structure; the **large separation** ($\Delta\nu \approx 135 \; \mu$Hz) depends on mean density, and the **small separation** is sensitive to core conditions
- The **turning point** depth depends on $\omega/l$ — low-$l$ modes penetrate deepest, enabling core measurements
- **Inversion techniques** (RLS, OLA) convert observed frequencies into internal sound speed and rotation profiles accurate to $< 0.2\%$
- Key discoveries: convection zone depth ($0.713 R_\odot$), rigid rotation of radiative zone, differential rotation of convection zone, tachocline, helium abundance
- **Local helioseismology** (ring diagrams, time-distance, holography) probes 3D structure: subsurface flows, sunspot interiors, and far-side active regions

---

## Practice Problems

### Problem 1: Mode Identification

A solar oscillation mode has frequency $\nu = 3.2$ mHz, angular degree $l = 100$, and radial order $n = 8$. (a) Is this a p-mode, g-mode, or f-mode? Explain. (b) Estimate the horizontal wavelength on the solar surface. (c) Using the turning point relation $c(r_t)/r_t = 2\pi\nu/\sqrt{l(l+1)}$ and assuming $c(r) \approx c_0 (r/R_\odot)^{-0.3}$ with $c_0 = 7$ km/s, estimate the turning point depth. (d) What physical quantity is this mode primarily sensitive to?

### Problem 2: Large Separation

(a) Using the sound speed profile $c(r) = c_c [1 - 0.9(r/R_\odot)^2]^{1/2}$ with $c_c = 500$ km/s, compute the acoustic travel time $\tau = \int_0^{R_\odot} dr/c(r)$ numerically (or analytically if possible). (b) Calculate the expected large separation $\Delta\nu = 1/(2\tau)$. (c) Compare with the observed value of $\sim 135 \; \mu$Hz. (d) If a star has twice the solar radius and the same mean sound speed, what large separation would you expect?

### Problem 3: Probing Different Depths

You have frequency measurements for modes at $l = 1$, $l = 20$, $l = 200$, and $l = 1000$, all with similar frequencies ($\nu \approx 3$ mHz). (a) Rank these modes by how deep they penetrate into the Sun. (b) Which mode(s) would be most useful for measuring the sound speed in the core? (c) Which mode(s) would be most useful for studying the structure just below the photosphere? (d) Why is it important to observe a wide range of $l$ values?

### Problem 4: Rotation Splitting

In a non-rotating Sun, modes with different $m$ values but the same $n$ and $l$ would have identical frequencies (degeneracy). Rotation lifts this degeneracy, producing a frequency splitting:

$$\nu_{nlm} = \nu_{nl0} + m \int_0^{R_\odot} K_{nl}(r) \Omega(r) \, dr$$

for a simplified case (ignoring latitudinal dependence). (a) If the Sun rotated rigidly at $\Omega/(2\pi) = 430$ nHz, what would the splitting between adjacent $m$ values be? (b) In practice, the splitting varies with $l$. Qualitatively explain why low-$l$ modes (which probe the rigidly rotating core) and high-$l$ modes (which probe the differentially rotating convection zone) would show different splitting patterns.

### Problem 5: Far-Side Imaging

A sound wave travels from a point on the near side of the Sun, through the interior, to a point on the far side, and back. (a) Using the mean sound speed $\bar{c} \approx 100$ km/s and a path length of approximately $2R_\odot$, estimate the round-trip travel time. (b) If an active region on the far side causes a local sound speed increase of 1%, by how much would the travel time change? (c) Given that travel times can be measured to a precision of about 1 second, is this detectable? (d) Why is far-side imaging useful for space weather forecasting?

---

[← Previous: Nuclear Energy Generation](02_Nuclear_Energy_Generation.md) | [Next: Photosphere →](04_Photosphere.md)

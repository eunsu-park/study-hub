# Photosphere

[← Previous: Helioseismology](03_Helioseismology.md) | [Next: Chromosphere and Transition Region →](05_Chromosphere_and_TR.md)

## Learning Objectives

1. Write the radiative transfer equation, define optical depth and source function, and derive the formal solution for emergent intensity
2. Explain the Eddington-Barbier relation and why we "see" to optical depth unity
3. Derive the limb darkening law and explain its physical origin in terms of the photospheric temperature gradient
4. Describe granulation and supergranulation patterns and their connection to subsurface convection
5. Explain spectral line formation, the Voigt profile, and the curve of growth
6. Discuss the First Ionization Potential (FIP) effect and its diagnostic use

---

## Why This Matters

The photosphere is the visible "surface" of the Sun — the thin layer ($\sim 500$ km thick) from which the photons we see are emitted. It is the boundary between the opaque interior and the transparent atmosphere, and it is where the Sun transitions from a plasma physics problem to a radiation physics problem. Everything we know about the solar interior from traditional astronomy (before helioseismology and neutrinos) came from analyzing photospheric radiation.

The photosphere is also where the physics of radiative transfer — one of the most beautiful and practical branches of astrophysics — comes into its own. The concepts of optical depth, source function, and limb darkening apply not just to the Sun but to every star, planet atmosphere, and interstellar cloud we observe. Mastering radiative transfer in the photosphere provides tools applicable across all of astrophysics.

> **Analogy**: The photosphere is like the surface of a swimming pool viewed from above. You can see into the water, but only so far — eventually the accumulated scattering makes it opaque. The depth you can see to depends on the water's clarity (opacity) and the angle you look at (limb darkening). On a clear day you see deeper; in murky water, shallower. The "bottom" you see is not a real surface but the depth at which the water becomes opaque — just as the photosphere is not a solid surface but the depth at which the plasma becomes opaque.

---

## 1. Radiative Transfer Basics

### Specific Intensity

The fundamental quantity in radiative transfer is the **specific intensity** $I_\nu$, defined as the energy passing through a unit area perpendicular to the beam, per unit time, per unit frequency, per unit solid angle:

$$I_\nu = \frac{dE}{dA \, dt \, d\nu \, d\Omega} \quad \left[\text{W m}^{-2} \text{Hz}^{-1} \text{sr}^{-1}\right]$$

Unlike flux, specific intensity is a **directional** quantity — it describes the radiation field as a function of both position and direction.

### Absorption and Emission

As a beam of radiation travels through a medium:

- **Absorption** removes energy at a rate proportional to $I_\nu$ and the **absorption coefficient** $\kappa_\nu$ (m$^{-1}$ or m$^2$/kg when defined per unit mass):

$$dI_\nu = -\kappa_\nu I_\nu \, ds$$

- **Emission** adds energy at a rate given by the **emission coefficient** $j_\nu$ (W m$^{-3}$ Hz$^{-1}$ sr$^{-1}$):

$$dI_\nu = +j_\nu \, ds$$

### The Radiative Transfer Equation

Combining absorption and emission:

$$\boxed{\frac{dI_\nu}{ds} = -\kappa_\nu I_\nu + j_\nu}$$

This is the fundamental equation of radiative transfer. It states that the intensity along a ray changes due to removal (absorption) and addition (emission) of photons.

### Optical Depth

The **optical depth** $\tau_\nu$ is defined along the ray path (measured from the observer, increasing inward):

$$d\tau_\nu = -\kappa_\nu \, ds \quad \text{(for rays directed outward toward observer)}$$

or equivalently, integrating inward from the surface:

$$\tau_\nu(s) = \int_s^{s_\text{surface}} \kappa_\nu \, ds'$$

Optical depth is the natural "distance" variable in radiative transfer — it measures how many mean free paths deep a point is.

- $\tau_\nu = 0$: the surface (observer's side)
- $\tau_\nu = 1$: one mean free path deep — photons from here have a $\sim 37\%$ ($= e^{-1}$) chance of escaping without further absorption
- $\tau_\nu \gg 1$: deep interior — photons are absorbed and re-emitted many times before escaping

### Source Function

The **source function** $S_\nu$ is the ratio of emission to absorption:

$$S_\nu = \frac{j_\nu}{\kappa_\nu}$$

The radiative transfer equation in terms of optical depth becomes:

$$\frac{dI_\nu}{d\tau_\nu} = I_\nu - S_\nu$$

This elegant form says: the intensity changes toward the source function. If $I_\nu > S_\nu$, the beam is absorbed (intensity decreases). If $I_\nu < S_\nu$, the beam is amplified (intensity increases). The medium always tries to bring the radiation toward $S_\nu$.

### Local Thermodynamic Equilibrium (LTE)

In **LTE**, the source function equals the **Planck function** at the local temperature:

$$S_\nu = B_\nu(T) = \frac{2h\nu^3}{c^2} \frac{1}{e^{h\nu/k_BT} - 1}$$

LTE holds when collisions dominate over radiation in determining the atomic level populations. This is an excellent approximation in the photosphere (the densities are high enough that collision rates greatly exceed radiative rates), though it breaks down in the chromosphere and corona.

---

## 2. The Eddington-Barbier Relation

### Formal Solution

The formal solution to the radiative transfer equation for the emergent intensity (at $\tau_\nu = 0$) from a semi-infinite atmosphere is:

$$I_\nu(0, \mu) = \int_0^\infty S_\nu(\tau_\nu) \, e^{-\tau_\nu/\mu} \, \frac{d\tau_\nu}{\mu}$$

where $\mu = \cos\theta$ is the cosine of the angle between the line of sight and the outward surface normal. At disk center, $\mu = 1$ (looking straight down); at the limb, $\mu \to 0$ (looking tangentially).

The exponential weighting $e^{-\tau_\nu/\mu}$ means that the contribution to the emergent intensity is dominated by layers near $\tau_\nu \approx \mu$.

### The Approximation

If the source function varies approximately linearly with optical depth:

$$S_\nu(\tau_\nu) \approx S_\nu(0) + \tau_\nu \left.\frac{dS_\nu}{d\tau_\nu}\right|_0$$

then substituting into the formal solution gives:

$$\boxed{I_\nu(0, \mu) \approx S_\nu(\tau_\nu = \mu)}$$

This is the **Eddington-Barbier relation**: the emergent intensity at angle $\theta$ approximately equals the source function at optical depth $\mu = \cos\theta$.

At disk center ($\mu = 1$): we see to $\tau_\nu \approx 1$.
At the limb ($\mu \to 0$): we see shallower layers ($\tau_\nu \ll 1$).

> **Physical intuition**: We "see" to optical depth unity because a photon emitted at $\tau_\nu = 1$ has a $\sim 37\%$ chance of escaping, while a photon from $\tau_\nu = 3$ has only a $5\%$ chance, and from $\tau_\nu = 10$, essentially zero chance. The photosphere is literally the depth from which photons can escape — and that depth is $\tau_\nu \approx 1$.

### Effective Temperature

The Sun's effective temperature $T_\text{eff} = 5778$ K is defined by:

$$L_\odot = 4\pi R_\odot^2 \sigma T_\text{eff}^4$$

The effective temperature corresponds to the temperature at the layer where $\tau_{5000} \approx 2/3$ (the Rosseland mean optical depth, accounting for the angular integration):

$$B_\nu(T_\text{eff}) \approx S_\nu(\tau \approx 2/3)$$

The photosphere is thus identified with the layer at $\tau \approx 2/3$.

---

## 3. Limb Darkening

### The Observation

When we look at the solar disk, the center appears brighter than the edge (limb). This is **limb darkening**, and it is a direct consequence of the Eddington-Barbier relation combined with the fact that temperature decreases outward.

At disk center ($\mu = 1$): we see to $\tau \approx 1$, which is deeper and hotter.
At the limb ($\mu \approx 0$): we see to $\tau \approx \mu \ll 1$, which is shallower and cooler.

Since the source function (Planck function) increases with temperature, the center is brighter.

### The Limb Darkening Law

For a linear source function $S_\nu(\tau) = a + b\tau$, the Eddington-Barbier relation gives:

$$I_\nu(0, \mu) = a + b\mu$$

The limb darkening profile is therefore:

$$\frac{I_\nu(\mu)}{I_\nu(1)} = \frac{a + b\mu}{a + b} = 1 - u_\nu(1 - \mu)$$

where the **limb darkening coefficient** is:

$$u_\nu = \frac{b}{a + b}$$

In the visible ($\lambda \approx 5000$ A): $u \approx 0.6$, meaning the limb intensity is about $0.4$ times the disk-center intensity.

### Wavelength Dependence

Limb darkening is **stronger at shorter wavelengths** (bluer light) and **weaker at longer wavelengths** (redder light):

| Wavelength | $u_\nu$ | Limb/Center Ratio |
|------------|---------|-------------------|
| 4000 A (blue) | ~0.75 | ~0.25 |
| 5000 A (green) | ~0.60 | ~0.40 |
| 8000 A (near-IR) | ~0.45 | ~0.55 |
| 1.6 $\mu$m (H$^-$ opacity min.) | ~0.30 | ~0.70 |

The physical reason: at shorter wavelengths, the opacity is higher (the atmosphere is more opaque), so the photon mean free path is shorter. This means the difference in depth between disk center and limb observations is larger, sampling a larger temperature difference.

At 1.6 $\mu$m, the H$^-$ opacity has a minimum, so we see deeper into the atmosphere at all positions, and the temperature contrast between center and limb is reduced.

### Beyond Linear Limb Darkening

More accurate parameterizations include:

**Quadratic law**:
$$\frac{I(\mu)}{I(1)} = 1 - a(1-\mu) - b(1-\mu)^2$$

**Four-parameter nonlinear law** (Claret 2000):
$$\frac{I(\mu)}{I(1)} = 1 - \sum_{k=1}^{4} c_k (1 - \mu^{k/2})$$

These higher-order laws are important for precise modeling of transiting exoplanet light curves, where the planet's shadow sweeps across the stellar disk and the limb darkening profile directly affects the shape of the transit.

---

## 4. Granulation

### Convective Origin

Granulation is the direct surface manifestation of convection in the outer layers of the Sun. Each granule is the top of a convective cell: hot plasma rises in the bright center, spreads horizontally at the surface, cools, and sinks back down in the dark intergranular lanes.

### Properties

| Property | Value | Notes |
|----------|-------|-------|
| Typical size | ~1000 km ($1.4''$) | Varies from 300-2000 km |
| Lifetime | 8-10 minutes | Granules appear, evolve, and fragment |
| Upflow velocity (center) | 1-2 km/s | Hot, bright material rising |
| Downflow velocity (lanes) | 2-3 km/s | Cool, dark material sinking |
| Temperature contrast | ~200 K | ~3.5% of $T_\text{eff}$ |
| Filling factor (bright/dark) | ~60% / 40% | Asymmetric due to mass conservation |
| Number on disk | ~4 million | At any given moment |
| Intensity contrast (rms) | ~14% (at 500 nm) | Observed; ~25% intrinsic (degraded by seeing) |

### The Asymmetry of Convection

A striking feature of solar granulation is the **asymmetry** between upflows and downflows:

- **Upflows** (granule centers): broad, gentle, warm
- **Downflows** (intergranular lanes): narrow, fast, cool

This asymmetry arises from the strong density stratification of the photosphere. The pressure scale height at the photosphere is only $H_P \approx 150$ km — much smaller than the granule size. As rising plasma reaches the surface:
1. It expands rapidly (the density drops by a factor of $e$ every 150 km)
2. Radiative cooling at the surface (now optically thin) removes energy efficiently
3. The cooled gas is denser than its surroundings and sinks
4. Sinking gas is compressed (density increases), becomes denser and accelerates
5. Mass conservation requires the narrow downflows to be faster to carry the same mass flux as the broad upflows

This asymmetry produces a characteristic **spectral line asymmetry** — the C-shaped bisector of photospheric absorption lines — which is a key diagnostic of convective velocity fields.

### Granular Evolution

Granules are not static. They follow a lifecycle:
1. **Birth**: A new granule appears as a bright point or as a fragment of a splitting granule
2. **Growth**: Expands to typical size over 2-3 minutes
3. **Maturity**: Stable bright cell with dark boundary; lasts a few minutes
4. **Fragmentation or fading**: Large granules often split into 2-4 smaller ones; small ones fade

The **power spectrum** of granulation peaks at spherical harmonic degree $l \approx 1000$, corresponding to the characteristic granular size. There is also a secondary peak at $l \approx 120$ due to **mesogranulation** (a controversial intermediate scale, $\sim 5000$-$10000$ km, debated in the literature).

### Magnetic Flux in Intergranular Lanes

The convective flow sweeps magnetic flux into the intergranular lanes, where it is concentrated into **magnetic flux tubes** (or flux elements) with field strengths of $\sim 1$-$2$ kG. These flux tubes are visible as bright points (G-band bright points) because:
1. The strong magnetic field suppresses convection inside the tube
2. The tube is evacuated (lower density at same height compared to surroundings, due to magnetic pressure)
3. Lateral radiative heating from the hot surrounding walls makes the tube interior bright

These small-scale magnetic elements are the building blocks of the solar magnetic field and play a role in chromospheric and coronal heating.

---

## 5. Supergranulation

### Discovery and Properties

**Supergranulation** was discovered by Hart (1954) and studied in detail by Leighton, Noyes, and Simon (1962). It is a larger-scale convective pattern:

| Property | Value | Comparison with Granulation |
|----------|-------|-----------------------------|
| Size | ~30,000 km ($35''$) | 30x larger |
| Lifetime | 1-2 days | 150x longer |
| Horizontal velocity | 300-500 m/s | Dominant motion is horizontal |
| Vertical velocity | ~30 m/s (small) | 50x smaller than granulation |
| Number on disk | ~2500 | |

### The Chromospheric Network

The most important consequence of supergranulation is the **chromospheric network**: the supergranular flow sweeps magnetic flux to the boundaries of supergranular cells, creating a network of enhanced magnetic field. This network is clearly visible in chromospheric lines (Ca II K, H$\alpha$) and EUV images as a bright web-like pattern.

The network magnetic field has an average strength of $\sim 20$-$50$ G (but concentrated into flux elements of $\sim 1$-$2$ kG within the network). The network outlines the supergranular cell boundaries and is a fundamental feature of the quiet-Sun magnetic field.

### Connection to Deeper Convection

The physical origin of supergranulation is debated. Two main hypotheses:

1. **Deep convective cells**: Supergranulation is the surface signature of convective cells rooted at the depth of the He II ionization zone ($\sim 20{,}000$ km below the surface). The ionization of He$^+$ to He$^{2+}$ at this depth creates a local increase in opacity, similar to how H ionization drives granular convection.

2. **Inverse cascade or collective interaction**: Supergranulation emerges from the interaction and self-organization of smaller-scale convective motions, rather than being driven by a specific thermodynamic instability.

Numerical simulations have had difficulty reproducing supergranulation from first principles, and the debate continues.

### Giant Cells?

Theory predicts that convection should also produce **giant cells** with sizes comparable to the solar radius or the depth of the convection zone ($\sim 200{,}000$ km). These have been very difficult to detect, with only marginal evidence from helioseismology (Hathaway et al., 2013, reported large-scale velocity patterns). Their elusive nature is one of the open questions in solar convection.

---

## 6. Spectral Line Formation

### Fraunhofer Lines

The solar spectrum is crossed by thousands of dark **absorption lines** — the Fraunhofer lines, first catalogued by Joseph von Fraunhofer in 1814. These lines arise because atoms in the photosphere absorb light at specific wavelengths corresponding to their electronic transitions.

Key Fraunhofer lines:

| Line | Wavelength (A) | Species | Origin |
|------|----------------|---------|--------|
| H$\alpha$ | 6563 | H I | Balmer series, $n=2 \to 3$ |
| D$_1$, D$_2$ | 5896, 5890 | Na I | Sodium resonance doublet |
| K, H | 3934, 3969 | Ca II | Calcium ionized, resonance |
| G | 4308 | CH, Fe I | Molecular band + iron |
| b$_1$, b$_2$ | 5173, 5169 | Mg I | Magnesium triplet |

### Line Profile: The Voigt Function

A spectral line is not infinitely narrow. Its profile $\phi(\nu)$ is broadened by three mechanisms:

**1. Natural (Lorentzian) broadening**: Due to the finite lifetime of the excited state (Heisenberg uncertainty principle):

$$\phi_L(\nu) = \frac{\gamma/(4\pi^2)}{(\nu - \nu_0)^2 + (\gamma/4\pi)^2}$$

where $\gamma$ is the damping rate ($\sim 10^8$ s$^{-1}$ for typical optical transitions). This produces broad Lorentzian wings.

**2. Thermal (Gaussian) broadening**: Due to the Maxwell-Boltzmann distribution of atomic velocities along the line of sight:

$$\phi_G(\nu) = \frac{1}{\Delta\nu_D \sqrt{\pi}} \exp\left(-\frac{(\nu - \nu_0)^2}{\Delta\nu_D^2}\right)$$

where the **Doppler width** is:

$$\Delta\nu_D = \frac{\nu_0}{c} \sqrt{\frac{2k_BT}{m_\text{atom}}}$$

For iron (Fe I) at $T = 5800$ K: $\Delta\nu_D / \nu_0 \sim 2 \times 10^{-6}$, corresponding to a velocity width of $\sim 0.6$ km/s.

**3. Pressure (collisional) broadening**: Interactions with neighboring particles perturb the energy levels, producing additional Lorentzian broadening with a width that depends on the local density (pressure).

The combined line profile is the **Voigt function** — the convolution of Gaussian and Lorentzian profiles:

$$\phi_V(\nu) = \frac{a}{\pi^{3/2} \Delta\nu_D} \int_{-\infty}^{\infty} \frac{e^{-y^2}}{(u-y)^2 + a^2} dy$$

where $u = (\nu - \nu_0)/\Delta\nu_D$ is the normalized frequency displacement and $a = \gamma/(4\pi\Delta\nu_D)$ is the **damping parameter** (ratio of Lorentzian to Gaussian widths).

The Voigt profile has:
- A **Gaussian core** (dominated by thermal broadening at line center)
- **Lorentzian wings** (dominated by damping far from line center)

### Equivalent Width and the Curve of Growth

The **equivalent width** $W$ of a spectral line is the width of a rectangle with the same area as the line profile (measured in absorption):

$$W = \int \frac{I_c - I_\nu}{I_c} d\nu$$

where $I_c$ is the continuum intensity. It measures the total absorption in the line, independent of spectral resolution.

The **curve of growth** relates equivalent width to the number of absorbing atoms (column density $N$):

1. **Weak lines** (linear part): $W \propto N$. Each atom added contributes proportionally to the absorption.
2. **Intermediate lines** (flat part): $W \propto \sqrt{\ln N}$. The line center is saturated (completely dark); adding more atoms broadens the Gaussian core only logarithmically.
3. **Strong lines** (damping part): $W \propto \sqrt{N}$. The Lorentzian wings start to contribute significantly; the line grows as the square root of abundance.

The curve of growth is the primary tool for determining **element abundances** from spectral analysis.

### Solar Composition from Spectroscopy

By measuring equivalent widths of thousands of spectral lines and fitting them to model atmospheres, the photospheric composition has been determined:

| Element | Abundance $\log(N/N_H) + 12$ | Mass fraction |
|---------|-------------------------------|---------------|
| H | 12.00 (by definition) | 73.5% |
| He | 10.93 | 24.9% |
| O | 8.69 | 0.73% |
| C | 8.43 | 0.26% |
| Ne | 7.93 | 0.17% |
| Fe | 7.50 | 0.13% |
| N | 7.83 | 0.089% |
| Si | 7.51 | 0.069% |
| Mg | 7.60 | 0.058% |
| S | 7.12 | 0.040% |

Note: Helium cannot be measured from photospheric spectral lines (it requires much higher temperatures for excitation). The photospheric He abundance is inferred indirectly from helioseismology and from the initial abundance calibrated by the SSM.

---

## 7. The FIP Effect

### Definition

The **First Ionization Potential (FIP) effect** is one of the most puzzling and diagnostic features of solar physics. It refers to the observation that elements with **low FIP** ($< 10$ eV) — such as Fe, Mg, Si, Ca — are systematically **enhanced** (by a factor of 3-4) in the corona and solar wind relative to their photospheric abundances. Elements with **high FIP** ($> 10$ eV) — such as O, Ne, Ar, C — have coronal abundances close to their photospheric values.

| Element | FIP (eV) | Coronal/Photospheric Ratio |
|---------|----------|---------------------------|
| Fe | 7.9 | ~3.5 (enhanced) |
| Mg | 7.6 | ~3.5 (enhanced) |
| Si | 8.2 | ~3.5 (enhanced) |
| Ca | 6.1 | ~3.5 (enhanced) |
| O | 13.6 | ~1.0 (unchanged) |
| C | 11.3 | ~1.2 (slightly enhanced) |
| Ne | 21.6 | ~1.0 (unchanged) |
| Ar | 15.8 | ~1.0 (unchanged) |
| He | 24.6 | ~0.5 (depleted!) |

The dividing line is at approximately 10 eV. The FIP effect is observed consistently in:
- Closed magnetic structures (coronal loops, active regions)
- The slow solar wind
- Solar energetic particles from gradual events

Interestingly, in open magnetic structures (coronal holes) and the fast solar wind, the FIP effect is often **absent or reduced**.

### Physical Mechanism

The leading explanation is the **ponderomotive force model** (Laming, 2004, 2015):

In the chromosphere, where the temperature is $6000$-$10{,}000$ K, low-FIP elements are mostly **ionized** while high-FIP elements are mostly **neutral**. Alfven waves propagating from the corona into the chromosphere and reflecting create a **ponderomotive force** — a net force on ions due to the gradient of the wave amplitude:

$$\mathbf{F}_\text{pond} = \frac{q^2}{2m\omega^2} \nabla \langle E^2 \rangle$$

This force acts on ions but not on neutrals. In regions where the ponderomotive force is directed upward (toward the corona), it preferentially lifts the ions of low-FIP elements into the corona, while neutral high-FIP elements are left behind.

The model predicts:
- **Closed loops** (waves reflect at both footpoints): strong FIP effect
- **Open field lines** (waves propagate outward without reflection): weak FIP effect

This is consistent with observations.

### Diagnostic Use

The FIP effect is a powerful diagnostic tool:

1. **Coronal vs photospheric origin**: Material showing the FIP effect has spent time in the corona (long enough for the fractionation to develop). Material without the FIP effect is "fresh" from the photosphere.
2. **Solar wind source identification**: Slow solar wind (FIP-enhanced) likely originates from closed-field regions; fast solar wind (near-photospheric composition) from coronal holes.
3. **CME source tracking**: Coronal mass ejections carry coronal material with the FIP effect, which can be detected in situ at 1 AU, confirming their coronal origin.

### Inverse FIP Effect

In some stellar coronae (particularly active M-dwarf stars), an **inverse FIP effect** is observed: low-FIP elements are *depleted* relative to high-FIP elements. This is thought to arise when the ponderomotive force acts downward (into the chromosphere) rather than upward, which can occur for different wave reflection conditions. The Sun itself may exhibit a weak inverse FIP effect in certain magnetic configurations.

---

## 8. Photospheric Magnetic Fields

### Magnetic Field Measurement

Photospheric magnetic fields are measured using the **Zeeman effect**: a spectral line splits into multiple components in the presence of a magnetic field. For a simple Zeeman triplet, the splitting is:

$$\Delta\lambda = \frac{e \lambda^2 B}{4\pi m_e c} g_\text{eff}$$

where $g_\text{eff}$ is the effective Lande g-factor of the transition. For a typical photospheric line ($\lambda = 6173$ A, $g_\text{eff} = 2.5$) in a sunspot field ($B = 3000$ G):

$$\Delta\lambda \approx 0.070 \text{ A}$$

This is measurable with modern spectropolarimeters (e.g., HMI on SDO, DKIST's ViSP).

### Stokes Parameters

The full magnetic field vector is measured using **spectropolarimetry** — observing the four Stokes parameters ($I$, $Q$, $U$, $V$):

- $I$: total intensity
- $Q$, $U$: linear polarization (sensitive to the transverse magnetic field component)
- $V$: circular polarization (sensitive to the line-of-sight magnetic field component)

Stokes $V$ (circular polarization) is the easiest to measure and gives the line-of-sight (longitudinal) magnetic field. Full vector magnetograms require measuring all four Stokes parameters with high precision.

### The Quiet Sun Magnetic Field

Even outside active regions, the photosphere harbors a complex magnetic field:

- **Network field**: Concentrated in intergranular lanes at supergranular boundaries. Flux elements of $10^{18}$-$10^{19}$ Mx (Maxwell), field strengths of $\sim 1$-$2$ kG.
- **Internetwork field**: Weak, mixed-polarity field within supergranular cells. Field strengths of $\sim 10$-$100$ G. Detected through Hanle effect (modification of spectral line polarization by magnetic fields too weak for Zeeman detection).
- **Ephemeral regions**: Small bipolar magnetic regions that emerge and cancel on timescales of hours to days. They carry magnetic flux of $\sim 10^{19}$-$10^{20}$ Mx.

The total unsigned magnetic flux in the quiet Sun is comparable to that in active regions, though distributed in much smaller elements. This ubiquitous magnetic field plays a role in chromospheric and coronal heating.

---

## Summary

- **Radiative transfer** governs the photosphere: the transfer equation $dI_\nu/ds = -\kappa_\nu I_\nu + j_\nu$ describes how specific intensity changes along a ray
- **Optical depth** $\tau_\nu$ measures how opaque the atmosphere is; we see to $\tau_\nu \approx 1$ (the **Eddington-Barbier relation**)
- **Limb darkening** ($I(\mu)/I(1) \approx 1 - u(1-\mu)$) arises because the limb reveals cooler, shallower layers; it is stronger at shorter wavelengths ($u \approx 0.6$ in visible)
- **Granulation** (1000 km cells, 8-10 min lifetime) is the surface signature of convection, with asymmetric upflows and downflows
- **Supergranulation** (30,000 km cells, 1-2 day lifetime) organizes the chromospheric magnetic network
- **Spectral lines** (Voigt profiles) enable abundance analysis via the curve of growth; the Sun is 73.5% H, 24.9% He by mass
- The **FIP effect** (3-4x enhancement of low-FIP elements in corona) is explained by ponderomotive forces acting on ions in the chromosphere and serves as a diagnostic of coronal vs photospheric material

---

## Practice Problems

### Problem 1: Eddington-Barbier Relation

Consider a model photosphere where the source function varies linearly with optical depth: $S(\tau) = S_0 + S_1 \tau$, with $S_0 = 2 \times 10^{13}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$ and $S_1 = 5 \times 10^{13}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$. (a) Use the formal solution to compute the emergent intensity $I(0, \mu)$ exactly. (b) Compare with the Eddington-Barbier approximation $I \approx S(\tau = \mu)$. (c) Calculate the ratio $I(\text{limb})/I(\text{center})$ for $\mu_\text{limb} = 0.2$. (d) What limb darkening coefficient $u$ does this correspond to?

### Problem 2: Photon Mean Free Path

In the visible continuum ($\lambda = 5000$ A), the dominant opacity source in the photosphere is the H$^-$ ion (negative hydrogen). At $\tau_{5000} = 1$, the temperature is approximately 6400 K and the density is $\rho \approx 3 \times 10^{-4}$ kg/m$^3$. (a) If the opacity is $\kappa_{5000} \approx 0.26$ m$^2$/kg, calculate the photon mean free path. (b) Express this in km and as a fraction of the pressure scale height ($H_P \approx 150$ km). (c) Explain why the photosphere is so thin compared to the solar radius.

### Problem 3: Granulation Velocities

From the granulation properties (upflow $v_u = 1.5$ km/s, downflow $v_d = 2.5$ km/s, bright area fraction $f = 0.6$), and using mass continuity ($\rho_u v_u A_u = \rho_d v_d A_d$): (a) Calculate the density ratio $\rho_d/\rho_u$ between downflow and upflow regions. (b) If the typical density at the photosphere is $\rho_0 = 3 \times 10^{-4}$ kg/m$^3$, estimate the mass flux (kg/m$^2$/s) in the upflow. (c) Estimate the kinetic energy flux carried by the upflows and downflows. Compare with the radiative flux $F_\text{rad} = \sigma T_\text{eff}^4$.

### Problem 4: Curve of Growth

For a spectral line with Doppler width $\Delta\lambda_D = 0.05$ A and damping parameter $a = 0.01$: (a) On the linear part of the curve of growth, by what factor does the equivalent width increase if the abundance doubles? (b) On the flat (saturated) part, approximately by what factor does $W$ increase for a doubling of abundance? (c) On the damping part, by what factor? (d) Explain why spectral analysis is most precise using lines on the linear part of the curve of growth.

### Problem 5: FIP Effect Diagnostics

A coronal mass ejection (CME) is observed in situ at 1 AU. The measured Fe/O ratio is $0.20 \pm 0.02$ (by number). The photospheric Fe/O ratio is $0.065$. (a) Calculate the Fe enhancement factor. (b) Given that Fe has FIP = 7.9 eV and O has FIP = 13.6 eV, is this consistent with the FIP effect? (c) If the CME material showed Fe/O = 0.07, what would you conclude about its origin? (d) How could you use the FIP effect to distinguish between slow solar wind and CME material?

---

[← Previous: Helioseismology](03_Helioseismology.md) | [Next: Chromosphere and Transition Region →](05_Chromosphere_and_TR.md)

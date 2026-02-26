# Corona

## Learning Objectives

- Describe the major morphological components of the solar corona (active regions, quiet Sun, coronal holes, streamers)
- Articulate the coronal heating problem and why it does not violate thermodynamics
- Compare and contrast wave-heating (AC) and reconnection-based (DC) heating mechanisms
- Understand coronal loop physics and derive the RTV scaling laws
- Explain thermal non-equilibrium and its manifestation as coronal rain
- Perform and interpret Differential Emission Measure (DEM) analysis from multi-channel EUV observations
- Discuss the role of nanoflares in the Parker braiding model

---

## 1. Coronal Structure and Morphology

The solar corona -- the Sun's outermost atmospheric layer -- extends from the top of the transition region outward into interplanetary space, eventually merging with the solar wind. Despite being heated to over a million kelvin, the corona has an extremely low density ($n_e \sim 10^8$-$10^9$ cm$^{-3}$ at its base), making it roughly $10^{-12}$ times as bright as the photosphere in visible light. This is why it is normally invisible except during total solar eclipses or when observed with a coronagraph.

### 1.1 Active Region Corona

Active regions (ARs) are the brightest features in the corona, appearing as complex systems of **coronal loops** -- arched magnetic flux tubes filled with hot, dense plasma. These loops connect regions of opposite magnetic polarity in the photosphere and are visible because the plasma is confined by the magnetic field ($\beta \ll 1$ in the corona, where $\beta = p_{\text{gas}}/p_{\text{mag}}$).

Active region loops span a range of temperatures:

- **Cool loops**: $T \sim 1$ MK, bright in Fe IX/X (SDO/AIA 171 A channel)
- **Warm loops**: $T \sim 2$-$3$ MK, bright in Fe XII-XIV (AIA 193, 211 A)
- **Hot loops**: $T \sim 3$-$10$ MK, bright in Fe XVIII-XXIV (AIA 94, 131 A), often associated with flares or newly emerged flux

The loop plasma has typical electron densities of $n_e \sim 10^9$-$10^{10}$ cm$^{-3}$, and loop lengths range from $\sim 10$ Mm for compact loops to $> 500$ Mm for large-scale connections.

### 1.2 Quiet Sun Corona

The quiet Sun corona, away from active regions, appears as a diffuse emission at temperatures of $\sim 1$-$1.5$ MK. High-resolution observations reveal that even the "quiet" corona is not truly uniform but consists of faint loop-like structures, reflecting the underlying mixed-polarity magnetic carpet.

The quiet Sun coronal emission measure is roughly an order of magnitude lower than in active regions, reflecting the weaker magnetic field and presumably lower heating rate.

### 1.3 Coronal Holes

**Coronal holes** are extended regions where the corona appears dark in EUV and X-ray images. They correspond to regions of **open magnetic field** -- field lines that extend outward into the heliosphere rather than closing back to the solar surface.

Key properties of coronal holes:

- Temperature: $\sim 0.8$-$1.0$ MK (slightly cooler than quiet Sun)
- Density: 2-3 times lower than quiet Sun corona
- Source of the **fast solar wind** ($v \sim 600$-$800$ km/s)
- Polar coronal holes are persistent features during solar minimum; equatorial holes appear and disappear throughout the cycle
- Boundaries are typically marked by streamers and are regions of complex magnetic topology

### 1.4 Streamers

**Helmet streamers** are large, bright coronal structures that extend outward for several solar radii. They have a characteristic shape: a broad base over a closed magnetic arcade, tapering to a narrow stalk that extends into the heliosphere.

At the top of the streamer arcade, the field transitions from closed to open, forming a **current sheet** where oppositely directed magnetic fields are separated by a thin layer of enhanced current density. The streamer belt is the source of the **slow solar wind** ($v \sim 300$-$400$ km/s) and is associated with intermittent release of small flux ropes (blobs) that propagate outward along the current sheet.

### 1.5 Prominences and Filaments

**Prominences** (called **filaments** when viewed against the disk) are perhaps the most visually striking coronal structures: they consist of cool, dense material ($T \sim 10^4$ K, $n_e \sim 10^{10}$-$10^{11}$ cm$^{-3}$) suspended in the hot corona ($T > 10^6$ K) by magnetic forces. The density contrast between a prominence and the surrounding corona is a factor of $\sim 100$, and the temperature contrast is $\sim 100$ in the opposite direction.

Prominences are supported against gravity by the magnetic tension of a dipped or helical magnetic field configuration. They typically lie along **polarity inversion lines** (PILs) and can remain stable for days to weeks before either draining back to the surface or erupting as part of a coronal mass ejection (CME).

---

## 2. The Coronal Heating Problem

### 2.1 Statement of the Problem

The corona's temperature of $1$-$3$ MK is roughly 200 times hotter than the photosphere at 5778 K. Since the corona is "above" the photosphere (further from the nuclear energy source), this temperature inversion seems to defy common sense and has prompted the question: how can the outer atmosphere be hotter than the surface?

This is the **coronal heating problem**, one of the longest-standing unsolved problems in solar physics, first articulated by Grotrian (1939) and Edlen (1943) when they identified coronal emission lines as coming from highly ionized iron (Fe X-XIV), implying temperatures exceeding one million kelvin.

### 2.2 Why Thermodynamics Is Not Violated

It is important to emphasize that the coronal heating problem does **not** violate the second law of thermodynamics. The second law prohibits spontaneous heat flow from a cooler to a hotter body, but it does not prohibit energy flow from a cooler body to a hotter one if the energy is not in the form of heat.

The key distinction is:

- The photosphere radiates $\sim 6.3 \times 10^7$ W/m$^2$ as thermal radiation
- The corona requires only $\sim 300$ W/m$^2$ (quiet Sun) to $\sim 10^4$ W/m$^2$ (active regions) to maintain its temperature
- This energy is supplied as **mechanical and magnetic energy** (waves, field stresses) generated in the convection zone, not as thermal radiation from the photosphere

The analogy is a campfire: the air above the fire can be hotter than the glowing embers below because convective and radiative energy transport carries energy upward from the combustion zone. The corona is heated by non-thermal energy that originates in the kinetic and magnetic energy of convective motions.

### 2.3 Energy Budget

The energy flux required to maintain the corona against radiative and conductive losses, plus the energy carried away by the solar wind, is:

| Region | Required flux (W/m$^2$) |
|--------|------------------------|
| Quiet Sun corona | $\sim 300$ |
| Coronal hole | $\sim 800$ (includes solar wind) |
| Active region corona | $\sim 10^4$ |

These values, while tiny compared to the photospheric luminous flux, must be supplied continuously and deposited specifically in the corona. The challenge is to identify the mechanism that transports energy from the convection zone to the corona and dissipates it there with the correct spatial and temporal distribution.

### 2.4 Two Paradigms: AC vs. DC Heating

Coronal heating theories are traditionally divided into two broad categories:

**AC (wave) heating**: energy is carried upward by magnetohydrodynamic waves generated by photospheric convective motions. The challenge is to get the waves to dissipate their energy in the corona rather than reflecting or passing through. The wave periods are comparable to or shorter than the Alfven travel time along a loop.

**DC (magnetic stress) heating**: photospheric motions slowly (compared to the Alfven travel time) braid and stress the coronal magnetic field, building up free energy in the form of electric currents. This energy is released in discrete events (reconnection, current sheet dissipation). The Parker nanoflare model is the canonical DC heating theory.

In reality, the distinction between AC and DC is not sharp -- it depends on the ratio of the driving timescale to the Alfven travel time along the coronal loop.

---

## 3. Wave Heating Mechanisms

### 3.1 Alfven Wave Propagation

Alfven waves -- transverse oscillations of magnetic field lines -- are the most promising wave heating candidate because they can carry large energy fluxes and are less susceptible to reflection than acoustic waves. The energy flux carried by an Alfven wave is:

$$F_A = \rho \langle \delta v^2 \rangle v_A$$

where $\rho$ is the mass density, $\delta v$ is the wave velocity amplitude, and $v_A = B/\sqrt{\mu_0 \rho}$ is the Alfven speed.

For photospheric conditions ($\rho \sim 2 \times 10^{-4}$ kg/m$^3$, $\delta v \sim 1$ km/s, $v_A \sim 1$ km/s), this gives $F_A \sim 2 \times 10^5$ W/m$^2$ -- more than enough to heat the corona.

### 3.2 The Reflection Problem

The main challenge for wave heating is getting Alfven waves from the photosphere into the corona. The Alfven speed increases by a factor of $\sim 10^3$ from the photosphere ($\sim 1$ km/s) to the corona ($\sim 1000$ km/s), with the steepest increase occurring in the transition region. This gradient in Alfven speed acts like a barrier, reflecting a large fraction of upward-propagating waves.

The reflection coefficient depends on the ratio of the wavelength to the scale length of the Alfven speed variation. Short-wavelength waves (high frequency) are transmitted more efficiently, but they also tend to have smaller amplitudes. The net result is that only a fraction of the photospheric wave energy flux reaches the corona.

### 3.3 Phase Mixing

Even if Alfven waves reach the corona, they must still be dissipated to heat the plasma. In a uniform medium, Alfven waves propagate without dissipation (in ideal MHD). **Phase mixing** provides a dissipation mechanism in non-uniform media.

Consider a bundle of magnetic field lines with different Alfven speeds (due to density or field strength variations across the bundle). Alfven waves on neighboring field lines, initially in phase, gradually become out of phase as they propagate. This creates increasingly small spatial scales across the magnetic field:

$$l_\perp(t) \sim l_\perp(0) \left(\frac{L}{v_A \, t_{\text{phase}}}\right)^{-1}$$

When $l_\perp$ becomes comparable to the resistive or viscous dissipation scale, the wave energy is converted to heat. The timescale for this process is:

$$t_{\text{pm}} \sim \left(\frac{6 \eta}{l_\perp^2 \, (\partial v_A / \partial x)^2}\right)^{1/3}$$

where $\eta$ is the magnetic diffusivity and $\partial v_A / \partial x$ is the transverse gradient of the Alfven speed.

### 3.4 Resonant Absorption

**Resonant absorption** is another mechanism for dissipating wave energy in non-uniform plasmas. When a kink-mode oscillation (a transverse displacement of the entire flux tube) encounters a resonant layer where the local Alfven frequency matches the driving frequency, energy is transferred from the collective kink mode to localized azimuthal Alfven oscillations.

The process can be summarized in three stages:

1. **Kink mode excitation**: photospheric motions excite transverse oscillations of coronal loops
2. **Resonant absorption**: at the loop boundary (where the density transitions from internal to external), the kink mode resonantly excites azimuthal Alfven waves, transferring energy to increasingly small scales
3. **Kelvin-Helmholtz instability**: the velocity shear at the resonant layer triggers KHI, generating turbulent eddies that cascade energy to dissipation scales

This mechanism is attractive because it naturally generates small scales from large-scale driving. Damping times of kink oscillations in coronal loops (observed to be $\sim 2$-$4$ periods) are consistent with resonant absorption theory.

### 3.5 Observational Evidence for Wave Heating

Modern space-based observatories have provided mounting evidence for waves in the corona:

- **CoMP** (Coronal Multi-channel Polarimeter): detected propagating Alfven-like waves throughout the corona with velocity amplitudes of $\sim 0.3$ km/s, carrying an estimated energy flux of $\sim 100$ W/m$^2$ -- insufficient to heat the corona alone, but possibly an underestimate due to line-of-sight integration
- **SDO/AIA**: transverse oscillations of coronal loops with periods of $\sim 2$-$10$ minutes and velocity amplitudes of $\sim 1$-$20$ km/s
- **Hinode/EIS**: non-thermal line widths in the corona ($\sim 20$-$30$ km/s) consistent with unresolved wave motions

Whether the observed wave energy flux is sufficient to heat the corona remains an open question. Many measurements provide only lower limits due to the difficulty of detecting waves along the line of sight.

---

## 4. Nanoflare Heating

### 4.1 Parker's Braiding Model

In a seminal 1988 paper, Eugene Parker proposed that the corona is heated by a multitude of tiny reconnection events called **nanoflares**. The basic idea is elegant:

1. Photospheric convective motions continuously shuffle the footpoints of coronal magnetic field lines
2. These random footpoint motions **braid** the coronal field, creating tangential discontinuities (current sheets) between adjacent flux tubes that have been wrapped around each other
3. Reconnection at these current sheets releases magnetic energy in small, impulsive bursts

Parker argued that the formation of current sheets (tangential discontinuities) is topologically inevitable when smooth footpoint motions are applied to a 3D magnetic field -- the field cannot remain smooth and must develop discontinuities.

### 4.2 Nanoflare Energetics

The term "nanoflare" refers to the energy scale: approximately $10^{24}$ ergs ($10^{17}$ J), which is about $10^{-9}$ times the energy of a large solar flare ($\sim 10^{32}$ ergs). To put this in perspective:

| Event | Energy (erg) | Frequency |
|-------|-------------|-----------|
| Large flare | $10^{31}$-$10^{32}$ | ~10/year (solar max) |
| Microflare | $10^{27}$-$10^{29}$ | ~1000/day |
| Nanoflare | $10^{24}$-$10^{26}$ | ~millions/second? |

### 4.3 The Frequency Distribution Question

A crucial question is whether nanoflares occur frequently enough, and with the right energy distribution, to heat the corona. If the frequency distribution of flare energies follows a power law:

$$\frac{dN}{dE} \propto E^{-\alpha}$$

then the total heating rate is dominated by the smallest events if $\alpha > 2$, and by the largest events if $\alpha < 2$.

For nanoflares to heat the corona, we need $\alpha > 2$ so that the energy budget is dominated by the most numerous, smallest events. Observational measurements of $\alpha$ from microflare and nanoflare statistics have yielded values in the range $1.5$-$2.5$, tantalizingly close to the critical threshold but not yet conclusive.

### 4.4 Observational Signatures

If the corona is heated by nanoflares, we expect specific observational signatures:

- **Hot plasma component**: each nanoflare heats a small strand to temperatures well above the average ($> 5$ MK), so the DEM should show a faint high-temperature tail. Searches for this hot component with RHESSI, NuSTAR, and SDO/AIA have found hints but remain inconclusive.
- **Impulsive brightenings**: individual nanoflares might be detectable as small, brief EUV or X-ray brightenings. The term "campfire" was coined by Solar Orbiter/EUI scientists for small-scale brightenings detected at high resolution.
- **Non-equilibrium ionization**: if heating is impulsive, the ionization state may lag behind the temperature changes, affecting spectral diagnostics.

---

## 5. Coronal Loop Physics

### 5.1 Why Loops?

The corona is structured by the magnetic field because the magnetic pressure vastly exceeds the gas pressure (low $\beta$). Plasma is effectively confined to move along field lines, and thermal conduction is $\sim 10^{12}$ times more efficient along the field than across it. Consequently, each magnetic flux tube behaves as an essentially independent atmosphere, and we observe the corona as a collection of individual loops.

### 5.2 Hydrostatic Loop Structure

For a static, symmetric coronal loop in hydrostatic equilibrium along the field, the pressure varies with height as:

$$p(h) = p_0 \exp\left(-\frac{h}{H}\right)$$

where $H$ is the pressure scale height:

$$H = \frac{2 k_B T}{m_p g} \approx 50 \left(\frac{T}{10^6 \text{ K}}\right) \text{ Mm}$$

Here the factor of 2 accounts for the equal electron and proton contributions to pressure ($p = 2 n k_B T$ for a fully ionized hydrogen plasma).

For a loop with apex height comparable to or less than $H$, the density variation along the loop is modest and the loop appears relatively uniform. For loops taller than $H$, the density drops significantly toward the apex, and the loop top may become gravitationally stratified.

### 5.3 RTV Scaling Laws

In 1978, Rosner, Tucker, and Vaiana (RTV) derived fundamental scaling laws for static coronal loops by solving the energy balance equation along a loop:

$$\frac{d}{ds}\left(\kappa_0 T^{5/2} \frac{dT}{ds}\right) = n_e^2 \Lambda(T) - E_H$$

where $s$ is the coordinate along the loop, $\kappa_0 T^{5/2}$ is the Spitzer thermal conductivity, $\Lambda(T)$ is the radiative loss function, and $E_H$ is the volumetric heating rate.

For a uniformly heated, symmetric loop with half-length $L$, boundary conditions $T = T_0$ at the footpoints and $dT/ds = 0$ at the apex (by symmetry), the RTV analysis yields two scaling laws relating the loop apex temperature $T_{\text{max}}$, the base pressure $p_0$, and the loop half-length $L$:

$$T_{\text{max}} \approx 1.4 \times 10^3 \left(p_0 \, L\right)^{1/3}$$

$$E_H \approx 10^5 \, p_0^{7/6} \, L^{-5/6}$$

where $T_{\text{max}}$ is in kelvin, $p_0$ in dyn/cm$^2$ (CGS), $L$ in cm, and $E_H$ in erg cm$^{-3}$ s$^{-1}$.

**Physical interpretation**: the first scaling law says that hotter loops must be either longer, denser, or both. A loop that is twice as long needs to be at a higher temperature to conduct enough energy from the corona to the footpoints. The second law relates the required heating rate to the loop parameters.

### 5.4 Implications of RTV Scaling

The RTV laws have been remarkably successful at organizing coronal observations:

- **Short, bright loops** in active region cores ($L \sim 20$ Mm, $T \sim 3$-$5$ MK): consistent with high pressure, strong heating
- **Long quiet Sun loops** ($L \sim 200$ Mm, $T \sim 1$-$1.5$ MK): consistent with lower pressure, weaker heating
- **Overdense loops**: many observed loops are denser than predicted by RTV, suggesting they are not in static equilibrium but are instead dynamically evolving (e.g., cooling from a higher temperature state following impulsive heating)

The widespread departure from static equilibrium is itself a clue about the heating mechanism: it suggests that heating is often impulsive (nanoflare-like) rather than steady.

---

## 6. Thermal Non-Equilibrium and Coronal Rain

### 6.1 The Concept of Thermal Non-Equilibrium

**Thermal non-equilibrium (TNE)** is a remarkable phenomenon that occurs in coronal loops when the heating is strongly concentrated near the footpoints. In such a configuration, the heating is deposited deep in the legs of the loop, and thermal conduction must transport the energy upward to the loop apex.

For sufficiently stratified (footpoint-concentrated) heating, the energy balance at the loop apex cannot be maintained: the conductive flux from below is insufficient to balance the radiative losses at the apex. The system has no static equilibrium solution, and instead undergoes **limit cycles** of heating and catastrophic cooling.

The cycle proceeds as follows:

1. **Heating phase**: footpoint heating drives evaporation, filling the loop with hot, dense plasma
2. **Thermal equilibrium attempt**: the loop tries to reach a steady state, but the apex cooling exceeds the conductive energy supply
3. **Catastrophic cooling**: the apex temperature drops, increasing radiative losses (since $\Lambda(T)$ increases as $T$ decreases in the $10^5$-$10^6$ K range), triggering a runaway cooling instability
4. **Condensation**: plasma cools to chromospheric temperatures ($\sim 10^4$ K), forming dense blobs
5. **Draining and recovery**: the condensations drain down the loop legs under gravity, the loop empties, and the cycle restarts

### 6.2 The Thermal Instability Criterion

The key physics driving the catastrophic cooling is the **thermal instability** (also known as radiative instability or condensation instability). In an optically thin plasma, the energy equation for a perturbation can be written:

$$\rho c_v \frac{\partial T}{\partial t} = E_H - n_e^2 \Lambda(T)$$

The condition for instability is that the cooling function increases faster than the heating as the temperature decreases:

$$\frac{\partial}{\partial T}\left[n_e^2 \Lambda(T)\right] < \frac{\partial E_H}{\partial T}$$

In the temperature range $10^5 < T < 10^7$ K, the radiative loss function $\Lambda(T)$ has a complex structure with peaks (primarily from line emission of C, O, N, Fe). In parts of this range, $\Lambda(T)$ increases as $T$ decreases, meaning that a cooling perturbation leads to more cooling -- a positive feedback loop.

If the heating is steady and does not increase to compensate, the cooling runs away and the plasma condenses.

### 6.3 Coronal Rain

The dramatic observational manifestation of thermal non-equilibrium is **coronal rain** -- cool, dense blobs of plasma ($T \sim 10^4$ K, $n_e \sim 10^{10}$-$10^{11}$ cm$^{-3}$) falling along coronal loop legs under gravity.

Observational properties of coronal rain:

- **Fall speeds**: 50-150 km/s (below free-fall, indicating gas pressure and magnetic curvature effects)
- **Blob sizes**: 300-700 km wide, 700-3000 km long
- **Observed in**: H-alpha (656.3 nm), Ca II (854.2 nm), SDO/AIA 304 A (He II, $T \sim 5 \times 10^4$ K)
- **Occurrence**: primarily in active region loops, but also in quiet Sun

Coronal rain provides a powerful diagnostic for the heating mechanism: its very existence implies that the heating in those particular loops is strongly stratified (concentrated at the footpoints). This is an important constraint, as it rules out uniform heating for loops exhibiting TNE.

### 6.4 Long-Period EUV Pulsations

The TNE limit cycles manifest as **long-period EUV pulsations** with periods of $\sim 2$-$16$ hours, observed as periodic intensity variations in coronal EUV channels. During the heating phase, the loop brightens in hot channels (AIA 94, 335 A); during the cooling phase, it brightens sequentially in progressively cooler channels (211 $\to$ 193 $\to$ 171 A), reflecting the decreasing temperature. The cooling phase often ends with coronal rain visible in 304 A and chromospheric lines.

These pulsations are now recognized as a common feature of active region loops and provide a new window into the temporal and spatial distribution of coronal heating.

---

## 7. Differential Emission Measure (DEM) Analysis

### 7.1 Definition and Physical Meaning

The **Differential Emission Measure** quantifies how much emitting plasma exists at each temperature along a line of sight. For an optically thin plasma, the DEM is defined as:

$$\text{DEM}(T) = n_e^2 \frac{dh}{dT} \quad [\text{cm}^{-5} \text{ K}^{-1}]$$

where the integral is along the line of sight $h$. The total emission measure over a temperature range is:

$$\text{EM} = \int_{T_1}^{T_2} \text{DEM}(T) \, dT = \int n_e^2 \, dh \quad [\text{cm}^{-5}]$$

### 7.2 Relation to Observed Intensities

The observed intensity in a given spectral line or broadband channel $i$ is:

$$I_i = \int_0^\infty R_i(T) \, \text{DEM}(T) \, dT$$

where $R_i(T)$ is the **temperature response function** of the instrument channel, which depends on atomic physics (ionization equilibrium, excitation rates, element abundances) and the instrument properties (effective area, filter transmission).

For SDO/AIA, the six coronal EUV channels sample different temperature ranges:

| Channel (A) | Peak Temperature (MK) | Primary Ion |
|-------------|----------------------|-------------|
| 94 | 6.3 (also 1.0) | Fe XVIII (Fe X) |
| 131 | 10.0 (also 0.4) | Fe XXI (Fe VIII) |
| 171 | 0.8 | Fe IX |
| 193 | 1.5 (also 16) | Fe XII (Fe XXIV) |
| 211 | 2.0 | Fe XIV |
| 335 | 2.5 | Fe XVI |

### 7.3 The Inversion Problem

Determining DEM$(T)$ from a set of observed intensities $\{I_i\}$ is an **inverse problem**: given the measurements and the response functions, we want to recover the underlying temperature distribution. This problem is mathematically **ill-posed** because:

1. We have a finite number of channels ($N = 6$ for AIA) but want a continuous function DEM$(T)$
2. The response functions are broad and overlapping, so different DEM shapes can produce similar intensities
3. Observational noise further limits the information content

The standard approach is **regularized inversion**: find the DEM that best fits the observations while satisfying some smoothness or positivity constraint. Common methods include:

- **Tikhonov regularization**: minimize $\chi^2 + \lambda \int |\nabla \text{DEM}|^2 dT$
- **Monte Carlo sampling**: explore the space of DEMs consistent with the data
- **Basis pursuit / sparse methods**: represent DEM as a sum of basis functions and enforce sparsity

### 7.4 Interpreting DEM Results

A DEM analysis reveals:

- **Peak temperature**: the dominant temperature of the plasma. For active region loops, typically 2-5 MK; for quiet Sun, 1-1.5 MK.
- **DEM width**: a broad DEM indicates multi-thermal plasma (many temperatures along the line of sight), while a narrow DEM indicates nearly isothermal plasma. Most coronal structures are multi-thermal.
- **High-temperature tail**: excess emission above $\sim 5$ MK may indicate nanoflare heating (plasma recently heated to high temperatures and now cooling).
- **DEM slope at low temperatures**: the power-law slope of DEM$(T)$ below the peak constrains the relationship between temperature and density in the transition region.

DEM analysis is one of the most powerful tools in solar coronal physics, providing quantitative constraints on the heating mechanism from readily available EUV observations.

---

## Practice Problems

**Problem 1: Coronal Energy Budget**

The quiet Sun corona requires an energy input of approximately 300 W/m$^2$. (a) What fraction of the total solar luminous flux ($L_\odot = 3.83 \times 10^{26}$ W, $R_\odot = 6.96 \times 10^8$ m) does this represent? (b) If this energy is supplied by Alfven waves with velocity amplitude $\delta v$ in the photosphere ($\rho = 2 \times 10^{-4}$ kg/m$^3$, $v_A = 1$ km/s), and only 1% of the wave energy flux reaches the corona, what $\delta v$ is required? (c) Is this velocity amplitude consistent with observed photospheric motions?

**Problem 2: RTV Scaling Laws**

A coronal loop has a half-length of $L = 50$ Mm and a base pressure of $p_0 = 0.5$ dyn/cm$^2$. (a) Use the RTV scaling law to predict the apex temperature. (b) Calculate the corresponding pressure scale height and determine whether the loop is gravitationally stratified (compare $L$ to $\pi H / 2$, the effective scale height along a semicircular loop). (c) If the loop length doubles while the pressure remains the same, by what factor does the temperature change?

**Problem 3: Nanoflare Frequency Distribution**

Suppose the frequency distribution of small-scale heating events follows $dN/dE = A E^{-\alpha}$ between $E_{\min} = 10^{24}$ erg and $E_{\max} = 10^{28}$ erg. (a) For $\alpha = 1.8$, calculate the total energy release rate $\dot{E} = \int E (dN/dE) dE$. Show that the largest events dominate. (b) Repeat for $\alpha = 2.3$ and show that the smallest events dominate. (c) Find the critical $\alpha$ value and explain its physical significance for coronal heating.

**Problem 4: Thermal Non-Equilibrium**

Consider a coronal loop of half-length $L = 80$ Mm with footpoint-concentrated heating that deposits most energy below an altitude of $h_0 = 10$ Mm. (a) Using the RTV scaling law, estimate the equilibrium temperature if the loop were uniformly heated at the same total rate. (b) Qualitatively explain why footpoint-concentrated heating can lead to thermal non-equilibrium at the loop apex. (c) If coronal rain blobs fall at 100 km/s from the loop apex along a semicircular loop, estimate the fall time. Compare this to typical TNE cycle periods of 2-10 hours.

**Problem 5: DEM Inversion**

SDO/AIA observes a coronal region in 6 channels, with measured intensities $I_i$ and response functions $R_i(T)$. (a) Write the integral equation relating $I_i$ to DEM$(T)$. (b) Explain why the inversion is ill-posed. (c) A DEM analysis reveals a peak at 1.5 MK with a high-temperature tail extending to 8 MK. Propose two physical scenarios that could produce this DEM shape (one involving steady heating, one involving impulsive nanoflares). (d) What additional observational test could distinguish between these scenarios?

---

**Previous**: [Chromosphere and Transition Region](./05_Chromosphere_and_Transition_Region.md) | **Next**: [Solar Magnetic Fields](./07_Solar_Magnetic_Fields.md)

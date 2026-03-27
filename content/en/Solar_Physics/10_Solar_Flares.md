# Solar Flares

## Learning Objectives

- Classify solar flares by GOES soft X-ray class and understand the logarithmic scale
- Describe the standard CSHKP flare model and its sequential stages
- Explain multi-wavelength flare emissions from radio through gamma-ray
- Understand the primary particle acceleration mechanisms in flares
- Analyze flare energetics and the partition of magnetic free energy
- Apply the thick-target model to relate electron spectra to hard X-ray emission
- Discuss quasi-periodic pulsations and their diagnostic potential

---

## 1. Flare Classification

Solar flares are sudden, intense brightenings on the Sun caused by the rapid release of magnetic energy stored in the coronal field. They are classified primarily by their peak soft X-ray intensity as measured by the GOES (Geostationary Operational Environmental Satellite) series.

### 1.1 GOES X-ray Classification

The GOES X-ray sensor monitors the Sun in the 1–8 Angstrom (0.1–0.8 nm) wavelength band. Flares are classified on a logarithmic scale:

| Class | Peak Flux (W/m$^2$) |
|-------|---------------------|
| A | $< 10^{-7}$ |
| B | $10^{-7}$ – $10^{-6}$ |
| C | $10^{-6}$ – $10^{-5}$ |
| M | $10^{-5}$ – $10^{-4}$ |
| X | $\geq 10^{-4}$ |

Each class is subdivided from 1 to 9 (e.g., C3.2, M7.1, X2.5), except the X class, which has no upper limit — the largest recorded events include X28+ (estimated X45) on November 4, 2003, and X9.3 on September 6, 2017.

The key intuition is that each letter represents a factor of 10 in X-ray intensity. An X1 flare is 10 times brighter than an M1 and 10,000 times brighter than an A1 in soft X-rays.

### 1.2 Occurrence Rates

Flare frequency follows a power-law distribution:

$$\frac{dN}{dE} \propto E^{-\alpha}$$

with $\alpha \approx 1.5$–$1.8$ for total energy $E$. At solar maximum, the Sun produces several C-class flares per day, a few M-class flares per week, and a few X-class flares per year. The most intense flares are genuinely rare — X10+ events occur perhaps once per decade.

### 1.3 Temporal Classification

Flares are also classified by their temporal profile:
- **Impulsive flares**: Short duration ($< 10$ minutes), compact, strong hard X-ray emission, associated with electron acceleration. Typically confined events (no CME).
- **Long-duration events (LDEs)**: Duration of hours, gradual decay, large arcade of post-flare loops, usually associated with CMEs. Also called "eruptive flares."

### 1.4 H-alpha Classification

Before the space age, flares were classified by their optical brightness in H-alpha (656.3 nm):
- **Importance**: S (subflare), 1, 2, 3, 4 based on area of the brightening
- **Brightness qualifier**: f (faint), n (normal), b (brilliant)

A typical designation like "2B" means a flare of importance 2 (100–250 millionths of the solar hemisphere) with brilliant brightness.

---

## 2. The Standard Flare Model (CSHKP)

The standard model of eruptive flares is named after the four researchers who independently contributed to its development: Carmichael (1964), Sturrock (1966), Hirayama (1974), and Kopp & Pneuman (1976). While simplified, it captures the essential physics and remains the framework within which most flare observations are interpreted.

### 2.1 Pre-Flare Configuration

Before the flare, the coronal magnetic field above a polarity inversion line (PIL) is stressed by photospheric motions — shearing, converging flows, or flux emergence. Free magnetic energy accumulates in the form of a sheared arcade or, more commonly, a magnetic flux rope lying above the PIL. The configuration is in a state of quasi-static equilibrium that is approaching a threshold of instability.

### 2.2 Eruption Onset

The equilibrium is broken by one of several mechanisms (torus instability, kink instability, or breakout reconnection — see Lesson 11 on CMEs). The flux rope begins to rise, stretching the overlying field lines into a vertical current sheet beneath it.

### 2.3 Magnetic Reconnection

Reconnection commences in the current sheet formed below the rising flux rope. This is the central engine of the flare:

- Oppositely directed magnetic field lines are brought together at the inflow speed $v_{\text{in}}$.
- They reconnect, converting magnetic energy into kinetic energy, thermal energy, and particle acceleration.
- The reconnection rate is characterized by the inflow Alfven Mach number $M_A = v_{\text{in}}/v_A \sim 0.01$–$0.1$ (consistent with Petschek-type fast reconnection rather than the much slower Sweet-Parker rate).
- The reconnection outflow speed approaches the local Alfven speed $v_A = B/\sqrt{4\pi\rho}$, typically $10^3$–$10^4$ km/s in the corona.

For a detailed treatment of magnetic reconnection physics, see MHD Lesson 06 (Magnetic Reconnection), Section 1.

### 2.4 Particle Acceleration and Chromospheric Response

Reconnection accelerates electrons and ions to high energies. These particles stream down along the newly reconnected field lines toward the chromosphere at nearly the speed of light:

1. **Hard X-ray footpoints**: Accelerated electrons (10–100+ keV) collide with the dense chromospheric plasma, producing hard X-ray bremsstrahlung emission at the footpoints of the flare loops.
2. **Chromospheric evaporation**: The impacting particles and conducted heat deposit enormous energy in the chromosphere, raising the temperature to $>10^7$ K. The heated plasma expands explosively upward into the coronal loop — this process is called chromospheric evaporation (despite being more accurately a hydrodynamic expansion than a phase change).
3. **Soft X-ray loops**: The evaporated hot plasma fills the newly reconnected loops, making them shine brightly in soft X-rays and EUV.

### 2.5 Post-Flare Phase

As reconnection proceeds, it works its way upward through the current sheet, reconnecting field lines at progressively greater heights:

- **Growing loop arcade**: Post-flare loops appear to grow upward over time, with the hottest, most recently reconnected loops at the top.
- **Flare ribbons**: The chromospheric footpoints of the reconnecting field lines form two bright ribbons on either side of the PIL. As reconnection proceeds upward, the ribbons separate — this is a direct observable signature of the reconnection process.
- **Gradual decay**: The soft X-ray emission decays over hours as the hot loops cool by conduction and radiation.

### 2.6 Limitations of the Standard Model

The CSHKP model is two-dimensional and highly idealized. Real flares involve:
- Three-dimensional field geometry (shear along the PIL, guide field effects)
- Multiple reconnection sites and fragmented current sheets
- Turbulence in the reconnection region
- Complex interactions between the erupting flux rope and the ambient field

---

## 3. Multi-Wavelength Observations

Solar flares emit across the entire electromagnetic spectrum, from radio waves to gamma-rays. Each wavelength regime reveals different physical processes and atmospheric layers.

### 3.1 Radio Emission

Solar radio emission during flares is extraordinarily rich and diagnostic:

- **Type III bursts** (MHz to kHz, fast drift): Produced by electron beams streaming outward along open field lines at ~$c/3$. The emission frequency drifts from high to low as the beam propagates into regions of decreasing plasma density ($f_p \propto n_e^{1/2}$). Drift rate: ~100 MHz/s at metric wavelengths.

- **Type II bursts** (slow drift): Produced at the plasma frequency ahead of a coronal/interplanetary shock (usually CME-driven). Drift rate: ~1 MHz/s. The fundamental and harmonic lanes are characteristic.

- **Gyrosynchrotron emission** (GHz, microwave): Mildly relativistic electrons (100 keV – few MeV) spiraling in the coronal magnetic field. The spectrum peaks at a few GHz, with the peak frequency depending on $B$ and the electron energy. This is the primary diagnostic of energetic electrons trapped in flare loops.

### 3.2 Optical and UV

- **H-alpha ribbons**: The classic flare signature — bright ribbons tracing the chromospheric footpoints of reconnecting field lines. The ribbons separate over time as reconnection proceeds upward.

- **White-light flares**: Continuum enhancement visible in integrated sunlight. Extremely energetic events — even the Carrington event of 1859, the largest recorded flare, was observed as a white-light brightening. The emission mechanism is debated (hydrogen recombination continuum, H$^-$ opacity enhancement, or direct heating of the photosphere).

- **UV/EUV (SDO/AIA)**: The Atmospheric Imaging Assembly on SDO provides multi-temperature imaging at 7 EUV and 2 UV channels, revealing the thermal structure of flare loops and ribbons with 12-second cadence and 0.6 arcsecond pixels.

### 3.3 Soft X-rays

Soft X-rays (0.1–10 keV, or 1–100 Angstroms) are produced by thermal bremsstrahlung and line emission from the hot ($10^6$–$10^8$ K) plasma filling flare loops. The GOES 1–8 Angstrom channel provides the standard flare classification. Imaging instruments (Hinode/XRT, formerly Yohkoh/SXT) show the morphology and evolution of flare loop arcades.

The thermal energy content of the soft X-ray emitting plasma is:

$$E_{\text{th}} = 3n_e k_B T V$$

where $n_e$ is the electron density, $T$ is the temperature, and $V$ is the volume. For a large flare: $n_e \sim 10^{11}$ cm$^{-3}$, $T \sim 2 \times 10^7$ K, $V \sim (10^9 \text{ cm})^3$ gives $E_{\text{th}} \sim 10^{31}$ erg.

### 3.4 Hard X-rays

Hard X-rays (>10 keV) are the smoking gun of particle acceleration. They are produced by non-thermal bremsstrahlung: accelerated electrons collide with ambient ions and emit photons with a power-law spectrum reflecting the electron energy distribution.

Key instruments: RHESSI (2002–2018), now Solar Orbiter/STIX (2020–present), and ASO-S/HXI.

The spatial distribution of hard X-ray emission reveals the flare geometry:
- **Footpoint sources**: Thick-target emission from accelerated electrons hitting the dense chromosphere (dominant at higher energies, >30 keV).
- **Coronal source**: Thin-target emission or thermal emission from the reconnection region and top of flare loops (dominant at lower energies, <25 keV).

### 3.5 Gamma-Rays

The most energetic flare emissions extend into the gamma-ray regime:

- **Nuclear de-excitation lines** (1–8 MeV): Accelerated protons and ions collide with ambient nuclei, exciting nuclear energy levels. The 2.223 MeV line from neutron capture on hydrogen is particularly prominent. Other lines include $^{12}$C at 4.44 MeV, $^{16}$O at 6.13 MeV, and $^{20}$Ne at 1.63 MeV.

- **Positron annihilation line** (0.511 MeV): From positrons produced by nuclear reactions or pion decay.

- **Pion-decay continuum** ($>50$ MeV): Accelerated protons with energies above the pion production threshold (~300 MeV) produce $\pi^0 \to 2\gamma$ and $\pi^\pm$ whose decay products include high-energy gamma rays. Detected by Fermi/LAT, sometimes lasting for hours after the impulsive phase.

---

## 4. Particle Acceleration

How the solar corona accelerates a large fraction of the flare energy into non-thermal particles is one of the outstanding problems in solar physics. Three main mechanisms are considered.

### 4.1 Direct Current (DC) Electric Field Acceleration

In the reconnection region, a large-scale electric field exists:

$$\mathbf{E} = -\frac{\mathbf{v}_{\text{in}} \times \mathbf{B}}{c}$$

For $v_{\text{in}} \sim 100$ km/s and $B \sim 100$ G, this gives $E \sim 10$ V/cm. Over a current sheet of length $L \sim 10^9$ cm, the total potential drop is:

$$\Delta V = E \times L \sim 10^{10} \text{ V} = 10 \text{ GV}$$

This is vastly more than enough to accelerate particles to the observed energies. The problem is not the total energy available but the acceleration efficiency and spectral form — a simple DC field would produce a monoenergetic beam, not the observed power-law distribution.

In practice, the current sheet is likely fragmented into many magnetic islands and turbulent structures, and particles are accelerated in a more complex, multi-stage manner within this fragmented reconnection region.

### 4.2 Stochastic (Fermi) Acceleration

In the turbulent reconnection outflow and in contracting/merging magnetic islands, particles undergo repeated scatterings off moving magnetic structures:

- **Second-order Fermi acceleration**: Particles gain energy in head-on collisions with magnetic mirrors and lose energy in following collisions. The net energy gain per scattering is:

$$\left\langle\frac{\Delta E}{E}\right\rangle \propto \left(\frac{v_A}{c}\right)^2$$

where $v_A$ is the Alfven speed. Though the fractional gain per scattering is small, the cumulative effect over many scatterings naturally produces a power-law energy distribution — consistent with the observed hard X-ray spectra.

- **First-order Fermi acceleration in contracting islands**: When magnetic islands contract after reconnection, particles trapped within them gain energy systematically (analogous to first-order Fermi acceleration at shocks). The energy gain rate is:

$$\frac{dE}{dt} \propto \frac{v_A}{L_{\text{island}}} E$$

This mechanism is efficient because the contraction speed is of order $v_A$, and it readily produces power-law distributions.

### 4.3 Betatron and Adiabatic Mechanisms

When a magnetic trap collapses (e.g., newly reconnected field lines contracting toward the surface), the conservation of the magnetic moment $\mu = mv_\perp^2/(2B)$ means that particles gain perpendicular energy as $B$ increases:

$$E_\perp \propto B$$

This betatron acceleration is efficient for electrons and can contribute to the hard X-ray emission from above-the-looptop sources.

### 4.4 Observed Particle Populations

The net result of these processes:
- **Electrons**: Accelerated to energies of 10–100 keV (bulk population) with a tail extending to MeV energies. The electron distribution is typically a power law above a low-energy cutoff.
- **Protons and ions**: Accelerated from ~1 MeV to GeV energies. Enrichment in $^3$He and heavy ions in impulsive flares points to resonant wave-particle interactions.

---

## 5. Hard X-ray Emission: Thick-Target Model

The thick-target model, introduced by Brown (1971), is the standard framework for interpreting hard X-ray observations from solar flares.

### 5.1 Physical Setup

Electrons are accelerated in the corona (at or near the reconnection site) and stream downward along magnetic field lines into the increasingly dense chromosphere. As they penetrate deeper, they lose energy primarily through Coulomb collisions with the ambient plasma. Eventually, they lose all their energy — hence "thick target" (the target stops the beam).

### 5.2 Energy Loss

The collisional energy loss rate for a fast electron in a cold plasma is:

$$\frac{dE}{dN} = -\frac{2\pi e^4 \ln\Lambda}{E}$$

where $N = \int n_e \, ds$ is the column depth (cm$^{-2}$) and $\ln\Lambda \approx 20$ is the Coulomb logarithm. An electron with initial energy $E_0$ stops after traversing a column depth:

$$N_{\text{stop}} = \frac{E_0^2}{4\pi e^4 \ln\Lambda} \approx 10^{17} E_0^2 \text{ cm}^{-2}$$

where $E_0$ is in keV. A 30 keV electron stops after $N \sim 10^{20}$ cm$^{-2}$, corresponding to the upper chromosphere.

### 5.3 Photon Spectrum

If the injected electron flux spectrum is a power law:

$$F(E_0) = A E_0^{-\delta} \quad \text{(electrons s}^{-1}\text{ keV}^{-1}\text{)}$$

then the thick-target bremsstrahlung photon spectrum is:

$$I(\varepsilon) \propto \varepsilon^{-(\gamma)} \quad \text{where } \gamma = \delta - 1$$

The photon spectral index $\gamma$ is one unit flatter than the injected electron spectral index $\delta$ because lower-energy electrons are more efficient emitters (they spend more time at energies where they radiate effectively). Typical observed values are $\gamma \approx 2$–$6$, implying $\delta \approx 3$–$7$.

### 5.4 Electron Numbers

The total electron flux required to explain observed hard X-ray emission is staggering:

$$\dot{N} = \int_{E_c}^{\infty} F(E_0) \, dE_0 \sim 10^{35}\text{–}10^{36} \text{ electrons/s}$$

for an M-class flare (where $E_c \sim 10$–$20$ keV is the low-energy cutoff). Over a 100-second impulsive phase, this means $\sim 10^{37}$–$10^{38}$ electrons must be accelerated. This is a substantial fraction of all the electrons in the flaring coronal loop — the "number problem" that challenges acceleration models.

### 5.5 The Neupert Effect

Neupert (1968) observed that the time profile of the soft X-ray emission resembles the time integral of the hard X-ray emission:

$$F_{\text{SXR}}(t) \propto \int_0^t F_{\text{HXR}}(t') \, dt'$$

The physical interpretation is elegant: hard X-rays mark the instantaneous rate of energy deposition by non-thermal electrons into the chromosphere. This energy drives chromospheric evaporation, filling the coronal loop with hot plasma. The soft X-ray flux reflects the accumulated thermal energy in the loop. Thus the time derivative of the soft X-ray light curve should match the hard X-ray light curve — a prediction well confirmed observationally (though not perfectly, indicating additional heating mechanisms).

---

## 6. Flare Energetics

Understanding how the released magnetic energy is distributed among different channels is crucial for flare physics and space weather impacts.

### 6.1 Total Energy Budget

The total energy released in flares spans a wide range:

| Flare Class | Typical Total Energy |
|-------------|---------------------|
| C | $10^{29}$–$10^{30}$ erg |
| M | $10^{30}$–$10^{31}$ erg |
| X | $10^{31}$–$10^{32}$ erg |
| Extreme (X10+) | $10^{32}$–$10^{33}$ erg |

### 6.2 Energy Partition

Emslie et al. (2012) performed a comprehensive study of 38 large eruptive flares and found the following approximate partition:

- **CME kinetic energy**: ~40% of the total released energy
- **Thermal energy** (hot plasma in loops): ~30%
- **Non-thermal electrons**: ~20%
- **Radiated energy** (bolometric): ~10%
- **Non-thermal ions**: ~1% (highly uncertain)

An important finding is that the CME kinetic energy is comparable to or exceeds the flare radiated energy, challenging the view that flares and CMEs are separate phenomena — they are different manifestations of the same magnetic energy release.

### 6.3 Magnetic Energy Estimate

The available magnetic free energy can be estimated as:

$$E_{\text{mag}} = \frac{B^2}{8\pi} \times V$$

For an active region with $B \sim 500$ G and a coronal volume $V \sim (10^9 \text{ cm})^3$:

$$E_{\text{mag}} = \frac{(500)^2}{8\pi} \times 10^{27} \approx 10^{32} \text{ erg}$$

This is an upper bound — only the free energy (excess above the potential field) is available for release, typically 10–30% of the total magnetic energy. A more refined estimate uses the difference between the observed non-potential field and the corresponding potential field:

$$E_{\text{free}} = \frac{1}{8\pi}\int (B_{\text{obs}}^2 - B_{\text{pot}}^2) \, dV$$

### 6.4 Energy Release Rate

For an X-class flare releasing $\sim 10^{32}$ erg in $\sim 10^3$ s:

$$P = \frac{E}{t} \sim 10^{29} \text{ erg/s} = 10^{22} \text{ W}$$

This is a significant fraction of the Sun's total luminosity ($L_\odot = 3.83 \times 10^{33}$ erg/s) concentrated in a tiny area. The energy release rate per unit area in the reconnection region can exceed $10^{12}$ erg cm$^{-2}$ s$^{-1}$.

---

## 7. Quasi-Periodic Pulsations (QPPs)

Many solar flares exhibit oscillatory modulation in their emission — quasi-periodic pulsations with periods ranging from sub-second to several minutes.

### 7.1 Observational Characteristics

QPPs have been detected across the electromagnetic spectrum:
- Hard X-rays: periodic modulation of non-thermal emission
- Microwaves: pulsations in gyrosynchrotron emission
- Soft X-rays and EUV: oscillations in thermal emission
- White light and Lyman-alpha: detected with high-cadence photometry

The periods range from fractions of a second to several minutes, with some events showing multiple simultaneous periodicities.

### 7.2 Physical Mechanisms

Several mechanisms can produce QPPs:

**MHD oscillations of flare loops**: The flare loop acts as a resonant cavity. The relevant modes include:
- **Kink mode** (transverse displacement): Period $P_{\text{kink}} \approx 2L/c_k$ where $L$ is the loop length and $c_k$ is the kink speed. Typical: 2–10 minutes.
- **Sausage mode** (radial compression): Period $P_{\text{saus}} \approx 2L/(j_{01} v_A)$ where $j_{01} \approx 2.4$ is the first zero of the Bessel function. Shorter periods: seconds to tens of seconds.

**Periodic reconnection**: Reconnection may proceed in a quasi-periodic manner due to:
- Tearing mode instability repeatedly forming and being ejected from the current sheet
- Oscillatory dynamics of the reconnection system

**Oscillatory particle acceleration**: External MHD oscillations modulate conditions in the acceleration region, producing periodic particle injection.

### 7.3 Diagnostic Potential

QPPs provide a unique diagnostic tool because the oscillation period depends on the physical parameters of the flare plasma:

$$P \propto L \sqrt{\rho} / B$$

By measuring the period and estimating two of the three parameters ($L$, $\rho$, $B$), the third can be inferred. This technique — coronal seismology applied to flares — is an active area of research.

---

## Practice Problems

1. **GOES Classification**: A flare has a peak GOES 1–8 Angstrom flux of $3.7 \times 10^{-5}$ W/m$^2$. (a) What is its GOES class? (b) How many times more intense is it than a C1 flare? (c) If the 1–8 Angstrom flux at the Sun-Earth distance is $F$, what is the total soft X-ray luminosity (assuming isotropic emission)?

2. **Reconnection Rate**: In the CSHKP model, the reconnection inflow speed is $v_{\text{in}} = M_A v_A$. For a coronal field of $B = 100$ G, density $n = 10^{9}$ cm$^{-3}$, and reconnection rate $M_A = 0.05$, calculate: (a) the Alfven speed $v_A$, (b) the inflow speed, (c) the energy release rate per unit area of the current sheet $\dot{E}/A = B^2 v_{\text{in}}/(4\pi)$.

3. **Thick-Target Problem**: A flare has an observed hard X-ray photon spectral index $\gamma = 4$ above 20 keV. (a) What is the injected electron spectral index $\delta$? (b) If the total electron flux above 20 keV is $\dot{N} = 5 \times 10^{35}$ electrons/s, calculate the total power in non-thermal electrons: $P = \dot{N} \langle E \rangle$ where $\langle E \rangle = E_c \delta/(\delta - 2)$ for a power-law.

4. **Flare Energy Budget**: An X5.0 flare releases a total energy of $5 \times 10^{32}$ erg. Using the Emslie et al. partition, estimate the energy in: (a) the associated CME, (b) thermal plasma, (c) non-thermal electrons, (d) total radiation. Express each in units of erg and in equivalent megatons of TNT ($1$ megaton $= 4.2 \times 10^{22}$ erg).

5. **QPP Diagnostics**: A flare loop of length $L = 5 \times 10^9$ cm shows quasi-periodic pulsations with period $P = 30$ s in hard X-rays. Assuming these are caused by the sausage mode of the loop with $P \approx 2a/v_A$ where $a = 10^8$ cm is the loop radius, estimate: (a) the Alfven speed in the loop, (b) the magnetic field strength if the electron density is $n_e = 10^{11}$ cm$^{-3}$.

---

**Previous**: [Solar Dynamo and Cycle](./09_Solar_Dynamo_and_Cycle.md) | **Next**: [Coronal Mass Ejections](./11_Coronal_Mass_Ejections.md)

# Modern Solar Missions

## Learning Objectives

- Describe the orbit, instrumentation, and major discoveries of Parker Solar Probe, including magnetic switchbacks and the sub-Alfvenic solar wind
- Summarize Solar Orbiter's unique orbital design for polar observations and its early science results including "campfires"
- Understand SDO's transformative impact through a decade of continuous, high-cadence, multi-wavelength solar monitoring
- Explain DKIST's ground-breaking capabilities for high-resolution photospheric, chromospheric, and coronal magnetic field measurements
- Discuss other active missions (STEREO-A, Hinode, IRIS) and their ongoing contributions
- Describe future mission concepts (Vigil, PUNCH, Aditya-L1, ASO-S) and their expected scientific and space weather contributions
- Appreciate how coordinated multi-mission observations advance solar physics beyond what any single observatory can achieve

---

## 1. Parker Solar Probe (2018-)

### 1.1 Mission Design and Rationale

Parker Solar Probe (PSP), launched on August 12, 2018, is humanity's first mission to "touch" the Sun. Named after Eugene Parker — the physicist who predicted the solar wind in 1958 — PSP is designed to answer three fundamental questions that have eluded solar physicists for decades:

1. **Why is the corona so hot?** The Sun's surface is ~5800 K, yet the corona exceeds 1 MK. What heats it?
2. **How is the solar wind accelerated?** What forces drive the solar wind to supersonic speeds?
3. **How are solar energetic particles produced?** What accelerates ions and electrons to relativistic energies?

To answer these questions, PSP makes repeated close approaches to the Sun, sampling the corona directly with in-situ instruments while also carrying a wide-field imager.

### 1.2 Orbit and Thermal Protection

PSP uses a series of Venus gravity assists (7 total) to progressively shrink its perihelion distance:
- First perihelion (Nov 2018): 35.7 $R_\odot$
- Closest perihelia (2024-2025): 9.86 $R_\odot$ (6.9 million km, approximately 0.046 AU)

At 9.86 $R_\odot$, the spacecraft reaches speeds of ~190 km/s relative to the Sun — the fastest human-made object. The solar intensity is approximately 475 times what it is at Earth's distance.

The **Thermal Protection System (TPS)** — a carbon-composite heat shield 2.4 m in diameter and 11.4 cm thick — faces the Sun, keeping the spacecraft body in its shadow at a comfortable ~30 degrees C while the Sun-facing surface endures temperatures of approximately 1370 degrees C. The shield consists of a carbon foam core sandwiched between carbon-carbon composite facesheets, coated with a white alumina layer to reflect as much sunlight as possible. An autonomous attitude control system ensures the shield always faces the Sun, even during communication blackouts.

**Worked example — TPS equilibrium temperature**: We can derive the shield temperature from radiative equilibrium. The solar intensity at distance $d$ from the Sun scales as the inverse square of distance:

$$S(d) = S_0 \left(\frac{1 \text{ AU}}{d}\right)^2$$

At closest approach ($d = 9.86 R_\odot = 0.046$ AU): $S = 1361 \times (1/0.046)^2 \approx 6.4 \times 10^5$ W/m$^2$ — about 475 times the solar constant at Earth. The TPS absorbs a fraction $(1 - \alpha)$ of this flux, where $\alpha \approx 0.6$ is the reflectivity of the alumina coating. In thermal equilibrium, the absorbed power equals the re-radiated power (from the Sun-facing surface only, since the back is shielded):

$$(1 - \alpha) \cdot S = \sigma T_{\text{eq}}^4$$

Solving: $T_{\text{eq}} = \left(\frac{(1 - 0.6) \times 6.4 \times 10^5}{5.67 \times 10^{-8}}\right)^{1/4} = \left(\frac{2.56 \times 10^5}{5.67 \times 10^{-8}}\right)^{1/4} \approx \left(4.52 \times 10^{12}\right)^{1/4} \approx 1460$ K $\approx 1190°$C.

The actual peak temperature (~1370°C $\approx$ 1640 K) is somewhat higher because (a) the coating reflectivity degrades with prolonged solar exposure, (b) the shield also absorbs some infrared radiation from the hot solar atmosphere, and (c) the effective emissivity of the carbon surface is less than 1. This calculation illustrates a key engineering constraint: even with 60% reflectivity, the shield must withstand temperatures exceeding the melting point of aluminum (660°C) and approaching that of steel (~1500°C), which is why carbon composites — stable to >3000°C — are essential.

### 1.3 Instruments

PSP carries four instrument suites:

**FIELDS**: Measures electric and magnetic fields and radio waves. Includes two fluxgate magnetometers, a search-coil magnetometer, and five electric field antennas extending beyond the heat shield's shadow. FIELDS has provided the first direct measurements of the magnetic field inside the corona.

**SWEAP (Solar Wind Electrons Alphas and Protons)**: Measures the velocity distribution functions of solar wind ions and electrons. Includes the Solar Probe Cup (SPC), a Faraday cup that peeks around the heat shield to directly sample the solar wind facing the Sun — the only instrument that intentionally looks at the Sun.

**WISPR (Wide-field Imager for Solar Probe)**: Two wide-field telescopes imaging the corona and inner heliosphere in white light, observing coronal streamers, CMEs, and the zodiacal dust cloud from within.

**IS$\odot$IS (Integrated Science Investigation of the Sun)**: Measures energetic particles over a broad energy range (tens of keV to hundreds of MeV). Consists of EPI-Lo (lower energies, time-of-flight) and EPI-Hi (higher energies, solid-state telescopes).

### 1.4 Major Discoveries

**Magnetic switchbacks**: The most unexpected finding from PSP's early orbits was the prevalence of "switchbacks" — rapid, large-amplitude deflections in the magnetic field, often reaching full reversals (the field momentarily points back toward the Sun). Switchbacks are associated with bursts of enhanced radial solar wind velocity ("velocity spikes").

The physical origin of switchbacks remains debated. Leading hypotheses include:
- **Interchange reconnection** at the base of the corona: When open field lines reconnect with closed loops, the newly opened field line has an S-shaped kink that propagates outward as a switchback
- **In-situ generation**: Velocity shear instabilities or turbulent evolution in the expanding solar wind
- **Coronal jets**: Small-scale eruptions that twist the field

Switchbacks appear to be a fundamental feature of the young solar wind, potentially carrying significant energy outward. Their prevalence suggests that the near-Sun solar wind is far more dynamic and structured than previously assumed.

**Sub-Alfvenic solar wind**: On April 28, 2021 (perihelion 8, at 18.8 $R_\odot$), and subsequently on closer passes, PSP crossed the **Alfven critical surface** — the boundary below which the solar wind speed falls below the local Alfven speed ($v_A = B/\sqrt{4\pi\rho}$). Below this surface, information can propagate sunward via Alfven waves; above it, the flow is super-Alfvenic and information travels only outward.

Crossing the Alfven surface was a milestone — PSP effectively entered the solar corona in a plasma physics sense. Key observations from the sub-Alfvenic crossings include:
- The Alfven surface is not smooth but wrinkled, with its height varying depending on coronal structure
- Inside the Alfven surface, the magnetic field is more radial and switchbacks are less frequent
- The transition from sub- to super-Alfvenic is often sharp, occurring at coronal streamers and pseudostreamers

**Dust depletion zone**: WISPR images revealed a gradual decrease in zodiacal dust density inward of approximately 10-20 $R_\odot$, consistent with theoretical predictions of a dust-free zone where solar heating sublimates dust grains. This is the first observational evidence for the inner boundary of the zodiacal dust cloud.

**Near-Sun energetic particles**: IS$\odot$IS detected energetic particle events very close to the Sun that were not observed at 1 AU, revealing localized acceleration processes (possibly at small-scale shocks or in reconnection events) that are masked by transport effects at greater distances.

**Slow solar wind variability**: Close to the Sun, the slow solar wind is remarkably structured, with rapid transitions in composition, speed, and magnetic topology. PSP data suggests the slow wind is not a single, uniform flow but a mixture of plasma from different coronal sources (streamer blobs, S-web corridors, and active region outflows).

---

## 2. Solar Orbiter (2020-)

### 2.1 Mission Design

Solar Orbiter (SolO), launched February 10, 2020, is an ESA-led mission (with NASA partnership) that combines close solar approaches with a unique orbital inclination that will eventually provide the first views of the Sun's polar regions.

**Orbit**: Solar Orbiter uses Venus gravity assists to both reduce its perihelion and increase its orbital inclination:
- Closest perihelion: 0.28 AU (approximately 60 $R_\odot$)
- Maximum inclination: ~33 degrees out of the ecliptic (achieved in the extended mission)

The high-inclination orbits are scientifically transformative. We have never imaged the Sun's poles, yet the polar regions are critical for understanding:
- The global magnetic flux transport cycle (flux cancellation at the poles reverses the dipole)
- The fast solar wind, which originates from polar coronal holes
- The polar dynamo contribution, if any

### 2.2 Instruments

Solar Orbiter carries 10 instruments spanning remote sensing and in-situ measurements:

**Remote sensing** (operated near perihelion):
- **EUI** (Extreme Ultraviolet Imager): Full-disk and high-resolution EUV imaging (17.4 nm, 30.4 nm)
- **PHI** (Polarimetric and Helioseismic Imager): Photospheric magnetograms and Dopplergrams — HMI equivalent from a different vantage point
- **SPICE** (Spectral Imaging of the Coronal Environment): EUV spectral imager for temperature, density, and velocity diagnostics
- **Metis**: Coronagraph imaging the corona in visible and UV Lyman-alpha simultaneously
- **SoloHI** (Heliospheric Imager): Wide-field white-light imager tracking the solar wind and CMEs
- **STIX** (Spectrometer/Telescope for Imaging X-rays): Hard X-ray imaging spectroscopy of solar flares

**In-situ**:
- **MAG**: Magnetometer
- **SWA** (Solar Wind Analyser): Proton, alpha, electron, and heavy ion analyzers
- **EPD** (Energetic Particle Detector): Suprathermal to relativistic particles
- **RPW** (Radio and Plasma Waves): Electric and magnetic field fluctuations, radio emissions

The combination of remote sensing and in-situ is Solar Orbiter's greatest strength: it can image structures on the Sun and simultaneously measure the particles and fields emanating from those structures, establishing direct cause-and-effect links.

### 2.3 Early Science Results

**"Campfires"**: Among the first images released from EUI's high-resolution channel were ubiquitous small-scale EUV brightenings at the edges of supergranular network cells, nicknamed "campfires." These features, typically 400-4000 km in size and lasting 10-200 seconds, are much smaller than previously cataloged coronal bright points. They may represent nanoflare-scale energy release events — the long-hypothesized small reconnection events that could collectively heat the quiet corona.

The estimated energy of individual campfires is $10^{24}$-$10^{25}$ erg, placing them in the nanoflare to microflare range. Whether they are numerous and frequent enough to account for coronal heating remains an active question requiring careful statistical analysis.

**Stealth CMEs observed close to Sun**: Solar Orbiter's coronagraph (Metis) and heliospheric imager (SoloHI) have observed "stealth CMEs" — eruptions with no obvious low-coronal signature in EUV images — from much closer than SOHO/LASCO, revealing faint structures that are invisible from 1 AU.

**Connecting remote sensing and in-situ**: A primary science goal is to connect solar wind properties measured in-situ to their specific source regions on the Sun. Early results have traced slow solar wind streams back to small coronal holes and active region boundaries using composition analysis (charge state ratios, FIP bias) and magnetic field mapping.

**Solar energetic particle events**: EPD has detected SEP events from very close to the Sun, providing constraints on acceleration timescales and seed populations that are smeared out by transport effects at 1 AU.

### 2.4 Upcoming Polar Phase

The most anticipated phase begins as Venus gravity assists progressively increase the orbital inclination above 20 degrees (starting ~2027) and eventually to ~33 degrees. From these vantage points, Solar Orbiter will:
- Image the polar coronal holes and their fine-scale magnetic structure
- Observe the polar crown filament channel and its role in the magnetic cycle
- Provide the first polar view of polar plumes, jets, and coronal bright points
- Enable far-side monitoring (the pole "sees" part of the far side invisible from Earth)

This polar perspective is expected to fundamentally advance our understanding of the solar dynamo and magnetic flux transport.

---

## 3. SDO: A Decade of Discovery (2010-)

### 3.1 A Revolution in Solar Observing

The Solar Dynamics Observatory, launched February 11, 2010, into a geosynchronous inclined orbit, has arguably transformed solar physics more than any other single mission. Its continuous, high-cadence, multi-wavelength coverage of the full solar disk created a dataset so rich that entirely new approaches — including machine learning — were needed to exploit it.

Before SDO, solar EUV images were taken every 10-15 minutes at a few wavelengths (SOHO/EIT). SDO/AIA produces images in 10 wavelength channels every 12 seconds with four times the spatial resolution. This 100-fold increase in data rate revealed dynamics that were completely invisible before: propagating EUV waves, coronal rain, the fine structure of flare energy release, and the continuous restructuring of the magnetic corona.

### 3.2 AIA: Multi-Thermal Imaging

AIA's seven EUV channels sample different temperatures:

| Channel (A) | Primary Ion | Peak Temperature (MK) | Primary Diagnostic |
|-------------|------------|----------------------|-------------------|
| 94 | Fe XVIII | 7.1 | Flare plasma, hot AR cores |
| 131 | Fe VIII, Fe XXI | 0.4, 11.0 | Transition region, flare plasma |
| 171 | Fe IX | 0.7 | Quiet corona, coronal loops |
| 193 | Fe XII, Fe XXIV | 1.6, 20.0 | Active region corona, flare plasma |
| 211 | Fe XIV | 2.0 | Active region corona |
| 304 | He II | 0.08 | Chromosphere, transition region |
| 335 | Fe XVI | 2.5 | Active region corona |

Each 4096 $\times$ 4096 pixel image covers the full disk with 0.6 arcsec per pixel (approximately 435 km). The 12-second cadence captures the dynamics of flares, oscillations, and eruptions in detail. AIA data has enabled:

- **DEM analysis**: Combining multiple channels to reconstruct the temperature distribution along each line of sight
- **EUV wave studies**: Tracking large-scale coronal propagating fronts at speeds of 200-1500 km/s
- **Coronal loop physics**: Heating, cooling, and the thermal non-equilibrium cycle (coronal rain)
- **Flare dynamics**: Impulsive phase ribbons, post-flare loops, sequential reconnection

### 3.3 HMI: Mapping the Magnetic Field

HMI produces three data products:
1. **Line-of-sight magnetograms** (45-second cadence): The radial component of the photospheric magnetic field
2. **Vector magnetograms** (12-minute cadence): All three components of the photospheric B-field, derived from full Stokes polarimetry of the Fe I 6173 A line
3. **Dopplergrams** (45-second cadence): Line-of-sight velocity maps for helioseismology

HMI magnetograms serve as the essential boundary condition for all coronal magnetic field models (PFSS, NLFFF, MHD). They are also the primary input for flare prediction models, which use active region magnetic parameters (total flux, flux near the polarity inversion line, shear, gradients) to forecast eruptions.

### 3.4 Machine Learning on SDO Data

SDO's massive dataset (~1.5 TB/day, >100 PB total as of 2025) has catalyzed the application of machine learning to solar physics:

- **Flare prediction**: Deep learning models trained on HMI magnetograms and AIA images to predict M- and X-class flares 24-48 hours in advance. Best models achieve True Skill Statistic (TSS) of ~0.5-0.8, significantly better than climatological baselines.
- **Image super-resolution**: Neural networks trained to enhance SDO resolution or to interpolate between AIA wavelengths.
- **DEM inversion**: ML-based DEM recovery from AIA channel images, orders of magnitude faster than traditional iterative methods.
- **Virtual instruments**: Models that generate synthetic EUV images from magnetograms (or vice versa), filling gaps in observational coverage.
- **Event detection**: Automated identification of flares, CMEs, coronal holes, active regions, filaments, and other features in the continuous data stream.

SDO has demonstrated that the future of solar physics lies at the intersection of comprehensive observational datasets and advanced computational analysis.

---

## 4. DKIST: Ground-Based Revolution (2020-)

### 4.1 Capabilities

The Daniel K. Inouye Solar Telescope (DKIST), situated at the summit of Haleakala on Maui, Hawaii (altitude 3048 m), began science operations in 2022. With a 4-meter primary mirror, it is the world's most powerful solar telescope.

**Diffraction-limited resolution**:
$$\theta = 1.22 \frac{\lambda}{D}$$

At 500 nm: $\theta = 0.031$ arcsec = 22 km on the Sun. At 1074.7 nm (Fe XIII coronal line): $\theta = 0.067$ arcsec = 49 km. These resolutions probe spatial scales at or below the fundamental scales of solar magnetoconvection (granulation ~1000 km, magnetic flux tubes ~100 km, current sheets potentially <100 km).

**Adaptive optics**: DKIST employs a deformable mirror with ~1600 actuators, correcting wavefront distortions from atmospheric turbulence approximately 2000 times per second. A correlating wavefront sensor locks onto granulation patterns to measure the wavefront. Multi-conjugate adaptive optics (MCAO) — correcting turbulence at multiple atmospheric heights — is planned for future upgrades to extend the corrected field of view.

### 4.2 Instrument Suite

DKIST's instruments are designed for comprehensive spectropolarimetric observations from the visible through the near-infrared:

- **VBI** (Visible Broadband Imager): High-cadence imaging in narrowband visible filters (e.g., H-alpha, Ca II K, G-band). Provides context imaging at the diffraction limit.
- **ViSP** (Visible Spectro-Polarimeter): Full Stokes polarimetry across the visible spectrum. Three independently configurable spectral arms allow simultaneous observation of multiple spectral lines, enabling multi-height atmospheric sampling.
- **DL-NIRSP** (Diffraction-Limited Near-Infrared Spectro-Polarimeter): Integral-field spectropolarimetry in the 0.5-2.5 $\mu$m range at the diffraction limit. Provides 2D spectral maps without scanning.
- **Cryo-NIRSP** (Cryogenic Near-Infrared Spectro-Polarimeter): Cooled to cryogenic temperatures to minimize thermal background emission. Designed specifically for coronal observations: the Fe XIII 1074.7 nm and Si X 1430.1 nm forbidden lines, which are faint and embedded in the bright sky background. This instrument enables routine measurement of the coronal magnetic field — arguably the most important unmeasured quantity in all of solar physics.

### 4.3 Science Highlights

**Sunspot fine structure**: DKIST's first images revealed convective detail within sunspot penumbrae at scales never before observed — individual convective filaments, their dark lanes, and the transition to umbral dots, all resolved at ~30 km.

**Chromospheric magnetic fields**: Using spectropolarimetry of the Ca II 8542 A and He I 10830 A lines, DKIST measures the magnetic field in the chromosphere — a region where the field is notoriously difficult to determine because the plasma is not in local thermodynamic equilibrium, making spectral line interpretation more complex than in the photosphere.

**Coronal magnetic field**: The Cryo-NIRSP instrument measures the linear polarization of the Fe XIII 1074.7 nm line, from which the plane-of-sky magnetic field direction can be inferred. Combined with emission line splitting or Zeeman broadening (for very strong fields), and coronal seismology from wave observations, DKIST is building the first comprehensive picture of the coronal B-field.

This is transformative because the coronal magnetic field controls essentially all coronal and heliospheric dynamics — from coronal heating to flares to CMEs — yet has been virtually unmeasurable until now. Previously, the coronal field was known only through extrapolation from photospheric measurements (PFSS, NLFFF models), which involve significant assumptions.

**Small-scale flux emergence**: DKIST observes the emergence of tiny magnetic bipoles (ephemeral regions and smaller) that are below the detection threshold of SDO/HMI. These small-scale flux elements may be crucial for:
- Chromospheric and coronal heating (reconnection with existing field)
- Building the magnetic "carpet" that pervades the quiet Sun
- Understanding the solar dynamo at its smallest scales

---

## 5. Other Active Missions

### 5.1 STEREO-A (2006-)

Although STEREO-B was lost in 2014, STEREO-A continues to provide a vital off-Sun-Earth-line viewpoint. As of 2025-2026, STEREO-A is approximately 20-30 degrees ahead of Earth in its orbit. This perspective:
- Reveals CMEs from the side that are directed toward Earth (which appear as halos in Earth-based coronagraphs, making their speed and morphology ambiguous)
- Provides advance warning of active regions rotating onto the Earth-facing disk
- Enables stereoscopic CME reconstruction when combined with SOHO/LASCO (from near Earth)
- Detects far-side active regions through EUV observations

### 5.2 Hinode (2006-)

After nearly two decades, Hinode continues to deliver unique science:
- **SOT**: Highest-resolution space-based optical observations (0.2 arcsec) of photospheric magnetic elements, sunspot evolution, and flux emergence
- **EIS**: Spectroscopic diagnostics of coronal and transition region plasma — velocities, temperatures, densities, and composition — that imaging instruments cannot provide
- **XRT**: Soft X-ray imaging revealing the hot coronal plasma associated with active regions, flares, and microflares

Hinode's long temporal baseline enables studies of magnetic field evolution over multiple solar cycles.

### 5.3 IRIS (2013-)

IRIS remains the only space mission dedicated to the chromosphere-corona interface. Its UV spectrograph resolves:
- Chromospheric dynamics (Mg II h & k lines) at sub-arcsecond scales
- Transition region flows and brightenings (Si IV, C II lines)
- The formation of spicules, jets, and UV bursts

IRIS has revealed that the chromosphere is far more dynamic than previously recognized, with ubiquitous heating events, rapid flows, and complex shock dynamics that challenge static atmospheric models.

### 5.4 Operational Space Weather Monitors

- **GOES-16/17/18** (geostationary orbit): Operational X-ray flux monitoring (the GOES classification of flares: A, B, C, M, X), SUVI (Solar Ultraviolet Imager) for full-disk EUV, and magnetometers for geomagnetic monitoring.
- **DSCOVR** (L1 Lagrange point): Measures real-time solar wind parameters (speed, density, magnetic field) at L1, providing approximately 15-45 minutes advance warning of geomagnetic storm conditions. Its Faraday cup and magnetometer are critical inputs for operational space weather models.

---

## 6. Future Missions and Concepts

### 6.1 ESA Vigil (L5 Mission, ~2031)

Vigil (formerly Lagrange) is an ESA space weather mission planned for the Sun-Earth L5 Lagrange point — trailing Earth by 60 degrees in its orbit. This vantage point provides a side view of the Sun-Earth line, which is transformative for space weather prediction:

**Coronagraph from L5**: Earth-directed CMEs, which appear as ambiguous "halo" events from Earth's perspective, would be seen from the side by Vigil's coronagraph. The CME's shape, speed, and direction would be directly measurable, potentially reducing CME arrival time prediction errors from the current ~12-18 hours to ~6-8 hours.

**Magnetograph from L5**: Vigil would observe active regions ~4-5 days before they rotate into Earth-facing view, providing early warning of potentially dangerous regions. Combined with PFSS modeling, this could enable predictive coronal and solar wind models with longer lead times.

**Heliospheric imager**: Tracking CMEs and solar wind structures between the Sun and Earth from the side.

Vigil represents a paradigm shift from "monitoring" to "predicting" space weather, and is considered a high-priority mission by the space weather community.

### 6.2 PUNCH (Polarimeter to Unify the Corona and Heliosphere, ~2025)

PUNCH is a NASA Small Explorer (SMEX) mission consisting of four small satellites flying in formation. Together, they image the transition from the corona into the young solar wind, filling the observational gap between coronagraphs (which see the corona out to ~30 $R_\odot$) and heliospheric imagers (which see the solar wind beyond ~15-30 $R_\odot$).

PUNCH will:
- Image the continuous flow of solar wind from the corona into the heliosphere
- Track CMEs in 3D using polarization (Thomson scattering polarization depends on the scattering angle, constraining the 3D location)
- Observe solar wind turbulence and structure formation in the critical transition region

### 6.3 Indian Aditya-L1 (2023-)

India's first dedicated solar observatory, launched September 2, 2023, and positioned at the Sun-Earth L1 point. Aditya-L1 carries seven payloads:
- **VELC** (Visible Emission Line Coronagraph): Imaging and spectroscopy of the inner corona (1.05-3 $R_\odot$), targeting temperature and velocity diagnostics
- **SUIT** (Solar Ultraviolet Imaging Telescope): Near-UV imaging (200-400 nm) of the photosphere and chromosphere
- **In-situ instruments**: Particle detectors and magnetometer for solar wind monitoring at L1

Aditya-L1 expands the global fleet of solar observatories and provides L1 solar wind monitoring complementary to DSCOVR.

### 6.4 Chinese ASO-S (Advanced Space-based Solar Observatory, 2022-)

ASO-S is China's first comprehensive solar observation satellite, designed to simultaneously observe the "one magnetic field and two eruptions" (solar magnetic field, solar flares, and CMEs):
- **FMG** (Full-disk vector Magnetograph): Photospheric magnetic field measurements
- **HXI** (Hard X-ray Imager): Solar flare X-ray imaging spectroscopy (sub-collimator technique)
- **LST** (Lyman-alpha Solar Telescope): Solar disk and corona imaging in Ly-alpha and white light

The simultaneous magnetograph + X-ray + white-light capability enables direct studies of the relationship between magnetic field evolution, flare energy release, and CME eruption.

### 6.5 Other Concepts

- **Solaris**: A proposed ESA mission for a close-to-Sun polar orbit, providing a bird's-eye view of the solar poles and the global magnetic flux distribution. The polar magnetic field is the strongest predictor of the next solar cycle's amplitude, yet it is currently observed only obliquely from the ecliptic.
- **MUSE** (Multi-slit Solar Explorer): A proposed NASA mission carrying a multi-slit EUV spectrograph with imaging cadence, enabling spectroscopic observations over a wide field of view simultaneously. This would solve the fundamental limitation of current slit spectrographs (like IRIS and Hinode/EIS), which must scan across the field of view and thus sacrifice either spatial coverage or temporal cadence.
- **COSMO** (Coronal Solar Magnetism Observatory): A ground-based facility (HAO/NCAR) designed for routine coronal magnetic field measurement using the K-coronagraph and the ChroMag chromosphere-corona magnetograph.

---

## 7. Coordinated Multi-Mission Observations

### 7.1 The Power of Combined Datasets

Modern solar physics has entered an era of coordinated campaigns where multiple missions simultaneously observe the same solar event or region. This synergy yields insights impossible from any single observatory:

**Example: A flare-CME event observed by the full fleet**
- SDO/AIA and HMI: Pre-flare magnetic configuration, flare ribbon evolution, post-flare loops
- IRIS: Chromospheric evaporation spectra (blueshifts in hot lines, redshifts in cool lines)
- Hinode/EIS: Coronal temperature, density, and velocity diagnostics in the flare loop system
- GOES: Soft X-ray classification and light curve
- SOHO/LASCO: CME white-light morphology and speed in the outer corona
- STEREO-A: Side-view of the CME for 3D reconstruction
- PSP or SolO (if well-positioned): In-situ detection of the CME and energetic particles at their heliocentric distance
- Radio arrays (LOFAR, EOVSA): Type II burst (shock), Type III burst (electron beams), microwave emission (accelerated electrons in magnetic field)
- DKIST: Photospheric and chromospheric magnetic field evolution at the highest resolution
- Neutron monitors (if GLE): Ground-level detection of the most energetic particles

This multi-vantage, multi-wavelength, multi-messenger approach is the gold standard of modern solar and heliospheric physics. The challenge is no longer obtaining data but rather integrating vast, heterogeneous datasets into coherent physical models.

### 7.2 Data Archives and Community Access

All major solar missions provide open data access:
- **JSOC** (Joint Science Operations Center): SDO data (AIA, HMI, EVE)
- **VSO** (Virtual Solar Observatory): Unified search across multiple missions
- **SOHO Archive**: LASCO, EIT, MDI data
- **STEREO Science Center**: SECCHI images and in-situ data
- **CDAWeb** (Coordinated Data Analysis Web): PSP and SolO in-situ data
- **DKIST Data Center**: Spectropolarimetric data products
- **Helioviewer**: Web-based visualization tool for browsing multi-mission solar images

The open data philosophy of solar physics has been a major factor in its rapid scientific progress, enabling researchers worldwide to contribute to discovery regardless of their involvement in specific missions.

---

## Practice Problems

**Problem 1**: Parker Solar Probe's closest perihelion is 9.86 $R_\odot$. Calculate the solar intensity (in W/m$^2$) at this distance, given the solar constant at 1 AU is $S_0 = 1361$ W/m$^2$ and 1 AU = 215 $R_\odot$. The TPS reflectivity is approximately 0.6 — estimate the power per unit area absorbed by the heat shield. If the shield re-radiates as a blackbody from one face, estimate its equilibrium temperature using the Stefan-Boltzmann law.

**Problem 2**: Solar Orbiter's maximum orbital inclination will be 33 degrees. At this inclination, what is the highest heliographic latitude that can be directly imaged? The Sun's rotation axis is tilted 7.25 degrees from the ecliptic normal — how does this affect the maximum observable latitude during favorable orbital geometry? Why has no previous mission imaged the solar poles?

**Problem 3**: SDO/AIA produces 4096 $\times$ 4096 pixel, 16-bit images in 10 channels every 12 seconds. Calculate the raw data rate in megabytes per second. Over SDO's 15+ year mission, estimate the total data volume (in petabytes). Discuss why machine learning became essential for analyzing this dataset.

**Problem 4**: DKIST's diffraction limit at 500 nm is 0.031 arcsec (22 km on the Sun). Typical photospheric granules are ~1000 km across. How many resolution elements fit across one granule? The smallest magnetic flux tubes (bright points) are ~100-200 km. Can DKIST resolve these? If DKIST observes the Fe XIII 1074.7 nm coronal line, what is the diffraction-limited resolution in km? Discuss why this is still valuable for coronal science despite being coarser than the visible resolution.

**Problem 5**: ESA's Vigil mission at L5 observes a CME from the side while SOHO/LASCO at L1 sees it as a halo CME. From L1, the halo CME's plane-of-sky speed is measured as 500 km/s. Vigil measures the true radial speed as 1200 km/s. Explain the discrepancy. If the CME is launched at a heliocentric distance of 2 $R_\odot$ and decelerates to 800 km/s by 1 AU due to drag, estimate the transit time to Earth. Compare this with the current prediction uncertainty of ~12-18 hours and discuss how Vigil improves the forecast.

---

**Previous**: [Solar Energetic Particles](./14_Solar_Energetic_Particles.md) | **Next**: [Projects](./16_Projects.md)

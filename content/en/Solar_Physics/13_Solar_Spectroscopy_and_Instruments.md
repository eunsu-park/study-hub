# Solar Spectroscopy and Instruments

## Learning Objectives

- Identify key Fraunhofer lines in the solar spectrum and explain their diagnostic value for probing different atmospheric layers
- Explain how spectroscopic techniques measure temperature, density, velocity, and elemental abundances in solar plasma
- Describe the capabilities and science goals of major ground-based solar observatories including DKIST
- Understand the design principles and scientific contributions of space-based solar observatories (SDO, SOHO, Hinode, IRIS, STEREO)
- Explain the operating principles of coronagraphs and heliospheric imagers for observing the extended corona
- Classify solar radio burst types and relate them to underlying physical processes
- Appreciate how multi-wavelength, multi-instrument observations combine to provide a comprehensive picture of solar activity

---

## 1. The Solar Spectrum

### 1.1 The Continuum: An Approximate Blackbody

The Sun radiates approximately as a blackbody with an effective temperature $T_{\text{eff}} = 5778$ K. This means the spectral radiance follows the Planck function:

$$B_\lambda(T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{hc/(\lambda k_B T)} - 1}$$

where $h$ is Planck's constant, $c$ the speed of light, $k_B$ Boltzmann's constant, and $\lambda$ the wavelength.

Wien's displacement law tells us that the peak emission occurs at:

$$\lambda_{\max} = \frac{b}{T} \approx \frac{2.898 \times 10^{-3} \text{ m K}}{5778 \text{ K}} \approx 501 \text{ nm}$$

This places the solar emission peak right in the middle of the visible spectrum — in the green-yellow region. It is no coincidence that human vision evolved to be most sensitive near this wavelength. The total luminosity follows the Stefan-Boltzmann law: $L_\odot = 4\pi R_\odot^2 \sigma T_{\text{eff}}^4 \approx 3.83 \times 10^{26}$ W.

However, the Sun is not a perfect blackbody. The continuum opacity varies with wavelength (primarily from H$^-$ bound-free and free-free absorption in the photosphere), so different wavelengths emerge from slightly different depths, each with a slightly different temperature. This produces subtle departures from a pure Planck curve.

### 1.2 Fraunhofer Lines: Fingerprints of the Solar Atmosphere

In 1814, Joseph von Fraunhofer catalogued hundreds of dark absorption lines superimposed on the solar continuum. These arise because atoms in the cooler overlying photospheric and chromospheric layers absorb photons at specific wavelengths corresponding to electronic transitions. Each line carries information about the physical conditions — temperature, density, velocity, magnetic field — at the height where it forms.

Think of the solar spectrum as a message written in multiple layers. The bright continuum is the "paper," generated deep in the photosphere. Each absorption line is a "stamp" imprinted by a specific atomic species at a specific height. By reading these stamps, we decode the temperature, density, and motion of the plasma at each layer.

### 1.3 Key Diagnostic Lines

Different spectral lines form at different heights and temperatures, providing a tomographic view of the solar atmosphere:

**Photospheric Lines (T ~ 4500-6000 K):**
- **Na I D lines** (589.0, 589.6 nm): Neutral sodium doublet, formed in the upper photosphere. Historically important for early solar studies.
- **Fe I 6173 A** (617.3 nm): Neutral iron line used by SDO/HMI for photospheric magnetic field measurements via the Zeeman effect. This particular line was chosen for its clean profile, moderate Lande g-factor ($g = 2.5$), and formation height near the surface where magnetic fields are strongest.

**Chromospheric Lines (T ~ 6000-20,000 K):**
- **H$\alpha$** (656.3 nm): The workhorse of chromospheric observation. This Balmer-series line (n=3 to n=2 transition of hydrogen) reveals prominences, filaments, flare ribbons, and chromospheric dynamics. Its wide wings allow Doppler imaging by scanning across the line profile.
- **Ca II K** (393.4 nm): Singly ionized calcium. The K-line core forms in the upper chromosphere, while its wings sample lower layers. The K$_1$, K$_2$, and K$_3$ features (minimum, emission peaks, and central reversal) directly encode the chromospheric temperature rise.
- **Ca II 8542 A**: Another Ca II line, increasingly used for chromospheric spectropolarimetry.
- **Mg II h & k** (279.6, 280.3 nm): UV lines observed by IRIS, analogous to Ca II H & K but formed slightly higher. Excellent chromospheric diagnostics.

**Transition Region Lines (T ~ 80,000-500,000 K):**
- **He II 304 A** (30.4 nm): Singly ionized helium, formed at approximately 80,000 K. One of the brightest EUV lines, used by SDO/AIA and SOHO/EIT for imaging the transition region and upper chromosphere.
- **C IV** (154.9 nm): Formed at ~100,000 K, a key IRIS diagnostic for the transition region.
- **O VI** (103.2 nm): Formed at ~300,000 K, important for SOHO/UVCS studies of the extended corona.

**Coronal Lines (T > 1 MK):**
- **Fe XII 195 A** (19.5 nm): Formed at ~1.5 MK, one of the primary SDO/AIA channels for imaging the quiet corona and active regions.
- **Fe IX 171 A** (17.1 nm): Formed at ~0.7 MK, excellent for imaging coronal loops and quiet Sun structures.
- **Fe XIV 211 A** (21.1 nm): Formed at ~2 MK, sensitive to active region corona.
- **Fe XVIII 94 A** (9.4 nm): Formed at ~7 MK, lights up during flares and in hot active region cores.
- **Fe XIII 1074.7 nm** (near-infrared): A forbidden coronal emission line, critically important because its linear polarization directly measures the coronal magnetic field direction. DKIST's Cryo-NIRSP is designed to exploit this line.

The progression from photospheric to coronal lines illustrates a profound observational strategy: by choosing the right spectral line, we can selectively observe any layer of the solar atmosphere, from the surface to millions of kilometers above it.

---

## 2. Spectroscopic Diagnostics

Spectral lines are not merely identifiers of atomic species. Their shapes, intensities, and ratios encode a wealth of physical information. Let us examine the major diagnostic techniques.

### 2.1 Temperature Diagnostics

**Line ratio method**: Consider two spectral lines from the same element but different ionization states — for example, Fe XII (formed at ~1.5 MK) and Fe XIV (formed at ~2 MK). In ionization equilibrium, the ratio of their emissivities depends strongly on temperature through the ionization balance:

$$\frac{I(\text{Fe XII})}{I(\text{Fe XIV})} = f(T_e)$$

This ratio changes rapidly with temperature, making it an excellent thermometer. The technique works because the ionization fraction of each ion peaks at a characteristic temperature, and the ratio of two such peaks varies monotonically over a useful range.

**Differential Emission Measure (DEM)**: For a more complete temperature picture, the DEM quantifies how much plasma exists at each temperature along the line of sight:

$$I_\lambda = \int G_\lambda(T) \cdot \text{DEM}(T) \, dT$$

where $G_\lambda(T)$ is the contribution function (encapsulating atomic physics) and $\text{DEM}(T) = n_e^2 \, dh/dT$ represents the temperature distribution of emitting material. By observing many lines simultaneously and inverting this equation, we recover the DEM — essentially a histogram of "how much stuff is at each temperature."

### 2.2 Density Diagnostics

Density-sensitive line ratios exploit a subtle property of atomic physics: some excited levels can be depopulated by either radiative decay or collisional de-excitation, depending on the electron density.

At low densities ($n_e \ll n_{\text{crit}}$), every excitation leads to a photon (the "coronal limit"). At high densities ($n_e \gg n_{\text{crit}}$), collisional de-excitation competes, altering the level populations and hence the emitted line ratios. The critical density is:

$$n_{\text{crit}} = \frac{A_{ij}}{q_{ij}}$$

where $A_{ij}$ is the **Einstein A coefficient** (the spontaneous radiative transition rate, in s$^{-1}$) — the probability per unit time that an ion in excited state $j$ will spontaneously emit a photon and decay to state $i$, and $q_{ij}$ is the **collisional de-excitation rate coefficient** (in cm$^3$ s$^{-1}$) — a measure of how efficiently electron impacts knock the ion from state $j$ back to state $i$ without producing a photon. The critical density is the electron density at which these two depopulation channels are equally fast: when $n_e \ll n_{\text{crit}}$, almost every excitation produces a photon (the "coronal limit"), and the line intensity scales as $n_e^2$; when $n_e \gg n_{\text{crit}}$, most excitations are collisionally quenched before they can radiate, driving the level populations toward their thermal (Boltzmann) values. This density dependence is what makes line ratios from levels with different $n_{\text{crit}}$ powerful density diagnostics.

A classic example is the **Si X 356/347 A ratio**: these two lines from Si X have different upper levels with different critical densities. Their ratio varies significantly over the density range $n_e \sim 10^8 - 10^{11}$ cm$^{-3}$, making it an excellent coronal density probe. Similarly, the Fe XIII 1074.7/1079.8 nm ratio is density-sensitive in the range relevant to the corona.

### 2.3 Velocity Diagnostics

**Doppler shift**: Motion along the line of sight shifts spectral lines:

$$\frac{\Delta\lambda}{\lambda_0} = \frac{v_{\text{LOS}}}{c}$$

Blueshifts indicate motion toward the observer; redshifts indicate motion away. Typical measurements:
- Photospheric convection: ~1-2 km/s (granulation)
- Chromospheric spicules: ~20-100 km/s
- Coronal mass ejections: ~100-3000 km/s
- Flare evaporation upflows: ~100-400 km/s (blueshifts in hot lines during flares)

SDO/HMI produces full-disk Dopplergrams every 45 seconds by measuring the shift of the Fe I 6173 A line, revealing the surface velocity field used for helioseismology.

**Line broadening**: The observed line width $\Delta\lambda_{\text{obs}}$ has multiple contributions:

$$\Delta\lambda_{\text{obs}}^2 = \Delta\lambda_{\text{thermal}}^2 + \Delta\lambda_{\text{non-thermal}}^2 + \Delta\lambda_{\text{instrumental}}^2$$

The thermal width depends on temperature and ion mass:

$$\Delta\lambda_{\text{thermal}} = \frac{\lambda_0}{c} \sqrt{\frac{2 k_B T}{m_i}}$$

After accounting for thermal and instrumental broadening, any excess width is attributed to non-thermal broadening — evidence for unresolved turbulent motions or wave amplitudes. This non-thermal velocity $\xi$ is a crucial diagnostic for coronal heating mechanisms:

$$\xi = \sqrt{\frac{2 k_B T_{\text{excess}}}{m_i}}$$

Typical non-thermal velocities in the corona are 20-40 km/s, potentially indicative of Alfven wave amplitudes.

### 2.4 Abundance Analysis

The equivalent width $W_\lambda$ of an absorption line — the total "area" it removes from the continuum — depends on the number of absorbing atoms:

$$W_\lambda = \int \left(1 - \frac{F_\lambda}{F_c}\right) d\lambda$$

where $F_\lambda$ is the line flux and $F_c$ is the continuum flux. The **curve of growth** relates $W_\lambda$ to column density, progressing through three regimes:
1. **Linear** (weak lines): $W_\lambda \propto N$ (column density)
2. **Flat** (saturated lines): $W_\lambda$ barely changes as $N$ increases (Doppler core is saturated)
3. **Square-root** (damped lines): $W_\lambda \propto \sqrt{N}$ (damping wings dominate)

Solar abundance analysis has revealed that the corona shows a systematic enhancement of elements with low first ionization potential (FIP < 10 eV) by factors of 2-4 compared to the photosphere. This "FIP effect" is a major constraint on models of chromospheric-coronal mass transport and is actively studied using spectroscopic data from missions like Hinode/EIS and IRIS.

---

## 3. Ground-Based Observatories

### 3.1 DKIST: The Daniel K. Inouye Solar Telescope

DKIST, located on Haleakala, Maui (altitude 3048 m), is the world's largest and most capable solar telescope. With a 4-meter primary mirror, it achieved first light in December 2019 and began science operations in 2022.

**Resolution**: The diffraction limit at 500 nm is:

$$\theta = 1.22 \frac{\lambda}{D} = 1.22 \times \frac{500 \times 10^{-9}}{4} \approx 0.03 \text{ arcsec}$$

At the Sun's distance, 1 arcsec corresponds to approximately 725 km, so 0.03 arcsec resolves structures as small as ~22 km — smaller than the photon mean free path in the photosphere. This is a revolutionary capability.

**Adaptive optics**: Earth's atmosphere blurs images (typical seeing: 0.5-2 arcsec). DKIST's adaptive optics system uses a deformable mirror that adjusts its shape hundreds of times per second to compensate for atmospheric turbulence, achieving diffraction-limited performance.

**Key instruments**:
- **VBI** (Visible Broadband Imager): High-resolution imaging in multiple visible channels
- **ViSP** (Visible Spectro-Polarimeter): Full Stokes polarimetry in the visible
- **DL-NIRSP** (Diffraction-Limited Near-IR Spectropolarimeter): NIR spectropolarimetry at the diffraction limit
- **Cryo-NIRSP** (Cryogenic Near-IR Spectropolarimeter): Cooled to reduce thermal background, designed for coronal observations including Fe XIII 1074.7 nm

**Science highlights**: DKIST has already delivered stunning images of sunspot fine structure, resolved convective cells smaller than ever before, and begun the long-awaited routine measurement of the coronal magnetic field — the single most important unmeasured quantity in solar physics.

### 3.2 GONG: Global Oscillation Network Group

GONG operates six identical instruments distributed around the world (Learmonth, Udaipur, El Teide, Cerro Tololo, Big Bear, Mauna Loa). This geographic distribution ensures nearly continuous coverage of solar oscillations, avoiding the day-night gaps that plague single-site observations.

GONG measures:
- Full-disk Dopplergrams (line-of-sight velocity maps) for helioseismology
- Full-disk magnetograms for space weather modeling (PFSS extrapolations)

The continuous helioseismic data from GONG has been instrumental in mapping the Sun's internal rotation, detecting the tachocline, and monitoring far-side active regions.

### 3.3 Other Major Facilities

- **Big Bear Solar Observatory** (BBSO, California): Hosts the 1.6m Goode Solar Telescope (GST), the largest operational solar telescope before DKIST. Pioneered high-resolution chromospheric observations and near-infrared spectropolarimetry.
- **GREGOR** (1.5m, Tenerife): European facility with excellent spectropolarimetric capabilities.
- **Swedish Solar Telescope** (SST, 1m, La Palma): Known for exceptional image quality and pioneering adaptive optics work.
- **Future: European Solar Telescope (EST)**: A planned 4.2m telescope for the Canary Islands, complementing DKIST with European access and advanced multi-conjugate adaptive optics.

### 3.4 Advantages and Limitations of Ground-Based Observing

Ground-based telescopes offer several advantages: large aperture (affordable at scales impossible for space), instrument flexibility (instruments can be upgraded or replaced), and long operational lifetimes. However, they are limited to visible and near-IR wavelengths (atmosphere absorbs UV, EUV, and X-rays), suffer from atmospheric seeing (partially corrected by AO), and operate only during daytime from each site (mitigated by networks like GONG).

---

## 4. Space-Based Observatories

Space eliminates atmospheric absorption and seeing, enabling observations in the ultraviolet, extreme ultraviolet, and X-ray wavelengths where the chromosphere, transition region, and corona shine brightest. The trade-off is smaller apertures, higher cost, and finite mission lifetimes.

### 4.1 SDO: Solar Dynamics Observatory (2010-)

SDO orbits Earth in a geosynchronous inclined orbit, providing nearly continuous solar coverage. It has fundamentally transformed solar physics by providing the first sustained, high-cadence, multi-wavelength view of the full solar disk.

**AIA (Atmospheric Imaging Assembly)**: Four telescopes with 10 wavelength channels:
- 7 EUV channels (94, 131, 171, 193, 211, 304, 335 A) sampling different temperatures from 60,000 K to 20 MK
- 2 UV channels (1600, 1700 A) for the upper photosphere and transition region
- 1 visible channel (4500 A) for photospheric context

Each channel produces 4096 $\times$ 4096 pixel images with 0.6 arcsec pixels every 12 seconds. That is roughly 1.5 terabytes of data per day — a data avalanche that spurred the application of machine learning to solar physics.

**HMI (Helioseismic and Magnetic Imager)**: Measures the photospheric magnetic field (full Stokes vector magnetograms every 12 minutes) and line-of-sight velocity (Dopplergrams every 45 seconds) using the Fe I 6173 A line. HMI data is essential for:
- Helioseismology (probing internal structure and rotation)
- Magnetic field boundary conditions for coronal models (PFSS, MHD)
- Active region monitoring for flare prediction
- Synoptic maps for solar wind modeling

**EVE (EUV Variability Experiment)**: Measures the total EUV irradiance spectrum (0.1-105 nm) with 10-second cadence, critical for understanding how solar EUV variations affect Earth's thermosphere and ionosphere.

### 4.2 SOHO: Solar and Heliospheric Observatory (1995-)

SOHO orbits the L1 Lagrange point, 1.5 million km sunward of Earth, providing an uninterrupted view of the Sun. After more than 30 years, it remains operational and scientifically productive.

**LASCO (Large Angle Spectrometric Coronagraph)**: Two coronagraphs still operating:
- **C2**: Field of view 1.5-6 $R_\odot$, revealing the inner corona and CME initiation
- **C3**: Field of view 3.7-30 $R_\odot$, tracking CMEs through the outer corona

LASCO has detected over 40,000 CMEs — the definitive CME catalog. It has also discovered over 4,500 sungrazing comets (mostly Kreutz group).

Other instruments: EIT (EUV imaging, predecessor to SDO/AIA), MDI (helioseismology, predecessor to HMI, now retired), UVCS (ultraviolet coronagraph spectrometer), SUMER (UV spectrometer).

### 4.3 Hinode (2006-)

A JAXA/NASA/ESA mission in Sun-synchronous polar orbit, providing uninterrupted viewing for 9 months per year.

- **SOT** (Solar Optical Telescope, 0.5m): Diffraction-limited (0.2 arcsec) visible imaging and spectropolarimetry. Provides the most detailed views of photospheric magnetic fine structure from space.
- **XRT** (X-Ray Telescope): Grazing-incidence X-ray imaging of the corona, sensitive to plasma above ~1 MK.
- **EIS** (EUV Imaging Spectrometer): Slit spectrometer covering 170-210 A and 250-290 A. Provides spectroscopic diagnostics (velocity, temperature, density) of the corona and transition region that imaging alone cannot.

Hinode/EIS data has been crucial for measuring coronal outflows at active region edges (possible slow solar wind sources), quantifying the FIP effect, and studying oscillations and waves in coronal loops.

### 4.4 IRIS: Interface Region Imaging Spectrograph (2013-)

IRIS targets the poorly understood interface between the chromosphere and corona — the region where the temperature jumps from ~6000 K to over 1 MK.

- **Capabilities**: UV spectrograph (133.2-135.8 nm, 138.9-140.7 nm) and slit-jaw imager (133, 140, 279.6, 283.4 nm)
- **Resolution**: 0.33 arcsec spatial, 1 second temporal cadence
- **Key spectral lines**: Mg II h & k (chromosphere), C II (transition region), Si IV (transition region, ~80,000 K)

IRIS has revealed ubiquitous small-scale dynamics in the chromosphere — rapid heating events, jets, and complex velocity patterns — that challenge our understanding of energy transport. Its combination of spectroscopic diagnostics with imaging context at high cadence makes it uniquely powerful for studying the chromosphere-corona connection.

### 4.5 STEREO: Solar Terrestrial Relations Observatory (2006-)

Two identical spacecraft launched into heliocentric orbits, one ahead (STEREO-A) and one behind (STEREO-B, contact lost 2014) Earth, gradually separating to provide stereoscopic views of the Sun and heliosphere.

**SECCHI instrument suite**:
- EUVI: EUV imager (similar to SOHO/EIT)
- COR1, COR2: Inner and outer coronagraphs
- HI-1, HI-2: Heliospheric imagers (wide-field cameras)

STEREO's stereoscopic perspective enabled the first 3D reconstruction of CME morphology and the first observation of CMEs propagating all the way from the Sun to Earth. Currently, STEREO-A continues to provide an off-Sun-Earth-line viewpoint that is invaluable for space weather monitoring.

---

## 5. Coronagraphs and Heliospheric Imagers

### 5.1 Why Coronagraphs Are Needed

The solar corona is approximately one million times fainter than the photospheric disk. From the ground, the corona is visible only during total solar eclipses, when the Moon acts as a natural occulting disk. Coronagraphs create artificial eclipses, blocking the photospheric light to reveal the faint corona.

The fundamental challenge is **scattered light**: even a tiny fraction of the photospheric intensity scattered by optical elements or diffraction around the occulting disk can overwhelm the coronal signal. Coronagraph design is essentially an exercise in stray light engineering.

### 5.2 The Lyot Coronagraph

Bernard Lyot invented the internally-occulted coronagraph in 1930. The key elements are:

1. **Objective lens** (O1): Forms an image of the Sun
2. **Occulting disk**: Placed at the focal plane, blocks the solar disk image
3. **Field lens**: Re-images the objective aperture
4. **Lyot stop**: Placed at the re-imaged aperture, blocks diffracted light from the objective edge
5. **Relay lens**: Forms the final coronal image

The Lyot stop is the crucial innovation: it intercepts the bright diffraction ring produced at the edge of the primary objective by the sharp occulting disk, dramatically reducing stray light.

### 5.3 Externally Occulted Coronagraphs

An alternative approach places the occulting disk ahead of the primary objective — an external occultation. This significantly reduces diffraction because the occulting disk is far from the entrance aperture, allowing sharper shadow boundaries.

SOHO/LASCO C2 and C3 use external occultation. The tradeoff is a larger inner field-of-view limit (the occultation must be oversized to accommodate pointing errors), but the stray light performance is superior, enabling observation of the faint outer corona.

### 5.4 The K-Corona and F-Corona

The coronagraph detects two components of coronal brightness:

- **K-corona** (Kontinuierlich, German for "continuous"): Photospheric light Thomson-scattered by free coronal electrons. The spectrum is a smoothed continuum (Fraunhofer lines are washed out by Doppler broadening from the hot electron velocities). The K-corona traces the coronal electron density distribution and is polarized (due to the geometry of Thomson scattering).
- **F-corona** (Fraunhofer): Photospheric light scattered by interplanetary dust. The Fraunhofer lines are preserved because the dust particles are cold and move slowly. The F-corona dominates beyond ~3-5 $R_\odot$.

Separating K from F components (using polarization) is essential for extracting the true coronal electron density.

### 5.5 Heliospheric Imagers

Heliospheric imagers (HIs) extend the coronagraph concept to much wider fields of view, tracking CMEs as they propagate through interplanetary space toward Earth and beyond.

STEREO's heliospheric imagers achieve this with wide-angle cameras and extremely careful baffling:
- **HI-1**: Field of view 15-84 $R_\odot$ (4-24 degrees from Sun center)
- **HI-2**: Field of view 66-318 $R_\odot$ (19-89 degrees from Sun center)

These instruments image CMEs by detecting Thomson-scattered photospheric light from the CME's electrons — the same physics as the K-corona, but at much greater distances. The challenge is separating the faint CME signal from the bright stellar background, zodiacal light, and stray light.

**Future: ESA Vigil mission** (planned ~2031): A spacecraft at the Sun-Earth L5 Lagrange point (trailing Earth by 60 degrees) will carry a coronagraph and heliospheric imager providing a side view of Earth-directed CMEs. This perspective is expected to dramatically improve CME arrival time predictions, which currently have uncertainties of ~12-18 hours.

---

## 6. Solar Radio Observations

### 6.1 Radio Emission Mechanisms

The Sun is a powerful radio source, but the emission mechanisms differ fundamentally from the optical/UV/EUV emission:

**Thermal bremsstrahlung (free-free)**: Electrons deflected by ions emit radiation. This produces the "quiet Sun" radio emission. The brightness temperature at frequency $\nu$ is related to the electron temperature of the optically thick layer at that frequency. Since the corona is optically thick at low radio frequencies, radio observations "see" coronal temperatures (~1-2 MK) at meter wavelengths, and chromospheric temperatures at millimeter wavelengths.

**Gyrosynchrotron emission**: Mildly relativistic electrons (energies ~100 keV to several MeV) spiraling in magnetic fields emit at harmonics of the gyrofrequency:

$$f_B = \frac{eB}{2\pi m_e c} \approx 2.8 \times 10^6 \, B \text{ Hz}$$

For $B = 100$ G, $f_B \approx 280$ MHz. Gyrosynchrotron emission from accelerated electrons during flares produces intense microwave bursts (typically 1-30 GHz), and the spectrum encodes both the magnetic field strength and the energetic electron distribution.

**Plasma emission**: The most dramatic solar radio bursts arise from a two-step process. Electron beams or shocks excite Langmuir (electrostatic) waves at the local plasma frequency:

$$f_p = \frac{1}{2\pi}\sqrt{\frac{n_e e^2}{\epsilon_0 m_e}} \approx 9 \sqrt{n_e} \text{ kHz}$$

where $n_e$ is in cm$^{-3}$. The physical meaning of the plasma frequency is fundamental: if you displace a slab of electrons from its equilibrium position relative to the (much heavier) ions, the resulting charge separation creates an electric field that pulls the electrons back. They overshoot, oscillate, and the natural frequency of this oscillation is $f_p$. In essence, $f_p$ is the frequency at which the electrostatic restoring force exactly balances electron inertia. The $\sqrt{n_e}$ dependence arises because higher density means a stronger restoring field (more charge separation per unit displacement), while the $\sqrt{1/m_e}$ factor reflects that lighter particles oscillate faster. Evaluating numerically: for a typical quiet corona with $n_e \sim 10^8$ cm$^{-3}$, $f_p \approx 90$ MHz (meter wavelengths), while for the dense chromosphere with $n_e \sim 10^{12}$ cm$^{-3}$, $f_p \approx 9$ GHz (microwave).

These Langmuir waves then convert to electromagnetic radiation at $f_p$ (fundamental) or $2f_p$ (harmonic). Because $n_e$ decreases with height in the corona, the emission frequency provides a proxy for height: higher frequency = denser plasma = lower altitude.

### 6.2 Solar Radio Burst Classification

Solar radio bursts are classified by their appearance on dynamic spectra (frequency vs. time plots):

**Type I (Noise Storms)**: Clusters of short-lived (~1 s), narrow-bandwidth bursts superimposed on a broadband continuum. Associated with active regions and their overlying magnetic fields. Can persist for hours to days. Mechanism likely involves electrons accelerated by small-scale reconnection events.

**Type II (Slow Drift Bursts)**: Emission bands that drift slowly (~1 MHz/s) from high to low frequency over tens of minutes. The slow drift corresponds to a disturbance moving outward through the corona at ~500-2000 km/s — the signature of a **CME-driven shock**. As the shock propagates to lower densities, the local plasma frequency decreases, producing the downward frequency drift. Type II bursts are important space weather indicators: they confirm that a CME is driving a shock capable of accelerating particles.

$$\frac{df}{dt} \propto \frac{dn_e}{dr} \cdot v_{\text{shock}}$$

**Type III (Fast Drift Bursts)**: Rapidly drifting emission (~20-100 MHz/s), sweeping from high to low frequency in seconds. Caused by electron beams (v ~ 0.1-0.5 c) streaming outward along open magnetic field lines, exciting plasma emission as they go. Type III bursts are closely associated with flares and indicate that energetic electrons have access to open field lines extending into the heliosphere. Interplanetary Type III bursts can be tracked from MHz frequencies near the Sun down to ~20 kHz at 1 AU.

**Type IV (Broadband Continuum)**: Broadband emission lasting minutes to hours, following major flares. Produced by electrons trapped in post-flare magnetic structures. The emission mechanism can include gyrosynchrotron, plasma emission, and synchrotron radiation, depending on the circumstances.

**Type V**: Brief broadband continuum following Type III bursts, lasting ~1-3 minutes. Less well understood; possibly due to electrons trapped behind the beam front.

### 6.3 Radio Instruments

Modern solar radio astronomy benefits from interferometric arrays that provide both spectral and spatial resolution:

- **LOFAR** (Low-Frequency Array, Netherlands): 10-240 MHz, arcsecond imaging at low frequencies, ideal for Type II and III burst studies
- **MWA** (Murchison Widefield Array, Australia): 80-300 MHz, wide field of view for solar burst imaging
- **VLA** (Very Large Array, New Mexico): 1-50 GHz, high-resolution imaging of active regions and flares
- **EOVSA** (Expanded Owens Valley Solar Array, California): 1-18 GHz, dedicated solar microwave spectral imaging, providing spatially-resolved microwave spectra of flares
- **e-CALLISTO** (Compound Astronomical Low-frequency Low-cost Instrument for Spectroscopy and Transportable Observatory): A global network of ~200 small radio spectrographs, providing continuous spectral monitoring of solar radio bursts

The combination of radio with EUV and X-ray observations provides a uniquely comprehensive view of particle acceleration and shock propagation during solar eruptions.

---

## 7. Multi-Wavelength Synthesis

The power of modern solar physics lies in combining observations across the electromagnetic spectrum. A single flare event, for instance, can be observed simultaneously in:

- **Radio** (Type III/IV): Tracks energetic electron beams and trapped populations
- **Hard X-rays** (RHESSI, Solar Orbiter/STIX): Images of electron impact sites (footpoints, loop-tops)
- **Soft X-rays** (Hinode/XRT, GOES): Hot flare plasma (~10-20 MK)
- **EUV** (SDO/AIA, multiple channels): Plasma at various temperatures, loop structures, waves
- **UV** (IRIS): Chromospheric evaporation, TR dynamics
- **Visible** (H$\alpha$, DKIST): Flare ribbons, chromospheric response
- **Magnetograms** (SDO/HMI): Pre-flare magnetic configuration, shear, flux cancellation

Each wavelength tells a different part of the story. Together, they provide a coherent narrative of energy release, particle acceleration, heating, and mass motion that no single instrument could deliver alone.

---

## Practice Problems

**Problem 1**: The Fe I 6173 A line is observed to be shifted by +0.021 A in a sunspot penumbra. Calculate the line-of-sight velocity component. Is the plasma moving toward or away from the observer? If this observation is near the solar limb, what component of the Evershed flow does this represent?

**Problem 2**: A coronal loop is observed in both the SDO/AIA 171 A channel (Fe IX, peak response at 0.7 MK) and the 193 A channel (Fe XII, peak response at 1.5 MK). The 171/193 intensity ratio is measured to be 0.8. Qualitatively, what does this tell you about the loop temperature? If the loop appears bright in 94 A (Fe XVIII, ~7 MK), what would you conclude?

**Problem 3**: A Type II radio burst begins at 150 MHz and drifts to 30 MHz over 10 minutes. Using the Newkirk coronal density model $n_e(r) = n_0 \times 10^{4.32 \times R_\odot/r}$ (with $n_0 = 4.2 \times 10^4$ cm$^{-3}$), estimate the heights at the start and end of the burst, and calculate the average shock speed.

**Problem 4**: An emission line from Si X is observed at two wavelengths: 356 A and 347 A. The 356/347 intensity ratio is measured to be 1.5 in region A and 3.0 in region B. Given that this ratio increases with electron density, which region has higher density? If the critical density for the 356 A line transition is $n_{\text{crit}} \sim 10^{10}$ cm$^{-3}$, discuss what the measured ratios imply about the density regime in each region.

**Problem 5**: DKIST's 4m mirror observes at 1074.7 nm (Fe XIII coronal line). Calculate the diffraction-limited angular resolution. Convert this to spatial resolution on the Sun (given 1 arcsec = 725 km). Compare with SDO/AIA's resolution of 1.2 arcsec at 17.1 nm. Why is the DKIST resolution at 1074.7 nm still useful despite being coarser than AIA in arcseconds?

---

**Previous**: [Solar Wind](./12_Solar_Wind.md) | **Next**: [Solar Energetic Particles](./14_Solar_Energetic_Particles.md)

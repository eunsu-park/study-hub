# Space Weather Indices

## Learning Objectives

- Derive the Dst index from ground magnetometer data and interpret its physical meaning in terms of ring current energy
- Understand the Kp/ap quasi-logarithmic geomagnetic index system and its relationship to disturbance amplitude
- Compute and interpret auroral electrojet indices (AE, AL, AU) from high-latitude magnetometer networks
- Explain solar activity indices (F10.7, Sunspot Number, TSI) and their role as proxies for solar output
- Classify space weather events using the NOAA G/S/R scales and connect each scale to its physical measurement
- Identify major data sources and archives for space weather research and operations
- Recognize the limitations of each index and when alternative measures are more appropriate

---

## 1. Geomagnetic Indices

Geomagnetic indices solve a fundamental problem in space physics: how do you reduce the complex, spatially varying, time-dependent magnetic field perturbations measured at the Earth's surface into a single number that characterizes the state of the magnetosphere?

The basic idea is straightforward. Magnetic disturbances at the ground arise from electric currents flowing in the magnetosphere and ionosphere. Different current systems produce characteristic signatures at different latitudes and local times. By selecting stations at appropriate locations and combining their measurements in specific ways, we can isolate the contribution of a particular current system and track its intensity over time.

Geomagnetic indices fall into two broad categories:

**Range indices** measure the amplitude of disturbance within a fixed time interval. They answer the question: "How disturbed was the magnetic field during this period?" The K-index family belongs here. A station records the maximum deviation from its quiet-day curve over a 3-hour window, and that range maps to an integer scale.

**Component indices** track the contribution of a specific magnetospheric current system by combining data from stations positioned to isolate that system's signature. The Dst index (ring current), AE index (auroral electrojets), and their variants belong here.

The choice of index depends on what physical process you want to monitor. A radiation belt modeler cares about Dst (ring current drives radial diffusion). An aurora forecaster watches AE (electrojet strength correlates with auroral activity). A power grid operator monitors dB/dt (rapid magnetic field changes induce geoelectric fields), which is not well captured by any single index.

---

## 2. Dst and SYM-H

### 2.1 The Dst Index

The Disturbance Storm Time (Dst) index is the workhorse of geomagnetic storm characterization. It has been computed continuously since 1957 and remains the primary measure of storm intensity.

**Station network.** Dst uses data from four near-equatorial magnetometer stations:

| Station | Code | Geographic Lat | Magnetic Lat |
|---------|------|---------------|-------------|
| Hermanus | HER | -34.4° | -33.3° |
| Kakioka | KAK | 36.2° | 26.0° |
| Honolulu | HON | 21.3° | 21.6° |
| San Juan | SJG | 18.1° | 28.0° |

These stations sit at low magnetic latitudes where the magnetic field perturbation from the symmetric ring current is strongest relative to auroral and high-latitude current systems. The four stations are distributed roughly evenly in longitude, which helps average out local time effects.

**Computation procedure:**

1. **Baseline removal.** For each station, establish a quiet-day baseline for the horizontal component $H$. This baseline accounts for the main field, secular variation, and quiet-day solar-driven currents (Sq). The deviation is $\Delta H = H_{\text{obs}} - H_{\text{quiet}}$.

2. **Secular variation correction.** The main geomagnetic field changes slowly over years to decades (secular variation). This trend must be subtracted to isolate magnetospheric contributions.

3. **Latitude normalization.** The horizontal perturbation from a symmetric equatorial ring current varies with magnetic latitude $\lambda$ as $\cos\lambda$. Each station's $\Delta H$ is divided by $\cos\lambda$ to normalize to the equator:

$$\Delta H_{\text{norm}} = \frac{\Delta H}{\cos\lambda}$$

4. **Averaging.** The hourly Dst is the average of the normalized $\Delta H$ values from all four (or available) stations:

$$\text{Dst} = \frac{1}{N}\sum_{i=1}^{N} \frac{\Delta H_i}{\cos\lambda_i}$$

### 2.2 Physical Interpretation

The Dst index is intimately connected to the total energy stored in the ring current through the Dessler-Parker-Sckopke relation:

$$\text{Dst} \approx -\frac{\mu_0}{4\pi} \frac{2 E_{RC}}{R_E^3 B_0}$$

where $E_{RC}$ is the total kinetic energy of the ring current particles, $R_E$ is the Earth's radius, and $B_0 \approx 3.1 \times 10^{-5}$ T is the equatorial surface field. This relation tells us that a Dst of $-100$ nT corresponds to a ring current energy of roughly $4 \times 10^{15}$ J.

The physical picture is elegant: energetic ions (primarily $\text{H}^+$ and $\text{O}^+$ at 10-200 keV) drift westward around the Earth, forming a toroidal current. This current produces a magnetic field that opposes the Earth's dipole at the equator, causing the negative Dst signature during storms.

### 2.3 Dst Corrections

The raw Dst contains contributions from current systems other than the ring current. The corrected index $\text{Dst}^*$ attempts to isolate the ring current contribution:

$$\text{Dst}^* = \frac{\text{Dst} - b\sqrt{P_{\text{dyn}}} + c}{1 + d}$$

The terms represent:

- **Magnetopause currents (Chapman-Ferraro):** Enhanced solar wind dynamic pressure $P_{\text{dyn}}$ compresses the magnetopause, strengthening the Chapman-Ferraro currents. These produce a positive $\Delta H$ at the equator, partially canceling the ring current depression. The correction $b\sqrt{P_{\text{dyn}}}$ (with $b \approx 7.26$ nT/$\sqrt{\text{nPa}}$) removes this contribution.

- **Magnetotail currents:** The cross-tail current sheet also contributes to equatorial $\Delta H$. The parameter $d$ accounts for the fraction of Dst attributable to tail currents (estimates range from 0.15 to 0.30). The constant $c$ is a quiet-time offset.

### 2.4 SYM-H Index

SYM-H is the high-resolution counterpart to Dst:

| Property | Dst | SYM-H |
|----------|-----|-------|
| Time resolution | 1 hour | 1 minute |
| Number of stations | 4 | ~6 |
| Provider | WDC Kyoto | ISGI |
| Available since | 1957 | 1981 |

SYM-H and Dst are highly correlated ($r > 0.95$) but SYM-H captures rapid variations — sudden commencements, substorm injections, and the fine structure of storm main phase — that hourly Dst smooths away. For modern research, SYM-H is generally preferred.

The complementary index ASY-H measures the asymmetric component of the ring current, which is strongest during the storm main phase when fresh injections create a partial ring current on the nightside.

### 2.5 Storm Classification

Geomagnetic storms are classified by their minimum Dst (or SYM-H):

| Category | Dst Range (nT) | Occurrence |
|----------|----------------|-----------|
| Weak | -30 to -50 | ~200/cycle |
| Moderate | -50 to -100 | ~100/cycle |
| Intense | -100 to -250 | ~30/cycle |
| Super-storm | < -250 | ~2-3/cycle |
| Extreme | < -500 | ~1-2/century |

The largest recorded storm was the Carrington Event of 1859, with estimated Dst around $-850$ to $-1050$ nT. The March 1989 event that caused the Quebec blackout reached $-589$ nT. The Halloween storms of 2003 reached $-422$ nT.

---

## 3. Kp and ap

### 3.1 The K-Index

The K-index is a local, quasi-logarithmic measure of geomagnetic disturbance. Each magnetometer station computes its own K value for every 3-hour Universal Time interval (00-03, 03-06, ..., 21-24 UT).

**Procedure:**

1. Remove the quiet-day variation (Sq curve) from the horizontal magnetic field components.
2. Measure the range (max minus min) of the residual within the 3-hour window.
3. Map this range to an integer K value (0 through 9) using a station-specific conversion table.

The conversion table is quasi-logarithmic: each step up in K roughly doubles the range. For a mid-latitude station, K = 9 corresponds to a range of about 500 nT, while K = 0 corresponds to less than about 5 nT.

Why quasi-logarithmic? Because geomagnetic disturbances span several orders of magnitude. A linear scale would either lack resolution at low activity or saturate during intense storms. The quasi-logarithmic mapping compresses the dynamic range while preserving the ability to distinguish between activity levels across the full spectrum.

### 3.2 From K to Kp

The planetary K-index (Kp) combines K values from 13 subauroral stations distributed in longitude around the Northern and Southern hemispheres. The process involves:

1. **Standardization.** Each station's K is converted to a standardized K (Ks) that accounts for the station's latitude. Higher-latitude stations naturally see larger disturbances, so their K thresholds are adjusted upward.

2. **Averaging.** Kp is the mean of the 13 standardized Ks values.

3. **Resolution.** Kp is reported in thirds: $0_o, 0_+, 1_-, 1_o, 1_+, \ldots, 9_-, 9_o$, giving 28 levels. The subscripts represent $-1/3, 0, +1/3$.

### 3.3 The ap and Ap Indices

The quasi-logarithmic Kp scale is inconvenient for quantitative work (you cannot meaningfully average logarithmic values). The equivalent amplitude $a_p$ converts each Kp value to a linear scale in nanoteslas:

| Kp | 0o | 1- | 1o | 1+ | 2- | 2o | 2+ | 3- | 3o | 3+ |
|----|----|----|----|----|----|----|----|----|----|----|
| ap | 0 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 12 | 15 |

| Kp | 4- | 4o | 4+ | 5- | 5o | 5+ | 6- | 6o | 6+ | 7- |
|----|----|----|----|----|----|----|----|----|----|----|
| ap | 18 | 22 | 27 | 32 | 39 | 48 | 56 | 67 | 80 | 94 |

| Kp | 7o | 7+ | 8- | 8o | 8+ | 9- | 9o |
|----|----|----|----|----|----|----|----|
| ap | 111 | 132 | 154 | 179 | 207 | 236 | 300+ |

The daily $A_p$ is simply the arithmetic mean of the eight 3-hourly $a_p$ values in a day. $A_p$ is widely used in thermospheric models (e.g., MSIS) to parameterize geomagnetic heating of the upper atmosphere.

### 3.4 Classification and Limitations

Activity classification based on Kp:

| Level | Kp Range | Description |
|-------|----------|-------------|
| Quiet | 0 - 2 | Minimal disturbance |
| Unsettled | 3 | Slightly above quiet |
| Active | 4 | Noticeable disturbance |
| Minor storm | 5 | NOAA G1 |
| Moderate storm | 6 | NOAA G2 |
| Strong storm | 7 | NOAA G3 |
| Severe storm | 8 | NOAA G4 |
| Extreme storm | 9 | NOAA G5 |

**Limitations of Kp:**
- The 3-hour cadence cannot resolve substorms (typical duration ~1-2 hours) or sudden impulses.
- Being a range index, it does not distinguish between different types of disturbance (storm vs. substorm vs. pulsations).
- Station distribution is biased toward the Northern Hemisphere and European longitudes.
- The quasi-logarithmic scale makes statistical analysis awkward; use $a_p$ instead.

---

## 4. AE, AL, AU

### 4.1 Auroral Electrojet Indices

The auroral electrojet (AE) family of indices monitors the intensity of ionospheric currents in the auroral zone. These currents flow in the E-region ionosphere (~100-120 km altitude) and are driven by magnetospheric convection and substorm processes.

**Station network.** Approximately 12 magnetometer stations are distributed in longitude along the auroral zone (magnetic latitude 65-70°N). This latitudinal band is chosen to lie beneath the main auroral electrojet currents.

**Computation:**

1. At each station, the $H$-component deviation from the quiet-day baseline is computed.
2. At each moment in time, find:
   - $\text{AU}(t) = \max_i [\Delta H_i(t)]$ — the largest positive deviation among all stations
   - $\text{AL}(t) = \min_i [\Delta H_i(t)]$ — the largest negative deviation
3. Derived indices:
   - $\text{AE} = \text{AU} - \text{AL}$ — total electrojet strength
   - $\text{AO} = \frac{\text{AU} + \text{AL}}{2}$ — asymmetric disturbance (analogous to ring current at auroral latitudes)

### 4.2 Physical Interpretation

The auroral electrojets are Hall currents driven by the convection electric field mapped from the magnetosphere:

- **Eastward electrojet** (afternoon sector): produces positive $\Delta H$ at the surface. Tracked by AU.
- **Westward electrojet** (morning/midnight sector): produces negative $\Delta H$. Tracked by AL.

During a substorm, the substorm current wedge diverts the cross-tail current through the ionosphere, dramatically enhancing the westward electrojet. This appears as a sharp negative excursion in AL (a "substorm onset" signature).

Typical values:
- Quiet: AE < 100 nT
- Moderate activity: AE ~ 200-500 nT
- Strong substorm: AE ~ 500-1500 nT
- Intense storm: AE > 2000 nT

### 4.3 The PC Index

The Polar Cap (PC) index takes a different approach. Instead of combining data from multiple stations, it uses a single station near each magnetic pole (Thule, Greenland for PCN; Vostok, Antarctica for PCS).

The PC index measures the magnetic perturbation projected onto the optimal direction determined by the statistical relationship between the perturbation and the interplanetary electric field $E_m = v B_T \sin^2(\theta/2)$, where $B_T$ is the transverse IMF and $\theta$ is the IMF clock angle.

$$\text{PC} = \frac{\Delta F_{\text{proj}} - \beta}{\alpha}$$

where $\alpha$ and $\beta$ are seasonally varying regression coefficients. The resulting index is dimensionally equivalent to mV/m and directly represents the transpolar electric field (convection strength).

The PC index has an important advantage: it responds within minutes to changes in the solar wind-magnetosphere coupling, making it useful for real-time monitoring. Its disadvantage is sensitivity to a single station's data quality.

### 4.4 SuperMAG Indices

The SuperMAG project addresses a fundamental limitation of the classical AE indices: only 12 stations sample the auroral zone. In reality, the auroral oval shifts in latitude depending on activity level, and 12 stations cannot fully resolve the spatial structure of the electrojets.

SuperMAG combines data from 300+ ground magnetometer stations worldwide to compute the SME (SuperMAG Electrojet) index. The SMU and SML components are analogous to AU and AL but use all available stations between 40° and 80° magnetic latitude, selecting the envelope of positive and negative deviations regardless of station identity.

Benefits of SME over AE:
- Better spatial coverage (fewer gaps in longitude)
- Captures electrojet activity even when the auroral oval shifts significantly equatorward during intense storms
- More accurate representation of total electrojet current

The trade-off is that SME is available only with post-processing delay and not in real time.

---

## 5. Solar Activity Indices

### 5.1 F10.7 Solar Radio Flux

The 10.7 cm (2800 MHz) solar radio flux, universally known as F10.7, is the most widely used index of solar activity for space weather applications.

**Measurement.** F10.7 has been measured daily at the Dominion Radio Astrophysical Observatory in Penticton, British Columbia, Canada since 1947. This unbroken 75+ year record makes it invaluable.

**Units.** Solar flux units (SFU), where $1 \text{ SFU} = 10^{-22} \text{ W m}^{-2} \text{ Hz}^{-1}$.

**Range over the solar cycle:**
- Solar minimum: ~65-70 SFU
- Solar maximum: ~150-300 SFU

**Why F10.7 matters.** The 10.7 cm emission originates from the chromosphere and low corona. It correlates extremely well ($r > 0.98$) with solar EUV (extreme ultraviolet) output, which is the primary driver of Earth's upper atmosphere heating and ionization. Since direct EUV measurements require space-based instruments (which were unavailable before the space age), F10.7 serves as the primary EUV proxy in atmospheric models like MSIS, IRI, and TIEGCM.

The relationship is not perfectly linear. At high activity levels, F10.7 tends to overestimate EUV relative to moderate activity. Some models now use the $S_{10.7}$ index or direct EUV proxies from the SOHO/SEM or SDO/EVE instruments.

### 5.2 Sunspot Number

The International Sunspot Number (ISN) is the oldest solar activity index, with records extending back to 1700 (and fragmentary data to 1610).

**Wolf number formula:**

$$R = k(10g + s)$$

where:
- $g$ = number of sunspot groups visible on the solar disk
- $s$ = total number of individual spots
- $k$ = observer-dependent correction factor (accounts for telescope, seeing, experience)

The factor of 10 weighting on groups reflects the empirical observation that sunspot groups are more reliably counted than individual spots. A single isolated spot still contributes $10 \times 1 + 1 = 11$.

**Version 2.0 (2015).** The sunspot number was recalibrated by SILSO (Royal Observatory of Belgium) to correct systematic errors in the historical record. Key changes:
- Removed the traditional division by 0.6 (Waldmeier's scaling)
- Corrected for a discontinuity around 1947
- Resulted in higher values for the earlier record

This recalibration has implications for understanding long-term solar variability and the possibility that previous solar cycles were as strong as recent ones.

### 5.3 Total Solar Irradiance (TSI)

TSI measures the total power per unit area from the Sun at 1 AU, integrated over all wavelengths.

$$\text{TSI} \approx 1361 \text{ W/m}^2$$

TSI varies by only about 0.1% ($\sim 1.4$ W/m$^2$) over the solar cycle, with the maximum at solar maximum. This seems paradoxical — more sunspots (which are dark) correlate with higher TSI — because the bright faculae that accompany sunspots more than compensate for the sunspot deficit.

On longer timescales, TSI variations are important for climate science. The Maunder Minimum (1645-1715), when sunspots nearly vanished, may have coincided with the "Little Ice Age," though the causal connection remains debated.

For space weather purposes, TSI is less relevant than spectral irradiance in the UV and EUV bands, which vary by much larger factors (up to 100% at Lyman-alpha, orders of magnitude at shorter EUV wavelengths).

---

## 6. NOAA Space Weather Scales

The Space Weather Prediction Center (SWPC) of NOAA developed three scales to communicate space weather impacts in a format accessible to non-specialists, analogous to the Saffir-Simpson hurricane scale.

### 6.1 G-Scale: Geomagnetic Storms

| Level | Kp | Description | Effects | Frequency (per cycle) |
|-------|-----|-------------|---------|----------------------|
| G1 | 5 | Minor | Weak power grid fluctuations; minor satellite operations impact; migratory animals affected | ~900 |
| G2 | 6 | Moderate | High-latitude power systems may have voltage alarms; spacecraft orbit corrections needed | ~360 |
| G3 | 7 | Strong | Voltage corrections required; surface charging on satellites; intermittent sat-nav issues | ~130 |
| G4 | 8 | Severe | Widespread voltage control problems; satellite tracking difficulties; HF radio intermittent | ~60 |
| G5 | 9 | Extreme | Widespread blackouts and transformer damage; satellite damage; complete HF radio blackout | ~4 |

### 6.2 S-Scale: Solar Radiation Storms

| Level | Flux (pfu)$^*$ | Description | Effects |
|-------|--------------|-------------|---------|
| S1 | 10 | Minor | Minor impact on HF in polar regions |
| S2 | $10^2$ | Moderate | Infrequent single-event upsets in satellites |
| S3 | $10^3$ | Strong | Degraded sat-nav accuracy; elevated radiation on high-latitude flights |
| S4 | $10^4$ | Severe | Satellite memory device problems; high radiation on polar flights |
| S5 | $10^5$ | Extreme | Satellite loss possible; complete HF blackout in polar regions; significant radiation hazard |

$^*$pfu = particle flux units ($\text{particles cm}^{-2} \text{s}^{-1} \text{sr}^{-1}$) for protons with energy > 10 MeV.

### 6.3 R-Scale: Radio Blackouts

| Level | X-ray Class | Description | Effects |
|-------|-------------|-------------|---------|
| R1 | M1 | Minor | Weak HF degradation on sunlit side; low-frequency navigation signals degraded |
| R2 | M5 | Moderate | Limited HF blackout on sunlit side; degraded low-frequency navigation for tens of minutes |
| R3 | X1 | Strong | Wide-area HF blackout for ~1 hour on sunlit side |
| R4 | X10 | Severe | HF blackout on most of sunlit side for 1-2 hours |
| R5 | X20+ | Extreme | Complete HF blackout on sunlit side lasting hours; low-frequency navigation errors |

Each scale links a physical measurement (Kp, proton flux, X-ray class) to practical impacts, making it straightforward for operators to understand the significance of a forecast.

---

## 7. Data Sources and Archives

### 7.1 OMNI Database

The OMNI database, maintained by NASA's Goddard Space Flight Center, is the gold standard for near-Earth solar wind data. It provides time-shifted, intercalibrated solar wind and magnetic field data at Earth's bow shock nose.

- **Time resolution:** 1-minute (high-res OMNI) and hourly
- **Coverage:** 1963 to present
- **Parameters:** $v$, $n$, $T$, $B$, $B_x$, $B_y$, $B_z$ (GSM/GSE), dynamic pressure, electric field, plasma beta, Alfven Mach number, Dst, Kp, AE, and more
- **Source missions:** ACE, Wind, DSCOVR, IMP-8, and many earlier spacecraft, cross-calibrated and time-shifted

OMNI is essential for statistical studies and model validation because it provides a uniform, quality-controlled dataset spanning the space age.

### 7.2 SuperMAG

SuperMAG is a worldwide collaboration of magnetometer networks providing a uniform dataset of ground magnetic field measurements.

- **Stations:** 300+ from multiple networks (INTERMAGNET, CANMOS, IMAGE, and others)
- **Processing:** Common baseline removal, coordinate rotation, quality control
- **Products:** SME/SMU/SML indices, magnetic field perturbation vectors
- **Access:** supermag.jhuapl.edu

### 7.3 GOES Satellites

The Geostationary Operational Environmental Satellites (GOES) provide real-time space weather data from geostationary orbit (6.6 $R_E$):

- **X-ray flux:** 1-8 A (long) and 0.5-4 A (short) channels — flare classification
- **Proton flux:** multiple energy channels (>1, >5, >10, >50, >100 MeV) — radiation storm detection
- **Magnetic field:** 3-axis magnetometer — magnetopause crossings, substorm signatures
- **Energetic particles:** >2 MeV electron flux — radiation belt monitoring

### 7.4 DSCOVR at L1

The Deep Space Climate Observatory (DSCOVR) orbits the Sun-Earth Lagrange point L1, approximately 1.5 million km upstream of Earth. It provides:

- **Real-time solar wind:** $v$, $n$, $T$, $\mathbf{B}$ with ~15-60 minutes lead time before arrival at Earth
- **Purpose:** operational space weather forecasting (replaced the aging ACE real-time system)
- **Instruments:** Faraday cup (plasma), fluxgate magnetometer

The L1 vantage point is the basis for all solar wind-based geomagnetic forecasting. The lead time depends on solar wind speed: ~60 min for slow wind (~350 km/s) and ~15 min for fast CME-driven flows (~1500 km/s).

### 7.5 CDAWeb and Other Archives

- **CDAWeb (Coordinated Data Analysis Web):** NASA's central portal for space physics mission data. Hosts data from 50+ missions.
- **WDC (World Data Centers):** Historical geomagnetic data, including Kyoto WDC for Dst and AE.
- **SDO/HMI:** Solar magnetic field data (magnetograms) — essential for flare prediction.
- **STEREO:** Multi-viewpoint solar observations — CME geometry.
- **Van Allen Probes (RBSP):** Radiation belt particle and field measurements (2012-2019).
- **MMS (Magnetospheric Multiscale):** High-resolution plasma measurements — reconnection studies.

---

## Practice Problems

**Problem 1.** A magnetometer station at magnetic latitude $\lambda = 30°$ measures a horizontal perturbation $\Delta H = -86.6$ nT during a geomagnetic storm. Calculate the contribution of this station to the Dst index. If all four Dst stations show similar corrected values, estimate the total ring current energy using the Dessler-Parker-Sckopke relation.

**Problem 2.** During a 3-hour interval, the maximum and minimum deviations of the horizontal field at a mid-latitude station are $+45$ nT and $-55$ nT (after quiet-day removal). The station's K=5 threshold is 70 nT. Determine the local K-index for this interval. If the global Kp turns out to be $5_o$, what is the corresponding $a_p$ value, and what NOAA G-scale level does this represent?

**Problem 3.** An auroral zone magnetometer network records the following maximum $H$-deviations at a particular minute: Station A: $+350$ nT, Station B: $-620$ nT, Station C: $+180$ nT, Station D: $-480$ nT, Station E: $+50$ nT. Compute AU, AL, AE, and AO. Identify which station is most likely under the eastward electrojet and which is under the westward electrojet.

**Problem 4.** The F10.7 index is 185 SFU. Using the approximate relationship $R \approx 1.1 \times (F_{10.7} - 67)$, estimate the sunspot number. If an observer sees 8 sunspot groups with a total of 63 individual spots, and the Wolf number matches your estimate, determine the observer's $k$-factor.

**Problem 5.** A solar proton event is measured with a peak flux of 5000 pfu (>10 MeV). Simultaneously, an X5 flare is in progress, and Kp reaches 8. Classify this event on all three NOAA scales (G, S, R). For each scale, describe one specific technological impact that operators should prepare for.

---

**Previous**: [Technological Impacts](./12_Technological_Impacts.md) | **Next**: [Forecasting Models](./14_Forecasting_Models.md)

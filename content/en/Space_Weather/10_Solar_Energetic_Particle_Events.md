# Solar Energetic Particle Events

## Learning Objectives

- Classify SEP events into impulsive and gradual types based on composition, duration, and acceleration mechanism
- Explain diffusive shock acceleration (first-order Fermi process) at CME-driven shocks and derive the resulting power-law spectrum
- Understand SEP transport in the heliosphere using the focused transport equation
- Apply velocity dispersion analysis to determine particle release times and path lengths
- Compute radiation doses from SEP spectra using stopping power and fluence integrals
- Assess radiation exposure risks for aviation crews on polar routes during SEP events
- Evaluate astronaut radiation hazards during SEP events for ISS, lunar, and deep-space missions

---

## 1. SEP Classification Review

Solar Energetic Particle (SEP) events are bursts of high-energy particles --- primarily protons, but also heavier ions and electrons --- accelerated to energies ranging from a few MeV to several GeV. These events represent one of the most significant radiation hazards in the space environment, threatening astronauts, satellite electronics, and even personnel on high-altitude polar flights.

The classical two-class paradigm, established by **Reames (1999)**, divides SEP events into two categories based on their acceleration mechanism and observed properties:

### 1.1 Impulsive SEP Events

Impulsive events are accelerated in **solar flares** through processes involving magnetic reconnection, wave-particle interactions, and stochastic acceleration:

| Property | Characteristic |
|----------|---------------|
| Duration | Hours (rapid onset, fast decay) |
| Longitude spread | Narrow (<30$^\circ$), magnetically connected |
| $^3$He/$^4$He ratio | Enhanced by factors of $10^3$--$10^4$ (vs. solar wind ~$5 \times 10^{-4}$) |
| Heavy ions (Fe/O) | Enhanced by factor ~10 |
| Electron-to-proton ratio | High (electron-rich) |
| Maximum proton energy | Typically <50 MeV |
| Radio association | Type III bursts (electron beams along open field lines) |
| Frequency | ~1000/year at solar maximum |

The dramatic $^3$He enrichment is one of the most remarkable phenomena in heliophysics. The mechanism involves **ion cyclotron resonance**: waves at frequencies near the $^3$He cyclotron frequency preferentially heat and accelerate $^3$He ions. Since $^3$He has a unique charge-to-mass ratio ($q/m = 2/3$ for $^3$He$^{2+}$, different from $^4$He$^{2+}$ with $q/m = 1/2$), it resonates with waves that other species do not.

Think of impulsive events as a focused, short burst --- like a flashbulb going off. The acceleration happens quickly, in a compact region, and affects only the particles that happen to be in the flare site.

### 1.2 Gradual SEP Events

Gradual events are accelerated at **shocks driven by coronal mass ejections** as they propagate outward through the corona and interplanetary medium:

| Property | Characteristic |
|----------|---------------|
| Duration | Days (gradual rise, slow decay) |
| Longitude spread | Wide (>60$^\circ$, sometimes 360$^\circ$) |
| $^3$He/$^4$He ratio | Solar/solar-wind values (~$5 \times 10^{-4}$) |
| Heavy ion composition | Approximately solar abundances |
| Proton-to-electron ratio | High (proton-dominated) |
| Maximum proton energy | Up to several GeV (Ground Level Enhancements) |
| Radio association | Type II bursts (shock signatures) |
| Frequency | ~10--20/year at solar maximum |

Gradual events are the ones that matter most for radiation hazards because they produce **higher fluxes** of **higher-energy protons** over **longer durations**. A single large gradual event can dominate the total radiation dose for an entire solar cycle.

Think of gradual events as a moving wall of acceleration --- the CME-driven shock sweeps through interplanetary space like a snowplow, continuously accelerating fresh particles as it goes.

### 1.3 The Mixed Reality

The two-class paradigm is a useful idealization, but reality is often more complex:

- Some events show **mixed** characteristics (impulsive composition but gradual time profile)
- A fast CME from the same active region as a flare can produce a **hybrid event** with both flare-accelerated and shock-accelerated populations
- **Seed particles**: CME shocks may re-accelerate suprathermal particles from prior impulsive events, imprinting impulsive-like composition on an otherwise gradual event
- **Twin-CME scenarios**: When two CMEs erupt in quick succession, the second shock propagates through the enhanced turbulence and seed population of the first, often producing the largest events

---

## 2. Diffusive Shock Acceleration (DSA)

### 2.1 The Physical Picture

Diffusive shock acceleration, also known as **first-order Fermi acceleration**, is the primary mechanism for producing the high-energy particles in gradual SEP events. The concept is beautifully simple:

Imagine a particle bouncing back and forth across a shock front. On each crossing, the particle gains energy because the upstream and downstream plasmas are converging (the shock is compressing the flow). Magnetic turbulence on both sides of the shock acts as "magnetic mirrors" that scatter the particle back toward the shock for another crossing.

The key ingredients are:
1. A **collisionless shock** (the CME-driven shock)
2. **Magnetic turbulence** upstream and downstream of the shock (scattering centers)
3. **Converging flows** across the shock (the source of energy gain)

### 2.2 Energy Gain Per Crossing

Consider a non-relativistic particle with velocity $v$ crossing a shock with upstream flow speed $u_1$ and downstream flow speed $u_2$ (in the shock frame). The fractional energy gain per complete cycle (upstream $\rightarrow$ downstream $\rightarrow$ upstream) is:

$$\frac{\Delta E}{E} \approx \frac{4(u_1 - u_2)}{3v} = \frac{4 \Delta u}{3v}$$

where $\Delta u = u_1 - u_2$ is the velocity difference across the shock.

This is called **first-order** Fermi acceleration because the energy gain is proportional to the first power of $\Delta u / v$. Compare this with second-order (stochastic) Fermi acceleration, where the energy gain per interaction is proportional to $(u/v)^2$ --- much slower.

The critical insight is that the energy gain is **systematic** (always positive), not random. Every time the particle crosses the shock, it gains energy, because the upstream and downstream media are always converging. This is unlike bouncing off randomly moving magnetic clouds, where some encounters accelerate and some decelerate.

### 2.3 The Power-Law Spectrum

The probability that a particle escapes downstream (and is lost from the acceleration process) after a given crossing is related to the **compression ratio** $r = u_1/u_2 = \rho_2/\rho_1$ of the shock.

The steady-state particle distribution that emerges from balancing acceleration against escape is a **power law in momentum**:

$$f(p) \propto p^{-\gamma} \quad \text{where} \quad \gamma = \frac{3r}{r-1}$$

This is one of the most celebrated results in astrophysics because it predicts a **universal power law** that depends only on the shock compression ratio, independent of the details of the scattering process.

For specific shock strengths:

| Compression Ratio $r$ | Spectral Index $\gamma$ | Differential Flux $j \propto E^{-\alpha}$ | Character |
|------------------------|------------------------|------------------------------------------|-----------|
| 2 (weak) | 6 | $\alpha = 2$ | Steep (soft) |
| 3 (moderate) | 4.5 | $\alpha = 1.75$ | Moderate |
| 4 (strong, gas limit) | 4 | $\alpha = 1.5$ | Flat (hard) |

For a strong shock in a monatomic ideal gas ($\gamma_{\text{gas}} = 5/3$), the maximum compression ratio is $r = 4$, giving $f \propto p^{-4}$, which translates to a differential energy flux $j(E) \propto E^{-1}$ for non-relativistic particles. This is a **very hard spectrum** --- it means the shock is remarkably efficient at accelerating particles to high energies.

In practice, observed SEP spectra often show:
- Spectral index $\alpha \approx 1.5$--3 (steeper than the strong shock limit due to finite shock geometry, escape, etc.)
- **Spectral breaks** or rollovers at high energies (where acceleration rate equals loss rate)
- **Streaming limits**: self-generated waves limit the intensity of particles escaping upstream

### 2.4 Maximum Energy

The maximum energy a particle can reach is determined by the balance between the **acceleration rate** and the **loss rate** (escape from the shock region or energy losses):

$$\frac{dE}{dt}\bigg|_{\text{accel}} \approx \frac{E \Delta u}{3 \lambda_\parallel}$$

where $\lambda_\parallel$ is the parallel mean free path (characterizing how quickly the particle diffuses back to the shock). The acceleration timescale is:

$$\tau_{\text{accel}} \sim \frac{3}{u_1 - u_2} \left(\frac{\kappa_1}{u_1} + \frac{\kappa_2}{u_2}\right)$$

where $\kappa_{1,2}$ are the diffusion coefficients upstream and downstream.

Typical maximum energies:
- **Slow CME** ($v_{\text{CME}} \sim 500$ km/s): $E_{\text{max}} \sim 1$--10 MeV
- **Fast CME** ($v_{\text{CME}} \sim 2000$ km/s): $E_{\text{max}} \sim 100$ MeV--1 GeV
- **Extreme events** (strongest shocks, favorable geometry): $E_{\text{max}} \sim$ several GeV (Ground Level Enhancement events)

**Quasi-parallel shocks** (where the magnetic field is nearly aligned with the shock normal) are more efficient at injecting thermal particles into the acceleration process and generally produce higher-energy particles than quasi-perpendicular shocks.

---

## 3. SEP Transport in the Heliosphere

Once accelerated at or near the Sun, SEP particles must travel through the heliosphere to reach Earth (or other observation points). The interplanetary magnetic field (IMF), structured as a Parker spiral, guides this transport, while magnetic turbulence scatters the particles.

### 3.1 The Focused Transport Equation

The standard framework for describing SEP propagation is the **focused transport equation**:

$$\frac{\partial f}{\partial t} + v\mu \frac{\partial f}{\partial s} + \frac{(1-\mu^2)v}{2L} \frac{\partial f}{\partial \mu} = \frac{\partial}{\partial \mu}\left(D_{\mu\mu} \frac{\partial f}{\partial \mu}\right)$$

where:
- $f(s, \mu, p, t)$ = particle distribution function
- $s$ = distance along the magnetic field line
- $\mu = \cos\alpha$ = cosine of the pitch angle (angle between particle velocity and magnetic field)
- $v$ = particle speed
- $L = -B/(dB/ds)$ = magnetic focusing length (characterizes how the field strength varies along the field line)
- $D_{\mu\mu}$ = pitch-angle diffusion coefficient (characterizes scattering by turbulence)

Each term has a clear physical meaning:

1. **$\partial f/\partial t$**: Time evolution
2. **$v\mu \, \partial f/\partial s$**: Free streaming along the field line (faster particles with $\mu \approx \pm 1$ travel faster)
3. **$\frac{(1-\mu^2)v}{2L} \frac{\partial f}{\partial \mu}$**: **Magnetic focusing** --- as the IMF expands outward, its strength decreases, and the magnetic mirror force focuses particles toward smaller pitch angles ($\mu \rightarrow \pm 1$), i.e., more field-aligned
4. **$\frac{\partial}{\partial \mu}(D_{\mu\mu} \frac{\partial f}{\partial \mu})$**: **Pitch-angle scattering** by turbulent fluctuations, which tends to isotropize the distribution

### 3.2 Two Limiting Transport Regimes

The competition between focusing and scattering defines two limiting regimes:

**Scatter-free (focusing-dominated) regime**: When the mean free path $\lambda_\parallel \gg L$ (weak scattering), particles stream nearly freely along the field line with minimal pitch-angle change. This produces:
- **Rapid onset** with clear velocity dispersion (fastest particles arrive first)
- **Strong anisotropy** (particles arrive primarily along the field direction)
- **Short event duration**

**Diffusive regime**: When $\lambda_\parallel \ll L$ (strong scattering), particles random-walk along the field line. The focused transport equation reduces approximately to a diffusion equation:

$$\frac{\partial f}{\partial t} \approx \kappa_\parallel \frac{\partial^2 f}{\partial s^2}$$

where $\kappa_\parallel = v\lambda_\parallel / 3$ is the parallel diffusion coefficient. This produces:
- **Slow, gradual onset** (particles take many scatterings to arrive)
- **Near-isotropic** distribution (particles arrive from all pitch angles)
- **Extended event duration** with slow decay

### 3.3 Mean Free Path

The parallel mean free path $\lambda_\parallel$ is a crucial but poorly constrained parameter:

$$\lambda_\parallel = \frac{3v}{8} \int_{-1}^{1} \frac{(1-\mu^2)^2}{D_{\mu\mu}} d\mu$$

Typical values for $\sim$10 MeV protons: $\lambda_\parallel \sim 0.08$--1 AU, with a common value around 0.1--0.3 AU.

The mean free path depends on:
- **Turbulence level**: More turbulence → shorter $\lambda_\parallel$ → more diffusive transport
- **Particle rigidity** (momentum per charge): Higher rigidity → longer $\lambda_\parallel$ (less affected by turbulence)
- **Turbulence geometry**: Slab vs. 2D vs. composite models give different scattering rates

### 3.4 Adiabatic Deceleration

As the solar wind expands, particles embedded in it lose energy adiabatically. For a particle with momentum $p$ in the expanding solar wind with speed $V_{sw}$:

$$\frac{dp}{dt} = -\frac{p}{3} (\nabla \cdot \mathbf{V}_{sw}) = -\frac{p V_{sw}}{3r}$$

for spherical expansion at heliocentric distance $r$. This effect is significant for low-energy particles (< few MeV) and long transport times, and it shifts the observed spectrum at 1 AU to lower energies compared to the source spectrum.

---

## 4. Velocity Dispersion and Onset Analysis

### 4.1 The Method

One of the most powerful observational techniques in SEP physics exploits the simple fact that faster particles arrive before slower ones. For a particle released at time $t_{\text{release}}$ from the Sun and traveling a path length $L_{\text{path}}$ along the magnetic field line with velocity $v$, the onset time at the observer is:

$$t_{\text{onset}}(v) = t_{\text{release}} + \frac{L_{\text{path}}}{v}$$

Rearranging:

$$t_{\text{onset}} = t_{\text{release}} + L_{\text{path}} \times \frac{1}{v}$$

This is a **linear equation** in $1/v$. By measuring the onset times at multiple energies (and hence velocities), one can plot $t_{\text{onset}}$ vs. $1/v$ and extract:
- **Slope** = $L_{\text{path}}$ (path length from Sun to observer)
- **Intercept** = $t_{\text{release}}$ (particle release time at the Sun)

### 4.2 What the Path Length Tells Us

For a well-connected event with minimal scattering, the path length along the Parker spiral from Sun to Earth is approximately:

$$L_{\text{Parker}} = \frac{r}{1 + (r\Omega/V_{sw})^2} \left[\sqrt{1 + \left(\frac{r\Omega}{V_{sw}}\right)^2} + \frac{V_{sw}}{r\Omega} \ln\left(\frac{r\Omega}{V_{sw}} + \sqrt{1 + \left(\frac{r\Omega}{V_{sw}}\right)^2}\right)\right]$$

For typical solar wind speed $V_{sw} = 400$ km/s, this gives $L_{\text{Parker}} \approx 1.15$ AU.

Observed values:
- $L_{\text{path}} \approx 1.0$--1.3 AU: nearly scatter-free transport (direct connection)
- $L_{\text{path}} \approx 1.5$--2+ AU: significant scattering (particles random-walk, increasing effective path)

### 4.3 Release Time Comparison

Comparing the inferred release time $t_{\text{release}}$ with the timing of associated solar events is revealing:

- $t_{\text{release}} \approx t_{\text{flare}}$: particles accelerated in the flare impulsive phase
- $t_{\text{release}} > t_{\text{flare}}$ by 10--30 minutes: particles accelerated by the CME-driven shock as it moves through the corona (the shock must build and become supercritical)
- Multiple release episodes: both flare and shock contribute at different times

This analysis has been critical in resolving the long-standing debate about whether flares or CME shocks are the primary accelerators in large SEP events (the answer: primarily shocks for gradual events, but with flare contributions in some cases).

### 4.4 Multi-Spacecraft Observations

With the **STEREO** twin spacecraft (separated by variable angles from Earth), combined with near-Earth observatories (**ACE**, **WIND**, **SOHO**), and more recently **Solar Orbiter** and **Parker Solar Probe**, velocity dispersion analysis can be performed simultaneously from multiple vantage points. This reveals:

- How the particle release depends on magnetic connection angle
- Whether the shock accelerates particles over a wide angular range simultaneously
- The role of cross-field transport in spreading particles to poorly connected observers

---

## 5. Radiation Dose Computation

### 5.1 Absorbed Dose

When energetic particles pass through matter (including human tissue), they deposit energy along their path through ionization and excitation of atoms. The **absorbed dose** quantifies this energy deposition:

$$D = \int_0^\infty \frac{dE}{dx}(E) \times \Phi(E) \, dE$$

where:
- $dE/dx$ = **stopping power** (energy lost per unit path length, also called Linear Energy Transfer or LET for ions), given by the **Bethe-Bloch formula**:

$$-\frac{dE}{dx} = \frac{4\pi n_e e^4 z^2}{m_e v^2} \left[\ln\frac{2m_e v^2}{I} - \ln(1-\beta^2) - \beta^2\right]$$

where $z$ is the particle charge number, $v$ its velocity, $n_e$ the electron density of the medium, $m_e$ the electron mass, and $I$ the mean ionization potential.

- $\Phi(E)$ = **fluence spectrum** (particles per unit area per unit energy, integrated over the event duration)

The unit of absorbed dose is the **Gray** (Gy):

$$1\;\text{Gy} = 1\;\text{J/kg}$$

### 5.2 Dose Equivalent and Radiation Weighting Factors

Different types of radiation cause different amounts of biological damage per unit of absorbed dose. A proton and a photon depositing the same energy do **not** cause the same biological harm. This is captured by the **dose equivalent**:

$$H = \sum_R w_R \times D_R$$

where $w_R$ is the **radiation weighting factor** for radiation type $R$ and $D_R$ is the absorbed dose from that radiation type.

| Radiation Type | Weighting Factor $w_R$ |
|----------------|----------------------|
| Photons (X-ray, $\gamma$) | 1 |
| Electrons, muons | 1 |
| Protons | 2 |
| Alpha particles | 20 |
| Heavy ions (C, O, Fe...) | 5--20 (depends on LET) |
| Neutrons | 5--20 (depends on energy) |

The unit of dose equivalent is the **Sievert** (Sv):

$$1\;\text{Sv} = 1\;\text{J/kg} \times w_R$$

The weighting factors reflect the fact that densely ionizing radiation (high LET) --- such as alpha particles and heavy ions --- causes more **clustered DNA damage** that is harder for cells to repair. A single heavy ion track can produce complex double-strand breaks in DNA that greatly increase the probability of cell death or carcinogenic mutation.

### 5.3 Depth-Dose Profile

For SEP events, the radiation environment varies significantly with depth in shielding material (or tissue). The critical concept is the **depth-dose curve**:

- **Protons (10--100 MeV)**: penetration depth ranges from ~1 mm to ~8 cm in water/tissue. Proton stopping power increases with decreasing energy (the **Bragg curve**), producing a dose peak near the end of range.
- **Skin dose**: dose at 0.01 mm depth --- sensitive to low-energy protons
- **Eye dose**: dose at 3 mm depth (lens of the eye)
- **BFO dose**: dose at blood-forming organs, approximated at 5 cm depth --- the regulatory-relevant quantity

Shielding effectiveness depends strongly on the spectrum: aluminum shielding of 1 g/cm$^2$ (about 3.7 mm) stops protons below ~30 MeV but is transparent to protons above ~100 MeV. This is why the highest-energy events (GLE events) are the most dangerous --- no practical amount of shielding can stop GeV protons.

---

## 6. Aviation Radiation Exposure

### 6.1 Background: Galactic Cosmic Rays

Even without SEP events, aircraft passengers and crew are exposed to elevated radiation from **galactic cosmic rays** (GCR) --- high-energy particles from outside the solar system that penetrate the atmosphere. The dose rate depends on:

- **Altitude**: higher altitude → less atmospheric shielding → higher dose. At cruise altitude (~10--12 km), the dose rate is roughly 50--100 times sea level.
- **Latitude**: the geomagnetic field deflects low-rigidity particles, providing maximum shielding at the equator and minimum at the poles. Polar routes receive ~3--6 $\mu$Sv/hr; equatorial routes ~1--2 $\mu$Sv/hr.
- **Solar cycle**: GCR flux is **anti-correlated** with solar activity (solar modulation). GCR dose is highest at solar minimum.

For a typical transatlantic flight (7 hours at mid-latitudes): dose $\approx$ 30--50 $\mu$Sv, roughly equivalent to a chest X-ray.

### 6.2 SEP Enhancement

During an SEP event, the radiation dose rate at aviation altitudes increases significantly, particularly for **polar routes** where geomagnetic shielding is weakest:

| Event Category | Additional Dose (polar route, 12 km) |
|---------------|--------------------------------------|
| Minor SEP | ~10 $\mu$Sv (negligible) |
| Moderate SEP | 0.01--0.1 mSv |
| Large SEP | 0.1--1 mSv |
| GLE (Ground Level Enhancement) | 1--10+ mSv |

For context, the Carrington-class event of 1859, if it occurred today, might deliver **10--20 mSv** for a single polar flight --- comparable to the annual occupational limit for radiation workers.

### 6.3 The CARI Model

The **CARI** (Civil Aerospace Medical Institute) model, maintained by the FAA, is the standard tool for computing aviation radiation doses:

- Inputs: flight route (departure/arrival airports, altitude profile), date, solar/geomagnetic conditions
- Physics: GCR transport through the atmosphere (LUIN code), geomagnetic cutoff rigidities, SEP contribution
- Output: effective dose for the flight in $\mu$Sv

The model is freely available online and is used by airlines to estimate crew cumulative doses for compliance with regulatory limits:
- **ICRP recommendation**: 20 mSv/year averaged over 5 years for occupational exposure
- **FAA guidance**: 20 mSv/year for aircrew, with 1 mSv/year recommended maximum for pregnant crew members
- **General public**: 1 mSv/year (the public limit is sometimes used for passenger protection during extreme events)

### 6.4 Airline Operational Response

When NOAA issues an SEP event warning (S-scale S2 or above), airlines operating polar routes may:

1. **Reduce altitude**: fly lower to gain more atmospheric shielding (increases fuel consumption)
2. **Reroute away from poles**: shift to lower-latitude great circle routes (adds distance and fuel, ~$10,000--$100,000 per flight)
3. **Restrict pregnant crew**: remove pregnant flight attendants from polar routes during events
4. **Cancel flights**: in extreme cases (rare, but considered for Carrington-class scenarios)

The tension between safety and economics is real: a major airline operates hundreds of flights daily, and rerouting even a fraction of polar flights during an event lasting several days costs millions.

---

## 7. Astronaut Radiation Exposure

### 7.1 ISS Radiation Environment

The International Space Station orbits at ~400 km altitude and ~51.6$^\circ$ inclination, within the protection of Earth's magnetosphere for most of its orbit. The primary radiation sources are:

- **GCR**: continuous background, ~0.3--0.5 mSv/day inside ISS
- **South Atlantic Anomaly (SAA)**: inner radiation belt protons at low altitude over South America, contributing intermittent dose spikes
- **SEP events**: episodic enhancement, can dominate dose for days during large events

ISS shielding varies by location:
- Typical: ~10 g/cm$^2$ aluminum equivalent
- Well-shielded areas (sleeping quarters): ~20 g/cm$^2$
- This shielding effectively stops protons below ~70--100 MeV

### 7.2 SEP Dose at ISS

During large SEP events, astronauts on the ISS may receive:

| Event Magnitude | BFO Dose (behind 10 g/cm$^2$ Al) |
|-----------------|-----------------------------------|
| Moderate (S2) | ~0.1--1 mSv |
| Large (S3--S4) | ~1--10 mSv |
| Extreme (S5, Carrington-class) | ~10--100 mSv |

For comparison, the monthly GCR dose is approximately 10--15 mSv, so a moderate SEP event adds roughly 10% to the monthly dose, while an extreme event could deliver a month's worth of additional dose in hours to days.

Current NASA limits for astronauts:

| Timescale | Dose Limit |
|-----------|-----------|
| 30-day | 250 mSv (BFO) |
| Annual | 500 mSv (BFO) |
| Career | 1--4 Sv (age and sex dependent, based on 3% REID) |

**REID** = Risk of Exposure-Induced Death (cancer mortality risk). NASA's career limit is set so that the additional cancer mortality risk does not exceed 3%.

### 7.3 EVA Hazard

Extravehicular activity (EVA) --- spacewalks --- represents the highest-risk scenario for astronaut radiation exposure during SEP events. The spacesuit provides minimal shielding:

- EMU (Extravehicular Mobility Unit) shielding: ~0.3 g/cm$^2$ (fabric, thermal layers)
- This stops protons below approximately 10 MeV only
- During a large SEP event: skin dose could reach **hundreds of mSv to Sv** level within hours

For this reason, EVAs are **terminated immediately** when an SEP event is detected, and astronauts retreat to the most heavily shielded area of the station. Real-time monitoring by instruments such as GOES energetic particle detectors and ISS-based dosimeters provides the critical early warning.

### 7.4 Lunar and Deep-Space Missions

Beyond LEO, astronauts lose the protection of Earth's magnetosphere entirely. This fundamentally changes the radiation risk calculus:

**Lunar surface**:
- No geomagnetic shielding (the Moon has no global magnetic field)
- Lunar regolith provides some shielding from below (2$\pi$ steradian reduction)
- During a large SEP event: unshielded astronaut could receive **lethal dose** (>1 Sv BFO) within hours
- **Storm shelter**: must be a mission requirement --- pre-positioned or excavated from regolith (~50 cm regolith provides ~50 g/cm$^2$)

**Mars transit**:
- ~6--9 months in interplanetary space, fully exposed to GCR and SEP
- GCR dose: ~0.3--0.5 Sv for one-way transit (already a significant fraction of career limit)
- Large SEP events: potential for acute radiation syndrome if shielding is inadequate
- Radiation is currently one of the **top three** technical challenges for human Mars missions (alongside propulsion and life support)

**The Apollo precedent**: During Apollo missions (1968--1972), astronauts were outside the magnetosphere for several days. By extraordinary luck, no large SEP event occurred during any Apollo mission. However, a major event occurred in **August 1972** --- between Apollo 16 (April) and Apollo 17 (December). Had astronauts been on the lunar surface during this event, they might have received a dose of 1--4 Sv (BFO), potentially causing acute radiation syndrome with symptoms ranging from nausea to death.

---

## Practice Problems

### Problem 1: DSA Spectral Index

A CME-driven shock has an upstream solar wind speed of 400 km/s and a shock speed of 1600 km/s in the solar frame.

(a) In the shock frame, what are the upstream flow speed $u_1$ and downstream flow speed $u_2$ (assuming the downstream plasma moves at the shock speed in the solar frame)? Hint: $u_1$ equals the shock speed minus the upstream wind speed.

(b) Calculate the compression ratio $r = u_1/u_2$ for a strong shock ($r = 4$) and for a moderate shock with $r = 3$.

(c) For each compression ratio, compute the spectral index $\gamma$ of the momentum distribution $f(p) \propto p^{-\gamma}$.

(d) Convert to the differential intensity spectral index: if $j(E) \propto E^{-\alpha}$, show that $\alpha = (\gamma - 1)/2$ for non-relativistic particles, and compute $\alpha$ for both cases.

### Problem 2: Velocity Dispersion

During an SEP event observed at 1 AU, the following onset times are recorded:

| Proton Energy (MeV) | $v/c$ | Onset Time (UT) |
|---------------------|-------|-----------------|
| 100 | 0.428 | 11:30 |
| 50 | 0.314 | 11:42 |
| 20 | 0.203 | 12:06 |
| 10 | 0.145 | 12:32 |

(a) Convert each velocity to km/s (using $c = 3 \times 10^5$ km/s).

(b) Plot (or tabulate) $t_{\text{onset}}$ vs. $1/v$ and determine the slope (path length) and intercept (release time) by linear regression or graphical estimation.

(c) Is the inferred path length consistent with scatter-free transport along the Parker spiral (~1.15 AU)? What does a longer path length imply?

(d) If the associated solar flare peaked at 11:05 UT, how much later were the particles released? What does this suggest about the acceleration mechanism?

### Problem 3: Radiation Dose from SEP Spectrum

An SEP event has a differential fluence spectrum $\Phi(E) = 10^9 \times E^{-2}$ protons/(cm$^2$ MeV) for proton energies 10 MeV $< E <$ 1000 MeV.

(a) Calculate the total proton fluence (integrated over energy):

$$F = \int_{10}^{1000} \Phi(E) \, dE$$

(b) The mean stopping power for protons in tissue over this energy range is approximately $\langle dE/dx \rangle \approx 5$ MeV$\cdot$cm$^2$/g. Estimate the absorbed dose:

$$D \approx \langle dE/dx \rangle \times F \times (1.6 \times 10^{-6}\;\text{erg/MeV}) \times (10^{-4}\;\text{J/erg}) / (10^{-3}\;\text{kg/g})$$

(c) Convert the absorbed dose to dose equivalent using $w_R = 2$ for protons. Express in mSv.

(d) Is this dose dangerous for an unshielded astronaut? Compare to the NASA 30-day limit of 250 mSv.

### Problem 4: Aviation Dose Estimate

A polar flight at 12 km altitude lasts 10 hours. The background GCR dose rate at this latitude and altitude is 5 $\mu$Sv/hr.

(a) Calculate the total GCR dose for the flight.

(b) During a moderate SEP event, the dose rate increases by an additional 20 $\mu$Sv/hr. What is the total dose (GCR + SEP)?

(c) If the airline reroutes the flight to an equatorial path (adding 3 hours flight time but reducing the GCR rate to 2 $\mu$Sv/hr and eliminating the SEP contribution at those latitudes), what is the total dose on the rerouted flight?

(d) A crew member flies 80 such polar flights per year. Compare the annual GCR dose (no SEP events) to the occupational limit of 20 mSv/year.

### Problem 5: Shielding Effectiveness

Protons are stopped in aluminum with a range (in g/cm$^2$) approximately given by:

$$R(E) \approx 0.0022 \times E^{1.77}\;\text{g/cm}^2$$

where $E$ is in MeV.

(a) Calculate the aluminum thickness (in g/cm$^2$ and in mm, using $\rho_{\text{Al}} = 2.7$ g/cm$^3$) needed to stop 30 MeV, 100 MeV, and 500 MeV protons.

(b) ISS shielding is approximately 10 g/cm$^2$ aluminum. What is the minimum proton energy that penetrates this shielding?

(c) An EVA suit provides ~0.3 g/cm$^2$ shielding. What is the minimum proton energy that penetrates the suit?

(d) Discuss why GLE events (with protons above 500 MeV) are particularly dangerous for astronauts even inside the ISS.

---

**Previous**: [Thermosphere and Satellite Drag](./09_Thermosphere_and_Satellite_Drag.md) | **Next**: [Geomagnetically Induced Currents](./11_Geomagnetically_Induced_Currents.md)

# Solar Energetic Particles

## Learning Objectives

- Distinguish between impulsive and gradual SEP events based on their composition, duration, and association with flares or CME-driven shocks
- Explain the physics of diffusive shock acceleration (DSA) at CME-driven shocks and derive the predicted spectral index
- Understand flare-related acceleration mechanisms and why they produce anomalous compositions (e.g., $^3$He enrichment)
- Describe SEP transport in the heliosphere using the focused transport equation and identify the key physical effects
- Analyze SEP event time-intensity profiles and explain velocity dispersion, reservoir effects, and longitude dependence
- Explain what Ground-Level Enhancements (GLEs) are and why they represent the most extreme SEP events
- Assess the radiation hazards posed by SEP events for astronauts, aviation, and spacecraft electronics

---

## 1. SEP Classification

### 1.1 The Two-Class Paradigm

Solar energetic particle (SEP) events — bursts of ions and electrons accelerated to high energies (keV to GeV) at or near the Sun — are among the most dramatic manifestations of solar activity. They were first detected in the 1940s as sudden increases in ground-level cosmic ray monitors during large flares.

Decades of observations have revealed that SEP events fall into two broad classes, each with distinct characteristics tracing back to fundamentally different acceleration mechanisms:

**Impulsive SEP Events:**
- **Duration**: Short, typically hours
- **Size**: Small, low intensity
- **Electron-rich**: High electron-to-proton ratio
- **$^3$He-rich**: The $^3$He/$^4$He ratio can be enhanced by factors of $10^3$ to $10^4$ relative to the solar value (~$5 \times 10^{-4}$). This is one of the most striking anomalies in all of space physics — $^3$He is a rare isotope, yet impulsive events can contain comparable numbers of $^3$He and $^4$He particles.
- **Heavy ion enhanced**: Fe/O ratio ~1, compared to the solar value of ~0.1. Elements heavier than iron can be enhanced by even larger factors.
- **High charge states**: e.g., Fe$^{20+}$ (indicating source temperatures >10 MK or stripping by dense plasma)
- **Association**: Solar flares, often microflares or small events
- **Longitude spread**: Narrow, typically $<30°$, centered on the magnetically connected footpoint
- **Frequency**: ~1000 events per year at solar maximum

**Gradual SEP Events:**
- **Duration**: Long, days to over a week
- **Size**: Can be very large, with intensities orders of magnitude above background
- **Proton-rich**: Dominated by protons, with approximately solar composition
- **Normal $^3$He**: $^3$He/$^4$He near solar value
- **Normal charge states**: e.g., Fe$^{10+}$-Fe$^{14+}$ (consistent with ambient coronal/solar wind temperatures of 1-2 MK)
- **Association**: CME-driven shocks (large, fast CMEs)
- **Longitude spread**: Wide, often $>60°$, sometimes approaching $360°$
- **Frequency**: ~10-20 large events per year at solar maximum

The physical picture: impulsive events result from acceleration in the compact flare reconnection region, while gradual events result from acceleration at the extended CME-driven shock that sweeps through the corona and interplanetary medium. Many events are "mixed," showing characteristics of both classes — a hybrid population that may result from flare-accelerated seed particles being further accelerated by a subsequent CME shock.

### 1.2 Why the Distinction Matters

The impulsive/gradual classification is not merely taxonomic. It has profound implications for:
- **Understanding acceleration physics**: Each class points to a different mechanism (reconnection vs. shock)
- **Space weather forecasting**: Gradual events pose the greatest radiation hazard, so identifying CME-driven shocks is essential
- **Tracing magnetic connectivity**: Impulsive events, being narrowly spread, are precise tracers of the magnetic field connecting the flare site to the observer

---

## 2. Acceleration at CME-Driven Shocks

### 2.1 The CME Shock as a Particle Accelerator

When a fast CME plows through the corona and solar wind at super-Alfvenic and super-magnetosonic speeds, it drives a collisionless shock ahead of it. This shock is a remarkably efficient particle accelerator — nature's largest accelerator within the heliosphere.

The key concept is **diffusive shock acceleration (DSA)**, also known as first-order Fermi acceleration. The basic idea, proposed by Axford, Leer, and Skadron (1977), Krymsky (1977), Bell (1978), and Blandford and Ostriker (1978), is elegantly simple:

1. A particle upstream of the shock is scattered by magnetic turbulence, sending it back across the shock
2. On crossing the shock, it encounters the downstream plasma moving toward it (in the shock frame), gaining energy from the converging flow
3. Downstream turbulence scatters it back upstream, where it again sees the upstream flow approaching
4. Each shock crossing yields an energy gain proportional to the velocity difference

Think of it like a ping-pong ball bouncing between two walls that are slowly closing in — each bounce increases the ball's speed.

### 2.2 Energy Gain per Cycle

In the shock rest frame, the upstream plasma approaches at velocity $u_1$ and the downstream plasma recedes at $u_2 = u_1/r$, where $r$ is the compression ratio. For a particle crossing the shock from upstream to downstream, the energy gain per round-trip cycle is:

$$\frac{\Delta E}{E} \approx \frac{4}{3} \frac{u_1 - u_2}{v} = \frac{4}{3} \frac{u_1(1 - 1/r)}{v}$$

for non-relativistic particles with speed $v \gg u_1$. This is a first-order process in $u/c$ (hence "first-order Fermi"), making it much more efficient than second-order (stochastic) Fermi acceleration where $\Delta E/E \propto (u/c)^2$.

### 2.3 The Resulting Spectrum

The remarkable prediction of DSA is a power-law energy spectrum. Particles that undergo more shock crossings reach higher energies, but each crossing has a probability of escape (being advected downstream with the flow). The competition between acceleration and escape produces:

$$f(E) \propto E^{-\gamma}$$

where the spectral index depends only on the compression ratio:

$$\gamma = \frac{r + 2}{2(r - 1)}$$

For a **strong shock** with Mach number $M \gg 1$, the adiabatic compression ratio for a monatomic ideal gas is $r = 4$ (the maximum for a single shock with $\Gamma = 5/3$), giving:

$$\gamma = \frac{4 + 2}{2(4 - 1)} = \frac{6}{6} = 1$$

This corresponds to a differential number spectrum $dN/dE \propto E^{-2}$ — a very hard (flat) spectrum. For weaker shocks ($r < 4$), the spectrum is softer (steeper).

This prediction is beautifully universal: it depends on no details of the scattering process, the magnetic field geometry, or the particle species. It is a robust consequence of the converging flow geometry. However, real SEP spectra often deviate from a pure power law, showing rollovers at high energies and broken power-law shapes.

### 2.4 Shock Geometry and Injection

The angle $\theta_{Bn}$ between the upstream magnetic field and the shock normal plays a critical role:

- **Quasi-parallel shocks** ($\theta_{Bn} < 45°$): Particles can stream back upstream along the field. Injection from the thermal pool is relatively easy, but the acceleration is slower because scattering in the upstream region must be strong.
- **Quasi-perpendicular shocks** ($\theta_{Bn} > 45°$): Particles gain energy faster (shock drift acceleration supplements DSA), but injection from the thermal pool is more difficult because particles must scatter across the magnetic field to return upstream.

The injection problem — how thermal particles gain enough energy to "enter" the DSA process — remains one of the outstanding puzzles. Suprathermal populations (particles already above thermal energies from earlier events or wave-particle interactions) may serve as seed particles, easing the injection requirement.

### 2.5 Maximum Energy

The maximum particle energy is limited by the condition that the particle's diffusion length (mean free path / acceleration time) not exceed the shock size:

$$E_{\max} \propto u_1 B R_{\text{shock}} / \kappa$$

where $\kappa$ is the diffusion coefficient. Typical CME-driven shocks in the corona can accelerate protons to ~100 MeV-1 GeV. The fastest, most extended shocks near the Sun produce the highest energies (GLEs).

---

## 3. Flare Particle Acceleration

### 3.1 Reconnection-Related Acceleration

In a solar flare, magnetic reconnection releases stored magnetic energy on timescales of minutes. The reconnection site contains strong electric fields — both the reconnection electric field in the current sheet and turbulent fields in the reconnection outflows.

**Direct electric field acceleration**: The reconnection electric field $E_{\text{rec}} \sim v_{\text{in}} B / c$ can directly accelerate particles. For reconnection inflows of $v_{\text{in}} \sim 100$ km/s and $B \sim 100$ G, $E_{\text{rec}} \sim 100$ V/m over distances of $\sim 10^4$ km, yielding energies of $\sim 1$ GeV in principle. However, the actual geometry is more complex — particles drift out of the current sheet, and the field is not uniform.

**Contracting magnetic islands**: In fragmented (plasmoid-dominated) reconnection, multiple magnetic islands form and merge. Particles trapped between contracting islands gain energy at each contraction — a first-order Fermi process within the reconnection layer. Simulations show this can efficiently produce power-law spectra.

### 3.2 Stochastic Acceleration and $^3$He Enrichment

Stochastic (second-order Fermi) acceleration involves resonant interaction between particles and MHD turbulence. Cascading turbulence in the reconnection outflow region provides a spectrum of wave modes.

The $^3$He enrichment is one of the most informative diagnostics. The gyrofrequency of an ion is:

$$\Omega = \frac{ZeB}{Am_p c}$$

where $Z$ is the charge and $A$ is the mass number. For $^3$He$^{2+}$: $\Omega(^3\text{He}^{2+}) = 2eB/(3m_p c)$. This frequency happens to fall between those of protons ($eB/m_p c$) and $^4$He$^{2+}$ ($eB/(2m_p c)$).

Ion cyclotron waves near the $^3$He gyrofrequency can resonantly couple to $^3$He$^{2+}$ ions, preferentially accelerating them while barely affecting the far more abundant protons and $^4$He. This selective heating through cyclotron resonance can produce the extreme $^3$He/$^4$He enhancements observed in impulsive events.

Similarly, heavy ion enhancements can arise from resonance with higher-harmonic cyclotron waves, preferentially accelerating ions with specific charge-to-mass ratios. The high charge states (e.g., Fe$^{20+}$) can result from either acceleration in very hot ($>10$ MK) flare plasma or from stripping as ions pass through dense regions after partial acceleration.

---

## 4. SEP Transport in the Heliosphere

### 4.1 The Focused Transport Equation

Once accelerated, SEPs must travel from the Sun to the observer (typically at 1 AU). The transport is governed by the **focused transport equation**:

$$\frac{\partial f}{\partial t} + v \mu \frac{\partial f}{\partial s} + \frac{(1 - \mu^2)}{2L} v \frac{\partial f}{\partial \mu} = \frac{\partial}{\partial \mu}\left(D_{\mu\mu} \frac{\partial f}{\partial \mu}\right) + Q$$

where:
- $f(s, \mu, v, t)$ is the particle distribution function
- $s$ is the distance along the magnetic field line
- $\mu = \cos\alpha$ is the pitch-angle cosine ($\alpha$ = angle between particle velocity and magnetic field)
- $v$ is the particle speed (assumed constant for scatter-dominated transport)
- $L = -B / (dB/ds)$ is the focusing length
- $D_{\mu\mu}$ is the pitch-angle diffusion coefficient
- $Q$ is the source term

Each term has a clear physical meaning:

### 4.2 Physical Effects

**Magnetic focusing** (third term on left): As a particle moves outward from the Sun, the magnetic field strength decreases ($B \propto 1/r^2$ for radial field). Conservation of the first adiabatic invariant ($\mu_m = mv_\perp^2/2B$) means $v_\perp$ decreases, so $v_\parallel$ increases — the particle is "focused" into a tighter beam around the field direction. This is analogous to a ball rolling down a widening funnel: it straightens out. The focusing length $L$ characterizes the spatial scale of this effect.

**Pitch-angle scattering** (right side): Magnetic turbulence — fluctuations in the interplanetary magnetic field — randomizes particle pitch angles. This opposes focusing and causes spatial diffusion. The pitch-angle diffusion coefficient $D_{\mu\mu}$ depends on the turbulence power spectrum and resonance conditions.

The interplay of focusing and scattering defines the transport regime:
- **Scatter-free**: If scattering is negligible, particles arrive as a focused beam with velocity dispersion (faster particles first). Onset is sharp.
- **Scatter-dominated** (diffusive): If scattering is strong, transport approaches spatial diffusion. Onset is gradual, profiles are broad.

Most real events fall between these extremes.

### 4.3 Mean Free Path

The scattering mean free path $\lambda$ quantifies the average distance a particle travels before its pitch angle is significantly randomized:

$$\lambda = \frac{3v}{4} \int_0^1 \frac{(1 - \mu^2)^2}{D_{\mu\mu}} d\mu$$

Typical values at 1 AU range from 0.08 to 0.3 AU for protons at ~10-100 MeV, but can vary by over an order of magnitude depending on the turbulence level. The mean free path tends to:
- Increase with particle rigidity (higher-energy particles "see" fewer resonant fluctuations)
- Decrease during disturbed periods (enhanced turbulence)
- Vary with heliocentric distance

**Adiabatic deceleration**: As the solar wind expands, particles embedded in it lose energy. For a radially expanding solar wind with speed $V_{sw}$:

$$\frac{dT}{dt} \approx -\frac{2}{3} \frac{V_{sw}}{r} T$$

This effect is significant for low-energy particles during prolonged events, reducing observed energies below the acceleration spectrum.

---

## 5. SEP Event Profiles

### 5.1 Velocity Dispersion

In the early phase of an event, faster particles arrive before slower ones. If particles are released simultaneously at the Sun and travel scatter-free along a field line of length $L$, the arrival time is:

$$t(v) = t_0 + L/v$$

Plotting onset time versus $1/v$ yields a straight line whose slope gives the path length $L$ and whose intercept gives the release time $t_0$. This "velocity dispersion analysis" (VDA) typically yields path lengths of 1.1-1.3 AU — longer than the 1 AU radial distance, consistent with the spiral geometry of the interplanetary magnetic field.

For a Parker spiral with solar wind speed $V_{sw} = 400$ km/s, the field line length from the Sun to 1 AU is:

$$L = \int_0^{r_E} \sqrt{1 + (\Omega r / V_{sw})^2} \, dr \approx 1.15 \text{ AU}$$

where $\Omega$ is the solar rotation rate.

### 5.2 Time-Intensity Profiles and Connection Longitude

The shape of the time-intensity profile depends critically on the observer's magnetic connection to the acceleration source:

- **Well-connected events** (source near W50-W60 for Parker spiral): Rapid rise (minutes to ~1 hour), sharp peak, gradual decay. The field line directly connects the shock/flare to the observer.
- **Poorly-connected events** (eastern sources): Slow rise over many hours, broad peak. Particles must diffuse across field lines to reach the observer.
- **Western events** (W80-W90): Can show very prompt onsets if the shock has a wide extent that reaches the connected field line quickly.

The **Parker spiral** geometry is key: the magnetic field line from the observer traces back to a solar longitude approximately $50°$-$60°$ west of the sub-observer point (for $V_{sw} \approx 400$ km/s). Events at this "best-connected" longitude produce the most prompt, intense signals.

### 5.3 The Reservoir Effect

Late in gradual SEP events (days after onset), a remarkable phenomenon occurs: particle intensities and spectra become nearly uniform across a broad region of the inner heliosphere. Observers at very different longitudes — even on opposite sides of the Sun — measure similar spectra.

This "reservoir" effect suggests that the particles fill a large volume bounded by magnetic structures (possibly the CME-driven shock itself or large-scale interplanetary structures). The trapping and redistribution homogenize the population. The reservoir is important because it means that SEP radiation hazards persist long after the flare and CME have passed.

---

## 6. Energy Spectra

### 6.1 Spectral Shapes

SEP energy spectra are not pure power laws. They typically exhibit:

**Low energies** ($E < E_{\text{break}}$): Approximate power law $dJ/dE \propto E^{-\gamma_1}$ with $\gamma_1 \sim 1-3$.

**High energies** ($E > E_{\text{break}}$): Steeper power law or exponential rollover. A common functional form is the **Band function** (a double power law with smooth connection):

$$\frac{dJ}{dE} \propto \begin{cases} E^{-\gamma_1} \exp(-E/E_0) & E < (\gamma_2 - \gamma_1) E_0 \\ E^{-\gamma_2} & E > (\gamma_2 - \gamma_1) E_0 \end{cases}$$

where $E_0$ is the e-folding energy characterizing the spectral break.

The spectral break energy varies from event to event: a few MeV for small events, up to ~100 MeV for the largest. It depends on the shock parameters, seed population, and scattering conditions.

### 6.2 The Streaming Limit

A remarkable self-regulating mechanism limits SEP intensities at high energies. High-energy protons streaming outward from the shock excite Alfven waves through the streaming instability. These waves, in turn, scatter the protons, limiting their outward flux. The result is an upper bound on near-Sun intensities — the **streaming limit**:

$$j_{\max}(E) \propto \frac{B}{r} v_A$$

where $v_A$ is the Alfven speed. This self-generated wave barrier acts like a traffic jam: when too many particles try to stream past, they create turbulence that slows them down. The streaming limit produces a spectral break — above a certain energy, the spectrum softens because the highest-energy particles are less effectively confined by self-generated waves.

Observationally, the streaming limit is evidenced by the fact that peak intensities at ~10-100 MeV show less event-to-event variation than lower-energy intensities: large and moderate events converge toward a common maximum.

---

## 7. Ground-Level Enhancements (GLEs)

### 7.1 What Are GLEs?

Ground-Level Enhancements are the rarest and most extreme SEP events. They occur when protons are accelerated to energies exceeding ~500 MeV — energetic enough to penetrate Earth's magnetosphere and atmosphere. These particles initiate nuclear cascades (hadronic showers) in the atmosphere, producing secondary neutrons, muons, and other particles that reach ground-level detectors.

The worldwide network of **neutron monitors** — detectors designed to measure cosmic ray secondary neutrons — records GLEs as sudden increases above the galactic cosmic ray background. A typical GLE shows a 5-50% increase in neutron monitor count rate, though the largest events (e.g., February 23, 1956 — GLE 5) recorded increases exceeding 4000%.

### 7.2 Statistics and Properties

Since systematic monitoring began in 1942, approximately 73 GLEs have been recorded (through 2025). Key statistics:
- **Frequency**: Roughly 1 per year at solar maximum, with none during deep solar minimum
- **Solar cycle dependence**: GLEs cluster near and just after solar maximum
- **Source events**: Always associated with both a major flare (typically X-class or strong M-class) and a fast CME (>1000 km/s). The combination suggests that both flare reconnection and shock acceleration contribute.
- **Longitude**: Preferentially from well-connected longitudes (W20-W90), though some events from eastern longitudes produce GLEs via cross-field diffusion or wide shock extent

### 7.3 Notable GLEs

**GLE 5 (1956 Feb 23)**: The largest GLE in the instrumental record. Neutron monitor increases of >4000% at some stations. Associated with an intense (likely >X10) flare from a highly active region.

**GLE 69 (2005 Jan 20)**: Among the fastest onset GLEs. Relativistic protons arrived at Earth within approximately 10 minutes of the flare onset, implying acceleration to $>1$ GeV within the first few minutes of the event. The rapid onset suggests acceleration very close to the Sun, possibly by the flare reconnection process rather than (or in addition to) the CME shock.

**GLE 72 (2017 Sep 10)**: The most recent GLE at the time of writing, associated with an X8.2 flare and a fast CME (~3000 km/s) from AR 12673. Notably, this was a limb event (source near W90), providing constraints on the acceleration site.

### 7.4 Scientific Importance

GLEs are uniquely important because:
1. They probe the highest-energy acceleration processes at the Sun
2. Neutron monitor data provides ground-truth measurements of the most energetic component
3. Their rapid onsets constrain the acceleration timescale and location
4. The anisotropy measured by the global neutron monitor network constrains the transport path
5. They represent the most hazardous radiation events for human spaceflight and aviation

---

## 8. Radiation Hazards

### 8.1 Radiation Dosimetry

To quantify biological radiation damage, several quantities are used:

**Absorbed dose** $D$: Energy deposited per unit mass of tissue:

$$D = \frac{\Delta E}{\Delta m}$$

Measured in Gray (Gy), where 1 Gy = 1 J/kg. For SEP protons at typical energies (10-100 MeV), 1 Gy roughly corresponds to a fluence of $\sim 10^9$-$10^{10}$ protons/cm$^2$.

**Linear Energy Transfer (LET)**: The energy deposited per unit path length as a charged particle traverses tissue:

$$\text{LET} = -\frac{dE}{dx}$$

For protons: LET increases as the proton slows down, peaking sharply at the Bragg peak just before the proton stops. At 100 MeV, LET ~ 0.7 keV/$\mu$m; at 1 MeV, LET ~ 25 keV/$\mu$m.

**Dose equivalent** $H$: Accounts for the varying biological effectiveness of different radiation types:

$$H = Q \times D$$

Measured in Sievert (Sv). The quality factor $Q$ depends on LET:
- Low-LET radiation (photons, high-energy protons): $Q \approx 1$
- High-LET radiation (heavy ions, low-energy protons): $Q$ up to 20

### 8.2 Astronaut Radiation Exposure

In space, outside the protection of Earth's magnetosphere and thick atmosphere, SEP events pose a serious radiation hazard. Astronauts receive radiation from three sources:
1. **Galactic cosmic rays (GCRs)**: Continuous low flux of highly energetic particles (~0.5-1 mSv/day in interplanetary space)
2. **SEP events**: Episodic, potentially very high dose rates
3. **Trapped radiation** (Van Allen belts): Relevant in Earth orbit

**Dose limits**: NASA career limits are 1-4 Sv (depending on age and sex at first exposure), with a 30-day limit of 0.25 Sv and a single-event limit of 0.15 Sv to blood-forming organs. A large SEP event (comparable to August 1972 — between Apollo 16 and 17) could deliver 0.1-1 Sv to an astronaut behind typical spacecraft shielding (~5 g/cm$^2$ Al equivalent) in hours to days.

For a deep-space mission (e.g., Mars transit), the absence of Earth's magnetosphere means that:
- Average GCR dose rate is ~0.5-0.8 mSv/day
- A single large SEP event could deliver months' worth of equivalent GCR dose in hours
- Storm shelters with enhanced shielding (~20 g/cm$^2$) can reduce SEP dose by ~10x

### 8.3 Aviation Radiation

At commercial aviation altitudes (10-12 km), Earth's atmosphere provides ~200-300 g/cm$^2$ of shielding, which stops most SEP protons below ~500 MeV. However, during GLEs, the highest-energy protons penetrate to flight altitudes, and secondary particle cascades increase the radiation field.

Typical radiation dose rates during a GLE for a polar flight:
- **Average GLE**: ~0.01-0.05 mSv additional dose (comparable to normal flight dose)
- **Large GLE**: ~0.1-1 mSv additional dose
- **Extreme Carrington-class event**: Potentially >1 mSv for a single polar flight (unclear, as no such event has occurred in the jet age)

Airlines may reroute polar flights to lower latitudes during large SEP events, where geomagnetic shielding (the geomagnetic cutoff rigidity) provides additional protection by deflecting lower-energy particles.

### 8.4 Spacecraft Electronics

SEPs also affect spacecraft through:
- **Single-Event Effects (SEEs)**: A single heavy ion or high-energy proton deposits enough charge in a transistor to flip a bit (Single-Event Upset, SEU), latch a circuit (Single-Event Latchup), or destroy a gate oxide (Single-Event Gate Rupture). SEE rates increase dramatically during large SEP events.
- **Total Ionizing Dose (TID)**: Cumulative damage from chronic radiation exposure degrades electronic components over the mission lifetime.
- **Displacement damage**: Energetic particles displace atoms in semiconductor crystal lattices, degrading solar cells and detectors.

Solar energetic particles are thus a multi-faceted hazard that drives the design of spacecraft shielding, mission planning, and real-time space weather monitoring.

---

## Practice Problems

**Problem 1**: A CME drives a shock with compression ratio $r = 3.5$ through the corona. Calculate the predicted DSA spectral index $\gamma$ for the accelerated proton distribution $f(E) \propto E^{-\gamma}$. Convert this to the differential intensity spectrum $dJ/dE \propto E^{-\delta}$, noting that $\delta = \gamma + 1$. How does this compare with the "universal" strong-shock limit? What physical conditions might produce a compression ratio less than 4?

**Problem 2**: An impulsive SEP event is observed with $^3$He/$^4$He = 0.5 (compared to solar value $5 \times 10^{-4}$). Calculate the enrichment factor. If the gyrofrequency of $^3$He$^{2+}$ is $\Omega = 2eB/(3m_p)$ and that of $^4$He$^{2+}$ is $\Omega = eB/(2m_p)$, calculate the ratio $\Omega(^3\text{He}^{2+})/\Omega(^4\text{He}^{2+})$. Explain qualitatively why ion cyclotron waves near this frequency range could preferentially accelerate $^3$He.

**Problem 3**: Velocity dispersion analysis of an SEP event shows that 100 MeV protons ($v = 0.43c$) arrive at $t_1 = 08{:}20$ UT and 10 MeV protons ($v = 0.14c$) arrive at $t_2 = 09{:}05$ UT. Assuming scatter-free propagation along a path of length $L$, calculate $L$ (in AU) and the solar release time $t_0$. The Parker spiral field line length for $V_{sw} = 400$ km/s is approximately 1.15 AU — comment on whether your derived path length is consistent.

**Problem 4**: An astronaut on a Mars transit mission is exposed to a large SEP event. The event delivers a proton fluence of $10^{10}$ protons/cm$^2$ above 10 MeV over 24 hours. Assuming an average LET of 2 keV/$\mu$m in tissue ($\rho = 1$ g/cm$^3$) and a quality factor $Q = 2$, estimate the absorbed dose (in Gy) and dose equivalent (in Sv) for a path length of 10 cm through tissue. Is this within NASA's 30-day limit? What shielding strategy could reduce this dose?

**Problem 5**: A Type II radio burst is observed simultaneously with a gradual SEP event. The burst starts at 200 MHz and drifts to 50 MHz over 8 minutes. Using $f_p \approx 9\sqrt{n_e}$ kHz (with $n_e$ in cm$^{-3}$), calculate the electron densities at the start and end of the burst. If the coronal density decreases as $n_e(r) \propto r^{-2}$, estimate the ratio of heliocentric distances and the average shock speed. Discuss why this shock is expected to efficiently accelerate particles.

---

**Previous**: [Solar Spectroscopy and Instruments](./13_Solar_Spectroscopy_and_Instruments.md) | **Next**: [Modern Solar Missions](./15_Modern_Solar_Missions.md)

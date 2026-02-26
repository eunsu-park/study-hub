# Magnetospheric Substorms

## Learning Objectives

- Describe the three phases of a magnetospheric substorm (growth, expansion, recovery) and their characteristic timescales
- Understand the competing onset mechanism models (near-Earth neutral line vs. current disruption) and the observational evidence for each
- Explain the physics of the substorm current wedge, including field-aligned currents and ionospheric closure
- Describe auroral emission mechanisms, including discrete and diffuse aurora, and relate them to magnetospheric processes
- Analyze the AE/AL/AU indices and their physical interpretation
- Distinguish substorms from geomagnetic storms and describe their complex relationship
- Identify dipolarization fronts and dispersionless particle injections as key expansion phase signatures

## 1. Substorm Phenomenology

A magnetospheric substorm is a transient disturbance of the magnetosphere-ionosphere system, lasting typically 1--3 hours, during which $10^{14}$--$10^{15}$ J of energy stored in the magnetotail is explosively released. Substorms are among the most fundamental modes of magnetospheric dynamics, occurring several times per day during geomagnetically active periods and even occasionally during relatively quiet times.

### 1.1 Historical Context

The concept of the auroral substorm was introduced by Syun-Ichi Akasofu in 1964, based on systematic analysis of all-sky camera images of the aurora. Akasofu identified a repeatable sequence of auroral morphological changes --- a quiet arc suddenly brightening and expanding poleward, followed by a gradual return to quiet conditions --- that he termed the "auroral substorm." The term "magnetospheric substorm" was later adopted to emphasize that the auroral displays are merely the visible manifestation of a global reconfiguration of the magnetosphere.

### 1.2 Substorms Are Not Small Storms

A common misconception is that substorms are simply smaller versions of geomagnetic storms. This is incorrect in several important ways:

- **Different timescales**: Substorms last 1--3 hours; storms last 1--7 days.
- **Different primary current system**: Substorms are associated with the substorm current wedge (field-aligned currents and ionospheric electrojet); storms are associated with the symmetric ring current.
- **Different energy storage**: Substorms release magnetotail lobe magnetic energy; storms build up ring current kinetic energy.
- **Different driving conditions**: Substorms can occur during modest southward $B_z$ intervals; storms require sustained, strong southward $B_z$.

That said, substorms and storms are intimately connected --- substorm injections contribute particles to the ring current, and storms typically contain multiple substorms. The nature of this relationship remains an active area of research.

### 1.3 Energy Budget

The energy released during a typical substorm is approximately $10^{14}$--$10^{15}$ J, distributed among several sinks:

| Energy Sink | Fraction | Amount |
|:---|:---:|:---:|
| Ionospheric Joule heating | $\sim$50% | $\sim 5 \times 10^{14}$ J |
| Particle precipitation | $\sim$20% | $\sim 2 \times 10^{14}$ J |
| Ring current injection | $\sim$15% | $\sim 1.5 \times 10^{14}$ J |
| Plasmoid ejection | $\sim$15% | $\sim 1.5 \times 10^{14}$ J |

The source of this energy is the magnetic energy stored in the magnetotail lobes during the growth phase, which is ultimately derived from the solar wind via dayside reconnection.

## 2. Growth Phase

The growth phase is the preparatory period during which energy from the solar wind is stored in the magnetotail, setting the stage for the explosive energy release of the expansion phase.

### 2.1 Energy Loading

The growth phase begins when the IMF turns southward, enabling magnetic reconnection at the dayside magnetopause. The reconnected magnetic flux is transported antisunward over the polar caps by the solar wind flow, adding open flux to the magnetotail lobes. Since the tail lobes contain only open magnetic field lines (connected to the solar wind at one end and Earth at the other), this flux addition increases the magnetic pressure in the lobes:

$$P_{lobe} = \frac{B_{lobe}^2}{2\mu_0}$$

The lobe magnetic field $B_{lobe}$ typically increases from $\sim$20 nT to $\sim$30--40 nT during the growth phase, representing a factor of 2--4 increase in magnetic pressure.

### 2.2 Tail Stretching and Current Sheet Thinning

The increased lobe pressure compresses the plasma sheet, causing two interrelated changes:

**Tail stretching**: The near-Earth magnetic field becomes increasingly tail-like, with the $B_x$ component increasing and the $B_z$ component decreasing. Field lines that were quasi-dipolar at $L \sim 8$--12 become highly stretched, with the neutral sheet (where $B_x$ changes sign) becoming more distant from the equatorial plane.

**Current sheet thinning**: The cross-tail current sheet, which separates the two tail lobes, thins dramatically during the growth phase. The half-thickness decreases from a quiet-time value of $\sim$5 $R_E$ to less than 1 $R_E$, and in some cases to ion-scale thicknesses of $\sim$1000 km or less. This thinning is critical because it brings the current sheet toward the threshold for instability.

The total energy stored in the magnetotail during the growth phase is approximately:

$$E_{tail} \approx \frac{B_{lobe}^2}{2\mu_0} \times V_{lobe} \approx 10^{15} \text{ J}$$

where $V_{lobe}$ is the volume of one tail lobe ($\sim 10^{24}$ m$^3$).

### 2.3 Auroral Signatures

During the growth phase, the auroral oval expands equatorward as open flux is added to the polar cap. The most equatorward discrete auroral arc moves steadily equatorward, reaching its lowest latitude just before substorm onset. This equatorward motion is observed as a gradual decrease in the latitude of the most equatorward arc, from typically $\sim$68$^\circ$ to $\sim$64$^\circ$ geomagnetic latitude.

### 2.4 Duration and Termination

The growth phase typically lasts 30--60 minutes, though it can extend to 1--2 hours in some cases. It terminates with substorm onset, which can be triggered by either (a) the internal state of the magnetotail reaching a critical threshold (loading instability), or (b) external triggers such as sudden changes in the solar wind (northward turning of $B_z$, pressure pulses). The relative importance of internal vs. external triggering remains debated.

## 3. Onset and Expansion Phase

The expansion phase is the explosive energy release episode that defines the substorm. It begins with a sudden onset and lasts 20--60 minutes, during which the magnetotail undergoes a dramatic reconfiguration.

### 3.1 Onset Signatures

Substorm onset is identified by a constellation of nearly simultaneous signatures:

- **Auroral**: Sudden brightening of the most equatorward arc near magnetic midnight, followed within minutes by poleward and westward expansion of the bright aurora (the "auroral bulge").

- **Magnetic (ground)**: Sharp negative excursion in the $H$-component at auroral-zone stations near midnight (the "negative bay" in AL index).

- **Magnetic (space)**: Dipolarization of the magnetic field at geosynchronous orbit --- $B_z$ increases suddenly as the field returns from a stretched to a more dipolar configuration.

- **Particle (space)**: Dispersionless injection of energetic particles at geosynchronous orbit near midnight.

- **Pi2 pulsations**: Damped oscillations with periods of 40--150 seconds, observed globally, thought to be cavity modes excited by the rapid reconfiguration.

The precise timing and causal ordering of these signatures are central to the onset mechanism debate.

### 3.2 The Near-Earth Neutral Line (NENL) Model

The NENL model, championed by Baker, McPherron, and others, proposes that the expansion phase is initiated by the onset of magnetic reconnection in the mid-tail at a geocentric distance of $\sim$20--30 $R_E$.

In this picture:

1. The thinned current sheet becomes unstable to the tearing instability, forming an X-line.
2. Reconnection produces an earthward flow jet (bursty bulk flow) and a tailward-ejected plasmoid.
3. The earthward flow brakes in the near-Earth region ($\sim$10 $R_E$), piling up magnetic flux and causing dipolarization.
4. Field-aligned currents develop at the flow-braking region, diverting cross-tail current into the ionosphere and establishing the substorm current wedge.

**Supporting evidence**: Tailward-moving plasmoids observed by spacecraft in the mid-tail; earthward bursty bulk flows (BBFs) with $v_x > 400$ km/s; reconnection signatures (Petschek-type outflows) at $\sim$20--30 $R_E$.

### 3.3 The Current Disruption (CD) Model

The CD model, advocated by Lui, Roux, and others, proposes that the onset begins with a current-driven instability in the near-Earth current sheet ($\sim$8--12 $R_E$), and that mid-tail reconnection is a consequence rather than a cause.

In this picture:

1. The thin current sheet at $\sim$8--12 $R_E$ becomes unstable to cross-field current instabilities (e.g., the ballooning/interchange instability or the cross-field current instability).
2. The instability disrupts the cross-tail current, diverting it along field lines into the ionosphere.
3. The resulting rarefaction propagates tailward, triggering reconnection at $\sim$20--30 $R_E$.

**Supporting evidence**: Onset signatures (auroral brightening, Pi2 pulsations, dipolarization) appear to originate in the near-Earth region and propagate tailward; current disruption has been observed at $\sim$8--10 $R_E$ before reconnection signatures appear at $\sim$20 $R_E$ in some events.

### 3.4 Current Understanding

The debate between NENL and CD models has evolved over decades. Current understanding suggests that **both processes occur**, and the question is not which one is "correct" but rather which one initiates the sequence in a given event and how they couple. Multi-point observations from missions like THEMIS (Time History of Events and Macroscale Interactions during Substorms), launched specifically to address this question, have shown that the timing is event-dependent, with some substorms showing clear tailward-to-earthward propagation (supporting CD-first) and others showing earthward flows preceding near-Earth disruption (supporting NENL-first).

The emerging synthesis view recognizes that the magnetotail is a coupled system in which instabilities at different locations can communicate via Alfven waves (propagation time $\sim$1--2 minutes between 10 and 20 $R_E$), making strict causal ordering difficult to establish with current measurement capabilities.

## 4. Dipolarization and Injection

Two of the most dramatic and consequential signatures of the substorm expansion phase are the dipolarization of the near-Earth magnetic field and the injection of energetic particles at geosynchronous orbit.

### 4.1 Dipolarization Fronts

A **dipolarization front (DF)** is a sharp, earthward-propagating boundary across which the magnetic field $B_z$ component increases abruptly (typically by 5--20 nT over a spatial scale of $\sim$1 $R_E$ or less). DFs are embedded in earthward-flowing plasma (bursty bulk flows) and represent the leading edge of the reconnection outflow as it compresses the pre-existing plasma ahead of it.

Key properties of dipolarization fronts:

- **Speed**: 200--400 km/s earthward
- **Thickness**: $\sim$800--2000 km (a few ion inertial lengths)
- **Magnetic signature**: Sharp increase in $B_z$, decrease in $B_x$
- **Electric field**: Enhanced dawn-to-dusk electric field at the front
- **Particle energization**: The electric field at the DF accelerates both ions and electrons, particularly through betatron (magnetic moment conservation) and Fermi (bounce) acceleration

DFs propagate from the reconnection region at $\sim$20--30 $R_E$ to the inner magnetosphere at $\sim$8--10 $R_E$, where they brake and accumulate, collectively restoring the near-Earth magnetic field to a more dipolar configuration.

### 4.2 Particle Injections

**Dispersionless injections** are sudden increases in the flux of energetic particles (10--300 keV) observed by geosynchronous spacecraft near midnight. They are called "dispersionless" because particles of all energies arrive simultaneously, implying that the injection boundary passed over the spacecraft rather than the particles drifting in from a distant source.

After the initial dispersionless injection:
- Electrons drift eastward due to gradient-curvature drift (energy-dependent speed)
- Ions drift westward

This produces **dispersed** injection signatures at other local times, where higher-energy particles arrive first (faster drift) and lower-energy particles arrive later. The dispersion pattern can be used to determine the injection time and local time of origin.

The injected particles contribute to the ring current and the radiation belts, providing the link between substorms and longer-term magnetospheric dynamics.

## 5. Substorm Current Wedge (SCW)

The substorm current wedge is one of the most important current systems in magnetospheric physics, representing the diversion of the cross-tail current through the ionosphere during the expansion phase.

### 5.1 Structure

During the growth phase, the cross-tail current flows from dawn to dusk across the magnetotail, closing through the magnetopause boundary. At substorm onset, a portion of this current is disrupted in the near-Earth tail and diverted along magnetic field lines into the ionosphere:

- **Downward (into ionosphere) field-aligned current**: On the duskside of the disruption region
- **Westward electrojet**: Intense ionospheric current flowing westward across the auroral zone (the same current that produces the negative bay in the AL index)
- **Upward (out of ionosphere) field-aligned current**: On the dawnside of the disruption region

The wedge typically spans 3--6 hours of local time centered near midnight, though it expands in both azimuthal extent and intensity during the expansion phase.

### 5.2 Magnetic Perturbation Pattern

The SCW produces a characteristic pattern of ground magnetic perturbations:

- **Below the electrojet** (auroral zone): Strong negative $\Delta H$ (the "negative bay"), positive $\Delta D$ east of center, negative $\Delta D$ west of center.
- **Equatorward of the electrojet** (mid-latitudes within the wedge): Positive $\Delta H$ from the field-aligned current system, which partially counteracts the overall ring current depression.
- **Low latitudes**: Weak positive $\Delta H$ from the reduced tail current (missing current replaced by SCW reduces the tail's contribution to ground perturbation).

### 5.3 Current Intensity

The total field-aligned current in the SCW is typically 1--3 MA (mega-amperes) for a moderate substorm, and can reach 5 MA or more for intense events. The current density in the ionospheric electrojet reaches $\sim$1 A/m, and the associated Joule heating rate can exceed 100 GW, representing the single largest energy dissipation mechanism during substorms.

### 5.4 Modern Refinements

The classical SCW picture of a simple wire-model current loop has been refined by observations from multi-spacecraft missions. The real SCW has a more complex structure:

- Multiple filamentary field-aligned currents rather than two monolithic sheets
- The upward FAC region is coincident with the auroral surge, which moves poleward and westward during the expansion
- Region 1 and Region 2 current systems interact with the SCW, creating a more complex closure pattern
- The current wedge is not static but evolves dynamically, expanding in local time and latitude throughout the expansion phase

## 6. Auroral Physics

The aurora is the most visually spectacular manifestation of substorm activity and provides critical remote-sensing information about magnetospheric processes. The different types and colors of aurora each tell a distinct physical story.

### 6.1 Discrete Aurora

Discrete auroral arcs are produced by electrons that have been accelerated along magnetic field lines by parallel electric fields. The acceleration occurs in the so-called **auroral acceleration region** at altitudes of $\sim$2000--10,000 km, where parallel potential drops of 1--10 kV develop.

The energy spectrum of the precipitating electrons displays a characteristic **inverted-V structure** when measured by satellites passing through the acceleration region: the electron energy peaks at the center of the arc (corresponding to the peak of the potential drop) and decreases toward the edges, producing a V shape when plotted as energy vs. latitude.

The mechanism for parallel electric field formation involves the interaction between field-aligned currents and the ionospheric plasma. When the magnetosphere demands upward current that exceeds what the ionospheric plasma can supply by thermal electrons alone (the Knight relation), a parallel potential drop develops to accelerate electrons upward (which is equivalent to accelerating magnetospheric electrons downward to carry the upward current).

### 6.2 Auroral Colors

The colors of the aurora directly reveal the atmospheric species being excited and the altitude of energy deposition:

**Green (557.7 nm)**: The most common auroral color. Produced by the $^1S \rightarrow ^1D$ forbidden transition of atomic oxygen (O I). Emission altitude: $\sim$100--200 km. The $^1S$ state has a radiative lifetime of 0.7 seconds, short enough that the excited atom can radiate before being collisionally deactivated.

**Red (630.0 nm)**: Produced by the $^1D \rightarrow ^3P$ forbidden transition of atomic oxygen. Emission altitude: $> 200$ km. The $^1D$ state has a much longer radiative lifetime of 110 seconds, so it can only radiate at high altitudes where the collision frequency is low enough that the atom is not deactivated before emission. Red aurora is associated with lower-energy precipitating electrons ($<1$ keV) that deposit their energy at higher altitudes.

**Blue/Violet (391.4 nm, 427.8 nm)**: Produced by the first negative band system of molecular nitrogen ions (N$_2^+$). These emissions require relatively energetic electrons ($>$10 keV) to penetrate to the low altitudes ($<$100 km) where N$_2$ is abundant. Blue/violet aurora often appears at the lower border of bright arcs.

**Proton aurora**: Produced by charge-exchange of precipitating protons with atmospheric neutrals. The resulting fast hydrogen atoms emit Lyman-alpha (121.6 nm, UV) and Balmer-alpha (656.3 nm) and Balmer-beta (486.1 nm) lines. These emissions are Doppler-broadened due to the high speed of the emitting atoms, which distinguishes them from geocoronal hydrogen emission.

### 6.3 Diffuse Aurora

Unlike discrete arcs, the diffuse aurora is produced by electrons scattered into the loss cone by wave-particle interactions in the equatorial magnetosphere, rather than by parallel electric fields. The dominant scattering mechanism is resonant interaction with **chorus waves** (whistler-mode waves at 0.1--0.8 of the electron cyclotron frequency).

The diffuse aurora is more spatially uniform than discrete aurora, typically forming a broad band equatorward of the discrete arcs. It accounts for roughly half of the total auroral electron energy input, despite being less visually striking. The diffuse aurora maps to the central plasma sheet and is always present to some degree, even during quiet conditions.

### 6.4 Pulsating Aurora

Pulsating aurora consists of irregular patches that brighten and dim quasi-periodically with periods of a few seconds to tens of seconds. It is most common during the recovery phase of substorms and is caused by modulation of electron precipitation by lower-band chorus waves. The chorus waves are generated at the magnetic equator and modulate the pitch-angle scattering rate of plasma sheet electrons, causing the precipitated flux to pulsate.

## 7. AE/AL/AU Indices

The auroral electrojet indices provide a quantitative measure of auroral-zone magnetic activity and are the primary indices used to identify and characterize substorms.

### 7.1 Definition and Computation

The AE indices are derived from the $H$-component recordings of approximately 12 magnetometers distributed along the auroral zone (geomagnetic latitude $\sim$65--70$^\circ$). At each time step (1-minute resolution):

1. The quiet-day baseline is subtracted from each station's $H$-component.
2. The **AU (upper) envelope** is defined as the maximum positive perturbation among all stations.
3. The **AL (lower) envelope** is defined as the maximum negative perturbation among all stations.
4. **AE** $=$ AU $-$ AL (always positive, measures total electrojet activity).
5. **AO** $= ($AU $+$ AL$)/2$ (midpoint, not commonly used).

### 7.2 Physical Interpretation

**AU** primarily measures the strength of the **eastward electrojet**, which flows in the afternoon sector and is associated with large-scale magnetospheric convection (DP2 current system). During geomagnetically active periods, AU typically ranges from 50 to 500 nT.

**AL** primarily measures the strength of the **westward electrojet**, which flows in the midnight-morning sector and is strongly enhanced during substorms by the substorm current wedge (DP1 current system). During substorm expansion phases, AL can reach $-1000$ to $-2000$ nT.

**AE** represents the total electrojet activity and combines both convection-driven (DP2) and substorm-driven (DP1) components. It is widely used as a proxy for overall magnetospheric energy dissipation, as the Joule heating in the ionosphere scales approximately as $\Sigma_P E^2 \propto \text{AE}^2$.

### 7.3 Substorm Identification in AE/AL

A substorm expansion phase onset is identified in the AL index as a **sharp negative bay** --- a sudden decrease in AL of at least $\sim$100 nT within a few minutes. The minimum AL during the expansion typically reaches $-500$ to $-1500$ nT. The onset time corresponds to the beginning of the negative excursion, which coincides (within minutes) with the auroral brightening seen by imagers.

Typical quiet-time values are AE $< 100$ nT. Values of AE $= 500$--$2000$ nT indicate substorm activity. Sustained AE $> 1000$ nT usually indicates storm-time substorm activity.

### 7.4 Limitations

The AE indices have several known limitations:

- **Station coverage**: 12 stations cannot fully resolve the spatial structure of the electrojet, especially during intense activity when the auroral oval moves equatorward of the station network.
- **Seasonal effects**: Ionospheric conductivity varies with season and solar illumination, affecting the electrojet intensity for a given level of magnetospheric driving.
- **Contamination**: AL can be affected by field-aligned current perturbations that do not represent electrojet enhancements.
- **Convection vs. substorm separation**: AE conflates the two current systems. The SME (SuperMAG) index uses $>$100 stations to mitigate coverage issues.

## 8. Storm-Substorm Relationship

The relationship between geomagnetic storms and substorms has been debated for decades and remains one of the fundamental open questions in magnetospheric physics.

### 8.1 The Classical View

The classical view, articulated in the 1970s--1980s, held that substorms are the building blocks of storms: each substorm injects energetic particles into the inner magnetosphere, gradually building up the ring current. In this picture, a storm is simply the accumulated effect of many substorms during a prolonged period of southward IMF.

This view has intuitive appeal because: (1) storms typically contain multiple substorms, (2) substorm injections clearly deliver particles to ring current distances, and (3) the ring current buildup during the main phase is temporally correlated with substorm activity.

### 8.2 Challenges to the Classical View

Several observations have challenged the simple "substorms cause storms" picture:

**Steady Magnetospheric Convection (SMC)**: During some intervals of sustained southward IMF, the magnetosphere reaches a quasi-steady state with continuous convection and particle injection without distinct substorm cycles. These SMC intervals can produce significant ring current enhancement (storm-like Dst decrease) without identifiable substorms in the AL index.

**Storms without substorms**: Some moderate geomagnetic storms occur with relatively few distinct substorms, suggesting that enhanced convection alone (without discrete substorm injections) can build the ring current.

**Substorms without storms**: Isolated substorms occur frequently during modest southward $B_z$ intervals and do not produce significant ring current enhancement, demonstrating that substorms are neither sufficient nor necessary for storms.

### 8.3 Sawtooth Events

**Sawtooth events** (or sawtooth injections) represent a striking mode of storm-time substorm activity. They consist of quasi-periodic ($\sim$2--4 hour period) global injections of energetic particles, observed simultaneously across the entire nightside at geosynchronous orbit, rather than the localized midnight injections of isolated substorms.

The "sawtooth" name comes from the appearance of the geosynchronous magnetic field: a gradual stretching (growth phase) followed by a sudden dipolarization (expansion), repeating with remarkable regularity. Sawtooth events occur predominantly during moderate to intense storms (Dst $< -50$ nT) and represent a strongly driven mode where the magnetotail undergoes forced periodic loading-unloading cycles.

### 8.4 Current Synthesis

The current understanding recognizes that storms and substorms are both responses to solar wind energy input, partially coupled but not in a simple cause-and-effect relationship:

- **Enhanced convection** (driven by sustained southward $B_z$) is the primary mechanism for ring current buildup during storms. This operates continuously, not in discrete substorm-related bursts.

- **Substorm injections** augment the convection-driven ring current by providing discrete pulses of energetic particles. They are particularly important for delivering higher-energy particles ($>50$ keV) to the inner magnetosphere.

- **The relative importance** of convection vs. substorm injection depends on the driving conditions. For very strong, sustained southward $B_z$ (CME magnetic clouds), enhanced convection dominates. For fluctuating $B_z$ (CIR/high-speed streams), substorm injections may be relatively more important.

- **Energy pathways differ**: Substorms primarily dissipate energy in the ionosphere (Joule heating), while storms primarily store energy in the ring current. The same solar wind energy input is partitioned differently depending on the dynamical mode.

## Practice Problems

**Problem 1: Growth Phase Energy**

During a substorm growth phase, the tail lobe magnetic field increases from 20 nT to 35 nT over 45 minutes. Assuming a lobe volume of $5 \times 10^{23}$ m$^3$ (for one lobe), calculate: (a) the magnetic energy stored in both lobes before and after the growth phase, (b) the net energy added to the tail, and (c) the average power input from the solar wind to the tail during the growth phase.

**Problem 2: Auroral Emission Altitude**

A precipitating electron with kinetic energy 5 keV enters the atmosphere. Using the empirical relation for the altitude of peak energy deposition, $h_{peak} \approx 130 - 10 \ln(E/\text{keV})$ km (where $E$ is in keV), calculate the peak emission altitude. At this altitude, will the dominant emission be green (557.7 nm, O I $^1S$) or red (630.0 nm, O I $^1D$)? Explain your reasoning in terms of collisional deactivation rates.

**Problem 3: Electrojet Current**

During a substorm, the AL index reaches $-1200$ nT. If the westward electrojet can be modeled as an infinite line current at an altitude of 110 km, estimate the total current in the electrojet. (Use the relation $\Delta H \approx \mu_0 I / (2\pi h)$ where $h$ is the altitude.) Compare this with the typical cross-tail current disrupted during a substorm.

**Problem 4: Injection Drift Dispersion**

A dispersionless injection of energetic particles occurs at midnight local time. Calculate the drift period for: (a) a 50 keV electron, and (b) a 50 keV proton, at $L = 6.6$ (geosynchronous orbit). Use the gradient-curvature drift period $T_d \approx \frac{2\pi q B_0 R_E^2}{3 L E}$ where $B_0 = 3.1 \times 10^{-5}$ T. If a geosynchronous satellite at the dawn meridian (6 hours from midnight) detects the injection, at what time delay after onset does it detect the electrons vs. the protons?

**Problem 5: Substorm vs. Storm Energy**

A moderate geomagnetic storm lasts 3 days with an average Dst of $-80$ nT. During this storm, 15 distinct substorms are identified, each releasing approximately $5 \times 10^{14}$ J. (a) Using the DPS relation, estimate the total ring current energy during the storm. (b) Calculate the total energy released by all substorms. (c) Compare the two values and discuss whether substorm energy release can account for the ring current energy, keeping in mind that substorms dissipate most of their energy in the ionosphere.

---

**Previous**: [Geomagnetic Storms](./05_Geomagnetic_Storms.md) | **Next**: [Radiation Belts](./07_Radiation_Belts.md)

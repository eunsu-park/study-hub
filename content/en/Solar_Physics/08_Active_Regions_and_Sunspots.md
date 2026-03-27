# Active Regions and Sunspots

## Learning Objectives

- Describe the fine structure of sunspots including umbra, penumbra, umbral dots, and Wilson depression
- Explain the Evershed effect and moat flow, including their physical mechanisms
- Understand the process of magnetic flux emergence from the convection zone to the corona
- Trace the lifecycle of an active region from emergence through decay
- Classify active regions using the McIntosh and Hale/Mt. Wilson systems
- Describe quiet Sun magnetism including ephemeral regions, the magnetic carpet, and internetwork fields
- Relate active region magnetic complexity to flare productivity

---

## 1. Sunspot Fine Structure

### 1.1 Overview: What Is a Sunspot?

Sunspots are the most prominent manifestations of solar magnetic activity visible on the photospheric surface. They appear as dark patches on the solar disk because the strong magnetic field within a sunspot ($\sim 2000$-$3500$ G) suppresses convective energy transport, reducing the local surface temperature by 1000-2000 K below the surrounding photosphere.

The energy flux radiated by a surface scales as $T^4$ (Stefan-Boltzmann law), so a sunspot umbra at 4000 K radiates only $(4000/5778)^4 \approx 23\%$ as much energy per unit area as the undisturbed photosphere. This is why sunspots appear dark -- they are still incredibly luminous in absolute terms (roughly as bright as the full Moon), but dim relative to their even brighter surroundings.

### 1.2 The Umbra

The **umbra** is the dark central region of a sunspot. Its key properties:

- **Temperature**: $T \approx 3500$-$4000$ K (lowest at the darkest core)
- **Magnetic field strength**: $B \approx 2000$-$3500$ G, with the strongest fields at the center
- **Field inclination**: nearly vertical (within $\sim 10°$-$20°$ of the local normal)
- **Size**: typically 5-15 Mm in diameter for a well-developed sunspot

The umbral magnetic field inhibits the vigorous overturning convection that transports most of the energy in the quiet photosphere. However, convection is not completely suppressed -- it is modified into a weaker, magnetically constrained form called **magneto-convection**.

**Umbral dots** are the primary evidence for residual convection in the umbra. These are small bright features ($\sim 200$-$300$ km diameter) scattered throughout the umbra, with temperatures a few hundred kelvin above the umbral background. They are the tops of thin, hot plumes where plasma rises along the magnetic field, overshoots, cools, and sinks back down. High-resolution observations from the Swedish Solar Telescope and DKIST reveal that umbral dots have:

- Upflow velocities of $\sim 1$ km/s at their centers
- Slight magnetic field reductions compared to the surrounding umbra
- Lifetimes of 10-30 minutes
- A filling factor of $\sim 10$-$30\%$ of the umbral area

The physical picture is that convection operates within the magnetic field by creating thin, elongated cells aligned with the vertical field. The strong field does not prevent convective instability entirely -- rather, it modifies the convective pattern and reduces its efficiency.

### 1.3 The Penumbra

The **penumbra** is the lighter, filamentary region surrounding the umbra, typically 5-10 Mm wide. It is one of the most complex structures in the solar atmosphere, with fine structure at scales below 100 km that challenges both observational resolution and theoretical understanding.

The penumbra has a distinctive **filamentary appearance**: alternating bright and dark radial filaments with widths of 150-300 km. These filaments reflect a fundamentally inhomogeneous magnetic and flow structure:

**Bright filaments (spines of the penumbra):**
- More vertical magnetic field ($\sim 40$-$50°$ from vertical)
- Hot plasma flowing upward (convective upflows)
- Higher temperature ($\sim 5500$-$5700$ K)
- Carry most of the penumbral energy flux

**Dark filaments (intra-spines):**
- More horizontal magnetic field ($\sim 70$-$90°$ from vertical)
- Radial outflows (the Evershed flow, see Section 2)
- Cooler temperature ($\sim 5000$-$5300$ K)
- Associated with overturning convection in which hot gas rises at the inner footpoint, flows radially outward, cools, and sinks at the outer footpoint

This two-component model is sometimes called the **uncombed penumbra** model (Solanki and Montavon 1993) or the **gappy penumbra** model, reflecting the interlocking of flux tubes with different inclinations.

### 1.4 Wilson Depression

The **Wilson depression** is the geometric depression of the $\tau = 1$ (unit optical depth) surface within a sunspot relative to the surrounding photosphere. Because the sunspot is cooler and the magnetic field provides additional pressure support (reducing the gas pressure and density), the surface of equal optical depth is pushed downward.

The depth of the Wilson depression can be estimated from pressure balance at the umbral boundary. At the edge of the sunspot, the total pressure must balance:

$$p_{\text{ext}}(z) = p_{\text{int}}(z) + \frac{B^2}{8\pi}$$

where $p_{\text{ext}}$ is the external gas pressure, $p_{\text{int}}$ is the internal gas pressure, and $B^2/(8\pi)$ is the magnetic pressure. The reduction in internal gas pressure means the density is lower inside the spot, so one must look deeper to reach the same optical depth.

Observational determinations of the Wilson depression yield values of approximately:

- **Umbra**: 400-700 km below the quiet photosphere
- **Penumbra**: 100-200 km depression

The Wilson depression was historically inferred from the geometric foreshortening of sunspots near the solar limb (the Wilson effect, observed by Alexander Wilson in 1769). Modern spectropolarimetric inversions provide more direct measurements.

---

## 2. Evershed and Moat Flows

### 2.1 The Evershed Effect

In 1909, John Evershed discovered a systematic Doppler shift pattern in sunspot spectra: spectral lines in the penumbra are blueshifted on the side of the spot facing disk center and redshifted on the opposite side. This pattern indicates a **radial outflow** in the penumbra, directed from the inner penumbral boundary (near the umbra) toward the outer boundary.

The **Evershed flow** has the following properties:

- **Velocity**: 1-6 km/s, with the fastest flows in the dark penumbral filaments
- **Height**: concentrated in the deep photosphere (below $\tau = 1$)
- **Magnetic alignment**: the flow is nearly aligned with the more horizontal magnetic field component
- **Radial direction**: always directed outward from the umbra, regardless of the spot's position on the disk
- **Temporal behavior**: remarkably steady, persisting as long as the penumbra exists

### 2.2 Physical Mechanism: Siphon Flow

The most widely accepted explanation for the Evershed flow is the **siphon flow model**. Consider a magnetic flux tube with its inner footpoint in the penumbra (high magnetic field, low gas pressure) and its outer footpoint outside the spot (lower magnetic field, higher gas pressure):

$$p_{\text{inner}} + \frac{B_{\text{inner}}^2}{8\pi} = p_{\text{outer}} + \frac{B_{\text{outer}}^2}{8\pi}$$

Since $B_{\text{inner}} > B_{\text{outer}}$, we have $p_{\text{inner}} < p_{\text{outer}}$, creating a pressure gradient that drives a flow from the outer footpoint to the inner footpoint along the tube. Wait -- this would be an inflow, opposite to what is observed!

The resolution is that the Evershed flow occurs in flux tubes that are embedded in a stratified atmosphere. The gas pressure difference at the footpoints depends on the height of the footpoints. If the inner footpoint is deeper than the outer footpoint (due to the Wilson depression), the hydrostatic gas pressure at the inner footpoint can exceed that at the outer footpoint even though the magnetic pressure is also higher there. The resulting pressure gradient drives the flow outward.

An alternative explanation is that the Evershed flow is a manifestation of overturning magneto-convection in the penumbra: hot gas rises at the inner (bright) end of a filament, flows radially outward as it cools, and sinks at the outer end.

### 2.3 Inverse Evershed Flow

In the **chromosphere**, the flow pattern reverses: there is an **inflow** directed from the outer penumbral boundary toward the umbra. This **inverse Evershed flow** has velocities of $\sim 1$-$3$ km/s and is observed in chromospheric lines such as H-alpha and Ca II 8542 A.

The inverse Evershed flow is interpreted as a siphon flow in higher-lying flux tubes where the pressure gradient (now determined by the different atmospheric conditions at chromospheric heights) drives plasma inward. The chromospheric magnetic field configuration differs from the photospheric one, with field lines that connect the outer penumbra to the superpenumbral chromosphere curving back toward the umbra.

### 2.4 The Moat Region and Moving Magnetic Features

Beyond the outer edge of the penumbra lies the **moat region**, an annular zone extending $\sim 10$-$20$ Mm from the sunspot boundary. The moat is characterized by:

- A radial outflow of $\sim 0.5$-$1.0$ km/s (much slower than the Evershed flow)
- The outward migration of small magnetic elements called **Moving Magnetic Features (MMFs)**
- Suppression of normal granulation in favor of slightly larger, more elongated convective cells

**Moving Magnetic Features (MMFs)** are small ($\sim 1$-$3$ arcsec), discrete magnetic elements that stream radially outward from the sunspot through the moat at speeds of $\sim 0.3$-$1.0$ km/s. They come in three distinct types:

**Type I (bipolar pairs)**: pairs of opposite-polarity elements moving outward together, with the leading element having the same polarity as the sunspot. These are interpreted as small $\Omega$-loops of flux detaching from the sunspot's peripheral field.

**Type II (unipolar, same polarity as spot)**: single elements with the same polarity as the sunspot, moving outward. These may represent fragments of the sunspot's magnetic flux being stripped away by the moat flow.

**Type III (unipolar, opposite polarity to spot)**: single elements with polarity opposite to the sunspot. These may be the return-flux ends of the sunspot's canopy field that have been swept outward.

MMFs play a crucial role in sunspot decay: they progressively remove magnetic flux from the spot, contributing to its gradual dissolution. The total flux carried away by MMFs is consistent with the observed decay rates of sunspots.

---

## 3. Active Region Emergence

### 3.1 Magnetic Buoyancy and Flux Tube Rise

Active regions form when **magnetic flux tubes** stored in the tachocline (the shear layer at the base of the convection zone, at $\sim 0.7 \, R_\odot$) become buoyantly unstable and rise through the convection zone to emerge at the photosphere.

The condition for magnetic buoyancy instability in a stratified atmosphere is that the magnetic pressure reduces the internal gas pressure so that the flux tube density is less than the external density. For a flux tube in temperature equilibrium with its surroundings, this requires:

$$B > B_{\text{eq}} = \sqrt{8\pi p_{\text{ext}}}$$

At the base of the convection zone, $B_{\text{eq}} \sim 10^4$-$10^5$ G. Thin flux tube simulations show that tubes with $B \sim 10^5$ G rise through the convection zone in a few months, forming an **$\Omega$-shaped loop** (because the central portion is more buoyant and rises faster than the ends).

### 3.2 The Coriolis Force and Joy's Law

As the flux tube rises through the rotating convection zone, it is deflected by the **Coriolis force**. The rising loop expands laterally, and the Coriolis force acts on this expansion to tilt the loop: the leading leg (in the direction of rotation) is pushed equatorward, while the following leg is pushed poleward.

This tilt is the origin of **Joy's law**: the empirical observation that bipolar active regions are tilted with respect to the east-west direction, with the leading polarity closer to the equator. The tilt angle increases with latitude, approximately as:

$$\gamma \approx \gamma_0 \sin\lambda$$

where $\lambda$ is the latitude and $\gamma_0 \approx 30°$-$35°$ is a constant. Joy's law tilt is a key ingredient in the Babcock-Leighton model of the solar dynamo, as it provides the mechanism for converting toroidal magnetic flux into poloidal flux.

### 3.3 Observational Sequence of Emergence

The emergence of a new active region unfolds as a characteristic sequence of observable events:

**Stage 1 -- Pre-emergence (hours before)**: subtle disturbances in the granulation pattern -- elongated, darkened lanes appear where the subsurface flux tube is pushing upward. Local helioseismology may detect subsurface flow anomalies.

**Stage 2 -- Initial emergence**: a small bipolar magnetic region appears in magnetograms, with the two polarities initially very close together ($\sim 5$-$10$ Mm separation). The magnetic flux increases rapidly, typically at a rate of $10^{19}$-$10^{20}$ Mx/hour.

**Stage 3 -- Separation and growth**: the opposite-polarity footpoints separate at velocities of $\sim 1$-$2$ km/s, driven by the emergence of the $\Omega$-loop. New flux continues to emerge, and the total unsigned flux grows from $\sim 10^{20}$ Mx to $\sim 10^{22}$ Mx over days.

**Stage 4 -- Arch filament system**: in the chromosphere, an **arch filament system (AFS)** appears -- a set of dark, arched absorption features in H-alpha connecting the two polarities. These are the chromospheric signatures of the newly emerged magnetic loops, with plasma draining down from the loop tops to the footpoints.

**Stage 5 -- Coronal loop formation**: EUV and X-ray observations show bright coronal loops connecting the two polarities, indicating that the emerged field has expanded to fill the coronal volume and is being heated to million-degree temperatures.

The entire process from first photospheric signature to fully developed coronal active region typically takes 1-3 days for a major active region.

### 3.4 Emergence Rate and Flux Content

The total magnetic flux in an active region spans several orders of magnitude:

| Category | Total flux (Mx) | Emergence timescale |
|----------|-----------------|-------------------|
| Ephemeral region | $10^{18}$-$10^{19}$ | Hours |
| Small AR | $10^{20}$-$10^{21}$ | 1-2 days |
| Large AR | $10^{21}$-$10^{22}$ | 2-5 days |
| Exceptional AR | $> 10^{22}$ | 5-10 days |

The largest active regions contain enough magnetic energy to power hundreds of major flares and CMEs throughout their lifetime.

---

## 4. Active Region Evolution and Decay

### 4.1 Mature Active Region Structure

A fully developed, mature active region typically shows a characteristic east-west asymmetry:

- **Leading spot** (westward in the direction of solar rotation): tends to be a single, well-formed sunspot with a regular penumbra. It is located closer to the equator than the following spots (Joy's law).
- **Following spots** (eastward): tend to be more fragmented, consisting of multiple smaller spots and pores rather than a single dominant spot. This asymmetry arises because the Coriolis force acts differently on the leading and following legs of the emerging flux tube.

The polarity pattern follows **Hale's law**: in a given solar cycle, leading spots in the northern hemisphere have one polarity (e.g., positive), while leading spots in the southern hemisphere have the opposite polarity (negative). This pattern reverses in the next solar cycle.

### 4.2 Joy's Law and the Babcock-Leighton Mechanism

Joy's law tilt is not merely an observational curiosity -- it is a fundamental ingredient in the solar dynamo. The **Babcock-Leighton mechanism** for generating the poloidal magnetic field works as follows:

1. A tilted bipolar active region emerges with the leading polarity closer to the equator
2. As the active region decays, its magnetic flux is dispersed by supergranular diffusion and meridional circulation
3. The leading polarity flux, being closer to the equator, is more likely to be transported across the equator and canceled with the opposite-polarity leading flux from the other hemisphere
4. The following polarity flux, being closer to the pole, is transported poleward by meridional flow
5. This preferentially sends following-polarity flux toward the poles, gradually reversing the polar field
6. The reversed polar field is the seed for the toroidal field of the next cycle (via differential rotation)

### 4.3 Active Region Decay

After reaching peak development (typically within a few days of initial emergence), an active region begins to decay. The decay process occurs on timescales of weeks to months and involves:

**Flux dispersal by turbulent diffusion**: the supergranular convective flow pattern continuously acts on the active region magnetic field, breaking it into smaller elements and dispersing them over an ever-larger area. The effective diffusivity is:

$$D_{\text{eff}} \approx \frac{1}{3} v_{\text{sg}} l_{\text{sg}} \approx \frac{1}{3} \times 0.3 \text{ km/s} \times 15 \text{ Mm} \approx 250 \text{ km}^2/\text{s}$$

where $v_{\text{sg}}$ and $l_{\text{sg}}$ are the supergranular velocity and scale.

**Flux cancellation**: when opposite-polarity magnetic elements are brought together by the flows, they cancel -- the photospheric field disappears as the flux submerges or reconnects. This is a primary mechanism for removing magnetic flux from the photosphere.

**MMF stripping**: as discussed in Section 2, moving magnetic features continuously remove flux from surviving sunspots.

The decay rate of sunspot area follows an approximately linear law:

$$\frac{dA}{dt} \approx -\text{const}$$

with typical rates of 10-50 MSH (millionths of the solar hemisphere) per day. A large sunspot with area 500 MSH thus survives for roughly 10-50 days.

---

## 5. Active Region Classification

### 5.1 McIntosh Classification (Z/p/c)

The **McIntosh classification** (McIntosh 1990) is a three-component system that describes the overall morphology of a sunspot group:

**Z -- Modified Zurich class** (A through H):
- **A**: unipolar, no penumbra
- **B**: bipolar, no penumbra
- **C**: bipolar, penumbra on one spot
- **D**: bipolar, penumbra on both spots, length $< 10°$
- **E**: bipolar, penumbra on both spots, length $10°$-$15°$
- **F**: bipolar, penumbra on both spots, length $> 15°$
- **H**: unipolar with penumbra (decayed AR with single surviving spot)

**p -- Penumbral type** of the largest spot:
- **x**: undefined (no penumbra)
- **r**: rudimentary
- **s**: small, symmetric
- **a**: small, asymmetric
- **h**: large, symmetric
- **k**: large, asymmetric

**c -- Compactness** of the spot distribution:
- **x**: undefined
- **o**: open (few spots, large separation)
- **i**: intermediate
- **c**: compact (many spots, densely packed)

For example, "Fkc" denotes a large bipolar group extending more than 15 degrees, with a large asymmetric penumbra on the dominant spot, and a compact spot distribution.

### 5.2 Hale/Mt. Wilson Magnetic Classification

The **Hale (Mt. Wilson) classification** is based on the magnetic polarity structure of the active region:

- **$\alpha$**: unipolar -- a single dominant polarity
- **$\beta$**: bipolar -- two well-separated regions of opposite polarity with a simple, well-defined polarity inversion line
- **$\gamma$**: complex -- the polarities are so intermingled that no single polarity inversion line can be drawn
- **$\delta$**: a special qualifier indicating that opposite-polarity umbrae share a common penumbra, separated by less than 2 degrees

The **$\beta\gamma\delta$** classification is the most flare-productive configuration. The $\delta$ designation is particularly significant because it indicates that opposite polarities are forced into close proximity -- this creates intense shear and strong electric currents along the polarity inversion line, storing large amounts of free magnetic energy.

### 5.3 Flare Productivity and Magnetic Complexity

The relationship between active region magnetic classification and flare productivity is well-established:

| Classification | X-class flare probability (per day) |
|---------------|-------------------------------------|
| $\alpha$, $\beta$ | $< 0.01$ |
| $\beta\gamma$ | $\sim 0.01$-$0.05$ |
| $\beta\delta$ | $\sim 0.05$-$0.1$ |
| $\beta\gamma\delta$ | $\sim 0.1$-$0.5$ |

These statistics are the basis of operational flare forecasting. Additional factors that increase flare probability include:

- Rapid flux emergence in an already-complex region
- Strong magnetic shear along the polarity inversion line
- High photospheric free energy (as measured by proxy parameters)
- Presence of a magnetic flux rope

### 5.4 NOAA Active Region Numbering

NOAA's Space Weather Prediction Center assigns a unique number to each sunspot group when it is first observed. The numbering has been continuous since 1972 (starting with AR 0001) and passed 10000 in 2002. As of 2024-2025, active region numbers are in the 13000-14000 range (Solar Cycle 25). A region retains its number even if it rotates off the visible disk and reappears 27 days later (one solar rotation), in which case it gets a new number.

---

## 6. Quiet Sun Magnetism

### 6.1 The Magnetic Carpet

The quiet Sun -- the vast expanse of the solar surface away from active regions -- was once thought to be essentially non-magnetic. Modern high-resolution observations have revealed that the quiet Sun is in fact permeated by magnetic fields at all scales, forming what is called the **magnetic carpet** (Title and Schrijver 1998).

The magnetic carpet consists of a constantly evolving pattern of small-scale magnetic flux concentrations that emerge, migrate, interact, cancel, and are replaced on timescales of hours. The total magnetic flux on the quiet Sun ($\sim 10^{23}$ Mx) actually exceeds the total flux in active regions during solar maximum.

Key properties of the magnetic carpet:

- **Flux recycling time**: approximately 14 hours -- the entire quiet Sun magnetic flux is replaced roughly every half day
- **Emergence rate**: $\sim 10^{24}$ Mx/day in the form of ephemeral regions and smaller bipoles
- **Energy implication**: the constant emergence and cancellation of small-scale flux may provide a significant fraction of the energy needed to heat the chromosphere and corona

### 6.2 Ephemeral Regions

**Ephemeral regions** are small, bipolar magnetic features with total flux of $10^{18}$-$10^{19}$ Mx and lifetimes of a few hours to about a day. They are the quiet Sun analog of active regions, appearing as small bipolar pairs in magnetograms that separate by a few megameters before canceling or being swept into the network.

Ephemeral regions emerge at a rate of roughly 600 per day over the entire Sun, corresponding to an emergence rate of $\sim 5 \times 10^{22}$ Mx/day. Unlike active regions, ephemeral regions show no preferred tilt angle (no Joy's law) and no latitude preference, suggesting they originate from a local, near-surface dynamo rather than from the deep-seated global dynamo that produces active regions.

### 6.3 Network Magnetic Elements

At the boundaries of supergranular cells, magnetic flux is concentrated into discrete, kilogauss-strength **network magnetic elements**. These elements have typical field strengths of 1000-1500 G despite being embedded in the quiet Sun, because the flux has been compressed into small cross-sections ($\sim 100$-$300$ km diameter) by the converging supergranular flows.

The concentration mechanism is known as **convective collapse** (Parker 1978): when a flux tube is compressed by external flows, its internal gas pressure decreases. If the flux tube is cooled (by reduced convective energy transport, as in a sunspot on a small scale), the internal pressure drops further, and the tube collapses to a smaller diameter and higher field strength until the magnetic pressure balances the external gas pressure.

Network magnetic elements are associated with:

- **G-band bright points**: tiny ($\sim 200$ km) brightenings visible in the G-band (4305 A, dominated by CH molecular lines). The brightness arises because the low density inside the flux tube allows radiation from deeper, hotter layers to escape through the tube walls.
- **Chromospheric network emission**: enhanced Ca II and Mg II emission due to heating associated with the concentrated magnetic flux
- **Spicule roots**: many Type I and Type II spicules originate from the magnetic network

### 6.4 Internetwork Fields

Between the network elements, in the interiors of supergranular cells, lies the **internetwork** -- a region of weak, mixed-polarity magnetic field that has proven extraordinarily difficult to characterize.

Current understanding of internetwork magnetic fields:

- **Field strength**: predominantly weak, in the range of 10-100 G (though isolated stronger concentrations exist)
- **Inclination**: highly inclined (more horizontal than vertical), in contrast to the predominantly vertical network fields
- **Distribution**: mixed polarity at scales below $\sim 1$ Mm, with no preferred orientation
- **Origin**: likely generated by a **local (surface) dynamo** driven by granular convection, independent of the global solar cycle

The internetwork magnetic field is the most poorly characterized component of solar magnetism because:

1. The weak fields produce Zeeman signals that are below the noise level of most magnetographs
2. The small-scale, mixed-polarity field partially cancels along the line of sight
3. Stokes $Q$ and $U$ signals (needed for transverse field) are very weak

Hanle effect measurements and high-sensitivity Stokes observations from Hinode and DKIST are gradually revealing the true nature of internetwork magnetism, with implications for small-scale dynamo theory and chromospheric/coronal energy supply.

### 6.5 Small-Scale Dynamo

The persistence of quiet Sun magnetic fields independent of the solar cycle, the lack of Joy's law tilt in ephemeral regions, and the presence of internetwork fields at all latitudes and solar cycle phases all point toward the operation of a **small-scale (or local) dynamo** driven by turbulent granular and supergranular convection.

In a small-scale dynamo, the kinetic energy of turbulent convection is converted directly into magnetic energy through the stretching, twisting, and folding of field lines by the flow. Unlike the global dynamo (which requires differential rotation and is organized on the scale of the Sun), the small-scale dynamo operates on the scale of granulation ($\sim 1$ Mm) and can generate magnetic energy as long as the magnetic Reynolds number $R_m = vl/\eta$ exceeds a critical threshold ($R_{m,\text{crit}} \sim 100$-$300$).

Numerical simulations of magneto-convection confirm that a small-scale dynamo operates in solar-like conditions, generating magnetic field with an energy spectrum that peaks at granular scales and a surface field strength distribution consistent with observations.

---

## Practice Problems

**Problem 1: Sunspot Brightness and Temperature**

A sunspot umbra has a temperature of $T_u = 3800$ K, while the surrounding photosphere has $T_p = 5778$ K. (a) Calculate the ratio of the umbral to photospheric surface brightness ($I_u/I_p$), assuming blackbody radiation. (b) A sunspot has an umbral area of $A_u = 100$ MSH ($1$ MSH $= 3.04 \times 10^{12}$ m$^2$). What is the "missing" luminosity -- the difference between what the area would radiate at the photospheric temperature versus the umbral temperature? (c) Observations show that sunspot groups do not cause a corresponding dip in total solar irradiance. Where does the "missing" energy go?

**Problem 2: Wilson Depression**

At the edge of a sunspot umbra ($B = 2500$ G), pressure balance requires $p_{\text{ext}} = p_{\text{int}} + B^2/(8\pi)$. (a) Calculate the magnetic pressure in dyn/cm$^2$ and in Pa. (b) If the external photospheric pressure at $\tau = 1$ is $p_{\text{ext}} = 1.2 \times 10^5$ dyn/cm$^2$, what fraction of the external pressure is provided by the magnetic field? (c) Using a photospheric pressure scale height of $H_p \approx 150$ km, estimate the Wilson depression (the depth below the external $\tau = 1$ surface at which the external pressure exceeds the surface external pressure by the magnetic pressure amount).

**Problem 3: Magnetic Flux Emergence**

An active region emerges with a flux growth rate of $d\Phi/dt = 5 \times 10^{19}$ Mx/hour for 48 hours. (a) What is the total flux of the active region after 48 hours? (b) Assuming the flux emerges in a circle of radius $R = 20$ Mm, what is the average unsigned vertical field strength at the photosphere? (c) The rise speed of the apex of the $\Omega$-loop from the tachocline ($r = 0.7 \, R_\odot$) takes approximately 2 months. Estimate the average rise velocity in km/s.

**Problem 4: Active Region Decay**

A sunspot has an initial area of $A_0 = 400$ MSH and decays linearly at a rate of $dA/dt = -25$ MSH/day. (a) How long does the sunspot survive? (b) If the sunspot magnetic flux is $\Phi = 5 \times 10^{21}$ Mx, and the flux is removed proportionally to area, what is the flux removal rate in Mx/s? (c) Compare this to the total quiet Sun flux recycling rate ($\sim 10^{24}$ Mx in 14 hours).

**Problem 5: Quiet Sun Magnetic Energy**

The quiet Sun has an average unsigned magnetic field of $\langle |B| \rangle \approx 20$ G over the internetwork and $\langle |B| \rangle \approx 200$ G in the network (which covers $\sim 10\%$ of the surface). (a) Calculate the average magnetic energy density (in erg/cm$^3$) for each component. (b) If the network field extends to a height of 2 Mm and the internetwork field to 500 km, estimate the total magnetic energy stored in the quiet Sun magnetosphere over the full solar surface ($A = 6.08 \times 10^{22}$ cm$^2$). (c) If the magnetic carpet recycles this flux every 14 hours, estimate the energy dissipation rate (in W/m$^2$) and compare it to the coronal heating requirement ($\sim 300$ W/m$^2$).

---

**Previous**: [Solar Magnetic Fields](./07_Solar_Magnetic_Fields.md) | **Next**: [Solar Dynamo and Cycle](./09_Solar_Dynamo_and_Cycle.md)

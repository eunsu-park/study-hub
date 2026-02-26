# Coronal Mass Ejections

## Learning Objectives

- Describe the observational characteristics of CMEs in coronagraph data
- Understand the primary CME initiation mechanisms including torus and kink instabilities
- Explain the structure of interplanetary CMEs (ICMEs) and magnetic clouds
- Analyze CME propagation using the drag-based model and MHD simulations
- Assess CME geo-effectiveness based on speed, magnetic field orientation, and impact geometry
- Discuss CME-CME interaction effects on space weather

---

## 1. CME Observations

Coronal mass ejections are enormous eruptions of magnetized plasma from the Sun's corona into the heliosphere. They are the most energetic events in the solar system after flares and represent the primary driver of severe space weather.

### 1.1 Coronagraph Observations

CMEs are observed primarily by coronagraphs — instruments that create an artificial eclipse by blocking the bright solar disk to reveal the faint corona. The workhorse instruments have been:

- **SOHO/LASCO C2**: Field of view 2–6 $R_\odot$, operational since 1996
- **SOHO/LASCO C3**: Field of view 3.7–30 $R_\odot$
- **STEREO/COR1 and COR2**: Twin spacecraft providing side views of the corona
- **K-Cor (Mauna Loa)**: Ground-based, 1.05–3 $R_\odot$

Coronagraphs detect Thomson-scattered white light from coronal electrons, so what we see in coronagraph images is the column-integrated electron density distribution projected onto the plane of the sky.

### 1.2 Three-Part Structure

The classical CME appearance in coronagraphs shows a three-part structure:

1. **Bright leading edge**: A sharp, bright front representing compressed coronal and solar wind plasma piled up ahead of the eruption. This is the CME shock/sheath region.
2. **Dark cavity**: A region of depleted density corresponding to the magnetic flux rope — the low-density, strong-field core of the eruption.
3. **Bright core**: A dense, bright structure within the cavity, often identified with erupting prominence material (cool, dense chromospheric plasma embedded in the flux rope).

Not all CMEs show this classic three-part structure. Some appear as narrow jets, diffuse clouds, or loop-like structures.

### 1.3 Halo CMEs

When a CME is directed along the Sun-Earth line (either toward or away from Earth), it appears as an expanding ring or halo around the occulting disk — a halo CME. A partial halo covers more than $120°$ of position angle. Halo CMEs directed toward Earth (front-side halos) are of primary concern for space weather, but disambiguating front-side from back-side halos requires additional data (e.g., on-disk signatures like flares, dimmings, EUV waves).

### 1.4 Statistical Properties

Over two solar cycles of LASCO observations:

| Property | Range | Median |
|----------|-------|--------|
| Speed | 200–3000 km/s | ~450 km/s |
| Angular width | $20°$–$360°$ | ~50° |
| Mass | $10^{12}$–$10^{13}$ kg | ~$2 \times 10^{12}$ kg |
| Kinetic energy | $10^{29}$–$10^{32}$ erg | ~$10^{30}$ erg |

The occurrence rate varies with the solar cycle: approximately 0.5 CMEs/day at solar minimum, increasing to about 3 CMEs/day at solar maximum. The speed distribution is roughly log-normal, with a tail of fast events ($>1500$ km/s) that are the most geo-effective.

### 1.5 On-Disk Signatures

Before a CME exits the coronagraph field of view, several associated phenomena can be observed on the solar disk:

- **Flare**: Bright emission in multiple wavelengths (but not all CMEs have associated flares, and not all flares produce CMEs)
- **Coronal dimmings**: Localized reductions in EUV/soft X-ray intensity near the eruption source, caused by the evacuation of coronal plasma
- **EUV wave**: A large-scale, quasi-circular bright front propagating across the disk at 200–400 km/s
- **Post-eruption arcade**: Growing system of bright loops spanning the polarity inversion line
- **Filament disappearance**: The associated prominence/filament erupts or partially erupts

---

## 2. CME Initiation Mechanisms

Understanding what triggers CMEs is one of the central problems in solar physics. Several mechanisms have been proposed, each likely operating in different circumstances.

### 2.1 Torus Instability

A magnetic flux rope embedded in an external (strapping) field experiences a hoop force directed outward and a restoring force from the external field. The balance is characterized by the decay index:

$$n = -\frac{\partial \ln B_{\text{ext}}}{\partial \ln h}$$

where $B_{\text{ext}}$ is the external poloidal field strength and $h$ is the height of the flux rope axis above the photosphere.

The torus instability occurs when the external field decreases sufficiently rapidly with height:

$$n > n_{\text{crit}} \approx 1.5$$

The physical intuition is straightforward: as the flux rope rises, the outward hoop force (which scales as $1/R$ for a current ring of radius $R$) decreases more slowly than the inward restoring force from the external field. When the external field drops off fast enough ($n > 1.5$), the restoring force cannot confine the rope, and it erupts.

Observational support: measurements of the coronal field decay index above active regions show that eruptions consistently occur when the flux rope reaches the height where $n \approx 1.5$.

### 2.2 Kink Instability

A magnetic flux rope with sufficient internal twist becomes unstable to a helical (kink) deformation. The critical condition is approximately:

$$\Phi > \Phi_{\text{crit}} \approx 3.5\pi$$

where $\Phi$ is the total twist of the flux rope (the angle through which field lines rotate about the axis over the length of the rope).

The kink instability causes the flux rope to writhe — converting internal twist into large-scale helical deformation. This can trigger reconnection with the overlying field and initiate eruption. However, the kink instability alone often does not produce full eruption — it may instead trigger the torus instability by lifting the flux rope to a critical height.

Observational signatures include S-shaped (sigmoid) morphology in soft X-ray and EUV images before eruption, and helical motion of erupting prominences.

### 2.3 Breakout Model

The breakout model (Antiochos et al., 1999) applies to multipolar magnetic configurations where a low-lying sheared arcade is confined by an overlying unsheared arcade:

1. Shearing motions at the PIL build up free energy in the core field.
2. The expanding core field presses against the overlying field, forming a current sheet at the null point (or quasi-separatrix layer).
3. Reconnection at this null ("breakout reconnection") removes overlying flux, reducing the confining field.
4. The core field erupts explosively once sufficient confining flux has been removed.
5. Flare reconnection then occurs below the erupting core, as in the standard CSHKP model.

The breakout model is attractive because it naturally produces the observed sequence of slow rise followed by sudden acceleration. It requires a specific multipolar topology, which may not be present in all erupting regions.

### 2.4 Flux Cancellation

Photospheric flux cancellation at the polarity inversion line can gradually build up a flux rope from a sheared arcade:

1. Converging flows bring opposite-polarity flux elements together at the PIL.
2. Reconnection between sheared field lines at the PIL creates flux rope field lines (with concave-up geometry) and small submerging loops.
3. The flux rope grows as more flux is added through continued cancellation.
4. Eventually, the flux rope becomes too massive or too twisted for the overlying field to confine, and it erupts via the torus or kink instability.

This mechanism provides a natural pathway for flux rope formation and eruption, and it is supported by extensive observational evidence of flux cancellation preceding CMEs.

For a more detailed theoretical treatment of these MHD instabilities, see MHD Lesson 06, Section 2 (MHD Instabilities and Eruptions).

---

## 3. CME Kinematics

The motion of a CME from initiation through interplanetary propagation can be broadly divided into three phases.

### 3.1 Initiation Phase (Slow Rise)

Before the impulsive eruption, the pre-CME structure (typically a prominence or flux rope) undergoes a slow rise with velocity $v < 10$ km/s. This phase can last hours to days and reflects the quasi-static evolution of the magnetic configuration toward the point of instability.

### 3.2 Impulsive Acceleration Phase

Once the instability threshold is crossed, the CME accelerates rapidly:
- Acceleration magnitude: 0.1–10 km/s$^2$ (occasionally higher)
- Duration: typically a few minutes to tens of minutes
- Occurs predominantly below ~3 $R_\odot$

The acceleration is driven by the Lorentz force — specifically, the imbalance between the upward magnetic pressure/tension force of the erupting flux rope and the downward tension of the overlying field.

A key observational result from Zhang and Dere (2006) is that the peak of the CME acceleration coincides temporally with the rise phase of the associated flare (the hard X-ray peak), providing strong evidence that the flare reconnection plays an active role in accelerating the CME — not merely a passive byproduct.

### 3.3 Propagation Phase

Beyond a few solar radii, the CME propagation is governed by its interaction with the ambient solar wind:
- Fast CMEs ($v_{\text{CME}} > v_{\text{sw}}$) decelerate due to aerodynamic drag
- Slow CMEs ($v_{\text{CME}} < v_{\text{sw}}$) accelerate toward the solar wind speed
- The asymptotic speed approaches the ambient solar wind speed ($\sim 400$–$600$ km/s)

---

## 4. Interplanetary CMEs (ICMEs)

When a CME propagates through the heliosphere and is detected by in-situ spacecraft instruments, it is called an interplanetary CME (ICME).

### 4.1 General ICME Signatures

In-situ measurements at 1 AU reveal several characteristic ICME signatures:
- Enhanced magnetic field magnitude ($B > 10$ nT, compared to ~5 nT ambient)
- Smooth, organized magnetic field rotation (in magnetic clouds)
- Low proton temperature (below the expected temperature for the observed speed)
- Low plasma beta ($\beta = 8\pi n k_B T / B^2 < 1$)
- Enhanced helium abundance ($N_{\alpha}/N_p > 5\%$, compared to ~4% in normal solar wind)
- Unusual charge states (e.g., Fe$^{16+}$, indicating formation temperature $> 3$ MK)
- Duration at 1 AU: typically 12–36 hours for the ICME body

### 4.2 Magnetic Clouds

A magnetic cloud (MC) is a subset of ICMEs (roughly 1/3) with particularly well-organized magnetic structure:

1. **Enhanced magnetic field**: $B > 10$ nT, often 15–30 nT
2. **Smooth rotation**: The magnetic field direction rotates smoothly through a large angle (often $>180°$) over the passage duration
3. **Low temperature**: Proton temperature well below the expected value

The magnetic field rotation is interpreted as the spacecraft passing through a large-scale magnetic flux rope. The simplest model is a constant-$\alpha$ force-free cylinder (Lundquist solution):

$$B_z = B_0 J_0(\alpha r), \qquad B_\theta = B_0 H J_1(\alpha r)$$

where $J_0$ and $J_1$ are Bessel functions, $B_0$ is the axial field strength, and $H = \pm 1$ is the chirality (handedness).

More sophisticated models include non-force-free cylinders, toroidal geometries, and reconstructions from Grad-Shafranov equilibria.

### 4.3 Non-Magnetic Cloud ICMEs

Approximately two-thirds of ICMEs do not show clean magnetic cloud signatures. This may be because:
- The spacecraft passes through the ICME flank rather than the center
- The flux rope has been significantly deformed by interactions
- Some ICMEs may not contain a well-formed flux rope (e.g., eruption of a sheared arcade without full flux rope formation)

### 4.4 Sheath Region

Fast ICMEs drive a shock ahead of them. Between the shock and the ICME leading edge lies the sheath:
- Compressed, heated solar wind plasma
- Turbulent, fluctuating magnetic field
- Duration: typically 6–12 hours at 1 AU
- Can be strongly geo-effective due to compressed $B_z$ southward fields

The sheath often causes the initial phase of geomagnetic storms, while the ICME body (especially if it is a magnetic cloud with sustained southward $B_z$) causes the main phase.

---

## 5. CME-CME Interaction

The heliosphere is not empty between CME events, and fast CMEs frequently interact with preceding slower CMEs or with stream interaction regions.

### 5.1 Collision and Merging

When a fast CME overtakes a slower one:
- **Compression**: The leading CME is compressed from behind, the trailing CME from the front
- **Momentum exchange**: The fast CME decelerates, the slow one accelerates (conservation of momentum)
- **Possible merging**: If the interaction is strong enough, the two CMEs can merge into a single, larger structure
- **Enhanced sheath**: The compressed region between the two CMEs can have very strong magnetic fields

### 5.2 Cannibalism

In extreme cases, a very fast CME can overtake and effectively absorb a much slower preceding CME. This "cannibalism" results in a single large, complex ICME at 1 AU that may be difficult to decompose into its constituent parts.

### 5.3 Preconditioning

A preceding CME can precondition the heliosphere for the following one:
- The first CME sweeps up solar wind, leaving a rarefied wake behind it
- The second CME propagates through this low-density channel with reduced drag
- Result: the second CME arrives at Earth faster than it would have in an undisturbed solar wind

This preconditioning effect is important for extreme space weather events, where a sequence of eruptions from the same active region produces a series of progressively faster-arriving CMEs.

### 5.4 Compound Events

Multiple interacting ICMEs arriving at Earth in succession can produce compound geomagnetic storms. The 2003 Halloween storms and the July 2012 event (detected by STEREO-A) are examples where the interaction of multiple CMEs produced conditions far more extreme than any single event would have.

---

## 6. CME Propagation Models

Predicting when and how a CME will arrive at Earth is a central challenge of space weather forecasting.

### 6.1 Drag-Based Model (DBM)

The simplest physics-based propagation model treats the CME as a rigid body moving through the solar wind and experiencing aerodynamic drag:

$$a = -\gamma(v - v_{\text{sw}})|v - v_{\text{sw}}|$$

where:
- $a$ is the CME acceleration
- $v$ is the CME speed
- $v_{\text{sw}}$ is the ambient solar wind speed
- $\gamma$ is the drag parameter: $\gamma = c_d A \rho_{\text{sw}} / M_{\text{CME}}$
- $c_d \sim 1$ is the drag coefficient
- $A$ is the CME cross-sectional area
- $\rho_{\text{sw}}$ is the solar wind density
- $M_{\text{CME}}$ is the CME mass

Typical values: $\gamma \sim 0.2$–$2 \times 10^{-7}$ km$^{-1}$.

The equation can be solved analytically:

$$r(t) = \pm \frac{1}{\gamma}\ln\left[1 \pm \gamma(v_0 - v_{\text{sw}})t\right] + v_{\text{sw}}t + r_0$$

where the $+$ sign applies for $v_0 > v_{\text{sw}}$ (decelerating CME) and $-$ for $v_0 < v_{\text{sw}}$ (accelerating CME).

The DBM is remarkably effective for its simplicity, with typical arrival time errors of $\pm 10$–$12$ hours for well-observed events.

### 6.2 Cone Model

The cone model approximates the CME as a cone with half-angle $\omega$ propagating radially from the Sun. Combined with the DBM, it provides a 3D kinematic model. The input parameters (speed, direction, angular width) are derived from coronagraph observations, often using geometric triangulation from multiple viewpoints (STEREO + SOHO).

### 6.3 Ensemble Modeling

To account for uncertainties in the input parameters, ensemble approaches run the propagation model many times with varied inputs:

- Speed: $v_0 \pm \sigma_v$ (typically $\sigma_v \sim 100$–$200$ km/s)
- Width: $\omega \pm \sigma_\omega$
- Direction: $(\phi, \theta) \pm \sigma_d$

The resulting distribution of arrival times provides a probabilistic forecast. This approach has significantly improved the reliability of space weather predictions.

### 6.4 MHD Heliospheric Models

Full 3D MHD simulations of CME propagation through a realistic solar wind background represent the state of the art:

- **ENLIL/WSA**: The most widely used operational model. WSA (Wang-Sheeley-Arge) provides the ambient solar wind, ENLIL solves the 3D MHD equations with a cone-model CME inserted at 21.5 $R_\odot$.
- **EUHFORIA** (European Heliospheric Forecasting Information Asset): More advanced, can inject flux-rope CMEs (not just cones), providing predictions of the internal magnetic field structure.
- **SWASTi** and other emerging models

Typical arrival time accuracy: $\pm 6$–$8$ hours for MHD models, compared to $\pm 10$–$12$ hours for the DBM. The major remaining challenge is predicting the magnetic field within the ICME, especially the crucial $B_z$ component.

---

## 7. Geo-effectiveness Assessment

Not all CMEs that hit Earth cause geomagnetic storms. The geo-effectiveness depends on several parameters.

### 7.1 Key Parameters

1. **Speed**: Determines the dynamic pressure ($\propto \rho v^2$) and energy input. Faster ICMEs compress the magnetosphere more and drive stronger storms.

2. **Southward magnetic field ($B_z < 0$)**: This is the single most important parameter. Southward interplanetary magnetic field reconnects with Earth's northward dayside magnetopause field, allowing solar wind energy to enter the magnetosphere. The reconnection rate scales as:

$$\Phi_{\text{reconnection}} \propto v \cdot B_s \cdot l$$

where $B_s$ is the southward field component and $l$ is the length of the reconnection line.

3. **Impact parameter**: Whether the Earth is hit by the ICME nose (direct hit) or flank (glancing blow). Glancing blows are less geo-effective.

4. **Duration**: Longer-duration southward $B_z$ produces more sustained energy input, leading to larger storms.

### 7.2 Empirical Relations

The Burton et al. (1975) equation and its variants relate the Dst (disturbance storm time) index to solar wind parameters:

$$\frac{dDst^*}{dt} = Q(t) - \frac{Dst^*}{\tau}$$

where $Dst^* = Dst - b\sqrt{P_{\text{dyn}}} + c$ is the pressure-corrected Dst, $\tau$ is the ring current decay time (~hours), and $Q(t)$ is the injection function. A commonly used empirical form:

$$Q = -4.4(v B_s - E_c) \quad \text{for } vB_s > E_c$$

where $E_c \approx 0.50$ mV/m is a threshold. This directly shows that $Dst_{\min} \propto -v \times B_s$: the product of speed and southward field determines the storm intensity.

### 7.3 The B_z Prediction Challenge

Predicting the magnetic field orientation within an ICME before it arrives at Earth is the "holy grail" of space weather forecasting. Current limitations:

- Coronagraph images show the density structure but not the magnetic field
- Faraday rotation measurements can constrain the field but are rarely available
- MHD models (EUHFORIA with flux rope insertion) can predict $B_z$, but require accurate knowledge of the source region magnetic structure
- L1 monitors (ACE, DSCOVR, Wind) provide only ~30–60 minutes of warning — often insufficient for mitigation

### 7.4 Extreme Events

The most extreme geo-effective events combine multiple unfavorable factors:
- Very fast CME ($>2000$ km/s)
- Sustained southward $B_z$ ($<-30$ nT) for many hours
- Direct nose impact
- Possible CME-CME interaction enhancing the fields

The Carrington event (1859) is estimated to have produced $Dst \approx -850$ nT. A comparable event today could cause widespread power grid failures, satellite damage, and communication disruptions with economic costs estimated at \$1–2 trillion.

---

## Practice Problems

1. **CME Mass and Energy**: A CME observed in LASCO C2 has an apparent speed of 1200 km/s and an estimated mass of $5 \times 10^{12}$ kg. (a) Calculate its kinetic energy. (b) If the CME also has an internal magnetic field of $B = 50$ mT (50 nT at 1 AU scaled back to the corona) filling a sphere of radius $2 \times 10^{10}$ cm, estimate the magnetic energy. (c) Compare the kinetic and magnetic energies and discuss which dominates at different heliocentric distances.

2. **Torus Instability**: The external (strapping) field above an active region varies as $B_{\text{ext}}(h) = B_0 (h/h_0)^{-n}$ where $B_0 = 200$ G at $h_0 = 5 \times 10^9$ cm. (a) For what decay index $n$ does the torus instability occur ($n_{\text{crit}} = 1.5$)? (b) If $n = 1.0$ at low heights and increases to $n = 2.0$ at $h = 1.5 \times 10^{10}$ cm, at approximately what height does the instability onset occur (assume $n$ varies linearly with $\ln h$)?

3. **Drag-Based Model**: A fast CME is launched at $r_0 = 20 R_\odot$ with initial speed $v_0 = 2000$ km/s into a solar wind of $v_{\text{sw}} = 400$ km/s. The drag parameter is $\gamma = 0.5 \times 10^{-7}$ km$^{-1}$. (a) Using $v(r) = v_{\text{sw}} + (v_0 - v_{\text{sw}})e^{-\gamma(r - r_0)}$, estimate the CME speed at 1 AU ($r = 215 R_\odot$). (b) Estimate the transit time from $r_0$ to 1 AU.

4. **Magnetic Cloud Fitting**: An ICME passes over a spacecraft at 1 AU. The measured $B_z$ component rotates smoothly from $+15$ nT to $-20$ nT to $+5$ nT over 24 hours. (a) Sketch the expected $B_z$ profile for a Lundquist flux rope model passing near the axis. (b) Estimate the impact parameter (closest approach distance to the rope axis relative to the rope radius) from the asymmetry of the profile. (c) What is the chirality (sign of helicity) if $B_y$ rotates from negative to positive?

5. **Geo-effectiveness**: Two ICMEs arrive at Earth. ICME-A has $v = 700$ km/s, $B_z = -15$ nT for 6 hours. ICME-B has $v = 500$ km/s, $B_z = -25$ nT for 4 hours. (a) Calculate $v \times B_s$ for each event. (b) Estimate which produces the larger Dst minimum using $Dst_{\min} \propto -v \times B_s \times \Delta t^{0.5}$. (c) Discuss whether the simple scaling captures all relevant physics.

---

**Previous**: [Solar Flares](./10_Solar_Flares.md) | **Next**: [Solar Wind](./12_Solar_Wind.md)

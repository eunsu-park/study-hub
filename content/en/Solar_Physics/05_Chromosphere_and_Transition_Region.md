# Chromosphere and Transition Region

## Why This Matters

The chromosphere and transition region are where some of the most fundamental puzzles in solar physics reside. This is the atmospheric zone where the temperature *increases* as you move away from the energy source — like a blanket that somehow gets hotter the farther it is from the fire. The chromosphere is where most of the Sun's ultraviolet radiation originates, where the seeds of coronal heating are planted, and where the transition to million-degree plasma occurs across a startlingly thin layer. Understanding this region is essential because it mediates all energy and mass transport between the visible surface and the corona: every coronal loop is rooted here, every solar wind stream passes through here, and every flare deposits much of its energy here.

---

## Learning Objectives

1. Describe the chromospheric temperature structure and the location of the temperature minimum
2. Identify key spectral diagnostics used to observe the chromosphere (H-alpha, Ca II, Mg II, He I)
3. Explain the two types of spicules and their distinct dynamic properties
4. Understand the transition region as a temperature regime rather than a geometric layer
5. Define the Differential Emission Measure (DEM) and its role in characterizing transition region plasma
6. Discuss the energy balance of the chromosphere including radiative losses and heating mechanisms
7. Relate the acoustic cutoff frequency to the dominant oscillation periods in the chromosphere

---

## 1. Temperature Structure Above the Photosphere

### 1.1 The Temperature Minimum

As we ascend from the visible surface of the Sun (the photosphere), the temperature does not simply continue to drop as one might naively expect from a hot body radiating into space. Instead, it decreases to a minimum of approximately 4400 K at an altitude of about 500 km above the photospheric $\tau_{500} = 1$ surface. This **temperature minimum region** marks the boundary between the photosphere and the chromosphere.

The temperature minimum can be understood from the perspective of radiative equilibrium. In the photosphere, the temperature gradient is set by radiative (and convective) energy transport. Above the photosphere, the decreasing density means less absorption of photospheric radiation, so the temperature drops. However, at some point, non-radiative heating mechanisms begin to deposit energy faster than the tenuous gas can radiate it away, and the temperature begins to rise.

### 1.2 The Chromospheric Temperature Rise

Above the temperature minimum, the temperature rises through the chromosphere from roughly 4400 K to about 25,000 K over a height range of approximately 2000 km. This increase is gradual compared to what comes next, but it is remarkable in itself: the gas is becoming hotter as we move away from the energy source.

The name **chromosphere** literally means "color sphere." It was coined because during total solar eclipses, just as the Moon covers the photospheric disk, a thin reddish ring becomes visible around the solar limb. This red color comes from hydrogen Balmer-alpha (H-alpha) emission at 656.3 nm, which is the dominant visible emission from chromospheric plasma at temperatures around 10,000 K.

### 1.3 The Transition Region

The most dramatic temperature change in the solar atmosphere occurs in the **transition region (TR)**, where the temperature skyrockets from approximately $2.5 \times 10^4$ K to over $10^6$ K across a remarkably thin layer of only about 100 km. This corresponds to a temperature gradient of order:

$$\frac{dT}{dh} \sim \frac{10^6 - 2.5 \times 10^4}{100 \text{ km}} \approx 10 \text{ K/m}$$

To appreciate how extreme this is, imagine walking 100 km and experiencing a temperature change from room temperature to the interior of a fusion reactor. The transition region is arguably the most poorly understood part of the solar atmosphere, precisely because its thinness and dynamism make it extraordinarily difficult to observe and model.

### 1.4 Physical Origin of the Temperature Profile

Why does this peculiar temperature profile exist? The fundamental reason is that the chromosphere and corona are heated by mechanical and magnetic energy originating in the convection zone. The temperature profile reflects a balance between:

- **Heating**: mechanical (acoustic waves, MHD waves) and magnetic (reconnection, current dissipation) energy deposition
- **Radiative cooling**: which is extremely efficient in the chromosphere (optically thick lines) but drops sharply in the transition region and corona (optically thin)
- **Thermal conduction**: which is highly efficient along magnetic field lines in the corona ($\kappa \propto T^{5/2}$) and conducts heat downward from the corona into the transition region

The steep transition region gradient is fundamentally set by the balance between downward thermal conduction from the hot corona and radiative losses in the cooler, denser plasma below.

---

## 2. Chromospheric Spectral Diagnostics

The chromosphere is best observed through specific spectral lines and continua that form at chromospheric temperatures. Each diagnostic provides a different window into chromospheric physics.

### 2.1 Hydrogen H-alpha (656.3 nm)

H-alpha is the workhorse of chromospheric observation. This Balmer series line (transition from $n = 3$ to $n = 2$) forms across a broad range of chromospheric heights and temperatures (roughly 6,000-15,000 K), making it sensitive to a wide variety of chromospheric structures.

In H-alpha images of the solar disk, one sees:

- **Filaments**: dark, elongated structures that are actually prominences seen in absorption against the disk. They trace polarity inversion lines (PILs) of the photospheric magnetic field.
- **Fibrils**: fine thread-like structures radiating from network magnetic elements, tracing the chromospheric magnetic field topology.
- **Plage**: bright regions overlying photospheric faculae in active regions, indicating enhanced chromospheric heating.

At the solar limb, H-alpha reveals prominences as bright structures extending into the corona, and spicules as a dynamic "burning prairie" at the chromospheric boundary.

### 2.2 Calcium II H and K Lines (396.8 nm, 393.4 nm)

The Ca II H and K resonance lines are among the strongest absorption lines in the solar spectrum (the H and K designations are Fraunhofer's original labels). These lines have complex profiles with:

- Broad absorption wings formed in the photosphere
- An emission reversal core formed in the chromosphere (the $H_2$ and $K_2$ peaks)
- A central absorption dip ($H_3$, $K_3$) formed in the upper chromosphere

The emission core brightness is strongly correlated with magnetic field strength, making Ca II a powerful diagnostic of chromospheric magnetic activity. The **chromospheric network** -- the pattern of enhanced emission outlining supergranular cell boundaries -- is beautifully visible in Ca II K filtergrams.

### 2.3 Magnesium II h and k Lines (279.6 nm, 280.3 nm)

The Mg II h and k lines are ultraviolet analogs of Ca II H and K, forming at similar heights but with several advantages: magnesium is about 18 times more abundant than calcium, and the UV wavelength means less photospheric contamination. These lines are the primary diagnostics of NASA's **IRIS** (Interface Region Imaging Spectrograph) mission, launched in 2013.

IRIS provides simultaneous imaging and spectroscopy at 0.33 arcsecond spatial resolution, revealing the chromosphere and transition region at unprecedented detail. The Mg II line profiles encode information about chromospheric velocity, temperature, and turbulence through their Doppler shifts, widths, and asymmetries.

### 2.4 Helium I 10830 Angstrom

The He I 10830 A triplet line has a unique formation mechanism: it requires **photoionization by coronal EUV radiation** followed by recombination into the triplet system. This means the line is dark (absorbing) only where the chromosphere is illuminated from above by coronal radiation. In coronal holes, where the corona is dim, the 10830 line is weak or absent.

This property makes He I 10830 an excellent diagnostic for mapping the coronal magnetic field topology from ground-based observations. It is also highly sensitive to magnetic fields via the Zeeman and Hanle effects.

### 2.5 Radio Continuum at Millimeter Wavelengths

At millimeter wavelengths (e.g., 1-3 mm, observed by ALMA), the solar emission is thermal bremsstrahlung (free-free emission) from the chromosphere. The brightness temperature directly traces the electron temperature at the height where the optical depth reaches unity. ALMA's high spatial resolution ($\sim 1$ arcsecond at 1 mm) provides a powerful thermometer for chromospheric plasma.

---

## 3. Chromospheric Fine Structure

### 3.1 Spicules: The Chromospheric Forest

Spicules are narrow, jet-like features that protrude from the chromospheric limb into the corona, giving the chromosphere the appearance of a "burning prairie" when viewed at the limb in H-alpha. They are among the most ubiquitous dynamic features of the solar atmosphere, with approximately $10^6$ spicules present on the Sun at any given time.

Modern high-resolution observations have revealed two distinct types of spicules with fundamentally different properties:

**Type I Spicules:**

- **Driving mechanism**: leakage of photospheric p-mode oscillations (5-minute oscillations) into the chromosphere
- **Rise speed**: 20-40 km/s
- **Maximum height**: 3-5 Mm above the limb
- **Lifetime**: 3-7 minutes
- **Period**: approximately 5 minutes, reflecting the driving p-mode oscillation
- **Behavior**: rise, reach maximum height, then **fall back** along the same trajectory
- **Occurrence**: predominantly at network boundaries

The connection between p-modes and Type I spicules is elegant. The photospheric 5-minute oscillations are acoustic waves trapped in the convection zone cavity. At the photosphere, the acoustic cutoff period is about 5.2 minutes (see Section 5), so the dominant 5-minute oscillations are evanescent. However, the inclined magnetic field at network boundaries effectively reduces the cutoff frequency (by a factor of $\cos\theta$ where $\theta$ is the field inclination), allowing the p-mode waves to propagate upward. These propagating waves steepen into shocks in the rapidly decreasing density of the chromosphere, driving material upward as spicules.

**Type II Spicules:**

- **Rise speed**: 50-100 km/s (much faster)
- **Maximum height**: 3-9 Mm
- **Lifetime**: approximately 1 minute (much shorter)
- **Behavior**: rise rapidly, then **fade from chromospheric passbands** without falling back
- **Fate**: appear to be heated to transition region or coronal temperatures

Type II spicules are particularly intriguing because their disappearance from chromospheric diagnostics without a corresponding downflow suggests they are heated beyond chromospheric temperatures. If the material is indeed heated to $\sim 10^5$ K or higher, Type II spicules could be a significant contributor to the mass and energy supply of the corona. However, this interpretation remains debated, with some studies suggesting they may simply move out of the instrumental passband due to rapid deceleration.

### 3.2 Mottles and Fibrils

When viewed on the solar disk rather than at the limb, the chromospheric counterparts of spicules are called **mottles** (in quiet Sun) and **fibrils** (in active regions). These are elongated dark or bright features in H-alpha that trace the magnetic field geometry in the chromosphere.

- **Dark mottles**: absorption features typically 5-10 Mm long, seen in the blue wing of H-alpha (indicating upflows)
- **Fibrils**: longer, thinner structures in active regions that bridge polarity inversion lines and connect regions of opposite magnetic polarity
- **Dynamic fibrils**: short, periodic jet-like features in active region plage, driven by magnetoacoustic shocks

### 3.3 Plage and the Chromospheric Network

**Plage** regions are areas of enhanced chromospheric emission overlying photospheric faculae in active regions. They appear bright in H-alpha, Ca II, and Mg II, indicating enhanced chromospheric heating due to the concentration of magnetic flux.

The **chromospheric network** outlines the boundaries of supergranular convection cells ($\sim 30$ Mm diameter). Magnetic flux is swept to these boundaries by supergranular flows, and the concentrated magnetic field leads to enhanced heating and emission. The network is visible as a pattern of bright emission lanes in Ca II K images, with darker **internetwork** cell interiors.

---

## 4. The Transition Region

### 4.1 Classical vs. Modern View

The classical picture of the transition region, developed in the 1970s from spatially unresolved EUV spectra, envisioned it as a thin, static, stratified layer where temperature increased monotonically with height. In this picture, the TR structure was determined entirely by the balance between downward thermal conduction from the corona and radiative losses.

The modern view, informed by high-resolution UV and EUV observations from SOHO, TRACE, SDO, and IRIS, is dramatically different. The transition region is:

- **Not a geometric layer**: TR-temperature plasma exists at many different heights, both low (near the chromosphere) and high (in coronal loops)
- **Highly dynamic**: spicules, jets, and explosive events continuously reshape the TR
- **Structured by the magnetic field**: the TR "moves" up and down along magnetic field lines as heating and cooling vary
- **Multi-thermal**: any given line of sight typically intersects plasma at many different temperatures

A more accurate description is that the transition region is a **temperature regime** ($\sim 2 \times 10^4$ to $10^6$ K) rather than a spatial layer, and plasma in this temperature range is found wherever the local energy balance demands it.

### 4.2 Key Transition Region Emission Lines

The transition region is observed primarily through UV and EUV emission lines formed at characteristic temperatures:

| Line | Wavelength | log T (K) | Notes |
|------|-----------|-----------|-------|
| C II | 1334, 1336 A | 4.4 | Lower TR, optically thick |
| Si IV | 1394, 1403 A | 4.9 | IRIS primary TR line |
| C IV | 1548, 1551 A | 5.0 | Classic TR doublet |
| O IV | 1401 A | 5.2 | Density diagnostic |
| N V | 1239, 1243 A | 5.3 | Upper TR |
| O VI | 1032, 1038 A | 5.5 | Upper TR, SUMER/UVCS |

Each line provides a "snapshot" of plasma at a specific temperature, and the combination of multiple lines allows reconstruction of the temperature distribution.

### 4.3 Differential Emission Measure

The **Differential Emission Measure (DEM)** is the fundamental quantity describing the thermal structure of optically thin plasma along a line of sight. It is defined as:

$$\text{DEM}(T) = n_e^2 \frac{dh}{dT}$$

where $n_e$ is the electron density and $dh/dT$ is the inverse of the temperature gradient along the line of sight. The DEM has units of cm$^{-5}$ K$^{-1}$ and tells us how much emitting material exists at each temperature.

Why does $n_e^2$ appear rather than just $n_e$? The dominant emission process in optically thin coronal and transition region plasma is **collisional excitation followed by radiative de-excitation**: a free electron collides with an ion, exciting it, and the ion then emits a photon. The collision rate is proportional to both the electron density and the ion density ($n_e \times n_i$), and since the plasma is nearly fully ionized with $n_i \approx n_e$, the emissivity scales as $n_e^2$. This is the same quadratic density scaling seen in bremsstrahlung (free-free) emission — a universal signature of two-body processes.

The factor $dh/dT$ represents the **column depth per unit temperature interval** — physically, how much "thickness" of atmosphere contains plasma at temperature $T$. Where the temperature gradient is steep (like the transition region), $dh/dT$ is small (plasma passes quickly through each temperature), and where the gradient is shallow, $dh/dT$ is large (plasma "lingers" at that temperature). The observed intensity of a spectral line formed at temperature $T$ is proportional to $\text{DEM}(T)$ weighted by the line's emissivity function — so by observing many lines formed at different temperatures and inverting the resulting system of equations, we reconstruct the DEM and thereby map the temperature structure of the plasma.

Physically, the DEM is large where:

- The density is high (bright emission), **and/or**
- The temperature gradient is shallow (plasma "lingers" at that temperature)

In the transition region, the DEM typically shows a characteristic shape: rising steeply from coronal temperatures down to $\sim 10^5$ K (roughly as $T^{-3}$ to $T^{-2}$), reflecting the increasing density at lower temperatures. Below $\sim 10^5$ K, the DEM behavior depends strongly on the magnetic environment.

### 4.4 Transition Region Dynamics

IRIS observations have revealed a wealth of dynamic phenomena in the transition region:

- **UV bursts**: compact, intense brightenings in Si IV with complex, broadened profiles indicating reconnection in the low atmosphere
- **Explosive events**: bidirectional jets seen as non-Gaussian line wings, associated with reconnection at network boundaries
- **Transition region moss**: the footpoints of hot ($>3$ MK) coronal loops, seen as dynamic bright patches in EUV 171 A images
- **Network jets**: small, repetitive jets from magnetic network elements reaching TR and coronal temperatures

---

## 5. Energy Balance

### 5.1 Radiative Losses

The chromosphere is an enormously efficient radiator. The total radiative loss from the chromosphere is approximately:

$$F_{\text{rad, chrom}} \approx 4 \text{ kW/m}^2$$

This is dominated by a few strong spectral lines (Ca II H&K, Mg II h&k, H-alpha, Lyman-alpha) and the hydrogen Lyman continuum. The chromosphere is optically thick in these lines, meaning radiative transfer is complex and NLTE (non-local thermodynamic equilibrium) effects are crucial.

The transition region, despite its extreme temperature gradient, has much smaller radiative losses:

$$F_{\text{rad, TR}} \approx 0.1 \text{ kW/m}^2$$

This is because the TR is optically thin and has a very small geometric extent. The emissivity of optically thin plasma (per unit volume) actually increases strongly with density squared, but the TR is so thin that the total column emission is modest.

For comparison, the total solar luminosity corresponds to a surface flux of $6.3 \times 10^7$ W/m$^2$, so the chromospheric radiative losses are only about $6 \times 10^{-5}$ of the photospheric output. This is why the coronal heating "problem" is really a problem of mechanism and detail, not of total energy -- the Sun has more than enough energy; the question is how it gets to the corona.

### 5.2 Heating Mechanisms

**Acoustic wave heating** plays an important role in the quiet Sun chromosphere. Acoustic waves generated by turbulent convection propagate upward and steepen into shocks due to the exponentially decreasing density. The shock dissipation deposits energy throughout the chromosphere.

However, acoustic heating alone appears insufficient to account for the full chromospheric energy budget, especially in active regions. **Magnetic heating** -- through mechanisms such as resistive dissipation of currents, reconnection, and MHD wave damping -- is believed to dominate in regions of significant magnetic field.

The relative importance of these mechanisms varies with location:

- **Quiet Sun internetwork**: primarily acoustic (p-mode shocks)
- **Quiet Sun network**: mix of acoustic and magnetic
- **Active region plage**: primarily magnetic
- **Active region umbra**: magneto-acoustic waves (3-min oscillations)

### 5.3 The Acoustic Cutoff Frequency

A key concept for understanding chromospheric dynamics is the **acoustic cutoff frequency**. In an isothermal, stratified atmosphere, only acoustic waves with frequency above the cutoff can propagate upward; lower-frequency waves are evanescent (their amplitude decays exponentially with height).

The acoustic cutoff angular frequency is:

$$\omega_{\text{ac}} = \frac{\gamma g}{2 c_s}$$

where $\gamma$ is the adiabatic index ($5/3$ for an ideal monatomic gas), $g$ is the gravitational acceleration ($274$ m/s$^2$ at the solar surface), and $c_s$ is the sound speed. For chromospheric conditions ($T \approx 7000$ K, $c_s \approx 8$ km/s):

$$\omega_{\text{ac}} \approx \frac{5/3 \times 274}{2 \times 8000} \approx 0.029 \text{ rad/s}$$

$$f_{\text{ac}} = \frac{\omega_{\text{ac}}}{2\pi} \approx 4.6 \text{ mHz}$$

corresponding to a cutoff **period** of about $P_{\text{ac}} = 1/f_{\text{ac}} \approx 3.6$ minutes.

This explains one of the most distinctive observational facts about solar oscillations:

- The **photosphere** is dominated by 5-minute oscillations ($f \approx 3.3$ mHz), which are the global p-modes trapped in the convection zone cavity
- The **chromosphere** is dominated by 3-minute oscillations ($f \approx 5.5$ mHz), which are above the cutoff and can propagate upward

The 5-minute p-modes, being below the cutoff, are evanescent in the chromosphere -- they cannot propagate upward and instead "leak" only a small fraction of their energy. The 3-minute waves, being above the cutoff, propagate freely, grow in amplitude due to the decreasing density ($v \propto \rho^{-1/2}$ for constant energy flux), and steepen into shocks that heat the chromosphere.

An important subtlety is that the effective cutoff frequency is modified by the magnetic field. In an inclined magnetic field making angle $\theta$ with the vertical, the effective cutoff frequency is reduced:

$$\omega_{\text{ac, eff}} = \omega_{\text{ac}} \cos\theta$$

This means that in regions of highly inclined field (such as at the edges of network flux concentrations), even 5-minute oscillations can propagate upward. This is why Type I spicules, driven by 5-minute oscillations, occur preferentially at network boundaries where the field is inclined.

---

## 6. Moreton and EIT Waves

### 6.1 Moreton Waves

In 1960, Gail Moreton discovered large-scale wave-like disturbances propagating across the solar chromosphere in H-alpha observations. These **Moreton waves** appear as arc-shaped brightenings (in the red wing of H-alpha) or darkenings (in the blue wing) sweeping away from flare sites at speeds of 500-1500 km/s.

The physical interpretation is that Moreton waves are the chromospheric footprint of a coronal shock wave. As the fast-mode MHD shock sweeps through the corona, it pushes down on the chromosphere, compressing and displacing the chromospheric material. The resulting downward velocity shift causes the characteristic H-alpha wing signatures.

The propagation speed of Moreton waves ($v \sim 500$-$1500$ km/s) is consistent with the fast-mode MHD speed in the corona:

$$v_f = \sqrt{c_s^2 + v_A^2}$$

where $c_s$ is the coronal sound speed ($\sim 150$-$200$ km/s for $T \sim 1$-$2$ MK) and $v_A$ is the Alfven speed ($\sim 500$-$1500$ km/s for typical coronal conditions).

### 6.2 EIT Waves

With the launch of SOHO in 1995 and its EIT (Extreme-ultraviolet Imaging Telescope) instrument, large-scale propagating disturbances were discovered in coronal EUV images, now called **EIT waves** (or more generally, coronal bright fronts). These appear as diffuse brightenings propagating outward from eruption sites, typically at speeds of 200-400 km/s -- notably slower than Moreton waves.

The interpretation of EIT waves remains debated, with two main models:

**Fast-mode MHD wave interpretation**: EIT waves are the coronal counterpart of Moreton waves -- large-scale fast magnetosonic waves. The lower apparent speed compared to Moreton waves may reflect projection effects, the wave slowing in regions of lower Alfven speed, or the EUV emission responding to a different part of the disturbance than H-alpha.

**CME-driven compression interpretation**: EIT waves are not true waves at all, but rather the compression front of the expanding CME flanks pushing into the surrounding corona. In this picture, the propagation speed reflects the lateral expansion speed of the CME, not the local wave speed.

SDO/AIA observations with high cadence (12 s) and multiple EUV channels have revealed that both interpretations may be partially correct: the initial fast component is a genuine fast-mode wave (sometimes seen as a Moreton wave in the chromosphere), while the slower, more diffuse trailing component is associated with CME restructuring of the coronal magnetic field.

---

## Practice Problems

**Problem 1: Temperature Gradient in the Transition Region**

The transition region spans a temperature range from $2.5 \times 10^4$ K to $10^6$ K over approximately 100 km. (a) Calculate the average temperature gradient $dT/dh$ in K/km. (b) If the thermal conductivity in this regime is $\kappa = \kappa_0 T^{5/2}$ with $\kappa_0 = 10^{-6}$ erg cm$^{-1}$ s$^{-1}$ K$^{-7/2}$, estimate the downward conductive heat flux at $T = 5 \times 10^5$ K assuming your average gradient. (c) Compare this to the chromospheric radiative loss rate of 4 kW/m$^2$.

**Problem 2: Acoustic Cutoff and Spicule Driving**

(a) Calculate the acoustic cutoff period for an isothermal atmosphere at $T = 6000$ K (use $\mu = 1.3 m_H$ for the mean molecular weight, $\gamma = 5/3$). (b) A magnetic flux tube is inclined at $\theta = 50^\circ$ from the vertical. What is the effective acoustic cutoff period in this tube? (c) Explain why this result is relevant for the formation of Type I spicules at network boundaries.

**Problem 3: Spicule Mass Flux**

Assume there are $N = 10^6$ spicules on the Sun at any time, each with an average diameter of 500 km, an upflow speed of 25 km/s, and an electron density of $n_e = 2 \times 10^{10}$ cm$^{-3}$. (a) Estimate the total mass flux carried by spicules (in kg/s). (b) The solar wind mass loss rate is approximately $2 \times 10^9$ kg/s. How does the spicule mass flux compare? (c) What does this imply about the fate of most spicule material?

**Problem 4: Differential Emission Measure**

Consider a simple model where the transition region is a slab of constant pressure $p = n_e k_B T$ (for a pure hydrogen plasma with $n_e = n_p$) and the temperature varies linearly from $T_1 = 10^5$ K to $T_2 = 10^6$ K over a height $\Delta h = 100$ km. (a) Express $n_e(T)$ in terms of pressure $p$ and temperature $T$. (b) Derive the DEM$(T) = n_e^2 \, dh/dT$ for this model. (c) At what temperature is the DEM largest, and why?

**Problem 5: Moreton Wave Speed**

A Moreton wave is observed to travel 400 Mm across the solar surface in 5.5 minutes. (a) Calculate the propagation speed. (b) If the coronal sound speed is 180 km/s at $T = 1.5$ MK, what Alfven speed is required to produce a fast-mode speed equal to the Moreton wave speed? (c) For a coronal electron density of $n_e = 5 \times 10^8$ cm$^{-3}$, what magnetic field strength does this Alfven speed imply?

---

**Previous**: [Photosphere](./04_Photosphere.md) | **Next**: [Corona](./06_Corona.md)

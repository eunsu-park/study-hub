# Magnetosphere Structure

## Learning Objectives

- Describe Earth's intrinsic magnetic field as a tilted dipole and understand its limitations
- Derive the magnetopause standoff distance from pressure balance and apply empirical models
- Explain the physics of the bow shock and magnetosheath, including shock type classification
- Characterize the magnetotail structure including lobes, plasma sheet, and neutral lines
- Describe the plasmasphere, its formation, erosion during storms, and detection methods
- Identify the magnetic cusp regions and their role in magnetosheath plasma access
- Define L-shells and magnetic coordinate systems used in magnetospheric physics

---

## 1. Earth's Intrinsic Magnetic Field

### 1.1 The Dipole Approximation

To first order, Earth's magnetic field can be approximated as a magnetic dipole. This approximation, while imperfect, captures the essential structure that organizes trapped particle motion, current systems, and the overall magnetospheric topology.

The magnetic field of a dipole with moment $\mathbf{M} = M \hat{z}$ (aligned with the magnetic axis) is:

$$\mathbf{B}(r, \theta) = \frac{\mu_0 M}{4\pi r^3} \left( 2\cos\theta \, \hat{r} + \sin\theta \, \hat{\theta} \right)$$

where $r$ is the radial distance from Earth's center and $\theta$ is the magnetic colatitude (angle from the magnetic pole). The field magnitude is:

$$B(r, \theta) = \frac{\mu_0 M}{4\pi r^3} \sqrt{1 + 3\cos^2\theta}$$

**Key parameters:**
- Magnetic dipole moment: $M = 8.0 \times 10^{22}$ A$\cdot$m$^2$ (equivalently $\sim 30.4$ $\mu$T $\cdot R_E^3$)
- Equatorial surface field ($\theta = 90°$, $r = R_E$): $B_0 \approx 31{,}000$ nT $= 0.31$ Gauss
- Polar surface field ($\theta = 0°$, $r = R_E$): $B_{pole} = 2B_0 \approx 62{,}000$ nT

The dipole axis is tilted approximately $11°$ from Earth's rotation axis, and offset by $\sim$500 km from Earth's center. This tilt and offset have important consequences: the magnetic equator does not coincide with the geographic equator, and the field strength varies with geographic longitude even at constant geographic latitude.

### 1.2 Field Line Equation

The equation of a dipole field line is remarkably simple. A field line that crosses the magnetic equator at distance $r = r_0$ follows:

$$r = r_0 \cos^2\lambda$$

where $\lambda$ is the magnetic latitude ($\lambda = 90° - \theta$). This equation tells us that a field line starting at the equator at $r_0 = 5 R_E$ reaches the surface (where $r = R_E$) at magnetic latitude:

$$\cos^2\lambda_{foot} = \frac{R_E}{r_0} = \frac{1}{5} \implies \lambda_{foot} \approx 63.4°$$

This is why aurora (which occurs on field lines mapping to the outer magnetosphere) is confined to high latitudes during quiet times, forming an "auroral oval" at $\sim$65–70$°$ magnetic latitude.

### 1.3 Limitations of the Dipole Model

The dipole approximation breaks down in several important ways:

- **Higher-order multipoles** — Earth's actual field includes quadrupole, octupole, and higher terms. The International Geomagnetic Reference Field (IGRF) uses spherical harmonic coefficients up to degree and order 13 to represent the internal field accurately.
- **South Atlantic Anomaly (SAA)** — A region over South America and the South Atlantic where the field is anomalously weak ($\sim$22,000 nT at the surface vs. $\sim$31,000 nT globally average). This weakness allows trapped radiation belt particles to dip to lower altitudes, creating enhanced radiation exposure for low-Earth orbit satellites and aircraft.
- **Secular variation** — The dipole moment is decreasing at approximately 5% per century. Over the past 2000 years, the moment has decreased by $\sim$30%. If this trend continued (which is not guaranteed), the dipole field would reach zero in $\sim$1600 years, though this does not mean the field would actually vanish — non-dipole components would persist.
- **External distortion** — Solar wind interaction compresses the dayside and stretches the nightside, producing a magnetosphere very different from a pure dipole (the subject of the rest of this lesson).

### 1.4 Physical Intuition

Think of Earth's magnetic field as a giant bar magnet buried at the center of the planet (tilted 11$°$ from the spin axis). The field lines emerge from the southern hemisphere (near the geographic south pole, which is actually the magnetic north pole — a confusing convention) and re-enter near the geographic north pole. This "bubble" of magnetic influence extends outward until it encounters the solar wind, creating the magnetosphere.

---

## 2. Magnetopause: Solar Wind Confinement

### 2.1 Pressure Balance

The magnetopause is the boundary where the outward magnetic pressure of Earth's field balances the inward dynamic pressure of the solar wind. This is the fundamental equation defining the magnetosphere's size.

At the subsolar point (the point on the magnetopause directly facing the Sun), the balance is:

$$\frac{1}{2} \rho_{sw} v_{sw}^2 = \frac{B_{mp}^2}{2\mu_0}$$

where $\rho_{sw}$ and $v_{sw}$ are the solar wind mass density and speed, and $B_{mp}$ is the magnetic field just inside the magnetopause.

For a dipole field, $B_{mp}$ at distance $r$ from Earth in the equatorial plane is:

$$B_{mp} = \frac{\mu_0 M}{4\pi r^3} \times f$$

where $f \approx 2$ accounts for the compression of field lines against the magnetopause (the Chapman-Ferraro surface current effectively doubles the field inside). Substituting:

$$\frac{1}{2} \rho_{sw} v_{sw}^2 = \frac{1}{2\mu_0}\left(\frac{2\mu_0 M}{4\pi R_{mp}^3}\right)^2$$

Solving for the standoff distance:

$$R_{mp} = \left(\frac{\mu_0 M^2}{\pi^2 \rho_{sw} v_{sw}^2}\right)^{1/6} \approx \left(\frac{B_0^2 R_E^6}{\mu_0 \rho_{sw} v_{sw}^2}\right)^{1/6}$$

### 2.2 Numerical Estimate

For typical solar wind conditions ($n = 5$ cm$^{-3}$, $v = 400$ km/s):

- Dynamic pressure: $P_{dyn} = \frac{1}{2} n m_p v^2 \approx \frac{1}{2}(5 \times 10^6)(1.67 \times 10^{-27})(4 \times 10^5)^2 \approx 1.3 \times 10^{-9}$ Pa $\approx 1.3$ nPa
- Standoff distance: $R_{mp} \approx 10 \, R_E$

The $1/6$ power dependence means the magnetopause position is remarkably insensitive to solar wind conditions. Doubling the dynamic pressure only moves the magnetopause inward by a factor of $2^{1/6} \approx 1.12$, or about 12%. This insensitivity arises because the dipole field strengthens rapidly ($\propto r^{-3}$) as the magnetopause is pushed closer to Earth.

### 2.3 Shue et al. (1998) Empirical Model

The most widely used empirical magnetopause model parameterizes the shape as:

$$r = R_0 \left(\frac{2}{1 + \cos\theta}\right)^{\alpha}$$

where $\theta$ is the angle from the Sun-Earth line, and:

$$R_0 = \left(11.4 + 0.013 B_z\right) D_p^{-1/6.6} \quad [R_E]$$
$$\alpha = \left(0.58 - 0.010 B_z\right)\left(1 + 0.010 D_p\right)$$

Here $D_p$ is the solar wind dynamic pressure in nPa and $B_z$ is the IMF north-south component in nT. The model captures two important physical effects:

- **Southward IMF ($B_z < 0$)** — Reconnection erodes magnetic flux, reducing $R_0$ (magnetopause moves closer to Earth)
- **High dynamic pressure** — Compresses the magnetopause (smaller $R_0$)

During extreme storms, $R_{mp}$ can compress to 6–7 $R_E$ or even less, bringing the magnetopause inside geostationary orbit (6.6 $R_E$). This exposes geostationary satellites to direct solar wind plasma.

### 2.4 Chapman-Ferraro Current

The magnetopause is not a passive boundary but carries an intense surface current — the Chapman-Ferraro current. This current flows from dawn to dusk on the dayside magnetopause, producing a magnetic field that:

- Adds to Earth's field inside the magnetopause (enhancing it)
- Cancels Earth's field outside the magnetopause (shielding it)

The current density is approximately:

$$K \sim \frac{\Delta B}{\mu_0} \sim \frac{2B_{mp}}{\mu_0}$$

For $B_{mp} \sim 50$ nT at the subsolar point: $K \sim 80$ mA/m. This current closes over the magnetopause surface, flowing from dusk to dawn over the polar regions.

---

## 3. Bow Shock

### 3.1 Formation and Location

The solar wind flows at super-magnetosonic speeds (faster than both the Alfven and sound speeds in the solar wind). When this supersonic flow encounters the magnetopause obstacle, a standing shock wave — the **bow shock** — forms upstream, analogous to the bow wave of a ship or the shock wave ahead of a supersonic aircraft.

The bow shock stands approximately 2–3 $R_E$ upstream of the subsolar magnetopause, at roughly 13 $R_E$ from Earth. Its shape roughly follows the magnetopause shape but is more blunt (larger flaring angle).

### 3.2 Shock Properties

The bow shock is primarily a fast-mode MHD shock. Its properties are governed by the Rankine-Hugoniot relations:

**Compression ratio** — For a strong hydrodynamic shock with ratio of specific heats $\gamma = 5/3$:

$$\frac{\rho_2}{\rho_1} = \frac{(\gamma + 1)M^2}{(\gamma - 1)M^2 + 2} \xrightarrow{M \gg 1} \frac{\gamma + 1}{\gamma - 1} = 4$$

The solar wind is typically compressed by a factor of $\sim$4 (density increases fourfold), heated substantially, and decelerated.

**Mach number** — The magnetosonic Mach number of the solar wind is:

$$M_{ms} = \frac{v_{sw}}{v_{ms}} = \frac{v_{sw}}{\sqrt{v_A^2 + c_s^2}}$$

Typically $M_{ms} \approx 6$–8, making Earth's bow shock a strong shock.

### 3.3 Quasi-parallel vs. Quasi-perpendicular Shock

The shock character depends on the angle $\theta_{Bn}$ between the upstream magnetic field and the shock normal:

- **Quasi-perpendicular** ($\theta_{Bn} > 45°$): Sharp, well-defined transition. Particles reflect specularly and are quickly swept downstream. Occurs on the dusk flank (typical Parker spiral geometry).
- **Quasi-parallel** ($\theta_{Bn} < 45°$): Diffuse, broad transition. Reflected particles escape upstream along field lines, generating an extended turbulent **foreshock** region. ULF (ultra-low frequency) waves, hot flow anomalies, and energized particle beams populate the foreshock. Occurs on the dawn flank.

The difference between these shock types has important implications for particle acceleration, wave generation, and downstream turbulence.

---

## 4. Magnetosheath

### 4.1 Properties

The magnetosheath is the region between the bow shock and the magnetopause — it contains solar wind plasma that has been shocked: decelerated, heated, and compressed. Typical magnetosheath properties:

| Parameter | Solar Wind | Magnetosheath |
|-----------|-----------|---------------|
| Speed | 400 km/s | 100–200 km/s |
| Density | 5 cm$^{-3}$ | 15–25 cm$^{-3}$ |
| Temperature | $10^5$ K | $10^6$ K |
| Magnetic field | 5 nT | 15–25 nT |
| $\beta = P/(B^2/2\mu_0)$ | $\sim$1 | 1–10 |

### 4.2 Turbulence and Wave Activity

The magnetosheath is highly turbulent, hosting several characteristic wave modes:

- **Mirror mode waves** — Compressive structures ($\delta B/B \sim 0.5$) generated by temperature anisotropy ($T_\perp > T_\parallel$), primarily in the quasi-perpendicular magnetosheath. Appear as "dips" or "peaks" in the magnetic field magnitude.
- **Lion roars** — Narrowband electromagnetic whistler-mode waves at $\sim$100 Hz, observed inside mirror mode troughs. Named for their characteristic sound when played through a speaker.
- **Alfven/ion cyclotron waves** — Generated by ion temperature anisotropy, more common in the quasi-parallel magnetosheath.

### 4.3 Flux Transfer Events (FTEs)

At the magnetopause, reconnection does not always proceed steadily. Instead, it often occurs in bursts, producing **Flux Transfer Events** — cylindrical magnetic flux tubes connecting the magnetosheath to the magnetosphere. FTE characteristics:

- Bipolar signature in the magnetopause normal component of $B$
- Diameter: $\sim$1 $R_E$
- Spacing: $\sim$8 minutes (quasi-periodic)
- Magnetic flux per FTE: $\sim$10$^{21}$ Mx ($10^{13}$ Wb)
- Move poleward away from the subsolar point after formation

FTEs are important because they represent the "unit" of reconnected flux and contribute to the total rate of magnetic flux transport from the dayside to the nightside.

---

## 5. Magnetotail

### 5.1 Overall Structure

The magnetotail extends more than 100 $R_E$ antisunward (possibly $>$1000 $R_E$ in some models), formed by the solar wind stretching open magnetic field lines that have been reconnected on the dayside. The tail has a remarkably organized structure:

**North Lobe** — Magnetic field directed earthward (toward Earth). Very low plasma density ($\sim$0.01 cm$^{-3}$), low $\beta$ ($\ll$ 1). Essentially vacuum-like, magnetically dominated.

**South Lobe** — Magnetic field directed tailward (away from Earth). Same low-density, low-$\beta$ character as the north lobe.

**Plasma Sheet** — A layer of hot ($T \sim$ 1–10 keV), relatively dense ($n \sim$ 0.1–1 cm$^{-3}$) plasma lying between the two lobes. The $\beta$ is typically $\sim$1–10. This is where most magnetotail dynamics occur — reconnection, particle acceleration, substorm activity.

**Cross-Tail Current Sheet** — A thin current sheet separating the north and south lobes, flowing from dawn to dusk. This current supports the reversal of $B_x$ across the midplane. Normal thickness: 1–5 $R_E$, but thins to $\sim$0.1 $R_E$ ($\sim$600 km) just before substorm onset.

### 5.2 Neutral Lines

**Near-Earth Neutral Line (NENL)** — Located at $\sim$20–30 $R_E$ downtail. This is where reconnection occurs during substorms, creating earthward-directed fast flows (bursty bulk flows, BBFs) and tailward-ejected plasmoids. The NENL is the engine of substorm dynamics.

**Distant Neutral Line (DNL)** — Located at $>$100 $R_E$. Where the last closed field lines of the magnetotail reconnect during quasi-steady conditions, maintaining the tail's equilibrium length.

### 5.3 Physical Intuition

Imagine the magnetotail as a stretched rubber band. The dayside reconnection "loads" the tail by transferring magnetic flux from the dayside to the nightside (adding tension to the rubber band). When the tail becomes too stretched and thin, reconnection at the NENL "unloads" the stored energy — this is the substorm expansion phase. The released energy drives fast plasma flows, particle injections, and aurora.

---

## 6. Plasmasphere

### 6.1 Formation and Structure

The plasmasphere is a torus-shaped region of cold ($<$ 1 eV $\sim$ 10$^4$ K), dense ($10^2$–$10^4$ cm$^{-3}$) plasma that corotates with Earth on closed dipole field lines. It is essentially an upward extension of the ionosphere:

- **Source**: Ionospheric O$^+$ and H$^+$ ions flow upward along field lines during the day, filling flux tubes. At night, plasma drains back to the ionosphere.
- **Equilibrium**: After several days of quiet conditions, an equilibrium density profile is established where the upward ionospheric flux balances loss processes.
- **Boundary**: The **plasmapause** — a sharp ($\sim$factor of 10–100) density gradient typically at $L \sim$ 4–5 during quiet times.

### 6.2 Plasmapause Formation

The plasmapause location is determined by the competition between corotation and convection electric fields:

- **Corotation electric field**: $\mathbf{E}_{cor} = -(\boldsymbol{\Omega} \times \mathbf{r}) \times \mathbf{B}$, directed radially inward. Drives $\mathbf{E} \times \mathbf{B}$ drift that keeps plasma corotating with Earth.
- **Convection electric field**: $\mathbf{E}_{conv}$, directed dawn-to-dusk, driven by the solar wind interaction. Drives sunward convection.

Where these two electric fields balance, the last closed drift path (LCDP) defines the plasmapause. Inside the LCDP, plasma circulates on closed orbits around Earth (corotation dominates). Outside, plasma is swept to the dayside magnetopause by convection.

During geomagnetic storms, the convection electric field intensifies, the LCDP contracts, and the plasmapause erodes inward — sometimes to $L \sim$ 2–3 during severe storms.

### 6.3 Erosion and Refilling

The plasmasphere is a dynamic structure:

- **Erosion** — During storms, the enhanced convection electric field strips plasma from the outer plasmasphere, forming **drainage plumes** — tongues of dense plasma extending from the dusk side toward the dayside magnetopause. These plumes have been dramatically imaged by the IMAGE/EUV instrument.
- **Refilling** — After storm activity subsides, the depleted flux tubes slowly refill from the ionosphere. Refilling timescales: $\sim$1 day for $L = 2$, $\sim$3–5 days for $L = 4$, $\sim$weeks for $L > 5$. The refilling rate depends on the ionospheric plasma density and the flux tube volume ($\propto L^4$ for a dipole).

### 6.4 Detection Methods

- **Whistler dispersion**: Lightning-generated whistler waves are dispersed by the plasmaspheric plasma (lower frequencies travel slower). The dispersion measures the total electron content along the path, constraining plasmaspheric density.
- **EUV imaging**: The IMAGE satellite's Extreme Ultraviolet (EUV) instrument imaged the entire plasmasphere in 30.4 nm He$^+$ emission, revealing drainage plumes, notches, and other structures for the first time.
- **In-situ density measurements**: Spacecraft (CRRES, Van Allen Probes) directly measure electron density from upper hybrid resonance frequency or spacecraft potential.

---

## 7. Cusps

### 7.1 Structure and Location

The magnetic cusps are funnel-shaped regions near the magnetic poles where the magnetic field topology allows direct access of magnetosheath plasma to the ionosphere. They are magnetic null regions (or near-null) where field lines from the dayside magnetopause converge toward the poles.

**Location**: The cusps are centered near $\sim$75–80$°$ magnetic latitude at local noon during quiet times. During southward IMF, the cusp shifts equatorward (to $\sim$70–75$°$) and widens due to enhanced dayside reconnection opening more field lines.

### 7.2 Cusp Phenomena

- **Cusp precipitation** — Magnetosheath ions and electrons stream directly into the cusp along open field lines, producing soft ($<$1 keV) particle precipitation and associated "cusp aurora" — diffuse, red/green aurora at midday high latitudes.
- **Diamagnetic cavities** — The inflowing magnetosheath plasma creates regions of enhanced plasma pressure and depressed magnetic field strength within the cusp.
- **Ion dispersion** — Ions of different energies precipitate at different latitudes due to the convection of open field lines over the polar cap while particles travel along the field lines. Higher energy ions (faster parallel velocity) precipitate at lower latitudes; lower energy ions precipitate at higher latitudes. This produces a latitude-energy dispersion signature that serves as a diagnostic of the reconnection rate.
- **Cusp as a particle entry point** — The cusp is the primary pathway for solar wind plasma to enter the magnetosphere directly. Some cusp ions become trapped in the outer magnetosphere, contributing to the warm plasma cloak.

---

## 8. Magnetic Coordinates and L-Shells

### 8.1 McIlwain L-Parameter

The most widely used coordinate for organizing magnetospheric phenomena is the McIlwain L-parameter (McIlwain, 1961). For a pure dipole field:

$$L = \frac{r_{eq}}{R_E}$$

where $r_{eq}$ is the equatorial crossing distance of a field line, measured in Earth radii. Equivalently, using the field line equation $r = L R_E \cos^2\lambda$:

$$L = \frac{r}{R_E \cos^2\lambda}$$

**Physical meaning**: $L$ labels a magnetic field line by where it crosses the magnetic equator. All points along a given field line share the same $L$-value (in a dipole).

### 8.2 Important L-Values

| L-value | Location/Feature |
|---------|-----------------|
| 1.0 | Earth's surface (equator) |
| 1.5–2.5 | Inner radiation belt (trapped protons) |
| 2.0–3.0 | Slot region (gap between radiation belts) |
| 3.0–7.0 | Outer radiation belt (trapped electrons) |
| 3.0–8.0 | Ring current region |
| $\sim$4–5 | Plasmapause (quiet time) |
| 6.6 | Geostationary orbit |
| $\sim$8–10 | Magnetopause (subsolar, quiet) |
| $\sim$10–12 | Auroral oval field line footprints |

### 8.3 Invariant Latitude

The invariant latitude $\Lambda$ is the magnetic latitude at which a given $L$-shell field line intersects Earth's surface:

$$\cos^2\Lambda = \frac{1}{L}$$

For example:
- $L = 4$: $\Lambda = 60°$
- $L = 6.6$ (GEO): $\Lambda = 67.1°$
- $L = 10$: $\Lambda = 71.6°$

Invariant latitude is used to map magnetospheric regions to ground locations — for example, the auroral oval maps to $\Lambda \sim$ 65–75$°$.

### 8.4 Beyond the Dipole: IGRF-Based Coordinates

For accurate work, especially at low altitudes where non-dipole components are significant, the actual geomagnetic field must be used:

- **$L^*$ (Roederer's L-star)**: Defined through the third adiabatic invariant $\Phi$ (magnetic flux enclosed by a drift shell). $L^*$ is constant along a particle drift orbit even in a non-dipole field.
- **Corrected geomagnetic coordinates**: Use IGRF field to trace field lines and define coordinate systems.
- **AACGM (Altitude-Adjusted Corrected Geomagnetic)**: Commonly used for ionospheric and auroral studies, accounts for the non-dipolar field.

The distinction between $L$ (dipole-based) and $L^*$ (invariant-based) becomes important for radiation belt studies where accurate drift shell identification is critical.

### 8.5 Magnetic Local Time (MLT)

Magnetic local time is the azimuthal coordinate in the magnetosphere, analogous to geographic local time but referenced to the magnetic dipole:

- MLT = 12 (noon): subsolar direction
- MLT = 0 (midnight): antisolar direction (tail)
- MLT = 6 (dawn): morning side
- MLT = 18 (dusk): evening side

MLT organizes many magnetospheric phenomena: the auroral oval, plasmaspheric drainage plumes (dusk), substorm injection (midnight to dawn), etc.

---

## 9. The Global Magnetospheric Picture

Putting it all together, the magnetosphere is a complex but organized system:

1. The **solar wind** flows supersonically toward Earth, creating a **bow shock** at $\sim$13 $R_E$.
2. Shocked solar wind fills the **magnetosheath** between the bow shock and **magnetopause** ($\sim$10 $R_E$).
3. The **Chapman-Ferraro current** at the magnetopause confines Earth's field.
4. On the dayside, the field is compressed; on the nightside, it is stretched into the **magnetotail** ($>$100 $R_E$).
5. The **plasma sheet** separates the two tail lobes, carrying the cross-tail current.
6. Inside, the cold, dense **plasmasphere** corotates on closed field lines out to $L \sim$ 4–5.
7. The **radiation belts** contain trapped energetic particles at $L \sim$ 1.5–7.
8. **Cusps** at high latitudes allow direct magnetosheath access.
9. **Birkeland currents** connect the magnetosphere to the ionosphere at auroral latitudes.

This structure is not static — it breathes, expands, contracts, and reconfigures in response to the ever-changing solar wind, creating the dynamic phenomena we call space weather.

---

## Practice Problems

1. **Dipole field calculation** — Calculate the magnetic field magnitude at: (a) the magnetic equator at the surface ($r = R_E$, $\theta = 90°$), (b) the north magnetic pole ($r = R_E$, $\theta = 0°$), and (c) the equatorial plane at geostationary orbit ($r = 6.6 R_E$, $\theta = 90°$). Use $M = 8.0 \times 10^{22}$ A$\cdot$m$^2$.

2. **Magnetopause standoff** — During the Halloween 2003 storm, the solar wind density reached $n = 50$ cm$^{-3}$ with speed $v = 700$ km/s. (a) Calculate the dynamic pressure. (b) Estimate the magnetopause standoff distance. (c) Was geostationary orbit ($r = 6.6 R_E$) exposed to direct solar wind? Compare with the Shue et al. model prediction using $B_z = -30$ nT.

3. **Plasmapause location** — The corotation electric field potential is $\Phi_{cor} = -\Omega_E B_0 R_E^3 / r$ (in the equatorial plane), and a simple dawn-dusk convection potential is $\Phi_{conv} = -E_0 r \sin\phi$ (where $\phi$ is azimuthal angle, 0 at noon). Find the plasmapause location on the dusk side ($\phi = -\pi/2$) for $E_0 = 0.3$ mV/m (quiet) and $E_0 = 1.5$ mV/m (storm). Use $\Omega_E = 7.27 \times 10^{-5}$ rad/s, $B_0 = 31{,}000$ nT.

4. **Invariant latitude mapping** — A satellite at geostationary orbit ($L = 6.6$) observes a particle injection event. (a) At what invariant latitude would the associated aurora appear on the ground? (b) At what geographic latitude would this occur on the nightside, accounting for the $11°$ dipole tilt? (c) During a storm, the injection is observed at $L = 4$. Where does the aurora shift to?

5. **Bow shock Mach number** — For solar wind conditions $v = 450$ km/s, $n = 8$ cm$^{-3}$, $B = 7$ nT, $T_p = 10^5$ K: (a) Calculate the Alfven speed $v_A = B/\sqrt{\mu_0 n m_p}$. (b) Calculate the sound speed $c_s = \sqrt{\gamma k_B T / m_p}$ for $\gamma = 5/3$. (c) Calculate the magnetosonic speed $v_{ms} = \sqrt{v_A^2 + c_s^2}$. (d) Determine the magnetosonic Mach number. Is this a strong shock?

---

**Previous**: [Introduction to Space Weather](./01_Introduction_to_Space_Weather.md) | **Next**: [Magnetospheric Current Systems](./03_Magnetospheric_Current_Systems.md)

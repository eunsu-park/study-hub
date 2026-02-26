# Solar Wind

## Learning Objectives

- Distinguish the properties and source regions of fast and slow solar wind
- Derive the Parker spiral geometry from the frozen-in field condition and solar rotation
- Explain the formation and structure of stream interaction regions (SIRs/CIRs)
- Understand the nature and spectral characteristics of solar wind turbulence
- Describe the solar wind acceleration problem and wave-driven solutions
- Analyze the heliospheric current sheet structure and its solar cycle variation

---

## 1. Solar Wind Properties

The solar wind is a continuous, supersonic outflow of plasma from the solar corona that fills the entire heliosphere. Its existence was predicted theoretically by Eugene Parker in 1958 and confirmed observationally by Mariner 2 in 1962.

### 1.1 Basic Parameters at 1 AU

At Earth's orbit ($r = 1$ AU $\approx 215 R_\odot$), the average solar wind properties are:

| Parameter | Typical Value |
|-----------|---------------|
| Number density $n$ | ~5 cm$^{-3}$ |
| Bulk speed $v$ | ~400 km/s |
| Proton temperature $T_p$ | ~$5 \times 10^4$ K |
| Electron temperature $T_e$ | ~$1.5 \times 10^5$ K |
| Magnetic field $B$ | ~5 nT |
| Plasma $\beta$ | ~1 |
| Alfven speed $v_A$ | ~50 km/s |
| Alfven Mach number $M_A$ | ~8 |
| Mass flux | ~$3 \times 10^8$ cm$^{-2}$ s$^{-1}$ |

The proton temperature is lower than the electron temperature, and both are far below what adiabatic expansion would predict — indicating continued heating of the solar wind as it propagates outward.

### 1.2 Two Types of Solar Wind

One of the most fundamental observations is that the solar wind comes in two distinct varieties:

**Fast solar wind** (~700–800 km/s):
- Source: coronal holes (open magnetic field regions)
- Density: relatively low (~3 cm$^{-3}$ at 1 AU)
- Variability: steady, uniform
- Composition: photospheric-like (low First Ionization Potential enhancement)
- Charge states: freeze-in temperature ~1–1.5 MK (e.g., O$^{6+}$/O$^{7+}$ ratio indicates low coronal temperature)
- Alfvenicity: highly Alfvenic (strong correlations between $\delta v$ and $\delta B$)

**Slow solar wind** (~300–400 km/s):
- Source: complex — streamer belt edges, active region periphery, or interchange reconnection
- Density: relatively high (~8–10 cm$^{-3}$ at 1 AU)
- Variability: highly variable, structured
- Composition: coronal-like (enhanced First Ionization Potential elements by factor ~3–4)
- Charge states: freeze-in temperature ~1.5–2 MK
- Alfvenicity: low (poor $\delta v$–$\delta B$ correlation)

### 1.3 Mass and Energy Loss

The total solar wind mass loss rate is:

$$\dot{M} = 4\pi r^2 n m_p v \approx 2 \times 10^{-14} \, M_\odot/\text{yr}$$

Over the Sun's remaining main-sequence lifetime (~5 Gyr), this amounts to only $\sim 10^{-4} M_\odot$ — negligible for stellar evolution but profoundly important for shaping the heliospheric environment.

The total kinetic energy flux of the solar wind:

$$L_{\text{KE}} = \frac{1}{2}\dot{M}v^2 \approx 10^{27} \text{ erg/s}$$

This is about $3 \times 10^{-7}$ of the solar luminosity — tiny in energy terms but carrying the magnetic field and particles that define the heliosphere.

---

## 2. Fast and Slow Wind: Source Regions

Understanding where and how the two types of solar wind originate remains one of the central questions in heliophysics.

### 2.1 Fast Wind from Coronal Holes

Coronal holes are extended regions of predominantly open (monopolar) magnetic field where the coronal plasma has unimpeded access to the heliosphere. They appear dark in EUV and soft X-ray images because the plasma density and temperature are lower than in the surrounding closed-field corona.

During solar minimum, large coronal holes dominate the polar regions, and the fast wind fills most of the heliosphere at high latitudes. Ulysses spacecraft, which flew over both solar poles, measured a bimodal solar wind: fast wind ($\sim 750$ km/s) at latitudes above $\sim 20°$, and slow, variable wind in the equatorial streamer belt region.

The fast wind has near-photospheric composition (low FIP bias), suggesting it originates from plasma that has spent minimal time in closed magnetic structures. The relatively low freeze-in charge states indicate that the source region has a modest coronal temperature (~1–1.5 MK), consistent with the dim appearance of coronal holes.

### 2.2 Slow Wind Origin: An Ongoing Debate

The slow wind origin is considerably more complex and remains actively debated. Several scenarios are proposed:

**Streamer edge model**: Slow wind emerges from the boundaries of helmet streamers, where closed field lines are slowly opened by the expanding corona. The plasma has been trapped in closed structures long enough to develop coronal composition (enhanced FIP elements).

**Active region expansion**: Open field lines rooted near active regions expand into the heliosphere. The proximity to hot, dense active region plasma explains the coronal composition.

**Interchange reconnection** (Fisk model): Open magnetic flux from coronal holes reconnects with closed loops at the coronal hole boundary. This opens previously closed field lines, releasing trapped coronal plasma as slow wind. This mechanism naturally explains the coronal composition and variability.

**S-web model** (Antiochos et al.): The "Separatrix-web" — a complex network of separatrices and quasi-separatrix layers (QSLs) in the coronal magnetic field — defines corridors along which slow wind can escape. Reconnection along these corridors releases plasma intermittently.

Parker Solar Probe observations in the inner heliosphere are providing new constraints on these models, with evidence for complex, structured slow wind sources.

### 2.3 Compositional Diagnostics

The composition of the solar wind provides a powerful diagnostic of its origin:

- **FIP (First Ionization Potential) effect**: Elements with low FIP ($<10$ eV: Fe, Mg, Si) are enhanced relative to high-FIP elements (O, Ne, Ar) in the slow wind by a factor of 3–4 compared to photospheric abundances. The fast wind has near-photospheric composition.

- **Charge state ratios**: The ionic charge states "freeze in" at the coronal height where the solar wind expansion timescale becomes shorter than the ionization/recombination timescale. Common diagnostics include O$^{7+}$/O$^{6+}$ and C$^{6+}$/C$^{5+}$, which indicate the electron temperature at the freeze-in point.

---

## 3. Parker Spiral

The large-scale structure of the interplanetary magnetic field (IMF) is one of the beautiful results in heliophysics — a direct consequence of the Sun's rotation combined with the radially flowing solar wind and the frozen-in flux condition.

### 3.1 The Physical Picture

Imagine a garden sprinkler: water shoots out radially, but because the sprinkler rotates, the water streams form spiral patterns on the lawn. The solar magnetic field behaves analogously — the field is frozen into the radially flowing solar wind, but the footpoints of the field lines rotate with the Sun (sidereal rotation period $P_\odot \approx 25.4$ days at the equator, angular velocity $\Omega_\odot = 2\pi/P_\odot \approx 2.87 \times 10^{-6}$ rad/s).

### 3.2 Derivation of the Spiral Geometry

In the rotating frame of the Sun, a solar wind parcel launched radially at time $t_0$ from a colatitude $\theta$ has position:

$$r(t) = r_0 + v_{\text{sw}}(t - t_0)$$

$$\phi(t) = \phi_0 - \Omega_\odot(t - t_0)$$

Eliminating time:

$$r - r_0 = -\frac{v_{\text{sw}}}{\Omega_\odot}(\phi - \phi_0)$$

This is the equation of an Archimedean spiral. The field line follows this spiral because of the frozen-in condition: the magnetic field is "frozen" into the plasma, so the field line connects plasma parcels that left the Sun at the same longitude but at different times.

### 3.3 Magnetic Field Components

From the frozen-in condition and flux conservation, the magnetic field components in spherical coordinates are:

$$B_r = B_0\left(\frac{r_0}{r}\right)^2$$

$$B_\phi = -B_r \frac{\Omega_\odot r \sin\theta}{v_{\text{sw}}}$$

The radial component decreases as $1/r^2$ (flux conservation through expanding spherical shells). The azimuthal component introduces the $1/r$ dependence because $B_\phi \propto B_r \times r \propto r^{-1}$.

The total field magnitude at large $r$ is dominated by $B_\phi$:

$$|B| \approx |B_\phi| = B_0\left(\frac{r_0}{r}\right)\frac{\Omega_\odot r_0\sin\theta}{v_{\text{sw}}} \propto \frac{1}{r}$$

### 3.4 The Garden-Hose Angle

The angle between the magnetic field and the radial direction — the garden-hose angle or spiral angle — is:

$$\tan\psi = \frac{|B_\phi|}{|B_r|} = \frac{\Omega_\odot r \sin\theta}{v_{\text{sw}}}$$

At 1 AU with $v_{\text{sw}} = 400$ km/s:

$$\tan\psi = \frac{2.87 \times 10^{-6} \times 1.5 \times 10^{11}}{4 \times 10^5} \approx 1.08$$

$$\psi \approx 47°$$

So at Earth's orbit, the magnetic field makes roughly a $45°$ angle with the Sun-Earth line for the typical slow solar wind. For fast wind ($v_{\text{sw}} = 750$ km/s), $\psi \approx 28°$ — more radial because the wind carries the field line further before the Sun rotates significantly.

### 3.5 Observational Verification

The Parker spiral has been thoroughly confirmed by decades of in-situ measurements. The distribution of the IMF direction at 1 AU shows two preferred orientations — away from the Sun in one spiral direction and toward the Sun in the opposite spiral direction — forming the two magnetic sectors (see Section 7 on the heliospheric current sheet).

---

## 4. Stream Interaction Regions (SIRs)

When fast solar wind from a coronal hole follows slow wind from the streamer belt (or vice versa), the resulting interaction creates large-scale structures in the heliosphere.

### 4.1 Formation Mechanism

Consider a rotating Sun with a coronal hole at one longitude emitting fast wind ($v_f \sim 750$ km/s) and a streamer at another longitude emitting slow wind ($v_s \sim 350$ km/s). As the Sun rotates, the fast wind is emitted behind the slow wind (in the sense of rotation). Because the fast wind catches up to the slow wind, a compression region forms at the interface.

The interaction cannot be resolved by simple overtaking because the magnetic field prevents interpenetration of the two streams. Instead:
- The fast wind is decelerated and deflected
- The slow wind is accelerated and compressed
- A pair of pressure waves forms at the boundaries

### 4.2 CIR Structure

If the coronal hole persists for multiple solar rotations, the interaction region recurs every ~27 days (as seen from Earth) — hence Corotating Interaction Region (CIR). The mature CIR structure consists of:

1. **Forward shock**: Propagating outward into the slow wind (in the rest frame of the Sun, propagating in the direction of solar rotation). Compresses and heats the slow wind.
2. **Stream interface**: The boundary between slow wind that has been compressed from behind and fast wind that has been compressed from ahead. Marked by a peak in total pressure and a shear in the flow direction.
3. **Reverse shock**: Propagating backward (in the solar wind frame) into the fast wind. Decelerates and heats the fast wind.

At 1 AU, the shocks are often not fully formed — they are still steepening pressure waves. Fully developed forward-reverse shock pairs are typically established by 2–3 AU, which is why Pioneer and Voyager observations in the outer heliosphere showed pronounced CIR signatures.

### 4.3 Geo-Effective Properties

CIRs produce moderate, recurrent geomagnetic activity:
- **Recurrence**: ~27-day periodicity, most prominent during the declining phase of the solar cycle when large low-latitude coronal holes are common
- **Storm intensity**: Typically $Dst \sim -50$ to $-100$ nT (moderate storms)
- **High-speed stream effects**: Prolonged intervals of enhanced solar wind speed drive long-duration magnetospheric convection
- **Radiation belt enhancement**: Sustained fast wind with Alfvenic fluctuations efficiently accelerates radiation belt electrons to relativistic energies ("killer electrons")

### 4.4 Radial Evolution

As CIRs propagate outward:
- Shocks strengthen and fully form by 2–3 AU
- The compression region broadens
- Forward and reverse shocks begin to interact with CIRs from adjacent streams
- By 10–15 AU, CIRs merge into large-scale merged interaction regions (MIRs) that can extend over many AU in the outer heliosphere

---

## 5. Solar Wind Turbulence

The solar wind is one of the best natural laboratories for studying magnetohydrodynamic turbulence, providing high-cadence in-situ measurements impossible in any other astrophysical plasma.

### 5.1 Power Spectrum

The magnetic field fluctuation power spectrum in the solar wind shows several distinct ranges:

$$P(f) \propto f^{-\alpha}$$

- **$1/f$ range** ($10^{-4}$–$10^{-3}$ Hz, corresponding to hours): $\alpha \approx 1$. This range is thought to contain the "energy reservoir" — fluctuations imprinted by the coronal source or generated by stream interactions. The $1/f$ spectrum may reflect a scale-invariant process at the Sun.

- **Inertial range** ($10^{-3}$–$1$ Hz, minutes to seconds): $\alpha \approx 5/3$ (Kolmogorov-like). This is the classic turbulent cascade range where energy is transferred from large to small scales without significant dissipation. The Kolmogorov scaling $\alpha = 5/3$ is observed remarkably consistently, though Iroshnikov-Kraichnan scaling ($\alpha = 3/2$) is also predicted for MHD turbulence.

The distinction between these two exponents is subtle but physically meaningful. The Kolmogorov spectrum assumes local energy transfer independent of the magnetic field, while the IK spectrum accounts for the Alfven effect — the tendency of counter-propagating Alfven wave packets to decorrelate on an Alfven time rather than an eddy turnover time.

- **Dissipation/kinetic range** ($>1$ Hz, sub-proton scales): Steeper spectrum, $\alpha \approx 2.5$–$4$. At the proton gyroscale ($\rho_p = v_{th,p}/\Omega_p \sim 100$ km at 1 AU) or proton inertial length ($d_p = c/\omega_{pp} \sim 100$ km), MHD breaks down and kinetic physics takes over. The turbulent energy is dissipated through various kinetic processes (cyclotron damping, Landau damping, stochastic heating).

### 5.2 Alfvenicity and Cross-Helicity

A fundamental property of solar wind turbulence is its Alfvenicity — the degree of correlation between velocity and magnetic field fluctuations.

Elsasser variables decompose the fluctuations into outward-propagating ($\mathbf{z}^+$) and inward-propagating ($\mathbf{z}^-$) Alfven waves:

$$\mathbf{z}^\pm = \delta\mathbf{v} \mp \frac{\delta\mathbf{B}}{\sqrt{4\pi\rho}}$$

The normalized cross-helicity quantifies the imbalance:

$$\sigma_c = \frac{e^+ - e^-}{e^+ + e^-}$$

where $e^\pm = |\mathbf{z}^\pm|^2/4$. In the fast solar wind, $\sigma_c \approx 0.8$–$1.0$ (predominantly outward-propagating Alfven waves). In the slow wind, $\sigma_c \approx 0$–$0.5$ (more balanced, indicating local turbulence generation or mixing).

### 5.3 Residual Energy

The residual energy measures the imbalance between kinetic and magnetic fluctuation energy:

$$\sigma_r = \frac{e_k - e_m}{e_k + e_m}$$

where $e_k = |\delta\mathbf{v}|^2/2$ and $e_m = |\delta\mathbf{B}|^2/(8\pi\rho)$. Observations typically show $\sigma_r \approx -0.2$ to $-0.5$ — a slight excess of magnetic over kinetic energy. This negative residual energy is a robust feature of solar wind turbulence and is related to the generation of magnetic structures (flux tubes, current sheets) by the turbulent cascade.

### 5.4 Intermittency

Solar wind turbulence is intermittent — the statistical properties become increasingly non-Gaussian at smaller scales. This manifests as:
- Heavy-tailed probability distributions of field increments $\delta B(\tau) = B(t + \tau) - B(t)$
- Enhanced kurtosis at small $\tau$
- Multifractal structure: different moments of the structure functions scale with different exponents
- Physical manifestation: thin current sheets and coherent structures embedded in the turbulent flow

These intermittent structures may play an important role in particle acceleration and heating.

---

## 6. Solar Wind Acceleration

How is the solar wind accelerated to supersonic speeds? This question, first posed and partially answered by Parker (1958), remains incompletely resolved, particularly for the fast wind.

### 6.1 Parker's Isothermal Model

Parker's original model treats the corona as a steady, spherically symmetric, isothermal flow. The momentum equation:

$$\rho v \frac{dv}{dr} = -\frac{dp}{dr} - \frac{GM_\odot\rho}{r^2}$$

combined with the isothermal equation of state $p = nk_BT = \rho c_s^2/m_p$ (where $c_s = \sqrt{k_BT/m_p}$ is the isothermal sound speed), yields the Parker wind equation:

$$\left(v - \frac{c_s^2}{v}\right)\frac{dv}{dr} = \frac{2c_s^2}{r} - \frac{GM_\odot}{r^2}$$

The critical point occurs where the right-hand side vanishes:

$$r_c = \frac{GM_\odot}{2c_s^2}$$

At this radius, the flow speed equals the sound speed ($v = c_s$). For $T = 10^6$ K, $r_c \approx 5.8 R_\odot$.

The transonic solution — the Parker wind — starts subsonic at the base, accelerates through the critical point, and becomes supersonic beyond it. This solution uniquely satisfies the boundary conditions of finite pressure at the base and zero pressure at infinity.

For a detailed derivation and discussion of the Parker wind equation, see MHD Lesson 11.

### 6.2 The Fast Wind Problem

Parker's isothermal model with $T = 1$–$2 \times 10^6$ K produces an asymptotic wind speed of $\sim 300$–$400$ km/s — adequate for the slow wind but far too slow for the fast wind (~750 km/s). Something beyond simple thermal pressure must accelerate the fast wind.

The discrepancy can be quantified through energy considerations. The total energy per unit mass at 1 AU is:

$$\frac{1}{2}v^2 + \frac{5}{2}\frac{k_BT}{m_p} - \frac{GM_\odot}{r_\odot} + \frac{v_A^2}{2} \approx 0$$

For $v = 750$ km/s, the gravitational binding energy is $\sim 2.2 \times 10^{11}$ erg/g, while the kinetic energy is $\sim 2.8 \times 10^{11}$ erg/g. The coronal thermal energy alone provides at most $\sim 1.5 \times 10^{11}$ erg/g for $T = 2$ MK — there is a clear energy deficit.

### 6.3 Wave-Driven Wind Models

The leading candidate for the additional energy source is Alfven waves:

1. **Generation**: Alfven waves are generated by the turbulent convective motions at the photosphere, which shake the magnetic field footpoints.

2. **Propagation**: The waves propagate upward along the open magnetic field lines, carrying an energy flux:

$$F_A = \frac{1}{2}\rho\langle\delta v^2\rangle v_A$$

Typical estimates give $F_A \sim 10^5$–$10^6$ erg cm$^{-2}$ s$^{-1}$ at the coronal base — sufficient to power the fast wind.

3. **Reflection**: As the Alfven speed changes with height (due to density stratification), some of the outward-propagating waves are reflected. The reflected (inward-propagating) waves interact nonlinearly with the outward waves, driving a turbulent cascade.

4. **Dissipation**: The turbulent cascade transfers energy to small scales where it is dissipated as heat, and the wave pressure gradient adds momentum to the flow:

$$\rho v \frac{dv}{dr} = -\frac{dp}{dr} - \frac{GM_\odot\rho}{r^2} - \frac{d}{dr}\left(\frac{\langle\delta B^2\rangle}{8\pi}\right) + \frac{\langle\delta B^2\rangle}{4\pi}\frac{1}{A}\frac{dA}{dr}$$

where $A(r)$ is the flux tube cross-sectional area. The last two terms represent the wave pressure gradient and the mirror force.

Modern wave-driven wind models (e.g., Cranmer et al. 2007, van der Holst et al. 2014) can reproduce both the fast and slow wind speeds with realistic wave amplitudes and coronal boundary conditions.

### 6.4 Parker Solar Probe Revelations

NASA's Parker Solar Probe (PSP), launched in 2018, has provided unprecedented measurements of the near-Sun solar wind (down to ~13 $R_\odot$ as of perihelion 18):

- **Switchbacks**: Rapid, large-amplitude reversals of the radial magnetic field. These are S-shaped bends in the field lines that propagate outward. Their origin is debated — they may be generated by interchange reconnection at the coronal base, by velocity shears in the expanding wind, or by the evolving turbulence itself.

- **Alfvenic spikes**: The switchbacks are highly Alfvenic (strong $\delta v$–$\delta B$ correlation), consistent with large-amplitude Alfven waves.

- **Dust-free zone**: Evidence for a dust depletion zone close to the Sun, as predicted by sublimation of interplanetary dust grains.

- **Sub-Alfvenic intervals**: During some close perihelia, PSP has entered regions where the solar wind speed drops below the local Alfven speed — the sub-Alfvenic corona. These passages allow direct sampling of the coronal source regions of the solar wind.

---

## 7. Heliospheric Current Sheet

The heliospheric current sheet (HCS) is the largest coherent structure in the solar system — a thin surface separating regions of opposite magnetic polarity that extends from the Sun to the outer boundary of the heliosphere.

### 7.1 Origin and Structure

The Sun's large-scale magnetic field resembles a tilted dipole (at least during solar minimum). The HCS is the extension of the coronal neutral line — where the radial magnetic field changes sign — into the heliosphere. Because the magnetic field is frozen into the solar wind, the HCS is carried outward as a thin current-carrying surface.

The current density in the HCS maintains the reversal of $B_r$ across the sheet:

$$\nabla \times \mathbf{B} = \frac{4\pi}{c}\mathbf{J}$$

The HCS thickness at 1 AU is typically $\sim 10^4$–$10^5$ km (much thinner than the ~1 AU scale of the heliosphere).

### 7.2 Ballerina Skirt Analogy

The HCS shape is often described using the "ballerina skirt" analogy. Just as a spinning dancer's skirt forms a wavy surface around her body, the rotating Sun's tilted magnetic dipole produces a wavy current sheet that undulates above and below the equatorial plane.

The HCS can be approximated as a warped surface at:

$$\theta_{\text{HCS}}(\phi, r) = \frac{\pi}{2} + \alpha\sin\left(\phi - \frac{\Omega_\odot r}{v_{\text{sw}}}\right)$$

where $\alpha$ is the tilt angle of the solar magnetic dipole axis relative to the rotation axis.

### 7.3 Solar Cycle Variation

The HCS tilt and complexity vary dramatically with the solar cycle:

- **Solar minimum**: The dipole is nearly aligned with the rotation axis ($\alpha \sim 10°$–$15°$). The HCS is relatively flat, close to the equatorial plane. The sector structure is simple (two sectors per rotation).

- **Solar maximum**: The dipole tilts to large angles ($\alpha \sim 70°$–$80°$), and higher-order multipoles become important. The HCS becomes highly warped and complex, reaching to high heliolatitudes. The sector structure can have four or more sectors per rotation.

### 7.4 Sector Structure

As Earth orbits the Sun (and the Sun rotates), the Earth crosses the HCS multiple times per solar rotation. Each crossing corresponds to a reversal of the predominant IMF direction — the magnetic sector boundary.

During solar minimum, the simple two-sector structure produces a ~13.5-day recurrence in the IMF polarity (half the ~27-day rotation period). During solar maximum, the sector structure is more complex with four or more sectors.

The sector structure has a subtle but measurable effect on geomagnetic activity. The Russell-McPherron effect causes enhanced geomagnetic activity near the equinoxes when the IMF has a component antiparallel to Earth's dipole in the GSM coordinate system.

### 7.5 Heliospheric Plasma Sheet

The HCS is embedded in a broader region of enhanced density and depressed speed — the heliospheric plasma sheet (HPS). The HPS has:
- Enhanced density: 2–5 times the ambient solar wind
- Slower speed: ~300–350 km/s
- Higher variability
- Enhanced $\beta$

The HPS corresponds to the extension of the coronal streamer belt into the heliosphere and is a primary source region for the slow solar wind.

---

## Practice Problems

1. **Solar Wind Mass Loss**: Calculate the solar wind mass loss rate $\dot{M} = 4\pi r^2 n m_p v$ at 1 AU using $n = 5$ cm$^{-3}$ and $v = 400$ km/s. (a) Express $\dot{M}$ in g/s and $M_\odot$/yr. (b) How long would it take to lose 1% of the Sun's mass at this rate? (c) How does this compare to the mass loss rate due to radiation ($L_\odot/c^2$)?

2. **Parker Spiral Angle**: (a) Derive the garden-hose angle $\psi$ at distance $r$ for solar wind speed $v_{\text{sw}}$ and solar rotation rate $\Omega_\odot$. (b) Calculate $\psi$ at 1 AU for $v_{\text{sw}} = 350$ km/s (slow wind) and $v_{\text{sw}} = 750$ km/s (fast wind). (c) At what distance does the spiral become nearly transverse ($\psi = 80°$) for 400 km/s wind? (d) Sketch the magnetic field lines in the ecliptic plane for both fast and slow wind sectors.

3. **CIR Compression**: A fast wind stream ($v_f = 700$ km/s, $n_f = 3$ cm$^{-3}$, $B_f = 4$ nT) overtakes a slow stream ($v_s = 350$ km/s, $n_s = 10$ cm$^{-3}$, $B_s = 5$ nT). (a) In the frame of the stream interface, what are the inflow speeds of the fast and slow wind? (b) Estimate the compression ratio across each shock using the Rankine-Hugoniot relations for a perpendicular MHD shock with Mach number $M \approx v_{\text{inflow}}/c_f$ where $c_f$ is the fast magnetosonic speed.

4. **Turbulence Spectrum**: The magnetic field power spectrum at 1 AU follows $P(f) = P_0 f^{-5/3}$ in the inertial range from $f_1 = 10^{-3}$ Hz to $f_2 = 0.5$ Hz, with $P_0 = 10$ nT$^2$/Hz at $f = 10^{-2}$ Hz. (a) Calculate the total fluctuation energy $\langle\delta B^2\rangle = \int_{f_1}^{f_2} P(f) \, df$. (b) If $n = 5$ cm$^{-3}$, calculate the ratio of fluctuation magnetic energy density to mean field energy density ($B_0 = 5$ nT). (c) Estimate the turbulent heating rate assuming the energy cascades at the Kolmogorov rate $\epsilon \sim \delta v^3/l$ where $l$ is the outer scale.

5. **Critical Point**: For Parker's isothermal solar wind model with coronal temperature $T = 1.5 \times 10^6$ K: (a) Calculate the isothermal sound speed $c_s = \sqrt{k_BT/m_p}$. (b) Find the critical radius $r_c = GM_\odot/(2c_s^2)$ in units of $R_\odot$. (c) If the wind speed at the critical point equals $c_s$, estimate the asymptotic wind speed far from the Sun using Bernoulli's equation: $\frac{1}{2}v_\infty^2 \approx 2c_s^2\ln(r_c/r_0) - \frac{GM_\odot}{r_0} + \frac{1}{2}c_s^2 + \frac{5}{2}c_s^2$ where $r_0 = R_\odot$. Discuss whether this is sufficient for the fast wind.

---

**Previous**: [Coronal Mass Ejections](./11_Coronal_Mass_Ejections.md) | **Next**: [Solar Spectroscopy and Instruments](./13_Solar_Spectroscopy_and_Instruments.md)

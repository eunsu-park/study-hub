# Magnetospheric Current Systems

## Learning Objectives

- Identify all major magnetospheric current systems and their interconnections
- Understand the Chapman-Ferraro magnetopause current and its role in magnetic confinement
- Describe the ring current, its composition, and its relationship to the Dst index
- Explain the cross-tail current sheet and its role in magnetotail equilibrium
- Derive the structure of Birkeland (field-aligned) currents and their Region 1/Region 2 organization
- Connect ionospheric Pedersen and Hall currents to their magnetospheric drivers
- Analyze the energy flow from the solar wind through current systems to ionospheric dissipation

---

## 1. Overview: Why Currents Matter

### 1.1 Currents as the Language of Magnetospheric Physics

Every geomagnetic disturbance — every storm, substorm, and quiet-time variation — is produced by electric currents flowing in the magnetosphere-ionosphere system. Ground magnetometers do not directly measure the magnetosphere; they measure the magnetic perturbations produced by currents. Therefore, understanding current systems is essential for interpreting observations and building predictive models.

From Ampere's law:

$$\mathbf{j} = \frac{1}{\mu_0} \nabla \times \mathbf{B}$$

Wherever the magnetic field has spatial gradients (which is everywhere in the magnetosphere), currents flow. The magnetosphere is not a vacuum with a few current sheets — it is a three-dimensional system of distributed and sheet currents that self-consistently maintain the magnetic field configuration.

### 1.2 Energy Perspective

The current systems are the pathways through which solar wind energy flows into and through the magnetosphere:

- Total power input during storms: $\sim$$10^{12}$–$10^{13}$ W (comparable to US electricity consumption)
- Current systems convert magnetic energy to particle kinetic energy and heat
- The partitioning of energy among current systems determines which effects (storms, substorms, aurora, GICs) are most prominent

### 1.3 Current Closure

A fundamental principle is that all currents must close — there are no magnetic monopoles. Every current system forms a closed loop, though the loop may span the entire magnetosphere-ionosphere system. This closure requirement constrains the possible current configurations and links apparently distant phenomena (e.g., the ring current in the equatorial plane is connected to the ionosphere through field-aligned currents).

---

## 2. Chapman-Ferraro Current (Magnetopause)

### 2.1 Physical Origin

The Chapman-Ferraro current is the electric current flowing on the magnetopause surface that maintains the separation between the terrestrial magnetic field and the solar wind. It was first proposed by Chapman and Ferraro (1931) as the mechanism by which a "stream of corpuscles" from the Sun is deflected by Earth's magnetic field.

The physical origin is straightforward: when solar wind particles encounter the increasing geomagnetic field, ions and electrons gyrate in opposite directions, creating a net current. Specifically:

- Solar wind ions (positive) are deflected duskward (westward looking from above the north pole)
- Solar wind electrons (negative) are deflected dawnward (eastward)
- The net current flows from dawn to dusk across the dayside magnetopause

### 2.2 Current Direction and Magnitude

On the dayside (subsolar region), the Chapman-Ferraro current flows from dawn to dusk. The current then closes over the polar regions of the magnetopause from dusk to dawn, forming a complete circuit.

The surface current density (current per unit length along the magnetopause) is:

$$\mathbf{K} = \frac{1}{\mu_0} \hat{n} \times [\mathbf{B}]$$

where $\hat{n}$ is the outward normal to the magnetopause and $[\mathbf{B}]$ is the jump in magnetic field across the boundary. For the subsolar point where $B_{inside} \approx 50$ nT and $B_{outside} \approx 0$ (idealized):

$$K \sim \frac{\Delta B}{\mu_0} \sim \frac{50 \times 10^{-9}}{4\pi \times 10^{-7}} \approx 40 \text{ mA/m}$$

In reality, the magnetosheath field is not zero, and the current density is $K \sim 40$–80 mA/m.

### 2.3 Magnetic Perturbation at Ground

The Chapman-Ferraro current produces a **positive** perturbation in the horizontal (H) component of the magnetic field at the ground (it adds to the dipole field). The magnitude depends on the magnetopause distance:

$$\Delta B_{CF} \sim \frac{\mu_0 K R_{mp}}{R_E^2} \sim +20\text{–}30 \text{ nT}$$

This positive perturbation is responsible for the **Storm Sudden Commencement (SSC)** — the sharp positive step in ground magnetometer H-component that marks the arrival of an interplanetary shock compressing the magnetopause.

During magnetopause compression events, the Chapman-Ferraro current intensifies (stronger pressure balance requires stronger current), producing a larger positive $\Delta H$.

---

## 3. Ring Current

### 3.1 Physical Origin

The ring current is a toroidal electric current carried by energetic (10–300 keV) ions and electrons trapped in the inner magnetosphere, drifting around Earth on closed paths. It is the most important current system for understanding geomagnetic storms.

The drift physics that generates the ring current involves gradient and curvature drifts:

$$\mathbf{v}_d = \frac{m v_\perp^2}{2qB^2}(\mathbf{B} \times \nabla B) + \frac{m v_\parallel^2}{qR_c^2 B^2}(\mathbf{R}_c \times \mathbf{B})$$

where $R_c$ is the radius of curvature of the field line. The key point is that the gradient-curvature drift is charge-dependent: **ions drift westward** and **electrons drift eastward** in the geomagnetic field. Since the current is $\mathbf{j} = nq\mathbf{v}_d$, and the ion drift is in the $-\hat{\phi}$ direction while the electron drift is in the $+\hat{\phi}$ direction, both species contribute to a net **westward** (dawn-to-dusk in the equatorial plane) current.

An analogy: imagine runners on a circular track. Positive ions run clockwise (westward, viewed from above the north pole) while negative electrons run counterclockwise (eastward). But since current direction is defined as the direction positive charges flow, the electron eastward motion contributes a westward conventional current. Both drift currents add up.

### 3.2 Composition and Energy

The ring current is carried by a mixture of ions:

| Species | Quiet Time | Storm Time | Source |
|---------|-----------|------------|--------|
| H$^+$ | ~80% | ~50% | Solar wind, ionosphere |
| O$^+$ | ~10% | ~40% | Ionosphere (upwelling) |
| He$^+$ | ~5% | ~5% | Solar wind |
| He$^{++}$ | ~5% | ~5% | Solar wind |

The dramatic increase in O$^+$ during storms reflects enhanced ionospheric outflow driven by storm-time heating. Oxygen ions, being 16 times heavier than protons at the same energy, carry significant current. The O$^+$ contribution can dominate during the main phase of intense storms.

**Total ring current energy:**
- Quiet time: $\sim$$2 \times 10^{14}$ J
- Moderate storm: $\sim$$5 \times 10^{14}$ J
- Intense storm: $\sim$$2 \times 10^{15}$ J
- Extreme storm: $>$$10^{16}$ J

### 3.3 Dessler-Parker-Sckopke (DPS) Relation

The most important result connecting the ring current to observations is the Dessler-Parker-Sckopke relation, which relates the magnetic field depression at Earth's center (essentially the Dst index) to the total energy of trapped particles:

$$\frac{\Delta B_z}{B_0} = -\frac{2}{3} \frac{E_{total}}{E_{mag}}$$

where $E_{total}$ is the total kinetic energy of all trapped particles and $E_{mag} = B_0^2 (4\pi R_E^3/3)/(2\mu_0)$ is the magnetic energy of the dipole field within $R_E$. Numerically:

$$\Delta B_z \text{ (nT)} \approx -3.98 \times 10^{-30} \times E_{total} \text{ (J)}$$

or equivalently:

$$\text{Dst (nT)} \approx -\frac{2\mu_0}{4\pi B_0 R_E^3} E_{total}$$

For a moderate storm with $E_{total} = 5 \times 10^{14}$ J: $\Delta B \approx -20$ nT.
For an intense storm with $E_{total} = 5 \times 10^{15}$ J: $\Delta B \approx -200$ nT.

The DPS relation is powerful because it connects a single ground-based measurement (Dst) to a global property of the magnetosphere (total trapped particle energy). However, it is an approximation — it assumes all particles are on closed drift paths, which is not true for the asymmetric (partial) ring current.

### 3.4 Symmetric vs. Asymmetric Ring Current

During a storm:

1. **Main phase injection** — Energetic particles are injected from the nightside plasma sheet (midnight to dawn sector). Initially, the enhanced current is concentrated on the nightside: this is the **asymmetric** (or partial) ring current.

2. **Drift and symmetrization** — Injected ions drift westward, spreading around the Earth. After several hours (a few drift periods), the current becomes roughly azimuthally symmetric: the **symmetric ring current**.

3. **Recovery phase** — The symmetric ring current decays through charge exchange (ring current ions capture electrons from cold geocoronal hydrogen, becoming energetic neutral atoms that escape the magnetic trap), Coulomb collisions, and wave-particle interactions. Recovery timescale: $\sim$7–10 hours for the fast component, days for the slow component.

The distinction between asymmetric and symmetric ring current is important for ground magnetometer interpretation: the asymmetric ring current produces a local-time-dependent perturbation (captured by the ASY-H index), while the symmetric ring current produces a uniform depression (captured by the SYM-H index, essentially high-resolution Dst).

---

## 4. Tail Current Sheet

### 4.1 Structure

The cross-tail current sheet is a dawn-to-dusk current flowing through the plasma sheet of the magnetotail. This current is essential for maintaining the tail structure: without it, the north and south lobe fields (pointing in opposite directions) would immediately reconnect and the tail would collapse.

**Current density**: Typically $j \sim 5$–10 nA/m$^2$ in the central plasma sheet.

**Thickness**: The current sheet thickness varies dramatically:
- Quiet time: $\sim$1–5 $R_E$ (thick, stable)
- Growth phase (pre-substorm): Thins to $\sim$0.1–0.5 $R_E$ as magnetic flux is loaded into the tail
- Just before substorm onset: Can thin to $\sim$0.1 $R_E$ ($\sim$600 km), approaching the ion gyroradius scale

### 4.2 Current Closure

The cross-tail current does not simply terminate at the magnetopause flanks. It connects to the magnetopause (Chapman-Ferraro) current, forming a large-scale circuit:

$$\text{Dawn magnetopause} \xrightarrow{\text{cross-tail}} \text{Dusk magnetopause} \xrightarrow{\text{return via magnetopause surface}} \text{Dawn magnetopause}$$

The total cross-tail current is $\sim$10$^6$ A (1 MA) during quiet times, increasing to several MA during disturbed periods.

### 4.3 Magnetic Effect

The cross-tail current produces two effects:

1. **Stretches the nightside field** — Adds a tailward component to $B_x$ in the lobes, elongating the magnetotail.
2. **Reduces $B_z$ at the neutral sheet** — The current opposes the northward dipole component in the equatorial plane, thinning the field and making it more tail-like.

From a practical perspective, the tail current contributes a **negative** perturbation to Dst (typically $-$10 to $-$20 nT), which must be subtracted when using Dst to estimate ring current energy.

### 4.4 Current Sheet Thinning and Substorm Onset

The process of current sheet thinning before substorm onset is one of the most studied (and debated) topics in magnetospheric physics. As the IMF remains southward, dayside reconnection continues adding open flux to the tail:

1. Tail lobes accumulate magnetic flux, increasing lobe field strength
2. Pressure balance requires the plasma sheet to compress and the current sheet to thin
3. When the current sheet thins to ion-scale thicknesses, the simple MHD description breaks down
4. Reconnection initiates at the NENL, triggering the substorm expansion phase

The Harris current sheet model provides a useful idealization:

$$B_x(z) = B_0 \tanh\left(\frac{z}{\delta}\right)$$

where $\delta$ is the half-thickness. The corresponding current density is:

$$j_y(z) = \frac{B_0}{\mu_0 \delta} \text{sech}^2\left(\frac{z}{\delta}\right)$$

As $\delta$ decreases (thinning), the peak current density increases proportionally.

---

## 5. Birkeland (Field-Aligned) Currents

### 5.1 Historical Context

The existence of field-aligned currents connecting the magnetosphere to the ionosphere was first proposed by Kristian Birkeland in 1908, based on his analysis of magnetic perturbations during auroral displays. His hypothesis was controversial for decades — many scientists believed the ionosphere-magnetosphere system could be described by purely perpendicular currents. The definitive confirmation came from satellite magnetometer measurements by Iijima and Potemra (1976), who mapped the systematic pattern of field-aligned currents using Triad satellite data.

### 5.2 Region 1 and Region 2 Currents

Birkeland currents are organized into two concentric rings at auroral latitudes:

**Region 1 (R1)** — Located at the poleward boundary of the auroral oval ($\sim$70–75$°$ magnetic latitude):
- Connected to the magnetopause boundary layer / low-latitude boundary layer (LLBL) and open-closed field line boundary
- **Dawn side**: current flows into the ionosphere (downward)
- **Dusk side**: current flows out of the ionosphere (upward)
- Driven by the solar wind dynamo: the $\mathbf{v}_{sw} \times \mathbf{B}_{IMF}$ electric field maps along field lines to the high-latitude ionosphere

**Region 2 (R2)** — Located at the equatorward boundary of the auroral oval ($\sim$65–70$°$ magnetic latitude):
- Connected to the inner magnetosphere, specifically the partial ring current region
- **Dawn side**: current flows out of the ionosphere (upward) — opposite to R1
- **Dusk side**: current flows into the ionosphere (downward) — opposite to R1
- Driven by pressure gradients in the inner magnetosphere: $\mathbf{j}_\parallel = -\frac{B}{B^2} \nabla \cdot \left(\frac{\mathbf{b} \times \nabla P}{nq\Omega}\right)$

### 5.3 Magnitudes

| Parameter | Quiet Time | Storm Time |
|-----------|-----------|------------|
| R1 total current (per hemisphere) | 1–2 MA | 5–10 MA |
| R2 total current (per hemisphere) | 0.5–1.5 MA | 3–8 MA |
| Peak current density | $\sim$1 $\mu$A/m$^2$ | $\sim$10 $\mu$A/m$^2$ |
| R1 latitude | $\sim$75$°$ | $\sim$65$°$ (expanded equatorward) |

During extreme storms, the total Birkeland current can exceed 20 MA per hemisphere, and the auroral oval can expand to mid-latitudes ($\sim$50$°$).

### 5.4 Current Circuit: Magnetosphere-Ionosphere Coupling

The Birkeland currents are the critical link connecting magnetospheric dynamics to ground-level effects. The complete circuit is:

$$\text{Magnetopause/LLBL generator} \xrightarrow{R1 \downarrow} \text{Ionosphere (Pedersen/Hall)} \xrightarrow{R2 \uparrow} \text{Inner magnetosphere/ring current} \xrightarrow{drift/pressure} \text{back to boundary}$$

This circuit transfers electromagnetic energy (Poynting flux) from the magnetosphere to the ionosphere, where it is dissipated as Joule heating and particle precipitation energy.

The importance of this coupling cannot be overstated: without field-aligned currents, the magnetosphere and ionosphere would be electrically decoupled, and most of the phenomena we associate with space weather (aurora, ionospheric storms, GICs) would not occur.

---

## 6. Ionospheric Currents

### 6.1 Conductivity Tensor

The ionosphere is a weakly ionized plasma where collisions between ions, electrons, and neutrals create an anisotropic conductivity. In the presence of a magnetic field $\mathbf{B}$, the conductivity is a tensor with three independent components:

**Parallel (direct) conductivity**:
$$\sigma_0 = \frac{ne^2}{m_e \nu_e} + \frac{ne^2}{m_i \nu_i}$$

This is the conductivity along $\mathbf{B}$. It is very large ($\sim$$10^4$ S/m), so field lines are essentially equipotentials — this is why magnetospheric electric fields map to the ionosphere.

**Pedersen conductivity** (along $\mathbf{E}_\perp$):
$$\sigma_P = ne\left(\frac{\nu_i \Omega_i}{\nu_i^2 + \Omega_i^2} + \frac{\nu_e \Omega_e}{\nu_e^2 + \Omega_e^2}\right)$$

**Hall conductivity** (along $-\mathbf{E} \times \mathbf{B}$ direction):
$$\sigma_H = ne\left(\frac{\Omega_i^2}{\nu_i^2 + \Omega_i^2} - \frac{\Omega_e^2}{\nu_e^2 + \Omega_e^2}\right)$$

where $\nu$ is the collision frequency with neutrals and $\Omega = eB/m$ is the gyrofrequency.

### 6.2 Height Dependence

The ionospheric conductivities are strongly height-dependent because collision frequencies decrease rapidly with altitude while gyrofrequencies are relatively constant:

- **Below 90 km** (D-region): Both ions and electrons are collision-dominated ($\nu \gg \Omega$). $\sigma_P$ and $\sigma_H$ are small.
- **90–150 km** (E-region): Ions are still collision-dominated ($\nu_i > \Omega_i$) but electrons are magnetized ($\nu_e < \Omega_e$). This is where $\sigma_P$ and $\sigma_H$ peak. The **E-region is where most ionospheric current flows**.
- **Above 150 km** (F-region): Both ions and electrons are magnetized. $\sigma_P$ decreases, $\sigma_H$ becomes negligible. $\sigma_0$ dominates.

### 6.3 Auroral Electrojet

The auroral electrojet is an intense, horizontally flowing current system in the auroral zone E-region ($\sim$100–130 km altitude). It consists primarily of Hall current driven by the convection electric field:

- **Eastward electrojet** — Flows in the dusk sector (afternoon to pre-midnight), driven by northward electric field in the dusk convection cell
- **Westward electrojet** — Flows in the dawn sector (midnight to morning), intensifies dramatically during substorms. The substorm electrojet current can exceed $10^6$ A.

The auroral electrojet produces the **AE (Auroral Electrojet) index** — the difference between the maximum positive (AU) and maximum negative (AL) perturbations measured by a ring of high-latitude magnetometers. AE is the primary measure of substorm activity.

The rapidly varying auroral electrojet is the primary source of $dB/dt$ that induces geoelectric fields and GICs on the ground.

### 6.4 Equatorial Electrojet

At the magnetic equator, a special geometry enhances the effective conductivity. The magnetic field is horizontal, and the convection/tidal electric field is eastward (during the day). The standard Pedersen current flows eastward. However, the associated Hall current flows vertically — but it cannot flow vertically because the ionosphere has finite vertical extent. The resulting charge accumulation creates a vertical polarization electric field that drives an additional Pedersen current in the eastward direction.

The effective conductivity at the equator — the **Cowling conductivity** — is:

$$\sigma_C = \sigma_P + \frac{\sigma_H^2}{\sigma_P}$$

which can be 5–10 times larger than $\sigma_P$ alone. This enhanced conductivity produces the **equatorial electrojet**: an intense, narrow ($\pm 3°$ latitude) eastward current at the magnetic equator. It is detected as an enhanced H-component at equatorial magnetometer stations (the "Huancayo anomaly").

### 6.5 Sq Current System

The quiet-day solar (Sq) current system is the background ionospheric current driven by thermospheric tidal winds:

- Solar heating creates a diurnal/semidiurnal tide in the neutral atmosphere
- Neutral winds drag ions across magnetic field lines (dynamo action), creating electric fields and currents
- Current pattern: two vortices (viewed from above the north pole) — counterclockwise in the morning hemisphere, clockwise in the afternoon hemisphere
- Magnetic perturbation at the ground: $\sim$20–50 nT, regular daily variation
- The Sq current system must be subtracted from magnetometer data before interpreting storm-time perturbations

---

## 7. Current System Coupling and Energy Flow

### 7.1 The Complete Circuit

All magnetospheric current systems are coupled into a coherent, closed-circuit system. The primary energy flow path during southward IMF driving is:

1. **Solar wind dynamo** — The motional electric field $\mathbf{E} = -\mathbf{v}_{sw} \times \mathbf{B}_{IMF}$ generates an EMF across the magnetosphere.
2. **Region 1 Birkeland currents** — Carry the current from the magnetopause/boundary layer down to the ionosphere along magnetic field lines.
3. **Ionospheric closure** — The Birkeland currents close in the ionosphere through Pedersen (along E) and Hall (perpendicular to E) currents. Energy is dissipated as Joule heating.
4. **Region 2 Birkeland currents** — Return current flows upward from the equatorward auroral boundary, connecting to the inner magnetosphere.
5. **Ring current/partial ring current** — R2 currents close through the azimuthal drift current in the inner magnetosphere.
6. **Return to boundary** — The circuit closes back at the magnetopause through the Chapman-Ferraro and partial ring current extension.

### 7.2 Poynting Flux

The electromagnetic energy transferred from the magnetosphere to the ionosphere is quantified by the Poynting flux:

$$\mathbf{S} = \frac{1}{\mu_0} \mathbf{E} \times \delta\mathbf{B}$$

where $\mathbf{E}$ is the convection electric field and $\delta\mathbf{B}$ is the magnetic perturbation from Birkeland currents. The Poynting flux is directed downward (into the ionosphere) in regions where $\mathbf{E}$ and $\delta\mathbf{B}$ are configured to give a downward $\mathbf{S}$.

### 7.3 Joule Heating

The dominant form of energy dissipation in the ionosphere is Joule (ohmic) heating:

$$Q_J = \int \sigma_P E^2 \, dA$$

where the integral is over the entire high-latitude ionosphere (both hemispheres). The height-integrated Pedersen conductivity $\Sigma_P = \int \sigma_P \, dz$ is typically 5–15 S in the sunlit ionosphere and 1–5 S on the nightside (lower due to reduced solar EUV ionization).

**Typical Joule heating rates:**
- Quiet time: $\sim$30–100 GW
- Moderate storm: $\sim$200–500 GW
- Intense storm: $\sim$500–2000 GW

For comparison, total US electricity generating capacity is $\sim$1100 GW. During major storms, the ionospheric Joule heating rate exceeds the entire US power grid capacity.

### 7.4 Energy Partitioning

During geomagnetic storms, the total energy input from the solar wind is partitioned approximately as:

| Channel | Fraction | Mechanism |
|---------|----------|-----------|
| Joule heating | ~35–45% | Ionospheric Pedersen current dissipation |
| Ring current | ~25–35% | Trapped energetic particle energy |
| Particle precipitation | ~10–15% | Auroral electrons and ions impacting atmosphere |
| Plasmoid ejection | ~10–15% | Magnetic energy ejected tailward during substorms |
| Other (waves, etc.) | ~5–10% | ULF/VLF waves, plasma heating |

These fractions vary with storm intensity, solar wind driving conditions, and the specific phase of the storm.

---

## 8. Summary

The magnetospheric current systems form an interconnected electrical circuit driven by the solar wind-magnetosphere interaction:

- The **Chapman-Ferraro current** at the magnetopause confines Earth's field (positive $\Delta H$ at ground)
- The **ring current** of trapped energetic particles produces the Dst depression during storms
- The **cross-tail current** maintains the magnetotail structure
- **Birkeland (field-aligned) currents** couple the magnetosphere to the ionosphere in Region 1/Region 2 pairs
- **Ionospheric currents** (Pedersen, Hall, electrojets) close the circuit and dissipate energy
- The total system transfers $\sim$$10^{12}$–$10^{13}$ W from the solar wind to the magnetosphere-ionosphere system during storms

---

## Practice Problems

1. **Ring current energy** — During a geomagnetic storm, the Dst index reaches $-200$ nT. Using the Dessler-Parker-Sckopke relation, calculate the total energy of the ring current particles. If 40% of this energy is carried by O$^+$ ions with average energy 50 keV, estimate the total number of O$^+$ ions in the ring current.

2. **Magnetopause current density** — The subsolar magnetopause has an internal field of $B_{in} = 60$ nT and an external (magnetosheath) field of $B_{out} = 20$ nT (with different directions). Calculate the surface current density $K$. If the magnetopause has a thickness of 500 km, estimate the volume current density $j$.

3. **Joule heating estimate** — The cross-polar cap potential is $\Phi_{PC} = 150$ kV and the polar cap radius is 15$°$ of latitude ($\sim$1670 km). (a) Estimate the average electric field in the polar cap. (b) If the height-integrated Pedersen conductance is $\Sigma_P = 8$ S, estimate the total Joule heating rate in one hemisphere. (c) Compare with the total energy input rate from the Akasofu $\epsilon$-parameter.

4. **Birkeland current mapping** — Region 1 Birkeland currents carry a total of 3 MA downward on the dawn side at 70$°$ invariant latitude, distributed over a 2$°$ latitude band. (a) Calculate the average current density if the current flows through a circular annulus at 110 km altitude. (b) Using the magnetic field mapping factor $B_{iono}/B_{eq} \approx 50$ (ratio of ionospheric to equatorial field), estimate the current density at the equatorial magnetopause where the current originates.

5. **Electrojet and AE index** — The westward auroral electrojet has a total current of 500 kA flowing at 110 km altitude, centered at 67$°$ magnetic latitude. Model the electrojet as an infinite line current. (a) Calculate the magnetic perturbation (H-component) directly below the electrojet at the ground (Earth's radius $R_E = 6371$ km). (b) If a substorm doubles this current, what is the change in the AL index? (c) Why is the infinite line current model a poor approximation, and in which direction does the error go?

---

**Previous**: [Magnetosphere Structure](./02_Magnetosphere_Structure.md) | **Next**: [Solar Wind-Magnetosphere Coupling](./04_Solar_Wind_Magnetosphere_Coupling.md)

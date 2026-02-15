# 6. Reconnection Applications

## Learning Objectives

By the end of this lesson, you should be able to:

1. Describe the CSHKP model of solar flares and identify the key reconnection signatures
2. Explain the role of reconnection in coronal mass ejections (CMEs) and flux rope eruption
3. Understand the Dungey cycle and magnetospheric substorms
4. Analyze tokamak sawtooth crashes and the Kadomtsev reconnection model
5. Explain magnetic island coalescence and its dynamics
6. Describe reconnection in astrophysical jets and other high-energy phenomena
7. Implement simple models of these reconnection applications in Python

## 1. Solar Flares

### 1.1 Observational Overview

Solar flares are the most powerful explosions in the solar system, releasing up to $10^{32}$ erg ($10^{25}$ J) of energy in minutes to hours. This is equivalent to billions of megatons of TNT, or 10 million volcanic eruptions occurring simultaneously.

**Key observational features:**

- **Electromagnetic emission**: From radio waves to gamma rays
  - Soft X-ray emission: Thermal plasma at 10–30 MK
  - Hard X-ray emission: Non-thermal electrons (bremsstrahlung)
  - H-alpha ribbons: Chromospheric brightening at footpoints
  - White light emission: Rare, seen only in most energetic flares

- **Particle acceleration**: Electrons to tens of MeV, ions to GeV
  - Relativistic electrons: Radio bursts (gyrosynchrotron, plasma emission)
  - Energetic protons: Nuclear gamma-ray lines (de-excitation)

- **Mass ejection**: Often associated with CMEs (but not always)

- **Time scales**:
  - Preflare phase: Minutes to hours (energy storage)
  - Impulsive phase: Seconds to minutes (energy release)
  - Gradual phase: Minutes to hours (cooling, gradual particle acceleration)

**Energy budget:**

The total energy released comes from magnetic energy stored in stressed coronal fields:

$$E_{mag} = \frac{B^2}{2\mu_0} \cdot V$$

For a flaring active region with $B \sim 0.01$ T, volume $V \sim (10^8 \text{ m})^3$:

$$E_{mag} \sim \frac{(0.01)^2}{2 \times 4\pi \times 10^{-7}} \times 10^{24} \sim 10^{26} \text{ J}$$

A significant fraction (10–50%) of this magnetic energy is released during the flare.

### 1.2 The CSHKP Standard Model

The standard model of eruptive solar flares is named after Carmichael, Sturrock, Hirayama, Kopp, and Pneuman (CSHKP model, developed 1964–1976). It invokes magnetic reconnection as the primary energy release mechanism.

**Cartoon structure:**

```
                        CME
                         ║
                    ╔════╩════╗
                    ║         ║  Erupting flux rope
                    ║    ☀    ║
                    ╚════╦════╝
                         ║
    ═══════════════════════════════════  Corona
                    X    ║  Reconnection point
                    ↕    ║  Current sheet
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                         ║  Post-flare loops
                    ┌────╨────┐
                    │         │  Flare arcade
                    │ ░░░░░░░ │  Hot plasma
                    └─────────┘
    ═══════════════════════════════════  Chromosphere
                    ↓         ↓  Downflows
               Ribbon 1    Ribbon 2  H-alpha brightening
```

**Key components:**

1. **Pre-eruption**: Magnetic flux rope (filament/prominence) is stored in the corona, held down by overlying arcade field.

2. **Loss of equilibrium**: Flux rope becomes unstable (e.g., torus instability, loss of equilibrium) and begins to rise.

3. **Current sheet formation**: As the flux rope rises, it stretches the underlying field, forming a vertical current sheet.

4. **Reconnection onset**: Reconnection begins in the current sheet, releasing magnetic energy.

5. **Post-flare loops**: Reconnected field lines form hot post-flare loops that rise sequentially.

6. **Footpoint heating**: Energetic particles and thermal conduction fronts travel down field lines, heating the chromosphere.

7. **Flare ribbons**: Heated chromospheric footpoints appear as bright H-alpha ribbons that separate as reconnection proceeds.

8. **Upward jet**: Reconnection outflow launches the CME upward.

**Energy release:**

Magnetic energy is converted to:
- **Kinetic energy**: Bulk plasma flows (outflows ~500–1000 km/s), CME kinetic energy
- **Thermal energy**: Heating of flare loops to 10–30 MK
- **Non-thermal particles**: Accelerated electrons and ions
- **Radiation**: X-rays, UV, optical, radio

The reconnection electric field accelerates particles, and turbulence in the reconnection region contributes to stochastic acceleration.

### 1.3 Reconnection Rate in Flares

The reconnection rate can be estimated from the **ribbon separation velocity**. As reconnection proceeds, the footpoints of the newly reconnected loops move apart. The ribbon separation speed $v_{sep}$ is related to the reconnection inflow speed $v_{in}$:

$$v_{sep} \approx v_{in} \frac{L_{corona}}{L_{ribbon}}$$

where $L_{corona}$ is the coronal height and $L_{ribbon}$ is the chromospheric footpoint separation.

Observed ribbon speeds are typically:

$$v_{sep} \sim 10\text{–}100 \text{ km/s}$$

With $L_{corona}/L_{ribbon} \sim 0.1$–1, this gives:

$$v_{in} \sim 10\text{–}100 \text{ km/s}$$

The Alfvén speed in the corona is $v_A \sim 1000$ km/s, so:

$$M_A = \frac{v_{in}}{v_A} \sim 0.01\text{–}0.1$$

This is consistent with fast reconnection (Petschek or Hall regime), not Sweet-Parker!

### 1.4 Observations from SDO and Other Missions

The Solar Dynamics Observatory (SDO), launched in 2010, has revolutionized flare observations with high-cadence (12 s), high-resolution (0.6 arcsec) images in extreme UV.

**Key findings:**

- **Supra-arcade downflows (SADs)**: Dark, tadpole-shaped structures falling into the flare arcade at ~100 km/s. Interpreted as density voids (plasmoids) ejected downward from the reconnection site.

- **Above-the-loop-top (ALT) hard X-ray sources**: Non-thermal X-ray emission above the flare loops, indicating particle acceleration at the reconnection site.

- **Reconnection inflow/outflow**: Doppler measurements (Hinode/EIS) show inflows of ~10 km/s and outflows of ~500 km/s, consistent with reconnection models.

- **Quasi-periodic pulsations (QPPs)**: Oscillations in flare emission with periods from sub-seconds to minutes. Possibly related to plasmoid ejections (see Lesson 7).

- **Magnetic flux rope structure**: Observations of sigmoids (S-shaped structures) and flux ropes before eruption, supporting the CSHKP model.

### 1.5 Python Example: CSHKP Model Diagram

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge, Ellipse
from matplotlib import patches

fig, ax = plt.subplots(figsize=(12, 14))

# Coordinate system: x horizontal, y vertical
# Chromosphere at y=0, corona y>0

# Chromosphere
ax.axhline(0, color='brown', linewidth=4, label='Chromosphere')
ax.fill_between([-6, 6], -0.5, 0, color='wheat', alpha=0.5)

# Flare ribbons
ribbon1 = Wedge((-2, 0), 0.5, 0, 180, color='red', alpha=0.7, label='Flare ribbons')
ribbon2 = Wedge((2, 0), 0.5, 0, 180, color='red', alpha=0.7)
ax.add_patch(ribbon1)
ax.add_patch(ribbon2)

# Post-flare loops
n_loops = 5
for i in range(n_loops):
    y_top = 1 + i * 0.5
    x_width = 1.5 + i * 0.3
    theta = np.linspace(0, np.pi, 50)
    x_loop = x_width * np.cos(theta)
    y_loop = y_top * np.sin(theta)
    ax.plot(x_loop, y_loop, color='orange', linewidth=2.5, alpha=0.8)

# Label one loop
ax.text(0, 1.8, 'Post-flare loops', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Current sheet
sheet_x = [0, 0]
sheet_y = [3.5, 7]
ax.plot(sheet_x, sheet_y, color='blue', linewidth=4, linestyle='--', label='Current sheet')

# X-point
ax.plot(0, 4.5, 'kx', markersize=25, markeredgewidth=4, label='Reconnection X-point')

# Inflow arrows
ax.annotate('', xy=(-0.3, 4.5), xytext=(-1.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
ax.annotate('', xy=(0.3, 4.5), xytext=(1.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
ax.text(-2.2, 4.5, 'Inflow', fontsize=11, color='green', weight='bold')

# Downward outflow
ax.annotate('', xy=(0, 3.8), xytext=(0, 3.2),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax.text(0.3, 3.5, 'Outflow\n(downward)', fontsize=11, color='red', weight='bold')

# Upward outflow
ax.annotate('', xy=(0, 5.2), xytext=(0, 5.8),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax.text(0.3, 5.5, 'Outflow\n(upward)', fontsize=11, color='red', weight='bold')

# Erupting flux rope (CME)
flux_rope = Ellipse((0, 8.5), 3, 1.5, color='purple', alpha=0.4, label='Erupting flux rope (CME)')
ax.add_patch(flux_rope)
ax.plot(0, 8.5, 'o', color='purple', markersize=10)

# CME upward arrow
ax.annotate('', xy=(0, 10), xytext=(0, 9.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
ax.text(0.5, 9.8, 'CME', fontsize=13, color='purple', weight='bold')

# Particle beams to chromosphere
for x_foot in [-2, 2]:
    ax.annotate('', xy=(x_foot, 0.1), xytext=(0, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='magenta', linestyle='dotted'))
ax.text(-3.5, 2, 'Energetic\nparticles', fontsize=11, color='magenta', weight='bold')

# Labels
ax.text(0, -1, 'Solar Flare: CSHKP Standard Model', fontsize=18, ha='center', weight='bold')
ax.text(-5, 7, 'Corona', fontsize=13, style='italic')
ax.text(-5, -0.3, 'Chromosphere', fontsize=13, style='italic', color='brown')

# Annotations
ax.text(-5, 9, 'Energy release: ~$10^{32}$ erg', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(-5, 8.2, 'Duration: minutes to hours', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(-5, 7.4, 'Reconnection rate: $M_A \\sim 0.01$–$0.1$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax.set_xlim(-6, 6)
ax.set_ylim(-1.5, 11)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=11)
ax.axis('off')

plt.tight_layout()
plt.savefig('solar_flare_cshkp_model.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 2. Coronal Mass Ejections (CMEs)

### 2.1 What is a CME?

A **coronal mass ejection** is a massive burst of solar plasma and magnetic field ejected into interplanetary space. CMEs are often (but not always) associated with solar flares.

**Typical properties:**

- **Mass**: $10^{15}$–$10^{16}$ g (billion to 10 billion tons)
- **Kinetic energy**: $10^{30}$–$10^{32}$ erg
- **Speed**: 200–3000 km/s (average ~500 km/s, fast events >1000 km/s)
- **Occurrence rate**: ~1 per day at solar maximum, ~1 per 5 days at solar minimum

**Structure:**

CMEs often have a three-part structure:
1. **Bright frontal loop**: Compressed sheath
2. **Dark cavity**: Low-density flux rope core
3. **Bright core**: Prominence material

### 2.2 CME Initiation and Reconnection

CMEs are believed to result from the eruption of coronal magnetic flux ropes. The initiation mechanisms include:

**1. Torus Instability:**

A flux rope becomes unstable when the overlying field decreases sufficiently fast with height. The critical condition is:

$$\frac{d \ln B_{external}}{d \ln h} < -\frac{3}{2}$$

where $h$ is height. This is called the **torus instability** criterion. When met, the hoop force (self-inductance) overcomes the restraining force, and the flux rope erupts.

**2. Flux Cancellation:**

Opposite-polarity magnetic flux at the photosphere cancels (reconnects), removing the "anchor" of the overlying field and allowing the flux rope to erupt.

**3. Breakout Model:**

A quadrupolar configuration undergoes reconnection at a null point above the flux rope, removing the restraining field and triggering eruption.

**4. Kink Instability:**

If the flux rope is twisted beyond a critical threshold (typically twist $\sim 2\pi$ radians per length), it becomes kink-unstable and erupts.

**Reconnection's role:**

During the eruption, reconnection in the current sheet below the CME (as in the CSHKP model) performs two functions:
1. **Releases magnetic energy**: Powers the eruption and flare
2. **Allows field topology change**: Permits the flux rope to disconnect from the Sun

### 2.3 Space Weather Impact

When a fast CME impacts Earth, it can cause significant space weather effects:

- **Geomagnetic storms**: Compressed magnetosphere, enhanced ring current, auroral activity
  - Strongest storms: Carrington Event (1859), Halloween storms (2003), St. Patrick's Day storm (2015)

- **Radiation hazards**: Energetic particles endanger astronauts and satellites

- **Technology disruption**:
  - Power grid failures (Quebec blackout, 1989)
  - Satellite damage and loss
  - GPS and communication disruptions
  - Aviation radiation exposure (polar routes)

The **transit time** from Sun to Earth is 1–3 days for typical CMEs, providing some warning time for protective measures.

### 2.4 Observations of CME Reconnection

White-light coronagraphs (SOHO/LASCO, STEREO/COR) observe CMEs propagating through the corona. Key reconnection signatures include:

- **Streamer blowout**: Eruption of the helmet streamer structure
- **Post-CME current sheet**: Faint ray-like structure trailing the CME
- **Inflow/outflow**: Doppler shifts in UV/EUV spectral lines

The Heliospheric Imagers on STEREO tracked CMEs all the way to Earth, revealing complex interactions and deflections.

## 3. Magnetospheric Substorms

### 3.1 The Dungey Cycle

The Dungey cycle (1961) is the fundamental model of solar wind-magnetosphere coupling, driven by **dayside and nightside reconnection**.

**Dayside reconnection:**

When the interplanetary magnetic field (IMF) has a southward component ($B_z < 0$), it can reconnect with the Earth's northward-pointing field at the **dayside magnetopause**:

```
    Solar Wind                  Magnetosphere

    IMF (southward)             Geomagnetic field
         ↓                            ↑
    ─ ─ ─X─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Magnetopause
         ↓                            ↓
       Reconnected field line
```

The newly reconnected field line is pulled tailward by the solar wind, transferring energy and flux to the magnetosphere.

**Nightside reconnection:**

In the magnetotail, the stretched field reconnects at the **near-Earth neutral line**:

```
         Tail lobe (north)
              ↑   ↑
    ─ ─ ─ ─ ─ ─X─ ─ ─ ─ ─ ─  Neutral line
              ↓   ↓
         Tail lobe (south)
```

Reconnection produces:
- **Earthward flow**: Plasma accelerated toward Earth (substorm injection)
- **Tailward flow**: Plasmoid ejected downtail

The cycle closes as the Earthward-moving flux returns to the dayside, completing the circulation.

**Energy budget:**

The solar wind delivers energy at a rate:

$$P \sim v B^2 / \mu_0 \times A_{cross}$$

where $A_{cross} \sim \pi R_M^2$ is the magnetospheric cross-section and $R_M \sim 10 R_E$ (Earth radii). For typical solar wind ($v = 400$ km/s, $B = 5$ nT):

$$P \sim 10^{11}\text{–}10^{12} \text{ W}$$

During substorms, stored energy (~$10^{15}$ J) is released in ~1 hour, giving power ~$10^{11}$ W.

### 3.2 Substorm Phases

A magnetospheric substorm is a global reconfiguration of the magnetosphere, typically lasting 2–3 hours.

**Growth phase** (30–60 min):

- Dayside reconnection transfers flux to the tail
- Magnetotail stretches and thins
- Energy stored in the tail lobes
- Auroral oval expands equatorward
- Cross-tail current intensifies

**Expansion phase** (30–60 min):

- Onset of nightside reconnection
- Sudden brightening of auroras (substorm onset)
- Westward traveling surge
- Dipolarization: Tail field becomes more dipolar
- Energetic particle injection into inner magnetosphere
- Bursty bulk flows (BBFs) in the plasma sheet

**Recovery phase** (~1 hour):

- Tail field relaxes to quiet-time configuration
- Auroral activity subsides
- Plasma sheet thickens

**Observational signatures:**

- **Ground magnetometers**: Negative bay in H-component (northward field decrease)
- **Auroral images**: Brightening and poleward expansion
- **In situ spacecraft**: Flow bursts, dipolarization fronts, particle injections
- **Auroral kilometric radiation (AKR)**: Intense radio emission

### 3.3 Near-Earth Neutral Line Model

The **Near-Earth Neutral Line (NENL) model** attributes substorm onset to reconnection forming at $X \sim -20$ to $-30 R_E$ in the magnetotail.

**Sequence:**

1. Growth phase: Current sheet thinning, pressure buildup
2. Onset: Reconnection initiates at the NENL
3. Expansion: Reconnection region extends in $X$ and $Y$ (dawn-dusk)
4. Earthward and tailward jets launched from X-line
5. Plasmoid (flux rope) ejected downtail
6. Dipolarization front propagates Earthward, delivering energetic particles

**Evidence:**

- Spacecraft observations of tailward-moving plasmoids
- Earthward bursty bulk flows with $v_x \sim 400$ km/s
- Traveling compression regions (TCRs)
- Hall magnetic field quadrupole (Cluster observations)

The reconnection rate during substorms is $M_A \sim 0.1$, indicating fast (collisionless) reconnection.

### 3.4 Python Example: Dungey Cycle Cartoon

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge, Arc

fig, ax = plt.subplots(figsize=(14, 10))

# Earth
earth = Circle((0, 0), 0.3, color='blue', alpha=0.7, label='Earth')
ax.add_patch(earth)

# Magnetopause (dayside and nightside)
theta_day = np.linspace(-np.pi/2, np.pi/2, 50)
x_day = 1.5 * np.cos(theta_day)
y_day = 1.5 * np.sin(theta_day)
ax.plot(x_day, y_day, 'k-', linewidth=3, label='Magnetopause')

# Tail magnetopause
tail_y_top = np.linspace(1.5, 1.2, 30)
tail_x_top = -np.linspace(0, 5, 30)
tail_y_bot = np.linspace(-1.5, -1.2, 30)
tail_x_bot = -np.linspace(0, 5, 30)
ax.plot(tail_x_top, tail_y_top, 'k-', linewidth=3)
ax.plot(tail_x_bot, tail_y_bot, 'k-', linewidth=3)

# Dayside X-line
ax.plot(1.5, 0, 'rx', markersize=20, markeredgewidth=4, label='Reconnection X-line')
ax.text(1.5, -0.5, 'Dayside\nreconnection', fontsize=11, ha='center', color='red', weight='bold')

# Nightside X-line
ax.plot(-3, 0, 'rx', markersize=20, markeredgewidth=4)
ax.text(-3, -0.5, 'Nightside\nreconnection', fontsize=11, ha='center', color='red', weight='bold')

# Solar wind
for y_sw in np.linspace(-2, 2, 5):
    ax.annotate('', xy=(2, y_sw), xytext=(4, y_sw),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
ax.text(4.5, 2.5, 'Solar Wind', fontsize=13, color='orange', weight='bold')

# Open field lines (dayside to tail)
# Freshly reconnected line
x_open1 = np.concatenate([np.linspace(1.5, 0, 20), np.linspace(0, -5, 30)])
y_open1 = np.concatenate([np.linspace(0, 1.8, 20), np.linspace(1.8, 1.2, 30)])
ax.plot(x_open1, y_open1, 'g-', linewidth=2, alpha=0.7)
ax.plot(x_open1, -y_open1, 'g-', linewidth=2, alpha=0.7)

# Add arrows to show tailward motion
ax.annotate('', xy=(-2, 1.5), xytext=(-1, 1.6),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text(-0.5, 2.2, 'Tailward\nconvection', fontsize=10, color='green', weight='bold')

# Closed field lines (dipolar)
for r in [0.6, 0.9, 1.2]:
    theta_closed = np.linspace(-np.pi/3, np.pi/3, 40)
    x_closed = r * np.cos(theta_closed)
    y_closed = r * np.sin(theta_closed)
    ax.plot(x_closed, y_closed, 'b--', linewidth=1.5, alpha=0.5)
    ax.plot(x_closed, -y_closed, 'b--', linewidth=1.5, alpha=0.5)

# Sunward return flow
ax.annotate('', xy=(0.8, 0.6), xytext=(-1, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
ax.text(-0.5, 1.0, 'Sunward\nreturn', fontsize=10, color='purple', weight='bold')

# Plasmoid ejection
plasmoid = Circle((-4.5, 0), 0.4, color='red', alpha=0.4)
ax.add_patch(plasmoid)
ax.annotate('', xy=(-5.5, 0), xytext=(-4.5, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax.text(-5.5, -0.6, 'Plasmoid', fontsize=11, color='red', weight='bold')

# Title and labels
ax.text(0, -3.5, 'Dungey Cycle: Solar Wind-Magnetosphere Coupling', fontsize=16, ha='center', weight='bold')
ax.text(3, -2.8, 'IMF $B_z < 0$ (southward)', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Axes and formatting
ax.set_xlim(-6, 5)
ax.set_ylim(-4, 3)
ax.set_aspect('equal')
ax.legend(loc='lower left', fontsize=11)
ax.axis('off')

plt.tight_layout()
plt.savefig('dungey_cycle.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 4. Tokamak Sawtooth Crashes

### 4.1 Sawtooth Oscillations

In tokamak plasmas, the core electron temperature often exhibits **sawtooth oscillations**: a slow rise followed by a sudden crash.

```
T_e
 |     /|     /|     /|
 |    / |    / |    / |
 |   /  |   /  |   /  |
 |  /   |  /   |  /   |
 | /    | /    | /    |
 |/_____|/_____|/_____|_____  Time
   Rise  Crash  Rise  Crash
```

**Characteristics:**

- **Rise phase**: Temperature increases steadily over 10–100 ms
- **Crash phase**: Temperature drops by ~50% in <100 μs
- **Inversion radius**: Radius inside which $T_e$ drops, outside which $T_e$ rises (redistribution)
- **q-profile**: Safety factor $q(r) = r B_\phi / (R B_\theta)$ drops below 1 at the core

**Physical picture:**

1. Rise phase: Central heating, peaked current profile, $q_0$ drops below 1
2. Trigger: Internal kink instability ($m/n = 1/1$ mode) develops
3. Reconnection: Helical magnetic surfaces reconnect
4. Crash: Rapid redistribution of heat and current
5. Reset: $q_0$ rises back above 1, cycle repeats

### 4.2 Kadomtsev Reconnection Model

The **Kadomtsev model** (1975) explains the crash as full reconnection of the $q = 1$ surface.

**Before reconnection:**

The $q = 1$ surface is a nested flux surface at radius $r_1$. Inside, field lines wind once toroidally for each poloidal turn. The core is magnetically isolated.

**After reconnection:**

The helical perturbation causes the $q = 1$ surface to become a **helical island**. Full reconnection merges the island O-point with the original core, creating a new flat $q \approx 1$ profile.

**Topology change:**

```
Before:                  After:

    Nested                  Flattened
    surfaces                profile
      ○                      ─────
     ╱ ╲                    ╱     ╲
    ○ ○ ○        ────>     ─────────
     ╲ ╱                    ╲     ╱
      ○                      ─────
     q=1                     q≈1
```

**Heat redistribution:**

The reconnection rapidly mixes the hot core plasma outward and cooler edge plasma inward, causing:
- Core temperature drop
- Edge temperature rise (within the inversion radius)
- Flatten temperature profile

**Reconnection rate:**

The crash time is ~10–100 μs, much faster than resistive diffusion (which would take seconds). This suggests:

$$M_A \sim 0.01\text{–}0.1$$

consistent with collisionless reconnection on electron kinetic scales.

### 4.3 Observations and Simulations

**Experimental evidence:**

- **Soft X-ray tomography**: Shows $m=1$ precursor oscillation, then rapid crash
- **ECE (Electron Cyclotron Emission)**: High-resolution $T_e$ profile measurements
- **Magnetic diagnostics**: Mirnov coils detect $m/n = 1/1$ mode growth
- **Partial vs full reconnection**: Not all crashes are complete; some show incomplete reconnection

**Numerical simulations:**

- Resistive MHD: Reproduces sawtooth cycle but crash is too slow
- Two-fluid/kinetic: Faster crash, closer to observations
- Extended MHD: Includes Hall, electron pressure, captures fast crash

Modern simulations (e.g., M3D, NIMROD) show that two-fluid effects significantly accelerate the crash compared to resistive MHD.

### 4.4 Python Example: Sawtooth Crash Simulation (1D Model)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple 1D model of sawtooth temperature evolution
# (Not a real reconnection simulation, just illustrative)

r = np.linspace(0, 1, 100)  # Normalized radius
t_rise = 100  # Number of time steps in rise phase
t_crash = 5   # Number of time steps in crash
n_cycles = 3

# Inversion radius
r_inv = 0.3

# Initial profile
T0 = 1 - r**2

# Storage
T_history = []
time_history = []

T = T0.copy()
time = 0

for cycle in range(n_cycles):
    # Rise phase: gradual central heating
    for i in range(t_rise):
        # Heat deposition in core
        heat_source = 0.01 * np.exp(-(r / 0.2)**2)
        # Diffusive cooling
        dTdr = np.gradient(T, r)
        d2Tdr2 = np.gradient(dTdr, r)
        cooling = 0.001 * d2Tdr2

        T += heat_source + cooling
        T_history.append(T.copy())
        time_history.append(time)
        time += 1

    # Crash phase: rapid flattening inside inversion radius
    T_before_crash = T.copy()
    for i in range(t_crash):
        # Average inside inversion radius
        inside = r < r_inv
        T_avg_inside = np.mean(T[inside])
        T[inside] = T_avg_inside

        # Slight increase outside (conservation)
        outside = r >= r_inv
        T[outside] += 0.05 * (T_before_crash[inside].mean() - T_avg_inside)

        T_history.append(T.copy())
        time_history.append(time)
        time += 1

# Convert to array
T_history = np.array(T_history)
time_history = np.array(time_history)

# Plot core temperature vs time
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Core temperature (sawtooth pattern)
ax = axes[0]
T_core = T_history[:, 0]
ax.plot(time_history, T_core, linewidth=2, color='darkblue')
ax.set_xlabel('Time (arbitrary units)', fontsize=13)
ax.set_ylabel('Core Temperature $T_e(r=0)$', fontsize=13)
ax.set_title('Sawtooth Oscillations: Core Temperature', fontsize=15)
ax.grid(True, alpha=0.3)

# Mark crashes
crash_indices = []
for i in range(1, len(T_core)):
    if T_core[i] < T_core[i-1] - 0.1:
        crash_indices.append(i)
for idx in crash_indices:
    ax.axvline(time_history[idx], color='red', linestyle='--', alpha=0.6)

# Panel 2: Radial profiles at different times
ax = axes[1]

# Plot profiles at selected times
times_to_plot = [50, 99, 102, 150, 199, 202]  # Before/after crashes
colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

for i, t_idx in enumerate(times_to_plot):
    label = f't = {time_history[t_idx]}'
    if t_idx in [99, 199]:
        label += ' (before crash)'
        linestyle = '-'
        linewidth = 2.5
    elif t_idx in [102, 202]:
        label += ' (after crash)'
        linestyle = '--'
        linewidth = 2.5
    else:
        linestyle = '-'
        linewidth = 1.5

    ax.plot(r, T_history[t_idx], color=colors[i], linestyle=linestyle,
            linewidth=linewidth, label=label)

ax.axvline(r_inv, color='black', linestyle=':', linewidth=2, label=f'Inversion radius ($r={r_inv}$)')
ax.set_xlabel('Normalized radius $r/a$', fontsize=13)
ax.set_ylabel('Temperature $T_e$', fontsize=13)
ax.set_title('Radial Temperature Profiles', fontsize=15)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sawtooth_crash_simulation.png', dpi=150)
plt.show()
```

## 5. Magnetic Island Coalescence

### 5.1 Physics of Island Coalescence

When two magnetic islands (O-points) with the same helicity are brought together, they can merge through reconnection at the X-point between them. This process is called **magnetic island coalescence**.

**Initial configuration:**

```
    O─────X─────O

  Island 1  X  Island 2
```

**During coalescence:**

Reconnection at the central X-point allows the islands to merge:

```
    O───────────O   (moving together)
          X

    Reconnection accelerates
```

**Final state:**

```
       ○○○
      ○   ○
      ○ O ○  Single large island
      ○   ○
       ○○○
```

### 5.2 Dynamics and Reconnection Rate

The islands are driven together by the magnetic tension force. As they approach, the X-point current sheet intensifies, and reconnection accelerates.

**Energy conversion:**

- **Initial state**: Magnetic energy stored in the two islands
- **Coalescence**: Reconnection releases magnetic energy
- **Final state**: One larger island + kinetic/thermal energy

**Reconnection rate:**

Simulations show that coalescence reconnection is fast, with $M_A \sim 0.1$, even in resistive MHD at high Lundquist number. This is because the Rutherford regime of island growth becomes explosive during coalescence.

**Applications:**

- **Tokamak disruptions**: Multiple tearing modes can coalesce, triggering disruption
- **Solar corona**: Interacting flux tubes/loops
- **Magnetotail**: Merging plasmoids

### 5.3 Python Example: 2D Island Coalescence

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple model: two magnetic islands approaching and merging

x = np.linspace(-4, 4, 100)
y = np.linspace(-2, 2, 80)
X, Y = np.meshgrid(x, y)

# Function to create an O-point (island) flux function
def island_flux(X, Y, x0, y0, size):
    return -np.exp(-((X - x0)**2 + (Y - y0)**2) / size**2)

# Time snapshots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

times = [0, 1, 2, 3]
separations = [2.5, 1.5, 0.8, 0]  # Island separation decreases

for ax, t, sep in zip(axes.flat, times, separations):
    # Two islands approaching
    psi1 = island_flux(X, Y, -sep/2, 0, 0.6)
    psi2 = island_flux(X, Y, sep/2, 0, 0.6)

    if sep > 0:
        # Before full merger
        psi = psi1 + psi2
        # Add a current sheet between them (X-point)
        sheet_contrib = 0.2 * np.exp(-X**2 / 0.1**2) * np.exp(-(Y)**2 / 2)
        psi += sheet_contrib
    else:
        # After merger: single large island
        psi = island_flux(X, Y, 0, 0, 1.0)

    # Add background field (hyperbolic)
    psi += 0.05 * X * Y

    # Compute magnetic field
    By = np.gradient(psi, x, axis=1)
    Bx = -np.gradient(psi, y, axis=0)

    # Plot
    contour_levels = np.linspace(psi.min(), psi.max(), 20)
    ax.contour(X, Y, psi, levels=contour_levels, colors='blue', linewidths=0.8)

    # Streamplot for field lines
    ax.streamplot(X, Y, Bx, By, color='black', linewidth=0.6, density=1.2, arrowsize=0.8)

    # Mark O-points
    if sep > 0:
        ax.plot(-sep/2, 0, 'go', markersize=12, label='O-point (island)')
        ax.plot(sep/2, 0, 'go', markersize=12)
        if sep > 0.5:
            ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='X-point')
    else:
        ax.plot(0, 0, 'go', markersize=12, label='Merged island')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title(f'Time $t = {t}$ (separation = {sep:.1f})', fontsize=13)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Magnetic Island Coalescence', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('island_coalescence.png', dpi=150)
plt.show()
```

## 6. Astrophysical Jets and Reconnection

### 6.1 Jets in Astrophysics

**Astrophysical jets** are highly collimated outflows observed in many systems:

- **Active Galactic Nuclei (AGN)**: Jets from supermassive black holes, extending Mpc (millions of parsecs)
- **Microquasars**: Stellar-mass black hole jets in X-ray binaries
- **Young Stellar Objects (YSOs)**: Protostellar jets (HH objects)
- **Pulsar wind nebulae**: Relativistic jets from pulsars (Crab Nebula)
- **Gamma-Ray Bursts (GRBs)**: Ultra-relativistic jets, Lorentz factors $\Gamma \sim 100$–1000

**Common features:**

- High collimation: Opening angles ~1°–10°
- Relativistic speeds: $v \sim 0.1c$ to $>0.99c$
- High power: Up to $10^{47}$ erg/s for AGN
- Magnetic fields: Strong ($B \sim 0.1$–10 G in AGN jets)

### 6.2 Reconnection-Driven Acceleration

Magnetic reconnection is a leading candidate for:

1. **Launching jets**: Converting magnetic energy to kinetic energy
2. **Accelerating jets**: Additional acceleration along the jet
3. **Particle acceleration**: Producing non-thermal particles (synchrotron emission)

**Launching mechanism:**

In the magnetosphere of a black hole or neutron star, a rotating, magnetized accretion disk can generate a large-scale poloidal field. Reconnection in the current sheet:
- Releases magnetic tension
- Drives outflows at Alfvén speed or higher
- Collimates the flow

**Reconnection in the jet:**

Instabilities (e.g., kink) in the jet can trigger reconnection:
- Magnetic energy dissipation
- Flares and blobs in jet emission (observed in blazar variability)
- Particle acceleration to non-thermal energies (power-law distributions)

### 6.3 Pulsar Magnetospheres

**Pulsars** are rotating neutron stars with ultra-strong magnetic fields ($B \sim 10^{12}$ G). The magnetospheric structure includes:

- **Closed zone**: Dipolar field lines that close within the light cylinder
- **Open zone**: Field lines that extend to infinity
- **Current sheet**: Forms in the equatorial plane beyond the light cylinder

**Reconnection in the striped wind:**

The pulsar wind has alternating magnetic polarity (striped wind). Reconnection in the current sheet:
- Converts magnetic energy (Poynting flux) to particle energy
- Produces the observed non-thermal emission (radio, optical, X-ray, gamma-ray)
- Explains the Crab Nebula's high-energy flares

**Sigma problem:**

Near the pulsar, the magnetization parameter $\sigma = B^2/(\mu_0 \rho c^2) \gg 1$ (magnetically dominated). But observations require $\sigma \sim 0.01$–0.1 at the termination shock. Reconnection in the striped wind is a leading solution to this **sigma problem**.

### 6.4 Gamma-Ray Bursts

**GRBs** are the brightest explosions in the universe, releasing $\sim 10^{51}$–$10^{54}$ erg in gamma-rays over seconds to minutes.

**Fireball model:**

- **Central engine**: Collapsar (massive star collapse) or merger (neutron star-neutron star/black hole)
- **Relativistic outflow**: Lorentz factor $\Gamma \sim 100$–1000
- **Internal shocks**: Reconnection and shocks in the jet produce prompt gamma-ray emission
- **External shock**: Jet interacts with ISM, producing afterglow

**Reconnection's role:**

- **Energy dissipation**: Reconnection in the jet converts magnetic energy to radiation
- **Particle acceleration**: Non-thermal electrons radiate gamma-rays (synchrotron, inverse Compton)
- **Time variability**: Reconnection plasmoids produce rapid variability (ms time scales)

Recent simulations (Uzdensky, Werner, Sironi et al.) show that relativistic reconnection can efficiently accelerate particles to the required non-thermal distributions.

## 7. Summary

Magnetic reconnection plays a central role in a wide range of phenomena:

1. **Solar flares**: The CSHKP model explains energy release (~$10^{32}$ erg) through reconnection in a current sheet. Observed reconnection rates are $M_A \sim 0.01$–$0.1$, consistent with fast reconnection. Observations from SDO and other missions reveal supra-arcade downflows, above-the-loop-top sources, and ribbon dynamics.

2. **Coronal mass ejections**: CME initiation involves flux rope eruption, often triggered by the torus instability. Reconnection below the erupting flux rope releases energy and allows topology change. CMEs pose significant space weather hazards.

3. **Magnetospheric substorms**: The Dungey cycle describes solar wind-magnetosphere coupling via dayside and nightside reconnection. Substorms involve energy storage in the tail lobes during the growth phase, followed by explosive release during the expansion phase. The Near-Earth Neutral Line model attributes onset to reconnection at $X \sim -20$ to $-30 R_E$.

4. **Tokamak sawtooth crashes**: Sawtooth oscillations result from the internal kink instability and reconnection at the $q = 1$ surface. The Kadomtsev model explains the crash as full reconnection, causing rapid heat redistribution. The fast crash time indicates collisionless reconnection.

5. **Magnetic island coalescence**: When two islands merge, reconnection at the intervening X-point is fast ($M_A \sim 0.1$), even in resistive MHD. This process is relevant to tokamak disruptions and solar/magnetospheric dynamics.

6. **Astrophysical jets**: Reconnection is implicated in jet launching, acceleration, and particle acceleration in AGN, pulsars, GRBs, and YSOs. Relativistic reconnection efficiently converts magnetic energy to particle energy in pulsar winds and GRB jets.

In all these applications, the reconnection rate is observed or inferred to be fast ($M_A \sim 0.01$–$0.1$), supporting the importance of collisionless (Hall, kinetic) reconnection physics.

## Practice Problems

1. **Solar flare energetics**:
   a) Estimate the magnetic energy in a flaring active region with $B = 0.02$ T, volume $V = (10^8 \text{ m})^3$.
   b) If 20% of this energy is released in a flare lasting 1000 s, what is the average power?
   c) Compare this to the total solar luminosity ($L_\odot = 3.8 \times 10^{26}$ W).

2. **Flare ribbon motion**:
   a) Observed flare ribbons separate at $v_{sep} = 50$ km/s. If the coronal height is $h = 10^7$ m and the footpoint separation is $d = 10^8$ m, estimate the reconnection inflow speed $v_{in}$.
   b) With an Alfvén speed $v_A = 1000$ km/s, calculate $M_A$.
   c) Is this consistent with Sweet-Parker, Petschek, or Hall reconnection?

3. **CME kinetic energy**:
   a) A CME has mass $M = 10^{15}$ g and speed $v = 1000$ km/s. Calculate its kinetic energy in ergs.
   b) If the CME is decelerated to $v = 500$ km/s by solar wind drag, how much energy is dissipated?
   c) Where does this energy go?

4. **Torus instability**:
   a) Explain the torus instability criterion: $d \ln B_{ext} / d \ln h < -3/2$.
   b) For a dipole field $B \propto r^{-3}$, calculate $d \ln B / d \ln r$ (treating $h \sim r$).
   c) Is a dipole field stable or unstable against the torus instability?

5. **Dungey cycle timescale**:
   a) If dayside reconnection transfers magnetic flux at rate $d\Phi/dt = E_{rec} \cdot L_y$, where $E_{rec} = v_{in} B_{sw}$ and $L_y \sim 20 R_E$, estimate $d\Phi/dt$ for $v_{in} = 100$ km/s, $B_{sw} = 5$ nT, $R_E = 6.4 \times 10^6$ m.
   b) The total flux in the tail lobes is $\Phi_{tail} \sim B_{lobe} \cdot A_{lobe} \sim 1$ GWb. How long does it take to load this flux?
   c) Compare to observed substorm growth phase duration (~30–60 min).

6. **Substorm energy release**:
   a) The magnetotail stores energy $\sim B^2 V / (2\mu_0)$. Estimate this for $B = 20$ nT, $V \sim (10 R_E)^3$.
   b) If this energy is released over 1 hour during a substorm, what is the average power?
   c) Compare to the solar wind input power (~$10^{11}$–$10^{12}$ W).

7. **Sawtooth crash time**:
   a) In a tokamak, the sawtooth crash time is $\tau_{crash} \sim 50$ μs. The minor radius is $a = 0.5$ m, $B = 3$ T, $n = 10^{20}$ m⁻³.
   b) Calculate the Alfvén time $\tau_A = a/v_A$.
   c) Estimate the reconnection rate $M_A \sim \tau_A / \tau_{crash}$.

8. **Island coalescence**:
   a) Two magnetic islands of width $w = 5$ cm are separated by distance $d = 10$ cm in a tokamak. The local Alfvén speed is $v_A = 10^6$ m/s.
   b) If they approach at speed $v \sim 0.1 v_A$, how long until they merge?
   c) During coalescence, what fraction of the magnetic energy is released (assume islands have energy $\propto w^2 B^2$, and the final island has $w_{final} = \sqrt{2} w$)?

9. **AGN jet power**:
   a) An AGN jet has radius $R = 10^{16}$ m, outflow speed $v = 0.5c$, and carries Poynting flux $S = B^2 v / \mu_0$ with $B = 1$ G.
   b) Calculate the jet power $P = S \cdot \pi R^2$.
   c) Compare to the Eddington luminosity for a $10^9 M_\odot$ black hole ($L_{Edd} \sim 10^{47}$ erg/s).

10. **Relativistic reconnection**:
    a) In a pulsar wind, the magnetization parameter is $\sigma = B^2/(\mu_0 \rho c^2) = 10^3$ near the light cylinder.
    b) If reconnection converts 50% of the magnetic energy to particle kinetic energy, what is the final $\sigma$?
    c) Is this sufficient to explain observations ($\sigma_{obs} \sim 0.01$–0.1)? If not, what else is needed?

## Navigation

Previous: [Reconnection Theory](./05_Reconnection_Theory.md) | Next: [Advanced Reconnection](./07_Advanced_Reconnection.md)

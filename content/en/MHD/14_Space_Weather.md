# 14. Space Weather MHD

## Learning Objectives

- Understand Earth's magnetosphere structure and solar wind interaction
- Analyze magnetopause standoff distance and bow shock formation
- Describe the Dungey cycle and magnetic reconnection processes
- Model magnetic storms and the Dst index
- Study CME propagation, interplanetary shocks, and arrival prediction
- Assess geomagnetically induced currents (GIC) and space weather impacts
- Implement Python models for magnetospheric physics and space weather forecasting

## 1. Introduction to Space Weather

Space weather refers to the variable conditions on the Sun and in space that can influence the performance and reliability of space-borne and ground-based technological systems, and can endanger human life or health.

### 1.1 Solar-Terrestrial Physics

The Sun-Earth system is a coupled MHD system spanning from the solar corona ($T \sim 10^6$ K, $B \sim 1-100$ G) to Earth's magnetosphere ($B \sim 0.01-1$ G) and ionosphere ($n_e \sim 10^{11}$ m$^{-3}$).

**Key components:**
- **Solar wind**: Supersonic, super-Alfvénic plasma flow from the Sun
- **Interplanetary magnetic field (IMF)**: Frozen-in magnetic field carried by solar wind
- **Earth's magnetosphere**: Region dominated by Earth's magnetic field
- **Magnetopause**: Boundary between solar wind and magnetosphere
- **Bow shock**: Shock wave where solar wind encounters magnetosphere

### 1.2 Solar Wind Parameters

Typical solar wind conditions at 1 AU (Earth's orbit):

```
Speed:          v_sw ~ 300-800 km/s (slow/fast wind)
Density:        n_sw ~ 5-10 cm^-3
Temperature:    T_sw ~ 10^5 K
Magnetic field: B_sw ~ 5 nT
Dynamic pressure: P_dyn = ρ v² ~ 1-5 nPa
```

The solar wind is highly variable, especially during solar storms.

### 1.3 Space Weather Hazards

**Impacts:**
1. **Satellite operations**: Radiation damage, surface charging, drag from thermospheric heating
2. **Communications**: HF radio blackouts, GPS errors
3. **Power grids**: Geomagnetically induced currents (GIC) can damage transformers
4. **Aviation**: Radiation exposure at high altitudes, communication disruptions
5. **Human health**: Astronaut radiation exposure

Major events:
- **Carrington Event (1859)**: Largest recorded geomagnetic storm
- **Quebec blackout (1989)**: GIC-induced power outage affecting millions
- **Halloween storms (2003)**: Satellite anomalies, power grid disturbances
- **Bastille Day storm (2000)**: Communications disruptions

## 2. Earth's Magnetosphere

### 2.1 Dipole Magnetic Field

Earth's intrinsic magnetic field is approximately a magnetic dipole:

```
B_r = -2 B_0 (R_E/r)³ cos θ
B_θ = -B_0 (R_E/r)³ sin θ
```

where:
- $B_0 \approx 3.12 \times 10^{-5}$ T (equatorial surface field)
- $R_E = 6371$ km (Earth radius)
- $r$ is radial distance, $\theta$ is magnetic colatitude

The dipole moment:

```
M_E ≈ 8 × 10^{22} A m²
```

In the absence of solar wind, the dipole field would extend to infinity. However, solar wind pressure compresses the field on the dayside and stretches it on the nightside.

### 2.2 Magnetopause

The **magnetopause** is the boundary where the solar wind dynamic pressure balances the magnetic pressure of Earth's field.

**Pressure balance:**

```
P_dyn = B²/(2 μ₀)
```

where $B$ is the magnetospheric field at the magnetopause.

For the subsolar point (nose of magnetosphere):

```
ρ_sw v_sw² = B_mp²/(2 μ₀)
```

### 2.3 Magnetopause Standoff Distance

The standoff distance $r_{mp}$ (distance from Earth center to subsolar magnetopause) is found by balancing pressures.

Using a simple dipole model:

```
B_mp ≈ B_0 (R_E/r_mp)³
```

Pressure balance:

```
ρ_sw v_sw² = B_0² (R_E/r_mp)⁶ / (2 μ₀)
```

Solving for $r_{mp}$:

```
r_mp = R_E (B_0² / (2 μ₀ ρ_sw v_sw²))^(1/6)
```

For typical solar wind conditions ($\rho_{sw} v_{sw}^2 \sim 2$ nPa):

```
r_mp ~ 10-12 R_E
```

During strong solar wind pressure (CME arrival), $r_{mp}$ can compress to $< 8 R_E$.

### 2.4 Chapman-Ferraro Current System

The magnetopause is not a discontinuity in $B$ but a thin current sheet (the **Chapman-Ferraro current**) that shields the magnetospheric field from the solar wind.

From $\nabla \times B = \mu_0 j$, the surface current density:

```
K = (1/μ₀) ∫ j dl ≈ (B_in - B_out) / μ₀
```

where $B_{in}$ is the magnetospheric field and $B_{out}$ is the magnetosheath field (just inside the bow shock).

The current flows around the magnetopause, creating a closed loop that confines Earth's field.

### 2.5 Bow Shock

The solar wind is supersonic ($M_s > 1$) and super-Alfvénic ($M_A > 1$), so a **bow shock** forms upstream of the magnetopause, similar to a shock wave ahead of a supersonic aircraft.

**Jump conditions** (Rankine-Hugoniot relations) across the shock:

```
[ρ v_n] = 0  (mass conservation)
[ρ v_n² + p + B_t²/(2μ₀)] = 0  (momentum)
[v_n (E + p/ρ) + (v×B)_n · B_t/μ₀] = 0  (energy)
[v_n B_t - v_t B_n] = 0  (magnetic field)
```

where $v_n, v_t$ are normal and tangential velocities, $B_n, B_t$ are normal and tangential fields.

For a perpendicular shock ($B \perp v$), the compression ratio:

```
ρ₂/ρ₁ = (γ+1) M_s² / ((γ-1) M_s² + 2)
```

For $\gamma = 5/3$ and strong shock ($M_s \gg 1$):

```
ρ₂/ρ₁ → 4
```

The **magnetosheath** is the region between the bow shock and magnetopause, containing shocked, compressed, and heated solar wind plasma.

### 2.6 Magnetotail

On the nightside, the solar wind stretches Earth's field into a long **magnetotail** extending hundreds of $R_E$ downstream.

The magnetotail consists of:
- **Tail lobes**: Nearly parallel field lines pointing away from Earth (north lobe) and toward Earth (south lobe)
- **Plasma sheet**: Thin current sheet separating the lobes, containing hot plasma
- **Neutral sheet**: Surface where $B_z = 0$ in the center of the plasma sheet

The tail current (flowing in the $\pm y$ direction) maintains the lobe fields.

## 3. Dungey Cycle and Magnetic Reconnection

### 3.1 Open Magnetosphere Concept

In the **closed magnetosphere** model, solar wind flows around Earth without penetrating the magnetopause. However, when the IMF has a southward component ($B_z < 0$), **magnetic reconnection** occurs at the dayside magnetopause, allowing solar wind plasma to enter the magnetosphere.

**Dungey cycle** (for southward IMF):

1. **Dayside reconnection**: IMF and magnetospheric field lines reconnect at the subsolar magnetopause
2. **Convection**: Reconnected field lines are swept over the polar caps by solar wind
3. **Tail storage**: Field lines accumulate in the magnetotail, storing energy
4. **Nightside reconnection**: Tail field lines reconnect in the plasma sheet
5. **Return flow**: Closed field lines convect back toward dayside, returning plasma

### 3.2 Magnetic Reconnection at Magnetopause

For reconnection to occur, the magnetic field on opposite sides of the magnetopause must have anti-parallel components.

**Reconnection rate:**

The reconnection electric field $E_{rec}$ determines the rate at which flux is transferred:

```
E_rec ~ 0.1 v_sw B_sw sin²(θ/2)
```

where $\theta$ is the IMF clock angle (angle between IMF and northward direction in $yz$ plane).

Maximum reconnection occurs for southward IMF ($B_z < 0$, $\theta = 180°$).

**Flux transfer events (FTEs):**

Bursty reconnection creates flux tubes of interconnected field lines that move along the magnetopause. These are observed as sudden pulses in magnetic field magnitude and direction.

### 3.3 Magnetospheric Convection

Reconnection drives large-scale plasma circulation in the magnetosphere:

- **Dayside**: Plasma and field lines move poleward
- **Polar cap**: Antisunward flow over the poles
- **Nightside**: Sunward return flow in the plasma sheet
- **Ring current**: Azimuthal drift of energetic particles around Earth

The electric field associated with convection:

```
E_conv = -v × B
```

Typical convection speeds: 100-1000 m/s, giving $E_{conv} \sim 0.01-0.1$ mV/m.

### 3.4 Substorms

A **magnetospheric substorm** is a transient energy release in the magnetotail, occurring when the tail becomes over-loaded with magnetic flux.

**Substorm phases:**

1. **Growth phase** (30-60 min): Dayside reconnection stores energy in tail, tail lobes stretch, plasma sheet thins
2. **Expansion phase** (10-30 min): Explosive release, nightside reconnection (near 10-15 $R_E$), auroral brightening, dipolarization (tail field becomes more dipole-like)
3. **Recovery phase** (30-60 min): System relaxes to pre-substorm state

**Auroral signature:**

Substorm onset is marked by sudden brightening and expansion of the aurora, caused by energetic electrons precipitating along field lines into the ionosphere.

## 4. Magnetic Storms

A **geomagnetic storm** is a major disturbance of Earth's magnetosphere driven by prolonged periods of southward IMF, often associated with CME arrival.

### 4.1 Ring Current

During a storm, enhanced magnetospheric convection injects large numbers of energetic ions (10-200 keV) into the inner magnetosphere. These particles drift azimuthally around Earth due to gradient and curvature drifts:

```
v_drift = (m v_⊥² + v_∥²) / (q B²) (B × ∇B) + (m v_∥²) / (q B³) B × (b · ∇)b
```

For ions: drift westward; for electrons: drift eastward.

The net result is a westward **ring current** at 2-7 $R_E$ altitude, which opposes Earth's dipole field.

### 4.2 Dst Index

The **Dst (Disturbance storm time)** index measures the depression of the horizontal component of the magnetic field at equatorial ground stations, providing a global measure of ring current intensity.

```
Dst = (sum of H component deviations at 4 equatorial stations) / 4
```

**Typical values:**
- Quiet conditions: $Dst \sim 0$ nT
- Moderate storm: $Dst < -50$ nT
- Intense storm: $Dst < -100$ nT
- Super-storm: $Dst < -250$ nT

The Carrington event (1859) is estimated to have reached $Dst \sim -1600$ nT.

**Storm phases:**

1. **Initial phase** (few hours): Magnetopause compression, slight increase in field ($Dst > 0$)
2. **Main phase** (few hours): Ring current injection, rapid decrease in $Dst$
3. **Recovery phase** (days): Ring current decay via charge exchange and precipitation

### 4.3 Dst-Solar Wind Coupling

Empirical formula (Burton et al. 1975):

```
dDst/dt = Q(E_sw) - Dst/τ
```

where:
- $Q(E_{sw})$ is the injection function, depending on solar wind electric field $E_{sw} = v_{sw} B_s$
- $B_s$ is the southward component of IMF ($B_s = -B_z$ if $B_z < 0$, else 0)
- $\tau \sim 8$ hours is the decay timescale

**Burton formula:**

```
Q = a (v_sw B_s - b)  if v_sw B_s > b, else 0
```

where $a \sim 10^{-3}$ nT/(mV/m) and $b \sim 0.5$ mV/m are empirical constants.

This simple model captures the main features: stronger storms for higher $v_{sw}$ and southward $B_z$.

### 4.4 Storm Impacts

**Radiation belt enhancement:**

Storms energize electrons in the outer radiation belt (L=4-7), creating hazards for satellites.

**Ionospheric disturbances:**

Enhanced currents and particle precipitation disturb the ionosphere, affecting radio communications and GPS accuracy.

**Auroral expansion:**

The aurora can extend to mid-latitudes (40-50° latitude) during major storms.

## 5. Coronal Mass Ejections (CMEs)

### 5.1 CME Characteristics

A **coronal mass ejection (CME)** is an eruption of plasma and magnetic field from the solar corona, releasing $10^{12}-10^{13}$ kg of material at speeds of 200-3000 km/s.

**Triggering mechanisms:**
- Magnetic flux rope eruption
- Shearing and twisting of coronal magnetic field
- Loss of equilibrium (e.g., torus instability)

**CME structure:**
- **Leading edge**: Bright rim in coronagraph images
- **Cavity**: Dark region (low density)
- **Core**: Bright core (prominence material)

### 5.2 Interplanetary CME (ICME)

Once a CME propagates into interplanetary space, it is called an **ICME**. At Earth (1 AU), ICMEs have characteristic signatures:

**Magnetic cloud:**

A subset of ICMEs with smooth rotation of the magnetic field vector over ~1 day, low plasma beta ($\beta < 1$), and low temperature. This indicates a flux rope structure.

**Sheath:**

If the CME is fast ($v > v_{sw}$), it drives a shock ahead of it. The region between the shock and the flux rope is the **sheath**, containing compressed and turbulent solar wind.

### 5.3 CME Propagation Models

**Drag-based model:**

The CME experiences aerodynamic drag in the solar wind:

```
dv/dt = -γ (v - v_sw)
```

where $\gamma$ is the drag coefficient:

```
γ ≈ C_d A / (2 M)
```

- $C_d \sim 1$ is the drag coefficient
- $A$ is the CME cross-sectional area
- $M$ is the CME mass

**Analytical solution:**

```
v(t) = v_sw + (v_0 - v_sw) exp(-γ t)
```

where $v_0$ is the initial CME speed.

The distance traveled:

```
r(t) = r_0 + v_sw t + (v_0 - v_sw)/γ (1 - exp(-γ t))
```

**Transit time to Earth:**

Setting $r(t_{arr}) = 1$ AU and solving for $t_{arr}$.

For typical parameters ($v_0 = 1000$ km/s, $v_{sw} = 400$ km/s, $\gamma^{-1} = 1$ day):

```
t_arr ~ 2-3 days
```

Faster CMEs arrive in 1-2 days; slower CMEs in 3-5 days.

### 5.4 MHD Simulation Models

Advanced space weather forecasting uses 3D MHD codes to simulate CME propagation:

**ENLIL** (NOAA):
- 3D MHD model from Sun to 2 AU
- Uses solar wind data from SOHO, ACE
- Provides arrival time, speed, density forecasts

**SUSANOO** (Japan):
- Global MHD simulation with adaptive mesh refinement
- Models CME-solar wind interaction

**Other models:**
- WSA-ENLIL (coupled coronal-heliospheric model)
- EUHFORIA (European heliospheric model)

These models typically predict arrival times with ~6-12 hour accuracy.

### 5.5 CME Orientation and Geoeffectiveness

The **geoeffectiveness** of a CME depends critically on the orientation of its magnetic field.

**Southward field ($B_z < 0$):**
- Strong dayside reconnection
- Efficient energy transfer to magnetosphere
- Strong storms (large negative $Dst$)

**Northward field ($B_z > 0$):**
- Weak or no dayside reconnection
- Minimal geomagnetic activity

Since the CME field orientation can only be measured when it arrives at L1 (1.5 million km from Earth, ~1 hour warning), forecasting storm intensity is challenging.

## 6. Geomagnetically Induced Currents (GIC)

### 6.1 Physical Mechanism

Rapid variations in the magnetospheric magnetic field (during storms or substorms) induce electric fields in the conductive Earth via Faraday's law:

```
∇ × E = -∂B/∂t
```

For a spatially uniform time-varying field $B(t)$ at ground level:

```
E ~ -L ∂B/∂t
```

where $L$ is a characteristic length scale.

These electric fields drive currents in conductors: power lines, pipelines, railway tracks, etc.

### 6.2 Ground Conductivity

The induced electric field depends on ground conductivity structure:

```
E(ω) = Z(ω) · H(ω)
```

where $Z(\omega)$ is the **surface impedance** (depends on ground conductivity profile) and $H$ is the horizontal magnetic field perturbation.

Regions with high resistivity (e.g., Precambrian shield rocks in Canada, Scandinavia) are more susceptible to large GIC.

### 6.3 GIC in Power Grids

GIC flows into transformers as quasi-DC current, causing:

1. **Half-cycle saturation**: DC current biases the transformer core, leading to asymmetric magnetization
2. **Increased reactive power demand**
3. **Heating**: Excessive heating can damage transformers
4. **Harmonics**: Distortion of AC waveform
5. **Voltage instability**: Can trigger cascading failures

**Quebec blackout (March 13, 1989):**

A major geomagnetic storm induced GIC in Hydro-Québec's grid. Within 90 seconds, the entire grid collapsed, leaving 6 million people without power for up to 9 hours.

### 6.4 GIC Magnitude Estimation

Empirical scaling:

```
GIC ~ (dB/dt) / R_earth
```

where $R_{earth}$ is the Earth's effective resistance (depends on ground conductivity).

For a storm with $dB/dt \sim 1000$ nT/min and resistivity $\rho \sim 1000$ Ω·m:

```
E ~ 1-10 V/km
```

Over a 100 km transmission line:

```
V ~ 100-1000 V
```

With line resistance $R \sim 0.1$ Ω:

```
GIC ~ 100-1000 A
```

This is quasi-DC superimposed on the AC grid.

### 6.5 GIC Mitigation

**Strategies:**
1. **Operational procedures**: Reduce loading during storms
2. **Neutral blocking devices**: Insert capacitors to block DC while passing AC
3. **Network reconfiguration**: Open vulnerable connections
4. **Improved forecasting**: Advance warning allows operators to prepare

Recent focus: Understanding 100-year and 500-year GIC events for grid resilience planning.

## 7. Space Weather Forecasting

### 7.1 Observational Assets

**Solar observations:**
- **SOHO, SDO**: Solar imaging (EUV, coronagraph)
- **STEREO**: 3D CME structure

**Solar wind monitors:**
- **ACE, DSCOVR**: At L1 (1.5 million km sunward), provides ~30-60 min warning
- Measures $v, n, B$ of solar wind before it reaches Earth

**Magnetospheric monitoring:**
- **Ground magnetometers**: Global network (SuperMAG)
- **Satellites**: GOES, THEMIS, MMS

### 7.2 Forecasting Workflow

1. **Solar monitoring**: Detect flares, CMEs
2. **CME propagation model**: Estimate arrival time using MHD or empirical models
3. **L1 data assimilation**: Refine prediction when ICME reaches L1
4. **Geomagnetic indices**: Predict Kp, Dst based on solar wind coupling functions
5. **Impact assessment**: Estimate GIC, radiation belt, ionospheric effects

### 7.3 Forecast Skill

**Arrival time**: Typically ±6-12 hours for CMEs
**Intensity (Dst)**: Correlation ~0.7-0.8 (limited by unknown CME $B_z$ orientation)
**Probabilistic forecasts**: Ensemble methods improve reliability

### 7.4 Operational Centers

- **NOAA Space Weather Prediction Center (SWPC)**: U.S. operational forecasts
- **ESA Space Situational Awareness (SSA)**: European forecasts
- **UKMO Space Weather Operations Centre (MOSWOC)**: UK forecasts
- **ISES (International Space Environment Service)**: Global coordination

## 8. Python Implementations

### 8.1 Magnetopause Standoff Distance

```python
import numpy as np
import matplotlib.pyplot as plt

def magnetopause_standoff(v_sw, n_sw, B_0=3.12e-5, R_E=6371e3):
    """
    Calculate magnetopause standoff distance.

    Parameters:
    v_sw : solar wind speed (m/s)
    n_sw : solar wind density (m^-3)
    B_0 : Earth's equatorial surface field (T)
    R_E : Earth radius (m)

    Returns:
    r_mp : magnetopause standoff distance (R_E)
    """
    mu_0 = 4 * np.pi * 1e-7
    m_p = 1.673e-27  # proton mass (kg)

    # Dynamic pressure = ρv²: the ram pressure of the solar wind plasma that
    # must be balanced by the magnetospheric field pressure — this is the
    # fundamental competition that sets the magnetopause location
    rho_sw = n_sw * m_p
    P_dyn = rho_sw * v_sw**2

    # The 1/6 power comes from the dipole field B ∝ r^{-3}; squaring gives
    # B² ∝ r^{-6}, so balancing B²/(2μ₀) = P_dyn and solving for r yields r ∝ P_dyn^{-1/6}
    # — a weak dependence, meaning you need a 64× pressure increase to halve r_mp
    r_mp = R_E * (B_0**2 / (2 * mu_0 * P_dyn))**(1/6)

    return r_mp / R_E  # Return in Earth radii

# Typical solar wind conditions
v_typical = 400e3  # m/s
n_typical = 5e6    # m^-3

r_mp_typical = magnetopause_standoff(v_typical, n_typical)
print(f"Typical solar wind: v = {v_typical/1e3:.0f} km/s, n = {n_typical/1e6:.1f} cm^-3")
print(f"Magnetopause standoff: r_mp = {r_mp_typical:.1f} R_E")

# CME impact (enhanced pressure)
v_cme = 800e3  # m/s
n_cme = 20e6   # m^-3

r_mp_cme = magnetopause_standoff(v_cme, n_cme)
print(f"\nCME arrival: v = {v_cme/1e3:.0f} km/s, n = {n_cme/1e6:.1f} cm^-3")
print(f"Magnetopause standoff: r_mp = {r_mp_cme:.1f} R_E (compressed!)")

# Parametric study: vary solar wind speed
v_scan = np.linspace(300e3, 1000e3, 100)
r_mp_scan = [magnetopause_standoff(v, n_typical) for v in v_scan]

plt.figure(figsize=(10, 6))
plt.plot(v_scan/1e3, r_mp_scan, 'b-', linewidth=2)
plt.axhline(y=r_mp_typical, color='g', linestyle='--', linewidth=1.5, label='Typical conditions')
plt.axhline(y=8, color='r', linestyle='--', linewidth=1.5, label='Geosynchronous orbit')
plt.xlabel('Solar Wind Speed (km/s)', fontsize=12)
plt.ylabel('Magnetopause Standoff Distance (R$_E$)', fontsize=12)
plt.title('Magnetopause Position vs Solar Wind Speed', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('magnetopause_standoff.png', dpi=150)
plt.show()
```

### 8.2 Bow Shock Properties

```python
def bow_shock_jump(M_s, gamma=5/3):
    """
    Calculate density compression ratio across bow shock.

    Parameters:
    M_s : sonic Mach number
    gamma : adiabatic index

    Returns:
    r : compression ratio (ρ₂/ρ₁)
    """
    r = (gamma + 1) * M_s**2 / ((gamma - 1) * M_s**2 + 2)
    return r

def sonic_mach_number(v_sw, T_sw):
    """
    Calculate sonic Mach number of solar wind.

    Parameters:
    v_sw : solar wind speed (m/s)
    T_sw : solar wind temperature (K)

    Returns:
    M_s : sonic Mach number
    """
    k_B = 1.381e-23  # J/K
    m_p = 1.673e-27  # kg
    gamma = 5/3

    c_s = np.sqrt(gamma * k_B * T_sw / m_p)
    M_s = v_sw / c_s

    return M_s

# Typical solar wind
T_sw = 1e5  # K
M_s_typical = sonic_mach_number(v_typical, T_sw)
r_typical = bow_shock_jump(M_s_typical)

print(f"\nBow shock (typical solar wind):")
print(f"Sonic Mach number: M_s = {M_s_typical:.1f}")
print(f"Density compression: ρ₂/ρ₁ = {r_typical:.2f}")

# Fast solar wind (CME)
M_s_cme = sonic_mach_number(v_cme, T_sw)
r_cme = bow_shock_jump(M_s_cme)

print(f"\nBow shock (CME):")
print(f"Sonic Mach number: M_s = {M_s_cme:.1f}")
print(f"Density compression: ρ₂/ρ₁ = {r_cme:.2f}")

# Mach number scan
M_s_scan = np.linspace(1.5, 10, 100)
r_scan = [bow_shock_jump(M) for M in M_s_scan]

plt.figure(figsize=(10, 6))
plt.plot(M_s_scan, r_scan, 'r-', linewidth=2)
plt.axhline(y=4, color='k', linestyle='--', linewidth=1.5, label='Strong shock limit (r=4)')
plt.xlabel('Sonic Mach Number $M_s$', fontsize=12)
plt.ylabel('Density Compression Ratio', fontsize=12)
plt.title('Bow Shock Density Jump vs Mach Number', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bow_shock_jump.png', dpi=150)
plt.show()
```

### 8.3 Dst Index Model (Burton)

```python
def dst_evolution(Dst_0, v_sw_series, B_z_series, dt, a=1e-3, b=0.5, tau=8*3600):
    """
    Evolve Dst index using Burton model.

    Parameters:
    Dst_0 : initial Dst (nT)
    v_sw_series : solar wind speed time series (m/s)
    B_z_series : IMF Bz time series (T)
    dt : timestep (s)
    a : injection parameter (nT/(mV/m))
    b : threshold (mV/m)
    tau : decay timescale (s)

    Returns:
    Dst_series : Dst evolution (nT)
    """
    N = len(v_sw_series)
    Dst_series = np.zeros(N)
    Dst = Dst_0

    for i in range(N):
        Dst_series[i] = Dst

        # Take the southward component (B_z < 0) because only southward IMF
        # drives dayside reconnection, which opens the magnetopause and allows
        # solar wind energy to enter and energize the ring current; northward
        # IMF closes the magnetopause and reconnection ceases
        B_s = max(-B_z_series[i], 0) * 1e9  # Convert to nT, take southward component
        # E_sw = v_sw * B_s is the interplanetary electric field (mV/m):
        # it represents the rate at which flux is transferred from the IMF into
        # the magnetosphere, so stronger electric field → faster ring current injection
        E_sw = v_sw_series[i] / 1e3 * B_s / 1e6  # mV/m

        # Threshold b ≈ 0.5 mV/m: below this, the magnetopause is closed enough
        # that reconnection is negligible and the ring current is not being driven;
        # this avoids unphysical Dst responses to very weak solar wind electric fields
        if E_sw > b:
            Q = a * (E_sw - b)
        else:
            Q = 0

        # The -Dst/tau decay term models ring current loss by charge exchange of
        # energetic ions with cold neutral hydrogen; τ ≈ 8 hours is the mean
        # lifetime before ions are neutralized and lost from the radiation belt
        dDst_dt = Q - Dst / tau

        Dst += dDst_dt * dt

    return Dst_series

# Simulate a magnetic storm
t_max = 5 * 24 * 3600  # 5 days
dt = 600  # 10 min
N = int(t_max / dt)
t_series = np.arange(N) * dt / 3600  # hours

# Solar wind scenario: CME arrival at t=12 hours
v_sw_series = np.ones(N) * 400e3  # m/s
B_z_series = np.ones(N) * 2e-9  # T (northward)

# CME arrival: 12-36 hours, enhanced speed and southward field
t_cme_start = int(12 * 3600 / dt)
t_cme_end = int(36 * 3600 / dt)
v_sw_series[t_cme_start:t_cme_end] = 600e3  # m/s
B_z_series[t_cme_start:t_cme_end] = -15e-9  # T (strongly southward)

# Evolve Dst
Dst_0 = 0  # nT (quiet conditions)
Dst_series = dst_evolution(Dst_0, v_sw_series, B_z_series, dt)

print(f"\nDst storm simulation:")
print(f"Minimum Dst: {np.min(Dst_series):.1f} nT at t = {t_series[np.argmin(Dst_series)]:.1f} hours")

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(t_series, v_sw_series/1e3, 'b-', linewidth=1.5)
ax1.set_ylabel('Solar Wind Speed (km/s)', fontsize=12)
ax1.set_title('Magnetic Storm Simulation (Burton Dst Model)', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(t_series, B_z_series*1e9, 'g-', linewidth=1.5)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax2.set_ylabel('IMF $B_z$ (nT)', fontsize=12)
ax2.grid(True, alpha=0.3)

ax3.plot(t_series, Dst_series, 'r-', linewidth=2)
ax3.axhline(y=-50, color='orange', linestyle='--', linewidth=1, label='Moderate storm')
ax3.axhline(y=-100, color='red', linestyle='--', linewidth=1, label='Intense storm')
ax3.set_xlabel('Time (hours)', fontsize=12)
ax3.set_ylabel('Dst (nT)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dst_storm_simulation.png', dpi=150)
plt.show()
```

### 8.4 CME Transit Time

```python
def cme_transit_time(v_0, v_sw=400e3, gamma_inv=86400, r_target=1.496e11):
    """
    Calculate CME arrival time using drag model.

    Parameters:
    v_0 : initial CME speed (m/s)
    v_sw : solar wind speed (m/s)
    gamma_inv : inverse drag coefficient (s)
    r_target : target distance (m, default 1 AU)

    Returns:
    t_arr : arrival time (hours)
    """
    # Solve r(t) = r_target for t
    # r(t) = v_sw * t + (v_0 - v_sw) * gamma_inv * (1 - exp(-t/gamma_inv))
    # Iterative solution
    t = 0
    dt = 600  # 10 min
    r = 0
    # Start the integration from 0.1 AU (not the solar surface) because the
    # drag-based model is only valid in the heliosphere beyond the CME initiation
    # region; the CME has already been accelerated/decelerated close to the Sun
    r_0 = 0.1 * r_target  # Start at 0.1 AU (close to Sun)

    while r < r_target:
        # The velocity decays exponentially toward v_sw: a fast CME is decelerated
        # by aerodynamic drag from the ambient solar wind, while a slow CME is
        # accelerated — both converge toward v_sw on a timescale γ^{-1} ≈ 1 day
        v = v_sw + (v_0 - v_sw) * np.exp(-t / gamma_inv)
        r += v * dt
        t += dt

    return t / 3600  # Convert to hours

# CME scenarios
v_0_slow = 500e3  # m/s
v_0_fast = 1200e3  # m/s

t_arr_slow = cme_transit_time(v_0_slow)
t_arr_fast = cme_transit_time(v_0_fast)

print(f"\nCME transit time to Earth (1 AU):")
print(f"Slow CME (v₀ = {v_0_slow/1e3:.0f} km/s): {t_arr_slow:.1f} hours ({t_arr_slow/24:.1f} days)")
print(f"Fast CME (v₀ = {v_0_fast/1e3:.0f} km/s): {t_arr_fast:.1f} hours ({t_arr_fast/24:.1f} days)")

# Parametric study
v_0_scan = np.linspace(400e3, 2000e3, 50)
t_arr_scan = [cme_transit_time(v_0) for v_0 in v_0_scan]

plt.figure(figsize=(10, 6))
plt.plot(v_0_scan/1e3, np.array(t_arr_scan)/24, 'b-', linewidth=2)
plt.xlabel('Initial CME Speed (km/s)', fontsize=12)
plt.ylabel('Transit Time to 1 AU (days)', fontsize=12)
plt.title('CME Arrival Time vs Initial Speed (Drag Model)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cme_transit_time.png', dpi=150)
plt.show()
```

### 8.5 GIC Estimation

```python
def gic_estimate(dB_dt, rho_earth=1000, L=100e3):
    """
    Estimate geomagnetically induced current.

    Parameters:
    dB_dt : magnetic field time derivative (nT/min)
    rho_earth : Earth resistivity (Ω·m)
    L : transmission line length (m)

    Returns:
    E : induced electric field (V/km)
    GIC : induced current (A, assuming 0.1 Ω line resistance)
    """
    mu_0 = 4 * np.pi * 1e-7

    # The induced E-field scales as √(ρ_earth/μ₀): ground resistivity amplifies
    # the GIC because high-resistivity crust (e.g., Precambrian shield in Canada)
    # forces the induced current to flow in surface conductors (power lines)
    # rather than dispersing through the ground — this is why Quebec and Finland
    # are more vulnerable than lower-latitude regions with conductive geology
    E = (dB_dt * 1e-9 / 60) * np.sqrt(rho_earth / mu_0) / 1000  # V/km

    # Voltage over line: longer lines accumulate more of the spatially extended
    # induced electric field, so 1000 km continental transmission lines are
    # much more at risk than short urban distribution lines
    V = E * L / 1e3  # V

    # GIC = V/R: the low resistance of power transformers (~0.1 Ω) means even
    # small induced voltages drive large quasi-DC currents, which saturate the
    # iron core and cause half-cycle distortion, reactive power demand, and heating
    R_line = 0.1  # Ω
    GIC = V / R_line  # A

    return E, GIC

# Quebec blackout scenario
dB_dt_quebec = 480  # nT/min (observed)

E_quebec, GIC_quebec = gic_estimate(dB_dt_quebec)
print(f"\nQuebec blackout (1989-03-13):")
print(f"dB/dt = {dB_dt_quebec} nT/min")
print(f"Induced electric field: E ~ {E_quebec:.2f} V/km")
print(f"GIC (100 km line): ~ {GIC_quebec:.0f} A")

# Carrington event estimate
dB_dt_carrington = 5000  # nT/min (estimated)

E_carrington, GIC_carrington = gic_estimate(dB_dt_carrington)
print(f"\nCarrington event (1859, estimated):")
print(f"dB/dt = {dB_dt_carrington} nT/min")
print(f"Induced electric field: E ~ {E_carrington:.2f} V/km")
print(f"GIC (100 km line): ~ {GIC_carrington:.0f} A")

# Parametric study
dB_dt_scan = np.linspace(10, 2000, 100)
GIC_scan = [gic_estimate(dB_dt)[1] for dB_dt in dB_dt_scan]

plt.figure(figsize=(10, 6))
plt.plot(dB_dt_scan, GIC_scan, 'm-', linewidth=2)
plt.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, label='Concern level (~100 A)')
plt.axvline(x=dB_dt_quebec, color='r', linestyle='--', linewidth=1.5, label='Quebec 1989')
plt.xlabel('dB/dt (nT/min)', fontsize=12)
plt.ylabel('GIC (A)', fontsize=12)
plt.title('Geomagnetically Induced Current vs dB/dt', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gic_estimate.png', dpi=150)
plt.show()
```

### 8.6 Reconnection Electric Field

```python
def reconnection_electric_field(v_sw, B_sw, theta):
    """
    Calculate reconnection electric field at magnetopause.

    Parameters:
    v_sw : solar wind speed (m/s)
    B_sw : IMF magnitude (T)
    theta : IMF clock angle (degrees, 0=northward, 180=southward)

    Returns:
    E_rec : reconnection electric field (mV/m)
    """
    theta_rad = np.deg2rad(theta)
    E_rec = 0.1 * v_sw / 1e3 * B_sw * 1e9 * np.sin(theta_rad / 2)**2  # mV/m
    return E_rec

# Northward IMF
E_rec_north = reconnection_electric_field(v_typical, 5e-9, 0)
print(f"\nReconnection electric field:")
print(f"Northward IMF (θ=0°): E_rec = {E_rec_north:.3f} mV/m (minimal reconnection)")

# Southward IMF
E_rec_south = reconnection_electric_field(v_typical, 5e-9, 180)
print(f"Southward IMF (θ=180°): E_rec = {E_rec_south:.2f} mV/m (strong reconnection)")

# Strong southward IMF (CME)
E_rec_cme = reconnection_electric_field(v_cme, 20e-9, 180)
print(f"CME with southward field: E_rec = {E_rec_cme:.2f} mV/m (very strong!)")

# Clock angle scan
theta_scan = np.linspace(0, 180, 100)
E_rec_scan = [reconnection_electric_field(v_typical, 5e-9, theta) for theta in theta_scan]

plt.figure(figsize=(10, 6))
plt.plot(theta_scan, E_rec_scan, 'purple', linewidth=2)
plt.xlabel('IMF Clock Angle (degrees)', fontsize=12)
plt.ylabel('Reconnection E-field (mV/m)', fontsize=12)
plt.title('Magnetopause Reconnection vs IMF Orientation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reconnection_efield.png', dpi=150)
plt.show()
```

## 9. Summary

This lesson covered the MHD physics of space weather:

1. **Earth's magnetosphere**: Dipole field compressed by solar wind, forming magnetopause, bow shock, and magnetotail
2. **Magnetopause**: Pressure balance determines standoff distance ($r_{mp} \sim 10 R_E$)
3. **Bow shock**: Shocks solar wind from supersonic to subsonic, compressing density by factor ~4
4. **Dungey cycle**: Magnetic reconnection drives magnetospheric convection and substorms
5. **Magnetic storms**: Ring current formation depresses Dst index, driven by southward IMF
6. **CMEs**: Eruptive release of solar plasma/field, propagate to Earth in 1-5 days
7. **GIC**: Induced currents in power grids from rapid B variations, can cause blackouts
8. **Forecasting**: Combination of solar observations, MHD models, and L1 monitors

Space weather MHD bridges scales from the Sun to Earth, requiring global modeling and real-time forecasting to protect critical infrastructure.

## Practice Problems

1. **Magnetopause compression**: During a CME, the solar wind speed increases to $v_{sw} = 900$ km/s and density to $n_{sw} = 30$ cm$^{-3}$. Calculate the magnetopause standoff distance. Does it compress inside geosynchronous orbit (6.6 $R_E$)?

2. **Bow shock**: For solar wind at $v_{sw} = 600$ km/s, $T_{sw} = 10^5$ K, calculate the sonic Mach number and the density compression ratio across the bow shock.

3. **Dst prediction**: Using the Burton model, estimate the minimum Dst for a storm with solar wind electric field $E_{sw} = v_{sw} B_s = 5$ mV/m sustained for 6 hours. Assume initial $Dst_0 = 0$, $a = 10^{-3}$ nT/(mV/m), $b = 0.5$ mV/m, $\tau = 8$ hours. Classify the storm (moderate: $< -50$ nT; intense: $< -100$ nT).

4. **CME transit**: A CME is launched at $v_0 = 1500$ km/s. Using the drag model with $v_{sw} = 400$ km/s and $\gamma^{-1} = 1$ day, estimate the arrival time at Earth (1 AU = 1.5 × 10$^{11}$ m). Express in hours and days.

5. **Reconnection rate**: Calculate the reconnection electric field for (a) northward IMF: $v_{sw} = 400$ km/s, $B_{sw} = 5$ nT, $\theta = 0°$; (b) southward IMF: same parameters, $\theta = 180°$. Compare the reconnection efficiency.

6. **GIC hazard**: During a storm, $dB/dt = 1000$ nT/min. Estimate the induced electric field and GIC in a 200 km transmission line (assume $\rho_{earth} = 1000$ Ω·m, $R_{line} = 0.2$ Ω). Is this a concern for the power grid?

7. **Substorm energy**: A substorm releases $10^{15}$ J of energy over 30 minutes. If this energy is deposited into the ionosphere at 100 km altitude over an area of $10^{12}$ m$^2$, estimate the energy flux (W/m$^2$). Compare to the solar constant (1360 W/m$^2$).

8. **Ring current**: The Dst index depression is proportional to the ring current energy: $Dst \sim -E_{ring} / (4 \times 10^{14} \text{ J/nT})$. For $Dst = -150$ nT, estimate the ring current energy. Express in joules.

9. **CME magnetic cloud**: A magnetic cloud has $B \sim 30$ nT, $n \sim 10$ cm$^{-3}$, $T \sim 10^4$ K. Calculate the plasma beta $\beta = 2 \mu_0 p / B^2$. Is this consistent with a flux rope structure ($\beta < 1$)?

10. **Space weather forecasting**: Explain why the CME arrival time can be predicted ~12 hours in advance, but the storm intensity (Dst minimum) is uncertain until the CME reaches L1. What is the critical missing information?

---

**Previous**: [Fusion MHD](./13_Fusion_MHD.md) | **Next**: [2D MHD Solver](./15_2D_MHD_Solver.md)

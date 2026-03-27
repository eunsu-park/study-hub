# 11. Solar MHD

## Learning Objectives

By the end of this lesson, you should be able to:

- Describe the solar structure and the role of magnetic fields in different layers
- Understand the physics of magnetic flux tubes and sunspots
- Explain the solar dynamo mechanism and the 11/22-year solar cycle
- Analyze the coronal heating problem and proposed solutions
- Derive and apply Parker's solar wind model
- Characterize solar wind turbulence and its properties
- Implement numerical models of solar phenomena (Parker wind, flux tubes, butterfly diagrams)

## 1. Solar Structure and Magnetic Fields

### 1.1 Layers of the Sun

The Sun is a complex, stratified plasma system spanning many orders of magnitude in density, temperature, and magnetic field strength.

**Interior:**

1. **Core** (r < 0.25 R☉):
   - Temperature: `T ~ 1.5 × 10⁷ K`
   - Density: `ρ ~ 150 g/cm³`
   - Nuclear fusion: `4 ¹H → ⁴He + energy`
   - Energy transport: radiation (photon diffusion)

2. **Radiative zone** (0.25 R☉ < r < 0.7 R☉):
   - Temperature: `T ~ 10⁷ - 10⁶ K`
   - Radiative diffusion dominates
   - Stable stratification (no convection)
   - Differential rotation established

3. **Tachocline** (r ~ 0.7 R☉):
   - Thin shear layer (~0.05 R☉)
   - Transition from radiative (solid-body rotation) to convective (differential rotation)
   - Site of **strong Ω-effect** (toroidal field generation)
   - Critical for solar dynamo

4. **Convection zone** (0.7 R☉ < r < 1.0 R☉):
   - Temperature: `T ~ 10⁶ - 6000 K`
   - Density: `ρ ~ 1 - 10⁻⁷ g/cm³`
   - Convective energy transport (granulation, supergranulation)
   - Source of **α-effect** (helical turbulence)
   - Differential rotation: equator faster than poles

**Atmosphere:**

5. **Photosphere** (surface, r ~ R☉):
   - Temperature: `T ~ 5800 K` (visible surface)
   - Density: `ρ ~ 10⁻⁷ g/cm³`
   - Optical depth `τ ~ 1`
   - Sunspots, faculae, granulation visible

6. **Chromosphere** (R☉ < r < R☉ + 2000 km):
   - Temperature: `T ~ 6000 - 20000 K` (increases with height)
   - Density: `ρ ~ 10⁻⁹ - 10⁻¹¹ g/cm³`
   - Emission lines (Hα, Ca II K)
   - Spicules, plages

7. **Transition region** (r ~ R☉ + 2000 km):
   - Rapid temperature increase: `T ~ 10⁴ → 10⁶ K` over ~100 km
   - Density drops sharply

8. **Corona** (r > R☉ + 2000 km):
   - Temperature: `T ~ 1-3 × 10⁶ K` (mysteriously hot!)
   - Density: `ρ ~ 10⁻¹² - 10⁻¹⁶ g/cm³`
   - Highly ionized (Fe XIV, etc.)
   - Closed magnetic loops and open field lines
   - Solar wind originates

**Magnetic field:**

- **Interior (convection zone):** `B ~ 10-100 G` (turbulent dynamo)
- **Tachocline:** `B ~ 10⁴ G` (strong toroidal field)
- **Photosphere (quiet Sun):** `B ~ 1-10 G`
- **Photosphere (sunspots):** `B ~ 3000 G` (0.3 T!)
- **Corona:** `B ~ 1-10 G` (extends to large scale)

### 1.2 Observational Techniques

**Magnetography:**

- **Zeeman effect:** Splitting of spectral lines in magnetic field
  - Line splitting: `Δλ ∝ B` (longitudinal component)
  - Circular polarization (Stokes V): longitudinal field
  - Linear polarization (Stokes Q, U): transverse field

- **Instruments:**
  - **SOHO (Solar and Heliospheric Observatory):** MDI (Michelson Doppler Imager)
  - **SDO (Solar Dynamics Observatory):** HMI (Helioseismic and Magnetic Imager)
    - Full-disk magnetograms every 45 seconds
    - Vector magnetic field maps
  - **Hinode:** SOT (Solar Optical Telescope) — high-resolution
  - **Daniel K. Inouye Solar Telescope (DKIST):** 4-meter aperture, highest resolution

**Helioseismology:**

- Study of solar oscillations (5-minute oscillations, p-modes, g-modes)
- Infer internal structure: rotation profile, sound speed, magnetic field (indirectly)
- **Key result:** Tachocline discovered via differential rotation profile

**Coronagraphy and EUV imaging:**

- SOHO/LASCO: white-light coronagraph (coronal mass ejections)
- SDO/AIA: Extreme UV imaging at multiple wavelengths (different temperatures)
- Hinode/XRT: X-ray imaging (hot coronal loops)

## 2. Magnetic Flux Tubes and Sunspots

### 2.1 Thin Flux Tube Approximation

A **magnetic flux tube** is a bundle of magnetic field lines, often approximated as a slender tube embedded in a field-free plasma.

**Assumptions:**

1. Tube radius `a ≪ L` (length scale of variation)
2. Internal field `B_i` is axial, external field `B_e = 0` (or weak)
3. Pressure balance across tube boundary

**Pressure balance:**

Inside and outside the tube, total pressure must balance:

```
p_i + B_i² / (2μ₀) = p_e
```

If `B_e = 0`:

```
p_i = p_e - B_i² / (2μ₀)
```

Since `p_i < p_e`, the gas inside is **cooler or less dense** (for ideal gas, `p = ρkT/m`).

**Buoyancy:**

If the tube is in hydrostatic equilibrium in a stratified atmosphere:

```
dp/dz = -ρg
```

For an external medium:

```
dp_e/dz = -ρ_e g
```

For the tube:

```
dp_i/dz = -ρ_i g
```

Subtracting:

```
d/dz(p_i - p_e) = -(ρ_i - ρ_e)g
```

With `p_i - p_e = -B_i²/(2μ₀)`:

```
-d/dz(B_i²/(2μ₀)) = -(ρ_i - ρ_e)g
```

If `ρ_i < ρ_e` (tube is less dense), the tube experiences **buoyancy force**:

```
F_buoy = (ρ_e - ρ_i) g V
```

This drives **magnetic buoyancy instability** → flux tubes rise through convection zone.

### 2.2 Magnetic Buoyancy and Flux Emergence

**Magnetic buoyancy instability:**

A horizontal magnetic flux tube in a stratified layer is unstable to undulations if:

```
B² / (2μ₀ρ) > c_s² H_p / γ
```

where:
- `c_s` = sound speed
- `H_p = p/(ρg)` = pressure scale height
- `γ` = adiabatic index

For strong fields in the convection zone, this condition is satisfied → flux tubes rise.

**Rise dynamics:**

The rise speed of a flux tube balances buoyancy and drag:

```
ρ_e (dv_z/dt) = (ρ_e - ρ_i) g - C_D (1/2) ρ_e v_z²
```

where `C_D` is a drag coefficient.

In the convection zone:
- Timescale: `τ_rise ~ H_p / v_z ~ months`
- Tubes with `B ~ 10⁴ G` generated at tachocline rise to surface

**Emergence at photosphere:**

Upon reaching the photosphere, the flux tube emerges as **bipolar active regions**:
- Leading spot (one polarity)
- Trailing spot (opposite polarity)
- Connected by magnetic loops above surface

### 2.3 Sunspots

**Structure:**

- **Umbra:** Dark central region
  - Temperature: `T ~ 4000 K` (vs 5800 K for quiet Sun)
  - Magnetic field: `B ~ 3000 G`, nearly vertical
  - Darkness due to suppressed convection (magnetic pressure inhibits motions)

- **Penumbra:** Surrounding region with radial filaments
  - Temperature: `T ~ 5000 K`
  - Magnetic field: `B ~ 1500 G`, inclined
  - **Evershed flow:** Radial outflow at `v ~ 1-2 km/s` (along inclined field)

**Sunspot cycle:**

- Sunspot number varies with ~11-year period
- Correlated with solar activity (flares, CMEs)
- Magnetic polarity reverses each cycle → 22-year magnetic cycle

**Hale's polarity law:**

In a given cycle:
- Northern hemisphere leading spots: one polarity
- Southern hemisphere leading spots: opposite polarity
- Reverses in next cycle

**Joy's law:**

Bipolar active regions are tilted:
- Leading spot closer to equator
- Trailing spot toward pole
- Tilt angle increases with latitude

This tilt is crucial for the **Babcock-Leighton mechanism** (see dynamo section).

## 3. Solar Dynamo

### 3.1 Observational Constraints

The solar dynamo must explain:

1. **11-year sunspot cycle** (22-year magnetic cycle)
2. **Butterfly diagram:** Sunspots appear at mid-latitudes (~30°) at cycle start, migrate toward equator
3. **Hale's law:** Polarity reversal every 11 years
4. **Polar field reversal:** Polar field reverses ~11 years, in phase with equator
5. **Differential rotation:** Ω(latitude, radius) from helioseismology
6. **Tachocline:** Strong radial shear `∂Ω/∂r` at base of convection zone

### 3.2 Differential Rotation Profile

From helioseismology (e.g., GONG, SOHO/MDI, SDO/HMI):

**Surface (photosphere):**

```
Ω(θ) ≈ Ω_eq (1 - a₂ cos²θ - a₄ cos⁴θ)
```

where:
- `Ω_eq ≈ 2π / 25 days` (equatorial rotation)
- `a₂ ≈ 0.14`, `a₄ ≈ 0.17`
- Poles rotate ~30% slower than equator

**Interior:**

- **Radiative zone (r < 0.7 R☉):** Nearly **solid-body rotation** Ω ~ const
- **Convection zone:** Latitude-dependent rotation, roughly constant on radial lines (cylindrical rotation)
- **Tachocline (r ~ 0.7 R☉):** Sharp transition
  - Strong **radial shear** `∂Ω/∂r`
  - This is where toroidal field is amplified (Ω-effect)

### 3.3 α-Ω Dynamo Mechanism

The classical **α-Ω dynamo** operates in two steps:

**1. Ω-effect (Toroidal Field Generation):**

Differential rotation shears poloidal field `B_p` into toroidal field `B_φ`:

```
∂B_φ/∂t ≈ r sin θ (B_r ∂Ω/∂r + B_θ / r ∂Ω/∂θ)
```

In the tachocline, `∂Ω/∂r` is large → strong amplification:

```
B_φ ~ B_p × (Ω_cycle × τ_cycle) ~ B_p × (10⁻⁶ rad/s × 10⁸ s) ~ 100 B_p
```

Starting from `B_p ~ 10 G`, this generates `B_φ ~ 10³-10⁴ G`.

**2. α-effect (Poloidal Field Regeneration):**

Helical convective turbulence in the convection zone converts toroidal → poloidal:

```
∂B_p/∂t ≈ ∇ × (α B_φ e_φ)
```

Sources of α:
- **Cyclonic convection:** Coriolis force acting on rising/sinking plumes → helical motion
- **Kinetic helicity:** `⟨u·(∇×u)⟩ ≠ 0` in rotating, stratified fluid

**Feedback loop:**

```
B_p → (Ω-effect) → B_φ → (α-effect) → B_p'
```

If the amplification per cycle exceeds diffusion, the field grows.

### 3.4 Babcock-Leighton Mechanism

An alternative to the turbulent α-effect:

**Mechanism:**

1. Strong toroidal field `B_φ` at tachocline becomes buoyantly unstable
2. Flux tubes rise and emerge as **tilted bipolar active regions** (Joy's law)
3. At the surface, following spot (poleward) has opposite polarity to preceding spot
4. Surface flows (meridional circulation, diffusion) transport flux:
   - Poleward migration of following-polarity flux → cancels and reverses polar field
   - This is equivalent to regenerating poloidal field from toroidal

**Poloidal field source:**

```
Source ∝ B_φ × sin(tilt angle) × (surface flux transport)
```

This is a **non-local α-effect** operating at the surface, not in the bulk.

**Advantages:**

- Directly observable (tilt, emergence)
- Explains polar field reversal
- Flux-transport dynamo models reproduce butterfly diagram well

### 3.5 Flux-Transport Dynamo

Modern solar dynamo models combine:

- **Ω-effect** at tachocline
- **Babcock-Leighton α-effect** at surface
- **Meridional circulation:** Poleward at surface, equatorward at base (conveyor belt)
  - Speed: `v_m ~ 10-20 m/s`
  - Period: `τ_m ~ R☉ / v_m ~ 10 years` (comparable to cycle!)
- **Turbulent diffusion:** In convection zone, `η_t ~ 10^{10-11} cm²/s`

**Cycle:**

1. Toroidal field `B_φ` builds up at tachocline (Ω-effect)
2. Flux emergence → tilted active regions at mid-latitudes
3. Surface diffusion + meridional flow → poleward flux migration, polar field reversal
4. Meridional flow advects surface poloidal field downward at poles
5. At tachocline, poloidal field is sheared by Ω → new toroidal field (opposite sign)
6. Cycle repeats

**Period:**

```
P ~ 2π √(D/α_eff Ω_eff)  or  P ~ circulation time
```

Tuning parameters (diffusivity, circulation speed) reproduces 11-year cycle.

### 3.6 Dynamo Saturation and Magnetic Activity

**Saturation:**

- Magnetic field cannot grow indefinitely
- **Magnetic buoyancy:** When `B_φ ~ 10⁴ G`, tubes become buoyant → rise and emerge → removes flux from tachocline
- **Lorentz force:** Strong field modifies flow (reduces shear, α-effect)

**Cyclic behavior:**

Saturation leads to **oscillatory solutions** rather than steady dynamo:
- Poloidal and toroidal fields oscillate with phase lag
- Butterfly diagram: equatorward migration due to dynamo wave propagation or flux transport

**Grand minima (Maunder Minimum):**

- Historical periods of very low sunspot activity (e.g., 1645-1715)
- Possibly due to:
  - Stochastic fluctuations in α-effect
  - Changes in meridional circulation
  - Transition between dynamo modes

Models with stochastic forcing can reproduce irregular cycles and grand minima.

## 4. Coronal Heating Problem

### 4.1 The Problem

**Observation:**

- Photosphere: `T ~ 5800 K`
- Chromosphere: `T ~ 10⁴ K`
- Corona: `T ~ 10⁶ K`

This **violates naive expectation** (temperature should decrease with distance from heat source).

**Energy budget:**

To maintain coronal temperature against radiative and conductive losses:

```
Energy input ≈ 10² - 10³ W/m² (in active regions)
             ≈ 10-100 W/m² (quiet Sun)
```

Total coronal heating power: `L_corona ~ 10²⁷ erg/s ~ 10²⁰ W` (compare to solar luminosity `L_☉ ~ 4×10²⁶ W`, so corona is `~0.01% L_☉`).

**Possible energy sources:**

- Photospheric motions (convection, granulation): `~ kW/m²` (ample energy!)
- Question: How is this energy **transported to the corona** and **dissipated** as heat?

### 4.2 Wave Heating Mechanisms

**Acoustic waves:**

- Convective motions generate sound waves
- Waves propagate upward, steepen into shocks (due to decreasing density)
- Shock dissipation → heating

**Problem:** Acoustic waves are mostly **reflected** at the steep temperature gradient (chromosphere/transition region) → insufficient flux reaches corona.

**Alfvén waves:**

Magnetic field lines are like strings that can support **Alfvén waves**:

```
v_A = B / √(μ₀ ρ)
```

In the corona, `B ~ 10 G`, `ρ ~ 10^{-12} g/cm³` → `v_A ~ 1000 km/s`.

**Wave generation:**

- Photospheric motions (granulation) shake magnetic field lines
- Excites Alfvén waves (transverse oscillations)
- Waves propagate along field lines into corona

**Dissipation mechanisms:**

1. **Phase mixing:**
   - Different field lines have different `v_A` → waves on adjacent lines get out of phase
   - Generates transverse gradients → viscous/resistive dissipation

2. **Resonant absorption:**
   - In stratified corona, local Alfvén frequency varies
   - Waves couple resonantly to localized oscillations → enhanced dissipation

3. **Turbulent cascade:**
   - Nonlinear interactions → energy cascades to small scales
   - Dissipates via resistivity or viscosity at small scales

**Energy flux:**

Alfvén wave energy flux:

```
F_A = (1/2) ρ v_A δv²
```

where `δv` is wave amplitude. For `δv ~ 10 km/s`, `ρ ~ 10^{-7} g/cm³` (chromosphere), `v_A ~ 100 km/s`:

```
F_A ~ 10³ W/m²  (sufficient for coronal heating!)
```

**Observational support:**

- Alfvénic fluctuations observed in corona (CoMP instrument, Parker Solar Probe)
- Waves seen propagating along coronal loops
- Damping timescales consistent with heating requirements

### 4.3 Magnetic Reconnection and Nanoflares

**Parker's nanoflare hypothesis (1988):**

- Photospheric motions continuously tangle coronal magnetic field
- Small-scale **current sheets** form
- Magnetic reconnection in these sheets releases energy in bursts (**nanoflares**)
- Many small events (each `E ~ 10²⁴ erg`, vs `10^{32}` erg for large flares) → statistical heating

**Energy release:**

Reconnection converts magnetic energy to kinetic and thermal energy:

```
Energy ~ B² / (2μ₀) × Volume
```

For a current sheet of size `L ~ 100 km`, `B ~ 10 G`:

```
E ~ (10⁻³ T)² / (2×4π×10⁻⁷) × (10⁵ m)³ ~ 10²⁴ erg  (nanoflare)
```

**Frequency:**

To supply `10²⁰ W` to the corona:

```
N × E / τ ~ 10²⁰ W
N ~ 10²⁰ W × τ / 10²⁴ erg ~ 10⁶ events/s  (if τ ~ 100 s)
```

**Challenges:**

- Individual nanoflares are below detection threshold
- Need to observe **statistical signatures** (e.g., non-thermal emission, heating rate distributions)

**Observational evidence:**

- EUV brightenings consistent with nanoflare statistics
- High-temperature (> 3 MK) plasma in active regions suggests impulsive heating
- But direct observation of individual nanoflares remains elusive

### 4.4 Current Understanding

Likely **both mechanisms contribute**:

- **Wave heating:** In quiet Sun, open field regions (coronal holes)
  - Alfvén waves propagate outward, dissipate
  - Relevant for slow and fast solar wind acceleration

- **Reconnection/nanoflares:** In active regions, closed loops
  - Braiding and reconnection in complex magnetic structures
  - Explains high-temperature emission

**Ongoing research:**

- High-resolution observations (DKIST, Solar Orbiter, Parker Solar Probe)
- MHD simulations of realistic coronal magnetic field
- Modeling wave propagation and dissipation
- Statistics of small-scale reconnection events

## 5. Solar Wind

### 5.1 Parker's Solar Wind Model (1958)

Eugene Parker proposed that the corona is not in **hydrostatic equilibrium** but instead expands supersonically into interplanetary space.

**Assumptions:**

1. **Steady, spherically symmetric, radial flow**
2. **Isothermal corona:** `T = T_c = const` (simplified)
3. **Ideal gas:** `p = ρkT/m`
4. **Energy equation:** Neglect heating/cooling (or assume balance)

**Equations:**

**Mass conservation:**

```
d/dr (ρ v r²) = 0  →  ρ v r² = const = Ṁ/(4π)
```

where `Ṁ` is mass loss rate.

**Momentum equation:**

```
ρ v dv/dr = -dp/dr - GMρ/r²
```

**Ideal gas:**

```
p = ρ c_s²
```

where `c_s² = kT/m` is sound speed squared.

**Combining:**

Substitute `ρ = Ṁ/(4πr²v)` and `p = (Ṁ/(4πr²v)) c_s²`:

```
v dv/dr = -c_s² (1/v dv/dr + 2/r) - GM/r²
```

Rearranging:

```
(v² - c_s²) (1/v dv/dr) = -(2c_s²/r + GM/r²)
```

**Critical point:**

At `r = r_c` where `v = c_s` (sonic point), the right-hand side must vanish:

```
2c_s²/r_c + GM/r_c² = 0  →  r_c = GM/(2c_s²)
```

For the Sun with `T_c = 10⁶ K`, `c_s ~ 100 km/s`:

```
r_c ~ (6.67×10⁻¹¹ × 2×10³⁰) / (2 × (10⁵)²) ~ 7×10⁹ m ~ 10 R☉
```

**Solution:**

The **transonic solution** passes through the critical point:
- **Subsonic** for `r < r_c` (near Sun)
- **Sonic** at `r = r_c`
- **Supersonic** for `r > r_c` (interplanetary space)

At Earth's orbit (1 AU ~ 215 R☉):
- Speed: `v ~ 400 km/s`
- Density: `n ~ 5 cm⁻³`
- Temperature: `T ~ 10⁵ K`

These match observations → Parker's model confirmed!

### 5.2 Non-Isothermal Models

**Problem with isothermal model:**

Energy conservation is not satisfied (would require continuous heating to maintain `T = const`).

**Polytropic model:**

Assume `p ∝ ρ^γ`:

```
p = K ρ^γ
```

where `γ` is polytropic index (γ=1 isothermal, γ=5/3 adiabatic).

For `γ < 3/2`, transonic solutions exist.

**Energy equation:**

More realistic models include:
- **Thermal conduction:** `q = -κ dT/dr`
- **Radiative cooling**
- **Coronal heating** (ad hoc function or wave dissipation)
- **Gravity work**

Full energy equation:

```
d/dr (r² ρ v [v²/2 + γ/(γ-1) p/ρ - GM/r]) = r² (Q_heat - Q_cool + Q_cond)
```

Numerical solutions find:
- Temperature peak near `r ~ 1-2 R☉`, then decreases
- Terminal speed depends on coronal temperature and heating profile

### 5.3 Fast and Slow Solar Wind

**Observations (Ulysses, Helios, Wind, ACE, Parker Solar Probe):**

Two types of solar wind:

**Slow solar wind:**
- Speed: `v ~ 300-400 km/s`
- Density: `n ~ 10 cm⁻³` at 1 AU
- Temperature: `T ~ 10⁵ K`
- Composition: Similar to photosphere, but enriched in low-FIP elements
- Source: **Streamer belt** near solar equator (closed-field regions)

**Fast solar wind:**
- Speed: `v ~ 700-800 km/s`
- Density: `n ~ 3 cm⁻³` at 1 AU
- Temperature: `T ~ 2×10⁵ K`
- Composition: Closer to photospheric
- Source: **Coronal holes** (open-field regions at poles)

**Acceleration mechanisms:**

- **Slow wind:** Possibly from reconnection at streamer tops, or waves in complex field
- **Fast wind:** Alfvén wave pressure (wave energy density acts as effective pressure)

**Alfvén wave-driven models:**

Wave pressure gradient accelerates wind:

```
F_wave ~ -d/dr (B δB / μ₀)
```

where `δB` is wave amplitude. This can drive fast wind to observed speeds.

### 5.4 Solar Wind Turbulence

**Observations:**

- Magnetic field and velocity fluctuations are highly turbulent
- Power spectra: `E(f) ∝ f^{-5/3}` (Kolmogorov-like) at MHD scales
- Steepening at kinetic scales (ion gyroradius, inertial length)

**Characteristics:**

- **Alfvénic correlations:** `δv ~ ±δB/√(μ₀ρ)` (outward/inward propagating Alfvén waves)
- **Anisotropy:** Fluctuations preferentially perpendicular to mean field
- **Intermittency:** Non-Gaussian statistics, coherent structures (current sheets, flux ropes)

**Energy cascade:**

- Injection at large scales (> 10⁶ km at 1 AU)
- Inertial range: `E(k) ∝ k^{-5/3}` (or `k^{-3/2}` Iroshnikov-Kraichnan)
- Dissipation at `k ρ_i ~ 1` (ion gyroscale) or `k λ_i ~ 1` (ion inertial length)

**Heating:**

Turbulent dissipation heats the solar wind:
- Proton temperature decreases slower than adiabatic (`T ∝ r^{-α}` with `α ~ 0.5` vs adiabatic `α ~ 4/3`)
- Perpendicular ion heating (temperature anisotropy)
- Electron heating at small scales

**Implications:**

- Solar wind is a natural laboratory for collisionless plasma turbulence
- Relevant for astrophysical plasmas (ISM, galaxy clusters, accretion flows)

## 6. Python Implementations

### 6.1 Parker Solar Wind Solution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

def parker_solar_wind():
    """
    Solve Parker's isothermal solar wind model.

    Equations:
      Mass: ρ v r² = const
      Momentum: v dv/dr = -c_s²/ρ dρ/dr - GM/r²

    Combining:
      (v² - c_s²) (1/v dv/dr) = -(2c_s²/r + GM/r²)
    """
    # Constants
    G = 6.674e-11      # m^3 kg^-1 s^-2
    M_sun = 1.989e30   # kg
    R_sun = 6.96e8     # m
    k_B = 1.38e-23     # J/K
    m_p = 1.67e-27     # kg (proton mass)

    # Coronal temperature
    T_corona = 1.5e6   # K
    # c_s = √(kT/m_p): the isothermal sound speed sets the scale of the
    # problem because the critical radius r_c = GM/(2c_s²) — a hotter
    # corona moves the sonic point inward, enabling faster acceleration
    # and higher terminal wind speed.
    c_s = np.sqrt(k_B * T_corona / m_p)  # Sound speed

    # Critical radius
    # r_c is where the flow transitions from subsonic to supersonic;
    # at this point the Parker equation is degenerate (0/0), so the
    # solution must pass through it smoothly — the "transonic" condition
    # that selects the physically correct branch among all possible solutions.
    r_c = G * M_sun / (2 * c_s**2)

    print(f"Sound speed: {c_s/1e3:.1f} km/s")
    print(f"Critical radius: {r_c/R_sun:.2f} R_sun")

    # Radial grid
    # Logarithmic spacing captures both the near-Sun acceleration region
    # (where dv/dr is large) and the distant solar wind (where v is nearly
    # constant) with equal fractional resolution at each radius.
    r = np.logspace(np.log10(R_sun), np.log10(215*R_sun), 1000)  # 1 R_sun to 1 AU

    # Define ODE: dv/dr
    def dv_dr(v, r):
        numerator = -(2*c_s**2 / r + G*M_sun / r**2)
        denominator = (v**2 - c_s**2) / v
        if abs(denominator) < 1e-10:
            return 0  # Near critical point
        return numerator / denominator

    # Find critical point solution
    # Start slightly supersonic just past r_c
    # We start at 1.01 r_c rather than exactly at r_c because the ODE has
    # a removable singularity at the critical point; a tiny offset lets the
    # integrator step past it without encountering a zero denominator.
    r_start_idx = np.argmin(np.abs(r - r_c * 1.01))
    r_start = r[r_start_idx]
    v_start = c_s * 1.01

    # Integrate outward from r_c
    r_out = r[r_start_idx:]
    v_out = odeint(dv_dr, v_start, r_out).flatten()

    # Integrate inward from r_c
    # Integrating both directions from r_c and then joining the pieces
    # guarantees we follow the unique transonic solution; all other
    # branches either stall subsonically or diverge supersonically before
    # reaching the corona, and are therefore unphysical.
    r_in = r[:r_start_idx+1][::-1]
    v_in_rev = odeint(dv_dr, c_s * 0.99, r_in).flatten()
    v_in = v_in_rev[::-1]

    # Combine
    r_sol = np.concatenate([r_in[:-1], r_out])
    v_sol = np.concatenate([v_in[:-1], v_out])

    # Density from mass conservation
    # ρ v r² = ρ_c c_s r_c²
    # This is the integrated continuity equation (Ṁ = const); as v increases
    # and r² grows, ρ must drop steeply — explaining why the solar wind is
    # a million times less dense at Earth than at the coronal base.
    rho_c = 1e-12  # kg/m^3 (arbitrary normalization)
    rho_sol = rho_c * c_s * r_c**2 / (v_sol * r_sol**2)

    # Number density
    n_sol = rho_sol / m_p  # m^-3

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Velocity
    ax1.plot(r_sol/R_sun, v_sol/1e3, 'b-', linewidth=2)
    ax1.axhline(c_s/1e3, color='r', linestyle='--', label='Sound speed')
    ax1.axvline(r_c/R_sun, color='g', linestyle='--', label='Critical radius')
    ax1.axvline(215, color='orange', linestyle='--', label='1 AU')
    ax1.set_xlabel('Radius ($R_\\odot$)', fontsize=12)
    ax1.set_ylabel('Velocity (km/s)', fontsize=12)
    ax1.set_title('Parker Solar Wind: Velocity Profile', fontsize=14)
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Density
    ax2.semilogy(r_sol/R_sun, n_sol/1e6, 'b-', linewidth=2)  # cm^-3
    ax2.axvline(r_c/R_sun, color='g', linestyle='--')
    ax2.axvline(215, color='orange', linestyle='--')
    ax2.set_xlabel('Radius ($R_\\odot$)', fontsize=12)
    ax2.set_ylabel('Number density (cm$^{-3}$)', fontsize=12)
    ax2.set_title('Parker Solar Wind: Density Profile', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    # Mach number
    M = v_sol / c_s
    ax3.plot(r_sol/R_sun, M, 'b-', linewidth=2)
    ax3.axhline(1, color='r', linestyle='--', label='M = 1')
    ax3.axvline(r_c/R_sun, color='g', linestyle='--')
    ax3.axvline(215, color='orange', linestyle='--')
    ax3.set_xlabel('Radius ($R_\\odot$)', fontsize=12)
    ax3.set_ylabel('Mach number', fontsize=12)
    ax3.set_title('Parker Solar Wind: Mach Number', fontsize=14)
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parker_solar_wind.png', dpi=150)
    plt.show()

    # Print values at 1 AU
    idx_1AU = np.argmin(np.abs(r_sol - 215*R_sun))
    print(f"\nAt 1 AU:")
    print(f"  Velocity: {v_sol[idx_1AU]/1e3:.1f} km/s")
    print(f"  Density: {n_sol[idx_1AU]/1e6:.2f} cm^-3")
    print(f"  Mach number: {M[idx_1AU]:.2f}")

parker_solar_wind()
```

### 6.2 Flux Tube Rise Calculation

```python
import numpy as np
import matplotlib.pyplot as plt

def flux_tube_rise():
    """
    Calculate the rise of a magnetic flux tube through the solar convection zone.

    Buoyancy equation:
      dv_z/dt = (Δρ/ρ_e) g - C_D v_z² / (2H_p)

    where Δρ = ρ_e - ρ_i (density deficit).
    """
    # Solar parameters
    g = 274  # m/s^2 (surface gravity)
    R_sun = 6.96e8  # m
    M_sun = 1.989e30  # kg

    # Convection zone
    r_bottom = 0.7 * R_sun  # Base (tachocline)
    r_top = R_sun           # Photosphere

    # Stratification (simplified exponential)
    # An exponential density profile with scale height H_p gives the
    # correct order-of-magnitude variation across the convection zone;
    # it also makes the buoyancy force larger near the tachocline (where
    # ρ_e is high) and smaller near the surface (where ρ_e drops sharply).
    H_p = 5e7  # Pressure scale height ~ 50 Mm
    rho_0 = 1e-1  # kg/m^3 (at r_bottom, approximate)

    def rho_ext(r):
        """External density (stratified atmosphere)."""
        return rho_0 * np.exp(-(r - r_bottom) / H_p)

    # Magnetic field strength
    B = 3e4 * 1e-4  # 30 kG in Tesla

    # Internal density: pressure balance
    # p_i + B²/(2μ₀) = p_e
    # Assume isothermal: p = ρ c_s²
    c_s = 1e4  # m/s (sound speed)
    mu_0 = 4*np.pi*1e-7

    def rho_int(rho_e, B):
        """Internal density from pressure balance."""
        # The flux tube interior must have lower gas pressure than the
        # exterior to balance the added magnetic pressure B²/(2μ₀); for
        # an isothermal gas (p=ρc_s²) this directly means ρ_i < ρ_e,
        # which is the source of the buoyancy driving the tube upward.
        p_e = rho_e * c_s**2
        p_i = p_e - B**2 / (2*mu_0)
        return p_i / c_s**2

    # Drag coefficient
    C_D = 0.5

    # Time evolution
    dt = 100  # seconds
    t_max = 1e7  # seconds (~100 days)
    Nt = int(t_max / dt)

    # Arrays
    t = np.zeros(Nt)
    z = np.zeros(Nt)  # Height above r_bottom
    v_z = np.zeros(Nt)

    # Initial conditions
    z[0] = 1e6  # Start 1 Mm above tachocline
    v_z[0] = 0

    # Integrate
    for n in range(Nt - 1):
        r = r_bottom + z[n]
        rho_e = rho_ext(r)
        rho_i = rho_int(rho_e, B)

        Delta_rho = rho_e - rho_i

        # Buoyancy acceleration
        # a_buoy = (Δρ/ρ_e) g is Archimedes' law in a compressible fluid:
        # the fractional density deficit times g gives the net upward
        # force per unit mass.  A strong field (large B) creates a large
        # Δρ and therefore vigorous buoyancy.
        a_buoy = (Delta_rho / rho_e) * g

        # Drag
        # The v_z²/(2H_p) form reflects aerodynamic drag normalized by
        # the pressure scale height: at terminal velocity the drag exactly
        # balances buoyancy, limiting how fast tubes can rise through
        # the convection zone and thus constraining the rise time.
        a_drag = -C_D * v_z[n]**2 / (2 * H_p)

        # Update velocity
        dv_dt = a_buoy + a_drag
        v_z[n+1] = v_z[n] + dt * dv_dt

        # Update position
        z[n+1] = z[n] + dt * v_z[n+1]

        # Store time
        t[n+1] = t[n] + dt

        # Stop if reaches surface
        if z[n+1] >= (r_top - r_bottom):
            z = z[:n+2]
            v_z = v_z[:n+2]
            t = t[:n+2]
            break

    # Convert to Mm and days
    z_Mm = z / 1e6
    t_days = t / 86400
    v_km_s = v_z / 1e3

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(t_days, z_Mm, 'b-', linewidth=2)
    ax1.axhline((r_top - r_bottom)/1e6, color='r', linestyle='--', label='Photosphere')
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.set_ylabel('Height above tachocline (Mm)', fontsize=12)
    ax1.set_title('Flux Tube Rise: Height vs Time', fontsize=14)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t_days, v_km_s, 'b-', linewidth=2)
    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.set_ylabel('Rise velocity (km/s)', fontsize=12)
    ax2.set_title('Flux Tube Rise: Velocity vs Time', fontsize=14)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('flux_tube_rise.png', dpi=150)
    plt.show()

    print(f"Rise time to photosphere: {t_days[-1]:.1f} days")
    print(f"Final velocity: {v_km_s[-1]:.2f} km/s")

flux_tube_rise()
```

### 6.3 Butterfly Diagram from Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt

def butterfly_diagram_synthetic():
    """
    Generate a synthetic solar butterfly diagram.

    Model:
      - Sunspot latitude: λ(t) = λ_max cos(2π t / P) exp(-t / τ_decay)
      - Equatorward migration with cycle
    """
    # Parameters
    P = 11  # years (cycle period)
    N_cycles = 3

    # Time array
    t = np.linspace(0, N_cycles * P, 1000)

    # Generate butterfly pattern
    # Latitude of sunspot emergence
    lambda_max = 35  # degrees (max latitude)
    tau_decay = P / 2  # decay timescale

    # For each cycle
    latitudes_N = []
    latitudes_S = []
    times = []

    for cycle in range(N_cycles):
        t_cycle = t[(t >= cycle*P) & (t < (cycle+1)*P)] - cycle*P

        # Latitude decreases from λ_max to ~5° during cycle
        lambda_t = lambda_max * (1 - t_cycle / P) + 5

        # Add some scatter
        N_spots = 200
        t_spots = np.random.uniform(0, P, N_spots)
        lambda_spots = lambda_max * (1 - t_spots / P) + 5 + np.random.normal(0, 3, N_spots)

        # Polarity reverses each cycle
        if cycle % 2 == 0:
            latitudes_N.extend(lambda_spots)
            latitudes_S.extend(-lambda_spots)
        else:
            latitudes_N.extend(-lambda_spots)
            latitudes_S.extend(lambda_spots)

        times.extend(t_spots + cycle * P)

    # Combine
    latitudes = np.concatenate([latitudes_N, latitudes_S])
    times_all = np.concatenate([times, times])

    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(times_all, latitudes, s=1, c='k', alpha=0.5)
    plt.axhline(0, color='r', linestyle='--', linewidth=0.5)
    for cycle in range(N_cycles + 1):
        plt.axvline(cycle * P, color='gray', linestyle='--', linewidth=0.5)

    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Latitude (degrees)', fontsize=14)
    plt.title('Synthetic Solar Butterfly Diagram', fontsize=16)
    plt.ylim(-40, 40)
    plt.grid(True, alpha=0.3)
    plt.savefig('butterfly_diagram_synthetic.png', dpi=150)
    plt.show()

butterfly_diagram_synthetic()
```

## 7. Summary

**Solar MHD** encompasses a rich variety of phenomena driven by magnetic fields:

1. **Solar structure:**
   - Convection zone (dynamo), tachocline (toroidal field generation), corona (mysteriously hot)
   - Magnetic field ranges from ~1 G (quiet Sun) to ~3000 G (sunspots)

2. **Magnetic flux tubes and sunspots:**
   - **Thin flux tube:** Pressure balance `p_i + B²/(2μ₀) = p_e`
   - **Magnetic buoyancy:** Less dense tubes rise through convection zone
   - **Sunspots:** Concentrations of strong vertical field, suppressed convection → cooler, darker

3. **Solar dynamo:**
   - **α-Ω mechanism:** Differential rotation (Ω) + helical turbulence (α) → cyclic field generation
   - **Babcock-Leighton:** Surface poloidal source from tilted bipolar regions
   - **Flux-transport dynamo:** Meridional circulation plays key role
   - **11-year cycle:** Butterfly diagram (equatorward migration), polar field reversal

4. **Coronal heating:**
   - **Problem:** Corona at ~10⁶ K, far above photosphere
   - **Wave heating:** Alfvén waves (phase mixing, resonant absorption, turbulence)
   - **Nanoflares:** Many small reconnection events (Parker's hypothesis)
   - Likely combination of both

5. **Solar wind:**
   - **Parker model:** Transonic expansion from hot corona
   - **Critical point:** `r_c ~ 10 R☉`, flow becomes supersonic
   - **Fast wind:** From coronal holes (~700 km/s)
   - **Slow wind:** From streamer belt (~400 km/s)
   - **Turbulence:** Alfvénic fluctuations, cascade, heating

The Sun is a natural laboratory for studying MHD processes, with applications to other stars, accretion disks, and planetary magnetospheres.

## Practice Problems

1. **Pressure Balance:** A sunspot has `B = 3000 G`. If the external gas pressure is `p_e = 10⁴ Pa`, what is the internal gas pressure `p_i`?

2. **Magnetic Buoyancy:** For a flux tube with `B = 10 kG` in the convection zone where `ρ_e = 0.1 g/cm³`, `c_s = 10 km/s`, compute the density deficit `Δρ = ρ_e - ρ_i` assuming isothermal pressure balance.

3. **Rise Time:** Using the density deficit from Problem 2 and `g = 274 m/s²`, estimate the buoyancy acceleration. How long does it take to rise 200 Mm (thickness of convection zone) if terminal velocity is reached?

4. **Solar Dynamo Number:** For `α = 0.1 m/s`, `ΔΩ = 10⁻⁶ rad/s`, `R = 5×10⁸ m`, `η_eff = 10¹⁰ cm²/s`, compute `D_αΩ`.

5. **Alfvén Speed in Corona:** For `B = 5 G`, `n = 10⁸ cm⁻³` (protons), compute the Alfvén speed in km/s.

6. **Critical Radius:** For coronal temperature `T = 2×10⁶ K`, compute the critical radius `r_c` in units of `R☉`.

7. **Solar Wind at 1 AU:** Using Parker's model with `T = 1.5×10⁶ K`, estimate the speed at 1 AU (215 R☉). Compare to observed ~400 km/s.

8. **Nanoflare Energy:** A current sheet of size `L = 100 km` with `B = 10 G` reconnects. Estimate the energy released in ergs.

9. **Python Exercise:** Modify the Parker solar wind code to include a polytropic relation `p ∝ ρ^{1.2}` instead of isothermal. How does the terminal speed change?

10. **Advanced:** Implement a simple 1D flux-transport dynamo model with Ω-effect at tachocline and Babcock-Leighton α-effect at surface. Include meridional circulation and diffusion. Reproduce a butterfly diagram.

---

**Previous:** [Turbulent Dynamo](./10_Turbulent_Dynamo.md) | **Next:** [Accretion Disk MHD](./12_Accretion_Disk_MHD.md)

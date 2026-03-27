# Solar Interior

[← Previous: Overview](00_Overview.md) | [Next: Nuclear Energy Generation →](02_Nuclear_Energy_Generation.md)

## Learning Objectives

1. Describe the Sun's fundamental properties (mass, radius, luminosity, age, composition) and their observational basis
2. Derive the equation of hydrostatic equilibrium and combine it with mass continuity to constrain the solar interior
3. Explain radiative energy transport, the role of opacity, and the photon random walk
4. State the Schwarzschild criterion for convective instability and describe mixing length theory
5. Characterize the tachocline as a rotational shear layer and explain its significance for the solar dynamo
6. Outline the inputs and predictions of the Standard Solar Model

---

## Why This Matters

The Sun's interior is forever hidden from direct observation — no telescope can peer beneath the photosphere. Yet understanding the interior is essential: it is where the Sun's energy originates, where its magnetic field is generated, and where the physical conditions (temperature, density, pressure) set the stage for everything we observe at the surface and beyond.

Solar interior physics draws on some of the deepest ideas in physics: nuclear reactions from quantum mechanics, radiative transfer from electrodynamics, convection from fluid dynamics, and equation of state from statistical mechanics. The Standard Solar Model — a numerical solution to the coupled equations of stellar structure — has been spectacularly validated by helioseismology (Lesson 03) and neutrino observations (Lesson 02), making it one of the most precisely tested models in astrophysics.

> **Analogy**: The solar interior is like the engine room of a ship. Passengers on deck (observers at the surface) cannot see the engines, but every vibration felt on deck, every change in speed, originates there. Helioseismology is like an ultrasound scan — using sound waves to image what lies beneath.

---

## 1. The Sun in Numbers

### Fundamental Parameters

The Sun's basic properties, determined through centuries of observation:

| Property | Value | How Measured |
|----------|-------|-------------|
| Mass ($M_\odot$) | $1.989 \times 10^{30}$ kg | Kepler's third law (planetary orbits) |
| Radius ($R_\odot$) | $6.957 \times 10^{8}$ m | Angular diameter + distance |
| Luminosity ($L_\odot$) | $3.828 \times 10^{26}$ W | Solar constant $\times$ $4\pi d^2$ |
| Effective temperature ($T_\text{eff}$) | 5778 K | Stefan-Boltzmann: $L = 4\pi R^2 \sigma T_\text{eff}^4$ |
| Age | $4.57 \times 10^9$ years | Meteorite radiometric dating |
| Surface gravity ($g_\odot$) | 274 m/s$^2$ | $GM/R^2$ |
| Escape velocity | 617.5 km/s | $\sqrt{2GM/R}$ |

### Composition

The Sun's composition by mass fraction:

- **Hydrogen (X)**: 73.5% — the primary fuel for nuclear fusion
- **Helium (Y)**: 24.9% — the fusion product (and primordial component)
- **Metals (Z)**: 1.6% — all elements heavier than helium (dominated by O, C, Ne, Fe)

These values refer to the present-day photospheric composition. The core has a lower X (more H has been converted to He) and higher Y than the surface. The precise metallicity (Z) has been a subject of active debate since the revision of solar abundances by Asplund et al. (2009), which lowered the O, C, and Ne abundances and created tension with helioseismology — the so-called **solar abundance problem**.

### Internal Structure Overview

The Sun's interior consists of distinct regions, defined by the dominant energy transport mechanism:

```
                    ┌──────────────┐
                    │   Corona     │  T ~ 10⁶ K
                    ├──────────────┤
                    │  Photosphere │  T ~ 5,800 K      ← visible surface
            ┌───────┼──────────────┤
            │       │  Convective  │  0.71-1.0 R☉     ← energy by convection
            │       │    Zone      │  T: 2×10⁶ → 5,800 K
            │  ┌────┼──────────────┤
            │  │    │  Tachocline  │  ~0.71 R☉        ← rotational shear layer
            │  │    ├──────────────┤
    Interior│  │    │  Radiative   │  0.25-0.71 R☉    ← energy by photon diffusion
            │  │    │    Zone      │  T: 7×10⁶ → 2×10⁶ K
            │  │    ├──────────────┤
            │  └────│    Core      │  0-0.25 R☉       ← nuclear fusion
            │       │              │  T: 1.57×10⁷ K center
            └───────┴──────────────┘
```

---

## 2. Hydrostatic Equilibrium

### The Fundamental Balance

The Sun is neither expanding nor contracting (on timescales much shorter than its nuclear evolution timescale). This means the outward pressure gradient force exactly balances the inward gravitational pull at every point. This is **hydrostatic equilibrium**.

Consider a thin spherical shell at radius $r$ with thickness $dr$, density $\rho(r)$, and pressure $P(r)$. The forces on this shell are:

- **Pressure gradient** (outward): $-\frac{dP}{dr} \cdot 4\pi r^2 \, dr$ (the pressure decreases outward, so $dP/dr < 0$, making this force outward)
- **Gravity** (inward): $-\frac{G m(r) \rho(r)}{r^2} \cdot 4\pi r^2 \, dr$

Setting the net force to zero:

$$\boxed{\frac{dP}{dr} = -\frac{G m(r) \rho(r)}{r^2}}$$

where $m(r)$ is the mass enclosed within radius $r$. This is the **equation of hydrostatic equilibrium** — one of the four fundamental equations of stellar structure.

> **Physical intuition**: Every layer of the Sun is like a balloon held in place by a tug-of-war between the weight of everything above it pushing down and the pressure from below pushing up. If gravity won, the Sun would collapse (this takes about 30 minutes — the **Kelvin-Helmholtz timescale**). If pressure won, the Sun would expand. The fact that neither happens tells us these forces are precisely balanced.

### Mass Continuity

The mass enclosed within radius $r$ satisfies:

$$\boxed{\frac{dm}{dr} = 4\pi r^2 \rho(r)}$$

This is simply the statement that the mass of a thin shell equals its volume ($4\pi r^2 \, dr$) times its density ($\rho$).

### Central Pressure Estimate

We can estimate the central pressure by approximating the hydrostatic equation with average values. Taking $\rho \approx \bar{\rho} = M_\odot / (\frac{4}{3}\pi R_\odot^3)$, $m(r) \approx M_\odot/2$, and $r \approx R_\odot/2$:

$$P_c \approx \frac{G M_\odot \bar{\rho}}{2} \cdot \frac{R_\odot}{R_\odot^2} = \frac{G M_\odot^2}{8\pi R_\odot^4}$$

Plugging in numbers:

$$P_c \approx \frac{(6.674 \times 10^{-11})(1.989 \times 10^{30})^2}{8\pi (6.957 \times 10^8)^4} \approx 4.5 \times 10^{13} \text{ Pa}$$

The actual central pressure from the Standard Solar Model is $P_c \approx 2.5 \times 10^{16}$ Pa — about 500 times larger than our crude estimate. The discrepancy arises because the real density profile is strongly centrally concentrated (the core density is $\sim 150$ g/cm$^3$, much higher than the mean density of $\sim 1.4$ g/cm$^3$).

### Central Temperature Estimate

Using the ideal gas law $P = \rho k_B T / (\mu m_H)$, where $\mu \approx 0.6$ is the mean molecular weight in the fully ionized core and $m_H$ is the hydrogen mass:

$$T_c = \frac{P_c \mu m_H}{\rho_c k_B}$$

With $P_c \approx 2.5 \times 10^{16}$ Pa and $\rho_c \approx 1.5 \times 10^5$ kg/m$^3$:

$$T_c \approx \frac{(2.5 \times 10^{16})(0.6)(1.67 \times 10^{-27})}{(1.5 \times 10^5)(1.381 \times 10^{-23})} \approx 1.2 \times 10^7 \text{ K}$$

The Standard Solar Model gives $T_c \approx 1.57 \times 10^7$ K, in reasonable agreement with our estimate.

---

## 3. Energy Transport: Radiative Zone

### The Radiative Diffusion Equation

In the radiative zone (from about $0.25 R_\odot$ to $0.71 R_\odot$), energy is transported outward by photons. However, the plasma is so opaque that photons travel only a very short distance before being absorbed and re-emitted in a random direction. This process is described by the **radiative diffusion equation**:

$$\boxed{\frac{dT}{dr} = -\frac{3 \kappa \rho}{16 \sigma T^3} \cdot \frac{L(r)}{4\pi r^2}}$$

where:
- $\kappa$ is the **opacity** (m$^2$/kg) — the cross-section per unit mass for photon absorption
- $\sigma$ is the Stefan-Boltzmann constant
- $L(r)$ is the luminosity (power) flowing through radius $r$
- The negative sign indicates temperature decreases outward

This is the third equation of stellar structure (after hydrostatic equilibrium and mass continuity). It tells us: the steeper the temperature gradient, the more energy flows outward; but high opacity ($\kappa$) requires a steeper gradient to transport the same luminosity.

> **Physical intuition**: Imagine trying to pass a message through a crowded room where each person can only whisper to their immediate neighbor. High opacity is like a very crowded room — the message (energy) moves slowly, requiring a large "incentive" (temperature gradient) to keep it flowing.

### Opacity Sources

Opacity determines how readily the plasma absorbs photons. The main sources in the solar interior are:

**Kramers' opacity** (approximate power law):

$$\kappa_\text{Kramers} \propto \rho T^{-3.5}$$

This is dominated by two processes:
- **Bound-free absorption** (photoionization): a photon ionizes an atom
- **Free-free absorption** (inverse bremsstrahlung): a photon is absorbed by an electron in the field of an ion

At very high temperatures ($T > 10^7$ K), **electron scattering** (Thomson scattering) becomes important:

$$\kappa_\text{es} = \frac{\sigma_T}{m_H} \frac{1+X}{2} \approx 0.02(1+X) \text{ m}^2/\text{kg} \approx 0.034 \text{ m}^2/\text{kg}$$

where $\sigma_T$ is the Thomson cross-section and $X$ is the hydrogen mass fraction. This is independent of temperature and density — a floor for the opacity.

Modern solar models use detailed opacity tables computed by the OPAL group (Iglesias & Rogers) or the Opacity Project, which account for millions of spectral lines from all elements.

### Photon Random Walk

A photon emitted in the core does not travel in a straight line to the surface. Instead, it undergoes a **random walk**, being absorbed and re-emitted roughly $N$ times, each time traveling a mean free path:

$$\ell_\text{mfp} = \frac{1}{\kappa \rho}$$

In the solar interior, $\ell_\text{mfp} \sim 1$ cm (incredibly short compared to $R_\odot \sim 7 \times 10^{10}$ cm).

After $N$ steps of length $\ell$, a random walk covers a net distance $d \sim \ell \sqrt{N}$. To traverse a distance $R_\odot$:

$$N \sim \left(\frac{R_\odot}{\ell_\text{mfp}}\right)^2 \sim (7 \times 10^{10})^2 \sim 5 \times 10^{21}$$

The total path length is $N \ell_\text{mfp} \sim 5 \times 10^{21}$ cm, and traveling at the speed of light, the escape time is:

$$t_\text{escape} \sim \frac{N \ell_\text{mfp}}{c} \sim \frac{5 \times 10^{21}}{3 \times 10^{10}} \sim 10^{11} \text{ s} \approx 170{,}000 \text{ years}$$

A photon born in the core takes roughly $10^5$ years to reach the surface — a staggering contrast with the 8.3 minutes it takes sunlight to reach Earth once it leaves the photosphere. The photon is "trapped" inside the Sun for most of this time, slowly diffusing outward.

---

## 4. Energy Transport: Convective Zone

### The Schwarzschild Criterion

At some radius, the temperature gradient required for radiative transport becomes too steep — the plasma becomes **convectively unstable**. The criterion for convective instability is the **Schwarzschild criterion**:

$$\boxed{\left|\frac{dT}{dr}\right|_\text{rad} > \left|\frac{dT}{dr}\right|_\text{ad}}$$

The actual (radiative) gradient exceeds the **adiabatic gradient** — the rate at which a rising blob of gas cools as it expands. When this happens, a displaced fluid element is buoyantly unstable: a rising blob is hotter (and less dense) than its surroundings, so it continues to rise. Convection sets in.

The adiabatic gradient for an ideal gas is:

$$\left|\frac{dT}{dr}\right|_\text{ad} = \frac{(\gamma - 1)}{\gamma} \frac{T}{P} \left|\frac{dP}{dr}\right| = \frac{g}{c_p}$$

where $\gamma$ is the adiabatic index (5/3 for a monatomic ideal gas), $g$ is the local gravitational acceleration, and $c_p$ is the specific heat at constant pressure.

> **Physical intuition**: Think of a pot of water on a stove. When the temperature difference between the bottom and top is small, heat flows by conduction (analogous to radiation). Turn up the heat, and at some point, the water starts to churn — convection takes over because it is a more efficient transport mechanism. The Schwarzschild criterion is the stellar physics version of "the stove is too hot for conduction alone."

### Why the Outer Sun Convects

In the Sun, convection begins at about $0.71 R_\odot$ (the base of the convection zone). Two factors trigger convective instability here:

1. **Increasing opacity**: As temperature drops below $\sim 2 \times 10^6$ K, partially ionized metals (especially Fe) dramatically increase the opacity. Higher $\kappa$ steepens the radiative gradient.
2. **Ionization zones**: Partial ionization of H and He lowers $\gamma$ below 5/3 (energy goes into ionization rather than heating), which reduces the adiabatic gradient.

Both effects conspire to violate the Schwarzschild criterion, triggering vigorous convection.

### Mixing Length Theory (MLT)

Convection in stellar interiors is turbulent and three-dimensional — a full simulation is computationally expensive (and only recently possible for portions of the Sun). The standard analytical approximation is **Mixing Length Theory** (Bohm-Vitense, 1958).

MLT assumes that convective elements (blobs) travel a characteristic distance $\ell = \alpha_\text{MLT} H_P$ before dissolving, where:
- $H_P = -P/(dP/dr) = P/(g\rho)$ is the **pressure scale height**
- $\alpha_\text{MLT}$ is a dimensionless parameter, typically $\alpha_\text{MLT} \approx 1.5$–$2.0$ (calibrated by matching the solar radius and luminosity)

The convective velocity is:

$$v_\text{conv} \sim \left(\frac{g \, \delta T \, \ell}{T}\right)^{1/2}$$

where $\delta T$ is the temperature excess of the convective element over its surroundings (the **superadiabatic excess**). The convective energy flux is:

$$F_\text{conv} = \rho c_p v_\text{conv} \, \delta T$$

In the deep convection zone, the superadiabatic excess is tiny ($\delta T / T \sim 10^{-6}$) and nearly all the luminosity is carried by convection. The temperature gradient is almost exactly adiabatic. Near the surface, however, the superadiabatic excess becomes large and the transition from convective to radiative transport occurs over a few hundred kilometers.

### Granulation: The Surface Signature

Convection is directly visible at the solar surface as **granulation**:

| Property | Value |
|----------|-------|
| Granule size | ~1000 km (1.4 arcsec) |
| Lifetime | 8–10 minutes |
| Bright center | Hot upflow, $v \approx 1$–$2$ km/s |
| Dark lanes | Cool downflow, $v \approx 2$–$3$ km/s |
| Temperature contrast | ~200 K between bright and dark |
| Number on disk | ~4 million at any time |

The asymmetry between upflows and downflows (downflows are faster and narrower) is a consequence of mass conservation and the stratified atmosphere: the density drops rapidly, so rising gas expands and decelerates, while sinking gas compresses and accelerates.

---

## 5. Tachocline

### Discovery and Location

The **tachocline** is a thin layer of strong rotational shear at the base of the convection zone, at approximately $0.71 R_\odot$. It was discovered through helioseismology (Lesson 03) and named by Spiegel and Zahn (1992).

**Key properties**:
- **Location**: centered at $r \approx 0.693 R_\odot$
- **Width**: $\Delta r \approx 0.04 R_\odot$ ($\sim 28{,}000$ km) — remarkably thin
- **Function**: transition from rigid rotation (radiative zone) to differential rotation (convective zone)

### Rotational Profile

Helioseismology reveals the Sun's internal rotation $\Omega(r, \theta)$, where $\theta$ is the colatitude:

- **Radiative zone** ($r < 0.71 R_\odot$): rotates nearly rigidly at $\Omega/(2\pi) \approx 430$ nHz ($\sim 27$-day period at all latitudes)
- **Convective zone** ($r > 0.71 R_\odot$): rotates **differentially** — the equator rotates faster ($\sim 460$ nHz, $\sim 25$-day period) than the poles ($\sim 340$ nHz, $\sim 34$-day period)

The transition occurs in the tachocline. The surface differential rotation is approximately:

$$\Omega(\theta) \approx \Omega_\text{eq} - \Delta\Omega \sin^2\theta$$

where $\Omega_\text{eq}/(2\pi) \approx 460$ nHz and $\Delta\Omega/(2\pi) \approx 120$ nHz.

### Significance for the Solar Dynamo

The tachocline is believed to play a crucial role in the **solar dynamo** (Lesson 09):

1. **Strong radial shear** ($\partial\Omega/\partial r$) stretches poloidal magnetic field lines into toroidal field — the **$\Omega$-effect**. This is the engine that generates the strong ($\sim 10^5$ G) toroidal fields that eventually emerge as sunspots.
2. **Stable stratification** below: the radiative zone is convectively stable, allowing magnetic fields to be stored and amplified without being immediately disrupted by turbulent convection.
3. **Thin width**: the thinness of the tachocline itself is a puzzle. Without some confinement mechanism (perhaps a primordial magnetic field in the radiative interior), the tachocline should spread inward on a timescale of $\sim 10^9$ years due to thermal diffusion.

> **Physical intuition**: The tachocline is like the contact zone between two conveyor belts moving at different speeds. The mechanical stress at the interface is enormous, and it is precisely this stress that winds up and amplifies magnetic fields — like twisting a rubber band by rotating one end while holding the other fixed.

---

## 6. Standard Solar Model

### Definition

The **Standard Solar Model (SSM)** is a numerical model of the Sun's interior that solves the four equations of stellar structure:

1. **Hydrostatic equilibrium**: $dP/dr = -G m \rho / r^2$
2. **Mass continuity**: $dm/dr = 4\pi r^2 \rho$
3. **Energy transport**: $dT/dr = -3\kappa\rho L/(16\sigma T^3 \cdot 4\pi r^2)$ (radiative) or $dT/dr \approx (dT/dr)_\text{ad}$ (convective)
4. **Luminosity gradient**: $dL/dr = 4\pi r^2 \rho \epsilon$ (where $\epsilon$ is the nuclear energy generation rate)

These are supplemented by:
- **Equation of state**: $P(\rho, T, X_i)$ — relates pressure to density, temperature, and composition
- **Opacity**: $\kappa(\rho, T, X_i)$ — from OPAL tables or similar
- **Nuclear reaction rates**: $\epsilon(\rho, T, X_i)$ — from laboratory measurements and theory
- **Composition evolution**: $dX_i/dt$ — tracks how nuclear reactions change the abundances over time

### Boundary Conditions

The SSM must satisfy:
- **Center** ($r = 0$): $m(0) = 0$, $L(0) = 0$
- **Surface** ($r = R_\odot$): $L(R_\odot) = L_\odot$, $T(R_\odot) = T_\text{eff}$, $m(R_\odot) = M_\odot$
- **Age**: The model must reproduce the present-day solar properties at $t = 4.57$ Gyr

### Key Predictions

The SSM, calibrated to match $L_\odot$, $R_\odot$, and $T_\text{eff}$ at the solar age, predicts:

| Quantity | SSM Prediction |
|----------|---------------|
| Central temperature | $T_c \approx 1.57 \times 10^7$ K |
| Central density | $\rho_c \approx 1.5 \times 10^5$ kg/m$^3$ |
| Central pressure | $P_c \approx 2.5 \times 10^{16}$ Pa |
| Core hydrogen fraction | $X_c \approx 0.34$ (depleted from initial 0.71) |
| Base of convection zone | $r_\text{BCZ} \approx 0.713 R_\odot$ |
| Helium abundance (surface) | $Y_s \approx 0.245$ |
| Sound speed profile | Agrees with helioseismology to $< 0.2\%$ |
| Neutrino fluxes | Consistent with SNO, Borexino measurements |

### The Solar Abundance Problem

The SSM's success depends critically on the input physics. A major challenge emerged in 2004-2009 when improved spectroscopic analyses (Asplund, Grevesse, Sauval, and collaborators) lowered the photospheric abundances of C, N, O, and Ne by 30-40%. The resulting "low-Z" SSM:

- Predicts a shallower convection zone ($r_\text{BCZ} = 0.725 R_\odot$ vs observed $0.713 R_\odot$)
- Gives a sound speed profile that disagrees with helioseismology at the $\sim 1\%$ level
- Predicts a surface helium abundance lower than observed

This **solar abundance problem** remains an active area of research. Possible resolutions include:
- Missing opacity sources (enhanced opacity at the base of the convection zone)
- Revised neon abundance (Ne is not directly measurable in the photosphere)
- Accretion of metal-poor material early in solar history
- Non-standard physics (e.g., enhanced diffusion, dark matter energy transport)

---

## 7. Equations of Stellar Structure: Summary

The four equations, plus supplementary relations, form a complete system:

$$\frac{dP}{dr} = -\frac{Gm\rho}{r^2} \quad \text{(hydrostatic equilibrium)}$$

$$\frac{dm}{dr} = 4\pi r^2 \rho \quad \text{(mass continuity)}$$

$$\frac{dT}{dr} = -\frac{3\kappa\rho}{16\sigma T^3} \frac{L}{4\pi r^2} \quad \text{(radiative)} \quad \text{or} \quad \frac{dT}{dr} = \left(1 - \frac{1}{\gamma}\right)\frac{T}{P}\frac{dP}{dr} \quad \text{(convective)}$$

$$\frac{dL}{dr} = 4\pi r^2 \rho \epsilon \quad \text{(energy generation)}$$

These four ODEs in four unknowns ($P$, $m$, $T$, $L$) as functions of $r$, supplemented by $P(\rho, T)$, $\kappa(\rho, T)$, $\epsilon(\rho, T)$, form a boundary value problem that is solved numerically to produce the SSM.

The elegance of stellar structure theory is that these four equations — force balance, mass bookkeeping, energy flow, and energy production — are sufficient to describe the interior of any star in hydrostatic and thermal equilibrium.

---

## Summary

- The Sun's interior extends from the core ($T_c \approx 1.57 \times 10^7$ K) through the radiative zone to the convection zone, bounded by the photosphere at $T \approx 5778$ K
- **Hydrostatic equilibrium** ($dP/dr = -Gm\rho/r^2$) balances gravity against the pressure gradient throughout the interior
- **Radiative transport** dominates from $0.25$ to $0.71 R_\odot$; photons random-walk outward with an escape time of $\sim 10^5$ years
- **Convection** takes over in the outer $0.29 R_\odot$ when the Schwarzschild criterion is violated; it is visible as granulation at the surface
- The **tachocline** at $0.71 R_\odot$ is a thin shear layer where rigid rotation transitions to differential rotation — critical for the solar dynamo
- The **Standard Solar Model** solves the coupled equations of stellar structure and has been validated by helioseismology and neutrino observations to remarkable precision

---

## Practice Problems

### Problem 1: Hydrostatic Equilibrium Estimate

Using the equation of hydrostatic equilibrium and assuming a uniform-density Sun ($\rho = \bar{\rho} = 3M_\odot/(4\pi R_\odot^3)$), integrate $dP/dr = -G m(r) \rho / r^2$ from the surface ($P = 0$) to the center to obtain the central pressure. Compare your answer with the SSM value of $2.5 \times 10^{16}$ Pa. Why is your estimate too low?

### Problem 2: Photon Random Walk

(a) Calculate the mean free path $\ell_\text{mfp} = 1/(\kappa \rho)$ at the center of the Sun, using $\kappa \approx 1$ m$^2$/kg and $\rho_c \approx 1.5 \times 10^5$ kg/m$^3$. (b) Estimate the number of scatterings $N$ needed to traverse $R_\odot$ by random walk. (c) Calculate the photon escape time. (d) If the nuclear reactions in the core were to suddenly stop, how long would it take for the luminosity to change noticeably at the surface?

### Problem 3: Schwarzschild Criterion

At the base of the convection zone ($r \approx 0.71 R_\odot$), the temperature is $T \approx 2.3 \times 10^6$ K and the pressure scale height is $H_P \approx 6 \times 10^7$ m. (a) Calculate the adiabatic temperature gradient $|dT/dr|_\text{ad} = g/c_p$ using $g \approx 500$ m/s$^2$ and $c_p \approx 3.5 k_B / (\mu m_H)$ with $\mu = 0.6$. (b) If the actual (radiative) gradient at this point is $7 \times 10^{-2}$ K/m, is this region convectively stable or unstable?

### Problem 4: Convective Velocity

Using mixing length theory, estimate the convective velocity near the base of the convection zone. Assume: superadiabatic excess $\delta T / T \sim 10^{-6}$, mixing length $\ell = 1.5 H_P$ with $H_P = 6 \times 10^7$ m, $g = 500$ m/s$^2$. Use $v \sim (g \, \delta T \, \ell / T)^{1/2}$. Express your answer in m/s and compare with the local sound speed $c_s \sim 2 \times 10^5$ m/s. Why is the convective velocity so much smaller than the sound speed?

### Problem 5: Tachocline Shear

The equatorial rotation rate at $r = 0.70 R_\odot$ is approximately 430 nHz, and at $r = 0.72 R_\odot$ it is approximately 460 nHz. (a) Calculate the radial shear $d\Omega/dr$ in the tachocline. (b) If a radial magnetic field line of length $L = R_\odot$ threads this shear region, estimate the toroidal field generated after one rotation period ($P \approx 27$ days) by the $\Omega$-effect: $B_\phi \approx B_r \cdot r \, (d\Omega/dr) \cdot P$. Assume $B_r \approx 1$ G. (c) How many rotation periods are needed to amplify the field to $10^5$ G?

---

[← Previous: Overview](00_Overview.md) | [Next: Nuclear Energy Generation →](02_Nuclear_Energy_Generation.md)

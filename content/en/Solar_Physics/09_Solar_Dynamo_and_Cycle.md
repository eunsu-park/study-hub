# Solar Dynamo and Cycle

## Learning Objectives

- Describe the phenomenology of the ~11-year sunspot cycle including the butterfly diagram and Waldmeier effect
- Derive the mean-field dynamo equations from the decomposition of the induction equation
- Explain the Babcock-Leighton mechanism for poloidal field regeneration
- Understand flux transport dynamo models and the role of meridional circulation
- Discuss solar cycle prediction methods and their physical basis
- Analyze the occurrence and causes of grand minima and maxima
- Connect differential rotation, turbulent convection, and magnetic field evolution into a coherent dynamo picture

---

## 1. Solar Cycle Phenomenology

The most striking regularity in solar activity is the roughly 11-year sunspot cycle, first recognized by Heinrich Schwabe in 1844 after 17 years of daily observations. This cycle governs nearly every aspect of solar variability, from the number and distribution of sunspots to the frequency of flares and coronal mass ejections.

### 1.1 The Sunspot Number Record

The international sunspot number $R$ is defined as:

$$R = k(10g + s)$$

where $g$ is the number of sunspot groups, $s$ is the total number of individual spots, and $k$ is an observer-dependent correction factor. The record extends back to ~1700 (with gaps), revealing a remarkably persistent oscillation with period $T \approx 11$ years, though individual cycles range from about 9 to 14 years.

### 1.2 The Butterfly Diagram

When sunspot latitudes are plotted against time, a striking pattern emerges — the Maunder butterfly diagram. At the beginning of each cycle, sunspots appear at latitudes around $\pm 30°$. As the cycle progresses, the active latitude band migrates toward the equator, reaching $\pm 5°$–$10°$ near cycle minimum. This is Spoerer's law: the zone of sunspot formation drifts equatorward over the course of a cycle.

The physical interpretation is deeply connected to the dynamo: the toroidal magnetic flux that produces sunspots is generated first at higher latitudes and progressively at lower latitudes as the cycle advances, reflecting the equatorward propagation of a dynamo wave.

### 1.3 Magnetic Polarity and the 22-Year Cycle

Hale's polarity law states that the leading and following polarities of bipolar active regions are opposite in the two hemispheres and reverse from one 11-year cycle to the next. The full magnetic cycle is therefore ~22 years (the Hale cycle). The Sun's polar magnetic field reverses at or near solar maximum — when the polar field is zero, the cycle is at its peak.

### 1.4 Empirical Regularities

Several empirical rules constrain dynamo models:

- **Waldmeier effect**: Cycles with shorter rise times tend to have higher maximum amplitudes. This anticorrelation between rise time and amplitude is one of the strongest constraints on nonlinear dynamo models.
- **Gnevyshev-Ohl rule**: Odd-numbered cycles tend to be stronger than the preceding even-numbered cycles, suggesting a ~22-year modulation.
- **Active longitudes**: Sunspot activity is not uniformly distributed in longitude; preferred longitudes persist for multiple rotations, indicating large-scale non-axisymmetric magnetic structure.

### 1.5 Beyond Sunspots

The solar cycle manifests in virtually every observable: the 10.7 cm radio flux ($F_{10.7}$), the total solar irradiance (TSI, varying by ~0.1% over the cycle), the EUV flux (varying by factors of 2–10), the coronal shape (round at maximum, elongated at minimum), and geomagnetic activity indices.

---

## 2. Differential Rotation

Differential rotation is the engine that converts poloidal magnetic field into toroidal field — the $\Omega$-effect. Understanding its profile is essential to understanding the solar dynamo.

### 2.1 Surface Differential Rotation

The solar surface does not rotate as a rigid body. The angular velocity depends on latitude $\theta$ (measured from the equator):

$$\Omega(\theta) = \Omega_{\text{eq}} - \Delta\Omega \sin^2\theta$$

where $\Omega_{\text{eq}} \approx 14.7°/\text{day}$ (sidereal) is the equatorial rotation rate and $\Delta\Omega \approx 2.9°/\text{day}$. The poles rotate at only about $10.0°/\text{day}$, roughly 30% slower than the equator.

A more precise fit uses additional terms:

$$\Omega(\theta) = A + B\sin^2\theta + C\sin^4\theta$$

with $A \approx 14.71°/\text{day}$, $B \approx -2.39°/\text{day}$, $C \approx -1.78°/\text{day}$.

### 2.2 Internal Rotation from Helioseismology

The revolution in our understanding came from helioseismology — the study of solar oscillation modes. By measuring how p-mode frequencies split due to rotation, the internal rotation profile $\Omega(r, \theta)$ has been mapped throughout the solar interior:

- **Convection zone** ($0.71 R_\odot < r < R_\odot$): Differential rotation similar to the surface, with roughly constant $\Omega$ on radial lines (not on cylinders as Taylor-Proudman theorem would predict — a major puzzle).
- **Radiative interior** ($r < 0.71 R_\odot$): Nearly rigid-body rotation at a rate intermediate between equatorial and polar surface rates.
- **Tachocline**: A thin shear layer at $r \approx 0.71 R_\odot$, only ~0.04 $R_\odot$ thick, where the transition from differential to rigid rotation occurs. This region of intense radial shear is believed to be the primary site of the $\Omega$-effect.
- **Near-Surface Shear Layer (NSSL)**: A region in the outer ~5% of the solar radius where $\Omega$ increases inward. This layer may also contribute to the dynamo.

### 2.3 The Omega-Effect

In the tachocline, the strong radial gradient $\partial\Omega/\partial r$ shears any pre-existing poloidal field into toroidal field. The rate of toroidal field generation is:

$$\frac{\partial B_\phi}{\partial t} \sim r\sin\theta \, B_r \frac{\partial\Omega}{\partial r}$$

Given $\Delta\Omega/\Delta r \sim 10^{-6}$ rad/s over $\Delta r \sim 0.04 R_\odot$, a modest poloidal field of ~1 G can be wound up to ~10⁴ G over a few years. The tachocline is ideally situated for this: below the turbulent convection zone, so the field can be stored and amplified without being quickly disrupted by convective motions.

---

## 3. Mean-Field Electrodynamics

Direct numerical simulation of the full solar dynamo problem remains extremely challenging due to the vast range of scales involved (from ~$10^{11}$ cm for the Sun's radius down to ~$10^4$ cm for the smallest turbulent eddies). Mean-field electrodynamics provides a tractable framework by separating the large-scale (mean) field from small-scale turbulent fluctuations.

### 3.1 Reynolds Decomposition

We decompose both velocity and magnetic field into mean and fluctuating parts:

$$\mathbf{B} = \langle\mathbf{B}\rangle + \mathbf{b}, \qquad \mathbf{v} = \langle\mathbf{v}\rangle + \mathbf{v}'$$

where $\langle \cdot \rangle$ denotes an appropriate averaging (e.g., azimuthal average in the solar case), and by definition $\langle\mathbf{b}\rangle = 0$, $\langle\mathbf{v}'\rangle = 0$.

### 3.2 The Mean-Field Induction Equation

Substituting the decomposition into the induction equation and averaging:

$$\frac{\partial\langle\mathbf{B}\rangle}{\partial t} = \nabla \times \left(\langle\mathbf{v}\rangle \times \langle\mathbf{B}\rangle + \boldsymbol{\mathcal{E}} - \eta\nabla\times\langle\mathbf{B}\rangle\right)$$

where the crucial new term is the mean electromotive force (EMF):

$$\boldsymbol{\mathcal{E}} = \langle\mathbf{v}' \times \mathbf{b}\rangle$$

This term encapsulates all the effects of turbulence on the mean field. The challenge is to express $\boldsymbol{\mathcal{E}}$ in terms of $\langle\mathbf{B}\rangle$ and its derivatives — the closure problem.

### 3.3 First-Order Smoothing Approximation

Under the assumption that the correlation time of turbulence is short (first-order smoothing or second-order correlation approximation), the EMF can be expanded:

$$\mathcal{E}_i = \alpha_{ij}\langle B_j\rangle + \beta_{ijk}\frac{\partial\langle B_j\rangle}{\partial x_k} + \cdots$$

For isotropic turbulence, this simplifies dramatically:

$$\boldsymbol{\mathcal{E}} = \alpha\langle\mathbf{B}\rangle - \beta\nabla\times\langle\mathbf{B}\rangle$$

### 3.4 The Alpha-Effect

The $\alpha$-coefficient is related to the kinetic helicity of the turbulence:

$$\alpha \approx -\frac{\tau_c}{3}\langle\mathbf{v}' \cdot (\nabla\times\mathbf{v}')\rangle$$

where $\tau_c$ is the correlation time of the turbulent eddies. The physical picture is illuminating: in the northern hemisphere, the Coriolis force imparts a systematic left-handed twist to rising convective plumes (and right-handed to sinking ones). This helical turbulence can bend and twist a toroidal field line into a loop that, when averaged, produces a component along the original poloidal direction.

The sign of $\alpha$ is typically positive in the northern hemisphere and negative in the southern hemisphere (though this is debated and may be more complex in reality).

### 3.5 The Beta-Effect: Turbulent Diffusivity

The $\beta$ term acts as a turbulent magnetic diffusivity:

$$\eta_t = \beta \approx \frac{\tau_c}{3}\langle v'^2\rangle$$

For the solar convection zone, $\eta_t \sim 10^{12}$–$10^{13}$ cm$^2$/s, which is many orders of magnitude larger than the molecular diffusivity $\eta \sim 10^4$ cm$^2$/s. This turbulent diffusivity sets the effective decay time of large-scale fields.

---

## 4. The Alpha-Omega Dynamo

The $\alpha\Omega$ dynamo is the classical framework for understanding the solar cycle. It operates through a two-step feedback loop that converts energy from differential rotation and helical turbulence into a self-sustaining oscillating magnetic field.

### 4.1 The Two-Step Process

1. **$\Omega$-effect**: Differential rotation (primarily in the tachocline) winds poloidal field into toroidal field. This step is efficient because the shear $\Delta\Omega$ is large and well-measured.

2. **$\alpha$-effect**: Helical turbulent convection twists toroidal field to regenerate poloidal field. This step is weaker but absolutely essential — without it, the dynamo would simply wind up the field and dissipate.

The mean-field induction equation for an axisymmetric $\alpha\Omega$ dynamo in spherical coordinates separates into equations for the toroidal field $B$ (represented by $B_\phi$) and the poloidal field (represented by the vector potential $A_\phi$):

$$\frac{\partial A}{\partial t} = \alpha B + \eta_t\left(\nabla^2 - \frac{1}{s^2}\right)A$$

$$\frac{\partial B}{\partial t} = s(\nabla\times(A\hat{\phi})) \cdot \nabla\Omega + \eta_t\left(\nabla^2 - \frac{1}{s^2}\right)B$$

where $s = r\sin\theta$.

### 4.2 The Dynamo Number

The behavior of the system depends on a single dimensionless parameter, the dynamo number:

$$N_D = \frac{\alpha_0 \Delta\Omega R^3}{\eta_t^2}$$

where $\alpha_0$ is the characteristic amplitude of the $\alpha$-effect, $\Delta\Omega$ is the differential rotation contrast, $R$ is the characteristic length scale, and $\eta_t$ is the turbulent diffusivity.

When $|N_D|$ exceeds a critical value $N_D^{\text{crit}}$ (typically $\sim 10$–$100$ depending on geometry), the dynamo produces oscillatory solutions — a propagating magnetic wave that reverses polarity every half-period.

### 4.3 Parker Dynamo Wave

Parker (1955) showed that an $\alpha\Omega$ dynamo produces propagating waves of magnetic activity. The direction of propagation is given by the Parker-Yoshimura sign rule:

$$\text{Propagation direction} \propto \alpha \nabla\Omega \times \hat{\phi}$$

For equatorward migration of the activity belt (as observed), we need:

- In the northern hemisphere: $\alpha > 0$ and $\partial\Omega/\partial r < 0$ (angular velocity decreasing outward), OR $\alpha < 0$ and $\partial\Omega/\partial r > 0$.

In the tachocline, helioseismology shows $\partial\Omega/\partial r > 0$ at low latitudes (angular velocity increases outward from the radiative interior). This requires $\alpha < 0$ in the northern hemisphere at the tachocline — opposite to what simple estimates of the convective $\alpha$-effect predict. This "sign problem" was one of the motivations for exploring the Babcock-Leighton mechanism as an alternative.

### 4.4 Nonlinear Saturation

The linear $\alpha\Omega$ dynamo would grow without bound once $N_D > N_D^{\text{crit}}$. In reality, the growing magnetic field must back-react on the flow to limit its own growth. The simplest prescription is $\alpha$-quenching:

$$\alpha = \frac{\alpha_0}{1 + (B/B_{\text{eq}})^2}$$

where $B_{\text{eq}} = \sqrt{4\pi\rho v'^2}$ is the equipartition field strength. More sophisticated treatments include the dynamical quenching formalism and Lorentz force feedback on differential rotation.

---

## 5. Babcock-Leighton Mechanism

The Babcock-Leighton (BL) mechanism provides an alternative to the turbulent $\alpha$-effect for regenerating poloidal field from toroidal field. It is based on the observed properties of bipolar magnetic regions (BMRs) and has become the leading paradigm for modern solar dynamo models.

### 5.1 Joy's Law and Tilted Bipolar Regions

When a toroidal flux tube rises through the convection zone and emerges at the surface, it produces a bipolar active region. Joy's law states that the line connecting the leading and following polarities is tilted with respect to the east-west direction by an angle:

$$\gamma \approx 0.5° \times \lambda$$

where $\lambda$ is the latitude in degrees. The leading polarity (closer to the equator) is at lower latitude, and the following polarity at slightly higher latitude.

The physical origin of the tilt is the Coriolis force acting on the diverging flows within the rising flux tube. This seemingly modest tilt has profound consequences for the global magnetic field evolution.

### 5.2 Surface Flux Transport

After emergence, the magnetic flux in the BMR is acted upon by surface flows:

$$\frac{\partial B_r}{\partial t} = -\Omega(\theta)\frac{\partial B_r}{\partial\phi} + \frac{1}{R_\odot\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta \, v_m B_r\right) + \frac{\eta_h}{R_\odot^2\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial B_r}{\partial\theta}\right) + S(\theta, \phi, t)$$

The four terms represent:

1. **Differential rotation**: Shears the BMR, stretching it in the east-west direction.
2. **Meridional flow**: A poleward surface flow of $v_m \sim 10$–$15$ m/s carries flux toward the poles. This is the crucial transport mechanism.
3. **Turbulent diffusion**: Supergranular diffusion with $\eta_h \approx 250$–$600$ km$^2$/s spreads the flux.
4. **Source term**: $S(\theta, \phi, t)$ represents new BMR emergences.

### 5.3 Polar Field Reversal

Because of Joy's law tilt, the following-polarity flux is systematically at higher latitude than the leading-polarity flux. Meridional flow preferentially carries the following-polarity flux poleward, while cross-equatorial diffusion allows leading-polarity flux from both hemispheres to cancel. Over the course of a cycle:

1. Following-polarity flux accumulates at the poles.
2. This cancels and eventually reverses the existing polar field.
3. The polar field reversal occurs near solar maximum.
4. The new polar field builds up through the declining phase, reaching maximum strength at solar minimum.

This reversed polar field is the seed for the next cycle's toroidal field — closing the dynamo loop. The beauty of the BL mechanism is that it is directly observable: we can watch the surface flux transport process happen in magnetogram movies.

### 5.4 Strengths and Challenges

The BL mechanism naturally explains:
- The correlation between polar field at minimum and next cycle amplitude
- The observed surface flux evolution
- The stochastic variability of the cycle (scatter in BMR tilt angles)

Challenges include:
- How does the surface poloidal field get transported down to the tachocline to be wound up?
- The mechanism relies on the statistical properties of many BMR emergences; individual large BMRs with anomalous tilts ("rogue" active regions) can significantly perturb the polar field.

---

## 6. Flux Transport Dynamo Models

Flux transport dynamos (FTDs) combine the $\Omega$-effect, the Babcock-Leighton source, and meridional circulation into a comprehensive model that reproduces many features of the solar cycle.

### 6.1 The Conveyor Belt Concept

The key insight is that meridional circulation acts as a conveyor belt connecting the surface (where poloidal field is generated by the BL mechanism) to the tachocline (where toroidal field is generated by the $\Omega$-effect):

1. At the surface, poleward meridional flow ($v_m \sim 10$–$15$ m/s) transports following-polarity flux to the poles.
2. At the poles, the flux submerges and is carried equatorward by a deep return flow.
3. At the tachocline, the poloidal field is wound up by differential rotation into toroidal field.
4. When the toroidal field is strong enough, flux tubes rise through the convection zone, emerge as new tilted BMRs, and the cycle repeats.

### 6.2 Cycle Period

In flux transport dynamo models, the cycle period is primarily determined by the meridional flow speed:

$$T_{\text{cyc}} \propto \frac{L}{v_m}$$

where $L$ is the path length of the meridional circulation. For a path from mid-latitudes to the pole, down to the base, and back to mid-latitudes, $L \sim 4 \times 10^{10}$ cm. With $v_m \sim 10$ m/s at the surface (and correspondingly slower at depth due to mass conservation):

$$T_{\text{cyc}} \sim \frac{4 \times 10^{10} \text{ cm}}{10^3 \text{ cm/s}} \sim 4 \times 10^7 \text{ s} \sim 1.3 \text{ yr}$$

This is too short — the discrepancy arises because the deep return flow is much slower than the surface flow (perhaps ~1–3 m/s), extending the effective transport time to ~10–12 years.

### 6.3 Model Results

Successful FTD models reproduce:
- The ~11-year period (with appropriate meridional flow speed)
- Equatorward migration of the toroidal field belt (butterfly diagram)
- Polar field reversal at maximum
- Phase relationships between toroidal and poloidal fields
- Asymmetries between northern and southern hemispheres

### 6.4 Sensitivity to Meridional Flow

A crucial prediction of FTD models is that the cycle period should vary inversely with meridional flow speed. Faster flow → shorter cycles → higher amplitude (because less time for diffusive decay). Observational tests of this prediction are ongoing, with some support from correlations between flow speed variations and cycle properties.

---

## 7. Cycle Prediction

Predicting the amplitude and timing of future solar cycles is both a fundamental science question and a practical necessity for space weather forecasting.

### 7.1 Precursor Methods

The most successful prediction method is based on the polar field precursor: the strength of the polar magnetic field at solar minimum is a strong predictor of the next cycle's amplitude, with a correlation coefficient of ~0.9.

The physical basis for this is straightforward in the flux transport dynamo framework: the polar field at minimum represents the poloidal field that will be wound into toroidal field by differential rotation, producing the sunspots of the next cycle.

For Cycle 25, the relatively weak polar field at the Cycle 24/25 minimum led to predictions of a moderate cycle, with $R_{\max} \approx 110$–$130$. These predictions have been broadly confirmed.

### 7.2 Dynamo-Based Prediction

Forward integration of flux transport dynamo models, initialized with observed surface magnetic field data, can make physics-based predictions. The key steps are:

1. Assimilate observed magnetogram data into the surface flux transport model.
2. Use the resulting poloidal field as an initial condition for the full dynamo model.
3. Integrate forward in time to predict toroidal field evolution.

This approach has shown skill comparable to or better than empirical precursor methods.

### 7.3 Machine Learning Approaches

Recent work has explored ML-based prediction using features such as:
- Sunspot number time series
- Polar faculae counts (proxy for polar field)
- Geomagnetic activity indices (aa, Ap)
- 10.7 cm radio flux

Neural networks and other ML methods can capture nonlinear relationships, but they are limited by the short training set (~24 well-observed cycles) and the risk of overfitting.

### 7.4 Fundamental Limits

The solar dynamo exhibits some degree of chaotic behavior, placing a fundamental limit on predictability. Current evidence suggests that:
- One cycle ahead: predictions are feasible with reasonable accuracy
- Two or more cycles ahead: predictive skill drops dramatically
- Long-term statistical properties (e.g., probability of a grand minimum in the next 50 years) may be estimable, but specific cycle-by-cycle predictions are not

---

## 8. Grand Minima and Maxima

The solar cycle is not perfectly regular. Extended periods of unusually low (grand minima) or high (grand maxima) activity punctuate the historical record.

### 8.1 The Maunder Minimum

The most famous grand minimum is the Maunder Minimum (1645–1715), during which very few sunspots were observed despite active telescopic observation. Key features:
- Estimated sunspot number $R < 5$ for most of the 70-year period
- The few sunspots that did appear were preferentially in the southern hemisphere
- Auroral activity was greatly reduced but not zero
- Coincided with the coldest phase of the Little Ice Age (though causation is debated)
- The cycle did not completely stop — there is evidence of a weak ~11-year periodicity persisting

### 8.2 Other Known Grand Minima

Cosmogenic isotope records ($^{14}$C in tree rings and $^{10}$Be in ice cores) extend the activity record back ~10,000 years. These proxies reveal additional grand minima:

| Grand Minimum | Approximate Period |
|---|---|
| Oort | 1010–1050 |
| Wolf | 1280–1340 |
| Spörer | 1390–1550 |
| Maunder | 1645–1715 |
| Dalton | 1790–1830 |

Statistical analysis by Usoskin et al. (2017) indicates that grand minima occur approximately 17% of the time, suggesting they are a normal mode of solar dynamo operation rather than exceptional events.

### 8.3 Grand Maxima

At the other extreme, periods of unusually high activity also occur. The Modern Maximum, centered on Cycle 19 (peak in 1958 with $R_{\max} \approx 285$), may represent such an episode, though this is debated. The cosmogenic isotope record shows that the level of activity in the second half of the 20th century was unusually high compared to the long-term average.

### 8.4 Physical Mechanism

What causes grand minima? In the Babcock-Leighton framework, the most promising explanation involves stochastic fluctuations in the BMR tilt angle. Joy's law gives only the average tilt; individual active regions have a large scatter around this mean. If by chance several cycles in a row produce less-than-average poloidal field (due to unfavorable tilt angle statistics or "rogue" active regions), the dynamo can enter a weakened state — a grand minimum.

Key supporting evidence:
- Simple FTD models with stochastic BL source produce grand minimum episodes with realistic frequency
- Recovery from grand minima occurs spontaneously when the stochastic fluctuations swing the other way
- The persistence of a weak cycle during the Maunder Minimum is consistent with a weakened but not extinguished dynamo

---

## Practice Problems

1. **Differential Rotation Shear**: The equatorial rotation rate is $\Omega_{\text{eq}} = 14.7°/\text{day}$ and the polar rate is $\Omega_{\text{pole}} = 10.0°/\text{day}$. Calculate the time required for an initially radial field line at $45°$ latitude to be wound into one complete toroidal loop (i.e., the leading end is $360°$ ahead of the trailing end). Use the profile $\Omega(\theta) = \Omega_{\text{eq}} - \Delta\Omega\sin^2\theta$.

2. **Dynamo Number Estimate**: Estimate the dynamo number $N_D = \alpha_0 \Delta\Omega R^3/\eta_t^2$ for the solar convection zone. Use $\alpha_0 \sim 1$ m/s, $\Delta\Omega \sim 10^{-6}$ rad/s, $R \sim 5 \times 10^{10}$ cm, and $\eta_t \sim 10^{12}$ cm$^2$/s. Is this above or below typical critical dynamo numbers?

3. **Meridional Flow and Cycle Period**: A flux transport dynamo model predicts $T_{\text{cyc}} \propto 1/v_m$. If the observed 11-year cycle corresponds to $v_m = 12$ m/s, what cycle period would result from $v_m = 8$ m/s? What about $v_m = 18$ m/s? Discuss the implications for cycle amplitude.

4. **Babcock-Leighton Source**: A bipolar magnetic region emerges at latitude $\lambda = 20°$ with a tilt angle of $\gamma = 10°$ (Joy's law), a pole separation of $d = 10^{10}$ cm, and a magnetic flux of $\Phi = 10^{22}$ Mx per polarity. Estimate the net north-south dipole moment contribution $m = \Phi \cdot d \sin\gamma$ from this single region. How many such regions per cycle ($\sim 2000$ major BMRs) are needed to reverse the polar dipole?

5. **Grand Minimum Probability**: If grand minima occur ~17% of the time on average, and the current quiet period has lasted 0 years (we are not in a grand minimum), what is the probability that the Sun will enter a grand minimum within the next 50 years? Assume that the probability of entering a grand minimum in any given 11-year cycle is $p = 0.02$ (roughly consistent with 17% occurrence rate and average duration of ~70 years). Model this as a geometric distribution.

---

**Previous**: [Active Regions and Sunspots](./08_Active_Regions_and_Sunspots.md) | **Next**: [Solar Flares](./10_Solar_Flares.md)

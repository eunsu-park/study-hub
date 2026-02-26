# Forecasting Models

## Learning Objectives

- Classify space weather models into empirical, physics-based, and semi-empirical categories and articulate the trade-offs of each
- Explain the WSA-ENLIL solar wind modeling chain from photospheric magnetogram to heliospheric conditions at Earth
- Describe physics-based and empirical magnetospheric models and their roles in storm forecasting
- Understand ionospheric models (IRI, SAMI3, TIEGCM) and their inputs, outputs, and limitations
- Explain radiation belt models that solve the Fokker-Planck equation for phase space density evolution
- Apply forecast verification metrics (POD, FAR, TSS, Brier Score, ROC) to evaluate model performance
- Distinguish between watch, warning, and alert products in the operational forecasting workflow

---

## 1. Model Classification

Space weather forecasting employs models spanning a wide range of complexity. Understanding the classification helps you choose the right tool for each problem.

**Empirical/statistical models** are built entirely from historical data. They identify correlations between inputs (e.g., solar wind parameters) and outputs (e.g., Dst) without solving any physics equations. Think of them as sophisticated pattern-matching: given conditions similar to what we have seen before, what typically happens?

Strengths: fast computation, easy to implement, can capture complex nonlinear relationships the physics does not fully describe. Weaknesses: cannot extrapolate beyond training data (dangerous for extreme events), provide no physical insight, may fail when the underlying system changes.

**Physics-based (first-principles) models** solve fundamental equations — MHD, Boltzmann/Vlasov, Maxwell — numerically on a computational grid. They embody our physical understanding of the system.

Strengths: can, in principle, predict events never seen before; provide spatial and temporal detail; physically self-consistent. Weaknesses: computationally expensive (hours to days of wall-clock time), sensitive to initial and boundary conditions, may miss physics not included in the equations (e.g., kinetic effects in MHD models).

**Semi-empirical models** use a physics-based framework but replace certain components with empirical parameterizations. For example, the Tsyganenko magnetic field models have the correct mathematical form for magnetospheric current systems but use empirically fitted coefficients. This approach is a pragmatic middle ground.

**Operational vs. research models.** An operational model must:
- Run faster than real time (a 1-hour forecast must compute in minutes)
- Handle missing or degraded inputs gracefully
- Produce output in a standardized, interpretable format
- Have documented skill scores and known failure modes

Research models can take days to run and require expert tuning. The pathway from research to operations is long and demanding.

---

## 2. Solar Wind Models

### 2.1 WSA (Wang-Sheeley-Arge) Model

The WSA model predicts the background solar wind speed at Earth from observations of the Sun's photospheric magnetic field. It is the standard coronal model in operational space weather forecasting.

**Three-step procedure:**

**Step 1: Potential Field Source Surface (PFSS) extrapolation.** Starting from a synoptic magnetogram (a full-surface map of the photospheric radial magnetic field $B_r$), solve for the potential (current-free) magnetic field between the photosphere ($r = R_\odot$) and the source surface ($r = R_{SS} \approx 2.5 R_\odot$):

$$\nabla^2 \Phi = 0 \quad \text{with} \quad B_r|_{R_\odot} = \text{observed}, \quad B_r|_{R_{SS}} = \text{radial}$$

The source surface condition forces field lines to become radial at $R_{SS}$, mimicking the effect of the solar wind flow. The solution gives the coronal magnetic field topology: open field regions (coronal holes) vs. closed loops.

**Step 2: Expansion factor.** For each open field line, compute the expansion factor:

$$f_s = \frac{(R_\odot / R_{SS})^2}{|B_r(R_\odot)| / |B_r(R_{SS})|}$$

A small $f_s$ means the flux tube expands slowly — it threads through a large coronal hole and carries fast wind. A large $f_s$ means rapid expansion near the boundary of a small coronal hole — associated with slow wind.

Additionally, the angular distance $\theta_b$ from the flux tube footpoint to the nearest coronal hole boundary is computed.

**Step 3: Empirical velocity relation.** The solar wind speed at $R_{SS}$ is:

$$v_{sw} = v_{\text{fast}} - v_{\text{range}} \cdot \frac{f_s^{1/\alpha}}{1 + f_s^{1/\alpha}} \cdot \left(1 - 0.8 \exp\left[-(\theta_b/\theta_0)^\beta\right]\right)^3$$

where $v_{\text{fast}} \approx 625$ km/s, $v_{\text{range}} \approx 240$ km/s, and $\alpha, \beta, \theta_0$ are empirically tuned parameters. The first factor captures the inverse relationship between expansion and speed. The second factor adjusts for proximity to coronal hole boundaries.

**Limitations of WSA:** It uses a potential field (no currents), so it cannot model the streamer belt or CME-related fields accurately. The synoptic magnetogram requires a full solar rotation (~27 days) to build, so far-side information is always ~2 weeks old.

### 2.2 ENLIL Model

ENLIL (named after the Sumerian god of wind) is a 3D time-dependent MHD model of the heliosphere developed by Dusan Odstrcil.

**Domain:** From the inner boundary at $21.5 R_\odot$ (where WSA provides input) to 1-10 AU.

**Equations:** ENLIL solves the ideal MHD equations in a rotating heliographic coordinate system:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

$$\frac{\partial (\rho \mathbf{v})}{\partial t} + \nabla \cdot \left[\rho \mathbf{v}\mathbf{v} + \left(p + \frac{B^2}{2\mu_0}\right)\mathbf{I} - \frac{\mathbf{B}\mathbf{B}}{\mu_0}\right] = -\rho \frac{GM_\odot}{r^2}\hat{r} + \rho \mathbf{a}_{\text{rot}}$$

$$\frac{\partial \mathbf{B}}{\partial t} - \nabla \times (\mathbf{v} \times \mathbf{B}) = 0$$

plus the energy equation with polytropic or more sophisticated closures.

**CME insertion:** CMEs are introduced as hydrodynamic pulses at the inner boundary — a "cone model" specifying the CME's direction, angular width, speed, and density enhancement. More advanced versions use the Gibson-Low flux rope model.

**Operational use:** ENLIL runs at NOAA/SWPC with WSA input, producing forecasts of solar wind conditions at Earth (and other planets) up to 4-5 days ahead. Typical run time: ~1 hour on a modern cluster.

**Performance:** For background solar wind, ENLIL captures the high-speed stream structure reasonably well. For CME arrival time, the typical error is $\pm 10-12$ hours, with occasional misses of $>24$ hours. The density and magnetic field predictions are less reliable than the speed.

### 2.3 Other Heliospheric Models

**EUHFORIA (European Heliospheric Forecasting Information Asset):** Developed at KU Leuven, EUHFORIA improves upon ENLIL by offering more realistic CME models, including linear force-free spheromak and non-linear flux rope (FRi3D) insertions. The improved CME geometry leads to better magnetic field predictions at Earth.

**SWASTi (Space Weather Adaptive SimulaTion):** Developed at the Indian Institute of Geomagnetism, SWASTi couples a coronal model with an MHD heliospheric solver and includes a module for solar energetic particle transport.

**Drag-based model (DBM):** A simple but remarkably effective model that treats CME propagation as aerodynamic drag in the ambient solar wind:

$$\frac{dv}{dt} = -\gamma (v - v_{sw})|v - v_{sw}|$$

where $\gamma$ is the drag coefficient (empirically $\sim 0.2 - 2.0 \times 10^{-7}$ km$^{-1}$) and $v_{sw}$ is the ambient wind speed. Despite its simplicity, DBM achieves CME arrival time errors comparable to full MHD models ($\pm 10-15$ hours) and runs in milliseconds.

---

## 3. Magnetospheric Models

### 3.1 Empirical Magnetic Field Models

The Tsyganenko family of models provides the magnetic field vector at any point in the magnetosphere as a function of external parameters.

**T89:** The simplest version, parameterized solely by Kp. Each current system (ring current, tail current, magnetopause currents, field-aligned currents) is represented by mathematical functions whose amplitudes scale with Kp.

**T96:** Adds explicit dependence on solar wind dynamic pressure $P_{\text{dyn}}$, IMF $B_y$ and $B_z$, and Dst. This allows the model to respond to specific solar wind conditions rather than just the aggregate Kp.

**TS04 (T04S):** Uses a "storm-time" formulation where the model parameters are driven by time histories of the solar wind input (integrated over the preceding hours), allowing it to represent the delayed response of the magnetosphere to changing conditions.

**TS07:** The most flexible version, using an expansion in basis functions fitted to a large database of satellite magnetic field measurements. No predefined current system shapes — the model learns the field structure from data.

**Applications:** Tsyganenko models are used for:
- Magnetic field line tracing (mapping ionospheric footpoints to equatorial plane)
- Computing adiabatic invariants ($\mu$, $J$, $L^*$) for radiation belt studies
- Background field for particle tracing codes
- Quick-look magnetospheric visualization

**Limitations:** These models give statistical averages for the given input parameters. They cannot capture the dynamics of individual substorms, the detailed structure of thin current sheets, or reconnection.

### 3.2 Global MHD Models

Global magnetospheric MHD models solve the ideal or resistive MHD equations in a domain extending from the dayside solar wind (~30 $R_E$ upstream) to the deep tail (~200 $R_E$), coupled to an ionospheric electrodynamics solver at the inner boundary (~2-3 $R_E$).

**Major models:**

| Model | Institution | Grid | Special Features |
|-------|------------|------|-----------------|
| SWMF/BATS-R-US | U. Michigan | Adaptive Cartesian | Multi-physics framework, operational at SWPC |
| LFM/CMIT | NCAR/Dartmouth | Distorted logically rectangular | High-resolution magnetopause, TING coupling |
| OpenGGCM | UNH | Stretched Cartesian | Open-source, Coupled Thermosphere-Ionosphere |
| GUMICS | FMI (Finland) | Adaptive Cartesian | Fully conservative scheme |

**SWMF (Space Weather Modeling Framework)** deserves special mention. It couples BATS-R-US (global MHD) with:
- RIM (Ridley Ionosphere Model) for ionospheric electrodynamics
- RCM (Rice Convection Model) for ring current physics
- CRCM or RAM-SCB for radiation belt dynamics

This coupled approach is necessary because ideal MHD cannot capture ring current energization (which involves gradient/curvature drift physics absent from MHD).

**What MHD models capture well:**
- Magnetopause location and shape
- Large-scale convection patterns
- Storm and substorm magnetic field reconfigurations
- Cross-polar cap potential

**What MHD models struggle with:**
- Magnetic reconnection (grid resolution dependent, often uses numerical resistivity)
- Radiation belt dynamics (no single-particle physics)
- Ring current (requires coupling to drift-kinetic models)
- Ionospheric outflow (often parameterized crudely)

### 3.3 Geospace Model Coupling

Modern geospace modeling is moving toward coupled frameworks that combine the strengths of different model types:

$$\text{Solar wind (ENLIL)} \rightarrow \text{Magnetosphere (MHD)} \rightarrow \text{Ionosphere (electrodynamics)} \rightarrow \text{Thermosphere (neutral dynamics)}$$

Each component passes boundary conditions to the next. The ionosphere-thermosphere coupling is bidirectional: the ionosphere provides conductance to the MHD model, while the MHD model provides electric fields and particle precipitation to the ionosphere.

The SWPC operational suite currently uses WSA-ENLIL for the solar wind and SWMF/BATS-R-US for the magnetosphere, with plans to add coupled thermosphere-ionosphere components.

---

## 4. Ionospheric Models

### 4.1 IRI (International Reference Ionosphere)

IRI is the international standard empirical model of the ionosphere, maintained by COSPAR and URSI. It provides monthly median values of ionospheric parameters as a function of location, time, and solar/geomagnetic activity.

**Inputs:** Geographic location, date, time, F10.7 (or sunspot number), and optionally Kp or ap.

**Outputs:**
- Electron density profile $n_e(h)$ from 50 to 2000 km
- Electron and ion temperatures $T_e(h)$, $T_i(h)$
- Ion composition ($\text{O}^+$, $\text{H}^+$, $\text{He}^+$, $\text{N}^+$, $\text{NO}^+$, $\text{O}_2^+$)
- Total electron content (TEC)

**Strengths:** Based on decades of ionosonde, incoherent scatter radar, and satellite data. Excellent for climatological studies and as a background model. Computationally trivial.

**Limitations:** Provides monthly medians, not instantaneous values. Storm-time performance is poor — during geomagnetic storms, the ionosphere can deviate dramatically from the median (positive and negative storm phases). IRI-2020 includes improved storm-time corrections, but capturing the full complexity of storm responses remains beyond an empirical model.

### 4.2 SAMI3 (NRL)

SAMI3 is a physics-based 3D ionosphere/plasmasphere model developed at the Naval Research Laboratory. The name stands for "Sami3 is Also a Model of the Ionosphere" (a recursive acronym).

**Physics:** SAMI3 solves the ion continuity, momentum, and energy equations for seven ion species ($\text{H}^+$, $\text{He}^+$, $\text{N}^+$, $\text{O}^+$, $\text{N}_2^+$, $\text{NO}^+$, $\text{O}_2^+$) along magnetic field lines, coupled through charge neutrality and current closure:

$$\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{v}_s) = P_s - L_s n_s$$

$$n_s m_s \frac{D\mathbf{v}_s}{Dt} = n_s q_s(\mathbf{E} + \mathbf{v}_s \times \mathbf{B}) - \nabla p_s + n_s m_s \mathbf{g} + \sum_t n_s m_s \nu_{st}(\mathbf{v}_t - \mathbf{v}_s)$$

The neutral atmosphere is specified by an empirical model (NRLMSISE-00 or TIEGCM output), and the electric field comes from either an empirical convection model or a coupled magnetospheric model.

**Applications:** Equatorial plasma bubbles, plasmaspheric drainage plumes, storm-time ionospheric redistribution, and GPS scintillation modeling.

### 4.3 TIEGCM and WACCM-X

**TIEGCM (Thermosphere-Ionosphere Electrodynamics General Circulation Model):** Developed at NCAR, TIEGCM is the community standard for upper atmosphere modeling. It covers ~97 to ~600 km altitude and solves the coupled thermosphere-ionosphere system self-consistently.

Key physics: Solar EUV heating, Joule heating from magnetospheric convection, ion-neutral coupling, neutral winds, composition changes, and dynamo electric fields.

**WACCM-X (Whole Atmosphere Community Climate Model - eXtended):** Extends the climate model WACCM upward to ~500-700 km, coupling the troposphere, stratosphere, mesosphere, and thermosphere-ionosphere in a single framework. This allows investigation of how lower atmosphere weather (e.g., sudden stratospheric warmings) affects the ionosphere — a topic of growing interest.

---

## 5. Radiation Belt Models

### 5.1 The Fokker-Planck Approach

Radiation belt electrons evolve under the influence of radial diffusion, local acceleration by wave-particle interactions, and losses to the atmosphere and magnetopause. The governing equation is the Fokker-Planck equation for the phase space density $f$ as a function of the three adiabatic invariants ($\mu$, $J$, $L^*$) and time:

$$\frac{\partial f}{\partial t} = L^{*2} \frac{\partial}{\partial L^*}\left(\frac{D_{L^*L^*}}{L^{*2}} \frac{\partial f}{\partial L^*}\right) + \left(\frac{\partial f}{\partial t}\right)_{\text{local}} - \frac{f}{\tau_{\text{loss}}} + S$$

The first term is radial diffusion (driven by ULF waves, with diffusion coefficient $D_{L^*L^*}$). The second term represents local acceleration and pitch angle scattering by chorus, hiss, and EMIC waves. The third is atmospheric loss. $S$ represents any sources.

### 5.2 Major Models

**Salammbo (ONERA, France):** One of the earliest radiation belt models solving the full 3D Fokker-Planck equation in ($\mu$, $J$, $L^*$) space. Includes radial diffusion, chorus wave acceleration, plasmaspheric hiss loss, and EMIC wave loss. Used for the European SPACECAST system.

**VERB-4D (UCLA):** Versatile Electron Radiation Belt code. Solves the Fokker-Planck equation with state-of-the-art diffusion coefficients computed from quasi-linear theory for realistic wave models. The "4D" refers to the three invariants plus time.

**Key inputs:**
- $D_{L^*L^*}(L^*, \text{Kp})$: radial diffusion coefficients, often parameterized following Brautigam & Albert (2000):

$$D_{L^*L^*} = D_0 \times 10^{(0.506 \text{Kp} - 9.325)} \times L^{*10}$$

- Wave amplitudes: chorus, hiss, and EMIC wave power spectral density, parameterized by AE or Kp
- Boundary conditions: measured flux at $L^*_{\min}$ (stable inner belt) and $L^*_{\max}$ (from GOES)

### 5.3 Operational Use

The SPACECAST system (EU) provides radiation belt nowcasts and 3-hour forecasts based on Salammbo. NASA's Community Coordinated Modeling Center (CCMC) runs several radiation belt models on demand.

**Challenges:**
- Wave models are still crude parameterizations of highly variable phenomena
- The plasmasphere boundary (plasmapause) location determines which waves dominate (hiss inside, chorus outside), introducing sensitivity to an imprecisely known quantity
- Coupling between the ring current (which modifies the magnetic field) and radiation belt dynamics is not fully captured in most models

---

## 6. Ensemble Modeling and Probabilistic Forecasts

### 6.1 Why Ensemble?

A single deterministic forecast gives one answer but no indication of confidence. In space weather, uncertainties are large:

- **Initial condition uncertainty:** CME speed measured from coronagraph images has errors of $\pm 100-200$ km/s. CME direction has errors of $\pm 10-20°$.
- **Model uncertainty:** Drag coefficient in DBM varies by an order of magnitude. MHD models use different numerical schemes and resolutions.
- **Boundary condition uncertainty:** The ambient solar wind state is imperfectly known, especially on the far side of the Sun.

### 6.2 Ensemble CME Arrival Time

The drag-based model is ideal for ensemble forecasting because it runs in milliseconds. The procedure:

1. Estimate CME initial speed $v_0$ and uncertainty $\sigma_v$ from observations.
2. Estimate drag coefficient $\gamma$ and its uncertainty $\sigma_\gamma$.
3. Draw $N$ samples (e.g., $N = 1000$) from the joint distribution of $(v_0, \gamma, v_{sw})$.
4. Run the DBM for each sample.
5. The resulting distribution of arrival times gives a probabilistic forecast:

$$P(\text{arrival between } t_1 \text{ and } t_2) = \frac{\text{number of ensemble members arriving in } [t_1, t_2]}{N}$$

This approach naturally captures the fact that faster CMEs have narrower arrival time distributions (less sensitive to drag), while slower CMEs have broader distributions.

### 6.3 Moving Toward Probabilistic Forecasting

NOAA/SWPC is transitioning from deterministic to probabilistic forecasts. Instead of "A G2 storm is expected on Tuesday," the forecast might say "There is a 60% chance of G2 or greater conditions on Tuesday."

This requires:
- Ensemble modeling infrastructure
- Calibrated probability estimates (so that events forecast at 60% actually occur about 60% of the time)
- User education (probabilistic forecasts are more informative but harder to interpret)

---

## 7. Verification Metrics

Forecast verification is essential for measuring model skill, comparing models, and communicating forecast quality to users. Different metrics are appropriate for different forecast types.

### 7.1 Continuous (Point) Forecasts

For forecasts of continuous variables like Dst or solar wind speed:

**Root Mean Square Error:**

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(F_i - O_i)^2}$$

where $F_i$ is the forecast and $O_i$ is the observation. RMSE penalizes large errors more heavily than small ones.

**Mean Absolute Error:**

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|F_i - O_i|$$

More robust to outliers than RMSE. If RMSE $\gg$ MAE, the forecast has occasional large misses.

**Correlation coefficient $r$:** Measures linear association. A forecast can have high $r$ but large bias. Always report $r$ alongside RMSE or MAE.

**Persistence baseline:** The simplest forecast is "tomorrow will be like today" ($F_{t+1} = O_t$). Any useful model must beat persistence.

### 7.2 Categorical (Yes/No) Forecasts

For binary event forecasts (e.g., "Will a G3+ storm occur in the next 24 hours?"), construct a contingency table:

|  | Event Observed | No Event Observed |
|--|---------------|-------------------|
| **Event Forecast** | Hits ($a$) | False Alarms ($b$) |
| **No Event Forecast** | Misses ($c$) | Correct Rejections ($d$) |

**Probability of Detection (Hit Rate):**

$$\text{POD} = \frac{a}{a + c}$$

What fraction of events were correctly forecast? POD = 1 is achievable by always forecasting "yes" — so POD alone is insufficient.

**False Alarm Ratio:**

$$\text{FAR} = \frac{b}{a + b}$$

What fraction of "yes" forecasts were wrong?

**True Skill Statistic (Hanssen-Kuipers / Peirce Skill Score):**

$$\text{TSS} = \frac{a}{a + c} - \frac{b}{b + d} = \text{POD} - \text{POFD}$$

where POFD is the Probability of False Detection ($b/(b+d)$). TSS ranges from $-1$ to $+1$; a value of 0 indicates no skill. TSS is equitable (not affected by the relative frequency of events vs. non-events), making it excellent for rare events.

**Heidke Skill Score:**

$$\text{HSS} = \frac{2(ad - bc)}{(a+c)(c+d) + (a+b)(b+d)}$$

Measures improvement over random forecasts. HSS = 0 for random, HSS = 1 for perfect.

### 7.3 Probabilistic Forecasts

**Brier Score:**

$$\text{BS} = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2$$

where $p_i$ is the forecast probability and $o_i \in \{0, 1\}$ is the observed outcome. BS = 0 is perfect; BS = 1 is worst. Think of it as the MSE for probabilities.

**Brier Skill Score:**

$$\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{ref}}}$$

where $\text{BS}_{\text{ref}}$ is the Brier Score of a reference forecast (typically climatology). BSS > 0 indicates skill above the reference.

**Reliability diagram:** Plot observed frequency against forecast probability in bins. A perfectly reliable forecast lies on the diagonal. Points above the diagonal indicate underforecasting (events occur more often than predicted); below indicates overforecasting.

**ROC (Receiver Operating Characteristic) curve:** Plot POD vs. POFD at varying probability thresholds. The area under the ROC curve (AUC) measures overall discrimination ability. AUC = 0.5 is no skill (random); AUC = 1.0 is perfect.

### 7.4 Choosing Metrics

No single metric tells the whole story. Best practice:
- Report multiple metrics
- Always compare to a reference (persistence, climatology)
- For rare events, prefer TSS over accuracy (which is dominated by correct rejections)
- For probabilistic forecasts, report both reliability (calibration) and resolution (sharpness)

---

## 8. Operational Forecasting Workflow

### 8.1 Watch-Warning-Alert System

Space weather forecasts follow a tiered urgency system, similar to terrestrial weather:

**Watch** (days in advance): Issued when conditions suggest elevated space weather risk. Examples:
- A complex active region (beta-gamma-delta magnetic classification) rotates onto the solar disk
- A CME is observed heading toward Earth
- A recurrent high-speed stream is expected based on 27-day recurrence

**Warning** (hours ahead): Issued when an event is in progress or imminent:
- CME has been confirmed Earth-directed; ENLIL model predicts arrival within 24-48 hours
- Solar wind enhancement is observed at L1 (DSCOVR), arrival at Earth in 15-60 minutes
- Solar proton event in progress

**Alert** (immediate): Threshold has been exceeded:
- Kp has reached 5+ (G1+ storm in progress)
- Proton flux exceeds 10 pfu (S1+ radiation storm)
- X-ray flux exceeds M1 (R1+ radio blackout)

### 8.2 The Human Forecaster

Despite advances in numerical modeling, human forecasters remain essential in space weather. They:

1. **Synthesize multiple data streams:** Satellite imagery, model output, ground observations, and real-time solar wind data.
2. **Apply expert judgment:** Models often disagree. The forecaster weighs each model's known strengths and weaknesses for the specific situation.
3. **Assess uncertainty:** "The model says G3, but the input data is uncertain, so I will forecast G2 with a note about G3 possibility."
4. **Communicate:** Translate technical output into actionable information for operators (power grid, airlines, satellite operators).

### 8.3 Forecast Verification in Practice

SWPC publishes quarterly and annual verification reports documenting their forecast skill. Key findings consistently show:

- **Flare forecasts:** Modest skill. Categorical forecasts of M-class flares achieve TSS ~0.3-0.5. The limitation is our inability to predict exactly when a complex active region will erupt.
- **CME arrival time:** $\pm 10-15$ hours with ENLIL, sometimes worse. The magnetic field inside the CME (especially $B_z$) remains largely unpredictable.
- **Geomagnetic storm intensity:** Once solar wind data is available at L1, 1-hour Dst forecasts are good (correlation > 0.9). Longer lead times degrade rapidly.
- **Radiation storms (SEPs):** Onset can often be identified within 30-60 minutes of a flare, but peak flux and duration are difficult.

---

## Practice Problems

**Problem 1.** A synoptic magnetogram shows a large coronal hole with a flux expansion factor $f_s = 3.5$ and a footpoint distance to the nearest coronal hole boundary $\theta_b = 15°$. Using the WSA empirical relation (with $v_{\text{fast}} = 625$ km/s, $v_{\text{range}} = 240$ km/s, $\alpha = 4.5$, $\theta_0 = 2.8°$, $\beta = 1.25$), estimate the solar wind speed at the source surface. Discuss whether this is fast or slow wind and what structures at Earth would be associated with it.

**Problem 2.** A CME is observed with initial speed $v_0 = 1200$ km/s. The ambient solar wind speed is $v_{sw} = 400$ km/s. Using the drag-based model with $\gamma = 0.5 \times 10^{-7}$ km$^{-1}$, estimate the CME speed at 1 AU and the transit time. Hint: the analytical solution for the DBM gives $v(t) = v_{sw} + (v_0 - v_{sw}) / [1 + \gamma |v_0 - v_{sw}| t]$ and $R(t) = v_{sw}t + \ln(1 + \gamma |v_0 - v_{sw}| t) / \gamma$.

**Problem 3.** A space weather model produces the following contingency table for G3+ storm forecasts over one year: Hits = 12, False Alarms = 8, Misses = 5, Correct Rejections = 340. Calculate POD, FAR, TSS, HSS, and the overall accuracy. Explain why accuracy is a misleading metric for this problem and why TSS is preferred.

**Problem 4.** A probabilistic CME arrival forecast gives a 70% chance of arrival within a 12-hour window. Over 50 such forecasts at the 70% level, the CME actually arrived in the specified window 42 times. Is the forecast reliable? Calculate the Brier Score for this single probability bin and plot the point on a reliability diagram.

**Problem 5.** Compare the Tsyganenko T96 model and SWMF/BATS-R-US for the following application: a satellite operator needs to know the magnetic field at geostationary orbit (6.6 $R_E$) during a moderate storm (Dst = $-80$ nT, $P_{\text{dyn}} = 5$ nPa, $B_z = -8$ nT). Discuss the expected accuracy, computation time, and appropriateness of each model. Which would you recommend and why?

---

**Previous**: [Space Weather Indices](./13_Space_Weather_Indices.md) | **Next**: [AI/ML for Space Weather](./15_AI_ML_for_Space_Weather.md)

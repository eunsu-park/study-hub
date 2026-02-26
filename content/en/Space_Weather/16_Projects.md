# Projects

## Learning Objectives

- Apply the Burton equation to predict Dst from solar wind inputs and compare with LSTM-based predictions
- Implement a 1D radial diffusion solver for radiation belt electron phase space density evolution
- Build a space weather monitoring dashboard integrating multiple data streams and models
- Integrate knowledge from multiple lessons (indices, forecasting, magnetospheric physics, radiation belts)
- Practice scientific data analysis including model validation, error analysis, and visualization

---

## 1. Project 1: Dst Prediction — Burton Model vs LSTM

### 1.1 Overview

This project implements two fundamentally different approaches to predicting the Dst geomagnetic index from upstream solar wind measurements: the physics-based Burton equation and a data-driven LSTM neural network. By building and comparing both, you will develop intuition for the strengths and weaknesses of each paradigm.

### 1.2 The Burton Equation

The Burton et al. (1975) model treats the ring current as a driven-dissipative system. The pressure-corrected Dst evolves according to:

$$\frac{d\text{Dst}^*}{dt} = Q(t) - \frac{\text{Dst}^*}{\tau}$$

where the injection function $Q$ and recovery timescale $\tau$ are:

$$Q(t) = \begin{cases} a \cdot (E_y - E_c) & \text{if } E_y > E_c \\ 0 & \text{if } E_y \leq E_c \end{cases}$$

with parameters:
- $E_y = -v_{sw} \cdot B_z$ (dawn-dusk convection electric field in mV/m, where $B_z$ is in GSM coordinates and negative means southward)
- $E_c = 0.5$ mV/m (critical electric field threshold for injection)
- $a = -4.5$ nT/(mV/m $\cdot$ hr) (injection efficiency)
- $\tau = 7.7$ hr (exponential decay time for ring current loss)

The pressure correction relates observed Dst to the ring current contribution:

$$\text{Dst}^* = \text{Dst} - b\sqrt{P_{\text{dyn}}} + c$$

where $P_{\text{dyn}} = m_p n v^2$ is the solar wind dynamic pressure, $b \approx 7.26$ nT/$\sqrt{\text{nPa}}$, and $c \approx 11$ nT.

### 1.3 Step-by-Step Implementation

**Step (a): Data Acquisition**

Obtain hourly OMNI data for 2000-2020 from NASA's OMNIWeb (omniweb.gsfc.nasa.gov). Required parameters:
- Solar wind speed $v$ (km/s)
- Proton number density $n$ (cm$^{-3}$)
- IMF magnitude $B$ (nT) and $B_z$ GSM component (nT)
- Observed Dst (nT)

Handle data gaps: OMNI uses fill values (e.g., 9999.9) for missing data. Implement linear interpolation for gaps shorter than 3 hours; exclude periods with longer gaps.

**Step (b): Compute Derived Quantities**

From the raw parameters, compute:

$$P_{\text{dyn}} = 1.6726 \times 10^{-6} \cdot n \cdot v^2 \quad [\text{nPa}]$$

$$E_y = -v \cdot B_z \times 10^{-3} \quad [\text{mV/m}]$$

$$\text{Dst}^* = \text{Dst} - 7.26\sqrt{P_{\text{dyn}}} + 11 \quad [\text{nT}]$$

Note the unit conversion for $E_y$: $v$ is in km/s and $B_z$ in nT, giving $E_y$ in $\mu$V/m; multiply by $10^{-3}$ to get mV/m. Alternatively, use consistent SI units throughout.

**Step (c): Numerical Integration of the Burton ODE**

Integrate the ODE using a simple Euler method (sufficient for hourly data) or 4th-order Runge-Kutta (more accurate):

**Euler method:** With time step $\Delta t = 1$ hour:

$$\text{Dst}^*(t + \Delta t) = \text{Dst}^*(t) + \left[Q(t) - \frac{\text{Dst}^*(t)}{\tau}\right] \Delta t$$

**Initial condition:** Set $\text{Dst}^*(t_0)$ to the observed value at the start of the integration period.

**Recover Dst:** After integration, convert back:

$$\text{Dst}(t) = \text{Dst}^*(t) + b\sqrt{P_{\text{dyn}}(t)} - c$$

**Step (d): LSTM Implementation**

Preprocess the solar wind time series:
1. Select input features: $v$, $n$, $B$, $B_z$, $P_{\text{dyn}}$, $E_y$
2. Normalize each feature to zero mean and unit variance (save the scaler for test data)
3. Create sliding windows: 24 hours of input (24 time steps $\times$ 6 features) mapped to Dst at $t + 1$ hour

Architecture:
- LSTM layer with 64 hidden units (return sequences = False)
- Dropout layer (rate = 0.2)
- Dense layer with 32 units and ReLU activation
- Output Dense layer with 1 unit (linear activation)

Training:
- Split: 2000-2015 training, 2015-2020 testing
- Loss: Mean Squared Error
- Optimizer: Adam (learning rate = $10^{-3}$)
- Batch size: 256
- Epochs: 50-100 with early stopping (patience = 10, monitor validation loss)
- Validation: 10% of training data (last portion, not random, to avoid temporal leakage)

**Step (e): Comparison**

For both models on the test set (2015-2020), compute:
- RMSE (nT)
- Pearson correlation coefficient $r$
- Scatter plot: predicted vs observed Dst with 1:1 line
- Time series overlay: observed Dst with both model predictions for selected storms

**Step (f): Storm Event Analysis**

Identify the 5-10 strongest storms in the test period (Dst < $-100$ nT). For each storm, compare:
- Does the model capture the Dst minimum? (Systematic underestimate means the model under-predicts peak storm intensity)
- How well is the recovery phase represented? The Burton model assumes exponential recovery; reality often shows faster initial recovery followed by a slower tail
- Which model handles multi-step storms (multiple dips) better?

### 1.4 Discussion Questions

Think about these as you analyze your results:

- The Burton model has 4 free parameters and encodes specific physics. The LSTM has thousands of parameters and learns from data. Under what circumstances would you prefer one over the other?
- What happens to the LSTM during a storm that is significantly stronger than anything in the training set? What happens to the Burton model?
- How would you modify the Burton equation to improve recovery phase predictions? (Hint: consider making $\tau$ activity-dependent, as in the O'Brien-McPhetridge formulation.)

---

## 2. Project 2: Radiation Belt Radial Diffusion

### 2.1 Overview

This project implements a 1D radial diffusion solver for the electron phase space density (PSD) in the radiation belt. Radial diffusion, driven by ULF wave activity, is one of the primary mechanisms for energizing and redistributing radiation belt electrons. By solving the diffusion equation numerically, you will observe how changes in geomagnetic activity (parameterized by Kp) drive enhancement and depletion of the radiation belt.

### 2.2 Governing Equation

The radial diffusion equation for electron PSD $f$ at fixed first and second adiabatic invariants ($\mu$, $J$):

$$\frac{\partial f}{\partial t} = L^2 \frac{\partial}{\partial L}\left(\frac{D_{LL}}{L^2} \frac{\partial f}{\partial L}\right) - \frac{f}{\tau_{\text{loss}}} + S$$

where:
- $f(L, t)$ is the phase space density at $L$-shell $L$ and time $t$
- $D_{LL}(L, \text{Kp})$ is the radial diffusion coefficient
- $\tau_{\text{loss}}(L, \text{Kp})$ is the loss timescale (atmospheric precipitation due to wave-particle scattering)
- $S(L, t)$ is a source term (we will set $S = 0$ for this project, considering only radial transport and loss)

### 2.3 Step-by-Step Implementation

**Step (a): Set Up the L-Grid**

Create a uniform grid in $L$-shell:
- $L_{\min} = 2.0$ (inner boundary — stable inner belt)
- $L_{\max} = 7.0$ (outer boundary — near geostationary orbit)
- $N = 100$ grid points
- $\Delta L = (L_{\max} - L_{\min}) / (N - 1) = 0.0505$

**Step (b): Radial Diffusion Coefficient**

Use the Brautigam & Albert (2000) magnetic component parameterization:

$$D_{LL}^M(L, \text{Kp}) = D_0 \times 10^{(0.506 \text{Kp} - 9.325)} \times L^{10}$$

with $D_0 = 1$ (in units of day$^{-1}$, or convert to s$^{-1}$ for SI computation).

The strong $L^{10}$ dependence means radial diffusion is overwhelmingly dominant at large $L$ and negligible in the inner belt. Physically, this reflects the fact that ULF wave amplitudes increase dramatically with distance from Earth.

For completeness, the electric component:

$$D_{LL}^E(L, \text{Kp}) = 0.26 \times 10^{(0.217 \text{Kp} - 5.197)} \times L^{8}$$

Use $D_{LL} = D_{LL}^M + D_{LL}^E$ for a more complete model, or just $D_{LL}^M$ for simplicity.

**Step (c): Loss Timescale**

Parameterize the loss timescale as a function of $L$ and Kp. A simplified model:

Inside the plasmasphere (plasmaspheric hiss dominates):

$$\tau_{\text{loss}}^{\text{hiss}}(L) = 10 \text{ days} \quad \text{for } L < L_{pp}$$

Outside the plasmasphere (chorus waves dominate):

$$\tau_{\text{loss}}^{\text{chorus}}(L, \text{Kp}) = \frac{3}{(\text{Kp} + 1)} \text{ days} \quad \text{for } L \geq L_{pp}$$

The plasmapause location depends on activity:

$$L_{pp} = 5.6 - 0.46 \times \text{Kp}$$

This is a crude but physically motivated parameterization: during storms (high Kp), the plasmasphere erodes, chorus waves extend inward, and loss timescales shorten.

**Step (d): Boundary Conditions**

- **Inner boundary** ($L = L_{\min} = 2.0$): Fixed (Dirichlet). The inner belt is stable, so $f(L_{\min}, t) = f_0$, where $f_0$ is the quiet-time value.
- **Outer boundary** ($L = L_{\max} = 7.0$): Time-varying. In reality, the PSD at the outer boundary is determined by solar wind-driven processes. For this project, use a simple model:

$$f(L_{\max}, t) = f_{\text{quiet}} \times \begin{cases} 5.0 & \text{if Kp} \geq 5 \text{ (enhanced source)} \\ 1.0 & \text{otherwise} \end{cases}$$

**Step (e): Initial Condition**

Start with a quiet-time PSD profile:

$$f_0(L) = f_{\text{peak}} \exp\left[-\frac{(L - L_{\text{peak}})^2}{2\sigma^2}\right]$$

with $f_{\text{peak}} = 1.0$ (normalized), $L_{\text{peak}} = 4.5$, $\sigma = 0.8$. This represents the typical quiet-time radiation belt peak in the outer zone.

**Step (f): Numerical Scheme**

Use the Crank-Nicolson method for stability. Transform the equation by defining $g = f / L^2$:

$$L^2 \frac{\partial (L^2 g)}{\partial t} = L^2 \frac{\partial}{\partial L}\left(\frac{D_{LL}}{L^2} \frac{\partial (L^2 g)}{\partial L}\right) - \frac{L^2 g}{\tau_{\text{loss}}}$$

Alternatively, work directly with $f$ and discretize:

$$\frac{f_i^{n+1} - f_i^n}{\Delta t} = \frac{1}{2}\left[\mathcal{D}(f^{n+1}) + \mathcal{D}(f^n)\right] - \frac{1}{2}\left[\frac{f_i^{n+1}}{\tau_i} + \frac{f_i^n}{\tau_i}\right]$$

where $\mathcal{D}(f)$ is the spatial diffusion operator discretized with central differences. This gives a tridiagonal system for $f^{n+1}$, solvable with the Thomas algorithm.

Time step: $\Delta t = 1$ hour (3600 s). Ensure the diffusion CFL condition $D_{LL} \Delta t / \Delta L^2 < 1$ is satisfied at the largest $L$ and highest Kp — if not, reduce $\Delta t$ or use the implicit scheme.

**Step (g): Drive with a Real Storm**

Obtain a Kp time series for a real geomagnetic storm (e.g., the 2003 Halloween storms, or the 2015 St. Patrick's Day storm). Run the simulation for a period covering quiet before the storm, the storm itself, and recovery (e.g., 10-14 days total).

### 2.4 Analysis

Create the following visualizations:
1. **$f(L, t)$ color map:** $L$ on the $y$-axis, time on the $x$-axis, PSD as color. This is the standard way to visualize radiation belt dynamics.
2. **PSD profiles at key times:** $f(L)$ at quiet before storm, during storm maximum, and several days into recovery. Identify where enhancement and depletion occur.
3. **Time profiles at fixed $L$:** $f(t)$ at $L = 4, 5, 6$ — showing the evolution at different radial distances.
4. **Kp overlay:** Plot Kp alongside $f$ to correlate driving with response.

### 2.5 Physical Interpretation

Your simulation should show:
- **Enhancement at $L \sim 4-5$:** Inward radial diffusion from the enhanced outer boundary, energizing electrons as they move to lower $L$ (conservation of the first adiabatic invariant in a stronger field).
- **Depletion at $L > 5-6$:** During the storm main phase, outward radial transport to the magnetopause (where particles are lost) and enhanced chorus wave loss create a dropout.
- **Slow recovery:** After the storm, radial diffusion gradually refills the depleted region from both boundaries.
- **Timescale separation:** The inner belt ($L < 3$) barely changes (negligible $D_{LL}$), while the outer belt ($L > 4$) responds dramatically within hours to days.

---

## 3. Project 3: Space Weather Dashboard

### 3.1 Overview

This project integrates multiple components from previous lessons into a monitoring dashboard that displays space weather conditions and simple model-based forecasts. The dashboard visualizes solar wind data, computes derived quantities, and provides at-a-glance assessment of space weather conditions.

### 3.2 Components

The dashboard has five main panels:

**Panel (a): Solar Wind Monitor**

Display time series of solar wind parameters from OMNI or DSCOVR data:
- Solar wind speed $v$ (km/s) — color-coded: green (<400), yellow (400-600), red (>600)
- Proton density $n$ (cm$^{-3}$) — log scale
- IMF magnitude $|B|$ (nT) and $B_z$ GSM (nT) — highlight southward ($B_z < 0$) in red
- Dynamic pressure $P_{\text{dyn}}$ (nPa)

Show the most recent 24-48 hours of data with the current value prominently displayed.

**Panel (b): Magnetopause Position**

Compute the subsolar magnetopause standoff distance using the Shue et al. (1998) model:

$$r = R_0 \left(\frac{2}{1 + \cos\theta}\right)^\alpha$$

where:

$$R_0 = \left(11.4 + K \cdot B_z\right) P_{\text{dyn}}^{-1/6.6}$$

with $K = 0.013$ for $B_z \geq 0$ and $K = 0.140$ for $B_z < 0$, and:

$$\alpha = (0.58 - 0.010 \cdot B_z)(1 + 0.010 \cdot P_{\text{dyn}})$$

Display $R_0$ (in $R_E$) and a schematic cross-section of the magnetosphere showing the magnetopause, with geostationary orbit (6.6 $R_E$) marked. When $R_0 < 6.6 R_E$, highlight the warning: geostationary satellites are outside the magnetopause and directly exposed to the solar wind.

**Panel (c): Dst Prediction**

Run the Burton model forward using current solar wind data to predict Dst for the next 6 hours:

1. Initialize from the current observed Dst value.
2. For the forecast period, assume solar wind conditions remain at current values (persistence forecast) or use a simple trend extrapolation.
3. Display current Dst with the 6-hour forecast trajectory.
4. Color-code by storm classification: green (> $-30$), yellow ($-30$ to $-50$), orange ($-50$ to $-100$), red ($-100$ to $-250$), purple (< $-250$).

**Panel (d): Aurora Probability**

Estimate the equatorward boundary of the auroral oval using the Feldstein relationship:

$$\Lambda_{\text{eq}} \approx 67° - 2° \times \text{Kp}$$

where $\Lambda_{\text{eq}}$ is the geomagnetic latitude of the equatorward boundary in degrees. Convert to geographic latitude using the magnetic field model.

Display:
- Current Kp value
- Equatorward boundary latitude
- A polar projection map showing the approximate auroral oval
- For specific cities, indicate whether aurora may be visible: e.g., if the observer is poleward of $\Lambda_{\text{eq}}$ and it is local nighttime

**Panel (e): NOAA Scale Indicators**

Display color-coded NOAA space weather scale status:

- **G-scale (Geomagnetic):** Based on current Kp. Show G0-G5 with appropriate color and description.
- **S-scale (Solar Radiation):** Based on GOES >10 MeV proton flux. Show S0-S5.
- **R-scale (Radio Blackout):** Based on GOES X-ray flux. Show R0-R5.

Each indicator should show: current level, description, and key impact.

### 3.3 Implementation Options

**Option A: Static dashboard (matplotlib)**

Use matplotlib with subplots for each panel. Generate a static image updated at regular intervals. This is the simplest approach and sufficient for learning the physics.

Layout suggestion:
- 3 rows, 2 columns
- Top-left: solar wind time series (4 subplots stacked)
- Top-right: magnetopause schematic
- Middle-left: Dst observation + forecast
- Middle-right: auroral oval map
- Bottom: NOAA scales (3 color bars)

**Option B: Interactive dashboard (Flask + Plotly)**

Build a web application using Flask for the backend and Plotly for interactive plots. This allows:
- Zooming and panning on time series
- Hovering for exact values
- Auto-refresh at configurable intervals
- Mobile-responsive layout

The Flask backend serves the data (loaded from OMNI files or a database) and runs the models. The frontend renders the plots using Plotly.js.

### 3.4 Data Source

For development and testing, use historical OMNI hourly data for a known storm event (e.g., the 2015 St. Patrick's Day storm, March 17-22, 2015). This allows you to verify your model outputs against known observations.

For a "live" demonstration, download recent OMNI data or use the DSCOVR real-time solar wind data feed (available from SWPC in JSON format).

### 3.5 Validation

- Compare your Burton model Dst prediction to the observed Dst for the test storm.
- Verify the Shue magnetopause model gives physically reasonable values: $R_0 \approx 10-11 R_E$ during quiet times, $R_0 \approx 6-8 R_E$ during storms.
- Check the auroral oval latitude against reported aurora observations during the test event.

---

## 4. Project Extensions

These extensions deepen the physics and increase the complexity of each project.

### 4.1 Extensions for Project 1

**Extension 1a: Newell Coupling Function**

Replace the simple $E_y$ driver in the Burton model with the Newell et al. (2007) coupling function:

$$\frac{d\Phi_{MP}}{dt} = v^{4/3} B_T^{2/3} \sin^{8/3}(\theta_c/2)$$

where $B_T = \sqrt{B_y^2 + B_z^2}$ is the transverse IMF and $\theta_c = \arctan(B_y / B_z)$ is the IMF clock angle (in GSM). This function better represents the rate of magnetic flux opening at the dayside magnetopause and is considered the best single-parameter predictor of magnetospheric activity. Compare the Dst prediction using this driver versus the original $E_y$.

**Extension 1b: Akasofu Epsilon Parameter**

Compare with the Akasofu $\epsilon$ parameter:

$$\epsilon = \frac{4\pi}{\mu_0} v B^2 \sin^4(\theta_c/2) l_0^2$$

where $l_0 \approx 7 R_E$ is an empirical length scale. The $\epsilon$ parameter has units of watts and estimates the total power input from the solar wind to the magnetosphere. Typical values: $10^{10}$ W (quiet) to $10^{13}$ W (extreme storm).

### 4.2 Extensions for Project 2

**Extension 2a: Local Acceleration by Chorus Waves**

Add a source term representing local acceleration of electrons by whistler-mode chorus waves. A simplified parameterization:

$$S(L, \text{Kp}) = \begin{cases} S_0 \times (\text{Kp}/5)^2 \times \exp\left[-\frac{(L - L_{\text{acc}})^2}{2 \times 0.5^2}\right] & \text{for } L > L_{pp} \\ 0 & \text{for } L \leq L_{pp} \end{cases}$$

with $L_{\text{acc}} = 4.5$ and $S_0$ chosen to produce realistic acceleration timescales (~1 day for strong chorus activity). This adds a local peak in PSD at $L \sim 4-5$ during storm recovery — a signature of internal acceleration observed by the Van Allen Probes.

Compare the simulation with and without local acceleration. Does the PSD profile develop a local peak (evidence of internal acceleration) or a monotonically increasing profile (evidence of radial diffusion dominating)?

**Extension 2b: Magnetopause Shadowing**

When the magnetopause is compressed to low $L$-shells during a storm, particles at $L > R_0$ (magnetopause distance) are lost on open drift paths. Implement this as:

$$\tau_{\text{mp}}(L, t) = \begin{cases} 0.5 \text{ hr} & \text{if } L > R_0(t) \\ \infty & \text{if } L \leq R_0(t) \end{cases}$$

where $R_0(t)$ comes from the Shue magnetopause model driven by observed $P_{\text{dyn}}$ and $B_z$. Combine with the wave-driven loss: $1/\tau_{\text{total}} = 1/\tau_{\text{wave}} + 1/\tau_{\text{mp}}$.

This produces the dramatic outer-zone flux dropout observed during storm main phase.

### 4.3 Extensions for Project 3

**Extension 3a: Satellite Orbit Overlay**

Add satellite orbits to the dashboard:
- **ISS** (altitude ~420 km, inclination 51.6°): Show when the ISS passes through the South Atlantic Anomaly (SAA) — the region where the inner radiation belt dips to low altitudes due to the offset between the geographic and magnetic axes. SAA boundaries: approximately 0°-60°W longitude, 15°-45°S latitude.
- **GPS constellation** (altitude ~20,200 km, $L \sim 4.2$): Show when GPS satellites are in the heart of the outer radiation belt and might experience single-event upsets.
- **GEO satellites** (altitude ~35,786 km, $L = 6.6$): Show when they might be outside the magnetopause during storm compression.

**Extension 3b: GIC Risk Indicator**

Add a geomagnetically induced current (GIC) risk panel. GIC risk correlates with $|dB/dt|$ at the ground, which in turn correlates with AE and the rate of change of Dst. A simple proxy:

$$\text{GIC risk} \propto \left|\frac{d\text{Dst}}{dt}\right| + 0.1 \times \text{AE}$$

Display as a color-coded meter with thresholds for power grid operator awareness.

---

## 5. References and Data Sources

### 5.1 Data Portals

| Resource | URL | Description |
|----------|-----|-------------|
| OMNI (NASA/GSFC) | omniweb.gsfc.nasa.gov | Multi-source solar wind data at bow shock nose, 1963-present |
| GOES real-time | swpc.noaa.gov/products/goes-magnetometer | Real-time X-ray, proton, magnetic field at GEO |
| SuperMAG | supermag.jhuapl.edu | Ground magnetometer network, 300+ stations |
| SWPC (NOAA) | swpc.noaa.gov | Operational forecasts, warnings, alerts |
| Van Allen Probes | rbsp-ect.newmexiconsortium.org | Radiation belt particle and field data (2012-2019) |
| SDO/HMI | jsoc.stanford.edu | Solar magnetograms and images |
| CDAWeb | cdaweb.gsfc.nasa.gov | Multi-mission space physics data archive |
| DSCOVR | swpc.noaa.gov/products/real-time-solar-wind | Real-time solar wind at L1 |

### 5.2 Python Packages

| Package | Purpose |
|---------|---------|
| spacepy | Space physics data analysis, coordinate transforms, radiation belt models |
| sunpy | Solar data access and analysis |
| geopack | Magnetic field models (Tsyganenko), coordinate transforms |
| cdflib | Read NASA CDF data files |
| astropy | General astronomy utilities, units, coordinates |
| heliopy (deprecated) / astrospice | Heliophysics data download |
| kamodo | CCMC model output access and visualization |

### 5.3 Key References

- Burton, R. K., McPhetres, R. L., & Russell, C. T. (1975). An empirical relationship between interplanetary conditions and Dst. Journal of Geophysical Research, 80(31), 4204-4214.
- Brautigam, D. H., & Albert, J. M. (2000). Radial diffusion analysis of outer radiation belt electrons during the October 9, 1990, magnetic storm. Journal of Geophysical Research, 105(A1), 291-309.
- Shue, J.-H., et al. (1998). Magnetopause location under extreme solar wind conditions. Journal of Geophysical Research, 103(A8), 17691-17700.
- Newell, P. T., et al. (2007). A nearly universal solar wind-magnetosphere coupling function. Journal of Geophysical Research, 112, A01206.
- Camporeale, E., Wing, S., & Johnson, J. R. (Eds.) (2018). Machine Learning Techniques for Space Weather. Elsevier.

---

## Exercises

### Exercise 1: Burton Model Warm-Up

Implement and validate the Burton model using a short, manually constructed solar wind time series.

**Steps:**

1. Create a synthetic 48-hour solar wind dataset with the following structure:
   - Hours 0-11: Quiet conditions — $v = 400$ km/s, $n = 5$ cm$^{-3}$, $B_z = +2$ nT
   - Hours 12-24: Storm main phase — $v = 600$ km/s, $n = 15$ cm$^{-3}$, $B_z = -10$ nT
   - Hours 25-48: Recovery — $v = 450$ km/s, $n = 8$ cm$^{-3}$, $B_z = +1$ nT

2. Compute $P_{\text{dyn}}$ and $E_y$ for each hour. Apply the pressure correction to obtain $\text{Dst}^*$.

3. Integrate the Burton ODE using the Euler method with $\Delta t = 1$ hour. Use Burton's canonical parameters ($a = -4.5$ nT/(mV/m·hr), $\tau = 7.7$ hr, $E_c = 0.5$ mV/m).

4. Convert $\text{Dst}^*$ back to Dst and plot the full time series alongside the solar wind inputs.

5. Verify your result: the Dst minimum should occur a few hours after the southward $B_z$ onset, and the recovery should be approximately exponential with a timescale of ~8 hours.

**Expected outcome:** A clear storm-time Dst depression (likely $-50$ to $-100$ nT range) with exponential recovery, confirming your implementation is physically correct before you move on to real data.

---

### Exercise 2: Statistical Evaluation of Dst Prediction Models

Extend Project 1 to perform a systematic statistical comparison between the Burton model and a persistence forecast.

**Steps:**

1. Download OMNI hourly data for 2010-2020 from OMNIWeb. Clean missing values using the fill-value mask.

2. Implement the Burton model as described in Section 1.3.

3. Implement a naive **persistence forecast**: predict that Dst at time $t+1$ equals Dst at time $t$ (i.e., $\hat{D}(t+1) = D(t)$). This is a common baseline in forecasting.

4. Compute the following skill scores for both models on the test period 2015-2020:
   - RMSE
   - Mean Absolute Error (MAE)
   - Pearson correlation coefficient $r$
   - Heidke Skill Score (HSS) for storm detection: define a storm as Dst < $-50$ nT

5. Produce a Taylor diagram comparing both models against observations. On a Taylor diagram, radial distance from the origin is the normalized standard deviation, the angle gives the correlation, and the distance from the "observed" reference point gives the centered RMSE.

6. Investigate how performance changes as a function of storm intensity. Bin all storm events by peak Dst intensity ($-50$ to $-100$ nT, $-100$ to $-200$ nT, < $-200$ nT) and compute RMSE in each bin separately.

**Discussion:** Does the Burton model outperform persistence? Under what storm intensity regime does the Burton model show the most improvement? Why might simple persistence be surprisingly competitive for short lead times?

---

### Exercise 3: Radiation Belt Response to a Real Storm

Run the radial diffusion solver from Project 2 for the 2003 Halloween Storm (October 29-31, 2003) and analyze the radiation belt dynamics.

**Steps:**

1. Download hourly Kp values for October 25 - November 5, 2003 from the GFZ Potsdam data service (kp.gfz-potsdam.de).

2. Set up the solver exactly as described in Section 2.3 with the Brautigam & Albert diffusion coefficients and the piecewise loss timescale model.

3. Run the simulation for the full 12-day period. Save the full $f(L, t)$ array at every hour.

4. Create the following four-panel figure:
   - Panel 1: Kp time series (top panel)
   - Panel 2: $f(L, t)$ color map (log scale) — use `pcolormesh` with `norm=LogNorm()`
   - Panel 3: $f(L)$ profiles at four times: quiet (Oct 25), storm onset (Oct 29 0 UT), storm peak (Oct 29 12 UT), recovery (Nov 3)
   - Panel 4: $f(t)$ at $L = 4.0$, $L = 5.0$, $L = 6.0$

5. Identify whether your simulation shows a flux dropout at high $L$ during the storm main phase (the signature of magnetopause shadowing — even without explicitly implementing it, the enhanced loss term should produce some dropout).

6. Compare your simulated PSD profile at recovery with the quiet-time profile. Is the recovery complete? Why or why not?

**Deliverable:** A four-panel figure and a 200-word interpretation of the radiation belt dynamics you observe.

---

### Exercise 4: Space Weather Dashboard for the 2015 St. Patrick's Day Storm

Build a static matplotlib dashboard (Option A from Section 3.3) driven by historical OMNI data for the 2015 St. Patrick's Day Storm (March 14-22, 2015).

**Steps:**

1. Download hourly OMNI data for March 14-22, 2015 from OMNIWeb. Required columns: $v$, $n$, $B$, $B_y$, $B_z$ (GSM), Dst, Kp.

2. Implement all five panels described in Section 3.2:
   - Panel a: 4-variable solar wind time series ($v$, $n$, $|B|$, $B_z$)
   - Panel b: Shue magnetopause standoff distance $R_0(t)$ over the event, with a horizontal line at $6.6 R_E$ (geostationary orbit)
   - Panel c: Observed Dst overlaid with the Burton model prediction
   - Panel d: Equatorward auroral boundary latitude from the Feldstein relation
   - Panel e: NOAA G-scale level as a step function of Kp

3. Add vertical dashed lines marking the storm sudden commencement (SSC) and Dst minimum.

4. For each panel, annotate any moment when a threshold is crossed:
   - Solar wind speed > 600 km/s
   - $B_z < -10$ nT sustained for > 3 hours
   - $R_0 < 8 R_E$
   - Auroral boundary < 55° latitude

5. Compute the overall RMSE between your Burton model Dst prediction and observed Dst over the storm period.

**Stretch goal:** Add a sixth panel showing the Newell coupling function $d\Phi_{MP}/dt$ alongside $E_y$. Discuss which better correlates with the observed Dst depression.

---

### Exercise 5: Coupling Function Comparison (Advanced)

Systematically compare three solar wind-magnetosphere coupling functions as drivers for the Burton model over the full 2000-2020 OMNI dataset, quantifying which produces the best Dst predictions across different storm categories.

**Steps:**

1. Implement all three coupling functions:
   - $E_y = -v \cdot B_z$ (standard Burton driver, mV/m)
   - Newell function: $\frac{d\Phi_{MP}}{dt} = v^{4/3} B_T^{2/3} \sin^{8/3}(\theta_c/2)$
   - Akasofu epsilon: $\epsilon = \frac{4\pi}{\mu_0} v B^2 \sin^4(\theta_c/2) l_0^2$ with $l_0 = 7 R_E$

2. For each coupling function, re-fit the injection efficiency parameter $a$ (Burton parameter) by minimizing RMSE over the training period 2000-2014. Keep $\tau = 7.7$ hr and $E_c = 0.5$ mV/m fixed (or also optimize $E_c$ as a threshold).

3. Evaluate all three fitted models on the test period 2015-2020. Compute RMSE and $r$ separately for:
   - Weak storms ($-30 < \text{Dst} < -50$ nT)
   - Moderate storms ($-50 < \text{Dst} < -100$ nT)
   - Intense storms (Dst < $-100$ nT)

4. Create a bar chart showing RMSE by storm category and coupling function. Which coupling function performs best for intense storms?

5. Analyze cases where the models disagree most: find 3-5 events where the Newell-driven prediction is substantially better (or worse) than the $E_y$-driven prediction. What is unusual about the solar wind conditions during those events?

**Discussion:** The Newell function weights the IMF clock angle more heavily than $E_y$ does. Based on your results, in what solar wind regimes does clock angle matter most for predicting Dst?

---

**Previous**: [AI/ML for Space Weather](./15_AI_ML_for_Space_Weather.md)

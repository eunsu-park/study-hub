# Projects

## Learning Objectives

- Apply solar physics concepts from multiple lessons to realistic computational projects
- Integrate knowledge of solar irradiance, magnetic fields, and flare physics into quantitative models
- Practice scientific data analysis using real solar datasets and observational archives
- Build intuition for the relationship between photospheric magnetic fields and coronal/heliospheric structure through PFSS modeling
- Develop skills in scientific modeling, from simple empirical fits to physics-based extrapolations
- Understand the challenges and evaluation metrics of solar flare prediction as a machine learning problem

---

## 1. Project 1: Solar Irradiance Model

> **Physics Context**: The Sun's total energy output is not truly constant — it varies by ~0.1% over the 11-year magnetic activity cycle and by larger fractions on shorter timescales. This project asks: *can we predict the Sun's brightness from a single observable (sunspot number)?* The answer reveals a beautiful competition between two magnetic phenomena — dark sunspots that block light and bright faculae that enhance it — and shows that the net effect is counter-intuitive: the Sun is actually *brighter* at solar maximum despite having more dark spots, because the facular brightening wins.

### 1.1 Background and Motivation

Total Solar Irradiance (TSI) — the total electromagnetic energy flux from the Sun at 1 AU — is approximately 1361 W/m$^2$, historically called the "solar constant." However, continuous space-based measurements since 1978 have revealed that TSI varies by approximately 0.1% ($\sim$1 W/m$^2$ peak-to-peak) over the 11-year solar cycle, and by larger fractions on shorter timescales during the passage of active regions.

Understanding TSI variability is important for two reasons:
1. **Climate science**: Even small, sustained changes in solar input can influence Earth's climate. The ~1 W/m$^2$ solar cycle variation translates to ~0.17 W/m$^2$ in global average forcing (accounting for geometry and albedo), comparable in magnitude to decade-scale changes in greenhouse forcing.
2. **Stellar physics**: The Sun serves as the calibrator for stellar variability models. Understanding what drives solar irradiance changes helps interpret photometric variations of other Sun-like stars.

### 1.2 Physics of TSI Variability

TSI variations are driven by the competing effects of two types of magnetic features on the solar surface:

**Sunspots** (dark features): A sunspot's umbra has a temperature of ~3500-4000 K (compared to the photospheric ~5800 K). Since radiated flux goes as $T^4$, the contrast is:

$$C_{\text{spot}} = 1 - \left(\frac{T_{\text{spot}}}{T_{\text{phot}}}\right)^4 \approx 1 - \left(\frac{4000}{5800}\right)^4 \approx 0.77$$

A sunspot radiates only about 23% as much as an equivalent area of quiet photosphere. Sunspots therefore reduce TSI. The deficit is proportional to the total sunspot area:

$$\Delta \text{TSI}_{\text{spot}} = -C_{\text{spot}} \cdot \frac{A_{\text{spot}}}{A_{\text{hemisphere}}} \cdot \text{TSI}_0$$

**Faculae** (bright features): Faculae are regions of enhanced magnetic field (plage in the chromosphere) that appear bright, especially near the solar limb. Their brightness excess arises because the magnetic field evacuates the flux tube interior, allowing the observer to see deeper (and therefore hotter) layers through the tube walls. The facular brightening overcompensates the sunspot darkening on cycle timescales because faculae are more widely distributed and persist longer than the spots that generate them.

The net TSI variation is the sum of these effects:

$$\frac{\Delta \text{TSI}}{\text{TSI}_0} = -\alpha \cdot f_{\text{spot}} + \beta \cdot f_{\text{fac}}$$

where:
- $f_{\text{spot}}$ is the fractional area of the hemisphere covered by sunspots
- $f_{\text{fac}}$ is the fractional area covered by faculae
- $\alpha \approx 0.3$ is the effective spot contrast (accounting for penumbra, which is less dark than umbra)
- $\beta \approx 0.001$ is the facular contrast per unit area (much smaller individually, but faculae cover a much larger area)

### 1.3 Simple Empirical Model

For a first-order model, we can use the **sunspot number** $R$ as a proxy for both spot and facular coverage:

$$f_{\text{spot}} \approx a_1 \cdot R$$
$$f_{\text{fac}} \approx a_2 \cdot R$$

where $a_1$ and $a_2$ are empirical scaling factors. Since facular areas scale with but exceed sunspot areas (typically by a factor of 10-20), and facular brightening dominates on cycle timescales, the net effect is:

$$\Delta \text{TSI} \approx c_0 + c_1 \cdot R$$

with $c_1 > 0$ — TSI increases at solar maximum despite more sunspots, because the facular brightening wins over the sunspot darkening when averaged over the full cycle.

On rotational timescales (~27 days), however, large sunspot groups can temporarily reduce TSI by up to 4-5 W/m$^2$ as they transit the visible disk.

### 1.4 Project Implementation

**Step 1: Obtain historical sunspot number data.** The SILSO (Sunspot Index and Long-term Solar Observations) database at the Royal Observatory of Belgium provides monthly and daily international sunspot numbers from 1700 to present. The Version 2.0 series (post-2015 recalibration) should be used.

**Step 2: Build the irradiance model.** Construct TSI as a function of monthly sunspot number:

$$\text{TSI}(t) = S_0 + c_1 \cdot R(t)$$

where $S_0 \approx 1360.5$ W/m$^2$ (quiet-Sun TSI) and $c_1$ is determined by fitting to measured TSI data.

**Step 3: Calibrate against measurements.** Compare with actual TSI measurements from SORCE/TIM (2003-2020), TSIS-1/TIM (2018-present), and the composite TSI record from PMOD/WRC (1978-present). Determine the best-fit $c_1$ using least-squares regression.

**Step 4: Evaluate model performance.** Calculate the correlation coefficient, RMS residual, and examine systematic failures (e.g., the model misses individual sunspot group transits because monthly $R$ smooths out rotational variations).

**Step 5: Analyze implications.** Plot TSI over multiple solar cycles. Estimate the radiative forcing change between Maunder Minimum (1645-1715, when $R \approx 0$) and modern solar maximum. Consider that the Maunder Minimum TSI decrease is estimated at 1-3 W/m$^2$, implying a global average forcing change of ~0.2-0.5 W/m$^2$ — small compared to current anthropogenic forcing of ~2.7 W/m$^2$, but not negligible for pre-industrial climate variability.

### 1.5 Physical Discussion Points

- Why does facular brightening dominate over sunspot darkening on cycle timescales?
- What happens to the "missing" energy blocked by sunspots? Is it stored and re-emitted later, or redistributed to surrounding areas?
- How does the sunspot area vs. sunspot number relationship affect model accuracy?
- What are the limitations of using a single proxy (sunspot number) for TSI modeling?
- How do the SATIRE (Spectral And Total Irradiance Reconstructions) and NRLTSI2 models improve upon this simple approach?

---

## 2. Project 2: Potential Field Source Surface (PFSS) Extrapolation

> **Physics Context**: The coronal magnetic field is the puppet master behind virtually all solar activity — it stores the energy released in flares, channels the solar wind, and shapes CMEs. Yet it is extraordinarily difficult to measure directly. This project tackles the question: *given only the photospheric magnetic field (which we can measure), can we reconstruct the 3D coronal field?* The PFSS model answers "yes, approximately" by assuming no currents flow in the corona (a current-free, or potential, field). Despite this drastic simplification, the model successfully reproduces coronal holes, helmet streamers, and the large-scale field topology that governs the heliosphere.

### 2.1 Background and Motivation

The photospheric magnetic field is relatively well measured (by SDO/HMI, GONG, etc.), but the coronal magnetic field — which controls coronal heating, solar wind structure, flare energy storage, and CME initiation — cannot be directly measured in most cases (DKIST's coronal measurements are a recent exception with limited spatial coverage).

The **Potential Field Source Surface (PFSS)** model is the simplest and most widely used method for extrapolating the photospheric field into the corona. Despite its simplicity (it assumes the corona is current-free), it reproduces the large-scale coronal structure remarkably well and is the standard starting point for solar wind and space weather models.

### 2.2 Mathematical Formulation

The PFSS model solves Laplace's equation in a spherical shell between the photosphere ($r = R_\odot$) and the "source surface" ($r = R_{ss}$, typically 2.5 $R_\odot$):

$$\nabla^2 \Phi = 0$$

where $\mathbf{B} = -\nabla\Phi$ (current-free condition, $\nabla \times \mathbf{B} = 0$).

**Boundary conditions:**
- **Inner boundary** ($r = R_\odot$): $B_r(R_\odot, \theta, \phi)$ is specified from the observed photospheric magnetogram
- **Outer boundary** ($r = R_{ss}$): $B_\theta = B_\phi = 0$ (purely radial field), simulating the effect of the solar wind stretching field lines open

The radial field condition at $R_{ss}$ is the key simplification. It represents the physical reality that beyond a certain height, the solar wind's dynamic pressure overcomes magnetic tension, forcing all field lines to be open and radial. The choice $R_{ss} = 2.5 R_\odot$ is empirical, balancing the need to open sufficient flux against preserving closed-field structures observed in the low corona.

### 2.3 Spherical Harmonic Solution

The solution to Laplace's equation in spherical coordinates is expressed as a sum of spherical harmonics:

$$\Phi(r, \theta, \phi) = R_\odot \sum_{l=1}^{l_{\max}} \sum_{m=0}^{l} \left[ a_{lm} \left(\frac{r}{R_\odot}\right)^l + b_{lm} \left(\frac{R_\odot}{r}\right)^{l+1} \right] Y_l^m(\theta, \phi)$$

where $Y_l^m(\theta, \phi) = P_l^m(\cos\theta)(c_{lm}\cos m\phi + d_{lm}\sin m\phi)$ are the real spherical harmonics, and $P_l^m$ are the associated Legendre polynomials.

The coefficients $a_{lm}$ and $b_{lm}$ are determined by applying both boundary conditions:

At $r = R_\odot$: The radial field component gives:

$$B_r(R_\odot, \theta, \phi) = -\frac{\partial \Phi}{\partial r}\bigg|_{r=R_\odot} = -\sum_{l,m} \left[ l \cdot a_{lm} - (l+1) \cdot b_{lm} \right] Y_l^m$$

At $r = R_{ss}$: The condition $B_\theta = B_\phi = 0$ implies $\Phi(R_{ss}) = \text{const}$, which gives:

$$a_{lm} \left(\frac{R_{ss}}{R_\odot}\right)^l + b_{lm} \left(\frac{R_\odot}{R_{ss}}\right)^{l+1} = 0$$

These two conditions uniquely determine $a_{lm}$ and $b_{lm}$ from the observed $B_r(R_\odot, \theta, \phi)$.

### 2.4 Project Implementation

**Step 1: Create a synthetic magnetogram.** Rather than using real (noisy, complex) data for a first implementation, construct a synthetic $B_r(R_\odot, \theta, \phi)$ on a latitude-longitude grid:

- **Dipole component**: $B_r \propto \cos\theta$ (mimicking the global solar dipole during solar minimum)
- **Quadrupole component**: $B_r \propto (3\cos^2\theta - 1)/2$ (representing the next-order term)
- **Optionally**: Add a localized bipolar active region (a pair of Gaussian spots of opposite polarity)

**Step 2: Compute spherical harmonic coefficients.** Decompose the synthetic magnetogram into spherical harmonics up to $l_{\max}$ (e.g., $l_{\max} = 15$ for a start). This involves numerical integration over the sphere:

$$g_{lm} = \frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!} \int B_r(R_\odot, \theta, \phi) P_l^m(\cos\theta) \cos(m\phi) \sin\theta \, d\theta \, d\phi$$

and similarly for $h_{lm}$ with $\sin(m\phi)$.

**Step 3: Evaluate the magnetic field in 3D.** Compute $B_r$, $B_\theta$, $B_\phi$ at any point $(r, \theta, \phi)$ in the shell $[R_\odot, R_{ss}]$ from the harmonic expansion and the determined $a_{lm}$, $b_{lm}$ coefficients.

**Step 4: Trace field lines.** Starting from footpoints on the photosphere, integrate the field line equation:

$$\frac{dr}{B_r} = \frac{r \, d\theta}{B_\theta} = \frac{r \sin\theta \, d\phi}{B_\phi}$$

Use a 4th-order Runge-Kutta (RK4) scheme. At each step, evaluate the magnetic field at the current position and advance the position by a small step along the field direction.

**Step 5: Classify field lines.**
- **Closed**: Field lines that start at one polarity on $R_\odot$ and return to the opposite polarity on $R_\odot$ without reaching $R_{ss}$
- **Open**: Field lines that extend from $R_\odot$ to $R_{ss}$ (they "escape" into the heliosphere)

**Step 6: Identify key structures.**
- **Coronal holes**: Regions of open field (outward or inward polarity). These are the sources of fast solar wind.
- **Streamer belt**: The boundary between open field regions of opposite polarity, where closed loops trap dense plasma
- **Heliospheric current sheet (HCS)**: The surface at $R_{ss}$ where $B_r$ changes sign, corresponding to the warped current sheet that extends throughout the heliosphere

### 2.5 Physical Discussion Points

- How does changing $R_{ss}$ affect the amount of open flux? What are the consequences for solar wind modeling?
- Why is the current-free assumption reasonable for the large-scale corona? When does it fail? (Answer: in active regions with strong shear and free energy)
- Compare your PFSS field line topology with SDO/AIA images at 193 A. Do the closed loops and coronal hole boundaries match qualitatively?
- What is the polarity of the heliospheric current sheet, and how does it relate to the interplanetary magnetic field sector structure?
- How would you extend this to a non-potential model (e.g., NLFFF — Non-Linear Force-Free Field)?

---

## 3. Project 3: Flare Prediction with Active Region Parameters

> **Physics Context**: Can we predict solar explosions before they happen? Flares release $10^{29}$–$10^{32}$ erg in minutes — energy stored in twisted, sheared coronal magnetic fields. This project explores whether measurable properties of the photospheric magnetic field (total flux, PIL complexity, magnetic shear) contain enough information to forecast flares. The result is humbling: even the best models achieve only moderate skill, suggesting that flare triggering involves a critical threshold — like an earthquake, the system is primed for failure but the precise moment of rupture may be inherently unpredictable. This connects solar physics to broader questions about predictability in complex systems.

### 3.1 Background and Motivation

Solar flare prediction is one of the "grand challenges" of space weather forecasting. The goal is to predict whether an active region will produce a significant flare (M-class or X-class, corresponding to peak soft X-ray flux $\geq 10^{-5}$ W/m$^2$) within a specified time window (typically 24 hours).

This is fundamentally a **binary classification** problem: given a set of features describing the current state of an active region's magnetic field, predict "flare" or "no flare." The challenge is extreme class imbalance — flaring active regions are rare (perhaps 1-5% of all active region-days produce M-class or larger flares), so a naive classifier that always predicts "no flare" achieves $>95\%$ accuracy but is useless.

### 3.2 Active Region Magnetic Parameters

The key insight is that flares are powered by free magnetic energy stored in the coronal field, and this energy is related to measurable photospheric magnetic field properties. The following parameters, computable from SDO/HMI vector magnetograms, have been shown to have predictive value:

**Total unsigned magnetic flux:**

$$\Phi_{\text{total}} = \int |B_r| \, dA$$

Larger active regions (more flux) have more energy available and tend to produce larger flares. However, $\Phi_{\text{total}}$ alone is a poor predictor because many large, well-ordered regions never flare.

**Flux near the polarity inversion line (R-value):**

$$R = \int_{\text{PIL}} |B_{\text{LOS}}| \, dA$$

The polarity inversion line (PIL) is where $B_r$ changes sign. Flux concentrated near the PIL indicates closely packed opposite polarities — a configuration prone to reconnection. The R-value, introduced by Schrijver (2007), is the total unsigned flux within a narrow band (e.g., 15 Mm) around the PIL, weighted by strong-field pixels. It is one of the single best predictors of major flares.

**Total magnetic field gradient along the PIL:**

$$G_{\text{PIL}} = \sum_{\text{PIL}} |\nabla B_{\text{LOS}}|$$

Strong gradients near the PIL indicate sharp transitions between opposite polarities — highly sheared configurations that store free energy.

**Magnetic shear angle:**

$$\Delta\psi = \arccos\left(\frac{\mathbf{B}_{\text{obs}} \cdot \mathbf{B}_{\text{pot}}}{|\mathbf{B}_{\text{obs}}||\mathbf{B}_{\text{pot}}|}\right)$$

The angle between the observed (non-potential) transverse field and the potential (current-free) transverse field. Large shear indicates strong electric currents and stored free energy. Regions with mean shear $> 40°$ along the PIL are significantly more flare-productive.

**Total photospheric free energy proxy:**

$$\rho_{\text{free}} = \sum (B_{\text{obs}} - B_{\text{pot}})^2$$

Proportional to the excess magnetic energy density above the potential field state. This directly estimates the energy available for release.

**Additional features**: Effective connected magnetic field, total unsigned current helicity, area of strong-field PIL, Ising energy (from statistical mechanics analogy), and time-derivative features (rate of flux emergence, shear evolution).

### 3.3 Classification Methods

**Logistic regression**: A simple baseline. The log-odds of flaring is modeled as a linear combination of features:

$$\log\frac{P(\text{flare})}{1 - P(\text{flare})} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots$$

Logistic regression is interpretable (the coefficients tell you which features matter most) and provides calibrated probabilities. It typically achieves TSS ~0.4-0.6 for M-class prediction.

**Random forest**: An ensemble of decision trees, each trained on a random subsample of data and features. Random forests handle non-linear relationships, feature interactions, and are robust to overfitting. They typically achieve TSS ~0.5-0.7 and provide feature importance rankings.

**More advanced**: Support vector machines, gradient boosting (XGBoost, LightGBM), and deep learning (CNNs on magnetogram images, LSTMs on time series of features) have been explored. The best models achieve TSS ~0.6-0.8, but no model consistently exceeds TSS ~0.8, suggesting fundamental limits related to the stochastic nature of flare triggering.

### 3.4 Evaluation Metrics

Standard classification metrics are inappropriate for highly imbalanced problems. The key metric is the **True Skill Statistic (TSS)**, also known as the Hanssen-Kuipers discriminant:

$$\text{TSS} = \frac{TP}{TP + FN} - \frac{FP}{FP + TN} = \text{TPR} - \text{FPR}$$

where TP = true positives, FN = false negatives, FP = false positives, TN = true negatives, TPR = true positive rate (sensitivity/recall), FPR = false positive rate.

TSS ranges from $-1$ to $+1$:
- $\text{TSS} = 1$: Perfect prediction
- $\text{TSS} = 0$: No skill (equivalent to random guessing or constant prediction)
- $\text{TSS} < 0$: Worse than random

The crucial property of TSS is that it is **base-rate independent** — its value does not depend on the relative frequency of flaring vs. non-flaring days. This makes it appropriate for the imbalanced flare prediction problem, where accuracy would be misleading.

Other useful metrics include:
- **Brier Skill Score (BSS)**: Measures probabilistic forecast quality
- **Reliability diagram**: Checks if predicted probabilities are well-calibrated
- **ROC curve and AUC**: Receiver operating characteristic, with area under the curve (AUC) summarizing overall discriminative ability

### 3.5 Project Implementation

**Step 1: Obtain data.** The SHARP (Space-weather HMI Active Region Patches) data product from SDO/HMI provides pre-computed active region parameters, including many of the features listed above. The SHARP data catalog is available through JSOC. For this project, download a subset (e.g., 1-2 years of data) including the SHARP keywords and corresponding GOES flare associations.

**Step 2: Prepare the dataset.** For each active region at each time step (e.g., every 12 hours):
- Features: SHARP parameters ($\Phi_{\text{total}}$, R-value, mean shear, total current, etc.)
- Label: Did this active region produce an M-class or larger flare within the next 24 hours? (Binary: 1/0)

**Step 3: Handle class imbalance.** Methods include:
- Oversampling the minority class (SMOTE)
- Undersampling the majority class
- Class-weighted loss function
- Threshold optimization on the validation set

**Step 4: Train and evaluate.** Use temporal cross-validation (train on earlier data, test on later data — never the reverse, to avoid data leakage). Compute TSS, AUC, and generate reliability diagrams.

**Step 5: Analyze feature importance.** Which features contribute most to prediction? Does R-value dominate? Is the time-derivative of flux emergence informative? Compare with published results.

**Step 6: Compare with climatological baseline.** The simplest "model" is the climatological flare rate: predict a flare with probability equal to the historical flare rate for each HALE class. What TSS does this achieve? Your model should significantly exceed this baseline.

### 3.6 Physical Discussion Points

- Why is flare prediction fundamentally limited? (Hint: the trigger mechanism may be stochastic — small perturbations can initiate or prevent flaring in a critically stressed active region.)
- What additional information might improve predictions? (Coronal magnetic field? Photospheric flow patterns? Data-driven MHD simulations?)
- How does the prediction horizon affect accuracy? (Shorter horizons are more accurate but less useful for mitigation.)
- What is the role of previous flaring history? (Active regions that have already flared tend to flare again — "flare productivity" is a useful feature.)
- How would you extend this to predict flare magnitude (C/M/X) rather than just occurrence?

---

## 4. Project Extensions

### 4.1 Extensions for Project 1 (Solar Irradiance)

- **Separate spot and facular components**: Instead of using $R$ as a proxy for both, use the Photometric Sunspot Index (PSI, derived from sunspot area and position) for the darkening component and the Mg II index or F10.7 radio flux for the facular brightening. This two-component model more accurately captures rotational and cycle variations.
- **Spectral irradiance**: Extend from TSI to spectral solar irradiance (SSI). UV irradiance varies by 5-10% over the solar cycle (much more than the 0.1% TSI variation), with significant implications for ozone chemistry. Use the NRLTSI2/NRLSSI2 model framework.
- **Long-term trends**: Investigate whether there is a secular trend in TSI beyond the 11-year cycle. The Gleissberg cycle (~80-90 years) and potential Grand Minima are relevant for long-term climate modeling.
- **Comparison with stellar photometry**: Apply your TSI model concepts to understand the photometric variability of Sun-like stars observed by Kepler/TESS.

### 4.2 Extensions for Project 2 (PFSS)

- **Animate over a Carrington rotation**: Compute PFSS solutions for a sequence of synoptic magnetograms spanning one solar rotation (~27 days). Animate the field line evolution to visualize the changing coronal topology.
- **Compare with SDO/AIA**: Overlay your computed open/closed field boundaries on SDO/AIA 193 A images. Coronal holes should correspond to open-field regions, and the bright streamer belt should correspond to the boundary between open-field domains.
- **Wang-Sheeley-Arge (WSA) model**: Use the PFSS solution to compute the expansion factor $f_s$ of each open flux tube:

$$f_s = \left(\frac{R_\odot}{R_{ss}}\right)^2 \frac{B(R_\odot)}{B(R_{ss})}$$

The WSA model relates $f_s$ and the angular distance from the nearest coronal hole boundary to the solar wind speed at 1 AU. Implement this empirical relationship and compare with observed solar wind speed (from OMNI database or DSCOVR).
- **Non-potential extension**: Implement the Current Sheet Source Surface (CSSS) model, which includes a horizontal current sheet at the source surface, providing a more realistic heliospheric field topology.

### 4.3 Extensions for Project 3 (Flare Prediction)

- **Multi-class prediction**: Extend from binary (M+/no-M+) to multi-class (C/M/X or quiet/C/M/X). Use ordinal regression or multi-class extensions of your classifier.
- **Time series features**: Instead of snapshot parameters, compute rolling statistics (mean, trend, variability) of SHARP parameters over 6, 12, 24, and 48-hour windows. Rapid changes in magnetic complexity may be precursors to flares.
- **Deep learning on magnetograms**: Train a CNN directly on HMI magnetogram image patches (rather than pre-computed scalar features). This allows the model to learn spatial patterns (PIL morphology, shear distribution) without manual feature engineering.
- **LSTM for temporal evolution**: Use a Long Short-Term Memory (LSTM) recurrent neural network on time sequences of SHARP features, allowing the model to learn temporal patterns of flux emergence and shear evolution that precede flares.
- **Ensemble approaches**: Combine logistic regression, random forest, and CNN predictions using stacking or blending (cross-reference with the Machine_Learning topic, particularly the stacking/blending lesson).

---

## 5. References and Data Sources

### 5.1 Data Archives

- **SILSO (Sunspot Index and Long-term Solar Observations)**: Monthly and daily international sunspot numbers, 1700-present. https://www.sidc.be/silso/
- **JSOC (Joint Science Operations Center)**: SDO/HMI magnetograms, Dopplergrams, AIA images, and SHARP active region data products. http://jsoc.stanford.edu/
- **GOES X-ray flux**: NOAA/NCEI Space Weather data, including the definitive GOES flare list used for training flare prediction models. https://www.ncei.noaa.gov/
- **OMNI database**: Solar wind parameters at 1 AU, combining data from multiple L1 monitors (ACE, Wind, DSCOVR). https://omniweb.gsfc.nasa.gov/
- **SOHO/LASCO CME catalog**: The comprehensive catalog of over 40,000 CMEs detected since 1996. https://cdaw.gsfc.nasa.gov/CME_list/
- **Parker Solar Probe and Solar Orbiter data**: SPDF/CDAWeb for in-situ data, individual instrument team websites for calibrated science products. https://cdaweb.gsfc.nasa.gov/

### 5.2 Software Tools

- **SunPy**: Open-source Python library for solar physics data analysis, including data retrieval (Fido), coordinate transformations, and visualization. https://sunpy.org/
- **pfsspy**: Python package for PFSS extrapolation, directly reading GONG and HMI synoptic maps. Well-documented and actively maintained.
- **aiapy**: SDO/AIA data analysis utilities (response functions, PSF deconvolution, DEM).
- **scikit-learn**: For machine learning classifiers (logistic regression, random forest, SVM, metrics).
- **Astropy**: General astronomical Python utilities (coordinates, units, FITS file handling).
- **Helioviewer / JHelioviewer**: Browser-based and Java-based tools for browsing multi-mission solar images with overlays and time-lapse capability.

### 5.3 Key References

- Schrijver, C. J. (2007): "A Characteristic Magnetic Field Pattern Associated with All Major Solar Flares and Its Use in Flare Forecasting." The paper introducing the R-value as a flare predictor.
- Altschuler, M. D. & Newkirk, G. (1969): "Magnetic Fields and the Structure of the Solar Corona." The original PFSS model paper.
- Lean, J. (2000): "Evolution of the Sun's Spectral Irradiance Since the Maunder Minimum." Foundational work on long-term irradiance reconstruction.
- Bobra, M. G. & Couvidat, S. (2015): "Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data with a Machine-Learning Algorithm." Machine learning applied to SHARP parameters.
- Riley, P. et al. (2019): "On the Relationship Between Sunspot Number and Solar Spectral Irradiance." Modern analysis of the sunspot-irradiance connection.
- Yeates, A. R. et al. (2018): "Global Non-potential Magnetic Models of the Solar Corona." Review of coronal magnetic field modeling beyond PFSS.

### 5.4 Cross-References to Other Topics

These projects naturally connect to several other topics in the study collection:

- **Machine_Learning**: Classification algorithms, evaluation metrics, feature engineering, handling class imbalance, ensemble methods (stacking/blending)
- **Data_Science**: Exploratory data analysis, time series analysis, statistical hypothesis testing
- **Deep_Learning**: CNN architectures for image classification (magnetogram-based flare prediction), LSTM for time series
- **Python**: Data handling (NumPy, Pandas), scientific computing (SciPy)
- **Numerical_Simulation**: ODE integration (RK4 for field line tracing), spherical harmonics
- **Mathematical_Methods**: Spherical harmonics, Legendre polynomials, Laplace's equation in spherical coordinates

---

## Practice Problems

**Problem 1**: Using the TSI model $\text{TSI}(t) = 1360.5 + 0.008 \times R(t)$ W/m$^2$ (where $R$ is the monthly sunspot number), calculate the TSI for (a) solar minimum ($R = 5$), (b) moderate solar maximum ($R = 120$), and (c) strong solar maximum ($R = 200$). What is the peak-to-peak variation? Express this as a percentage of $S_0$. What global average radiative forcing change does this correspond to (use the factor 0.175 to convert TSI change to global average forcing)?

**Problem 2**: In the PFSS model, the source surface is typically set at $R_{ss} = 2.5 R_\odot$. For a pure dipole field, the radial field at the source surface is $B_r(R_{ss}) = B_0 (R_\odot/R_{ss})^3 \cos\theta$ (for the potential field in a spherical shell). Calculate the total unsigned open magnetic flux $\Phi_{\text{open}} = \int |B_r(R_{ss})| \, dA_{ss}$ in terms of $B_0$ and $R_\odot$. If $B_0 = 5$ G (typical quiet-Sun polar field), calculate $\Phi_{\text{open}}$ in Mx. Compare with the observed value of $\sim 3 \times 10^{22}$ Mx.

**Problem 3**: An active region has the following SHARP parameters: total unsigned flux $\Phi = 5 \times 10^{22}$ Mx, R-value $= 3 \times 10^{21}$ Mx, mean shear angle $= 45°$. A logistic regression model gives $\log(P/(1-P)) = -5.0 + 2.0 \times \log_{10}(R/10^{20}) + 0.05 \times \bar{\psi}$. Calculate the predicted probability of an M-class flare within 24 hours. If the climatological rate is 3%, does this active region have elevated risk?

**Problem 4**: You are evaluating a flare prediction model that, on a test set of 1000 active region-days (30 flaring, 970 non-flaring), produces TP = 20, FN = 10, FP = 100, TN = 870. Calculate the accuracy, TPR (recall), FPR, precision, F1 score, and TSS. A colleague's model on the same test set gives TP = 25, FN = 5, FP = 200, TN = 770. Compare the two models using TSS and discuss which is more useful operationally.

---

**Previous**: [Modern Solar Missions](./15_Modern_Solar_Missions.md)

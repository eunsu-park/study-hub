# Solar Magnetic Fields

## Learning Objectives

- Explain the Zeeman effect and how it is used to measure photospheric magnetic fields
- Describe Stokes polarimetry and the inversion techniques used to derive magnetic field vectors
- Derive the Potential Field Source Surface (PFSS) model equations and understand its assumptions
- Understand the Non-Linear Force-Free Field (NLFFF) extrapolation and why the corona is force-free
- Define magnetic helicity and free energy, and explain their role in solar eruptions
- Describe magnetic topology concepts including null points, separatrices, and quasi-separatrix layers (QSLs)
- Relate magnetic field measurements and models to flare and CME prediction

---

## 1. Magnetic Field Measurement: Zeeman Effect

### 1.1 Physical Basis

The **Zeeman effect**, discovered by Pieter Zeeman in 1896, is the splitting of spectral lines in the presence of a magnetic field. It is the primary method for measuring magnetic fields on the Sun and remains one of the most direct measurements in all of astrophysics.

In quantum mechanical terms, a magnetic field lifts the degeneracy of atomic energy levels with different magnetic quantum numbers $m_J$. For a transition between upper level $J_u$ and lower level $J_l$, the frequency shift of a component with $\Delta m_J = m_{J,u} - m_{J,l}$ is:

$$\Delta \nu = \frac{e B}{4\pi m_e} (g_u m_{J,u} - g_l m_{J,l})$$

where $g_u$ and $g_l$ are the Lande g-factors of the upper and lower levels. In terms of wavelength:

$$\Delta \lambda = \frac{e \lambda^2 B}{4\pi m_e c} (g_u m_{J,u} - g_l m_{J,l})$$

### 1.2 Normal Zeeman Effect

The simplest case is a transition between $J = 0$ and $J = 1$ levels (or more generally, when both levels have the same Lande factor). This produces the **normal Zeeman triplet**:

- **$\pi$ component** ($\Delta m_J = 0$): unshifted, linearly polarized parallel to the magnetic field (projected onto the plane of the sky)
- **$\sigma^+$ component** ($\Delta m_J = +1$): shifted to longer wavelength, circularly polarized when viewed along the field
- **$\sigma^-$ component** ($\Delta m_J = -1$): shifted to shorter wavelength, circularly polarized (opposite sense)

The wavelength splitting of the $\sigma$ components from line center is:

$$\Delta \lambda_B = \frac{e \lambda_0^2 B}{4\pi m_e c} g_{\text{eff}}$$

where $g_{\text{eff}}$ is the effective Lande g-factor of the transition. Numerically:

$$\Delta \lambda_B = 4.67 \times 10^{-13} \, g_{\text{eff}} \, \lambda_0^2 \, B$$

with $\Delta \lambda_B$ in angstroms, $\lambda_0$ in angstroms, and $B$ in gauss.

**Example**: For the Fe I 6173 A line used by SDO/HMI ($g_{\text{eff}} = 2.5$):

$$\Delta \lambda_B = 4.67 \times 10^{-13} \times 2.5 \times (6173)^2 \times B = 0.044 \, B \text{ mA}$$

For a 1000 G (0.1 T) sunspot field, $\Delta \lambda_B \approx 44$ mA, which is measurable but much smaller than the thermal Doppler width of the line ($\sim 100$ mA). For weak fields (below a few hundred gauss), the splitting is unresolved and one must rely on the polarization signal rather than the splitting itself.

### 1.3 Anomalous Zeeman Effect

When the upper and lower levels have different Lande factors or when $J > 1$, the transition splits into more than three components -- this is the **anomalous Zeeman effect** (which is actually the more common case). The splitting pattern depends on the specific quantum numbers involved, and the effective Lande factor provides a weighted average splitting.

The Lande g-factor for a level with quantum numbers $L$, $S$, $J$ in LS coupling is:

$$g_J = 1 + \frac{J(J+1) + S(S+1) - L(L+1)}{2J(J+1)}$$

Spectral lines with large $g_{\text{eff}}$ are preferred for magnetometry because they are more sensitive to magnetic fields. The most commonly used solar magnetic field diagnostics include:

| Line | $\lambda$ (A) | $g_{\text{eff}}$ | Used by |
|------|----------|-----------|---------|
| Fe I | 6173 | 2.5 | SDO/HMI |
| Fe I | 6302.5 | 2.5 | Hinode/SP |
| Fe I | 5250.2 | 3.0 | Ground-based |
| Fe I | 15648 | 3.0 | Infrared, large splitting |

### 1.4 The Hanle Effect

For weak magnetic fields (below $\sim 100$ G), the Zeeman effect becomes too small to detect. In this regime, the **Hanle effect** provides an alternative diagnostic. The Hanle effect is the modification of the polarization of resonantly scattered radiation by a magnetic field.

In the absence of a magnetic field, resonance scattering of the limb-darkened solar radiation produces linear polarization. A magnetic field modifies this polarization by rotating the scattering polarization plane and depolarizing the signal. The sensitivity range is:

$$B_{\text{Hanle}} \sim \frac{h}{4\pi \mu_B g_J t_{\text{life}}}$$

where $t_{\text{life}}$ is the lifetime of the upper level of the transition. For typical chromospheric lines, the Hanle effect is sensitive to fields in the range of 1-100 G, complementing the Zeeman effect which works best above $\sim 100$ G.

The Hanle effect is particularly valuable for measuring magnetic fields in prominences and the quiet Sun chromosphere, where the field is too weak for reliable Zeeman measurements.

---

## 2. Stokes Polarimetry

### 2.1 The Stokes Parameters

The polarization state of light is completely described by the four **Stokes parameters** $(I, Q, U, V)$:

- **$I$**: total intensity
- **$Q$**: linear polarization along a reference direction (positive for polarization along the reference axis, negative for perpendicular)
- **$U$**: linear polarization at $\pm 45^\circ$ to the reference direction
- **$V$**: circular polarization (positive for right-hand circular, negative for left-hand circular)

These are directly measurable quantities defined in terms of time-averaged products of the electric field components:

$$I = \langle E_x^2 \rangle + \langle E_y^2 \rangle$$
$$Q = \langle E_x^2 \rangle - \langle E_y^2 \rangle$$
$$U = 2 \langle E_x E_y \cos\delta \rangle$$
$$V = 2 \langle E_x E_y \sin\delta \rangle$$

where $\delta$ is the phase difference between the two orthogonal electric field components.

### 2.2 Magnetic Field Information in Stokes Profiles

The Stokes profiles of a magnetically split spectral line encode the full vector magnetic field:

- **Stokes $V$** (circular polarization): proportional to the line-of-sight (longitudinal) component of $B$. In the weak-field regime, $V(\lambda) \propto g_{\text{eff}} B_{\text{los}} \, \partial I / \partial \lambda$. This is the easiest signal to measure and is the basis for **longitudinal magnetograms**.

- **Stokes $Q$ and $U$** (linear polarization): proportional to the transverse component of $B$ squared ($B_\perp^2$). These signals are typically 10-100 times weaker than $V$, making transverse field measurements much more challenging. The azimuth of the transverse field is given by $\chi = \frac{1}{2} \arctan(U/Q)$, but this suffers from a **180-degree ambiguity** (cannot distinguish between $\chi$ and $\chi + 180^\circ$).

### 2.3 Polarized Radiative Transfer

The propagation of polarized light through a magnetized atmosphere is described by the **polarized radiative transfer equation**:

$$\frac{d\mathbf{I}}{d\tau_c} = \mathbf{K} (\mathbf{I} - \mathbf{S})$$

where $\mathbf{I} = (I, Q, U, V)^T$ is the Stokes vector, $\mathbf{K}$ is the $4 \times 4$ absorption matrix (which includes both absorption and anomalous dispersion terms that depend on the magnetic field), $\mathbf{S}$ is the source function vector, and $\tau_c$ is the continuum optical depth.

This is the **Unno-Rachkovsky equation**, first derived by Unno (1956) and extended by Rachkovsky (1962). The absorption matrix $\mathbf{K}$ contains the Zeeman-split absorption profiles and the magneto-optical (anomalous dispersion) terms that couple the different Stokes parameters.

### 2.4 Milne-Eddington Inversion

The simplest practical approach to extracting the magnetic field from observed Stokes profiles is the **Milne-Eddington (ME) inversion**. This assumes that the atmospheric parameters (magnetic field strength $B$, inclination $\gamma$, azimuth $\chi$, line-of-sight velocity $v_{\text{los}}$, Doppler width, line strength, etc.) are **constant with depth**. Under this assumption, the Unno-Rachkovsky equation has an **analytical solution**, giving explicit Stokes profiles as functions of the parameters.

The inversion proceeds by:

1. Computing model Stokes profiles for a set of parameters
2. Comparing to observed profiles via a $\chi^2$ merit function
3. Iteratively adjusting parameters (typically via Levenberg-Marquardt) to minimize $\chi^2$

The ME inversion is fast and robust, making it the standard for processing large data volumes (e.g., SDO/HMI processes $\sim 16$ million pixels every 12 minutes). However, it cannot capture atmospheric gradients (e.g., the variation of field strength with height).

### 2.5 Stratified Inversions

For more detailed analysis, **stratified inversions** solve the full polarized radiative transfer equation numerically, allowing atmospheric parameters to vary with optical depth. Major codes include:

- **SIR** (Stokes Inversion based on Response functions): parameterizes atmosphere on a grid of optical depths, uses response functions for efficient optimization
- **NICOLE**: NLTE inversion code that handles chromospheric lines
- **STiC**: recently developed code combining photospheric and chromospheric diagnostics

These inversions yield the 3D magnetic field vector as a function of height in the atmosphere, but are computationally expensive and require high-quality spectropolarimetric data.

### 2.6 Current Solar Magnetographs

Major instruments providing routine magnetic field measurements:

- **SDO/HMI**: full-disk, Fe I 6173 A, 6 wavelength points, Stokes $I$ and $V$ (with occasional full Stokes). Provides line-of-sight magnetograms every 45 s and vector magnetograms every 12 min.
- **Hinode/SP**: slit spectrograph, Fe I 6301.5/6302.5 A, full Stokes, 0.32" resolution. The gold standard for vector magnetic field measurements.
- **DKIST**: 4-meter ground-based telescope, multiple instruments, unprecedented polarimetric sensitivity.
- **Solar Orbiter/PHI**: full Stokes, Fe I 6173 A, providing out-of-ecliptic viewpoints.

---

## 3. Potential Field Source Surface (PFSS) Model

### 3.1 Motivation and Assumptions

While we can measure the magnetic field directly only at the photosphere (and to some extent in the chromosphere), we need to know the 3D magnetic field throughout the corona to understand coronal structure, solar wind source regions, and eruption physics. **Magnetic field extrapolation** models compute the 3D coronal field from photospheric boundary data.

The simplest and most widely used extrapolation is the **Potential Field Source Surface (PFSS)** model, which makes two assumptions:

1. **Current-free corona**: $\nabla \times \mathbf{B} = 0$ (the field is a potential field, with minimum energy for a given photospheric flux distribution)
2. **Source surface**: at a spherical surface of radius $R_{\text{ss}}$ (typically $2.5 \, R_\odot$), the field is forced to be purely radial ($B_\theta = B_\phi = 0$)

The first assumption means the corona carries no electric currents, which is a reasonable zeroth-order approximation for the large-scale field but fails in active regions where currents are important. The second assumption mimics the effect of the solar wind stretching field lines radially at large distances.

### 3.2 Mathematical Formulation

Since $\nabla \times \mathbf{B} = 0$, we can write $\mathbf{B} = -\nabla \Phi$ where $\Phi$ is a scalar potential. Combined with $\nabla \cdot \mathbf{B} = 0$, this gives **Laplace's equation**:

$$\nabla^2 \Phi = 0$$

In spherical coordinates $(r, \theta, \phi)$, the general solution is:

$$\Phi(r, \theta, \phi) = \sum_{l=1}^{l_{\max}} \sum_{m=0}^{l} \left[a_{lm} r^l + b_{lm} r^{-(l+1)}\right] P_l^m(\cos\theta) \left[c_{lm} \cos m\phi + d_{lm} \sin m\phi\right]$$

where $P_l^m$ are the associated Legendre polynomials and the coefficients are determined by the boundary conditions.

### 3.3 Boundary Conditions

**Inner boundary** ($r = R_\odot$): the radial component of the magnetic field matches the observed photospheric magnetogram:

$$B_r(R_\odot, \theta, \phi) = -\left.\frac{\partial \Phi}{\partial r}\right|_{r=R_\odot} = B_r^{\text{obs}}(\theta, \phi)$$

**Outer boundary** ($r = R_{\text{ss}}$): the field is radial, meaning the tangential components vanish:

$$B_\theta(R_{\text{ss}}, \theta, \phi) = 0, \quad B_\phi(R_{\text{ss}}, \theta, \phi) = 0$$

These two conditions fully determine the coefficients $a_{lm}$, $b_{lm}$, $c_{lm}$, $d_{lm}$ in terms of the spherical harmonic decomposition of the observed magnetogram.

### 3.4 Solution and Field Line Topology

With the potential determined, the magnetic field components are computed by differentiation:

$$B_r = -\frac{\partial \Phi}{\partial r}, \quad B_\theta = -\frac{1}{r}\frac{\partial \Phi}{\partial \theta}, \quad B_\phi = -\frac{1}{r \sin\theta}\frac{\partial \Phi}{\partial \phi}$$

Field lines are then traced by solving the ODE:

$$\frac{dr}{B_r} = \frac{r \, d\theta}{B_\theta} = \frac{r \sin\theta \, d\phi}{B_\phi}$$

The PFSS model classifies field lines into:

- **Closed field**: both footpoints on the solar surface (below $R_{\text{ss}}$)
- **Open field**: one footpoint on the solar surface, the other reaching $R_{\text{ss}}$ -- these correspond to coronal holes and solar wind source regions

### 3.5 Limitations

The PFSS model, while extremely useful for global coronal modeling, has significant limitations:

- **No currents**: cannot reproduce sheared or twisted fields in active regions, which are precisely the structures that store energy for flares and CMEs
- **Static**: no time evolution, no dynamics
- **Source surface height**: the choice of $R_{\text{ss}} = 2.5 \, R_\odot$ is empirical and may not be optimal for all phases of the solar cycle
- **Photospheric input**: uses line-of-sight magnetograms (often synoptic maps assembled over a full solar rotation), which cannot capture the Sun's far-side field

Despite these limitations, the PFSS model remains the standard tool for global coronal field modeling and solar wind connectivity studies.

---

## 4. Non-Linear Force-Free Field (NLFFF)

### 4.1 Why Force-Free?

In the solar corona, the magnetic pressure greatly exceeds the gas pressure. The plasma beta parameter:

$$\beta = \frac{p_{\text{gas}}}{p_{\text{mag}}} = \frac{2 \mu_0 n k_B T}{B^2}$$

takes typical values of $10^{-4}$ to $10^{-2}$ in the low corona. This means the magnetic force dominates all other forces, and the corona must be approximately **force-free**: the net Lorentz force on the plasma is negligible:

$$\mathbf{J} \times \mathbf{B} \approx 0$$

This condition implies that the current density $\mathbf{J}$ is parallel (or antiparallel) to the magnetic field:

$$\nabla \times \mathbf{B} = \alpha \mathbf{B}$$

where $\alpha(\mathbf{x})$ is a scalar function. Taking the divergence of both sides and using $\nabla \cdot \mathbf{B} = 0$:

$$\mathbf{B} \cdot \nabla \alpha = 0$$

This means $\alpha$ is constant along each field line, but can vary from one field line to another.

### 4.2 Special Cases

- **$\alpha = 0$ everywhere**: potential field (current-free), minimum energy state
- **$\alpha = \text{constant}$**: linear force-free field (LFFF), easier to solve but not realistic
- **$\alpha(\mathbf{x})$ varies in space**: non-linear force-free field (NLFFF), most realistic but hardest to compute

### 4.3 NLFFF Computational Methods

Several numerical methods have been developed to compute NLFFF extrapolations:

**Grad-Rubin method**: iteratively solves two sub-problems: (1) given $\alpha$ on field lines, compute $\mathbf{B}$ from $\nabla \times \mathbf{B} = \alpha \mathbf{B}$; (2) given $\mathbf{B}$, update $\alpha$ from boundary data. This method is well-posed but sensitive to boundary data quality.

**Optimization method** (Wheatland, Sturrock, Roumeliotis 2000): minimizes a functional that combines the force-free condition and divergence-free condition:

$$\mathcal{L} = \int_V \left[\frac{|(\nabla \times \mathbf{B}) \times \mathbf{B}|^2}{B^2} + |\nabla \cdot \mathbf{B}|^2\right] dV$$

The minimum $\mathcal{L} = 0$ corresponds to a perfect NLFFF solution.

**Magneto-frictional method**: evolves the magnetic field via an artificial velocity proportional to the Lorentz force: $\mathbf{v} = \nu^{-1} \mathbf{J} \times \mathbf{B}$. The field relaxes toward a force-free state as the residual force drives the evolution.

### 4.4 The Boundary Data Challenge

A fundamental challenge for NLFFF extrapolation is that the boundary data (photospheric vector magnetograms) come from a layer that is **not force-free**. The photosphere has $\beta \sim 1$, meaning gas pressure forces are comparable to magnetic forces. This creates inconsistencies:

- The measured photospheric field does not satisfy $\mathbf{J} \times \mathbf{B} = 0$
- Noise in the transverse field measurements introduces spurious currents
- The 180-degree azimuth ambiguity must be resolved before extrapolation

**Preprocessing** techniques have been developed to modify the photospheric boundary data to be more consistent with a force-free state while remaining close to the observations. This is done by minimizing a functional that penalizes both departure from force-freeness and departure from the data.

---

## 5. Magnetic Helicity

### 5.1 Definition

**Magnetic helicity** is a topological quantity that measures the degree of linkage, twist, and writhe in a magnetic field configuration. It is defined as:

$$H = \int_V \mathbf{A} \cdot \mathbf{B} \, dV$$

where $\mathbf{A}$ is the magnetic vector potential ($\mathbf{B} = \nabla \times \mathbf{A}$). Helicity has units of Mx$^2$ (Maxwell squared, or Wb$^2$ in SI).

The physical meaning of helicity can be understood through simple examples:

- **Two linked flux tubes**: each carrying flux $\Phi$, the mutual helicity is $H = \pm 2\Phi^2$ (sign depends on the sense of linkage)
- **A twisted flux tube**: with $N$ turns of twist over its length, the self-helicity is $H = N \Phi^2$

### 5.2 Gauge Invariance and Relative Helicity

A subtlety is that $\mathbf{A}$ is not unique -- it is defined only up to a gauge transformation $\mathbf{A} \to \mathbf{A} + \nabla \chi$. For a closed volume (where $\mathbf{B} \cdot \hat{n} = 0$ on the boundary), the helicity is gauge-invariant. But the solar corona is not a closed volume -- magnetic flux threads the photospheric boundary.

The solution is to use **relative helicity** (Berger and Field 1984):

$$H_R = \int_V (\mathbf{A} + \mathbf{A}_p) \cdot (\mathbf{B} - \mathbf{B}_p) \, dV$$

where $\mathbf{B}_p$ is a reference field (usually the potential field with the same boundary flux) and $\mathbf{A}_p$ is its vector potential. This quantity is gauge-invariant even for open volumes and measures the helicity relative to the potential field state.

### 5.3 Helicity Injection

Helicity is injected into the corona through the photospheric boundary by two processes:

$$\frac{dH}{dt} = 2 \int_S (\mathbf{A}_p \cdot \mathbf{v}_t) B_n \, dS - 2 \int_S (\mathbf{A}_p \cdot \mathbf{B}_t) v_n \, dS$$

- **First term (shearing)**: horizontal motions shuffling the footpoints of existing coronal field lines, twisting and braiding them
- **Second term (emergence)**: vertical transport of already-twisted flux tubes from below the photosphere into the corona

Observationally, the helicity injection rate can be computed from time sequences of photospheric magnetograms and velocity fields (from local correlation tracking or DAVE -- Differential Affine Velocity Estimator).

### 5.4 Conservation and the Hemispheric Rule

A remarkable property of magnetic helicity is that it is approximately conserved even during magnetic reconnection (Taylor 1974). While reconnection can change the topology of field lines (converting mutual helicity to self-helicity and vice versa), the total helicity is nearly preserved because helicity dissipation occurs on the resistive timescale, which is much longer than the reconnection timescale in high-Lundquist-number plasmas.

This conservation has profound implications:

- Helicity accumulates in the corona over time
- It cannot be efficiently removed by reconnection
- The primary removal mechanism is **bodily ejection** via CMEs

The **hemispheric helicity rule** is an observed pattern: active regions in the northern hemisphere tend to have negative helicity (left-handed twist), while those in the southern hemisphere tend to have positive helicity (right-handed twist). This pattern is thought to reflect the effect of the Coriolis force on rising flux tubes in the convection zone.

---

## 6. Magnetic Free Energy

### 6.1 Definition and Significance

The **magnetic free energy** is defined as the excess magnetic energy above the minimum-energy (potential) field configuration:

$$E_{\text{free}} = E_{\text{total}} - E_{\text{potential}} = \frac{1}{2\mu_0}\int_V B^2 \, dV - \frac{1}{2\mu_0}\int_V B_p^2 \, dV$$

where $\mathbf{B}_p$ is the potential field matching the same boundary conditions. The potential field has the minimum energy for a given boundary flux distribution (Thomson's theorem), so $E_{\text{free}} \geq 0$.

Only the free energy is available to power solar eruptions (flares and CMEs). The potential field energy is "locked" by the boundary conditions and cannot be released without changing the photospheric flux distribution.

### 6.2 Typical Values

For major solar flares and CMEs, the free energy budget is:

| Eruption class | $E_{\text{free}}$ (erg) | Fraction released |
|---------------|----------------------|-------------------|
| C-class flare | $10^{29}$-$10^{30}$ | 10-30% |
| M-class flare | $10^{30}$-$10^{31}$ | 10-30% |
| X-class flare | $10^{31}$-$10^{32}$ | 10-50% |

The fraction of free energy released in a single event is typically 10-50%, meaning substantial free energy remains even after a major eruption.

### 6.3 Proxy Measures

Since computing the full 3D coronal field is difficult, several photospheric proxy measures for free energy have been developed:

- **Magnetic shear angle**: the angle between the observed transverse field and the potential field transverse component. Large shear (especially along the PIL) indicates stored free energy.
- **Photospheric excess energy density**: $\rho_{\text{free}} = (B_{\text{obs}}^2 - B_p^2) / (8\pi)$ evaluated at the photosphere
- **Schrijver's $R$ parameter**: the unsigned flux near strong-gradient polarity inversion lines, empirically correlated with flare productivity
- **Total unsigned current helicity**: $\int |J_z \cdot B_z| \, dA$, related to the degree of twist

---

## 7. Magnetic Topology

### 7.1 Why Topology Matters

The topology of the coronal magnetic field determines where **reconnection** can occur, which in turn determines where flares happen, how field lines reconfigure during eruptions, and how energy is released. Reconnection occurs preferentially at locations where the magnetic connectivity changes discontinuously or very rapidly.

### 7.2 Null Points

A **magnetic null point** is a location where $\mathbf{B} = 0$. Near a 3D null point, the field has a characteristic structure:

- **Spine lines**: two field lines that approach (or recede from) the null along a single direction
- **Fan surface**: a surface of field lines that radiates outward (or inward) from the null in a plane

Null points are important sites for reconnection because the vanishing field means the frozen-in condition of ideal MHD breaks down easily. Reconnection at a null point transfers magnetic flux between the domains separated by the fan surface.

The number and location of coronal null points can be found from the PFSS or NLFFF models. Typical active regions may contain several null points, often located above polarity inversion lines or between interacting flux systems.

### 7.3 Separatrices and Separators

A **separatrix** is a surface in the magnetic field that separates regions of distinct magnetic connectivity. Field lines on opposite sides of a separatrix connect to different photospheric regions.

Where two separatrix surfaces intersect, they form a **separator** -- a field line that connects two null points. Separators are lines in 3D where four distinct connectivity domains meet, and they are natural sites for current sheet formation and reconnection.

### 7.4 Quasi-Separatrix Layers (QSLs)

In many realistic magnetic configurations, true null points and separatrices may be absent, yet there are still regions where the magnetic connectivity changes very rapidly. These are called **Quasi-Separatrix Layers (QSLs)** (Priest and Demoulin 1995).

QSLs are quantified by the **squashing factor** $Q$, which measures how much a small flux tube deforms as it is mapped from one footpoint to the other. If we map footpoint coordinates $(x_1, y_1)$ on one polarity to $(x_2, y_2)$ on the conjugate polarity via field line tracing, the squashing factor is:

$$Q = \frac{a^2 + b^2 + c^2 + d^2}{|ad - bc|}$$

where $a = \partial x_2/\partial x_1$, $b = \partial x_2/\partial y_1$, $c = \partial y_2/\partial x_1$, $d = \partial y_2/\partial y_1$ are the elements of the Jacobian of the field line mapping.

For smooth field line mapping, $Q = 2$ (minimum value). True separatrices have $Q \to \infty$. QSLs are defined as regions where $Q \gg 2$, typically $Q > 10^2$ or higher.

### 7.5 QSLs and Reconnection

QSLs are the preferred sites for current sheet formation in the corona. The physical reason is clear: when photospheric motions displace the footpoints, field lines on opposite sides of a QSL are displaced by very different amounts (because the mapping is strongly deforming). This differential displacement generates strong gradients in the magnetic field across the QSL, which by Ampere's law correspond to intense current sheets.

These current sheets are where magnetic reconnection occurs, releasing stored magnetic energy and accelerating particles. The observed locations of flare ribbons -- the chromospheric brightenings produced by energetic particles precipitating along post-reconnection field lines -- map remarkably well onto the photospheric footprints of QSLs, providing strong evidence for the QSL-reconnection paradigm.

### 7.6 Hyperbolic Flux Tubes

A special class of QSL structure is the **Hyperbolic Flux Tube (HFT)**, which occurs at the intersection of two QSLs. An HFT has a cross-section that is hyperbolic in shape: field lines are compressed in one direction and expanded in the other. This geometry is particularly efficient at generating thin current sheets and is believed to be the 3D generalization of the 2D X-point where reconnection occurs in classical models.

---

## Practice Problems

**Problem 1: Zeeman Splitting Calculation**

(a) Calculate the Zeeman splitting $\Delta\lambda_B$ for the Fe I 6302.5 A line ($g_{\text{eff}} = 2.5$) in a sunspot with $B = 2500$ G. (b) The thermal Doppler width of this line at $T = 4500$ K is approximately $\Delta\lambda_D = \lambda_0 \sqrt{2k_BT / (m_{\text{Fe}} c^2)}$. Calculate $\Delta\lambda_D$ and compare it to $\Delta\lambda_B$. (c) Is the Zeeman splitting resolved in this sunspot? What about for a quiet Sun field of $B = 50$ G?

**Problem 2: PFSS Model**

Consider the lowest-order ($l = 1$, $m = 0$) PFSS solution, which represents a magnetic dipole. (a) Write the general solution for $\Phi(r, \theta)$ with the two radial functions. (b) Apply the source surface boundary condition at $r = R_{\text{ss}}$ to express $b_{10}$ in terms of $a_{10}$. (c) Show that for $R_{\text{ss}} \to \infty$, you recover the standard dipole field, and for $R_{\text{ss}} \to R_\odot$, the field becomes purely radial at the surface.

**Problem 3: Force-Free Parameter**

An active region has a magnetic field with $|\mathbf{B}| = 100$ G and a current density $|\mathbf{J}| = 10$ mA/m$^2$. (a) Calculate the force-free parameter $\alpha = \mu_0 J / B$ (in units of Mm$^{-1}$). (b) The characteristic length scale of the field is $L = 1/\alpha$. Is this scale larger or smaller than a typical active region ($\sim 100$ Mm)? (c) Explain what it means physically for $\alpha$ to be large vs. small.

**Problem 4: Magnetic Helicity and CMEs**

An active region injects helicity at a rate of $dH/dt = 5 \times 10^{40}$ Mx$^2$/hour. (a) How much helicity accumulates over 3 days? (b) If a CME removes helicity of order $H_{\text{CME}} \sim 10^{42}$ Mx$^2$, how many days of accumulation does this represent? (c) A typical active region exists for 2-4 weeks. Estimate the total helicity accumulated and the number of CMEs needed to remove it. Discuss the implications for the relationship between helicity buildup and CME occurrence.

**Problem 5: Squashing Factor and Reconnection**

Consider a simplified 2D magnetic field line mapping where footpoint $(x_1, y_1)$ maps to $(x_2, y_2) = (x_1 + L \tanh(x_1/w), y_1)$, where $L$ is a displacement amplitude and $w$ is a characteristic width. (a) Compute the Jacobian elements $a, b, c, d$ of this mapping. (b) Derive the squashing factor $Q$ as a function of $x_1$. (c) Where is $Q$ maximized, and what is $Q_{\max}$ in terms of $L/w$? (d) For $L = 50$ Mm and $w = 1$ Mm, calculate $Q_{\max}$ and comment on whether this would constitute a QSL.

---

**Previous**: [Corona](./06_Corona.md) | **Next**: [Active Regions and Sunspots](./08_Active_Regions_and_Sunspots.md)

# 17. Spectroscopy

[← Previous: 16. Adaptive Optics](16_Adaptive_Optics.md) | [Overview →](00_Overview.md)

---

In 1814, Joseph von Fraunhofer pointed a telescope through a prism at the Sun and noticed something remarkable: the rainbow of solar colors was crossed by hundreds of dark lines at specific wavelengths. These **Fraunhofer lines** — absorption signatures of elements in the solar atmosphere — opened a new way of studying matter through its interaction with light. By the 1860s, Kirchhoff and Bunsen had established that every element produces a unique pattern of spectral lines, and spectroscopy became the most powerful tool in both chemistry and astrophysics. Today it remains so: from determining the composition of distant galaxies to measuring blood oxygen levels, from quality control in semiconductor fabrication to detecting explosives at airports.

**Spectroscopy** is the study of how matter absorbs, emits, and scatters electromagnetic radiation as a function of wavelength (or equivalently, frequency or energy). This lesson covers the physical principles behind spectral lines, the instruments that resolve them, and the analytical techniques that extract information from spectra. We connect to earlier lessons on diffraction (L06), interference (L05), and optical instruments (L04), showing how these fundamental phenomena underpin every spectrometer design.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Explain the origin of atomic and molecular spectra from quantized energy transitions and state Kirchhoff's three laws of spectroscopy
2. Derive Einstein's A and B coefficients for radiative transitions and relate them to absorption and emission rates
3. Describe the three main spectral line broadening mechanisms (natural, Doppler, pressure) and compute the Voigt profile
4. Analyze the resolving power and free spectral range of prism and diffraction grating spectrometers
5. Explain the operating principle and performance of the Fabry-Pérot interferometer and Fourier transform spectrometer
6. Apply the Beer-Lambert law to quantitative absorption spectroscopy and calculate concentrations from absorbance data
7. Describe fluorescence, Raman, and laser spectroscopy techniques and their applications
8. Implement spectral line fitting, spectrometer simulation, and Fabry-Pérot analysis in Python

---

## Table of Contents

1. [Fundamentals of Spectra](#1-fundamentals-of-spectra)
2. [Radiative Transitions](#2-radiative-transitions)
3. [Spectral Line Broadening](#3-spectral-line-broadening)
4. [Dispersive Spectrometers](#4-dispersive-spectrometers)
5. [Interferometric Spectrometers](#5-interferometric-spectrometers)
6. [Absorption Spectroscopy](#6-absorption-spectroscopy)
7. [Emission and Fluorescence Spectroscopy](#7-emission-and-fluorescence-spectroscopy)
8. [Raman Spectroscopy](#8-raman-spectroscopy)
9. [Modern Techniques](#9-modern-techniques)
10. [Python Examples](#10-python-examples)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Fundamentals of Spectra

### 1.1 Atomic Energy Levels

Atoms have discrete energy levels determined by quantum mechanics. The energy of a photon emitted or absorbed in a transition between levels $E_1$ and $E_2$ is:

$$h\nu = E_2 - E_1 \qquad \Leftrightarrow \qquad \lambda = \frac{hc}{E_2 - E_1}$$

For hydrogen, the energy levels are:

$$E_n = -\frac{13.6\,\text{eV}}{n^2}$$

giving the familiar Balmer series (visible), Lyman series (UV), and Paschen series (IR).

### 1.2 Molecular Spectra

Molecules have additional degrees of freedom — rotation and vibration — producing much richer spectra:

- **Rotational spectra** (microwave, far-IR): $E_J = BJ(J+1)$ where $B = \hbar^2/(2I)$ is the rotational constant
- **Vibrational spectra** (IR): $E_v = \hbar\omega_0(v + 1/2)$ for harmonic oscillator approximation
- **Electronic spectra** (visible, UV): Transitions between electronic states, typically accompanied by vibrational structure (vibronic bands)

The total energy is approximately:

$$E \approx E_{\text{electronic}} + E_{\text{vibrational}} + E_{\text{rotational}}$$

with $E_{\text{elec}} \gg E_{\text{vib}} \gg E_{\text{rot}}$.

### 1.3 Kirchhoff's Laws of Spectroscopy

Gustav Kirchhoff (1859) formulated three empirical laws:

1. **A hot, dense body** (solid, liquid, or dense gas) emits a continuous spectrum (blackbody radiation)
2. **A hot, low-density gas** emits light at specific wavelengths — an **emission line spectrum**
3. **A cool gas in front of a continuous source** absorbs light at the same specific wavelengths — an **absorption line spectrum**

> **Analogy**: Think of spectral lines as a barcode for each element. Just as a supermarket scanner identifies products by their unique barcode pattern, spectroscopists identify atoms and molecules by their unique spectral line pattern. Every element has its own "barcode" — a set of wavelengths where it absorbs or emits light — and this fingerprint is as unique as a human fingerprint.

### 1.4 The Electromagnetic Spectrum and Spectroscopic Regions

| Region | Wavelength | Transitions | Technique |
|--------|:----------:|-------------|-----------|
| Gamma/X-ray | < 10 nm | Inner-shell electrons | X-ray spectroscopy |
| UV | 10–400 nm | Valence electrons | UV-Vis spectrophotometry |
| Visible | 400–700 nm | Valence electrons | Optical spectroscopy |
| Near-IR | 0.7–2.5 $\mu$m | Overtone vibrations | NIR spectroscopy |
| Mid-IR | 2.5–25 $\mu$m | Fundamental vibrations | FTIR, IR absorption |
| Far-IR/THz | 25–1000 $\mu$m | Rotations | THz spectroscopy |
| Microwave | 1 mm–1 m | Molecular rotations | Microwave spectroscopy |
| Radio | > 1 m | Hyperfine (21 cm H line) | Radio astronomy |

---

## 2. Radiative Transitions

### 2.1 Einstein Coefficients

In 1917, Einstein showed that the interaction between radiation and matter involves three processes:

**Spontaneous emission** (rate $A_{21}$): An atom in excited state 2 spontaneously decays to state 1, emitting a photon. The spontaneous emission rate is:

$$\left(\frac{dN_2}{dt}\right)_{\text{spont}} = -A_{21} N_2$$

The Einstein A coefficient has units of s$^{-1}$ and is the inverse of the radiative lifetime: $\tau_{\text{rad}} = 1/A_{21}$.

**Stimulated emission** (rate $B_{21}$): A photon of the right frequency stimulates an excited atom to emit an identical photon. The rate is proportional to the radiation energy density $u(\nu)$:

$$\left(\frac{dN_2}{dt}\right)_{\text{stim}} = -B_{21} u(\nu) N_2$$

**Absorption** (rate $B_{12}$): A photon is absorbed, exciting an atom from state 1 to state 2:

$$\left(\frac{dN_1}{dt}\right)_{\text{abs}} = -B_{12} u(\nu) N_1$$

### 2.2 Relations Between Coefficients

In thermal equilibrium, the population ratio is given by the Boltzmann distribution:

$$\frac{N_2}{N_1} = \frac{g_2}{g_1}\exp\left(-\frac{h\nu}{k_BT}\right)$$

Requiring that the Einstein relations reproduce the Planck blackbody spectrum, one obtains:

$$g_1 B_{12} = g_2 B_{21}$$

$$A_{21} = \frac{8\pi h\nu^3}{c^3} B_{21}$$

where $g_1$, $g_2$ are the statistical weights (degeneracies) of the two levels.

### 2.3 Oscillator Strength

The **oscillator strength** $f_{12}$ is a dimensionless measure of the transition probability:

$$f_{12} = \frac{m_e c}{8\pi^2 e^2 \nu^2} \frac{g_2}{g_1} A_{21}$$

Strong transitions have $f \sim 1$ (e.g., the sodium D lines: $f = 0.65$). Forbidden transitions have $f \ll 1$ (e.g., nebular [O III] lines: $f \sim 10^{-8}$).

### 2.4 Selection Rules

Not all transitions between energy levels are allowed. The electric dipole selection rules for atoms are:

$$\Delta l = \pm 1, \quad \Delta m_l = 0, \pm 1, \quad \Delta S = 0$$

Transitions violating these rules are "forbidden" — they still occur via magnetic dipole or electric quadrupole interactions, but at much lower rates (longer lifetimes).

---

## 3. Spectral Line Broadening

### 3.1 Natural Broadening (Lorentzian)

The finite radiative lifetime $\tau$ of an excited state leads to an energy uncertainty $\Delta E \sim \hbar/\tau$ via the Heisenberg uncertainty principle. This produces a **Lorentzian** line profile:

$$\phi_L(\nu) = \frac{1}{\pi} \frac{\gamma/2}{(\nu - \nu_0)^2 + (\gamma/2)^2}$$

where $\gamma = 1/(2\pi\tau)$ is the half-width at half-maximum (HWHM) in Hz. Natural broadening is typically very small ($\Delta\nu \sim 10^7$ Hz, i.e., $\Delta\lambda \sim 10^{-5}$ nm at visible wavelengths) and is usually negligible compared to other mechanisms.

### 3.2 Doppler Broadening (Gaussian)

Thermal motion of atoms causes a distribution of Doppler shifts. For a Maxwell-Boltzmann velocity distribution at temperature $T$, the line profile is **Gaussian**:

$$\phi_G(\nu) = \frac{1}{\sigma_D\sqrt{2\pi}} \exp\left[-\frac{(\nu - \nu_0)^2}{2\sigma_D^2}\right]$$

where the Doppler width (standard deviation) is:

$$\sigma_D = \frac{\nu_0}{c}\sqrt{\frac{k_BT}{m}}$$

and the FWHM is:

$$\Delta\nu_D = 2\sqrt{2\ln 2}\,\sigma_D = \frac{\nu_0}{c}\sqrt{\frac{8k_BT\ln 2}{m}}$$

For the hydrogen Balmer-$\alpha$ line at $T = 5000$ K: $\Delta\lambda_D \approx 0.04$ nm.

### 3.3 Pressure Broadening (Lorentzian)

Collisions with neighboring atoms or electrons interrupt the emission process, shortening the effective coherence time. This produces another Lorentzian profile with width:

$$\gamma_{\text{pressure}} \propto N_{\text{perturber}} \, v_{\text{rel}} \, \sigma_{\text{col}}$$

where $N$ is the perturber density, $v_{\text{rel}}$ is the relative velocity, and $\sigma_{\text{col}}$ is the collision cross-section. Pressure broadening dominates in dense plasmas and stellar atmospheres.

### 3.4 The Voigt Profile

When both Doppler and Lorentzian broadening are significant, the resulting profile is a **convolution** of the two:

$$\phi_V(\nu) = \phi_G * \phi_L = \int_{-\infty}^{\infty} \phi_G(\nu') \phi_L(\nu - \nu') \, d\nu'$$

This is the **Voigt profile**, which can be expressed using the **Faddeeva function** $w(z)$:

$$\phi_V(\nu) = \frac{1}{\sigma_D\sqrt{2\pi}} \text{Re}[w(z)], \qquad z = \frac{(\nu - \nu_0) + i\gamma/2}{\sigma_D\sqrt{2}}$$

where $w(z) = e^{-z^2}\text{erfc}(-iz)$ is the complex error function.

The Voigt profile has a Gaussian core (dominated by Doppler near $\nu_0$) and Lorentzian wings (dominated by pressure broadening far from $\nu_0$).

### 3.5 Equivalent Width

The **equivalent width** $W_\lambda$ measures the total absorption of a spectral line, defined as the width of a rectangle with height equal to the continuum that has the same area as the line:

$$W_\lambda = \int \frac{I_c - I(\lambda)}{I_c} d\lambda$$

where $I_c$ is the continuum intensity. The equivalent width is independent of instrumental resolution and directly relates to the column density of absorbers (via the **curve of growth**).

---

## 4. Dispersive Spectrometers

### 4.1 Prism Spectrometers

A prism separates wavelengths through material dispersion: the refractive index varies with wavelength ($dn/d\lambda < 0$ in the visible, known as normal dispersion).

**Angular dispersion** of a prism with apex angle $A$:

$$\frac{d\theta}{d\lambda} = \frac{t}{d} \frac{dn}{d\lambda}$$

where $t$ is the base length and $d$ is the beam diameter at the exit face. Equivalently, for minimum deviation:

$$\frac{d\theta}{d\lambda} = \frac{2\sin(A/2)}{\cos[(\delta_{\min}+A)/2]} \frac{dn}{d\lambda}$$

**Resolving power**:

$$R = \frac{\lambda}{\Delta\lambda} = t \frac{dn}{d\lambda}$$

For a 60 mm BK7 prism at 500 nm: $dn/d\lambda \approx -0.04\,\mu\text{m}^{-1}$, giving $R \approx 2400$.

### 4.2 Diffraction Grating Spectrometers

Diffraction gratings are the workhorse of modern spectroscopy, offering higher resolving power and broader wavelength coverage than prisms.

The **grating equation** (reviewed from L06):

$$d(\sin\theta_i + \sin\theta_m) = m\lambda$$

where $d$ is the groove spacing, $\theta_i$ is the incidence angle, $\theta_m$ is the diffraction angle for order $m$.

**Angular dispersion**:

$$\frac{d\theta_m}{d\lambda} = \frac{m}{d\cos\theta_m}$$

**Resolving power**:

$$R = mN$$

where $N$ is the total number of illuminated grooves. A grating with 1200 grooves/mm, 80 mm wide ($N = 96{,}000$) in first order gives $R = 96{,}000$ — sufficient to resolve the hyperfine structure of many spectral lines.

### 4.3 Blazed Gratings

A plain grating distributes light across many orders, wasting most of the energy. A **blazed grating** has a sawtooth groove profile that concentrates light into a specific order. The **blaze angle** $\theta_B$ is chosen so that the specular reflection from each groove facet coincides with the desired diffraction order:

$$\lambda_{\text{blaze}} = \frac{2d\sin\theta_B}{m}$$

Blazed gratings achieve >70% efficiency at the blaze wavelength.

### 4.4 Free Spectral Range

The **free spectral range** (FSR) is the wavelength range over which adjacent diffraction orders do not overlap:

$$\text{FSR} = \frac{\lambda}{m}$$

Higher orders give better resolving power but smaller FSR. Order-sorting filters or cross-dispersers are used to isolate the desired order.

### 4.5 Spectrometer Configurations

| Configuration | Description | Application |
|---------------|-------------|-------------|
| Czerny-Turner | Two concave mirrors + plane grating | General-purpose lab spectrometer |
| Littrow | Grating acts as both disperser and retroreflector | Compact high-resolution |
| Echelle | High blaze angle, high order ($m \sim 50$–$100$), cross-dispersed | High-resolution stellar spectroscopy |
| Rowland circle | Concave grating (disperser + focuser) | X-ray, EUV spectroscopy |

> **Analogy**: A diffraction grating works like a venetian blind for light. Sunlight passing through the blind's slats creates rainbow patterns on the opposite wall — that's diffraction splitting white light into colors. The grating's grooves play the role of the slats, but engineered with nanometer precision to separate wavelengths with extraordinary accuracy.

---

## 5. Interferometric Spectrometers

### 5.1 Fabry-Pérot Interferometer

The **Fabry-Pérot interferometer** (FPI) consists of two parallel, partially reflecting surfaces separated by a gap $d$. Light bounces back and forth between the surfaces, and constructive interference occurs when:

$$2nd\cos\theta = m\lambda \qquad (m = 1, 2, 3, \ldots)$$

where $n$ is the refractive index of the gap medium and $\theta$ is the angle of incidence.

The transmitted intensity follows the **Airy function**:

$$T = \frac{1}{1 + F\sin^2(\delta/2)}$$

where:

$$\delta = \frac{4\pi nd\cos\theta}{\lambda}, \qquad F = \frac{4R}{(1-R)^2}$$

and $R$ is the mirror reflectance. The parameter $F$ is called the **coefficient of finesse** (not to be confused with the finesse itself).

### 5.2 Finesse and Resolution

The **finesse** $\mathcal{F}$ is the ratio of FSR to linewidth:

$$\mathcal{F} = \frac{\pi\sqrt{R}}{1-R} = \frac{\pi\sqrt{F}}{2}$$

| Reflectance $R$ | Finesse $\mathcal{F}$ |
|:----------------:|:--------------------:|
| 0.5 | 4.4 |
| 0.9 | 30 |
| 0.95 | 61 |
| 0.99 | 313 |

The FSR and resolving power are:

$$\text{FSR} = \frac{\lambda^2}{2nd}, \qquad R_{\text{FP}} = \frac{2nd}{\lambda} \mathcal{F} = m\mathcal{F}$$

An FPI with $d = 5$ mm, $R = 0.95$, at $\lambda = 500$ nm gives $m = 20{,}000$, $\mathcal{F} = 61$, and $R_{\text{FP}} = 1.2 \times 10^6$ — comparable to the best echelle spectrographs.

### 5.3 Fourier Transform Spectroscopy (FTS)

A **Fourier transform spectrometer** is based on a Michelson interferometer. As one mirror scans through a range of path differences $\delta$, the detector records the **interferogram** — the intensity as a function of $\delta$:

$$I(\delta) = \int_0^{\infty} B(\nu)[1 + \cos(2\pi\nu\delta)] \, d\nu$$

where $B(\nu)$ is the spectral intensity. The spectrum is recovered by Fourier transforming the interferogram:

$$B(\nu) = 2\int_0^{\infty} [I(\delta) - \langle I \rangle] \cos(2\pi\nu\delta) \, d\delta$$

**Advantages of FTS**:
- **Fellgett (multiplex) advantage**: All wavelengths measured simultaneously (SNR $\propto \sqrt{N}$ improvement over scanning)
- **Jacquinot (throughput) advantage**: Circular apertures can be large (no slits needed)
- **Connes advantage**: Wavelength calibration via a reference laser (He-Ne) is extremely precise

**Resolution**: The spectral resolution is limited by the maximum path difference $\delta_{\max}$:

$$\Delta\nu = \frac{1}{2\delta_{\max}}$$

An FTS scanning 10 cm path difference achieves $\Delta\nu = 0.05\,\text{cm}^{-1}$ ($\Delta\lambda \approx 10^{-3}$ nm at 500 nm), i.e., $R \approx 500{,}000$.

FTS is the standard technique in the mid-infrared (FTIR spectroscopy), where thermal sources are weak and the multiplex advantage is critical.

---

## 6. Absorption Spectroscopy

### 6.1 The Beer-Lambert Law

When light passes through an absorbing medium, the transmitted intensity decreases exponentially:

$$I(\lambda) = I_0(\lambda) \exp[-\alpha(\lambda) l] = I_0(\lambda) \exp[-\varepsilon(\lambda) c l]$$

where:
- $\alpha(\lambda)$ is the absorption coefficient (m$^{-1}$)
- $\varepsilon(\lambda)$ is the molar absorption coefficient (L mol$^{-1}$ cm$^{-1}$)
- $c$ is the concentration (mol/L)
- $l$ is the path length (cm)

### 6.2 Absorbance and Transmittance

The **transmittance** and **absorbance** are:

$$T = \frac{I}{I_0}, \qquad A = -\log_{10}(T) = \varepsilon c l$$

The absorbance is linear in concentration — this is the basis of quantitative analysis. A UV-Vis spectrophotometer measures $A(\lambda)$ and uses calibration curves to determine concentrations.

### 6.3 Deviations from Beer-Lambert

The law assumes:
- Monochromatic light (polychromatic light causes nonlinearity at high absorbance)
- Dilute solutions (intermolecular interactions at high concentration shift spectra)
- No scattering or fluorescence
- Uniform path through the sample

At high concentrations ($A > 2$, i.e., $T < 1\%$), stray light in the spectrometer causes apparent deviations.

### 6.4 Differential Optical Absorption Spectroscopy (DOAS)

DOAS removes broadband absorption features (Rayleigh scattering, aerosols) by fitting only the narrow spectral structure of each absorber. This technique is widely used for atmospheric trace gas monitoring (NO$_2$, SO$_2$, O$_3$, HCHO).

---

## 7. Emission and Fluorescence Spectroscopy

### 7.1 Emission Spectroscopy

In emission spectroscopy, atoms or molecules are excited (thermally, electrically, or optically) and the emitted light is analyzed. Examples:

- **Flame emission**: Sample is aspirated into a flame; alkali metals produce characteristic colors (Na: yellow, K: violet, Li: red)
- **Arc/spark emission**: High-energy excitation for metals and alloys
- **ICP-OES**: Inductively coupled plasma optical emission spectroscopy — the workhorse of elemental analysis

### 7.2 Fluorescence Spectroscopy

**Fluorescence** is the emission of light following electronic excitation. The process is described by the **Jablonski diagram**:

```
        S₂ ────── fast internal conversion
        |
        ▼
S₁ ──── ────── vibrational relaxation (ps)
│    ╲
│     ╲ intersystem crossing
│      ╲
│       T₁ ──── phosphorescence (ms-s)
│
▼ fluorescence (ns)
S₀
```

Key properties:

- **Stokes shift**: Fluorescence is always at longer wavelength (lower energy) than excitation, because of vibrational relaxation in the excited state
- **Quantum yield**: $\Phi = k_r / (k_r + k_{nr})$ where $k_r$ is the radiative rate and $k_{nr}$ includes all non-radiative decay channels
- **Fluorescence lifetime**: $\tau_f = 1/(k_r + k_{nr})$, typically 1–100 ns for organic fluorophores

### 7.3 Applications

| Technique | Application |
|-----------|-------------|
| Laser-induced fluorescence (LIF) | Combustion diagnostics, flow visualization |
| Fluorescence microscopy | Cell biology, medical imaging |
| Time-resolved fluorescence | Protein dynamics, FRET distance measurements |
| Fluorescence correlation spectroscopy | Diffusion coefficients, binding kinetics |

---

## 8. Raman Spectroscopy

### 8.1 The Raman Effect

When monochromatic light scatters off molecules, most photons scatter elastically (**Rayleigh scattering**) at the same frequency. However, a small fraction ($\sim 10^{-6}$) scatter inelastically — the photon gains or loses energy by exciting or de-exciting a molecular vibration:

$$\nu_{\text{scattered}} = \nu_0 \pm \nu_{\text{vib}}$$

- **Stokes lines** ($\nu_0 - \nu_{\text{vib}}$): Photon loses energy; molecule gains a vibrational quantum
- **Anti-Stokes lines** ($\nu_0 + \nu_{\text{vib}}$): Photon gains energy; molecule loses a vibrational quantum

The intensity ratio of anti-Stokes to Stokes is governed by the Boltzmann factor:

$$\frac{I_{\text{aS}}}{I_{\text{S}}} = \left(\frac{\nu_0 + \nu_{\text{vib}}}{\nu_0 - \nu_{\text{vib}}}\right)^4 \exp\left(-\frac{h\nu_{\text{vib}}}{k_BT}\right)$$

### 8.2 Raman vs. IR Spectroscopy

Both probe molecular vibrations but through different mechanisms:

| Property | Raman | IR Absorption |
|----------|:-----:|:-------------:|
| Mechanism | Polarizability change | Dipole moment change |
| Selection rule | $\Delta\alpha \neq 0$ | $\Delta\mu \neq 0$ |
| Water interference | Minimal | Strong |
| Sample preparation | Minimal | Often requires thin films/KBr pellets |
| Homonuclear molecules | Active (O$_2$, N$_2$) | Inactive |
| Spatial resolution | $\sim 1\,\mu$m (confocal) | $\sim 10\,\mu$m |

The two techniques are **complementary**: modes that are Raman-active tend to be IR-inactive in centrosymmetric molecules (rule of mutual exclusion).

### 8.3 Enhanced Raman Techniques

- **SERS** (Surface-Enhanced Raman Spectroscopy): Metal nanostructures amplify the Raman signal by $10^6$–$10^{10}$ through plasmonic enhancement. Enables single-molecule detection.
- **CARS** (Coherent Anti-Stokes Raman Spectroscopy): Two laser beams generate a coherent anti-Stokes signal. Much stronger than spontaneous Raman; used in combustion diagnostics and microscopy.

---

## 9. Modern Techniques

### 9.1 Laser Spectroscopy

Lasers enable spectroscopic techniques impossible with conventional sources:

- **LIBS** (Laser-Induced Breakdown Spectroscopy): A high-power laser pulse ablates and ionizes the sample surface; the resulting plasma emission reveals elemental composition. Used for remote analysis (Mars rovers), art authentication, and industrial sorting.
- **Cavity Ring-Down Spectroscopy (CRDS)**: A laser pulse bounces between high-reflectivity mirrors ($R > 99.99\%$). The decay time depends on intracavity absorption. Detects absorptions as small as $10^{-10}$ cm$^{-1}$ — ideal for trace gas sensing.
- **Saturated absorption spectroscopy**: A strong "pump" beam saturates the transition, creating a narrow "Lamb dip" in the Doppler-broadened profile. Enables sub-Doppler resolution for frequency standards.

### 9.2 Time-Resolved Spectroscopy

Ultrafast lasers (femtosecond to attosecond) enable the study of dynamics:

- **Pump-probe spectroscopy**: A pump pulse excites the sample; a probe pulse measures the absorption at a variable delay. Time resolution limited by pulse duration ($\sim 10$ fs).
- **Transient absorption**: Measures changes in optical density $\Delta A(t, \lambda)$ after excitation. Maps excited state dynamics, charge transfer, and relaxation pathways.

### 9.3 Astronomical Spectroscopy

Spectroscopy is fundamental to astrophysics:

- **Redshift**: Doppler shift of spectral lines measures recession velocity: $z = \Delta\lambda/\lambda = v/c$ (non-relativistic)
- **Stellar classification**: Spectral types (O, B, A, F, G, K, M) determined by absorption lines
- **Radial velocity method**: Detection of exoplanets from periodic Doppler shifts (precision ~1 m/s with modern echelle spectrographs)
- **Chemical abundances**: Equivalent widths and spectral synthesis determine elemental abundances in stellar atmospheres

### 9.4 Imaging Spectroscopy

**Hyperspectral imaging** combines spectroscopy with spatial information, producing a data cube $(x, y, \lambda)$:

- **Remote sensing**: Mineral mapping, vegetation health, atmospheric monitoring
- **Medical imaging**: Tissue oxygenation, cancer detection
- **Art conservation**: Pigment identification without sampling

---

## 10. Python Examples

### 10.1 Spectral Line Profiles

```python
import numpy as np

def lorentzian(nu: np.ndarray, nu0: float, gamma: float) -> np.ndarray:
    """Lorentzian (natural/pressure broadened) line profile.

    L(nu) = (1/pi) * (gamma/2) / ((nu - nu0)^2 + (gamma/2)^2)

    Parameters
    ----------
    nu : array  — Frequency grid
    nu0 : float  — Line center frequency
    gamma : float  — Full width at half maximum (FWHM)
    """
    return (1 / np.pi) * (gamma / 2) / ((nu - nu0)**2 + (gamma / 2)**2)


def gaussian(nu: np.ndarray, nu0: float, sigma: float) -> np.ndarray:
    """Gaussian (Doppler-broadened) line profile.

    G(nu) = (1/(sigma*sqrt(2*pi))) * exp(-(nu-nu0)^2 / (2*sigma^2))

    Parameters
    ----------
    nu : array  — Frequency grid
    nu0 : float  — Line center frequency
    sigma : float  — Standard deviation (Doppler width)
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((nu - nu0) / sigma)**2
    )


def voigt(nu: np.ndarray, nu0: float, sigma: float,
          gamma: float) -> np.ndarray:
    """Voigt profile via numerical convolution of Gaussian and Lorentzian.

    The Voigt profile is the convolution G * L, representing a line
    broadened by both Doppler (Gaussian) and pressure (Lorentzian)
    mechanisms simultaneously.

    Uses direct numerical convolution (no scipy dependency).
    """
    # Evaluate Lorentzian on the same grid
    L = lorentzian(nu, nu0, gamma)
    # Convolve with Gaussian kernel
    dnu = nu[1] - nu[0]
    # Gaussian kernel centered at zero
    kernel_half = int(5 * sigma / dnu)
    nu_kernel = np.arange(-kernel_half, kernel_half + 1) * dnu
    G_kernel = gaussian(nu_kernel + nu0, nu0, sigma) * dnu
    G_kernel /= G_kernel.sum()  # normalize
    V = np.convolve(L, G_kernel, mode='same')
    return V
```

### 10.2 Diffraction Grating Spectrometer

```python
def grating_angles(wavelengths: np.ndarray, d: float, theta_i: float,
                   m: int = 1) -> np.ndarray:
    """Compute diffraction angles for a grating spectrometer.

    Grating equation: d * (sin(theta_i) + sin(theta_m)) = m * lambda

    Parameters
    ----------
    wavelengths : array  — Wavelengths in same units as d
    d : float  — Groove spacing
    theta_i : float  — Incidence angle in radians
    m : int  — Diffraction order

    Returns
    -------
    theta_m : array  — Diffraction angles (radians); NaN where no solution exists
    """
    sin_theta_m = m * wavelengths / d - np.sin(theta_i)
    # Physical solutions require |sin(theta_m)| <= 1
    valid = np.abs(sin_theta_m) <= 1.0
    theta_m = np.full_like(wavelengths, np.nan)
    theta_m[valid] = np.arcsin(sin_theta_m[valid])
    return theta_m


def resolving_power_grating(N: int, m: int = 1) -> float:
    """Resolving power of a diffraction grating: R = m * N."""
    return m * N


def focal_plane_positions(wavelengths: np.ndarray, d: float,
                          theta_i: float, f: float,
                          m: int = 1) -> np.ndarray:
    """Map wavelengths to positions on the focal plane of a spectrometer.

    Parameters
    ----------
    f : float  — Focal length of the camera mirror/lens
    Returns x-positions on the detector (same units as f).
    """
    theta_m = grating_angles(wavelengths, d, theta_i, m)
    # Linear dispersion: x = f * theta_m (small angle approx for deviation)
    theta_center = np.nanmedian(theta_m)
    x = f * np.tan(theta_m - theta_center)
    return x
```

### 10.3 Fabry-Pérot Transmission

```python
def fabry_perot_transmission(wavelength: np.ndarray, d: float,
                              R: float, n: float = 1.0,
                              theta: float = 0.0) -> np.ndarray:
    """Compute Fabry-Pérot etalon transmission (Airy function).

    T = 1 / (1 + F * sin^2(delta/2))
    delta = 4*pi*n*d*cos(theta) / lambda
    F = 4*R / (1 - R)^2

    Parameters
    ----------
    wavelength : array  — Wavelength(s)
    d : float  — Mirror separation (same units as wavelength)
    R : float  — Mirror reflectance (0 to 1)
    n : float  — Refractive index of gap medium
    theta : float  — Angle of incidence (radians)
    """
    delta = 4 * np.pi * n * d * np.cos(theta) / wavelength
    F = 4 * R / (1 - R)**2
    T = 1.0 / (1.0 + F * np.sin(delta / 2)**2)
    return T


def finesse(R: float) -> float:
    """Reflectance finesse of a Fabry-Pérot interferometer."""
    return np.pi * np.sqrt(R) / (1 - R)
```

### 10.4 Beer-Lambert Absorption

```python
def beer_lambert(wavelengths: np.ndarray,
                 epsilon: np.ndarray | float,
                 c: float, l: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute transmittance and absorbance using Beer-Lambert law.

    A = epsilon * c * l
    T = 10^(-A)

    Parameters
    ----------
    wavelengths : array  — Wavelength grid (for labeling; not used in calculation)
    epsilon : array or float  — Molar absorption coefficient(s) (L/(mol*cm))
    c : float  — Concentration (mol/L)
    l : float  — Path length (cm)

    Returns
    -------
    transmittance, absorbance : arrays
    """
    epsilon = np.asarray(epsilon)
    A = epsilon * c * l
    T = 10.0**(-A)
    return T, A
```

---

## 11. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Photon energy | $E = h\nu = hc/\lambda$ |
| Kirchhoff's laws | Continuous, emission, absorption spectra |
| Einstein coefficients | $A_{21} = (8\pi h\nu^3/c^3)B_{21}$; $g_1 B_{12} = g_2 B_{21}$ |
| Natural broadening | Lorentzian: $\gamma = 1/(2\pi\tau)$ |
| Doppler broadening | Gaussian: $\sigma_D = (\nu_0/c)\sqrt{k_BT/m}$ |
| Voigt profile | Convolution of Gaussian and Lorentzian |
| Grating equation | $d(\sin\theta_i + \sin\theta_m) = m\lambda$ |
| Grating resolving power | $R = mN$ |
| FP transmission | $T = [1 + F\sin^2(\delta/2)]^{-1}$ |
| FP finesse | $\mathcal{F} = \pi\sqrt{R}/(1-R)$ |
| FTS resolution | $\Delta\nu = 1/(2\delta_{\max})$ |
| Beer-Lambert law | $A = \varepsilon c l$; $T = 10^{-A}$ |
| Raman shift | $\nu_{\text{scattered}} = \nu_0 \pm \nu_{\text{vib}}$ |
| Stokes shift | Fluorescence redshifted from excitation |

---

## 12. Exercises

### Exercise 1: Line Profile Analysis

The hydrogen Balmer-$\alpha$ line (656.28 nm) is observed in a gas discharge at temperature $T = 10{,}000$ K and electron density $n_e = 10^{21}\,\text{m}^{-3}$. (a) Calculate the Doppler width $\Delta\lambda_D$. (b) The pressure broadening width is approximately $\Delta\lambda_P \approx 0.02$ nm at this density. Calculate the Lorentzian FWHM in frequency units. (c) Plot the Voigt profile and compare it with pure Gaussian and pure Lorentzian profiles of the same widths. (d) At what distance from line center does the Lorentzian wing dominate over the Gaussian?

### Exercise 2: Grating Spectrometer Design

Design a Czerny-Turner spectrometer to resolve the sodium D doublet (589.0 nm and 589.6 nm, separation 0.6 nm). (a) What resolving power is needed? (b) For a 1200 grooves/mm grating in first order, what minimum illuminated width is required? (c) If the camera focal length is 500 mm, what is the linear separation of the two lines on the detector? (d) What is the free spectral range in first order? In second order?

### Exercise 3: Fabry-Pérot Spectroscopy

A Fabry-Pérot etalon has mirror separation $d = 10$ mm, reflectance $R = 0.97$, and gap refractive index $n = 1.0$. (a) Calculate the finesse, FSR (in nm at $\lambda = 500$ nm), and resolving power. (b) Plot the transmission function over a 1 nm wavelength range around 500 nm. (c) How many transmission peaks fall within this range? (d) If the etalon is scanned by changing the gap spacing by $\Delta d = 250$ nm (using a piezoelectric actuator), over what wavelength range does the transmission peak scan?

### Exercise 4: Quantitative Absorption Analysis

A solution of potassium permanganate (KMnO$_4$) has a peak molar absorption coefficient $\varepsilon = 2{,}455\,\text{L}\,\text{mol}^{-1}\,\text{cm}^{-1}$ at 525 nm. (a) Plot the absorbance and transmittance at 525 nm for concentrations $c = 0.01, 0.02, 0.05, 0.1, 0.2$ mmol/L in a 1 cm cuvette. (b) If the measured transmittance is 35%, what is the concentration? (c) At what concentration does the absorbance reach $A = 2$ (1% transmittance)? (d) Discuss what happens to the accuracy of concentration measurements at very high and very low absorbance values.

---

## 13. References

1. Hecht, E. (2017). *Optics* (5th ed.). Pearson. — Chapter 9 on interference (Fabry-Pérot), Chapter 10 on diffraction (gratings).
2. Demtröder, W. (2015). *Laser Spectroscopy 1: Basic Principles* (5th ed.). Springer. — Comprehensive reference on spectroscopic techniques and laser methods.
3. Banwell, C. N., & McCash, E. M. (1994). *Fundamentals of Molecular Spectroscopy* (4th ed.). McGraw-Hill. — Accessible introduction to rotational, vibrational, and electronic spectroscopy.
4. Hollas, J. M. (2004). *Modern Spectroscopy* (4th ed.). Wiley. — Covers all major spectroscopic techniques with clear explanations.
5. Thorne, A. P., Litzén, U., & Johansson, S. (1999). *Spectrophysics: Principles and Applications*. Springer. — Detailed treatment of spectroscopic instruments and techniques.
6. Griffiths, P. R., & de Haseth, J. A. (2007). *Fourier Transform Infrared Spectrometry* (2nd ed.). Wiley. — The standard FTIR reference.
7. Ferraro, J. R., Nakamoto, K., & Brown, C. W. (2003). *Introductory Raman Spectroscopy* (2nd ed.). Academic Press. — Accessible Raman spectroscopy text.
8. Gray, D. F. (2022). *The Observation and Analysis of Stellar Photospheres* (4th ed.). Cambridge University Press. — Stellar spectroscopy including equivalent widths, curve of growth, and abundance analysis.

---

[← Previous: 16. Adaptive Optics](16_Adaptive_Optics.md) | [Overview →](00_Overview.md)

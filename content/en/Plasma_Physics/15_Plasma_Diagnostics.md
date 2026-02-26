# 15. Plasma Diagnostics

## Learning Objectives

- Understand the principles of Langmuir probe measurements for density, temperature, and plasma potential
- Explain Thomson scattering (incoherent and coherent) for non-perturbative density and temperature diagnostics
- Apply interferometry and reflectometry for line-integrated and local density measurements
- Use spectroscopy to measure temperature, density, flow velocity, and magnetic field
- Understand magnetic diagnostics for current, stored energy, and internal magnetic field measurements
- Describe low-temperature plasma diagnostics for industrial applications

## 1. Why Plasma Diagnostics?

### 1.1 The Challenge of Plasma Measurement

Plasmas present unique measurement challenges:

1. **Extreme environments**: High temperature (10⁶–10⁸ K), low density (10¹⁴–10²¹ m⁻³), strong magnetic fields (1–10 T)

2. **Sensitivity to perturbations**: Inserting a probe can cool the plasma, introduce impurities, or perturb the equilibrium

3. **No direct access**: In fusion devices, the plasma is confined by magnetic fields and surrounded by vacuum vessels

4. **Multi-scale**: Need to measure phenomena from mm (turbulence) to meters (equilibrium)

5. **Time-varying**: Plasmas evolve on time scales from nanoseconds (waves) to seconds (discharge duration)

**Solution**: Plasma diagnostics use a combination of:
- **Non-perturbative** techniques (light scattering, spectroscopy, interferometry)
- **Minimally perturbative** techniques (small probes, edge measurements)
- **Passive** diagnostics (observe natural emissions)
- **Active** diagnostics (inject particles/waves and observe response)

### 1.2 Diagnostic Goals

Different physics questions require different diagnostics:

| Quantity | Why it matters | Diagnostic |
|----------|----------------|------------|
| $n_e$ | Plasma density, confinement | Interferometry, Thomson scattering, Langmuir probe |
| $T_e$, $T_i$ | Energy content, confinement | Thomson scattering, spectroscopy, CXRS |
| $\mathbf{B}$ | Equilibrium, stability | Magnetic coils, MSE, Zeeman splitting |
| $I_p$ | Plasma current, stability | Rogowski coil |
| $Z_{eff}$ | Impurity content, radiation | Bremsstrahlung, spectroscopy |
| $\mathbf{v}$ | Flows, rotation, transport | Doppler spectroscopy, CXRS |
| Fluctuations | Turbulence, instabilities | Probes, reflectometry, BES |

A modern fusion device uses **dozens** of diagnostics operating simultaneously to build a complete picture of the plasma.

## 2. Langmuir Probe

### 2.1 Principle

A **Langmuir probe** (named after Irving Langmuir, 1920s) is a simple metal electrode inserted into the plasma. By sweeping the probe voltage $V$ and measuring the current $I$, we can deduce $n_e$, $T_e$, and the plasma potential $V_p$.

**Advantages**:
- Simple and inexpensive
- Direct local measurement
- Fast time response (μs)

**Disadvantages**:
- Perturbative (heats/cools plasma locally)
- Only works in edge plasmas (would melt in core)
- Requires careful interpretation (sheath theory)

### 2.2 Sheath Formation

When a probe is inserted into a plasma, a **sheath** (Debye sheath) forms around it. Electrons, being more mobile than ions, initially flow to the probe, charging it negatively. This repels further electrons and attracts ions, creating a quasi-neutral plasma with a thin non-neutral sheath.

The sheath thickness is $\sim$ few Debye lengths:
$$\lambda_D = \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}}$$

Typical $\lambda_D \sim 10$–$100$ μm.

### 2.3 I-V Characteristic

The current-voltage characteristic has three regions:

```
           I
           ^
           |     Electron saturation
         Ie|  ___________________
           | /
           |/
           |   Electron retardation
           |       /|
     ------|------/-|----------> V
           |     /  |V_f  V_p
           |    /   |
           |   /    |
           |  /     Ion saturation
     -Ii   | /
           |/______________
```

**1. Ion saturation region** ($V \ll V_p$):

The probe is biased negatively relative to the plasma. Electrons are repelled; only ions reach the probe. The ion current is:

$$I_{\text{sat},i} = -e n_i u_B A_p$$

where $A_p$ is the probe area and $u_B$ is the **Bohm velocity**:

$$u_B = \sqrt{\frac{k_B T_e}{m_i}}$$

The Bohm criterion states that ions must enter the sheath with at least this velocity for a stable sheath.

**2. Electron retardation region** ($V_f < V < V_p$):

As the voltage increases, some electrons can overcome the potential barrier. The electron current follows a Boltzmann distribution:

$$I_e = I_{e,\text{sat}} \exp\left( \frac{e(V - V_p)}{k_B T_e} \right)$$

The total current is:
$$I = I_e + I_i \approx I_{e,\text{sat}} \exp\left( \frac{e(V - V_p)}{k_B T_e} \right) - I_{\text{sat},i}$$

At the **floating potential** $V_f$, the total current is zero: $I = 0$.

**3. Electron saturation region** ($V > V_p$):

The probe collects all electrons reaching it. For a small probe (much smaller than electron mean free path), the current saturates at:

$$I_{e,\text{sat}} = \frac{1}{4} e n_e \bar{v}_e A_p$$

where $\bar{v}_e = \sqrt{8 k_B T_e / (\pi m_e)}$ is the mean electron speed.

### 2.4 Extracting Plasma Parameters

From the I-V curve:

**Electron temperature** $T_e$:

Plot $\ln(I_e)$ vs. $V$ in the electron retardation region. The slope is:

$$\frac{d \ln I_e}{dV} = \frac{e}{k_B T_e}$$

So:
$$T_e = \frac{e}{k_B} \left( \frac{d \ln I_e}{dV} \right)^{-1}$$

**Plasma potential** $V_p$:

The "knee" of the I-V curve (where the slope changes from exponential to flat) is the plasma potential. More precisely, $V_p$ is where the second derivative $d^2I/dV^2$ has a maximum.

**Ion density** $n_i$:

From the ion saturation current:
$$n_i = \frac{I_{\text{sat},i}}{e u_B A_p} = \frac{I_{\text{sat},i}}{e A_p} \sqrt{\frac{m_i}{k_B T_e}}$$

**Electron density** $n_e$:

From quasi-neutrality, $n_e \approx n_i$.

### 2.5 Complications

Real Langmuir probes face several complications:

1. **Magnetic field**: In magnetized plasmas, the sheath is elongated along $\mathbf{B}$. The collection area is $A_\parallel \sim \pi r^2$ (cross-field) or $A_\perp \sim 2\pi r L$ (along field), depending on orientation.

2. **Flowing plasma**: If the plasma flows relative to the probe, the I-V curve is distorted. Use a **Mach probe** (two probes facing opposite directions) to measure flow.

3. **Secondary electron emission**: Energetic ions can knock out electrons from the probe, adding a spurious current.

4. **Collisions in sheath**: If the sheath is thick compared to the mean free path, ion-neutral collisions occur, reducing ion current.

5. **RF oscillations**: In RF-heated plasmas, the probe potential oscillates at the RF frequency, complicating interpretation.

### 2.6 Variants: Double and Triple Probes

**Double probe**: Two floating probes biased relative to each other, not relative to the wall. Avoids the need to draw large currents from the power supply. Used in flowing plasmas.

**Triple probe**: Three probes at different biases, measured simultaneously. Allows measurement of $T_e$ and $n_e$ from a single time instant (no sweeping), useful for fast fluctuations.

## 3. Thomson Scattering

### 3.1 Principle

**Thomson scattering** is the scattering of electromagnetic waves by free electrons. A high-power laser is fired through the plasma, and the scattered light is collected and analyzed.

**Advantages**:
- Non-perturbative (photons don't heat plasma)
- Measures $n_e$ and $T_e$ simultaneously
- High spatial resolution (mm)
- Absolute calibration (no reference plasma needed)

**Disadvantages**:
- Expensive (requires high-power laser and sensitive detectors)
- Complex analysis
- Limited time resolution (laser repetition rate)

### 3.2 Scattering Regimes

The scattering depends on the parameter:

$$\alpha = \frac{1}{k \lambda_D}$$

where $k = |\mathbf{k}_s - \mathbf{k}_i|$ is the scattering wave vector (difference between scattered and incident photon momenta).

**1. Incoherent scattering** ($\alpha \ll 1$, or $k \lambda_D \gg 1$):

Scattering from individual electrons. The electron density fluctuations are uncorrelated. The scattered spectrum reflects the electron velocity distribution.

**2. Coherent scattering** ($\alpha \gg 1$, or $k \lambda_D \ll 1$):

Scattering from collective plasma oscillations (ion acoustic waves, electron plasma waves). Electrons scatter coherently, enhancing the signal.

### 3.3 Incoherent Thomson Scattering

For a **Maxwellian** electron distribution, the scattered spectrum is:

$$S(\omega) \propto \exp\left( -\frac{(\omega - \omega_0)^2}{2 k^2 v_{te}^2} \right)$$

where $\omega_0$ is the laser frequency and $v_{te} = \sqrt{k_B T_e / m_e}$ is the electron thermal velocity.

The spectrum is **Doppler-broadened** by the electron motion:

$$\Delta \omega = k v_{te} = k \sqrt{\frac{k_B T_e}{m_e}}$$

**Measuring** $T_e$:

Fit the scattered spectrum to a Gaussian. The width gives $T_e$:

$$T_e = \frac{m_e (\Delta \omega)^2}{k^2 k_B}$$

**Measuring** $n_e$:

The total scattered power is:

$$P_s = P_i \sigma_T n_e \Delta V \Delta \Omega$$

where:
- $P_i$: incident laser power
- $\sigma_T = 6.65 \times 10^{-29}$ m²: Thomson cross-section
- $n_e$: electron density
- $\Delta V$: scattering volume
- $\Delta \Omega$: solid angle of collection optics

By measuring $P_s$ and knowing $P_i$, $\Delta V$, $\Delta \Omega$, we can deduce $n_e$.

### 3.4 Coherent Thomson Scattering (Collective Scattering)

When $k \lambda_D < 1$, scattering is from **collective fluctuations**. The scattered spectrum has peaks at the ion acoustic wave frequency:

$$\omega_{ia} \approx k c_s = k \sqrt{\frac{k_B (T_e + T_i)}{m_i}}$$

The spectrum shows:
- **Central peak**: electron plasma wave (Langmuir oscillation)
- **Side peaks**: ion acoustic waves (blue- and red-shifted)

From the side peak positions, we get $c_s$ → $(T_e + T_i)$.
From the peak widths, we get Landau damping → $T_i$.

This allows measurement of **ion temperature** $T_i$, which is hard to get from incoherent scattering.

### 3.5 Thomson Scattering in Fusion Devices

**ITER Thomson scattering system**:
- Laser: Nd:YAG (1064 nm), 6 J/pulse, 20 Hz
- ~100 spatial points along the beam
- Time resolution: 50 ms (one shot per 50 ms)
- Measures $n_e$ (10¹⁸–10²¹ m⁻³) and $T_e$ (0.1–50 keV)

Thomson scattering is the **gold standard** for core $n_e$ and $T_e$ profiles in tokamaks.

## 4. Interferometry and Reflectometry

### 4.1 Microwave Interferometry

A microwave or laser beam passes through the plasma. The phase shift is proportional to the **line-integrated density**:

$$\Delta \phi = \frac{2\pi}{\lambda} \int n_e \, dl$$

(neglecting the wavelength dependence, which is weak).

More precisely:
$$\Delta \phi = \frac{e^2}{2 \epsilon_0 m_e \omega c} \int n_e \, dl$$

where $\omega$ is the beam frequency.

**Measuring** $\bar{n}_e L$:

From the phase shift, we get the **line-integrated density**:

$$\int n_e \, dl = \frac{2 \epsilon_0 m_e \omega c}{e^2} \Delta \phi$$

To get the density profile $n_e(r)$, we need **multiple chords** at different impact parameters. Then use **Abel inversion** (assuming cylindrical symmetry) or **tomographic inversion** (2D).

**Abel inversion**:

For a cylindrically symmetric plasma:

$$\int n_e \, dl = 2 \int_r^a n_e(r') \frac{r' \, dr'}{\sqrt{r'^2 - r^2}}$$

where $r$ is the impact parameter. Inverting:

$$n_e(r) = -\frac{1}{\pi} \int_r^a \frac{d}{dr'} \left( \int n_e \, dl \right) \frac{dr'}{\sqrt{r'^2 - r^2}}$$

### 4.2 Reflectometry

Instead of passing through the plasma, send a microwave that **reflects** at the plasma cutoff layer:

$$\omega^2 = \omega_{pe}^2 = \frac{n_e e^2}{\epsilon_0 m_e}$$

The cutoff density is:
$$n_c = \frac{\epsilon_0 m_e \omega^2}{e^2} \approx 1.24 \times 10^{10} \, f^2 \quad (\text{m}^{-3}, \, f \text{ in GHz})$$

By sweeping the frequency, we probe different density layers. The time delay gives the position of the cutoff layer.

**Advantages**:
- **Local** measurement (reflects at specific density)
- Fast (can measure fluctuations, turbulence)

**Disadvantages**:
- Complex interpretation (phase jumps, multiple reflections)
- Limited to edge/gradient regions

**Density fluctuation measurement**:

Reflectometry is excellent for measuring **turbulence**. The scattered signal fluctuates due to density fluctuations:

$$\frac{\delta n_e}{n_e} \sim \text{few percent}$$

This is used to study edge turbulence in tokamaks.

### 4.3 Faraday Rotation

In a magnetized plasma, the plane of polarization of linearly polarized light rotates:

$$\theta = \frac{e^3}{2 \epsilon_0 m_e^2 \omega^2 c} \int n_e B_\parallel \, dl$$

This measures $\int n_e B_\parallel \, dl$, giving information on the magnetic field (if $n_e$ is known from interferometry).

## 5. Spectroscopy

### 5.1 Line Emission

Atoms and ions emit characteristic spectral lines when electrons transition between energy levels. By observing these lines, we can:

- **Identify species**: Each element has unique lines (e.g., H$_\alpha$ 656.3 nm, He II 468.6 nm)
- **Measure temperature**: Line intensity ratios, Doppler broadening
- **Measure density**: Stark broadening, line ratios
- **Measure flow velocity**: Doppler shift
- **Measure magnetic field**: Zeeman splitting

### 5.2 Doppler Broadening → Temperature

Thermal motion causes **Doppler broadening**:

$$\Delta \lambda = \lambda_0 \sqrt{\frac{2 k_B T}{m c^2}}$$

where $m$ is the ion mass.

**Measuring** $T_i$:

Fit the spectral line to a Gaussian:

$$I(\lambda) = I_0 \exp\left[ -\frac{(\lambda - \lambda_0)^2}{2 (\Delta \lambda)^2} \right]$$

From $\Delta \lambda$, deduce $T_i$:

$$T_i = \frac{m c^2}{2 k_B} \left( \frac{\Delta \lambda}{\lambda_0} \right)^2$$

**Example**: For C⁶⁺ (fully stripped carbon) at $T_i = 1$ keV, $\lambda_0 = 529$ nm:

$$\Delta \lambda = 529 \times 10^{-9} \sqrt{\frac{2 \times 1.6 \times 10^{-16}}{12 \times 1.67 \times 10^{-27} \times (3 \times 10^8)^2}} \approx 0.01 \text{ nm}$$

This requires a high-resolution spectrometer ($R = \lambda / \Delta \lambda \sim 50{,}000$).

### 5.3 Stark Broadening → Density

**Stark broadening** occurs when the electric fields from nearby ions and electrons shift the energy levels. The line width is proportional to $n_e^{2/3}$:

$$\Delta \lambda_{Stark} \propto n_e^{2/3}$$

For hydrogen Balmer lines (H$_\alpha$, H$_\beta$), empirical formulas exist:

$$\Delta \lambda_{H\alpha} (\text{nm}) \approx 4 \times 10^{-16} n_e^{2/3}$$

**Measuring** $n_e$:

Observe the H$_\alpha$ line width (after subtracting Doppler and instrumental broadening):

$$n_e = \left( \frac{\Delta \lambda_{H\alpha}}{4 \times 10^{-16}} \right)^{3/2}$$

This is commonly used in edge plasmas and low-temperature plasmas.

### 5.4 Doppler Shift → Flow Velocity

A moving plasma shifts the spectral line:

$$\Delta \lambda = \lambda_0 \frac{v_{\parallel}}{c}$$

where $v_\parallel$ is the velocity component along the line of sight.

**Measuring** $v$:

$$v_{\parallel} = c \frac{\Delta \lambda}{\lambda_0}$$

**Example**: For toroidal rotation in a tokamak, observe C⁶⁺ emission. If $\Delta \lambda = 0.05$ nm at $\lambda_0 = 529$ nm:

$$v_{\parallel} = 3 \times 10^8 \times \frac{0.05 \times 10^{-9}}{529 \times 10^{-9}} \approx 28 \text{ km/s}$$

Typical tokamak rotation velocities are 10–100 km/s.

### 5.5 Charge Exchange Recombination Spectroscopy (CXRS)

A brilliant technique to measure **ion temperature** and **flow velocity** in the core plasma:

1. Inject a beam of **neutral atoms** (usually deuterium) into the plasma.
2. Fast ions in the plasma undergo **charge exchange** with neutrals:
   $$\text{D}^+ + \text{D}^0 \to \text{D}^0 + \text{D}^+$$
   or for impurities (e.g., carbon):
   $$\text{C}^{6+} + \text{D}^0 \to \text{C}^{5+} + \text{D}^+$$
3. The newly formed $\text{C}^{5+}$ is in an excited state and emits light (e.g., 529 nm).
4. This light has the Doppler shift and broadening of the **bulk ions**, giving $T_i$ and $v_i$.

**Advantages**:
- Measures core $T_i$ and $v_i$ (not possible with edge spectroscopy)
- High spatial resolution (along beam path)

**Disadvantages**:
- Requires neutral beam injection (perturbs plasma)
- Complex calibration

CXRS is essential for **rotation measurements** in tokamaks, which affect MHD stability and turbulence.

### 5.6 Zeeman Splitting → Magnetic Field

In a magnetic field, spectral lines split into multiple components due to the **Zeeman effect**:

$$\Delta E = \mu_B g_J m_J B$$

where $\mu_B$ is the Bohr magneton, $g_J$ is the Landé g-factor, $m_J$ is the magnetic quantum number.

The wavelength splitting is:

$$\Delta \lambda = \lambda_0^2 \frac{e B}{4\pi m_e c^2}$$

**Example**: For H$_\alpha$ ($\lambda_0 = 656.3$ nm) in $B = 1$ T:

$$\Delta \lambda \approx (656.3 \times 10^{-9})^2 \times \frac{1.6 \times 10^{-19} \times 1}{4\pi \times 9.11 \times 10^{-31} \times (3 \times 10^8)^2} \approx 0.014 \text{ nm}$$

This is detectable with high-resolution spectroscopy.

**Motional Stark Effect (MSE)**:

In a tokamak, the neutral beam atoms see a Lorentz-transformed electric field:

$$\mathbf{E}' = -\mathbf{v}_{beam} \times \mathbf{B}$$

This causes Stark splitting, whose polarization depends on the direction of $\mathbf{B}$. By measuring the polarization angle, we get the **pitch angle** of the magnetic field:

$$\tan \theta = \frac{B_\theta}{B_\phi}$$

This gives the **current density profile** via Ampère's law:

$$J_\phi = \frac{1}{\mu_0} \frac{\partial B_\theta}{\partial r}$$

MSE is crucial for measuring $q(r)$, the safety factor profile.

## 6. Magnetic Diagnostics

### 6.1 Rogowski Coil

A **Rogowski coil** is a toroidal coil wrapped around the plasma. From Ampère's law:

$$\oint \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{enclosed}$$

The coil measures the time derivative of the plasma current:

$$V_{coil} = -\frac{d\Phi}{dt} = -\mu_0 A N \frac{dI_p}{dt}$$

where $N$ is the number of turns, $A$ is the cross-sectional area.

Integrating in time:

$$I_p(t) = -\frac{1}{\mu_0 A N} \int V_{coil} \, dt$$

**Accuracy**: The Rogowski coil gives the **total plasma current**, essential for equilibrium reconstruction and discharge control.

### 6.2 Magnetic Pickup Coils

Small coils (flux loops) measure the local magnetic field:

$$V_{coil} = -\frac{d\Phi}{dt} = -A \frac{dB}{dt}$$

By placing many coils at different positions, we reconstruct the 2D poloidal field $B_\theta(r, \theta)$.

**Grad-Shafranov equilibrium reconstruction**:

Combine magnetic measurements with pressure (from Thomson scattering) to solve the Grad-Shafranov equation:

$$\Delta^* \psi = -\mu_0 r^2 \frac{dp}{d\psi} - F \frac{dF}{d\psi}$$

This gives the magnetic flux surfaces and $q(r)$.

### 6.3 Diamagnetic Loop

A poloidal loop measures the **diamagnetic flux**:

$$\Phi_{dia} = \int \mathbf{B}_\theta \cdot d\mathbf{A}$$

The diamagnetic effect reduces $B_\theta$ when plasma pressure is present:

$$B_\theta^{vac} - B_\theta^{plasma} \propto p$$

The stored energy is:

$$W = \frac{3}{2} \int p \, dV \propto \Phi_{dia}$$

This gives a fast, simple measurement of **stored energy**, crucial for fusion performance ($Q = P_{fusion} / P_{input} \propto W$).

### 6.4 Motional Stark Effect (MSE)

As discussed in Section 5.6, MSE measures the **internal magnetic field** from Stark splitting of neutral beam emission. This is one of the few ways to measure $B(r)$ inside the plasma.

## 7. Low-Temperature and Industrial Plasmas

### 7.1 Glow Discharge

A **glow discharge** is a low-pressure gas discharge with a characteristic glow (visible light emission). It's used in:
- Plasma processing (etching, deposition)
- Lighting (neon signs, fluorescent lamps)
- Displays (plasma TVs, now obsolete)

**Paschen's Law**: The breakdown voltage $V_b$ (minimum voltage to sustain discharge) depends on the product $p d$ (pressure × gap distance):

$$V_b = \frac{B p d}{\ln(A p d) - \ln(\ln(1 + 1/\gamma))}$$

where $A$ and $B$ are gas-dependent constants, and $\gamma$ is the secondary emission coefficient.

There's a **minimum** breakdown voltage at a specific $p d$ (Paschen minimum).

**Example**: For air at $p d \approx 0.5$ Torr·cm, $V_b \approx 300$ V.

### 7.2 RF Plasmas

**Capacitively Coupled Plasma (CCP)**:

Two parallel electrodes driven by RF (typically 13.56 MHz). Ions respond to the time-averaged potential; electrons oscillate at the RF frequency.

**Inductively Coupled Plasma (ICP)**:

RF current in an external coil induces a time-varying magnetic field, which induces an azimuthal electric field → drives plasma current → heats electrons.

ICP achieves higher density ($10^{17}$–$10^{18}$ m⁻³) than CCP.

### 7.3 Plasma Processing Diagnostics

**Optical Emission Spectroscopy (OES)**:

Monitor the emission lines to track:
- **Etch endpoint**: When the substrate is etched through, emission from the underlying layer appears
- **Gas composition**: Detect impurities, monitor precursor dissociation

**Langmuir Probe**:

Measure $n_e$, $T_e$ in the chamber. Must be at **floating potential** to avoid sputtering.

**Quadrupole Mass Spectrometer (QMS)**:

Sample neutral and ion species, identify chemical reactions.

**Laser-Induced Fluorescence (LIF)**:

Excite neutral species with a tunable laser, measure fluorescence → get density and velocity of specific species.

### 7.4 Applications

- **Semiconductor manufacturing**: Plasma etching (anisotropic, selective), PECVD (plasma-enhanced chemical vapor deposition)
- **Surface treatment**: Cleaning, activation, coating
- **Sterilization**: Low-temperature plasma kills bacteria without heat
- **Lighting**: Energy-efficient (fluorescent, LED plasma)
- **Materials processing**: Nitriding, carburizing, hardening

## 8. Python Code Examples

### 8.1 Langmuir Probe I-V Curve Fitting

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate synthetic I-V data
def langmuir_current(V, n_e, T_e, V_p, V_f, A_p):
    """
    Langmuir probe current model.
    """
    e = 1.6e-19
    k_B = 1.38e-23
    m_e = 9.11e-31
    m_i = 1.67e-27  # proton

    # Bohm velocity
    u_B = np.sqrt(k_B * T_e / m_i)

    # Ion saturation current
    I_sat_i = e * n_e * u_B * A_p

    # Electron current (Boltzmann)
    v_bar_e = np.sqrt(8 * k_B * T_e / (np.pi * m_e))
    I_sat_e = 0.25 * e * n_e * v_bar_e * A_p

    # Total current
    I = np.where(V < V_p,
                 I_sat_e * np.exp(e * (V - V_p) / (k_B * T_e)) - I_sat_i,
                 I_sat_e - I_sat_i)

    return I

# True parameters
n_e_true = 1e16  # m^-3
T_e_true = 3.0   # eV
V_p_true = 10.0  # V
V_f_true = 5.0   # V (not used directly, but implicit)
A_p = 1e-4       # m^2 (1 cm^2)

# Generate data
V = np.linspace(-20, 20, 100)
I_true = langmuir_current(V, n_e_true, T_e_true * 1.6e-19, V_p_true, V_f_true, A_p)
I_noisy = I_true + np.random.normal(0, 0.1e-6, len(V))

# Fit in electron retardation region only: below V_p the electron current
# follows a Boltzmann exponential, so a linear fit in log-space gives T_e
# directly. Including the saturation region would break the linearity.
mask = (V > 0) & (V < V_p_true)
V_fit = V[mask]
I_fit = I_noisy[mask]

# Adding 1e-6 A shifts the origin to avoid log(0) or log(negative) from noise.
# The shift is ~1000× smaller than the saturation current, so it does not
# significantly bias the slope and hence the inferred temperature.
I_e_approx = I_fit + 1e-6  # shift to avoid log(negative)
ln_I = np.log(np.abs(I_e_approx))

# Linear fit: ln(I) = (e/k_B T_e) * V + const
# The slope encodes T_e because the Boltzmann factor gives I_e ∝ exp(eV/kT_e).
p = np.polyfit(V_fit, ln_I, 1)
slope = p[0]

e = 1.6e-19
k_B = 1.38e-23
T_e_fit = e / (k_B * slope)

print("Langmuir Probe Analysis:")
print(f"  True T_e = {T_e_true:.2f} eV")
print(f"  Fitted T_e = {T_e_fit:.2f} eV")
print()

# Find V_p via the second derivative: the I-V curve transitions from exponential
# (electron retardation) to flat (saturation) at V_p. The second derivative
# peaks sharply at this inflection point, providing a robust numerical criterion
# that does not require manual identification of the "knee".
dI_dV = np.gradient(I_noisy, V)
d2I_dV2 = np.gradient(dI_dV, V)
idx_Vp = np.argmax(d2I_dV2)
V_p_fit = V[idx_Vp]

print(f"  True V_p = {V_p_true:.2f} V")
print(f"  Fitted V_p = {V_p_fit:.2f} V")
print()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# I-V curve
axes[0].plot(V, I_true * 1e6, 'b-', linewidth=2, label='True')
axes[0].plot(V, I_noisy * 1e6, 'r.', markersize=4, label='Noisy data')
axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[0].axvline(V_p_true, color='g', linestyle='--', linewidth=2, label=f'V_p = {V_p_true} V')
axes[0].set_xlabel('Probe voltage V (V)', fontsize=12)
axes[0].set_ylabel('Current I (μA)', fontsize=12)
axes[0].set_title('Langmuir Probe I-V Characteristic', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# ln(I) vs V (electron retardation region)
axes[1].plot(V_fit, ln_I, 'bo', markersize=5, label='Data')
axes[1].plot(V_fit, np.polyval(p, V_fit), 'r-', linewidth=2,
             label=f'Fit: slope = {slope:.2f} V⁻¹\nT_e = {T_e_fit:.2f} eV')
axes[1].set_xlabel('Probe voltage V (V)', fontsize=12)
axes[1].set_ylabel('ln(I)', fontsize=12)
axes[1].set_title('Electron Retardation Region (Temperature Fit)', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('langmuir_probe.png', dpi=150)
plt.show()
```

### 8.2 Interferometry: Density Profile from Phase Shift

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

def abel_inversion(r_impact, line_integral):
    """
    Abel inversion to get radial profile from line-integrated data.

    Assumes cylindrical symmetry: n(r).
    Given: ∫ n(r) dl for different impact parameters r.
    """
    # Sort by impact parameter
    idx = np.argsort(r_impact)
    r = r_impact[idx]
    L = line_integral[idx]

    # Numerical derivative of the line-integral: Abel inversion requires dL/dr,
    # not L itself. np.gradient uses second-order central differences, which
    # suppresses noise amplification better than a simple finite difference.
    dL_dr = np.gradient(L, r)

    # Abel inversion: n(r) = -(1/π) ∫_r^a (dL/dr') / sqrt(r'^2 - r^2) dr'
    n_r = np.zeros_like(r)

    for i in range(len(r)):
        ri = r[i]
        # Integrate from ri to r_max
        # The 1e-10 regularizer prevents the integrand from diverging at r' = ri
        # (the square-root singularity at the lower limit). This is an integrable
        # singularity in the exact Abel transform, but finite discretization makes
        # the denominator exactly zero; 1e-10 is much smaller than dr² so it does
        # not bias the integral.
        integrand = dL_dr[i:] / np.sqrt(r[i:]**2 - ri**2 + 1e-10)  # avoid division by zero
        n_r[i] = -1/np.pi * np.trapz(integrand, r[i:])

    return r, n_r

# Synthetic density profile (parabolic)
a = 0.5  # plasma radius (m)
n_0 = 1e20  # peak density (m^-3)

r_true = np.linspace(0, a, 100)
n_true = n_0 * (1 - (r_true / a)**2)**2

# Compute line-integrated density for different chords
N_chords = 20
r_impact = np.linspace(0, 0.9*a, N_chords)
line_integral = np.zeros(N_chords)

for i, r_imp in enumerate(r_impact):
    # Integrate along the chord
    # For cylindrical symmetry: ∫ n dl = 2 ∫_r_imp^a n(r) r dr / sqrt(r^2 - r_imp^2)
    # Starting at r_imp + 1e-6 rather than r_imp avoids the geometric singularity
    # at the chord tangent point where sqrt(r^2 - r_imp^2) → 0. The offset is
    # negligible relative to the plasma radius a ~ 0.5 m.
    r_chord = np.linspace(r_imp + 1e-6, a, 200)
    # The factor r/sqrt(r^2 - r_imp^2) is the chord-path Jacobian: it converts
    # integration in r (radial coordinate) to integration along the line of sight,
    # accounting for the oblique angle the chord makes with each radial shell.
    integrand = n_0 * (1 - (r_chord/a)**2)**2 * r_chord / np.sqrt(r_chord**2 - r_imp**2)
    line_integral[i] = 2 * np.trapz(integrand, r_chord)

# Add noise
line_integral_noisy = line_integral + np.random.normal(0, 0.02 * n_0 * a, N_chords)

# Abel inversion
r_inverted, n_inverted = abel_inversion(r_impact, line_integral_noisy)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Line-integrated density
axes[0].plot(r_impact * 100, line_integral / (n_0 * a), 'bo', markersize=7, label='True')
axes[0].plot(r_impact * 100, line_integral_noisy / (n_0 * a), 'rx', markersize=7, label='Noisy')
axes[0].set_xlabel('Impact parameter r (cm)', fontsize=12)
axes[0].set_ylabel('Line-integrated density / (n₀ a)', fontsize=12)
axes[0].set_title('Interferometry Measurements', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Density profile
axes[1].plot(r_true * 100, n_true / n_0, 'b-', linewidth=2, label='True profile')
axes[1].plot(r_inverted * 100, np.abs(n_inverted) / n_0, 'r--', linewidth=2, label='Abel inverted')
axes[1].set_xlabel('Radius r (cm)', fontsize=12)
axes[1].set_ylabel('Density n / n₀', fontsize=12)
axes[1].set_title('Reconstructed Density Profile', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('interferometry_inversion.png', dpi=150)
plt.show()

print("Interferometry and Abel Inversion:")
print(f"  Number of chords: {N_chords}")
print(f"  Peak density (true): {n_0:.2e} m⁻³")
print(f"  Peak density (inverted): {np.max(np.abs(n_inverted)):.2e} m⁻³")
print(f"  Relative error: {100 * (np.max(np.abs(n_inverted)) - n_0) / n_0:.1f}%")
```

### 8.3 Doppler Broadening: Fit Spectral Line → Temperature

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    """Gaussian function."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def doppler_width(T, m, lambda_0):
    """
    Doppler width (FWHM) of spectral line.

    T: temperature (eV)
    m: ion mass (kg)
    lambda_0: rest wavelength (m)
    """
    k_B = 1.38e-23
    c = 3e8
    e = 1.6e-19

    T_J = T * e
    sigma_v = np.sqrt(k_B * T_J / m)  # velocity dispersion
    sigma_lambda = lambda_0 * sigma_v / c  # wavelength dispersion

    return sigma_lambda

# Simulate spectral line (C^6+ at 529 nm)
lambda_0 = 529e-9  # m
m_C = 12 * 1.67e-27  # kg
T_true = 1000  # eV (1 keV)

sigma_true = doppler_width(T_true, m_C, lambda_0)

# ±3σ range covers 99.7% of the Gaussian line profile; extending further would
# waste spectral resolution on noise-dominated wings without improving the fit.
lambda_grid = np.linspace(lambda_0 - 3*sigma_true, lambda_0 + 3*sigma_true, 200)

# True spectrum
I_true = gaussian(lambda_grid, 1.0, lambda_0, sigma_true)

# Add noise
I_noisy = I_true + np.random.normal(0, 0.02, len(lambda_grid))

# Fit Gaussian
# Initial guess uses sigma_true * 1.2 (slightly wider than expected): starting
# too narrow can cause the optimizer to get trapped in a local minimum if noise
# creates a narrow false peak; starting slightly wide avoids this.
p0 = [1.0, lambda_0, sigma_true * 1.2]  # initial guess
popt, pcov = curve_fit(gaussian, lambda_grid, I_noisy, p0=p0)

A_fit, mu_fit, sigma_fit = popt

# Infer temperature by inverting σ_λ = λ₀ σ_v/c and σ_v = sqrt(kT/m):
# T = m c² (σ_λ/λ₀)² / k_B. The division by 1.6e-19 converts Joules to eV.
T_fit = (m_C * c**2 / k_B) * (sigma_fit / lambda_0)**2 / 1.6e-19  # eV

print("Doppler Broadening Analysis:")
print(f"  True temperature: {T_true} eV")
print(f"  Fitted temperature: {T_fit:.1f} eV")
print(f"  True sigma: {sigma_true * 1e12:.3f} pm")
print(f"  Fitted sigma: {sigma_fit * 1e12:.3f} pm")
print()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(lambda_grid * 1e9, I_true, 'b-', linewidth=2, label='True (T = 1000 eV)')
plt.plot(lambda_grid * 1e9, I_noisy, 'r.', markersize=5, label='Noisy data')
plt.plot(lambda_grid * 1e9, gaussian(lambda_grid, *popt), 'g--', linewidth=2,
         label=f'Gaussian fit (T = {T_fit:.0f} eV)')

plt.xlabel('Wavelength λ (nm)', fontsize=12)
plt.ylabel('Intensity (normalized)', fontsize=12)
plt.title('Doppler Broadening of C⁶⁺ Line (529 nm)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('doppler_broadening.png', dpi=150)
plt.show()
```

## Summary

In this lesson, we surveyed the key plasma diagnostic techniques:

1. **Langmuir probe**: Simple, local measurement of $n_e$, $T_e$, $V_p$ via I-V characteristics. Useful for edge plasmas but perturbative.

2. **Thomson scattering**: Gold standard for core $n_e$ and $T_e$. Incoherent scattering gives electron distribution; coherent scattering gives ion temperature via collective modes.

3. **Interferometry and reflectometry**: Microwave diagnostics for density. Interferometry gives line-integrated density; reflectometry gives local density and fluctuations.

4. **Spectroscopy**: Rich information from line emission. Doppler broadening → temperature, Stark broadening → density, Doppler shift → flow, Zeeman splitting → magnetic field.

5. **CXRS**: Charge exchange recombination spectroscopy measures core $T_i$ and rotation by observing impurity emission from neutral beam.

6. **Magnetic diagnostics**: Rogowski coil (total current), pickup coils (poloidal field), diamagnetic loop (stored energy), MSE (internal field).

7. **Low-temperature plasmas**: Langmuir probes, OES, mass spectrometry for industrial plasma processing (etching, deposition, sterilization).

Modern fusion experiments use **integrated diagnostics**: combining multiple techniques to build a complete picture of the plasma state. Data fusion and Bayesian inference are emerging tools to combine disparate measurements.

## Practice Problems

### Problem 1: Langmuir Probe in Edge Plasma
A Langmuir probe in the edge of a tokamak measures:
- Ion saturation current: $I_{sat,i} = -5$ mA
- Probe area: $A_p = 2$ mm²
- Electron temperature (from slope): $T_e = 20$ eV

Calculate:
(a) The Bohm velocity $u_B$.
(b) The ion density $n_i$.
(c) The floating potential $V_f$ (assume $T_e = T_i$ and singly charged ions).

### Problem 2: Thomson Scattering Spectrum
A Thomson scattering system uses a Nd:YAG laser ($\lambda = 1064$ nm) at 90° scattering angle. The scattered spectrum shows a Gaussian width of $\Delta \lambda = 2$ nm.

(a) Calculate the electron temperature $T_e$.
(b) If the scattered power is $P_s = 10^{-9}$ W for incident power $P_i = 1$ J/pulse, scattering volume $\Delta V = 1$ mm³, and collection solid angle $\Delta \Omega = 0.01$ sr, estimate the electron density $n_e$.

### Problem 3: Interferometry Abel Inversion
An interferometer measures line-integrated density along 5 chords through a cylindrical plasma (radius $a = 10$ cm):

| Impact parameter $r$ (cm) | Line integral $\int n_e \, dl$ (10¹⁸ m⁻²) |
|---------------------------|--------------------------------------------|
| 0                         | 10.0                                       |
| 3                         | 9.5                                        |
| 5                         | 8.0                                        |
| 7                         | 5.0                                        |
| 9                         | 2.0                                        |

(a) Assuming a parabolic profile $n_e(r) = n_0 (1 - r^2/a^2)^\alpha$, estimate $n_0$ and $\alpha$ by fitting to the data.
(b) Use Abel inversion (numerically or analytically) to reconstruct $n_e(r)$ at $r = 0, 5, 10$ cm.

### Problem 4: Doppler Spectroscopy
The C⁶⁺ line at $\lambda_0 = 529.0$ nm is observed in a tokamak plasma. The measured spectrum has:
- Peak wavelength: $\lambda_{peak} = 529.05$ nm
- FWHM: $\Delta \lambda_{FWHM} = 0.02$ nm

(a) Calculate the toroidal rotation velocity (assume line of sight is perpendicular to toroidal direction).
(b) Calculate the ion temperature $T_i$ from the Doppler broadening (assume carbon mass $m = 12$ amu).
(c) If there's also Stark broadening of $\Delta \lambda_{Stark} = 0.005$ nm, what is the intrinsic Doppler width?

### Problem 5: Magnetic Diagnostics
A tokamak has a Rogowski coil measuring plasma current. The coil has $N = 1000$ turns, cross-sectional area $A = 1$ cm², and the induced voltage is $V_{coil} = -50$ mV during the ramp-up phase (1 s duration).

(a) Calculate the rate of change of plasma current $dI_p / dt$.
(b) If the current starts at zero, what is the final plasma current $I_p$ after 1 s (assuming constant $dI_p/dt$)?
(c) A diamagnetic loop measures stored energy $W = 10$ MJ. For a plasma volume $V = 100$ m³, estimate the average pressure $\langle p \rangle$.

---

**Previous**: [From Kinetic to MHD](./14_From_Kinetic_to_MHD.md) | **Next**: [Projects](./16_Projects.md)

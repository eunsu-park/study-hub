# 11. Electromagnetic Waves in Plasma

## Learning Objectives

- Understand electromagnetic wave propagation in unmagnetized plasmas and the plasma cutoff
- Master the Stix cold plasma dielectric tensor for magnetized plasmas
- Derive dispersion relations for R-wave, L-wave, O-mode, and X-mode
- Analyze whistler wave dispersion and its applications
- Construct and interpret the CMA (Clemmow-Mullaly-Allis) diagram
- Apply Faraday rotation to measure magnetic fields in plasmas

## Introduction

Unlike electrostatic waves (which involve only $\mathbf{E}$ perturbations), **electromagnetic (EM) waves** have both electric and magnetic field components:

$$\mathbf{E}_1 \neq -\nabla\phi, \quad \mathbf{B}_1 \neq 0$$

EM waves in plasmas exhibit rich physics:
- **Cutoffs**: frequencies below which waves cannot propagate (evanescent)
- **Resonances**: frequencies where wave properties diverge
- **Polarization**: wave electric field can be linear, circular, or elliptical
- **Mode conversion**: one wave type transforms into another

These waves are crucial for:
- **Plasma heating**: ECRH (electron cyclotron resonance heating), ICRH (ion cyclotron)
- **Diagnostics**: interferometry, reflectometry, polarimetry
- **Communications**: ionospheric propagation, whistler waves
- **Astrophysics**: pulsar emissions, solar radio bursts

We start with the simplest case (unmagnetized plasma) and build up to the full magnetized plasma theory using the **Stix formalism**.

## 1. EM Waves in Unmagnetized Plasma

### 1.1 Maxwell Equations and Wave Equation

Starting from Maxwell's equations for fields $\propto e^{i(\mathbf{k}\cdot\mathbf{x} - \omega t)}$:

$$\mathbf{k} \times \mathbf{E}_1 = \omega \mathbf{B}_1$$

$$\mathbf{k} \times \mathbf{B}_1 = -\frac{\omega}{c^2}(\mathbf{E}_1 + \mathbf{E}_{\text{plasma}})$$

where $\mathbf{E}_{\text{plasma}}$ is the electric field from plasma currents.

For a cold plasma, the plasma current is:
$$\mathbf{j}_1 = -n_0 e \mathbf{v}_1 = -i\frac{n_0 e^2}{m_e \omega}\mathbf{E}_1 = -i\epsilon_0\omega_{pe}^2/\omega \cdot \mathbf{E}_1$$

This gives:
$$\mathbf{k} \times \mathbf{B}_1 = -\frac{\omega}{c^2}\left(1 - \frac{\omega_{pe}^2}{\omega^2}\right)\mathbf{E}_1$$

Taking $\mathbf{k} \times$ of the first equation:
$$\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) = \omega \mathbf{k} \times \mathbf{B}_1$$

Using the vector identity $\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) = \mathbf{k}(\mathbf{k}\cdot\mathbf{E}_1) - k^2\mathbf{E}_1$:

For a **transverse wave** ($\mathbf{k}\cdot\mathbf{E}_1 = 0$):

$$-k^2 \mathbf{E}_1 = -\frac{\omega^2}{c^2}\left(1 - \frac{\omega_{pe}^2}{\omega^2}\right)\mathbf{E}_1$$

This gives the **dispersion relation**:

$$\boxed{\omega^2 = \omega_{pe}^2 + k^2 c^2}$$

or equivalently:

$$\boxed{k^2 c^2 = \omega^2 - \omega_{pe}^2}$$

### 1.2 Cutoff and Refractive Index

**Cutoff**: The frequency $\omega = \omega_{pe}$ is a **cutoff**. For $\omega < \omega_{pe}$:
$$k^2 < 0 \Rightarrow k = i\kappa$$

The wave becomes **evanescent** (exponentially decaying in space):
$$\mathbf{E}_1 \propto e^{-\kappa x} e^{-i\omega t}$$

The **penetration depth** (skin depth) is:
$$\delta = \frac{1}{\kappa} = \frac{c}{\sqrt{\omega_{pe}^2 - \omega^2}}$$

For $\omega \ll \omega_{pe}$:
$$\delta \approx \frac{c}{\omega_{pe}}$$

This is why low-frequency radio waves cannot penetrate the ionosphere ($\omega_{pe} \sim 2\pi \times 10$ MHz).

**Refractive index**: Define $n = kc/\omega$:

$$\boxed{n^2 = 1 - \frac{\omega_{pe}^2}{\omega^2}}$$

For $\omega > \omega_{pe}$: $n < 1$ (phase velocity $v_\phi = c/n > c$!)

This does not violate relativity because **information** travels at the group velocity:
$$v_g = \frac{d\omega}{dk} = \frac{k c^2}{\omega} = c \sqrt{1 - \frac{\omega_{pe}^2}{\omega^2}} < c$$

### 1.3 Physical Picture: Oscillating Electrons

```
EM wave enters plasma:
E-field → accelerates electrons → oscillating current
Current → generates secondary E-field (out of phase)
Net effect: modifies wave propagation

Low ω (ω < ωpe):  Electrons respond fast enough to cancel E-field
                   → wave cannot propagate (reflected)

High ω (ω > ωpe): Electrons cannot respond fast enough
                   → wave propagates (with modified c)
```

### 1.4 Ionospheric Applications

The ionosphere has density profile $n(h)$ increasing with height:

```
Height (km)     Density (m^-3)      f_pe (MHz)
   100             10^11               0.3
   200             10^12               3
   300             10^13              10
```

An AM radio wave at 1 MHz ($< 10$ MHz) reflects at the height where $f = f_{pe}(h)$. This enables over-the-horizon communication.

FM radio at 100 MHz ($> f_{pe,\max}$) passes through the ionosphere (line-of-sight only).

## 2. Cold Magnetized Plasma: Stix Formalism

### 2.1 Dielectric Tensor

In a magnetized plasma with $\mathbf{B}_0 = B_0\hat{z}$, the plasma response is **anisotropic**. The displacement is:

$$\mathbf{D} = \epsilon_0 \overleftrightarrow{K} \cdot \mathbf{E}$$

where $\overleftrightarrow{K}$ is the **dielectric tensor**.

For a cold plasma, the equation of motion for species $s$ is:

$$-i\omega m_s \mathbf{v}_s = e_s(\mathbf{E}_1 + \mathbf{v}_s \times \mathbf{B}_0)$$

Solving for $\mathbf{v}_s$ and substituting into $\mathbf{j} = \sum_s n_0 e_s \mathbf{v}_s$, we get $\overleftrightarrow{K}$.

In the **Stix notation**, the tensor has the form:

$$\overleftrightarrow{K} = \begin{pmatrix} S & -iD & 0 \\ iD & S & 0 \\ 0 & 0 & P \end{pmatrix}$$

where:

$$\boxed{S = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}}$$

$$\boxed{D = \sum_s \frac{\omega_{cs}}{\omega} \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}}$$

$$\boxed{P = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}}$$

Here $\omega_{cs} = e_s B_0/m_s$ is the cyclotron frequency (positive for electrons, negative for ions by our convention).

**Convenient combinations**:
$$R = S + D$$
$$L = S - D$$

$R$ corresponds to **right-hand circular** polarization, $L$ to **left-hand circular**.

### 2.2 Wave Equation

The wave equation is:

$$\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) + \frac{\omega^2}{c^2}\overleftrightarrow{K}\cdot\mathbf{E}_1 = 0$$

Using $\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) = \mathbf{k}(\mathbf{k}\cdot\mathbf{E}_1) - k^2\mathbf{E}_1$ and defining the **refractive index** $\mathbf{n} = \mathbf{k}c/\omega$:

$$\mathbf{n} \times (\mathbf{n} \times \mathbf{E}_1) + \overleftrightarrow{K}\cdot\mathbf{E}_1 = 0$$

This is the **Appleton-Hartree equation** in tensor form.

The **dispersion relation** $D(n, \omega) = 0$ comes from requiring $\det[\mathbf{n} \times (\mathbf{n} \times \overleftrightarrow{I}) + \overleftrightarrow{K}] = 0$.

For propagation at angle $\theta$ to $\mathbf{B}_0$:

$$A n^4 - B n^2 + C = 0$$

where $A, B, C$ are complicated functions of $S, D, P, \theta$. We focus on two special cases.

## 3. Parallel Propagation ($\mathbf{k} \parallel \mathbf{B}_0$)

### 3.1 Circularly Polarized Modes

For $\mathbf{k} = k\hat{z}$ (along $\mathbf{B}_0$), look for solutions with $\mathbf{E}_1 = E_x\hat{x} + E_y\hat{y}$.

The wave equation gives:
$$-n^2 E_x + S E_x - iD E_y = 0$$
$$-n^2 E_y + iD E_x + S E_y = 0$$

Combining $E_\pm = E_x \pm iE_y$:

$$(S \mp D - n^2)E_\pm = 0$$

Two modes:
1. **R-wave** (right circular): $E_+ \propto e^{ikz}$, $n^2 = R = S + D$
2. **L-wave** (left circular): $E_- \propto e^{ikz}$, $n^2 = L = S - D$

For electron-ion plasma:

$$R = 1 - \frac{\omega_{pe}^2}{\omega(\omega - \omega_{ce})} - \frac{\omega_{pi}^2}{\omega(\omega + \omega_{ci})}$$

$$L = 1 - \frac{\omega_{pe}^2}{\omega(\omega + \omega_{ce})} - \frac{\omega_{pi}^2}{\omega(\omega - \omega_{ci})}$$

### 3.2 Electron Cyclotron Resonance

The R-wave has a **resonance** at $\omega = \omega_{ce}$:

$$n^2 = R \to \infty \quad \text{as } \omega \to \omega_{ce}$$

At resonance:
- Wave energy is absorbed by electrons gyrating in resonance with the wave
- Wavelength $\lambda = 2\pi/k \to 0$ (infinite $k$)
- Group velocity $v_g \to 0$ (energy deposited locally)

This is the basis for **Electron Cyclotron Resonance Heating (ECRH)**:
- Frequency: $f = n \times f_{ce}$ (typically $n=1$ or $2$)
- For ITER: $B \sim 5.3$ T $\Rightarrow f_{ce} \sim 140$ GHz
- Launched as X-mode or O-mode, absorbed at cyclotron layer

### 3.3 Ion Cyclotron Resonance

The L-wave has a resonance at $\omega = \omega_{ci}$:

$$n^2 = L \to \infty \quad \text{as } \omega \to \omega_{ci}$$

For **Ion Cyclotron Resonance Heating (ICRH)**:
- Frequency: $f \sim 30-100$ MHz for tokamaks
- Can selectively heat ion species (minority heating)
- Used in combination with mode conversion

### 3.4 Whistler Waves

In the frequency range $\omega_{ci} \ll \omega \ll \omega_{ce}$, the R-wave becomes the **whistler mode**.

Approximate dispersion (neglecting ions, $\omega_{ce} \gg \omega$):

$$n^2 = R \approx \frac{\omega_{pe}^2}{\omega(\omega_{ce} - \omega)} \approx \frac{\omega_{pe}^2}{\omega \omega_{ce}}$$

Thus:
$$\boxed{k = \frac{\omega}{c}\sqrt{\frac{\omega_{pe}^2}{\omega \omega_{ce}}}}$$

or:
$$\omega \approx \frac{k^2 c^2 \omega_{ce}}{\omega_{pe}^2}$$

Properties:
- **Highly dispersive**: $\omega \propto k^2$
- **Right-hand polarized**: $\mathbf{E}_1$ rotates in same sense as electrons
- **Phase velocity**: $v_\phi = \omega/k \propto \omega^{1/2}$ (higher frequency travels faster)
- **Group velocity**: $v_g = d\omega/dk \propto \omega$ (also dispersive)

**Discovery**: During WWII, audio signals from lightning were detected on radio receivers, exhibiting a characteristic **descending tone** (falling "whistle"):
- Lightning generates broadband EM waves
- Higher frequencies travel faster along Earth's magnetic field lines
- Arrive earlier → "whistling" sound

Whistlers propagate along magnetic field lines from one hemisphere to the other, providing information about the magnetosphere.

**Applications**:
- Magnetospheric diagnostics
- VLF communication with submarines
- Wave-particle interactions (radiation belt dynamics)

## 4. Perpendicular Propagation ($\mathbf{k} \perp \mathbf{B}_0$)

### 4.1 Ordinary and Extraordinary Modes

For $\mathbf{k} = k\hat{x}$, $\mathbf{B}_0 = B_0\hat{z}$, there are two independent modes:

**Ordinary mode (O-mode)**: $\mathbf{E}_1 = E_z\hat{z}$ (parallel to $\mathbf{B}_0$)
- No $\mathbf{v} \times \mathbf{B}$ force → behaves like unmagnetized plasma
- Dispersion: $$\boxed{n^2 = P = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}}$$
- Cutoff at $\omega_{pe}$ (or $\omega_R = \sqrt{\omega_{pe}^2 + \omega_{pi}^2}$ for exact treatment)

**Extraordinary mode (X-mode)**: $\mathbf{E}_1$ in $x$-$y$ plane (perpendicular to $\mathbf{B}_0$)
- $\mathbf{v} \times \mathbf{B}$ force couples motion
- Dispersion: $$\boxed{n^2 = \frac{RL}{S} = \frac{(S+D)(S-D)}{S}}$$

More explicitly:
$$n^2 = 1 - \frac{\omega_{pe}^2(\omega^2 - \omega_{UH}^2)}{\omega^2(\omega^2 - \omega_{UH}^2 - \omega_{ce}^2)}$$

where $\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2$ is the upper hybrid frequency.

### 4.2 X-mode Cutoffs and Resonances

The X-mode has two **cutoffs** (where $n^2 = 0$):

$$\omega_R = \frac{1}{2}\left(\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2}\right)$$ (right-hand cutoff)

$$\omega_L = \frac{1}{2}\left(-\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2}\right)$$ (left-hand cutoff)

And one **resonance** (where $n^2 \to \infty$):

$$\omega = \omega_{UH} = \sqrt{\omega_{pe}^2 + \omega_{ce}^2}$$ (upper hybrid resonance)

Near the upper hybrid layer, the wavelength $\lambda \to 0$ and the wave energy is absorbed (converted to electrostatic upper hybrid waves).

### 4.3 Mode Accessibility

The O-mode and X-mode have different accessibility to high-density plasmas:

**O-mode**:
- Cutoff at $\omega = \omega_{pe}$
- Cannot propagate if $n > n_c$ where $\omega_{pe}(n_c) = \omega$
- Limited to densities $n < n_c = \epsilon_0 m_e \omega^2/e^2$

**X-mode**:
- Cutoff at $\omega_R > \omega_{pe}$ (higher than O-mode)
- Can access higher densities: $n < n_c^{X-mode} > n_c^{O-mode}$
- Used for overdense plasma heating

For ECRH at $\omega = 2\omega_{ce}$:
- O-mode cutoff: $n_c = 4 n_{ce}$ where $n_{ce} = \epsilon_0 m_e \omega_{ce}^2/e^2$
- X-mode cutoff: higher (allows higher density access)

## 5. CMA Diagram

### 5.1 Construction

The **Clemmow-Mullaly-Allis (CMA) diagram** is a map of wave propagation regimes in parameter space.

Axes:
- Horizontal: $X = \omega_{pe}^2/\omega^2$ (density effect)
- Vertical: $Y = \omega_{ce}/\omega$ (magnetic field effect)

For each point $(X, Y)$, the diagram shows which modes (R, L, O, X) can propagate or are evanescent.

**Boundaries**:
- **Cutoffs**: curves where $n^2 = 0$ (boundary between propagation and evanescence)
- **Resonances**: curves where $n^2 \to \infty$ (absorption)

Key curves:
- O-mode cutoff: $X = 1$ (vertical line)
- R-wave cutoff: $R = 0 \Rightarrow X = 1 - Y$
- L-wave cutoff: $L = 0 \Rightarrow X = 1 + Y$
- Upper hybrid: $X = 1 - Y^2$ (for perpendicular propagation)

### 5.2 Regions and Mode Characteristics

```
Y (ω_ce/ω)
   ↑
   |        R, L, O propagate
   |     /
   | R  /   O
   | cutoff
   |  /
   |/______________→ X (ω_pe²/ω²)
    1
```

Different regions:
- **Region I** ($X < 1 - Y$): All modes propagate
- **Region II** ($1 - Y < X < 1$): R-wave evanescent, L and O propagate
- **Region III** ($X > 1$): O-mode evanescent

The full diagram (including ion effects and perpendicular propagation) has $\sim 10$ distinct regions.

### 5.3 Applications

The CMA diagram is used to:
- Design heating and current drive systems (choose appropriate mode)
- Plan diagnostic systems (reflectometry, interferometry)
- Understand ionospheric propagation
- Analyze whistler propagation in magnetosphere

For example, in fusion plasmas:
- High density, high $B$ → $(X, Y)$ in region where X-mode needed
- Low density edge → O-mode accessible

## 6. Faraday Rotation

### 6.1 Theory

When a **linearly polarized** wave propagates through a magnetized plasma, the plane of polarization **rotates**. This is **Faraday rotation**.

Physical origin:
- Linear polarization = superposition of R-wave and L-wave with equal amplitude
- R and L have different refractive indices: $n_R \neq n_L$
- Phase difference accumulates: $\Delta\phi = (k_R - k_L) L$
- Polarization plane rotates by angle $\theta = \Delta\phi/2$

For $\omega \gg \omega_{pe}, \omega_{ce}$:

$$n_R - n_L \approx \frac{\omega_{pe}^2 \omega_{ce}}{\omega^3}$$

The rotation angle over distance $L$ is:

$$\boxed{\theta = \frac{\omega_{pe}^2 \omega_{ce}}{2c\omega^2} L = \frac{e^3}{2\epsilon_0 m_e^2 c \omega^2} \int_0^L n_e B_\parallel \, dl}$$

where $B_\parallel$ is the component of $\mathbf{B}$ along the ray path.

The **rotation measure** is:

$$RM = \frac{e^3}{2\pi \epsilon_0 m_e^2 c} \int n_e B_\parallel \, dl$$

In practical units:
$$RM \approx 2.63 \times 10^{-13} \int n_e(\text{cm}^{-3}) B_\parallel(\text{G}) \, dl(\text{pc}) \quad (\text{rad/m}^2)$$

### 6.2 Astrophysical Applications

Faraday rotation is used to measure magnetic fields in:

**Pulsars**:
- Measure $RM$ from multifrequency polarization observations
- Infer $\int n_e B_\parallel dl$ along line of sight
- Map Galactic magnetic field structure

**Active Galactic Nuclei (AGN)**:
- Jets have tangled magnetic fields
- Faraday rotation gives field strength and structure

**Intracluster Medium**:
- Galaxy clusters: $n_e \sim 10^{-3}$ cm$^{-3}$, $L \sim$ Mpc
- Measure $B \sim \mu$G from rotation measures

**Tokamak diagnostics**:
- Polarimetry measures $\int n_e B_\parallel dl$
- Combined with interferometry ($\int n_e dl$), can infer $B$ profile

### 6.3 Dispersion and Depolarization

At lower frequencies, the rotation angle is larger: $\theta \propto \omega^{-2}$.

For broadband emission with $\Delta\omega$, different frequencies rotate by different amounts, causing **depolarization**:

$$\Delta\theta \approx 2\theta \frac{\Delta\omega}{\omega}$$

If $\Delta\theta \gtrsim \pi/2$, the net polarization is scrambled.

This sets a **depolarization frequency**:
$$\omega_{\text{depol}} \sim \left(\frac{e^3 n_e B_\parallel L}{\epsilon_0 m_e^2 c}\right)^{1/2}$$

## 7. Python Implementation

### 7.1 Dispersion in Unmagnetized Plasma

```python
import numpy as np
import matplotlib.pyplot as plt

def unmagnetized_dispersion(k, omega_pe):
    """
    EM wave dispersion in unmagnetized plasma: ω² = ω_pe² + k²c².
    """
    c = 3e8  # m/s
    omega = np.sqrt(omega_pe**2 + k**2 * c**2)
    return omega

# Parameters
n = 1e19  # m^-3
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m
m_e = 9.109e-31  # kg
c = 3e8  # m/s

omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
f_pe = omega_pe / (2 * np.pi)

print(f"Plasma frequency: f_pe = {f_pe / 1e9:.2f} GHz")
print(f"Cutoff wavelength: λ_c = {c / f_pe:.2f} m")

# Wavenumber
k = np.linspace(0, 5, 1000) * omega_pe / c

# Dispersion
omega = unmagnetized_dispersion(k, omega_pe)

# Refractive index
n_refr = k * c / omega

# Group velocity
v_g = c**2 * k / omega

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Dispersion relation
ax1.plot(k * c / omega_pe, omega / omega_pe, 'b-', linewidth=2, label='Plasma')
ax1.plot(k * c / omega_pe, k * c / omega_pe, 'k--', linewidth=1.5,
         label='Vacuum ($\\omega = kc$)')
ax1.axhline(1, color='r', linestyle=':', linewidth=1.5, label='Cutoff ($\\omega_{pe}$)')
ax1.set_xlabel('$kc/\\omega_{pe}$', fontsize=13)
ax1.set_ylabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax1.set_title('EM Wave Dispersion in Plasma', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 6])

# Refractive index
ax2.plot(omega / omega_pe, n_refr, 'b-', linewidth=2)
ax2.axhline(1, color='k', linestyle='--', alpha=0.5, label='Vacuum')
ax2.axvline(1, color='r', linestyle=':', linewidth=1.5, label='Cutoff')
ax2.set_xlabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax2.set_ylabel('Refractive Index $n$', fontsize=13)
ax2.set_title('Refractive Index: $n = \\sqrt{1 - \\omega_{pe}^2/\\omega^2}$', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1, 3])
ax2.set_ylim([0, 1.2])

# Phase velocity
v_phase = omega / k
ax3.plot(omega / omega_pe, v_phase / c, 'b-', linewidth=2, label='$v_\\phi$')
ax3.axhline(1, color='k', linestyle='--', alpha=0.5, label='$c$')
ax3.set_xlabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax3.set_ylabel('$v_\\phi / c$', fontsize=13)
ax3.set_title('Phase Velocity (superluminal!)', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([1, 3])
ax3.set_ylim([0.5, 3])

# Group velocity
ax4.plot(omega / omega_pe, v_g / c, 'r-', linewidth=2, label='$v_g$')
ax4.axhline(1, color='k', linestyle='--', alpha=0.5, label='$c$')
ax4.set_xlabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax4.set_ylabel('$v_g / c$', fontsize=13)
ax4.set_title('Group Velocity (subluminal)', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([1, 3])
ax4.set_ylim([0, 1.2])

plt.tight_layout()
plt.savefig('unmagnetized_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 7.2 Stix Parameters and Mode Dispersion

```python
def stix_parameters(omega, n, B, Z=1, A=1):
    """
    Compute Stix parameters S, D, P for cold magnetized plasma.

    Parameters:
    -----------
    omega : float or array
        Wave frequency (rad/s)
    n : float
        Density (m^-3)
    B : float
        Magnetic field (T)
    Z : int
        Ion charge
    A : float
        Ion mass number

    Returns:
    --------
    dict with S, D, P, R, L
    """
    m_i = A * 1.673e-27

    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    omega_pi = np.sqrt(n * Z**2 * e**2 / (epsilon_0 * m_i))
    omega_ce = e * B / m_e
    omega_ci = Z * e * B / m_i

    # S, D, P are the three independent components of the cold-plasma dielectric tensor.
    # S (sum) is the diagonal element coupling x and y components; the resonance
    # denominator ω² - ω_cs² diverges at cyclotron frequency, reflecting the
    # singular response of gyrating particles near their natural frequency.
    S = 1 - omega_pe**2 / (omega**2 - omega_ce**2) - \
        omega_pi**2 / (omega**2 - omega_ci**2)

    # D (difference) is the off-diagonal gyrotropic element that breaks left/right
    # symmetry; it vanishes without a magnetic field (ω_cs → 0) and causes Faraday rotation.
    D = omega_ce * omega_pe**2 / (omega * (omega**2 - omega_ce**2)) + \
        omega_ci * omega_pi**2 / (omega * (omega**2 - omega_ci**2))

    # P (plasma) governs response along B where cyclotron motion is irrelevant;
    # it reduces to the unmagnetised dielectric (1 - ω_pe²/ω²) for electrons alone.
    P = 1 - omega_pe**2 / omega**2 - omega_pi**2 / omega**2

    # R and L are the refractive indices squared for right and left circular polarisations
    # along B; they factor from the determinant of the wave equation for parallel propagation.
    R = S + D
    L = S - D

    return {
        'S': S, 'D': D, 'P': P, 'R': R, 'L': L,
        'omega_pe': omega_pe, 'omega_pi': omega_pi,
        'omega_ce': omega_ce, 'omega_ci': omega_ci
    }

# Parameters
B = 2.0  # T
n = 5e19  # m^-3

params = stix_parameters(1, n, B, Z=1, A=2)  # Dummy ω for frequencies

f_ce = params['omega_ce'] / (2 * np.pi)
f_ci = params['omega_ci'] / (2 * np.pi)
f_pe = params['omega_pe'] / (2 * np.pi)

print(f"Electron cyclotron frequency: f_ce = {f_ce / 1e9:.2f} GHz")
print(f"Ion cyclotron frequency: f_ci = {f_ci / 1e6:.2f} MHz")
print(f"Electron plasma frequency: f_pe = {f_pe / 1e9:.2f} GHz")

# Scanning 0.1-100 GHz covers all physically relevant resonances (f_ci ~ MHz is
# far below, f_ce ~ 56 GHz sits mid-range, f_pe ~ GHz depending on density).
# The 5000-point resolution resolves the sharp sign changes in S and D near f_ce
# that mark the transition between propagating and evanescent regimes.
f = np.linspace(0.1, 100, 5000) * 1e9  # Hz
omega = 2 * np.pi * f

params = stix_parameters(omega, n, B, Z=1, A=2)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Stix parameters
ax1.plot(f / 1e9, params['S'], 'b-', linewidth=2, label='S')
ax1.plot(f / 1e9, params['P'], 'r-', linewidth=2, label='P')
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(f_ce / 1e9, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Frequency (GHz)', fontsize=13)
ax1.set_ylabel('Stix Parameter', fontsize=13)
ax1.set_title('S and P Parameters', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])
ax1.set_ylim([-5, 5])

ax2.plot(f / 1e9, params['D'], 'g-', linewidth=2)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(f_ce / 1e9, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Frequency (GHz)', fontsize=13)
ax2.set_ylabel('D Parameter', fontsize=13)
ax2.set_title('D Parameter', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 100])

# R and L (refractive index squared for parallel propagation)
R_positive = np.where(params['R'] > 0, params['R'], np.nan)
L_positive = np.where(params['L'] > 0, params['L'], np.nan)

ax3.semilogy(f / 1e9, R_positive, 'b-', linewidth=2, label='R (R-wave)')
ax3.semilogy(f / 1e9, L_positive, 'r-', linewidth=2, label='L (L-wave)')
ax3.axvline(f_ce / 1e9, color='b', linestyle=':', alpha=0.5,
            label='$f_{ce}$ (R resonance)')
ax3.axvline(f_ci / 1e6 / 1e3, color='r', linestyle=':', alpha=0.5,
            label='$f_{ci}$ (L resonance)')
ax3.set_xlabel('Frequency (GHz)', fontsize=13)
ax3.set_ylabel('$n^2$ (R, L modes)', fontsize=13)
ax3.set_title('Parallel Propagation: $n^2 = R, L$', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, which='both', alpha=0.3)
ax3.set_xlim([0, 100])
ax3.set_ylim([0.01, 1000])

# O-mode and X-mode (perpendicular)
n2_O = params['P']
n2_X = params['R'] * params['L'] / params['S']

O_positive = np.where(n2_O > 0, n2_O, np.nan)
X_positive = np.where(n2_X > 0, n2_X, np.nan)

ax4.semilogy(f / 1e9, O_positive, 'b-', linewidth=2, label='O-mode')
ax4.semilogy(f / 1e9, X_positive, 'r-', linewidth=2, label='X-mode')
ax4.axvline(f_pe / 1e9, color='b', linestyle=':', alpha=0.5,
            label='$f_{pe}$ (O cutoff)')
ax4.set_xlabel('Frequency (GHz)', fontsize=13)
ax4.set_ylabel('$n^2$ (O, X modes)', fontsize=13)
ax4.set_title('Perpendicular Propagation: O and X modes', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, which='both', alpha=0.3)
ax4.set_xlim([0, 100])
ax4.set_ylim([0.01, 100])

plt.tight_layout()
plt.savefig('stix_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 7.3 CMA Diagram

```python
def cma_diagram():
    """
    Generate CMA (Clemmow-Mullaly-Allis) diagram.
    """
    X = np.linspace(0, 3, 500)
    Y = np.linspace(0, 2, 500)
    XX, YY = np.meshgrid(X, Y)

    # Cutoff and resonance curves
    # O-mode cutoff: X = 1
    # R-wave cutoff: R = 0 → X = 1 - Y (for single species, approx)
    # L-wave cutoff: L = 0 → X = 1 + Y
    # Upper hybrid resonance: X = 1 - Y²

    fig, ax = plt.subplots(figsize=(10, 8))

    # Define regions based on cutoffs
    # (Simplified for electrons only)

    # O-mode propagates when X < 1
    O_propagate = XX < 1

    # R-wave propagates when R > 0 (approx X < 1 - Y for Y << 1)
    # More accurately, solve R = 0
    R_cutoff_Y = np.linspace(0, 1.5, 100)
    R_cutoff_X = 1 - R_cutoff_Y

    # L-wave cutoff: X = 1 + Y
    L_cutoff_Y = np.linspace(0, 1.5, 100)
    L_cutoff_X = 1 + L_cutoff_Y

    # Upper hybrid resonance: X = 1 - Y²
    UH_res_Y = np.linspace(0, 1.5, 100)
    UH_res_X = 1 - UH_res_Y**2

    # Each boundary is where a wave transitions from propagating to evanescent (cutoffs)
    # or from finite to infinite wavenumber (resonances). Plotting them in (X, Y) space
    # is powerful because a plasma's path through the diagram as density or B changes
    # shows which modes it passes through — essential for antenna/launching design.
    ax.plot([1, 1], [0, 2], 'b-', linewidth=2, label='O-mode cutoff ($X=1$)')
    ax.plot(R_cutoff_X, R_cutoff_Y, 'r-', linewidth=2,
            label='R-wave cutoff ($X=1-Y$)')
    ax.plot(L_cutoff_X, L_cutoff_Y, 'g-', linewidth=2,
            label='L-wave cutoff ($X=1+Y$)')
    # The upper hybrid resonance sits inside the O-mode propagating region (X < 1),
    # which is why X-mode can reach it from the low-density side while O-mode cannot.
    ax.plot(UH_res_X, UH_res_Y, 'm--', linewidth=2,
            label='Upper hybrid res ($X=1-Y^2$)')

    # Add shaded regions
    ax.fill_betweenx([0, 2], 0, 1, alpha=0.1, color='blue', label='O propagates')
    ax.fill_between(R_cutoff_X, 0, R_cutoff_Y, alpha=0.1, color='red',
                     label='R evanescent')

    ax.set_xlabel('$X = \\omega_{pe}^2 / \\omega^2$', fontsize=14)
    ax.set_ylabel('$Y = \\omega_{ce} / \\omega$', fontsize=14)
    ax.set_title('CMA Diagram (Simplified, Electrons Only)', fontsize=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 2])

    # Annotate regions
    ax.text(0.5, 0.5, 'All modes\npropagate', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(1.5, 0.3, 'O-mode\nevanescent', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.3, 1.5, 'R-wave\nevanescent', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig('cma_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()

cma_diagram()
```

### 7.4 Whistler Wave Dispersion

```python
def whistler_dispersion(k, omega_pe, omega_ce):
    """
    Whistler wave dispersion: ω ≈ k²c²ω_ce/ω_pe².
    """
    c = 3e8
    # The ω ∝ k² scaling (anomalous dispersion) comes from the R-wave expression
    # n² = R ≈ ω_pe² / (ω ω_ce) in the whistler limit ω_ci << ω << ω_ce;
    # rearranging kc/ω = n gives ω = k²c²ω_ce / ω_pe² — higher frequencies
    # have larger phase velocity, so they travel faster and arrive first (descending tone).
    omega = k**2 * c**2 * omega_ce / omega_pe**2
    # Ion contribution to R is neglected here because at whistler frequencies
    # ω >> ω_ci, so ions are too slow to respond to the wave and χ_i ≈ 0.
    # The cap at 0.5 ω_ce enforces the validity of the whistler approximation;
    # near ω_ce the full R expression must be used as the resonance divergence matters.
    omega = np.minimum(omega, 0.5 * omega_ce)  # Limit to whistler range
    return omega

# Magnetospheric parameters are chosen to place f_pe and f_ce both in the kHz range:
# at these low densities (n ~ 10^6 m^-3) f_pe ~ 9 kHz, while B ~ 50 μT gives
# f_ce ~ 1.4 MHz — satisfying ω_ci << ω << ω_ce for audio-frequency whistlers.
n = 1e6  # m^-3 (magnetosphere)
B = 5e-5  # T (Earth's magnetic field at magnetosphere)

omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
omega_ce = e * B / m_e

f_pe = omega_pe / (2 * np.pi)
f_ce = omega_ce / (2 * np.pi)

print(f"Magnetospheric parameters:")
print(f"  f_pe = {f_pe / 1e3:.1f} kHz")
print(f"  f_ce = {f_ce / 1e3:.1f} kHz")

# Wavenumber
k = np.linspace(1e-7, 1e-5, 500)  # m^-1

omega = whistler_dispersion(k, omega_pe, omega_ce)
f = omega / (2 * np.pi)

# Phase and group velocity
v_phase = omega / k
v_group = 2 * omega / k

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Dispersion
ax1.plot(k * 1e6, f / 1e3, 'b-', linewidth=2)
ax1.axhline(f_ce / (2e3), color='r', linestyle='--',
            label='$f_{ce}/2$ (upper limit)')
ax1.set_xlabel('Wavenumber $k$ ($\\mu$m$^{-1}$)', fontsize=13)
ax1.set_ylabel('Frequency (kHz)', fontsize=13)
ax1.set_title('Whistler Dispersion: $\\omega \\propto k^2$', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Phase velocity
ax2.plot(f / 1e3, v_phase / c, 'b-', linewidth=2)
ax2.set_xlabel('Frequency (kHz)', fontsize=13)
ax2.set_ylabel('$v_\\phi / c$', fontsize=13)
ax2.set_title('Phase Velocity (higher f travels faster)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Group velocity
ax3.plot(f / 1e3, v_group / c, 'r-', linewidth=2)
ax3.set_xlabel('Frequency (kHz)', fontsize=13)
ax3.set_ylabel('$v_g / c$', fontsize=13)
ax3.set_title('Group Velocity: $v_g = 2v_\\phi$', fontsize=14)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('whistler_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()

# Simulate whistler arrival times
print("\nWhistler arrival times from lightning (L = 10,000 km):")
L = 1e7  # m
for f_khz in [1, 2, 5, 10]:
    f_val = f_khz * 1e3
    idx = np.argmin(np.abs(f / 1e3 - f_khz))
    t_arrival = L / v_group[idx]
    print(f"  {f_khz} kHz: {t_arrival:.2f} s")
```

## Summary

Electromagnetic waves in plasmas exhibit rich behavior governed by the interplay of plasma frequency, cyclotron frequency, and wave frequency:

**Unmagnetized plasma**:
- Dispersion: $\omega^2 = \omega_{pe}^2 + k^2c^2$
- Cutoff at $\omega_{pe}$: waves with $\omega < \omega_{pe}$ cannot propagate
- Refractive index $n < 1$: phase velocity $> c$, but group velocity $< c$
- Applications: ionospheric reflection, plasma density diagnostics

**Magnetized plasma (Stix formalism)**:
- Dielectric tensor with components $S$, $D$, $P$
- Anisotropic response leads to multiple wave modes
- Combinations $R = S + D$, $L = S - D$ for circular polarizations

**Parallel propagation**:
- R-wave (right circular): resonance at $\omega = \omega_{ce}$ → ECRH
- L-wave (left circular): resonance at $\omega = \omega_{ci}$ → ICRH
- Whistler waves ($\omega_{ci} \ll \omega \ll \omega_{ce}$): highly dispersive, $\omega \propto k^2$
- Applications: magnetospheric physics, VLF communication

**Perpendicular propagation**:
- O-mode: $\mathbf{E} \parallel \mathbf{B}$, cutoff at $\omega_{pe}$
- X-mode: $\mathbf{E} \perp \mathbf{B}$, cutoffs at $\omega_R, \omega_L$, resonance at $\omega_{UH}$
- Mode accessibility: X-mode reaches higher densities than O-mode

**CMA diagram**:
- Maps propagation regions in $(X, Y)$ space where $X = \omega_{pe}^2/\omega^2$, $Y = \omega_{ce}/\omega$
- Shows cutoffs, resonances, and mode boundaries
- Essential tool for wave heating and diagnostic design

**Faraday rotation**:
- Rotation angle: $\theta \propto \int n_e B_\parallel dl \cdot \omega^{-2}$
- Measures magnetic fields in astrophysical plasmas
- Used in fusion polarimetry for current profile measurements

These wave phenomena enable:
- Plasma heating (ECRH, ICRH, LHCD)
- Current drive (non-inductive operation)
- Diagnostics (interferometry, reflectometry, polarimetry)
- Communications (ionospheric propagation)
- Astrophysical observations (pulsar RM, AGN jets)

## Practice Problems

### Problem 1: Ionospheric Reflection
The ionosphere has a peak density of $n_e = 10^{12}$ m$^{-3}$.

(a) Calculate the plasma frequency $f_{pe}$ at the peak.

(b) An AM radio station broadcasts at 1 MHz. Will the signal reflect off the ionosphere or pass through?

(c) At what minimum frequency will signals pass through the ionosphere?

(d) The ionosphere density varies with time of day. During the day, $n_e$ increases by a factor of 10. How does this affect AM radio propagation?

### Problem 2: ECRH System Design
A tokamak has $B_0 = 3.5$ T on axis. You are designing an ECRH system for central heating.

(a) Calculate the electron cyclotron frequency $f_{ce}$ and wavelength in vacuum.

(b) The system will use 2nd harmonic ($\omega = 2\omega_{ce}$). What frequency should the gyrotron produce?

(c) For a density $n = 5 \times 10^{19}$ m$^{-3}$, calculate the O-mode cutoff density at $2f_{ce}$. Can the wave reach the center?

(d) If O-mode cannot reach the center, explain how X-mode or electron Bernstein wave (EBW) could be used instead.

### Problem 3: Whistler Propagation
A lightning stroke at the magnetic equator generates a broadband signal that propagates along field lines to the opposite hemisphere.

(a) For $B = 5 \times 10^{-5}$ T, $n = 10^7$ m$^{-3}$, calculate $f_{ce}$ and $f_{pe}$.

(b) Show that the frequency range $f_{ci} \ll f \ll f_{ce}$ is satisfied for $f \sim 1-10$ kHz (assume hydrogen ions).

(c) Calculate the group velocity at $f = 5$ kHz.

(d) If the path length is $L = 10,000$ km, how long does it take for the 5 kHz component to arrive? How about the 1 kHz component? Explain the "whistling" sound.

### Problem 4: CMA Diagram Regions
For a plasma with $f_{pe} = 50$ GHz and $f_{ce} = 70$ GHz, determine which modes can propagate at the following frequencies:

(a) $f = 40$ GHz (X-band radar)

(b) $f = 75$ GHz (W-band)

(c) $f = 120$ GHz (ECRH at 2nd harmonic)

(d) For each frequency, calculate $X$ and $Y$, and identify the accessible modes (R, L, O, X).

### Problem 5: Faraday Rotation Measurement
A polarized wave at $\lambda = 6$ cm (5 GHz) propagates through a plasma slab with $n_e = 10^{18}$ m$^{-3}$, $B_\parallel = 0.1$ T, $L = 1$ m.

(a) Calculate the Faraday rotation angle $\theta$.

(b) If measurements are made at two wavelengths ($\lambda_1 = 6$ cm, $\lambda_2 = 3$ cm), what is the difference in rotation angles $\Delta\theta$?

(c) From $\Delta\theta$, derive the rotation measure $RM = \theta/\lambda^2$.

(d) If only $RM$ is measured (not $n_e$ and $B$ separately), what additional measurement would you need to determine $B_\parallel$?

---

**Previous**: [10. Electrostatic Waves](./10_Electrostatic_Waves.md)
**Next**: [12. Wave Heating and Instabilities](./12_Wave_Heating_and_Instabilities.md)

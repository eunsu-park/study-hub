# 10. Electromagnetic Waves in Matter

[← Previous: 09. EM Waves in Vacuum](09_EM_Waves_Vacuum.md) | [Next: 11. Reflection and Refraction →](11_Reflection_and_Refraction.md)

## Learning Objectives

1. Derive the wave equation in linear dielectric and conducting media from Maxwell's equations
2. Understand complex permittivity and its connection to absorption and dispersion
3. Apply the Drude model to describe electromagnetic response of metals
4. Calculate skin depth and understand its practical significance
5. Work with the complex refractive index and absorption coefficient
6. Derive and interpret the Kramers-Kronig relations linking real and imaginary parts of the response
7. Simulate wave propagation in dispersive media using Python

When an electromagnetic wave enters a material, it no longer travels unimpeded at the speed of light. The wave's electric field drives the bound charges in dielectrics and the free charges in conductors, creating polarization and conduction currents that feed back into Maxwell's equations. The result is a rich set of phenomena: the wave may slow down (refraction), lose energy (absorption), or travel at different speeds depending on its frequency (dispersion). Understanding these effects is essential for designing optical fibers, radar absorbers, and virtually every photonic device.

> **Analogy**: Think of a crowd of people (the medium) through which a wave (like a "stadium wave") propagates. In vacuum, the wave moves at maximum speed with no resistance. In a dense crowd (high permittivity), people respond slowly — the wave slows down. If people are clumsy and bump into each other (conductivity), some energy is lost to friction with each oscillation. The frequency dependence of these crowd dynamics is dispersion.

---

## 1. Maxwell's Equations in Linear Media

### 1.1 Constitutive Relations

In a linear, isotropic, homogeneous (LIH) medium, the material response is captured by:

$$\mathbf{D} = \epsilon \mathbf{E}, \quad \mathbf{B} = \mu \mathbf{H}, \quad \mathbf{J}_f = \sigma \mathbf{E}$$

where $\epsilon$ is the permittivity, $\mu$ the permeability, and $\sigma$ the conductivity. We write:

$$\epsilon = \epsilon_0 \epsilon_r, \quad \mu = \mu_0 \mu_r$$

with $\epsilon_r$ and $\mu_r$ being the relative permittivity and permeability.

### 1.2 Wave Equation in a Dielectric

Starting from the source-free Maxwell equations in matter:

$$\nabla \times \mathbf{E} = -\mu \frac{\partial \mathbf{H}}{\partial t}, \quad \nabla \times \mathbf{H} = \epsilon \frac{\partial \mathbf{E}}{\partial t} + \sigma \mathbf{E}$$

Taking the curl of the first equation and substituting the second:

$$\nabla^2 \mathbf{E} = \mu \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2} + \mu \sigma \frac{\partial \mathbf{E}}{\partial t}$$

This is the **damped wave equation**. The first term on the right gives wave propagation (with speed $v = 1/\sqrt{\mu\epsilon}$), while the second term represents ohmic losses.

For a lossless dielectric ($\sigma = 0$):

$$\nabla^2 \mathbf{E} = \mu \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2}$$

The wave speed in the medium is:

$$v = \frac{1}{\sqrt{\mu \epsilon}} = \frac{c}{\sqrt{\mu_r \epsilon_r}} = \frac{c}{n}$$

where $n = \sqrt{\mu_r \epsilon_r}$ is the **refractive index**.

### 1.3 Plane Wave Ansatz in a Conducting Medium

Assume a plane wave solution $\mathbf{E} = \mathbf{E}_0 \, e^{i(\tilde{k}z - \omega t)}$ where $\tilde{k}$ is a complex wave number. Substituting into the damped wave equation:

$$-\tilde{k}^2 = -\mu \epsilon \omega^2 + i \mu \sigma \omega$$

$$\tilde{k}^2 = \mu \epsilon \omega^2 - i \mu \sigma \omega = \mu \omega^2 \left(\epsilon - i\frac{\sigma}{\omega}\right)$$

This motivates defining a **complex permittivity**.

---

## 2. Complex Permittivity

### 2.1 Definition

We absorb the conductivity into a single complex quantity:

$$\tilde{\epsilon}(\omega) = \epsilon(\omega) + i\frac{\sigma(\omega)}{\omega} = \epsilon'(\omega) + i\epsilon''(\omega)$$

where $\epsilon' = \operatorname{Re}(\tilde{\epsilon})$ and $\epsilon'' = \operatorname{Im}(\tilde{\epsilon})$. Now the wave equation takes the simple form:

$$\tilde{k}^2 = \mu \tilde{\epsilon} \, \omega^2$$

The real part $\epsilon'$ determines the phase velocity; the imaginary part $\epsilon''$ determines the absorption.

### 2.2 Sign Convention

Different textbooks use different sign conventions for the time dependence ($e^{-i\omega t}$ vs $e^{+i\omega t}$). With our convention $e^{i(\tilde{k}z - \omega t)}$:

- $\epsilon'' > 0$ corresponds to **loss** (absorption)
- A positive imaginary part of $\tilde{k}$ gives exponential decay along $z$

This is the physicists' convention (Griffiths, Jackson). Engineers often use $e^{j(\omega t - kz)}$, which flips the sign of $\epsilon''$.

### 2.3 Lorentz Oscillator Model

The simplest microscopic model for dielectric response treats bound electrons as damped harmonic oscillators driven by the wave's electric field:

$$m\ddot{x} + m\gamma\dot{x} + m\omega_0^2 x = -eE_0 e^{-i\omega t}$$

where $\omega_0$ is the natural frequency and $\gamma$ is the damping rate. The steady-state solution gives the dipole moment $p = -ex$, and summing over $N$ oscillators per unit volume:

$$\epsilon_r(\omega) = 1 + \frac{Ne^2}{m\epsilon_0} \frac{1}{\omega_0^2 - \omega^2 - i\gamma\omega}$$

For multiple resonances at frequencies $\omega_j$ with oscillator strengths $f_j$:

$$\epsilon_r(\omega) = 1 + \frac{Ne^2}{m\epsilon_0} \sum_j \frac{f_j}{\omega_j^2 - \omega^2 - i\gamma_j\omega}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def lorentz_permittivity(omega, omega_0, gamma, omega_p):
    """
    Compute complex relative permittivity using the Lorentz oscillator model.

    Parameters:
        omega   : angular frequency array (rad/s)
        omega_0 : resonance frequency (rad/s)
        gamma   : damping rate (rad/s)
        omega_p : plasma frequency = sqrt(Ne^2 / m eps_0) (rad/s)

    Returns:
        Complex epsilon_r(omega)

    Why this model matters: it captures the essential physics of how
    bound electrons respond to an oscillating E-field, producing both
    dispersion (varying n) and absorption near resonance.
    """
    return 1.0 + omega_p**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

# Parameters for a typical glass-like material
omega_0 = 6e15       # UV resonance ~6 PHz
gamma = 1e14         # damping ~0.1 PHz
omega_p = 4e15       # plasma frequency

omega = np.linspace(1e14, 1.2e16, 5000)
eps = lorentz_permittivity(omega, omega_0, gamma, omega_p)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Real part: determines refractive index
axes[0].plot(omega / 1e15, eps.real, 'b-', linewidth=1.5)
axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=omega_0 / 1e15, color='r', linestyle=':', label=f'$\\omega_0$')
axes[0].set_ylabel("$\\epsilon'$ (real part)")
axes[0].set_title("Lorentz Oscillator Model: Complex Permittivity")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Imaginary part: determines absorption
axes[1].plot(omega / 1e15, eps.imag, 'r-', linewidth=1.5)
axes[1].axvline(x=omega_0 / 1e15, color='r', linestyle=':', label=f'$\\omega_0$')
axes[1].set_xlabel("Frequency (PHz)")
axes[1].set_ylabel("$\\epsilon''$ (imaginary part)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lorentz_permittivity.png", dpi=150)
plt.show()
```

Near resonance ($\omega \approx \omega_0$), the imaginary part peaks (maximum absorption) and the real part swings from above 1 to below 1 — this is **anomalous dispersion**, where the refractive index decreases with increasing frequency.

---

## 3. Dispersion

### 3.1 Normal vs. Anomalous Dispersion

Away from resonances, $\epsilon'$ increases with frequency — this is **normal dispersion**. A prism separates colors because blue light (higher $\omega$) has a larger refractive index and bends more.

Near a resonance, $\epsilon'$ **decreases** with frequency — **anomalous dispersion**. In this narrow band, the group velocity can exceed $c$ or even become negative, but the signal velocity remains below $c$ (no faster-than-light information transfer).

### 3.2 Phase and Group Velocity

The complex wave number is $\tilde{k} = k + i\kappa$, so:

$$\mathbf{E} = \mathbf{E}_0 \, e^{-\kappa z} e^{i(kz - \omega t)}$$

- **Phase velocity**: $v_p = \omega / k$ (speed of constant-phase surfaces)
- **Group velocity**: $v_g = d\omega / dk$ (speed of a wave packet's envelope)

In a dispersive medium, $v_p \neq v_g$. For the Lorentz model far from resonance:

$$v_g = \frac{c}{n + \omega \, dn/d\omega}$$

### 3.3 Group Velocity Dispersion (GVD)

The second derivative $d^2k/d\omega^2$ determines how a pulse broadens as it propagates. This is critical in fiber optics:

$$\text{GVD} = \frac{d^2 k}{d\omega^2} = \frac{1}{c}\left(2\frac{dn}{d\omega} + \omega\frac{d^2n}{d\omega^2}\right)$$

- GVD > 0: normal dispersion (longer wavelengths travel faster)
- GVD < 0: anomalous dispersion (shorter wavelengths travel faster)

```python
def compute_group_velocity(omega, eps_r, mu_r=1.0):
    """
    Compute phase velocity, group velocity, and GVD from complex permittivity.

    Why we compute these numerically: analytical expressions exist for
    simple models, but numerical differentiation generalizes to arbitrary
    epsilon(omega) from experiments or multi-resonance models.
    """
    c = 3e8  # speed of light (m/s)

    # Complex refractive index
    n_complex = np.sqrt(eps_r * mu_r)
    n = n_complex.real  # real refractive index
    kappa = n_complex.imag  # extinction coefficient

    # Phase velocity
    v_phase = c / n

    # Group velocity via numerical derivative dn/domega
    dn_domega = np.gradient(n, omega)
    v_group = c / (n + omega * dn_domega)

    # GVD via second derivative of k = n*omega/c
    k_real = n * omega / c
    d2k_domega2 = np.gradient(np.gradient(k_real, omega), omega)

    return v_phase, v_group, d2k_domega2

# Using the Lorentz model from above (far from resonance region)
mask = (omega < 4e15) | (omega > 8e15)  # avoid resonance region
omega_disp = omega[mask]
eps_disp = eps[mask]

v_ph, v_gr, gvd = compute_group_velocity(omega_disp, eps_disp)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(omega_disp / 1e15, v_ph / 3e8, 'b-', label='$v_p / c$')
axes[0].axhline(y=1, color='gray', linestyle='--')
axes[0].set_ylabel('$v_p / c$')
axes[0].set_title('Dispersion Properties')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega_disp / 1e15, v_gr / 3e8, 'r-', label='$v_g / c$')
axes[1].axhline(y=1, color='gray', linestyle='--')
axes[1].set_ylabel('$v_g / c$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(omega_disp / 1e15, gvd * 1e30, 'g-', label='GVD')
axes[2].axhline(y=0, color='gray', linestyle='--')
axes[2].set_xlabel('Frequency (PHz)')
axes[2].set_ylabel('GVD (fs$^2$/mm)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dispersion_properties.png", dpi=150)
plt.show()
```

---

## 4. The Drude Model for Metals

### 4.1 Free Electron Response

In metals, conduction electrons are essentially free ($\omega_0 = 0$). The equation of motion simplifies to:

$$m\ddot{x} + m\gamma\dot{x} = -eE_0 e^{-i\omega t}$$

This gives the **Drude permittivity**:

$$\epsilon_r(\omega) = 1 - \frac{\omega_p^2}{\omega^2 + i\gamma\omega}$$

where $\omega_p = \sqrt{Ne^2 / m\epsilon_0}$ is the **plasma frequency** — the natural oscillation frequency of the electron gas.

### 4.2 Key Features

Separating real and imaginary parts:

$$\epsilon'(\omega) = 1 - \frac{\omega_p^2}{\omega^2 + \gamma^2}, \quad \epsilon''(\omega) = \frac{\omega_p^2 \gamma}{\omega(\omega^2 + \gamma^2)}$$

**Low frequency** ($\omega \ll \gamma$): The imaginary part dominates, and the material behaves as a conductor. The Drude conductivity is:

$$\sigma(\omega) = \frac{Ne^2}{m(\gamma - i\omega)} \quad \Rightarrow \quad \sigma_0 = \frac{Ne^2}{m\gamma} \quad (\text{DC conductivity})$$

**High frequency** ($\omega \gg \gamma$): The damping is negligible, and:

$$\epsilon_r(\omega) \approx 1 - \frac{\omega_p^2}{\omega^2}$$

- For $\omega < \omega_p$: $\epsilon_r < 0$, waves are evanescent (metals are opaque)
- For $\omega > \omega_p$: $\epsilon_r > 0$, waves propagate (metals become transparent)

This explains why metals are shiny in the visible but transparent to X-rays, and why alkali metals have UV transparency edges.

### 4.3 Plasma Frequency of Common Metals

| Metal | $\omega_p$ (PHz) | $\lambda_p$ (nm) | UV transparent? |
|-------|-------------------|-------------------|-----------------|
| Na    | 8.8               | 214               | Yes ($\lambda < 214$ nm) |
| Al    | 22.9              | 82                | Yes (deep UV) |
| Au    | 13.7              | 138               | Interband transitions complicate |
| Ag    | 14.0              | 135               | Cleanest Drude metal |

```python
def drude_permittivity(omega, omega_p, gamma):
    """
    Drude model for metallic permittivity.

    Why the Drude model: despite its simplicity (free electrons + damping),
    it accurately predicts optical properties of simple metals like Al, Na, Ag
    from IR through visible frequencies.
    """
    return 1.0 - omega_p**2 / (omega**2 + 1j * gamma * omega)

# Silver parameters
omega_p_Ag = 14.0e15   # plasma frequency ~14 PHz
gamma_Ag = 0.032e15    # damping rate ~32 THz (low for Ag)

omega_metal = np.linspace(0.1e15, 25e15, 5000)
eps_Ag = drude_permittivity(omega_metal, omega_p_Ag, gamma_Ag)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(omega_metal / 1e15, eps_Ag.real, 'b-', linewidth=1.5, label="$\\epsilon'$")
ax.plot(omega_metal / 1e15, eps_Ag.imag, 'r--', linewidth=1.5, label="$\\epsilon''$")
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=omega_p_Ag / 1e15, color='green', linestyle=':', label=f'$\\omega_p$ = {omega_p_Ag/1e15:.1f} PHz')

# Shade the opaque region where epsilon' < 0
ax.axvspan(0, omega_p_Ag / 1e15, alpha=0.1, color='gray', label='Opaque region')

ax.set_xlabel('Frequency (PHz)')
ax.set_ylabel('$\\epsilon_r$')
ax.set_title('Drude Model: Silver Permittivity')
ax.set_ylim(-10, 5)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("drude_silver.png", dpi=150)
plt.show()
```

---

## 5. Skin Depth

### 5.1 Derivation

When a wave enters a conducting medium, the complex wave number is:

$$\tilde{k} = \omega\sqrt{\mu\tilde{\epsilon}} = \omega\sqrt{\mu\left(\epsilon + i\frac{\sigma}{\omega}\right)}$$

For a good conductor ($\sigma \gg \omega\epsilon$):

$$\tilde{k} \approx \sqrt{\frac{\mu\sigma\omega}{2}}(1 + i)$$

The field decays as $e^{-z/\delta}$ where the **skin depth** is:

$$\boxed{\delta = \sqrt{\frac{2}{\mu\sigma\omega}}}$$

The skin depth tells us how far the wave penetrates before its amplitude falls to $1/e \approx 37\%$.

### 5.2 Physical Interpretation

The skin depth decreases with:
- **Higher frequency** ($\delta \propto 1/\sqrt{\omega}$): high-frequency fields are confined to the surface
- **Higher conductivity** ($\delta \propto 1/\sqrt{\sigma}$): better conductors reflect more, absorb in a thinner layer
- **Higher permeability** ($\delta \propto 1/\sqrt{\mu}$): magnetic materials have even smaller skin depths

### 5.3 Practical Values

| Material | $\sigma$ (S/m) | $\delta$ at 60 Hz | $\delta$ at 1 MHz | $\delta$ at 1 GHz |
|----------|-----------------|--------------------|--------------------|---------------------|
| Copper   | $5.96 \times 10^7$ | 8.5 mm | 0.066 mm | 2.1 $\mu$m |
| Aluminum | $3.77 \times 10^7$ | 10.7 mm | 0.083 mm | 2.6 $\mu$m |
| Seawater | 4.0              | 32.5 m  | 0.25 m   | 7.9 mm |

This is why submarines use extremely low frequency (ELF) radio for communication, and why microwave ovens use a metal mesh with holes smaller than the skin depth at 2.45 GHz.

```python
def skin_depth(freq, sigma, mu_r=1.0):
    """
    Calculate electromagnetic skin depth.

    Why skin depth matters: it determines shielding effectiveness,
    heating depth in induction furnaces, and the minimum thickness
    of conductive coatings on waveguides.
    """
    mu = 4 * np.pi * 1e-7 * mu_r  # permeability (H/m)
    omega = 2 * np.pi * freq
    return np.sqrt(2.0 / (mu * sigma * omega))

# Compute skin depth vs frequency for copper
freq = np.logspace(1, 12, 1000)  # 10 Hz to 1 THz
sigma_Cu = 5.96e7  # copper conductivity (S/m)
delta_Cu = skin_depth(freq, sigma_Cu)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(freq, delta_Cu * 1e3, 'b-', linewidth=2, label='Copper')

# Add reference lines for common frequencies
ref_freqs = {'60 Hz': 60, '1 kHz': 1e3, '1 MHz': 1e6, '1 GHz': 1e9}
for name, f in ref_freqs.items():
    d = skin_depth(f, sigma_Cu) * 1e3
    ax.plot(f, d, 'ro', markersize=8)
    ax.annotate(f'{name}\n$\\delta$ = {d:.2g} mm',
                xy=(f, d), xytext=(f * 3, d * 2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Skin depth (mm)')
ax.set_title('Skin Depth in Copper vs. Frequency')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("skin_depth_copper.png", dpi=150)
plt.show()
```

---

## 6. Complex Refractive Index and Absorption

### 6.1 Definition

The complex refractive index is:

$$\tilde{n} = n + i\kappa$$

where $n$ is the (real) refractive index and $\kappa$ is the **extinction coefficient**. They relate to the complex permittivity by:

$$\tilde{n}^2 = \tilde{\epsilon}_r \mu_r$$

For non-magnetic materials ($\mu_r = 1$):

$$n^2 - \kappa^2 = \epsilon', \quad 2n\kappa = \epsilon''$$

Solving these:

$$n = \sqrt{\frac{|\tilde{\epsilon}_r| + \epsilon'}{2}}, \quad \kappa = \sqrt{\frac{|\tilde{\epsilon}_r| - \epsilon'}{2}}$$

### 6.2 Absorption Coefficient

The wave intensity decays as:

$$I(z) = I_0 \, e^{-\alpha z}$$

where the **absorption coefficient** (or attenuation coefficient) is:

$$\alpha = \frac{2\omega\kappa}{c} = \frac{2\kappa\omega}{c}$$

The factor of 2 arises because intensity is proportional to $|E|^2$, while the field amplitude decays as $e^{-\kappa\omega z/c}$.

### 6.3 Beer-Lambert Law

In spectroscopy, absorption is often expressed as:

$$A = \log_{10}\left(\frac{I_0}{I}\right) = \frac{\alpha z}{2.303}$$

The **molar absorption coefficient** $\varepsilon_m$ relates to the microscopic absorption:

$$\alpha = \varepsilon_m c_{\text{mol}} \ln 10$$

where $c_{\text{mol}}$ is the molar concentration.

```python
def complex_refractive_index(eps_complex, mu_r=1.0):
    """
    Compute n and kappa from complex permittivity.

    Why separate n and kappa: experiments measure reflectance and
    transmittance, which map directly to n and kappa rather than
    epsilon' and epsilon''.
    """
    n_complex = np.sqrt(eps_complex * mu_r)
    return n_complex.real, n_complex.imag

# Demonstrate absorption in glass with a Lorentz resonance
omega_abs = np.linspace(1e14, 1.2e16, 5000)
eps_glass = lorentz_permittivity(omega_abs, omega_0=6e15, gamma=1e14, omega_p=4e15)

n_glass, kappa_glass = complex_refractive_index(eps_glass)
c = 3e8
alpha_glass = 2 * omega_abs * kappa_glass / c  # absorption coefficient (1/m)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(omega_abs / 1e15, n_glass, 'b-', linewidth=1.5)
axes[0].set_ylabel('Refractive index $n$')
axes[0].set_title('Complex Refractive Index (Lorentz Model)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega_abs / 1e15, alpha_glass * 1e-6, 'r-', linewidth=1.5)
axes[1].set_xlabel('Frequency (PHz)')
axes[1].set_ylabel('Absorption coeff. $\\alpha$ (10$^6$ m$^{-1}$)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("complex_refractive_index.png", dpi=150)
plt.show()
```

---

## 7. Kramers-Kronig Relations

### 7.1 Causality and Analyticity

The most profound constraint on the dielectric response comes from **causality**: the polarization at time $t$ cannot depend on the electric field at future times. Mathematically, the response function $\chi(t)$ vanishes for $t < 0$.

This causality condition, combined with the assumption that $\chi(\omega)$ is analytic in the upper half of the complex $\omega$-plane and decays at infinity, leads to the **Kramers-Kronig (KK) relations**:

$$\epsilon'(\omega) - 1 = \frac{2}{\pi} \, \mathcal{P} \int_0^{\infty} \frac{\omega' \epsilon''(\omega')}{\omega'^2 - \omega^2} \, d\omega'$$

$$\epsilon''(\omega) = -\frac{2\omega}{\pi} \, \mathcal{P} \int_0^{\infty} \frac{\epsilon'(\omega') - 1}{\omega'^2 - \omega^2} \, d\omega'$$

where $\mathcal{P}$ denotes the Cauchy principal value.

### 7.2 Physical Meaning

The KK relations say that **dispersion and absorption are not independent** — knowing $\epsilon''(\omega)$ at all frequencies completely determines $\epsilon'(\omega)$, and vice versa. This is a powerful experimental tool: if you measure the absorption spectrum, you can compute the refractive index without a separate measurement.

> **Analogy**: Kramers-Kronig relations are like a holistic health check. If you know how much energy a system absorbs at every frequency (the "symptoms"), you can deduce exactly how it refracts light (the "diagnosis"). You cannot have absorption without dispersion, just as you cannot have symptoms without an underlying condition.

### 7.3 Sum Rules

The KK relations imply sum rules, such as:

$$\int_0^{\infty} \omega \, \epsilon''(\omega) \, d\omega = \frac{\pi}{2} \omega_p^2$$

This **f-sum rule** connects the total absorption to the electron density, regardless of the specific model.

```python
from scipy.integrate import simpson

def kramers_kronig_real(omega, eps_imag):
    """
    Compute epsilon'(omega) from epsilon''(omega) using Kramers-Kronig.

    Why KK is useful: in experiments, you often measure only absorption
    (eps'') via transmission. KK lets you recover eps' (refractive index)
    without an independent reflectance measurement.
    """
    eps_real = np.ones_like(omega)  # start with the 1 (vacuum contribution)
    domega = omega[1] - omega[0]

    for i, w in enumerate(omega):
        # Cauchy principal value: skip the singularity at omega' = omega
        integrand = omega * eps_imag / (omega**2 - w**2 + 1e-30)
        # Zero out the singular point
        if i < len(omega):
            integrand[i] = 0.0
        eps_real[i] += (2.0 / np.pi) * simpson(integrand, x=omega)

    return eps_real

# Verify KK on the Lorentz model
omega_kk = np.linspace(0.01e15, 1.5e16, 2000)
eps_lorentz = lorentz_permittivity(omega_kk, omega_0=6e15, gamma=1e14, omega_p=4e15)

# Use exact imaginary part to reconstruct real part
eps_real_kk = kramers_kronig_real(omega_kk, eps_lorentz.imag)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(omega_kk / 1e15, eps_lorentz.real, 'b-', linewidth=2, label="Exact $\\epsilon'$")
ax.plot(omega_kk / 1e15, eps_real_kk, 'r--', linewidth=2, label="KK from $\\epsilon''$")
ax.set_xlabel('Frequency (PHz)')
ax.set_ylabel("$\\epsilon'$")
ax.set_title("Kramers-Kronig Verification: Recovering $\\epsilon'$ from $\\epsilon''$")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kramers_kronig_verification.png", dpi=150)
plt.show()
```

---

## 8. Wave Propagation in Dispersive Media: Simulation

Let us simulate a Gaussian pulse propagating through a dispersive medium and observe pulse broadening.

```python
def simulate_pulse_in_dispersive_medium(n_func, omega_center, bandwidth,
                                         z_max, c=3e8, N=4096):
    """
    Simulate Gaussian pulse propagation in a dispersive medium.

    Strategy:
    1. Construct pulse in frequency domain
    2. Multiply by transfer function exp(i * k(omega) * z)
    3. IFFT back to time domain

    Why frequency-domain approach: it naturally handles arbitrary dispersion
    relations without the numerical instabilities of direct PDE solvers.
    """
    # Frequency grid
    domega = 2 * bandwidth / N
    omega = omega_center + np.linspace(-bandwidth, bandwidth, N)

    # Gaussian pulse spectrum centered at omega_center
    pulse_spectrum = np.exp(-0.5 * ((omega - omega_center) / (bandwidth / 10))**2)

    # Propagation distances
    distances = [0, z_max / 4, z_max / 2, z_max]

    # Time grid (from inverse FFT)
    dt = 2 * np.pi / (N * domega)
    t = np.arange(N) * dt
    t -= t[N // 2]  # center the time axis

    fig, axes = plt.subplots(len(distances), 1, figsize=(10, 10), sharex=True)

    for ax, z in zip(axes, distances):
        # Transfer function: phase accumulated over distance z
        n_vals = n_func(omega)
        k_vals = n_vals * omega / c
        propagated = pulse_spectrum * np.exp(1j * k_vals * z)

        # Inverse FFT to get time-domain pulse
        pulse_time = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(propagated)))
        intensity = np.abs(pulse_time)**2
        intensity /= intensity.max()

        ax.plot(t * 1e15, intensity, 'b-', linewidth=1.5)
        ax.set_ylabel('Intensity (norm.)')
        ax.set_title(f'z = {z*1e6:.0f} $\\mu$m')
        ax.set_xlim(-500, 500)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (fs)')
    plt.suptitle('Gaussian Pulse in Dispersive Medium', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("pulse_dispersion.png", dpi=150)
    plt.show()

# Refractive index function from Lorentz model (away from resonance)
def n_dispersive(omega):
    eps = lorentz_permittivity(omega, omega_0=6e15, gamma=1e14, omega_p=4e15)
    return np.sqrt(eps).real

simulate_pulse_in_dispersive_medium(
    n_func=n_dispersive,
    omega_center=3e15,       # visible light ~3 PHz
    bandwidth=2e15,          # broad bandwidth for short pulse
    z_max=100e-6             # 100 micrometers
)
```

---

## Summary

| Concept | Key Formula | Physical Meaning |
|---------|-------------|------------------|
| Complex permittivity | $\tilde{\epsilon} = \epsilon' + i\epsilon''$ | Combines dispersion and absorption |
| Lorentz model | $\epsilon_r = 1 + \omega_p^2/(\omega_0^2 - \omega^2 - i\gamma\omega)$ | Bound electron response |
| Drude model | $\epsilon_r = 1 - \omega_p^2/(\omega^2 + i\gamma\omega)$ | Free electron response |
| Skin depth | $\delta = \sqrt{2/\mu\sigma\omega}$ | Penetration depth in conductors |
| Complex refractive index | $\tilde{n} = n + i\kappa$ | Phase velocity and attenuation |
| Absorption coefficient | $\alpha = 2\omega\kappa/c$ | Intensity decay rate |
| Kramers-Kronig | $\epsilon' \leftrightarrow \epsilon''$ via integral | Causality links dispersion and absorption |

---

## Exercises

### Exercise 1: Drude Model for Aluminum
Aluminum has a plasma frequency $\omega_p = 2.29 \times 10^{16}$ rad/s and damping rate $\gamma = 1.22 \times 10^{14}$ rad/s. (a) Plot $\epsilon_r(\omega)$ from 0 to $3\omega_p$. (b) Find the frequency at which aluminum becomes transparent. (c) Calculate the skin depth at 1 GHz and at visible frequencies (500 nm).

### Exercise 2: Multi-Resonance Lorentz Model
Glass has absorption resonances in the UV and IR. Model glass using two Lorentz oscillators: one at $\omega_1 = 1.5 \times 10^{16}$ rad/s (UV) with $\gamma_1 = 10^{14}$ rad/s, $f_1 = 0.6$, and one at $\omega_2 = 6 \times 10^{13}$ rad/s (IR) with $\gamma_2 = 5 \times 10^{12}$ rad/s, $f_2 = 0.4$. Use $\omega_p = 2 \times 10^{16}$ rad/s. Plot $n(\omega)$ and $\kappa(\omega)$ in the visible range. Does the model predict that glass is transparent to visible light?

### Exercise 3: Kramers-Kronig Numerical Verification
Take the Drude model for silver ($\omega_p = 14.0 \times 10^{15}$ rad/s, $\gamma = 3.2 \times 10^{13}$ rad/s). (a) Compute $\epsilon''(\omega)$ analytically. (b) Use the Kramers-Kronig integral to numerically reconstruct $\epsilon'(\omega)$. (c) Compare with the exact Drude $\epsilon'(\omega)$. Discuss sources of numerical error.

### Exercise 4: Pulse Broadening
A 10 fs Gaussian pulse at 800 nm center wavelength propagates through 1 cm of BK7 glass (GVD $\approx 44.7$ fs$^2$/mm at 800 nm). (a) Estimate the output pulse duration using the formula $\tau_{\text{out}} = \tau_{\text{in}} \sqrt{1 + (4\ln 2 \cdot \text{GVD} \cdot L / \tau_{\text{in}}^2)^2}$. (b) Simulate the propagation numerically and verify your analytical estimate.

---

[← Previous: 09. EM Waves in Vacuum](09_EM_Waves_Vacuum.md) | [Next: 11. Reflection and Refraction →](11_Reflection_and_Refraction.md)

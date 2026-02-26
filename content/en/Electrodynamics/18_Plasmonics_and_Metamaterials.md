# 18. Applications — Plasmonics and Metamaterials

[← Previous: 17. Electromagnetic Scattering](17_Electromagnetic_Scattering.md) | [Next: Overview →](00_Overview.md)

## Learning Objectives

1. Understand surface plasmon polaritons (SPPs) and their dispersion relation at metal-dielectric interfaces
2. Analyze localized surface plasmon resonances (LSPRs) in metallic nanoparticles
3. Explain how nanoparticle geometry determines the resonance wavelength
4. Define metamaterials and understand how sub-wavelength structures create effective material properties
5. Derive the conditions for negative refractive index and understand Veselago's predictions
6. Describe split-ring resonators and wire arrays as building blocks for metamaterials
7. Explore applications: biosensors, electromagnetic cloaking, super-resolution imaging, and photonic crystals

The preceding lessons built the foundations of classical electrodynamics: Maxwell's equations, wave propagation, scattering, and radiation. This final lesson shows how these foundations enable two of the most exciting frontiers of modern photonics. **Plasmonics** exploits the collective oscillation of free electrons in metals to confine light to nanometer scales — far below the diffraction limit — enabling ultra-sensitive biosensors and nanoscale optical circuits. **Metamaterials** take a bolder approach, engineering artificial electromagnetic materials with properties impossible in nature, including negative refractive index, electromagnetic invisibility cloaks, and perfect lenses. Both fields illustrate the creative power of Maxwell's equations when combined with modern nanofabrication.

> **Analogy**: Plasmonics is like conducting an orchestra where the musicians are electrons. When the conductor (incident light) hits the right frequency, all the electrons in a metal nanoparticle oscillate in perfect unison, creating an intense resonance that concentrates the "sound" (electromagnetic field) into a tiny volume. Metamaterials, meanwhile, are like building a new instrument from scratch — by arranging tiny resonant structures in a pattern smaller than the wavelength, you create a material that "plays notes" (has electromagnetic properties) that no natural material can produce.

---

## 1. Surface Plasmon Polaritons

### 1.1 What is a Surface Plasmon?

A **surface plasmon polariton (SPP)** is an electromagnetic wave that propagates along a metal-dielectric interface, with its field exponentially decaying into both media. It is a coupled oscillation of the electromagnetic field and the free-electron plasma at the surface.

The key feature: SPPs have a wavelength **shorter** than free-space light at the same frequency. This allows confinement of light below the diffraction limit.

### 1.2 Dispersion Relation

Consider a planar interface between a metal ($\epsilon_m(\omega)$, typically negative) and a dielectric ($\epsilon_d > 0$). The SPP dispersion relation is obtained by matching boundary conditions for a TM surface wave:

$$\boxed{k_{\text{SPP}} = \frac{\omega}{c}\sqrt{\frac{\epsilon_m \epsilon_d}{\epsilon_m + \epsilon_d}}}$$

Using the Drude model for the metal ($\epsilon_m = 1 - \omega_p^2/\omega^2$ for the lossless case):

- For $\omega \to 0$: $k_{\text{SPP}} \approx (\omega/c)\sqrt{\epsilon_d}$ (approaches the light line)
- For $\omega \to \omega_{sp} = \omega_p/\sqrt{1 + \epsilon_d}$: $k_{\text{SPP}} \to \infty$ (asymptotic surface plasmon frequency)

The SPP always lies to the right of the light line $\omega = ck/\sqrt{\epsilon_d}$, meaning **it cannot be excited by freely propagating light** — special coupling mechanisms are needed.

### 1.3 Field Profile

The fields decay exponentially away from the interface:

$$\mathbf{E} \propto e^{ik_{\text{SPP}}x - \kappa_d z} \quad (z > 0, \text{ dielectric})$$

$$\mathbf{E} \propto e^{ik_{\text{SPP}}x + \kappa_m z} \quad (z < 0, \text{ metal})$$

where $\kappa_{d,m} = \sqrt{k_{\text{SPP}}^2 - \epsilon_{d,m}\omega^2/c^2}$ are the decay constants. The penetration depth into the dielectric is typically 100-500 nm, while penetration into the metal is limited to the skin depth (~20-30 nm for noble metals).

### 1.4 Excitation Methods

Since $k_{\text{SPP}} > k_0\sqrt{\epsilon_d}$, free-space light cannot directly excite SPPs. Common coupling schemes:

- **Prism coupling (Kretschmann configuration)**: Evanescent wave from total internal reflection in a prism provides the extra momentum
- **Grating coupling**: A periodic surface structure adds $\pm n G$ to the photon wave vector ($G = 2\pi/\Lambda$ is the grating vector)
- **Near-field excitation**: A nano-tip or subwavelength aperture provides a broad spectrum of wave vectors

```python
import numpy as np
import matplotlib.pyplot as plt

def spp_dispersion(omega, omega_p, gamma=0, eps_d=1.0):
    """
    Compute SPP dispersion relation at a metal-dielectric interface.

    Parameters:
        omega   : angular frequency array (rad/s)
        omega_p : metal plasma frequency (rad/s)
        gamma   : metal damping rate (rad/s)
        eps_d   : dielectric constant of the dielectric medium

    Why SPP dispersion: it reveals the fundamental limits of
    light confinement — SPPs can have arbitrarily short wavelengths
    near the surface plasmon frequency, but at the cost of increasing loss.
    """
    c = 3e8

    # Drude permittivity
    eps_m = 1 - omega_p**2 / (omega**2 + 1j * gamma * omega)

    # SPP wave vector (complex for lossy metal)
    k_spp = (omega / c) * np.sqrt(eps_m * eps_d / (eps_m + eps_d))

    # Light line
    k_light = omega * np.sqrt(eps_d) / c

    # Surface plasmon frequency
    omega_sp = omega_p / np.sqrt(1 + eps_d)

    return k_spp, k_light, omega_sp

# Silver parameters
omega_p = 14.0e15   # plasma frequency (rad/s)
gamma = 3.2e13      # damping (rad/s)
omega = np.linspace(0.01e15, 12e15, 2000)

# Air-silver interface
k_spp_air, k_light_air, omega_sp_air = spp_dispersion(omega, omega_p, gamma, eps_d=1.0)

# Glass-silver interface
k_spp_glass, k_light_glass, omega_sp_glass = spp_dispersion(omega, omega_p, gamma, eps_d=2.25)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Dispersion diagram
ax = axes[0]
ax.plot(k_spp_air.real / 1e7, omega / 1e15, 'b-', linewidth=2, label='SPP (air-Ag)')
ax.plot(k_light_air / 1e7, omega / 1e15, 'b--', linewidth=1, alpha=0.5, label='Light line (air)')
ax.plot(k_spp_glass.real / 1e7, omega / 1e15, 'r-', linewidth=2, label='SPP (glass-Ag)')
ax.plot(k_light_glass / 1e7, omega / 1e15, 'r--', linewidth=1, alpha=0.5, label='Light line (glass)')

ax.axhline(y=omega_sp_air / 1e15, color='blue', linestyle=':', alpha=0.5)
ax.axhline(y=omega_sp_glass / 1e15, color='red', linestyle=':', alpha=0.5)

ax.set_xlabel('$k$ (10$^7$ rad/m)')
ax.set_ylabel('Frequency (PHz)')
ax.set_title('Surface Plasmon Polariton Dispersion')
ax.set_xlim(0, 5)
ax.set_ylim(0, 10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Propagation length (1/e decay of intensity)
ax = axes[1]
L_spp_air = 1.0 / (2 * k_spp_air.imag)
L_spp_glass = 1.0 / (2 * k_spp_glass.imag)

valid_air = (omega > 0.5e15) & (omega < omega_sp_air * 0.95)
valid_glass = (omega > 0.5e15) & (omega < omega_sp_glass * 0.95)

ax.semilogy(omega[valid_air] / 1e15, L_spp_air[valid_air] * 1e6,
            'b-', linewidth=2, label='Air-Ag')
ax.semilogy(omega[valid_glass] / 1e15, L_spp_glass[valid_glass] * 1e6,
            'r-', linewidth=2, label='Glass-Ag')

ax.set_xlabel('Frequency (PHz)')
ax.set_ylabel('Propagation length ($\\mu$m)')
ax.set_title('SPP Propagation Length')
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim(0.1, 1000)

plt.tight_layout()
plt.savefig("spp_dispersion.png", dpi=150)
plt.show()
```

---

## 2. Localized Surface Plasmon Resonance (LSPR)

### 2.1 Nanoparticle Resonance

When a metallic nanoparticle (radius $a \ll \lambda$) is illuminated, the conduction electrons oscillate collectively. Using the quasi-static approximation, the electric field inside and outside the sphere is:

$$\mathbf{E}_{\text{in}} = \frac{3\epsilon_d}{\epsilon_m + 2\epsilon_d}\mathbf{E}_0$$

$$\mathbf{E}_{\text{out}} = \mathbf{E}_0 + \frac{3\hat{r}(\hat{r}\cdot\mathbf{p}) - \mathbf{p}}{4\pi\epsilon_0\epsilon_d r^3}$$

where $\mathbf{p} = 4\pi\epsilon_0\epsilon_d a^3 \frac{\epsilon_m - \epsilon_d}{\epsilon_m + 2\epsilon_d}\mathbf{E}_0$ is the induced dipole moment.

### 2.2 The Frohlich Condition

The resonance occurs when the denominator is minimized:

$$\boxed{\text{Re}[\epsilon_m(\omega)] = -2\epsilon_d}$$

This is the **Frohlich condition**. For a Drude metal in vacuum ($\epsilon_d = 1$):

$$\omega_{\text{LSPR}} = \frac{\omega_p}{\sqrt{3}}$$

The resonance frequency depends on:
- **Particle material**: Different metals have different $\omega_p$ and damping
- **Surrounding medium**: Higher $\epsilon_d$ redshifts the resonance
- **Particle shape**: Ellipsoids, rods, and other shapes have different depolarization factors

### 2.3 Field Enhancement

At resonance, the local electric field can be enhanced by factors of 10-1000 compared to the incident field. This **near-field enhancement** is the basis for:
- Surface-enhanced Raman spectroscopy (SERS): Enhancement $\propto |E/E_0|^4$, achieving $10^6$-$10^{10}$ times stronger Raman signals
- Localized surface plasmon resonance biosensors: Spectral shift upon binding of analyte molecules
- Plasmonic solar cells: Enhanced absorption in thin-film photovoltaics

```python
def nanoparticle_polarizability(omega, omega_p, gamma, a, eps_d=1.0):
    """
    Compute the polarizability of a metallic nanosphere.

    Parameters:
        omega   : angular frequency (rad/s)
        omega_p : plasma frequency (rad/s)
        gamma   : damping rate (rad/s)
        a       : sphere radius (m)
        eps_d   : dielectric constant of surrounding medium

    Why polarizability: it determines the absorption and scattering
    cross sections of the nanoparticle, which are the measurable
    quantities in optical experiments.
    """
    eps_m = 1 - omega_p**2 / (omega**2 + 1j * gamma * omega)
    alpha = 4 * np.pi * a**3 * (eps_m - eps_d) / (eps_m + 2 * eps_d)
    return alpha

def nanoparticle_cross_sections(omega, omega_p, gamma, a, eps_d=1.0):
    """Compute absorption and scattering cross sections."""
    c = 3e8
    k = omega * np.sqrt(eps_d) / c
    alpha = nanoparticle_polarizability(omega, omega_p, gamma, a, eps_d)

    # Absorption cross section (dominant for small particles)
    sigma_abs = k * np.imag(alpha)

    # Scattering cross section (proportional to a^6)
    sigma_sca = k**4 / (6 * np.pi) * np.abs(alpha)**2

    sigma_ext = sigma_abs + sigma_sca
    return sigma_ext, sigma_abs, sigma_sca

def plot_lspr():
    """Visualize LSPR for different metals and environments."""
    omega = np.linspace(1e15, 8e15, 2000)
    a = 20e-9  # 20 nm radius

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Silver nanoparticle in different media
    ax = axes[0, 0]
    for eps_d, label, color in [(1.0, 'Vacuum', 'blue'),
                                 (1.77, 'Water', 'green'),
                                 (2.25, 'Glass', 'red')]:
        sigma_ext, _, _ = nanoparticle_cross_sections(
            omega, 14e15, 3.2e13, a, eps_d)
        # Convert to nm² and normalize
        ax.plot(2*np.pi*3e8/omega * 1e9, sigma_ext * 1e18,
                color=color, linewidth=2, label=label)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('$\\sigma_{\\mathrm{ext}}$ (nm²)')
    ax.set_title('Silver NP (20 nm radius) in Different Media')
    ax.set_xlim(300, 800)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Gold nanoparticle (interband transitions modify the response)
    ax = axes[0, 1]
    omega_p_Au = 13.7e15
    gamma_Au = 1.075e14  # higher damping than Ag
    for a_nm in [10, 20, 40, 80]:
        a_val = a_nm * 1e-9
        sigma_ext, _, _ = nanoparticle_cross_sections(
            omega, omega_p_Au, gamma_Au, a_val, eps_d=1.0)
        ax.plot(2*np.pi*3e8/omega * 1e9, sigma_ext / (np.pi * a_val**2),
                linewidth=2, label=f'{a_nm} nm')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('$Q_{\\mathrm{ext}}$')
    ax.set_title('Gold NP: Size Dependence')
    ax.set_xlim(300, 800)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Field enhancement at resonance
    ax = axes[1, 0]
    eps_d = 1.0
    omega_res = omega[np.argmax(np.abs(
        nanoparticle_polarizability(omega, 14e15, 3.2e13, 20e-9, eps_d)))]

    # Field enhancement as a function of position along the axis
    r = np.linspace(1.01 * a, 10 * a, 200)
    # On-axis enhancement (theta=0): |E/E0|^2 = |1 + 2*alpha/(4*pi*r^3)|^2
    alpha_res = nanoparticle_polarizability(
        np.array([omega_res]), 14e15, 3.2e13, a, eps_d)[0]
    enhancement = np.abs(1 + 2 * alpha_res / (4 * np.pi * r**3))**2

    ax.semilogy(r / a, enhancement, 'b-', linewidth=2)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='red', linestyle=':', label='Particle surface')
    ax.set_xlabel('Distance from center ($r/a$)')
    ax.set_ylabel('$|E/E_0|^2$')
    ax.set_title('Near-Field Enhancement (Ag, on-axis)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # Biosensor application: spectral shift
    ax = axes[1, 1]
    eps_d_values = np.linspace(1.0, 2.5, 20)
    lambda_res = np.zeros(len(eps_d_values))

    for i, eps_d in enumerate(eps_d_values):
        sigma_ext, _, _ = nanoparticle_cross_sections(
            omega, 14e15, 3.2e13, 20e-9, eps_d)
        lambda_res[i] = 2 * np.pi * 3e8 / omega[np.argmax(sigma_ext)] * 1e9

    ax.plot(eps_d_values, lambda_res, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Surrounding $\\epsilon_d$')
    ax.set_ylabel('LSPR wavelength (nm)')
    ax.set_title('LSPR Sensitivity to Environment')
    ax.grid(True, alpha=0.3)

    # Sensitivity
    sensitivity = np.gradient(lambda_res, eps_d_values)
    ax2 = ax.twinx()
    ax2.plot(eps_d_values, sensitivity, 'r--', linewidth=1.5, label='Sensitivity')
    ax2.set_ylabel('Sensitivity (nm/RIU)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.suptitle('Localized Surface Plasmon Resonance', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("lspr_properties.png", dpi=150)
    plt.show()

plot_lspr()
```

---

## 3. Metamaterials

### 3.1 Concept

**Metamaterials** are artificially structured materials with electromagnetic properties not found in nature. The key idea: arrange sub-wavelength resonant elements (meta-atoms) in a periodic pattern. When the unit cell is much smaller than the wavelength ($a \ll \lambda$), the medium can be described by effective permittivity $\epsilon_{\text{eff}}(\omega)$ and permeability $\mu_{\text{eff}}(\omega)$.

### 3.2 Negative Refractive Index

In 1968, Veselago showed that a material with **simultaneously negative** $\epsilon$ and $\mu$ would have:

$$n = -\sqrt{|\epsilon_r||\mu_r|}$$

This **negative refractive index** leads to remarkable consequences:
- Snell's law gives refraction to the **same side** of the normal
- The Poynting vector (energy flow) is antiparallel to the wave vector (phase propagation)
- The Doppler effect is reversed
- Cherenkov radiation is emitted backward

### 3.3 Building Blocks

**Wire arrays** (metallic wires): provide negative $\epsilon_{\text{eff}}$ below a plasma frequency that depends on wire spacing and radius:

$$\epsilon_{\text{eff}} = 1 - \frac{\omega_{p,\text{eff}}^2}{\omega^2}, \quad \omega_{p,\text{eff}}^2 = \frac{2\pi c^2}{a^2 \ln(a/r)}$$

**Split-ring resonators (SRRs)**: provide negative $\mu_{\text{eff}}$ near the resonance frequency:

$$\mu_{\text{eff}} = 1 - \frac{F\omega^2}{\omega^2 - \omega_0^2 + i\gamma\omega}$$

where $F$ is a filling factor and $\omega_0$ is the LC resonance of the ring.

By combining wire arrays and SRRs, one can achieve a frequency band where both $\epsilon_{\text{eff}} < 0$ and $\mu_{\text{eff}} < 0$, yielding a negative index metamaterial.

```python
def metamaterial_effective_properties(omega, omega_p_eff, omega_0_srr,
                                      gamma_srr, F=0.5, gamma_wire=0):
    """
    Compute effective permittivity and permeability of a metamaterial
    consisting of wire arrays + split-ring resonators.

    Parameters:
        omega       : angular frequency (rad/s)
        omega_p_eff : effective plasma frequency of wire array (rad/s)
        omega_0_srr : resonance frequency of SRRs (rad/s)
        gamma_srr   : damping of SRRs (rad/s)
        F           : filling factor of SRRs
        gamma_wire  : damping of wire array (rad/s)

    Why effective medium: when meta-atoms are much smaller than lambda,
    the metamaterial behaves like a continuous medium with engineered
    epsilon and mu — exactly as an ordinary dielectric behaves as
    continuous despite being made of discrete atoms.
    """
    # Wire array: effective permittivity (Drude-like)
    eps_eff = 1 - omega_p_eff**2 / (omega**2 + 1j * gamma_wire * omega)

    # SRR: effective permeability (Lorentz-like)
    mu_eff = 1 - F * omega**2 / (omega**2 - omega_0_srr**2 + 1j * gamma_srr * omega)

    # Effective refractive index
    n_eff = np.sqrt(eps_eff * mu_eff)

    # Choose the correct branch: Re(n) < 0 when both eps and mu are negative
    # This is the key physics: the sign of n must be chosen consistently
    # with causality (Im(n) > 0 for passive media)
    n_eff = np.where(
        (np.real(eps_eff) < 0) & (np.real(mu_eff) < 0),
        -np.abs(n_eff.real) + 1j * np.abs(n_eff.imag),
        n_eff
    )

    return eps_eff, mu_eff, n_eff


def plot_metamaterial():
    """Visualize metamaterial effective properties and negative index band."""
    omega = np.linspace(0.1e10, 20e10, 2000)

    omega_p_eff = 12e10   # wire array plasma frequency ~12 GHz
    omega_0_srr = 8e10    # SRR resonance ~8 GHz
    gamma_srr = 0.3e10    # SRR damping
    gamma_wire = 0.1e10   # wire damping

    eps_eff, mu_eff, n_eff = metamaterial_effective_properties(
        omega, omega_p_eff, omega_0_srr, gamma_srr,
        F=0.5, gamma_wire=gamma_wire)

    freq_GHz = omega / (2 * np.pi * 1e9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Effective permittivity
    ax = axes[0, 0]
    ax.plot(freq_GHz, eps_eff.real, 'b-', linewidth=2, label="$\\epsilon'$")
    ax.plot(freq_GHz, eps_eff.imag, 'b--', linewidth=1.5, label="$\\epsilon''$")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$\\epsilon_{\\mathrm{eff}}$')
    ax.set_title('Effective Permittivity (Wire Array)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 3)

    # Effective permeability
    ax = axes[0, 1]
    ax.plot(freq_GHz, mu_eff.real, 'r-', linewidth=2, label="$\\mu'$")
    ax.plot(freq_GHz, mu_eff.imag, 'r--', linewidth=1.5, label="$\\mu''$")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$\\mu_{\\mathrm{eff}}$')
    ax.set_title('Effective Permeability (SRRs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 3)

    # Effective refractive index
    ax = axes[1, 0]
    ax.plot(freq_GHz, n_eff.real, 'k-', linewidth=2, label="$n'$ (real)")
    ax.plot(freq_GHz, n_eff.imag, 'k--', linewidth=1.5, label="$n''$ (imag)")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Shade the negative-index band
    neg_idx = (np.real(eps_eff) < 0) & (np.real(mu_eff) < 0)
    if np.any(neg_idx):
        freq_neg = freq_GHz[neg_idx]
        ax.axvspan(freq_neg.min(), freq_neg.max(), alpha=0.2, color='green',
                   label='Negative index band')

    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$n_{\\mathrm{eff}}$')
    ax.set_title('Effective Refractive Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)

    # Property map
    ax = axes[1, 1]
    eps_r = eps_eff.real
    mu_r = mu_eff.real

    # Color regions by quadrant
    ax.scatter(eps_r, mu_r, c=freq_GHz, cmap='viridis', s=3, alpha=0.5)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)

    # Label quadrants
    ax.text(1.5, 1.5, 'Standard\nmaterials', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(-2, -2, 'Negative\nindex', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(-2, 1.5, '$\\epsilon < 0$\n(metals)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax.text(1.5, -2, '$\\mu < 0$\n(SRRs only)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    ax.set_xlabel("$\\epsilon'_{\\mathrm{eff}}$")
    ax.set_ylabel("$\\mu'_{\\mathrm{eff}}$")
    ax.set_title('$\\epsilon$-$\\mu$ Phase Space')
    ax.set_xlim(-4, 3)
    ax.set_ylim(-4, 3)
    ax.grid(True, alpha=0.3)
    cb = plt.colorbar(ax.collections[0], ax=ax, label='Frequency (GHz)')

    plt.suptitle('Metamaterial Effective Properties', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("metamaterial_properties.png", dpi=150)
    plt.show()

plot_metamaterial()
```

---

## 4. Electromagnetic Cloaking

### 4.1 Transformation Optics

**Transformation optics** exploits the form-invariance of Maxwell's equations under coordinate transformations. A coordinate transformation $\mathbf{r} \to \mathbf{r}'$ maps to an equivalent problem in physical space with transformed material parameters:

$$\epsilon'_{ij} = \frac{J_i^{\ a} J_j^{\ b} \epsilon_{ab}}{\det(\mathbf{J})}, \quad \mu'_{ij} = \frac{J_i^{\ a} J_j^{\ b} \mu_{ab}}{\det(\mathbf{J})}$$

where $J_i^{\ a} = \partial x'^i / \partial x^a$ is the Jacobian of the transformation.

### 4.2 Cylindrical Cloak

To cloak a cylinder of radius $a$, we compress the region $0 \leq r \leq b$ into the annulus $a \leq r' \leq b$ using the transformation:

$$r' = a + r\frac{b - a}{b}, \quad \theta' = \theta, \quad z' = z$$

This requires anisotropic, inhomogeneous material parameters in the cloak shell:

$$\epsilon_r' = \mu_r' = \frac{r' - a}{r'}, \quad \epsilon_\theta' = \mu_\theta' = \frac{r'}{r' - a}, \quad \epsilon_z' = \mu_z' = \left(\frac{b}{b - a}\right)^2 \frac{r' - a}{r'}$$

The inner boundary ($r' = a$) has $\epsilon_r' = \mu_r' = 0$ and $\epsilon_\theta' = \mu_\theta' \to \infty$ — extreme parameter values that are challenging to realize.

### 4.3 Practical Limitations

- **Bandwidth**: Cloaking requires dispersive materials, limiting bandwidth
- **Loss**: Real metamaterials have loss, causing shadows
- **Size**: Only demonstrated at microwave frequencies so far (centimeter-scale)
- **Imperfect parameters**: Simplified designs reduce performance

> **Analogy**: Imagine a river flowing around a rock. If you shape the riverbed carefully, the water flows smoothly around the rock and reconverges downstream as if the rock were not there. Transformation optics does the same for light — it designs the "riverbed" (material parameters) so that electromagnetic waves flow smoothly around the cloaked object.

---

## 5. Photonic Crystals

### 5.1 Concept

Photonic crystals are periodic dielectric structures with a period comparable to the wavelength. They create **photonic band gaps** — frequency ranges where no electromagnetic modes can propagate, analogous to electronic band gaps in semiconductors.

### 5.2 1D Photonic Crystal (Bragg Mirror)

A multilayer stack of alternating high and low refractive index layers ($n_H$, $n_L$) with quarter-wave thickness creates a reflection band centered at the design wavelength.

The bandgap width for a quarter-wave stack is approximately:

$$\frac{\Delta\omega}{\omega_0} = \frac{4}{\pi}\arcsin\left(\frac{n_H - n_L}{n_H + n_L}\right)$$

### 5.3 2D and 3D Photonic Crystals

In 2D and 3D, the band structure depends on the lattice symmetry:
- **2D triangular lattice of air holes in dielectric**: Large TE bandgap
- **2D square lattice of dielectric rods**: Large TM bandgap
- **3D diamond lattice (woodpile structure)**: Complete 3D bandgap

```python
def photonic_crystal_1d(n_H, n_L, N_periods, wavelength_range, lambda_design):
    """
    Compute reflectance of a 1D photonic crystal (Bragg mirror)
    using the transfer matrix method.

    Why photonic crystals: they enable complete control over light
    propagation — waveguiding around sharp bends, ultra-high-Q cavities,
    and engineered group velocity for slow light applications.
    """
    d_H = lambda_design / (4 * n_H)
    d_L = lambda_design / (4 * n_L)

    R = np.zeros(len(wavelength_range))

    for idx, lam in enumerate(wavelength_range):
        M = np.eye(2, dtype=complex)

        for _ in range(N_periods):
            # High-index layer
            delta_H = 2 * np.pi * n_H * d_H / lam
            M_H = np.array([
                [np.cos(delta_H), -1j * np.sin(delta_H) / n_H],
                [-1j * n_H * np.sin(delta_H), np.cos(delta_H)]
            ])

            # Low-index layer
            delta_L = 2 * np.pi * n_L * d_L / lam
            M_L = np.array([
                [np.cos(delta_L), -1j * np.sin(delta_L) / n_L],
                [-1j * n_L * np.sin(delta_L), np.cos(delta_L)]
            ])

            M = M @ M_H @ M_L

        # Reflection coefficient (air substrate)
        n_sub = 1.52  # glass substrate
        n_0 = 1.0     # air

        r = ((M[0,0] + M[0,1]*n_sub)*n_0 - (M[1,0] + M[1,1]*n_sub)) / \
            ((M[0,0] + M[0,1]*n_sub)*n_0 + (M[1,0] + M[1,1]*n_sub))

        R[idx] = np.abs(r)**2

    return R


def plot_photonic_crystal():
    """Demonstrate photonic band gap in a 1D photonic crystal."""
    lam = np.linspace(300, 900, 1000) * 1e-9  # nm
    lambda_design = 550e-9  # green

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Effect of number of periods
    ax = axes[0]
    for N in [2, 5, 10, 20]:
        R = photonic_crystal_1d(n_H=2.3, n_L=1.38, N_periods=N,
                                wavelength_range=lam, lambda_design=lambda_design)
        ax.plot(lam * 1e9, R * 100, linewidth=1.5, label=f'{N} periods')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('1D Photonic Crystal: Effect of Period Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effect of index contrast
    ax = axes[1]
    for n_H, n_L, label in [(1.5, 1.38, 'Low contrast'),
                              (2.0, 1.38, 'Medium'),
                              (2.3, 1.38, 'TiO$_2$/MgF$_2$'),
                              (3.5, 1.38, 'High contrast')]:
        R = photonic_crystal_1d(n_H=n_H, n_L=n_L, N_periods=10,
                                wavelength_range=lam, lambda_design=lambda_design)
        ax.plot(lam * 1e9, R * 100, linewidth=1.5,
                label=f'$n_H$={n_H}, $n_L$={n_L}')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('1D Photonic Crystal: Effect of Index Contrast')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Photonic Crystal Band Gaps', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("photonic_crystal.png", dpi=150)
    plt.show()

plot_photonic_crystal()
```

---

## 6. Applications Summary

### 6.1 Biosensors

LSPR-based biosensors detect binding of molecules to nanoparticle surfaces through a shift in the resonance wavelength. Typical sensitivity: 100-400 nm per refractive index unit (RIU), with detection limits approaching single molecules.

### 6.2 Super-Resolution Imaging

The **Veselago-Pendry perfect lens** uses a slab of negative-index material to amplify evanescent waves (which carry sub-wavelength spatial information) and reconstruct a perfect image. Although a true perfect lens remains impractical, near-field superlenses using silver films have demonstrated resolution below $\lambda/6$.

### 6.3 Optical Computing

Plasmonic waveguides can carry signals in nanometer-wide channels, potentially replacing electronic interconnects in dense integrated circuits. Current challenges: propagation loss (plasmon propagation length is typically 10-100 $\mu$m).

### 6.4 Energy Harvesting

Metamaterial perfect absorbers can be engineered to absorb specific wavelengths with near-100% efficiency, useful for thermophotovoltaics and thermal emitters.

| Application | Technology | Status |
|-------------|-----------|--------|
| LSPR biosensors | Gold nanoparticles | Commercial products available |
| SERS | Nanostructured substrates | Widely used in analytical chemistry |
| Metamaterial absorbers | Split-ring + wire arrays | Microwave/THz demonstrated |
| Electromagnetic cloaking | Transformation optics | Proof-of-concept at microwave |
| Superlens | Silver thin film | Near-field demonstrated at UV |
| Photonic crystals | 2D/3D periodic structures | Optical fibers, LEDs, lasers |

---

## Summary

| Concept | Key Formula/Principle | Physical Meaning |
|---------|----------------------|------------------|
| SPP dispersion | $k_{\text{SPP}} = (\omega/c)\sqrt{\epsilon_m\epsilon_d/(\epsilon_m + \epsilon_d)}$ | Surface wave at metal-dielectric interface |
| LSPR condition | $\text{Re}[\epsilon_m] = -2\epsilon_d$ | Nanoparticle resonance |
| Field enhancement | $|E/E_0|^2 \gg 1$ at resonance | Basis for SERS, biosensors |
| Negative index | $\epsilon < 0$ and $\mu < 0$ simultaneously | Reversed Snell's law |
| Wire array | $\epsilon_{\text{eff}} = 1 - \omega_{p,\text{eff}}^2/\omega^2$ | Artificial plasma |
| SRR | $\mu_{\text{eff}} = 1 - F\omega^2/(\omega^2 - \omega_0^2 + i\gamma\omega)$ | Magnetic resonance |
| Transformation optics | $\epsilon', \mu'$ from Jacobian | Coordinate-based material design |
| Photonic band gap | $\Delta\omega/\omega_0 \propto (n_H - n_L)/(n_H + n_L)$ | Forbidden frequency range |

---

## Exercises

### Exercise 1: SPP Excitation Design
Design a Kretschmann prism coupler to excite SPPs on a 50 nm silver film at $\lambda = 633$ nm (He-Ne laser). (a) Compute $k_{\text{SPP}}$ using the Drude model for silver. (b) Determine the required angle of incidence in a BK7 glass prism ($n = 1.515$). (c) Plot the angular reflectance spectrum (sharp dip at the SPP coupling angle). (d) How does the coupling angle change when a 10 nm protein layer ($n = 1.45$) is adsorbed on the silver surface?

### Exercise 2: Nanoparticle Shape Effects
A gold nanorod can be modeled as a prolate spheroid with aspect ratio $R = a/b$ (major/minor axis). The LSPR for the long axis occurs approximately when $\text{Re}[\epsilon_m] = -(1 + 1/L_a)\epsilon_d$, where $L_a$ is the depolarization factor along the long axis. (a) Compute $L_a$ for aspect ratios 1 (sphere), 2, 3, 4, and 5. (b) Plot the LSPR wavelength as a function of aspect ratio. (c) Explain why gold nanorods are tunable from visible to near-IR.

### Exercise 3: Metamaterial Band Structure
For a metamaterial with $\omega_{p,\text{eff}} = 12$ GHz, $\omega_0 = 8$ GHz, $\gamma = 0.3$ GHz, and $F = 0.5$, compute: (a) The frequency band where $n_{\text{eff}} < 0$. (b) The figure of merit FOM $= |n'|/n''$ as a function of frequency in the negative-index band. (c) What is the maximum FOM, and at what frequency does it occur?

### Exercise 4: Photonic Crystal Defect
Introduce a "defect" in a 1D photonic crystal by changing the thickness of one layer. Using the transfer matrix method, show that a sharp transmission peak appears within the band gap. (a) Compute the Q-factor of this defect mode as a function of the number of periods on each side. (b) How does the defect mode frequency change when the defect layer thickness is varied?

### Exercise 5: Cloaking Performance
Implement a 2D FDTD simulation of a cylindrical electromagnetic cloak. Use the simplified reduced-parameter cloak (only $\epsilon_r$ and $\epsilon_\theta$ vary, $\mu = 1$). (a) Simulate a plane wave scattering from an uncloaked PEC cylinder. (b) Add the cloak and observe the reduction in scattering. (c) Measure the total scattering cross section with and without the cloak. (d) How does the cloaking performance degrade as frequency deviates from the design frequency?

---

[← Previous: 17. Electromagnetic Scattering](17_Electromagnetic_Scattering.md) | [Next: Overview →](00_Overview.md)

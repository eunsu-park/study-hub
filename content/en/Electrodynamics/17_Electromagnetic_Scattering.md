# 17. Electromagnetic Scattering

[← Previous: 16. Computational Electrodynamics](16_Computational_Electrodynamics.md) | [Next: 18. Plasmonics and Metamaterials →](18_Plasmonics_and_Metamaterials.md)

## Learning Objectives

1. Formulate the electromagnetic scattering problem with incident, scattered, and total fields
2. Derive Rayleigh scattering for particles much smaller than the wavelength and explain why the sky is blue
3. Understand Mie theory for scattering by spherical particles of arbitrary size
4. Define and compute scattering, absorption, and extinction cross sections
5. Apply the Born approximation for weak scatterers and the optical theorem
6. Compute differential scattering cross sections and understand angular distributions
7. Connect scattering theory to practical applications in radar, lidar, and medical imaging

When an electromagnetic wave encounters an obstacle — a raindrop, a nanoparticle, an aircraft — part of the wave is redirected in various directions. This is **scattering**, and it shapes everything we see: the blue sky, the white clouds, the red sunset, the rainbow. Scattering also underpins technologies from radar (detecting aircraft) to lidar (atmospheric profiling) to medical imaging (optical coherence tomography). In this lesson, we develop the theory of electromagnetic scattering from first principles, starting with the simplest case (small particles, Rayleigh regime) and progressing to the exact solution for spheres (Mie theory). Along the way, we learn the language of cross sections and scattering amplitudes that connects theory to experiment.

> **Analogy**: Throw a tennis ball (the incident wave) at a basketball (the scatterer). The tennis ball bounces off in a direction that depends on where it hits. If you throw many tennis balls from the same direction, they scatter into a characteristic pattern around the basketball. The "scattering cross section" is the effective target area — it tells you how much of the incoming flux is redirected. Remarkably, for electromagnetic waves, the cross section can be much larger or smaller than the physical size of the scatterer, depending on the wavelength.

---

## 1. The Scattering Problem

### 1.1 Setup

An incident plane wave $\mathbf{E}_i = \mathbf{E}_0 e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$ impinges on a scatterer of finite extent. The total field everywhere is:

$$\mathbf{E}_{\text{total}} = \mathbf{E}_i + \mathbf{E}_s$$

where $\mathbf{E}_s$ is the **scattered field**. Far from the scatterer ($r \to \infty$), the scattered field has the form of an outgoing spherical wave:

$$\mathbf{E}_s \to f(\theta, \phi) \frac{e^{ikr}}{r} \hat{e}_s$$

The function $f(\theta, \phi)$ is the **scattering amplitude** — it encodes all the angular information about the scattering process.

### 1.2 Cross Sections

The **differential scattering cross section** is defined as the power scattered per unit solid angle, normalized by the incident intensity:

$$\frac{d\sigma_s}{d\Omega} = |f(\theta, \phi)|^2$$

The **total scattering cross section** integrates over all angles:

$$\sigma_s = \int |f(\theta, \phi)|^2 \, d\Omega$$

If the scatterer also absorbs energy, the **absorption cross section** $\sigma_a$ accounts for that, and the **extinction cross section** is:

$$\sigma_{\text{ext}} = \sigma_s + \sigma_a$$

The extinction cross section determines how much the incident beam is attenuated (both scattering and absorption).

### 1.3 Size Parameter

The physics of scattering depends critically on the ratio of particle size to wavelength, captured by the **size parameter**:

$$x = \frac{2\pi a}{\lambda} = ka$$

where $a$ is the particle radius. Three regimes emerge:

| Regime | Condition | Key Feature |
|--------|-----------|-------------|
| Rayleigh | $x \ll 1$ | Scattering $\propto \lambda^{-4}$, isotropic-ish |
| Mie (resonant) | $x \sim 1$ | Complex resonances, strong forward scattering |
| Geometric optics | $x \gg 1$ | Ray tracing, $\sigma \approx 2\pi a^2$ |

---

## 2. Rayleigh Scattering

### 2.1 Induced Dipole

When a small particle ($a \ll \lambda$) is placed in a uniform electric field, it develops an induced dipole moment:

$$\mathbf{p} = \epsilon_0 \alpha_E \mathbf{E}$$

where $\alpha_E$ is the electric polarizability. For a dielectric sphere of radius $a$ and relative permittivity $\epsilon_r$:

$$\alpha_E = 4\pi a^3 \frac{\epsilon_r - 1}{\epsilon_r + 2}$$

The factor $(\epsilon_r - 1)/(\epsilon_r + 2)$ is the **Clausius-Mossotti** factor.

### 2.2 Scattering Cross Section

The oscillating dipole radiates with the pattern derived in Lesson 13. The total scattered power gives:

$$\boxed{\sigma_{\text{Rayleigh}} = \frac{8\pi}{3}\left(\frac{2\pi}{\lambda}\right)^4 a^6 \left|\frac{\epsilon_r - 1}{\epsilon_r + 2}\right|^2 = \frac{128\pi^5}{3} \frac{a^6}{\lambda^4}\left|\frac{\epsilon_r - 1}{\epsilon_r + 2}\right|^2}$$

The crucial features are:
- **$\lambda^{-4}$ dependence**: Blue light ($\lambda \approx 450$ nm) scatters $(700/450)^4 \approx 5.7$ times more than red light ($\lambda \approx 700$ nm)
- **$a^6$ dependence**: Scattering is extremely sensitive to particle size

### 2.3 Why the Sky is Blue (and Sunsets are Red)

Sunlight entering the atmosphere encounters nitrogen and oxygen molecules ($a \sim 0.1$ nm, $\lambda \sim 500$ nm, so $x \sim 10^{-3}$). The $\lambda^{-4}$ law means blue light is scattered about 6 times more than red light. When you look at the sky (away from the sun), you see predominantly scattered blue light.

At sunset, the direct sunlight passes through a much thicker slice of atmosphere. The blue light has been scattered away, leaving the transmitted light enriched in red and orange.

```python
import numpy as np
import matplotlib.pyplot as plt

def rayleigh_cross_section(a, wavelength, eps_r):
    """
    Compute Rayleigh scattering cross section for a small sphere.

    Parameters:
        a          : sphere radius (m)
        wavelength : wavelength of light (m)
        eps_r      : relative permittivity of sphere

    Why Rayleigh: it's the simplest scattering theory and explains
    everyday phenomena like sky color, haze, and why fine particles
    scatter light so differently from large ones.
    """
    k = 2 * np.pi / wavelength
    cm = (eps_r - 1) / (eps_r + 2)
    return (128 * np.pi**5 / 3) * a**6 / wavelength**4 * np.abs(cm)**2

def rayleigh_differential(theta, polarization='unpolarized'):
    """
    Differential cross section pattern for Rayleigh scattering.

    For unpolarized incident light, the scattered intensity at angle theta is:
    dσ/dΩ ∝ (1 + cos²θ) / 2
    """
    if polarization == 'parallel':
        return np.cos(theta)**2
    elif polarization == 'perpendicular':
        return np.ones_like(theta)
    else:  # unpolarized
        return 0.5 * (1 + np.cos(theta)**2)

# Demonstrate λ^{-4} dependence
wavelengths = np.linspace(380, 780, 200) * 1e-9  # visible spectrum
a_N2 = 0.1e-9   # effective radius of N2 molecule
eps_r_air = 1.00029  # relative permittivity of air (approximately)

sigma_rayleigh = rayleigh_cross_section(a_N2, wavelengths, eps_r_air)

# Normalize to green (550 nm)
idx_green = np.argmin(np.abs(wavelengths - 550e-9))
sigma_norm = sigma_rayleigh / sigma_rayleigh[idx_green]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Spectrum of scattered light
ax = axes[0]
# Color mapping for visible spectrum
colors = plt.cm.rainbow(np.linspace(0, 1, len(wavelengths)))
for i in range(len(wavelengths) - 1):
    ax.fill_between(wavelengths[i:i+2] * 1e9, 0, sigma_norm[i:i+2],
                    color=colors[i], alpha=0.8)
ax.plot(wavelengths * 1e9, sigma_norm, 'k-', linewidth=1)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Relative scattering cross section')
ax.set_title('Rayleigh Scattering: $\\sigma \\propto \\lambda^{-4}$')
ax.grid(True, alpha=0.3)

# Annotate blue vs red
ax.annotate('Blue scatters\n~6x more', xy=(450, sigma_norm[np.argmin(np.abs(wavelengths - 450e-9))]),
            xytext=(500, 4), arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')

# Angular pattern
ax = axes[1]
theta = np.linspace(0, 2 * np.pi, 500)
r_unpol = rayleigh_differential(theta, 'unpolarized')
r_para = rayleigh_differential(theta, 'parallel')
r_perp = rayleigh_differential(theta, 'perpendicular')

ax_polar = fig.add_subplot(1, 3, 2, projection='polar')
axes[1].remove()
ax_polar.plot(theta, r_unpol, 'k-', linewidth=2, label='Unpolarized')
ax_polar.plot(theta, r_para, 'b--', linewidth=1.5, label='$\\parallel$ polarized')
ax_polar.plot(theta, r_perp, 'r:', linewidth=1.5, label='$\\perp$ polarized')
ax_polar.set_title('Rayleigh Angular Pattern', pad=20)
ax_polar.set_theta_zero_location('E')
ax_polar.legend(loc='lower right', fontsize=9)

# Sky color at different sun positions
ax = axes[2]
# Transmission through atmosphere of thickness L
# I/I_0 = exp(-n σ L) where n is number density
n_air = 2.5e25   # molecules/m^3 at sea level
L_zenith = 8000   # effective atmosphere height (m)

sun_angles = [0, 30, 60, 80, 85]  # degrees from zenith
for angle in sun_angles:
    L = L_zenith / max(np.cos(np.radians(angle)), 0.01)
    # Use approximate cross section for N2 (scaled)
    sigma_approx = 5e-31 * (550e-9 / wavelengths)**4
    transmission = np.exp(-n_air * sigma_approx * L)

    ax.plot(wavelengths * 1e9, transmission, linewidth=1.5,
            label=f'Zenith angle = {angle}°')

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Transmission')
ax.set_title('Direct Sunlight Through Atmosphere')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rayleigh_scattering.png", dpi=150)
plt.show()
```

---

## 3. Mie Theory

### 3.1 Exact Solution for Spheres

Mie theory (1908) provides the exact solution for scattering of a plane wave by a homogeneous sphere of arbitrary size. The incident, scattered, and internal fields are expanded in **vector spherical harmonics**:

$$\mathbf{E}_s = \sum_{n=1}^{\infty} E_n \left(i a_n \mathbf{N}_{e1n}^{(3)} - b_n \mathbf{M}_{o1n}^{(3)}\right)$$

where $\mathbf{M}$ and $\mathbf{N}$ are vector spherical wave functions, and $a_n$, $b_n$ are the **Mie coefficients**:

$$a_n = \frac{m\psi_n(mx)\psi_n'(x) - \psi_n(x)\psi_n'(mx)}{m\psi_n(mx)\xi_n'(x) - \xi_n(x)\psi_n'(mx)}$$

$$b_n = \frac{\psi_n(mx)\psi_n'(x) - m\psi_n(x)\psi_n'(mx)}{\psi_n(mx)\xi_n'(x) - m\xi_n(x)\psi_n'(mx)}$$

Here $m = n_{\text{sphere}}/n_{\text{medium}}$ is the relative refractive index, $x = ka$ is the size parameter, and $\psi_n$, $\xi_n$ are Riccati-Bessel functions.

### 3.2 Cross Sections from Mie Coefficients

$$\sigma_{\text{ext}} = \frac{2\pi}{k^2}\sum_{n=1}^{\infty}(2n+1)\,\text{Re}(a_n + b_n)$$

$$\sigma_s = \frac{2\pi}{k^2}\sum_{n=1}^{\infty}(2n+1)(|a_n|^2 + |b_n|^2)$$

$$\sigma_a = \sigma_{\text{ext}} - \sigma_s$$

### 3.3 Efficiency Factors

It is conventional to normalize by the geometric cross section $\pi a^2$:

$$Q_{\text{ext}} = \frac{\sigma_{\text{ext}}}{\pi a^2}, \quad Q_s = \frac{\sigma_s}{\pi a^2}, \quad Q_a = \frac{\sigma_a}{\pi a^2}$$

Remarkably, $Q_{\text{ext}}$ can exceed 2, meaning the particle "shadows" more area than its physical size. This is the **extinction paradox**, resolved by understanding that diffraction around the particle also removes energy from the forward beam.

```python
from scipy.special import spherical_jn, spherical_yn

def riccati_bessel_jn(n, x):
    """Riccati-Bessel function psi_n(x) = x * j_n(x)."""
    return x * spherical_jn(n, x)

def riccati_bessel_jn_deriv(n, x):
    """Derivative of psi_n(x)."""
    return spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)

def riccati_bessel_hn(n, x):
    """Riccati-Bessel function xi_n(x) = x * h_n^(1)(x)."""
    return x * (spherical_jn(n, x) + 1j * spherical_yn(n, x))

def riccati_bessel_hn_deriv(n, x):
    """Derivative of xi_n(x)."""
    jn = spherical_jn(n, x)
    jn_d = spherical_jn(n, x, derivative=True)
    yn = spherical_yn(n, x)
    yn_d = spherical_yn(n, x, derivative=True)
    return (jn + 1j * yn) + x * (jn_d + 1j * yn_d)

def mie_coefficients(m, x, n_max=None):
    """
    Compute Mie scattering coefficients a_n and b_n.

    Parameters:
        m     : relative refractive index (complex)
        x     : size parameter (2*pi*a/lambda)
        n_max : number of terms (default: x + 4*x^(1/3) + 2)

    Why Mie theory: it's the exact solution for spheres, serving as
    the benchmark for approximate methods and the foundation for
    understanding scattering by non-spherical particles.
    """
    if n_max is None:
        n_max = int(x + 4 * x**(1/3) + 2)
    n_max = max(n_max, 3)

    mx = m * x
    a_n = np.zeros(n_max, dtype=complex)
    b_n = np.zeros(n_max, dtype=complex)

    for n in range(1, n_max + 1):
        psi_mx = riccati_bessel_jn(n, mx)
        psi_mx_d = riccati_bessel_jn_deriv(n, mx)
        psi_x = riccati_bessel_jn(n, x)
        psi_x_d = riccati_bessel_jn_deriv(n, x)
        xi_x = riccati_bessel_hn(n, x)
        xi_x_d = riccati_bessel_hn_deriv(n, x)

        a_n[n-1] = (m * psi_mx * psi_x_d - psi_x * psi_mx_d) / \
                    (m * psi_mx * xi_x_d - xi_x * psi_mx_d)
        b_n[n-1] = (psi_mx * psi_x_d - m * psi_x * psi_mx_d) / \
                    (psi_mx * xi_x_d - m * xi_x * psi_mx_d)

    return a_n, b_n

def mie_cross_sections(m, x):
    """Compute Mie scattering, extinction, and absorption efficiencies."""
    a_n, b_n = mie_coefficients(m, x)
    n_terms = len(a_n)
    n_arr = np.arange(1, n_terms + 1)

    Q_ext = (2 / x**2) * np.sum((2 * n_arr + 1) * np.real(a_n + b_n))
    Q_sca = (2 / x**2) * np.sum((2 * n_arr + 1) * (np.abs(a_n)**2 + np.abs(b_n)**2))
    Q_abs = Q_ext - Q_sca

    return Q_ext, Q_sca, Q_abs


def plot_mie_efficiency():
    """
    Plot Mie efficiency factors vs size parameter for different
    refractive indices.
    """
    x_range = np.linspace(0.01, 30, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Different refractive indices
    indices = [
        (1.33 + 0j, 'Water droplet ($n$ = 1.33)'),
        (1.50 + 0j, 'Glass sphere ($n$ = 1.50)'),
        (1.50 + 0.1j, 'Absorbing sphere ($n$ = 1.50 + 0.1i)'),
        (2.00 + 1.00j, 'Metallic sphere ($n$ = 2.0 + 1.0i)')
    ]

    for ax, (m, label) in zip(axes.flat, indices):
        Q_ext_arr = np.zeros(len(x_range))
        Q_sca_arr = np.zeros(len(x_range))
        Q_abs_arr = np.zeros(len(x_range))

        for i, x in enumerate(x_range):
            if x < 0.05:
                continue
            try:
                Q_ext_arr[i], Q_sca_arr[i], Q_abs_arr[i] = mie_cross_sections(m, x)
            except (ValueError, ZeroDivisionError):
                pass

        ax.plot(x_range, Q_ext_arr, 'k-', linewidth=2, label='$Q_{\\mathrm{ext}}$')
        ax.plot(x_range, Q_sca_arr, 'b-', linewidth=1.5, label='$Q_{\\mathrm{sca}}$')
        if np.any(Q_abs_arr > 0.01):
            ax.plot(x_range, Q_abs_arr, 'r--', linewidth=1.5, label='$Q_{\\mathrm{abs}}$')
        ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='$Q_{\\mathrm{ext}} = 2$ (limit)')
        ax.set_xlabel('Size parameter $x = 2\\pi a / \\lambda$')
        ax.set_ylabel('Efficiency $Q$')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(5, 1.1 * np.max(Q_ext_arr)))

    plt.suptitle('Mie Scattering Efficiency vs Size Parameter', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("mie_efficiency.png", dpi=150)
    plt.show()

plot_mie_efficiency()
```

---

## 4. The Born Approximation

### 4.1 Weak Scattering

When the scatterer is "weak" (small contrast in refractive index, or thin), we can approximate the total field inside the scatterer by the incident field. This is the **Born approximation**:

$$\mathbf{E}_s(\mathbf{r}) \approx \frac{k^2}{4\pi}\int \Delta\epsilon_r(\mathbf{r}') \, \mathbf{E}_i(\mathbf{r}') \frac{e^{ik|\mathbf{r}-\mathbf{r}'|}}{|\mathbf{r}-\mathbf{r}'|} \, d^3r'$$

where $\Delta\epsilon_r = \epsilon_r - 1$ is the dielectric contrast.

### 4.2 Far-Field Born Approximation

In the far field, $|\mathbf{r} - \mathbf{r}'| \approx r - \hat{r}\cdot\mathbf{r}'$, giving:

$$f(\theta, \phi) \propto \int \Delta\epsilon_r(\mathbf{r}') \, e^{-i\mathbf{q}\cdot\mathbf{r}'} \, d^3r'$$

where $\mathbf{q} = \mathbf{k}_s - \mathbf{k}_i$ is the **scattering vector** ($|\mathbf{q}| = 2k\sin(\theta/2)$ for elastic scattering). The scattering amplitude is the **Fourier transform** of the dielectric contrast evaluated at $\mathbf{q}$.

This powerful result connects scattering measurements to the spatial structure of the scatterer — the basis for X-ray crystallography, radar imaging, and inverse scattering.

### 4.3 Validity

The Born approximation is valid when:

$$|\Delta\epsilon_r| \cdot k \cdot a \ll 1$$

That is, the phase shift accumulated through the scatterer must be small.

```python
def born_scattering_cross_section(k, a, delta_eps):
    """
    Compute the Born approximation scattering cross section for
    a homogeneous dielectric sphere.

    The result is:
    σ = (k^4 / 6π) * |delta_eps|^2 * V^2 * [3(sin(qa) - qa*cos(qa))/(qa)^3]^2

    averaged over angles, where V = 4πa^3/3.

    Why Born approximation: it gives analytical insight into scattering
    by arbitrary shapes, since the cross section is essentially the
    Fourier transform of the object's shape.
    """
    theta = np.linspace(0.001, np.pi, 500)
    q = 2 * k * np.sin(theta / 2)

    # Form factor for a sphere: F(q) = 3[sin(qa) - qa*cos(qa)] / (qa)^3
    qa = q * a
    F = np.where(qa < 0.01, 1.0 - qa**2/10,
                  3 * (np.sin(qa) - qa * np.cos(qa)) / qa**3)

    V = 4 * np.pi * a**3 / 3

    # Differential cross section
    dsigma = (k**4 / (4 * np.pi)**2) * np.abs(delta_eps)**2 * V**2 * F**2

    # Total cross section (integrate over solid angle)
    sigma_total = 2 * np.pi * np.trapz(dsigma * np.sin(theta), theta)

    return theta, dsigma, sigma_total


def compare_born_vs_mie():
    """Compare Born approximation with exact Mie theory."""
    # Weak scatterer: n = 1.05 (delta_eps = n^2 - 1 ≈ 0.1025)
    n_sphere = 1.05
    m = n_sphere
    delta_eps = n_sphere**2 - 1

    x_values = np.linspace(0.1, 10, 50)
    wavelength = 500e-9  # 500 nm

    Q_mie_arr = np.zeros(len(x_values))
    Q_born_arr = np.zeros(len(x_values))

    for i, x in enumerate(x_values):
        a = x * wavelength / (2 * np.pi)
        k = 2 * np.pi / wavelength

        # Mie
        _, Q_mie_arr[i], _ = mie_cross_sections(m, x)

        # Born
        _, _, sigma_born = born_scattering_cross_section(k, a, delta_eps)
        Q_born_arr[i] = sigma_born / (np.pi * a**2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, Q_mie_arr, 'b-', linewidth=2, label='Mie (exact)')
    ax.plot(x_values, Q_born_arr, 'r--', linewidth=2, label='Born approximation')
    ax.set_xlabel('Size parameter $x$')
    ax.set_ylabel('Scattering efficiency $Q_{\\mathrm{sca}}$')
    ax.set_title(f'Born vs Mie: Weak Scatterer ($n$ = {n_sphere})')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Annotate validity region
    ax.axvspan(0, 2, alpha=0.1, color='green')
    ax.text(1, ax.get_ylim()[1] * 0.9, 'Born valid', fontsize=11,
            ha='center', color='green')

    plt.tight_layout()
    plt.savefig("born_vs_mie.png", dpi=150)
    plt.show()

compare_born_vs_mie()
```

---

## 5. The Optical Theorem

### 5.1 Statement

The **optical theorem** relates the total extinction cross section to the forward scattering amplitude:

$$\boxed{\sigma_{\text{ext}} = \frac{4\pi}{k}\,\text{Im}[f(0)]}$$

where $f(0)$ is the scattering amplitude in the exact forward direction ($\theta = 0$).

### 5.2 Physical Meaning

The forward-scattered wave interferes destructively with the incident wave, creating a "shadow" behind the scatterer. The amount of power removed from the incident beam (extinction) is entirely determined by this forward interference — regardless of how the scattered power is distributed in angle.

### 5.3 Consequences

- Extinction is always at least as large as absorption: $\sigma_{\text{ext}} \geq \sigma_a$
- For a purely scattering particle (no absorption), $\sigma_{\text{ext}} = \sigma_s$
- In the large-particle limit ($x \to \infty$), $\sigma_{\text{ext}} \to 2\pi a^2$ — twice the geometric cross section (the extinction paradox)

---

## 6. Angular Scattering Patterns

### 6.1 Small Particles (Rayleigh Regime)

For $x \ll 1$, the scattering is nearly symmetric between forward and backward:

$$\frac{d\sigma}{d\Omega} \propto (1 + \cos^2\theta)$$

### 6.2 Large Particles (Mie Regime)

As $x$ increases, the scattering becomes increasingly peaked in the forward direction. The forward lobe narrows as $\Delta\theta \sim 1/x \sim \lambda/(2\pi a)$, and complex interference fringes appear at larger angles.

```python
def mie_scattering_pattern(m, x, theta):
    """
    Compute Mie angular scattering pattern (intensity functions S1, S2).

    Why angular patterns: they are directly measurable in experiments
    (nephelometers, goniometers) and provide information about
    particle size, shape, and refractive index.
    """
    from scipy.special import lpmv

    a_n, b_n = mie_coefficients(m, x)
    n_max = len(a_n)
    cos_theta = np.cos(theta)

    S1 = np.zeros(len(theta), dtype=complex)
    S2 = np.zeros(len(theta), dtype=complex)

    for n in range(1, n_max + 1):
        # Angular functions pi_n and tau_n
        # pi_n = P_n^1 / sin(theta)
        # tau_n = d/d(theta) P_n^1

        P1 = lpmv(1, n, cos_theta)
        sin_theta = np.sin(theta)
        sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
        pi_n = P1 / sin_theta
        # tau_n requires numerical derivative or recursion
        # Simple finite difference for tau_n
        dtheta = 1e-6
        P1_plus = lpmv(1, n, np.cos(theta + dtheta))
        tau_n = -(P1_plus - P1) / dtheta

        prefactor = (2 * n + 1) / (n * (n + 1))
        S1 += prefactor * (a_n[n-1] * pi_n + b_n[n-1] * tau_n)
        S2 += prefactor * (a_n[n-1] * tau_n + b_n[n-1] * pi_n)

    return np.abs(S1)**2, np.abs(S2)**2


def plot_angular_patterns():
    """Plot angular scattering patterns for different size parameters."""
    theta = np.linspace(0.01, np.pi, 500)
    m = 1.33  # water droplet

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    x_values = [0.1, 1.0, 5.0, 20.0]

    for ax, x in zip(axes.flat, x_values):
        try:
            S1, S2 = mie_scattering_pattern(m, x, theta)
            S_unpol = 0.5 * (S1 + S2)

            ax_polar = fig.add_subplot(2, 2, list(axes.flat).index(ax) + 1,
                                       projection='polar')
            ax.remove()

            # Use log scale for pattern
            pattern_db = 10 * np.log10(S_unpol / S_unpol.max() + 1e-10)
            pattern_plot = np.clip(pattern_db + 40, 0, 40)  # shift for plotting

            ax_polar.plot(theta, pattern_plot, 'b-', linewidth=1.5)
            ax_polar.plot(-theta + 2*np.pi, pattern_plot, 'b-', linewidth=1.5)
            ax_polar.set_title(f'$x$ = {x} ($a/\\lambda$ = {x/(2*np.pi):.3f})',
                              pad=15)
            ax_polar.set_theta_zero_location('E')

        except Exception:
            ax.text(0.5, 0.5, f'x = {x}\n(computation error)',
                    ha='center', va='center', transform=ax.transAxes)

    plt.suptitle(f'Mie Scattering Angular Pattern (water, $n$ = {m})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("mie_angular_patterns.png", dpi=150)
    plt.show()

plot_angular_patterns()
```

---

## 7. Applications

### 7.1 Radar

Radar works by transmitting microwave pulses and detecting the backscattered signal ($\theta = \pi$). The **radar cross section** (RCS) is:

$$\sigma_{\text{RCS}} = 4\pi r^2 \frac{|\mathbf{E}_s|^2}{|\mathbf{E}_i|^2}\bigg|_{\theta=\pi}$$

Stealth aircraft are designed to minimize RCS through geometry (angled surfaces) and materials (radar-absorbing coatings).

### 7.2 Lidar and Atmospheric Remote Sensing

Lidar uses short laser pulses to profile the atmosphere. Backscattered light carries information about:
- **Aerosol concentration** (Mie scattering from particles)
- **Molecular density** (Rayleigh scattering from air molecules)
- **Wind speed** (Doppler shift of backscattered light)
- **Temperature** (broadening of the Rayleigh line)

### 7.3 Medical Imaging

Optical coherence tomography (OCT) uses near-IR light scattering to image tissue at micrometer resolution. The forward-scattering regime ($x \sim 1-10$ for cells) determines the imaging depth and contrast.

### 7.4 Nanoparticle Characterization

Dynamic light scattering (DLS) measures the size distribution of nanoparticles in solution by analyzing the temporal fluctuations of scattered light intensity. Particles diffuse at rates inversely proportional to their size (Stokes-Einstein relation), causing intensity fluctuations that are decoded by autocorrelation analysis.

---

## Summary

| Concept | Key Formula | Physical Meaning |
|---------|-------------|------------------|
| Size parameter | $x = 2\pi a / \lambda$ | Determines scattering regime |
| Rayleigh cross section | $\sigma \propto a^6 / \lambda^4$ | Small particles; blue sky |
| Mie coefficients | $a_n, b_n$ from Riccati-Bessel functions | Exact sphere solution |
| Extinction efficiency | $Q_{\text{ext}} = \sigma_{\text{ext}} / \pi a^2$ | Can exceed 2 (paradox) |
| Born approximation | $f \propto \text{FT}[\Delta\epsilon_r]$ at $\mathbf{q}$ | Weak scatterer; Fourier relation |
| Optical theorem | $\sigma_{\text{ext}} = (4\pi/k)\,\text{Im}[f(0)]$ | Forward scattering determines extinction |
| Forward peaking | $\Delta\theta \sim \lambda / (2\pi a)$ | Large particles scatter forward |

---

## Exercises

### Exercise 1: Sunset Simulation
Model the transmission of sunlight through the atmosphere as a function of solar zenith angle. Use Rayleigh scattering cross sections for N$_2$ and O$_2$, with the atmosphere modeled as an exponential density profile ($n(h) = n_0 e^{-h/H}$, $H = 8.5$ km). Plot the transmitted spectrum for zenith angles 0, 45, 70, 85, and 90 degrees. Compute the "color temperature" of the transmitted light at each angle.

### Exercise 2: Mie Resonances
Plot $Q_{\text{ext}}$ for a glass sphere ($n = 1.5$) as a function of size parameter from $x = 0$ to $x = 50$. Identify the resonance peaks and explain their origin in terms of the Mie coefficients $a_n$ and $b_n$. Which multipole order $n$ dominates each resonance?

### Exercise 3: Cloud Opacity
Cloud droplets have a typical radius of 10 $\mu$m and concentration $\sim 300$ cm$^{-3}$. (a) Compute $Q_{\text{ext}}$ for visible light ($\lambda = 550$ nm) using Mie theory ($n = 1.33$). (b) Calculate the extinction coefficient $\alpha = n_{\text{droplet}} \sigma_{\text{ext}}$. (c) What is the optical depth of a 1-km-thick cloud? Is this consistent with clouds being visually opaque?

### Exercise 4: Born Approximation Validity
For a sphere with $n = 1.05$, compare Born and Mie cross sections as a function of $x$ from 0 to 20. At what $x$ does the Born approximation error exceed 10%? Repeat for $n = 1.2$ and $n = 1.5$. Verify the rule of thumb $|\Delta\epsilon_r| \cdot x \ll 1$.

### Exercise 5: Radar Cross Section
A metallic sphere of radius 1 m is illuminated by a 10 GHz radar. (a) Compute the size parameter. (b) Calculate $Q_{\text{ext}}$ using Mie theory with $n = 10 + 10i$ (approximate metal). (c) Estimate the RCS. (d) The radar equation gives the received power as $P_r = P_t G^2 \lambda^2 \sigma / (4\pi)^3 R^4$. For $P_t = 1$ MW, $G = 40$ dBi, $R = 100$ km, compute $P_r$.

---

[← Previous: 16. Computational Electrodynamics](16_Computational_Electrodynamics.md) | [Next: 18. Plasmonics and Metamaterials →](18_Plasmonics_and_Metamaterials.md)

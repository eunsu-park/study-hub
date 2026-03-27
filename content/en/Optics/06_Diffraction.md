# 06. Diffraction

[← Previous: 05. Wave Optics — Interference](05_Wave_Optics_Interference.md) | [Next: 07. Polarization →](07_Polarization.md)

---

## Learning Objectives

1. State the Huygens-Fresnel principle and explain how it accounts for diffraction phenomena
2. Derive and analyze the Fraunhofer diffraction pattern of a single slit, including the intensity formula and minima conditions
3. Calculate the Airy pattern for a circular aperture and apply it to determine the resolution limit of optical systems
4. Analyze diffraction gratings — their resolving power, free spectral range, and use in spectroscopy
5. Distinguish between Fraunhofer (far-field) and Fresnel (near-field) diffraction and identify the Fresnel number as the criterion
6. Explain how diffraction sets the ultimate resolution limit of all optical systems
7. Apply diffraction theory to practical systems: spectrometers, X-ray crystallography, and acoustic analogs

---

## Why This Matters

Diffraction is the reason you cannot simply magnify forever to see smaller and smaller details. It is the fundamental resolution barrier of every optical system — from your smartphone camera to the Hubble Space Telescope to a lithography machine printing transistors on silicon. Understanding diffraction is essential for designing spectrometers (which separate wavelengths using diffraction gratings), for interpreting X-ray crystallography data (which reveals molecular structures through diffraction), and for appreciating why we need electron microscopes or near-field techniques to image features smaller than the wavelength of light.

> **Analogy**: Imagine dropping a pebble into a pond near a wall with a narrow gap. The wave does not simply pass straight through the gap like a bullet through a hole — instead, it spreads out in a semicircular pattern on the other side. The narrower the gap relative to the wavelength, the more the wave spreads. This is diffraction. Light does the same thing: when it passes through an aperture comparable to its wavelength, it refuses to travel in straight lines.

---

## 1. The Huygens-Fresnel Principle

### 1.1 Huygens' Construction (1690)

Christiaan Huygens proposed that every point on a wavefront acts as a source of secondary spherical wavelets. The new wavefront at a later time is the envelope (tangent surface) of all these wavelets.

### 1.2 Fresnel's Enhancement (1818)

Fresnel made Huygens' principle quantitative by adding two crucial ingredients:

1. **Superposition with phase**: Each wavelet carries a phase determined by the distance it has traveled
2. **Obliquity factor**: An inclination factor $K(\chi) = \frac{1}{2}(1 + \cos\chi)$ that suppresses backward propagation

The Huygens-Fresnel integral gives the field at point $P$ from an aperture $\Sigma$:

$$E(P) = -\frac{i}{\lambda} \iint_\Sigma E(Q) \frac{e^{ikr}}{r} K(\chi) \, dA$$

where $Q$ is a point on the aperture, $r$ is the distance from $Q$ to $P$, and $K(\chi)$ is the obliquity factor.

### 1.3 Fraunhofer vs. Fresnel Diffraction

The classification depends on the **Fresnel number**:

$$N_F = \frac{a^2}{\lambda L}$$

where $a$ is the aperture size and $L$ is the observation distance.

| Regime | Fresnel Number | Characteristics |
|--------|---------------|-----------------|
| Fresnel (near-field) | $N_F \geq 1$ | Complex patterns, phase curvature matters |
| Fraunhofer (far-field) | $N_F \ll 1$ | Simpler patterns, equivalent to Fourier transform |

**Fraunhofer condition**: $L \gg a^2/\lambda$. For a 1 mm slit with visible light ($\lambda = 500$ nm): $L \gg 2$ m. A lens at the aperture can bring the Fraunhofer pattern to its focal plane, regardless of distance — this is the standard configuration for spectrometers.

---

## 2. Single-Slit Fraunhofer Diffraction

### 2.1 Setup and Geometry

A plane wave illuminates a slit of width $a$. We observe the intensity pattern at a distant screen (or at the focal plane of a lens).

Using the Huygens-Fresnel integral with the Fraunhofer approximation, the electric field amplitude at angle $\theta$:

$$E(\theta) = E_0 \frac{\sin\beta}{\beta}$$

where:

$$\beta = \frac{\pi a \sin\theta}{\lambda}$$

The function $\text{sinc}(\beta/\pi) = \sin(\beta)/\beta$ is the **sinc function** — one of the most important functions in optics and signal processing.

### 2.2 Intensity Pattern

The intensity is proportional to $|E|^2$:

$$I(\theta) = I_0 \left(\frac{\sin\beta}{\beta}\right)^2 = I_0 \,\text{sinc}^2\left(\frac{a\sin\theta}{\lambda}\right)$$

**Minima** (zeros of intensity) occur when:

$$a\sin\theta = m\lambda, \qquad m = \pm 1, \pm 2, \pm 3, \ldots$$

Note: $m = 0$ gives the central maximum, not a minimum.

**Central maximum**: The angular half-width of the central peak is:

$$\Delta\theta = \frac{\lambda}{a}$$

This is the full width from the first minimum on one side to the first minimum on the other side: $2\lambda/a$.

### 2.3 Secondary Maxima

The secondary maxima occur approximately at $\beta \approx (m + \frac{1}{2})\pi$ for $m = 1, 2, 3, \ldots$

| Maximum | Position $a\sin\theta/\lambda$ | Relative Intensity |
|---------|-------------------------------|-------------------|
| Central | 0 | 1.000 |
| 1st secondary | $\approx 1.43$ | 0.0472 (4.72%) |
| 2nd secondary | $\approx 2.46$ | 0.0165 (1.65%) |
| 3rd secondary | $\approx 3.47$ | 0.0083 (0.83%) |

The secondary maxima drop off rapidly — the central maximum contains about 90% of the total diffracted energy.

```python
import numpy as np
import matplotlib.pyplot as plt

# Single-slit Fraunhofer diffraction: intensity pattern
# The sinc^2 function is the Fourier transform of a rectangular aperture

def single_slit_intensity(theta, a, wavelength):
    """
    Calculate normalized single-slit diffraction intensity.
    theta: angle (radians)
    a: slit width (meters)
    wavelength: wavelength (meters)
    Returns: I/I_0
    """
    beta = np.pi * a * np.sin(theta) / wavelength
    # Handle the singularity at beta = 0 (central maximum)
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.where(np.abs(beta) < 1e-10, 1.0, np.sin(beta) / beta)
    return sinc**2

wavelength = 550e-9  # green light (m)
slit_widths = [5e-6, 20e-6, 100e-6]  # 5 um, 20 um, 100 um

theta = np.linspace(-0.15, 0.15, 2000)  # angle in radians

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: linear scale — shows the narrow central peak and secondary maxima
for a in slit_widths:
    I = single_slit_intensity(theta, a, wavelength)
    ax1.plot(np.rad2deg(theta), I, linewidth=1.5,
             label=f'a = {a*1e6:.0f} μm')

ax1.set_xlabel('Angle (degrees)', fontsize=12)
ax1.set_ylabel('Normalized Intensity $I/I_0$', fontsize=12)
ax1.set_title('Single-Slit Diffraction (Linear Scale)', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-8, 8)

# Right: log scale — reveals the secondary maxima structure
for a in slit_widths:
    I = single_slit_intensity(theta, a, wavelength)
    I_db = 10 * np.log10(np.maximum(I, 1e-10))  # convert to dB
    ax2.plot(np.rad2deg(theta), I_db, linewidth=1.5,
             label=f'a = {a*1e6:.0f} μm')

ax2.set_xlabel('Angle (degrees)', fontsize=12)
ax2.set_ylabel('Intensity (dB)', fontsize=12)
ax2.set_title('Single-Slit Diffraction (Log Scale)', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-8, 8)
ax2.set_ylim(-40, 2)

plt.tight_layout()
plt.savefig('single_slit_diffraction.png', dpi=150)
plt.show()
```

---

## 3. Circular Aperture: The Airy Pattern

### 3.1 The Airy Disk

For a circular aperture of diameter $D$, the Fraunhofer diffraction pattern is circularly symmetric and described by:

$$I(\theta) = I_0 \left[\frac{2J_1(x)}{x}\right]^2$$

where $x = \frac{\pi D\sin\theta}{\lambda}$ and $J_1$ is the first-order Bessel function of the first kind.

The central bright disk is called the **Airy disk**, and the surrounding rings are **Airy rings**.

### 3.2 Key Features

**First dark ring** (first zero of $J_1$) at $x = 3.8317$:

$$\sin\theta_1 = 1.22\frac{\lambda}{D}$$

This is the basis of the **Rayleigh criterion** (see Lesson 04). The angular radius of the Airy disk determines the diffraction-limited resolution of any circular-aperture optical system.

**Airy disk radius** on a focal plane:

$$r_{\text{Airy}} = 1.22\frac{\lambda f}{D} = 1.22\lambda N$$

where $N = f/D$ is the f-number.

### 3.3 Encircled Energy

The Airy pattern concentrates most energy in the central disk:

| Feature | Encircled Energy |
|---------|-----------------|
| Airy disk (to 1st dark ring) | 83.8% |
| To 2nd dark ring | 91.0% |
| To 3rd dark ring | 93.8% |

The remaining $\sim 16\%$ is spread across the Airy rings, causing a faint halo around bright point sources.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# Airy pattern: diffraction pattern of a circular aperture
# This is the point spread function (PSF) of any diffraction-limited telescope or camera

def airy_pattern(x):
    """
    Normalized Airy pattern: [2*J1(x)/x]^2
    x = pi * D * sin(theta) / lambda
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(x) < 1e-10, 1.0, (2 * j1(x) / x)**2)
    return result

# 1D profile
x = np.linspace(-15, 15, 1000)
I_airy = airy_pattern(x)

# 2D pattern
xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
rr = np.sqrt(xx**2 + yy**2)
I_2d = airy_pattern(rr)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Left: 1D intensity profile (linear)
ax1.plot(x, I_airy, 'b-', linewidth=2)
ax1.set_xlabel('$x = \\pi D \\sin\\theta / \\lambda$', fontsize=12)
ax1.set_ylabel('$I / I_0$', fontsize=12)
ax1.set_title('Airy Pattern (1D Profile)', fontsize=13)
ax1.axvline(3.83, color='red', linestyle='--', alpha=0.5, label='1st zero (3.83)')
ax1.axvline(-3.83, color='red', linestyle='--', alpha=0.5)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Middle: 2D pattern (log scale to show rings)
im = ax2.imshow(np.log10(np.maximum(I_2d, 1e-6)), extent=[-10, 10, -10, 10],
                cmap='inferno', vmin=-4, vmax=0)
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$y$', fontsize=12)
ax2.set_title('Airy Pattern (2D, Log Scale)', fontsize=13)
# Draw circles at the dark rings
for zero in [3.83, 7.02, 10.17]:
    circle = plt.Circle((0, 0), zero, fill=False, color='white', linewidth=0.5, linestyle='--')
    ax2.add_patch(circle)
plt.colorbar(im, ax=ax2, label='$\\log_{10}(I/I_0)$')

# Right: encircled energy as a function of radius
r_values = np.linspace(0, 15, 500)
# Numerically integrate the Airy pattern in 2D to get encircled energy
# E(r) = integral of I(rho) * 2*pi*rho from 0 to r
dr = r_values[1] - r_values[0]
rho = np.linspace(0, 15, 2000)
I_rho = airy_pattern(rho)
# Cumulative integral: E(r) = integral_0^r I(rho) * rho * d(rho) (normalized)
encircled = np.cumsum(I_rho * rho * (rho[1] - rho[0]))
encircled = encircled / encircled[-1]  # normalize to 1

ax3.plot(rho, encircled * 100, 'b-', linewidth=2)
ax3.axhline(83.8, color='r', linestyle='--', alpha=0.5, label='83.8% (Airy disk)')
ax3.axvline(3.83, color='r', linestyle=':', alpha=0.5)
ax3.set_xlabel('Radius $x = \\pi D \\sin\\theta / \\lambda$', fontsize=12)
ax3.set_ylabel('Encircled Energy (%)', fontsize=12)
ax3.set_title('Encircled Energy', fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 15)
ax3.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('airy_pattern.png', dpi=150)
plt.show()
```

---

## 4. Diffraction Grating

### 4.1 Multiple-Slit Diffraction

A diffraction grating consists of $N$ parallel, equally spaced slits (or grooves), separated by distance $d$ (the **grating period** or **grating spacing**). The intensity pattern combines single-slit diffraction with multi-slit interference:

$$I(\theta) = I_0 \left(\frac{\sin\beta}{\beta}\right)^2 \left(\frac{\sin(N\gamma)}{\sin\gamma}\right)^2$$

where:
- $\beta = \frac{\pi a\sin\theta}{\lambda}$ (single-slit diffraction envelope, $a$ = slit width)
- $\gamma = \frac{\pi d\sin\theta}{\lambda}$ (inter-slit interference)

The term $(\sin N\gamma / \sin\gamma)^2$ produces sharp **principal maxima** at:

$$d\sin\theta = m\lambda, \qquad m = 0, \pm 1, \pm 2, \ldots$$

Between adjacent principal maxima there are $N - 2$ secondary maxima and $N - 1$ minima.

### 4.2 Grating Characteristics

**Angular dispersion** — how much the angle changes per unit wavelength:

$$\frac{d\theta}{d\lambda} = \frac{m}{d\cos\theta}$$

Higher order $m$ and smaller grating period $d$ give greater dispersion.

**Resolving power** — the ability to distinguish two closely spaced wavelengths:

$$\mathcal{R} = \frac{\lambda}{\Delta\lambda} = mN$$

where $m$ is the diffraction order and $N$ is the total number of grooves illuminated. A grating with 1000 grooves/mm, illuminated over a 5 cm width, has $N = 50{,}000$. In the first order: $\mathcal{R} = 50{,}000$, meaning it can resolve wavelength differences of $\Delta\lambda = 550/50{,}000 = 0.011$ nm.

**Free spectral range** — the wavelength range in a given order before overlap with the next order:

$$\Delta\lambda_{\text{FSR}} = \frac{\lambda}{m}$$

### 4.3 Blazed Gratings

A simple grating wastes most light in the zero-order ($m = 0$), which provides no spectral information. A **blazed grating** has grooves tilted at an angle $\theta_b$ (the blaze angle), redirecting the single-slit diffraction envelope to peak at a desired diffraction order. This dramatically increases efficiency in that order.

The blaze condition:

$$d(\sin\theta_i + \sin\theta_m) = m\lambda \quad \text{and} \quad \theta_b = \frac{\theta_i + \theta_m}{2}$$

For the Littrow configuration ($\theta_i = \theta_m$): $2d\sin\theta_b = m\lambda$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Diffraction grating: intensity pattern showing principal maxima
# and the single-slit envelope

def grating_intensity(theta, a, d, N, wavelength):
    """
    Calculate the intensity pattern of an N-slit diffraction grating.

    theta: angle (radians)
    a: slit width (m)
    d: grating period / slit spacing (m)
    N: number of slits
    wavelength: wavelength (m)

    Returns: I / I_0 (normalized to single-slit central maximum)
    """
    # Single-slit diffraction factor
    beta = np.pi * a * np.sin(theta) / wavelength
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_factor = np.where(np.abs(beta) < 1e-12, 1.0, np.sin(beta) / beta)

    # Multi-slit interference factor
    gamma = np.pi * d * np.sin(theta) / wavelength
    with np.errstate(divide='ignore', invalid='ignore'):
        # sin(N*gamma) / sin(gamma) → N when gamma → m*pi
        array_factor = np.where(
            np.abs(np.sin(gamma)) < 1e-12,
            N * np.cos(N * gamma) / np.cos(gamma),  # L'Hopital's rule
            np.sin(N * gamma) / np.sin(gamma)
        )

    return (sinc_factor * array_factor / N)**2 * N**2

wavelength = 550e-9    # green light
d = 2e-6               # grating period: 2 um (500 grooves/mm)
a = d * 0.4            # slit width: 40% of period

theta = np.linspace(-0.3, 0.3, 10000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Show patterns for different numbers of slits
N_values = [2, 5, 20, 100]

for ax, N in zip(axes.flat, N_values):
    I = grating_intensity(theta, a, d, N, wavelength)
    I_envelope = grating_intensity(theta, a, a, 1, wavelength)  # single-slit envelope

    ax.plot(np.rad2deg(theta), I / I.max(), 'b-', linewidth=1,
            label=f'N = {N} slits')
    ax.plot(np.rad2deg(theta), I_envelope / I_envelope.max(), 'r--', linewidth=1,
            alpha=0.5, label='Single-slit envelope')

    ax.set_xlabel('Angle (degrees)', fontsize=11)
    ax.set_ylabel('Normalized Intensity', fontsize=11)
    ax.set_title(f'N = {N} slits, d = {d*1e6:.1f} μm', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.02, 1.1)

    # Mark diffraction orders
    for m in range(-3, 4):
        theta_m = np.arcsin(m * wavelength / d) if abs(m * wavelength / d) < 1 else None
        if theta_m is not None:
            ax.axvline(np.rad2deg(theta_m), color='gray', linestyle=':', alpha=0.3)

plt.suptitle(f'Diffraction Grating Patterns (λ = {wavelength*1e9:.0f} nm)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diffraction_grating.png', dpi=150)
plt.show()
```

---

## 5. Fresnel Diffraction

### 5.1 The Fresnel Regime

When the observation point is close enough that the Fresnel number $N_F = a^2/(\lambda L) \geq 1$, we cannot ignore the curvature of the wavefronts. The diffraction pattern depends on the exact distance to the screen and can be quite complex.

### 5.2 Fresnel Zones

Fresnel's ingenious approach: divide the wavefront into concentric annular regions (Fresnel zones) such that consecutive zones differ in path length by $\lambda/2$.

The radius of the $m$-th Fresnel zone:

$$r_m = \sqrt{m\lambda z}$$

where $z$ is the distance from the aperture to the observation point.

Properties:
- Adjacent zones contribute nearly equal amplitudes but opposite phases
- The total field from all zones is approximately half the first zone's contribution
- Blocking alternate zones (a **zone plate**) acts as a focusing lens

### 5.3 Fresnel Zone Plate

A Fresnel zone plate is a diffractive optical element that blocks (or phase-shifts) alternate Fresnel zones. It focuses light like a lens, with focal length:

$$f_m = \frac{r_1^2}{\lambda} = \frac{r_m^2}{m\lambda}$$

Zone plates are used in X-ray microscopy (where refractive lenses are impractical because $n \approx 1$ for X-rays) and in astronomical radio telescopes.

### 5.4 Diffraction by a Straight Edge

Fresnel diffraction at a straight edge produces a characteristic pattern: the intensity does not abruptly go from bright to dark at the geometric shadow boundary. Instead:

- **Outside the shadow**: The intensity oscillates around the unobstructed value, with fringes that become less prominent further from the edge
- **At the shadow boundary**: The intensity is exactly 25% of the unobstructed value ($I = I_0/4$)
- **Inside the shadow**: The intensity decays smoothly to zero

This is described mathematically by the **Fresnel integrals** $C(u)$ and $S(u)$, which trace a Cornu spiral in the complex plane.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# Fresnel diffraction at a straight edge
# The intensity near the geometric shadow boundary

# Fresnel integrals: S(u) and C(u)
u = np.linspace(-5, 5, 2000)  # dimensionless Fresnel parameter

# The Fresnel integrals give us the complex amplitude
S, C = fresnel(u)

# Complex amplitude: A(u) = (C(u) + 1/2) + i*(S(u) + 1/2) relative to total
# For a semi-infinite screen (straight edge), the field at position u is:
# E(u) = (1 + i)/2 * [(C(u) + 1/2) + i*(S(u) + 1/2)]
# but more directly, the intensity is:
# I/I_0 = 1/2 * [(C(u) + 1/2)^2 + (S(u) + 1/2)^2]
I_norm = 0.5 * ((C + 0.5)**2 + (S + 0.5)**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: intensity profile at the straight edge
ax1.plot(u, I_norm, 'b-', linewidth=2)
ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Unobstructed ($I_0$)')
ax1.axhline(0.25, color='red', linestyle=':', alpha=0.5, label='Shadow edge ($I_0/4$)')
ax1.axvline(0, color='green', linestyle='--', alpha=0.5, label='Geometric shadow boundary')
ax1.fill_betweenx([0, 1.5], -5, 0, alpha=0.05, color='gray')

ax1.set_xlabel('Fresnel parameter $u$', fontsize=12)
ax1.set_ylabel('$I / I_0$', fontsize=12)
ax1.set_title('Fresnel Diffraction at a Straight Edge', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.5)
ax1.text(-3, 0.1, 'Shadow\nregion', fontsize=10, ha='center', color='gray')
ax1.text(3, 0.8, 'Illuminated\nregion', fontsize=10, ha='center', color='gray')

# Right: Cornu spiral (C(u) vs S(u))
ax2.plot(C, S, 'b-', linewidth=1.5)
ax2.plot(0.5, 0.5, 'ro', markersize=8, label='$u \\to +\\infty$')
ax2.plot(-0.5, -0.5, 'go', markersize=8, label='$u \\to -\\infty$')

# Mark some u values along the spiral
for u_mark in [-3, -2, -1, 0, 1, 2, 3]:
    s_val, c_val = fresnel(u_mark)
    ax2.plot(c_val, s_val, 'ko', markersize=4)
    ax2.annotate(f'u={u_mark}', xy=(c_val, s_val), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('C(u)', fontsize=12)
ax2.set_ylabel('S(u)', fontsize=12)
ax2.set_title('Cornu Spiral', fontsize=13)
ax2.set_aspect('equal')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fresnel_edge_diffraction.png', dpi=150)
plt.show()
```

---

## 6. Diffraction and Resolution Limits

### 6.1 The Diffraction Limit

Every imaging system has a minimum resolvable feature size set by diffraction:

$$\Delta x_{\min} = \frac{0.61\lambda}{\text{NA}}$$

for a microscope objective, or equivalently:

$$\theta_{\min} = 1.22\frac{\lambda}{D}$$

for a telescope or camera with aperture $D$.

This is a *fundamental* limit — no amount of lens polishing, aberration correction, or computational processing can beat it using conventional far-field imaging. (Super-resolution techniques like STED, PALM/STORM, and structured illumination achieve resolution below the diffraction limit by using clever tricks to distinguish nearby fluorescent molecules.)

### 6.2 The Relationship Between Diffraction and Fourier Transforms

A profound result: the Fraunhofer diffraction pattern is the **Fourier transform** of the aperture function.

If the aperture has a transmission function $t(x, y)$, then the far-field electric field is:

$$E(k_x, k_y) \propto \mathcal{F}\{t(x, y)\}$$

where $k_x = \frac{2\pi}{\lambda}\sin\theta_x$ and $k_y = \frac{2\pi}{\lambda}\sin\theta_y$ are spatial frequencies.

This means:
- **Narrow aperture** (small in space) → **wide diffraction pattern** (spread in spatial frequency)
- **Wide aperture** (large in space) → **narrow diffraction pattern** (sharp in spatial frequency)

This is the spatial analog of the time-frequency uncertainty relation: $\Delta x \cdot \Delta k_x \geq 2\pi$.

### 6.3 Diffraction Limit of Telescopes and Cameras

For a circular aperture of diameter $D$, the optical transfer function (OTF) is zero beyond the cutoff spatial frequency:

$$f_{\text{cutoff}} = \frac{D}{\lambda f} = \frac{1}{\lambda N}$$

where $N = f/D$ is the f-number. Features with spatial frequencies above $f_{\text{cutoff}}$ are simply not transmitted — they are irrecoverably lost.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstration: Fraunhofer diffraction as Fourier transform
# Compare the diffraction patterns of different aperture shapes

def compute_diffraction_2d(aperture, pad_factor=4):
    """
    Compute the Fraunhofer diffraction pattern of a 2D aperture.
    Uses FFT (since Fraunhofer diffraction = Fourier transform).

    aperture: 2D numpy array (transmission function)
    pad_factor: zero-padding factor for better resolution
    Returns: intensity pattern (2D)
    """
    N = aperture.shape[0]
    padded = np.zeros((N * pad_factor, N * pad_factor))
    start = (N * pad_factor - N) // 2
    padded[start:start+N, start:start+N] = aperture

    # 2D FFT (shifted so zero frequency is at center)
    E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    I = np.abs(E)**2
    I = I / I.max()  # normalize
    return I

N = 256
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Define four aperture shapes
apertures = {
    'Circular': R <= 0.5,
    'Square': (np.abs(X) <= 0.3) & (np.abs(Y) <= 0.3),
    'Slit (horizontal)': (np.abs(X) <= 0.4) & (np.abs(Y) <= 0.05),
    'Annular': (R >= 0.3) & (R <= 0.5),
}

for i, (name, aperture) in enumerate(apertures.items()):
    aperture = aperture.astype(float)
    I_diff = compute_diffraction_2d(aperture, pad_factor=4)

    # Top row: aperture
    axes[0, i].imshow(aperture, extent=[-1, 1, -1, 1], cmap='gray')
    axes[0, i].set_title(f'Aperture: {name}', fontsize=11)
    axes[0, i].set_xlabel('x')
    axes[0, i].set_ylabel('y')

    # Bottom row: diffraction pattern (log scale)
    M = I_diff.shape[0]
    extent = [-1, 1, -1, 1]  # normalized spatial frequency
    axes[1, i].imshow(np.log10(np.maximum(I_diff, 1e-6)),
                      extent=extent, cmap='inferno', vmin=-4, vmax=0)
    axes[1, i].set_title(f'Diffraction pattern (log)', fontsize=11)
    axes[1, i].set_xlabel('$k_x$')
    axes[1, i].set_ylabel('$k_y$')

plt.suptitle('Fraunhofer Diffraction = Fourier Transform of Aperture',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diffraction_fourier_transform.png', dpi=150)
plt.show()
```

---

## 7. Applications of Diffraction

### 7.1 X-Ray Crystallography

When X-rays ($\lambda \sim 0.1$ nm) are directed at a crystal, the regular atomic lattice acts as a three-dimensional diffraction grating. The condition for constructive interference (Bragg's law):

$$2d\sin\theta = n\lambda$$

where $d$ is the lattice spacing. By analyzing the positions and intensities of the diffraction spots, the crystal structure can be determined. This technique revealed the structures of DNA, proteins, and countless other molecules.

### 7.2 Spectrometers

Diffraction gratings are the heart of most spectrometers. Light enters through a slit, is collimated, diffracted by a grating, and focused onto a detector array. Each wavelength is focused to a different position:

$$x = f_{\text{camera}} \cdot \sin\theta_m \approx f_{\text{camera}} \cdot \frac{m\lambda}{d}$$

The grating equation determines the calibration, the resolving power determines the spectral resolution, and the blaze angle determines the efficiency.

### 7.3 Acoustic and Water Wave Analogs

Diffraction is universal to all wave phenomena. Sound diffracts around corners (you can hear someone around a hallway corner). Water waves diffract through harbor openings. These macroscopic analogs make diffraction directly observable and provide powerful pedagogical demonstrations.

### 7.4 Diffraction in Everyday Life

- **CDs/DVDs**: The track spacing ($\sim 1.6$ $\mu$m for CD) acts as a reflection grating, producing rainbow colors
- **Camera star bursts**: Diffraction around the aperture blades of a camera iris creates pointed "starburst" patterns from bright point sources
- **Holographic security labels**: Micro-structured patterns produce angle-dependent colors through diffraction

---

## Exercises

### Exercise 1: Single-Slit Analysis

A single slit of width 0.1 mm is illuminated by monochromatic light at 633 nm. The diffraction pattern is observed on a screen 2 m away.

(a) What is the width of the central maximum (distance between first minima on each side)?

(b) At what angle is the third minimum?

(c) Calculate the Fresnel number. Is the Fraunhofer approximation valid at this distance?

### Exercise 2: Airy Disk and Resolution

The Hubble Space Telescope has a 2.4 m primary mirror.

(a) Calculate the angular diameter of the Airy disk at $\lambda = 500$ nm.

(b) What is the smallest feature size it can resolve on the Moon's surface (distance 384,000 km)?

(c) What is the spatial resolution on the detector, given an effective focal length of 57.6 m?

(d) Compare with the HST's pixel size of 25 $\mu$m. Is the detector adequately sampling the Airy disk?

### Exercise 3: Diffraction Grating Spectroscopy

A grating has 600 grooves/mm and is 5 cm wide.

(a) What is the total number of grooves illuminated?

(b) Calculate the resolving power in the first and second orders.

(c) Can this grating resolve the sodium D doublet ($\lambda_1 = 589.0$ nm, $\lambda_2 = 589.6$ nm) in the first order? In the second order?

(d) What is the angular separation of the sodium doublet in the second order?

(e) What is the free spectral range in the second order at 589 nm?

### Exercise 4: Fresnel Zone Plate

Design a Fresnel zone plate to act as a lens with focal length $f = 1$ m at $\lambda = 550$ nm.

(a) What is the radius of the first zone?

(b) What is the radius of the 10th zone?

(c) How many zones are needed to achieve a 1 cm radius plate?

(d) What is the minimum feature size (width of the outermost zone)? Is this manufacturable with standard lithography?

### Exercise 5: Diffraction-Limited Photography

A camera has a 50 mm lens and a sensor with 4 $\mu$m pixel pitch.

(a) At what f-number does the Airy disk diameter equal the pixel size? (Use $\lambda = 550$ nm.)

(b) What happens to image sharpness as you stop down below this f-number?

(c) Calculate the maximum useful resolution (line pairs per mm) at $f$/2.8 and $f$/16.

---

## Summary

| Concept | Key Formula / Fact |
|---------|-------------------|
| Huygens-Fresnel principle | Every point on a wavefront is a source of secondary wavelets |
| Fresnel number | $N_F = a^2/(\lambda L)$; Fraunhofer when $N_F \ll 1$ |
| Single-slit minima | $a\sin\theta = m\lambda$, $m = \pm 1, \pm 2, \ldots$ |
| Single-slit intensity | $I = I_0 (\sin\beta/\beta)^2$, $\beta = \pi a\sin\theta/\lambda$ |
| Airy pattern (circular) | $I = I_0 [2J_1(x)/x]^2$; first zero at $1.22\lambda/D$ |
| Diffraction limit | $\Delta x = 0.61\lambda/\text{NA}$; $\theta_R = 1.22\lambda/D$ |
| Grating maxima | $d\sin\theta = m\lambda$ |
| Grating resolving power | $\mathcal{R} = mN$ ($m$ = order, $N$ = total grooves) |
| Grating angular dispersion | $d\theta/d\lambda = m/(d\cos\theta)$ |
| Grating FSR | $\Delta\lambda_{\text{FSR}} = \lambda/m$ |
| Fresnel zone radius | $r_m = \sqrt{m\lambda z}$ |
| Fraunhofer ↔ Fourier | Far-field pattern is the Fourier transform of the aperture |
| Bragg's law | $2d\sin\theta = n\lambda$ (X-ray diffraction) |

---

[← Previous: 05. Wave Optics — Interference](05_Wave_Optics_Interference.md) | [Next: 07. Polarization →](07_Polarization.md)

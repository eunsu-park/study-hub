# 10. Fourier Optics

[← Previous: 09. Fiber Optics](09_Fiber_Optics.md) | [Next: 11. Holography →](11_Holography.md)

---

Fourier optics is the stunning realization that a simple glass lens performs the same mathematical operation as a Fourier transform — converting spatial patterns into spatial frequencies, and vice versa, at the speed of light. This insight, which connects physical optics to linear systems theory, revolutionized optical instrument design, image processing, and our understanding of how microscopes and cameras fundamentally work.

When Ernst Abbe formulated his theory of microscope imaging in the 1870s, he showed that resolution is determined not by lens quality alone but by how many diffraction orders (spatial frequencies) the lens can collect. When Frits Zernike invented phase contrast microscopy in the 1930s (Nobel Prize, 1953), he used Fourier optics principles to make transparent biological specimens visible. Today, Fourier optics underpins everything from adaptive optics in telescopes to computational imaging in smartphones.

This lesson develops the Fourier-transform relationship between diffraction and propagation, shows how a lens implements the Fourier transform, and explores spatial filtering, optical transfer functions, and coherent imaging theory.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Describe scalar diffraction as a linear, shift-invariant system and identify its impulse response and transfer function
2. Derive the angular spectrum representation and the transfer function of free-space propagation
3. Prove that a thin lens performs a Fourier transform in its back focal plane
4. Design a 4f spatial filtering system and predict the effect of various filter masks
5. Define and relate the point spread function (PSF), optical transfer function (OTF), and modulation transfer function (MTF)
6. Explain Abbe's theory of image formation and the resolution limit in coherent and incoherent imaging
7. Describe the principle of phase contrast microscopy as a spatial filtering operation

---

## Table of Contents

1. [Diffraction as a Linear System](#1-diffraction-as-a-linear-system)
2. [Angular Spectrum Representation](#2-angular-spectrum-representation)
3. [Fresnel and Fraunhofer Diffraction Revisited](#3-fresnel-and-fraunhofer-diffraction-revisited)
4. [The Lens as Fourier Transformer](#4-the-lens-as-fourier-transformer)
5. [The 4f System and Spatial Filtering](#5-the-4f-system-and-spatial-filtering)
6. [Optical Transfer Function](#6-optical-transfer-function)
7. [Abbe Theory of Image Formation](#7-abbe-theory-of-image-formation)
8. [Phase Contrast Microscopy](#8-phase-contrast-microscopy)
9. [Python Examples](#9-python-examples)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Diffraction as a Linear System

### 1.1 The Linear Systems Perspective

The central insight of Fourier optics is this: **free-space propagation of light is a linear, shift-invariant (LSI) system**. This means we can characterize it completely by its impulse response $h(x, y)$ or, equivalently, its transfer function $H(f_x, f_y)$.

Given an input field $U_{\text{in}}(x, y)$ at plane $z = 0$, the output field at plane $z = d$ is:

$$U_{\text{out}}(x, y) = U_{\text{in}}(x, y) * h(x, y; d)$$

where $*$ denotes 2D convolution and $h$ is the propagation impulse response (the field produced by a point source).

In the frequency domain:

$$\tilde{U}_{\text{out}}(f_x, f_y) = \tilde{U}_{\text{in}}(f_x, f_y) \cdot H(f_x, f_y; d)$$

> **Analogy**: Think of a complex scene as a musical chord — a superposition of many pure tones (spatial frequencies). Free-space propagation is like an audio equalizer: it modifies each frequency component independently (via the transfer function $H$), without creating new frequencies. The output is simply the input with each spatial frequency amplitude and phase adjusted according to how well it propagates. A lens, in this analogy, is like a spectrum analyzer — it separates all the tones and displays them side by side.

### 1.2 Scalar Diffraction Theory

We work within the **scalar approximation**: treating light as a scalar complex field $U(\mathbf{r}) = |U|e^{i\phi}$, ignoring polarization. This is valid when:
- Feature sizes are much larger than the wavelength
- We work in the paraxial regime (angles < ~20°)
- We are not near sharp material boundaries

The scalar field satisfies the Helmholtz equation:

$$(\nabla^2 + k^2)U = 0, \quad k = \frac{2\pi}{\lambda}$$

### 1.3 The Huygens-Fresnel Principle (Formalized)

Every point on a wavefront acts as a secondary source. The Rayleigh-Sommerfeld diffraction integral formalizes this:

$$U(x, y, d) = \frac{1}{i\lambda}\iint U(x', y', 0)\frac{e^{ikr}}{r}\cos\theta\,dx'dy'$$

where $r = \sqrt{(x-x')^2 + (y-y')^2 + d^2}$ is the distance from source to observation point.

---

## 2. Angular Spectrum Representation

### 2.1 Decomposition into Plane Waves

Any monochromatic field $U(x, y, 0)$ can be decomposed into plane waves via the 2D Fourier transform:

$$\tilde{U}(f_x, f_y; 0) = \iint U(x, y, 0)\,e^{-i2\pi(f_x x + f_y y)}\,dx\,dy$$

Each spatial frequency $(f_x, f_y)$ corresponds to a plane wave propagating at angles:

$$\sin\alpha = \lambda f_x, \quad \sin\beta = \lambda f_y$$

where $\alpha$ and $\beta$ are the angles with respect to the $z$-axis in the $xz$ and $yz$ planes.

### 2.2 Transfer Function of Free Space

Each plane wave component propagates through distance $d$ by acquiring a phase:

$$H(f_x, f_y; d) = \exp\!\left(i2\pi d\sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}\right)$$

This is exact (within the scalar approximation). Two important regimes:

**Propagating waves**: When $f_x^2 + f_y^2 < 1/\lambda^2$, $H$ is a pure phase factor — the wave propagates without loss.

**Evanescent waves**: When $f_x^2 + f_y^2 > 1/\lambda^2$, the square root becomes imaginary, and $H$ decays exponentially:

$$H \propto \exp\!\left(-2\pi d\sqrt{f_x^2 + f_y^2 - 1/\lambda^2}\right)$$

These high-frequency components carry sub-wavelength detail but decay within a fraction of a wavelength — the fundamental reason conventional optics cannot resolve features smaller than $\sim\lambda/2$.

### 2.3 The Paraxial (Fresnel) Approximation

For small angles ($f_x^2 + f_y^2 \ll 1/\lambda^2$), we Taylor-expand the square root:

$$H(f_x, f_y; d) \approx e^{ikd}\exp\!\left(-i\pi\lambda d(f_x^2 + f_y^2)\right)$$

This is the **Fresnel transfer function** — a quadratic phase in frequency space. It corresponds to a quadratic phase in the spatial domain (the Fresnel propagation kernel).

---

## 3. Fresnel and Fraunhofer Diffraction Revisited

### 3.1 Fresnel Diffraction

Under the paraxial approximation, the propagated field is:

$$U(x, y, d) = \frac{e^{ikd}}{i\lambda d}\iint U(x', y', 0)\,\exp\!\left(\frac{ik}{2d}\left[(x-x')^2 + (y-y')^2\right]\right)dx'dy'$$

This is a convolution with the Fresnel kernel $h_F(x,y) = \frac{e^{ikd}}{i\lambda d}\exp\!\left(\frac{ik(x^2+y^2)}{2d}\right)$.

### 3.2 Fraunhofer Diffraction

When the observation distance is very large ($d \gg \frac{k(x'^2 + y'^2)_{\max}}{2}$, i.e., $d \gg a^2/\lambda$ where $a$ is the aperture size), the quadratic phase in the integral becomes negligible, and:

$$U(x, y, d) = \frac{e^{ikd}}{i\lambda d}e^{\frac{ik(x^2+y^2)}{2d}}\tilde{U}\!\left(\frac{x}{\lambda d}, \frac{y}{\lambda d}\right)$$

The far-field pattern is the **Fourier transform** of the aperture field, evaluated at spatial frequencies $f_x = x/(\lambda d)$, $f_y = y/(\lambda d)$.

**Key insight**: Fraunhofer diffraction is a physical Fourier transform, with the transform variable being the observation angle.

### 3.3 The Fresnel Number

The Fresnel number $N_F = a^2/(\lambda d)$ classifies the diffraction regime:
- $N_F \gg 1$: Near-field (Fresnel diffraction)
- $N_F \ll 1$: Far-field (Fraunhofer diffraction)
- $N_F \sim 1$: Transition region

---

## 4. The Lens as Fourier Transformer

### 4.1 Phase Transformation by a Thin Lens

A thin lens of focal length $f$ introduces a quadratic phase:

$$t_{\text{lens}}(x, y) = \exp\!\left(-\frac{ik(x^2 + y^2)}{2f}\right)$$

This is the same mathematical form as the Fresnel propagation kernel — but with opposite sign. The lens exactly compensates the quadratic phase accumulated during propagation over distance $f$.

### 4.2 Fourier Transform Property

Place an input transparency $U_{\text{in}}(x, y)$ in the **front focal plane** of a lens ($z = -f$). The field in the **back focal plane** ($z = +f$) is:

$$\boxed{U_f(x, y) = \frac{1}{i\lambda f}\tilde{U}_{\text{in}}\!\left(\frac{x}{\lambda f}, \frac{y}{\lambda f}\right)}$$

This is an **exact** Fourier transform (within the paraxial approximation), with no quadratic phase prefactor. The lens brings the Fraunhofer far field from infinity to the focal plane.

**Physical meaning**: Each point $(x, y)$ in the focal plane corresponds to a spatial frequency component of the input:

$$f_x = \frac{x}{\lambda f}, \quad f_y = \frac{y}{\lambda f}$$

### 4.3 Proof Sketch

The field just after the input plane propagates distance $f$ to the lens (Fresnel propagation), passes through the lens (phase multiplication), then propagates another distance $f$ to the output plane (Fresnel propagation). The three quadratic phases cancel perfectly, leaving only the Fourier integral. This cancellation is why the focal-plane result is a clean Fourier transform.

### 4.4 Scaling Properties

The spatial frequency resolution in the Fourier plane is:

$$\delta x = \lambda f \cdot \delta f_x$$

So a longer focal length $f$ spreads the spectrum over a larger area (higher spectral resolution), while a shorter $f$ compresses it (lower resolution but wider field of view).

---

## 5. The 4f System and Spatial Filtering

### 5.1 The 4f Configuration

The **4f system** consists of two lenses separated by the sum of their focal lengths, with the total length being $4f$ (hence the name, assuming identical lenses):

```
Input       Lens 1      Fourier       Lens 2      Output
plane         f          plane          f          plane
  │           │            │            │            │
  │     f     │     f      │     f      │     f      │
  │←────────→│←─────────→│←─────────→│←──────────→│
  │           │            │            │            │
  U_in      FT           F(fx,fy)     FT^(-1)     U_out
```

- Lens 1 computes the Fourier transform of $U_{\text{in}}$ at the Fourier plane
- A spatial filter (mask) $H(f_x, f_y)$ is placed in the Fourier plane
- Lens 2 computes the inverse Fourier transform, producing the filtered output

The output is:

$$U_{\text{out}}(x, y) = \mathcal{F}^{-1}\!\left\{\tilde{U}_{\text{in}}(f_x, f_y) \cdot H(f_x, f_y)\right\}$$

This is **optical convolution** — the optical implementation of a linear spatial filter. The entire operation happens at the speed of light, in parallel across all pixels simultaneously.

### 5.2 Spatial Filtering Examples

**Low-pass filter** (pinhole in Fourier plane): Blocks high spatial frequencies, smoothing the image. Removes sharp edges and noise.

$$H(f_x, f_y) = \begin{cases} 1 & \sqrt{f_x^2 + f_y^2} \leq f_c \\ 0 & \text{otherwise} \end{cases}$$

**High-pass filter** (opaque disk in Fourier plane): Blocks low spatial frequencies, enhancing edges and fine detail.

**Band-pass filter** (annular opening): Passes only a specific range of spatial frequencies.

**Directional filter** (slit in Fourier plane): Passes frequencies in one orientation, removing features with a specific directionality (e.g., removing horizontal scan lines).

> **Analogy**: The 4f system is like a graphic equalizer for images. Just as a music equalizer displays the audio spectrum on a row of sliders — letting you boost the bass (low frequencies) or treble (high frequencies) — the 4f system displays the image's spatial spectrum in the Fourier plane, where you can physically block or modify any spatial frequency band with a mask.

### 5.3 Historical Note: Abbe-Porter Experiment

In 1906, Albert Porter demonstrated Abbe's theory by placing various masks in the Fourier plane of a microscope. A mesh object produced a regular array of dots in the Fourier plane (the diffraction orders). By blocking selected dots, he could make horizontal or vertical lines disappear from the image — a spectacular demonstration that the image is assembled from its spatial frequency components.

---

## 6. Optical Transfer Function

### 6.1 Coherent vs. Incoherent Imaging

The theory of imaging depends fundamentally on whether the illumination is **coherent** or **incoherent**:

**Coherent imaging** (e.g., laser illumination): The optical system is linear in complex field amplitude.

$$U_{\text{img}}(x, y) = h(x, y) * U_{\text{obj}}(x, y)$$

where $h$ is the amplitude point spread function (coherent PSF).

**Incoherent imaging** (e.g., natural light): The system is linear in intensity.

$$I_{\text{img}}(x, y) = |h(x, y)|^2 * I_{\text{obj}}(x, y)$$

### 6.2 Point Spread Function (PSF)

The **PSF** is the image of a point source — the impulse response of the imaging system.

For a circular aperture of diameter $D$, the coherent PSF is the Airy pattern:

$$h(r) \propto \frac{2J_1(\pi Dr/(\lambda f))}{\pi Dr/(\lambda f)}$$

The intensity PSF (incoherent) is $|h(r)|^2$, the Airy disk. The first zero occurs at:

$$r_{\text{Airy}} = 1.22\frac{\lambda f}{D}$$

This is the **Rayleigh resolution criterion**: two point sources are just resolved when the maximum of one falls on the first minimum of the other.

### 6.3 Coherent Transfer Function (CTF)

For coherent imaging, the transfer function in the frequency domain is:

$$H_{\text{coh}}(f_x, f_y) = P(\lambda f f_x, \lambda f f_y)$$

where $P$ is the pupil function. For a circular aperture of radius $a$:

$$H_{\text{coh}}(f_r) = \begin{cases} 1 & f_r \leq f_c = a/(\lambda f) \\ 0 & f_r > f_c \end{cases}$$

The coherent system is a perfect low-pass filter with sharp cutoff at $f_c$.

### 6.4 Optical Transfer Function (OTF) — Incoherent

For incoherent imaging, the transfer function is the **OTF**, defined as:

$$\text{OTF}(f_x, f_y) = \frac{\iint P(\xi, \eta)\,P^*(\xi - \lambda f f_x, \eta - \lambda f f_y)\,d\xi\,d\eta}{\iint |P(\xi, \eta)|^2\,d\xi\,d\eta}$$

This is the **autocorrelation** of the pupil function, normalized to unity at the origin.

### 6.5 Modulation Transfer Function (MTF)

The **MTF** is the magnitude of the OTF:

$$\text{MTF}(f_x, f_y) = |\text{OTF}(f_x, f_y)|$$

It describes how much contrast survives at each spatial frequency. MTF = 1 means full contrast; MTF = 0 means the spatial frequency is completely lost.

For a circular aperture (aberration-free), the OTF/MTF has the form:

$$\text{MTF}(f_r) = \frac{2}{\pi}\left[\arccos\!\left(\frac{f_r}{2f_c}\right) - \frac{f_r}{2f_c}\sqrt{1 - \left(\frac{f_r}{2f_c}\right)^2}\right]$$

for $f_r \leq 2f_c$, and zero otherwise. The incoherent cutoff frequency is **twice** the coherent cutoff: $f_{\text{max}} = 2f_c = D/(\lambda f)$.

### 6.6 Coherent vs. Incoherent Resolution

| Property | Coherent | Incoherent |
|----------|----------|------------|
| Linear in | Amplitude $U$ | Intensity $I$ |
| Transfer function | CTF (pupil) | OTF (autocorrelation of pupil) |
| Cutoff frequency | $f_c = a/(\lambda f)$ | $2f_c = D/(\lambda f)$ |
| Phase response | Present | Present (phase transfer function, PTF) |
| Edge ringing | Yes (coherent artifacts) | No (smoother) |

The incoherent system passes frequencies up to $2f_c$ — double the coherent cutoff — but with diminishing contrast. Which is "better" depends on the application: incoherent imaging has higher cutoff but no phase information; coherent imaging preserves phase but is plagued by speckle.

---

## 7. Abbe Theory of Image Formation

### 7.1 Abbe's Two-Step Model

Ernst Abbe (1873) described microscope imaging as a **two-step process**:

1. **Step 1**: The object diffracts the illuminating beam, producing diffraction orders. Each order corresponds to a spatial frequency of the object.

2. **Step 2**: The microscope objective lens collects some of these diffraction orders and recombines them to form the image. Only the orders that enter the lens contribute to the image.

```
  Illumination    Object     Objective    Back focal    Image
   plane         (grating)    lens         plane        plane
     │              │           │            │            │
   ──┼──────────→   │ ↗ +1     │            │            │
     │              │ → 0      │        · +1 │            │
   ──┼──────────→   │ ↘ -1     │        · 0  │            │
     │              │           │        · -1 │            │
     │              │           │            │            │
     Coherent    Diffraction    Lens      Fourier      Recombined
     plane wave   orders       collects   plane        orders form
                               orders                  the image
```

### 7.2 Resolution Limit

For a grating with period $d$, the first-order diffraction angle is $\sin\theta_1 = \lambda/d$. The objective can collect this order only if $\sin\theta_1 \leq \text{NA}$:

$$d_{\min} = \frac{\lambda}{\text{NA}} \quad (\text{coherent, normal incidence})$$

With oblique illumination (filling the condenser NA), the limit improves to:

$$\boxed{d_{\min} = \frac{\lambda}{2\,\text{NA}}} \quad (\text{Abbe resolution limit})$$

This is the fundamental resolution limit of far-field optical microscopy. For visible light ($\lambda \approx 500\,\text{nm}$) and a high-NA oil immersion objective ($\text{NA} = 1.4$):

$$d_{\min} = \frac{500}{2 \times 1.4} \approx 180\,\text{nm}$$

### 7.3 Beyond the Abbe Limit

Modern super-resolution techniques (STED, PALM, STORM) circumvent this limit by exploiting nonlinear fluorescence responses or stochastic single-molecule localization. These methods earned the 2014 Nobel Prize in Chemistry (Betzig, Hell, Moerner).

---

## 8. Phase Contrast Microscopy

### 8.1 The Problem: Phase Objects

Many biological specimens (cells, bacteria) are nearly transparent — they absorb very little light but alter its phase as it passes through regions of different refractive index. A standard bright-field microscope detects intensity, not phase, so these specimens are nearly invisible.

Mathematically, a thin phase object has transmittance:

$$t(x, y) = e^{i\phi(x, y)} \approx 1 + i\phi(x, y) \quad (\text{for small } \phi)$$

The intensity is $|t|^2 = 1 + \phi^2 \approx 1$ — the phase information is lost.

### 8.2 Zernike's Solution

Frits Zernike (1934) realized that the problem was one of spatial filtering. In the Fourier plane:

- The "1" (DC component, undiffracted light) becomes a bright spot at the center
- The "$i\phi$" terms (diffracted light carrying the phase information) spread across the plane

The diffracted light is $\pi/2$ out of phase with the DC component (the factor of $i$). Interference between them does not produce intensity variations because the phase shift is $\pi/2$ (imagine: $|1 + i\epsilon|^2 \approx 1$).

Zernike's insight: **shift the DC component by an additional $\pm\pi/2$** using a small phase plate (a phase ring) at the center of the Fourier plane:

$$\text{Phase plate}: \quad H(0,0) = e^{\pm i\pi/2} = \pm i$$

Now the total field becomes:

$$U_{\text{out}} \propto \pm i + i\phi = i(\pm 1 + \phi)$$

And the intensity is:

$$I \propto |(\pm 1 + \phi)|^2 = 1 \pm 2\phi + \phi^2 \approx 1 \pm 2\phi$$

The phase $\phi$ now appears **linearly in intensity** — the phase object becomes visible!

- **Positive phase contrast** ($+\pi/2$ shift): Phase-advanced regions appear bright on a gray background
- **Negative phase contrast** ($-\pi/2$ shift): Phase-advanced regions appear dark

### 8.3 Practical Implementation

In a microscope:
1. An annular condenser aperture produces a hollow cone of illumination
2. The undiffracted light passes through a matching annular phase ring in the objective's back focal plane
3. The diffracted light mostly misses the ring and passes through unaffected
4. The $\pi/2$ phase shift of the ring converts phase differences to intensity differences

This requires no staining, no preparation — living cells can be observed in real time.

---

## 9. Python Examples

### 9.1 Fraunhofer Diffraction and the Fourier Transform

```python
import numpy as np
import matplotlib.pyplot as plt

def fraunhofer_diffraction(aperture, wavelength, z, pixel_size):
    """
    Compute the Fraunhofer diffraction pattern of an aperture.

    In the far field, the diffracted field is the Fourier transform
    of the aperture function. We use FFT to compute this numerically.
    The physical coordinates in the observation plane are related to
    spatial frequencies by (x, y) = lambda * z * (fx, fy).
    """
    N = aperture.shape[0]

    # Compute 2D FFT — this IS the Fraunhofer integral
    # fftshift centers the zero-frequency component
    U_far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))

    # Intensity pattern
    I_far = np.abs(U_far)**2
    I_far /= I_far.max()  # Normalize to peak

    # Physical coordinates in observation plane
    df = 1.0 / (N * pixel_size)  # Frequency spacing
    fx = np.arange(-N//2, N//2) * df
    x_obs = wavelength * z * fx  # Physical coordinates

    return I_far, x_obs

# --- Simulate diffraction from different apertures ---
N = 1024
pixel_size = 1e-6  # 1 µm per pixel
wavelength = 500e-9  # 500 nm (green light)
z = 1.0  # 1 meter propagation

# Coordinate grid
x = np.arange(-N//2, N//2) * pixel_size
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

# Circular aperture (diameter = 200 µm)
D = 200e-6
aperture_circ = (R <= D/2).astype(float)

# Square aperture (side = 200 µm)
a = 200e-6
aperture_sq = ((np.abs(X) <= a/2) & (np.abs(Y) <= a/2)).astype(float)

# Double slit (slit width = 50 µm, separation = 200 µm)
slit_w = 50e-6
slit_sep = 200e-6
slit_h = 400e-6
aperture_ds = (
    ((np.abs(X - slit_sep/2) <= slit_w/2) | (np.abs(X + slit_sep/2) <= slit_w/2))
    & (np.abs(Y) <= slit_h/2)
).astype(float)

fig, axes = plt.subplots(3, 2, figsize=(12, 14))
apertures = [aperture_circ, aperture_sq, aperture_ds]
titles = ['Circular Aperture', 'Square Aperture', 'Double Slit']

for i, (ap, title) in enumerate(zip(apertures, titles)):
    # Aperture
    extent_ap = [x[0]*1e3, x[-1]*1e3, x[0]*1e3, x[-1]*1e3]
    axes[i, 0].imshow(ap, cmap='gray', extent=extent_ap)
    axes[i, 0].set_title(f'{title} (Object Plane)', fontsize=11)
    axes[i, 0].set_xlabel('x [mm]')
    axes[i, 0].set_ylabel('y [mm]')

    # Diffraction pattern (log scale for visibility)
    I_far, x_obs = fraunhofer_diffraction(ap, wavelength, z, pixel_size)
    extent_diff = [x_obs[0]*1e3, x_obs[-1]*1e3, x_obs[0]*1e3, x_obs[-1]*1e3]
    axes[i, 1].imshow(np.log10(I_far + 1e-6), cmap='inferno',
                       extent=extent_diff, vmin=-4, vmax=0)
    axes[i, 1].set_title(f'{title} (Fraunhofer Pattern, log)', fontsize=11)
    axes[i, 1].set_xlabel('x [mm]')
    axes[i, 1].set_ylabel('y [mm]')
    axes[i, 1].set_xlim(-5, 5)
    axes[i, 1].set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('fraunhofer_diffraction.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.2 4f Spatial Filtering System

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import face  # Test image

def spatial_filter_4f(image, filter_mask):
    """
    Simulate a 4f spatial filtering system.

    Steps:
    1. Lens 1: Fourier transform the input image
    2. Multiply by the filter mask in the Fourier plane
    3. Lens 2: inverse Fourier transform to get the filtered output

    This is optically equivalent to convolution with the inverse FT of the mask.
    """
    # Fourier transform (Lens 1)
    spectrum = np.fft.fftshift(np.fft.fft2(image))

    # Apply filter in Fourier plane
    filtered_spectrum = spectrum * filter_mask

    # Inverse Fourier transform (Lens 2)
    output = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))

    return np.abs(output), np.abs(spectrum), np.abs(filtered_spectrum)

# Load and prepare test image (grayscale)
img = face(gray=True).astype(float)
img = img[:512, :512]  # Crop to square
img /= img.max()

Ny, Nx = img.shape
fy = np.fft.fftshift(np.fft.fftfreq(Ny))
fx = np.fft.fftshift(np.fft.fftfreq(Nx))
FX, FY = np.meshgrid(fx, fy)
FR = np.sqrt(FX**2 + FY**2)

# Define filter masks
# 1. Low-pass: keep only low spatial frequencies (blur)
cutoff_low = 0.05  # Normalized frequency
mask_lowpass = (FR <= cutoff_low).astype(float)

# 2. High-pass: remove low frequencies (edge detection)
cutoff_high = 0.02
mask_highpass = (FR > cutoff_high).astype(float)

# 3. Band-pass: keep middle frequencies
mask_bandpass = ((FR > 0.02) & (FR < 0.1)).astype(float)

# 4. Directional: vertical slit (pass horizontal details only)
mask_directional = (np.abs(FX) < 0.01).astype(float)

filters = [mask_lowpass, mask_highpass, mask_bandpass, mask_directional]
names = ['Low-pass (blur)', 'High-pass (edges)', 'Band-pass', 'Vertical slit']

fig, axes = plt.subplots(4, 3, figsize=(14, 16))
fig.suptitle('4f Spatial Filtering System', fontsize=16, y=1.01)

for i, (mask, name) in enumerate(zip(filters, names)):
    output, spectrum, filt_spectrum = spatial_filter_4f(img, mask)

    # Filter mask
    axes[i, 0].imshow(mask, cmap='gray', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[i, 0].set_title(f'Filter: {name}', fontsize=10)
    axes[i, 0].set_xlabel('fx')
    axes[i, 0].set_ylabel('fy')

    # Filtered spectrum
    axes[i, 1].imshow(np.log10(filt_spectrum + 1), cmap='inferno')
    axes[i, 1].set_title('Filtered Spectrum (log)', fontsize=10)
    axes[i, 1].axis('off')

    # Output image
    axes[i, 2].imshow(output, cmap='gray')
    axes[i, 2].set_title('Output Image', fontsize=10)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('4f_spatial_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 MTF of a Circular Aperture

```python
import numpy as np
import matplotlib.pyplot as plt

def mtf_circular(f_norm):
    """
    Analytical MTF for a diffraction-limited circular aperture.

    The MTF is the autocorrelation of the circular pupil, normalized
    to unity at zero frequency. For an aberration-free system, this
    represents the theoretical maximum contrast at each spatial frequency.
    Below this curve, you can never do better without super-resolution tricks.
    """
    # Normalized frequency: f_norm = f / f_cutoff, range [0, 1]
    f = np.clip(f_norm, 0, 1)
    mtf = (2/np.pi) * (np.arccos(f) - f * np.sqrt(1 - f**2))
    return mtf

f = np.linspace(0, 1, 500)
mtf_vals = mtf_circular(f)

# Compare with coherent transfer function (CTF)
ctf = np.where(f <= 0.5, 1.0, 0.0)  # CTF cuts off at f_c = f_max/2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f, mtf_vals, 'b-', linewidth=2, label='Incoherent MTF')
ax.plot(f, ctf, 'r--', linewidth=2, label='Coherent CTF (at f/f_inc_cutoff)')
ax.fill_between(f, 0, mtf_vals, alpha=0.1, color='blue')

ax.set_xlabel('Normalized spatial frequency (f / f_cutoff)', fontsize=12)
ax.set_ylabel('Transfer function value', fontsize=12)
ax.set_title('MTF and CTF of a Circular Aperture (Aberration-Free)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.05, 1.1)

# Annotate
ax.annotate('Incoherent cutoff\n(f = D/λf)', xy=(1.0, 0), fontsize=9,
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='blue'),
            xytext=(0.85, 0.15))
ax.annotate('Coherent cutoff\n(f = D/2λf)', xy=(0.5, 0), fontsize=9,
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='red'),
            xytext=(0.35, 0.15))

plt.tight_layout()
plt.savefig('mtf_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Diffraction as LSI system | $\tilde{U}_{\text{out}} = \tilde{U}_{\text{in}} \cdot H(f_x, f_y)$ |
| Free-space transfer function | $H = \exp\!\left(i2\pi d\sqrt{1/\lambda^2 - f_x^2 - f_y^2}\right)$ |
| Evanescent waves | $f_x^2 + f_y^2 > 1/\lambda^2 \Rightarrow$ exponential decay |
| Fresnel approximation | $H \approx e^{ikd}\exp(-i\pi\lambda d(f_x^2+f_y^2))$ |
| Fraunhofer diffraction | Far-field $\propto$ Fourier transform of aperture |
| Lens as FT | $U_f = \frac{1}{i\lambda f}\tilde{U}_{\text{in}}(x/\lambda f, y/\lambda f)$ |
| 4f system | Spatial filtering: FT → filter → inverse FT |
| PSF (circular aperture) | Airy disk; first zero at $r = 1.22\lambda f/D$ |
| Coherent cutoff | $f_c = a/(\lambda f) = D/(2\lambda f)$ |
| Incoherent cutoff | $2f_c = D/(\lambda f)$ — twice the coherent cutoff |
| MTF | $|\text{OTF}|$; describes contrast vs. spatial frequency |
| Abbe resolution | $d_{\min} = \lambda/(2\,\text{NA})$ |
| Phase contrast | $\pi/2$ phase shift of DC converts phase → intensity |

---

## 11. Exercises

### Exercise 1: Angular Spectrum Propagation

A uniform plane wave ($\lambda = 633\,\text{nm}$) illuminates a slit of width $a = 100\,\mu\text{m}$.

(a) Write the angular spectrum (Fourier transform) of the slit transmittance.
(b) At what spatial frequency do evanescent waves begin?
(c) Use Python to propagate the angular spectrum through $d = 1\,\text{cm}$ and $d = 1\,\text{m}$. Compare the results and identify which is in the Fresnel regime and which in the Fraunhofer regime.

### Exercise 2: 4f System Design

Design a 4f system using lenses with $f = 200\,\text{mm}$ at $\lambda = 532\,\text{nm}$.

(a) An object has spatial frequency components from 0 to 50 lines/mm. What is the physical extent of the Fourier spectrum in the filter plane?
(b) You want to build a low-pass filter that passes spatial frequencies up to 10 lines/mm. What is the required pinhole diameter?
(c) Implement this filter in Python and show its effect on a test image containing a grid pattern.

### Exercise 3: MTF Analysis

A camera lens has $f = 50\,\text{mm}$, $f/\# = 2.8$ (aperture $D = 17.9\,\text{mm}$), used at $\lambda = 550\,\text{nm}$.

(a) Calculate the diffraction-limited MTF cutoff frequency (incoherent).
(b) Plot the diffraction-limited MTF.
(c) The detector pixel pitch is 5 $\mu\text{m}$. At what spatial frequency does the detector's Nyquist limit occur? Is the system diffraction-limited or detector-limited?

### Exercise 4: Phase Contrast

Simulate Zernike phase contrast microscopy:

(a) Create a 2D phase object: $t(x,y) = \exp(i\phi(x,y))$ where $\phi$ has a circular region with $\phi = 0.3$ rad on a background of $\phi = 0$.
(b) Compute and display the bright-field intensity image (no filter). Verify it is nearly uniform.
(c) Apply a $\pi/2$ phase ring filter to the DC component in the Fourier plane. Display the resulting image and verify that the phase object is now visible.

---

## 12. References

1. Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W. H. Freeman. — The standard reference for Fourier optics.
2. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapters 4-5.
3. Hecht, E. (2017). *Optics* (5th ed.). Pearson. — Chapter 11.
4. Born, M., & Wolf, E. (2019). *Principles of Optics* (7th expanded ed.). Cambridge University Press.
5. Abbe, E. (1873). "Beitrage zur Theorie des Mikroskops und der mikroskopischen Wahrnehmung." *Archiv fur Mikroskopische Anatomie*, 9, 413-468.
6. Zernike, F. (1955). Nobel Lecture: "How I Discovered Phase Contrast."

---

[← Previous: 09. Fiber Optics](09_Fiber_Optics.md) | [Next: 11. Holography →](11_Holography.md)

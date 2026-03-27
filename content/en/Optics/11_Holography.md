# 11. Holography

[← Previous: 10. Fourier Optics](10_Fourier_Optics.md) | [Next: 12. Nonlinear Optics →](12_Nonlinear_Optics.md)

---

A photograph records the intensity of light — how bright each point of a scene is. But it throws away the phase — the timing of the light waves that encodes depth and three-dimensional structure. Holography, invented by Dennis Gabor in 1948 (Nobel Prize, 1971), is the technique for recording **both** amplitude and phase, enabling the faithful reconstruction of a complete three-dimensional light field.

The word "holography" comes from the Greek *holos* (whole) and *graphe* (writing) — whole writing, or complete recording. A hologram does not merely display a 3D picture; it recreates the original light wave so perfectly that your eyes cannot tell the difference between looking at the hologram and looking at the real object. You can shift your viewpoint and see around corners of the object. Different parts of the hologram show different perspectives. If you cut a hologram in half, each half still contains the entire scene (at reduced resolution).

This lesson covers the physics of holographic recording and reconstruction, the major types of holograms, and modern digital holography techniques.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

1. Explain how a hologram encodes both amplitude and phase information through interference with a reference beam
2. Derive the reconstruction process and identify the real image, virtual image, and conjugate beam terms
3. Compare Gabor (inline) and Leith-Upatnieks (off-axis) holography configurations and their advantages
4. Distinguish thin and thick (volume) holograms and explain Bragg selectivity in volume holograms
5. Describe the principles of digital holography: recording on CCD/CMOS sensors and numerical reconstruction
6. Identify key applications of holography in 3D display, data storage, interferometry, and security
7. Perform numerical holographic recording and reconstruction using Python

---

## Table of Contents

1. [The Holographic Principle](#1-the-holographic-principle)
2. [Recording a Hologram](#2-recording-a-hologram)
3. [Reconstructing the Wavefront](#3-reconstructing-the-wavefront)
4. [Gabor (Inline) Holography](#4-gabor-inline-holography)
5. [Leith-Upatnieks (Off-Axis) Holography](#5-leith-upatnieks-off-axis-holography)
6. [Thin vs. Thick Holograms](#6-thin-vs-thick-holograms)
7. [Volume Holograms and Bragg Diffraction](#7-volume-holograms-and-bragg-diffraction)
8. [Digital Holography](#8-digital-holography)
9. [Applications](#9-applications)
10. [Python Examples](#10-python-examples)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. The Holographic Principle

### 1.1 The Phase Problem

Optical detectors (film, CCD, eyes) measure **intensity** $I = |U|^2$. The phase of the optical field is lost. If we could record $U = |U|e^{i\phi}$ directly, we could reconstruct the full 3D wavefront. But we cannot measure phase directly.

### 1.2 Gabor's Insight

Gabor realized that phase information can be encoded as intensity variations through **interference**. By adding a known **reference wave** $U_R$ to the unknown **object wave** $U_O$, the total intensity pattern contains the phase:

$$I = |U_R + U_O|^2 = |U_R|^2 + |U_O|^2 + U_R^*U_O + U_R U_O^*$$

The cross terms $U_R^*U_O$ and $U_R U_O^*$ encode the relative phase between the reference and object waves. This interference pattern — the **hologram** — is recorded on a photosensitive medium.

> **Analogy**: Imagine you want to record a complex melody (the object wave) but your recorder can only capture volume levels, not pitch (phase). Gabor's trick is to play a simple, known tone (the reference wave) at the same time. The beats and interference patterns between the melody and the tone encode the pitch information in the volume variations. To hear the original melody again, you play back the recording while simultaneously playing the same reference tone — the interference recreates the missing pitch information.

### 1.3 Requirements for Holography

1. **Coherent light source**: Both beams must be mutually coherent. The coherence length must exceed the maximum path difference. This is why holography became practical only after the invention of the laser (1960).

2. **Stable recording geometry**: Sub-wavelength stability during exposure ($< \lambda/4 \approx 150\,\text{nm}$). Any vibration blurs the interference fringes.

3. **High-resolution recording medium**: The fringe spacing can be as fine as $\lambda/(2\sin\theta) \approx 0.3\,\mu\text{m}$, requiring resolution > 3000 lines/mm (far beyond ordinary photography at ~200 lines/mm).

---

## 2. Recording a Hologram

### 2.1 Setup

A laser beam is split into two paths:
- **Reference beam** $U_R$: propagates directly to the recording medium
- **Object beam** $U_O$: illuminates the object, and the scattered/reflected light reaches the recording medium

```
                    Beam
    Laser ─────────splitter──────────→ Reference beam (U_R)
                      │                     │
                      ↓                     │
                   Object                   │
                      │                     │
                   Scattered                │
                   light (U_O)              │
                      │                     │
                      ↓                     ↓
                    ┌─────────────────────────┐
                    │     Recording medium     │
                    │  (film / photopolymer)   │
                    └─────────────────────────┘
```

### 2.2 Interference Pattern

At the recording plane, the intensity is:

$$I(x, y) = |U_R + U_O|^2 = I_R + I_O + U_R^*U_O + U_R U_O^*$$

where:
- $I_R = |U_R|^2$: reference beam intensity (uniform for a plane wave)
- $I_O = |U_O|^2$: object beam intensity (contains only amplitude info)
- $U_R^*U_O$: holographic term (encodes object phase relative to reference)
- $U_R U_O^*$: conjugate holographic term

### 2.3 Recording Medium Response

The recording medium (photographic film, photopolymer, photorefractive crystal) responds to the intensity pattern. For a linear recording medium, the **amplitude transmittance** after development is proportional to the exposure:

$$t(x, y) = t_0 + \beta I(x, y)$$

where $t_0$ is the bias transmittance and $\beta$ is the sensitivity (negative for film). The transmittance becomes:

$$t = t_0 + \beta(I_R + I_O) + \beta U_R^*U_O + \beta U_R U_O^*$$

The third and fourth terms are the holographic content.

---

## 3. Reconstructing the Wavefront

### 3.1 Illumination with the Reference Beam

To reconstruct, illuminate the hologram with the original reference beam $U_R$. The transmitted field is:

$$U_{\text{trans}} = t \cdot U_R = (t_0 + \beta I_R + \beta I_O)U_R + \beta|U_R|^2 U_O + \beta U_R^2 U_O^*$$

The three terms are:

1. **Zeroth order** $(t_0 + \beta I_R + \beta I_O)U_R$: A modified version of the reference beam (DC term, attenuated and modulated). This propagates straight through.

2. **Virtual image** $\beta|U_R|^2 U_O$: Proportional to $U_O$ (with a constant factor $\beta I_R$ for a plane-wave reference). This is an exact copy of the original object wave — a viewer looking through the hologram sees the object in its original 3D position **behind** the hologram.

3. **Conjugate (real) image** $\beta U_R^2 U_O^*$: Proportional to the complex conjugate of the object wave. This converges to form a real image **in front of** the hologram (pseudoscopic — inside-out depth).

### 3.2 The Key Result

The second term faithfully reconstructs the original object wavefront, including all amplitude and phase information. An observer looking through the hologram sees a three-dimensional image indistinguishable from the original scene.

### 3.3 Why This Works

The magic is that the reference beam acts as a **local oscillator** (in signal processing language) or a **heterodyne detector**. The cross-term $U_R^*U_O$ records the phase of $U_O$ as intensity modulation. Re-illumination with $U_R$ reverses the process: $U_R \cdot U_R^*U_O = |U_R|^2 U_O \propto U_O$.

---

## 4. Gabor (Inline) Holography

### 4.1 Configuration

In Gabor's original scheme (1948), the reference and object beams travel along the same axis. The object is partially transparent, and the unscattered part of the illuminating beam serves as the reference:

```
   Laser ────→ Object (partially transparent) ────→ Recording medium
              │                                      │
              └── Scattered (U_O) ──────────────────→│
              └── Unscattered (U_R) ────────────────→│
```

### 4.2 Advantages

- Simple setup, no separate reference beam path
- Only one coherent beam needed
- Compact geometry

### 4.3 Disadvantages

The three reconstructed terms (zeroth order, virtual image, conjugate image) all propagate **along the same axis**. They overlap spatially, causing:
- The DC term floods the image with unwanted light
- The conjugate image creates a defocused "twin image" superimposed on the virtual image

This twin-image problem severely limited Gabor's original holography. It was practical mainly for specialized applications like electron holography (Gabor's original motivation was improving electron microscope resolution).

---

## 5. Leith-Upatnieks (Off-Axis) Holography

### 5.1 The Off-Axis Solution (1962)

Emmett Leith and Juris Upatnieks introduced the **off-axis reference beam** — the reference wave arrives at an angle $\theta$ to the object beam. This simple but profound modification solved the twin-image problem.

For a plane-wave reference at angle $\theta$:

$$U_R = A_R\,e^{ikx\sin\theta}$$

The interference pattern now contains spatial carrier fringes at frequency $f_c = \sin\theta/\lambda$.

### 5.2 Spatial Separation of Terms

Upon reconstruction, the three terms propagate in different directions:

- **Zeroth order**: Continues along the reference beam direction
- **Virtual image (+1 order)**: Diffracted into the original object beam direction
- **Conjugate image (-1 order)**: Diffracted to the opposite side

```
                     Virtual image (U_O)
                   ╱
                 ╱ angle θ
   Reference ──╳────────→ Zeroth order (DC)
   beam        ╲
                 ╲ angle -θ
                   ╲
                     Conjugate image (U_O*)
```

The three terms are **spatially separated** as long as the angle $\theta$ is large enough:

$$\sin\theta > \frac{3}{2}\frac{\lambda}{d_{\min}}$$

where $d_{\min}$ is the smallest feature in the object. This condition ensures the three diffracted beams do not overlap in the output plane.

### 5.3 Practical Significance

The Leith-Upatnieks off-axis geometry made high-quality holography practical. Combined with the laser (invented 1960), it opened the door to all modern holographic applications. Their 1964 paper producing 3D holograms of real objects is a landmark in optics.

> **Analogy**: The difference between Gabor and Leith-Upatnieks holography is like the difference between AM radio and FM radio. In AM (Gabor), the signal is superimposed directly on the carrier, and any noise or overlapping stations cause interference. In FM (Leith-Upatnieks), the signal is encoded on a carrier at a specific frequency offset, which can be cleanly separated from other signals using a tuned filter. The off-axis angle is the "carrier frequency" that separates the useful signal from the noise.

---

## 6. Thin vs. Thick Holograms

### 6.1 The Q Parameter

The distinction between thin and thick holograms is governed by the **Klein-Cook Q parameter**:

$$Q = \frac{2\pi\lambda d}{n\Lambda^2}$$

where $d$ is the hologram thickness, $\Lambda$ is the fringe spacing, and $n$ is the refractive index.

- **Thin hologram** ($Q < 1$): Behaves as a 2D grating. Multiple diffraction orders are produced. Raman-Nath regime.
- **Thick (volume) hologram** ($Q > 10$): Behaves as a 3D grating. Only one diffraction order is efficiently produced (Bragg diffraction). Bragg regime.

### 6.2 Thin Holograms

- Recorded on thin photographic film (~5-15 $\mu$m)
- Multiple diffraction orders at angles $\sin\theta_m = m\lambda/\Lambda$
- Low diffraction efficiency (typically < 6% for amplitude holograms)
- Can be either amplitude or phase type

### 6.3 Phase vs. Amplitude Holograms

**Amplitude hologram**: Modulates the absorption of the medium. Maximum diffraction efficiency: $\eta_{\max} = 6.25\%$ (thin) — most of the light goes into the zeroth order or is absorbed.

**Phase hologram**: Modulates the refractive index or thickness without absorption. Maximum diffraction efficiency: $\eta_{\max} = 33.9\%$ (thin, Raman-Nath). Phase holograms are preferred because they are more efficient and brighter.

---

## 7. Volume Holograms and Bragg Diffraction

### 7.1 Bragg Condition

In a thick hologram, the interference fringes form a 3D grating — a periodic structure throughout the volume of the recording medium. Efficient diffraction occurs only when the **Bragg condition** is satisfied:

$$\boxed{2n\Lambda\sin\theta_B = \lambda}$$

where $\Lambda$ is the fringe spacing, $\theta_B$ is the Bragg angle (measured from the fringe planes), and $n$ is the refractive index.

### 7.2 Angular and Wavelength Selectivity

The Bragg condition imposes strict selectivity:

**Angular selectivity**: The hologram diffracts efficiently only within a narrow range of reconstruction angles:

$$\Delta\theta \approx \frac{\Lambda}{d}$$

For a 1 mm thick hologram with $\Lambda = 0.5\,\mu\text{m}$: $\Delta\theta \approx 0.5\,\text{mrad} \approx 0.03°$.

**Wavelength selectivity**: Only a narrow wavelength band is diffracted:

$$\Delta\lambda \approx \frac{\lambda\Lambda}{d}$$

This selectivity is what makes volume holograms useful for dense data storage and WDM filters.

### 7.3 Diffraction Efficiency

For a thick phase hologram (Kogelnik's coupled wave theory):

$$\eta = \sin^2\!\left(\frac{\pi\Delta n\, d}{\lambda\cos\theta_B}\right)$$

where $\Delta n$ is the refractive index modulation. When $\frac{\pi\Delta n\, d}{\lambda\cos\theta_B} = \frac{\pi}{2}$, efficiency reaches **100%** — all incident light is diffracted into a single order. This is dramatically better than thin holograms.

### 7.4 Reflection vs. Transmission Volume Holograms

**Transmission hologram**: Reference and object beams enter from the same side. Fringes are approximately perpendicular to the surface. Reconstructed with laser light from behind.

**Reflection hologram** (Denisyuk hologram): Reference and object beams enter from opposite sides. Fringes are approximately parallel to the surface. Can be viewed with white light — the Bragg wavelength selectivity acts as a color filter, selecting the correct wavelength from the broad spectrum. This is how museum display holograms and security holograms work.

---

## 8. Digital Holography

### 8.1 From Film to Sensor

In digital holography, the photographic film is replaced by a CCD or CMOS sensor. The hologram (interference pattern) is recorded as a digital image and the reconstruction is performed numerically on a computer.

### 8.2 Recording

The setup is similar to classical holography, but the recording medium is a digital sensor:

```
   Laser ────→ BS ────→ Reference beam
                │
                ↓
             Object ────→ CCD/CMOS sensor
                          (records I(x,y))
```

**Sampling requirement**: The sensor pixel pitch $\Delta p$ must satisfy the Nyquist condition for the highest fringe frequency:

$$\Delta p < \frac{\lambda}{2\sin\theta}$$

For $\lambda = 633\,\text{nm}$ and $\theta = 5°$: $\Delta p < 3.6\,\mu\text{m}$. Modern sensors with 1-5 $\mu\text{m}$ pixels can handle moderate off-axis angles.

### 8.3 Numerical Reconstruction

The recorded hologram $I(x, y)$ is multiplied by a numerical reference wave $U_R(x, y)$ and then propagated to the reconstruction plane using the Fresnel propagation integral (computed via FFT):

$$U_{\text{recon}}(x', y') = \mathcal{F}^{-1}\left\{\mathcal{F}\left\{I(x,y) \cdot U_R(x,y)\right\} \cdot H(f_x, f_y; d)\right\}$$

where $H$ is the Fresnel transfer function and $d$ is the reconstruction distance.

### 8.4 Advantages of Digital Holography

1. **Quantitative phase imaging**: The numerically reconstructed complex field gives both amplitude AND phase — enabling precise measurement of surface profiles, refractive index distributions, etc.

2. **Numerical refocusing**: A single recorded hologram can be numerically propagated to any distance $d$, allowing post-capture focusing without mechanical adjustment.

3. **Aberration correction**: Optical aberrations can be numerically compensated in post-processing.

4. **No chemical processing**: No darkroom, instant results.

5. **Time-resolved**: Fast sensors enable dynamic holography (digital holographic microscopy of living cells, vibration analysis).

---

## 9. Applications

### 9.1 3D Display

- **Art and museum displays**: Large reflection holograms viewable under white light
- **Head-up displays (HUD)**: Holographic optical elements in automotive and aviation
- **Holographic video**: Displays that generate dynamic holograms in real-time (MIT Media Lab, Looking Glass Factory)

### 9.2 Holographic Data Storage

Volume holograms can store data throughout the thickness of the medium, not just on a 2D surface. Each page of data is stored as a hologram at a different angle (angular multiplexing) or wavelength. Theoretical capacity: ~1 TB per disc (compared to 25-100 GB for Blu-ray).

### 9.3 Holographic Interferometry

Two holograms recorded at different times (or a double exposure) interfere to reveal minute changes in the object — deformations, vibrations, or refractive index changes. Sensitivity: sub-wavelength ($< \lambda/10 \approx 50\,\text{nm}$).

Applications: non-destructive testing of aircraft components, tire inspection, vibration mode analysis of musical instruments and loudspeakers.

### 9.4 Security Holograms

The rainbow holograms on credit cards, banknotes, and passports are embossed surface-relief holograms. They are extremely difficult to counterfeit because:
- They require coherent recording equipment
- Each hologram is a complex 3D microstructure
- Mass production requires an expensive electroformed master

### 9.5 Holographic Optical Elements (HOEs)

Holograms that function as optical elements: lenses, mirrors, gratings, beam splitters. Advantages: lightweight, flat, can combine multiple functions in one element. Used in head-mounted displays, barcode scanners, solar concentrators.

### 9.6 Digital Holographic Microscopy (DHM)

Quantitative phase imaging of biological cells without staining. Measures cell thickness, dry mass, and refractive index distribution. Real-time monitoring of cell dynamics, growth, and response to stimuli.

---

## 10. Python Examples

### 10.1 Digital Holographic Recording and Reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt

def create_object_wave(N, pixel_size, wavelength, object_distance):
    """
    Create a simple object wave: two point sources at different depths.

    We model each point source as a spherical wave emanating from
    a specific (x, y, z) position. The interference of multiple
    point sources at different depths creates a complex wavefront
    that encodes genuine 3D information — exactly what holography
    is designed to capture.
    """
    x = np.arange(-N//2, N//2) * pixel_size
    X, Y = np.meshgrid(x, x)
    k = 2 * np.pi / wavelength

    # Point source 1 — centered, distance d1
    d1 = object_distance
    r1 = np.sqrt(X**2 + Y**2 + d1**2)
    U1 = np.exp(1j * k * r1) / r1

    # Point source 2 — offset, distance d2 (different depth)
    d2 = object_distance * 1.2
    x_off, y_off = 50 * pixel_size, 30 * pixel_size
    r2 = np.sqrt((X - x_off)**2 + (Y - y_off)**2 + d2**2)
    U2 = 0.7 * np.exp(1j * k * r2) / r2  # Slightly dimmer

    return U1 + U2

def create_reference_wave(N, pixel_size, wavelength, angle_deg):
    """
    Create an off-axis plane wave reference beam.

    The reference angle must be large enough to spatially separate
    the three diffraction orders (DC, virtual image, conjugate image)
    upon reconstruction — this is the Leith-Upatnieks geometry.
    """
    x = np.arange(-N//2, N//2) * pixel_size
    X, Y = np.meshgrid(x, x)
    k = 2 * np.pi / wavelength
    angle = np.radians(angle_deg)

    # Plane wave tilted in x-direction
    U_R = np.exp(1j * k * X * np.sin(angle))
    return U_R

def fresnel_propagate(field, wavelength, pixel_size, distance):
    """
    Propagate a complex field using the angular spectrum method.

    This is the core numerical engine of digital holography.
    We decompose the field into plane waves (FFT), propagate
    each one by multiplying with the free-space transfer function,
    then recombine (inverse FFT). Exact within the scalar approximation.
    """
    N = field.shape[0]
    k = 2 * np.pi / wavelength

    # Spatial frequency grid
    df = 1.0 / (N * pixel_size)
    fx = np.arange(-N//2, N//2) * df
    FX, FY = np.meshgrid(fx, fx)

    # Transfer function of free space (angular spectrum)
    # Evanescent waves (fr > 1/lambda) are automatically handled
    # by the square root becoming imaginary → exponential decay
    arg = (1.0 / wavelength)**2 - FX**2 - FY**2
    H = np.exp(1j * 2 * np.pi * distance * np.sqrt(np.maximum(arg, 0).astype(complex)))
    H[arg < 0] = 0  # Kill evanescent waves explicitly

    # Propagate: FFT → multiply by H → inverse FFT
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    propagated = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum * H)))

    return propagated

# --- Parameters ---
N = 1024          # Grid size (pixels)
pixel_size = 5e-6 # 5 µm pixel pitch (typical CCD)
wavelength = 633e-9  # He-Ne laser, 633 nm
object_distance = 0.05  # 5 cm from object to hologram plane
ref_angle = 1.5   # Reference beam angle (degrees)

# --- Recording ---
U_obj = create_object_wave(N, pixel_size, wavelength, object_distance)
U_ref = create_reference_wave(N, pixel_size, wavelength, ref_angle)

# Hologram = interference pattern (intensity at the sensor)
hologram = np.abs(U_ref + U_obj)**2

# --- Reconstruction ---
# Multiply hologram by (numerical) reference wave
U_recon_field = hologram * U_ref

# Propagate back to object plane
U_reconstructed = fresnel_propagate(U_recon_field, wavelength, pixel_size,
                                     -object_distance)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Original object intensity
axes[0, 0].imshow(np.abs(U_obj)**2, cmap='hot')
axes[0, 0].set_title('Object Wave Intensity', fontsize=12)
axes[0, 0].axis('off')

# Recorded hologram
axes[0, 1].imshow(hologram, cmap='gray')
axes[0, 1].set_title('Recorded Hologram (Interference Pattern)', fontsize=12)
axes[0, 1].axis('off')

# Zoomed view of hologram fringes
axes[1, 0].imshow(hologram[N//2-50:N//2+50, N//2-50:N//2+50], cmap='gray')
axes[1, 0].set_title('Hologram Fringes (Zoomed)', fontsize=12)
axes[1, 0].axis('off')

# Reconstructed image
I_recon = np.abs(U_reconstructed)**2
axes[1, 1].imshow(I_recon, cmap='hot')
axes[1, 1].set_title('Reconstructed Image (Amplitude²)', fontsize=12)
axes[1, 1].axis('off')

plt.suptitle('Digital Holography: Recording and Reconstruction', fontsize=14)
plt.tight_layout()
plt.savefig('digital_holography.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Numerical refocusing: propagate to different distances ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
distances = [0.8, 0.9, 1.0, 1.1]  # Relative to object_distance

for i, rel_d in enumerate(distances):
    d = -object_distance * rel_d
    U_refocus = fresnel_propagate(U_recon_field, wavelength, pixel_size, d)
    axes[i].imshow(np.abs(U_refocus)**2, cmap='hot')
    axes[i].set_title(f'd = {rel_d:.1f} × d_obj', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Numerical Refocusing from a Single Hologram', fontsize=13)
plt.tight_layout()
plt.savefig('numerical_refocusing.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.2 Holographic Interference Fringes Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def hologram_fringe_spacing(wavelength, angle_deg):
    """
    Calculate the fringe spacing for an off-axis hologram.

    The fringe period Lambda = lambda / sin(theta) determines
    the spatial frequency of the carrier. The sensor pixel pitch
    must be smaller than Lambda/2 to satisfy Nyquist sampling.
    """
    angle_rad = np.radians(angle_deg)
    if np.sin(angle_rad) == 0:
        return np.inf
    return wavelength / np.sin(angle_rad)

# Analyze fringe spacing vs. reference angle
angles = np.linspace(0.5, 30, 100)
wavelengths = [633e-9, 532e-9, 405e-9]  # Red, Green, Blue
colors = ['red', 'green', 'blue']
labels = ['633 nm (He-Ne)', '532 nm (Nd:YAG 2ω)', '405 nm (diode)']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for wl, color, label in zip(wavelengths, colors, labels):
    spacings = [hologram_fringe_spacing(wl, a) * 1e6 for a in angles]  # µm
    ax1.semilogy(angles, spacings, color=color, linewidth=2, label=label)

# Mark typical sensor pixel pitches
for pitch, name in [(1.67, '1.67 µm'), (3.45, '3.45 µm'), (5.0, '5.0 µm')]:
    ax1.axhline(y=2*pitch, color='gray', linestyle='--', alpha=0.5)
    ax1.text(25, 2*pitch*1.1, f'Nyquist for {name} pixel',
             fontsize=8, color='gray')

ax1.set_xlabel('Reference beam angle (degrees)', fontsize=11)
ax1.set_ylabel('Fringe spacing (µm)', fontsize=11)
ax1.set_title('Hologram Fringe Spacing vs. Reference Angle', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(1, 100)

# Maximum recordable angle for different pixel sizes
pixel_pitches = np.linspace(1, 10, 100)  # µm
for wl, color, label in zip(wavelengths, colors, labels):
    # Nyquist: pixel_pitch < lambda / (2*sin(theta))
    # => sin(theta_max) = lambda / (2*pixel_pitch)
    sin_theta_max = (wl * 1e6) / (2 * pixel_pitches)
    theta_max = np.degrees(np.arcsin(np.clip(sin_theta_max, 0, 1)))
    ax2.plot(pixel_pitches, theta_max, color=color, linewidth=2, label=label)

ax2.set_xlabel('Sensor pixel pitch (µm)', fontsize=11)
ax2.set_ylabel('Maximum reference angle (degrees)', fontsize=11)
ax2.set_title('Maximum Off-Axis Angle vs. Pixel Pitch', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hologram_fringe_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.3 Volume Hologram Bragg Selectivity

```python
import numpy as np
import matplotlib.pyplot as plt

def bragg_efficiency(delta_theta, thickness, wavelength, n, fringe_spacing):
    """
    Approximate angular selectivity of a volume hologram.

    Kogelnik's coupled wave theory predicts that the diffraction
    efficiency drops as the reconstruction angle deviates from the
    Bragg angle. The sinc-like profile has a width inversely
    proportional to the hologram thickness — thicker holograms
    are more selective, enabling denser multiplexing.
    """
    # Bragg angle
    theta_B = np.arcsin(wavelength / (2 * n * fringe_spacing))

    # Detuning parameter (simplified)
    xi = np.pi * n * thickness * delta_theta / wavelength

    # Efficiency (sinc² envelope, simplified model)
    eta = np.sinc(xi / np.pi)**2
    return eta

# Parameters
wavelength = 532e-9  # Green laser
n = 1.5              # Typical photopolymer
fringe_spacing = 0.5e-6  # 0.5 µm

# Angular selectivity for different thicknesses
delta_theta = np.linspace(-5e-3, 5e-3, 1000)  # radians

fig, ax = plt.subplots(figsize=(10, 6))

for thickness_mm in [0.01, 0.1, 1.0]:
    thickness = thickness_mm * 1e-3  # Convert to meters
    eta = bragg_efficiency(delta_theta, thickness, wavelength, n, fringe_spacing)
    ax.plot(np.degrees(delta_theta) * 1000, eta, linewidth=2,
            label=f'd = {thickness_mm} mm')

ax.set_xlabel('Angular detuning from Bragg angle (millidegrees)', fontsize=11)
ax.set_ylabel('Relative diffraction efficiency', fontsize=11)
ax.set_title('Volume Hologram Angular Selectivity', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-300, 300)

plt.tight_layout()
plt.savefig('bragg_selectivity.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Holographic recording | $I = \|U_R + U_O\|^2 = I_R + I_O + U_R^*U_O + U_RU_O^*$ |
| Reconstruction | Illuminate with $U_R$: get $\|U_R\|^2 U_O$ (virtual image) |
| Gabor (inline) | Simple setup; twin-image problem (all orders overlap) |
| Leith-Upatnieks (off-axis) | Off-axis reference; spatially separates three orders |
| Thin hologram | $Q < 1$; multiple diffraction orders; $\eta_{\max} = 33.9\%$ (phase) |
| Volume hologram | $Q > 10$; single Bragg order; $\eta$ up to 100% |
| Bragg condition | $2n\Lambda\sin\theta_B = \lambda$ |
| Angular selectivity | $\Delta\theta \approx \Lambda/d$ (narrows with thickness) |
| Digital holography | Record on CCD → numerical reconstruction via FFT |
| Numerical refocusing | Single hologram → reconstruct at any depth $d$ |
| Phase imaging | Reconstructed complex field gives quantitative phase |

---

## 12. Exercises

### Exercise 1: Hologram Resolution Requirements

You want to record an off-axis hologram at $\lambda = 532\,\text{nm}$ with a reference beam angle of $\theta = 10°$.

(a) Calculate the fringe spacing.
(b) What is the minimum resolution (lines/mm) needed for the recording medium?
(c) Your CCD sensor has 3.45 $\mu$m pixel pitch. Can it record this hologram?
(d) What is the maximum off-axis angle this sensor can handle at 532 nm?

### Exercise 2: Volume Hologram Design

You want to design a reflection hologram for display at $\lambda = 633\,\text{nm}$ ($n = 1.5$).

(a) Calculate the fringe spacing for normal incidence ($\theta_B = 90°$, fringes parallel to surface).
(b) For a 20 $\mu$m thick emulsion, calculate the wavelength selectivity $\Delta\lambda$.
(c) Will this hologram reconstruct well under white light? Explain.
(d) What refractive index modulation $\Delta n$ is needed for 100% diffraction efficiency?

### Exercise 3: Digital Holography Simulation

Modify the Python code from Section 10.1 to:
(a) Create a more complex object: the letter "H" made of multiple point sources.
(b) Record and reconstruct the hologram.
(c) Demonstrate numerical refocusing by placing some point sources at different depths.
(d) Add noise to the recorded hologram (simulating camera noise) and observe the effect on reconstruction quality.

### Exercise 4: Gabor vs. Off-Axis Comparison

Using Python, simulate both Gabor (inline) and Leith-Upatnieks (off-axis) holography for the same object.

(a) For the Gabor case, show the twin-image artifact.
(b) For the off-axis case, show the spatial separation of the three terms in the Fourier domain.
(c) Apply a spatial filter in the Fourier domain to isolate the virtual image term in the off-axis case.
(d) Compare the reconstructed image quality for both methods.

---

## 13. References

1. Gabor, D. (1948). "A new microscopic principle." *Nature*, 161, 777-778.
2. Leith, E. N., & Upatnieks, J. (1962). "Reconstructed wavefronts and communication theory." *JOSA*, 52(10), 1123-1130.
3. Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W. H. Freeman. — Chapter 9.
4. Hariharan, P. (2002). *Basics of Holography*. Cambridge University Press.
5. Schnars, U., & Jueptner, W. (2005). *Digital Holography*. Springer.
6. Kogelnik, H. (1969). "Coupled wave theory for thick hologram gratings." *Bell System Technical Journal*, 48(9), 2909-2947.

---

[← Previous: 10. Fourier Optics](10_Fourier_Optics.md) | [Next: 12. Nonlinear Optics →](12_Nonlinear_Optics.md)

# 05. Wave Optics — Interference

[← Previous: 04. Optical Instruments](04_Optical_Instruments.md) | [Next: 06. Diffraction →](06_Diffraction.md)

---

## Learning Objectives

1. Apply the superposition principle to calculate interference patterns from two or more coherent sources
2. Derive the intensity distribution of Young's double-slit experiment and predict fringe spacing
3. Analyze thin-film interference including phase shifts upon reflection and conditions for constructive/destructive interference
4. Explain the operation of the Michelson interferometer and its applications in metrology
5. Distinguish between temporal and spatial coherence and relate them to source properties
6. Calculate coherence length and coherence time from the spectral width of a light source
7. Apply interference concepts to practical systems: anti-reflection coatings, Newton's rings, Fabry-Perot etalons

---

## Why This Matters

Interference is the definitive signature of wave behavior. It was Young's double-slit experiment in 1801 that settled the centuries-old debate about the nature of light in favor of the wave theory. Today, interference underpins some of humanity's most precise measurements: LIGO detects gravitational waves by measuring length changes of $10^{-18}$ m using laser interferometry — a ten-thousandth the diameter of a proton. Anti-reflection coatings on every camera lens and pair of eyeglasses rely on thin-film interference. Fiber-optic sensors, optical coherence tomography (OCT) in medicine, and the calibration of the meter itself all depend on understanding interference.

> **Analogy**: Interference is like two speakers playing the same note in a room. At some locations the sound waves add up (loud spots), at others they cancel (quiet spots). Now replace "sound" with "light" and "loud/quiet" with "bright/dark" — you get interference fringes. The key requirement is the same: the two sources must be coherent (maintaining a constant phase relationship).

---

## 1. The Superposition Principle

### 1.1 Statement

When two or more waves overlap in space, the resulting electric field is the vector sum of the individual fields:

$$\mathbf{E}_{\text{total}}(\mathbf{r}, t) = \mathbf{E}_1(\mathbf{r}, t) + \mathbf{E}_2(\mathbf{r}, t) + \cdots$$

This linearity is a direct consequence of Maxwell's equations being linear in vacuum and in linear media.

### 1.2 Intensity of Superposed Waves

For two monochromatic waves with the same frequency $\omega$ and polarization:

$$E_1 = E_{01} \cos(k r_1 - \omega t + \phi_1)$$
$$E_2 = E_{02} \cos(k r_2 - \omega t + \phi_2)$$

The time-averaged intensity at a point where both waves are present:

$$I = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos\delta$$

where $\delta$ is the **phase difference**:

$$\delta = k(r_2 - r_1) + (\phi_2 - \phi_1) = \frac{2\pi}{\lambda}\Delta r + \Delta\phi$$

and $\Delta r = r_2 - r_1$ is the **path difference**.

**Key cases**:
- $\delta = 0, \pm 2\pi, \pm 4\pi, \ldots$ → **Constructive interference**: $I = I_1 + I_2 + 2\sqrt{I_1 I_2} = (\sqrt{I_1} + \sqrt{I_2})^2$
- $\delta = \pm\pi, \pm 3\pi, \ldots$ → **Destructive interference**: $I = (\sqrt{I_1} - \sqrt{I_2})^2$

For equal intensities $I_1 = I_2 = I_0$:

$$I = 2I_0(1 + \cos\delta) = 4I_0 \cos^2\left(\frac{\delta}{2}\right)$$

The maximum intensity is $4I_0$ (not $2I_0$) — energy is not created or destroyed but *redistributed* from dark fringes to bright fringes.

```python
import numpy as np
import matplotlib.pyplot as plt

# Interference of two plane waves: intensity as a function of phase difference
# Shows how intensity oscillates between 0 and 4*I_0

delta = np.linspace(-4*np.pi, 4*np.pi, 500)

# Equal intensity case: I = 4*I_0 * cos^2(delta/2)
I_equal = 4 * np.cos(delta / 2)**2

# Unequal intensity case: I1 = 1, I2 = 0.25
I1, I2 = 1.0, 0.25
I_unequal = I1 + I2 + 2*np.sqrt(I1*I2) * np.cos(delta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: equal intensities
ax1.plot(delta/np.pi, I_equal, 'b-', linewidth=2)
ax1.set_xlabel('Phase difference $\\delta / \\pi$', fontsize=12)
ax1.set_ylabel('Intensity $I / I_0$', fontsize=12)
ax1.set_title('Interference: Equal Intensities ($I_1 = I_2 = I_0$)', fontsize=13)
ax1.axhline(2, color='gray', linestyle='--', alpha=0.5, label='$2I_0$ (no interference)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.2, 4.5)

# Right: unequal intensities
ax2.plot(delta/np.pi, I_unequal, 'r-', linewidth=2)
ax2.set_xlabel('Phase difference $\\delta / \\pi$', fontsize=12)
ax2.set_ylabel('Intensity', fontsize=12)
ax2.set_title('Interference: Unequal Intensities ($I_1 = 1, I_2 = 0.25$)', fontsize=13)
ax2.axhline(I1 + I2, color='gray', linestyle='--', alpha=0.5,
            label=f'$I_1 + I_2$ = {I1+I2} (no interference)')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Mark the visibility/contrast
I_max = (np.sqrt(I1) + np.sqrt(I2))**2
I_min = (np.sqrt(I1) - np.sqrt(I2))**2
V = (I_max - I_min) / (I_max + I_min)
ax2.annotate(f'$I_{{max}}$ = {I_max:.2f}\n$I_{{min}}$ = {I_min:.2f}\nVisibility V = {V:.2f}',
             xy=(0, I_max), xytext=(1.5, I_max-0.3), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig('superposition_intensity.png', dpi=150)
plt.show()
```

### 1.3 Fringe Visibility (Contrast)

The **visibility** (or **contrast**) of an interference pattern quantifies how well-defined the fringes are:

$$V = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$$

For equal-intensity beams: $V = 1$ (perfect contrast).
For unequal intensities: $V = \frac{2\sqrt{I_1 I_2}}{I_1 + I_2} < 1$.

Visibility also decreases with partial coherence (see Section 6).

---

## 2. Young's Double-Slit Experiment

### 2.1 Setup and Analysis

Thomas Young's experiment (1801) is one of the most important experiments in physics. Monochromatic light illuminates two narrow slits separated by distance $d$. On a screen at distance $L \gg d$, an interference pattern appears.

The path difference to a point at angle $\theta$ on the screen:

$$\Delta r = d\sin\theta$$

**Bright fringes** (constructive interference):

$$d\sin\theta = m\lambda, \qquad m = 0, \pm 1, \pm 2, \ldots$$

**Dark fringes** (destructive interference):

$$d\sin\theta = \left(m + \frac{1}{2}\right)\lambda, \qquad m = 0, \pm 1, \pm 2, \ldots$$

### 2.2 Fringe Spacing

For small angles ($\sin\theta \approx \tan\theta = y/L$), the position of the $m$-th bright fringe on the screen:

$$y_m = \frac{m\lambda L}{d}$$

The **fringe spacing** (distance between adjacent bright fringes):

$$\Delta y = \frac{\lambda L}{d}$$

This is one of the most elegant results in optics: the fringe spacing is directly proportional to the wavelength and inversely proportional to the slit separation.

### 2.3 Intensity Distribution

The intensity pattern for two identical narrow slits:

$$I(\theta) = 4I_0 \cos^2\left(\frac{\pi d \sin\theta}{\lambda}\right)$$

where $I_0$ is the intensity from a single slit. The $\cos^2$ pattern has maxima at $d\sin\theta = m\lambda$ and zeros at $d\sin\theta = (m+\frac{1}{2})\lambda$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Young's double-slit experiment: intensity pattern on the screen

wavelength = 550e-9      # green light (m)
d = 0.2e-3               # slit separation: 0.2 mm
L = 1.0                  # screen distance: 1 m

# Position on screen
y = np.linspace(-0.01, 0.01, 1000)  # ±10 mm
theta = np.arctan(y / L)             # angle (exact)

# Two-slit interference pattern (ignoring single-slit diffraction envelope)
delta = 2 * np.pi * d * np.sin(theta) / wavelength
I_two_slit = 4 * np.cos(delta / 2)**2

# Fringe spacing
fringe_spacing = wavelength * L / d * 1000  # in mm
print(f"Fringe spacing: {fringe_spacing:.2f} mm")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top: intensity vs position on screen
ax1.plot(y * 1000, I_two_slit, 'b-', linewidth=1.5)
ax1.set_xlabel('Position on screen (mm)', fontsize=12)
ax1.set_ylabel('Intensity $I / I_0$', fontsize=12)
ax1.set_title(f"Young's Double-Slit: d = {d*1e3:.1f} mm, L = {L:.1f} m, "
              f"λ = {wavelength*1e9:.0f} nm\nFringe spacing = {fringe_spacing:.2f} mm",
              fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 4.5)

# Mark the central maximum and first-order fringes
for m in range(-3, 4):
    y_m = m * wavelength * L / d * 1000  # in mm
    if abs(y_m) < 10:
        ax1.axvline(y_m, color='red', linestyle=':', alpha=0.4)
        ax1.text(y_m, 4.2, f'm={m}', ha='center', fontsize=8, color='red')

# Bottom: 2D visualization of the interference pattern
Y, X = np.meshgrid(np.linspace(-8, 8, 400), np.linspace(-3, 3, 100))
theta_2d = np.arctan(Y / (L * 1000))
I_2d = 4 * np.cos(np.pi * d * np.sin(theta_2d) / wavelength)**2
ax2.imshow(I_2d, extent=[-8, 8, -3, 3], aspect='auto', cmap='inferno',
           vmin=0, vmax=4)
ax2.set_xlabel('Position on screen (mm)', fontsize=12)
ax2.set_ylabel('Vertical position (mm)', fontsize=12)
ax2.set_title('Fringe Pattern (2D View)', fontsize=13)

plt.tight_layout()
plt.savefig('youngs_double_slit.png', dpi=150)
plt.show()
```

---

## 3. Thin-Film Interference

### 3.1 Phase Shifts upon Reflection

When light reflects at an interface, the phase shift depends on the refractive indices:

- **Reflection from a denser medium** ($n_1 < n_2$): Phase shift of $\pi$ (half wavelength)
- **Reflection from a less dense medium** ($n_1 > n_2$): No phase shift

This is analogous to a wave on a string: reflection from a fixed end (denser medium) inverts the pulse; reflection from a free end (less dense medium) does not.

### 3.2 Conditions for Thin-Film Interference

Consider a thin film of thickness $t$ and refractive index $n_f$ sandwiched between media with indices $n_1$ (above) and $n_2$ (below). Light reflects from both the top and bottom surfaces of the film.

The total phase difference between the two reflected beams:

$$\delta = \frac{2\pi}{\lambda} \cdot 2n_f t\cos\theta_t + \delta_{\text{reflection}}$$

where $\theta_t$ is the refraction angle inside the film and $\delta_{\text{reflection}}$ accounts for phase shifts at the two interfaces.

**Common case: Air-film-glass** ($n_1 < n_f < n_2$):
- Top surface: $\pi$ phase shift (air → film, reflecting from denser medium)
- Bottom surface: $\pi$ phase shift (film → glass, reflecting from denser medium)
- Net reflection phase shift: $\pi - \pi = 0$ (the two $\pi$ shifts cancel!)

**Constructive reflection**: $2n_f t\cos\theta_t = m\lambda$ ($m = 0, 1, 2, \ldots$)

**Common case: Air-film-air** ($n_1 = n_2 < n_f$, e.g., soap film):
- Top surface: $\pi$ phase shift
- Bottom surface: no phase shift
- Net reflection phase shift: $\pi$

**Constructive reflection**: $2n_f t\cos\theta_t = (m + \frac{1}{2})\lambda$

> **Analogy**: Think of the two reflected beams as two runners doing laps around a track. One runner (reflected from the top) starts immediately; the other (reflected from the bottom) first runs down through the film and back up before starting. If they arrive back at the start line in step, you get a bright reflection (constructive). If one is half a lap ahead of the other, they cancel (destructive). The "half-lap head start" from phase shifts on reflection changes the conditions for being in step.

### 3.3 Anti-Reflection Coatings

To minimize reflection from a glass surface ($n_g \approx 1.52$), apply a coating of thickness $t$ and index $n_c$ such that:

1. **Amplitude matching**: $n_c = \sqrt{n_{\text{air}} \cdot n_g} = \sqrt{1.52} \approx 1.23$ (minimum reflection when both reflected beams have equal amplitude)
2. **Destructive interference**: $2n_c t = \frac{\lambda}{2}$ → $t = \frac{\lambda}{4n_c}$ (quarter-wave thickness)

For $\lambda = 550$ nm and $n_c = 1.23$: $t = 112$ nm.

MgF$_2$ ($n = 1.38$) is the most common single-layer coating. The ideal $n_c = 1.23$ is difficult to achieve, so MgF$_2$ is a practical compromise.

Multi-layer coatings (stacks of alternating high-$n$ and low-$n$ layers) can achieve reflectance below 0.1% over a broad wavelength range — these are **broadband anti-reflection (BBAR)** coatings.

```python
import numpy as np
import matplotlib.pyplot as plt

# Thin-film interference: reflectance of a single-layer anti-reflection coating
# as a function of wavelength

def single_layer_reflectance(wavelength, t, n_coat, n_glass, n_air=1.0):
    """
    Calculate reflectance of a single-layer coating on glass.
    Uses the exact formula from thin-film optics (normal incidence).

    wavelength: in nm
    t: coating thickness in nm
    n_coat: coating refractive index
    n_glass: glass refractive index
    n_air: ambient refractive index (usually 1.0)
    """
    # Fresnel reflection coefficients at each interface (normal incidence)
    r1 = (n_air - n_coat) / (n_air + n_coat)   # air → coating
    r2 = (n_coat - n_glass) / (n_coat + n_glass)  # coating → glass

    # Phase accumulated in the film (round trip)
    delta = 4 * np.pi * n_coat * t / wavelength

    # Total reflectance (Airy formula for a single film)
    numerator = r1**2 + r2**2 + 2 * r1 * r2 * np.cos(delta)
    denominator = 1 + r1**2 * r2**2 + 2 * r1 * r2 * np.cos(delta)
    R = numerator / denominator
    return R

# Wavelength range: 350 to 800 nm (covering visible spectrum)
wavelengths = np.linspace(350, 800, 500)
n_glass = 1.52

# Design wavelength: 550 nm (green, center of visible spectrum)
lambda_design = 550.0

# Case 1: MgF2 coating (n = 1.38), quarter-wave at 550 nm
n_mgf2 = 1.38
t_mgf2 = lambda_design / (4 * n_mgf2)

# Case 2: Ideal coating (n = sqrt(1.52) ≈ 1.233)
n_ideal = np.sqrt(n_glass)
t_ideal = lambda_design / (4 * n_ideal)

# Case 3: No coating (bare glass)
R_bare = ((1 - n_glass) / (1 + n_glass))**2

R_mgf2 = single_layer_reflectance(wavelengths, t_mgf2, n_mgf2, n_glass)
R_ideal = single_layer_reflectance(wavelengths, t_ideal, n_ideal, n_glass)

fig, ax = plt.subplots(figsize=(12, 6))

ax.axhline(R_bare * 100, color='gray', linestyle='--', linewidth=1.5,
           label=f'Bare glass ({R_bare*100:.1f}%)')
ax.plot(wavelengths, R_mgf2 * 100, 'b-', linewidth=2,
        label=f'MgF$_2$ (n={n_mgf2}, t={t_mgf2:.0f} nm)')
ax.plot(wavelengths, R_ideal * 100, 'r-', linewidth=2,
        label=f'Ideal (n={n_ideal:.3f}, t={t_ideal:.0f} nm)')

# Shade the visible spectrum for reference
ax.axvspan(380, 700, alpha=0.08, color='yellow', label='Visible range')

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Reflectance (%)', fontsize=12)
ax.set_title('Single-Layer Anti-Reflection Coating Performance', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 5)
ax.set_xlim(350, 800)

# Annotate the minimum
idx_min_mgf2 = np.argmin(R_mgf2)
ax.annotate(f'Min: {R_mgf2[idx_min_mgf2]*100:.2f}% at {wavelengths[idx_min_mgf2]:.0f} nm',
            xy=(wavelengths[idx_min_mgf2], R_mgf2[idx_min_mgf2]*100),
            xytext=(wavelengths[idx_min_mgf2]+80, 1.5),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('anti_reflection_coating.png', dpi=150)
plt.show()
```

### 3.4 Newton's Rings

When a plano-convex lens is placed on a flat glass surface, the thin air gap between them varies with radial distance. This produces circular interference fringes called **Newton's rings**.

The air gap thickness at radius $r$ from the contact point, for a lens with radius of curvature $R$:

$$t(r) = \frac{r^2}{2R}$$

Bright rings (in reflected light, accounting for the $\pi$ phase shift at the lower interface):

$$r_m = \sqrt{\left(m + \frac{1}{2}\right)\lambda R}, \qquad m = 0, 1, 2, \ldots$$

Dark rings:

$$r_m = \sqrt{m \lambda R}, \qquad m = 0, 1, 2, \ldots$$

The central spot is **dark** in reflection (due to the $\pi$ phase shift at the glass-air-glass interface) and **bright** in transmission.

### 3.5 Soap Bubbles and Oil Films

The iridescent colors of soap bubbles and oil films on water arise from thin-film interference. The film thickness varies across the surface, so different regions satisfy the constructive interference condition for different wavelengths, producing a rainbow of colors.

For a soap film ($n \approx 1.33$) in air ($n_1 = n_2 = 1$):
- Top surface reflection: $\pi$ phase shift
- Bottom surface reflection: no phase shift
- Constructive reflection: $2nt = (m + \frac{1}{2})\lambda$

A film of thickness 300 nm strongly reflects blue ($\lambda \approx 400$ nm for $m = 0$) and appears blue. As the film thins (due to gravity), the colors shift and eventually the film becomes so thin that it reflects no visible light — appearing black just before it pops.

---

## 4. The Michelson Interferometer

### 4.1 Design and Operation

The Michelson interferometer splits a beam of light into two paths, reflects each back, and recombines them to form interference fringes. It is one of the most versatile and precise optical instruments ever devised.

**Components**:
1. **Beam splitter**: A half-silvered mirror that transmits half the light and reflects half
2. **Mirror 1** (fixed): Reflects the transmitted beam back
3. **Mirror 2** (movable): Reflects the reflected beam back
4. **Detector/screen**: Where the recombined beams interfere

The optical path difference between the two arms:

$$\Delta = 2(d_1 - d_2)$$

where $d_1$ and $d_2$ are the distances from the beam splitter to each mirror.

### 4.2 Fringe Patterns

**Circular fringes** (equal inclination fringes): When the mirrors are exactly perpendicular, rings of equal path difference appear centered on the optical axis. The condition for bright rings:

$$2d\cos\theta = m\lambda$$

where $d = |d_1 - d_2|$ is the path difference at normal incidence.

**Straight fringes** (equal thickness fringes): When one mirror is slightly tilted, the effective air wedge produces parallel fringes.

As mirror 2 is translated by $\lambda/2$, each fringe shifts by one full fringe spacing. By counting fringes, one can measure displacements with sub-wavelength precision.

### 4.3 Applications

**Metrology**: Measuring lengths with precision better than $\lambda/10 \approx 50$ nm.

**Spectroscopy (FTIR)**: Fourier-transform infrared spectroscopy uses a scanning Michelson interferometer. The interferogram (intensity vs. mirror displacement) is the Fourier transform of the spectrum.

**Gravitational wave detection**: LIGO uses kilometer-scale Michelson interferometers with Fabry-Perot cavities in each arm, achieving displacement sensitivity of $10^{-18}$ m.

**Michelson-Morley experiment (1887)**: The most famous null result in physics. The experiment sought to detect Earth's motion through the "luminiferous aether" by looking for a directional dependence of the speed of light. The null result led directly to Einstein's special relativity.

```python
import numpy as np
import matplotlib.pyplot as plt

# Michelson interferometer: simulate circular fringe patterns
# for different mirror separations

def michelson_fringes(d_um, wavelength_nm, max_angle_deg=5, N_points=500):
    """
    Calculate the Michelson interferometer fringe pattern.

    d_um: path difference in micrometers
    wavelength_nm: wavelength in nanometers
    max_angle_deg: maximum viewing angle in degrees
    N_points: number of points in each dimension

    Returns: 2D intensity array and extent
    """
    wavelength_um = wavelength_nm / 1000  # convert to micrometers

    # Create 2D grid of angles
    theta = np.linspace(-np.deg2rad(max_angle_deg),
                         np.deg2rad(max_angle_deg), N_points)
    TX, TY = np.meshgrid(theta, theta)
    angle = np.sqrt(TX**2 + TY**2)  # radial angle

    # Phase difference: delta = 4*pi*d*cos(theta) / lambda
    delta = 4 * np.pi * d_um * np.cos(angle) / wavelength_um

    # Intensity: I = I_0 * (1 + cos(delta)) / 2 (for 50/50 beam splitter)
    I = (1 + np.cos(delta)) / 2

    extent = [-max_angle_deg, max_angle_deg, -max_angle_deg, max_angle_deg]
    return I, extent

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Show fringe patterns for increasing path differences
# Larger d means more fringes (higher order at center)
path_diffs = [5, 20, 50, 100]  # micrometers

for ax, d in zip(axes, path_diffs):
    I, extent = michelson_fringes(d, 550)
    ax.imshow(I, extent=extent, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'd = {d} μm\n({2*d/0.55:.0f} fringes)', fontsize=11)
    ax.set_xlabel('Angle (deg)', fontsize=9)
    ax.set_ylabel('Angle (deg)', fontsize=9)

plt.suptitle('Michelson Interferometer: Circular Fringes (λ = 550 nm)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('michelson_fringes.png', dpi=150)
plt.show()
```

---

## 5. Multiple-Beam Interference: Fabry-Perot Etalon

### 5.1 From Two Beams to Many Beams

In thin-film interference, we considered only two reflected beams. In reality, light bounces back and forth multiple times inside the film. When the surface reflectivity is high ($R \to 1$), these multiple reflections become significant and produce much sharper fringes than the two-beam pattern.

### 5.2 The Airy Function

For a Fabry-Perot etalon (two parallel, highly reflective mirrors separated by distance $d$), the transmitted intensity is:

$$I_t = \frac{I_0}{1 + F\sin^2(\delta/2)}$$

where:
- $\delta = \frac{4\pi n d \cos\theta}{\lambda}$ is the round-trip phase
- $F = \frac{4R}{(1-R)^2}$ is the **coefficient of finesse**
- $R$ is the reflectance of each mirror

This is the **Airy function**. It has sharp transmission peaks (bright fringes) at $\delta = 2m\pi$ and broad, dark regions in between.

### 5.3 Finesse and Resolution

The **finesse** $\mathcal{F}$ measures the sharpness of the transmission peaks:

$$\mathcal{F} = \frac{\pi\sqrt{F}}{2} = \frac{\pi\sqrt{R}}{1-R}$$

| Mirror Reflectance $R$ | Finesse $\mathcal{F}$ | Coefficient $F$ |
|:-----------------------:|:---------------------:|:---------------:|
| 0.04 (bare glass) | 0.64 | 0.17 |
| 0.50 | 4.4 | 8.0 |
| 0.90 | 30 | 360 |
| 0.99 | 313 | 39,600 |

The **free spectral range** (FSR) is the frequency separation between adjacent transmission peaks:

$$\Delta\nu_{\text{FSR}} = \frac{c}{2nd}$$

The **spectral resolution** is:

$$\delta\nu = \frac{\Delta\nu_{\text{FSR}}}{\mathcal{F}} = \frac{c}{2nd\mathcal{F}}$$

The **resolving power**:

$$\mathcal{R} = \frac{\nu}{\delta\nu} = m \cdot \mathcal{F}$$

where $m$ is the interference order. A Fabry-Perot etalon with $\mathcal{F} = 30$ operating at order $m = 10^4$ can resolve spectral features with $\mathcal{R} = 3 \times 10^5$ — far exceeding diffraction gratings.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fabry-Perot etalon: transmission (Airy function) for different reflectivities
# Shows how higher reflectivity produces sharper transmission peaks

delta = np.linspace(0, 6 * np.pi, 1000)  # phase in radians

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

reflectivities = [0.04, 0.30, 0.70, 0.90, 0.97]

for R in reflectivities:
    F = 4 * R / (1 - R)**2  # coefficient of finesse
    finesse = np.pi * np.sqrt(R) / (1 - R)
    # Airy function: transmitted intensity
    I_t = 1 / (1 + F * np.sin(delta / 2)**2)

    ax1.plot(delta / np.pi, I_t, linewidth=1.5 + R,
             label=f'R = {R:.2f} ($\\mathcal{{F}}$ = {finesse:.1f})')

ax1.set_xlabel('Phase $\\delta / \\pi$', fontsize=12)
ax1.set_ylabel('Transmitted Intensity $I_t / I_0$', fontsize=12)
ax1.set_title('Fabry-Perot Transmission (Airy Function)', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.1)

# Right: resolving two closely spaced wavelengths with a Fabry-Perot
# Two wavelengths differ by delta_lambda, which maps to a phase shift
R = 0.90
F = 4 * R / (1 - R)**2
finesse = np.pi * np.sqrt(R) / (1 - R)

# Fine phase grid around a single order
phase_fine = np.linspace(1.8 * np.pi, 2.2 * np.pi, 1000)

# Two wavelengths separated by different amounts
separations = [0.02*np.pi, 0.05*np.pi, 0.10*np.pi]  # phase separations

for sep in separations:
    I1 = 1 / (1 + F * np.sin(phase_fine / 2)**2)
    I2 = 1 / (1 + F * np.sin((phase_fine - sep) / 2)**2)
    I_total = I1 + I2

    # Normalize for display
    ax2.plot(phase_fine / np.pi, I_total / I_total.max(), linewidth=2,
             label=f'$\\Delta\\delta = {sep/np.pi:.2f}\\pi$')

ax2.set_xlabel('Phase $\\delta / \\pi$', fontsize=12)
ax2.set_ylabel('Combined Intensity (normalized)', fontsize=12)
ax2.set_title(f'Resolving Two Wavelengths (R={R}, $\\mathcal{{F}}$={finesse:.0f})', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fabry_perot.png', dpi=150)
plt.show()
```

---

## 6. Coherence

### 6.1 Why Coherence Matters

Real light sources are never perfectly monochromatic or perfectly point-like. The degree to which a source can produce stable interference fringes is quantified by its **coherence**.

### 6.2 Temporal Coherence

**Temporal coherence** measures how well-correlated the wave is with a delayed version of itself. It is related to the spectral width $\Delta\nu$ of the source:

$$\tau_c = \frac{1}{\Delta\nu} \qquad (\text{coherence time})$$

$$\ell_c = c\tau_c = \frac{c}{\Delta\nu} = \frac{\lambda^2}{\Delta\lambda} \qquad (\text{coherence length})$$

| Source | $\Delta\lambda$ | Coherence Length |
|--------|-----------------|-----------------|
| White light | $\sim 300$ nm | $\sim 1$ $\mu$m |
| LED | $\sim 30$ nm | $\sim 10$ $\mu$m |
| Sodium lamp (D line) | $\sim 0.02$ nm | $\sim 15$ mm |
| HeNe laser | $\sim 0.001$ nm | $\sim 30$ cm |
| Stabilized laser | $\sim 10^{-9}$ nm | $\sim 300$ km |

For a Michelson interferometer, fringes are visible only when the path difference $\Delta < \ell_c$. This is why white light produces only a few colored fringes near zero path difference, while a laser produces millions of fringes.

### 6.3 Spatial Coherence

**Spatial coherence** measures how well-correlated the wave is at two different points in space perpendicular to the propagation direction. It depends on the angular size $\Delta\theta$ of the source as seen from the observation point:

$$d_c = \frac{\lambda}{\Delta\theta} \qquad (\text{coherence width})$$

For Young's double-slit experiment to produce clear fringes, the slit separation must be less than the coherence width: $d < d_c$.

The Sun ($\Delta\theta \approx 0.5° = 0.0087$ rad) has a spatial coherence width of about:

$$d_c = \frac{550 \text{ nm}}{0.0087} \approx 63 \text{ μm}$$

This is why sunlight produces weak interference in most setups — the slits must be extremely close.

### 6.4 The Van Cittert-Zernike Theorem

This theorem provides a quantitative link between the spatial intensity distribution of a source and the coherence of the field it produces. The spatial coherence function (complex degree of coherence) is the normalized Fourier transform of the source intensity distribution.

---

## 7. Interference in Practice

### 7.1 Optical Testing

Interferometers are the standard tool for testing optical surfaces. A Twyman-Green interferometer (a variant of the Michelson) compares the wavefront reflected from a test surface against a reference flat. Deviations from straightness in the fringe pattern reveal surface imperfections with nanometer precision.

### 7.2 Optical Coherence Tomography (OCT)

OCT uses a Michelson interferometer with a low-coherence (broad bandwidth) source to produce cross-sectional images of biological tissue. The short coherence length ($\sim 10$ $\mu$m) acts as a depth gate: only light from a specific depth in the tissue interferes with the reference beam. By scanning the reference mirror, a depth profile is built up.

### 7.3 Gravitational Wave Detection (LIGO)

LIGO's interferometers have 4-km arms with Fabry-Perot cavities (effective path length $\sim 1200$ km). A passing gravitational wave stretches one arm while compressing the other, creating a differential path change of $\sim 10^{-18}$ m — about one-thousandth the diameter of a proton. This is detected as a shift in the interference pattern.

---

## Exercises

### Exercise 1: Double-Slit Parameters

In a Young's double-slit experiment using 632.8 nm light (HeNe laser), the slits are separated by 0.15 mm and the screen is 2.0 m away.

(a) Calculate the fringe spacing on the screen.

(b) What is the angular position of the 5th bright fringe from the center?

(c) If one slit is covered with a neutral density filter that reduces its intensity to 25% of the other slit, what is the fringe visibility?

### Exercise 2: Anti-Reflection Coating Design

Design a single-layer anti-reflection coating for a glass lens ($n_g = 1.72$, high-index glass) optimized for $\lambda = 550$ nm.

(a) What is the ideal coating refractive index?

(b) What thickness should the coating be?

(c) Calculate the residual reflectance at 550 nm if MgF$_2$ ($n = 1.38$) is used instead of the ideal material.

(d) At what wavelength does the MgF$_2$ coating achieve zero reflectance for this glass?

### Exercise 3: Michelson Interferometer

A Michelson interferometer illuminated by a sodium lamp ($\lambda = 589.0$ nm and $\lambda = 589.6$ nm — the sodium doublet) shows fringes that periodically fade in and out as one mirror is translated.

(a) Explain why the fringes fade. (Hint: the two wavelengths produce overlapping fringe patterns.)

(b) At what path difference do the fringes first disappear?

(c) Calculate the coherence length of the sodium doublet.

### Exercise 4: Fabry-Perot Etalon

A Fabry-Perot etalon has mirror reflectivity $R = 0.95$, spacing $d = 5$ mm, and is used with green light ($\lambda = 546$ nm).

(a) Calculate the finesse.

(b) Calculate the free spectral range (in frequency and wavelength).

(c) What is the minimum wavelength difference that can be resolved?

(d) What is the resolving power?

### Exercise 5: Coherence and Fringe Visibility

A Michelson interferometer is illuminated by a source with a Gaussian spectral profile centered at $\lambda_0 = 800$ nm with a full-width at half-maximum (FWHM) spectral width of $\Delta\lambda = 40$ nm.

(a) Calculate the coherence length.

(b) At what path difference does the fringe visibility drop to $1/e$ of its maximum?

(c) How many fringes are visible before the contrast drops below 50%?

---

## Summary

| Concept | Key Formula / Fact |
|---------|-------------------|
| Superposition | $\mathbf{E}_{\text{total}} = \mathbf{E}_1 + \mathbf{E}_2$; intensity includes cross term |
| Two-beam interference | $I = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos\delta$ |
| Fringe visibility | $V = (I_{\max} - I_{\min})/(I_{\max} + I_{\min})$ |
| Double-slit bright fringes | $d\sin\theta = m\lambda$; fringe spacing $\Delta y = \lambda L/d$ |
| Thin-film reflection phase | $\pi$ shift at denser medium; no shift at less dense |
| Anti-reflection coating | Quarter-wave thickness: $t = \lambda/(4n_c)$; ideal $n_c = \sqrt{n_g}$ |
| Newton's rings (dark) | $r_m = \sqrt{m\lambda R}$ |
| Michelson interferometer | OPD = $2(d_1 - d_2)$; fringes shift by one per $\lambda/2$ mirror motion |
| Fabry-Perot (Airy function) | $I_t = I_0/[1 + F\sin^2(\delta/2)]$; $\mathcal{F} = \pi\sqrt{R}/(1-R)$ |
| Free spectral range | $\Delta\nu_{\text{FSR}} = c/(2nd)$ |
| Coherence length | $\ell_c = c/\Delta\nu = \lambda^2/\Delta\lambda$ |
| Spatial coherence width | $d_c = \lambda/\Delta\theta$ |

---

[← Previous: 04. Optical Instruments](04_Optical_Instruments.md) | [Next: 06. Diffraction →](06_Diffraction.md)

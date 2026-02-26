# 04. Optical Instruments

[← Previous: 03. Mirrors and Lenses](03_Mirrors_and_Lenses.md) | [Next: 05. Wave Optics — Interference →](05_Wave_Optics_Interference.md)

---

## Learning Objectives

1. Analyze the optics of magnifying glasses, microscopes, and telescopes using the thin lens equation
2. Calculate angular magnification for visual instruments and distinguish it from lateral magnification
3. Apply the Rayleigh criterion to determine the resolution limit of optical systems
4. Explain the optical design of cameras including f-number, depth of field, and exposure
5. Describe the optics of the human eye, common vision defects, and corrective lenses
6. Compare refracting and reflecting telescope designs and understand their trade-offs
7. Evaluate practical limits — diffraction, aberrations, detector noise — that constrain instrument performance

---

## Why This Matters

Optical instruments extend human vision from the subatomic to the cosmic. Microscopes revealed bacteria, cells, and viruses, revolutionizing medicine. Telescopes revealed galaxies, the cosmic microwave background, and exoplanets, revolutionizing our understanding of the universe. The camera — now embedded in every smartphone — is arguably the most impactful optical instrument in daily life. Understanding how these instruments work, and what limits their performance, is essential for anyone working with imaging, sensing, or photonics.

> **Analogy**: An optical instrument is like a translator for light. Your eye can only process a narrow range of angular sizes (about 1 arcminute resolution) and intensities. A microscope "translates" the angular size of a bacterium from 0.001° (invisible) to 10° (easily visible). A telescope "translates" the faint light of a distant galaxy from an undetectable whisper to a measurable signal. Each instrument reshapes the light to fit the narrow bandwidth of human perception.

---

## 1. The Human Eye

### 1.1 Optical Design

The human eye is a remarkable optical instrument:

| Component | Function |
|-----------|----------|
| Cornea | Primary refracting surface ($n \approx 1.376$, power $\approx$ 43 D) |
| Aqueous humor | Transparent fluid ($n \approx 1.336$) |
| Iris / Pupil | Aperture stop (2–8 mm diameter) |
| Crystalline lens | Variable-focus lens ($n \approx 1.39$–$1.41$, power $\approx$ 15–30 D) |
| Vitreous humor | Transparent gel ($n \approx 1.337$) |
| Retina | Detector (rods for dim light, cones for color) |

Total optical power: approximately 60 D, giving a focal length of about 17 mm.

The eye accommodates (changes focus) by changing the shape of the crystalline lens:
- **Far point** (relaxed eye): Focused at infinity (healthy eye). The farthest point you can see clearly.
- **Near point** (maximum accommodation): About 25 cm for a young adult. The closest point you can focus on.

### 1.2 Visual Acuity and Angular Resolution

The minimum angle of resolution of the human eye is approximately:

$$\theta_{\min} \approx 1' = \frac{1}{60}° \approx 2.9 \times 10^{-4} \text{ rad}$$

This corresponds to a spacing of about 5 $\mu$m on the retina — roughly the diameter of a foveal cone cell. For a 3 mm pupil, the diffraction limit at 550 nm is:

$$\theta_{\text{diff}} = 1.22 \frac{\lambda}{D} = 1.22 \times \frac{550 \times 10^{-9}}{3 \times 10^{-3}} \approx 2.2 \times 10^{-4} \text{ rad} \approx 0.77'$$

The eye's resolution is impressively close to its diffraction limit — evolution has optimized the retinal mosaic to match.

### 1.3 Common Vision Defects

| Defect | Cause | Image Position | Correction |
|--------|-------|---------------|------------|
| **Myopia** (nearsightedness) | Eyeball too long or cornea too curved | In front of retina | Diverging lens (negative power) |
| **Hyperopia** (farsightedness) | Eyeball too short or cornea too flat | Behind retina | Converging lens (positive power) |
| **Presbyopia** | Lens loses flexibility with age | Near point recedes | Reading glasses (converging lens) |
| **Astigmatism** | Cornea not spherical (different curvature in different meridians) | Line foci at different distances | Cylindrical or toric lens |

```python
import numpy as np
import matplotlib.pyplot as plt

# Corrective lens prescriptions: relationship between
# vision defect and lens power needed

# For myopia: far point is closer than infinity
# Lens power needed: P = -1/far_point (in meters)
far_points_m = np.linspace(0.2, 5.0, 100)  # far point from 20 cm to 5 m
P_myopia = -1 / far_points_m

# For hyperopia/presbyopia: near point is farther than 25 cm
# Lens power needed to bring near point back to 25 cm:
# P = 1/0.25 - 1/near_point (near point > 0.25 m)
near_points_m = np.linspace(0.3, 3.0, 100)
P_hyperopia = 1/0.25 - 1/near_points_m

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Myopia correction
ax1.plot(far_points_m * 100, P_myopia, 'b-', linewidth=2)
ax1.set_xlabel('Uncorrected Far Point (cm)', fontsize=12)
ax1.set_ylabel('Lens Power (Diopters)', fontsize=12)
ax1.set_title('Myopia Correction', fontsize=13)
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.grid(True, alpha=0.3)
# Typical prescription range
ax1.axhspan(-6, -1, alpha=0.1, color='blue', label='Typical range (-1 to -6 D)')
ax1.legend(fontsize=10)

# Right: Hyperopia/Presbyopia correction
ax2.plot(near_points_m * 100, P_hyperopia, 'r-', linewidth=2)
ax2.set_xlabel('Uncorrected Near Point (cm)', fontsize=12)
ax2.set_ylabel('Lens Power (Diopters)', fontsize=12)
ax2.set_title('Hyperopia / Presbyopia Correction\n(bring near point to 25 cm)', fontsize=13)
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.grid(True, alpha=0.3)
ax2.axhspan(0.5, 3.0, alpha=0.1, color='red', label='Typical range (+0.5 to +3 D)')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('vision_correction.png', dpi=150)
plt.show()
```

---

## 2. The Magnifying Glass (Simple Magnifier)

### 2.1 Angular Magnification

A converging lens used as a magnifying glass does not bring the object "closer" — it increases the **angular size** of the object as seen by the eye. The angular magnification is defined as:

$$M = \frac{\theta_{\text{with lens}}}{\theta_{\text{without lens}}}$$

Without the lens, the maximum angular size is achieved at the near point ($D = 25$ cm):

$$\theta_0 = \frac{h}{D}$$

With the lens (focal length $f$), the object is placed at or inside the focal point:

**Image at infinity** (relaxed eye, object at $f$):

$$M_\infty = \frac{D}{f} = \frac{25 \text{ cm}}{f}$$

**Image at near point** (maximum magnification, object slightly inside $f$):

$$M_{\max} = \frac{D}{f} + 1 = \frac{25}{f} + 1$$

A typical magnifying glass with $f = 5$ cm gives $M_\infty = 5\times$.

### 2.2 Practical Limits

The magnifying glass is limited to about $M \leq 10\times$ because shorter focal lengths require bringing the eye very close to the lens, introducing severe aberrations. For higher magnification, we need a compound microscope.

---

## 3. The Compound Microscope

### 3.1 Optical Design

A compound microscope uses two lens groups:

1. **Objective lens** (short focal length $f_o$): Creates a magnified real image of the specimen inside the tube
2. **Eyepiece** (ocular, focal length $f_e$): Acts as a magnifying glass to view the intermediate image

The intermediate image is formed at a distance $L$ (the **tube length**, typically 160 mm or infinity in modern microscopes) from the objective's rear focal point.

### 3.2 Total Magnification

The total angular magnification is the product of the objective's lateral magnification and the eyepiece's angular magnification:

$$M_{\text{total}} = m_o \times M_e = -\frac{L}{f_o} \times \frac{D}{f_e}$$

For a $40\times$ objective ($f_o = 4$ mm) and $10\times$ eyepiece ($f_e = 25$ mm) with $L = 160$ mm:

$$M_{\text{total}} = -\frac{160}{4} \times \frac{25}{25} = -40 \times 10 = -400\times$$

The negative sign indicates the final image is inverted (which is corrected in practice by additional prisms or relay lenses).

### 3.3 Resolution: The Real Limit

The useful magnification of a microscope is limited not by the lenses but by **diffraction**. The smallest resolvable feature (Abbe diffraction limit) is:

$$d_{\min} = \frac{0.61\lambda}{\text{NA}} = \frac{0.61\lambda}{n\sin\alpha}$$

where **NA** (Numerical Aperture) = $n \sin\alpha$, with $n$ being the refractive index of the medium between the specimen and objective, and $\alpha$ the half-angle of the maximum cone of light collected.

| Objective | NA | Resolution (550 nm) |
|-----------|-----|-------------------|
| 4$\times$ (dry) | 0.10 | 3.4 $\mu$m |
| 10$\times$ (dry) | 0.25 | 1.3 $\mu$m |
| 40$\times$ (dry) | 0.65 | 0.52 $\mu$m |
| 100$\times$ (oil, $n=1.52$) | 1.25 | 0.27 $\mu$m |

**Empty magnification**: Magnification beyond about $500 \times \text{NA}$ to $1000 \times \text{NA}$ reveals no new detail — you just magnify the blur. This is empty (or useless) magnification.

> **Analogy**: Resolution is like the pixel count of a digital image. Magnification is like zooming in on the screen. You can zoom in as much as you want, but once you exceed the native resolution, all you see are bigger pixels — no new detail. The microscope objective's NA determines the "pixel count" of the optical image.

```python
import numpy as np
import matplotlib.pyplot as plt

# Microscope resolution: Airy disk size as a function of numerical aperture
# Shows why oil immersion objectives are essential for high resolution

wavelength = 550e-9  # green light (m)

NA_values = np.linspace(0.05, 1.4, 200)

# Abbe resolution limit: d_min = 0.61 * lambda / NA
d_min = 0.61 * wavelength / NA_values * 1e6  # convert to micrometers

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: resolution vs NA
ax1.plot(NA_values, d_min, 'b-', linewidth=2)
ax1.set_xlabel('Numerical Aperture (NA)', fontsize=12)
ax1.set_ylabel('Minimum Resolvable Feature ($\\mu$m)', fontsize=12)
ax1.set_title('Microscope Resolution Limit (Abbe Criterion)', fontsize=13)
ax1.set_ylim(0, 8)
ax1.grid(True, alpha=0.3)

# Mark common objectives
objectives = [
    (0.10, '4x dry'),
    (0.25, '10x dry'),
    (0.65, '40x dry'),
    (0.95, '100x dry'),
    (1.25, '100x oil'),
    (1.40, '100x oil max'),
]
for na, label in objectives:
    d = 0.61 * wavelength / na * 1e6
    ax1.plot(na, d, 'ro', markersize=6)
    ax1.annotate(f'  {label}\n  d={d:.2f}μm', xy=(na, d), fontsize=8)

# Indicate the air limit (NA max = 1.0 for dry objectives)
ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='Air limit (NA=1)')
ax1.legend(fontsize=10)

# Right: useful magnification range
ax2.fill_between([0, 1.5], [0, 0], [500*0, 500*1.5], alpha=0.15, color='red',
                 label='Below useful (< 500·NA)')
ax2.fill_between([0, 1.5], [500*0, 500*1.5], [1000*0, 1000*1.5], alpha=0.15, color='green',
                 label='Useful range (500-1000·NA)')
ax2.fill_between([0, 1.5], [1000*0, 1000*1.5], [2000, 2000], alpha=0.15, color='orange',
                 label='Empty magnification (> 1000·NA)')

NA_range = np.linspace(0, 1.5, 100)
ax2.plot(NA_range, 500 * NA_range, 'r--', linewidth=1.5)
ax2.plot(NA_range, 1000 * NA_range, 'g--', linewidth=1.5)

ax2.set_xlabel('Numerical Aperture (NA)', fontsize=12)
ax2.set_ylabel('Total Magnification', fontsize=12)
ax2.set_title('Useful vs. Empty Magnification', fontsize=13)
ax2.set_xlim(0, 1.5)
ax2.set_ylim(0, 2000)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('microscope_resolution.png', dpi=150)
plt.show()
```

---

## 4. Telescopes

### 4.1 Refracting Telescope (Keplerian)

A refracting telescope uses an objective lens (large $f_o$) to form a real image at its focal plane, and an eyepiece (small $f_e$) to magnify that image:

$$M = -\frac{f_o}{f_e}$$

The total length of the telescope is approximately $f_o + f_e$.

**Example**: A telescope with $f_o = 1000$ mm and $f_e = 25$ mm gives $M = -40\times$.

### 4.2 Reflecting Telescope (Newtonian)

Isaac Newton invented the reflecting telescope to avoid chromatic aberration (which plagued early refractors). A concave primary mirror collects and focuses light, and a small flat secondary mirror redirects the beam to a side-mounted eyepiece.

**Advantages over refractors**:
- No chromatic aberration (mirrors reflect all wavelengths the same way)
- Can be made much larger (a lens must be supported at its edges; a mirror can be supported from behind)
- Less glass needed (only one reflecting surface vs. four refracting surfaces)

### 4.3 Cassegrain Telescope

The Cassegrain design uses a concave primary mirror and a convex secondary mirror. Light enters the tube, reflects off the primary, bounces off the secondary back through a hole in the primary, and reaches a focus behind the primary mirror.

**Advantages**: Very compact tube length for a given focal length. The effective focal length is $f_{\text{eff}} = f_{\text{primary}} \times |m_{\text{secondary}}|$, where $m_{\text{secondary}}$ is the magnification of the secondary.

| Telescope | Primary | Secondary | Advantage |
|-----------|---------|-----------|-----------|
| Newtonian | Concave (parabolic) | Flat (diagonal) | Simple, affordable |
| Cassegrain | Concave (parabolic) | Convex (hyperbolic) | Compact, long $f_{\text{eff}}$ |
| Ritchey-Chretien | Concave (hyperbolic) | Convex (hyperbolic) | Wide field, no coma (used by HST, JWST) |
| Schmidt-Cassegrain | Spherical + corrector plate | Convex | Very compact, mass-produced |

### 4.4 Angular Resolution: Rayleigh Criterion

The diffraction-limited angular resolution of any telescope (or camera) with circular aperture $D$ is:

$$\theta_R = 1.22 \frac{\lambda}{D}$$

This is the **Rayleigh criterion** — two point sources are just resolved when the central maximum of one coincides with the first minimum of the other's Airy pattern.

| Instrument | Aperture | Resolution (550 nm) |
|-----------|----------|-------------------|
| Human eye | 5 mm | 27.5" (arcseconds) |
| Binoculars (50 mm) | 50 mm | 2.8" |
| Amateur telescope (200 mm) | 200 mm | 0.69" |
| Hubble Space Telescope | 2.4 m | 0.056" |
| James Webb Space Telescope | 6.5 m | 0.07" (at 2 $\mu$m IR) |

> **Note**: Ground-based telescopes are usually limited by atmospheric turbulence (seeing, typically 1"–2") rather than diffraction. **Adaptive optics** corrects for atmospheric distortion in real time using deformable mirrors, approaching the diffraction limit.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function of the first kind, order 1

# Airy pattern: the diffraction-limited point spread function
# of a circular aperture (telescope, camera, etc.)

def airy_intensity(theta, D, wavelength):
    """
    Calculate the normalized Airy pattern intensity.
    theta: angle from axis (radians)
    D: aperture diameter (meters)
    wavelength: wavelength (meters)
    Returns: I/I_max
    """
    # x = pi * D * sin(theta) / wavelength ≈ pi * D * theta / lambda for small theta
    x = np.pi * D * theta / wavelength
    # I(x) = [2 * J1(x) / x]^2, with I(0) = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.where(np.abs(x) < 1e-10, 1.0, (2 * j1(x) / x)**2)
    return pattern

# Parameters
wavelength = 550e-9  # green light
D_telescope = 0.2    # 200 mm amateur telescope (8-inch)

# Angular range (in arcseconds)
theta_arcsec = np.linspace(-3, 3, 1000)
theta_rad = theta_arcsec * np.pi / (180 * 3600)  # convert arcsec to radians

# Single star Airy pattern
I_single = airy_intensity(theta_rad, D_telescope, wavelength)

# Two stars separated by exactly the Rayleigh criterion
theta_R = 1.22 * wavelength / D_telescope  # Rayleigh limit in radians
sep_arcsec = theta_R * 180 * 3600 / np.pi
print(f"Rayleigh limit: {sep_arcsec:.2f} arcseconds for D={D_telescope*1000:.0f}mm at λ={wavelength*1e9:.0f}nm")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

separations = [0.5 * sep_arcsec, sep_arcsec, 2.0 * sep_arcsec]
labels = ['Unresolved\n(sep < Rayleigh)', 'Just Resolved\n(sep = Rayleigh)', 'Well Resolved\n(sep > Rayleigh)']

for ax, sep, label in zip(axes, separations, labels):
    # Two star Airy patterns, separated by 'sep' arcseconds
    sep_rad = sep * np.pi / (180 * 3600)
    I_star1 = airy_intensity(theta_rad - sep_rad/2, D_telescope, wavelength)
    I_star2 = airy_intensity(theta_rad + sep_rad/2, D_telescope, wavelength)
    I_combined = I_star1 + I_star2

    ax.plot(theta_arcsec, I_star1, 'b--', alpha=0.5, linewidth=1, label='Star 1')
    ax.plot(theta_arcsec, I_star2, 'r--', alpha=0.5, linewidth=1, label='Star 2')
    ax.plot(theta_arcsec, I_combined, 'k-', linewidth=2, label='Combined')
    ax.fill_between(theta_arcsec, 0, I_combined, alpha=0.1, color='gray')

    ax.set_xlabel('Angle (arcseconds)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title(f'{label}\nsep = {sep:.2f}"', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.2)

plt.suptitle(f'Rayleigh Criterion (D={D_telescope*1000:.0f}mm, λ={wavelength*1e9:.0f}nm)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rayleigh_criterion.png', dpi=150)
plt.show()
```

---

## 5. The Camera

### 5.1 Basic Camera Optics

A camera focuses light from a scene onto a detector (film or digital sensor) using a lens system. The key parameters are:

**Focal length** $f$: Determines the field of view and image magnification.
- Wide-angle: $f < 35$ mm (35mm equivalent) — large field of view
- Normal: $f \approx 50$ mm — similar to human eye perspective
- Telephoto: $f > 70$ mm — narrow field of view, high magnification

**F-number** (f-stop):

$$N = \frac{f}{D}$$

where $D$ is the aperture diameter. Common f-numbers: $f$/1.4, $f$/2, $f$/2.8, $f$/4, $f$/5.6, $f$/8, $f$/11, $f$/16.

Each stop doubles (or halves) the light-gathering area because:

$$\text{Area} \propto D^2 \propto \frac{f^2}{N^2}$$

Going from $f$/2.8 to $f$/4 halves the light (one stop darker).

### 5.2 Depth of Field

Not all objects at different distances can be perfectly focused simultaneously. The **depth of field** (DOF) is the range of distances that appear acceptably sharp:

$$\text{DOF} \approx \frac{2 N c s^2}{f^2}$$

where $N$ is the f-number, $c$ is the circle of confusion diameter (typically $\sim 30$ $\mu$m for full-frame), and $s$ is the focus distance.

**Key relationships**:
- Larger $N$ (smaller aperture) → larger DOF → more in focus
- Shorter $f$ (wide-angle) → larger DOF
- Greater $s$ (focus distance) → larger DOF

This is why portrait photographers use large apertures ($f$/1.4–$f$/2.8) to create a blurred background (bokeh), while landscape photographers use small apertures ($f$/8–$f$/16) to keep everything sharp.

### 5.3 Exposure and the Exposure Triangle

Correct exposure depends on three settings:

1. **Aperture** ($N$): Controls light amount and DOF
2. **Shutter speed** ($t$): Controls motion blur and total light
3. **ISO sensitivity**: Controls sensor gain and noise

The exposure value (EV) is:

$$\text{EV} = \log_2\left(\frac{N^2}{t}\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Depth of field calculation for different camera settings
# Shows the relationship between f-number, focal length, and DOF

def depth_of_field(f_mm, N, s_m, c_mm=0.030):
    """
    Calculate depth of field.
    f_mm: focal length in mm
    N: f-number
    s_m: subject distance in meters
    c_mm: circle of confusion diameter in mm (0.030 for full-frame)
    Returns: (near_limit, far_limit, DOF) in meters
    """
    f_m = f_mm / 1000
    c_m = c_mm / 1000

    # Hyperfocal distance: H = f^2 / (N * c) + f
    H = f_m**2 / (N * c_m) + f_m

    # Near and far limits of acceptable sharpness
    near = s_m * (H - f_m) / (H + s_m - 2*f_m)
    if H > s_m:
        far = s_m * (H - f_m) / (H - s_m)
    else:
        far = float('inf')  # everything to infinity is in focus

    DOF = far - near if np.isfinite(far) else float('inf')
    return near, far, DOF

# Compare DOF for different scenarios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: DOF vs f-number for 50mm lens at 3m focus distance
f_numbers = np.array([1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22])
near_limits = []
far_limits = []
for N in f_numbers:
    near, far, dof = depth_of_field(50, N, 3.0)
    near_limits.append(near)
    far_limits.append(min(far, 20))  # cap for plotting

ax1.fill_between(f_numbers, near_limits, far_limits, alpha=0.3, color='green', label='In-focus range')
ax1.plot(f_numbers, near_limits, 'b-o', linewidth=1.5, markersize=4, label='Near limit')
ax1.plot(f_numbers, far_limits, 'r-o', linewidth=1.5, markersize=4, label='Far limit')
ax1.axhline(3.0, color='gray', linestyle='--', alpha=0.5, label='Focus distance (3m)')
ax1.set_xlabel('F-number', fontsize=12)
ax1.set_ylabel('Distance (m)', fontsize=12)
ax1.set_title('Depth of Field vs F-number\n(50mm lens, subject at 3m)', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 15)

# Right: DOF vs focal length at f/4, subject at 3m
focal_lengths = np.array([24, 35, 50, 85, 100, 135, 200])
near_limits2 = []
far_limits2 = []
for fl in focal_lengths:
    near, far, dof = depth_of_field(fl, 4.0, 3.0)
    near_limits2.append(near)
    far_limits2.append(min(far, 30))

ax2.fill_between(focal_lengths, near_limits2, far_limits2, alpha=0.3, color='green')
ax2.plot(focal_lengths, near_limits2, 'b-o', linewidth=1.5, markersize=4, label='Near limit')
ax2.plot(focal_lengths, far_limits2, 'r-o', linewidth=1.5, markersize=4, label='Far limit')
ax2.axhline(3.0, color='gray', linestyle='--', alpha=0.5, label='Focus distance (3m)')
ax2.set_xlabel('Focal Length (mm)', fontsize=12)
ax2.set_ylabel('Distance (m)', fontsize=12)
ax2.set_title('Depth of Field vs Focal Length\n(f/4, subject at 3m)', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 15)

plt.tight_layout()
plt.savefig('depth_of_field.png', dpi=150)
plt.show()
```

---

## 6. Resolution Limits of Optical Systems

### 6.1 The Rayleigh Criterion (Revisited)

For any diffraction-limited imaging system with a circular aperture of diameter $D$:

$$\theta_R = 1.22\frac{\lambda}{D}$$

The spatial resolution on the image plane is:

$$\Delta x = 1.22\frac{\lambda f}{D} = 1.22 \lambda N$$

where $N = f/D$ is the f-number. This means:
- A camera at $f$/2 resolves features as small as $1.22 \times 0.55 \times 2 \approx 1.3$ $\mu$m
- A camera at $f$/11 resolves features as small as $\approx 7.4$ $\mu$m

Modern smartphone cameras with $\sim 1$ $\mu$m pixel pitch actually begin to be limited by diffraction at small apertures.

### 6.2 The Sparrow Criterion

The Rayleigh criterion is somewhat arbitrary. The **Sparrow criterion** defines the resolution limit as the separation at which the combined intensity profile no longer has a dip between the two peaks:

$$\theta_S = 0.95 \frac{\lambda}{D} \approx 0.78 \, \theta_R$$

The Sparrow limit is about 22% tighter than the Rayleigh limit.

### 6.3 Dawes Limit (Empirical for Telescopes)

For visual observation of double stars through a telescope with aperture $D$ (in mm):

$$\theta_{\text{Dawes}} \approx \frac{116}{D} \text{ (arcseconds, } D \text{ in mm)}$$

This empirical limit accounts for both diffraction and the contrast sensitivity of the human eye.

---

## 7. Comparison of Optical Instruments

| Instrument | Key Metric | Typical Value | Limiting Factor |
|-----------|-----------|---------------|----------------|
| Magnifying glass | Angular magnification | 2–10$\times$ | Aberrations at high $M$ |
| Compound microscope | Resolution (NA) | 0.2–1.4 NA | Diffraction ($d \geq 0.61\lambda$/NA) |
| Refracting telescope | Angular magnification | $f_o/f_e$ | Chromatic aberration, size |
| Reflecting telescope | Light gathering ($\propto D^2$) | $D$ up to 10 m | Mirror fabrication, seeing |
| Camera | Spatial resolution | $1.22\lambda N$ | Diffraction, pixel size, noise |
| Human eye | Angular resolution | $\sim 1'$ | Cone spacing, aberrations |

---

## 8. Advanced Topics: Adaptive Optics

Ground-based telescopes have their resolution degraded by atmospheric turbulence — air cells of different temperatures act as a random collection of weak lenses, blurring the image. The **Fried parameter** $r_0$ characterizes the turbulence:

$$\theta_{\text{seeing}} \sim \frac{\lambda}{r_0}$$

Typically $r_0 \sim 10$–$20$ cm at visible wavelengths, so the seeing limit is about 0.5"–1.5" — far worse than the diffraction limit of a large telescope.

**Adaptive optics (AO)** corrects this in real time:
1. A **wavefront sensor** measures the distortion (using a natural or laser guide star)
2. A **deformable mirror** (with hundreds to thousands of actuators) adjusts its shape to cancel the distortion
3. The corrected image approaches the diffraction limit

Modern AO systems on 8-10 m telescopes routinely achieve 0.05"–0.1" resolution in the near-infrared.

---

## Exercises

### Exercise 1: Eye Correction

A person has a far point of 50 cm (myopia) and a near point of 15 cm.

(a) What lens power (in diopters) is needed to correct the myopia (bring the far point to infinity)?

(b) With the corrective lens in place, what is the new near point?

(c) At age 55, this person's accommodation decreases so the near point becomes 100 cm. What additional lens power (bifocal add) is needed for reading at 25 cm?

### Exercise 2: Microscope Design

A microscope has a $40\times$ objective (NA = 0.65, $f_o = 4.5$ mm) and a $10\times$ eyepiece ($f_e = 25$ mm).

(a) What is the total magnification?

(b) What is the theoretical resolution at $\lambda = 550$ nm?

(c) Is the total magnification within the useful range, or is some of it "empty magnification"?

(d) How would the resolution change if oil immersion ($n = 1.52$) were used?

### Exercise 3: Telescope Comparison

Compare two telescopes for observing Jupiter:
- Telescope A: $D = 100$ mm refractor, $f = 900$ mm, eyepiece $f_e = 9$ mm
- Telescope B: $D = 200$ mm Newtonian reflector, $f = 1200$ mm, eyepiece $f_e = 12$ mm

(a) Calculate the magnification of each.

(b) Calculate the diffraction-limited angular resolution of each (at 550 nm).

(c) Which telescope shows more detail, and why? (Assume seeing is 2".)

(d) Calculate the exit pupil of each. Which is better for night observing?

### Exercise 4: Camera Settings

You are photographing a landscape at a focus distance of 5 m with a 35 mm lens.

(a) What f-number gives a depth of field from 2 m to infinity? (Hint: the far limit reaches infinity at the hyperfocal distance.)

(b) What is the diffraction-limited resolution at this f-number?

(c) If your sensor has 5 $\mu$m pixels, is the system diffraction-limited or pixel-limited?

### Exercise 5: Satellite Imaging

A spy satellite orbits at 200 km altitude and carries a telescope with $D = 2.4$ m.

(a) What is the theoretical angular resolution at $\lambda = 550$ nm?

(b) What ground distance does this correspond to at 200 km altitude?

(c) Can this satellite read a car license plate (character height $\sim 8$ cm)? Why or why not?

---

## Summary

| Concept | Key Formula / Fact |
|---------|-------------------|
| Eye optics | Total power $\approx$ 60 D; near point 25 cm; resolution $\sim 1'$ |
| Magnifying glass | $M = D/f$ (relaxed); $M = D/f + 1$ (max); $D = 25$ cm |
| Compound microscope | $M = -(L/f_o)(D/f_e)$; resolution $d = 0.61\lambda$/NA |
| Numerical aperture | NA = $n\sin\alpha$; oil immersion increases NA above 1.0 |
| Useful magnification | $500\cdot$NA to $1000\cdot$NA; beyond this is empty magnification |
| Telescope magnification | $M = -f_o/f_e$ (Keplerian) |
| Rayleigh criterion | $\theta_R = 1.22\lambda/D$ — angular resolution of circular aperture |
| Camera f-number | $N = f/D$; each stop doubles light; smaller $N$ = brighter |
| Depth of field | DOF $\approx 2Ncs^2/f^2$; larger $N$ or shorter $f$ → more DOF |
| Exposure value | EV = $\log_2(N^2/t)$ |
| Adaptive optics | Deformable mirror corrects atmospheric turbulence → near diffraction limit |

---

[← Previous: 03. Mirrors and Lenses](03_Mirrors_and_Lenses.md) | [Next: 05. Wave Optics — Interference →](05_Wave_Optics_Interference.md)

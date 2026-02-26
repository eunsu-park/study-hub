# 03. Mirrors and Lenses

[← Previous: 02. Geometric Optics Fundamentals](02_Geometric_Optics_Fundamentals.md) | [Next: 04. Optical Instruments →](04_Optical_Instruments.md)

---

## Learning Objectives

1. Apply the mirror equation and thin lens equation to locate images and determine magnification
2. Derive the lensmaker's equation and understand its dependence on refractive index and curvature
3. Distinguish between real and virtual images and predict which type an optical element produces
4. Trace principal rays through mirrors and thin lenses to construct ray diagrams
5. Identify the five primary aberrations (Seidel aberrations) and their physical origins
6. Use the matrix (ABCD) method for systematic analysis of multi-element optical systems
7. Solve compound optical systems with multiple lenses in series

---

## Why This Matters

Mirrors and lenses are the building blocks of every optical instrument — from the reading glasses correcting your vision to the 6.5-meter primary mirror of the James Webb Space Telescope. Understanding how curved surfaces form images is not just an academic exercise; it is the practical foundation for designing cameras, microscopes, projectors, laser cavities, and fiber-optic couplers. The thin lens equation, despite its simplicity, captures the essential physics of image formation.

> **Analogy**: A lens is like a highway toll plaza for light waves. All the waves from a single source point arrive at the lens with different phases (because they traveled different distances). The lens adds just the right amount of extra phase to each wavelet — more in the thinner parts, less in the thicker parts — so that they all arrive in phase at the image point. The lens is a "phase corrector" that redirects a diverging wavefront into a converging one.

---

## 1. Curved Mirrors

### 1.1 Concave (Converging) Mirrors

A concave mirror has a reflecting surface that curves inward (like the inside of a spoon). Parallel rays converge to a **focal point** $F$ located at half the radius of curvature:

$$f = \frac{R}{2}$$

where $R$ is the radius of curvature and $f$ is the focal length.

The **mirror equation** relates object distance $s$, image distance $s'$, and focal length $f$:

$$\frac{1}{s} + \frac{1}{s'} = \frac{1}{f} = \frac{2}{R}$$

**Lateral magnification**:

$$m = -\frac{s'}{s}$$

The negative sign means that when both $s$ and $s'$ are positive (real object, real image), the image is inverted.

### 1.2 Convex (Diverging) Mirrors

A convex mirror has a reflecting surface that curves outward. Parallel rays diverge after reflection, appearing to come from a virtual focal point behind the mirror. Using our sign convention:

$$f = -\frac{|R|}{2} \quad (\text{negative for convex mirrors})$$

Convex mirrors always produce virtual, upright, reduced images — which is why they are used as side mirrors on vehicles ("objects in mirror are closer than they appear").

### 1.3 Principal Ray Tracing for Mirrors

Three principal rays suffice to locate an image:

1. **Parallel ray**: Arrives parallel to the optical axis → reflects through the focal point $F$
2. **Focal ray**: Passes through $F$ → reflects parallel to the optical axis
3. **Center ray**: Strikes the center of the mirror → reflects symmetrically (angle in = angle out)

The image forms where any two of these rays intersect.

```python
import numpy as np
import matplotlib.pyplot as plt

# Ray tracing for a concave mirror
# Demonstrates image formation for objects at different distances

def mirror_image(s, f):
    """
    Calculate image distance and magnification for a curved mirror.
    s: object distance (positive for real object)
    f: focal length (positive for concave, negative for convex)
    Returns: (s', m) — image distance and magnification
    """
    if abs(1/f - 1/s) < 1e-10:
        return float('inf'), float('inf')  # object at focal point
    s_prime = 1 / (1/f - 1/s)
    m = -s_prime / s
    return s_prime, m

# Mirror parameters
f = 10.0  # focal length (cm), concave mirror
R = 2 * f

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Four cases: object at different positions relative to the mirror
cases = [
    ('Object beyond C (s > 2f)', 30.0),
    ('Object at C (s = 2f)', 20.0),
    ('Object between F and C (f < s < 2f)', 15.0),
    ('Object inside F (s < f)', 5.0),
]

for ax, (title, s) in zip(axes.flat, cases):
    s_prime, m = mirror_image(s, f)

    # Draw the mirror (vertical line at x=0 with curved cap)
    mirror_y = np.linspace(-8, 8, 100)
    mirror_x = mirror_y**2 / (4 * f) * 0.3  # slight curvature for visualization
    ax.plot(-mirror_x, mirror_y, 'k-', linewidth=3)

    # Draw optical axis
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    # Mark focal point and center of curvature
    ax.plot(-f, 0, 'ro', markersize=8, label=f'F (f={f})')
    ax.plot(-R, 0, 'bs', markersize=6, label=f'C (R={R})')

    # Object (arrow at -s)
    obj_height = 3.0
    ax.annotate('', xy=(-s, obj_height), xytext=(-s, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(-s, obj_height + 0.5, 'Object', ha='center', fontsize=9, color='green')

    # Principal rays (from tip of object)
    # Ray 1: parallel to axis → reflects through F
    ax.plot([-s, 0], [obj_height, obj_height], 'b-', linewidth=1.2)
    if np.isfinite(s_prime) and s_prime > 0:
        # Real image: ray goes through F to the image
        ax.plot([0, -s_prime], [obj_height, m * obj_height], 'b-', linewidth=1.2)
    else:
        # Virtual image or at infinity
        ax.plot([0, -f], [obj_height, 0], 'b-', linewidth=1.2)
        if np.isfinite(s_prime):
            ax.plot([0, 15], [obj_height, obj_height + 15 * obj_height / f], 'b--', linewidth=1)

    # Ray 2: through F → reflects parallel
    if abs(s - f) > 0.1:
        ax.plot([-s, 0], [obj_height, obj_height * (1 - s/f) + obj_height * s/f * 0], 'r-', linewidth=1.2)

    # Image (if real)
    if np.isfinite(s_prime) and abs(s_prime) < 100:
        img_height = m * obj_height
        if s_prime > 0:  # real image
            ax.annotate('', xy=(-s_prime, img_height), xytext=(-s_prime, 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
            img_type = 'Real'
        else:  # virtual image
            ax.annotate('', xy=(abs(s_prime), img_height), xytext=(abs(s_prime), 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.5, linestyle='--'))
            img_type = 'Virtual'
        ax.set_title(f"{title}\n$s'$={s_prime:.1f}, m={m:.2f} ({img_type})", fontsize=11)
    else:
        ax.set_title(f"{title}\nImage at infinity", fontsize=11)

    ax.set_xlim(-35, 15)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Position (cm)', fontsize=10)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)

plt.suptitle('Concave Mirror: Image Formation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('concave_mirror_cases.png', dpi=150)
plt.show()
```

---

## 2. Thin Lenses

### 2.1 The Thin Lens Equation

A thin lens (thickness negligible compared to the radii of curvature) obeys the same mathematical form as the mirror equation:

$$\frac{1}{s} + \frac{1}{s'} = \frac{1}{f}$$

with the important distinction that image distances are positive on the *opposite* side of the lens from the object (transmitted light side).

**Lateral magnification**:

$$m = -\frac{s'}{s}$$

### 2.2 The Lensmaker's Equation

The focal length of a thin lens depends on the refractive index $n$ of the lens material and the radii of curvature $R_1$ and $R_2$ of its two surfaces:

$$\frac{1}{f} = (n - 1)\left(\frac{1}{R_1} - \frac{1}{R_2}\right)$$

Sign convention: $R > 0$ if the center of curvature is on the transmission side (right side for light traveling left to right).

| Lens Type | Shape | $R_1$ | $R_2$ | $f$ |
|-----------|-------|-------|-------|-----|
| Biconvex | () | + | - | + (converging) |
| Plano-convex | D | + | $\infty$ | + (converging) |
| Biconcave | )( | - | + | - (diverging) |
| Plano-concave | ( | $\infty$ | + | - (diverging) |
| Meniscus (converging) | )( | + | + ($R_1 < R_2$) | + |
| Meniscus (diverging) | )( | + | + ($R_1 > R_2$) | - |

### 2.3 Optical Power

The **optical power** of a lens is the reciprocal of the focal length:

$$P = \frac{1}{f}$$

measured in **diopters** (D) when $f$ is in meters. A +2 D lens has $f = 0.5$ m. Eyeglass prescriptions use diopters.

### 2.4 Principal Ray Tracing for Thin Lenses

For a converging lens, the three principal rays are:

1. **Parallel ray**: Arrives parallel to axis → refracts through the back focal point $F'$
2. **Focal ray**: Passes through the front focal point $F$ → refracts parallel to axis
3. **Central ray**: Passes through the center of the lens → continues undeviated

```python
import numpy as np
import matplotlib.pyplot as plt

# Thin lens image formation: demonstrate converging and diverging cases
# Plot principal ray diagrams

def thin_lens_image(s, f):
    """Calculate image distance and magnification for a thin lens."""
    s_prime = 1 / (1/f - 1/s) if abs(1/f - 1/s) > 1e-10 else float('inf')
    m = -s_prime / s if s != 0 else float('inf')
    return s_prime, m

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Converging lens (f > 0) ---
ax = axes[0]
f_conv = 8.0   # cm
s_obj = 14.0   # cm (object beyond F)
s_prime, m = thin_lens_image(s_obj, f_conv)

# Draw lens (thin vertical double-arrow)
lens_height = 7
ax.annotate('', xy=(0, lens_height), xytext=(0, -lens_height),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax.axhline(0, color='gray', linewidth=0.5)

# Focal points
ax.plot(-f_conv, 0, 'ro', markersize=8, zorder=5)
ax.plot(f_conv, 0, 'ro', markersize=8, zorder=5)
ax.text(-f_conv, -1, 'F', ha='center', fontsize=10, color='red')
ax.text(f_conv, -1, "F'", ha='center', fontsize=10, color='red')

# Object
h_obj = 4.0
ax.annotate('', xy=(-s_obj, h_obj), xytext=(-s_obj, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))

# Image
h_img = m * h_obj
ax.annotate('', xy=(s_prime, h_img), xytext=(s_prime, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

# Ray 1: parallel → through F'
ax.plot([-s_obj, 0, s_prime + 5], [h_obj, h_obj, h_obj - (s_prime + 5) * h_obj / f_conv + h_obj],
        'b-', linewidth=1.2, alpha=0.7)
# Simplified: parallel ray bends through F'
ax.plot([-s_obj, 0], [h_obj, h_obj], 'b-', linewidth=1.2)
ax.plot([0, s_prime], [h_obj, h_img], 'b-', linewidth=1.2)

# Ray 2: through center (undeviated)
ax.plot([-s_obj, s_prime], [h_obj, h_img], 'orange', linewidth=1.2)

# Ray 3: through F → parallel after lens
ax.plot([-s_obj, 0], [h_obj, h_obj * (1 - s_obj / (-f_conv))], 'g-', linewidth=1.2, alpha=0.5)

ax.set_xlim(-22, 22)
ax.set_ylim(-8, 8)
ax.set_title(f"Converging Lens: f={f_conv}, s={s_obj}, s'={s_prime:.1f}, m={m:.2f}", fontsize=11)
ax.set_xlabel('Position (cm)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# --- Diverging lens (f < 0) ---
ax = axes[1]
f_div = -8.0
s_obj = 14.0
s_prime, m = thin_lens_image(s_obj, f_div)

# Draw lens (thin vertical concave shape)
ax.annotate('', xy=(0, lens_height), xytext=(0, -lens_height),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.axhline(0, color='gray', linewidth=0.5)

# Focal points (virtual for diverging lens)
ax.plot(f_div, 0, 'ro', markersize=8, zorder=5)
ax.plot(-f_div, 0, 'ro', markersize=8, zorder=5)
ax.text(f_div, -1, "F'", ha='center', fontsize=10, color='red')
ax.text(-f_div, -1, 'F', ha='center', fontsize=10, color='red')

# Object
ax.annotate('', xy=(-s_obj, h_obj), xytext=(-s_obj, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))

# Image (virtual — on the same side as object)
h_img = m * h_obj
ax.annotate('', xy=(s_prime, h_img), xytext=(s_prime, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5, linestyle='--'))

# Ray through center
ax.plot([-s_obj, 15], [h_obj, h_obj + (15 + s_obj) * (h_img - h_obj) / (s_prime + s_obj)],
        'orange', linewidth=1.2)

ax.set_xlim(-22, 22)
ax.set_ylim(-8, 8)
ax.set_title(f"Diverging Lens: f={f_div}, s={s_obj}, s'={s_prime:.1f}, m={m:.2f}", fontsize=11)
ax.set_xlabel('Position (cm)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('thin_lens_ray_diagrams.png', dpi=150)
plt.show()
```

---

## 3. Image Characteristics

### 3.1 Real vs. Virtual Images

| Property | Real Image | Virtual Image |
|----------|-----------|---------------|
| Formation | Light rays actually converge at the image point | Light rays appear to diverge from the image point |
| Can be projected? | Yes (onto a screen) | No |
| $s'$ sign | Positive (same side as transmitted light) | Negative (same side as incident light) |
| Example | Projector, camera sensor | Magnifying glass, flat mirror |

### 3.2 Image Formation Summary

**Converging lens/concave mirror** ($f > 0$):

| Object Position | Image Position | Image Type | Orientation | Size |
|----------------|---------------|------------|-------------|------|
| $s > 2f$ | $f < s' < 2f$ | Real | Inverted | Reduced |
| $s = 2f$ | $s' = 2f$ | Real | Inverted | Same size |
| $f < s < 2f$ | $s' > 2f$ | Real | Inverted | Magnified |
| $s = f$ | $s' = \infty$ | — | — | — |
| $s < f$ | $s' < 0$ (virtual) | Virtual | Upright | Magnified |

**Diverging lens/convex mirror** ($f < 0$): Always produces a virtual, upright, reduced image.

---

## 4. Compound Lens Systems

### 4.1 Two Thin Lenses in Contact

When two thin lenses with focal lengths $f_1$ and $f_2$ are placed in contact, the combined focal length is:

$$\frac{1}{f_{\text{total}}} = \frac{1}{f_1} + \frac{1}{f_2}$$

or equivalently, the powers add:

$$P_{\text{total}} = P_1 + P_2$$

This is how achromatic doublets work: a converging crown glass lens and a diverging flint glass lens are combined to cancel chromatic aberration while maintaining net converging power.

### 4.2 Two Thin Lenses Separated by Distance $d$

For two lenses separated by distance $d$:

$$\frac{1}{f_{\text{eff}}} = \frac{1}{f_1} + \frac{1}{f_2} - \frac{d}{f_1 f_2}$$

The image from the first lens becomes the object for the second lens. If the first lens produces an image at distance $s_1'$, the object distance for the second lens is:

$$s_2 = d - s_1'$$

Note: If $s_2 < 0$, the "object" for the second lens is virtual (the image from the first lens lies beyond the second lens).

### 4.3 The ABCD Matrix Method

Any paraxial optical element can be represented by a $2 \times 2$ **ray transfer matrix** that acts on the ray vector $\begin{pmatrix} y \\ \theta \end{pmatrix}$, where $y$ is the ray height and $\theta$ is the ray angle:

$$\begin{pmatrix} y_{\text{out}} \\ \theta_{\text{out}} \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} y_{\text{in}} \\ \theta_{\text{in}} \end{pmatrix}$$

Common matrices:

| Element | Matrix |
|---------|--------|
| Free space (distance $d$) | $\begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}$ |
| Thin lens (focal length $f$) | $\begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$ |
| Curved mirror (radius $R$) | $\begin{pmatrix} 1 & 0 \\ -2/R & 1 \end{pmatrix}$ |
| Flat interface ($n_1 \to n_2$) | $\begin{pmatrix} 1 & 0 \\ 0 & n_1/n_2 \end{pmatrix}$ |
| Curved interface (radius $R$, $n_1 \to n_2$) | $\begin{pmatrix} 1 & 0 \\ (n_1-n_2)/(n_2 R) & n_1/n_2 \end{pmatrix}$ |

For a system of elements, multiply the matrices from right to left (first element on the right):

$$M_{\text{system}} = M_N \cdot M_{N-1} \cdots M_2 \cdot M_1$$

The effective focal length is: $f_{\text{eff}} = -1/C$ where $C$ is the (2,1) element of the system matrix.

```python
import numpy as np

# ABCD Matrix method: trace rays through a compound optical system
# Example: two thin lenses separated by a distance

def free_space(d):
    """Transfer matrix for propagation through free space of distance d."""
    return np.array([[1, d],
                     [0, 1]])

def thin_lens(f):
    """Transfer matrix for a thin lens with focal length f."""
    return np.array([[1, 0],
                     [-1/f, 1]])

def curved_mirror(R):
    """Transfer matrix for a curved mirror with radius of curvature R."""
    return np.array([[1, 0],
                     [-2/R, 1]])

# Example system: two converging lenses
f1 = 10.0   # cm, first lens
f2 = 20.0   # cm, second lens
d = 15.0    # cm, separation between lenses

# System matrix: M = L2 * D * L1
# (multiply right to left: light hits L1 first, propagates distance d, then hits L2)
M = thin_lens(f2) @ free_space(d) @ thin_lens(f1)

print("System matrix M:")
print(f"  A = {M[0,0]:.4f}")
print(f"  B = {M[0,1]:.4f}")
print(f"  C = {M[1,0]:.4f}")
print(f"  D = {M[1,1]:.4f}")
print()

# Effective focal length from the C element
f_eff = -1 / M[1, 0]
print(f"Effective focal length: {f_eff:.2f} cm")

# Compare with the formula: 1/f_eff = 1/f1 + 1/f2 - d/(f1*f2)
f_formula = 1 / (1/f1 + 1/f2 - d/(f1*f2))
print(f"Formula result:         {f_formula:.2f} cm")
print()

# Trace a specific ray: object at s=25 cm from first lens, height y=2 cm
s = 25.0
y_in = 2.0
theta_in = -y_in / s  # angle for a ray from the object tip to the lens center

# Full system: free_space(s) on input side, then the optical system
# Then find where the output ray crosses the axis (image location)
ray_in = np.array([y_in, theta_in])

# After the two-lens system:
ray_out = M @ ray_in
y_out, theta_out = ray_out

# The image is where y = 0 after the last lens
# y_out + theta_out * s_prime = 0  =>  s_prime = -y_out / theta_out
if abs(theta_out) > 1e-10:
    s_image = -y_out / theta_out
    m = (y_out + theta_out * s_image) / y_in  # actually should be y_img / y_obj
    # Better: magnification from full matrix approach
    print(f"Object distance from L1: {s:.1f} cm")
    print(f"Image distance from L2:  {s_image:.2f} cm")
    print(f"Output ray: y = {y_out:.4f} cm, theta = {theta_out:.6f} rad")
```

---

## 5. Aberrations

Real lenses and mirrors deviate from the ideal thin-lens behavior. These deviations are called **aberrations**. The five primary monochromatic aberrations (Seidel aberrations) arise from the paraxial approximation breaking down for rays at larger angles or heights.

### 5.1 Spherical Aberration

**Cause**: Rays far from the optical axis are focused at a different point than paraxial rays.

**Effect**: Blurred images even on-axis. The amount of spherical aberration scales as $h^4$ (fourth power of ray height).

**Correction**: Aspheric surfaces, aperture stops (limiting ray height), combining positive and negative elements.

### 5.2 Coma

**Cause**: Off-axis points imaged through different annular zones of the lens are magnified differently.

**Effect**: Point sources off-axis appear as comet-shaped ("coma") blurs.

**Correction**: Aplanatic lens design (corrected for both spherical aberration and coma).

### 5.3 Astigmatism

**Cause**: For off-axis object points, the lens has different effective curvatures in the tangential and sagittal planes.

**Effect**: A point source forms two perpendicular line images at different distances. Between them is the "circle of least confusion."

**Correction**: Anastigmatic lens designs, curved image planes.

### 5.4 Field Curvature (Petzval Curvature)

**Cause**: Even after correcting astigmatism, the image of a flat object lies on a curved surface (the Petzval surface).

**Effect**: Center and edges of the field cannot be simultaneously in focus on a flat detector.

**Correction**: Field-flattening lenses, meniscus elements (the Petzval sum $\sum 1/(n_i f_i)$ must be minimized).

### 5.5 Distortion

**Cause**: Magnification varies with distance from the optical axis.

**Effect**: Straight lines appear curved. Two types:
- **Barrel distortion**: Magnification decreases away from center (wide-angle lenses)
- **Pincushion distortion**: Magnification increases away from center (telephoto lenses)

**Correction**: Symmetric lens arrangements, computational correction (common in smartphones).

### 5.6 Chromatic Aberration

This is a *polychromatic* aberration caused by dispersion — different wavelengths have different focal lengths.

**Longitudinal chromatic aberration**: Different colors focus at different distances along the axis.

**Lateral chromatic aberration**: Different colors have different magnifications, producing color fringes.

**Correction**: Achromatic doublets (crown + flint glass), apochromatic triplets, diffractive-refractive hybrids.

```python
import numpy as np
import matplotlib.pyplot as plt

# Spherical aberration: rays at different heights focus at different points
# Compare a perfect lens (all rays focus at f) vs. real spherical lens

def trace_spherical_lens(y_in, R, n_lens, thickness):
    """
    Simple ray trace through a thick symmetric biconvex lens.
    Uses exact Snell's law (not paraxial approximation) to show spherical aberration.

    y_in: input ray height (parallel to axis)
    R: radius of curvature of both surfaces (|R1| = |R2| = R)
    n_lens: refractive index of lens
    thickness: lens center thickness
    """
    n_air = 1.0

    # First surface: center at (R, 0), ray hits at height y_in
    # Find intersection with sphere: (x - R)^2 + y^2 = R^2 for the left surface
    x1 = R - np.sqrt(R**2 - y_in**2)

    # Normal at this point: pointing from center (R, 0) toward (x1, y_in)
    nx1 = (x1 - R) / R
    ny1 = y_in / R
    # Angle of incidence (ray is horizontal, so sin(theta_i) = |ny1| effectively)
    sin_theta1 = abs(ny1)  # horizontal ray hitting curved surface
    sin_theta1_r = sin_theta1 / n_lens  # Snell's law: refracted angle
    theta1_r = np.arcsin(sin_theta1_r)
    theta1 = np.arcsin(sin_theta1)

    # Refracted ray direction (2D rotation)
    # The deviation angle
    delta1 = theta1 - theta1_r  # bending toward normal

    # Simplified: the refracted ray slope inside the lens
    slope_inside = np.tan(-delta1) if y_in > 0 else np.tan(delta1)

    # Second surface: center at (thickness - R, 0) for symmetric biconvex
    # Approximate: the ray exits and crosses the axis at some distance
    # For a full trace we'd need the second surface intersection too

    # Use the thin lens formula with correction for ray height (exact trace)
    # For simplicity, use the paraxial focal length + spherical aberration term
    f_paraxial = R / (2 * (n_lens - 1))  # thin lens approximation for symmetric biconvex
    # Longitudinal spherical aberration scales as y^2
    # (3rd order: the marginal focus differs from paraxial focus)
    SA_coeff = 0.002  # empirical coefficient for demonstration
    f_actual = f_paraxial * (1 - SA_coeff * y_in**2)

    return f_actual

# Parameters
R = 50.0      # mm radius of curvature
n_glass = 1.5
thickness = 5.0

y_values = np.linspace(0.1, 20, 100)  # ray heights from 0.1 to 20 mm
f_paraxial = R / (2 * (n_glass - 1))  # = 50 mm for these values

# Calculate focus position for each ray height
# Using the exact formula for a spherical lens longitudinal aberration:
# Delta_f ≈ -h^2 / (2 * f * n * (n-1)) * [n^2 + (n-1)^2 * (3n+2)/(n-1)] ...
# Simplified: LSA ≈ -k * h^2 where k depends on lens shape and n
k_sa = 1 / (8 * f_paraxial * (n_glass - 1)**2)  # approximate coefficient
focus_positions = f_paraxial - k_sa * y_values**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: focus position vs ray height (shows spherical aberration)
ax1.plot(y_values, focus_positions, 'b-', linewidth=2)
ax1.axhline(f_paraxial, color='r', linestyle='--', label=f'Paraxial focus: {f_paraxial:.1f} mm')
ax1.set_xlabel('Ray height (mm)', fontsize=12)
ax1.set_ylabel('Focus position (mm)', fontsize=12)
ax1.set_title('Longitudinal Spherical Aberration', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: spot diagram — ray intersections in the focal plane
# Simulate 2D: rays enter at various heights and azimuthal angles
N_rays = 500
heights = np.random.uniform(0, 15, N_rays)  # random ray heights
azimuths = np.random.uniform(0, 2*np.pi, N_rays)  # random azimuthal angles

# Each ray's focus error (how far from paraxial focus it lands)
delta_f = -k_sa * heights**2
# In the paraxial focal plane, the ray misses the center by:
# transverse error ≈ delta_f * (height / f_paraxial)
x_spot = delta_f * np.cos(azimuths) * heights / f_paraxial
y_spot = delta_f * np.sin(azimuths) * heights / f_paraxial

ax2.scatter(x_spot, y_spot, s=1, alpha=0.5, c=heights, cmap='viridis')
ax2.set_xlabel('x (mm)', fontsize=12)
ax2.set_ylabel('y (mm)', fontsize=12)
ax2.set_title('Spot Diagram at Paraxial Focus\n(color = ray height)', fontsize=13)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(ax2.collections[0], ax=ax2, label='Ray height (mm)')

plt.tight_layout()
plt.savefig('spherical_aberration.png', dpi=150)
plt.show()
```

---

## 6. Chromatic Aberration and Achromatic Doublets

### 6.1 The Problem

Since $n(\lambda)$ varies with wavelength, a single lens has a different focal length for each color:

$$f(\lambda) = \frac{R_1 R_2}{(n(\lambda) - 1)(R_2 - R_1)}$$

Blue light focuses closer to the lens than red light (for normal dispersion), creating colored halos around images.

### 6.2 Achromatic Doublet Design

An achromatic doublet combines a crown glass lens (low dispersion, $V_1$ large) with a flint glass lens (high dispersion, $V_2$ small) such that the chromatic aberrations cancel:

$$\frac{P_1}{V_1} + \frac{P_2}{V_2} = 0$$

where $P_i = 1/f_i$ are the individual powers. Combined with the total power requirement $P_1 + P_2 = P_{\text{total}}$, we can solve for both:

$$P_1 = P_{\text{total}} \frac{V_1}{V_1 - V_2}, \qquad P_2 = -P_{\text{total}} \frac{V_2}{V_1 - V_2}$$

```python
import numpy as np

# Design an achromatic doublet lens
# Crown glass (BK7) + Flint glass (SF2)

# Abbe numbers and refractive indices
V_crown = 64.17   # BK7 Abbe number (low dispersion)
V_flint = 33.85   # SF2 Abbe number (high dispersion)
n_d_crown = 1.5168  # at 587.6 nm (d-line)
n_d_flint = 1.6477

# Desired total focal length
f_total = 100.0  # mm
P_total = 1 / f_total  # total power (1/mm)

# Individual powers that cancel chromatic aberration
P_crown = P_total * V_crown / (V_crown - V_flint)
P_flint = -P_total * V_flint / (V_crown - V_flint)

f_crown = 1 / P_crown
f_flint = 1 / P_flint

print("Achromatic Doublet Design")
print("=" * 40)
print(f"Total focal length:  {f_total:.1f} mm")
print(f"Crown lens (BK7):    f = {f_crown:.1f} mm  (P = {P_crown:.6f} mm⁻¹)")
print(f"Flint lens (SF2):    f = {f_flint:.1f} mm  (P = {P_flint:.6f} mm⁻¹)")
print(f"\nVerification:")
print(f"  P_crown + P_flint = {P_crown + P_flint:.6f} mm⁻¹ (should be {P_total:.6f})")
print(f"  P_crown/V_crown + P_flint/V_flint = {P_crown/V_crown + P_flint/V_flint:.8f} (should be ~0)")

# The crown element is converging (positive power) and the flint is diverging (negative power)
# Together they form a net converging doublet with minimal chromatic aberration
```

---

## 7. Thick Lenses and Cardinal Points

### 7.1 Principal Planes

For a thick lens or compound system, the thin lens equation still applies if we measure distances from the **principal planes** $H$ and $H'$ rather than from the lens surfaces:

$$\frac{1}{s_H} + \frac{1}{s'_{H'}} = \frac{1}{f}$$

where $s_H$ is the object distance measured from $H$ and $s'_{H'}$ is the image distance measured from $H'$.

### 7.2 The Six Cardinal Points

A thick lens has six cardinal points:

1. **Two focal points** ($F$, $F'$): Where parallel rays converge
2. **Two principal points** ($H$, $H'$): Where the lens appears to bend rays
3. **Two nodal points** ($N$, $N'$): Where an oblique ray passes through undeviated in angle

For a lens in the same medium on both sides: $N = H$ and $N' = H'$.

The ABCD matrix uniquely determines all cardinal points:

$$f = -\frac{1}{C}, \qquad \text{BFD} = -\frac{A}{C}, \qquad \text{FFD} = \frac{D}{C}$$

where BFD is the back focal distance (from last surface to $F'$) and FFD is the front focal distance (from first surface to $F$).

---

## Exercises

### Exercise 1: Mirror Problems

A concave mirror has a radius of curvature of 40 cm.

(a) Where is the image of an object placed 60 cm from the mirror? Is it real or virtual?

(b) What is the magnification? Is the image upright or inverted?

(c) Where must the object be placed to produce a virtual image that is 3 times the size of the object?

### Exercise 2: Lensmaker's Equation

Design a thin lens with a focal length of 15 cm using glass with $n = 1.52$.

(a) If the lens is biconvex with equal radii, what is $R$?

(b) If one surface is flat (plano-convex), what is the radius of the curved surface?

(c) Calculate the optical power in diopters for each design.

### Exercise 3: Compound System

Two thin lenses ($f_1 = 10$ cm, $f_2 = -20$ cm) are separated by 8 cm. An object is placed 15 cm to the left of the first lens.

(a) Find the final image position using sequential application of the thin lens equation.

(b) Verify your answer using the ABCD matrix method.

(c) What is the total magnification of the system?

### Exercise 4: Achromatic Doublet

You need to design an achromatic doublet with $f = 200$ mm using BK7 crown glass ($n_d = 1.517$, $V_d = 64.2$) and SF11 flint glass ($n_d = 1.785$, $V_d = 25.7$).

(a) Calculate the required focal lengths of the crown and flint elements.

(b) Using the lensmaker's equation, find suitable radii of curvature for each element (assume the cemented surface has a common radius).

(c) What is the residual secondary spectrum (difference in focal length between blue and red)?

### Exercise 5: Aberration Identification

For each scenario, identify the dominant aberration and suggest a correction:

(a) A telescope shows star images that are sharp at the center but elongated radially at the edges.

(b) A microscope objective produces colored fringes around high-contrast features.

(c) A wide-angle camera lens makes straight buildings appear curved at the frame edges.

(d) A large-aperture astronomical mirror produces blurred star images that improve when the aperture is reduced.

---

## Summary

| Concept | Key Formula / Fact |
|---------|-------------------|
| Mirror equation | $1/s + 1/s' = 1/f = 2/R$ |
| Thin lens equation | $1/s + 1/s' = 1/f$ |
| Lensmaker's equation | $1/f = (n-1)(1/R_1 - 1/R_2)$ |
| Magnification | $m = -s'/s$ |
| Optical power | $P = 1/f$ (diopters when $f$ in meters) |
| Lenses in contact | $1/f_{\text{total}} = 1/f_1 + 1/f_2$ |
| Separated lenses | $1/f_{\text{eff}} = 1/f_1 + 1/f_2 - d/(f_1 f_2)$ |
| ABCD matrix (lens) | $\begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$ |
| ABCD matrix (space) | $\begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}$ |
| Achromatic condition | $P_1/V_1 + P_2/V_2 = 0$ |
| Seidel aberrations | Spherical, coma, astigmatism, field curvature, distortion |
| Chromatic aberration | Different $\lambda$ → different $f$; corrected with achromatic doublets |

---

[← Previous: 02. Geometric Optics Fundamentals](02_Geometric_Optics_Fundamentals.md) | [Next: 04. Optical Instruments →](04_Optical_Instruments.md)

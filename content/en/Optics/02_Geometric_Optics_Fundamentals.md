# 02. Geometric Optics Fundamentals

[← Previous: 01. Nature of Light](01_Nature_of_Light.md) | [Next: 03. Mirrors and Lenses →](03_Mirrors_and_Lenses.md)

---

## Learning Objectives

1. Apply the ray model of light and understand its domain of validity (feature sizes $\gg \lambda$)
2. Derive Snell's law and the law of reflection from Fermat's principle of least time
3. Solve refraction problems at planar and curved interfaces using Snell's law
4. Calculate the critical angle for total internal reflection and explain its applications
5. Analyze prism dispersion and trace rays through multi-surface optical systems
6. Describe the eikonal equation as the bridge between wave optics and ray optics
7. Explain natural phenomena (mirages, rainbows, fiber optics) using geometric optics

---

## Why This Matters

Geometric optics is the workhorse of optical design. From the eyeglasses on your face to the camera in your phone to the mirrors in a space telescope, nearly all practical optical systems are designed using ray tracing — the systematic application of reflection and refraction at each surface. Even advanced computational optics and lens design software (Zemax, Code V) start from geometric optics before adding wave corrections. Mastering ray optics gives you the tools to understand and design most optical systems you will ever encounter.

> **Analogy**: Geometric optics is to wave optics what Newtonian mechanics is to quantum mechanics. It is an approximation that works brilliantly when objects are much larger than the wavelength of light — just as Newton's laws work brilliantly when objects are much larger than atomic scales. You use the simpler theory everywhere it applies, and bring in the full theory only when you must.

---

## 1. The Ray Model of Light

### 1.1 When Rays Work

In **geometric optics** (also called ray optics), light propagates as rays — straight lines in homogeneous media that bend at interfaces and curve in graded-index media. This model is valid when:

$$\text{Feature size} \gg \lambda$$

For visible light ($\lambda \sim 400$–$700$ nm), geometric optics works for objects larger than roughly 10 $\mu$m. Below this scale, diffraction becomes important and we need wave optics (Lessons 05–06).

### 1.2 Sign Conventions

Geometric optics calculations require consistent sign conventions. We use the **real-is-positive** convention:

| Quantity | Positive | Negative |
|----------|----------|----------|
| Object distance $s$ | Object on incoming side | Object on outgoing side (virtual) |
| Image distance $s'$ | Image on outgoing side (real) | Image on incoming side (virtual) |
| Focal length $f$ | Converging (concave mirror, convex lens) | Diverging (convex mirror, concave lens) |
| Radius of curvature $R$ | Center of curvature on outgoing side | Center of curvature on incoming side |
| Height $y$ | Above optical axis | Below optical axis |

Light travels from left to right by convention.

---

## 2. Fermat's Principle

### 2.1 Statement

**Fermat's Principle** (Principle of Least Time): Light travels between two points along the path for which the optical path length (OPL) is *stationary* — meaning the first-order variation vanishes:

$$\delta \int_A^B n(\mathbf{r}) \, ds = 0$$

In most cases, "stationary" means a *minimum* (the path of least time), though it can also be a maximum or saddle point (e.g., reflections from concave mirrors).

The **optical path length** from $A$ to $B$ is:

$$\text{OPL} = \int_A^B n(\mathbf{r}) \, ds$$

where $n(\mathbf{r})$ is the spatially varying refractive index. In a homogeneous medium, $\text{OPL} = n \cdot d$, where $d$ is the geometric distance.

### 2.2 Physical Interpretation

In vacuum, light always takes the shortest geometric path (a straight line). In the presence of media, light may take a bent path because the *time* along the bent path (through the faster medium) can be less than the *time* along the straight-line path (partly through the slower medium).

> **Analogy**: Imagine a lifeguard on the beach who spots a drowning swimmer offshore. The lifeguard runs faster on sand than she swims in water. The fastest route is *not* a straight line to the swimmer; instead, she runs along the beach until the angle is just right, then enters the water. Fermat's principle says light "chooses" its path the same way — by minimizing travel time.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fermat's principle: find the path that minimizes travel time
# A lifeguard analogy — running on beach (fast) then swimming (slow)

# Setup
beach_length = 20.0     # meters (horizontal extent)
lifeguard_pos = (0, 5)  # position on beach (x=0, y=5 from water line)
swimmer_pos = (15, -8)  # position in water (x=15, y=-8 from water line)

v_sand = 5.0   # running speed on sand (m/s)
v_water = 1.5  # swimming speed in water (m/s)

# Water line is at y = 0. The lifeguard enters water at point (x_entry, 0).
# We vary x_entry to find the minimum-time path.

x_entry = np.linspace(0, 20, 500)

# Distance on sand from lifeguard to entry point
d_sand = np.sqrt((x_entry - lifeguard_pos[0])**2 + lifeguard_pos[1]**2)
# Distance in water from entry point to swimmer
d_water = np.sqrt((swimmer_pos[0] - x_entry)**2 + swimmer_pos[1]**2)

# Total travel time = distance/speed for each segment
time_total = d_sand / v_sand + d_water / v_water

# Find the optimal entry point (minimum time)
idx_opt = np.argmin(time_total)
x_opt = x_entry[idx_opt]
t_opt = time_total[idx_opt]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: time vs entry point — shows clear minimum
ax1.plot(x_entry, time_total, 'b-', linewidth=2)
ax1.axvline(x_opt, color='r', linestyle='--', label=f'Optimal: x = {x_opt:.1f} m')
ax1.set_xlabel('Entry point x (m)', fontsize=12)
ax1.set_ylabel('Total time (s)', fontsize=12)
ax1.set_title("Fermat's Principle: Minimum Travel Time", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right panel: physical path — lifeguard → entry → swimmer
ax2.axhspan(-10, 0, alpha=0.2, color='cyan', label='Water')
ax2.axhspan(0, 8, alpha=0.15, color='sandybrown', label='Beach')
ax2.axhline(0, color='blue', linewidth=1)

# Draw three paths: straight line, optimal, and a suboptimal
for x_e, color, label, ls in [(swimmer_pos[0], 'gray', 'Straight line', '--'),
                                (x_opt, 'red', f'Optimal (t={t_opt:.2f}s)', '-'),
                                (3.0, 'orange', 'Suboptimal', ':')]:
    ax2.plot([lifeguard_pos[0], x_e], [lifeguard_pos[1], 0], color=color, linestyle=ls, linewidth=2)
    ax2.plot([x_e, swimmer_pos[0]], [0, swimmer_pos[1]], color=color, linestyle=ls, linewidth=2)

ax2.plot(*lifeguard_pos, 'ko', markersize=10, label='Lifeguard')
ax2.plot(*swimmer_pos, 'bx', markersize=12, markeredgewidth=3, label='Swimmer')
ax2.set_xlabel('x (m)', fontsize=12)
ax2.set_ylabel('y (m)', fontsize=12)
ax2.set_title('Path Geometry (Top View)', fontsize=13)
ax2.legend(fontsize=9, loc='upper left')
ax2.set_xlim(-2, 22)
ax2.set_ylim(-10, 8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fermat_principle.png', dpi=150)
plt.show()

# Verify: at the optimal point, sin(theta_sand)/sin(theta_water) = v_sand/v_water
# This is the lifeguard version of Snell's law!
theta_sand = np.arctan(x_opt / lifeguard_pos[1])
theta_water = np.arctan((swimmer_pos[0] - x_opt) / abs(swimmer_pos[1]))
print(f"sin(θ_sand)/sin(θ_water) = {np.sin(theta_sand)/np.sin(theta_water):.3f}")
print(f"v_sand/v_water = {v_sand/v_water:.3f}")
```

---

## 3. Law of Reflection

### 3.1 Statement

When a ray strikes a smooth surface, the reflected ray lies in the **plane of incidence** (the plane containing the incident ray and the surface normal), and:

$$\theta_i = \theta_r$$

The angle of incidence $\theta_i$ equals the angle of reflection $\theta_r$, both measured from the surface normal.

### 3.2 Derivation from Fermat's Principle

Consider a ray traveling from point $A$ to a mirror surface and then to point $B$. Let the mirror be along the $x$-axis, with $A$ at $(0, a)$ and $B$ at $(d, b)$. The ray hits the mirror at point $P = (x, 0)$.

The total path length is:

$$L(x) = \sqrt{x^2 + a^2} + \sqrt{(d-x)^2 + b^2}$$

Setting $dL/dx = 0$:

$$\frac{x}{\sqrt{x^2 + a^2}} = \frac{d-x}{\sqrt{(d-x)^2 + b^2}}$$

The left side is $\sin\theta_i$ and the right side is $\sin\theta_r$, giving us $\theta_i = \theta_r$.

### 3.3 Types of Reflection

- **Specular reflection**: From smooth surfaces (mirrors, calm water). Produces clear images.
- **Diffuse reflection**: From rough surfaces (paper, walls). Scatters light in all directions. This is why we can see objects from any viewing angle — they scatter ambient light diffusely.

The boundary between specular and diffuse reflection depends on the surface roughness relative to the wavelength. If surface irregularities are much smaller than $\lambda$, the reflection is specular.

---

## 4. Snell's Law of Refraction

### 4.1 Statement

When light passes from a medium with refractive index $n_1$ into a medium with refractive index $n_2$, the refracted ray obeys:

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

where $\theta_1$ is the angle of incidence and $\theta_2$ is the angle of refraction, both measured from the normal.

### 4.2 Derivation from Fermat's Principle

Consider a ray from $A$ in medium 1 to $B$ in medium 2, crossing the interface at point $P$. The optical path length is:

$$\text{OPL}(x) = n_1 \sqrt{x^2 + a^2} + n_2 \sqrt{(d-x)^2 + b^2}$$

Setting $d(\text{OPL})/dx = 0$:

$$n_1 \frac{x}{\sqrt{x^2 + a^2}} = n_2 \frac{d-x}{\sqrt{(d-x)^2 + b^2}}$$

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

### 4.3 Consequences of Snell's Law

**Going from less dense to more dense medium** ($n_1 < n_2$): The ray bends *toward* the normal ($\theta_2 < \theta_1$).

**Going from more dense to less dense medium** ($n_1 > n_2$): The ray bends *away* from the normal ($\theta_2 > \theta_1$).

**At normal incidence** ($\theta_1 = 0$): No bending occurs. $\theta_2 = 0$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize refraction at a flat interface using Snell's law
# Shows how the refracted angle depends on the angle of incidence

n1, n2 = 1.0, 1.5  # air to glass

theta1_deg = np.linspace(0, 89, 200)
theta1_rad = np.deg2rad(theta1_deg)

# Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
sin_theta2 = (n1 / n2) * np.sin(theta1_rad)
theta2_rad = np.arcsin(sin_theta2)
theta2_deg = np.rad2deg(theta2_rad)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: angle relationship
ax1.plot(theta1_deg, theta2_deg, 'b-', linewidth=2, label=f'$n_1={n1}$ → $n_2={n2}$ (air→glass)')
ax1.plot(theta1_deg, theta1_deg, 'k--', alpha=0.3, label='No refraction ($n_1=n_2$)')
ax1.set_xlabel('Angle of incidence $\\theta_1$ (degrees)', fontsize=12)
ax1.set_ylabel('Angle of refraction $\\theta_2$ (degrees)', fontsize=12)
ax1.set_title("Snell's Law: Refraction Angles", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: ray diagram showing incident and refracted rays
ax2.axhspan(-5, 0, alpha=0.15, color='lightblue', label=f'Medium 2 (n={n2})')
ax2.axhspan(0, 5, alpha=0.05, color='white', label=f'Medium 1 (n={n1})')
ax2.axhline(0, color='gray', linewidth=2)

# Draw normal (dashed vertical)
ax2.plot([0, 0], [-4, 4], 'k--', linewidth=1, alpha=0.5, label='Normal')

# Draw rays for several angles
for theta1_val in [15, 30, 45, 60, 75]:
    t1 = np.deg2rad(theta1_val)
    t2 = np.arcsin((n1 / n2) * np.sin(t1))

    # Incident ray: comes from upper-left, hits origin
    x_inc = -4 * np.sin(t1)
    y_inc = 4 * np.cos(t1)
    ax2.annotate('', xy=(0, 0), xytext=(x_inc, y_inc),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Refracted ray: leaves origin, goes into lower medium
    x_ref = 4 * np.sin(t2)
    y_ref = -4 * np.cos(t2)
    ax2.annotate('', xy=(x_ref, y_ref), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax2.text(x_inc - 0.3, y_inc + 0.2, f'{theta1_val}°', fontsize=8, color='blue')

ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Ray Diagram: Air → Glass', fontsize=13)
ax2.set_aspect('equal')
ax2.legend(fontsize=9, loc='lower left')

plt.tight_layout()
plt.savefig('snells_law.png', dpi=150)
plt.show()
```

---

## 5. Total Internal Reflection

### 5.1 Critical Angle

When light travels from a denser medium to a less dense medium ($n_1 > n_2$), there exists a **critical angle** $\theta_c$ beyond which all light is reflected:

$$\sin\theta_c = \frac{n_2}{n_1}$$

For $\theta_1 > \theta_c$, Snell's law gives $\sin\theta_2 > 1$, which has no real solution — no transmitted ray exists.

| Interface | $n_1$ | $n_2$ | Critical Angle |
|-----------|-------|-------|---------------|
| Glass → Air | 1.50 | 1.00 | 41.8° |
| Water → Air | 1.33 | 1.00 | 48.8° |
| Diamond → Air | 2.42 | 1.00 | 24.4° |
| Optical fiber (core → cladding) | 1.48 | 1.46 | 80.6° |

Diamond's very low critical angle means light bouncing inside the gem undergoes many total internal reflections before escaping, producing the characteristic brilliance and "fire."

### 5.2 Applications of Total Internal Reflection

**Optical fibers**: Light is guided along thin glass fibers by repeated total internal reflection at the core-cladding interface. This is the backbone of modern telecommunications.

**Prisms**: Right-angle prisms can redirect light by 90° or 180° using total internal reflection, which is more efficient than metallic mirrors (no absorption loss).

**Fingerprint sensors**: Frustrated total internal reflection (FTIR) detects where skin ridges touch a glass surface, disrupting the TIR condition.

### 5.3 Evanescent Waves

Even when total internal reflection occurs, the electromagnetic field does not abruptly vanish at the interface. Instead, an **evanescent wave** penetrates into the second medium, decaying exponentially:

$$E(z) = E_0 \exp\left(-\frac{z}{\delta}\right)$$

where the penetration depth is:

$$\delta = \frac{\lambda}{2\pi\sqrt{n_1^2 \sin^2\theta_1 - n_2^2}}$$

The evanescent wave carries no net energy into the second medium under normal TIR. However, if another dense medium is brought close (within $\sim \lambda$), the evanescent wave can couple into it — this is **frustrated total internal reflection** (FTIR), analogous to quantum tunneling.

```python
import numpy as np
import matplotlib.pyplot as plt

# Evanescent wave: intensity decays exponentially beyond the interface
# during total internal reflection

n1 = 1.50   # glass
n2 = 1.00   # air
wavelength = 550e-9  # green light (550 nm)

# Calculate penetration depth for several angles beyond the critical angle
theta_c = np.arcsin(n2 / n1)
angles_deg = [42, 50, 60, 70, 80]  # all > critical angle (41.8°)

z = np.linspace(0, 2000, 500)  # distance into medium 2, in nm

fig, ax = plt.subplots(figsize=(10, 6))

for angle_deg in angles_deg:
    theta = np.deg2rad(angle_deg)

    # Penetration depth: how far the evanescent field extends
    denom = np.sqrt(n1**2 * np.sin(theta)**2 - n2**2)
    delta = wavelength / (2 * np.pi * denom)  # in meters
    delta_nm = delta * 1e9  # convert to nm for plotting

    # Intensity decays as exp(-2z/delta) since I ~ E^2
    intensity = np.exp(-2 * z / delta_nm)

    ax.plot(z, intensity, linewidth=2,
            label=f'$\\theta$ = {angle_deg}° ($\\delta$ = {delta_nm:.0f} nm)')

ax.set_xlabel('Distance into medium 2 (nm)', fontsize=12)
ax.set_ylabel('Relative Intensity $I/I_0$', fontsize=12)
ax.set_title(f'Evanescent Wave Decay (glass→air, $\\theta_c$ = {np.rad2deg(theta_c):.1f}°)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('evanescent_wave.png', dpi=150)
plt.show()
```

---

## 6. Prism Optics

### 6.1 Refraction Through a Prism

A prism with apex angle $A$ bends light by a deviation angle $\delta$. For a ray at minimum deviation:

$$n = \frac{\sin\left(\frac{A + \delta_{\min}}{2}\right)}{\sin\left(\frac{A}{2}\right)}$$

This is a standard method for measuring refractive indices with high accuracy.

### 6.2 Prism Dispersion

Because $n$ depends on wavelength, a prism separates white light into its spectral components:

$$\frac{d\delta}{d\lambda} = \frac{d\delta}{dn} \cdot \frac{dn}{d\lambda}$$

The geometric factor $d\delta/dn$ depends on the prism angle and orientation. The material factor $dn/d\lambda$ depends on the glass type (higher dispersion for flint glass than crown glass).

### 6.3 Rainbow Formation

A raindrop acts as a spherical prism. Parallel rays from the sun enter the drop at different heights and undergo:

1. **Refraction** at the front surface
2. **Reflection** at the back surface (one for primary, two for secondary)
3. **Refraction** at the front surface again on exit

The deviation angle varies with impact parameter, and there is a **minimum deviation** (Descartes ray) at about 138° — meaning the maximum concentration of light appears at $180° - 138° = 42°$ from the antisolar point.

```python
import numpy as np
import matplotlib.pyplot as plt

# Rainbow optics: trace rays through a spherical water droplet
# and find the Descartes minimum deviation angle

def rainbow_deviation(b, n, k=1):
    """
    Calculate the deviation angle for a ray hitting a spherical droplet.

    b: impact parameter (0 to 1, normalized to droplet radius)
    n: refractive index of water
    k: number of internal reflections (1 for primary, 2 for secondary)

    Returns: deviation angle in degrees
    """
    # Angle of incidence from the impact parameter
    theta_i = np.arcsin(b)
    # Angle of refraction (Snell's law)
    theta_r = np.arcsin(b / n)
    # Total deviation: accounts for entry refraction, k internal reflections, exit refraction
    deviation = 2 * theta_i - 2 * (k + 1) * theta_r + k * np.pi
    return np.rad2deg(deviation)

b = np.linspace(0.01, 0.99, 1000)  # impact parameter

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate for different wavelengths (different n values for water)
wavelengths = {
    'Red (700nm)': 1.3312,
    'Yellow (580nm)': 1.3335,
    'Blue (450nm)': 1.3400,
}

for label, n_water in wavelengths.items():
    dev = rainbow_deviation(b, n_water, k=1)
    ax.plot(b, dev, linewidth=2, label=label)

    # Find the minimum deviation (Descartes ray)
    idx_min = np.argmin(np.abs(np.gradient(dev)))
    ax.plot(b[idx_min], dev[idx_min], 'o', markersize=8)
    ax.annotate(f'  min = {dev[idx_min]:.1f}°',
                xy=(b[idx_min], dev[idx_min]), fontsize=9)

ax.set_xlabel('Impact parameter b (normalized)', fontsize=12)
ax.set_ylabel('Deviation angle (degrees)', fontsize=12)
ax.set_title('Primary Rainbow: Deviation Angle vs Impact Parameter', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(135, 180)
plt.tight_layout()
plt.savefig('rainbow_deviation.png', dpi=150)
plt.show()
```

---

## 7. The Eikonal Equation

### 7.1 From Waves to Rays

The eikonal equation provides the rigorous mathematical bridge between wave optics and ray optics. Starting from the wave equation and assuming a slowly varying envelope:

$$\mathbf{E}(\mathbf{r}) = \mathbf{E}_0(\mathbf{r}) \exp\left(i k_0 S(\mathbf{r})\right)$$

where $S(\mathbf{r})$ is the **eikonal** (optical path function) and $k_0 = 2\pi/\lambda_0$ is the free-space wave number. In the limit $\lambda \to 0$ (geometric optics limit), the wave equation reduces to the **eikonal equation**:

$$|\nabla S|^2 = n^2(\mathbf{r})$$

### 7.2 Physical Meaning

The surfaces of constant $S$ are the **wavefronts**, and the rays are the curves perpendicular to these wavefronts:

$$\frac{d\mathbf{r}}{ds} = \frac{\nabla S}{n}$$

This equation tells us that rays curve toward regions of higher refractive index. In a homogeneous medium ($n$ = constant), $\nabla S$ is constant and rays are straight lines.

### 7.3 Ray Equation in Graded-Index Media

For a medium with spatially varying $n(\mathbf{r})$, the ray path satisfies:

$$\frac{d}{ds}\left(n \frac{d\mathbf{r}}{ds}\right) = \nabla n$$

This is the **ray equation** — a second-order ODE that, given initial position and direction, determines the complete ray path.

**Application: Mirages**

In a hot desert, the air near the ground is hotter and less dense, so $n$ decreases toward the ground. The ray equation predicts that rays curve *upward* away from the surface. An observer sees the sky reflected in what appears to be a puddle of water — a mirage.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ray tracing in a graded-index medium (mirage simulation)
# Temperature gradient near hot ground creates a vertical n gradient

def n_profile(y):
    """
    Refractive index profile for air above a hot surface.
    n decreases toward the ground due to temperature-induced density decrease.
    This is a simplified exponential model.
    """
    n_ground = 1.000250   # slightly lower n near the hot ground
    n_inf = 1.000293      # standard air n at altitude
    scale_height = 2.0     # meters — height over which the gradient relaxes
    return n_inf - (n_inf - n_ground) * np.exp(-y / scale_height)

def dn_dy(y):
    """Derivative of n with respect to y (vertical gradient)."""
    n_ground = 1.000250
    n_inf = 1.000293
    scale_height = 2.0
    return (n_inf - n_ground) / scale_height * np.exp(-y / scale_height)

def ray_equations(s, state):
    """
    ODE system for the ray equation in 2D: d/ds(n dr/ds) = grad(n)
    state = [x, y, dx/ds, dy/ds]
    Using the parameterization where ds is arc length.
    """
    x, y, dx_ds, dy_ds = state
    n = n_profile(y)
    dndx = 0        # n doesn't depend on x (horizontal homogeneity)
    dndy = dn_dy(y)

    # d/ds(n * dx/ds) = dn/dx => n * d2x/ds2 + dn/ds * dx/ds = dn/dx
    # But dn/ds = (dn/dy) * (dy/ds), and similarly we need the full equations
    # Simplified: d2x/ds2 = (1/n)(dndx - dndy*dy_ds*dx_ds/n)
    # d2y/ds2 = (1/n)(dndy - (dndy*dy_ds^2 + dndx*dx_ds*dy_ds)/n)

    # Cleaner approach: let p = n*dx/ds, q = n*dy/ds
    # dp/ds = dndx, dq/ds = dndy
    # But we track (x, y, vx, vy) where vx=dx/ds, vy=dy/ds
    d2x_ds2 = (dndx - (dndx * dx_ds + dndy * dy_ds) * dx_ds) / n
    d2y_ds2 = (dndy - (dndx * dx_ds + dndy * dy_ds) * dy_ds) / n

    return [dx_ds, dy_ds, d2x_ds2, d2y_ds2]

# Launch rays from x=0 at various downward angles
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top panel: ray paths (mirage effect)
for angle_deg in [-0.02, -0.04, -0.06, -0.08, -0.10]:
    angle_rad = np.deg2rad(angle_deg)
    y0 = 1.7  # observer eye height (meters)

    # Initial conditions: position (0, y0), direction (cos(a), sin(a))
    initial = [0, y0, np.cos(angle_rad), np.sin(angle_rad)]

    # Integrate the ray equation along the arc length parameter
    sol = solve_ivp(ray_equations, [0, 5000], initial,
                    max_step=1.0, events=None, dense_output=True)

    # Only plot rays that stay above ground (y > 0)
    mask = sol.y[1] > 0
    ax1.plot(sol.y[0][mask], sol.y[1][mask], linewidth=1.5,
             label=f'{angle_deg}°')

ax1.set_xlabel('Horizontal distance (m)', fontsize=12)
ax1.set_ylabel('Height (m)', fontsize=12)
ax1.set_title('Ray Paths in a Mirage (Hot Ground)', fontsize=13)
ax1.legend(fontsize=9, title='Launch angle')
ax1.set_ylim(0, 2)
ax1.grid(True, alpha=0.3)

# Bottom panel: refractive index profile
y_range = np.linspace(0, 5, 200)
ax2.plot(n_profile(y_range), y_range, 'b-', linewidth=2)
ax2.set_xlabel('Refractive index n', fontsize=12)
ax2.set_ylabel('Height (m)', fontsize=12)
ax2.set_title('Refractive Index Profile (n decreases near hot ground)', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mirage_ray_tracing.png', dpi=150)
plt.show()
```

---

## 8. Refraction at Curved Surfaces

### 8.1 Single Refracting Surface

For a spherical interface of radius $R$ separating media with indices $n_1$ and $n_2$:

$$\frac{n_1}{s} + \frac{n_2}{s'} = \frac{n_2 - n_1}{R}$$

where $s$ is the object distance and $s'$ is the image distance. This is the foundation for the thin lens equation (Lesson 03).

### 8.2 Apparent Depth

When looking into a pool of water, objects appear closer to the surface than they actually are. For near-normal viewing:

$$\text{Apparent depth} = \frac{\text{Actual depth}}{n}$$

A coin at 2 m depth in water ($n = 1.33$) appears to be at about 1.5 m depth.

---

## Exercises

### Exercise 1: Fermat's Principle Derivation

A light ray travels from point $A = (0, 3)$ in air ($n_1 = 1$) to point $B = (6, -4)$ in glass ($n_2 = 1.5$), crossing the flat interface at $y = 0$.

(a) Write the optical path length as a function of the crossing point $x$.

(b) Take the derivative and set it to zero to find the optimal $x$.

(c) Verify that the resulting angles satisfy Snell's law.

### Exercise 2: Total Internal Reflection

An optical fiber has a core refractive index of 1.48 and cladding index of 1.46.

(a) Calculate the critical angle at the core-cladding interface.

(b) What is the maximum acceptance angle (numerical aperture) for light entering the fiber from air?

(c) If the fiber is bent, how does the critical angle condition change? At what bend radius does light begin to leak?

### Exercise 3: Prism Analysis

A 60° equilateral prism is made of flint glass with $n = 1.62$ at 589 nm.

(a) Calculate the minimum deviation angle $\delta_{\min}$.

(b) At what angle of incidence does minimum deviation occur?

(c) If $n = 1.64$ at 450 nm, what is the angular dispersion between 589 nm and 450 nm light at minimum deviation?

### Exercise 4: Evanescent Wave

Light at 633 nm travels in glass ($n_1 = 1.52$) and hits the glass-air interface at 45°.

(a) Verify that total internal reflection occurs.

(b) Calculate the evanescent wave penetration depth $\delta$.

(c) If a second glass surface is placed at a distance of $\delta/2$ from the first, approximately what fraction of the intensity tunnels through?

### Exercise 5: Mirage Estimation

On a hot road, the air temperature gradient creates a refractive index profile that can be approximated as $n(y) = n_0(1 + \alpha y)$ where $y$ is height above the road, $n_0 = 1.000250$, and $\alpha = 1.7 \times 10^{-5}$ m$^{-1}$.

(a) At what angle must a ray be launched horizontally from $y = 1.5$ m for it to just graze the road surface ($y = 0$)?

(b) How far away (horizontally) does the mirage appear?

---

## Summary

| Concept | Key Formula / Fact |
|---------|-------------------|
| Ray model validity | Feature size $\gg \lambda$ |
| Fermat's principle | $\delta \int n \, ds = 0$ — light takes the path of stationary optical path length |
| Optical path length | $\text{OPL} = \int n \, ds = n \cdot d$ (homogeneous medium) |
| Law of reflection | $\theta_i = \theta_r$ |
| Snell's law | $n_1 \sin\theta_1 = n_2 \sin\theta_2$ |
| Critical angle | $\sin\theta_c = n_2/n_1$ (requires $n_1 > n_2$) |
| Evanescent wave depth | $\delta = \lambda / (2\pi\sqrt{n_1^2\sin^2\theta - n_2^2})$ |
| Prism minimum deviation | $n = \sin\frac{A+\delta_{\min}}{2} / \sin\frac{A}{2}$ |
| Eikonal equation | $|\nabla S|^2 = n^2(\mathbf{r})$ — bridge from wave to ray optics |
| Ray equation | $\frac{d}{ds}(n\frac{d\mathbf{r}}{ds}) = \nabla n$ — rays curve toward higher $n$ |
| Curved surface refraction | $n_1/s + n_2/s' = (n_2 - n_1)/R$ |
| Apparent depth | depth$_{\text{apparent}} = $ depth$_{\text{actual}} / n$ |

---

[← Previous: 01. Nature of Light](01_Nature_of_Light.md) | [Next: 03. Mirrors and Lenses →](03_Mirrors_and_Lenses.md)

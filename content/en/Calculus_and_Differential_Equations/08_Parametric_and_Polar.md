# 08. Parametric Curves and Polar Coordinates

## Learning Objectives

- Represent curves using parametric equations and compute velocity, acceleration, and slope
- Calculate arc length for curves defined parametrically
- Convert between Cartesian and polar coordinate systems and identify standard polar curves
- Compute areas and arc lengths in polar coordinates
- Visualize parametric and polar curves using Python (Matplotlib)

---

## 1. Parametric Equations

Sometimes a curve in the plane cannot be described as $y = f(x)$. A **parametric representation** uses a third variable -- the **parameter** $t$ -- to describe both coordinates simultaneously:

$$x = f(t), \quad y = g(t), \quad a \le t \le b$$

Think of $t$ as time: as the clock ticks from $a$ to $b$, the point $(x(t), y(t))$ traces out a path. This is exactly how an animation works -- each frame has a time stamp that determines where the dot appears.

**Example: The Circle**

The unit circle $x^2 + y^2 = 1$ is not a function of $x$ (it fails the vertical line test), but parametrically:

$$x = \cos t, \quad y = \sin t, \quad 0 \le t \le 2\pi$$

### 1.1 Velocity and Acceleration

If $t$ represents time, the **velocity vector** is:

$$\mathbf{v}(t) = \left(\frac{dx}{dt},\, \frac{dy}{dt}\right)$$

The **speed** (magnitude of velocity) is:

$$\|\mathbf{v}(t)\| = \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2}$$

The **acceleration vector** is:

$$\mathbf{a}(t) = \left(\frac{d^2x}{dt^2},\, \frac{d^2y}{dt^2}\right)$$

**Example: Projectile Motion**

A ball launched at angle $\theta$ with initial speed $v_0$ (ignoring air resistance):

$$x(t) = v_0 \cos\theta \cdot t, \quad y(t) = v_0 \sin\theta \cdot t - \tfrac{1}{2}g t^2$$

- $v_0$: initial speed (m/s)
- $\theta$: launch angle (radians)
- $g$: gravitational acceleration ($\approx 9.8\,\text{m/s}^2$)

The velocity components are $\dot{x} = v_0 \cos\theta$ (constant) and $\dot{y} = v_0 \sin\theta - gt$ (linearly decreasing).

### 1.2 Slope of Parametric Curves

To find the slope $dy/dx$ without eliminating the parameter, use the **chain rule**:

$$\frac{dy}{dx} = \frac{dy/dt}{dx/dt} \quad \text{(provided } dx/dt \neq 0\text{)}$$

This is a powerful trick: instead of solving for $y$ in terms of $x$, we divide derivatives with respect to $t$.

For the **second derivative** (concavity):

$$\frac{d^2y}{dx^2} = \frac{\frac{d}{dt}\!\left(\frac{dy}{dx}\right)}{dx/dt}$$

Note: this is NOT $\frac{d^2y/dt^2}{d^2x/dt^2}$ -- a common mistake.

**Example:** For $x = t^2$, $y = t^3$:

$$\frac{dy}{dx} = \frac{3t^2}{2t} = \frac{3t}{2}$$

$$\frac{d^2y}{dx^2} = \frac{\frac{d}{dt}(3t/2)}{2t} = \frac{3/2}{2t} = \frac{3}{4t}$$

### 1.3 Arc Length of Parametric Curves

The length of a parametric curve from $t = a$ to $t = b$ is found by summing up infinitesimal distance elements $ds$:

$$L = \int_a^b \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2}\, dt$$

**Intuition:** At each instant, the point moves $dx$ horizontally and $dy$ vertically. By the Pythagorean theorem, the tiny distance traveled is $\sqrt{dx^2 + dy^2}$. Dividing by $dt$ and integrating gives the total distance.

**Example: Circumference of a Circle**

For $x = R\cos t$, $y = R\sin t$, $0 \le t \le 2\pi$:

$$L = \int_0^{2\pi} \sqrt{R^2\sin^2 t + R^2\cos^2 t}\, dt = \int_0^{2\pi} R\, dt = 2\pi R$$

Exactly the well-known formula.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Parametric curve examples ---

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Projectile motion
v0, theta, g = 20.0, np.radians(45), 9.8
t_flight = 2 * v0 * np.sin(theta) / g  # total flight time
t = np.linspace(0, t_flight, 200)
x_proj = v0 * np.cos(theta) * t
y_proj = v0 * np.sin(theta) * t - 0.5 * g * t**2

axes[0].plot(x_proj, y_proj, 'b-', linewidth=2)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_title('Projectile Motion')
axes[0].set_aspect('equal')
axes[0].grid(True)

# 2. Cycloid: the curve traced by a point on a rolling wheel
t = np.linspace(0, 4 * np.pi, 500)
R = 1.0
x_cyc = R * (t - np.sin(t))
y_cyc = R * (1 - np.cos(t))

axes[1].plot(x_cyc, y_cyc, 'r-', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Cycloid (Rolling Wheel)')
axes[1].set_aspect('equal')
axes[1].grid(True)

# 3. Lissajous figure: superposition of two perpendicular oscillations
t = np.linspace(0, 2 * np.pi, 1000)
x_lis = np.sin(3 * t)
y_lis = np.sin(4 * t + np.pi / 4)

axes[2].plot(x_lis, y_lis, 'g-', linewidth=1)
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Lissajous Figure (3:4)')
axes[2].set_aspect('equal')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# --- Arc length computation ---
# Arc length of one arch of cycloid: x = t - sin(t), y = 1 - cos(t), 0 <= t <= 2pi
def cycloid_speed(t):
    """Speed along the cycloid: sqrt((dx/dt)^2 + (dy/dt)^2)."""
    dxdt = 1 - np.cos(t)
    dydt = np.sin(t)
    return np.sqrt(dxdt**2 + dydt**2)

arc_length, _ = quad(cycloid_speed, 0, 2 * np.pi)
print(f"Arc length of one cycloid arch: {arc_length:.6f}")
print(f"Theoretical value (8R = 8):     {8 * R:.6f}")
```

---

## 2. Polar Coordinates

### 2.1 The Polar System

Instead of locating a point by its horizontal and vertical distances $(x, y)$, **polar coordinates** use:

- $r$: distance from the origin (the **pole**)
- $\theta$: angle measured counter-clockwise from the positive $x$-axis

**Conversion formulas:**

$$x = r\cos\theta, \quad y = r\sin\theta$$

$$r = \sqrt{x^2 + y^2}, \quad \theta = \arctan\!\left(\frac{y}{x}\right)$$

The $\arctan$ formula requires care with quadrants; in Python, use `np.arctan2(y, x)`.

**Analogy:** Think of polar coordinates as radar: you specify "how far" ($r$) and "in which direction" ($\theta$) rather than "how far east" and "how far north."

### 2.2 Common Polar Curves

| Curve | Equation | Shape |
|-------|----------|-------|
| Circle | $r = a$ | Circle of radius $a$ centered at origin |
| Cardioid | $r = a(1 + \cos\theta)$ | Heart-shaped curve |
| Rose ($n$ petals) | $r = a\cos(n\theta)$ | $n$ petals if $n$ odd, $2n$ if $n$ even |
| Limaçon | $r = a + b\cos\theta$ | Varies with $a/b$ ratio |
| Archimedean Spiral | $r = a\theta$ | Evenly spaced spiral |
| Logarithmic Spiral | $r = ae^{b\theta}$ | Self-similar spiral (nautilus shell) |

### 2.3 Area in Polar Coordinates

The area enclosed by a polar curve $r = f(\theta)$ from $\theta = \alpha$ to $\theta = \beta$ is:

$$A = \frac{1}{2}\int_\alpha^\beta r^2\, d\theta = \frac{1}{2}\int_\alpha^\beta [f(\theta)]^2\, d\theta$$

**Derivation:** A thin "pie slice" at angle $\theta$ with angular width $d\theta$ approximates a triangle with base $r\,d\theta$ and height $r$, giving area $\tfrac{1}{2}r^2\,d\theta$.

**Example:** Area enclosed by the cardioid $r = 1 + \cos\theta$:

$$A = \frac{1}{2}\int_0^{2\pi}(1 + \cos\theta)^2\, d\theta = \frac{1}{2}\int_0^{2\pi}(1 + 2\cos\theta + \cos^2\theta)\, d\theta = \frac{3\pi}{2}$$

### 2.4 Arc Length in Polar Coordinates

For a polar curve $r = f(\theta)$, the arc length from $\theta = \alpha$ to $\theta = \beta$ is:

$$L = \int_\alpha^\beta \sqrt{r^2 + \left(\frac{dr}{d\theta}\right)^2}\, d\theta$$

**Derivation:** Start from the parametric arc length formula with $x = r\cos\theta$, $y = r\sin\theta$ and parameter $\theta$. After expanding and simplifying using $\cos^2\theta + \sin^2\theta = 1$, this formula emerges.

**Example:** Arc length of the Archimedean spiral $r = \theta$ from $\theta = 0$ to $\theta = 2\pi$:

$$L = \int_0^{2\pi} \sqrt{\theta^2 + 1}\, d\theta$$

This integral can be evaluated using the substitution $\theta = \sinh u$ or numerically.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                          subplot_kw={'projection': 'polar'})

# 1. Cardioid
theta = np.linspace(0, 2 * np.pi, 500)
r_card = 1 + np.cos(theta)
axes[0, 0].plot(theta, r_card, 'b-', linewidth=2)
axes[0, 0].set_title('Cardioid\n$r = 1 + \\cos\\theta$', pad=15)

# 2. Rose curve (3 petals)
r_rose3 = np.cos(3 * theta)
axes[0, 1].plot(theta, r_rose3, 'r-', linewidth=2)
axes[0, 1].set_title('Rose (3 petals)\n$r = \\cos(3\\theta)$', pad=15)

# 3. Rose curve (4 petals, n=2 even gives 2n=4)
r_rose4 = np.cos(2 * theta)
axes[0, 2].plot(theta, r_rose4, 'g-', linewidth=2)
axes[0, 2].set_title('Rose (4 petals)\n$r = \\cos(2\\theta)$', pad=15)

# 4. Limaçon with inner loop (a/b < 1)
r_lima = 0.5 + np.cos(theta)
axes[1, 0].plot(theta, r_lima, 'm-', linewidth=2)
axes[1, 0].set_title('Limaçon\n$r = 0.5 + \\cos\\theta$', pad=15)

# 5. Archimedean spiral
theta_sp = np.linspace(0, 4 * np.pi, 500)
r_arch = 0.3 * theta_sp
axes[1, 1].plot(theta_sp, r_arch, 'c-', linewidth=2)
axes[1, 1].set_title('Archimedean Spiral\n$r = 0.3\\theta$', pad=15)

# 6. Logarithmic spiral
r_log = np.exp(0.1 * theta_sp)
axes[1, 2].plot(theta_sp, r_log, color='orange', linewidth=2)
axes[1, 2].set_title('Log Spiral\n$r = e^{0.1\\theta}$', pad=15)

plt.tight_layout()
plt.show()

# --- Area and arc length computations ---

# Area of cardioid r = 1 + cos(theta)
area_card, _ = quad(lambda th: 0.5 * (1 + np.cos(th))**2, 0, 2 * np.pi)
print(f"Cardioid area:       {area_card:.6f}")
print(f"Theoretical (3π/2):  {1.5 * np.pi:.6f}")

# Arc length of Archimedean spiral r = theta, 0 to 2pi
def spiral_ds(th):
    """Integrand for polar arc length: sqrt(r^2 + (dr/dtheta)^2)."""
    r = th
    drdt = 1.0
    return np.sqrt(r**2 + drdt**2)

arc_spiral, _ = quad(spiral_ds, 0, 2 * np.pi)
print(f"Archimedean spiral arc length (0 to 2π): {arc_spiral:.6f}")
```

---

## 3. Slope and Tangent Lines for Polar Curves

A polar curve $r = f(\theta)$ can be viewed parametrically with $\theta$ as the parameter:

$$x = r\cos\theta = f(\theta)\cos\theta, \quad y = r\sin\theta = f(\theta)\sin\theta$$

Applying the parametric slope formula:

$$\frac{dy}{dx} = \frac{dy/d\theta}{dx/d\theta} = \frac{f'(\theta)\sin\theta + f(\theta)\cos\theta}{f'(\theta)\cos\theta - f(\theta)\sin\theta}$$

where $f'(\theta) = dr/d\theta$.

**Horizontal tangent** occurs when $dy/d\theta = 0$ (and $dx/d\theta \neq 0$).

**Vertical tangent** occurs when $dx/d\theta = 0$ (and $dy/d\theta \neq 0$).

```python
import numpy as np
import matplotlib.pyplot as plt

# Find tangent lines on the cardioid r = 1 + cos(theta)
theta_vals = np.linspace(0, 2 * np.pi, 1000)
r_vals = 1 + np.cos(theta_vals)

# Convert to Cartesian for plotting
x = r_vals * np.cos(theta_vals)
y = r_vals * np.sin(theta_vals)

# Compute dy/dtheta and dx/dtheta
dr = -np.sin(theta_vals)
dxdt = dr * np.cos(theta_vals) - r_vals * np.sin(theta_vals)
dydt = dr * np.sin(theta_vals) + r_vals * np.cos(theta_vals)

# Find approximate horizontal tangent points (dy/dtheta ≈ 0)
horiz_mask = np.abs(dydt) < 0.01
# Filter out points where dx/dtheta is also near zero (cusps)
horiz_mask &= np.abs(dxdt) > 0.1

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, 'b-', linewidth=2, label='Cardioid')

# Mark special points
ax.plot(x[horiz_mask], y[horiz_mask], 'ro', markersize=6,
        label='Horizontal tangent')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Tangent Lines on Cardioid')
ax.set_aspect('equal')
ax.legend()
ax.grid(True)
plt.show()
```

---

## 4. Applications and Connections

**Parametric curves appear everywhere:**
- Robotics: joint angles parameterize end-effector paths
- Computer graphics: Bézier curves and splines are parametric
- Physics: orbital mechanics uses time as the parameter
- Engineering: CNC machining paths are specified parametrically

**Polar coordinates simplify problems with circular symmetry:**
- Antenna radiation patterns
- Planetary orbits (Kepler's laws use polar form)
- Spiral structures in nature (nautilus shells, galaxies)

---

## Practice Problems

**1.** A particle moves along the curve $x = e^t\cos t$, $y = e^t\sin t$ for $0 \le t \le 2\pi$.
   - (a) Find the velocity and speed at $t = 0$.
   - (b) Show that the speed is $\sqrt{2}\,e^t$.
   - (c) Find the total arc length.

**2.** For the parametric curve $x = 2\cos t + \cos 2t$, $y = 2\sin t + \sin 2t$ (an **epicycloid**):
   - (a) Find $dy/dx$ at $t = \pi/4$.
   - (b) Find all values of $t$ where the tangent is horizontal on $[0, 2\pi)$.
   - (c) Compute the total arc length numerically.

**3.** Find the area enclosed by one petal of the rose curve $r = \sin(3\theta)$. (Hint: one petal spans $0 \le \theta \le \pi/3$.)

**4.** Compute the arc length of the logarithmic spiral $r = e^{0.2\theta}$ from $\theta = 0$ to $\theta = 4\pi$. Verify numerically with `scipy.integrate.quad`.

**5.** A satellite follows the polar orbit $r = \frac{p}{1 + e\cos\theta}$ (conic section in polar form) with semi-latus rectum $p = 7000\,\text{km}$ and eccentricity $e = 0.1$.
   - (a) Find the periapsis (closest approach) and apoapsis (farthest distance).
   - (b) Plot the orbit in polar coordinates.
   - (c) Find the area swept by the radius vector from $\theta = 0$ to $\theta = \pi$ (Kepler's second law).

---

## References

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapters 10.1-10.4
- **Gilbert Strang**, *Calculus*, Chapter 9 (Polar Coordinates and Complex Numbers)
- **3Blue1Brown**, "Parametric Curves" (visual intuition)
- **Matplotlib polar plot documentation**: https://matplotlib.org/stable/gallery/pie_and_polar_charts/index.html

---

[Previous: Sequences and Series](./07_Sequences_and_Series.md) | [Next: Multivariable Functions](./09_Multivariable_Functions.md)

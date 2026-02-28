# Applications of Integration

## Learning Objectives

- **Calculate** the area between two curves by identifying which function is on top and integrating the difference
- **Compute** volumes of solids of revolution using disk, washer, and shell methods
- **Derive** and evaluate arc length formulas for curves given in Cartesian and parametric form
- **Calculate** surface areas of revolution and relate them to arc length
- **Apply** integration to physical problems including work, hydrostatic force, and center of mass

## Introduction

Integration was born from the need to calculate areas, but its applications extend far beyond. Every time you need to add up infinitely many infinitesimal contributions -- slicing a solid into thin disks, unrolling a curve into tiny line segments, or summing forces across a surface -- integration is the tool.

This lesson covers the geometric and physical applications of definite integrals. These problems all share the same strategy: slice the object into thin pieces, write an expression for each piece, and integrate to sum them all up.

## Area Between Curves

### Two Curves: Top Minus Bottom

If $f(x) \geq g(x)$ on $[a, b]$, the area between the curves is:

$$A = \int_a^b [f(x) - g(x)] \, dx$$

**Why top minus bottom?** Each thin vertical strip has height $f(x) - g(x)$ and width $dx$. We sum (integrate) all these strips from $a$ to $b$.

**Important:** If the curves cross within $[a, b]$, you must split the integral at each intersection point and swap which function is "on top."

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x_sym = sp.Symbol('x')

# Find area between y = x^2 and y = x + 2
f = x_sym + 2
g = x_sym**2

# Step 1: Find intersection points
intersections = sp.solve(f - g, x_sym)
a, b = float(intersections[0]), float(intersections[1])
print(f"Intersection points: x = {intersections}")

# Step 2: Determine which is on top (f > g between intersections)
# At x = 0: f(0) = 2, g(0) = 0, so f is on top

# Step 3: Integrate
area = sp.integrate(f - g, (x_sym, intersections[0], intersections[1]))
print(f"Area = integral from {a} to {b} of [(x+2) - x^2] dx = {area} = {float(area):.4f}")

# Visualization
x_vals = np.linspace(-2.5, 3.5, 500)
f_vals = x_vals + 2
g_vals = x_vals**2

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_vals, f_vals, 'b-', linewidth=2, label='$y = x + 2$')
ax.plot(x_vals, g_vals, 'r-', linewidth=2, label='$y = x^2$')

# Shade the area between curves
x_fill = np.linspace(a, b, 300)
ax.fill_between(x_fill, x_fill + 2, x_fill**2, alpha=0.3, color='green',
                label=f'Area = {float(area):.2f}')

ax.plot([a, b], [a+2, b+2], 'ko', markersize=8, zorder=5)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Area Between Two Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('area_between_curves.png', dpi=150)
plt.show()
```

## Volumes of Revolution

When a region is rotated around an axis, it sweeps out a 3D solid. We compute the volume by slicing this solid into thin pieces.

### Disk Method

When rotating around the $x$-axis, each cross-section perpendicular to the axis is a **disk** (filled circle):

$$V = \int_a^b \pi [f(x)]^2 \, dx$$

- $f(x)$: the radius of the disk at position $x$
- $[f(x)]^2$: the area of a circular cross-section ($\pi r^2$)
- $dx$: the thickness of each disk

**Analogy:** Imagine stacking infinitely many thin coins. Each coin has a different radius determined by the curve.

### Washer Method

If there is a hole in the center (rotation of a region between two curves), each cross-section is a **washer** (annulus):

$$V = \int_a^b \pi \left([R(x)]^2 - [r(x)]^2\right) dx$$

- $R(x)$: outer radius (farther from axis)
- $r(x)$: inner radius (closer to axis)

### Shell Method

Sometimes it is easier to use **cylindrical shells** instead of disks. When rotating around the $y$-axis, a thin vertical strip at position $x$ sweeps out a cylindrical shell:

$$V = \int_a^b 2\pi x \cdot f(x) \, dx$$

- $2\pi x$: the circumference of the shell (at distance $x$ from the axis)
- $f(x)$: the height of the shell
- $dx$: the thickness

**When to use which:**
- **Disk/Washer**: cross-sections perpendicular to the axis of rotation are simple
- **Shell**: cross-sections parallel to the axis of rotation are simpler

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

x_sym = sp.Symbol('x')

# Example: Rotate y = sqrt(x) from x=0 to x=4 around the x-axis
# Disk method: V = pi * integral_0^4 (sqrt(x))^2 dx = pi * integral_0^4 x dx
V_disk = sp.pi * sp.integrate(x_sym, (x_sym, 0, 4))
print(f"Disk method: V = pi * integral_0^4 x dx = {V_disk} = {float(V_disk):.4f}")

# Shell method for the same solid (rotating around x-axis):
# We must express x as a function of y: x = y^2, and y ranges from 0 to 2
y_sym = sp.Symbol('y')
V_shell = 2 * sp.pi * sp.integrate(y_sym * (4 - y_sym**2), (y_sym, 0, 2))
print(f"Shell method: V = 2pi * integral_0^2 y*(4-y^2) dy = {V_shell} = {float(V_shell):.4f}")
print(f"Both methods agree: {V_disk == V_shell}")

# 3D visualization of the solid of revolution
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface of revolution
theta = np.linspace(0, 2*np.pi, 100)
x_3d = np.linspace(0, 4, 100)
Theta, X = np.meshgrid(theta, x_3d)

# y = sqrt(x) rotated around x-axis gives r = sqrt(x)
R = np.sqrt(X)
Y = R * np.cos(Theta)
Z = R * np.sin(Theta)

ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_title('Solid of Revolution: $y = \\sqrt{x}$ rotated around $x$-axis')
plt.tight_layout()
plt.savefig('volume_revolution_3d.png', dpi=150)
plt.show()
```

### Washer Example

Find the volume obtained by rotating the region between $y = x$ and $y = x^2$ (from $x=0$ to $x=1$) around the $x$-axis.

$$V = \int_0^1 \pi\left[(x)^2 - (x^2)^2\right] dx = \pi \int_0^1 (x^2 - x^4) \, dx = \pi\left[\frac{x^3}{3} - \frac{x^5}{5}\right]_0^1 = \frac{2\pi}{15}$$

```python
import sympy as sp

x = sp.Symbol('x')

# Washer method: region between y=x and y=x^2, rotated about x-axis
# Outer radius R(x) = x (farther from axis), inner radius r(x) = x^2
V_washer = sp.pi * sp.integrate(x**2 - x**4, (x, 0, 1))
print(f"Washer volume: {V_washer} = {float(V_washer):.6f}")
```

## Arc Length

### Cartesian Form

The length of a curve $y = f(x)$ from $x = a$ to $x = b$ is:

$$L = \int_a^b \sqrt{1 + [f'(x)]^2} \, dx$$

**Derivation:** A tiny piece of the curve has horizontal length $dx$ and vertical length $dy = f'(x) \, dx$. By the Pythagorean theorem, the arc length element is:

$$ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx$$

Integrating $ds$ gives the total arc length.

### Parametric Form

If the curve is given by $x = x(t)$, $y = y(t)$ for $t \in [\alpha, \beta]$:

$$L = \int_\alpha^\beta \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2} \, dt$$

```python
import numpy as np
import sympy as sp
from scipy import integrate

x = sp.Symbol('x')

# Arc length of y = x^(3/2) from x=0 to x=4
f = x**sp.Rational(3, 2)
f_prime = sp.diff(f, x)
integrand = sp.sqrt(1 + f_prime**2)
print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")
print(f"Arc length integrand: sqrt(1 + (f')^2) = {sp.simplify(integrand)}")

# Symbolic integration
L_exact = sp.integrate(integrand, (x, 0, 4))
print(f"Exact arc length: {L_exact} = {float(L_exact):.6f}")

# Numerical verification using scipy
f_numeric = lambda t: np.sqrt(1 + (1.5 * np.sqrt(t))**2)
L_numerical, _ = integrate.quad(f_numeric, 0, 4)
print(f"Numerical arc length: {L_numerical:.6f}")

# Parametric example: circle x = cos(t), y = sin(t), t in [0, 2pi]
t = sp.Symbol('t')
x_param = sp.cos(t)
y_param = sp.sin(t)
ds = sp.sqrt(sp.diff(x_param, t)**2 + sp.diff(y_param, t)**2)
L_circle = sp.integrate(ds, (t, 0, 2*sp.pi))
print(f"\nCircle circumference (parametric): {L_circle} = {float(L_circle):.6f}")
```

## Surface Area of Revolution

When a curve $y = f(x)$ is rotated around the $x$-axis, the surface area is:

$$S = \int_a^b 2\pi f(x) \sqrt{1 + [f'(x)]^2} \, dx$$

**Intuition:** Each arc length element $ds$ sweeps out a thin band (frustum) of circumference $2\pi f(x)$. The area of this band is $2\pi f(x) \, ds$.

This combines two ideas:
- The arc length element $ds = \sqrt{1 + [f'(x)]^2} \, dx$
- The circumference $2\pi r$ of the circle traced by each point

```python
import numpy as np
import sympy as sp

x = sp.Symbol('x')

# Surface area of y = sqrt(x) from x=0 to x=1, rotated about x-axis
f = sp.sqrt(x)
f_prime = sp.diff(f, x)
integrand = 2 * sp.pi * f * sp.sqrt(1 + f_prime**2)

print(f"Surface area integrand: {sp.simplify(integrand)}")
SA = sp.integrate(integrand, (x, 0, 1))
print(f"Surface area = {SA} = {float(SA):.6f}")

# Verify: surface area of a sphere (rotate y = sqrt(r^2 - x^2) about x-axis)
r = sp.Symbol('r', positive=True)
f_sphere = sp.sqrt(r**2 - x**2)
f_sphere_prime = sp.diff(f_sphere, x)
integrand_sphere = 2 * sp.pi * f_sphere * sp.sqrt(1 + f_sphere_prime**2)
SA_sphere = sp.integrate(sp.simplify(integrand_sphere), (x, -r, r))
print(f"\nSurface area of sphere: {SA_sphere}")
# Should give 4*pi*r^2, the well-known formula
```

## Physical Applications

### Work

In physics, **work** is the integral of force over distance:

$$W = \int_a^b F(x) \, dx$$

- $F(x)$: force as a function of position (in Newtons)
- $dx$: infinitesimal displacement
- $W$: total work (in Joules)

**Example: Spring work.** Hooke's law says $F(x) = kx$ where $k$ is the spring constant and $x$ is the displacement from equilibrium. The work to stretch a spring from $x = 0$ to $x = d$:

$$W = \int_0^d kx \, dx = \frac{1}{2} k d^2$$

```python
import sympy as sp

x = sp.Symbol('x')
k = sp.Symbol('k', positive=True)
d = sp.Symbol('d', positive=True)

# Work to stretch a spring from 0 to d
W_spring = sp.integrate(k * x, (x, 0, d))
print(f"Work to stretch spring: W = {W_spring}")

# Numerical example: k = 200 N/m, stretch 0.3 m
W_numeric = W_spring.subs([(k, 200), (d, 0.3)])
print(f"With k=200 N/m, d=0.3 m: W = {float(W_numeric)} J")
```

### Hydrostatic Pressure and Force

A dam or submerged plate experiences pressure that increases with depth. The force on a horizontal strip at depth $h$ below the surface:

$$dF = \rho g h \cdot w(h) \, dh$$

where $\rho$ is the fluid density ($\approx 1000$ kg/m$^3$ for water), $g \approx 9.8$ m/s$^2$, and $w(h)$ is the width of the plate at depth $h$.

Total force:

$$F = \int_0^H \rho g h \cdot w(h) \, dh$$

### Center of Mass

For a lamina (thin flat plate) with density $\rho$ bounded by $y = f(x)$ on $[a, b]$:

$$\bar{x} = \frac{\int_a^b x \cdot f(x) \, dx}{\int_a^b f(x) \, dx}, \qquad \bar{y} = \frac{\frac{1}{2}\int_a^b [f(x)]^2 \, dx}{\int_a^b f(x) \, dx}$$

- $\bar{x}$: the $x$-coordinate of the center of mass (weighted average of $x$)
- $\bar{y}$: the $y$-coordinate (each horizontal strip's center is at height $f(x)/2$)

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol('x')

# Center of mass of a semicircular lamina: y = sqrt(1 - x^2)
f = sp.sqrt(1 - x**2)

# Total area (mass, assuming uniform density)
area = sp.integrate(f, (x, -1, 1))  # Should be pi/2
print(f"Area = {area}")

# x-bar: by symmetry, should be 0
x_bar = sp.integrate(x * f, (x, -1, 1)) / area
print(f"x_bar = {x_bar}")

# y-bar
y_bar = sp.Rational(1, 2) * sp.integrate(f**2, (x, -1, 1)) / area
print(f"y_bar = {y_bar} = {float(y_bar):.6f}")
# y_bar = 4/(3*pi) ≈ 0.4244

# Visualize with center of mass marked
theta = np.linspace(0, np.pi, 200)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

fig, ax = plt.subplots(figsize=(8, 6))
ax.fill(np.append(x_circle, x_circle[-1]),
        np.append(y_circle, 0), alpha=0.3, color='blue')
ax.plot(x_circle, y_circle, 'b-', linewidth=2)
ax.plot([-1, 1], [0, 0], 'b-', linewidth=2)
ax.plot(float(x_bar), float(y_bar), 'r*', markersize=15, zorder=5,
        label=f'Center of mass: (0, {float(y_bar):.4f})')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Center of Mass of a Semicircular Lamina')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('center_of_mass.png', dpi=150)
plt.show()
```

## Summary

- **Area between curves**: $\int_a^b [f(x) - g(x)] \, dx$ -- always subtract the lower from the upper function
- **Disk method**: $V = \pi \int [f(x)]^2 \, dx$ -- for solids with no hole, slicing perpendicular to the axis
- **Washer method**: $V = \pi \int [R^2 - r^2] \, dx$ -- for solids with a hole
- **Shell method**: $V = 2\pi \int x \cdot f(x) \, dx$ -- cylindrical shells, often easier when revolving around the $y$-axis
- **Arc length**: $L = \int \sqrt{1 + [f'(x)]^2} \, dx$ -- Pythagorean theorem applied infinitesimally
- **Surface area**: $S = 2\pi \int f(x) \sqrt{1 + [f'(x)]^2} \, dx$ -- arc length times circumference
- **Physical applications**: work, hydrostatic force, and center of mass all follow the "slice, approximate, integrate" paradigm

## Practice Problems

### Problem 1: Area Between Curves

Find the area of the region enclosed by $y = \sin x$ and $y = \cos x$ between $x = 0$ and $x = \pi/2$. (Hint: determine where the curves intersect and which is on top in each sub-interval.)

### Problem 2: Volume by Disk/Washer

The region bounded by $y = \sqrt{x}$, $y = 0$, and $x = 4$ is rotated about the $x$-axis. Find the volume using:
(a) The disk method
(b) Verify your answer with Python

### Problem 3: Volume by Shell Method

The region bounded by $y = x - x^2$ and $y = 0$ is rotated about the $y$-axis. Use the shell method to find the volume. Then compute the same volume using the washer method (with respect to $y$) and verify they agree.

### Problem 4: Arc Length

Compute the arc length of $y = \frac{x^2}{2} - \frac{\ln x}{4}$ from $x = 1$ to $x = 2$. (This integral simplifies nicely -- show why.)

### Problem 5: Physical Application

A conical tank (point down) has height 6 m and top radius 3 m, and is filled with water ($\rho = 1000$ kg/m$^3$). Compute the work required to pump all the water out of the top of the tank.

(Hint: A thin horizontal slice of water at height $y$ from the bottom has radius $r = y/2$ and must be lifted a distance $(6 - y)$.)

## References

- Stewart, *Calculus: Early Transcendentals*, Ch. 6 (Applications of Integration)
- [3Blue1Brown: Volumes of Revolution](https://www.youtube.com/watch?v=rjLJIVoQxz4)
- [Paul's Online Notes: Applications of Integrals](https://tutorial.math.lamar.edu/Classes/CalcI/AreaBetweenCurves.aspx)

---

[Previous: Integration Techniques](./05_Integration_Techniques.md) | [Next: Sequences and Series](./07_Sequences_and_Series.md)

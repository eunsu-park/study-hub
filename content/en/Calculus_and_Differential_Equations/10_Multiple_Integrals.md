# 10. Multiple Integrals

## Learning Objectives

- Evaluate double integrals over rectangular and general regions using iterated integrals
- Apply Fubini's theorem to change the order of integration
- Transform double and triple integrals into polar, cylindrical, and spherical coordinates
- Use the Jacobian to perform general changes of variables in multiple integrals
- Compute physical quantities (mass, center of mass, moments of inertia) using multiple integrals

---

## 1. Double Integrals over Rectangles

### 1.1 Definition

The **double integral** of $f(x, y)$ over a rectangle $R = [a, b] \times [c, d]$ is defined as the limit of Riemann sums:

$$\iint_R f(x, y)\, dA = \lim_{m,n \to \infty} \sum_{i=1}^{m}\sum_{j=1}^{n} f(x_i^*, y_j^*)\,\Delta A$$

where $\Delta A = \Delta x\, \Delta y$ is the area of each small sub-rectangle.

**Geometric interpretation:** When $f(x, y) \ge 0$, the double integral gives the **volume** under the surface $z = f(x, y)$ and above the region $R$.

**Analogy:** A single integral $\int_a^b f(x)\,dx$ adds up thin vertical strips of area. A double integral adds up tiny vertical columns of volume -- stacking them across an entire 2D region.

### 1.2 Iterated Integrals

**Fubini's Theorem** (rectangular case): If $f$ is continuous on $R = [a, b] \times [c, d]$, then:

$$\iint_R f(x, y)\, dA = \int_a^b \left[\int_c^d f(x, y)\, dy\right] dx = \int_c^d \left[\int_a^b f(x, y)\, dx\right] dy$$

The two iterated integrals give the same result. This converts a 2D integration problem into two successive 1D integrations.

**Example:**

$$\iint_R xy^2\, dA, \quad R = [0, 2] \times [1, 3]$$

$$= \int_0^2 \left[\int_1^3 xy^2\, dy\right] dx = \int_0^2 x\left[\frac{y^3}{3}\right]_1^3 dx = \int_0^2 x \cdot \frac{26}{3}\, dx = \frac{26}{3}\left[\frac{x^2}{2}\right]_0^2 = \frac{52}{3}$$

---

## 2. Double Integrals over General Regions

### 2.1 Type I and Type II Regions

Not all regions are rectangles. We distinguish two convenient types:

**Type I** (bounded by curves $y = g_1(x)$ and $y = g_2(x)$):

$$\iint_D f(x, y)\, dA = \int_a^b \int_{g_1(x)}^{g_2(x)} f(x, y)\, dy\, dx$$

**Type II** (bounded by curves $x = h_1(y)$ and $x = h_2(y)$):

$$\iint_D f(x, y)\, dA = \int_c^d \int_{h_1(y)}^{h_2(y)} f(x, y)\, dx\, dy$$

### 2.2 Changing the Order of Integration

Sometimes one order of integration leads to an integral that is difficult or impossible to evaluate in closed form, while the other order is straightforward.

**Strategy:** Sketch the region $D$, identify the boundaries, and re-express them for the other order.

**Example:** Evaluate $\int_0^1 \int_x^1 e^{y^2}\, dy\, dx$.

The inner integral $\int_x^1 e^{y^2}\,dy$ has no elementary antiderivative. Switch the order:

- Region: $0 \le x \le y$, $0 \le y \le 1$ (a triangle below $y = x$ line, above $x$-axis)
- Reversed: $\int_0^1 \int_0^y e^{y^2}\, dx\, dy = \int_0^1 y\,e^{y^2}\, dy = \frac{1}{2}(e - 1)$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# --- Double integral over a triangular region ---
# Region: 0 <= x <= y, 0 <= y <= 1

# Numerical computation using scipy
result, error = dblquad(
    lambda x, y: np.exp(y**2),  # integrand f(x, y)
    0, 1,                        # y limits: 0 to 1
    lambda y: 0,                 # x lower limit: 0
    lambda y: y                  # x upper limit: y
)

print(f"Numerical result: {result:.8f}")
print(f"Exact value (e-1)/2: {(np.e - 1) / 2:.8f}")

# Visualize the region
fig, ax = plt.subplots(figsize=(6, 6))
# Fill the triangular region
triangle = plt.Polygon([(0, 0), (0, 1), (1, 1)], alpha=0.3, color='blue')
ax.add_patch(triangle)
ax.plot([0, 1], [0, 1], 'b-', linewidth=2, label='$y = x$')
ax.plot([0, 0], [0, 1], 'b-', linewidth=2)
ax.plot([0, 1], [1, 1], 'b-', linewidth=2)

# Annotate
ax.annotate('Type I: $x \\leq y \\leq 1$', xy=(0.15, 0.6), fontsize=12)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Region for Changing Order of Integration')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 3. Double Integrals in Polar Coordinates

When the region $D$ has **circular symmetry**, polar coordinates simplify the integral dramatically.

### 3.1 The Transformation

Substituting $x = r\cos\theta$, $y = r\sin\theta$:

$$\iint_D f(x, y)\, dA = \int_\alpha^\beta \int_{r_1(\theta)}^{r_2(\theta)} f(r\cos\theta,\, r\sin\theta)\, r\, dr\, d\theta$$

**The extra factor $r$** is crucial. It comes from the **Jacobian** of the coordinate transformation:

$$dA = dx\,dy = r\,dr\,d\theta$$

**Why the $r$ factor?** A small "box" in polar coordinates is not a rectangle but a curved wedge. Its area is approximately $(dr)(r\,d\theta) = r\,dr\,d\theta$. The further from the origin, the larger the arc length.

**Example:** Evaluate $\iint_D e^{-(x^2 + y^2)}\, dA$ where $D$ is the disk $x^2 + y^2 \le 4$.

In polar: $x^2 + y^2 = r^2$, and $D$ becomes $0 \le r \le 2$, $0 \le \theta \le 2\pi$:

$$\int_0^{2\pi}\int_0^2 e^{-r^2}\, r\, dr\, d\theta = 2\pi \int_0^2 r\,e^{-r^2}\, dr = 2\pi\left[-\frac{1}{2}e^{-r^2}\right]_0^2 = \pi(1 - e^{-4})$$

This technique is key to evaluating the **Gaussian integral** $\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}$.

---

## 4. Triple Integrals

### 4.1 Cartesian Coordinates

The triple integral of $f(x, y, z)$ over a region $E$ in 3D space:

$$\iiint_E f(x, y, z)\, dV$$

is evaluated as an iterated integral with three levels of nesting. The order depends on how $E$ is described.

**Example:** Integrate $f = xyz$ over the tetrahedron bounded by $x = 0$, $y = 0$, $z = 0$, and $x + y + z = 1$:

$$\int_0^1 \int_0^{1-x} \int_0^{1-x-y} xyz\, dz\, dy\, dx$$

### 4.2 Cylindrical Coordinates

**Cylindrical coordinates** $(r, \theta, z)$ extend polar coordinates by adding a vertical axis:

$$x = r\cos\theta, \quad y = r\sin\theta, \quad z = z$$

$$dV = r\, dr\, d\theta\, dz$$

**Best for:** regions with circular cross-sections (cylinders, cones, etc.).

### 4.3 Spherical Coordinates

**Spherical coordinates** $(\rho, \theta, \phi)$:

- $\rho$: distance from origin
- $\theta$: azimuthal angle (same as polar $\theta$)
- $\phi$: polar angle measured from the positive $z$-axis

$$x = \rho\sin\phi\cos\theta, \quad y = \rho\sin\phi\sin\theta, \quad z = \rho\cos\phi$$

$$dV = \rho^2 \sin\phi\, d\rho\, d\phi\, d\theta$$

The volume element $\rho^2\sin\phi$ accounts for the distortion of a small "box" in spherical coordinates.

**Best for:** regions with spherical symmetry (balls, spherical shells).

**Example:** Volume of a sphere of radius $R$:

$$V = \int_0^{2\pi}\int_0^{\pi}\int_0^R \rho^2\sin\phi\, d\rho\, d\phi\, d\theta = 2\pi \cdot 2 \cdot \frac{R^3}{3} = \frac{4}{3}\pi R^3$$

```python
import numpy as np
from scipy.integrate import tplquad, dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Triple integral: volume of intersection of sphere and cylinder ---
# Sphere: x^2 + y^2 + z^2 <= 4 (radius 2)
# Cylinder: x^2 + y^2 <= 1 (radius 1)
# We compute the volume inside both using cylindrical coordinates.

# In cylindrical: 0 <= r <= 1, 0 <= theta <= 2pi,
# -sqrt(4 - r^2) <= z <= sqrt(4 - r^2)

def integrand_cyl(z, r, theta):
    """Volume element in cylindrical: r * dz * dr * dtheta."""
    return r

volume, error = tplquad(
    integrand_cyl,
    0, 2 * np.pi,              # theta limits
    lambda theta: 0,            # r lower
    lambda theta: 1.0,          # r upper
    lambda theta, r: -np.sqrt(4 - r**2),  # z lower
    lambda theta, r: np.sqrt(4 - r**2)    # z upper
)

print(f"Volume of sphere-cylinder intersection: {volume:.6f}")
# Analytical: 2*pi * integral_0^1 of 2*sqrt(4-r^2) * r dr
# = 2*pi * [-2/3 * (4-r^2)^(3/2)]_0^1 = 2*pi * (16/3 - 2*sqrt(3)*3/3)
analytical = 2 * np.pi * (2/3) * (4**(3/2) - 3**(3/2))
print(f"Analytical value:                       {analytical:.6f}")

# --- Spherical coordinates: mass of a hemisphere with density rho(x,y,z) = z ---
# Region: x^2+y^2+z^2 <= R^2, z >= 0
# In spherical: rho*cos(phi) * rho^2*sin(phi)

R = 2.0

def integrand_sph(rho, phi, theta):
    """Density z = rho*cos(phi) times volume element rho^2*sin(phi)."""
    return rho * np.cos(phi) * rho**2 * np.sin(phi)

mass, error = tplquad(
    integrand_sph,
    0, 2 * np.pi,                 # theta
    lambda theta: 0,               # phi lower
    lambda theta: np.pi / 2,       # phi upper (hemisphere z >= 0)
    lambda theta, phi: 0,          # rho lower
    lambda theta, phi: R           # rho upper
)

print(f"\nMass of hemisphere (density=z, R={R}): {mass:.6f}")
print(f"Analytical (pi*R^4/4):                 {np.pi * R**4 / 4:.6f}")
```

---

## 5. Jacobian and Change of Variables

### 5.1 The General Formula

For a transformation $(x, y) = T(u, v)$, meaning $x = x(u, v)$ and $y = y(u, v)$:

$$\iint_R f(x, y)\, dx\, dy = \iint_S f(x(u,v),\, y(u,v))\, |J|\, du\, dv$$

where $J$ is the **Jacobian determinant**:

$$J = \frac{\partial(x, y)}{\partial(u, v)} = \begin{vmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{vmatrix}$$

**The Jacobian measures how area distorts** under the transformation. If $|J| = 2$, a unit square in $(u, v)$ space maps to a region with area 2 in $(x, y)$ space.

### 5.2 Polar Coordinates as a Special Case

For polar coordinates $x = r\cos\theta$, $y = r\sin\theta$:

$$J = \begin{vmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{vmatrix} = r\cos^2\theta + r\sin^2\theta = r$$

This confirms $dA = |J|\, dr\, d\theta = r\, dr\, d\theta$.

### 5.3 Triple Integral Jacobians

For transformations $(x, y, z) = T(u, v, w)$:

$$J = \frac{\partial(x, y, z)}{\partial(u, v, w)} = \begin{vmatrix} x_u & x_v & x_w \\ y_u & y_v & y_w \\ z_u & z_v & z_w \end{vmatrix}$$

**Spherical coordinates:** $J = \rho^2\sin\phi$ (as used above).

**Cylindrical coordinates:** $J = r$ (same factor as polar).

```python
import numpy as np
from sympy import symbols, cos, sin, Matrix, simplify

# --- Verify Jacobians symbolically ---
r, theta, phi, rho = symbols('r theta phi rho', positive=True)

# Polar coordinates
x_pol = r * cos(theta)
y_pol = r * sin(theta)
J_polar = Matrix([
    [x_pol.diff(r), x_pol.diff(theta)],
    [y_pol.diff(r), y_pol.diff(theta)]
]).det()
print(f"Jacobian (polar): {simplify(J_polar)}")  # r

# Spherical coordinates
x_sph = rho * sin(phi) * cos(theta)
y_sph = rho * sin(phi) * sin(theta)
z_sph = rho * cos(phi)
J_sph = Matrix([
    [x_sph.diff(rho), x_sph.diff(phi), x_sph.diff(theta)],
    [y_sph.diff(rho), y_sph.diff(phi), y_sph.diff(theta)],
    [z_sph.diff(rho), z_sph.diff(phi), z_sph.diff(theta)]
]).det()
print(f"Jacobian (spherical): {simplify(J_sph)}")  # rho^2*sin(phi)

# Cylindrical coordinates
z_var = symbols('z')
x_cyl = r * cos(theta)
y_cyl = r * sin(theta)
z_cyl = z_var
J_cyl = Matrix([
    [x_cyl.diff(r), x_cyl.diff(theta), x_cyl.diff(z_var)],
    [y_cyl.diff(r), y_cyl.diff(theta), y_cyl.diff(z_var)],
    [z_cyl.diff(r), z_cyl.diff(theta), z_cyl.diff(z_var)]
]).det()
print(f"Jacobian (cylindrical): {simplify(J_cyl)}")  # r
```

---

## 6. Applications

### 6.1 Mass and Center of Mass

For a lamina (thin plate) occupying region $D$ with density $\rho(x, y)$:

| Quantity | Formula |
|----------|---------|
| Mass | $m = \iint_D \rho(x,y)\, dA$ |
| Center of mass $\bar{x}$ | $\bar{x} = \frac{1}{m}\iint_D x\,\rho(x,y)\, dA$ |
| Center of mass $\bar{y}$ | $\bar{y} = \frac{1}{m}\iint_D y\,\rho(x,y)\, dA$ |

### 6.2 Moments of Inertia

| Quantity | Formula |
|----------|---------|
| $I_x$ (about $x$-axis) | $I_x = \iint_D y^2 \rho(x,y)\, dA$ |
| $I_y$ (about $y$-axis) | $I_y = \iint_D x^2 \rho(x,y)\, dA$ |
| $I_0$ (about origin) | $I_0 = \iint_D (x^2 + y^2) \rho(x,y)\, dA = I_x + I_y$ |

These formulas extend to 3D with triple integrals.

```python
import numpy as np
from scipy.integrate import dblquad

# --- Center of mass of a semicircular lamina ---
# Region: x^2 + y^2 <= R^2, y >= 0
# Density: rho(x, y) = 1 (uniform)

R = 2.0

# Mass
mass, _ = dblquad(
    lambda y, x: 1.0,
    -R, R,
    lambda x: 0,
    lambda x: np.sqrt(max(R**2 - x**2, 0))
)

# First moment about x-axis: integral of y * rho dA
My, _ = dblquad(
    lambda y, x: y,
    -R, R,
    lambda x: 0,
    lambda x: np.sqrt(max(R**2 - x**2, 0))
)

y_bar = My / mass

print(f"Mass of semicircular lamina: {mass:.6f}")
print(f"Analytical mass (pi*R^2/2):  {np.pi * R**2 / 2:.6f}")
print(f"Center of mass y_bar:        {y_bar:.6f}")
print(f"Analytical (4R / 3pi):       {4 * R / (3 * np.pi):.6f}")
print(f"By symmetry, x_bar = 0")
```

---

## 7. Cross-References

- **Mathematical Methods Lesson 06** covers curvilinear coordinates (cylindrical, spherical, general orthogonal) in greater depth, including scale factors and differential operators.
- **Lesson 09 (Multivariable Functions)** introduced partial derivatives and the gradient, which underpin the variable transformations in this lesson.
- **Lesson 11 (Vector Calculus)** extends integration to line and surface integrals.

---

## Practice Problems

**1.** Evaluate $\iint_R (x^2 + y)\, dA$ where $R = [0, 1] \times [0, 2]$. Verify by computing in both orders of integration.

**2.** Evaluate $\int_0^1 \int_{\sqrt{y}}^1 \sin(x^2)\, dx\, dy$ by switching the order of integration.
   (Hint: sketch the region first.)

**3.** Use polar coordinates to evaluate $\iint_D (x^2 + y^2)^{3/2}\, dA$ where $D$ is the annulus $1 \le x^2 + y^2 \le 4$.

**4.** Find the volume of the solid bounded above by the paraboloid $z = 4 - x^2 - y^2$ and below by the plane $z = 0$.
   - (a) Set up and evaluate using Cartesian coordinates.
   - (b) Set up and evaluate using polar coordinates (should be much easier).

**5.** Compute the mass and center of mass of a solid hemisphere $x^2 + y^2 + z^2 \le R^2$, $z \ge 0$, with density $\rho(x, y, z) = z$. Use spherical coordinates. Verify numerically with `scipy.integrate.tplquad`.

---

## References

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapters 15.1-15.9
- **Jerrold E. Marsden & Anthony Tromba**, *Vector Calculus*, 6th Edition, Chapter 5
- **SciPy Integration Documentation**: https://docs.scipy.org/doc/scipy/reference/integrate.html
- **Khan Academy**, "Double and Triple Integrals"

---

[Previous: Multivariable Functions](./09_Multivariable_Functions.md) | [Next: Vector Calculus](./11_Vector_Calculus.md)

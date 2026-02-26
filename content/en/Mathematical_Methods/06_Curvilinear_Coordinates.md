# 06. Curvilinear Coordinates and Multiple Integrals

## Learning Objectives

- Understand the definition and calculation methods of **multiple integrals**, and master the technique of changing the order of integration
- Perform integration variable substitution in coordinate transformations using the **Jacobian**
- Derive and apply coordinate transformations, volume/area elements, and differential operators in **cylindrical coordinates** and **spherical coordinates**
- Understand the general representation of scale factors and differential operators in **general curvilinear coordinate systems**
- Select and apply appropriate coordinate systems to physics problems (moment of inertia, electric field, gravitational field)

---

## 1. Double and Triple Integrals

### 1.1 Definition and Calculation of Double Integrals

A double integral integrates a function $f(x, y)$ over a two-dimensional region $R$:

$$\iint_R f(x, y) \, dA = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i, y_i) \, \Delta A_i$$

In rectangular coordinates, the area element is $dA = dx \, dy$, so we calculate using iterated integrals:

$$\iint_R f(x, y) \, dA = \int_a^b \left[ \int_{g_1(x)}^{g_2(x)} f(x, y) \, dy \right] dx$$

where $a \le x \le b$, and for each $x$, $g_1(x) \le y \le g_2(x)$.

**Example**: Double integral of $f(x, y) = x + y$ over the triangular region $0 \le x \le 1$, $0 \le y \le x$

$$\int_0^1 \int_0^x (x + y) \, dy \, dx = \int_0^1 \left[ xy + \frac{y^2}{2} \right]_0^x dx = \int_0^1 \frac{3x^2}{2} \, dx = \frac{1}{2}$$

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- Numerical double integral computation ---
# f(x, y) = x + y, region: 0 <= x <= 1, 0 <= y <= x
f = lambda y, x: x + y
result, error = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: x)
print(f"Double integral result: {result:.6f} (error: {error:.2e})")
# Output: Double integral result: 0.500000 (error: 5.55e-15)

# --- Visualize integration region ---
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
triangle = Polygon([[0, 0], [1, 0], [1, 1]], alpha=0.3, color='steelblue')
ax.add_patch(triangle)
ax.set_xlim(-0.1, 1.3)
ax.set_ylim(-0.1, 1.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Integration region: 0 ≤ y ≤ x, 0 ≤ x ≤ 1')
ax.set_aspect('equal')
ax.plot([0, 1], [0, 1], 'r-', linewidth=2, label='y = x')
ax.legend()
plt.tight_layout()
plt.savefig('double_integral_region.png', dpi=150)
plt.show()
```

### 1.2 Triple Integrals

A triple integral integrates a function over a three-dimensional region $V$:

$$\iiint_V f(x, y, z) \, dV = \int_a^b \int_{g_1(x)}^{g_2(x)} \int_{h_1(x,y)}^{h_2(x,y)} f(x, y, z) \, dz \, dy \, dx$$

In rectangular coordinates, the volume element is $dV = dx \, dy \, dz$.

**Example**: Triple integral of $f = 1$ inside the unit sphere $x^2 + y^2 + z^2 \le 1$ (volume of a sphere)

```python
# Volume of unit sphere: triple integral (Cartesian coordinates)
def sphere_volume_cartesian():
    f = lambda z, y, x: 1.0
    x_lo, x_hi = -1, 1
    y_lo = lambda x: -np.sqrt(1 - x**2)
    y_hi = lambda x:  np.sqrt(1 - x**2)
    z_lo = lambda x, y: -np.sqrt(max(0, 1 - x**2 - y**2))
    z_hi = lambda x, y:  np.sqrt(max(0, 1 - x**2 - y**2))

    result, error = integrate.tplquad(f, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    return result

V = sphere_volume_cartesian()
print(f"Volume of unit sphere (numerical): {V:.6f}")
print(f"Analytical result (4π/3):          {4*np.pi/3:.6f}")
```

### 1.3 Changing the Order of Integration

Changing the order of integration in a double integral can greatly simplify the calculation. The key is to **re-describe the boundaries of the integration region** according to the new order.

**Example**: $\int_0^1 \int_y^1 e^{x^2} \, dx \, dy$

The integral in the $x$ direction comes first, but the indefinite integral of $e^{x^2}$ cannot be expressed as an elementary function. If we change the order:

- Original region: $0 \le y \le 1$, $y \le x \le 1$ (triangle: $0 \le y \le x \le 1$)
- After change: $0 \le x \le 1$, $0 \le y \le x$

$$\int_0^1 \int_0^x e^{x^2} \, dy \, dx = \int_0^1 x \, e^{x^2} \, dx = \frac{1}{2}(e - 1)$$

```python
import sympy as sp

x, y = sp.symbols('x y', positive=True)

# Integrate after changing order
inner = sp.integrate(sp.exp(x**2), (y, 0, x))   # ∫₀ˣ e^{x²} dy = x·e^{x²}
result = sp.integrate(inner, (x, 0, 1))           # ∫₀¹ x·e^{x²} dx
print(f"Result after changing order: {result}")
# Output: -1/2 + E/2  i.e., (e-1)/2
print(f"Numerical value: {float(result):.6f}")
```

---

## 2. Coordinate Transformations and Jacobian

### 2.1 Jacobian Determinant

For a coordinate transformation $(x, y) \to (u, v)$, where $x = x(u, v)$ and $y = y(u, v)$, the **Jacobian** is:

$$J = \frac{\partial(x, y)}{\partial(u, v)} = \begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} \\[8pt] \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} \end{vmatrix}$$

The Jacobian represents the **area (or volume) scaling ratio** due to the coordinate transformation.

In three dimensions:

$$J = \frac{\partial(x, y, z)}{\partial(u, v, w)} = \begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} & \dfrac{\partial x}{\partial w} \\[6pt] \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} & \dfrac{\partial y}{\partial w} \\[6pt] \dfrac{\partial z}{\partial u} & \dfrac{\partial z}{\partial v} & \dfrac{\partial z}{\partial w} \end{vmatrix}$$

### 2.2 General Coordinate Transformation Formula

The transformation formula for a double integral under the coordinate transformation $(u, v) \to (x, y)$:

$$\iint_R f(x, y) \, dx \, dy = \iint_{R'} f(x(u,v), y(u,v)) \, |J| \, du \, dv$$

where $|J|$ is the **absolute value** of the Jacobian.

### 2.3 Integration Using Variable Substitution

```python
import sympy as sp

u, v = sp.symbols('u v', positive=True)

# Example: polar coordinate transformation x = r cos(θ), y = r sin(θ)
r, theta = sp.symbols('r theta', positive=True)
x_expr = r * sp.cos(theta)
y_expr = r * sp.sin(theta)

# Compute Jacobian
J = sp.Matrix([
    [sp.diff(x_expr, r), sp.diff(x_expr, theta)],
    [sp.diff(y_expr, r), sp.diff(y_expr, theta)]
])
jacobian_det = J.det().simplify()
print(f"Polar coordinate Jacobian: J = {jacobian_det}")
# Output: Polar coordinate Jacobian: J = r

# Print Jacobian matrix
print(f"\nJacobian matrix:")
sp.pprint(J)

# --- Elliptic coordinate transformation example ---
# x = a·u·cos(v), y = b·u·sin(v)
a, b = sp.symbols('a b', positive=True)
x_ellip = a * u * sp.cos(v)
y_ellip = b * u * sp.sin(v)

J_ellip = sp.Matrix([
    [sp.diff(x_ellip, u), sp.diff(x_ellip, v)],
    [sp.diff(y_ellip, u), sp.diff(y_ellip, v)]
])
det_ellip = J_ellip.det().simplify()
print(f"\nElliptic coordinate Jacobian: J = {det_ellip}")
# Output: Elliptic coordinate Jacobian: J = a*b*u
```

---

## 3. Cylindrical Coordinates

### 3.1 Coordinate Definition and Transformation

Cylindrical coordinates $(\rho, \phi, z)$ add a $z$ axis to 2D polar coordinates:

$$x = \rho \cos\phi, \quad y = \rho \sin\phi, \quad z = z$$

Inverse transformation:

$$\rho = \sqrt{x^2 + y^2}, \quad \phi = \arctan\left(\frac{y}{x}\right), \quad z = z$$

Range: $\rho \ge 0$, $0 \le \phi < 2\pi$, $-\infty < z < \infty$

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cylindrical_to_cartesian(rho, phi, z):
    """Cylindrical coordinates → Cartesian coordinates"""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

def cartesian_to_cylindrical(x, y, z):
    """Cartesian coordinates → Cylindrical coordinates"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z

# --- Visualize cylindrical coordinate system ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ρ = const surface (cylinder)
phi_grid = np.linspace(0, 2*np.pi, 50)
z_grid = np.linspace(-2, 2, 20)
PHI, Z = np.meshgrid(phi_grid, z_grid)
for rho_val in [0.5, 1.0, 1.5]:
    X = rho_val * np.cos(PHI)
    Y = rho_val * np.sin(PHI)
    ax.plot_surface(X, Y, Z, alpha=0.15, color='blue')

# φ = const surface (half-plane)
rho_grid = np.linspace(0, 2, 20)
RHO, Z2 = np.meshgrid(rho_grid, z_grid)
for phi_val in [0, np.pi/3, 2*np.pi/3, np.pi]:
    X2 = RHO * np.cos(phi_val)
    Y2 = RHO * np.sin(phi_val)
    ax.plot_surface(X2, Y2, Z2, alpha=0.1, color='red')

# z = const surface (horizontal plane)
RHO3, PHI3 = np.meshgrid(rho_grid, phi_grid)
X3 = RHO3 * np.cos(PHI3)
Y3 = RHO3 * np.sin(PHI3)
for z_val in [-1, 0, 1]:
    Z3 = np.full_like(X3, z_val)
    ax.plot_surface(X3, Y3, Z3, alpha=0.1, color='green')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Cylindrical coordinate surfaces: ρ=const(blue), φ=const(red), z=const(green)')
plt.tight_layout()
plt.savefig('cylindrical_coords.png', dpi=150)
plt.show()
```

### 3.2 Volume and Area Elements

The Jacobian in cylindrical coordinates:

$$J = \frac{\partial(x, y, z)}{\partial(\rho, \phi, z)} = \begin{vmatrix} \cos\phi & -\rho\sin\phi & 0 \\ \sin\phi & \rho\cos\phi & 0 \\ 0 & 0 & 1 \end{vmatrix} = \rho$$

Therefore, the **volume element** is:

$$dV = \rho \, d\rho \, d\phi \, dz$$

**Area elements**:
- $\rho = \text{const}$ surface (cylindrical side): $dA = \rho \, d\phi \, dz$
- $z = \text{const}$ surface (horizontal plane): $dA = \rho \, d\rho \, d\phi$
- $\phi = \text{const}$ surface (half-plane): $dA = d\rho \, dz$

**Example**: Volume of a cylinder with radius $R$ and height $H$

$$V = \int_0^H \int_0^{2\pi} \int_0^R \rho \, d\rho \, d\phi \, dz = \pi R^2 H$$

```python
import sympy as sp

rho, phi, z = sp.symbols('rho phi z', positive=True)
R, H = sp.symbols('R H', positive=True)

# Volume of cylinder
V = sp.integrate(rho, (rho, 0, R), (phi, 0, 2*sp.pi), (z, 0, H))
print(f"Volume of cylinder: V = {V}")
# Output: V = π·R²·H

# Verify Jacobian
x_cyl = rho * sp.cos(phi)
y_cyl = rho * sp.sin(phi)
z_cyl = z

J_cyl = sp.Matrix([
    [sp.diff(x_cyl, rho), sp.diff(x_cyl, phi), sp.diff(x_cyl, z)],
    [sp.diff(y_cyl, rho), sp.diff(y_cyl, phi), sp.diff(y_cyl, z)],
    [sp.diff(z_cyl, rho), sp.diff(z_cyl, phi), sp.diff(z_cyl, z)]
])
print(f"Jacobian det = {J_cyl.det().simplify()}")
# Output: Jacobian det = rho
```

### 3.3 Gradient, Divergence, and Curl in Cylindrical Coordinates

In cylindrical coordinates, we use unit vectors $\hat{\boldsymbol{\rho}}$, $\hat{\boldsymbol{\phi}}$, $\hat{\mathbf{z}}$.

**Gradient**:

$$\nabla f = \frac{\partial f}{\partial \rho}\hat{\boldsymbol{\rho}} + \frac{1}{\rho}\frac{\partial f}{\partial \phi}\hat{\boldsymbol{\phi}} + \frac{\partial f}{\partial z}\hat{\mathbf{z}}$$

**Divergence**:

$$\nabla \cdot \mathbf{F} = \frac{1}{\rho}\frac{\partial}{\partial \rho}(\rho F_\rho) + \frac{1}{\rho}\frac{\partial F_\phi}{\partial \phi} + \frac{\partial F_z}{\partial z}$$

**Curl**:

$$\nabla \times \mathbf{F} = \left(\frac{1}{\rho}\frac{\partial F_z}{\partial \phi} - \frac{\partial F_\phi}{\partial z}\right)\hat{\boldsymbol{\rho}} + \left(\frac{\partial F_\rho}{\partial z} - \frac{\partial F_z}{\partial \rho}\right)\hat{\boldsymbol{\phi}} + \frac{1}{\rho}\left(\frac{\partial}{\partial \rho}(\rho F_\phi) - \frac{\partial F_\rho}{\partial \phi}\right)\hat{\mathbf{z}}$$

**Laplacian**:

$$\nabla^2 f = \frac{1}{\rho}\frac{\partial}{\partial \rho}\left(\rho \frac{\partial f}{\partial \rho}\right) + \frac{1}{\rho^2}\frac{\partial^2 f}{\partial \phi^2} + \frac{\partial^2 f}{\partial z^2}$$

### 3.4 Application Example

**Example**: Magnetic field of an infinitely long straight wire (current $I$)

By Ampère's law, $\mathbf{B} = \frac{\mu_0 I}{2\pi \rho}\hat{\boldsymbol{\phi}}$. We verify the divergence and curl.

```python
import sympy as sp
from sympy.vector import CoordSys3D

# SymPy vector system is Cartesian-based; implement cylindrical differential operators directly
rho, phi, z = sp.symbols('rho phi z', positive=True)
mu_0, I = sp.symbols('mu_0 I', positive=True)

# B = (μ₀I / 2πρ) φ̂  →  B_rho = 0, B_phi = μ₀I/(2πρ), B_z = 0
B_rho = 0
B_phi = mu_0 * I / (2 * sp.pi * rho)
B_z = 0

# Divergence (cylindrical coordinates)
div_B = (1/rho) * sp.diff(rho * B_rho, rho) + \
        (1/rho) * sp.diff(B_phi, phi) + \
        sp.diff(B_z, z)
print(f"∇·B = {sp.simplify(div_B)}")
# Output: ∇·B = 0  (Maxwell's equation ∇·B = 0 satisfied)

# Curl (cylindrical coordinates) - only z component (others are 0)
curl_B_z = (1/rho) * (sp.diff(rho * B_phi, rho) - sp.diff(B_rho, phi))
print(f"(∇×B)_z = {sp.simplify(curl_B_z)}")
# 0 for ρ ≠ 0 (no current outside the wire)
```

---

## 4. Spherical Coordinates

### 4.1 Coordinate Definition and Transformation

Spherical coordinates $(r, \theta, \phi)$:

$$x = r \sin\theta \cos\phi, \quad y = r \sin\theta \sin\phi, \quad z = r \cos\theta$$

Inverse transformation:

$$r = \sqrt{x^2 + y^2 + z^2}, \quad \theta = \arccos\left(\frac{z}{r}\right), \quad \phi = \arctan\left(\frac{y}{x}\right)$$

Range: $r \ge 0$, $0 \le \theta \le \pi$ (polar angle), $0 \le \phi < 2\pi$ (azimuthal angle)

> **Note**: In physics, the convention is to use $\theta$ for the polar angle and $\phi$ for the azimuthal angle. In mathematics, the opposite convention is sometimes used.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(r, theta, phi):
    """Spherical coordinates → Cartesian coordinates"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    """Cartesian coordinates → Spherical coordinates"""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / np.where(r > 0, r, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi

# --- Visualize spherical coordinate surfaces ---
fig = plt.figure(figsize=(12, 5))

# (a) r = const (sphere)
ax1 = fig.add_subplot(131, projection='3d')
theta_g = np.linspace(0, np.pi, 30)
phi_g = np.linspace(0, 2*np.pi, 30)
THETA, PHI = np.meshgrid(theta_g, phi_g)
for r_val in [0.5, 1.0, 1.5]:
    X = r_val * np.sin(THETA) * np.cos(PHI)
    Y = r_val * np.sin(THETA) * np.sin(PHI)
    Z = r_val * np.cos(THETA)
    ax1.plot_surface(X, Y, Z, alpha=0.2, color='blue')
ax1.set_title('r = const (sphere)')

# (b) θ = const (cone)
ax2 = fig.add_subplot(132, projection='3d')
r_g = np.linspace(0, 2, 20)
R_grid, PHI2 = np.meshgrid(r_g, phi_g)
for theta_val in [np.pi/6, np.pi/3, np.pi/2]:
    X2 = R_grid * np.sin(theta_val) * np.cos(PHI2)
    Y2 = R_grid * np.sin(theta_val) * np.sin(PHI2)
    Z2 = R_grid * np.cos(theta_val)
    ax2.plot_surface(X2, Y2, Z2, alpha=0.2, color='red')
ax2.set_title('θ = const (cone)')

# (c) φ = const (half-plane)
ax3 = fig.add_subplot(133, projection='3d')
R3, THETA3 = np.meshgrid(r_g, theta_g)
for phi_val in [0, np.pi/3, 2*np.pi/3, np.pi]:
    X3 = R3 * np.sin(THETA3) * np.cos(phi_val)
    Y3 = R3 * np.sin(THETA3) * np.sin(phi_val)
    Z3 = R3 * np.cos(THETA3)
    ax3.plot_surface(X3, Y3, Z3, alpha=0.2, color='green')
ax3.set_title('φ = const (half-plane)')

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plt.suptitle('Spherical coordinate surfaces', fontsize=14)
plt.tight_layout()
plt.savefig('spherical_coords.png', dpi=150)
plt.show()
```

### 4.2 Volume and Area Elements

The Jacobian in spherical coordinates:

$$J = \frac{\partial(x, y, z)}{\partial(r, \theta, \phi)} = r^2 \sin\theta$$

> **Derivation**: Calculate the determinant of the $3 \times 3$ Jacobian matrix directly, or understand geometrically as the infinitesimal volume element $dr \cdot (r \, d\theta) \cdot (r\sin\theta \, d\phi)$.

Therefore, the **volume element**:

$$dV = r^2 \sin\theta \, dr \, d\theta \, d\phi$$

**Area elements**:
- $r = \text{const}$ surface (sphere): $dA = r^2 \sin\theta \, d\theta \, d\phi$
- $\theta = \text{const}$ surface (cone): $dA = r \sin\theta \, dr \, d\phi$
- $\phi = \text{const}$ surface (half-plane): $dA = r \, dr \, d\theta$

**Example**: Volume and surface area of a sphere

```python
import sympy as sp

r, theta, phi = sp.symbols('r theta phi', positive=True)
R = sp.Symbol('R', positive=True)

# Verify Jacobian
x_sph = r * sp.sin(theta) * sp.cos(phi)
y_sph = r * sp.sin(theta) * sp.sin(phi)
z_sph = r * sp.cos(theta)

J_sph = sp.Matrix([
    [sp.diff(x_sph, r), sp.diff(x_sph, theta), sp.diff(x_sph, phi)],
    [sp.diff(y_sph, r), sp.diff(y_sph, theta), sp.diff(y_sph, phi)],
    [sp.diff(z_sph, r), sp.diff(z_sph, theta), sp.diff(z_sph, phi)]
])
det_J = sp.trigsimp(J_sph.det())
print(f"Spherical coordinate Jacobian: {det_J}")
# Output: r**2*sin(theta)

# Volume of sphere
V = sp.integrate(r**2 * sp.sin(theta), (r, 0, R), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
print(f"Volume of sphere: V = {V}")
# Output: 4*pi*R**3/3

# Surface area of sphere (r = R fixed)
S = sp.integrate(R**2 * sp.sin(theta), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
print(f"Surface area of sphere: S = {S}")
# Output: 4*pi*R**2
```

### 4.3 Gradient, Divergence, and Curl in Spherical Coordinates

**Gradient**:

$$\nabla f = \frac{\partial f}{\partial r}\hat{\mathbf{r}} + \frac{1}{r}\frac{\partial f}{\partial \theta}\hat{\boldsymbol{\theta}} + \frac{1}{r\sin\theta}\frac{\partial f}{\partial \phi}\hat{\boldsymbol{\phi}}$$

**Divergence**:

$$\nabla \cdot \mathbf{F} = \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 F_r) + \frac{1}{r\sin\theta}\frac{\partial}{\partial \theta}(\sin\theta \, F_\theta) + \frac{1}{r\sin\theta}\frac{\partial F_\phi}{\partial \phi}$$

**Curl**:

$$\nabla \times \mathbf{F} = \frac{1}{r\sin\theta}\left[\frac{\partial}{\partial \theta}(\sin\theta \, F_\phi) - \frac{\partial F_\theta}{\partial \phi}\right]\hat{\mathbf{r}} + \frac{1}{r}\left[\frac{1}{\sin\theta}\frac{\partial F_r}{\partial \phi} - \frac{\partial}{\partial r}(r F_\phi)\right]\hat{\boldsymbol{\theta}} + \frac{1}{r}\left[\frac{\partial}{\partial r}(r F_\theta) - \frac{\partial F_r}{\partial \theta}\right]\hat{\boldsymbol{\phi}}$$

**Laplacian**:

$$\nabla^2 f = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 \frac{\partial f}{\partial r}\right) + \frac{1}{r^2 \sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta \frac{\partial f}{\partial \theta}\right) + \frac{1}{r^2 \sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}$$

### 4.4 Application Example

**Example**: Laplacian of the Coulomb potential $\Phi = \frac{q}{4\pi\epsilon_0 r}$

```python
import sympy as sp

r, theta, phi = sp.symbols('r theta phi', positive=True)
q, eps0 = sp.symbols('q epsilon_0', positive=True)

# Coulomb potential
Phi = q / (4 * sp.pi * eps0 * r)

# Laplacian in spherical coordinates (r > 0 region)
laplacian_Phi = (1/r**2) * sp.diff(r**2 * sp.diff(Phi, r), r) + \
                (1/(r**2 * sp.sin(theta))) * sp.diff(sp.sin(theta) * sp.diff(Phi, theta), theta) + \
                (1/(r**2 * sp.sin(theta)**2)) * sp.diff(Phi, phi, 2)

result = sp.simplify(laplacian_Phi)
print(f"∇²Φ = {result}  (r > 0)")
# Output: 0  (satisfies Laplace's equation outside the origin)
# At the origin: ∇²(1/r) = -4πδ(r) (Dirac delta function)

# --- Verify divergence theorem in spherical coordinates ---
# E = -∇Φ = (q/4πε₀r²) r̂
E_r = q / (4 * sp.pi * eps0 * r**2)

# Divergence (only r component due to spherical symmetry)
div_E = (1/r**2) * sp.diff(r**2 * E_r, r)
print(f"∇·E = {sp.simplify(div_E)}  (r > 0)")
# Output: 0 (no charge in this region)
```

**Example**: Solid angle integration in spherical coordinates

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Solid angle element dΩ = sin(θ) dθ dφ
# Total solid angle of sphere: ∫∫ sin(θ) dθ dφ = 4π steradians

# Solid angle subtended by cone (θ ≤ α)
alpha_values = np.linspace(0, np.pi, 100)
solid_angles = 2 * np.pi * (1 - np.cos(alpha_values))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Solid angle vs half-apex angle
axes[0].plot(np.degrees(alpha_values), solid_angles, 'b-', linewidth=2)
axes[0].axhline(y=4*np.pi, color='r', linestyle='--', label='Full sphere = 4π sr')
axes[0].axhline(y=2*np.pi, color='g', linestyle='--', label='Hemisphere = 2π sr')
axes[0].set_xlabel('Half-apex angle α (degrees)')
axes[0].set_ylabel('Solid angle Ω (sr)')
axes[0].set_title('Solid angle of cone')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) Visualize area element on sphere
ax2 = fig.add_subplot(122, projection='3d')
theta_g = np.linspace(0, np.pi, 40)
phi_g = np.linspace(0, 2*np.pi, 40)
THETA, PHI = np.meshgrid(theta_g, phi_g)
X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

# Color proportional to sin(θ) to represent area element density
colors = plt.cm.viridis(np.sin(THETA) / np.sin(THETA).max())
ax2.plot_surface(X, Y, Z, facecolors=colors, alpha=0.7)
ax2.set_title('Area element density ∝ sinθ\n(brighter near equator)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

plt.tight_layout()
plt.savefig('solid_angle.png', dpi=150)
plt.show()
```

---

## 5. General Curvilinear Coordinates

### 5.1 Scale Factors

For a general curvilinear coordinate system $(q_1, q_2, q_3)$ and rectangular coordinates $(x, y, z)$ related by $\mathbf{r} = \mathbf{r}(q_1, q_2, q_3)$, the **scale factor** $h_i$ is:

$$h_i = \left|\frac{\partial \mathbf{r}}{\partial q_i}\right|$$

The scale factor represents how much the actual distance changes when the coordinate $q_i$ changes by one unit.

**Infinitesimal displacement**:

$$d\mathbf{r} = h_1 \, dq_1 \, \hat{\mathbf{e}}_1 + h_2 \, dq_2 \, \hat{\mathbf{e}}_2 + h_3 \, dq_3 \, \hat{\mathbf{e}}_3$$

**Volume element**:

$$dV = h_1 h_2 h_3 \, dq_1 \, dq_2 \, dq_3$$

| Coordinate System | $(q_1, q_2, q_3)$ | $(h_1, h_2, h_3)$ |
|--------|-------------------|-------------------|
| Cartesian | $(x, y, z)$ | $(1, 1, 1)$ |
| Cylindrical | $(\rho, \phi, z)$ | $(1, \rho, 1)$ |
| Spherical | $(r, \theta, \phi)$ | $(1, r, r\sin\theta)$ |

```python
import sympy as sp

# --- Scale factor computation function ---
def compute_scale_factors(coords, transform):
    """
    Compute scale factors for a curvilinear coordinate system.

    Parameters:
        coords: list of curvilinear coordinate variables [q1, q2, q3]
        transform: Cartesian coordinates [x(q), y(q), z(q)]

    Returns:
        scale factors [h1, h2, h3]
    """
    r = sp.Matrix(transform)
    scale_factors = []
    for q in coords:
        dr_dq = r.diff(q)
        h = sp.sqrt(dr_dq.dot(dr_dq)).simplify()
        # Simplify trigonometric expressions
        h = sp.trigsimp(h)
        scale_factors.append(h)
    return scale_factors

# Cylindrical coordinates
rho, phi, z = sp.symbols('rho phi z', positive=True)
h_cyl = compute_scale_factors(
    [rho, phi, z],
    [rho * sp.cos(phi), rho * sp.sin(phi), z]
)
print(f"Cylindrical coordinate scale factors: h_ρ={h_cyl[0]}, h_φ={h_cyl[1]}, h_z={h_cyl[2]}")
# Output: h_ρ=1, h_φ=rho, h_z=1

# Spherical coordinates
r, theta = sp.symbols('r theta', positive=True)
h_sph = compute_scale_factors(
    [r, theta, phi],
    [r * sp.sin(theta) * sp.cos(phi),
     r * sp.sin(theta) * sp.sin(phi),
     r * sp.cos(theta)]
)
print(f"Spherical coordinate scale factors: h_r={h_sph[0]}, h_θ={h_sph[1]}, h_φ={h_sph[2]}")
# Output: h_r=1, h_θ=r, h_φ=r*sin(theta)

# Parabolic cylindrical coordinates: x = (u²-v²)/2, y = uv, z = z
u, v = sp.symbols('u v', positive=True)
h_parab = compute_scale_factors(
    [u, v, z],
    [(u**2 - v**2)/2, u*v, z]
)
print(f"Parabolic cylindrical coordinate scale factors: h_u={h_parab[0]}, h_v={h_parab[1]}, h_z={h_parab[2]}")
# Output: h_u=sqrt(u²+v²), h_v=sqrt(u²+v²), h_z=1
```

### 5.2 General Differential Operators

General expressions for differential operators in orthogonal curvilinear coordinates:

**Gradient**:

$$\nabla f = \frac{1}{h_1}\frac{\partial f}{\partial q_1}\hat{\mathbf{e}}_1 + \frac{1}{h_2}\frac{\partial f}{\partial q_2}\hat{\mathbf{e}}_2 + \frac{1}{h_3}\frac{\partial f}{\partial q_3}\hat{\mathbf{e}}_3$$

**Divergence**:

$$\nabla \cdot \mathbf{F} = \frac{1}{h_1 h_2 h_3}\left[\frac{\partial}{\partial q_1}(h_2 h_3 F_1) + \frac{\partial}{\partial q_2}(h_1 h_3 F_2) + \frac{\partial}{\partial q_3}(h_1 h_2 F_3)\right]$$

**Curl**:

$$\nabla \times \mathbf{F} = \frac{1}{h_1 h_2 h_3}\begin{vmatrix} h_1\hat{\mathbf{e}}_1 & h_2\hat{\mathbf{e}}_2 & h_3\hat{\mathbf{e}}_3 \\[4pt] \dfrac{\partial}{\partial q_1} & \dfrac{\partial}{\partial q_2} & \dfrac{\partial}{\partial q_3} \\[4pt] h_1 F_1 & h_2 F_2 & h_3 F_3 \end{vmatrix}$$

**Laplacian**:

$$\nabla^2 f = \frac{1}{h_1 h_2 h_3}\left[\frac{\partial}{\partial q_1}\left(\frac{h_2 h_3}{h_1}\frac{\partial f}{\partial q_1}\right) + \frac{\partial}{\partial q_2}\left(\frac{h_1 h_3}{h_2}\frac{\partial f}{\partial q_2}\right) + \frac{\partial}{\partial q_3}\left(\frac{h_1 h_2}{h_3}\frac{\partial f}{\partial q_3}\right)\right]$$

```python
import sympy as sp

def laplacian_curvilinear(f, coords, scale_factors):
    """
    Compute Laplacian in a general orthogonal curvilinear coordinate system.

    Parameters:
        f: scalar function
        coords: [q1, q2, q3]
        scale_factors: [h1, h2, h3]
    """
    q1, q2, q3 = coords
    h1, h2, h3 = scale_factors

    term1 = sp.diff((h2*h3/h1) * sp.diff(f, q1), q1)
    term2 = sp.diff((h1*h3/h2) * sp.diff(f, q2), q2)
    term3 = sp.diff((h1*h2/h3) * sp.diff(f, q3), q3)

    return sp.simplify((term1 + term2 + term3) / (h1 * h2 * h3))

# Verify Laplacian in spherical coordinates: f = 1/r
r, theta, phi = sp.symbols('r theta phi', positive=True)
f = 1 / r

lap_f = laplacian_curvilinear(
    f,
    [r, theta, phi],
    [1, r, r * sp.sin(theta)]
)
print(f"∇²(1/r) = {lap_f}  (r > 0)")
# Output: 0 (excluding origin)

# Verify Laplacian: f = r² cos(θ) = rz → ∇²f = 0 (harmonic function)
f2 = r**2 * sp.cos(theta)
lap_f2 = laplacian_curvilinear(
    f2,
    [r, theta, phi],
    [1, r, r * sp.sin(theta)]
)
print(f"∇²(r²cosθ) = {lap_f2}")
# ∇²(r²cosθ) should be 0 (but actually this is not harmonic)
# Correct harmonic function: r cos(θ) = z
f3 = r * sp.cos(theta)
lap_f3 = laplacian_curvilinear(
    f3,
    [r, theta, phi],
    [1, r, r * sp.sin(theta)]
)
print(f"∇²(r·cosθ) = {lap_f3}")
# Output: 0 (z = r·cosθ is a harmonic function)
```

### 5.3 Orthogonal Curvilinear Coordinates

The condition for orthogonal curvilinear coordinates: coordinate surfaces are mutually orthogonal. Mathematically:

$$\frac{\partial \mathbf{r}}{\partial q_i} \cdot \frac{\partial \mathbf{r}}{\partial q_j} = 0 \quad (i \ne j)$$

When this condition is satisfied, the metric tensor becomes a diagonal matrix, greatly simplifying the differential operators.

**List of major orthogonal coordinate systems**:

| Coordinate System | Variables | Scale Factors | Main Use |
|--------|------|------------|----------|
| Cartesian | $(x,y,z)$ | $(1,1,1)$ | General |
| Cylindrical | $(\rho,\phi,z)$ | $(1,\rho,1)$ | Axisymmetric problems |
| Spherical | $(r,\theta,\phi)$ | $(1,r,r\sin\theta)$ | Spherically symmetric problems |
| Elliptic cylindrical | $(u,v,z)$ | $(\cdot,\cdot,1)$ | Elliptical boundaries |
| Parabolic cylindrical | $(u,v,z)$ | $(\sqrt{u^2+v^2},\sqrt{u^2+v^2},1)$ | Parabolic boundaries |
| Prolate spheroidal | $(\xi,\eta,\phi)$ | Complex | Problems with eccentricity |

---

## 6. Physics Applications

### 6.1 Moment of Inertia Calculation

The moment of inertia of an object characterizes the mass distribution about an axis of rotation:

$$I = \iiint_V \rho(\mathbf{r}) \, d^2 \, dV$$

where $d$ is the distance to the rotation axis and $\rho(\mathbf{r})$ is the mass density.

**Example 1**: Moment of inertia of a solid sphere with uniform density $\rho_0$ and radius $R$ (about the $z$ axis)

In spherical coordinates, the distance to the $z$ axis: $d = r\sin\theta$

$$I_z = \rho_0 \int_0^{2\pi} \int_0^{\pi} \int_0^R (r\sin\theta)^2 \cdot r^2 \sin\theta \, dr \, d\theta \, d\phi$$

```python
import sympy as sp

r, theta, phi = sp.symbols('r theta phi', positive=True)
R, rho0, M = sp.symbols('R rho_0 M', positive=True)

# Moment of inertia about z-axis
integrand = rho0 * (r * sp.sin(theta))**2 * r**2 * sp.sin(theta)
I_z = sp.integrate(integrand, (r, 0, R), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
I_z_simplified = sp.simplify(I_z)
print(f"I_z = {I_z_simplified}")

# Substitute total mass M = (4/3)πR³ρ₀
M_expr = sp.Rational(4, 3) * sp.pi * R**3 * rho0
I_z_M = I_z_simplified.subs(rho0, M / (sp.Rational(4, 3) * sp.pi * R**3))
I_z_final = sp.simplify(I_z_M)
print(f"I_z = {I_z_final}")
# Output: 2*M*R²/5

print(f"\n--- Moments of inertia for various shapes ---")

# Cylinder (radius R, height H, about z-axis)
rho_cyl, z_cyl = sp.symbols('rho z', positive=True)
H = sp.Symbol('H', positive=True)
I_cylinder = sp.integrate(
    rho0 * rho_cyl**2 * rho_cyl,  # d² * dV/dρdφdz where d=ρ
    (rho_cyl, 0, R), (phi, 0, 2*sp.pi), (z_cyl, 0, H)
)
M_cyl = sp.pi * R**2 * H * rho0
I_cyl_M = sp.simplify(I_cylinder.subs(rho0, M / (sp.pi * R**2 * H)))
print(f"Cylinder (z-axis): I = {I_cyl_M}")
# Output: M*R²/2

# Hollow spherical shell (radius R, about z-axis)
# Surface integral: I = ∫ (R sinθ)² σ R² sinθ dθ dφ
sigma = sp.Symbol('sigma', positive=True)
I_shell = sp.integrate(
    sigma * (R * sp.sin(theta))**2 * R**2 * sp.sin(theta),
    (theta, 0, sp.pi), (phi, 0, 2*sp.pi)
)
M_shell = 4 * sp.pi * R**2 * sigma
I_shell_M = sp.simplify(I_shell.subs(sigma, M / (4 * sp.pi * R**2)))
print(f"Spherical shell (z-axis): I = {I_shell_M}")
# Output: 2*M*R²/3
```

### 6.2 Electric Field of Spherically Symmetric Charge Distribution

Gauss's law: $\oint \mathbf{E} \cdot d\mathbf{A} = \frac{Q_{\text{enc}}}{\epsilon_0}$

For spherically symmetric charge distributions, spherical coordinates are a natural choice.

**Example**: Electric field of a sphere with uniform charge density $\rho_e$ and radius $R$

- $r > R$: $E(r) = \frac{Q}{4\pi\epsilon_0 r^2}$ (same as a point charge)
- $r < R$: $E(r) = \frac{\rho_e \, r}{3\epsilon_0} = \frac{Q r}{4\pi\epsilon_0 R^3}$

```python
import numpy as np
import matplotlib.pyplot as plt

# Electric field of a uniformly charged sphere
Q = 1.0        # Total charge (arbitrary units)
R = 1.0        # Sphere radius
eps0 = 1.0     # Permittivity (simplified units)

r = np.linspace(0.01, 3.0, 500)

# Electric field magnitude
E = np.where(
    r < R,
    Q * r / (4 * np.pi * eps0 * R**3),      # Interior: E ∝ r
    Q / (4 * np.pi * eps0 * r**2)            # Exterior: E ∝ 1/r²
)

# Electric potential
V_inside = Q / (8 * np.pi * eps0 * R) * (3 - r**2 / R**2)
V_outside = Q / (4 * np.pi * eps0 * r)
V = np.where(r < R, V_inside, V_outside)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Electric field
axes[0].plot(r/R, E * (4*np.pi*eps0*R**2/Q), 'b-', linewidth=2)
axes[0].axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='r = R')
axes[0].set_xlabel('r / R')
axes[0].set_ylabel('E × (4πε₀R²/Q)')
axes[0].set_title('Electric field of uniformly charged sphere')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].annotate('E ∝ r', xy=(0.5, 0.3), fontsize=12, color='blue')
axes[0].annotate('E ∝ 1/r²', xy=(1.8, 0.25), fontsize=12, color='blue')

# (b) Electric potential
axes[1].plot(r/R, V * (4*np.pi*eps0*R/Q), 'r-', linewidth=2)
axes[1].axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='r = R')
axes[1].set_xlabel('r / R')
axes[1].set_ylabel('V × (4πε₀R/Q)')
axes[1].set_title('Electric potential of uniformly charged sphere')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uniform_sphere_field.png', dpi=150)
plt.show()

# --- Numerical verification of Gauss's law ---
from scipy import integrate

rho_e = 3 * Q / (4 * np.pi * R**3)  # Uniform charge density

# Electric flux through sphere of radius r_test
r_test_values = [0.3, 0.7, 1.0, 1.5, 2.0]

print("Gauss's law verification:")
print(f"{'r/R':>6s}  {'Q_enc':>10s}  {'Φ = Q_enc/ε₀':>14s}  {'E·4πr²':>10s}")
print("-" * 48)

for r_test in r_test_values:
    if r_test <= R:
        Q_enc = rho_e * (4/3) * np.pi * r_test**3
    else:
        Q_enc = Q
    flux = Q_enc / eps0
    E_at_r = Q_enc / (4 * np.pi * eps0 * r_test**2)
    E_times_area = E_at_r * 4 * np.pi * r_test**2
    print(f"{r_test/R:6.2f}  {Q_enc:10.4f}  {flux:14.4f}  {E_times_area:10.4f}")
```

### 6.3 Preview of Spherical Harmonics

When separating variables in Laplace's equation $\nabla^2 f = 0$ in spherical coordinates, the angular part of the solution gives **spherical harmonics** $Y_l^m(\theta, \phi)$:

$$Y_l^m(\theta, \phi) = N_{lm} \, P_l^m(\cos\theta) \, e^{im\phi}$$

where $P_l^m$ is the associated Legendre function and $N_{lm}$ is a normalization constant.

Spherical harmonics are used extensively in quantum mechanics (hydrogen atom orbitals), electromagnetism (multipole expansion), and geophysics (gravitational field modeling).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D

# --- Visualize spherical harmonics ---
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)

fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                          subplot_kw={'projection': '3d'})

harmonics = [
    (0, 0, '$Y_0^0$'),
    (1, 0, '$Y_1^0$'),
    (1, 1, '$Y_1^1$ (real)'),
    (2, 0, '$Y_2^0$'),
    (2, 1, '$Y_2^1$ (real)'),
    (2, 2, '$Y_2^2$ (real)'),
]

for idx, (l, m, title) in enumerate(harmonics):
    ax = axes[idx // 3][idx % 3]

    # Note: scipy sph_harm takes (m, l, φ, θ) order
    Y = sph_harm(m, l, PHI, THETA)

    # Real spherical harmonics
    if m > 0:
        Y_real = np.real(Y) * np.sqrt(2) * (-1)**m
    elif m < 0:
        Y_real = np.imag(Y) * np.sqrt(2) * (-1)**m
    else:
        Y_real = np.real(Y)

    # Use |Y| as radius, sign encoded as color
    R = np.abs(Y_real)
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    colors = np.where(Y_real >= 0, 'steelblue', 'coral')
    # Show positive/negative regions in different colors
    norm = plt.Normalize(vmin=-np.max(np.abs(Y_real)), vmax=np.max(np.abs(Y_real)))
    facecolors = plt.cm.RdBu(norm(Y_real))

    ax.plot_surface(X, Y_coord, Z, facecolors=facecolors, alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_box_aspect([1, 1, 1])
    # Hide axis tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.suptitle('Spherical harmonics $Y_l^m(\\theta, \\phi)$\n(blue: positive, red: negative)', fontsize=16)
plt.tight_layout()
plt.savefig('spherical_harmonics.png', dpi=150)
plt.show()
```

> **Note**: Detailed theory of spherical harmonics will be covered with Legendre functions in [09. Series Solutions and Special Functions](09_Series_Solutions_Special_Functions.md).

---

## Practice Problems

### Problem 1: Jacobian Calculation

Find the Jacobian for the following coordinate transformations:

(a) $x = u^2 - v^2$, $y = 2uv$ (parabolic coordinates)

(b) $x = e^u \cos v$, $y = e^u \sin v$ (logarithmic polar coordinates)

**Hint**: Calculate the determinant of the Jacobian matrix.

### Problem 2: Changing the Order of Integration

Change the order of integration and evaluate:

$$\int_0^4 \int_{\sqrt{y}}^{2} \frac{1}{x^3 + 1} \, dx \, dy$$

**Hint**: The integration region is $0 \le \sqrt{y} \le x \le 2$, which becomes $0 \le y \le x^2$, $0 \le x \le 2$.

### Problem 3: Cylindrical Coordinate Integration

Using cylindrical coordinates, calculate:

(a) The volume of a cone with radius $R$ and height $H$: $z = H(1 - \rho/R)$

(b) If the cone has uniform density $\rho_0$, find the moment of inertia about the $z$ axis through the apex

### Problem 4: Spherical Coordinate Application

Find the center of mass $\bar{z}$ of an object with uniform density $\rho_0$ in the upper hemisphere ($z > 0$) of a sphere with radius $R$.

$$\bar{z} = \frac{\iiint z \, \rho_0 \, dV}{\iiint \rho_0 \, dV}$$

**Answer**: $\bar{z} = 3R/8$

### Problem 5: General Curvilinear Coordinates

The transformation for toroidal coordinates $(\tau, \sigma, \phi)$ is given by:

$$x = \frac{a \sinh\tau \cos\phi}{\cosh\tau - \cos\sigma}, \quad y = \frac{a \sinh\tau \sin\phi}{\cosh\tau - \cos\sigma}, \quad z = \frac{a \sin\sigma}{\cosh\tau - \cos\sigma}$$

Find the scale factors $h_\tau$, $h_\sigma$, $h_\phi$ for this coordinate system (you may use SymPy).

### Problem 6: Comprehensive Physics Application

(a) If a surface charge density $\sigma(\theta) = \sigma_0 \cos\theta$ is distributed on the surface of a sphere with radius $R$, find the total charge.

(b) Find the electric potential at the center of the sphere due to this charge distribution.

**Hint**: In (a), use $\int_0^{\pi} \cos\theta \sin\theta \, d\theta = 0$. This is a dipole distribution.

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 5. Wiley.
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapters 2-3. Academic Press.
3. **Griffiths, D. J.** (2017). *Introduction to Electrodynamics*, 4th ed., Chapter 1 (vector analysis and coordinate systems). Cambridge University Press.

### Key Formula Summary

| Item | Cylindrical $(\rho, \phi, z)$ | Spherical $(r, \theta, \phi)$ |
|------|---------------------------|-------------------------------|
| Scale factors | $(1, \rho, 1)$ | $(1, r, r\sin\theta)$ |
| Volume element | $\rho \, d\rho \, d\phi \, dz$ | $r^2\sin\theta \, dr \, d\theta \, d\phi$ |
| $\nabla f$ (r component) | $\partial f/\partial\rho$ | $\partial f/\partial r$ |
| $\nabla \cdot \mathbf{F}$ | $\frac{1}{\rho}\partial_\rho(\rho F_\rho) + \cdots$ | $\frac{1}{r^2}\partial_r(r^2 F_r) + \cdots$ |
| Jacobian | $\rho$ | $r^2\sin\theta$ |

### Online Resources
1. **MIT OCW 18.02**: Multivariable Calculus (double/triple integrals, coordinate transformations)
2. **Paul's Online Math Notes**: Cylindrical and Spherical Coordinates
3. **Wolfram MathWorld**: Curvilinear Coordinates

---

## Next Lesson

[05. Fourier Series](05_Fourier_Series.md) covers Fourier series, which expands periodic functions as a series of trigonometric functions. It is a fundamental tool for eigenfunction expansion when applying the separation of variables method in coordinate systems.

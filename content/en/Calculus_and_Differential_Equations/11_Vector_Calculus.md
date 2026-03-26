# 11. Vector Calculus

## Learning Objectives

- Describe vector fields and compute divergence and curl
- Evaluate line integrals of scalar and vector fields along curves
- Determine whether a vector field is conservative and find its potential function
- State and apply Green's theorem, Stokes' theorem, and the Divergence theorem
- Verify the fundamental theorems of vector calculus computationally using Python

---

## 1. Vector Fields

A **vector field** assigns a vector to each point in space.

**In 2D:** $\mathbf{F}(x, y) = P(x, y)\,\hat{\mathbf{i}} + Q(x, y)\,\hat{\mathbf{j}}$

**In 3D:** $\mathbf{F}(x, y, z) = P\,\hat{\mathbf{i}} + Q\,\hat{\mathbf{j}} + R\,\hat{\mathbf{k}}$

**Examples from physics:**
- **Gravitational field:** $\mathbf{F} = -\frac{GMm}{r^3}\mathbf{r}$ (points toward the mass)
- **Velocity field:** at each point in a fluid, $\mathbf{v}(x, y, z)$ gives the local velocity
- **Electric field:** $\mathbf{E} = -\nabla V$ (gradient of electric potential)

### 1.1 Divergence and Curl

The **divergence** measures the net "outflow" of a vector field at a point:

$$\text{div}\,\mathbf{F} = \nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$

**Analogy:** Imagine the vector field as fluid flow. Positive divergence means fluid is being created (a source, like a faucet); negative divergence means fluid is being absorbed (a sink, like a drain).

The **curl** measures the rotational tendency:

$$\text{curl}\,\mathbf{F} = \nabla \times \mathbf{F} = \begin{vmatrix} \hat{\mathbf{i}} & \hat{\mathbf{j}} & \hat{\mathbf{k}} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$$

**Analogy:** Place a tiny paddlewheel in the flow. The curl tells you the axis and speed of rotation. If the curl is zero everywhere, the flow has no "swirl."

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Vector field visualization ---
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Source field: F = (x, y) -- divergence = 2, curl = 0
axes[0].quiver(X, Y, X, Y, color='blue', alpha=0.7)
axes[0].set_title('Source: $\\mathbf{F} = (x, y)$\ndiv = 2, curl = 0')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 2. Rotational field: F = (-y, x) -- divergence = 0, curl = 2k
axes[1].quiver(X, Y, -Y, X, color='red', alpha=0.7)
axes[1].set_title('Rotation: $\\mathbf{F} = (-y, x)$\ndiv = 0, curl = 2')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# 3. Saddle field: F = (x, -y) -- divergence = 0, curl = 0
axes[2].quiver(X, Y, X, -Y, color='green', alpha=0.7)
axes[2].set_title('Saddle: $\\mathbf{F} = (x, -y)$\ndiv = 0, curl = 0')
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. Line Integrals

### 2.1 Scalar Line Integrals

The **line integral of a scalar function** $f$ along a curve $C$ parameterized by $\mathbf{r}(t) = (x(t), y(t))$ for $a \le t \le b$:

$$\int_C f\, ds = \int_a^b f(\mathbf{r}(t))\,\|\mathbf{r}'(t)\|\, dt$$

where $ds = \|\mathbf{r}'(t)\|\, dt$ is the arc length element.

**Physical meaning:** If $f(x, y)$ is the linear density of a wire shaped like $C$, then $\int_C f\,ds$ gives the total **mass** of the wire.

### 2.2 Vector Line Integrals (Work)

The **line integral of a vector field** $\mathbf{F}$ along $C$:

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t)\, dt$$

**Physical meaning:** This is the **work** done by force $\mathbf{F}$ as a particle moves along $C$. The dot product $\mathbf{F} \cdot d\mathbf{r}$ picks out the component of force in the direction of motion.

In component form:

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_C P\,dx + Q\,dy$$

**Example:** Find the work done by $\mathbf{F} = (y, x)$ along the parabola $y = x^2$ from $(0, 0)$ to $(1, 1)$.

Parameterize: $x = t$, $y = t^2$, $0 \le t \le 1$. Then $dx = dt$, $dy = 2t\,dt$.

$$W = \int_0^1 (t^2\,dt + t\cdot 2t\,dt) = \int_0^1 3t^2\,dt = 1$$

```python
import numpy as np
from scipy.integrate import quad

# --- Line integral computation ---
# F = (y, x) along y = x^2 from (0,0) to (1,1)

def work_integrand(t):
    """Integrand for F dot r'(t) where r(t) = (t, t^2)."""
    x, y = t, t**2
    dx_dt, dy_dt = 1.0, 2 * t
    P, Q = y, x   # F = (y, x)
    return P * dx_dt + Q * dy_dt  # F dot r'

work, _ = quad(work_integrand, 0, 1)
print(f"Work along parabola: {work:.6f}")  # Should be 1.0

# Compare: work along straight line y = x from (0,0) to (1,1)
def work_line(t):
    """Integrand along the straight line r(t) = (t, t)."""
    x, y = t, t
    P, Q = y, x
    return P * 1.0 + Q * 1.0  # dx/dt = dy/dt = 1

work2, _ = quad(work_line, 0, 1)
print(f"Work along straight line: {work2:.6f}")  # Also 1.0!
# Both give 1.0 -- this field is conservative (spoiler for Section 3)
```

---

## 3. Conservative Fields and Potential Functions

### 3.1 Definition

A vector field $\mathbf{F}$ is **conservative** if there exists a scalar function $\varphi$ (the **potential function**) such that:

$$\mathbf{F} = \nabla\varphi$$

Equivalently, $P = \partial\varphi/\partial x$, $Q = \partial\varphi/\partial y$.

### 3.2 Path Independence

The **Fundamental Theorem for Line Integrals** states: if $\mathbf{F} = \nabla\varphi$, then

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \varphi(\mathbf{r}(b)) - \varphi(\mathbf{r}(a))$$

The integral depends only on the **endpoints**, not the path -- just like the fundamental theorem of calculus.

**Consequence:** For any **closed curve** $C$ (loop), $\oint_C \mathbf{F} \cdot d\mathbf{r} = 0$.

### 3.3 Test for Conservativeness

In 2D, $\mathbf{F} = (P, Q)$ is conservative on a simply connected domain if and only if:

$$\frac{\partial P}{\partial y} = \frac{\partial Q}{\partial x}$$

In 3D, $\mathbf{F} = (P, Q, R)$ is conservative if and only if $\nabla \times \mathbf{F} = \mathbf{0}$.

### 3.4 Finding the Potential Function

Given $\mathbf{F} = (2xy + z, x^2, x)$:

1. Integrate $P$ with respect to $x$: $\varphi = x^2 y + xz + g(y, z)$
2. Differentiate with respect to $y$: $\varphi_y = x^2 + g_y = Q = x^2$, so $g_y = 0$
3. Differentiate with respect to $z$: $\varphi_z = x + g_z = R = x$, so $g_z = 0$
4. Therefore $\varphi = x^2 y + xz + C$

```python
from sympy import symbols, diff, integrate, simplify

x, y, z = symbols('x y z')

# --- Test conservativeness and find potential function ---
P = 2*x*y + z
Q = x**2
R = x

# Check curl = 0
curl_x = diff(R, y) - diff(Q, z)  # dR/dy - dQ/dz
curl_y = diff(P, z) - diff(R, x)  # dP/dz - dR/dx
curl_z = diff(Q, x) - diff(P, y)  # dQ/dx - dP/dy

print(f"curl F = ({curl_x}, {curl_y}, {curl_z})")
# Should be (0, 0, 0) for conservative field

# Find potential: integrate P w.r.t. x
phi = integrate(P, x)  # x^2*y + x*z
print(f"After integrating P w.r.t. x: phi = {phi} + g(y,z)")

# Check: d(phi)/dy should equal Q
phi_y = diff(phi, y)
g_y = simplify(Q - phi_y)
print(f"g_y = Q - phi_y = {g_y}")  # 0

# Check: d(phi)/dz should equal R
phi_z = diff(phi, z)
g_z = simplify(R - phi_z)
print(f"g_z = R - phi_z = {g_z}")  # 0

print(f"\nPotential function: phi = {phi}")
```

---

## 4. Green's Theorem

**Green's theorem** connects a **line integral** around a closed curve to a **double integral** over the enclosed region.

### 4.1 Statement

Let $C$ be a simple closed curve (traversed counter-clockwise) enclosing region $D$, and let $P, Q$ have continuous partial derivatives. Then:

$$\oint_C P\,dx + Q\,dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

**Left side:** Work done by $(P, Q)$ around the loop.
**Right side:** Integral of the "microscopic circulation" (the 2D curl) over the enclosed area.

**Intuition:** Green's theorem says that the total circulation around the boundary equals the sum of all the tiny rotations inside. It is the 2D special case of Stokes' theorem.

### 4.2 Applications

**Area formula:** Setting $P = -y/2$, $Q = x/2$:

$$A = \frac{1}{2}\oint_C x\,dy - y\,dx$$

This is how planimeters measure area and how the **shoelace formula** for polygon area works.

```python
import numpy as np
from scipy.integrate import dblquad, quad

# --- Green's theorem verification ---
# F = (-y^2, x^2), C = unit circle (counterclockwise)
# Line integral: integral_C -y^2 dx + x^2 dy

# Parameterize: x = cos(t), y = sin(t), 0 <= t <= 2pi
def line_integrand(t):
    """Integrand for the line integral around the unit circle."""
    x, y = np.cos(t), np.sin(t)
    dx_dt, dy_dt = -np.sin(t), np.cos(t)
    P = -y**2
    Q = x**2
    return P * dx_dt + Q * dy_dt

line_result, _ = quad(line_integrand, 0, 2 * np.pi)
print(f"Line integral:   {line_result:.6f}")

# Double integral: integral_D (dQ/dx - dP/dy) dA
# dQ/dx = 2x, dP/dy = -2y
# Integrand = 2x + 2y = 2(x + y)
# Over the unit disk

area_result, _ = dblquad(
    lambda y, x: 2 * (x + y),
    -1, 1,
    lambda x: -np.sqrt(1 - x**2),
    lambda x: np.sqrt(1 - x**2)
)
print(f"Double integral: {area_result:.6f}")
print(f"Green's theorem verified: {np.isclose(line_result, area_result)}")
```

---

## 5. Surface Integrals

### 5.1 Parametric Surfaces

A surface $S$ can be parameterized as $\mathbf{r}(u, v) = (x(u,v), y(u,v), z(u,v))$ for $(u, v) \in D$.

The **surface area element** is:

$$dS = \left\|\frac{\partial\mathbf{r}}{\partial u} \times \frac{\partial\mathbf{r}}{\partial v}\right\| du\, dv$$

The cross product $\mathbf{r}_u \times \mathbf{r}_v$ gives the normal vector, and its magnitude gives the "stretching factor" of the parameterization.

### 5.2 Flux Integrals

The **flux** of a vector field $\mathbf{F}$ through a surface $S$ with outward normal $\hat{\mathbf{n}}$:

$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_S \mathbf{F} \cdot \hat{\mathbf{n}}\, dS = \iint_D \mathbf{F} \cdot (\mathbf{r}_u \times \mathbf{r}_v)\, du\, dv$$

**Physical meaning:** Flux measures the "flow rate" of the field through the surface. If $\mathbf{F}$ is a velocity field, the flux is the volume of fluid crossing $S$ per unit time.

---

## 6. Stokes' Theorem

**Stokes' theorem** generalizes Green's theorem to surfaces in 3D.

### 6.1 Statement

Let $S$ be an oriented surface bounded by a simple closed curve $C$ (with compatible orientation). Then:

$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

**Left side:** Circulation of $\mathbf{F}$ around the boundary curve.
**Right side:** Flux of the curl through the surface.

**Intuition:** The total circulation around the edge equals the sum of all microscopic rotations across the surface -- even when the surface is curved in 3D.

**Special case:** When $S$ is flat (in the $xy$-plane), Stokes' theorem reduces to Green's theorem.

```python
import numpy as np
from scipy.integrate import dblquad, quad

# --- Stokes' theorem verification ---
# F = (z, x, y)
# Surface S: portion of z = 1 - x^2 - y^2 above z = 0
# Boundary C: the circle x^2 + y^2 = 1, z = 0

# 1. Line integral around C: x = cos(t), y = sin(t), z = 0
def stokes_line(t):
    x, y, z = np.cos(t), np.sin(t), 0.0
    dx, dy, dz = -np.sin(t), np.cos(t), 0.0
    # F = (z, x, y) = (0, cos(t), sin(t))
    return z * dx + x * dy + y * dz

line_val, _ = quad(stokes_line, 0, 2 * np.pi)
print(f"Line integral (circulation): {line_val:.6f}")

# 2. Surface integral of curl(F) dot dS
# curl(F) = (dR/dy - dQ/dz, dP/dz - dR/dx, dQ/dx - dP/dy)
#          = (1 - 1, 1 - 0, 1 - 0) = (0, 1, 1)
# Wait, let's recalculate:
# P=z, Q=x, R=y
# curl_x = dR/dy - dQ/dz = 1 - 0 = 1
# curl_y = dP/dz - dR/dx = 1 - 0 = 1
# curl_z = dQ/dx - dP/dy = 1 - 0 = 1
# curl(F) = (1, 1, 1)

# Surface z = 1 - x^2 - y^2 above z=0, parameterized by (x, y)
# Normal: (-dz/dx, -dz/dy, 1) = (2x, 2y, 1) (outward/upward)
# curl(F) dot n = 1*(2x) + 1*(2y) + 1*(1) = 2x + 2y + 1

def surface_integrand(y, x):
    """curl(F) dot (outward normal) for z = 1 - x^2 - y^2."""
    return 2 * x + 2 * y + 1

surface_val, _ = dblquad(
    surface_integrand,
    -1, 1,
    lambda x: -np.sqrt(max(1 - x**2, 0)),
    lambda x: np.sqrt(max(1 - x**2, 0))
)
print(f"Surface integral (curl flux): {surface_val:.6f}")
print(f"Stokes' theorem verified: {np.isclose(line_val, surface_val)}")
# Both should equal pi
print(f"Analytical value: pi = {np.pi:.6f}")
```

---

## 7. The Divergence Theorem (Gauss's Theorem)

### 7.1 Statement

Let $E$ be a solid region bounded by a closed surface $S$ (outward normal). Then:

$$\oiint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_E \nabla \cdot \mathbf{F}\, dV$$

**Left side:** Total flux of $\mathbf{F}$ out through the surface.
**Right side:** Total divergence (source strength) inside the volume.

**Intuition:** If you have sources inside a closed box, the net outflow through the walls equals the total source strength inside. This is the 3D analogue of the fundamental theorem of calculus.

### 7.2 Applications

**Gauss's law in electrostatics:**

$$\oiint_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\varepsilon_0}$$

This relates the electric flux through a surface to the enclosed charge -- a direct consequence of the divergence theorem applied to $\nabla \cdot \mathbf{E} = \rho / \varepsilon_0$.

### 7.3 The Big Picture

The three major theorems form a hierarchy:

| Theorem | Dimension | Relates |
|---------|-----------|---------|
| Fundamental Theorem of Calculus | 1D | $\int_a^b f'(x)\,dx = f(b) - f(a)$ |
| Green's / Stokes' | 2D/3D | Boundary integral = interior curl integral |
| Divergence Theorem | 3D | Boundary flux = interior divergence integral |

All three are instances of the **generalized Stokes' theorem**: $\int_{\partial\Omega} \omega = \int_\Omega d\omega$. The pattern is always: **integrating a derivative over a region equals integrating the original over the boundary**.

```python
import numpy as np
from scipy.integrate import tplquad, dblquad

# --- Divergence theorem verification ---
# F = (x^2, y^2, z^2)
# Region E: unit sphere x^2 + y^2 + z^2 <= 1
# div(F) = 2x + 2y + 2z

# Volume integral of div(F) over the unit sphere (spherical coords)
def div_integrand(rho, phi, theta):
    """div(F) * rho^2 * sin(phi) in spherical coordinates."""
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    div_F = 2 * x + 2 * y + 2 * z
    return div_F * rho**2 * np.sin(phi)

vol_integral, _ = tplquad(
    div_integrand,
    0, 2 * np.pi,                  # theta
    lambda t: 0, lambda t: np.pi,  # phi
    lambda t, p: 0, lambda t, p: 1 # rho
)
print(f"Volume integral of div(F): {vol_integral:.6f}")

# By symmetry, integral of 2x, 2y, 2z over the sphere are each 0
# (odd functions over symmetric domain)
# So the divergence theorem gives flux = 0
print(f"Expected (by symmetry): 0.000000")
print(f"Verified: {np.isclose(vol_integral, 0, atol=1e-10)}")

# Let's try F = (x, y, z) instead: div(F) = 3
# Volume of unit sphere = 4*pi/3
# Volume integral = 3 * 4*pi/3 = 4*pi
def div_integrand_2(rho, phi, theta):
    """div(F) = 3, times volume element."""
    return 3 * rho**2 * np.sin(phi)

vol2, _ = tplquad(
    div_integrand_2,
    0, 2 * np.pi,
    lambda t: 0, lambda t: np.pi,
    lambda t, p: 0, lambda t, p: 1
)
print(f"\nFor F=(x,y,z): volume integral of div(F) = {vol2:.6f}")
print(f"Analytical (4*pi): {4 * np.pi:.6f}")
```

---

## 8. Cross-References

- **Mathematical Methods Lesson 05** provides a comprehensive treatment of vector analysis including the nabla operator in different coordinate systems, Helmholtz decomposition, and proofs of the integral theorems.
- **Electrodynamics Lesson 01-06** applies Green's, Stokes', and Divergence theorems extensively to Maxwell's equations.
- **Lesson 10 (Multiple Integrals)** covers the double and triple integration techniques used throughout this lesson.

---

## Practice Problems

**1.** Let $\mathbf{F} = (x^2 y, xy^2)$. Verify Green's theorem by computing both the line integral around the triangle with vertices $(0,0)$, $(1,0)$, $(0,1)$ and the corresponding double integral.

**2.** Determine whether $\mathbf{F} = (2xy + z^2, x^2 + 2yz, 2xz + y^2)$ is conservative. If so, find the potential function $\varphi$ and evaluate $\int_C \mathbf{F} \cdot d\mathbf{r}$ from $(0,0,0)$ to $(1,2,3)$.

**3.** Use the Divergence theorem to evaluate $\oiint_S \mathbf{F} \cdot d\mathbf{S}$ where $\mathbf{F} = (x^3, y^3, z^3)$ and $S$ is the sphere $x^2 + y^2 + z^2 = 4$.

**4.** Use Stokes' theorem to evaluate $\oint_C \mathbf{F} \cdot d\mathbf{r}$ where $\mathbf{F} = (y^2, z^2, x^2)$ and $C$ is the triangle formed by $(1,0,0)$, $(0,1,0)$, $(0,0,1)$ (oriented counterclockwise when viewed from above).

**5.** Show that the flux of $\mathbf{F} = \mathbf{r}/\|\mathbf{r}\|^3$ through any closed surface enclosing the origin is $4\pi$. (Hint: this is Gauss's law for a point charge.) What happens if the surface does not enclose the origin?

---

## References

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapter 16
- **Jerrold E. Marsden & Anthony Tromba**, *Vector Calculus*, 6th Edition, Chapters 7-8
- **H.M. Schey**, *Div, Grad, Curl, and All That*, 4th Edition (excellent intuitive treatment)
- **3Blue1Brown**, "Divergence and Curl" (visual intuition)

---

[Previous: Multiple Integrals](./10_Multiple_Integrals.md) | [Next: First-Order Ordinary Differential Equations](./12_First_Order_ODE.md)

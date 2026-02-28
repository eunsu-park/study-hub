# 09. Multivariable Functions

## Learning Objectives

- Define functions of several variables and interpret level curves and level surfaces geometrically
- Compute partial derivatives and directional derivatives, and explain the gradient vector's role
- Apply the multivariable chain rule to composite functions
- Construct tangent planes and linear approximations for surfaces
- Classify critical points using the second derivative test for functions of two variables

---

## 1. Functions of Several Variables

A function $f: \mathbb{R}^n \to \mathbb{R}$ assigns a single real number to each point in $n$-dimensional space.

**Two variables:** $z = f(x, y)$ maps a point $(x, y)$ in the plane to a height $z$. The graph is a **surface** in 3D.

**Analogy:** Think of $f(x, y)$ as a topographic map. The input $(x, y)$ is your GPS location; the output $z$ is your elevation. The map itself shows **level curves** (contour lines) -- curves where $f(x, y) = c$ for constant $c$.

**Example:**

$$f(x, y) = x^2 + y^2$$

- The graph is a **paraboloid** opening upward.
- Level curves: $x^2 + y^2 = c$ are circles of radius $\sqrt{c}$.
- Closely spaced level curves mean the surface is steep (like tightly packed contour lines on a mountain).

**Three variables:** $w = f(x, y, z)$ cannot be graphed directly (would need 4D), but we can visualize **level surfaces** $f(x, y, z) = c$.

**Example:** The temperature field $T(x, y, z) = 100 - x^2 - y^2 - z^2$ has level surfaces (isotherms) that are concentric spheres.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Level curves (contour plot) ---
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Saddle function: f(x,y) = x^2 - y^2
Z = X**2 - Y**2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Contour plot (level curves)
cs = axes[0].contour(X, Y, Z, levels=15, cmap='RdBu_r')
axes[0].clabel(cs, inline=True, fontsize=8)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Level Curves of $f(x,y) = x^2 - y^2$')
axes[0].set_aspect('equal')

# Filled contour for better visualization
cf = axes[1].contourf(X, Y, Z, levels=20, cmap='RdBu_r')
plt.colorbar(cf, ax=axes[1], label='f(x,y)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Filled Contour Plot')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

# --- 3D surface plot ---
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# Paraboloid
ax1 = fig.add_subplot(121, projection='3d')
Z1 = X**2 + Y**2
ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Paraboloid: $z = x^2 + y^2$')

# Saddle surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='RdBu_r', alpha=0.8)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Saddle: $z = x^2 - y^2$')

plt.tight_layout()
plt.show()
```

---

## 2. Partial Derivatives

### 2.1 Definition and Geometric Meaning

The **partial derivative** of $f(x, y)$ with respect to $x$ is:

$$\frac{\partial f}{\partial x} = f_x = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}$$

**Key idea:** Hold all other variables constant and differentiate with respect to one variable at a time.

**Geometric interpretation:** $f_x(a, b)$ is the slope of the surface $z = f(x, y)$ at the point $(a, b)$ in the $x$-direction. Imagine slicing the surface with a plane $y = b$; the resulting curve has slope $f_x$.

**Example:** For $f(x, y) = x^2 y + \sin(xy)$:

$$f_x = 2xy + y\cos(xy) \quad \text{(treat } y \text{ as constant)}$$

$$f_y = x^2 + x\cos(xy) \quad \text{(treat } x \text{ as constant)}$$

### 2.2 Higher-Order Partial Derivatives

Second-order partial derivatives:

$$f_{xx} = \frac{\partial^2 f}{\partial x^2}, \quad f_{yy} = \frac{\partial^2 f}{\partial y^2}, \quad f_{xy} = \frac{\partial^2 f}{\partial y\,\partial x}, \quad f_{yx} = \frac{\partial^2 f}{\partial x\,\partial y}$$

**Clairaut's Theorem:** If $f_{xy}$ and $f_{yx}$ are both continuous, then $f_{xy} = f_{yx}$. This means the order of differentiation doesn't matter -- a tremendously useful fact.

---

## 3. Directional Derivatives and the Gradient

### 3.1 Directional Derivative

The partial derivatives give rates of change along the coordinate axes. What about an **arbitrary direction**?

The **directional derivative** of $f$ at $(a, b)$ in the direction of unit vector $\hat{\mathbf{u}} = (u_1, u_2)$ is:

$$D_{\hat{\mathbf{u}}} f = \lim_{h \to 0} \frac{f(a + hu_1,\, b + hu_2) - f(a, b)}{h}$$

If $f$ is differentiable, this simplifies beautifully:

$$D_{\hat{\mathbf{u}}} f = f_x \, u_1 + f_y \, u_2 = \nabla f \cdot \hat{\mathbf{u}}$$

### 3.2 The Gradient Vector

The **gradient** of $f(x, y)$ is the vector of partial derivatives:

$$\nabla f = \left(\frac{\partial f}{\partial x},\, \frac{\partial f}{\partial y}\right) = f_x \,\hat{\mathbf{i}} + f_y \,\hat{\mathbf{j}}$$

**Three fundamental properties of the gradient:**

1. **Direction of steepest ascent:** $\nabla f$ points in the direction where $f$ increases fastest.
2. **Magnitude = maximum rate:** $\|\nabla f\|$ equals the maximum directional derivative.
3. **Perpendicular to level curves:** $\nabla f$ is always orthogonal to the level curve $f(x,y) = c$ passing through the point.

**Analogy:** Imagine standing on a hillside. The gradient vector tells you: "This is the steepest uphill direction, and this is how steep it is." Water flows in the direction of $-\nabla f$ (steepest descent) -- this is exactly the principle behind gradient descent in machine learning.

**Example:** For $f(x, y) = x^2 + 4y^2$ at the point $(1, 1)$:

$$\nabla f = (2x, 8y) \big|_{(1,1)} = (2, 8)$$

- Steepest ascent direction: $(2, 8) / \|(2,8)\| = (1/\sqrt{17},\, 4/\sqrt{17})$
- Maximum rate of increase: $\|(2, 8)\| = \sqrt{68} = 2\sqrt{17}$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gradient field visualization ---
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Function: f(x,y) = x^2 + 4y^2 (elliptic paraboloid)
Z = X**2 + 4 * Y**2

# Gradient components
dfdx = 2 * X      # partial f / partial x
dfdy = 8 * Y      # partial f / partial y

fig, ax = plt.subplots(figsize=(8, 8))

# Draw level curves in the background
x_fine = np.linspace(-2, 2, 200)
y_fine = np.linspace(-2, 2, 200)
Xf, Yf = np.meshgrid(x_fine, y_fine)
Zf = Xf**2 + 4 * Yf**2
cs = ax.contour(Xf, Yf, Zf, levels=10, cmap='Blues', alpha=0.6)
ax.clabel(cs, inline=True, fontsize=8)

# Draw gradient vectors (arrows point in direction of steepest ascent)
ax.quiver(X, Y, dfdx, dfdy, color='red', alpha=0.7,
          scale=60, width=0.004)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradient Field of $f(x,y) = x^2 + 4y^2$\n'
             'Arrows = $\\nabla f$ (steepest ascent), perpendicular to contours')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. Chain Rule for Multivariable Functions

### 4.1 Single Parameter

If $z = f(x, y)$ where $x = x(t)$ and $y = y(t)$, then:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

**Intuition:** The total rate of change of $z$ has contributions from both the $x$-path and the $y$-path. Each contribution is the sensitivity ($\partial f/\partial x$ or $\partial f/\partial y$) times the rate of change of that variable.

### 4.2 Two Parameters

If $z = f(x, y)$ where $x = x(s, t)$ and $y = y(s, t)$, then:

$$\frac{\partial z}{\partial s} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial s} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial s}$$

$$\frac{\partial z}{\partial t} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial t} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial t}$$

A helpful mnemonic is the **tree diagram**: draw a tree from $z$ branching to $x$ and $y$, then from each to $s$ and $t$. Multiply along each branch and sum over all paths.

**Example:** Let $z = x^2 y$ with $x = s\cos t$, $y = s\sin t$:

$$\frac{\partial z}{\partial s} = 2xy \cdot \cos t + x^2 \cdot \sin t$$

### 4.3 Implicit Differentiation

If $F(x, y) = 0$ defines $y$ implicitly as a function of $x$, then:

$$\frac{dy}{dx} = -\frac{F_x}{F_y} \quad \text{(provided } F_y \neq 0\text{)}$$

This follows from differentiating $F(x, y(x)) = 0$ with the chain rule.

---

## 5. Tangent Planes and Linear Approximation

### 5.1 Tangent Plane

The **tangent plane** to $z = f(x, y)$ at the point $(a, b, f(a,b))$ is:

$$z - f(a,b) = f_x(a,b)(x - a) + f_y(a,b)(y - b)$$

This is the 2D analogue of the tangent line $y - f(a) = f'(a)(x - a)$.

### 5.2 Linear Approximation

Near the point $(a, b)$, we can approximate $f$ by its tangent plane:

$$f(x, y) \approx f(a, b) + f_x(a,b)(x - a) + f_y(a,b)(y - b)$$

Equivalently, the **total differential** is:

$$df = f_x\,dx + f_y\,dy$$

**Application:** If $z = \sqrt{x^2 + y^2}$ (distance from origin), approximate $z$ at $(3.02, 3.97)$ using the tangent plane at $(3, 4)$:

$$f(3, 4) = 5, \quad f_x = \frac{x}{\sqrt{x^2+y^2}} = \frac{3}{5}, \quad f_y = \frac{4}{5}$$

$$f(3.02, 3.97) \approx 5 + \frac{3}{5}(0.02) + \frac{4}{5}(-0.03) = 5 + 0.012 - 0.024 = 4.988$$

Exact value: $\sqrt{3.02^2 + 3.97^2} = \sqrt{24.8813} \approx 4.98812$ -- the linear approximation is very close.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Tangent plane visualization ---
def f(x, y):
    return np.sin(x) * np.cos(y)

def tangent_plane(x, y, a, b):
    """Tangent plane to f at point (a, b)."""
    f0 = f(a, b)
    fx = np.cos(a) * np.cos(b)   # partial f / partial x
    fy = -np.sin(a) * np.sin(b)  # partial f / partial y
    return f0 + fx * (x - a) + fy * (y - b)

a, b = 1.0, 0.5  # point of tangency

x = np.linspace(-1, 3, 100)
y = np.linspace(-1.5, 2.5, 100)
X, Y = np.meshgrid(x, y)

Z_surface = f(X, Y)
Z_tangent = tangent_plane(X, Y, a, b)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with some transparency
ax.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=0.6)

# Plot tangent plane (clip to a small region for clarity)
mask = (np.abs(X - a) < 1.2) & (np.abs(Y - b) < 1.2)
Z_plane_clipped = np.where(mask, Z_tangent, np.nan)
ax.plot_surface(X, Y, Z_plane_clipped, color='red', alpha=0.4)

# Mark the point of tangency
ax.scatter([a], [b], [f(a, b)], color='black', s=80, zorder=5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'Surface $z = \\sin(x)\\cos(y)$ and Tangent Plane at ({a}, {b})')
plt.tight_layout()
plt.show()
```

---

## 6. Critical Points and the Second Derivative Test

### 6.1 Finding Critical Points

A **critical point** of $f(x, y)$ is where both partial derivatives vanish:

$$f_x(a, b) = 0 \quad \text{and} \quad f_y(a, b) = 0$$

At critical points, the tangent plane is horizontal.

### 6.2 Second Derivative Test

To classify a critical point $(a, b)$, compute the **discriminant** (Hessian determinant):

$$D = f_{xx}(a,b)\,f_{yy}(a,b) - [f_{xy}(a,b)]^2$$

| Condition | Classification |
|-----------|----------------|
| $D > 0$ and $f_{xx} > 0$ | Local minimum |
| $D > 0$ and $f_{xx} < 0$ | Local maximum |
| $D < 0$ | Saddle point |
| $D = 0$ | Inconclusive (need further analysis) |

**Why does this work?** The Hessian matrix $H = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{pmatrix}$ captures the curvature of $f$ in all directions. If both eigenvalues of $H$ are positive (positive definite), we have a minimum; if both negative, a maximum; if they differ in sign, a saddle.

**Example:** $f(x, y) = x^3 - 3xy + y^3$

$$f_x = 3x^2 - 3y = 0 \implies y = x^2$$

$$f_y = -3x + 3y^2 = 0 \implies x = y^2$$

Substituting: $x = (x^2)^2 = x^4$, so $x(x^3 - 1) = 0$, giving $x = 0$ or $x = 1$.

- At $(0, 0)$: $D = (0)(0) - (-3)^2 = -9 < 0$ -- **saddle point**
- At $(1, 1)$: $D = (6)(6) - (-3)^2 = 27 > 0$ and $f_{xx} = 6 > 0$ -- **local minimum**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, diff, solve, Matrix

# --- Symbolic critical point analysis ---
x, y = symbols('x y', real=True)
f = x**3 - 3*x*y + y**3

fx = diff(f, x)
fy = diff(f, y)
print(f"f_x = {fx}")
print(f"f_y = {fy}")

# Find critical points
critical_pts = solve([fx, fy], [x, y])
print(f"Critical points: {critical_pts}")

# Second derivative test
fxx = diff(f, x, 2)
fyy = diff(f, y, 2)
fxy = diff(f, x, y)

for pt in critical_pts:
    D_val = fxx.subs([(x, pt[0]), (y, pt[1])]) * \
            fyy.subs([(x, pt[0]), (y, pt[1])]) - \
            fxy.subs([(x, pt[0]), (y, pt[1])])**2
    fxx_val = fxx.subs([(x, pt[0]), (y, pt[1])])
    print(f"\nAt {pt}: D = {D_val}, f_xx = {fxx_val}")
    if D_val > 0 and fxx_val > 0:
        print("  -> Local minimum")
    elif D_val > 0 and fxx_val < 0:
        print("  -> Local maximum")
    elif D_val < 0:
        print("  -> Saddle point")
    else:
        print("  -> Inconclusive")
```

---

## 7. Cross-References

- **Mathematical Methods Lesson 04** covers **Lagrange multipliers** for constrained optimization, extending the unconstrained optimization treated here.
- **Mathematical Methods Lesson 05** provides a deeper treatment of vector analysis including div, grad, and curl in different coordinate systems.
- **Math for AI Lesson 08** discusses gradient descent optimization, which directly applies the gradient concept from this lesson.

---

## Practice Problems

**1.** For $f(x, y) = \ln(x^2 + y^2)$:
   - (a) Find $\nabla f$ and show it points radially outward.
   - (b) Compute the directional derivative at $(1, 1)$ in the direction of $(3, 4)$.
   - (c) At which points is $\nabla f$ undefined?

**2.** Let $w = xy + yz + zx$ with $x = t$, $y = t^2$, $z = t^3$. Use the chain rule to find $dw/dt$ and verify by first substituting and then differentiating directly.

**3.** Find and classify all critical points of $f(x, y) = 2x^3 + 6xy^2 - 3y^3 - 150x$.

**4.** The ideal gas law $PV = nRT$ implicitly defines $P$ as a function of $V$ and $T$.
   - (a) Find $\partial P/\partial V$ and $\partial P/\partial T$ using implicit differentiation.
   - (b) Verify that $\frac{\partial P}{\partial V}\frac{\partial V}{\partial T}\frac{\partial T}{\partial P} = -1$ (the **cyclic relation**).

**5.** Use the linear approximation of $f(x, y) = \sqrt{x}\,e^y$ at $(4, 0)$ to estimate $f(4.1, -0.05)$. Compare with the exact value.

---

## References

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapters 14.1-14.7
- **Jerrold E. Marsden & Anthony Tromba**, *Vector Calculus*, 6th Edition, Chapter 2
- **George B. Thomas**, *Thomas' Calculus*, Chapters 14-15
- **Khan Academy**, "Multivariable Calculus" (interactive visualizations)

---

[Previous: Parametric Curves and Polar Coordinates](./08_Parametric_and_Polar.md) | [Next: Multiple Integrals](./10_Multiple_Integrals.md)

# 17. Calculus of Variations

## Learning Objectives

- Understand the concept of functionals and their difference from ordinary functions
- Derive the Euler-Lagrange equation and apply it to various special forms
- Solve classical variational problems such as the brachistochrone, geodesics, and catenary
- Apply Lagrange multipliers to constrained variational problems
- Understand the variational foundations of Lagrangian and Hamiltonian mechanics
- Recognize modern applications of calculus of variations (finite element method, optimal control, Rayleigh-Ritz method)

> **Importance in physics**: Calculus of variations is the mathematical foundation of the principle of least action: "nature minimizes action." It reformulates Newtonian mechanics into Lagrangian and Hamiltonian forms and plays a central role throughout physics and engineering, including Fermat's principle (optics), minimal surfaces (soap films), and optimal path problems. The path integral in quantum mechanics, Einstein's equations in general relativity, and the Lagrangian density in classical field theory all rest on calculus of variations.

---

## 1. Basic Concepts of Calculus of Variations

### 1.1 Distinction Between Functionals and Functions

An ordinary function $f: \mathbb{R} \to \mathbb{R}$ takes a **number** as input and returns a **number**. In contrast, a **functional** $J: \mathcal{F} \to \mathbb{R}$ takes a **function** as input and returns a **number**.

$$J[y] = \int_a^b F(x, y(x), y'(x))\, dx$$

Here $F$ is the **integrand**, and $y(x)$ belongs to the space $\mathcal{F}$ of **admissible functions**. Admissible functions satisfy:

- $y(x) \in C^2[a, b]$ (twice continuously differentiable)
- Boundary conditions: $y(a) = y_a$, $y(b) = y_b$ (fixed endpoints)

### 1.2 Variation $\delta y$ and First Variation

The **variation** of an admissible function $y(x)$ is defined for an arbitrary function $\eta(x)$ that vanishes at the boundaries:

$$\delta y(x) = \epsilon \eta(x), \quad \eta(a) = \eta(b) = 0$$

The **first variation** corresponds to differentiation of the functional:

$$\delta J = \left.\frac{d}{d\epsilon}J[y + \epsilon\eta]\right|_{\epsilon=0} = \int_a^b \left(\frac{\partial F}{\partial y}\eta + \frac{\partial F}{\partial y'}\eta'\right) dx$$

For the functional to have an extremum, $\delta J = 0$ must hold for all admissible variations $\eta$.

```python
import numpy as np
import matplotlib.pyplot as plt

def evaluate_functional(F, y_func, x_range, N=1000):
    """Numerically evaluate functional J[y] = ∫F(x, y, y')dx"""
    a, b = x_range
    x = np.linspace(a, b, N)
    h = (b - a) / (N - 1)

    y = y_func(x)
    # Compute y' using central differences
    yp = np.gradient(y, x)

    # Trapezoidal integration
    integrand = F(x, y, yp)
    return np.trapz(integrand, x)

# Example: arc length functional J[y] = ∫√(1 + y'²)dx
F_arclength = lambda x, y, yp: np.sqrt(1 + yp**2)

# Straight line y = x vs parabola y = x²  (interval [0, 1])
y_line = lambda x: x
y_parabola = lambda x: x**2

J_line = evaluate_functional(F_arclength, y_line, (0, 1))
J_parab = evaluate_functional(F_arclength, y_parabola, (0, 1))

print(f"Arc length of straight line: J[y=x]     = {J_line:.6f}")
print(f"Arc length of parabola:      J[y=x²]   = {J_parab:.6f}")
print(f"Straight line is shorter (shortest path): {J_line < J_parab}")
```

---

## 2. Euler-Lagrange Equation

### 2.1 Derivation

Apply integration by parts to the condition $\delta J = 0$:

$$\delta J = \int_a^b \left(\frac{\partial F}{\partial y}\eta + \frac{\partial F}{\partial y'}\eta'\right) dx$$

Integrating the second term by parts:

$$\int_a^b \frac{\partial F}{\partial y'}\eta'\, dx = \left[\frac{\partial F}{\partial y'}\eta\right]_a^b - \int_a^b \frac{d}{dx}\frac{\partial F}{\partial y'}\eta\, dx$$

The boundary term vanishes since $\eta(a) = \eta(b) = 0$. Therefore:

$$\delta J = \int_a^b \left(\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'}\right)\eta\, dx = 0$$

**Fundamental lemma of calculus of variations**: If the integral is zero for all $\eta$, the integrand must be zero. Thus the **Euler-Lagrange equation** is derived:

$$\boxed{\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0}$$

### 2.2 Special Cases

**Case 1**: When $F$ does not depend on $y$ ($F = F(x, y')$):

$$\frac{\partial F}{\partial y'} = C \quad \text{(constant)}$$

**Case 2**: When $F$ does not explicitly depend on $x$ ($F = F(y, y')$) -- **Beltrami identity**:

$$\boxed{F - y'\frac{\partial F}{\partial y'} = C}$$

This follows from $\frac{d}{dx}\left(F - y'F_{y'}\right) = -xF_x = 0$.

```python
import sympy as sp

x = sp.Symbol('x')
y = sp.Function('y')(x)
yp = y.diff(x)

def euler_lagrange(F, y_func, x_var):
    """Symbolically derive the Euler-Lagrange equation"""
    yp = y_func.diff(x_var)
    # ∂F/∂y
    dF_dy = sp.diff(F, y_func)
    # d/dx(∂F/∂y')
    dF_dyp = sp.diff(F, yp)
    d_dx_dF_dyp = sp.diff(dF_dyp, x_var)

    el_eq = sp.simplify(dF_dy - d_dx_dF_dyp)
    return sp.Eq(el_eq, 0)

# Example 1: Shortest distance — F = √(1 + y'²)
F1 = sp.sqrt(1 + yp**2)
el1 = euler_lagrange(F1, y, x)
print("=== Shortest distance problem ===")
print(f"Euler-Lagrange: {el1}")
print("Solution: y'' = 0  →  y = ax + b (straight line)")

# Example 2: Minimal surface of revolution — F = y√(1 + y'²)
F2 = y * sp.sqrt(1 + yp**2)
el2 = euler_lagrange(F2, y, x)
print(f"\n=== Minimal surface of revolution ===")
print(f"Euler-Lagrange: {el2}")
print("Solution: y = c₁ cosh((x - c₂)/c₁) (catenary)")
```

---

## 3. Classical Variational Problems

### 3.1 Brachistochrone Problem

Find the fastest path for a particle sliding without friction from point $A(0,0)$ to point $B(x_1, y_1)$ in a gravitational field. ($y$-axis points downward.)

By energy conservation, the speed is $v = \sqrt{2gy}$, and the travel time is:

$$T[y] = \int_0^{x_1} \frac{\sqrt{1 + y'^2}}{\sqrt{2gy}}\, dx$$

Since $F = \sqrt{(1 + y'^2)/(2gy)}$ does not explicitly depend on $x$, apply the Beltrami identity:

$$F - y'F_{y'} = C \implies \frac{1}{\sqrt{2gy(1 + y'^2)}} = C$$

Simplifying gives $y(1 + y'^2) = \frac{1}{2gC^2} = 2R$ (constant). Solving with parameter $\theta$ yields a **cycloid**:

$$\boxed{x = R(\theta - \sin\theta), \quad y = R(1 - \cos\theta)}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def brachistochrone(x1, y1, N=500):
    """Compute the brachistochrone (cycloid)"""
    def equations(params):
        R, theta1 = params
        eq1 = R * (theta1 - np.sin(theta1)) - x1
        eq2 = R * (1 - np.cos(theta1)) - y1
        return [eq1, eq2]

    R, theta1 = fsolve(equations, [1.0, np.pi])

    theta = np.linspace(0, theta1, N)
    x_cyc = R * (theta - np.sin(theta))
    y_cyc = R * (1 - np.cos(theta))
    return x_cyc, y_cyc, R, theta1

# Comparison: cycloid vs straight line vs parabola
x1, y1 = 1.0, 0.6
x_cyc, y_cyc, R, th1 = brachistochrone(x1, y1)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Straight line
ax.plot([0, x1], [0, y1], 'b--', label='Straight line', linewidth=2)

# Parabola (y = ax²)
x_par = np.linspace(0, x1, 200)
a_par = y1 / x1**2
y_par = a_par * x_par**2
ax.plot(x_par, y_par, 'g-.', label='Parabola', linewidth=2)

# Cycloid
ax.plot(x_cyc, y_cyc, 'r-', label='Cycloid (fastest)', linewidth=2.5)

ax.set_xlabel('x')
ax.set_ylabel('y (downward direction)')
ax.set_title('Brachistochrone Problem')
ax.legend()
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('brachistochrone.png', dpi=150)
plt.show()

# Comparison of travel times along each path
g = 9.81
def travel_time(x_path, y_path):
    """Calculate travel time along a path"""
    ds = np.sqrt(np.diff(x_path)**2 + np.diff(y_path)**2)
    v = np.sqrt(2 * g * (y_path[:-1] + y_path[1:]) / 2)
    v = np.where(v < 1e-10, 1e-10, v)  # avoid division by zero
    return np.sum(ds / v)

t_line = travel_time(np.linspace(0, x1, 500), np.linspace(0, y1, 500))
t_par = travel_time(x_par, y_par)
t_cyc = travel_time(x_cyc, y_cyc)

print(f"Straight line travel time:  {t_line:.4f} s")
print(f"Parabola travel time:       {t_par:.4f} s")
print(f"Cycloid travel time:        {t_cyc:.4f} s (minimum)")
```

### 3.2 Geodesics

**Geodesics on a plane**: Shortest distance between two points. From $F = \sqrt{1 + y'^2}$, we get $y'' = 0$, i.e., a **straight line**.

**Geodesics on a sphere**: In spherical coordinates $(\theta, \phi)$, the arc length is:

$$L = R\int \sqrt{d\theta^2 + \sin^2\theta\, d\phi^2} = R\int_{\phi_1}^{\phi_2} \sqrt{\theta'^2 + \sin^2\theta}\, d\phi$$

Solving the Euler-Lagrange equation yields a **great circle**.

### 3.3 Catenary

Find the shape of a uniform-density chain hanging with fixed endpoints that minimizes potential energy:

$$U[y] = \rho g \int_{-a}^{a} y\sqrt{1 + y'^2}\, dx$$

Under the constraint that the chain length $L = \int \sqrt{1 + y'^2}\, dx$ is constant, applying the Beltrami identity to $F = y\sqrt{1 + y'^2}$ gives:

$$y = c_1 \cosh\left(\frac{x - c_2}{c_1}\right)$$

This is the **catenary**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Draw catenary
def catenary(x, c1, c2=0):
    """Catenary: y = c₁ cosh((x - c₂)/c₁)"""
    return c1 * np.cosh((x - c2) / c1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Catenaries with various parameters
x = np.linspace(-2, 2, 300)
for c in [0.5, 1.0, 1.5, 2.0]:
    axes[0].plot(x, catenary(x, c), label=f'$c_1 = {c}$')
axes[0].set_title('Catenaries with various parameters')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) Catenary vs parabola comparison
c1 = 1.0
y_cat = catenary(x, c1) - c1  # shift lowest point to 0
y_par = 0.5 * x**2            # similarly shaped parabola
axes[1].plot(x, y_cat, 'r-', linewidth=2, label='Catenary')
axes[1].plot(x, y_par, 'b--', linewidth=2, label='Parabola')
axes[1].set_title('Catenary vs Parabola')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('catenary.png', dpi=150)
plt.show()
```

### 3.4 Isoperimetric Problem

Among closed curves with given perimeter $L$, find the shape that maximizes area. The solution is a **circle**.

Area: $A = \frac{1}{2}\oint (x\, dy - y\, dx)$, perimeter constraint: $L = \oint \sqrt{dx^2 + dy^2}$

### 3.5 Minimal Surface (Soap Film Problem)

The minimal surface of revolution formed by a soap film between two coaxial circular rings. Functional:

$$A[y] = 2\pi\int_a^b y\sqrt{1 + y'^2}\, dx$$

Solving with the Beltrami identity gives a **catenoid**: $y = c\cosh\left(\frac{x - x_0}{c}\right)$

---

## 4. Multiple Variables and Higher Derivatives

### 4.1 Multi-variable Euler-Lagrange Equations

For a functional with multiple dependent variables $y_1(x), y_2(x), \ldots, y_n(x)$:

$$J[y_1, \ldots, y_n] = \int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\, dx$$

An independent Euler-Lagrange equation holds for each $y_i$:

$$\frac{\partial F}{\partial y_i} - \frac{d}{dx}\frac{\partial F}{\partial y_i'} = 0, \quad i = 1, 2, \ldots, n$$

### 4.2 Higher Derivatives: Euler-Poisson Equation

When $F$ includes up to $y''$:

$$J[y] = \int_a^b F(x, y, y', y'')\, dx$$

**Euler-Poisson equation**:

$$\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} + \frac{d^2}{dx^2}\frac{\partial F}{\partial y''} = 0$$

Boundary conditions require $y$ and $y'$ to be fixed at both endpoints (4 boundary conditions).

**Example**: For the bending energy of an elastic beam $J[y] = \int_0^L \frac{EI}{2}(y'')^2\, dx$, we get $y'''' = 0$, i.e., a cubic polynomial.

### 4.3 Multiple Independent Variables

For a functional with $u(x, y)$:

$$J[u] = \iint_D F(x, y, u, u_x, u_y)\, dx\, dy$$

Euler-Lagrange equation:

$$\frac{\partial F}{\partial u} - \frac{\partial}{\partial x}\frac{\partial F}{\partial u_x} - \frac{\partial}{\partial y}\frac{\partial F}{\partial u_y} = 0$$

**Example**: Minimizing the Dirichlet functional $J[u] = \iint (u_x^2 + u_y^2)\, dx\, dy$ yields $\nabla^2 u = 0$ (Laplace equation).

```python
import sympy as sp

x = sp.Symbol('x')
y1 = sp.Function('y1')(x)
y2 = sp.Function('y2')(x)

# Two-variable Lagrangian example: coupled oscillator
# L = ½m(ẏ₁² + ẏ₂²) - ½k(y₁² + y₂²) - ½k'(y₁ - y₂)²
m, k, kp = sp.symbols('m k k_prime', positive=True)
y1p, y2p = y1.diff(x), y2.diff(x)

T = sp.Rational(1, 2) * m * (y1p**2 + y2p**2)
V = sp.Rational(1, 2) * k * (y1**2 + y2**2) + sp.Rational(1, 2) * kp * (y1 - y2)**2
L = T - V

# Euler-Lagrange equation for each variable
for yi, name in [(y1, 'y₁'), (y2, 'y₂')]:
    yip = yi.diff(x)
    dL_dyi = sp.diff(L, yi)
    dL_dyip = sp.diff(L, yip)
    d_dx = sp.diff(dL_dyip, x)
    eq = sp.simplify(dL_dyi - d_dx)
    print(f"E-L for {name}: {eq} = 0")
```

---

## 5. Constrained Variational Problems

### 5.1 Isoperimetric Constraints

Extremize the functional $J[y] = \int_a^b F(x, y, y')\, dx$ subject to:

$$K[y] = \int_a^b G(x, y, y')\, dx = \ell \quad \text{(given constant)}$$

**Lagrange multiplier method**: Define auxiliary functional $\bar{J} = J + \lambda K$ and apply the Euler-Lagrange equation to $\bar{F} = F + \lambda G$:

$$\frac{\partial \bar{F}}{\partial y} - \frac{d}{dx}\frac{\partial \bar{F}}{\partial y'} = 0$$

The unknown $\lambda$ is determined from the constraint $K = \ell$.

### 5.2 Holonomic Constraints

Algebraic constraints of the form $g(x, y_1, y_2) = 0$:

$$\frac{\partial F}{\partial y_i} - \frac{d}{dx}\frac{\partial F}{\partial y_i'} + \lambda(x)\frac{\partial g}{\partial y_i} = 0$$

where $\lambda(x)$ is a multiplier function that may vary point-by-point.

### 5.3 Example: Maximum Area Curve with Fixed Length

Maximize the area under a curve with given length $L$:

$$J[y] = \int_0^a y\, dx, \quad K[y] = \int_0^a \sqrt{1 + y'^2}\, dx = L$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def isoperimetric_circle(a, L):
    """Isoperimetric problem: curve of maximum area on [0, a] with length L (circular arc)"""
    # Circular arc: radius R, center (a/2, y_c)
    # Determine R from L = 2R·arcsin(a/(2R))
    def eq(R):
        if R < a / 2:
            return 1e10
        return 2 * R * np.arcsin(a / (2 * R)) - L

    R = fsolve(eq, L / np.pi)[0]

    theta = np.linspace(-np.arcsin(a / (2 * R)), np.arcsin(a / (2 * R)), 300)
    x_arc = R * np.sin(theta) + a / 2
    y_arc = R * np.cos(theta) - R * np.cos(np.arcsin(a / (2 * R)))

    return x_arc, y_arc, R

# Interval length 2, curve length π
a, L = 2.0, np.pi
x_arc, y_arc, R = isoperimetric_circle(a, L)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_arc, y_arc, 'r-', linewidth=2, label=f'Circular arc (R={R:.3f})')
ax.fill_between(x_arc, 0, y_arc, alpha=0.2, color='red')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_title('Isoperimetric problem: maximum area for given length')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('isoperimetric.png', dpi=150)
plt.show()
```

---

## 6. Lagrangian Mechanics

### 6.1 Principle of Least Action

**Hamilton's principle**: The actual path of a mechanical system is the one that makes the **action** stationary.

$$S[q] = \int_{t_1}^{t_2} L(q, \dot{q}, t)\, dt, \quad \delta S = 0$$

**Lagrangian**: $L = T - V$ (kinetic energy - potential energy)

**Generalized coordinates** $q_i$: Independent variables describing the system's degrees of freedom. They automatically handle constraints.

The **Lagrange equations of motion** for each generalized coordinate:

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0, \quad i = 1, 2, \ldots, n$$

### 6.2 Simple Pendulum

Generalized coordinate: angle $\theta$

$$T = \frac{1}{2}ml^2\dot{\theta}^2, \quad V = -mgl\cos\theta$$

$$L = \frac{1}{2}ml^2\dot{\theta}^2 + mgl\cos\theta$$

Euler-Lagrange equation: $ml^2\ddot{\theta} + mgl\sin\theta = 0$, i.e., $\ddot{\theta} + \frac{g}{l}\sin\theta = 0$

```python
import sympy as sp

t = sp.Symbol('t')
m, l, g = sp.symbols('m l g', positive=True)
theta = sp.Function('theta')(t)
theta_dot = theta.diff(t)

# Lagrangian
T = sp.Rational(1, 2) * m * l**2 * theta_dot**2
V = -m * g * l * sp.cos(theta)
Lag = T - V

# Derive Euler-Lagrange equation
dL_dtheta = sp.diff(Lag, theta)
dL_dthetadot = sp.diff(Lag, theta_dot)
d_dt = sp.diff(dL_dthetadot, t)

eq = sp.simplify(d_dt - dL_dtheta)
print("=== Simple pendulum equation of motion ===")
print(f"  {eq} = 0")
# Simplified: ml²θ̈ + mgl·sin(θ) = 0
```

### 6.3 Double Pendulum

Generalized coordinates: $\theta_1, \theta_2$

$$x_1 = l_1\sin\theta_1, \quad y_1 = -l_1\cos\theta_1$$

$$x_2 = l_1\sin\theta_1 + l_2\sin\theta_2, \quad y_2 = -l_1\cos\theta_1 - l_2\cos\theta_2$$

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def double_pendulum_eom(t, state, m1, m2, l1, l2, g=9.81):
    """Equations of motion for the double pendulum (derived via Lagrangian mechanics)
    state = [theta1, theta2, omega1, omega2]
    """
    th1, th2, w1, w2 = state
    dth = th1 - th2
    cos_d, sin_d = np.cos(dth), np.sin(dth)
    M = m1 + m2

    # Angular accelerations using inverse of mass matrix
    den = M * l1 - m2 * l1 * cos_d**2

    alpha1 = (-m2 * l2 * w2**2 * sin_d
              - m2 * g * np.sin(th2) * cos_d
              + M * g * np.sin(th1)) / (-den)

    den2 = (l2 / l1) * den

    alpha2 = (M * l1 * w1**2 * sin_d
              + M * g * np.sin(th1) * cos_d
              - M * g * np.sin(th2)) / den2

    return [w1, w2, alpha1, alpha2]

# Simulation
m1, m2, l1, l2 = 1.0, 1.0, 1.0, 1.0
state0 = [np.pi / 2, np.pi / 2, 0, 0]  # initial angles 90°, angular velocities 0
t_span = (0, 20)
t_eval = np.linspace(*t_span, 2000)

sol = solve_ivp(double_pendulum_eom, t_span, state0, t_eval=t_eval,
                args=(m1, m2, l1, l2), method='RK45', rtol=1e-10)

# Convert to Cartesian coordinates
x1 = l1 * np.sin(sol.y[0])
y1_pos = -l1 * np.cos(sol.y[0])
x2 = x1 + l2 * np.sin(sol.y[1])
y2_pos = y1_pos - l2 * np.cos(sol.y[1])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trajectory of tip
axes[0].plot(x2, y2_pos, 'b-', linewidth=0.3, alpha=0.7)
axes[0].set_title('Double pendulum: tip trajectory')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Angle vs time
axes[1].plot(sol.t, sol.y[0], label=r'$\theta_1$')
axes[1].plot(sol.t, sol.y[1], label=r'$\theta_2$')
axes[1].set_title('Double pendulum: angle vs time')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Angle (rad)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum.png', dpi=150)
plt.show()
```

### 6.4 Central Force Problem

Generalized coordinates: $(r, \theta)$ (polar coordinates)

$$L = \frac{1}{2}m(\dot{r}^2 + r^2\dot{\theta}^2) - V(r)$$

Since $\theta$ is a cyclic coordinate ($L$ does not explicitly depend on $\theta$):

$$p_\theta = \frac{\partial L}{\partial \dot{\theta}} = mr^2\dot{\theta} = \text{const} \quad \text{(angular momentum conservation)}$$

### 6.5 Atwood Machine

Two masses $m_1, m_2$ connected by a string over a pulley. Generalized coordinate: $x$ (displacement of one mass).

$$T = \frac{1}{2}(m_1 + m_2)\dot{x}^2, \quad V = -(m_1 - m_2)gx$$

$$L = \frac{1}{2}(m_1 + m_2)\dot{x}^2 + (m_1 - m_2)gx$$

Euler-Lagrange equation: $(m_1 + m_2)\ddot{x} = (m_1 - m_2)g$, i.e., $\ddot{x} = \frac{m_1 - m_2}{m_1 + m_2}g$

### 6.6 Noether's Theorem

If the Lagrangian is invariant under a continuous transformation, there exists a corresponding conserved quantity:

| Symmetry (transformation) | Conserved quantity |
|-------------|--------|
| Time translation ($t \to t + \epsilon$) | Energy |
| Space translation ($x \to x + \epsilon$) | Momentum |
| Rotation ($\theta \to \theta + \epsilon$) | Angular momentum |

---

## 7. Hamiltonian Mechanics Basics

### 7.1 Legendre Transform

Transformation from Lagrangian $L(q, \dot{q}, t)$ to **Hamiltonian** $H(q, p, t)$:

Generalized momentum: $p_i = \frac{\partial L}{\partial \dot{q}_i}$

$$\boxed{H(q, p, t) = \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)}$$

Here $\dot{q}_i$ is expressed in terms of $q, p$ by inverting $p_i = \partial L / \partial \dot{q}_i$.

For conservative systems (time-independent), $H = T + V$ (total energy).

### 7.2 Hamilton's Canonical Equations

$$\boxed{\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}}$$

While Lagrange equations are $n$ second-order ODEs, Hamilton equations are $2n$ first-order ODEs.

### 7.3 Phase Space and Poisson Brackets

**Phase space**: $2n$-dimensional space consisting of $(q_1, \ldots, q_n, p_1, \ldots, p_n)$.

**Poisson bracket**:

$$\{A, B\} = \sum_i \left(\frac{\partial A}{\partial q_i}\frac{\partial B}{\partial p_i} - \frac{\partial A}{\partial p_i}\frac{\partial B}{\partial q_i}\right)$$

Time evolution: $\dot{A} = \{A, H\} + \frac{\partial A}{\partial t}$. If conserved, $\{A, H\} = 0$.

### 7.4 Connection to Quantum Mechanics

In quantum mechanics, Poisson brackets are replaced by commutators:

$$\{A, B\}_{\text{Poisson}} \to \frac{1}{i\hbar}[\hat{A}, \hat{B}]$$

The canonical commutation relation $\{q, p\} = 1$ corresponds to quantum mechanics' $[\hat{q}, \hat{p}] = i\hbar$. This is the starting point of **canonical quantization**.

```python
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Symbolic: Hamiltonian mechanics of 1D harmonic oscillator ===
t = sp.Symbol('t')
m_sym, k_sym = sp.symbols('m k', positive=True)
q_sym = sp.Function('q')(t)
p_sym = sp.Function('p')(t)

# Hamiltonian: H = p²/(2m) + kq²/2
H = p_sym**2 / (2 * m_sym) + k_sym * q_sym**2 / 2

# Hamilton's canonical equations
q_dot = sp.diff(H, p_sym)      # dq/dt = ∂H/∂p = p/m
p_dot = -sp.diff(H, q_sym)     # dp/dt = -∂H/∂q = -kq

print("=== Harmonic oscillator Hamilton equations ===")
print(f"  dq/dt = {q_dot}")
print(f"  dp/dt = {p_dot}")

# === Numerical solution: phase space trajectories ===
def harmonic_hamiltonian(t, state, m=1.0, k=1.0):
    q, p = state
    dqdt = p / m       # ∂H/∂p
    dpdt = -k * q      # -∂H/∂q
    return [dqdt, dpdt]

# Phase space trajectories for various initial conditions
fig, ax = plt.subplots(figsize=(7, 7))
for E0 in [0.5, 1.0, 2.0, 4.0]:
    q0 = np.sqrt(2 * E0)  # maximum displacement when p0 = 0
    sol = solve_ivp(harmonic_hamiltonian, (0, 10), [q0, 0],
                    t_eval=np.linspace(0, 10, 1000))
    ax.plot(sol.y[0], sol.y[1], label=f'E = {E0}')

ax.set_xlabel('q (position)')
ax.set_ylabel('p (momentum)')
ax.set_title('Harmonic oscillator phase space')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('phase_space.png', dpi=150)
plt.show()
```

---

## 8. Modern Applications of Calculus of Variations

### 8.1 Fermat's Principle (Optics)

Light travels between two points along the path where the **optical path length** is stationary:

$$\delta \int_A^B n(x, y)\, ds = 0$$

where $n$ is the refractive index. This derives Snell's law:

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

```python
import numpy as np
import matplotlib.pyplot as plt

def snell_law_demo():
    """Visualize Fermat's principle and Snell's law"""
    n1, n2 = 1.0, 1.5  # refractive indices (air → glass)
    y_source, y_target = 2.0, -2.0
    x_source, x_target = -1.0, 1.5

    # Compute optical path length varying incidence point x on boundary
    x_boundary = np.linspace(-2, 3, 1000)
    optical_path = []
    for xb in x_boundary:
        d1 = np.sqrt((xb - x_source)**2 + y_source**2)
        d2 = np.sqrt((x_target - xb)**2 + y_target**2)
        opl = n1 * d1 + n2 * d2
        optical_path.append(opl)

    optical_path = np.array(optical_path)
    x_opt = x_boundary[np.argmin(optical_path)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Optical path length vs incidence point
    axes[0].plot(x_boundary, optical_path, 'b-')
    axes[0].axvline(x_opt, color='r', linestyle='--', label=f'minimum x={x_opt:.3f}')
    axes[0].set_xlabel('Incidence point x on boundary')
    axes[0].set_ylabel('Optical path length')
    axes[0].set_title("Fermat's principle: minimizing optical path length")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) Light path
    axes[1].fill_between([-3, 4], 0, 3, alpha=0.1, color='blue', label=f'Medium 1 (n={n1})')
    axes[1].fill_between([-3, 4], -3, 0, alpha=0.1, color='orange', label=f'Medium 2 (n={n2})')
    axes[1].plot([x_source, x_opt], [y_source, 0], 'r-', linewidth=2)
    axes[1].plot([x_opt, x_target], [0, y_target], 'r-', linewidth=2)
    axes[1].axhline(y=0, color='k', linewidth=1)
    axes[1].plot(x_source, y_source, 'ko', markersize=8)
    axes[1].plot(x_target, y_target, 'ko', markersize=8)
    axes[1].set_title("Snell's law (Fermat's principle)")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fermat_snell.png', dpi=150)
    plt.show()

    # Verify Snell's law
    theta1 = np.arctan2(abs(x_opt - x_source), y_source)
    theta2 = np.arctan2(abs(x_target - x_opt), abs(y_target))
    print(f"Incidence angle: {np.degrees(theta1):.2f}°, Refraction angle: {np.degrees(theta2):.2f}°")
    print(f"n₁ sin θ₁ = {n1 * np.sin(theta1):.4f}")
    print(f"n₂ sin θ₂ = {n2 * np.sin(theta2):.4f}")

snell_law_demo()
```

### 8.2 Rayleigh-Ritz Method

A **direct method** for finding approximate solutions to variational problems. Assume the solution as a linear combination of basis functions:

$$y(x) \approx \sum_{i=1}^{N} c_i \phi_i(x)$$

Convert the functional into a function of $c_i$ and solve $\partial J / \partial c_i = 0$ to determine coefficients.

```python
import numpy as np
import matplotlib.pyplot as plt

def rayleigh_ritz_beam(N_basis=5, L_beam=1.0, q0=1.0, EI=1.0):
    """
    Compute deflection of a simply supported beam using the Rayleigh-Ritz method.
    Functional: J[y] = ∫[EI/2·(y'')² - q₀·y]dx
    Boundary conditions: y(0) = y(L) = 0, y''(0) = y''(L) = 0
    Basis functions: φₙ(x) = sin(nπx/L)
    """
    x = np.linspace(0, L_beam, 500)

    # Exact solution: y = q₀/(24EI) · x(L³ - 2Lx² + x³)
    y_exact = q0 / (24 * EI) * x * (L_beam**3 - 2 * L_beam * x**2 + x**3)

    # Rayleigh-Ritz approximation
    y_approx = np.zeros_like(x)
    coefficients = []

    for n in range(1, N_basis + 1):
        kn = n * np.pi / L_beam
        # Diagonal entry of stiffness matrix
        K_nn = EI * kn**4 * L_beam / 2
        # Load vector: only odd n contribute
        if n % 2 == 1:
            f_n = 2 * q0 * L_beam / (n * np.pi)
        else:
            f_n = 0
        c_n = f_n / K_nn
        coefficients.append(c_n)
        y_approx += c_n * np.sin(n * np.pi * x / L_beam)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Deflection comparison
    axes[0].plot(x, y_exact, 'r-', linewidth=2, label='Exact solution')
    axes[0].plot(x, y_approx, 'b--', linewidth=2, label=f'R-R ({N_basis} terms)')
    axes[0].set_title('Rayleigh-Ritz method: beam deflection')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y(x)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Convergence analysis
    errors = []
    N_range = range(1, 20)
    for N in N_range:
        y_rr = np.zeros_like(x)
        for n in range(1, N + 1):
            kn = n * np.pi / L_beam
            K_nn = EI * kn**4 * L_beam / 2
            f_n = 2 * q0 * L_beam / (n * np.pi) if n % 2 == 1 else 0
            y_rr += (f_n / K_nn) * np.sin(n * np.pi * x / L_beam)
        err = np.max(np.abs(y_rr - y_exact))
        errors.append(err)

    axes[1].semilogy(list(N_range), errors, 'ko-')
    axes[1].set_title('Error convergence vs number of basis functions')
    axes[1].set_xlabel('Number of basis functions N')
    axes[1].set_ylabel('Maximum error')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rayleigh_ritz.png', dpi=150)
    plt.show()

    return coefficients

coeffs = rayleigh_ritz_beam(N_basis=7)
print("Rayleigh-Ritz coefficients:", [f"{c:.6e}" for c in coeffs])
```

### 8.3 Optimal Control Theory (Overview)

**Problem**: Minimize a cost functional with respect to state $\mathbf{x}(t)$ and control $\mathbf{u}(t)$:

$$J = \int_{t_0}^{t_f} L(\mathbf{x}, \mathbf{u}, t)\, dt$$

Constraint: $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, t)$ (state equation)

**Pontryagin's maximum principle**: Define Hamiltonian $H = L + \boldsymbol{\lambda}^T \mathbf{f}$ and the optimal control minimizes $H$ with respect to $\mathbf{u}$.

### 8.4 Connection to Finite Element Method

The weak form of variational problems is the starting point of the finite element method (FEM). For the Dirichlet problem:

$$\text{Minimize: } J[u] = \frac{1}{2}\int_\Omega |\nabla u|^2\, d\Omega - \int_\Omega fu\, d\Omega$$

the Euler-Lagrange equation is $-\nabla^2 u = f$ (Poisson equation). FEM discretizes this variational problem with finite-dimensional basis functions to obtain approximate solutions.

---

## Practice Problems

### Basic Problems

1. Find the Euler-Lagrange equations for the following functionals:
   - (a) $J[y] = \int_0^1 (y'^2 + 2yy')\, dx$, $y(0)=0$, $y(1)=1$
   - (b) $J[y] = \int_0^{\pi} (y'^2 - y^2)\, dx$, $y(0)=0$, $y(\pi)=0$
   - (c) $J[y] = \int_1^2 \frac{\sqrt{1 + y'^2}}{x}\, dx$

2. Use the Beltrami identity to find the extremal curve of $J[y] = \int_0^1 (y'^2 + y^2)\, dx$. ($y(0)=0$, $y(1)=1$)

3. Find the curve connecting two points $(0, 0)$ and $(a, 0)$ with given arc length $L$ that maximizes the area with the $x$-axis.

### Intermediate Problems

4. Prove that geodesics on a sphere are great circles using the Euler-Lagrange equation. (Hint: Use $\phi$ as the independent variable)

5. **Brachistochrone**: Find the cycloid from point $(0, 0)$ to $(1, 1)$ parametrically and compare the travel time with a straight path.

6. Derive the Lagrangian for a double pendulum and find the equations of motion for $\theta_1$, $\theta_2$.

7. For a one-dimensional harmonic oscillator $L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2$:
   - (a) Find the Lagrange equation of motion
   - (b) Transform to Hamiltonian form
   - (c) Show that Hamilton's canonical equations agree with (a)

### Advanced Problems

8. **Elastic beam**: Model the bending of a cantilever beam as $J[y] = \int_0^L \left[\frac{EI}{2}(y'')^2 - qy\right] dx$. Find the solution with boundary conditions $y(0) = y'(0) = 0$, $y''(L) = y'''(L) = 0$.

9. **Variational principle for electric field**: For a given charge distribution $\rho(\mathbf{r})$, show that the Euler-Lagrange equation of the functional $J[\phi] = \int \left[\frac{\epsilon_0}{2}|\nabla\phi|^2 + \rho\phi\right] d^3r$ is the Poisson equation.

10. Use the Rayleigh-Ritz method to find an approximate solution of $y'' + y = 1$ ($y(0) = y(\pi) = 0$) as $y \approx c_1\sin x + c_2\sin 2x + c_3\sin 3x$ and compare with the exact solution.

---

## Advanced Topics

### Second Variation and Sufficient Conditions

The first variation $\delta J = 0$ is a **necessary condition**. To verify it is actually a minimum, examine the **second variation**:

$$\delta^2 J = \int_a^b \left(P\eta^2 + Q\eta'^2\right) dx$$

where $P = F_{yy} - \frac{d}{dx}F_{yy'}$, $Q = F_{y'y'}$. If $\delta^2 J > 0$, it's a minimum; if $\delta^2 J < 0$, a maximum.

**Legendre condition**: $F_{y'y'} > 0$ is a necessary condition for weak minimum.

**Jacobi condition**: If the solution of the Jacobi equation $(Q\eta')' - P\eta = 0$ has no zeros in the interval $(a, b)$, it is a sufficient condition for strong minimum.

### Direct Methods

- **Ritz method**: Minimize the functional in a finite-dimensional subspace
- **Galerkin method**: Set the residual orthogonal to test functions
- **Finite element method (FEM)**: Ritz/Galerkin method using piecewise basis functions

### Variational Inequalities

Problems with inequality constraints such as $y(x) \geq \psi(x)$ (obstacle problems). Closely related to free boundary problems.

### References

1. **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 9
2. **Gelfand, I. M. & Fomin, S. V.** *Calculus of Variations* (Dover) -- classic textbook on calculus of variations
3. **Goldstein, Poole, Safko** *Classical Mechanics*, 3rd ed. -- Lagrangian/Hamiltonian mechanics
4. **Lanczos, C.** *The Variational Principles of Mechanics* (Dover) -- physical interpretation of variational principles
5. **Arfken, Weber** *Mathematical Methods for Physicists*, 7th ed., Ch. 22

### Summary of Key Formulas

| Formula | Description |
|------|------|
| $\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0$ | Euler-Lagrange equation |
| $F - y'F_{y'} = C$ | Beltrami identity ($F$ independent of $x$) |
| $F_{y'} = C$ | When $F$ is independent of $y$ |
| $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$ | Lagrange equation of motion |
| $H = \sum p_i\dot{q}_i - L$ | Hamiltonian definition |
| $\dot{q} = \partial H/\partial p$, $\dot{p} = -\partial H/\partial q$ | Hamilton canonical equations |
| $x = R(\theta - \sin\theta)$, $y = R(1-\cos\theta)$ | Cycloid (brachistochrone) |

---

**Previous**: [16. Green's Functions](16_Greens_Functions.md)
**Next**: [18. Tensor Analysis](18_Tensor_Analysis.md)

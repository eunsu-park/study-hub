# 14. Complex Analysis

## Learning Objectives

- Understand the differentiability of complex functions and the Cauchy-Riemann conditions
- Calculate complex integrals using Cauchy's integral theorem and integral formulas
- Expand complex functions into series through Taylor and Laurent series
- Efficiently solve real integral problems using the residue theorem
- Understand the concept of conformal mapping and apply it to physics problems

> **Importance in Physics**: Complex analysis is a core tool across all of physics, including wave functions in quantum mechanics, potential theory in electrodynamics, stream functions in fluid mechanics, and frequency analysis in signal processing. In particular, integral calculations via the residue theorem are among the most frequently used techniques in theoretical physics.

---

## 1. Analytic Functions

### 1.1 Complex Differentiation and the Cauchy-Riemann Conditions

For a complex function $f(z) = u(x, y) + iv(x, y)$ to be **differentiable** at a point $z_0$, the limit

$$f'(z_0) = \lim_{\Delta z \to 0} \frac{f(z_0 + \Delta z) - f(z_0)}{\Delta z}$$

must converge to the same value regardless of the direction from which $\Delta z$ approaches. This leads to the **Cauchy-Riemann equations**:

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

A function that is differentiable at every point in a region is called **analytic** in that region.

**Polar form** ($z = re^{i\theta}$):

$$\frac{\partial u}{\partial r} = \frac{1}{r}\frac{\partial v}{\partial \theta}, \quad \frac{1}{r}\frac{\partial u}{\partial \theta} = -\frac{\partial v}{\partial r}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def check_cauchy_riemann(u_func, v_func, x, y, h=1e-7):
    """Numerically verify the Cauchy-Riemann conditions"""
    du_dx = (u_func(x + h, y) - u_func(x - h, y)) / (2 * h)
    du_dy = (u_func(x, y + h) - u_func(x, y - h)) / (2 * h)
    dv_dx = (v_func(x + h, y) - v_func(x - h, y)) / (2 * h)
    dv_dy = (v_func(x, y + h) - v_func(x, y - h)) / (2 * h)

    cond1 = np.abs(du_dx - dv_dy)   # ∂u/∂x = ∂v/∂y
    cond2 = np.abs(du_dy + dv_dx)   # ∂u/∂y = -∂v/∂x

    print(f"Point ({x}, {y}): |∂u/∂x - ∂v/∂y| = {cond1:.2e}, "
          f"|∂u/∂y + ∂v/∂x| = {cond2:.2e}")
    return cond1 < 1e-5 and cond2 < 1e-5

# f(z) = z² = (x² - y²) + i(2xy)
u = lambda x, y: x**2 - y**2
v = lambda x, y: 2 * x * y

print("=== Cauchy-Riemann verification for f(z) = z² ===")
for pt in [(1, 1), (2, -1), (0.5, 3)]:
    analytic = check_cauchy_riemann(u, v, *pt)
    print(f"  Analytic: {analytic}")
```

### 1.2 Harmonic Functions

The real part $u$ and imaginary part $v$ of an analytic function each satisfy **Laplace's equation**:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0, \quad \nabla^2 v = 0$$

Such functions are called **harmonic functions**, and $u$ and $v$ are **harmonic conjugates** of each other.

**Physical meaning**: In two-dimensional electrostatics, the potential $\phi(x,y)$ satisfies Laplace's equation and is therefore a harmonic function. The real part of an analytic function can be interpreted as the potential and the imaginary part as the electric field line function.

```python
import sympy as sp

x, y = sp.symbols('x y', real=True)

def is_harmonic(expr):
    """Check if the Laplacian is 0"""
    lap = sp.simplify(sp.diff(expr, x, 2) + sp.diff(expr, y, 2))
    return lap, lap == 0

# Real and imaginary parts of f(z) = z³
u_expr = x**3 - 3*x*y**2      # Re(z³)
v_expr = 3*x**2*y - y**3      # Im(z³)

for name, expr in [("Re(z³)", u_expr), ("Im(z³)", v_expr)]:
    lap, harmonic = is_harmonic(expr)
    print(f"{name} = {expr}: ∇² = {lap}, harmonic: {harmonic}")
```

### 1.3 Examples of Analytic Functions

Analytic functions frequently appearing in physics:

| Function | Real part $u$ | Imaginary part $v$ | Physical application |
|------|-----------|-----------|------------|
| $e^z$ | $e^x \cos y$ | $e^x \sin y$ | Waves, damping |
| $\ln z$ | $\ln r$ | $\theta$ | Line charge potential |
| $z^n$ | $r^n \cos n\theta$ | $r^n \sin n\theta$ | Multipole expansion |
| $1/z$ | $x/(x^2+y^2)$ | $-y/(x^2+y^2)$ | Point charge, source/sink |

```python
from matplotlib.colors import hsv_to_rgb

def domain_coloring(f, xlim=(-2, 2), ylim=(-2, 2), N=500):
    """Domain coloring visualization of complex function (phase→color, magnitude→brightness)"""
    xv = np.linspace(*xlim, N)
    yv = np.linspace(*ylim, N)
    X, Y = np.meshgrid(xv, yv)
    Z = X + 1j * Y

    with np.errstate(divide='ignore', invalid='ignore'):
        W = f(Z)

    H = (np.angle(W) + np.pi) / (2 * np.pi)
    V = 1 - 1 / (1 + np.abs(W)**0.3)
    HSV = np.stack([H, np.ones_like(H), V], axis=-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(hsv_to_rgb(HSV), extent=[*xlim, *ylim], origin='lower')
    ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
    plt.tight_layout(); plt.show()

# domain_coloring(lambda z: z**3)       # domain coloring of z³
# domain_coloring(lambda z: np.exp(z))  # domain coloring of e^z
```

---

## 2. Complex Integrals

### 2.1 Contour Integrals

Integrating a complex function along a path $C$ is called a **contour integral**:

$$\oint_C f(z)\, dz = \int_a^b f(z(t))\, z'(t)\, dt$$

**Key result**: For counterclockwise integration over $|z - z_0| = r$:

$$\oint \frac{dz}{(z - z_0)^n} = \begin{cases} 2\pi i & n = 1 \\ 0 & n \neq 1 \end{cases}$$

```python
def contour_integral_circle(f, z0=0, r=1, N=10000):
    """Contour integral over circular path |z-z0|=r (numerical computation)"""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    dt = 2*np.pi / N
    z = z0 + r * np.exp(1j * t)
    dz_dt = 1j * r * np.exp(1j * t)
    return np.sum(f(z) * dz_dt) * dt

# ∮ 1/z dz = 2πi
I1 = contour_integral_circle(lambda z: 1/z)
print(f"∮ 1/z dz = {I1:.6f}  (theoretical: {2*np.pi*1j:.6f})")

# ∮ 1/z² dz = 0
I2 = contour_integral_circle(lambda z: 1/z**2)
print(f"∮ 1/z² dz = {I2:.6f}  (theoretical: 0)")

# ∮ e^z/z dz = 2πi (Cauchy's formula: f(0)=e⁰=1)
I3 = contour_integral_circle(lambda z: np.exp(z)/z)
print(f"∮ e^z/z dz = {I3:.6f}  (theoretical: {2*np.pi*1j:.6f})")
```

### 2.2 Cauchy's Integral Theorem

If $f(z)$ is analytic in a simply connected region $D$, then for any closed curve $C$ in $D$:

$$\oint_C f(z)\, dz = 0$$

**Physical meaning**: This is equivalent to the circulation of a conservative force being zero. The integral of an analytic function is path-independent.

### 2.3 Cauchy's Integral Formula

If $f(z)$ is analytic inside $C$ and $z_0$ is an interior point:

$$f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0}\, dz$$

**Generalization** (nth derivative):

$$f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^{n+1}}\, dz$$

This formula means that analytic functions are infinitely differentiable, and boundary values determine interior values.

```python
from math import factorial

def cauchy_derivative(f, z0, n=0, r=1, N=10000):
    """Compute f^(n)(z0) using Cauchy's integral formula"""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    dt = 2*np.pi / N
    z = z0 + r * np.exp(1j * t)
    dz_dt = 1j * r * np.exp(1j * t)
    integrand = f(z) / (z - z0)**(n + 1) * dz_dt
    return np.sum(integrand) * dt * factorial(n) / (2*np.pi*1j)

# Verification: f(z) = sin(z)
f = lambda z: np.sin(z)
print("Cauchy formula verification: f(z) = sin(z)")
print(f"  f(0)   = {cauchy_derivative(f, 0, 0):.6f}  (exact: 0)")
print(f"  f'(0)  = {cauchy_derivative(f, 0, 1):.6f}  (exact: 1)")
print(f"  f''(0) = {cauchy_derivative(f, 0, 2):.6f}  (exact: 0)")
print(f"  f'''(0)= {cauchy_derivative(f, 0, 3):.6f}  (exact: -1)")
```

---

## 3. Series Expansions

### 3.1 Taylor Series

If $f(z)$ is analytic inside a circle of radius $R$ centered at $z_0$:

$$f(z) = \sum_{n=0}^{\infty} a_n (z - z_0)^n, \quad a_n = \frac{f^{(n)}(z_0)}{n!}$$

The radius of convergence $R$ is the distance from $z_0$ to the nearest singularity.

**Example**: The radius of convergence of the Taylor series of $f(z) = 1/(1+z^2)$ around $z=0$ is $R=1$. This is because there are poles at $z = \pm i$. Although there are no singularities on the real axis, singularities in the complex plane determine the radius of convergence.

### 3.2 Laurent Series

If $f(z)$ is analytic in an annular region $r < |z - z_0| < R$:

$$f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n$$

Terms with $n < 0$ are called the **principal part**, and specifically $a_{-1}$ is called the **residue**.

```python
z = sp.Symbol('z')

# Laurent series: e^z / z³ (around z=0)
f1 = sp.exp(z) / z**3
print("Laurent series of e^z/z³:")
print(f"  {sp.series(f1, z, 0, n=5)}")
print(f"  Residue (z⁻¹ coefficient) = 1/2\n")

# Laurent series: 1/(z(z-1)) (around z=0)
f2 = 1 / (z * (z - 1))
print("Laurent series of 1/(z(z-1)) (around z=0):")
print(f"  {sp.series(f2, z, 0, n=4)}")

# around z=1
w = sp.Symbol('w')
print("\nLaurent series of 1/(z(z-1)) (around z=1, w=z-1):")
print(f"  {sp.series(f2.subs(z, w+1), w, 0, n=4)}")
```

### 3.3 Classification of Singularities (Removable, Pole, Essential)

| Type | Number of principal part terms | Example | $\lim_{z \to z_0} f(z)$ |
|------|-------------|------|------------------------|
| **Removable singularity** | 0 | $\sin z / z$ at $z=0$ | Finite value |
| **Pole of order $m$** | $m$ | $1/z^m$ at $z=0$ | $\infty$ |
| **Essential singularity** | Infinitely many | $e^{1/z}$ at $z=0$ | Does not exist |

**Casorati-Weierstrass theorem**: Near an essential singularity, the function takes on almost every complex value.

```python
# Singularity classification verification
cases = [
    ("sin(z)/z (z=0, removable)", sp.sin(z)/z, z, 0),
    ("1/(z-1)³ (z=1, pole of order 3)", 1/(z-1)**3, z, 1),
    ("exp(1/z) (z=0, essential)", sp.exp(1/z), z, 0),
]

for name, expr, var, pt in cases:
    if pt != 0:
        w = sp.Symbol('w')
        s = sp.series(expr.subs(var, w + pt), w, 0, n=5)
    else:
        s = sp.series(expr, var, 0, n=5)
    print(f"{name}:\n  {s}\n")
```

---

## 4. Residue Theorem

### 4.1 Definition and Calculation of Residues

The **residue** of $f(z)$ at the point $z_0$ is the $a_{-1}$ coefficient of the Laurent series:

$$\text{Res}_{z=z_0} f(z) = \frac{1}{2\pi i} \oint_C f(z)\, dz$$

**Methods for calculating residues**:

1. **Simple pole**: $\text{Res}_{z=z_0} f = \lim_{z \to z_0} (z - z_0) f(z)$
2. **Pole of order $m$**: $\text{Res}_{z=z_0} f = \frac{1}{(m-1)!} \lim_{z \to z_0} \frac{d^{m-1}}{dz^{m-1}} [(z - z_0)^m f(z)]$
3. **Form $p/q$** (simple pole): $\text{Res}_{z=z_0} \frac{p}{q} = \frac{p(z_0)}{q'(z_0)}$

```python
z = sp.Symbol('z')

examples = [
    ("1/(z²+1)", 1/(z**2+1), sp.I),
    ("1/(z²+1)", 1/(z**2+1), -sp.I),
    ("e^z/z²", sp.exp(z)/z**2, 0),
    ("z/(z²-3z+2)", z/(z**2-3*z+2), 1),
    ("z/(z²-3z+2)", z/(z**2-3*z+2), 2),
]

print("=== Residue calculation ===")
for name, expr, z0 in examples:
    print(f"Res[{name}, z={z0}] = {sp.residue(expr, z, z0)}")
```

### 4.2 Residue Theorem

If $f(z)$ is analytic inside a closed curve $C$ except for a finite number of singularities $z_1, \ldots, z_n$:

$$\oint_C f(z)\, dz = 2\pi i \sum_{k=1}^{n} \text{Res}_{z=z_k} f(z)$$

### 4.3 Jordan's Lemma

When calculating real integrals using the residue theorem, we must verify that the contribution from the infinite semicircular path is zero. **Jordan's lemma** guarantees this.

**Theorem**: If $f(z) \to 0$ uniformly as $|z| \to \infty$ (upper half-plane), then for $a > 0$:

$$\lim_{R \to \infty} \int_{C_R} f(z) e^{iaz}\, dz = 0$$

where $C_R$ is a semicircle of radius $R$ in the upper half-plane. The key is that $e^{iaz} = e^{ia(x+iy)} = e^{iax}e^{-ay}$, so in the upper half-plane ($y > 0$), $e^{-ay}$ decays exponentially.

> **Note**: If $a < 0$, use the lower half-plane semicircle, and in this case the path direction is clockwise so $(-2\pi i)$ is multiplied to the residues.

### 4.4 Four Types of Real Integral Calculations

The most important application of the residue theorem is the calculation of real integrals. We systematically classify them according to the form of the integral.

#### Type 1: Rational functions of trigonometric functions — $\int_0^{2\pi} R(\cos\theta, \sin\theta)\, d\theta$

Substitution $z = e^{i\theta}$: $\cos\theta = (z + z^{-1})/2$, $\sin\theta = (z - z^{-1})/(2i)$, $d\theta = dz/(iz)$.

The integral transforms into a contour integral over the unit circle $|z| = 1$. Calculate only the residues at poles **inside** the unit circle.

```python
from scipy.integrate import quad
import sympy as sp
import numpy as np

z = sp.Symbol('z')

# --- Type 1: ∫₀²π dθ/(2 + cosθ) ---
print("=== Type 1: ∫₀²π dθ/(2 + cosθ) ===")
integrand1 = 2 / (sp.I * (z**2 + 4*z + 1))
inner_pole = -2 + sp.sqrt(3)  # pole with |z| < 1
res1 = sp.residue(integrand1, z, inner_pole)
result1 = sp.simplify(2 * sp.pi * sp.I * res1)
print(f"Residue theorem: {result1} = {float(result1):.6f}")
num1, _ = quad(lambda t: 1/(2 + np.cos(t)), 0, 2*np.pi)
print(f"Numerical verification: {num1:.6f}\n")
```

#### Type 2: Rational functions — $\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)}\, dx$

Condition: $\deg(Q) \geq \deg(P) + 2$ (integral converges), $Q(x) \neq 0$ on the real axis.

Use an upper half-plane semicircular path. As $R \to \infty$, the integral over the semicircle becomes 0 (since $f(z) \to 0$ sufficiently fast):

$$\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)}\, dx = 2\pi i \sum_{\text{Im}(z_k) > 0} \text{Res}_{z=z_k} \frac{P(z)}{Q(z)}$$

```python
# --- Type 2: ∫₋∞^∞ dx/(x²+1)² ---
print("=== Type 2: ∫₋∞^∞ dx/(x²+1)² ===")
res2 = sp.residue(1/(z**2+1)**2, z, sp.I)
result2 = sp.simplify(2 * sp.pi * sp.I * res2)
print(f"Residue theorem: {result2} = {float(result2):.6f}")
num2, _ = quad(lambda x: 1/(x**2+1)**2, -100, 100)
print(f"Numerical verification: {num2:.6f}\n")
```

#### Type 3: Fourier-type integrals — $\int_{-\infty}^{\infty} f(x) e^{iax}\, dx$ ($a > 0$)

Apply Jordan's lemma. If $f(z) \to 0$ as $|z| \to \infty$ (first-order is sufficient), the contribution from the upper half-plane semicircle is 0:

$$\int_{-\infty}^{\infty} f(x) e^{iax}\, dx = 2\pi i \sum_{\text{Im}(z_k) > 0} \text{Res}_{z=z_k} f(z) e^{iaz}$$

Integrals containing $\cos(ax)$ or $\sin(ax)$ use $e^{iax}$ and then take the real or imaginary part.

```python
# --- Type 3: ∫₋∞^∞ cos(x)/(x²+1) dx = π/e ---
print("=== Type 3: ∫₋∞^∞ cos(x)/(x²+1) dx ===")
f3 = sp.exp(sp.I*z) / (z**2 + 1)
res3 = sp.residue(f3, z, sp.I)
result3 = sp.simplify(2 * sp.pi * sp.I * res3)
print(f"Residue theorem: Re({result3}) = π/e = {float(sp.pi/sp.E):.6f}")
num3, _ = quad(lambda x: np.cos(x)/(x**2+1), -100, 100)
print(f"Numerical verification: {num3:.6f}\n")
```

#### Type 4: Branch cut integrals — $\int_0^{\infty} x^{a-1} f(x)\, dx$ ($0 < a < 1$)

If the integrand contains $x^a$ (where $a$ is not an integer), a **branch cut** is needed. Use a keyhole contour.

**Representative example**: $\int_0^{\infty} \frac{x^{a-1}}{1+x}\, dx = \frac{\pi}{\sin(\pi a)}$ ($0 < a < 1$)

**Solution strategy**:
1. Choose the positive real axis as the branch cut for $f(z) = z^{a-1}/(1+z)$
2. Keyhole contour: above branch cut → large circle → below branch cut → small circle
3. Below the branch cut, $z^{a-1} = |z|^{a-1} e^{2\pi i(a-1)}$
4. Residue at $z = -1$: $e^{i\pi(a-1)} = -e^{i\pi a}$

$$\int_0^{\infty} \frac{x^{a-1}}{1+x} dx - e^{2\pi i(a-1)} \int_0^{\infty} \frac{x^{a-1}}{1+x} dx = 2\pi i \cdot (-e^{i\pi a})$$

$$(1 - e^{2\pi i(a-1)}) I = -2\pi i e^{i\pi a} \implies I = \frac{\pi}{\sin(\pi a)}$$

```python
# --- Type 4: ∫₀^∞ x^{a-1}/(1+x) dx = π/sin(πa) ---
print("=== Type 4: ∫₀^∞ x^{a-1}/(1+x) dx ===")
for a in [0.25, 0.5, 0.75]:
    theory = np.pi / np.sin(np.pi * a)
    numerical, _ = quad(lambda x: x**(a-1)/(1+x), 0, np.inf)
    print(f"  a = {a}: π/sin(πa) = {theory:.6f}, numerical = {numerical:.6f}")
```

---

## 5. Conformal Mapping

### 5.1 Definition and Properties of Conformal Mapping

An analytic function $w = f(z)$ with $f'(z_0) \neq 0$ is **conformal** in the neighborhood of $z_0$:

- **Angle preservation**: The angle of intersection of two curves is preserved after mapping
- **Laplace equation invariant**: Harmonic functions remain harmonic after mapping
- **Boundary condition preservation**: Physical boundary conditions remain valid after mapping

### 5.2 Möbius Transformations

$$w = \frac{az + b}{cz + d}, \quad ad - bc \neq 0$$

**Properties**: Maps circles and lines to circles and lines, uniquely determined by three points.

```python
def mobius_transform(z, a, b, c, d):
    """Mobius transformation w = (az+b)/(cz+d)"""
    return (a*z + b) / (c*z + d)

def plot_mobius(a, b, c, d, title="Mobius transformation"):
    """Visualization of Mobius transformation of grid lines"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t = np.linspace(-2, 2, 300)

    for x0 in np.linspace(-2, 2, 9):
        zv = x0 + 1j*t; wv = mobius_transform(zv, a, b, c, d)
        axes[0].plot(zv.real, zv.imag, 'b-', alpha=0.4, lw=0.7)
        m = np.abs(wv) < 10
        axes[1].plot(wv.real[m], wv.imag[m], 'b-', alpha=0.4, lw=0.7)

    for y0 in np.linspace(-2, 2, 9):
        zv = t + 1j*y0; wv = mobius_transform(zv, a, b, c, d)
        axes[0].plot(zv.real, zv.imag, 'r-', alpha=0.4, lw=0.7)
        m = np.abs(wv) < 10
        axes[1].plot(wv.real[m], wv.imag[m], 'r-', alpha=0.4, lw=0.7)

    for ax, lab in zip(axes, ['z-plane', 'w-plane']):
        ax.set_xlim(-4,4); ax.set_ylim(-4,4)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.set_title(lab)
    fig.suptitle(f'{title}: w=({a}z+{b})/({c}z+{d})')
    plt.tight_layout(); plt.show()

# Cayley transform (upper half-plane → unit disk): w = (z-i)/(z+i)
# plot_mobius(1, -1j, 1, 1j, "Cayley transform")
```

### 5.3 Physics Applications (Fluid Mechanics, Electric Field)

**Complex potential**: In two-dimensional incompressible irrotational flow:

$$W(z) = \phi(x, y) + i\psi(x, y), \quad \frac{dW}{dz} = v_x - iv_y$$

$\phi$ is the velocity potential, $\psi$ is the streamline function.

| Flow | $W(z)$ | Physical meaning |
|------|--------|------------|
| Uniform flow | $Uz$ | Velocity $U$ |
| Source/sink | $(Q/2\pi)\ln z$ | Strength $Q$ |
| Vortex | $(-i\Gamma/2\pi)\ln z$ | Circulation $\Gamma$ |
| Doublet | $\mu/z$ | Dipole |
| Around cylinder | $U(z + a^2/z)$ | Radius $a$ |

```python
def plot_flow(W_func, xlim=(-3,3), ylim=(-3,3), N=400, title="Streamline plot"):
    """Visualization of streamlines and equipotential lines from complex potential"""
    xv = np.linspace(*xlim, N); yv = np.linspace(*ylim, N)
    X, Y = np.meshgrid(xv, yv); Z = X + 1j*Y

    with np.errstate(divide='ignore', invalid='ignore'):
        W = W_func(Z)
    phi, psi = np.real(W), np.imag(W)
    mask = np.abs(W) > 50; phi[mask] = np.nan; psi[mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].contour(X, Y, psi, levels=30, colors='blue', linewidths=0.8)
    axes[0].set_title(f'{title} - streamlines (ψ=const)')
    axes[1].contour(X, Y, phi, levels=30, colors='red', linewidths=0.8)
    axes[1].set_title(f'{title} - equipotential lines (φ=const)')
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout(); plt.show()

# Flow around cylinder with circulation
U, a, Gamma = 1.0, 1.0, 2*np.pi
W_cyl = lambda z: U*(z + a**2/z) - 1j*Gamma/(2*np.pi)*np.log(z)
# plot_flow(W_cyl, title="Flow around cylinder with circulation")

# Electric dipole (+q at z=d, -q at z=-d)
W_dipole = lambda z: -1/(2*np.pi) * (np.log(z-0.5) - np.log(z+0.5))
# plot_flow(W_dipole, title="Electric dipole potential")
```

**Joukowski transformation and lift**: $w = z + c^2/z$ maps a circle to an airfoil. Lift per unit length by the Kutta-Joukowski theorem:

$$L = \rho U \Gamma$$

This result is elegantly derived from the residue theorem.

### 5.4 Schwarz-Christoffel Mapping

**Problem**: Find a conformal mapping that maps the upper half-plane to a polygonal region.

**Schwarz-Christoffel formula**: If points $x_1, x_2, \ldots, x_n$ on the real axis of the upper half-plane map to vertices $w_1, w_2, \ldots, w_n$ of a polygon, and the interior angle at each vertex is $\alpha_k \pi$:

$$\frac{dw}{dz} = A \prod_{k=1}^{n} (z - x_k)^{\alpha_k - 1}$$

where $A$ is a complex constant.

**Example**: The mapping from the upper half-plane to a rectangle is expressed in terms of elliptic integrals and is used in electrostatics to calculate the fringing field of parallel plate capacitors.

---

## 6. Analytic Continuation

### 6.1 Basic Concept

When a function $f(z)$ is defined in a region $D_1$, if an analytic function $g(z)$ in a larger region $D_2 \supset D_1$ agrees with $f$ in $D_1$, then $g$ is called an **analytic continuation** of $f$.

**Uniqueness**: If an analytic continuation exists, it is unique (by the identity theorem).

### 6.2 Physics Applications

**Gamma function**: Originally $\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} dt$ is only defined for $\text{Re}(z) > 0$, but using the recurrence relation $\Gamma(z) = \Gamma(z+1)/z$, it can be analytically continued to the entire complex plane except for negative integers.

**Riemann zeta function**: $\zeta(s) = \sum_{n=1}^{\infty} n^{-s}$ converges for $\text{Re}(s) > 1$, but through analytic continuation it extends to the entire complex plane except $s = 1$. This is used in physics for **zeta function regularization** (Casimir effect, etc.).

```python
# Visualization of analytic continuation of the gamma function
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

x = np.linspace(-4.5, 5, 2000)
y = np.array([gamma(xi) if abs(xi - round(xi)) > 0.02 or xi > 0.5
              else np.nan for xi in x])

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=1.5)
plt.ylim(-10, 10)
for n in range(0, -5, -1):
    plt.axvline(x=n, color='red', linewidth=0.5, linestyle='--', alpha=0.5)

plt.xlabel('z')
plt.ylabel('Γ(z)')
plt.title('Analytic continuation of the gamma function (poles at negative integers)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

---

## Exercises

### Basic Problems

1. Determine whether the following functions are analytic using the Cauchy-Riemann conditions:
   - (a) $f(z) = z^3$
   - (b) $f(z) = |z|^2$
   - (c) $f(z) = \bar{z}$
   - (d) $f(z) = e^{-z}\sin z$

2. Find the harmonic conjugate $v(x, y)$ of $u(x, y) = x^3 - 3xy^2 + 2x$.

3. Calculate the following integrals ($C$: circle of radius 2 centered at origin):
   - (a) $\oint_C e^z/z^2\, dz$
   - (b) $\oint_C \cos z/z^3\, dz$
   - (c) $\oint_C z^2/((z-1)(z+2))\, dz$

### Intermediate Problems

4. Calculate using the residue theorem:
   - (a) $\displaystyle\int_0^{2\pi} \frac{d\theta}{5 + 4\cos\theta}$
   - (b) $\displaystyle\int_0^{\infty} \frac{x^2}{(x^2+1)(x^2+4)}\, dx$
   - (c) $\displaystyle\int_0^{\infty} \frac{\cos 3x}{x^2 + 1}\, dx$

5. Find the residues of $f(z) = z/((z-1)^2(z+2))$ at all singularities, and find the integral value over $|z|=3$.

### Advanced Problems

6. **Fluid mechanics**: When there is a source at $z=a$ and a sink at $z=-a$, and the $x$-axis is a wall, find the streamline function in the upper half-plane using the method of images.

7. **Quantum mechanics**: Derive the density of states $\rho(E) = -\text{Im}\, G(E)/\pi$ from the Green's function $G(E) = (E - H + i\epsilon)^{-1}$ using the residue theorem.

8. **Solution** (Problem 4a):

```python
z = sp.Symbol('z')
# ∫₀²π dθ/(5+4cosθ) → ∮ 1/(i(2z²+5z+2)) dz
integrand = 1 / (sp.I * (2*z**2 + 5*z + 2))
poles = sp.solve(2*z**2 + 5*z + 2, z)
print(f"Poles: {poles}")  # z=-1/2 (interior), z=-2 (exterior)
res = sp.residue(integrand, z, sp.Rational(-1, 2))
print(f"Integral value: {sp.simplify(2*sp.pi*sp.I*res)}")  # 2π/3
```

---

## References

### Textbooks
- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 14
- **Arfken, Weber** *Mathematical Methods for Physicists*, Ch. 6-7
- **Churchill, Brown** *Complex Variables and Applications*

### Supplementary Materials
- **Needham, T.** *Visual Complex Analysis* - Geometric intuition
- **Ablowitz, Fokas** *Complex Variables: Introduction and Applications* - Physics applications

### Summary of Key Formulas

| Formula | Condition |
|------|------|
| Cauchy-Riemann: $u_x = v_y$, $u_y = -v_x$ | Differentiable |
| Cauchy's theorem: $\oint_C f\, dz = 0$ | Analytic in simply connected region |
| Cauchy's formula: $f(z_0) = \frac{1}{2\pi i}\oint \frac{f}{z-z_0} dz$ | $f$ analytic, $z_0$ interior |
| Residue theorem: $\oint f\, dz = 2\pi i \sum \text{Res}$ | Finite singularities |

---

## Next Lesson

In [15. Laplace Transform](15_Laplace_Transform.md), we will cover the definition and properties of the Laplace transform, inverse transforms, and applications to differential equations.

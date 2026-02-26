# 02. Complex Numbers

> **Boas Chapter 2** — In the physical sciences, complex numbers are more than mere mathematical tools; they are the language of wave phenomena, AC circuits, quantum mechanics, and other core areas of physics.

---

## Learning Objectives

After completing this lesson, you will be able to:

- **Freely convert between algebraic and geometric representations** of complex numbers (Cartesian, polar, exponential forms)
- **Derive trigonometric identities and compute nth roots** using Euler's formula and De Moivre's theorem
- **Understand and compute the properties** of complex functions (exponential, trigonometric, hyperbolic, logarithmic)
- **Physics applications**: Calculate impedance in AC circuits, represent waves in complex form, handle quantum mechanical wavefunctions
- **Grasp the geometric meaning of 2D transformations** using complex numbers and understand the basic principles of conformal mapping

---

## 1. Fundamentals of Complex Numbers

### 1.1 Imaginary Unit and Definition of Complex Numbers

Real numbers alone cannot solve equations like $x^2 + 1 = 0$. We define the **imaginary unit** $i$ as:

$$
i^2 = -1, \quad i = \sqrt{-1}
$$

A **complex number** $z$ consists of two real numbers $a$ and $b$:

$$
z = a + bi
$$

- $a = \text{Re}(z)$: **real part**
- $b = \text{Im}(z)$: **imaginary part**

> **Note**: In engineering, to avoid confusion with current $i$, the imaginary unit is denoted as $j$. Python also uses `j`.

**Complex conjugate**:

$$
\bar{z} = z^* = a - bi
$$

**Modulus (absolute value)**:

$$
|z| = \sqrt{a^2 + b^2} = \sqrt{z \cdot \bar{z}}
$$

```python
import numpy as np

# Basic complex number operations
z1 = 3 + 4j
z2 = 1 - 2j

print(f"z1 = {z1}")
print(f"Real part: {z1.real}, Imaginary part: {z1.imag}")
print(f"Conjugate: {z1.conjugate()}")
print(f"|z1| = {abs(z1):.4f}")  # sqrt(9 + 16) = 5.0

# Useful properties
print(f"\nz1 * conj(z1) = {z1 * z1.conjugate()}")  # |z1|^2 = 25
print(f"|z1|^2 = {abs(z1)**2}")
```

### 1.2 Complex Plane (Argand Diagram)

A complex number $z = a + bi$ is represented as a point $(a, b)$ in a two-dimensional plane. This plane is called the **complex plane** or **Argand diagram**.

- **Horizontal axis**: real axis
- **Vertical axis**: imaginary axis
- **Distance from origin to point $z$**: $|z|$ (modulus)
- **Angle with real axis**: $\theta = \arg(z)$ (argument)

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Array of complex numbers
points = {
    r'$3+4i$': 3+4j,
    r'$-2+3i$': -2+3j,
    r'$-1-2i$': -1-2j,
    r'$4-i$': 4-1j,
    r'$2i$': 2j,
    r'$-3$': -3+0j,
}

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for (label, z), color in zip(points.items(), colors):
    # Plot point
    ax.plot(z.real, z.imag, 'o', color=color, markersize=10, zorder=5)
    ax.annotate(label, (z.real, z.imag), textcoords="offset points",
                xytext=(10, 10), fontsize=12, color=color)
    # Arrow from origin
    ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Axis settings
ax.axhline(y=0, color='k', linewidth=0.8)
ax.axvline(x=0, color='k', linewidth=0.8)
ax.set_xlim(-5, 6)
ax.set_ylim(-4, 6)
ax.set_xlabel('Re(z)', fontsize=13)
ax.set_ylabel('Im(z)', fontsize=13)
ax.set_title('Complex Plane (Argand Diagram)', fontsize=15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_plane.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.3 Arithmetic Operations on Complex Numbers

For two complex numbers $z_1 = a + bi$, $z_2 = c + di$:

**Addition/Subtraction**: operate on real and imaginary parts separately

$$
z_1 \pm z_2 = (a \pm c) + (b \pm d)i
$$

**Multiplication**: expand using $i^2 = -1$

$$
z_1 \cdot z_2 = (ac - bd) + (ad + bc)i
$$

**Division**: rationalize the denominator with the conjugate

$$
\frac{z_1}{z_2} = \frac{z_1 \cdot \bar{z_2}}{|z_2|^2} = \frac{(ac + bd) + (bc - ad)i}{c^2 + d^2}
$$

```python
import numpy as np

z1 = 3 + 4j
z2 = 1 - 2j

print("=== Complex arithmetic operations ===")
print(f"z1 + z2 = {z1 + z2}")          # (4+2j)
print(f"z1 - z2 = {z1 - z2}")          # (2+6j)
print(f"z1 * z2 = {z1 * z2}")          # (11-2j) = (3+8)+(4-6)i
print(f"z1 / z2 = {z1 / z2}")          # (-1+2j)

# Division verification: manual computation
numerator = z1 * z2.conjugate()
denominator = abs(z2)**2
print(f"\nManual division verification:")
print(f"  z1 * conj(z2) = {numerator}")
print(f"  |z2|^2 = {denominator}")
print(f"  result = {numerator / denominator}")
```

---

## 2. Polar and Exponential Representations

### 2.1 Polar Form (r, $\theta$)

A complex number $z = a + bi$ in polar form is:

$$
z = r(\cos\theta + i\sin\theta)
$$

where:
- $r = |z| = \sqrt{a^2 + b^2}$ : modulus
- $\theta = \arg(z) = \arctan\left(\frac{b}{a}\right)$ : argument

**Inverse transformation**:

$$
a = r\cos\theta, \quad b = r\sin\theta
$$

> **Note**: $\arctan(b/a)$ alone cannot distinguish quadrants, so in programming we use `np.arctan2(b, a)` or `np.angle(z)`.

### 2.2 Euler's Formula

One of the most beautiful formulas in mathematics, **Euler's formula**:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

Using this, complex numbers can be expressed in **exponential form**:

$$
z = re^{i\theta}
$$

**Euler's identity** (setting $\theta = \pi$):

$$
e^{i\pi} + 1 = 0
$$

This identity connects five fundamental constants in mathematics ($e$, $i$, $\pi$, $1$, $0$) in a single equation.

**Proof (using Taylor series)**:

$$
e^{i\theta} = \sum_{n=0}^{\infty} \frac{(i\theta)^n}{n!}
= \underbrace{\left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right)}_{\cos\theta}
+ i\underbrace{\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)}_{\sin\theta}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Euler's formula visualization: e^{i*theta} is a point on the unit circle
theta = np.linspace(0, 2*np.pi, 300)
z = np.exp(1j * theta)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) e^{i*theta} on the unit circle
ax = axes[0]
ax.plot(z.real, z.imag, 'b-', linewidth=2)

# Mark special angles
special_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi,
                  3*np.pi/2]
labels = [r'$1$', r'$e^{i\pi/6}$', r'$e^{i\pi/4}$', r'$e^{i\pi/3}$',
          r'$e^{i\pi/2}=i$', r'$e^{i\pi}=-1$', r'$e^{i3\pi/2}=-i$']

for angle, label in zip(special_angles, labels):
    w = np.exp(1j * angle)
    ax.plot(w.real, w.imag, 'ro', markersize=8, zorder=5)
    offset = (15 * np.cos(angle), 15 * np.sin(angle))
    ax.annotate(label, (w.real, w.imag), textcoords="offset points",
                xytext=offset, fontsize=10, ha='center')

ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect('equal')
ax.set_title(r'$e^{i\theta}$: Complex numbers on the unit circle', fontsize=14)
ax.set_xlabel('Re', fontsize=12)
ax.set_ylabel('Im', fontsize=12)
ax.grid(True, alpha=0.3)

# (b) Taylor series convergence of Euler's formula
ax2 = axes[1]
theta_val = np.pi / 3  # 60 degrees

n_terms_list = range(1, 12)
partial_real = []
partial_imag = []
cumsum = 0 + 0j

for n in range(20):
    cumsum += (1j * theta_val)**n / np.math.factorial(n)
    if n + 1 <= 11:
        partial_real.append(cumsum.real)
        partial_imag.append(cumsum.imag)

exact = np.exp(1j * theta_val)
ax2.axhline(y=exact.real, color='blue', linestyle='--', alpha=0.5,
            label=f'cos(pi/3) = {exact.real:.4f}')
ax2.axhline(y=exact.imag, color='red', linestyle='--', alpha=0.5,
            label=f'sin(pi/3) = {exact.imag:.4f}')
ax2.plot(range(1, 12), partial_real, 'bo-', label='Partial sum (real part)')
ax2.plot(range(1, 12), partial_imag, 'rs-', label='Partial sum (imag part)')
ax2.set_xlabel('Number of terms', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.set_title(r'Convergence of Taylor series for $e^{i\pi/3}$', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('euler_formula.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 Exponential Form and Multiplication/Division

The exponential form makes multiplication and division extremely simple.

For $z_1 = r_1 e^{i\theta_1}$, $z_2 = r_2 e^{i\theta_2}$:

**Multiplication**: multiply moduli, add arguments

$$
z_1 \cdot z_2 = r_1 r_2 \, e^{i(\theta_1 + \theta_2)}
$$

**Division**: divide moduli, subtract arguments

$$
\frac{z_1}{z_2} = \frac{r_1}{r_2} \, e^{i(\theta_1 - \theta_2)}
$$

**Powers**: raise modulus to power, multiply argument

$$
z^n = r^n e^{in\theta}
$$

```python
import numpy as np

# Conversion between polar/exponential forms
z = 1 + 1j * np.sqrt(3)  # = 2 * e^{i*pi/3}

r = abs(z)
theta = np.angle(z)  # radians

print(f"z = {z}")
print(f"|z| = {r:.4f}")
print(f"arg(z) = {theta:.4f} rad = {np.degrees(theta):.1f}°")
print(f"Exponential form: {r:.4f} * exp(i * {theta:.4f})")

# Multiplication example
z1 = 2 * np.exp(1j * np.pi/4)   # r=2, theta=45°
z2 = 3 * np.exp(1j * np.pi/6)   # r=3, theta=30°
z_product = z1 * z2

print(f"\n=== Multiplication ===")
print(f"z1 = 2*exp(i*pi/4), z2 = 3*exp(i*pi/6)")
print(f"z1*z2 = {z_product:.4f}")
print(f"|z1*z2| = {abs(z_product):.4f} (= 2*3 = 6)")
print(f"arg(z1*z2) = {np.degrees(np.angle(z_product)):.1f}° (= 45+30 = 75°)")
```

---

## 3. De Moivre's Theorem and nth Roots

### 3.1 De Moivre's Theorem

Directly derived from Euler's formula, this important theorem states:

$$
(\cos\theta + i\sin\theta)^n = \cos(n\theta) + i\sin(n\theta)
$$

This theorem has powerful applications:
- Deriving **multiple angle formulas**
- Proving **trigonometric identities**
- Computing **nth roots**

**Example**: Express $\cos(3\theta)$ in terms of $\cos\theta$

From De Moivre's theorem with $n = 3$:

$$
\cos(3\theta) + i\sin(3\theta) = (\cos\theta + i\sin\theta)^3
$$

Expand the right side with the binomial theorem and compare real parts:

$$
\cos(3\theta) = \cos^3\theta - 3\cos\theta\sin^2\theta = 4\cos^3\theta - 3\cos\theta
$$

```python
import sympy as sp

# Using De Moivre's theorem via SymPy to derive multiple-angle formulas
theta = sp.Symbol('theta', real=True)

# Expand (cos(theta) + i*sin(theta))^n to derive cos(n*theta), sin(n*theta)
for n in [2, 3, 4]:
    expr = sp.expand((sp.cos(theta) + sp.I * sp.sin(theta))**n)
    real_part = sp.re(expr)
    imag_part = sp.im(expr)

    # Substitute sin^2 = 1 - cos^2 to express solely in terms of cos
    real_simplified = sp.simplify(
        real_part.rewrite(sp.cos).expand(trig=True)
    )
    imag_simplified = sp.simplify(
        imag_part.rewrite(sp.sin).expand(trig=True)
    )

    print(f"=== n = {n} ===")
    print(f"  cos({n}*theta) = {sp.trigsimp(real_part)}")
    print(f"  sin({n}*theta) = {sp.trigsimp(imag_part)}")
    print()

# Numerical verification
import numpy as np
t = np.pi / 7
for n in [2, 3, 4]:
    lhs = np.cos(n * t)
    rhs = (np.exp(1j * t)**n).real
    print(f"cos({n}*pi/7): direct={lhs:.10f}, De Moivre={rhs:.10f}, diff={abs(lhs-rhs):.2e}")
```

### 3.2 nth Roots

To find the **nth roots** of a complex number $w$, i.e., solve $z^n = w$:

Let $w = Re^{i\Phi}$ (where $R = |w|$, $\Phi = \arg(w)$):

$$
z_k = R^{1/n} \exp\left(i\frac{\Phi + 2\pi k}{n}\right), \quad k = 0, 1, 2, \ldots, n-1
$$

Key insight: There exist **$n$ distinct roots**, equally spaced on a circle of radius $R^{1/n}$.

```python
import numpy as np
import matplotlib.pyplot as plt

def nth_roots(w, n):
    """Find all nth roots of complex number w."""
    R = abs(w)
    Phi = np.angle(w)
    roots = []
    for k in range(n):
        z_k = R**(1/n) * np.exp(1j * (Phi + 2*np.pi*k) / n)
        roots.append(z_k)
    return np.array(roots)

# Example: cube roots of (1+i)
w = 1 + 1j
n = 3
roots = nth_roots(w, n)

print(f"The {n} roots of w = {w}:")
for k, root in enumerate(roots):
    print(f"  z_{k} = {root:.6f}")
    print(f"       = {abs(root):.4f} * exp(i * {np.degrees(np.angle(root)):.2f}°)")
    # Verification
    print(f"       verify: z_{k}^{n} = {root**n:.6f}")
    print()

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
test_cases = [
    (8+0j, 3, r'$\sqrt[3]{8}$'),
    (1+1j, 4, r'$\sqrt[4]{1+i}$'),
    (-1+0j, 5, r'$\sqrt[5]{-1}$'),
]

for ax, (w, n, title) in zip(axes, test_cases):
    roots = nth_roots(w, n)
    R = abs(w)**(1/n)

    # Draw circle
    circle_theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(circle_theta), R*np.sin(circle_theta),
            'b--', alpha=0.3, linewidth=1)

    # Mark roots
    for k, z_k in enumerate(roots):
        ax.plot(z_k.real, z_k.imag, 'ro', markersize=10, zorder=5)
        ax.annotate(f'$z_{k}$', (z_k.real, z_k.imag),
                    textcoords="offset points", xytext=(8, 8), fontsize=11)

    # Connect as regular polygon
    polygon = np.append(roots, roots[0])
    ax.plot(polygon.real, polygon.imag, 'r-', alpha=0.4, linewidth=1)

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

plt.suptitle('nth roots of complex numbers: evenly spaced on a circle', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('nth_roots.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.3 Roots of Unity

The special case $w = 1$: roots of $z^n = 1$ are called **nth roots of unity**.

$$
\omega_k = e^{2\pi i k / n}, \quad k = 0, 1, \ldots, n-1
$$

**Primitive nth root**: Setting $\omega = e^{2\pi i / n}$

$$
\omega_k = \omega^k, \quad k = 0, 1, \ldots, n-1
$$

**Key properties**:

1. $\omega^n = 1$
2. $1 + \omega + \omega^2 + \cdots + \omega^{n-1} = 0$ (sum of roots is 0)
3. $\omega_j \cdot \omega_k = \omega_{(j+k) \bmod n}$ (group structure)

> **Application**: nth roots of unity are fundamental to the **Discrete Fourier Transform** (DFT), essential in signal processing.

```python
import numpy as np

# Verify properties of nth roots of unity
for n in [3, 4, 6, 8]:
    omega = np.exp(2j * np.pi / n)
    roots = np.array([omega**k for k in range(n)])

    print(f"=== {n}th roots of unity ===")
    print(f"  Primitive root omega = exp(2*pi*i/{n}) = {omega:.6f}")
    print(f"  omega^{n} = {omega**n:.6f} (verify = 1)")
    print(f"  Sum of roots = {roots.sum():.6f} (verify = 0)")
    print(f"  Product of roots = {np.prod(roots):.6f}")
    print()
```

---

## 4. Complex Functions

### 4.1 Complex Exponential Function

For a complex number $z = x + iy$, the complex exponential function is:

$$
e^z = e^{x+iy} = e^x(\cos y + i\sin y)
$$

**Key properties**:
- $|e^z| = e^x = e^{\text{Re}(z)}$ (modulus depends only on real part)
- $\arg(e^z) = y = \text{Im}(z)$
- $e^z$ has **period $2\pi i$**: $e^{z + 2\pi i} = e^z$

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualization of the complex exponential function: image of e^z
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) Image of vertical lines x = const: circles centered at origin
ax = axes[0]
y = np.linspace(0, 2*np.pi, 200)
for x_val in [-1, -0.5, 0, 0.5, 1]:
    w = np.exp(x_val + 1j*y)
    ax.plot(w.real, w.imag, label=f'x={x_val}')

ax.set_aspect('equal')
ax.set_title(r'$e^{x+iy}$: image of vertical lines $x=\mathrm{const}$', fontsize=13)
ax.set_xlabel('Re', fontsize=12)
ax.set_ylabel('Im', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) Image of horizontal lines y = const: rays from origin
ax2 = axes[1]
x = np.linspace(-2, 2, 200)
for y_val in np.linspace(0, 2*np.pi, 9)[:-1]:
    w = np.exp(x + 1j*y_val)
    ax2.plot(w.real, w.imag, label=f'y={y_val:.2f}')

ax2.set_xlim(-8, 8)
ax2.set_ylim(-8, 8)
ax2.set_aspect('equal')
ax2.set_title(r'$e^{x+iy}$: image of horizontal lines $y=\mathrm{const}$', fontsize=13)
ax2.set_xlabel('Re', fontsize=12)
ax2.set_ylabel('Im', fontsize=12)
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_exp.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.2 Complex Trigonometric and Hyperbolic Functions

Using Euler's formula, trigonometric and hyperbolic functions can be defined in terms of exponentials.

**Complex trigonometric functions**:

$$
\cos z = \frac{e^{iz} + e^{-iz}}{2}, \quad \sin z = \frac{e^{iz} - e^{-iz}}{2i}
$$

**Complex hyperbolic functions**:

$$
\cosh z = \frac{e^z + e^{-z}}{2}, \quad \sinh z = \frac{e^z - e^{-z}}{2}
$$

**Relationship between trigonometric and hyperbolic functions**:

$$
\cos(iz) = \cosh z, \quad \sin(iz) = i\sinh z
$$

$$
\cosh(iz) = \cos z, \quad \sinh(iz) = i\sin z
$$

> **Important**: For real numbers, $|\sin x| \leq 1$, $|\cos x| \leq 1$, but this restriction disappears for complex numbers. For example, $\cos(i) = \cosh(1) \approx 1.543$.

```python
import numpy as np

# Verify relationships between complex trig and hyperbolic functions
z_values = [1+1j, 2-0.5j, 0+2j, np.pi/4+0j]

print("=== Complex trigonometric functions ===")
for z in z_values:
    # Compute cos(z) directly from the exponential definition
    cos_euler = (np.exp(1j*z) + np.exp(-1j*z)) / 2
    cos_numpy = np.cos(z)
    print(f"z = {z:.4f}")
    print(f"  cos(z) [Euler] = {cos_euler:.6f}")
    print(f"  cos(z) [NumPy] = {cos_numpy:.6f}")
    print(f"  diff = {abs(cos_euler - cos_numpy):.2e}")
    print()

# Verify cos(iz) = cosh(z)
z = 1.5 + 0.7j
print("=== Relation verification ===")
print(f"cos(iz) = {np.cos(1j*z):.8f}")
print(f"cosh(z) = {np.cosh(z):.8f}")
print(f"sin(iz) = {np.sin(1j*z):.8f}")
print(f"i*sinh(z) = {1j*np.sinh(z):.8f}")
```

### 4.3 Complex Logarithm

The logarithm of a complex number $z = re^{i\theta}$:

$$
\ln z = \ln r + i\theta = \ln|z| + i\arg(z)
$$

**Multi-valued function**: Since the argument $\theta$ has freedom up to integer multiples of $2\pi$:

$$
\text{Ln}(z) = \ln|z| + i(\theta + 2n\pi), \quad n \in \mathbb{Z}
$$

- **Principal value**: Restricting $-\pi < \theta \leq \pi$, denoted as $\text{Log}(z)$
- **Branch**: Restricting the range of the argument to make a multi-valued function single-valued
- **Branch point**: $z = 0$ (where logarithm is undefined)
- **Branch cut**: Conventionally the negative real axis

```python
import numpy as np

# Multi-valuedness of the complex logarithm
z = -1 + 0j

print("=== Multi-valuedness of ln(-1) ===")
print(f"Principal value: Log(-1) = {np.log(-1+0j)}")  # i*pi

for n in range(-2, 3):
    val = np.log(abs(z)) + 1j * (np.pi + 2*np.pi*n)
    # Verify: e^val = z?
    check = np.exp(val)
    print(f"  n={n:+d}: ln(-1) = {val:.6f}, exp(ln(-1)) = {check:.6f}")

# Verify properties of complex logarithm
z1 = 2 + 3j
z2 = 1 - 1j
print(f"\n=== Logarithm properties (principal value) ===")
print(f"Log(z1*z2) = {np.log(z1*z2):.6f}")
print(f"Log(z1) + Log(z2) = {np.log(z1) + np.log(z2):.6f}")
print("Note: with principal value, Log(z1*z2) != Log(z1) + Log(z2) may occur")
print(f"diff = {abs(np.log(z1*z2) - np.log(z1) - np.log(z2)):.6e}")
```

---

## 5. Physics Applications

### 5.1 AC Circuits (Impedance)

Complex numbers are essential for analyzing alternating current (AC) circuits. Representing voltage/current with complex exponentials transforms differential equations into algebraic equations.

**Complex impedance** $Z$:

| Element | Impedance | Phase |
|------|----------|------|
| Resistor $R$ | $Z_R = R$ | $0$ |
| Inductor $L$ | $Z_L = i\omega L$ | $+90°$ |
| Capacitor $C$ | $Z_C = \frac{1}{i\omega C} = -\frac{i}{\omega C}$ | $-90°$ |

**Total impedance** of series RLC circuit:

$$
Z = R + i\left(\omega L - \frac{1}{\omega C}\right)
$$

- $|Z|$: magnitude of impedance (voltage amplitude / current amplitude)
- $\arg(Z)$: phase difference between voltage and current
- **Resonance**: When $\omega L = 1/(\omega C)$, $Z = R$ (pure resistance)

$$
\omega_0 = \frac{1}{\sqrt{LC}} \quad \text{(resonance frequency)}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Impedance analysis of RLC series circuit
R = 100      # Ohm
L = 0.1      # Henry
C = 1e-6     # Farad

omega_0 = 1 / np.sqrt(L * C)  # resonance angular frequency
f_0 = omega_0 / (2 * np.pi)   # resonance frequency

print(f"Resonance frequency: f_0 = {f_0:.1f} Hz")
print(f"Resonance angular frequency: omega_0 = {omega_0:.1f} rad/s")

# Frequency range
omega = np.linspace(100, 20000, 2000)
Z = R + 1j * (omega * L - 1/(omega * C))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) |Z| vs omega
ax = axes[0, 0]
ax.semilogy(omega, np.abs(Z), 'b-', linewidth=2)
ax.axvline(x=omega_0, color='r', linestyle='--', alpha=0.7,
           label=f'$\\omega_0$ = {omega_0:.0f} rad/s')
ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
ax.set_ylabel(r'$|Z|$ ($\Omega$)', fontsize=12)
ax.set_title('Impedance magnitude', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (b) arg(Z) vs omega
ax = axes[0, 1]
ax.plot(omega, np.degrees(np.angle(Z)), 'g-', linewidth=2)
ax.axvline(x=omega_0, color='r', linestyle='--', alpha=0.7)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
ax.set_ylabel(r'$\arg(Z)$ (degrees)', fontsize=12)
ax.set_title('Phase angle', fontsize=13)
ax.grid(True, alpha=0.3)

# (c) Current response (V_0 = 1V)
V0 = 1.0  # voltage amplitude
I = V0 / Z

ax = axes[1, 0]
ax.plot(omega, np.abs(I) * 1000, 'm-', linewidth=2)
ax.axvline(x=omega_0, color='r', linestyle='--', alpha=0.7,
           label=f'Resonance: I_max = {1000*V0/R:.1f} mA')
ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
ax.set_ylabel(r'$|I|$ (mA)', fontsize=12)
ax.set_title('Current response (resonance curve)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (d) Impedance locus in the complex plane
ax = axes[1, 1]
ax.plot(Z.real, Z.imag, 'b-', linewidth=2)
ax.plot(R, 0, 'ro', markersize=10, zorder=5, label=f'Resonance point ($\\omega_0$)')
ax.set_xlabel(r'Re$(Z)$ ($\Omega$)', fontsize=12)
ax.set_ylabel(r'Im$(Z)$ ($\Omega$)', fontsize=12)
ax.set_title('Impedance locus (Nyquist plot)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle(f'RLC series circuit (R={R}Ω, L={L*1000}mH, C={C*1e6}μF)',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('rlc_circuit.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 Wave Representation in Complex Form

A one-dimensional traveling wave is concisely expressed using complex numbers:

$$
\psi(x, t) = A e^{i(kx - \omega t)}
$$

where:
- $A$: complex amplitude (includes magnitude and initial phase)
- $k$: wave number, $k = 2\pi/\lambda$
- $\omega$: angular frequency, $\omega = 2\pi f$

The actual physical quantity is the **real part**:

$$
\text{Re}[\psi] = |A| \cos(kx - \omega t + \phi)
$$

where $\phi = \arg(A)$.

**Advantages of complex representation**:
1. Differentiation is simple: $\frac{\partial \psi}{\partial t} = -i\omega\psi$
2. Superposition and interference calculations are straightforward
3. Phase relationships are explicit

```python
import numpy as np
import matplotlib.pyplot as plt

# Superposition of two waves (interference)
x = np.linspace(0, 10, 500)
t = 0

# Wave 1: A1 = 1, k1 = 2, omega1 = 3
A1, k1, omega1 = 1.0, 2.0, 3.0
psi1 = A1 * np.exp(1j * (k1*x - omega1*t))

# Wave 2: A2 = 0.8, k2 = 2.2, omega2 = 3.1 (slightly different wavenumber/frequency)
A2, k2, omega2 = 0.8, 2.2, 3.1
psi2 = A2 * np.exp(1j * (k2*x - omega2*t))

# Superposition
psi_total = psi1 + psi2

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(x, psi1.real, 'b-', linewidth=1.5, label='Wave 1')
axes[0].plot(x, np.abs(psi1)*np.ones_like(x), 'b--', alpha=0.3)
axes[0].set_ylabel('Re(ψ₁)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, psi2.real, 'r-', linewidth=1.5, label='Wave 2')
axes[1].plot(x, np.abs(psi2)*np.ones_like(x), 'r--', alpha=0.3)
axes[1].set_ylabel('Re(ψ₂)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].plot(x, psi_total.real, 'purple', linewidth=1.5, label='Superposition (ψ₁ + ψ₂)')
axes[2].plot(x, np.abs(psi_total), 'k--', alpha=0.5, label='Envelope |ψ|')
axes[2].plot(x, -np.abs(psi_total), 'k--', alpha=0.5)
axes[2].set_ylabel('Re(ψ₁ + ψ₂)', fontsize=12)
axes[2].set_xlabel('x', fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Complex wave representation and Beat phenomenon', fontsize=14)
plt.tight_layout()
plt.savefig('wave_superposition.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.3 Quantum Mechanical Wavefunctions

In quantum mechanics, the state of a particle is described by a **complex wavefunction** $\Psi(x, t)$.

**Free particle wavefunction**:

$$
\Psi(x, t) = A \exp\left[i\left(kx - \frac{\hbar k^2}{2m}t\right)\right]
$$

**Probability density**: $|\Psi(x,t)|^2 = \Psi^* \Psi$ (observable quantities are always real)

**Schrödinger equation**:

$$
i\hbar \frac{\partial \Psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \Psi}{\partial x^2} + V(x)\Psi
$$

> **Intrinsically complex**: The Schrödinger equation explicitly contains $i$. In quantum mechanics, complex numbers are not merely convenient — they are **essential to physics**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Time evolution of a Gaussian wave packet
hbar = 1.0  # natural units
m = 1.0
sigma_0 = 1.0   # initial wave packet width
k_0 = 5.0       # mean wavenumber (momentum)

x = np.linspace(-10, 20, 1000)

def gaussian_wavepacket(x, t, sigma_0, k_0, m, hbar):
    """Analytic solution for the Gaussian wave packet."""
    sigma_t = sigma_0 * np.sqrt(1 + (hbar*t/(2*m*sigma_0**2))**2)
    phase_factor = hbar * t / (2 * m * sigma_0**2)

    psi = (2*np.pi*sigma_0**2)**(-0.25) * (sigma_0 / sigma_t) * np.exp(
        -(x - hbar*k_0*t/m)**2 / (4*sigma_0*sigma_t) *
        (sigma_0/sigma_t) *
        np.exp(-1j * np.arctan(phase_factor)/2)
    ) * np.exp(1j * (k_0*x - hbar*k_0**2*t/(2*m)))

    # Normalize
    norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi / norm if norm > 0 else psi

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

times = [0, 1, 2, 3, 4]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))

for t, color in zip(times, colors):
    psi = gaussian_wavepacket(x, t, sigma_0, k_0, m, hbar)
    prob = np.abs(psi)**2

    axes[0].plot(x, psi.real, color=color, linewidth=1.5,
                 label=f't = {t}', alpha=0.8)
    axes[1].plot(x, prob, color=color, linewidth=1.5,
                 label=f't = {t}', alpha=0.8)

axes[0].set_ylabel(r'Re($\Psi$)', fontsize=13)
axes[0].set_title('Time evolution of the Gaussian wave packet', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('x', fontsize=13)
axes[1].set_ylabel(r'$|\Psi|^2$', fontsize=13)
axes[1].set_title('Probability density (wave packet spreading)', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wavepacket.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.4 Deriving Trigonometric Identities

Euler's formula allows us to systematically derive complex trigonometric identities.

**Method**: Use $e^{i\theta} = \cos\theta + i\sin\theta$, expand using exponential laws, then compare real and imaginary parts.

**Example 1**: Addition formulas

$$
e^{i(\alpha + \beta)} = e^{i\alpha} \cdot e^{i\beta}
$$

Left side: $\cos(\alpha+\beta) + i\sin(\alpha+\beta)$

Right side: $(\cos\alpha + i\sin\alpha)(\cos\beta + i\sin\beta)$

Comparing real parts: $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$

Comparing imaginary parts: $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$

**Example 2**: Expressing $\cos^n\theta$ in terms of multiple angles

$$
\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}
$$

Therefore:

$$
\cos^n\theta = \frac{1}{2^n}(e^{i\theta} + e^{-i\theta})^n
$$

Expand with the binomial theorem to obtain multiple angle formulas.

```python
import sympy as sp

theta, alpha, beta = sp.symbols('theta alpha beta', real=True)

# Deriving trigonometric identities using Euler's formula
print("=== Deriving the addition formulas ===")
lhs = sp.exp(sp.I * (alpha + beta))
rhs = sp.exp(sp.I * alpha) * sp.exp(sp.I * beta)

# Expand rhs
rhs_expanded = sp.expand(rhs, complex=True)
rhs_trig = sp.expand((sp.cos(alpha) + sp.I*sp.sin(alpha)) *
                      (sp.cos(beta) + sp.I*sp.sin(beta)))

print(f"Real part: cos(a+b) = {sp.re(rhs_trig)}")
print(f"Imaginary part: sin(a+b) = {sp.im(rhs_trig)}")

# Express cos^n(theta) in terms of multiple angles
print("\n=== cos^n(theta) expansion ===")
for n in [2, 3, 4]:
    # cos(theta) = (e^{it} + e^{-it}) / 2
    t = sp.Symbol('t')
    expr = ((sp.exp(sp.I*t) + sp.exp(-sp.I*t)) / 2)**n
    expanded = sp.expand(expr)
    # Use e^{ikt} + e^{-ikt} = 2*cos(kt)
    result = sp.simplify(sp.trigsimp(expanded.rewrite(sp.cos)))
    print(f"  cos^{n}(theta) = {result.subs(t, theta)}")

# Numerical verification
import numpy as np
theta_val = np.pi / 5
print(f"\n=== Numerical verification (theta = pi/5) ===")
print(f"cos^3(theta) = {np.cos(theta_val)**3:.8f}")
print(f"(3*cos(theta) + cos(3*theta))/4 = "
      f"{(3*np.cos(theta_val) + np.cos(3*theta_val))/4:.8f}")
```

---

## 6. Complex Numbers and 2D Transformations

### 6.1 Rotation and Scaling

Multiplication by complex numbers geometrically corresponds to **rotation** and **scaling**.

Multiplying by $w = e^{i\phi}$ produces a **counterclockwise rotation** by angle $\phi$:

$$
z' = e^{i\phi} z
$$

More generally, multiplying by $w = re^{i\phi}$:
- Scaling by $r$
- Rotation by $\phi$

**Conformal mapping**: Transformations by analytic complex functions **preserve angles**. This property is fundamental in fluid dynamics and electromagnetics.

```python
import numpy as np
import matplotlib.pyplot as plt

# Rotation and scaling via complex multiplication
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original shape: triangle
triangle = np.array([1+0j, 0+1j, -0.5-0.5j, 1+0j])

# (a) Pure rotation: multiply by e^{i*pi/4}
angle = np.pi / 4
w_rotate = np.exp(1j * angle)

ax = axes[0]
ax.plot(triangle.real, triangle.imag, 'b-o', linewidth=2,
        markersize=8, label='Original')
rotated = w_rotate * triangle
ax.plot(rotated.real, rotated.imag, 'r-o', linewidth=2,
        markersize=8, label=f'Rotation ({np.degrees(angle):.0f}°)')
ax.set_title(r'Rotation: $z \mapsto e^{i\pi/4} z$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 2)

# (b) Scaling + rotation: (1+i)*z = sqrt(2)*e^{i*pi/4}*z
w_scale_rotate = 1 + 1j

ax = axes[1]
ax.plot(triangle.real, triangle.imag, 'b-o', linewidth=2,
        markersize=8, label='Original')
transformed = w_scale_rotate * triangle
ax.plot(transformed.real, transformed.imag, 'r-o', linewidth=2,
        markersize=8, label=r'$(1+i) \cdot z$')
ax.set_title(r'Scaling+rotation: $z \mapsto (1+i)z$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 3)

# (c) z^2 transformation (nonlinear, conformal)
theta = np.linspace(0, 2*np.pi, 200)
r_vals = [0.5, 0.75, 1.0]

ax = axes[2]
for r in r_vals:
    z_circle = r * np.exp(1j * theta)
    w_mapped = z_circle**2
    ax.plot(z_circle.real, z_circle.imag, 'b-', alpha=0.5, linewidth=1)
    ax.plot(w_mapped.real, w_mapped.imag, 'r-', alpha=0.7, linewidth=1.5)

ax.plot([], [], 'b-', label='Original (circle)', linewidth=2)
ax.plot([], [], 'r-', label=r'$z^2$ mapping', linewidth=2)
ax.set_title(r'Nonlinear conformal map: $z \mapsto z^2$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_transformations.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 Joukowski Transform (Aerodynamics)

The **Joukowski (Zhukovsky) transform** is a representative conformal mapping used in aerodynamics to analyze flow around airfoil cross-sections:

$$
w = z + \frac{c^2}{z}
$$

- Transforms a circle ($z$-plane) to an airfoil shape ($w$-plane)
- Slightly shifting the circle center generates various airfoil shapes
- Allows analytical calculation of velocity field and pressure distribution

```python
import numpy as np
import matplotlib.pyplot as plt

def joukowski(z, c=1.0):
    """Joukowski transform: w = z + c^2/z"""
    return z + c**2 / z

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

theta = np.linspace(0, 2*np.pi, 500)
c = 1.0

# Airfoil shape changes with different circle center offsets
offsets = [
    (0.0, 0.0, 'Circle (center at origin)'),  # perfect circle -> flat plate
    (-0.1, 0.1, 'Slightly shifted circle'),    # asymmetric airfoil
    (-0.15, 0.15, 'More shifted circle'),       # thicker airfoil
]

for ax, (dx, dy, title) in zip(axes, offsets):
    # z-plane: shifted circle
    R = np.sqrt((c - dx)**2 + dy**2) + 0.02  # circle enclosing c
    z = (dx + dy*1j) + R * np.exp(1j * theta)

    # w-plane: Joukowski transform
    w = joukowski(z, c)

    # Show both z-plane and w-plane
    ax.plot(z.real, z.imag, 'b-', linewidth=1.5, alpha=0.5,
            label=r'$z$-plane (circle)')
    ax.plot(w.real, w.imag, 'r-', linewidth=2.5,
            label=r'$w$-plane (airfoil)')

    # Mark singular points
    ax.plot(c, 0, 'ko', markersize=6)
    ax.plot(-c, 0, 'ko', markersize=6)

    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.5, 2.5)

plt.suptitle(r'Joukowski transform: $w = z + c^2/z$ (aerodynamics application)',
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('joukowski.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Flow field visualization** (streamlines):

```python
import numpy as np
import matplotlib.pyplot as plt

# Flow field around a Joukowski airfoil
c = 1.0
dx, dy = -0.1, 0.08
R = np.sqrt((c - dx)**2 + dy**2) + 0.01
center = dx + dy*1j

# Complex potential: uniform flow + flow around cylinder + circulation
U_inf = 1.0  # freestream velocity
Gamma = 4 * np.pi * U_inf * R * np.sin(
    np.arctan2(dy, c - dx) + np.arcsin(Gamma_approx := 0.1)
) if False else 2.5  # Circulation from Kutta condition

# Create grid (z-plane)
x_grid = np.linspace(-3, 4, 400)
y_grid = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_grid, y_grid)
z_grid = X + 1j*Y

# Mask interior of circle
mask = np.abs(z_grid - center) < R

# Complex velocity (in z-plane)
# w(z) = U*(z - center) + U*R^2/(z - center) + i*Gamma/(2*pi)*log(z - center)
zeta = z_grid - center
F = U_inf * zeta + U_inf * R**2 / zeta - 1j*Gamma/(2*np.pi)*np.log(zeta)

# Streamlines = Im(F) = const
psi = F.imag
psi[mask] = np.nan

# Transform to w-plane
w_grid = joukowski(z_grid, c)
w_grid[mask] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# z-plane flow
ax = axes[0]
levels = np.linspace(-4, 4, 40)
ax.contour(X, Y, psi, levels=levels, colors='steelblue', linewidths=0.7)
circle_plot = center + R * np.exp(1j * np.linspace(0, 2*np.pi, 200))
ax.fill(circle_plot.real, circle_plot.imag, color='lightgray', zorder=3)
ax.plot(circle_plot.real, circle_plot.imag, 'k-', linewidth=2, zorder=4)
ax.set_title('z-plane: flow around a cylinder', fontsize=13)
ax.set_aspect('equal')
ax.set_xlim(-3, 4)
ax.set_ylim(-3, 3)

# w-plane flow (around airfoil)
ax = axes[1]
# Airfoil boundary
theta_wing = np.linspace(0, 2*np.pi, 500)
z_wing = center + R * np.exp(1j * theta_wing)
w_wing = joukowski(z_wing, c)

ax.contour(w_grid.real, w_grid.imag, psi, levels=levels,
           colors='steelblue', linewidths=0.7)
ax.fill(w_wing.real, w_wing.imag, color='lightgray', zorder=3)
ax.plot(w_wing.real, w_wing.imag, 'k-', linewidth=2, zorder=4)
ax.set_title('w-plane: flow around airfoil (Joukowski transform)', fontsize=13)
ax.set_aspect('equal')
ax.set_xlim(-3.5, 4.5)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('joukowski_flow.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Practice Problems

### Problem 1: Polar Coordinate Conversion

Express the following complex numbers in polar form $re^{i\theta}$ ($-\pi < \theta \leq \pi$):

(a) $z = -1 + i$
(b) $z = -3 - 3\sqrt{3}\,i$
(c) $z = 5i$

### Problem 2: De Moivre's Theorem Application

Use De Moivre's theorem to express $\sin(4\theta)$ in terms of $\sin\theta$ and $\cos\theta$.

### Problem 3: nth Roots

Find all roots of $z^4 = -16$. Plot them in the complex plane and express the results in the form $a + bi$.

### Problem 4: Complex Logarithm

Compute the following (principal values):

(a) $\ln(-e)$
(b) $\ln(1 + i)$
(c) $i^i = e^{i \ln i}$

### Problem 5: AC Circuit Analysis

For a series RLC circuit with $R = 50\,\Omega$, $L = 20\,\text{mH}$, $C = 10\,\mu\text{F}$ driven by $V(t) = 10\cos(\omega t)$ (V):

(a) Find the resonance frequency $f_0$.
(b) Find the magnitude and phase of impedance $Z$ at $f = 500\,\text{Hz}$.
(c) Find the maximum current at resonance.

### Problem 6: Conformal Mapping

For the Joukowski transform $w = z + 1/z$:

(a) Find the parametric representation of the curve in the $w$-plane when the circle $|z| = 2$ is transformed.
(b) Show that this curve is an ellipse and find the lengths of the major and minor axes.

---

<details>
<summary><strong>Solutions (click to expand)</strong></summary>

```python
import numpy as np

# === Problem 1 solution ===
print("=== Problem 1: Polar coordinate conversion ===\n")

problems_1 = {
    '(a) -1 + i': -1 + 1j,
    '(b) -3 - 3*sqrt(3)*i': -3 - 3*np.sqrt(3)*1j,
    '(c) 5i': 5j,
}

for label, z in problems_1.items():
    r = abs(z)
    theta = np.angle(z)
    print(f"{label}")
    print(f"  z = {z}")
    print(f"  r = {r:.6f}, theta = {theta:.6f} rad = {np.degrees(theta):.2f}°")
    print(f"  Polar form: {r:.4f} * exp(i * {theta:.4f})")
    print()

# === Problem 3 solution ===
print("=== Problem 3: z^4 = -16 ===\n")
w = -16 + 0j
n = 4
R = abs(w)**(1/n)   # 16^(1/4) = 2
Phi = np.angle(w)    # pi

for k in range(n):
    z_k = R * np.exp(1j * (Phi + 2*np.pi*k) / n)
    print(f"z_{k} = {z_k.real:+.6f} {z_k.imag:+.6f}i")
    print(f"     = {R:.4f} * exp(i * {np.degrees((Phi + 2*np.pi*k)/n):.1f}°)")
    print(f"     verify: z^4 = {z_k**4:.6f}")
    print()

# === Problem 4 solution ===
print("=== Problem 4: Complex logarithm ===\n")

# (a) ln(-e)
z_a = -np.e + 0j
print(f"(a) ln(-e) = {np.log(z_a):.6f}")
print(f"    = ln(e) + i*pi = 1 + i*pi = {1 + 1j*np.pi:.6f}\n")

# (b) ln(1+i)
z_b = 1 + 1j
print(f"(b) ln(1+i) = {np.log(z_b):.6f}")
print(f"    = ln(sqrt(2)) + i*pi/4 = {np.log(np.sqrt(2)) + 1j*np.pi/4:.6f}\n")

# (c) i^i
z_c = 1j ** 1j
print(f"(c) i^i = {z_c:.10f}")
print(f"    = exp(i * ln(i)) = exp(i * i*pi/2) = exp(-pi/2)")
print(f"    = {np.exp(-np.pi/2):.10f}")

# === Problem 5 solution ===
print("\n=== Problem 5: RLC circuit ===\n")
R = 50
L = 20e-3
C = 10e-6

# (a)
f_0 = 1 / (2*np.pi*np.sqrt(L*C))
print(f"(a) Resonance frequency: f_0 = {f_0:.2f} Hz")

# (b)
f = 500
omega = 2 * np.pi * f
Z = R + 1j*(omega*L - 1/(omega*C))
print(f"(b) At f = {f} Hz:")
print(f"    Z = {Z:.4f}")
print(f"    |Z| = {abs(Z):.4f} Ohm")
print(f"    arg(Z) = {np.degrees(np.angle(Z)):.2f}°")

# (c)
V0 = 10
I_max = V0 / R
print(f"(c) At resonance: I_max = V0/R = {I_max:.4f} A = {I_max*1000:.1f} mA")
```

</details>

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 2. Wiley.
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 6. Academic Press.
3. **Needham, T.** (1997). *Visual Complex Analysis*. Oxford University Press. — Excellent for geometric understanding of complex numbers

### Online Resources
1. **MIT OCW 18.04**: Complex Variables with Applications
2. **3Blue1Brown**: "What is Euler's Formula?" (YouTube) — Intuitive understanding of Euler's formula
3. **Better Explained**: *An Intuitive Guide to Imaginary Numbers*

### Related Lessons
- [01. Infinite Series](01_Infinite_Series.md): Taylor series (used in Euler's formula proof)
- [12. Complex Analysis](12_Complex_Analysis.md): Analytic functions, Cauchy's theorem, residue theorem (advanced topics from this lesson)
- [05. Fourier Series](05_Fourier_Series.md): Connection between nth roots of unity and DFT

---

## Next Lesson

[03. Vector Analysis](03_Vector_Analysis.md) covers differentiation and integration of scalar and vector fields. We will learn gradient, divergence, curl operators and Stokes'/Gauss' theorems.

# 07. Fourier Series

## Learning Objectives

- Understand the concept of **Fourier series** that decomposes **periodic functions** into a series of trigonometric functions, and its mathematical foundation (orthogonality)
- Calculate **Fourier coefficients** $a_0, a_n, b_n$, and apply them to representative waveforms such as square waves and sawtooth waves
- Understand **Dirichlet conditions**, **pointwise and uniform convergence**, and **Gibbs phenomenon**, and analyze convergence behavior
- Apply **half-range expansion** and **Parseval's theorem** to solve physics problems
- Understand how Fourier series serve as a key tool in **physics applications** such as vibrating strings, heat conduction, and electromagnetic waves

---

## 1. Basic Concepts of Fourier Series

### 1.1 Periodic Functions and Orthogonality

A **periodic function** is a function that satisfies the following for a positive constant $T > 0$:

$$f(x + T) = f(x) \quad \text{for all } x$$

where $T$ is called the **period**. The smallest positive period is called the **fundamental period**.

Typical periodic functions include $\sin x$ (period $2\pi$), $\cos x$ (period $2\pi$), $\tan x$ (period $\pi$), etc.

**Orthogonality** is a key concept in Fourier series. Two functions $f(x)$ and $g(x)$ are **orthogonal** on the interval $[a, b]$ if:

$$\int_a^b f(x) g(x) \, dx = 0$$

This is satisfied. This extends the orthogonality condition $\mathbf{a} \cdot \mathbf{b} = 0$ for vector inner products to function spaces.

### 1.2 Orthogonality of Trigonometric Functions

On the interval $[-\pi, \pi]$ (or any one period interval), the set of trigonometric functions $\{1, \cos x, \sin x, \cos 2x, \sin 2x, \ldots\}$ are mutually orthogonal:

$$\int_{-\pi}^{\pi} \cos(mx) \cos(nx) \, dx = \begin{cases} 0 & m \neq n \\ \pi & m = n \neq 0 \\ 2\pi & m = n = 0 \end{cases}$$

$$\int_{-\pi}^{\pi} \sin(mx) \sin(nx) \, dx = \begin{cases} 0 & m \neq n \\ \pi & m = n \neq 0 \end{cases}$$

$$\int_{-\pi}^{\pi} \cos(mx) \sin(nx) \, dx = 0 \quad \text{(for all } m, n \text{)}$$

This orthogonality relationship means that $\cos(mx)$ and $\sin(nx)$ point in different "directions" like basis vectors.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Numerical verification of orthogonality of trigonometric functions ---
x = np.linspace(-np.pi, np.pi, 10000)
dx = x[1] - x[0]

# Integral of cos(mx) * cos(nx)
print("=== Orthogonality check: cos(mx) * cos(nx) ===")
for m in range(0, 4):
    for n in range(0, 4):
        integral = np.trapz(np.cos(m * x) * np.cos(n * x), x)
        if abs(integral) > 1e-10:
            print(f"  m={m}, n={n}: {integral:.4f}")

# Integral of cos(mx) * sin(nx) (always 0)
print("\n=== Orthogonality check: cos(mx) * sin(nx) ===")
for m in range(1, 4):
    for n in range(1, 4):
        integral = np.trapz(np.cos(m * x) * np.sin(n * x), x)
        print(f"  m={m}, n={n}: {integral:.6f}")  # all ≈ 0

# Output:
# === Orthogonality check: cos(mx) * cos(nx) ===
#   m=0, n=0: 6.2832  (= 2*pi)
#   m=1, n=1: 3.1416  (= pi)
#   m=2, n=2: 3.1416  (= pi)
#   m=3, n=3: 3.1416  (= pi)
# === Orthogonality check: cos(mx) * sin(nx) ===
#   m=1, n=1: 0.000000
#   ...
```

### 1.3 Definition of Fourier Series

The **Fourier series** of a function $f(x)$ with period $2\pi$ is:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

where $a_0/2$ corresponds to the average value of $f(x)$. The coefficients $a_n$ and $b_n$ are determined using orthogonality.

For a function with period $2L$:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\!\left(\frac{n\pi x}{L}\right) + b_n \sin\!\left(\frac{n\pi x}{L}\right) \right]$$

---

## 2. Calculating Fourier Coefficients

### 2.1 Formulas for $a_0$, $a_n$, $b_n$

Using the orthogonality relations, the Fourier coefficients of a function $f(x)$ with period $2\pi$ are:

$$a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \, dx$$

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx \quad (n = 1, 2, 3, \ldots)$$

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx \quad (n = 1, 2, 3, \ldots)$$

**Derivation**: Multiply both sides by $\cos(mx)$ and integrate over $[-\pi, \pi]$. By orthogonality, only the $n = m$ term remains:

$$\int_{-\pi}^{\pi} f(x) \cos(mx) \, dx = a_m \cdot \pi \quad \Rightarrow \quad a_m = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(mx) \, dx$$

$b_n$ is derived in the same way by multiplying by $\sin(mx)$.

For a function with period $2L$:

$$a_0 = \frac{1}{L} \int_{-L}^{L} f(x) \, dx, \quad a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\!\left(\frac{n\pi x}{L}\right) dx, \quad b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\!\left(\frac{n\pi x}{L}\right) dx$$

### 2.2 Calculation Example: Square Wave

Define a **square wave** with period $2\pi$ as:

$$f(x) = \begin{cases} 1 & 0 < x < \pi \\ -1 & -\pi < x < 0 \end{cases}$$

$f(x)$ is an **odd function**, so $a_0 = 0$ and $a_n = 0$ (odd function $\times$ even function = odd function, so the integral over a symmetric interval is 0).

Calculating $b_n$:

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx = \frac{2}{\pi} \int_0^{\pi} \sin(nx) \, dx = \frac{2}{\pi} \left[-\frac{\cos(nx)}{n}\right]_0^{\pi}$$

$$= \frac{2}{n\pi} (1 - \cos(n\pi)) = \begin{cases} \frac{4}{n\pi} & n \text{ odd} \\ 0 & n \text{ even} \end{cases}$$

Therefore, the Fourier series of the square wave is:

$$f(x) = \frac{4}{\pi} \left( \sin x + \frac{\sin 3x}{3} + \frac{\sin 5x}{5} + \cdots \right) = \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{\sin\left[(2k+1)x\right]}{2k+1}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Visualize convergence of Fourier series for square wave ---
x = np.linspace(-2 * np.pi, 2 * np.pi, 2000)

# Original square wave (period 2*pi)
square_wave = np.sign(np.sin(x))

# Partial sum of Fourier series (up to N terms)
def fourier_square(x, N):
    """Partial sum of Fourier series for square wave (N sine terms)"""
    result = np.zeros_like(x)
    for k in range(N):
        n = 2 * k + 1  # odd terms only
        result += np.sin(n * x) / n
    return (4 / np.pi) * result

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
terms = [1, 3, 10, 50]

for ax, N in zip(axes.flat, terms):
    ax.plot(x, square_wave, 'k--', alpha=0.3, linewidth=1, label='Original function')
    ax.plot(x, fourier_square(x, N), 'b-', linewidth=1.5, label=f'N={N}')
    ax.set_title(f'Square wave Fourier series: {N} terms')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_square_wave.png', dpi=150)
plt.show()
```

### 2.3 Calculation Example: Sawtooth Wave

A **sawtooth wave** with period $2\pi$:

$$f(x) = x \quad (-\pi < x < \pi)$$

This is also an odd function, so $a_0 = 0$ and $a_n = 0$.

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} x \sin(nx) \, dx = \frac{2}{\pi} \int_0^{\pi} x \sin(nx) \, dx$$

Performing integration by parts:

$$= \frac{2}{\pi} \left[-\frac{x \cos(nx)}{n} + \frac{\sin(nx)}{n^2}\right]_0^{\pi} = \frac{2}{\pi} \cdot \left(-\frac{\pi \cos(n\pi)}{n}\right) = -\frac{2 \cos(n\pi)}{n} = \frac{2(-1)^{n+1}}{n}$$

Therefore:

$$f(x) = 2 \sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin(nx) = 2\left(\sin x - \frac{\sin 2x}{2} + \frac{\sin 3x}{3} - \cdots\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, pi, integrate, simplify

# --- Analytical coefficient computation using SymPy ---
x_sym, n_sym = symbols('x n', real=True)

# b_n coefficients for sawtooth wave
b_n_expr = (1 / pi) * integrate(x_sym * sin(n_sym * x_sym), (x_sym, -pi, pi))
print(f"b_n = {simplify(b_n_expr)}")
# Output: b_n = -2*cos(pi*n)/n = 2*(-1)^(n+1)/n

# --- Visualize Fourier series for sawtooth wave ---
x = np.linspace(-2 * np.pi, 2 * np.pi, 2000)

# Original sawtooth wave (period 2*pi)
sawtooth = x - 2 * np.pi * np.round(x / (2 * np.pi))

def fourier_sawtooth(x, N):
    """Partial sum of Fourier series for sawtooth wave"""
    result = np.zeros_like(x)
    for n in range(1, N + 1):
        result += ((-1) ** (n + 1) / n) * np.sin(n * x)
    return 2 * result

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, sawtooth, 'k--', alpha=0.3, linewidth=1, label='Original function')
for N, color in [(3, 'blue'), (10, 'green'), (50, 'red')]:
    ax.plot(x, fourier_sawtooth(x, N), color=color, linewidth=1.2, label=f'N={N}')

ax.set_title('Fourier series of sawtooth wave')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fourier_sawtooth_wave.png', dpi=150)
plt.show()
```

---

## 3. Convergence and Gibbs Phenomenon

### 3.1 Dirichlet Conditions

The **sufficient conditions** for the convergence of the Fourier series of a function $f(x)$ are called the **Dirichlet conditions**:

1. $f(x)$ has only a **finite number of discontinuities** in one period
2. $f(x)$ has only a **finite number of extrema (maxima and minima)** in one period
3. $\int_{-\pi}^{\pi} |f(x)| \, dx < \infty$ (**absolutely integrable**)

If these conditions are satisfied:
- At a **continuous point** $x$: The Fourier series converges to $f(x)$
- At a **discontinuous point** $x_0$: The Fourier series converges to the **arithmetic mean** of the left and right limits

$$S(x_0) = \frac{f(x_0^+) + f(x_0^-)}{2}$$

> Most functions dealt with in physical sciences satisfy the Dirichlet conditions.

### 3.2 Pointwise and Uniform Convergence

**Pointwise convergence**: At each point $x$, individually $S_N(x) \to f(x)$

$$\forall x, \forall \varepsilon > 0, \exists N_0(x, \varepsilon): N > N_0 \Rightarrow |S_N(x) - f(x)| < \varepsilon$$

where $N_0$ can vary depending on $x$.

**Uniform convergence**: Simultaneously at all points, converges independently of $x$

$$\forall \varepsilon > 0, \exists N_0(\varepsilon): N > N_0 \Rightarrow |S_N(x) - f(x)| < \varepsilon \quad \forall x$$

The Fourier series of a function with discontinuities **converges pointwise** but not **uniformly** (due to the Gibbs phenomenon).
The Fourier series of a continuous and piecewise smooth function **converges uniformly**.

### 3.3 Gibbs Phenomenon

Near a discontinuity, the partial sum of the Fourier series oscillates (overshoots) about **9%** above the original function value, and this overshoot does not disappear no matter how many terms are added. This is called the **Gibbs phenomenon**.

Exact overshoot ratio:

$$\text{overshoot} \approx \frac{2}{\pi} \int_0^{\pi} \frac{\sin t}{t} \, dt - 1 \approx 0.0895 \approx 8.95\%$$

This is a ratio relative to the size of the discontinuity (jump size). For a square wave, the jump size is 2, so the maximum overshoot of $S_N$ is about $1 + 2 \times 0.0895 = 1.179$.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Visualize Gibbs phenomenon ---
x = np.linspace(-0.5, 0.5, 5000)

def fourier_square_detail(x, N):
    """Partial sum of square wave (zoom near discontinuity at x=0)"""
    result = np.zeros_like(x)
    for k in range(N):
        n = 2 * k + 1
        result += np.sin(n * x) / n
    return (4 / np.pi) * result

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (left) Behavior near discontinuity for various N
for N in [10, 50, 200, 1000]:
    axes[0].plot(x, fourier_square_detail(x, N), linewidth=1, label=f'N={N}')
axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[0].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('Gibbs phenomenon: near discontinuity (x=0)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('$S_N(x)$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (right) Theoretical overshoot calculation
# Si(pi) = integral_0^pi sin(t)/t dt
Si_pi, _ = quad(lambda t: np.sin(t) / t, 0, np.pi)
overshoot = 2 * Si_pi / np.pi - 1
print(f"Theoretical overshoot: {overshoot:.4f} ({overshoot*100:.2f}%)")

# Maximum value as a function of N
N_values = np.arange(5, 505, 5)
max_values = []
x_fine = np.linspace(0.0001, 0.5, 10000)
for N in N_values:
    S_N = fourier_square_detail(x_fine, N)
    max_values.append(np.max(S_N))

axes[1].plot(N_values, max_values, 'b-', linewidth=1)
axes[1].axhline(y=1 + overshoot * 2, color='r', linestyle='--',
                label=f'Theoretical: {1 + overshoot*2:.4f}')
axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1].set_title('Maximum value vs number of terms (N)')
axes[1].set_xlabel('N (number of terms)')
axes[1].set_ylabel('$\\max S_N(x)$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gibbs_phenomenon.png', dpi=150)
plt.show()
```

---

## 4. Half-Range Expansion and Symmetry

### 4.1 Even Functions and Cosine Series

If $f(x)$ is an **even function**, i.e., $f(-x) = f(x)$:

- $f(x) \sin(nx)$ is an odd function, so $b_n = 0$
- The Fourier series contains only a **cosine series**:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos(nx)$$

$$a_n = \frac{2}{\pi} \int_0^{\pi} f(x) \cos(nx) \, dx$$

**Example**: $f(x) = |x|$ (period $2\pi$)

$$a_0 = \frac{2}{\pi} \int_0^{\pi} x \, dx = \pi$$

$$a_n = \frac{2}{\pi} \int_0^{\pi} x \cos(nx) \, dx = \frac{2}{n^2\pi} (\cos(n\pi) - 1) = \begin{cases} -\frac{4}{n^2\pi} & n \text{ odd} \\ 0 & n \text{ even} \end{cases}$$

$$|x| = \frac{\pi}{2} - \frac{4}{\pi}\left(\cos x + \frac{\cos 3x}{9} + \frac{\cos 5x}{25} + \cdots\right)$$

### 4.2 Odd Functions and Sine Series

If $f(x)$ is an **odd function**, i.e., $f(-x) = -f(x)$:

- $f(x) \cos(nx)$ is an odd function, so $a_n = 0$ (for all $n \ge 0$)
- The Fourier series contains only a **sine series**:

$$f(x) = \sum_{n=1}^{\infty} b_n \sin(nx)$$

$$b_n = \frac{2}{\pi} \int_0^{\pi} f(x) \sin(nx) \, dx$$

The square wave and sawtooth wave discussed earlier are examples of sine series for odd functions.

### 4.3 Half-Range Expansion

There are two ways to extend a function $f(x)$ defined only on the interval $[0, L]$ to the entire interval $[-L, L]$:

1. **Even extension**: Extend as $f(-x) = f(x)$ → **Cosine series**
2. **Odd extension**: Extend as $f(-x) = -f(x)$ → **Sine series**

In physics problems, the appropriate extension is chosen according to boundary conditions:
- **Dirichlet boundary condition** ($f(0) = f(L) = 0$): Use sine series
- **Neumann boundary condition** ($f'(0) = f'(L) = 0$): Use cosine series

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Visualize half-range expansion ---
# f(x) = x(pi - x), defined for 0 < x < pi

x_half = np.linspace(0, np.pi, 500)
f_half = x_half * (np.pi - x_half)

# Even extension (cosine series)
def cosine_expansion(x, N):
    """Cosine series for f(x) = x(pi - x)"""
    # a_0 = (2/pi) * integral_0^pi x(pi-x) dx = pi^2/3
    a0 = np.pi**2 / 3
    result = a0 / 2 * np.ones_like(x)
    for n in range(1, N + 1):
        # a_n: a_n = (2/pi) * integral_0^pi x(pi-x)*cos(nx) dx
        if n % 2 == 0:
            a_n = -4 / (n**2)
        else:
            a_n = -4 / (n**2)
        # Exact: a_n = -4/(n^2) (i.e. -2(1 + (-1)^n)/(n^2) for all n)
        a_n = -2 * (1 + (-1)**n) / (n**2)
        result += a_n * np.cos(n * x)
    return result

# Odd extension (sine series)
def sine_expansion(x, N):
    """Sine series for f(x) = x(pi - x)"""
    result = np.zeros_like(x)
    for n in range(1, N + 1):
        # b_n = (2/pi) * integral_0^pi x(pi-x)*sin(nx) dx
        if n % 2 == 1:
            b_n = 8 / (n**3 * np.pi)
        else:
            b_n = 0
        result += b_n * np.sin(n * x)
    return result

x_full = np.linspace(-np.pi, np.pi, 1000)
N = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Even extension
f_even = np.abs(x_full) * (np.pi - np.abs(x_full))
axes[0].plot(x_full, f_even, 'k--', alpha=0.3, label='Even extension (original)')
axes[0].plot(x_full, cosine_expansion(x_full, N), 'b-', label=f'Cosine series (N={N})')
axes[0].set_title('Even extension → Cosine series')
axes[0].set_xlabel('x')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Odd extension
f_odd = np.where(x_full >= 0,
                 x_full * (np.pi - x_full),
                 -(-x_full) * (np.pi - (-x_full)))
axes[1].plot(x_full, f_odd, 'k--', alpha=0.3, label='Odd extension (original)')
axes[1].plot(x_full, sine_expansion(x_full, N), 'r-', label=f'Sine series (N={N})')
axes[1].set_title('Odd extension → Sine series')
axes[1].set_xlabel('x')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('half_range_expansion.png', dpi=150)
plt.show()
```

---

## 5. Parseval's Theorem

### 5.1 Energy and Parseval's Identity

**Parseval's theorem** expresses that the "energy" of a function is the same in the time/space domain and the frequency domain:

$$\frac{1}{\pi} \int_{-\pi}^{\pi} |f(x)|^2 \, dx = \frac{a_0^2}{2} + \sum_{n=1}^{\infty} (a_n^2 + b_n^2)$$

Physical interpretation:
- **Left side**: "Mean square" (proportional to energy) over one period
- **Right side**: Sum of energy (square of amplitude) of each frequency component
- Total energy equals the sum of the energy of each frequency component

This is an infinite-dimensional extension of the Pythagorean theorem: just as the square of a vector's length is the sum of the squares of its components, the "norm squared" of a function is the sum of the squares of each Fourier component.

### 5.2 Application: Calculating Series Sums

Parseval's theorem can be used to calculate the sum of specific series.

**Example 1**: Applying Parseval's theorem to a square wave

For a square wave $f(x) = \pm 1$:
- Left side: $\frac{1}{\pi} \int_{-\pi}^{\pi} 1 \, dx = 2$
- Right side: $\sum_{k=0}^{\infty} b_{2k+1}^2 = \sum_{k=0}^{\infty} \left(\frac{4}{(2k+1)\pi}\right)^2 = \frac{16}{\pi^2} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2}$

Therefore:

$$2 = \frac{16}{\pi^2} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} \quad \Rightarrow \quad \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} = \frac{\pi^2}{8}$$

From this, we can obtain the famous result:

$$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{1}{1^2} + \frac{1}{2^2} + \frac{1}{3^2} + \cdots = \frac{\pi^2}{6}$$

(Derived by adding the sum of even terms $= \frac{1}{4}\sum 1/n^2$ to the sum of odd terms $\pi^2/8$.)

**Example 2**: Applying Parseval's theorem to sawtooth wave $f(x) = x$

$$\frac{1}{\pi}\int_{-\pi}^{\pi} x^2 \, dx = \frac{2\pi^2}{3} = \sum_{n=1}^{\infty} \frac{4}{n^2} \quad \Rightarrow \quad \sum_{n=1}^{\infty}\frac{1}{n^2} = \frac{\pi^2}{6}$$

```python
import numpy as np

# --- Numerical verification of Parseval's theorem ---
x = np.linspace(-np.pi, np.pi, 100000)
dx = x[1] - x[0]

# Square wave
f_square = np.sign(np.sin(x + 1e-15))
lhs = np.trapz(f_square**2, x) / np.pi
print(f"Square wave |f|^2 integral / pi = {lhs:.6f}")

# Right side: sum of squares of Fourier coefficients
rhs = 0
for k in range(10000):
    n = 2 * k + 1
    b_n = 4 / (n * np.pi)
    rhs += b_n**2
print(f"sum b_n^2 = {rhs:.6f}")
print(f"Error: {abs(lhs - rhs):.2e}")

# Compute series sums
series_sum = sum(1 / (2*k+1)**2 for k in range(100000))
print(f"\nsum 1/(2k+1)^2 = {series_sum:.10f}")
print(f"pi^2/8         = {np.pi**2/8:.10f}")

series_all = sum(1 / n**2 for n in range(1, 100001))
print(f"\nsum 1/n^2      = {series_all:.10f}")
print(f"pi^2/6         = {np.pi**2/6:.10f}")

# Output:
# Square wave |f|^2 integral / pi = 2.000000
# sum b_n^2 = 1.999999
# Error: 5.07e-05
# sum 1/(2k+1)^2 = 1.2337005501
# pi^2/8         = 1.2337005501
# sum 1/n^2      = 1.6449240669
# pi^2/6         = 1.6449340668
```

---

## 6. Complex Fourier Series

### 6.1 Complex Exponential Form

Using Euler's formula $e^{inx} = \cos(nx) + i\sin(nx)$, the Fourier series can be written in a more compact **complex exponential** form:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n \, e^{inx}$$

where the complex Fourier coefficients $c_n$ are:

$$c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) \, e^{-inx} \, dx$$

Relationship with real coefficients:

$$c_0 = \frac{a_0}{2}, \quad c_n = \frac{a_n - ib_n}{2}, \quad c_{-n} = \frac{a_n + ib_n}{2} = c_n^* \quad (n > 0)$$

If $f(x)$ is a real function, then $c_{-n} = c_n^*$ (complex conjugate).

Advantages of the complex form:
- Mathematically more compact and symmetric
- Simultaneously represents positive and negative frequencies
- Natural extension to the continuous Fourier transform

### 6.2 Frequency Spectrum

The **frequency spectrum** is a graph showing the magnitude of each frequency component:

- **Amplitude spectrum**: $|c_n|$ vs. $n$
- **Phase spectrum**: $\arg(c_n)$ vs. $n$
- **Power spectrum**: $|c_n|^2$ vs. $n$

Complex form of Parseval's theorem:

$$\frac{1}{2\pi} \int_{-\pi}^{\pi} |f(x)|^2 \, dx = \sum_{n=-\infty}^{\infty} |c_n|^2$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Complex Fourier coefficients and frequency spectrum ---
# Compare spectra of square, sawtooth, and triangle waves

x = np.linspace(-np.pi, np.pi, 10000, endpoint=False)
dx = x[1] - x[0]

# Define waveforms
square = np.sign(np.sin(x + 1e-15))
sawtooth = x / np.pi  # normalized
triangle = 1 - 2 * np.abs(x) / np.pi

# Numerically compute complex Fourier coefficients
def compute_cn(f, x, n_max):
    """Numerically compute complex Fourier coefficients c_n"""
    cn = {}
    for n in range(-n_max, n_max + 1):
        integrand = f * np.exp(-1j * n * x)
        cn[n] = np.trapz(integrand, x) / (2 * np.pi)
    return cn

n_max = 20
cn_square = compute_cn(square, x, n_max)
cn_sawtooth = compute_cn(sawtooth, x, n_max)
cn_triangle = compute_cn(triangle, x, n_max)

# Visualize spectra
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
waveforms = [('Square wave', cn_square), ('Sawtooth wave', cn_sawtooth), ('Triangle wave', cn_triangle)]

for j, (name, cn) in enumerate(waveforms):
    ns = sorted(cn.keys())
    amplitudes = [np.abs(cn[n]) for n in ns]
    phases = [np.angle(cn[n]) for n in ns]

    # Amplitude spectrum
    axes[0, j].stem(ns, amplitudes, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0, j].set_title(f'{name} - Amplitude spectrum')
    axes[0, j].set_xlabel('n (frequency)')
    axes[0, j].set_ylabel('$|c_n|$')
    axes[0, j].grid(True, alpha=0.3)

    # Phase spectrum
    # Filter out components with very small amplitude
    significant = [n for n in ns if np.abs(cn[n]) > 1e-10]
    sig_phases = [np.angle(cn[n]) for n in significant]
    axes[1, j].stem(significant, sig_phases, linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[1, j].set_title(f'{name} - Phase spectrum')
    axes[1, j].set_xlabel('n (frequency)')
    axes[1, j].set_ylabel('$\\arg(c_n)$')
    axes[1, j].set_ylim(-np.pi - 0.3, np.pi + 0.3)
    axes[1, j].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_spectra.png', dpi=150)
plt.show()
```

---

## 7. Physics Applications

### 7.1 Vibrating String (Standing Waves)

The vibration of a string of length $L$ fixed at both ends is described by the **wave equation**:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

Given boundary conditions $u(0, t) = u(L, t) = 0$ and initial conditions $u(x, 0) = f(x)$, $\frac{\partial u}{\partial t}(x, 0) = 0$, the solution by separation of variables is:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\!\left(\frac{n\pi x}{L}\right) \cos\!\left(\frac{n\pi ct}{L}\right)$$

where $b_n$ are the **sine series (half-range expansion)** coefficients of the initial displacement $f(x)$:

$$b_n = \frac{2}{L} \int_0^L f(x) \sin\!\left(\frac{n\pi x}{L}\right) dx$$

Physical meaning:
- $n = 1$: **Fundamental frequency** $f_1 = c/(2L)$
- $n = 2, 3, \ldots$: **Harmonics** $f_n = n f_1$
- The magnitude of each $b_n$ determines the contribution of that harmonic

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Vibrating string: Fourier decomposition of initial displacement ---
L = 1.0  # String length
c = 1.0  # Wave speed
N = 20   # Number of harmonics

# Initial displacement: triangle (plucked at center)
def initial_displacement(x, L):
    """Triangular initial displacement: maximum at center"""
    return np.where(x <= L/2, 2*x/L, 2*(L-x)/L)

# Compute sine series coefficients
x_int = np.linspace(0, L, 10000)
f_init = initial_displacement(x_int, L)

b_n = np.zeros(N + 1)
for n in range(1, N + 1):
    integrand = f_init * np.sin(n * np.pi * x_int / L)
    b_n[n] = 2 / L * np.trapz(integrand, x_int)

print("Harmonic coefficients b_n:")
for n in range(1, 8):
    print(f"  b_{n} = {b_n[n]:+.6f}")

# Displacement at time t
def string_displacement(x, t, b_n, L, c):
    u = np.zeros_like(x)
    for n in range(1, len(b_n)):
        u += b_n[n] * np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)
    return u

# Visualize string shape at multiple times
x = np.linspace(0, L, 500)
times = [0, 0.1, 0.25, 0.4, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for t in times:
    u = string_displacement(x, t, b_n, L, c)
    axes[0].plot(x, u, label=f't = {t:.2f}')

axes[0].set_title('Vibrating string: time evolution')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u(x, t)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Harmonic spectrum
axes[1].bar(range(1, N + 1), np.abs(b_n[1:]), color='steelblue', alpha=0.7)
axes[1].set_title('Harmonic coefficients $|b_n|$')
axes[1].set_xlabel('Harmonic number n')
axes[1].set_ylabel('$|b_n|$')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vibrating_string.png', dpi=150)
plt.show()
```

### 7.2 Initial Conditions in Heat Conduction Problems

The **heat equation** for a one-dimensional rod of length $L$:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

where $\alpha$ is the **thermal diffusivity**.

The solution for boundary conditions $u(0, t) = u(L, t) = 0$ (fixed temperature at both ends) and initial condition $u(x, 0) = f(x)$:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\!\left(\frac{n\pi x}{L}\right) \exp\!\left(-\frac{n^2 \pi^2 \alpha t}{L^2}\right)$$

Unlike the wave equation, in the heat equation, harmonic components undergo **exponential decay** at a rate proportional to $n^2$. Therefore:
- High-frequency components decay quickly
- Over time, only the fundamental mode ($n = 1$) remains
- This is the mathematical expression of the phenomenon of heat "spreading"

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier series solution to heat conduction problem ---
L = 1.0
alpha = 0.01  # Thermal diffusivity

# Initial temperature: heat source concentrated at center
def initial_temp(x, L):
    """Gaussian initial temperature distribution"""
    return np.exp(-50 * (x - L/2)**2)

# Sine series coefficients
x_int = np.linspace(0, L, 10000)
f_init = initial_temp(x_int, L)
N = 50

b_n = np.zeros(N + 1)
for n in range(1, N + 1):
    integrand = f_init * np.sin(n * np.pi * x_int / L)
    b_n[n] = 2 / L * np.trapz(integrand, x_int)

# Heat equation solution
def heat_solution(x, t, b_n, L, alpha):
    u = np.zeros_like(x)
    for n in range(1, len(b_n)):
        decay = np.exp(-n**2 * np.pi**2 * alpha * t / L**2)
        u += b_n[n] * np.sin(n * np.pi * x / L) * decay
    return u

# Visualization
x = np.linspace(0, L, 500)
times = [0, 0.5, 1.0, 2.0, 5.0, 10.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for t in times:
    u = heat_solution(x, t, b_n, L, alpha)
    axes[0].plot(x, u, label=f't = {t:.1f}')

axes[0].set_title('Heat conduction: time evolution of temperature')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u(x, t)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Decay of each mode
t_range = np.linspace(0, 10, 200)
for n in [1, 2, 3, 5, 10]:
    decay = np.abs(b_n[n]) * np.exp(-n**2 * np.pi**2 * alpha * t_range / L**2)
    axes[1].plot(t_range, decay, label=f'n={n} (|b_{n}|={np.abs(b_n[n]):.3f})')

axes[1].set_title('Exponential decay of harmonic modes')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('$|b_n| e^{-n^2 \\pi^2 \\alpha t / L^2}$')
axes[1].legend()
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heat_conduction_fourier.png', dpi=150)
plt.show()
```

### 7.3 Frequency Analysis of Electromagnetic Waves

Frequency component analysis of periodic electromagnetic signals is an important application of Fourier series.

**AM modulated signal (Amplitude Modulated signal)**:

$$s(t) = [1 + m \cos(\omega_m t)] \cos(\omega_c t)$$

where $\omega_c$ is the **carrier frequency**, $\omega_m$ is the **modulation frequency**, and $m$ is the **modulation index**.

Expanding the product of trigonometric functions:

$$s(t) = \cos(\omega_c t) + \frac{m}{2}\cos[(\omega_c + \omega_m)t] + \frac{m}{2}\cos[(\omega_c - \omega_m)t]$$

Thus, the product in the time domain appears as frequency components of **sum and difference** in the frequency domain.

**Fourier series of a periodic pulse train**:

For a rectangular pulse train with period $T$ and width $\tau$:

$$f(t) = \frac{\tau}{T} + \frac{2}{T}\sum_{n=1}^{\infty} \frac{\sin(n\pi\tau/T)}{n\pi/T} \cos\!\left(\frac{2\pi n t}{T}\right)$$

where a $\text{sinc}$ function-shaped envelope determines the frequency spectrum.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- EM wave frequency analysis ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) AM modulated signal
t = np.linspace(0, 1, 10000)
fc = 50   # Carrier frequency (Hz)
fm = 5    # Modulation frequency (Hz)
m = 0.7   # Modulation index

signal = (1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2 * np.pi * fc * t)

axes[0, 0].plot(t[:2000], signal[:2000], 'b-', linewidth=0.5)
axes[0, 0].set_title('AM modulated signal (time domain)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('s(t)')
axes[0, 0].grid(True, alpha=0.3)

# Spectral analysis using FFT
N_fft = len(t)
freq = np.fft.fftfreq(N_fft, d=t[1] - t[0])
S = np.fft.fft(signal) / N_fft

# Show only positive frequencies
pos_mask = freq > 0
axes[0, 1].plot(freq[pos_mask], 2 * np.abs(S[pos_mask]), 'r-', linewidth=1)
axes[0, 1].set_title('AM modulated signal (frequency spectrum)')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_xlim(0, 100)
axes[0, 1].grid(True, alpha=0.3)
# Mark carrier (fc), upper sideband (fc+fm), lower sideband (fc-fm)
for f_label, label in [(fc, '$f_c$'), (fc+fm, '$f_c+f_m$'), (fc-fm, '$f_c-f_m$')]:
    axes[0, 1].axvline(x=f_label, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].text(f_label, 0.45, label, ha='center', fontsize=9)

# (2) Rectangular pulse train spectrum
T_pulse = 0.1    # Period (s)
tau = 0.02        # Pulse width (s)
t2 = np.linspace(0, 0.5, 10000)

# Generate rectangular pulse train
pulse_train = np.zeros_like(t2)
for k in range(int(0.5 / T_pulse) + 1):
    center = k * T_pulse
    pulse_train[(t2 >= center) & (t2 < center + tau)] = 1.0

axes[1, 0].plot(t2, pulse_train, 'b-', linewidth=1)
axes[1, 0].set_title(f'Rectangular pulse train (T={T_pulse}s, τ={tau}s)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('f(t)')
axes[1, 0].set_ylim(-0.2, 1.4)
axes[1, 0].grid(True, alpha=0.3)

# FFT of rectangular pulse train
N_fft2 = len(t2)
freq2 = np.fft.fftfreq(N_fft2, d=t2[1] - t2[0])
S2 = np.fft.fft(pulse_train) / N_fft2

pos_mask2 = freq2 > 0
axes[1, 1].stem(freq2[pos_mask2][:50], 2 * np.abs(S2[pos_mask2][:50]),
                linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1, 1].set_title('Frequency spectrum of rectangular pulse train')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].grid(True, alpha=0.3)

# Show sinc envelope
f_env = np.linspace(0.1, freq2[pos_mask2][49], 500)
envelope = tau / T_pulse * np.abs(np.sinc(f_env * tau))
axes[1, 1].plot(f_env, 2 * envelope, 'b--', alpha=0.5, linewidth=2, label='sinc envelope')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('em_wave_fourier.png', dpi=150)
plt.show()
```

---

## Practice Problems

### Problem 1: Fourier Coefficient Calculation

Find the Fourier series of the function $f(x) = x^2$ with period $2\pi$ ($-\pi < x < \pi$).

**Hint**: $f(x) = x^2$ is an even function, so only the cosine series remains. Perform integration by parts twice.

**Expected result**: $x^2 = \frac{\pi^2}{3} + 4\sum_{n=1}^{\infty} \frac{(-1)^n}{n^2}\cos(nx)$

### Problem 2: Application of Parseval's Theorem

Apply Parseval's theorem to the result of Problem 1 to find the value of $\sum_{n=1}^{\infty} \frac{1}{n^4}$.

**Hint**: Start from $\frac{1}{\pi}\int_{-\pi}^{\pi} x^4 \, dx = \frac{a_0^2}{2} + \sum a_n^2$.

**Expected result**: $\sum_{n=1}^{\infty} \frac{1}{n^4} = \frac{\pi^4}{90}$

### Problem 3: Half-Range Expansion

Expand $f(x) = \cos x$ on the interval $[0, \pi]$ as a **sine series**.

**Hint**: Calculate $b_n$ after odd extension. Distinguish between the case when $n = 1$ and when $n \neq 1$.

### Problem 4: Gibbs Phenomenon Analysis

For the Fourier series of the sawtooth wave $f(x) = x$ ($-\pi < x < \pi$), show that the overshoot of the $N$-term partial sum near $x = \pi$ is related to $\text{Si}(\pi) \approx 1.8519$.

**Hint**: Write $S_N(x)$ in integral form and analyze the limit as $x \to \pi^-$.

### Problem 5: Physics Application — Vibrating String

A string of length $L = 1$ m has an initial displacement of $f(x) = A\sin(\pi x/L)\sin(3\pi x/L)$ and is released from rest.

(a) Expand the initial displacement as a sine series. (Use trigonometric formulas to convert products to differences)

(b) Find $u(x, t)$ and show that this vibration is the sum of two standing waves.

### Problem 6: Heat Conduction

The initial temperature of a rod of length $L$ is $u(x, 0) = u_0$ (constant), and for $t > 0$, both ends are maintained at $0$°C.

(a) Find the sine series expansion of $u_0$.

(b) Analyze the rate at which the temperature approaches 0 as $t \to \infty$. (Decay time constant of the fundamental mode $\tau_1 = L^2 / (\pi^2 \alpha)$)

```python
# --- Auxiliary code for practice problems ---
import numpy as np
from sympy import symbols, cos, sin, pi, integrate, simplify, Rational

x, n = symbols('x n', real=True, positive=True)

# Problem 1: Fourier coefficients of f(x) = x^2
print("=== Problem 1: f(x) = x^2 ===")
a0 = (1 / pi) * integrate(x**2, (x, -pi, pi))
print(f"a_0 = {a0}")

a_n = (1 / pi) * integrate(x**2 * cos(n * x), (x, -pi, pi))
a_n_simplified = simplify(a_n)
print(f"a_n = {a_n_simplified}")

# Problem 2: Applying Parseval's theorem
print("\n=== Problem 2: Parseval's theorem ===")
lhs = (1 / pi) * integrate(x**4, (x, -pi, pi))
print(f"(1/pi) * integral x^4 dx = {lhs}")

# a_0^2/2 + sum a_n^2 = 2*pi^4/5
# pi^4/9/2 + 16 * sum 1/n^4 = 2*pi^4/5
# => sum 1/n^4 = (2*pi^4/5 - pi^4/18) / 16 = pi^4/90
print(f"sum 1/n^4 = pi^4/90 = {float(pi**4/90):.10f}")
print(f"Numerical check: {sum(1/k**4 for k in range(1, 100001)):.10f}")
```

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 7. Wiley.
   - Main reference for this lesson
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 19. Academic Press.
   - More rigorous mathematical treatment
3. **Kreyszig, E.** (2011). *Advanced Engineering Mathematics*, 10th ed., Chapters 11-12. Wiley.
   - Engineering perspective on Fourier analysis

### Online Resources
1. **3Blue1Brown**: *But what is a Fourier series?* — Intuitive visualization
2. **MIT OCW 18.03**: Fourier Series lectures
3. **Paul's Online Math Notes**: Fourier Series reference
4. **Wolfram MathWorld**: Fourier Series definitions and formulas

### Key Formula Summary

| Formula | Expression |
|------|------|
| Fourier series | $f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}[a_n\cos(nx) + b_n\sin(nx)]$ |
| $a_n$ | $\frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx$ |
| $b_n$ | $\frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx$ |
| Complex form | $f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}$ |
| Parseval | $\frac{1}{\pi}\int_{-\pi}^{\pi}|f|^2\,dx = \frac{a_0^2}{2} + \sum(a_n^2 + b_n^2)$ |

---

## Next Lesson

[06. Fourier Transforms](06_Fourier_Transforms.md) extends Fourier series, which is limited to periodic functions, to **non-periodic functions** using the **Fourier transform**. We will study the continuous Fourier transform, discrete Fourier transform (DFT), fast Fourier transform (FFT), and the convolution theorem.

# 11. Series Solutions and Special Functions

## Learning Objectives

- Distinguish between **ordinary points** and **singular points**, and apply the **power series method** around ordinary points
- Solve for series solutions around **regular singular points** using the **Frobenius method**, and handle the three cases of the **indicial equation**
- Understand the properties, orthogonality, and recurrence relations of **Bessel functions** $J_n(x)$, $Y_n(x)$ and apply them to cylindrical coordinate problems
- Understand Rodrigues' formula, generating functions, and orthogonality of **Legendre polynomials** $P_l(x)$ and apply them to spherical coordinate problems
- Understand how **Hermite functions** and **Laguerre functions** appear in quantum mechanics problems (harmonic oscillator and hydrogen atom)
- Know the definitions, properties, and relationships between **gamma functions** and **beta functions**, and apply them to integral calculations

---

## 1. Power Series Method

### 1.1 Ordinary Points and Singular Points

Consider the standard form of a second-order linear ODE:

$$y'' + P(x)y' + Q(x)y = 0$$

Classification at point $x = x_0$:

| Classification | Condition | Example |
|---|---|---|
| **Ordinary point** | $P(x)$, $Q(x)$ are analytic at $x_0$ | $x_0 = 0$ for $y'' + y = 0$ |
| **Regular singular point** | $(x-x_0)P(x)$, $(x-x_0)^2 Q(x)$ are analytic | $x_0 = 0$ for Bessel's equation |
| **Irregular singular point** | Neither of the above | $x_0 = 0$ for $x^3 y'' + y = 0$ |

At ordinary points, we use the **power series method**; at regular singular points, we use the **Frobenius method**.

```python
import sympy as sp

# --- Singular point classification ---
x = sp.Symbol('x')

# Example: Bessel's equation x^2 y'' + x y' + (x^2 - n^2) y = 0
# Standard form: y'' + (1/x) y' + (1 - n^2/x^2) y = 0
n = sp.Symbol('n', positive=True)
P_bessel = 1 / x
Q_bessel = 1 - n**2 / x**2

# Classification at x = 0
xP = sp.simplify(x * P_bessel)   # x * (1/x) = 1 (analytic)
x2Q = sp.simplify(x**2 * Q_bessel)  # x^2 * (1 - n^2/x^2) = x^2 - n^2 (analytic)

print(f"Bessel's equation (x=0):")
print(f"  P(x) = {P_bessel}  →  singular at x=0")
print(f"  x·P(x) = {xP}  →  analytic")
print(f"  x²·Q(x) = {x2Q}  →  analytic")
print(f"  Conclusion: x=0 is a regular singular point")

# Example: Airy equation y'' - x y = 0
P_airy = 0
Q_airy = -x

print(f"\nAiry equation (x=0):")
print(f"  P(x) = {P_airy}, Q(x) = {Q_airy}")
print(f"  Both analytic at x=0 → x=0 is an ordinary point")
```

### 1.2 Series Solutions Around Ordinary Points

Around an ordinary point $x_0 = 0$, we assume a solution as a power series:

$$y = \sum_{n=0}^{\infty} a_n x^n$$

Substitute this into the ODE to derive a **recurrence relation** and determine the coefficients $a_n$ sequentially.

**Example: Airy Equation** $y'' - xy = 0$

$$\sum_{n=2}^{\infty} n(n-1) a_n x^{n-2} - \sum_{n=0}^{\infty} a_n x^{n+1} = 0$$

Aligning the indices (with substitution $m = n-2$):

$$\sum_{m=0}^{\infty} (m+2)(m+1) a_{m+2} x^m - \sum_{m=1}^{\infty} a_{m-1} x^m = 0$$

$x^0$ term: $2 \cdot 1 \cdot a_2 = 0 \;\Rightarrow\; a_2 = 0$

$x^m$ ($m \geq 1$) term: $(m+2)(m+1) a_{m+2} = a_{m-1}$

$$\boxed{a_{m+2} = \frac{a_{m-1}}{(m+2)(m+1)}, \quad m \geq 1}$$

$a_0$ and $a_1$ are free parameters (initial conditions), so we obtain two linearly independent solutions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# --- Power series solution of the Airy equation ---
def airy_series(x_val, a0, a1, N_terms=30):
    """Power series solution of the Airy equation y'' - xy = 0"""
    a = np.zeros(N_terms)
    a[0] = a0
    a[1] = a1
    a[2] = 0  # a_2 = 0 always

    # Recurrence relation: a[m+2] = a[m-1] / ((m+2)*(m+1)), m >= 1
    for m in range(1, N_terms - 2):
        a[m + 2] = a[m - 1] / ((m + 2) * (m + 1))

    result = sum(a[k] * x_val**k for k in range(N_terms))
    return result

# Compare with SciPy Airy functions
x = np.linspace(-15, 5, 500)
Ai, Aip, Bi, Bip = airy(x)

# Series solution (within convergence range)
x_series = np.linspace(-8, 5, 300)
y_series_Ai = np.array([airy_series(xi, 1, 0, N_terms=60) for xi in x_series])
y_series_Bi = np.array([airy_series(xi, 0, 1, N_terms=60) for xi in x_series])

# Normalization constants (to match SciPy Airy functions)
c_Ai = airy(0)[0]  # Ai(0)
c_Bi = airy(0)[2]  # Bi(0)
c_Ai_prime = airy(0)[1]  # Ai'(0)
c_Bi_prime = airy(0)[3]  # Bi'(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ai(x): Airy function of the first kind
axes[0].plot(x, Ai, 'b-', linewidth=2, label='Ai(x) (SciPy)')
y_approx = c_Ai * np.array([airy_series(xi, 1, 0, 60) for xi in x_series]) \
         + c_Ai_prime * np.array([airy_series(xi, 0, 1, 60) for xi in x_series])
axes[0].plot(x_series, y_approx, 'r--', linewidth=1.5, label='Series solution (60 terms)')
axes[0].set_xlim(-15, 5)
axes[0].set_ylim(-0.6, 0.8)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Ai(x)')
axes[0].set_title('Airy Function Ai(x)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bi(x): Airy function of the second kind
axes[1].plot(x, Bi, 'b-', linewidth=2, label='Bi(x) (SciPy)')
axes[1].set_xlim(-15, 5)
axes[1].set_ylim(-0.6, 1.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Bi(x)')
axes[1].set_title('Airy Function Bi(x)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Solutions of the Airy Equation $y\'\' - xy = 0$', fontsize=14)
plt.tight_layout()
plt.savefig('airy_functions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2. Frobenius Method

### 2.1 Regular Singular Points

Around a regular singular point $x_0 = 0$, a general power series may not converge. Instead, we assume a **Frobenius series**:

$$y = x^s \sum_{n=0}^{\infty} a_n x^n = \sum_{n=0}^{\infty} a_n x^{n+s}, \quad a_0 \neq 0$$

Here, $s$ is an unknown **index** to be determined by substituting into the ODE.

### 2.2 Indicial Equation

Write the ODE in the form $x^2 y'' + x b(x) y' + c(x) y = 0$ (where $b(x) = xP(x)$, $c(x) = x^2 Q(x)$), and substitute the Frobenius series. The coefficient of the lowest power ($x^s$) gives the **indicial equation**:

$$\boxed{s(s-1) + b_0 s + c_0 = 0}$$

where $b_0 = b(0)$, $c_0 = c(0)$. The two roots $s_1, s_2$ ($s_1 \geq s_2$) of this quadratic equation determine the form of the series solutions.

### 2.3 Three Cases

The difference $s_1 - s_2$ between the two roots of the indicial equation leads to three cases:

| Case | Condition | Form of Solutions |
|---|---|---|
| **Case 1** | $s_1 - s_2 \notin \mathbb{Z}$ (non-integer) | $y_1 = x^{s_1}\sum a_n x^n$, $y_2 = x^{s_2}\sum b_n x^n$ |
| **Case 2** | $s_1 = s_2$ (repeated root) | $y_1 = x^s \sum a_n x^n$, $y_2 = y_1 \ln x + x^s \sum b_n x^n$ |
| **Case 3** | $s_1 - s_2 \in \mathbb{Z}^+$ (positive integer) | $y_2$ may or may not contain a $\ln x$ term |

**Example: Bessel's Equation** $x^2 y'' + x y' + (x^2 - \nu^2)y = 0$

$b(x) = 1$, $c(x) = x^2 - \nu^2$, so $b_0 = 1$, $c_0 = -\nu^2$.

Indicial equation: $s(s-1) + s - \nu^2 = s^2 - \nu^2 = 0$

$$s_1 = \nu, \quad s_2 = -\nu$$

```python
import sympy as sp

# --- Frobenius method: indicial equation of Bessel's equation ---
s, nu = sp.symbols('s nu', real=True)

# Indicial equation: s(s-1) + b0*s + c0 = 0
b0 = 1        # b(0) = 1
c0 = -nu**2   # c(0) = -nu^2

indicial_eq = s*(s - 1) + b0*s + c0
print(f"Indicial equation: {sp.expand(indicial_eq)} = 0")
roots = sp.solve(indicial_eq, s)
print(f"Indicial roots: s1 = {roots[1]}, s2 = {roots[0]}")

# nu = 0: repeated root (Case 2) → second kind Bessel has ln term
# nu = 1/2: non-integer difference (Case 1) → two independent series solutions
# nu = 1: integer difference (Case 3) → requires care
for nu_val in [0, sp.Rational(1, 2), 1, 2]:
    diff = 2 * nu_val  # s1 - s2 = 2*nu
    case = "Case 2 (repeated root)" if diff == 0 else \
           ("Case 1 (non-integer)" if not diff.is_integer else "Case 3 (integer difference)")
    print(f"  nu = {nu_val}: s1-s2 = {diff} → {case}")
```

---

## 3. Bessel Functions

### 3.1 Bessel's Equation

**Bessel's equation** arises naturally when separating variables in Laplace's equation or Helmholtz equation in cylindrical coordinates:

$$x^2 y'' + x y' + (x^2 - \nu^2) y = 0$$

Here, $\nu \geq 0$ is the **order**. This equation is one of the most commonly encountered second-order ODEs in physics.

### 3.2 Bessel Functions of the First and Second Kind

The **Bessel function of the first kind** $J_\nu(x)$ is the solution with index $s = \nu$ obtained by the Frobenius method:

$$J_\nu(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k! \, \Gamma(k + \nu + 1)} \left(\frac{x}{2}\right)^{2k + \nu}$$

The **Bessel function of the second kind** $Y_\nu(x)$ (Neumann function) is the second solution linearly independent of $J_\nu$:

$$Y_\nu(x) = \frac{J_\nu(x) \cos(\nu\pi) - J_{-\nu}(x)}{\sin(\nu\pi)}$$

When $\nu$ is an integer, this formula becomes the indeterminate form $0/0$, so it is defined as a limit.

**Key Properties**:
- $J_\nu(x)$: finite at $x = 0$ (when $\nu \geq 0$)
- $Y_\nu(x)$: diverges to $-\infty$ as $x \to 0^+$
- Asymptotic behavior ($x \to \infty$): $J_\nu(x) \sim \sqrt{\frac{2}{\pi x}} \cos\left(x - \frac{\nu\pi}{2} - \frac{\pi}{4}\right)$

**Recurrence relations**:

$$J_{\nu-1}(x) + J_{\nu+1}(x) = \frac{2\nu}{x} J_\nu(x)$$

$$J_{\nu-1}(x) - J_{\nu+1}(x) = 2 J_\nu'(x)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jn_zeros

# --- Bessel function visualization ---
x = np.linspace(0.01, 20, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bessel functions of the first kind
for nu in [0, 1, 2, 3]:
    axes[0].plot(x, jv(nu, x), linewidth=1.5, label=f'$J_{{{nu}}}(x)$')
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].set_xlim(0, 20)
axes[0].set_ylim(-0.5, 1.1)
axes[0].set_xlabel('x')
axes[0].set_ylabel('$J_\\nu(x)$')
axes[0].set_title('Bessel Functions of the First Kind $J_\\nu(x)$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bessel functions of the second kind
for nu in [0, 1, 2]:
    axes[1].plot(x, yv(nu, x), linewidth=1.5, label=f'$Y_{{{nu}}}(x)$')
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].set_xlim(0, 20)
axes[1].set_ylim(-1.5, 0.8)
axes[1].set_xlabel('x')
axes[1].set_ylabel('$Y_\\nu(x)$')
axes[1].set_title('Bessel Functions of the Second Kind $Y_\\nu(x)$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bessel_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Recurrence relation verification ---
nu_test = 2
x_test = np.linspace(1, 10, 100)

# J_{nu-1} + J_{nu+1} = (2*nu/x) * J_nu
lhs = jv(nu_test - 1, x_test) + jv(nu_test + 1, x_test)
rhs = (2 * nu_test / x_test) * jv(nu_test, x_test)
print(f"Recurrence relation verification (nu={nu_test}):")
print(f"  max|LHS - RHS| = {np.max(np.abs(lhs - rhs)):.2e}")
```

### 3.3 Orthogonality and Series Expansion

If the zeros of $J_\nu$ are denoted $\alpha_{\nu,1} < \alpha_{\nu,2} < \cdots$, then the following **orthogonality** holds on the interval $[0, a]$:

$$\int_0^a x \, J_\nu\!\left(\frac{\alpha_{\nu,m}}{a} x\right) J_\nu\!\left(\frac{\alpha_{\nu,n}}{a} x\right) dx = \begin{cases} 0 & m \neq n \\ \frac{a^2}{2} [J_{\nu+1}(\alpha_{\nu,n})]^2 & m = n \end{cases}$$

Using this, a function $f(r)$ defined on the interval $[0, a]$ can be expanded in a **Fourier-Bessel series**:

$$f(r) = \sum_{n=1}^{\infty} c_n J_\nu\!\left(\frac{\alpha_{\nu,n}}{a} r\right)$$

$$c_n = \frac{2}{a^2 [J_{\nu+1}(\alpha_{\nu,n})]^2} \int_0^a r \, f(r) \, J_\nu\!\left(\frac{\alpha_{\nu,n}}{a} r\right) dr$$

```python
import numpy as np
from scipy.special import jv, jn_zeros
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- Fourier-Bessel series example: expand f(r) = 1 in J_0 series ---
a = 1.0  # interval [0, a]
nu = 0
N_terms = 20

# Zeros of J_0
zeros = jn_zeros(nu, N_terms)

def compute_bessel_coeff(n_idx, func, a, nu):
    """Compute Fourier-Bessel coefficient c_n"""
    alpha_n = zeros[n_idx]
    norm = (a**2 / 2) * jv(nu + 1, alpha_n)**2

    integrand = lambda r: r * func(r) * jv(nu, alpha_n * r / a)
    integral, _ = quad(integrand, 0, a)

    return integral / norm

# f(r) = 1
f = lambda r: 1.0
coeffs = [compute_bessel_coeff(n, f, a, nu) for n in range(N_terms)]

# Reconstruct series
r = np.linspace(0, a, 200)
f_approx = {5: None, 10: None, 20: None}

for N in f_approx:
    f_approx[N] = sum(coeffs[n] * jv(nu, zeros[n] * r / a) for n in range(N))

plt.figure(figsize=(10, 5))
plt.axhline(y=1.0, color='k', linewidth=2, label='$f(r) = 1$')
for N, color in zip([5, 10, 20], ['red', 'blue', 'green']):
    plt.plot(r, f_approx[N], color=color, linewidth=1.5,
             label=f'Bessel series ({N} terms)')
plt.xlabel('r')
plt.ylabel('f(r)')
plt.title('Fourier-Bessel Series Expansion: $f(r) = 1$ on $[0, 1]$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fourier_bessel_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.4 Physics Applications (Circular Membrane Vibration, Cylindrical Problems)

**Circular membrane vibration**: The vibration of a circular membrane of radius $a$ with fixed boundary is obtained by solving the 2D wave equation in polar coordinates:

$$u(r, \theta, t) = \sum_{m=0}^{\infty} \sum_{n=1}^{\infty} J_m\!\left(\frac{\alpha_{m,n}}{a} r\right) \left(A_{mn} \cos m\theta + B_{mn} \sin m\theta\right) \cos(\omega_{mn} t)$$

where $\omega_{mn} = c \, \alpha_{m,n}/a$ and $c$ is the wave speed. The boundary condition $u(a, \theta, t) = 0$ requires $J_m(\alpha_{m,n}) = 0$.

```python
import numpy as np
from scipy.special import jv, jn_zeros
import matplotlib.pyplot as plt

# --- Visualization of standing vibration modes of a circular membrane ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                         subplot_kw={'projection': 'polar'})

modes = [(0, 1), (0, 2), (0, 3),
         (1, 1), (1, 2), (2, 1)]

for idx, (m, n) in enumerate(modes):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    alpha_mn = jn_zeros(m, n)[-1]  # n-th zero

    r = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 200)
    R, Theta = np.meshgrid(r, theta)

    Z = jv(m, alpha_mn * R) * np.cos(m * Theta)

    c = ax.pcolormesh(Theta, R, Z, cmap='RdBu_r', shading='auto')
    ax.set_title(f'Mode ($m$={m}, $n$={n})\n'
                 f'$\\alpha_{{{m},{n}}}$ = {alpha_mn:.3f}',
                 fontsize=11, pad=12)
    ax.set_rticks([])

plt.suptitle('Circular Membrane Vibration Modes — $J_m(\\alpha_{m,n} r/a) \\cos(m\\theta)$',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('circular_membrane_modes.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Frequency ratios of normal modes ---
print("Circular membrane normal mode frequency ratios (f_mn / f_01):")
alpha_01 = jn_zeros(0, 1)[0]
for m in range(3):
    for n in range(1, 4):
        alpha_mn = jn_zeros(m, n)[-1]
        ratio = alpha_mn / alpha_01
        print(f"  (m={m}, n={n}): alpha = {alpha_mn:.4f}, "
              f"f/f_01 = {ratio:.4f}")
```

**Physical meaning**: The frequency ratios of a circular membrane vibration are inharmonic — this is why drums do not produce clear pitches like pianos.

### 3.5 Modified Bessel Functions

The **modified Bessel equation** appears when exponential behavior in the $z$-direction emerges in cylindrical coordinate problems:

$$x^2 y'' + x y' - (x^2 + \nu^2) y = 0$$

This is obtained from Bessel's equation by substituting $x \to ix$, and the solutions are **modified Bessel functions**:

- **Modified Bessel function of the first kind** $I_\nu(x) = i^{-\nu} J_\nu(ix)$: **exponentially increasing** as $x \to \infty$ ($I_\nu(x) \sim \frac{e^x}{\sqrt{2\pi x}}$)
- **Modified Bessel function of the second kind** $K_\nu(x)$: **exponentially decreasing** as $x \to \infty$ ($K_\nu(x) \sim \sqrt{\frac{\pi}{2x}} e^{-x}$)

Series representation:

$$I_\nu(x) = \sum_{k=0}^{\infty} \frac{1}{k! \, \Gamma(k+\nu+1)} \left(\frac{x}{2}\right)^{2k+\nu}$$

(Compared to Bessel $J_\nu$: no $(-1)^k$, so all terms are positive → monotonically increasing without oscillation)

**Physics applications**:
- Attenuated modes in cylindrical waveguides
- Steady-state heat conduction in circular cross-sections
- Yukawa potential $\propto K_0(mr)/r$
- Diffusion problems reducing to the modified Bessel equation

```python
import numpy as np
from scipy.special import iv, kv, jv
import matplotlib.pyplot as plt

x = np.linspace(0.01, 5, 300)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# I_nu(x): modified Bessel of the first kind
for nu in [0, 1, 2]:
    axes[0].plot(x, iv(nu, x), linewidth=2, label=f'$I_{{{nu}}}(x)$')
axes[0].set_ylim(0, 10)
axes[0].set_xlabel('x'); axes[0].set_ylabel('$I_\\nu(x)$')
axes[0].set_title('Modified Bessel Functions of the First Kind $I_\\nu(x)$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# K_nu(x): modified Bessel of the second kind
for nu in [0, 1, 2]:
    axes[1].plot(x, kv(nu, x), linewidth=2, label=f'$K_{{{nu}}}(x)$')
axes[1].set_ylim(0, 5)
axes[1].set_xlabel('x'); axes[1].set_ylabel('$K_\\nu(x)$')
axes[1].set_title('Modified Bessel Functions of the Second Kind $K_\\nu(x)$')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modified_bessel.png', dpi=150, bbox_inches='tight')
plt.show()

# J_nu vs I_nu comparison: oscillating vs monotonically increasing
fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, jv(0, x), 'b-', linewidth=2, label='$J_0(x)$ (oscillating)')
ax.plot(x, iv(0, x), 'r-', linewidth=2, label='$I_0(x)$ (monotonically increasing)')
ax.plot(x, kv(0, x), 'g-', linewidth=2, label='$K_0(x)$ (monotonically decreasing)')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_ylim(-0.5, 5)
ax.set_title('Bessel $J_0$ vs Modified Bessel $I_0$, $K_0$')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. Legendre Polynomials

### 4.1 Legendre's Equation

**Legendre's equation** appears when separating variables in Laplace's equation in spherical coordinates:

$$(1 - x^2) y'' - 2x y' + l(l+1) y = 0$$

where $x = \cos\theta$ and $l$ is a non-negative integer. The points $x = \pm 1$ are regular singular points.

When $l$ is a non-negative integer, one series solution terminates as a **polynomial**, which is the **Legendre polynomial** $P_l(x)$.

### 4.2 Rodrigues' Formula and Generating Function

**Rodrigues' formula**:

$$P_l(x) = \frac{1}{2^l l!} \frac{d^l}{dx^l} (x^2 - 1)^l$$

First few Legendre polynomials:

| $l$ | $P_l(x)$ |
|---|---|
| 0 | $1$ |
| 1 | $x$ |
| 2 | $\frac{1}{2}(3x^2 - 1)$ |
| 3 | $\frac{1}{2}(5x^3 - 3x)$ |
| 4 | $\frac{1}{8}(35x^4 - 30x^2 + 3)$ |

**Generating function**:

$$\frac{1}{\sqrt{1 - 2xt + t^2}} = \sum_{l=0}^{\infty} P_l(x) t^l, \quad |t| < 1$$

The physical meaning of this generating function: it appears when expanding $\frac{1}{|\mathbf{r} - \mathbf{r}'|}$, and is the core of **multipole expansion**.

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import legendre

# --- Legendre polynomials: Rodrigues' formula with SymPy ---
x = sp.Symbol('x')

def rodrigues(l):
    """Compute P_l(x) using Rodrigues' formula"""
    return sp.simplify(
        sp.diff((x**2 - 1)**l, x, l) / (2**l * sp.factorial(l))
    )

print("Legendre polynomials via Rodrigues' formula:")
for l in range(6):
    Pl = rodrigues(l)
    print(f"  P_{l}(x) = {Pl}")

# --- Visualization ---
x_vals = np.linspace(-1, 1, 500)

plt.figure(figsize=(10, 6))
for l in range(6):
    Pl = legendre(l)
    plt.plot(x_vals, Pl(x_vals), linewidth=1.5, label=f'$P_{{{l}}}(x)$')

plt.xlabel('$x$')
plt.ylabel('$P_l(x)$')
plt.title('Legendre Polynomials $P_l(x)$, $l = 0, 1, \\ldots, 5$')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('legendre_polynomials.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 Orthogonality and Series Expansion

Legendre polynomials are **orthogonal** on the interval $[-1, 1]$:

$$\int_{-1}^{1} P_m(x) P_n(x) \, dx = \frac{2}{2n+1} \delta_{mn}$$

Therefore, a function $f(x)$ defined on the interval $[-1, 1]$ can be expanded in a **Legendre series**:

$$f(x) = \sum_{l=0}^{\infty} c_l P_l(x), \quad c_l = \frac{2l+1}{2} \int_{-1}^{1} f(x) P_l(x) \, dx$$

```python
import numpy as np
from scipy.special import legendre
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- Legendre series expansion: f(x) = |x| ---
def legendre_coeff(l, func):
    """Compute Legendre series coefficient c_l"""
    Pl = legendre(l)
    integrand = lambda x: func(x) * Pl(x)
    integral, _ = quad(integrand, -1, 1)
    return (2*l + 1) / 2 * integral

f = lambda x: np.abs(x)
N_max = 20
coeffs = [legendre_coeff(l, f) for l in range(N_max)]

# |x| is even, so odd l coefficients are 0
print("Legendre series coefficients (f(x) = |x|):")
for l in range(8):
    print(f"  c_{l} = {coeffs[l]:.6f}", "(= 0 theoretically)" if l % 2 == 1 else "")

# Reconstruct series
x = np.linspace(-1, 1, 500)

plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(x), 'k-', linewidth=2.5, label='$f(x) = |x|$')

for N, color in [(3, 'red'), (7, 'blue'), (15, 'green')]:
    f_approx = sum(coeffs[l] * legendre(l)(x) for l in range(N))
    plt.plot(x, f_approx, color=color, linewidth=1.5,
             label=f'Legendre series ({N} terms)')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Legendre Series Expansion: $f(x) = |x|$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('legendre_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.4 Physics Applications (Multipole Expansion, Spherically Symmetric Problems)

**Multipole Expansion**: The potential at a far distance from a charge distribution is expanded in Legendre polynomials.

When a point charge $q$ is located at a distance $d$ from the origin, the potential at an observation point $r > d$:

$$\Phi(r, \theta) = \frac{q}{4\pi\epsilon_0} \frac{1}{|\mathbf{r} - \mathbf{d}|} = \frac{q}{4\pi\epsilon_0} \sum_{l=0}^{\infty} \frac{d^l}{r^{l+1}} P_l(\cos\theta)$$

- $l = 0$: **monopole** — total charge
- $l = 1$: **dipole** — $\propto 1/r^2$
- $l = 2$: **quadrupole** — $\propto 1/r^3$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# --- Multipole expansion visualization ---
# Point charge q located at distance d on the z-axis
q = 1.0
d = 0.5  # charge position (along z from origin)

# Polar coordinate grid (region r > d)
r = np.linspace(0.8, 3.0, 100)
theta = np.linspace(0, np.pi, 100)
R, Theta = np.meshgrid(r, theta)

# Exact potential
X = R * np.sin(Theta)
Z = R * np.cos(Theta)
dist = np.sqrt(X**2 + (Z - d)**2)
Phi_exact = q / dist

# Multipole approximations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Monopole (l=0)', 'Monopole+Dipole (l=0,1)', 'Multipole (l=0,...,4)']
L_max_list = [0, 1, 4]

for idx, L_max in enumerate(L_max_list):
    Phi_approx = np.zeros_like(R)
    for l in range(L_max + 1):
        Pl = legendre(l)
        Phi_approx += q * (d**l / R**(l + 1)) * Pl(np.cos(Theta))

    # Contour plot in r-theta plane (x-z cross section)
    ax = axes[idx]
    levels = np.linspace(0.2, 2.0, 15)
    c1 = ax.contour(X, Z, Phi_exact, levels=levels,
                    colors='gray', linewidths=0.5, linestyles='--')
    c2 = ax.contour(X, Z, Phi_approx, levels=levels,
                    colors='blue', linewidths=1.0)
    ax.plot(0, d, 'ro', markersize=8, label='Point charge')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(titles[idx])
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle('Multipole Expansion: Exact (gray dashed) vs Approximation (blue solid)', fontsize=13)
plt.tight_layout()
plt.savefig('multipole_expansion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.5 Associated Legendre Functions and Spherical Harmonics

#### Associated Legendre Functions

In 3D spherical coordinate problems involving both $\theta$- and $\phi$-dependence, the **associated Legendre equation** appears:

$$(1 - x^2) y'' - 2x y' + \left[l(l+1) - \frac{m^2}{1-x^2}\right] y = 0$$

where $x = \cos\theta$, $l = 0, 1, 2, \ldots$, $m = -l, \ldots, l$.

The solutions are **associated Legendre functions** $P_l^m(x)$, given by an extension of Rodrigues' formula:

$$P_l^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_l(x), \quad m \geq 0$$

For negative $m$:

$$P_l^{-m}(x) = (-1)^m \frac{(l-m)!}{(l+m)!} P_l^m(x)$$

First few associated Legendre functions:

| $P_l^m(\cos\theta)$ | Explicit Form |
|---|---|
| $P_0^0$ | $1$ |
| $P_1^0$ | $\cos\theta$ |
| $P_1^1$ | $-\sin\theta$ |
| $P_2^0$ | $\frac{1}{2}(3\cos^2\theta - 1)$ |
| $P_2^1$ | $-3\sin\theta\cos\theta$ |
| $P_2^2$ | $3\sin^2\theta$ |

**Orthogonality**:

$$\int_{-1}^{1} P_l^m(x) P_{l'}^m(x) dx = \frac{2}{2l+1} \frac{(l+m)!}{(l-m)!} \delta_{ll'}$$

```python
import numpy as np
from scipy.special import lpmv
import matplotlib.pyplot as plt

# Associated Legendre function visualization
theta = np.linspace(0, np.pi, 200)
x = np.cos(theta)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
funcs = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

for idx, (l, m) in enumerate(funcs):
    ax = axes[idx // 3, idx % 3]
    Plm = lpmv(m, l, x)
    ax.plot(theta * 180 / np.pi, Plm, 'b-', linewidth=2)
    ax.set_title(f'$P_{{{l}}}^{{{m}}}(\\cos\\theta)$')
    ax.set_xlabel('$\\theta$ (degrees)')
    ax.grid(True, alpha=0.3)

plt.suptitle('Associated Legendre Functions', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Orthogonality verification (same m, different l)
print("=== Orthogonality: ∫P_l^m P_l'^m dx (m=1) ===")
x_fine = np.linspace(-1, 1, 5000)
for l1 in range(1, 5):
    for l2 in range(l1, 5):
        inner = np.trapz(lpmv(1, l1, x_fine) * lpmv(1, l2, x_fine), x_fine)
        print(f"  <P_{l1}^1, P_{l2}^1> = {inner:.6f}")
```

#### Spherical Harmonics

Combining $P_l^m$ with $e^{im\phi}$ yields **spherical harmonics** $Y_l^m(\theta, \phi)$:

$$Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}} \, P_l^m(\cos\theta) \, e^{im\phi}$$

These are **eigenfunctions of the Laplacian** on the sphere:

$$\nabla^2_{S^2} Y_l^m = -l(l+1) Y_l^m$$

**Key Properties**:
- **Orthonormality**: $\int_0^{2\pi}\int_0^{\pi} Y_l^m(\theta,\phi)^* Y_{l'}^{m'}(\theta,\phi) \sin\theta \, d\theta \, d\phi = \delta_{ll'}\delta_{mm'}$
- **Completeness**: Any function on the sphere $f(\theta, \phi) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} c_{lm} Y_l^m(\theta, \phi)$
- **Symmetry relation**: $Y_l^{-m} = (-1)^m (Y_l^m)^*$
- **Addition theorem**: $P_l(\cos\gamma) = \frac{4\pi}{2l+1} \sum_{m=-l}^{l} Y_l^{m*}(\theta', \phi') Y_l^m(\theta, \phi)$

**Physics applications**:
- Angular part of hydrogen atom wavefunctions: $\psi_{nlm} = R_{nl}(r) Y_l^m(\theta, \phi)$
- Multipole expansion: potential, gravitational potential spherical expansion
- Geophysics: spherical harmonic analysis of geomagnetic field, gravitational field

```python
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# Spherical harmonics |Y_l^m|² visualization
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
Theta, Phi = np.meshgrid(theta, phi)

fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                         subplot_kw={'projection': '3d'})

harmonics = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

for idx, (l, m) in enumerate(harmonics):
    ax = axes[idx // 3, idx % 3]

    # scipy sph_harm: sph_harm(m, l, phi, theta)
    Y = sph_harm(m, l, Phi, Theta)
    r = np.abs(Y)

    X = r * np.sin(Theta) * np.cos(Phi)
    Y_coord = r * np.sin(Theta) * np.sin(Phi)
    Z = r * np.cos(Theta)

    # Color by phase
    fcolors = Y.real
    fmax = np.max(np.abs(fcolors))
    if fmax > 0:
        fcolors = fcolors / fmax

    ax.plot_surface(X, Y_coord, Z, facecolors=plt.cm.RdBu_r((fcolors + 1)/2),
                    rstride=2, cstride=2, alpha=0.8)
    ax.set_title(f'$Y_{{{l}}}^{{{m}}}(\\theta, \\phi)$')
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(-0.5, 0.5)

plt.suptitle('Spherical Harmonics (colored by real part)', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# Orthonormality verification
print("=== Spherical Harmonics Orthonormality ===")
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0]
for l1, m1 in [(0,0), (1,0), (1,1), (2,0)]:
    for l2, m2 in [(0,0), (1,0), (1,1), (2,0)]:
        Y1 = sph_harm(m1, l1, Phi, Theta)
        Y2 = sph_harm(m2, l2, Phi, Theta)
        integrand = np.conj(Y1) * Y2 * np.sin(Theta)
        inner = np.trapz(np.trapz(integrand, phi, axis=0), theta)
        if abs(inner) > 0.01 or (l1==l2 and m1==m2):
            print(f"  <Y_{l1}^{m1}|Y_{l2}^{m2}> = {inner.real:.4f}")
```

---

## 5. Hermite and Laguerre Functions

### 5.1 Hermite Functions and Quantum Harmonic Oscillator

**Hermite equation**:

$$y'' - 2xy' + 2ny = 0, \quad n = 0, 1, 2, \ldots$$

The polynomial solution is the **Hermite polynomial** $H_n(x)$:

$$H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2}$$

First few: $H_0 = 1$, $H_1 = 2x$, $H_2 = 4x^2 - 2$, $H_3 = 8x^3 - 12x$

The Schrödinger equation for the **quantum harmonic oscillator**:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + \frac{1}{2}m\omega^2 x^2 \psi = E\psi$$

Introducing the dimensionless variable $\xi = \sqrt{m\omega/\hbar} \, x$, the solutions are:

$$\psi_n(\xi) = \frac{1}{\sqrt{2^n n! \sqrt{\pi}}} H_n(\xi) e^{-\xi^2/2}, \quad E_n = \hbar\omega\left(n + \frac{1}{2}\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial, sqrt, pi

# --- Quantum harmonic oscillator wavefunctions ---
def psi_n(xi, n):
    """n-th harmonic oscillator wavefunction"""
    Hn = hermite(n)
    norm = 1.0 / sqrt(2**n * factorial(n) * sqrt(pi))
    return norm * Hn(xi) * np.exp(-xi**2 / 2)

xi = np.linspace(-6, 6, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Wavefunctions
for n in range(5):
    axes[0].plot(xi, psi_n(xi, n) + n, linewidth=1.5,
                 label=f'$\\psi_{{{n}}}$ ($E_{n} = {n+0.5}\\hbar\\omega$)')
    axes[0].axhline(y=n, color='gray', linewidth=0.3, linestyle='--')

axes[0].set_xlabel('$\\xi = \\sqrt{m\\omega/\\hbar} \\, x$')
axes[0].set_ylabel('$\\psi_n(\\xi)$ + offset')
axes[0].set_title('Quantum Harmonic Oscillator Wavefunctions')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, alpha=0.2)

# Probability densities
for n in range(5):
    prob = psi_n(xi, n)**2
    axes[1].fill_between(xi, n, prob + n, alpha=0.3)
    axes[1].plot(xi, prob + n, linewidth=1.5, label=f'$|\\psi_{{{n}}}|^2$')
    axes[1].axhline(y=n, color='gray', linewidth=0.3, linestyle='--')

# Classical probability density (n=4, for comparison)
n_class = 4
A = np.sqrt(2 * n_class + 1)
xi_class = np.linspace(-A + 0.01, A - 0.01, 300)
prob_class = 1.0 / (np.pi * np.sqrt(A**2 - xi_class**2))
axes[1].plot(xi_class, prob_class + n_class, 'k--', linewidth=1.5,
             label='Classical ($n=4$)')

axes[1].set_xlabel('$\\xi$')
axes[1].set_ylabel('$|\\psi_n|^2$ + offset')
axes[1].set_title('Probability Density (Quantum vs Classical Correspondence)')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('quantum_harmonic_oscillator.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 Laguerre Functions and Hydrogen Atom

**Associated Laguerre equation**:

$$x y'' + (k + 1 - x) y' + n y = 0$$

The solutions are **associated Laguerre polynomials** $L_n^k(x)$.

The radial wavefunction for the **hydrogen atom**:

$$R_{nl}(r) = N_{nl} \left(\frac{2r}{na_0}\right)^l e^{-r/(na_0)} L_{n-l-1}^{2l+1}\!\left(\frac{2r}{na_0}\right)$$

where $n$ is the principal quantum number, $l$ is the orbital quantum number, and $a_0$ is the Bohr radius. Energy levels:

$$E_n = -\frac{13.6 \text{ eV}}{n^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre, factorial
from math import sqrt

# --- Hydrogen atom radial wavefunctions ---
a0 = 1.0  # in units of Bohr radius

def R_nl(r, n, l):
    """Hydrogen atom radial wavefunction R_nl(r)"""
    rho = 2 * r / (n * a0)
    norm = sqrt((2 / (n * a0))**3 * factorial(n - l - 1)
                / (2 * n * factorial(n + l)))
    return norm * rho**l * np.exp(-rho / 2) * assoc_laguerre(rho, n - l - 1, 2*l + 1)

r = np.linspace(0, 25, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Radial wavefunction R_nl(r)
states = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
for n, l in states:
    axes[0].plot(r, R_nl(r, n, l), linewidth=1.5,
                 label=f'$R_{{{n}{l}}}$ ($n$={n}, $l$={l})')

axes[0].set_xlabel('$r / a_0$')
axes[0].set_ylabel('$R_{nl}(r)$')
axes[0].set_title('Hydrogen Atom Radial Wavefunctions')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linewidth=0.5)

# Radial probability density r^2 |R_nl|^2
for n, l in states:
    prob = r**2 * R_nl(r, n, l)**2
    axes[1].plot(r, prob, linewidth=1.5,
                 label=f'$r^2|R_{{{n}{l}}}|^2$')

axes[1].set_xlabel('$r / a_0$')
axes[1].set_ylabel('$r^2 |R_{nl}|^2$')
axes[1].set_title('Radial Probability Density')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hydrogen_radial.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. Gamma and Beta Functions

### 6.1 Definition and Properties of the Gamma Function

The **gamma function** extends the factorial function to real (and complex) numbers:

$$\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} \, dt, \quad \text{Re}(z) > 0$$

**Key Properties**:

| Property | Formula |
|---|---|
| Recurrence relation | $\Gamma(z+1) = z \Gamma(z)$ |
| Relationship with factorial | $\Gamma(n+1) = n!$ ($n$ is a non-negative integer) |
| Half-integer value | $\Gamma(1/2) = \sqrt{\pi}$ |
| Reflection formula | $\Gamma(z)\Gamma(1-z) = \frac{\pi}{\sin(\pi z)}$ |
| Duplication formula (Legendre) | $\Gamma(z)\Gamma(z+\frac{1}{2}) = \frac{\sqrt{\pi}}{2^{2z-1}}\Gamma(2z)$ |
| Stirling approximation | $\Gamma(z+1) \sim \sqrt{2\pi z} \left(\frac{z}{e}\right)^z$ ($z \to \infty$) |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# --- Gamma function visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gamma function on the real axis
x = np.linspace(-4.5, 5, 1000)
y = np.array([gamma(xi) if abs(xi - round(xi)) > 0.01 or xi > 0
              else np.nan for xi in x])

axes[0].plot(x, y, 'b-', linewidth=1.5)
# Values at positive integers (factorials)
for n in range(1, 6):
    axes[0].plot(n, gamma(n), 'ro', markersize=6)
    axes[0].annotate(f'{n-1}! = {int(gamma(n))}',
                     xy=(n, gamma(n)), xytext=(n+0.2, gamma(n)+2),
                     fontsize=9)

axes[0].set_xlim(-4.5, 5.5)
axes[0].set_ylim(-10, 25)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axvline(x=0, color='k', linewidth=0.5)
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$\\Gamma(x)$')
axes[0].set_title('Gamma Function $\\Gamma(x)$')
axes[0].grid(True, alpha=0.3)

# Stirling approximation comparison
n_vals = np.arange(1, 15)
exact = np.array([gamma(n + 1) for n in n_vals])
stirling = np.sqrt(2 * np.pi * n_vals) * (n_vals / np.e)**n_vals
rel_error = np.abs(stirling - exact) / exact

axes[1].semilogy(n_vals, rel_error, 'bo-', linewidth=1.5, markersize=6)
axes[1].set_xlabel('$n$')
axes[1].set_ylabel('Relative error')
axes[1].set_title('Accuracy of Stirling Approximation $n! \\approx \\sqrt{2\\pi n}(n/e)^n$')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gamma_function.png', dpi=150, bbox_inches='tight')
plt.show()

# Reflection formula verification
print("Gamma function reflection formula verification: Gamma(z) * Gamma(1-z) = pi/sin(pi*z)")
for z in [0.25, 0.5, 0.75, 1.3]:
    lhs = gamma(z) * gamma(1 - z)
    rhs = np.pi / np.sin(np.pi * z)
    print(f"  z = {z}: LHS = {lhs:.8f}, RHS = {rhs:.8f}, "
          f"diff = {abs(lhs-rhs):.2e}")
```

### 6.2 Beta Function

The **beta function** is defined as:

$$B(p, q) = \int_0^1 t^{p-1}(1-t)^{q-1} \, dt, \quad p, q > 0$$

Relationship with the gamma function:

$$\boxed{B(p, q) = \frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}}$$

This formula is very useful, allowing complex integrals to be computed directly using gamma function values.

**Application example**: $\int_0^{\pi/2} \sin^m\theta \cos^n\theta \, d\theta = \frac{1}{2} B\!\left(\frac{m+1}{2}, \frac{n+1}{2}\right)$

```python
import numpy as np
from scipy.special import gamma, beta

# --- Integral computation using the beta function ---
print("Integral computation using the beta function:")
print("=" * 55)

# Example 1: int_0^1 x^3 (1-x)^4 dx = B(4, 5) = 3!*4!/8!
result = beta(4, 5)
exact = 6 * 24 / 40320
print(f"\n∫₀¹ x³(1-x)⁴ dx = B(4,5) = {result:.8f}")
print(f"  = 3!·4!/8! = {exact:.8f}")

# Example 2: int_0^{pi/2} sin^5(theta) d(theta)
# = (1/2) B(3, 1/2) = (1/2) * Gamma(3)*Gamma(1/2) / Gamma(7/2)
m = 5
integral = 0.5 * beta((m + 1)/2, 0.5)
print(f"\n∫₀^(π/2) sin⁵θ dθ = (1/2)B(3, 1/2) = {integral:.8f}")
print(f"  theoretical value = 8/15 = {8/15:.8f}")

# Example 3: Generalized Gaussian integral
# int_0^inf x^(2n) e^(-x^2) dx = Gamma(n + 1/2) / 2
print(f"\nGeneralized Gaussian integral:")
for n in range(5):
    val = gamma(n + 0.5) / 2
    print(f"  ∫₀^∞ x^{2*n} e^(-x²) dx = Γ({n}+1/2)/2 = {val:.6f}")

# Stirling formula: verify Gamma(1/2) = sqrt(pi)
print(f"\nΓ(1/2) = {gamma(0.5):.10f}")
print(f"√π     = {np.sqrt(np.pi):.10f}")
```

---

## Practice Problems

### Basic Problems

**Problem 1**: Classify the ordinary points and singular points of the following ODEs.
- (a) $(1-x^2)y'' - 2xy' + 6y = 0$ (Legendre's equation, $l=2$)
- (b) $xy'' + (1-x)y' + ny = 0$ (Laguerre's equation)
- (c) $y'' + \frac{1}{x}y' + \left(1 - \frac{4}{x^2}\right)y = 0$ (Bessel's equation, $\nu=2$)

<details>
<summary>Solution Hint</summary>

(a) Standard form: $P(x) = \frac{-2x}{1-x^2}$, $Q(x) = \frac{6}{1-x^2}$

- $x = 0$: ordinary point ($P$, $Q$ analytic)
- $x = \pm 1$: singular points ($P$, $Q$ diverge)
  - $(x-1)P(x) = \frac{-2x}{-(1+x)} \to 1$ (analytic), $(x-1)^2 Q(x) \to 0$ (analytic) → regular singular point

(b) Standard form: $P(x) = (1-x)/x$, $Q(x) = n/x$

- $x = 0$: $xP(x) = 1-x$ (analytic), $x^2 Q(x) = nx$ (analytic) → regular singular point

(c) $x = 0$: $xP(x) = 1$ (analytic), $x^2 Q(x) = x^2 - 4$ (analytic) → regular singular point

</details>

**Problem 2**: For the Airy equation $y'' - xy = 0$, find the first 6 non-zero terms in the power series solution (cases $a_0 = 1$, $a_1 = 0$ and $a_0 = 0$, $a_1 = 1$).

<details>
<summary>Solution Hint</summary>

Recurrence relation: $a_{m+2} = \frac{a_{m-1}}{(m+2)(m+1)}$, $a_2 = 0$

**Case 1** ($a_0 = 1$, $a_1 = 0$): $a_3 = \frac{1}{6}$, $a_6 = \frac{1}{180}$, $a_9 = \frac{1}{12960}$, ...

$$y_1 = 1 + \frac{x^3}{6} + \frac{x^6}{180} + \frac{x^9}{12960} + \cdots$$

**Case 2** ($a_0 = 0$, $a_1 = 1$): $a_4 = \frac{1}{12}$, $a_7 = \frac{1}{504}$, $a_{10} = \frac{1}{45360}$, ...

$$y_2 = x + \frac{x^4}{12} + \frac{x^7}{504} + \frac{x^{10}}{45360} + \cdots$$

```python
import sympy as sp
x = sp.Symbol('x')
y = sp.Function('y')
sol = sp.dsolve(y(x).diff(x, 2) - x*y(x), y(x), hint='power_series', n=12)
print(sol)
```

</details>

### Application Problems

**Problem 3** (Bessel functions): For a circular membrane of radius $a = 1$, find the first 5 eigenfrequency ratios $\omega_n / \omega_1$ for axisymmetric ($m = 0$) modes.

<details>
<summary>Solution Hint</summary>

Frequencies for axisymmetric modes: $\omega_n = c \, \alpha_{0,n} / a$

where $\alpha_{0,n}$ is the $n$-th positive root of $J_0(x) = 0$.

```python
from scipy.special import jn_zeros

zeros_J0 = jn_zeros(0, 5)
print("Zeros of J_0(x):", zeros_J0)
print("Frequency ratios:")
for n in range(5):
    print(f"  omega_{n+1}/omega_1 = {zeros_J0[n]/zeros_J0[0]:.4f}")
```

</details>

**Problem 4** (Legendre polynomials): Using Legendre polynomials, expand the potential of an electric dipole with charge $+q$ at $(0, 0, d)$ and $-q$ at $(0, 0, -d)$ in the region $r \gg d$, and show that the dipole term ($l = 1$) dominates.

<details>
<summary>Solution Hint</summary>

$$\Phi = \frac{q}{4\pi\epsilon_0}\left(\frac{1}{|\mathbf{r} - d\hat{z}|} - \frac{1}{|\mathbf{r} + d\hat{z}|}\right)$$

Expand each in a Legendre series:

$$\Phi = \frac{q}{4\pi\epsilon_0} \sum_{l=0}^{\infty} \frac{d^l}{r^{l+1}} P_l(\cos\theta) [1 - (-1)^l]$$

When $l$ is even, $[1 - 1] = 0$, so only odd $l$ remain. The $l = 1$ (dipole) term:

$$\Phi_{\text{dipole}} = \frac{q \cdot 2d}{4\pi\epsilon_0} \frac{\cos\theta}{r^2} = \frac{p \cos\theta}{4\pi\epsilon_0 r^2}$$

where $p = 2qd$ is the dipole moment.

</details>

**Problem 5** (Gamma/Beta functions): Calculate the following integrals using gamma and beta functions.

- (a) $\int_0^{\infty} x^4 e^{-x^2} dx$
- (b) $\int_0^1 x^2 \sqrt{1 - x^2} \, dx$
- (c) For the volume of an $n$-dimensional hypersphere $V_n(R) = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} R^n$, find $V_1, V_2, V_3, V_4$.

<details>
<summary>Solution Hint</summary>

(a) Substitute $t = x^2$: $\frac{1}{2}\int_0^{\infty} t^{3/2} e^{-t} dt = \frac{1}{2}\Gamma(5/2) = \frac{1}{2} \cdot \frac{3}{2} \cdot \frac{1}{2} \cdot \sqrt{\pi} = \frac{3\sqrt{\pi}}{8}$

(b) Substitute $x = \sin\theta$: $\int_0^{\pi/2} \sin^2\theta \cos^2\theta \, d\theta = \frac{1}{2}B(3/2, 3/2) = \frac{\pi}{16}$

(c)
```python
from scipy.special import gamma
import numpy as np

for n in range(1, 5):
    V = np.pi**(n/2) / gamma(n/2 + 1)
    print(f"V_{n}(R=1) = pi^({n}/2) / Gamma({n}/2 + 1) = {V:.6f}")
# V_1 = 2 (line segment), V_2 = pi (circle), V_3 = 4pi/3 (sphere), V_4 = pi^2/2
```

</details>

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapters 11-12. Wiley.
   - Main reference for this lesson
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapters 7-11. Academic Press.
   - Graduate-level theory of special functions
3. **Watson, G. N.** (1995). *A Treatise on the Theory of Bessel Functions*, 2nd ed. Cambridge University Press.
   - Classical reference on Bessel functions
4. **Griffiths, D. J.** (2018). *Introduction to Quantum Mechanics*, 3rd ed. Cambridge University Press.
   - Quantum mechanics applications of Hermite/Laguerre functions

### Online Resources
1. **NIST Digital Library of Mathematical Functions**: https://dlmf.nist.gov/
   - Standard reference for formulas, properties, and numerical data of special functions
2. **MIT OCW 18.03**: Differential Equations — Power Series Solutions
3. **3Blue1Brown**: Intuitive understanding of special functions

### Core Library Documentation
1. **SciPy `scipy.special`**: https://docs.scipy.org/doc/scipy/reference/special.html
   - `jv`, `yv` (Bessel), `legendre` (Legendre), `hermite` (Hermite), `gamma`, `beta`, etc.
2. **SymPy Special Functions**: https://docs.sympy.org/latest/modules/functions/special.html

---

## Next Lesson

In [12. Sturm-Liouville Theory](12_Sturm_Liouville_Theory.md), we will show that Bessel functions, Legendre polynomials, Hermite functions, and others covered in this lesson are all special cases of the **Sturm-Liouville eigenvalue problem**, and cover the mathematical foundation of **completeness of orthogonal function systems** and **generalized Fourier series**.

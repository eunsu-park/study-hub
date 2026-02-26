# 12. Sturm-Liouville Theory

## Learning Objectives

- Understand the standard form of **Sturm-Liouville problems** and transform general second-order ODEs into self-adjoint form
- Prove and utilize key results of S-L theory: **reality** of eigenvalues, **orthogonality** of eigenfunctions, and **completeness**
- Expand arbitrary functions in **generalized Fourier series** using eigenfunctions
- Identify which S-L problem has trigonometric functions, Bessel functions, and Legendre polynomials as its eigenfunctions
- Perform **weighted inner product** and **Gram-Schmidt orthogonalization**
- Understand the role of S-L theory in **physics applications** such as heat equations, vibration problems, and quantum mechanics

---

## 1. Sturm-Liouville Problems

### 1.1 Self-Adjoint Form

The **Sturm-Liouville equation** is the standard form of a second-order ODE:

$$\frac{d}{dx}\left[p(x)\frac{dy}{dx}\right] + q(x)y + \lambda w(x) y = 0$$

where:
- $p(x) > 0$: coefficient function (can vanish at endpoints)
- $q(x)$: potential term
- $w(x) > 0$: **weight function** or **density function**
- $\lambda$: **eigenvalue** parameter

In operator notation:

$$\mathcal{L}[y] = -\lambda w(x) y, \quad \mathcal{L} = \frac{d}{dx}\left[p(x)\frac{d}{dx}\right] + q(x)$$

$\mathcal{L}$ is called the **Sturm-Liouville operator**, which has the **self-adjoint** property under appropriate boundary conditions.

**Transforming a general 2nd-order ODE to S-L form**: For $a(x)y'' + b(x)y' + c(x)y + \lambda d(x)y = 0$, multiply by **integrating factor** $\mu(x) = \frac{1}{a}\exp\left(\int \frac{b}{a}dx\right)$ to get $p = \mu a$, $q = \mu c$, $w = \mu d$.

```python
import sympy as sp

x = sp.Symbol('x')
# Example: Convert Hermite equation y'' - 2xy' + 2ny = 0 to S-L form
a_x, b_x = sp.Integer(1), -2 * x

integrating_factor = sp.exp(sp.integrate(b_x / a_x, x))
p_x = integrating_factor * a_x
print(f"p(x) = {p_x}")   # exp(-x^2)
print(f"w(x) = {integrating_factor}")  # exp(-x^2)
# S-L form: d/dx[exp(-x^2) y'] + 2n exp(-x^2) y = 0
```

### 1.2 Boundary Conditions

S-L problems are defined on an interval $[a, b]$ with accompanying **boundary conditions**.

**Regular S-L problem**: $p, q, w$ continuous, $p > 0$, $w > 0$ on $[a,b]$, separated boundary conditions:

$$\alpha_1 y(a) + \alpha_2 y'(a) = 0, \quad \beta_1 y(b) + \beta_2 y'(b) = 0$$

**Singular S-L problem**: When $p(x)$ vanishes at endpoints or the interval is infinite. **Boundedness** or **square integrability** of solutions replaces boundary conditions.

| Type | Condition | Physical Example |
|---|---|---|
| Dirichlet | $y(a) = 0, \; y(b) = 0$ | String fixed at both ends |
| Neumann | $y'(a) = 0, \; y'(b) = 0$ | String free at both ends |
| Robin | $\alpha y + \beta y' = 0$ | Convective heat transfer |
| Periodic | $y(a) = y(b), \; y'(a) = y'(b)$ | Circular boundary |

### 1.3 Eigenvalues and Eigenfunctions

Solving the S-L equation with boundary conditions, nontrivial solutions exist only for specific $\lambda$ values:

$$\mathcal{L}[y_n] = -\lambda_n w(x) y_n, \quad n = 0, 1, 2, \ldots$$

**Example**: $y'' + \lambda y = 0$, $y(0) = 0$, $y(L) = 0$ ($p=1, q=0, w=1$)

Eigenvalues: $\lambda_n = (n\pi/L)^2$, Eigenfunctions: $y_n(x) = \sin(n\pi x/L)$, $n = 1, 2, 3, \ldots$

```python
import numpy as np
import matplotlib.pyplot as plt

L = np.pi
x = np.linspace(0, L, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for n, ax in zip(range(1, 5), axes.flat):
    y_n = np.sin(n * x)
    ax.plot(x, y_n, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_title(f'$y_{n}(x) = \\sin({n}x)$, $\\lambda_{n} = {n**2}$')
    ax.grid(True, alpha=0.3)

plt.suptitle('Eigenfunctions with Dirichlet Boundary Conditions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 2. Sturm-Liouville Theorems

### 2.1 Reality of Eigenvalues and Orthogonality of Eigenfunctions

**Theorem 1 (Reality)**: All eigenvalues of a regular S-L problem are **real**.

**Proof outline**: Multiply the equation for $y$ by $\bar{y}$ and the equation for $\bar{y}$ by $y$, then subtract:

$$(\lambda - \bar{\lambda})\int_a^b w|y|^2 dx = [p(\bar{y}y' - y\bar{y}')]_a^b = 0$$

Since $w > 0$, $|y|^2 > 0$, we have $\lambda = \bar{\lambda}$. $\blacksquare$

**Theorem 2 (Orthogonality)**: If $\lambda_m \neq \lambda_n$, eigenfunctions are orthogonal with respect to the weight function:

$$\int_a^b w(x) y_m(x) y_n(x) \, dx = 0$$

**Proof**: Similarly, $(\lambda_m - \lambda_n)\int_a^b w y_m y_n dx = 0$, and since $\lambda_m \neq \lambda_n$, the integral = 0. $\blacksquare$

```python
import numpy as np

L = np.pi
x = np.linspace(0, L, 10000)

print("=== Orthogonality verification: int_0^pi sin(mx) sin(nx) dx ===")
for m in range(1, 5):
    for n in range(1, 5):
        inner = np.trapz(np.sin(m*x) * np.sin(n*x), x)
        status = "= π/2" if m == n else "= 0"
        print(f"  <y_{m}, y_{n}> = {inner:.6f}  ({status})")
```

### 2.2 Completeness of Eigenfunctions

**Theorem 3**: The eigenfunctions $\{y_n\}$ of a regular S-L problem are **complete** in $L^2_w[a,b]$. For any piecewise smooth function $f(x)$:

$$f(x) = \sum_{n=1}^{\infty} c_n y_n(x), \quad \lim_{N\to\infty} \int_a^b w \left| f - \sum_{n=1}^{N} c_n y_n \right|^2 dx = 0$$

**Parseval's identity**: $\int_a^b w |f|^2 dx = \sum_{n=1}^{\infty} |c_n|^2 \|y_n\|^2$

### 2.3 Generalized Fourier Series

Expanding an arbitrary function in eigenfunctions $\{y_n\}$ is called a **generalized Fourier series**:

$$f(x) = \sum_{n=1}^{\infty} c_n y_n(x), \quad c_n = \frac{\int_a^b w(x) f(x) y_n(x) dx}{\int_a^b w(x) y_n^2(x) dx}$$

This is the function space generalization of vector projection $c_n = \frac{\mathbf{f} \cdot \mathbf{e}_n}{\mathbf{e}_n \cdot \mathbf{e}_n}$.

```python
import numpy as np
import matplotlib.pyplot as plt

L = np.pi
x = np.linspace(0, L, 1000)
f = lambda t: t  # f(x) = x

def compute_cn(n):
    return 2.0 / L * np.trapz(f(x) * np.sin(n * x), x)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for N, ax in zip([1, 3, 10, 50], axes.flat):
    approx = sum(compute_cn(n) * np.sin(n*x) for n in range(1, N+1))
    ax.plot(x, f(x), 'k--', linewidth=1.5, label='$f(x) = x$')
    ax.plot(x, approx, 'b-', linewidth=2, label=f'$N = {N}$')
    ax.set_title(f'Generalized Fourier Series ($N = {N}$)')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Eigenfunction Expansion of $f(x) = x$', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 2.4 Rayleigh Quotient and Variational Principle

S-L eigenvalues can be characterized **variationally**. The **Rayleigh quotient** for a function $y$ satisfying boundary conditions:

$$R[y] = \frac{-[p y y']_a^b + \int_a^b \left[p(y')^2 - q y^2\right] dx}{\int_a^b w y^2 dx}$$

For Dirichlet boundary conditions $y(a) = y(b) = 0$, the boundary term vanishes:

$$R[y] = \frac{\int_a^b \left[p(y')^2 - q y^2\right] dx}{\int_a^b w y^2 dx}$$

**Key properties**:
- **Minimum principle**: $R[y] \geq \lambda_1$ (smallest eigenvalue). Equality holds when $y = y_1$
- **Exactness**: $R[y_n] = \lambda_n$ (exact eigenvalue at eigenfunctions)
- **Upper bound**: Any trial function gives an upper bound for $\lambda_1$

This property is the foundation of the **Ritz method**: adjusting parameters of a trial function to minimize $R$ yields a good approximation of the eigenvalue.

```python
import numpy as np

L = np.pi
x = np.linspace(0, L, 1000)

# y'' + λy = 0, y(0) = y(π) = 0  →  p=1, q=0, w=1
# Exact λ₁ = 1

# Trial function 1: φ₁(x) = x(π - x)
phi1 = x * (L - x)
phi1_prime = L - 2*x
R1 = np.trapz(phi1_prime**2, x) / np.trapz(phi1**2, x)

# Trial function 2: φ₂(x) = sin(πx/L) (exact eigenfunction)
phi2 = np.sin(np.pi * x / L)
phi2_prime = (np.pi / L) * np.cos(np.pi * x / L)
R2 = np.trapz(phi2_prime**2, x) / np.trapz(phi2**2, x)

print(f"Trial function 1: x(π-x)")
print(f"  R₁ = {R1:.6f}  (exact value 1.0, error {abs(R1-1)*100:.2f}%)")
print(f"Trial function 2: sin(πx/π) (exact)")
print(f"  R₂ = {R2:.6f}  (exact value 1.0)")
print(f"\nR ≥ λ₁ confirmed: {R1:.6f} ≥ {1.0} ✓")
```

---

## 3. Major Examples

### 3.1 Trigonometric Functions (Fourier Series Reinterpreted)

A general **Fourier series** is the S-L problem $y'' + \lambda y = 0$ with periodic boundary conditions:

| Eigenvalue | Eigenfunction | Normalization Constant |
|---|---|---|
| $\lambda_0 = 0$ | $y_0 = 1$ | $\|y_0\|^2 = 2L$ |
| $\lambda_n = (n\pi/L)^2$ | $\cos(n\pi x/L), \; \sin(n\pi x/L)$ | $\|y_n\|^2 = L$ |

Thus, the Fourier series on $[-L, L]$, $f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}[a_n \cos\frac{n\pi x}{L} + b_n \sin\frac{n\pi x}{L}]$, is exactly a generalized Fourier series in S-L eigenfunctions.

### 3.2 Bessel Functions in S-L Form

The **Bessel equation** $x^2 y'' + xy' + (x^2 - \nu^2) y = 0$ in S-L form:

$$\frac{d}{dx}\left[x\frac{dy}{dx}\right] - \frac{\nu^2}{x}y + \lambda x \, y = 0$$

$p(x) = x$, $q(x) = -\nu^2/x$, $w(x) = x$. Since $p(0) = 0$, this is a **singular S-L problem**.

Boundary conditions: $|y(0)| < \infty$, $y(a) = 0$. Eigenvalues $\lambda_{n\nu} = (\alpha_{n\nu}/a)^2$ ($\alpha_{n\nu}$ are zeros of $J_\nu$).

**Orthogonality**: $\int_0^a x J_\nu(\frac{\alpha_{m\nu}}{a}x) J_\nu(\frac{\alpha_{n\nu}}{a}x) dx = 0$ ($m \neq n$)

```python
from scipy.special import jn_zeros, jv
import numpy as np
import matplotlib.pyplot as plt

nu, a = 0, 1.0
zeros = jn_zeros(nu, 5)
x = np.linspace(0, a, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for n in range(5):
    ax1.plot(x, jv(nu, zeros[n]*x), linewidth=2,
             label=f'$J_0(\\alpha_{{{n+1}}} x)$, $\\alpha={zeros[n]:.2f}$')
ax1.axhline(0, color='k', linewidth=0.5)
ax1.set_title('Bessel Function Eigenfunctions'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Orthogonality matrix
ortho = np.array([[np.trapz(x * jv(nu, zeros[m]*x) * jv(nu, zeros[n]*x), x)
                    for n in range(5)] for m in range(5)])
ax2.imshow(ortho, cmap='RdBu_r', vmin=-0.1, vmax=0.3)
ax2.set_title('Orthogonality Matrix (weight $w=x$)'); plt.colorbar(ax2.images[0], ax=ax2)
plt.tight_layout(); plt.show()
```

### 3.3 Legendre Polynomials in S-L Form

The **Legendre equation** $(1-x^2)y'' - 2xy' + l(l+1)y = 0$ in S-L form:

$$\frac{d}{dx}\left[(1-x^2)\frac{dy}{dx}\right] + l(l+1)y = 0$$

$p(x) = 1-x^2$, $q = 0$, $w = 1$, $\lambda = l(l+1)$. Since $p(\pm 1) = 0$, this is a singular S-L problem.

Requiring boundedness, solutions exist only when $l = 0, 1, 2, \ldots$: the **Legendre polynomials** $P_l(x)$.

**Orthogonality**: $\int_{-1}^{1} P_m(x) P_n(x) dx = \frac{2}{2n+1} \delta_{mn}$

```python
from scipy.special import legendre
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for l in range(6):
    ax1.plot(x, legendre(l)(x), linewidth=2, label=f'$P_{l}(x)$')
ax1.set_title('Legendre Polynomials'); ax1.legend(); ax1.grid(True, alpha=0.3)

N = 6
ortho = np.array([[np.trapz(legendre(m)(x)*legendre(n)(x), x)
                    for n in range(N)] for m in range(N)])
ax2.imshow(ortho, cmap='RdBu_r', vmin=-0.2, vmax=1.0)
ax2.set_title('Orthogonality Matrix $\\langle P_m, P_n \\rangle$')
plt.tight_layout(); plt.show()
```

### 3.4 Singular Sturm-Liouville Problems in Detail

**Singular S-L problems** occur when regular conditions are violated, and they appear more frequently in physics. Three causes:

1. **$p(x) = 0$ at endpoints**: Legendre, Bessel, etc.
2. **Infinite interval**: Hermite $(-\infty, \infty)$, Laguerre $[0, \infty)$
3. **Divergence of $q(x)$ or $w(x)$**: Chebyshev's $w = (1-x^2)^{-1/2}$

Boundary conditions are replaced by **boundedness** or **square integrability** $\int_a^b w |y|^2 dx < \infty$ of solutions.

| Equation | Interval | Singular Cause | Boundary Condition |
|---|---|---|---|
| Bessel | $[0, a]$ | $p(0) = 0$ | $|y(0)| < \infty$, $y(a) = 0$ |
| Legendre | $[-1, 1]$ | $p(\pm 1) = 0$ | $|y(\pm 1)| < \infty$ |
| Hermite | $(-\infty, \infty)$ | Infinite interval | $y \to 0$ as $|x| \to \infty$ |
| Laguerre | $[0, \infty)$ | Semi-infinite | $|y(0)| < \infty$, $y \to 0$ as $x \to \infty$ |
| Chebyshev | $[-1, 1]$ | $w(\pm 1) = \infty$ | $|y(\pm 1)| < \infty$ |

**Key result (Singular S-L theorem)**: Under appropriate boundary conditions, singular S-L problems also satisfy all key theorems of the regular case (real eigenvalues, orthogonality, completeness).

> **Key to the proof**: The boundary term $[p(\bar{y}_m y_n' - y_n \bar{y}_m')]_a^b$ automatically vanishes due to singular boundary conditions. For example, in Legendre's case, $p(\pm 1) = 1 - (\pm 1)^2 = 0$, so the boundary term vanishes.

```python
import numpy as np
from scipy.special import eval_hermite
import matplotlib.pyplot as plt

# Hermite functions: infinite-interval singular S-L
# d/dx[e^{-x²} y'] + 2n e^{-x²} y = 0
x = np.linspace(-5, 5, 500)
w = np.exp(-x**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for n in range(5):
    Hn = eval_hermite(n, x)
    ax1.plot(x, Hn * np.sqrt(w), linewidth=1.5,
             label=f'$H_{n}(x) e^{{-x^2/2}}$')

ax1.set_title('Hermite Functions (weighted eigenfunctions)')
ax1.set_xlabel('x'); ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Square integrability: ∫w|H_n|² dx < ∞
for n in range(4):
    Hn = eval_hermite(n, x)
    integrand = w * Hn**2
    ax2.fill_between(x, 0, integrand, alpha=0.3, label=f'$w H_{n}^2$')
    area = np.trapz(integrand, x)
    print(f"||H_{n}||²_w = {area:.4f}  (theory: {np.sqrt(np.pi) * 2**n * np.math.factorial(n):.4f})")

ax2.set_title('$w(x)|H_n(x)|^2$ (square integrable)')
ax2.set_xlabel('x'); ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. Weight Functions and Inner Products

### 4.1 Weighted Inner Product

**Weighted inner product**: $\langle f, g \rangle_w = \int_a^b w(x) f(x) g(x) dx$, **Norm**: $\|f\|_w = \sqrt{\langle f, f \rangle_w}$

| S-L Problem | Interval | Weight Function $w(x)$ | Eigenfunctions |
|---|---|---|---|
| Trigonometric (Fourier) | $[-L, L]$ | $1$ | $\cos, \sin$ |
| Legendre | $[-1, 1]$ | $1$ | $P_l(x)$ |
| Chebyshev | $[-1, 1]$ | $(1-x^2)^{-1/2}$ | $T_n(x)$ |
| Hermite | $(-\infty, \infty)$ | $e^{-x^2}$ | $H_n(x)$ |
| Laguerre | $[0, \infty)$ | $e^{-x}$ | $L_n(x)$ |
| Bessel | $[0, a]$ | $x$ | $J_\nu(\alpha_{n\nu}x/a)$ |

```python
import numpy as np
from scipy.special import eval_hermite
import math

# Weighted orthogonality verification of Hermite polynomials
x = np.linspace(-5, 5, 5000)
w = np.exp(-x**2)

print("=== Hermite Polynomial Weighted Orthogonality ===")
for m in range(5):
    for n in range(m, 5):
        inner = np.trapz(w * eval_hermite(m, x) * eval_hermite(n, x), x)
        if m == n:
            theory = math.sqrt(math.pi) * 2**n * math.factorial(n)
            print(f"  <H_{m}, H_{n}>_w = {inner:.4f}  (theory: {theory:.4f})")
        elif abs(inner) > 0.01:
            print(f"  <H_{m}, H_{n}>_w = {inner:.4f}  (should be ~0)")
```

### 4.2 Gram-Schmidt Orthogonalization

Construct weighted orthogonal functions $\{\phi_n\}$ from linearly independent functions $\{f_0, f_1, \ldots\}$:

$$\phi_0 = f_0, \quad \phi_n = f_n - \sum_{k=0}^{n-1} \frac{\langle f_n, \phi_k \rangle_w}{\langle \phi_k, \phi_k \rangle_w} \phi_k$$

**Example**: Apply $\{1, x, x^2, \ldots\}$ with $w=1$, interval $[-1,1]$ $\rightarrow$ **Legendre polynomials**.

```python
import numpy as np
import matplotlib.pyplot as plt

def gram_schmidt(basis_funcs, x, w):
    ortho = []
    for f_n in basis_funcs:
        phi = f_n(x).copy()
        for phi_k in ortho:
            coeff = np.trapz(w * f_n(x) * phi_k, x) / np.trapz(w * phi_k**2, x)
            phi -= coeff * phi_k
        ortho.append(phi)
    return ortho

x = np.linspace(-1, 1, 2000)
monomials = [lambda x, k=k: x**k for k in range(5)]
ortho_polys = gram_schmidt(monomials, x, np.ones_like(x))

# Orthogonality verification
N = len(ortho_polys)
mat = np.array([[np.trapz(ortho_polys[i]*ortho_polys[j], x)
                 for j in range(N)] for i in range(N)])

plt.figure(figsize=(6, 5))
plt.imshow(np.abs(mat), cmap='viridis', norm=plt.matplotlib.colors.LogNorm(1e-15, 2))
plt.title('Gram-Schmidt Orthogonality Matrix $|\\langle\\phi_m, \\phi_n\\rangle|$')
plt.colorbar(); plt.show()

```

### 4.3 Sturm Comparison Theorem

The **Sturm comparison theorem** is a tool for comparing the **oscillatory properties** of solutions under different potentials.

**Theorem**: For two equations with the same $p(x)$:

$$[p(x)u']' + q_1(x) u = 0, \quad [p(x)v']' + q_2(x) v = 0$$

where $q_1(x) \geq q_2(x)$ ($q_1 \not\equiv q_2$), then between two consecutive zeros of $v$, there exists at least one zero of $u$.

**Intuition**: Stronger "restoring force" ($q$) leads to faster oscillation of the solution.

**Physical applications**:
- **Quantum mechanics**: Higher energy $E$ state → larger effective $q(x)$ → increased wavefunction nodes
- **General result**: The $n$-th eigenfunction has exactly $n-1$ zeros (oscillation theorem)

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# y'' + q(x) y = 0 comparison of two cases
def ode(t, Y, q_func):
    y, yp = Y
    return [yp, -q_func(t) * y]

x_span = (0, 4*np.pi)
x_eval = np.linspace(*x_span, 2000)
y0 = [0, 1]

# q₁ = 4 (fast oscillation), q₂ = 1 (slow oscillation)
sol1 = solve_ivp(ode, x_span, y0, args=(lambda x: 4.0,),
                 t_eval=x_eval, rtol=1e-10)
sol2 = solve_ivp(ode, x_span, y0, args=(lambda x: 1.0,),
                 t_eval=x_eval, rtol=1e-10)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sol1.t, sol1.y[0], 'b-', linewidth=1.5, label='$q_1 = 4$: $\\sin(2x)$')
ax.plot(sol2.t, sol2.y[0], 'r-', linewidth=1.5, label='$q_2 = 1$: $\\sin(x)$')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('x'); ax.set_ylabel('y(x)')
ax.set_title('Sturm Comparison Theorem: larger $q$ leads to faster oscillation')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Compare number of zeros (interval 0 to 4π)
zeros1 = np.where(np.diff(np.sign(sol1.y[0])))[0]
zeros2 = np.where(np.diff(np.sign(sol2.y[0])))[0]
print(f"Number of zeros for q₁=4 solution: {len(zeros1)}  (theory: {int(4*np.pi/(np.pi/2))})")
print(f"Number of zeros for q₂=1 solution: {len(zeros2)}  (theory: {int(4*np.pi/np.pi)})")
```

---

## 5. Physics Applications

### 5.1 Eigenfunction Expansion of the Heat Equation

**Heat equation** for a rod of length $L$: $\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}$

Boundary: $u(0,t) = u(L,t) = 0$, Initial: $u(x,0) = f(x)$

Separation of variables $u = X(x)T(t)$ $\rightarrow$ S-L problem $X'' + \lambda X = 0$:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\frac{n\pi x}{L} e^{-\alpha^2 (n\pi/L)^2 t}, \quad b_n = \frac{2}{L}\int_0^L f(x) \sin\frac{n\pi x}{L} dx$$

```python
import numpy as np
import matplotlib.pyplot as plt

L, alpha2, N = 1.0, 0.01, 50
x = np.linspace(0, L, 200)
f = lambda t: np.sin(np.pi*t/L) + 0.5*np.sin(3*np.pi*t/L)

def bn(n):
    return 2/L * np.trapz(f(x) * np.sin(n*np.pi*x/L), x)

def u(x_val, t):
    return sum(bn(n)*np.sin(n*np.pi*x_val/L)*np.exp(-alpha2*(n*np.pi/L)**2*t)
               for n in range(1, N+1))

fig, ax = plt.subplots(figsize=(10, 6))
for t_val, c in zip([0, 0.5, 2, 5, 10, 30], plt.cm.coolwarm(np.linspace(0,1,6))):
    ax.plot(x, u(x, t_val), color=c, linewidth=2, label=f'$t={t_val}$')

ax.set_xlabel('$x$'); ax.set_ylabel('$u(x,t)$')
ax.set_title('Eigenfunction Expansion Solution of the Heat Equation', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

### 5.2 Normal Modes of Vibration Problems

**Wave equation** for a string fixed at both ends: $\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$

The same S-L problem with a different time component: $T_n(t) = A_n\cos(\omega_n t) + B_n\sin(\omega_n t)$

$\omega_n = cn\pi/L$ is the $n$-th **natural frequency**, and $X_n T_n$ is the **normal mode**.

```python
import numpy as np
import matplotlib.pyplot as plt

L, c = 1.0, 1.0
x = np.linspace(0, L, 300)

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for mode_n, ax in zip([1, 2, 3], axes):
    omega = c * mode_n * np.pi / L
    X_n = np.sin(mode_n * np.pi * x / L)
    for t in [0, 0.1, 0.25, 0.4]:
        ax.plot(x, X_n*np.cos(omega*t), linewidth=1.5, label=f't={t}')
    ax.set_ylabel(f'Mode {mode_n}')
    ax.set_title(f'$\\omega_{mode_n} = {omega:.3f}$ rad/s')
    ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('$x$')
plt.suptitle('Normal Modes of a String', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

### 5.3 Energy Eigenstates in Quantum Mechanics

**Time-independent Schrödinger equation**: $-\frac{\hbar^2}{2m}\psi'' + V(x)\psi = E\psi$

S-L form: $p = \hbar^2/(2m)$, $q = -V(x)$, $w = 1$, $\lambda = E$

**Infinite square well** ($V=0$ for $0<x<L$):
- Energy: $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$
- Wavefunction: $\psi_n(x) = \sqrt{2/L}\sin(n\pi x/L)$

**Physical meaning of S-L**:
- Real eigenvalues $\rightarrow$ observables (energy) are real
- Orthogonality $\rightarrow$ different energy states are orthogonal
- Completeness $\rightarrow$ arbitrary state = superposition of energy eigenstates

```python
import numpy as np
import matplotlib.pyplot as plt

L, hbar, m = 1.0, 1.0, 1.0
x = np.linspace(0, L, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for n in range(1, 6):
    E_n = (n*np.pi*hbar)**2 / (2*m*L**2)
    psi = np.sqrt(2/L) * np.sin(n*np.pi*x/L)
    # Wavefunction (displayed above energy baseline)
    ax1.axhline(E_n, color='gray', linewidth=0.5, linestyle='--')
    ax1.plot(x, E_n + 3*psi, linewidth=2, label=f'$n={n}, E={E_n:.1f}$')
    ax1.fill_between(x, E_n, E_n + 3*psi, alpha=0.1)
    # Probability density
    ax2.plot(x, psi**2, linewidth=2, label=f'$|\\psi_{n}|^2$')

ax1.set_title('Energy Eigenstates'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax2.set_title('Probability Density'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.suptitle('Quantum Mechanics: Application of S-L Theory', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Orthonormality verification
print("\n=== Quantum Eigenstate Orthonormality ===")
for i in range(1, 4):
    for j in range(i, 4):
        psi_i = np.sqrt(2/L)*np.sin(i*np.pi*x/L)
        psi_j = np.sqrt(2/L)*np.sin(j*np.pi*x/L)
        print(f"  <psi_{i}|psi_{j}> = {np.trapz(psi_i*psi_j, x):.6f}")
```

**Harmonic oscillator**: For $V = \frac{1}{2}m\omega^2 x^2$, the equation reduces to the Hermite equation with eigenfunctions as Hermite functions, $E_n = \hbar\omega(n+1/2)$.

---

## Exercises

### Basic Problems

**Problem 1.** Transform the following to S-L standard form and find $p, q, w$.
(a) Chebyshev: $(1-x^2)y'' - xy' + n^2 y = 0$
(b) Laguerre: $xy'' + (1-x)y' + ny = 0$

**Problem 2.** Find the eigenvalues and eigenfunctions of $y'' + \lambda y = 0$, $y'(0) = 0$, $y'(L) = 0$ (Neumann boundary conditions).

**Problem 3.** Expand $f(x) = \begin{cases} 1 & 0 < x < \pi/2 \\ 0 & \pi/2 < x < \pi \end{cases}$ in a $\sin(nx)$ series on $[0, \pi]$.

### Advanced Problems

**Problem 4.** Perform Gram-Schmidt on $\{1, x, x^2\}$ with $w(x) = (1-x^2)^{-1/2}$, interval $[-1,1]$ to derive the Chebyshev polynomials $T_0, T_1, T_2$.

**Problem 5.** For the heat equation with initial temperature $f(x) = 4x(1-x)$, $L=1$, $\alpha^2=0.01$, find the solution and plot the temperature at $t=0, 0.1, 1, 10$.

**Problem 6 (Quantum mechanics).** For an infinite square well with $\Psi(x,0) = Ax(L-x)$:
(a) Find normalization constant $A$, (b) expansion coefficients $c_n$, (c) $\langle E \rangle = \sum_n |c_n|^2 E_n$.

**Problem 7 (Rayleigh quotient).** For the problem $y'' + \lambda y = 0$, $y(0) = y(L) = 0$ with trial function $\phi(x) = x^2(L-x)^2$, compute the Rayleigh quotient analytically and compare with the exact $\lambda_1 = (\pi/L)^2$. (Hint: compute $\int_0^L [x^2(L-x)^2]^2 dx$ and $\int_0^L [\phi']^2 dx$ separately)

**Problem 8 (Sturm comparison).** In the Schrödinger equation $\psi'' + [E - V(x)]\psi = 0$, when interpreting effective potential as $q(x) = E - V(x)$, use the Sturm comparison theorem to predict the number of zeros of the $E_3$ state wavefunction for $V(x) = x^2$ (harmonic oscillator).

---

## References

- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd Ed., Ch. 12
- **Arfken, Weber, Harris.** *Mathematical Methods for Physicists*, 7th Ed., Ch. 10
- **Riley, Hobson, Bence.** *Mathematical Methods for Physics and Engineering*, Ch. 17
- **Zettl, A.** *Sturm-Liouville Theory*, AMS (2005)

---

## Next Lesson

- [13. Partial Differential Equations](13_Partial_Differential_Equations.md) - PDE solutions where S-L theory serves as a key tool in the method of separation of variables

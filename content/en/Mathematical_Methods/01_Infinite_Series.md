# 01. Infinite Series and Convergence

## Learning Objectives

- Define **convergence and divergence of series** and determine convergence using partial sums
- Apply major convergence tests: **comparison test, ratio test, root test, integral test, alternating series test**
- Find the radius of convergence for **power series** and perform term-by-term differentiation/integration
- Approximate functions using **Taylor series and Maclaurin series** and apply them to physics problems
- Understand the concepts of **asymptotic series** and **Stirling's approximation** and utilize them in physical applications

---

## 1. Basic Concepts of Series

### 1.1 Sequences and Series

A **sequence** is a function from natural numbers to real (or complex) numbers:

$$a_1, a_2, a_3, \ldots, a_n, \ldots$$

A sequence $\{a_n\}$ converges to a specific value $L$ if:

$$\lim_{n \to \infty} a_n = L$$

This means that for sufficiently large $n$, $a_n$ becomes arbitrarily close to $L$.

A **series** is the sum of the terms of a sequence:

$$S = \sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots$$

**Necessary condition** for series convergence: $\lim_{n \to \infty} a_n = 0$

> **Note**: This is only a necessary condition, not sufficient. The harmonic series $\sum \frac{1}{n}$ has $a_n \to 0$ but diverges.

### 1.2 Partial Sums and Convergence

A **partial sum** $S_N$ is the sum of the first $N$ terms of the series:

$$S_N = \sum_{n=1}^{N} a_n = a_1 + a_2 + \cdots + a_N$$

**Convergence** of a series: If the sequence of partial sums $\{S_N\}$ converges to a finite value $S$, we say the series $\sum a_n$ converges to $S$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: Partial sums of a geometric series
# sum_{n=0}^{inf} r^n = 1/(1-r)  (|r| < 1)

def geometric_partial_sums(r, N_max):
    """Compute partial sums of the geometric series."""
    n_values = np.arange(N_max + 1)
    terms = r ** n_values
    partial_sums = np.cumsum(terms)
    return n_values, partial_sums

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Convergent case: |r| < 1
r_conv = 0.5
n_vals, S_n = geometric_partial_sums(r_conv, 20)
exact = 1 / (1 - r_conv)
axes[0].plot(n_vals, S_n, 'bo-', markersize=4, label=f'S_N (r={r_conv})')
axes[0].axhline(y=exact, color='r', linestyle='--', label=f'S = {exact:.4f}')
axes[0].set_xlabel('N')
axes[0].set_ylabel('S_N')
axes[0].set_title(f'Convergent geometric series (r = {r_conv})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Harmonic series vs p-series comparison
N_max = 100
n = np.arange(1, N_max + 1)
harmonic = np.cumsum(1.0 / n)           # p = 1 (diverges)
p2_series = np.cumsum(1.0 / n**2)       # p = 2 (converges)
p3_series = np.cumsum(1.0 / n**3)       # p = 3 (converges)

axes[1].plot(n, harmonic, 'r-', label='p=1 (harmonic series, diverges)')
axes[1].plot(n, p2_series, 'b-', label=f'p=2 (converges, pi^2/6 = {np.pi**2/6:.4f})')
axes[1].plot(n, p3_series, 'g-', label='p=3 (converges)')
axes[1].axhline(y=np.pi**2/6, color='b', linestyle='--', alpha=0.5)
axes[1].set_xlabel('N')
axes[1].set_ylabel('S_N')
axes[1].set_title('p-series comparison: sum(1/n^p)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('series_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Key series results**:

| Series | Convergence condition | Sum (when convergent) |
|------|-----------|-------------|
| Geometric series $\sum r^n$ | $|r| < 1$ | $\frac{1}{1-r}$ |
| p-series $\sum \frac{1}{n^p}$ | $p > 1$ | $\zeta(p)$ |
| Harmonic series $\sum \frac{1}{n}$ | Diverges | - |

---

## 2. Convergence Tests

### 2.1 Comparison Test

If $0 \leq a_n \leq b_n$ (for sufficiently large $n$), then:

- $\sum b_n$ converges $\Rightarrow$ $\sum a_n$ converges
- $\sum a_n$ diverges $\Rightarrow$ $\sum b_n$ diverges

**Limit Comparison Test**: If $a_n > 0$, $b_n > 0$ and

$$\lim_{n \to \infty} \frac{a_n}{b_n} = L \quad (0 < L < \infty)$$

then $\sum a_n$ and $\sum b_n$ either both converge or both diverge.

### 2.2 Ratio Test

$$\rho = \lim_{n \to \infty} \left|\frac{a_{n+1}}{a_n}\right|$$

- $\rho < 1$ implies **absolutely convergent**
- $\rho > 1$ implies **divergent**
- $\rho = 1$ implies **inconclusive** (use another test)

> **Physics Tip**: The ratio test is particularly useful for series containing factorials or exponential functions.

### 2.3 Root Test

$$\rho = \lim_{n \to \infty} |a_n|^{1/n}$$

- $\rho < 1$ implies absolutely convergent
- $\rho > 1$ implies divergent
- $\rho = 1$ implies inconclusive

### 2.4 Integral Test

If $f(x)$ is positive, continuous, and decreasing on $[1, \infty)$ with $f(n) = a_n$, then:

$$\sum_{n=1}^{\infty} a_n \quad \text{and} \quad \int_1^{\infty} f(x) \, dx \quad \text{both converge or both diverge}$$

**Example**: Deriving the convergence condition for p-series

$$\int_1^{\infty} \frac{1}{x^p} \, dx = \left[\frac{x^{1-p}}{1-p}\right]_1^{\infty}$$

If $p > 1$, the integral converges, so $\sum \frac{1}{n^p}$ also converges.

### 2.5 Alternating Series Test

An alternating series $\sum (-1)^n b_n$ ($b_n > 0$) converges if:

1. $b_{n+1} \leq b_n$ (monotone decreasing)
2. $\lim_{n \to \infty} b_n = 0$

Error bound for alternating series: $|S - S_N| \leq b_{N+1}$ (less than or equal to the absolute value of the next term)

```python
import numpy as np
import sympy as sp

# Automated convergence test tool
def test_convergence(a_n_func, name="series"):
    """Apply the ratio test and root test to the given series."""
    n = sp.Symbol('n', positive=True, integer=True)
    a_n = a_n_func(n)

    print(f"=== {name}: sum a_n, a_n = {a_n} ===\n")

    # Ratio test
    ratio = sp.simplify(a_n_func(n + 1) / a_n)
    rho_ratio = sp.limit(sp.Abs(ratio), n, sp.oo)
    print(f"Ratio test: |a_(n+1)/a_n| -> {rho_ratio}")
    if rho_ratio < 1:
        print("  => Absolutely convergent\n")
    elif rho_ratio > 1:
        print("  => Divergent\n")
    else:
        print("  => Inconclusive\n")

    # Root test
    root = sp.Abs(a_n) ** (sp.Rational(1, n))
    rho_root = sp.limit(root, n, sp.oo)
    print(f"Root test: |a_n|^(1/n) -> {rho_root}")
    if rho_root < 1:
        print("  => Absolutely convergent\n")
    elif rho_root > 1:
        print("  => Divergent\n")
    else:
        print("  => Inconclusive\n")

# Test examples
n = sp.Symbol('n', positive=True, integer=True)

# 1) sum n! / n^n  (converges)
test_convergence(lambda n: sp.factorial(n) / n**n, "n!/n^n")

# 2) sum 1/n^2  (converges)
test_convergence(lambda n: 1 / n**2, "1/n^2")

# 3) sum n^2 / 2^n  (converges)
test_convergence(lambda n: n**2 / 2**n, "n^2/2^n")
```

```python
import numpy as np
import matplotlib.pyplot as plt

# Alternating series visualization: ln(2) = sum_{n=1}^{inf} (-1)^{n+1} / n
N = 30
n = np.arange(1, N + 1)
terms = (-1.0) ** (n + 1) / n
partial_sums = np.cumsum(terms)

plt.figure(figsize=(10, 5))
plt.plot(n, partial_sums, 'bo-', markersize=5, label='Partial sum S_N')
plt.axhline(y=np.log(2), color='r', linestyle='--',
            label=f'ln(2) = {np.log(2):.6f}')

# Show error bounds
for i in range(len(n)):
    error_bound = 1.0 / (n[i] + 1)
    plt.plot([n[i], n[i]],
             [partial_sums[i] - error_bound, partial_sums[i] + error_bound],
             'g-', alpha=0.3, linewidth=2)

plt.xlabel('N')
plt.ylabel('S_N')
plt.title('Alternating series: ln(2) = 1 - 1/2 + 1/3 - 1/4 + ...')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alternating_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3. Power Series

### 3.1 Radius of Convergence

A **power series** has the form:

$$f(x) = \sum_{n=0}^{\infty} c_n (x - a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots$$

where $a$ is the **center** and $c_n$ are **coefficients**.

**Radius of convergence** $R$:

$$R = \lim_{n \to \infty} \left|\frac{c_n}{c_{n+1}}\right| \quad \text{or} \quad R = \frac{1}{\lim_{n \to \infty} |c_n|^{1/n}}$$

- $|x - a| < R$ implies absolute convergence
- $|x - a| > R$ implies divergence
- $|x - a| = R$ requires separate investigation

**Interval of convergence**:

```
      Diverges    Converges    Diverges
  -------|========|========|--------->  x
        a-R       a       a+R
         <---- R ---->
```

### 3.2 Term-by-Term Differentiation and Integration

Within the interval of convergence, power series can be differentiated and integrated **term-by-term**:

$$f'(x) = \sum_{n=1}^{\infty} n c_n (x-a)^{n-1}$$

$$\int f(x) \, dx = C + \sum_{n=0}^{\infty} \frac{c_n (x-a)^{n+1}}{n+1}$$

The **radius of convergence remains the same** after differentiation or integration (though convergence at endpoints may change).

```python
import sympy as sp

x = sp.Symbol('x')

# Example: Computing the radius of convergence for power series
print("=== Radius of convergence for power series ===\n")

# 1) sum x^n / n!  (exponential function)
n = sp.Symbol('n', positive=True, integer=True)
c_n = 1 / sp.factorial(n)
c_n1 = 1 / sp.factorial(n + 1)
R1 = sp.limit(sp.Abs(c_n / c_n1), n, sp.oo)
print(f"sum x^n/n!: R = {R1}  (converges for all x)")

# 2) sum n * x^n  (R = 1)
c_n = n
c_n1 = n + 1
R2 = sp.limit(sp.Abs(c_n / c_n1), n, sp.oo)
print(f"sum n*x^n: R = {R2}")

# 3) sum n! * x^n  (R = 0, converges only at x=0)
c_n = sp.factorial(n)
c_n1 = sp.factorial(n + 1)
R3 = sp.limit(sp.Abs(c_n / c_n1), n, sp.oo)
print(f"sum n!*x^n: R = {R3}  (converges only at x=0)")

# Term-by-term differentiation visualization
print("\n=== Term-by-term differentiation: d/dx [1/(1-x)] = 1/(1-x)^2 ===")
# 1/(1-x) = sum x^n  =>  differentiating gives  1/(1-x)^2 = sum n*x^{n-1}
x_vals = np.linspace(-0.9, 0.9, 200)

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original function and series
for N in [3, 5, 10, 20]:
    series_sum = sum(x_vals**k for k in range(N))
    axes[0].plot(x_vals, series_sum, alpha=0.7, label=f'N={N}')

axes[0].plot(x_vals, 1/(1-x_vals), 'k--', linewidth=2, label='1/(1-x)')
axes[0].set_ylim(-5, 15)
axes[0].set_title('1/(1-x) = sum x^n')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Differentiated function and series
for N in [3, 5, 10, 20]:
    deriv_sum = sum(k * x_vals**(k-1) for k in range(1, N))
    axes[1].plot(x_vals, deriv_sum, alpha=0.7, label=f'N={N}')

axes[1].plot(x_vals, 1/(1-x_vals)**2, 'k--', linewidth=2, label='1/(1-x)^2')
axes[1].set_ylim(-5, 30)
axes[1].set_title('1/(1-x)^2 = sum n*x^{n-1}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_series_differentiation.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. Taylor Series and Maclaurin Series

### 4.1 Taylor Expansion

If a function $f(x)$ is infinitely differentiable near $x = a$, it can be expanded as a **Taylor series**:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n$$

$$= f(a) + f'(a)(x-a) + \frac{f''(a)}{2!} (x-a)^2 + \frac{f'''(a)}{3!} (x-a)^3 + \cdots$$

When $a = 0$, this is called a **Maclaurin series**.

**Taylor remainder** $R_N$: the error after expanding to $N$ terms

$$R_N(x) = \frac{f^{(N+1)}(c)}{(N+1)!} (x-a)^{N+1} \quad \text{(Lagrange remainder, } a < c < x\text{)}$$

### 4.2 Series Representations of Common Functions

Most frequently used series expansions in physics:

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots \quad \text{(all } x\text{)}$$

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots \quad \text{(all } x\text{)}$$

$$\cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots \quad \text{(all } x\text{)}$$

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots \quad (-1 < x \leq 1)$$

$$(1+x)^p = 1 + px + \frac{p(p-1)}{2!} x^2 + \cdots \quad (|x| < 1, \text{ binomial series})$$

$$\frac{1}{1-x} = 1 + x + x^2 + x^3 + \cdots \quad (|x| < 1)$$

$$\arctan(x) = x - \frac{x^3}{3} + \frac{x^5}{5} - \frac{x^7}{7} + \cdots \quad (|x| \leq 1)$$

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Taylor expansion using SymPy
x = sp.Symbol('x')

functions = {
    'e^x': sp.exp(x),
    'sin(x)': sp.sin(x),
    'cos(x)': sp.cos(x),
    'ln(1+x)': sp.ln(1 + x),
}

print("=== Taylor series of common functions (a=0, up to 6th order) ===\n")
for name, func in functions.items():
    taylor = sp.series(func, x, 0, n=7)
    print(f"{name} = {taylor}\n")

# Taylor series convergence visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 500)

# Taylor approximation of sin(x)
ax = axes[0, 0]
ax.plot(x_vals, np.sin(x_vals), 'k-', linewidth=2, label='sin(x)')
for N in [1, 3, 5, 7, 9]:
    approx = sum((-1)**k * x_vals**(2*k+1) / np.math.factorial(2*k+1)
                 for k in range(N))
    ax.plot(x_vals, approx, alpha=0.7, label=f'order {2*N-1}')
ax.set_ylim(-2, 2)
ax.set_title('Taylor approximation of sin(x)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Taylor approximation of cos(x)
ax = axes[0, 1]
ax.plot(x_vals, np.cos(x_vals), 'k-', linewidth=2, label='cos(x)')
for N in [1, 2, 4, 6, 8]:
    approx = sum((-1)**k * x_vals**(2*k) / np.math.factorial(2*k)
                 for k in range(N))
    ax.plot(x_vals, approx, alpha=0.7, label=f'order {2*N-2}')
ax.set_ylim(-2, 2)
ax.set_title('Taylor approximation of cos(x)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Taylor approximation of e^x
x_exp = np.linspace(-3, 3, 300)
ax = axes[1, 0]
ax.plot(x_exp, np.exp(x_exp), 'k-', linewidth=2, label='e^x')
for N in [1, 2, 4, 6, 10]:
    approx = sum(x_exp**k / np.math.factorial(k) for k in range(N + 1))
    ax.plot(x_exp, approx, alpha=0.7, label=f'order {N}')
ax.set_ylim(-5, 20)
ax.set_title('Taylor approximation of e^x')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Taylor approximation of ln(1+x)
x_ln = np.linspace(-0.95, 2, 300)
ax = axes[1, 1]
ax.plot(x_ln, np.log(1 + x_ln), 'k-', linewidth=2, label='ln(1+x)')
for N in [1, 3, 5, 10, 20]:
    x_series = np.linspace(-0.95, 0.95, 300)  # within convergence interval
    approx = sum((-1)**(k+1) * x_series**k / k for k in range(1, N + 1))
    ax.plot(x_series, approx, alpha=0.7, label=f'order {N}')
ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5, label='x=1 (boundary)')
ax.axvline(x=-1, color='gray', linestyle=':', alpha=0.5, label='x=-1 (boundary)')
ax.set_ylim(-4, 2)
ax.set_title('Taylor approximation of ln(1+x)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('taylor_series_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 Approximation Using Series

The most important application of series expansion in physics is **small-quantity approximation**.

When $\varepsilon \ll 1$:

$$(1 + \varepsilon)^p \approx 1 + p\varepsilon + \frac{p(p-1)}{2} \varepsilon^2 + \cdots$$

$$e^\varepsilon \approx 1 + \varepsilon + \frac{\varepsilon^2}{2} + \cdots$$

$$\sin(\varepsilon) \approx \varepsilon - \frac{\varepsilon^3}{6} + \cdots$$

$$\cos(\varepsilon) \approx 1 - \frac{\varepsilon^2}{2} + \cdots$$

$$\tan(\varepsilon) \approx \varepsilon + \frac{\varepsilon^3}{3} + \cdots$$

$$\frac{1}{1+\varepsilon} \approx 1 - \varepsilon + \varepsilon^2 - \cdots$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Accuracy comparison of small-quantity approximations
epsilon = np.linspace(0, 1, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Approximation of sqrt(1 + epsilon) = (1+epsilon)^{1/2}
exact = np.sqrt(1 + epsilon)
order1 = 1 + epsilon / 2                       # 1st-order approximation
order2 = 1 + epsilon / 2 - epsilon**2 / 8      # 2nd-order approximation
order3 = 1 + epsilon/2 - epsilon**2/8 + epsilon**3/16  # 3rd-order

axes[0].plot(epsilon, exact, 'k-', linewidth=2, label='Exact')
axes[0].plot(epsilon, order1, 'b--', label='1st order: 1 + e/2')
axes[0].plot(epsilon, order2, 'r--', label='2nd order: 1 + e/2 - e^2/8')
axes[0].plot(epsilon, order3, 'g--', label='3rd order')
axes[0].set_xlabel('epsilon')
axes[0].set_ylabel('sqrt(1 + epsilon)')
axes[0].set_title('Binomial series approximation: (1+e)^{1/2}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Relative error
rel_err1 = np.abs(order1 - exact) / exact * 100
rel_err2 = np.abs(order2 - exact) / exact * 100
rel_err3 = np.abs(order3 - exact) / exact * 100

axes[1].semilogy(epsilon, rel_err1, 'b-', label='1st-order approx')
axes[1].semilogy(epsilon, rel_err2, 'r-', label='2nd-order approx')
axes[1].semilogy(epsilon, rel_err3, 'g-', label='3rd-order approx')
axes[1].set_xlabel('epsilon')
axes[1].set_ylabel('Relative error (%)')
axes[1].set_title('Relative error by approximation order')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('approximation_error.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. Asymptotic Series

### 5.1 Divergent but Useful Series

An **asymptotic series** diverges when summed to infinity, but finite partial sums provide excellent approximations to a function.

Asymptotic expansion of a function $f(x)$:

$$f(x) \sim \sum_{n=0}^{\infty} \frac{a_n}{x^n} \quad (x \to \infty)$$

This means that for the partial sum of $N$ terms, the following holds:

$$\lim_{x \to \infty} x^N \left[f(x) - \sum_{n=0}^{N} \frac{a_n}{x^n}\right] = 0 \quad \text{(for each } N\text{)}$$

> **Key Point**: The entire series diverges, but truncating at the **optimal number of terms** yields an excellent approximation.
> Generally, the optimal truncation point is where the terms stop decreasing and start increasing again.

### 5.2 Stirling's Approximation

**Stirling's approximation** is an asymptotic approximation for factorials:

$$n! \sim \sqrt{2\pi n} \left(\frac{n}{e}\right)^n \quad (n \to \infty)$$

$$\ln(n!) \sim n\ln(n) - n + \frac{1}{2}\ln(2\pi n)$$

This approximation is crucial in statistical mechanics for calculating Boltzmann entropy, combinatorial problems, etc.

More precise Stirling series:

$$n! \sim \sqrt{2\pi n} \left(\frac{n}{e}\right)^n \left(1 + \frac{1}{12n} + \frac{1}{288n^2} - \frac{139}{51840n^3} + \cdots\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial

# Accuracy of Stirling's approximation
n_vals = np.arange(1, 51)

# Exact ln(n!)
exact_lnfact = np.array([np.sum(np.log(np.arange(1, n+1))) for n in n_vals])

# Stirling's approximation (multiple orders)
stirling_0 = n_vals * np.log(n_vals) - n_vals  # simplest form
stirling_1 = stirling_0 + 0.5 * np.log(2 * np.pi * n_vals)  # 1st-order correction
stirling_2 = stirling_1 + 1 / (12 * n_vals)  # 2nd-order correction

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Absolute value comparison
axes[0].plot(n_vals, exact_lnfact, 'k-', linewidth=2, label='ln(n!) exact')
axes[0].plot(n_vals, stirling_0, 'b--', label='n*ln(n) - n')
axes[0].plot(n_vals, stirling_1, 'r--', label='+ (1/2)ln(2*pi*n)')
axes[0].set_xlabel('n')
axes[0].set_ylabel('ln(n!)')
axes[0].set_title("Stirling's approximation")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Relative error
rel_err0 = np.abs(stirling_0 - exact_lnfact) / exact_lnfact * 100
rel_err1 = np.abs(stirling_1 - exact_lnfact) / exact_lnfact * 100
rel_err2 = np.abs(stirling_2 - exact_lnfact) / exact_lnfact * 100

axes[1].semilogy(n_vals, rel_err0, 'b-', label='0th order: n*ln(n)-n')
axes[1].semilogy(n_vals, rel_err1, 'r-', label='1st order: + ln(sqrt(2*pi*n))')
axes[1].semilogy(n_vals, rel_err2, 'g-', label='2nd order: + 1/(12n)')
axes[1].set_xlabel('n')
axes[1].set_ylabel('Relative error (%)')
axes[1].set_title("Relative error of Stirling's approximation")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stirling_approximation.png', dpi=150, bbox_inches='tight')
plt.show()

# Statistical mechanics application: Boltzmann entropy
print("=== Statistical mechanics application: Ideal gas entropy ===\n")
print("When N particles have W microstates:")
print("S = k_B * ln(W)")
print("Using Stirling's approximation:")
print("ln(N!) ~ N*ln(N) - N")
print()
print("Example: N = 10^23 (Avogadro's number scale)")
N = 1e23
print(f"ln(N!) ~ N*ln(N) - N = {N * np.log(N) - N:.4e}")
print("Direct computation is impossible, but easy with Stirling's approximation!")
```

---

## 6. Applications in Physics

### 6.1 Period of a Pendulum (Series Expansion)

The exact period of a simple pendulum is given by a complete elliptic integral:

$$T = 4\sqrt{\frac{L}{g}} K\left(\sin\frac{\theta_0}{2}\right)$$

where $K(k)$ is the complete elliptic integral of the first kind. Expanding as a series:

$$T = 2\pi\sqrt{\frac{L}{g}} \left[1 + \frac{1}{4}\sin^2\frac{\theta_0}{2} + \frac{9}{64}\sin^4\frac{\theta_0}{2} + \cdots\right]$$

$$\approx T_0 \left[1 + \frac{\theta_0^2}{16} + \frac{11\theta_0^4}{3072} + \cdots\right]$$

where $T_0 = 2\pi\sqrt{L/g}$ is the period in the small-angle approximation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

# Pendulum period: exact vs series approximation
g = 9.81  # m/s^2
L = 1.0   # m
T0 = 2 * np.pi * np.sqrt(L / g)  # small-angle approximation period

theta0 = np.linspace(0.01, np.pi * 0.95, 200)  # initial angle (rad)
k = np.sin(theta0 / 2)

# Exact period (complete elliptic integral)
T_exact = 4 * np.sqrt(L / g) * ellipk(k**2)

# Series approximations (from small-angle to higher-order)
T_approx0 = T0 * np.ones_like(theta0)                          # 0th order (small-angle)
T_approx2 = T0 * (1 + (1/4) * k**2)                           # 2nd order
T_approx4 = T0 * (1 + (1/4)*k**2 + (9/64)*k**4)               # 4th order
T_approx6 = T0 * (1 + (1/4)*k**2 + (9/64)*k**4 + (25/256)*k**6)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Period comparison
theta_deg = np.degrees(theta0)
axes[0].plot(theta_deg, T_exact / T0, 'k-', linewidth=2, label='Exact')
axes[0].plot(theta_deg, T_approx0 / T0, 'b--', label='0th order (small-angle)')
axes[0].plot(theta_deg, T_approx2 / T0, 'r--', label='2nd-order correction')
axes[0].plot(theta_deg, T_approx4 / T0, 'g--', label='4th-order correction')
axes[0].plot(theta_deg, T_approx6 / T0, 'm--', label='6th-order correction')
axes[0].set_xlabel('Initial angle (degrees)')
axes[0].set_ylabel('T / T_0')
axes[0].set_title('Pendulum period: series expansion approximation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Relative error
axes[1].semilogy(theta_deg, np.abs(T_approx0 - T_exact)/T_exact * 100,
                 'b-', label='0th order (small-angle)')
axes[1].semilogy(theta_deg, np.abs(T_approx2 - T_exact)/T_exact * 100,
                 'r-', label='2nd order')
axes[1].semilogy(theta_deg, np.abs(T_approx4 - T_exact)/T_exact * 100,
                 'g-', label='4th order')
axes[1].semilogy(theta_deg, np.abs(T_approx6 - T_exact)/T_exact * 100,
                 'm-', label='6th order')
axes[1].set_xlabel('Initial angle (degrees)')
axes[1].set_ylabel('Relative error (%)')
axes[1].set_title('Relative error of series approximation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pendulum_period.png', dpi=150, bbox_inches='tight')
plt.show()

# Numerical results
print("=== Pendulum period comparison (L=1m) ===\n")
print(f"{'Angle':>6s}  {'Exact (s)':>10s}  {'Small-angle (s)':>15s}  {'Error (%)':>9s}")
print("-" * 45)
for angle_deg in [5, 10, 30, 45, 60, 90, 120, 150]:
    angle_rad = np.radians(angle_deg)
    k_val = np.sin(angle_rad / 2)
    T_ex = 4 * np.sqrt(L / g) * ellipk(k_val**2)
    err = (T0 - T_ex) / T_ex * 100
    print(f"{angle_deg:>5d}   {T_ex:>10.6f}  {T0:>12.6f}  {err:>8.3f}")
```

### 6.2 Non-relativistic Approximation of Relativistic Energy

Einstein's relativistic energy-momentum relation:

$$E = \gamma mc^2 = \frac{mc^2}{\sqrt{1 - v^2/c^2}}$$

Expanding the kinetic energy $K = E - mc^2$ as a series (when $\beta = v/c \ll 1$):

$$\gamma = \frac{1}{\sqrt{1 - \beta^2}} = 1 + \frac{1}{2}\beta^2 + \frac{3}{8}\beta^4 + \frac{5}{16}\beta^6 + \cdots$$

$$K = mc^2 (\gamma - 1)$$

$$= \underbrace{\frac{1}{2}mv^2}_{\text{Newtonian}} + \underbrace{\frac{3}{8}m\frac{v^4}{c^2} + \frac{5}{16}m\frac{v^6}{c^4} + \cdots}_{\text{Relativistic correction}}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Series approximation of relativistic kinetic energy
beta = np.linspace(0.001, 0.99, 500)  # v/c

# Exact relativistic kinetic energy: K/(mc^2) = gamma - 1
gamma = 1 / np.sqrt(1 - beta**2)
K_exact = gamma - 1  # in units of mc^2

# Series approximation (binomial expansion)
K_order2 = 0.5 * beta**2                                         # Newtonian
K_order4 = 0.5 * beta**2 + (3/8) * beta**4                      # 1st-order correction
K_order6 = 0.5 * beta**2 + (3/8) * beta**4 + (5/16) * beta**6  # 2nd-order correction

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Kinetic energy comparison
axes[0].plot(beta, K_exact, 'k-', linewidth=2, label='Relativistic (exact)')
axes[0].plot(beta, K_order2, 'b--', label='Newtonian: (1/2)mv^2')
axes[0].plot(beta, K_order4, 'r--', label='1st-order correction')
axes[0].plot(beta, K_order6, 'g--', label='2nd-order correction')
axes[0].set_xlabel('beta = v/c')
axes[0].set_ylabel('K / (mc^2)')
axes[0].set_title('Kinetic energy: Relativistic vs Newtonian')
axes[0].set_ylim(0, 5)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Relative error of Newtonian mechanics
beta_low = np.linspace(0.001, 0.5, 300)
gamma_low = 1 / np.sqrt(1 - beta_low**2)
K_exact_low = gamma_low - 1
K_newton = 0.5 * beta_low**2
K_corr1 = 0.5 * beta_low**2 + (3/8) * beta_low**4

rel_err_newton = np.abs(K_newton - K_exact_low) / K_exact_low * 100
rel_err_corr1 = np.abs(K_corr1 - K_exact_low) / K_exact_low * 100

axes[1].semilogy(beta_low, rel_err_newton, 'b-', label='Newtonian error')
axes[1].semilogy(beta_low, rel_err_corr1, 'r-', label='Error after 1st-order correction')
axes[1].axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='1% error line')
axes[1].set_xlabel('beta = v/c')
axes[1].set_ylabel('Relative error (%)')
axes[1].set_title('Valid range of Newtonian mechanics')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('relativistic_energy.png', dpi=150, bbox_inches='tight')
plt.show()

# Print results
print("=== Valid velocity range for Newtonian mechanics ===\n")
print(f"{'v/c':>6s}  {'Relativistic K':>14s}  {'Newtonian K':>11s}  {'Error (%)':>9s}")
print("-" * 42)
for b in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    g = 1 / np.sqrt(1 - b**2)
    K_rel = g - 1
    K_new = 0.5 * b**2
    err = abs(K_new - K_rel) / K_rel * 100
    print(f"{b:>6.2f}  {K_rel:>12.6f}  {K_new:>10.6f}  {err:>8.3f}")
```

### 6.3 Electric Multipole Expansion

When a point charge $q$ is located at distance $d$ from the origin, the potential at a point far from the origin ($r \gg d$) can be expanded as a series.

```
    Observation point P(r, theta)
    *
    |\
    | \  r
    |  \
    |   \
    |theta\
    |------*--- Charge q (distance d, on z-axis)
    |
    Origin O
```

Distance to observation point:

$$\frac{1}{|\mathbf{r} - \mathbf{d}|} = \frac{1}{r} \sum_{l=0}^{\infty} \left(\frac{d}{r}\right)^l P_l(\cos\theta)$$

where $P_l$ are Legendre polynomials.

For an electric dipole ($+q$ and $-q$ separated by distance $d$):

$$V(r, \theta) \approx \frac{1}{4\pi\varepsilon_0} \frac{p\cos\theta}{r^2} \quad (r \gg d)$$

where $p = qd$ is the dipole moment.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# Multipole Expansion visualization
# Series expansion of 1/|r - d*z_hat| when charge q is at z-axis position d = 1

def exact_potential_ratio(r, theta, d=1.0):
    """Exact potential ratio: r / |r - d*z_hat| (excluding the 1/r factor)"""
    # |r - d*z_hat|^2 = r^2 - 2*r*d*cos(theta) + d^2
    dist = np.sqrt(r**2 - 2*r*d*np.cos(theta) + d**2)
    return r / dist

def multipole_approx(r, theta, d=1.0, L_max=5):
    """Multipole expansion: sum_{l=0}^{L_max} (d/r)^l * P_l(cos(theta))"""
    result = np.zeros_like(r)
    cos_theta = np.cos(theta)
    for l in range(L_max + 1):
        Pl = legendre(l)
        result += (d / r)**l * Pl(cos_theta)
    return result

# Accuracy of multipole expansion as a function of r/d ratio
r_values = np.linspace(1.5, 10, 200)
theta = np.pi / 4  # 45 degrees

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

exact = exact_potential_ratio(r_values, theta)

for L_max in [0, 1, 2, 5, 10]:
    approx = multipole_approx(r_values, theta, L_max=L_max)
    axes[0].plot(r_values, approx, label=f'L_max = {L_max}')

axes[0].plot(r_values, exact, 'k--', linewidth=2, label='Exact')
axes[0].set_xlabel('r/d')
axes[0].set_ylabel('r * V(r) / (kq)')
axes[0].set_title(f'Multipole expansion (theta = 45 degrees)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Electric dipole potential pattern (2D)
r_grid = np.linspace(0.5, 5, 200)
theta_grid = np.linspace(0, 2*np.pi, 200)
R, Theta = np.meshgrid(r_grid, theta_grid)
X = R * np.sin(Theta)
Z = R * np.cos(Theta)

# Dipole potential: V ~ cos(theta)/r^2 (units omitted)
V_dipole = np.cos(Theta) / R**2

axes[1].contourf(X, Z, V_dipole, levels=np.linspace(-2, 2, 41),
                 cmap='RdBu_r', extend='both')
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title('Electric dipole potential pattern')
axes[1].set_aspect('equal')
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)
# Mark charge positions
axes[1].plot(0, 0.1, 'r+', markersize=15, markeredgewidth=2)
axes[1].plot(0, -0.1, 'b_', markersize=15, markeredgewidth=2)

plt.tight_layout()
plt.savefig('multipole_expansion.png', dpi=150, bbox_inches='tight')
plt.show()

# First few Legendre polynomials
print("=== Legendre polynomials P_l(x) ===\n")
import sympy as sp
x_sym = sp.Symbol('x')
for l in range(6):
    Pl = sp.legendre(l, x_sym)
    print(f"P_{l}(x) = {sp.expand(Pl)}")
```

---

## Practice Problems

### Problem 1. Convergence Test (Basic)

Determine the convergence/divergence of the following series. State the test used.

(a) $\sum_{n=1}^{\infty} \frac{n^2}{3^n}$

(b) $\sum_{n=2}^{\infty} \frac{1}{n \ln^2(n)}$

(c) $\sum_{n=1}^{\infty} \frac{(-1)^n n}{n^2 + 1}$

**Hint**: (a) ratio test, (b) integral test, (c) alternating series test

### Problem 2. Radius of Convergence (Basic)

Find the radius of convergence $R$ for the following power series.

(a) $\sum_{n=0}^{\infty} \frac{n! x^n}{(2n)!}$

(b) $\sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n \cdot 3^n}$

### Problem 3. Taylor Series Approximation (Intermediate)

Expand $f(x) = \sqrt{1 + x}$ as a Taylor series about $x = 0$ up to 3rd order, and:

(a) Find the approximate value of $\sqrt{1.1}$ (substitute $x = 0.1$)

(b) Calculate the relative error by comparing with the exact value

(c) Use Python to plot a comparison of errors for 1st, 2nd, and 3rd order approximations

### Problem 4. Physics Application (Intermediate)

For a pendulum with maximum angle $\theta_0 = 30$ degrees:

(a) Calculate the exact period (elliptic integral)

(b) Compare with the small-angle approximation period and find the error

(c) Find the error when including the 2nd order correction term

### Problem 5. Stirling's Approximation (Intermediate)

(a) Use Stirling's approximation to find $\log_{10}(100!)$

(b) Compare with Python's exact calculation

(c) For an ideal gas with $N$ molecules having $W$ microstates $W = N! / (n_1! n_2! \cdots n_k!)$, explain why Stirling's approximation is essential

### Problem 6. Comprehensive Application (Advanced)

When an electron has velocity $v = 0.1c$:

(a) Find the exact relativistic kinetic energy in $mc^2$ units

(b) Compare with Newtonian kinetic energy and find the relative error

(c) Calculate how much the error reduces when including the 1st order relativistic correction term $\frac{3}{8}m\frac{v^4}{c^2}$

(d) Find the maximum $v/c$ for which Newtonian mechanics has an error less than 1%

```python
# Practice problem solution helper
import numpy as np
import sympy as sp

# Problem 1(a) verification
n = sp.Symbol('n', positive=True, integer=True)
a_n = n**2 / 3**n
ratio = sp.simplify(sp.Abs((n+1)**2 / 3**(n+1)) / (n**2 / 3**n))
rho = sp.limit(ratio, n, sp.oo)
print(f"Problem 1(a): ratio = {rho} < 1 => converges")

# Problem 4 verification
from scipy.special import ellipk
L, g = 1.0, 9.81
theta0 = np.radians(30)
T0 = 2 * np.pi * np.sqrt(L / g)
k = np.sin(theta0 / 2)
T_exact = 4 * np.sqrt(L / g) * ellipk(k**2)
print(f"\nProblem 4: T_exact = {T_exact:.6f}s, T_0 = {T0:.6f}s")
print(f"Small-angle approx error = {abs(T0 - T_exact)/T_exact * 100:.4f}%")

T_corrected = T0 * (1 + (1/4) * k**2)
print(f"Error after 2nd-order correction = {abs(T_corrected - T_exact)/T_exact * 100:.6f}%")

# Problem 5 verification
import math
stirling_log10_100 = (100 * np.log(100) - 100 + 0.5 * np.log(200 * np.pi)) / np.log(10)
exact_log10_100 = np.log10(float(math.factorial(100)))
print(f"\nProblem 5: log10(100!) Stirling = {stirling_log10_100:.4f}")
print(f"           log10(100!) exact    = {exact_log10_100:.4f}")
print(f"           error = {abs(stirling_log10_100 - exact_log10_100):.6f}")
```

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 1. Wiley.
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 5. Academic Press.
3. **Riley, K. F. et al.** (2006). *Mathematical Methods for Physics and Engineering*, Chapter 4. Cambridge University Press.

### Online Resources
1. **MIT OCW 18.01SC**: Single Variable Calculus - Sequences and Series
2. **Paul's Online Math Notes**: Series & Sequences (https://tutorial.math.lamar.edu/)
3. **3Blue1Brown**: Taylor series visualization

### Python Tools
- `sympy.series()`: symbolic Taylor series calculation
- `scipy.special.ellipk()`: complete elliptic integral
- `scipy.special.factorial()`: factorial calculation
- `numpy.cumsum()`: partial sum calculation

---

## Next Lesson

[02. Complex Numbers](02_Complex_Numbers.md) covers algebraic operations with complex numbers, polar and exponential representations, De Moivre's theorem, and Euler's formula which is essential in physics. We will use Taylor expansion learned in series to derive $e^{ix} = \cos(x) + i\sin(x)$.

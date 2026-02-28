# Sequences and Series

## Learning Objectives

- **Determine** convergence or divergence of sequences using limit laws, the Squeeze Theorem, and the Monotone Convergence Theorem
- **Apply** convergence tests (comparison, ratio, root, integral, alternating series) to determine whether an infinite series converges
- **Find** the radius and interval of convergence for power series
- **Derive** Taylor and Maclaurin series for common functions and bound the approximation error
- **Implement** Taylor polynomial approximations in Python and visualize convergence behavior

## Introduction

Can you add up infinitely many numbers and get a finite result? The answer, surprisingly, is often yes. Consider Zeno's paradox: to walk across a room, you must first cross half the room, then half of what remains, then half again, and so on. The distances form the series:

$$\frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{16} + \cdots = 1$$

This infinite sum equals exactly 1. The theory of sequences and series makes this rigorous and provides tools to determine when infinite sums converge, how quickly they converge, and how to use them to represent functions.

Series are not just theoretical curiosities. Taylor series underpin how computers evaluate $\sin$, $\cos$, $e^x$, and $\ln$ internally. Power series solve differential equations that resist closed-form solutions. Fourier series decompose signals into frequencies. This lesson builds the foundations for all of these.

> **Cross-reference:** The [Mathematical Methods](../Mathematical_Methods/00_Overview.md) topic (Lesson 01) covers infinite series and convergence from a physicist's perspective, with emphasis on asymptotic analysis and advanced summation techniques.

## Sequences

A **sequence** is an ordered list of numbers: $a_1, a_2, a_3, \ldots$ or equivalently a function $a : \mathbb{N} \to \mathbb{R}$.

### Convergence of Sequences

A sequence $\{a_n\}$ **converges** to limit $L$ if for every $\varepsilon > 0$, there exists an integer $N$ such that:

$$n > N \implies |a_n - L| < \varepsilon$$

We write $\lim_{n \to \infty} a_n = L$. If no such $L$ exists, the sequence **diverges**.

### Useful Sequence Limits

| Sequence $a_n$ | Limit | Condition |
|-----------------|-------|-----------|
| $\frac{1}{n^p}$ | $0$ | $p > 0$ |
| $r^n$ | $0$ | $\|r\| < 1$ |
| $n^{1/n}$ | $1$ | -- |
| $\left(1 + \frac{1}{n}\right)^n$ | $e$ | Definition of $e$ |
| $\frac{n!}{n^n}$ | $0$ | Stirling's approximation |
| $\frac{\ln n}{n}$ | $0$ | Logarithm grows slower than linear |

### Monotone Convergence Theorem

If a sequence is **monotone** (always increasing or always decreasing) **and bounded**, then it converges.

This is powerful because it guarantees convergence without requiring us to find the limit. For example, the sequence $a_n = \left(1 + \frac{1}{n}\right)^n$ is increasing and bounded above by 3, so it converges. Its limit turns out to be $e$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize convergence of several sequences
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
n = np.arange(1, 51)

# (1 + 1/n)^n -> e
a1 = (1 + 1/n)**n
axes[0, 0].stem(n, a1, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].axhline(y=np.e, color='red', linestyle='--', label=f'$e \\approx {np.e:.4f}$')
axes[0, 0].set_title('$(1 + 1/n)^n \\to e$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# n^(1/n) -> 1
a2 = n**(1/n)
axes[0, 1].stem(n, a2, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 1].axhline(y=1, color='red', linestyle='--', label='Limit = 1')
axes[0, 1].set_title('$n^{1/n} \\to 1$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (-1)^n / n -> 0 (oscillating but converging)
a3 = (-1)**n / n
axes[1, 0].stem(n, a3, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 0].axhline(y=0, color='red', linestyle='--', label='Limit = 0')
axes[1, 0].set_title('$(-1)^n / n \\to 0$')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# (-1)^n diverges (oscillates without settling)
a4 = (-1.0)**n
axes[1, 1].stem(n, a4, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 1].set_title('$(-1)^n$ diverges (oscillates)')
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('$n$')

plt.tight_layout()
plt.savefig('sequence_convergence.png', dpi=150)
plt.show()
```

## Infinite Series

An **infinite series** is the sum of the terms of a sequence:

$$\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots$$

The **partial sums** are $S_N = \sum_{n=1}^{N} a_n$. The series converges if the sequence of partial sums $\{S_N\}$ converges:

$$\sum_{n=1}^{\infty} a_n = \lim_{N \to \infty} S_N$$

### Geometric Series

The most fundamental series:

$$\sum_{n=0}^{\infty} r^n = \frac{1}{1 - r}, \quad |r| < 1$$

**Derivation:** $S_N = 1 + r + r^2 + \cdots + r^N = \frac{1 - r^{N+1}}{1 - r}$. As $N \to \infty$, $r^{N+1} \to 0$ when $|r| < 1$.

More generally: $\sum_{n=0}^{\infty} ar^n = \frac{a}{1-r}$ for $|r| < 1$.

### Harmonic Series

$$\sum_{n=1}^{\infty} \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \cdots = \infty$$

Despite $a_n \to 0$, this series diverges! The terms shrink too slowly. This is a crucial warning: $a_n \to 0$ is **necessary** but not **sufficient** for convergence.

### The Divergence Test (nth Term Test)

If $\lim_{n \to \infty} a_n \neq 0$, then $\sum a_n$ diverges.

**Contrapositive:** If $\sum a_n$ converges, then $a_n \to 0$. But again, $a_n \to 0$ does NOT guarantee convergence (harmonic series is the counterexample).

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare partial sums: geometric (converges) vs harmonic (diverges)
N = 100
n = np.arange(1, N + 1)

# Geometric series: sum of (1/2)^n
geometric_partial = np.cumsum(0.5**n)

# Harmonic series: sum of 1/n
harmonic_partial = np.cumsum(1.0 / n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(n, geometric_partial, 'b-', linewidth=2)
ax1.axhline(y=1.0, color='red', linestyle='--',
            label='Limit = 1')
ax1.set_xlabel('$N$ (number of terms)')
ax1.set_ylabel('Partial sum $S_N$')
ax1.set_title('Geometric Series $\\sum (1/2)^n$: Converges to 1')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(n, harmonic_partial, 'r-', linewidth=2)
ax2.set_xlabel('$N$ (number of terms)')
ax2.set_ylabel('Partial sum $S_N$')
ax2.set_title('Harmonic Series $\\sum 1/n$: Diverges (slowly!)')
ax2.grid(True, alpha=0.3)
# Note: after 100 terms, the harmonic series is only about 5.2.
# It takes about e^(10^6) terms to reach a partial sum of 10^6.
# This is the slowest possible divergence.

plt.tight_layout()
plt.savefig('series_convergence_comparison.png', dpi=150)
plt.show()
```

## Convergence Tests

### Comparison Test

If $0 \leq a_n \leq b_n$ for all $n$:
- If $\sum b_n$ converges, then $\sum a_n$ converges (bounded by something finite)
- If $\sum a_n$ diverges, then $\sum b_n$ diverges (bigger than something infinite)

**Example:** $\sum \frac{1}{n^2 + 1}$ converges because $\frac{1}{n^2+1} < \frac{1}{n^2}$ and $\sum \frac{1}{n^2}$ converges (it is a $p$-series with $p = 2 > 1$).

### Limit Comparison Test

If $a_n, b_n > 0$ and $\lim_{n \to \infty} \frac{a_n}{b_n} = L$ where $0 < L < \infty$, then $\sum a_n$ and $\sum b_n$ either both converge or both diverge.

**Intuition:** If $a_n/b_n \to L$, then for large $n$, $a_n \approx L \cdot b_n$, so the series behave the same way.

### Ratio Test

$$L = \lim_{n \to \infty} \left|\frac{a_{n+1}}{a_n}\right|$$

- $L < 1$: series converges (absolutely)
- $L > 1$: series diverges
- $L = 1$: test is inconclusive

**Best for:** Series involving factorials, exponentials, or products. The ratio test is the workhorse of convergence testing.

### Root Test

$$L = \lim_{n \to \infty} \sqrt[n]{|a_n|}$$

Same conclusion as the ratio test. Useful when $a_n$ has the form $(...)^n$.

### Integral Test

If $f(x)$ is positive, continuous, and decreasing on $[1, \infty)$ with $f(n) = a_n$, then:

$$\sum_{n=1}^{\infty} a_n \text{ converges} \iff \int_1^{\infty} f(x) \, dx \text{ converges}$$

**Application:** The $p$-series $\sum \frac{1}{n^p}$ converges iff $p > 1$ (matches the $p$-test for improper integrals).

### Alternating Series Test

If $\{b_n\}$ is positive, decreasing, and $b_n \to 0$, then the alternating series:

$$\sum_{n=1}^{\infty} (-1)^{n+1} b_n = b_1 - b_2 + b_3 - b_4 + \cdots$$

converges. Moreover, the error after $N$ terms is bounded by $|R_N| \leq b_{N+1}$.

**Example:** The alternating harmonic series $\sum \frac{(-1)^{n+1}}{n} = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots = \ln 2$.

```python
import numpy as np
import sympy as sp

n = sp.Symbol('n', positive=True, integer=True)

# Demonstrate the ratio test on several series
print("=== Ratio Test Examples ===\n")

# 1. sum of n! / n^n
a_n = sp.factorial(n) / n**n
ratio = sp.simplify(a_n.subs(n, n+1) / a_n)
L = sp.limit(ratio, n, sp.oo)
print(f"Series: sum n!/n^n")
print(f"  Ratio: a_(n+1)/a_n simplified -> limit = {L}")
print(f"  L = {float(L):.4f} < 1, so series CONVERGES\n")

# 2. sum of 2^n / n!
a_n = 2**n / sp.factorial(n)
ratio = sp.simplify(a_n.subs(n, n+1) / a_n)
L = sp.limit(ratio, n, sp.oo)
print(f"Series: sum 2^n/n!")
print(f"  Ratio limit = {L}")
print(f"  L = 0 < 1, so series CONVERGES (sum = e^2 - 1)\n")

# 3. sum of n^2 / 2^n
a_n = n**2 / 2**n
ratio = sp.simplify(a_n.subs(n, n+1) / a_n)
L = sp.limit(ratio, n, sp.oo)
print(f"Series: sum n^2/2^n")
print(f"  Ratio limit = {L}")
print(f"  L = 1/2 < 1, so series CONVERGES\n")

# Alternating harmonic series partial sums -> ln(2)
N_values = np.arange(1, 201)
partial_sums = np.cumsum([(-1)**(k+1) / k for k in N_values])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(N_values, partial_sums, 'b-', linewidth=1, alpha=0.7)
ax.axhline(y=np.log(2), color='red', linestyle='--',
           label=f'$\\ln 2 \\approx {np.log(2):.6f}$')
ax.set_xlabel('Number of terms $N$')
ax.set_ylabel('Partial sum $S_N$')
ax.set_title('Alternating Harmonic Series: $\\sum (-1)^{n+1}/n = \\ln 2$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alternating_harmonic.png', dpi=150)
plt.show()
```

### Convergence Test Decision Guide

```
Is a_n -> 0?
  NO  --> Series DIVERGES (divergence test)
  YES --> Does NOT guarantee convergence; apply further tests:

Is it a geometric series (a*r^n)?
  YES --> Converges iff |r| < 1

Is it a p-series (1/n^p)?
  YES --> Converges iff p > 1

Does it involve factorials or exponentials?
  YES --> Try the RATIO TEST

Does a_n have the form (...)^n?
  YES --> Try the ROOT TEST

Can you bound a_n by a known convergent/divergent series?
  YES --> COMPARISON TEST or LIMIT COMPARISON

Is it alternating with decreasing terms?
  YES --> ALTERNATING SERIES TEST

Can you integrate f(x) where f(n) = a_n?
  YES --> INTEGRAL TEST

None work clearly?
  --> Try rewriting a_n, partial fractions, or direct partial sum computation
```

## Power Series

A **power series** centered at $a$ is:

$$\sum_{n=0}^{\infty} c_n (x - a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots$$

This is a "function as an infinite polynomial." It converges for some values of $x$ and diverges for others.

### Radius of Convergence

Every power series has a **radius of convergence** $R$ such that:
- The series converges absolutely for $|x - a| < R$
- The series diverges for $|x - a| > R$
- At $|x - a| = R$ (the boundary), convergence must be checked separately

The radius is found using the ratio test:

$$R = \lim_{n \to \infty} \left|\frac{c_n}{c_{n+1}}\right| \quad \text{or equivalently} \quad \frac{1}{R} = \lim_{n \to \infty} \left|\frac{c_{n+1}}{c_n}\right|$$

**Example:** For $\sum \frac{x^n}{n!}$: $\frac{1}{R} = \lim \frac{n!}{(n+1)!} = \lim \frac{1}{n+1} = 0$, so $R = \infty$. This series converges for all $x$ (it represents $e^x$).

**Example:** For $\sum n! \, x^n$: $\frac{1}{R} = \lim \frac{(n+1)!}{n!} = \lim (n+1) = \infty$, so $R = 0$. This series converges only at $x = 0$.

## Taylor and Maclaurin Series

### Definition

The **Taylor series** of $f(x)$ centered at $a$ is:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n$$

When $a = 0$, this is called a **Maclaurin series**.

**Key insight:** The Taylor series is constructed so that the $n$th partial sum (the Taylor polynomial $T_n$) matches $f$ and its first $n$ derivatives at $x = a$. Each additional term captures finer local behavior.

### Important Maclaurin Series

These should be memorized -- they appear throughout mathematics and science:

| Function | Maclaurin Series | Radius $R$ |
|----------|------------------|------------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$ | $\infty$ |
| $\sin x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$ | $\infty$ |
| $\cos x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$ | $\infty$ |
| $\frac{1}{1-x}$ | $\sum_{n=0}^{\infty} x^n = 1 + x + x^2 + x^3 + \cdots$ | $1$ |
| $\ln(1+x)$ | $\sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$ | $1$ |
| $(1+x)^\alpha$ | $\sum_{n=0}^{\infty} \binom{\alpha}{n} x^n$ (binomial series) | $1$ |
| $\arctan x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{2n+1} = x - \frac{x^3}{3} + \frac{x^5}{5} - \cdots$ | $1$ |

### Taylor Remainder Theorem

The error of the $n$th-degree Taylor polynomial is:

$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!} (x - a)^{n+1}$$

for some $c$ between $a$ and $x$. This is called the **Lagrange remainder**. It gives us an upper bound on the approximation error:

$$|R_n(x)| \leq \frac{M}{(n+1)!} |x - a|^{n+1}$$

where $M = \max |f^{(n+1)}(t)|$ for $t$ between $a$ and $x$.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def taylor_polynomial(f, x, a, n):
    """
    Compute the nth-degree Taylor polynomial of f centered at a.

    This builds the polynomial term by term, each term adding
    more accuracy near x = a. The factorial in the denominator
    ensures the derivatives match.
    """
    poly = 0
    for k in range(n + 1):
        coeff = sp.diff(f, x, k).subs(x, a) / sp.factorial(k)
        poly += coeff * (x - a)**k
    return poly

x = sp.Symbol('x')
f = sp.sin(x)

# Compute Taylor polynomials of sin(x) at a=0 for various degrees
print("Taylor polynomials of sin(x) centered at 0:")
for degree in [1, 3, 5, 7, 9]:
    T = taylor_polynomial(f, x, 0, degree)
    print(f"  T_{degree}(x) = {T}")

# Visualization: Taylor polynomials converging to sin(x)
x_vals = np.linspace(-2*np.pi, 2*np.pi, 500)

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(x_vals, np.sin(x_vals), 'k-', linewidth=3, label='$\\sin(x)$')

colors = ['red', 'orange', 'green', 'blue', 'purple']
degrees = [1, 3, 5, 7, 9]

for degree, color in zip(degrees, colors):
    T = taylor_polynomial(f, x, 0, degree)
    T_func = sp.lambdify(x, T, 'numpy')
    y_taylor = T_func(x_vals)
    # Clip extreme values for visualization (Taylor polynomials diverge far from center)
    y_taylor = np.clip(y_taylor, -3, 3)
    ax.plot(x_vals, y_taylor, '--', color=color, linewidth=1.5,
            label=f'$T_{{{degree}}}(x)$')

ax.set_ylim(-3, 3)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Taylor Polynomials of $\\sin(x)$ at $a = 0$')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('taylor_sin.png', dpi=150)
plt.show()
```

### How Computers Calculate sin(x)

Modern CPUs and math libraries use Taylor-like polynomial approximations (often Chebyshev or minimax polynomials) to compute trigonometric functions. The idea is:

1. **Reduce** the argument to a small interval (e.g., $[0, \pi/4]$) using symmetry
2. **Approximate** using a polynomial of degree 7-13 (enough for double precision)
3. The polynomial is pre-computed to minimize the maximum error

```python
import numpy as np

def sin_taylor(x, n_terms=10):
    """
    Approximate sin(x) using its Maclaurin series.

    This demonstrates the principle behind how computers evaluate
    trigonometric functions, though real implementations use
    optimized polynomial approximations (minimax/Chebyshev).
    """
    # First, reduce x to [-pi, pi] using periodicity
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi

    result = 0.0
    for k in range(n_terms):
        # Each term: (-1)^k * x^(2k+1) / (2k+1)!
        term = ((-1)**k * x**(2*k + 1)) / np.math.factorial(2*k + 1)
        result += term
    return result

# Test accuracy
test_values = [0.1, 0.5, 1.0, np.pi/4, np.pi/2, np.pi, 3.0, 10.0]
print(f"{'x':>8} {'Taylor (10 terms)':>20} {'np.sin(x)':>15} {'Error':>12}")
print("-" * 58)
for x in test_values:
    approx = sin_taylor(x, n_terms=10)
    exact = np.sin(x)
    print(f"{x:>8.4f} {approx:>20.15f} {exact:>15.15f} {abs(approx-exact):>12.2e}")
```

### Taylor Series for Computing e

```python
import numpy as np

def compute_e(n_terms=20):
    """
    Compute e using its Maclaurin series: e = sum_{n=0}^{inf} 1/n!

    The factorial in the denominator makes this series converge
    extremely fast. With just 20 terms, we get 18 digits of accuracy.
    """
    e_approx = 0.0
    for n in range(n_terms):
        term = 1.0 / np.math.factorial(n)
        e_approx += term
        if n < 12:
            print(f"  n={n:>2d}: term = 1/{n}! = {term:.15f}, "
                  f"partial sum = {e_approx:.15f}")
    return e_approx

print("Computing e via Taylor series:")
result = compute_e(20)
print(f"\nResult:  {result:.18f}")
print(f"np.e:    {np.e:.18f}")
print(f"Error:   {abs(result - np.e):.2e}")
```

## Summary

- **Sequences** converge if their terms approach a finite limit; the Monotone Convergence Theorem guarantees convergence for bounded monotone sequences
- **Series** are sums of sequence terms; convergence means the partial sums approach a finite limit
- The **divergence test** is a quick first check: if $a_n \not\to 0$, the series diverges
- **Convergence tests** (comparison, ratio, root, integral, alternating) each have their ideal use cases -- the ratio test is most versatile
- **Power series** $\sum c_n(x-a)^n$ converge inside a disk of radius $R$ centered at $a$
- **Taylor series** represent smooth functions as infinite polynomials; the Taylor remainder bounds approximation error
- Key series to memorize: $e^x$, $\sin x$, $\cos x$, $\frac{1}{1-x}$, $\ln(1+x)$
- Computers use polynomial approximations (descendants of Taylor series) to evaluate transcendental functions

## Practice Problems

### Problem 1: Sequence Convergence

Determine whether each sequence converges or diverges. If it converges, find the limit.

(a) $a_n = \frac{n^2 + 3n}{2n^2 - 1}$

(b) $a_n = \frac{(-1)^n n}{n + 1}$

(c) $a_n = \left(1 + \frac{3}{n}\right)^n$

### Problem 2: Series Convergence Tests

Determine whether each series converges or diverges, stating which test you used.

(a) $\sum_{n=1}^{\infty} \frac{n^2}{3^n}$ (ratio test)

(b) $\sum_{n=2}^{\infty} \frac{1}{n \ln n}$ (integral test)

(c) $\sum_{n=1}^{\infty} \frac{(-1)^n}{\sqrt{n}}$ (alternating series test)

(d) $\sum_{n=1}^{\infty} \frac{n!}{n^n}$ (ratio test)

### Problem 3: Radius of Convergence

Find the radius and interval of convergence for each power series:

(a) $\sum_{n=0}^{\infty} \frac{(x-3)^n}{n \cdot 2^n}$

(b) $\sum_{n=0}^{\infty} \frac{n! \, x^n}{n^n}$

### Problem 4: Taylor Series Derivation

(a) Derive the Maclaurin series for $f(x) = \frac{1}{1+x^2}$ by substituting $-x^2$ into the geometric series $\frac{1}{1-u} = \sum u^n$.

(b) Integrate the result term by term to obtain the series for $\arctan x$.

(c) Use $\arctan(1) = \pi/4$ to derive the Leibniz formula: $\frac{\pi}{4} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots$

(d) Write Python code to compute $\pi$ using this series. How many terms are needed for 6-digit accuracy?

### Problem 5: Taylor Polynomial Error Bound

Use the Taylor remainder theorem to determine the degree $n$ needed so that the Maclaurin polynomial $T_n(x)$ for $e^x$ approximates $e^{0.5}$ with error less than $10^{-8}$.

Then verify computationally: compute $T_n(0.5)$ for increasing $n$ and compare with the exact value of $e^{0.5}$.

## References

- Stewart, *Calculus: Early Transcendentals*, Ch. 11 (Infinite Sequences and Series)
- [3Blue1Brown: Taylor Series](https://www.youtube.com/watch?v=3d6DsjIBzJ4)
- [Paul's Online Notes: Series and Sequences](https://tutorial.math.lamar.edu/Classes/CalcII/SeriesIntro.aspx)
- See also: [Mathematical Methods L01](../Mathematical_Methods/01_Infinite_Series.md) for advanced convergence topics

---

[Previous: Applications of Integration](./06_Applications_of_Integration.md) | [Next: Parametric Curves and Polar Coordinates](./08_Parametric_and_Polar.md)

# Integration Fundamentals

## Learning Objectives

- **Approximate** definite integrals using Riemann sums (left, right, midpoint, trapezoidal) and explain how finer partitions improve accuracy
- **State** both parts of the Fundamental Theorem of Calculus and explain why they connect differentiation and integration
- **Compute** antiderivatives for polynomial, exponential, trigonometric, and basic rational functions
- **Evaluate** definite integrals using the FTC and interpret the result as signed area
- **Compare** numerical integration methods in Python (Riemann sums, trapezoidal rule, Simpson's rule, `scipy.integrate.quad`)

## Introduction

If the derivative answers "how fast is it changing?", the integral answers "how much has accumulated?" A car's speedometer gives velocity (derivative of position); the odometer gives total distance (integral of velocity). A rain gauge measures the total rainfall (integral of the rain rate). Your bank balance is the integral of deposits and withdrawals over time.

Integration is the inverse of differentiation, but it is also much more: it is a way to add up infinitely many infinitesimal pieces to compute areas, volumes, averages, probabilities, and physical quantities like work and energy.

## Riemann Sums: Building Intuition

The key idea is to approximate the area under a curve by slicing it into thin rectangles.

Given a function $f(x)$ on $[a, b]$, divide the interval into $n$ equal subintervals of width $\Delta x = \frac{b - a}{n}$.

### Types of Riemann Sums

| Method | Sample Point | Formula |
|--------|-------------|---------|
| **Left** | Left endpoint $x_i = a + i \cdot \Delta x$ | $L_n = \sum_{i=0}^{n-1} f(x_i) \Delta x$ |
| **Right** | Right endpoint $x_{i+1} = a + (i+1) \cdot \Delta x$ | $R_n = \sum_{i=1}^{n} f(x_i) \Delta x$ |
| **Midpoint** | Midpoint $\bar{x}_i = a + (i + \frac{1}{2}) \Delta x$ | $M_n = \sum_{i=0}^{n-1} f(\bar{x}_i) \Delta x$ |
| **Trapezoidal** | Average of left and right | $T_n = \frac{L_n + R_n}{2}$ |

As $n \to \infty$ (rectangles get thinner), all four methods converge to the same value: the **definite integral**.

```python
import numpy as np
import matplotlib.pyplot as plt

def riemann_sum_visualization(f, a, b, n=10, method='left'):
    """
    Visualize a Riemann sum approximation.

    The rectangles show how we approximate the curved area with
    flat-topped boxes. More rectangles = better approximation.
    """
    x = np.linspace(a, b, 1000)
    dx = (b - a) / n

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the actual function
    ax.plot(x, f(x), 'b-', linewidth=2, label='$f(x)$', zorder=3)

    total = 0
    for i in range(n):
        xi = a + i * dx
        if method == 'left':
            height = f(xi)
        elif method == 'right':
            height = f(xi + dx)
        elif method == 'midpoint':
            height = f(xi + dx/2)

        total += height * dx

        # Draw rectangle
        rect = plt.Rectangle((xi, 0), dx, height,
                              facecolor='skyblue', edgecolor='navy',
                              alpha=0.5, linewidth=1)
        ax.add_patch(rect)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title(f'{method.capitalize()} Riemann Sum (n={n}): '
                 f'$\\sum \\approx {total:.4f}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'riemann_{method}_{n}.png', dpi=150)
    plt.show()
    return total

# Example: integral of x^2 from 0 to 1 (exact answer: 1/3)
f = lambda x: x**2
for n in [5, 10, 50]:
    approx = riemann_sum_visualization(f, 0, 1, n, method='midpoint')
    print(f"  n={n}: approximation = {approx:.6f}, error = {abs(approx - 1/3):.6f}")
```

### Convergence of Riemann Sums

```python
import numpy as np

def compare_riemann_methods(f, a, b, exact, n_values):
    """
    Compare how quickly different Riemann sum methods converge.

    This demonstrates that midpoint and trapezoidal converge faster
    (O(1/n^2)) than left and right (O(1/n)). Simpson's rule, which
    combines them, converges even faster (O(1/n^4)).
    """
    print(f"{'n':>8} {'Left':>12} {'Right':>12} {'Midpoint':>12} "
          f"{'Trapezoid':>12} {'Simpson':>12}")
    print("-" * 72)

    for n in n_values:
        dx = (b - a) / n
        x_left = np.linspace(a, b - dx, n)
        x_right = np.linspace(a + dx, b, n)
        x_mid = np.linspace(a + dx/2, b - dx/2, n)

        L = np.sum(f(x_left)) * dx
        R = np.sum(f(x_right)) * dx
        M = np.sum(f(x_mid)) * dx
        T = (L + R) / 2
        S = (2*M + T) / 3  # Simpson's rule combines midpoint and trapezoidal

        print(f"{n:>8d} {abs(L-exact):>12.2e} {abs(R-exact):>12.2e} "
              f"{abs(M-exact):>12.2e} {abs(T-exact):>12.2e} {abs(S-exact):>12.2e}")

# Integral of sin(x) from 0 to pi, exact answer = 2
f = lambda x: np.sin(x)
compare_riemann_methods(f, 0, np.pi, exact=2.0,
                        n_values=[10, 50, 100, 500, 1000, 5000])
```

## The Definite Integral

As $n \to \infty$, the Riemann sum becomes the **definite integral**:

$$\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta x$$

**Notation explained:**
- $\int$ : elongated S for "sum" (Leibniz's notation)
- $a, b$: lower and upper limits of integration
- $f(x)$: the integrand (the function being integrated)
- $dx$: an infinitesimally small width (the limit of $\Delta x$)

**Interpretation:** The definite integral gives the **signed area** between the curve and the $x$-axis:
- Area above the $x$-axis counts as **positive**
- Area below the $x$-axis counts as **negative**

### Properties of Definite Integrals

| Property | Statement |
|----------|-----------|
| **Linearity** | $\int_a^b [cf(x) + dg(x)] \, dx = c\int_a^b f(x)\,dx + d\int_a^b g(x)\,dx$ |
| **Additivity** | $\int_a^b f(x)\,dx + \int_b^c f(x)\,dx = \int_a^c f(x)\,dx$ |
| **Reversal** | $\int_a^b f(x)\,dx = -\int_b^a f(x)\,dx$ |
| **Comparison** | If $f(x) \geq g(x)$ on $[a,b]$, then $\int_a^b f\,dx \geq \int_a^b g\,dx$ |
| **Zero width** | $\int_a^a f(x)\,dx = 0$ |

## Antiderivatives and Indefinite Integrals

An **antiderivative** of $f(x)$ is any function $F(x)$ such that $F'(x) = f(x)$.

The **indefinite integral** collects all antiderivatives:

$$\int f(x) \, dx = F(x) + C$$

where $C$ is the **constant of integration** -- a reminder that any constant vanishes under differentiation.

### Basic Antiderivative Table

| $f(x)$ | $\int f(x)\,dx$ |
|---------|-----------------|
| $x^n$ ($n \neq -1$) | $\frac{x^{n+1}}{n+1} + C$ |
| $x^{-1} = 1/x$ | $\ln|x| + C$ |
| $e^x$ | $e^x + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |
| $\sec^2 x$ | $\tan x + C$ |
| $\frac{1}{1+x^2}$ | $\arctan x + C$ |
| $\frac{1}{\sqrt{1-x^2}}$ | $\arcsin x + C$ |

```python
import sympy as sp

x = sp.Symbol('x')

# SymPy computes antiderivatives (indefinite integrals)
expressions = [x**3, sp.sin(x), sp.exp(x), 1/(1 + x**2), 1/x]

for expr in expressions:
    antideriv = sp.integrate(expr, x)
    # Verify by differentiating the result
    check = sp.diff(antideriv, x)
    print(f"integral of {expr} = {antideriv}")
    print(f"  Verification: d/dx[{antideriv}] = {sp.simplify(check)}\n")
```

## The Fundamental Theorem of Calculus

This is one of the most important theorems in all of mathematics. It reveals that differentiation and integration are inverse operations.

### Part 1 (Differentiation of an Integral)

If $f$ is continuous on $[a, b]$ and we define:

$$F(x) = \int_a^x f(t) \, dt$$

then $F'(x) = f(x)$.

**In words:** The derivative of an accumulation function is the original function. If $f(t)$ is a rate (say, water flowing into a tank at rate $f(t)$ liters/hour), then $F(x) = \int_a^x f(t)\,dt$ is the total water accumulated from time $a$ to time $x$, and the rate of accumulation at time $x$ is exactly $f(x)$.

### Part 2 (Evaluation of a Definite Integral)

If $f$ is continuous on $[a, b]$ and $F$ is any antiderivative of $f$ (i.e., $F' = f$), then:

$$\int_a^b f(x) \, dx = F(b) - F(a)$$

**This is revolutionary:** Instead of computing a limit of Riemann sums (which requires infinite subdivision), we simply find an antiderivative and evaluate at the endpoints. A sum of infinitely many infinitesimal pieces reduces to simple subtraction.

**Notation:** We write $F(x) \Big|_a^b = F(b) - F(a)$.

**Example:**

$$\int_0^{\pi} \sin x \, dx = [-\cos x]_0^{\pi} = -\cos(\pi) - (-\cos(0)) = -(-1) + 1 = 2$$

This is the total area under one arch of the sine curve.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Visualize FTC Part 1: the accumulation function
x_sym = sp.Symbol('x')
t_sym = sp.Symbol('t')

# f(t) = sin(t), accumulation F(x) = integral from 0 to x of sin(t) dt
f_expr = sp.sin(t_sym)
F_expr = sp.integrate(f_expr, (t_sym, 0, x_sym))  # = 1 - cos(x)
F_prime = sp.diff(F_expr, x_sym)  # Should equal sin(x)

print(f"f(t) = {f_expr}")
print(f"F(x) = integral_0^x sin(t) dt = {F_expr}")
print(f"F'(x) = {F_prime}")
print(f"FTC Part 1 verified: F'(x) = f(x) = sin(x)")

# Visualize both f and F
x_vals = np.linspace(0, 4*np.pi, 500)
f_vals = np.sin(x_vals)
F_vals = 1 - np.cos(x_vals)  # The accumulation function

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Original function
ax1.plot(x_vals, f_vals, 'b-', linewidth=2)
ax1.fill_between(x_vals, f_vals, 0, where=(f_vals >= 0),
                  alpha=0.3, color='green', label='Positive area')
ax1.fill_between(x_vals, f_vals, 0, where=(f_vals < 0),
                  alpha=0.3, color='red', label='Negative area')
ax1.set_ylabel('$f(t) = \\sin(t)$')
ax1.set_title('Integrand and Accumulation Function (FTC Part 1)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accumulation function
ax2.plot(x_vals, F_vals, 'r-', linewidth=2,
         label='$F(x) = \\int_0^x \\sin(t)\\,dt = 1 - \\cos(x)$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$F(x)$')
ax2.legend()
ax2.grid(True, alpha=0.3)
# Notice: F increases when f > 0, decreases when f < 0, and
# has extrema where f crosses zero. This is FTC Part 1 in action.

plt.tight_layout()
plt.savefig('ftc_accumulation.png', dpi=150)
plt.show()
```

## Numerical Integration with SciPy

For functions without closed-form antiderivatives, numerical integration is essential.

```python
import numpy as np
from scipy import integrate

# Example: integral of e^(-x^2) from 0 to infinity
# This is related to the Gaussian integral: sqrt(pi)/2

# scipy.integrate.quad: adaptive quadrature (gold standard)
result, error = integrate.quad(lambda x: np.exp(-x**2), 0, np.inf)
print(f"integral of exp(-x^2) from 0 to inf:")
print(f"  Numerical: {result:.15f}")
print(f"  Exact:     {np.sqrt(np.pi)/2:.15f}")
print(f"  Estimated error: {error:.2e}")

# Compare methods on a simpler integral: integral of sin(x) from 0 to pi
f = lambda x: np.sin(x)
exact = 2.0

# Trapezoidal rule (numpy)
for n in [10, 100, 1000]:
    x = np.linspace(0, np.pi, n+1)
    trap = np.trapz(f(x), x)
    print(f"\n  Trapezoidal (n={n:>4d}): {trap:.12f}, error = {abs(trap-exact):.2e}")

# Simpson's rule (scipy)
for n in [10, 100, 1000]:
    x = np.linspace(0, np.pi, n+1)
    simp = integrate.simpson(f(x), x=x)
    print(f"  Simpson's  (n={n:>4d}): {simp:.12f}, error = {abs(simp-exact):.2e}")

# Adaptive quadrature (scipy) -- usually the best choice
quad_result, quad_error = integrate.quad(f, 0, np.pi)
print(f"\n  Adaptive quad:       {quad_result:.15f}, est. error = {quad_error:.2e}")
```

## Summary

- **Riemann sums** approximate integrals by partitioning the domain into rectangles; finer partitions yield better approximations
- The **definite integral** $\int_a^b f(x)\,dx$ is the limit of Riemann sums and represents signed area
- **Antiderivatives** are the reverse of derivatives: $F'(x) = f(x) \implies \int f(x)\,dx = F(x) + C$
- **FTC Part 1**: The derivative of an accumulation function $\int_a^x f(t)\,dt$ equals $f(x)$
- **FTC Part 2**: $\int_a^b f(x)\,dx = F(b) - F(a)$ -- evaluation replaces infinite summation
- For functions without closed-form antiderivatives, **numerical methods** (trapezoidal, Simpson's, adaptive quadrature) provide accurate approximations

## Practice Problems

### Problem 1: Riemann Sum Computation

Compute the left, right, and midpoint Riemann sums for $\int_0^2 x^3 \, dx$ with $n = 4$ subintervals. Compare each with the exact value. Which method is closest?

### Problem 2: FTC Application

Use the Fundamental Theorem of Calculus to evaluate:

(a) $\int_1^4 (3\sqrt{x} - 1/x) \, dx$

(b) $\int_0^{\pi/4} \sec^2 \theta \, d\theta$

(c) $\frac{d}{dx} \int_0^{x^2} \sin(t^2) \, dt$ (Hint: apply the chain rule with FTC Part 1)

### Problem 3: Signed Area Interpretation

Compute $\int_0^{2\pi} \sin x \, dx$. Explain why the result is 0, even though the sine function clearly has area under it. Then compute the *total unsigned area* $\int_0^{2\pi} |\sin x| \, dx$.

### Problem 4: Numerical Integration Comparison

Write Python code to approximate $\int_0^1 e^{-x^2} dx$ using:
- Left Riemann sum with $n = 1000$
- Trapezoidal rule with $n = 1000$
- Simpson's rule with $n = 1000$
- `scipy.integrate.quad`

Compare all four results. Which method gives the most accurate answer with the fewest function evaluations?

### Problem 5: Proving the FTC

Given that FTC Part 1 says $\frac{d}{dx}\int_a^x f(t)\,dt = f(x)$, prove FTC Part 2: $\int_a^b f(x)\,dx = F(b) - F(a)$ where $F' = f$.

(Hint: Define $G(x) = \int_a^x f(t)\,dt$. By Part 1, $G'(x) = f(x) = F'(x)$. What can you conclude about $G(x) - F(x)$?)

## References

- Stewart, *Calculus: Early Transcendentals*, Ch. 5 (Integrals)
- [3Blue1Brown: Integration and the Fundamental Theorem](https://www.youtube.com/watch?v=rfG8ce4nNh0)
- [Paul's Online Notes: Integrals](https://tutorial.math.lamar.edu/Classes/CalcI/IntegralsIntro.aspx)

---

[Previous: Applications of Derivatives](./03_Applications_of_Derivatives.md) | [Next: Integration Techniques](./05_Integration_Techniques.md)

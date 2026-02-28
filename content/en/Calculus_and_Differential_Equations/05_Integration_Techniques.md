# Integration Techniques

## Learning Objectives

- **Apply** $u$-substitution to evaluate integrals involving composite functions
- **Perform** integration by parts and recognize when to apply it (LIATE rule)
- **Decompose** rational functions using partial fractions and integrate each term
- **Evaluate** trigonometric integrals using power-reduction and trigonometric substitution
- **Determine** convergence or divergence of improper integrals (Type I and Type II)

## Introduction

The Fundamental Theorem of Calculus tells us that every definite integral can be evaluated by finding an antiderivative. The catch? Finding antiderivatives is often much harder than finding derivatives. While differentiation is algorithmic (apply the rules mechanically), integration is more of an art -- it requires pattern recognition, clever substitutions, and sometimes creative tricks.

Think of it like this: differentiation is like breaking a vase (easy, systematic), while integration is like reassembling the pieces (requires insight into how they fit together).

This lesson covers the essential techniques that handle the vast majority of integrals encountered in practice.

## Substitution (u-Substitution)

### The Idea

$u$-substitution is the integration counterpart of the chain rule. If we see a composite function and its inner derivative, we can simplify.

**Chain rule (differentiation):** $\frac{d}{dx} F(g(x)) = F'(g(x)) \cdot g'(x)$

**Substitution (integration):** $\int F'(g(x)) \cdot g'(x) \, dx = F(g(x)) + C$

### The Procedure

1. **Identify** an inner function $u = g(x)$
2. **Compute** $du = g'(x) \, dx$
3. **Rewrite** the integral entirely in terms of $u$
4. **Integrate** with respect to $u$
5. **Substitute** back to $x$

### Example 1: Basic Substitution

$$\int 2x \cos(x^2) \, dx$$

Let $u = x^2$, so $du = 2x \, dx$. The integral becomes:

$$\int \cos(u) \, du = \sin(u) + C = \sin(x^2) + C$$

### Example 2: Adjusting Constants

$$\int x e^{3x^2} \, dx$$

Let $u = 3x^2$, so $du = 6x \, dx$, which means $x \, dx = \frac{du}{6}$.

$$\int e^u \cdot \frac{du}{6} = \frac{1}{6} e^u + C = \frac{1}{6} e^{3x^2} + C$$

### Example 3: Definite Integral with Substitution

For definite integrals, change the limits along with the variable:

$$\int_0^1 x \sqrt{1 - x^2} \, dx$$

Let $u = 1 - x^2$, $du = -2x \, dx$. When $x = 0$: $u = 1$. When $x = 1$: $u = 0$.

$$\int_1^0 \sqrt{u} \cdot \left(-\frac{du}{2}\right) = \frac{1}{2} \int_0^1 u^{1/2} \, du = \frac{1}{2} \cdot \frac{2}{3} u^{3/2} \Big|_0^1 = \frac{1}{3}$$

```python
import sympy as sp

x = sp.Symbol('x')

# SymPy handles substitution automatically
integrals = [
    2*x * sp.cos(x**2),
    x * sp.exp(3*x**2),
    x * sp.sqrt(1 - x**2),
]

for expr in integrals:
    result = sp.integrate(expr, x)
    print(f"integral of {expr} dx = {result}")

# Definite integral with substitution
print(f"\nintegral_0^1 x*sqrt(1-x^2) dx = "
      f"{sp.integrate(x * sp.sqrt(1 - x**2), (x, 0, 1))}")
```

## Integration by Parts

### The Formula

Integration by parts is the integration counterpart of the product rule:

$$\int u \, dv = uv - \int v \, du$$

**Derivation:** Start from the product rule $\frac{d}{dx}(uv) = u\frac{dv}{dx} + v\frac{du}{dx}$, integrate both sides, and rearrange.

### The LIATE Rule

When choosing which factor is $u$ (to differentiate) and which is $dv$ (to integrate), use the **LIATE** priority:

| Priority | Type | Examples |
|----------|------|----------|
| 1 (highest) | **L**ogarithmic | $\ln x$, $\log x$ |
| 2 | **I**nverse trig | $\arctan x$, $\arcsin x$ |
| 3 | **A**lgebraic | $x^2$, $3x + 1$ |
| 4 | **T**rigonometric | $\sin x$, $\cos x$ |
| 5 (lowest) | **E**xponential | $e^x$, $2^x$ |

Choose $u$ as the factor highest on the list (it will simplify when differentiated).

### Example 1: $\int x e^x \, dx$

Using LIATE: $u = x$ (Algebraic), $dv = e^x \, dx$ (Exponential).

Then $du = dx$, $v = e^x$.

$$\int x e^x \, dx = x e^x - \int e^x \, dx = x e^x - e^x + C = e^x(x - 1) + C$$

### Example 2: $\int x^2 \sin x \, dx$ (Apply Twice)

First application: $u = x^2$, $dv = \sin x \, dx$, so $du = 2x \, dx$, $v = -\cos x$.

$$\int x^2 \sin x \, dx = -x^2 \cos x + \int 2x \cos x \, dx$$

Second application to $\int 2x \cos x \, dx$: $u = 2x$, $dv = \cos x \, dx$.

$$= -x^2 \cos x + 2x \sin x - \int 2 \sin x \, dx = -x^2 \cos x + 2x \sin x + 2\cos x + C$$

### Example 3: $\int \ln x \, dx$ (Logarithmic)

$u = \ln x$, $dv = dx$, so $du = \frac{1}{x} dx$, $v = x$.

$$\int \ln x \, dx = x \ln x - \int x \cdot \frac{1}{x} \, dx = x \ln x - x + C$$

```python
import sympy as sp

x = sp.Symbol('x')

# Integration by parts examples
by_parts_examples = [
    (x * sp.exp(x), "x * e^x"),
    (x**2 * sp.sin(x), "x^2 * sin(x)"),
    (sp.log(x), "ln(x)"),
    (x * sp.log(x), "x * ln(x)"),
    (sp.exp(x) * sp.sin(x), "e^x * sin(x)"),  # Requires the "cycling" trick
]

for expr, name in by_parts_examples:
    result = sp.integrate(expr, x)
    print(f"integral of {name} dx = {result}")
```

## Partial Fraction Decomposition

### The Method

Any rational function $\frac{P(x)}{Q(x)}$ (where $\deg P < \deg Q$) can be decomposed into simpler fractions that are easy to integrate individually.

**Step 1:** Factor the denominator completely.
**Step 2:** Write the decomposition based on factor types:

| Factor Type | Decomposition |
|-------------|---------------|
| Linear: $(ax + b)$ | $\frac{A}{ax + b}$ |
| Repeated linear: $(ax + b)^n$ | $\frac{A_1}{ax+b} + \frac{A_2}{(ax+b)^2} + \cdots + \frac{A_n}{(ax+b)^n}$ |
| Irreducible quadratic: $(ax^2+bx+c)$ | $\frac{Ax + B}{ax^2+bx+c}$ |

**Step 3:** Solve for constants by multiplying through and equating coefficients.

### Example: $\int \frac{x+5}{x^2+x-2} \, dx$

Factor: $x^2 + x - 2 = (x+2)(x-1)$.

Decompose: $\frac{x+5}{(x+2)(x-1)} = \frac{A}{x+2} + \frac{B}{x-1}$

Multiply through: $x + 5 = A(x-1) + B(x+2)$

Set $x = 1$: $6 = 3B \implies B = 2$.
Set $x = -2$: $3 = -3A \implies A = -1$.

$$\int \frac{x+5}{x^2+x-2} \, dx = \int \left(\frac{-1}{x+2} + \frac{2}{x-1}\right) dx = -\ln|x+2| + 2\ln|x-1| + C$$

```python
import sympy as sp

x = sp.Symbol('x')

# Partial fraction decomposition
expr = (x + 5) / (x**2 + x - 2)
decomposed = sp.apart(expr, x)
print(f"Partial fractions of {expr}:")
print(f"  = {decomposed}")

# Integrate
result = sp.integrate(expr, x)
print(f"  integral = {result}")

# More complex example: repeated and irreducible factors
expr2 = (2*x**2 + 3) / (x**3 - x)
decomposed2 = sp.apart(expr2, x)
print(f"\nPartial fractions of {expr2}:")
print(f"  = {decomposed2}")
print(f"  integral = {sp.integrate(expr2, x)}")
```

## Trigonometric Integrals

### Powers of Sine and Cosine

For $\int \sin^m x \cos^n x \, dx$:

- If $m$ is odd: save one $\sin x$, convert rest to $\cos$ using $\sin^2 x = 1 - \cos^2 x$, substitute $u = \cos x$
- If $n$ is odd: save one $\cos x$, convert rest to $\sin$ using $\cos^2 x = 1 - \sin^2 x$, substitute $u = \sin x$
- If both even: use power-reduction identities:
  - $\sin^2 x = \frac{1 - \cos 2x}{2}$
  - $\cos^2 x = \frac{1 + \cos 2x}{2}$

**Example:** $\int \sin^3 x \cos^2 x \, dx$

Since $m = 3$ is odd, save one $\sin x$:

$$\int \sin^2 x \cos^2 x \sin x \, dx = \int (1 - \cos^2 x) \cos^2 x \sin x \, dx$$

Let $u = \cos x$, $du = -\sin x \, dx$:

$$-\int (1 - u^2) u^2 \, du = -\int (u^2 - u^4) \, du = -\frac{u^3}{3} + \frac{u^5}{5} + C = -\frac{\cos^3 x}{3} + \frac{\cos^5 x}{5} + C$$

### Trigonometric Substitution

For integrals involving $\sqrt{a^2 - x^2}$, $\sqrt{a^2 + x^2}$, or $\sqrt{x^2 - a^2}$:

| Expression | Substitution | Identity Used |
|------------|-------------|---------------|
| $\sqrt{a^2 - x^2}$ | $x = a\sin\theta$ | $1 - \sin^2\theta = \cos^2\theta$ |
| $\sqrt{a^2 + x^2}$ | $x = a\tan\theta$ | $1 + \tan^2\theta = \sec^2\theta$ |
| $\sqrt{x^2 - a^2}$ | $x = a\sec\theta$ | $\sec^2\theta - 1 = \tan^2\theta$ |

**Example:** $\int \frac{dx}{\sqrt{4 - x^2}}$

Let $x = 2\sin\theta$, $dx = 2\cos\theta \, d\theta$:

$$\int \frac{2\cos\theta \, d\theta}{\sqrt{4 - 4\sin^2\theta}} = \int \frac{2\cos\theta \, d\theta}{2\cos\theta} = \int d\theta = \theta + C = \arcsin\frac{x}{2} + C$$

```python
import sympy as sp

x = sp.Symbol('x')

# Trigonometric integrals
trig_examples = [
    (sp.sin(x)**3 * sp.cos(x)**2, "sin^3(x) cos^2(x)"),
    (sp.sin(x)**2 * sp.cos(x)**2, "sin^2(x) cos^2(x)"),
    (1 / sp.sqrt(4 - x**2), "1/sqrt(4 - x^2)"),
    (sp.sqrt(1 + x**2), "sqrt(1 + x^2)"),
]

for expr, name in trig_examples:
    result = sp.integrate(expr, x)
    print(f"integral of {name} dx = {sp.simplify(result)}")
```

## Improper Integrals

An integral is **improper** if either:
- **Type I:** One or both limits of integration are infinite
- **Type II:** The integrand has an infinite discontinuity within $[a, b]$

### Type I: Infinite Limits

$$\int_a^{\infty} f(x) \, dx = \lim_{t \to \infty} \int_a^t f(x) \, dx$$

If the limit exists and is finite, the integral **converges**; otherwise, it **diverges**.

**Example (Convergent):**

$$\int_1^{\infty} \frac{1}{x^2} \, dx = \lim_{t \to \infty} \left[-\frac{1}{x}\right]_1^t = \lim_{t \to \infty} \left(-\frac{1}{t} + 1\right) = 1$$

**Example (Divergent):**

$$\int_1^{\infty} \frac{1}{x} \, dx = \lim_{t \to \infty} [\ln x]_1^t = \lim_{t \to \infty} \ln t = \infty$$

### The $p$-Test for $\int_1^{\infty} \frac{1}{x^p} dx$

- Converges if $p > 1$
- Diverges if $p \leq 1$

This is one of the most frequently used convergence tests, analogous to the $p$-series test.

### Type II: Discontinuous Integrand

If $f$ has a discontinuity at $c \in [a, b]$:

$$\int_a^b f(x) \, dx = \lim_{\epsilon \to 0^+} \int_a^{c-\epsilon} f(x) \, dx + \lim_{\epsilon \to 0^+} \int_{c+\epsilon}^b f(x) \, dx$$

**Example:**

$$\int_0^1 \frac{1}{\sqrt{x}} \, dx = \lim_{\epsilon \to 0^+} \int_\epsilon^1 x^{-1/2} \, dx = \lim_{\epsilon \to 0^+} [2\sqrt{x}]_\epsilon^1 = 2 - \lim_{\epsilon \to 0^+} 2\sqrt{\epsilon} = 2$$

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol('x')

# Type I: Convergent vs divergent
print("Type I improper integrals:")
print(f"  integral_1^inf 1/x^2 dx = {sp.integrate(1/x**2, (x, 1, sp.oo))}")
print(f"  integral_1^inf 1/x dx = {sp.integrate(1/x, (x, 1, sp.oo))}")

# The Gaussian integral -- one of the most important improper integrals
print(f"\n  integral_0^inf e^(-x^2) dx = {sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))}")

# Type II: Discontinuous integrand
print(f"\nType II improper integrals:")
print(f"  integral_0^1 1/sqrt(x) dx = {sp.integrate(1/sp.sqrt(x), (x, 0, 1))}")

# Visualize convergence: partial integrals approaching the limit
t_values = np.linspace(1, 50, 200)

# 1/x^2 converges to 1
partial_convergent = 1 - 1/t_values

# 1/x diverges
partial_divergent = np.log(t_values)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(t_values, partial_convergent, 'b-', linewidth=2)
ax1.axhline(y=1, color='red', linestyle='--', label='Limit = 1')
ax1.set_xlabel('Upper limit $t$')
ax1.set_ylabel('$\\int_1^t x^{-2} \\, dx$')
ax1.set_title('Convergent: $\\int_1^\\infty x^{-2} \\, dx = 1$')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_values, partial_divergent, 'r-', linewidth=2)
ax2.set_xlabel('Upper limit $t$')
ax2.set_ylabel('$\\int_1^t x^{-1} \\, dx$')
ax2.set_title('Divergent: $\\int_1^\\infty x^{-1} \\, dx = \\infty$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improper_integrals.png', dpi=150)
plt.show()
```

## Choosing the Right Technique

A decision guide for selecting the appropriate integration method:

```
Is the integrand a rational function P(x)/Q(x)?
  YES --> Partial fractions (if deg P >= deg Q, do polynomial division first)

Does it contain sqrt(a^2 - x^2), sqrt(a^2 + x^2), or sqrt(x^2 - a^2)?
  YES --> Trigonometric substitution

Is there a "function and its derivative" pattern?
  YES --> u-substitution

Is it a product of two different types of functions?
  YES --> Integration by parts (use LIATE for choosing u)

Does it involve powers of sin and cos?
  YES --> Use odd-power or power-reduction strategies

None of the above?
  --> Try SymPy or numerical integration
```

## Summary

- **$u$-Substitution** undoes the chain rule: look for $f(g(x)) \cdot g'(x)$ patterns
- **Integration by parts** undoes the product rule: use the LIATE rule to choose $u$ and $dv$
- **Partial fractions** decompose rational functions into simpler terms that integrate to logarithms and arctangents
- **Trigonometric integrals** use identities and substitutions to handle powers and roots
- **Improper integrals** extend definite integrals to infinite domains or singular integrands; convergence depends on how fast the integrand decays
- When manual techniques fail, **SymPy** and **scipy.integrate.quad** provide reliable computational alternatives

## Practice Problems

### Problem 1: Substitution

Evaluate each integral:

(a) $\int \frac{e^{\sqrt{x}}}{\sqrt{x}} \, dx$

(b) $\int_0^{\pi/2} \cos x \cdot e^{\sin x} \, dx$

(c) $\int \frac{x}{(x^2+1)^3} \, dx$

### Problem 2: Integration by Parts

Evaluate:

(a) $\int x^2 e^{-x} \, dx$

(b) $\int e^x \cos x \, dx$ (Hint: apply by-parts twice, then solve for the integral algebraically)

(c) $\int \arctan x \, dx$

### Problem 3: Partial Fractions

Evaluate:

(a) $\int \frac{3x+1}{x^2-5x+6} \, dx$

(b) $\int \frac{x^2 + 1}{x(x-1)^2} \, dx$

### Problem 4: Trigonometric Substitution

Evaluate $\int \frac{x^2}{\sqrt{9-x^2}} \, dx$ using the substitution $x = 3\sin\theta$.

### Problem 5: Improper Integral Convergence

Determine whether each integral converges or diverges. If it converges, find its value.

(a) $\int_0^{\infty} x e^{-x} \, dx$

(b) $\int_0^1 \frac{1}{x^{2/3}} \, dx$

(c) $\int_2^{\infty} \frac{1}{x \ln^2 x} \, dx$

Verify your answers using SymPy.

## References

- Stewart, *Calculus: Early Transcendentals*, Ch. 7 (Techniques of Integration)
- [Paul's Online Notes: Integration Techniques](https://tutorial.math.lamar.edu/Classes/CalcII/IntTechIntro.aspx)
- [MIT OCW 18.01: Techniques of Integration](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)

---

[Previous: Integration Fundamentals](./04_Integration_Fundamentals.md) | [Next: Applications of Integration](./06_Applications_of_Integration.md)

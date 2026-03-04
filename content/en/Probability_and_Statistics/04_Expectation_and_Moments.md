# Expectation and Moments

## Learning Objectives

After completing this lesson, you will be able to:

1. Compute the expected value of discrete and continuous random variables
2. Apply the linearity of expectation to simplify calculations
3. Compute variance using both the definition and the shortcut formula
4. Define and interpret higher moments (skewness and kurtosis)
5. Derive moments from the moment generating function (MGF)
6. Apply Markov's, Chebyshev's, and Jensen's inequalities to bound probabilities
7. Implement moment computations in Python

---

## Overview

The expectation (mean) is the single most important summary of a random variable's distribution -- it tells us the "center of mass." Variance measures spread, and higher moments capture shape. The moment generating function provides an elegant algebraic tool that encodes all moments in one function. Probability inequalities let us derive useful bounds even when the full distribution is unknown.

---

## Table of Contents

1. [Expected Value](#1-expected-value)
2. [Properties of Expectation](#2-properties-of-expectation)
3. [Variance](#3-variance)
4. [Higher Moments: Skewness and Kurtosis](#4-higher-moments-skewness-and-kurtosis)
5. [Moment Generating Function](#5-moment-generating-function)
6. [Probability Inequalities](#6-probability-inequalities)
7. [Python Examples](#7-python-examples)
8. [Key Takeaways](#8-key-takeaways)

---

## 1. Expected Value

### Discrete Case

If $X$ is a discrete random variable with PMF $p_X(x)$, the **expected value** (or **mean**) is:

$$E[X] = \mu_X = \sum_{x} x \, p_X(x)$$

provided $\sum_{x} |x| \, p_X(x) < \infty$ (absolute convergence).

### Continuous Case

If $X$ is a continuous random variable with PDF $f_X(x)$:

$$E[X] = \mu_X = \int_{-\infty}^{\infty} x \, f_X(x) \, dx$$

provided $\int_{-\infty}^{\infty} |x| \, f_X(x) \, dx < \infty$.

### Law of the Unconscious Statistician (LOTUS)

To compute $E[g(X)]$ without first finding the distribution of $Y = g(X)$:

**Discrete**: $E[g(X)] = \sum_{x} g(x) \, p_X(x)$

**Continuous**: $E[g(X)] = \int_{-\infty}^{\infty} g(x) \, f_X(x) \, dx$

This is extremely useful -- we can compute expectations of transformed variables directly from the original distribution.

### Example: Expected Sum of Two Dice

Let $X$ be the sum. Using the PMF from Lesson 03:

$$E[X] = \sum_{x=2}^{12} x \cdot p_X(x) = 2 \cdot \frac{1}{36} + 3 \cdot \frac{2}{36} + \cdots + 12 \cdot \frac{1}{36} = 7$$

Alternatively, let $X = D_1 + D_2$ where each die has $E[D_i] = 3.5$, so $E[X] = 3.5 + 3.5 = 7$ by linearity.

---

## 2. Properties of Expectation

### Linearity of Expectation

For any random variables $X$ and $Y$ (even if dependent!) and constants $a, b, c$:

$$E[aX + bY + c] = aE[X] + bE[Y] + c$$

This extends to any finite sum:

$$E\left[\sum_{i=1}^{n} a_i X_i\right] = \sum_{i=1}^{n} a_i E[X_i]$$

**Linearity holds regardless of dependence** -- this is one of the most powerful properties in probability.

### Monotonicity

If $X \leq Y$ (for all outcomes), then $E[X] \leq E[Y]$.

### Expectation of a Constant

$$E[c] = c$$

### Non-Negative Random Variables

If $X \geq 0$, then $E[X] \geq 0$.

### Expectation of a Product (Independent Case Only)

If $X$ and $Y$ are **independent**:

$$E[XY] = E[X] \cdot E[Y]$$

**Warning**: This does NOT hold for dependent random variables in general.

---

## 3. Variance

### Definition

The **variance** of $X$ measures the expected squared deviation from the mean:

$$\text{Var}(X) = \sigma_X^2 = E\left[(X - \mu_X)^2\right]$$

### Shortcut (Computational) Formula

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

*Proof*:

$$E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

### Standard Deviation

$$\sigma_X = \sqrt{\text{Var}(X)}$$

The standard deviation has the same units as $X$, making it more interpretable than variance.

### Properties of Variance

1. $\text{Var}(X) \geq 0$ always; $\text{Var}(X) = 0$ iff $X$ is a constant a.s.
2. $\text{Var}(aX + b) = a^2 \, \text{Var}(X)$ (constants shift; scaling squares)
3. If $X$ and $Y$ are **independent**: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$
4. In general: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$

### Example: Variance of a Fair Die

$E[X] = 3.5$ and $E[X^2] = \frac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6} \approx 15.167$

$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - \left(\frac{7}{2}\right)^2 = \frac{91}{6} - \frac{49}{4} = \frac{35}{12} \approx 2.917$$

---

## 4. Higher Moments: Skewness and Kurtosis

### The $k$-th Moment

The **$k$-th moment** of $X$ about the origin:

$$\mu_k' = E[X^k]$$

The **$k$-th central moment**:

$$\mu_k = E[(X - \mu)^k]$$

Note: $\mu_1 = 0$ always, $\mu_2 = \text{Var}(X)$.

### Skewness

**Skewness** measures asymmetry of the distribution:

$$\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3} = \frac{\mu_3}{\sigma^3}$$

- $\gamma_1 > 0$: right-skewed (long right tail)
- $\gamma_1 = 0$: symmetric
- $\gamma_1 < 0$: left-skewed (long left tail)

### Kurtosis

**Kurtosis** measures the heaviness of tails relative to the normal distribution:

$$\gamma_2 = \frac{E[(X - \mu)^4]}{\sigma^4} = \frac{\mu_4}{\sigma^4}$$

The normal distribution has $\gamma_2 = 3$. The **excess kurtosis** is $\gamma_2 - 3$:

- Excess > 0 (**leptokurtic**): heavier tails than normal
- Excess = 0 (**mesokurtic**): normal-like tails
- Excess < 0 (**platykurtic**): lighter tails than normal

---

## 5. Moment Generating Function

### Definition

The **moment generating function** (MGF) of $X$ is:

$$M_X(t) = E[e^{tX}]$$

defined for all $t$ in some open interval containing 0.

**Discrete**: $M_X(t) = \sum_{x} e^{tx} \, p_X(x)$

**Continuous**: $M_X(t) = \int_{-\infty}^{\infty} e^{tx} \, f_X(x) \, dx$

### Moment Extraction

The key property -- moments can be extracted by differentiation:

$$E[X^k] = M_X^{(k)}(0) = \left.\frac{d^k}{dt^k} M_X(t)\right|_{t=0}$$

*Proof sketch*: $e^{tX} = \sum_{k=0}^{\infty} \frac{(tX)^k}{k!}$, so $M_X(t) = \sum_{k=0}^{\infty} \frac{t^k}{k!} E[X^k]$. Differentiating $k$ times and setting $t = 0$ isolates $E[X^k]$.

### Uniqueness Theorem

If two random variables have the same MGF in a neighborhood of $t = 0$, then they have the same distribution. This makes MGFs a powerful tool for identifying distributions.

### MGF Properties

1. **Constant**: $M_c(t) = e^{ct}$
2. **Linear transformation**: $M_{aX+b}(t) = e^{bt} M_X(at)$
3. **Sum of independent RVs**: If $X \perp Y$, then $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$

### Example: MGF of Bernoulli($p$)

$$M_X(t) = E[e^{tX}] = e^{t \cdot 0}(1-p) + e^{t \cdot 1}p = (1-p) + pe^t$$

First moment: $M_X'(t) = pe^t$, so $E[X] = M_X'(0) = p$.

Second moment: $M_X''(t) = pe^t$, so $E[X^2] = M_X''(0) = p$.

Variance: $\text{Var}(X) = p - p^2 = p(1-p)$.

---

## 6. Probability Inequalities

### Markov's Inequality

If $X \geq 0$ and $a > 0$:

$$P(X \geq a) \leq \frac{E[X]}{a}$$

*Proof*: $E[X] = E[X \cdot \mathbf{1}_{X \geq a}] + E[X \cdot \mathbf{1}_{X < a}] \geq E[X \cdot \mathbf{1}_{X \geq a}] \geq a \cdot P(X \geq a)$.

**Example**: If the average number of server requests per minute is 100, then $P(X \geq 500) \leq 100/500 = 0.20$.

### Chebyshev's Inequality

For any random variable $X$ with finite mean $\mu$ and variance $\sigma^2$, and for $k > 0$:

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

Equivalently:

$$P(|X - \mu| \geq a) \leq \frac{\sigma^2}{a^2}$$

*Proof*: Apply Markov's inequality to the non-negative random variable $(X - \mu)^2$.

**Example**: A factory produces bolts with mean length 10 cm and standard deviation 0.1 cm. The probability a bolt deviates by more than 0.3 cm from the mean:

$$P(|X - 10| \geq 0.3) \leq \frac{(0.1)^2}{(0.3)^2} = \frac{1}{9} \approx 0.111$$

### Jensen's Inequality

If $g$ is a **convex** function (i.e., $g''(x) \geq 0$), then:

$$E[g(X)] \geq g(E[X])$$

If $g$ is **concave** ($g''(x) \leq 0$), the inequality reverses.

**Examples**:

- $g(x) = x^2$ is convex: $E[X^2] \geq (E[X])^2$ (i.e., $\text{Var}(X) \geq 0$)
- $g(x) = \ln(x)$ is concave: $E[\ln X] \leq \ln(E[X])$ (used in information theory)
- $g(x) = e^x$ is convex: $E[e^X] \geq e^{E[X]}$

---

## 7. Python Examples

### Computing Moments from a PMF

```python
def compute_moments(values, probs):
    """Compute mean, variance, skewness, and kurtosis from a PMF."""
    # Mean
    mean = sum(x * p for x, p in zip(values, probs))

    # Variance
    e_x2 = sum(x**2 * p for x, p in zip(values, probs))
    var = e_x2 - mean**2

    # Standard deviation
    std = var ** 0.5

    # Skewness
    mu3 = sum((x - mean)**3 * p for x, p in zip(values, probs))
    skew = mu3 / std**3 if std > 0 else 0

    # Kurtosis
    mu4 = sum((x - mean)**4 * p for x, p in zip(values, probs))
    kurt = mu4 / std**4 if std > 0 else 0
    excess_kurt = kurt - 3

    print(f"E[X]            = {mean:.4f}")
    print(f"Var(X)          = {var:.4f}")
    print(f"Std(X)          = {std:.4f}")
    print(f"Skewness        = {skew:.4f}")
    print(f"Kurtosis        = {kurt:.4f}")
    print(f"Excess Kurtosis = {excess_kurt:.4f}")
    return mean, var, skew, kurt

# Fair die
values = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6
compute_moments(values, probs)
# E[X] = 3.5, Var = 2.9167, Skew = 0, Excess Kurt = -1.2686
```

### Moment Generating Function Numerically

```python
import math

def mgf_bernoulli(t, p):
    """MGF of Bernoulli(p): (1-p) + p*e^t"""
    return (1 - p) + p * math.exp(t)

def numerical_derivative(f, x, h=1e-7):
    """Central difference approximation of f'(x)."""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_second_derivative(f, x, h=1e-5):
    """Central difference approximation of f''(x)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

p = 0.4
mgf = lambda t: mgf_bernoulli(t, p)

# E[X] = M'(0)
e_x = numerical_derivative(mgf, 0)
print(f"E[X] from MGF: {e_x:.6f}  (exact: {p})")

# E[X^2] = M''(0)
e_x2 = numerical_second_derivative(mgf, 0)
print(f"E[X^2] from MGF: {e_x2:.6f}  (exact: {p})")

# Var(X) = E[X^2] - (E[X])^2
var = e_x2 - e_x**2
print(f"Var(X) from MGF: {var:.6f}  (exact: {p*(1-p):.6f})")
```

### Linearity of Expectation: Simulation

```python
import random

def demonstrate_linearity(n=500000):
    """Show E[2X + 3Y + 5] = 2E[X] + 3E[Y] + 5 even for dependent X, Y."""
    random.seed(42)
    sum_x = sum_y = sum_combo = 0

    for _ in range(n):
        x = random.gauss(0, 1)  # standard normal approximation
        y = x + random.gauss(0, 0.5)  # Y depends on X!
        z = 2 * x + 3 * y + 5

        sum_x += x
        sum_y += y
        sum_combo += z

    e_x = sum_x / n
    e_y = sum_y / n
    e_z = sum_combo / n
    expected = 2 * e_x + 3 * e_y + 5

    print(f"E[X]             = {e_x:.4f}")
    print(f"E[Y]             = {e_y:.4f}")
    print(f"E[2X + 3Y + 5]   = {e_z:.4f}")
    print(f"2E[X] + 3E[Y] + 5 = {expected:.4f}")
    print(f"Difference       = {abs(e_z - expected):.6f}")

demonstrate_linearity()
```

### Chebyshev's Inequality Verification

```python
import random

def verify_chebyshev(n=1_000_000):
    """Verify Chebyshev's inequality with simulated data."""
    random.seed(99)
    # Exponential distribution: mean=1, var=1, std=1
    samples = [-math.log(1 - random.random()) for _ in range(n)]

    import math
    mean = sum(samples) / n
    var = sum((x - mean)**2 for x in samples) / (n - 1)
    std = math.sqrt(var)

    print(f"Sample mean = {mean:.4f}, Sample std = {std:.4f}\n")
    print(f"{'k':>4}  {'Chebyshev bound':>16}  {'Empirical P':>13}")
    print("-" * 38)
    for k in [1, 1.5, 2, 3, 4]:
        bound = 1 / k**2
        empirical = sum(1 for x in samples if abs(x - mean) >= k * std) / n
        print(f"{k:4.1f}  {bound:16.4f}  {empirical:13.4f}")

verify_chebyshev()
```

### Variance of Sum: Independent vs. Dependent

```python
import random
import math

def variance_of_sum(n=500000):
    """Demonstrate Var(X+Y) = Var(X) + Var(Y) for independent X, Y."""
    random.seed(0)

    # Independent case
    xs = [random.gauss(2, 3) for _ in range(n)]
    ys = [random.gauss(5, 4) for _ in range(n)]
    sums_indep = [x + y for x, y in zip(xs, ys)]

    var_x = sum((x - 2)**2 for x in xs) / n
    var_y = sum((y - 5)**2 for y in ys) / n
    var_sum = sum((s - 7)**2 for s in sums_indep) / n

    print("=== Independent Case ===")
    print(f"Var(X)     = {var_x:.4f}  (theoretical: 9)")
    print(f"Var(Y)     = {var_y:.4f}  (theoretical: 16)")
    print(f"Var(X+Y)   = {var_sum:.4f}  (theoretical: 25)")
    print(f"Var(X)+Var(Y) = {var_x + var_y:.4f}\n")

    # Dependent case: Y = 2X + noise
    xs2 = [random.gauss(0, 1) for _ in range(n)]
    ys2 = [2 * x + random.gauss(0, 0.5) for x in xs2]
    sums_dep = [x + y for x, y in zip(xs2, ys2)]

    m_x = sum(xs2) / n
    m_y = sum(ys2) / n
    m_s = sum(sums_dep) / n
    var_x2 = sum((x - m_x)**2 for x in xs2) / n
    var_y2 = sum((y - m_y)**2 for y in ys2) / n
    var_sum2 = sum((s - m_s)**2 for s in sums_dep) / n

    print("=== Dependent Case (Y = 2X + noise) ===")
    print(f"Var(X)       = {var_x2:.4f}")
    print(f"Var(Y)       = {var_y2:.4f}")
    print(f"Var(X+Y)     = {var_sum2:.4f}")
    print(f"Var(X)+Var(Y) = {var_x2 + var_y2:.4f}")
    print(f"Var(X+Y) != Var(X)+Var(Y) when dependent!")

variance_of_sum()
```

---

## 8. Key Takeaways

1. **Expected value** $E[X]$ is the probability-weighted average of all possible values. It represents the long-run average over many repetitions.

2. **Linearity of expectation** $E[aX + bY] = aE[X] + bE[Y]$ holds **always**, regardless of dependence. This is arguably the most useful property in probability.

3. **Variance** $\text{Var}(X) = E[X^2] - (E[X])^2$ measures spread. The shortcut formula is almost always easier to compute. Variance is additive only for independent (or uncorrelated) random variables.

4. **Skewness** measures asymmetry; **kurtosis** measures tail heaviness relative to the normal distribution.

5. **The MGF** $M_X(t) = E[e^{tX}]$ encodes all moments: $E[X^k] = M_X^{(k)}(0)$. If two random variables share the same MGF, they share the same distribution. The MGF of a sum of independent RVs is the product of their individual MGFs.

6. **Probability inequalities** provide distribution-free bounds:
   - **Markov**: $P(X \geq a) \leq E[X]/a$ (requires $X \geq 0$)
   - **Chebyshev**: $P(|X - \mu| \geq k\sigma) \leq 1/k^2$ (requires only mean and variance)
   - **Jensen**: $E[g(X)] \geq g(E[X])$ for convex $g$

---

*Previous: [03 - Random Variables and Distributions](./03_Random_Variables_and_Distributions.md) | Next: [05 - Joint Distributions](./05_Joint_Distributions.md)*

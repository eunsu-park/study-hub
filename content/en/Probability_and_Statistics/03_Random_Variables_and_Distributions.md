# Random Variables and Distributions

## Learning Objectives

After completing this lesson, you will be able to:

1. Define a random variable as a measurable function from the sample space to the real line
2. Distinguish between discrete and continuous random variables
3. Specify distributions via PMF, PDF, and CDF
4. State and verify the properties that valid PMFs, PDFs, and CDFs must satisfy
5. Compute probabilities of intervals using the CDF
6. Define and compute quantile functions (inverse CDF)
7. Simulate random variables in Python using the standard library

---

## Overview

A random variable transforms abstract experimental outcomes into numbers, enabling us to use the full power of calculus and algebra in probability. This lesson formalizes the concept and introduces the three main ways to describe a random variable's distribution: the probability mass function (PMF), probability density function (PDF), and cumulative distribution function (CDF).

---

## Table of Contents

1. [Definition of a Random Variable](#1-definition-of-a-random-variable)
2. [Discrete Random Variables](#2-discrete-random-variables)
3. [Continuous Random Variables](#3-continuous-random-variables)
4. [Cumulative Distribution Function](#4-cumulative-distribution-function)
5. [Quantile Function](#5-quantile-function)
6. [Mixed Distributions](#6-mixed-distributions)
7. [Functions of a Random Variable](#7-functions-of-a-random-variable)
8. [Python Examples](#8-python-examples)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. Definition of a Random Variable

### Formal Definition

A **random variable** $X$ is a function that maps each outcome $\omega$ in the sample space $\Omega$ to a real number:

$$X : \Omega \to \mathbb{R}$$

Technically, $X$ must be **measurable** -- for every Borel set $B \subseteq \mathbb{R}$, the preimage $\{\omega \in \Omega : X(\omega) \in B\}$ must be an event in the sigma-algebra $\mathcal{F}$.

### Intuition

The sample space might consist of abstract outcomes (like "the third transistor fails first"). A random variable assigns a number to each outcome, letting us work with real numbers rather than abstract labels.

**Example**: Roll two dice. Let $X$ = sum of the faces. The sample space has 36 outcomes, but $X$ takes values in $\{2, 3, \ldots, 12\}$.

### Notation Convention

- Random variables: uppercase letters ($X$, $Y$, $Z$)
- Observed values: lowercase letters ($x$, $y$, $z$)
- $P(X = x)$ or $P(X \leq x)$ means the probability of the event $\{\omega : X(\omega) = x\}$ or $\{\omega : X(\omega) \leq x\}$

---

## 2. Discrete Random Variables

### Definition

A random variable $X$ is **discrete** if it takes values in a finite or countably infinite set $\{x_1, x_2, x_3, \ldots\}$.

### Probability Mass Function (PMF)

The **PMF** $p_X(x)$ gives the probability that $X$ equals $x$:

$$p_X(x) = P(X = x)$$

### PMF Properties

A valid PMF must satisfy:

1. **Non-negativity**: $p_X(x) \geq 0$ for all $x$
2. **Normalization**: $\sum_{\text{all } x} p_X(x) = 1$
3. **Probability of a set**: $P(X \in A) = \sum_{x \in A} p_X(x)$

### Example: Sum of Two Dice

| $x$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-----|---|---|---|---|---|---|---|---|----|----|-----|
| $p_X(x)$ | $\frac{1}{36}$ | $\frac{2}{36}$ | $\frac{3}{36}$ | $\frac{4}{36}$ | $\frac{5}{36}$ | $\frac{6}{36}$ | $\frac{5}{36}$ | $\frac{4}{36}$ | $\frac{3}{36}$ | $\frac{2}{36}$ | $\frac{1}{36}$ |

Verification: $1 + 2 + 3 + 4 + 5 + 6 + 5 + 4 + 3 + 2 + 1 = 36$, so $\sum p_X(x) = 36/36 = 1$.

---

## 3. Continuous Random Variables

### Definition

A random variable $X$ is **continuous** if there exists a non-negative function $f_X(x)$ such that for any interval $[a, b]$:

$$P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$$

### Probability Density Function (PDF)

The function $f_X(x)$ is called the **probability density function**.

### PDF Properties

1. **Non-negativity**: $f_X(x) \geq 0$ for all $x$
2. **Normalization**: $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$
3. **Probability of an interval**: $P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$

### Critical Distinction from PMF

For a continuous random variable, the probability of any single point is **zero**:

$$P(X = x) = \int_x^x f_X(t) \, dt = 0$$

Consequently, for continuous $X$:

$$P(a \leq X \leq b) = P(a < X < b) = P(a \leq X < b) = P(a < X \leq b)$$

**Note**: $f_X(x)$ is a **density**, not a probability. It can exceed 1 (e.g., $f_X(x) = 3$ on $[0, 1/3]$). Only the integral over an interval gives a probability.

### Example: Uniform Distribution on $[0, 1]$

$$f_X(x) = \begin{cases} 1 & \text{if } 0 \leq x \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

$$P(0.2 \leq X \leq 0.7) = \int_{0.2}^{0.7} 1 \, dx = 0.5$$

---

## 4. Cumulative Distribution Function

### Definition

The **CDF** $F_X(x)$ of a random variable $X$ is defined for all $x \in \mathbb{R}$:

$$F_X(x) = P(X \leq x)$$

The CDF provides a **universal** description that works for discrete, continuous, and mixed random variables.

### CDF Properties

Every valid CDF satisfies:

1. **Non-decreasing**: If $a < b$, then $F_X(a) \leq F_X(b)$
2. **Right-continuous**: $\lim_{x \to a^+} F_X(x) = F_X(a)$
3. **Limits**: $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to \infty} F_X(x) = 1$
4. **Bounded**: $0 \leq F_X(x) \leq 1$ for all $x$

### Computing Probabilities from the CDF

$$P(a < X \leq b) = F_X(b) - F_X(a)$$

$$P(X > a) = 1 - F_X(a)$$

$$P(X = a) = F_X(a) - \lim_{x \to a^-} F_X(x) \quad (\text{size of jump at } a)$$

### CDF for Discrete Random Variables

For a discrete $X$ with PMF $p_X$:

$$F_X(x) = \sum_{x_i \leq x} p_X(x_i)$$

The CDF is a **staircase function** with jumps at each value in the support.

### CDF for Continuous Random Variables

For a continuous $X$ with PDF $f_X$:

$$F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt$$

The CDF is **continuous** (no jumps), and $f_X(x) = F_X'(x)$ wherever the derivative exists.

### Example: Exponential Distribution

PDF: $f_X(x) = \lambda e^{-\lambda x}$ for $x \geq 0$ (with $\lambda > 0$).

CDF:

$$F_X(x) = \int_0^x \lambda e^{-\lambda t} \, dt = 1 - e^{-\lambda x}, \quad x \geq 0$$

$$P(1 \leq X \leq 3) = F_X(3) - F_X(1) = (1 - e^{-3\lambda}) - (1 - e^{-\lambda}) = e^{-\lambda} - e^{-3\lambda}$$

---

## 5. Quantile Function

### Definition

The **quantile function** (or inverse CDF) $F_X^{-1}(p)$ for $0 < p < 1$ is:

$$F_X^{-1}(p) = \inf\{x : F_X(x) \geq p\}$$

For strictly increasing, continuous CDFs, this simplifies to solving $F_X(x) = p$ for $x$.

### Special Quantiles

| Name | Value of $p$ |
|------|-------------|
| Median | $p = 0.5$ |
| First quartile ($Q_1$) | $p = 0.25$ |
| Third quartile ($Q_3$) | $p = 0.75$ |
| $k$-th percentile | $p = k/100$ |

### Example

For $X \sim \text{Exponential}(\lambda)$:

$$F_X(x) = 1 - e^{-\lambda x} = p \implies x = -\frac{\ln(1 - p)}{\lambda}$$

Median: $x_{0.5} = \frac{\ln 2}{\lambda}$

### Inverse Transform Sampling

If $U \sim \text{Uniform}(0, 1)$, then $X = F_X^{-1}(U)$ has CDF $F_X$. This is a fundamental technique for simulating random variables.

---

## 6. Mixed Distributions

A **mixed** random variable has a CDF that is partly continuous and partly has jumps. It cannot be described by a PMF alone or a PDF alone.

**Example**: A call center wait time $X$:

- With probability 0.3, the call is answered immediately ($X = 0$)
- With probability 0.7, the wait is exponentially distributed

$$F_X(x) = \begin{cases} 0 & x < 0 \\ 0.3 + 0.7(1 - e^{-\lambda x}) & x \geq 0 \end{cases}$$

Note the jump of size 0.3 at $x = 0$, followed by a continuous increase.

---

## 7. Functions of a Random Variable

If $X$ is a random variable and $g : \mathbb{R} \to \mathbb{R}$ is a (measurable) function, then $Y = g(X)$ is also a random variable.

### Discrete Case

If $X$ is discrete with PMF $p_X$, then $Y = g(X)$ has PMF:

$$p_Y(y) = \sum_{\{x : g(x) = y\}} p_X(x)$$

### Continuous Case (Monotone $g$)

If $g$ is strictly monotone and differentiable with inverse $g^{-1}$, then:

$$f_Y(y) = f_X(g^{-1}(y)) \left| \frac{d}{dy} g^{-1}(y) \right|$$

This is the **change of variables** formula (explored in detail in Lesson 08).

---

## 8. Python Examples

### Discrete Random Variable: Dice Sum PMF

```python
from collections import Counter

def dice_sum_pmf():
    """Compute and display the PMF of the sum of two fair dice."""
    outcomes = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
    sums = [d1 + d2 for d1, d2 in outcomes]
    counts = Counter(sums)

    print("x   P(X=x)     Fraction")
    print("-" * 30)
    total = len(outcomes)  # 36
    for x in range(2, 13):
        prob = counts[x] / total
        print(f"{x:2d}  {prob:.4f}     {counts[x]}/{total}")

    # Verify normalization
    assert sum(counts.values()) == total

dice_sum_pmf()
```

### CDF Computation

```python
def dice_sum_cdf():
    """Compute the CDF of the sum of two fair dice."""
    from collections import Counter

    outcomes = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
    sums = [d1 + d2 for d1, d2 in outcomes]
    counts = Counter(sums)

    total = 36
    cumulative = 0.0
    print("x   F(x) = P(X <= x)")
    print("-" * 25)
    for x in range(2, 13):
        cumulative += counts[x] / total
        print(f"{x:2d}  {cumulative:.4f}")

    # P(5 < X <= 9) = F(9) - F(5)
    f9 = sum(counts[k] for k in range(2, 10)) / total
    f5 = sum(counts[k] for k in range(2, 6)) / total
    print(f"\nP(5 < X <= 9) = F(9) - F(5) = {f9:.4f} - {f5:.4f} = {f9 - f5:.4f}")

dice_sum_cdf()
```

### Inverse Transform Sampling

```python
import random
import math

def inverse_transform_exponential(lam=1.0, n=10000):
    """Generate Exponential(lam) samples via inverse transform."""
    random.seed(42)
    samples = []
    for _ in range(n):
        u = random.random()          # U ~ Uniform(0, 1)
        x = -math.log(1 - u) / lam   # F^{-1}(u) for Exponential
        samples.append(x)

    # Check empirical mean and variance
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)

    print(f"Exponential(lambda={lam}) via inverse transform:")
    print(f"  Theoretical mean = {1/lam:.4f},  Sample mean = {mean:.4f}")
    print(f"  Theoretical var  = {1/lam**2:.4f},  Sample var  = {var:.4f}")

    # Compute empirical CDF at a few points
    for t in [0.5, 1.0, 2.0]:
        empirical = sum(1 for x in samples if x <= t) / n
        theoretical = 1 - math.exp(-lam * t)
        print(f"  F({t}) = {theoretical:.4f} (theoretical), {empirical:.4f} (empirical)")

inverse_transform_exponential()
```

### Simulating a Mixed Distribution

```python
import random
import math

def mixed_distribution_sim(n=100000, lam=2.0):
    """Simulate a mixed distribution: P(X=0)=0.3, else Exp(lam)."""
    random.seed(7)
    samples = []
    for _ in range(n):
        if random.random() < 0.3:
            samples.append(0.0)  # Point mass at 0
        else:
            u = random.random()
            samples.append(-math.log(1 - u) / lam)  # Exponential

    # Empirical P(X = 0)
    p_zero = sum(1 for x in samples if x == 0.0) / n
    mean = sum(samples) / n

    # Theoretical mean: 0.3*0 + 0.7*(1/lam) = 0.7/lam
    print(f"P(X = 0): empirical = {p_zero:.4f}, theoretical = 0.3000")
    print(f"E[X]:     empirical = {mean:.4f}, theoretical = {0.7/lam:.4f}")

mixed_distribution_sim()
```

### Verifying PDF Normalization Numerically

```python
def trapezoidal_integrate(f, a, b, n=10000):
    """Numerical integration using the trapezoidal rule."""
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h

import math

# Verify that the standard normal PDF integrates to 1
def standard_normal_pdf(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

# Integrate from -10 to 10 (effectively -inf to inf for the normal)
result = trapezoidal_integrate(standard_normal_pdf, -10, 10, n=100000)
print(f"Integral of standard normal PDF from -10 to 10: {result:.8f}")
# Should be very close to 1.0
```

---

## 9. Key Takeaways

1. **A random variable** is a function $X: \Omega \to \mathbb{R}$ that maps outcomes to numbers. It is the central object in probability theory.

2. **Discrete random variables** have countable support and are described by a **PMF** where $p_X(x) = P(X = x)$.

3. **Continuous random variables** have uncountable support and are described by a **PDF** where $P(a \leq X \leq b) = \int_a^b f_X(x)\,dx$. The probability at any single point is zero.

4. **The CDF** $F_X(x) = P(X \leq x)$ is universal -- it works for discrete, continuous, and mixed random variables. It is non-decreasing, right-continuous, and bounded between 0 and 1.

5. **The quantile function** inverts the CDF: $F_X^{-1}(p)$ gives the value below which a proportion $p$ of the distribution lies. It enables **inverse transform sampling**.

6. **Mixed distributions** combine point masses and continuous densities. They arise naturally in applications (e.g., insurance claims that may be zero).

---

*Previous: [02 - Probability Axioms and Rules](./02_Probability_Axioms_and_Rules.md) | Next: [04 - Expectation and Moments](./04_Expectation_and_Moments.md)*

# Joint Distributions

## Learning Objectives

After completing this lesson, you will be able to:

1. Define and work with joint PMFs and joint PDFs for two or more random variables
2. Derive marginal distributions from a joint distribution
3. Compute and interpret conditional distributions
4. Test for independence of random variables using the factorization criterion
5. Compute covariance and the Pearson correlation coefficient
6. Apply the Law of Iterated Expectation (tower property)
7. Implement joint distribution computations in Python

---

## Overview

Real-world problems involve multiple random variables simultaneously. A joint distribution captures the full probabilistic relationship between two or more random variables, including any dependence structure. From a joint distribution, we can extract marginals (individual variable distributions), conditionals (how one variable behaves given knowledge of another), and summary measures like covariance and correlation.

---

## Table of Contents

1. [Joint PMF (Discrete Case)](#1-joint-pmf-discrete-case)
2. [Joint PDF (Continuous Case)](#2-joint-pdf-continuous-case)
3. [Marginal Distributions](#3-marginal-distributions)
4. [Conditional Distributions](#4-conditional-distributions)
5. [Independence of Random Variables](#5-independence-of-random-variables)
6. [Covariance and Correlation](#6-covariance-and-correlation)
7. [Conditional Expectation and the Tower Property](#7-conditional-expectation-and-the-tower-property)
8. [Python Examples](#8-python-examples)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. Joint PMF (Discrete Case)

### Definition

For discrete random variables $X$ and $Y$, the **joint PMF** is:

$$p_{X,Y}(x, y) = P(X = x, Y = y)$$

### Properties

1. **Non-negativity**: $p_{X,Y}(x, y) \geq 0$ for all $x, y$
2. **Normalization**: $\sum_{x}\sum_{y} p_{X,Y}(x, y) = 1$
3. **Probability of a set**: $P((X,Y) \in A) = \sum_{(x,y) \in A} p_{X,Y}(x, y)$

### Example: Joint PMF Table

Consider two coins: $X$ = number of heads on coin 1 (fair), $Y$ = number of heads on coin 2 (biased, $P(H) = 0.7$).

|  | $Y=0$ | $Y=1$ | Marginal $p_X$ |
|---|---|---|---|
| $X=0$ | 0.15 | 0.35 | 0.50 |
| $X=1$ | 0.15 | 0.35 | 0.50 |
| Marginal $p_Y$ | 0.30 | 0.70 | 1.00 |

Here $X$ and $Y$ are independent (we will verify this formally in Section 5).

---

## 2. Joint PDF (Continuous Case)

### Definition

For continuous random variables $X$ and $Y$, the **joint PDF** $f_{X,Y}(x, y)$ satisfies:

$$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) \, dx \, dy$$

### Properties

1. **Non-negativity**: $f_{X,Y}(x, y) \geq 0$ for all $x, y$
2. **Normalization**: $\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx \, dy = 1$

### Example: Uniform on the Unit Square

$$f_{X,Y}(x, y) = \begin{cases} 1 & \text{if } 0 \leq x \leq 1 \text{ and } 0 \leq y \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

$$P(X + Y \leq 1) = \int_0^1 \int_0^{1-x} 1 \, dy \, dx = \int_0^1 (1-x) \, dx = \frac{1}{2}$$

### Joint CDF

$$F_{X,Y}(x, y) = P(X \leq x, Y \leq y)$$

For the continuous case: $f_{X,Y}(x, y) = \frac{\partial^2}{\partial x \, \partial y} F_{X,Y}(x, y)$.

---

## 3. Marginal Distributions

### Discrete Case

The **marginal PMF** of $X$ is obtained by summing over all values of $Y$:

$$p_X(x) = \sum_{y} p_{X,Y}(x, y)$$

Similarly: $p_Y(y) = \sum_{x} p_{X,Y}(x, y)$

### Continuous Case

The **marginal PDF** of $X$ is obtained by integrating out $Y$:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy$$

### Example: Non-Uniform Joint Distribution

Let $f_{X,Y}(x,y) = 6(1-y)$ for $0 \leq x \leq y \leq 1$.

Marginal of $X$:

$$f_X(x) = \int_x^1 6(1-y) \, dy = 6\left[(1-y) \cdot (-1)\Big|_{y=x}^{y=1} - \int_x^1 (-1) \, dy\right]$$

Using direct integration:

$$f_X(x) = \int_x^1 6(1-y) \, dy = 6\left[y - \frac{y^2}{2}\right]_x^1 = 6\left[\frac{1}{2} - x + \frac{x^2}{2}\right] = 3(1 - x)^2$$

Marginal of $Y$:

$$f_Y(y) = \int_0^y 6(1-y) \, dx = 6y(1-y), \quad 0 \leq y \leq 1$$

---

## 4. Conditional Distributions

### Discrete Case

The **conditional PMF** of $Y$ given $X = x$ (where $p_X(x) > 0$):

$$p_{Y|X}(y \mid x) = \frac{p_{X,Y}(x, y)}{p_X(x)}$$

### Continuous Case

The **conditional PDF** of $Y$ given $X = x$ (where $f_X(x) > 0$):

$$f_{Y|X}(y \mid x) = \frac{f_{X,Y}(x, y)}{f_X(x)}$$

### Properties

Conditional distributions are proper distributions:

- $p_{Y|X}(y \mid x) \geq 0$ and $\sum_y p_{Y|X}(y \mid x) = 1$ (discrete)
- $f_{Y|X}(y \mid x) \geq 0$ and $\int f_{Y|X}(y \mid x) \, dy = 1$ (continuous)

### Example

Using the joint $f_{X,Y}(x,y) = 6(1-y)$ for $0 \leq x \leq y \leq 1$:

$$f_{Y|X}(y \mid x) = \frac{6(1-y)}{3(1-x)^2} = \frac{2(1-y)}{(1-x)^2}, \quad x \leq y \leq 1$$

This is the conditional density of $Y$ given $X = x$. Note how it depends on $x$.

---

## 5. Independence of Random Variables

### Definition

Random variables $X$ and $Y$ are **independent** if and only if their joint distribution factors into the product of marginals:

**Discrete**: $p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y)$ for all $x, y$

**Continuous**: $f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y)$ for all $x, y$

### Equivalent Conditions

Any one of the following implies the others:

1. $F_{X,Y}(x, y) = F_X(x) \cdot F_Y(y)$ for all $x, y$
2. The joint PMF/PDF factors as above
3. $f_{Y|X}(y \mid x) = f_Y(y)$ for all $x, y$ (conditional equals marginal)
4. $E[g(X)h(Y)] = E[g(X)] \cdot E[h(Y)]$ for all functions $g, h$

### Checking Independence: The Factorization Criterion

For continuous $(X, Y)$, they are independent iff $f_{X,Y}(x, y)$ can be written as $g(x) \cdot h(y)$ for some functions $g$ and $h$ (and the support is a Cartesian product).

**Example**: $f_{X,Y}(x,y) = 6(1-y)$ on $\{0 \leq x \leq y \leq 1\}$. The support is **not** a rectangle (it depends on the relationship $x \leq y$), so $X$ and $Y$ are **not** independent.

---

## 6. Covariance and Correlation

### Covariance

The **covariance** of $X$ and $Y$ measures their linear association:

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

### Properties of Covariance

1. $\text{Cov}(X, X) = \text{Var}(X)$
2. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ (symmetric)
3. $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
4. $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$ (bilinearity)
5. If $X \perp Y$, then $\text{Cov}(X, Y) = 0$

**Warning**: $\text{Cov}(X, Y) = 0$ does NOT imply independence. Uncorrelated variables can still be dependent.

**Classic counterexample**: Let $X \sim \text{Uniform}(-1, 1)$ and $Y = X^2$. Then $\text{Cov}(X, Y) = E[X^3] - E[X]E[X^2] = 0 - 0 = 0$, but $Y$ is completely determined by $X$.

### Correlation Coefficient

The **Pearson correlation coefficient** normalizes covariance to $[-1, 1]$:

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- $\rho = 1$: perfect positive linear relationship ($Y = aX + b$, $a > 0$)
- $\rho = -1$: perfect negative linear relationship ($Y = aX + b$, $a < 0$)
- $\rho = 0$: uncorrelated (no linear relationship, but possibly nonlinear dependence)

### Variance of a Sum (General Case)

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

More generally:

$$\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j)$$

---

## 7. Conditional Expectation and the Tower Property

### Conditional Expectation

The **conditional expectation** of $Y$ given $X = x$ is:

**Discrete**: $E[Y \mid X = x] = \sum_y y \, p_{Y|X}(y \mid x)$

**Continuous**: $E[Y \mid X = x] = \int_{-\infty}^{\infty} y \, f_{Y|X}(y \mid x) \, dy$

$E[Y \mid X]$ (as a function of the random variable $X$) is itself a random variable.

### Law of Iterated Expectation (Tower Property)

$$E[Y] = E\big[E[Y \mid X]\big]$$

In the discrete case:

$$E[Y] = \sum_x E[Y \mid X = x] \, p_X(x)$$

This is one of the most powerful tools in probability. It lets us decompose a complex expectation by first conditioning on another variable.

### Example: Random Number of Coin Flips

Let $N \sim \text{Poisson}(\lambda)$ be the number of fair coin flips, and $Y$ = number of heads.

Given $N = n$: $Y \mid N = n \sim \text{Binomial}(n, 0.5)$, so $E[Y \mid N = n] = 0.5n$.

By the tower property:

$$E[Y] = E[E[Y \mid N]] = E[0.5N] = 0.5 \, E[N] = 0.5\lambda$$

### Conditional Variance Formula

$$\text{Var}(Y) = E[\text{Var}(Y \mid X)] + \text{Var}(E[Y \mid X])$$

This decomposes total variance into:

- **Within-group variance**: $E[\text{Var}(Y \mid X)]$ (average variance within each $X$ level)
- **Between-group variance**: $\text{Var}(E[Y \mid X])$ (variability of the group means)

---

## 8. Python Examples

### Joint PMF Table

```python
def joint_pmf_example():
    """Work with a joint PMF stored as a 2D dictionary."""
    # Joint distribution of X (rows) and Y (columns)
    # X: number of defective items in batch (0, 1, 2)
    # Y: number returned by customer (0, 1)
    joint = {
        (0, 0): 0.40, (0, 1): 0.00,
        (1, 0): 0.15, (1, 1): 0.20,
        (2, 0): 0.05, (2, 1): 0.20,
    }

    # Verify normalization
    total = sum(joint.values())
    print(f"Sum of joint PMF: {total:.2f}")

    # Marginal of X
    x_values = sorted(set(x for x, y in joint))
    y_values = sorted(set(y for x, y in joint))

    print("\nMarginal of X:")
    p_x = {}
    for x in x_values:
        p_x[x] = sum(joint.get((x, y), 0) for y in y_values)
        print(f"  P(X={x}) = {p_x[x]:.2f}")

    # Marginal of Y
    print("\nMarginal of Y:")
    p_y = {}
    for y in y_values:
        p_y[y] = sum(joint.get((x, y), 0) for x in x_values)
        print(f"  P(Y={y}) = {p_y[y]:.2f}")

    # Conditional distribution: P(Y|X=1)
    print("\nConditional P(Y | X=1):")
    for y in y_values:
        cond = joint.get((1, y), 0) / p_x[1]
        print(f"  P(Y={y} | X=1) = {cond:.4f}")

    # Check independence: P(X=x, Y=y) == P(X=x)*P(Y=y)?
    print("\nIndependence check:")
    independent = True
    for x in x_values:
        for y in y_values:
            product = p_x[x] * p_y[y]
            actual = joint.get((x, y), 0)
            match = abs(actual - product) < 1e-10
            if not match:
                independent = False
            print(f"  P(X={x},Y={y})={actual:.2f}  vs  P(X={x})*P(Y={y})={product:.4f}  {'OK' if match else 'MISMATCH'}")
    print(f"  Independent? {independent}")

joint_pmf_example()
```

### Covariance and Correlation from Simulation

```python
import random
import math

def covariance_simulation(n=500000):
    """Estimate covariance and correlation from simulated data."""
    random.seed(42)
    xs = []
    ys = []

    for _ in range(n):
        x = random.gauss(10, 3)      # X ~ Normal(10, 9)
        y = 2 * x + random.gauss(0, 2)  # Y = 2X + noise
        xs.append(x)
        ys.append(y)

    # Means
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    # Variances
    var_x = sum((x - mean_x)**2 for x in xs) / (n - 1)
    var_y = sum((y - mean_y)**2 for y in ys) / (n - 1)

    # Covariance
    cov_xy = sum((x - mean_x) * (y - mean_y)
                 for x, y in zip(xs, ys)) / (n - 1)

    # Correlation
    rho = cov_xy / (math.sqrt(var_x) * math.sqrt(var_y))

    print(f"E[X]     = {mean_x:.4f}  (theoretical: 10)")
    print(f"E[Y]     = {mean_y:.4f}  (theoretical: 20)")
    print(f"Var(X)   = {var_x:.4f}  (theoretical: 9)")
    print(f"Var(Y)   = {var_y:.4f}  (theoretical: 4*9+4 = 40)")
    print(f"Cov(X,Y) = {cov_xy:.4f}  (theoretical: 2*9 = 18)")
    print(f"rho(X,Y) = {rho:.4f}  (theoretical: 18/sqrt(9*40) ~ 0.9487)")

covariance_simulation()
```

### Uncorrelated but Dependent

```python
import random

def uncorrelated_dependent(n=500000):
    """Demonstrate Cov=0 does not imply independence."""
    random.seed(10)
    xs = []
    ys = []

    for _ in range(n):
        x = random.uniform(-1, 1)  # X ~ Uniform(-1, 1)
        y = x ** 2                 # Y = X^2 (fully dependent!)
        xs.append(x)
        ys.append(y)

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov_xy = sum((x - mean_x) * (y - mean_y)
                 for x, y in zip(xs, ys)) / (n - 1)

    print(f"E[X]     = {mean_x:.6f}  (theoretical: 0)")
    print(f"E[Y]     = {mean_y:.6f}  (theoretical: 1/3)")
    print(f"Cov(X,Y) = {cov_xy:.6f}  (theoretical: 0)")
    print(f"\nY = X^2, so Y is completely determined by X,")
    print(f"yet Cov(X,Y) ~ 0. Uncorrelated != Independent!")

uncorrelated_dependent()
```

### Law of Iterated Expectation

```python
import random
import math

def tower_property_demo(lam=5.0, p=0.5, n=200000):
    """Demonstrate E[Y] = E[E[Y|N]] with N ~ Poisson, Y|N ~ Binomial."""
    random.seed(42)
    ys = []

    for _ in range(n):
        # Generate N ~ Poisson(lam) using inverse transform
        L = math.exp(-lam)
        k = 0
        prob = 1.0
        while prob > L:
            k += 1
            prob *= random.random()
        nn = k - 1  # Poisson sample

        # Given N=nn, generate Y ~ Binomial(nn, p)
        y = sum(1 for _ in range(nn) if random.random() < p)
        ys.append(y)

    empirical_mean = sum(ys) / n
    theoretical_mean = lam * p  # E[Y] = E[E[Y|N]] = E[pN] = p*lam

    print(f"Tower property: E[Y] = E[E[Y|N]] = p * lambda")
    print(f"  lambda = {lam}, p = {p}")
    print(f"  Theoretical E[Y] = {theoretical_mean:.4f}")
    print(f"  Empirical   E[Y] = {empirical_mean:.4f}")

tower_property_demo()
```

### Conditional Variance Formula

```python
import random
import math

def conditional_variance_demo(n=200000):
    """Verify Var(Y) = E[Var(Y|X)] + Var(E[Y|X])."""
    random.seed(55)
    xs = []
    ys = []

    for _ in range(n):
        # X takes values 1, 2, 3 equally likely
        x = random.choice([1, 2, 3])
        # Y | X=x ~ Normal(x, x)  (mean=x, variance=x)
        y = random.gauss(x, math.sqrt(x))
        xs.append(x)
        ys.append(y)

    # Total variance
    mean_y = sum(ys) / n
    var_y_total = sum((y - mean_y)**2 for y in ys) / (n - 1)

    # E[Var(Y|X)]: average of within-group variances
    # Var(E[Y|X]): variance of group means
    group_means = {}
    group_vars = {}
    for x_val in [1, 2, 3]:
        group = [y for x, y in zip(xs, ys) if x == x_val]
        gm = sum(group) / len(group)
        gv = sum((y - gm)**2 for y in group) / (len(group) - 1)
        group_means[x_val] = gm
        group_vars[x_val] = gv

    e_var_y_given_x = sum(group_vars[x] for x in [1, 2, 3]) / 3
    overall_mean_of_means = sum(group_means[x] for x in [1, 2, 3]) / 3
    var_e_y_given_x = sum((group_means[x] - overall_mean_of_means)**2
                          for x in [1, 2, 3]) / 2  # sample variance

    # Theoretical: E[Var(Y|X)] = E[X] = 2, Var(E[Y|X]) = Var(X) = 2/3
    print(f"Total Var(Y)     = {var_y_total:.4f}  (theoretical: E[X]+Var(X) = 2+2/3 ~ 2.6667)")
    print(f"E[Var(Y|X)]      = {e_var_y_given_x:.4f}  (theoretical: 2.0)")
    print(f"Var(E[Y|X])      = {var_e_y_given_x:.4f}  (theoretical: 2/3 ~ 0.6667)")
    print(f"Sum              = {e_var_y_given_x + var_e_y_given_x:.4f}")

conditional_variance_demo()
```

---

## 9. Key Takeaways

1. **Joint distributions** describe the simultaneous behavior of two or more random variables. A joint PMF or PDF contains strictly more information than the individual marginals.

2. **Marginal distributions** are obtained by summing (discrete) or integrating (continuous) out the other variables. Information about dependence is lost in the process.

3. **Conditional distributions** describe one variable's behavior given knowledge of another: $f_{Y|X}(y|x) = f_{X,Y}(x,y)/f_X(x)$.

4. **Independence** means the joint factors into marginals: $f_{X,Y}(x,y) = f_X(x)f_Y(y)$. This is the strongest form of "no relationship."

5. **Covariance** measures linear association: $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$. The **correlation** $\rho$ normalizes this to $[-1, 1]$. Zero covariance does **not** imply independence.

6. **The Law of Iterated Expectation** $E[Y] = E[E[Y|X]]$ is a powerful decomposition tool. Its variance analog $\text{Var}(Y) = E[\text{Var}(Y|X)] + \text{Var}(E[Y|X])$ decomposes total variability into within-group and between-group components.

---

*Previous: [04 - Expectation and Moments](./04_Expectation_and_Moments.md) | Next: [06 - Discrete Distribution Families](./06_Discrete_Distribution_Families.md)*

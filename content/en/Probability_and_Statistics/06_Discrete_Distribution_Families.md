# Discrete Distribution Families

## Learning Objectives

After completing this lesson, you will be able to:

1. State the PMF, mean, variance, and MGF for each of the six major discrete distributions
2. Identify which distribution models a given real-world scenario
3. Apply the Poisson approximation to the Binomial distribution
4. Prove and apply the memoryless property of the Geometric distribution
5. Explain the relationships between these distribution families
6. Simulate each distribution in Python using the standard library

---

## Overview

Discrete probability distributions are the building blocks of stochastic modeling. Rather than specifying a PMF from scratch for every problem, we match our scenario to a named distribution family whose properties are well-understood. This lesson covers the six most important families, their parameters, properties, and interconnections.

---

## Table of Contents

1. [Bernoulli Distribution](#1-bernoulli-distribution)
2. [Binomial Distribution](#2-binomial-distribution)
3. [Poisson Distribution](#3-poisson-distribution)
4. [Geometric Distribution](#4-geometric-distribution)
5. [Negative Binomial Distribution](#5-negative-binomial-distribution)
6. [Hypergeometric Distribution](#6-hypergeometric-distribution)
7. [Relationships Between Distributions](#7-relationships-between-distributions)
8. [Python Examples](#8-python-examples)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. Bernoulli Distribution

### Setup

A single trial with two outcomes: **success** (1) with probability $p$, or **failure** (0) with probability $1-p$.

### Notation

$$X \sim \text{Bernoulli}(p), \quad 0 \leq p \leq 1$$

### PMF

$$p_X(x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

| $x$ | 0 | 1 |
|-----|---|---|
| $P(X=x)$ | $1-p$ | $p$ |

### Properties

| Property | Value |
|----------|-------|
| Mean | $E[X] = p$ |
| Variance | $\text{Var}(X) = p(1-p)$ |
| MGF | $M_X(t) = (1-p) + pe^t$ |

### Use Cases

- Coin flip (fair: $p = 0.5$)
- Default/no-default on a loan
- Defective/non-defective item

---

## 2. Binomial Distribution

### Setup

Count the number of successes in $n$ **independent** Bernoulli trials, each with success probability $p$.

### Notation

$$X \sim \text{Binomial}(n, p)$$

### PMF

$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, 2, \ldots, n$$

### Properties

| Property | Value |
|----------|-------|
| Mean | $E[X] = np$ |
| Variance | $\text{Var}(X) = np(1-p)$ |
| MGF | $M_X(t) = [(1-p) + pe^t]^n$ |

### Derivation of the Mean (via Linearity)

Write $X = \sum_{i=1}^n X_i$ where $X_i \sim \text{Bernoulli}(p)$ are independent. Then:

$$E[X] = \sum_{i=1}^n E[X_i] = np$$

$$\text{Var}(X) = \sum_{i=1}^n \text{Var}(X_i) = np(1-p)$$

### Example

A fair coin is flipped 10 times. What is $P(\text{exactly 3 heads})$?

$$P(X = 3) = \binom{10}{3} (0.5)^3 (0.5)^7 = 120 \cdot \frac{1}{1024} = \frac{120}{1024} \approx 0.1172$$

### Use Cases

- Number of heads in $n$ coin flips
- Number of defective items in a sample (with replacement)
- Number of patients who respond to treatment out of $n$

---

## 3. Poisson Distribution

### Setup

Models the number of events occurring in a fixed interval of time or space, when events occur independently at a constant average rate $\lambda$.

### Notation

$$X \sim \text{Poisson}(\lambda), \quad \lambda > 0$$

### PMF

$$p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

### Properties

| Property | Value |
|----------|-------|
| Mean | $E[X] = \lambda$ |
| Variance | $\text{Var}(X) = \lambda$ |
| MGF | $M_X(t) = e^{\lambda(e^t - 1)}$ |

A distinctive feature: the mean and variance are **equal** ($= \lambda$).

### Poisson Approximation to Binomial

When $n$ is large, $p$ is small, and $\lambda = np$ is moderate:

$$\binom{n}{k} p^k (1-p)^{n-k} \approx \frac{\lambda^k e^{-\lambda}}{k!}$$

**Rule of thumb**: Use the Poisson approximation when $n \geq 20$ and $p \leq 0.05$ (or $n \geq 100$ and $np \leq 10$).

### Example: Poisson Approximation

A batch of 1000 items has a 0.2% defect rate. Approximate $P(\text{exactly 3 defectives})$:

$\lambda = np = 1000 \times 0.002 = 2$

$$P(X = 3) \approx \frac{2^3 e^{-2}}{3!} = \frac{8 \cdot 0.1353}{6} \approx 0.1804$$

### Additive Property

If $X \sim \text{Poisson}(\lambda_1)$ and $Y \sim \text{Poisson}(\lambda_2)$ are independent, then:

$$X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$$

### Use Cases

- Number of emails per hour
- Number of car accidents per day at an intersection
- Number of mutations in a DNA segment
- Number of photons hitting a detector per second

---

## 4. Geometric Distribution

### Setup

Count the number of trials until the **first success** in a sequence of independent Bernoulli trials.

### Notation

$$X \sim \text{Geometric}(p)$$

(Convention: $X$ = trial number of the first success, so $X \in \{1, 2, 3, \ldots\}$.)

### PMF

$$p_X(k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

**Alternative convention**: Some texts define $Y$ = number of failures before first success ($Y = X - 1$, $Y \in \{0, 1, 2, \ldots\}$), giving $p_Y(k) = (1-p)^k p$.

### Properties (first-success convention)

| Property | Value |
|----------|-------|
| Mean | $E[X] = 1/p$ |
| Variance | $\text{Var}(X) = (1-p)/p^2$ |
| MGF | $M_X(t) = \frac{pe^t}{1 - (1-p)e^t}$ for $t < -\ln(1-p)$ |

### The Memoryless Property

The Geometric distribution is the **only** discrete distribution with the memoryless property:

$$P(X > m + n \mid X > m) = P(X > n)$$

**Interpretation**: Given that you have already failed $m$ times, the probability of needing at least $n$ more trials is the same as if you were starting fresh. Past failures provide no information about future success.

*Proof*:

$$P(X > m + n \mid X > m) = \frac{P(X > m + n)}{P(X > m)} = \frac{(1-p)^{m+n}}{(1-p)^m} = (1-p)^n = P(X > n)$$

### Example

A die is rolled repeatedly. Let $X$ = number of rolls until the first 6. Then $X \sim \text{Geometric}(1/6)$.

$$E[X] = 6, \quad P(X > 12 \mid X > 6) = P(X > 6) = (5/6)^6 \approx 0.335$$

### Use Cases

- Number of attempts until first success
- Waiting time for first event in discrete time
- Number of components tested until first defective

---

## 5. Negative Binomial Distribution

### Setup

Count the number of trials until the $r$-th success in a sequence of independent Bernoulli trials.

### Notation

$$X \sim \text{NegBin}(r, p)$$

($X$ = trial number of the $r$-th success, $X \in \{r, r+1, r+2, \ldots\}$.)

### PMF

$$p_X(k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, r+2, \ldots$$

**Intuition**: Among the first $k-1$ trials, exactly $r-1$ are successes (hence $\binom{k-1}{r-1}$), and the $k$-th trial is the $r$-th success (contributing the final factor of $p$).

### Properties

| Property | Value |
|----------|-------|
| Mean | $E[X] = r/p$ |
| Variance | $\text{Var}(X) = r(1-p)/p^2$ |
| MGF | $M_X(t) = \left(\frac{pe^t}{1-(1-p)e^t}\right)^r$ for $t < -\ln(1-p)$ |

### Connection to Geometric

$\text{NegBin}(1, p) = \text{Geometric}(p)$.

If $X_1, X_2, \ldots, X_r$ are independent $\text{Geometric}(p)$ random variables, then:

$$X_1 + X_2 + \cdots + X_r \sim \text{NegBin}(r, p)$$

### Use Cases

- Number of patients screened until finding $r$ eligible participants
- Number of sales calls until closing $r$ deals
- Overdispersed count data (when variance exceeds mean, unlike Poisson)

---

## 6. Hypergeometric Distribution

### Setup

Draw $n$ items **without replacement** from a population of $N$ items containing $K$ successes. Count the number of successes drawn.

### Notation

$$X \sim \text{Hypergeometric}(N, K, n)$$

### PMF

$$p_X(k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}, \quad k = \max(0, n-N+K), \ldots, \min(n, K)$$

### Properties

| Property | Value |
|----------|-------|
| Mean | $E[X] = n \cdot K/N$ |
| Variance | $\text{Var}(X) = n \cdot \frac{K}{N} \cdot \frac{N-K}{N} \cdot \frac{N-n}{N-1}$ |

The factor $\frac{N-n}{N-1}$ is called the **finite population correction** (FPC). When $N \gg n$, the FPC approaches 1 and the Hypergeometric approaches the Binomial.

### Comparison with Binomial

| Feature | Binomial | Hypergeometric |
|---------|----------|----------------|
| Sampling | With replacement | Without replacement |
| Trials | Independent | Dependent |
| Variance | $np(1-p)$ | $np(1-p) \cdot \frac{N-n}{N-1}$ |
| Approximation | -- | Approaches Binomial as $N \to \infty$ |

### Example

A deck has 52 cards, 4 aces. Draw 5 cards without replacement. What is $P(\text{exactly 2 aces})$?

$$P(X = 2) = \frac{\binom{4}{2}\binom{48}{3}}{\binom{52}{5}} = \frac{6 \times 17296}{2598960} = \frac{103776}{2598960} \approx 0.0399$$

### Use Cases

- Quality control sampling from a finite lot
- Card drawing problems
- Capture-recapture in ecology
- Fisher's exact test for $2 \times 2$ tables

---

## 7. Relationships Between Distributions

```
Bernoulli(p)
    |
    | Sum of n independent copies
    v
Binomial(n, p)
    |
    | n -> inf, p -> 0, np = lambda
    v
Poisson(lambda)

Geometric(p)   = NegBin(1, p)
    |
    | Sum of r independent copies
    v
NegBin(r, p)

Hypergeometric(N, K, n)
    |
    | N -> inf with K/N = p
    v
Binomial(n, p)
```

### Summary of Limiting Relationships

| From | To | Condition |
|------|----|-----------|
| Binomial$(n, p)$ | Poisson$(\lambda)$ | $n \to \infty$, $p \to 0$, $np = \lambda$ |
| Hypergeometric$(N, K, n)$ | Binomial$(n, p)$ | $N \to \infty$, $K/N \to p$ |
| Binomial$(n, p)$ | Normal$(np, np(1-p))$ | $n \to \infty$ (CLT, Lesson 11) |

---

## 8. Python Examples

### Simulating Bernoulli and Binomial

```python
import random
import math

def simulate_binomial(n, p, num_trials=100000):
    """Simulate Binomial(n, p) and compare to theoretical values."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        x = sum(1 for _ in range(n) if random.random() < p)
        samples.append(x)

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    print(f"Binomial(n={n}, p={p})")
    print(f"  Theoretical: mean={n*p:.4f}, var={n*p*(1-p):.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")

simulate_binomial(20, 0.3)
```

### Poisson Simulation and Approximation

```python
import random
import math

def simulate_poisson(lam, num_trials=100000):
    """Simulate Poisson(lam) using the inverse transform method."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        L = math.exp(-lam)
        k = 0
        prob = 1.0
        while prob > L:
            k += 1
            prob *= random.random()
        samples.append(k - 1)

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    print(f"Poisson(lambda={lam})")
    print(f"  Theoretical: mean={lam:.4f}, var={lam:.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")
    return samples

samples = simulate_poisson(5.0)
```

### Poisson Approximation to Binomial

```python
import math

def poisson_approximation_comparison(n, p, k_max=15):
    """Compare exact Binomial PMF with Poisson approximation."""
    lam = n * p
    print(f"Binomial(n={n}, p={p}) vs Poisson(lambda={lam})")
    print(f"{'k':>3}  {'Binomial':>12}  {'Poisson':>12}  {'Abs Error':>12}")
    print("-" * 45)

    for k in range(k_max + 1):
        # Exact Binomial
        binom_pmf = (math.comb(n, k) * p**k * (1-p)**(n-k))

        # Poisson approximation
        poisson_pmf = (lam**k * math.exp(-lam)) / math.factorial(k)

        error = abs(binom_pmf - poisson_pmf)
        print(f"{k:3d}  {binom_pmf:12.8f}  {poisson_pmf:12.8f}  {error:12.8f}")

# Good approximation: large n, small p
poisson_approximation_comparison(n=100, p=0.03)
print()
# Poor approximation: small n, large p
poisson_approximation_comparison(n=10, p=0.3)
```

### Geometric Distribution and Memoryless Property

```python
import random

def simulate_geometric(p, num_trials=200000):
    """Simulate Geometric(p) and verify memoryless property."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        k = 1
        while random.random() >= p:  # Keep going until success
            k += 1
        samples.append(k)

    emp_mean = sum(samples) / num_trials
    print(f"Geometric(p={p})")
    print(f"  Theoretical mean = {1/p:.4f}")
    print(f"  Empirical mean   = {emp_mean:.4f}")

    # Verify memoryless property: P(X > m+n | X > m) = P(X > n)
    m, n = 5, 3
    count_gt_m = sum(1 for x in samples if x > m)
    count_gt_m_plus_n = sum(1 for x in samples if x > m + n)
    count_gt_n = sum(1 for x in samples if x > n)

    cond_prob = count_gt_m_plus_n / count_gt_m if count_gt_m > 0 else 0
    uncond_prob = count_gt_n / num_trials

    print(f"\n  Memoryless property (m={m}, n={n}):")
    print(f"  P(X>{m+n} | X>{m}) = {cond_prob:.4f}")
    print(f"  P(X>{n})           = {uncond_prob:.4f}")
    print(f"  Theoretical        = {(1-p)**n:.4f}")

simulate_geometric(p=1/6)
```

### Negative Binomial

```python
import random

def simulate_negative_binomial(r, p, num_trials=100000):
    """Simulate NegBin(r, p) as sum of r Geometric(p) variables."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        total = 0
        for _ in range(r):
            # One Geometric(p) sample
            k = 1
            while random.random() >= p:
                k += 1
            total += k
        samples.append(total)

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    print(f"NegBin(r={r}, p={p})")
    print(f"  Theoretical: mean={r/p:.4f}, var={r*(1-p)/p**2:.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")

simulate_negative_binomial(r=5, p=0.4)
```

### Hypergeometric

```python
import random
import math

def hypergeometric_pmf(k, N, K, n):
    """Compute P(X=k) for Hypergeometric(N, K, n)."""
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)

def simulate_hypergeometric(N, K, n, num_trials=100000):
    """Simulate Hypergeometric by drawing without replacement."""
    random.seed(42)
    population = [1] * K + [0] * (N - K)  # 1 = success, 0 = failure
    samples = []

    for _ in range(num_trials):
        draw = random.sample(population, n)
        samples.append(sum(draw))

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    theo_mean = n * K / N
    theo_var = n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))

    print(f"Hypergeometric(N={N}, K={K}, n={n})")
    print(f"  Theoretical: mean={theo_mean:.4f}, var={theo_var:.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")

    # PMF comparison
    print(f"\n  {'k':>3}  {'Exact PMF':>12}  {'Empirical':>12}")
    print("  " + "-" * 30)
    from collections import Counter
    counts = Counter(samples)
    k_min = max(0, n - (N - K))
    k_max = min(n, K)
    for k in range(k_min, k_max + 1):
        exact = hypergeometric_pmf(k, N, K, n)
        emp = counts.get(k, 0) / num_trials
        print(f"  {k:3d}  {exact:12.6f}  {emp:12.6f}")

# Card example: 52 cards, 4 aces, draw 5
simulate_hypergeometric(N=52, K=4, n=5)
```

### Complete Distribution Summary

```python
import math

def distribution_summary():
    """Print a summary table of all discrete distributions."""
    distributions = [
        ("Bernoulli(p)", "p", "p(1-p)", "{0,1}"),
        ("Binomial(n,p)", "np", "np(1-p)", "{0,...,n}"),
        ("Poisson(lam)", "lam", "lam", "{0,1,2,...}"),
        ("Geometric(p)", "1/p", "(1-p)/p^2", "{1,2,3,...}"),
        ("NegBin(r,p)", "r/p", "r(1-p)/p^2", "{r,r+1,...}"),
        ("Hypergeo(N,K,n)", "nK/N", "nK(N-K)(N-n)/[N^2(N-1)]", "{0,...,min(n,K)}"),
    ]

    header = f"{'Distribution':<20} {'Mean':<12} {'Variance':<18} {'Support':<15}"
    print(header)
    print("-" * len(header))
    for name, mean, var, support in distributions:
        print(f"{name:<20} {mean:<12} {var:<18} {support:<15}")

distribution_summary()
```

---

## 9. Key Takeaways

1. **Bernoulli** is the atomic building block: a single binary trial. Everything else builds on it.

2. **Binomial** counts successes in $n$ independent trials. Its mean $np$ and variance $np(1-p)$ follow elegantly from linearity of expectation and independence.

3. **Poisson** models rare event counts. Its defining feature is $E[X] = \text{Var}(X) = \lambda$. It approximates Binomial when $n$ is large and $p$ is small.

4. **Geometric** models waiting time until first success. Its memoryless property ($P(X > m+n \mid X > m) = P(X > n)$) makes it the discrete analog of the exponential distribution.

5. **Negative Binomial** generalizes Geometric to waiting for the $r$-th success. It equals the sum of $r$ independent Geometric random variables.

6. **Hypergeometric** handles sampling without replacement from a finite population. It converges to Binomial as the population size grows, with the finite population correction factor bridging the gap.

7. **Choosing the right distribution** depends on the problem structure:
   - Fixed $n$ trials, independent, with replacement -> **Binomial**
   - Count of rare events in fixed interval -> **Poisson**
   - Waiting for first/r-th success -> **Geometric / Negative Binomial**
   - Sampling without replacement -> **Hypergeometric**

---

*Previous: [05 - Joint Distributions](./05_Joint_Distributions.md) | Next: [07 - Continuous Distribution Families](./07_Continuous_Distribution_Families.md)*

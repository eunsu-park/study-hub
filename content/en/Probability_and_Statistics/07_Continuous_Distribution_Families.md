# Continuous Distribution Families

**Previous**: [Discrete Distribution Families](./06_Discrete_Distribution_Families.md) | **Next**: [Transformations of Random Variables](./08_Transformations_of_Random_Variables.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. State the PDF, CDF, mean, variance, and MGF for each major continuous distribution
2. Apply the Uniform, Normal, Exponential, and Gamma families to model real-world phenomena
3. Explain the memoryless property of the Exponential distribution
4. Describe the Beta distribution as a natural prior for proportions
5. Derive how Chi-squared, Student's t, and F distributions arise from the Normal
6. Map the relationships connecting these distribution families

---

Continuous distributions assign probabilities to intervals of real numbers rather than individual points. This lesson surveys the most important continuous families, each characterised by a probability density function (PDF) $f(x)$ satisfying $f(x) \ge 0$ and $\int_{-\infty}^{\infty} f(x)\,dx = 1$.

---

## 1. Uniform Distribution — $X \sim \text{Uniform}(a, b)$

### 1.1 Definition and PDF

The simplest continuous distribution assigns equal density over a finite interval $[a, b]$.

$$f(x) = \frac{1}{b - a}, \quad a \le x \le b$$

### 1.2 CDF

$$F(x) = \begin{cases} 0 & x < a \\ \frac{x - a}{b - a} & a \le x \le b \\ 1 & x > b \end{cases}$$

### 1.3 Moments and MGF

| Property | Value |
|----------|-------|
| Mean | $E[X] = \frac{a + b}{2}$ |
| Variance | $\text{Var}(X) = \frac{(b - a)^2}{12}$ |
| MGF | $M_X(t) = \frac{e^{tb} - e^{ta}}{t(b - a)}$ for $t \ne 0$ |

### 1.4 Use Cases

- Modelling complete ignorance about a parameter's location
- Random number generation: most pseudo-random generators produce $\text{Uniform}(0,1)$
- Inverse-transform sampling starts with $U \sim \text{Uniform}(0,1)$

---

## 2. Normal (Gaussian) Distribution — $X \sim N(\mu, \sigma^2)$

### 2.1 PDF

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad -\infty < x < \infty$$

### 2.2 Standard Normal

When $\mu = 0$ and $\sigma^2 = 1$, write $Z \sim N(0,1)$. Any normal variable can be standardised:

$$Z = \frac{X - \mu}{\sigma}$$

The CDF of the standard normal is denoted $\Phi(z)$ and has no closed-form expression.

### 2.3 The 68-95-99.7 Rule

For a normal distribution:

- $P(\mu - \sigma \le X \le \mu + \sigma) \approx 0.6827$
- $P(\mu - 2\sigma \le X \le \mu + 2\sigma) \approx 0.9545$
- $P(\mu - 3\sigma \le X \le \mu + 3\sigma) \approx 0.9973$

This empirical rule provides a quick way to assess probabilities without tables.

### 2.4 CDF

$$F(x) = \Phi\!\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]$$

### 2.5 Moments and MGF

| Property | Value |
|----------|-------|
| Mean | $E[X] = \mu$ |
| Variance | $\text{Var}(X) = \sigma^2$ |
| MGF | $M_X(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$ |

### 2.6 Key Properties

- **Closure under linear transformation**: If $X \sim N(\mu, \sigma^2)$ then $aX + b \sim N(a\mu + b,\, a^2\sigma^2)$.
- **Sum of independents**: If $X_i \sim N(\mu_i, \sigma_i^2)$ independently, then $\sum X_i \sim N\!\left(\sum \mu_i, \sum \sigma_i^2\right)$.
- The normal is the **maximum entropy** distribution for a given mean and variance.

---

## 3. Exponential Distribution — $X \sim \text{Exp}(\lambda)$

### 3.1 PDF and CDF

$$f(x) = \lambda e^{-\lambda x}, \quad x \ge 0$$

$$F(x) = 1 - e^{-\lambda x}, \quad x \ge 0$$

Here $\lambda > 0$ is the rate parameter. Some texts parameterise by the mean $\beta = 1/\lambda$.

### 3.2 Moments and MGF

| Property | Value |
|----------|-------|
| Mean | $E[X] = 1/\lambda$ |
| Variance | $\text{Var}(X) = 1/\lambda^2$ |
| MGF | $M_X(t) = \frac{\lambda}{\lambda - t}$ for $t < \lambda$ |

### 3.3 Memoryless Property

The exponential is the **only** continuous distribution that is memoryless:

$$P(X > s + t \mid X > s) = P(X > t) \quad \text{for all } s, t \ge 0$$

This means the remaining lifetime does not depend on how long the process has already lasted.

### 3.4 Connection to Poisson Process

If events occur according to a Poisson process with rate $\lambda$, then the waiting time between successive events follows $\text{Exp}(\lambda)$. Equivalently, the number of events in a time interval of length $t$ is $\text{Poisson}(\lambda t)$.

---

## 4. Gamma Distribution — $X \sim \text{Gamma}(\alpha, \beta)$

### 4.1 PDF

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0$$

where $\alpha > 0$ is the shape parameter, $\beta > 0$ is the rate parameter, and $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\,dt$ is the gamma function.

### 4.2 CDF

No closed-form expression in general; computed via the lower incomplete gamma function:

$$F(x) = \frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)}$$

### 4.3 Moments and MGF

| Property | Value |
|----------|-------|
| Mean | $E[X] = \alpha / \beta$ |
| Variance | $\text{Var}(X) = \alpha / \beta^2$ |
| MGF | $M_X(t) = \left(\frac{\beta}{\beta - t}\right)^\alpha$ for $t < \beta$ |

### 4.4 Special Cases

- **Exponential**: $\text{Gamma}(1, \lambda) = \text{Exp}(\lambda)$
- **Erlang**: When $\alpha = n$ is a positive integer, the distribution is called the Erlang distribution. It models the waiting time for the $n$-th event in a Poisson process.
- **Chi-squared**: $\text{Gamma}(k/2, 1/2) = \chi^2(k)$

### 4.5 Additive Property

If $X_1 \sim \text{Gamma}(\alpha_1, \beta)$ and $X_2 \sim \text{Gamma}(\alpha_2, \beta)$ are independent (same rate), then:

$$X_1 + X_2 \sim \text{Gamma}(\alpha_1 + \alpha_2, \beta)$$

---

## 5. Beta Distribution — $X \sim \text{Beta}(\alpha, \beta)$

### 5.1 PDF

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 < x < 1$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$ is the beta function.

### 5.2 Moments

| Property | Value |
|----------|-------|
| Mean | $E[X] = \frac{\alpha}{\alpha + \beta}$ |
| Variance | $\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

### 5.3 Special Cases and Shapes

- $\text{Beta}(1, 1) = \text{Uniform}(0, 1)$
- $\alpha = \beta$: symmetric about $1/2$
- $\alpha > \beta$: skewed left (mass towards 1)
- $\alpha < \beta$: skewed right (mass towards 0)
- $\alpha, \beta < 1$: U-shaped (mass at both extremes)

### 5.4 Use Cases

The Beta distribution is the **conjugate prior** for the Bernoulli and Binomial likelihoods. If the prior is $p \sim \text{Beta}(\alpha, \beta)$ and we observe $k$ successes in $n$ trials, the posterior is:

$$p \mid \text{data} \sim \text{Beta}(\alpha + k, \beta + n - k)$$

This makes it the standard choice for modelling unknown proportions in Bayesian statistics.

---

## 6. Chi-Squared Distribution — $X \sim \chi^2(k)$

### 6.1 Definition

If $Z_1, Z_2, \ldots, Z_k$ are independent standard normal variables, then:

$$X = Z_1^2 + Z_2^2 + \cdots + Z_k^2 \sim \chi^2(k)$$

The parameter $k$ is called the **degrees of freedom**.

### 6.2 PDF

$$f(x) = \frac{1}{2^{k/2}\,\Gamma(k/2)} x^{k/2 - 1} e^{-x/2}, \quad x > 0$$

This is simply $\text{Gamma}(k/2, 1/2)$.

### 6.3 Moments and MGF

| Property | Value |
|----------|-------|
| Mean | $E[X] = k$ |
| Variance | $\text{Var}(X) = 2k$ |
| MGF | $M_X(t) = (1 - 2t)^{-k/2}$ for $t < 1/2$ |

### 6.4 Additive Property

If $X_1 \sim \chi^2(k_1)$ and $X_2 \sim \chi^2(k_2)$ are independent, then $X_1 + X_2 \sim \chi^2(k_1 + k_2)$.

### 6.5 Applications

- Testing goodness-of-fit (Pearson's chi-squared test)
- Constructing confidence intervals for the variance of a normal population
- Sample variance: if $X_i \sim N(\mu, \sigma^2)$, then $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$

---

## 7. Student's t-Distribution — $T \sim t(\nu)$

### 7.1 Definition

If $Z \sim N(0,1)$ and $V \sim \chi^2(\nu)$ are independent, then:

$$T = \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)$$

The parameter $\nu$ is the degrees of freedom.

### 7.2 PDF

$$f(t) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-(\nu+1)/2}, \quad -\infty < t < \infty$$

### 7.3 Moments

| Property | Value |
|----------|-------|
| Mean | $E[T] = 0$ (for $\nu > 1$) |
| Variance | $\text{Var}(T) = \frac{\nu}{\nu - 2}$ (for $\nu > 2$) |

The MGF exists only for $\nu > 1$ and has no simple closed form.

### 7.4 Key Properties

- **Heavy tails**: The $t$-distribution has heavier tails than the standard normal, making extreme values more likely.
- **Convergence**: As $\nu \to \infty$, $t(\nu) \to N(0,1)$. In practice, for $\nu \ge 30$ the $t$ is well approximated by the standard normal.
- **Application**: Used when estimating the mean of a normal population with **unknown variance** and small sample sizes.

---

## 8. F-Distribution — $F \sim F(d_1, d_2)$

### 8.1 Definition

If $U \sim \chi^2(d_1)$ and $V \sim \chi^2(d_2)$ are independent, then:

$$F = \frac{U/d_1}{V/d_2} \sim F(d_1, d_2)$$

### 8.2 PDF

$$f(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x\,B(d_1/2, d_2/2)}, \quad x > 0$$

### 8.3 Moments

| Property | Value |
|----------|-------|
| Mean | $E[F] = \frac{d_2}{d_2 - 2}$ (for $d_2 > 2$) |
| Variance | $\text{Var}(F) = \frac{2d_2^2(d_1 + d_2 - 2)}{d_1(d_2-2)^2(d_2-4)}$ (for $d_2 > 4$) |

### 8.4 Connection to t-Distribution

If $T \sim t(\nu)$, then $T^2 \sim F(1, \nu)$.

### 8.5 Applications

- **ANOVA** (Analysis of Variance): comparing means across multiple groups
- **Regression**: testing whether a set of predictors is jointly significant
- Comparing two population variances

---

## 9. Relationships Between Distribution Families

Understanding how distributions connect is essential for both theory and practice.

```
Bernoulli(p) --sum--> Binomial(n,p) --CLT--> Normal(np, np(1-p))
                                                   |
Poisson(λ) --waiting--> Exponential(λ) = Gamma(1,λ)  |
                                  |                    |
                             Gamma(α,β) <-- sum of Exp  |
                                  |                    |
                           Chi-squared(k) = Gamma(k/2, 1/2)
                                  |                    |
                                  +-----> t(ν) = Z / sqrt(χ²/ν)
                                  |
                                  +-----> F(d1,d2) = (χ²₁/d1) / (χ²₂/d2)

Uniform(0,1) --inverse CDF--> any distribution
Beta(1,1) = Uniform(0,1)
```

Key relationships in summary:

1. $\text{Exp}(\lambda)$ is $\text{Gamma}(1, \lambda)$
2. $\chi^2(k)$ is $\text{Gamma}(k/2, 1/2)$
3. Sum of independent $\chi^2$ is $\chi^2$ (additive degrees of freedom)
4. $t(\nu)$ involves a ratio of Normal to $\sqrt{\chi^2/\nu}$
5. $F(d_1, d_2)$ is a ratio of two independent $\chi^2$ variables (each divided by df)
6. As $\nu \to \infty$, $t(\nu) \to N(0,1)$
7. $\text{Beta}(1,1) = \text{Uniform}(0,1)$

---

## 10. Python Examples

### 10.1 Plotting PDFs of Key Distributions

```python
import math

def normal_pdf(x, mu=0.0, sigma=1.0):
    """Standard or general normal PDF."""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    return coeff * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def exponential_pdf(x, lam=1.0):
    """Exponential PDF with rate parameter lambda."""
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)

def uniform_pdf(x, a=0.0, b=1.0):
    """Uniform PDF on [a, b]."""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

# Evaluate normal PDF at several points
xs = [mu * 0.5 for mu in range(-8, 9)]  # -4.0 to 4.0
for x in xs:
    print(f"  N(0,1) at x={x:5.1f}: {normal_pdf(x):.6f}")
```

### 10.2 Verifying the 68-95-99.7 Rule by Simulation

```python
import random
import math

random.seed(42)
n = 100_000
mu, sigma = 5.0, 2.0
samples = [random.gauss(mu, sigma) for _ in range(n)]

within_1 = sum(1 for x in samples if abs(x - mu) <= sigma) / n
within_2 = sum(1 for x in samples if abs(x - mu) <= 2 * sigma) / n
within_3 = sum(1 for x in samples if abs(x - mu) <= 3 * sigma) / n

print(f"Within 1 sigma: {within_1:.4f}  (expected ~0.6827)")
print(f"Within 2 sigma: {within_2:.4f}  (expected ~0.9545)")
print(f"Within 3 sigma: {within_3:.4f}  (expected ~0.9973)")
```

### 10.3 Demonstrating the Memoryless Property

```python
import random

random.seed(123)
lam = 0.5
n = 200_000
samples = [random.expovariate(lam) for _ in range(n)]

s = 2.0
t = 3.0

# P(X > s + t | X > s) should equal P(X > t)
exceed_s = [x for x in samples if x > s]
conditional = sum(1 for x in exceed_s if x > s + t) / len(exceed_s)
marginal = sum(1 for x in samples if x > t) / n

print(f"P(X > {s+t} | X > {s}) = {conditional:.4f}")
print(f"P(X > {t})             = {marginal:.4f}")
print(f"Theoretical P(X > {t}) = {math.exp(-lam * t):.4f}")
```

### 10.4 Gamma Distribution: Sum of Exponentials

```python
import random
import math

random.seed(99)
lam = 2.0
alpha = 5  # shape = number of exponentials to sum
n = 100_000

# Sum of alpha independent Exp(lam) is Gamma(alpha, lam)
gamma_samples = []
for _ in range(n):
    total = sum(random.expovariate(lam) for _ in range(alpha))
    gamma_samples.append(total)

sample_mean = sum(gamma_samples) / n
sample_var = sum((x - sample_mean) ** 2 for x in gamma_samples) / (n - 1)

print(f"Sample mean:     {sample_mean:.4f}  (theoretical: {alpha / lam:.4f})")
print(f"Sample variance: {sample_var:.4f}  (theoretical: {alpha / lam**2:.4f})")
```

### 10.5 Chi-Squared from Squared Normals

```python
import random

random.seed(77)
k = 6  # degrees of freedom
n = 100_000

chi2_samples = []
for _ in range(n):
    val = sum(random.gauss(0, 1) ** 2 for _ in range(k))
    chi2_samples.append(val)

mean_est = sum(chi2_samples) / n
var_est = sum((x - mean_est) ** 2 for x in chi2_samples) / (n - 1)

print(f"Chi-squared({k}):")
print(f"  Sample mean:     {mean_est:.4f}  (theoretical: {k})")
print(f"  Sample variance: {var_est:.4f}  (theoretical: {2 * k})")
```

### 10.6 Student's t from Normal and Chi-Squared

```python
import random
import math

random.seed(55)
nu = 5  # degrees of freedom
n = 100_000

t_samples = []
for _ in range(n):
    z = random.gauss(0, 1)
    v = sum(random.gauss(0, 1) ** 2 for _ in range(nu))
    t_val = z / math.sqrt(v / nu)
    t_samples.append(t_val)

mean_est = sum(t_samples) / n
var_est = sum((x - mean_est) ** 2 for x in t_samples) / (n - 1)
theoretical_var = nu / (nu - 2) if nu > 2 else float('inf')

print(f"Student's t({nu}):")
print(f"  Sample mean:     {mean_est:.4f}  (theoretical: 0)")
print(f"  Sample variance: {var_est:.4f}  (theoretical: {theoretical_var:.4f})")
```

---

## Key Takeaways

1. **Normal distribution** is the cornerstone of statistics; its bell curve arises naturally via the Central Limit Theorem.
2. **Exponential** is the continuous analogue of the geometric distribution and is uniquely memoryless.
3. **Gamma** generalises the Exponential; the Chi-squared is a special Gamma.
4. **Beta** lives on $[0, 1]$ and is the conjugate prior for binomial data, making it central to Bayesian analysis.
5. **Chi-squared, t, and F** distributions are derived from the normal and underpin classical hypothesis testing and confidence intervals.
6. Many distributions are related through sums, ratios, limits, or special parameter choices. Understanding these connections helps you choose the right model and derive sampling distributions.

---

*Next lesson: [Transformations of Random Variables](./08_Transformations_of_Random_Variables.md)*

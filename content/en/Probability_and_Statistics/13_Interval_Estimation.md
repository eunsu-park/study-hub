# Interval Estimation

**Previous**: [Point Estimation](./12_Point_Estimation.md) | **Next**: [Hypothesis Testing](./14_Hypothesis_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why point estimates alone are insufficient and intervals are needed
2. Define and correctly interpret confidence intervals in the frequentist framework
3. Construct confidence intervals for the mean of a normal distribution (known and unknown variance)
4. Derive confidence intervals for variance using the chi-squared distribution
5. Compute confidence intervals for proportions using Wald, Wilson, and Clopper-Pearson methods
6. Build intervals for the difference of means and ratio of variances
7. Determine the required sample size for a desired margin of error
8. Apply bootstrap methods (percentile, BCa) for confidence interval construction

---

A point estimate $\hat{\theta}$ gives a single best guess for a parameter, but it tells us nothing about the uncertainty of that guess. Interval estimation addresses this by providing a range of plausible values along with a measure of confidence that the true parameter lies within that range.

---

## 1. Limitations of Point Estimates

A point estimate $\hat{\theta}$ of a parameter $\theta$ is a single number computed from sample data. While useful, it has key shortcomings:

- **No measure of precision**: Two samples may yield identical point estimates but with vastly different variabilities.
- **Sampling variability**: Different samples produce different estimates. Without a range, we cannot express how much the estimate might vary.
- **Decision-making risk**: Acting on a single number without understanding its uncertainty can lead to poor decisions.

**Example**: Suppose we estimate the average response time of a server as $\hat{\mu} = 120$ ms. Is the true mean likely between 115 and 125 ms, or between 50 and 190 ms? The point estimate alone cannot answer this.

---

## 2. Confidence Interval: Definition and Interpretation

### 2.1 Definition

A **confidence interval** (CI) at confidence level $1 - \alpha$ is a random interval $[L(X), U(X)]$ such that:

$$P(L(X) \leq \theta \leq U(X)) = 1 - \alpha$$

where $L(X)$ and $U(X)$ are statistics (functions of the sample data), and $\theta$ is the fixed but unknown parameter.

### 2.2 Frequentist Interpretation

The correct interpretation is about the **procedure**, not the specific interval:

> If we were to repeat the sampling process many times and compute a 95% CI each time, approximately 95% of those intervals would contain the true parameter $\theta$.

A specific computed interval, say $[112, 128]$, either contains $\theta$ or it does not. We do **not** say "there is a 95% probability that $\theta$ lies in $[112, 128]$."

### 2.3 Width and Precision

The width of a CI depends on:

- **Confidence level** $1 - \alpha$: higher confidence yields wider intervals.
- **Sample size** $n$: larger samples yield narrower intervals.
- **Population variability** $\sigma$: more variability yields wider intervals.

---

## 3. Pivot Quantities

A **pivot quantity** is a function $Q(X, \theta)$ of the data and the parameter whose distribution is completely known (does not depend on $\theta$).

**General method**: If $Q(X, \theta)$ is a pivot and $P(a \leq Q \leq b) = 1 - \alpha$, then inverting the inequality gives a confidence interval for $\theta$.

**Example**: If $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with $\sigma$ known, then:

$$Q = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)$$

This is a pivot because its distribution does not depend on $\mu$. Inverting $P(-z_{\alpha/2} \leq Q \leq z_{\alpha/2}) = 1 - \alpha$ gives:

$$\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

---

## 4. CI for Normal Mean

### 4.1 Known Variance (z-Interval)

When $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with $\sigma^2$ known:

$$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

where $z_{\alpha/2}$ is the upper $\alpha/2$ quantile of the standard normal. For a 95% CI, $z_{0.025} = 1.96$.

### 4.2 Unknown Variance (t-Interval)

When $\sigma^2$ is unknown, we replace it with the sample standard deviation $S$. The pivot becomes:

$$T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}$$

The $(1-\alpha)$ CI is:

$$\bar{X} \pm t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}$$

The t-distribution has heavier tails than the standard normal, producing wider intervals that account for the additional uncertainty of estimating $\sigma$.

```python
import math
import statistics

def z_confidence_interval(data, sigma, confidence=0.95):
    """CI for mean with known population std dev."""
    n = len(data)
    x_bar = sum(data) / n
    # Approximate z-value using inverse error function
    # For common levels: 0.90->1.645, 0.95->1.960, 0.99->2.576
    z_values = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_values.get(confidence, 1.960)
    margin = z * sigma / math.sqrt(n)
    return (x_bar - margin, x_bar + margin)

def t_confidence_interval(data, confidence=0.95):
    """CI for mean with unknown variance using t-distribution."""
    n = len(data)
    x_bar = statistics.mean(data)
    s = statistics.stdev(data)
    # t critical values for common df (approximation for moderate n)
    # For large n, t approaches z
    from math import sqrt
    # Use normal approximation for demonstration
    z_values = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    # Adjust for small samples with Satterthwaite-like correction
    df = n - 1
    z = z_values.get(confidence, 1.960)
    # Simple t-approximation: t ≈ z + (z + z^3)/(4*df) for moderate df
    t_approx = z + (z + z**3) / (4 * df) if df < 30 else z
    margin = t_approx * s / sqrt(n)
    return (x_bar - margin, x_bar + margin)

# Example: Server response times (ms)
data = [118, 122, 131, 115, 127, 124, 119, 130, 126, 121]
print("Sample mean:", statistics.mean(data))
print("Sample std:", statistics.stdev(data))
print("t-interval (95%):", t_confidence_interval(data, 0.95))
```

---

## 5. CI for Normal Variance

If $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$, the pivot for $\sigma^2$ is:

$$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$$

The $(1 - \alpha)$ CI for $\sigma^2$ is:

$$\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}, \quad \frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}\right]$$

Note the asymmetry: the chi-squared distribution is right-skewed, so the CI for variance is not symmetric about $S^2$.

---

## 6. CI for Proportions

Let $X \sim \text{Binomial}(n, p)$ and $\hat{p} = X/n$.

### 6.1 Wald Interval

The simplest (but often inaccurate for small $n$ or extreme $p$):

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**Problem**: Coverage can be well below the nominal level, especially when $p$ is near 0 or 1, or $n$ is small.

### 6.2 Wilson Score Interval

A more reliable alternative that inverts the score test:

$$\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

This has much better coverage properties than the Wald interval.

### 6.3 Clopper-Pearson (Exact) Interval

Based on inverting the binomial test. It guarantees at least $1-\alpha$ coverage (conservative):

$$\left[B\left(\frac{\alpha}{2}; x, n-x+1\right), \quad B\left(1 - \frac{\alpha}{2}; x+1, n-x\right)\right]$$

where $B(q; a, b)$ is the $q$-th quantile of the $\text{Beta}(a, b)$ distribution.

```python
import math

def wald_interval(x, n, confidence=0.95):
    """Wald confidence interval for proportion."""
    p_hat = x / n
    z = 1.960 if confidence == 0.95 else 1.645
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n)
    return (max(0, p_hat - margin), min(1, p_hat + margin))

def wilson_interval(x, n, confidence=0.95):
    """Wilson score confidence interval for proportion."""
    p_hat = x / n
    z = 1.960 if confidence == 0.95 else 1.645
    z2 = z ** 2
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2)) / denom
    return (center - spread, center + spread)

# Example: 23 successes out of 100 trials
x, n = 23, 100
print(f"Wald:   {wald_interval(x, n)}")
print(f"Wilson: {wilson_interval(x, n)}")

# Small sample: 2 out of 10 -- Wald breaks down
x2, n2 = 2, 10
print(f"\nSmall sample (x=2, n=10):")
print(f"Wald:   {wald_interval(x2, n2)}")
print(f"Wilson: {wilson_interval(x2, n2)}")
```

---

## 7. CI for Difference of Means and Ratio of Variances

### 7.1 Difference of Two Means (Independent Samples)

For independent samples from $N(\mu_1, \sigma_1^2)$ and $N(\mu_2, \sigma_2^2)$:

**Equal variances** (pooled t-interval):

$$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\, n_1+n_2-2} \cdot S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$

where $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$.

**Unequal variances** (Welch's t-interval):

$$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\, \nu} \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}$$

where the degrees of freedom $\nu$ are given by the Welch-Satterthwaite formula.

### 7.2 Ratio of Two Variances

The pivot is:

$$F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} \sim F_{n_1-1, n_2-1}$$

The CI for $\sigma_1^2 / \sigma_2^2$ is:

$$\left[\frac{S_1^2}{S_2^2} \cdot \frac{1}{F_{\alpha/2,\, n_1-1,\, n_2-1}}, \quad \frac{S_1^2}{S_2^2} \cdot \frac{1}{F_{1-\alpha/2,\, n_1-1,\, n_2-1}}\right]$$

---

## 8. Sample Size Determination

### 8.1 For Estimating a Mean

To achieve a margin of error $E$ at confidence level $1-\alpha$ with known $\sigma$:

$$n \geq \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2$$

**Example**: If $\sigma = 15$, desired margin $E = 3$, and $\alpha = 0.05$:

$$n \geq \left(\frac{1.96 \times 15}{3}\right)^2 = (9.8)^2 = 96.04 \implies n = 97$$

### 8.2 For Estimating a Proportion

$$n \geq \left(\frac{z_{\alpha/2}}{E}\right)^2 \hat{p}(1-\hat{p})$$

If no prior estimate of $p$ is available, use $\hat{p} = 0.5$ (worst case):

$$n \geq \frac{z_{\alpha/2}^2}{4E^2}$$

```python
import math

def sample_size_mean(sigma, margin, confidence=0.95):
    """Minimum sample size for estimating a mean."""
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence]
    n = (z * sigma / margin) ** 2
    return math.ceil(n)

def sample_size_proportion(margin, p_hat=0.5, confidence=0.95):
    """Minimum sample size for estimating a proportion."""
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence]
    n = (z ** 2) * p_hat * (1 - p_hat) / (margin ** 2)
    return math.ceil(n)

print("Mean (sigma=15, E=3):", sample_size_mean(15, 3))
print("Proportion (E=0.03):", sample_size_proportion(0.03))
print("Proportion (E=0.05, p=0.2):", sample_size_proportion(0.05, p_hat=0.2))
```

---

## 9. Bootstrap Confidence Intervals

When the sampling distribution of an estimator is difficult to derive analytically, **bootstrap** methods offer a computational alternative.

### 9.1 Bootstrap Principle

1. From the original sample of size $n$, draw $B$ bootstrap samples (sampling with replacement).
2. Compute the statistic of interest $\hat{\theta}^*_b$ for each bootstrap sample $b = 1, \ldots, B$.
3. Use the empirical distribution of $\{\hat{\theta}^*_1, \ldots, \hat{\theta}^*_B\}$ to estimate the sampling distribution.

### 9.2 Percentile Method

The $(1-\alpha)$ percentile bootstrap CI is simply:

$$[\hat{\theta}^*_{(\alpha/2)}, \quad \hat{\theta}^*_{(1-\alpha/2)}]$$

where $\hat{\theta}^*_{(q)}$ is the $q$-th quantile of the bootstrap distribution.

### 9.3 BCa (Bias-Corrected and Accelerated)

The BCa method adjusts for bias and skewness in the bootstrap distribution, providing better coverage than the simple percentile method. It uses a bias-correction factor $z_0$ and an acceleration factor $a$.

```python
import random
import statistics

def bootstrap_percentile_ci(data, statistic_fn, B=10000, confidence=0.95, seed=42):
    """Bootstrap percentile confidence interval.

    Args:
        data: list of observations
        statistic_fn: callable that computes the statistic from a sample
        B: number of bootstrap replicates
        confidence: confidence level
        seed: random seed for reproducibility
    """
    random.seed(seed)
    n = len(data)
    alpha = 1 - confidence

    boot_stats = []
    for _ in range(B):
        boot_sample = random.choices(data, k=n)
        boot_stats.append(statistic_fn(boot_sample))

    boot_stats.sort()
    lower_idx = int(B * alpha / 2)
    upper_idx = int(B * (1 - alpha / 2))
    return (boot_stats[lower_idx], boot_stats[upper_idx])

# Example: CI for the median
data = [2.3, 4.1, 1.8, 5.7, 3.2, 6.4, 2.9, 4.8, 3.5, 7.1,
        2.1, 5.3, 3.8, 4.5, 6.0, 1.9, 3.1, 5.5, 4.2, 3.7]

ci_mean = bootstrap_percentile_ci(data, statistics.mean)
ci_median = bootstrap_percentile_ci(data, statistics.median)
print(f"Bootstrap CI for mean:   {ci_mean}")
print(f"Bootstrap CI for median: {ci_median}")
print(f"Sample mean:   {statistics.mean(data):.3f}")
print(f"Sample median: {statistics.median(data):.3f}")
```

---

## 10. Key Takeaways

| Concept | Key Point |
|---|---|
| Point estimate limitation | Gives no measure of uncertainty |
| Confidence level $1-\alpha$ | Proportion of CIs that capture $\theta$ in repeated sampling |
| Pivot quantity | Function of data and parameter with a known distribution |
| z-interval | Use when $\sigma$ is known; based on standard normal |
| t-interval | Use when $\sigma$ is unknown; wider due to extra uncertainty |
| Chi-squared CI | For variance; asymmetric interval |
| Wald vs Wilson | Wilson is preferred for proportions, especially small $n$ |
| Sample size | $n \propto (z \cdot \sigma / E)^2$; increases quadratically as $E$ decreases |
| Bootstrap CI | Non-parametric; useful when analytical CIs are unavailable |

**Common pitfall**: Saying "there is a 95% probability that $\theta$ is in this interval." The correct frequentist statement is about the long-run coverage of the **procedure**.

---

**Previous**: [Point Estimation](./12_Point_Estimation.md) | **Next**: [Hypothesis Testing](./14_Hypothesis_Testing.md)

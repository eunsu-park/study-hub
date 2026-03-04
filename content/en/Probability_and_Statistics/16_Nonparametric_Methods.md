# Nonparametric Methods

**Previous**: [Bayesian Inference](./15_Bayesian_Inference.md) | **Next**: [Regression and ANOVA](./17_Regression_and_ANOVA.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain when and why nonparametric methods are preferable to parametric ones
2. Apply the sign test for a population median
3. Perform the Wilcoxon signed-rank test for paired data
4. Use the Mann-Whitney U test for comparing two independent groups
5. Conduct the Kruskal-Wallis test for multiple groups
6. Compute Spearman's rank correlation coefficient
7. Understand Kernel Density Estimation and bandwidth selection
8. Implement permutation tests for exact and approximate inference
9. Apply bootstrap methods for general inference
10. Perform the Kolmogorov-Smirnov test for distributional comparisons

---

Nonparametric methods make minimal assumptions about the underlying probability distribution of the data. They are especially valuable when distributional assumptions (e.g., normality) are questionable, when data are ordinal, or when the sample size is small and distribution shape cannot be verified.

---

## 1. Why Nonparametric?

### 1.1 Limitations of Parametric Methods

Parametric tests (t-test, F-test, etc.) assume specific distributional forms. When these assumptions are violated:

- Type I error rates may be inflated or deflated.
- Power may drop substantially.
- Confidence intervals may have incorrect coverage.

### 1.2 Advantages of Nonparametric Methods

- **Distribution-free**: Valid under weaker assumptions.
- **Robust**: Less sensitive to outliers and heavy tails.
- **Flexible**: Can handle ordinal data and non-standard distributions.
- **Exact tests available**: For small samples, exact p-values can be computed.

### 1.3 Trade-offs

- **Lower power** when parametric assumptions actually hold (asymptotic relative efficiency of Wilcoxon vs. t-test is $3/\pi \approx 0.955$ for normal data).
- **Less precise estimates** in some settings.
- **Harder to incorporate covariates** compared to regression models.

---

## 2. Sign Test

### 2.1 Setting

Test the population median: $H_0: m = m_0$ vs. $H_1: m \neq m_0$.

### 2.2 Procedure

1. Compute $D_i = X_i - m_0$ for each observation.
2. Discard zeros; let $n'$ be the remaining count.
3. Count $S^+ = $ number of positive $D_i$.
4. Under $H_0$, $S^+ \sim \text{Binomial}(n', 0.5)$.
5. Reject if $S^+$ falls in the tails of this binomial distribution.

The sign test is the simplest nonparametric test. It uses only the signs of the differences, discarding magnitude information.

```python
import math

def sign_test(data, m_0):
    """Two-sided sign test for median = m_0."""
    diffs = [x - m_0 for x in data if x != m_0]
    n = len(diffs)
    s_plus = sum(1 for d in diffs if d > 0)

    # P-value: 2 * P(X >= max(s_plus, n-s_plus)) under Binomial(n, 0.5)
    k = max(s_plus, n - s_plus)
    # Compute binomial tail probability
    p_tail = 0
    for i in range(k, n + 1):
        # Binomial coefficient * 0.5^n
        binom_coeff = math.comb(n, i)
        p_tail += binom_coeff * (0.5 ** n)
    p_value = 2 * p_tail
    p_value = min(p_value, 1.0)

    return {"s_plus": s_plus, "n": n, "p_value": p_value}

# Example: test if median weight is 70 kg
weights = [68, 72, 65, 74, 71, 69, 73, 67, 75, 70, 66, 77, 63, 72, 71]
result = sign_test(weights, m_0=70)
print(f"Sign test: S+ = {result['s_plus']}, n = {result['n']}, p = {result['p_value']:.4f}")
```

---

## 3. Wilcoxon Signed-Rank Test

### 3.1 Setting

A more powerful alternative to the sign test for paired data or testing a median. It uses both the signs and the magnitudes (ranks) of the differences.

### 3.2 Procedure

1. Compute $D_i = X_i - m_0$ (or $D_i = X_i - Y_i$ for paired data).
2. Discard zeros.
3. Rank the $|D_i|$ from smallest to largest.
4. The test statistic is $W^+ = \sum_{D_i > 0} R_i$ (sum of ranks of positive differences).
5. Under $H_0$, the distribution of $W^+$ is symmetric.

For large $n$, the normal approximation is:

$$Z = \frac{W^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}}$$

```python
def wilcoxon_signed_rank(data, m_0=0):
    """Wilcoxon signed-rank test (two-sided, normal approximation)."""
    diffs = [(x - m_0) for x in data if x != m_0]
    n = len(diffs)

    # Rank absolute differences
    abs_diffs = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_diffs.sort()
    ranks = [0] * n
    for rank, (_, idx) in enumerate(abs_diffs, 1):
        ranks[idx] = rank

    # W+: sum of ranks where diff is positive
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)

    # Normal approximation
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mean_w) / std_w

    # Two-sided p-value (normal approximation)
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p_value = 2 * (1 - phi(abs(z)))

    return {"W_plus": w_plus, "z": z, "p_value": p_value}

# Example: paired data (before vs after)
before = [125, 130, 118, 140, 135, 128, 132, 137, 122, 145, 138, 127]
after  = [118, 125, 112, 133, 130, 122, 128, 131, 117, 138, 131, 120]
diffs = [a - b for a, b in zip(before, after)]
result = wilcoxon_signed_rank(diffs, m_0=0)
print(f"Wilcoxon signed-rank: W+ = {result['W_plus']}, z = {result['z']:.3f}, p = {result['p_value']:.4f}")
```

---

## 4. Mann-Whitney U Test

### 4.1 Setting

Compare two independent groups without assuming normality. Tests whether one distribution is stochastically greater than the other.

$H_0: P(X > Y) = 0.5$ (the two distributions are identical)

### 4.2 Procedure

1. Combine both samples and rank all observations from 1 to $N = n_1 + n_2$.
2. Compute $R_1 = $ sum of ranks in group 1.
3. $U_1 = R_1 - n_1(n_1+1)/2$.
4. $U_2 = n_1 n_2 - U_1$.
5. The test statistic is $U = \min(U_1, U_2)$.

For large samples, use the normal approximation:

$$Z = \frac{U_1 - n_1 n_2 / 2}{\sqrt{n_1 n_2 (n_1 + n_2 + 1) / 12}}$$

```python
def mann_whitney_u(group1, group2):
    """Mann-Whitney U test (two-sided, normal approximation)."""
    n1, n2 = len(group1), len(group2)
    combined = [(val, 1) for val in group1] + [(val, 2) for val in group2]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handle ties with average rank)
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # average of ranks i+1 to j
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 1)
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1

    # Normal approximation
    mu_u = n1 * n2 / 2
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu_u) / sigma_u

    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p_value = 2 * (1 - phi(abs(z)))

    return {"U1": u1, "U2": u2, "z": z, "p_value": p_value}

# Example: compare two teaching methods
method_a = [78, 82, 85, 71, 90, 76, 88, 83, 79, 86]
method_b = [65, 72, 80, 68, 75, 70, 74, 69, 77, 73]
result = mann_whitney_u(method_a, method_b)
print(f"Mann-Whitney U: U1={result['U1']:.0f}, z={result['z']:.3f}, p={result['p_value']:.4f}")
```

---

## 5. Kruskal-Wallis Test

### 5.1 Setting

Extends the Mann-Whitney U test to compare $k \geq 3$ independent groups. It is the nonparametric analog of one-way ANOVA.

$H_0$: All $k$ groups have the same distribution. $H_1$: At least one group differs.

### 5.2 Test Statistic

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

where $R_i$ is the sum of ranks in group $i$, $n_i$ is the size of group $i$, and $N = \sum n_i$.

Under $H_0$, $H \sim \chi^2_{k-1}$ approximately for large samples.

```python
def kruskal_wallis(*groups):
    """Kruskal-Wallis H test (normal approximation)."""
    k = len(groups)
    all_data = []
    for i, group in enumerate(groups):
        for val in group:
            all_data.append((val, i))
    all_data.sort(key=lambda x: x[0])
    N = len(all_data)

    # Assign average ranks for ties
    ranks = [0.0] * N
    i = 0
    while i < N:
        j = i
        while j < N and all_data[j][0] == all_data[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for idx in range(i, j):
            ranks[idx] = avg_rank
        i = j

    # Sum of ranks per group
    rank_sums = [0.0] * k
    group_sizes = [len(g) for g in groups]
    for idx in range(N):
        group_id = all_data[idx][1]
        rank_sums[group_id] += ranks[idx]

    H = (12 / (N * (N + 1))) * sum(r**2 / n for r, n in zip(rank_sums, group_sizes)) - 3 * (N + 1)
    df = k - 1
    return {"H": H, "df": df}

# Example: compare 3 fertilizers on crop yield
fert_a = [45, 52, 48, 55, 50]
fert_b = [60, 58, 65, 62, 57]
fert_c = [40, 42, 38, 44, 41]
result = kruskal_wallis(fert_a, fert_b, fert_c)
print(f"Kruskal-Wallis: H={result['H']:.3f}, df={result['df']}")
```

---

## 6. Spearman Rank Correlation

### 6.1 Definition

Spearman's rank correlation $r_s$ measures the strength and direction of the monotonic relationship between two variables.

$$r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

where $d_i = \text{rank}(X_i) - \text{rank}(Y_i)$.

When there are no tied ranks, $r_s$ is exactly the Pearson correlation computed on the ranks.

### 6.2 Properties

- $-1 \leq r_s \leq 1$
- $r_s = 1$: perfect monotonically increasing relationship
- $r_s = -1$: perfect monotonically decreasing relationship
- More robust to outliers than Pearson's $r$

```python
def spearman_rank_correlation(x, y):
    """Compute Spearman's rank correlation coefficient."""
    assert len(x) == len(y)
    n = len(x)

    def rank_data(data):
        indexed = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and data[indexed[j]] == data[indexed[i]]:
                j += 1
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                ranks[indexed[k]] = avg_rank
            i = j
        return ranks

    rx = rank_data(x)
    ry = rank_data(y)
    d_sq = sum((a - b)**2 for a, b in zip(rx, ry))
    rs = 1 - 6 * d_sq / (n * (n**2 - 1))
    return rs

# Example
hours_studied = [2, 4, 6, 8, 10, 1, 3, 5, 7, 9]
test_scores   = [55, 70, 80, 88, 95, 50, 65, 75, 85, 92]
rs = spearman_rank_correlation(hours_studied, test_scores)
print(f"Spearman rs = {rs:.4f}")
```

---

## 7. Kernel Density Estimation (KDE)

### 7.1 Definition

KDE estimates the probability density function $f(x)$ from a sample:

$$\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right)$$

where $K$ is a kernel function (e.g., Gaussian) and $h > 0$ is the **bandwidth**.

### 7.2 Common Kernels

| Kernel | $K(u)$ |
|---|---|
| Gaussian | $\frac{1}{\sqrt{2\pi}} e^{-u^2/2}$ |
| Epanechnikov | $\frac{3}{4}(1 - u^2)$ for $|u| \leq 1$ |
| Uniform | $\frac{1}{2}$ for $|u| \leq 1$ |

### 7.3 Bandwidth Selection

The bandwidth $h$ is the most critical parameter:

- **Too small**: Undersmoothed; noisy, spiky estimate.
- **Too large**: Oversmoothed; washes out features.

**Silverman's rule of thumb** (for Gaussian kernel):

$$h = 1.06 \cdot \hat{\sigma} \cdot n^{-1/5}$$

where $\hat{\sigma}$ is the sample standard deviation.

```python
def gaussian_kde(data, x_grid, bandwidth=None):
    """Gaussian Kernel Density Estimation.

    Args:
        data: list of observations
        x_grid: list of points at which to evaluate the density
        bandwidth: bandwidth h; if None, uses Silverman's rule
    """
    n = len(data)
    if bandwidth is None:
        # Silverman's rule of thumb
        mean_val = sum(data) / n
        std_val = math.sqrt(sum((x - mean_val)**2 for x in data) / (n - 1))
        bandwidth = 1.06 * std_val * n**(-0.2)

    def gaussian_kernel(u):
        return math.exp(-0.5 * u**2) / math.sqrt(2 * math.pi)

    density = []
    for x in x_grid:
        val = sum(gaussian_kernel((x - xi) / bandwidth) for xi in data) / (n * bandwidth)
        density.append(val)
    return density, bandwidth

# Example: estimate density of bimodal data
import random
random.seed(42)
data = [random.gauss(2, 0.8) for _ in range(50)] + [random.gauss(5, 1.0) for _ in range(50)]
x_grid = [i * 0.1 for i in range(-20, 100)]

density, h = gaussian_kde(data, x_grid)
print(f"Bandwidth (Silverman): {h:.3f}")
peak_x = x_grid[density.index(max(density))]
print(f"Highest density at x = {peak_x:.1f}")
```

---

## 8. Permutation Tests

### 8.1 Idea

Under $H_0$ (e.g., no difference between groups), the group labels are exchangeable. We can compute the test statistic for all (or many) permutations of the labels and compare the observed statistic to this permutation distribution.

### 8.2 Exact Permutation Test

Enumerate all $\binom{N}{n_1}$ possible assignments. Feasible only for small $N$.

### 8.3 Approximate (Monte Carlo) Permutation Test

Randomly shuffle labels $B$ times (e.g., $B = 10000$) to approximate the permutation distribution.

```python
import random

def permutation_test(group1, group2, n_permutations=10000, seed=42):
    """Two-sided permutation test for difference in means."""
    random.seed(seed)
    combined = group1 + group2
    n1 = len(group1)
    observed_diff = abs(sum(group1)/n1 - sum(group2)/len(group2))

    count_extreme = 0
    for _ in range(n_permutations):
        random.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        perm_diff = abs(sum(perm_g1)/n1 - sum(perm_g2)/len(perm_g2))
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return {"observed_diff": observed_diff, "p_value": p_value}

# Example
treatment = [5.2, 6.1, 4.8, 5.5, 6.3, 5.9, 6.0, 5.7]
control   = [4.1, 3.8, 4.5, 4.0, 3.9, 4.3, 4.2, 3.7]
result = permutation_test(treatment, control)
print(f"Observed diff: {result['observed_diff']:.3f}, p-value: {result['p_value']:.4f}")
```

---

## 9. Bootstrap Methods

### 9.1 General Framework

The bootstrap resamples **with replacement** from the observed data to approximate the sampling distribution of any statistic.

1. Draw $B$ bootstrap samples of size $n$ (with replacement).
2. Compute the statistic $\hat{\theta}^*_b$ for each.
3. Use the distribution of $\{\hat{\theta}^*_b\}$ for inference (CIs, standard errors, hypothesis tests).

### 9.2 Bootstrap Hypothesis Test

To test $H_0: \theta = \theta_0$:
1. Shift the data so that $H_0$ is true (e.g., subtract $\bar{x} - \theta_0$ from each observation).
2. Bootstrap from the shifted data.
3. Compute the p-value as the proportion of bootstrap statistics as extreme as the observed one.

```python
import random
import statistics

def bootstrap_test_mean(data, mu_0, B=10000, seed=42):
    """Bootstrap hypothesis test for H0: mean = mu_0."""
    random.seed(seed)
    n = len(data)
    x_bar = statistics.mean(data)
    observed_stat = abs(x_bar - mu_0)

    # Shift data to enforce H0
    shifted = [x - x_bar + mu_0 for x in data]

    count = 0
    for _ in range(B):
        boot_sample = random.choices(shifted, k=n)
        boot_mean = statistics.mean(boot_sample)
        if abs(boot_mean - mu_0) >= observed_stat:
            count += 1

    return {"observed_mean": x_bar, "p_value": count / B}

data = [23.1, 25.4, 22.8, 24.0, 26.1, 23.5, 24.8, 25.0, 22.5, 24.3]
result = bootstrap_test_mean(data, mu_0=24.0)
print(f"Mean = {result['observed_mean']:.2f}, Bootstrap p = {result['p_value']:.4f}")
```

---

## 10. Kolmogorov-Smirnov Test

### 10.1 One-Sample KS Test

Tests whether a sample comes from a specified distribution $F_0$:

$$D_n = \sup_x |F_n(x) - F_0(x)|$$

where $F_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(X_i \leq x)$ is the empirical CDF.

### 10.2 Two-Sample KS Test

Compares two empirical CDFs:

$$D_{n,m} = \sup_x |F_n(x) - G_m(x)|$$

Under $H_0$, the distribution of the scaled statistic is known asymptotically.

```python
def ks_two_sample(sample1, sample2):
    """Two-sample Kolmogorov-Smirnov test statistic."""
    s1 = sorted(sample1)
    s2 = sorted(sample2)
    n1, n2 = len(s1), len(s2)

    # Merge and compute ECDFs
    all_vals = sorted(set(s1 + s2))
    max_diff = 0.0
    for x in all_vals:
        ecdf1 = sum(1 for v in s1 if v <= x) / n1
        ecdf2 = sum(1 for v in s2 if v <= x) / n2
        diff = abs(ecdf1 - ecdf2)
        if diff > max_diff:
            max_diff = diff

    # Approximate critical value (alpha=0.05)
    c_alpha = 1.36  # for alpha=0.05
    critical = c_alpha * math.sqrt((n1 + n2) / (n1 * n2))

    return {"D": max_diff, "critical_value_05": critical,
            "reject_H0": max_diff > critical}

# Example: do two samples come from the same distribution?
random.seed(42)
sample_a = [random.gauss(0, 1) for _ in range(50)]
sample_b = [random.gauss(0.5, 1) for _ in range(50)]
result = ks_two_sample(sample_a, sample_b)
print(f"KS D = {result['D']:.4f}, critical (5%) = {result['critical_value_05']:.4f}")
print(f"Reject H0: {result['reject_H0']}")
```

---

## 11. Key Takeaways

| Method | Use Case | Parametric Analog |
|---|---|---|
| Sign test | Test median; ordinal data | One-sample t-test |
| Wilcoxon signed-rank | Paired differences; symmetric distribution | Paired t-test |
| Mann-Whitney U | Compare two independent groups | Two-sample t-test |
| Kruskal-Wallis | Compare $k \geq 3$ groups | One-way ANOVA |
| Spearman $r_s$ | Monotonic association | Pearson $r$ |
| KDE | Estimate density without parametric model | Parametric density fit |
| Permutation test | Any hypothesis; exact or approximate | Varies |
| Bootstrap | General inference (CIs, tests) | Varies |
| Kolmogorov-Smirnov | Compare distributions | Likelihood ratio |

**Practical guideline**: Use parametric tests when their assumptions hold (for maximum power). Use nonparametric alternatives when assumptions are questionable or data are ordinal. Permutation and bootstrap methods are versatile tools that work in nearly any setting.

---

**Previous**: [Bayesian Inference](./15_Bayesian_Inference.md) | **Next**: [Regression and ANOVA](./17_Regression_and_ANOVA.md)

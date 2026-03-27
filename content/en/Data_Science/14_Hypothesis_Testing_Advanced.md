# 14. Advanced Hypothesis Testing

[Previous: Confidence Intervals](./13_Confidence_Intervals.md) | [Next: ANOVA](./15_ANOVA.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between Type I error, Type II error, and statistical power, and visualize their relationships
2. Calculate and interpret effect sizes (Cohen's d, Hedges' g, Glass's delta) for group comparisons
3. Perform power analysis to determine the minimum sample size needed for a study
4. Explain why multiple testing inflates the family-wise error rate and calculate the theoretical FWER
5. Apply Bonferroni, Holm, and Benjamini-Hochberg corrections to a set of p-values
6. Compare FWER-controlling and FDR-controlling methods in terms of conservatism and discovery rates
7. Implement comprehensive hypothesis testing workflows using `pingouin` and `statsmodels`

---

> **How surprised should you be?** Think of a p-value as a "surprise meter." You assume nothing interesting is happening (the null hypothesis), then ask: "If that were true, how surprising is the data I just saw?" A tiny p-value means the data would be *very* surprising under the null -- surprising enough that you start doubting the assumption itself. The p-value doesn't tell you the probability of the hypothesis being true; it tells you how incompatible the data is with the null scenario.

Knowing that a result is "statistically significant" is only the beginning. How large is the effect? Was the study powerful enough to detect it? And when you run dozens of tests at once, how many of your "discoveries" are false alarms? This lesson equips you with the tools to answer these deeper questions -- power analysis for planning studies, effect sizes for measuring practical importance, and multiple-testing corrections for controlling error rates across many simultaneous comparisons.

---

## 1. Review of Hypothesis Testing

### 1.1 Basic Framework of Hypothesis Testing

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Basic elements of hypothesis testing
# H₀: null hypothesis - no effect
# H₁: alternative hypothesis - there is an effect
# α: significance level - usually 0.05
# p-value: probability of obtaining results as extreme as observed, assuming H₀ is true

# Example: One-sample t-test
np.random.seed(42)
sample = np.random.normal(102, 15, 30)  # True mean 102
mu_0 = 100  # Null hypothesis: mean = 100

# Perform t-test
t_stat, p_value = stats.ttest_1samp(sample, mu_0)

print("One-sample t-test:")
print(f"Sample mean: {sample.mean():.2f}")
print(f"Null hypothesis mean: {mu_0}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ Reject null hypothesis (α=0.05)")
else:
    print("→ Fail to reject null hypothesis (α=0.05)")
```

### 1.2 Type I and Type II Errors

```python
# Type I Error: Reject H₀ when it's true → α (significance level)
# Type II Error: Fail to reject H₀ when it's false → β
# Power: 1 - β = probability of rejecting H₀ when it's false

def visualize_errors(mu_0, mu_1, sigma, n, alpha=0.05):
    """Visualize Type I and Type II errors"""
    se = sigma / np.sqrt(n)

    # Distribution under null hypothesis (H₀: μ = μ₀)
    null_dist = stats.norm(mu_0, se)

    # Distribution under alternative hypothesis (H₁: μ = μ₁)
    alt_dist = stats.norm(mu_1, se)

    # Critical values (two-tailed test)
    critical_upper = null_dist.ppf(1 - alpha/2)
    critical_lower = null_dist.ppf(alpha/2)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.linspace(mu_0 - 4*se, mu_1 + 4*se, 500)

    # Null hypothesis distribution
    ax.plot(x, null_dist.pdf(x), 'b-', lw=2, label=f'H₀: μ = {mu_0}')
    ax.fill_between(x, null_dist.pdf(x),
                    where=(x >= critical_upper) | (x <= critical_lower),
                    alpha=0.3, color='red', label=f'Type I Error (α = {alpha})')

    # Alternative hypothesis distribution
    ax.plot(x, alt_dist.pdf(x), 'g-', lw=2, label=f'H₁: μ = {mu_1}')
    ax.fill_between(x, alt_dist.pdf(x),
                    where=(x < critical_upper) & (x > critical_lower),
                    alpha=0.3, color='orange', label='Type II Error (β)')

    # Mark critical values
    ax.axvline(critical_upper, color='red', linestyle='--', lw=1.5)
    ax.axvline(critical_lower, color='red', linestyle='--', lw=1.5)

    # Calculate β and Power
    # Why this formula equals β: we integrate the alternative distribution between the
    # two rejection boundaries.  The area under alt_dist in that middle region is the
    # probability that the test statistic falls in the "fail to reject" zone *even though
    # H₁ is true* — that is the definition of a Type II error.  Power = 1 - β is the
    # complementary probability: correctly rejecting H₀ when H₁ is true.
    beta = alt_dist.cdf(critical_upper) - alt_dist.cdf(critical_lower)
    power = 1 - beta

    ax.set_xlabel('Sample mean', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Errors in Hypothesis Testing\nβ = {beta:.3f}, Power = {power:.3f}', fontsize=14)
    ax.legend(fontsize=10)

    return beta, power

# Example
mu_0, mu_1 = 100, 105
sigma = 15
n = 30

beta, power = visualize_errors(mu_0, mu_1, sigma, n, alpha=0.05)
print(f"Type II error (β): {beta:.3f}")
print(f"Power: {power:.3f}")
plt.show()
```

---

## 2. Power and Effect Size

### 2.1 Effect Size

```python
# Cohen's d: Standardize mean difference by standard deviation
# d = (μ₁ - μ₀) / σ

def cohens_d(group1, group2):
    """Calculate Cohen's d for two groups"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

# Example
np.random.seed(42)
control = np.random.normal(100, 15, 50)
treatment = np.random.normal(107, 15, 50)

d = cohens_d(treatment, control)

print("Cohen's d calculation:")
print(f"Control mean: {control.mean():.2f}")
print(f"Treatment mean: {treatment.mean():.2f}")
print(f"Cohen's d: {d:.3f}")

# Interpretation guidelines (Cohen, 1988)
print("\nCohen's d interpretation guidelines:")
print("  |d| ≈ 0.2: small effect")
print("  |d| ≈ 0.5: medium effect")
print("  |d| ≈ 0.8: large effect")
```

```python
# Various effect size measures

def effect_sizes(group1, group2):
    """Calculate various effect sizes"""
    # Cohen's d
    d = cohens_d(group1, group2)

    # Hedges' g (small sample correction)
    # Why correct Cohen's d? With small samples, the pooled standard deviation tends to
    # underestimate the population σ, making |d| slightly inflated.  The correction
    # factor 1 - 3/(4N-9) shrinks d toward zero to account for this bias.
    # As N grows, the factor approaches 1 and Hedges' g converges to Cohen's d —
    # so for large samples (n > 20 per group) the two measures are virtually identical.
    n1, n2 = len(group1), len(group2)
    correction = 1 - (3 / (4*(n1+n2) - 9))
    g = d * correction

    # Glass's delta (using control group std only)
    delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)

    # r (convert to correlation coefficient)
    t, _ = stats.ttest_ind(group1, group2)
    df = n1 + n2 - 2
    r = np.sqrt(t**2 / (t**2 + df))

    return {
        "Cohen's d": d,
        "Hedges' g": g,
        "Glass's delta": delta,
        "r (effect size)": r
    }

effects = effect_sizes(treatment, control)
print("\nVarious effect size measures:")
for name, value in effects.items():
    print(f"  {name}: {value:.4f}")
```

### 2.2 Effect Size Calculation Using pingouin

```python
import pingouin as pg

# Easy effect size calculation with pingouin
np.random.seed(42)
group1 = np.random.normal(100, 15, 40)
group2 = np.random.normal(108, 15, 45)

# t-test with effect size
result = pg.ttest(group1, group2, correction=False)
print("pingouin t-test results:")
print(result.to_string())

# Calculate effect size separately
d = pg.compute_effsize(group1, group2, eftype='cohen')
hedges = pg.compute_effsize(group1, group2, eftype='hedges')
cles = pg.compute_effsize(group1, group2, eftype='CLES')  # Common Language Effect Size

print(f"\nEffect sizes:")
print(f"  Cohen's d: {d:.4f}")
print(f"  Hedges' g: {hedges:.4f}")
print(f"  CLES: {cles:.4f}")
print(f"  (CLES: probability that a randomly selected value from group1 is greater than group2)")
```

### 2.3 Power Analysis

```python
# Power: Probability of rejecting H₀ when H₁ is true

def power_ttest(mu_0, mu_1, sigma, n, alpha=0.05, alternative='two-sided'):
    """Calculate power of t-test"""
    se = sigma / np.sqrt(n)
    d = (mu_1 - mu_0) / sigma  # Effect size

    if alternative == 'two-sided':
        critical_upper = stats.norm.ppf(1 - alpha/2)
        critical_lower = -critical_upper

        # Use non-central t-distribution (approximately normal)
        ncp = d * np.sqrt(n)  # Non-centrality parameter
        power = 1 - (stats.norm.cdf(critical_upper - ncp) -
                     stats.norm.cdf(critical_lower - ncp))
    elif alternative == 'greater':
        critical = stats.norm.ppf(1 - alpha)
        ncp = d * np.sqrt(n)
        power = 1 - stats.norm.cdf(critical - ncp)
    else:  # 'less'
        critical = stats.norm.ppf(alpha)
        ncp = d * np.sqrt(n)
        power = stats.norm.cdf(critical - ncp)

    return power

# Example
mu_0, mu_1 = 100, 105
sigma = 15
n = 30

power = power_ttest(mu_0, mu_1, sigma, n, alpha=0.05)
print(f"Power analysis:")
print(f"H₀: μ = {mu_0}, H₁: μ = {mu_1}")
print(f"σ = {sigma}, n = {n}, α = 0.05")
print(f"Power = {power:.4f}")

# Power change with sample size
sample_sizes = np.arange(10, 201, 5)
powers = [power_ttest(mu_0, mu_1, sigma, n) for n in sample_sizes]

plt.figure(figsize=(10, 5))
plt.plot(sample_sizes, powers, 'b-', linewidth=2)
plt.axhline(0.8, color='r', linestyle='--', label='Power = 0.8')
plt.xlabel('Sample size (n)')
plt.ylabel('Power')
plt.title('Relationship Between Sample Size and Power')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Find minimum sample size for 80% power
for n in sample_sizes:
    if power_ttest(mu_0, mu_1, sigma, n) >= 0.8:
        print(f"\nMinimum sample size for 80% power: {n}")
        break
```

---

## 3. Sample Size Determination

### 3.1 Sample Size Calculation Using statsmodels

```python
from statsmodels.stats.power import TTestIndPower, TTestPower

# Power analysis for independent samples t-test
power_analysis = TTestIndPower()

# Calculate required sample size (per group)
effect_size = 0.5  # Medium effect size
alpha = 0.05
power = 0.8

n = power_analysis.solve_power(effect_size=effect_size,
                                alpha=alpha,
                                power=power,
                                ratio=1,  # Ratio of two group sizes
                                alternative='two-sided')

print("Independent samples t-test sample size calculation:")
print(f"Effect size (d): {effect_size}")
print(f"Significance level (α): {alpha}")
print(f"Power (1-β): {power}")
print(f"Required sample size (per group): {np.ceil(n):.0f}")
print(f"Total required sample size: {np.ceil(n)*2:.0f}")

# Visualize power curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sample size vs power (various effect sizes)
sample_sizes = np.arange(10, 151)
effect_sizes = [0.2, 0.5, 0.8]
colors = ['blue', 'green', 'red']

for d, color in zip(effect_sizes, colors):
    powers = [power_analysis.solve_power(effect_size=d, nobs1=n, alpha=0.05)
              for n in sample_sizes]
    axes[0].plot(sample_sizes, powers, color=color, linewidth=2, label=f'd = {d}')

axes[0].axhline(0.8, color='k', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Sample size (per group)')
axes[0].set_ylabel('Power')
axes[0].set_title('Sample Size and Power')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Effect size vs required sample size
effect_range = np.linspace(0.1, 1.5, 50)
required_n = [power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.8)
              for d in effect_range]

axes[1].plot(effect_range, required_n, 'b-', linewidth=2)
axes[1].set_xlabel('Effect size (d)')
axes[1].set_ylabel('Required sample size (per group)')
axes[1].set_title('Effect Size and Required Sample Size (Power=0.8)')
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 500)

# Mark key effect sizes
for d in [0.2, 0.5, 0.8]:
    n_req = power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.8)
    axes[1].scatter([d], [n_req], color='red', s=100, zorder=5)
    axes[1].annotate(f'd={d}\nn={n_req:.0f}', (d, n_req), xytext=(d+0.05, n_req+30))

plt.tight_layout()
plt.show()
```

### 3.2 Sample Size for Proportion Comparison

```python
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

# Sample size for comparing two proportions
p1 = 0.30  # Control group proportion
p2 = 0.40  # Treatment group proportion (expected)
alpha = 0.05
power = 0.80

# Calculate effect size (Cohen's h)
h = proportion_effectsize(p2, p1)
print(f"Cohen's h: {h:.4f}")

# Required sample size
n = zt_ind_solve_power(effect_size=h, alpha=alpha, power=power)

print(f"\nSample size calculation for comparing two proportions:")
print(f"Control group proportion: {p1}")
print(f"Treatment group proportion (expected): {p2}")
print(f"Required sample size (per group): {np.ceil(n):.0f}")
```

### 3.3 Power Analysis Using pingouin

```python
# pingouin power functions

# t-test power
power_result = pg.power_ttest(d=0.5, n=30, power=None, alpha=0.05,
                               alternative='two-sided')
print(f"Power (d=0.5, n=30): {power_result:.4f}")

# Required sample size
n_result = pg.power_ttest(d=0.5, n=None, power=0.8, alpha=0.05,
                           alternative='two-sided')
print(f"Required sample size (d=0.5, power=0.8): {n_result:.1f}")

# Detectable effect size
d_result = pg.power_ttest(d=None, n=50, power=0.8, alpha=0.05,
                           alternative='two-sided')
print(f"Detectable effect size (n=50, power=0.8): {d_result:.4f}")

# Correlation analysis power
power_corr = pg.power_corr(r=0.3, n=50, power=None, alpha=0.05)
print(f"\nCorrelation analysis power (r=0.3, n=50): {power_corr:.4f}")

n_corr = pg.power_corr(r=0.3, n=None, power=0.8, alpha=0.05)
print(f"Correlation analysis required sample size (r=0.3, power=0.8): {n_corr:.1f}")
```

---

## 4. Multiple Testing Problem

### 4.1 Problem of Multiple Testing

```python
# Type I error rate increases with multiple tests

def multiple_testing_demo(n_tests, alpha=0.05, n_simulations=10000):
    """Multiple testing simulation"""
    np.random.seed(42)

    at_least_one_significant = 0

    for _ in range(n_simulations):
        # Situation where all H₀ are true (no effect)
        p_values = np.random.uniform(0, 1, n_tests)

        # Type I error if at least one is significant
        if np.any(p_values < alpha):
            at_least_one_significant += 1

    familywise_error_rate = at_least_one_significant / n_simulations

    # Theoretical FWER: 1 - (1-α)^n
    theoretical_fwer = 1 - (1 - alpha) ** n_tests

    return familywise_error_rate, theoretical_fwer

# Simulation for various numbers of tests
test_counts = [1, 5, 10, 20, 50, 100]

print("Type I error rate (FWER) with multiple testing:")
print("-" * 50)
print(f"{'Test count':<12} {'Simulation':<15} {'Theoretical':<15}")
print("-" * 50)

for n_tests in test_counts:
    simulated, theoretical = multiple_testing_demo(n_tests)
    print(f"{n_tests:<12} {simulated:<15.4f} {theoretical:<15.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

n_tests_range = np.arange(1, 101)
fwer = 1 - (1 - 0.05) ** n_tests_range

ax.plot(n_tests_range, fwer, 'b-', linewidth=2)
ax.axhline(0.05, color='r', linestyle='--', label='α = 0.05')
ax.fill_between(n_tests_range, 0.05, fwer, alpha=0.3, color='red')
ax.set_xlabel('Number of tests')
ax.set_ylabel('FWER (Family-Wise Error Rate)')
ax.set_title('Multiple Testing Problem: FWER Increase')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

print(f"\nFWER with 20 tests: {1 - (1-0.05)**20:.2%}")
print(f"FWER with 100 tests: {1 - (1-0.05)**100:.2%}")
```

### 4.2 Bonferroni Correction

```python
from statsmodels.stats.multitest import multipletests

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni correction"""
    n = len(p_values)
    adjusted_alpha = alpha / n
    reject = p_values < adjusted_alpha
    adjusted_p = np.minimum(p_values * n, 1.0)
    return reject, adjusted_p, adjusted_alpha

# Example: 10 p-values
np.random.seed(42)
p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.06, 0.10, 0.15, 0.35, 0.50, 0.80])

# Bonferroni correction
reject, adjusted_p, adj_alpha = bonferroni_correction(p_values)

print("Bonferroni correction:")
print(f"Original α = 0.05, Corrected α = {adj_alpha:.4f}")
print("-" * 60)
print(f"{'i':<5} {'p-value':<12} {'Adjusted p':<15} {'Reject':<10}")
print("-" * 60)
for i, (p, adj_p, rej) in enumerate(zip(p_values, adjusted_p, reject), 1):
    print(f"{i:<5} {p:<12.4f} {adj_p:<15.4f} {'Yes' if rej else 'No':<10}")

# Using statsmodels
reject_sm, adjusted_p_sm, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print(f"\nstatsmodels result verification: {np.allclose(adjusted_p, adjusted_p_sm)}")
```

### 4.3 Holm-Bonferroni Correction (Step-down)

```python
def holm_correction(p_values, alpha=0.05):
    """Holm-Bonferroni correction (step-down)"""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    reject = np.zeros(n, dtype=bool)
    adjusted_p = np.zeros(n)

    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        # Holm's corrected α: α / (n - i)
        holm_alpha = alpha / (n - i)

        if p < holm_alpha:
            reject[idx] = True
        else:
            # Fail to reject all subsequent hypotheses
            break

        # Adjusted p-value
        adjusted_p[idx] = min((n - i) * p, 1.0)

    # Ensure monotonicity
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i-1]
        adjusted_p[idx] = max(adjusted_p[idx], adjusted_p[prev_idx])

    return reject, adjusted_p

# Comparison
p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.06, 0.10, 0.15, 0.35, 0.50, 0.80])

reject_bonf, adj_p_bonf, _ = bonferroni_correction(p_values)
reject_holm, adj_p_holm = holm_correction(p_values)

# Verify with statsmodels
_, adj_p_holm_sm, _, _ = multipletests(p_values, alpha=0.05, method='holm')

print("Bonferroni vs Holm comparison:")
print("-" * 70)
print(f"{'p-value':<10} {'Bonf p':<12} {'Bonf Reject':<12} {'Holm p':<12} {'Holm Reject':<10}")
print("-" * 70)
for p, bp, br, hp, hr in zip(p_values, adj_p_bonf, reject_bonf,
                              adj_p_holm_sm, reject_holm):
    print(f"{p:<10.4f} {bp:<12.4f} {'Yes' if br else 'No':<12} {hp:<12.4f} {'Yes' if hr else 'No':<10}")

print(f"\n→ Holm method is less conservative than Bonferroni (more rejections possible)")
```

### 4.4 False Discovery Rate (FDR) Correction

```python
def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction"""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH threshold: (i/n) * α
    # Why rank-scaled thresholds? The BH procedure controls the *expected* fraction of
    # false discoveries among all rejections (FDR), not the probability of any false
    # discovery (FWER).  By giving higher-ranked (smaller) p-values a proportionally
    # tighter threshold, it allows more discoveries than Bonferroni while bounding the
    # proportion of those discoveries that are spurious.
    bh_thresholds = (np.arange(1, n+1) / n) * alpha

    # Rejection decision: find the largest rank where p ≤ threshold, then reject all
    # hypotheses ranked up to that point (not just those below their own threshold).
    # This "keep going until the last crossing" rule is what makes BH coherent —
    # rejecting a weaker hypothesis also requires rejecting all stronger ones.
    below_threshold = sorted_p <= bh_thresholds
    if np.any(below_threshold):
        max_i = np.max(np.where(below_threshold)[0])
        reject = np.zeros(n, dtype=bool)
        reject[sorted_indices[:max_i+1]] = True
    else:
        reject = np.zeros(n, dtype=bool)

    # Adjusted p-values
    adjusted_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adjusted_p[idx] = sorted_p[i] * n / (i + 1)

    # Ensure monotonicity (reverse order)
    for i in range(n-2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i+1]
        adjusted_p[idx] = min(adjusted_p[idx], adjusted_p[next_idx])

    adjusted_p = np.minimum(adjusted_p, 1.0)

    return reject, adjusted_p

# Comparison
p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.06, 0.10, 0.15, 0.35, 0.50, 0.80])

# Using statsmodels
_, adj_p_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("FDR (Benjamini-Hochberg) correction:")
print("-" * 60)
print(f"{'Rank':<6} {'p-value':<10} {'BH threshold':<12} {'Adjusted p':<15}")
print("-" * 60)

sorted_p = np.sort(p_values)
n = len(p_values)
for i, p in enumerate(sorted_p, 1):
    threshold = (i / n) * 0.05
    adj_p = p_values[np.argsort(p_values)[i-1]]
    print(f"{i:<6} {p:<10.4f} {threshold:<12.4f} {adj_p_bh[np.argsort(p_values)[i-1]]:<15.4f}")
```

### 4.5 Comprehensive Comparison of Multiple Testing Methods

```python
# Compare all methods
methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']
method_names = ['Bonferroni', 'Holm', 'BH (FDR)', 'BY (FDR)']

p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.055, 0.10])

print("Comparison of multiple testing correction methods:")
print("=" * 80)
print(f"{'Original p':<15}", end='')
for name in method_names:
    print(f"{name:<15}", end='')
print("\n" + "=" * 80)

results = {}
for method in methods:
    _, adj_p, _, _ = multipletests(p_values, alpha=0.05, method=method)
    results[method] = adj_p

for i, p in enumerate(p_values):
    print(f"{p:<15.4f}", end='')
    for method in methods:
        adj = results[method][i]
        sig = "*" if adj < 0.05 else ""
        print(f"{adj:<13.4f}{sig:<2}", end='')
    print()

print("=" * 80)
print("* indicates: adjusted p-value < 0.05 (significant)")

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(p_values))
width = 0.15

for i, (method, name) in enumerate(zip(methods, method_names)):
    adj_p = results[method]
    ax.bar(x + i*width, adj_p, width, label=name, alpha=0.8)

ax.bar(x - width, p_values, width, label='Original', color='gray', alpha=0.5)
ax.axhline(0.05, color='red', linestyle='--', label='α = 0.05')

ax.set_xlabel('Hypothesis')
ax.set_ylabel('p-value')
ax.set_title('Comparison of Multiple Testing Correction Methods')
ax.set_xticks(x + width)
ax.set_xticklabels([f'H{i+1}' for i in range(len(p_values))])
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 4.6 Real Data Example

```python
# Gene expression data simulation
np.random.seed(42)

n_genes = 100
n_samples = 20

# Most genes have no difference (H₀ true)
# Only 10 genes are truly differentially expressed (H₁ true)
truly_different = 10

# Generate p-values
p_values = np.zeros(n_genes)

# Genes where H₀ is true (uniform distribution p-values)
p_values[truly_different:] = np.random.uniform(0, 1, n_genes - truly_different)

# Genes where H₁ is true (tendency toward low p-values)
p_values[:truly_different] = np.random.beta(0.5, 5, truly_different) * 0.1

# Multiple testing correction
_, adj_p_bonf, _, _ = multipletests(p_values, method='bonferroni')
_, adj_p_bh, _, _ = multipletests(p_values, method='fdr_bh')

# Compare results
def count_discoveries(p_values, adj_p, alpha=0.05, n_true=truly_different):
    """Count discoveries"""
    significant_original = p_values < alpha
    significant_adjusted = adj_p < alpha

    # True positives discovered among truly positive
    tp_original = np.sum(significant_original[:n_true])
    tp_adjusted = np.sum(significant_adjusted[:n_true])

    # False positives discovered among truly negative
    fp_original = np.sum(significant_original[n_true:])
    fp_adjusted = np.sum(significant_adjusted[n_true:])

    return {
        'total_discoveries': np.sum(significant_adjusted),
        'true_positives': tp_adjusted,
        'false_positives': fp_adjusted,
        'false_discovery_rate': fp_adjusted / max(np.sum(significant_adjusted), 1)
    }

print(f"Simulation: {n_genes} genes, {truly_different} truly differentially expressed")
print("=" * 60)

# No correction
results_none = count_discoveries(p_values, p_values)
print(f"\nNo correction:")
print(f"  Total discoveries: {np.sum(p_values < 0.05)}")
print(f"  True Positives: {np.sum(p_values[:truly_different] < 0.05)}")
print(f"  False Positives: {np.sum(p_values[truly_different:] < 0.05)}")

# Bonferroni
results_bonf = count_discoveries(p_values, adj_p_bonf)
print(f"\nBonferroni correction:")
print(f"  Total discoveries: {results_bonf['total_discoveries']}")
print(f"  True Positives: {results_bonf['true_positives']}")
print(f"  False Positives: {results_bonf['false_positives']}")

# BH (FDR)
results_bh = count_discoveries(p_values, adj_p_bh)
print(f"\nBH (FDR) correction:")
print(f"  Total discoveries: {results_bh['total_discoveries']}")
print(f"  True Positives: {results_bh['true_positives']}")
print(f"  False Positives: {results_bh['false_positives']}")
print(f"  Actual FDR: {results_bh['false_discovery_rate']:.2%}")
```

---

## 5. Comprehensive Analysis Using pingouin

### 5.1 Various Tests

```python
import pingouin as pg
import pandas as pd

# Prepare data
np.random.seed(42)
df = pd.DataFrame({
    'group': ['A']*30 + ['B']*30 + ['C']*30,
    'value': np.concatenate([
        np.random.normal(100, 15, 30),
        np.random.normal(105, 15, 30),
        np.random.normal(110, 15, 30)
    ])
})

# Normality test
print("Normality test (Shapiro-Wilk):")
for group in ['A', 'B', 'C']:
    data = df[df['group'] == group]['value']
    result = pg.normality(data)
    print(f"  Group {group}: W={result['W'].values[0]:.4f}, p={result['pval'].values[0]:.4f}")

# Homogeneity of variance test
print("\nHomogeneity of variance test (Levene):")
levene_result = pg.homoscedasticity(df, dv='value', group='group')
print(levene_result)

# t-test (A vs B)
print("\nt-test (A vs B):")
ttest_result = pg.ttest(df[df['group']=='A']['value'],
                        df[df['group']=='B']['value'])
print(ttest_result)

# Non-parametric test (Mann-Whitney)
print("\nMann-Whitney U test (A vs B):")
mwu_result = pg.mwu(df[df['group']=='A']['value'],
                    df[df['group']=='B']['value'])
print(mwu_result)
```

### 5.2 Effect Size Interpretation Guide

```python
# Effect size interpretation function
def interpret_effect_size(d, measure='cohen_d'):
    """Interpret effect size"""
    d = abs(d)

    if measure == 'cohen_d':
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    elif measure == 'r':
        if d < 0.1:
            return "negligible"
        elif d < 0.3:
            return "small"
        elif d < 0.5:
            return "medium"
        else:
            return "large"

# Example
effect_sizes = [0.15, 0.35, 0.65, 0.95]

print("Cohen's d interpretation:")
print("-" * 40)
for d in effect_sizes:
    print(f"d = {d:.2f}: {interpret_effect_size(d)}")
```

---

## Practice Problems

### Problem 1: Power Analysis
When the mean difference between two groups is 5 points and standard deviation is 12:
- (a) What is the power when testing with 30 people per group?
- (b) What is the minimum sample size for 80% power?

### Problem 2: Multiple Testing
Apply Bonferroni and BH corrections to the following 10 p-values.
```python
p_values = [0.001, 0.005, 0.010, 0.020, 0.040, 0.080, 0.120, 0.200, 0.500, 0.800]
```

### Problem 3: Effect Size
Calculate and interpret Cohen's d and Hedges' g for two group data.
```python
group1 = [23, 25, 27, 22, 28, 26, 24, 29, 25, 27]
group2 = [30, 32, 28, 31, 33, 29, 35, 31, 30, 32]
```

---

## Summary

| Concept | Key Content | Python |
|---------|-------------|--------|
| Type I error (α) | Reject true H₀ | Significance level |
| Type II error (β) | Fail to reject false H₀ | 1 - power |
| Power | 1 - β | `pg.power_ttest()` |
| Cohen's d | Effect size | `pg.compute_effsize()` |
| Bonferroni | FWER correction | `multipletests(method='bonferroni')` |
| Holm | FWER correction (less conservative) | `multipletests(method='holm')` |
| BH (FDR) | FDR correction | `multipletests(method='fdr_bh')` |

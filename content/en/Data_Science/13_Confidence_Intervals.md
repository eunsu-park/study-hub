# 13. Confidence Intervals

[Previous: Sampling and Estimation](./12_Sampling_and_Estimation.md) | [Next: Advanced Hypothesis Testing](./14_Hypothesis_Testing_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the correct frequentist interpretation of a confidence interval and common misconceptions
2. Construct Z-intervals and t-intervals for a population mean and describe when each applies
3. Calculate confidence intervals for population proportions using normal approximation and the Wilson method
4. Derive confidence intervals for population variance using the chi-square distribution
5. Compute confidence intervals for the difference of two means (independent and paired) and two proportions
6. Implement bootstrap confidence intervals (percentile and BCa methods) for arbitrary statistics
7. Apply `scipy.stats` and `statsmodels` functions to compute confidence intervals efficiently

---

A point estimate like a sample mean gives you a single best guess, but it tells you nothing about how uncertain that guess is. Confidence intervals quantify that uncertainty by providing a plausible range for the true parameter. Understanding how to construct, interpret, and choose among different interval methods is essential for honest reporting of any data analysis result.

---

## 1. Basic Concepts of Interval Estimation

### 1.1 Definition of Confidence Interval

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Correct interpretation of confidence intervals
# Meaning of "95% confidence interval":
# - If we draw samples 100 times using the same method and construct intervals,
# - About 95 of them will contain the parameter

# Understanding through simulation
np.random.seed(42)

# Parameter setting (unknown in practice)
mu = 100
sigma = 15
n = 30
confidence_level = 0.95

# Draw 100 samples and calculate confidence intervals
n_simulations = 100
intervals = []
contains_mu = 0

z_critical = stats.norm.ppf((1 + confidence_level) / 2)

for i in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    x_bar = sample.mean()
    se = sigma / np.sqrt(n)  # Assuming known population std

    lower = x_bar - z_critical * se
    upper = x_bar + z_critical * se

    intervals.append((lower, upper))
    if lower <= mu <= upper:
        contains_mu += 1

print(f"Out of 100 95% confidence intervals, {contains_mu} contain the population mean")

# Visualization (first 20 intervals)
fig, ax = plt.subplots(figsize=(12, 8))

for i in range(20):
    lower, upper = intervals[i]
    color = 'blue' if lower <= mu <= upper else 'red'
    ax.plot([lower, upper], [i, i], color=color, linewidth=2)
    ax.scatter([(lower + upper)/2], [i], color=color, s=30)

ax.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'μ = {mu}')
ax.set_xlabel('Value')
ax.set_ylabel('Simulation number')
ax.set_title('Meaning of Confidence Intervals: Intervals not containing the mean (red) among 20')
ax.legend()
ax.set_xlim(90, 110)
plt.show()
```

### 1.2 Relationship Between Confidence Level and Interval Width

```python
# Confidence level ↑ → Interval width ↑
# Sample size ↑ → Interval width ↓

np.random.seed(42)
sample = np.random.normal(100, 15, 50)
x_bar = sample.mean()
s = sample.std(ddof=1)
n = len(sample)
se = s / np.sqrt(n)

confidence_levels = [0.80, 0.90, 0.95, 0.99]

print("Confidence intervals by confidence level:")
print("-" * 60)
print(f"Sample mean = {x_bar:.2f}, Standard error = {se:.2f}")
print("-" * 60)

for cl in confidence_levels:
    t_critical = stats.t.ppf((1 + cl) / 2, df=n-1)
    margin = t_critical * se
    lower = x_bar - margin
    upper = x_bar + margin
    width = upper - lower
    print(f"{int(cl*100)}% CI: [{lower:.2f}, {upper:.2f}], Width = {width:.2f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(confidence_levels)))
for i, (cl, color) in enumerate(zip(confidence_levels, colors)):
    t_critical = stats.t.ppf((1 + cl) / 2, df=n-1)
    margin = t_critical * se
    ax.barh(i, 2*margin, left=x_bar-margin, height=0.6, color=color,
            label=f'{int(cl*100)}% CI', alpha=0.7)

ax.axvline(x_bar, color='red', linestyle='-', linewidth=2, label='Sample mean')
ax.set_yticks(range(len(confidence_levels)))
ax.set_yticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
ax.set_xlabel('Value')
ax.set_ylabel('Confidence level')
ax.set_title('Relationship Between Confidence Level and Interval Width')
ax.legend(loc='upper right')
plt.show()
```

---

## 2. Confidence Interval for Population Mean

### 2.1 When Population Variance is Known (Z-interval)

```python
def ci_mean_z(sample, sigma, confidence=0.95):
    """Confidence interval for population mean when variance is known (Z-interval)"""
    n = len(sample)
    x_bar = np.mean(sample)
    se = sigma / np.sqrt(n)

    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin = z_critical * se

    return x_bar - margin, x_bar + margin, margin

# Example: Monitoring defect rate in manufacturing process
# Historical data shows σ = 2.5
np.random.seed(42)
sigma_known = 2.5
sample = np.random.normal(50, sigma_known, 40)

lower, upper, margin = ci_mean_z(sample, sigma_known, 0.95)

print("When population variance is known (Z-interval):")
print(f"Sample mean: {sample.mean():.3f}")
print(f"95% confidence interval: [{lower:.3f}, {upper:.3f}]")
print(f"Margin of error: ±{margin:.3f}")
```

### 2.2 When Population Variance is Unknown (t-interval)

```python
def ci_mean_t(sample, confidence=0.95):
    """Confidence interval for population mean when variance is unknown (t-interval)"""
    n = len(sample)
    x_bar = np.mean(sample)
    s = np.std(sample, ddof=1)
    se = s / np.sqrt(n)

    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_critical * se

    return x_bar - margin, x_bar + margin, margin

# Example
np.random.seed(42)
sample = np.random.normal(50, 2.5, 40)

lower_t, upper_t, margin_t = ci_mean_t(sample, 0.95)

print("When population variance is unknown (t-interval):")
print(f"Sample mean: {sample.mean():.3f}")
print(f"Sample standard deviation: {sample.std(ddof=1):.3f}")
print(f"95% confidence interval: [{lower_t:.3f}, {upper_t:.3f}]")
print(f"Margin of error: ±{margin_t:.3f}")

# Using scipy.stats
sem = stats.sem(sample)  # Standard error
ci_scipy = stats.t.interval(0.95, df=len(sample)-1, loc=sample.mean(), scale=sem)
print(f"\nscipy result: [{ci_scipy[0]:.3f}, {ci_scipy[1]:.3f}]")
```

### 2.3 Understanding t-distribution

```python
# t-distribution: Used when sample size is small
# t = (X̄ - μ) / (S/√n) ~ t(n-1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# t-distribution vs normal distribution
x = np.linspace(-4, 4, 200)
axes[0].plot(x, stats.norm.pdf(x), 'b-', lw=2, label='N(0,1)')

dfs = [3, 10, 30]
colors = ['red', 'green', 'purple']
for df, color in zip(dfs, colors):
    axes[0].plot(x, stats.t.pdf(x, df), linestyle='--', lw=2,
                 color=color, label=f't(df={df})')

axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].set_title('t-distribution vs Standard Normal Distribution')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Critical value change by degrees of freedom
dfs_range = np.arange(2, 101)
t_criticals = [stats.t.ppf(0.975, df) for df in dfs_range]
z_critical = stats.norm.ppf(0.975)

axes[1].plot(dfs_range, t_criticals, 'b-', lw=2, label='t(0.975, df)')
axes[1].axhline(z_critical, color='r', linestyle='--', lw=2,
                label=f'z(0.975) = {z_critical:.3f}')
axes[1].set_xlabel('Degrees of freedom (df)')
axes[1].set_ylabel('Critical value')
axes[1].set_title('95% Critical Value by Degrees of Freedom')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Tabular summary
print("95% confidence interval critical values by degrees of freedom:")
print("-" * 30)
for df in [5, 10, 20, 30, 50, 100, np.inf]:
    if df == np.inf:
        t_val = stats.norm.ppf(0.975)
        print(f"df = ∞ (normal): t = {t_val:.4f}")
    else:
        t_val = stats.t.ppf(0.975, df)
        print(f"df = {df:3.0f}: t = {t_val:.4f}")
```

---

## 3. Confidence Interval for Population Proportion

### 3.1 Confidence Interval Using Normal Approximation

```python
def ci_proportion(successes, n, confidence=0.95):
    """Confidence interval for population proportion (normal approximation)"""
    p_hat = successes / n
    se = np.sqrt(p_hat * (1 - p_hat) / n)

    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin = z_critical * se

    return p_hat - margin, p_hat + margin

# Example: Opinion poll
# 420 out of 1000 people agree
successes = 420
n = 1000

lower, upper = ci_proportion(successes, n, 0.95)

print("95% confidence interval for population proportion (normal approximation):")
print(f"Sample proportion: p̂ = {successes/n:.3f}")
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
print(f"Interpretation: Population proportion is estimated to be between {lower*100:.1f}% and {upper*100:.1f}%")

# Check normal approximation condition: np ≥ 10 and n(1-p) ≥ 10
p_hat = successes / n
print(f"\nNormal approximation condition check:")
print(f"np = {n * p_hat:.1f} ≥ 10? {n * p_hat >= 10}")
print(f"n(1-p) = {n * (1-p_hat):.1f} ≥ 10? {n * (1-p_hat) >= 10}")
```

### 3.2 Wilson Confidence Interval (Improved Method)

```python
def ci_proportion_wilson(successes, n, confidence=0.95):
    """Wilson confidence interval (more accurate method)"""
    p_hat = successes / n
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))

    return center - margin, center + margin

# Comparison with small sample
successes = 8
n = 20
p_hat = successes / n

lower_normal, upper_normal = ci_proportion(successes, n, 0.95)
lower_wilson, upper_wilson = ci_proportion_wilson(successes, n, 0.95)

print(f"Sample proportion: p̂ = {p_hat:.3f} (n={n})")
print(f"\nNormal approximation CI: [{lower_normal:.3f}, {upper_normal:.3f}]")
print(f"Wilson CI:                [{lower_wilson:.3f}, {upper_wilson:.3f}]")

# scipy's proportion_confint
from statsmodels.stats.proportion import proportion_confint

ci_wilson = proportion_confint(successes, n, alpha=0.05, method='wilson')
ci_normal = proportion_confint(successes, n, alpha=0.05, method='normal')

print(f"\nstatsmodels results:")
print(f"  Normal: [{ci_normal[0]:.3f}, {ci_normal[1]:.3f}]")
print(f"  Wilson: [{ci_wilson[0]:.3f}, {ci_wilson[1]:.3f}]")
```

### 3.3 Comparison of Proportion Confidence Intervals

```python
from statsmodels.stats.proportion import proportion_confint

# Comparison of various methods
successes = 15
n = 50
p_hat = successes / n

methods = ['normal', 'wilson', 'jeffreys', 'agresti_coull', 'beta']

print(f"Sample proportion = {p_hat:.3f} (successes={successes}, n={n})")
print("\nComparison of proportion confidence interval methods:")
print("-" * 50)

for method in methods:
    lower, upper = proportion_confint(successes, n, alpha=0.05, method=method)
    width = upper - lower
    print(f"{method:<15}: [{lower:.4f}, {upper:.4f}], Width={width:.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

y_positions = range(len(methods))
for i, method in enumerate(methods):
    lower, upper = proportion_confint(successes, n, alpha=0.05, method=method)
    ax.barh(i, upper-lower, left=lower, height=0.6, alpha=0.7)
    ax.scatter([p_hat], [i], color='red', s=50, zorder=5)

ax.axvline(p_hat, color='red', linestyle='--', alpha=0.5, label='Sample proportion')
ax.set_yticks(y_positions)
ax.set_yticklabels(methods)
ax.set_xlabel('Proportion')
ax.set_title('Comparison of Proportion Confidence Interval Methods')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## 4. Confidence Interval for Population Variance

### 4.1 Confidence Interval Using Chi-Square Distribution

```python
def ci_variance(sample, confidence=0.95):
    """Confidence interval for population variance (assuming normal population)"""
    n = len(sample)
    s_squared = np.var(sample, ddof=1)

    alpha = 1 - confidence
    chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)

    # Confidence interval: ((n-1)s² / χ²_upper, (n-1)s² / χ²_lower)
    var_lower = (n - 1) * s_squared / chi2_upper
    var_upper = (n - 1) * s_squared / chi2_lower

    return var_lower, var_upper

def ci_std(sample, confidence=0.95):
    """Confidence interval for population standard deviation"""
    var_lower, var_upper = ci_variance(sample, confidence)
    return np.sqrt(var_lower), np.sqrt(var_upper)

# Example: Variance estimation in quality control
np.random.seed(42)
sample = np.random.normal(100, 5, 30)  # True σ = 5

s_squared = np.var(sample, ddof=1)
s = np.std(sample, ddof=1)

var_lower, var_upper = ci_variance(sample, 0.95)
std_lower, std_upper = ci_std(sample, 0.95)

print("95% confidence interval for population variance:")
print(f"Sample variance: s² = {s_squared:.3f}")
print(f"Variance CI: [{var_lower:.3f}, {var_upper:.3f}]")

print(f"\n95% confidence interval for population standard deviation:")
print(f"Sample standard deviation: s = {s:.3f}")
print(f"Standard deviation CI: [{std_lower:.3f}, {std_upper:.3f}]")
print(f"Does the interval contain true σ = 5? {std_lower <= 5 <= std_upper}")
```

### 4.2 Asymmetry of Chi-Square Distribution

```python
# Why variance confidence intervals are not centered around the mean

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chi-square distribution
df = 20
x = np.linspace(0, 50, 200)
chi2_pdf = stats.chi2.pdf(x, df)

# Critical values
alpha = 0.05
chi2_lower = stats.chi2.ppf(alpha/2, df)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df)

axes[0].plot(x, chi2_pdf, 'b-', lw=2)
axes[0].fill_between(x, chi2_pdf, where=(x >= chi2_lower) & (x <= chi2_upper),
                      alpha=0.3, color='blue')
axes[0].axvline(chi2_lower, color='r', linestyle='--', label=f'χ²_L = {chi2_lower:.2f}')
axes[0].axvline(chi2_upper, color='r', linestyle='--', label=f'χ²_U = {chi2_upper:.2f}')
axes[0].axvline(df, color='g', linestyle=':', label=f'E[χ²] = {df}')
axes[0].set_xlabel('χ²')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Chi-Square Distribution (df={df})')
axes[0].legend()

# Simulation of variance confidence intervals
np.random.seed(42)
true_variance = 25  # σ² = 25
n = 21  # df = 20
n_simulations = 1000

var_estimates = []
ci_lowers = []
ci_uppers = []

for _ in range(n_simulations):
    sample = np.random.normal(0, np.sqrt(true_variance), n)
    s2 = np.var(sample, ddof=1)
    var_estimates.append(s2)

    lower = (n-1) * s2 / chi2_upper
    upper = (n-1) * s2 / chi2_lower
    ci_lowers.append(lower)
    ci_uppers.append(upper)

axes[1].hist(var_estimates, bins=50, density=True, alpha=0.7, label='Sample variance distribution')
axes[1].axvline(true_variance, color='r', linestyle='--', lw=2,
                label=f'σ² = {true_variance}')
axes[1].axvline(np.mean(var_estimates), color='g', linestyle=':',
                label=f'E[s²] = {np.mean(var_estimates):.2f}')
axes[1].set_xlabel('Sample variance')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Sample Variance')
axes[1].legend()

plt.tight_layout()
plt.show()

# Coverage check
coverage = np.mean([(l <= true_variance <= u)
                    for l, u in zip(ci_lowers, ci_uppers)])
print(f"Actual coverage of 95% confidence interval: {coverage*100:.1f}%")
```

---

## 5. Confidence Intervals for Comparing Two Groups

### 5.1 Confidence Interval for Difference of Two Population Means

```python
def ci_two_means_independent(sample1, sample2, confidence=0.95, equal_var=True):
    """Confidence interval for difference of two independent sample means"""
    n1, n2 = len(sample1), len(sample2)
    x1_bar, x2_bar = sample1.mean(), sample2.mean()
    s1, s2 = sample1.std(ddof=1), sample2.std(ddof=1)

    if equal_var:
        # Pooled variance
        sp_squared = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)
        se = np.sqrt(sp_squared * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test (no equal variance assumption)
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        # Welch-Satterthwaite degrees of freedom
        df = (s1**2/n1 + s2**2/n2)**2 / \
             ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_critical * se
    diff = x1_bar - x2_bar

    return diff - margin, diff + margin, diff, df

# Example: A/B test results
np.random.seed(42)
group_A = np.random.normal(105, 15, 50)  # Treatment group
group_B = np.random.normal(100, 15, 50)  # Control group

# Equal variance assumption
lower_eq, upper_eq, diff, df = ci_two_means_independent(group_A, group_B,
                                                        equal_var=True)
print("95% confidence interval for difference of two population means:")
print(f"Group A mean: {group_A.mean():.2f}, Group B mean: {group_B.mean():.2f}")
print(f"Difference (A - B): {diff:.2f}")
print(f"Equal variance CI (df={df:.0f}): [{lower_eq:.2f}, {upper_eq:.2f}]")

# No equal variance assumption (Welch)
lower_welch, upper_welch, diff, df_welch = ci_two_means_independent(group_A, group_B,
                                                                     equal_var=False)
print(f"Welch CI (df={df_welch:.1f}): [{lower_welch:.2f}, {upper_welch:.2f}]")

# scipy verification
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(group_A, group_B, equal_var=False)
print(f"\nscipy t-test statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
```

### 5.2 Confidence Interval for Paired Sample Mean Difference

```python
def ci_paired_difference(before, after, confidence=0.95):
    """Confidence interval for paired sample mean difference"""
    diff = after - before
    n = len(diff)
    d_bar = diff.mean()
    s_d = diff.std(ddof=1)
    se = s_d / np.sqrt(n)

    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_critical * se

    return d_bar - margin, d_bar + margin, d_bar

# Example: Weight before and after diet program
np.random.seed(42)
weight_before = np.random.normal(75, 10, 30)
weight_after = weight_before - np.random.normal(3, 2, 30)  # Average 3kg reduction

lower, upper, mean_diff = ci_paired_difference(weight_before, weight_after, 0.95)

print("95% confidence interval for paired sample mean difference:")
print(f"Before: {weight_before.mean():.2f} kg, After: {weight_after.mean():.2f} kg")
print(f"Mean change: {mean_diff:.2f} kg")
print(f"95% CI: [{lower:.2f}, {upper:.2f}] kg")

if upper < 0:
    print("→ Interval does not contain 0, weight reduction is significant")
```

### 5.3 Confidence Interval for Difference of Two Population Proportions

```python
def ci_two_proportions(successes1, n1, successes2, n2, confidence=0.95):
    """Confidence interval for difference of two population proportions"""
    p1 = successes1 / n1
    p2 = successes2 / n2
    diff = p1 - p2

    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin = z_critical * se

    return diff - margin, diff + margin, diff

# Example: Comparing click-through rates of two ads
# Ad A: 1000 impressions, 150 clicks
# Ad B: 1200 impressions, 132 clicks

lower, upper, diff = ci_two_proportions(150, 1000, 132, 1200, 0.95)

print("95% confidence interval for difference of two population proportions:")
print(f"Ad A click rate: {150/1000:.3f}")
print(f"Ad B click rate: {132/1200:.3f}")
print(f"Difference: {diff:.4f} ({diff*100:.2f}%p)")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

if lower > 0:
    print("→ Ad A's click rate is significantly higher")
elif upper < 0:
    print("→ Ad B's click rate is significantly higher")
else:
    print("→ No significant difference in click rates between the two ads")
```

---

## 6. Bootstrap Confidence Intervals

### 6.1 Introduction to Bootstrap Method

```python
def bootstrap_ci(data, statistic_func, confidence=0.95, n_bootstrap=10000):
    """Bootstrap confidence interval (percentile method)"""
    np.random.seed(42)
    n = len(data)
    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        # Generate bootstrap sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_statistics.append(statistic_func(bootstrap_sample))

    # Percentile confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_statistics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2))

    return lower, upper, np.array(bootstrap_statistics)

# Example: Bootstrap confidence interval for mean
np.random.seed(42)
sample = np.random.exponential(scale=5, size=50)  # Asymmetric distribution

lower_boot, upper_boot, boot_means = bootstrap_ci(sample, np.mean, 0.95)

# Compare with traditional t-distribution method
lower_t, upper_t, _ = ci_mean_t(sample, 0.95)

print("Comparison of 95% confidence intervals for mean:")
print(f"Sample mean: {sample.mean():.3f}")
print(f"t-distribution:  [{lower_t:.3f}, {upper_t:.3f}]")
print(f"Bootstrap:       [{lower_boot:.3f}, {upper_boot:.3f}]")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original data
axes[0].hist(sample, bins=20, density=True, alpha=0.7, edgecolor='black')
axes[0].axvline(sample.mean(), color='r', linestyle='--', linewidth=2, label='Sample mean')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].set_title('Original Data (Exponential Distribution)')
axes[0].legend()

# Bootstrap distribution
axes[1].hist(boot_means, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1].axvline(lower_boot, color='r', linestyle='--', label=f'95% CI: [{lower_boot:.2f}, {upper_boot:.2f}]')
axes[1].axvline(upper_boot, color='r', linestyle='--')
axes[1].axvline(sample.mean(), color='g', linestyle='-', linewidth=2, label='Sample mean')
axes[1].set_xlabel('Bootstrap sample mean')
axes[1].set_ylabel('Density')
axes[1].set_title('Bootstrap Distribution (10,000 iterations)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 6.2 Various Bootstrap Methods

```python
def bootstrap_ci_bca(data, statistic_func, confidence=0.95, n_bootstrap=10000):
    """BCa (Bias-Corrected and Accelerated) bootstrap confidence interval"""
    from scipy.stats import norm

    np.random.seed(42)
    n = len(data)
    original_stat = statistic_func(data)

    # Bootstrap statistics
    boot_stats = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_func(boot_sample))
    boot_stats = np.array(boot_stats)

    # Bias correction
    z0 = norm.ppf(np.mean(boot_stats < original_stat))

    # Acceleration coefficient - using jackknife
    jackknife_stats = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_stats.append(statistic_func(jack_sample))
    jackknife_stats = np.array(jackknife_stats)
    jack_mean = jackknife_stats.mean()
    a = np.sum((jack_mean - jackknife_stats)**3) / \
        (6 * (np.sum((jack_mean - jackknife_stats)**2))**1.5 + 1e-10)

    # Calculate BCa percentiles
    alpha = 1 - confidence
    z_alpha_lower = norm.ppf(alpha / 2)
    z_alpha_upper = norm.ppf(1 - alpha / 2)

    alpha_lower = norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    alpha_upper = norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

    lower = np.percentile(boot_stats, 100 * alpha_lower)
    upper = np.percentile(boot_stats, 100 * alpha_upper)

    return lower, upper

# Comparison
np.random.seed(42)
sample = np.random.exponential(scale=5, size=30)

# Percentile method
lower_pct, upper_pct, _ = bootstrap_ci(sample, np.mean, 0.95)

# BCa method
lower_bca, upper_bca = bootstrap_ci_bca(sample, np.mean, 0.95)

print("Comparison of bootstrap methods:")
print(f"Sample mean: {sample.mean():.3f}")
print(f"Percentile:  [{lower_pct:.3f}, {upper_pct:.3f}]")
print(f"BCa:         [{lower_bca:.3f}, {upper_bca:.3f}]")
```

### 6.3 Bootstrap Confidence Interval for Median

```python
# Using bootstrap for statistics with unknown distributions like median

np.random.seed(42)
sample = np.random.lognormal(mean=3, sigma=0.5, size=100)

lower_median, upper_median, boot_medians = bootstrap_ci(sample, np.median, 0.95)

print("95% bootstrap confidence interval for median:")
print(f"Sample median: {np.median(sample):.3f}")
print(f"95% CI: [{lower_median:.3f}, {upper_median:.3f}]")

# Bootstrap confidence interval for correlation coefficient
np.random.seed(42)
x = np.random.normal(0, 1, 50)
y = 0.7 * x + np.random.normal(0, 0.5, 50)
data_combined = np.column_stack([x, y])

def correlation(data):
    return np.corrcoef(data[:, 0], data[:, 1])[0, 1]

lower_corr, upper_corr, boot_corrs = bootstrap_ci(data_combined, correlation, 0.95)

print(f"\n95% bootstrap confidence interval for correlation coefficient:")
print(f"Sample correlation coefficient: {np.corrcoef(x, y)[0, 1]:.3f}")
print(f"95% CI: [{lower_corr:.3f}, {upper_corr:.3f}]")
```

---

## 7. Confidence Interval Calculation Using scipy

### 7.1 Basic Confidence Interval Functions

```python
# Using scipy.stats interval method

# Confidence interval for population mean from normal distribution
np.random.seed(42)
sample = np.random.normal(100, 15, 50)

# Using t-distribution
mean = sample.mean()
sem = stats.sem(sample)  # Standard error

ci_95 = stats.t.interval(confidence=0.95, df=len(sample)-1, loc=mean, scale=sem)
ci_99 = stats.t.interval(confidence=0.99, df=len(sample)-1, loc=mean, scale=sem)

print("Using scipy.stats.t.interval:")
print(f"Sample mean: {mean:.2f}")
print(f"95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
print(f"99% CI: [{ci_99[0]:.2f}, {ci_99[1]:.2f}]")

# bootstrap method (scipy >= 1.9)
from scipy.stats import bootstrap

np.random.seed(42)
sample_tuple = (sample,)  # Must be passed as tuple
result = bootstrap(sample_tuple, np.mean, confidence_level=0.95, n_resamples=9999)

print(f"\nUsing scipy.stats.bootstrap:")
print(f"95% CI: [{result.confidence_interval.low:.2f}, {result.confidence_interval.high:.2f}]")
```

### 7.2 Using statsmodels

```python
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

# Calculate confidence interval with DescrStatsW
np.random.seed(42)
sample = np.random.normal(100, 15, 50)

d = DescrStatsW(sample)

print("Using statsmodels DescrStatsW:")
print(f"Sample mean: {d.mean:.2f}")
print(f"95% CI: {d.tconfint_mean(alpha=0.05)}")
print(f"99% CI: {d.tconfint_mean(alpha=0.01)}")

# Comparing two samples
group1 = np.random.normal(100, 15, 40)
group2 = np.random.normal(105, 15, 45)

from statsmodels.stats.weightstats import CompareMeans

cm = CompareMeans(DescrStatsW(group1), DescrStatsW(group2))
print(f"\n95% CI for difference of two means: {cm.tconfint_diff(alpha=0.05)}")
```

---

## Practice Problems

### Problem 1: t-confidence Interval
Construct a 95% confidence interval for the population mean from the following sample.
```python
sample = [23, 25, 28, 22, 26, 24, 27, 25, 29, 24]
```

### Problem 2: Confidence Interval for Proportion
In a survey of 500 people, 230 agreed.
- (a) Construct a 95% confidence interval for the population proportion (normal approximation)
- (b) Recalculate using the Wilson method
- (c) How does the interval width change if the sample size is increased to 2000?

### Problem 3: Bootstrap
Construct 95% bootstrap confidence intervals for both the mean and median from the following asymmetric distribution.
```python
np.random.seed(42)
data = np.random.exponential(10, 40)
```

---

## Summary

| CI Type | Condition | Formula | Python |
|---------|-----------|---------|--------|
| Mean (σ known) | Z-interval | x̄ ± z·σ/√n | `stats.norm.interval()` |
| Mean (σ unknown) | t-interval | x̄ ± t·s/√n | `stats.t.interval()` |
| Proportion | np ≥ 10 | p̂ ± z·√(p̂(1-p̂)/n) | `proportion_confint()` |
| Variance | Normal population | χ² distribution | Direct calculation |
| Bootstrap | Nonparametric | Resampling | `stats.bootstrap()` |

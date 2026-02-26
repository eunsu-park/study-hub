# 23. Nonparametric Statistics

[Previous: Multivariate Analysis](./22_Multivariate_Analysis.md) | [Next: Experimental Design](./24_Experimental_Design.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify situations where nonparametric tests are more appropriate than their parametric counterparts
2. Apply the Mann-Whitney U test to compare two independent groups and interpret the rank-biserial effect size
3. Apply the Wilcoxon signed-rank test for paired samples and one-sample median tests
4. Compare three or more groups using the Kruskal-Wallis H test and perform Bonferroni-corrected post-hoc comparisons
5. Distinguish between Spearman and Kendall rank correlations and select the appropriate one for a given data type
6. Calculate and interpret effect sizes for nonparametric tests (rank-biserial r, eta-squared, Kendall's W)
7. Implement a comprehensive nonparametric analysis workflow that includes normality testing, test selection, and visualization

---

Not all data plays by the rules of the normal distribution. Skewed distributions, small samples, ordinal scales, and stubborn outliers can all violate the assumptions that parametric tests depend on. Nonparametric methods free you from these distributional constraints by working with ranks rather than raw values, giving you robust and valid inference even when the textbook assumptions break down.

---

## 1. When Nonparametric Tests Are Needed

### 1.1 When to Use Nonparametric Tests?

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

def when_to_use_nonparametric():
    """When to use nonparametric tests"""
    print("""
    ================================================
    When to Use Nonparametric Tests
    ================================================

    1. Violation of Normality
       - Data does not follow a normal distribution
       - Rejection by Shapiro-Wilk test, etc.

    2. Small Sample Size
       - When n < 30, CLT is difficult to apply
       - Normality tests also lack power

    3. Ordinal/Rank Data
       - Likert scales (1-5 points)
       - Ranked data

    4. Presence of Outliers
       - Compare medians instead of means (which are sensitive to extreme values)
       - Rank-based methods are robust to outliers

    5. Violation of Homogeneity Assumption
       - Violation of variance homogeneity (equal variance) assumption

    ================================================
    Advantages and Disadvantages of Nonparametric Tests
    ================================================

    Advantages:
    - No distributional assumptions required
    - Robust to outliers
    - Suitable for ordinal data

    Disadvantages:
    - Lower power than parametric tests when assumptions are met
    - Effect size interpretation can be difficult
    - Difficult to apply to some complex designs
    """)

when_to_use_nonparametric()

# Normality test example
def check_normality(data, alpha=0.05):
    """Perform normality test"""
    # Shapiro-Wilk
    stat_sw, p_sw = stats.shapiro(data)

    # D'Agostino-Pearson
    if len(data) >= 20:
        stat_da, p_da = stats.normaltest(data)
    else:
        stat_da, p_da = np.nan, np.nan

    print("=== Normality Tests ===")
    print(f"Sample size: {len(data)}")
    print(f"Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f}")
    if not np.isnan(p_da):
        print(f"D'Agostino-Pearson: p={p_da:.4f}")

    is_normal = p_sw > alpha
    print(f"Conclusion: {'Can be considered normal' if is_normal else 'Not normal'} (α={alpha})")

    return is_normal

# Normal data
normal_data = np.random.normal(50, 10, 30)
check_normality(normal_data)

print()

# Non-normal data (exponential distribution)
skewed_data = np.random.exponential(10, 30)
check_normality(skewed_data)
```

### 1.2 Parametric vs Nonparametric Test Correspondence

| Parametric Test | Nonparametric Test | Situation |
|----------------|-------------------|-----------|
| 1-sample t-test | Wilcoxon signed-rank test | Single sample, median test |
| Independent t-test | Mann-Whitney U | Two independent samples |
| Paired t-test | Wilcoxon signed-rank test | Paired samples |
| One-way ANOVA | Kruskal-Wallis H | 3+ independent samples |
| Repeated measures ANOVA | Friedman | 3+ paired samples |
| Pearson correlation | Spearman/Kendall | Correlation |

---

## 2. Mann-Whitney U Test

### 2.1 Concept

**Purpose**: Compare distributions of two independent samples (median or distribution location)

**Hypotheses**:
- H₀: The two groups have identical distributions
- H₁: The two groups have different distributions (or one is probabilistically larger)

**Test statistic**: U (based on rank sum)

```python
def mann_whitney_example():
    """Mann-Whitney U test example"""
    np.random.seed(42)

    # Scenario: Compare treatment effects of two groups (normality violation)
    # Group A: Control
    # Group B: Treatment

    group_a = np.random.exponential(20, 25)  # Non-normal
    group_b = np.random.exponential(25, 25) + 5  # Non-normal, larger values

    print("=== Mann-Whitney U Test ===")
    print(f"\nGroup A: n={len(group_a)}, median={np.median(group_a):.2f}")
    print(f"Group B: n={len(group_b)}, median={np.median(group_b):.2f}")

    # Normality tests
    print("\nNormality tests:")
    _, p_a = stats.shapiro(group_a)
    _, p_b = stats.shapiro(group_b)
    print(f"  Group A: p={p_a:.4f}")
    print(f"  Group B: p={p_b:.4f}")

    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

    print(f"\nMann-Whitney U test:")
    print(f"  U statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("  Conclusion: Significant difference between groups (p < 0.05)")
    else:
        print("  Conclusion: No significant difference between groups (p >= 0.05)")

    # Effect size: rank-biserial correlation
    n1, n2 = len(group_a), len(group_b)
    r = 1 - (2 * statistic) / (n1 * n2)
    print(f"\nEffect size (rank-biserial r): {r:.3f}")
    print(f"  |r| < 0.1: Small effect")
    print(f"  0.1 <= |r| < 0.3: Medium effect")
    print(f"  |r| >= 0.3: Large effect")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Boxplot
    ax = axes[0]
    ax.boxplot([group_a, group_b], labels=['Group A', 'Group B'])
    ax.set_ylabel('Value')
    ax.set_title('Boxplot')
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(group_a, bins=15, alpha=0.5, label='Group A', density=True)
    ax.hist(group_b, bins=15, alpha=0.5, label='Group B', density=True)
    ax.axvline(np.median(group_a), color='blue', linestyle='--', label='A median')
    ax.axvline(np.median(group_b), color='orange', linestyle='--', label='B median')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rank distribution
    ax = axes[2]
    all_data = np.concatenate([group_a, group_b])
    ranks = stats.rankdata(all_data)
    ranks_a = ranks[:len(group_a)]
    ranks_b = ranks[len(group_a):]
    ax.hist(ranks_a, bins=15, alpha=0.5, label='Group A ranks')
    ax.hist(ranks_b, bins=15, alpha=0.5, label='Group B ranks')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title('Rank Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return group_a, group_b

group_a, group_b = mann_whitney_example()
```

### 2.2 One-sided Tests

```python
# One-sided test: Test if Group B is larger than Group A
stat_greater, p_greater = stats.mannwhitneyu(group_a, group_b, alternative='less')
print(f"One-sided test (B > A): p = {p_greater:.4f}")

stat_less, p_less = stats.mannwhitneyu(group_a, group_b, alternative='greater')
print(f"One-sided test (A > B): p = {p_less:.4f}")
```

---

## 3. Wilcoxon Signed-Rank Test

### 3.1 Paired Sample Comparison

**Purpose**: Test if the difference between paired measurements is zero

**Use case**: Before/after comparison, paired samples

```python
def wilcoxon_signed_rank_example():
    """Wilcoxon signed-rank test example"""
    np.random.seed(42)

    # Scenario: Weight change before and after diet program
    n = 20
    before = np.random.normal(80, 10, n)
    # Mean 3kg reduction effect + non-normal error
    after = before - 3 + np.random.exponential(2, n) - np.random.exponential(2, n)

    diff = after - before

    print("=== Wilcoxon Signed-Rank Test ===")
    print(f"\nSample size: {n}")
    print(f"Before: mean={before.mean():.2f}, median={np.median(before):.2f}")
    print(f"After: mean={after.mean():.2f}, median={np.median(after):.2f}")
    print(f"Difference: mean={diff.mean():.2f}, median={np.median(diff):.2f}")

    # Normality test on differences
    _, p_norm = stats.shapiro(diff)
    print(f"\nNormality of differences: p={p_norm:.4f}")

    # Wilcoxon test
    statistic, p_value = stats.wilcoxon(before, after, alternative='two-sided')

    print(f"\nWilcoxon test:")
    print(f"  W statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # Comparison with paired t-test
    t_stat, t_p = stats.ttest_rel(before, after)
    print(f"\nPaired t-test (for comparison):")
    print(f"  t statistic: {t_stat:.2f}")
    print(f"  p-value: {t_p:.4f}")

    # Effect size
    r = statistic / (n * (n + 1) / 2)
    print(f"\nEffect size (r): {abs(1-2*r):.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Before/After comparison
    ax = axes[0]
    ax.boxplot([before, after], labels=['Before', 'After'])
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Before/After Comparison')
    ax.grid(True, alpha=0.3)

    # Individual changes
    ax = axes[1]
    for i in range(n):
        ax.plot([0, 1], [before[i], after[i]], 'b-', alpha=0.5)
    ax.plot([0, 1], [before.mean(), after.mean()], 'r-', linewidth=2, label='Mean')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before', 'After'])
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Individual Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference distribution
    ax = axes[2]
    ax.hist(diff, bins=10, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='No change')
    ax.axvline(np.median(diff), color='g', linestyle='-', label=f'Median={np.median(diff):.2f}')
    ax.set_xlabel('Difference (After - Before)')
    ax.set_ylabel('Density')
    ax.set_title('Difference Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return before, after

before, after = wilcoxon_signed_rank_example()
```

### 3.2 One-sample Test (Median Test)

```python
# One-sample: Test if median equals a specific value
def one_sample_wilcoxon(data, hypothesized_median):
    """
    One-sample Wilcoxon test
    H0: median = hypothesized_median
    """
    diff_from_median = data - hypothesized_median

    # Exclude zero values
    diff_from_median = diff_from_median[diff_from_median != 0]

    if len(diff_from_median) == 0:
        print("All values equal the hypothesized median.")
        return

    stat, p_value = stats.wilcoxon(diff_from_median)

    print(f"=== One-sample Wilcoxon Test ===")
    print(f"H0: median = {hypothesized_median}")
    print(f"Sample median: {np.median(data):.2f}")
    print(f"W statistic: {stat:.2f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: Median is significantly different from {hypothesized_median}")
    else:
        print(f"Conclusion: Insufficient evidence that median differs from {hypothesized_median}")

# Example
sample_data = np.random.exponential(10, 30) + 5
one_sample_wilcoxon(sample_data, hypothesized_median=10)
```

---

## 4. Kruskal-Wallis H Test

### 4.1 Concept

**Purpose**: Compare distributions of 3 or more independent groups (nonparametric alternative to one-way ANOVA)

**Hypotheses**:
- H₀: All groups have identical distributions
- H₁: At least one group differs

```python
def kruskal_wallis_example():
    """Kruskal-Wallis H test example"""
    np.random.seed(42)

    # Scenario: Compare effects of 3 teaching methods
    method_a = np.random.exponential(10, 25) + 60  # Traditional
    method_b = np.random.exponential(10, 25) + 65  # Online
    method_c = np.random.exponential(10, 25) + 70  # Blended

    print("=== Kruskal-Wallis H Test ===")
    print(f"\nMethod A: n={len(method_a)}, median={np.median(method_a):.2f}")
    print(f"Method B: n={len(method_b)}, median={np.median(method_b):.2f}")
    print(f"Method C: n={len(method_c)}, median={np.median(method_c):.2f}")

    # Normality tests
    print("\nNormality tests:")
    for name, data in [('A', method_a), ('B', method_b), ('C', method_c)]:
        _, p = stats.shapiro(data)
        print(f"  {name}: p={p:.4f}")

    # Kruskal-Wallis test
    H_stat, p_value = stats.kruskal(method_a, method_b, method_c)

    print(f"\nKruskal-Wallis test:")
    print(f"  H statistic: {H_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # Effect size: eta-squared
    N = len(method_a) + len(method_b) + len(method_c)
    k = 3
    eta_sq = (H_stat - k + 1) / (N - k)
    print(f"\nEffect size (η²): {eta_sq:.3f}")
    print("  0.01: Small effect")
    print("  0.06: Medium effect")
    print("  0.14: Large effect")

    # Comparison with one-way ANOVA
    F_stat, anova_p = stats.f_oneway(method_a, method_b, method_c)
    print(f"\nOne-way ANOVA (for comparison):")
    print(f"  F statistic: {F_stat:.2f}")
    print(f"  p-value: {anova_p:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    ax = axes[0]
    ax.boxplot([method_a, method_b, method_c],
               labels=['Method A', 'Method B', 'Method C'])
    ax.set_ylabel('Score')
    ax.set_title('Score Distribution by Teaching Method')
    ax.grid(True, alpha=0.3)

    # Violin plot
    ax = axes[1]
    parts = ax.violinplot([method_a, method_b, method_c], positions=[1, 2, 3],
                           showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Method A', 'Method B', 'Method C'])
    ax.set_ylabel('Score')
    ax.set_title('Violin Plot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return method_a, method_b, method_c

method_a, method_b, method_c = kruskal_wallis_example()
```

### 4.2 Post-hoc Tests

```python
from itertools import combinations
from scipy.stats import mannwhitneyu

def dunn_test_alternative(groups, group_names, alpha=0.05):
    """
    Alternative to Dunn test: Bonferroni-corrected Mann-Whitney U tests
    """
    n_comparisons = len(list(combinations(range(len(groups)), 2)))
    adjusted_alpha = alpha / n_comparisons

    print(f"=== Post-hoc Tests (Bonferroni correction) ===")
    print(f"Number of comparisons: {n_comparisons}")
    print(f"Adjusted α: {adjusted_alpha:.4f}")
    print()

    results = []
    for (i, j) in combinations(range(len(groups)), 2):
        stat, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        significant = p < adjusted_alpha
        results.append({
            'comparison': f'{group_names[i]} vs {group_names[j]}',
            'U': stat,
            'p-value': p,
            'significant': significant
        })
        print(f"{group_names[i]} vs {group_names[j]}: U={stat:.1f}, p={p:.4f} {'*' if significant else ''}")

    return pd.DataFrame(results)

groups = [method_a, method_b, method_c]
group_names = ['A', 'B', 'C']
posthoc_results = dunn_test_alternative(groups, group_names)
```

---

## 5. Friedman Test

### 5.1 Concept

**Purpose**: Compare 3 or more conditions in repeated measures or blocked designs

**Use case**: Same subjects measured under multiple conditions

```python
def friedman_example():
    """Friedman test example"""
    np.random.seed(42)

    # Scenario: Same students' scores on 3 exams
    n_students = 20

    # Generate correlated data (multiple exams for same students)
    ability = np.random.normal(70, 10, n_students)  # Baseline ability
    exam1 = ability + np.random.normal(0, 5, n_students)
    exam2 = ability + np.random.normal(3, 5, n_students)  # Slightly harder
    exam3 = ability + np.random.normal(6, 5, n_students)  # Harder

    print("=== Friedman Test ===")
    print(f"\nNumber of students: {n_students}")
    print(f"Exam 1: median={np.median(exam1):.2f}")
    print(f"Exam 2: median={np.median(exam2):.2f}")
    print(f"Exam 3: median={np.median(exam3):.2f}")

    # Friedman test
    stat, p_value = stats.friedmanchisquare(exam1, exam2, exam3)

    print(f"\nFriedman test:")
    print(f"  χ² statistic: {stat:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # Effect size: Kendall's W
    k = 3  # Number of conditions
    W = stat / (n_students * (k - 1))
    print(f"\nEffect size (Kendall's W): {W:.3f}")
    print("  W < 0.3: Weak agreement")
    print("  0.3 <= W < 0.5: Moderate agreement")
    print("  W >= 0.5: Strong agreement")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    ax = axes[0]
    ax.boxplot([exam1, exam2, exam3], labels=['Exam 1', 'Exam 2', 'Exam 3'])
    ax.set_ylabel('Score')
    ax.set_title('Score Distribution by Exam')
    ax.grid(True, alpha=0.3)

    # Individual profiles
    ax = axes[1]
    for i in range(min(10, n_students)):  # First 10 students only
        ax.plot([1, 2, 3], [exam1[i], exam2[i], exam3[i]], 'o-', alpha=0.5)
    ax.plot([1, 2, 3], [exam1.mean(), exam2.mean(), exam3.mean()],
            'rs-', linewidth=2, markersize=8, label='Mean')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Exam 1', 'Exam 2', 'Exam 3'])
    ax.set_ylabel('Score')
    ax.set_title('Individual Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return exam1, exam2, exam3

exam1, exam2, exam3 = friedman_example()
```

### 5.2 Nemenyi Post-hoc Test

```python
def nemenyi_posthoc(groups, group_names, alpha=0.05):
    """
    Nemenyi post-hoc test (after Friedman)
    Bonferroni-corrected Wilcoxon signed-rank tests
    """
    n_comparisons = len(list(combinations(range(len(groups)), 2)))
    adjusted_alpha = alpha / n_comparisons

    print(f"=== Nemenyi Post-hoc Test ===")
    print(f"Number of comparisons: {n_comparisons}")
    print(f"Adjusted α: {adjusted_alpha:.4f}")
    print()

    results = []
    for (i, j) in combinations(range(len(groups)), 2):
        stat, p = stats.wilcoxon(groups[i], groups[j])
        significant = p < adjusted_alpha
        results.append({
            'comparison': f'{group_names[i]} vs {group_names[j]}',
            'W': stat,
            'p-value': p,
            'significant': significant
        })
        print(f"{group_names[i]} vs {group_names[j]}: W={stat:.1f}, p={p:.4f} {'*' if significant else ''}")

    return pd.DataFrame(results)

groups = [exam1, exam2, exam3]
group_names = ['Exam1', 'Exam2', 'Exam3']
nemenyi_results = nemenyi_posthoc(groups, group_names)
```

---

## 6. Nonparametric Correlation (Spearman, Kendall)

### 6.1 Spearman Rank Correlation

**Characteristics**: Rank-based, measures monotonic relationships (need not be linear)

```python
def spearman_correlation_example():
    """Spearman correlation example"""
    np.random.seed(42)

    # Nonlinear relationship data
    n = 50
    x = np.random.uniform(0, 10, n)
    y = np.log(x + 1) + np.random.normal(0, 0.3, n)  # Log relationship + noise

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print("=== Correlation Analysis ===")
    print(f"\nPearson correlation: r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman correlation: ρ={spearman_r:.4f}, p={spearman_p:.4f}")
    print("\nSpearman is more appropriate for nonlinear relationships")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original data
    ax = axes[0]
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Original Data\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}')
    ax.grid(True, alpha=0.3)

    # Rank transformation
    ax = axes[1]
    x_ranks = stats.rankdata(x)
    y_ranks = stats.rankdata(y)
    ax.scatter(x_ranks, y_ranks, alpha=0.7)
    ax.set_xlabel('X Rank')
    ax.set_ylabel('Y Rank')
    ax.set_title('After Rank Transformation')
    ax.grid(True, alpha=0.3)

    # With outlier
    ax = axes[2]
    x_outlier = np.append(x, [50])  # Extreme outlier
    y_outlier = np.append(y, [y.mean()])

    pearson_out, _ = stats.pearsonr(x_outlier, y_outlier)
    spearman_out, _ = stats.spearmanr(x_outlier, y_outlier)

    ax.scatter(x_outlier[:-1], y_outlier[:-1], alpha=0.7, label='Normal data')
    ax.scatter(x_outlier[-1], y_outlier[-1], color='r', s=100, label='Outlier', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'With Outlier\nPearson={pearson_out:.3f}, Spearman={spearman_out:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return x, y

x, y = spearman_correlation_example()
```

### 6.2 Kendall Rank Correlation

**Characteristics**: Based on pairwise comparisons, more robust than Spearman

```python
def kendall_correlation_example():
    """Kendall tau correlation example"""
    np.random.seed(42)

    # Ordinal data (Likert scale, etc.)
    n = 30
    rater1 = np.random.randint(1, 6, n)  # 1-5 points
    # Somewhat consistent but not identical
    rater2 = np.clip(rater1 + np.random.randint(-1, 2, n), 1, 5)

    # Pearson (inappropriate)
    pearson_r, _ = stats.pearsonr(rater1, rater2)

    # Spearman
    spearman_r, _ = stats.spearmanr(rater1, rater2)

    # Kendall
    kendall_tau, kendall_p = stats.kendalltau(rater1, rater2)

    print("=== Ordinal Data Correlation Analysis ===")
    print(f"\nRater 1 distribution: {np.bincount(rater1)[1:]}")
    print(f"Rater 2 distribution: {np.bincount(rater2)[1:]}")
    print(f"\nPearson r: {pearson_r:.4f} (inappropriate for ordinal data)")
    print(f"Spearman ρ: {spearman_r:.4f}")
    print(f"Kendall τ: {kendall_tau:.4f}, p={kendall_p:.4f}")

    print("\nInterpretation:")
    print("  τ = 0: Equal concordant/discordant pairs")
    print("  τ = 1: Perfect rank agreement")
    print("  τ = -1: Perfect rank disagreement")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))

    # Add jitter (to prevent overlapping same scores)
    jitter1 = rater1 + np.random.uniform(-0.1, 0.1, n)
    jitter2 = rater2 + np.random.uniform(-0.1, 0.1, n)

    ax.scatter(jitter1, jitter2, alpha=0.7, s=50)
    ax.plot([0, 6], [0, 6], 'r--', label='Perfect agreement line')
    ax.set_xlabel('Rater 1')
    ax.set_ylabel('Rater 2')
    ax.set_title(f'Inter-rater Agreement\nKendall τ = {kendall_tau:.3f}')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()

    return rater1, rater2

rater1, rater2 = kendall_correlation_example()
```

### 6.3 Correlation Coefficient Comparison

```python
def compare_correlations():
    """Compare three correlation coefficients"""
    print("""
    =================================================
    Correlation Coefficient Comparison
    =================================================

    | Characteristic | Pearson | Spearman | Kendall |
    |----------------|---------|----------|---------|
    | Relationship | Linear | Monotonic | Monotonic |
    | Data type | Continuous | Ordinal/Continuous | Ordinal |
    | Outliers | Sensitive | Robust | Very robust |
    | Ties handling | N/A | Average rank | Correctable |
    | Computation | O(n) | O(n log n) | O(n²) |
    | Range | [-1, 1] | [-1, 1] | [-1, 1] |

    Selection criteria:
    - Continuous & linear relationship → Pearson
    - Non-normal/outliers/nonlinear → Spearman
    - Ordinal/ranked data → Kendall
    - Small sample size → Kendall
    """)

compare_correlations()
```

---

## 7. Practice Examples

### 7.1 Comprehensive Nonparametric Analysis

```python
def comprehensive_nonparametric_analysis(data_dict, alpha=0.05):
    """
    Perform comprehensive nonparametric analysis

    Parameters:
    -----------
    data_dict : dict
        Format: {group_name: data}
    """
    print("="*60)
    print("Comprehensive Nonparametric Analysis")
    print("="*60)

    groups = list(data_dict.values())
    group_names = list(data_dict.keys())
    n_groups = len(groups)

    # 1. Descriptive statistics
    print("\n[1] Descriptive Statistics")
    for name, data in data_dict.items():
        print(f"  {name}: n={len(data)}, median={np.median(data):.2f}, "
              f"IQR={np.percentile(data, 75)-np.percentile(data, 25):.2f}")

    # 2. Normality tests
    print("\n[2] Normality Tests (Shapiro-Wilk)")
    all_normal = True
    for name, data in data_dict.items():
        _, p = stats.shapiro(data)
        is_normal = p > alpha
        all_normal = all_normal and is_normal
        print(f"  {name}: p={p:.4f} {'(normal)' if is_normal else '(non-normal)'}")

    # 3. Select and perform appropriate test
    print(f"\n[3] Group Comparison Test (number of groups: {n_groups})")

    if n_groups == 2:
        # Two groups: Mann-Whitney U
        stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        print(f"  Mann-Whitney U: U={stat:.2f}, p={p:.4f}")
        test_name = "Mann-Whitney U"
    else:
        # Three or more groups: Kruskal-Wallis
        stat, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis H: H={stat:.2f}, p={p:.4f}")
        test_name = "Kruskal-Wallis"

        # Post-hoc if significant
        if p < alpha:
            print("\n  Post-hoc tests (Bonferroni correction):")
            n_comp = n_groups * (n_groups - 1) // 2
            adj_alpha = alpha / n_comp
            for (i, j) in combinations(range(n_groups), 2):
                _, ph_p = stats.mannwhitneyu(groups[i], groups[j])
                sig = '*' if ph_p < adj_alpha else ''
                print(f"    {group_names[i]} vs {group_names[j]}: p={ph_p:.4f} {sig}")

    # 4. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    ax = axes[0]
    ax.boxplot(groups, labels=group_names)
    ax.set_ylabel('Value')
    ax.set_title(f'{test_name} Test\np = {p:.4f}')
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    for name, data in data_dict.items():
        ax.hist(data, bins=15, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Usage example
np.random.seed(42)
data = {
    'Control': np.random.exponential(10, 30) + 50,
    'Treatment A': np.random.exponential(10, 30) + 55,
    'Treatment B': np.random.exponential(10, 30) + 60
}
comprehensive_nonparametric_analysis(data)
```

---

## 8. Practice Problems

### Problem 1: Test Selection
Select an appropriate nonparametric test for each situation:
1. Comparing test scores between male and female students (skewed score distribution)
2. Comparing waiting times at 3 hospitals
3. Comparing pain scores before and after treatment for the same patients

### Problem 2: Mann-Whitney U
Given data for two groups:
- Group A: [23, 28, 31, 35, 39, 42]
- Group B: [18, 22, 25, 29, 33]

Calculate the U statistic manually and compare with scipy results.

### Problem 3: Correlation Analysis
For 10 pairs of ranked data:
1. Calculate Spearman correlation coefficient
2. Calculate Kendall tau
3. Interpret the difference between the two coefficients

### Problem 4: Kruskal-Wallis Post-hoc
When Kruskal-Wallis is significant in comparing 4 groups:
1. Number of post-hoc comparisons needed
2. Bonferroni-corrected significance level
3. Determine which pairs differ significantly

---

## 9. Key Summary

### Test Selection Flowchart

```
Check data type
    │
    ├── Normality satisfied? ───┬── Yes → Parametric test
    │                           └── No → Nonparametric test
    │
Nonparametric test selection:
    │
    ├── 2 independent groups → Mann-Whitney U
    │
    ├── 2 paired groups → Wilcoxon signed-rank
    │
    ├── 3+ independent groups → Kruskal-Wallis H → Post-hoc: Dunn
    │
    ├── 3+ paired groups → Friedman → Post-hoc: Nemenyi
    │
    └── Correlation → Spearman (continuous) / Kendall (ordinal)
```

### scipy.stats Functions

| Test | Function |
|------|----------|
| Mann-Whitney U | `mannwhitneyu(x, y)` |
| Wilcoxon | `wilcoxon(x, y)` |
| Kruskal-Wallis | `kruskal(g1, g2, g3, ...)` |
| Friedman | `friedmanchisquare(g1, g2, g3, ...)` |
| Spearman | `spearmanr(x, y)` |
| Kendall | `kendalltau(x, y)` |

### Effect Size Interpretation

| Test | Effect Size | Small | Medium | Large |
|------|------------|-------|--------|-------|
| Mann-Whitney | rank-biserial r | 0.1 | 0.3 | 0.5 |
| Kruskal-Wallis | η² | 0.01 | 0.06 | 0.14 |
| Friedman | Kendall's W | 0.1 | 0.3 | 0.5 |

### Next Chapter Preview

In Chapter 14, **Experimental Design**, we'll cover:
- Basic principles of experimental design
- A/B testing
- Sample size determination (power analysis)
- Sequential testing

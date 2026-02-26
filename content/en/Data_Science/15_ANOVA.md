# 15. Analysis of Variance (ANOVA)

[Previous: Advanced Hypothesis Testing](./14_Hypothesis_Testing_Advanced.md) | [Next: Advanced Regression Analysis](./16_Regression_Analysis_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why ANOVA is preferred over multiple t-tests for comparing three or more group means
2. Decompose total variation into between-group and within-group components and compute the F-statistic
3. Verify ANOVA assumptions (normality, homogeneity of variance) and apply Welch's ANOVA when assumptions are violated
4. Perform post-hoc tests (Tukey HSD, Bonferroni, Dunnett, Games-Howell) to identify which groups differ
5. Implement two-way ANOVA to analyze main effects and interaction effects between two factors
6. Apply repeated measures and mixed-design ANOVA for within-subject and combined designs
7. Identify when to use nonparametric alternatives (Kruskal-Wallis, Friedman) and calculate ANOVA effect sizes

---

> **Signal vs. Noise.** Imagine you're in a noisy room trying to hear three different speakers. ANOVA asks: "Is the difference between what the speakers are saying (between-group variance) louder than the background chatter (within-group variance)?" If the signal-to-noise ratio is high enough, you conclude the speakers are genuinely saying different things -- not just being drowned out by random noise.

When you need to compare more than two groups, running pairwise t-tests quickly inflates your false-positive rate. ANOVA provides an elegant single-test framework that partitions variability into meaningful sources -- the signal you care about versus the noise you cannot control. Mastering ANOVA and its extensions unlocks the ability to design and analyze experiments with multiple factors and repeated measurements.

---

## 1. One-way ANOVA

### 1.1 Basic Concepts

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Basic idea of ANOVA:
# Total variation = Between-group variation + Within-group variation
# H₀: μ₁ = μ₂ = ... = μₖ (all group means are equal)
# H₁: At least one group mean differs

# Generate example data
np.random.seed(42)

# Compare three teaching methods
method_A = np.random.normal(75, 10, 25)  # Traditional teaching
method_B = np.random.normal(80, 10, 25)  # Discussion-based
method_C = np.random.normal(85, 10, 25)  # Flipped learning

# Create DataFrame
df = pd.DataFrame({
    'score': np.concatenate([method_A, method_B, method_C]),
    'method': ['A']*25 + ['B']*25 + ['C']*25
})

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=df, x='method', y='score', ax=axes[0], palette='Set2')
axes[0].set_xlabel('Teaching Method')
axes[0].set_ylabel('Score')
axes[0].set_title('Score Distribution by Teaching Method')

# Individual scores and group means
for i, method in enumerate(['A', 'B', 'C']):
    data = df[df['method'] == method]['score']
    axes[1].scatter([i]*len(data), data, alpha=0.5, s=30)
    axes[1].scatter([i], [data.mean()], color='red', s=100, marker='D', zorder=5)

axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['A', 'B', 'C'])
axes[1].set_xlabel('Teaching Method')
axes[1].set_ylabel('Score')
axes[1].set_title('Individual Scores and Group Means')
axes[1].axhline(df['score'].mean(), color='blue', linestyle='--', label='Grand Mean')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Grand mean: {df['score'].mean():.2f}")
print(f"Group means:")
print(df.groupby('method')['score'].mean())
```

### 1.2 ANOVA Calculation

```python
def one_way_anova_manual(groups):
    """Manual one-way ANOVA calculation"""
    # All data
    all_data = np.concatenate(groups)
    n_total = len(all_data)
    k = len(groups)  # Number of groups
    grand_mean = all_data.mean()

    # Between-group variation (SSB: Sum of Squares Between)
    SSB = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)

    # Within-group variation (SSW: Sum of Squares Within)
    SSW = sum(sum((x - g.mean())**2 for x in g) for g in groups)

    # Total variation (SST: Sum of Squares Total)
    SST = sum((x - grand_mean)**2 for x in all_data)

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1

    # Mean Squares
    MSB = SSB / df_between
    MSW = SSW / df_within

    # F-statistic
    F_stat = MSB / MSW

    # p-value
    p_value = 1 - stats.f.cdf(F_stat, df_between, df_within)

    return {
        'SSB': SSB, 'SSW': SSW, 'SST': SST,
        'df_between': df_between, 'df_within': df_within,
        'MSB': MSB, 'MSW': MSW,
        'F_stat': F_stat, 'p_value': p_value
    }

# Calculate
groups = [method_A, method_B, method_C]
result = one_way_anova_manual(groups)

print("One-way ANOVA Results (manual calculation):")
print("=" * 60)
print("ANOVA Table:")
print("-" * 60)
print(f"{'Source':<12} {'SS':<12} {'df':<8} {'MS':<12} {'F':<10} {'p-value':<10}")
print("-" * 60)
print(f"{'Treatment':<12} {result['SSB']:<12.2f} {result['df_between']:<8} "
      f"{result['MSB']:<12.2f} {result['F_stat']:<10.3f} {result['p_value']:<10.4f}")
print(f"{'Error':<12} {result['SSW']:<12.2f} {result['df_within']:<8} {result['MSW']:<12.2f}")
print(f"{'Total':<12} {result['SST']:<12.2f} {result['df_between'] + result['df_within']:<8}")
print("=" * 60)
```

### 1.3 Using scipy and statsmodels

```python
# scipy.stats.f_oneway
F_scipy, p_scipy = stats.f_oneway(method_A, method_B, method_C)
print(f"scipy.stats.f_oneway:")
print(f"  F = {F_scipy:.3f}, p = {p_scipy:.4f}")

# ANOVA using statsmodels OLS
model = ols('score ~ C(method)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(f"\nstatsmodels ANOVA Table:")
print(anova_table)

# pingouin
anova_pg = pg.anova(data=df, dv='score', between='method', detailed=True)
print(f"\npingouin ANOVA:")
print(anova_pg)
```

### 1.4 Understanding F-distribution

```python
# F-distribution: ratio of two chi-square variables
# F = (χ²₁/df₁) / (χ²₂/df₂)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F-distribution (various degrees of freedom)
x = np.linspace(0, 5, 200)
df_pairs = [(2, 20), (5, 20), (10, 20), (5, 50)]

for df1, df2 in df_pairs:
    axes[0].plot(x, stats.f.pdf(x, df1, df2), label=f'F({df1},{df2})')

axes[0].set_xlabel('F')
axes[0].set_ylabel('Density')
axes[0].set_title('F-distribution (various df)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Current analysis F-distribution and observed F-value
df1, df2 = result['df_between'], result['df_within']
x = np.linspace(0, 8, 200)
y = stats.f.pdf(x, df1, df2)

axes[1].plot(x, y, 'b-', lw=2, label=f'F({df1},{df2})')
axes[1].fill_between(x, y, where=(x >= result['F_stat']), alpha=0.3, color='red',
                      label=f'p-value = {result["p_value"]:.4f}')
axes[1].axvline(result['F_stat'], color='red', linestyle='--',
                label=f'F = {result["F_stat"]:.2f}')

# Critical value
f_critical = stats.f.ppf(0.95, df1, df2)
axes[1].axvline(f_critical, color='green', linestyle=':',
                label=f'F_crit (α=0.05) = {f_critical:.2f}')

axes[1].set_xlabel('F')
axes[1].set_ylabel('Density')
axes[1].set_title('F-distribution and Test Results')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. Checking ANOVA Assumptions

### 2.1 Normality Test

```python
# Normality test for each group

print("Normality Test (Shapiro-Wilk):")
print("-" * 40)

for method in ['A', 'B', 'C']:
    data = df[df['method'] == method]['score']
    stat, p = stats.shapiro(data)
    print(f"Group {method}: W = {stat:.4f}, p = {p:.4f}")

# Residual normality (more important)
residuals = model.resid
stat, p = stats.shapiro(residuals)
print(f"\nResidual normality: W = {stat:.4f}, p = {p:.4f}")

# Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Q-Q plot by group
for i, method in enumerate(['A', 'B', 'C']):
    data = df[df['method'] == method]['score']
    stats.probplot(data, dist="norm", plot=axes[0])

axes[0].set_title('Q-Q Plot by Group')

# Residual Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Residual Q-Q Plot')

plt.tight_layout()
plt.show()
```

### 2.2 Homogeneity of Variance Test

```python
# Levene's test (median-based, more robust)
stat_levene, p_levene = stats.levene(method_A, method_B, method_C)
print(f"Levene's Test: F = {stat_levene:.4f}, p = {p_levene:.4f}")

# Bartlett's test (assumes normality)
stat_bartlett, p_bartlett = stats.bartlett(method_A, method_B, method_C)
print(f"Bartlett's Test: χ² = {stat_bartlett:.4f}, p = {p_bartlett:.4f}")

# pingouin
homoscedasticity_result = pg.homoscedasticity(df, dv='score', group='method')
print(f"\npingouin homogeneity test:")
print(homoscedasticity_result)

# Residual vs fitted values plot
fig, ax = plt.subplots(figsize=(10, 5))

fitted = model.fittedvalues
residuals = model.resid

ax.scatter(fitted, residuals, alpha=0.6)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Fitted values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted (check homoscedasticity)')

# Show variance for each group
for method in ['A', 'B', 'C']:
    data = df[df['method'] == method]
    mean_val = data['score'].mean()
    ax.axvline(mean_val, color='gray', linestyle=':', alpha=0.5)

plt.show()

print("\nStandard deviation by group:")
print(df.groupby('method')['score'].std())
```

### 2.3 When Homogeneity is Violated: Welch's ANOVA

```python
# Case where homogeneity of variance is violated
np.random.seed(42)

# Groups with different variances
group1 = np.random.normal(50, 5, 30)   # σ = 5
group2 = np.random.normal(55, 15, 30)  # σ = 15
group3 = np.random.normal(60, 25, 30)  # σ = 25

df_hetero = pd.DataFrame({
    'value': np.concatenate([group1, group2, group3]),
    'group': ['G1']*30 + ['G2']*30 + ['G3']*30
})

# Homogeneity test
stat, p = stats.levene(group1, group2, group3)
print(f"Levene's Test: p = {p:.4f} (homogeneity violated)")

# Regular ANOVA
F_normal, p_normal = stats.f_oneway(group1, group2, group3)
print(f"\nRegular ANOVA: F = {F_normal:.3f}, p = {p_normal:.4f}")

# Welch's ANOVA (pingouin)
welch_result = pg.welch_anova(data=df_hetero, dv='value', between='group')
print(f"\nWelch's ANOVA:")
print(welch_result)

# Games-Howell post-hoc test (no homogeneity assumption)
games_howell = pg.pairwise_gameshowell(data=df_hetero, dv='value', between='group')
print(f"\nGames-Howell post-hoc:")
print(games_howell)
```

---

## 3. Post-hoc Tests

### 3.1 Tukey HSD

```python
# If ANOVA is significant → identify which groups differ
# Tukey's Honest Significant Difference

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Using original data
tukey_result = pairwise_tukeyhsd(df['score'], df['method'], alpha=0.05)

print("Tukey HSD post-hoc:")
print(tukey_result)

# Visualization
fig = tukey_result.plot_simultaneous(figsize=(10, 4))
plt.title('Tukey HSD: 95% Confidence Intervals')
plt.xlabel('Score')
plt.tight_layout()
plt.show()

# Using pingouin
tukey_pg = pg.pairwise_tukey(data=df, dv='score', between='method')
print("\npingouin Tukey HSD:")
print(tukey_pg)
```

### 3.2 Various Post-hoc Methods

```python
# Comparison of various post-hoc methods

print("Comparison of Various Post-hoc Methods:")
print("=" * 70)

# 1. Bonferroni
bonf = pg.pairwise_tests(data=df, dv='score', between='method',
                          padjust='bonf', effsize='cohen')
print("\n1. Bonferroni:")
print(bonf[['A', 'B', 'T', 'p-unc', 'p-corr', 'cohen']])

# 2. Holm
holm = pg.pairwise_tests(data=df, dv='score', between='method',
                          padjust='holm', effsize='cohen')
print("\n2. Holm:")
print(holm[['A', 'B', 'p-unc', 'p-corr']])

# 3. Scheffe (more conservative)
# Requires direct calculation with statsmodels

# 4. Dunnett (control group comparison)
# Set A as control
from scipy.stats import dunnett

dunnett_result = dunnett(method_B, method_C, control=method_A)
print("\n4. Dunnett (A is control):")
print(f"  B vs A: statistic={dunnett_result.statistic[0]:.3f}, p={dunnett_result.pvalue[0]:.4f}")
print(f"  C vs A: statistic={dunnett_result.statistic[1]:.3f}, p={dunnett_result.pvalue[1]:.4f}")
```

### 3.3 Effect Size

```python
# ANOVA effect size

# Eta-squared (η²)
ss_between = result['SSB']
ss_total = result['SST']
eta_squared = ss_between / ss_total

# Omega-squared (ω²) - bias corrected
ms_within = result['MSW']
n_total = len(df)
k = 3
omega_squared = (ss_between - (k-1)*ms_within) / (ss_total + ms_within)

# Partial eta-squared (same as eta-squared in this case)
partial_eta_squared = ss_between / (ss_between + result['SSW'])

print("ANOVA Effect Size:")
print("-" * 40)
print(f"η² (Eta-squared): {eta_squared:.4f}")
print(f"ω² (Omega-squared): {omega_squared:.4f}")
print(f"Partial η²: {partial_eta_squared:.4f}")

print("\nInterpretation Guidelines (Cohen):")
print("  η² ≈ 0.01: small effect")
print("  η² ≈ 0.06: medium effect")
print("  η² ≈ 0.14: large effect")

# Effect size from pingouin result
print(f"\npingouin η²: {anova_pg['np2'].values[0]:.4f}")
```

---

## 4. Two-way ANOVA

### 4.1 Basic Concepts

```python
# Analyze effects of two independent variables (factors) simultaneously
# Main Effect + Interaction

np.random.seed(42)

# Example: Teaching method (A, B) × Study time (Low, High) effects
n_per_cell = 20

data = {
    'score': [],
    'method': [],
    'time': []
}

# 2x2 design
# Including interaction effect
effects = {
    ('A', 'Low'): 70,
    ('A', 'High'): 80,
    ('B', 'Low'): 75,
    ('B', 'High'): 90,  # Synergy effect of B + High
}

for (method, time), mean in effects.items():
    scores = np.random.normal(mean, 8, n_per_cell)
    data['score'].extend(scores)
    data['method'].extend([method] * n_per_cell)
    data['time'].extend([time] * n_per_cell)

df_two = pd.DataFrame(data)

# Check cell means
cell_means = df_two.groupby(['method', 'time'])['score'].mean().unstack()
print("Cell Means:")
print(cell_means)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=df_two, x='method', y='score', hue='time', ax=axes[0])
axes[0].set_title('Teaching Method × Study Time')
axes[0].legend(title='Study Time')

# Interaction plot
for time in ['Low', 'High']:
    means = df_two[df_two['time'] == time].groupby('method')['score'].mean()
    axes[1].plot(['A', 'B'], means.values, 'o-', label=time, markersize=10)

axes[1].set_xlabel('Teaching Method')
axes[1].set_ylabel('Mean Score')
axes[1].set_title('Interaction Plot')
axes[1].legend(title='Study Time')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.2 Two-way ANOVA Analysis

```python
# Two-way ANOVA using statsmodels
model_two = ols('score ~ C(method) * C(time)', data=df_two).fit()
anova_two = sm.stats.anova_lm(model_two, typ=2)

print("Two-way ANOVA Results:")
print(anova_two)

# pingouin
anova_two_pg = pg.anova(data=df_two, dv='score', between=['method', 'time'])
print("\npingouin Two-way ANOVA:")
print(anova_two_pg)

# Effect interpretation
print("\nEffect Interpretation:")
print("-" * 50)
for idx, row in anova_two.iterrows():
    if 'Residual' not in idx:
        sig = "***" if row['PR(>F)'] < 0.001 else ("**" if row['PR(>F)'] < 0.01 else ("*" if row['PR(>F)'] < 0.05 else ""))
        print(f"{idx}: F = {row['F']:.2f}, p = {row['PR(>F)']:.4f} {sig}")
```

### 4.3 Interpreting Interaction Effects

```python
# When interaction is significant: simple main effects analysis

# Teaching method effect by study time
print("Simple Main Effects Analysis: Teaching Method Effect by Study Time")
print("-" * 50)

for time in ['Low', 'High']:
    subset = df_two[df_two['time'] == time]
    t_stat, p_val = stats.ttest_ind(
        subset[subset['method'] == 'A']['score'],
        subset[subset['method'] == 'B']['score']
    )
    print(f"{time} Study Time: t = {t_stat:.3f}, p = {p_val:.4f}")

# Study time effect by teaching method
print("\nSimple Main Effects Analysis: Study Time Effect by Teaching Method")
print("-" * 50)

for method in ['A', 'B']:
    subset = df_two[df_two['method'] == method]
    t_stat, p_val = stats.ttest_ind(
        subset[subset['time'] == 'Low']['score'],
        subset[subset['time'] == 'High']['score']
    )
    d = pg.compute_effsize(
        subset[subset['time'] == 'High']['score'],
        subset[subset['time'] == 'Low']['score'],
        eftype='cohen'
    )
    print(f"Method {method}: t = {t_stat:.3f}, p = {p_val:.4f}, d = {d:.3f}")
```

### 4.4 No Interaction Case

```python
# Generate data without interaction
np.random.seed(42)

data_no_int = {
    'score': [],
    'method': [],
    'time': []
}

# No interaction: effects are independent
effects_no_int = {
    ('A', 'Low'): 70,
    ('A', 'High'): 80,   # +10 (time effect)
    ('B', 'Low'): 78,    # +8 (method effect)
    ('B', 'High'): 88,   # +8 + 10 (additive)
}

for (method, time), mean in effects_no_int.items():
    scores = np.random.normal(mean, 8, 20)
    data_no_int['score'].extend(scores)
    data_no_int['method'].extend([method] * 20)
    data_no_int['time'].extend([time] * 20)

df_no_int = pd.DataFrame(data_no_int)

# Interaction plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# With interaction
for time in ['Low', 'High']:
    means = df_two[df_two['time'] == time].groupby('method')['score'].mean()
    axes[0].plot(['A', 'B'], means.values, 'o-', label=time, markersize=10)
axes[0].set_xlabel('Teaching Method')
axes[0].set_ylabel('Mean Score')
axes[0].set_title('With Interaction (lines cross/non-parallel)')
axes[0].legend(title='Study Time')
axes[0].grid(alpha=0.3)

# Without interaction
for time in ['Low', 'High']:
    means = df_no_int[df_no_int['time'] == time].groupby('method')['score'].mean()
    axes[1].plot(['A', 'B'], means.values, 'o-', label=time, markersize=10)
axes[1].set_xlabel('Teaching Method')
axes[1].set_ylabel('Mean Score')
axes[1].set_title('No Interaction (lines parallel)')
axes[1].legend(title='Study Time')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ANOVA comparison
print("ANOVA without interaction:")
model_no_int = ols('score ~ C(method) * C(time)', data=df_no_int).fit()
print(sm.stats.anova_lm(model_no_int, typ=2))
```

---

## 5. Repeated Measures ANOVA

### 5.1 Basic Concepts

```python
# Same subjects measured under multiple conditions
# Within-subjects design

np.random.seed(42)

# Example: same subjects measured under 3 drug conditions
n_subjects = 20

# Individual differences (subject effect)
subject_effect = np.random.normal(0, 10, n_subjects)

# Mean effect of each condition
condition_effects = {'Placebo': 0, 'Drug_A': 5, 'Drug_B': 10}

data_rm = []
for subj in range(n_subjects):
    for condition, effect in condition_effects.items():
        score = 50 + subject_effect[subj] + effect + np.random.normal(0, 5)
        data_rm.append({
            'subject': f'S{subj+1:02d}',
            'condition': condition,
            'score': score
        })

df_rm = pd.DataFrame(data_rm)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=df_rm, x='condition', y='score', ax=axes[0],
            order=['Placebo', 'Drug_A', 'Drug_B'])
axes[0].set_title('Score Distribution by Condition')

# Individual changes
for subj in df_rm['subject'].unique()[:10]:  # First 10 subjects only
    subj_data = df_rm[df_rm['subject'] == subj]
    subj_data = subj_data.set_index('condition').loc[['Placebo', 'Drug_A', 'Drug_B']]
    axes[1].plot(range(3), subj_data['score'].values, 'o-', alpha=0.5)

# Add mean
means = df_rm.groupby('condition')['score'].mean()
means = means.loc[['Placebo', 'Drug_A', 'Drug_B']]
axes[1].plot(range(3), means.values, 'rs-', markersize=12, linewidth=3, label='Mean')

axes[1].set_xticks(range(3))
axes[1].set_xticklabels(['Placebo', 'Drug_A', 'Drug_B'])
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Score')
axes[1].set_title('Individual Change Patterns')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 5.2 Repeated Measures ANOVA Analysis

```python
# Repeated measures ANOVA using pingouin
rm_anova = pg.rm_anova(data=df_rm, dv='score', within='condition',
                        subject='subject', detailed=True)

print("Repeated Measures ANOVA Results:")
print(rm_anova)

# Sphericity test (Mauchly's test)
# Assumption for repeated measures ANOVA
print("\nSphericity Assumption:")
print("  If sphericity is violated, use Greenhouse-Geisser or Huynh-Feldt correction")
print(f"  GG epsilon: {rm_anova['eps'].values[0]:.4f}")

# Post-hoc
posthoc_rm = pg.pairwise_tests(data=df_rm, dv='score', within='condition',
                                subject='subject', padjust='bonf')
print("\nPost-hoc (Bonferroni):")
print(posthoc_rm[['A', 'B', 'T', 'p-unc', 'p-corr', 'BF10']])
```

### 5.3 Mixed Design ANOVA

```python
# Mixed design: between-subject factor + within-subject factor
np.random.seed(42)

# 2 (group: experimental/control) × 3 (time: pre/mid/post) mixed design
n_per_group = 15

data_mixed = []
for group, group_effect in [('Experimental', 5), ('Control', 0)]:
    for subj in range(n_per_group):
        subject_id = f'{group[0]}_{subj+1:02d}'
        base = 50 + np.random.normal(0, 8)

        for time_idx, (time, time_effect) in enumerate([('Pre', 0), ('Mid', 3), ('Post', 6)]):
            # Interaction: experimental group has greater effect over time
            interaction = time_idx * 3 if group == 'Experimental' else 0
            score = base + group_effect + time_effect + interaction + np.random.normal(0, 5)

            data_mixed.append({
                'subject': subject_id,
                'group': group,
                'time': time,
                'score': score
            })

df_mixed = pd.DataFrame(data_mixed)

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

for group in ['Experimental', 'Control']:
    means = df_mixed[df_mixed['group'] == group].groupby('time')['score'].mean()
    sems = df_mixed[df_mixed['group'] == group].groupby('time')['score'].sem()
    means = means.loc[['Pre', 'Mid', 'Post']]
    sems = sems.loc[['Pre', 'Mid', 'Post']]

    ax.errorbar(range(3), means.values, yerr=sems.values, fmt='o-',
                label=group, markersize=10, capsize=5)

ax.set_xticks(range(3))
ax.set_xticklabels(['Pre', 'Mid', 'Post'])
ax.set_xlabel('Time')
ax.set_ylabel('Score')
ax.set_title('Mixed Design: Group × Time')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# Mixed ANOVA
mixed_anova = pg.mixed_anova(data=df_mixed, dv='score', between='group',
                              within='time', subject='subject')
print("Mixed Design ANOVA:")
print(mixed_anova)
```

---

## 6. Nonparametric Alternatives

### 6.1 Kruskal-Wallis Test

```python
# When normality assumption is not met

# Generate non-normal data
np.random.seed(42)
group1_nonnorm = np.random.exponential(10, 30)
group2_nonnorm = np.random.exponential(15, 30)
group3_nonnorm = np.random.exponential(20, 30)

# Normality test
print("Normality Test:")
for i, g in enumerate([group1_nonnorm, group2_nonnorm, group3_nonnorm], 1):
    stat, p = stats.shapiro(g)
    print(f"  Group {i}: p = {p:.4f}")

# Kruskal-Wallis test
H_stat, p_kw = stats.kruskal(group1_nonnorm, group2_nonnorm, group3_nonnorm)
print(f"\nKruskal-Wallis Test:")
print(f"  H = {H_stat:.3f}, p = {p_kw:.4f}")

# Comparison: one-way ANOVA
F_stat, p_anova = stats.f_oneway(group1_nonnorm, group2_nonnorm, group3_nonnorm)
print(f"\nOne-way ANOVA (for comparison):")
print(f"  F = {F_stat:.3f}, p = {p_anova:.4f}")

# Post-hoc: Dunn's test
df_nonnorm = pd.DataFrame({
    'value': np.concatenate([group1_nonnorm, group2_nonnorm, group3_nonnorm]),
    'group': ['G1']*30 + ['G2']*30 + ['G3']*30
})

dunn_result = pg.pairwise_tests(data=df_nonnorm, dv='value', between='group',
                                 parametric=False, padjust='bonf')
print("\nDunn's Post-hoc:")
print(dunn_result[['A', 'B', 'U-val', 'p-unc', 'p-corr']])
```

### 6.2 Friedman Test

```python
# Nonparametric alternative for repeated measures

# Non-normal repeated measures data
np.random.seed(42)
n_subjects = 20

cond1 = np.random.exponential(10, n_subjects)
cond2 = np.random.exponential(15, n_subjects) + cond1 * 0.3  # Correlated measures
cond3 = np.random.exponential(20, n_subjects) + cond1 * 0.3

# Friedman test
chi2_stat, p_friedman = stats.friedmanchisquare(cond1, cond2, cond3)
print(f"Friedman Test:")
print(f"  χ² = {chi2_stat:.3f}, p = {p_friedman:.4f}")

# Effect size: Kendall's W
n_conditions = 3
W = chi2_stat / (n_subjects * (n_conditions - 1))
print(f"  Kendall's W = {W:.4f}")
```

---

## Practice Problems

### Problem 1: One-way ANOVA
Analyze the effect of three fertilizers (A, B, C) on plant growth.
```python
fertilizer_A = [20, 22, 19, 24, 25, 23, 21, 22, 26, 24]
fertilizer_B = [28, 30, 27, 29, 31, 28, 30, 29, 32, 31]
fertilizer_C = [25, 27, 26, 28, 24, 26, 27, 25, 29, 26]
```

### Problem 2: Two-way ANOVA
Analyze the effects of Gender (Male/Female) × Learning Method (Online/Offline).
- Interpret main effects and interaction.
- Plot interaction diagram.

### Problem 3: Post-hoc Test
If Problem 1 results are significant, perform Tukey HSD post-hoc test and interpret.

---

## Summary

| ANOVA Type | Design | Python Function |
|------------|--------|-----------------|
| One-way | 1 factor, k levels | `stats.f_oneway()`, `pg.anova()` |
| Two-way | 2 factors | `ols() + anova_lm()` |
| Repeated Measures | Within-subject factor | `pg.rm_anova()` |
| Mixed Design | Between+Within | `pg.mixed_anova()` |
| Kruskal-Wallis | Nonparametric one-way | `stats.kruskal()` |
| Friedman | Nonparametric repeated | `stats.friedmanchisquare()` |

| Post-hoc | Feature | Function |
|----------|---------|----------|
| Tukey HSD | All pairwise comparisons | `pairwise_tukeyhsd()` |
| Bonferroni | Conservative | `pg.pairwise_tests(padjust='bonf')` |
| Dunnett | Control group comparison | `stats.dunnett()` |
| Games-Howell | No homogeneity | `pg.pairwise_gameshowell()` |

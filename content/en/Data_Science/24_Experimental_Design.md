# 24. Experimental Design

[Previous: Nonparametric Statistics](./23_Nonparametric_Statistics.md) | [Next: Practical Projects](./25_Practical_Projects.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the three core principles of experimental design: randomization, replication, and blocking
2. Implement simple and stratified randomization for participant assignment
3. Describe the A/B testing workflow from hypothesis formulation through decision making
4. Calculate required sample sizes using power analysis for two-proportion comparisons
5. Compare frequentist and Bayesian approaches to A/B test analysis
6. Identify the peeking problem and apply alpha spending functions (O'Brien-Fleming, Pocock) for sequential testing
7. Apply multiple comparison corrections (Bonferroni, Holm-Bonferroni) and explain why they are necessary
8. Evaluate common A/B testing pitfalls including novelty effects, network effects, and Simpson's paradox

---

Observational data can reveal correlations, but only well-designed experiments can establish causation. Whether you are running an A/B test on a website, evaluating a medical treatment, or comparing teaching methods, the principles of experimental design determine whether your conclusions are trustworthy. Getting the design right before collecting data -- randomization, sufficient sample size, proper controls -- saves you from drawing false conclusions that no amount of post-hoc analysis can fix.

---

## 1. Basic Principles of Experimental Design

### 1.1 Three Core Principles

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm, t

np.random.seed(42)

def experimental_design_principles():
    """Three core principles of experimental design"""
    print("""
    =================================================
    Three Core Principles of Experimental Design
    =================================================

    1. Randomization
    ─────────────────────────────
    - Randomly assign subjects to treatment groups
    - Equally distribute effects of confounding variables
    - Foundation for causal inference

    Examples:
    - Coin flip for A/B group assignment
    - Computer-generated random numbers
    - Blocked randomization (stratify then randomize)

    2. Replication
    ─────────────────────────────
    - Sufficient number of independent observations
    - Ensure statistical power
    - Enable variability estimation

    Considerations:
    - Sample size calculation (power analysis)
    - Cost-benefit trade-off
    - Practical constraints

    3. Blocking
    ─────────────────────────────
    - Group subjects by known sources of variation
    - Random assignment within blocks
    - Reduce error, increase power

    Examples:
    - Block by gender → random assignment within each block
    - Stratify by age group
    - Region, time period, etc.

    =================================================
    Additional Principles
    =================================================

    - Control: Include control group
    - Blinding: Single/double blind
    - Balance: Equal allocation across groups
    """)

experimental_design_principles()
```

### 1.2 Implementing Randomization

```python
def randomize_participants(participants, n_groups=2, method='simple', block_var=None):
    """
    Randomize participants

    Parameters:
    -----------
    participants : DataFrame
        Participant information
    n_groups : int
        Number of groups
    method : str
        'simple' - Simple randomization
        'stratified' - Stratified randomization
    block_var : str
        Stratification variable (when method='stratified')
    """
    n = len(participants)
    result = participants.copy()

    if method == 'simple':
        # Simple random assignment
        assignments = np.random.choice(range(n_groups), size=n)
        result['group'] = assignments

    elif method == 'stratified' and block_var is not None:
        # Stratified random assignment
        result['group'] = -1
        for block_value in participants[block_var].unique():
            mask = participants[block_var] == block_value
            block_n = mask.sum()
            assignments = np.random.choice(range(n_groups), size=block_n)
            result.loc[mask, 'group'] = assignments

    return result

# Example: 100 participants
np.random.seed(42)
participants = pd.DataFrame({
    'id': range(100),
    'age': np.random.choice(['young', 'middle', 'old'], 100),
    'gender': np.random.choice(['M', 'F'], 100)
})

# Simple randomization
simple_rand = randomize_participants(participants, n_groups=2, method='simple')

# Stratified randomization (by gender)
stratified_rand = randomize_participants(participants, n_groups=2,
                                          method='stratified', block_var='gender')

print("=== Simple Randomization Results ===")
print(pd.crosstab(simple_rand['gender'], simple_rand['group']))

print("\n=== Stratified Randomization Results (by gender) ===")
print(pd.crosstab(stratified_rand['gender'], stratified_rand['group']))
```

### 1.3 Types of Experimental Designs

```python
def experimental_design_types():
    """Major types of experimental designs"""
    print("""
    =================================================
    Experimental Design Types
    =================================================

    1. Completely Randomized Design
       - Simplest design
       - Complete random assignment of subjects to treatments
       - Analysis: Independent t-test, one-way ANOVA

    2. Randomized Block Design
       - Stratify by blocking variable then randomize
       - All treatment levels included within each block
       - Analysis: Two-way ANOVA (removing block effect)

    3. Factorial Design
       - Study combined effects of multiple factors
       - Can detect interaction effects
       - Analysis: Multi-way ANOVA

    4. Crossover Design
       - Subjects receive all treatments sequentially
       - Controls for between-subject variation
       - Watch for carryover effects

    5. Split-Plot Design
       - One factor applied to whole units, another to sub-units
       - Common in agriculture and engineering
    """)

experimental_design_types()
```

---

## 2. A/B Testing Theory

### 2.1 A/B Testing Overview

```python
def ab_test_overview():
    """A/B testing overview"""
    print("""
    =================================================
    A/B Testing
    =================================================

    Definition:
    - Randomized controlled experiment comparing two versions (A, B)
    - Most widely used experimental method in web/app

    Terminology:
    - Control (A): Existing version (control group)
    - Treatment (B): New version (treatment group)
    - Conversion Rate: Proportion of goal actions
    - Lift: (B - A) / A

    Process:
    1. Formulate hypothesis
    2. Define metrics
    3. Calculate sample size
    4. Run experiment
    5. Statistical analysis
    6. Decision making

    Cautions:
    - Unit consistency (users vs sessions vs pageviews)
    - Experiment duration (minimum 1-2 weeks, consider day-of-week effects)
    - Multiple comparison correction
    - Network effects (spillover)
    """)

ab_test_overview()
```

### 2.2 A/B Test Analysis

```python
class ABTest:
    """A/B test analysis class"""

    def __init__(self, control_visitors, control_conversions,
                 treatment_visitors, treatment_conversions):
        self.n_c = control_visitors
        self.x_c = control_conversions
        self.n_t = treatment_visitors
        self.x_t = treatment_conversions

        self.p_c = self.x_c / self.n_c
        self.p_t = self.x_t / self.n_t

    def z_test(self, alternative='two-sided'):
        """Z-test for two proportions"""
        # Pooled proportion
        p_pooled = (self.x_c + self.x_t) / (self.n_c + self.n_t)

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/self.n_c + 1/self.n_t))

        # Z statistic
        z = (self.p_t - self.p_c) / se

        # p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - norm.cdf(abs(z)))
        elif alternative == 'greater':  # treatment > control
            p_value = 1 - norm.cdf(z)
        else:  # treatment < control
            p_value = norm.cdf(z)

        return z, p_value

    def confidence_interval(self, alpha=0.05):
        """Confidence interval for the difference"""
        diff = self.p_t - self.p_c

        # Variance for each proportion
        var_c = self.p_c * (1 - self.p_c) / self.n_c
        var_t = self.p_t * (1 - self.p_t) / self.n_t
        se = np.sqrt(var_c + var_t)

        z_crit = norm.ppf(1 - alpha/2)
        ci_lower = diff - z_crit * se
        ci_upper = diff + z_crit * se

        return diff, (ci_lower, ci_upper)

    def lift(self):
        """Calculate lift"""
        if self.p_c == 0:
            return np.inf
        return (self.p_t - self.p_c) / self.p_c

    def summary(self):
        """Summary of results"""
        print("=== A/B Test Summary ===")
        print(f"\nControl:   {self.x_c:,}/{self.n_c:,} = {self.p_c:.4f} ({self.p_c*100:.2f}%)")
        print(f"Treatment: {self.x_t:,}/{self.n_t:,} = {self.p_t:.4f} ({self.p_t*100:.2f}%)")

        z, p_value = self.z_test()
        diff, ci = self.confidence_interval()
        lift = self.lift()

        print(f"\nDifference: {diff:.4f} ({diff*100:.2f}%p)")
        print(f"Lift: {lift*100:.2f}%")
        print(f"95% CI: ({ci[0]*100:.2f}%p, {ci[1]*100:.2f}%p)")
        print(f"\nZ statistic: {z:.3f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("\nConclusion: Statistically significant difference (p < 0.05)")
            if diff > 0:
                print("Treatment is significantly higher than Control")
            else:
                print("Treatment is significantly lower than Control")
        else:
            print("\nConclusion: No statistically significant difference (p >= 0.05)")


# Example: Button color A/B test
ab_test = ABTest(
    control_visitors=10000,
    control_conversions=350,
    treatment_visitors=10000,
    treatment_conversions=420
)
ab_test.summary()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Conversion rate comparison
ax = axes[0]
bars = ax.bar(['Control', 'Treatment'], [ab_test.p_c, ab_test.p_t], alpha=0.7)
ax.set_ylabel('Conversion Rate')
ax.set_title('A/B Test: Conversion Rate Comparison')

# Add error bars
se_c = np.sqrt(ab_test.p_c * (1 - ab_test.p_c) / ab_test.n_c)
se_t = np.sqrt(ab_test.p_t * (1 - ab_test.p_t) / ab_test.n_t)
ax.errorbar(['Control', 'Treatment'], [ab_test.p_c, ab_test.p_t],
            yerr=[1.96*se_c, 1.96*se_t], fmt='none', color='black', capsize=5)
ax.grid(True, alpha=0.3, axis='y')

# Confidence interval of difference
ax = axes[1]
diff, ci = ab_test.confidence_interval()
ax.errorbar([0], [diff], yerr=[[diff - ci[0]], [ci[1] - diff]],
            fmt='o', markersize=10, capsize=10, capthick=2)
ax.axhline(0, color='r', linestyle='--', label='No difference')
ax.set_xlim(-1, 1)
ax.set_ylabel('Conversion Rate Difference')
ax.set_title(f'95% Confidence Interval of Difference\n({ci[0]:.4f}, {ci[1]:.4f})')
ax.set_xticks([])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.3 Bayesian A/B Testing

```python
def bayesian_ab_test(n_c, x_c, n_t, x_t, alpha_prior=1, beta_prior=1, n_samples=100000):
    """
    Bayesian A/B test

    Conversion rate estimation using Beta prior
    """
    # Posterior distribution (Beta-Binomial conjugate)
    alpha_c = alpha_prior + x_c
    beta_c = beta_prior + n_c - x_c
    alpha_t = alpha_prior + x_t
    beta_t = beta_prior + n_t - x_t

    # Sample from posterior
    samples_c = np.random.beta(alpha_c, beta_c, n_samples)
    samples_t = np.random.beta(alpha_t, beta_t, n_samples)

    # P(Treatment > Control)
    prob_t_better = np.mean(samples_t > samples_c)

    # Expected lift
    lift_samples = (samples_t - samples_c) / samples_c
    expected_lift = np.mean(lift_samples)
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])

    print("=== Bayesian A/B Test ===")
    print(f"\nP(Treatment > Control): {prob_t_better:.4f} ({prob_t_better*100:.1f}%)")
    print(f"Expected lift: {expected_lift*100:.2f}%")
    print(f"Lift 95% CI: ({lift_ci[0]*100:.2f}%, {lift_ci[1]*100:.2f}%)")

    # Decision criteria
    print("\nDecision:")
    if prob_t_better > 0.95:
        print("  → Recommend adopting Treatment (P > 95%)")
    elif prob_t_better < 0.05:
        print("  → Recommend keeping Control (P < 5%)")
    else:
        print("  → Need more data")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Posterior distribution comparison
    ax = axes[0]
    x_range = np.linspace(0, 0.1, 200)
    ax.plot(x_range, stats.beta(alpha_c, beta_c).pdf(x_range), label='Control')
    ax.plot(x_range, stats.beta(alpha_t, beta_t).pdf(x_range), label='Treatment')
    ax.fill_between(x_range, stats.beta(alpha_c, beta_c).pdf(x_range), alpha=0.3)
    ax.fill_between(x_range, stats.beta(alpha_t, beta_t).pdf(x_range), alpha=0.3)
    ax.set_xlabel('Conversion Rate')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distribution of Conversion Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference distribution
    ax = axes[1]
    diff_samples = samples_t - samples_c
    ax.hist(diff_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='No difference')
    ax.axvline(np.mean(diff_samples), color='g', linestyle='-',
               label=f'Mean: {np.mean(diff_samples):.4f}')
    ax.set_xlabel('Conversion Rate Difference (T - C)')
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior Distribution of Difference\nP(T>C)={prob_t_better:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lift distribution
    ax = axes[2]
    lift_samples_clipped = np.clip(lift_samples, -1, 2)
    ax.hist(lift_samples_clipped, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='0%')
    ax.axvline(expected_lift, color='g', linestyle='-',
               label=f'Expected: {expected_lift*100:.1f}%')
    ax.set_xlabel('Lift')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distribution of Lift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return prob_t_better, expected_lift

# Bayesian analysis
prob_better, exp_lift = bayesian_ab_test(10000, 350, 10000, 420)
```

---

## 3. Sample Size Determination (Power Analysis)

### 3.1 Power Analysis Concepts

```python
def power_analysis_concepts():
    """Core concepts of power analysis"""
    print("""
    =================================================
    Power Analysis
    =================================================

    Four elements (calculate one from the other three):
    ─────────────────────────────────────────────────
    1. Effect Size
       - Minimum effect to detect
       - Example: conversion rate difference 0.02 (2%p)

    2. Significance Level α
       - Type I error probability
       - Typically 0.05

    3. Power 1-β
       - Probability of detecting effect when it exists
       - Typically 0.80 (minimum) ~ 0.90

    4. Sample Size n
       - Required number of observations

    Calculation flow:
    ─────────────────────────────────────────────────
    Effect size + α + (1-β) → n (prior design)
    n + α + (1-β) → Minimum detectable effect (sensitivity)
    n + α + Effect size → Achieved power (post-hoc)

    Rules of thumb:
    ─────────────────────────────────────────────────
    - Power < 80%: Underpowered
    - Power 80-90%: Generally recommended
    - Power > 90%: High power study
    """)

power_analysis_concepts()
```

### 3.2 Sample Size for Two Proportion Comparison

```python
def sample_size_two_proportions(p1, p2, alpha=0.05, power=0.80, ratio=1):
    """
    Calculate sample size for comparing two proportions

    Parameters:
    -----------
    p1 : float
        Control conversion rate (baseline)
    p2 : float
        Treatment conversion rate (target)
    alpha : float
        Significance level
    power : float
        Statistical power
    ratio : float
        n2/n1 ratio (default 1 = equal size)

    Returns:
    --------
    n1, n2 : int
        Required sample size for each group
    """
    # Effect size
    effect = abs(p2 - p1)
    p_pooled = (p1 + ratio * p2) / (1 + ratio)

    # Z values
    z_alpha = norm.ppf(1 - alpha/2)  # Two-sided
    z_beta = norm.ppf(power)

    # Sample size formula
    numerator = (z_alpha * np.sqrt((1 + ratio) * p_pooled * (1 - p_pooled)) +
                 z_beta * np.sqrt(p1 * (1 - p1) + ratio * p2 * (1 - p2)))**2
    n1 = numerator / (effect**2 * ratio)
    n2 = n1 * ratio

    return int(np.ceil(n1)), int(np.ceil(n2))


def plot_sample_size_analysis(p1_base, effects, alpha=0.05, power=0.80):
    """Required sample size by effect size"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Effect size vs sample size
    ax = axes[0]
    sample_sizes = []
    for effect in effects:
        p2 = p1_base + effect
        n1, _ = sample_size_two_proportions(p1_base, p2, alpha, power)
        sample_sizes.append(n1)

    ax.plot(np.array(effects)*100, sample_sizes, 'bo-', linewidth=2)
    ax.set_xlabel('Effect Size (Conversion Rate Difference %p)')
    ax.set_ylabel('Required Sample Size per Group')
    ax.set_title(f'Effect Size vs Sample Size\n(Baseline rate={p1_base:.1%}, α={alpha}, power={power})')
    ax.grid(True, alpha=0.3)

    # Log scale
    ax.set_yscale('log')
    for i, (eff, n) in enumerate(zip(effects, sample_sizes)):
        ax.annotate(f'{n:,}', (eff*100, n), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # Power vs sample size
    ax = axes[1]
    effect_fixed = 0.02  # Fixed at 2%p
    p2_fixed = p1_base + effect_fixed
    powers = np.linspace(0.5, 0.95, 10)
    sample_sizes_power = []

    for pwr in powers:
        n1, _ = sample_size_two_proportions(p1_base, p2_fixed, alpha, pwr)
        sample_sizes_power.append(n1)

    ax.plot(powers*100, sample_sizes_power, 'go-', linewidth=2)
    ax.set_xlabel('Power (%)')
    ax.set_ylabel('Required Sample Size per Group')
    ax.set_title(f'Power vs Sample Size\n(Effect size={effect_fixed:.1%}p)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Example
p1 = 0.05  # Baseline conversion rate 5%
effects = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]  # 0.5%p ~ 3%p

print("=== Sample Size Calculation ===")
print(f"Baseline conversion rate: {p1:.1%}")
print(f"α = 0.05, Power = 0.80")
print()
for effect in effects:
    p2 = p1 + effect
    n1, n2 = sample_size_two_proportions(p1, p2)
    print(f"Effect {effect*100:.1f}%p (relative {effect/p1*100:.0f}%): n1={n1:,}, n2={n2:,}, total={n1+n2:,}")

plot_sample_size_analysis(p1, effects)
```

### 3.3 Power Analysis with statsmodels

```python
from statsmodels.stats.power import TTestPower, NormalIndPower, tt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

def statsmodels_power_analysis():
    """Power analysis using statsmodels"""

    # 1. t-test power analysis
    print("=== t-test Power Analysis ===")

    # Calculate effect size (Cohen's d)
    # d = (μ1 - μ2) / σ
    mean_diff = 5
    std = 15
    d = mean_diff / std
    print(f"Cohen's d = {d:.3f}")

    # Required sample size
    power_analysis = TTestPower()
    n = power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.80,
                                     alternative='two-sided')
    print(f"Required sample size (per group): {int(np.ceil(n))}")

    # Achieved power
    achieved_power = power_analysis.power(effect_size=d, nobs=100, alpha=0.05,
                                           alternative='two-sided')
    print(f"Power at n=100: {achieved_power:.3f}")

    # 2. Proportion test power analysis
    print("\n=== Proportion Test Power Analysis ===")

    p1 = 0.05
    p2 = 0.07
    effect = proportion_effectsize(p1, p2)
    print(f"Effect size (h): {effect:.3f}")

    # Required sample size
    power_prop = NormalIndPower()
    n_prop = power_prop.solve_power(effect_size=effect, alpha=0.05, power=0.80,
                                      alternative='two-sided', ratio=1)
    print(f"Required sample size (per group): {int(np.ceil(n_prop))}")

    return n, n_prop

n_t, n_prop = statsmodels_power_analysis()
```

### 3.4 Power Curves

```python
def plot_power_curve(effect_sizes, n_per_group, alpha=0.05):
    """Visualize power curves"""

    power_analysis = NormalIndPower()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Effect size vs power (n fixed)
    ax = axes[0]
    for n in n_per_group:
        powers = [power_analysis.power(effect_size=es, nobs=n, alpha=alpha,
                                        alternative='two-sided', ratio=1)
                  for es in effect_sizes]
        ax.plot(effect_sizes, powers, '-o', label=f'n={n}')

    ax.axhline(0.80, color='r', linestyle='--', alpha=0.5, label='Power=0.80')
    ax.set_xlabel('Effect Size (Cohen\'s h)')
    ax.set_ylabel('Power')
    ax.set_title('Power Curves (by Sample Size)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Sample size vs power (effect size fixed)
    ax = axes[1]
    n_range = np.arange(50, 1001, 50)
    effect_fixed = [0.1, 0.2, 0.3, 0.5]

    for es in effect_fixed:
        powers = [power_analysis.power(effect_size=es, nobs=n, alpha=alpha,
                                        alternative='two-sided', ratio=1)
                  for n in n_range]
        ax.plot(n_range, powers, '-', label=f'h={es}')

    ax.axhline(0.80, color='r', linestyle='--', alpha=0.5, label='Power=0.80')
    ax.set_xlabel('Sample Size per Group')
    ax.set_ylabel('Power')
    ax.set_title('Power Curves (by Effect Size)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

effect_sizes = np.linspace(0.05, 0.5, 20)
n_per_group = [50, 100, 200, 500]
plot_power_curve(effect_sizes, n_per_group)
```

---

## 4. Sequential Testing

### 4.1 Why Sequential Testing?

```python
def sequential_testing_motivation():
    """Need for sequential testing"""
    print("""
    =================================================
    Sequential Testing
    =================================================

    Problem: Peeking Problem
    ─────────────────────────────────────────────────
    - Checking results mid-experiment increases Type I error rate
    - Designed at α=0.05, but 5 interim looks → actual rate ~14%

    Examples:
    - 1 look: α = 0.05
    - 5 looks: α ≈ 0.14
    - 10 looks: α ≈ 0.19

    Solutions:
    ─────────────────────────────────────────────────
    1. Fixed sample test: Wait until predetermined n
    2. Sequential test: Allow interim looks with correction
       - O'Brien-Fleming
       - Pocock
       - Alpha spending functions

    Advantages:
    - Early stopping if effect is clear → Save cost/time
    - Quick stop if no effect
    - Maintain statistical validity
    """)

sequential_testing_motivation()
```

### 4.2 Peeking Problem Simulation

```python
def simulate_peeking_problem(n_simulations=10000, n_total=1000, n_looks=5):
    """
    Peeking problem simulation:
    How often is significance found when null is true (no actual difference)
    """
    np.random.seed(42)
    alpha = 0.05

    # Interim look points
    look_points = np.linspace(n_total // n_looks, n_total, n_looks).astype(int)

    false_positives_fixed = 0  # Fixed sample (last look only)
    false_positives_peeking = 0  # All looks

    for _ in range(n_simulations):
        # Generate data under null hypothesis (two groups identical)
        control = np.random.binomial(1, 0.1, n_total)
        treatment = np.random.binomial(1, 0.1, n_total)

        # Peeking: Test at each look
        for look in look_points:
            x_c = control[:look].sum()
            x_t = treatment[:look].sum()
            n = look

            # Proportions
            p_c = x_c / n
            p_t = x_t / n
            p_pooled = (x_c + x_t) / (2 * n)

            if p_pooled > 0 and p_pooled < 1:
                se = np.sqrt(p_pooled * (1 - p_pooled) * 2 / n)
                z = (p_t - p_c) / se if se > 0 else 0
                p_value = 2 * (1 - norm.cdf(abs(z)))

                if p_value < alpha:
                    false_positives_peeking += 1
                    break  # Stop once significant

        # Fixed sample: Last look only
        x_c = control.sum()
        x_t = treatment.sum()
        p_c = x_c / n_total
        p_t = x_t / n_total
        p_pooled = (x_c + x_t) / (2 * n_total)

        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * 2 / n_total)
            z = (p_t - p_c) / se if se > 0 else 0
            p_value = 2 * (1 - norm.cdf(abs(z)))

            if p_value < alpha:
                false_positives_fixed += 1

    fpr_fixed = false_positives_fixed / n_simulations
    fpr_peeking = false_positives_peeking / n_simulations

    print("=== Peeking Problem Simulation ===")
    print(f"Number of simulations: {n_simulations:,}")
    print(f"Total sample size: {n_total}")
    print(f"Number of interim looks: {n_looks}")
    print(f"Target α: {alpha}")
    print(f"\nFixed sample test false positive rate: {fpr_fixed:.4f} ({fpr_fixed*100:.2f}%)")
    print(f"Peeking test false positive rate: {fpr_peeking:.4f} ({fpr_peeking*100:.2f}%)")
    print(f"False positive rate inflation: {(fpr_peeking/alpha - 1)*100:.1f}%")

    return fpr_fixed, fpr_peeking

fpr_fixed, fpr_peeking = simulate_peeking_problem()
```

### 4.3 Alpha Spending Functions

```python
def alpha_spending_pocock(t, alpha=0.05):
    """Pocock alpha spending function"""
    return alpha * np.log(1 + (np.e - 1) * t)

def alpha_spending_obrien_fleming(t, alpha=0.05):
    """O'Brien-Fleming alpha spending function"""
    return 2 * (1 - norm.cdf(norm.ppf(1 - alpha/2) / np.sqrt(t)))

def plot_alpha_spending():
    """Visualize alpha spending functions"""
    t = np.linspace(0.01, 1, 100)
    alpha = 0.05

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, alpha_spending_pocock(t, alpha), label='Pocock', linewidth=2)
    ax.plot(t, alpha_spending_obrien_fleming(t, alpha), label="O'Brien-Fleming", linewidth=2)
    ax.plot(t, t * alpha, '--', label='Linear (reference)', alpha=0.5)
    ax.axhline(alpha, color='r', linestyle=':', label=f'Total α={alpha}')

    ax.set_xlabel('Information Fraction (Current/Final)')
    ax.set_ylabel('Cumulative α Spent')
    ax.set_title('Alpha Spending Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, alpha * 1.1)

    plt.show()

    print("=== Alpha Spending Function Comparison ===")
    print("\nPocock:")
    print("  - Same threshold at each analysis")
    print("  - More lenient for early stopping")
    print("  - More conservative at final analysis")

    print("\nO'Brien-Fleming:")
    print("  - Very conservative early (high threshold)")
    print("  - Similar to fixed sample later")
    print("  - Early stopping only for extreme effects")

plot_alpha_spending()
```

### 4.4 Sequential Test Implementation

```python
class SequentialTest:
    """Sequential A/B test"""

    def __init__(self, max_n, n_looks, alpha=0.05, spending='obrien_fleming'):
        """
        Parameters:
        -----------
        max_n : int
            Maximum sample size (per group)
        n_looks : int
            Number of interim analyses
        alpha : float
            Overall significance level
        spending : str
            'pocock' or 'obrien_fleming'
        """
        self.max_n = max_n
        self.n_looks = n_looks
        self.alpha = alpha
        self.spending = spending

        # Analysis times
        self.look_times = np.linspace(1/n_looks, 1, n_looks)

        # Alpha to use at each analysis
        self.alphas = self._compute_alphas()

    def _compute_alphas(self):
        """Calculate alpha for each analysis"""
        if self.spending == 'pocock':
            cumulative = [alpha_spending_pocock(t, self.alpha) for t in self.look_times]
        else:
            cumulative = [alpha_spending_obrien_fleming(t, self.alpha) for t in self.look_times]

        # Incremental alpha
        alphas = [cumulative[0]]
        for i in range(1, len(cumulative)):
            alphas.append(cumulative[i] - cumulative[i-1])

        return alphas

    def critical_values(self):
        """Critical Z values for each analysis"""
        return [norm.ppf(1 - a/2) for a in self.alphas]

    def summary(self):
        """Analysis plan summary"""
        print("=== Sequential Test Plan ===")
        print(f"Maximum sample: {self.max_n} (per group)")
        print(f"Interim analyses: {self.n_looks} times")
        print(f"Overall α: {self.alpha}")
        print(f"Spending: {self.spending}")

        print("\nPlan by analysis:")
        print("-" * 50)
        print(f"{'Look':<6} {'n':<10} {'Cumulative α':<12} {'Increment α':<12} {'Z Critical':<10}")
        print("-" * 50)

        cumulative_alpha = 0
        z_crits = self.critical_values()

        for i, (t, a) in enumerate(zip(self.look_times, self.alphas)):
            n = int(t * self.max_n)
            cumulative_alpha += a
            print(f"{i+1:<6} {n:<10} {cumulative_alpha:<12.4f} {a:<12.4f} {z_crits[i]:<10.3f}")


# Example
seq_test = SequentialTest(max_n=5000, n_looks=5, alpha=0.05, spending='obrien_fleming')
seq_test.summary()

print("\n")

seq_test_pocock = SequentialTest(max_n=5000, n_looks=5, alpha=0.05, spending='pocock')
seq_test_pocock.summary()
```

---

## 5. Common Pitfalls and Cautions

### 5.1 Multiple Comparisons Problem

```python
def multiple_comparisons_problem():
    """Multiple comparisons problem"""
    print("""
    =================================================
    Multiple Comparisons Problem
    =================================================

    Problem:
    - Conducting multiple tests simultaneously increases Type I error
    - Probability of at least one false positive with k tests: 1 - (1-α)^k

    Examples (α=0.05):
    - 1 test: 5%
    - 5 tests: 23%
    - 10 tests: 40%
    - 20 tests: 64%

    Correction methods:
    ─────────────────────────────────────────────────
    1. Bonferroni: α' = α/k (most conservative)
    2. Holm-Bonferroni: Sequential Bonferroni
    3. Benjamini-Hochberg (FDR): Control false discovery rate
    4. Pre-registration: Specify one primary hypothesis
    """)

    # Visualization
    k_values = range(1, 21)
    alpha = 0.05

    fwer = [1 - (1 - alpha)**k for k in k_values]
    bonferroni = [min(alpha * k, 1.0) for k in k_values]  # Pre-correction tolerance

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, fwer, 'b-o', label='Without correction (FWER)')
    ax.axhline(alpha, color='r', linestyle='--', label=f'Target α={alpha}')
    ax.set_xlabel('Number of Tests')
    ax.set_ylabel('Probability of At Least One False Positive')
    ax.set_title('Multiple Comparisons Problem: Tests vs False Positive Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.7)

    plt.show()

multiple_comparisons_problem()
```

### 5.2 Other Cautions

```python
def common_pitfalls():
    """Common A/B testing pitfalls"""
    print("""
    =================================================
    A/B Testing Cautions
    =================================================

    1. Peeking (Interim Checking)
       - Problem: Check until desired result appears
       - Solution: Sequential testing or fixed sample

    2. Multiple Comparisons
       - Problem: Testing multiple metrics/segments
       - Solution: Pre-registration, correction, specify primary metric

    3. Inadequate Sample Size
       - Problem: Too small → fail to detect effect, too large → wasteful
       - Solution: Prior power analysis

    4. Novelty Effect
       - Problem: Newness itself causes temporary effect
       - Solution: Sufficient experiment duration

    5. Network Effects (Spillover)
       - Problem: Interaction between groups
       - Solution: Cluster randomization

    6. Simpson's Paradox
       - Problem: Overall vs segment-wise results conflict
       - Solution: Stratified analysis, causal graphs

    7. Ignoring Practical Significance
       - Problem: Statistical significance ≠ Practical importance
       - Solution: Consider effect size, CI, business impact

    8. Insufficient Power
       - Problem: "No effect" ≠ "Null hypothesis is true"
       - Solution: Report power, equivalence testing
    """)

common_pitfalls()
```

---

## 6. Practice Examples

### 6.1 Complete Experimental Design

```python
def complete_ab_test_workflow():
    """Complete A/B test workflow"""

    print("="*60)
    print("A/B Test Workflow")
    print("="*60)

    # 1. Formulate hypothesis
    print("\n[Step 1] Formulate Hypothesis")
    print("  H0: New button color has no effect on conversion rate")
    print("  H1: New button color changes conversion rate")

    # 2. Define metrics
    print("\n[Step 2] Define Metrics")
    baseline_rate = 0.05  # 5%
    mde = 0.01  # Minimum detectable effect: 1%p
    print(f"  Baseline conversion rate: {baseline_rate:.1%}")
    print(f"  MDE (Minimum Detectable Effect): {mde:.1%}p")

    # 3. Calculate sample size
    print("\n[Step 3] Calculate Sample Size")
    target_rate = baseline_rate + mde
    n1, n2 = sample_size_two_proportions(baseline_rate, target_rate, alpha=0.05, power=0.80)
    print(f"  Required sample size: {n1:,} (per group)")
    print(f"  Total required traffic: {n1 + n2:,}")

    # 4. Run experiment (simulation)
    print("\n[Step 4] Run Experiment (Simulation)")
    np.random.seed(42)
    n_control = n1
    n_treatment = n2
    x_control = np.random.binomial(n_control, baseline_rate)
    x_treatment = np.random.binomial(n_treatment, baseline_rate + mde * 0.8)  # Actual effect is 80% of MDE

    print(f"  Control: {x_control:,}/{n_control:,} = {x_control/n_control:.2%}")
    print(f"  Treatment: {x_treatment:,}/{n_treatment:,} = {x_treatment/n_treatment:.2%}")

    # 5. Analysis
    print("\n[Step 5] Analysis")
    ab = ABTest(n_control, x_control, n_treatment, x_treatment)
    z, p = ab.z_test()
    diff, ci = ab.confidence_interval()
    lift = ab.lift()

    print(f"  Difference: {diff:.4f} ({diff*100:.2f}%p)")
    print(f"  Lift: {lift*100:.2f}%")
    print(f"  95% CI: ({ci[0]*100:.2f}%p, {ci[1]*100:.2f}%p)")
    print(f"  Z statistic: {z:.3f}")
    print(f"  p-value: {p:.4f}")

    # 6. Decision making
    print("\n[Step 6] Decision Making")
    if p < 0.05:
        if diff > 0:
            print("  Conclusion: Adopt Treatment (statistically significant improvement)")
        else:
            print("  Conclusion: Keep Control (Treatment is worse)")
    else:
        print("  Conclusion: Decision deferred (no significant difference)")
        print("  Considerations: Increase sample size or test another variant")

    return ab

ab_result = complete_ab_test_workflow()
```

---

## 7. Practice Problems

### Problem 1: Sample Size Calculation
If baseline conversion rate is 3%, and you want to detect a minimum 20% relative lift (to 3.6%):
1. Required sample size at α=0.05, Power=0.80
2. Sample size change if increasing Power to 0.90
3. Sample size change if reducing MDE to 10% lift

### Problem 2: Experiment Duration Estimation
If daily traffic is 10,000 visits with 50:50 split:
1. Days needed to achieve sample size from Problem 1
2. Minimum weeks for experiment considering weekend effects?

### Problem 3: Sequential Test Design
If planning 5 interim analyses:
1. Each analysis threshold using O'Brien-Fleming method
2. Compare with Pocock method
3. Early stopping condition at first analysis

### Problem 4: Multiple Comparison Correction
If analyzing A/B test results across 5 segments (age groups):
1. Bonferroni-corrected significance level
2. Conclusion if one segment shows p=0.02
3. How would pre-registration change this?

---

## 8. Key Summary

### Experimental Design Checklist

1. [ ] Clear hypothesis and metric definition
2. [ ] Sample size determination via power analysis
3. [ ] Randomization method selection
4. [ ] Experiment duration setting (consider weekly effects)
5. [ ] Interim analysis plan (sequential testing)
6. [ ] Multiple comparison considerations
7. [ ] Pre-registration

### Sample Size Formula (Proportion Comparison)

$$n = \frac{(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)})^2}{(p_1-p_2)^2}$$

### Power Relationships

| Factor | Sample Size When Increased |
|--------|---------------------------|
| Effect size ↑ | Decreases |
| Power ↑ | Increases |
| α ↓ (more strict) | Increases |
| Variance ↑ | Increases |

### Sequential Testing

| Method | Early | Late |
|--------|-------|------|
| O'Brien-Fleming | Very conservative | Similar to fixed sample |
| Pocock | Constant | More conservative |

### Python Libraries

```python
from statsmodels.stats.power import TTestPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy.stats import norm

# Sample size calculation
power_analysis = NormalIndPower()
n = power_analysis.solve_power(effect_size=h, alpha=0.05, power=0.80)
```

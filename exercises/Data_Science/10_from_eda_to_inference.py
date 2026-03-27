"""
Exercises for Lesson 10: From EDA to Statistical Inference
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Sampling Variability ===
# Problem: Demonstrate that different samples yield different estimates.
#   Draw 20 samples of size n=50 from a population (mean=100, std=15),
#   compute the sample mean for each, and report the range and standard
#   error of the sampling distribution.
def exercise_1():
    """Solution demonstrating sampling variability and the sampling distribution.

    When we draw multiple samples from the same population, each sample
    mean differs due to random variation. The standard error (SE) of the
    mean quantifies this variability: SE = sigma / sqrt(n). As n grows
    the SE shrinks, giving us tighter estimates.
    """
    np.random.seed(42)
    pop_mean = 100.0
    pop_std = 15.0
    population = np.random.normal(pop_mean, pop_std, size=100_000)

    n_samples = 20
    sample_size = 50
    sample_means = []

    print(f"Population: mean={pop_mean}, std={pop_std}")
    print(f"Drawing {n_samples} samples of size {sample_size}:\n")

    for i in range(n_samples):
        idx = np.random.choice(len(population), size=sample_size, replace=False)
        sample = population[idx]
        m = sample.mean()
        sample_means.append(m)
        print(f"  Sample {i+1:2d}: mean = {m:.2f}")

    sample_means = np.array(sample_means)
    print(f"\nSampling distribution of the mean:")
    print(f"  Min sample mean:  {sample_means.min():.2f}")
    print(f"  Max sample mean:  {sample_means.max():.2f}")
    print(f"  Range:            {sample_means.max() - sample_means.min():.2f}")
    print(f"  Mean of means:    {sample_means.mean():.2f}")
    print(f"  SD of means:      {sample_means.std(ddof=1):.2f}")

    # Theoretical standard error
    se_theoretical = pop_std / np.sqrt(sample_size)
    print(f"\nTheoretical SE = sigma/sqrt(n) = {pop_std}/{np.sqrt(sample_size):.2f} = {se_theoretical:.2f}")
    print(f"Observed SD of sample means: {sample_means.std(ddof=1):.2f}")
    print("These should be close, confirming the Central Limit Theorem.")


# === Exercise 2: Confidence Interval Construction ===
# Problem: Construct a 95% confidence interval for the mean of a sample
#   using the t-distribution. Then verify coverage by repeating many times.
def exercise_2():
    """Solution building and verifying t-based confidence intervals.

    A 95% CI means that if we repeated the sampling procedure many times,
    approximately 95% of the constructed intervals would contain the true
    population mean. We verify this empirically.
    """
    np.random.seed(123)
    true_mean = 50.0
    true_std = 10.0
    n = 30
    alpha = 0.05

    # Single CI construction
    sample = np.random.normal(true_mean, true_std, size=n)
    x_bar = sample.mean()
    s = sample.std(ddof=1)
    se = s / np.sqrt(n)

    # t critical value for 95% CI with n-1 degrees of freedom
    # Using the inverse CDF of the t-distribution
    from math import sqrt
    df = n - 1

    # Approximate t critical value using the normal approximation
    # (adequate for df >= 30; for smaller df we compute iteratively)
    # For df=29, t_0.025 ~ 2.045
    # We compute via the Wilson-Hilferty approximation of the t quantile
    z = 1.96  # z_{alpha/2}
    t_crit = z * (1 + (z * z - 3) / (20 * df)
                  + (7 * z**4 - 111 * z**2 + 264) / (1680 * df**2))

    ci_lower = x_bar - t_crit * se
    ci_upper = x_bar + t_crit * se

    print(f"Sample: n={n}, mean={x_bar:.2f}, std={s:.2f}")
    print(f"Standard error: {se:.2f}")
    print(f"t-critical (approx, df={df}): {t_crit:.3f}")
    print(f"95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
    print(f"True mean ({true_mean}) in CI? "
          f"{'Yes' if ci_lower <= true_mean <= ci_upper else 'No'}")

    # Coverage simulation: repeat 1000 times
    n_sim = 1000
    covers = 0
    for _ in range(n_sim):
        sim_sample = np.random.normal(true_mean, true_std, size=n)
        sim_mean = sim_sample.mean()
        sim_se = sim_sample.std(ddof=1) / np.sqrt(n)
        lo = sim_mean - t_crit * sim_se
        hi = sim_mean + t_crit * sim_se
        if lo <= true_mean <= hi:
            covers += 1

    coverage = covers / n_sim
    print(f"\nCoverage simulation ({n_sim} repetitions):")
    print(f"  Intervals containing true mean: {covers}/{n_sim}")
    print(f"  Empirical coverage: {coverage:.3f}")
    print(f"  Expected coverage:  0.950")
    print(f"  Difference:         {abs(coverage - 0.95):.3f}")


# === Exercise 3: Hypothesis Test Decision Framework ===
# Problem: Implement a function that selects the appropriate test
#   based on data type, number of groups, and whether observations are paired.
#   Then run a two-sample test on synthetic data.
def exercise_3():
    """Solution implementing a test selection guide and a manual two-sample test.

    Choosing the right statistical test depends on:
    - Data type (continuous vs categorical)
    - Number of groups
    - Whether samples are paired or independent
    - Whether assumptions (normality, equal variance) hold
    """
    def suggest_test(data_type, num_groups, paired=False):
        """Return the recommended statistical test name."""
        if data_type == "continuous":
            if num_groups == 1:
                return "One-sample t-test"
            elif num_groups == 2:
                return "Paired t-test" if paired else "Independent two-sample t-test"
            else:
                return "One-way ANOVA"
        elif data_type == "categorical":
            if num_groups == 1:
                return "One-proportion z-test"
            elif num_groups == 2:
                return "Chi-square test / Two-proportion z-test"
            else:
                return "Chi-square test for independence"
        return "Unknown"

    scenarios = [
        ("continuous", 1, False, "Test if mean exam score equals 70"),
        ("continuous", 2, False, "Compare heights of two regions"),
        ("continuous", 2, True, "Before/after blood pressure"),
        ("continuous", 3, False, "Compare sales across 3 stores"),
        ("categorical", 2, False, "Compare conversion rates A vs B"),
    ]

    print("Test Selection Guide:\n")
    for dtype, ngrp, paired, desc in scenarios:
        test = suggest_test(dtype, ngrp, paired)
        print(f"  {desc}")
        print(f"    -> {test}\n")

    # Perform an independent two-sample t-test manually
    np.random.seed(99)
    group_a = np.random.normal(75, 10, size=40)
    group_b = np.random.normal(80, 10, size=40)

    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = group_a.mean(), group_b.mean()
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)

    # Pooled standard error for equal-variance t-test
    sp2 = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    se_diff = np.sqrt(sp2 * (1.0 / n_a + 1.0 / n_b))
    t_stat = (mean_b - mean_a) / se_diff
    df = n_a + n_b - 2

    print("Independent two-sample t-test (manual):")
    print(f"  Group A: n={n_a}, mean={mean_a:.2f}, var={var_a:.2f}")
    print(f"  Group B: n={n_b}, mean={mean_b:.2f}, var={var_b:.2f}")
    print(f"  Pooled variance: {sp2:.2f}")
    print(f"  SE of difference: {se_diff:.2f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  df: {df}")

    # Approximate p-value using normal approximation (valid for large df)
    # For two-tailed test
    abs_t = abs(t_stat)
    # Use the complementary error function for normal tail probability
    from math import erfc
    p_approx = erfc(abs_t / np.sqrt(2))
    print(f"  Approximate two-tailed p-value: {p_approx:.4f}")
    print(f"  Significant at alpha=0.05? {'Yes' if p_approx < 0.05 else 'No'}")

    # Effect size (Cohen's d)
    cohens_d = (mean_b - mean_a) / np.sqrt(sp2)
    print(f"\n  Cohen's d: {cohens_d:.4f}")
    if abs(cohens_d) >= 0.8:
        size_label = "large"
    elif abs(cohens_d) >= 0.5:
        size_label = "medium"
    else:
        size_label = "small"
    print(f"  Effect size: {size_label}")


# === Exercise 4: Statistical vs Practical Significance ===
# Problem: Show that a large sample can make a tiny effect statistically
#   significant even when it has no practical importance. Compare a small
#   and a large sample with the same true difference.
def exercise_4():
    """Solution contrasting statistical and practical significance.

    With enough data, even trivially small differences become
    'statistically significant'. Effect size measures (Cohen's d)
    and confidence intervals help assess whether a result matters
    in practice, not just statistically.
    """
    np.random.seed(7)
    true_diff = 0.3  # Very small true difference
    sigma = 10.0

    from math import erfc

    for label, n in [("Small sample (n=30)", 30), ("Large sample (n=10000)", 10000)]:
        group1 = np.random.normal(100, sigma, n)
        group2 = np.random.normal(100 + true_diff, sigma, n)

        m1, m2 = group1.mean(), group2.mean()
        var1 = group1.var(ddof=1)
        var2 = group2.var(ddof=1)

        # Pooled t-test
        sp2 = ((n - 1) * var1 + (n - 1) * var2) / (2 * n - 2)
        se = np.sqrt(sp2 * 2.0 / n)
        t_val = (m2 - m1) / se
        p_val = erfc(abs(t_val) / np.sqrt(2))

        # Cohen's d
        d = (m2 - m1) / np.sqrt(sp2)

        print(f"{label}:")
        print(f"  Observed difference: {m2 - m1:.4f}")
        print(f"  t-statistic: {t_val:.4f}")
        print(f"  p-value (approx): {p_val:.4f}")
        print(f"  Significant at alpha=0.05? {'Yes' if p_val < 0.05 else 'No'}")
        print(f"  Cohen's d: {d:.4f}")
        if abs(d) >= 0.2:
            d_label = "small or larger"
        else:
            d_label = "negligible"
        print(f"  Practical importance: {d_label}")
        print()

    print("Lesson: Always report effect sizes alongside p-values.")
    print("A statistically significant result with negligible effect size")
    print("may not warrant action in a real-world context.")


if __name__ == "__main__":
    print("=== Exercise 1: Sampling Variability ===")
    exercise_1()
    print("\n=== Exercise 2: Confidence Interval Construction ===")
    exercise_2()
    print("\n=== Exercise 3: Hypothesis Test Decision Framework ===")
    exercise_3()
    print("\n=== Exercise 4: Statistical vs Practical Significance ===")
    exercise_4()
    print("\nAll exercises completed!")

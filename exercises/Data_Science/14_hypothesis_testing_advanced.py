"""
Exercises for Lesson 14: Hypothesis Testing Advanced
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


# === Exercise 1: Power Analysis ===
# Problem: Mean difference = 5, SD = 12.
#   (a) Power with n=30 per group?
#   (b) Minimum n for 80% power?
def exercise_1():
    """Solution for statistical power analysis using the non-central t-distribution.

    Power = P(reject H0 | H1 is true).
    For a two-sample t-test:
        - Effect size (Cohen's d) = mean_diff / SD
        - Non-centrality parameter delta = d * sqrt(n/2)
    Higher power means lower risk of Type II error (failing to detect a real effect).
    """
    mean_diff = 5
    sd = 12
    alpha = 0.05

    # Cohen's d = standardized effect size
    d = mean_diff / sd
    print(f"Mean difference: {mean_diff}")
    print(f"Standard deviation: {sd}")
    print(f"Cohen's d = {mean_diff}/{sd} = {d:.4f}")
    print(f"  (d < 0.2: small, 0.2-0.8: medium, > 0.8: large)")

    # (a) Power with n=30 per group
    n = 30
    df = 2 * n - 2  # degrees of freedom for two-sample t-test

    # Non-centrality parameter: how far the true distribution is from H0
    ncp = d * np.sqrt(n / 2)

    # Critical value under H0
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Power = P(|T| > t_crit | H1) using the non-central t-distribution
    # Power = 1 - P(-t_crit < T < t_crit | ncp)
    power = 1 - (stats.nct.cdf(t_crit, df, ncp) - stats.nct.cdf(-t_crit, df, ncp))

    print(f"\n(a) Power with n={n} per group:")
    print(f"    df = {df}")
    print(f"    Non-centrality parameter: {ncp:.4f}")
    print(f"    t-critical: {t_crit:.4f}")
    print(f"    Power = {power:.4f} ({power*100:.1f}%)")

    # (b) Minimum n for 80% power
    # Iterative search: increase n until power >= 0.80
    target_power = 0.80
    for n_try in range(5, 500):
        df_try = 2 * n_try - 2
        ncp_try = d * np.sqrt(n_try / 2)
        t_crit_try = stats.t.ppf(1 - alpha / 2, df_try)
        power_try = 1 - (stats.nct.cdf(t_crit_try, df_try, ncp_try)
                         - stats.nct.cdf(-t_crit_try, df_try, ncp_try))
        if power_try >= target_power:
            print(f"\n(b) Minimum n for {target_power*100:.0f}% power:")
            print(f"    n = {n_try} per group (total = {2*n_try})")
            print(f"    Achieved power = {power_try:.4f}")
            break

    # Verification using the approximate formula: n ~ (z_alpha/2 + z_beta)^2 * 2 / d^2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(target_power)
    n_approx = int(np.ceil(2 * (z_alpha + z_beta)**2 / d**2))
    print(f"\n    Approximate formula: n ~ {n_approx}")
    print(f"    (z_alpha/2 + z_beta)^2 * 2 / d^2 = "
          f"({z_alpha:.2f} + {z_beta:.2f})^2 * 2 / {d:.4f}^2")


# === Exercise 2: Multiple Testing ===
# Problem: Apply Bonferroni and BH corrections to 10 p-values.
def exercise_2():
    """Solution for multiple testing correction methods.

    When performing m simultaneous tests, the probability of at least one
    false positive (family-wise error rate, FWER) inflates. Corrections:
    - Bonferroni: controls FWER by dividing alpha by m (conservative).
    - Benjamini-Hochberg: controls FDR (false discovery rate), less conservative.
    """
    p_values = np.array([0.001, 0.005, 0.010, 0.020, 0.040,
                         0.080, 0.120, 0.200, 0.500, 0.800])
    alpha = 0.05
    m = len(p_values)

    print(f"Original p-values: {p_values.tolist()}")
    print(f"Number of tests: m = {m}")
    print(f"Significance level: alpha = {alpha}")

    # Bonferroni correction: reject if p_i < alpha/m
    # Equivalently, adjusted p_i = min(m * p_i, 1)
    reject_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=alpha,
                                                   method='bonferroni')
    print(f"\nBonferroni correction (controls FWER):")
    print(f"  Adjusted alpha: {alpha/m:.4f}")
    print(f"  Adjusted p-values: {np.round(pvals_bonf, 4).tolist()}")
    print(f"  Rejections: {reject_bonf.tolist()}")
    print(f"  Number rejected: {reject_bonf.sum()}")

    # Benjamini-Hochberg (BH) correction: controls FDR
    # Sort p-values, reject if p_(i) <= i/m * alpha
    reject_bh, pvals_bh, _, _ = multipletests(p_values, alpha=alpha,
                                               method='fdr_bh')
    print(f"\nBenjamini-Hochberg correction (controls FDR):")
    print(f"  Adjusted p-values: {np.round(pvals_bh, 4).tolist()}")
    print(f"  Rejections: {reject_bh.tolist()}")
    print(f"  Number rejected: {reject_bh.sum()}")

    # Comparison summary
    print(f"\nComparison:")
    print(f"  {'p-value':<10} {'Bonferroni':<12} {'BH (FDR)':<12} {'Unadjusted':<12}")
    print(f"  {'-'*46}")
    for i in range(m):
        unadj = "Reject" if p_values[i] < alpha else "Fail"
        bonf = "Reject" if reject_bonf[i] else "Fail"
        bh = "Reject" if reject_bh[i] else "Fail"
        print(f"  {p_values[i]:<10.3f} {bonf:<12} {bh:<12} {unadj:<12}")

    print(f"\n  Unadjusted: {(p_values < alpha).sum()} rejections")
    print(f"  Bonferroni: {reject_bonf.sum()} rejections (most conservative)")
    print(f"  BH (FDR):   {reject_bh.sum()} rejections (less conservative)")


# === Exercise 3: Effect Size ===
# Problem: Calculate Cohen's d and Hedges' g for two groups.
def exercise_3():
    """Solution computing standardized effect sizes.

    Cohen's d uses pooled standard deviation assuming equal variances.
    Hedges' g applies a correction factor for small-sample bias.
    Both measure "how many standard deviations apart" the group means are.
    """
    group1 = np.array([23, 25, 27, 22, 28, 26, 24, 29, 25, 27])
    group2 = np.array([30, 32, 28, 31, 33, 29, 35, 31, 30, 32])

    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)

    print(f"Group 1: mean={m1:.2f}, sd={s1:.4f}, n={n1}")
    print(f"Group 2: mean={m2:.2f}, sd={s2:.4f}, n={n2}")
    print(f"Mean difference: {m2 - m1:.2f}")

    # Pooled standard deviation (assuming equal variances)
    # s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    # Cohen's d
    cohens_d = (m2 - m1) / s_pooled
    print(f"\nCohen's d = (m2 - m1) / s_pooled = {m2-m1:.2f} / {s_pooled:.4f} = {cohens_d:.4f}")

    # Interpretation
    if abs(cohens_d) >= 0.8:
        interpretation = "large"
    elif abs(cohens_d) >= 0.5:
        interpretation = "medium"
    else:
        interpretation = "small"
    print(f"  Interpretation: {interpretation} effect size")

    # Hedges' g: bias-corrected version for small samples
    # Correction factor: J = 1 - 3/(4(n1+n2-2) - 1)
    df = n1 + n2 - 2
    J = 1 - 3 / (4 * df - 1)
    hedges_g = cohens_d * J

    print(f"\nHedges' g (small-sample corrected):")
    print(f"  Correction factor J = 1 - 3/(4*{df} - 1) = {J:.6f}")
    print(f"  Hedges' g = Cohen's d * J = {cohens_d:.4f} * {J:.6f} = {hedges_g:.4f}")
    print(f"  Note: Hedges' g is slightly smaller than Cohen's d,")
    print(f"  correcting for the upward bias in small samples.")

    # Confirm with a t-test
    t_stat, p_val = stats.ttest_ind(group1, group2)
    print(f"\nIndependent t-test: t={t_stat:.4f}, p={p_val:.6f}")
    print(f"  Highly significant difference between groups.")


if __name__ == "__main__":
    print("=== Exercise 1: Power Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Multiple Testing ===")
    exercise_2()
    print("\n=== Exercise 3: Effect Size ===")
    exercise_3()
    print("\nAll exercises completed!")

"""
Exercises for Lesson 13: Confidence Intervals
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
from scipy import stats


# === Exercise 1: t-Confidence Interval ===
# Problem: Construct a 95% CI for the population mean from a sample.
def exercise_1():
    """Solution constructing a t-based confidence interval for the mean.

    When the population standard deviation is unknown (the usual case),
    we use the t-distribution with n-1 degrees of freedom.
    The t-distribution has heavier tails than the normal, producing
    wider intervals that account for the added uncertainty from estimating sigma.
    """
    sample = np.array([23, 25, 28, 22, 26, 24, 27, 25, 29, 24])
    n = len(sample)
    x_bar = sample.mean()
    s = sample.std(ddof=1)  # ddof=1 for unbiased sample standard deviation
    se = s / np.sqrt(n)

    print(f"Sample: {sample.tolist()}")
    print(f"n = {n}")
    print(f"x_bar = {x_bar:.4f}")
    print(f"s = {s:.4f}")
    print(f"SE = s/sqrt(n) = {se:.4f}")

    # t-critical value for 95% CI with n-1 df
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    ci_lower = x_bar - t_crit * se
    ci_upper = x_bar + t_crit * se

    print(f"\nt-critical value (df={n-1}): {t_crit:.4f}")
    print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"Margin of error: {t_crit * se:.4f}")

    # Verification using scipy
    ci_scipy = stats.t.interval(0.95, df=n - 1, loc=x_bar, scale=se)
    print(f"\nScipy verification: ({ci_scipy[0]:.4f}, {ci_scipy[1]:.4f})")

    # Interpretation: We are 95% confident that the true population mean
    # falls within this interval.
    print(f"\nInterpretation: We are 95% confident the population mean")
    print(f"lies between {ci_lower:.2f} and {ci_upper:.2f}.")


# === Exercise 2: Confidence Interval for Proportion ===
# Problem: 230 out of 500 agreed.
#   (a) 95% CI using normal approximation (Wald)
#   (b) Wilson method
#   (c) What happens with n=2000?
def exercise_2():
    """Solution for confidence intervals of proportions.

    The Wald interval (normal approximation) is simple but can have poor
    coverage near 0 or 1. The Wilson interval is preferred for smaller
    samples or extreme proportions because it corrects for this.
    """
    x, n = 230, 500
    p_hat = x / n
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)

    print(f"Survey: {x} agreed out of {n}")
    print(f"p_hat = {p_hat:.4f}")

    # (a) Wald (normal approximation) CI
    # p_hat +/- z * sqrt(p_hat * (1 - p_hat) / n)
    se_wald = np.sqrt(p_hat * (1 - p_hat) / n)
    ci_wald = (p_hat - z * se_wald, p_hat + z * se_wald)
    print(f"\n(a) Wald CI (normal approximation):")
    print(f"    SE = sqrt(p*(1-p)/n) = {se_wald:.6f}")
    print(f"    95% CI: ({ci_wald[0]:.6f}, {ci_wald[1]:.6f})")
    print(f"    Width: {ci_wald[1] - ci_wald[0]:.6f}")

    # (b) Wilson CI
    # Wilson interval: (p_hat + z^2/(2n) +/- z*sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2))) / (1 + z^2/n)
    center = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z / (1 + z**2 / n)) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    ci_wilson = (center - margin, center + margin)
    print(f"\n(b) Wilson CI:")
    print(f"    95% CI: ({ci_wilson[0]:.6f}, {ci_wilson[1]:.6f})")
    print(f"    Width: {ci_wilson[1] - ci_wilson[0]:.6f}")
    print(f"    Note: Wilson CI is slightly shifted toward 0.5 (shrinkage)")

    # (c) Larger sample size: n=2000, same proportion
    n2 = 2000
    x2 = int(p_hat * n2)
    se_wald_2 = np.sqrt(p_hat * (1 - p_hat) / n2)
    ci_wald_2 = (p_hat - z * se_wald_2, p_hat + z * se_wald_2)
    print(f"\n(c) With n={n2} (same p_hat={p_hat:.4f}):")
    print(f"    SE = {se_wald_2:.6f}")
    print(f"    95% CI: ({ci_wald_2[0]:.6f}, {ci_wald_2[1]:.6f})")
    print(f"    Width: {ci_wald_2[1] - ci_wald_2[0]:.6f}")
    print(f"\n    Width ratio (n={n} vs n={n2}): "
          f"{(ci_wald[1]-ci_wald[0])/(ci_wald_2[1]-ci_wald_2[0]):.2f}x")
    print(f"    Width decreases proportionally to 1/sqrt(n): "
          f"sqrt({n2}/{n}) = {np.sqrt(n2/n):.2f}")


# === Exercise 3: Bootstrap ===
# Problem: 95% bootstrap CIs for both mean and median of an exponential distribution.
def exercise_3():
    """Solution using bootstrap resampling for nonparametric CI construction.

    Bootstrap is especially valuable for:
    - Statistics without closed-form CI formulas (e.g., median, trimmed mean)
    - Skewed distributions where normal approximation is poor
    - Small samples where asymptotic theory is unreliable
    """
    np.random.seed(42)
    data = np.random.exponential(10, 40)

    print(f"Data: exponential(scale=10), n={len(data)}")
    print(f"Sample mean: {data.mean():.4f}")
    print(f"Sample median: {np.median(data):.4f}")
    print(f"Skewness: {stats.skew(data):.4f} (right-skewed)")

    # Bootstrap resampling
    n_bootstrap = 10_000
    rng = np.random.default_rng(42)
    boot_means = np.empty(n_bootstrap)
    boot_medians = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement (same size as original)
        boot_sample = rng.choice(data, size=len(data), replace=True)
        boot_means[i] = boot_sample.mean()
        boot_medians[i] = np.median(boot_sample)

    # Percentile method: use the 2.5th and 97.5th percentiles of the
    # bootstrap distribution as CI bounds
    ci_mean = np.percentile(boot_means, [2.5, 97.5])
    ci_median = np.percentile(boot_medians, [2.5, 97.5])

    print(f"\nBootstrap 95% CI (percentile method, B={n_bootstrap:,}):")
    print(f"  Mean:   ({ci_mean[0]:.4f}, {ci_mean[1]:.4f})")
    print(f"  Median: ({ci_median[0]:.4f}, {ci_median[1]:.4f})")

    # Compare with parametric t-interval for the mean
    se_mean = data.std(ddof=1) / np.sqrt(len(data))
    t_crit = stats.t.ppf(0.975, df=len(data) - 1)
    ci_t = (data.mean() - t_crit * se_mean, data.mean() + t_crit * se_mean)
    print(f"\nParametric t-CI for mean: ({ci_t[0]:.4f}, {ci_t[1]:.4f})")
    print(f"  Note: t-CI assumes normality, which is violated here (skewed data).")
    print(f"  Bootstrap CI is more appropriate for skewed distributions.")

    # BCa (Bias-Corrected and Accelerated) bootstrap via scipy
    boot_result_mean = stats.bootstrap(
        (data,), np.mean, n_resamples=n_bootstrap, random_state=42
    )
    boot_result_median = stats.bootstrap(
        (data,), np.median, n_resamples=n_bootstrap, random_state=42
    )
    print(f"\nScipy bootstrap (BCa) 95% CI:")
    print(f"  Mean:   ({boot_result_mean.confidence_interval.low:.4f}, "
          f"{boot_result_mean.confidence_interval.high:.4f})")
    print(f"  Median: ({boot_result_median.confidence_interval.low:.4f}, "
          f"{boot_result_median.confidence_interval.high:.4f})")


if __name__ == "__main__":
    print("=== Exercise 1: t-Confidence Interval ===")
    exercise_1()
    print("\n=== Exercise 2: Confidence Interval for Proportion ===")
    exercise_2()
    print("\n=== Exercise 3: Bootstrap ===")
    exercise_3()
    print("\nAll exercises completed!")

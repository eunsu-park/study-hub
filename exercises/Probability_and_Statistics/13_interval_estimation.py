"""
Probability and Statistics — Interval Estimation
Exercises covering z/t confidence interval computation, sample size
determination, coverage simulation, and bootstrap confidence intervals.
"""
import math
import random
from typing import Callable, List, Tuple


# === Helpers ===

def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def z_quantile(p: float, tol: float = 1e-8) -> float:
    """Inverse normal CDF via bisection."""
    lo, hi = -6.0, 6.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# === Exercise 1: Z and T Confidence Interval Computation ===
def exercise_1() -> None:
    """Compute 95% z-interval (sigma known) and t-interval (sigma unknown)
    for a sample, and compare their widths."""
    print("=== Exercise 1: Z and T Confidence Interval Computation ===")

    random.seed(42)
    true_mu = 50.0
    true_sigma = 10.0
    n = 25
    data = [random.gauss(true_mu, true_sigma) for _ in range(n)]

    x_bar = sum(data) / n
    s = math.sqrt(sum((x - x_bar) ** 2 for x in data) / (n - 1))
    confidence = 0.95
    alpha = 1 - confidence

    # Z-interval (sigma known)
    z_crit = z_quantile(1 - alpha / 2)
    z_margin = z_crit * true_sigma / math.sqrt(n)
    z_lo = x_bar - z_margin
    z_hi = x_bar + z_margin

    # T-interval (sigma unknown) with Cornish-Fisher t approximation
    nu = n - 1
    z = z_quantile(1 - alpha / 2)
    t_approx = z + (z ** 3 + z) / (4 * nu) + \
               (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * nu ** 2)
    t_margin = t_approx * s / math.sqrt(n)
    t_lo = x_bar - t_margin
    t_hi = x_bar + t_margin

    print(f"\n  Sample: n={n}, x_bar={x_bar:.3f}, s={s:.3f}")
    print(f"  True mu={true_mu}, sigma={true_sigma}")

    print(f"\n  Z-interval (sigma={true_sigma} known):")
    print(f"    z_crit = {z_crit:.4f}")
    print(f"    Margin = {z_crit:.4f} * {true_sigma}/{math.sqrt(n):.3f} = {z_margin:.3f}")
    print(f"    95% CI: ({z_lo:.3f}, {z_hi:.3f})")
    print(f"    Contains mu={true_mu}? {'Yes' if z_lo <= true_mu <= z_hi else 'No'}")

    print(f"\n  T-interval (sigma unknown):")
    print(f"    t_approx (df={nu}) = {t_approx:.4f}")
    print(f"    Margin = {t_approx:.4f} * {s:.3f}/{math.sqrt(n):.3f} = {t_margin:.3f}")
    print(f"    95% CI: ({t_lo:.3f}, {t_hi:.3f})")
    print(f"    Contains mu={true_mu}? {'Yes' if t_lo <= true_mu <= t_hi else 'No'}")

    print(f"\n  Width comparison:")
    print(f"    Z-interval width = {2 * z_margin:.3f}")
    print(f"    T-interval width = {2 * t_margin:.3f}")
    print(f"    T-interval is wider due to uncertainty in sigma.\n")


# === Exercise 2: Sample Size Determination ===
def exercise_2() -> None:
    """Determine the required sample size to achieve a desired margin of
    error for both mean and proportion CIs."""
    print("=== Exercise 2: Sample Size Determination ===")

    # For mean: n >= (z * sigma / E)^2
    sigma = 10.0
    print(f"\n  (a) Sample size for mean CI (sigma={sigma}, 95% confidence):")
    print(f"      Formula: n >= (z * sigma / E)^2")
    z_crit = z_quantile(0.975)

    print(f"\n  {'Margin E':>10} {'n required':>12}")
    print(f"  {'--------':>10} {'----------':>12}")
    for e in [5.0, 3.0, 2.0, 1.0, 0.5]:
        n_req = math.ceil((z_crit * sigma / e) ** 2)
        print(f"  {e:>10.1f} {n_req:>12}")

    # For proportion: n >= z^2 * p(1-p) / E^2
    print(f"\n  (b) Sample size for proportion CI (95% confidence):")
    print(f"      Formula: n >= z^2 * p*(1-p) / E^2")
    print(f"\n  {'Margin E':>10} {'n (p=0.5)':>12} {'n (p=0.1)':>12}")
    print(f"  {'--------':>10} {'---------':>12} {'---------':>12}")
    for e in [0.05, 0.03, 0.02, 0.01]:
        n_05 = math.ceil(z_crit ** 2 * 0.5 * 0.5 / e ** 2)
        n_01 = math.ceil(z_crit ** 2 * 0.1 * 0.9 / e ** 2)
        print(f"  {e:>10.2f} {n_05:>12} {n_01:>12}")

    print(f"\n  Key insight: p=0.5 is worst-case (largest n).")
    print(f"  Conservative approach: use p=0.5 when p is unknown.\n")


# === Exercise 3: Coverage Simulation ===
def exercise_3() -> None:
    """Simulate many confidence intervals to verify that the empirical
    coverage rate matches the nominal confidence level."""
    print("=== Exercise 3: Coverage Simulation ===")

    random.seed(42)
    true_mu = 100.0
    sigma = 15.0
    n = 30
    n_sims = 5000

    z_crit = z_quantile(0.975)

    print(f"\n  Setup: true mu={true_mu}, sigma={sigma}, n={n}")
    print(f"  Simulating {n_sims:,} confidence intervals at 90%, 95%, 99% levels")

    for confidence in [0.90, 0.95, 0.99]:
        z_c = z_quantile(1 - (1 - confidence) / 2)
        margin = z_c * sigma / math.sqrt(n)
        covered = 0
        for _ in range(n_sims):
            sample = [random.gauss(true_mu, sigma) for _ in range(n)]
            x_bar = sum(sample) / n
            lo = x_bar - margin
            hi = x_bar + margin
            if lo <= true_mu <= hi:
                covered += 1
        coverage = covered / n_sims
        print(f"\n    {confidence*100:.0f}% CI (z={z_c:.3f}, margin={margin:.3f}):")
        print(f"      Empirical coverage: {coverage:.4f} "
              f"({'OK' if abs(coverage - confidence) < 0.02 else 'DRIFT'})")

    # Also check t-interval coverage with unknown sigma
    print(f"\n  T-interval coverage check (sigma unknown, n={n}):")
    confidence = 0.95
    z = z_quantile(0.975)
    nu = n - 1
    t_approx = z + (z ** 3 + z) / (4 * nu) + \
               (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * nu ** 2)
    covered = 0
    for _ in range(n_sims):
        sample = [random.gauss(true_mu, sigma) for _ in range(n)]
        x_bar = sum(sample) / n
        s = math.sqrt(sum((x - x_bar) ** 2 for x in sample) / (n - 1))
        margin = t_approx * s / math.sqrt(n)
        if x_bar - margin <= true_mu <= x_bar + margin:
            covered += 1
    print(f"    Empirical coverage: {covered / n_sims:.4f} (nominal: 0.95)")
    print()


# === Exercise 4: Bootstrap Confidence Interval ===
def exercise_4() -> None:
    """Construct a bootstrap percentile CI for the median of a skewed
    distribution, and compare with the bootstrap CI for the mean."""
    print("=== Exercise 4: Bootstrap Confidence Interval ===")

    random.seed(42)
    # Skewed data from Exponential(0.5): true median = ln(2)/0.5 = 1.3863
    n = 40
    lam = 0.5
    data = [random.expovariate(lam) for _ in range(n)]
    true_median = math.log(2) / lam

    data_sorted = sorted(data)
    sample_median = (data_sorted[n // 2 - 1] + data_sorted[n // 2]) / 2
    sample_mean = sum(data) / n

    def median_fn(d: List[float]) -> float:
        s = sorted(d)
        m = len(s)
        if m % 2 == 0:
            return (s[m // 2 - 1] + s[m // 2]) / 2
        return s[m // 2]

    def mean_fn(d: List[float]) -> float:
        return sum(d) / len(d)

    def bootstrap_ci(data: List[float], stat_fn: Callable,
                     B: int = 5000, confidence: float = 0.95) -> Tuple[float, float]:
        """Percentile bootstrap CI."""
        boot_stats = []
        m = len(data)
        for _ in range(B):
            resample = [data[random.randint(0, m - 1)] for _ in range(m)]
            boot_stats.append(stat_fn(resample))
        boot_stats.sort()
        alpha = 1 - confidence
        lo_idx = int(B * alpha / 2)
        hi_idx = int(B * (1 - alpha / 2))
        return boot_stats[lo_idx], boot_stats[hi_idx]

    print(f"\n  Data: {n} samples from Exponential({lam})")
    print(f"  True median = ln(2)/{lam} = {true_median:.4f}")
    print(f"  Sample median = {sample_median:.4f}")
    print(f"  Sample mean = {sample_mean:.4f}")

    B = 5000
    med_lo, med_hi = bootstrap_ci(data, median_fn, B)
    mean_lo, mean_hi = bootstrap_ci(data, mean_fn, B)

    print(f"\n  Bootstrap percentile CI (B={B}):")
    print(f"    Median: ({med_lo:.4f}, {med_hi:.4f})  width={med_hi - med_lo:.4f}")
    print(f"      Contains true median? "
          f"{'Yes' if med_lo <= true_median <= med_hi else 'No'}")
    print(f"    Mean:   ({mean_lo:.4f}, {mean_hi:.4f})  width={mean_hi - mean_lo:.4f}")

    # Bootstrap SE of median
    boot_medians = []
    for _ in range(B):
        resample = [data[random.randint(0, n - 1)] for _ in range(n)]
        boot_medians.append(median_fn(resample))
    boot_se = math.sqrt(sum((m - sum(boot_medians) / B) ** 2
                            for m in boot_medians) / (B - 1))
    print(f"\n  Bootstrap SE of median = {boot_se:.4f}")
    print(f"  (No closed-form SE for median -- bootstrap is ideal here.)\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

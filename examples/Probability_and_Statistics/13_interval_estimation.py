"""
Interval Estimation

Demonstrates:
1. Z-interval and T-interval for Normal Mean
2. Confidence Interval for Proportion
3. Sample Size Determination
4. Bootstrap Confidence Intervals

Theory:
- CI for μ (σ known): x̄ ± z_{α/2} · σ/√n
- CI for μ (σ unknown): x̄ ± t_{α/2,n-1} · s/√n
- CI for p (Wald): p̂ ± z_{α/2} · √(p̂(1-p̂)/n)
- Bootstrap: resample with replacement → percentile CI

Adapted from Probability and Statistics Lesson 13.
"""

import math
import random


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def z_quantile(p: float, tol: float = 1e-8) -> float:
    """Inverse normal CDF via bisection."""
    lo, hi = -6, 6
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────
# 1. Z-INTERVAL AND T-INTERVAL
# ─────────────────────────────────────────────────

def z_confidence_interval(data: list[float], sigma: float,
                           confidence: float = 0.95) -> tuple:
    """CI for μ when σ is known."""
    n = len(data)
    x_bar = sum(data) / n
    alpha = 1 - confidence
    z = z_quantile(1 - alpha / 2)
    margin = z * sigma / math.sqrt(n)
    return x_bar - margin, x_bar + margin, margin


def t_confidence_interval(data: list[float],
                           confidence: float = 0.95) -> tuple:
    """CI for μ when σ is unknown (approximate t-quantile)."""
    n = len(data)
    x_bar = sum(data) / n
    s = math.sqrt(sum((x - x_bar)**2 for x in data) / (n - 1))
    alpha = 1 - confidence
    # Approximate t-quantile (good for n ≥ 30)
    z = z_quantile(1 - alpha / 2)
    # Cornish-Fisher expansion for better t approximation
    nu = n - 1
    t_approx = z + (z**3 + z) / (4 * nu) + (5*z**5 + 16*z**3 + 3*z) / (96 * nu**2)
    margin = t_approx * s / math.sqrt(n)
    return x_bar - margin, x_bar + margin, margin


def demo_ci_mean():
    print("=" * 60)
    print("  Confidence Intervals for Mean")
    print("=" * 60)

    random.seed(42)
    true_mu = 50
    true_sigma = 10
    data = [random.gauss(true_mu, true_sigma) for _ in range(30)]
    x_bar = sum(data) / len(data)

    print(f"\n  Sample: n={len(data)}, x̄={x_bar:.2f}, true μ={true_mu}")

    # Z-interval (σ known)
    lo, hi, m = z_confidence_interval(data, true_sigma, 0.95)
    print(f"\n  Z-interval (σ={true_sigma} known):")
    print(f"  95% CI: ({lo:.2f}, {hi:.2f}), margin={m:.2f}")
    print(f"  Contains μ={true_mu}? {'Yes' if lo <= true_mu <= hi else 'No'}")

    # T-interval (σ unknown)
    lo, hi, m = t_confidence_interval(data, 0.95)
    print(f"\n  T-interval (σ unknown):")
    print(f"  95% CI: ({lo:.2f}, {hi:.2f}), margin={m:.2f}")

    # Coverage simulation
    print(f"\n  Coverage simulation (1000 CIs at 95%):")
    covered = 0
    for _ in range(1000):
        sample = [random.gauss(true_mu, true_sigma) for _ in range(30)]
        lo, hi, _ = z_confidence_interval(sample, true_sigma, 0.95)
        if lo <= true_mu <= hi:
            covered += 1
    print(f"  Actual coverage: {covered/10:.1f}% (target: 95%)")


# ─────────────────────────────────────────────────
# 2. CONFIDENCE INTERVAL FOR PROPORTION
# ─────────────────────────────────────────────────

def wald_ci(successes: int, n: int,
             confidence: float = 0.95) -> tuple:
    """Wald CI: p̂ ± z·√(p̂(1-p̂)/n)."""
    p_hat = successes / n
    z = z_quantile(1 - (1 - confidence) / 2)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat - margin, p_hat + margin


def wilson_ci(successes: int, n: int,
               confidence: float = 0.95) -> tuple:
    """Wilson score interval (better for small n or extreme p)."""
    p_hat = successes / n
    z = z_quantile(1 - (1 - confidence) / 2)
    z2 = z ** 2
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2)) / denom
    return center - margin, center + margin


def demo_ci_proportion():
    print("\n" + "=" * 60)
    print("  Confidence Interval for Proportion")
    print("=" * 60)

    cases = [
        (30, 100, "Moderate sample"),
        (3, 10, "Small sample"),
        (95, 100, "Extreme proportion"),
    ]

    for successes, n, desc in cases:
        p_hat = successes / n
        w_lo, w_hi = wald_ci(successes, n)
        ws_lo, ws_hi = wilson_ci(successes, n)
        print(f"\n  {desc}: {successes}/{n} (p̂={p_hat:.2f})")
        print(f"  Wald:   ({max(0,w_lo):.4f}, {min(1,w_hi):.4f})")
        print(f"  Wilson: ({ws_lo:.4f}, {ws_hi:.4f})")


# ─────────────────────────────────────────────────
# 3. SAMPLE SIZE DETERMINATION
# ─────────────────────────────────────────────────

def sample_size_mean(sigma: float, margin: float,
                      confidence: float = 0.95) -> int:
    """n ≥ (z·σ/E)² for desired margin E."""
    z = z_quantile(1 - (1 - confidence) / 2)
    return math.ceil((z * sigma / margin) ** 2)


def sample_size_proportion(margin: float,
                            confidence: float = 0.95,
                            p_est: float = 0.5) -> int:
    """n ≥ z²·p(1-p)/E² for desired margin E."""
    z = z_quantile(1 - (1 - confidence) / 2)
    return math.ceil(z**2 * p_est * (1 - p_est) / margin**2)


def demo_sample_size():
    print("\n" + "=" * 60)
    print("  Sample Size Determination")
    print("=" * 60)

    # For mean
    print(f"\n  Required n for mean (σ=10, 95% CI):")
    print(f"  {'Margin E':>10} {'n':>8}")
    print(f"  {'─' * 20}")
    for e in [5, 3, 2, 1, 0.5]:
        n = sample_size_mean(10, e)
        print(f"  {e:>10.1f} {n:>8}")

    # For proportion
    print(f"\n  Required n for proportion (95% CI):")
    print(f"  {'Margin E':>10} {'n (p=0.5)':>10} {'n (p=0.1)':>10}")
    print(f"  {'─' * 32}")
    for e in [0.05, 0.03, 0.02, 0.01]:
        n1 = sample_size_proportion(e, p_est=0.5)
        n2 = sample_size_proportion(e, p_est=0.1)
        print(f"  {e:>10.2f} {n1:>10} {n2:>10}")


# ─────────────────────────────────────────────────
# 4. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────

def bootstrap_ci(data: list[float], stat_fn,
                  B: int = 5000, confidence: float = 0.95) -> tuple:
    """Percentile bootstrap CI."""
    boot_stats = []
    n = len(data)
    for _ in range(B):
        resample = [data[random.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(stat_fn(resample))

    boot_stats.sort()
    alpha = 1 - confidence
    lo_idx = int(B * alpha / 2)
    hi_idx = int(B * (1 - alpha / 2))
    return boot_stats[lo_idx], boot_stats[hi_idx]


def demo_bootstrap():
    print("\n" + "=" * 60)
    print("  Bootstrap Confidence Intervals")
    print("=" * 60)

    random.seed(42)

    # Median CI (no closed form)
    data = [random.expovariate(0.5) for _ in range(50)]
    data.sort()
    median = data[len(data) // 2]

    mean_fn = lambda d: sum(d) / len(d)
    median_fn = lambda d: sorted(d)[len(d) // 2]
    var_fn = lambda d: sum((x - sum(d)/len(d))**2 for x in d) / (len(d) - 1)

    print(f"\n  Exponential sample (n=50):")
    for name, fn, point in [("Mean", mean_fn, mean_fn(data)),
                             ("Median", median_fn, median),
                             ("Variance", var_fn, var_fn(data))]:
        lo, hi = bootstrap_ci(data, fn)
        print(f"  {name}: {point:.3f}, 95% CI: ({lo:.3f}, {hi:.3f})")

    # Compare bootstrap vs parametric for mean
    true_mu = 2.0  # 1/λ for Exp(0.5)
    data_m = [random.expovariate(0.5) for _ in range(30)]
    lo_t, hi_t, _ = t_confidence_interval(data_m)
    lo_b, hi_b = bootstrap_ci(data_m, mean_fn)
    print(f"\n  Comparison for mean (n=30, Exponential):")
    print(f"  T-interval:  ({lo_t:.3f}, {hi_t:.3f})")
    print(f"  Bootstrap:   ({lo_b:.3f}, {hi_b:.3f})")


if __name__ == "__main__":
    demo_ci_mean()
    demo_ci_proportion()
    demo_sample_size()
    demo_bootstrap()

"""
Hypothesis Testing

Demonstrates:
1. Z-test and T-test
2. Chi-squared Goodness of Fit
3. Power Analysis
4. Multiple Testing Correction

Theory:
- H₀ vs H₁, Type I (α) and Type II (β) errors
- Test statistic → p-value → decision
- Power = 1 - β = P(reject H₀ | H₁ true)
- Bonferroni: reject if p < α/m

Adapted from Probability and Statistics Lesson 14.
"""

import math
import random


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def z_quantile(p: float, tol: float = 1e-8) -> float:
    lo, hi = -6, 6
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────
# 1. Z-TEST AND T-TEST
# ─────────────────────────────────────────────────

def z_test(data: list[float], mu0: float, sigma: float,
           alternative: str = "two-sided") -> dict:
    """One-sample z-test (σ known)."""
    n = len(data)
    x_bar = sum(data) / n
    z = (x_bar - mu0) / (sigma / math.sqrt(n))

    if alternative == "two-sided":
        p_value = 2 * (1 - phi(abs(z)))
    elif alternative == "greater":
        p_value = 1 - phi(z)
    else:  # less
        p_value = phi(z)

    return {"z": z, "p_value": p_value, "x_bar": x_bar, "n": n}


def t_test_one_sample(data: list[float], mu0: float,
                       alternative: str = "two-sided") -> dict:
    """One-sample t-test (σ unknown, approximate p)."""
    n = len(data)
    x_bar = sum(data) / n
    s = math.sqrt(sum((x - x_bar)**2 for x in data) / (n - 1))
    t = (x_bar - mu0) / (s / math.sqrt(n))

    # Approximate p-value using normal (good for n ≥ 30)
    if alternative == "two-sided":
        p_value = 2 * (1 - phi(abs(t)))
    elif alternative == "greater":
        p_value = 1 - phi(t)
    else:
        p_value = phi(t)

    return {"t": t, "p_value": p_value, "x_bar": x_bar, "s": s, "df": n - 1}


def t_test_two_sample(data1: list[float], data2: list[float]) -> dict:
    """Two-sample t-test (Welch's approximation)."""
    n1, n2 = len(data1), len(data2)
    m1 = sum(data1) / n1
    m2 = sum(data2) / n2
    s1 = math.sqrt(sum((x - m1)**2 for x in data1) / (n1 - 1))
    s2 = math.sqrt(sum((x - m2)**2 for x in data2) / (n2 - 1))

    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    p_value = 2 * (1 - phi(abs(t)))

    return {"t": t, "p_value": p_value, "diff": m1 - m2, "se": se}


def demo_tests():
    print("=" * 60)
    print("  Z-test and T-test")
    print("=" * 60)

    random.seed(42)

    # One-sample z-test
    true_mu = 52  # slightly above null
    sigma = 10
    data = [random.gauss(true_mu, sigma) for _ in range(40)]
    result = z_test(data, mu0=50, sigma=sigma)
    print(f"\n  One-sample z-test: H₀: μ = 50 vs H₁: μ ≠ 50")
    print(f"  n={result['n']}, x̄={result['x_bar']:.2f}, z={result['z']:.3f}, p={result['p_value']:.4f}")
    print(f"  Decision (α=0.05): {'Reject H₀' if result['p_value'] < 0.05 else 'Fail to reject H₀'}")

    # One-sample t-test
    result = t_test_one_sample(data, mu0=50)
    print(f"\n  One-sample t-test: H₀: μ = 50")
    print(f"  t={result['t']:.3f}, p={result['p_value']:.4f}, df={result['df']}")

    # Two-sample t-test
    group_a = [random.gauss(100, 15) for _ in range(35)]
    group_b = [random.gauss(108, 15) for _ in range(40)]
    result = t_test_two_sample(group_a, group_b)
    print(f"\n  Two-sample t-test: H₀: μ₁ = μ₂")
    print(f"  diff={result['diff']:.2f}, t={result['t']:.3f}, p={result['p_value']:.4f}")


# ─────────────────────────────────────────────────
# 2. CHI-SQUARED GOODNESS OF FIT
# ─────────────────────────────────────────────────

def chi_squared_gof(observed: list[int],
                     expected: list[float]) -> dict:
    """Chi-squared goodness-of-fit test."""
    chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1

    # Approximate p-value using Wilson-Hilferty
    if df > 0:
        z = ((chi2 / df)**(1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
        p_value = 1 - phi(z)
    else:
        p_value = 1.0

    return {"chi2": chi2, "df": df, "p_value": p_value}


def demo_chi_squared():
    print("\n" + "=" * 60)
    print("  Chi-squared Goodness of Fit")
    print("=" * 60)

    # Fair die test
    observed = [18, 15, 23, 20, 14, 10]
    n = sum(observed)
    expected = [n / 6] * 6

    result = chi_squared_gof(observed, expected)
    print(f"\n  Die fairness test (n={n}):")
    print(f"  Observed: {observed}")
    print(f"  Expected: {[f'{e:.1f}' for e in expected]}")
    print(f"  χ² = {result['chi2']:.3f}, df={result['df']}, p={result['p_value']:.4f}")
    print(f"  Decision: {'Reject (not fair)' if result['p_value'] < 0.05 else 'Fail to reject (fair)'}")

    # Hardy-Weinberg test
    print(f"\n  Hardy-Weinberg equilibrium test:")
    # Genotypes: AA=120, Aa=200, aa=80, total=400
    # Under HW: p=allele freq of A
    obs = [120, 200, 80]
    n_total = sum(obs)
    p_a = (2*obs[0] + obs[1]) / (2*n_total)
    exp_hw = [n_total * p_a**2, n_total * 2*p_a*(1-p_a), n_total * (1-p_a)**2]
    result = chi_squared_gof(obs, exp_hw)
    print(f"  Observed: AA={obs[0]}, Aa={obs[1]}, aa={obs[2]}")
    print(f"  Expected: AA={exp_hw[0]:.1f}, Aa={exp_hw[1]:.1f}, aa={exp_hw[2]:.1f}")
    print(f"  χ² = {result['chi2']:.3f}, p={result['p_value']:.4f}")


# ─────────────────────────────────────────────────
# 3. POWER ANALYSIS
# ─────────────────────────────────────────────────

def power_z_test(n: int, mu0: float, mu1: float,
                  sigma: float, alpha: float = 0.05) -> float:
    """Power of one-sided z-test."""
    z_alpha = z_quantile(1 - alpha)
    delta = (mu1 - mu0) / (sigma / math.sqrt(n))
    power = 1 - phi(z_alpha - delta)
    return power


def sample_size_for_power(mu0: float, mu1: float, sigma: float,
                           alpha: float = 0.05,
                           target_power: float = 0.80) -> int:
    """Find n for desired power (one-sided z-test)."""
    z_a = z_quantile(1 - alpha)
    z_b = z_quantile(target_power)
    n = ((z_a + z_b) * sigma / (mu1 - mu0)) ** 2
    return math.ceil(n)


def demo_power():
    print("\n" + "=" * 60)
    print("  Power Analysis")
    print("=" * 60)

    mu0, sigma = 50, 10

    print(f"\n  Power vs sample size (H₀: μ={mu0}, σ={sigma}, α=0.05):")
    print(f"  {'n':>6} {'μ₁=52':>8} {'μ₁=55':>8} {'μ₁=60':>8}")
    print(f"  {'─' * 32}")
    for n in [10, 20, 30, 50, 100, 200]:
        powers = [power_z_test(n, mu0, mu1, sigma) for mu1 in [52, 55, 60]]
        print(f"  {n:>6} {powers[0]:>8.3f} {powers[1]:>8.3f} {powers[2]:>8.3f}")

    # Sample size for 80% power
    print(f"\n  Sample size for 80% power:")
    for mu1 in [52, 53, 55, 60]:
        n = sample_size_for_power(mu0, mu1, sigma)
        print(f"  μ₁={mu1}: n={n}")


# ─────────────────────────────────────────────────
# 4. MULTIPLE TESTING CORRECTION
# ─────────────────────────────────────────────────

def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Bonferroni correction: reject if p < α/m."""
    m = len(p_values)
    return [p < alpha / m for p in p_values]


def benjamini_hochberg(p_values: list[float],
                        alpha: float = 0.05) -> list[bool]:
    """BH procedure for FDR control."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * m

    max_k = 0
    for k, (orig_idx, p) in enumerate(indexed, 1):
        if p <= k / m * alpha:
            max_k = k

    for k, (orig_idx, p) in enumerate(indexed, 1):
        if k <= max_k:
            results[orig_idx] = True

    return results


def demo_multiple_testing():
    print("\n" + "=" * 60)
    print("  Multiple Testing Correction")
    print("=" * 60)

    random.seed(42)

    # Simulate 20 tests: 3 truly different, 17 null
    p_values = []
    true_effects = []
    for i in range(20):
        if i < 3:  # true effect
            data = [random.gauss(2, 1) for _ in range(30)]
            true_effects.append(True)
        else:  # null
            data = [random.gauss(0, 1) for _ in range(30)]
            true_effects.append(False)
        x_bar = sum(data) / len(data)
        z = x_bar / (1 / math.sqrt(30))
        p = 2 * (1 - phi(abs(z)))
        p_values.append(p)

    naive = [p < 0.05 for p in p_values]
    bonf = bonferroni(p_values)
    bh = benjamini_hochberg(p_values)

    print(f"\n  20 tests (3 true effects, 17 null):")
    print(f"  {'Method':<15} {'Reject':>7} {'True+':>7} {'False+':>7}")
    print(f"  {'─' * 38}")
    for name, results in [("No correction", naive),
                          ("Bonferroni", bonf),
                          ("BH (FDR)", bh)]:
        tp = sum(1 for r, t in zip(results, true_effects) if r and t)
        fp = sum(1 for r, t in zip(results, true_effects) if r and not t)
        total = sum(results)
        print(f"  {name:<15} {total:>7} {tp:>7} {fp:>7}")


if __name__ == "__main__":
    demo_tests()
    demo_chi_squared()
    demo_power()
    demo_multiple_testing()

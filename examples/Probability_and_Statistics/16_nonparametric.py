"""
Nonparametric Methods

Demonstrates:
1. Sign Test and Wilcoxon Signed-Rank Test
2. Mann-Whitney U Test
3. Permutation Test
4. Kernel Density Estimation (KDE)

Theory:
- Nonparametric tests don't assume a specific distribution
- Sign test: counts signs of (xi - m₀)
- Wilcoxon: uses ranks of |xi - m₀|
- Permutation test: exact p-value by enumerating permutations

Adapted from Probability and Statistics Lesson 16.
"""

import math
import random
from itertools import combinations


# ─────────────────────────────────────────────────
# 1. SIGN TEST AND WILCOXON SIGNED-RANK
# ─────────────────────────────────────────────────

def sign_test(data: list[float], m0: float = 0) -> dict:
    """Sign test for median = m0."""
    diffs = [x - m0 for x in data if x != m0]
    n = len(diffs)
    n_plus = sum(1 for d in diffs if d > 0)
    n_minus = n - n_plus

    # P-value via binomial (two-sided)
    # P(X ≥ max(n+,n-)) where X ~ Binomial(n, 0.5)
    k = max(n_plus, n_minus)
    p_value = 0
    for i in range(k, n + 1):
        p_value += math.comb(n, i) * 0.5 ** n
    p_value *= 2  # two-sided
    p_value = min(p_value, 1.0)

    return {"n_plus": n_plus, "n_minus": n_minus, "n": n, "p_value": p_value}


def wilcoxon_signed_rank(data: list[float], m0: float = 0) -> dict:
    """Wilcoxon signed-rank test for median = m0."""
    diffs = [(x - m0, i) for i, x in enumerate(data) if x != m0]
    n = len(diffs)

    # Rank by absolute value
    ranked = sorted(diffs, key=lambda d: abs(d[0]))
    ranks = {}
    for rank, (d, idx) in enumerate(ranked, 1):
        ranks[idx] = rank

    w_plus = sum(ranks[idx] for d, idx in diffs if d > 0)
    w_minus = sum(ranks[idx] for d, idx in diffs if d < 0)
    w = min(w_plus, w_minus)

    # Normal approximation for p-value (n ≥ 10)
    mu_w = n * (n + 1) / 4
    sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu_w) / sigma_w if sigma_w > 0 else 0
    p_value = 2 * (0.5 * (1 + math.erf(z / math.sqrt(2))))

    return {"W+": w_plus, "W-": w_minus, "W": w, "z": z, "p_value": p_value}


def demo_sign_wilcoxon():
    print("=" * 60)
    print("  Sign Test and Wilcoxon Signed-Rank Test")
    print("=" * 60)

    # Test if median weight loss > 0
    random.seed(42)
    weight_loss = [2.1, -0.5, 3.4, 1.2, 0.8, -1.1, 2.5, 1.8, 0.3, 4.2,
                   -0.2, 1.5, 2.8, 0.6, 3.1]

    sign_result = sign_test(weight_loss, m0=0)
    wilc_result = wilcoxon_signed_rank(weight_loss, m0=0)

    print(f"\n  Weight loss data (n={len(weight_loss)}):")
    print(f"  Median = {sorted(weight_loss)[len(weight_loss)//2]:.1f}")

    print(f"\n  Sign test (H₀: median = 0):")
    print(f"  n+ = {sign_result['n_plus']}, n- = {sign_result['n_minus']}")
    print(f"  p-value = {sign_result['p_value']:.4f}")

    print(f"\n  Wilcoxon signed-rank test:")
    print(f"  W+ = {wilc_result['W+']:.0f}, W- = {wilc_result['W-']:.0f}")
    print(f"  z = {wilc_result['z']:.3f}, p-value = {wilc_result['p_value']:.4f}")


# ─────────────────────────────────────────────────
# 2. MANN-WHITNEY U TEST
# ─────────────────────────────────────────────────

def mann_whitney_u(group1: list[float],
                    group2: list[float]) -> dict:
    """Mann-Whitney U test for equal distributions."""
    n1, n2 = len(group1), len(group2)

    # Combine and rank
    combined = [(x, 0) for x in group1] + [(x, 1) for x in group2]
    combined.sort(key=lambda t: t[0])

    rank_sum = [0, 0]
    for rank, (val, group) in enumerate(combined, 1):
        rank_sum[group] += rank

    u1 = rank_sum[0] - n1 * (n1 + 1) / 2
    u2 = rank_sum[1] - n2 * (n2 + 1) / 2
    u = min(u1, u2)

    # Normal approximation
    mu_u = n1 * n2 / 2
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu_u) / sigma_u if sigma_u > 0 else 0
    p_value = 2 * (0.5 * (1 + math.erf(z / math.sqrt(2))))

    return {"U1": u1, "U2": u2, "U": u, "z": z, "p_value": p_value}


def demo_mann_whitney():
    print("\n" + "=" * 60)
    print("  Mann-Whitney U Test")
    print("=" * 60)

    # Treatment vs control
    treatment = [85, 92, 78, 88, 95, 82, 90, 87, 93, 86]
    control = [72, 80, 68, 75, 82, 70, 78, 74, 77, 71]

    result = mann_whitney_u(treatment, control)
    print(f"\n  Treatment: {treatment}")
    print(f"  Control:   {control}")
    print(f"\n  U₁={result['U1']:.0f}, U₂={result['U2']:.0f}")
    print(f"  z = {result['z']:.3f}, p-value = {result['p_value']:.4f}")
    print(f"  Decision: {'Reject H₀' if result['p_value'] < 0.05 else 'Fail to reject'} (α=0.05)")


# ─────────────────────────────────────────────────
# 3. PERMUTATION TEST
# ─────────────────────────────────────────────────

def permutation_test(group1: list[float], group2: list[float],
                      n_perms: int = 10000) -> dict:
    """Two-sample permutation test for difference in means."""
    observed_diff = (sum(group1) / len(group1) -
                     sum(group2) / len(group2))
    combined = group1 + group2
    n1 = len(group1)

    count_extreme = 0
    for _ in range(n_perms):
        random.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        perm_diff = sum(perm_g1) / n1 - sum(perm_g2) / len(perm_g2)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_perms
    return {"observed_diff": observed_diff, "p_value": p_value}


def demo_permutation():
    print("\n" + "=" * 60)
    print("  Permutation Test")
    print("=" * 60)

    random.seed(42)
    group_a = [23, 28, 31, 25, 27, 30, 29, 26]
    group_b = [19, 22, 20, 24, 18, 21, 23, 17]

    result = permutation_test(group_a, group_b, n_perms=10000)
    print(f"\n  Group A: {group_a}")
    print(f"  Group B: {group_b}")
    print(f"  Observed diff = {result['observed_diff']:.2f}")
    print(f"  Permutation p-value = {result['p_value']:.4f}")
    print(f"  (10000 permutations)")

    # Compare with t-test approximation
    n1, n2 = len(group_a), len(group_b)
    m1, m2 = sum(group_a)/n1, sum(group_b)/n2
    s1 = math.sqrt(sum((x-m1)**2 for x in group_a)/(n1-1))
    s2 = math.sqrt(sum((x-m2)**2 for x in group_b)/(n2-1))
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    t = (m1 - m2) / se
    p_t = 2 * (1 - 0.5*(1 + math.erf(abs(t)/math.sqrt(2))))
    print(f"\n  Welch t-test: t={t:.3f}, p≈{p_t:.4f}")
    print(f"  (Permutation test is more robust — no normality assumed)")


# ─────────────────────────────────────────────────
# 4. KERNEL DENSITY ESTIMATION
# ─────────────────────────────────────────────────

def gaussian_kernel(x: float) -> float:
    """Standard Gaussian kernel."""
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def kde(data: list[float], x: float, bandwidth: float) -> float:
    """KDE estimate at point x."""
    n = len(data)
    return sum(gaussian_kernel((x - xi) / bandwidth) for xi in data) / (n * bandwidth)


def silverman_bandwidth(data: list[float]) -> float:
    """Silverman's rule of thumb: h = 1.06·σ̂·n^{-1/5}."""
    n = len(data)
    mean = sum(data) / n
    std = math.sqrt(sum((x - mean)**2 for x in data) / n)
    return 1.06 * std * n ** (-0.2)


def demo_kde():
    print("\n" + "=" * 60)
    print("  Kernel Density Estimation")
    print("=" * 60)

    random.seed(42)
    # Bimodal data
    data = ([random.gauss(2, 0.8) for _ in range(100)] +
            [random.gauss(6, 1.2) for _ in range(100)])

    h = silverman_bandwidth(data)
    print(f"\n  Bimodal data (n={len(data)})")
    print(f"  Silverman bandwidth: h = {h:.3f}")

    # Compare bandwidths
    print(f"\n  KDE with different bandwidths:")
    x_range = [i * 0.5 for i in range(-2, 22)]

    for bw, label in [(0.3, "h=0.3 (under)"), (h, f"h={h:.2f} (Silverman)"),
                       (2.0, "h=2.0 (over)")]:
        print(f"\n  {label}:")
        max_density = max(kde(data, x, bw) for x in x_range)
        for x in x_range:
            density = kde(data, x, bw)
            bar_len = int(density / max_density * 30) if max_density > 0 else 0
            print(f"  {x:>5.1f} {'█' * bar_len}")


if __name__ == "__main__":
    demo_sign_wilcoxon()
    demo_mann_whitney()
    demo_permutation()
    demo_kde()

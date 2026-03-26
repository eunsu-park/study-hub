"""
Probability and Statistics — Nonparametric Methods
Exercises covering sign test, Wilcoxon signed-rank test, permutation test,
and kernel density estimation.
"""
import math
import random
from typing import List, Tuple


# === Exercise 1: Sign Test for Median ===
def exercise_1() -> None:
    """Implement a sign test to test whether the median of a population
    equals a hypothesized value, using the binomial distribution for
    the exact p-value."""
    print("=== Exercise 1: Sign Test for Median ===")

    # Data: weight changes after a diet program (kg)
    data = [2.3, -0.5, 1.8, 3.1, 0.7, -1.2, 2.5, 0.9, 1.4, -0.3,
            3.7, 1.1, 0.2, 2.8, -0.8]
    m0 = 0.0  # H0: median = 0

    # Remove zeros and count signs
    diffs = [x - m0 for x in data if x != m0]
    n = len(diffs)
    n_plus = sum(1 for d in diffs if d > 0)
    n_minus = n - n_plus

    # Two-sided exact p-value using Binomial(n, 0.5)
    k = max(n_plus, n_minus)
    p_value = 0.0
    for i in range(k, n + 1):
        p_value += math.comb(n, i) * 0.5 ** n
    p_value *= 2  # two-sided
    p_value = min(p_value, 1.0)

    print(f"\n  Data: {data}")
    print(f"  H0: median = {m0}  vs  H1: median != {m0}")
    print(f"\n  Non-zero differences: n = {n}")
    print(f"  Positive signs (n+) = {n_plus}")
    print(f"  Negative signs (n-) = {n_minus}")
    print(f"\n  Under H0, n+ ~ Binomial({n}, 0.5)")
    print(f"  P(X >= {k}) = sum_{{i={k}}}^{{{n}}} C({n},i) * 0.5^{n}")
    print(f"  Two-sided p-value = {p_value:.4f}")

    alpha = 0.05
    print(f"\n  Decision (alpha={alpha}): "
          f"{'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
    if p_value < alpha:
        print(f"  Evidence that the median weight change differs from {m0}.")

    data_sorted = sorted(data)
    median = data_sorted[len(data_sorted) // 2]
    print(f"\n  Sample median = {median:.1f}\n")


# === Exercise 2: Wilcoxon Signed-Rank Test ===
def exercise_2() -> None:
    """Implement the Wilcoxon signed-rank test for paired data, using
    signed ranks and a normal approximation for the p-value."""
    print("=== Exercise 2: Wilcoxon Signed-Rank Test ===")

    # Paired data: before and after measurements
    before = [125, 130, 118, 140, 135, 128, 122, 132, 137, 126,
              129, 133, 121, 138, 131]
    after = [120, 126, 115, 132, 130, 125, 118, 128, 130, 123,
             125, 129, 117, 133, 127]
    diffs = [a - b for a, b in zip(after, before)]

    print(f"\n  Paired data: n={len(before)} subjects")
    print(f"  H0: median difference = 0 (no treatment effect)")

    # Remove zeros
    nonzero = [(d, i) for i, d in enumerate(diffs) if d != 0]
    n = len(nonzero)

    # Rank by absolute value
    ranked = sorted(nonzero, key=lambda x: abs(x[0]))
    signed_ranks: List[Tuple[int, int, float]] = []
    for rank, (d, idx) in enumerate(ranked, 1):
        sign = 1 if d > 0 else -1
        signed_ranks.append((rank, sign, d))

    w_plus = sum(r for r, s, _ in signed_ranks if s > 0)
    w_minus = sum(r for r, s, _ in signed_ranks if s < 0)
    w = min(w_plus, w_minus)

    print(f"\n  {'Rank':>6} {'|Diff|':>8} {'Sign':>6} {'Signed Rank':>12}")
    print(f"  {'----':>6} {'------':>8} {'----':>6} {'-----------':>12}")
    for rank, sign, d in signed_ranks:
        print(f"  {rank:>6} {abs(d):>8.1f} {'  +' if sign > 0 else '  -':>6} "
              f"{rank * sign:>12}")

    print(f"\n  W+ = {w_plus}")
    print(f"  W- = {w_minus}")
    print(f"  W = min(W+, W-) = {w}")

    # Normal approximation
    mu_w = n * (n + 1) / 4
    sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu_w) / sigma_w if sigma_w > 0 else 0
    # p-value: W is small when H1 is true
    p_value = 2 * (0.5 * (1 + math.erf(z / math.sqrt(2))))
    p_value = min(p_value, 1.0)

    print(f"\n  Normal approximation:")
    print(f"    E[W] = n(n+1)/4 = {mu_w:.1f}")
    print(f"    SD[W] = sqrt(n(n+1)(2n+1)/24) = {sigma_w:.3f}")
    print(f"    z = (W - E[W]) / SD[W] = {z:.4f}")
    print(f"    p-value (approx) = {p_value:.4f}")

    alpha = 0.05
    print(f"\n  Decision (alpha={alpha}): "
          f"{'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
    print()


# === Exercise 3: Permutation Test ===
def exercise_3() -> None:
    """Implement a two-sample permutation test for the difference in means,
    comparing results with a parametric t-test approximation."""
    print("=== Exercise 3: Permutation Test (Two-Sample) ===")

    random.seed(42)

    group_a = [23, 28, 31, 25, 27, 30, 29, 26]
    group_b = [19, 22, 20, 24, 18, 21, 23, 17]

    n1 = len(group_a)
    n2 = len(group_b)
    obs_diff = sum(group_a) / n1 - sum(group_b) / n2

    print(f"\n  Group A: {group_a}")
    print(f"  Group B: {group_b}")
    print(f"  Observed difference in means = {obs_diff:.4f}")
    print(f"\n  H0: The two groups come from the same distribution")

    # Permutation test
    n_perms = 10000
    combined = group_a + group_b
    count_extreme = 0

    for _ in range(n_perms):
        random.shuffle(combined)
        perm_a = combined[:n1]
        perm_b = combined[n1:]
        perm_diff = sum(perm_a) / n1 - sum(perm_b) / n2
        if abs(perm_diff) >= abs(obs_diff):
            count_extreme += 1

    p_perm = count_extreme / n_perms

    print(f"\n  Permutation test ({n_perms:,} permutations):")
    print(f"    P(|diff| >= {abs(obs_diff):.4f}) = {p_perm:.4f}")

    # Parametric t-test for comparison
    m1 = sum(group_a) / n1
    m2 = sum(group_b) / n2
    s1 = math.sqrt(sum((x - m1) ** 2 for x in group_a) / (n1 - 1))
    s2 = math.sqrt(sum((x - m2) ** 2 for x in group_b) / (n2 - 1))
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    t_stat = (m1 - m2) / se
    p_t = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    print(f"\n  Welch t-test comparison:")
    print(f"    t = {t_stat:.4f}, p (approx) = {p_t:.4f}")

    alpha = 0.05
    print(f"\n  Decision (alpha={alpha}):")
    print(f"    Permutation: {'Reject H0' if p_perm < alpha else 'Fail to reject H0'}")
    print(f"    T-test:      {'Reject H0' if p_t < alpha else 'Fail to reject H0'}")
    print(f"\n  Permutation test makes no distributional assumptions.\n")


# === Exercise 4: Kernel Density Estimation ===
def exercise_4() -> None:
    """Implement Gaussian kernel density estimation with Silverman's
    bandwidth rule and compare different bandwidth choices."""
    print("=== Exercise 4: Kernel Density Estimation ===")

    random.seed(42)
    # Bimodal distribution: mixture of two normals
    data: List[float] = []
    for _ in range(80):
        data.append(random.gauss(2.0, 0.8))
    for _ in range(80):
        data.append(random.gauss(6.0, 1.0))

    n = len(data)
    mean = sum(data) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in data) / n)

    def gaussian_kernel(u: float) -> float:
        return math.exp(-0.5 * u ** 2) / math.sqrt(2 * math.pi)

    def kde(x: float, bandwidth: float) -> float:
        return sum(gaussian_kernel((x - xi) / bandwidth)
                   for xi in data) / (n * bandwidth)

    # Silverman's rule of thumb
    h_silverman = 1.06 * std * n ** (-0.2)

    print(f"\n  Bimodal data: n={n}, mean={mean:.2f}, std={std:.2f}")
    print(f"  Silverman bandwidth: h = 1.06 * {std:.2f} * {n}^(-0.2) "
          f"= {h_silverman:.3f}")

    x_range = [0.5 * i for i in range(-2, 20)]

    bandwidths = [
        (0.3, "h=0.3 (undersmoothed)"),
        (h_silverman, f"h={h_silverman:.2f} (Silverman)"),
        (2.0, "h=2.0 (oversmoothed)"),
    ]

    for bw, label in bandwidths:
        print(f"\n  {label}:")
        densities = [kde(x, bw) for x in x_range]
        max_d = max(densities) if densities else 1
        for x, d in zip(x_range, densities):
            bar_len = int(d / max_d * 30) if max_d > 0 else 0
            bar = "#" * bar_len
            print(f"    {x:>5.1f} | {bar}")

    print(f"\n  Observations:")
    print(f"  - Undersmoothed (h=0.3): noisy, shows spurious peaks")
    print(f"  - Silverman: reasonable trade-off")
    print(f"  - Oversmoothed (h=2.0): misses the bimodal structure\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

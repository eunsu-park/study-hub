"""
Probability and Statistics — Hypothesis Testing
Exercises covering z-test statistic and p-value, t-test for means,
chi-squared goodness-of-fit, and power analysis.
"""
import math
import random
from typing import Dict, List, Tuple


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


# === Exercise 1: Z-Test Statistic and P-Value ===
def exercise_1() -> None:
    """Perform a one-sample z-test (sigma known) with two-sided and
    one-sided alternatives, computing test statistics and p-values."""
    print("=== Exercise 1: Z-Test Statistic and P-Value ===")

    # Scenario: A manufacturer claims mean weight = 500g, sigma = 12g.
    # Sample of n=36 has x_bar = 496.5g.
    mu_0 = 500.0
    sigma = 12.0
    n = 36
    x_bar = 496.5

    z_stat = (x_bar - mu_0) / (sigma / math.sqrt(n))

    # Two-sided p-value
    p_two = 2 * (1 - phi(abs(z_stat)))
    # One-sided (less than) p-value
    p_less = phi(z_stat)
    # One-sided (greater than) p-value
    p_greater = 1 - phi(z_stat)

    print(f"\n  Scenario: Manufacturer claims mu = {mu_0}g, sigma = {sigma}g")
    print(f"  Sample: n={n}, x_bar={x_bar}g")
    print(f"\n  Test statistic:")
    print(f"    z = (x_bar - mu_0) / (sigma/sqrt(n))")
    print(f"      = ({x_bar} - {mu_0}) / ({sigma}/sqrt({n}))")
    print(f"      = {x_bar - mu_0} / {sigma / math.sqrt(n):.4f}")
    print(f"      = {z_stat:.4f}")

    alpha = 0.05
    print(f"\n  P-values (alpha = {alpha}):")
    print(f"    Two-sided  (H1: mu != {mu_0}): p = {p_two:.4f} "
          f"{'-> Reject' if p_two < alpha else '-> Fail to reject'}")
    print(f"    Left-sided (H1: mu < {mu_0}):  p = {p_less:.4f} "
          f"{'-> Reject' if p_less < alpha else '-> Fail to reject'}")
    print(f"    Right-sided(H1: mu > {mu_0}):  p = {p_greater:.4f} "
          f"{'-> Reject' if p_greater < alpha else '-> Fail to reject'}")

    # Simulation verification of p-value
    random.seed(42)
    n_sims = 50000
    extreme_count = 0
    for _ in range(n_sims):
        sample = [random.gauss(mu_0, sigma) for _ in range(n)]
        s_bar = sum(sample) / n
        z_sim = (s_bar - mu_0) / (sigma / math.sqrt(n))
        if abs(z_sim) >= abs(z_stat):
            extreme_count += 1

    p_simulated = extreme_count / n_sims
    print(f"\n  Simulated two-sided p-value: {p_simulated:.4f} "
          f"(analytic: {p_two:.4f})\n")


# === Exercise 2: T-Test for Comparing Two Means ===
def exercise_2() -> None:
    """Perform a two-sample Welch's t-test to compare means of two groups."""
    print("=== Exercise 2: T-Test for Comparing Two Means ===")

    # Drug trial: control vs treatment
    random.seed(42)
    control = [random.gauss(120, 12) for _ in range(20)]
    treatment = [random.gauss(112, 10) for _ in range(22)]

    n1 = len(control)
    n2 = len(treatment)
    m1 = sum(control) / n1
    m2 = sum(treatment) / n2
    s1_sq = sum((x - m1) ** 2 for x in control) / (n1 - 1)
    s2_sq = sum((x - m2) ** 2 for x in treatment) / (n2 - 1)
    s1 = math.sqrt(s1_sq)
    s2 = math.sqrt(s2_sq)

    # Welch's t-statistic
    se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    denom = (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
    df = num / denom

    # Approximate p-value using normal approximation (good for df >= 30)
    p_value = 2 * (1 - phi(abs(t_stat)))

    print(f"\n  Drug trial: control (n={n1}) vs treatment (n={n2})")
    print(f"  Control:   mean={m1:.2f}, s={s1:.2f}")
    print(f"  Treatment: mean={m2:.2f}, s={s2:.2f}")
    print(f"\n  Welch's t-statistic:")
    print(f"    t = (x_bar1 - x_bar2) / sqrt(s1^2/n1 + s2^2/n2)")
    print(f"      = ({m1:.2f} - {m2:.2f}) / {se:.4f}")
    print(f"      = {t_stat:.4f}")
    print(f"    df (Welch-Satterthwaite) = {df:.2f}")
    print(f"    p-value (approx) = {p_value:.4f}")

    alpha = 0.05
    print(f"\n  Decision (alpha={alpha}): "
          f"{'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
    if p_value < alpha:
        print(f"  The treatment group has a significantly different mean.")
    print()


# === Exercise 3: Chi-Squared Goodness-of-Fit ===
def exercise_3() -> None:
    """Test whether observed letter frequencies in a text match a known
    English letter distribution using the chi-squared test."""
    print("=== Exercise 3: Chi-Squared Goodness-of-Fit ===")

    # Observed letter frequencies in a text sample
    observed = [82, 15, 28, 43, 127, 22, 20, 61, 70, 2,
                8, 40, 24, 67, 75, 19, 1, 60, 63, 91,
                28, 10, 24, 2, 20, 1]
    total = sum(observed)
    k = len(observed)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Expected English letter frequencies (approximate)
    english_freq = [
        0.082, 0.015, 0.028, 0.043, 0.127, 0.022, 0.020, 0.061, 0.070, 0.002,
        0.008, 0.040, 0.024, 0.067, 0.075, 0.019, 0.001, 0.060, 0.063, 0.091,
        0.028, 0.010, 0.024, 0.002, 0.020, 0.001
    ]

    expected = [total * f for f in english_freq]

    # Chi-squared statistic
    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi2 += (o - e) ** 2 / e

    df = k - 1

    # Approximate p-value using Wilson-Hilferty transformation
    if df > 0:
        z_wh = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / \
               math.sqrt(2 / (9 * df))
        p_value = 1 - phi(z_wh)
    else:
        p_value = 1.0

    print(f"\n  Testing letter frequencies against English distribution")
    print(f"  Total letters: {total}, Categories: {k}")

    # Show top contributors
    contributions = [(letters[i], observed[i], expected[i],
                      (observed[i] - expected[i]) ** 2 / expected[i] if expected[i] > 0 else 0)
                     for i in range(k)]
    contributions.sort(key=lambda x: -x[3])

    print(f"\n  Top 5 contributors to chi-squared:")
    print(f"  {'Letter':>8} {'Observed':>10} {'Expected':>10} {'(O-E)^2/E':>10}")
    print(f"  {'------':>8} {'--------':>10} {'--------':>10} {'---------':>10}")
    for letter, obs, exp, contrib in contributions[:5]:
        print(f"  {letter:>8} {obs:>10} {exp:>10.2f} {contrib:>10.4f}")

    print(f"\n  Chi-squared = {chi2:.4f}")
    print(f"  df = {df}")
    print(f"  p-value (approx) = {p_value:.4f}")

    alpha = 0.05
    print(f"\n  Decision (alpha={alpha}): "
          f"{'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
    if p_value >= alpha:
        print(f"  The text's letter frequencies are consistent with English.\n")
    else:
        print(f"  The text's letter frequencies differ from English.\n")


# === Exercise 4: Power Analysis ===
def exercise_4() -> None:
    """Compute the power of a one-sided z-test for various sample sizes
    and effect sizes, and determine required n for a target power."""
    print("=== Exercise 4: Power Analysis ===")

    mu_0 = 50.0
    sigma = 10.0
    alpha = 0.05
    z_alpha = z_quantile(1 - alpha)

    def power_z(n: int, mu_1: float) -> float:
        """Power of one-sided z-test: P(reject H0 | mu = mu_1)."""
        delta = (mu_1 - mu_0) / (sigma / math.sqrt(n))
        return 1 - phi(z_alpha - delta)

    def required_n(mu_1: float, target_power: float = 0.80) -> int:
        """Minimum n for desired power."""
        z_beta = z_quantile(target_power)
        n_exact = ((z_alpha + z_beta) * sigma / (mu_1 - mu_0)) ** 2
        return math.ceil(n_exact)

    print(f"\n  H0: mu = {mu_0}, sigma = {sigma}, alpha = {alpha}")
    print(f"  Power = P(reject H0 | H1 true)")

    # Power table
    print(f"\n  Power vs sample size:")
    print(f"  {'n':>6}  {'mu1=52':>8}  {'mu1=55':>8}  {'mu1=60':>8}")
    print(f"  {'-----':>6}  {'------':>8}  {'------':>8}  {'------':>8}")
    for n in [10, 20, 30, 50, 100, 200]:
        p1 = power_z(n, 52)
        p2 = power_z(n, 55)
        p3 = power_z(n, 60)
        print(f"  {n:>6}  {p1:>8.4f}  {p2:>8.4f}  {p3:>8.4f}")

    # Required sample size for 80% power
    print(f"\n  Required n for 80% power:")
    for mu_1 in [52, 53, 55, 60]:
        n_req = required_n(mu_1)
        actual_power = power_z(n_req, mu_1)
        print(f"    mu1={mu_1}: n={n_req} (actual power = {actual_power:.4f})")

    # Simulation verification
    random.seed(42)
    mu_1 = 55.0
    n = 30
    n_sims = 10000
    reject_count = 0
    for _ in range(n_sims):
        sample = [random.gauss(mu_1, sigma) for _ in range(n)]
        x_bar = sum(sample) / n
        z_stat = (x_bar - mu_0) / (sigma / math.sqrt(n))
        if z_stat > z_alpha:
            reject_count += 1

    sim_power = reject_count / n_sims
    th_power = power_z(n, mu_1)
    print(f"\n  Verification (mu1={mu_1}, n={n}):")
    print(f"    Theoretical power = {th_power:.4f}")
    print(f"    Simulated power   = {sim_power:.4f}\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

"""
Law of Large Numbers and Central Limit Theorem — Exercises

Topics covered:
- Monte Carlo estimation via LLN
- CLT convergence for Bernoulli averages
- Normal approximation to Binomial with continuity correction
- Sample size determination for Monte Carlo precision
"""

import math
import random
from typing import List


# === Exercise 1: Monte Carlo Estimate of Pi ===
def exercise_1() -> None:
    """Monte Carlo estimate of pi using random points in unit square."""
    print("=" * 70)
    print("Exercise 1: Monte Carlo Estimate of Pi")
    print("=" * 70)
    print()
    print("Problem:")
    print("  Generate N random points (x, y) uniformly in [0,1]x[0,1].")
    print("  Count the fraction inside the unit quarter circle (x^2 + y^2 <= 1).")
    print("  By LLN, this fraction -> pi/4, so pi_hat = 4 * fraction.")
    print("  Estimate pi for N = 100, 1000, 10000, 100000, 1000000.")
    print()

    random.seed(42)
    sample_sizes: List[int] = [100, 1000, 10000, 100000, 1000000]

    print("Solution:")
    print(f"  True pi = {math.pi:.10f}")
    print()
    print(f"  {'N':>10}  {'pi_hat':>12}  {'|Error|':>10}  {'Rel Error %':>12}")
    print(f"  {'-' * 10}  {'-' * 12}  {'-' * 10}  {'-' * 12}")

    for n in sample_sizes:
        inside: int = 0
        for _ in range(n):
            x: float = random.random()
            y: float = random.random()
            if x * x + y * y <= 1.0:
                inside += 1
        pi_hat: float = 4.0 * inside / n
        error: float = abs(pi_hat - math.pi)
        rel_error: float = error / math.pi * 100
        print(f"  {n:>10}  {pi_hat:>12.6f}  {error:>10.6f}  {rel_error:>11.4f}%")

    print()
    print("  By LLN, the estimate converges to pi as N -> infinity.")
    print("  Error decreases roughly as 1/sqrt(N) (Monte Carlo rate).")
    print()


# === Exercise 2: CLT for Bernoulli Averages ===
def exercise_2() -> None:
    """CLT simulation: average of n Bernoulli(0.3), show convergence to Normal."""
    print("=" * 70)
    print("Exercise 2: CLT for Bernoulli(0.3) Averages")
    print("=" * 70)
    print()
    print("Problem:")
    print("  Let X1,...,Xn ~ Bernoulli(0.3). The sample mean X̄n has:")
    print("  E[X̄n] = 0.3, Var(X̄n) = 0.3*0.7/n = 0.21/n.")
    print("  By CLT, Zn = (X̄n - 0.3)/sqrt(0.21/n) -> N(0,1).")
    print("  Simulate 10000 values of Zn for n = 5, 30, 100 and check normality.")
    print()

    random.seed(42)
    p: float = 0.3
    mu: float = p
    sigma_sq: float = p * (1.0 - p)
    sample_sizes: List[int] = [5, 30, 100]
    num_sims: int = 10000

    def standard_normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    print("Solution:")
    print(f"  p = {p}, mu = {mu}, sigma^2 = {sigma_sq}")
    print()

    for n in sample_sizes:
        z_values: List[float] = []
        se: float = math.sqrt(sigma_sq / n)

        for _ in range(num_sims):
            x_bar: float = sum(1 for _ in range(n) if random.random() < p) / n
            z: float = (x_bar - mu) / se
            z_values.append(z)

        z_values.sort()

        # Compute summary statistics
        z_mean: float = sum(z_values) / num_sims
        z_var: float = sum((z - z_mean) ** 2 for z in z_values) / (num_sims - 1)
        z_skew_num: float = sum((z - z_mean) ** 3 for z in z_values) / num_sims
        z_skew: float = z_skew_num / (z_var ** 1.5) if z_var > 0 else 0.0

        print(f"  n = {n}:")
        print(f"    Mean of Zn:     {z_mean:>8.4f}  (expected: 0)")
        print(f"    Variance of Zn: {z_var:>8.4f}  (expected: 1)")
        print(f"    Skewness of Zn: {z_skew:>8.4f}  (expected: 0)")

        # CDF comparison at key quantiles
        quantiles: List[float] = [-1.96, -1.0, 0.0, 1.0, 1.96]
        print(f"    {'z':>6}  {'Empirical':>10}  {'Phi(z)':>8}  {'|Diff|':>8}")
        for zp in quantiles:
            emp: float = sum(1 for zv in z_values if zv <= zp) / num_sims
            theo: float = standard_normal_cdf(zp)
            print(f"    {zp:>6.2f}  {emp:>10.4f}  {theo:>8.4f}  {abs(emp - theo):>8.4f}")
        print()

    print("  As n grows, Zn approaches N(0,1) — skewness decreases toward 0.")
    print()


# === Exercise 3: Normal Approximation to Binomial ===
def exercise_3() -> None:
    """Normal approximation to Binomial(100, 0.3) with continuity correction."""
    print("=" * 70)
    print("Exercise 3: Normal Approximation to Binomial(100, 0.3)")
    print("=" * 70)
    print()
    print("Problem:")
    print("  X ~ Binomial(100, 0.3). Compute P(25 <= X <= 35) using:")
    print("  (a) Exact binomial probability (summing PMF)")
    print("  (b) Normal approximation without continuity correction")
    print("  (c) Normal approximation with continuity correction")
    print()

    n: int = 100
    p: float = 0.3
    mu: float = n * p
    sigma: float = math.sqrt(n * p * (1.0 - p))

    def standard_normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def binom_pmf(k: int, n: int, p: float) -> float:
        """Compute binomial PMF using log to avoid overflow."""
        log_pmf: float = 0.0
        log_pmf += math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        log_pmf += k * math.log(p) + (n - k) * math.log(1.0 - p)
        return math.exp(log_pmf)

    print("Solution:")
    print(f"  n = {n}, p = {p}")
    print(f"  mu = n*p = {mu:.1f}")
    print(f"  sigma = sqrt(n*p*(1-p)) = {sigma:.4f}")
    print()

    # (a) Exact
    exact_prob: float = sum(binom_pmf(k, n, p) for k in range(25, 36))
    print(f"  (a) Exact: P(25 <= X <= 35) = {exact_prob:.6f}")

    # (b) Normal without continuity correction
    z_low: float = (25 - mu) / sigma
    z_high: float = (35 - mu) / sigma
    approx_no_cc: float = standard_normal_cdf(z_high) - standard_normal_cdf(z_low)
    print(f"  (b) Normal (no CC): P({z_low:.4f} <= Z <= {z_high:.4f}) = {approx_no_cc:.6f}")

    # (c) Normal with continuity correction
    z_low_cc: float = (24.5 - mu) / sigma
    z_high_cc: float = (35.5 - mu) / sigma
    approx_cc: float = standard_normal_cdf(z_high_cc) - standard_normal_cdf(z_low_cc)
    print(f"  (c) Normal (CC):    P({z_low_cc:.4f} <= Z <= {z_high_cc:.4f}) = {approx_cc:.6f}")

    print()
    print(f"  Error without CC: {abs(approx_no_cc - exact_prob):.6f}")
    print(f"  Error with CC:    {abs(approx_cc - exact_prob):.6f}")
    print(f"  Continuity correction improves accuracy by ~{abs(abs(approx_no_cc - exact_prob) - abs(approx_cc - exact_prob)) / abs(approx_no_cc - exact_prob) * 100:.1f}%.")
    print()


# === Exercise 4: Minimum Sample Size for Monte Carlo Error ===
def exercise_4() -> None:
    """Determine minimum sample size for Monte Carlo error < 0.01 at 95% confidence."""
    print("=" * 70)
    print("Exercise 4: Minimum Sample Size for Monte Carlo Precision")
    print("=" * 70)
    print()
    print("Problem:")
    print("  We estimate E[g(X)] by the sample mean of g(X1),...,g(Xn).")
    print("  By CLT, the 95% CI half-width is z_{0.025} * sigma / sqrt(n).")
    print("  Find min n such that half-width < 0.01, given:")
    print("  (a) sigma = 0.5  (known)")
    print("  (b) sigma = 1.0  (known)")
    print("  (c) sigma unknown — estimate via pilot sample from Uniform(0,1).")
    print()

    z_alpha_half: float = 1.96  # z_{0.025}
    target_error: float = 0.01

    print("Solution:")
    print(f"  z_{{0.025}} = {z_alpha_half}")
    print(f"  Target half-width < {target_error}")
    print(f"  Required n >= (z * sigma / epsilon)^2")
    print()

    # (a) sigma = 0.5
    sigma_a: float = 0.5
    n_a: int = math.ceil((z_alpha_half * sigma_a / target_error) ** 2)
    print(f"  (a) sigma = {sigma_a}:")
    print(f"      n >= ({z_alpha_half} * {sigma_a} / {target_error})^2 = {(z_alpha_half * sigma_a / target_error) ** 2:.1f}")
    print(f"      Minimum n = {n_a}")
    print()

    # (b) sigma = 1.0
    sigma_b: float = 1.0
    n_b: int = math.ceil((z_alpha_half * sigma_b / target_error) ** 2)
    print(f"  (b) sigma = {sigma_b}:")
    print(f"      n >= ({z_alpha_half} * {sigma_b} / {target_error})^2 = {(z_alpha_half * sigma_b / target_error) ** 2:.1f}")
    print(f"      Minimum n = {n_b}")
    print()

    # (c) Pilot sample to estimate sigma
    random.seed(42)
    pilot_size: int = 100
    pilot: List[float] = [random.random() for _ in range(pilot_size)]
    pilot_mean: float = sum(pilot) / pilot_size
    pilot_var: float = sum((x - pilot_mean) ** 2 for x in pilot) / (pilot_size - 1)
    sigma_hat: float = math.sqrt(pilot_var)
    n_c: int = math.ceil((z_alpha_half * sigma_hat / target_error) ** 2)

    # Theoretical sigma for U(0,1): sqrt(1/12) = 0.2887
    sigma_theory: float = math.sqrt(1.0 / 12.0)

    print(f"  (c) Pilot sample (n_pilot = {pilot_size}) from Uniform(0,1):")
    print(f"      Estimated sigma = {sigma_hat:.4f} (theoretical: {sigma_theory:.4f})")
    print(f"      n >= ({z_alpha_half} * {sigma_hat:.4f} / {target_error})^2 = {(z_alpha_half * sigma_hat / target_error) ** 2:.1f}")
    print(f"      Minimum n = {n_c}")
    print()
    print("  Note: Larger sigma requires larger n. Doubling sigma quadruples n.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

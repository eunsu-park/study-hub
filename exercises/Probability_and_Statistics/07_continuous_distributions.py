"""
Probability and Statistics — Continuous Distributions
Exercises covering Normal, Exponential, Gamma, and Beta distributions
with numerical computation and simulation.
"""
import math
import random
from typing import List


# === Exercise 1: Normal Distribution — P(mu-2sigma < X < mu+2sigma) ===
def exercise_1() -> None:
    """Compute P(mu - 2*sigma < X < mu + 2*sigma) for X ~ N(100, 15^2)
    using numerical integration of the PDF."""
    print("=== Exercise 1: Normal — 2-Sigma Probability ===")

    mu = 100.0
    sigma = 15.0

    def normal_pdf(x: float, mu: float, sigma: float) -> float:
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    # Integrate from mu - 2*sigma to mu + 2*sigma
    a = mu - 2 * sigma
    b = mu + 2 * sigma
    n_steps = 100_000
    h = (b - a) / n_steps

    integral = 0.5 * (normal_pdf(a, mu, sigma) + normal_pdf(b, mu, sigma))
    for i in range(1, n_steps):
        x = a + i * h
        integral += normal_pdf(x, mu, sigma)
    integral *= h

    # Expected: erf(2/sqrt(2)) = erf(sqrt(2))
    expected = math.erf(2 / math.sqrt(2))

    print(f"  X ~ N(mu={mu}, sigma={sigma})")
    print(f"  Interval: [{a:.0f}, {b:.0f}]  (mu +/- 2*sigma)")
    print(f"  P({a:.0f} < X < {b:.0f}):")
    print(f"    Numerical integration: {integral:.8f}")
    print(f"    Via erf:               {expected:.8f}")
    print(f"    Difference:            {abs(integral - expected):.2e}")
    print(f"  Approximately {integral * 100:.2f}% of data falls within 2 sigma.\n")


# === Exercise 2: Exponential — Memoryless Property ===
def exercise_2() -> None:
    """For X ~ Exponential(lambda=0.5), compute P(X > 3 | X > 1) and
    verify it equals P(X > 2), demonstrating the memoryless property."""
    print("=== Exercise 2: Exponential — Memoryless Property ===")

    lam = 0.5

    # P(X > t) = e^{-lambda*t}
    def survival(t: float) -> float:
        return math.exp(-lam * t)

    s = 1.0
    t = 2.0

    # P(X > s+t | X > s) = P(X > s+t) / P(X > s) = e^{-lam*(s+t)} / e^{-lam*s}
    p_conditional = survival(s + t) / survival(s)
    p_direct = survival(t)

    print(f"  X ~ Exponential(lambda={lam})")
    print(f"  P(X > {s+t:.0f}) = e^(-{lam}*{s+t:.0f}) = {survival(s+t):.8f}")
    print(f"  P(X > {s:.0f}) = e^(-{lam}*{s:.0f}) = {survival(s):.8f}")
    print(f"  P(X > {s+t:.0f} | X > {s:.0f}) = P(X>{s+t:.0f})/P(X>{s:.0f}) "
          f"= {p_conditional:.8f}")
    print(f"  P(X > {t:.0f})              = {p_direct:.8f}")
    print(f"  Difference: {abs(p_conditional - p_direct):.2e}")
    print(f"  Memoryless property verified: "
          f"P(X > s+t | X > s) = P(X > t).")

    # Simulation verification
    n = 200_000
    random.seed(55)
    samples = [-math.log(1 - random.random()) / lam for _ in range(n)]

    count_gt_s = sum(1 for x in samples if x > s)
    count_gt_s_plus_t = sum(1 for x in samples if x > s + t)
    count_gt_t = sum(1 for x in samples if x > t)

    p_cond_sim = count_gt_s_plus_t / count_gt_s if count_gt_s > 0 else 0
    p_direct_sim = count_gt_t / n

    print(f"  Simulation ({n:,} samples):")
    print(f"    P(X>{s+t:.0f}|X>{s:.0f}) = {p_cond_sim:.6f},  "
          f"P(X>{t:.0f}) = {p_direct_sim:.6f}\n")


# === Exercise 3: Gamma Distribution via Sum of Exponentials ===
def exercise_3() -> None:
    """Generate Gamma(alpha=5, beta=1) samples as the sum of 5 independent
    Exponential(1) random variables.  Compute and verify the moments."""
    print("=== Exercise 3: Gamma(5,1) as Sum of 5 Exponentials ===")

    alpha = 5
    beta = 1.0  # rate parameter
    n = 100_000
    random.seed(88)

    samples: List[float] = []
    for _ in range(n):
        # Sum of alpha independent Exp(beta) gives Gamma(alpha, beta)
        gamma_sample = sum(-math.log(1 - random.random()) / beta
                          for _ in range(alpha))
        samples.append(gamma_sample)

    # Theoretical moments for Gamma(alpha, beta) with rate parameterization
    # Mean = alpha/beta, Var = alpha/beta^2
    mean_theory = alpha / beta
    var_theory = alpha / beta**2

    mean_emp = sum(samples) / n
    var_emp = sum((x - mean_emp) ** 2 for x in samples) / (n - 1)

    print(f"  Gamma(alpha={alpha}, beta={beta}) via sum of {alpha} Exp({beta}) r.v.s")
    print(f"  Generated {n:,} samples.")
    print(f"  Theoretical: E[X] = {mean_theory:.4f}, Var(X) = {var_theory:.4f}")
    print(f"  Empirical:   E[X] = {mean_emp:.4f}, Var(X) = {var_emp:.4f}")

    # Also check the shape via percentiles
    sorted_samples = sorted(samples)
    p25 = sorted_samples[int(0.25 * n)]
    p50 = sorted_samples[int(0.50 * n)]
    p75 = sorted_samples[int(0.75 * n)]
    print(f"  Empirical quartiles: Q1={p25:.3f}, Q2(median)={p50:.3f}, Q3={p75:.3f}")

    # Skewness for Gamma = 2/sqrt(alpha)
    skew_theory = 2 / math.sqrt(alpha)
    m3 = sum((x - mean_emp) ** 3 for x in samples) / n
    std_emp = math.sqrt(var_emp)
    skew_emp = m3 / std_emp**3
    print(f"  Theoretical skewness = 2/sqrt({alpha}) = {skew_theory:.4f}")
    print(f"  Empirical skewness   = {skew_emp:.4f}\n")


# === Exercise 4: Beta Distribution — Mean and Variance ===
def exercise_4() -> None:
    """Compute the mean and variance of Beta(3, 7) analytically and verify
    by simulation.  Generate Beta samples using the Gamma-Beta connection."""
    print("=== Exercise 4: Beta(3, 7) — Mean and Variance ===")

    a, b = 3.0, 7.0
    n = 200_000
    random.seed(33)

    # Analytical formulas
    mean_theory = a / (a + b)
    var_theory = (a * b) / ((a + b) ** 2 * (a + b + 1))

    print(f"  Beta(a={a:.0f}, b={b:.0f})")
    print(f"  Theoretical: E[X] = a/(a+b) = {mean_theory:.6f}")
    print(f"  Theoretical: Var(X) = ab/((a+b)^2(a+b+1)) = {var_theory:.6f}")

    # Simulate using Gamma-Beta relation:
    # If Y1 ~ Gamma(a,1), Y2 ~ Gamma(b,1), then Y1/(Y1+Y2) ~ Beta(a,b)
    def gamma_sample(shape: float) -> float:
        """Generate Gamma(shape, 1) by summing Exp(1) for integer shape."""
        return sum(-math.log(1 - random.random()) for _ in range(int(shape)))

    samples: List[float] = []
    for _ in range(n):
        y1 = gamma_sample(a)
        y2 = gamma_sample(b)
        samples.append(y1 / (y1 + y2))

    mean_emp = sum(samples) / n
    var_emp = sum((x - mean_emp) ** 2 for x in samples) / (n - 1)

    print(f"\n  Simulated {n:,} samples via Gamma-Beta connection.")
    print(f"  Empirical:   E[X] = {mean_emp:.6f}")
    print(f"  Empirical:   Var(X) = {var_emp:.6f}")
    print(f"  Mean diff:   {abs(mean_theory - mean_emp):.6f}")
    print(f"  Var diff:    {abs(var_theory - var_emp):.6f}")

    # Mode for Beta(a,b) when a,b > 1: (a-1)/(a+b-2)
    mode_theory = (a - 1) / (a + b - 2)
    print(f"\n  Mode = (a-1)/(a+b-2) = {mode_theory:.6f}")

    # Empirical mode via histogram binning
    n_bins = 50
    bin_width = 1.0 / n_bins
    bins = [0] * n_bins
    for x in samples:
        idx = min(int(x / bin_width), n_bins - 1)
        bins[idx] += 1
    mode_bin = max(range(n_bins), key=lambda i: bins[i])
    mode_emp = (mode_bin + 0.5) * bin_width
    print(f"  Empirical mode (binned) ~ {mode_emp:.4f}\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

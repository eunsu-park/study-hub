"""
Probability and Statistics — Random Variables
Exercises covering PMF, CDF, inverse CDF transform, numerical integration
for continuous distributions, and quantiles of discrete distributions.
"""
import math
import random
from typing import Dict, List, Tuple


# === Exercise 1: PMF and CDF for Sum of Two Dice ===
def exercise_1() -> None:
    """Build the PMF for the sum of two fair six-sided dice.  Then compute
    the CDF and display both."""
    print("=== Exercise 1: PMF and CDF for Sum of Two Dice ===")

    # Enumerate all outcomes
    pmf: Dict[int, float] = {}
    total_outcomes = 36  # 6 * 6
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            s = d1 + d2
            pmf[s] = pmf.get(s, 0) + 1 / total_outcomes

    # Build CDF
    cdf: Dict[int, float] = {}
    cumulative = 0.0
    for s in sorted(pmf):
        cumulative += pmf[s]
        cdf[s] = cumulative

    # Display
    print(f"  {'Sum':>4} {'PMF':>10} {'CDF':>10}")
    print(f"  {'----':>4} {'----------':>10} {'----------':>10}")
    for s in sorted(pmf):
        bar = "#" * int(pmf[s] * 60)
        print(f"  {s:>4} {pmf[s]:>10.4f} {cdf[s]:>10.4f}  {bar}")

    # Sanity check
    assert abs(sum(pmf.values()) - 1.0) < 1e-12, "PMF does not sum to 1"
    assert abs(cdf[12] - 1.0) < 1e-12, "CDF at max should be 1"
    print(f"  PMF sums to {sum(pmf.values()):.6f} (verified).\n")


# === Exercise 2: Inverse CDF Transform — Uniform to Exponential ===
def exercise_2() -> None:
    """Generate Exponential(lambda=2) samples from Uniform(0,1) using the
    inverse CDF method.  F(x) = 1 - e^{-lx}  =>  x = -ln(1-u)/l."""
    print("=== Exercise 2: Inverse CDF Transform — Exponential ===")

    lam = 2.0  # rate parameter
    n = 50_000
    random.seed(123)

    samples: List[float] = []
    for _ in range(n):
        u = random.random()
        x = -math.log(1 - u) / lam
        samples.append(x)

    # Theoretical moments
    theoretical_mean = 1 / lam
    theoretical_var = 1 / lam**2

    # Empirical moments
    empirical_mean = sum(samples) / n
    empirical_var = sum((x - empirical_mean) ** 2 for x in samples) / (n - 1)

    print(f"  Lambda = {lam}")
    print(f"  Generated {n:,} samples via inverse CDF.")
    print(f"  Theoretical mean = {theoretical_mean:.4f}, "
          f"Empirical mean = {empirical_mean:.4f}")
    print(f"  Theoretical var  = {theoretical_var:.4f}, "
          f"Empirical var  = {empirical_var:.4f}")

    # Check P(X > 1) theoretically vs empirically
    p_gt1_theory = math.exp(-lam * 1)
    p_gt1_empirical = sum(1 for x in samples if x > 1) / n
    print(f"  P(X > 1): theory = {p_gt1_theory:.4f}, "
          f"empirical = {p_gt1_empirical:.4f}\n")


# === Exercise 3: P(1 < X < 3) for X ~ N(2, 1) Numerically ===
def exercise_3() -> None:
    """Compute P(1 < X < 3) for X ~ Normal(mu=2, sigma^2=1) using numerical
    integration (trapezoidal rule) of the PDF."""
    print("=== Exercise 3: P(1 < X < 3) for X ~ N(2,1) ===")

    mu = 2.0
    sigma = 1.0

    def normal_pdf(x: float, mu: float, sigma: float) -> float:
        """Standard normal PDF."""
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    # Trapezoidal integration
    a, b = 1.0, 3.0
    n_steps = 100_000
    h = (b - a) / n_steps

    integral = 0.5 * (normal_pdf(a, mu, sigma) + normal_pdf(b, mu, sigma))
    for i in range(1, n_steps):
        x = a + i * h
        integral += normal_pdf(x, mu, sigma)
    integral *= h

    # This should be close to P(|Z| < 1) for Z ~ N(0,1) = erf(1/sqrt(2))
    expected = math.erf(1.0 / math.sqrt(2))

    print(f"  X ~ N({mu}, {sigma}^2)")
    print(f"  P(1 < X < 3) by trapezoidal rule ({n_steps:,} steps): {integral:.8f}")
    print(f"  Expected (erf-based):                              {expected:.8f}")
    print(f"  Difference: {abs(integral - expected):.2e}\n")


# === Exercise 4: Median of a Discrete PMF ===
def exercise_4() -> None:
    """Given a discrete PMF, find the median (smallest x such that
    CDF(x) >= 0.5)."""
    print("=== Exercise 4: Median of a Discrete PMF ===")

    # PMF for a custom discrete random variable
    pmf: List[Tuple[int, float]] = [
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.25),
        (5, 0.20),
        (6, 0.15),
        (7, 0.10),
    ]

    # Verify PMF sums to 1
    total_prob = sum(p for _, p in pmf)
    assert abs(total_prob - 1.0) < 1e-12, f"PMF sums to {total_prob}, not 1"

    # Compute CDF and find median
    print(f"  {'x':>4} {'P(X=x)':>8} {'CDF(x)':>8}")
    print(f"  {'----':>4} {'--------':>8} {'--------':>8}")

    cumulative = 0.0
    median: int = pmf[0][0]
    median_found = False
    for x, p in pmf:
        cumulative += p
        marker = ""
        if cumulative >= 0.5 and not median_found:
            median = x
            median_found = True
            marker = " <-- median"
        print(f"  {x:>4} {p:>8.4f} {cumulative:>8.4f}{marker}")

    # Also compute mean for comparison
    mean = sum(x * p for x, p in pmf)

    print(f"\n  Median = {median}")
    print(f"  Mean   = {mean:.2f}")
    print(f"  The median is the smallest x with CDF(x) >= 0.5.\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

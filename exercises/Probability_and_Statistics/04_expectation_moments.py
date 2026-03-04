"""
Probability and Statistics — Expectation and Moments
Exercises covering expected value, variance, the Chebyshev inequality,
skewness, and kurtosis.
"""
import math
import random
from typing import Dict, List


# === Exercise 1: E[X] and Var(X) for a Loaded Die ===
def exercise_1() -> None:
    """Compute the expected value E[X] and variance Var(X) for a loaded die
    with a given PMF."""
    print("=== Exercise 1: E[X] and Var(X) for a Loaded Die ===")

    # Loaded die PMF: face -> probability
    pmf: Dict[int, float] = {
        1: 0.10,
        2: 0.10,
        3: 0.10,
        4: 0.15,
        5: 0.25,
        6: 0.30,
    }

    # Verify probabilities sum to 1
    assert abs(sum(pmf.values()) - 1.0) < 1e-12

    # E[X] = sum(x * P(X=x))
    ex = sum(x * p for x, p in pmf.items())

    # E[X^2] = sum(x^2 * P(X=x))
    ex2 = sum(x**2 * p for x, p in pmf.items())

    # Var(X) = E[X^2] - (E[X])^2
    var_x = ex2 - ex**2

    # Standard deviation
    std_x = math.sqrt(var_x)

    print("  PMF of loaded die:")
    for face, prob in sorted(pmf.items()):
        bar = "#" * int(prob * 40)
        print(f"    Face {face}: P = {prob:.2f}  {bar}")

    print(f"\n  E[X]      = {ex:.4f}")
    print(f"  E[X^2]    = {ex2:.4f}")
    print(f"  Var(X)    = E[X^2] - (E[X])^2 = {ex2:.4f} - {ex:.4f}^2 = {var_x:.4f}")
    print(f"  Std(X)    = {std_x:.4f}")

    # Compare with fair die
    fair_ex = 3.5
    fair_var = 35 / 12
    print(f"\n  Fair die: E[X] = {fair_ex:.4f}, Var(X) = {fair_var:.4f}")
    print(f"  Loaded die is {'more' if var_x > fair_var else 'less'} "
          f"variable than a fair die.\n")


# === Exercise 2: Verify Var(aX+b) = a^2 * Var(X) by Simulation ===
def exercise_2() -> None:
    """Generate X ~ Uniform(0,10), set Y = 3X + 5, and verify by simulation
    that Var(Y) = 9 * Var(X)."""
    print("=== Exercise 2: Verify Var(aX + b) = a^2 Var(X) ===")

    a, b = 3.0, 5.0
    n = 100_000
    random.seed(77)

    # Generate X ~ Uniform(0, 10)
    X: List[float] = [random.uniform(0, 10) for _ in range(n)]
    Y: List[float] = [a * x + b for x in X]

    # Empirical means
    mean_x = sum(X) / n
    mean_y = sum(Y) / n

    # Empirical variances (sample variance with n-1)
    var_x = sum((x - mean_x) ** 2 for x in X) / (n - 1)
    var_y = sum((y - mean_y) ** 2 for y in Y) / (n - 1)

    # Theoretical: Uniform(0,10) => Var = (10-0)^2 / 12 = 100/12
    var_x_theory = 100 / 12
    var_y_theory = a**2 * var_x_theory

    ratio = var_y / var_x

    print(f"  X ~ Uniform(0, 10),  Y = {a:.0f}X + {b:.0f}")
    print(f"  n = {n:,} samples")
    print(f"  Theoretical Var(X) = {var_x_theory:.4f},  Empirical = {var_x:.4f}")
    print(f"  Theoretical Var(Y) = {var_y_theory:.4f},  Empirical = {var_y:.4f}")
    print(f"  Var(Y)/Var(X) = {ratio:.4f}  (should be a^2 = {a**2:.1f})")
    print(f"  Var(aX+b) = a^2 * Var(X) verified.\n")


# === Exercise 3: Chebyshev's Inequality ===
def exercise_3() -> None:
    """Use Chebyshev's inequality to bound P(|X - mu| > 2*sigma), then
    compare with simulation for different distributions."""
    print("=== Exercise 3: Chebyshev's Inequality ===")

    # Chebyshev bound: P(|X - mu| > k*sigma) <= 1/k^2
    k = 2.0
    chebyshev_bound = 1 / k**2

    print(f"  Chebyshev bound: P(|X - mu| > {k:.0f}*sigma) <= 1/{k:.0f}^2 "
          f"= {chebyshev_bound:.4f}")
    print()

    n = 200_000
    random.seed(99)

    # Test with multiple distributions
    distributions = {
        "Uniform(0,1)": lambda: random.uniform(0, 1),
        "Exponential(1)": lambda: -math.log(1 - random.random()),
        "Normal(0,1) approx": lambda: sum(random.random() for _ in range(12)) - 6,
    }

    for name, sampler in distributions.items():
        samples = [sampler() for _ in range(n)]
        mu = sum(samples) / n
        sigma = math.sqrt(sum((x - mu) ** 2 for x in samples) / (n - 1))

        # Count violations
        violations = sum(1 for x in samples if abs(x - mu) > k * sigma)
        p_empirical = violations / n

        print(f"  {name}:")
        print(f"    mu = {mu:.4f}, sigma = {sigma:.4f}")
        print(f"    P(|X-mu| > {k:.0f}*sigma) = {p_empirical:.6f}  "
              f"(Chebyshev bound: {chebyshev_bound:.4f})")
        within = "YES" if p_empirical <= chebyshev_bound else "NO"
        print(f"    Within Chebyshev bound? {within}")

    print()


# === Exercise 4: Skewness and Kurtosis from a Sample ===
def exercise_4() -> None:
    """Compute the sample skewness and excess kurtosis from generated data.
    Compare a symmetric vs. skewed distribution."""
    print("=== Exercise 4: Skewness and Kurtosis ===")

    n = 100_000
    random.seed(42)

    def compute_moments(data: List[float]) -> tuple:
        """Compute mean, variance, skewness, excess kurtosis."""
        m = len(data)
        mean = sum(data) / m
        # Central moments
        m2 = sum((x - mean) ** 2 for x in data) / m
        m3 = sum((x - mean) ** 3 for x in data) / m
        m4 = sum((x - mean) ** 4 for x in data) / m

        std = math.sqrt(m2)
        skewness = m3 / std**3 if std > 0 else 0.0
        kurtosis = (m4 / std**4) - 3  # excess kurtosis
        return mean, m2, skewness, kurtosis

    # Distribution 1: Approximately symmetric (Normal via CLT)
    symmetric_data = [sum(random.random() for _ in range(12)) - 6 for _ in range(n)]

    # Distribution 2: Right-skewed (Exponential via inverse CDF)
    skewed_data = [-math.log(1 - random.random()) for _ in range(n)]

    for name, data, expected_skew, expected_kurt in [
        ("Normal(0,1) approx", symmetric_data, 0.0, 0.0),
        ("Exponential(1)", skewed_data, 2.0, 6.0),
    ]:
        mean, var, skew, kurt = compute_moments(data)
        print(f"  {name} ({n:,} samples):")
        print(f"    Mean     = {mean:>8.4f}")
        print(f"    Variance = {var:>8.4f}")
        print(f"    Skewness = {skew:>8.4f}  (theoretical: {expected_skew:.1f})")
        print(f"    Excess Kurtosis = {kurt:>8.4f}  (theoretical: {expected_kurt:.1f})")
        print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

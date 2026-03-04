"""
Probability and Statistics — Transformations of Random Variables
Exercises covering Jacobian calculation, CDF technique application,
Box-Muller verification, and order statistics of a uniform sample.
"""
import math
import random
from typing import List, Tuple


# === Exercise 1: Jacobian Calculation for Exponential Transform ===
def exercise_1() -> None:
    """Given X ~ Uniform(0,1), verify that Y = -ln(X)/lambda is Exp(lambda)
    by computing the Jacobian and comparing simulated vs theoretical moments."""
    print("=== Exercise 1: Jacobian Calculation for Exponential Transform ===")

    # Theory: Y = g(X) = -ln(X)/lambda, X ~ Uniform(0,1)
    # Inverse: X = g^{-1}(Y) = exp(-lambda*Y)
    # Jacobian: |dX/dY| = lambda * exp(-lambda*Y)
    # f_Y(y) = f_X(g^{-1}(y)) * |dX/dY| = 1 * lambda*exp(-lambda*y) for y > 0
    # => Y ~ Exp(lambda)

    lam = 3.0
    print(f"\n  Transformation: Y = -ln(X)/{lam}, X ~ Uniform(0,1)")
    print(f"  Inverse: X = exp(-{lam}*Y)")
    print(f"  Jacobian: |dX/dY| = {lam}*exp(-{lam}*Y)")
    print(f"  f_Y(y) = 1 * {lam}*exp(-{lam}*y) = Exp({lam}) PDF")

    random.seed(42)
    n = 100000
    uniform_samples = [random.random() for _ in range(n)]
    y_samples = [-math.log(u) / lam for u in uniform_samples]

    # Simulated moments
    mean_sim = sum(y_samples) / n
    var_sim = sum((y - mean_sim) ** 2 for y in y_samples) / n

    # Theoretical: E[Y] = 1/lambda, Var(Y) = 1/lambda^2
    mean_th = 1 / lam
    var_th = 1 / lam ** 2

    print(f"\n  Simulation (n={n:,}):")
    print(f"  E[Y]   = {mean_sim:.5f}  (theory: {mean_th:.5f})")
    print(f"  Var(Y) = {var_sim:.5f}  (theory: {var_th:.5f})")

    # Verify PDF at selected points via histogram density
    print(f"\n  PDF verification f_Y(y) = {lam}*exp(-{lam}*y):")
    bin_width = 0.1
    for y_center in [0.05, 0.15, 0.35, 0.55, 0.95]:
        count = sum(1 for y in y_samples
                    if y_center - bin_width / 2 <= y < y_center + bin_width / 2)
        density_sim = count / (n * bin_width)
        density_th = lam * math.exp(-lam * y_center)
        print(f"    y={y_center:.2f}: sim={density_sim:.3f}, theory={density_th:.3f}")
    print()


# === Exercise 2: CDF Technique — Square of a Normal ===
def exercise_2() -> None:
    """Apply the CDF technique to derive Y = X^2 where X ~ N(0,1),
    and verify via simulation that Y ~ chi-squared(1)."""
    print("=== Exercise 2: CDF Technique -- Y = X^2 for X ~ N(0,1) ===")

    # Theory:
    # F_Y(y) = P(X^2 <= y) = P(-sqrt(y) <= X <= sqrt(y))
    #         = 2*Phi(sqrt(y)) - 1   for y > 0
    # f_Y(y) = phi(sqrt(y)) / sqrt(y) = chi-squared(1) density

    print("\n  Derivation via CDF technique:")
    print("  F_Y(y) = P(X^2 <= y) = 2*Phi(sqrt(y)) - 1  for y > 0")
    print("  f_Y(y) = phi(sqrt(y)) / sqrt(y)")
    print("         = [1/sqrt(2*pi)] * y^{-1/2} * exp(-y/2)")
    print("         = chi-squared(1) density")

    random.seed(42)
    n = 100000
    x_samples = [random.gauss(0, 1) for _ in range(n)]
    y_samples = [x ** 2 for x in x_samples]

    mean_sim = sum(y_samples) / n
    var_sim = sum((y - mean_sim) ** 2 for y in y_samples) / n

    # chi-squared(1): E[Y] = 1, Var(Y) = 2
    print(f"\n  Simulation (n={n:,}):")
    print(f"  E[Y]   = {mean_sim:.4f}  (theory: 1.0000)")
    print(f"  Var(Y) = {var_sim:.4f}  (theory: 2.0000)")

    # CDF comparison
    def phi_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    print(f"\n  CDF verification P(Y <= y):")
    for y_val in [0.5, 1.0, 2.0, 3.84, 6.63]:
        cdf_sim = sum(1 for y in y_samples if y <= y_val) / n
        cdf_th = 2 * phi_cdf(math.sqrt(y_val)) - 1
        print(f"    y={y_val:<5}: sim={cdf_sim:.4f}, theory={cdf_th:.4f}")
    print()


# === Exercise 3: Box-Muller Transform Verification ===
def exercise_3() -> None:
    """Implement the Box-Muller transform and verify that generated pairs
    (Z1, Z2) are independent standard normals."""
    print("=== Exercise 3: Box-Muller Transform Verification ===")

    def box_muller(u1: float, u2: float) -> Tuple[float, float]:
        """Transform two Uniform(0,1) into two independent N(0,1)."""
        r = math.sqrt(-2 * math.log(u1))
        theta = 2 * math.pi * u2
        z1 = r * math.cos(theta)
        z2 = r * math.sin(theta)
        return z1, z2

    random.seed(42)
    n = 100000
    z1_list: List[float] = []
    z2_list: List[float] = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z1, z2 = box_muller(u1, u2)
        z1_list.append(z1)
        z2_list.append(z2)

    # Verify moments
    mean_z1 = sum(z1_list) / n
    mean_z2 = sum(z2_list) / n
    var_z1 = sum((z - mean_z1) ** 2 for z in z1_list) / n
    var_z2 = sum((z - mean_z2) ** 2 for z in z2_list) / n
    cov_12 = sum((a - mean_z1) * (b - mean_z2)
                 for a, b in zip(z1_list, z2_list)) / n
    rho = cov_12 / (math.sqrt(var_z1) * math.sqrt(var_z2))

    print(f"\n  Box-Muller: U1,U2 ~ Uniform(0,1)")
    print(f"  Z1 = sqrt(-2 ln U1) * cos(2*pi*U2)")
    print(f"  Z2 = sqrt(-2 ln U1) * sin(2*pi*U2)")
    print(f"\n  Simulation ({n:,} pairs):")
    print(f"  {'':>5} {'Mean':>10} {'Var':>10} {'(Theory)':>10}")
    print(f"  {'Z1':>5} {mean_z1:>10.5f} {var_z1:>10.5f} {'0, 1':>10}")
    print(f"  {'Z2':>5} {mean_z2:>10.5f} {var_z2:>10.5f} {'0, 1':>10}")
    print(f"\n  Cov(Z1,Z2) = {cov_12:.5f}  (theory: 0)")
    print(f"  Corr(Z1,Z2) = {rho:.5f}  (theory: 0)")

    # Check normality: proportion within 1, 2, 3 sigma
    print(f"\n  Proportion within k sigma (Z1):")
    for k in [1, 2, 3]:
        prop_sim = sum(1 for z in z1_list if abs(z) <= k) / n
        prop_th = 2 * (0.5 * (1 + math.erf(k / math.sqrt(2)))) - 1
        print(f"    |Z1| <= {k}: sim={prop_sim:.4f}, theory={prop_th:.4f}")

    assert abs(mean_z1) < 0.02, "Z1 mean not close to 0"
    assert abs(var_z1 - 1) < 0.05, "Z1 variance not close to 1"
    assert abs(rho) < 0.02, "Z1 and Z2 not independent"
    print(f"\n  All checks passed.\n")


# === Exercise 4: Order Statistics of Uniform Sample ===
def exercise_4() -> None:
    """Compute and verify the expected values and variances of order statistics
    X_(k) from a Uniform(0,1) sample of size n=7."""
    print("=== Exercise 4: Order Statistics of Uniform(0,1) Sample ===")

    n = 7  # sample size
    n_sims = 50000

    # Theory for Uniform(0,1):
    # E[X_(k)] = k / (n+1)
    # Var(X_(k)) = k*(n-k+1) / ((n+1)^2*(n+2))
    # X_(k) ~ Beta(k, n-k+1)

    random.seed(42)

    print(f"\n  Sample size n={n}, {n_sims:,} simulations")
    print(f"  X_(k) ~ Beta(k, n-k+1) for Uniform(0,1)")
    print(f"\n  {'k':>4} {'E[X_(k)]':>10} {'E[sim]':>10} "
          f"{'Var[X_(k)]':>12} {'Var[sim]':>10}")
    print(f"  {'---':>4} {'----------':>10} {'------':>10} "
          f"{'----------':>12} {'--------':>10}")

    all_order_stats: dict = {k: [] for k in range(1, n + 1)}
    for _ in range(n_sims):
        sample = sorted(random.random() for _ in range(n))
        for k in range(1, n + 1):
            all_order_stats[k].append(sample[k - 1])

    for k in range(1, n + 1):
        e_th = k / (n + 1)
        var_th = k * (n - k + 1) / ((n + 1) ** 2 * (n + 2))
        vals = all_order_stats[k]
        e_sim = sum(vals) / n_sims
        var_sim = sum((v - e_sim) ** 2 for v in vals) / n_sims
        print(f"  {k:>4} {e_th:>10.5f} {e_sim:>10.5f} "
              f"{var_th:>12.6f} {var_sim:>10.6f}")

    # Verify range R = X_(n) - X_(1)
    ranges = [all_order_stats[n][i] - all_order_stats[1][i]
              for i in range(n_sims)]
    range_mean = sum(ranges) / n_sims
    range_th = (n - 1) / (n + 1)
    print(f"\n  Range R = X_({n}) - X_(1):")
    print(f"  E[R] = {range_mean:.5f}  (theory: (n-1)/(n+1) = {range_th:.5f})")

    # Verify median X_(4) is symmetric around 0.5
    median_vals = all_order_stats[4]
    median_mean = sum(median_vals) / n_sims
    print(f"\n  Median X_(4): E[median] = {median_mean:.5f}  (theory: 0.5)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

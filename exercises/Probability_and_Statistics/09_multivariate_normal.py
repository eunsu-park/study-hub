"""
Probability and Statistics — Multivariate Normal Distribution
Exercises covering covariance matrix construction, conditional distribution
computation, independence check, and Mahalanobis distance.
"""
import math
import random
from typing import Dict, List, Tuple


# === Helpers: 2x2 Matrix Operations ===

def cholesky_2x2(sigma: List[List[float]]) -> List[List[float]]:
    """Cholesky decomposition for 2x2 positive-definite matrix: Sigma = L L^T."""
    l11 = math.sqrt(sigma[0][0])
    l21 = sigma[1][0] / l11
    l22 = math.sqrt(sigma[1][1] - l21 ** 2)
    return [[l11, 0.0], [l21, l22]]


def mat_vec_2x2(m: List[List[float]], v: List[float]) -> List[float]:
    """2x2 matrix times 2-vector."""
    return [m[0][0] * v[0] + m[0][1] * v[1],
            m[1][0] * v[0] + m[1][1] * v[1]]


def inv_2x2(m: List[List[float]]) -> List[List[float]]:
    """Inverse of a 2x2 matrix."""
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return [[m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det]]


def generate_bvn(mu: List[float], sigma: List[List[float]],
                 n: int) -> List[Tuple[float, float]]:
    """Generate n bivariate normal samples using Cholesky decomposition."""
    L = cholesky_2x2(sigma)
    samples = []
    for _ in range(n):
        z = [random.gauss(0, 1), random.gauss(0, 1)]
        x = mat_vec_2x2(L, z)
        samples.append((x[0] + mu[0], x[1] + mu[1]))
    return samples


# === Exercise 1: Covariance Matrix Construction and Verification ===
def exercise_1() -> None:
    """Construct a covariance matrix from given standard deviations and
    correlation, generate bivariate normal samples, and verify empirical
    statistics match the parameters."""
    print("=== Exercise 1: Covariance Matrix Construction ===")

    sigma1 = 3.0
    sigma2 = 5.0
    rho = 0.6
    mu = [10.0, 20.0]

    # Build covariance matrix
    cov12 = rho * sigma1 * sigma2
    Sigma = [[sigma1 ** 2, cov12],
             [cov12, sigma2 ** 2]]

    print(f"\n  Given: sigma1={sigma1}, sigma2={sigma2}, rho={rho}")
    print(f"  mu = {mu}")
    print(f"  Covariance matrix:")
    print(f"    Sigma = [[{Sigma[0][0]:>6.1f}, {Sigma[0][1]:>6.1f}],")
    print(f"             [{Sigma[1][0]:>6.1f}, {Sigma[1][1]:>6.1f}]]")

    # Verify positive definiteness: det(Sigma) > 0
    det = Sigma[0][0] * Sigma[1][1] - Sigma[0][1] * Sigma[1][0]
    print(f"\n  det(Sigma) = {det:.2f} > 0: Positive definite.")

    # Generate samples and verify
    random.seed(42)
    n = 100000
    samples = generate_bvn(mu, Sigma, n)
    x1 = [s[0] for s in samples]
    x2 = [s[1] for s in samples]

    m1 = sum(x1) / n
    m2 = sum(x2) / n
    v1 = sum((x - m1) ** 2 for x in x1) / n
    v2 = sum((x - m2) ** 2 for x in x2) / n
    c12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2)) / n
    rho_emp = c12 / (math.sqrt(v1) * math.sqrt(v2))

    print(f"\n  Simulation ({n:,} samples):")
    print(f"  {'':>14} {'Theory':>10} {'Empirical':>10}")
    print(f"  {'E[X1]':>14} {mu[0]:>10.4f} {m1:>10.4f}")
    print(f"  {'E[X2]':>14} {mu[1]:>10.4f} {m2:>10.4f}")
    print(f"  {'Var(X1)':>14} {sigma1**2:>10.4f} {v1:>10.4f}")
    print(f"  {'Var(X2)':>14} {sigma2**2:>10.4f} {v2:>10.4f}")
    print(f"  {'Cov(X1,X2)':>14} {cov12:>10.4f} {c12:>10.4f}")
    print(f"  {'rho':>14} {rho:>10.4f} {rho_emp:>10.4f}")

    assert abs(rho_emp - rho) < 0.02, "Correlation mismatch"
    print(f"\n  All statistics match theoretical values.\n")


# === Exercise 2: Conditional Distribution Computation ===
def exercise_2() -> None:
    """Compute the conditional distribution X1|X2=x2 for a bivariate normal
    and verify by simulation."""
    print("=== Exercise 2: Conditional Distribution X1|X2=x2 ===")

    mu = [5.0, 3.0]
    sigma = [[4.0, 2.1], [2.1, 2.25]]

    # Conditional: X1|X2=x2 ~ N(mu_cond, var_cond)
    # mu_cond = mu1 + Sigma12/Sigma22 * (x2 - mu2)
    # var_cond = Sigma11 - Sigma12^2/Sigma22

    var_cond = sigma[0][0] - sigma[0][1] ** 2 / sigma[1][1]
    regression_coeff = sigma[0][1] / sigma[1][1]

    print(f"\n  Joint: N(mu={mu}, Sigma=[[4, 2.1],[2.1, 2.25]])")
    print(f"\n  Conditional formula:")
    print(f"    E[X1|X2=x2] = mu1 + (Sigma12/Sigma22)*(x2 - mu2)")
    print(f"                = {mu[0]} + ({sigma[0][1]}/{sigma[1][1]})*(x2 - {mu[1]})")
    print(f"                = {mu[0]} + {regression_coeff:.4f}*(x2 - {mu[1]})")
    print(f"    Var(X1|X2)  = Sigma11 - Sigma12^2/Sigma22")
    print(f"                = {sigma[0][0]} - {sigma[0][1]}^2/{sigma[1][1]}")
    print(f"                = {var_cond:.4f}")

    print(f"\n  Conditional distributions for selected x2 values:")
    print(f"  {'x2':>6} {'E[X1|X2=x2]':>14} {'Var(X1|X2)':>12}")
    print(f"  {'------':>6} {'-----------':>14} {'----------':>12}")
    for x2_val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        cond_mean = mu[0] + regression_coeff * (x2_val - mu[1])
        print(f"  {x2_val:>6.1f} {cond_mean:>14.4f} {var_cond:>12.4f}")

    # Simulation verification for x2 = 4.0
    x2_target = 4.0
    cond_mean_th = mu[0] + regression_coeff * (x2_target - mu[1])

    random.seed(42)
    samples = generate_bvn(mu, sigma, 200000)
    tolerance = 0.1
    filtered = [s[0] for s in samples if abs(s[1] - x2_target) < tolerance]

    if len(filtered) > 100:
        sim_mean = sum(filtered) / len(filtered)
        sim_var = sum((x - sim_mean) ** 2 for x in filtered) / len(filtered)
        print(f"\n  Verification (X2 ~ {x2_target}, n={len(filtered)}):")
        print(f"    Theory:     E[X1|X2={x2_target}] = {cond_mean_th:.4f}, "
              f"Var = {var_cond:.4f}")
        print(f"    Simulation: E[X1|X2~{x2_target}] = {sim_mean:.4f}, "
              f"Var = {sim_var:.4f}")
    print()


# === Exercise 3: Independence Check for Bivariate Normal ===
def exercise_3() -> None:
    """Check whether two jointly normal random variables are independent
    by examining the covariance (off-diagonal) entry."""
    print("=== Exercise 3: Independence Check for Bivariate Normal ===")

    # For jointly normal RVs: independence <=> zero correlation
    cases: List[Tuple[str, List[List[float]]]] = [
        ("Case 1: Independent", [[4.0, 0.0], [0.0, 9.0]]),
        ("Case 2: Dependent (rho=0.5)", [[4.0, 3.0], [3.0, 9.0]]),
        ("Case 3: Strongly dependent (rho=-0.8)", [[1.0, -0.8], [-0.8, 1.0]]),
    ]

    for label, Sigma in cases:
        print(f"\n  {label}")
        print(f"    Sigma = [[{Sigma[0][0]:.1f}, {Sigma[0][1]:.1f}],")
        print(f"             [{Sigma[1][0]:.1f}, {Sigma[1][1]:.1f}]]")

        cov_12 = Sigma[0][1]
        sd1 = math.sqrt(Sigma[0][0])
        sd2 = math.sqrt(Sigma[1][1])
        rho = cov_12 / (sd1 * sd2)

        independent = abs(cov_12) < 1e-12
        print(f"    Cov(X1,X2) = {cov_12:.4f}")
        print(f"    rho = {rho:.4f}")
        print(f"    Independent? {'YES' if independent else 'NO'}")

        if not independent:
            print(f"    (For jointly normal, Cov=0 iff independent)")

    # Empirical demonstration: generate from independent vs dependent
    random.seed(42)
    n = 50000

    # Independent case
    mu = [0.0, 0.0]
    Sigma_indep = [[4.0, 0.0], [0.0, 9.0]]
    samples = generate_bvn(mu, Sigma_indep, n)
    x1 = [s[0] for s in samples]
    x2 = [s[1] for s in samples]
    m1 = sum(x1) / n
    m2 = sum(x2) / n
    cov_emp = sum((a - m1) * (b - m2) for a, b in zip(x1, x2)) / n
    rho_emp = cov_emp / (math.sqrt(sum((a - m1)**2 for a in x1) / n) *
                         math.sqrt(sum((b - m2)**2 for b in x2) / n))

    print(f"\n  Empirical check (n={n:,}):")
    print(f"    Independent case:  rho_emp = {rho_emp:.5f} (theory: 0)")

    # Dependent case
    Sigma_dep = [[4.0, 3.0], [3.0, 9.0]]
    samples = generate_bvn(mu, Sigma_dep, n)
    x1 = [s[0] for s in samples]
    x2 = [s[1] for s in samples]
    m1 = sum(x1) / n
    m2 = sum(x2) / n
    cov_emp = sum((a - m1) * (b - m2) for a, b in zip(x1, x2)) / n
    rho_emp = cov_emp / (math.sqrt(sum((a - m1)**2 for a in x1) / n) *
                         math.sqrt(sum((b - m2)**2 for b in x2) / n))

    print(f"    Dependent case:    rho_emp = {rho_emp:.5f} (theory: 0.5)")
    print()


# === Exercise 4: Mahalanobis Distance ===
def exercise_4() -> None:
    """Compute the Mahalanobis distance for several test points from a
    bivariate normal, and verify that d^2 ~ chi-squared(2) by simulation."""
    print("=== Exercise 4: Mahalanobis Distance ===")

    mu = [0.0, 0.0]
    Sigma = [[4.0, 1.5], [1.5, 1.0]]
    Sigma_inv = inv_2x2(Sigma)

    def mahalanobis_sq(x: List[float]) -> float:
        """Compute d^2 = (x-mu)^T Sigma^{-1} (x-mu)."""
        diff = [x[0] - mu[0], x[1] - mu[1]]
        temp = mat_vec_2x2(Sigma_inv, diff)
        return diff[0] * temp[0] + diff[1] * temp[1]

    points = [
        [0, 0], [2, 0], [0, 1], [2, 1], [4, 2], [-1, -1],
    ]

    print(f"\n  Sigma = [[4, 1.5],[1.5, 1]], mu = {mu}")
    print(f"\n  {'Point':>12} {'d^2':>10} {'d':>10} {'Euclidean':>10}")
    print(f"  {'-' * 44}")
    for p in points:
        d2 = mahalanobis_sq(p)
        d = math.sqrt(max(d2, 0))
        d_euc = math.sqrt(p[0] ** 2 + p[1] ** 2)
        print(f"  ({p[0]:>3},{p[1]:>3})   {d2:>10.4f} {d:>10.4f} {d_euc:>10.4f}")

    # Verify d^2 ~ chi-squared(2) via simulation
    random.seed(42)
    n = 50000
    samples = generate_bvn(mu, Sigma, n)
    d2_values = [mahalanobis_sq(list(s)) for s in samples]

    mean_d2 = sum(d2_values) / n
    # P(d^2 <= c) for chi-squared(2): 1 - exp(-c/2)
    within_2 = sum(1 for d in d2_values if d <= 2) / n
    within_5991 = sum(1 for d in d2_values if d <= 5.991) / n

    print(f"\n  d^2 distribution verification (should be chi-squared(2)):")
    print(f"  E[d^2] = {mean_d2:.4f}  (theory: 2)")
    print(f"  P(d^2 <= 2)     = {within_2:.4f}  (theory: {1 - math.exp(-1):.4f})")
    print(f"  P(d^2 <= 5.991) = {within_5991:.4f}  (theory: 0.9500)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

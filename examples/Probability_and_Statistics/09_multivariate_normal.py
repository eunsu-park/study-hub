"""
Multivariate Normal Distribution

Demonstrates:
1. Bivariate Normal Generation (Cholesky)
2. Conditional Distribution
3. Mahalanobis Distance
4. Linear Transformations

Theory:
- X ~ N(μ, Σ) where Σ is positive definite covariance matrix
- Marginals are normal: X_i ~ N(μ_i, Σ_{ii})
- Conditional: X₁|X₂=x₂ ~ N(μ₁ + Σ₁₂Σ₂₂⁻¹(x₂-μ₂), Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁)
- Mahalanobis: d² = (x-μ)ᵀΣ⁻¹(x-μ)

Adapted from Probability and Statistics Lesson 09.
"""

import math
import random


# ─────────────────────────────────────────────────
# HELPER: 2x2 Matrix Operations
# ─────────────────────────────────────────────────

def cholesky_2x2(sigma: list[list[float]]) -> list[list[float]]:
    """Cholesky decomposition for 2x2 positive-definite matrix.
    Σ = LLᵀ where L is lower triangular."""
    a = sigma[0][0]
    b = sigma[1][0]
    d = sigma[1][1]
    l11 = math.sqrt(a)
    l21 = b / l11
    l22 = math.sqrt(d - l21**2)
    return [[l11, 0], [l21, l22]]


def mat_vec_2x2(m: list[list[float]], v: list[float]) -> list[float]:
    """2x2 matrix × 2-vector."""
    return [m[0][0]*v[0] + m[0][1]*v[1],
            m[1][0]*v[0] + m[1][1]*v[1]]


def det_2x2(m: list[list[float]]) -> float:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def inv_2x2(m: list[list[float]]) -> list[list[float]]:
    d = det_2x2(m)
    return [[m[1][1]/d, -m[0][1]/d],
            [-m[1][0]/d, m[0][0]/d]]


# ─────────────────────────────────────────────────
# 1. BIVARIATE NORMAL GENERATION
# ─────────────────────────────────────────────────

def generate_bivariate_normal(mu: list[float],
                               sigma: list[list[float]],
                               n: int) -> list[tuple[float, float]]:
    """Generate n samples from N(μ, Σ) using Cholesky."""
    L = cholesky_2x2(sigma)
    samples = []
    for _ in range(n):
        z = [random.gauss(0, 1), random.gauss(0, 1)]
        x = mat_vec_2x2(L, z)
        samples.append((x[0] + mu[0], x[1] + mu[1]))
    return samples


def demo_bivariate():
    print("=" * 60)
    print("  Bivariate Normal Generation (Cholesky)")
    print("=" * 60)

    mu = [5.0, 3.0]
    rho = 0.7
    sigma = [[4.0, 0.7 * 2 * 1.5],
             [0.7 * 2 * 1.5, 2.25]]

    random.seed(42)
    samples = generate_bivariate_normal(mu, sigma, 10000)

    x_data = [s[0] for s in samples]
    y_data = [s[1] for s in samples]

    mx = sum(x_data) / len(x_data)
    my = sum(y_data) / len(y_data)
    vx = sum((x - mx)**2 for x in x_data) / len(x_data)
    vy = sum((y - my)**2 for y in y_data) / len(y_data)
    cov_xy = sum((x - mx)*(y - my) for x, y in zip(x_data, y_data)) / len(x_data)
    rho_sim = cov_xy / (math.sqrt(vx) * math.sqrt(vy))

    print(f"\n  Parameters: μ={mu}, ρ={rho}")
    print(f"  Σ = [[{sigma[0][0]}, {sigma[0][1]:.2f}],")
    print(f"       [{sigma[1][0]:.2f}, {sigma[1][1]}]]")
    print(f"\n  Simulated statistics (n=10000):")
    print(f"  E[X₁]={mx:.3f} (th:{mu[0]}), E[X₂]={my:.3f} (th:{mu[1]})")
    print(f"  Var(X₁)={vx:.3f} (th:{sigma[0][0]}), Var(X₂)={vy:.3f} (th:{sigma[1][1]})")
    print(f"  ρ = {rho_sim:.4f} (th:{rho})")

    # ASCII scatter plot
    print(f"\n  Scatter plot (first 500 samples):")
    grid = [[' '] * 40 for _ in range(20)]
    for x, y in samples[:500]:
        col = int((x - 0) / 10 * 39)
        row = int((y - -2) / 8 * 19)
        col = max(0, min(39, col))
        row = max(0, min(19, row))
        grid[19 - row][col] = '·'
    for row in grid:
        print("  " + "".join(row))


# ─────────────────────────────────────────────────
# 2. CONDITIONAL DISTRIBUTION
# ─────────────────────────────────────────────────

def conditional_normal_2d(mu: list[float], sigma: list[list[float]],
                           x2_val: float) -> tuple[float, float]:
    """Compute conditional mean and variance of X₁|X₂=x₂.
    μ_{1|2} = μ₁ + Σ₁₂/Σ₂₂ · (x₂ - μ₂)
    σ²_{1|2} = Σ₁₁ - Σ₁₂²/Σ₂₂
    """
    cond_mean = mu[0] + sigma[0][1] / sigma[1][1] * (x2_val - mu[1])
    cond_var = sigma[0][0] - sigma[0][1]**2 / sigma[1][1]
    return cond_mean, cond_var


def demo_conditional():
    print("\n" + "=" * 60)
    print("  Conditional Distribution X₁|X₂=x₂")
    print("=" * 60)

    mu = [5.0, 3.0]
    sigma = [[4.0, 2.1], [2.1, 2.25]]

    print(f"\n  Joint: N(μ={mu}, Σ=[[4, 2.1],[2.1, 2.25]])")
    for x2 in [1.0, 3.0, 5.0]:
        cm, cv = conditional_normal_2d(mu, sigma, x2)
        print(f"  X₁|X₂={x2}: N({cm:.2f}, {cv:.3f})")

    # Verify by filtering simulation
    random.seed(42)
    samples = generate_bivariate_normal(mu, sigma, 100000)
    x2_target = 3.0
    tolerance = 0.1
    filtered = [s[0] for s in samples if abs(s[1] - x2_target) < tolerance]
    if filtered:
        sim_mean = sum(filtered) / len(filtered)
        sim_var = sum((x - sim_mean)**2 for x in filtered) / len(filtered)
        th_mean, th_var = conditional_normal_2d(mu, sigma, x2_target)
        print(f"\n  Verification (X₂ ≈ {x2_target}, n={len(filtered)}):")
        print(f"  Sim mean={sim_mean:.3f} (th:{th_mean:.3f})")
        print(f"  Sim var={sim_var:.3f} (th:{th_var:.3f})")


# ─────────────────────────────────────────────────
# 3. MAHALANOBIS DISTANCE
# ─────────────────────────────────────────────────

def mahalanobis_2d(x: list[float], mu: list[float],
                    sigma: list[list[float]]) -> float:
    """d² = (x-μ)ᵀ Σ⁻¹ (x-μ)."""
    inv = inv_2x2(sigma)
    diff = [x[0] - mu[0], x[1] - mu[1]]
    temp = mat_vec_2x2(inv, diff)
    return diff[0] * temp[0] + diff[1] * temp[1]


def demo_mahalanobis():
    print("\n" + "=" * 60)
    print("  Mahalanobis Distance")
    print("=" * 60)

    mu = [0, 0]
    sigma = [[4.0, 1.5], [1.5, 1.0]]

    points = [
        [0, 0], [2, 0], [0, 1], [2, 1], [4, 2], [-1, -1],
    ]

    print(f"\n  Σ = [[4, 1.5],[1.5, 1]]")
    print(f"  {'Point':>12} {'d²_Mah':>10} {'d_Euclid':>10}")
    print(f"  {'─' * 34}")
    for p in points:
        d2 = mahalanobis_2d(p, mu, sigma)
        d_euc = math.sqrt(p[0]**2 + p[1]**2)
        print(f"  ({p[0]:>3},{p[1]:>3})   {d2:>10.3f} {d_euc:>10.3f}")

    # d² ~ χ²(2) for MVN
    random.seed(42)
    samples = generate_bivariate_normal(mu, sigma, 10000)
    d2_values = [mahalanobis_2d(list(s), mu, sigma) for s in samples]
    mean_d2 = sum(d2_values) / len(d2_values)
    within_1 = sum(1 for d in d2_values if d <= 1) / len(d2_values)
    within_6 = sum(1 for d in d2_values if d <= 5.991) / len(d2_values)

    print(f"\n  d² distribution (should be χ²(2)):")
    print(f"  E[d²] = {mean_d2:.3f} (theory: 2)")
    print(f"  P(d²≤1) = {within_1:.3f}")
    print(f"  P(d²≤5.991) = {within_6:.3f} (theory: 0.95 for χ²(2))")


# ─────────────────────────────────────────────────
# 4. LINEAR TRANSFORMATIONS
# ─────────────────────────────────────────────────

def demo_linear_transform():
    print("\n" + "=" * 60)
    print("  Linear Transformations")
    print("=" * 60)

    print("""
  If X ~ N(μ, Σ), then AX + b ~ N(Aμ + b, AΣAᵀ)

  Example: X ~ N([0,0], [[1, 0.5],[0.5, 1]])
  A = [[2, 0],[1, 1]], b = [1, -1]
  Y = AX + b ~ N(Aμ+b, AΣAᵀ)
""")
    mu = [0, 0]
    sigma = [[1, 0.5], [0.5, 1]]
    A = [[2, 0], [1, 1]]
    b = [1, -1]

    # Aμ + b
    new_mu = [A[0][0]*mu[0] + A[0][1]*mu[1] + b[0],
              A[1][0]*mu[0] + A[1][1]*mu[1] + b[1]]

    # AΣAᵀ
    AS = [[A[0][0]*sigma[0][0]+A[0][1]*sigma[1][0],
           A[0][0]*sigma[0][1]+A[0][1]*sigma[1][1]],
          [A[1][0]*sigma[0][0]+A[1][1]*sigma[1][0],
           A[1][0]*sigma[0][1]+A[1][1]*sigma[1][1]]]
    new_sigma = [[AS[0][0]*A[0][0]+AS[0][1]*A[0][1],
                  AS[0][0]*A[1][0]+AS[0][1]*A[1][1]],
                 [AS[1][0]*A[0][0]+AS[1][1]*A[0][1],
                  AS[1][0]*A[1][0]+AS[1][1]*A[1][1]]]

    print(f"  New μ = {new_mu}")
    print(f"  New Σ = [[{new_sigma[0][0]:.2f}, {new_sigma[0][1]:.2f}],")
    print(f"           [{new_sigma[1][0]:.2f}, {new_sigma[1][1]:.2f}]]")

    # Verify by simulation
    random.seed(42)
    x_samples = generate_bivariate_normal(mu, sigma, 10000)
    y_samples = [(A[0][0]*x+A[0][1]*y+b[0], A[1][0]*x+A[1][1]*y+b[1])
                 for x, y in x_samples]

    y1 = [s[0] for s in y_samples]
    y2 = [s[1] for s in y_samples]
    sm1 = sum(y1) / len(y1)
    sm2 = sum(y2) / len(y2)
    print(f"\n  Simulation: E[Y₁]={sm1:.3f} (th:{new_mu[0]}), "
          f"E[Y₂]={sm2:.3f} (th:{new_mu[1]})")


if __name__ == "__main__":
    demo_bivariate()
    demo_conditional()
    demo_mahalanobis()
    demo_linear_transform()

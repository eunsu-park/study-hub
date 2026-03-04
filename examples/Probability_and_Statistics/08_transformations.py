"""
Transformations of Random Variables

Demonstrates:
1. CDF Technique
2. PDF Transformation with Jacobian
3. Box-Muller Transform
4. Order Statistics

Theory:
- CDF technique: F_Y(y) = P(g(X) ≤ y)
- PDF technique: f_Y(y) = f_X(g⁻¹(y)) · |dg⁻¹/dy|
- Box-Muller: U₁,U₂ ~ Uniform → Z₁,Z₂ ~ Normal
- k-th order statistic: f_{X_{(k)}}(x) = k·C(n,k)·[F(x)]^{k-1}·[1-F(x)]^{n-k}·f(x)

Adapted from Probability and Statistics Lesson 08.
"""

import math
import random
from collections import Counter


# ─────────────────────────────────────────────────
# 1. CDF TECHNIQUE
# ─────────────────────────────────────────────────

def demo_cdf_technique():
    print("=" * 60)
    print("  CDF Technique: Y = g(X)")
    print("=" * 60)

    # X ~ Uniform(0,1), Y = -ln(X)/λ → Y ~ Exponential(λ)
    random.seed(42)
    n = 50000
    lam = 2.0
    uniform_samples = [random.random() for _ in range(n)]
    exp_samples = [-math.log(u) / lam for u in uniform_samples]

    mean_sim = sum(exp_samples) / n
    var_sim = sum((x - mean_sim)**2 for x in exp_samples) / n

    print(f"\n  X ~ Uniform(0,1), Y = -ln(X)/{lam}")
    print(f"  Theory: Y ~ Exponential(λ={lam})")
    print(f"  E[Y] = {mean_sim:.4f} (theory: {1/lam:.4f})")
    print(f"  Var(Y) = {var_sim:.4f} (theory: {1/lam**2:.4f})")

    # X ~ Uniform(0,1), Y = X² → f_Y(y) = 1/(2√y) on [0,1]
    sq_samples = [u**2 for u in uniform_samples]
    mean_sq = sum(sq_samples) / n
    print(f"\n  X ~ Uniform(0,1), Y = X²")
    print(f"  E[Y] = {mean_sq:.4f} (theory: {1/3:.4f})")
    print(f"  f_Y(y) = 1/(2√y) for 0 < y < 1")


# ─────────────────────────────────────────────────
# 2. PDF TRANSFORMATION WITH JACOBIAN
# ─────────────────────────────────────────────────

def demo_jacobian():
    print("\n" + "=" * 60)
    print("  PDF Transformation (Jacobian)")
    print("=" * 60)

    print("""
  For monotonic g: f_Y(y) = f_X(g⁻¹(y)) · |d(g⁻¹)/dy|

  Example 1: X ~ N(0,1), Y = eˣ (log-normal)
    g⁻¹(y) = ln(y), |d(g⁻¹)/dy| = 1/y
    f_Y(y) = (1/y) · φ(ln(y))  for y > 0
""")

    # Verify via simulation
    random.seed(42)
    n = 50000
    normal_samples = [random.gauss(0, 1) for _ in range(n)]
    lognormal_samples = [math.exp(x) for x in normal_samples]

    mean_ln = sum(lognormal_samples) / n
    th_mean = math.exp(0.5)  # E[e^X] = e^{μ+σ²/2}
    print(f"  Log-normal simulation:")
    print(f"  E[Y] = {mean_ln:.4f} (theory: e^0.5 = {th_mean:.4f})")

    # Example 2: X ~ Uniform(0,π), Y = sin(X)
    print(f"\n  Example 2: X ~ Uniform(0,π), Y = sin(X)")
    sin_samples = [math.sin(random.uniform(0, math.pi)) for _ in range(n)]
    mean_sin = sum(sin_samples) / n
    print(f"  E[sin(X)] = {mean_sin:.4f} (theory: 2/π = {2/math.pi:.4f})")


# ─────────────────────────────────────────────────
# 3. BOX-MULLER TRANSFORM
# ─────────────────────────────────────────────────

def box_muller() -> tuple[float, float]:
    """Generate two independent N(0,1) from two Uniform(0,1)."""
    u1 = random.random()
    u2 = random.random()
    r = math.sqrt(-2 * math.log(u1))
    theta = 2 * math.pi * u2
    z1 = r * math.cos(theta)
    z2 = r * math.sin(theta)
    return z1, z2


def demo_box_muller():
    print("\n" + "=" * 60)
    print("  Box-Muller Transform")
    print("=" * 60)

    print("""
  U₁, U₂ ~ Uniform(0,1) independently
  Z₁ = √(-2 ln U₁) · cos(2πU₂)
  Z₂ = √(-2 ln U₁) · sin(2πU₂)
  → Z₁, Z₂ ~ N(0,1) independently
""")

    random.seed(42)
    n = 50000
    z1_samples, z2_samples = [], []
    for _ in range(n):
        z1, z2 = box_muller()
        z1_samples.append(z1)
        z2_samples.append(z2)

    for name, samples in [("Z₁", z1_samples), ("Z₂", z2_samples)]:
        mean = sum(samples) / n
        var = sum((x - mean)**2 for x in samples) / n
        print(f"  {name}: mean={mean:>7.4f} (→0), var={var:>6.4f} (→1)")

    # Check independence: correlation ≈ 0
    m1 = sum(z1_samples) / n
    m2 = sum(z2_samples) / n
    cov = sum((a - m1)*(b - m2) for a, b in zip(z1_samples, z2_samples)) / n
    print(f"  Cov(Z₁,Z₂) = {cov:.4f} (→0)")


# ─────────────────────────────────────────────────
# 4. ORDER STATISTICS
# ─────────────────────────────────────────────────

def order_stat_pdf_uniform(x: float, k: int, n: int) -> float:
    """PDF of k-th order statistic from Uniform(0,1).
    f_{X(k)}(x) = k·C(n,k)·x^{k-1}·(1-x)^{n-k} for 0<x<1.
    This is Beta(k, n-k+1).
    """
    if x <= 0 or x >= 1:
        return 0
    return (math.factorial(n) / (math.factorial(k-1) * math.factorial(n-k))
            * x**(k-1) * (1-x)**(n-k))


def demo_order_statistics():
    print("\n" + "=" * 60)
    print("  Order Statistics")
    print("=" * 60)

    random.seed(42)
    n_samples = 10000
    n = 5  # sample size

    print(f"\n  Sample size n={n} from Uniform(0,1)")
    print(f"  X₍₁₎ = min, X₍₃₎ = median, X₍₅₎ = max")

    # Simulate
    stats = {1: [], 3: [], 5: []}
    for _ in range(n_samples):
        sample = sorted([random.random() for _ in range(n)])
        for k in stats:
            stats[k].append(sample[k-1])

    print(f"\n  {'Stat':>8} {'E[sim]':>8} {'E[theory]':>10} {'Var[sim]':>10}")
    print(f"  {'─' * 38}")
    for k in [1, 3, 5]:
        sim_mean = sum(stats[k]) / n_samples
        sim_var = sum((x - sim_mean)**2 for x in stats[k]) / n_samples
        # E[X(k)] = k/(n+1) for Uniform(0,1)
        th_mean = k / (n + 1)
        print(f"  X₍{k}₎{'':<4} {sim_mean:>8.4f} {th_mean:>10.4f} {sim_var:>10.4f}")

    # Range: R = X(n) - X(1)
    ranges = [stats[5][i] - stats[1][i] for i in range(n_samples)]
    r_mean = sum(ranges) / n_samples
    th_r_mean = (n - 1) / (n + 1)
    print(f"\n  Range R = X₍₅₎ - X₍₁₎:")
    print(f"  E[R] = {r_mean:.4f} (theory: (n-1)/(n+1) = {th_r_mean:.4f})")

    # Median of odd sample
    print(f"\n  Sample median X₍₃₎ ~ Beta(3,3):")
    print(f"  E[median] = 0.5 (symmetric), simulated = {sum(stats[3])/n_samples:.4f}")


if __name__ == "__main__":
    demo_cdf_technique()
    demo_jacobian()
    demo_box_muller()
    demo_order_statistics()

"""
Continuous Distribution Families

Demonstrates:
1. Uniform and Normal Distributions
2. Exponential and Gamma
3. Beta Distribution
4. Chi-squared, t, and F Distributions

Theory:
- Normal: f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
- Exponential: f(x) = λe^{-λx}, memoryless
- Gamma: f(x) = (β^α/Γ(α)) x^{α-1} e^{-βx}
- Beta: f(x) = x^{α-1}(1-x)^{β-1} / B(α,β)

Adapted from Probability and Statistics Lesson 07.
"""

import math
import random


# ─────────────────────────────────────────────────
# HELPER: Numerical Integration
# ─────────────────────────────────────────────────

def integrate(f, a: float, b: float, n: int = 2000) -> float:
    """Simpson's rule."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * f(x)
    return s * h / 3


def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ─────────────────────────────────────────────────
# 1. UNIFORM AND NORMAL
# ─────────────────────────────────────────────────

def uniform_pdf(x: float, a: float, b: float) -> float:
    return 1 / (b - a) if a <= x <= b else 0


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))


def demo_uniform_normal():
    print("=" * 60)
    print("  Uniform and Normal Distributions")
    print("=" * 60)

    # Uniform
    a, b = 2, 8
    mean = (a + b) / 2
    var = (b - a)**2 / 12
    print(f"\n  Uniform({a}, {b}): E[X]={mean}, Var={var:.2f}")

    # Normal properties
    print(f"\n  Standard Normal N(0,1):")
    print(f"  P(Z<0)     = {phi(0):.4f}")
    print(f"  P(Z<1.96)  = {phi(1.96):.4f}")
    print(f"  P(-1<Z<1)  = {phi(1) - phi(-1):.4f}  (68.27%)")
    print(f"  P(-2<Z<2)  = {phi(2) - phi(-2):.4f}  (95.45%)")
    print(f"  P(-3<Z<3)  = {phi(3) - phi(-3):.4f}  (99.73%)")

    # Visualize Normal PDF
    print(f"\n  N(0,1) PDF:")
    for row in range(20, 0, -1):
        threshold = row * 0.02
        line = "  "
        for col in range(-30, 31):
            x = col / 10
            if normal_pdf(x) >= threshold:
                line += "█"
            else:
                line += " "
        if any(c != " " for c in line[2:]):
            print(line)


# ─────────────────────────────────────────────────
# 2. EXPONENTIAL AND GAMMA
# ─────────────────────────────────────────────────

def exponential_pdf(x: float, lam: float) -> float:
    return lam * math.exp(-lam * x) if x >= 0 else 0


def gamma_pdf(x: float, alpha: float, beta: float) -> float:
    """f(x) = (β^α / Γ(α)) x^{α-1} e^{-βx}."""
    if x <= 0:
        return 0
    log_pdf = alpha * math.log(beta) - math.lgamma(alpha) + (alpha - 1) * math.log(x) - beta * x
    return math.exp(log_pdf)


def demo_exponential_gamma():
    print("\n" + "=" * 60)
    print("  Exponential and Gamma Distributions")
    print("=" * 60)

    # Exponential
    lam = 0.5
    print(f"\n  Exponential(λ={lam}): E[X]={1/lam}, Var={1/lam**2}")
    print(f"  Memoryless: P(X>s+t|X>s) = P(X>t)")

    print(f"\n  CDF F(x) = 1 - e^(-λx):")
    for x in [1, 2, 3, 5, 10]:
        cdf = 1 - math.exp(-lam * x)
        print(f"  F({x:>2}) = {cdf:.4f}")

    # Gamma
    print(f"\n  Gamma family (rate parameterization):")
    configs = [(1, 1, "= Exponential(1)"),
               (2, 1, "= Erlang(2,1)"),
               (5, 1, ""), (3, 2, "")]
    for alpha, beta, note in configs:
        mean = alpha / beta
        var = alpha / beta**2
        print(f"  Gamma(α={alpha},β={beta}): E={mean:.1f}, Var={var:.2f} {note}")

    # Simulate: sum of exponentials → Gamma
    random.seed(42)
    n = 10000
    alpha = 5
    lam_g = 2.0
    gamma_samples = [sum(random.expovariate(lam_g) for _ in range(alpha))
                     for _ in range(n)]
    sim_mean = sum(gamma_samples) / n
    sim_var = sum((x - sim_mean)**2 for x in gamma_samples) / n
    print(f"\n  Gamma({alpha},{lam_g}) via sum of {alpha} Exp({lam_g}):")
    print(f"  Theory: E={alpha/lam_g:.2f}, Var={alpha/lam_g**2:.2f}")
    print(f"  Sim:    E={sim_mean:.2f}, Var={sim_var:.2f}")


# ─────────────────────────────────────────────────
# 3. BETA DISTRIBUTION
# ─────────────────────────────────────────────────

def beta_pdf(x: float, alpha: float, beta_param: float) -> float:
    """f(x) = x^{α-1}(1-x)^{β-1} / B(α,β)."""
    if x <= 0 or x >= 1:
        return 0
    log_pdf = ((alpha - 1) * math.log(x) + (beta_param - 1) * math.log(1 - x)
               - math.lgamma(alpha) - math.lgamma(beta_param)
               + math.lgamma(alpha + beta_param))
    return math.exp(log_pdf)


def demo_beta():
    print("\n" + "=" * 60)
    print("  Beta Distribution")
    print("=" * 60)

    configs = [
        (1, 1, "Uniform(0,1)"),
        (2, 5, "Left-skewed"),
        (5, 2, "Right-skewed"),
        (5, 5, "Symmetric, peaked"),
        (0.5, 0.5, "U-shaped"),
    ]

    print(f"\n  {'Config':<14} {'E[X]':>6} {'Var':>8} {'Mode':>6} {'Shape'}")
    print(f"  {'─' * 50}")
    for a, b, shape in configs:
        mean = a / (a + b)
        var = (a * b) / ((a + b)**2 * (a + b + 1))
        mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else float('nan')
        tag = f"Beta({a},{b})"
        print(f"  {tag:<14} {mean:>6.3f} {var:>8.5f} {mode:>6.3f} {shape}")

    # Bayesian coin: Beta prior updating
    print(f"\n  Bayesian coin flip:")
    alpha, beta_p = 1, 1  # Uniform prior
    flips = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # 7 heads, 3 tails
    print(f"  Prior: Beta({alpha},{beta_p}), E[p] = {alpha/(alpha+beta_p):.3f}")
    for i, flip in enumerate(flips, 1):
        if flip == 1:
            alpha += 1
        else:
            beta_p += 1
        if i in [1, 5, 10]:
            print(f"  After {i:>2} flips: Beta({alpha},{beta_p}), "
                  f"E[p] = {alpha/(alpha+beta_p):.3f}")


# ─────────────────────────────────────────────────
# 4. CHI-SQUARED, t, AND F DISTRIBUTIONS
# ─────────────────────────────────────────────────

def demo_chi_t_f():
    print("\n" + "=" * 60)
    print("  Chi-squared, t, and F Distributions")
    print("=" * 60)

    # Chi-squared: sum of squared standard normals
    random.seed(42)
    n = 10000

    print(f"\n  Chi-squared(k): sum of k standard normal² values")
    for k in [1, 2, 5, 10, 30]:
        samples = [sum(random.gauss(0, 1)**2 for _ in range(k))
                   for _ in range(n)]
        sim_mean = sum(samples) / n
        sim_var = sum((x - sim_mean)**2 for x in samples) / n
        print(f"  χ²({k:>2}): E={sim_mean:>6.2f} (th:{k}), "
              f"Var={sim_var:>7.2f} (th:{2*k})")

    # t-distribution: Z / √(χ²/k)
    print(f"\n  Student's t(ν): Z / √(χ²_ν/ν)")
    for nu in [1, 5, 10, 30]:
        samples = []
        for _ in range(n):
            z = random.gauss(0, 1)
            chi2 = sum(random.gauss(0, 1)**2 for _ in range(nu))
            samples.append(z / math.sqrt(chi2 / nu))
        sim_var = sum(x**2 for x in samples) / n
        th_var = nu / (nu - 2) if nu > 2 else float('inf')
        print(f"  t({nu:>2}): Var={sim_var:>6.2f} (th:{th_var:>6.2f}), "
              f"heavier tails than Normal")

    # F-distribution: (χ²_d1/d1) / (χ²_d2/d2)
    print(f"\n  F(d1,d2): (χ²_d1/d1) / (χ²_d2/d2)")
    for d1, d2 in [(5, 10), (10, 20), (1, 30)]:
        samples = []
        for _ in range(n):
            c1 = sum(random.gauss(0, 1)**2 for _ in range(d1)) / d1
            c2 = sum(random.gauss(0, 1)**2 for _ in range(d2)) / d2
            samples.append(c1 / c2)
        sim_mean = sum(samples) / n
        th_mean = d2 / (d2 - 2) if d2 > 2 else float('inf')
        print(f"  F({d1},{d2}): E={sim_mean:>5.2f} (th:{th_mean:>5.2f})")

    print(f"\n  Relationships:")
    print(f"  t²(ν) = F(1,ν)")
    print(f"  χ²(k)/k → 1 as k→∞ (by LLN)")
    print(f"  t(ν) → N(0,1) as ν→∞")


if __name__ == "__main__":
    demo_uniform_normal()
    demo_exponential_gamma()
    demo_beta()
    demo_chi_t_f()

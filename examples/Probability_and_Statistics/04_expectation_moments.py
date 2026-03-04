"""
Expectation and Moments

Demonstrates:
1. Expected Value Calculator (Discrete and Continuous)
2. Moment Generating Function (MGF)
3. Chebyshev's Inequality Demonstration
4. Skewness and Kurtosis

Theory:
- E[X] = Σ x·p(x) (discrete) or ∫ x·f(x)dx (continuous)
- Var(X) = E[X²] - (E[X])²
- MGF: M(t) = E[e^{tX}], E[X^n] = M^(n)(0)
- Chebyshev: P(|X-μ| ≥ kσ) ≤ 1/k²

Adapted from Probability and Statistics Lesson 04.
"""

import math
import random


# ─────────────────────────────────────────────────
# 1. EXPECTED VALUE CALCULATOR
# ─────────────────────────────────────────────────

def expected_value_discrete(pmf: dict[float, float]) -> float:
    """E[X] = Σ x·P(X=x)."""
    return sum(x * p for x, p in pmf.items())


def variance_discrete(pmf: dict[float, float]) -> float:
    """Var(X) = E[X²] - (E[X])²."""
    ex = expected_value_discrete(pmf)
    ex2 = sum(x ** 2 * p for x, p in pmf.items())
    return ex2 - ex ** 2


def expected_value_continuous(f, a: float, b: float,
                               n: int = 2000) -> float:
    """E[X] = ∫ x·f(x)dx via Simpson's rule."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    g = lambda x: x * f(x)
    total = g(a) + g(b)
    for i in range(1, n):
        x = a + i * h
        total += (4 if i % 2 == 1 else 2) * g(x)
    return total * h / 3


def demo_expectation():
    print("=" * 60)
    print("  Expected Value and Variance")
    print("=" * 60)

    # Fair die
    die_pmf = {i: 1/6 for i in range(1, 7)}
    ex = expected_value_discrete(die_pmf)
    var = variance_discrete(die_pmf)
    print(f"\n  Fair die:")
    print(f"  E[X] = {ex:.4f} (theory: 3.5)")
    print(f"  Var(X) = {var:.4f} (theory: {35/12:.4f})")

    # Loaded die
    loaded = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.5}
    ex_l = expected_value_discrete(loaded)
    var_l = variance_discrete(loaded)
    print(f"\n  Loaded die (P(6)=0.5):")
    print(f"  E[X] = {ex_l:.4f}")
    print(f"  Var(X) = {var_l:.4f}")

    # Continuous: Exponential(λ=2) on [0, 20]
    lam = 2.0
    pdf = lambda x: lam * math.exp(-lam * x)
    ex_c = expected_value_continuous(pdf, 0, 20)
    print(f"\n  Exponential(λ={lam}):")
    print(f"  E[X] = {ex_c:.4f} (theory: {1/lam:.4f})")

    # Linearity of expectation
    print(f"\n  Linearity: E[aX+b] = a·E[X] + b")
    a, b = 3, 5
    print(f"  E[{a}X+{b}] for fair die = {a}·{ex:.1f}+{b} = {a*ex+b:.1f}")


# ─────────────────────────────────────────────────
# 2. MOMENT GENERATING FUNCTION
# ─────────────────────────────────────────────────

def mgf_discrete(pmf: dict[float, float], t: float) -> float:
    """M(t) = E[e^{tX}] = Σ e^{tx} · P(X=x)."""
    return sum(math.exp(t * x) * p for x, p in pmf.items())


def moments_from_mgf(pmf: dict[float, float],
                      order: int = 4, h: float = 1e-5) -> list[float]:
    """Extract moments via numerical differentiation of MGF at t=0."""
    moments = []
    for k in range(1, order + 1):
        # Central difference for k-th derivative
        if k == 1:
            m = (mgf_discrete(pmf, h) - mgf_discrete(pmf, -h)) / (2 * h)
        elif k == 2:
            m = (mgf_discrete(pmf, h) - 2 * mgf_discrete(pmf, 0) + mgf_discrete(pmf, -h)) / (h ** 2)
        elif k == 3:
            m = (mgf_discrete(pmf, 2*h) - 2*mgf_discrete(pmf, h) + 2*mgf_discrete(pmf, -h) - mgf_discrete(pmf, -2*h)) / (2 * h**3)
        elif k == 4:
            m = (mgf_discrete(pmf, 2*h) - 4*mgf_discrete(pmf, h) + 6*mgf_discrete(pmf, 0) - 4*mgf_discrete(pmf, -h) + mgf_discrete(pmf, -2*h)) / (h**4)
        else:
            m = 0
        moments.append(m)
    return moments


def demo_mgf():
    print("\n" + "=" * 60)
    print("  Moment Generating Function")
    print("=" * 60)

    die_pmf = {i: 1/6 for i in range(1, 7)}

    print(f"\n  Fair die MGF M(t) at various t:")
    for t in [0.0, 0.1, 0.5, 1.0]:
        m = mgf_discrete(die_pmf, t)
        print(f"  M({t}) = {m:.6f}")

    print(f"\n  Moments extracted from MGF (numerical):")
    moments = moments_from_mgf(die_pmf, order=4)
    labels = ["E[X]", "E[X²]", "E[X³]", "E[X⁴]"]
    direct = [
        sum(x * p for x, p in die_pmf.items()),
        sum(x**2 * p for x, p in die_pmf.items()),
        sum(x**3 * p for x, p in die_pmf.items()),
        sum(x**4 * p for x, p in die_pmf.items()),
    ]
    for label, mgf_val, dir_val in zip(labels, moments, direct):
        print(f"  {label}: MGF = {mgf_val:.4f}, Direct = {dir_val:.4f}")


# ─────────────────────────────────────────────────
# 3. CHEBYSHEV'S INEQUALITY
# ─────────────────────────────────────────────────

def chebyshev_bound(k: float) -> float:
    """P(|X-μ| ≥ kσ) ≤ 1/k²."""
    return 1.0 / (k ** 2)


def demo_chebyshev():
    print("\n" + "=" * 60)
    print("  Chebyshev's Inequality")
    print("=" * 60)

    print(f"\n  Chebyshev bound: P(|X-μ| ≥ kσ) ≤ 1/k²")
    print(f"  {'k':>4} {'Bound':>10} {'Normal (actual)':>16}")
    print(f"  {'─' * 32}")

    for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        bound = chebyshev_bound(k)
        # For comparison: actual probability for Normal
        # P(|Z| ≥ k) computed via simulation
        normal_actual = 2 * (1 - _phi(k))
        print(f"  {k:>4.1f} {bound:>10.4f} {normal_actual:>16.6f}")

    # Simulation
    print(f"\n  Simulation: Uniform(0,1) — μ=0.5, σ²=1/12")
    random.seed(42)
    n = 100_000
    samples = [random.random() for _ in range(n)]
    mu = 0.5
    sigma = math.sqrt(1/12)

    for k in [1.0, 2.0, 3.0]:
        actual = sum(1 for x in samples if abs(x - mu) >= k * sigma) / n
        bound = chebyshev_bound(k)
        print(f"  k={k:.0f}: Actual={actual:.4f}, Bound={bound:.4f}")


def _phi(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ─────────────────────────────────────────────────
# 4. SKEWNESS AND KURTOSIS
# ─────────────────────────────────────────────────

def compute_moments(data: list[float]) -> dict:
    """Compute mean, variance, skewness, kurtosis from data."""
    n = len(data)
    mean = sum(data) / n
    m2 = sum((x - mean) ** 2 for x in data) / n
    m3 = sum((x - mean) ** 3 for x in data) / n
    m4 = sum((x - mean) ** 4 for x in data) / n
    sd = math.sqrt(m2)

    skewness = m3 / (sd ** 3) if sd > 0 else 0
    kurtosis = m4 / (sd ** 4) if sd > 0 else 0
    excess_kurtosis = kurtosis - 3  # Normal has kurtosis = 3

    return {
        "mean": mean, "variance": m2, "std": sd,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": excess_kurtosis,
    }


def demo_skewness_kurtosis():
    print("\n" + "=" * 60)
    print("  Skewness and Kurtosis")
    print("=" * 60)

    random.seed(42)
    n = 50000

    distributions = {
        "Normal(0,1)": [random.gauss(0, 1) for _ in range(n)],
        "Exponential(1)": [random.expovariate(1) for _ in range(n)],
        "Uniform(0,1)": [random.random() for _ in range(n)],
    }

    print(f"\n  {'Distribution':<18} {'Skew':>8} {'Excess Kurt':>12} {'Shape':>15}")
    print(f"  {'─' * 55}")
    for name, data in distributions.items():
        m = compute_moments(data)
        shape = ("symmetric" if abs(m["skewness"]) < 0.3
                 else "right-skewed" if m["skewness"] > 0
                 else "left-skewed")
        print(f"  {name:<18} {m['skewness']:>8.3f} {m['excess_kurtosis']:>12.3f} {shape:>15}")

    print(f"\n  Reference values:")
    print(f"  Normal:      skew=0, excess_kurt=0 (mesokurtic)")
    print(f"  Exponential: skew=2, excess_kurt=6 (leptokurtic)")
    print(f"  Uniform:     skew=0, excess_kurt=-1.2 (platykurtic)")


if __name__ == "__main__":
    demo_expectation()
    demo_mgf()
    demo_chebyshev()
    demo_skewness_kurtosis()

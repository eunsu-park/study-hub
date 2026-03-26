"""
Random Variables and Distributions

Demonstrates:
1. PMF Simulator (Discrete)
2. PDF/CDF via Numerical Integration (Continuous)
3. CDF Properties and Quantile Function
4. Distribution Comparison (Simulation)

Theory:
- PMF: p(x) = P(X = x), Σp(x) = 1
- PDF: f(x) ≥ 0, ∫f(x)dx = 1, P(a≤X≤b) = ∫ₐᵇf(x)dx
- CDF: F(x) = P(X ≤ x), non-decreasing, right-continuous

Adapted from Probability and Statistics Lesson 03.
"""

import random
import math
from collections import Counter


# ─────────────────────────────────────────────────
# 1. DISCRETE PMF SIMULATOR
# ─────────────────────────────────────────────────

def simulate_pmf(pmf: dict[int, float], n: int = 10000) -> dict:
    """Simulate samples and compare with theoretical PMF."""
    values = list(pmf.keys())
    weights = list(pmf.values())
    samples = random.choices(values, weights=weights, k=n)
    counts = Counter(samples)
    empirical = {v: counts.get(v, 0) / n for v in values}
    return {"theoretical": pmf, "empirical": empirical}


def demo_discrete():
    print("=" * 60)
    print("  Discrete Random Variable (PMF)")
    print("=" * 60)

    # Custom die (loaded)
    pmf = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.5}
    random.seed(42)
    result = simulate_pmf(pmf, n=50000)

    print(f"\n  Loaded die PMF (P(6) = 0.5):")
    print(f"  {'x':>4} {'P(X=x)':>10} {'Empirical':>10} {'Error':>10}")
    print(f"  {'─' * 36}")
    for x in sorted(pmf):
        th = result["theoretical"][x]
        em = result["empirical"].get(x, 0)
        print(f"  {x:>4} {th:>10.4f} {em:>10.4f} {abs(th-em):>10.4f}")

    # CDF from PMF
    print(f"\n  CDF F(x) = P(X ≤ x):")
    cumsum = 0.0
    for x in sorted(pmf):
        cumsum += pmf[x]
        print(f"  F({x}) = {cumsum:.2f}")


# ─────────────────────────────────────────────────
# 2. CONTINUOUS PDF AND CDF
# ─────────────────────────────────────────────────

def numerical_integrate(f, a: float, b: float,
                        n: int = 1000) -> float:
    """Simpson's rule numerical integration."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += (4 if i % 2 == 1 else 2) * f(x)
    return total * h / 3


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def exponential_pdf(x: float, lam: float = 1) -> float:
    """Exponential PDF."""
    return lam * math.exp(-lam * x) if x >= 0 else 0


def demo_continuous():
    print("\n" + "=" * 60)
    print("  Continuous Random Variable (PDF/CDF)")
    print("=" * 60)

    # Normal distribution
    print(f"\n  Standard Normal N(0,1):")
    print(f"  {'x':>6} {'f(x)':>10} {'F(x)':>10}")
    print(f"  {'─' * 28}")
    for x_val in [-3, -2, -1, 0, 1, 2, 3]:
        fx = normal_pdf(x_val)
        Fx = numerical_integrate(lambda t: normal_pdf(t), -8, x_val)
        print(f"  {x_val:>6} {fx:>10.6f} {Fx:>10.6f}")

    # Key probabilities
    print(f"\n  68-95-99.7 rule check:")
    for k in [1, 2, 3]:
        p = numerical_integrate(lambda t: normal_pdf(t), -k, k)
        print(f"  P(-{k} < Z < {k}) = {p:.4f}")

    # Exponential distribution
    print(f"\n  Exponential(λ=2) — CDF: F(x) = 1 - e^(-λx)")
    lam = 2.0
    for x_val in [0.5, 1.0, 2.0, 5.0]:
        fx = exponential_pdf(x_val, lam)
        Fx = 1 - math.exp(-lam * x_val)  # closed form
        print(f"  F({x_val}) = {Fx:.4f}  (PDF f({x_val}) = {fx:.4f})")


# ─────────────────────────────────────────────────
# 3. CDF PROPERTIES AND QUANTILE FUNCTION
# ─────────────────────────────────────────────────

def quantile_normal(p: float, mu: float = 0, sigma: float = 1,
                    tol: float = 1e-8) -> float:
    """Inverse CDF of normal distribution (bisection method)."""
    lo, hi = mu - 6 * sigma, mu + 6 * sigma
    while hi - lo > tol:
        mid = (lo + hi) / 2
        cdf = numerical_integrate(lambda t: normal_pdf(t, mu, sigma), mu - 6*sigma, mid)
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def demo_quantile():
    print("\n" + "=" * 60)
    print("  Quantile Function (Inverse CDF)")
    print("=" * 60)

    print(f"\n  Standard Normal Quantiles:")
    for p in [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]:
        q = quantile_normal(p)
        print(f"  Q({p:.3f}) = z = {q:>7.4f}")

    # Exponential quantile (closed form)
    print(f"\n  Exponential(λ=1) Quantiles: Q(p) = -ln(1-p)/λ")
    for p in [0.25, 0.50, 0.75, 0.90, 0.95]:
        q = -math.log(1 - p)
        print(f"  Q({p:.2f}) = {q:.4f}")


# ─────────────────────────────────────────────────
# 4. DISTRIBUTION COMPARISON (SIMULATION)
# ─────────────────────────────────────────────────

def ascii_histogram(data: list[float], bins: int = 20,
                    width: int = 40) -> list[str]:
    """Create ASCII histogram."""
    lo = min(data)
    hi = max(data)
    bin_width = (hi - lo) / bins
    counts = [0] * bins
    for x in data:
        idx = min(int((x - lo) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts) if counts else 1
    lines = []
    for i, c in enumerate(counts):
        left = lo + i * bin_width
        bar_len = int(c / max_count * width)
        lines.append(f"  [{left:>6.2f}] {'█' * bar_len} {c}")
    return lines


def demo_comparison():
    print("\n" + "=" * 60)
    print("  Distribution Comparison (Simulation)")
    print("=" * 60)

    random.seed(42)
    n = 10000

    # Uniform vs Normal vs Exponential
    uniform = [random.uniform(0, 10) for _ in range(n)]
    normal = [random.gauss(5, 1.5) for _ in range(n)]
    exponential = [random.expovariate(0.5) for _ in range(n)]

    for name, data in [("Uniform(0,10)", uniform),
                       ("Normal(5,1.5²)", normal),
                       ("Exponential(λ=0.5)", exponential)]:
        mean = sum(data) / len(data)
        var = sum((x - mean) ** 2 for x in data) / len(data)
        print(f"\n  {name}:")
        print(f"  Mean = {mean:.3f}, Var = {var:.3f}, SD = {math.sqrt(var):.3f}")
        for line in ascii_histogram(data, bins=15):
            print(line)


if __name__ == "__main__":
    demo_discrete()
    demo_continuous()
    demo_quantile()
    demo_comparison()

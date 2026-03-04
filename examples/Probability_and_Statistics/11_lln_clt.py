"""
Law of Large Numbers and Central Limit Theorem

Demonstrates:
1. Weak Law of Large Numbers (WLLN)
2. Strong Law of Large Numbers (SLLN)
3. Central Limit Theorem (CLT) Visualization
4. Normal Approximation with Continuity Correction

Theory:
- WLLN: X̄n → μ in probability
- SLLN: X̄n → μ almost surely
- CLT: √n(X̄n - μ)/σ → N(0,1) in distribution
- Berry-Esseen: |Fn(x) - Φ(x)| ≤ Cρ/(σ³√n) where ρ = E[|X-μ|³]

Adapted from Probability and Statistics Lesson 11.
"""

import random
import math


def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ─────────────────────────────────────────────────
# 1. WEAK LAW OF LARGE NUMBERS
# ─────────────────────────────────────────────────

def demo_wlln():
    print("=" * 60)
    print("  Weak Law of Large Numbers")
    print("=" * 60)

    random.seed(42)

    # Various distributions, all showing X̄n → μ
    distributions = [
        ("Uniform(0,1)", lambda: random.random(), 0.5),
        ("Exponential(2)", lambda: random.expovariate(2), 0.5),
        ("Bernoulli(0.3)", lambda: 1 if random.random() < 0.3 else 0, 0.3),
    ]

    for name, gen, mu in distributions:
        print(f"\n  {name}: μ = {mu}")
        print(f"  {'n':>8} {'X̄n':>10} {'|X̄n - μ|':>12}")
        print(f"  {'─' * 32}")

        total = 0
        for n in [10, 100, 1000, 10000, 100000]:
            while len(range(n)) > 0:
                # generate up to n
                total = sum(gen() for _ in range(n))
                break
            x_bar = total / n
            print(f"  {n:>8} {x_bar:>10.6f} {abs(x_bar - mu):>12.6f}")

    # Monte Carlo integration: ∫₀¹ sin(πx) dx = 2/π
    print(f"\n  Monte Carlo: ∫₀¹ sin(πx) dx")
    true_val = 2 / math.pi
    for n in [100, 1000, 10000, 100000]:
        est = sum(math.sin(math.pi * random.random()) for _ in range(n)) / n
        print(f"  n={n:>6}: {est:.6f} (true: {true_val:.6f}, err: {abs(est-true_val):.6f})")


# ─────────────────────────────────────────────────
# 2. STRONG LAW OF LARGE NUMBERS
# ─────────────────────────────────────────────────

def demo_slln():
    print("\n" + "=" * 60)
    print("  Strong Law: Sample Paths")
    print("=" * 60)

    random.seed(42)
    max_n = 500
    n_paths = 4
    mu = 0.5  # Uniform(0,1) mean

    # Generate paths
    paths = []
    for _ in range(n_paths):
        running_sum = 0
        path = []
        for i in range(1, max_n + 1):
            running_sum += random.random()
            path.append(running_sum / i)
        paths.append(path)

    # ASCII visualization
    print(f"\n  X̄n vs n for Uniform(0,1), μ=0.5:")
    rows = 15
    for r in range(rows, 0, -1):
        val = 0.3 + (r / rows) * 0.4  # range [0.3, 0.7]
        line = f"  {val:.2f} |"
        for n_idx in range(0, max_n, max_n // 50):
            chars = []
            for p_idx, path in enumerate(paths):
                if abs(path[n_idx] - val) < 0.015:
                    chars.append(str(p_idx + 1))
            line += chars[0] if chars else " "
        print(line)

    print(f"  {'─' * 55}")
    print(f"       n=1{'':>23}n={max_n//2}{'':>20}n={max_n}")
    print(f"\n  All paths converge to μ = 0.5 (a.s. convergence)")


# ─────────────────────────────────────────────────
# 3. CENTRAL LIMIT THEOREM VISUALIZATION
# ─────────────────────────────────────────────────

def demo_clt():
    print("\n" + "=" * 60)
    print("  Central Limit Theorem")
    print("=" * 60)

    random.seed(42)
    mu_exp = 1.0  # Exponential(1) mean
    sigma_exp = 1.0  # Exponential(1) std

    for n in [2, 5, 20, 50]:
        print(f"\n  n = {n} (Exponential(1) → standardized):")

        z_samples = []
        for _ in range(10000):
            sample = [random.expovariate(1.0) for _ in range(n)]
            x_bar = sum(sample) / n
            z = math.sqrt(n) * (x_bar - mu_exp) / sigma_exp
            z_samples.append(z)

        # Histogram
        bins = 20
        lo, hi = -4, 4
        bin_width = (hi - lo) / bins
        counts = [0] * bins
        for z in z_samples:
            idx = int((z - lo) / bin_width)
            if 0 <= idx < bins:
                counts[idx] += 1

        max_count = max(counts) if counts else 1
        for i, c in enumerate(counts):
            left = lo + i * bin_width
            bar_len = int(c / max_count * 30)
            # Expected from N(0,1)
            expected = 10000 * (phi(left + bin_width) - phi(left))
            exp_len = int(expected / max_count * 30)
            bar = "█" * bar_len
            exp_bar = "·" * max(0, exp_len - bar_len)
            print(f"  [{left:>5.1f}] {bar}{exp_bar}")

        # KS-like check
        z_samples.sort()
        max_diff = 0
        for i, z in enumerate(z_samples):
            empirical = (i + 1) / len(z_samples)
            theoretical = phi(z)
            max_diff = max(max_diff, abs(empirical - theoretical))
        print(f"  Max |Fn - Φ| = {max_diff:.4f}")


# ─────────────────────────────────────────────────
# 4. NORMAL APPROXIMATION
# ─────────────────────────────────────────────────

def demo_normal_approximation():
    print("\n" + "=" * 60)
    print("  Normal Approximation (Continuity Correction)")
    print("=" * 60)

    # Binomial(100, 0.3) → Normal(30, 21)
    n, p = 100, 0.3
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))

    print(f"\n  Binomial(n={n}, p={p}) ≈ N(μ={mu}, σ²={sigma**2:.1f})")

    # Exact binomial (using log for stability)
    def binom_pmf(k):
        return math.exp(math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
                       + k*math.log(p) + (n-k)*math.log(1-p))

    queries = [
        ("P(X ≤ 25)", lambda: sum(binom_pmf(k) for k in range(26)),
         lambda: phi((25.5 - mu) / sigma), lambda: phi((25 - mu) / sigma)),
        ("P(X = 30)", lambda: binom_pmf(30),
         lambda: phi((30.5 - mu)/sigma) - phi((29.5 - mu)/sigma), lambda: 0),
        ("P(X ≥ 35)", lambda: sum(binom_pmf(k) for k in range(35, n+1)),
         lambda: 1 - phi((34.5 - mu) / sigma), lambda: 1 - phi((35 - mu) / sigma)),
    ]

    print(f"\n  {'Query':<14} {'Exact':>10} {'w/ correction':>14} {'w/o':>10}")
    print(f"  {'─' * 50}")
    for name, exact_fn, corr_fn, no_corr_fn in queries:
        exact = exact_fn()
        with_corr = corr_fn()
        without = no_corr_fn()
        print(f"  {name:<14} {exact:>10.6f} {with_corr:>14.6f} {without:>10.6f}")

    print(f"\n  Continuity correction improves approximation for discrete → continuous")


if __name__ == "__main__":
    demo_wlln()
    demo_slln()
    demo_clt()
    demo_normal_approximation()

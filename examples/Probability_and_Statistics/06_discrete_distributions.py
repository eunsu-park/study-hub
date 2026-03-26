"""
Discrete Distribution Families

Demonstrates:
1. Bernoulli and Binomial
2. Poisson (with Binomial approximation)
3. Geometric (memoryless property)
4. Negative Binomial and Hypergeometric

Theory:
- Binomial(n,p): P(X=k) = C(n,k)p^k(1-p)^{n-k}
- Poisson(λ): P(X=k) = e^{-λ}λ^k/k!
- Geometric(p): P(X=k) = (1-p)^{k-1}p, memoryless
- Hypergeometric(N,K,n): P(X=k) = C(K,k)C(N-K,n-k)/C(N,n)

Adapted from Probability and Statistics Lesson 06.
"""

import math
import random
from collections import Counter


# ─────────────────────────────────────────────────
# 1. BERNOULLI AND BINOMIAL
# ─────────────────────────────────────────────────

def binomial_pmf(k: int, n: int, p: float) -> float:
    """P(X=k) = C(n,k) p^k (1-p)^{n-k}."""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def demo_binomial():
    print("=" * 60)
    print("  Bernoulli and Binomial Distribution")
    print("=" * 60)

    n, p = 10, 0.3
    mean = n * p
    var = n * p * (1 - p)
    print(f"\n  Binomial(n={n}, p={p}): E[X]={mean:.1f}, Var(X)={var:.2f}")

    print(f"\n  {'k':>4} {'P(X=k)':>10} {'P(X≤k)':>10} {'Bar'}")
    print(f"  {'─' * 45}")
    cdf = 0.0
    for k in range(n + 1):
        pmf = binomial_pmf(k, n, p)
        cdf += pmf
        bar = "█" * int(pmf * 80)
        print(f"  {k:>4} {pmf:>10.4f} {cdf:>10.4f} {bar}")

    # Simulation check
    random.seed(42)
    samples = [sum(random.random() < p for _ in range(n)) for _ in range(10000)]
    sim_mean = sum(samples) / len(samples)
    print(f"\n  Simulation (10000): mean={sim_mean:.2f} (theory: {mean:.1f})")


# ─────────────────────────────────────────────────
# 2. POISSON DISTRIBUTION
# ─────────────────────────────────────────────────

def poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) = e^{-λ} λ^k / k!"""
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def demo_poisson():
    print("\n" + "=" * 60)
    print("  Poisson Distribution")
    print("=" * 60)

    lam = 5.0
    print(f"\n  Poisson(λ={lam}): E[X]={lam}, Var(X)={lam}")

    print(f"\n  {'k':>4} {'P(X=k)':>10} {'Bar'}")
    print(f"  {'─' * 35}")
    for k in range(16):
        pmf = poisson_pmf(k, lam)
        bar = "█" * int(pmf * 60)
        print(f"  {k:>4} {pmf:>10.4f} {bar}")

    # Poisson approximation to Binomial
    print(f"\n  Poisson ≈ Binomial when n large, p small, np moderate:")
    n, p = 1000, 0.005
    lam_approx = n * p
    print(f"  Binomial(n={n}, p={p}) vs Poisson(λ={lam_approx})")
    print(f"  {'k':>4} {'Binomial':>10} {'Poisson':>10} {'|Error|':>10}")
    print(f"  {'─' * 36}")
    for k in range(11):
        b = binomial_pmf(k, n, p)
        po = poisson_pmf(k, lam_approx)
        print(f"  {k:>4} {b:>10.6f} {po:>10.6f} {abs(b-po):>10.6f}")


# ─────────────────────────────────────────────────
# 3. GEOMETRIC DISTRIBUTION
# ─────────────────────────────────────────────────

def geometric_pmf(k: int, p: float) -> float:
    """P(X=k) = (1-p)^{k-1} p, k=1,2,..."""
    return ((1 - p) ** (k - 1)) * p


def demo_geometric():
    print("\n" + "=" * 60)
    print("  Geometric Distribution (Memoryless)")
    print("=" * 60)

    p = 0.2
    mean = 1 / p
    var = (1 - p) / (p ** 2)
    print(f"\n  Geometric(p={p}): E[X]={mean:.1f}, Var(X)={var:.2f}")

    print(f"\n  {'k':>4} {'P(X=k)':>10} {'P(X≤k)':>10}")
    print(f"  {'─' * 26}")
    cdf = 0.0
    for k in range(1, 16):
        pmf = geometric_pmf(k, p)
        cdf += pmf
        print(f"  {k:>4} {pmf:>10.4f} {cdf:>10.4f}")

    # Memoryless property verification
    print(f"\n  Memoryless: P(X>s+t | X>s) = P(X>t)")
    s, t = 3, 2
    p_gt_s = (1 - p) ** s
    p_gt_st = (1 - p) ** (s + t)
    p_gt_t = (1 - p) ** t
    print(f"  P(X>{s+t} | X>{s}) = P(X>{s+t})/P(X>{s})")
    print(f"  = {p_gt_st:.4f} / {p_gt_s:.4f} = {p_gt_st/p_gt_s:.4f}")
    print(f"  P(X>{t}) = {p_gt_t:.4f}  ✓")


# ─────────────────────────────────────────────────
# 4. NEGATIVE BINOMIAL AND HYPERGEOMETRIC
# ─────────────────────────────────────────────────

def neg_binomial_pmf(k: int, r: int, p: float) -> float:
    """P(X=k) = C(k-1,r-1) p^r (1-p)^{k-r}, k=r,r+1,..."""
    if k < r:
        return 0.0
    return math.comb(k - 1, r - 1) * (p ** r) * ((1 - p) ** (k - r))


def hypergeometric_pmf(k: int, N: int, K: int, n: int) -> float:
    """P(X=k) = C(K,k)C(N-K,n-k) / C(N,n)."""
    if k < max(0, n - (N - K)) or k > min(K, n):
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def demo_neg_binomial():
    print("\n" + "=" * 60)
    print("  Negative Binomial Distribution")
    print("=" * 60)

    r, p = 3, 0.4
    mean = r / p
    var = r * (1 - p) / (p ** 2)
    print(f"\n  NegBin(r={r}, p={p}): E[X]={mean:.1f}, Var(X)={var:.2f}")
    print(f"  (Wait for {r}th success)")

    print(f"\n  {'k':>4} {'P(X=k)':>10} {'Bar'}")
    print(f"  {'─' * 35}")
    for k in range(r, 20):
        pmf = neg_binomial_pmf(k, r, p)
        bar = "█" * int(pmf * 60)
        print(f"  {k:>4} {pmf:>10.4f} {bar}")


def demo_hypergeometric():
    print("\n" + "=" * 60)
    print("  Hypergeometric Distribution")
    print("=" * 60)

    # Deck: 52 cards, 4 aces, draw 5
    N, K, n = 52, 4, 5
    mean = n * K / N
    print(f"\n  Cards: N={N}, K={K} aces, draw n={n}")
    print(f"  E[X] = nK/N = {mean:.3f}")

    print(f"\n  {'k':>4} {'P(X=k)':>10} {'Bar'}")
    print(f"  {'─' * 35}")
    for k in range(min(K, n) + 1):
        pmf = hypergeometric_pmf(k, N, K, n)
        bar = "█" * int(pmf * 60)
        print(f"  {k:>4} {pmf:>10.6f} {bar}")

    # Compare with Binomial (with replacement)
    p = K / N
    print(f"\n  Hypergeometric vs Binomial(n={n}, p={K}/{N}):")
    print(f"  {'k':>4} {'Hyper':>10} {'Binomial':>10}")
    print(f"  {'─' * 26}")
    for k in range(min(K, n) + 1):
        h = hypergeometric_pmf(k, N, K, n)
        b = binomial_pmf(k, n, p)
        print(f"  {k:>4} {h:>10.6f} {b:>10.6f}")


# ─────────────────────────────────────────────────
# 5. DISTRIBUTION RELATIONSHIPS
# ─────────────────────────────────────────────────

def demo_relationships():
    print("\n" + "=" * 60)
    print("  Distribution Relationships")
    print("=" * 60)

    print("""
  Bernoulli(p) ─── n trials ───→ Binomial(n,p)
       │                              │
       │                         n→∞, p→0
       │                         np=λ fixed
       │                              ↓
       └── wait for 1st ──→ Geometric(p)   Poisson(λ)
                │
           wait for r-th ──→ NegBin(r,p)

  Hypergeometric(N,K,n) → Binomial(n,p) as N→∞ with K/N→p
  Binomial(n,p) → Poisson(λ) as n→∞, p→0, np→λ
  Binomial(n,p) → Normal(np, np(1-p)) as n→∞ (CLT)
""")


if __name__ == "__main__":
    demo_binomial()
    demo_poisson()
    demo_geometric()
    demo_neg_binomial()
    demo_hypergeometric()
    demo_relationships()

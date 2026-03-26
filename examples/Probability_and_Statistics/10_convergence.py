"""
Convergence Concepts

Demonstrates:
1. Convergence in Probability
2. Almost Sure Convergence
3. Convergence in Distribution
4. Relationships between Convergence Types

Theory:
- Prob: P(|Xn-X| > ε) → 0 for all ε > 0
- a.s.: P(lim Xn = X) = 1
- Distribution: Fn(x) → F(x) at all continuity points
- Lp: E[|Xn-X|^p] → 0
- a.s. → prob → dist; Lp → prob

Adapted from Probability and Statistics Lesson 10.
"""

import random
import math


# ─────────────────────────────────────────────────
# 1. CONVERGENCE IN PROBABILITY
# ─────────────────────────────────────────────────

def demo_convergence_probability():
    print("=" * 60)
    print("  Convergence in Probability")
    print("=" * 60)

    # X̄n → μ (sample mean → true mean)
    random.seed(42)
    mu, sigma = 5.0, 2.0
    eps = 0.5

    print(f"\n  X ~ N({mu}, {sigma}²), X̄n → {mu} in probability")
    print(f"  P(|X̄n - μ| > {eps}) should → 0")
    print(f"\n  {'n':>8} {'P(|X̄n-μ|>ε)':>14} {'|X̄n-μ| (one run)':>18}")
    print(f"  {'─' * 42}")

    for n in [10, 50, 100, 500, 1000, 5000]:
        violations = 0
        trials = 2000
        last_dev = 0
        for _ in range(trials):
            sample_mean = sum(random.gauss(mu, sigma) for _ in range(n)) / n
            if abs(sample_mean - mu) > eps:
                violations += 1
            last_dev = abs(sample_mean - mu)
        prob = violations / trials
        print(f"  {n:>8} {prob:>14.4f} {last_dev:>18.6f}")


# ─────────────────────────────────────────────────
# 2. ALMOST SURE CONVERGENCE
# ─────────────────────────────────────────────────

def demo_as_convergence():
    print("\n" + "=" * 60)
    print("  Almost Sure Convergence")
    print("=" * 60)

    # Xn = max of first n uniform samples → 1 a.s.
    random.seed(42)
    n_paths = 5
    max_n = 200

    print(f"\n  Xn = max(U₁,...,Un) where Ui ~ Uniform(0,1)")
    print(f"  Xn → 1 almost surely")

    paths = []
    for _ in range(n_paths):
        current_max = 0
        path = []
        for i in range(1, max_n + 1):
            u = random.random()
            current_max = max(current_max, u)
            path.append(current_max)
        paths.append(path)

    print(f"\n  {'n':>6}", end="")
    for j in range(n_paths):
        print(f"  {'Path '+str(j+1):>10}", end="")
    print()
    print(f"  {'─' * (6 + 12 * n_paths)}")

    for n in [1, 5, 10, 20, 50, 100, 200]:
        print(f"  {n:>6}", end="")
        for path in paths:
            print(f"  {path[n-1]:>10.6f}", end="")
        print()

    print(f"\n  All paths converge to 1 (a.s. convergence)")


# ─────────────────────────────────────────────────
# 3. CONVERGENCE IN DISTRIBUTION
# ─────────────────────────────────────────────────

def demo_convergence_distribution():
    print("\n" + "=" * 60)
    print("  Convergence in Distribution")
    print("=" * 60)

    # (X̄n - μ)/(σ/√n) → N(0,1) (CLT)
    random.seed(42)
    mu, sigma = 3.0, 1.0  # Exponential(1/3) approximated

    print(f"\n  CLT: √n(X̄n - μ)/σ → N(0,1) in distribution")
    print(f"  Using Exponential(λ=1): μ=1, σ=1")

    for n in [5, 20, 50, 200]:
        standardized = []
        for _ in range(5000):
            sample = [random.expovariate(1.0) for _ in range(n)]
            x_bar = sum(sample) / n
            z = math.sqrt(n) * (x_bar - 1.0) / 1.0
            standardized.append(z)

        # Check quantiles against N(0,1)
        standardized.sort()
        q25 = standardized[len(standardized)//4]
        q50 = standardized[len(standardized)//2]
        q75 = standardized[3*len(standardized)//4]

        print(f"  n={n:>3}: Q25={q25:>6.3f}({-0.674:>6.3f}), "
              f"Q50={q50:>6.3f}({0:>6.3f}), "
              f"Q75={q75:>6.3f}({0.674:>6.3f})")

    print(f"  (N(0,1) quantiles in parentheses)")


# ─────────────────────────────────────────────────
# 4. RELATIONSHIPS AND COUNTEREXAMPLES
# ─────────────────────────────────────────────────

def demo_relationships():
    print("\n" + "=" * 60)
    print("  Convergence Relationships")
    print("=" * 60)

    print("""
  Hierarchy:
    a.s. convergence ──→ convergence in probability ──→ convergence in distribution
         ↑                        ↑
    Lp convergence ────────────────┘

  Key facts:
  • a.s. → prob (but not reverse in general)
  • prob → dist (but not reverse in general)
  • Lp → prob (but not reverse)
  • dist + constant limit → prob (special case)
""")

    # Counterexample: convergence in prob but not a.s.
    # Sliding indicator: Xn = 1 on interval of length 1/n cycling over [0,1]
    random.seed(42)
    n = 10000

    print(f"  Counterexample: conv. in prob ↛ a.s.")
    print(f"  Typewriter sequence on [0,1]:")
    print(f"  X₁=1 on [0,1], X₂=1 on [0,1/2], X₃=1 on [1/2,1],")
    print(f"  X₄=1 on [0,1/3], X₅=1 on [1/3,2/3], ...")
    print(f"  → Xn → 0 in probability (indicator shrinks)")
    print(f"  → NOT a.s. (every ω ∈ [0,1] hit infinitely often)")

    # Slutsky's theorem demo
    print(f"\n  Slutsky's Theorem:")
    print(f"  If Xn →d X and Yn →p c, then Xn+Yn →d X+c")
    print(f"  and Xn·Yn →d c·X")

    xn = [random.gauss(0, 1) for _ in range(n)]
    # Yn → 2 in probability
    yn = [2 + random.gauss(0, 0.01) for _ in range(n)]
    sum_samples = [x + y for x, y in zip(xn, yn)]
    s_mean = sum(sum_samples) / n
    s_var = sum((x - s_mean)**2 for x in sum_samples) / n
    print(f"  Xn ~ N(0,1), Yn → 2 in prob")
    print(f"  Xn + Yn: mean={s_mean:.3f} (→2), var={s_var:.3f} (→1)")


if __name__ == "__main__":
    demo_convergence_probability()
    demo_as_convergence()
    demo_convergence_distribution()
    demo_relationships()

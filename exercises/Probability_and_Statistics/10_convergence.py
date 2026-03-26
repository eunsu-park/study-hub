"""
Convergence of Random Variables — Exercises

Topics covered:
- Convergence in probability (Weak Law of Large Numbers)
- Almost sure convergence
- Convergence in distribution (Central Limit Theorem)
- Slutsky's theorem
"""

import math
import random
from typing import List, Tuple


# === Exercise 1: Convergence in Probability — WLLN for Exponential(1) ===
def exercise_1() -> None:
    """Simulate X̄n for Exponential(1) samples, show P(|X̄n - 1| > 0.1) -> 0."""
    print("=" * 70)
    print("Exercise 1: Convergence in Probability — WLLN for Exponential(1)")
    print("=" * 70)
    print()
    print("Problem:")
    print("  Let X1, X2, ... be i.i.d. Exponential(1) with E[Xi] = 1.")
    print("  Estimate P(|X̄n - 1| > 0.1) for n = 10, 50, 100, 500, 1000, 5000.")
    print("  Show that this probability converges to 0 (WLLN).")
    print()

    random.seed(42)
    epsilon: float = 0.1
    sample_sizes: List[int] = [10, 50, 100, 500, 1000, 5000]
    num_trials: int = 10000

    print("Solution:")
    print(f"  Epsilon = {epsilon}, Number of trials = {num_trials}")
    print(f"  {'n':>6}  P(|X̄n - 1| > {epsilon})")
    print(f"  {'-' * 6}  {'-' * 25}")

    for n in sample_sizes:
        count_exceed: int = 0
        for _ in range(num_trials):
            # Generate n Exponential(1) samples using inverse transform
            samples: List[float] = [-math.log(random.random()) for _ in range(n)]
            sample_mean: float = sum(samples) / n
            if abs(sample_mean - 1.0) > epsilon:
                count_exceed += 1
        prob: float = count_exceed / num_trials
        print(f"  {n:>6}  {prob:.4f}")

    print()
    print("  As n increases, P(|X̄n - 1| > 0.1) -> 0, confirming WLLN.")
    print()


# === Exercise 2: Almost Sure Convergence — max(U1,...,Un) -> 1 ===
def exercise_2() -> None:
    """Simulate max(U1,...,Un) for Uniform(0,1), show a.s. convergence to 1."""
    print("=" * 70)
    print("Exercise 2: Almost Sure Convergence — max of Uniform samples")
    print("=" * 70)
    print()
    print("Problem:")
    print("  Let U1, U2, ... be i.i.d. Uniform(0,1). Define Mn = max(U1,...,Un).")
    print("  Show Mn -> 1 almost surely by tracking Mn across one long sequence.")
    print("  Theory: P(Mn <= x) = x^n, so P(Mn <= 1-eps) = (1-eps)^n -> 0.")
    print()

    random.seed(42)
    num_paths: int = 5
    checkpoints: List[int] = [10, 50, 100, 500, 1000, 5000, 10000]

    print("Solution:")
    print(f"  Simulating {num_paths} independent sample paths.")
    print()

    header: str = f"  {'n':>6}"
    for p in range(1, num_paths + 1):
        header += f"  {'Path ' + str(p):>10}"
    print(header)
    print(f"  {'-' * 6}" + f"  {'-' * 10}" * num_paths)

    for path_idx in range(num_paths):
        random.seed(42 + path_idx)
        running_max: float = 0.0
        sample_idx: int = 0
        checkpoint_idx: int = 0
        results: List[Tuple[int, float]] = []

        for i in range(1, checkpoints[-1] + 1):
            u: float = random.random()
            if u > running_max:
                running_max = u
            if i == checkpoints[checkpoint_idx]:
                results.append((i, running_max))
                checkpoint_idx += 1
                if checkpoint_idx >= len(checkpoints):
                    break

        if path_idx == 0:
            all_results: List[List[Tuple[int, float]]] = []
        all_results.append(results)

    for row_idx in range(len(checkpoints)):
        n: int = checkpoints[row_idx]
        line: str = f"  {n:>6}"
        for path_results in all_results:
            line += f"  {path_results[row_idx][1]:>10.6f}"
        print(line)

    print()
    print("  All paths show Mn -> 1 as n -> infinity (almost sure convergence).")
    print(f"  Gap from 1 at n=10000: ~{1.0 - all_results[0][-1][1]:.6f}")
    print()


# === Exercise 3: Convergence in Distribution — CLT Verification ===
def exercise_3() -> None:
    """Compute standardized sample means, compare CDF with Phi for various n."""
    print("=" * 70)
    print("Exercise 3: Convergence in Distribution — CLT CDF Comparison")
    print("=" * 70)
    print()
    print("Problem:")
    print("  For X ~ Exponential(1) (mu=1, sigma=1), compute the standardized")
    print("  sample mean: Zn = sqrt(n)*(X̄n - 1)/1.")
    print("  Compare empirical CDF of Zn with standard normal CDF Phi(z).")
    print()

    random.seed(42)
    sample_sizes: List[int] = [5, 30, 100, 500]
    num_simulations: int = 10000
    eval_points: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    def standard_normal_cdf(z: float) -> float:
        """Approximate Phi(z) using the error function."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    print("Solution:")
    print("  Phi(z) = standard normal CDF (target)")
    print()

    for n in sample_sizes:
        z_values: List[float] = []
        for _ in range(num_simulations):
            samples: List[float] = [-math.log(random.random()) for _ in range(n)]
            x_bar: float = sum(samples) / n
            z: float = math.sqrt(n) * (x_bar - 1.0)
            z_values.append(z)
        z_values.sort()

        print(f"  n = {n}:")
        print(f"    {'z':>6}  {'Empirical CDF':>14}  {'Phi(z)':>8}  {'|Diff|':>8}")
        print(f"    {'-' * 6}  {'-' * 14}  {'-' * 8}  {'-' * 8}")

        max_diff: float = 0.0
        for zp in eval_points:
            count_below: int = sum(1 for zv in z_values if zv <= zp)
            empirical: float = count_below / num_simulations
            theoretical: float = standard_normal_cdf(zp)
            diff: float = abs(empirical - theoretical)
            if diff > max_diff:
                max_diff = diff
            print(f"    {zp:>6.1f}  {empirical:>14.4f}  {theoretical:>8.4f}  {diff:>8.4f}")

        print(f"    Max |diff| = {max_diff:.4f}")
        print()

    print("  As n increases, empirical CDF converges to Phi(z), confirming CLT.")
    print()


# === Exercise 4: Slutsky's Theorem Verification ===
def exercise_4() -> None:
    """Verify Slutsky's theorem: Xn ~ N(0,1), Yn -> 2 in prob, check Xn + Yn."""
    print("=" * 70)
    print("Exercise 4: Slutsky's Theorem Verification")
    print("=" * 70)
    print()
    print("Problem:")
    print("  Let Xn ~ N(0,1) (does not depend on n) and Yn = 2 + 1/n * Epsilon")
    print("  where Epsilon ~ N(0,1). Then Yn -> 2 in probability.")
    print("  By Slutsky's theorem, Xn + Yn -> N(2,1) in distribution.")
    print("  Verify by comparing empirical CDF of Xn + Yn with N(2,1) CDF.")
    print()

    random.seed(42)
    sample_sizes: List[int] = [5, 50, 500, 5000]
    num_simulations: int = 10000
    eval_points: List[float] = [0.0, 1.0, 2.0, 3.0, 4.0]

    def normal_cdf(x: float, mu: float, sigma: float) -> float:
        """CDF of N(mu, sigma^2)."""
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

    def box_muller() -> Tuple[float, float]:
        """Generate two independent N(0,1) samples."""
        u1: float = random.random()
        u2: float = random.random()
        # Avoid log(0)
        while u1 == 0.0:
            u1 = random.random()
        z1: float = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z2: float = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        return z1, z2

    print("Solution:")
    print("  Target distribution: N(2, 1)")
    print()

    for n in sample_sizes:
        sums: List[float] = []
        for _ in range(num_simulations):
            xn, eps = box_muller()
            yn: float = 2.0 + (1.0 / n) * eps
            sums.append(xn + yn)
        sums.sort()

        print(f"  n = {n} (Yn = 2 + (1/{n})*eps):")
        print(f"    {'z':>6}  {'Empirical':>10}  {'N(2,1) CDF':>10}  {'|Diff|':>8}")
        print(f"    {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

        for zp in eval_points:
            count_below: int = sum(1 for s in sums if s <= zp)
            empirical: float = count_below / num_simulations
            theoretical: float = normal_cdf(zp, 2.0, 1.0)
            diff: float = abs(empirical - theoretical)
            print(f"    {zp:>6.1f}  {empirical:>10.4f}  {theoretical:>10.4f}  {diff:>8.4f}")
        print()

    print("  As n -> inf, Yn -> 2 in prob, so Xn + Yn -> N(2,1) by Slutsky.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

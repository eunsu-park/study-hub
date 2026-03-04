"""
Probability and Statistics — Discrete Distributions
Exercises covering Binomial, Poisson approximation, Geometric,
Hypergeometric, and Negative Binomial distributions.
"""
import math
import random
from typing import List, Tuple


# === Exercise 1: Binomial Distribution — PMF and Mode ===
def exercise_1() -> None:
    """Compute P(X=k) for X ~ Binomial(n=20, p=0.3) for k=0..20.
    Find the mode (most probable value)."""
    print("=== Exercise 1: Binomial(20, 0.3) PMF and Mode ===")

    n = 20
    p = 0.3

    def binom_pmf(k: int, n: int, p: float) -> float:
        """P(X=k) = C(n,k) * p^k * (1-p)^(n-k)"""
        coeff = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        return coeff * (p ** k) * ((1 - p) ** (n - k))

    # Compute full PMF
    pmf: List[Tuple[int, float]] = []
    for k in range(n + 1):
        pmf.append((k, binom_pmf(k, n, p)))

    # Find mode
    mode_k, mode_p = max(pmf, key=lambda t: t[1])

    # Display
    print(f"  X ~ Binomial(n={n}, p={p})")
    print(f"  E[X] = np = {n * p:.1f}")
    print(f"  Var(X) = np(1-p) = {n * p * (1 - p):.2f}")
    print()
    print(f"  {'k':>4} {'P(X=k)':>10} {'Bar':}")
    for k, pk in pmf:
        bar = "#" * int(pk * 100)
        marker = " <-- mode" if k == mode_k else ""
        print(f"  {k:>4} {pk:>10.6f}  {bar}{marker}")

    # Verify PMF sums to 1
    total = sum(pk for _, pk in pmf)
    assert abs(total - 1.0) < 1e-10, f"PMF sums to {total}"
    print(f"\n  Sum of PMF = {total:.10f}")
    print(f"  Mode at k = {mode_k} with P(X={mode_k}) = {mode_p:.6f}\n")


# === Exercise 2: Poisson Approximation to Binomial ===
def exercise_2() -> None:
    """Compare Binomial(n=1000, p=0.003) with Poisson(lambda=3) to
    demonstrate the Poisson approximation for rare events."""
    print("=== Exercise 2: Poisson Approximation to Binomial ===")

    n = 1000
    p = 0.003
    lam = n * p  # = 3.0

    def binom_pmf(k: int, n: int, p: float) -> float:
        coeff = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        return coeff * (p ** k) * ((1 - p) ** (n - k))

    def poisson_pmf(k: int, lam: float) -> float:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    print(f"  Binomial(n={n}, p={p})  vs  Poisson(lambda={lam})")
    print(f"  {'k':>4} {'Binomial':>12} {'Poisson':>12} {'|Diff|':>12}")
    print(f"  {'----':>4} {'----------':>12} {'----------':>12} {'----------':>12}")

    max_diff = 0.0
    for k in range(11):
        bp = binom_pmf(k, n, p)
        pp = poisson_pmf(k, lam)
        diff = abs(bp - pp)
        max_diff = max(max_diff, diff)
        print(f"  {k:>4} {bp:>12.8f} {pp:>12.8f} {diff:>12.8f}")

    print(f"\n  Maximum absolute difference (k=0..10): {max_diff:.8f}")
    print(f"  The Poisson approximation is excellent when n is large "
          f"and p is small.\n")


# === Exercise 3: Geometric Distribution — Memoryless Property ===
def exercise_3() -> None:
    """Simulate waiting time for the first head with P(Head)=0.15.
    Verify the memoryless property: P(X > s+t | X > s) = P(X > t)."""
    print("=== Exercise 3: Geometric Distribution — Memoryless Property ===")

    p = 0.15
    n_simulations = 200_000
    random.seed(42)

    # Simulate geometric random variables (number of trials until first success)
    def geometric_sample(p: float) -> int:
        """Return the trial number of the first success (1-indexed)."""
        count = 1
        while random.random() > p:
            count += 1
        return count

    samples: List[int] = [geometric_sample(p) for _ in range(n_simulations)]

    # Empirical mean and variance
    mean_emp = sum(samples) / n_simulations
    var_emp = sum((x - mean_emp) ** 2 for x in samples) / (n_simulations - 1)
    mean_theory = 1 / p
    var_theory = (1 - p) / p**2

    print(f"  X ~ Geometric(p={p})")
    print(f"  Theoretical: E[X] = {mean_theory:.4f}, Var(X) = {var_theory:.4f}")
    print(f"  Empirical:   E[X] = {mean_emp:.4f}, Var(X) = {var_emp:.4f}")
    print()

    # Verify memoryless property: P(X > s+t | X > s) should equal P(X > t)
    s, t = 5, 3

    # P(X > t) = (1-p)^t
    p_gt_t_theory = (1 - p) ** t
    p_gt_t_empirical = sum(1 for x in samples if x > t) / n_simulations

    # P(X > s+t | X > s) via conditional counting
    count_gt_s = sum(1 for x in samples if x > s)
    count_gt_s_plus_t = sum(1 for x in samples if x > s + t)
    p_conditional = count_gt_s_plus_t / count_gt_s if count_gt_s > 0 else 0

    print(f"  Memoryless property test (s={s}, t={t}):")
    print(f"    P(X > {t})              = (1-p)^{t} = {p_gt_t_theory:.6f}")
    print(f"    P(X > {t}) empirical    = {p_gt_t_empirical:.6f}")
    print(f"    P(X > {s+t} | X > {s}) empirical = {p_conditional:.6f}")
    print(f"    Difference: {abs(p_gt_t_theory - p_conditional):.6f}")
    print(f"    Memoryless property: P(X > s+t | X > s) = P(X > t) "
          f"approximately verified.\n")


# === Exercise 4: Hypergeometric Distribution ===
def exercise_4() -> None:
    """Compute the probability of drawing exactly 2 aces in a 5-card hand
    from a standard 52-card deck using the hypergeometric distribution."""
    print("=== Exercise 4: Hypergeometric — 2 Aces in 5-Card Hand ===")

    N = 52   # population size
    K = 4    # number of success states (aces)
    n = 5    # number of draws
    k = 2    # desired successes

    def hypergeometric_pmf(k: int, N: int, K: int, n: int) -> float:
        """P(X=k) = C(K,k)*C(N-K,n-k) / C(N,n)"""
        def comb(a: int, b: int) -> int:
            if b < 0 or b > a:
                return 0
            return math.factorial(a) // (math.factorial(b) * math.factorial(a - b))

        return comb(K, k) * comb(N - K, n - k) / comb(N, n)

    # Compute for all possible k values
    print(f"  Drawing n={n} cards from N={N}, K={K} aces in deck.")
    print(f"  {'k':>4} {'P(X=k)':>12}")
    for ki in range(min(K, n) + 1):
        prob = hypergeometric_pmf(ki, N, K, n)
        marker = " <-- target" if ki == k else ""
        print(f"  {ki:>4} {prob:>12.8f}{marker}")

    target_prob = hypergeometric_pmf(k, N, K, n)

    # Expected value E[X] = nK/N
    ex = n * K / N
    # Variance
    var = n * K * (N - K) * (N - n) / (N**2 * (N - 1))

    print(f"\n  P(exactly {k} aces) = {target_prob:.8f}")
    print(f"  ≈ 1 in {1/target_prob:.1f}")
    print(f"  E[X] = nK/N = {ex:.4f}")
    print(f"  Var(X) = {var:.4f}\n")


# === Exercise 5: Negative Binomial Distribution ===
def exercise_5() -> None:
    """Simulate the Negative Binomial distribution: number of trials to
    achieve the 5th success with p=0.4.  Compare with theoretical moments."""
    print("=== Exercise 5: Negative Binomial — Time to 5th Success ===")

    r = 5     # required successes
    p = 0.4   # probability of success
    n_simulations = 200_000
    random.seed(101)

    def negative_binomial_sample(r: int, p: float) -> int:
        """Return the number of trials to get r successes."""
        successes = 0
        trials = 0
        while successes < r:
            trials += 1
            if random.random() < p:
                successes += 1
        return trials

    samples: List[int] = [negative_binomial_sample(r, p)
                          for _ in range(n_simulations)]

    # Theoretical moments (for total trials interpretation: NB in terms of trials)
    mean_theory = r / p
    var_theory = r * (1 - p) / p**2

    mean_emp = sum(samples) / n_simulations
    var_emp = sum((x - mean_emp) ** 2 for x in samples) / (n_simulations - 1)

    print(f"  NegBin(r={r}, p={p}): trials until {r}th success")
    print(f"  Theoretical: E[X] = r/p = {mean_theory:.4f}, "
          f"Var(X) = r(1-p)/p^2 = {var_theory:.4f}")
    print(f"  Empirical:   E[X] = {mean_emp:.4f}, Var(X) = {var_emp:.4f}")

    # PMF for first few values using the formula
    # P(X=k) = C(k-1, r-1) * p^r * (1-p)^(k-r)  for k >= r
    print(f"\n  {'k':>4} {'P(X=k) theory':>14} {'Empirical':>12}")
    from collections import Counter
    freq = Counter(samples)

    for k in range(r, r + 15):
        coeff = math.factorial(k - 1) // (
            math.factorial(r - 1) * math.factorial(k - r)
        )
        prob_theory = coeff * (p ** r) * ((1 - p) ** (k - r))
        prob_emp = freq.get(k, 0) / n_simulations
        print(f"  {k:>4} {prob_theory:>14.8f} {prob_emp:>12.6f}")

    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

"""
Combinatorics and Counting

Demonstrates:
1. Permutation and Combination Calculator
2. Multinomial Coefficients
3. Inclusion-Exclusion Principle Simulation
4. Derangement Counter

Theory:
- Permutation P(n,r) = n! / (n-r)!  (order matters)
- Combination C(n,r) = n! / (r!(n-r)!)  (order doesn't matter)
- Inclusion-Exclusion: |A∪B∪C| = |A|+|B|+|C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|

Adapted from Probability and Statistics Lesson 01.
"""

from math import factorial, comb, perm
from itertools import permutations, combinations
from functools import reduce


# ─────────────────────────────────────────────────
# 1. PERMUTATION AND COMBINATION CALCULATOR
# ─────────────────────────────────────────────────

def permutation_count(n: int, r: int) -> int:
    """P(n, r) = n! / (n-r)!"""
    return perm(n, r)


def combination_count(n: int, r: int) -> int:
    """C(n, r) = n! / (r!(n-r)!)"""
    return comb(n, r)


def combinations_with_repetition(n: int, r: int) -> int:
    """Stars and bars: C(n+r-1, r)."""
    return comb(n + r - 1, r)


def demo_basic_counting():
    print("=" * 60)
    print("  Permutations and Combinations")
    print("=" * 60)

    n, r = 10, 3
    print(f"\n  n = {n}, r = {r}")
    print(f"  P({n},{r}) = {permutation_count(n, r)}")
    print(f"  C({n},{r}) = {combination_count(n, r)}")
    print(f"  C_rep({n},{r}) = {combinations_with_repetition(n, r)}")

    # Verify by enumeration for small case
    n_small, r_small = 5, 2
    perms = list(permutations(range(n_small), r_small))
    combs = list(combinations(range(n_small), r_small))
    print(f"\n  Enumeration check (n=5, r=2):")
    print(f"  Permutations: {len(perms)} (formula: {permutation_count(5, 2)})")
    print(f"  Combinations: {len(combs)} (formula: {combination_count(5, 2)})")

    # Poker hand example
    print(f"\n  Poker hand (5 from 52):")
    print(f"  Total hands: {comb(52, 5):,}")
    print(f"  Flush (same suit): {4 * comb(13, 5):,}")
    print(f"  Four of a kind: {13 * 48:,}")


# ─────────────────────────────────────────────────
# 2. MULTINOMIAL COEFFICIENTS
# ─────────────────────────────────────────────────

def multinomial(n: int, groups: list[int]) -> int:
    """Multinomial coefficient: n! / (k1! * k2! * ... * km!)."""
    if sum(groups) != n:
        raise ValueError("Group sizes must sum to n")
    return factorial(n) // reduce(lambda a, b: a * b,
                                   (factorial(k) for k in groups))


def demo_multinomial():
    print("\n" + "=" * 60)
    print("  Multinomial Coefficients")
    print("=" * 60)

    # Arranging letters in "MISSISSIPPI"
    word = "MISSISSIPPI"
    from collections import Counter
    counts = Counter(word)
    groups = list(counts.values())
    n = len(word)

    print(f"\n  Word: {word}")
    print(f"  Letter counts: {dict(counts)}")
    print(f"  Arrangements: {n}! / {' × '.join(f'{v}!' for v in groups)}")
    print(f"  = {multinomial(n, groups):,}")

    # Dividing 12 people into groups of 4, 4, 4
    print(f"\n  Divide 12 people into 3 groups of 4:")
    ways = multinomial(12, [4, 4, 4])
    # If groups are indistinguishable, divide by 3!
    print(f"  Distinguishable groups: {ways:,}")
    print(f"  Indistinguishable groups: {ways // factorial(3):,}")


# ─────────────────────────────────────────────────
# 3. INCLUSION-EXCLUSION PRINCIPLE
# ─────────────────────────────────────────────────

def inclusion_exclusion(sets: list[set]) -> int:
    """Count |A₁ ∪ A₂ ∪ ... ∪ Aₙ| via inclusion-exclusion."""
    n = len(sets)
    total = 0
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            intersection = sets[combo[0]]
            for i in combo[1:]:
                intersection = intersection & sets[i]
            if k % 2 == 1:
                total += len(intersection)
            else:
                total -= len(intersection)
    return total


def euler_totient(n: int) -> int:
    """Euler's totient via inclusion-exclusion on prime factors."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def demo_inclusion_exclusion():
    print("\n" + "=" * 60)
    print("  Inclusion-Exclusion Principle")
    print("=" * 60)

    # Students in clubs
    math_club = {1, 2, 3, 4, 5, 6}
    cs_club = {4, 5, 6, 7, 8}
    physics_club = {2, 6, 8, 9, 10}

    sets = [math_club, cs_club, physics_club]
    union_size = inclusion_exclusion(sets)

    print(f"\n  Math club:    {sorted(math_club)}")
    print(f"  CS club:      {sorted(cs_club)}")
    print(f"  Physics club: {sorted(physics_club)}")
    print(f"\n  |Math ∪ CS ∪ Physics| = {union_size}")
    print(f"  Direct: {len(math_club | cs_club | physics_club)}")

    # Step by step
    print(f"\n  Step-by-step:")
    print(f"  + |M| + |C| + |P| = {len(math_club)} + {len(cs_club)} + {len(physics_club)} = {len(math_club)+len(cs_club)+len(physics_club)}")
    mc = math_club & cs_club
    mp = math_club & physics_club
    cp = cs_club & physics_club
    print(f"  - |M∩C| - |M∩P| - |C∩P| = {len(mc)} + {len(mp)} + {len(cp)} = {len(mc)+len(mp)+len(cp)}")
    mcp = math_club & cs_club & physics_club
    print(f"  + |M∩C∩P| = {len(mcp)}")

    # Euler's totient
    print(f"\n  Euler's totient (inclusion-exclusion on primes):")
    for n in [12, 30, 100]:
        print(f"  φ({n}) = {euler_totient(n)}")


# ─────────────────────────────────────────────────
# 4. DERANGEMENTS
# ─────────────────────────────────────────────────

def count_derangements(n: int) -> int:
    """D(n): permutations where no element is in its original position."""
    if n == 0:
        return 1
    if n == 1:
        return 0
    # D(n) = (n-1)(D(n-1) + D(n-2))
    dp = [1, 0]
    for i in range(2, n + 1):
        dp.append((i - 1) * (dp[-1] + dp[-2]))
    return dp[n]


def demo_derangements():
    print("\n" + "=" * 60)
    print("  Derangements")
    print("=" * 60)

    print(f"\n  {'n':<5} {'D(n)':<12} {'n!':<12} {'D(n)/n!':<10}")
    print(f"  {'─' * 38}")
    for n in range(1, 11):
        d = count_derangements(n)
        f = factorial(n)
        ratio = d / f if f > 0 else 0
        print(f"  {n:<5} {d:<12} {f:<12} {ratio:<10.6f}")

    print(f"\n  D(n)/n! → 1/e ≈ 0.367879 as n → ∞")

    # Verify for n=4 by enumeration
    import itertools
    n = 4
    original = list(range(n))
    derangements = [p for p in permutations(original) if all(p[i] != i for i in range(n))]
    print(f"\n  Derangements of [0,1,2,3]: {len(derangements)} (formula: {count_derangements(4)})")
    for d in derangements[:5]:
        print(f"    {list(d)}")


if __name__ == "__main__":
    demo_basic_counting()
    demo_multinomial()
    demo_inclusion_exclusion()
    demo_derangements()

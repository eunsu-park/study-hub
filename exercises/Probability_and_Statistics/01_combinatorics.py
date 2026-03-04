"""
Probability and Statistics — Combinatorics
Exercises covering permutations, combinations, multinomial coefficients,
inclusion-exclusion, derangements, and stars-and-bars.
"""
import math
from typing import List


# === Exercise 1: Implement nCr and nPr from Scratch ===
def exercise_1() -> None:
    """Implement nCr (combinations) and nPr (permutations) without using
    math.comb or math.perm.  Verify against known values."""
    print("=== Exercise 1: Implement nCr and nPr from Scratch ===")

    def factorial(n: int) -> int:
        """Compute n! iteratively."""
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def nPr(n: int, r: int) -> int:
        """Number of r-permutations of n objects: n! / (n-r)!"""
        if r < 0 or r > n:
            return 0
        result = 1
        for i in range(n, n - r, -1):
            result *= i
        return result

    def nCr(n: int, r: int) -> int:
        """Number of r-combinations of n objects: n! / (r!(n-r)!)"""
        if r < 0 or r > n:
            return 0
        # Use the smaller of r and n-r for efficiency
        r = min(r, n - r)
        numerator = 1
        denominator = 1
        for i in range(r):
            numerator *= (n - i)
            denominator *= (i + 1)
        return numerator // denominator

    # Test cases
    test_cases = [(10, 3), (52, 5), (20, 7), (5, 0), (5, 5)]
    for n, r in test_cases:
        p = nPr(n, r)
        c = nCr(n, r)
        # Verify against math.factorial
        p_expected = factorial(n) // factorial(n - r)
        c_expected = factorial(n) // (factorial(r) * factorial(n - r))
        assert p == p_expected, f"nPr({n},{r}) mismatch"
        assert c == c_expected, f"nCr({n},{r}) mismatch"
        print(f"  P({n},{r}) = {p:>12,}    C({n},{r}) = {c:>10,}")

    print(f"  All {len(test_cases)} test cases passed.\n")


# === Exercise 2: Multinomial Coefficient for "STATISTICS" ===
def exercise_2() -> None:
    """Compute the number of distinct anagrams of 'STATISTICS' using the
    multinomial coefficient: n! / (n1! * n2! * ... * nk!)."""
    print('=== Exercise 2: Multinomial Coefficient for "STATISTICS" ===')

    word = "STATISTICS"
    n = len(word)

    # Count frequency of each letter
    freq: dict[str, int] = {}
    for ch in word:
        freq[ch] = freq.get(ch, 0) + 1

    print(f"  Word: {word}  (length {n})")
    print(f"  Letter frequencies: {dict(sorted(freq.items()))}")

    # Compute multinomial coefficient
    numerator = math.factorial(n)
    denominator = 1
    for count in freq.values():
        denominator *= math.factorial(count)

    result = numerator // denominator
    print(f"  Multinomial coefficient = {n}! / ", end="")
    print(" * ".join(f"{c}!" for c in sorted(freq.values(), reverse=True)))
    print(f"                         = {numerator} / {denominator}")
    print(f"                         = {result:,}\n")


# === Exercise 3: Inclusion-Exclusion Principle ===
def exercise_3() -> None:
    """Count integers from 1 to 1000 divisible by 2, 3, or 5 using the
    inclusion-exclusion principle."""
    print("=== Exercise 3: Inclusion-Exclusion (Divisible by 2, 3, or 5) ===")

    N = 1000

    # |A_i| = floor(N / d_i)
    d2 = N // 2      # divisible by 2
    d3 = N // 3      # divisible by 3
    d5 = N // 5      # divisible by 5

    # |A_i ∩ A_j| = floor(N / lcm(d_i, d_j))
    d6 = N // 6      # divisible by 2 and 3
    d10 = N // 10    # divisible by 2 and 5
    d15 = N // 15    # divisible by 3 and 5

    # |A_1 ∩ A_2 ∩ A_3| = floor(N / lcm(2,3,5))
    d30 = N // 30    # divisible by 2, 3, and 5

    # Inclusion-exclusion formula
    count_ie = (d2 + d3 + d5) - (d6 + d10 + d15) + d30

    # Brute-force verification
    count_brute = sum(1 for i in range(1, N + 1)
                      if i % 2 == 0 or i % 3 == 0 or i % 5 == 0)

    print(f"  Range: 1 to {N}")
    print(f"  |div by 2| = {d2},  |div by 3| = {d3},  |div by 5| = {d5}")
    print(f"  |div by 6| = {d6},  |div by 10| = {d10}, |div by 15| = {d15}")
    print(f"  |div by 30| = {d30}")
    print(f"  Inclusion-exclusion: {d2}+{d3}+{d5} - {d6}-{d10}-{d15} + {d30} = {count_ie}")
    print(f"  Brute-force count:  {count_brute}")
    assert count_ie == count_brute, "Mismatch!"
    print(f"  Results match.\n")


# === Exercise 4: Derangements ===
def exercise_4() -> None:
    """Count the number of derangements (permutations with no fixed points)
    of [1, 2, ..., 8] using the subfactorial formula."""
    print("=== Exercise 4: Derangements of [1..8] ===")

    n = 8

    # Method 1: Subfactorial via inclusion-exclusion
    # D(n) = n! * sum_{k=0}^{n} (-1)^k / k!
    def derangements_formula(n: int) -> int:
        total = 0
        for k in range(n + 1):
            total += ((-1) ** k) * math.factorial(n) // math.factorial(k)
        return total

    # Method 2: Recurrence D(n) = (n-1)(D(n-1) + D(n-2)), D(0)=1, D(1)=0
    def derangements_recurrence(n: int) -> int:
        if n == 0:
            return 1
        if n == 1:
            return 0
        d_prev2 = 1  # D(0)
        d_prev1 = 0  # D(1)
        for i in range(2, n + 1):
            d_curr = (i - 1) * (d_prev1 + d_prev2)
            d_prev2 = d_prev1
            d_prev1 = d_curr
        return d_prev1

    d_formula = derangements_formula(n)
    d_recurrence = derangements_recurrence(n)

    assert d_formula == d_recurrence, "Methods disagree!"

    total_perms = math.factorial(n)
    prob = d_formula / total_perms

    print(f"  n = {n}")
    print(f"  D({n}) via inclusion-exclusion = {d_formula:,}")
    print(f"  D({n}) via recurrence          = {d_recurrence:,}")
    print(f"  Total permutations: {n}! = {total_perms:,}")
    print(f"  P(derangement) = {d_formula}/{total_perms} = {prob:.6f}")
    print(f"  Approaches 1/e = {1/math.e:.6f}\n")


# === Exercise 5: Stars and Bars ===
def exercise_5() -> None:
    """Count the number of ways to distribute 10 identical balls into 4
    distinct boxes using the stars-and-bars formula.  Also enumerate
    solutions where each box must have at least 1 ball."""
    print("=== Exercise 5: Stars and Bars (10 balls, 4 boxes) ===")

    n = 10  # balls
    k = 4   # boxes

    # Unrestricted: C(n + k - 1, k - 1)
    unrestricted = math.factorial(n + k - 1) // (
        math.factorial(k - 1) * math.factorial(n)
    )

    # With at least 1 per box: substitute y_i = x_i - 1, need y_1+...+y_k = n-k
    # so C(n-1, k-1)
    at_least_one = math.factorial(n - 1) // (
        math.factorial(k - 1) * math.factorial(n - k)
    )

    # Brute-force verification (unrestricted)
    brute_count = 0
    for x1 in range(n + 1):
        for x2 in range(n - x1 + 1):
            for x3 in range(n - x1 - x2 + 1):
                x4 = n - x1 - x2 - x3
                if x4 >= 0:
                    brute_count += 1

    # Brute-force verification (at least 1 each)
    brute_at_least_one = 0
    for x1 in range(1, n + 1):
        for x2 in range(1, n - x1 + 1):
            for x3 in range(1, n - x1 - x2 + 1):
                x4 = n - x1 - x2 - x3
                if x4 >= 1:
                    brute_at_least_one += 1

    print(f"  Balls (n) = {n}, Boxes (k) = {k}")
    print(f"  Unrestricted:     C({n+k-1},{k-1}) = {unrestricted}")
    print(f"    Brute-force:    {brute_count}")
    assert unrestricted == brute_count
    print(f"  At least 1 each:  C({n-1},{k-1}) = {at_least_one}")
    print(f"    Brute-force:    {brute_at_least_one}")
    assert at_least_one == brute_at_least_one
    print(f"  Results match.\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

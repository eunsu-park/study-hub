"""
Exercises for Lesson 01: Number Theory Foundations
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

from math import gcd, isqrt
import random


def extended_gcd(a, b):
    """Extended Euclidean Algorithm. Returns (gcd, x, y) such that a*x + b*y = gcd(a, b)."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


def mod_inverse(a, n):
    """Compute modular inverse of a modulo n using Extended GCD."""
    g, x, _ = extended_gcd(a, n)
    if g != 1:
        raise ValueError(f"No modular inverse: gcd({a}, {n}) = {g}")
    return x % n


def euler_totient(n):
    """Compute Euler's totient function phi(n) via prime factorization."""
    result = n
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def miller_rabin(n, k=20):
    """Miller-Rabin primality test. Returns True if n is probably prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def exercise_1():
    """Exercise 1: Modular Arithmetic Warm-up (Basic)

    Compute by hand (then verify with Python):
    1. 17^3 mod 13
    2. 7^(-1) mod 31
    3. phi(360)
    """
    # 1. 17^3 mod 13
    # Mathematical reasoning:
    # 17 mod 13 = 4, so 17^3 mod 13 = 4^3 mod 13 = 64 mod 13
    # 64 = 4*13 + 12, so 64 mod 13 = 12
    result_1 = pow(17, 3, 13)
    print(f"  17^3 mod 13 = {result_1}")
    print(f"  Reasoning: 17 ≡ 4 (mod 13), 4^3 = 64, 64 mod 13 = {64 % 13}")

    # 2. 7^(-1) mod 31
    # We need x such that 7x ≡ 1 (mod 31)
    # Using Extended GCD: gcd(7, 31) = 1, so inverse exists
    # By Fermat's little theorem (31 is prime): 7^(-1) ≡ 7^(31-2) = 7^29 (mod 31)
    inv = mod_inverse(7, 31)
    print(f"\n  7^(-1) mod 31 = {inv}")
    print(f"  Verification: 7 * {inv} mod 31 = {(7 * inv) % 31}")

    # Alternative via Fermat's little theorem
    inv_fermat = pow(7, 29, 31)
    print(f"  Via Fermat: 7^29 mod 31 = {inv_fermat}")

    # 3. phi(360)
    # 360 = 2^3 * 3^2 * 5
    # phi(360) = 360 * (1 - 1/2) * (1 - 1/3) * (1 - 1/5)
    #          = 360 * 1/2 * 2/3 * 4/5 = 360 * 8/30 = 96
    phi_360 = euler_totient(360)
    print(f"\n  phi(360) = {phi_360}")
    print("  Reasoning: 360 = 2^3 * 3^2 * 5")
    print("  phi(360) = 360 * (1 - 1/2) * (1 - 1/3) * (1 - 1/5)")
    print(f"           = 360 * 1/2 * 2/3 * 4/5 = {360 * 1 * 2 * 4 // (2 * 3 * 5)}")


def exercise_2():
    """Exercise 2: Extended GCD Application (Intermediate)

    Use the Extended Euclidean Algorithm to solve:
    1234x + 567y = gcd(1234, 567)
    Then find 1234^(-1) mod 567 (if it exists).
    """
    a, b = 1234, 567

    # Compute Extended GCD
    g, x, y = extended_gcd(a, b)
    print(f"  gcd({a}, {b}) = {g}")
    print(f"  Bezout coefficients: x = {x}, y = {y}")
    print(f"  Verification: {a}*{x} + {b}*{y} = {a * x + b * y}")

    # Find modular inverse of 1234 mod 567
    if g == 1:
        inv = x % b
        print(f"\n  {a}^(-1) mod {b} = {inv}")
        print(f"  Verification: {a} * {inv} mod {b} = {(a * inv) % b}")
    else:
        print(f"\n  gcd({a}, {b}) = {g} != 1")
        print(f"  Therefore {a}^(-1) mod {b} does NOT exist.")
        # Since gcd(1234, 567) = 1, the inverse should exist.
        # Let's check: 1234 = 2*617, 567 = 7*81 = 7*3^4
        # gcd(1234, 567): 1234 mod 567 = 100, 567 mod 100 = 67, 100 mod 67 = 33,
        # 67 mod 33 = 1, 33 mod 1 = 0. So gcd = 1.
        print("  (Double-checking factorizations...)")


def exercise_3():
    """Exercise 3: CRT Problem (Intermediate)

    A number leaves remainder 2 when divided by 3, remainder 3 when divided by 5,
    and remainder 4 when divided by 7. Find the smallest positive number.

    System: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 4 (mod 7)
    """
    remainders = [2, 3, 4]
    moduli = [3, 5, 7]

    # Method 1: CRT Algorithm
    N = 1
    for m in moduli:
        N *= m  # N = 3*5*7 = 105

    x = 0
    print("  CRT Construction:")
    for i, (a_i, n_i) in enumerate(zip(remainders, moduli)):
        N_i = N // n_i
        N_i_inv = pow(N_i, -1, n_i)
        term = a_i * N_i * N_i_inv
        print(f"    a_{i}={a_i}, n_{i}={n_i}, N_{i}={N_i}, N_{i}^(-1) mod {n_i}={N_i_inv}, term={term}")
        x += term

    x = x % N
    print(f"\n  CRT solution: x = {x}")
    print(f"  Verification: {x} mod 3 = {x % 3}, {x} mod 5 = {x % 5}, {x} mod 7 = {x % 7}")

    # Method 2: Brute force verification
    print("\n  Brute force search:")
    for candidate in range(N):
        if candidate % 3 == 2 and candidate % 5 == 3 and candidate % 7 == 4:
            print(f"    Found: x = {candidate}")
            print(f"    CRT and brute force agree: {candidate == x}")
            break


def exercise_4():
    """Exercise 4: Miller-Rabin Analysis (Challenging)

    1. Show that 561 is a Carmichael number.
    2. Show Miller-Rabin correctly identifies 561 as composite.
    3. How many rounds needed for false positive probability < 2^(-128)?
    """
    n = 561
    # 561 = 3 * 11 * 17

    # Part 1: Verify 561 is a Carmichael number
    # A Carmichael number satisfies: a^(n-1) ≡ 1 (mod n) for all a coprime to n.
    # By Korselt's criterion: n is Carmichael iff n is square-free and
    # for every prime p dividing n, (p-1) | (n-1).
    print("  Part 1: Carmichael number verification for 561 = 3 * 11 * 17")
    print(f"    n-1 = {n - 1}")
    print(f"    (3-1) = 2 divides 560: {560 % 2 == 0}")
    print(f"    (11-1) = 10 divides 560: {560 % 10 == 0}")
    print(f"    (17-1) = 16 divides 560: {560 % 16 == 0}")

    # Verify Fermat test passes for some bases
    test_bases = [2, 5, 10]
    print("\n    Fermat test (a^560 mod 561):")
    for a in test_bases:
        if gcd(a, n) == 1:
            result = pow(a, n - 1, n)
            print(f"      {a}^560 mod 561 = {result} {'(passes Fermat)' if result == 1 else '(FAILS)'}")

    # Part 2: Miller-Rabin catches 561
    # n-1 = 560 = 2^4 * 35, so s=4, d=35
    print("\n  Part 2: Miller-Rabin on 561")
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    print(f"    n-1 = 560 = 2^{s} * {d}")

    # Try witness a=2
    for a in [2, 3, 5, 7]:
        x = pow(a, d, n)
        sequence = [x]
        is_composite = True
        if x == 1 or x == n - 1:
            is_composite = False
        else:
            for _ in range(s - 1):
                x = pow(x, 2, n)
                sequence.append(x)
                if x == n - 1:
                    is_composite = False
                    break

        status = "COMPOSITE (witness found!)" if is_composite else "inconclusive"
        print(f"    Witness a={a}: sequence = {sequence} -> {status}")

    # Part 3: Rounds needed for error < 2^(-128)
    # Each Miller-Rabin round has error probability <= 1/4
    # After k rounds: error <= (1/4)^k = 2^(-2k)
    # Need 2^(-2k) < 2^(-128), so 2k > 128, k > 64
    print("\n  Part 3: Rounds for P(false positive) < 2^(-128)")
    print("    Each round: P(error) <= 1/4 = 2^(-2)")
    print("    After k rounds: P(error) <= (1/4)^k = 2^(-2k)")
    print("    Need: 2^(-2k) < 2^(-128)")
    print("    Therefore: k > 64 rounds needed")


def exercise_5():
    """Exercise 5: Discrete Logarithm (Challenging)

    1. Find all generators of Z_23^*.
    2. Solve 5^x ≡ 8 (mod 23) using baby-step giant-step.
    3. Explain security of 256-bit vs 2048-bit prime DLP.
    """
    import math

    p = 23

    # Part 1: Find all generators of Z_23^*
    # Z_23^* has order p-1 = 22 = 2 * 11
    # g is a generator iff g^(22/2) ≢ 1 and g^(22/11) ≢ 1 (mod 23)
    print(f"  Part 1: Generators of Z_{p}^*")
    print(f"    |Z_{p}^*| = {p - 1} = 2 * 11")
    print(f"    Number of generators = phi(22) = phi(2)*phi(11) = 1*10 = 10")

    generators = []
    for g in range(1, p):
        # Check g^((p-1)/q) != 1 for each prime factor q of p-1
        is_gen = True
        for q in [2, 11]:
            if pow(g, (p - 1) // q, p) == 1:
                is_gen = False
                break
        if is_gen:
            generators.append(g)

    print(f"    Generators: {generators}")
    print(f"    Count: {len(generators)}")

    # Verify one generator by listing all powers
    g = generators[0]
    powers = sorted([pow(g, i, p) for i in range(p - 1)])
    print(f"    Verification for g={g}: powers = {powers}")

    # Part 2: Baby-step giant-step for 5^x ≡ 8 (mod 23)
    print(f"\n  Part 2: Solve 5^x ≡ 8 (mod 23)")
    g, h = 5, 8

    m = math.isqrt(p - 1) + 1
    print(f"    m = ceil(sqrt({p - 1})) = {m}")

    # Baby steps: g^j for j = 0, ..., m-1
    baby_table = {}
    power = 1
    for j in range(m):
        baby_table[power] = j
        power = (power * g) % p

    print(f"    Baby steps: {baby_table}")

    # Giant steps: h * (g^(-m))^i
    g_inv_m = pow(g, -m, p)
    gamma = h
    x = None
    for i in range(m):
        if gamma in baby_table:
            x = i * m + baby_table[gamma]
            print(f"    Giant step i={i}: gamma={gamma} found in table at j={baby_table[gamma]}")
            print(f"    x = {i}*{m} + {baby_table[gamma]} = {x}")
            break
        gamma = (gamma * g_inv_m) % p

    if x is not None:
        print(f"    Verification: 5^{x} mod 23 = {pow(5, x, 23)}")

    # Part 3: Security analysis
    print("\n  Part 3: DLP Security Analysis")
    print("    256-bit prime p:")
    print("      - Baby-step giant-step: O(2^128) operations ~ feasible with ~2^128 space")
    print("      - Index calculus: sub-exponential, practical with specialized hardware")
    print("      - Verdict: INSECURE (equivalent to ~128-bit security at best)")
    print()
    print("    2048-bit prime p:")
    print("      - Baby-step giant-step: O(2^1024) -- completely infeasible")
    print("      - Index calculus: L(1/3, 1.923) ~ 2^112 -- still infeasible")
    print("      - Verdict: SECURE (approximately 112-bit security)")
    print()
    print("    Note: NIST recommends 2048-bit DH primes for 112-bit security.")
    print("    For 128-bit security, use 3072-bit primes (or switch to ECDH with 256-bit curves).")


if __name__ == "__main__":
    print("=== Exercise 1: Modular Arithmetic Warm-up ===")
    exercise_1()

    print("\n=== Exercise 2: Extended GCD Application ===")
    exercise_2()

    print("\n=== Exercise 3: CRT Problem ===")
    exercise_3()

    print("\n=== Exercise 4: Miller-Rabin Analysis ===")
    exercise_4()

    print("\n=== Exercise 5: Discrete Logarithm ===")
    exercise_5()

    print("\nAll exercises completed!")

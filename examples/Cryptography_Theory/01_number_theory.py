"""
Number Theory Foundations for Cryptography
==========================================
Core mathematical building blocks: GCD, modular arithmetic, primality testing, CRT.
These algorithms underpin RSA, Diffie-Hellman, and virtually all public-key cryptosystems.
"""

from __future__ import annotations
import random
import math


# ---------------------------------------------------------------------------
# Extended Euclidean Algorithm
# ---------------------------------------------------------------------------

# Why: The extended GCD is essential for computing modular inverses,
# which are needed in RSA (computing d from e) and elliptic curve arithmetic.
def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (gcd, x, y) such that a*x + b*y = gcd(a, b).

    Uses the iterative form to avoid Python recursion limits for large inputs.
    """
    # Why: Iterative version avoids stack overflow on very large integers
    # (e.g., 2048-bit RSA moduli) and is slightly faster than recursive.
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t  # gcd, x, y


# ---------------------------------------------------------------------------
# Modular Inverse
# ---------------------------------------------------------------------------

# Why: Modular inverse is the "division" operation in modular arithmetic.
# RSA decryption key d = e^{-1} mod phi(n) — without this, RSA doesn't work.
def mod_inverse(a: int, m: int) -> int:
    """Return x such that (a * x) % m == 1. Raises ValueError if no inverse."""
    gcd, x, _ = extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError(f"No modular inverse: gcd({a}, {m}) = {gcd}")
    return x % m


# ---------------------------------------------------------------------------
# Fast Modular Exponentiation (Square-and-Multiply)
# ---------------------------------------------------------------------------

# Why: Naive computation of a^b mod n would require b multiplications.
# Square-and-multiply reduces this to O(log b) multiplications, making
# RSA encryption (e ~ 65537) and signatures feasible.
def mod_pow(base: int, exp: int, mod: int) -> int:
    """Compute base^exp mod mod using the square-and-multiply algorithm."""
    if mod == 1:
        return 0
    result = 1
    base = base % mod
    while exp > 0:
        # Why: If the current bit of the exponent is 1, multiply the result.
        # This processes the exponent bit-by-bit from LSB to MSB.
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


# ---------------------------------------------------------------------------
# Miller-Rabin Primality Test
# ---------------------------------------------------------------------------

# Why: Deterministic primality tests (trial division) are too slow for
# cryptographic-size numbers (512+ bits). Miller-Rabin is probabilistic
# but with k rounds, the error probability is at most 4^{-k}.
def miller_rabin(n: int, rounds: int = 20) -> bool:
    """Probabilistic primality test. Returns True if n is probably prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    # Why: Write n-1 = 2^s * d where d is odd. This factorization is the
    # basis of the Miller-Rabin witness test — it exploits Fermat's little
    # theorem and the fact that x^2 = 1 (mod p) implies x = +/-1 for prime p.
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = mod_pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = mod_pow(x, 2, n)
            if x == n - 1:
                break
        else:
            # Why: If we never hit n-1, then a is a witness that n is composite.
            return False

    return True


# ---------------------------------------------------------------------------
# Chinese Remainder Theorem
# ---------------------------------------------------------------------------

# Why: CRT allows us to solve a system of congruences with coprime moduli.
# In RSA, CRT speeds up decryption by ~4x by working mod p and mod q
# separately rather than mod n = p*q (RSA-CRT optimization).
def chinese_remainder_theorem(
    remainders: list[int], moduli: list[int]
) -> int:
    """Solve x = r_i (mod m_i) for all i. Moduli must be pairwise coprime."""
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")

    M = math.prod(moduli)
    x = 0
    for r_i, m_i in zip(remainders, moduli):
        M_i = M // m_i
        # Why: M_i * y_i = 1 (mod m_i) gives us the "selector" that is
        # 1 mod m_i and 0 mod all other moduli — the CRT magic ingredient.
        y_i = mod_inverse(M_i, m_i)
        x += r_i * M_i * y_i

    return x % M


# ---------------------------------------------------------------------------
# Large Prime Generation
# ---------------------------------------------------------------------------

# Why: RSA and DH require large primes. We generate random odd numbers
# and test with Miller-Rabin until we find one. For n-bit primes, the
# prime number theorem says we need ~n*ln(2) candidates on average.
def generate_prime(bits: int = 512) -> int:
    """Generate a probable prime of the specified bit length."""
    while True:
        # Why: Set the top bit to ensure the number is exactly `bits` bits,
        # and set the bottom bit to make it odd (even numbers > 2 aren't prime).
        candidate = random.getrandbits(bits)
        candidate |= (1 << (bits - 1)) | 1  # ensure top bit and odd

        if miller_rabin(candidate, rounds=20):
            return candidate


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Number Theory Foundations for Cryptography")
    print("=" * 65)

    # Extended GCD
    a, b = 240, 46
    gcd, x, y = extended_gcd(a, b)
    print(f"\n[Extended GCD]")
    print(f"  gcd({a}, {b}) = {gcd}")
    print(f"  {a}*({x}) + {b}*({y}) = {a * x + b * y}")

    # Modular inverse
    a, m = 17, 3120
    inv = mod_inverse(a, m)
    print(f"\n[Modular Inverse]")
    print(f"  {a}^(-1) mod {m} = {inv}")
    print(f"  Verification: {a} * {inv} mod {m} = {(a * inv) % m}")

    # Fast modular exponentiation
    base, exp, mod = 7, 256, 13
    result = mod_pow(base, exp, mod)
    print(f"\n[Fast Modular Exponentiation]")
    print(f"  {base}^{exp} mod {mod} = {result}")
    print(f"  Verification (Python built-in): {pow(base, exp, mod)}")

    # Miller-Rabin
    print(f"\n[Miller-Rabin Primality Test]")
    test_numbers = [2, 17, 561, 1009, 1729, 104729]
    for n in test_numbers:
        label = "PRIME" if miller_rabin(n) else "COMPOSITE"
        print(f"  {n:>8} -> {label}")

    # CRT
    print(f"\n[Chinese Remainder Theorem]")
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    x = chinese_remainder_theorem(remainders, moduli)
    print(f"  System: x = {remainders[0]} (mod {moduli[0]}), "
          f"x = {remainders[1]} (mod {moduli[1]}), "
          f"x = {remainders[2]} (mod {moduli[2]})")
    print(f"  Solution: x = {x}")
    for r, m in zip(remainders, moduli):
        print(f"  Verification: {x} mod {m} = {x % m} (expected {r})")

    # Large prime generation
    print(f"\n[Large Prime Generation]")
    bits = 256
    print(f"  Generating a {bits}-bit probable prime...")
    p = generate_prime(bits)
    print(f"  p = {p}")
    print(f"  Bit length: {p.bit_length()}")
    print(f"  Miller-Rabin check: {'PRIME' if miller_rabin(p) else 'COMPOSITE'}")

    print(f"\n{'=' * 65}")

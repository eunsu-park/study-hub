"""
Exercises for Lesson 05: The RSA Cryptosystem
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
import time
from math import gcd, isqrt


def extended_gcd(a, b):
    """Extended Euclidean Algorithm."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def miller_rabin(n, k=20):
    """Miller-Rabin primality test."""
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True


def generate_prime(bits):
    """Generate a random prime of specified bit length."""
    while True:
        n = random.getrandbits(bits)
        n |= (1 << (bits - 1)) | 1
        if miller_rabin(n):
            return n


def exercise_1():
    """Exercise 1: RSA by Hand (Basic)

    Using p=61, q=53, e=17:
    1. Compute n, phi(n), d
    2. Encrypt m=65
    3. Decrypt c
    4. Sign m=42 and verify
    """
    p, q = 61, 53
    e = 17

    # Step 1: Compute n, phi(n), d
    n = p * q
    phi_n = (p - 1) * (q - 1)
    d = pow(e, -1, phi_n)

    print(f"  Step 1: Key generation")
    print(f"    p = {p}, q = {q}")
    print(f"    n = p * q = {n}")
    print(f"    phi(n) = (p-1)(q-1) = {phi_n}")
    print(f"    e = {e}")
    print(f"    d = e^(-1) mod phi(n) = {d}")
    print(f"    Verify: e*d mod phi(n) = {(e * d) % phi_n}")

    # Step 2: Encrypt m=65
    m = 65
    c = pow(m, e, n)
    print(f"\n  Step 2: Encryption")
    print(f"    m = {m}")
    print(f"    c = m^e mod n = {m}^{e} mod {n} = {c}")

    # Step 3: Decrypt
    m_recovered = pow(c, d, n)
    print(f"\n  Step 3: Decryption")
    print(f"    m = c^d mod n = {c}^{d} mod {n} = {m_recovered}")
    print(f"    Correct: {m_recovered == m}")

    # Step 4: Sign m=42
    m_sign = 42
    # Signature: sigma = m^d mod n
    sigma = pow(m_sign, d, n)
    # Verify: m == sigma^e mod n
    verified = pow(sigma, e, n)

    print(f"\n  Step 4: Signing")
    print(f"    Message: {m_sign}")
    print(f"    Signature: sigma = {m_sign}^{d} mod {n} = {sigma}")
    print(f"    Verification: sigma^e mod n = {sigma}^{e} mod {n} = {verified}")
    print(f"    Valid: {verified == m_sign}")


def exercise_2():
    """Exercise 2: RSA Key Generation Timing (Intermediate)

    Time RSA key generation and operations for various key sizes.
    """
    print(f"  RSA Performance Benchmark")
    print(f"  {'Key Size':>10} {'Keygen':>10} {'Encrypt':>10} {'Decrypt':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for bits in [512, 1024, 2048]:
        # Key generation
        start = time.time()
        half = bits // 2
        while True:
            p = generate_prime(half)
            q = generate_prime(half)
            if p != q:
                n = p * q
                phi_n = (p - 1) * (q - 1)
                e = 65537
                if gcd(e, phi_n) == 1:
                    d = pow(e, -1, phi_n)
                    break
        keygen_time = time.time() - start

        # Encryption (32-byte message as integer)
        m = random.getrandbits(min(bits - 16, 256))
        start = time.time()
        for _ in range(100):
            c = pow(m, e, n)
        enc_time = (time.time() - start) / 100

        # Decryption
        start = time.time()
        for _ in range(10):
            m_dec = pow(c, d, n)
        dec_time = (time.time() - start) / 10

        print(f"  {bits:>8}b {keygen_time:>9.3f}s {enc_time*1000:>9.2f}ms {dec_time*1000:>9.2f}ms")

    # Extrapolation for larger key sizes
    print(f"\n  Note: Modular exponentiation is O(k^3) in bit length k.")
    print(f"  Doubling key size increases decryption time by ~8x.")
    print(f"  Estimated 4096-bit RSA decryption: ~8x of 2048-bit time.")
    print(f"  Estimated 8192-bit RSA decryption: ~64x of 2048-bit time.")


def exercise_3():
    """Exercise 3: Common Modulus Attack (Intermediate)

    Given the same message encrypted with same n but different e1, e2,
    recover the message using only public information.
    """
    # Generate RSA modulus
    p = generate_prime(256)
    q = generate_prime(256)
    n = p * q

    e1 = 17
    e2 = 65537
    assert gcd(e1, e2) == 1, "e1 and e2 must be coprime"

    # Encrypt the same message with both keys
    m = random.randrange(2, n - 1)
    c1 = pow(m, e1, n)
    c2 = pow(m, e2, n)

    print(f"  Common Modulus Attack")
    print(f"    n = {str(n)[:40]}... ({n.bit_length()} bits)")
    print(f"    e1 = {e1}, e2 = {e2}")
    print(f"    m = {str(m)[:40]}...")

    # Attack: find a, b such that a*e1 + b*e2 = 1
    g, a, b = extended_gcd(e1, e2)
    assert g == 1

    print(f"\n    Extended GCD: {a}*{e1} + {b}*{e2} = {a*e1 + b*e2}")

    # Recover m = c1^a * c2^b mod n
    # Handle negative exponents: c^(-k) = (c^(-1))^k mod n
    if a < 0:
        c1_inv = pow(c1, -1, n)
        recovered = (pow(c1_inv, -a, n) * pow(c2, b, n)) % n
    elif b < 0:
        c2_inv = pow(c2, -1, n)
        recovered = (pow(c1, a, n) * pow(c2_inv, -b, n)) % n
    else:
        recovered = (pow(c1, a, n) * pow(c2, b, n)) % n

    print(f"    Recovered m = {str(recovered)[:40]}...")
    print(f"    Match: {recovered == m}")
    print(f"\n    Attack requires: c1, c2, e1, e2, n (all public!)")
    print(f"    Does NOT require: d1, d2, p, q (private)")
    print(f"    Lesson: NEVER reuse the same modulus with different public exponents.")


def exercise_4():
    """Exercise 4: Wiener's Attack (Challenging)

    1. Generate RSA with small d (d < n^(1/4) / 3).
    2. Recover d using continued fractions.
    """
    # Generate RSA with deliberately small d
    p = generate_prime(128)
    q = generate_prime(128)
    n = p * q
    phi_n = (p - 1) * (q - 1)

    # Choose small d and compute e = d^(-1) mod phi(n)
    # d must be coprime with phi(n)
    while True:
        d = random.randrange(3, isqrt(isqrt(n)) // 3)  # d < n^(1/4) / 3
        if d % 2 == 1 and gcd(d, phi_n) == 1:
            break

    e = pow(d, -1, phi_n)

    print(f"  Wiener's Attack")
    print(f"    n = {n}")
    print(f"    e = {e}")
    print(f"    (secret d = {d})")
    print(f"    n^(1/4)/3 = {isqrt(isqrt(n)) // 3}")
    print(f"    d < n^(1/4)/3: {d < isqrt(isqrt(n)) // 3}")

    # Continued fraction expansion of e/n
    def continued_fraction(num, den):
        cf = []
        while den:
            q = num // den
            cf.append(q)
            num, den = den, num - q * den
        return cf

    def convergents(cf):
        convs = []
        h_prev, h_curr = 0, 1
        k_prev, k_curr = 1, 0
        for a in cf:
            h_prev, h_curr = h_curr, a * h_curr + h_prev
            k_prev, k_curr = k_curr, a * k_curr + k_prev
            convs.append((h_curr, k_curr))
        return convs

    cf = continued_fraction(e, n)
    convs = convergents(cf)

    recovered_d = None
    for k_guess, d_guess in convs:
        if k_guess == 0:
            continue
        if d_guess <= 0:
            continue
        # Check: (e * d_guess - 1) must be divisible by k_guess
        if (e * d_guess - 1) % k_guess != 0:
            continue
        phi_guess = (e * d_guess - 1) // k_guess
        # phi(n) = n - p - q + 1, so p + q = n - phi_guess + 1
        s = n - phi_guess + 1
        # p, q are roots of x^2 - s*x + n = 0
        discriminant = s * s - 4 * n
        if discriminant < 0:
            continue
        sqrt_disc = isqrt(discriminant)
        if sqrt_disc * sqrt_disc == discriminant:
            p_test = (s + sqrt_disc) // 2
            q_test = (s - sqrt_disc) // 2
            if p_test * q_test == n:
                recovered_d = d_guess
                print(f"\n    Recovered d = {recovered_d}")
                print(f"    Correct: {recovered_d == d}")
                print(f"    Also recovered p = {p_test}, q = {q_test}")
                print(f"    Correct factors: {p_test == p or p_test == q}")
                break

    if recovered_d is None:
        print(f"    Attack failed (d may be too large for this n)")

    # Find threshold
    print(f"\n    Wiener's threshold: d < n^(1/4) / 3")
    print(f"    For this n ({n.bit_length()} bits): max d ~ {isqrt(isqrt(n)) // 3}")
    print(f"    Lesson: private exponent d must be large (~same bit length as n).")


def exercise_5():
    """Exercise 5: RSA-CRT Fault Attack (Challenging)

    Simulate the Bellcore attack: a fault in CRT decryption reveals factors.
    """
    # Generate RSA key
    p = generate_prime(128)
    q = generate_prime(128)
    n = p * q
    e = 65537
    phi_n = (p - 1) * (q - 1)
    d = pow(e, -1, phi_n)

    # CRT parameters
    dp = d % (p - 1)
    dq = d % (q - 1)
    q_inv = pow(q, -1, p)

    # Correct CRT decryption
    m = random.randrange(2, n - 1)
    c = pow(m, e, n)

    m_p = pow(c, dp, p)
    m_q = pow(c, dq, q)
    h = (q_inv * (m_p - m_q)) % p
    m_correct = m_q + h * q

    print(f"  RSA-CRT Fault Attack (Bellcore Attack)")
    print(f"    p = {p}")
    print(f"    q = {q}")
    print(f"    n = {n}")
    print(f"    Correct decryption: {m_correct == m}")

    # Simulate fault: m_p is computed incorrectly
    # In practice, faults can be induced by voltage glitching, etc.
    m_p_faulty = (m_p + random.randrange(1, p)) % p  # Inject error
    h_faulty = (q_inv * (m_p_faulty - m_q)) % p
    m_faulty = m_q + h_faulty * q

    print(f"\n    Faulty decryption (error in m_p computation):")
    print(f"    m_faulty = {m_faulty}")
    print(f"    m_faulty != m: {m_faulty != m}")

    # Attack: gcd(m_faulty^e - c, n) should reveal a factor
    # Mathematical reasoning:
    # m_faulty is correct mod q (m_q was computed correctly)
    # m_faulty is WRONG mod p (m_p was faulty)
    # So: m_faulty^e ≡ c (mod q) but m_faulty^e ≢ c (mod p)
    # Therefore: gcd(m_faulty^e - c, n) = q

    diff = (pow(m_faulty, e, n) - c) % n
    factor = gcd(diff, n)

    print(f"\n    Attack:")
    print(f"    Compute: diff = m_faulty^e - c mod n")
    print(f"    gcd(diff, n) = {factor}")
    print(f"    Factor found: {factor == p or factor == q}")
    if factor != 1 and factor != n:
        other_factor = n // factor
        print(f"    p = {factor}, q = {other_factor}")
        print(f"    Verification: p * q = n: {factor * other_factor == n}")

    print(f"\n    Countermeasure:")
    print(f"    After CRT decryption, verify: m^e ≡ c (mod n)")
    print(f"    If verification fails, a fault was detected -> abort.")
    print(f"    This adds one modular exponentiation but prevents the attack.")


if __name__ == "__main__":
    print("=== Exercise 1: RSA by Hand ===")
    exercise_1()

    print("\n=== Exercise 2: RSA Key Generation Timing ===")
    exercise_2()

    print("\n=== Exercise 3: Common Modulus Attack ===")
    exercise_3()

    print("\n=== Exercise 4: Wiener's Attack ===")
    exercise_4()

    print("\n=== Exercise 5: RSA-CRT Fault Attack ===")
    exercise_5()

    print("\nAll exercises completed!")

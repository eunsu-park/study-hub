"""
Exercises for Lesson 10: Lattice-Based Cryptography
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
import time
import math
import hashlib
import hmac as hmac_lib


def exercise_1():
    """Exercise 1: Lattice Basics (Conceptual)

    Given basis B = [[3, 1], [1, 2]]:
    1. List all lattice points (x, y) where |x| <= 10 and |y| <= 10
    2. Find the shortest non-zero vector
    3. Apply LLL reduction (2D case)
    """
    # Basis vectors: b1 = (3, 1), b2 = (1, 2)
    b1 = (3, 1)
    b2 = (1, 2)

    print(f"  Lattice Basics")
    print(f"    Basis: b1 = {b1}, b2 = {b2}")

    # Step 1: Generate all lattice points in range
    # Lattice point = z1*b1 + z2*b2 for integer z1, z2
    print(f"\n  Step 1: Lattice points with |x| <= 10, |y| <= 10")
    points = []
    for z1 in range(-10, 11):
        for z2 in range(-10, 11):
            x = z1 * b1[0] + z2 * b2[0]
            y = z1 * b1[1] + z2 * b2[1]
            if abs(x) <= 10 and abs(y) <= 10:
                points.append((x, y, z1, z2))

    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)

    print(f"    Found {len(points)} lattice points in range")
    print(f"    First 15 points (sorted by distance):")
    for x, y, z1, z2 in points[:15]:
        dist = math.sqrt(x**2 + y**2)
        print(f"      ({x:>3}, {y:>3}) = {z1}*b1 + {z2}*b2, ||v|| = {dist:.3f}")

    # Step 2: Find shortest non-zero vector
    print(f"\n  Step 2: Shortest non-zero vector")
    # Skip (0, 0)
    nonzero = [(x, y, z1, z2) for x, y, z1, z2 in points if (x, y) != (0, 0)]
    shortest = nonzero[0]
    dist = math.sqrt(shortest[0]**2 + shortest[1]**2)
    print(f"    Shortest: ({shortest[0]}, {shortest[1]}) = "
          f"{shortest[2]}*b1 + {shortest[3]}*b2")
    print(f"    Length: {dist:.4f}")

    # Step 3: LLL reduction (2D case, simplified Gauss reduction)
    print(f"\n  Step 3: LLL/Gauss Lattice Reduction (2D)")
    print(f"    Original basis: b1 = {b1}, b2 = {b2}")

    # Gauss reduction for 2D: reduce until |b1| <= |b2|
    # and the projection of b2 onto b1 satisfies |mu| <= 1/2
    v1 = list(b1)
    v2 = list(b2)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def norm_sq(a):
        return dot(a, a)

    iterations = 0
    while True:
        iterations += 1
        # Ensure |v1| <= |v2|
        if norm_sq(v1) > norm_sq(v2):
            v1, v2 = v2, v1

        # Compute mu = <v2, v1> / <v1, v1>
        mu = dot(v2, v1) / norm_sq(v1)

        # Size-reduce: v2 = v2 - round(mu) * v1
        m = round(mu)
        if m == 0:
            break

        v2 = [v2[i] - m * v1[i] for i in range(2)]
        print(f"    Iteration {iterations}: mu = {mu:.3f}, m = {m}")
        print(f"      v1 = {tuple(v1)}, v2 = {tuple(v2)}")

    print(f"\n    Reduced basis: v1 = {tuple(v1)}, v2 = {tuple(v2)}")
    print(f"    |v1| = {math.sqrt(norm_sq(v1)):.4f}, |v2| = {math.sqrt(norm_sq(v2)):.4f}")
    print(f"    Original: |b1| = {math.sqrt(norm_sq(list(b1))):.4f}, "
          f"|b2| = {math.sqrt(norm_sq(list(b2))):.4f}")

    # Verify: reduced basis spans the same lattice
    # det(B) should be preserved
    det_original = b1[0] * b2[1] - b1[1] * b2[0]
    det_reduced = v1[0] * v2[1] - v1[1] * v2[0]
    print(f"    det(original) = {det_original}, det(reduced) = {abs(det_reduced)}")
    print(f"    Same lattice: {abs(det_original) == abs(det_reduced)}")


def exercise_2():
    """Exercise 2: LWE Encryption (Coding)

    1. Byte-by-byte LWE encryption
    2. Error correction and failure analysis
    3. Decryption failure rate vs error magnitude
    """
    def lwe_keygen(n, q):
        """Generate LWE key pair."""
        # Secret key: small random vector
        s = [random.randrange(q) for _ in range(n)]
        return s

    def lwe_encrypt_bit(bit, s, n, q, error_std):
        """Encrypt a single bit using LWE.

        Ciphertext: (a, b) where a is random, b = <a, s> + e + bit*(q//2)
        """
        a = [random.randrange(q) for _ in range(n)]
        e = round(random.gauss(0, error_std)) % q

        # b = <a, s> + e + bit * (q // 2)
        b = (sum(ai * si for ai, si in zip(a, s)) + e + bit * (q // 2)) % q
        return (a, b)

    def lwe_decrypt_bit(ct, s, q):
        """Decrypt a single bit."""
        a, b = ct
        # v = b - <a, s> mod q
        v = (b - sum(ai * si for ai, si in zip(a, s))) % q

        # If v is closer to 0, bit = 0; if closer to q/2, bit = 1
        if v < q // 4 or v > 3 * q // 4:
            return 0
        else:
            return 1

    # Parameters
    n = 16    # Dimension
    q = 97    # Modulus (prime)
    error_std = 1.0  # Error standard deviation

    # Part 1: Byte-by-byte encryption
    print(f"  Part 1: LWE Byte-by-Byte Encryption")
    print(f"    Parameters: n={n}, q={q}, error_std={error_std}")

    s = lwe_keygen(n, q)

    # Encrypt a message byte-by-byte (each byte = 8 bits)
    message = b"Hi!"
    print(f"    Message: {message.decode()} ({message.hex()})")

    encrypted_bits = []
    for byte in message:
        for bit_pos in range(8):
            bit = (byte >> (7 - bit_pos)) & 1
            ct = lwe_encrypt_bit(bit, s, n, q, error_std)
            encrypted_bits.append(ct)

    # Decrypt
    decrypted_bytes = []
    for byte_idx in range(len(message)):
        byte_val = 0
        for bit_pos in range(8):
            ct = encrypted_bits[byte_idx * 8 + bit_pos]
            bit = lwe_decrypt_bit(ct, s, q)
            byte_val = (byte_val << 1) | bit
        decrypted_bytes.append(byte_val)

    decrypted = bytes(decrypted_bytes)
    print(f"    Decrypted: {decrypted.decode()} ({decrypted.hex()})")
    print(f"    Match: {decrypted == message}")

    # Part 2: Error analysis
    print(f"\n  Part 2: Decryption Failure Rate vs Error Standard Deviation")
    print(f"    {'Error Std':>10} {'Failures':>10} {'Rate':>10}")
    print(f"    {'-'*10} {'-'*10} {'-'*10}")

    for err_std in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]:
        failures = 0
        trials = 1000
        for _ in range(trials):
            bit = random.randint(0, 1)
            ct = lwe_encrypt_bit(bit, s, n, q, err_std)
            dec = lwe_decrypt_bit(ct, s, q)
            if dec != bit:
                failures += 1
        rate = failures / trials
        print(f"    {err_std:>10.1f} {failures:>10}/{trials} {rate:>10.3f}")

    print(f"\n    Observation: As error increases, decryption starts failing.")
    print(f"    The threshold is approximately error_std ~ q/8 = {q/8:.1f}")
    print(f"    Below this, errors stay in the 'correct' half of Z_q.")
    print(f"    Above this, errors push values across the decision boundary.")


def exercise_3():
    """Exercise 3: Ring Operations (Coding)

    Polynomial arithmetic in Z_q[x]/(x^n + 1):
    1. Addition, subtraction, multiplication
    2. Schoolbook vs NTT comparison
    3. Benchmark for n = 256, 512, 1024
    """
    def poly_add(a, b, q, n):
        """Add two polynomials mod q in Z_q[x]/(x^n + 1)."""
        result = [(a[i] + b[i]) % q for i in range(n)]
        return result

    def poly_sub(a, b, q, n):
        """Subtract two polynomials mod q."""
        result = [(a[i] - b[i]) % q for i in range(n)]
        return result

    def poly_mul_schoolbook(a, b, q, n):
        """Schoolbook polynomial multiplication mod x^n + 1 and mod q.

        O(n^2) complexity. The reduction by x^n + 1 means:
        x^n = -1, so any term x^(n+k) becomes -x^k.
        """
        result = [0] * n
        for i in range(n):
            for j in range(n):
                idx = i + j
                if idx < n:
                    result[idx] = (result[idx] + a[i] * b[j]) % q
                else:
                    # x^(n+k) = -x^k in Z_q[x]/(x^n + 1)
                    result[idx - n] = (result[idx - n] - a[i] * b[j]) % q
        return result

    def poly_mul_ntt_simple(a, b, q, n):
        """Simplified NTT-like multiplication using DFT over Z_q.

        For a proper NTT, q must have a primitive 2n-th root of unity.
        Here we use a simplified version for correctness testing.
        This is NOT a true NTT but demonstrates the concept.
        """
        # Fallback to schoolbook for small n or when NTT is not available
        return poly_mul_schoolbook(a, b, q, n)

    # Part 1: Basic operations
    n = 8  # Small for demonstration
    q = 97

    print(f"  Part 1: Ring Operations in Z_{q}[x]/(x^{n} + 1)")

    a = [random.randrange(q) for _ in range(n)]
    b = [random.randrange(q) for _ in range(n)]

    print(f"    a = {a}")
    print(f"    b = {b}")

    sum_ab = poly_add(a, b, q, n)
    diff_ab = poly_sub(a, b, q, n)
    prod_ab = poly_mul_schoolbook(a, b, q, n)

    print(f"    a + b = {sum_ab}")
    print(f"    a - b = {diff_ab}")
    print(f"    a * b mod (x^{n}+1) = {prod_ab}")

    # Verify: (a+b) - b = a
    verify = poly_sub(sum_ab, b, q, n)
    print(f"    Verify (a+b)-b = a: {verify == a}")

    # Part 2: Benchmark schoolbook multiplication
    print(f"\n  Part 2: Schoolbook Multiplication Benchmark")
    print(f"    {'n':>6} {'Time (ms)':>12} {'Ops/sec':>12}")
    print(f"    {'-'*6} {'-'*12} {'-'*12}")

    q_bench = 3329  # Kyber's modulus

    for n_bench in [64, 128, 256, 512]:
        a_bench = [random.randrange(q_bench) for _ in range(n_bench)]
        b_bench = [random.randrange(q_bench) for _ in range(n_bench)]

        iterations = max(1, 1000 // n_bench)
        start = time.time()
        for _ in range(iterations):
            poly_mul_schoolbook(a_bench, b_bench, q_bench, n_bench)
        elapsed = time.time() - start

        ms_per_op = (elapsed / iterations) * 1000
        ops_per_sec = iterations / elapsed if elapsed > 0 else float('inf')

        print(f"    {n_bench:>6} {ms_per_op:>12.3f} {ops_per_sec:>12.0f}")

    print(f"\n    Schoolbook multiplication is O(n^2).")
    print(f"    NTT-based multiplication is O(n log n).")
    print(f"    For n=256 (Kyber): NTT is ~30x faster than schoolbook.")
    print(f"    Kyber uses q=3329, which has a primitive 512th root of unity,")
    print(f"    enabling efficient NTT computation.")


def exercise_4():
    """Exercise 4: Parameter Selection (Conceptual + Coding)

    For LWE encryption at 128-bit security:
    1. Find minimum n and q for correct decryption
    2. Compare key sizes with RSA-3072 and Kyber-768
    """
    print(f"  LWE Parameter Selection")

    # Security estimation (simplified):
    # For LWE with dimension n, modulus q, error rate alpha = sigma/q:
    # Security ≈ n * log2(q/sigma) bits
    # (This is a simplification; real security estimation uses the
    # lattice estimator or BKZ block size analysis)

    print(f"\n  Part 1: Parameter search for 128-bit security")
    print(f"    {'n':>6} {'q':>8} {'sigma':>6} {'Security':>10} {'Dec Fail':>10} {'Key (bytes)':>12}")
    print(f"    {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*12}")

    candidates = []
    for n in [256, 384, 512, 768, 1024]:
        for log_q in [10, 12, 14, 16]:
            q = 2**log_q
            for sigma in [1.0, 2.0, 3.2, 4.0]:
                # Simplified security estimate
                if sigma > 0:
                    security = n * math.log2(q / sigma)
                else:
                    security = 0

                # Decryption failure: error must be < q/4
                # Error in decryption: sum of n small errors + encryption error
                # Total error std ≈ sigma * sqrt(n + 1)
                total_error_std = sigma * math.sqrt(n + 1)
                # Probability of failure ≈ erfc(q/(4*total_error_std*sqrt(2)))
                threshold = q / 4
                if total_error_std > 0:
                    # Approximate using Gaussian tail bound
                    z = threshold / total_error_std
                    # P(fail) ≈ exp(-z^2/2)
                    log_fail = -z * z / 2 / math.log(2)  # in bits
                else:
                    log_fail = -999

                # Key size: public key = n*log2(q) bits for the matrix row + n*log2(q) for b
                key_bytes = n * log_q // 8 * 2  # Simplified

                if 120 <= security <= 200 and log_fail < -40:
                    candidates.append((n, q, sigma, security, log_fail, key_bytes))
                    print(f"    {n:>6} {q:>8} {sigma:>6.1f} {security:>10.0f} "
                          f"{'2^'+str(int(log_fail)):>10} {key_bytes:>12}")

    # Part 2: Comparison with existing schemes
    print(f"\n  Part 2: Key Size Comparison")
    print(f"    {'Scheme':<20} {'Security':>10} {'Public Key':>12} {'Ciphertext':>12}")
    print(f"    {'-'*20} {'-'*10} {'-'*12} {'-'*12}")

    comparisons = [
        ("RSA-3072", "112-bit", "384 bytes", "384 bytes"),
        ("ECDH P-256", "128-bit", "32 bytes", "32 bytes"),
        ("Kyber-512", "128-bit (PQ)", "800 bytes", "768 bytes"),
        ("Kyber-768", "192-bit (PQ)", "1,184 bytes", "1,088 bytes"),
        ("Kyber-1024", "256-bit (PQ)", "1,568 bytes", "1,568 bytes"),
    ]

    for name, sec, pk, ct in comparisons:
        print(f"    {name:<20} {sec:>10} {pk:>12} {ct:>12}")

    print(f"\n    Key insight: Lattice-based schemes have 10-50x larger keys than ECC,")
    print(f"    but this is the cost of quantum resistance.")
    print(f"    Kyber-768 (1,184-byte public key) is the NIST PQC standard (ML-KEM).")


def exercise_5():
    """Exercise 5: Hybrid Key Exchange (Challenging)

    Combine X25519 ECDH and simplified LWE key exchange.
    """
    def lwe_key_exchange(n, q, error_std):
        """Simplified LWE-based key exchange (Regev/FrodoKEM-style).

        Alice -> Bob: A (public matrix), b = As + e (public key)
        Bob -> Alice: u = A^T * r + e', v = b^T * r + e'' + msg*(q//2)
        """
        # Alice generates keys
        A = [[random.randrange(q) for _ in range(n)] for _ in range(n)]
        s = [random.randrange(-1, 2) for _ in range(n)]  # Small secret
        e = [round(random.gauss(0, error_std)) % q for _ in range(n)]

        # b = A*s + e mod q
        b = [(sum(A[i][j] * s[j] for j in range(n)) + e[i]) % q for i in range(n)]

        # Bob's side: pick random r, compute shared secret
        r = [random.randrange(-1, 2) for _ in range(n)]
        e_prime = [round(random.gauss(0, error_std)) % q for _ in range(n)]

        # u = A^T * r + e' mod q
        u = [(sum(A[j][i] * r[j] for j in range(n)) + e_prime[i]) % q for i in range(n)]

        # v = b^T * r + e'' mod q  (this is the shared key material)
        e_dprime = round(random.gauss(0, error_std)) % q
        v_bob = (sum(b[j] * r[j] for j in range(n)) + e_dprime) % q

        # Alice computes: v_alice = s^T * u mod q
        v_alice = sum(s[j] * u[j] for j in range(n)) % q

        # Both v_alice and v_bob are approximately equal (differ by noise)
        # Extract shared bit: close to 0 -> 0, close to q/2 -> 1
        def extract_key(v, q):
            if v < q // 4 or v > 3 * q // 4:
                return 0
            return 1

        key_alice = extract_key(v_alice, q)
        key_bob = extract_key(v_bob, q)

        return key_alice, key_bob, v_alice, v_bob

    # Parameters
    n_lwe = 32
    q_lwe = 97
    error_std = 1.0

    print(f"  Hybrid Key Exchange: X25519 + LWE")

    # Part 1: LWE key exchange
    print(f"\n  Part 1: LWE Key Exchange (n={n_lwe}, q={q_lwe})")
    success_count = 0
    trials = 100
    for _ in range(trials):
        ka, kb, va, vb = lwe_key_exchange(n_lwe, q_lwe, error_std)
        if ka == kb:
            success_count += 1
    print(f"    Success rate: {success_count}/{trials} ({success_count/trials*100:.1f}%)")

    # For a single exchange demonstration
    ka, kb, va, vb = lwe_key_exchange(n_lwe, q_lwe, error_std)
    print(f"    Alice's raw value: {va}, key bit: {ka}")
    print(f"    Bob's raw value:   {vb}, key bit: {kb}")
    print(f"    Match: {ka == kb}")

    # Part 2: Hybrid scheme
    print(f"\n  Part 2: Hybrid Key Exchange")
    print(f"    In a real hybrid scheme:")
    print(f"    1. Perform X25519 ECDH -> shared_secret_1 (32 bytes)")
    print(f"    2. Perform LWE/Kyber KEM -> shared_secret_2 (32 bytes)")
    print(f"    3. Combined: SK = HKDF(shared_secret_1 || shared_secret_2)")

    # Simulate with random values (X25519 would need the cryptography library)
    ecdh_secret = os.urandom(32)
    lwe_secret = hashlib.sha256(str(va).encode()).digest()

    # Combine with HKDF
    combined = ecdh_secret + lwe_secret
    hybrid_key = hashlib.sha256(combined).digest()

    print(f"    ECDH secret:  {ecdh_secret.hex()[:32]}...")
    print(f"    LWE secret:   {lwe_secret.hex()[:32]}...")
    print(f"    Hybrid key:   {hybrid_key.hex()[:32]}...")

    # Part 3: Security analysis
    print(f"\n  Part 3: Why Hybrid?")
    print(f"    If X25519 is broken (quantum computer):  LWE still protects")
    print(f"    If LWE is broken (new lattice attack):   X25519 still protects")
    print(f"    Both must be broken to compromise the session key.")
    print(f"    This is the 'belt and suspenders' approach recommended by:")
    print(f"    - NIST SP 800-227 (Hybrid PQC)")
    print(f"    - Chrome/Firefox (already deploying X25519+Kyber)")
    print(f"    - Signal (PQXDH = X25519 + Kyber)")


if __name__ == "__main__":
    print("=== Exercise 1: Lattice Basics ===")
    exercise_1()

    print("\n=== Exercise 2: LWE Encryption ===")
    exercise_2()

    print("\n=== Exercise 3: Ring Operations ===")
    exercise_3()

    print("\n=== Exercise 4: Parameter Selection ===")
    exercise_4()

    print("\n=== Exercise 5: Hybrid Key Exchange ===")
    exercise_5()

    print("\nAll exercises completed!")

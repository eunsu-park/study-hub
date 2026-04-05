"""
Exercises for Lesson 13: Homomorphic Encryption
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
import math
import hashlib
import time
from math import gcd


def exercise_1():
    """Exercise 1: RSA Homomorphism (Coding)

    1. Implement textbook RSA (small primes for demo)
    2. Demonstrate multiplicative homomorphism
    3. Show additive homomorphism does NOT hold
    4. Explain semantic security failure of textbook RSA
    """
    print(f"  RSA Multiplicative Homomorphism")

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def generate_rsa_keys(p, q):
        """Generate RSA keys from two primes."""
        n = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        if gcd(e, phi) != 1:
            # Find a valid e
            for candidate in range(3, phi, 2):
                if gcd(candidate, phi) == 1:
                    e = candidate
                    break
        d = pow(e, -1, phi)
        return (e, n), (d, n)

    # Use small primes for demonstration
    p_rsa, q_rsa = 61, 53
    pub, priv = generate_rsa_keys(p_rsa, q_rsa)
    e, n = pub
    d, _ = priv

    print(f"    RSA parameters: p={p_rsa}, q={q_rsa}, N={n}")
    print(f"    Public key e = {e}")
    print(f"    Private key d = {d}")

    # Part 1: Basic encryption/decryption
    print(f"\n  Part 1: Basic Encryption/Decryption")
    for m in [7, 42, 100]:
        c = pow(m, e, n)
        m_dec = pow(c, d, n)
        print(f"    m={m}: Enc(m)={c}, Dec(Enc(m))={m_dec}, correct={m == m_dec}")

    # Part 2: Multiplicative homomorphism
    print(f"\n  Part 2: Multiplicative Homomorphism")
    m1, m2 = 7, 13
    c1 = pow(m1, e, n)
    c2 = pow(m2, e, n)

    # Multiply ciphertexts
    c_product = (c1 * c2) % n
    dec_product = pow(c_product, d, n)

    print(f"    m1={m1}, m2={m2}")
    print(f"    Enc(m1) = {c1}")
    print(f"    Enc(m2) = {c2}")
    print(f"    Enc(m1) * Enc(m2) mod N = {c_product}")
    print(f"    Dec(Enc(m1) * Enc(m2)) = {dec_product}")
    print(f"    m1 * m2 = {m1 * m2}")
    print(f"    Multiplicative homomorphism: {dec_product == m1 * m2}")

    # Chain: multiply 3 values
    m3 = 5
    c3 = pow(m3, e, n)
    c_triple = (c1 * c2 * c3) % n
    dec_triple = pow(c_triple, d, n)
    print(f"\n    Chain: m1*m2*m3 = {m1}*{m2}*{m3} = {m1*m2*m3}")
    print(f"    Dec(Enc(m1)*Enc(m2)*Enc(m3)) = {dec_triple}")
    print(f"    Correct: {dec_triple == m1 * m2 * m3}")

    # Part 3: Additive homomorphism does NOT hold
    print(f"\n  Part 3: Additive Homomorphism (DOES NOT HOLD)")
    c_sum = (c1 + c2) % n
    dec_sum = pow(c_sum, d, n)

    print(f"    Enc(m1) + Enc(m2) mod N = {c_sum}")
    print(f"    Dec(Enc(m1) + Enc(m2)) = {dec_sum}")
    print(f"    m1 + m2 = {m1 + m2}")
    print(f"    Additive homomorphism: {dec_sum == m1 + m2}")
    print(f"    RSA is NOT additively homomorphic!")

    # Part 4: Semantic security
    print(f"\n  Part 4: Why Textbook RSA is NOT Semantically Secure")
    print(f"    Problem 1: Deterministic encryption")
    c_same_1 = pow(42, e, n)
    c_same_2 = pow(42, e, n)
    print(f"    Enc(42) = {c_same_1}")
    print(f"    Enc(42) = {c_same_2}")
    print(f"    Same plaintext -> same ciphertext: {c_same_1 == c_same_2}")
    print(f"    An attacker can detect repeated messages!")
    print(f"")
    print(f"    Problem 2: Malleability via homomorphism")
    print(f"    If attacker knows Enc(m), they can compute Enc(2*m)")
    print(f"    without knowing m:")
    c_m = pow(42, e, n)
    c_2 = pow(2, e, n)
    c_2m = (c_m * c_2) % n
    dec_2m = pow(c_2m, d, n)
    print(f"    Enc(42) * Enc(2) mod N -> Dec = {dec_2m} = 2*42 = {2*42}")
    print(f"")
    print(f"    Fix: Use RSA-OAEP padding (randomized, not homomorphic)")


def exercise_2():
    """Exercise 2: Simplified Paillier Applications (Coding)

    Using a simplified additive homomorphic scheme:
    1. Private voting
    2. Encrypted mean computation
    3. Analysis of encrypted variance feasibility
    """
    print(f"  Paillier Applications (Simplified Additive HE)")

    # Simplified additive HE using modular arithmetic for demonstration
    # In production, use actual Paillier with large primes
    class SimpleAdditiveHE:
        """Simplified additive HE for demonstration.

        Uses a simple masking scheme: Enc(m) = m + r*N mod N^2
        This is NOT cryptographically secure -- just for demonstrating
        the homomorphic property. Use real Paillier for security.
        """

        def __init__(self, N=997):
            self.N = N
            self.N_sq = N * N

        def encrypt(self, m):
            r = random.randint(1, self.N - 1)
            return (m + r * self.N) % self.N_sq

        def decrypt(self, c):
            return c % self.N

        def add_encrypted(self, c1, c2):
            return (c1 + c2) % self.N_sq

        def scalar_multiply(self, c, k):
            return (c * k) % self.N_sq

    he = SimpleAdditiveHE(N=997)

    # Verify basic operation
    m1, m2 = 42, 58
    c1, c2 = he.encrypt(m1), he.encrypt(m2)
    c_sum = he.add_encrypted(c1, c2)
    dec_sum = he.decrypt(c_sum)
    print(f"    Basic test: Dec(Enc({m1}) + Enc({m2})) = {dec_sum}, "
          f"expected {m1 + m2}, correct={dec_sum == m1 + m2}")

    # Part 1: Private Voting
    print(f"\n  Part 1: Private Voting System")
    print(f"    Each voter encrypts 1 (yes) or 0 (no)")
    print(f"    Tally = sum of all encrypted votes (homomorphic)")

    votes = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    num_voters = len(votes)
    yes_count = sum(votes)
    no_count = num_voters - yes_count

    print(f"    Voters: {num_voters}")
    print(f"    True tally: {yes_count} yes, {no_count} no")

    # Each voter encrypts their vote
    encrypted_votes = [he.encrypt(v) for v in votes]

    # Election authority sums encrypted votes (never sees individual votes)
    encrypted_tally = encrypted_votes[0]
    for ev in encrypted_votes[1:]:
        encrypted_tally = he.add_encrypted(encrypted_tally, ev)

    # Decrypt only the total
    decrypted_tally = he.decrypt(encrypted_tally)
    print(f"    Decrypted tally: {decrypted_tally} yes votes")
    print(f"    Correct: {decrypted_tally == yes_count}")

    # Verify: each encrypted vote looks different (randomized)
    ev1a = he.encrypt(1)
    ev1b = he.encrypt(1)
    print(f"    Enc(1) = {ev1a}, Enc(1) = {ev1b}, "
          f"same? {ev1a == ev1b} (should be False)")

    # Part 2: Encrypted Mean
    print(f"\n  Part 2: Encrypted Mean Computation")
    salaries = [50000, 65000, 72000, 48000, 55000, 80000, 61000, 59000]
    n_emp = len(salaries)
    true_sum = sum(salaries)
    true_mean = true_sum / n_emp

    print(f"    {n_emp} employees, true mean salary: ${true_mean:,.0f}")

    # Encrypt each salary
    encrypted_salaries = [he.encrypt(s % he.N) for s in salaries]

    # Sum homomorphically
    encrypted_sum = encrypted_salaries[0]
    for es in encrypted_salaries[1:]:
        encrypted_sum = he.add_encrypted(encrypted_sum, es)

    # Decrypt the sum
    decrypted_sum = he.decrypt(encrypted_sum)
    adjusted_sum = true_sum % he.N  # account for modular arithmetic
    print(f"    Decrypted sum mod N: {decrypted_sum}")
    print(f"    True sum mod N: {adjusted_sum}")
    print(f"    Match: {decrypted_sum == adjusted_sum}")

    # For exact mean: need scalar multiplication by n^(-1) mod N
    n_inv = pow(n_emp, -1, he.N)
    encrypted_mean = he.scalar_multiply(encrypted_sum, n_inv)
    decrypted_mean = he.decrypt(encrypted_mean)
    expected_mean = (adjusted_sum * n_inv) % he.N

    print(f"    n^(-1) mod N = {n_inv}")
    print(f"    Decrypted mean (mod N): {decrypted_mean}")
    print(f"    Expected mean (mod N): {expected_mean}")
    print(f"    Match: {decrypted_mean == expected_mean}")

    # Part 3: Can we compute encrypted variance?
    print(f"\n  Part 3: Encrypted Variance -- Can We Do It?")
    print(f"    Variance = E[X^2] - E[X]^2")
    print(f"    This requires:")
    print(f"    1. Squaring encrypted values: Enc(x_i) -> Enc(x_i^2)")
    print(f"       This is a MULTIPLICATION of two encrypted values!")
    print(f"    2. Additively homomorphic schemes (Paillier) cannot do this.")
    print(f"")
    print(f"    Options:")
    print(f"    a) Use FHE (BFV/CKKS): supports both + and *")
    print(f"       Enc(x)^2 is possible, then sum the squares")
    print(f"    b) Interactive protocol: client sends Enc(x_i^2) alongside Enc(x_i)")
    print(f"       Server computes sum of Enc(x_i^2) homomorphically")
    print(f"       But client must be trusted to send consistent values")
    print(f"    c) Hybrid: use Paillier for sums, ZKP to prove consistency")
    print(f"       Each client proves Enc(x_i^2) is the square of Enc(x_i)")
    print(f"")

    # Demonstrate option (b): client pre-computes squares
    enc_squares = [he.encrypt((s * s) % he.N) for s in salaries]
    enc_sum_sq = enc_squares[0]
    for es in enc_squares[1:]:
        enc_sum_sq = he.add_encrypted(enc_sum_sq, es)

    dec_sum_sq = he.decrypt(enc_sum_sq)
    true_sum_sq = sum(s * s for s in salaries) % he.N
    print(f"    Option (b) demo: sum of squares")
    print(f"    Dec(sum(Enc(x_i^2))) = {dec_sum_sq}")
    print(f"    True sum(x_i^2) mod N = {true_sum_sq}")
    print(f"    Match: {dec_sum_sq == true_sum_sq}")


def exercise_3():
    """Exercise 3: Noise Budget Simulation (Coding + Conceptual)

    Simulate noise growth in LWE-based homomorphic encryption.
    """
    print(f"  Noise Budget Simulation")

    initial_noise = 3.0
    q = 2**32
    threshold = q / 4.0

    print(f"    Initial noise: {initial_noise}")
    print(f"    Modulus q: 2^32 = {q}")
    print(f"    Decryption threshold: q/4 = {threshold:.0f}")

    # Part 1: Addition chain
    print(f"\n  Part 1: Addition Chain")
    print(f"    After addition: noise_new = noise_a + noise_b")
    print(f"    {'Step':>6} {'Noise':>15} {'Budget Used':>15} {'Status':>10}")
    print(f"    {'-'*6} {'-'*15} {'-'*15} {'-'*10}")

    noise = initial_noise
    add_max_depth = 0
    for step in range(1, 40):
        noise = noise + initial_noise  # adding a fresh ciphertext each time
        budget_pct = noise / threshold * 100
        status = "OK" if noise < threshold else "FAIL"
        if step <= 10 or step % 5 == 0 or noise >= threshold:
            print(f"    {step:>6} {noise:>15.1f} {budget_pct:>14.6f}% {status:>10}")
        if noise < threshold:
            add_max_depth = step
        if noise >= threshold:
            break

    print(f"    Max additions before failure: {add_max_depth}")
    # Theoretical: noise = (steps+1) * initial_noise < q/4
    # steps < q/(4*initial_noise) - 1
    theoretical_max = int(threshold / initial_noise) - 1
    print(f"    Theoretical max: floor(q/4/e0) - 1 = {theoretical_max}")

    # Part 2: Multiplication chain
    print(f"\n  Part 2: Multiplication Chain")
    print(f"    After multiplication: noise_new ~ noise_a * noise_b")
    print(f"    {'Step':>6} {'Noise':>15} {'log2(noise)':>15} {'Status':>10}")
    print(f"    {'-'*6} {'-'*15} {'-'*15} {'-'*10}")

    noise = initial_noise
    mul_max_depth = 0
    for step in range(1, 40):
        noise = noise * initial_noise  # simplified: multiply by fresh noise
        log_noise = math.log2(noise) if noise > 0 else 0
        log_thresh = math.log2(threshold)
        status = "OK" if noise < threshold else "FAIL"
        if step <= 10 or noise >= threshold:
            print(f"    {step:>6} {noise:>15.1f} {log_noise:>15.2f} {status:>10}")
        if noise < threshold:
            mul_max_depth = step
        if noise >= threshold:
            break

    print(f"    Max multiplications before failure: {mul_max_depth}")
    # Theoretical: noise = initial^(steps+1) < q/4
    # (steps+1) * log(initial) < log(q/4)
    theoretical_mul = int(math.log(threshold) / math.log(initial_noise)) - 1
    print(f"    Theoretical max: floor(log(q/4)/log(e0)) - 1 = {theoretical_mul}")

    # Part 3: Realistic multiplication (noise squares each step)
    print(f"\n  Part 3: Realistic Noise Growth (noise squares per multiplication)")
    print(f"    After mult of two depth-d ciphertexts: noise ~ noise^2")
    print(f"    {'Step':>6} {'log2(noise)':>15} {'Budget':>15} {'Status':>10}")
    print(f"    {'-'*6} {'-'*15} {'-'*15} {'-'*10}")

    log_noise = math.log2(initial_noise)
    log_thresh = math.log2(threshold)
    sq_max_depth = 0

    for step in range(1, 35):
        log_noise = log_noise * 2  # noise squares (doubles in log space)
        budget = log_noise / log_thresh * 100
        status = "OK" if log_noise < log_thresh else "FAIL"
        print(f"    {step:>6} {log_noise:>15.2f} {budget:>14.2f}% {status:>10}")
        if log_noise < log_thresh:
            sq_max_depth = step
        if log_noise >= log_thresh:
            break

    print(f"    Max multiplications (squaring noise): {sq_max_depth}")
    print(f"    This is why deep circuits need bootstrapping!")

    # Part 4: Mixed circuit (3-layer neural network approximation)
    print(f"\n  Part 4: Mixed Circuit (3-Layer Neural Network)")
    print(f"    Layer 1: 4 additions (weighted sum), 1 multiplication (activation)")
    print(f"    Layer 2: 4 additions, 1 multiplication")
    print(f"    Layer 3: 2 additions, 1 multiplication")

    noise = initial_noise
    log_q = math.log2(q)
    steps = []

    # Layer 1: 4 additions + 1 multiplication
    for _ in range(4):
        noise = noise + initial_noise
    steps.append(("L1 add x4", noise, math.log2(noise)))
    noise = noise * noise  # polynomial activation approximation
    steps.append(("L1 mult", noise, math.log2(noise)))

    # Layer 2: 4 additions + 1 multiplication
    for _ in range(4):
        noise = noise + noise * 0.1  # adding smaller-noise terms
    steps.append(("L2 add x4", noise, math.log2(noise)))
    noise = noise * noise
    steps.append(("L2 mult", noise, math.log2(noise)))

    # Layer 3: 2 additions + 1 multiplication
    for _ in range(2):
        noise = noise + noise * 0.1
    steps.append(("L3 add x2", noise, math.log2(noise)))
    noise = noise * noise
    steps.append(("L3 mult", noise, math.log2(noise)))

    print(f"    {'Operation':<14} {'Noise':>15} {'log2(noise)':>12} {'Status':>8}")
    print(f"    {'-'*14} {'-'*15} {'-'*12} {'-'*8}")
    for op, n_val, log_n in steps:
        status = "OK" if n_val < threshold else "FAIL"
        if n_val < 1e15:
            print(f"    {op:<14} {n_val:>15.1f} {log_n:>12.2f} {status:>8}")
        else:
            print(f"    {op:<14} {'overflow':>15} {log_n:>12.2f} {status:>8}")

    print(f"\n    Conclusion: Even a 3-layer network exhausts the noise budget.")
    print(f"    Solutions:")
    print(f"    1. Bootstrapping: reset noise after each layer (~1000x slower)")
    print(f"    2. Leveled FHE: set q large enough for known depth")
    print(f"    3. CKKS rescaling: manage noise by scaling down after each mult")


def exercise_4():
    """Exercise 4: Encrypted Linear Model Inference (Coding)

    Build a complete encrypted linear inference pipeline.
    """
    print(f"  Encrypted Linear Model Inference")

    # Simplified additive HE for demo
    class SimpleHE:
        def __init__(self, N=10007):
            self.N = N
            self.N_sq = N * N

        def encrypt(self, m):
            r = random.randint(1, self.N - 1)
            return (m % self.N + r * self.N) % self.N_sq

        def decrypt(self, c):
            result = c % self.N
            if result > self.N // 2:
                result -= self.N
            return result

        def add_encrypted(self, c1, c2):
            return (c1 + c2) % self.N_sq

        def scalar_multiply(self, c, k):
            return (c * k) % self.N_sq

    he = SimpleHE(N=10007)

    # Part 1: Train a simple linear model (plaintext)
    print(f"\n  Part 1: Plaintext Linear Model")

    # Simple 3-feature model: predict house price category
    # Features: [size_score, location_score, age_score] (0-100 scale)
    # Weights learned by "training"
    weights = [5, 3, -2]  # size matters most, age is negative
    bias = 10

    test_data = [
        ([80, 60, 20], "premium"),
        ([40, 30, 70], "budget"),
        ([90, 90, 10], "luxury"),
        ([50, 50, 50], "standard"),
        ([20, 80, 90], "location"),
    ]

    print(f"    Model: y = {weights[0]}*x1 + {weights[1]}*x2 + "
          f"({weights[2]})*x3 + {bias}")
    print(f"    Threshold: y >= 300 -> 'expensive', else 'affordable'")
    print(f"")
    print(f"    {'Features':<25} {'Score':>8} {'Class':>12}")
    print(f"    {'-'*25} {'-'*8} {'-'*12}")

    plaintext_predictions = []
    for features, label in test_data:
        score = sum(w * x for w, x in zip(weights, features)) + bias
        prediction = "expensive" if score >= 300 else "affordable"
        plaintext_predictions.append(score)
        feat_str = str(features)
        print(f"    {feat_str:<25} {score:>8} {prediction:>12}")

    # Part 2: Encrypted inference
    print(f"\n  Part 2: Encrypted Inference (server never sees features)")

    encrypted_predictions = []
    for features, label in test_data:
        # Client encrypts features
        enc_features = [he.encrypt(x) for x in features]

        # Server computes weighted sum homomorphically
        # Server knows weights (public model), not features
        weighted = [he.scalar_multiply(ef, w) for ef, w in zip(enc_features, weights)]
        enc_sum = weighted[0]
        for term in weighted[1:]:
            enc_sum = he.add_encrypted(enc_sum, term)

        # Add bias
        enc_bias = he.encrypt(bias)
        enc_result = he.add_encrypted(enc_sum, enc_bias)

        # Client decrypts
        dec_score = he.decrypt(enc_result)
        encrypted_predictions.append(dec_score)

    print(f"    {'Features':<25} {'Plain':>8} {'Encrypted':>10} {'Match':>8}")
    print(f"    {'-'*25} {'-'*8} {'-'*10} {'-'*8}")

    all_match = True
    for i, (features, label) in enumerate(test_data):
        feat_str = str(features)
        match = plaintext_predictions[i] == encrypted_predictions[i]
        if not match:
            all_match = False
        print(f"    {feat_str:<25} {plaintext_predictions[i]:>8} "
              f"{encrypted_predictions[i]:>10} {str(match):>8}")

    print(f"    All predictions match: {all_match}")

    # Part 3: Privacy analysis
    print(f"\n  Part 3: Privacy Analysis")
    print(f"    What the SERVER sees:")
    print(f"      - Model weights: {weights} (public)")
    print(f"      - Encrypted features: large random-looking numbers")
    print(f"      - Encrypted result: large random-looking number")
    print(f"    What the SERVER learns: NOTHING about client data")
    print(f"")
    print(f"    What the CLIENT sees:")
    print(f"      - Their own features (private)")
    print(f"      - Decrypted prediction score")
    print(f"    What the CLIENT learns about the model:")
    print(f"      - The prediction for their specific input")
    print(f"      - With many queries, could reverse-engineer weights")
    print(f"      - Mitigation: rate limiting, differential privacy on outputs")


def exercise_5():
    """Exercise 5: FHE Scheme Comparison (Conceptual + Coding)

    Recommend the best HE scheme for various applications.
    """
    print(f"  FHE Scheme Comparison and Recommendations")

    # Part 1: Scheme capabilities matrix
    print(f"\n  Part 1: Scheme Capabilities")
    print(f"    {'Scheme':<12} {'Add':>6} {'Mult':>6} {'Depth':>10} "
          f"{'Type':>12} {'Speed':>10}")
    print(f"    {'-'*12} {'-'*6} {'-'*6} {'-'*10} {'-'*12} {'-'*10}")

    schemes = [
        ("Paillier", "Yes", "No", "Unlimited", "Additive PHE", "Fast"),
        ("ElGamal", "No", "Yes", "Unlimited", "Mult PHE", "Fast"),
        ("BFV", "Yes", "Yes", "Bounded", "Leveled FHE", "Moderate"),
        ("CKKS", "Yes", "Yes", "Bounded", "Approx FHE", "Moderate"),
        ("TFHE", "Yes", "Yes", "Unlimited*", "Gate FHE", "Slow"),
    ]

    for name, add, mult, depth, stype, speed in schemes:
        print(f"    {name:<12} {add:>6} {mult:>6} {depth:>10} "
              f"{stype:>12} {speed:>10}")

    # Part 2: Application-specific recommendations
    print(f"\n  Part 2: Application Recommendations")

    applications = [
        {
            "name": "Average salary computation",
            "operations": "Sum N values, divide by N",
            "recommendation": "Paillier",
            "reason": (
                "Only needs addition (summing) and scalar multiplication "
                "(dividing by N = multiplying by N^(-1) mod p). "
                "Paillier is the simplest and fastest scheme that supports this. "
                "No need for FHE overhead."
            ),
        },
        {
            "name": "Neural network on medical images",
            "operations": "Matrix multiply, ReLU/polynomial activation, convolution",
            "recommendation": "CKKS",
            "reason": (
                "Neural networks use floating-point multiply-accumulate. "
                "CKKS natively supports approximate real arithmetic. "
                "SIMD batching encodes N/2 pixels per ciphertext. "
                "Polynomial approximation of ReLU works with CKKS rescaling."
            ),
        },
        {
            "name": "Private Boolean database search",
            "operations": "AND, OR, NOT gates on encrypted bits",
            "recommendation": "TFHE",
            "reason": (
                "TFHE evaluates Boolean circuits gate-by-gate with "
                "programmable bootstrapping after each gate. "
                "Each gate operation resets noise, enabling unlimited depth. "
                "Boolean operations map directly to TFHE's native operations."
            ),
        },
        {
            "name": "Encrypted statistics (mean, median, std dev)",
            "operations": "Sum, comparison (median), multiply (variance)",
            "recommendation": "BFV or CKKS",
            "reason": (
                "Mean: Paillier suffices. But median requires comparisons "
                "(sorting network), and std dev requires squaring. "
                "Both need FHE. BFV for exact integers, CKKS for approximate "
                "real numbers. CKKS is preferred if small errors are acceptable."
            ),
        },
        {
            "name": "Homomorphic AES evaluation",
            "operations": "S-box lookups, MixColumns, ShiftRows, XOR",
            "recommendation": "TFHE",
            "reason": (
                "AES is a Boolean circuit (operates on bits/bytes). "
                "TFHE's gate-by-gate bootstrapping handles AES's non-linear "
                "S-box without noise explosion. "
                "This is used in 'transciphering': encrypt with AES, then "
                "homomorphically decrypt into FHE ciphertext format."
            ),
        },
    ]

    for app in applications:
        print(f"\n    Application: {app['name']}")
        print(f"    Operations: {app['operations']}")
        print(f"    Recommended: {app['recommendation']}")
        print(f"    Reason: {app['reason']}")

    # Part 3: Performance simulation
    print(f"\n  Part 3: Performance Comparison (Simulated)")
    print(f"    Task: Compute mean of 1000 encrypted integers")
    print(f"")
    print(f"    {'Scheme':<12} {'Add (us)':>10} {'Total add':>12} "
          f"{'Total time':>12} {'Overhead':>10}")
    print(f"    {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    n_values = 1000
    # Estimated times per operation (microseconds)
    scheme_times = [
        ("Paillier", 50, 0),       # 50 us per add, no mult needed
        ("BFV", 100, 10000),       # 100 us per add, 10 ms per mult
        ("CKKS", 80, 8000),        # 80 us per add
        ("TFHE", 5000, 50000),     # 5 ms per gate (bootstrapped)
    ]

    for name, add_us, mult_us in scheme_times:
        total_add_us = add_us * (n_values - 1)
        total_us = total_add_us + mult_us  # one scalar mult for mean
        total_ms = total_us / 1000
        plaintext_ns = n_values  # ~1 ns per addition
        overhead = total_us * 1000 / plaintext_ns if plaintext_ns > 0 else 0
        print(f"    {name:<12} {add_us:>10} {total_add_us:>12,} "
              f"{total_ms:>10.0f} ms {overhead:>9.0f}x")

    print(f"\n    Paillier is ~50-100x faster than FHE for addition-only tasks.")
    print(f"    Always use the simplest scheme that supports your operations.")


if __name__ == "__main__":
    print("=== Exercise 1: RSA Homomorphism ===")
    exercise_1()

    print("\n=== Exercise 2: Paillier Applications ===")
    exercise_2()

    print("\n=== Exercise 3: Noise Budget Simulation ===")
    exercise_3()

    print("\n=== Exercise 4: Encrypted Linear Model ===")
    exercise_4()

    print("\n=== Exercise 5: FHE Scheme Comparison ===")
    exercise_5()

    print("\nAll exercises completed!")

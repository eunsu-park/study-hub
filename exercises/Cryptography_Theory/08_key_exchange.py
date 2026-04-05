"""
Exercises for Lesson 08: Key Exchange
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import hashlib
import hmac as hmac_lib
import math
import os
import random


def exercise_1():
    """Exercise 1: Small DH by Hand (Conceptual)

    Given p=23, g=5, a=6, b=15:
    1. Compute Alice's public value A
    2. Compute Bob's public value B
    3. Compute shared secret from both sides
    4. Eavesdropper analysis
    """
    p, g = 23, 5
    a, b = 6, 15

    # Step 1: Alice's public value
    # A = g^a mod p = 5^6 mod 23
    # 5^2 = 25 mod 23 = 2
    # 5^4 = 2^2 = 4
    # 5^6 = 5^4 * 5^2 = 4 * 2 = 8
    A = pow(g, a, p)
    print(f"  Step 1: Alice's public value")
    print(f"    A = g^a mod p = {g}^{a} mod {p}")
    print(f"    5^2 mod 23 = {pow(5,2,23)}")
    print(f"    5^4 mod 23 = {pow(5,4,23)}")
    print(f"    5^6 mod 23 = {pow(5,6,23)}")
    print(f"    A = {A}")

    # Step 2: Bob's public value
    # B = g^b mod p = 5^15 mod 23
    B = pow(g, b, p)
    print(f"\n  Step 2: Bob's public value")
    print(f"    B = g^b mod p = {g}^{b} mod {p}")
    print(f"    B = {B}")

    # Step 3: Shared secret from both sides
    # Alice: s = B^a mod p = {B}^{a} mod {p}
    s_alice = pow(B, a, p)
    # Bob:   s = A^b mod p = {A}^{b} mod {p}
    s_bob = pow(A, b, p)
    print(f"\n  Step 3: Shared secret computation")
    print(f"    Alice: s = B^a mod p = {B}^{a} mod {p} = {s_alice}")
    print(f"    Bob:   s = A^b mod p = {A}^{b} mod {p} = {s_bob}")
    print(f"    Match: {s_alice == s_bob}")
    print(f"    Mathematical proof: B^a = (g^b)^a = g^(ab) = (g^a)^b = A^b")

    # Step 4: Eavesdropper analysis
    print(f"\n  Step 4: Eavesdropper (Eve) analysis")
    print(f"    Eve sees: p={p}, g={g}, A={A}, B={B}")
    print(f"    Eve must solve: {g}^x ≡ {A} (mod {p}) to find a")
    print(f"    Or solve:       {g}^x ≡ {B} (mod {p}) to find b")
    print(f"\n    Brute force search:")
    attempts = 0
    for x in range(1, p):
        attempts += 1
        if pow(g, x, p) == A:
            print(f"      Found a = {x} after {attempts} attempts")
            break
    print(f"      s = B^a mod p = {pow(B, x, p)} (correct: {pow(B, x, p) == s_alice})")

    # Scaling analysis
    print(f"\n    Scaling:")
    print(f"    For p with k bits, brute force requires O(p) = O(2^k) operations")
    print(f"    Baby-step giant-step: O(sqrt(p)) = O(2^(k/2)) time + space")
    print(f"    For k=23: ~{p} brute force, ~{math.isqrt(p)} BSGS")
    print(f"    For k=2048: 2^2048 brute force (infeasible)")
    print(f"    Index calculus: sub-exponential, L(1/3) ~ 2^112 for 2048-bit p")


def exercise_2():
    """Exercise 2: DH Parameter Validation (Coding)

    Write validate_dh_params(p, g, A) that checks:
    - p is prime
    - g generates a large subgroup
    - A is in valid range [2, p-2]
    - A^q = 1 mod p for safe primes
    """
    def is_prime(n, k=20):
        """Miller-Rabin primality test."""
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0: return False
        s, d = 0, n - 1
        while d % 2 == 0: d //= 2; s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else: return False
        return True

    def validate_dh_params(p, g, A, verbose=True):
        """Validate Diffie-Hellman parameters.

        For safe primes p = 2q + 1:
        - p and q must be prime
        - g must generate the order-q subgroup (g^q ≡ 1 mod p)
        - A must be in [2, p-2] (reject trivial values)
        - A^q ≡ 1 mod p (A is in the correct subgroup)
        """
        errors = []

        # Check 1: p is prime
        p_prime = is_prime(p)
        if verbose:
            print(f"    Check 1: p is prime: {p_prime}")
        if not p_prime:
            errors.append("p is not prime")

        # Check 2: p is a safe prime (p = 2q + 1 where q is prime)
        q = (p - 1) // 2
        q_prime = is_prime(q)
        if verbose:
            print(f"    Check 2: q = (p-1)/2 = {q} is prime: {q_prime}")
        if not q_prime:
            errors.append("p is not a safe prime")

        # Check 3: g generates a large subgroup
        # For safe primes, the subgroup orders are: 1, 2, q, p-1
        # g should have order q (or p-1)
        g_order_q = pow(g, q, p) == 1
        g_trivial = g <= 1 or g >= p - 1
        if verbose:
            print(f"    Check 3: g^q mod p = {pow(g, q, p)} (should be 1): {g_order_q}")
            print(f"    Check 3: g is not trivial (1 or p-1): {not g_trivial}")
        if g_trivial:
            errors.append("g is a trivial value")

        # Check 4: A is in valid range
        a_valid_range = 2 <= A <= p - 2
        if verbose:
            print(f"    Check 4: A in [2, p-2]: {a_valid_range} (A={A})")
        if not a_valid_range:
            errors.append(f"A={A} is not in [2, p-2]")

        # Check 5: A is in the correct subgroup
        a_subgroup = pow(A, q, p) == 1
        if verbose:
            print(f"    Check 5: A^q mod p = {pow(A, q, p)} (should be 1): {a_subgroup}")
        if not a_subgroup:
            errors.append("A is not in the order-q subgroup")

        is_valid = len(errors) == 0
        if verbose:
            print(f"    Result: {'VALID' if is_valid else 'INVALID'}")
            for err in errors:
                print(f"      Error: {err}")

        return is_valid

    # Test with a known safe prime: p = 23, q = 11
    # 23 is prime, 11 is prime, so p = 2*11 + 1 is a safe prime
    print(f"  DH Parameter Validation")

    # Test 1: Valid parameters
    p = 23
    g = 2  # g=2, order of 2 in Z_23*: 2^11 mod 23 = 1, so order is 11
    a_private = random.randrange(2, p - 1)
    A = pow(g, a_private, p)

    print(f"\n  Test 1: Valid parameters (p=23, g=2)")
    valid = validate_dh_params(p, g, A)

    # Test 2: Invalid - A = 0
    print(f"\n  Test 2: Invalid A = 0")
    validate_dh_params(p, g, 0)

    # Test 3: Invalid - A = 1 (trivial)
    print(f"\n  Test 3: Invalid A = 1 (trivial)")
    validate_dh_params(p, g, 1)

    # Test 4: Invalid - A = p - 1
    print(f"\n  Test 4: Invalid A = p-1 = {p-1}")
    validate_dh_params(p, g, p - 1)

    # Test 5: Non-prime p
    print(f"\n  Test 5: Non-prime p = 22")
    validate_dh_params(22, 2, 5)

    # Test 6: Larger safe prime
    # p = 7919 is prime, q = 3959 is prime, so p is a safe prime
    p2 = 7919
    q2 = (p2 - 1) // 2
    print(f"\n  Test 6: Larger safe prime p={p2}, q={q2}")
    print(f"    p prime: {is_prime(p2)}, q prime: {is_prime(q2)}")
    g2 = 2
    a2 = random.randrange(2, p2 - 1)
    A2 = pow(g2, a2, p2)
    validate_dh_params(p2, g2, A2)


def exercise_3():
    """Exercise 3: Forward Secrecy Simulation (Coding)

    1. Static key mode: reuse DH keys
    2. Ephemeral key mode: fresh keys per message
    Show: compromise one session key reveals different amounts
    """
    # Use a small prime for demonstration
    p = 7919  # Safe prime
    g = 2

    print(f"  Forward Secrecy Simulation")
    print(f"    p = {p}, g = {g}")

    num_messages = 5

    # Mode 1: Static key (same key pair for all messages)
    print(f"\n  Mode 1: Static keys (reused for all {num_messages} messages)")
    static_a = random.randrange(2, p - 1)
    static_b = random.randrange(2, p - 1)
    static_A = pow(g, static_a, p)
    static_B = pow(g, static_b, p)
    static_shared = pow(static_B, static_a, p)

    static_keys = []
    for i in range(num_messages):
        # All messages use the same shared secret
        msg_key = hashlib.sha256(
            f"{static_shared}-msg{i}".encode()
        ).hexdigest()[:16]
        static_keys.append(msg_key)
        print(f"    Message {i}: key = {msg_key}")

    # Simulate compromise: attacker learns the static private key a
    print(f"\n    COMPROMISE: attacker learns Alice's private key a = {static_a}")
    print(f"    Attacker computes shared secret: B^a mod p = {pow(static_B, static_a, p)}")
    compromised_count = 0
    for i in range(num_messages):
        recovered_key = hashlib.sha256(
            f"{static_shared}-msg{i}".encode()
        ).hexdigest()[:16]
        match = recovered_key == static_keys[i]
        if match:
            compromised_count += 1
    print(f"    Messages compromised: {compromised_count}/{num_messages} (ALL!)")

    # Mode 2: Ephemeral keys (fresh per message)
    print(f"\n  Mode 2: Ephemeral keys (fresh for each of {num_messages} messages)")
    eph_keys = []
    eph_privates = []
    for i in range(num_messages):
        a_i = random.randrange(2, p - 1)
        b_i = random.randrange(2, p - 1)
        A_i = pow(g, a_i, p)
        B_i = pow(g, b_i, p)
        shared_i = pow(B_i, a_i, p)
        msg_key = hashlib.sha256(str(shared_i).encode()).hexdigest()[:16]
        eph_keys.append(msg_key)
        eph_privates.append((a_i, b_i, B_i))
        print(f"    Message {i}: key = {msg_key} (fresh DH)")

    # Simulate compromise: attacker learns one session's private key
    compromised_session = 2
    a_comp, _, B_comp = eph_privates[compromised_session]
    recovered_shared = pow(B_comp, a_comp, p)
    recovered_key = hashlib.sha256(str(recovered_shared).encode()).hexdigest()[:16]

    print(f"\n    COMPROMISE: attacker learns session {compromised_session}'s private key")
    compromised_count = 0
    for i in range(num_messages):
        if i == compromised_session:
            match = recovered_key == eph_keys[i]
            if match: compromised_count += 1
            print(f"    Message {i}: {'COMPROMISED' if match else 'safe'}")
        else:
            print(f"    Message {i}: SAFE (different ephemeral keys)")

    print(f"    Messages compromised: {compromised_count}/{num_messages} (only 1!)")

    # Summary
    print(f"\n  Summary:")
    print(f"    Static keys:    1 key compromise -> ALL messages exposed")
    print(f"    Ephemeral keys: 1 key compromise -> only 1 message exposed")
    print(f"    This is Forward Secrecy: past sessions remain secure")
    print(f"    even if long-term keys are compromised.")


def exercise_4():
    """Exercise 4: HKDF Test Vectors (Coding)

    Implement HKDF from scratch and verify against RFC 5869 test vectors.
    """
    def hkdf_extract(salt, ikm, hash_func=hashlib.sha256):
        """HKDF-Extract: PRK = HMAC-Hash(salt, IKM)."""
        if not salt:
            salt = b'\x00' * hash_func().digest_size
        return hmac_lib.new(salt, ikm, hash_func).digest()

    def hkdf_expand(prk, info, length, hash_func=hashlib.sha256):
        """HKDF-Expand: OKM = T(1) || T(2) || ... || T(N)."""
        hash_len = hash_func().digest_size
        n = math.ceil(length / hash_len)
        okm = b""
        t = b""
        for i in range(1, n + 1):
            t = hmac_lib.new(prk, t + info + bytes([i]), hash_func).digest()
            okm += t
        return okm[:length]

    def hkdf(ikm, length, salt=b"", info=b"", hash_func=hashlib.sha256):
        """Full HKDF: extract then expand."""
        prk = hkdf_extract(salt, ikm, hash_func)
        return hkdf_expand(prk, info, length, hash_func)

    print(f"  HKDF Implementation vs RFC 5869 Test Vectors")

    # Test Case 1 (RFC 5869, Appendix A.1): SHA-256
    ikm_1 = bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b")
    salt_1 = bytes.fromhex("000102030405060708090a0b0c")
    info_1 = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    L_1 = 42

    expected_prk_1 = bytes.fromhex(
        "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
    )
    expected_okm_1 = bytes.fromhex(
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865"
    )

    prk_1 = hkdf_extract(salt_1, ikm_1)
    okm_1 = hkdf_expand(prk_1, info_1, L_1)

    print(f"\n  Test Case 1 (SHA-256, basic):")
    print(f"    PRK match: {prk_1 == expected_prk_1}")
    print(f"    OKM match: {okm_1 == expected_okm_1}")
    if prk_1 != expected_prk_1:
        print(f"    PRK got:      {prk_1.hex()}")
        print(f"    PRK expected: {expected_prk_1.hex()}")
    if okm_1 != expected_okm_1:
        print(f"    OKM got:      {okm_1.hex()}")
        print(f"    OKM expected: {expected_okm_1.hex()}")

    # Test Case 2 (RFC 5869, Appendix A.2): SHA-256, longer inputs
    ikm_2 = bytes.fromhex(
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
        "404142434445464748494a4b4c4d4e4f"
    )
    salt_2 = bytes.fromhex(
        "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f"
        "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
        "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
    )
    info_2 = bytes.fromhex(
        "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdce"
        "cfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedee"
        "eff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"
    )
    L_2 = 82

    expected_prk_2 = bytes.fromhex(
        "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244"
    )
    expected_okm_2 = bytes.fromhex(
        "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f1d87"
    )

    prk_2 = hkdf_extract(salt_2, ikm_2)
    okm_2 = hkdf_expand(prk_2, info_2, L_2)

    print(f"\n  Test Case 2 (SHA-256, longer inputs):")
    print(f"    PRK match: {prk_2 == expected_prk_2}")
    print(f"    OKM match: {okm_2 == expected_okm_2}")

    # Test Case 3 (RFC 5869, Appendix A.3): SHA-256, no salt, no info
    ikm_3 = bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b")
    salt_3 = b""
    info_3 = b""
    L_3 = 42

    expected_prk_3 = bytes.fromhex(
        "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04"
    )
    expected_okm_3 = bytes.fromhex(
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8"
    )

    prk_3 = hkdf_extract(salt_3, ikm_3)
    okm_3 = hkdf_expand(prk_3, info_3, L_3)

    print(f"\n  Test Case 3 (SHA-256, no salt, no info):")
    print(f"    PRK match: {prk_3 == expected_prk_3}")
    print(f"    OKM match: {okm_3 == expected_okm_3}")

    all_pass = (prk_1 == expected_prk_1 and okm_1 == expected_okm_1 and
                prk_2 == expected_prk_2 and okm_2 == expected_okm_2 and
                prk_3 == expected_prk_3 and okm_3 == expected_okm_3)
    print(f"\n  All test vectors pass: {all_pass}")


def exercise_5():
    """Exercise 5: X3DH Walkthrough (Challenging)

    Simplified X3DH protocol:
    1. Bob publishes IK_B, SPK_B, OPK_B
    2. Alice performs four DH operations
    3. Alice encrypts initial message
    4. Bob decrypts
    """
    # Use a small prime for DH (educational; real X3DH uses Curve25519)
    p = 7919  # Safe prime
    g = 2

    def dh(private, public):
        """Compute DH shared secret."""
        return pow(public, private, p)

    def kdf(inputs, info=b""):
        """KDF over concatenated DH outputs."""
        combined = b"".join(s.to_bytes(16, 'big') for s in inputs)
        return hashlib.sha256(combined + info).digest()

    print(f"  X3DH Protocol Walkthrough")
    print(f"    p = {p}, g = {g}")

    # Step 1: Bob publishes keys to the server
    print(f"\n  Step 1: Bob publishes keys")

    # Identity key (long-term)
    ik_b_priv = random.randrange(2, p - 1)
    IK_B = pow(g, ik_b_priv, p)

    # Signed pre-key (medium-term, rotated periodically)
    spk_b_priv = random.randrange(2, p - 1)
    SPK_B = pow(g, spk_b_priv, p)
    # In practice, SPK_B is signed by IK_B (proving Bob created it)
    spk_signature = hashlib.sha256(
        f"SPK:{SPK_B}:signed_by:{IK_B}".encode()
    ).hexdigest()[:16]

    # One-time pre-key (used once, then deleted)
    opk_b_priv = random.randrange(2, p - 1)
    OPK_B = pow(g, opk_b_priv, p)

    print(f"    IK_B  = {IK_B} (identity, long-term)")
    print(f"    SPK_B = {SPK_B} (signed pre-key, medium-term)")
    print(f"    SPK signature: {spk_signature}")
    print(f"    OPK_B = {OPK_B} (one-time, consumed after use)")

    # Step 2: Alice performs X3DH
    print(f"\n  Step 2: Alice initiates X3DH")

    # Alice's identity key
    ik_a_priv = random.randrange(2, p - 1)
    IK_A = pow(g, ik_a_priv, p)

    # Alice's ephemeral key (fresh for this session)
    ek_a_priv = random.randrange(2, p - 1)
    EK_A = pow(g, ek_a_priv, p)

    print(f"    IK_A  = {IK_A} (Alice's identity key)")
    print(f"    EK_A  = {EK_A} (Alice's ephemeral key)")

    # Four DH operations
    # DH1: IK_A with SPK_B (mutual authentication)
    dh1 = dh(ik_a_priv, SPK_B)
    # DH2: EK_A with IK_B (forward secrecy for Alice's identity)
    dh2 = dh(ek_a_priv, IK_B)
    # DH3: EK_A with SPK_B (forward secrecy)
    dh3 = dh(ek_a_priv, SPK_B)
    # DH4: EK_A with OPK_B (replay protection)
    dh4 = dh(ek_a_priv, OPK_B)

    print(f"\n    DH1 = DH(IK_A, SPK_B) = {dh1}  (mutual authentication)")
    print(f"    DH2 = DH(EK_A, IK_B)  = {dh2}  (forward secrecy if IK_A compromised)")
    print(f"    DH3 = DH(EK_A, SPK_B) = {dh3}  (forward secrecy)")
    print(f"    DH4 = DH(EK_A, OPK_B) = {dh4}  (replay protection)")

    # Derive shared key
    SK_alice = kdf([dh1, dh2, dh3, dh4], info=b"X3DH")
    print(f"\n    SK (Alice) = KDF(DH1||DH2||DH3||DH4) = {SK_alice.hex()[:32]}...")

    # Step 3: Alice encrypts initial message
    print(f"\n  Step 3: Alice sends initial message")
    initial_message = b"Hello Bob, this is Alice!"

    # Simple XOR encryption with derived key (in practice, use AES-GCM)
    encrypted = bytes(m ^ k for m, k in zip(initial_message, SK_alice * 2))
    print(f"    Plaintext:  {initial_message.decode()}")
    print(f"    Encrypted:  {encrypted.hex()[:32]}...")
    print(f"    Alice sends: (IK_A, EK_A, encrypted message)")

    # Step 4: Bob receives and derives the same key
    print(f"\n  Step 4: Bob receives and decrypts")

    # Bob recomputes the four DH operations
    bob_dh1 = dh(spk_b_priv, IK_A)   # SPK_B_priv with IK_A
    bob_dh2 = dh(ik_b_priv, EK_A)    # IK_B_priv with EK_A
    bob_dh3 = dh(spk_b_priv, EK_A)   # SPK_B_priv with EK_A
    bob_dh4 = dh(opk_b_priv, EK_A)   # OPK_B_priv with EK_A

    SK_bob = kdf([bob_dh1, bob_dh2, bob_dh3, bob_dh4], info=b"X3DH")
    print(f"    SK (Bob)   = KDF(DH1||DH2||DH3||DH4) = {SK_bob.hex()[:32]}...")
    print(f"    Keys match: {SK_alice == SK_bob}")

    # Bob decrypts
    decrypted = bytes(c ^ k for c, k in zip(encrypted, SK_bob * 2))
    print(f"    Decrypted:  {decrypted.decode()}")
    print(f"    Correct: {decrypted == initial_message}")

    # Security properties
    print(f"\n  Security Properties:")
    print(f"    - Mutual authentication: DH1 uses both identity keys")
    print(f"    - Forward secrecy: EK_A is ephemeral (deleted after use)")
    print(f"    - Replay protection: OPK_B is consumed (server deletes it)")
    print(f"    - Asynchronous: Bob doesn't need to be online")
    print(f"    - If IK_A compromised later: DH2, DH3, DH4 still protect")
    print(f"    - If IK_B compromised later: DH1, DH3, DH4 still protect")


if __name__ == "__main__":
    print("=== Exercise 1: Small DH by Hand ===")
    exercise_1()

    print("\n=== Exercise 2: DH Parameter Validation ===")
    exercise_2()

    print("\n=== Exercise 3: Forward Secrecy Simulation ===")
    exercise_3()

    print("\n=== Exercise 4: HKDF Test Vectors ===")
    exercise_4()

    print("\n=== Exercise 5: X3DH Walkthrough ===")
    exercise_5()

    print("\nAll exercises completed!")

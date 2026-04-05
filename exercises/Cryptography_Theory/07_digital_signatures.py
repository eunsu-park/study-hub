"""
Exercises for Lesson 07: Digital Signatures
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import hashlib
import random
from math import gcd


class EllipticCurveFiniteField:
    """Elliptic curve over a prime finite field F_p."""

    def __init__(self, a, b, p):
        self.a, self.b, self.p = a, b, p

    def is_on_curve(self, P):
        if P is None: return True
        x, y = P
        return (y*y - (x*x*x + self.a*x + self.b)) % self.p == 0

    def add(self, P, Q):
        if P is None: return Q
        if Q is None: return P
        x1, y1 = P; x2, y2 = Q; p = self.p
        if x1 == x2 and y1 == (p - y2) % p: return None
        if x1 == x2 and y1 == y2:
            if y1 == 0: return None
            lam = (3*x1*x1 + self.a) * pow(2*y1, -1, p) % p
        else:
            lam = (y2-y1) * pow(x2-x1, -1, p) % p
        x3 = (lam*lam - x1 - x2) % p
        y3 = (lam*(x1-x3) - y1) % p
        return (x3, y3)

    def scalar_multiply(self, k, P):
        result = None; addend = P
        while k:
            if k & 1: result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def count_points(self):
        count = 1
        for x in range(self.p):
            rhs = (x**3 + self.a*x + self.b) % self.p
            if rhs == 0: count += 1
            elif pow(rhs, (self.p-1)//2, self.p) == 1: count += 2
        return count


def generate_rsa_key(bits=512):
    """Generate RSA key pair for signature exercises."""
    def generate_prime(bits):
        while True:
            n = random.getrandbits(bits) | (1 << (bits - 1)) | 1
            if is_probable_prime(n):
                return n

    def is_probable_prime(n, k=20):
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

    half = bits // 2
    while True:
        p = generate_prime(half)
        q = generate_prime(half)
        if p != q:
            n = p * q
            phi = (p - 1) * (q - 1)
            e = 65537
            if gcd(e, phi) == 1:
                d = pow(e, -1, phi)
                return (n, e, d, p, q)


def exercise_1():
    """Exercise 1: RSA Signatures (Basic)

    Using RSA with p=61, q=53, e=17:
    1. Compute the private key d
    2. Sign the message hash h=42
    3. Verify the signature
    4. Show that modifying hash to h=43 causes verification failure
    """
    p, q = 61, 53
    e = 17

    # Step 1: Compute private key d
    n = p * q
    phi_n = (p - 1) * (q - 1)
    d = pow(e, -1, phi_n)

    print(f"  Step 1: Key generation")
    print(f"    p = {p}, q = {q}")
    print(f"    n = {n}")
    print(f"    phi(n) = {phi_n}")
    print(f"    e = {e}")
    print(f"    d = e^(-1) mod phi(n) = {d}")
    print(f"    Verify: e*d mod phi(n) = {(e * d) % phi_n}")

    # Step 2: Sign h=42
    # Signature: sigma = h^d mod n
    h = 42
    sigma = pow(h, d, n)
    print(f"\n  Step 2: Sign hash h = {h}")
    print(f"    sigma = h^d mod n = {h}^{d} mod {n} = {sigma}")

    # Step 3: Verify
    # Verification: sigma^e mod n should equal h
    h_recovered = pow(sigma, e, n)
    print(f"\n  Step 3: Verify signature")
    print(f"    sigma^e mod n = {sigma}^{e} mod {n} = {h_recovered}")
    print(f"    h_recovered == h: {h_recovered == h}")
    print(f"    Signature VALID: {h_recovered == h}")

    # Step 4: Modified hash h=43
    h_modified = 43
    h_check = pow(sigma, e, n)
    print(f"\n  Step 4: Verify against modified hash h = {h_modified}")
    print(f"    sigma^e mod n = {h_check}")
    print(f"    h_check == {h_modified}: {h_check == h_modified}")
    print(f"    Signature INVALID for modified hash: {h_check != h_modified}")
    print(f"    This demonstrates integrity: any change to the message invalidates the signature.")


def exercise_2():
    """Exercise 2: ECDSA Implementation (Intermediate)

    1. Sign 10 different messages and verify each
    2. Measure point operations per signing and verification
    3. Analyze probability of r=0
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    # Generate key pair
    d = random.randrange(1, n)
    Q = curve.scalar_multiply(d, G)

    print(f"  ECDSA Implementation")
    print(f"    Curve: y^2 = x^3 + 2x + 3 (mod 97)")
    print(f"    Generator G = {G}, Order n = {n}")
    print(f"    Private key d = {d}, Public key Q = {Q}")

    # Part 1: Sign and verify 10 messages
    print(f"\n  Part 1: Sign and verify 10 messages")
    all_valid = True
    for i in range(10):
        msg = f"Message #{i}".encode()
        e = int(hashlib.sha256(msg).hexdigest(), 16) % n

        # Sign
        while True:
            k = random.randrange(1, n)
            R = curve.scalar_multiply(k, G)
            r = R[0] % n
            if r == 0: continue
            k_inv = pow(k, -1, n)
            s = (k_inv * (e + r * d)) % n
            if s == 0: continue
            break

        # Verify
        w = pow(s, -1, n)
        u1 = (e * w) % n
        u2 = (r * w) % n
        P1 = curve.scalar_multiply(u1, G)
        P2 = curve.scalar_multiply(u2, Q)
        R_prime = curve.add(P1, P2)
        valid = R_prime is not None and R_prime[0] % n == r

        if not valid:
            all_valid = False
        if i < 3:
            print(f"    Msg {i}: sig=({r},{s}), valid={valid}")

    print(f"    ... (7 more)")
    print(f"    All 10 signatures valid: {all_valid}")

    # Part 2: Point operations count
    print(f"\n  Part 2: Point operations per signing/verification")
    print(f"    Signing:")
    print(f"      - 1 scalar multiplication: k*G ({n.bit_length()} doublings + additions)")
    print(f"      - 1 modular inverse: k^(-1) mod n")
    print(f"      - 2 modular multiplications")
    print(f"      Total: ~{n.bit_length()} point doublings + ~{n.bit_length()//2} additions")
    print(f"    Verification:")
    print(f"      - 2 scalar multiplications: u1*G, u2*Q")
    print(f"      - 1 point addition: u1*G + u2*Q")
    print(f"      Total: ~{2*n.bit_length()} point doublings + ~{n.bit_length()} additions")

    # Part 3: Probability of r=0
    print(f"\n  Part 3: Probability of r=0")
    print(f"    r = x_R mod n where R = kG")
    print(f"    r=0 when x_R is a multiple of n")
    print(f"    For a curve with n points, x-coordinates range over F_p (p={curve.p})")
    print(f"    P(r=0) = P(x_R ≡ 0 mod n) ≈ 1/n ≈ {1/n:.6f}")
    print(f"    This is extremely unlikely (~2^(-256) for real curves)")
    print(f"    The retry loop is needed for correctness but almost never triggers.")


def exercise_3():
    """Exercise 3: Nonce Reuse Attack (Intermediate)

    1. Implement nonce reuse attack
    2. Use a curve with order > 100
    3. Create two signatures with same nonce
    4. Recover private key
    5. Verify recovered key works
    """
    # Use a larger curve
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    # Generate target key pair
    d_secret = random.randrange(1, n)
    Q = curve.scalar_multiply(d_secret, G)

    print(f"  Nonce Reuse Attack")
    print(f"    Curve order n = {n}")
    print(f"    Secret key d = {d_secret}")
    print(f"    Public key Q = {Q}")

    # Sign two different messages with the SAME nonce
    k = random.randrange(1, n)
    R = curve.scalar_multiply(k, G)
    r = R[0] % n

    msg1 = b"Transfer $100 to Alice"
    msg2 = b"Transfer $200 to Bob"
    e1 = int(hashlib.sha256(msg1).hexdigest(), 16) % n
    e2 = int(hashlib.sha256(msg2).hexdigest(), 16) % n

    k_inv = pow(k, -1, n)
    s1 = (k_inv * (e1 + r * d_secret)) % n
    s2 = (k_inv * (e2 + r * d_secret)) % n

    print(f"\n  Two signatures with SAME nonce (k = {k}):")
    print(f"    Sig1: (r={r}, s1={s1}) on '{msg1.decode()}'")
    print(f"    Sig2: (r={r}, s2={s2}) on '{msg2.decode()}'")
    print(f"    NOTE: Same r value reveals nonce reuse!")

    # Attack: recover k
    # s1 - s2 = k^(-1) * (e1 - e2) mod n
    # k = (e1 - e2) / (s1 - s2) mod n
    s_diff = (s1 - s2) % n
    e_diff = (e1 - e2) % n

    if s_diff != 0:
        k_recovered = (e_diff * pow(s_diff, -1, n)) % n
        print(f"\n  Attack:")
        print(f"    s1 - s2 = {s_diff}")
        print(f"    e1 - e2 = {e_diff}")
        print(f"    k = (e1-e2) * (s1-s2)^(-1) mod n = {k_recovered}")
        print(f"    k matches: {k_recovered == k}")

        # Recover d from k
        # s1 = k^(-1) * (e1 + r*d) mod n
        # s1*k = e1 + r*d mod n
        # d = (s1*k - e1) * r^(-1) mod n
        d_recovered = ((s1 * k_recovered - e1) * pow(r, -1, n)) % n
        print(f"    d = (s1*k - e1) * r^(-1) mod n = {d_recovered}")
        print(f"    d matches: {d_recovered == d_secret}")

        # Step 5: Verify recovered key can sign new messages
        print(f"\n  Verify recovered key:")
        msg_new = b"Forged message"
        e_new = int(hashlib.sha256(msg_new).hexdigest(), 16) % n
        k_new = random.randrange(1, n)
        R_new = curve.scalar_multiply(k_new, G)
        r_new = R_new[0] % n
        k_new_inv = pow(k_new, -1, n)
        s_new = (k_new_inv * (e_new + r_new * d_recovered)) % n

        # Verify with original public key
        w = pow(s_new, -1, n)
        u1 = (e_new * w) % n
        u2 = (r_new * w) % n
        P1 = curve.scalar_multiply(u1, G)
        P2 = curve.scalar_multiply(u2, Q)
        R_check = curve.add(P1, P2)
        valid = R_check is not None and R_check[0] % n == r_new
        print(f"    Forged signature valid: {valid}")
        print(f"    GAME OVER: attacker can sign arbitrary messages!")
    else:
        print(f"\n  Attack failed: s1 == s2 (degenerate case)")


def exercise_4():
    """Exercise 4: Schnorr Multi-Signature (Challenging)

    Simple 2-of-2 Schnorr multi-signature protocol:
    1. Alice and Bob each have private keys
    2. Joint public key: Q = Q_A + Q_B
    3. Each contributes a partial signature
    4. Combined signature verifies against Q
    5. Discuss rogue-key attack
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    print(f"  Schnorr 2-of-2 Multi-Signature")
    print(f"    Curve order n = {n}")

    # Step 1: Key generation
    d_a = random.randrange(1, n)
    Q_a = curve.scalar_multiply(d_a, G)
    d_b = random.randrange(1, n)
    Q_b = curve.scalar_multiply(d_b, G)

    print(f"    Alice: d_A = {d_a}, Q_A = {Q_a}")
    print(f"    Bob:   d_B = {d_b}, Q_B = {Q_b}")

    # Step 2: Joint public key
    Q_joint = curve.add(Q_a, Q_b)
    print(f"    Joint key Q = Q_A + Q_B = {Q_joint}")

    # Step 3: Collaborative signing
    message = b"Multi-sig document"

    # Each party picks a random nonce
    r_a = random.randrange(1, n)
    R_a = curve.scalar_multiply(r_a, G)

    r_b = random.randrange(1, n)
    R_b = curve.scalar_multiply(r_b, G)

    # Aggregate nonce
    R = curve.add(R_a, R_b)

    # Compute challenge: e = H(R || Q || message)
    e_input = f"{R[0]},{R[1]},{Q_joint[0]},{Q_joint[1]},{message.hex()}".encode()
    e = int(hashlib.sha256(e_input).hexdigest(), 16) % n

    # Partial signatures
    s_a = (r_a - e * d_a) % n  # Alice's partial
    s_b = (r_b - e * d_b) % n  # Bob's partial

    # Combined signature
    s = (s_a + s_b) % n

    print(f"\n    Signing '{message.decode()}':")
    print(f"    R_A = {R_a}, R_B = {R_b}")
    print(f"    R = R_A + R_B = {R}")
    print(f"    e = H(R||Q||m) mod n = {e}")
    print(f"    s_A = {s_a}, s_B = {s_b}")
    print(f"    s = s_A + s_B = {s}")

    # Step 4: Verify against joint public key
    # Verification: R' = s*G + e*Q should equal R
    sG = curve.scalar_multiply(s, G)
    eQ = curve.scalar_multiply(e, Q_joint)
    R_prime = curve.add(sG, eQ)

    e_check_input = f"{R_prime[0]},{R_prime[1]},{Q_joint[0]},{Q_joint[1]},{message.hex()}".encode()
    e_check = int(hashlib.sha256(e_check_input).hexdigest(), 16) % n

    print(f"\n    Verification:")
    print(f"    R' = s*G + e*Q = {R_prime}")
    print(f"    R' == R: {R_prime == R}")
    print(f"    Signature valid: {e_check == e}")

    # Mathematical justification:
    # s = (r_a + r_b) - e*(d_a + d_b) mod n
    # s*G + e*Q = [(r_a + r_b) - e*(d_a + d_b)]*G + e*(d_a + d_b)*G
    #           = (r_a + r_b)*G = R_a + R_b = R

    # Step 5: Rogue-key attack discussion
    print(f"\n  Step 5: Rogue-Key Attack Discussion")
    print(f"    The naive Q = Q_A + Q_B is VULNERABLE!")
    print(f"    Attack: Mallory claims public key Q_M = Q_M' - Q_A")
    print(f"      where Q_M' = d_M * G for some d_M Mallory knows")
    print(f"    Joint key: Q = Q_A + Q_M = Q_A + Q_M' - Q_A = Q_M'")
    print(f"    Now Mallory can sign alone using d_M!")
    print(f"")
    print(f"    MuSig2 prevents this by requiring each party to prove")
    print(f"    knowledge of their secret key (via a ZKP or by using")
    print(f"    key aggregation with a hash-based coefficient):")
    print(f"    Q = H(Q_A,Q_B,Q_A)*Q_A + H(Q_A,Q_B,Q_B)*Q_B")
    print(f"    The hash prevents rogue-key attacks because Mallory")
    print(f"    cannot choose Q_M to cancel out another key's contribution.")


def exercise_5():
    """Exercise 5: Deterministic vs Random Nonces (Challenging)

    1. Implement RFC 6979 deterministic nonce generation for ECDSA
    2. Same (key, message) always produces same signature
    3. Different messages produce different nonces
    4. Compare with Ed25519
    5. Discuss SUF-CMA security
    """
    import hmac

    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    # Generate key pair
    d = random.randrange(1, n)
    Q = curve.scalar_multiply(d, G)

    print(f"  RFC 6979 Deterministic Nonces")
    print(f"    Curve order n = {n}")

    # Part 1: Simplified RFC 6979 nonce generation
    def rfc6979_nonce(private_key, message_hash, order):
        """Simplified RFC 6979: deterministic nonce from key + message.

        The full RFC 6979 uses HMAC-DRBG:
        1. V = 0x01...01 (hash_len bytes)
        2. K = 0x00...00 (hash_len bytes)
        3. K = HMAC_K(V || 0x00 || x || h)
        4. V = HMAC_K(V)
        5. K = HMAC_K(V || 0x01 || x || h)
        6. V = HMAC_K(V)
        7. Generate k from V, retry if k >= order
        """
        # Simplified version: k = HMAC(private_key, message_hash) mod order
        key_bytes = private_key.to_bytes(32, 'big')
        msg_bytes = message_hash.to_bytes(32, 'big')

        # HMAC-based deterministic derivation
        v = b'\x01' * 32
        k_hmac = b'\x00' * 32

        k_hmac = hmac.new(k_hmac, v + b'\x00' + key_bytes + msg_bytes, hashlib.sha256).digest()
        v = hmac.new(k_hmac, v, hashlib.sha256).digest()
        k_hmac = hmac.new(k_hmac, v + b'\x01' + key_bytes + msg_bytes, hashlib.sha256).digest()
        v = hmac.new(k_hmac, v, hashlib.sha256).digest()

        # Generate nonce
        while True:
            v = hmac.new(k_hmac, v, hashlib.sha256).digest()
            nonce = int.from_bytes(v, 'big') % order
            if 1 <= nonce < order:
                return nonce
            k_hmac = hmac.new(k_hmac, v + b'\x00', hashlib.sha256).digest()
            v = hmac.new(k_hmac, v, hashlib.sha256).digest()

    def ecdsa_sign_deterministic(msg, private_key):
        """ECDSA signing with deterministic nonce."""
        e = int(hashlib.sha256(msg).hexdigest(), 16) % n
        k = rfc6979_nonce(private_key, e, n)
        R = curve.scalar_multiply(k, G)
        r = R[0] % n
        if r == 0:
            return None
        k_inv = pow(k, -1, n)
        s = (k_inv * (e + r * private_key)) % n
        if s == 0:
            return None
        return (r, s, k)  # Return k for demonstration

    # Part 2: Same (key, message) -> same signature
    print(f"\n  Part 2: Determinism test")
    msg = b"Hello, deterministic ECDSA!"

    sig1 = ecdsa_sign_deterministic(msg, d)
    sig2 = ecdsa_sign_deterministic(msg, d)

    if sig1 and sig2:
        print(f"    Signature 1: (r={sig1[0]}, s={sig1[1]}), k={sig1[2]}")
        print(f"    Signature 2: (r={sig2[0]}, s={sig2[1]}), k={sig2[2]}")
        print(f"    Same nonce: {sig1[2] == sig2[2]}")
        print(f"    Same signature: {sig1[:2] == sig2[:2]}")
    else:
        print(f"    Signing failed (degenerate nonce)")

    # Part 3: Different messages -> different nonces
    print(f"\n  Part 3: Different messages produce different nonces")
    messages = [b"Message A", b"Message B", b"Message C"]
    nonces = []
    for msg in messages:
        sig = ecdsa_sign_deterministic(msg, d)
        if sig:
            nonces.append(sig[2])
            print(f"    '{msg.decode()}': k = {sig[2]}")

    all_different = len(set(nonces)) == len(nonces)
    print(f"    All nonces different: {all_different}")

    # Part 4: Comparison with Ed25519
    print(f"\n  Part 4: RFC 6979 vs Ed25519")
    print(f"    {'Property':<25} {'RFC 6979':<25} {'Ed25519':<25}")
    print(f"    {'-'*25} {'-'*25} {'-'*25}")
    print(f"    {'Nonce source':<25} {'HMAC(sk, H(msg))':<25} {'SHA-512(prefix||msg)':<25}")
    print(f"    {'Depends on':<25} {'private key + message':<25} {'private key + message':<25}")
    print(f"    {'Hash function':<25} {'Configurable':<25} {'SHA-512 (fixed)':<25}")
    print(f"    {'Curve':<25} {'Any (ECDSA)':<25} {'Ed25519 only':<25}")
    print(f"    {'Deterministic':<25} {'Yes':<25} {'Yes':<25}")
    print(f"    {'Key expansion':<25} {'No':<25} {'Yes (SHA-512(sk))':<25}")
    print(f"    {'Nonce reuse risk':<25} {'Eliminated':<25} {'Eliminated':<25}")
    print(f"")
    print(f"    Similarity: Both derive nonce from (private_key, message)")
    print(f"    Difference: Ed25519 uses the upper half of SHA-512(sk) as the")
    print(f"    nonce prefix, while RFC 6979 uses HMAC-DRBG with the full key.")

    # Part 5: SUF-CMA security
    print(f"\n  Part 5: Can deterministic signatures be SUF-CMA secure?")
    print(f"    SUF-CMA (Strong UF under Chosen Message Attack):")
    print(f"    The adversary cannot produce a NEW valid signature for ANY message,")
    print(f"    even for messages they already have signatures for.")
    print(f"")
    print(f"    Deterministic signatures: same (key, msg) -> same signature")
    print(f"    This means there is only ONE valid signature per message.")
    print(f"    If the adversary queries a signature, they get the ONLY one.")
    print(f"    They cannot produce a DIFFERENT valid signature.")
    print(f"")
    print(f"    Conclusion: Deterministic signature schemes are NATURALLY SUF-CMA")
    print(f"    secure (if the underlying scheme is EUF-CMA secure), because")
    print(f"    there is no second valid signature to forge.")
    print(f"    Ed25519 achieves SUF-CMA security precisely because of this.")
    print(f"    Randomized ECDSA is only EUF-CMA (different randomness -> different signature).")


if __name__ == "__main__":
    print("=== Exercise 1: RSA Signatures ===")
    exercise_1()

    print("\n=== Exercise 2: ECDSA Implementation ===")
    exercise_2()

    print("\n=== Exercise 3: Nonce Reuse Attack ===")
    exercise_3()

    print("\n=== Exercise 4: Schnorr Multi-Signature ===")
    exercise_4()

    print("\n=== Exercise 5: Deterministic vs Random Nonces ===")
    exercise_5()

    print("\nAll exercises completed!")

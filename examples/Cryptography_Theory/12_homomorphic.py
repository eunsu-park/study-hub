"""
Homomorphic Encryption: The Paillier Cryptosystem
==================================================
Demonstrates additive homomorphic encryption — performing addition
on encrypted values without ever decrypting them. Includes a
practical voting example.
"""

from __future__ import annotations
import math
import random


# ---------------------------------------------------------------------------
# Number Theory Helpers
# ---------------------------------------------------------------------------

def mod_pow(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation."""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        exp >>= 1
        base = base * base % mod
    return result


def mod_inverse(a: int, m: int) -> int:
    """Modular inverse via extended GCD."""
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No inverse for {a} mod {m}")
    return x % m


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def _is_prime(n: int) -> bool:
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0: return False
    return True


def _generate_prime(bits: int) -> int:
    while True:
        c = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_prime(c):
            return c


# ---------------------------------------------------------------------------
# Paillier Cryptosystem
# ---------------------------------------------------------------------------

# Why: Paillier encryption is additively homomorphic, meaning:
#   Enc(a) * Enc(b) mod n^2 = Enc(a + b mod n)
#   Enc(a)^k mod n^2 = Enc(a * k mod n)
# This allows computation on encrypted data — the foundation of
# privacy-preserving computation, secure voting, and confidential
# machine learning (federated learning with encrypted gradients).

class PaillierPublicKey:
    """Paillier public key (n, g)."""

    def __init__(self, n: int, g: int) -> None:
        self.n = n
        self.n_sq = n * n  # n^2 is the ciphertext space
        self.g = g

    # Why: Encryption uses randomness r to achieve semantic security
    # (same plaintext encrypts to different ciphertexts each time).
    # c = g^m * r^n mod n^2, where r is random in Z_n*.
    def encrypt(self, plaintext: int) -> int:
        """Encrypt a plaintext message m in [0, n-1].

        c = g^m * r^n mod n^2
        """
        if plaintext < 0 or plaintext >= self.n:
            raise ValueError(f"Plaintext must be in [0, {self.n - 1}]")

        # Why: r must be coprime to n and random. This randomization
        # is what makes Paillier semantically secure — identical
        # plaintexts produce different ciphertexts.
        while True:
            r = random.randrange(1, self.n)
            if math.gcd(r, self.n) == 1:
                break

        # c = g^m * r^n mod n^2
        gm = mod_pow(self.g, plaintext, self.n_sq)
        rn = mod_pow(r, self.n, self.n_sq)
        return (gm * rn) % self.n_sq


class PaillierPrivateKey:
    """Paillier private key (lambda, mu)."""

    def __init__(self, lam: int, mu: int, n: int) -> None:
        self.lam = lam  # lambda = lcm(p-1, q-1)
        self.mu = mu    # mu = L(g^lambda mod n^2)^{-1} mod n
        self.n = n
        self.n_sq = n * n

    def decrypt(self, ciphertext: int) -> int:
        """Decrypt a ciphertext.

        m = L(c^lambda mod n^2) * mu mod n
        where L(x) = (x - 1) / n
        """
        # Why: The L function extracts the "exponent" from the ciphertext.
        # L(c^lambda mod n^2) = L(g^{m*lambda} mod n^2) because r^{n*lambda}
        # = 1 mod n^2 (by Carmichael's theorem). The mu factor normalizes.
        x = mod_pow(ciphertext, self.lam, self.n_sq)
        l_x = (x - 1) // self.n  # L function
        return (l_x * self.mu) % self.n


def paillier_keygen(
    bits: int = 64,
) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Generate a Paillier key pair.

    Args:
        bits: bit length of each prime (n will be ~2*bits)
    """
    # Why: Like RSA, Paillier uses n = p*q where p, q are large primes.
    # The security relies on the Decisional Composite Residuosity
    # Assumption (DCRA): it's hard to distinguish n-th residues mod n^2
    # from random elements.
    p = _generate_prime(bits)
    q = _generate_prime(bits)
    while p == q:
        q = _generate_prime(bits)

    n = p * q
    n_sq = n * n

    # lambda = lcm(p-1, q-1)
    lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)

    # g = n + 1 is the simplest valid generator
    # Why: g = n+1 simplifies computation because L(g^lambda mod n^2)
    # has a nice closed form. Any g in Z_{n^2}* with order divisible
    # by n would work, but g = n+1 is standard.
    g = n + 1

    # mu = L(g^lambda mod n^2)^{-1} mod n
    x = mod_pow(g, lam, n_sq)
    l_x = (x - 1) // n
    mu = mod_inverse(l_x, n)

    pub = PaillierPublicKey(n, g)
    priv = PaillierPrivateKey(lam, mu, n)
    return pub, priv


# ---------------------------------------------------------------------------
# Homomorphic Operations
# ---------------------------------------------------------------------------

# Why: These operations work on CIPHERTEXTS — no decryption needed!
# The server performing these operations never sees the plaintext.

def homomorphic_add(
    c1: int, c2: int, pub: PaillierPublicKey
) -> int:
    """Compute Enc(m1 + m2) from Enc(m1) and Enc(m2).

    c1 * c2 mod n^2 = Enc(m1 + m2 mod n)
    """
    # Why: This follows from the math: c1 = g^{m1}*r1^n, c2 = g^{m2}*r2^n.
    # c1*c2 = g^{m1+m2} * (r1*r2)^n mod n^2 = Enc(m1+m2) with randomness r1*r2.
    return (c1 * c2) % pub.n_sq


def homomorphic_scalar_mult(
    c: int, k: int, pub: PaillierPublicKey
) -> int:
    """Compute Enc(m * k) from Enc(m) and plaintext scalar k.

    c^k mod n^2 = Enc(m * k mod n)
    """
    # Why: c^k = (g^m * r^n)^k = g^{mk} * r^{nk} = Enc(mk).
    # Note: k is a PLAINTEXT scalar, not encrypted. The server needs
    # to know k but doesn't learn m.
    return mod_pow(c, k, pub.n_sq)


# ---------------------------------------------------------------------------
# Secure Voting Example
# ---------------------------------------------------------------------------

# Why: Paillier is ideal for voting because:
# 1) Each voter encrypts their vote (0 or 1)
# 2) The election authority multiplies all ciphertexts (homomorphic sum)
# 3) Only the final sum is decrypted — individual votes stay hidden
# 4) No one (not even the authority) can link votes to voters

def secure_voting_demo(
    pub: PaillierPublicKey,
    priv: PaillierPrivateKey,
    votes: list[int],
) -> int:
    """Simulate a secure election using Paillier homomorphic encryption."""
    print(f"\n  Voters: {len(votes)}")
    print(f"  Votes (secret): {votes}")

    # Each voter encrypts their vote
    encrypted_votes = [pub.encrypt(v) for v in votes]
    print(f"  Encrypted votes: [{', '.join(str(c)[:12] + '...' for c in encrypted_votes)}]")

    # Tally: homomorphic sum of all encrypted votes
    tally = encrypted_votes[0]
    for c in encrypted_votes[1:]:
        tally = homomorphic_add(tally, c, pub)

    # Decrypt only the final tally
    result = priv.decrypt(tally)
    print(f"  Encrypted tally: {str(tally)[:20]}...")
    print(f"  Decrypted tally: {result}")
    print(f"  Expected tally:  {sum(votes)}")
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Homomorphic Encryption: Paillier Cryptosystem")
    print("=" * 65)

    # Key generation
    print("\n[Key Generation]")
    pub, priv = paillier_keygen(bits=32)
    print(f"  n = {pub.n} ({pub.n.bit_length()} bits)")
    print(f"  g = n + 1 = {pub.g}")

    # Basic encryption/decryption
    print("\n[Basic Encrypt/Decrypt]")
    for m in [0, 1, 42, 100, 9999]:
        c = pub.encrypt(m)
        d = priv.decrypt(c)
        print(f"  Enc({m:>5}) = {str(c)[:20]}...  Dec = {d}")

    # Semantic security: same plaintext, different ciphertext
    print("\n[Semantic Security]")
    c1 = pub.encrypt(42)
    c2 = pub.encrypt(42)
    print(f"  Enc(42) attempt 1: {str(c1)[:30]}...")
    print(f"  Enc(42) attempt 2: {str(c2)[:30]}...")
    print(f"  Same ciphertext?   {c1 == c2}")
    print(f"  Both decrypt to 42: {priv.decrypt(c1) == 42 and priv.decrypt(c2) == 42}")

    # --- Additive Homomorphism ---
    print("\n[Additive Homomorphism: Enc(a) * Enc(b) = Enc(a+b)]")
    a, b = 17, 25
    enc_a = pub.encrypt(a)
    enc_b = pub.encrypt(b)
    enc_sum = homomorphic_add(enc_a, enc_b, pub)
    dec_sum = priv.decrypt(enc_sum)
    print(f"  a = {a}, b = {b}")
    print(f"  Enc(a) * Enc(b) mod n^2 -> decrypt = {dec_sum}")
    print(f"  a + b = {a + b}")
    print(f"  Match: {dec_sum == a + b}")

    # Multiple additions
    print("\n  Adding 5 values homomorphically:")
    values = [10, 20, 30, 40, 50]
    encrypted = [pub.encrypt(v) for v in values]
    running = encrypted[0]
    for c in encrypted[1:]:
        running = homomorphic_add(running, c, pub)
    result = priv.decrypt(running)
    print(f"  Values: {values}")
    print(f"  Homomorphic sum: {result}")
    print(f"  Expected:        {sum(values)}")

    # --- Scalar Multiplication ---
    print("\n[Scalar Multiplication: Enc(a)^k = Enc(a*k)]")
    a, k = 7, 6
    enc_a = pub.encrypt(a)
    enc_product = homomorphic_scalar_mult(enc_a, k, pub)
    dec_product = priv.decrypt(enc_product)
    print(f"  a = {a}, k = {k}")
    print(f"  Enc(a)^k mod n^2 -> decrypt = {dec_product}")
    print(f"  a * k = {a * k}")
    print(f"  Match: {dec_product == a * k}")

    # --- Secure Voting ---
    print(f"\n{'=' * 65}")
    print("\n[Secure Voting with Paillier]")

    # Scenario: 7 voters, vote 0 (no) or 1 (yes)
    votes = [1, 0, 1, 1, 0, 1, 1]  # 5 yes, 2 no
    tally = secure_voting_demo(pub, priv, votes)
    print(f"\n  Result: {tally} out of {len(votes)} voted YES")
    print(f"  Individual votes remain encrypted and private!")

    # --- Weighted Voting ---
    print("\n[Weighted Voting (scalar multiplication)]")
    voter_weights = [3, 1, 2, 1, 2]
    voter_choices = [1, 0, 1, 1, 0]  # weighted: 3+0+2+1+0 = 6
    print(f"  Weights:  {voter_weights}")
    print(f"  Choices:  {voter_choices}")

    weighted_sum = pub.encrypt(0)
    for weight, choice in zip(voter_weights, voter_choices):
        enc_vote = pub.encrypt(choice)
        weighted_vote = homomorphic_scalar_mult(enc_vote, weight, pub)
        weighted_sum = homomorphic_add(weighted_sum, weighted_vote, pub)

    weighted_result = priv.decrypt(weighted_sum)
    expected = sum(w * c for w, c in zip(voter_weights, voter_choices))
    print(f"  Weighted tally: {weighted_result}")
    print(f"  Expected:       {expected}")
    print(f"  Match: {weighted_result == expected}")

    print(f"\n{'=' * 65}")

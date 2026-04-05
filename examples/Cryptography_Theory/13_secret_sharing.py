"""
Secret Sharing: Shamir's Scheme and Commitments
================================================
Split a secret into n shares where any k shares can reconstruct it,
but k-1 shares reveal nothing. Uses Lagrange interpolation over
a finite field.
"""

from __future__ import annotations
import hashlib
import random
import secrets


# ---------------------------------------------------------------------------
# Finite Field Arithmetic
# ---------------------------------------------------------------------------

# Why: We work over a prime field GF(p) instead of the integers because:
# 1) Polynomial interpolation over integers leaks information (share values
#    constrain the polynomial), but over GF(p) all values are equally likely.
# 2) The field must be large enough that the secret fits in one element.
# This is a critical security requirement — without it, the scheme fails
# to be information-theoretically secure.
PRIME = 2**127 - 1  # Mersenne prime for efficiency


def _mod_inverse(a: int, p: int) -> int:
    """Modular inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p."""
    # Why: For a prime p, Fermat's little theorem gives a^{p-1} = 1 mod p,
    # so a^{-1} = a^{p-2} mod p. This is simpler than extended GCD for
    # prime fields and avoids the recursive/iterative algorithm.
    return pow(a, p - 2, p)


# ---------------------------------------------------------------------------
# Shamir's Secret Sharing
# ---------------------------------------------------------------------------

# Why: Shamir's scheme (1979) is information-theoretically secure — even
# an adversary with unlimited computing power learns NOTHING from k-1
# shares. This is because a polynomial of degree k-1 is uniquely
# determined by k points, but k-1 points are consistent with every
# possible secret. It's used in key management, multi-party computation,
# and cryptocurrency wallets (e.g., "2-of-3 multisig").

def create_shares(
    secret: int,
    n: int,
    k: int,
    prime: int = PRIME,
) -> list[tuple[int, int]]:
    """Split a secret into n shares with threshold k.

    Any k shares can reconstruct the secret; k-1 shares reveal nothing.
    Uses a random polynomial of degree k-1 where f(0) = secret.

    Returns list of (x, y) pairs where y = f(x) mod prime.
    """
    if k > n:
        raise ValueError("Threshold k must be <= total shares n")
    if secret >= prime:
        raise ValueError(f"Secret must be < {prime}")

    # Why: The polynomial f(x) = a_0 + a_1*x + a_2*x^2 + ... + a_{k-1}*x^{k-1}
    # where a_0 = secret and a_1...a_{k-1} are random. The secret is f(0),
    # so knowing k points lets you interpolate f and recover f(0).
    # With only k-1 points, any value of f(0) is equally likely.
    coefficients = [secret]
    for _ in range(k - 1):
        coefficients.append(random.randrange(1, prime))

    # Why: We evaluate at x = 1, 2, ..., n (never at x = 0, because
    # f(0) = secret, and giving that away defeats the purpose).
    shares = []
    for i in range(1, n + 1):
        y = 0
        for j, coeff in enumerate(coefficients):
            y = (y + coeff * pow(i, j, prime)) % prime
        shares.append((i, y))

    return shares


# ---------------------------------------------------------------------------
# Lagrange Interpolation
# ---------------------------------------------------------------------------

# Why: Lagrange interpolation reconstructs a polynomial from k points.
# We only need f(0) (the secret), not the full polynomial. The formula:
# f(0) = sum(y_i * product((0 - x_j)/(x_i - x_j) for j != i) for each i)
# This is computed entirely in GF(p) for security.

def reconstruct_secret(
    shares: list[tuple[int, int]],
    prime: int = PRIME,
) -> int:
    """Reconstruct the secret from k or more shares using Lagrange interpolation.

    Evaluates the interpolating polynomial at x = 0.
    """
    k = len(shares)
    secret = 0

    for i in range(k):
        x_i, y_i = shares[i]

        # Why: The Lagrange basis polynomial L_i(0) is the product of
        # (0 - x_j) / (x_i - x_j) for all j != i. This gives L_i(x_i) = 1
        # and L_i(x_j) = 0 for j != i, so f(0) = sum(y_i * L_i(0)).
        numerator = 1
        denominator = 1
        for j in range(k):
            if i == j:
                continue
            x_j = shares[j][0]
            numerator = (numerator * (-x_j)) % prime
            denominator = (denominator * (x_i - x_j)) % prime

        lagrange_coeff = (numerator * _mod_inverse(denominator, prime)) % prime
        secret = (secret + y_i * lagrange_coeff) % prime

    return secret


# ---------------------------------------------------------------------------
# Commitment Scheme (Hash-Based)
# ---------------------------------------------------------------------------

# Why: Commitments let you "lock in" a value now and reveal it later.
# Properties: (1) Hiding — the commitment reveals nothing about the value.
# (2) Binding — you can't change the value after committing.
# Used in ZKPs, secure coin flipping, and verifiable secret sharing.

class Commitment:
    """Hash-based commitment scheme."""

    @staticmethod
    def commit(value: bytes) -> tuple[bytes, bytes]:
        """Create a commitment to a value.

        Returns (commitment, randomness) where commitment = H(randomness || value).
        """
        # Why: The randomness (nonce) provides hiding — without it, the
        # verifier could brute-force the value by trying all possibilities
        # and checking against the commitment hash.
        randomness = secrets.token_bytes(32)
        commitment = hashlib.sha256(randomness + value).digest()
        return commitment, randomness

    @staticmethod
    def verify(
        commitment: bytes,
        value: bytes,
        randomness: bytes,
    ) -> bool:
        """Verify that a commitment matches the revealed value."""
        expected = hashlib.sha256(randomness + value).digest()
        return commitment == expected


# ---------------------------------------------------------------------------
# Verifiable Secret Sharing (Feldman's Scheme)
# ---------------------------------------------------------------------------

# Why: Basic Shamir's scheme has a problem: a dishonest dealer can give
# invalid shares. Feldman's VSS adds commitments to the polynomial
# coefficients so that each shareholder can verify their share is correct
# without learning the secret or other shares.

def feldman_vss_create(
    secret: int,
    n: int,
    k: int,
    p: int,
    g: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Create Feldman VSS shares with public commitments.

    Args:
        p: prime for the group Z_p*
        g: generator of Z_p*

    Returns (shares, commitments) where commitments[i] = g^{a_i} mod p.
    """
    # Generate polynomial coefficients
    q = (p - 1) // 2  # subgroup order for safe prime
    coefficients = [secret % q]
    for _ in range(k - 1):
        coefficients.append(random.randrange(1, q))

    # Why: Public commitments C_j = g^{a_j} mod p let anyone verify
    # that share (i, y_i) satisfies g^{y_i} = product(C_j^{i^j})
    # without revealing the coefficients a_j.
    commitments = [pow(g, c, p) for c in coefficients]

    # Create shares (evaluated mod q, the subgroup order)
    shares = []
    for i in range(1, n + 1):
        y = 0
        for j, coeff in enumerate(coefficients):
            y = (y + coeff * pow(i, j, q)) % q
        shares.append((i, y))

    return shares, commitments


def feldman_vss_verify(
    share: tuple[int, int],
    commitments: list[int],
    p: int,
    g: int,
) -> bool:
    """Verify a Feldman VSS share against the public commitments."""
    x, y = share
    # Check: g^y == product(C_j^{x^j}) mod p
    lhs = pow(g, y, p)
    rhs = 1
    for j, cj in enumerate(commitments):
        rhs = (rhs * pow(cj, pow(x, j), p)) % p
    return lhs == rhs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Secret Sharing: Shamir's Scheme")
    print("=" * 65)

    # Basic Shamir's Secret Sharing
    print("\n[Shamir's Secret Sharing (3-of-5)]")
    secret = 123456789
    n, k = 5, 3
    print(f"  Secret: {secret}")
    print(f"  Splitting into {n} shares (threshold = {k})")

    shares = create_shares(secret, n, k)
    for i, (x, y) in enumerate(shares):
        print(f"  Share {i + 1}: (x={x}, y={y})")

    # Reconstruct from k=3 shares
    print(f"\n  Reconstructing from 3 shares (1, 3, 5):")
    subset = [shares[0], shares[2], shares[4]]
    recovered = reconstruct_secret(subset)
    print(f"  Recovered secret: {recovered}")
    print(f"  Match: {recovered == secret}")

    # Different subset
    print(f"\n  Reconstructing from 3 shares (2, 3, 4):")
    subset2 = [shares[1], shares[2], shares[3]]
    recovered2 = reconstruct_secret(subset2)
    print(f"  Recovered secret: {recovered2}")
    print(f"  Match: {recovered2 == secret}")

    # Too few shares
    print(f"\n  Attempting with only 2 shares (below threshold):")
    subset_small = [shares[0], shares[1]]
    wrong_secret = reconstruct_secret(subset_small)
    print(f"  Got: {wrong_secret}")
    print(f"  Match: {wrong_secret == secret} (expected: False)")
    print(f"  With k-1 shares, any secret value is equally possible!")

    # All shares
    print(f"\n  Reconstructing from all 5 shares:")
    recovered_all = reconstruct_secret(shares)
    print(f"  Recovered: {recovered_all}")
    print(f"  Match: {recovered_all == secret}")

    # --- Different Threshold Schemes ---
    print(f"\n{'=' * 65}")
    print(f"\n[Threshold Variations]")
    for n_val, k_val in [(3, 2), (7, 4), (10, 7)]:
        s = create_shares(secret, n_val, k_val)
        # Pick exactly k random shares
        chosen = random.sample(s, k_val)
        rec = reconstruct_secret(chosen)
        print(f"  {k_val}-of-{n_val}: "
              f"recovered = {rec == secret} "
              f"(used shares at x = {[x for x, _ in chosen]})")

    # --- Commitment Scheme ---
    print(f"\n{'=' * 65}")
    print(f"\n[Hash-Based Commitment Scheme]")
    value = b"my secret vote: YES"
    commitment, randomness = Commitment.commit(value)
    print(f"  Value: '{value.decode()}'")
    print(f"  Commitment: {commitment.hex()[:32]}...")
    print(f"  (Commitment reveals nothing about the value)")

    # Verify
    print(f"\n  Reveal phase:")
    valid = Commitment.verify(commitment, value, randomness)
    print(f"  Verified: {valid}")

    # Tampered value
    tampered = b"my secret vote: NO!"
    valid_t = Commitment.verify(commitment, tampered, randomness)
    print(f"  Tampered value verified: {valid_t}")
    print(f"  (Cannot change value after committing — binding property)")

    # --- Verifiable Secret Sharing ---
    print(f"\n{'=' * 65}")
    print(f"\n[Feldman's Verifiable Secret Sharing]")

    # Use small parameters for demo
    # Safe prime p = 2*q + 1
    vss_p = 1019  # safe prime
    vss_q = 509
    vss_g = 2     # generator
    # Verify generator
    assert pow(vss_g, vss_q, vss_p) == 1, "Bad generator"
    assert pow(vss_g, 2, vss_p) != 1, "Trivial generator"

    vss_secret = 42
    vss_n, vss_k = 5, 3
    print(f"  Secret: {vss_secret}")
    print(f"  Group: Z_{vss_p}*, generator g = {vss_g}")

    vss_shares, commitments = feldman_vss_create(
        vss_secret, vss_n, vss_k, vss_p, vss_g
    )

    print(f"  Commitments: {commitments}")
    print(f"  Shares and verification:")
    for share in vss_shares:
        valid = feldman_vss_verify(share, commitments, vss_p, vss_g)
        print(f"    Share (x={share[0]}, y={share[1]}): "
              f"{'VALID' if valid else 'INVALID'}")

    # Reconstruct
    vss_subset = vss_shares[:vss_k]
    vss_recovered = reconstruct_secret(vss_subset, prime=vss_q)
    print(f"\n  Reconstructed from {vss_k} shares: {vss_recovered}")
    print(f"  Match: {vss_recovered == vss_secret}")

    # Summary
    print(f"\n[Secret Sharing Applications]")
    print(f"  - Key management: Split master key across locations")
    print(f"  - Cryptocurrency: 2-of-3 wallet backup")
    print(f"  - Nuclear launch: Multiple authorization required")
    print(f"  - Secure MPC: Compute on shared secrets jointly")

    print(f"\n{'=' * 65}")

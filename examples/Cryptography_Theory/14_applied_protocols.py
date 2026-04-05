"""
Applied Cryptographic Protocols: TLS 1.3, Signal, MPC
=====================================================
Demonstrates how cryptographic primitives compose into
real-world protocols. Includes simplified TLS handshake,
Double Ratchet concept, and commitment schemes.
"""

from __future__ import annotations
import hashlib
import hmac
import os
import secrets


# ---------------------------------------------------------------------------
# 1. Simplified TLS 1.3 Handshake
# ---------------------------------------------------------------------------

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-Extract: derive a pseudorandom key from input keying material."""
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-Expand: expand a PRK into output keying material."""
    blocks = []
    prev = b""
    for i in range(1, (length + 31) // 32 + 1):
        prev = hmac.new(prk, prev + info + bytes([i]), hashlib.sha256).digest()
        blocks.append(prev)
    return b"".join(blocks)[:length]


def tls13_handshake_simulation() -> None:
    """Simulate a simplified TLS 1.3 handshake key schedule."""
    print("  TLS 1.3 Handshake Simulation:")

    # Step 1: Key exchange (simulated ECDHE)
    client_private = secrets.token_bytes(32)
    server_private = secrets.token_bytes(32)
    # In real TLS, this would be X25519 or P-256; we simulate a shared secret
    shared_secret = hashlib.sha256(client_private + server_private).digest()
    print(f"  1. ECDHE shared secret: {shared_secret[:8].hex()}...")

    # Step 2: Early secret (from PSK or zeros)
    psk = b"\x00" * 32  # No pre-shared key
    early_secret = hkdf_extract(b"\x00" * 32, psk)
    print(f"  2. Early secret:        {early_secret[:8].hex()}...")

    # Step 3: Handshake secret
    derived = hkdf_expand(early_secret, b"derived", 32)
    handshake_secret = hkdf_extract(derived, shared_secret)
    print(f"  3. Handshake secret:    {handshake_secret[:8].hex()}...")

    # Step 4: Client/Server handshake traffic keys
    client_hs_traffic = hkdf_expand(handshake_secret, b"c hs traffic", 32)
    server_hs_traffic = hkdf_expand(handshake_secret, b"s hs traffic", 32)
    print(f"  4. Client HS key:       {client_hs_traffic[:8].hex()}...")
    print(f"     Server HS key:       {server_hs_traffic[:8].hex()}...")

    # Step 5: Application traffic keys
    derived2 = hkdf_expand(handshake_secret, b"derived", 32)
    master_secret = hkdf_extract(derived2, b"\x00" * 32)
    client_app_traffic = hkdf_expand(master_secret, b"c ap traffic", 32)
    server_app_traffic = hkdf_expand(master_secret, b"s ap traffic", 32)
    print(f"  5. Client App key:      {client_app_traffic[:8].hex()}...")
    print(f"     Server App key:      {server_app_traffic[:8].hex()}...")
    print("  Handshake complete — 1-RTT")


# ---------------------------------------------------------------------------
# 2. Double Ratchet (Signal Protocol Concept)
# ---------------------------------------------------------------------------

class SimpleRatchet:
    """Simplified symmetric ratchet for forward secrecy demonstration."""

    def __init__(self, root_key: bytes):
        self.chain_key = root_key

    def advance(self) -> bytes:
        """Advance the ratchet: derive message key and new chain key."""
        message_key = hmac.new(self.chain_key, b"message", hashlib.sha256).digest()
        self.chain_key = hmac.new(self.chain_key, b"chain", hashlib.sha256).digest()
        return message_key


def double_ratchet_demo() -> None:
    """Demonstrate the forward secrecy property of the Double Ratchet."""
    print("\n  Double Ratchet (Signal Protocol Concept):")

    root = secrets.token_bytes(32)
    alice = SimpleRatchet(root)
    bob = SimpleRatchet(root)

    # Alice sends 3 messages
    for i in range(3):
        alice_key = alice.advance()
        bob_key = bob.advance()
        match = alice_key == bob_key
        print(f"  Message {i+1}: Alice key={alice_key[:6].hex()}... Bob key={bob_key[:6].hex()}... Match={match}")

    print()
    print("  Forward secrecy: compromising current key cannot recover past keys.")
    print("  Each message uses a unique key derived from the chain ratchet.")
    print("  The DH ratchet (not shown) provides post-compromise security.")


# ---------------------------------------------------------------------------
# 3. Commitment Scheme
# ---------------------------------------------------------------------------

def commit(value: bytes, randomness: bytes | None = None) -> tuple[bytes, bytes]:
    """Create a cryptographic commitment to a value."""
    r = randomness or secrets.token_bytes(32)
    commitment = hashlib.sha256(value + r).digest()
    return commitment, r


def verify_commitment(commitment: bytes, value: bytes, randomness: bytes) -> bool:
    """Verify that a commitment opens to the claimed value."""
    return hashlib.sha256(value + randomness).digest() == commitment


def commitment_demo() -> None:
    """Demonstrate hiding and binding properties of commitments."""
    print("\n  Commitment Scheme:")

    # Alice commits to her choice
    choice = b"heads"
    c, r = commit(choice)
    print(f"  Alice commits: {c.hex()[:16]}...")
    print("  (Bob cannot determine the choice from the commitment — hiding)")

    # Later, Alice reveals
    valid = verify_commitment(c, choice, r)
    print(f"  Alice reveals: '{choice.decode()}' with randomness {r.hex()[:16]}...")
    print(f"  Verification: {valid}")

    # Alice cannot change her mind (binding)
    fake = verify_commitment(c, b"tails", r)
    print(f"  Trying to claim 'tails': {fake} (binding property)")


# ---------------------------------------------------------------------------
# 4. Shamir's Secret Sharing
# ---------------------------------------------------------------------------

PRIME = 2**127 - 1  # Mersenne prime


def _eval_poly(coeffs: list[int], x: int) -> int:
    """Evaluate polynomial at x modulo PRIME."""
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % PRIME
    return result


def split_secret(secret: int, n: int, k: int) -> list[tuple[int, int]]:
    """Split a secret into n shares with threshold k."""
    coeffs = [secret] + [secrets.randbelow(PRIME) for _ in range(k - 1)]
    return [(i, _eval_poly(coeffs, i)) for i in range(1, n + 1)]


def reconstruct_secret(shares: list[tuple[int, int]]) -> int:
    """Reconstruct secret from k shares using Lagrange interpolation."""
    k = len(shares)
    secret = 0
    for i in range(k):
        xi, yi = shares[i]
        num = den = 1
        for j in range(k):
            if i != j:
                xj = shares[j][0]
                num = (num * (-xj)) % PRIME
                den = (den * (xi - xj)) % PRIME
        secret = (secret + yi * num * pow(den, PRIME - 2, PRIME)) % PRIME
    return secret


def secret_sharing_demo() -> None:
    """Demonstrate (3,5) threshold secret sharing."""
    print("\n  Shamir's Secret Sharing (3-of-5):")

    secret = 42
    shares = split_secret(secret, n=5, k=3)
    for i, (x, y) in enumerate(shares):
        print(f"  Share {i+1}: ({x}, {y % 10000}...)")

    # Reconstruct with 3 shares
    recovered = reconstruct_secret(shares[:3])
    print(f"  Reconstructed (shares 1,2,3): {recovered}")

    # Reconstruct with different 3 shares
    recovered2 = reconstruct_secret([shares[0], shares[2], shares[4]])
    print(f"  Reconstructed (shares 1,3,5): {recovered2}")

    # Cannot reconstruct with only 2 shares
    wrong = reconstruct_secret(shares[:2])
    print(f"  With only 2 shares: {wrong} (incorrect — need ≥3)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Applied Cryptographic Protocols")
    print("=" * 60)
    tls13_handshake_simulation()
    double_ratchet_demo()
    commitment_demo()
    secret_sharing_demo()

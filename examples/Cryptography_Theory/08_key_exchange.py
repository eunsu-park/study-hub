"""
Key Exchange Protocols: Diffie-Hellman, ECDH, and Key Derivation
================================================================
Demonstrates how two parties establish a shared secret over an
insecure channel, and how raw shared secrets are converted into
usable symmetric keys via key derivation functions.
"""

from __future__ import annotations
import hashlib
import hmac
import random
import struct

# Import elliptic curve module (with inline fallback)
try:
    from importlib import import_module
    _ec = import_module("06_elliptic_curves")
    Point = _ec.Point
    EllipticCurve = _ec.EllipticCurve
    get_demo_curve = _ec.get_demo_curve
    ECDHParty = _ec.ECDHParty
except (ImportError, ModuleNotFoundError):
    # Minimal fallback
    def _extended_gcd(a, b):
        old_r, r = a, b
        old_s, s = 1, 0
        while r:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
        return old_r, old_s

    def _mod_inverse(a, p):
        g, x = _extended_gcd(a % p, p)
        return x % p

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Point:
        x: int | None; y: int | None
        @property
        def is_infinity(self): return self.x is None
        def __repr__(self): return "O" if self.is_infinity else f"({self.x}, {self.y})"

    INFINITY = Point(None, None)

    class EllipticCurve:
        def __init__(self, a, b, p, G, n):
            self.a, self.b, self.p, self.G, self.n = a, b, p, G, n
        def add(self, P, Q):
            if P.is_infinity: return Q
            if Q.is_infinity: return P
            if P.x == Q.x and P.y != Q.y: return INFINITY
            p = self.p
            if P == Q:
                s = (3*P.x*P.x + self.a) * _mod_inverse(2*P.y, p) % p
            else:
                s = (Q.y - P.y) * _mod_inverse(Q.x - P.x, p) % p
            xr = (s*s - P.x - Q.x) % p
            yr = (s*(P.x - xr) - P.y) % p
            return Point(xr, yr)
        def scalar_multiply(self, k, P):
            R, A = INFINITY, P
            while k > 0:
                if k & 1: R = self.add(R, A)
                A = self.add(A, A); k >>= 1
            return R

    def get_demo_curve():
        return EllipticCurve(2, 3, 97, Point(3, 6), 89)

    class ECDHParty:
        def __init__(self, name, curve):
            self.name = name; self.curve = curve
            self.private_key = random.randrange(1, curve.n)
            self.public_key = curve.scalar_multiply(self.private_key, curve.G)
        def compute_shared_secret(self, other_pub):
            return self.curve.scalar_multiply(self.private_key, other_pub)


# ---------------------------------------------------------------------------
# Classic Diffie-Hellman (over integers mod p)
# ---------------------------------------------------------------------------

# Why: Diffie-Hellman (1976) was the FIRST public key exchange protocol.
# It allows two parties to agree on a shared secret over an insecure channel.
# Security relies on the Decisional Diffie-Hellman (DDH) assumption:
# given g, g^a, g^b, it's hard to distinguish g^{ab} from a random element.

# Why: We need a safe prime p = 2q + 1 where q is also prime. This ensures
# the multiplicative group Z_p* has a large prime-order subgroup, preventing
# small-subgroup attacks where an attacker forces the shared secret into
# a small set of values.
def _is_prime(n: int) -> bool:
    """Simple primality test for small numbers."""
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


def find_safe_prime(min_val: int = 100, max_val: int = 10000) -> int:
    """Find a safe prime p = 2q + 1 where q is also prime."""
    for q in range(min_val, max_val):
        if _is_prime(q):
            p = 2 * q + 1
            if _is_prime(p):
                return p
    raise ValueError("No safe prime found in range")


def find_generator(p: int) -> int:
    """Find a generator of the multiplicative group Z_p*."""
    # Why: A generator g has order p-1 in Z_p*. For safe prime p = 2q+1,
    # the possible orders are 1, 2, q, or 2q. We check that g is not
    # of order 1 or 2, guaranteeing it generates the full group.
    for g in range(2, p):
        if pow(g, 2, p) != 1 and pow(g, (p - 1) // 2, p) != 1:
            return g
    raise ValueError("No generator found")


class DHParty:
    """One party in a classic Diffie-Hellman key exchange."""

    def __init__(self, name: str, p: int, g: int) -> None:
        self.name = name
        self.p = p
        self.g = g
        # Why: Private key is random in [2, p-2]. We exclude 0, 1, p-1
        # because they produce trivial public keys (1, g, or g^{-1}).
        self.private_key = random.randrange(2, p - 1)
        self.public_key = pow(g, self.private_key, p)

    def compute_shared_secret(self, other_public: int) -> int:
        """Compute shared secret: other_public^private mod p."""
        return pow(other_public, self.private_key, self.p)


# ---------------------------------------------------------------------------
# Key Derivation Function (simplified HKDF)
# ---------------------------------------------------------------------------

# Why: Raw DH/ECDH shared secrets should NEVER be used directly as
# encryption keys. They have structure (e.g., they're elements of a
# specific group) and may have biased bits. A KDF like HKDF extracts
# uniform randomness and derives keys of the desired length.

def hkdf_extract(salt: bytes, input_key_material: bytes) -> bytes:
    """HKDF-Extract: extract a pseudorandom key from input material.

    Uses HMAC-SHA256 as the extraction function.
    """
    # Why: The extract step "concentrates" the entropy from the input
    # material into a fixed-size pseudorandom key (PRK). The salt
    # adds domain separation and additional randomness.
    if not salt:
        salt = b"\x00" * 32
    return hmac.new(salt, input_key_material, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """HKDF-Expand: expand a PRK into output keying material.

    Parameters:
        prk: pseudorandom key from HKDF-Extract
        info: context/application-specific info (e.g., "encryption key")
        length: desired output length in bytes
    """
    # Why: The expand step derives multiple keys from a single PRK using
    # different 'info' strings. This lets you derive encryption key,
    # MAC key, and IV from the same shared secret without correlation.
    hash_len = 32  # SHA-256 output
    n_blocks = (length + hash_len - 1) // hash_len

    okm = b""
    t = b""  # T(0) = empty

    for i in range(1, n_blocks + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t

    return okm[:length]


def hkdf(
    input_key_material: bytes,
    length: int,
    salt: bytes = b"",
    info: bytes = b"",
) -> bytes:
    """Full HKDF: Extract-then-Expand."""
    prk = hkdf_extract(salt, input_key_material)
    return hkdf_expand(prk, info, length)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Key Exchange Protocols")
    print("=" * 65)

    # --- Classic Diffie-Hellman ---
    print("\n[Classic Diffie-Hellman Key Exchange]")
    p = find_safe_prime(500)
    g = find_generator(p)
    print(f"  Public parameters: p = {p}, g = {g}")

    alice = DHParty("Alice", p, g)
    bob = DHParty("Bob", p, g)

    print(f"\n  Alice: private = {alice.private_key}, "
          f"public = g^a = {alice.public_key}")
    print(f"  Bob:   private = {bob.private_key}, "
          f"public = g^b = {bob.public_key}")

    alice_secret = alice.compute_shared_secret(bob.public_key)
    bob_secret = bob.compute_shared_secret(alice.public_key)

    print(f"\n  Alice computes: Bob_pub^a mod p = {alice_secret}")
    print(f"  Bob computes:   Alice_pub^b mod p = {bob_secret}")
    print(f"  Match: {alice_secret == bob_secret}")

    # Eve's view
    print(f"\n  Eve sees: p={p}, g={g}, "
          f"g^a={alice.public_key}, g^b={bob.public_key}")
    print(f"  Eve needs to solve discrete log to find a or b!")

    # --- ECDH ---
    print(f"\n{'=' * 65}")
    print(f"\n[ECDH Key Exchange (Elliptic Curve)]")
    curve = get_demo_curve()
    print(f"  Curve: y^2 = x^3 + {curve.a}x + {curve.b} (mod {curve.p})")
    print(f"  Generator: {curve.G}, Order: {curve.n}")

    alice_ec = ECDHParty("Alice", curve)
    bob_ec = ECDHParty("Bob", curve)

    print(f"\n  Alice: private = {alice_ec.private_key}, "
          f"public = {alice_ec.public_key}")
    print(f"  Bob:   private = {bob_ec.private_key}, "
          f"public = {bob_ec.public_key}")

    alice_ec_secret = alice_ec.compute_shared_secret(bob_ec.public_key)
    bob_ec_secret = bob_ec.compute_shared_secret(alice_ec.public_key)

    print(f"\n  Alice's shared secret: {alice_ec_secret}")
    print(f"  Bob's shared secret:   {bob_ec_secret}")
    print(f"  Match: {alice_ec_secret == bob_ec_secret}")

    # Why: ECDH offers the same security as DH but with much smaller keys.
    # A 256-bit ECC key provides security equivalent to a 3072-bit RSA/DH key.
    print(f"\n  [ECDH vs DH Security Comparison]")
    print(f"  {'Security Level':<20} {'DH key size':<15} {'ECDH key size':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'80 bits':<20} {'1024 bits':<15} {'160 bits':<15}")
    print(f"  {'128 bits':<20} {'3072 bits':<15} {'256 bits':<15}")
    print(f"  {'192 bits':<20} {'7680 bits':<15} {'384 bits':<15}")
    print(f"  {'256 bits':<20} {'15360 bits':<15} {'512 bits':<15}")

    # --- Key Derivation ---
    print(f"\n{'=' * 65}")
    print(f"\n[Key Derivation Function (HKDF)]")

    # Derive keys from the DH shared secret
    shared_secret = alice_secret.to_bytes(
        (alice_secret.bit_length() + 7) // 8, "big"
    )
    salt = b"application-salt"
    print(f"  Raw shared secret: {shared_secret.hex()}")

    # Derive different keys for different purposes
    enc_key = hkdf(shared_secret, 32, salt, b"encryption-key")
    mac_key = hkdf(shared_secret, 32, salt, b"mac-key")
    iv = hkdf(shared_secret, 16, salt, b"initialization-vector")

    print(f"\n  Derived encryption key (32 bytes): {enc_key.hex()}")
    print(f"  Derived MAC key (32 bytes):        {mac_key.hex()}")
    print(f"  Derived IV (16 bytes):             {iv.hex()}")
    print(f"  Keys are independent (different 'info' parameter)")

    # Verify HKDF consistency
    enc_key2 = hkdf(shared_secret, 32, salt, b"encryption-key")
    print(f"\n  Deterministic: same inputs -> same output: {enc_key == enc_key2}")

    # Show that different salts give different keys
    enc_key_alt = hkdf(shared_secret, 32, b"different-salt", b"encryption-key")
    print(f"  Different salt -> different key: {enc_key != enc_key_alt}")

    print(f"\n{'=' * 65}")

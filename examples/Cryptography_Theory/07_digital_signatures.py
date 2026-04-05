"""
Digital Signatures: ECDSA and Schnorr
======================================
Implements ECDSA signing/verification, demonstrates the devastating
nonce-reuse vulnerability, and shows the Schnorr signature scheme.
"""

from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass

# Import elliptic curve module (with inline fallback)
try:
    from importlib import import_module
    _ec = import_module("06_elliptic_curves")
    Point = _ec.Point
    INFINITY = _ec.INFINITY
    EllipticCurve = _ec.EllipticCurve
    get_demo_curve = _ec.get_demo_curve
    mod_inverse = _ec.mod_inverse
except (ImportError, ModuleNotFoundError):
    # Minimal fallback — see 06_elliptic_curves.py for full implementation
    def _extended_gcd(a, b):
        old_r, r = a, b
        old_s, s = 1, 0
        while r:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
        return old_r, old_s

    def mod_inverse(a, p):
        g, x = _extended_gcd(a % p, p)
        if g != 1:
            raise ValueError(f"No inverse for {a} mod {p}")
        return x % p

    @dataclass(frozen=True)
    class Point:
        x: int | None
        y: int | None
        @property
        def is_infinity(self):
            return self.x is None
        def __repr__(self):
            return "O" if self.is_infinity else f"({self.x}, {self.y})"

    INFINITY = Point(None, None)

    class EllipticCurve:
        def __init__(self, a, b, p, G, n):
            self.a, self.b, self.p, self.G, self.n = a, b, p, G, n
        def add(self, P, Q):
            if P.is_infinity: return Q
            if Q.is_infinity: return P
            if P.x == Q.x and P.y != Q.y: return INFINITY
            if P == Q:
                s = (3*P.x*P.x + self.a) * mod_inverse(2*P.y, self.p) % self.p
            else:
                s = (Q.y - P.y) * mod_inverse(Q.x - P.x, self.p) % self.p
            xr = (s*s - P.x - Q.x) % self.p
            yr = (s*(P.x - xr) - P.y) % self.p
            return Point(xr, yr)
        def scalar_multiply(self, k, P):
            if k == 0: return INFINITY
            R, A = INFINITY, P
            while k > 0:
                if k & 1: R = self.add(R, A)
                A = self.add(A, A)
                k >>= 1
            return R

    def get_demo_curve():
        return EllipticCurve(2, 3, 97, Point(3, 6), 89)


# ---------------------------------------------------------------------------
# Hash-to-scalar helper
# ---------------------------------------------------------------------------

def hash_to_scalar(data: bytes, n: int) -> int:
    """Hash arbitrary data to a scalar in [1, n-1]."""
    # Why: We need to map messages to scalars for signing. Using SHA-256
    # and reducing mod n gives a uniform distribution (approximately,
    # since 2^256 >> n for our small demo curve).
    h = int.from_bytes(hashlib.sha256(data).digest(), "big")
    return (h % (n - 1)) + 1  # ensure non-zero


# ---------------------------------------------------------------------------
# ECDSA
# ---------------------------------------------------------------------------

# Why: ECDSA is the most widely deployed elliptic curve signature scheme
# (Bitcoin, TLS, code signing). It's a variant of the ElGamal signature
# adapted to elliptic curves. Each signature uses a random nonce k;
# if k is ever reused or predictable, the private key is completely exposed.

def ecdsa_sign(
    message: bytes,
    private_key: int,
    curve: EllipticCurve,
) -> tuple[int, int]:
    """Sign a message using ECDSA.

    Returns (r, s) signature pair.
    """
    n = curve.n

    while True:
        # Why: k MUST be a cryptographically random nonce, unique per signature.
        # If k is reused across two messages, the private key can be extracted
        # (demonstrated below). RFC 6979 defines deterministic k generation.
        k = random.randrange(1, n)

        # Compute r = (k * G).x mod n
        R = curve.scalar_multiply(k, curve.G)
        r = R.x % n
        if r == 0:
            continue

        # Compute message hash
        z = hash_to_scalar(message, n)

        # Why: s = k^{-1} * (z + r*d) mod n, where d is the private key.
        # This equation binds the message hash z to the private key d
        # through the random nonce k. Verification works because the
        # verifier can reconstruct R from (r, s) and the public key.
        s = (mod_inverse(k, n) * (z + r * private_key)) % n
        if s == 0:
            continue

        return r, s


def ecdsa_verify(
    message: bytes,
    signature: tuple[int, int],
    public_key: Point,
    curve: EllipticCurve,
) -> bool:
    """Verify an ECDSA signature."""
    r, s = signature
    n = curve.n

    if not (1 <= r < n and 1 <= s < n):
        return False

    z = hash_to_scalar(message, n)

    # Why: The verification equation reconstructs the nonce point R:
    # w = s^{-1}, u1 = z*w, u2 = r*w
    # R' = u1*G + u2*Q (where Q is public key)
    # Valid iff R'.x mod n == r
    # This works because: u1*G + u2*Q = (z*s^{-1})*G + (r*s^{-1})*(d*G)
    # = s^{-1}*(z + r*d)*G = s^{-1}*(s*k)*G = k*G = R
    w = mod_inverse(s, n)
    u1 = (z * w) % n
    u2 = (r * w) % n

    P = curve.add(
        curve.scalar_multiply(u1, curve.G),
        curve.scalar_multiply(u2, public_key),
    )

    if P.is_infinity:
        return False

    return P.x % n == r


# ---------------------------------------------------------------------------
# Nonce Reuse Attack
# ---------------------------------------------------------------------------

# Why: This is one of the most important practical attacks in cryptography.
# In 2010, Sony's PlayStation 3 code signing key was extracted because
# they used a CONSTANT nonce for every ECDSA signature. Given two
# signatures (r, s1) and (r, s2) with the same k, the private key
# can be computed algebraically.
def ecdsa_nonce_reuse_attack(
    msg1: bytes, sig1: tuple[int, int],
    msg2: bytes, sig2: tuple[int, int],
    curve: EllipticCurve,
) -> int | None:
    """Extract the private key from two signatures that used the same nonce.

    Returns the private key, or None if signatures used different nonces.
    """
    r1, s1 = sig1
    r2, s2 = sig2
    n = curve.n

    # Why: If both signatures used the same k, then r1 == r2 (since
    # r = (k*G).x). From s1 = k^{-1}(z1 + r*d) and s2 = k^{-1}(z2 + r*d),
    # we get: s1 - s2 = k^{-1}(z1 - z2), so k = (z1 - z2) / (s1 - s2).
    # Then d = (s1*k - z1) / r.
    if r1 != r2:
        return None  # different nonces

    z1 = hash_to_scalar(msg1, n)
    z2 = hash_to_scalar(msg2, n)

    # Recover k
    ds = (s1 - s2) % n
    if ds == 0:
        return None
    k = ((z1 - z2) * mod_inverse(ds, n)) % n

    # Recover private key d
    d = ((s1 * k - z1) * mod_inverse(r1, n)) % n
    return d


# ---------------------------------------------------------------------------
# Schnorr Signature
# ---------------------------------------------------------------------------

# Why: Schnorr signatures are simpler and more elegant than ECDSA. They
# have a clean security proof under the random oracle model, support
# efficient multi-signatures (MuSig), and are linear (enabling signature
# aggregation). Bitcoin adopted Schnorr in the Taproot upgrade (2021).
def schnorr_sign(
    message: bytes,
    private_key: int,
    curve: EllipticCurve,
) -> tuple[Point, int]:
    """Schnorr signature: returns (R, s) where R is a curve point."""
    n = curve.n

    # Why: Like ECDSA, we need a random nonce. But in Schnorr, the
    # signature structure is simpler: s = k - e*d (mod n),
    # where e = H(R || message). No modular inverse needed!
    k = random.randrange(1, n)
    R = curve.scalar_multiply(k, curve.G)

    # Challenge: e = H(R || msg)
    e_data = f"{R.x},{R.y}".encode() + message
    e = hash_to_scalar(e_data, n)

    # s = k - e * private_key (mod n)
    s = (k - e * private_key) % n

    return R, s


def schnorr_verify(
    message: bytes,
    signature: tuple[Point, int],
    public_key: Point,
    curve: EllipticCurve,
) -> bool:
    """Verify a Schnorr signature."""
    R, s = signature
    n = curve.n

    # Recompute challenge
    e_data = f"{R.x},{R.y}".encode() + message
    e = hash_to_scalar(e_data, n)

    # Why: Verification checks s*G + e*Q == R, which expands to
    # (k - e*d)*G + e*(d*G) = k*G - e*d*G + e*d*G = k*G = R. QED.
    lhs = curve.add(
        curve.scalar_multiply(s, curve.G),
        curve.scalar_multiply(e, public_key),
    )

    return lhs == R


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Digital Signatures: ECDSA & Schnorr")
    print("=" * 65)

    curve = get_demo_curve()
    print(f"\n[Curve: y^2 = x^3 + {curve.a}x + {curve.b} mod {curve.p}, n={curve.n}]")

    # Generate keypair
    private_key = random.randrange(1, curve.n)
    public_key = curve.scalar_multiply(private_key, curve.G)
    print(f"  Private key: {private_key}")
    print(f"  Public key:  {public_key}")

    # --- ECDSA ---
    print(f"\n[ECDSA Sign & Verify]")
    msg = b"Transfer 10 BTC to Alice"
    r, s = ecdsa_sign(msg, private_key, curve)
    print(f"  Message: '{msg.decode()}'")
    print(f"  Signature: (r={r}, s={s})")

    valid = ecdsa_verify(msg, (r, s), public_key, curve)
    print(f"  Verification: {'VALID' if valid else 'INVALID'}")

    # Tampered message
    tampered = b"Transfer 10 BTC to Eve!"
    valid_t = ecdsa_verify(tampered, (r, s), public_key, curve)
    print(f"  Tampered msg verification: {'VALID' if valid_t else 'INVALID'}")

    # --- Nonce Reuse Attack ---
    print(f"\n[ECDSA Nonce Reuse Attack]")
    print(f"  Demonstrating why nonce reuse is CATASTROPHIC:")

    # Sign two messages with the SAME nonce (by fixing random seed)
    fixed_seed = random.randrange(1, 10**9)

    random.seed(fixed_seed)
    msg1 = b"Message one"
    sig1 = ecdsa_sign(msg1, private_key, curve)

    random.seed(fixed_seed)  # same seed -> same k
    msg2 = b"Message two"
    sig2 = ecdsa_sign(msg2, private_key, curve)

    print(f"  Signature 1: r={sig1[0]}, s={sig1[1]}")
    print(f"  Signature 2: r={sig2[0]}, s={sig2[1]}")
    print(f"  Same r value (same nonce): {sig1[0] == sig2[0]}")

    recovered_key = ecdsa_nonce_reuse_attack(msg1, sig1, msg2, sig2, curve)
    print(f"\n  Actual private key:    {private_key}")
    print(f"  Recovered private key: {recovered_key}")
    print(f"  Key recovered: {recovered_key == private_key}")
    print(f"  LESSON: NEVER reuse a nonce in ECDSA!")

    # Reseed properly
    random.seed()

    # --- Schnorr ---
    print(f"\n[Schnorr Signature]")
    msg = b"I approve this transaction"
    R, s_val = schnorr_sign(msg, private_key, curve)
    print(f"  Message: '{msg.decode()}'")
    print(f"  Signature: R={R}, s={s_val}")

    valid = schnorr_verify(msg, (R, s_val), public_key, curve)
    print(f"  Verification: {'VALID' if valid else 'INVALID'}")

    # Compare ECDSA vs Schnorr
    print(f"\n[ECDSA vs Schnorr Comparison]")
    print(f"  {'Property':<30} {'ECDSA':<15} {'Schnorr':<15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Signature components':<30} {'(r, s) ints':<15} {'(R, s) pt+int':<15}")
    print(f"  {'Needs modular inverse':<30} {'Yes':<15} {'No':<15}")
    print(f"  {'Security proof':<30} {'Complex':<15} {'Simple (RO)':<15}")
    print(f"  {'Multi-sig support':<30} {'Difficult':<15} {'Native':<15}")
    print(f"  {'Signature aggregation':<30} {'No':<15} {'Yes':<15}")

    print(f"\n{'=' * 65}")

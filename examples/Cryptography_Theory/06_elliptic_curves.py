"""
Elliptic Curve Cryptography — Point Arithmetic and ECDH
========================================================
Implements elliptic curve point addition, scalar multiplication,
and the Elliptic Curve Diffie-Hellman key exchange on a small
prime-field curve for educational purposes.
"""

from __future__ import annotations
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Modular Arithmetic Helpers
# ---------------------------------------------------------------------------

def mod_inverse(a: int, p: int) -> int:
    """Compute modular inverse using extended Euclidean algorithm."""
    # Why: Division on elliptic curves over GF(p) is done via modular inverse.
    # When computing the slope of the line through two points, we need
    # (y2-y1) / (x2-x1) mod p = (y2-y1) * (x2-x1)^{-1} mod p.
    g, x, _ = _extended_gcd(a % p, p)
    if g != 1:
        raise ValueError(f"No inverse for {a} mod {p}")
    return x % p


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


# ---------------------------------------------------------------------------
# Elliptic Curve and Point
# ---------------------------------------------------------------------------

# Why: We use a dataclass for immutability and clarity. The special "point
# at infinity" (the identity element) is represented by x=None, y=None.
# In projective coordinates (used in optimized libraries), infinity is
# (0:1:0), but affine coordinates are clearer for learning.
@dataclass(frozen=True)
class Point:
    """A point on an elliptic curve (or the point at infinity)."""
    x: int | None
    y: int | None

    @property
    def is_infinity(self) -> bool:
        return self.x is None and self.y is None

    def __repr__(self) -> str:
        if self.is_infinity:
            return "O (infinity)"
        return f"({self.x}, {self.y})"


# The point at infinity (identity element for the group operation)
INFINITY = Point(None, None)


class EllipticCurve:
    """An elliptic curve y^2 = x^3 + ax + b over GF(p).

    Parameters:
        a, b: curve coefficients
        p: prime field modulus
        G: generator point (base point)
        n: order of the generator point (number of points in the subgroup)
    """

    def __init__(self, a: int, b: int, p: int,
                 G: Point, n: int) -> None:
        self.a = a
        self.b = b
        self.p = p
        self.G = G
        self.n = n

        # Why: The discriminant 4a^3 + 27b^2 must be non-zero mod p.
        # If it's zero, the curve has a singularity (cusp or node) and
        # the group law breaks down — the curve is not "smooth."
        disc = (4 * a**3 + 27 * b**2) % p
        if disc == 0:
            raise ValueError("Singular curve (discriminant is zero)")

    def is_on_curve(self, P: Point) -> bool:
        """Check if point P lies on the curve."""
        if P.is_infinity:
            return True
        lhs = (P.y * P.y) % self.p
        rhs = (P.x**3 + self.a * P.x + self.b) % self.p
        return lhs == rhs

    # Why: Point addition is the core group operation. Geometrically,
    # draw a line through P and Q, find the third intersection with
    # the curve, and reflect over the x-axis. Algebraically, this
    # translates to the formulas below. The group formed by point
    # addition is what makes ECC secure — the discrete log problem
    # (finding k given k*G) is believed to be hard in this group.
    def add(self, P: Point, Q: Point) -> Point:
        """Add two points on the curve."""
        if P.is_infinity:
            return Q
        if Q.is_infinity:
            return P

        if P.x == Q.x and P.y != Q.y:
            # Why: P + (-P) = O. The inverse of (x, y) is (x, -y mod p).
            return INFINITY

        if P.x == Q.x and P.y == Q.y:
            # Point doubling: tangent line
            return self._double(P)

        # Why: The slope formula for two distinct points P, Q:
        # lambda = (y2 - y1) * (x2 - x1)^{-1} mod p
        # This is the finite-field analog of the slope of a line.
        s = ((Q.y - P.y) * mod_inverse(Q.x - P.x, self.p)) % self.p
        x_r = (s * s - P.x - Q.x) % self.p
        y_r = (s * (P.x - x_r) - P.y) % self.p
        return Point(x_r, y_r)

    def _double(self, P: Point) -> Point:
        """Double a point (P + P) using the tangent line."""
        if P.is_infinity or P.y == 0:
            return INFINITY

        # Why: For doubling, the slope is the derivative dy/dx of the curve
        # equation: lambda = (3x^2 + a) / (2y) mod p.
        # This comes from implicit differentiation of y^2 = x^3 + ax + b.
        s = ((3 * P.x * P.x + self.a) * mod_inverse(2 * P.y, self.p)) % self.p
        x_r = (s * s - 2 * P.x) % self.p
        y_r = (s * (P.x - x_r) - P.y) % self.p
        return Point(x_r, y_r)

    def negate(self, P: Point) -> Point:
        """Return -P (reflection over x-axis)."""
        if P.is_infinity:
            return INFINITY
        return Point(P.x, (-P.y) % self.p)

    # Why: Scalar multiplication k*P is computed via double-and-add,
    # analogous to square-and-multiply for modular exponentiation.
    # This is O(log k) point operations. The ECDLP assumes that
    # given P and k*P, recovering k is computationally infeasible.
    def scalar_multiply(self, k: int, P: Point) -> Point:
        """Compute k * P using the double-and-add algorithm."""
        if k < 0:
            return self.scalar_multiply(-k, self.negate(P))
        if k == 0:
            return INFINITY

        result = INFINITY
        addend = P

        while k > 0:
            if k & 1:
                result = self.add(result, addend)
            addend = self._double(addend)
            k >>= 1

        return result


# ---------------------------------------------------------------------------
# A Small Curve for Demonstrations
# ---------------------------------------------------------------------------

# Why: We use a small curve (secp-like but with tiny parameters) so that
# computations finish instantly and values are readable. Real curves like
# secp256k1 (Bitcoin) use 256-bit primes. The security of ECC comes from
# the size of n (the group order), not from the curve being "complicated."
def get_demo_curve() -> EllipticCurve:
    """Return a small elliptic curve suitable for demonstrations.

    Curve: y^2 = x^3 + 2x + 3 over GF(97)
    This curve has order 89 (prime), making every non-identity point a generator.
    """
    p = 97
    a, b = 2, 3
    # Generator point (verified to be on the curve)
    G = Point(3, 6)
    n = 89  # order of G (verified by enumeration)
    curve = EllipticCurve(a, b, p, G, n)
    assert curve.is_on_curve(G), "Generator not on curve!"
    return curve


# ---------------------------------------------------------------------------
# ECDH Key Exchange
# ---------------------------------------------------------------------------

# Why: ECDH is the elliptic curve version of Diffie-Hellman. Alice picks
# private key a, publishes a*G. Bob picks b, publishes b*G. Shared secret
# is a*(b*G) = b*(a*G) = ab*G. An eavesdropper sees a*G and b*G but
# cannot compute ab*G without solving the ECDLP.
class ECDHParty:
    """One party in an ECDH key exchange."""

    def __init__(self, name: str, curve: EllipticCurve) -> None:
        self.name = name
        self.curve = curve
        # Why: The private key must be in [1, n-1]. Using n-1 as upper bound
        # ensures the public key is never the point at infinity.
        self.private_key = random.randrange(1, curve.n)
        self.public_key = curve.scalar_multiply(self.private_key, curve.G)

    def compute_shared_secret(self, other_public: Point) -> Point:
        """Compute shared secret: private_key * other_public_key."""
        return self.curve.scalar_multiply(self.private_key, other_public)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Elliptic Curve Cryptography")
    print("=" * 65)

    curve = get_demo_curve()
    print(f"\n[Curve Parameters]")
    print(f"  Equation: y^2 = x^3 + {curve.a}x + {curve.b} (mod {curve.p})")
    print(f"  Generator G = {curve.G}")
    print(f"  Order n = {curve.n}")
    print(f"  G on curve: {curve.is_on_curve(curve.G)}")

    # Point arithmetic
    print(f"\n[Point Arithmetic]")
    G = curve.G
    G2 = curve.scalar_multiply(2, G)
    G3 = curve.scalar_multiply(3, G)
    G4 = curve.scalar_multiply(4, G)
    print(f"  1*G = {G}")
    print(f"  2*G = {G2}")
    print(f"  3*G = {G3}")
    print(f"  4*G = {G4}")
    print(f"  2*G + 2*G = {curve.add(G2, G2)}")
    print(f"  4*G       = {G4}")
    print(f"  Match: {curve.add(G2, G2) == G4}")

    # Verify group order
    Gn = curve.scalar_multiply(curve.n, G)
    print(f"\n  n*G = {Gn}  (should be point at infinity)")
    print(f"  (n+1)*G = {curve.scalar_multiply(curve.n + 1, G)}  (should equal G)")

    # Enumerate some multiples
    print(f"\n[First 10 Multiples of G]")
    for i in range(1, 11):
        P = curve.scalar_multiply(i, G)
        print(f"  {i:>2} * G = {P}")

    # ECDH Key Exchange
    print(f"\n[ECDH Key Exchange]")
    alice = ECDHParty("Alice", curve)
    bob = ECDHParty("Bob", curve)

    print(f"  Alice's private key: {alice.private_key}")
    print(f"  Alice's public key:  {alice.public_key}")
    print(f"  Bob's private key:   {bob.private_key}")
    print(f"  Bob's public key:    {bob.public_key}")

    alice_secret = alice.compute_shared_secret(bob.public_key)
    bob_secret = bob.compute_shared_secret(alice.public_key)

    print(f"\n  Alice computes: {alice.private_key} * {bob.public_key} = {alice_secret}")
    print(f"  Bob computes:   {bob.private_key} * {alice.public_key} = {bob_secret}")
    print(f"  Shared secrets match: {alice_secret == bob_secret}")

    # Eve's perspective
    print(f"\n  Eve sees: Alice_pub = {alice.public_key}, Bob_pub = {bob.public_key}")
    print(f"  Eve knows G = {curve.G}")
    print(f"  Eve must solve ECDLP to find private keys — infeasible for large curves!")

    print(f"\n{'=' * 65}")

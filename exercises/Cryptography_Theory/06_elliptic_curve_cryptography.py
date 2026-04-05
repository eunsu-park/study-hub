"""
Exercises for Lesson 06: Elliptic Curve Cryptography
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
import time
from math import gcd, isqrt


class EllipticCurveFiniteField:
    """Elliptic curve over a prime finite field F_p.

    Curve equation: y^2 = x^3 + ax + b (mod p)
    """

    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
        assert (4 * a**3 + 27 * b**2) % p != 0, "Singular curve!"

    def is_on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - (x * x * x + self.a * x + self.b)) % self.p == 0

    def add(self, P, Q):
        """Point addition over F_p."""
        if P is None: return Q
        if Q is None: return P

        x1, y1 = P
        x2, y2 = Q
        p = self.p

        if x1 == x2 and y1 == (p - y2) % p:
            return None  # P + (-P) = O

        if x1 == x2 and y1 == y2:
            if y1 == 0:
                return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, p) % p
        else:
            lam = (y2 - y1) * pow(x2 - x1, -1, p) % p

        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def negate(self, P):
        if P is None: return None
        return (P[0], (-P[1]) % self.p)

    def scalar_multiply(self, k, P):
        """Double-and-add scalar multiplication."""
        if k == 0 or P is None:
            return None
        if k < 0:
            return self.scalar_multiply(-k, self.negate(P))

        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def count_points(self):
        """Brute force point count (small p only)."""
        count = 1  # Point at infinity
        for x in range(self.p):
            rhs = (x**3 + self.a * x + self.b) % self.p
            if rhs == 0:
                count += 1
            elif pow(rhs, (self.p - 1) // 2, self.p) == 1:
                count += 2
        return count

    def find_all_points(self):
        """Find all points on the curve (small p only)."""
        points = [None]  # Include point at infinity
        for x in range(self.p):
            rhs = (x**3 + self.a * x + self.b) % self.p
            if rhs == 0:
                points.append((x, 0))
            elif pow(rhs, (self.p - 1) // 2, self.p) == 1:
                # Find square root using Tonelli-Shanks (simplified for odd p)
                y = pow(rhs, (self.p + 1) // 4, self.p)
                if (y * y) % self.p == rhs:
                    points.append((x, y))
                    points.append((x, (self.p - y) % self.p))
                else:
                    # Fallback: brute force sqrt
                    for y in range(self.p):
                        if (y * y) % self.p == rhs:
                            points.append((x, y))
                            points.append((x, (self.p - y) % self.p))
                            break
        return points


class TwistedEdwardsCurve:
    """Twisted Edwards curve: ax^2 + y^2 = 1 + dx^2y^2 over F_p."""

    def __init__(self, a, d, p):
        self.a = a
        self.d = d
        self.p = p

    def is_on_curve(self, P):
        if P is None:
            return True
        x, y = P
        lhs = (self.a * x * x + y * y) % self.p
        rhs = (1 + self.d * x * x * y * y) % self.p
        return lhs == rhs

    def add(self, P, Q):
        """Complete addition formula -- no special cases."""
        if P is None: return Q
        if Q is None: return P

        x1, y1 = P
        x2, y2 = Q
        p = self.p

        x_num = (x1 * y2 + x2 * y1) % p
        y_num = (y1 * y2 - self.a * x1 * x2) % p

        common = (self.d * x1 * x2 * y1 * y2) % p
        x_den = (1 + common) % p
        y_den = (1 - common) % p

        x3 = (x_num * pow(x_den, -1, p)) % p
        y3 = (y_num * pow(y_den, -1, p)) % p
        return (x3, y3)

    def scalar_multiply(self, k, P):
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result


def exercise_1():
    """Exercise 1: Point Arithmetic (Basic)

    On the curve y^2 = x^3 + 2x + 3 over F_97:
    1. Verify that P = (3, 6) is on the curve
    2. Compute 2P (point doubling)
    3. Compute 3P = 2P + P
    4. Find the order of P (smallest k such that kP = O)
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    P = (3, 6)

    # Step 1: Verify P is on the curve
    # y^2 = 6^2 = 36
    # x^3 + 2x + 3 = 27 + 6 + 3 = 36
    lhs = (P[1] ** 2) % 97
    rhs = (P[0] ** 3 + 2 * P[0] + 3) % 97
    print(f"  Step 1: Verify P = {P} is on y^2 = x^3 + 2x + 3 (mod 97)")
    print(f"    y^2 mod 97 = {lhs}")
    print(f"    x^3 + 2x + 3 mod 97 = {rhs}")
    print(f"    On curve: {curve.is_on_curve(P)}")

    # Step 2: Compute 2P
    # Point doubling: lambda = (3*x1^2 + a) / (2*y1) mod p
    # lambda = (3*9 + 2) / (2*6) = 29 / 12 mod 97
    # 12^(-1) mod 97: pow(12, -1, 97) = 89 (since 12*89 = 1068 = 11*97 + 1)
    # lambda = 29 * 89 mod 97 = 2581 mod 97 = 2581 - 26*97 = 2581 - 2522 = 59
    two_P = curve.add(P, P)
    print(f"\n  Step 2: Point doubling 2P")
    print(f"    lambda = (3*{P[0]}^2 + 2) * (2*{P[1]})^(-1) mod 97")
    lam_num = (3 * P[0]**2 + 2) % 97
    lam_den_inv = pow(2 * P[1], -1, 97)
    lam = (lam_num * lam_den_inv) % 97
    print(f"    lambda = {lam_num} * {lam_den_inv} mod 97 = {lam}")
    print(f"    2P = {two_P}")
    print(f"    On curve: {curve.is_on_curve(two_P)}")

    # Step 3: Compute 3P
    three_P = curve.add(two_P, P)
    print(f"\n  Step 3: 3P = 2P + P")
    print(f"    3P = {three_P}")
    print(f"    On curve: {curve.is_on_curve(three_P)}")

    # Step 4: Find order of P
    print(f"\n  Step 4: Find order of P")
    Q = P
    order = 1
    while Q is not None:
        Q = curve.add(Q, P)
        order += 1
        if order > 200:
            print(f"    Order > 200, stopping search")
            break
    print(f"    Order of P = {order}")
    print(f"    Verify: {order}P = {curve.scalar_multiply(order, P)}")

    # Also show the total number of curve points and check divisibility
    n_points = curve.count_points()
    print(f"    Total curve points (including O): {n_points}")
    print(f"    ord(P) divides #E: {n_points % order == 0}")


def exercise_2():
    """Exercise 2: ECDH Implementation (Intermediate)

    1. Implement ECDH key exchange
    2. Have Alice and Bob generate key pairs and compute shared secrets
    3. Verify the shared secrets match
    4. Show eavesdropper cannot compute shared secret without solving ECDLP
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    print(f"  ECDH Key Exchange")
    print(f"    Curve: y^2 = x^3 + 2x + 3 (mod 97)")
    print(f"    Generator G = {G}")
    print(f"    Group order n = {n}")

    # Step 1: Key generation
    # Alice picks random private key, computes public key
    d_alice = random.randrange(1, n)
    Q_alice = curve.scalar_multiply(d_alice, G)

    # Bob picks random private key, computes public key
    d_bob = random.randrange(1, n)
    Q_bob = curve.scalar_multiply(d_bob, G)

    print(f"\n    Alice: private key d_A = {d_alice}, public key Q_A = {Q_alice}")
    print(f"    Bob:   private key d_B = {d_bob}, public key Q_B = {Q_bob}")

    # Step 2: Shared secret computation
    # Alice computes S = d_A * Q_B
    S_alice = curve.scalar_multiply(d_alice, Q_bob)
    # Bob computes S = d_B * Q_A
    S_bob = curve.scalar_multiply(d_bob, Q_alice)

    print(f"\n    Alice computes: d_A * Q_B = {d_alice} * {Q_bob} = {S_alice}")
    print(f"    Bob computes:   d_B * Q_A = {d_bob} * {Q_alice} = {S_bob}")
    print(f"    Shared secrets match: {S_alice == S_bob}")

    # Mathematical proof: S_alice = d_A * (d_B * G) = (d_A * d_B) * G
    #                      S_bob   = d_B * (d_A * G) = (d_B * d_A) * G
    # Since scalar multiplication is commutative: d_A * d_B = d_B * d_A

    # Step 3: Eavesdropper's perspective
    print(f"\n    Eavesdropper sees:")
    print(f"      G = {G}")
    print(f"      Q_A = {Q_alice}")
    print(f"      Q_B = {Q_bob}")
    print(f"    To compute S, the eavesdropper must solve:")
    print(f"      Find d_A such that d_A * G = Q_A  (ECDLP)")
    print(f"    For this small curve, brute force is possible:")

    # Brute force ECDLP (only feasible for small curves)
    for k in range(1, n + 1):
        if curve.scalar_multiply(k, G) == Q_alice:
            print(f"      Brute force found d_A = {k} (matches: {k == d_alice})")
            break

    print(f"    For a 256-bit curve, brute force requires ~2^128 operations")
    print(f"    (Pollard's rho), which is computationally infeasible.")


def exercise_3():
    """Exercise 3: Curve Point Counting (Intermediate)

    1. Count all points on y^2 = x^3 + x + 1 over F_p for p = 5, 7, 11, 13, 17, 19, 23
    2. Verify Hasse's bound for each
    3. Analyze the relationship between #E and p
    """
    primes = [5, 7, 11, 13, 17, 19, 23]
    a, b = 1, 1  # y^2 = x^3 + x + 1

    print(f"  Point Counting: y^2 = x^3 + x + 1 over F_p")
    print(f"  {'p':>4} {'#E':>5} {'p+1':>5} {'|#E-(p+1)|':>12} {'2*sqrt(p)':>10} {'Hasse OK':>10}")
    print(f"  {'-'*4} {'-'*5} {'-'*5} {'-'*12} {'-'*10} {'-'*10}")

    counts = []
    for p in primes:
        # Check that the discriminant is non-zero
        disc = (4 * a**3 + 27 * b**2) % p
        if disc == 0:
            print(f"  {p:>4}  Singular curve (discriminant = 0), skipping")
            continue

        curve = EllipticCurveFiniteField(a=a, b=b, p=p)
        n_points = curve.count_points()
        counts.append((p, n_points))

        hasse_diff = abs(n_points - (p + 1))
        hasse_bound = 2 * isqrt(p)
        # More precise: 2*sqrt(p)
        import math
        hasse_bound_precise = 2 * math.sqrt(p)
        hasse_ok = hasse_diff <= hasse_bound_precise

        print(f"  {p:>4} {n_points:>5} {p+1:>5} {hasse_diff:>12} {hasse_bound_precise:>10.2f} {'Yes':>10 if hasse_ok else 'No':>10}")

    # Analysis
    print(f"\n  Analysis:")
    print(f"    Hasse's theorem: |#E - (p+1)| <= 2*sqrt(p)")
    print(f"    This means #E is approximately p+1, with deviation bounded by 2*sqrt(p).")
    print(f"    As p grows, the relative deviation shrinks (2*sqrt(p)/p -> 0).")
    print(f"    The relationship #E vs p is roughly linear (#E ~ p).")

    # Show the ratio
    print(f"\n    Ratios #E / (p+1):")
    for p, n in counts:
        print(f"      p={p:>3}: #E/(p+1) = {n/(p+1):.3f}")


def exercise_4():
    """Exercise 4: Edwards vs Weierstrass (Challenging)

    1. Implement both Weierstrass and twisted Edwards point addition
    2. Count conditional branches in each
    3. Time 10,000 scalar multiplications on each form
    4. Explain why Edwards curves are preferred for constant-time
    """
    # Use a prime where we can find both Weierstrass and Edwards curve points
    p = 101

    # Weierstrass curve: y^2 = x^3 + 2x + 3 (mod 101)
    w_curve = EllipticCurveFiniteField(a=2, b=3, p=p)

    # Twisted Edwards curve: ax^2 + y^2 = 1 + dx^2y^2 (mod 101)
    # a = -1 (= 100 mod 101), d = 3
    e_a = p - 1  # -1 mod p
    e_d = 3
    e_curve = TwistedEdwardsCurve(a=e_a, d=e_d, p=p)

    # Find a valid point on each curve
    w_P = (3, 6) if w_curve.is_on_curve((3, 6)) else None
    if w_P is None:
        for x in range(p):
            rhs = (x**3 + 2*x + 3) % p
            if pow(rhs, (p-1)//2, p) == 1:
                y = pow(rhs, (p+1)//4, p)
                if (y*y) % p == rhs:
                    w_P = (x, y)
                    break

    e_P = None
    for x in range(p):
        for y in range(p):
            if e_curve.is_on_curve((x, y)) and (x, y) != (0, 1):
                e_P = (x, y)
                break
        if e_P:
            break

    print(f"  Edwards vs Weierstrass Comparison")
    print(f"  Field: F_{p}")

    # Part 1: Branch count analysis
    print(f"\n  Part 1: Conditional branches in point addition")
    print(f"    Weierstrass add():")
    print(f"      - P is None? (1 branch)")
    print(f"      - Q is None? (1 branch)")
    print(f"      - P == -Q? (1 branch)")
    print(f"      - P == Q? (1 branch, for doubling vs addition)")
    print(f"      - y1 == 0 in doubling? (1 branch)")
    print(f"      Total: 5 conditional branches")
    print(f"    Edwards add():")
    print(f"      - P is None? (1 branch)")
    print(f"      - Q is None? (1 branch)")
    print(f"      Total: 2 branches (only for identity element)")
    print(f"    The complete addition formula eliminates 3 branches!")

    # Part 2: Timing comparison
    print(f"\n  Part 2: Timing 10,000 scalar multiplications")

    if w_P and w_curve.is_on_curve(w_P):
        start = time.time()
        for _ in range(10000):
            w_curve.scalar_multiply(42, w_P)
        w_time = time.time() - start
        print(f"    Weierstrass: {w_time:.3f}s ({w_time/10000*1e6:.1f} us/op)")
    else:
        print(f"    Weierstrass: no valid point found")
        w_time = float('inf')

    if e_P and e_curve.is_on_curve(e_P):
        start = time.time()
        for _ in range(10000):
            e_curve.scalar_multiply(42, e_P)
        e_time = time.time() - start
        print(f"    Edwards:     {e_time:.3f}s ({e_time/10000*1e6:.1f} us/op)")
    else:
        print(f"    Edwards: no valid point found")
        e_time = float('inf')

    if w_time != float('inf') and e_time != float('inf'):
        if w_time < e_time:
            print(f"    Weierstrass is {e_time/w_time:.2f}x faster in this test")
        else:
            print(f"    Edwards is {w_time/e_time:.2f}x faster in this test")

    # Part 3: Why Edwards is preferred
    print(f"\n  Part 3: Why Edwards curves are preferred for constant-time")
    print(f"    1. COMPLETE addition formula: works for ALL point pairs")
    print(f"       - No special cases for P=Q, P=-Q, P=O")
    print(f"       - This means no data-dependent branches")
    print(f"    2. Side-channel resistance:")
    print(f"       - Each branch is a potential timing leak")
    print(f"       - An attacker measuring execution time can determine")
    print(f"         which branch was taken, revealing secret key bits")
    print(f"    3. Simpler code = fewer implementation bugs")
    print(f"    4. Ed25519 (twisted Edwards) is the most widely deployed")
    print(f"       signature curve (SSH, Signal, WireGuard, Tor)")


def exercise_5():
    """Exercise 5: Security Level Analysis (Challenging)

    1. For P-256, P-384, P-521, Curve25519:
       - State group order n
       - Compute security level = floor(log2(sqrt(n)))
       - Compare with equivalent RSA key size
    2. Quantum computer impact analysis
    """
    import math

    curves = [
        {
            "name": "NIST P-256 (secp256r1)",
            "order": 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551,
            "rsa_equiv": 3072,
        },
        {
            "name": "NIST P-384 (secp384r1)",
            "order": 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973,
            "rsa_equiv": 7680,
        },
        {
            "name": "NIST P-521 (secp521r1)",
            "order": 0x01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6FB71E91386409,
            "rsa_equiv": 15360,
        },
        {
            "name": "Curve25519 / X25519",
            # Order of the prime-order subgroup (cofactor 8)
            "order": 2**252 + 27742317777372353535851937790883648493,
            "rsa_equiv": 3072,
        },
    ]

    print(f"  Security Level Analysis")
    print(f"  {'Curve':<25} {'Order bits':>10} {'Security':>10} {'RSA equiv':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for c in curves:
        n = c["order"]
        n_bits = n.bit_length()
        # Security level = floor(log2(sqrt(n)))
        # Pollard's rho: O(sqrt(n)) operations
        security_bits = n_bits // 2
        rsa = c["rsa_equiv"]
        print(f"  {c['name']:<25} {n_bits:>10} {security_bits:>10} {rsa:>10}")

    print(f"\n  Key insight: ECC achieves equivalent security with much smaller keys.")
    print(f"  P-256 (256 bits) ~ RSA-3072 (3072 bits) = 12x smaller keys")
    print(f"  P-521 (521 bits) ~ RSA-15360 (15360 bits) = 30x smaller keys")

    # Part 2: Quantum computer impact
    print(f"\n  Quantum Computer Impact")
    print(f"  {'='*55}")
    print(f"  Shor's algorithm breaks ECDLP in polynomial time.")
    print(f"  Required qubits: approximately 6n logical qubits for n-bit curve")
    print(f"  (with error correction, physical qubits ~1000x more)")
    print(f"")

    for c in curves:
        n_bits = c["order"].bit_length()
        logical_qubits = 6 * n_bits  # Rough estimate for ECDLP
        physical_qubits = logical_qubits * 1000  # Error correction overhead
        print(f"  {c['name']:<25}")
        print(f"    Logical qubits needed:  ~{logical_qubits}")
        print(f"    Physical qubits needed: ~{physical_qubits:,}")

    print(f"\n  If 4,000 logical qubits become available:")
    for c in curves:
        n_bits = c["order"].bit_length()
        logical_needed = 6 * n_bits
        secure = logical_needed > 4000
        status = "STILL SECURE" if secure else "BROKEN"
        print(f"    {c['name']:<25}: needs {logical_needed} qubits -> {status}")

    print(f"\n  Conclusion: With 4,000 logical qubits, ALL these curves are broken.")
    print(f"  P-256 needs ~1,536 logical qubits (broken first)")
    print(f"  P-521 needs ~3,126 logical qubits (also broken)")
    print(f"  Migration to lattice-based crypto (Kyber, Dilithium) is essential.")


if __name__ == "__main__":
    print("=== Exercise 1: Point Arithmetic ===")
    exercise_1()

    print("\n=== Exercise 2: ECDH Implementation ===")
    exercise_2()

    print("\n=== Exercise 3: Curve Point Counting ===")
    exercise_3()

    print("\n=== Exercise 4: Edwards vs Weierstrass ===")
    exercise_4()

    print("\n=== Exercise 5: Security Level Analysis ===")
    exercise_5()

    print("\nAll exercises completed!")

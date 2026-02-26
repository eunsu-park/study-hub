# Lesson 6: Elliptic Curve Cryptography

**Previous**: [RSA Cryptosystem](./05_RSA_Cryptosystem.md) | **Next**: [Digital Signatures](./07_Digital_Signatures.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the geometric intuition behind elliptic curve point addition over the reals
2. Define elliptic curves over finite fields and perform point addition algebraically
3. Implement scalar multiplication using the double-and-add algorithm
4. Explain why the Elliptic Curve Discrete Logarithm Problem (ECDLP) is hard
5. Compare ECC and RSA in terms of key sizes, performance, and security levels
6. Identify the most important named curves (secp256k1, Curve25519, Ed25519) and their design philosophies

---

Elliptic Curve Cryptography (ECC) achieves the same security as RSA with dramatically smaller keys — a 256-bit ECC key provides security equivalent to a 3072-bit RSA key. This means faster computations, less bandwidth, and smaller certificates. Since its introduction by Koblitz and Miller in 1985, ECC has become the dominant public-key algorithm in TLS 1.3, SSH, Bitcoin, and Signal. This lesson builds the mathematical machinery from geometric intuition through finite field arithmetic to practical key generation.

> **Analogy:** Point addition on an elliptic curve is like a billiard ball bouncing off the curve — draw a line through two points, find where it hits the curve again, and reflect. Easy to compute the next point, but given a starting point and an ending point, tracing backward through thousands of bounces is nearly impossible. This one-way property is the ECDLP.

## Table of Contents

1. [Elliptic Curves Over the Reals](#1-elliptic-curves-over-the-reals)
2. [The Group Law: Point Addition](#2-the-group-law-point-addition)
3. [Elliptic Curves Over Finite Fields](#3-elliptic-curves-over-finite-fields)
4. [Scalar Multiplication](#4-scalar-multiplication)
5. [The Elliptic Curve Discrete Logarithm Problem](#5-the-elliptic-curve-discrete-logarithm-problem)
6. [Named Curves and Standards](#6-named-curves-and-standards)
7. [Montgomery and Edwards Curves](#7-montgomery-and-edwards-curves)
8. [ECC Key Pairs and ECDH](#8-ecc-key-pairs-and-ecdh)
9. [ECC vs RSA Comparison](#9-ecc-vs-rsa-comparison)
10. [Exercises](#10-exercises)

---

## 1. Elliptic Curves Over the Reals

### 1.1 The Weierstrass Equation

An elliptic curve in **short Weierstrass form** is defined by:

$$y^2 = x^3 + ax + b$$

where the **discriminant** $\Delta = -16(4a^3 + 27b^2) \neq 0$ (ensures no singular points — no cusps or self-intersections).

### 1.2 Visualization

```python
import numpy as np

def plot_elliptic_curve_text(a, b, x_range=(-3, 3), resolution=40):
    """Text-based visualization of an elliptic curve y^2 = x^3 + ax + b.

    Why visualize: the geometric interpretation makes the group law
    intuitive before we move to finite fields where visualization
    is impossible.
    """
    disc = -16 * (4 * a**3 + 27 * b**2)
    print(f"Curve: y^2 = x^3 + {a}x + {b}")
    print(f"Discriminant: {disc} {'(valid)' if disc != 0 else '(SINGULAR!)'}")
    print()

    # Compute points on the curve
    points = []
    for x in np.linspace(x_range[0], x_range[1], 1000):
        rhs = x**3 + a*x + b
        if rhs >= 0:
            y = np.sqrt(rhs)
            points.append((x, y))
            points.append((x, -y))

    if not points:
        print("No real points in range")
        return

    # Print key properties
    print(f"Number of real points plotted: {len(points)}")
    print(f"Curve is symmetric about x-axis (if (x,y) is on curve, so is (x,-y))")
    print()

    # Some sample points
    for x_test in [0, 1, 2]:
        rhs = x_test**3 + a*x_test + b
        if rhs >= 0:
            y_test = np.sqrt(rhs)
            print(f"  Point: ({x_test}, {y_test:.4f})")
            print(f"  Verify: {y_test**2:.4f} = {x_test**3 + a*x_test + b:.4f}")

# Common curves
plot_elliptic_curve_text(a=-1, b=1)   # y^2 = x^3 - x + 1
print()
plot_elliptic_curve_text(a=0, b=7)    # secp256k1 (Bitcoin)
```

### 1.3 The Point at Infinity

The curve includes a special **point at infinity** $\mathcal{O}$, which serves as the **identity element** of the group. Geometrically, it is the point where all vertical lines meet.

---

## 2. The Group Law: Point Addition

### 2.1 Geometric Construction

Given two points $P = (x_1, y_1)$ and $Q = (x_2, y_2)$ on the curve:

**Point Addition ($P \neq Q$):**
1. Draw the line through $P$ and $Q$
2. This line intersects the curve at a third point $R'$
3. Reflect $R'$ across the x-axis to get $R = P + Q$

**Point Doubling ($P = Q$):**
1. Draw the tangent line at $P$
2. This line intersects the curve at a second point $R'$
3. Reflect $R'$ across the x-axis to get $R = 2P$

### 2.2 Algebraic Formulas

For $P = (x_1, y_1)$ and $Q = (x_2, y_2)$:

**Addition ($P \neq Q$):**

$$\lambda = \frac{y_2 - y_1}{x_2 - x_1}$$

**Doubling ($P = Q$):**

$$\lambda = \frac{3x_1^2 + a}{2y_1}$$

**Result point:**

$$x_3 = \lambda^2 - x_1 - x_2$$
$$y_3 = \lambda(x_1 - x_3) - y_1$$

**Special cases:**
- $P + \mathcal{O} = P$ (identity)
- $P + (-P) = \mathcal{O}$ where $-P = (x_1, -y_1)$ (inverse)

```python
class EllipticCurveReal:
    """Elliptic curve over the real numbers (for geometric intuition).

    Why start with reals: the geometry of point addition is visible
    and intuitive over R. Once you understand the geometry, the
    algebraic formulas carry directly to finite fields.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        assert -16 * (4 * a**3 + 27 * b**2) != 0, "Singular curve!"

    def is_on_curve(self, P):
        if P is None:  # Point at infinity
            return True
        x, y = P
        return abs(y**2 - (x**3 + self.a * x + self.b)) < 1e-10

    def add(self, P, Q):
        """Add two points on the curve.

        Why reflection across x-axis: without it, the third intersection
        point would not form a group (associativity would fail). The
        reflection is what makes (E, +) an abelian group.
        """
        if P is None: return Q
        if Q is None: return P

        x1, y1 = P
        x2, y2 = Q

        if abs(x1 - x2) < 1e-10 and abs(y1 + y2) < 1e-10:
            return None  # P + (-P) = O

        if abs(x1 - x2) < 1e-10 and abs(y1 - y2) < 1e-10:
            # Point doubling
            if abs(y1) < 1e-10:
                return None  # Tangent is vertical
            lam = (3 * x1**2 + self.a) / (2 * y1)
        else:
            # Point addition
            lam = (y2 - y1) / (x2 - x1)

        x3 = lam**2 - x1 - x2
        y3 = lam * (x1 - x3) - y1

        return (x3, y3)

    def negate(self, P):
        if P is None: return None
        return (P[0], -P[1])

# Demonstrate point addition over reals
curve = EllipticCurveReal(a=-1, b=1)
P = (0, 1)  # On curve: 1 = 0 + 0 + 1 ✓
Q = (1, 1)  # On curve: 1 = 1 - 1 + 1 ✓

print(f"P = {P}, on curve: {curve.is_on_curve(P)}")
print(f"Q = {Q}, on curve: {curve.is_on_curve(Q)}")

R = curve.add(P, Q)
print(f"P + Q = ({R[0]:.6f}, {R[1]:.6f}), on curve: {curve.is_on_curve(R)}")

# Point doubling
D = curve.add(P, P)
print(f"2P = ({D[0]:.6f}, {D[1]:.6f}), on curve: {curve.is_on_curve(D)}")
```

### 2.3 Group Properties

The set of points on an elliptic curve, together with point addition, forms an **abelian group**:

| Property | Meaning |
|----------|---------|
| Closure | $P + Q$ is on the curve |
| Associativity | $(P + Q) + R = P + (Q + R)$ |
| Identity | $P + \mathcal{O} = P$ |
| Inverse | $P + (-P) = \mathcal{O}$ |
| Commutativity | $P + Q = Q + P$ |

---

## 3. Elliptic Curves Over Finite Fields

### 3.1 From Reals to $\mathbb{F}_p$

For cryptography, we work over a **prime finite field** $\mathbb{F}_p$ where $p$ is a large prime. The curve equation becomes:

$$y^2 \equiv x^3 + ax + b \pmod{p}$$

All arithmetic (addition, subtraction, multiplication, division) is performed modulo $p$. Division uses modular inverses (Lesson 1, Section 3).

### 3.2 Key Differences from Reals

| Property | Over $\mathbb{R}$ | Over $\mathbb{F}_p$ |
|----------|-------------------|---------------------|
| Points | Continuous curve | Discrete set of $(x, y)$ pairs |
| Number of points | Infinite | Finite ($\approx p$ by Hasse's theorem) |
| Visualization | Smooth curve | Scattered dots |
| Division | Normal division | Modular inverse |
| Square roots | $\sqrt{\cdot}$ | Tonelli-Shanks algorithm |

### 3.3 Hasse's Theorem

The number of points on $E(\mathbb{F}_p)$, denoted $\#E(\mathbb{F}_p)$, satisfies:

$$|\ \#E(\mathbb{F}_p) - (p + 1)\ | \leq 2\sqrt{p}$$

So the number of points is approximately $p$.

```python
class EllipticCurveFiniteField:
    """Elliptic curve over a prime finite field F_p.

    Why finite fields for crypto: finite fields have a fixed number
    of elements, making the discrete log problem well-defined. Over
    the reals, there would be no discrete log problem.
    """

    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
        # Verify discriminant is non-zero
        assert (4 * a**3 + 27 * b**2) % p != 0, "Singular curve!"

    def is_on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - (x * x * x + self.a * x + self.b)) % self.p == 0

    def add(self, P, Q):
        """Point addition over F_p.

        Why modular inverse instead of division: in F_p, there is
        no 'division' — we multiply by the modular inverse. This
        is computed using the Extended Euclidean Algorithm (L01).
        """
        if P is None: return Q
        if Q is None: return P

        x1, y1 = P
        x2, y2 = Q
        p = self.p

        if x1 == x2 and y1 == (p - y2) % p:
            return None  # P + (-P) = O

        if x1 == x2 and y1 == y2:
            # Point doubling
            if y1 == 0:
                return None
            # lambda = (3*x1^2 + a) * (2*y1)^(-1) mod p
            num = (3 * x1 * x1 + self.a) % p
            den = pow(2 * y1, -1, p)  # Modular inverse
            lam = (num * den) % p
        else:
            # Point addition
            num = (y2 - y1) % p
            den = pow(x2 - x1, -1, p)
            lam = (num * den) % p

        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p

        return (x3, y3)

    def negate(self, P):
        if P is None: return None
        return (P[0], (-P[1]) % self.p)

    def scalar_multiply(self, k, P):
        """Compute k*P using double-and-add (see Section 4)."""
        result = None  # Point at infinity
        addend = P

        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1

        return result

    def count_points(self):
        """Count all points on the curve (brute force — only for small p).

        Why brute force is okay for small p: for cryptographic curves
        (p ~ 2^256), we use Schoof's algorithm or precomputed values.
        """
        count = 1  # Point at infinity
        for x in range(self.p):
            rhs = (x**3 + self.a * x + self.b) % self.p
            # Check if rhs is a quadratic residue
            if rhs == 0:
                count += 1  # (x, 0)
            elif pow(rhs, (self.p - 1) // 2, self.p) == 1:
                count += 2  # (x, y) and (x, -y)
        return count

# Example: small curve for demonstration
curve = EllipticCurveFiniteField(a=2, b=3, p=97)

P = (3, 6)
assert curve.is_on_curve(P), "P not on curve!"
print(f"P = {P}, on curve: {curve.is_on_curve(P)}")

# Point addition
Q = curve.add(P, P)
print(f"2P = {Q}, on curve: {curve.is_on_curve(Q)}")

R = curve.add(Q, P)
print(f"3P = {R}, on curve: {curve.is_on_curve(R)}")

# Count points (Hasse bound: |#E - (p+1)| <= 2*sqrt(p))
n_points = curve.count_points()
import math
hasse_bound = 2 * math.isqrt(97)
print(f"\nNumber of points: {n_points}")
print(f"p + 1 = {98}, Hasse bound: {98} ± {hasse_bound}")
print(f"|{n_points} - 98| = {abs(n_points - 98)} <= {hasse_bound}: {abs(n_points - 98) <= hasse_bound}")
```

---

## 4. Scalar Multiplication

### 4.1 Definition

**Scalar multiplication** computes $kP = P + P + \cdots + P$ ($k$ times). This is the fundamental operation in ECC — it is the elliptic curve analog of modular exponentiation in RSA.

### 4.2 Double-and-Add Algorithm

Just as RSA uses square-and-multiply for $a^k \bmod n$ (Lesson 1, Section 7), ECC uses **double-and-add** for $kP$:

```python
def double_and_add(curve, k, P):
    """Scalar multiplication using double-and-add.

    Why this is O(log k): we process one bit of k per iteration,
    performing at most 2 group operations (one doubling + one addition)
    per bit. For a 256-bit scalar, this is at most 512 point operations.

    Compare: naive addition (P + P + ... + P) would require k-1
    additions — for k ~ 2^256, this is impossibly slow.
    """
    if k == 0:
        return None  # Point at infinity
    if k < 0:
        return double_and_add(curve, -k, curve.negate(P))

    result = None  # Accumulator, starts at O
    addend = P     # Current power of 2 times P

    bits_processed = 0
    while k:
        if k & 1:
            result = curve.add(result, addend)
        addend = curve.add(addend, addend)  # Double
        k >>= 1
        bits_processed += 1

    return result

# Verify: compute 10P step by step
curve = EllipticCurveFiniteField(a=2, b=3, p=97)
P = (3, 6)

# Method 1: double-and-add
R1 = double_and_add(curve, 10, P)

# Method 2: repeated addition
R2 = P
for _ in range(9):
    R2 = curve.add(R2, P)

print(f"10P (double-and-add): {R1}")
print(f"10P (repeated add):   {R2}")
print(f"Match: {R1 == R2}")
```

### 4.3 Side-Channel Resistance: Montgomery Ladder

The double-and-add algorithm leaks information through timing and power analysis (the `if k & 1` branch takes different time). The **Montgomery ladder** performs the same number of operations regardless of the bit value:

```python
def montgomery_ladder(curve, k, P):
    """Constant-time scalar multiplication using the Montgomery ladder.

    Why constant-time matters: in a side-channel attack, an attacker
    measures the time or power consumption of scalar multiplication
    to determine individual bits of the private key k. The Montgomery
    ladder ensures every iteration performs exactly one addition and
    one doubling, regardless of the bit value.
    """
    R0 = None  # Point at infinity
    R1 = P

    for i in range(k.bit_length() - 1, -1, -1):
        if (k >> i) & 1:
            R0 = curve.add(R0, R1)
            R1 = curve.add(R1, R1)
        else:
            R1 = curve.add(R0, R1)
            R0 = curve.add(R0, R0)

    return R0

# Verify
R = montgomery_ladder(curve, 10, P)
print(f"10P (Montgomery): {R}")
```

---

## 5. The Elliptic Curve Discrete Logarithm Problem

### 5.1 Definition

Given points $P$ and $Q = kP$ on an elliptic curve, find the scalar $k$.

$$\text{Given } P \text{ and } Q = kP, \text{ find } k$$

### 5.2 Why ECDLP Is Harder Than DLP

For the discrete log problem in $\mathbb{Z}_p^*$ (Lesson 1), sub-exponential algorithms exist (index calculus). For ECDLP, **no sub-exponential algorithm is known**:

| Problem | Best Algorithm | Complexity |
|---------|---------------|------------|
| DLP in $\mathbb{Z}_p^*$ | Index Calculus | $L_p(1/3)$ (sub-exponential) |
| ECDLP | Pollard's Rho | $O(\sqrt{n})$ (fully exponential) |
| Factoring (RSA) | GNFS | $L_n(1/3)$ (sub-exponential) |

This is why ECC achieves equivalent security with much smaller parameters.

### 5.3 Pollard's Rho for ECDLP

The best known algorithm for ECDLP on general curves:

```python
def pollard_rho_ecdlp(curve, P, Q, order):
    """Pollard's rho algorithm for solving ECDLP: find k such that Q = kP.

    Why O(sqrt(n)): the algorithm creates a pseudo-random walk on
    the group. By the birthday paradox, after ~sqrt(n) steps,
    we expect a collision, which yields the discrete log.

    For a 256-bit curve, sqrt(2^256) = 2^128 — this is computationally
    infeasible, which is why 256-bit ECC is considered secure.
    """
    import random

    def step(R, a, b):
        """Partition the group into 3 sets and define the walk."""
        x = R[0] if R else 0
        partition = x % 3

        if partition == 0:
            # R = R + P
            return curve.add(R, P), (a + 1) % order, b
        elif partition == 1:
            # R = 2R
            return curve.add(R, R), (2 * a) % order, (2 * b) % order
        else:
            # R = R + Q
            return curve.add(R, Q), a, (b + 1) % order

    # Floyd's cycle detection
    # Tortoise
    aT, bT = 1, 0
    T = curve.scalar_multiply(1, P)  # T = 1*P + 0*Q = P

    # Hare (moves twice as fast)
    aH, bH = 1, 0
    H = curve.scalar_multiply(1, P)

    for _ in range(order):
        # Tortoise: one step
        T, aT, bT = step(T, aT, bT)

        # Hare: two steps
        H, aH, bH = step(H, aH, bH)
        H, aH, bH = step(H, aH, bH)

        if T == H:
            # Collision: aT*P + bT*Q = aH*P + bH*Q
            # (aT - aH)*P = (bH - bT)*Q = (bH - bT)*k*P
            # So k = (aT - aH) / (bH - bT) mod order
            num = (aT - aH) % order
            den = (bH - bT) % order
            if den == 0:
                continue  # Degenerate collision, restart
            k = (num * pow(den, -1, order)) % order
            # Verify
            if curve.scalar_multiply(k, P) == Q:
                return k

    return None  # Should not reach here for correct inputs

# Demo on a small curve
curve = EllipticCurveFiniteField(a=2, b=3, p=97)
P = (3, 6)
k_secret = 15
Q = curve.scalar_multiply(k_secret, P)

order = curve.count_points()  # For small curves, brute-force order
print(f"P = {P}, Q = {k_secret}*P = {Q}")
print(f"Curve order: {order}")

k_found = pollard_rho_ecdlp(curve, P, Q, order)
if k_found is not None:
    print(f"Recovered k = {k_found} (correct: {k_found == k_secret})")
else:
    print("Failed to find k (try again — probabilistic algorithm)")
```

---

## 6. Named Curves and Standards

### 6.1 NIST Curves

| Curve | Field Size | Security Level | Usage |
|-------|-----------|----------------|-------|
| P-256 (secp256r1) | 256-bit | 128-bit | TLS, most common |
| P-384 (secp384r1) | 384-bit | 192-bit | Government, high security |
| P-521 (secp521r1) | 521-bit | 256-bit | Maximum security |

**Controversy:** The NIST curves use "random" parameters generated by the NSA. After the Dual_EC_DRBG backdoor scandal (2013), some researchers distrust NIST curves, preferring independently generated curves.

### 6.2 secp256k1 (Bitcoin)

$$y^2 = x^3 + 7 \quad \text{over } \mathbb{F}_p$$

where $p = 2^{256} - 2^{32} - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1$.

- Used by Bitcoin, Ethereum, and other cryptocurrencies
- Chosen for efficiency ($a = 0$ simplifies doubling formulas)
- Parameters are "verifiably random" (derived from simple formula)

### 6.3 Curve25519 and Ed25519

**Curve25519** (Daniel J. Bernstein, 2006):
- Montgomery curve: $y^2 = x^3 + 486662x^2 + x$ over $\mathbb{F}_{2^{255}-19}$
- Designed for Diffie-Hellman key exchange (X25519)
- Constant-time by design, resistant to timing attacks

**Ed25519** (Bernstein et al., 2011):
- Twisted Edwards curve birationally equivalent to Curve25519
- Designed for digital signatures (EdDSA)
- Deterministic signatures (no random nonce — prevents PS3 disaster, Lesson 7)
- Used in SSH, Signal, WireGuard, Tor

```python
def named_curves_info():
    """Summary of major named curves and their properties.

    Why different curves for different purposes:
    - NIST P-256: backward compatibility, government compliance
    - secp256k1: cryptocurrency ecosystem, verifiable parameters
    - Curve25519/Ed25519: modern design, constant-time, no trust issues
    """
    curves = [
        {
            "name": "NIST P-256 (secp256r1)",
            "form": "Short Weierstrass",
            "field": "2^256 - 2^224 + 2^192 + 2^96 - 1",
            "security": "128-bit",
            "usage": "TLS, PKI, government",
            "trust": "NSA-generated parameters (controversial)",
        },
        {
            "name": "secp256k1",
            "form": "Short Weierstrass (a=0, b=7)",
            "field": "2^256 - 2^32 - 977",
            "security": "128-bit",
            "usage": "Bitcoin, Ethereum",
            "trust": "Verifiably random (Koblitz curve)",
        },
        {
            "name": "Curve25519 / X25519",
            "form": "Montgomery",
            "field": "2^255 - 19",
            "security": "128-bit",
            "usage": "Key exchange (TLS, SSH, WireGuard)",
            "trust": "Independently designed, fully rigid",
        },
        {
            "name": "Ed25519",
            "form": "Twisted Edwards",
            "field": "2^255 - 19 (same as Curve25519)",
            "security": "128-bit",
            "usage": "Signatures (SSH, Signal, Tor)",
            "trust": "Independently designed, deterministic",
        },
    ]

    for c in curves:
        print(f"--- {c['name']} ---")
        for k, v in c.items():
            if k != 'name':
                print(f"  {k:>10}: {v}")
        print()

named_curves_info()
```

---

## 7. Montgomery and Edwards Curves

### 7.1 Montgomery Curves

A **Montgomery curve** has the form:

$$By^2 = x^3 + Ax^2 + x$$

**Key advantage:** The Montgomery ladder naturally uses only the $x$-coordinate, making scalar multiplication:
- Constant-time (no branches on secret bits)
- Efficient (no $y$-coordinate needed until the final step)
- Resistant to side-channel attacks by construction

### 7.2 Edwards Curves

A **twisted Edwards curve** has the form:

$$ax^2 + y^2 = 1 + dx^2y^2$$

**Key advantage:** The addition formula is **complete** — it works for all pairs of points without special cases (no checking for $P = Q$, $P = -Q$, etc.):

$$x_3 = \frac{x_1 y_2 + x_2 y_1}{1 + d x_1 x_2 y_1 y_2}, \quad y_3 = \frac{y_1 y_2 - a x_1 x_2}{1 - d x_1 x_2 y_1 y_2}$$

```python
class TwistedEdwardsCurve:
    """Twisted Edwards curve: ax^2 + y^2 = 1 + dx^2y^2 over F_p.

    Why Edwards curves are superior for implementations:
    1. Complete addition formula — no special cases, no branching
    2. Fastest known point addition among curve forms
    3. Naturally constant-time (no if/else on point coordinates)
    4. Ed25519 is the most widely deployed signature curve
    """

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
        """Complete addition formula — works for ALL point pairs.

        Why 'complete' matters: in Weierstrass form, the addition
        formula has special cases (P = Q, P = -Q, P = O) that
        require branching. Each branch is a potential side-channel
        leak. Edwards curves eliminate all special cases.
        """
        if P is None: return Q
        if Q is None: return P

        x1, y1 = P
        x2, y2 = Q
        p = self.p

        # Numerators
        x_num = (x1 * y2 + x2 * y1) % p
        y_num = (y1 * y2 - self.a * x1 * x2) % p

        # Denominators
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

# Example: small twisted Edwards curve
# a = -1, d = -121665/121666 mod p for Ed25519
# For demo, use small parameters
p = 101
a = p - 1  # -1 mod p = 100
d = 3

edwards = TwistedEdwardsCurve(a=a, d=d, p=p)

# Find a point on the curve
for x in range(p):
    for y in range(p):
        if edwards.is_on_curve((x, y)):
            P = (x, y)
            if P != (0, 1):  # Skip identity
                break
    else:
        continue
    break

print(f"Curve: {a}x^2 + y^2 = 1 + {d}x^2y^2 (mod {p})")
print(f"Point P = {P}, on curve: {edwards.is_on_curve(P)}")

Q = edwards.add(P, P)
print(f"2P = {Q}, on curve: {edwards.is_on_curve(Q)}")

R = edwards.scalar_multiply(5, P)
print(f"5P = {R}, on curve: {edwards.is_on_curve(R)}")
```

### 7.3 Birational Equivalence

Montgomery, Edwards, and Weierstrass forms are **birationally equivalent** — there exist rational maps between them. Curve25519 (Montgomery) and Ed25519 (Edwards) represent the same underlying group; the different forms are chosen for different computational advantages.

---

## 8. ECC Key Pairs and ECDH

### 8.1 Key Generation

1. Choose a standardized curve $E$ with base point $G$ and order $n$
2. Choose a random private key $d \in \{1, 2, \ldots, n-1\}$
3. Compute the public key $Q = dG$

```python
def ecc_keygen(curve, G, order):
    """Generate an ECC key pair.

    Why this is secure: recovering d from Q = dG requires solving
    the ECDLP, which takes O(sqrt(n)) ≈ 2^128 operations for a
    256-bit curve. This is computationally infeasible.
    """
    import random
    d = random.randrange(1, order)
    Q = curve.scalar_multiply(d, G)
    return d, Q  # (private_key, public_key)
```

### 8.2 Elliptic Curve Diffie-Hellman (ECDH)

ECDH allows two parties to establish a shared secret over an insecure channel:

1. Alice: private key $d_A$, public key $Q_A = d_A G$
2. Bob: private key $d_B$, public key $Q_B = d_B G$
3. Shared secret: $S = d_A Q_B = d_A (d_B G) = d_B (d_A G) = d_B Q_A$

```python
def ecdh_demo(curve, G, order):
    """Demonstrate Elliptic Curve Diffie-Hellman key exchange.

    Why ECDH over classical DH: same security with smaller parameters.
    A 256-bit ECC key exchange is as secure as a 3072-bit classical
    DH exchange, but uses ~12x less bandwidth.
    """
    # Alice's key pair
    d_alice, Q_alice = ecc_keygen(curve, G, order)

    # Bob's key pair
    d_bob, Q_bob = ecc_keygen(curve, G, order)

    # Shared secret computation
    S_alice = curve.scalar_multiply(d_alice, Q_bob)  # Alice computes
    S_bob = curve.scalar_multiply(d_bob, Q_alice)     # Bob computes

    print(f"Alice's public key: {Q_alice}")
    print(f"Bob's public key:   {Q_bob}")
    print(f"Alice's shared secret: {S_alice}")
    print(f"Bob's shared secret:   {S_bob}")
    print(f"Secrets match: {S_alice == S_bob}")

    # An eavesdropper knows G, Q_alice, Q_bob but cannot compute S
    # without solving ECDLP

# Demo with our small curve
curve = EllipticCurveFiniteField(a=2, b=3, p=97)
G = (3, 6)
order = curve.count_points()
ecdh_demo(curve, G, order)
```

---

## 9. ECC vs RSA Comparison

### 9.1 Key Size Comparison

| Security (bits) | RSA key | ECC key | RSA/ECC ratio |
|-----------------|---------|---------|---------------|
| 80 | 1024 | 160 | 6.4x |
| 112 | 2048 | 224 | 9.1x |
| 128 | 3072 | 256 | 12x |
| 192 | 7680 | 384 | 20x |
| 256 | 15360 | 521 | 29.5x |

### 9.2 Performance Comparison

| Operation | RSA-2048 | ECDSA P-256 | Winner |
|-----------|----------|-------------|--------|
| Key generation | ~100ms | ~1ms | ECC (100x) |
| Signing | ~2ms | ~1ms | ECC (2x) |
| Verification | ~0.1ms | ~2ms | RSA (20x) |
| Key size | 256 bytes | 32 bytes | ECC (8x) |

RSA verification is faster because $e = 65537$ is small, but ECC wins on all other metrics.

### 9.3 When to Use Each

| Use Case | Recommendation | Reason |
|----------|---------------|--------|
| TLS 1.3 | ECDHE + ECDSA | Default, fastest handshake |
| SSH keys | Ed25519 | Smallest keys, fastest |
| Code signing | ECDSA or RSA | Verification-heavy → RSA OK |
| Bitcoin | secp256k1 | Ecosystem standardized |
| Legacy systems | RSA-2048 | Compatibility requirement |
| Post-quantum prep | None (both broken by Shor) | Migrate to lattice-based |

---

## 10. Exercises

### Exercise 1: Point Arithmetic (Basic)

On the curve $y^2 = x^3 + 2x + 3$ over $\mathbb{F}_{97}$:
1. Verify that $P = (3, 6)$ is on the curve
2. Compute $2P$ (point doubling)
3. Compute $3P = 2P + P$
4. Find the order of $P$ (smallest $k$ such that $kP = \mathcal{O}$)

### Exercise 2: ECDH Implementation (Intermediate)

1. Implement ECDH key exchange using the `EllipticCurveFiniteField` class
2. Have Alice and Bob generate key pairs and compute shared secrets
3. Verify the shared secrets match
4. Show that an eavesdropper with both public keys cannot compute the shared secret without solving ECDLP

### Exercise 3: Curve Point Counting (Intermediate)

1. Count all points on $y^2 = x^3 + x + 1$ over $\mathbb{F}_p$ for $p = 5, 7, 11, 13, 17, 19, 23$
2. Verify Hasse's bound for each: $|\ \#E - (p+1)\ | \leq 2\sqrt{p}$
3. Plot $\#E(F_p)$ vs $p$. Does the relationship appear linear?

### Exercise 4: Edwards vs Weierstrass (Challenging)

1. Implement both Weierstrass and twisted Edwards point addition
2. For each form, count the number of conditional branches in the addition code
3. Time 10,000 scalar multiplications on each form for the same underlying group
4. Explain why Edwards curves are preferred for constant-time implementations

### Exercise 5: Security Level Analysis (Challenging)

1. For each of the curves P-256, P-384, P-521, Curve25519:
   - State the group order $n$
   - Compute the security level as $\lfloor\log_2(\sqrt{n})\rfloor$
   - Compare with the equivalent RSA key size
2. If quantum computers with 4000 logical qubits become available, which of these curves remain secure? (Research the qubit requirements for Shor's algorithm applied to ECDLP)

---

**Previous**: [RSA Cryptosystem](./05_RSA_Cryptosystem.md) | **Next**: [Digital Signatures](./07_Digital_Signatures.md)

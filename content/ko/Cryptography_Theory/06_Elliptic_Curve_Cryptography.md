# 레슨 6: 타원곡선 암호

**이전**: [RSA 암호체계](./05_RSA_Cryptosystem.md) | **다음**: [디지털 서명](./07_Digital_Signatures.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 실수 위에서 타원곡선 점 덧셈의 기하학적 직관을 설명할 수 있다
2. 유한체 위의 타원곡선을 정의하고 점 덧셈을 대수적으로 수행할 수 있다
3. 이중-덧셈(double-and-add) 알고리즘을 사용한 스칼라 곱셈(scalar multiplication)을 구현할 수 있다
4. 타원곡선 이산 로그 문제(ECDLP, Elliptic Curve Discrete Logarithm Problem)가 어려운 이유를 설명할 수 있다
5. 키 크기, 성능, 보안 수준 면에서 ECC와 RSA를 비교할 수 있다
6. 가장 중요한 명명된 곡선(secp256k1, Curve25519, Ed25519)과 그 설계 철학을 식별할 수 있다

---

타원곡선 암호(ECC, Elliptic Curve Cryptography)는 RSA와 동일한 보안을 훨씬 작은 키로 달성합니다. 256비트 ECC 키는 3072비트 RSA 키와 동등한 보안을 제공합니다. 이는 더 빠른 연산, 더 적은 대역폭, 더 작은 인증서를 의미합니다. 1985년 코블리츠(Koblitz)와 밀러(Miller)가 도입한 이래, ECC는 TLS 1.3, SSH, 비트코인, Signal에서 지배적인 공개 키 알고리즘이 되었습니다. 이 레슨은 기하학적 직관에서 유한체 산술까지, 실용적인 키 생성에 이르는 수학적 체계를 구축합니다.

> **비유:** 타원곡선의 점 덧셈은 곡선에서 당구공이 튀기는 것과 같습니다. 두 점을 지나는 직선을 그어 곡선과 만나는 세 번째 점을 찾고 반사시킵니다. 다음 점을 계산하기는 쉽지만, 시작 점과 끝 점이 주어졌을 때 수천 번의 튀김을 역으로 추적하는 것은 거의 불가능합니다. 이 단방향 성질이 바로 ECDLP입니다.

## 목차

1. [실수 위의 타원곡선](#1-실수-위의-타원곡선)
2. [군 법칙: 점 덧셈](#2-군-법칙-점-덧셈)
3. [유한체 위의 타원곡선](#3-유한체-위의-타원곡선)
4. [스칼라 곱셈](#4-스칼라-곱셈)
5. [타원곡선 이산 로그 문제](#5-타원곡선-이산-로그-문제)
6. [명명된 곡선과 표준](#6-명명된-곡선과-표준)
7. [몽고메리 곡선과 에드워즈 곡선](#7-몽고메리-곡선과-에드워즈-곡선)
8. [ECC 키 쌍과 ECDH](#8-ecc-키-쌍과-ecdh)
9. [ECC vs RSA 비교](#9-ecc-vs-rsa-비교)
10. [연습 문제](#10-연습-문제)

---

## 1. 실수 위의 타원곡선

### 1.1 바이어슈트라스 방정식

**짧은 바이어슈트라스 형식(short Weierstrass form)**의 타원곡선은 다음으로 정의됩니다:

$$y^2 = x^3 + ax + b$$

여기서 **판별식(discriminant)** $\Delta = -16(4a^3 + 27b^2) \neq 0$ (특이점 없음을 보장 — 뾰족점(cusp)이나 자기 교차(self-intersection) 없음).

### 1.2 시각화

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

### 1.3 무한원점

곡선에는 특별한 **무한원점(point at infinity)** $\mathcal{O}$가 포함됩니다. 이는 군의 **항등원(identity element)**으로 기능합니다. 기하학적으로, 모든 수직선이 만나는 점입니다.

---

## 2. 군 법칙: 점 덧셈

### 2.1 기하학적 구성

곡선 위의 두 점 $P = (x_1, y_1)$과 $Q = (x_2, y_2)$가 주어지면:

**점 덧셈($P \neq Q$):**
1. $P$와 $Q$를 지나는 직선을 긋는다
2. 이 직선은 곡선과 세 번째 점 $R'$에서 교차한다
3. $R'$를 x축에 대해 반사하여 $R = P + Q$를 구한다

**점 두 배($P = Q$):**
1. $P$에서의 접선을 긋는다
2. 이 직선은 곡선과 두 번째 점 $R'$에서 교차한다
3. $R'$를 x축에 대해 반사하여 $R = 2P$를 구한다

### 2.2 대수적 공식

$P = (x_1, y_1)$과 $Q = (x_2, y_2)$에 대해:

**덧셈($P \neq Q$):**

$$\lambda = \frac{y_2 - y_1}{x_2 - x_1}$$

**두 배($P = Q$):**

$$\lambda = \frac{3x_1^2 + a}{2y_1}$$

**결과 점:**

$$x_3 = \lambda^2 - x_1 - x_2$$
$$y_3 = \lambda(x_1 - x_3) - y_1$$

**특수 경우:**
- $P + \mathcal{O} = P$ (항등원)
- $P + (-P) = \mathcal{O}$ (여기서 $-P = (x_1, -y_1)$은 역원)

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

### 2.3 군 성질

타원곡선 위의 점들의 집합과 점 덧셈은 **아벨 군(abelian group)**을 이룹니다:

| 성질 | 의미 |
|------|------|
| 닫힘성(Closure) | $P + Q$는 곡선 위에 있다 |
| 결합법칙(Associativity) | $(P + Q) + R = P + (Q + R)$ |
| 항등원(Identity) | $P + \mathcal{O} = P$ |
| 역원(Inverse) | $P + (-P) = \mathcal{O}$ |
| 교환법칙(Commutativity) | $P + Q = Q + P$ |

---

## 3. 유한체 위의 타원곡선

### 3.1 실수에서 $\mathbb{F}_p$로

암호학에서는 $p$가 큰 소수인 **소수 유한체(prime finite field)** $\mathbb{F}_p$ 위에서 작업합니다. 곡선 방정식은 다음이 됩니다:

$$y^2 \equiv x^3 + ax + b \pmod{p}$$

모든 산술 연산(덧셈, 뺄셈, 곱셈, 나눗셈)은 모듈로 $p$로 수행됩니다. 나눗셈에는 모듈러 역원(modular inverse)을 사용합니다 (레슨 1, 섹션 3).

### 3.2 실수와의 주요 차이점

| 성질 | $\mathbb{R}$ 위 | $\mathbb{F}_p$ 위 |
|------|-----------------|-------------------|
| 점 | 연속 곡선 | 이산 $(x, y)$ 쌍의 집합 |
| 점의 수 | 무한 | 유한 (하세 정리에 의해 $\approx p$) |
| 시각화 | 매끄러운 곡선 | 분산된 점들 |
| 나눗셈 | 일반 나눗셈 | 모듈러 역원 |
| 제곱근 | $\sqrt{\cdot}$ | 토넬리-샹크스(Tonelli-Shanks) 알고리즘 |

### 3.3 하세의 정리

$E(\mathbb{F}_p)$의 점 개수 $\#E(\mathbb{F}_p)$는 다음을 만족합니다:

$$|\ \#E(\mathbb{F}_p) - (p + 1)\ | \leq 2\sqrt{p}$$

따라서 점의 수는 대략 $p$에 가깝습니다.

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

## 4. 스칼라 곱셈

### 4.1 정의

**스칼라 곱셈(scalar multiplication)**은 $kP = P + P + \cdots + P$ ($k$번)을 계산합니다. 이것은 ECC의 근본 연산으로, RSA의 모듈러 지수승(modular exponentiation)에 해당하는 타원곡선 유사 연산입니다.

### 4.2 이중-덧셈 알고리즘

RSA가 $a^k \bmod n$에 제곱-후-곱셈(square-and-multiply)을 사용하는 것처럼 (레슨 1, 섹션 7), ECC는 $kP$에 **이중-덧셈(double-and-add)**을 사용합니다:

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

### 4.3 부채널 저항: 몽고메리 래더

이중-덧셈 알고리즘은 타이밍 분석 및 전력 분석을 통해 정보를 누출합니다 (`if k & 1` 분기의 실행 시간이 다름). **몽고메리 래더(Montgomery ladder)**는 비트 값에 무관하게 항상 동일한 수의 연산을 수행합니다:

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

## 5. 타원곡선 이산 로그 문제

### 5.1 정의

타원곡선 위의 점 $P$와 $Q = kP$가 주어졌을 때 스칼라 $k$를 구하는 문제입니다.

$$\text{주어진 } P \text{와 } Q = kP, \text{ } k \text{를 구하라}$$

### 5.2 ECDLP가 DLP보다 어려운 이유

$\mathbb{Z}_p^*$에서의 이산 로그 문제(Discrete Logarithm Problem)(레슨 1)에는 준지수적(sub-exponential) 알고리즘이 존재합니다 (지표 계산법(index calculus)). ECDLP에 대해서는 **준지수적 알고리즘이 알려져 있지 않습니다**:

| 문제 | 최선 알고리즘 | 복잡도 |
|------|------------|--------|
| $\mathbb{Z}_p^*$에서의 DLP | 지표 계산법(Index Calculus) | $L_p(1/3)$ (준지수적) |
| ECDLP | 폴라드의 로(Pollard's Rho) | $O(\sqrt{n})$ (완전 지수적) |
| 인수분해(RSA) | GNFS | $L_n(1/3)$ (준지수적) |

이것이 ECC가 훨씬 작은 파라미터로 동등한 보안을 달성하는 이유입니다.

### 5.3 ECDLP에 대한 폴라드의 로

일반 곡선에 대한 ECDLP의 최선 알고리즘:

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

## 6. 명명된 곡선과 표준

### 6.1 NIST 곡선

| 곡선 | 필드 크기 | 보안 수준 | 용도 |
|------|----------|----------|------|
| P-256 (secp256r1) | 256비트 | 128비트 | TLS, 가장 일반적 |
| P-384 (secp384r1) | 384비트 | 192비트 | 정부, 고보안 |
| P-521 (secp521r1) | 521비트 | 256비트 | 최대 보안 |

**논란:** NIST 곡선은 NSA가 생성한 "무작위" 파라미터를 사용합니다. 2013년 Dual_EC_DRBG 백도어 스캔들 이후 일부 연구자들은 NIST 곡선을 불신하며, 독립적으로 생성된 곡선을 선호합니다.

### 6.2 secp256k1 (비트코인)

$$y^2 = x^3 + 7 \quad \text{over } \mathbb{F}_p$$

여기서 $p = 2^{256} - 2^{32} - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1$.

- 비트코인, 이더리움 및 기타 암호화폐에서 사용
- 효율성을 위해 선택됨 ($a = 0$이 점 두 배 공식을 단순화)
- 파라미터가 "검증 가능하게 무작위" (단순한 공식으로 유도됨)

### 6.3 Curve25519와 Ed25519

**Curve25519** (다니엘 J. 번스타인(Daniel J. Bernstein), 2006):
- 몽고메리 곡선(Montgomery curve): $y^2 = x^3 + 486662x^2 + x$ over $\mathbb{F}_{2^{255}-19}$
- 디피-헬만 키 교환(Diffie-Hellman key exchange)용으로 설계 (X25519)
- 설계상 상수 시간, 타이밍 공격에 저항

**Ed25519** (번스타인 외, 2011):
- Curve25519와 쌍유리적으로 동등(birationally equivalent)한 비틀린 에드워즈 곡선(Twisted Edwards curve)
- 디지털 서명(digital signatures)용으로 설계 (EdDSA)
- 결정론적 서명 (무작위 논스(nonce) 불필요 — PS3 재앙 방지, 레슨 7)
- SSH, Signal, WireGuard, Tor에서 사용

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

## 7. 몽고메리 곡선과 에드워즈 곡선

### 7.1 몽고메리 곡선

**몽고메리 곡선(Montgomery curve)**은 다음 형태를 가집니다:

$$By^2 = x^3 + Ax^2 + x$$

**핵심 장점:** 몽고메리 래더는 자연스럽게 $x$좌표만 사용하여 스칼라 곱셈을:
- 상수 시간으로 (비밀 비트에 분기 없음)
- 효율적으로 ($y$좌표가 마지막 단계까지 불필요)
- 구조적으로 부채널 공격에 저항하게 만듭니다

### 7.2 에드워즈 곡선

**비틀린 에드워즈 곡선(twisted Edwards curve)**은 다음 형태를 가집니다:

$$ax^2 + y^2 = 1 + dx^2y^2$$

**핵심 장점:** 덧셈 공식이 **완전(complete)**합니다 — 모든 점 쌍에 대해 특수 경우 없이 동작합니다 ($P = Q$, $P = -Q$ 등을 확인할 필요 없음):

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

### 7.3 쌍유리 동치

몽고메리, 에드워즈, 바이어슈트라스 형식은 **쌍유리적으로 동치(birationally equivalent)**입니다 — 이들 사이에 유리 사상(rational map)이 존재합니다. Curve25519(몽고메리)와 Ed25519(에드워즈)는 동일한 군을 나타내며, 서로 다른 형식은 각각의 계산상 이점을 위해 선택된 것입니다.

---

## 8. ECC 키 쌍과 ECDH

### 8.1 키 생성

1. 기저점(base point) $G$와 위수(order) $n$을 가진 표준화된 곡선 $E$를 선택한다
2. 무작위 개인 키 $d \in \{1, 2, \ldots, n-1\}$를 선택한다
3. 공개 키 $Q = dG$를 계산한다

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

### 8.2 타원곡선 디피-헬만(ECDH)

ECDH(Elliptic Curve Diffie-Hellman)는 두 당사자가 안전하지 않은 채널을 통해 공유 비밀을 설정할 수 있게 합니다:

1. 앨리스: 개인 키 $d_A$, 공개 키 $Q_A = d_A G$
2. 밥: 개인 키 $d_B$, 공개 키 $Q_B = d_B G$
3. 공유 비밀: $S = d_A Q_B = d_A (d_B G) = d_B (d_A G) = d_B Q_A$

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

## 9. ECC vs RSA 비교

### 9.1 키 크기 비교

| 보안 수준(비트) | RSA 키 | ECC 키 | RSA/ECC 비율 |
|----------------|--------|--------|-------------|
| 80 | 1024 | 160 | 6.4배 |
| 112 | 2048 | 224 | 9.1배 |
| 128 | 3072 | 256 | 12배 |
| 192 | 7680 | 384 | 20배 |
| 256 | 15360 | 521 | 29.5배 |

### 9.2 성능 비교

| 연산 | RSA-2048 | ECDSA P-256 | 우위 |
|------|----------|-------------|------|
| 키 생성 | ~100ms | ~1ms | ECC (100배) |
| 서명 | ~2ms | ~1ms | ECC (2배) |
| 검증 | ~0.1ms | ~2ms | RSA (20배) |
| 키 크기 | 256바이트 | 32바이트 | ECC (8배) |

RSA 검증은 $e = 65537$이 작아서 더 빠르지만, 그 외 모든 지표에서 ECC가 우위입니다.

### 9.3 각각을 사용할 때

| 사용 사례 | 권장 사항 | 이유 |
|----------|----------|------|
| TLS 1.3 | ECDHE + ECDSA | 기본값, 가장 빠른 핸드셰이크 |
| SSH 키 | Ed25519 | 가장 작은 키, 가장 빠름 |
| 코드 서명 | ECDSA 또는 RSA | 검증 위주 → RSA 허용 |
| 비트코인 | secp256k1 | 생태계 표준화 |
| 레거시 시스템 | RSA-2048 | 호환성 요건 |
| 포스트 양자 준비 | 해당 없음 (두 방식 모두 쇼어 알고리즘에 취약) | 격자 기반으로 마이그레이션 필요 |

---

## 10. 연습 문제

### 연습 문제 1: 점 산술 (기초)

$\mathbb{F}_{97}$ 위의 곡선 $y^2 = x^3 + 2x + 3$에서:
1. $P = (3, 6)$이 곡선 위에 있음을 확인하라
2. $2P$를 계산하라 (점 두 배)
3. $3P = 2P + P$를 계산하라
4. $P$의 위수를 구하라 ($kP = \mathcal{O}$가 되는 최소 $k$)

### 연습 문제 2: ECDH 구현 (중급)

1. `EllipticCurveFiniteField` 클래스를 사용해 ECDH 키 교환을 구현하라
2. 앨리스와 밥이 키 쌍을 생성하고 공유 비밀을 계산하게 하라
3. 공유 비밀이 일치함을 확인하라
4. 두 공개 키를 가진 도청자가 ECDLP를 풀지 않고서는 공유 비밀을 계산할 수 없음을 보여라

### 연습 문제 3: 곡선 점 개수 세기 (중급)

1. $p = 5, 7, 11, 13, 17, 19, 23$에 대해 $\mathbb{F}_p$ 위의 $y^2 = x^3 + x + 1$의 모든 점을 세라
2. 각각에 대해 하세의 경계를 검증하라: $|\ \#E - (p+1)\ | \leq 2\sqrt{p}$
3. $\#E(F_p)$ 대 $p$ 그래프를 그려라. 관계가 선형으로 보이는가?

### 연습 문제 4: 에드워즈 vs 바이어슈트라스 (도전)

1. 바이어슈트라스와 비틀린 에드워즈 점 덧셈을 모두 구현하라
2. 각 형식의 덧셈 코드에서 조건 분기의 수를 세라
3. 동일한 군에 대해 각 형식으로 10,000회 스칼라 곱셈을 수행하고 시간을 재라
4. 에드워즈 곡선이 상수 시간 구현에서 선호되는 이유를 설명하라

### 연습 문제 5: 보안 수준 분석 (도전)

1. P-256, P-384, P-521, Curve25519 각각에 대해:
   - 군의 위수 $n$을 명시하라
   - 보안 수준을 $\lfloor\log_2(\sqrt{n})\rfloor$로 계산하라
   - 동등한 RSA 키 크기와 비교하라
2. 4000개의 논리 큐비트를 가진 양자 컴퓨터가 가용해진다면, 이 곡선들 중 어떤 것이 안전하게 유지되는가? (ECDLP에 쇼어 알고리즘을 적용할 때 필요한 큐비트 요건을 조사하라)

---

**이전**: [RSA 암호체계](./05_RSA_Cryptosystem.md) | **다음**: [디지털 서명](./07_Digital_Signatures.md)

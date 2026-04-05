# 레슨 7: 디지털 서명

**이전**: [타원곡선 암호](./06_Elliptic_Curve_Cryptography.md) | **다음**: [키 교환](./08_Key_Exchange.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 디지털 서명의 보안 목표(인증, 무결성, 부인 방지)를 설명하고 MAC과 비교한다
2. PSS 패딩을 사용한 RSA 서명을 설명하고 구현한다
3. ECDSA를 구현하고 서명 및 검증 알고리즘을 추적한다
4. ECDSA에서 논스 재사용이 왜 치명적인지 플레이스테이션 3 해킹 사례 연구를 통해 설명한다
5. EdDSA/Ed25519를 설명하고 결정론적 논스가 취약점의 전체 유형을 제거하는 이유를 설명한다
6. Schnorr 서명, 블라인드 서명, 집계 서명을 비교한다

---

디지털 서명은 손으로 쓴 서명의 암호학적 동등물이지만, 훨씬 더 강력하다. 손으로 쓴 서명은 위조되거나, 복사되거나, 문서에서 분리될 수 있지만, 디지털 서명은 서명자의 신원을 메시지의 정확한 내용에 수학적으로 결합한다. 아무리 작은 수정이라도 서명을 무효화한다. 디지털 서명은 코드 서명, 인증 기관, 블록체인 트랜잭션, 안전한 소프트웨어 업데이트의 기반이다. 이 레슨에서는 RSA-PSS부터 ECDSA, 현대적인 Ed25519까지 주요 서명 방식을 다룬다.

> **비유:** 디지털 서명은 봉랍 인장(wax seal)과 같다 — 발신자의 신원을 증명하고 메시지가 변조되지 않았음을 보장한다. 하지만 봉랍 인장과 달리, 누구나 (공개 키를 사용하여) 검증할 수 있으며, 특정 메시지에 수학적으로 결합되어 있어 단 1비트만 변경해도 인장이 깨진다.

## 목차

1. [디지털 서명 개념](#1-디지털-서명-개념)
2. [RSA 서명](#2-rsa-서명)
3. [RSA-PSS: 확률적 서명 방식](#3-rsa-pss-확률적-서명-방식)
4. [ECDSA: 타원곡선 디지털 서명 알고리즘](#4-ecdsa-타원곡선-디지털-서명-알고리즘)
5. [논스 재사용: 플레이스테이션 3 재앙](#5-논스-재사용-플레이스테이션-3-재앙)
6. [EdDSA와 Ed25519](#6-eddsa와-ed25519)
7. [Schnorr 서명](#7-schnorr-서명)
8. [고급 서명 방식](#8-고급-서명-방식)
9. [보안 모델](#9-보안-모델)
10. [연습 문제](#10-연습-문제)

---

## 1. 디지털 서명 개념

### 1.1 서명이 제공하는 것

| 속성 | 설명 | MAC 제공 여부 | 서명 제공 여부 |
|------|------|--------------|--------------|
| **인증(Authentication)** | 메시지를 보낸 사람 확인 | 예 | 예 |
| **무결성(Integrity)** | 메시지 변조 여부 감지 | 예 | 예 |
| **부인 방지(Non-repudiation)** | 서명자가 서명을 부인할 수 없음 | **아니오** | **예** |
| **공개 검증 가능성(Public verifiability)** | 누구나 검증 가능 | **아니오** (공유 비밀) | **예** (공개 키) |

부인 방지(Non-repudiation)가 핵심적인 차이점이다. HMAC에서는 양 당사자가 키를 공유하므로 어느 쪽이든 MAC을 생성했을 수 있다. 디지털 서명에서는 개인 키를 보유한 사람만 서명을 생성할 수 있다.

### 1.2 서명 방식의 구성 요소

디지털 서명 방식은 세 가지 알고리즘으로 구성된다:

1. **KeyGen()** $\rightarrow$ (공개 키 $pk$, 개인 키 $sk$)
2. **Sign($sk$, $m$)** $\rightarrow$ 서명 $\sigma$
3. **Verify($pk$, $m$, $\sigma$)** $\rightarrow$ 수락/거부

**정확성:** 임의의 $(pk, sk) \leftarrow \text{KeyGen}()$와 임의의 메시지 $m$에 대해:

$$\text{Verify}(pk, m, \text{Sign}(sk, m)) = \text{accept}$$

### 1.3 해시-후-서명 패러다임(Hash-Then-Sign Paradigm)

모든 실용적인 서명 방식은 메시지 자체가 아닌 메시지의 **해시**에 서명한다:

$$\sigma = \text{Sign}(sk, H(m))$$

**먼저 해시하는 이유:**
- 효율성: 임의 길이의 메시지 대신 고정 크기의 다이제스트에 서명
- 보안: 서명 함수의 구조를 이용하는 대수적 공격 방지 (레슨 5, 4절에서 다룬 교과서 RSA 참조)

---

## 2. RSA 서명

### 2.1 교과서 RSA 서명(Textbook RSA Signatures)

**서명:** $\sigma = H(m)^d \bmod n$

**검증:** $\sigma^e \bmod n = H(m)$인지 확인

레슨 5에서 논의된 바와 같이, 교과서 RSA 서명은 실존적 위조(existential forgery)에 취약하다. 서명자는 패딩 방식을 사용해야 한다.

### 2.2 PKCS#1 v1.5 서명

가장 널리 배포된 RSA 서명 패딩 방식 (신규 설계에서는 더 이상 권장되지 않음):

```
0x00 || 0x01 || [0xFF 바이트들] || 0x00 || [해시의 DigestInfo ASN.1 인코딩]
```

```python
import hashlib

def pkcs1_v15_sign(message, n, d):
    """PKCS#1 v1.5 signature with SHA-256.

    Why the DigestInfo encoding: it binds the hash algorithm identifier
    to the signature, preventing cross-algorithm attacks. Without it,
    an attacker could claim the signature was made with a different
    (weaker) hash function.
    """
    # SHA-256 DigestInfo header (DER-encoded ASN.1)
    DIGEST_INFO_SHA256 = bytes.fromhex(
        "3031300d060960864801650304020105000420"
    )

    h = hashlib.sha256(message).digest()
    k = (n.bit_length() + 7) // 8  # Modulus size in bytes

    # Construct padded message
    T = DIGEST_INFO_SHA256 + h
    ps_len = k - len(T) - 3
    if ps_len < 8:
        raise ValueError("Modulus too small for this message")

    em = b'\x00\x01' + b'\xff' * ps_len + b'\x00' + T

    # Sign: m^d mod n
    m_int = int.from_bytes(em, 'big')
    sig = pow(m_int, d, n)

    return sig.to_bytes(k, 'big')

def pkcs1_v15_verify(message, signature, n, e):
    """PKCS#1 v1.5 signature verification."""
    DIGEST_INFO_SHA256 = bytes.fromhex(
        "3031300d060960864801650304020105000420"
    )

    k = (n.bit_length() + 7) // 8
    sig_int = int.from_bytes(signature, 'big')
    em_int = pow(sig_int, e, n)
    em = em_int.to_bytes(k, 'big')

    # Verify padding structure
    if em[0:2] != b'\x00\x01':
        return False

    # Find 0x00 separator
    sep_idx = em.index(b'\x00', 2)
    if em[2:sep_idx] != b'\xff' * (sep_idx - 2):
        return False

    # Extract and verify digest info + hash
    T = em[sep_idx + 1:]
    expected_hash = hashlib.sha256(message).digest()
    expected_T = DIGEST_INFO_SHA256 + expected_hash

    return T == expected_T
```

### 2.3 블라이헨바허의 서명 위조(Bleichenbacher's Signature Forgery, 2006)

일부 구현에서는 PKCS#1 v1.5 서명을 잘못 검증하여 패딩 끝의 쓰레기 바이트를 무시하고 패딩 시작 부분의 해시만 확인했다. 이로 인해 $e = 3$ (작은 공개 지수)일 때 위조가 가능했다.

**교훈:** 패딩된 메시지를 항상 **엄격하게** 파싱하라 — 시작 부분만이 아니라 전체 구조를 검증해야 한다.

---

## 3. RSA-PSS: 확률적 서명 방식

### 3.1 PKCS#1 v1.5 대신 PSS를 사용하는 이유

| 속성 | PKCS#1 v1.5 | RSA-PSS |
|------|-------------|---------|
| 무작위성 | 아니오 (결정론적) | 예 (무작위 솔트) |
| 보안 증명 | 휴리스틱 | 증명 가능한 보안 (랜덤 오라클 모델) |
| 표준 상태 | 레거시 | 권장 |

PSS는 무작위 솔트를 추가하여 동일한 메시지라도 매번 다른 서명을 생성한다. 이는 랜덤 오라클 모델에서 증명 가능한 보안을 제공한다.

### 3.2 PSS 구조 (단순화)

```
mHash = Hash(message)
M' = (8개의 영바이트) || mHash || salt
H = Hash(M')
DB = padding || 0x01 || salt
maskedDB = DB XOR MGF1(H)
EM = maskedDB || H || 0xBC
```

```python
def rsa_pss_concept():
    """Explain RSA-PSS structure and its security advantages.

    Why randomized signatures:
    1. Provable security reduction to RSA problem
    2. Each signature is unique (even for the same message)
    3. Prevents deterministic-signature attacks
    4. Recommended by NIST SP 800-131A and PKCS#1 v2.2
    """
    import os

    message = b"Sign this document"
    salt = os.urandom(32)  # Random salt for each signature

    # Step 1: Hash the message
    m_hash = hashlib.sha256(message).digest()

    # Step 2: Create M' = (8 zero bytes) || mHash || salt
    m_prime = b'\x00' * 8 + m_hash + salt

    # Step 3: H = Hash(M')
    H = hashlib.sha256(m_prime).digest()

    print(f"Message hash: {m_hash.hex()[:32]}...")
    print(f"Salt:         {salt.hex()[:32]}...")
    print(f"H (from M'):  {H.hex()[:32]}...")
    print()
    print("Two signatures of the same message produce different H values")
    print("because the salt is random each time.")
    print()

    # Sign again with different salt
    salt2 = os.urandom(32)
    m_prime2 = b'\x00' * 8 + m_hash + salt2
    H2 = hashlib.sha256(m_prime2).digest()
    print(f"H (salt 2):   {H2.hex()[:32]}...")
    print(f"H values differ: {H != H2}")

rsa_pss_concept()
```

---

## 4. ECDSA: 타원곡선 디지털 서명 알고리즘

### 4.1 매개변수

- $\mathbb{F}_p$ 위의 타원곡선 $E$와 위수(order) $n$의 기저점 $G$
- 개인 키: $d \in \{1, \ldots, n-1\}$
- 공개 키: $Q = dG$

### 4.2 서명 알고리즘

메시지 $m$에 서명하려면:

1. $e = H(m)$을 계산한다 ($n$의 비트 길이로 잘라냄)
2. **무작위 논스(nonce)** $k \in \{1, \ldots, n-1\}$를 선택한다
3. $R = kG = (x_R, y_R)$를 계산한다
4. $r = x_R \bmod n$을 계산한다 ($r = 0$이면 새로운 $k$를 선택)
5. $s = k^{-1}(e + rd) \bmod n$을 계산한다 ($s = 0$이면 새로운 $k$를 선택)
6. 서명은 $(r, s)$

### 4.3 검증 알고리즘

공개 키 $Q$로 메시지 $m$에 대한 서명 $(r, s)$를 검증하려면:

1. $e = H(m)$을 계산한다
2. $w = s^{-1} \bmod n$을 계산한다
3. $u_1 = ew \bmod n$과 $u_2 = rw \bmod n$을 계산한다
4. $R' = u_1 G + u_2 Q$를 계산한다
5. $R'_x \equiv r \pmod{n}$이면 수락한다

### 4.4 검증이 성립하는 이유

$$R' = u_1 G + u_2 Q = ewG + rwQ = ewG + rw(dG) = (ew + rwd)G$$

$w = s^{-1}$이고 $s = k^{-1}(e + rd)$이므로:

$$ew + rwd = es^{-1} + rds^{-1} = (e + rd)s^{-1} = (e + rd) \cdot \frac{k}{e + rd} = k$$

따라서 $R' = kG = R$이고 $R'_x = r$. $\checkmark$

```python
class ECDSA:
    """ECDSA implementation using the EllipticCurveFiniteField class from Lesson 6.

    WARNING: Educational implementation. Production ECDSA must use
    constant-time operations, validated curves, and RFC 6979 nonces.
    """

    def __init__(self, curve, G, n):
        """
        Args:
            curve: EllipticCurveFiniteField instance
            G: base point (generator)
            n: order of G
        """
        self.curve = curve
        self.G = G
        self.n = n

    def keygen(self):
        """Generate ECDSA key pair."""
        import random
        d = random.randrange(1, self.n)
        Q = self.curve.scalar_multiply(d, self.G)
        return d, Q

    def sign(self, message, private_key):
        """ECDSA signing.

        Why the nonce k MUST be:
        1. Random (or deterministic via RFC 6979)
        2. Unique per signature
        3. Secret (never revealed)
        Violating ANY of these leads to private key recovery (Section 5).
        """
        import random

        d = private_key
        e = int(hashlib.sha256(message).hexdigest(), 16) % self.n

        while True:
            k = random.randrange(1, self.n)
            R = self.curve.scalar_multiply(k, self.G)
            r = R[0] % self.n

            if r == 0:
                continue

            k_inv = pow(k, -1, self.n)
            s = (k_inv * (e + r * d)) % self.n

            if s == 0:
                continue

            return (r, s)

    def verify(self, message, signature, public_key):
        """ECDSA verification.

        Why we compute u1*G + u2*Q:
        - This is a linear combination of the base point and public key
        - If the signature is valid, this recovers the point R = kG
        - The x-coordinate of R must equal r from the signature
        """
        r, s = signature
        Q = public_key

        if not (1 <= r < self.n and 1 <= s < self.n):
            return False

        e = int(hashlib.sha256(message).hexdigest(), 16) % self.n

        w = pow(s, -1, self.n)
        u1 = (e * w) % self.n
        u2 = (r * w) % self.n

        # R' = u1*G + u2*Q
        P1 = self.curve.scalar_multiply(u1, self.G)
        P2 = self.curve.scalar_multiply(u2, Q)
        R_prime = self.curve.add(P1, P2)

        if R_prime is None:
            return False

        return R_prime[0] % self.n == r


# Need the EllipticCurveFiniteField class from Lesson 6
class EllipticCurveFiniteField:
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


# Demo
curve = EllipticCurveFiniteField(a=2, b=3, p=97)
G = (3, 6)
n = curve.count_points()  # Group order

ecdsa = ECDSA(curve, G, n)
d, Q = ecdsa.keygen()

message = b"Hello, ECDSA!"
signature = ecdsa.sign(message, d)

print(f"Private key d = {d}")
print(f"Public key Q = {Q}")
print(f"Signature (r, s) = {signature}")
print(f"Verification: {ecdsa.verify(message, signature, Q)}")
print(f"Tampered: {ecdsa.verify(b'Hello, ECDSA?', signature, Q)}")
```

---

## 5. 논스 재사용: 플레이스테이션 3 재앙

### 5.1 공격

2010년, fail0verflow 팀은 소니가 플레이스테이션 3 펌웨어의 모든 ECDSA 서명에 **동일한 논스 $k$**를 사용했음을 밝혔다. 이로 인해 소니의 개인 서명 키가 완전히 복구될 수 있었다.

### 5.2 수학적 원리

두 서명 $(r_1, s_1)$과 $(r_2, s_2)$가 같은 논스 $k$를 사용하는 경우:

$R = kG$가 동일하므로 $r_1 = r_2 = r$이다.

$$s_1 = k^{-1}(e_1 + rd) \bmod n$$
$$s_2 = k^{-1}(e_2 + rd) \bmod n$$

빼면:

$$s_1 - s_2 = k^{-1}(e_1 - e_2) \bmod n$$

$k$를 구하면:

$$k = (e_1 - e_2)(s_1 - s_2)^{-1} \bmod n$$

$k$를 알면 $d$를 복구한다:

$$d = r^{-1}(s_1 k - e_1) \bmod n$$

```python
def nonce_reuse_attack_demo():
    """Demonstrate ECDSA private key recovery from nonce reuse.

    Why this is catastrophic: a SINGLE reused nonce across ANY two
    signatures reveals the private key permanently. The attacker
    then has complete control — they can sign arbitrary messages,
    forge certificates, or steal cryptocurrency.

    Real-world victims:
    - PlayStation 3 firmware signing key (2010)
    - Android Bitcoin wallet app (2013): reused k → stolen coins
    - Numerous smart contract vulnerabilities
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    # Generate target's key pair
    import random
    d_secret = random.randrange(1, n)
    Q = curve.scalar_multiply(d_secret, G)

    # Victim signs two messages with the SAME nonce (the vulnerability)
    k = random.randrange(1, n)  # Reused nonce!
    R = curve.scalar_multiply(k, G)
    r = R[0] % n

    msg1 = b"Firmware v4.0"
    msg2 = b"Firmware v4.1"
    e1 = int(hashlib.sha256(msg1).hexdigest(), 16) % n
    e2 = int(hashlib.sha256(msg2).hexdigest(), 16) % n

    k_inv = pow(k, -1, n)
    s1 = (k_inv * (e1 + r * d_secret)) % n
    s2 = (k_inv * (e2 + r * d_secret)) % n

    print(f"Two signatures with the same r (same nonce reused):")
    print(f"  Signature 1: (r={r}, s1={s1})")
    print(f"  Signature 2: (r={r}, s2={s2})")
    print(f"  (r values are identical → nonce reuse detected!)")
    print()

    # ATTACK: recover k from the two signatures
    k_recovered = ((e1 - e2) * pow(s1 - s2, -1, n)) % n
    print(f"  Recovered k = {k_recovered} (actual k = {k}, match: {k_recovered == k})")

    # ATTACK: recover private key d from k
    d_recovered = (pow(r, -1, n) * (s1 * k_recovered - e1)) % n
    print(f"  Recovered d = {d_recovered} (actual d = {d_secret}, match: {d_recovered == d_secret})")
    print()
    print("  GAME OVER: attacker now has the private signing key!")

nonce_reuse_attack_demo()
```

### 5.3 배운 교훈

1. **논스를 절대 재사용하지 마라.** 각 서명은 새롭고 예측 불가능한 $k$를 사용해야 한다.
2. **편향된 논스도 위험하다.** $k$ 비트의 부분적인 지식만 있어도 격자 공격(lattice attacks)이 가능하다.
3. **결정론적 논스**(RFC 6979)는 이 유형의 취약점 전체를 제거한다 — 개인 키와 메시지로부터 $k$를 결정론적으로 유도하므로, 동일한 $(d, m)$은 항상 같은 서명을 생성하고, 다른 메시지는 다른 논스를 생성한다.

---

## 6. EdDSA와 Ed25519

### 6.1 동기

ECDSA의 가장 큰 약점은 논스이다: 무작위이고, 고유하며, 비밀이어야 한다. 단 한 번의 실패도 치명적이다. **EdDSA**는 논스를 **결정론적**으로 만들어 이 문제를 해결한다.

### 6.2 Ed25519 매개변수

| 매개변수 | 값 |
|---------|---|
| 곡선 | 비틀린 에드워즈(Twisted Edwards): $-x^2 + y^2 = 1 + dx^2y^2$, $\mathbb{F}_{2^{255}-19}$ 위 |
| 기저점 | $G$ (표준, 완전히 명시됨) |
| 위수 | $n = 2^{252} + 27742317777372353535851937790883648493$ |
| 해시 | SHA-512 |
| 키 크기 | 256비트 개인 키, 256비트 공개 키 |
| 서명 크기 | 512비트 (64바이트) |

### 6.3 EdDSA 서명 알고리즘

1. **키 확장:** $h = \text{SHA-512}(sk)$. 하위 256비트를 스칼라 $a$ (비트 클램핑 포함)로, 상위 256비트를 논스 프리픽스 $\text{prefix}$로 사용한다.
2. **논스 계산:** $r = \text{SHA-512}(\text{prefix} \| m) \bmod n$
3. **$R = rG$ 계산**
4. **$S = (r + \text{SHA-512}(R \| pk \| m) \cdot a) \bmod n$ 계산**
5. **서명:** $(R, S)$

```python
def ed25519_concept():
    """Explain Ed25519's deterministic nonce generation.

    Why deterministic nonces are revolutionary:
    1. No random number generator needed during signing
    2. Nonce reuse is IMPOSSIBLE (same message → same nonce → same signature)
    3. Different messages → different nonces (SHA-512 is collision-resistant)
    4. The nonce depends on the private key, so an attacker without the
       private key cannot predict or influence the nonce
    5. Eliminates the ENTIRE class of nonce-related vulnerabilities

    Why Ed25519 over ECDSA:
    - Deterministic: no nonce disaster possible
    - Faster: ~3x faster signing, ~2x faster verification than ECDSA P-256
    - Smaller: 64-byte signatures (vs 70+ for ECDSA DER-encoded)
    - Simpler: fewer parameters, fewer implementation pitfalls
    - Constant-time: Edwards curve addition has no branches
    """
    print("Ed25519 Signing Process:")
    print("=" * 50)
    print()
    print("1. Key expansion: h = SHA-512(private_key)")
    print("   a = lower_256_bits(h)  (scalar, with bit clamping)")
    print("   prefix = upper_256_bits(h)  (nonce seed)")
    print()
    print("2. Nonce: r = SHA-512(prefix || message) mod n")
    print("   → Deterministic! Same (key, message) → same nonce")
    print("   → Different messages → different nonces")
    print("   → No RNG needed during signing")
    print()
    print("3. R = r * G  (nonce point)")
    print()
    print("4. S = (r + SHA-512(R || pk || message) * a) mod n")
    print()
    print("5. Signature = (R, S)  [64 bytes total]")
    print()
    print("Verification: check that S*G = R + SHA-512(R || pk || m) * Q")
    print()
    print("Adoption:")
    print("  - SSH (default key type since OpenSSH 6.5)")
    print("  - Signal Protocol")
    print("  - WireGuard")
    print("  - Tor")
    print("  - Solana, Cardano, Polkadot blockchains")

ed25519_concept()
```

### 6.4 Ed25519 검증

공개 키 $Q$로 메시지 $m$에 대한 서명 $(R, S)$를 검증하려면:

$$S \cdot G \stackrel{?}{=} R + H(R \| Q \| m) \cdot Q$$

이는 다음과 동치이다:

$$(r + H(R \| Q \| m) \cdot a) \cdot G = r \cdot G + H(R \| Q \| m) \cdot a \cdot G = R + H(R \| Q \| m) \cdot Q$$

### 6.5 비트 클램핑(Bit Clamping)

Ed25519는 개인 스칼라 $a$를 "클램핑"한다:
- 최하위 3비트를 지운다 (코팩터(cofactor)인 8의 배수가 되도록 보장)
- 최상위 비트를 지운다
- 두 번째로 높은 비트를 설정한다

```python
def bit_clamping_explanation():
    """Explain why Ed25519 clamps the private scalar.

    Why clamp:
    1. Clearing low 3 bits (cofactor 8): prevents small-subgroup attacks.
       The curve has cofactor h=8, meaning the full group has small
       subgroups of order 2, 4, 8. Clamping ensures the scalar is a
       multiple of 8, so multiplication always lands in the prime-order
       subgroup.

    2. Setting bit 255: ensures constant-time scalar multiplication.
       The Montgomery ladder needs a fixed number of iterations, which
       is determined by the position of the highest set bit.

    3. Clearing bit 256: ensures the scalar is less than 2^255,
       preventing modular reduction issues.
    """
    example_key = bytearray(range(32))  # Toy example

    # Clamp
    example_key[0] &= 248   # Clear lowest 3 bits (248 = 11111000)
    example_key[31] &= 127  # Clear highest bit    (127 = 01111111)
    example_key[31] |= 64   # Set second-highest   (64  = 01000000)

    print(f"Before clamping: byte[0]  = xxxx_xxxx")
    print(f"After clamping:  byte[0]  = xxxx_x000  (multiple of 8)")
    print(f"Before clamping: byte[31] = xxxx_xxxx")
    print(f"After clamping:  byte[31] = 01xx_xxxx  (bit 254 set, bit 255 clear)")

bit_clamping_explanation()
```

---

## 7. Schnorr 서명

### 7.1 개요

Schnorr 서명(1989)은 가장 단순한 이산 로그(discrete-log) 기반 서명 방식이다. 특허 제한(2008년 만료)으로 인해 역사적으로 덜 사용되었지만, 최근 비트코인(BIP 340, Taproot 업그레이드, 2021)에서 주목을 받고 있다.

### 7.2 알고리즘

**매개변수:** 생성원 $g$와 소수 위수 $q$를 가진 그룹 $\mathbb{Z}_p^*$, 또는 타원곡선 그룹.

**서명:**
1. 무작위 논스 $k$를 선택한다
2. $R = g^k$ (또는 ECC의 경우 $R = kG$)
3. $e = H(R \| m)$
4. $s = k - ed \bmod q$
5. 서명: $(e, s)$ 또는 $(R, s)$

**검증:**
1. $R' = g^s \cdot y^e$ (여기서 $y = g^d$가 공개 키)
2. $e \stackrel{?}{=} H(R' \| m)$을 확인한다

```python
def schnorr_signature_demo():
    """Demonstrate Schnorr signatures on an elliptic curve.

    Why Schnorr signatures matter:
    1. Simplest DL-based signature — minimal code, minimal attack surface
    2. Provably secure under the DL assumption (in the ROM)
    3. Linear structure enables advanced features:
       - Multi-signatures (MuSig): n signers, 1 combined signature
       - Threshold signatures: t-of-n signing
       - Adaptor signatures: conditional signatures for atomic swaps
    4. Adopted by Bitcoin (BIP 340) for Taproot in 2021
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    import random

    # Key generation
    d = random.randrange(1, n)
    Q = curve.scalar_multiply(d, G)

    # Signing
    message = b"Hello, Schnorr!"
    k = random.randrange(1, n)
    R = curve.scalar_multiply(k, G)

    # e = H(R || message) mod n
    e_input = f"{R[0]},{R[1]},{message.hex()}".encode()
    e = int(hashlib.sha256(e_input).hexdigest(), 16) % n

    s = (k - e * d) % n

    print(f"Schnorr Signature:")
    print(f"  Private key d = {d}")
    print(f"  Public key Q = {Q}")
    print(f"  Nonce point R = {R}")
    print(f"  e = {e}, s = {s}")

    # Verification: R' = s*G + e*Q should equal R
    sG = curve.scalar_multiply(s, G)
    eQ = curve.scalar_multiply(e, Q)
    R_prime = curve.add(sG, eQ)

    # Recompute e' = H(R' || message)
    e_prime_input = f"{R_prime[0]},{R_prime[1]},{message.hex()}".encode()
    e_prime = int(hashlib.sha256(e_prime_input).hexdigest(), 16) % n

    print(f"  Verification: R' = {R_prime}")
    print(f"  R' == R: {R_prime == R}")
    print(f"  e' == e: {e_prime == e}")
    print(f"  Signature valid: {e_prime == e}")

schnorr_signature_demo()
```

### 7.3 Schnorr vs ECDSA

| 속성 | ECDSA | Schnorr |
|------|-------|---------|
| 보안 증명 | 휴리스틱 | 증명 가능 (랜덤 오라클 모델) |
| 선형성(Linearity) | 아니오 | 예 (다중 서명 가능) |
| 서명 크기 | ~72바이트 (DER) | 64바이트 |
| 일괄 검증(Batch verification) | 가능하나 복잡 | 자연스럽고 효율적 |
| 다중 서명(Multi-signatures) | 복잡 (MuSig2) | 우아함 |
| 특허 상태 | 항상 무료 | 2008년까지 특허 |
| 채택 | 광범위 | 성장 중 (비트코인 Taproot) |

---

## 8. 고급 서명 방식

### 8.1 블라인드 서명(Blind Signatures)

**블라인드 서명**은 서명자가 메시지 내용을 보지 않고 사용자가 메시지에 서명을 받을 수 있게 한다. 데이비드 차움(David Chaum, 1982)이 익명 디지털 화폐를 위해 도입했다.

```python
def blind_signature_concept():
    """Explain blind signatures and their applications.

    How it works (RSA-based):
    1. Alice has message m
    2. Alice blinds: m' = m * r^e mod n  (r is random blinding factor)
    3. Signer signs: s' = (m')^d = m^d * r mod n
    4. Alice unblinds: s = s' * r^(-1) = m^d mod n
    5. Result: valid signature on m, but signer never saw m!

    Applications:
    - Anonymous digital cash (eCash, David Chaum)
    - Voting systems (vote without revealing to authority)
    - Certificate issuance (prove attribute without revealing identity)
    """
    print("Blind Signature Protocol (RSA-based):")
    print("=" * 50)
    print()
    print("Alice                         Signer")
    print("  |                             |")
    print("  | m' = m * r^e mod n          |")
    print("  |------------ m' ------------>|")
    print("  |                             | s' = (m')^d mod n")
    print("  |<----------- s' -------------|")
    print("  | s = s' * r^(-1) mod n       |")
    print("  |                             |")
    print("  | s is valid signature on m   |")
    print("  | Signer never saw m!         |")

blind_signature_concept()
```

### 8.2 집계 서명(Aggregate Signatures)

**집계 서명**은 $n$개의 메시지에 대한 $n$명의 서명자로부터의 $n$개 서명을 모든 $n$개의 공개 키에 대해 검증할 수 있는 단일 서명으로 결합한다.

| 방식 | 집계 크기 | 검증 |
|------|----------|------|
| 개별 서명 | $n \times 64$바이트 | $n$번 검증 |
| BLS 집계 | 48바이트 (상수!) | 1번 페어링(pairing) 확인 |
| Schnorr 다중 서명 | 64바이트 (동일 메시지) | 1번 검증 |

**응용:**
- 블록체인: 수천 개의 트랜잭션 서명을 하나로 압축
- 인증서 투명성(Certificate Transparency): 검증자 서명 집계
- IoT: 자원이 제한된 기기가 압축된 증명 전송

### 8.3 임계값 서명(Threshold Signatures)

$(t, n)$ **임계값 서명** 방식은 $n$개의 당사자 중 임의의 $t$개가 협력하여 유효한 서명을 생성할 수 있지만, $t - 1$개 이하의 당사자는 불가능하다.

**핵심 속성:** 결과 서명은 단일 당사자 서명과 구별할 수 없다 — 검증자는 여러 당사자가 참여했다는 것을 알 필요가 없다.

**응용:**
- 암호화폐 다중 당사자 보관
- 분산 인증 기관
- 기업 서명 정책 (예: 5명 임원 중 3명이 승인해야 함)

---

## 9. 보안 모델

### 9.1 EUF-CMA: 선택 메시지 공격 하의 실존적 위조 불가능성

디지털 서명의 표준 보안 정의:

**게임:**
1. 도전자가 $(pk, sk)$를 생성하고 $pk$를 공격자에게 제공한다
2. 공격자는 선택한 임의의 메시지에 대해 서명을 요청할 수 있다 (오라클 접근)
3. 공격자가 쿼리된 적 없는 $m^*$에 대해 유효한 서명 $(m^*, \sigma^*)$를 생성하면 승리한다

방식이 **EUF-CMA 안전**하다는 것은 어떤 효율적인 공격자도 무시할 수 없는 확률로 이 게임에서 승리할 수 없음을 의미한다.

### 9.2 SUF-CMA: 강한 위조 불가능성(Strong Unforgeability)

더 강력한 개념: 공격자는 이전에 서명된 메시지에 대해서도 **새로운** 유효한 서명을 생성할 수 없다. 즉, 쿼리된 임의의 메시지 $m$에 대해 공격자는 다른 유효한 서명 $\sigma' \neq \sigma$를 생성할 수 없다.

### 9.3 비교

| 방식 | EUF-CMA | SUF-CMA | 모델 |
|------|---------|---------|------|
| RSA-PKCS#1 v1.5 | 예 (가정) | 아니오 | 휴리스틱 |
| RSA-PSS | 예 | 예 | 랜덤 오라클 |
| ECDSA | 예 (가정) | 아니오 | 일반 그룹 |
| EdDSA | 예 | 예 (검사 포함) | 랜덤 오라클 |
| Schnorr | 예 | 예 | 랜덤 오라클 |

```python
def security_model_demo():
    """Demonstrate the EUF-CMA game conceptually.

    Why EUF-CMA is the minimum standard:
    - The attacker gets to see many valid signatures (realistic —
      public servers sign millions of messages)
    - Despite all these examples, the attacker still cannot forge
      a signature on any NEW message
    - This models a very powerful attacker; if a scheme is EUF-CMA
      secure, it resists all weaker attacks too
    """
    print("EUF-CMA Security Game:")
    print("=" * 50)
    print()
    print("Setup: Challenger generates (pk, sk), gives pk to Adversary")
    print()
    print("Phase 1 (Learning):")
    print("  Adversary → Challenger: 'Sign m_1'")
    print("  Challenger → Adversary: σ_1 = Sign(sk, m_1)")
    print("  Adversary → Challenger: 'Sign m_2'")
    print("  Challenger → Adversary: σ_2 = Sign(sk, m_2)")
    print("  ... (polynomially many queries)")
    print()
    print("Phase 2 (Forgery):")
    print("  Adversary outputs (m*, σ*)")
    print("  Adversary WINS if:")
    print("    1. Verify(pk, m*, σ*) = accept")
    print("    2. m* was NOT queried in Phase 1")
    print()
    print("A scheme is EUF-CMA secure if no efficient")
    print("adversary can win with non-negligible probability.")

security_model_demo()
```

---

## 10. 연습 문제

### 연습 1: RSA 서명 (기초)

$p = 61$, $q = 53$, $e = 17$인 RSA를 사용하여:
1. 개인 키 $d$를 계산하라
2. 메시지 해시 $h = 42$ (숫자로)에 서명하라
3. 공개 키를 사용하여 서명을 검증하라
4. 해시를 $h = 43$으로 수정하면 검증이 실패함을 보여라

### 연습 2: ECDSA 구현 (중급)

1. 4절의 `ECDSA` 클래스를 사용하여 10개의 다른 메시지에 서명하고 각각을 검증하라
2. 서명 및 검증당 평균 포인트 연산 횟수를 측정하라
3. 첫 번째 논스 $k$가 $r = 0$을 생성할 확률은 얼마인가? 재시도 루프가 필요한 이유는?

### 연습 3: 논스 재사용 공격 (중급)

1. 5절의 논스 재사용 공격을 구현하라
2. 위수가 100보다 큰 곡선을 사용하라
3. 동일하게 (비밀리에 선택된) 논스로 두 개의 서명을 생성하라
4. 두 서명으로부터 개인 키를 복구하라
5. 복구된 키가 새 메시지에 올바르게 서명할 수 있음을 검증하라

### 연습 4: Schnorr 다중 서명 (도전)

간단한 2-of-2 Schnorr 다중 서명 프로토콜을 구현하라:
1. 앨리스와 밥은 각각 개인 키 $d_A$와 $d_B$를 가진다
2. 공동 공개 키: $Q = Q_A + Q_B$
3. 서명: 각 당사자가 부분 서명을 기여한다
4. 결합된 서명은 $Q$에 대해 검증된다
5. 논의: 순진한 키 집계 ($Q = Q_A + Q_B$)가 왜 악성 키 공격(rogue-key attack)에 취약한가? MuSig2는 이를 어떻게 방지하는가?

### 연습 5: 결정론적 vs 무작위 논스 (도전)

1. ECDSA를 위한 RFC 6979 결정론적 논스 생성을 구현하라
2. 동일한 (키, 메시지)가 항상 같은 서명을 생성함을 검증하라
3. 다른 메시지는 다른 논스를 생성함을 검증하라
4. Ed25519의 접근 방식과 비교하라: 어떻게 유사한가? 어떻게 다른가?
5. 논의: 결정론적 서명 방식이 SUF-CMA 안전할 수 있는가? 이유는?

---

**이전**: [타원곡선 암호](./06_Elliptic_Curve_Cryptography.md) | **다음**: [키 교환](./08_Key_Exchange.md)

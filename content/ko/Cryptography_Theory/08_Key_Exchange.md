# 레슨 8: 키 교환

**이전**: [디지털 서명](./07_Digital_Signatures.md) | **다음**: [PKI와 인증서](./09_PKI_and_Certificates.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 키 배포 문제(key distribution problem)를 설명하고 그것이 왜 안전한 통신의 근본 문제인지 이해한다
2. 이산 로그(discrete logarithm) 가정으로부터 Diffie-Hellman 키 교환 프로토콜을 유도한다
3. 고전적 DH와 타원곡선 DH(ECDH) 모두를 Python으로 구현한다
4. 인증된 키 교환과 인증되지 않은 키 교환을 비교하고 중간자(man-in-the-middle) 취약점을 파악한다
5. 전방 비밀성(forward secrecy)을 설명하고 왜 임시(ephemeral) 키가 현대 프로토콜에 중요한지 이해한다
6. Signal 프로토콜의 X3DH 핸드셰이크와 보안 속성을 설명한다
7. 공유 비밀로부터 사용 가능한 키를 만들기 위해 키 유도 함수(HKDF)를 적용한다

---

지금까지 공부한 모든 암호 시스템 — 대칭 암호(레슨 3), 공개 키 암호(레슨 5), 디지털 서명(레슨 7) — 은 통신 당사자가 이미 어떤 비밀을 공유하거나 서로의 공개 키를 알고 있다고 가정한다. 그런데 도청자가 모든 비트를 듣고 있는 안전하지 않은 채널을 통해 통신하는 두 낯선 사람이 어떻게 비밀 자체를 전송하지 않고 공유 비밀에 합의할 수 있을까? 이것이 **키 배포 문제(key distribution problem)**이며, 그 우아한 해결책인 Diffie-Hellman 프로토콜은 암호학 역사상 가장 중요한 발명이라 할 수 있다. 이것은 현대 안전한 통신의 전체 구조를 가능하게 만들었다.

## 목차

1. [키 배포 문제](#1-키-배포-문제)
2. [Diffie-Hellman 키 교환](#2-diffie-hellman-키-교환)
3. [타원곡선 Diffie-Hellman (ECDH)](#3-타원곡선-diffie-hellman-ecdh)
4. [중간자 공격과 인증된 키 교환](#4-중간자-공격과-인증된-키-교환)
5. [전방 비밀성과 임시 키](#5-전방-비밀성과-임시-키)
6. [Signal 프로토콜과 X3DH](#6-signal-프로토콜과-x3dh)
7. [키 유도 함수](#7-키-유도-함수)
8. [요약](#8-요약)
9. [연습 문제](#9-연습-문제)

---

## 1. 키 배포 문제

### 1.1 근본적인 딜레마

앨리스가 AES-256(레슨 3)을 사용하여 밥에게 암호화된 메시지를 보내려 한다고 가정하자. 그녀는 앨리스와 밥만 아는 256비트 키가 필요하다. 하지만 어떻게 이 키를 밥에게 전달하는가?

- **물리적 전달(Physical courier)**: 안전하지만 규모가 커지면 비실용적이다. 한 번도 방문하지 않은 웹사이트와 어떻게 키를 교환하는가?
- **신뢰할 수 있는 제3자(Kerberos 모델)**: 모두가 신뢰하는 중앙 서버가 필요하다 — 단일 실패 지점.
- **사전 공유 키(Pre-shared keys)**: 소규모 그룹에서는 작동하지만 $O(n^2)$으로 확장된다. 1,000명의 사용자 네트워크에는 $\frac{1000 \times 999}{2} = 499,500$개의 고유 키가 필요하다.

### 1.2 혁신적인 아이디어

1976년, 휘트필드 디피(Whitfield Diffie)와 마틴 헬먼(Martin Hellman)은 "암호학의 새로운 방향(New Directions in Cryptography)"을 발표하여, 두 당사자가 공개 채널을 통해 비밀 자체를 전송하지 않고 공유 비밀에 합의할 수 있다고 제안했다. 이는 혁명적이었다 — 정보 교환에 대한 직관을 위반하는 것처럼 보였다.

> **비유:** Diffie-Hellman은 페인트 색을 섞는 것과 같다. 앨리스와 밥은 공통 베이스 색에 동의한다(공개). 앨리스는 자신의 비밀 색을 추가하여 혼합물을 밥에게 보내고; 밥은 자신의 비밀 색을 추가하여 혼합물을 앨리스에게 보낸다. 이제 각자는 상대방의 혼합물에 자신의 비밀 색을 추가한다. 둘 다 같은 최종 색에 도달한다 — 하지만 두 중간 혼합물만 본 도청자는 개별 비밀 색을 복원하기 위해 혼합물을 역으로 분리할 수 없다.

---

## 2. Diffie-Hellman 키 교환

### 2.1 수학적 기반

DH의 보안은 **이산 로그 문제(Discrete Logarithm Problem, DLP)**에 기반한다:

소수 $p$, $\mathbb{Z}_p^*$의 생성원 $g$, 그리고 값 $A = g^a \bmod p$가 주어졌을 때, 큰 $p$에 대해 $a$를 찾는 것은 계산적으로 불가능하다.

관련된 **결정론적 Diffie-Hellman(Decisional Diffie-Hellman, DDH) 가정**은 다음을 말한다:

$$
(g^a, g^b, g^{ab}) \approx_c (g^a, g^b, g^r)
$$

여기서 $r$은 무작위이다. 즉, $g^a$와 $g^b$만 아는 사람에게는 $g^{ab}$가 무작위처럼 보인다.

### 2.2 프로토콜

**공개 매개변수**: 큰 소수 $p$와 $\mathbb{Z}_p^*$의 생성원 $g$.

| 단계 | 앨리스 | 채널 | 밥 |
|------|-------|------|---|
| 1 | 무작위 $a \in \{2, \ldots, p-2\}$ 선택 | | 무작위 $b \in \{2, \ldots, p-2\}$ 선택 |
| 2 | $A = g^a \bmod p$ 계산 | $A \longrightarrow$ | |
| 3 | | $\longleftarrow B$ | $B = g^b \bmod p$ 계산 |
| 4 | $s = B^a \bmod p$ 계산 | | $s = A^b \bmod p$ 계산 |

**정확성**: 다음이 성립하므로 둘 다 같은 값을 계산한다:

$$
B^a = (g^b)^a = g^{ab} = (g^a)^b = A^b \pmod{p}
$$

### 2.3 구현

```python
"""
Diffie-Hellman Key Exchange — Educational Implementation

Why we use Python's built-in pow(base, exp, mod):
  Python's three-argument pow() uses fast modular exponentiation
  (square-and-multiply), making it efficient even for large numbers.
"""

import secrets
import hashlib


def generate_dh_parameters():
    """
    Generate DH parameters (p, g).

    In production, use well-known groups like RFC 3526 Group 14 (2048-bit).
    Here we use a small safe prime for demonstration.

    Why a safe prime? If p = 2q + 1 where q is also prime, the subgroup
    of order q has no small subgroups to leak information through.
    """
    # RFC 3526 Group 14 (2048-bit MODP) — truncated for readability
    # In practice, you'd use the full value
    p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
    g = 2
    return p, g


def dh_key_exchange():
    """Simulate a complete Diffie-Hellman key exchange."""
    p, g = generate_dh_parameters()

    # Alice's side
    a = secrets.randbelow(p - 2) + 2  # Why +2: avoid 0 and 1
    A = pow(g, a, p)                   # Alice's public value

    # Bob's side
    b = secrets.randbelow(p - 2) + 2
    B = pow(g, b, p)                   # Bob's public value

    # Key agreement
    # Why both computations yield the same result:
    # Alice computes B^a mod p = g^(ba) mod p
    # Bob computes A^b mod p = g^(ab) mod p
    # Since ab = ba, the shared secret is identical.
    shared_secret_alice = pow(B, a, p)
    shared_secret_bob = pow(A, b, p)

    assert shared_secret_alice == shared_secret_bob, "Key exchange failed!"

    # Why we hash the shared secret:
    # The raw DH output has mathematical structure (it's an element of Z_p*).
    # Hashing extracts a uniformly random key from this structured value.
    key = hashlib.sha256(
        shared_secret_alice.to_bytes(256, 'big')
    ).digest()

    print(f"DH parameters: p has {p.bit_length()} bits, g = {g}")
    print(f"Alice's public value A: {hex(A)[:20]}...")
    print(f"Bob's public value B:   {hex(B)[:20]}...")
    print(f"Shared secret matches:  {shared_secret_alice == shared_secret_bob}")
    print(f"Derived 256-bit key:    {key.hex()[:32]}...")

    return key


if __name__ == "__main__":
    dh_key_exchange()
```

### 2.4 매개변수 선택

DH 매개변수를 부주의하게 선택하면 치명적인 결과를 초래할 수 있다:

| 매개변수 | 최소값 | 권장값 | 이유 |
|---------|-------|-------|------|
| $p$ 비트 길이 | 1024 | 2048+ | 1024비트 DLP는 국가 수준의 공격자에게 가능 (Logjam 공격) |
| $g$ | 2 또는 5 | 2 | 작은 생성원은 괜찮음; 보안은 $p$에 의존 |
| 개인 지수 $a$ | 160비트 | 256비트 | 목표 보안 레벨의 최소 두 배 이상이어야 함 |

> **경고**: 많은 구현이 모든 연결에 동일한 DH 매개변수를 재사용한다. 2015년 Logjam 공격은 일반적으로 사용되는 1024비트 소수에 대한 이산 로그 테이블을 사전 계산하면 TLS 서버의 80%에 대한 연결을 깰 수 있음을 보여주었다.

---

## 3. 타원곡선 Diffie-Hellman (ECDH)

### 3.1 타원곡선을 사용하는 이유

고전적 DH는 적절한 보안을 위해 2048비트 이상의 소수가 필요하다. 타원곡선 DH는 훨씬 작은 키로 동등한 보안을 달성한다:

| 보안 레벨 | DH 키 크기 | ECDH 키 크기 | 비율 |
|----------|-----------|-------------|------|
| 128비트 | 3072비트 | 256비트 | 12:1 |
| 192비트 | 7680비트 | 384비트 | 20:1 |
| 256비트 | 15360비트 | 512비트 | 30:1 |

### 3.2 ECDH 프로토콜

$\mathbb{Z}_p^*$에서 작업하는 대신, 위수(order) $n$의 기저점 $G$를 가진 $\mathbb{F}_p$ 위의 타원곡선 $E$에서 작업한다 (ECC 기초는 레슨 6 참조).

| 단계 | 앨리스 | 밥 |
|------|-------|---|
| 1 | 무작위 $a \in \{1, \ldots, n-1\}$ 선택 | 무작위 $b \in \{1, \ldots, n-1\}$ 선택 |
| 2 | $A = aG$ 계산 (점 곱셈) | $B = bG$ 계산 |
| 3 | $A$를 밥에게 전송 | $B$를 앨리스에게 전송 |
| 4 | $S = aB = a(bG)$ 계산 | $S = bA = b(aG)$ 계산 |

점 곱셈이 교환 가능하므로 둘 다 같은 점 $S = abG$에 도달한다. 공유 비밀은 일반적으로 $S$의 $x$좌표이다.

### 3.3 `cryptography` 라이브러리를 사용한 ECDH

```python
"""
ECDH Key Exchange using the cryptography library.

Why SECP256R1 (P-256)? It's the NIST-recommended curve for 128-bit
security, widely supported in TLS, and hardware-accelerated on many CPUs.
Curve25519 is the modern alternative preferred by many cryptographers
for its simpler, constant-time implementation.
"""

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def ecdh_key_exchange():
    """Demonstrate ECDH using NIST P-256 curve."""

    # Alice generates her key pair
    # Why ec.SECP256R1(): 256-bit prime-order curve, equivalent to 3072-bit RSA
    alice_private = ec.generate_private_key(ec.SECP256R1())
    alice_public = alice_private.public_key()

    # Bob generates his key pair
    bob_private = ec.generate_private_key(ec.SECP256R1())
    bob_public = bob_private.public_key()

    # Key agreement
    # Why ECDH().exchange() instead of manual math?
    # The library handles point validation (rejecting invalid curve points),
    # constant-time scalar multiplication, and proper encoding.
    alice_shared = alice_private.exchange(ec.ECDH(), bob_public)
    bob_shared = bob_private.exchange(ec.ECDH(), alice_public)

    assert alice_shared == bob_shared

    # Why HKDF? The raw ECDH output (x-coordinate of the shared point)
    # is not uniformly random — it's biased by the curve equation.
    # HKDF extracts and expands it into a proper symmetric key.
    alice_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,            # 256-bit key
        salt=None,            # Optional but recommended in practice
        info=b"ecdh-demo",    # Context separation (binds key to purpose)
    ).derive(alice_shared)

    print(f"ECDH shared secret (hex): {alice_shared.hex()[:32]}...")
    print(f"Derived key (hex):        {alice_key.hex()}")
    return alice_key


def ecdh_curve25519():
    """ECDH using Curve25519 (modern, preferred curve)."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    # Why Curve25519? Designed by Daniel Bernstein with:
    # - Built-in resistance to timing attacks (constant-time by design)
    # - No need for point validation (every 32-byte string is valid)
    # - Faster than P-256 in software implementations
    alice_private = X25519PrivateKey.generate()
    alice_public = alice_private.public_key()

    bob_private = X25519PrivateKey.generate()
    bob_public = bob_private.public_key()

    alice_shared = alice_private.exchange(bob_public)
    bob_shared = bob_private.exchange(alice_public)

    assert alice_shared == bob_shared
    print(f"X25519 shared secret: {alice_shared.hex()[:32]}...")
    return alice_shared


if __name__ == "__main__":
    print("=== ECDH with P-256 ===")
    ecdh_key_exchange()
    print("\n=== ECDH with Curve25519 ===")
    ecdh_curve25519()
```

---

## 4. 중간자 공격과 인증된 키 교환

### 4.1 순수 DH에 대한 중간자(MitM) 공격

순수 Diffie-Hellman은 어느 당사자도 상대방을 인증하지 않기 때문에 **중간자(man-in-the-middle, MitM) 공격**에 취약하다:

```
Alice                    Eve (attacker)                Bob
  |                          |                          |
  |--- A = g^a mod p ------->|                          |
  |                          |--- E1 = g^e1 mod p ----->|
  |                          |<--- B = g^b mod p -------|
  |<--- E2 = g^e2 mod p ----|                          |
  |                          |                          |
  s1 = E2^a (Alice-Eve)     s1 = A^e2, s2 = B^e1      s2 = E1^b (Eve-Bob)
```

이브는 앨리스와 밥 각각과 별도의 공유 비밀을 구성한다. 그녀는 모든 메시지를 복호화하고, 읽고, 재암호화하여 전달한다. 앨리스도 밥도 감청을 감지하지 못한다.

### 4.2 STS(Station-to-Station) 프로토콜

STS 프로토콜은 DH 교환 값에 서명함으로써 인증을 추가한다:

1. 앨리스가 $A = g^a \bmod p$를 보낸다
2. 밥이 $B = g^b \bmod p$와 함께 $K = g^{ab}$로 암호화된 $\text{Sign}_{B}(B \| A)$를 보낸다
3. 앨리스는 밥의 서명을 검증한 후, $K$로 암호화된 $\text{Sign}_{A}(A \| B)$를 보낸다

이는 신원(서명을 통해)을 교환 값에 결합하여, 이브가 자신의 값을 대체하는 것을 방지한다.

### 4.3 기타 인증된 키 교환 프로토콜

| 프로토콜 | 인증 방법 | 사용처 |
|---------|---------|-------|
| STS | 디지털 서명 | IPsec (IKE) |
| SIGMA | 서명 + MAC | IKEv2 |
| HMQV | 암묵적 (서명 불필요) | 학술, 일부 스마트카드 |
| TLS 1.3 | 서버 인증서 + 선택적 클라이언트 인증서 | HTTPS, 모든 웹 브라우저 |
| Noise 프레임워크 | 다양한 패턴 (XX, IK, NK 등) | WireGuard, Signal |

---

## 5. 전방 비밀성과 임시 키

### 5.1 장기 키만으로는 부족한 이유

시나리오를 생각해 보자: 서버가 몇 년 동안 정적 RSA 키 쌍을 사용한다. 공격자가 모든 암호화된 트래픽을 기록한다. 몇 년 후, 서버가 침해되어 개인 키가 도난당한다. 이제 공격자는 **이전에 기록된 모든 트래픽**을 복호화할 수 있다.

> **비유:** 키 교환에 정적 키를 사용하는 것은 보낸 모든 편지를 잠그기 위해 동일한 마스터 키를 사용하는 것과 같다. 누군가 마스터 키를 훔치면, 과거, 현재, 미래의 모든 편지를 열 수 있다.

### 5.2 전방 비밀성(Forward Secrecy)

**전방 비밀성(Forward Secrecy, FS)** (또는 **완전 전방 비밀성(perfect forward secrecy, PFS)**이라고도 함)은 장기 키가 노출되더라도 과거 세션 키는 노출되지 않음을 보장한다.

메커니즘은 간단하다: 각 세션마다 **임시(한번 사용하는) DH 키**를 사용하는 것이다.

```
Session 1: Alice generates a_1, Bob generates b_1 → K_1 = g^(a_1 * b_1)
Session 2: Alice generates a_2, Bob generates b_2 → K_2 = g^(a_2 * b_2)
...
```

각 세션 후, $a_i$와 $b_i$는 안전하게 삭제된다. 공격자가 나중에 앨리스의 장기 서명 키를 획득하더라도, $a_1$이나 $b_1$을 복구할 수 없으므로 $K_1$을 계산할 수 없다.

### 5.3 TLS에서의 임시 DH (DHE / ECDHE)

TLS 1.3에서는 **모든** 키 교환이 임시 키를 사용한다 (정적 RSA 키 교환 모드는 완전히 제거되었다). 명명 규칙:

- **DH**: 정적 Diffie-Hellman (더 이상 사용되지 않음)
- **DHE**: 임시 Diffie-Hellman (전방 비밀성)
- **ECDHE**: 임시 타원곡선 DH (전방 비밀성, 더 작은 키)

```python
"""
Demonstrating ephemeral key exchange sessions.

Why delete private keys after use? Each session's privacy depends
on the secrecy of its ephemeral key. Once the shared secret is
derived, the private key serves no further purpose and becomes a
liability if retained.
"""

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def ephemeral_session(session_id: int) -> bytes:
    """Simulate one ephemeral key exchange session."""
    # Generate fresh keys — never reused
    alice_eph = X25519PrivateKey.generate()
    bob_eph = X25519PrivateKey.generate()

    alice_pub = alice_eph.public_key()
    bob_pub = bob_eph.public_key()

    # Key agreement
    shared = alice_eph.exchange(bob_pub)

    # Derive session key
    session_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"session-{session_id}".encode(),
    ).derive(shared)

    # In a real system, alice_eph and bob_eph would be securely
    # zeroed from memory here. Python doesn't guarantee this,
    # but C/Rust implementations use explicit zeroization.
    del alice_eph, bob_eph

    return session_key


# Each session produces an independent key
for i in range(3):
    key = ephemeral_session(i)
    print(f"Session {i} key: {key.hex()[:16]}...")
    # Compromising one session key reveals nothing about others
```

### 5.4 스노든 이후의 영향

2013년 스노든 폭로는 정보기관들이 키가 노출될 경우 나중에 복호화할 목적으로 암호화된 트래픽을 대규모로 기록하고 있었음을 보여주었다. 이로 인해 전방 비밀성이 시급한 과제가 되었다:

- **2013년 이전**: HTTPS 사이트의 약 30%가 전방 비밀성 지원
- **2013년 이후**: 90% 이상 채택, TLS 1.3에서 필수화
- **"지금 수집하고 나중에 복호화(Harvest now, decrypt later)"**: 이 위협은 양자 내성 암호(post-quantum) 마이그레이션(레슨 11)에서도 여전히 관련이 있다

---

## 6. Signal 프로토콜과 X3DH

### 6.1 비동기 키 교환 문제

표준 DH는 두 당사자가 동시에 온라인 상태여야 한다. 하지만 메시징 앱은 비동기적으로 작동해야 한다 — 앨리스는 밥의 전화기가 꺼져 있는 동안 메시지를 보낸다. Signal 프로토콜은 **X3DH(Extended Triple Diffie-Hellman, 확장된 3중 Diffie-Hellman)**으로 이 문제를 해결한다.

### 6.2 X3DH 키 합의

밥은 서버에 여러 종류의 키를 사전 게시한다:

- **IK_B**: 신원 키(Identity key, 장기, 인증용)
- **SPK_B**: 서명된 사전 키(Signed pre-key, 중기, 주기적으로 교체)
- **OPK_B**: 일회용 사전 키(One-time pre-keys, 각각 정확히 한 번 사용)

앨리스가 오프라인 상태의 밥에게 메시지를 보내려 할 때:

$$
\text{SK} = \text{KDF}(\text{DH}(IK_A, SPK_B) \| \text{DH}(EK_A, IK_B) \| \text{DH}(EK_A, SPK_B) \| \text{DH}(EK_A, OPK_B))
$$

여기서 $EK_A$는 이 세션을 위해 앨리스가 생성한 임시 키이다.

### 6.3 네 번의 DH 연산이 필요한 이유

각 DH 계산은 서로 다른 보안 속성을 제공한다:

| DH 쌍 | 속성 |
|-------|------|
| $\text{DH}(IK_A, SPK_B)$ | 상호 인증 (두 장기 키가 모두 관여) |
| $\text{DH}(EK_A, IK_B)$ | 앨리스의 IK가 노출되더라도 과거 세션은 안전 |
| $\text{DH}(EK_A, SPK_B)$ | 전방 비밀성 (사용 후 임시 키 삭제) |
| $\text{DH}(EK_A, OPK_B)$ | 재전송 공격 방지 (일회용 키 소비) |

### 6.4 이중 래칫(Double Ratchet) 개요

X3DH가 초기 공유 비밀을 구성한 후, **이중 래칫(Double Ratchet) 알고리즘**은 각 메시지가 고유한 키를 사용하도록 보장한다. 다음 두 가지를 결합한다:

1. **DH 래칫**: 모든 메시지 왕복마다 새로운 DH 교환 (각 메시지에 대한 전방 비밀성 제공)
2. **대칭 래칫**: 단일 DH 라운드 내의 메시지를 위한 해시 기반 키 체인 (순서 및 효율성 제공)

이중 래칫은 레슨 14(응용 암호 프로토콜)에서 자세히 살펴본다.

---

## 7. 키 유도 함수

### 7.1 공유 비밀을 직접 사용하지 않는 이유

DH/ECDH의 원시 출력은 수학적 구조를 가지며 균일하게 무작위하지 않다. **키 유도 함수(Key Derivation Function, KDF)**는 이를 하나 이상의 암호학적으로 강력한 키로 변환한다.

### 7.2 HKDF (HMAC 기반 KDF)

HKDF(RFC 5869)는 두 단계로 작동한다:

1. **추출(Extract)**: 입력 키 재료를 의사 무작위 키로 압축
   $$\text{PRK} = \text{HMAC-Hash}(\text{salt}, \text{IKM})$$

2. **확장(Expand)**: 원하는 길이의 출력 키 재료 생성
   $$\text{OKM} = T_1 \| T_2 \| \ldots$$
   여기서 $T_i = \text{HMAC-Hash}(\text{PRK}, T_{i-1} \| \text{info} \| i)$

```python
"""
HKDF: Extracting multiple keys from a single shared secret.

Why separate extract and expand? The extract step handles potentially
biased or non-uniform input (like a DH shared secret). The expand step
then generates as many independent-looking keys as needed from the
extracted PRK.
"""

import hmac
import hashlib
import math


def hkdf_extract(salt: bytes, ikm: bytes, hash_algo=hashlib.sha256) -> bytes:
    """
    Extract: condense input keying material into a PRK.

    Why salt? It ensures that even if the IKM has some structure,
    the PRK is uniformly distributed. If no salt is available,
    a string of zeros (hash length) is used.
    """
    if not salt:
        salt = b'\x00' * hash_algo().digest_size
    return hmac.new(salt, ikm, hash_algo).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int,
                hash_algo=hashlib.sha256) -> bytes:
    """
    Expand: derive output keying material from PRK.

    Why info? It binds the derived key to a specific context.
    Using different info values from the same PRK produces
    independent keys (e.g., one for encryption, one for MAC).
    """
    hash_len = hash_algo().digest_size
    n = math.ceil(length / hash_len)

    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hash_algo).digest()
        okm += t

    return okm[:length]


def hkdf(ikm: bytes, length: int, salt: bytes = b"",
          info: bytes = b"") -> bytes:
    """Full HKDF: extract then expand."""
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)


# Example: derive encryption key and MAC key from one DH secret
dh_shared_secret = bytes.fromhex(
    "a3f2b8c1d4e5f607182930a1b2c3d4e5f60718293041526374859607a8b9c0d1"
)

enc_key = hkdf(dh_shared_secret, 32, info=b"encryption")
mac_key = hkdf(dh_shared_secret, 32, info=b"authentication")

print(f"Encryption key: {enc_key.hex()}")
print(f"MAC key:        {mac_key.hex()}")
# These keys are cryptographically independent despite sharing a source
assert enc_key != mac_key
```

### 7.3 기타 KDF

| KDF | 기반 | 사용처 |
|-----|------|-------|
| HKDF | HMAC | TLS 1.3, Signal, Noise |
| NIST SP 800-108 | HMAC 또는 CMAC | 정부 시스템 |
| PBKDF2 | HMAC (반복) | 패스워드 해싱 (하지만 Argon2 권장) |
| X963KDF | 해시 함수 | ECIES (레슨 6) |

> **참고**: HKDF는 DH 공유 비밀과 같은 높은 엔트로피 입력을 위해 설계되었다. 패스워드(낮은 엔트로피)의 경우에는 Argon2나 scrypt를 사용하라 — 이들은 의도적으로 시간과 메모리를 낭비하여 무차별 대입(brute force)에 저항한다.

---

## 8. 요약

| 개념 | 핵심 내용 |
|------|---------|
| 키 배포 문제 | DH 없이는 안전하지 않은 채널을 통해 키를 안전하게 전송할 수 없음 |
| Diffie-Hellman | $g^{ab} \bmod p$ — 우아하지만 인증이 필요함 |
| ECDH | 타원곡선에서의 동일한 아이디어; 256비트 키가 3072비트 DH와 동등 |
| 중간자 공격 | 순수 DH는 취약; 인증(STS, 인증서)이 필수 |
| 전방 비밀성 | 임시 키로 장기 키가 유출되더라도 과거 세션이 안전하게 유지 |
| X3DH | 비동기 인증 키 교환 (Signal 프로토콜) |
| HKDF | 공유 비밀로부터 키를 유도하기 위한 추출-후-확장 패러다임 |

---

## 9. 연습 문제

### 연습 1: 손으로 계산하는 소규모 DH (개념)

$p = 23$, $g = 5$, $a = 6$, $b = 15$가 주어졌을 때:

1. 앨리스의 공개 값 $A = g^a \bmod p$를 계산하라
2. 밥의 공개 값 $B = g^b \bmod p$를 계산하라
3. 양쪽에서 공유 비밀을 계산하고 일치함을 검증하라
4. 도청자 이브로서, $A$와 $B$는 보이지만 $a$나 $b$는 보이지 않는다. 공유 비밀을 찾으려고 시도하라. 무차별 대입에 몇 번의 연산이 필요한가? $p$의 크기에 따라 어떻게 확장되는가?

### 연습 2: DH 매개변수 검증 (코딩)

다음을 확인하는 Python 함수 `validate_dh_params(p, g, A)`를 작성하라:
- $p$가 소수인지
- $g$가 $\mathbb{Z}_p^*$의 큰 부분군의 생성원인지
- $A$가 $[2, p-2]$ 범위에 있는지 (사소한 값 0, 1, $p-1$ 거부)
- 안전 소수에 대해 $A^q \equiv 1 \pmod{p}$ (여기서 $q = (p-1)/2$)

### 연습 3: 전방 비밀성 시뮬레이션 (코딩)

두 가지 모드로 채팅 시뮬레이션을 구현하라:
1. **정적 키 모드**: 모든 메시지에 동일한 DH 키 쌍 재사용
2. **임시 키 모드**: 각 메시지마다 새 키 생성

한 세션 키를 노출시키는 공격자를 시뮬레이션하라. 정적 모드에서는 모든 메시지가 노출되지만, 임시 모드에서는 한 메시지만 노출됨을 보여라.

### 연습 4: HKDF 테스트 벡터 (코딩)

라이브러리를 사용하지 않고 처음부터 HKDF를 구현하고, RFC 5869 부록 A의 테스트 벡터로 구현을 검증하라. 세 가지 테스트 케이스를 모두 통과해야 한다.

### 연습 5: X3DH 단계별 진행 (도전)

단순화된 X3DH 프로토콜을 구현하라:
1. 밥이 $IK_B$, $SPK_B$ ($IK_B$로 서명됨), 그리고 $OPK_B$ 집합을 게시한다
2. 앨리스가 네 번의 DH 연산을 수행하고 공유 키를 유도한다
3. 앨리스가 이 키로 초기 메시지를 암호화한다
4. 밥이 수신하여 복호화한다

보안보다 정확성에 집중하라 (프로덕션 구현에서 필요한 일부 확인은 생략해도 됨). 앨리스와 밥이 같은 키를 유도함을 검증하라.

---

**이전**: [디지털 서명](./07_Digital_Signatures.md) | **다음**: [PKI와 인증서](./09_PKI_and_Certificates.md)

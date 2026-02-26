# 레슨 14: 응용 암호 프로토콜(Applied Cryptographic Protocols)

**이전**: [← 동형 암호](13_Homomorphic_Encryption.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 완전한 TLS 1.3 핸드셰이크(Handshake)를 추적하고 각 암호 연산을 설명한다
2. Signal 프로토콜의 이중 래칫(Double Ratchet) 알고리즘과 그 보안 특성을 설명한다
3. 안전한 다자간 계산(MPC, Secure Multi-Party Computation)을 설명하고 Yao의 혼돈 회로(Garbled Circuits)를 개념적으로 구현한다
4. 샤미르 비밀 공유(Shamir's Secret Sharing)를 구현하고 임계값(Threshold) 특성을 이해한다
5. 커밋먼트 스킴(Commitment Scheme)을 설계하고 암호 프로토콜에서의 역할을 설명한다
6. 불확지 전송(Oblivious Transfer)과 MPC에서의 기초적 역할을 설명한다
7. 실제 프로토콜이 앞선 레슨의 기본 요소들을 어떻게 결합하는지 분석한다

---

이 과정을 통해 우리는 개별 암호 기본 요소들을 학습했습니다 — 대칭 암호(레슨 3), 해시 함수(레슨 4), 공개 키 암호화(레슨 5), 디지털 서명(레슨 7), 키 교환(레슨 8), 그리고 영지식 증명(ZKP, 레슨 12)과 동형 암호(레슨 13) 같은 고급 구조들. 실제로는 이 기본 요소들 중 어느 것도 독립적으로 동작하지 않습니다. 실세계의 보안은 기본 요소들을 **결합**하여 프로토콜로 만듦으로써 달성됩니다 — 한 기본 요소의 출력이 다른 기본 요소의 입력이 되는, 신중하게 조율된 연산의 순서. 이 결합 과정에서 단 하나의 설계 결함이 전체 시스템을 안전하지 않게 만들 수 있으며, 개별 기본 요소들이 아무리 강력해도 마찬가지입니다.

이 레슨에서는 오늘날 사용되는 가장 중요한 암호 프로토콜들을 살펴보고, 우리가 학습한 기본 요소들이 어떻게 함께 엮이는지 보여줍니다.

> **비유:** Signal 프로토콜은 릴레이를 통해 수업 중에 쪽지를 전달하는 것과 같습니다. 각 쪽지는 다르게 암호화되어 있어, 누군가가 하나를 가로채더라도 과거나 미래의 쪽지를 읽을 수 없습니다. 이것이 전방 비밀성(Forward Secrecy)과 미래 비밀성(Future Secrecy, 사후 침해 보안(Post-Compromise Security)이라고도 함)이 결합된 것입니다 — 자기 파괴되는 메시지 체인의 암호학적 등가물.

## 목차

1. [TLS 1.3: 인터넷의 보안 계층](#1-tls-13-인터넷의-보안-계층)
2. [Signal 프로토콜](#2-signal-프로토콜)
3. [안전한 다자간 계산(MPC)](#3-안전한-다자간-계산mpc)
4. [샤미르 비밀 공유](#4-샤미르-비밀-공유)
5. [커밋먼트 스킴](#5-커밋먼트-스킴)
6. [불확지 전송](#6-불확지-전송)
7. [임계값 서명](#7-임계값-서명)
8. [프라이빗 정보 검색](#8-프라이빗-정보-검색)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)

---

## 1. TLS 1.3: 인터넷의 보안 계층

### 1.1 TLS란 무엇인가?

전송 계층 보안(TLS, Transport Layer Security)은 HTTPS, 이메일(SMTP/IMAP with STARTTLS), DNS-over-TLS 등 사실상 모든 인터넷 통신을 보호합니다. TLS 1.3(RFC 8446, 2018년 발표)은 레거시 요소를 제거하고 보안과 성능을 모두 개선한 대대적인 재설계입니다.

### 1.2 TLS 1.3 vs. TLS 1.2

| 특성 | TLS 1.2 | TLS 1.3 |
|------|---------|---------|
| 첫 데이터까지의 왕복 횟수 | 2 RTT | **1 RTT** (0-RTT 가능) |
| 키 교환 | RSA 또는 DHE/ECDHE | **ECDHE만** (전방 비밀성 필수) |
| 암호 스위트(Cipher Suites) | 37개 이상 (다수 불안전) | **5개** (모두 인증된 암호화) |
| 압축 | 지원 | **제거** (CRIME 공격 방지) |
| 재협상(Renegotiation) | 지원 | **제거** (보안 문제) |
| 0-RTT | 불가 | **선택적** (재전송 위험 있음) |

### 1.3 전체 핸드셰이크(1-RTT)

```
Client                                              Server

ClientHello
  + key_share (ECDHE public key)
  + supported_versions (TLS 1.3)
  + cipher_suites
  + signature_algorithms
  --------------------------------------------->

                                              ServerHello
                                  + key_share (ECDHE public key)
                           {EncryptedExtensions}
                           {CertificateRequest*}
                           {Certificate}
                           {CertificateVerify}
                           {Finished}
  <---------------------------------------------

{Certificate*}
{CertificateVerify*}
{Finished}
  --------------------------------------------->

[Application Data]            <---->         [Application Data]
```

`{...}` 안의 항목들은 ECDHE 공유 비밀에서 파생된 핸드셰이크 키로 암호화됩니다.

### 1.4 TLS 1.3의 키 파생

TLS 1.3은 HKDF(레슨 8)를 기반으로 한 정교한 키 스케줄을 사용합니다:

```
                  0
                  |
                  v
    PSK ->  HKDF-Extract = Early Secret
                  |
                  +-> Derive-Secret(., "ext binder" | "res binder", "")
                  |                     = binder_key
                  +-> Derive-Secret(., "c e traffic", ClientHello)
                  |                     = client_early_traffic_secret
                  +-> Derive-Secret(., "e exp master", ClientHello)
                  |                     = early_exporter_master_secret
                  v
            Derive-Secret(., "derived", "")
                  |
                  v
  (EC)DHE -> HKDF-Extract = Handshake Secret
                  |
                  +-> Derive-Secret(., "c hs traffic", ClientHello..ServerHello)
                  |                     = client_handshake_traffic_secret
                  +-> Derive-Secret(., "s hs traffic", ClientHello..ServerHello)
                  |                     = server_handshake_traffic_secret
                  v
            Derive-Secret(., "derived", "")
                  |
                  v
       0 -> HKDF-Extract = Master Secret
                  |
                  +-> Derive-Secret(., "c ap traffic", ClientHello..Server Finished)
                  |                     = client_application_traffic_secret
                  +-> Derive-Secret(., "s ap traffic", ClientHello..Server Finished)
                                        = server_application_traffic_secret
```

### 1.5 이 설계의 이유

- **방향별 별도 키**: 클라이언트→서버와 서버→클라이언트가 다른 키를 사용하여 반사 공격(Reflection Attack)을 방지
- **핸드셰이크 키 vs. 애플리케이션 키 분리**: 이를 분리하면 키 노출로 인한 피해를 제한
- **트랜스크립트 바인딩(Transcript Binding)**: 파생된 각 키에 이전 모든 메시지의 해시가 포함되어 메시지 변조나 순서 조작을 방지
- **PSK 지원**: 사전 공유 키(Pre-Shared Key)로 0-RTT 재개 가능 (단, 재전송 취약성 비용 발생)

### 1.6 0-RTT 재개

0-RTT 모드에서 클라이언트는 첫 번째 메시지에서 (서버 응답 전에) 애플리케이션 데이터를 전송합니다:

```
Client                                              Server
ClientHello + early_data (encrypted with PSK)
  --------------------------------------------->
```

**위험**: 0-RTT 데이터는 ClientHello를 캡처한 공격자에 의해 **재전송(Replay)**될 수 있습니다. TLS 1.3은 서버가 0-RTT 데이터를 잠재적으로 재전송된 것으로 처리해야 한다고 명시합니다 — 멱등(Idempotent) 연산(예: GET 요청, POST는 제외)에만 사용해야 합니다.

---

## 2. Signal 프로토콜

### 2.1 개요

Signal 프로토콜(Signal, WhatsApp, Facebook Messenger, Google Messages에서 사용)은 다음을 제공합니다:
- **종단 간 암호화(End-to-End Encryption)**: 발신자와 수신자만 메시지를 읽을 수 있음
- **전방 비밀성(Forward Secrecy)**: 현재 키 침해가 과거 메시지를 노출하지 않음
- **사후 침해 보안(Post-Compromise Security)** (미래 비밀성): 침해 후에도 새 키를 교환하면 보안이 복원됨
- **비동기 키 교환**: 수신자가 오프라인일 때도 작동 (X3DH, 레슨 8을 통해)

### 2.2 이중 래칫(Double Ratchet) 알고리즘

X3DH(레슨 8)가 초기 공유 비밀을 수립한 후, 이중 래칫은 두 개의 연동된 "래칫"을 통해 지속적인 메시지 보안을 유지합니다:

**1. DH 래칫** (비대칭):
- 각 당사자는 메시지 왕복마다 새 임시 DH 키 쌍을 생성
- 새로운 DH 공유 비밀이 계산되어 메시지 수준의 전방 비밀성 제공

**2. 대칭 래칫** (KDF 체인):
- DH 래칫 단계 사이에 해시 체인이 메시지별 키를 파생
- 각 단계: `(chain_key, message_key) = KDF(chain_key_prev)`
- 메시지 키는 한 번 사용 후 삭제

```
Alice sends:                    Bob receives:
  DH_A1 →  msg1 (chain_key_A1) ← DH_B0
  DH_A1 →  msg2 (chain_key_A1)
                                Bob sends:
                                  DH_B1 → msg3 (chain_key_B1) ← DH_A1
  Alice receives:
  DH_A1 ← msg3 (chain_key_B1) ← DH_B1

  Alice sends (new DH ratchet step):
  DH_A2 →  msg4 (chain_key_A2) ← DH_B1
```

### 2.3 두 래칫이 필요한 이유

| 래칫 | 목적 | 보안 특성 |
|------|------|----------|
| DH 래칫 | 새 공유 비밀 생성 | 전방 비밀성 + 사후 침해 보안 |
| 대칭 래칫 | 메시지별 키 효율적 파생 | 메시지 간 키 분리 |

DH 래칫은 비용이 많이 들지만(키 교환 필요) 가장 강력한 보안을 제공합니다. 대칭 래칫은 저렴하며(해시만 필요) DH 단계 사이의 간격을 채웁니다.

### 2.4 순서 없는 메시지 처리

대칭 래칫은 순서가 바뀐 메시지 처리를 가능하게 합니다: 메시지 5가 메시지 4보다 먼저 도착하면, 수신자는 위치 4의 체인 키를 저장하고 메시지 5의 키를 계산할 수 있습니다. 메시지 4가 결국 도착하면, 저장된 체인 키를 사용하여 해당 키를 파생합니다.

---

## 3. 안전한 다자간 계산(MPC)

### 3.1 백만장자 문제(Millionaires' Problem)

Yao의 백만장자 문제(1982): 두 백만장자가 서로의 실제 재산을 공개하지 않고 누가 더 부유한지 알고 싶습니다. 더 일반적으로: 각자 비공개 입력 $x_i$를 가진 $n$명의 당사자가 출력 이상의 정보를 누구도 알지 못하면서 $f(x_1, \ldots, x_n)$을 계산하고 싶습니다.

### 3.2 보안 모델

| 모델 | 공격 방식 | 보장 |
|------|----------|------|
| **반정직(Semi-Honest)** (정직하지만 호기심 있는) | 프로토콜을 따르지만 트랜스크립트에서 추론 시도 | 수동 공격자에 대한 프라이버시 |
| **악의적(Malicious)** | 프로토콜에서 임의로 벗어날 수 있음 | 능동적 공격자에 대한 프라이버시 + 정확성 |
| **은밀(Covert)** | 잡히지 않으면 벗어남 | 속임이 확률 $\epsilon$으로 탐지됨 |

### 3.3 Yao의 혼돈 회로(Garbled Circuits)

2자간 계산(Two-Party Computation)을 위한 Yao의 프로토콜(1986):

1. **회로 표현**: $f$를 불리언 회로로 표현
2. **혼돈화(Garbling)**: 한 당사자("가블러(Garbler)")가 각 와이어 값(0 또는 1)을 임의의 레이블로 교체하고, 각 게이트에 대해 암호화된 진리표(Truth Table)를 생성
3. **평가(Evaluation)**: 다른 당사자("평가자(Evaluator)")가 자신의 입력에 대한 레이블을 (불확지 전송으로, 6절) 얻고 혼돈 회로를 게이트별로 평가
4. **결과**: 평가자가 출력 레이블을 얻으면, 가블러가 이를 0 또는 1로 매핑

```python
"""
Simplified Garbled Circuit demonstration.

This is a conceptual implementation showing the core idea.
A real garbled circuit system uses point-and-permute optimization,
free-XOR technique, and half-gates for efficiency.

Why garbled circuits? They allow two parties to compute ANY function
on their joint inputs while revealing only the output. This is the
most general form of two-party computation.
"""

import secrets
import hashlib
from typing import Dict, Tuple


def garble_and_gate(
    wire_a_labels: Tuple[bytes, bytes],  # (label_0, label_1) for wire A
    wire_b_labels: Tuple[bytes, bytes],  # (label_0, label_1) for wire B
    wire_c_labels: Tuple[bytes, bytes],  # (label_0, label_1) for output wire C
) -> list[bytes]:
    """
    Create a garbled AND gate.

    The truth table for AND: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1

    Each row encrypts the output label under the two input labels.
    Why double encryption? The evaluator needs both input labels to
    decrypt the correct output label. Having only one label reveals nothing.
    """
    garbled_table = []

    and_truth = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]

    for a_val, b_val, c_val in and_truth:
        # Double-encrypt the output label
        key_a = wire_a_labels[a_val]
        key_b = wire_b_labels[b_val]
        output_label = wire_c_labels[c_val]

        # Encrypt: H(key_a || key_b) XOR output_label
        # Why hash-based encryption? It's a simple PRF construction.
        # Real implementations use AES for efficiency.
        h = hashlib.sha256(key_a + key_b).digest()[:len(output_label)]
        encrypted = bytes(a ^ b for a, b in zip(h, output_label))
        garbled_table.append(encrypted)

    # Randomly permute the table rows
    # Why shuffle? Without shuffling, the position in the table
    # leaks information about the input values.
    indices = list(range(4))
    for i in range(3, 0, -1):
        j = secrets.randbelow(i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    return [garbled_table[i] for i in indices]


def evaluate_garbled_gate(
    garbled_table: list[bytes],
    label_a: bytes,
    label_b: bytes,
    label_length: int = 16,
) -> bytes:
    """
    Evaluate a garbled gate given input labels.

    The evaluator tries all rows (only one will decrypt correctly).
    Why try all rows? The table is shuffled, so the evaluator
    doesn't know which row corresponds to their inputs.
    """
    for encrypted_row in garbled_table:
        h = hashlib.sha256(label_a + label_b).digest()[:label_length]
        decrypted = bytes(a ^ b for a, b in zip(h, encrypted_row))
        # In a real implementation, we'd use point-and-permute to
        # avoid trying all rows (only try the one indicated by
        # permutation bits appended to labels)
        if _is_valid_label(decrypted):
            return decrypted

    raise ValueError("No valid decryption found")


def _is_valid_label(label: bytes) -> bool:
    """
    Check if a decrypted label is valid.

    In this simplified version, we use a simple heuristic.
    Real implementations use point-and-permute with explicit
    validity markers.
    """
    # For demonstration: labels have a known prefix
    return label[:2] == b'\xAB\xCD'


def demo_garbled_circuit():
    """Demonstrate a garbled AND gate."""
    label_length = 16

    def make_label():
        return b'\xAB\xCD' + secrets.token_bytes(label_length - 2)

    # Generate random labels for each wire value
    wire_a = (make_label(), make_label())  # (label_0, label_1)
    wire_b = (make_label(), make_label())
    wire_c = (make_label(), make_label())

    # Garbler creates the garbled table
    garbled_table = garble_and_gate(wire_a, wire_b, wire_c)

    # Test all input combinations
    print("=== Garbled AND Gate ===")
    for a_val in [0, 1]:
        for b_val in [0, 1]:
            result_label = evaluate_garbled_gate(
                garbled_table, wire_a[a_val], wire_b[b_val], label_length
            )
            # Map output label back to bit
            result_bit = 0 if result_label == wire_c[0] else 1
            expected = a_val & b_val
            print(f"  AND({a_val}, {b_val}) = {result_bit} "
                  f"(expected: {expected}, "
                  f"{'CORRECT' if result_bit == expected else 'WRONG'})")


if __name__ == "__main__":
    demo_garbled_circuit()
```

---

## 4. 샤미르 비밀 공유

### 4.1 문제

비밀 $S$를 $n$명의 당사자 사이에 분할하되:
- $t$명(임계값) 이상의 당사자가 $S$를 복원할 수 있어야 함
- $t$명 미만의 당사자는 $S$에 대해 **아무것도** 알 수 없어야 함 (정보 이론적 보안)

### 4.2 스킴

샤미르 비밀 공유(Shamir's Secret Sharing, 1979)는 다항식 보간법(Polynomial Interpolation)을 기반으로 합니다:

1. **공유**: $f(0) = S$이고 차수가 $t-1$인 임의의 다항식 $f(x)$를 선택:
   $$f(x) = S + a_1 x + a_2 x^2 + \cdots + a_{t-1} x^{t-1}$$
2. **배포**: 당사자 $i$에게 쉐어(Share) $(i, f(i))$를 제공
3. **복원**: 임의의 $t$개 쉐어가 다항식을 유일하게 결정하며 (라그랑주 보간법(Lagrange Interpolation)을 통해), $S = f(0)$을 복원

### 4.3 $t-1$개의 점이 불충분한 이유

차수 $t-1$인 다항식은 $t-1$개의 점을 무한히 많은 방법으로 통과합니다. 가능한 각 다항식은 서로 다른 비밀 값에 해당하며, 모두 동일하게 가능합니다. 이것이 **완전한** 비밀성입니다 — $t-1$개의 쉐어만으로는 비밀이 어떤 값이든 동일한 확률을 가집니다.

### 4.4 구현

```python
"""
Shamir's Secret Sharing Scheme.

Why work in a finite field (mod p)? In the integers, interpolation
produces fractions. In a prime field Z_p, division is always possible
(every non-zero element has a multiplicative inverse), and the
information-theoretic security proof holds exactly.
"""

import secrets
from typing import List, Tuple


class ShamirSecretSharing:
    """Shamir's (t, n) threshold secret sharing over Z_p."""

    def __init__(self, prime: int = None):
        """
        Initialize with a prime modulus.

        Why a large prime? The prime must be larger than the secret
        and the number of shares. A 256-bit prime works for most
        applications (can share any 256-bit secret).
        """
        if prime is None:
            # Mersenne prime 2^127 - 1 (good for demonstration)
            self.p = (1 << 127) - 1
        else:
            self.p = prime

    def split(self, secret: int, threshold: int,
              num_shares: int) -> List[Tuple[int, int]]:
        """
        Split a secret into shares.

        Parameters:
            secret: The secret value (must be < self.p)
            threshold: Minimum shares needed to reconstruct (t)
            num_shares: Total number of shares to generate (n)

        Returns:
            List of (x, y) shares
        """
        assert 0 <= secret < self.p, f"Secret must be in [0, {self.p})"
        assert threshold <= num_shares, "Threshold cannot exceed num_shares"
        assert threshold >= 2, "Threshold must be at least 2"

        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1}
        coefficients = [secret]  # a0 = secret
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(self.p))

        # Evaluate polynomial at x = 1, 2, ..., n
        shares = []
        for i in range(1, num_shares + 1):
            y = self._evaluate_polynomial(coefficients, i)
            shares.append((i, y))

        return shares

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct the secret from shares using Lagrange interpolation.

        Why Lagrange interpolation? Given t points on a degree-(t-1)
        polynomial, Lagrange interpolation recovers the unique
        polynomial passing through all points. The secret is f(0).
        """
        # Lagrange interpolation at x = 0
        secret = 0

        for i, (xi, yi) in enumerate(shares):
            # Compute Lagrange basis polynomial L_i(0)
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(shares):
                if i != j:
                    # L_i(0) = product of (0 - xj) / (xi - xj)
                    numerator = (numerator * (-xj)) % self.p
                    denominator = (denominator * (xi - xj)) % self.p

            # Why modular inverse? In Z_p, division is multiplication
            # by the modular inverse. This exists for all non-zero
            # elements because p is prime (Fermat's little theorem).
            lagrange_coeff = (numerator * pow(denominator, -1, self.p)) % self.p
            secret = (secret + yi * lagrange_coeff) % self.p

        return secret

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at x using Horner's method."""
        # Horner's method: f(x) = (...((a_{t-1}*x + a_{t-2})*x + a_{t-3})*x + ... + a_0)
        # Why Horner's? It evaluates a degree-d polynomial with only d
        # multiplications instead of d*(d+1)/2 with naive evaluation.
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.p
        return result


def demo_secret_sharing():
    """Demonstrate Shamir's Secret Sharing."""
    sss = ShamirSecretSharing()

    secret = 42
    threshold = 3  # Need at least 3 shares to reconstruct
    num_shares = 5  # Distribute 5 shares total

    print(f"Secret: {secret}")
    print(f"Threshold: {threshold}-of-{num_shares}\n")

    # Split
    shares = sss.split(secret, threshold, num_shares)
    for i, (x, y) in enumerate(shares):
        print(f"  Share {i+1}: ({x}, {y % 1000}...)  [truncated]")

    # Reconstruct with exactly threshold shares
    print(f"\n--- Reconstruct with {threshold} shares ---")
    recovered = sss.reconstruct(shares[:threshold])
    print(f"  Recovered secret: {recovered}")
    print(f"  Correct: {recovered == secret}")

    # Reconstruct with more than threshold shares
    print(f"\n--- Reconstruct with {threshold + 1} shares ---")
    recovered = sss.reconstruct(shares[:threshold + 1])
    print(f"  Recovered secret: {recovered}")
    print(f"  Correct: {recovered == secret}")

    # Attempt with fewer than threshold shares
    print(f"\n--- Reconstruct with {threshold - 1} shares ---")
    wrong = sss.reconstruct(shares[:threshold - 1])
    print(f"  Result: {wrong}")
    print(f"  Correct: {wrong == secret}")
    print(f"  (With fewer than t shares, the result is random — no information leaked)")

    # Different subsets of t shares all work
    print(f"\n--- Different subsets of {threshold} shares ---")
    import itertools
    for combo in list(itertools.combinations(shares, threshold))[:3]:
        recovered = sss.reconstruct(list(combo))
        indices = [s[0] for s in combo]
        print(f"  Shares {indices}: recovered = {recovered}, correct = {recovered == secret}")


if __name__ == "__main__":
    demo_secret_sharing()
```

### 4.5 응용

| 응용 분야 | 임계값 | 사용 사례 |
|----------|-------|---------|
| 기업 키 에스크로(Key Escrow) | 이사회 5명 중 3명 | CEO 무력화 시 키 복구 |
| 암호화폐 지갑 | 3개 기기 중 2개 | 단일 기기 침해 방지 |
| 핵 발사 코드 | 장교 2명 중 2명 | 이중 인원 무결성(Dual-Person Integrity) |
| 분산 서명 | 검증자 n명 중 t명 | 임계값 서명 (7절) |

---

## 5. 커밋먼트 스킴

### 5.1 정의

**커밋먼트 스킴(Commitment Scheme)**은 당사자가 값을 숨긴 채로 커밋하고 나중에 커밋먼트를 공개(열기)할 수 있게 합니다.

두 가지 특성:
- **은닉성(Hiding)**: 커밋먼트는 값에 대해 아무것도 드러내지 않음
- **결합성(Binding)**: 커미터(Committer)는 커밋 후 값을 변경할 수 없음

### 5.2 해시 기반 커밋먼트

가장 단순한 구성:

$$
\text{Commit}(m, r) = H(m \| r) \quad \text{where } r \text{ is random}
$$

- **은닉성**: $H$가 $m$을 숨김 (랜덤 오라클 모델)
- **결합성**: 충돌 저항성(Collision Resistance)이 동일한 커밋먼트를 가진 $m' \neq m$을 찾는 것을 방지

### 5.3 페더슨 커밋먼트(Pedersen Commitment)

추가적인 대수적 특성을 가진 커밋먼트:

$$
C = g^m h^r \pmod{p}
$$

여기서 $g, h$는 이산 로그(Discrete Log) 관계를 알 수 없는 생성원입니다.

**핵심 특성**: 페더슨 커밋먼트는 **동형(Homomorphic)**입니다:

$$
C_1 \cdot C_2 = g^{m_1 + m_2} h^{r_1 + r_2} = \text{Commit}(m_1 + m_2, r_1 + r_2)
$$

이를 통해 커밋된 덧셈이 가능합니다 — 커밋된 값들을 공개하지 않고 합에 대한 명제를 증명할 수 있습니다.

```python
"""
Commitment scheme implementations.

Why commitment schemes? They are building blocks for:
- Zero-knowledge proofs (Lesson 12: graph coloring ZKP uses commitments)
- Auction protocols (bid without revealing, then reveal after deadline)
- Coin-flipping over the phone (fair randomness without trust)
- MPC protocols (commit to inputs before seeing others' inputs)
"""

import secrets
import hashlib


class HashCommitment:
    """Hash-based commitment scheme."""

    @staticmethod
    def commit(message: bytes) -> tuple[bytes, bytes]:
        """
        Commit to a message.

        Returns (commitment, opening_info).
        The opening_info includes the randomness needed to verify.
        """
        randomness = secrets.token_bytes(32)
        commitment = hashlib.sha256(message + randomness).digest()
        return commitment, randomness

    @staticmethod
    def verify(commitment: bytes, message: bytes,
               randomness: bytes) -> bool:
        """Verify that a commitment opens to the claimed message."""
        expected = hashlib.sha256(message + randomness).digest()
        # Why constant-time comparison? Prevents timing attacks that
        # could leak information about the commitment value.
        return secrets.compare_digest(commitment, expected)


class PedersenCommitment:
    """
    Pedersen commitment over Z_p*.

    Properties:
    - Perfectly hiding (information-theoretic)
    - Computationally binding (under DLP assumption)
    - Homomorphic: Commit(a) * Commit(b) = Commit(a+b)
    """

    def __init__(self, p: int, g: int, h: int):
        """
        Initialize with group parameters.

        CRITICAL: The discrete log of h with respect to g must be
        unknown. If someone knows log_g(h), they can break binding
        (open a commitment to any value).
        """
        self.p = p
        self.g = g
        self.h = h

    def commit(self, value: int) -> tuple[int, int]:
        """Commit to a value. Returns (commitment, randomness)."""
        r = secrets.randbelow(self.p - 1) + 1
        c = (pow(self.g, value, self.p) * pow(self.h, r, self.p)) % self.p
        return c, r

    def verify(self, commitment: int, value: int, randomness: int) -> bool:
        """Verify a Pedersen commitment opening."""
        expected = (pow(self.g, value, self.p) *
                    pow(self.h, randomness, self.p)) % self.p
        return commitment == expected

    def add_commitments(self, c1: int, c2: int) -> int:
        """
        Homomorphically add two commitments.

        If c1 = Commit(v1, r1) and c2 = Commit(v2, r2),
        then c1 * c2 = Commit(v1 + v2, r1 + r2).
        """
        return (c1 * c2) % self.p


def demo_commitments():
    """Demonstrate commitment schemes."""

    # Hash commitment
    print("=== Hash Commitment ===")
    msg = b"My secret bid: $1000"
    comm, randomness = HashCommitment.commit(msg)
    print(f"Commitment: {comm.hex()[:16]}...")
    print(f"Verify (correct):  {HashCommitment.verify(comm, msg, randomness)}")
    print(f"Verify (tampered): {HashCommitment.verify(comm, b'$2000', randomness)}")

    # Pedersen commitment (small prime for demo)
    print("\n=== Pedersen Commitment ===")
    p = 23
    g = 4   # Generator
    h = 9   # Another generator (unknown DLP relation to g)

    pc = PedersenCommitment(p, g, h)

    v1, v2 = 5, 7
    c1, r1 = pc.commit(v1)
    c2, r2 = pc.commit(v2)

    print(f"Commit({v1}) = {c1}")
    print(f"Commit({v2}) = {c2}")
    print(f"Verify c1: {pc.verify(c1, v1, r1)}")
    print(f"Verify c2: {pc.verify(c2, v2, r2)}")

    # Homomorphic property
    c_sum = pc.add_commitments(c1, c2)
    r_sum = (r1 + r2) % (p - 1)
    print(f"\nHomomorphic addition:")
    print(f"Commit({v1}) * Commit({v2}) = {c_sum}")
    print(f"Verify as Commit({v1 + v2}): {pc.verify(c_sum, v1 + v2, r_sum)}")


if __name__ == "__main__":
    demo_commitments()
```

---

## 6. 불확지 전송

### 6.1 정의

**1-of-2 불확지 전송(Oblivious Transfer, OT)**: 발신자는 두 메시지 $(m_0, m_1)$를 가집니다. 수신자는 선택 비트 $b$를 가집니다. 프로토콜 후:
- 수신자는 $m_b$ (자신이 선택한 메시지)를 알게 됨
- 발신자는 $b$에 대해 아무것도 알지 못함 (어떤 메시지가 선택되었는지)
- 수신자는 $m_{1-b}$ (다른 메시지)에 대해 아무것도 알지 못함

### 6.2 OT가 중요한 이유

OT는 암호학에서 근본적인 구성 요소입니다. Kilian(1988)은 OT만으로도 **모든** 안전한 2자간 계산을 구현하기에 충분하다는 것을 증명했습니다 — OT를 "보편(Universal)" 기본 요소로 만들어 줍니다.

Yao의 혼돈 회로(3절)에서 OT는 가블러(Garbler)가 평가자(Evaluator)의 어떤 비트를 갖고 있는지 알지 못하면서 평가자가 자신의 입력 비트에 대한 레이블을 얻는 데 사용됩니다.

### 6.3 단순 OT 프로토콜 (RSA 기반)

| 단계 | 발신자 ($m_0, m_1$ 보유) | 수신자 ($m_b$ 원함) |
|------|------------------------|-------------------|
| 1 | RSA 키 쌍 $(e, d, N)$ 생성; $(e, N)$과 임의 값 $x_0, x_1$ 전송 | |
| 2 | | 임의 $k$ 선택; $v = x_b + k^e \bmod N$ 계산; $v$ 전송 |
| 3 | $k_0 = (v - x_0)^d \bmod N$과 $k_1 = (v - x_1)^d \bmod N$ 계산 | |
| 4 | $m_0' = m_0 + k_0$과 $m_1' = m_1 + k_1$ 전송 | |
| 5 | | $m_b = m_b' - k$ 계산 (수신자는 선택한 메시지에 대해서만 $k$를 알고 있음) |

수신자는 $k_b = ((x_b + k^e) - x_b)^d = k$이기 때문에 $m_b$를 복원할 수 있습니다. 다른 메시지의 경우, $(v - x_{1-b})^d$는 무작위로 보이는 값을 생성합니다 — 수신자는 $k_{1-b}$를 복원할 수 없습니다.

### 6.4 OT 확장(OT Extension)

많은 OT를 수행하는 것은 각각 공개 키 연산이 필요하기 때문에 비용이 많이 듭니다. **OT 확장**(Ishai 외, 2003)은 소수의 "기본 OT"(예: 128개)를 사용하여 대칭 키 연산(해시)만으로 임의 개수의 OT를 생성합니다. 이는 대규모 MPC를 실용적으로 만듭니다.

---

## 7. 임계값 서명

### 7.1 동기

표준 디지털 서명은 단일 실패 지점을 가집니다: 개인 키가 침해되면 공격자가 무엇이든 서명할 수 있습니다. **임계값 서명(Threshold Signatures)**은 서명 키를 $n$명의 당사자 사이에 분산하여 유효한 서명을 위해 $t$-of-$n$이 협력하도록 합니다.

### 7.2 특성

- **임계값**: 임의의 $t$명이 서명 가능; $t$명 미만은 불가
- **비대화형(Non-Interactive)**: (일부 스킴에서) 서명자들이 독립적으로 부분 서명 생성
- **구분 불가(Indistinguishable)**: 최종 서명은 표준 단일 서명자 서명과 동일

### 7.3 응용

| 시스템 | 임계값 | 목적 |
|--------|-------|------|
| 암호화폐 보관 | 키 보유자 5명 중 3명 | 대규모 잔액 보호 |
| 인증 기관(CA) | HSM 3개 중 2개 | 단일 HSM 침해 방지 |
| 분산 검증자 | n개 노드 중 2/3 | 비잔틴 내결함성(Byzantine Fault Tolerance) |
| 정부 시스템 | 공무원 n명 중 m명 | 내부자 위협 방지 |

### 7.4 임계값 ECDSA 및 슈노르(Schnorr)

임계값 슈노르 서명(Threshold Schnorr Signatures)은 슈노르의 선형 구조 때문에 임계값 ECDSA보다 단순합니다:

$$
s = r + c \cdot x = (r_1 + \ldots + r_t) + c \cdot (x_1 + \ldots + x_t)
$$

각 당사자 $i$가 부분 서명 $s_i = r_i + c \cdot x_i$를 계산하고, 부분 서명들을 단순히 더합니다. ECDSA는 곱셈 구조 때문에 더 복잡한 MPC 기법이 필요합니다.

---

## 8. 프라이빗 정보 검색

### 8.1 문제

사용자가 서버에게 어떤 항목이 조회되었는지 알리지 않고 데이터베이스를 조회하고 싶습니다. 단순한 해결책: 전체 데이터베이스를 다운로드 — 정확하지만 대용량 데이터베이스에는 비실용적입니다.

### 8.2 PIR 접근 방식

| 접근 방식 | 통신량 | 서버 계산 | 신뢰 |
|----------|-------|---------|------|
| 단순 (전체 다운로드) | $O(N)$ | $O(1)$ | 없음 |
| 계산적 PIR(cPIR) | $O(\text{polylog}(N))$ | $O(N)$ | 계산 난이도 |
| IT-PIR (다중 서버) | $O(\text{polylog}(N))$ | $O(N)$ | 공모하지 않는 서버 |
| HE 기반 PIR | $O(\sqrt{N})$ | $O(N)$ | LWE 난이도 |

### 8.3 HE 기반 PIR의 동작 방식

1. 클라이언트가 쿼리 벡터 생성: $\mathbf{q} = (0, 0, \ldots, 1, \ldots, 0)$, 여기서 1은 위치 $i$ (원하는 항목)에 있음
2. 클라이언트가 FHE로 $\mathbf{q}$ 암호화: $\overline{\mathbf{q}} = \text{Enc}(\mathbf{q})$
3. 서버가 $\overline{\text{result}} = \overline{\mathbf{q}} \cdot \mathbf{DB}$ 계산 (암호화된 쿼리와 데이터베이스의 내적)
4. 클라이언트가 복호화: $\text{result} = \text{Dec}(\overline{\text{result}}) = \text{DB}[i]$

쿼리가 암호화되어 있어 서버는 아무것도 알 수 없습니다 — 항목 1에 대한 쿼리와 항목 1000에 대한 쿼리를 구분할 수 없습니다.

---

## 9. 요약

| 프로토콜 | 사용된 기본 요소 | 핵심 특성 |
|---------|---------------|---------|
| TLS 1.3 | ECDHE, HKDF, AEAD, 인증서 | 전방 비밀성, 1-RTT 핸드셰이크 |
| Signal (이중 래칫) | X3DH, DH 래칫, KDF 체인 | 전방 + 사후 침해 보안 |
| MPC (혼돈 회로) | OT, 대칭 암호화, 해시 | 공동 입력에 대해 공개 없이 계산 |
| 샤미르 비밀 공유 | $\mathbb{F}_p$ 위의 다항식 보간법 | $t$-of-$n$ 복원, 완전 비밀성 |
| 커밋먼트 스킴 | 해시 함수, 이산 로그 문제(DLP) | 숨기고 공개; 동형 (페더슨) |
| 불확지 전송 | 공개 키 암호 | MPC의 보편 기본 요소 |
| 임계값 서명 | 비밀 공유 + 서명 | 분산 신뢰, 단일 실패 지점 없음 |
| PIR | HE 또는 다중 서버 | 쿼리를 공개하지 않고 데이터베이스 조회 |

반복되는 주제: 어떤 기본 요소도 혼자 동작하지 않습니다. 안전한 시스템은 신중한 결합을 필요로 합니다 — 그리고 결합 자체가 안전성이 증명되어야 하며, 개별 요소만으로는 충분하지 않습니다.

---

## 10. 연습 문제

### 연습 1: TLS 1.3 키 스케줄 (코딩)

TLS 1.3 키 파생 스케줄을 구현하세요:
1. 사전 공유 키(PSK)와 ECDHE 공유 비밀이 주어졌을 때, HKDF를 사용하여 얼리 시크릿(Early Secret), 핸드셰이크 시크릿, 마스터 시크릿을 계산
2. 클라이언트와 서버 핸드셰이크 트래픽 시크릿을 파생
3. RFC 8448의 테스트 벡터에 대해 구현을 검증

### 연습 2: 이중 래칫 시뮬레이션 (코딩)

단순화된 이중 래칫을 구현하세요:
1. 공유 루트 키로 초기화 (X3DH 이후를 시뮬레이션)
2. 대칭 래칫 구현 (메시지별 키를 위한 KDF 체인)
3. DH 래칫 구현 (발신자 변경 시 새 DH 교환)
4. 하나의 메시지 키를 침해해도 과거 또는 미래 메시지 키가 노출되지 않음을 보임
5. 순서가 바뀐 메시지 전달을 시뮬레이션하고 올바른 복호화를 보임

### 연습 3: 비밀 공유 응용 (코딩)

샤미르 구현을 확장하세요:
1. **능동적 비밀 공유(Proactive Secret Sharing)** 구현: 비밀을 변경하지 않고 주기적으로 쉐어를 갱신 (동일한 상수항을 가진 새 임의 다항식 생성)
2. **검증 가능한 비밀 공유(Feldman VSS, Verifiable Secret Sharing)** 구현: 각 쉐어에 커밋먼트가 포함되어 주주가 쉐어를 공개하지 않고 일관성을 검증 가능
3. 쉐어 갱신 시연: 쉐어를 생성하고, 갱신하고, 이전 및 새 쉐어가 동일한 비밀을 복원함을 검증

### 연습 4: 안전한 동전 뒤집기 (코딩)

Alice와 Bob 사이의 안전한 동전 뒤집기 프로토콜을 구현하세요:
1. Alice가 임의 비트 $a$에 커밋
2. Bob이 자신의 임의 비트 $b$를 (평문으로) 전송
3. Alice가 커밋먼트를 공개
4. 동전 뒤집기 결과는 $a \oplus b$

다음을 보이세요:
- 어느 당사자도 결과를 편향할 수 없음 (결합성이 Alice의 $a$ 변경을 방지; $a \oplus b$는 $a$ 또는 $b$ 중 하나가 임의이면 균등하게 랜덤)
- 해시 기반과 페더슨 커밋먼트 모두를 사용하여 구현
- 순서를 반대로 하면 어떻게 되는가 (Bob이 먼저 커밋)?

### 연습 5: 프로토콜 결합 분석 (심화)

단순화된 "안전한 경매" 프로토콜을 고려하세요:
1. 각 입찰자가 Paillier 암호화(레슨 13)로 입찰금을 암호화
2. 경매인이 모든 입찰의 합을 동형적으로 계산
3. 경매인이 ZKP(레슨 12)를 사용하여 합을 올바르게 계산했음을 증명
4. 낙찰 입찰이 모든 입찰을 비교하는 혼돈 회로에 의해 결정됨

분석하세요:
- 이 프로토콜이 의존하는 암호학적 가정은 무엇인가?
- 경매인이 악의적이면 어떻게 되는가? 추가적으로 어떤 메커니즘이 필요한가?
- 효율성을 개선하기 위해 다른 기본 요소로 어떤 구성 요소를 대체할 수 있는가?
- 동률 입찰을 처리하는 수정된 프로토콜을 설계하라
- 입찰자 수 $n$에 대한 통신 복잡도는 얼마인가?

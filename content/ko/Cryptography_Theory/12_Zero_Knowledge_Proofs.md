# 레슨 12: 영지식 증명(Zero-Knowledge Proofs)

[← 이전: 포스트 양자 암호학](./11_Post_Quantum_Cryptography.md) | [다음: 동형 암호 →](./13_Homomorphic_Encryption.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 구체적인 비유와 형식적 정의를 사용하여 영지식 증명(Zero-Knowledge Proof)의 개념 설명하기
2. ZKP의 세 가지 속성인 완전성(Completeness), 건전성(Soundness), 영지식성(Zero-Knowledge) 정의하기
3. 슈노르 식별 프로토콜(Schnorr Identification Protocol)을 구현하고 보안 환원 이해하기
4. 피아트-샤미르 휴리스틱(Fiat-Shamir Heuristic)을 적용하여 대화형 증명을 비대화형으로 변환하기
5. zk-SNARK와 zk-STARK를 구분하고 각각의 트레이드오프 이해하기
6. 그래프 3-색칠(Graph 3-Coloring)을 위한 간단한 영지식 증명 구현하기
7. 블록체인, 신원, 투표 시스템에서 ZKP의 실제 응용 사례 식별하기

---

당신이 알고 있는 것을 밝히지 않고도 그것을 알고 있음을 증명할 수 있을까요? 역설적으로 보이는 이 질문은 정밀한 수학적 답변을 가집니다: **영지식 증명(Zero-Knowledge Proofs, ZKP)**입니다. 1985년 Goldwasser, Micali, Rackoff가 도입한 ZKP는 이론 컴퓨터 과학에서 가장 놀랍고 강력한 아이디어 중 하나입니다. 순수 이론 개념에서 프라이버시 보존 암호화폐, 탈중앙화 신원 시스템, 검증 가능한 계산을 지원하는 실용적인 도구로 변모했습니다. 이 레슨에서는 간단한 예시에서 시작하여 디지털 프라이버시를 재편하고 있는 최첨단 증명 시스템까지 직관을 쌓아나갑니다.

## 목차

1. [직관: 밝히지 않고 증명하기](#1-직관-밝히지-않고-증명하기)
2. [형식적 정의](#2-형식적-정의)
3. [슈노르 식별 프로토콜](#3-슈노르-식별-프로토콜)
4. [피아트-샤미르 휴리스틱](#4-피아트-샤미르-휴리스틱)
5. [그래프 3-색칠을 위한 영지식 증명](#5-그래프-3-색칠을-위한-영지식-증명)
6. [zk-SNARK](#6-zk-snark)
7. [zk-STARK](#7-zk-stark)
8. [응용](#8-응용)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)

---

## 1. 직관: 밝히지 않고 증명하기

### 1.1 알리바바 동굴

고전적인 비유 (Quisquater 등, 1990):

> **비유:** 영지식 증명은 마법의 문을 여는 비밀 단어를 알고 있다는 것을 — 그 단어 자체를 밝히지 않고 — 증명하는 것과 같습니다. 뒤쪽에 잠긴 문이 있는 고리 모양의 동굴을 상상하세요. Alice(증명자)는 왼쪽 또는 오른쪽 경로 중 하나로 들어갑니다. Bob(검증자)은 Alice가 어느 쪽으로 나오길 원하는지 외칩니다. Alice가 비밀 단어를 안다면, 필요할 경우 문을 열어 항상 응할 수 있습니다. 모른다면, 올바르게 추측할 확률은 50%에 불과합니다. 20라운드 후, Bob은 $1 - 2^{-20}$의 확신으로 Alice가 단어를 안다고 믿게 되지만, 그 단어가 무엇인지는 전혀 알지 못합니다.

```
         Entrance
         /      \
        /        \
       A          B
       |          |
       |  [DOOR]  |
       |          |
        \        /
         \      /
```

### 1.2 더 친숙한 예시들

| 시나리오 | 증명되는 것 | 밝혀지지 않는 것 |
|----------|---------------|---------------------|
| 나이 인증 | "나는 18세 이상이다" | 정확한 나이, 생년월일 |
| 지급 능력 증명 | "나는 100만 달러 이상의 자산이 있다" | 정확한 잔액, 계좌 정보 |
| 비밀번호 인증 | "나는 비밀번호를 안다" | 비밀번호 자체 |
| 자격 증명 확인 | "나는 유효한 운전면허증이 있다" | 면허 번호, 주소 |

### 1.3 ZKP가 중요한 이유

- **프라이버시**: 기반 데이터를 노출하지 않고 진술을 증명
- **확장성**: 다시 실행하지 않고 계산을 검증 (간결한 증명)
- **규제 준수**: 영업 비밀을 밝히지 않고 준수를 증명

---

## 2. 형식적 정의

### 2.1 대화형 증명 시스템

언어 $L$에 대한 **대화형 증명 시스템(Interactive Proof System)**은 두 확률적 다항 시간 기계로 구성됩니다:
- **증명자(Prover)** $P$: 무제한 계산 능력을 보유 (또는 증인 $w$를 알고 있음)
- **검증자(Verifier)** $V$: 다항 시간 내에 실행

두 기계는 메시지를 교환하고, 마지막에 $V$가 수락 또는 거부를 출력합니다.

### 2.2 세 가지 속성

언어 $L$에 대한 증명 시스템 $(P, V)$:

**1. 완전성(Completeness)**: 진술이 참이면, 정직한 증명자는 정직한 검증자를 설득할 수 있습니다.

$$
x \in L \implies \Pr[V \text{ accepts after interacting with } P] \geq 1 - \text{negl}(\lambda)
$$

**2. 건전성(Soundness)**: 진술이 거짓이면, 어떤 부정직한 증명자도 검증자를 설득할 수 없습니다 (무시 가능한 확률 제외).

$$
x \notin L \implies \forall P^*, \Pr[V \text{ accepts after interacting with } P^*] \leq \text{negl}(\lambda)
$$

**3. 영지식성(Zero-Knowledge)**: 검증자는 진술의 참/거짓 이상은 아무것도 배우지 못합니다. 형식적으로, 진술만 주어지고(증인 없이) 실제 대화와 구별할 수 없는 기록을 생성할 수 있는 시뮬레이터 $S$가 존재합니다.

$$
\text{View}_V(P(w) \leftrightarrow V)(x) \approx_c S(x)
$$

### 2.3 영지식의 종류

| 유형 | 구별 불가능성 | 강도 |
|------|---------------------|----------|
| **완전 영지식(Perfect ZK)** | 동일한 분포 | 가장 강함 (정보 이론적) |
| **통계적 영지식(Statistical ZK)** | 무시 가능한 통계적 거리 | 강함 |
| **계산적 영지식(Computational ZK)** | 다항 시간 알고리즘으로 구별 불가 | 표준 (대부분의 실용적 방식) |

---

## 3. 슈노르 식별 프로토콜

### 3.1 설정

- **공개 파라미터**: 소수 $p$, 위수 $q$의 부분군의 생성원 $g$
- **증명자의 비밀**: $x \in \mathbb{Z}_q$
- **증명자의 공개 키**: $y = g^x \bmod p$
- **목표**: 증명자는 검증자에게 자신이 $x$($y$의 이산 대수)를 안다는 것을 증명

### 3.2 프로토콜 단계

| 단계 | 증명자 | | 검증자 |
|------|--------|-|----------|
| 1 | 임의의 $r \in \mathbb{Z}_q$ 선택 | | |
| 2 | $R = g^r \bmod p$ 계산 | $\xrightarrow{R}$ | |
| 3 | | $\xleftarrow{c}$ | 임의의 챌린지 $c \in \mathbb{Z}_q$ 선택 |
| 4 | $s = r + cx \bmod q$ 계산 | $\xrightarrow{s}$ | |
| 5 | | | 검증: $g^s \equiv R \cdot y^c \pmod{p}$ |

### 3.3 작동 원리

**완전성(Completeness)**: $g^s = g^{r+cx} = g^r \cdot g^{cx} = R \cdot (g^x)^c = R \cdot y^c$

**건전성(Soundness)**: 증명자가 같은 $R$에 대해 두 개의 다른 챌린지 $c_1, c_2$에 답할 수 있다면, $s_1 - s_2 = (c_1 - c_2)x$이므로, $x = (s_1 - s_2)(c_1 - c_2)^{-1} \bmod q$가 드러납니다. 이는 증명자가 $x$를 "알고" 있음을 입증합니다.

**영지식성(Zero-Knowledge)**: 시뮬레이터는 $x$ 없이도 유효해 보이는 기록을 생성할 수 있습니다:
1. 임의의 $s$와 $c$ 선택
2. $R = g^s \cdot y^{-c}$ 계산
3. 삼중쌍 $(R, c, s)$는 실제 기록과 구별 불가

### 3.4 구현

```python
"""
Schnorr Zero-Knowledge Identification Protocol.

Why Schnorr and not other ZKP protocols? Schnorr is the simplest
honest-verifier ZKP for discrete log knowledge, and it's the building
block for Schnorr signatures (used in Bitcoin's Taproot, EdDSA).
"""

import secrets
import hashlib


class SchnorrZKP:
    """
    Schnorr Zero-Knowledge Proof of knowledge of discrete log.

    Security relies on the hardness of the discrete logarithm problem
    in the subgroup of order q in Z_p*.
    """

    def __init__(self, p: int, q: int, g: int):
        """
        Initialize with group parameters.

        Why separate p and q? We work in a subgroup of order q (prime)
        inside Z_p*. This ensures all operations are in a prime-order group,
        which prevents small-subgroup attacks.
        """
        self.p = p  # Large prime
        self.q = q  # Prime order of subgroup
        self.g = g  # Generator of subgroup

    def keygen(self) -> tuple[int, int]:
        """Generate (public_key, secret_key) pair."""
        x = secrets.randbelow(self.q - 1) + 1  # Secret key
        y = pow(self.g, x, self.p)               # Public key
        return y, x

    def prover_commit(self, secret_key: int) -> tuple[int, int]:
        """
        Prover's first message: commitment R = g^r mod p.

        Why random r? The randomness r is essential for zero-knowledge.
        Without it (or if r is reused), the secret key leaks.
        This is exactly what happened with the PlayStation 3 ECDSA hack
        — Sony reused the nonce, and the private key was extracted.
        """
        r = secrets.randbelow(self.q - 1) + 1
        R = pow(self.g, r, self.p)
        return R, r  # r is kept secret by prover

    def verifier_challenge(self) -> int:
        """Verifier's random challenge."""
        return secrets.randbelow(self.q - 1) + 1

    def prover_respond(self, r: int, c: int, secret_key: int) -> int:
        """
        Prover's response: s = r + c*x mod q.

        Why mod q (not mod p)? We're doing arithmetic in the exponent,
        which is modulo the group order q (by Fermat's little theorem).
        """
        s = (r + c * secret_key) % self.q
        return s

    def verify(self, public_key: int, R: int, c: int, s: int) -> bool:
        """
        Verify: g^s == R * y^c (mod p).

        Why this equation? It checks that s = r + cx without
        knowing r or x individually. The verifier can compute
        both sides using only public information.
        """
        lhs = pow(self.g, s, self.p)
        rhs = (R * pow(public_key, c, self.p)) % self.p
        return lhs == rhs

    def simulate(self, public_key: int) -> tuple[int, int, int]:
        """
        Simulator: produce a valid-looking transcript WITHOUT the secret.

        This proves zero-knowledge: if a simulator can produce
        indistinguishable transcripts, the real interaction reveals
        nothing beyond what the verifier could compute alone.
        """
        s = secrets.randbelow(self.q - 1) + 1
        c = secrets.randbelow(self.q - 1) + 1

        # Compute R = g^s * y^(-c) mod p
        # Why this formula? We need R such that g^s = R * y^c,
        # i.e., R = g^s * y^(-c). This "reverse engineers" a valid R.
        y_inv_c = pow(public_key, self.q - c, self.p)  # y^(-c) mod p
        R = (pow(self.g, s, self.p) * y_inv_c) % self.p

        return R, c, s


def demo_schnorr():
    """Demonstrate the Schnorr ZKP protocol."""

    # Small parameters for demonstration
    # In practice: p ~ 2048 bits, q ~ 256 bits
    # Here: p = 23, q = 11, g = 4 (g has order 11 in Z_23*)
    p, q, g = 23, 11, 4

    zkp = SchnorrZKP(p, q, g)

    # Key generation
    public_key, secret_key = zkp.keygen()
    print(f"Public key y = {public_key}, Secret key x = {secret_key}")
    print(f"Verification: g^x mod p = {pow(g, secret_key, p)} == {public_key}")

    # Run protocol multiple rounds
    print("\n--- Interactive Protocol (5 rounds) ---")
    for i in range(5):
        # Prover commits
        R, r = zkp.prover_commit(secret_key)

        # Verifier challenges
        c = zkp.verifier_challenge()

        # Prover responds
        s = zkp.prover_respond(r, c, secret_key)

        # Verifier checks
        valid = zkp.verify(public_key, R, c, s)
        print(f"  Round {i+1}: R={R:3d}, c={c:3d}, s={s:3d} → {'ACCEPT' if valid else 'REJECT'}")

    # Demonstrate simulation (no secret key needed!)
    print("\n--- Simulated Transcripts (no secret key!) ---")
    for i in range(5):
        R, c, s = zkp.simulate(public_key)
        valid = zkp.verify(public_key, R, c, s)
        print(f"  Simulated {i+1}: R={R:3d}, c={c:3d}, s={s:3d} → {'VALID' if valid else 'INVALID'}")

    print("\n  Note: Simulated transcripts are indistinguishable from real ones!")


if __name__ == "__main__":
    demo_schnorr()
```

---

## 4. 피아트-샤미르 휴리스틱

### 4.1 대화형 증명의 문제점

대화형 증명은 증명자와 검증자 사이의 실시간 통신이 필요합니다. 이는 많은 응용에서 비실용적입니다:
- 블록체인: 증명은 누구든지, 언제든지 검증 가능해야 함
- 디지털 서명: "검증자"는 나중에 서명을 확인하는 누구든지
- 배치 검증: 수천 개의 증명을 하나씩 검증하는 것은 느림

### 4.2 변환 방법

**피아트-샤미르 휴리스틱(Fiat-Shamir Heuristic)** (1986)은 검증자의 임의 챌린지를 해시 함수의 출력으로 대체합니다:

$$
c = H(R \| \text{message} \| \text{context})
$$

해시 함수가 결정론적이고 예측 불가능하기 때문에, "가상 검증자" 역할을 합니다 — 증명자는 $R$을 선택할 때 챌린지를 예측할 수 없으므로 건전성이 보존됩니다.

### 4.3 슈노르 ZKP에서 슈노르 서명으로

피아트-샤미르를 슈노르 프로토콜에 적용하면 **슈노르 서명(Schnorr Signature)**(레슨 7에서 논의)이 됩니다:

1. 임의의 $r$ 선택, $R = g^r$ 계산
2. 챌린지 계산: $c = H(R \| m)$
3. 응답 계산: $s = r + cx$
4. 서명: $(R, s)$ 또는 $(c, s)$

검증: $c = H(R \| m)$에서 $g^s = R \cdot y^c$ 확인.

```python
"""
Fiat-Shamir transform: from interactive Schnorr ZKP to non-interactive proof.

Why hash-based challenge? The key insight is that the hash function
is a "random oracle" — its output is unpredictable until you commit
to the input. Since R is committed before c is computed, the prover
cannot cheat by choosing R to match a predetermined c.
"""

import secrets
import hashlib


def fiat_shamir_schnorr_prove(p: int, q: int, g: int,
                                x: int, message: bytes) -> tuple[int, int, int]:
    """
    Non-interactive Schnorr proof via Fiat-Shamir.

    Returns (R, c, s) that anyone can verify without interaction.
    """
    # Step 1: Commit
    r = secrets.randbelow(q - 1) + 1
    R = pow(g, r, p)

    # Step 2: Challenge via hash (replaces verifier)
    # Why include the message? It binds the proof to a specific context,
    # preventing the proof from being replayed in a different context.
    hash_input = f"{R}{message.hex()}".encode()
    c_bytes = hashlib.sha256(hash_input).digest()
    c = int.from_bytes(c_bytes, 'big') % q

    # Step 3: Respond
    s = (r + c * x) % q

    return R, c, s


def fiat_shamir_schnorr_verify(p: int, q: int, g: int,
                                  y: int, message: bytes,
                                  R: int, c: int, s: int) -> bool:
    """
    Verify a non-interactive Schnorr proof.

    Anyone can run this with just the public key and proof —
    no interaction with the prover needed.
    """
    # Recompute challenge
    hash_input = f"{R}{message.hex()}".encode()
    c_check_bytes = hashlib.sha256(hash_input).digest()
    c_check = int.from_bytes(c_check_bytes, 'big') % q

    if c != c_check:
        return False  # Challenge doesn't match

    # Verify g^s == R * y^c (mod p)
    lhs = pow(g, s, p)
    rhs = (R * pow(y, c, p)) % p
    return lhs == rhs


# Demo
p, q, g = 23, 11, 4
x = 7  # Secret
y = pow(g, x, p)  # Public key

message = b"I know the secret key"

# Prove (non-interactively)
R, c, s = fiat_shamir_schnorr_prove(p, q, g, x, message)
print(f"Proof: R={R}, c={c}, s={s}")

# Verify (anyone can do this)
valid = fiat_shamir_schnorr_verify(p, q, g, y, message, R, c, s)
print(f"Verification: {'VALID' if valid else 'INVALID'}")

# Tampered message fails
valid2 = fiat_shamir_schnorr_verify(p, q, g, y, b"Different message", R, c, s)
print(f"Tampered verification: {'VALID' if valid2 else 'INVALID'}")
```

### 4.4 피아트-샤미르의 보안

**랜덤 오라클 모델(Random Oracle Model)**(해시 함수가 진정으로 임의의 함수로 모델링되는 경우)에서 피아트-샤미르는 건전성을 보존합니다. 표준 모델(랜덤 오라클 없이)에서는 상황이 더 복잡합니다 — 피아트-샤미르가 안전하지 않은 인위적인 예가 존재하지만, 슈노르와 같은 실용적인 방식에서는 안전한 것으로 간주됩니다.

---

## 5. 그래프 3-색칠을 위한 영지식 증명

### 5.1 왜 그래프 색칠인가?

그래프 3-색칠 문제는 NP-완전 문제입니다. Goldreich, Micali, Wigderson (1986)은 그래프 3-색칠로 환원하여 NP 진술에 영지식 증명이 존재함을 보였습니다. 이는 기초적인 결과입니다: 그래프 3-색칠에 ZKP가 있다면, **모든** NP 진술에 ZKP가 있습니다.

### 5.2 프로토콜

그래프 $G = (V, E)$와 유효한 3-색칠 $\chi: V \to \{1, 2, 3\}$가 주어졌을 때:

**각 라운드:**
1. **증명자**: 3가지 색상을 임의로 순열합니다 (실제 색칠이 숨겨지도록). 암호학적 커밋먼트(Commitment)를 사용하여 각 꼭짓점의 (순열된) 색상에 커밋합니다.
2. **검증자**: 임의의 간선 $(u, v)$을 선택합니다.
3. **증명자**: $u$와 $v$에 대한 커밋먼트를 공개합니다.
4. **검증자**: $u$와 $v$가 서로 다른 색상인지 확인합니다.

**건전성**: 색칠이 유효하지 않다면, 적어도 하나의 간선에 같은 색상의 끝점이 있습니다. 검증자는 라운드당 $\geq 1/|E|$의 확률로 이를 잡아냅니다. $k \cdot |E|$라운드 후 속임수 확률은 $(1 - 1/|E|)^{k \cdot |E|} \approx e^{-k}$로 감소합니다.

**영지식성**: 색상 순열은 검증자가 매 라운드마다 임의의 색상만 보도록 합니다 — 하나의 간선의 두 색상을 배워도 실제 색칠에 대해서는 아무것도 알 수 없습니다.

### 5.3 구현

```python
"""
Zero-Knowledge Proof for Graph 3-Coloring.

Why graph coloring? It's NP-complete, so a ZKP for it implies ZKPs
exist for ALL NP problems (via polynomial-time reductions). This is
the most general ZKP construction, though not the most efficient.
"""

import secrets
import hashlib
import json
from typing import Optional


class Commitment:
    """
    Simple hash-based commitment scheme.

    Commit(x) = H(x || r) where r is random.

    Why commitment? It provides two properties:
    - Hiding: The commitment reveals nothing about x
    - Binding: The committer cannot change x after committing
    """

    @staticmethod
    def commit(value: int) -> tuple[bytes, bytes]:
        """Return (commitment, opening) for a value."""
        randomness = secrets.token_bytes(32)
        data = json.dumps({"value": value, "r": randomness.hex()}).encode()
        commitment = hashlib.sha256(data).digest()
        return commitment, data  # data is the opening

    @staticmethod
    def verify(commitment: bytes, opening: bytes, expected_value: int) -> bool:
        """Verify a commitment opens to the expected value."""
        # Check the hash matches
        if hashlib.sha256(opening).digest() != commitment:
            return False
        # Check the value matches
        data = json.loads(opening)
        return data["value"] == expected_value


class GraphColoringZKP:
    """
    Interactive ZKP for graph 3-coloring.

    The prover convinces the verifier that they know a valid
    3-coloring of a graph without revealing the coloring.
    """

    def __init__(self, edges: list[tuple[int, int]], num_vertices: int):
        self.edges = edges
        self.num_vertices = num_vertices

    def prover_commit(self, coloring: dict[int, int]) -> tuple[dict, dict]:
        """
        Prover's first step: permute colors and commit.

        Why permute? Without permutation, the verifier would learn
        the actual colors over many rounds. The random permutation
        ensures each round reveals only random color pairs.
        """
        # Random permutation of {1, 2, 3}
        perm = [1, 2, 3]
        # Fisher-Yates shuffle
        for i in range(2, 0, -1):
            j = secrets.randbelow(i + 1)
            perm[i], perm[j] = perm[j], perm[i]

        # Apply permutation and commit
        commitments = {}
        openings = {}
        for vertex, color in coloring.items():
            permuted_color = perm[color - 1]  # Colors are 1-indexed
            comm, opening = Commitment.commit(permuted_color)
            commitments[vertex] = comm
            openings[vertex] = (opening, permuted_color)

        return commitments, openings

    def verifier_challenge(self) -> tuple[int, int]:
        """Verifier chooses a random edge."""
        idx = secrets.randbelow(len(self.edges))
        return self.edges[idx]

    def prover_respond(self, openings: dict,
                        edge: tuple[int, int]) -> tuple:
        """Prover reveals colors for the challenged edge."""
        u, v = edge
        opening_u, color_u = openings[u]
        opening_v, color_v = openings[v]
        return (opening_u, color_u), (opening_v, color_v)

    def verify_round(self, commitments: dict,
                      edge: tuple[int, int],
                      response: tuple) -> bool:
        """
        Verify one round of the protocol.

        Checks:
        1. Commitments open correctly
        2. The two vertices have different colors
        3. Colors are valid (1, 2, or 3)
        """
        u, v = edge
        (opening_u, color_u), (opening_v, color_v) = response

        # Check commitments
        if not Commitment.verify(commitments[u], opening_u, color_u):
            return False
        if not Commitment.verify(commitments[v], opening_v, color_v):
            return False

        # Check colors are valid
        if color_u not in {1, 2, 3} or color_v not in {1, 2, 3}:
            return False

        # Check colors are different (the actual 3-coloring property)
        return color_u != color_v


def demo_graph_coloring_zkp():
    """Demonstrate ZKP for graph 3-coloring."""

    # Example: A simple graph (triangle + extra vertex)
    #   0 --- 1
    #   |   / |
    #   |  /  |
    #   | /   |
    #   2 --- 3
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    num_vertices = 4

    # Valid 3-coloring (the prover's secret)
    coloring = {0: 1, 1: 2, 2: 3, 3: 1}  # R, G, B, R

    # Verify coloring is actually valid
    for u, v in edges:
        assert coloring[u] != coloring[v], f"Invalid coloring at edge ({u},{v})"

    zkp = GraphColoringZKP(edges, num_vertices)

    # Run multiple rounds
    num_rounds = 20
    all_passed = True

    print(f"Graph: {num_vertices} vertices, {len(edges)} edges")
    print(f"Running {num_rounds} rounds of ZKP...\n")

    for round_num in range(num_rounds):
        # Step 1: Prover commits
        commitments, openings = zkp.prover_commit(coloring)

        # Step 2: Verifier challenges
        edge = zkp.verifier_challenge()

        # Step 3: Prover responds
        response = zkp.prover_respond(openings, edge)

        # Step 4: Verifier checks
        valid = zkp.verify_round(commitments, edge, response)

        if not valid:
            all_passed = False
            print(f"  Round {round_num + 1}: FAILED (edge {edge})")
        else:
            u, v = edge
            (_, cu), (_, cv) = response
            print(f"  Round {round_num + 1}: PASS (edge {edge}, "
                  f"colors: {cu}, {cv})")

    # Soundness: probability of cheating prover passing all rounds
    cheat_prob = (1 - 1/len(edges)) ** num_rounds
    print(f"\nAll rounds passed: {all_passed}")
    print(f"Soundness: cheating prover succeeds with prob <= "
          f"{cheat_prob:.6f}")
    print(f"Verifier confidence: {1 - cheat_prob:.6f}")


if __name__ == "__main__":
    demo_graph_coloring_zkp()
```

---

## 6. zk-SNARK

### 6.1 zk-SNARK란?

**zk-SNARK**: Zero-Knowledge **Succinct** Non-interactive **Argument** of Knowledge (간결한 비대화형 지식 증명)

- **간결함(Succinct)**: 증명 크기가 $O(\log n)$ 또는 심지어 $O(1)$ — 계산 크기에 무관하게 상수
- **비대화형(Non-interactive)**: 증명자에서 검증자로의 단일 메시지
- **인수(Argument)**: 계산적으로 건전 (무제한 공격자가 아닌 다항 시간 공격자에 안전)
- **지식(Knowledge)**: 증명자는 단순히 진술을 설득하는 것이 아니라 증인을 "알고" 있어야 함

### 6.2 파이프라인 (단순화)

```
Computation → Arithmetic Circuit → R1CS → QAP → SNARK Proof
```

1. **계산**: 진술을 유한체(Finite Field) 위의 산술 회로로 표현
2. **R1CS** (Rank-1 Constraint System): 회로를 $\mathbf{a} \cdot \mathbf{w} \times \mathbf{b} \cdot \mathbf{w} = \mathbf{c} \cdot \mathbf{w}$ 형태의 제약 시스템으로 변환
3. **QAP** (Quadratic Arithmetic Program): 제약을 다항식으로 보간(Interpolate)
4. **SNARK**: 타원 곡선 페어링(Elliptic Curve Pairing)을 사용하여 다항식 항등식이 성립함을 간결하게 증명

### 6.3 신뢰 설정(Trusted Setup)

대부분의 zk-SNARK 구성(특히 Groth16)은 **신뢰 설정 세리머니(Trusted Setup Ceremony)**가 필요합니다:

- **구조화된 참조 문자열(SRS, Structured Reference String)**이 비밀 무작위성을 사용하여 생성됨
- 비밀 무작위성("독성 폐기물(Toxic Waste)")은 반드시 폐기되어야 함
- 누군가 독성 폐기물을 안다면 거짓 증명을 만들 수 있음

이것이 zk-SNARK에 대한 주요 비판입니다. 완화 방법:
- **MPC 세리머니**: 여러 참가자가 무작위성을 기여; 적어도 한 참가자가 정직하면 설정이 안전 (예: Zcash의 세리머니에는 수천 명이 참여)
- **범용 설정**: PLONK와 같은 방식은 어떤 회로에도 작동하는 범용 SRS를 사용 (모든 응용에 대해 한 번의 세리머니)
- **투명한 설정** (zk-STARK): 신뢰 설정 불필요

### 6.4 성능 특성

| 속성 | Groth16 | PLONK | 일반적인 값 |
|----------|---------|-------|---------------|
| 증명 크기 | 3개 군 원소 | ~10개 군 원소 | 128~512바이트 |
| 검증 시간 | 3회 페어링 | ~10회 페어링 | 2~10 ms |
| 증명자 시간 | $O(n \log n)$ | $O(n \log n)$ | 수 초에서 수 분 |
| 신뢰 설정 | 회로별 | 범용 | 일회성 |

---

## 7. zk-STARK

### 7.1 zk-STARK란?

**zk-STARK**: Zero-Knowledge **Scalable** **Transparent** Argument of Knowledge (확장 가능하고 투명한 지식 증명)

- **확장 가능(Scalable)**: 증명자 시간은 준선형 $O(n \log^2 n)$; 검증자 시간은 $O(\log^2 n)$
- **투명(Transparent)**: 신뢰 설정 불필요 — 해시 함수만 사용
- **포스트양자(Post-Quantum)**: 충돌 저항 해싱에 기반한 보안 (양자 컴퓨터에도 안전)

### 7.2 SNARK와의 주요 차이점

| 속성 | zk-SNARK | zk-STARK |
|----------|-----------|-----------|
| 신뢰 설정 | 필요 (대부분) | **불필요** |
| 증명 크기 | ~128~512바이트 | ~40~200 KB |
| 검증 시간 | ~2~5 ms | ~10~50 ms |
| 포스트양자 | 아니오 (페어링/DLP에 의존) | **예** (해시 기반) |
| 암호학적 가정 | 타원 곡선 페어링, 지식 가정 | 충돌 저항 해싱 |
| 증명자 복잡도 | $O(n \log n)$ | $O(n \log^2 n)$ |

### 7.3 트레이드오프

- **SNARK 우세**: 증명 크기가 중요할 때 (블록체인 온체인 검증, 대역폭 제한)
- **STARK 우세**: 신뢰가 중요할 때 (세리머니 없음), 양자 저항이 필요할 때, 또는 오프체인에서 증명이 검증될 때

### 7.4 하이브리드 방식

현대 시스템은 종종 기법을 결합합니다:
- **재귀 SNARK(Recursive SNARK)**: 다른 SNARK를 검증하는 SNARK (STARK를 압축)
- **SNARK-of-STARK**: 계산의 대부분에는 STARK를 사용하고, 온체인 검증을 위해 SNARK로 증명을 압축

---

## 8. 응용

### 8.1 블록체인 프라이버시

**Zcash** (2016): zk-SNARK를 사용하여 거래가 유효함을 증명 (입력 = 출력, 발신자에게 충분한 자금이 있음) — 발신자, 수신자 또는 금액을 공개하지 않고. 이는 "프라이빗 화폐"입니다 — 디지털 현금과 같습니다.

**zkRollup** (이더리움 L2): 수천 건의 거래를 오프체인에서 배치 처리하고, 정확성에 대한 단일 SNARK/STARK 증명을 생성하여 체인 위에는 증명만 게시합니다. 이는 기본 체인과 동일한 보안으로 확장성을 제공합니다 (1000배 처리량 향상).

### 8.2 탈중앙화 신원

**ZKP가 적용된 검증 가능한 자격 증명(Verifiable Credentials):**
- "나는 18세 이상이다"를 증명 — 생년월일 공개 없이
- "나는 면허가 있는 의사다"를 증명 — 면허 번호 공개 없이
- "나는 EU에 거주한다"를 증명 — 주소 공개 없이

표준: W3C Verifiable Credentials, BBS+ 서명 (선택적 공개 허용)

### 8.3 투표

ZKP는 **검증 가능하면서도 비밀인** 투표를 가능하게 합니다:
- 각 투표자는 자격이 있고 정확히 한 번 투표했음을 증명
- 개별 투표를 공개하지 않고 집계를 검증 가능
- 시스템: Vocdoni, MACI (Minimum Anti-Collusion Infrastructure)

### 8.4 검증 가능한 계산

계산을 신뢰할 수 없는 서버에 아웃소싱하고 결과를 효율적으로 검증:
- 클라우드 컴퓨팅: AWS가 실제로 계산을 올바르게 실행했는지 검증
- 머신러닝: 데이터를 공개하지 않고 특정 데이터에 대해 모델이 훈련되었음을 증명
- 공급망: 제조 프로세스를 공개하지 않고 제품이 사양을 충족함을 증명

---

## 9. 요약

| 개념 | 핵심 요점 |
|---------|-------------|
| 영지식 | 기반 증인을 공개하지 않고 진술을 증명 |
| 완전성/건전성/영지식성 | 모든 ZKP가 만족해야 하는 세 가지 속성 |
| 슈노르 프로토콜 | 이산 대수 지식에 대한 간단한 ZKP; 슈노르 서명의 기초 |
| 피아트-샤미르 | 해시 함수를 사용하여 대화형 증명을 비대화형으로 변환 |
| 그래프 색칠 ZKP | 모든 NP 진술이 ZKP를 가질 수 있음을 증명 |
| zk-SNARK | 매우 작은 증명 (~128바이트), 빠른 검증, 하지만 신뢰 설정 필요 |
| zk-STARK | 더 큰 증명 (~100 KB), 신뢰 설정 불필요, 포스트양자 |
| 응용 | 프라이버시 (Zcash), 확장성 (zkRollup), 신원, 투표 |

---

## 10. 연습 문제

### 연습 문제 1: 슈노르 프로토콜 보안 (개념)

1. 증명자가 두 가지 다른 챌린지에 대해 동일한 무작위 $r$을 재사용하면 어떤 일이 발생하는지 설명하세요. 검증자가 비밀 키 $x$를 어떻게 추출할 수 있는지 보여주세요.
2. 2010년 PlayStation 3 해킹은 ECDSA에서 이 취약점을 정확히 악용했습니다. 어떻게 작동했는지 조사하고 요약하세요.
3. $r$이 암호학적으로 안전한 난수 생성기를 사용하여 생성되어야 하고 현재 시간으로 시드된 PRNG가 아니어야 하는 이유는 무엇인가요?

### 연습 문제 2: 피아트-샤미르 구현 (코딩)

피아트-샤미르 슈노르 예시를 확장하여 완전한 비대화형 ZKP 시스템 만들기:
1. 임의의 (안전한) 소수 군 파라미터에 대한 비밀 키 지식 증명 허용
2. `prove(secret_key, message)` 함수와 `verify(public_key, message, proof)` 함수 포함
3. 서로 다른 메시지에 대한 증명이 연결 불가능함을 보여주기 (검증자는 두 증명이 같은 증명자로부터 왔는지 알 수 없어야 함)

### 연습 문제 3: 그래프 색칠 건전성 (코딩 + 개념)

1. **유효하지 않은** 색칠(적어도 하나의 간선의 양 끝점이 같은 색상)을 사용하도록 그래프 색칠 ZKP를 수정
2. 증가하는 라운드 수(10, 50, 100, 500)에 대해 프로토콜 실행
3. 라운드 수 대비 부정직한 증명자가 잡힐 확률 플롯
4. 99.99% 신뢰도를 위해 몇 라운드가 필요한가요?

### 연습 문제 4: 커밋먼트 방식 (코딩)

더 완전한 커밋먼트 방식 구현:
1. **페더슨 커밋먼트(Pedersen Commitment)**: $C = g^v h^r$ (계산적으로 숨김, 완전 결합)
2. 페더슨 커밋먼트가 준동형(Homomorphic)임을 보여주기: $C_1 \cdot C_2 = \text{Commit}(v_1 + v_2, r_1 + r_2)$
3. 해시 기반 커밋먼트 대신 페더슨 커밋먼트를 그래프 색칠 ZKP에 사용
4. 준동형 커밋먼트가 효율적인 ZKP를 구축하는 데 왜 중요한지 설명

### 연습 문제 5: ZKP 응용 설계 (심화)

암호화폐 거래소에 대한 간소화된 "지급 능력 증명(Proof of Solvency)" ZKP 시스템 설계:
1. 거래소는 계정 잔액의 머클 트리를 가지고 있음 (잎은 $(account\_id, balance)$)
2. 거래소는 모든 잔액의 합이 공개적으로 알려진 총액과 같음을 증명해야 함
3. 개별 계정 잔액은 공개되지 않아야 함
4. 프로토콜 개요 스케치 (증명자는 무엇을 계산하나요? 검증자는 무엇을 확인하나요?)
5. 어떤 ZKP 시스템을 사용하겠으며 그 이유는? (슈노르, 그래프 색칠, SNARK, STARK?)

# 레슨 11: 포스트양자 암호(Post-Quantum Cryptography)

**이전**: [격자 기반 암호](./10_Lattice_Based_Cryptography.md) | **다음**: [영지식 증명](./12_Zero_Knowledge_Proofs.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 쇼어 알고리즘(Shor's algorithm)이 RSA, Diffie-Hellman, ECC를 어떻게 깨는지 설명하기
2. 그로버 알고리즘(Grover's algorithm)과 대칭 암호에 미치는 영향 설명하기
3. NIST 포스트양자 암호(Post-Quantum Cryptography) 표준화 과정 추적 및 선정된 알고리즘 식별하기
4. 포스트양자 암호의 4가지 계열(격자, 코드, 해시, 아이소제니)을 비교하기
5. SIKE가 깨진 이유와 그것이 주는 암호학적 신뢰에 대한 교훈을 설명하기
6. 조직을 위한 하이브리드 암호 마이그레이션 전략을 설계하기
7. 암호 민첩성(Crypto Agility) 개념을 시스템에 적용하여 미래에 대비하기

---

오늘날 인터넷에 배포된 모든 공개 키 암호 시스템 — RSA, Diffie-Hellman, ECDSA, EdDSA — 은 정수 인수분해 또는 이산 대수 문제의 어려움에 기반합니다. 1994년 Peter Shor는 충분히 큰 양자 컴퓨터가 두 문제를 모두 다항 시간 내에 풀 수 있음을 증명했습니다. 현재의 양자 컴퓨터는 아직 너무 소규모이고 오류가 많지만, 위협은 "지금 수집하고 나중에 복호화(harvest now, decrypt later)" 모델을 따릅니다: 오늘날의 암호화된 트래픽을 기록하는 적대자는 양자 컴퓨터가 성숙하면 이를 복호화할 수 있습니다. 이 레슨은 양자 위협 환경과 이에 맞서기 위해 설계된 새로운 암호 시스템을 개관합니다.

> **비유:** 포스트양자 암호는 기후 변화가 닥치기 전에 홍수 방벽을 미리 쌓는 것과 같습니다. 양자 컴퓨터가 RSA를 깰 때까지 마이그레이션을 미룰 수는 없습니다 — 전환에는 수년이 걸리고, 기록된 데이터는 수십 년 동안 비밀로 유지되어야 할 수 있습니다.

## 목차

1. [양자 컴퓨팅 위협](#1-양자-컴퓨팅-위협)
2. [쇼어 알고리즘](#2-쇼어-알고리즘)
3. [그로버 알고리즘](#3-그로버-알고리즘)
4. [NIST PQC 표준화](#4-nist-pqc-표준화)
5. [격자 기반 암호 (선정 알고리즘)](#5-격자-기반-암호-선정-알고리즘)
6. [코드 기반 암호](#6-코드-기반-암호)
7. [해시 기반 서명](#7-해시-기반-서명)
8. [아이소제니 기반 암호 (경계의 사례)](#8-아이소제니-기반-암호-경계의-사례)
9. [하이브리드 방식과 암호 민첩성](#9-하이브리드-방식과-암호-민첩성)
10. [마이그레이션 로드맵](#10-마이그레이션-로드맵)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)

---

## 1. 양자 컴퓨팅 위협

### 1.1 암호학자를 위한 양자 컴퓨팅 기초

고전 컴퓨터는 비트(0 또는 1)를 조작합니다. 양자 컴퓨터는 중첩 상태(superposition)에 존재할 수 있는 **큐비트(qubit)**를 조작합니다:

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1
$$

두 가지 양자 현상이 계산상의 이점을 가능하게 합니다:

- **중첩(Superposition)**: $n$개의 큐비트로 $2^n$개의 상태를 동시에 처리
- **얽힘(Entanglement)**: 큐비트는 고전적인 방식으로는 설명할 수 없는 방식으로 상관될 수 있음

### 1.2 암호학에 대한 영향

| 암호학적 기본 요소 | 고전적 보안 | 양자 영향 |
|------------------------|-------------------|----------------|
| AES-128 | 128비트 | 64비트 (그로버) — 키 길이 두 배 필요 |
| AES-256 | 256비트 | 128비트 (그로버) — 여전히 안전 |
| SHA-256 | 256비트 프리이미지 | 128비트 (그로버) — 여전히 안전 |
| RSA-2048 | ~112비트 | **깨짐** (쇼어) |
| ECDSA P-256 | 128비트 | **깨짐** (쇼어) |
| DH-2048 | ~112비트 | **깨짐** (쇼어) |
| Kyber-768 | 192비트 (PQ) | 192비트 — 저항하도록 설계됨 |

### 1.3 타임라인 추정

양자 컴퓨터가 RSA-2048을 깰 만큼 충분히 커지려면 언제까지 걸릴까요?

- **낙관적** (양자 컴퓨팅 기업들): 2030~2035년
- **주류** (학계 컨센서스): 2035~2050년
- **보수적**: 2050년 이후
- **NSA/NIST 입장**: 타임라인과 관계없이 지금 당장 마이그레이션 시작

핵심 통찰: 위협이 15~20년 후라도, 오늘 암호화된 데이터는 30년 이상 기밀로 유지되어야 할 수 있습니다(의료 기록, 기밀 정보, 영업 비밀). 대규모 조직의 마이그레이션 자체에는 5~10년이 걸립니다.

---

## 2. 쇼어 알고리즘

### 2.1 핵심 아이디어

쇼어 알고리즘은 **양자 푸리에 변환(Quantum Fourier Transform, QFT)**을 사용하여 함수의 **주기(period)**를 효율적으로 찾습니다. 인수분해와 이산 대수 문제 모두 주기 찾기로 환원될 수 있습니다.

**주기 찾기를 통한 인수분해:**
1. 인수분해할 $N$이 주어지면, 임의의 $a < N$을 선택
2. $f(x) = a^x \bmod N$의 주기 $r$을 찾음 (즉, $a^r \equiv 1 \pmod{N}$을 만족하는 가장 작은 $r$)
3. $r$이 짝수이면, $\gcd(a^{r/2} \pm 1, N)$을 계산 — 이는 높은 확률로 $N$의 인수를 산출

양자 부분은 2단계입니다: 고전적으로 주기를 찾으려면 지수 시간이 필요하지만, QFT는 다항 시간 내에 수행합니다.

### 2.2 개념 시연

```python
"""
Shor's algorithm — classical simulation of the period-finding step.

Why a classical simulation? A real quantum implementation requires
a quantum computer. This code demonstrates the mathematical reduction
from factoring to period-finding, which is the classical part of Shor's.

WARNING: This is exponentially slow for large N (it brute-forces the
period classically). The quantum speedup replaces this brute force
with polynomial-time QFT.
"""

import math
import random


def classical_period_finding(a: int, N: int) -> int:
    """
    Find the period r of f(x) = a^x mod N classically.

    This is the step that Shor's quantum algorithm accelerates
    from O(exp(n)) to O(poly(n)).
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            return -1  # No period found (shouldn't happen if gcd(a,N)=1)
    return r


def shors_factoring(N: int, max_attempts: int = 20) -> tuple[int, int]:
    """
    Simulate Shor's algorithm to factor N.

    Steps:
    1. Choose random a
    2. Check gcd(a, N) — if > 1, we got lucky
    3. Find period r of a^x mod N
    4. If r is even, try gcd(a^(r/2) ± 1, N)
    """
    if N % 2 == 0:
        return 2, N // 2

    for attempt in range(max_attempts):
        a = random.randint(2, N - 1)

        # Lucky case: a shares a factor with N
        g = math.gcd(a, N)
        if g > 1:
            print(f"  Attempt {attempt + 1}: Lucky! gcd({a}, {N}) = {g}")
            return g, N // g

        # Find the period (quantum computer does this efficiently)
        r = classical_period_finding(a, N)

        if r == -1 or r % 2 != 0:
            print(f"  Attempt {attempt + 1}: a={a}, r={r} (odd or not found, retry)")
            continue

        # Try to extract a factor
        # Why a^(r/2)? Because a^r ≡ 1 (mod N) means
        # (a^(r/2))^2 ≡ 1 (mod N), so (a^(r/2) - 1)(a^(r/2) + 1) ≡ 0 (mod N)
        x = pow(a, r // 2, N)

        if x == N - 1:  # Trivial: x ≡ -1 (mod N)
            print(f"  Attempt {attempt + 1}: a={a}, r={r} (trivial, retry)")
            continue

        factor1 = math.gcd(x - 1, N)
        factor2 = math.gcd(x + 1, N)

        if factor1 > 1 and factor1 < N:
            print(f"  Attempt {attempt + 1}: a={a}, r={r} → "
                  f"factors: {factor1} × {N // factor1}")
            return factor1, N // factor1

        if factor2 > 1 and factor2 < N:
            print(f"  Attempt {attempt + 1}: a={a}, r={r} → "
                  f"factors: {factor2} × {N // factor2}")
            return factor2, N // factor2

    return None, None


def demo_shors():
    """Demonstrate Shor's factoring on small numbers."""
    test_numbers = [15, 21, 35, 77, 91, 143, 221, 323]

    for N in test_numbers:
        print(f"\nFactoring N = {N}:")
        p, q = shors_factoring(N)
        if p and q:
            assert p * q == N, f"Factoring failed: {p} × {q} ≠ {N}"
            print(f"  Result: {N} = {p} × {q}")
        else:
            print(f"  Failed to factor {N}")


if __name__ == "__main__":
    demo_shors()
```

### 2.3 RSA를 깨기 위한 자원 추정

쇼어 알고리즘으로 2048비트 RSA 모듈러스를 인수분해하려면:

| 자원 | 추정치 |
|----------|----------|
| 논리적 큐비트 | ~4,000 |
| 물리적 큐비트 (오류 수정 포함) | ~4,000,000 (현재 오류율 기준) |
| 게이트 연산 | ~$10^{12}$ |
| 실제 소요 시간 | 수 시간에서 수 일 |

현재 최신 기술: IBM의 가장 큰 프로세서는 ~1,000개의 물리적 큐비트를 갖추고 있습니다. 격차는 여전히 크지만 좁혀지고 있습니다.

### 2.4 ECC와 DH 공격

쇼어 알고리즘은 이산 대수 문제도 효율적으로 풉니다:
- **DH/DSA**: $g^a \bmod p$에서 $a$를 $O((\log p)^3)$에 찾음
- **ECDLP**: 타원 곡선 위에서 $kG$로부터 $k$를 $O((\log n)^3)$에 찾음

실제로 ECC는 관련된 숫자가 더 작기 때문에(256비트 vs 2048비트) RSA보다 **더 적은** 큐비트로 깰 수 있습니다.

---

## 3. 그로버 알고리즘

### 3.1 탐색에 대한 이차 속도 향상

그로버 알고리즘은 $N$개의 항목으로 이루어진 비구조화 데이터베이스에서 대상 항목을 고전적인 $O(N)$에 비해 $O(\sqrt{N})$번의 질의만으로 찾습니다.

### 3.2 대칭 암호에 대한 영향

| 암호 | 고전적 보안 | 포스트양자 보안 | 대응 |
|--------|-------------------|----------------------|--------|
| AES-128 | $2^{128}$ | $2^{64}$ | AES-256으로 업그레이드 |
| AES-256 | $2^{256}$ | $2^{128}$ | 이미 충분 |
| ChaCha20 | $2^{256}$ | $2^{128}$ | 이미 충분 |

### 3.3 해시 함수에 대한 영향

| 해시 | 프리이미지 (고전) | 프리이미지 (양자) | 충돌 (양자) |
|------|---------------------|-------------------|---------------------|
| SHA-256 | $2^{256}$ | $2^{128}$ | $2^{85}$ (BHT 알고리즘) |
| SHA-384 | $2^{384}$ | $2^{192}$ | $2^{128}$ |
| SHA-512 | $2^{512}$ | $2^{256}$ | $2^{170}$ |

SHA-256은 프리이미지 저항성은 유지되지만 충돌 저항성이 감소합니다. 장기적인 보안을 위해서는 SHA-384 또는 SHA-512가 권장됩니다.

### 3.4 그로버가 쇼어보다 덜 우려스러운 이유

- **쇼어**: 지수적 속도 향상 → RSA/ECC를 완전히 파괴 → **대체** 필요
- **그로버**: 이차적 속도 향상 → 보안 비트를 절반으로 줄임 → **파라미터 두 배 증가**만으로 충분

---

## 4. NIST PQC 표준화

### 4.1 타임라인

| 날짜 | 마일스톤 |
|------|-----------|
| 2016 | NIST가 PQC 제안 공모 |
| 2017 | 82개 제출물 접수 |
| 2019 | 2라운드: 26개 후보 |
| 2020 | 3라운드: 15개 후보 (결선 진출자 7개 + 대안 8개) |
| 2022 | 선정 알고리즘 발표 |
| 2024 | FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA) 발행 |
| 2024+ | 추가 서명을 위한 4라운드 계속 |

### 4.2 선정 알고리즘 및 표준

| 표준 | 알고리즘 | 계열 | 목적 |
|----------|-----------|--------|---------|
| **FIPS 203** (ML-KEM) | CRYSTALS-Kyber | 격자 | 키 캡슐화 |
| **FIPS 204** (ML-DSA) | CRYSTALS-Dilithium | 격자 | 디지털 서명 |
| **FIPS 205** (SLH-DSA) | SPHINCS+ | 해시 기반 | 디지털 서명 (무상태) |
| 4라운드 | FALCON | 격자 | 서명 (소형) |
| 4라운드 | BIKE, HQC, Classic McEliece | 코드 기반 | KEM (다양성) |

### 4.3 격자가 지배한 이유

- **성능**: 격자 방식은 빠름 (종종 RSA보다 빠름)
- **키 크기**: 관리 가능 (1~2 KB, 코드 기반의 100+ KB에 비해)
- **다용도성**: 동일한 프레임워크로 KEM, 서명, 고급 프로토콜(FHE, ZKP) 지원
- **이론적 기반**: 최악의 경우에서 평균적인 경우로의 강력한 환원 관계

---

## 5. 격자 기반 암호 (선정 알고리즘)

레슨 10에서 자세히 다루었습니다. PQC 맥락에서의 핵심 요점:

### 5.1 ML-KEM (Kyber) 요약

- Module-LWE 기반 키 캡슐화
- 공개 키: 보안 수준에 따라 800~1568바이트
- 암호문: 768~1568바이트
- Chrome, Signal, iMessage에 이미 배포됨

### 5.2 ML-DSA (Dilithium) 요약

- Module-LWE 기반 디지털 서명
- 서명: 2420~4595바이트
- 대부분의 벤치마크에서 가장 빠른 PQC 서명 방식

### 5.3 FALCON

- 가우시안 샘플링을 사용하는 NTRU 격자 기반 서명
- Dilithium보다 **훨씬 작은 서명** (~NIST 1레벨에서 666바이트 vs 2420바이트)
- 더 복잡한 구현 (가우시안 샘플링을 위한 배정밀도 부동소수점 필요)
- 후속 라운드에서 표준화 중

---

## 6. 코드 기반 암호

### 6.1 아이디어

오류 정정 코드(Error-Correcting Code)는 전송된 데이터의 오류를 감지하고 수정할 수 있습니다. 코드 기반 암호는 이를 역으로 이용합니다: 코드 구조가 트랩도어(trapdoor)이며, 이를 알지 못하면 복호화가 NP-난해 문제가 됩니다.

### 6.2 McEliece 암호 시스템 (1978)

가장 오래된 깨지지 않은 포스트양자 암호 시스템:

1. **키 생성**: 생성 행렬 $G$를 가진 이진 Goppa 코드를 선택합니다. 이를 섞습니다: $G' = SGP$ (여기서 $S$는 역행렬이 존재하고 $P$는 순열 행렬)
2. **암호화**: $c = mG' + e$ (여기서 $e$는 무게 $t$의 임의 오류 벡터)
3. **복호화**: $P^{-1}$을 적용하고, Goppa 코드로 복호화한 후, $S^{-1}$을 적용

**단점**: 공개 키가 엄청나게 큽니다.

| 방식 | 공개 키 | 암호문 | 보안 수준 |
|--------|-----------|------------|----------------|
| Classic McEliece-348864 | **261 KB** | 128 B | NIST 레벨 1 |
| Classic McEliece-6960119 | **1,044 KB** | 226 B | NIST 레벨 5 |

> 261 KB 공개 키는 Kyber-768의 1.2 KB 키보다 ~200배 큽니다. 이로 인해 McEliece는 대부분의 인터넷 프로토콜에는 비실용적이지만, 키를 미리 배포하는 특수 응용 프로그램에는 적합합니다.

### 6.3 BIKE와 HQC

BIKE(Bit Flipping Key Encapsulation)와 HQC(Hamming Quasi-Cyclic)는 훨씬 작은 키(~2~5 KB)를 가진 코드 기반 KEM이지만, 신중하게 분석해야 하는 무시할 수 없는 복호화 실패율을 가지고 있습니다.

---

## 7. 해시 기반 서명

### 7.1 매력

해시 기반 서명은 암호학에서 가장 보수적이고 잘 이해된 가정인 해시 함수의 충돌 저항성**만으로** 보안을 도출합니다. SHA-256이 안전하다면, 이러한 서명도 안전합니다.

### 7.2 일회용 서명 (Lamport/Winternitz)

**Lamport의 일회용 서명** (1979):
1. $2n$개의 임의 값 생성: $(\text{sk}_0^1, \text{sk}_1^1, \ldots, \text{sk}_0^n, \text{sk}_1^n)$
2. 공개 키: 각 값을 해시 $(\text{pk}_0^1 = H(\text{sk}_0^1), \ldots)$
3. 메시지 해시의 비트 $b_i$에 서명: $\text{sk}_{b_i}^i$를 공개
4. 검증: 공개된 각 값을 해시하고 공개 키와 비교

**문제**: 각 키 쌍은 정확히 **하나의** 메시지에만 서명할 수 있습니다. 재사용하면 개인 키가 누출됩니다.

### 7.3 머클 트리(Merkle Tree): 일회용 키에서 다중 서명으로

Ralph Merkle의 통찰: $2^h$개의 일회용 키 쌍을 이진 해시 트리의 잎으로 구성합니다. 루트 해시가 공개 키입니다.

```
         Root (public key)
        /                \
      H01                H23
     /    \            /    \
   H0      H1       H2      H3
   |       |        |       |
  OTS_0   OTS_1   OTS_2   OTS_3
```

키 $i$로 서명하려면: OTS 서명과 인증 경로(잎에서 루트까지의 형제 해시)를 제공합니다.

**제한**: 트리의 용량은 유한합니다($2^h$개의 서명). 서명자는 어떤 키를 사용했는지 추적해야 합니다.

### 7.4 XMSS (eXtended Merkle Signature Scheme)

XMSS (RFC 8391)는 **상태 유지(stateful)** 해시 기반 서명 방식입니다:
- Winternitz 일회용 서명의 머클 트리 사용
- 서명자는 상태(다음 미사용 키의 인덱스)를 유지해야 함
- 매우 작은 서명 (~2.5 KB)과 빠른 검증

**위험**: 상태가 손실되거나 복제되면(예: VM 스냅샷 복원), 동일한 OTS 키가 재사용될 수 있어 보안이 완전히 깨집니다.

### 7.5 SPHINCS+ (SLH-DSA)

SPHINCS+ (FIPS 205에서 SLH-DSA로 표준화)는 **무상태(stateless)** 해시 기반 서명입니다:
- 잎에 FORS(Forest of Random Subsets)를 가진 머클 트리의 하이퍼 트리 사용
- 관리할 상태 없음 — 동일한 키로 무제한 메시지에 안전하게 서명 가능
- **트레이드오프**: 더 큰 서명 (~파라미터에 따라 8~50 KB)

```python
"""
Compare PQC signature sizes.

Why SPHINCS+ despite large signatures? It provides the most conservative
security assumption (only hash function security) and is stateless.
For environments where state management is risky (distributed systems,
HSMs with backup), SPHINCS+ is the safest choice.
"""


def compare_pqc_signatures():
    """Compare signature scheme sizes and properties."""

    schemes = [
        {
            "name": "ECDSA P-256",
            "family": "Elliptic Curve",
            "pubkey_bytes": 64,
            "sig_bytes": 64,
            "quantum_safe": False,
            "stateful": False,
            "assumption": "ECDLP",
        },
        {
            "name": "Ed25519",
            "family": "Elliptic Curve",
            "pubkey_bytes": 32,
            "sig_bytes": 64,
            "quantum_safe": False,
            "stateful": False,
            "assumption": "ECDLP",
        },
        {
            "name": "ML-DSA-65 (Dilithium3)",
            "family": "Lattice",
            "pubkey_bytes": 1952,
            "sig_bytes": 3293,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "Module-LWE",
        },
        {
            "name": "FALCON-512",
            "family": "Lattice",
            "pubkey_bytes": 897,
            "sig_bytes": 666,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "NTRU lattice",
        },
        {
            "name": "SLH-DSA-SHA2-128s (SPHINCS+)",
            "family": "Hash-based",
            "pubkey_bytes": 32,
            "sig_bytes": 7856,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "Hash function",
        },
        {
            "name": "SLH-DSA-SHA2-128f (SPHINCS+)",
            "family": "Hash-based",
            "pubkey_bytes": 32,
            "sig_bytes": 17088,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "Hash function (fast variant)",
        },
        {
            "name": "XMSS (h=10)",
            "family": "Hash-based",
            "pubkey_bytes": 64,
            "sig_bytes": 2500,
            "quantum_safe": True,
            "stateful": True,
            "assumption": "Hash function",
        },
    ]

    print(f"{'Scheme':<35} {'Family':<15} {'PubKey':>8} {'Sig':>8} "
          f"{'PQ':>4} {'State':>6}")
    print("-" * 85)
    for s in schemes:
        pq = "Yes" if s["quantum_safe"] else "No"
        st = "Yes" if s["stateful"] else "No"
        print(f"{s['name']:<35} {s['family']:<15} "
              f"{s['pubkey_bytes']:>6} B {s['sig_bytes']:>6} B "
              f"{pq:>4} {st:>6}")

    # Highlight the trade-off
    print("\n--- Key Trade-offs ---")
    print("FALCON: Smallest PQ signatures but complex implementation")
    print("Dilithium: Best general-purpose PQ signature (balanced)")
    print("SPHINCS+: Most conservative assumption but largest signatures")
    print("XMSS: Small signatures but requires careful state management")


if __name__ == "__main__":
    compare_pqc_signatures()
```

---

## 8. 아이소제니 기반 암호 (경계의 사례)

### 8.1 아이소제니(Isogeny)란?

**아이소제니**는 타원 곡선 사이의 구조를 보존하는 사상(map)입니다. 아이소제니 기반 암호의 보안은 두 개의 초특이(supersingular) 타원 곡선 사이에서 아이소제니를 찾는 어려움에 기반했습니다.

### 8.2 SIKE/SIDH

**SIDH**(Supersingular Isogeny Diffie-Hellman)와 그 KEM 변형인 **SIKE**는 놀랍도록 작은 키 크기를 제공했습니다:

| 방식 | 공개 키 | 암호문 |
|--------|-----------|------------|
| SIKE-p434 | 330 B | 346 B |
| Kyber-512 | 800 B | 768 B |

SIKE는 NIST 경쟁의 4라운드에 진출했습니다.

### 8.3 충격적인 공격 (2022)

2022년 7월, Wouter Castryck과 Thomas Decru는 단일 랩톱에서 **몇 분** 만에 SIKE를 깨는 치명적인 공격을 발표했습니다. 이 공격은 SIDH가 아이소제니와 함께 공개한 보조 비틀림 점(torsion point) 정보를 이용했습니다.

**핵심 교훈:**

1. **다양성이 중요**: 단일 가정에 의존하는 것은 위험합니다. NIST는 현명하게도 여러 계열에서 알고리즘을 선정했습니다.
2. **암호분석적 성숙도**: 격자 문제는 40년 이상 연구되었고, 아이소제니 문제는 ~10년 연구되었습니다. 더 많은 검토는 더 많은 공격을 발견합니다.
3. **보조 정보는 위험**: SIDH의 취약점은 핵심 아이소제니 문제가 아니라 프로토콜을 작동시키기 위해 필요한 추가 정보(비틀림 점)에서 비롯됐습니다.

> **참고**: SIKE의 공격이 모든 아이소제니 기반 암호를 무효화하는 것은 아닙니다. 새로운 제안(CSIDH, SQISign 등)은 비틀림 점을 공개하지 않아 취약점을 피하지만, 더 느리고 키가 더 큽니다.

---

## 9. 하이브리드 방식과 암호 민첩성

### 9.1 왜 하이브리드인가?

전환 기간 동안 다음 중 어느 것도 완전히 신뢰할 수 없습니다:
- **고전 암호만**: 미래의 양자 컴퓨터에 취약
- **PQC만**: 비교적 새롭고, 발견되지 않은 취약점이 있을 수 있음 (SIKE가 보여준 것처럼)

**하이브리드 방식**은 둘 다 결합합니다:

$$
\text{SharedSecret} = \text{KDF}(\text{ECDH\_secret} \| \text{Kyber\_secret})
$$

어느 한 방식이 깨지더라도, 나머지 방식이 여전히 보안을 제공합니다. 둘 다 안전하다면, 결합된 방식은 더 강한 것만큼은 안전합니다.

### 9.2 실제 하이브리드 배포

| 시스템 | 고전 | 포스트양자 | 도입 시기 |
|--------|-----------|-------------|-------|
| Chrome/TLS | X25519 | Kyber-768 | 2023년 8월 |
| Signal | X25519 | Kyber-1024 | 2023년 9월 |
| Apple iMessage | ECDH P-256 | Kyber-768 | 2024년 2월 |
| Cloudflare | X25519 | Kyber-768 | 2023년 |
| AWS KMS | ECDH | Kyber-768 | 2024년 |

### 9.3 암호 민첩성(Crypto Agility)

**암호 민첩성**은 전체 시스템을 재설계하지 않고 암호 알고리즘을 교체할 수 있는 능력입니다. 이를 위해서는 다음이 필요합니다:

1. **알고리즘 협상**: 프로토콜은 런타임에 알고리즘을 협상해야 함 (TLS 암호 스위트가 이미 이를 수행함)
2. **추상화 레이어**: 애플리케이션 코드는 특정 알고리즘을 하드코딩하지 않아야 함
3. **설정 기반**: 알고리즘은 컴파일 시 고정이 아닌 설정 가능해야 함
4. **점진적 업그레이드**: 전환 기간 동안 이전 알고리즘과 새 알고리즘을 모두 지원

```python
"""
Crypto agility pattern — algorithm-agnostic key exchange.

Why crypto agility? When a cryptographic algorithm is broken (like SIKE)
or deprecated, the system should be able to switch algorithms with
configuration changes, not code rewrites.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class KeyPair:
    public_key: bytes
    private_key: bytes


@dataclass
class KEMResult:
    shared_secret: bytes
    ciphertext: bytes


class KEMScheme(ABC):
    """Abstract KEM interface — any algorithm can implement this."""

    @abstractmethod
    def keygen(self) -> KeyPair:
        """Generate a key pair."""
        ...

    @abstractmethod
    def encapsulate(self, public_key: bytes) -> KEMResult:
        """Encapsulate: generate shared secret and ciphertext."""
        ...

    @abstractmethod
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate: recover shared secret from ciphertext."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class HybridKEM(KEMScheme):
    """
    Combine two KEMs for hybrid security.

    Why combine with KDF? Simple concatenation might have subtle issues
    if one scheme's output is correlated with the other's. The KDF
    ensures the combined secret is uniformly random.
    """

    def __init__(self, classical: KEMScheme, post_quantum: KEMScheme):
        self.classical = classical
        self.post_quantum = post_quantum

    @property
    def name(self) -> str:
        return f"Hybrid({self.classical.name}+{self.post_quantum.name})"

    def keygen(self) -> KeyPair:
        ck = self.classical.keygen()
        pk = self.post_quantum.keygen()
        # In practice, keys would be serialized with length prefixes
        return KeyPair(
            public_key=ck.public_key + pk.public_key,
            private_key=ck.private_key + pk.private_key,
        )

    def encapsulate(self, public_key: bytes) -> KEMResult:
        # Split public key (in practice, use proper serialization)
        mid = len(public_key) // 2
        c_result = self.classical.encapsulate(public_key[:mid])
        p_result = self.post_quantum.encapsulate(public_key[mid:])

        import hashlib
        combined_secret = hashlib.sha256(
            c_result.shared_secret + p_result.shared_secret
        ).digest()

        return KEMResult(
            shared_secret=combined_secret,
            ciphertext=c_result.ciphertext + p_result.ciphertext,
        )

    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        mid_key = len(private_key) // 2
        mid_ct = len(ciphertext) // 2

        c_secret = self.classical.decapsulate(
            private_key[:mid_key], ciphertext[:mid_ct]
        )
        p_secret = self.post_quantum.decapsulate(
            private_key[mid_key:], ciphertext[mid_ct:]
        )

        import hashlib
        return hashlib.sha256(c_secret + p_secret).digest()


# Usage:
# kem = HybridKEM(X25519KEM(), KyberKEM())
# keys = kem.keygen()
# result = kem.encapsulate(keys.public_key)
# recovered = kem.decapsulate(keys.private_key, result.ciphertext)
# assert result.shared_secret == recovered
```

---

## 10. 마이그레이션 로드맵

### 10.1 PQC 마이그레이션 단계

| 단계 | 타임라인 | 작업 |
|-------|----------|---------|
| **1. 인벤토리** | 현재 | 모든 암호학적 사용 목록 작성 (라이브러리, 프로토콜, 인증서, 하드웨어) |
| **2. 평가** | 현재 | 가장 위험성이 높은 시스템 식별 (수명이 긴 비밀, 컴플라이언스 요건) |
| **3. 테스트** | 현재~2026 | 테스트 환경에 하이브리드 방식 배포, 성능 영향 벤치마크 |
| **4. 하이브리드 배포** | 2025~2028 | 키 교환에 하이브리드 (고전 + PQC)를 프로덕션에 배포 |
| **5. PQC 서명** | 2026~2030 | 인증서 체인, 코드 서명을 PQC 서명으로 마이그레이션 |
| **6. 완전한 PQC** | 2030+ | 고전 전용 암호 스위트 제거 |

### 10.2 도전 과제

- **인증서 체인**: TLS 체인의 각 인증서가 ~2~3 KB 증가 (Dilithium). 3개 인증서 체인은 모든 TLS 핸드셰이크에 ~7~9 KB를 추가합니다.
- **임베디드/IoT**: 제한된 장치는 PQC를 위한 리소스가 부족할 수 있음
- **HSM**: 하드웨어 보안 모듈(Hardware Security Module)은 새 알고리즘을 위한 펌웨어 업데이트 필요
- **컴플라이언스**: FIPS, Common Criteria 등 규정이 업데이트되어야 함
- **상호 운용성**: 모든 연결의 양쪽이 PQC를 지원해야 함

### 10.3 정부 의무화

- **NSA CNSA 2.0 (2022)**: 모든 국가 안보 시스템은 2035년까지 PQC로 마이그레이션해야 함
- **백악관 메모 NSM-10 (2022)**: 연방 기관은 양자 취약 암호학을 목록화해야 함
- **NIST IR 8547 (2024)**: PQC 표준으로의 전환 지침
- **EU**: 유럽 사이버보안 인증 프레임워크에 PQC 요건 포함

---

## 11. 요약

| 개념 | 핵심 요점 |
|---------|-------------|
| 쇼어 알고리즘 | 양자 컴퓨터에서 RSA, DH, ECC를 다항 시간 내에 파괴 |
| 그로버 알고리즘 | 대칭 암호의 보안을 절반으로 줄임; 키 길이를 두 배로 늘리면 충분 |
| NIST 표준 | ML-KEM (Kyber), ML-DSA (Dilithium), SLH-DSA (SPHINCS+) |
| 격자 기반 | 성능, 키 크기, 다용도성의 최적 균형 |
| 코드 기반 | 보수적이지만 큰 키 (McEliece: 261 KB) |
| 해시 기반 | 가장 보수적인 가정; SPHINCS+는 무상태이지만 서명이 큼 |
| SIKE | 2022년에 깨짐; 다양성과 암호분석적 성숙도의 필요성을 보여줌 |
| 하이브리드 방식 | 전환 기간 동안 고전 + PQC 결합 |
| 암호 민첩성 | 재작성 없이 알고리즘을 교체할 수 있도록 시스템 설계 |

---

## 12. 연습 문제

### 연습 문제 1: 양자 위협 평가 (개념)

다음 각 시스템에 대해 양자 위협 수준(높음/중간/낮음)을 평가하고 마이그레이션 우선순위를 권장하세요:
1. 병원의 전자 의료 기록 시스템 (30년 보존)
2. 실시간 게임 플랫폼 (세션 키가 몇 분 동안만 유효)
3. 정부 정보 기관의 기밀 통신
4. 블로그 게시물을 제공하는 공개 웹사이트 (모든 콘텐츠가 이미 공개)
5. 장기적으로 개인 키를 저장하는 암호화폐 지갑

### 연습 문제 2: 쇼어 알고리즘 탐구 (코딩)

이 레슨의 고전 주기 찾기 시뮬레이션을 사용하여:
1. 1000 이하의 모든 반소수(두 소수의 곱)를 인수분해하기
2. 각각에 대해 필요한 시도 횟수 기록하기
3. $N$의 크기 대비 성공률 플롯하기
4. 알고리즘이 여러 번 시도해야 하는 이유는? (수학적으로 설명하세요.)

### 연습 문제 3: PQC 키 크기 영향 (코딩)

TLS 핸드셰이크에 대한 PQC의 영향을 측정하는 시뮬레이션 작성:
1. 고전 알고리즘(X25519 + Ed25519)을 사용한 TLS 1.3 핸드셰이크 모델링
2. PQC(Kyber-768 + Dilithium3)로 대체
3. 각 경우에 전송되는 총 바이트 계산
4. 깊이 3의 인증서 체인에서 각 핸드셰이크는 얼마나 커지나요?
5. 어떤 네트워크 지연에서 크기 증가가 눈에 띄는 지연을 유발하나요?

### 연습 문제 4: 하이브리드 KEM 구현 (코딩)

X25519와 단순화된 LWE 방식을 결합하는 하이브리드 KEM 구현:
1. X25519에 `cryptography` 라이브러리 사용
2. 레슨 10의 LWE 구현 사용
3. HKDF로 공유 비밀 결합
4. 한 방식을 "깨도" (비밀을 누출시켜) 하이브리드 공유 비밀이 여전히 안전함을 보여주는 테스트 케이스 작성

### 연습 문제 5: 암호 민첩성 감사 (심화)

오픈 소스 프로젝트(예: 웹 프레임워크, SSH 라이브러리, VPN 클라이언트)를 선택하고 암호 민첩성을 감사하세요:
1. 암호 알고리즘은 어디에 명시되어 있나요? (설정? 소스 코드? 상수?)
2. Kyber 지원을 추가하기가 얼마나 어려울까요?
3. 알고리즘에 무관한 암호에 대한 어떤 추상화가 존재하나요 (또는 없나요)?
4. 해당 프로젝트에 PQC 지원을 추가하는 간략한 마이그레이션 계획을 작성하세요.

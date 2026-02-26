# 레슨 10: 격자 기반 암호

**이전**: [PKI와 인증서](./09_PKI_and_Certificates.md) | **다음**: [포스트양자 암호](./11_Post_Quantum_Cryptography.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 격자(Lattice)를 수학적으로 정의하고 저차원에서 시각화할 수 있다
2. 핵심 격자 문제(SVP, CVP, GapSVP)와 이것이 어렵다고 여겨지는 이유를 설명할 수 있다
3. 오류 포함 학습(LWE) 문제와 그 링 변형(Ring-LWE)을 설명할 수 있다
4. Python으로 간단한 LWE 기반 암호화 방식을 구현할 수 있다
5. CRYSTALS-Kyber(ML-KEM)가 LWE로부터 키 캡슐화 메커니즘을 구성하는 방식을 개략적으로 설명할 수 있다
6. 격자 기반 암호를 보안, 키 크기, 성능 측면에서 RSA 및 ECC와 비교할 수 있다
7. 격자 문제가 양자 공격에 저항한다고 여겨지는 이유를 설명할 수 있다

---

지금까지 공부한 암호 시스템들 — RSA(레슨 5), Diffie-Hellman(레슨 8), ECC(레슨 6) — 은 모두 정수 인수분해와 이산 로그라는 정수론적 문제의 어려움에서 보안을 이끌어냅니다. 1994년 Peter Shor는 충분히 강력한 양자 컴퓨터가 두 문제 모두를 효율적으로 해결하여 이 시스템들을 무용지물로 만들 수 있다는 것을 보였습니다. 격자 기반 암호는 고차원 공간의 기하학적 문제에 기반한 대안적 토대를 제공합니다 — 고전 알고리즘과 양자 알고리즘 모두에 저항하는 것으로 보이는 문제들입니다. 이것은 먼 미래의 이야기가 아닙니다: NIST는 2024년에 격자 기반 방식을 주요 포스트양자 표준으로 선정했으며, 이미 배포가 진행 중입니다.

## 목차

1. [격자란 무엇인가?](#1-격자란-무엇인가)
2. [어려운 격자 문제](#2-어려운-격자-문제)
3. [오류 포함 학습(LWE)](#3-오류-포함-학습lwe)
4. [Ring-LWE와 Module-LWE](#4-ring-lwe와-module-lwe)
5. [NTRU: 역사적 관점](#5-ntru-역사적-관점)
6. [CRYSTALS-Kyber (ML-KEM)](#6-crystals-kyber-ml-kem)
7. [CRYSTALS-Dilithium (ML-DSA)](#7-crystals-dilithium-ml-dsa)
8. [고전 암호학과의 비교](#8-고전-암호학과의-비교)
9. [격자 문제가 양자 공격에 저항하는 이유](#9-격자-문제가-양자-공격에-저항하는-이유)
10. [요약](#10-요약)
11. [연습 문제](#11-연습-문제)

---

## 1. 격자란 무엇인가?

### 1.1 공식 정의

$\mathbb{R}^n$에서의 **격자(lattice)** $\mathcal{L}$은 선형 독립인 기저 벡터 집합 $\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n \in \mathbb{R}^n$의 모든 정수 선형 결합의 집합입니다:

$$
\mathcal{L}(\mathbf{B}) = \left\{ \sum_{i=1}^{n} z_i \mathbf{b}_i \mid z_i \in \mathbb{Z} \right\}
$$

여기서 $\mathbf{B} = [\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n]$은 **기저 행렬(basis matrix)**입니다.

### 1.2 직관

> **비유:** 고차원 격자에서 최단 벡터를 찾는 것은 수백만 개의 역이 있고 지도도 없는 도시에서 가장 가까운 지하철역을 찾는 것과 같습니다. 2D 격자에서는 그냥 주변을 둘러보고 가장 가까운 곳으로 걸어가면 됩니다. 하지만 1000차원의 "도시"에서는 탐색해야 할 방향이 너무 많아서 가장 빠른 컴퓨터(및 양자 컴퓨터)도 최단 경로를 효율적으로 찾을 수 없습니다.

### 1.3 2D 시각화

```python
"""
Visualize a 2D lattice and its basis vectors.

Why start with 2D? Lattice problems are easy in low dimensions
(solvable in polynomial time up to ~4-5D). The difficulty explosion
in high dimensions is what makes lattice crypto possible.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_lattice_2d(basis: np.ndarray, range_val: int = 5,
                     title: str = "2D Lattice"):
    """
    Plot a 2D lattice with its basis vectors.

    Parameters:
        basis: 2x2 matrix where columns are basis vectors
        range_val: range of integer coefficients to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Generate lattice points as integer combinations of basis vectors
    # Why nested loops? Each lattice point is z1*b1 + z2*b2
    points = []
    for z1 in range(-range_val, range_val + 1):
        for z2 in range(-range_val, range_val + 1):
            point = z1 * basis[:, 0] + z2 * basis[:, 1]
            points.append(point)

    points = np.array(points)

    # Plot lattice points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=20, zorder=3)

    # Plot basis vectors as arrows from origin
    origin = np.array([0, 0])
    ax.annotate('', xy=basis[:, 0], xytext=origin,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=basis[:, 1], xytext=origin,
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.annotate(f'b1={basis[:, 0]}', xy=basis[:, 0], fontsize=10, color='red')
    ax.annotate(f'b2={basis[:, 1]}', xy=basis[:, 1], fontsize=10, color='green')

    # Highlight shortest vector
    distances = np.linalg.norm(points, axis=1)
    distances[distances == 0] = np.inf  # Exclude origin
    shortest_idx = np.argmin(distances)
    ax.scatter(*points[shortest_idx], c='orange', s=100, zorder=4,
               label=f'Shortest: {points[shortest_idx]}')

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig("lattice_2d.png", dpi=100)
    plt.close()
    print(f"Lattice plotted with {len(points)} points")
    print(f"Shortest non-zero vector: {points[shortest_idx]}, "
          f"length: {distances[shortest_idx]:.4f}")


# "Good" basis — nearly orthogonal, easy to find shortest vector
good_basis = np.array([[1, 0],
                        [0, 1]])

# "Bad" basis — same lattice, but skewed, harder to work with
bad_basis = np.array([[1, 47],
                       [0, 53]])

print("=== Good Basis (Z^2 standard) ===")
plot_lattice_2d(good_basis, title="Good Basis")

print("\n=== Bad Basis (same lattice, skewed) ===")
# Why does basis quality matter? An adversary given the bad basis
# cannot easily find short vectors, while the good basis makes it trivial.
# Lattice crypto uses bad bases as public keys and good bases as private keys.
plot_lattice_2d(bad_basis, range_val=3, title="Bad Basis (same lattice)")
```

### 1.4 핵심 속성: 동일한 격자, 다른 기저

격자는 무한히 많은 기저를 가집니다. 두 기저 $\mathbf{B}_1$과 $\mathbf{B}_2$가 동일한 격자를 생성하는 것은 $\mathbf{B}_2 = \mathbf{B}_1 \mathbf{U}$ (여기서 $\mathbf{U}$는 유니모듈러 행렬(unimodular matrix), $\det(\mathbf{U}) = \pm 1$, 정수 원소)인 경우에만 해당합니다.

이는 같은 수를 다른 기저로 표현할 수 있는 것과 유사합니다 — 기저의 품질이 격자 문제 풀이에 매우 중요합니다.

---

## 2. 어려운 격자 문제

### 2.1 최단 벡터 문제(Shortest Vector Problem, SVP)

**문제**: 격자 기저 $\mathbf{B}$가 주어졌을 때, 가장 짧은 0이 아닌 격자 벡터를 찾아라.

$$
\text{Find } \mathbf{v} \in \mathcal{L}(\mathbf{B}) \setminus \{\mathbf{0}\} \text{ that minimizes } \|\mathbf{v}\|
$$

SVP는 랜덤화된 환원(randomized reduction)에서 NP-hard입니다. 임의의 격자에 대해 다항 시간 내에 이를 해결하는 알려진 알고리즘(고전 또는 양자)이 없습니다.

### 2.2 최근접 벡터 문제(Closest Vector Problem, CVP)

**문제**: 격자 기저 $\mathbf{B}$와 목표 점 $\mathbf{t}$가 주어졌을 때, $\mathbf{t}$에 가장 가까운 격자 점을 찾아라.

$$
\text{Find } \mathbf{v} \in \mathcal{L}(\mathbf{B}) \text{ that minimizes } \|\mathbf{v} - \mathbf{t}\|
$$

CVP는 SVP만큼 어렵거나 그 이상입니다 (SVP에서 CVP로의 다항 시간 환원이 존재합니다).

### 2.3 근사 버전 (GapSVP, ApproxCVP)

암호학에서는 보통 인수 $\gamma$ 이내에서 최단 벡터에 가까운 벡터를 찾아야 하는 **근사** 버전을 다룹니다:

$$
\gamma\text{-GapSVP}: \text{Find } \mathbf{v} \text{ with } \|\mathbf{v}\| \leq \gamma \cdot \lambda_1(\mathcal{L})
$$

여기서 $\lambda_1(\mathcal{L})$은 가장 짧은 0이 아닌 벡터의 길이입니다.

다항 $\gamma$에 대해서도 근사 문제는 여전히 어렵다고 여겨집니다.

### 2.4 현재까지 알려진 최선의 알고리즘

| 알고리즘 | 시간 복잡도 | 유형 |
|---------|-----------|------|
| LLL (1982) | $\text{poly}(n)$ | $2^{O(n)}$ 근사 |
| BKZ (Block Korkine-Zolotarev) | $2^{O(n)}$ | 더 나은 근사, 조절 가능한 블록 크기 |
| 체 알고리즘(Sieving) | $2^{O(n)}$ | 정확한 SVP에 대한 최선의 점근 복잡도 |
| 양자 체 알고리즘 | $2^{O(n)}$ | 상수 인수 개선에 불과 |

핵심 관찰: 양자 알고리즘조차 격자 문제에 대해 **상수 인수** 속도 향상만 달성합니다. 이는 Shor의 알고리즘이 인수분해와 이산 로그에 제공하는 **지수적** 속도 향상과 대비됩니다.

---

## 3. 오류 포함 학습(LWE)

### 3.1 LWE 문제

2005년 Oded Regev가 소개한 LWE(Learning With Errors)는 현대 격자 암호학의 기반이 되었습니다.

**설정**: 차원 $n$, 모듈러스 $q$, 오류 분포 $\chi$ (일반적으로 작은 표준편차 $\sigma$를 가진 이산 가우시안)를 선택합니다.

**문제**: 다음 형태의 많은 샘플이 주어졌을 때

$$
(\mathbf{a}_i, b_i = \langle \mathbf{a}_i, \mathbf{s} \rangle + e_i \pmod{q})
$$

여기서 $\mathbf{a}_i \in \mathbb{Z}_q^n$은 무작위, $\mathbf{s} \in \mathbb{Z}_q^n$은 비밀 벡터, $e_i \leftarrow \chi$는 작은 오류일 때, $\mathbf{s}$를 찾아라.

### 3.2 LWE가 어려운 이유

오류 $e_i$ 없이는 이것은 단순한 연립 일차 방정식이며 가우스 소거법으로 $O(n^3)$에 풀 수 있습니다. 작은 오류 항들이 시스템을 **잡음이 있게** 만들고, 잡음 있는 선형 방정식에서 $\mathbf{s}$를 복원하는 것은 계산적으로 어려운 격자 문제와 동등합니다.

**Regev의 정리**: LWE를 푸는 것은 최악의 경우 근사 격자 문제(GapSVP)를 다항 시간에 푸는 것만큼 어렵습니다.

> **비유:** 선의 기울기를 결정하려는데 모든 측정값에 작은 무작위 오류가 있다고 상상해 보세요. 정확한 측정값으로는 두 점으로 충분합니다. 잡음이 있으면 많은 측정값과 통계적 기법이 필요합니다. LWE는 이것과 같지만, $n$이 수백 또는 수천인 $n$차원에서 이루어집니다 — 잡음이 시스템을 통계적으로 더 어렵게 만드는 것이 아니라 지수적으로 더 어렵게 만듭니다.

### 3.3 LWE 암호화 방식

```python
"""
A simple LWE-based public-key encryption scheme.

This is the Regev encryption scheme — the conceptual ancestor of
all modern lattice-based crypto. It encrypts one bit at a time.

Why such small parameters here? Real implementations use n=512 or
n=1024 with q around 2^12 to 2^32. We use small values to make
the computation transparent.
"""

import numpy as np
import secrets


def lwe_keygen(n: int = 8, q: int = 97, num_samples: int = 20):
    """
    Generate LWE key pair.

    Parameters:
        n: dimension of the secret vector (security parameter)
        q: modulus (must be prime, much larger than the error)
        num_samples: number of LWE samples (rows of A)

    Returns:
        public_key: (A, b)  where b = A*s + e mod q
        secret_key: s
    """
    # Secret key: random vector in Z_q^n
    s = np.array([secrets.randbelow(q) for _ in range(n)])

    # Public matrix: random m x n matrix
    A = np.array([[secrets.randbelow(q) for _ in range(n)]
                   for _ in range(num_samples)])

    # Error vector: small Gaussian noise
    # Why small error? The error must be small enough that decryption
    # works (roughly |e| < q/4), but large enough that the LWE problem
    # is hard. This tension determines the parameters.
    sigma = 2.0
    e = np.round(np.random.normal(0, sigma, num_samples)).astype(int) % q

    # Public key: b = A*s + e mod q
    b = (A @ s + e) % q

    return (A, b), s


def lwe_encrypt(public_key, bit: int, q: int = 97):
    """
    Encrypt a single bit (0 or 1).

    Why a random subset sum? The encryption creates a new LWE sample
    by combining random rows of the public key. This "rerandomizes"
    the public key, ensuring each ciphertext looks fresh.
    """
    A, b = public_key
    m = len(b)

    # Choose a random binary vector (subset selector)
    r = np.array([secrets.randbelow(2) for _ in range(m)])

    # Ciphertext: (u, v) where
    #   u = r^T * A mod q  (n-dimensional vector)
    #   v = r^T * b + bit * floor(q/2) mod q  (scalar)
    u = (r @ A) % q
    v = (r @ b + bit * (q // 2)) % q

    return u, v


def lwe_decrypt(secret_key, ciphertext, q: int = 97) -> int:
    """
    Decrypt a ciphertext.

    Why does this work? v - u*s = r^T*(A*s + e) + bit*q/2 - (r^T*A)*s
                                = r^T*e + bit*q/2
    Since r^T*e is small (bounded by m*max_error), and q/2 is large,
    we can determine the bit by checking if the result is closer to
    0 (bit=0) or q/2 (bit=1).
    """
    u, v = ciphertext

    # Compute v - <u, s> mod q
    decrypted = (v - u @ secret_key) % q

    # Decision: closer to 0 → bit 0, closer to q/2 → bit 1
    if decrypted > q // 4 and decrypted < 3 * q // 4:
        return 1
    else:
        return 0


def demo_lwe():
    """Demonstrate LWE encryption and decryption."""
    n, q = 16, 97
    num_samples = 40

    print(f"LWE parameters: n={n}, q={q}, samples={num_samples}")

    # Key generation
    public_key, secret_key = lwe_keygen(n, q, num_samples)
    print(f"Secret key (first 5): {secret_key[:5]}")

    # Encrypt and decrypt multiple bits
    test_bits = [0, 1, 1, 0, 1, 0, 0, 1]
    decrypted_bits = []

    for bit in test_bits:
        ct = lwe_encrypt(public_key, bit, q)
        dec = lwe_decrypt(secret_key, ct, q)
        decrypted_bits.append(dec)

    print(f"Original:  {test_bits}")
    print(f"Decrypted: {decrypted_bits}")
    print(f"All correct: {test_bits == decrypted_bits}")


if __name__ == "__main__":
    demo_lwe()
```

### 3.4 복호화 정확성 분석

복호화 잡음은 $r^T \mathbf{e}$이며, 여기서 $r \in \{0,1\}^m$이고 $\mathbf{e}$의 원소들은 $B$로 유계됩니다. 잡음의 크기는 다음으로 유계됩니다:

$$
|r^T \mathbf{e}| \leq m \cdot B
$$

올바른 복호화를 위해 $m \cdot B < q/4$가 필요합니다. 이는 보안 매개변수(큰 $m$과 $B$ 필요)와 정확성(작은 $m \cdot B$가 $q$에 비해 필요) 사이의 관계를 제약합니다.

---

## 4. Ring-LWE와 Module-LWE

### 4.1 LWE의 효율성 문제

표준 LWE는 공개 행렬 $\mathbf{A}$가 $\mathbb{Z}_q$에서 $m \times n$ 개의 무작위 원소이기 때문에 키 크기가 큽니다. 128비트 보안을 위해:
- $n \approx 512$, $m \approx 1024$
- 공개 키 $\approx$ 500 KB (비실용적으로 큼)

### 4.2 Ring-LWE

**Ring-LWE**는 벡터와 행렬을 다음 링의 다항식으로 대체합니다:

$$
R_q = \mathbb{Z}_q[x] / (x^n + 1)
$$

여기서 $n$은 2의 거듭제곱입니다. 이 링에서의 곱셈은 수론 변환(Number Theoretic Transform, NTT)을 사용하여 $O(n \log n)$에 계산할 수 있습니다. NTT는 FFT의 모듈러 유사체입니다.

**키 크기 감소**: 단일 링 원소(차수 $n-1$의 다항식)가 전체 $n \times n$ 행렬을 대체하여 키 크기를 $n$ 인수만큼 줄입니다.

### 4.3 Module-LWE

Module-LWE는 LWE와 Ring-LWE의 중간으로, 링 원소들의 작은 행렬을 다룹니다. $k \times k$ 링 원소 행렬에서:

- $k = 1$: Ring-LWE
- $k = n$: 표준 LWE (다항식 구조 포함)
- $k = 2, 3, 4$: Module-LWE (Kyber/Dilithium에 사용)

**Module-LWE의 이유는?** Ring-LWE의 완전한 대수적 구조에 의존하지 않고 유연한 보안/성능 트레이드오프를 제공합니다.

### 4.4 매개변수 비교

| 방식 | 키 크기 | 암호문 크기 | 보안 가정 |
|------|--------|-----------|---------|
| LWE ($n=512$) | ~500 KB | ~500 KB | LWE (가장 강력) |
| Ring-LWE ($n=1024$) | ~1 KB | ~1 KB | Ring-LWE (더 구조화됨) |
| Module-LWE ($k=3, n=256$) | ~1.5 KB | ~1.5 KB | Module-LWE (균형) |
| RSA-2048 | 256 B | 256 B | 인수분해 |
| ECC P-256 | 64 B | 64 B | ECDLP |

---

## 5. NTRU: 역사적 관점

### 5.1 최초의 실용적 격자 방식

1996년 Hoffstein, Pipher, Silverman이 소개한 NTRU는 격자 문제에 기반한 최초의 실용적 공개 키 암호 시스템으로, LWE보다 거의 10년 앞섰습니다.

### 5.2 NTRU 동작 방식 (단순화)

NTRU는 두 모듈러스 $p$와 $q$를 가진 다항식 링 $\mathbb{Z}[x]/(x^N - 1)$에서 동작합니다:

1. **키 생성**: 작은 계수를 가진 무작위 다항식 $f$와 $g$를 선택합니다. 공개 키는 $h = p \cdot g \cdot f^{-1} \pmod{q}$입니다.
2. **암호화**: 메시지 다항식 $m$과 무작위 블라인딩 다항식 $r$에 대해: $c = r \cdot h + m \pmod{q}$.
3. **복호화**: $f$를 곱하고 $p$로 축소합니다.

### 5.3 NTRU의 유산

NTRU는 이후 모든 격자 방식에 영향을 미쳤지만 NIST 표준화에는 선정되지 않았습니다:
- 특허 이력 (현재 만료됨)
- 동등한 보안에서 Kyber보다 약간 비효율적
- (Module-)LWE에 비해 덜 이해된 보안 환원

NTRU는 NTRUEncrypt와 NTRU-HRSS로 NIST PQC 경쟁에 제출되어 최종 라운드에 진출했습니다.

---

## 6. CRYSTALS-Kyber (ML-KEM)

### 6.1 NIST의 선택

2024년 NIST는 CRYSTALS-Kyber를 FIPS 203에서 **ML-KEM**(Module-Lattice-based Key Encapsulation Mechanism)으로 표준화했습니다. 이는 주요 포스트양자 키 교환 메커니즘으로, TLS, SSH, VPN 및 사실상 모든 인터넷 프로토콜에 사용될 예정입니다.

### 6.2 KEM vs. 키 교환

KEM(Key Encapsulation Mechanism, 키 캡슐화 메커니즘)은 키 교환과 약간 다릅니다:

| | 키 교환 (DH) | KEM (Kyber) |
|---|---|---|
| **상호작용** | 양측 모두 무작위성 기여 | 한 측이 무작위 키를 캡슐화 |
| **출력** | 양측 입력에서 도출된 공유 비밀 | 캡슐화자가 생성한 공유 비밀 |
| **왕복 횟수** | 1회 왕복 | 1개 메시지(캡슐화) + 1개 응답(수락) |

### 6.3 Kyber 개요

**매개변수** (Kyber-768, NIST 보안 레벨 3):
- 모듈 차원: $k = 3$
- 다항식 차수: $n = 256$
- 모듈러스: $q = 3329$
- 오류 분포: $\eta = 2$인 중심 이항 분포

**키 생성**:
1. 무작위 행렬 $\mathbf{A} \in R_q^{k \times k}$ 샘플링
2. 비밀 $\mathbf{s} \in R_q^k$ 및 오류 $\mathbf{e} \in R_q^k$ (작음) 샘플링
3. 공개 키: $(\mathbf{A}, \mathbf{t} = \mathbf{A}\mathbf{s} + \mathbf{e})$
4. 비밀 키: $\mathbf{s}$

**캡슐화** (앨리스 → 밥):
1. 무작위 $\mathbf{r}, \mathbf{e}_1, e_2$ (작음) 샘플링
2. $\mathbf{u} = \mathbf{A}^T\mathbf{r} + \mathbf{e}_1$ 계산
3. $v = \mathbf{t}^T\mathbf{r} + e_2 + \lceil q/2 \rceil \cdot m$ 계산
4. 암호문: $(\mathbf{u}, v)$

**역캡슐화** (밥):
1. $v - \mathbf{s}^T\mathbf{u}$ 계산
2. 반올림하여 $m$ 복원

### 6.4 Kyber 매개변수 집합

| 매개변수 집합 | NIST 레벨 | $k$ | 공개 키 | 암호문 | 공유 비밀 |
|------------|---------|-----|--------|--------|---------|
| Kyber-512 (ML-KEM-512) | 1 (AES-128) | 2 | 800 B | 768 B | 32 B |
| Kyber-768 (ML-KEM-768) | 3 (AES-192) | 3 | 1,184 B | 1,088 B | 32 B |
| Kyber-1024 (ML-KEM-1024) | 5 (AES-256) | 4 | 1,568 B | 1,568 B | 32 B |

### 6.5 실제 Kyber 배포

Kyber-768은 이미 배포되고 있습니다:
- **Google Chrome** (2023년 8월): TLS에서 하이브리드 X25519+Kyber-768
- **Signal** (2023년 9월): PQXDH (Kyber를 사용한 포스트양자 X3DH)
- **Apple iMessage** (2024년 2월): Kyber-768을 사용한 PQ3 프로토콜
- **AWS KMS** (2024년): Kyber를 사용한 하이브리드 TLS

---

## 7. CRYSTALS-Dilithium (ML-DSA)

### 7.1 격자 기반 디지털 서명

CRYSTALS-Dilithium은 FIPS 204에서 **ML-DSA**(Module-Lattice-based Digital Signature Algorithm)로 표준화되었습니다. ECDSA와 EdDSA를 디지털 서명에서 대체(또는 보완)합니다.

### 7.2 중단을 포함한 Fiat-Shamir(Fiat-Shamir with Aborts)

Dilithium은 **중단을 포함한 Fiat-Shamir** 기법을 사용합니다:

1. 서명자가 무작위 마스킹 벡터 $\mathbf{y}$ 생성
2. 커밋먼트 $\mathbf{w} = \mathbf{A}\mathbf{y}$ 계산
3. 챌린지 $c = H(\mathbf{w}, \text{message})$ 도출
4. 응답 $\mathbf{z} = \mathbf{y} + c\mathbf{s}$ 계산
5. **중단 확인**: $\mathbf{z}$가 너무 크면 ($\mathbf{s}$에 대한 정보를 유출할 수 있으면) 폐기하고 재시작

중단 단계는 필수적입니다 — 없으면 공격자가 많은 서명에서 통계적으로 비밀 키를 복원할 수 있습니다.

### 7.3 Dilithium 크기

| 매개변수 집합 | NIST 레벨 | 공개 키 | 서명 | 서명 시간 |
|------------|---------|--------|------|---------|
| Dilithium2 (ML-DSA-44) | 2 | 1,312 B | 2,420 B | ~0.5 ms |
| Dilithium3 (ML-DSA-65) | 3 | 1,952 B | 3,293 B | ~0.8 ms |
| Dilithium5 (ML-DSA-87) | 5 | 2,592 B | 4,595 B | ~1.0 ms |

ECDSA P-256(64B 서명)과 비교하면 Dilithium 서명은 ~40~70배 더 큽니다. 이는 제약된 환경에서 중요한 고려사항입니다.

---

## 8. 고전 암호학과의 비교

### 8.1 크기 비교

```python
"""
Compare key and ciphertext/signature sizes across crypto families.

Why size matters: Larger keys and ciphertexts increase bandwidth,
storage, and latency. This is especially critical for:
- TLS handshakes (adding ~1-2 KB)
- IoT devices (limited bandwidth)
- Certificate chains (each cert grows)
- Blockchain (every node stores every signature)
"""


def compare_sizes():
    """Compare sizes of different cryptographic schemes."""

    schemes = {
        "RSA-2048": {
            "type": "KEM + Signature",
            "public_key": 256,
            "private_key": 256,
            "ciphertext_or_sig": 256,
            "security_level": "112-bit",
            "quantum_safe": False,
        },
        "RSA-3072": {
            "type": "KEM + Signature",
            "public_key": 384,
            "private_key": 384,
            "ciphertext_or_sig": 384,
            "security_level": "128-bit",
            "quantum_safe": False,
        },
        "ECDSA P-256": {
            "type": "Signature",
            "public_key": 64,
            "private_key": 32,
            "ciphertext_or_sig": 64,
            "security_level": "128-bit",
            "quantum_safe": False,
        },
        "X25519 + Ed25519": {
            "type": "KEM + Signature",
            "public_key": 32,
            "private_key": 32,
            "ciphertext_or_sig": 64,
            "security_level": "128-bit",
            "quantum_safe": False,
        },
        "Kyber-768 (ML-KEM)": {
            "type": "KEM",
            "public_key": 1184,
            "private_key": 2400,
            "ciphertext_or_sig": 1088,
            "security_level": "192-bit (PQ)",
            "quantum_safe": True,
        },
        "Dilithium3 (ML-DSA)": {
            "type": "Signature",
            "public_key": 1952,
            "private_key": 4000,
            "ciphertext_or_sig": 3293,
            "security_level": "192-bit (PQ)",
            "quantum_safe": True,
        },
    }

    print(f"{'Scheme':<25} {'Type':<18} {'PubKey':>8} {'CT/Sig':>8} "
          f"{'Security':<14} {'PQ?'}")
    print("-" * 90)

    for name, data in schemes.items():
        pq = "Yes" if data["quantum_safe"] else "No"
        print(f"{name:<25} {data['type']:<18} "
              f"{data['public_key']:>6} B {data['ciphertext_or_sig']:>6} B "
              f"{data['security_level']:<14} {pq}")


if __name__ == "__main__":
    compare_sizes()
```

### 8.2 성능 비교

| 연산 | RSA-2048 | ECDSA P-256 | Kyber-768 | Dilithium3 |
|------|---------|------------|---------|----------|
| 키 생성 | ~1 ms | ~0.1 ms | ~0.05 ms | ~0.3 ms |
| 암호화/캡슐화 | ~0.1 ms | ~0.2 ms | ~0.07 ms | N/A |
| 복호화/역캡슐화 | ~1 ms | ~0.2 ms | ~0.08 ms | N/A |
| 서명 | N/A | ~0.2 ms | N/A | ~0.8 ms |
| 검증 | N/A | ~0.3 ms | N/A | ~0.3 ms |

격자 방식은 놀랍도록 **빠릅니다** — 종종 RSA보다 빠릅니다. 트레이드오프는 속도가 아닌 **크기**에 있습니다.

---

## 9. 격자 문제가 양자 공격에 저항하는 이유

### 9.1 Shor의 알고리즘과 그 한계

Shor의 알고리즘은 모듈러 지수 연산의 **주기 구조**를 이용하여 인수분해와 이산 로그를 효율적으로 찾습니다. 구체적으로, 양자 푸리에 변환(Quantum Fourier Transform)을 사용하여 $f(x) = a^x \bmod N$의 주기를 찾습니다.

격자 문제에는 이러한 주기 구조가 없습니다. SVP나 LWE를 Shor의 알고리즘이 활용할 수 있는 숨겨진 부분군 문제(hidden subgroup problem)로 공식화하는 방법이 알려져 있지 않습니다.

### 9.2 Grover의 알고리즘

Grover의 알고리즘은 비구조적 탐색에 대해 일반적인 $O(\sqrt{N})$ 속도 향상을 제공합니다. 격자 문제에 적용하면:
- 차원 $n$에서 브루트포스 SVP: 고전 $2^{O(n)}$, 양자 $2^{O(n/2)}$
- 이는 지수에서의 상수 인수이지 질적 개선이 아닙니다

Grover에 대비한 보안을 유지하려면 차원을 ~2배 늘리면 됩니다 (대칭 암호의 키 크기를 2배로 늘리는 것과 동등).

### 9.3 양자 격자 공격의 현황

격자 문제에 대한 최선의 알려진 양자 알고리즘(양자 체 알고리즘, 양자 BKZ)은 고전 알고리즘 대비 기껏해야 작은 다항 속도 향상을 달성합니다. 지수적 양자 속도 향상은 알려져 있지 않으며, 격자 문제의 구조에 기반한 이론적 주장에 의해 존재하지 않을 수도 있습니다.

---

## 10. 요약

| 개념 | 핵심 내용 |
|------|---------|
| 격자 | $\mathbb{R}^n$에서 점들의 규칙적인 격자; 고차원에서 어려운 문제 발생 |
| SVP/CVP | 최단/최근접 벡터 찾기 — 고전 및 양자 모두에서 어렵다고 여겨짐 |
| LWE | 잡음 있는 선형 방정식; 최악의 경우 격자 문제로 환원 |
| Ring/Module-LWE | 실용적 키 크기를 가능하게 하는 구조화된 변형 |
| Kyber (ML-KEM) | NIST의 표준 PQ 키 캡슐화; ~1.2 KB 키 |
| Dilithium (ML-DSA) | NIST의 표준 PQ 서명; ~2~3 KB 서명 |
| 양자 저항 | 격자 문제에 대한 알려진 지수적 양자 속도 향상 없음 |

---

## 11. 연습 문제

### 연습 1: 격자 기초 (개념)

기저 $\mathbf{B} = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$가 주어졌을 때:

1. $|x| \leq 10$이고 $|y| \leq 10$인 모든 격자 점 $(x, y)$를 나열하세요
2. 육안으로 가장 짧은 0이 아닌 벡터를 찾으세요
3. 2D의 경우 LLL 알고리즘을 손으로 적용하여 동일한 격자에 대해 "더 직교에 가까운" 대안적 기저를 찾으세요

### 연습 2: LWE 암호화 (코딩)

LWE 암호화 데모를 다중 비트 메시지를 암호화하도록 확장하세요:
1. 한 번에 8비트씩 암호화하는 바이트 단위 암호화 방식 구현
2. 오류 수정 추가: 오류 표준편차를 늘리면 어떻게 되나요? 어느 시점에서 복호화가 실패하기 시작하나요?
3. 고정된 $n$과 $q$에서 오류 표준편차의 함수로 복호화 실패율을 그래프로 그리세요

### 연습 3: 링 연산 (코딩)

$\mathbb{Z}_q[x]/(x^n + 1)$에서 다항식 산술을 구현하세요:
1. $x^n + 1$ 및 $q$를 법으로 하는 다항식의 덧셈, 뺄셈, 곱셈
2. 교과서식 곱셈($O(n^2)$)과 NTT 기반 곱셈($O(n \log n)$)을 구현하고 비교
3. $n = 256, 512, 1024$에 대해 두 방식 모두 벤치마크

### 연습 4: 매개변수 선택 (개념 + 코딩)

LWE 기반 암호화 방식에 대해:
1. 보안 레벨을 128비트로 고정 (즉, 콘크리트 보안 추정에 의해 $n \cdot \log_2 q \cdot \sigma \geq 128$)
2. 오류 확률 $< 2^{-40}$으로 올바른 복호화를 허용하는 최소 $n$과 $q$를 찾으세요
3. 결과 키 크기를 RSA-3072 및 Kyber-768과 비교하세요

### 연습 5: 하이브리드 키 교환 (심화)

X25519와 단순화된 Kyber 유사 방식을 결합하는 하이브리드 키 교환을 구현하세요:
1. X25519 ECDH와 LWE 기반 키 교환을 모두 수행
2. HKDF(레슨 8)를 사용하여 두 공유 비밀을 결합
3. 포스트양자 전환 기간에 하이브리드 방식이 권장되는 이유 설명: 두 방식 중 하나가 깨진다면 어떻게 되나요?

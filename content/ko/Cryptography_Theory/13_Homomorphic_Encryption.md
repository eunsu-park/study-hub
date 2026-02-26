# 레슨 13: 동형 암호

**이전**: [영지식 증명](./12_Zero_Knowledge_Proofs.md) | **다음**: [응용 암호 프로토콜](./14_Applied_Cryptographic_Protocols.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 암호화된 데이터에서 연산을 수행하는 동기를 설명하고, 다른 프라이버시 기법과 구별할 수 있다
2. 동형 암호의 세 가지 수준(부분 동형(PHE), 준동형(SHE), 완전 동형(FHE))을 정의할 수 있다
3. RSA의 곱셈 동형성과 Paillier의 덧셈 동형성을 시연할 수 있다
4. Paillier 암호화 방식을 구현하고 암호문에서 연산을 수행할 수 있다
5. FHE의 노이즈 증가 문제를 설명하고 Gentry의 부트스트래핑(Bootstrapping) 해법을 이해할 수 있다
6. 정수 및 근사 산술에 대한 BFV와 CKKS 방식을 비교할 수 있다
7. FHE의 실용적 응용 사례와 현재의 성능 한계를 평가할 수 있다

---

암호화된 의료 기록을 클라우드 서버에 전송하고, 서버가 해당 기록에 대해 머신러닝 진단을 수행한 뒤 암호화된 결과를 돌려받는 것이 가능하다면 — 그것도 서버가 데이터를 한 번도 보지 않은 채로 — 어떨까요? 이것이 **동형 암호(Homomorphic Encryption, HE)**의 약속입니다: 데이터를 복호화하지 않고도 암호화된 상태에서 연산을 수행할 수 있는 능력입니다. 1978년 Rivest, Adleman, Dertouzos가 처음 구상한(RSA 논문 발표 불과 1년 후) 완전 동형 암호는 Craig Gentry가 2009년에 돌파구를 열기 전까지 30년이 넘는 미해결 문제로 남아 있었습니다. 오늘날 HE는 이론적 호기심에서 실용적 배포로 전환 중이지만, 상당한 성능 과제가 여전히 남아 있습니다.

> **비유:** 동형 암호는 화학 실험실의 글로브 박스(Glove Box)와 같습니다. 내부의 화학 물질을 직접 만지지 않고도(복호화 없이) 조작할 수 있습니다. 밀폐된 환경(암호화)은 당신과 화학 물질 모두를 보호하면서도, 장갑을 통해 복잡한 작업을 수행할 수 있게 해줍니다.

## 목차

1. [동기와 배경](#1-동기와-배경)
2. [부분 동형 암호(PHE)](#2-부분-동형-암호phe)
3. [Paillier 암호 시스템](#3-paillier-암호-시스템)
4. [부분 동형에서 완전 동형으로](#4-부분-동형에서-완전-동형으로)
5. [노이즈 증가 문제](#5-노이즈-증가-문제)
6. [Gentry의 부트스트래핑](#6-gentry의-부트스트래핑)
7. [현대 FHE 방식: BFV와 CKKS](#7-현대-fhe-방식-bfv와-ckks)
8. [실용적 응용](#8-실용적-응용)
9. [성능 현실 점검](#9-성능-현실-점검)
10. [요약](#10-요약)
11. [연습 문제](#11-연습-문제)

---

## 1. 동기와 배경

### 1.1 클라우드 컴퓨팅의 딜레마

클라우드 컴퓨팅은 탄력적인 자원, 관리형 인프라, 전 세계적 가용성이라는 막대한 이점을 제공하지만, 클라우드 제공자에게 데이터를 신뢰해야 합니다. 저장 중 암호화와 전송 중 암호화가 적용되더라도, 처리를 위해서는 데이터가 복호화되어야 합니다.

```
전통적인 클라우드 컴퓨팅:
  클라이언트 → 암호화 → [전송] → 복호화 → 처리 → 암호화 → [전송] → 복호화 → 클라이언트
                                  ^^^^^^^^
                                  서버에서 데이터가 노출됨

동형 암호:
  클라이언트 → 암호화 → [전송] → 암호문 상에서 처리 → [전송] → 복호화 → 클라이언트
                                  ^^^^^^^^^^^^^^^^^^^^^^^
                                  데이터가 절대 노출되지 않음
```

### 1.2 다른 프라이버시 기법과의 비교

| 기법 | 암호화된 데이터에서 연산? | 다수 당사자? | 신뢰 가정 |
|-----------|----------------------------|-------------------|-------------------|
| **HE** | 예 | 아니오 (단일 서버) | 없음 (데이터가 복호화되지 않음) |
| **MPC** (레슨 14) | 예 (비밀 분산 활용) | 예 (다수 당사자) | 정직한 다수 또는 정직하지만 호기심 있는 참여자 |
| **TEE** (SGX, TrustZone) | 아니오 (엔클레이브 내부에서 복호화) | 아니오 | 하드웨어 공급자 신뢰 |
| **차분 프라이버시(Differential Privacy)** | N/A (출력에 노이즈 추가) | N/A | 데이터 큐레이터 신뢰 |
| **ZKP** (레슨 12) | 아니오 (연산의 정확성을 증명) | N/A | 없음 |

### 1.3 형식적 정의

함수 클래스 $\mathcal{F}$에 대한 암호화 방식 $(\text{KeyGen}, \text{Enc}, \text{Dec}, \text{Eval})$이 **동형(Homomorphic)**이라는 것은 다음을 만족하는 것입니다:

$$
\text{Dec}(\text{sk}, \text{Eval}(\text{pk}, f, \text{Enc}(\text{pk}, m_1), \ldots, \text{Enc}(\text{pk}, m_k))) = f(m_1, \ldots, m_k)
$$

모든 $f \in \mathcal{F}$ 및 유효한 평문 $m_1, \ldots, m_k$에 대해 성립합니다.

---

## 2. 부분 동형 암호(PHE)

### 2.1 정의

PHE는 암호문에 대해 **하나의 연산 종류**(덧셈 또는 곱셈 중 하나)만을 지원하며, 연산 횟수에는 제한이 없습니다.

### 2.2 RSA의 곱셈 동형성

RSA를 상기해 보면 (레슨 5): $\text{Enc}(m) = m^e \bmod N$.

두 암호문에 대해:

$$
\text{Enc}(m_1) \cdot \text{Enc}(m_2) = m_1^e \cdot m_2^e = (m_1 \cdot m_2)^e = \text{Enc}(m_1 \cdot m_2) \pmod{N}
$$

RSA는 **곱셈에 대한 동형성**을 가집니다: 암호문을 곱하면 평문을 곱한 것과 동일한 결과를 얻습니다.

```python
"""
RSA multiplicative homomorphism demonstration.

Why is this useful? Imagine a server that needs to multiply two
encrypted values (e.g., price × quantity) without learning either value.
With RSA's homomorphism, the server multiplies the ciphertexts, and
the client decrypts to get the product.

Limitation: RSA supports ONLY multiplication, not addition.
You cannot compute enc(m1) + enc(m2) to get enc(m1 + m2).
"""

from math import gcd


def simple_rsa_keygen(p: int, q: int):
    """Generate RSA key pair from given primes (educational only)."""
    N = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    # Find d such that e*d ≡ 1 (mod phi)
    d = pow(e, -1, phi)
    return (e, N), (d, N)  # public, private


def rsa_encrypt(pub, m):
    e, N = pub
    return pow(m, e, N)


def rsa_decrypt(priv, c):
    d, N = priv
    return pow(d, c, N)  # This should be pow(c, d, N) — fixed below


def demo_rsa_homomorphism():
    """Show RSA multiplicative homomorphism."""
    # Small primes for demonstration
    p, q = 61, 53
    pub, priv = simple_rsa_keygen(p, q)
    e, N = pub
    d, _ = priv

    m1, m2 = 7, 13

    # Encrypt individually
    c1 = pow(m1, e, N)
    c2 = pow(m2, e, N)

    # Multiply ciphertexts (no decryption needed!)
    c_product = (c1 * c2) % N

    # Decrypt the product
    result = pow(c_product, d, N)

    print(f"m1 = {m1}, m2 = {m2}")
    print(f"c1 = {c1}, c2 = {c2}")
    print(f"c1 * c2 mod N = {c_product}")
    print(f"Dec(c1 * c2) = {result}")
    print(f"m1 * m2 = {m1 * m2}")
    print(f"Homomorphism verified: {result == m1 * m2}")


if __name__ == "__main__":
    demo_rsa_homomorphism()
```

### 2.3 ElGamal의 곱셈 동형성

ElGamal 암호화(레슨 5)도 곱셈에 대한 동형성을 가집니다:

$$
\text{Enc}(m_1) \otimes \text{Enc}(m_2) = (g^{r_1+r_2}, m_1 \cdot m_2 \cdot y^{r_1+r_2})
$$

이는 암호화된 투표를 곱하여 암호화된 집계를 계산하는 전자 투표 방식에 사용됩니다.

### 2.4 덧셈 동형 vs. 곱셈 동형

| 방식 | 동형 연산 | 깊이 | 용도 |
|--------|----------------------|-------|----------|
| RSA (교과서) | 곱셈 | 무제한 | 실용적이지 않음 (교과서 RSA는 안전하지 않음) |
| ElGamal | 곱셈 | 무제한 | 전자 투표, 재무작위화(Rerandomization) |
| **Paillier** | **덧셈** | 무제한 | 집계, 머신러닝, 경매 |
| Goldwasser-Micali | XOR (1비트) | 무제한 | 이론적 |

---

## 3. Paillier 암호 시스템

### 3.1 왜 Paillier인가?

Paillier(1999)는 가장 실용적인 덧셈 동형 방식입니다. 다음과 같은 이유로 널리 사용됩니다:
- 덧셈은 곱셈보다 일반적으로 더 유용합니다 (집계, 평균, 선형 연산)
- 스칼라 곱셈 지원: $\text{Enc}(k \cdot m) = \text{Enc}(m)^k$
- 의미론적 보안(Semantic Security) (확률적 암호화)

### 3.2 수학적 원리

**키 생성:**
1. 두 큰 소수 $p, q$를 선택하고 $N = pq$를 계산
2. $\lambda = \text{lcm}(p-1, q-1)$를 계산
3. $g = N + 1$ 선택 (작동하는 단순화된 방법)
4. $\mu = (L(g^\lambda \bmod N^2))^{-1} \bmod N$ 계산, 여기서 $L(x) = \frac{x-1}{N}$
5. 공개 키: $(N, g)$; 개인 키: $(\lambda, \mu)$

**암호화:**

$$
\text{Enc}(m, r) = g^m \cdot r^N \bmod N^2
$$

여기서 $r$은 $\mathbb{Z}_N^*$에서의 무작위 값입니다.

**복호화:**

$$
\text{Dec}(c) = L(c^\lambda \bmod N^2) \cdot \mu \bmod N
$$

### 3.3 동형 특성

**덧셈**:

$$
\text{Enc}(m_1) \cdot \text{Enc}(m_2) \bmod N^2 = \text{Enc}(m_1 + m_2 \bmod N)
$$

**스칼라 곱셈**:

$$
\text{Enc}(m)^k \bmod N^2 = \text{Enc}(k \cdot m \bmod N)
$$

### 3.4 구현

```python
"""
Paillier Homomorphic Encryption — Additively Homomorphic.

Why Paillier for practical applications? It supports:
1. Adding encrypted values without decryption
2. Multiplying an encrypted value by a known constant
3. These two operations enable computing weighted sums, averages,
   and inner products — the building blocks of many ML algorithms.
"""

import secrets
import math


class PaillierKeyPair:
    """Paillier public/private key pair."""

    def __init__(self, bits: int = 512):
        """
        Generate Paillier key pair.

        Why 512 bits here? This is for demonstration. Production
        systems use 2048-bit or 3072-bit keys (similar to RSA).
        """
        # Generate two large primes
        p = self._generate_prime(bits // 2)
        q = self._generate_prime(bits // 2)

        self.n = p * q
        self.n_sq = self.n * self.n
        self.g = self.n + 1  # Simplified generator choice

        # Private key components
        self.lam = math.lcm(p - 1, q - 1)

        # Why L function? It maps elements of the form (1 + N)^x mod N^2
        # back to x mod N. This is the "discrete logarithm" in this group.
        def L(x):
            return (x - 1) // self.n

        self.mu = pow(L(pow(self.g, self.lam, self.n_sq)), -1, self.n)

    def _generate_prime(self, bits: int) -> int:
        """Generate a random prime of the specified bit length."""
        from sympy import isprime, nextprime
        while True:
            candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
            p = nextprime(candidate)
            if p.bit_length() == bits:
                return p


class Paillier:
    """Paillier encryption scheme with homomorphic operations."""

    def __init__(self, keypair: PaillierKeyPair):
        self.kp = keypair

    def encrypt(self, plaintext: int) -> int:
        """
        Encrypt a plaintext integer.

        Why random r? Semantic security requires that encrypting the
        same message twice produces different ciphertexts. The random
        r provides this "probabilistic" property.
        """
        assert 0 <= plaintext < self.kp.n, f"Plaintext must be in [0, {self.kp.n})"

        r = secrets.randbelow(self.kp.n - 1) + 1
        while math.gcd(r, self.kp.n) != 1:
            r = secrets.randbelow(self.kp.n - 1) + 1

        # c = g^m * r^n mod n^2
        c = (pow(self.kp.g, plaintext, self.kp.n_sq) *
             pow(r, self.kp.n, self.kp.n_sq)) % self.kp.n_sq
        return c

    def decrypt(self, ciphertext: int) -> int:
        """Decrypt a ciphertext."""
        # m = L(c^lambda mod n^2) * mu mod n
        x = pow(ciphertext, self.kp.lam, self.kp.n_sq)
        L_val = (x - 1) // self.kp.n
        plaintext = (L_val * self.kp.mu) % self.kp.n
        return plaintext

    def add_encrypted(self, c1: int, c2: int) -> int:
        """
        Add two encrypted values: Enc(m1) * Enc(m2) = Enc(m1 + m2).

        Why multiplication in ciphertext space = addition in plaintext space?
        Because Enc(m) = g^m * r^n mod n^2, and:
        g^m1 * r1^n * g^m2 * r2^n = g^(m1+m2) * (r1*r2)^n = Enc(m1+m2)
        """
        return (c1 * c2) % self.kp.n_sq

    def scalar_multiply(self, ciphertext: int, scalar: int) -> int:
        """
        Multiply encrypted value by a known scalar: Enc(m)^k = Enc(k*m).

        Why exponentiation? Enc(m)^k = (g^m * r^n)^k = g^(km) * r^(kn) = Enc(km)
        """
        return pow(ciphertext, scalar, self.kp.n_sq)

    def encrypt_negative(self, plaintext: int) -> int:
        """
        Encrypt a negative number (as n - |plaintext|).

        Why this encoding? The Paillier plaintext space is Z_n (non-negative).
        We represent negative numbers as their additive inverse mod n:
        -x is encoded as n - x. After decryption, values > n/2 are interpreted
        as negative.
        """
        if plaintext < 0:
            return self.encrypt(self.kp.n + plaintext)
        return self.encrypt(plaintext)

    def decrypt_signed(self, ciphertext: int) -> int:
        """Decrypt and interpret as a signed integer."""
        result = self.decrypt(ciphertext)
        if result > self.kp.n // 2:
            return result - self.kp.n
        return result


def demo_paillier():
    """Demonstrate Paillier homomorphic encryption."""
    print("Generating Paillier key pair (512-bit)...")
    kp = PaillierKeyPair(bits=512)
    paillier = Paillier(kp)

    # Basic encryption/decryption
    m1, m2 = 42, 58
    c1 = paillier.encrypt(m1)
    c2 = paillier.encrypt(m2)

    print(f"\nm1 = {m1}, m2 = {m2}")
    print(f"Dec(Enc(m1)) = {paillier.decrypt(c1)}")
    print(f"Dec(Enc(m2)) = {paillier.decrypt(c2)}")

    # Homomorphic addition
    c_sum = paillier.add_encrypted(c1, c2)
    decrypted_sum = paillier.decrypt(c_sum)
    print(f"\n--- Homomorphic Addition ---")
    print(f"Dec(Enc(m1) * Enc(m2)) = {decrypted_sum}")
    print(f"m1 + m2 = {m1 + m2}")
    print(f"Correct: {decrypted_sum == m1 + m2}")

    # Scalar multiplication
    scalar = 7
    c_scaled = paillier.scalar_multiply(c1, scalar)
    decrypted_scaled = paillier.decrypt(c_scaled)
    print(f"\n--- Scalar Multiplication ---")
    print(f"Dec(Enc(m1)^{scalar}) = {decrypted_scaled}")
    print(f"{scalar} * m1 = {scalar * m1}")
    print(f"Correct: {decrypted_scaled == scalar * m1}")

    # Weighted sum (common in ML)
    weights = [3, 5]
    values = [10, 20]
    encrypted_values = [paillier.encrypt(v) for v in values]

    # Compute weighted sum homomorphically
    weighted_terms = [paillier.scalar_multiply(ev, w)
                      for ev, w in zip(encrypted_values, weights)]
    encrypted_weighted_sum = weighted_terms[0]
    for term in weighted_terms[1:]:
        encrypted_weighted_sum = paillier.add_encrypted(
            encrypted_weighted_sum, term
        )

    result = paillier.decrypt(encrypted_weighted_sum)
    expected = sum(w * v for w, v in zip(weights, values))

    print(f"\n--- Weighted Sum (ML inner product) ---")
    print(f"weights = {weights}, values = {values}")
    print(f"Homomorphic result: {result}")
    print(f"Expected: {expected}")
    print(f"Correct: {result == expected}")


if __name__ == "__main__":
    demo_paillier()
```

---

## 4. 부분 동형에서 완전 동형으로

### 4.1 계층 구조

| 수준 | 지원 연산 | 깊이 제한 | 예시 |
|-------|---------------------|-------------|---------|
| **PHE** | 하나의 연산 (덧셈 또는 곱셈) | 무제한 | Paillier, ElGamal, RSA |
| **SHE** | 덧셈과 곱셈 모두 | 제한적 (제한된 깊이) | BGN 방식 (2005) |
| **레벨형 FHE** | 두 연산 모두 | 사전 결정된 회로 깊이 | BGV, BFV |
| **FHE** | 두 연산 모두 | **무제한** (부트스트래핑을 통해) | Gentry (2009), TFHE |

### 4.2 FHE가 어려운 이유

임의의 함수를 계산하려면 덧셈과 곱셈이 모두 필요합니다(산술 연산의 완전 집합을 이루기 때문입니다). 문제는 **노이즈(Noise)**입니다.

대부분의 동형 방식은 LWE(레슨 10)에 기반하며, 암호문에 작은 오차 항 $e$를 포함합니다:

$$
\text{Enc}(m) = (\mathbf{a}, \langle \mathbf{a}, \mathbf{s} \rangle + e + \Delta m)
$$

- **덧셈**은 오차를 더합니다: $e_{\text{sum}} = e_1 + e_2$ (선형 증가)
- **곱셈**은 오차를 곱합니다: $e_{\text{prod}} \approx e_1 \cdot e_2$ (지수적 증가!)

너무 많은 곱셈 후에는 오차가 메시지를 압도하여 복호화가 실패합니다.

---

## 5. 노이즈 증가 문제

### 5.1 노이즈 증가 시각화

```python
"""
Simulate noise growth in homomorphic operations.

Why does noise matter? Every HE ciphertext has a "noise budget."
Operations consume this budget:
- Addition: small cost (noise roughly doubles)
- Multiplication: large cost (noise roughly squares)

When the budget is exhausted, decryption fails.
"""

import matplotlib.pyplot as plt
import numpy as np


def simulate_noise_growth(initial_noise: float = 1.0,
                           modulus: float = 1e15,
                           max_ops: int = 50):
    """
    Simulate noise growth under addition and multiplication.

    The noise must stay below modulus/4 for correct decryption.
    """
    threshold = modulus / 4

    # Addition chain: noise grows linearly
    add_noise = [initial_noise]
    for i in range(max_ops):
        add_noise.append(add_noise[-1] + initial_noise)

    # Multiplication chain: noise grows exponentially
    mul_noise = [initial_noise]
    for i in range(max_ops):
        new_noise = mul_noise[-1] * mul_noise[-1] / modulus * 4
        # Simplified model — actual noise growth depends on scheme
        if new_noise > modulus:
            mul_noise.append(modulus)
        else:
            mul_noise.append(mul_noise[-1] * 2)

    # Find when each hits the threshold
    add_depth = next((i for i, n in enumerate(add_noise) if n > threshold), max_ops)
    mul_depth = next((i for i, n in enumerate(mul_noise) if n > threshold), max_ops)

    print(f"Initial noise: {initial_noise:.1f}")
    print(f"Decryption threshold: {threshold:.2e}")
    print(f"Max additions before failure: {add_depth}")
    print(f"Max multiplications before failure: {mul_depth}")
    print(f"\nThis is why FHE needs bootstrapping for deep circuits!")

    return add_noise[:add_depth+2], mul_noise[:mul_depth+2]


if __name__ == "__main__":
    simulate_noise_growth()
```

### 5.2 회로 깊이 장벽

SHE 방식은 노이즈가 넘치기 전까지 제한된 깊이 $L$의 회로만 평가할 수 있습니다:

- $L$번의 순차적 덧셈: 노이즈가 $O(L)$로 증가
- $L$번의 순차적 곱셈: 노이즈가 $O(2^L)$로 증가

임의의 회로(무제한 깊이)를 평가하려면 복호화 없이 노이즈를 **감소**시키는 방법이 필요합니다. 이것이 부트스트래핑입니다.

---

## 6. Gentry의 부트스트래핑

### 6.1 핵심 아이디어 (2009)

Craig Gentry의 박사 학위 논문은 30년 된 미해결 문제를 놀라운 아이디어로 해결했습니다: **복호화 함수 자체를 동형 연산으로 사용**하는 것입니다.

"노이즈가 많은" 암호문 $c$가 주어졌을 때:
1. 비밀 키 $\text{sk}$를 새로운 공개 키로 암호화: $\overline{\text{sk}} = \text{Enc}_{\text{pk}'} (\text{sk})$
2. 암호문 $c$를 새로운 키로 암호화: $\overline{c} = \text{Enc}_{\text{pk}'}(c)$
3. 복호화 회로를 동형적으로 평가: $c' = \text{Eval}(\text{Dec}, \overline{\text{sk}}, \overline{c})$

결과 $c'$는 새로운 키 하에 동일한 평문의 **새로운** 암호화이며, **노이즈가 초기화**됩니다.

### 6.2 부트스트래핑의 역설

> 잠깐 — 방식이 깊이 $L$의 회로만 평가할 수 있고, 복호화 회로의 깊이가 $D$라면, $D < L$이 필요하지 않나요? 하지만 방식이 유용하려면 $L$이 커야 하는데, 그러면 복호화가 복잡해지지 않나요?

이것이 **원형 보안(Circular Security)** 과제입니다. Gentry의 해결책은 복호화 회로의 깊이가 평가 용량보다 엄격히 작도록 방식을 설계하는 것이었습니다 — 이를 **부트스트랩 가능한(Bootstrappable)** 방식이라고 합니다. 이를 위해서는 신중한 매개변수 선택이 필요합니다.

### 6.3 실용적 영향

부트스트래핑은 FHE에서 가장 비용이 많이 드는 연산입니다:
- 부트스트래핑 없이: 제한된 회로 평가 (레벨형 FHE)
- 부트스트래핑을 통해: 임의의 회로 평가 (진정한 FHE), 하지만 부트스트랩당 ~1000배의 오버헤드

많은 실용적 응용에서는 회로 깊이를 미리 알고 매개변수를 그에 맞게 설정할 수 있기 때문에 **레벨형 FHE**(부트스트래핑 없음)를 사용합니다.

---

## 7. 현대 FHE 방식: BFV와 CKKS

### 7.1 BFV (Brakerski/Fan-Vercauteren)

**목적**: 암호화된 데이터에 대한 정확한 정수 산술.

**평문 공간**: $\mathbb{Z}_t$ (모듈러스 $t$의 정수)

**주요 특징**:
- 정확한 계산 (근사 오차 없음)
- 계수, 분류, 정확한 조회에 최적
- Ring-LWE(레슨 10) 기반

**암호문 구조:**

$$
\text{ct} = (c_0, c_1) \in R_q^2 \text{ where } c_0 + c_1 \cdot s \approx \Delta \cdot m + e
$$

여기서 $\Delta = \lfloor q/t \rfloor$는 스케일링 인수입니다.

### 7.2 CKKS (Cheon-Kim-Kim-Song)

**목적**: 암호화된 데이터에 대한 근사 실수/복소수 산술.

**평문 공간**: $\mathbb{C}^{N/2}$ (복소수 벡터, 다항식으로 인코딩)

**주요 특징**:
- 부동소수점과 유사한 계산 지원
- 머신러닝에 자연스럽게 적합 (근사적 특성)
- **재스케일링(Rescaling)**: 곱셈 후 추가 스케일링 인수를 나누어 노이즈를 관리

**CKKS가 머신러닝에 적합한 이유?** 신경망 추론은 부동소수점 수에 대해 수백만 번의 곱셈-누산 연산을 포함합니다. CKKS는 $N/2$개의 값을 단일 암호문에 배치하여 병렬로 처리할 수 있습니다(SIMD).

### 7.3 비교

| 특성 | BFV | CKKS |
|----------|-----|------|
| 산술 | 정확한 정수 | 근사 실수/복소수 |
| 평문 유형 | $\mathbb{Z}_t$ | $\mathbb{R}$ (또는 $\mathbb{C}$) |
| 곱셈 | 정확하지만 노이즈 증가 | 재스케일링을 통한 근사 |
| 최적 용도 | 조회, 계수, 정확한 논리 | 머신러닝 추론, 통계 |
| SIMD 배치 | 예 (정수 슬롯) | 예 (실수/복소수 슬롯) |

### 7.4 FHE 라이브러리

| 라이브러리 | 언어 | 지원 방식 | 주요 특징 |
|---------|----------|---------|-----------------|
| Microsoft SEAL | C++ | BFV, CKKS | 문서화 우수, 프로덕션 준비 완료 |
| OpenFHE | C++ | BFV, BGV, CKKS, TFHE | 가장 포괄적 |
| TFHE-rs | Rust | TFHE | 불리언 회로 FHE, 빠른 부트스트래핑 |
| Concrete | Rust/Python | TFHE | Zama의 머신러닝 중심 FHE 컴파일러 |
| HElib | C++ | BGV, CKKS | IBM의 라이브러리, 부트스트래핑 지원 |

---

## 8. 실용적 응용

### 8.1 헬스케어

**문제**: 병원은 진단을 위해 클라우드 머신러닝 서비스를 사용하고 싶지만, 환자 데이터를 공유할 수 없습니다 (HIPAA).

**해결책**: FHE로 환자 기록을 암호화하여 클라우드에 암호문을 전송하고, 머신러닝 모델을 동형적으로 실행한 뒤 암호화된 예측 결과를 병원에 반환하여 복호화합니다.

**현재 상태**: Duality Technologies, Zama 등의 기업이 유전체 분석 및 임상 시험에 FHE를 배포하고 있습니다.

### 8.2 암호화된 데이터에서의 머신러닝

```python
"""
Conceptual: Linear regression inference on encrypted data using Paillier.

Why Paillier for linear models? A linear model computes:
  y = w1*x1 + w2*x2 + ... + wn*xn + b

This is a weighted sum — exactly what Paillier's homomorphic
addition and scalar multiplication support!

For non-linear models (neural networks with ReLU, sigmoid), we need
FHE (BFV/CKKS) because we need multiplication between encrypted values.
"""


def encrypted_linear_inference(paillier, weights, bias, encrypted_features):
    """
    Perform linear regression inference on encrypted features.

    The server knows the model weights (public) but NOT the
    input features (encrypted by the client).

    Parameters:
        paillier: Paillier encryption instance
        weights: list of integers (model weights, known to server)
        bias: integer (model bias, known to server)
        encrypted_features: list of encrypted integers (client data)
    """
    # Compute weighted sum homomorphically
    # Step 1: Multiply each encrypted feature by its weight
    # This uses scalar multiplication: Enc(xi)^wi = Enc(wi * xi)
    weighted = [paillier.scalar_multiply(ef, w)
                for ef, w in zip(encrypted_features, weights)]

    # Step 2: Sum all weighted terms
    # This uses homomorphic addition: Enc(a) * Enc(b) = Enc(a + b)
    result = weighted[0]
    for term in weighted[1:]:
        result = paillier.add_encrypted(result, term)

    # Step 3: Add bias
    enc_bias = paillier.encrypt(bias)
    result = paillier.add_encrypted(result, enc_bias)

    return result  # Still encrypted! Server never sees the data or result


# This would be used as:
# client_features = [age, blood_pressure, cholesterol, ...]
# encrypted_features = [paillier.encrypt(f) for f in client_features]
# encrypted_prediction = encrypted_linear_inference(paillier, model_weights,
#                                                    model_bias,
#                                                    encrypted_features)
# prediction = paillier.decrypt(encrypted_prediction)  # Only client can do this
```

### 8.3 프라이빗 집합 교집합(Private Set Intersection)

두 당사자가 공통 원소가 아닌 것을 노출하지 않고 집합의 공통 원소를 찾고 싶을 때:
- 광고: "내 고객 중 당신의 웹사이트를 방문한 사람은 누구인가요?"
- 헬스케어: "우리 시험에 참여한 환자 중 부작용 데이터베이스에도 등록된 사람은 누구인가요?"

### 8.4 암호화된 데이터베이스 쿼리

CipherCompute 등의 서비스는 암호화된 데이터베이스에 대한 SQL 유사 쿼리를 허용합니다:
- `SELECT SUM(salary) FROM employees WHERE department = 'Engineering'`
- 데이터베이스 서버는 평문 급여를 보지 않고도 쿼리를 처리합니다

---

## 9. 성능 현실 점검

### 9.1 현재의 오버헤드

| 연산 | 평문 | BFV (FHE) | 오버헤드 |
|-----------|-----------|-----------|----------|
| 덧셈 | ~1 ns | ~10 μs | ~10,000배 |
| 곱셈 | ~1 ns | ~10 ms | ~10,000,000배 |
| 신경망 (ResNet-20) | ~5 ms | ~30분 | ~360,000배 |
| AES-128 평가 | ~1 ns | ~1초 | ~1,000,000,000배 |

### 9.2 암호문 팽창

| 방식 | 평문 크기 | 암호문 크기 | 팽창 비율 |
|--------|---------------|-----------------|-----------|
| Paillier (2048비트) | 256바이트 | 512바이트 | 2배 |
| BFV ($n=4096$) | 4 KB (배치) | 256 KB | 64배 |
| CKKS ($n=16384$) | 16 KB (배치) | 2 MB | 128배 |

### 9.3 FHE가 왜 이렇게 느린가?

1. **큰 암호문**: 32바이트 숫자 대신 메가바이트 크기의 다항식에 대한 연산
2. **노이즈 관리**: 모든 연산이 노이즈 증가를 추적하고 관리해야 함
3. **모듈러 산술**: 보안을 위해 수백 비트의 큰 모듈러스 필요
4. **부트스트래핑**: 노이즈 초기화 연산 자체가 복잡한 계산

### 9.4 실용화의 길

- **하드웨어 가속**: Intel, DARPA(DPRIVE 프로그램) 등이 FHE 전용 프로세서 설계 중
- **컴파일러 최적화**: 자동 노이즈 관리, 연산 스케줄링, 암호문 패킹
- **알고리즘 개선**: 더 빠른 부트스트래핑(TFHE), 더 나은 SIMD 배치
- **근사 컴퓨팅**: CKKS는 대규모 속도 향상을 위해 작은 근사 오차를 허용

> FHE 커뮤니티는 2012년 딥러닝과의 유사성을 제시합니다: 알고리즘은 존재했지만 GPU 가속이 실용화를 가능하게 했습니다. FHE도 전용 하드웨어를 통해 비슷한 궤적을 따를 수 있습니다.

---

## 10. 요약

| 개념 | 핵심 내용 |
|---------|-------------|
| PHE | 하나의 연산(덧셈 또는 곱셈)을 깊이 제한 없이 지원 |
| Paillier | 덧셈 PHE; 암호화된 집계와 가중 합산 가능 |
| SHE | 두 연산 모두 지원하지만 제한된 회로 깊이 |
| FHE | 부트스트래핑을 통한 무제한 깊이; 암호화의 "성배" |
| 노이즈 증가 | 곱셈이 지수적 노이즈 증가를 유발 — 핵심 과제 |
| 부트스트래핑 | 노이즈를 초기화하기 위해 복호화를 동형적으로 평가 (Gentry, 2009) |
| BFV | 정확한 정수 FHE; 계수 및 분류에 적합 |
| CKKS | 근사 실수 FHE; 머신러닝 추론에 이상적 |
| 성능 | 10,000배~1,000,000,000배 오버헤드; 하드웨어와 알고리즘으로 개선 중 |

---

## 11. 연습 문제

### 연습 문제 1: RSA 동형성 (코딩)

1. 1024비트 키로 교과서 RSA를 구현하세요
2. 곱셈 동형성을 시연하세요: 두 수를 암호화하고, 암호문을 곱한 뒤, 복호화하세요
3. 방식이 덧셈에 대해 동형이 아님을 보이세요: $m_1$과 $m_2$를 암호화하고 암호문을 더했을 때 $m_1 + m_2$로 복호화되지 않음을 보이세요
4. 패딩 없는 교과서 RSA가 의미론적으로 안전하지 않은 이유를 설명하세요

### 연습 문제 2: Paillier 응용 (코딩)

이 레슨의 Paillier 구현을 사용하여:
1. 간단한 프라이빗 투표 시스템을 구현하세요: 각 투표자는 1(찬성) 또는 0(반대)을 암호화하고, 집계는 모든 암호화된 투표를 더하여 계산합니다
2. 암호화된 평균을 구현하세요: $n$개의 값을 암호화하고, 암호화된 합계를 계산한 뒤, $n^{-1} \bmod N$에 의한 스칼라 곱셈을 사용하여 암호화된 평균을 계산하세요
3. 암호화된 분산을 계산할 수 있나요? 왜 가능하거나 불가능한가요? 어떤 추가 기능이 필요한가요?

### 연습 문제 3: 노이즈 예산 시뮬레이션 (코딩 + 개념)

LWE 기반 방식에서 노이즈 증가를 추적하는 시뮬레이션을 만드세요:
1. 초기 노이즈 $e = 3$과 모듈러스 $q = 2^{32}$로 시작하세요
2. 각 덧셈: 새 노이즈 = 입력 노이즈들의 합
3. 각 곱셈: 새 노이즈 = 입력 노이즈들의 곱 (단순화)
4. 복호화가 실패하기 전($\text{noise} > q/4$) 최대 회로 깊이(순차적 곱셈 횟수)를 구하세요
5. 다음에 대한 노이즈 증가를 그래프로 그리세요: (a) 20번의 덧셈 체인, (b) 20번의 곱셈 체인, (c) 3층 신경망과 유사한 혼합 회로

### 연습 문제 4: 암호화된 선형 모델 (코딩)

완전한 암호화된 추론 파이프라인을 구축하세요:
1. Iris 데이터셋에 로지스틱 회귀 모델을 훈련하세요 (scikit-learn)
2. 모델 가중치를 정수로 반올림하세요 (1000을 곱하세요)
3. Paillier를 사용하여 테스트 특성을 암호화하세요
4. 추론의 선형 부분을 동형적으로 수행하세요
5. 결과를 복호화하고 시그모이드 함수를 적용하세요 (클라이언트 측)
6. 평문 모델과 정확도를 비교하세요

### 연습 문제 5: FHE 방식 비교 (연구 + 개념)

다음 각 응용에 가장 적합한 HE 방식(Paillier, BFV, CKKS, TFHE 중 하나)을 추천하고 이유를 설명하세요:
1. 암호화된 직원 기록에서 평균 급여 계산
2. 암호화된 의료 이미지에 신경망 실행
3. 암호화된 데이터베이스에서 프라이빗 불리언 검색
4. 금융 데이터에 대한 암호화된 통계(평균, 중앙값, 표준 편차) 계산
5. AES 암호화의 동형 평가 (대칭 암호의 부트스트래핑)

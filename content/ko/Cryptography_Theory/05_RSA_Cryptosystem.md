# 레슨 5: RSA 암호체계

**이전**: [해시 함수](./04_Hash_Functions.md) | **다음**: [타원곡선 암호](./06_Elliptic_Curve_Cryptography.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. RSA 키 쌍을 생성하고 각 파라미터($p$, $q$, $n$, $e$, $d$)의 역할을 설명할 수 있다
2. RSA 암호화와 복호화가 역연산임을 수학적으로 증명할 수 있다
3. RSA 암호화, 복호화, 서명, 검증을 처음부터 구현할 수 있다
4. 교과서 RSA 공격(소 지수 공격, 공통 모듈러스, 위너 공격)을 설명하고 패딩이 필수적인 이유를 설명할 수 있다
5. PKCS#1 v1.5 패딩과 OAEP를 비교하고 블라이헨바허 공격을 설명할 수 있다
6. RSA 키 크기 권장 사항과 양자 컴퓨팅의 영향을 평가할 수 있다

---

1977년 리베스트(Rivest), 샤미르(Shamir), 애들먼(Adleman)이 발표한 RSA는 최초의 실용적인 공개 키 암호체계이며, 오늘날에도 가장 광범위하게 배포된 알고리즘 중 하나입니다. 모든 TLS 연결, 코드 서명 인증서, SSH 키 교환에 RSA가 관여할 수 있습니다. RSA의 우아함은 심오한 수학적 통찰에 있습니다: 두 개의 큰 소수를 곱하는 것은 쉽지만, 그 곱을 인수분해하는 것은 극도로 어렵습니다. 이 레슨은 레슨 1의 수론 기초 위에 RSA를 구축하고, RSA의 수학적 아름다움과 실용적 함정을 모두 드러냅니다.

> **비유:** RSA는 우편함과 같습니다. 누구나 편지를 넣을 수 있지만(공개 키로 암호화), 실물 열쇠를 가진 소유자만 열 수 있습니다(개인 키로 복호화). RSA의 "어려움"이란, 우편함을 만드는 것(소수를 곱하는 것)은 간단하지만, 침입하는 것(곱을 인수분해하는 것)은 거의 불가능하다는 점입니다.

## 목차

1. [키 생성](#1-키-생성)
2. [암호화와 복호화](#2-암호화와-복호화)
3. [정확성 증명](#3-정확성-증명)
4. [RSA 서명](#4-rsa-서명)
5. [교과서 RSA 공격](#5-교과서-rsa-공격)
6. [패딩: PKCS#1과 OAEP](#6-패딩-pkcs1과-oaep)
7. [블라이헨바허 공격](#7-블라이헨바허-공격)
8. [키 크기와 성능](#8-키-크기와-성능)
9. [완전한 RSA 구현](#9-완전한-rsa-구현)
10. [연습 문제](#10-연습-문제)

---

## 1. 키 생성

### 1.1 알고리즘

1. 두 개의 크고 서로 다른 소수 $p$와 $q$를 선택한다 (RSA-2048의 경우 각각 일반적으로 1024비트)
2. $n = p \cdot q$를 계산한다 (**모듈러스(modulus)**)
3. $\phi(n) = (p-1)(q-1)$를 계산한다 (오일러의 토션트(Euler's totient), 더 정확하게는 $\lambda(n) = \text{lcm}(p-1, q-1)$)
4. $1 < e < \phi(n)$이고 $\gcd(e, \phi(n)) = 1$을 만족하는 $e$를 선택한다 (**공개 지수(public exponent)**)
5. $d = e^{-1} \bmod \phi(n)$을 계산한다 (**개인 지수(private exponent)**)

**공개 키:** $(n, e)$
**개인 키:** $(n, d)$ (또는 동등하게, $p$, $q$, $d$, CRT 파라미터)

### 1.2 파라미터 선택

| 파라미터 | 일반적인 값 | 근거 |
|----------|-------------|------|
| $e$ | 65537 ($2^{16} + 1$) | 작은 해밍 가중치(Hamming weight) → 빠른 암호화; 소 지수 공격을 방지할 만큼 충분히 큰 값 |
| $p, q$ | 각각 1024비트 이상 | $n$의 인수분해가 불가능해야 함 |
| $\|p - q\|$ | 크게 | 페르마 인수분해($p \approx q$를 이용하는) 방지 |

```python
import random

def generate_prime(bits, k=20):
    """Generate a random prime of the specified bit length.

    Uses Miller-Rabin primality test from Lesson 1.
    """
    def miller_rabin(n, k):
        if n < 2: return False
        if n == 2 or n == 3: return True
        if n % 2 == 0: return False
        s, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else:
                return False
        return True

    while True:
        n = random.getrandbits(bits)
        n |= (1 << (bits - 1)) | 1
        if miller_rabin(n, k):
            return n

def rsa_keygen(bits=2048):
    """Generate an RSA key pair.

    Why e = 65537:
    - Binary: 10000000000000001 (only 2 bits set)
    - Fast modular exponentiation: only 17 squarings + 1 multiply
    - Large enough to prevent small-exponent attacks
    - Standard choice across all major implementations

    Why phi(n) vs lambda(n):
    - Textbooks use phi(n) = (p-1)(q-1)
    - PKCS#1 recommends lambda(n) = lcm(p-1, q-1)
    - Both work; lambda(n) gives a smaller d (slightly faster)
    """
    from math import gcd

    half_bits = bits // 2

    while True:
        p = generate_prime(half_bits)
        q = generate_prime(half_bits)

        # Ensure p != q and |p - q| is large (prevents Fermat factoring)
        if p == q:
            continue
        if abs(p - q).bit_length() < half_bits - 10:
            continue  # p and q too close together

        n = p * q
        phi_n = (p - 1) * (q - 1)

        e = 65537
        if gcd(e, phi_n) != 1:
            continue  # Extremely rare, but check anyway

        d = pow(e, -1, phi_n)  # Modular inverse (Python 3.8+)

        public_key = (n, e)
        private_key = (n, d, p, q)
        return public_key, private_key

# Generate a small key pair for demonstration
pub, priv = rsa_keygen(512)  # 512-bit for speed; use 2048+ in production
n, e = pub
_, d, p, q = priv

print(f"p = {p}")
print(f"q = {q}")
print(f"n = {n}")
print(f"e = {e}")
print(f"d = {d}")
print(f"n bit length: {n.bit_length()}")
```

### 1.3 $p$와 $q$를 비밀로 유지해야 하는 이유

$p$와 $q$를 알면 $\phi(n) = (p-1)(q-1)$을 계산할 수 있고, 이로부터 $d = e^{-1} \bmod \phi(n)$을 즉시 구할 수 있습니다. 마찬가지로 $\phi(n)$과 $n$을 알면 $n$을 인수분해할 수 있습니다 ($p + q = n - \phi(n) + 1$이고 $p \cdot q = n$이므로 이차방정식이 성립합니다).

---

## 2. 암호화와 복호화

### 2.1 교과서 RSA

**암호화:** 공개 키 $(n, e)$와 평문 $m \in \{0, 1, \ldots, n-1\}$이 주어지면:

$$c = m^e \bmod n$$

**복호화:** 개인 키 $(n, d)$와 암호문 $c$가 주어지면:

$$m = c^d \bmod n$$

```python
def rsa_encrypt(m, public_key):
    """Textbook RSA encryption.

    Why modular exponentiation: Python's pow(m, e, n) uses the
    square-and-multiply algorithm (Lesson 1, Section 7), making
    this efficient even for 2048-bit numbers.
    """
    n, e = public_key
    if m >= n:
        raise ValueError("Message must be less than modulus n")
    return pow(m, e, n)

def rsa_decrypt(c, private_key):
    """Textbook RSA decryption."""
    n, d = private_key[0], private_key[1]
    return pow(c, d, n)

# Encrypt and decrypt a number
message = 42
ciphertext = rsa_encrypt(message, pub)
decrypted = rsa_decrypt(ciphertext, priv)

print(f"Message:    {message}")
print(f"Ciphertext: {ciphertext}")
print(f"Decrypted:  {decrypted}")
print(f"Correct:    {message == decrypted}")
```

### 2.2 CRT 최적화

RSA 복호화는 중국인의 나머지 정리(Chinese Remainder Theorem, CRT)(레슨 1, 섹션 6)를 이용해 약 4배 가속할 수 있습니다:

$$m_p = c^{d \bmod (p-1)} \bmod p$$
$$m_q = c^{d \bmod (q-1)} \bmod q$$
$$m = \text{CRT}(m_p, m_q, p, q)$$

```python
def rsa_decrypt_crt(c, private_key):
    """RSA decryption with CRT optimization.

    Why CRT is ~4x faster:
    - Two exponentiations with half-size exponents and moduli
    - Exponentiation cost is O(k^3) where k is bit length
    - Half the bits → (1/2)^3 = 1/8 the cost per exponentiation
    - Two exponentiations: 2 * (1/8) = 1/4 the total cost
    """
    n, d, p, q = private_key

    # Precompute (these can be stored with the private key)
    dp = d % (p - 1)
    dq = d % (q - 1)
    q_inv = pow(q, -1, p)

    # Decrypt modulo p and q separately
    m_p = pow(c, dp, p)
    m_q = pow(c, dq, q)

    # Combine using Garner's formula (efficient CRT for 2 moduli)
    h = (q_inv * (m_p - m_q)) % p
    m = m_q + h * q

    return m

# Verify CRT decryption matches standard decryption
ciphertext = rsa_encrypt(42, pub)
m1 = rsa_decrypt(ciphertext, priv)
m2 = rsa_decrypt_crt(ciphertext, priv)
print(f"Standard: {m1}, CRT: {m2}, Match: {m1 == m2}")
```

---

## 3. 정확성 증명

### 3.1 $m^{ed} \equiv m \pmod{n}$인 이유

암호화된 메시지를 복호화하면 원래 메시지가 복원됨을 보여야 합니다.

**주어진 조건:** $ed \equiv 1 \pmod{\phi(n)}$, 즉 어떤 정수 $k$에 대해 $ed = 1 + k\phi(n)$

**경우 1: $\gcd(m, n) = 1$**

오일러의 정리(Euler's theorem)(레슨 1, 섹션 5)에 의해:

$$m^{\phi(n)} \equiv 1 \pmod{n}$$

따라서:

$$m^{ed} = m^{1 + k\phi(n)} = m \cdot (m^{\phi(n)})^k \equiv m \cdot 1^k = m \pmod{n}$$

**경우 2: $\gcd(m, n) \neq 1$**

$n = pq$이므로, $\gcd(m, n) \neq 1$이면 $p \mid m$ 또는 $q \mid m$ (단, $m < n$이므로 둘 다는 아님).

$p \mid m$이라 가정하면 ($m \equiv 0 \pmod{p}$):
- $m^{ed} \equiv 0 \equiv m \pmod{p}$
- $\pmod{q}$의 경우: $\gcd(m, q) = 1$이므로 페르마 정리에 의해 $m^{q-1} \equiv 1 \pmod{q}$, 따라서 $m^{ed} \equiv m \pmod{q}$

CRT에 의해 $m^{ed} \equiv m \pmod{n}$. $\blacksquare$

### 3.2 카마이클 함수 사용

$\phi(n)$ 대신 $\lambda(n) = \text{lcm}(p-1, q-1)$을 사용해도 증명이 성립합니다. $\phi(n)$은 항상 $\lambda(n)$의 배수이므로, $ed \equiv 1 \pmod{\lambda(n)}$이면 같은 논리로 정확성이 보장됩니다.

---

## 4. RSA 서명

### 4.1 해시-후-서명(Hash-Then-Sign)

RSA 서명은 개인 키를 사용해 메시지 해시를 "암호화"합니다:

$$\text{서명: } \sigma = H(m)^d \bmod n$$
$$\text{검증: } H(m) \stackrel{?}{=} \sigma^e \bmod n$$

```python
import hashlib

def rsa_sign(message, private_key):
    """RSA signature: hash the message, then 'decrypt' the hash.

    Why hash first: signing the raw message would be slow (message
    could be gigabytes) and insecure (textbook RSA has algebraic
    structure that enables forgery — see Section 5).
    """
    n, d = private_key[0], private_key[1]
    h = int(hashlib.sha256(message).hexdigest(), 16) % n
    signature = pow(h, d, n)
    return signature

def rsa_verify(message, signature, public_key):
    """RSA signature verification: 'encrypt' the signature and compare."""
    n, e = public_key
    h = int(hashlib.sha256(message).hexdigest(), 16) % n
    recovered = pow(signature, e, n)
    return h == recovered

# Sign and verify
message = b"This is an important document."
sig = rsa_sign(message, priv)
valid = rsa_verify(message, sig, pub)
print(f"Signature valid: {valid}")

# Tampered message fails verification
tampered = b"This is an important document!"
valid_tampered = rsa_verify(tampered, sig, pub)
print(f"Tampered signature valid: {valid_tampered}")
```

### 4.2 교과서 RSA 서명의 실존적 위조

해시 없이 서명하면 공격자가 서명을 위조할 수 있습니다:

**공격 1: 서명을 먼저 선택하기.** 임의의 $\sigma$를 선택하고 $m = \sigma^e \bmod n$을 계산합니다. 그러면 $(m, \sigma)$는 유효한 메시지-서명 쌍이 됩니다 (다만 $m$은 무작위 쓰레기값).

**공격 2: 곱셈적 성질 이용.** $m_1$에 대한 서명 $\sigma_1$과 $m_2$에 대한 서명 $\sigma_2$가 주어지면:

$$\sigma_1 \cdot \sigma_2 = (m_1^d)(m_2^d) = (m_1 m_2)^d \bmod n$$

이것은 $m_1 \cdot m_2 \bmod n$에 대한 유효한 서명입니다. 해시를 사용하면 $H(m_1 \cdot m_2) \neq H(m_1) \cdot H(m_2)$이므로 이 공격이 차단됩니다 (해시 함수는 곱셈적이지 않습니다).

---

## 5. 교과서 RSA 공격

패딩 없는 교과서 RSA ($c = m^e \bmod n$)는 여러 공격에 취약합니다:

### 5.1 소 공개 지수 공격

$e = 3$이고 $m^3 < n$ (메시지가 작은 경우), $c = m^3$은 정수 범위에서 계산됩니다 (모듈러 환산이 일어나지 않음). 공격자는 단순히 $m = \sqrt[3]{c}$를 계산합니다.

```python
def small_exponent_attack():
    """Demonstrate the small-e attack on textbook RSA.

    Why e=3 is dangerous without padding: if the message is small
    enough that m^3 < n, the modular reduction never happens, and
    the attacker can compute the cube root over the integers.
    """
    # Generate RSA with e=3 (for demonstration)
    from math import gcd, isqrt

    # Small example
    p, q = 61, 53
    n = p * q  # 3233
    e = 3
    phi_n = (p-1) * (q-1)  # 3120
    # Verify gcd(3, 3120) = 3 — actually, this doesn't work for e=3 with these primes
    # Let's use primes where it works
    p, q = 59, 71
    n = p * q  # 4189
    phi_n = (p-1) * (q-1)  # 4060
    assert gcd(3, phi_n) == 1
    d = pow(3, -1, phi_n)

    # Small message: m^3 < n
    m = 15
    c = pow(m, 3, n)
    print(f"n = {n}, e = 3, m = {m}, c = {c}")
    print(f"m^3 = {m**3}, n = {n}, m^3 < n: {m**3 < n}")

    if m**3 < n:
        # Attacker computes integer cube root
        recovered = round(c ** (1/3))
        # More precise method for large numbers:
        def integer_cube_root(x):
            if x < 0: return -integer_cube_root(-x)
            if x == 0: return 0
            guess = int(x ** (1/3)) + 1
            while guess ** 3 > x: guess -= 1
            return guess

        recovered = integer_cube_root(c)
        print(f"Recovered m = {recovered} (correct: {recovered == m})")

small_exponent_attack()
```

### 5.2 공통 모듈러스 공격

같은 메시지가 동일한 모듈러스에서 $\gcd(e_1, e_2) = 1$인 서로 다른 공개 지수 $e_1, e_2$로 암호화된 경우:

$$c_1 = m^{e_1} \bmod n, \quad c_2 = m^{e_2} \bmod n$$

$ae_1 + be_2 = 1$을 만족하는 $a, b$를 구하면 (확장 유클리드 알고리즘(Extended GCD)):

$$c_1^a \cdot c_2^b = m^{ae_1 + be_2} = m^1 = m \pmod{n}$$

### 5.3 하스타드의 브로드캐스트 공격

같은 메시지 $m$이 $e$개의 서로 다른 공개 키로 암호화된 경우 (모두 $e = 3$):

$$c_1 = m^3 \bmod n_1, \quad c_2 = m^3 \bmod n_2, \quad c_3 = m^3 \bmod n_3$$

CRT를 이용해 $c' = m^3 \bmod (n_1 n_2 n_3)$을 구합니다. $m < \min(n_i)$이므로 $m^3 < n_1 n_2 n_3$이고, 따라서 $c' = m^3$은 정수 범위에서 성립하여 $m = \sqrt[3]{c'}$로 복원할 수 있습니다.

### 5.4 위너 공격

$d < \frac{1}{3} n^{1/4}$이면, $e/n$의 연분수 전개(continued fraction expansion)가 $d$를 드러냅니다.

```python
def wieners_attack_demo():
    """Demonstrate Wiener's attack on RSA with small d.

    Why small d is dangerous: the key equation ed ≡ 1 (mod phi(n))
    can be written as ed = 1 + k*phi(n), giving e/n ≈ k/d.
    The continued fraction expansion of e/n reveals k/d when
    d is small (d < n^(1/4) / 3).
    """
    def continued_fraction(numerator, denominator):
        """Compute the continued fraction expansion of num/den."""
        cf = []
        while denominator:
            q = numerator // denominator
            cf.append(q)
            numerator, denominator = denominator, numerator - q * denominator
        return cf

    def convergents(cf):
        """Compute convergents from a continued fraction."""
        convs = []
        h_prev, h_curr = 0, 1
        k_prev, k_curr = 1, 0
        for a in cf:
            h_prev, h_curr = h_curr, a * h_curr + h_prev
            k_prev, k_curr = k_curr, a * k_curr + k_prev
            convs.append((h_curr, k_curr))
        return convs

    # Example with small d (vulnerable to Wiener's attack)
    p, q = 1009, 3643
    n = p * q  # 3675787
    phi_n = (p - 1) * (q - 1)
    d = 29  # Deliberately small
    e = pow(d, -1, phi_n)

    print(f"n = {n}, e = {e}, d = {d}")
    print(f"d < n^(1/4)/3 = {n**(1/4)/3:.1f}: {d < n**(1/4)/3}")

    # Attack: compute continued fraction of e/n
    cf = continued_fraction(e, n)
    convs = convergents(cf)

    for k, d_guess in convs:
        if k == 0:
            continue
        # Check if d_guess could be the private exponent
        if (e * d_guess - 1) % k == 0:
            phi_guess = (e * d_guess - 1) // k
            # Verify: n - phi + 1 should be p + q, and p*q should be n
            s = n - phi_guess + 1  # p + q
            # p and q are roots of x^2 - s*x + n = 0
            discriminant = s * s - 4 * n
            if discriminant >= 0:
                from math import isqrt
                sqrt_disc = isqrt(discriminant)
                if sqrt_disc * sqrt_disc == discriminant:
                    print(f"Recovered d = {d_guess} (correct: {d_guess == d})")
                    p_recovered = (s + sqrt_disc) // 2
                    q_recovered = (s - sqrt_disc) // 2
                    print(f"Recovered p = {p_recovered}, q = {q_recovered}")
                    break

wieners_attack_demo()
```

---

## 6. 패딩: PKCS#1과 OAEP

### 6.1 패딩이 필수적인 이유

교과서 RSA는 결정론적입니다 ($E(m) = m^e \bmod n$). 이는 다음을 의미합니다:
- 동일한 메시지는 동일한 암호문을 생성합니다 (의미론적 안전성(semantic security) 없음)
- 곱셈적 성질이 선택 암호문 공격을 가능하게 합니다
- 작은 메시지는 거듭제곱 근 추출에 취약합니다

패딩은 무작위성과 구조를 추가하여 이러한 공격을 방지합니다.

### 6.2 PKCS#1 v1.5

가장 오래된 광범위 배포 RSA 패딩 방식 (여전히 많은 시스템에서 사용됨):

```
0x00 || 0x02 || [random non-zero bytes] || 0x00 || [message]
```

- `0x02` 바이트는 암호화를 나타냅니다 (서명에는 `0x01` 사용)
- 최소 8바이트의 랜덤 비-영(non-zero) 패딩 바이트
- `0x00` 구분자가 실제 메시지의 시작을 표시합니다

**문제:** 블라이헨바허 공격(섹션 7)에 취약합니다.

### 6.3 OAEP (최적 비대칭 암호화 패딩)

OAEP(Optimal Asymmetric Encryption Padding)(벨라레-로거웨이(Bellare-Rogaway), 1994; PKCS#1 v2.2에 표준화)는 **CCA 안전성(선택 암호문 공격에 대한 안전성)**을 제공합니다:

```
         ┌──── message ────┐
         │ m || 00...01    │
         │   (padded)      │
seed ──→ [G]  ──XOR──→ maskedDB
  │                         │
  └────────── [H] ←────────┘
         maskedSeed

OAEP output = 0x00 || maskedSeed || maskedDB
```

여기서 $G$와 $H$는 마스크 생성 함수(mask generation functions)(일반적으로 SHA-256 기반)입니다.

```python
def oaep_concept():
    """Explain OAEP's structure and security properties.

    Why OAEP over PKCS#1 v1.5:
    1. Provably CCA-secure (in the random oracle model)
    2. Randomized: same message → different ciphertexts each time
    3. All-or-nothing transform: partial information about the
       plaintext cannot be extracted from the ciphertext
    4. Not vulnerable to Bleichenbacher's attack
    """
    import hashlib
    import os

    def mgf1(seed, length, hash_func=hashlib.sha256):
        """Mask Generation Function 1 (MGF1) per PKCS#1.

        Why MGF1: it stretches a short seed into an arbitrarily long
        mask, similar to a key derivation function. Using a hash
        ensures the mask appears random.
        """
        mask = b''
        counter = 0
        while len(mask) < length:
            c = counter.to_bytes(4, 'big')
            mask += hash_func(seed + c).digest()
            counter += 1
        return mask[:length]

    # OAEP encoding (simplified)
    k = 64  # RSA modulus size in bytes (512-bit for demo)
    h_len = 32  # SHA-256 output length
    message = b"Hello, OAEP!"
    max_msg_len = k - 2 * h_len - 2

    if len(message) > max_msg_len:
        raise ValueError(f"Message too long (max {max_msg_len} bytes)")

    # Create data block: lHash || PS || 0x01 || message
    l_hash = hashlib.sha256(b'').digest()  # Hash of empty label
    ps = b'\x00' * (max_msg_len - len(message))  # Padding
    db = l_hash + ps + b'\x01' + message

    # Generate random seed
    seed = os.urandom(h_len)

    # Mask db with seed
    db_mask = mgf1(seed, len(db))
    masked_db = bytes(a ^ b for a, b in zip(db, db_mask))

    # Mask seed with masked_db
    seed_mask = mgf1(masked_db, h_len)
    masked_seed = bytes(a ^ b for a, b in zip(seed, seed_mask))

    # OAEP encoded message
    em = b'\x00' + masked_seed + masked_db

    print(f"Message: {message}")
    print(f"OAEP encoded ({len(em)} bytes): {em[:20].hex()}...")
    print(f"Encoded same message again (different randomness):")

    seed2 = os.urandom(h_len)
    db_mask2 = mgf1(seed2, len(db))
    masked_db2 = bytes(a ^ b for a, b in zip(db, db_mask2))
    seed_mask2 = mgf1(masked_db2, h_len)
    masked_seed2 = bytes(a ^ b for a, b in zip(seed2, seed_mask2))
    em2 = b'\x00' + masked_seed2 + masked_db2

    print(f"  {em2[:20].hex()}...")
    print(f"Different encodings: {em != em2}")

oaep_concept()
```

---

## 7. 블라이헨바허 공격

### 7.1 개요

1998년 다니엘 블라이헨바허(Daniel Bleichenbacher)는 PKCS#1 v1.5가 **적응적 선택 암호문 공격(adaptive chosen-ciphertext attack)**에 취약하다는 사실을 증명했습니다. 서버가 복호화된 암호문이 유효한 PKCS#1 v1.5 형식인지 여부를 공개하면 (오류 메시지나 타이밍 차이와 같은 부채널을 통해서라도), 공격자는 임의의 암호문을 복호화할 수 있습니다.

### 7.2 동작 원리

1. 공격자는 복호화하고 싶은 암호문 $c$를 가지고 있습니다
2. 공격자는 배수 $s$를 선택하고 $c' = c \cdot s^e \bmod n$을 서버에 전송합니다
3. RSA의 곱셈적 성질에 의해: $\text{Dec}(c') = m \cdot s \bmod n$
4. 서버는 $m \cdot s$가 유효한 PKCS#1 형식인지 확인합니다 (`0x00 0x02`로 시작하는지)
5. 유효/무효 응답이 $m$의 가능한 값을 점진적으로 좁혀나갑니다
6. 약 100만 번의 쿼리 후 공격자는 $m$을 완전히 복원합니다

### 7.3 영향과 대응 방법

- **DROWN 공격(2016):** SSLv2 서버를 이용해 TLS 연결을 해독
- **ROBOT 공격(2017):** 많은 TLS 구현이 여전히 취약함을 발견
- **대응:** OAEP 사용, 또는 상수 시간 PKCS#1 v1.5 처리 구현 (극히 어려움)

```python
def bleichenbacher_concept():
    """Explain the Bleichenbacher attack strategy.

    Why this attack is devastating:
    - Does NOT require finding factors of n
    - Does NOT require any weakness in AES or the hash function
    - Only requires a padding oracle (valid/invalid distinction)
    - ~1M queries to decrypt a message (trivially fast)

    This attack is why PKCS#1 v1.5 encryption is deprecated and
    OAEP is mandatory for new applications.
    """
    print("Bleichenbacher's Attack (1998):")
    print("=" * 50)
    print()
    print("1. Attacker captures ciphertext c = m^e mod n")
    print("2. Attacker sends c' = c * s^e mod n (for various s)")
    print("3. Server decrypts: m' = m * s mod n")
    print("4. Server checks PKCS#1 format: does m' start with 0x00 0x02?")
    print("5. Server response (YES/NO) narrows the range of m")
    print()
    print("After ~1 million queries:")
    print("  - Range of m narrows to a single value")
    print("  - Attacker has fully decrypted c without the private key")
    print()
    print("Defenses:")
    print("  - Use OAEP (not PKCS#1 v1.5) for encryption")
    print("  - Constant-time processing (no timing side channels)")
    print("  - TLS 1.3 removed RSA key exchange entirely")

bleichenbacher_concept()
```

---

## 8. 키 크기와 성능

### 8.1 RSA 키 크기 권장 사항

| 보안 수준(비트) | RSA 키 크기 | 동등한 AES | 상태 |
|----------------|------------|-----------|------|
| 80 | 1024 | - | **지원 중단됨** |
| 112 | 2048 | AES-128 | 최소 허용 수준 |
| 128 | 3072 | AES-128 | 권장 |
| 192 | 7680 | AES-192 | 보수적 선택 |
| 256 | 15360 | AES-256 | 2030년 이후+ |

### 8.2 RSA vs ECC 성능

동등한 보안 수준에서 RSA 키는 ECC 키보다 훨씬 빠르게 커집니다:

| 보안 수준 | RSA 키 | ECC 키 | 비율 |
|----------|--------|--------|------|
| 128비트 | 3072비트 | 256비트 | 12:1 |
| 192비트 | 7680비트 | 384비트 | 20:1 |
| 256비트 | 15360비트 | 521비트 | 30:1 |

RSA 복호화/서명은 동등한 보안 수준에서 ECC보다 훨씬 느립니다. 이것이 현대 시스템(TLS 1.3, SSH)이 RSA 대신 ECDSA/Ed25519를 점점 더 선호하는 주요 이유입니다 (레슨 6, 7).

### 8.3 양자 컴퓨팅 위협

양자 컴퓨터의 쇼어 알고리즘(Shor's algorithm)은 $n$을 다항 시간에 인수분해할 수 있어 RSA를 완전히 파괴합니다. 암호학적으로 유의미한 양자 컴퓨터는 현재 존재하지 않지만, 이 위협은 다음을 촉발하고 있습니다:

1. **지금 수집, 나중에 복호화:** 공격자들이 오늘날 RSA 암호화 데이터를 기록해 두었다가 양자 컴퓨터가 가용해지면 복호화할 수 있습니다
2. **포스트 양자 전환:** NIST가 RSA를 대체하기 위해 격자 기반 알고리즘(Lesson 10)을 표준화했습니다

---

## 9. 완전한 RSA 구현

```python
import hashlib
import os
import random

class RSA:
    """Complete RSA implementation for educational purposes.

    WARNING: This is for learning only. In production, use a
    well-audited library like `cryptography` (Python) or OpenSSL.

    Why not roll your own crypto:
    - Side-channel attacks (timing, power analysis)
    - Subtle mathematical attacks (Bleichenbacher, Manger)
    - Random number generation vulnerabilities
    - Padding implementation bugs
    """

    def __init__(self, bits=2048):
        self.bits = bits
        self.public_key, self.private_key = self._keygen()

    def _keygen(self):
        from math import gcd
        half = self.bits // 2

        while True:
            p = self._generate_prime(half)
            q = self._generate_prime(half)
            if p == q or abs(p - q).bit_length() < half - 10:
                continue

            n = p * q
            phi = (p - 1) * (q - 1)
            e = 65537

            if gcd(e, phi) != 1:
                continue

            d = pow(e, -1, phi)
            return (n, e), (n, d, p, q)

    def _generate_prime(self, bits):
        while True:
            n = random.getrandbits(bits)
            n |= (1 << (bits - 1)) | 1
            if self._miller_rabin(n):
                return n

    def _miller_rabin(self, n, k=20):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0: return False
        s, d = 0, n - 1
        while d % 2 == 0:
            d //= 2; s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else: return False
        return True

    def encrypt(self, plaintext_bytes):
        """Encrypt bytes using RSA with simple PKCS#1-like padding."""
        n, e = self.public_key
        k = (n.bit_length() + 7) // 8  # Modulus size in bytes

        # Simple padding: 0x00 || 0x02 || random || 0x00 || message
        max_msg = k - 11  # 8 bytes minimum random padding
        if len(plaintext_bytes) > max_msg:
            raise ValueError(f"Message too long (max {max_msg} bytes)")

        ps = bytes(random.randint(1, 255) for _ in range(k - len(plaintext_bytes) - 3))
        em = b'\x00\x02' + ps + b'\x00' + plaintext_bytes

        m = int.from_bytes(em, 'big')
        c = pow(m, e, n)
        return c.to_bytes(k, 'big')

    def decrypt(self, ciphertext_bytes):
        """Decrypt bytes using RSA-CRT with padding removal."""
        n, d, p, q = self.private_key
        c = int.from_bytes(ciphertext_bytes, 'big')

        # CRT decryption
        dp, dq = d % (p - 1), d % (q - 1)
        q_inv = pow(q, -1, p)
        m_p, m_q = pow(c, dp, p), pow(c, dq, q)
        h = (q_inv * (m_p - m_q)) % p
        m = m_q + h * q

        k = (n.bit_length() + 7) // 8
        em = m.to_bytes(k, 'big')

        # Remove padding
        if em[0:2] != b'\x00\x02':
            raise ValueError("Decryption error")
        sep = em.index(b'\x00', 2)
        return em[sep + 1:]

    def sign(self, message):
        """Sign a message using RSA (hash-then-sign)."""
        n, d = self.private_key[0], self.private_key[1]
        h = int(hashlib.sha256(message).hexdigest(), 16)
        # Ensure hash < n
        h = h % n
        sig = pow(h, d, n)
        k = (n.bit_length() + 7) // 8
        return sig.to_bytes(k, 'big')

    def verify(self, message, signature):
        """Verify an RSA signature."""
        n, e = self.public_key
        sig_int = int.from_bytes(signature, 'big')
        recovered = pow(sig_int, e, n)
        h = int(hashlib.sha256(message).hexdigest(), 16) % n
        return recovered == h

# Full demonstration
rsa = RSA(bits=1024)  # 1024-bit for speed in demo

# Encryption / Decryption
message = b"RSA encryption works!"
ct = rsa.encrypt(message)
pt = rsa.decrypt(ct)
print(f"Encrypt/Decrypt: {pt.decode()}")

# Signing / Verification
sig = rsa.sign(b"Sign this document")
valid = rsa.verify(b"Sign this document", sig)
print(f"Signature valid: {valid}")

tampered = rsa.verify(b"Sign this ducument", sig)  # typo in 'document'
print(f"Tampered valid:  {tampered}")
```

---

## 10. 연습 문제

### 연습 문제 1: 손으로 RSA 계산하기 (기초)

$p = 61$, $q = 53$, $e = 17$을 사용하여:
1. $n$, $\phi(n)$, $d$를 계산하라
2. $m = 65$를 암호화하라. 암호문 $c$는 무엇인가?
3. $c$를 복호화하라. $m = 65$가 복원됨을 확인하라.
4. $m = 42$에 서명하라. 서명을 검증하라.

### 연습 문제 2: RSA 키 생성 시간 측정 (중급)

1. 512, 1024, 2048, 4096비트에 대한 RSA 키 생성 시간을 측정하라.
2. 각 키 크기에 대해 32바이트 메시지의 암호화와 복호화 시간을 측정하라.
3. 키 생성 시간과 복호화 시간 대비 키 크기를 그래프로 나타내라.
4. 외삽: 8192비트 RSA 복호화는 얼마나 걸릴까?

### 연습 문제 3: 공통 모듈러스 공격 (중급)

공통 모듈러스 공격을 구현하라:
1. RSA 모듈러스 $n = pq$를 하나 생성하라
2. 두 공개 키를 만들어라: $(n, e_1 = 17)$과 $(n, e_2 = 65537)$ ($\gcd(e_1, e_2) = 1$)
3. 같은 메시지 $m$을 두 키로 암호화하라
4. $c_1$, $c_2$, $e_1$, $e_2$, $n$만 사용하여 ($d_1$, $d_2$ 없이) $m$을 복원하라

### 연습 문제 4: 위너 공격 (도전)

1. $d < n^{1/4} / 3$인 RSA 키 쌍을 생성하라 (먼저 작은 $d$를 선택하고 $e$를 계산)
2. 연분수를 이용한 위너 공격을 구현하라
3. $(n, e)$만으로 $d$가 복원됨을 확인하라
4. 임계값을 찾아라: 공격으로 복원할 수 있는 최대 $d$는 얼마인가?

### 연습 문제 5: RSA-CRT 오류 공격 (도전)

RSA-CRT에 대한 벨코어 공격(Bellcore attack)을 조사하고 설명하라:
1. CRT 계산 중 하나의 오류가 발생하는 경우 (예: $m_p$가 잘못되고 $m_q$는 올바른 경우), 공격자가 $n$을 인수분해할 수 있음을 보여라
2. 시뮬레이션을 구현하라: CRT 복호화에 오류를 주입하고 $p$와 $q$를 복원하라
3. 대응 방법을 설명하라: CRT 복호화 후 $m^e \equiv c \pmod{n}$을 검증하라

---

**이전**: [해시 함수](./04_Hash_Functions.md) | **다음**: [타원곡선 암호](./06_Elliptic_Curve_Cryptography.md)

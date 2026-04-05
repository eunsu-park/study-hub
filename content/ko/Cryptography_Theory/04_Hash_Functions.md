# 레슨 4: 해시 함수

**이전**: [블록 암호 모드](./03_Block_Cipher_Modes.md) | **다음**: [RSA 암호체계](./05_RSA_Cryptosystem.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 암호학적 해시 함수(Cryptographic Hash Function)의 세 가지 보안 속성(원상 저항성, 제2 원상 저항성, 충돌 저항성)을 정의할 수 있다
2. 머클-담가드(Merkle-Damgard) 구성을 설명하고 SHA-256의 압축 함수를 통해 데이터 흐름을 추적할 수 있다
3. 생일 경계(Birthday Bound)를 유도하고 주어진 해시 출력 크기에 대한 충돌 확률을 추정할 수 있다
4. 길이 확장 공격(Length Extension Attack)을 설명하고, SHA-3의 스펀지(Sponge) 구성이 왜 이에 면역인지 설명할 수 있다
5. 해시 함수로부터 HMAC을 구현하고, 단순 키 접두사 해싱이 왜 안전하지 않은지 설명할 수 있다
6. 패스워드 해싱 알고리즘(bcrypt, scrypt, Argon2)을 비교하고 메모리 난이도(Memory-Hardness)를 설명할 수 있다

---

> **비유:** 해시 함수는 지문 채취기와 같습니다. 사람이 주어지면 고유한 지문을 생성하지만, 지문으로부터 사람을 재구성할 수는 없습니다. 두 사람이 같은 지문을 가져서는 안 되듯이, 두 메시지가 같은 해시를 생성해서는 안 됩니다. 손가락을 스캔하기는 쉽지만 지문으로부터 사람을 만드는 것이 불가능하듯이, 해싱은 빠르지만 역산은 불가능합니다.

해시 함수는 암호학 전반에서 등장합니다: 디지털 서명(레슨 7)에서는 서명 전에 메시지를 해싱하고, 패스워드 저장은 느린 해시 함수에 의존하며, 블록체인 무결성은 해시 체인에 달려 있고, TLS는 메시지 인증에 HMAC을 사용합니다. 해시 함수를 깊이 이해하는 것은 이후 모든 내용의 전제 지식입니다.

## 목차

1. [암호학적 해시 함수란?](#1-암호학적-해시-함수란)
2. [보안 속성](#2-보안-속성)
3. [생일 공격](#3-생일-공격)
4. [머클-담가드 구성](#4-머클-담가드-구성)
5. [SHA-256 내부 구조](#5-sha-256-내부-구조)
6. [길이 확장 공격](#6-길이-확장-공격)
7. [SHA-3와 스펀지 구성](#7-sha-3와-스펀지-구성)
8. [HMAC: 키 기반 해싱](#8-hmac-키-기반-해싱)
9. [패스워드 해싱](#9-패스워드-해싱)
10. [연습 문제](#10-연습-문제)

---

## 1. 암호학적 해시 함수란?

**암호학적 해시 함수** $H$는 임의 길이의 입력을 고정 길이의 출력으로 매핑합니다:

$$H: \{0, 1\}^* \rightarrow \{0, 1\}^n$$

SHA-256의 경우 $n = 256$ (32바이트)입니다. 출력은 **해시(Hash)**, **다이제스트(Digest)**, 또는 **지문(Fingerprint)**이라고 부릅니다.

### 1.1 기본 속성

| 속성 | 설명 |
|------|------|
| **결정론적** | 동일한 입력은 항상 동일한 출력을 생성 |
| **빠름** | $H(m)$ 계산이 효율적 |
| **고정 출력** | 입력 크기에 관계없이 출력은 항상 $n$ 비트 |
| **눈사태 효과(Avalanche Effect)** | 입력 1비트 변화 시 출력 비트의 ~50%가 뒤집힘 |

```python
import hashlib

def hash_demo():
    """Demonstrate basic hash function properties."""
    # Deterministic: same input → same output
    h1 = hashlib.sha256(b"Hello").hexdigest()
    h2 = hashlib.sha256(b"Hello").hexdigest()
    print(f"Deterministic: {h1 == h2}")  # True

    # Fixed output: any input → 256 bits (64 hex chars)
    for msg in [b"", b"Hi", b"A" * 10000]:
        h = hashlib.sha256(msg).hexdigest()
        print(f"  len(input)={len(msg):>5}, hash={h[:16]}... ({len(h)} hex chars)")

    # Avalanche: 1-bit difference → ~50% bits change
    h_a = int(hashlib.sha256(b"Hello").hexdigest(), 16)
    h_b = int(hashlib.sha256(b"Iello").hexdigest(), 16)
    diff_bits = bin(h_a ^ h_b).count('1')
    print(f"Avalanche: {diff_bits}/256 bits differ ({diff_bits/256*100:.1f}%)")

hash_demo()
```

---

## 2. 보안 속성

### 2.1 원상 저항성(Preimage Resistance) — 단방향성

해시 값 $h$가 주어졌을 때, $H(m) = h$를 만족하는 메시지 $m$을 찾는 것이 계산적으로 불가능합니다.

$$\text{Given } h, \text{ find } m \text{ such that } H(m) = h$$

**예상 노력:** $n$비트 해시에 대해 $O(2^n)$.

### 2.2 제2 원상 저항성(Second Preimage Resistance)

메시지 $m_1$이 주어졌을 때, $H(m_1) = H(m_2)$를 만족하는 다른 메시지 $m_2 \neq m_1$을 찾는 것이 계산적으로 불가능합니다.

$$\text{Given } m_1, \text{ find } m_2 \neq m_1 \text{ such that } H(m_1) = H(m_2)$$

**예상 노력:** $n$비트 해시에 대해 $O(2^n)$.

### 2.3 충돌 저항성(Collision Resistance)

$H(m_1) = H(m_2)$를 만족하는 **임의의** 서로 다른 두 메시지 $m_1 \neq m_2$를 찾는 것이 계산적으로 불가능합니다.

$$\text{Find any } m_1 \neq m_2 \text{ such that } H(m_1) = H(m_2)$$

**예상 노력:** $O(2^{n/2})$ — 원상 저항성보다 훨씬 적습니다! 이것이 **생일 경계(Birthday Bound)**(3절)입니다.

### 2.4 속성 간의 관계

```
Collision Resistance → Second Preimage Resistance → (weak) Preimage Resistance

If you can find collisions easily, you can find second preimages easily.
But second preimage resistance does NOT imply collision resistance.
```

| 해시 | 출력 비트 | 원상 | 제2 원상 | 충돌 |
|------|----------|------|----------|------|
| MD5 | 128 | $2^{128}$ (안전) | $2^{128}$ (안전) | **깨짐** ($2^{18}$ — 초 단위) |
| SHA-1 | 160 | $2^{160}$ (안전) | $2^{160}$ (안전) | **깨짐** ($2^{63}$) |
| SHA-256 | 256 | $2^{256}$ | $2^{256}$ | $2^{128}$ (안전) |
| SHA-3-256 | 256 | $2^{256}$ | $2^{256}$ | $2^{128}$ (안전) |

---

## 3. 생일 공격

### 3.1 생일 역설(Birthday Paradox)

23명이 있는 방에서 두 명이 같은 생일을 공유할 확률이 50% 이상입니다. $23 \ll 365$이므로 이는 놀라운 결과입니다.

일반 공식: $N$개의 값에서 $k$개의 무작위 샘플을 뽑을 때 충돌 확률은 다음과 같습니다:

$$P(\text{collision}) \approx 1 - e^{-k^2/(2N)}$$

$P = 0.5$로 설정하면 $k \approx \sqrt{N} \cdot 1.177$입니다.

### 3.2 해시 함수에 대한 시사점

$N = 2^n$개의 가능한 출력을 가진 $n$비트 해시 함수의 경우, 약 $2^{n/2}$번의 해싱 후 충돌이 예상됩니다.

| 해시 크기 | 충돌 발생 시점 | 보안 수준 |
|----------|--------------|-----------|
| 128비트 | $2^{64}$ | 안전하지 않음 (실현 가능) |
| 160비트 | $2^{80}$ | 경계선 |
| 256비트 | $2^{128}$ | 안전 |
| 512비트 | $2^{256}$ | 매우 안전 |

이것이 MD5(128비트)와 SHA-1(160비트)이 충돌 저항성 측면에서 깨졌다고 여겨지는 이유입니다.

### 3.3 생일 공격 시뮬레이션

```python
import random
import math

def birthday_attack_simulation(hash_bits=16, trials=1000):
    """Simulate the birthday attack on a small hash function.

    Why simulate: this demonstrates that finding collisions requires
    O(2^(n/2)) effort, NOT O(2^n). For a 256-bit hash, 2^128 is
    beyond reach, but for a 64-bit hash, 2^32 is trivially achievable.

    We use a small hash size (16 bits) so the simulation completes quickly.
    """
    hash_space = 2 ** hash_bits
    total_hashes = 0

    for _ in range(trials):
        seen = set()
        count = 0
        while True:
            h = random.randint(0, hash_space - 1)
            count += 1
            if h in seen:
                break
            seen.add(h)
        total_hashes += count

    avg = total_hashes / trials
    expected = math.sqrt(math.pi / 2 * hash_space)

    print(f"Hash size: {hash_bits} bits ({hash_space} values)")
    print(f"Average hashes to collision: {avg:.1f}")
    print(f"Theoretical (sqrt(pi/2 * N)):  {expected:.1f}")
    print(f"Ratio: {avg/expected:.3f}")

birthday_attack_simulation(hash_bits=16, trials=500)
# Expected: ~320 hashes to find a collision in a 16-bit hash
```

---

## 4. 머클-담가드 구성

### 4.1 동작 방식

가장 널리 사용되는 해시 함수들(MD5, SHA-1, SHA-256)은 **머클-담가드(Merkle-Damgard) 구성**을 따릅니다:

1. 메시지를 패딩하고 고정 크기의 블록 $M_1, M_2, \ldots, M_L$로 분할
2. 고정된 **초기화 벡터(IV)**로 시작
3. 각 블록을 **압축 함수(Compression Function)** $f$로 처리:

$$H_0 = IV$$
$$H_i = f(H_{i-1}, M_i) \quad \text{for } i = 1, \ldots, L$$
$$H(M) = H_L$$

```
M_1        M_2        M_3        M_L
 ↓          ↓          ↓          ↓
IV → [f] → h_1 → [f] → h_2 → [f] → ... → [f] → H(M)
```

### 4.2 동작 원리

**정리 (머클-담가드):** 압축 함수 $f$가 충돌 저항성을 가진다면, 그로부터 구축된 해시 함수 $H$도 충돌 저항성을 가집니다.

즉, 고정 입력 크기에 대한 안전한 **압축 함수**만 설계하면 되고, 머클-담가드 구성이 이를 임의 길이 입력으로 확장합니다.

```python
def merkle_damgard_demo(message, block_size=4):
    """Demonstrate the Merkle-Damgard construction with a toy compression function.

    Why Merkle-Damgard: it reduces the problem of building a hash function
    for arbitrary inputs to building a compression function for fixed inputs.
    This modular approach enabled decades of hash function design.
    """
    # Pad message to multiple of block_size
    # Append length (simplified: just pad with zeros + length byte)
    msg_len = len(message)
    padded = message + b'\x80'  # padding start marker
    while (len(padded) + 1) % block_size != 0:
        padded += b'\x00'
    padded += bytes([msg_len % 256])  # simplified length encoding

    # Split into blocks
    blocks = [padded[i:i+block_size] for i in range(0, len(padded), block_size)]

    # Toy compression function (NOT cryptographically secure!)
    def compress(state, block):
        """A toy compression function for illustration.
        Real compression functions (SHA-256) use bit rotations,
        additions, and Boolean functions over 32-bit words."""
        result = state
        for b in block:
            result = ((result * 31) + b) & 0xFFFFFFFF
        return result

    # Process blocks
    IV = 0x6A09E667  # SHA-256's actual first IV constant
    state = IV

    print(f"Message: {message}")
    print(f"Padded:  {padded.hex()} ({len(blocks)} blocks)")
    print(f"IV: 0x{state:08X}")

    for i, block in enumerate(blocks):
        state = compress(state, block)
        print(f"After block {i+1} ({block.hex()}): 0x{state:08X}")

    print(f"Final hash: 0x{state:08X}")
    return state

merkle_damgard_demo(b"Hello, World!")
```

---

## 5. SHA-256 내부 구조

### 5.1 개요

SHA-256은 512비트(64바이트)의 메시지 블록을 처리하고 256비트 상태(32비트 워드 8개)를 유지합니다.

| 파라미터 | 값 |
|---------|-----|
| 블록 크기 | 512비트 (64바이트) |
| 출력 크기 | 256비트 (32바이트) |
| 워드 크기 | 32비트 |
| 라운드 수 | 64 |
| 연산 | AND, OR, XOR, NOT, $2^{32}$ 모듈로 덧셈, 회전 |

### 5.2 메시지 패딩

1. `1` 비트 추가
2. 길이 $\equiv 448 \pmod{512}$가 될 때까지 `0` 비트 추가
3. 원본 메시지 길이를 64비트 빅엔디언 정수로 추가

### 5.3 메시지 스케줄

16워드(512비트) 블록을 64워드로 확장:

$$W_t = \begin{cases} M_t & \text{for } 0 \leq t \leq 15 \\ \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16} & \text{for } 16 \leq t \leq 63 \end{cases}$$

여기서:
$$\sigma_0(x) = \text{ROTR}^7(x) \oplus \text{ROTR}^{18}(x) \oplus \text{SHR}^3(x)$$
$$\sigma_1(x) = \text{ROTR}^{17}(x) \oplus \text{ROTR}^{19}(x) \oplus \text{SHR}^{10}(x)$$

### 5.4 압축 함수

각 라운드는 8개의 작업 변수 $a, b, c, d, e, f, g, h$를 다음과 같이 업데이트합니다:

$$T_1 = h + \Sigma_1(e) + Ch(e, f, g) + K_t + W_t$$
$$T_2 = \Sigma_0(a) + Maj(a, b, c)$$
$$h = g, \quad g = f, \quad f = e, \quad e = d + T_1$$
$$d = c, \quad c = b, \quad b = a, \quad a = T_1 + T_2$$

여기서:
$$\Sigma_0(a) = \text{ROTR}^2(a) \oplus \text{ROTR}^{13}(a) \oplus \text{ROTR}^{22}(a)$$
$$\Sigma_1(e) = \text{ROTR}^6(e) \oplus \text{ROTR}^{11}(e) \oplus \text{ROTR}^{25}(e)$$
$$Ch(e, f, g) = (e \wedge f) \oplus (\neg e \wedge g)$$
$$Maj(a, b, c) = (a \wedge b) \oplus (a \wedge c) \oplus (b \wedge c)$$

```python
def sha256_compression_demo():
    """Demonstrate one round of SHA-256 compression.

    Why these specific operations:
    - Ch (choice): e chooses between f and g bits — non-linear
    - Maj (majority): output bit is the majority of a, b, c — non-linear
    - Sigma functions: rotations + XOR spread bit influence (diffusion)
    - Addition mod 2^32: provides non-linearity (carries)
    - Round constants K_t: derived from cube roots of primes (nothing-up-my-sleeve numbers)
    """
    def rotr(x, n, w=32):
        """Right rotation of a w-bit word."""
        return ((x >> n) | (x << (w - n))) & ((1 << w) - 1)

    def ch(e, f, g):
        return (e & f) ^ (~e & g) & 0xFFFFFFFF

    def maj(a, b, c):
        return (a & b) ^ (a & c) ^ (b & c)

    def sigma0(a):
        return rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)

    def sigma1(e):
        return rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)

    # SHA-256 initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
    H = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ]

    # First 8 round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5
    ]

    a, b, c, d, e, f, g, h = H
    W_0 = 0x48656C6C  # "Hell" in ASCII (first word of "Hello...")

    print("Initial state:")
    print(f"  a={a:08x} b={b:08x} c={c:08x} d={d:08x}")
    print(f"  e={e:08x} f={f:08x} g={g:08x} h={h:08x}")

    # One round
    T1 = (h + sigma1(e) + ch(e, f, g) + K[0] + W_0) & 0xFFFFFFFF
    T2 = (sigma0(a) + maj(a, b, c)) & 0xFFFFFFFF

    h, g, f, e = g, f, e, (d + T1) & 0xFFFFFFFF
    d, c, b, a = c, b, a, (T1 + T2) & 0xFFFFFFFF

    print("\nAfter round 0:")
    print(f"  a={a:08x} b={b:08x} c={c:08x} d={d:08x}")
    print(f"  e={e:08x} f={f:08x} g={g:08x} h={h:08x}")

sha256_compression_demo()
```

### 5.5 왜 이런 상수를 사용하는가?

초기 해시 값은 $\sqrt{2}, \sqrt{3}, \sqrt{5}, \sqrt{7}, \sqrt{11}, \sqrt{13}, \sqrt{17}, \sqrt{19}$의 소수점 이하 부분입니다.

라운드 상수 $K_t$는 처음 64개의 소수 $p$에 대한 $\sqrt[3]{p}$의 소수점 이하 부분입니다.

이것들은 **nothing-up-my-sleeve 숫자**라고 불립니다. 잘 알려진 수학 상수로부터 유도되어, 상수에 숨겨진 백도어가 없음을 증명합니다.

---

## 6. 길이 확장 공격

### 6.1 취약점

머클-담가드 해시의 경우, $m$을 몰라도 $H(m)$과 $|m|$을 알면 다음을 계산할 수 있습니다:

$$H(m \| \text{padding} \| m')$$

**$m$을 모르고도** 임의의 접미사 $m'$에 대해 계산 가능합니다.

### 6.2 동작 방식

$H(m) = h_L$은 압축 함수의 최종 상태이므로, 공격자는:
1. $H(m)$을 시작 상태로 사용
2. 추가 블록 $m'$ 처리 계속
3. 결과는 $m \| \text{padding} \| m'$의 유효한 해시

```python
def length_extension_concept():
    """Explain the length extension attack.

    Why this is dangerous: naive MAC construction H(key || message) is
    vulnerable. An attacker who sees MAC = H(key || msg) can compute
    H(key || msg || padding || attacker_data) without knowing the key.

    Real-world impact: Flickr API signing (2009) used H(secret || params),
    allowing attackers to append additional API parameters.
    """
    import hashlib

    # Scenario: server computes MAC = SHA256(secret_key || message)
    secret_key = b"supersecret"
    message = b"amount=100&to=alice"

    mac = hashlib.sha256(secret_key + message).hexdigest()
    print(f"Original MAC: {mac}")
    print(f"Original msg: {message.decode()}")

    # Attacker wants to extend: amount=100&to=alice[padding]&to=eve
    # The attacker can compute the new MAC without knowing secret_key!
    # They use the original MAC as the internal state and continue hashing.

    print("\nLength extension attack:")
    print("Attacker can compute SHA256(key || msg || padding || &to=eve)")
    print("without knowing the key, using only the original MAC as state.")
    print("\nThis is why HMAC exists (Section 8).")

length_extension_concept()
```

### 6.3 영향받는 해시 함수

| 해시 | 길이 확장에 취약? |
|------|-----------------|
| MD5 | 예 |
| SHA-1 | 예 |
| SHA-256, SHA-512 | 예 |
| SHA-3 (Keccak) | **아니오** (스펀지 구성) |
| BLAKE2, BLAKE3 | **아니오** (다른 최종화 방식) |

---

## 7. SHA-3와 스펀지 구성

### 7.1 동기

SHA-1이 이론적으로 깨진 후(2005년), NIST는 SHA-3 경쟁을 시작했습니다. **Keccak**이 2012년에 우승하며, 근본적으로 다른 설계인 **스펀지(Sponge) 구성**을 도입했습니다.

### 7.2 스펀지 구성

스펀지는 $b = r + c$ 비트의 상태에서 동작하며:
- $r$ = **속도(Rate)** (단계당 흡수/짜내는 비트 수)
- $c$ = **용량(Capacity)** (보안을 제공하는 비트; 직접 노출되지 않음)

```
Absorbing phase:
M_1        M_2        M_3
 ↓          ↓          ↓
[0...0] → [f] → s_1 → [f] → s_2 → [f] → s_3
  ↑XOR      ↑XOR        ↑XOR
  M_1       M_2         M_3

Squeezing phase:
s_3 → [f] → output_1 → [f] → output_2 → ...
```

```python
def sponge_concept_demo():
    """Demonstrate the sponge construction concept.

    Why sponge > Merkle-Damgard:
    1. No length extension attack — the capacity bits are never output
    2. Variable output length — can produce any number of output bits
    3. Simpler security proof — security depends only on the permutation f
    4. Versatile — same construction for hash, MAC, PRNG, stream cipher
    """
    # Toy sponge with b=16, r=8, c=8 (real Keccak: b=1600, r=1088, c=512)
    r = 8  # rate (bits absorbed per step)
    c = 8  # capacity (security parameter)
    b = r + c  # total state size

    state = 0  # b-bit state, initially all zeros

    def permutation(state):
        """Toy permutation — Keccak uses 24 rounds of theta, rho, pi, chi, iota."""
        # Simple mixing (NOT cryptographic — for illustration only)
        state = ((state * 0xABCD) ^ (state >> 3) ^ (state << 5)) & ((1 << b) - 1)
        return state

    message_blocks = [0x48, 0x65, 0x6C]  # "Hel" as bytes

    # Absorbing phase
    print("=== Absorbing ===")
    for i, block in enumerate(message_blocks):
        # XOR message into the RATE portion of the state (not the capacity!)
        state ^= (block << c)  # Message goes into the rate (upper) bits
        state = permutation(state)
        print(f"After absorbing block {i} (0x{block:02x}): state = 0x{state:04x}")

    # Squeezing phase
    print("\n=== Squeezing ===")
    output = []
    for i in range(2):
        # Extract from the RATE portion only
        output_block = (state >> c) & ((1 << r) - 1)
        output.append(output_block)
        print(f"Squeeze {i}: 0x{output_block:02x}")
        state = permutation(state)

    print(f"\nHash output: {''.join(f'{b:02x}' for b in output)}")
    print("\nKey insight: capacity bits are NEVER directly output,")
    print("so the internal state cannot be reconstructed from the hash.")

sponge_concept_demo()
```

### 7.3 Keccak 세부사항

Keccak은 $5 \times 5 \times 64 = 1600$비트 상태와 순열 $f = \iota \circ \chi \circ \pi \circ \rho \circ \theta$의 24라운드를 사용합니다:

| 단계 | 연산 | 목적 |
|------|------|------|
| $\theta$ | 열 패리티 혼합 | 확산(Diffusion) |
| $\rho$ | 비트 회전 (레인마다 다름) | 확산 |
| $\pi$ | 레인 순열 | 확산 |
| $\chi$ | 비선형 매핑 ($a \oplus (\neg b \wedge c)$) | 혼돈(Confusion) |
| $\iota$ | 라운드 상수와 XOR | 대칭성 파괴 |

---

## 8. HMAC: 키 기반 해싱

### 8.1 왜 H(키 || 메시지)가 안되는가?

6절에서 보여준 것처럼, $H(\text{key} \| \text{message})$는 머클-담가드 해시에 대한 길이 확장 공격에 취약합니다.

### 8.2 HMAC 구성

**HMAC**(Hash-based Message Authentication Code, RFC 2104):

$$\text{HMAC}(K, m) = H\bigl((K' \oplus \text{opad}) \| H((K' \oplus \text{ipad}) \| m)\bigr)$$

여기서:
- $K'$ = 해시 블록 크기로 패딩된 키 (또는 $K$가 블록 크기보다 크면 $H(K)$)
- $\text{ipad}$ = `0x36` 반복
- $\text{opad}$ = `0x5C` 반복

```python
import hashlib
import hmac as hmac_lib

def hmac_from_scratch(key, message, hash_func=hashlib.sha256, block_size=64):
    """HMAC implementation from scratch.

    Why two hash applications:
    - Inner hash: H(K XOR ipad || message) — prevents length extension
    - Outer hash: H(K XOR opad || inner_hash) — prevents key recovery
      even if the inner hash has weaknesses

    Why XOR with ipad/opad: ensures the key is used differently in
    the inner and outer hash calls, preventing related-key attacks.
    """
    # If key is longer than block size, hash it first
    if len(key) > block_size:
        key = hash_func(key).digest()

    # Pad key to block_size
    key = key + b'\x00' * (block_size - len(key))

    # Inner and outer padded keys
    o_key_pad = bytes(k ^ 0x5C for k in key)  # K XOR opad
    i_key_pad = bytes(k ^ 0x36 for k in key)  # K XOR ipad

    # HMAC = H(o_key_pad || H(i_key_pad || message))
    inner_hash = hash_func(i_key_pad + message).digest()
    outer_hash = hash_func(o_key_pad + inner_hash).hexdigest()

    return outer_hash

# Verify against Python's hmac library
key = b"my_secret_key"
message = b"Hello, HMAC!"

our_hmac = hmac_from_scratch(key, message)
lib_hmac = hmac_lib.new(key, message, hashlib.sha256).hexdigest()

print(f"Our HMAC:     {our_hmac}")
print(f"Library HMAC: {lib_hmac}")
print(f"Match: {our_hmac == lib_hmac}")
```

### 8.3 HMAC 보안

HMAC은 기반 해시 함수에 일부 약점이 있어도 안전합니다:
- 길이 확장에 내성 (설계상)
- 보안 증명: 압축 함수가 PRF(의사 난수 함수)라면 HMAC도 PRF
- MD5-HMAC도 여전히 안전하다고 여겨짐 (MD5 충돌 공격에도 불구하고)

---

## 9. 패스워드 해싱

### 9.1 왜 패스워드에 SHA-256을 사용하면 안 되는가?

SHA-256은 **너무 빠릅니다**. 현대 GPU는 초당 수십억 번의 SHA-256 해시를 계산할 수 있어, 패스워드 데이터베이스에 대한 빠른 무차별 대입 또는 사전 공격이 가능합니다.

| 공격 | SHA-256 속도 | 8자 패스워드 소요 시간 |
|------|------------|---------------------|
| CPU (단일) | ~20M/s | 수 시간 |
| GPU (RTX 4090) | ~10B/s | 수 초 |
| ASIC (비트코인 채굴기) | ~100T/s | 마이크로초 |

### 9.2 패스워드 해싱 요구사항

좋은 패스워드 해시 함수는 다음을 만족해야 합니다:
1. **느림** — 의도적인 계산 비용 (조정 가능한 작업 인수)
2. **메모리 난이도(Memory-Hard)** — 상당한 메모리 필요 (GPU/ASIC 공격 저항)
3. **솔트(Salt)** — 패스워드마다 고유한 무작위 솔트 (레인보우 테이블 방지)

### 9.3 bcrypt

- **기반:** Blowfish 암호 (Provos & Mazieres, 1999)
- **파라미터:** 비용 인수 (지수적: cost=12는 $2^{12}$ 라운드)
- **메모리:** ~4 KB (작음, 주요 강점은 CPU 비용)
- **출력 형식:** `$2b$12$salt22chars.hash31chars`

### 9.4 scrypt

- **기반:** PBKDF2 + 순차 메모리 난이도 함수 (Percival, 2009)
- **파라미터:** N (CPU/메모리 비용), r (블록 크기), p (병렬성)
- **메모리:** 설정 가능 (기본 ~16 MB)
- **bcrypt 대비 장점:** 메모리 난이도로 GPU 공격 억제

### 9.5 Argon2

**패스워드 해싱 경쟁(2015) 우승작.** 세 가지 변형:

| 변형 | 최적화 대상 | 사용 사례 |
|------|-----------|---------|
| Argon2d | GPU 저항 | 암호화폐 채굴 |
| Argon2i | 사이드채널 저항 | 공유 서버에서의 패스워드 해싱 |
| Argon2id | 두 가지 모두 (하이브리드) | **권장 기본값** |

```python
def password_hashing_comparison():
    """Compare password hashing approaches.

    Why Argon2id is recommended (OWASP 2024):
    1. Tunable memory, time, and parallelism parameters
    2. Resists GPU, FPGA, and ASIC attacks via memory-hardness
    3. Winner of an open, peer-reviewed competition
    4. Hybrid design (Argon2id) resists both side-channel and GPU attacks
    """
    import hashlib
    import os
    import time

    password = b"correct horse battery staple"
    salt = os.urandom(16)

    # Naive: SHA-256 (TOO FAST for passwords)
    start = time.time()
    for _ in range(100_000):
        hashlib.sha256(salt + password).digest()
    sha_time = time.time() - start
    print(f"SHA-256 x100000: {sha_time:.3f}s ({100_000/sha_time:.0f} hashes/s)")

    # Better: PBKDF2 (iterated hashing)
    start = time.time()
    hashlib.pbkdf2_hmac('sha256', password, salt, iterations=600_000)
    pbkdf2_time = time.time() - start
    print(f"PBKDF2 (600k iterations): {pbkdf2_time:.3f}s")

    # Best: Argon2 (requires `pip install argon2-cffi`)
    # from argon2 import PasswordHasher
    # ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
    # hash = ph.hash(password.decode())

    print("\nRecommended settings (OWASP 2024):")
    print("  Argon2id: time=3, memory=64MB, parallelism=4")
    print("  PBKDF2-SHA256: 600,000 iterations (if Argon2 unavailable)")
    print("  bcrypt: cost factor 12+")

password_hashing_comparison()
```

### 9.6 솔트(Salt)와 페퍼(Pepper)

| 개념 | 저장 위치 | 목적 |
|------|----------|------|
| **솔트** | 해시와 함께 저장 | 레인보우 테이블 방지; 사용자마다 고유 |
| **페퍼** | 별도 저장 (환경 변수, HSM) | 비밀 추가; DB 유출에 대한 보호 |

---

## 10. 연습 문제

### 연습 문제 1: 해시 속성 (기본)

1. "Hello"와 "Hello " (끝에 공백 포함)의 SHA-256을 계산하세요. 몇 비트가 다른가요?
2. SHA-256이 "0000"으로 시작하는 메시지(4개의 16진수 제로 문자 = 16개의 제로 비트)를 찾으세요. 몇 번의 시도가 필요했나요?
3. (2)번을 비트코인 채굴 난이도와 연관 지어 설명하세요.

### 연습 문제 2: 생일 공격 (중급)

1. 잘린 해시에 대한 생일 공격을 구현하세요: SHA-256의 첫 4바이트(32비트)만 사용하세요.
2. 충돌을 찾으세요. 몇 번의 해싱이 필요했나요?
3. 이론적인 $\sqrt{2^{32}} \approx 65536$과 비교하세요.

### 연습 문제 3: HMAC 검증 (중급)

1. `hashlib.sha256`만 사용해 HMAC-SHA256을 처음부터 구현하세요.
2. 10개의 다른 키/메시지 쌍에 대해 `hmac.new(key, msg, hashlib.sha256)`와 비교하여 테스트하세요.
3. 같은 입력에 대해 H(키 || 메시지)와 HMAC(키, 메시지)가 다른 결과를 생성함을 시연하세요.

### 연습 문제 4: 길이 확장 공격 (심화)

1. `SHA-256("secret" || "data") = <해시>`와 `len("secret") = 6`이 주어졌을 때, "secret"을 모르고 `SHA-256("secret" || "data" || padding || "evil")`을 계산하세요.
2. `hashpumpy` 또는 `hlextend` 라이브러리를 사용해 결과를 검증하세요.
3. HMAC이 이 공격에 면역임을 보이세요.

### 연습 문제 5: 패스워드 크래킹 (심화)

1. 1000개의 일반 패스워드를 SHA-256으로 해싱하세요 (솔트 없이). 해시가 주어졌을 때 일치 항목을 찾는 데 얼마나 걸리는지 측정하세요.
2. PBKDF2 (100,000회 반복)로 반복하세요. 시간을 비교하세요.
3. 무작위 16바이트 솔트를 추가하세요. 사전 계산된 레인보우 테이블이 이제 왜 쓸모없는지 설명하세요.
4. 계산하세요: 64MB 메모리를 사용하는 Argon2id에서 공격자가 $10^6$개의 패스워드를 병렬로 시도하려면 총 얼마나 많은 RAM이 필요한가요?

---

**이전**: [블록 암호 모드](./03_Block_Cipher_Modes.md) | **다음**: [RSA 암호체계](./05_RSA_Cryptosystem.md)

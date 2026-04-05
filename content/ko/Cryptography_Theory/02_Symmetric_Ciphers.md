# 레슨 2: 대칭 암호

**이전**: [수론 기초](./01_Number_Theory_Foundations.md) | **다음**: [블록 암호 모드](./03_Block_Cipher_Modes.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 케르크호프스 원칙(Kerckhoffs' Principle)과 암호 설계에 대한 함의를 설명할 수 있습니다
2. 파이스텔 네트워크(Feistel Network) 구조를 설명하고 단순화된 파이스텔 암호를 구현할 수 있습니다
3. 샘플 데이터에 대해 네 가지 AES 연산(SubBytes, ShiftRows, MixColumns, AddRoundKey)을 추적할 수 있습니다
4. 혼돈(Confusion)과 확산(Diffusion)이 블록 암호의 보안을 어떻게 제공하는지 설명할 수 있습니다
5. DES와 AES를 구조, 키 크기, 보안 측면에서 비교할 수 있습니다
6. 대칭 암호에 대한 일반적인 공격(브루트 포스, 차분 분석, 선형 분석)을 식별할 수 있습니다

---

대칭 암호(Symmetric Cipher)는 현대 암호학의 일꾼입니다. 동영상을 스트리밍하거나, 메시지를 보내거나, 은행에 접속할 때, 대량의 데이터 암호화는 거의 항상 AES와 같은 대칭 암호가 처리합니다 — 대용량 데이터에는 너무 느린 RSA나 ECC가 아닙니다. AES의 내부 동작 방식을 이해하면 왜 25년 이상의 암호 분석 검증을 견뎌냈는지, 그리고 DES와 같은 초기 암호들이 왜 기술 발전에 무릎을 꿇었는지 알 수 있습니다.

## 목차

1. [역사적 암호들](#1-역사적-암호들)
2. [케르크호프스 원칙](#2-케르크호프스-원칙)
3. [혼돈과 확산](#3-혼돈과-확산)
4. [파이스텔 네트워크](#4-파이스텔-네트워크)
5. [DES: 데이터 암호화 표준](#5-des-데이터-암호화-표준)
6. [AES: 고급 암호화 표준](#6-aes-고급-암호화-표준)
7. [AES 내부 동작 단계별 설명](#7-aes-내부-동작-단계별-설명)
8. [보안 분석](#8-보안-분석)
9. [연습 문제](#9-연습-문제)

---

## 1. 역사적 암호들

### 1.1 카이사르 암호(Caesar Cipher)

가장 단순한 치환 암호: 각 문자를 고정된 양만큼 이동합니다.

$$E(x) = (x + k) \bmod 26$$
$$D(y) = (y - k) \bmod 26$$

```python
def caesar_encrypt(plaintext, key):
    """Caesar cipher encryption.

    Why this is insecure: only 26 possible keys. An attacker can
    try all 26 shifts in microseconds (brute force). Additionally,
    letter frequency analysis instantly reveals the key.
    """
    result = []
    for char in plaintext.upper():
        if char.isalpha():
            shifted = (ord(char) - ord('A') + key) % 26
            result.append(chr(shifted + ord('A')))
        else:
            result.append(char)
    return ''.join(result)

def caesar_decrypt(ciphertext, key):
    return caesar_encrypt(ciphertext, -key)

# Example
plaintext = "HELLO WORLD"
key = 3
ct = caesar_encrypt(plaintext, key)
pt = caesar_decrypt(ct, key)
print(f"Plaintext:  {plaintext}")
print(f"Ciphertext: {ct}")   # KHOOR ZRUOG
print(f"Decrypted:  {pt}")   # HELLO WORLD
```

### 1.2 비즈네르 암호(Vigenere Cipher)

키워드를 사용하여 각 위치의 이동량을 결정하는 다중 치환 암호(Polyalphabetic Cipher):

```python
def vigenere_encrypt(plaintext, key):
    """Vigenere cipher — polyalphabetic substitution.

    Why it was called 'le chiffre indechiffrable': unlike Caesar,
    the same letter encrypts to different ciphertext letters depending
    on its position. This defeats simple frequency analysis.

    Why it's still insecure: Kasiski examination can determine the
    key length, after which each position becomes an independent
    Caesar cipher vulnerable to frequency analysis.
    """
    result = []
    key = key.upper()
    key_index = 0
    for char in plaintext.upper():
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            encrypted = (ord(char) - ord('A') + shift) % 26
            result.append(chr(encrypted + ord('A')))
            key_index += 1
        else:
            result.append(char)
    return ''.join(result)

ct = vigenere_encrypt("ATTACK AT DAWN", "LEMON")
print(f"Ciphertext: {ct}")  # LXFOPV EF RNHR
```

### 1.3 역사에서 얻은 교훈

| 암호 | 키 공간 | 공격 방법 | 교훈 |
|------|---------|-----------|------|
| 카이사르 | 26 | 브루트 포스(Brute Force) | 키 공간이 너무 작음 |
| 단일 치환(Substitution) | $26!$ | 빈도 분석(Frequency Analysis) | 문자 패턴 보존은 치명적 |
| 비즈네르 | $26^k$ | 카시스키 검사(Kasiski) + 빈도 분석 | 반복 키가 패턴 생성 |
| 일회용 패드(One-Time Pad) | $26^n$ ($n$ = 메시지 길이) | **해독 불가** (올바르게 사용할 경우) | 완전 비밀성은 메시지만큼 긴 키 요구 |

**일회용 패드(One-Time Pad)**는 완전 비밀성(Perfect Secrecy)을 달성합니다(Shannon, 1949). 그러나 키가 메시지만큼 길어야 하고 재사용할 수 없어 실용적이지 않습니다. 현대 암호들은 **계산적 보안(Computational Security)**을 목표로 합니다: 현실적인 계산 자원으로 해독이 불가능한 수준.

---

## 2. 케르크호프스 원칙

> **케르크호프스 원칙(Kerckhoffs' Principle, 1883):** 암호 시스템은 키를 제외한 모든 것이 공개되어도 안전해야 합니다.

현대적 표현: **암호의 보안은 알고리즘의 비밀이 아닌, 오직 키의 비밀에만 의존해야 합니다.**

**왜 중요한가:**
- 알고리즘은 역공학(Reverse Engineering)되거나, 유출되거나, 공개될 수 있습니다
- 공개된 알고리즘은 더 많은 검증을 받으므로 더 신뢰할 수 있습니다
- 비밀 알고리즘은 잘못된 보안감("보안을 통한 불명확성", Security Through Obscurity)을 만들어냅니다
- AES, ChaCha20, 그리고 모든 현대 암호들은 완전히 공개된 사양을 가집니다

**역사적 관행과의 대조:** 많은 역사적 암호들은 알고리즘을 비밀로 유지하는 데 의존했습니다. 알고리즘이 발견되면 암호는 즉시 해독되었습니다.

---

## 3. 혼돈과 확산

클로드 섀넌(Claude Shannon, 1949)은 안전한 암호가 반드시 갖추어야 할 두 가지 근본적 성질을 규명했습니다:

### 3.1 혼돈(Confusion)

**혼돈**은 키와 암호문 사이의 관계를 최대한 복잡하게 만듭니다. 각 암호문 비트는 키의 여러 부분에 의존해야 합니다.

- **구현 방법:** 치환 연산(AES의 S-박스)
- **목표:** 일부 암호문을 아는 공격자가 키에 대한 정보를 추론할 수 없게 함

### 3.2 확산(Diffusion)

**확산**은 각 평문 비트의 영향을 많은 암호문 비트에 퍼뜨립니다. 평문 비트 하나를 변경하면 암호문 비트의 약 절반이 변경되어야 합니다.

- **구현 방법:** 순열 및 혼합 연산(AES의 ShiftRows, MixColumns)
- **목표:** 평문의 중복성을 암호문 전체에 분산

> **비유:** 혼돈은 레시피를 뒤죽박죽으로 만드는 것(어떤 재료가 사용되었는지 파악하기 어렵게)이고, 확산은 반죽을 완전히 저어 주는 것(각 재료를 혼합물 전체에 퍼뜨리는 것)이라고 생각하세요. 좋은 암호는 이 두 가지를 반복적으로 수행합니다.

```python
def avalanche_demo():
    """Demonstrate the avalanche effect — a hallmark of good diffusion.

    A 1-bit change in input should cause ~50% of output bits to flip.
    """
    from hashlib import sha256

    msg1 = b"Hello"
    msg2 = b"Iello"  # One bit difference ('H' vs 'I' differ by 1 bit)

    h1 = int(sha256(msg1).hexdigest(), 16)
    h2 = int(sha256(msg2).hexdigest(), 16)

    # Count differing bits
    diff = bin(h1 ^ h2).count('1')
    total = 256  # SHA-256 output size

    print(f"Bits changed: {diff}/{total} ({diff/total*100:.1f}%)")
    # Expect approximately 128 bits (50%) to differ

avalanche_demo()
```

---

## 4. 파이스텔 네트워크

### 4.1 구조

**파이스텔 네트워크(Feistel Network)**는 블록 암호를 구성하는 일반적인 방법입니다. 블록을 두 절반으로 나누고 여러 라운드를 적용합니다:

```
Input: (L_0, R_0)

Round i:
    L_i = R_{i-1}
    R_i = L_{i-1} XOR F(R_{i-1}, K_i)

Output: (L_n, R_n)
```

여기서 $F$는 **라운드 함수(Round Function)**이고, $K_i$는 마스터 키에서 키 스케줄(Key Schedule)을 통해 파생된 **라운드 키(Round Key)**입니다.

> **비유:** 파이스텔 네트워크는 조립 라인 위의 일련의 믹서들과 같습니다. 각 믹서는 오른쪽 절반을 취해 비밀 재료(라운드 키)로 변환하고, 그 결과를 왼쪽 절반에 혼합합니다. 그런 다음 다음 믹서를 위해 두 절반이 교환됩니다. 충분한 라운드 이후에는 원본 데이터가 완전히 뒤섞입니다.

### 4.2 핵심 성질: 역함수 가능성(Invertibility)

파이스텔 구조는 라운드 함수 $F$에 관계없이 **항상 역함수 가능**합니다. 복호화하려면 단순히 라운드 순서를 역전시키면 됩니다:

$$R_{i-1} = L_i$$
$$L_{i-1} = R_i \oplus F(L_i, K_i)$$

이는 $F$가 역함수 가능할 필요가 없음을 의미합니다 — 해시 함수를 포함한 어떤 함수든 될 수 있습니다. 이것은 강력한 설계 장점입니다.

### 4.3 구현

```python
def feistel_round(left, right, round_key, f):
    """One round of a Feistel cipher.

    Why XOR: XOR is its own inverse (a XOR b XOR b = a), which
    makes decryption structurally identical to encryption — just
    reverse the round key order. No separate decryption logic needed.
    """
    new_left = right
    new_right = left ^ f(right, round_key)
    return new_left, new_right

def simple_round_function(half_block, round_key):
    """A toy round function for demonstration.

    In real ciphers (DES), this involves S-boxes, permutations,
    and expansion. Here we use a simple mixing function.
    """
    # Rotate, XOR with key, and add non-linearity
    mixed = ((half_block * 31337) ^ round_key) & 0xFFFFFFFF
    return mixed

def feistel_encrypt(plaintext_block, round_keys):
    """Encrypt a 64-bit block using a Feistel network.

    Why 16 rounds: security analysis shows that fewer rounds leave
    detectable statistical patterns. DES uses 16 rounds; research
    showed that 15 rounds of DES have a detectable weakness.
    """
    left = (plaintext_block >> 32) & 0xFFFFFFFF
    right = plaintext_block & 0xFFFFFFFF

    for key in round_keys:
        left, right = feistel_round(left, right, key, simple_round_function)

    # Swap halves at the end (standard convention)
    return (right << 32) | left

def feistel_decrypt(ciphertext_block, round_keys):
    """Decrypt: same structure, reversed key order.

    Why same structure: this is the beauty of Feistel networks —
    encryption and decryption use identical hardware/code, just
    with reversed key schedule. This halves implementation cost.
    """
    left = (ciphertext_block >> 32) & 0xFFFFFFFF
    right = ciphertext_block & 0xFFFFFFFF

    # Reverse round keys for decryption
    for key in reversed(round_keys):
        left, right = feistel_round(left, right, key, simple_round_function)

    return (right << 32) | left

# Demonstrate encryption and decryption
import random
round_keys = [random.getrandbits(32) for _ in range(16)]
plaintext = 0xDEADBEEFCAFEBABE

ciphertext = feistel_encrypt(plaintext, round_keys)
decrypted = feistel_decrypt(ciphertext, round_keys)

print(f"Plaintext:  0x{plaintext:016X}")
print(f"Ciphertext: 0x{ciphertext:016X}")
print(f"Decrypted:  0x{decrypted:016X}")
print(f"Match: {plaintext == decrypted}")
```

---

## 5. DES: 데이터 암호화 표준

### 5.1 개요

DES(Data Encryption Standard)는 1977년에 연방 표준으로 발표되었습니다(IBM이 NSA의 수정과 함께 설계).

| 매개변수 | 값 |
|----------|-----|
| 블록 크기 | 64비트 |
| 키 크기 | 56비트 (패리티 포함 시 64비트) |
| 라운드 수 | 16 |
| 구조 | 파이스텔 네트워크 |

### 5.2 DES 라운드 함수

각 DES 라운드는 다음을 적용합니다:
1. **확장(Expansion, E):** 32비트 절반을 48비트로 확장
2. **키 혼합(Key Mixing):** 48비트 라운드 키와 XOR
3. **S-박스 치환(S-Box Substitution):** 8개의 6비트→4비트 S-박스(비선형, 혼돈 제공)
4. **순열(Permutation, P):** 고정 비트 순열(확산 제공)

### 5.3 DES가 쇠퇴한 이유

- **56비트 키:** 1998년 EFF의 "Deep Crack" 기계가 25만 달러로 56시간 만에 DES 키를 브루트 포스했습니다. 오늘날에는 일반 하드웨어로 몇 시간이면 가능합니다.
- **3DES(Triple DES):** 삼중 DES($E_{K_1}(D_{K_2}(E_{K_1}(m)))$)는 유효 키 길이를 112비트로 확장했지만 3배 느립니다.
- DES는 2005년 NIST에 의해 공식적으로 폐기되고 AES로 대체되었습니다.

### 5.4 NSA 논란

NSA는 IBM의 원래 S-박스 설계를 수정했습니다. 이는 음모론을 불러일으켰지만, 나중에 NSA가 실제로 차분 암호 분석(Differential Cryptanalysis)에 대해 S-박스를 **강화**했다는 사실이 밝혀졌습니다 — 이 기법은 1990년까지 공개적으로 알려지지 않았지만 NSA는 이미 내부적으로 발견해 두었습니다.

---

## 6. AES: 고급 암호화 표준

### 6.1 역사

1997년 NIST는 DES를 대체하기 위한 공개 공모를 시작했습니다. 3년에 걸쳐 15개의 후보를 평가한 끝에, **레인달(Rijndael)**(Joan Daemen과 Vincent Rijmen 작성)이 2001년에 선정되었습니다.

| 매개변수 | AES-128 | AES-192 | AES-256 |
|----------|---------|---------|---------|
| 블록 크기 | 128비트 | 128비트 | 128비트 |
| 키 크기 | 128비트 | 192비트 | 256비트 |
| 라운드 수 | 10 | 12 | 14 |

### 6.2 AES는 파이스텔 네트워크가 아닙니다

DES와 달리, AES는 **치환-순열 네트워크(Substitution-Permutation Network, SPN)**를 사용합니다. 블록의 모든 비트가 매 라운드에서 변환됩니다(파이스텔에서는 라운드당 블록의 절반만 수정). 이 덕분에 AES는 더 적은 라운드에서 완전한 확산을 달성합니다.

### 6.3 상태 행렬(State Matrix)

AES는 **상태(State)**라 불리는 $4 \times 4$ 바이트 행렬(128비트 = 16바이트)에 대해 동작합니다:

$$\begin{bmatrix} s_{0,0} & s_{0,1} & s_{0,2} & s_{0,3} \\ s_{1,0} & s_{1,1} & s_{1,2} & s_{1,3} \\ s_{2,0} & s_{2,1} & s_{2,2} & s_{2,3} \\ s_{3,0} & s_{3,1} & s_{3,2} & s_{3,3} \end{bmatrix}$$

바이트는 **열 우선(Column-Major)** 방식으로 채워집니다: 평문의 첫 4바이트가 첫 번째 열을 채웁니다.

> **비유:** AES는 일련의 산업용 믹서와 같습니다 — 각 라운드는 데이터를 더욱 철저히 혼합합니다. SubBytes는 각 재료의 색깔을 바꾸고, ShiftRows는 컨베이어 벨트 위에서 재배치하며, MixColumns는 각 열 내의 재료들을 혼합하고, AddRoundKey는 비밀 키로 혼합물에 양념을 더합니다.

---

## 7. AES 내부 동작 단계별 설명

각 AES 라운드는 순서대로 네 가지 연산을 적용합니다. 마지막 라운드는 MixColumns를 생략합니다.

### 7.1 SubBytes (치환)

각 바이트는 고정된 **S-박스(S-Box)** — $16 \times 16$ 룩업 테이블을 사용하여 교체됩니다.

S-박스는 수학적으로 구성됩니다:
1. $GF(2^8)$(256개 원소를 가진 유한체)에서 바이트의 곱셈 역원을 계산
2. $GF(2)$ 위에서 아핀 변환(Affine Transformation) 적용

```python
# AES S-box (first 4 rows shown; full table is 256 entries)
AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]

def sub_bytes(state):
    """AES SubBytes: replace each byte using the S-box.

    Why the S-box uses GF(2^8) inverses: the multiplicative inverse
    in a finite field is highly non-linear, which provides maximum
    confusion. The affine transform prevents fixed points (no byte
    maps to itself) and adds algebraic complexity.
    """
    return [[AES_SBOX[byte] for byte in row] for row in state]

# Example: SubBytes on a single byte
print(f"SubBytes(0x00) = 0x{AES_SBOX[0x00]:02x}")  # 0x63
print(f"SubBytes(0x53) = 0x{AES_SBOX[0x53]:02x}")  # 0xed
```

**SubBytes가 혼돈을 제공하는 이유:** S-박스는 고정점(Fixed Point)이 없고($S(x) \neq x$, 모든 $x$에 대해), 반대 고정점도 없으며($S(x) \neq \overline{x}$), 높은 비선형성(Non-Linearity)을 가집니다. 이것이 AES를 선형 및 차분 암호 분석에 강하게 만드는 비선형 요소를 제공합니다.

### 7.2 ShiftRows (순열)

상태 행렬의 각 행이 행 인덱스만큼 좌측으로 순환 이동됩니다:

- 0행: 이동 없음
- 1행: 1칸 좌측 이동
- 2행: 2칸 좌측 이동
- 3행: 3칸 좌측 이동

```python
def shift_rows(state):
    """AES ShiftRows: cyclically shift each row.

    Why different shift amounts per row: this ensures that bytes
    from different columns are mixed in the MixColumns step.
    Without ShiftRows, each column would be processed independently,
    and there would be no inter-column diffusion.
    """
    shifted = [row[:] for row in state]  # Deep copy
    for i in range(4):
        shifted[i] = state[i][i:] + state[i][:i]  # Left rotate by i
    return shifted

# Visual example
state = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15]
]
result = shift_rows(state)
for row in result:
    print(row)
# [0, 1, 2, 3]      — no shift
# [5, 6, 7, 4]      — shift left 1
# [10, 11, 8, 9]    — shift left 2
# [15, 12, 13, 14]  — shift left 3
```

### 7.3 MixColumns (확산)

각 열은 $GF(2^8)$ 위의 다항식으로 취급되어 고정된 다항식과 곱해집니다:

$$c(x) = 3x^3 + x^2 + x + 2$$

행렬 형식으로:

$$\begin{bmatrix} 2 & 3 & 1 & 1 \\ 1 & 2 & 3 & 1 \\ 1 & 1 & 2 & 3 \\ 3 & 1 & 1 & 2 \end{bmatrix} \begin{bmatrix} s_{0,j} \\ s_{1,j} \\ s_{2,j} \\ s_{3,j} \end{bmatrix}$$

모든 산술 연산은 기약 다항식(Irreducible Polynomial) $x^8 + x^4 + x^3 + x + 1$을 사용하는 $GF(2^8)$에서 수행됩니다.

```python
def gf_multiply(a, b):
    """Multiply two bytes in GF(2^8).

    Why GF(2^8): this finite field has exactly 256 elements (one per
    byte value), making it natural for byte-oriented operations.
    The irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11B)
    was chosen because it is the lexicographically first irreducible
    polynomial of degree 8 — a convention, not a security requirement.
    """
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        carry = a & 0x80  # Check if high bit is set
        a = (a << 1) & 0xFF
        if carry:
            a ^= 0x1B  # Reduce modulo x^8 + x^4 + x^3 + x + 1
        b >>= 1
    return result

def mix_columns(state):
    """AES MixColumns: mix bytes within each column.

    Why this specific matrix: the coefficients {1, 2, 3} are the
    simplest values that make the matrix invertible over GF(2^8)
    and provide maximum diffusion (every output byte depends on
    all four input bytes). The matrix is an MDS (Maximum Distance
    Separable) code.
    """
    MIX_MATRIX = [
        [2, 3, 1, 1],
        [1, 2, 3, 1],
        [1, 1, 2, 3],
        [3, 1, 1, 2]
    ]

    new_state = [[0]*4 for _ in range(4)]
    for col in range(4):
        for row in range(4):
            val = 0
            for k in range(4):
                val ^= gf_multiply(MIX_MATRIX[row][k], state[k][col])
            new_state[row][col] = val
    return new_state
```

**MixColumns가 확산을 제공하는 이유:** MixColumns 이후에 열의 각 바이트는 해당 열의 4개 바이트 모두에 의존합니다. ShiftRows(열 간에 바이트를 이동)와 결합하면, 단 두 라운드 만에 모든 출력 바이트가 모든 입력 바이트에 의존하게 됩니다.

### 7.4 AddRoundKey

키 스케줄을 통해 마스터 키에서 파생된 라운드 키와 상태를 XOR합니다:

```python
def add_round_key(state, round_key):
    """AES AddRoundKey: XOR state with the round key.

    Why XOR and not addition: XOR is its own inverse, is extremely
    fast in hardware (single gate per bit), and does not introduce
    carries between bit positions. It is the simplest way to inject
    key material into the state.
    """
    return [
        [state[row][col] ^ round_key[row][col] for col in range(4)]
        for row in range(4)
    ]
```

### 7.5 키 스케줄(Key Schedule)

AES 키 스케줄은 128비트 마스터 키를 11개의 라운드 키(초기 1개 + 10 라운드)로 확장합니다:

```python
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

def key_expansion(key_bytes):
    """AES-128 key expansion: 16 bytes → 176 bytes (11 round keys).

    Why key expansion is needed: using the same key for every round
    would allow round-by-round attacks. The key schedule ensures that
    each round key is different and that a change in the master key
    cascades to all round keys (avalanche in the key space).
    """
    # Convert 16 bytes to 4 words (columns)
    w = []
    for i in range(4):
        w.append(key_bytes[4*i : 4*i+4])

    for i in range(4, 44):  # 44 words for AES-128
        temp = list(w[i-1])
        if i % 4 == 0:
            # RotWord + SubWord + Rcon
            temp = temp[1:] + temp[:1]  # Rotate left
            temp = [AES_SBOX[b] for b in temp]  # SubBytes
            temp[0] ^= RCON[i // 4 - 1]
        w.append(bytes(temp))

    # Group into round keys (4 words = 16 bytes = one round key)
    round_keys = []
    for r in range(11):
        rk = [[0]*4 for _ in range(4)]
        for col in range(4):
            for row in range(4):
                rk[row][col] = w[r*4 + col][row]
        round_keys.append(rk)

    return round_keys
```

### 7.6 전체 조합

```python
def aes_encrypt_block(plaintext_bytes, key_bytes):
    """AES-128 encryption of a single 16-byte block.

    Round structure:
    - Initial round:     AddRoundKey
    - Rounds 1-9:       SubBytes → ShiftRows → MixColumns → AddRoundKey
    - Final round (10): SubBytes → ShiftRows → AddRoundKey (no MixColumns)

    Why no MixColumns in the last round: omitting it makes the
    structure more symmetric with decryption, and adding it would
    not improve security (it would be followed only by AddRoundKey,
    which an attacker could absorb into the key).
    """
    # Convert bytes to 4x4 state matrix (column-major)
    state = [[0]*4 for _ in range(4)]
    for col in range(4):
        for row in range(4):
            state[row][col] = plaintext_bytes[col * 4 + row]

    round_keys = key_expansion(key_bytes)

    # Initial round key addition
    state = add_round_key(state, round_keys[0])

    # Rounds 1-9
    for r in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[r])

    # Final round (no MixColumns)
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[10])

    # Convert state matrix back to bytes
    output = bytearray(16)
    for col in range(4):
        for row in range(4):
            output[col * 4 + row] = state[row][col]

    return bytes(output)

# Example from NIST FIPS 197 Appendix B
key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
plaintext = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
ciphertext = aes_encrypt_block(plaintext, key)
print(f"Ciphertext: {ciphertext.hex()}")
# Expected: 3925841d02dc09fbdc118597196a0b32
```

---

## 8. 보안 분석

### 8.1 AES 보안 현황

2026년 현재, **완전한 AES에 대한 실용적인 공격은 발견되지 않았습니다.**

| 공격 | 최선의 결과 | 실용적? |
|------|-------------|---------|
| AES-128 브루트 포스 | $2^{128}$번 연산 | 아니요 (가장 빠른 슈퍼컴퓨터로 $10^{24}$년) |
| 바이클리크(Biclique) 공격 | AES-128에 대해 $2^{126.1}$ | 아니요 (무시할 수 있는 개선) |
| 연관 키(Related-Key) 공격 | AES-256에 대해 이론적 | 아니요 (연관 키 요구, 비현실적 모델) |
| 부채널(Side-Channel, 타이밍) | 캐시 타이밍으로 키 복구 | 예, 그러나 알고리즘이 아닌 **구현** 공격 |

### 8.2 일반적인 공격 유형

**차분 암호 분석(Differential Cryptanalysis)** — 입력 차분이 암호를 통해 어떻게 전파되는지 분석합니다. AES S-박스는 특히 이에 저항하도록 설계되었습니다.

**선형 암호 분석(Linear Cryptanalysis)** — 선형 방정식으로 암호를 근사합니다. AES S-박스의 비선형성이 높은 저항력을 제공합니다.

**부채널 공격(Side-Channel Attacks)** — 알고리즘 자체가 아닌 구현 세부 사항(타이밍, 전력 소비, 전자기 방사)을 이용합니다. 대응책으로는 상수 시간(Constant-Time) 구현과 마스킹(Masking)이 있습니다.

### 8.3 키 크기 권고사항

| 사용 사례 | 최소 키 크기 | 권장 사항 |
|----------|-------------|-----------|
| 단기 데이터 (< 5년) | AES-128 | AES-128 |
| 장기 데이터 (> 10년) | AES-128 | AES-256 |
| 양자 내성(Quantum-Resistant) | AES-256 | AES-256 (그로버 알고리즘이 유효 키 크기를 절반으로 줄임) |

---

## 9. 연습 문제

### 연습 문제 1: 빈도 분석 (기초)

"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" 메시지를 카이사르 암호(키=7)와 비즈네르 암호(키="CRYPTO")로 암호화하세요. 각 암호문에서 문자 빈도를 세어보세요. 어떤 암호가 평문 빈도 분포를 더 잘 숨기나요?

### 연습 문제 2: 파이스텔 암호 (중급)

1. 64비트 블록과 128비트 키를 가진 4라운드 파이스텔 암호를 구현하세요.
2. 평문 블록을 암호화한 후 복호화하세요. 원래 평문을 복구하는지 확인하세요.
3. 라운드가 1개뿐이면 어떻게 될까요? 암호문에서 평문에 대한 정보를 복구할 수 있나요?

### 연습 문제 3: AES 단계별 추적 (중급)

7.6절의 AES 구현을 사용하여, 처음 두 라운드에 대해 각 연산(SubBytes, ShiftRows, MixColumns, AddRoundKey) 이후의 상태 행렬을 출력하세요. 출력 결과를 NIST FIPS 197 부록 B 테스트 벡터와 비교하세요.

### 연습 문제 4: 눈사태 효과(Avalanche Effect) (심화)

1. 정확히 1비트 차이가 나는 두 16바이트 평문을 AES로 암호화하세요.
2. 1라운드, 2라운드, 10라운드 이후에 암호문에서 몇 비트가 다른지 세어보세요.
3. 라운드 수 대 다른 비트 수를 그래프로 그려보세요. 50% 근접 확산을 달성하는 데 몇 라운드가 필요한가요?

### 연습 문제 5: S-박스 분석 (심화)

1. AES S-박스에 고정점이 없음을 확인하세요: 모든 $x$에 대해 $S(x) \neq x$.
2. 반대 고정점도 없음을 확인하세요: 모든 $x$에 대해 $S(x) \neq \overline{x}$.
3. S-박스의 **비선형성(Non-Linearity)**을 계산하세요: 각 선형 근사 $a \cdot x \oplus b \cdot S(x) = 0$에 대해, 256개의 입력 중 몇 개가 이를 만족하는지 세어보세요. 128에서 최대 편차가 선형성 측도를 제공합니다.

---

**이전**: [수론 기초](./01_Number_Theory_Foundations.md) | **다음**: [블록 암호 모드](./03_Block_Cipher_Modes.md)

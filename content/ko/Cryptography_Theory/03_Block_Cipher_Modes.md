# 레슨 3: 블록 암호 운용 모드

**이전**: [대칭 암호](./02_Symmetric_Ciphers.md) | **다음**: [해시 함수](./04_Hash_Functions.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. ECB 모드가 왜 안전하지 않은지 설명하고 "펭귄 문제"를 시각적으로 시연할 수 있다
2. CBC, CTR, GCM 모드의 동작 방식과 각각의 트레이드오프를 설명할 수 있다
3. 블록 암호 기본 연산으로부터 CTR 모드 암호화를 구현할 수 있다
4. 인증 암호화(Authenticated Encryption)를 설명하고, Encrypt-then-MAC이 선호되는 이유를 설명할 수 있다
5. 패딩 오라클 공격(Padding Oracle Attack)을 설명하고, PKCS#7 패딩이 위험할 수 있는 이유를 설명할 수 있다
6. 스트림 유사 모드에서 논스(Nonce)/IV 재사용이 초래하는 치명적 결과를 식별할 수 있다

---

AES와 같은 블록 암호는 고정 크기의 블록(AES의 경우 128비트)을 암호화합니다. 실제 메시지가 정확히 128비트인 경우는 거의 없습니다. 블록 암호 **운용 모드(Modes of Operation)**는 블록 암호를 사용해 임의 길이의 메시지를 안전하게 암호화하는 방법을 정의합니다. 잘못된 모드를 선택하거나 올바른 모드를 잘못 사용하면 TLS의 BEAST 공격부터 플레이스테이션 네트워크 침해까지 실제 세계에서 치명적인 취약점이 발생해 왔습니다. 이 레슨에서는 각 주요 모드, 그 존재 이유, 그리고 안전하게 사용하는 방법을 설명합니다.

## 목차

1. [문제: 한 블록을 넘는 암호화](#1-문제-한-블록을-넘는-암호화)
2. [ECB 모드 — 잘못된 답](#2-ecb-모드--잘못된-답)
3. [CBC 모드](#3-cbc-모드)
4. [패딩과 PKCS#7](#4-패딩과-pkcs7)
5. [패딩 오라클 공격](#5-패딩-오라클-공격)
6. [CTR 모드](#6-ctr-모드)
7. [인증 암호화](#7-인증-암호화)
8. [GCM 모드](#8-gcm-모드)
9. [모드 비교 및 선택 가이드](#9-모드-비교-및-선택-가이드)
10. [연습 문제](#10-연습-문제)

---

## 1. 문제: 한 블록을 넘는 암호화

AES는 한 번에 정확히 16바이트를 암호화합니다. 메시지가 100바이트라면 다음을 결정해야 합니다:

1. 메시지를 블록으로 분할하는 방법
2. 각 블록을 독립적으로 암호화할지, 이전 블록과 연결할지 여부
3. 길이가 블록 크기의 배수가 아닌 메시지 처리 방법 (패딩)
4. 동일한 평문 블록이 동일한 암호문 블록을 생성하지 않도록 보장하는 방법

이러한 결정들이 **운용 모드**를 구성합니다. 이를 잘못 처리하면 완벽한 블록 암호의 보안성도 완전히 무효화될 수 있습니다.

---

## 2. ECB 모드 — 잘못된 답

### 2.1 ECB 동작 방식

**전자 코드북(ECB, Electronic Codebook)** 모드는 각 블록을 독립적으로 암호화합니다:

$$C_i = E_K(P_i)$$

```
P_1 → [AES_K] → C_1
P_2 → [AES_K] → C_2
P_3 → [AES_K] → C_3
...
```

> **비유:** ECB는 모든 스파이에게 똑같은 변장을 사용하는 것과 같습니다. 두 스파이가 같은 모습이라면(동일한 평문 블록), 변장도 동일하게 됩니다. 관찰자는 즉시 패턴을 알아챌 수 있습니다.

### 2.2 펭귄 문제(Penguin Problem)

ECB의 약점을 가장 잘 보여주는 예시: ECB 모드로 비트맵 이미지를 암호화합니다. 동일한 픽셀 블록이 동일한 암호문 블록을 생성하기 때문에, 원본 이미지의 윤곽이 여전히 선명하게 보입니다.

```python
def ecb_penguin_demo():
    """Demonstrate ECB mode's pattern-preserving weakness.

    Why this matters: any data with repeated patterns (images, structured
    data, database records with common fields) will leak those patterns
    through ECB encryption. This is not a theoretical concern — it has
    been exploited in real attacks.
    """
    import os

    # Simulate: create a "image" with repeating patterns
    # Row of white (0xFF) bytes and black (0x00) bytes
    white_block = bytes([0xFF] * 16)
    black_block = bytes([0x00] * 16)

    # Simple pattern: white-black stripes
    plaintext_image = (white_block + black_block) * 8

    # Encrypt each block independently (ECB)
    key = os.urandom(16)

    # Using Python's built-in AES would require a library;
    # instead, we'll demonstrate the concept with XOR as a toy cipher
    def toy_encrypt(block, key):
        """Toy 'encryption' — in real code, use AES."""
        return bytes(b ^ k for b, k in zip(block, key))

    ecb_ciphertext = b''
    for i in range(0, len(plaintext_image), 16):
        block = plaintext_image[i:i+16]
        ecb_ciphertext += toy_encrypt(block, key)

    # Check: do identical plaintext blocks produce identical ciphertext blocks?
    blocks = [ecb_ciphertext[i:i+16] for i in range(0, len(ecb_ciphertext), 16)]
    unique_blocks = len(set(blocks))
    total_blocks = len(blocks)

    print(f"Total blocks: {total_blocks}")
    print(f"Unique ciphertext blocks: {unique_blocks}")
    # Only 2 unique blocks! Pattern completely preserved.
    print(f"Pattern leaked: {'YES' if unique_blocks < total_blocks else 'NO'}")

ecb_penguin_demo()
```

### 2.3 ECB가 (제한적으로) 허용되는 경우

ECB는 단일 블록을 암호화할 때(예: 단일 AES 키 암호화)에만 허용됩니다. **어떤** 다중 블록 메시지에도 ECB를 절대 사용해서는 안 됩니다.

---

## 3. CBC 모드

### 3.1 CBC 동작 방식

**암호 블록 체이닝(CBC, Cipher Block Chaining)** 모드는 암호화 전에 각 평문 블록을 이전 암호문 블록과 XOR 연산합니다:

$$C_0 = IV$$
$$C_i = E_K(P_i \oplus C_{i-1})$$

```
IV ─────────┐
             ↓
P_1 → [XOR] → [AES_K] → C_1 ─────────┐
                                        ↓
P_2 ────────────────────→ [XOR] → [AES_K] → C_2 ─────┐
                                                        ↓
P_3 ────────────────────────────────────→ [XOR] → [AES_K] → C_3
```

복호화:

$$P_i = D_K(C_i) \oplus C_{i-1}$$

> **비유:** CBC는 변장들을 서로 연결하는 것과 같습니다. 각 스파이의 변장은 이전 스파이의 변장에 영향을 받습니다. 두 스파이가 같은 모습이더라도, 체인에서 서로 다른 위치에 있으므로 변장이 달라집니다.

### 3.2 초기화 벡터(IV, Initialization Vector)

IV는 $C_0$의 역할을 하며 다음 조건을 만족해야 합니다:
- **예측 불가능** (암호화마다 무작위)
- 메시지마다 **유일**
- **비밀이 아님** — 일반적으로 암호문 앞에 붙여서 전송

**왜 예측 불가능해야 하는가?** 공격자가 IV를 예측할 수 있다면 선택 평문 공격(Chosen-Plaintext Attack)을 가할 수 있습니다(TLS 1.0의 BEAST 공격은 예측 가능한 IV를 악용했습니다).

```python
import os

def cbc_encrypt(plaintext_blocks, key, iv):
    """CBC mode encryption (using XOR as a toy cipher for demonstration).

    Why CBC over ECB: identical plaintext blocks produce different
    ciphertext because each block is XORed with the previous ciphertext
    (which varies due to position and preceding content).
    """
    def toy_encrypt(block, key):
        return bytes(b ^ k for b, k in zip(block, key))

    ciphertext_blocks = []
    prev = iv

    for block in plaintext_blocks:
        # XOR with previous ciphertext block (chaining)
        xored = bytes(p ^ c for p, c in zip(block, prev))
        encrypted = toy_encrypt(xored, key)
        ciphertext_blocks.append(encrypted)
        prev = encrypted

    return ciphertext_blocks

def cbc_decrypt(ciphertext_blocks, key, iv):
    """CBC mode decryption."""
    def toy_decrypt(block, key):
        return bytes(b ^ k for b, k in zip(block, key))

    plaintext_blocks = []
    prev = iv

    for block in ciphertext_blocks:
        decrypted = toy_decrypt(block, key)
        plaintext = bytes(d ^ p for d, p in zip(decrypted, prev))
        plaintext_blocks.append(plaintext)
        prev = block

    return plaintext_blocks

# Demo: identical blocks produce different ciphertext in CBC
key = os.urandom(16)
iv = os.urandom(16)
identical_blocks = [bytes([0xAA] * 16)] * 4

ct = cbc_encrypt(identical_blocks, key, iv)
print("CBC ciphertext blocks (all plaintext blocks are identical):")
for i, block in enumerate(ct):
    print(f"  Block {i}: {block.hex()}")
# All ciphertext blocks are different!
```

### 3.3 CBC의 한계

- **순차 암호화:** 병렬화 불가 (각 블록이 이전 블록에 의존)
- **오류 전파:** $C_i$에서 비트 오류 발생 시 $P_i$ 전체와 $P_{i+1}$의 한 비트가 손상됨
- **패딩 필요:** 메시지는 블록 크기의 배수로 패딩되어야 함
- **무결성 보호 없음:** 공격자가 암호문을 탐지 없이 수정 가능 (5절 참조)

---

## 4. 패딩과 PKCS#7

### 4.1 문제

마지막 평문 블록이 16바이트보다 짧은 경우 패딩이 필요합니다. 패딩은 **모호하지 않아야** 합니다 — 수신자가 안정적으로 제거할 수 있어야 합니다.

### 4.2 PKCS#7 패딩

$n$바이트가 부족할 때, 각각 $n$ 값을 가진 $n$개의 바이트로 패딩합니다:

- 1바이트 부족: `01`로 패딩
- 2바이트 부족: `02 02`로 패딩
- 3바이트 부족: `03 03 03`으로 패딩
- 완전한 블록: `10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10` (16 = 0x10)의 전체 블록 추가

```python
def pkcs7_pad(data, block_size=16):
    """PKCS#7 padding.

    Why always pad (even if aligned): if the message happens to end with
    bytes that look like padding (e.g., 0x01), the receiver needs to
    distinguish real data from padding. Always adding a pad block
    removes ambiguity.
    """
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def pkcs7_unpad(data, block_size=16):
    """Remove PKCS#7 padding.

    Why validate carefully: improper validation enables padding oracle
    attacks (Section 5). Always check that all padding bytes have the
    correct value and that the pad length is valid.
    """
    if len(data) == 0 or len(data) % block_size != 0:
        raise ValueError("Invalid padded data length")

    pad_len = data[-1]

    if pad_len == 0 or pad_len > block_size:
        raise ValueError("Invalid padding value")

    # Verify ALL padding bytes have the correct value
    for i in range(pad_len):
        if data[-(i+1)] != pad_len:
            raise ValueError("Invalid padding")

    return data[:-pad_len]

# Examples
print(pkcs7_pad(b"Hello").hex())        # 48656c6c6f0b0b0b0b0b0b0b0b0b0b0b
print(pkcs7_pad(b"Hello World!!!!!").hex())  # Full block → adds 16 bytes of padding
print(pkcs7_unpad(pkcs7_pad(b"Hello")))  # b'Hello'
```

---

## 5. 패딩 오라클 공격

### 5.1 취약점

2002년 Serge Vaudenay는 복호화 시스템이 패딩의 유효성을 노출(**패딩 오라클**)하면, 공격자가 **키를 알지 못해도** 전체 암호문을 복호화할 수 있음을 발견했습니다.

오라클은 다음과 같이 미묘할 수 있습니다:
- 다른 HTTP 오류 코드 (400 vs 500)
- 다른 응답 시간 (패딩 확인이 MAC 확인 전후에 실패)
- 다른 오류 메시지

### 5.2 작동 방식 (단순화)

CBC 복호화를 상기합니다: $P_i = D_K(C_i) \oplus C_{i-1}$.

공격자는 $C_{i-1}$을 바이트별로 수정하고 수정된 암호문을 오라클에 제출합니다. 오라클이 유효한 패딩을 보고하면, 공격자는 한 번에 한 바이트씩 $D_K(C_i)$를 추론할 수 있습니다.

```python
def padding_oracle_demo():
    """Conceptual demonstration of the padding oracle attack.

    Why this works: the attacker controls C_{i-1} (the IV or previous
    ciphertext block). By systematically modifying its last byte and
    checking if padding is valid, the attacker can determine D_K(C_i)[15].
    Then they work backward through each byte.

    Attack complexity: 256 * block_size * num_blocks = O(256 * 16 * n)
    This is trivially fast — not a brute force on the key!
    """
    import os

    block_size = 16

    # Secret plaintext (attacker doesn't know this)
    secret = b"Secret message!!"  # Exactly 16 bytes
    padded = pkcs7_pad(secret)

    # Simulated encryption (using XOR for simplicity)
    key = os.urandom(16)
    iv = os.urandom(16)

    # Encrypt: C = E_K(P XOR IV)
    def encrypt_block(block, key):
        return bytes(b ^ k for b, k in zip(block, key))

    c1 = encrypt_block(bytes(p ^ i for p, i in zip(padded[:16], iv)), key)
    c2 = encrypt_block(bytes(p ^ c for p, c in zip(padded[16:32], c1)), key)

    # The padding oracle: returns True if padding is valid after decryption
    def oracle(modified_iv, ciphertext_block):
        """This oracle is the vulnerability. In real systems, it might be
        an HTTP 200 vs 500 response, or a timing difference."""
        decrypted = bytes(b ^ k for b, k in zip(ciphertext_block, key))
        plaintext = bytes(d ^ i for d, i in zip(decrypted, modified_iv))
        try:
            pkcs7_unpad(plaintext)
            return True
        except ValueError:
            return False

    # Attack: recover the first plaintext block byte by byte
    # We attack C1 by manipulating the IV
    intermediate = bytearray(16)  # D_K(C1) — what we're recovering

    print("Padding Oracle Attack (recovering first block):")

    for byte_pos in range(15, -1, -1):
        pad_value = 16 - byte_pos  # Target padding value

        # Set already-known bytes to produce correct padding
        modified_iv = bytearray(iv)
        for j in range(byte_pos + 1, 16):
            modified_iv[j] = intermediate[j] ^ pad_value

        # Try all 256 values for the target byte
        for guess in range(256):
            modified_iv[byte_pos] = guess
            if oracle(bytes(modified_iv), c1):
                # Found it! intermediate[byte_pos] XOR guess = pad_value
                intermediate[byte_pos] = guess ^ pad_value
                recovered_byte = intermediate[byte_pos] ^ iv[byte_pos]
                print(f"  Byte {byte_pos}: 0x{recovered_byte:02x} ('{chr(recovered_byte)}')")
                break

    # Recover full plaintext
    recovered = bytes(i ^ v for i, v in zip(intermediate, iv))
    print(f"\nRecovered plaintext: {recovered}")

padding_oracle_demo()
```

### 5.3 대응책

1. **인증 암호화 사용** (GCM, CCM) — 최선의 해결책
2. **복호화 전에 항상 MAC 검증** (Encrypt-then-MAC)
3. **상수 시간 비교** — 타이밍으로 패딩 유효성을 절대 노출하지 않음
4. **TLS 1.3**은 CBC 암호 스위트를 완전히 제거함

---

## 6. CTR 모드

### 6.1 CTR 동작 방식

**카운터(CTR, Counter)** 모드는 블록 암호를 스트림 암호로 변환합니다. 연속적인 카운터 값을 암호화하고 결과를 평문과 XOR 연산합니다:

$$C_i = P_i \oplus E_K(\text{Nonce} \| \text{Counter}_i)$$

```
Nonce||0 → [AES_K] → Keystream_0 ⊕ P_0 → C_0
Nonce||1 → [AES_K] → Keystream_1 ⊕ P_1 → C_1
Nonce||2 → [AES_K] → Keystream_2 ⊕ P_2 → C_2
```

복호화는 암호화와 동일합니다(XOR은 자기 역원).

### 6.2 CTR 모드의 장점

| 장점 | 설명 |
|------|------|
| **병렬화 가능** | 모든 블록을 동시에 암호화/복호화 가능 |
| **임의 접근** | 블록 $0$부터 $i-1$까지 복호화하지 않고 블록 $i$ 복호화 가능 |
| **패딩 불필요** | 키스트림과 XOR — 임의 길이에 작동 |
| **사전 계산 가능** | 평문을 알기 전에 키스트림 생성 가능 |
| **암호화/복호화 동일 코드** | XOR은 대칭적 |

```python
import os
import struct

def ctr_encrypt(plaintext, key, nonce):
    """CTR mode encryption.

    Why CTR is preferred over CBC in modern protocols:
    1. Parallelizable (huge performance win on multi-core/GPU)
    2. No padding needed (no padding oracle attacks)
    3. Random access (useful for disk encryption, seeking in streams)

    The nonce MUST be unique per message with the same key.
    Reusing a nonce is catastrophic (see Section 6.3).
    """
    def toy_encrypt(block, key):
        """Toy block cipher — replace with AES in real code."""
        return bytes(b ^ k for b, k in zip(block, key))

    ciphertext = bytearray()
    block_size = 16

    for i in range(0, len(plaintext), block_size):
        # Construct counter block: nonce (8 bytes) || counter (8 bytes)
        counter_block = nonce + struct.pack('>Q', i // block_size)

        # Generate keystream block
        keystream = toy_encrypt(counter_block, key)

        # XOR plaintext with keystream
        block = plaintext[i:i+block_size]
        for j in range(len(block)):
            ciphertext.append(block[j] ^ keystream[j])

    return bytes(ciphertext)

# CTR decryption is identical to encryption
ctr_decrypt = ctr_encrypt

# Demo
key = os.urandom(16)
nonce = os.urandom(8)
message = b"CTR mode needs no padding! Any length works."

ct = ctr_encrypt(message, key, nonce)
pt = ctr_decrypt(ct, key, nonce)
print(f"Plaintext:  {message}")
print(f"Ciphertext: {ct.hex()}")
print(f"Decrypted:  {pt}")
print(f"Match: {message == pt}")
```

### 6.3 논스 재사용: 치명적 실패

같은 키로 두 개의 다른 메시지에 동일한 논스를 사용하면:

$$C_1 = P_1 \oplus \text{Keystream}$$
$$C_2 = P_2 \oplus \text{Keystream}$$

그러면:

$$C_1 \oplus C_2 = P_1 \oplus P_2$$

키스트림이 상쇄되어 공격자는 두 평문의 XOR 값을 얻게 됩니다. 자연어 텍스트의 경우, 이는 크립 드래깅(Crib Dragging)을 통해 쉽게 악용될 수 있습니다.

```python
def nonce_reuse_attack():
    """Demonstrate why nonce reuse is catastrophic in CTR mode.

    Real-world examples of nonce reuse disasters:
    - WEP (WiFi) used a 24-bit IV, leading to frequent reuse
    - Microsoft PPTP VPN reused keystreams
    - Some TLS implementations had nonce reuse bugs
    """
    key = os.urandom(16)
    nonce = os.urandom(8)  # Same nonce for both messages!

    msg1 = b"Transfer $10000 to Alice"
    msg2 = b"The secret code is 4242"

    ct1 = ctr_encrypt(msg1, key, nonce)
    ct2 = ctr_encrypt(msg2, key, nonce)

    # Attacker XORs the two ciphertexts
    xor_result = bytes(a ^ b for a, b in zip(ct1, ct2))

    # This equals P1 XOR P2 — the key is completely eliminated!
    expected = bytes(a ^ b for a, b in zip(msg1, msg2))
    print(f"C1 XOR C2 = {xor_result.hex()}")
    print(f"P1 XOR P2 = {expected.hex()}")
    print(f"Key eliminated: {xor_result[:len(expected)] == expected}")

nonce_reuse_attack()
```

---

## 7. 인증 암호화

### 7.1 암호화만으로는 부족한 이유

암호화는 **기밀성(Confidentiality)**을 제공하지만, **무결성(Integrity)**은 제공하지 않습니다. 공격자는 암호문을 수정할 수 있고, 수신자는 변조를 감지하지 못한 채 다른(공격자가 선택한) 평문으로 복호화하게 됩니다.

### 7.2 세 가지 접근법

| 접근법 | 구성 | 보안성 |
|--------|------|--------|
| **Encrypt-and-MAC** | $C = E_K(P)$, $T = MAC_{K'}(P)$ | 안전하지 않음 (MAC이 평문 정보를 유출할 수 있음) |
| **MAC-then-Encrypt** | $T = MAC_{K'}(P)$, $C = E_K(P \| T)$ | 취약함 (MAC에 대한 패딩 오라클) |
| **Encrypt-then-MAC** | $C = E_K(P)$, $T = MAC_{K'}(C)$ | **안전** (표준 권장 방식) |

> **비유:** Encrypt-then-MAC은 편지를 봉투에 밀봉(암호화)한 다음, 겉면에 변조 방지 씰(MAC)을 붙이는 것과 같습니다. 봉투를 수정하는 사람은 씰을 부수게 되고, 수신자는 봉투를 열기 전에 메시지를 거부합니다.

### 7.3 AEAD: 연관 데이터 인증 암호화

현대 암호는 암호화와 인증을 단일 연산으로 결합합니다. **AEAD(Authenticated Encryption with Associated Data)**는 평문으로 전송되지만 변조되어서는 안 되는 추가 데이터(헤더, 메타데이터)도 인증합니다.

$$\text{AEAD}(K, N, P, A) \rightarrow (C, T)$$

여기서:
- $K$ = 키
- $N$ = 논스
- $P$ = 평문 (암호화 및 인증됨)
- $A$ = 연관 데이터 (인증되지만 암호화되지 않음)
- $C$ = 암호문
- $T$ = 인증 태그

---

## 8. GCM 모드

### 8.1 갈루아/카운터 모드(Galois/Counter Mode)

**GCM**은 암호화에는 CTR 모드를, 인증에는 다항식 해시(GHASH)를 결합합니다. TLS 1.2와 1.3에서 가장 널리 사용되는 AEAD 모드입니다.

```
Counter Mode (Encryption):
Nonce||1 → [AES_K] → ⊕ P_1 → C_1
Nonce||2 → [AES_K] → ⊕ P_2 → C_2

GHASH (Authentication):
(A_1 ⊕ H) · H → (... ⊕ C_1) · H → (... ⊕ C_2) · H → (... ⊕ len) · H → GHASH

Tag:
T = GHASH ⊕ E_K(Nonce||0)
```

### 8.2 GHASH: 인증 컴포넌트

GHASH는 $GF(2^{128})$ 위에서의 다항식 평가입니다:

$$\text{GHASH}(H, A, C) = X_{m+n+1}$$

여기서 $H = E_K(0^{128})$은 해시 키이고:

$$X_i = (X_{i-1} \oplus B_i) \cdot H$$

$B_i$는 연관 데이터, 암호문, 길이 블록의 블록들입니다.

```python
def gcm_concept_demo():
    """Conceptual demonstration of GCM's encrypt-then-authenticate structure.

    Why GCM is preferred:
    1. Single pass: encryption and authentication happen simultaneously
    2. Parallelizable: inherits CTR mode's parallelism
    3. Standardized: NIST SP 800-38D, used in TLS 1.2/1.3, IPsec, SSH
    4. Hardware acceleration: AES-NI + CLMUL instructions on modern CPUs

    In practice, always use a library (e.g., cryptography.hazmat for Python).
    """
    # In practice, you would use:
    # from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    # key = AESGCM.generate_key(bit_length=256)
    # aesgcm = AESGCM(key)
    # ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    # pt = aesgcm.decrypt(nonce, ct, associated_data)

    # Simplified structure demonstration
    print("GCM Structure:")
    print("1. Encrypt plaintext with CTR mode (Nonce||Counter)")
    print("2. Compute GHASH over: AAD || Ciphertext || Lengths")
    print("3. Tag = GHASH XOR E_K(Nonce||0)")
    print()
    print("Verification:")
    print("1. Recompute Tag from received ciphertext + AAD")
    print("2. Compare with received Tag (constant-time!)")
    print("3. If match: decrypt. If not: reject (do NOT decrypt)")
    print()
    print("Key property: tag verification BEFORE decryption")
    print("prevents padding oracle and chosen-ciphertext attacks")

gcm_concept_demo()
```

### 8.3 GCM 보안 요구사항

1. **96비트 논스** (권장): 같은 키로 암호화할 때마다 유일해야 함
2. **논스 재사용은 치명적:** GHASH 키 $H$가 노출되어 인증 태그 위조 가능
3. **최대 메시지 크기:** 메시지당 $2^{39} - 256$ 비트
4. **키당 최대 메시지 수:** $2^{32}$ (무작위 논스 사용 시), 충돌 확률을 $2^{-32}$ 이하로 유지

### 8.4 다른 AEAD 모드

| 모드 | 구조 | 비고 |
|------|------|------|
| **GCM** | CTR + GHASH | 가장 일반적; 하드웨어 가속 지원 |
| **CCM** | CTR + CBC-MAC | WiFi(WPA2), 블루투스에서 사용 |
| **ChaCha20-Poly1305** | ChaCha20 + Poly1305 | 소프트웨어 친화적; AES-NI 없을 때 TLS에서 사용 |
| **OCB** | 병렬 암호화 + 인증 | 가장 빠름; 특허 문제 (현재 오픈소스는 무료) |
| **SIV** | 합성 IV | 논스 오용에 내성 (우아한 성능 저하) |

---

## 9. 모드 비교 및 선택 가이드

### 9.1 요약 표

| 모드 | 기밀성 | 무결성 | 병렬화 | 패딩 | 논스 재사용 영향 |
|------|--------|--------|--------|------|-----------------|
| ECB | 약함 | 없음 | 가능 | 필요 | 해당 없음 (논스 없음) |
| CBC | 강함 | 없음 | 복호화만 가능 | 필요 | IV 예측 가능 → CPA |
| CTR | 강함 | 없음 | 가능 | 불필요 | 완전 붕괴 (키스트림 재사용) |
| GCM | 강함 | **있음** | 가능 | 불필요 | 완전 붕괴 (키 + 위조) |
| CCM | 강함 | **있음** | 불가 | 불필요 | 완전 붕괴 |

### 9.2 결정 순서도

```
Need encryption?
├── Yes → Need authentication too?
│         ├── Yes → Use GCM (or ChaCha20-Poly1305)
│         └── No  → Are you sure? (You almost certainly need authentication)
│                   └── Use CTR if you have a separate MAC, otherwise use GCM
└── No  → Need authentication only? → Use HMAC (see Lesson 4)
```

### 9.3 흔한 실수

| 실수 | 결과 | 해결책 |
|------|------|--------|
| ECB 사용 | 패턴 유출 | GCM 사용 |
| CTR/GCM에서 논스 재사용 | 키스트림 재사용, 위조 | 무작위 논스 + 카운터 사용; 키 교체 |
| MAC 없이 CBC 사용 | 패딩 오라클, 비트 플리핑 | GCM 사용 (또는 Encrypt-then-MAC) |
| 예측 가능한 CBC IV | 선택 평문 공격 (BEAST) | 메시지마다 무작위 IV |
| MAC-then-Encrypt | MAC에도 불구한 패딩 오라클 | Encrypt-then-MAC 또는 AEAD |

```python
def mode_selection_guide(need_auth, need_parallel, hardware_aes):
    """Recommend the best mode for a given scenario.

    Why this matters: choosing the wrong mode is the #1 cause of
    symmetric crypto vulnerabilities in real applications.
    """
    if need_auth:
        if hardware_aes:
            print("Recommendation: AES-256-GCM")
            print("  - Hardware-accelerated AES-NI + CLMUL")
            print("  - TLS 1.3 default cipher suite")
        else:
            print("Recommendation: ChaCha20-Poly1305")
            print("  - Fast in software (no AES-NI needed)")
            print("  - Used by Google (QUIC), Cloudflare, WireGuard")
    else:
        if need_parallel:
            print("Recommendation: AES-CTR (with separate HMAC)")
            print("  - But seriously, just use GCM")
        else:
            print("Recommendation: AES-CBC (with encrypt-then-MAC)")
            print("  - Legacy; prefer GCM for new designs")

mode_selection_guide(need_auth=True, need_parallel=True, hardware_aes=True)
```

---

## 10. 연습 문제

### 연습 문제 1: ECB 패턴 탐지 (기본)

10개의 평문 블록 목록을 만들되, 그 중 4개가 동일하게 합니다. ECB와 CBC 모드로 암호화하세요. 각 경우에서 고유한 암호문 블록 수를 세고, 왜 다른지 설명하세요.

### 연습 문제 2: CTR 모드 구현 (중급)

실제 AES 블록 암호를 위해 Python의 `cryptography` 라이브러리를 사용해 CTR 모드 암호화를 구현하세요. 메시지를 암호화한 후 재암호화(CTR 복호화는 암호화와 동일)를 통해 복호화하고, 왕복 변환을 검증하세요.

### 연습 문제 3: 논스 재사용 공격 (중급)

1. 동일한 논스로 CTR 모드를 사용해 두 개의 알려진 메시지를 암호화하세요.
2. 두 암호문만 가지고 두 평문의 XOR 값을 복원하세요.
3. 한 평문을 알고 있다면, 다른 평문을 복원하는 방법을 보이세요.
4. 논의: 이것이 일회용 패드(One-Time Pad)와 어떻게 연관되며, 논스 재사용이 왜 키 재사용과 동일한지 설명하세요.

### 연습 문제 4: 패딩 오라클 시뮬레이터 (심화)

완전한 패딩 오라클 공격을 구현하세요:
1. CBC 암호문을 복호화하고 유효한 패딩에 대해 True/False를 반환하는 "오라클 서버"를 생성하세요.
2. 오라클만 사용해 평문을 바이트별로 복원하는 공격자를 작성하세요.
3. 3블록 메시지를 복호화하는 데 필요한 총 오라클 쿼리 수를 세세요.
4. 예상 쿼리 수가 대략 $3 \times 16 \times 128 = 6144$임을 검증하세요.

### 연습 문제 5: 논스 재사용 하에서의 GCM 태그 위조 (심화)

같은 논스로 두 개의 GCM 암호화를 관찰한 공격자가 다음을 할 수 있는 방법을 수식으로 연구하고 설명하세요:
1. GHASH 키 $H$ 복원
2. 임의 메시지에 대한 유효한 인증 태그 위조
3. SIV 모드(AES-GCM-SIV)가 논스 재사용 하에서 왜 치명적으로 실패하는 대신 성능을 우아하게 저하시키는지 설명하세요.

---

**이전**: [대칭 암호](./02_Symmetric_Ciphers.md) | **다음**: [해시 함수](./04_Hash_Functions.md)

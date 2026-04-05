# Cryptography 기초

**이전**: [01. Security 기초](./01_Security_Fundamentals.md) | **다음**: [03. Hashing과 데이터 무결성](./03_Hashing_and_Integrity.md)

---

암호학은 의도된 당사자만 접근할 수 있도록 통신과 데이터를 보호하는 관행입니다. 이 레슨은 현대 암호학의 두 가지 주요 분야인 대칭 및 비대칭 암호화와 키 교환 프로토콜, 디지털 서명, 그리고 실용적인 Python 구현을 다룹니다. 이 레슨이 끝나면 주어진 문제에 대해 올바른 암호화 기본 요소를 선택하고 가장 일반적인 함정을 피할 수 있게 됩니다.

**난이도**: ⭐⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 대칭 및 비대칭 암호화의 차이점 이해하기
2. Python에서 AES-GCM과 ChaCha20-Poly1305 암호화 구현하기
3. 비대칭 암호학을 위한 RSA, ECDSA, Ed25519 이해하기
4. Diffie-Hellman과 ECDH 키 교환 구현하기
5. 디지털 서명 생성 및 검증하기
6. 일반적인 암호학적 함정 인식하고 피하기
7. 현대 암호학 권장 사항 적용하기

---

## 목차

1. [암호학 개요](#1-암호학-개요)
2. [대칭 암호화](#2-대칭-암호화)
3. [블록 암호 운영 모드](#3-블록-암호-운영-모드)
4. [AES-GCM: 현대 표준](#4-aes-gcm-현대-표준)
5. [ChaCha20-Poly1305](#5-chacha20-poly1305)
6. [비대칭 암호화](#6-비대칭-암호화)
7. [RSA](#7-rsa)
8. [타원 곡선 암호학](#8-타원-곡선-암호학)
9. [키 교환](#9-키-교환)
10. [디지털 서명](#10-디지털-서명)
11. [일반적인 함정과 피하는 방법](#11-일반적인-함정과-피하는-방법)
12. [현대 권장 사항](#12-현대-권장-사항)
13. [연습 문제](#13-연습-문제)
14. [참고 자료](#14-참고-자료)

---

## 1. 암호학 개요

### 1.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                    현대 암호학 분류                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Cryptography                                                        │
│  ├── Symmetric (공유 키)                                             │
│  │   ├── Block Ciphers                                               │
│  │   │   ├── AES (128/192/256-bit key)                               │
│  │   │   ├── Modes: ECB, CBC, CTR, GCM, CCM                        │
│  │   │   └── Legacy: DES, 3DES, Blowfish (피할 것)                  │
│  │   └── Stream Ciphers                                              │
│  │       ├── ChaCha20(-Poly1305)                                    │
│  │       └── Legacy: RC4 (깨짐, 피할 것)                            │
│  │                                                                   │
│  ├── Asymmetric (공개/개인 키 쌍)                                    │
│  │   ├── Encryption                                                  │
│  │   │   ├── RSA-OAEP                                               │
│  │   │   └── ECIES (Elliptic Curve Integrated Encryption)           │
│  │   ├── Digital Signatures                                          │
│  │   │   ├── RSA-PSS                                                │
│  │   │   ├── ECDSA (secp256r1, secp384r1)                          │
│  │   │   └── Ed25519 / Ed448                                       │
│  │   └── Key Exchange                                                │
│  │       ├── Diffie-Hellman (DH)                                    │
│  │       ├── ECDH (X25519, P-256)                                   │
│  │       └── Post-quantum: ML-KEM (CRYSTALS-Kyber)                 │
│  │                                                                   │
│  ├── Hash Functions (레슨 03에서 다룸)                               │
│  │   ├── SHA-2 (SHA-256, SHA-512)                                   │
│  │   ├── SHA-3 (Keccak)                                             │
│  │   └── BLAKE2/BLAKE3                                              │
│  │                                                                   │
│  └── Key Derivation Functions                                        │
│      ├── HKDF                                                        │
│      ├── PBKDF2                                                      │
│      └── scrypt / Argon2 (비밀번호 기반)                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 개념

```
┌─────────────────────────────────────────────────────────────────────┐
│                    핵심 암호학 개념                                  │
├────────────────────┬────────────────────────────────────────────────┤
│ 용어               │ 정의                                           │
├────────────────────┼────────────────────────────────────────────────┤
│ Plaintext          │ 원본, 암호화되지 않은 데이터                   │
│ Ciphertext         │ 암호화된 (읽을 수 없는) 출력                   │
│ Key                │ 암호화/복호화에 사용되는 비밀 값               │
│ Nonce / IV         │ 한 번만 사용되는 숫자; 동일한 평문이           │
│                    │ 동일한 암호문을 생성하는 것을 방지             │
│ Authenticated enc. │ 무결성도 검증하는 암호화 (AEAD)               │
│ Key derivation     │ 비밀번호나 다른 키로부터                       │
│                    │ 암호화 키를 유도                               │
│ Forward secrecy    │ 장기 키의 침해가 과거                          │
│                    │ 세션 키를 침해하지 않음                        │
└────────────────────┴────────────────────────────────────────────────┘
```

### 1.3 Python Cryptography 라이브러리 설정

이 레슨의 모든 코드 예제는 높은 수준의 레시피와 낮은 수준의 기본 요소를 제공하는 `cryptography` 라이브러리를 사용합니다.

```bash
pip install cryptography
```

```python
# Verify installation
import cryptography
print(f"cryptography version: {cryptography.__version__}")

# The library has two main layers:
# 1. High-level (recipes): cryptography.fernet, cryptography.hazmat.primitives.kdf
# 2. Low-level (hazmat): cryptography.hazmat.primitives.ciphers, asymmetric, etc.
#
# "hazmat" stands for "hazardous materials" - these primitives can be
# misused. Always prefer high-level APIs when available.
```

---

## 2. 대칭 암호화

대칭 암호화는 암호화와 복호화 모두에 동일한 키를 사용합니다. 빠르고 대량의 데이터를 암호화하는 데 적합합니다.

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│Plaintext │──Key──▶│ Encrypt  │────────▶│Ciphertext│
│ "Hello"  │         │ (AES)    │         │ 0xA3F1.. │
└──────────┘         └──────────┘         └──────────┘

┌──────────┐         ┌──────────┐         ┌──────────┐
│Ciphertext│──Key──▶│ Decrypt  │────────▶│Plaintext │
│ 0xA3F1.. │         │ (AES)    │         │ "Hello"  │
└──────────┘         └──────────┘         └──────────┘

         두 작업 모두 동일한 키 사용!
```

### 2.1 AES (Advanced Encryption Standard)

AES는 가장 널리 사용되는 대칭 암호입니다. 128비트 블록에서 작동하며 128, 192 또는 256비트의 키 크기를 지원합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AES 키 크기                                  │
├──────────┬──────────┬──────────┬────────────────────────────────────┤
│ 키 크기  │ 라운드   │ 보안     │ 사용 사례                          │
├──────────┼──────────┼──────────┼────────────────────────────────────┤
│ 128-bit  │ 10       │ ~128 bit │ 범용, 빠름                         │
│ 192-bit  │ 12       │ ~192 bit │ 실제로 거의 사용되지 않음          │
│ 256-bit  │ 14       │ ~256 bit │ 정부/군사, 양자 이후               │
│          │          │          │ 저항 마진                          │
└──────────┴──────────┴──────────┴────────────────────────────────────┘
```

### 2.2 Fernet: 높은 수준의 대칭 암호화

간단한 사용 사례의 경우 `Fernet` 클래스는 간단한 API로 인증된 암호화를 제공합니다.

```python
from cryptography.fernet import Fernet
import base64

# Generate a random key (URL-safe base64 encoded)
key = Fernet.generate_key()
print(f"Key: {key.decode()}")
print(f"Key length: {len(base64.urlsafe_b64decode(key))} bytes = "
      f"{len(base64.urlsafe_b64decode(key)) * 8} bits")

# Create cipher
cipher = Fernet(key)

# Encrypt
plaintext = b"Sensitive financial data: account balance $50,000"
ciphertext = cipher.encrypt(plaintext)
print(f"\nPlaintext:  {plaintext.decode()}")
print(f"Ciphertext: {ciphertext[:60]}...")
print(f"Ciphertext length: {len(ciphertext)} bytes")

# Decrypt
decrypted = cipher.decrypt(ciphertext)
assert decrypted == plaintext
print(f"Decrypted:  {decrypted.decode()}")

# Fernet includes a timestamp - you can set a TTL (time-to-live)
import time
token = cipher.encrypt(b"temporary secret")
# time.sleep(2)  # Uncomment to test expiration
try:
    cipher.decrypt(token, ttl=60)  # Valid for 60 seconds
    print("\nToken is still valid")
except Exception as e:
    print(f"\nToken expired: {e}")
```

**Fernet이 내부적으로 하는 일:**
```
1. 랜덤 128비트 IV 생성
2. AES-128-CBC로 암호화
3. HMAC-SHA256으로 인증
4. 연결: version || timestamp || IV || ciphertext || HMAC
5. 결과를 Base64 인코딩
```

---

## 3. 블록 암호 운영 모드

블록 암호(AES 같은)는 고정 크기 블록을 암호화합니다. 운영 모드는 한 블록보다 긴 메시지를 처리하는 방법을 정의합니다.

### 3.1 ECB 모드 (절대 사용하지 마세요)

ECB(Electronic Codebook)는 각 블록을 독립적으로 암호화합니다. 동일한 평문 블록은 동일한 암호문 블록을 생성하여 패턴을 노출합니다.

```
┌──────────────────────────────────────────────────────────────────┐
│                    ECB 모드 - 왜 깨졌는가                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  평문 블록:       [AAAA] [BBBB] [AAAA] [CCCC] [AAAA]            │
│                      │      │      │      │      │              │
│                     AES    AES    AES    AES    AES             │
│                      │      │      │      │      │              │
│  암호문 블록:     [X1X1] [Y2Y2] [X1X1] [Z3Z3] [X1X1]           │
│                                                                  │
│  문제: 동일한 평문 블록 → 동일한 암호문!                        │
│  공격자는 복호화 없이 패턴을 볼 수 있습니다.                    │
│                                                                  │
│  고전적 예: ECB 암호화된 비트맵 이미지는 모양을 보존합니다       │
│  인접한 동일한 픽셀이 동일한 암호문을 생성하기 때문입니다.       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

```python
# Demonstration: ECB leaks patterns
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os

key = os.urandom(32)  # AES-256

# ECB mode (DO NOT USE in production - for demonstration only)
def ecb_encrypt_block(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt a single block with AES-ECB. For demonstration only!"""
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    encryptor = cipher.encryptor()
    return encryptor.update(plaintext) + encryptor.finalize()

# Same plaintext block produces same ciphertext
block1 = b"AAAAAAAAAAAAAAAA"  # 16 bytes = 1 AES block
block2 = b"BBBBBBBBBBBBBBBB"

ct1a = ecb_encrypt_block(key, block1)
ct1b = ecb_encrypt_block(key, block1)  # Same input
ct2  = ecb_encrypt_block(key, block2)

print("ECB Pattern Leak Demonstration:")
print(f"  Block 'AAA...' → {ct1a.hex()[:32]}...")
print(f"  Block 'AAA...' → {ct1b.hex()[:32]}... (SAME ciphertext!)")
print(f"  Block 'BBB...' → {ct2.hex()[:32]}... (different)")
print(f"  ct1a == ct1b: {ct1a == ct1b}")  # True - this is the problem
```

### 3.2 CBC 모드 (레거시, 주의해서 사용)

CBC(Cipher Block Chaining)는 암호화하기 전에 각 평문 블록을 이전 암호문 블록과 XOR합니다. 랜덤 IV가 필요합니다.

```
┌──────────────────────────────────────────────────────────────────┐
│                        CBC 모드                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│     IV ──┐                                                      │
│          ▼                                                      │
│  P1 ──▶ XOR ──▶ AES ──▶ C1                                    │
│                          │                                      │
│                          ▼                                      │
│  P2 ──────────▶ XOR ──▶ AES ──▶ C2                            │
│                                  │                              │
│                                  ▼                              │
│  P3 ──────────────────▶ XOR ──▶ AES ──▶ C3                    │
│                                                                  │
│  각 암호문 블록은 모든 이전 블록에 의존합니다.                   │
│  랜덤 IV는 동일한 평문이 다르게 암호화되도록 보장합니다.         │
│                                                                  │
│  ⚠ 패딩 필요 (예: PKCS#7)                                      │
│  ⚠ 인증되지 않으면 패딩 오라클 공격에 취약                      │
│  ⚠ 암호화를 병렬화할 수 없음                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import os

def aes_cbc_encrypt(key: bytes, plaintext: bytes) -> tuple:
    """AES-CBC encryption with PKCS7 padding. Returns (iv, ciphertext)."""
    iv = os.urandom(16)  # Random 128-bit IV

    # Pad plaintext to block size
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()

    # Encrypt
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    return iv, ciphertext

def aes_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    """AES-CBC decryption with PKCS7 unpadding."""
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove padding
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()

    return plaintext

# Usage
key = os.urandom(32)  # AES-256
message = b"CBC mode requires padding and a random IV for each message"

iv, ct = aes_cbc_encrypt(key, message)
print(f"IV:         {iv.hex()}")
print(f"Ciphertext: {ct.hex()[:64]}...")

pt = aes_cbc_decrypt(key, iv, ct)
print(f"Decrypted:  {pt.decode()}")

# Same plaintext, different IV → different ciphertext
iv2, ct2 = aes_cbc_encrypt(key, message)
print(f"\nSame message, new IV:")
print(f"  ct1: {ct.hex()[:32]}...")
print(f"  ct2: {ct2.hex()[:32]}...")
print(f"  Same? {ct == ct2}")  # False - good!
```

### 3.3 CTR 모드

CTR(Counter) 모드는 블록 암호를 스트림 암호로 변환합니다. 병렬화 가능하며 패딩이 필요하지 않습니다.

```
┌──────────────────────────────────────────────────────────────────┐
│                        CTR 모드                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Nonce|Counter=0 ──▶ AES ──▶ Keystream0 ──XOR──▶ C0            │
│                                               ▲                  │
│                                               │                  │
│                                              P0                  │
│                                                                  │
│  Nonce|Counter=1 ──▶ AES ──▶ Keystream1 ──XOR──▶ C1            │
│                                               ▲                  │
│                                               │                  │
│                                              P1                  │
│                                                                  │
│  ✓ 병렬화 가능 (암호화 및 복호화)                               │
│  ✓ 패딩 필요 없음                                               │
│  ✓ 모든 블록으로 시크 가능                                      │
│  ⚠ Nonce 재사용은 치명적 (두 평문의 XOR 누출)                  │
│  ⚠ 인증 없음 (대신 GCM 사용)                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. AES-GCM: 현대 표준

GCM(Galois/Counter Mode)은 CTR 모드 암호화와 GMAC 인증을 결합합니다. 단일 작업으로 기밀성과 무결성을 모두 제공합니다. 이를 Authenticated Encryption with Associated Data(AEAD)라고 합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AES-GCM (AEAD)                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  입력:  Key, Nonce (96-bit), Plaintext, AAD (선택)                  │
│  출력: Ciphertext, Authentication Tag (128-bit)                     │
│                                                                      │
│  ┌─────────┐     ┌─────────────┐     ┌────────────┐                │
│  │  Nonce   │────▶│  AES-CTR    │────▶│ Ciphertext │                │
│  └─────────┘     │  Encryption │     └──────┬─────┘                │
│                  └─────────────┘            │                       │
│  ┌─────────┐                               │                       │
│  │  AAD    │──────────┐                    │                       │
│  │(header) │          ▼                    ▼                       │
│  └─────────┘     ┌─────────────┐     ┌──────────┐                  │
│                  │   GHASH     │────▶│   Auth   │                  │
│                  │  (GMAC)    │     │   Tag    │                  │
│                  └─────────────┘     └──────────┘                  │
│                                                                      │
│  AAD = Associated Authenticated Data                                │
│  - 인증되지만 암호화되지 않음                                        │
│  - 예: 메시지 헤더, 패킷 시퀀스 번호                                │
│  - AAD 변조는 인증 실패를 유발                                      │
│                                                                      │
│  ✓ AEAD: 기밀성 + 무결성 + 진정성                                  │
│  ✓ 병렬화 가능                                                      │
│  ✓ 하드웨어 가속 (AES-NI)                                          │
│  ⚠ Nonce는 키당 고유해야 함 (절대 재사용 금지!)                    │
│  ⚠ 96비트 nonce는 키당 ~2^32 암호화로 제한                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 AES-GCM 구현

```python
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def aes_gcm_encrypt(key: bytes, plaintext: bytes,
                     aad: bytes = None) -> tuple:
    """
    Encrypt with AES-GCM.
    Returns (nonce, ciphertext_with_tag).
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce (NIST recommended)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ciphertext

def aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes,
                     aad: bytes = None) -> bytes:
    """
    Decrypt with AES-GCM.
    Raises InvalidTag if authentication fails.
    """
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)

# Generate a 256-bit key
key = AESGCM.generate_key(bit_length=256)
print(f"Key: {key.hex()} ({len(key) * 8} bits)")

# Encrypt a message
message = b"Top secret: The missile codes are 12345"
aad = b"message-id: 42, timestamp: 2026-01-15"  # Authenticated but not encrypted

nonce, ciphertext = aes_gcm_encrypt(key, message, aad)
print(f"\nNonce:      {nonce.hex()}")
print(f"Ciphertext: {ciphertext.hex()[:64]}...")
print(f"CT length:  {len(ciphertext)} bytes "
      f"(plaintext: {len(message)} + tag: 16)")

# Decrypt
plaintext = aes_gcm_decrypt(key, nonce, ciphertext, aad)
print(f"Decrypted:  {plaintext.decode()}")

# Tamper with ciphertext → authentication fails
tampered_ct = bytearray(ciphertext)
tampered_ct[0] ^= 0xFF  # Flip bits in first byte
try:
    aes_gcm_decrypt(key, nonce, bytes(tampered_ct), aad)
    print("ERROR: Should have failed!")
except Exception as e:
    print(f"\nTamper detected: {type(e).__name__}")

# Tamper with AAD → authentication fails
try:
    aes_gcm_decrypt(key, nonce, ciphertext, b"tampered AAD")
    print("ERROR: Should have failed!")
except Exception as e:
    print(f"AAD tamper detected: {type(e).__name__}")
```

### 4.2 AES-GCM을 사용한 파일 암호화

```python
import os
import struct
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pathlib import Path

class FileEncryptor:
    """Encrypt/decrypt files using AES-256-GCM."""

    CHUNK_SIZE = 64 * 1024  # 64 KB chunks
    NONCE_SIZE = 12
    TAG_SIZE = 16

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("Key must be 256 bits (32 bytes)")
        self.aesgcm = AESGCM(key)

    def encrypt_file(self, input_path: str, output_path: str) -> dict:
        """
        Encrypt a file chunk by chunk.
        Format: [nonce (12B)][chunk_count (4B)][encrypted_chunk1][encrypted_chunk2]...
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        file_size = input_file.stat().st_size
        chunks_written = 0

        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            # Write file nonce (used as base; each chunk gets nonce + counter)
            base_nonce = os.urandom(self.NONCE_SIZE)
            fout.write(base_nonce)

            # Placeholder for chunk count
            chunk_count_pos = fout.tell()
            fout.write(struct.pack('<I', 0))  # Will update later

            while True:
                chunk = fin.read(self.CHUNK_SIZE)
                if not chunk:
                    break

                # Derive unique nonce for this chunk
                chunk_nonce = self._derive_chunk_nonce(base_nonce, chunks_written)

                # AAD includes chunk index to prevent reordering
                aad = struct.pack('<I', chunks_written)

                encrypted = self.aesgcm.encrypt(chunk_nonce, chunk, aad)

                # Write: [length (4B)][encrypted_data]
                fout.write(struct.pack('<I', len(encrypted)))
                fout.write(encrypted)
                chunks_written += 1

            # Update chunk count
            fout.seek(chunk_count_pos)
            fout.write(struct.pack('<I', chunks_written))

        return {
            "input_size": file_size,
            "chunks": chunks_written,
            "output_size": Path(output_path).stat().st_size
        }

    def decrypt_file(self, input_path: str, output_path: str) -> dict:
        """Decrypt a file encrypted with encrypt_file."""
        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            base_nonce = fin.read(self.NONCE_SIZE)
            chunk_count = struct.unpack('<I', fin.read(4))[0]

            for i in range(chunk_count):
                chunk_nonce = self._derive_chunk_nonce(base_nonce, i)
                aad = struct.pack('<I', i)

                enc_len = struct.unpack('<I', fin.read(4))[0]
                encrypted = fin.read(enc_len)

                decrypted = self.aesgcm.decrypt(chunk_nonce, encrypted, aad)
                fout.write(decrypted)

        return {"chunks_decrypted": chunk_count}

    def _derive_chunk_nonce(self, base_nonce: bytes, chunk_index: int) -> bytes:
        """Derive a unique nonce for each chunk by XORing with chunk index."""
        nonce_int = int.from_bytes(base_nonce, 'big') ^ chunk_index
        return nonce_int.to_bytes(self.NONCE_SIZE, 'big')

# Usage example
key = AESGCM.generate_key(bit_length=256)
encryptor = FileEncryptor(key)

# Create a test file
test_data = b"Hello, encrypted world! " * 10000  # ~240 KB
with open("/tmp/test_plain.bin", "wb") as f:
    f.write(test_data)

# Encrypt
info = encryptor.encrypt_file("/tmp/test_plain.bin", "/tmp/test_encrypted.bin")
print(f"Encrypted: {info}")

# Decrypt
info = encryptor.decrypt_file("/tmp/test_encrypted.bin", "/tmp/test_decrypted.bin")
print(f"Decrypted: {info}")

# Verify
with open("/tmp/test_decrypted.bin", "rb") as f:
    decrypted_data = f.read()
assert decrypted_data == test_data
print("Verification: OK - decrypted data matches original")

# Clean up
for f in ["/tmp/test_plain.bin", "/tmp/test_encrypted.bin", "/tmp/test_decrypted.bin"]:
    Path(f).unlink(missing_ok=True)
```

---

## 5. ChaCha20-Poly1305

ChaCha20-Poly1305는 ChaCha20 스트림 암호와 Poly1305 MAC을 결합한 AEAD 암호입니다. AES-GCM의 주요 대안입니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                  AES-GCM vs ChaCha20-Poly1305                        │
├──────────────────┬──────────────────────┬────────────────────────────┤
│                  │ AES-256-GCM          │ ChaCha20-Poly1305          │
├──────────────────┼──────────────────────┼────────────────────────────┤
│ 키 크기          │ 256 bits             │ 256 bits                   │
│ Nonce 크기       │ 96 bits              │ 96 bits                    │
│ Tag 크기         │ 128 bits             │ 128 bits                   │
│ 속도 (HW 가속)   │ 매우 빠름 (AES-NI)   │ AES-NI로 느림              │
│ 속도 (소프트웨어)│ 느림                 │ 빠름 (특수 HW 불필요)      │
│ 모바일/임베디드  │ HW 지원 필요         │ 우수 (순수 소프트웨어)     │
│ 사이드 채널      │ 주의 필요 (T-테이블) │ 본질적으로 상수 시간       │
│ 사용처           │ TLS, IPsec, 디스크 암│ TLS (Google/CF), WireGuard│
│ Nonce 오용       │ 치명적               │ 치명적                     │
│ 양자 이후        │ PQ 저항 없음         │ PQ 저항 없음               │
│                  │ (단독으로)           │ (단독으로)                 │
└──────────────────┴──────────────────────┴────────────────────────────┘
```

### 5.1 ChaCha20-Poly1305 구현

```python
import os
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# Generate a 256-bit key
key = ChaCha20Poly1305.generate_key()
chacha = ChaCha20Poly1305(key)

# Encrypt
nonce = os.urandom(12)  # 96-bit nonce
message = b"ChaCha20-Poly1305 is great for mobile and embedded devices"
aad = b"metadata: device=mobile, version=1"

ciphertext = chacha.encrypt(nonce, message, aad)
print(f"Plaintext:  {message.decode()}")
print(f"Ciphertext: {ciphertext.hex()[:64]}...")
print(f"CT length:  {len(ciphertext)} (plaintext {len(message)} + tag 16)")

# Decrypt
plaintext = chacha.decrypt(nonce, ciphertext, aad)
assert plaintext == message
print(f"Decrypted:  {plaintext.decode()}")

# Tamper detection
tampered = bytearray(ciphertext)
tampered[-1] ^= 0x01
try:
    chacha.decrypt(nonce, bytes(tampered), aad)
except Exception as e:
    print(f"Tamper detected: {type(e).__name__}")
```

### 5.2 XChaCha20-Poly1305 (확장 Nonce)

XChaCha20은 192비트 nonce를 사용합니다(96비트 대비). 이는 현실적인 충돌 위험 없이 무작위로 생성될 수 있을 만큼 충분히 큽니다. 이는 nonce 관리 부담을 제거합니다.

```python
# XChaCha20 is available through libsodium bindings (PyNaCl)
# pip install pynacl

import nacl.secret
import nacl.utils

# XChaCha20-Poly1305 with 192-bit random nonce
key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)  # 256-bit
box = nacl.secret.SecretBox(key)

# Encrypt - nonce is generated automatically (192-bit, random-safe)
message = b"With XChaCha20, random nonces are always safe"
encrypted = box.encrypt(message)

print(f"Nonce size: {box.NONCE_SIZE} bytes = {box.NONCE_SIZE * 8} bits")
print(f"Encrypted length: {len(encrypted)} bytes")

# Decrypt
decrypted = box.decrypt(encrypted)
print(f"Decrypted: {decrypted.decode()}")
```

---

## 6. 비대칭 암호화

비대칭(공개키) 암호화는 키 쌍을 사용합니다: 암호화(또는 검증)를 위한 공개 키와 복호화(또는 서명)를 위한 개인 키입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    비대칭 암호학                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  키 생성                                                             │
│  ┌──────────┐                                                       │
│  │ KeyGen() │──▶ Private Key (비밀로 유지!)                         │
│  │          │──▶ Public Key  (자유롭게 공유)                        │
│  └──────────┘                                                       │
│                                                                      │
│  암호화 (누구나 → 키 소유자)                                        │
│  ┌───────────┐   Public Key   ┌──────────┐                          │
│  │ Plaintext │──────────────▶│ Encrypt  │──▶ Ciphertext            │
│  └───────────┘               └──────────┘                          │
│                                                                      │
│  복호화 (키 소유자만)                                               │
│  ┌───────────┐   Private Key  ┌──────────┐                          │
│  │Ciphertext │──────────────▶│ Decrypt  │──▶ Plaintext             │
│  └───────────┘               └──────────┘                          │
│                                                                      │
│  서명 (키 소유자 → 누구나 검증 가능)                               │
│  ┌───────────┐   Private Key  ┌──────────┐                          │
│  │ Message   │──────────────▶│  Sign    │──▶ Signature             │
│  └───────────┘               └──────────┘                          │
│                                                                      │
│  검증 (공개 키를 가진 누구나)                                       │
│  ┌───────────┐   Public Key   ┌──────────┐                          │
│  │ Message + │──────────────▶│ Verify   │──▶ Valid / Invalid       │
│  │ Signature │               └──────────┘                          │
│  └───────────┘                                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.1 하이브리드 암호화

비대칭 암호화는 느리고 암호화할 수 있는 데이터 양이 제한됩니다. 실제로는 **하이브리드 암호화**를 사용합니다: 대칭 키로 데이터를 암호화한 다음 수신자의 공개 키로 대칭 키를 암호화합니다.

```
┌──────────────────────────────────────────────────────────────────┐
│                    하이브리드 암호화                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  발신자:                                                         │
│  1. 랜덤 대칭 키 생성 (예: AES-256)                             │
│  2. 대칭 키로 데이터 암호화 (AES-GCM)                           │
│  3. 수신자의 공개 키로 대칭 키 암호화 (RSA)                     │
│  4. 전송: [encrypted_key] + [encrypted_data]                    │
│                                                                  │
│  수신자:                                                         │
│  1. 개인 키로 대칭 키 복호화 (RSA)                              │
│  2. 대칭 키로 데이터 복호화 (AES-GCM)                           │
│                                                                  │
│  이것은 다음을 제공합니다:                                       │
│  - 대량 데이터를 위한 대칭 암호화의 속도                        │
│  - 키 배포를 위한 공개 키 암호화의 편리함                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. RSA

RSA(Rivest-Shamir-Adleman)는 가장 널리 배포된 비대칭 알고리즘입니다. 보안은 큰 정수를 인수분해하는 어려움에 기반합니다.

### 7.1 RSA 키 생성 및 암호화

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# Generate RSA key pair
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,  # 2048 minimum, 4096 recommended
)
public_key = private_key.public_key()

# Display key info
print(f"Key size: {private_key.key_size} bits")
print(f"Public exponent: {private_key.private_numbers().public_numbers.e}")

# Serialize keys (PEM format)
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.BestAvailableEncryption(b"my-password")
)
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)
print(f"\nPublic key (first 80 chars):\n{public_pem.decode()[:80]}...")

# Encrypt with public key (using OAEP padding - ALWAYS use OAEP, never PKCS1v15)
message = b"Secret message for RSA encryption"
ciphertext = public_key.encrypt(
    message,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
print(f"\nCiphertext: {ciphertext.hex()[:64]}...")
print(f"CT length: {len(ciphertext)} bytes")

# Decrypt with private key
plaintext = private_key.decrypt(
    ciphertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
print(f"Decrypted: {plaintext.decode()}")
```

### 7.2 RSA 하이브리드 암호화

```python
import os
import json
import base64
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class HybridEncryptor:
    """RSA + AES-GCM hybrid encryption."""

    def __init__(self, public_key=None, private_key=None):
        self.public_key = public_key
        self.private_key = private_key

    @classmethod
    def generate_keypair(cls):
        """Generate a new RSA key pair and return an encryptor."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        return cls(
            public_key=private_key.public_key(),
            private_key=private_key,
        )

    def encrypt(self, plaintext: bytes) -> dict:
        """Encrypt data using hybrid encryption."""
        if not self.public_key:
            raise ValueError("Public key required for encryption")

        # 1. Generate random AES-256 key
        aes_key = AESGCM.generate_key(bit_length=256)

        # 2. Encrypt data with AES-GCM
        nonce = os.urandom(12)
        aesgcm = AESGCM(aes_key)
        encrypted_data = aesgcm.encrypt(nonce, plaintext, None)

        # 3. Encrypt AES key with RSA public key
        encrypted_key = self.public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 4. Package everything together
        return {
            "encrypted_key": base64.b64encode(encrypted_key).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(encrypted_data).decode(),
        }

    def decrypt(self, package: dict) -> bytes:
        """Decrypt hybrid-encrypted data."""
        if not self.private_key:
            raise ValueError("Private key required for decryption")

        # 1. Decrypt AES key with RSA private key
        encrypted_key = base64.b64decode(package["encrypted_key"])
        aes_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 2. Decrypt data with AES-GCM
        nonce = base64.b64decode(package["nonce"])
        ciphertext = base64.b64decode(package["ciphertext"])
        aesgcm = AESGCM(aes_key)
        return aesgcm.decrypt(nonce, ciphertext, None)

# Usage
encryptor = HybridEncryptor.generate_keypair()

# Encrypt a large message
large_message = b"A" * 100000  # 100 KB - too large for raw RSA
package = encryptor.encrypt(large_message)

print("Hybrid Encryption Package:")
print(f"  Encrypted key length: {len(base64.b64decode(package['encrypted_key']))} bytes")
print(f"  Nonce: {package['nonce']}")
print(f"  Ciphertext length: {len(base64.b64decode(package['ciphertext']))} bytes")

# Decrypt
decrypted = encryptor.decrypt(package)
assert decrypted == large_message
print(f"\nDecrypted successfully: {len(decrypted)} bytes match original")
```

---

## 8. 타원 곡선 암호학

ECC는 훨씬 작은 키 크기로 RSA와 동일한 보안을 제공하여 더 빠르고 효율적입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│              RSA vs 타원 곡선 키 크기                                │
├──────────────────┬──────────────────┬────────────────────────────────┤
│ 보안 수준        │ RSA 키 크기      │ ECC 키 크기                    │
├──────────────────┼──────────────────┼────────────────────────────────┤
│ 128-bit          │ 3072 bits        │ 256 bits (P-256/secp256r1)     │
│ 192-bit          │ 7680 bits        │ 384 bits (P-384/secp384r1)     │
│ 256-bit          │ 15360 bits       │ 521 bits (P-521/secp521r1)     │
├──────────────────┴──────────────────┴────────────────────────────────┤
│ ECC는 동등한 보안을 위해 ~10-15배 작습니다!                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.1 ECDSA (타원 곡선 디지털 서명 알고리즘)

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

# Generate key pair using P-256 curve (NIST recommended)
private_key = ec.generate_private_key(ec.SECP256R1())
public_key = private_key.public_key()

print(f"Curve: {private_key.curve.name}")
print(f"Key size: {private_key.curve.key_size} bits")

# Sign a message
message = b"This message is signed with ECDSA"
signature = private_key.sign(
    message,
    ec.ECDSA(hashes.SHA256())
)
print(f"\nSignature: {signature.hex()[:64]}...")
print(f"Signature length: {len(signature)} bytes")  # ~70-72 bytes for P-256

# Verify
try:
    public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
    print("Signature valid!")
except Exception:
    print("Signature invalid!")

# Tampered message fails
try:
    public_key.verify(signature, b"tampered message", ec.ECDSA(hashes.SHA256()))
    print("ERROR: Should have failed!")
except Exception:
    print("Tampered message: signature verification failed (expected)")
```

### 8.2 Ed25519 (현대 서명 알고리즘)

Ed25519는 Curve25519를 사용하는 현대 EdDSA 서명 스킴입니다. 결정론적이고(서명 중 랜덤 nonce 불필요), 빠르며, 사이드 채널 공격에 강합니다.

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# Generate Ed25519 key pair
private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Sign (no hash algorithm parameter needed - it uses SHA-512 internally)
message = b"Ed25519 is the recommended signature algorithm for new systems"
signature = private_key.sign(message)

print(f"Signature: {signature.hex()}")
print(f"Signature length: {len(signature)} bytes")  # Always 64 bytes

# Verify
try:
    public_key.verify(signature, message)
    print("Ed25519 signature valid!")
except Exception as e:
    print(f"Invalid: {e}")

# Key sizes are small
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption
)

pub_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
priv_bytes = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
print(f"\nPublic key:  {len(pub_bytes)} bytes ({len(pub_bytes) * 8} bits)")
print(f"Private key: {len(priv_bytes)} bytes ({len(priv_bytes) * 8} bits)")
print(f"Signature:   {len(signature)} bytes ({len(signature) * 8} bits)")
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              서명 알고리즘 비교                                      │
├──────────────┬──────────┬──────────┬──────────┬─────────────────────┤
│              │ RSA-2048 │ECDSA P256│ Ed25519  │ Ed448               │
├──────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ 공개 키 크기 │ 256 B    │ 64 B     │ 32 B     │ 57 B                │
│ 서명 크기    │ 256 B    │ ~72 B    │ 64 B     │ 114 B               │
│ 서명 속도    │ 느림     │ 빠름     │ 매우 빠름│ 빠름                │
│ 검증 속도    │ 빠름     │ 보통     │ 빠름     │ 빠름                │
│ 결정론적     │ No       │ No*      │ Yes      │ Yes                 │
│ 사이드 채널  │ 주의     │ 주의     │ 본질적   │ 본질적              │
│ 저항         │ 필요     │ 필요     │          │                     │
│ 표준         │ PKCS#1   │ FIPS     │ RFC 8032 │ RFC 8032            │
├──────────────┴──────────┴──────────┴──────────┴─────────────────────┤
│ * RFC 6979는 결정론적 ECDSA를 제공하지만 모든 구현이 사용하지 않음 │
│ 권장 사항: 새 시스템에는 Ed25519 사용                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 키 교환

키 교환 프로토콜은 두 당사자가 안전하지 않은 채널을 통해 공유 비밀을 설정할 수 있도록 합니다.

### 9.1 Diffie-Hellman 키 교환

```
┌──────────────────────────────────────────────────────────────────────┐
│               Diffie-Hellman 키 교환                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Alice                                        Bob                   │
│   ─────                                        ───                   │
│                                                                      │
│   1. 개인 키 선택: a                           1. 개인 키 선택: b    │
│   2. 계산: A = g^a mod p                       2. 계산: B = g^b      │
│                                                    mod p             │
│                                                                      │
│   3. A 전송 ──────────────────────────────────▶ A 수신               │
│      B 수신 ◀────────────────────────────────── 4. B 전송            │
│                                                                      │
│   5. 계산:                                     5. 계산:              │
│      shared = B^a mod p                           shared = A^b mod p│
│             = (g^b)^a mod p                              = (g^a)^b  │
│             = g^(ab) mod p                                  mod p   │
│                                                          = g^(ab)   │
│                                                            mod p    │
│                                                                      │
│   두 당사자 모두 동일한 공유 비밀에 도달: g^(ab) mod p              │
│                                                                      │
│   Eve가 보는 것: g, p, A=g^a, B=g^b                                │
│   A로부터 a를 계산하려면 이산 로그 문제를 풀어야 하며               │
│   큰 소수에 대해 계산적으로 불가능하다고 믿어집니다.                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 ECDH (타원 곡선 Diffie-Hellman)

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def ecdh_key_exchange():
    """
    Demonstrate ECDH key exchange between Alice and Bob.
    Uses X25519-equivalent (P-256 shown here for compatibility).
    """
    # Alice generates her key pair
    alice_private = ec.generate_private_key(ec.SECP256R1())
    alice_public = alice_private.public_key()

    # Bob generates his key pair
    bob_private = ec.generate_private_key(ec.SECP256R1())
    bob_public = bob_private.public_key()

    # Alice computes shared secret using her private key + Bob's public key
    alice_shared = alice_private.exchange(ec.ECDH(), bob_public)

    # Bob computes shared secret using his private key + Alice's public key
    bob_shared = bob_private.exchange(ec.ECDH(), alice_public)

    # Both shared secrets are identical
    assert alice_shared == bob_shared
    print(f"ECDH shared secret: {alice_shared.hex()[:32]}...")
    print(f"Shared secret length: {len(alice_shared)} bytes")

    # Derive actual encryption key from shared secret using HKDF
    # (raw ECDH output should NOT be used directly as an encryption key)
    alice_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key for AES-256
        salt=None,
        info=b"ecdh-derived-key-v1",
    ).derive(alice_shared)

    bob_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"ecdh-derived-key-v1",
    ).derive(bob_shared)

    assert alice_key == bob_key
    print(f"Derived AES key: {alice_key.hex()}")

    return alice_key

derived_key = ecdh_key_exchange()
```

### 9.3 X25519 키 교환

X25519는 현대 애플리케이션에 권장되는 ECDH 곡선입니다(TLS 1.3, WireGuard, Signal에서 사용).

```python
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

# Alice
alice_private = X25519PrivateKey.generate()
alice_public = alice_private.public_key()

# Bob
bob_private = X25519PrivateKey.generate()
bob_public = bob_private.public_key()

# Exchange
alice_shared = alice_private.exchange(bob_public)
bob_shared = bob_private.exchange(alice_public)

assert alice_shared == bob_shared
print(f"X25519 shared secret: {alice_shared.hex()}")

# Derive keys using HKDF
def derive_keys(shared_secret: bytes, context: bytes) -> dict:
    """Derive separate keys for encryption and MAC from shared secret."""
    # Encryption key
    enc_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context + b"-enc",
    ).derive(shared_secret)

    # MAC key (for additional message authentication if needed)
    mac_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context + b"-mac",
    ).derive(shared_secret)

    return {"encryption_key": enc_key, "mac_key": mac_key}

keys = derive_keys(alice_shared, b"session-2026-01-15")
print(f"Encryption key: {keys['encryption_key'].hex()}")
print(f"MAC key:        {keys['mac_key'].hex()}")
```

---

## 10. 디지털 서명

디지털 서명은 인증, 무결성 및 부인 방지를 제공합니다. 누구나 메시지가 특정 개인 키의 소유자에 의해 생성되었으며 수정되지 않았음을 확인할 수 있습니다.

### 10.1 디지털 서명 작동 방식

```
┌──────────────────────────────────────────────────────────────────────┐
│                    디지털 서명 프로세스                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  서명 (작성자가 개인 키를 사용하여):                                │
│                                                                      │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐  │
│  │ Message  │────▶│  Hash    │────▶│  Sign    │────▶│ Signature │  │
│  │          │     │ (SHA-256)│     │(Private  │     │           │  │
│  └──────────┘     └──────────┘     │  Key)    │     └───────────┘  │
│                                    └──────────┘                     │
│                                                                      │
│  검증 (작성자의 공개 키를 사용하여 누구나):                         │
│                                                                      │
│  ┌──────────┐     ┌──────────┐                                      │
│  │ Message  │────▶│  Hash    │─────┐                                │
│  └──────────┘     │ (SHA-256)│     │                                │
│                   └──────────┘     ▼                                │
│                                ┌──────────┐     ┌──────────┐       │
│  ┌───────────┐                 │  Verify  │────▶│ Valid /  │       │
│  │ Signature │────────────────▶│(Public   │     │ Invalid  │       │
│  └───────────┘                 │  Key)    │     └──────────┘       │
│                                └──────────┘                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.2 실습: 문서 서명 시스템

```python
import json
import time
import base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption
)
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class SignedDocument:
    """A document with a digital signature."""
    content: str
    author: str
    timestamp: float
    public_key: str  # Base64-encoded public key
    signature: str   # Base64-encoded signature

class DocumentSigner:
    """Sign and verify documents using Ed25519."""

    def __init__(self):
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def get_public_key_b64(self) -> str:
        """Get base64-encoded public key for sharing."""
        raw = self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(raw).decode()

    def sign_document(self, content: str, author: str) -> SignedDocument:
        """Sign a document and return it with the signature."""
        timestamp = time.time()

        # Create canonical message to sign
        message = self._canonical_message(content, author, timestamp)

        # Sign
        signature = self.private_key.sign(message)

        return SignedDocument(
            content=content,
            author=author,
            timestamp=timestamp,
            public_key=self.get_public_key_b64(),
            signature=base64.b64encode(signature).decode(),
        )

    @staticmethod
    def verify_document(doc: SignedDocument) -> dict:
        """Verify a signed document. Returns verification result."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            # Reconstruct the canonical message
            message = DocumentSigner._canonical_message(
                doc.content, doc.author, doc.timestamp
            )

            # Decode public key and signature
            pub_bytes = base64.b64decode(doc.public_key)
            signature = base64.b64decode(doc.signature)

            # Load public key
            public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)

            # Verify
            public_key.verify(signature, message)

            return {
                "valid": True,
                "author": doc.author,
                "signed_at": time.ctime(doc.timestamp),
                "public_key": doc.public_key[:20] + "...",
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
            }

    @staticmethod
    def _canonical_message(content: str, author: str, timestamp: float) -> bytes:
        """Create a canonical byte representation for signing."""
        canonical = json.dumps({
            "content": content,
            "author": author,
            "timestamp": timestamp,
        }, sort_keys=True, separators=(',', ':')).encode()
        return canonical

# Create signers for Alice and Bob
alice_signer = DocumentSigner()
bob_signer = DocumentSigner()

# Alice signs a document
doc = alice_signer.sign_document(
    content="I, Alice, hereby agree to pay Bob $100.",
    author="Alice"
)
print("Signed Document:")
print(f"  Content:   {doc.content}")
print(f"  Author:    {doc.author}")
print(f"  Timestamp: {time.ctime(doc.timestamp)}")
print(f"  Signature: {doc.signature[:40]}...")

# Anyone can verify using Alice's public key (embedded in document)
result = DocumentSigner.verify_document(doc)
print(f"\nVerification: {result}")

# Tamper with the document
tampered_doc = SignedDocument(
    content="I, Alice, hereby agree to pay Bob $10000.",  # Changed!
    author=doc.author,
    timestamp=doc.timestamp,
    public_key=doc.public_key,
    signature=doc.signature,
)
tamper_result = DocumentSigner.verify_document(tampered_doc)
print(f"\nTampered verification: {tamper_result}")
```

---

## 11. 일반적인 함정과 피하는 방법

### 11.1 암호학의 치명적인 죄악

```
┌─────────────────────────────────────────────────────────────────────┐
│          주요 암호학적 함정 (및 피하는 방법)                         │
├────┬────────────────────┬───────────────────────────────────────────┤
│ #  │ 함정               │ 올바른 접근법                             │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 1  │ ECB 모드 사용      │ AES-GCM 또는 ChaCha20-Poly1305 사용      │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 2  │ 동일한 키로        │ 메시지당 랜덤 nonce (또는 카운터)        │
│    │ nonces/IV 재사용   │ 랜덤 안전 nonce를 위해 XChaCha20 사용    │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 3  │ 인증 없이 암호화   │ 항상 AEAD 사용 (GCM, Poly1305)           │
│    │                    │ 원시 CBC/CTR 사용 금지                   │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 4  │ 자체 암호화 구현   │ 확립된 라이브러리 사용 (cryptography,    │
│    │                    │ libsodium/NaCl, OpenSSL)                  │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 5  │ 약하거나 예측      │ os.urandom() 또는 secrets 모듈 사용     │
│    │ 가능한 난수        │ 암호화에 random.random() 절대 사용 금지  │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 6  │ 소스 코드에        │ 키 관리 사용 (AWS KMS, Vault)            │
│    │ 하드코딩된 키      │ 최소한 환경 변수                         │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 7  │ 비밀번호에         │ 해시에는 SHA-256+, 비밀번호에는          │
│    │ MD5/SHA-1 사용     │ bcrypt/argon2 사용                       │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 8  │ PKCS#1 v1.5        │ 암호화에는 RSA-OAEP 사용                 │
│    │ 패딩으로 RSA 사용  │ 서명에는 RSA-PSS 사용                    │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 9  │ ==로 MAC 비교      │ 상수 시간 비교를 위해                    │
│    │                    │ hmac.compare_digest() 사용               │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 10 │ 키 로테이션 없음   │ 키 로테이션 일정 구현                    │
│    │                    │ 키 버전 관리 사용                        │
└────┴────────────────────┴───────────────────────────────────────────┘
```

### 11.2 Nonce 재사용 재앙

```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# Demonstration: Why nonce reuse with CTR/GCM is catastrophic
key = os.urandom(32)
nonce = os.urandom(16)  # Same nonce used twice - BAD!

def ctr_encrypt(key, nonce, plaintext):
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce))
    encryptor = cipher.encryptor()
    return encryptor.update(plaintext) + encryptor.finalize()

# Two messages encrypted with the SAME key and nonce
msg1 = b"Attack at dawn!!!"  # 17 bytes
msg2 = b"Retreat at night!"  # 17 bytes

ct1 = ctr_encrypt(key, nonce, msg1)
ct2 = ctr_encrypt(key, nonce, msg2)

# XOR of two ciphertexts = XOR of two plaintexts!
# In CTR mode: ct = plaintext XOR keystream
# So: ct1 XOR ct2 = (msg1 XOR keystream) XOR (msg2 XOR keystream)
#                  = msg1 XOR msg2  (keystream cancels out!)

xor_result = bytes(a ^ b for a, b in zip(ct1, ct2))
expected = bytes(a ^ b for a, b in zip(msg1, msg2))

print("Nonce Reuse Attack Demonstration:")
print(f"  ct1 XOR ct2:  {xor_result.hex()}")
print(f"  msg1 XOR msg2: {expected.hex()}")
print(f"  Match: {xor_result == expected}")
print()
print("  An attacker who knows msg1 can recover msg2:")
recovered = bytes(a ^ b for a, b in zip(xor_result, msg1))
print(f"  Recovered msg2: {recovered.decode()}")
print()
print("  LESSON: Never reuse a nonce with the same key!")
```

### 11.3 안전한 난수 vs 안전하지 않은 난수

```python
import random
import secrets
import os

# INSECURE - never use for cryptography
# random.random() uses Mersenne Twister (MT19937)
# It is deterministic and predictable if you observe enough outputs
insecure_key = bytes([random.randint(0, 255) for _ in range(32)])
print(f"INSECURE key: {insecure_key.hex()}")
print(f"  Source: random.random() - Mersenne Twister (PREDICTABLE)")

# SECURE - use for cryptography
secure_key = os.urandom(32)
print(f"\nSECURE key:   {secure_key.hex()}")
print(f"  Source: os.urandom() - OS CSPRNG (/dev/urandom)")

# ALSO SECURE - Python 3.6+ secrets module
secure_token = secrets.token_bytes(32)
print(f"\nSECURE key:   {secure_token.hex()}")
print(f"  Source: secrets.token_bytes() - wraps os.urandom()")

# For URL-safe tokens
url_token = secrets.token_urlsafe(32)
print(f"\nURL-safe token: {url_token}")

# For comparison tokens (timing-safe)
a = secrets.token_bytes(32)
b = secrets.token_bytes(32)
print(f"\nConstant-time comparison: {secrets.compare_digest(a, b)}")
```

---

## 12. 현대 권장 사항

### 12.1 알고리즘 선택 가이드 (2025+)

```
┌─────────────────────────────────────────────────────────────────────┐
│               현대 암호학 권장 사항                                  │
├─────────────────┬───────────────────────────────────────────────────┤
│ 사용 사례       │ 권장 알고리즘                                     │
├─────────────────┼───────────────────────────────────────────────────┤
│ 대칭 암호화     │ AES-256-GCM (하드웨어 사용) 또는                 │
│                 │ ChaCha20-Poly1305 (하드웨어 없음 / 모바일)       │
│                 │ 랜덤 nonce 필요 시 XChaCha20-Poly1305            │
├─────────────────┼───────────────────────────────────────────────────┤
│ 키 교환         │ X25519 (Curve25519와 ECDH)                       │
│                 │ ML-KEM (양자 이후, X25519와 하이브리드)          │
├─────────────────┼───────────────────────────────────────────────────┤
│ 디지털 서명     │ Ed25519 (범용)                                   │
│                 │ Ed448 (더 높은 보안 마진)                        │
│                 │ ECDSA P-256 (레거시 호환성)                      │
├─────────────────┼───────────────────────────────────────────────────┤
│ 해싱            │ SHA-256 / SHA-3-256 (범용)                       │
│                 │ BLAKE3 (속도 중요)                               │
├─────────────────┼───────────────────────────────────────────────────┤
│ 비밀번호 해시   │ Argon2id (선호)                                  │
│                 │ bcrypt (널리 지원됨)                             │
│                 │ scrypt (메모리 하드 대안)                        │
├─────────────────┼───────────────────────────────────────────────────┤
│ 키 유도         │ HKDF-SHA256 (고 엔트로피 입력에서)               │
│                 │ Argon2id (비밀번호에서)                          │
├─────────────────┼───────────────────────────────────────────────────┤
│ TLS             │ TLS 1.3 with X25519 + AES-256-GCM                │
│                 │ 또는 ChaCha20-Poly1305                           │
├─────────────────┼───────────────────────────────────────────────────┤
│ 피해야 할 것    │ MD5, SHA-1, DES, 3DES, RC4, RSA-1024,           │
│                 │ ECB 모드, PKCS#1 v1.5, 사용자 정의 알고리즘      │
└─────────────────┴───────────────────────────────────────────────────┘
```

### 12.2 키 크기 권장 사항

```
┌─────────────────────────────────────────────────────────────────────┐
│              최소 키 크기 (NIST / ANSSI 2025+)                       │
├─────────────────────┬───────────────────────────────────────────────┤
│ 알고리즘            │ 최소 키 크기                                  │
├─────────────────────┼───────────────────────────────────────────────┤
│ AES                 │ 128-bit (양자 이후 마진을 위해 256-bit)      │
│ RSA (필요 시)       │ 3072-bit (4096 권장)                         │
│ ECDSA / ECDH        │ P-256 / Curve25519 (256-bit)                │
│ EdDSA               │ Ed25519 (256-bit)                            │
│ 해시 출력           │ 256-bit (SHA-256, SHA-3-256, BLAKE2b-256)   │
│ HMAC 키             │ 해시 출력 크기와 동일 (256-bit)              │
└─────────────────────┴───────────────────────────────────────────────┘
```

### 12.3 양자 이후 암호학

```
┌─────────────────────────────────────────────────────────────────────┐
│              양자 이후 암호학 (PQC)                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  문제: 양자 컴퓨터 (Shor 알고리즘)는 다음을 깨뜨릴 것입니다:        │
│  - RSA (인수분해)                                                   │
│  - ECDSA/ECDH (타원 곡선 이산 로그)                                │
│  - DH (이산 로그)                                                   │
│                                                                      │
│  양자에 영향받지 않음:                                              │
│  - AES (Grover 알고리즘이 유효 키 크기를 절반으로:                 │
│    AES-256 → ~128비트 보안, 여전히 안전)                          │
│  - SHA-256 (유사한 절반, 여전히 적절함)                            │
│                                                                      │
│  NIST PQC 표준 (2024 확정):                                        │
│  ├── ML-KEM (CRYSTALS-Kyber) — 키 캡슐화                          │
│  ├── ML-DSA (CRYSTALS-Dilithium) — 디지털 서명                    │
│  ├── SLH-DSA (SPHINCS+) — 해시 기반 서명                         │
│  └── FN-DSA (FALCON) — 격자 기반 서명                            │
│                                                                      │
│  현재 권장 사항: 하이브리드 모드                                    │
│  - 키 교환에 X25519 + ML-KEM을 함께 사용                          │
│  - 고전 또는 PQ 알고리즘이 깨져도 여전히 안전                      │
│  - Chrome, Firefox, Cloudflare가 이미 하이브리드 PQ 지원          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 13. 연습 문제

### 연습 문제 1: 대칭 암호화 (초급)

다음을 수행하는 Python 함수를 작성하세요:
1. 평문 문자열과 비밀번호를 입력으로 받기
2. PBKDF2를 사용하여 비밀번호로부터 AES-256 키 유도 (랜덤 salt 사용)
3. AES-GCM으로 평문 암호화
4. salt + nonce + ciphertext를 포함하는 단일 base64 인코딩 문자열 반환
5. 해당하는 복호화 함수 작성

힌트:
```python
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# Use iterations=600000, salt_size=16, nonce_size=12
```

### 연습 문제 2: 하이브리드 암호화 (중급)

PGP와 유사한 암호화의 간단한 버전을 구현하세요:
1. Alice가 RSA-4096 키 쌍 생성
2. Bob이 RSA-4096 키 쌍 생성
3. Alice가 Bob에게 서명되고 암호화된 메시지를 보내려고 함:
   - Alice의 개인 키로 메시지 서명
   - 랜덤 AES-256 키 생성
   - AES-GCM으로 메시지 암호화
   - Bob의 공개 RSA 키로 AES 키 암호화
   - 패키지: encrypted_key + nonce + ciphertext + signature + alice_public_key
4. Bob이 패키지를 받아서:
   - 자신의 개인 RSA 키로 AES 키 복호화
   - AES-GCM으로 메시지 복호화
   - Alice의 공개 키를 사용하여 서명 검증

### 연습 문제 3: 키 교환 프로토콜 (중급)

Alice와 Bob 간의 안전한 채팅 시뮬레이션:
1. 두 당사자 모두 X25519 키 교환 수행
2. HKDF를 사용하여 각 방향에 대한 별도 키 유도 (Alice-to-Bob, Bob-to-Alice)
3. 각 메시지는 고유한 nonce를 받음 (카운터 사용)
4. 메시지는 ChaCha20-Poly1305로 암호화됨
5. 재생 공격을 감지하기 위한 메시지 카운터 포함

### 연습 문제 4: Nonce 재사용 공격 (고급)

동일한 키와 nonce를 사용하여 AES-CTR로 암호화된 두 개의 암호문이 주어졌을 때:
```
ct1 = bytes.fromhex("a1b2c3d4e5f6071829")
ct2 = bytes.fromhex("b4a3d2c5f4e7162738")
```
그리고 plaintext1이 `b"plaintext"`임을 알 때:
1. plaintext2 복구
2. 이 공격이 수학적으로 왜 작동하는지 설명
3. 이 취약점을 방지하는 방법 설명

### 연습 문제 5: 디지털 서명 검증 (고급)

간단한 코드 서명 시스템 구축:
1. "발행자"가 Ed25519로 Python 스크립트 서명
2. "실행자"가 스크립트 실행 전에 서명 검증
3. 신뢰할 수 있는 공개 키 레지스트리 유지
4. 키 로테이션 처리 (이전 서명은 이전 키로 여전히 유효해야 함)
5. 타임스탬프 검증 추가 (30일 이상 된 서명 거부)

### 연습 문제 6: 암호학적 감사 (고급)

다음 코드를 검토하고 모든 암호학적 취약점을 식별하세요. 최소 8개의 문제가 있습니다:

```python
import hashlib
import base64
from Crypto.Cipher import AES  # PyCryptodome

def encrypt_message(password, message):
    key = hashlib.md5(password.encode()).digest()  # 128-bit key from MD5
    iv = b'\x00' * 16  # Static IV
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Manual padding
    pad_len = 16 - (len(message) % 16)
    padded = message + chr(pad_len) * pad_len

    encrypted = cipher.encrypt(padded.encode())
    return base64.b64encode(encrypted).decode()

def verify_password(stored_hash, password):
    return hashlib.sha256(password.encode()).hexdigest() == stored_hash
```

각 취약점에 대해 다음을 설명하세요:
- 무엇이 잘못되었는지
- 왜 위험한지
- 어떻게 수정하는지

---

## 14. 참고 자료

- Ferguson, Schneier, Kohno. *Cryptography Engineering*. Wiley, 2010.
- Bernstein, D.J. "Curve25519: New Diffie-Hellman Speed Records". 2006.
- NIST SP 800-175B: Guideline for Using Cryptographic Standards
- NIST Post-Quantum Cryptography - https://csrc.nist.gov/projects/post-quantum-cryptography
- Python `cryptography` library docs - https://cryptography.io/
- Latacora, "Cryptographic Right Answers" (2018, updated regularly)
- RFC 8032: Edwards-Curve Digital Signature Algorithm (EdDSA)
- RFC 8439: ChaCha20 and Poly1305 for IETF Protocols

---

**이전**: [01. Security 기초](./01_Security_Fundamentals.md) | **다음**: [03. Hashing과 데이터 무결성](./03_Hashing_and_Integrity.md)

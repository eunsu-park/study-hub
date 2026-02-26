# Cryptography Basics

**Previous**: [01. Security Fundamentals](./01_Security_Fundamentals.md) | **Next**: [03. Hashing and Data Integrity](./03_Hashing_and_Integrity.md)

---

Cryptography is the practice of securing communication and data so that only intended parties can access it. This lesson covers the two main branches of modern cryptography -- symmetric and asymmetric encryption -- along with key exchange protocols, digital signatures, and practical Python implementations. By the end, you will be able to choose the right cryptographic primitive for a given problem and avoid the most common pitfalls.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the difference between symmetric and asymmetric encryption
2. Implement AES-GCM and ChaCha20-Poly1305 encryption in Python
3. Understand RSA, ECDSA, and Ed25519 for asymmetric cryptography
4. Implement Diffie-Hellman and ECDH key exchange
5. Create and verify digital signatures
6. Recognize and avoid common cryptographic pitfalls
7. Apply modern cryptographic recommendations

---

## Table of Contents

1. [Cryptography Overview](#1-cryptography-overview)
2. [Symmetric Encryption](#2-symmetric-encryption)
3. [Block Cipher Modes of Operation](#3-block-cipher-modes-of-operation)
4. [AES-GCM: The Modern Standard](#4-aes-gcm-the-modern-standard)
5. [ChaCha20-Poly1305](#5-chacha20-poly1305)
6. [Asymmetric Encryption](#6-asymmetric-encryption)
7. [RSA](#7-rsa)
8. [Elliptic Curve Cryptography](#8-elliptic-curve-cryptography)
9. [Key Exchange](#9-key-exchange)
10. [Digital Signatures](#10-digital-signatures)
11. [Common Pitfalls and How to Avoid Them](#11-common-pitfalls-and-how-to-avoid-them)
12. [Modern Recommendations](#12-modern-recommendations)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Cryptography Overview

### 1.1 The Landscape

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Modern Cryptography Taxonomy                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Cryptography                                                        │
│  ├── Symmetric (shared key)                                          │
│  │   ├── Block Ciphers                                               │
│  │   │   ├── AES (128/192/256-bit key)                               │
│  │   │   ├── Modes: ECB, CBC, CTR, GCM, CCM                        │
│  │   │   └── Legacy: DES, 3DES, Blowfish (avoid)                   │
│  │   └── Stream Ciphers                                              │
│  │       ├── ChaCha20(-Poly1305)                                    │
│  │       └── Legacy: RC4 (broken, avoid)                            │
│  │                                                                   │
│  ├── Asymmetric (public/private key pair)                            │
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
│  ├── Hash Functions (covered in Lesson 03)                           │
│  │   ├── SHA-2 (SHA-256, SHA-512)                                   │
│  │   ├── SHA-3 (Keccak)                                             │
│  │   └── BLAKE2/BLAKE3                                              │
│  │                                                                   │
│  └── Key Derivation Functions                                        │
│      ├── HKDF                                                        │
│      ├── PBKDF2                                                      │
│      └── scrypt / Argon2 (password-based)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Concepts

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Core Cryptographic Concepts                       │
├────────────────────┬────────────────────────────────────────────────┤
│ Term               │ Definition                                     │
├────────────────────┼────────────────────────────────────────────────┤
│ Plaintext          │ The original, unencrypted data                 │
│ Ciphertext         │ The encrypted (unreadable) output              │
│ Key                │ Secret value used for encryption/decryption    │
│ Nonce / IV         │ Number used once; prevents identical plaintext │
│                    │ from producing identical ciphertext            │
│ Authenticated enc. │ Encryption that also verifies integrity (AEAD)│
│ Key derivation     │ Deriving a cryptographic key from a password  │
│                    │ or another key                                │
│ Forward secrecy    │ Compromise of long-term keys does not         │
│                    │ compromise past session keys                  │
└────────────────────┴────────────────────────────────────────────────┘
```

### 1.3 Python Cryptography Library Setup

All code examples in this lesson use the `cryptography` library, which provides both high-level recipes and low-level primitives.

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

## 2. Symmetric Encryption

Symmetric encryption uses the same key for both encryption and decryption. It is fast and suitable for encrypting large amounts of data.

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│Plaintext │──Key──▶│ Encrypt  │────────▶│Ciphertext│
│ "Hello"  │         │ (AES)    │         │ 0xA3F1.. │
└──────────┘         └──────────┘         └──────────┘

┌──────────┐         ┌──────────┐         ┌──────────┐
│Ciphertext│──Key──▶│ Decrypt  │────────▶│Plaintext │
│ 0xA3F1.. │         │ (AES)    │         │ "Hello"  │
└──────────┘         └──────────┘         └──────────┘

         Same key used for both operations!
```

### 2.1 AES (Advanced Encryption Standard)

AES is the most widely used symmetric cipher. It operates on 128-bit blocks and supports key sizes of 128, 192, or 256 bits.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AES Key Sizes                                │
├──────────┬──────────┬──────────┬────────────────────────────────────┤
│ Key Size │ Rounds   │ Security │ Use Case                           │
├──────────┼──────────┼──────────┼────────────────────────────────────┤
│ 128-bit  │ 10       │ ~128 bit │ General purpose, fast              │
│ 192-bit  │ 12       │ ~192 bit │ Rarely used in practice            │
│ 256-bit  │ 14       │ ~256 bit │ Government/military, post-quantum  │
│          │          │          │ resistance margin                   │
└──────────┴──────────┴──────────┴────────────────────────────────────┘
```

### 2.2 Fernet: High-Level Symmetric Encryption

For simple use cases, the `Fernet` class provides authenticated encryption with a simple API.

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

**What Fernet does under the hood:**
```
1. Generate random 128-bit IV
2. Encrypt with AES-128-CBC
3. HMAC-SHA256 for authentication
4. Concatenate: version || timestamp || IV || ciphertext || HMAC
5. Base64-encode the result
```

---

## 3. Block Cipher Modes of Operation

A block cipher (like AES) encrypts fixed-size blocks. Modes of operation define how to handle messages longer than one block.

### 3.1 ECB Mode (NEVER Use This)

ECB (Electronic Codebook) encrypts each block independently. Identical plaintext blocks produce identical ciphertext blocks, leaking patterns.

```
┌──────────────────────────────────────────────────────────────────┐
│                    ECB Mode - WHY IT IS BROKEN                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Plaintext blocks:    [AAAA] [BBBB] [AAAA] [CCCC] [AAAA]       │
│                          │      │      │      │      │          │
│                         AES    AES    AES    AES    AES         │
│                          │      │      │      │      │          │
│  Ciphertext blocks:   [X1X1] [Y2Y2] [X1X1] [Z3Z3] [X1X1]     │
│                                                                  │
│  Problem: Identical plaintext blocks → identical ciphertext!    │
│  An attacker can see patterns without decrypting.               │
│                                                                  │
│  Classic example: ECB-encrypted bitmap image preserves shapes   │
│  because adjacent identical pixels produce identical ciphertext.│
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

### 3.2 CBC Mode (Legacy, Use with Care)

CBC (Cipher Block Chaining) XORs each plaintext block with the previous ciphertext block before encryption. Requires a random IV.

```
┌──────────────────────────────────────────────────────────────────┐
│                        CBC Mode                                   │
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
│  Each ciphertext block depends on ALL previous blocks.          │
│  Random IV ensures same plaintext encrypts differently.         │
│                                                                  │
│  ⚠ Requires padding (e.g., PKCS#7)                             │
│  ⚠ Vulnerable to padding oracle attacks if not authenticated    │
│  ⚠ Not parallelizable for encryption                           │
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

### 3.3 CTR Mode

CTR (Counter) mode turns a block cipher into a stream cipher. It is parallelizable and does not require padding.

```
┌──────────────────────────────────────────────────────────────────┐
│                        CTR Mode                                   │
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
│  ✓ Parallelizable (encryption and decryption)                   │
│  ✓ No padding needed                                            │
│  ✓ Can seek to any block                                        │
│  ⚠ Nonce reuse is catastrophic (XOR of two plaintexts leaked)   │
│  ⚠ No authentication (use GCM instead)                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. AES-GCM: The Modern Standard

GCM (Galois/Counter Mode) combines CTR mode encryption with GMAC authentication. It provides both confidentiality and integrity in a single operation -- this is called Authenticated Encryption with Associated Data (AEAD).

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AES-GCM (AEAD)                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input:  Key, Nonce (96-bit), Plaintext, AAD (optional)             │
│  Output: Ciphertext, Authentication Tag (128-bit)                   │
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
│  - Authenticated but NOT encrypted                                  │
│  - Example: message headers, packet sequence numbers                │
│  - Tampering with AAD causes authentication to fail                 │
│                                                                      │
│  ✓ AEAD: Confidentiality + Integrity + Authenticity                │
│  ✓ Parallelizable                                                   │
│  ✓ Hardware acceleration (AES-NI)                                   │
│  ⚠ Nonce MUST be unique per key (never reuse!)                     │
│  ⚠ 96-bit nonce limits to ~2^32 encryptions per key               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 AES-GCM Implementation

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

### 4.2 File Encryption with AES-GCM

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

ChaCha20-Poly1305 is an AEAD cipher that combines the ChaCha20 stream cipher with the Poly1305 MAC. It is the primary alternative to AES-GCM.

```
┌──────────────────────────────────────────────────────────────────────┐
│                  AES-GCM vs ChaCha20-Poly1305                        │
├──────────────────┬──────────────────────┬────────────────────────────┤
│                  │ AES-256-GCM          │ ChaCha20-Poly1305          │
├──────────────────┼──────────────────────┼────────────────────────────┤
│ Key size         │ 256 bits             │ 256 bits                   │
│ Nonce size       │ 96 bits              │ 96 bits                    │
│ Tag size         │ 128 bits             │ 128 bits                   │
│ Speed (HW accel) │ Very fast (AES-NI)   │ Slower with AES-NI         │
│ Speed (software) │ Slower               │ Faster (no special HW)     │
│ Mobile/embedded  │ Needs HW support     │ Excellent (pure software)  │
│ Side channels    │ Needs care (T-tables)│ Inherently constant-time   │
│ Used by          │ TLS, IPsec, disk enc │ TLS (Google/CF), WireGuard│
│ Nonce misuse     │ Catastrophic         │ Catastrophic               │
│ Post-quantum     │ Neither provides PQ  │ Neither provides PQ        │
│                  │ resistance alone     │ resistance alone           │
└──────────────────┴──────────────────────┴────────────────────────────┘
```

### 5.1 ChaCha20-Poly1305 Implementation

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

### 5.2 XChaCha20-Poly1305 (Extended Nonce)

XChaCha20 uses a 192-bit nonce (vs 96-bit), which is large enough to be randomly generated without realistic collision risk. This eliminates the nonce-management burden.

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

## 6. Asymmetric Encryption

Asymmetric (public-key) cryptography uses a key pair: a public key for encryption (or verification) and a private key for decryption (or signing).

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Asymmetric Cryptography                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Key Generation                                                      │
│  ┌──────────┐                                                       │
│  │ KeyGen() │──▶ Private Key (keep secret!)                         │
│  │          │──▶ Public Key  (share freely)                         │
│  └──────────┘                                                       │
│                                                                      │
│  Encryption (anyone → key owner)                                    │
│  ┌───────────┐   Public Key   ┌──────────┐                          │
│  │ Plaintext │──────────────▶│ Encrypt  │──▶ Ciphertext            │
│  └───────────┘               └──────────┘                          │
│                                                                      │
│  Decryption (only key owner)                                        │
│  ┌───────────┐   Private Key  ┌──────────┐                          │
│  │Ciphertext │──────────────▶│ Decrypt  │──▶ Plaintext             │
│  └───────────┘               └──────────┘                          │
│                                                                      │
│  Signing (key owner → anyone can verify)                            │
│  ┌───────────┐   Private Key  ┌──────────┐                          │
│  │ Message   │──────────────▶│  Sign    │──▶ Signature             │
│  └───────────┘               └──────────┘                          │
│                                                                      │
│  Verification (anyone with public key)                              │
│  ┌───────────┐   Public Key   ┌──────────┐                          │
│  │ Message + │──────────────▶│ Verify   │──▶ Valid / Invalid       │
│  │ Signature │               └──────────┘                          │
│  └───────────┘                                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.1 Hybrid Encryption

Asymmetric encryption is slow and limited in the amount of data it can encrypt. In practice, we use **hybrid encryption**: encrypt the data with a symmetric key, then encrypt the symmetric key with the recipient's public key.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Hybrid Encryption                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sender:                                                        │
│  1. Generate random symmetric key (e.g., AES-256)               │
│  2. Encrypt data with symmetric key (AES-GCM)                   │
│  3. Encrypt symmetric key with recipient's public key (RSA)     │
│  4. Send: [encrypted_key] + [encrypted_data]                    │
│                                                                  │
│  Recipient:                                                      │
│  1. Decrypt symmetric key with private key (RSA)                │
│  2. Decrypt data with symmetric key (AES-GCM)                   │
│                                                                  │
│  This gives us:                                                  │
│  - Speed of symmetric encryption for bulk data                  │
│  - Convenience of public-key encryption for key distribution    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. RSA

RSA (Rivest-Shamir-Adleman) is the most widely deployed asymmetric algorithm. Its security is based on the difficulty of factoring large integers.

### 7.1 RSA Key Generation and Encryption

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

### 7.2 RSA Hybrid Encryption

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

## 8. Elliptic Curve Cryptography

ECC provides the same security as RSA with much smaller key sizes, making it faster and more efficient.

```
┌─────────────────────────────────────────────────────────────────────┐
│              RSA vs Elliptic Curve Key Sizes                         │
├──────────────────┬──────────────────┬────────────────────────────────┤
│ Security Level   │ RSA Key Size     │ ECC Key Size                   │
├──────────────────┼──────────────────┼────────────────────────────────┤
│ 128-bit          │ 3072 bits        │ 256 bits (P-256/secp256r1)     │
│ 192-bit          │ 7680 bits        │ 384 bits (P-384/secp384r1)     │
│ 256-bit          │ 15360 bits       │ 521 bits (P-521/secp521r1)     │
├──────────────────┴──────────────────┴────────────────────────────────┤
│ ECC is ~10-15x smaller for equivalent security!                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.1 ECDSA (Elliptic Curve Digital Signature Algorithm)

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

### 8.2 Ed25519 (Modern Signature Algorithm)

Ed25519 is a modern EdDSA signature scheme using Curve25519. It is deterministic (no random nonce needed during signing), fast, and resistant to side-channel attacks.

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
│              Signature Algorithm Comparison                          │
├──────────────┬──────────┬──────────┬──────────┬─────────────────────┤
│              │ RSA-2048 │ECDSA P256│ Ed25519  │ Ed448               │
├──────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ Pub key size │ 256 B    │ 64 B     │ 32 B     │ 57 B                │
│ Sig size     │ 256 B    │ ~72 B    │ 64 B     │ 114 B               │
│ Sign speed   │ Slow     │ Fast     │ Very fast│ Fast                │
│ Verify speed │ Fast     │ Moderate │ Fast     │ Fast                │
│ Deterministic│ No       │ No*      │ Yes      │ Yes                 │
│ Side-channel │ Needs    │ Needs    │ Inherent │ Inherent            │
│ resistance   │ care     │ care     │          │                     │
│ Standard     │ PKCS#1   │ FIPS     │ RFC 8032 │ RFC 8032            │
├──────────────┴──────────┴──────────┴──────────┴─────────────────────┤
│ * RFC 6979 provides deterministic ECDSA, but not all impls use it  │
│ Recommendation: Use Ed25519 for new systems                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Exchange

Key exchange protocols allow two parties to establish a shared secret over an insecure channel.

### 9.1 Diffie-Hellman Key Exchange

```
┌──────────────────────────────────────────────────────────────────────┐
│               Diffie-Hellman Key Exchange                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Alice                                        Bob                   │
│   ─────                                        ───                   │
│                                                                      │
│   1. Choose private key: a                     1. Choose private: b  │
│   2. Compute: A = g^a mod p                    2. Compute: B = g^b   │
│                                                    mod p             │
│                                                                      │
│   3. Send A ──────────────────────────────────▶ Receive A            │
│      Receive B ◀────────────────────────────── 4. Send B             │
│                                                                      │
│   5. Compute:                                  5. Compute:           │
│      shared = B^a mod p                           shared = A^b mod p │
│             = (g^b)^a mod p                              = (g^a)^b   │
│             = g^(ab) mod p                                  mod p    │
│                                                          = g^(ab)    │
│                                                            mod p     │
│                                                                      │
│   Both arrive at the SAME shared secret: g^(ab) mod p               │
│                                                                      │
│   Eve sees: g, p, A=g^a, B=g^b                                     │
│   Computing a from A requires solving the Discrete Logarithm        │
│   Problem -- believed to be computationally infeasible for          │
│   large primes.                                                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 ECDH (Elliptic Curve Diffie-Hellman)

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

### 9.3 X25519 Key Exchange

X25519 is the recommended ECDH curve for modern applications (used in TLS 1.3, WireGuard, Signal).

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

## 10. Digital Signatures

Digital signatures provide authentication, integrity, and non-repudiation. They allow anyone to verify that a message was created by the holder of a specific private key and has not been modified.

### 10.1 How Digital Signatures Work

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Digital Signature Process                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SIGNING (by the author, using their private key):                  │
│                                                                      │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐  │
│  │ Message  │────▶│  Hash    │────▶│  Sign    │────▶│ Signature │  │
│  │          │     │ (SHA-256)│     │(Private  │     │           │  │
│  └──────────┘     └──────────┘     │  Key)    │     └───────────┘  │
│                                    └──────────┘                     │
│                                                                      │
│  VERIFICATION (by anyone, using the author's public key):           │
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

### 10.2 Practical: Document Signing System

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

## 11. Common Pitfalls and How to Avoid Them

### 11.1 The Deadly Sins of Cryptography

```
┌─────────────────────────────────────────────────────────────────────┐
│          Top Cryptographic Pitfalls (and How to Avoid Them)          │
├────┬────────────────────┬───────────────────────────────────────────┤
│ #  │ Pitfall            │ Correct Approach                          │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 1  │ Using ECB mode     │ Use AES-GCM or ChaCha20-Poly1305         │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 2  │ Reusing nonces/IVs │ Random nonce per message (or counter)     │
│    │ with the same key  │ Use XChaCha20 for random-safe nonces     │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 3  │ Encrypt without    │ Always use AEAD (GCM, Poly1305)          │
│    │ authenticating     │ Never use raw CBC/CTR                     │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 4  │ Rolling your own   │ Use established libraries (cryptography,  │
│    │ crypto             │ libsodium/NaCl, OpenSSL)                  │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 5  │ Weak/predictable   │ Use os.urandom() or secrets module       │
│    │ random numbers     │ NEVER use random.random() for crypto     │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 6  │ Hardcoded keys     │ Use key management (AWS KMS, Vault)      │
│    │ in source code     │ Environment variables at minimum          │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 7  │ Using MD5/SHA-1    │ Use SHA-256+ for hashes, bcrypt/argon2   │
│    │ for passwords      │ for passwords                             │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 8  │ RSA with PKCS#1    │ Use RSA-OAEP for encryption              │
│    │ v1.5 padding       │ Use RSA-PSS for signatures               │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 9  │ Comparing MACs     │ Use hmac.compare_digest() for            │
│    │ with ==            │ constant-time comparison                   │
├────┼────────────────────┼───────────────────────────────────────────┤
│ 10 │ Not rotating keys  │ Implement key rotation schedules         │
│    │                    │ Use key versioning                        │
└────┴────────────────────┴───────────────────────────────────────────┘
```

### 11.2 Nonce Reuse Disaster

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

### 11.3 Secure vs Insecure Random

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

## 12. Modern Recommendations

### 12.1 Algorithm Selection Guide (2025+)

```
┌─────────────────────────────────────────────────────────────────────┐
│               Modern Cryptographic Recommendations                   │
├─────────────────┬───────────────────────────────────────────────────┤
│ Use Case        │ Recommended Algorithm(s)                          │
├─────────────────┼───────────────────────────────────────────────────┤
│ Symmetric enc.  │ AES-256-GCM (with hardware) or                   │
│                 │ ChaCha20-Poly1305 (without hardware / mobile)     │
│                 │ XChaCha20-Poly1305 if random nonces needed       │
├─────────────────┼───────────────────────────────────────────────────┤
│ Key exchange    │ X25519 (ECDH with Curve25519)                    │
│                 │ ML-KEM (post-quantum, hybrid with X25519)        │
├─────────────────┼───────────────────────────────────────────────────┤
│ Digital sig.    │ Ed25519 (general purpose)                        │
│                 │ Ed448 (higher security margin)                   │
│                 │ ECDSA P-256 (legacy compatibility)               │
├─────────────────┼───────────────────────────────────────────────────┤
│ Hashing         │ SHA-256 / SHA-3-256 (general)                    │
│                 │ BLAKE3 (speed-critical)                          │
├─────────────────┼───────────────────────────────────────────────────┤
│ Password hash   │ Argon2id (preferred)                             │
│                 │ bcrypt (widely supported)                        │
│                 │ scrypt (memory-hard alternative)                 │
├─────────────────┼───────────────────────────────────────────────────┤
│ Key derivation  │ HKDF-SHA256 (from high-entropy input)            │
│                 │ Argon2id (from passwords)                        │
├─────────────────┼───────────────────────────────────────────────────┤
│ TLS             │ TLS 1.3 with X25519 + AES-256-GCM               │
│                 │ or ChaCha20-Poly1305                              │
├─────────────────┼───────────────────────────────────────────────────┤
│ AVOID           │ MD5, SHA-1, DES, 3DES, RC4, RSA-1024,           │
│                 │ ECB mode, PKCS#1 v1.5, custom algorithms        │
└─────────────────┴───────────────────────────────────────────────────┘
```

### 12.2 Key Size Recommendations

```
┌─────────────────────────────────────────────────────────────────────┐
│              Minimum Key Sizes (NIST / ANSSI 2025+)                  │
├─────────────────────┬───────────────────────────────────────────────┤
│ Algorithm           │ Minimum Key Size                              │
├─────────────────────┼───────────────────────────────────────────────┤
│ AES                 │ 128-bit (256-bit for post-quantum margin)    │
│ RSA (if you must)   │ 3072-bit (4096 recommended)                  │
│ ECDSA / ECDH        │ P-256 / Curve25519 (256-bit)                │
│ EdDSA               │ Ed25519 (256-bit)                            │
│ Hash output         │ 256-bit (SHA-256, SHA-3-256, BLAKE2b-256)   │
│ HMAC key            │ Same as hash output size (256-bit)           │
└─────────────────────┴───────────────────────────────────────────────┘
```

### 12.3 Post-Quantum Cryptography

```
┌─────────────────────────────────────────────────────────────────────┐
│              Post-Quantum Cryptography (PQC)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem: Quantum computers (Shor's algorithm) will break:          │
│  - RSA (factoring)                                                  │
│  - ECDSA/ECDH (elliptic curve discrete log)                        │
│  - DH (discrete log)                                                │
│                                                                      │
│  NOT affected by quantum:                                            │
│  - AES (Grover's algorithm halves effective key size:               │
│    AES-256 → ~128-bit security, still safe)                        │
│  - SHA-256 (similar halving, still adequate)                        │
│                                                                      │
│  NIST PQC Standards (finalized 2024):                               │
│  ├── ML-KEM (CRYSTALS-Kyber) — key encapsulation                   │
│  ├── ML-DSA (CRYSTALS-Dilithium) — digital signatures              │
│  ├── SLH-DSA (SPHINCS+) — hash-based signatures                   │
│  └── FN-DSA (FALCON) — lattice-based signatures                   │
│                                                                      │
│  Current recommendation: Hybrid mode                                │
│  - Use X25519 + ML-KEM together for key exchange                   │
│  - If classical OR PQ algorithm is broken, you are still safe      │
│  - Chrome, Firefox, and Cloudflare already support hybrid PQ       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 13. Exercises

### Exercise 1: Symmetric Encryption (Beginner)

Write a Python function that:
1. Takes a plaintext string and a password as input
2. Derives an AES-256 key from the password using PBKDF2 (with a random salt)
3. Encrypts the plaintext with AES-GCM
4. Returns a single base64-encoded string containing: salt + nonce + ciphertext
5. Write the corresponding decryption function

Hints:
```python
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# Use iterations=600000, salt_size=16, nonce_size=12
```

### Exercise 2: Hybrid Encryption (Intermediate)

Implement a simplified version of PGP-like encryption:
1. Alice generates an RSA-4096 key pair
2. Bob generates an RSA-4096 key pair
3. Alice wants to send a signed and encrypted message to Bob:
   - Sign the message with Alice's private key
   - Generate a random AES-256 key
   - Encrypt the message with AES-GCM
   - Encrypt the AES key with Bob's public RSA key
   - Package: encrypted_key + nonce + ciphertext + signature + alice_public_key
4. Bob receives the package and:
   - Decrypts the AES key with his private RSA key
   - Decrypts the message with AES-GCM
   - Verifies the signature using Alice's public key

### Exercise 3: Key Exchange Protocol (Intermediate)

Simulate a secure chat between Alice and Bob:
1. Both perform X25519 key exchange
2. Derive separate keys for each direction (Alice-to-Bob, Bob-to-Alice) using HKDF
3. Each message gets a unique nonce (use a counter)
4. Messages are encrypted with ChaCha20-Poly1305
5. Include a message counter to detect replay attacks

### Exercise 4: Nonce Reuse Attack (Advanced)

Given two ciphertexts encrypted with AES-CTR using the same key and nonce:
```
ct1 = bytes.fromhex("a1b2c3d4e5f6071829")
ct2 = bytes.fromhex("b4a3d2c5f4e7162738")
```
And knowing that plaintext1 is `b"plaintext"`:
1. Recover plaintext2
2. Explain why this attack works mathematically
3. Describe how to prevent this vulnerability

### Exercise 5: Digital Signature Verification (Advanced)

Build a simple code-signing system:
1. A "publisher" signs Python scripts with Ed25519
2. A "runner" verifies signatures before executing scripts
3. Maintain a registry of trusted public keys
4. Handle key rotation (old signatures should remain valid with old keys)
5. Add timestamp verification (reject signatures older than 30 days)

### Exercise 6: Cryptographic Audit (Advanced)

Review the following code and identify ALL cryptographic vulnerabilities. There are at least 8 issues:

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

For each vulnerability, explain:
- What is wrong
- Why it is dangerous
- How to fix it

---

## References

- Ferguson, Schneier, Kohno. *Cryptography Engineering*. Wiley, 2010.
- Bernstein, D.J. "Curve25519: New Diffie-Hellman Speed Records". 2006.
- NIST SP 800-175B: Guideline for Using Cryptographic Standards
- NIST Post-Quantum Cryptography - https://csrc.nist.gov/projects/post-quantum-cryptography
- Python `cryptography` library docs - https://cryptography.io/
- Latacora, "Cryptographic Right Answers" (2018, updated regularly)
- RFC 8032: Edwards-Curve Digital Signature Algorithm (EdDSA)
- RFC 8439: ChaCha20 and Poly1305 for IETF Protocols

---

**Previous**: [01. Security Fundamentals](./01_Security_Fundamentals.md) | **Next**: [03. Hashing and Data Integrity](./03_Hashing_and_Integrity.md)

# Lesson 3: Block Cipher Modes of Operation

**Previous**: [Symmetric Ciphers](./02_Symmetric_Ciphers.md) | **Next**: [Hash Functions](./04_Hash_Functions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why ECB mode is insecure and demonstrate the "penguin problem" visually
2. Describe the operation of CBC, CTR, and GCM modes and their respective trade-offs
3. Implement CTR mode encryption from a block cipher primitive
4. Explain authenticated encryption and why encrypt-then-MAC is preferred
5. Describe the padding oracle attack and why PKCS#7 padding can be dangerous
6. Identify the catastrophic consequences of nonce/IV reuse in stream-like modes

---

A block cipher like AES encrypts fixed-size blocks (128 bits for AES). Real messages are almost never exactly 128 bits. Block cipher **modes of operation** define how to use a block cipher to securely encrypt messages of arbitrary length. Choosing the wrong mode — or misusing the right one — has led to devastating real-world vulnerabilities, from the BEAST attack on TLS to the PlayStation Network breach. This lesson explains each major mode, why it exists, and how to use it safely.

## Table of Contents

1. [The Problem: Encrypting Beyond One Block](#1-the-problem-encrypting-beyond-one-block)
2. [ECB Mode — The Wrong Answer](#2-ecb-mode--the-wrong-answer)
3. [CBC Mode](#3-cbc-mode)
4. [Padding and PKCS#7](#4-padding-and-pkcs7)
5. [The Padding Oracle Attack](#5-the-padding-oracle-attack)
6. [CTR Mode](#6-ctr-mode)
7. [Authenticated Encryption](#7-authenticated-encryption)
8. [GCM Mode](#8-gcm-mode)
9. [Mode Comparison and Selection Guide](#9-mode-comparison-and-selection-guide)
10. [Exercises](#10-exercises)

---

## 1. The Problem: Encrypting Beyond One Block

AES encrypts exactly 16 bytes at a time. If your message is 100 bytes, you need to:

1. Decide how to split the message into blocks
2. Decide whether each block is encrypted independently or linked to previous blocks
3. Handle messages whose length is not a multiple of the block size (padding)
4. Ensure that identical plaintext blocks do not produce identical ciphertext blocks

These decisions constitute the **mode of operation**. Getting them wrong can completely negate the security of even a perfect block cipher.

---

## 2. ECB Mode — The Wrong Answer

### 2.1 How ECB Works

**Electronic Codebook (ECB)** mode encrypts each block independently:

$$C_i = E_K(P_i)$$

```
P_1 → [AES_K] → C_1
P_2 → [AES_K] → C_2
P_3 → [AES_K] → C_3
...
```

> **Analogy:** ECB is like using the same disguise for every spy — if two spies look alike (identical plaintext blocks), their disguises will be identical too. An observer can immediately spot the pattern.

### 2.2 The Penguin Problem

The most famous demonstration of ECB's weakness: encrypt a bitmap image with ECB mode. Because identical pixel blocks produce identical ciphertext blocks, the outline of the original image remains clearly visible.

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

### 2.3 When ECB Is (Marginally) Acceptable

ECB is acceptable **only** when encrypting a single block (e.g., encrypting a single AES key). For **any** multi-block message, ECB must never be used.

---

## 3. CBC Mode

### 3.1 How CBC Works

**Cipher Block Chaining (CBC)** XORs each plaintext block with the previous ciphertext block before encryption:

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

Decryption:

$$P_i = D_K(C_i) \oplus C_{i-1}$$

> **Analogy:** CBC chains disguises together — each spy's disguise is influenced by the previous spy's disguise. Even if two spies look alike, their disguises will differ because they are in different positions in the chain.

### 3.2 The Initialization Vector (IV)

The IV serves as $C_0$ and must be:
- **Unpredictable** (random for each encryption)
- **Unique** per message
- **Not secret** — it is typically prepended to the ciphertext

**Why unpredictable?** If an attacker can predict the IV, they can mount a chosen-plaintext attack (the BEAST attack on TLS 1.0 exploited predictable IVs).

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

### 3.3 CBC Limitations

- **Sequential encryption:** Cannot parallelize (each block depends on the previous)
- **Error propagation:** A bit error in $C_i$ corrupts all of $P_i$ and one bit of $P_{i+1}$
- **Requires padding:** Message must be padded to a multiple of the block size
- **No integrity protection:** An attacker can modify ciphertext without detection (see Section 5)

---

## 4. Padding and PKCS#7

### 4.1 The Problem

If the last plaintext block is shorter than 16 bytes, we must pad it. The padding must be **unambiguous** — the receiver must be able to remove it reliably.

### 4.2 PKCS#7 Padding

Pad with $n$ bytes, each with value $n$:

- 1 byte short: pad with `01`
- 2 bytes short: pad with `02 02`
- 3 bytes short: pad with `03 03 03`
- Full block: add entire block of `10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10` (16 = 0x10)

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

## 5. The Padding Oracle Attack

### 5.1 The Vulnerability

In 2002, Serge Vaudenay discovered that if a decryption system reveals whether the padding is valid (a **padding oracle**), an attacker can decrypt the entire ciphertext **without knowing the key**.

The oracle can be as subtle as:
- A different HTTP error code (400 vs 500)
- A different response time (padding check fails before or after MAC check)
- A different error message

### 5.2 How It Works (Simplified)

Recall CBC decryption: $P_i = D_K(C_i) \oplus C_{i-1}$.

The attacker modifies $C_{i-1}$ byte by byte and submits the modified ciphertext to the oracle. When the oracle reports valid padding, the attacker can deduce $D_K(C_i)$ one byte at a time.

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

### 5.3 Countermeasures

1. **Use authenticated encryption** (GCM, CCM) — the best solution
2. **Always verify MAC before decrypting** (encrypt-then-MAC)
3. **Constant-time comparison** — never reveal padding validity through timing
4. **TLS 1.3** removed CBC cipher suites entirely

---

## 6. CTR Mode

### 6.1 How CTR Works

**Counter (CTR)** mode turns a block cipher into a stream cipher. It encrypts successive counter values and XORs the result with the plaintext:

$$C_i = P_i \oplus E_K(\text{Nonce} \| \text{Counter}_i)$$

```
Nonce||0 → [AES_K] → Keystream_0 ⊕ P_0 → C_0
Nonce||1 → [AES_K] → Keystream_1 ⊕ P_1 → C_1
Nonce||2 → [AES_K] → Keystream_2 ⊕ P_2 → C_2
```

Decryption is identical to encryption (XOR is its own inverse).

### 6.2 Advantages of CTR Mode

| Advantage | Explanation |
|-----------|-------------|
| **Parallelizable** | All blocks can be encrypted/decrypted simultaneously |
| **Random access** | Can decrypt block $i$ without decrypting blocks $0$ to $i-1$ |
| **No padding needed** | XOR with keystream — works on any length |
| **Pre-computable** | Keystream can be generated before plaintext is known |
| **Same code for encrypt/decrypt** | XOR is symmetric |

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

### 6.3 Nonce Reuse: A Catastrophic Failure

If the same nonce is used with the same key for two different messages:

$$C_1 = P_1 \oplus \text{Keystream}$$
$$C_2 = P_2 \oplus \text{Keystream}$$

Then:

$$C_1 \oplus C_2 = P_1 \oplus P_2$$

The keystream cancels out, and the attacker obtains the XOR of the two plaintexts. For natural language text, this is easily exploitable via crib dragging.

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

## 7. Authenticated Encryption

### 7.1 The Problem with Encryption Alone

Encryption provides **confidentiality** but NOT **integrity**. An attacker can modify ciphertext, and the receiver will decrypt it to a different (attacker-chosen) plaintext without detecting the tampering.

### 7.2 Three Approaches

| Approach | Construction | Security |
|----------|-------------|----------|
| **Encrypt-and-MAC** | $C = E_K(P)$, $T = MAC_{K'}(P)$ | Insecure (MAC may leak plaintext info) |
| **MAC-then-Encrypt** | $T = MAC_{K'}(P)$, $C = E_K(P \| T)$ | Fragile (padding oracle on MAC) |
| **Encrypt-then-MAC** | $C = E_K(P)$, $T = MAC_{K'}(C)$ | **Secure** (standard recommendation) |

> **Analogy:** Encrypt-then-MAC is like sealing a letter in an envelope (encryption) and then putting a tamper-evident seal on the outside (MAC). Anyone who modifies the envelope breaks the seal, and the receiver rejects the message before even opening it.

### 7.3 AEAD: Authenticated Encryption with Associated Data

Modern ciphers combine encryption and authentication in a single operation. **AEAD** also authenticates additional data (headers, metadata) that is transmitted in plaintext but must not be tampered with.

$$\text{AEAD}(K, N, P, A) \rightarrow (C, T)$$

Where:
- $K$ = key
- $N$ = nonce
- $P$ = plaintext (encrypted and authenticated)
- $A$ = associated data (authenticated but not encrypted)
- $C$ = ciphertext
- $T$ = authentication tag

---

## 8. GCM Mode

### 8.1 Galois/Counter Mode

**GCM** combines CTR mode for encryption with a polynomial hash (GHASH) for authentication. It is the most widely used AEAD mode in TLS 1.2 and 1.3.

```
Counter Mode (Encryption):
Nonce||1 → [AES_K] → ⊕ P_1 → C_1
Nonce||2 → [AES_K] → ⊕ P_2 → C_2

GHASH (Authentication):
(A_1 ⊕ H) · H → (... ⊕ C_1) · H → (... ⊕ C_2) · H → (... ⊕ len) · H → GHASH

Tag:
T = GHASH ⊕ E_K(Nonce||0)
```

### 8.2 GHASH: The Authentication Component

GHASH is a polynomial evaluation over $GF(2^{128})$:

$$\text{GHASH}(H, A, C) = X_{m+n+1}$$

where $H = E_K(0^{128})$ is the hash key and:

$$X_i = (X_{i-1} \oplus B_i) \cdot H$$

with $B_i$ being the blocks of associated data, ciphertext, and a length block.

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

### 8.3 GCM Security Requirements

1. **96-bit nonce** (recommended): Unique per encryption with the same key
2. **Nonce reuse is catastrophic:** Reveals the GHASH key $H$, allowing forgery of authentication tags
3. **Maximum message size:** $2^{39} - 256$ bits per message
4. **Maximum messages per key:** $2^{32}$ (with random nonces) to keep collision probability below $2^{-32}$

### 8.4 Other AEAD Modes

| Mode | Structure | Notes |
|------|-----------|-------|
| **GCM** | CTR + GHASH | Most common; hardware-accelerated |
| **CCM** | CTR + CBC-MAC | Used in WiFi (WPA2), Bluetooth |
| **ChaCha20-Poly1305** | ChaCha20 + Poly1305 | Software-friendly; used in TLS when AES-NI unavailable |
| **OCB** | Parallel encrypt + authenticate | Fastest; patent issues (now free for open source) |
| **SIV** | Synthetic IV | Nonce-misuse resistant (degrades gracefully) |

---

## 9. Mode Comparison and Selection Guide

### 9.1 Summary Table

| Mode | Confidentiality | Integrity | Parallelizable | Padding | Nonce Reuse Impact |
|------|----------------|-----------|----------------|---------|-------------------|
| ECB | Weak | No | Yes | Required | N/A (no nonce) |
| CBC | Strong | No | Decrypt only | Required | IV predictability → CPA |
| CTR | Strong | No | Yes | Not needed | Total break (keystream reuse) |
| GCM | Strong | **Yes** | Yes | Not needed | Total break (key + forgery) |
| CCM | Strong | **Yes** | No | Not needed | Total break |

### 9.2 Decision Flowchart

```
Need encryption?
├── Yes → Need authentication too?
│         ├── Yes → Use GCM (or ChaCha20-Poly1305)
│         └── No  → Are you sure? (You almost certainly need authentication)
│                   └── Use CTR if you have a separate MAC, otherwise use GCM
└── No  → Need authentication only? → Use HMAC (see Lesson 4)
```

### 9.3 Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Using ECB | Pattern leakage | Use GCM |
| Reusing nonce in CTR/GCM | Keystream reuse, forgery | Use random nonce + counter; rotate keys |
| CBC without MAC | Padding oracle, bit flipping | Use GCM (or encrypt-then-MAC) |
| Predictable CBC IV | Chosen plaintext attack (BEAST) | Random IV per message |
| MAC-then-encrypt | Padding oracle despite MAC | Encrypt-then-MAC or AEAD |

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

## 10. Exercises

### Exercise 1: ECB Pattern Detection (Basic)

Create a list of 10 plaintext blocks where 4 blocks are identical. Encrypt with ECB and CBC modes. Count the number of unique ciphertext blocks in each case. Explain why they differ.

### Exercise 2: CTR Mode Implementation (Intermediate)

Implement CTR mode encryption using Python's `cryptography` library for the actual AES block cipher. Encrypt a message, then decrypt it by re-encrypting (since CTR decryption equals encryption). Verify the round-trip.

### Exercise 3: Nonce Reuse Attack (Intermediate)

1. Encrypt two known messages with CTR mode using the same nonce.
2. Given only the two ciphertexts, recover the XOR of the two plaintexts.
3. If you know one plaintext, show how to recover the other.
4. Discuss: how does this relate to the one-time pad and why nonce reuse is equivalent to key reuse?

### Exercise 4: Padding Oracle Simulator (Challenging)

Implement a complete padding oracle attack:
1. Create an "oracle server" that decrypts CBC ciphertext and returns True/False for valid padding.
2. Write an attacker that, using only the oracle, recovers the plaintext byte by byte.
3. Count the total number of oracle queries needed to decrypt a 3-block message.
4. Verify that the expected number is approximately $3 \times 16 \times 128 = 6144$ queries.

### Exercise 5: GCM Tag Forgery Under Nonce Reuse (Challenging)

Research and explain (with equations) how an attacker who observes two GCM encryptions with the same nonce can:
1. Recover the GHASH key $H$
2. Forge valid authentication tags for arbitrary messages
3. Why does SIV mode (AES-GCM-SIV) degrade gracefully under nonce reuse instead of catastrophically failing?

---

**Previous**: [Symmetric Ciphers](./02_Symmetric_Ciphers.md) | **Next**: [Hash Functions](./04_Hash_Functions.md)

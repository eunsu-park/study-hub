# Lesson 2: Symmetric Ciphers

**Previous**: [Number Theory Foundations](./01_Number_Theory_Foundations.md) | **Next**: [Block Cipher Modes](./03_Block_Cipher_Modes.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Kerckhoffs' principle and its implications for cipher design
2. Describe the Feistel network structure and implement a simplified Feistel cipher
3. Trace the four AES operations (SubBytes, ShiftRows, MixColumns, AddRoundKey) on sample data
4. Explain how confusion and diffusion provide security in block ciphers
5. Compare DES and AES in terms of structure, key size, and security
6. Identify common attacks on symmetric ciphers (brute force, differential, linear cryptanalysis)

---

Symmetric ciphers are the workhorses of modern cryptography. When you stream a video, send a message, or access your bank, the bulk data encryption is almost always handled by a symmetric cipher like AES — not by RSA or ECC, which are too slow for large data. Understanding how AES works internally reveals why it has withstood over 25 years of cryptanalytic scrutiny, and why earlier ciphers like DES fell to advancing technology.

## Table of Contents

1. [Historical Ciphers](#1-historical-ciphers)
2. [Kerckhoffs' Principle](#2-kerckhoffs-principle)
3. [Confusion and Diffusion](#3-confusion-and-diffusion)
4. [Feistel Networks](#4-feistel-networks)
5. [DES: The Data Encryption Standard](#5-des-the-data-encryption-standard)
6. [AES: The Advanced Encryption Standard](#6-aes-the-advanced-encryption-standard)
7. [AES Internals Step by Step](#7-aes-internals-step-by-step)
8. [Security Analysis](#8-security-analysis)
9. [Exercises](#9-exercises)

---

## 1. Historical Ciphers

### 1.1 Caesar Cipher

The simplest substitution cipher: shift each letter by a fixed amount.

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

### 1.2 Vigenere Cipher

A polyalphabetic cipher using a keyword to determine the shift for each position:

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

### 1.3 Lessons from History

| Cipher | Key Space | Attack | Lesson |
|--------|-----------|--------|--------|
| Caesar | 26 | Brute force | Key space too small |
| Substitution | $26!$ | Frequency analysis | Preserving letter patterns is fatal |
| Vigenere | $26^k$ | Kasiski + frequency | Repeating key creates patterns |
| One-Time Pad | $26^n$ ($n$ = msg length) | **Unbreakable** (if used correctly) | Perfect secrecy requires key as long as message |

The **one-time pad** achieves perfect secrecy (Shannon, 1949) but is impractical because the key must be as long as the message and never reused. Modern ciphers aim for **computational security**: breaking them is infeasible given realistic computational resources.

---

## 2. Kerckhoffs' Principle

> **Kerckhoffs' Principle (1883):** A cryptosystem should be secure even if everything about the system, except the key, is public knowledge.

Modern formulation: **The security of a cipher must depend solely on the secrecy of the key, not on the secrecy of the algorithm.**

**Why this matters:**
- Algorithms get reverse-engineered, leaked, or published
- Public algorithms receive more scrutiny and are more trustworthy
- Secret algorithms create a false sense of security ("security through obscurity")
- AES, ChaCha20, and all modern ciphers are fully public specifications

**Contrast with historical practice:** Many historical ciphers relied on keeping the algorithm secret. Once the algorithm was discovered, the cipher was immediately broken.

---

## 3. Confusion and Diffusion

Claude Shannon (1949) identified two fundamental properties that a secure cipher must have:

### 3.1 Confusion

**Confusion** makes the relationship between the key and the ciphertext as complex as possible. Each ciphertext bit should depend on several parts of the key.

- **Implementation:** Substitution operations (S-boxes in AES)
- **Goal:** An attacker who knows some ciphertext cannot deduce information about the key

### 3.2 Diffusion

**Diffusion** spreads the influence of each plaintext bit across many ciphertext bits. Changing one plaintext bit should change approximately half the ciphertext bits.

- **Implementation:** Permutation and mixing operations (ShiftRows, MixColumns in AES)
- **Goal:** Redundancy in the plaintext is dispersed across the ciphertext

> **Analogy:** Think of confusion as scrambling the recipe (making it hard to figure out what ingredients were used) and diffusion as thoroughly stirring the batter (spreading each ingredient throughout the mixture). A good cipher does both, repeatedly.

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

## 4. Feistel Networks

### 4.1 Structure

A **Feistel network** is a general method for building block ciphers. It splits the block into two halves and applies multiple rounds:

```
Input: (L_0, R_0)

Round i:
    L_i = R_{i-1}
    R_i = L_{i-1} XOR F(R_{i-1}, K_i)

Output: (L_n, R_n)
```

Where $F$ is a **round function** and $K_i$ is the **round key** (derived from the master key via a key schedule).

> **Analogy:** A Feistel network is like a series of mixers on an assembly line. Each mixer takes the right half, transforms it with a secret ingredient (round key), and mixes the result into the left half. Then the halves swap for the next mixer. After enough rounds, the original data is thoroughly scrambled.

### 4.2 Key Property: Invertibility

The Feistel structure is **always invertible**, regardless of the round function $F$. To decrypt, simply reverse the round order:

$$R_{i-1} = L_i$$
$$L_{i-1} = R_i \oplus F(L_i, K_i)$$

This means $F$ does **not** need to be invertible — it can be any function, even a hash function. This is a powerful design advantage.

### 4.3 Implementation

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

## 5. DES: The Data Encryption Standard

### 5.1 Overview

DES was published as a federal standard in 1977 (designed by IBM with NSA modifications).

| Parameter | Value |
|-----------|-------|
| Block size | 64 bits |
| Key size | 56 bits (64 bits with parity) |
| Rounds | 16 |
| Structure | Feistel network |

### 5.2 DES Round Function

Each DES round applies:
1. **Expansion (E):** Expand 32-bit half to 48 bits
2. **Key mixing:** XOR with 48-bit round key
3. **S-box substitution:** Eight 6-to-4-bit S-boxes (non-linear, provides confusion)
4. **Permutation (P):** Fixed bit permutation (provides diffusion)

### 5.3 Why DES Fell

- **56-bit key:** In 1998, the EFF's "Deep Crack" machine brute-forced a DES key in 56 hours for $250,000. Today, it can be done in hours on commodity hardware.
- **3DES:** Triple DES ($E_{K_1}(D_{K_2}(E_{K_1}(m)))$) extended the effective key length to 112 bits but is 3x slower.
- DES was officially retired by NIST in 2005, replaced by AES.

### 5.4 The NSA Controversy

The NSA modified IBM's original S-box designs. This sparked conspiracy theories, but it was later discovered that the NSA had actually **strengthened** the S-boxes against differential cryptanalysis — a technique that was publicly unknown until 1990 but that the NSA had already discovered internally.

---

## 6. AES: The Advanced Encryption Standard

### 6.1 History

In 1997, NIST initiated a public competition to replace DES. After evaluating 15 candidates over 3 years, **Rijndael** (by Joan Daemen and Vincent Rijmen) was selected in 2001.

| Parameter | AES-128 | AES-192 | AES-256 |
|-----------|---------|---------|---------|
| Block size | 128 bits | 128 bits | 128 bits |
| Key size | 128 bits | 192 bits | 256 bits |
| Rounds | 10 | 12 | 14 |

### 6.2 AES is NOT a Feistel Network

Unlike DES, AES uses a **substitution-permutation network (SPN)**. Every bit of the block is transformed in every round (in Feistel, only half the block is modified per round). This means AES achieves full diffusion in fewer rounds.

### 6.3 The State Matrix

AES operates on a $4 \times 4$ matrix of bytes (128 bits = 16 bytes), called the **state**:

$$\begin{bmatrix} s_{0,0} & s_{0,1} & s_{0,2} & s_{0,3} \\ s_{1,0} & s_{1,1} & s_{1,2} & s_{1,3} \\ s_{2,0} & s_{2,1} & s_{2,2} & s_{2,3} \\ s_{3,0} & s_{3,1} & s_{3,2} & s_{3,3} \end{bmatrix}$$

Bytes are filled **column-major**: the first 4 bytes of plaintext fill the first column.

> **Analogy:** AES is like a series of industrial mixing machines — each round stirs the data more thoroughly. SubBytes changes the color of each ingredient, ShiftRows rearranges them on the conveyor belt, MixColumns blends ingredients within each column, and AddRoundKey seasons the mixture with the secret key.

---

## 7. AES Internals Step by Step

Each AES round applies four operations in order. The last round omits MixColumns.

### 7.1 SubBytes (Substitution)

Each byte is replaced using a fixed **S-box** — a $16 \times 16$ lookup table.

The S-box is constructed mathematically:
1. Compute the multiplicative inverse of the byte in $GF(2^8)$ (the finite field with 256 elements)
2. Apply an affine transformation over $GF(2)$

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

**Why SubBytes provides confusion:** The S-box has no fixed points ($S(x) \neq x$ for all $x$), no opposite fixed points ($S(x) \neq \overline{x}$), and high non-linearity. It provides the non-linear element that makes AES resistant to linear and differential cryptanalysis.

### 7.2 ShiftRows (Permutation)

Each row of the state matrix is cyclically shifted left by its row index:

- Row 0: no shift
- Row 1: shift left by 1
- Row 2: shift left by 2
- Row 3: shift left by 3

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

### 7.3 MixColumns (Diffusion)

Each column is treated as a polynomial over $GF(2^8)$ and multiplied by a fixed polynomial:

$$c(x) = 3x^3 + x^2 + x + 2$$

In matrix form:

$$\begin{bmatrix} 2 & 3 & 1 & 1 \\ 1 & 2 & 3 & 1 \\ 1 & 1 & 2 & 3 \\ 3 & 1 & 1 & 2 \end{bmatrix} \begin{bmatrix} s_{0,j} \\ s_{1,j} \\ s_{2,j} \\ s_{3,j} \end{bmatrix}$$

All arithmetic is in $GF(2^8)$ with the irreducible polynomial $x^8 + x^4 + x^3 + x + 1$.

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

**Why MixColumns provides diffusion:** After MixColumns, each byte in a column depends on all four bytes of that column. Combined with ShiftRows (which moves bytes between columns), after just two rounds every output byte depends on every input byte.

### 7.4 AddRoundKey

XOR the state with the round key (derived from the master key via the key schedule):

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

### 7.5 Key Schedule

The AES key schedule expands the 128-bit master key into 11 round keys (one initial + 10 rounds):

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

### 7.6 Putting It All Together

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

## 8. Security Analysis

### 8.1 AES Security Status

As of 2026, **no practical attack on full AES** has been found.

| Attack | Best Result | Practical? |
|--------|-------------|------------|
| Brute force on AES-128 | $2^{128}$ operations | No ($10^{24}$ years on fastest supercomputer) |
| Biclique attack | $2^{126.1}$ for AES-128 | No (negligible improvement) |
| Related-key attacks | Theoretical on AES-256 | No (requires related keys, unrealistic model) |
| Side-channel (timing) | Key recovery from cache timing | Yes, but **implementation** attack, not algorithm |

### 8.2 Common Attack Types

**Differential Cryptanalysis** — Analyze how input differences propagate through the cipher. AES S-boxes are specifically designed to resist this.

**Linear Cryptanalysis** — Approximate the cipher with linear equations. The non-linearity of AES S-boxes provides high resistance.

**Side-Channel Attacks** — Exploit implementation details (timing, power consumption, electromagnetic emissions) rather than the algorithm itself. Countermeasures include constant-time implementations and masking.

### 8.3 Key Size Recommendations

| Use Case | Minimum Key Size | Recommended |
|----------|-----------------|-------------|
| Short-term data (< 5 years) | AES-128 | AES-128 |
| Long-term data (> 10 years) | AES-128 | AES-256 |
| Quantum-resistant | AES-256 | AES-256 (Grover halves effective key size) |

---

## 9. Exercises

### Exercise 1: Frequency Analysis (Basic)

Encrypt the message "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" with a Caesar cipher (key=7) and a Vigenere cipher (key="CRYPTO"). Count the letter frequencies in each ciphertext. Which cipher better hides the plaintext frequency distribution?

### Exercise 2: Feistel Cipher (Intermediate)

1. Implement a 4-round Feistel cipher with a 64-bit block and 128-bit key.
2. Encrypt a plaintext block, then decrypt it. Verify you recover the original plaintext.
3. What happens if you use only 1 round? Can you recover any information about the plaintext from the ciphertext?

### Exercise 3: AES Step Trace (Intermediate)

Using the AES implementation from Section 7.6, print the state matrix after each operation (SubBytes, ShiftRows, MixColumns, AddRoundKey) for the first two rounds. Compare your output with the NIST FIPS 197 Appendix B test vectors.

### Exercise 4: Avalanche Effect (Challenging)

1. Encrypt two 16-byte plaintexts that differ by exactly one bit using AES.
2. Count how many bits differ in the ciphertext after round 1, round 2, and round 10.
3. Plot the number of differing bits vs. round number. How many rounds are needed to achieve near-50% diffusion?

### Exercise 5: S-Box Analysis (Challenging)

1. Verify that the AES S-box has no fixed points: $S(x) \neq x$ for all $x$.
2. Verify that it has no opposite fixed points: $S(x) \neq \overline{x}$ for all $x$.
3. Compute the **non-linearity** of the S-box: for each linear approximation $a \cdot x \oplus b \cdot S(x) = 0$, count how many of the 256 inputs satisfy it. The maximum deviation from 128 gives the linearity measure.

---

**Previous**: [Number Theory Foundations](./01_Number_Theory_Foundations.md) | **Next**: [Block Cipher Modes](./03_Block_Cipher_Modes.md)

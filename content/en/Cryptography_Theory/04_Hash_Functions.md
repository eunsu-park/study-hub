# Lesson 4: Hash Functions

**Previous**: [Block Cipher Modes](./03_Block_Cipher_Modes.md) | **Next**: [RSA Cryptosystem](./05_RSA_Cryptosystem.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the three security properties of cryptographic hash functions (preimage, second preimage, collision resistance)
2. Explain the Merkle-Damgard construction and trace data through SHA-256's compression function
3. Derive the birthday bound and estimate collision probabilities for a given hash output size
4. Describe the length extension attack and explain why SHA-3's sponge construction is immune to it
5. Implement HMAC from a hash function and explain why naive key-prefixed hashing is insecure
6. Compare password hashing algorithms (bcrypt, scrypt, Argon2) and explain memory-hardness

---

> **Analogy:** A hash function is a fingerprint machine — given a person, it produces a unique fingerprint, but you cannot reconstruct the person from the fingerprint. Just as no two people should share the same fingerprint, no two messages should produce the same hash. And just as it is easy to scan a finger but impossible to build a person from a fingerprint, hashing is fast but inverting is infeasible.

Hash functions appear everywhere in cryptography: digital signatures hash the message before signing (Lesson 7), password storage relies on slow hash functions, blockchain integrity depends on hash chains, and TLS uses HMAC for message authentication. Understanding hash functions deeply is prerequisite knowledge for everything that follows.

## Table of Contents

1. [What Is a Cryptographic Hash Function?](#1-what-is-a-cryptographic-hash-function)
2. [Security Properties](#2-security-properties)
3. [The Birthday Attack](#3-the-birthday-attack)
4. [Merkle-Damgard Construction](#4-merkle-damgard-construction)
5. [SHA-256 Internals](#5-sha-256-internals)
6. [Length Extension Attack](#6-length-extension-attack)
7. [SHA-3 and the Sponge Construction](#7-sha-3-and-the-sponge-construction)
8. [HMAC: Keyed Hashing](#8-hmac-keyed-hashing)
9. [Password Hashing](#9-password-hashing)
10. [Exercises](#10-exercises)

---

## 1. What Is a Cryptographic Hash Function?

A **cryptographic hash function** $H$ maps an arbitrary-length input to a fixed-length output:

$$H: \{0, 1\}^* \rightarrow \{0, 1\}^n$$

For SHA-256, $n = 256$ (32 bytes). The output is called the **hash**, **digest**, or **fingerprint**.

### 1.1 Basic Properties

| Property | Description |
|----------|-------------|
| **Deterministic** | Same input always produces the same output |
| **Fast** | Computing $H(m)$ is efficient |
| **Fixed output** | Output is always $n$ bits regardless of input size |
| **Avalanche effect** | A 1-bit change in input flips ~50% of output bits |

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

## 2. Security Properties

### 2.1 Preimage Resistance (One-Way)

Given a hash value $h$, it is computationally infeasible to find any message $m$ such that $H(m) = h$.

$$\text{Given } h, \text{ find } m \text{ such that } H(m) = h$$

**Expected effort:** $O(2^n)$ for an $n$-bit hash.

### 2.2 Second Preimage Resistance

Given a message $m_1$, it is computationally infeasible to find a different message $m_2 \neq m_1$ such that $H(m_1) = H(m_2)$.

$$\text{Given } m_1, \text{ find } m_2 \neq m_1 \text{ such that } H(m_1) = H(m_2)$$

**Expected effort:** $O(2^n)$ for an $n$-bit hash.

### 2.3 Collision Resistance

It is computationally infeasible to find **any** two distinct messages $m_1 \neq m_2$ such that $H(m_1) = H(m_2)$.

$$\text{Find any } m_1 \neq m_2 \text{ such that } H(m_1) = H(m_2)$$

**Expected effort:** $O(2^{n/2})$ — much less than preimage resistance! This is the **birthday bound** (Section 3).

### 2.4 Relationship Between Properties

```
Collision Resistance → Second Preimage Resistance → (weak) Preimage Resistance

If you can find collisions easily, you can find second preimages easily.
But second preimage resistance does NOT imply collision resistance.
```

| Hash | Output bits | Preimage | Second Preimage | Collision |
|------|------------|----------|-----------------|-----------|
| MD5 | 128 | $2^{128}$ (safe) | $2^{128}$ (safe) | **BROKEN** ($2^{18}$ — seconds) |
| SHA-1 | 160 | $2^{160}$ (safe) | $2^{160}$ (safe) | **BROKEN** ($2^{63}$) |
| SHA-256 | 256 | $2^{256}$ | $2^{256}$ | $2^{128}$ (safe) |
| SHA-3-256 | 256 | $2^{256}$ | $2^{256}$ | $2^{128}$ (safe) |

---

## 3. The Birthday Attack

### 3.1 The Birthday Paradox

In a room of 23 people, there is a >50% chance that two share a birthday. This is surprising because $23 \ll 365$.

The general formula: with $k$ random samples from $N$ values, the probability of a collision is approximately:

$$P(\text{collision}) \approx 1 - e^{-k^2/(2N)}$$

Setting $P = 0.5$ gives $k \approx \sqrt{N} \cdot 1.177$.

### 3.2 Implications for Hash Functions

For an $n$-bit hash function with $N = 2^n$ possible outputs, we expect a collision after approximately $2^{n/2}$ hashes.

| Hash size | Collision at | Security level |
|-----------|-------------|----------------|
| 128 bits | $2^{64}$ | Insecure (feasible) |
| 160 bits | $2^{80}$ | Marginal |
| 256 bits | $2^{128}$ | Secure |
| 512 bits | $2^{256}$ | Very secure |

This is why MD5 (128-bit) and SHA-1 (160-bit) are considered broken for collision resistance.

### 3.3 Birthday Attack Simulation

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

## 4. Merkle-Damgard Construction

### 4.1 How It Works

Most widely-used hash functions (MD5, SHA-1, SHA-256) follow the **Merkle-Damgard construction**:

1. Pad and split the message into fixed-size blocks $M_1, M_2, \ldots, M_L$
2. Start with a fixed **initialization vector** (IV)
3. Process each block through a **compression function** $f$:

$$H_0 = IV$$
$$H_i = f(H_{i-1}, M_i) \quad \text{for } i = 1, \ldots, L$$
$$H(M) = H_L$$

```
M_1        M_2        M_3        M_L
 ↓          ↓          ↓          ↓
IV → [f] → h_1 → [f] → h_2 → [f] → ... → [f] → H(M)
```

### 4.2 Why It Works

**Theorem (Merkle-Damgard):** If the compression function $f$ is collision-resistant, then the hash function $H$ built from it is also collision-resistant.

This means we only need to design a secure **compression function** for a fixed input size, and the Merkle-Damgard construction extends it to arbitrary-length inputs.

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

## 5. SHA-256 Internals

### 5.1 Overview

SHA-256 processes 512-bit (64-byte) message blocks and maintains a 256-bit state (eight 32-bit words).

| Parameter | Value |
|-----------|-------|
| Block size | 512 bits (64 bytes) |
| Output size | 256 bits (32 bytes) |
| Word size | 32 bits |
| Rounds | 64 |
| Operations | AND, OR, XOR, NOT, addition mod $2^{32}$, rotate |

### 5.2 Message Padding

1. Append a `1` bit
2. Append `0` bits until length $\equiv 448 \pmod{512}$
3. Append the original message length as a 64-bit big-endian integer

### 5.3 Message Schedule

Expand the 16-word (512-bit) block into 64 words:

$$W_t = \begin{cases} M_t & \text{for } 0 \leq t \leq 15 \\ \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16} & \text{for } 16 \leq t \leq 63 \end{cases}$$

Where:
$$\sigma_0(x) = \text{ROTR}^7(x) \oplus \text{ROTR}^{18}(x) \oplus \text{SHR}^3(x)$$
$$\sigma_1(x) = \text{ROTR}^{17}(x) \oplus \text{ROTR}^{19}(x) \oplus \text{SHR}^{10}(x)$$

### 5.4 Compression Function

Each round updates eight working variables $a, b, c, d, e, f, g, h$ using:

$$T_1 = h + \Sigma_1(e) + Ch(e, f, g) + K_t + W_t$$
$$T_2 = \Sigma_0(a) + Maj(a, b, c)$$
$$h = g, \quad g = f, \quad f = e, \quad e = d + T_1$$
$$d = c, \quad c = b, \quad b = a, \quad a = T_1 + T_2$$

Where:
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

### 5.5 Why These Constants?

The initial hash values are the fractional parts of $\sqrt{2}, \sqrt{3}, \sqrt{5}, \sqrt{7}, \sqrt{11}, \sqrt{13}, \sqrt{17}, \sqrt{19}$.

The round constants $K_t$ are the fractional parts of $\sqrt[3]{p}$ for the first 64 primes $p$.

These are called **nothing-up-my-sleeve numbers** — derived from well-known mathematical constants, proving there are no hidden backdoors in the constants.

---

## 6. Length Extension Attack

### 6.1 The Vulnerability

For Merkle-Damgard hashes, knowing $H(m)$ and $|m|$ (but not $m$) allows computing:

$$H(m \| \text{padding} \| m')$$

for any suffix $m'$, **without knowing $m$**.

### 6.2 How It Works

Since $H(m) = h_L$ is the final state of the compression function, an attacker can:
1. Use $H(m)$ as the starting state
2. Continue processing additional blocks $m'$
3. The result is a valid hash of $m \| \text{padding} \| m'$

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

### 6.3 Affected Hash Functions

| Hash | Vulnerable to Length Extension? |
|------|---------------------------------|
| MD5 | Yes |
| SHA-1 | Yes |
| SHA-256, SHA-512 | Yes |
| SHA-3 (Keccak) | **No** (sponge construction) |
| BLAKE2, BLAKE3 | **No** (different finalization) |

---

## 7. SHA-3 and the Sponge Construction

### 7.1 Motivation

After SHA-1 was theoretically broken (2005), NIST launched a competition for SHA-3. **Keccak** won in 2012, introducing a fundamentally different design: the **sponge construction**.

### 7.2 Sponge Construction

A sponge operates on a state of $b = r + c$ bits, where:
- $r$ = **rate** (bits absorbed/squeezed per step)
- $c$ = **capacity** (bits that provide security; never directly exposed)

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

### 7.3 Keccak Specifics

Keccak uses a $5 \times 5 \times 64 = 1600$-bit state and 24 rounds of the permutation $f = \iota \circ \chi \circ \pi \circ \rho \circ \theta$:

| Step | Operation | Purpose |
|------|-----------|---------|
| $\theta$ | Column parity mixing | Diffusion |
| $\rho$ | Bit rotation (different per lane) | Diffusion |
| $\pi$ | Lane permutation | Diffusion |
| $\chi$ | Non-linear mapping ($a \oplus (\neg b \wedge c)$) | Confusion |
| $\iota$ | XOR with round constant | Break symmetry |

---

## 8. HMAC: Keyed Hashing

### 8.1 Why Not H(key || message)?

As shown in Section 6, $H(\text{key} \| \text{message})$ is vulnerable to length extension attacks for Merkle-Damgard hashes.

### 8.2 HMAC Construction

**HMAC** (Hash-based Message Authentication Code, RFC 2104):

$$\text{HMAC}(K, m) = H\bigl((K' \oplus \text{opad}) \| H((K' \oplus \text{ipad}) \| m)\bigr)$$

Where:
- $K'$ = key padded to the hash block size (or $H(K)$ if $K$ is longer than block size)
- $\text{ipad}$ = `0x36` repeated
- $\text{opad}$ = `0x5C` repeated

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

### 8.3 HMAC Security

HMAC is secure even if the underlying hash function has some weaknesses:
- Resistant to length extension (by design)
- Security proof: HMAC is a PRF (pseudorandom function) if the compression function is a PRF
- Even MD5-HMAC is still considered secure (despite MD5 collision attacks)

---

## 9. Password Hashing

### 9.1 Why Not SHA-256 for Passwords?

SHA-256 is **too fast**. A modern GPU can compute billions of SHA-256 hashes per second, enabling rapid brute-force or dictionary attacks on password databases.

| Attack | SHA-256 rate | Time for 8-char password |
|--------|-------------|--------------------------|
| CPU (single) | ~20M/s | Hours |
| GPU (RTX 4090) | ~10B/s | Seconds |
| ASIC (Bitcoin miner) | ~100T/s | Microseconds |

### 9.2 Password Hashing Requirements

A good password hash function must be:
1. **Slow** — deliberate computational cost (tunable work factor)
2. **Memory-hard** — require significant memory (resists GPU/ASIC attacks)
3. **Salted** — unique random salt per password (prevents rainbow tables)

### 9.3 bcrypt

- **Based on:** Blowfish cipher (Provos & Mazieres, 1999)
- **Parameters:** Cost factor (exponential: cost=12 means $2^{12}$ rounds)
- **Memory:** ~4 KB (small, main strength is CPU cost)
- **Output format:** `$2b$12$salt22chars.hash31chars`

### 9.4 scrypt

- **Based on:** PBKDF2 + Sequential Memory-Hard Function (Percival, 2009)
- **Parameters:** N (CPU/memory cost), r (block size), p (parallelism)
- **Memory:** Configurable (default ~16 MB)
- **Advantage over bcrypt:** Memory-hardness deters GPU attacks

### 9.5 Argon2

**Winner of the Password Hashing Competition (2015).** Three variants:

| Variant | Optimized For | Use Case |
|---------|---------------|----------|
| Argon2d | GPU resistance | Cryptocurrency mining |
| Argon2i | Side-channel resistance | Password hashing on shared servers |
| Argon2id | Both (hybrid) | **Recommended default** |

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

### 9.6 Salt and Pepper

| Concept | Storage | Purpose |
|---------|---------|---------|
| **Salt** | Stored with hash | Prevents rainbow tables; unique per user |
| **Pepper** | Stored separately (env var, HSM) | Adds secret; protects against DB leaks |

---

## 10. Exercises

### Exercise 1: Hash Properties (Basic)

1. Compute SHA-256 of "Hello" and "Hello " (with trailing space). How many bits differ?
2. Find a message whose SHA-256 starts with "0000" (4 zero hex chars = 16 zero bits). How many attempts did it take?
3. Relate part (2) to Bitcoin mining difficulty.

### Exercise 2: Birthday Attack (Intermediate)

1. Implement a birthday attack on a truncated hash: use only the first 4 bytes (32 bits) of SHA-256.
2. Find a collision. How many hashes were needed?
3. Compare with the theoretical $\sqrt{2^{32}} \approx 65536$.

### Exercise 3: HMAC Verification (Intermediate)

1. Implement HMAC-SHA256 from scratch (using only `hashlib.sha256`).
2. Test it against `hmac.new(key, msg, hashlib.sha256)` for 10 different key/message pairs.
3. Demonstrate that H(key || msg) and HMAC(key, msg) produce different results for the same inputs.

### Exercise 4: Length Extension (Challenging)

1. Given `SHA-256("secret" || "data") = <hash>` and `len("secret") = 6`, compute `SHA-256("secret" || "data" || padding || "evil")` without knowing "secret".
2. Use the `hashpumpy` or `hlextend` library to verify your result.
3. Show that HMAC is immune to this attack.

### Exercise 5: Password Cracking (Challenging)

1. Hash 1000 common passwords with SHA-256 (no salt). Time how long it takes to find a match given a hash.
2. Repeat with PBKDF2 (100,000 iterations). Compare the time.
3. Add random 16-byte salts. Explain why a precomputed rainbow table is now useless.
4. Calculate: for Argon2id with 64MB memory, how much total RAM would an attacker need to try $10^6$ passwords in parallel?

---

**Previous**: [Block Cipher Modes](./03_Block_Cipher_Modes.md) | **Next**: [RSA Cryptosystem](./05_RSA_Cryptosystem.md)

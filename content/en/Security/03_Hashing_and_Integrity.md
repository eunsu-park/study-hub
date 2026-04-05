# Hashing and Data Integrity

**Previous**: [02. Cryptography Basics](./02_Cryptography_Basics.md) | **Next**: [04. TLS/SSL and Public Key Infrastructure](./04_TLS_and_PKI.md)

---

Hash functions are the workhorses of modern security. They appear everywhere: password storage, digital signatures, message authentication, blockchain, software distribution, and data deduplication. This lesson covers hash functions in depth, from the mathematical properties that make them useful to practical implementations of password hashing, HMACs, and Merkle trees.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the properties of cryptographic hash functions
2. Implement hashing with SHA-256, SHA-3, BLAKE2, and BLAKE3
3. Store passwords securely using bcrypt, scrypt, and Argon2
4. Construct and verify HMACs for message authentication
5. Build and understand Merkle trees
6. Implement content-addressable storage
7. Recognize and prevent timing attacks using constant-time comparison

---

## Table of Contents

1. [What Is a Hash Function?](#1-what-is-a-hash-function)
2. [Cryptographic Hash Properties](#2-cryptographic-hash-properties)
3. [Hash Function Survey](#3-hash-function-survey)
4. [Python Hashing with hashlib](#4-python-hashing-with-hashlib)
5. [Password Hashing](#5-password-hashing)
6. [HMAC: Message Authentication](#6-hmac-message-authentication)
7. [Merkle Trees](#7-merkle-trees)
8. [Content-Addressable Storage](#8-content-addressable-storage)
9. [Timing Attacks and Constant-Time Comparison](#9-timing-attacks-and-constant-time-comparison)
10. [Hash Function Attacks and Mitigations](#10-hash-function-attacks-and-mitigations)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. What Is a Hash Function?

A hash function takes an arbitrary-length input and produces a fixed-length output (the "hash" or "digest"). A **cryptographic** hash function has additional security properties that make it suitable for security applications.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Hash Function                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Input (any size)           Hash Function         Output (fixed)    │
│   ──────────────            ──────────────         ──────────────    │
│                                                                      │
│   "Hello"           ──────▶  SHA-256  ──────▶  2cf24dba5fb0...      │
│   (5 bytes)                                    (32 bytes / 256 bits) │
│                                                                      │
│   "Hello World"     ──────▶  SHA-256  ──────▶  a591a6d40bf4...      │
│   (11 bytes)                                   (32 bytes / 256 bits) │
│                                                                      │
│   War and Peace     ──────▶  SHA-256  ──────▶  b28f8b893c45...      │
│   (~3.2 MB)                                    (32 bytes / 256 bits) │
│                                                                      │
│   Key insight: Regardless of input size, the output is ALWAYS       │
│   the same fixed size.                                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 Non-Cryptographic vs Cryptographic Hashes

```
┌─────────────────────────────────────────────────────────────────────┐
│        Non-Cryptographic vs Cryptographic Hash Functions             │
├──────────────────┬──────────────────────┬───────────────────────────┤
│                  │ Non-Cryptographic    │ Cryptographic              │
├──────────────────┼──────────────────────┼───────────────────────────┤
│ Purpose          │ Hash tables, checksums│ Security applications     │
│ Speed            │ Extremely fast       │ Intentionally slower      │
│ Collision resist.│ Weak / none          │ Strong (required)         │
│ Pre-image resist.│ Not guaranteed       │ Strong (required)         │
│ Examples         │ CRC32, MurmurHash,  │ SHA-256, SHA-3, BLAKE2,  │
│                  │ xxHash, FNV          │ BLAKE3                    │
│ Use cases        │ Hash maps, checksums,│ Passwords, signatures,   │
│                  │ deduplication        │ certificates, HMAC       │
├──────────────────┴──────────────────────┴───────────────────────────┤
│ RULE: NEVER use a non-cryptographic hash for security purposes.    │
│ NEVER use a cryptographic hash where speed matters and security    │
│ does not (e.g., hash tables).                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cryptographic Hash Properties

A secure cryptographic hash function must satisfy three properties:

```
┌──────────────────────────────────────────────────────────────────────┐
│             Three Properties of Cryptographic Hashes                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PRE-IMAGE RESISTANCE (one-way)                                  │
│     Given hash h, it is infeasible to find any m such that          │
│     H(m) = h.                                                       │
│                                                                      │
│     Hash ──╳──▶ Original input                                     │
│     (You cannot reverse a hash to get the input)                    │
│                                                                      │
│  2. SECOND PRE-IMAGE RESISTANCE (weak collision resistance)         │
│     Given m₁, it is infeasible to find m₂ ≠ m₁ such that          │
│     H(m₁) = H(m₂).                                                 │
│                                                                      │
│     "Hello" → abc123    Cannot find another input → abc123          │
│     (Cannot find a second input with the same hash)                 │
│                                                                      │
│  3. COLLISION RESISTANCE (strong collision resistance)               │
│     It is infeasible to find ANY two distinct messages m₁ ≠ m₂     │
│     such that H(m₁) = H(m₂).                                       │
│                                                                      │
│     Cannot find ANY pair of inputs with the same hash               │
│     (This is strictly stronger than second pre-image resistance)    │
│                                                                      │
│  Security levels (for an n-bit hash):                               │
│  • Pre-image:    O(2^n) work (birthday paradox does not apply)     │
│  • 2nd pre-image: O(2^n) work                                      │
│  • Collision:     O(2^(n/2)) work (birthday paradox)               │
│                                                                      │
│  This is why SHA-256 provides 128-bit collision resistance          │
│  (2^128 ≈ 3.4 × 10^38 operations).                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 The Avalanche Effect

A good hash function exhibits the **avalanche effect**: a tiny change in input produces a drastically different output.

```python
import hashlib

# Demonstrate the avalanche effect
msg1 = b"Hello, World!"
msg2 = b"Hello, World?"  # Changed '!' to '?'
msg3 = b"Hello, World! " # Added a space

hash1 = hashlib.sha256(msg1).hexdigest()
hash2 = hashlib.sha256(msg2).hexdigest()
hash3 = hashlib.sha256(msg3).hexdigest()

print("Avalanche Effect Demonstration (SHA-256):")
print(f"  '{msg1.decode()}' → {hash1}")
print(f"  '{msg2.decode()}' → {hash2}")
print(f"  '{msg3.decode()}'→ {hash3}")

# Count differing bits
def bit_difference(hex1: str, hex2: str) -> tuple:
    """Count the number of differing bits between two hex strings."""
    b1 = int(hex1, 16)
    b2 = int(hex2, 16)
    xor = b1 ^ b2
    diff_bits = bin(xor).count('1')
    total_bits = len(hex1) * 4
    return diff_bits, total_bits

diff, total = bit_difference(hash1, hash2)
print(f"\n  Bits changed (1 char diff): {diff}/{total} "
      f"({diff/total*100:.1f}%)")

diff2, _ = bit_difference(hash1, hash3)
print(f"  Bits changed (space added): {diff2}/{total} "
      f"({diff2/total*100:.1f}%)")

# Ideal: ~50% of bits differ for any change
print(f"  Expected for random: ~{total//2} bits ({50.0}%)")
```

---

## 3. Hash Function Survey

### 3.1 Hash Function Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                Hash Function Comparison Table                        │
├────────────┬────────┬───────────┬──────────┬────────────────────────┤
│ Algorithm  │ Output │ Speed     │ Status   │ Notes                  │
│            │ (bits) │ (GB/s)*   │          │                        │
├────────────┼────────┼───────────┼──────────┼────────────────────────┤
│ MD5        │ 128    │ ~5.0      │ BROKEN   │ Collisions found 2004  │
│ SHA-1      │ 160    │ ~3.0      │ BROKEN   │ SHAttered attack 2017  │
│ SHA-256    │ 256    │ ~1.5      │ SECURE   │ Most widely used       │
│ SHA-512    │ 512    │ ~2.0†     │ SECURE   │ Faster on 64-bit CPUs  │
│ SHA-3-256  │ 256    │ ~0.5      │ SECURE   │ Different construction  │
│ BLAKE2b    │ 256    │ ~3.5      │ SECURE   │ Faster than SHA-256    │
│ BLAKE2s    │ 256    │ ~2.0      │ SECURE   │ Optimized for 32-bit   │
│ BLAKE3     │ 256    │ ~10.0     │ SECURE   │ Parallelizable, newest │
├────────────┴────────┴───────────┴──────────┴────────────────────────┤
│ * Approximate throughput on modern x86-64 with hardware support     │
│ † SHA-512 is faster than SHA-256 on 64-bit processors              │
│                                                                      │
│ Recommendation: SHA-256 for interoperability, BLAKE2b/BLAKE3 for   │
│ performance-sensitive applications                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 SHA-2 Family

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SHA-2 Family                                  │
├──────────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ Variant      │ Output   │ Block    │ Word     │ Rounds              │
│              │ (bits)   │ (bits)   │ (bits)   │                     │
├──────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ SHA-224      │ 224      │ 512      │ 32       │ 64                  │
│ SHA-256      │ 256      │ 512      │ 32       │ 64                  │
│ SHA-384      │ 384      │ 1024     │ 64       │ 80                  │
│ SHA-512      │ 512      │ 1024     │ 64       │ 80                  │
│ SHA-512/256  │ 256      │ 1024     │ 64       │ 80                  │
├──────────────┴──────────┴──────────┴──────────┴─────────────────────┤
│ SHA-256 and SHA-512 are the most commonly used.                     │
│ SHA-512/256 gives SHA-256 output size with SHA-512's speed on       │
│ 64-bit processors and is not vulnerable to length-extension.        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 SHA-3 (Keccak)

SHA-3 uses the **sponge construction**, which is fundamentally different from SHA-2's Merkle-Damgard construction. This diversity is valuable: if SHA-2 is ever broken, SHA-3 will likely remain secure.

```
┌──────────────────────────────────────────────────────────────────────┐
│              SHA-3 Sponge Construction (Simplified)                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  State = [0...0]  (1600 bits for Keccak)                            │
│                                                                      │
│  ABSORBING PHASE:                                                    │
│  ┌───────┐   XOR    ┌───────┐   XOR    ┌───────┐                   │
│  │ msg₁  │──────▶ f ──────▶│ msg₂  │──────▶ f ──────▶ ...        │
│  └───────┘   State  └───────┘   State  └───────┘                   │
│                                                                      │
│  SQUEEZING PHASE:                                                    │
│  ... ──▶ f ──▶ [output₁] ──▶ f ──▶ [output₂] ──▶ ...             │
│                                                                      │
│  f = Keccak permutation (5 sub-rounds × 24 rounds)                 │
│                                                                      │
│  Key advantage: NOT vulnerable to length-extension attacks           │
│  (unlike SHA-256 / SHA-512)                                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 BLAKE2 and BLAKE3

```
┌─────────────────────────────────────────────────────────────────────┐
│                   BLAKE2 and BLAKE3                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BLAKE2 (RFC 7693):                                                 │
│  ├── BLAKE2b: Optimized for 64-bit platforms (up to 64 bytes)      │
│  ├── BLAKE2s: Optimized for 32-bit platforms (up to 32 bytes)      │
│  ├── Built-in keying (can act as a MAC without HMAC)               │
│  ├── Built-in personalization and salt support                      │
│  ├── Tree hashing mode for parallelism                             │
│  └── Used in: Argon2 password hash, WireGuard, libsodium           │
│                                                                      │
│  BLAKE3 (2020):                                                     │
│  ├── Based on BLAKE2 but redesigned for speed                      │
│  ├── Merkle tree structure (inherently parallelizable)             │
│  ├── Single algorithm for hash, MAC, KDF, XOF                     │
│  ├── 256-bit output (extendable)                                   │
│  ├── ~10x faster than SHA-256 on modern CPUs                       │
│  └── Used in: Bao (verified streaming), many new projects          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Python Hashing with hashlib

### 4.1 Basic Hashing

```python
import hashlib

message = b"The quick brown fox jumps over the lazy dog"

# SHA-256 (most common)
sha256 = hashlib.sha256(message).hexdigest()
print(f"SHA-256:   {sha256}")

# SHA-512
sha512 = hashlib.sha512(message).hexdigest()
print(f"SHA-512:   {sha512[:64]}...{sha512[-8:]}")

# SHA-3-256
sha3_256 = hashlib.sha3_256(message).hexdigest()
print(f"SHA-3-256: {sha3_256}")

# BLAKE2b (variable output size, up to 64 bytes)
blake2b = hashlib.blake2b(message, digest_size=32).hexdigest()
print(f"BLAKE2b:   {blake2b}")

# BLAKE2s (variable output size, up to 32 bytes)
blake2s = hashlib.blake2s(message, digest_size=32).hexdigest()
print(f"BLAKE2s:   {blake2s}")

# List all available algorithms
print(f"\nAvailable: {sorted(hashlib.algorithms_available)[:10]}...")
```

### 4.2 Incremental Hashing (Streaming)

For large files, you should hash data incrementally rather than loading everything into memory:

```python
import hashlib
from pathlib import Path

def hash_file(filepath: str, algorithm: str = "sha256",
              chunk_size: int = 8192) -> str:
    """Hash a file incrementally without loading it all into memory."""
    hasher = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

# Create a test file
test_file = "/tmp/test_hash_file.bin"
with open(test_file, 'wb') as f:
    for i in range(1000):
        f.write(f"Line {i}: some data for hashing\n".encode())

# Hash it
sha256_hash = hash_file(test_file, "sha256")
sha3_hash = hash_file(test_file, "sha3_256")
blake2_hash = hash_file(test_file, "blake2b")

print(f"File size: {Path(test_file).stat().st_size:,} bytes")
print(f"SHA-256:   {sha256_hash}")
print(f"SHA-3-256: {sha3_hash}")
print(f"BLAKE2b:   {blake2_hash}")

# Equivalent to reading all at once (but memory-efficient)
data = Path(test_file).read_bytes()
assert hashlib.sha256(data).hexdigest() == sha256_hash
print("\nIncremental hash matches full-read hash: OK")

# Clean up
Path(test_file).unlink()
```

### 4.3 BLAKE2 with Key (Keyed Hashing)

BLAKE2 has built-in support for keyed hashing, making it usable as a MAC without the HMAC construction:

```python
import hashlib
import os

# BLAKE2b with key (acts as a MAC)
key = os.urandom(32)  # 256-bit key
message = b"Authenticate this message"

# Keyed hash
mac = hashlib.blake2b(message, key=key, digest_size=32).hexdigest()
print(f"BLAKE2b keyed hash: {mac}")

# Verify
verification = hashlib.blake2b(message, key=key, digest_size=32).hexdigest()
print(f"Verification match: {mac == verification}")

# With personalization (domain separation)
mac1 = hashlib.blake2b(
    b"shared data",
    key=key,
    person=b"payment-v1",  # Max 16 bytes
    digest_size=32
).hexdigest()

mac2 = hashlib.blake2b(
    b"shared data",
    key=key,
    person=b"session-v1",  # Different domain
    digest_size=32
).hexdigest()

print(f"\nSame data, different personalization:")
print(f"  payment context: {mac1[:32]}...")
print(f"  session context: {mac2[:32]}...")
print(f"  Same? {mac1 == mac2}")  # False - different domains
```

### 4.4 BLAKE3

```python
# pip install blake3
import blake3

message = b"BLAKE3 is extremely fast"

# Basic hash
digest = blake3.blake3(message).hexdigest()
print(f"BLAKE3: {digest}")

# Keyed hash (for MAC)
key = b"0" * 32  # 32-byte key required
keyed = blake3.blake3(message, key=key).hexdigest()
print(f"BLAKE3 keyed: {keyed}")

# Key derivation
derived = blake3.blake3(
    b"input key material",
    derive_key_context="my-app 2026-01-15 encryption key"
).hexdigest()
print(f"BLAKE3 KDF: {derived}")

# Incremental hashing
hasher = blake3.blake3()
hasher.update(b"Hello, ")
hasher.update(b"World!")
print(f"BLAKE3 incremental: {hasher.hexdigest()}")

# Extendable output (XOF) - get any number of bytes
digest_64 = blake3.blake3(message).hexdigest(length=64)
print(f"BLAKE3 64-byte: {digest_64}")
```

---

## 5. Password Hashing

Password hashing is fundamentally different from general-purpose hashing. Passwords have low entropy (humans choose predictable passwords), so an attacker who obtains password hashes can try to brute-force them. Password hashing algorithms are designed to be **deliberately slow** to make brute-force attacks impractical.

```
┌──────────────────────────────────────────────────────────────────────┐
│        Why General-Purpose Hashes Are BAD for Passwords              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SHA-256 speed on modern GPU: ~10 billion hashes/second             │
│                                                                      │
│  Password "P@ssw0rd":                                                │
│    SHA-256 → d74ff0ee8da3b9806b18c877d...                           │
│    Time to brute-force 8-char password: seconds to minutes          │
│                                                                      │
│  bcrypt (cost=12):                                                  │
│    bcrypt → $2b$12$LJ3m4ys3Tdb2vNQhk9Oy...                        │
│    Time to brute-force: years to centuries                          │
│                                                                      │
│  KEY DIFFERENCES:                                                    │
│  • Password hashes include a random SALT (prevents rainbow tables)  │
│  • Password hashes have a tunable WORK FACTOR (adapts to hardware) │
│  • Password hashes may require large MEMORY (defeats GPU attacks)   │
│                                                                      │
│  NEVER hash passwords with SHA-256, MD5, or any general hash!      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.1 bcrypt

```python
# pip install bcrypt
import bcrypt
import time

password = b"correct-horse-battery-staple"

# Hash a password
# bcrypt automatically generates a random 16-byte salt
# cost=12 means 2^12 = 4096 iterations
start = time.time()
hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))
elapsed = time.time() - start

print(f"Password:  {password.decode()}")
print(f"Hash:      {hashed.decode()}")
print(f"Time:      {elapsed:.3f}s")
print()

# Anatomy of a bcrypt hash:
# $2b$12$LJ3m4ys3Tdb2vNQhk9OyAeKK3b3eAGQjT5xKp2JFe5cF5NY5U/a2e
# ├──┤├─┤├──────────────────────┤├──────────────────────────────────┤
# Algo Cost       Salt (22 chars)         Hash (31 chars)
#
# $2b = bcrypt version
# $12 = cost factor (2^12 iterations)

# Verify a password
start = time.time()
is_valid = bcrypt.checkpw(password, hashed)
elapsed = time.time() - start
print(f"Valid password: {is_valid} ({elapsed:.3f}s)")

# Wrong password
is_valid = bcrypt.checkpw(b"wrong-password", hashed)
print(f"Wrong password: {is_valid}")

# Each hash is unique even for the same password (different salt)
hash2 = bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))
print(f"\nHash 1: {hashed.decode()}")
print(f"Hash 2: {hash2.decode()}")
print(f"Same?   {hashed == hash2}")  # False (different salts)
# But both verify the same password
print(f"Both verify 'correct-horse-battery-staple': "
      f"{bcrypt.checkpw(password, hashed) and bcrypt.checkpw(password, hash2)}")
```

### 5.2 scrypt

scrypt is **memory-hard**: it requires a large amount of memory proportional to the cost parameter, making it resistant to GPU and ASIC attacks.

```python
import hashlib
import os
import time

password = b"my-secure-password"
salt = os.urandom(16)

# scrypt parameters:
# n = CPU/memory cost (must be power of 2). Higher = slower + more memory.
# r = block size (8 is standard)
# p = parallelism (1 is standard)
# Memory usage ≈ 128 * n * r bytes

start = time.time()
derived = hashlib.scrypt(
    password,
    salt=salt,
    n=2**14,    # 16384 iterations
    r=8,        # Block size
    p=1,        # Parallelism
    dklen=32    # Output 256 bits
)
elapsed = time.time() - start

print(f"scrypt derived key: {derived.hex()}")
print(f"Salt:              {salt.hex()}")
print(f"Time:              {elapsed:.3f}s")
print(f"Memory used:       ~{128 * 2**14 * 8 / 1024 / 1024:.0f} MB")

# Verify
derived2 = hashlib.scrypt(password, salt=salt, n=2**14, r=8, p=1, dklen=32)
print(f"Verification:      {derived == derived2}")
```

### 5.3 Argon2 (Recommended)

Argon2 won the 2015 Password Hashing Competition and is the currently recommended algorithm. It comes in three variants:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Argon2 Variants                                │
├──────────────┬──────────────────────────────────────────────────────┤
│ Variant      │ Description                                         │
├──────────────┼──────────────────────────────────────────────────────┤
│ Argon2d      │ Data-dependent memory access. Faster, more GPU-     │
│              │ resistant, but vulnerable to side-channel attacks.   │
│              │ Best for cryptocurrency mining, NOT for passwords.   │
├──────────────┼──────────────────────────────────────────────────────┤
│ Argon2i      │ Data-independent memory access. Resistant to side-  │
│              │ channel attacks. Better for password hashing in      │
│              │ shared environments.                                │
├──────────────┼──────────────────────────────────────────────────────┤
│ Argon2id     │ Hybrid: first pass is Argon2i, subsequent passes   │
│              │ are Argon2d. RECOMMENDED for password hashing.      │
│              │ Best of both worlds.                                │
└──────────────┴──────────────────────────────────────────────────────┘
```

```python
# pip install argon2-cffi
from argon2 import PasswordHasher, Type
import time

# Create a hasher with recommended parameters
# Adjust time_cost and memory_cost based on your server's capabilities
# Target: ~0.5-1.0 seconds per hash
ph = PasswordHasher(
    time_cost=3,          # Number of iterations
    memory_cost=65536,    # Memory in KB (64 MB)
    parallelism=4,        # Number of threads
    hash_len=32,          # Output hash length
    salt_len=16,          # Salt length
    type=Type.ID,         # Argon2id (recommended)
)

password = "correct-horse-battery-staple"

# Hash
start = time.time()
hashed = ph.hash(password)
elapsed = time.time() - start

print(f"Argon2id hash: {hashed}")
print(f"Time: {elapsed:.3f}s")
print()

# Anatomy of an Argon2 hash string:
# $argon2id$v=19$m=65536,t=3,p=4$c2FsdDEyMzQ1Njc4$aXaShZ7S2X3yBqBvP5WF4w
# ├───────┤├───┤├──────────────┤├──────────────────┤├──────────────────────────┤
#   Algo    Ver    Parameters         Salt (b64)           Hash (b64)

# Verify
try:
    is_valid = ph.verify(hashed, password)
    print(f"Valid password: {is_valid}")
except Exception as e:
    print(f"Verification failed: {e}")

# Wrong password
try:
    ph.verify(hashed, "wrong-password")
except Exception as e:
    print(f"Wrong password: {type(e).__name__}")

# Check if rehash is needed (parameters changed)
if ph.check_needs_rehash(hashed):
    print("Hash needs rehash with updated parameters")
else:
    print("Hash parameters are current")
```

### 5.4 Password Hashing Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│              Password Hashing Algorithm Comparison                    │
├─────────────┬─────────┬───────────────┬──────────┬──────────────────┤
│ Algorithm   │ Memory  │ GPU/ASIC      │ Side-Ch. │ Recommendation   │
│             │ Hard?   │ Resistant?    │ Safe?    │                  │
├─────────────┼─────────┼───────────────┼──────────┼──────────────────┤
│ bcrypt      │ No      │ Moderate      │ Yes      │ Good (mature)    │
│ scrypt      │ Yes     │ Good          │ Partial  │ Good             │
│ Argon2id    │ Yes     │ Best          │ Yes      │ Best (preferred) │
│ PBKDF2      │ No      │ Poor          │ Yes      │ Legacy only      │
├─────────────┴─────────┴───────────────┴──────────┴──────────────────┤
│                                                                      │
│ Recommended parameters (2025+, targeting ~0.5s on server):          │
│ • Argon2id: m=64MB, t=3, p=4                                       │
│ • bcrypt:   cost=12 to 14                                           │
│ • scrypt:   N=2^15, r=8, p=1                                       │
│ • PBKDF2:   600,000+ iterations with SHA-256                        │
│                                                                      │
│ OWASP recommends: Argon2id first, bcrypt second.                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.5 Complete Password Storage System

```python
from argon2 import PasswordHasher, Type, exceptions
from dataclasses import dataclass, field
from typing import Optional, Dict
import secrets
import time

@dataclass
class UserRecord:
    username: str
    password_hash: str
    created_at: float
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None

class PasswordStore:
    """Production-quality password storage using Argon2id."""

    MAX_ATTEMPTS = 5
    LOCKOUT_SECONDS = 300  # 5 minutes
    MIN_PASSWORD_LENGTH = 12

    def __init__(self):
        self.hasher = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16,
            type=Type.ID,
        )
        self.users: Dict[str, UserRecord] = {}

    def _validate_password_strength(self, password: str) -> list:
        """Check password against basic strength requirements."""
        issues = []
        if len(password) < self.MIN_PASSWORD_LENGTH:
            issues.append(f"Password must be at least {self.MIN_PASSWORD_LENGTH} chars")
        if password.lower() == password:
            issues.append("Password must contain uppercase letters")
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain digits")

        # Check against common passwords (abbreviated list)
        common = {"password", "123456", "qwerty", "admin", "letmein"}
        if password.lower() in common:
            issues.append("Password is too common")

        return issues

    def register(self, username: str, password: str) -> dict:
        """Register a new user."""
        if username in self.users:
            return {"success": False, "error": "Username already exists"}

        issues = self._validate_password_strength(password)
        if issues:
            return {"success": False, "error": "Weak password", "issues": issues}

        hashed = self.hasher.hash(password)
        self.users[username] = UserRecord(
            username=username,
            password_hash=hashed,
            created_at=time.time(),
        )
        return {"success": True, "message": f"User {username} registered"}

    def authenticate(self, username: str, password: str) -> dict:
        """Authenticate a user."""
        if username not in self.users:
            # Perform a dummy hash to prevent timing-based user enumeration
            self.hasher.hash("dummy-to-waste-time")
            return {"success": False, "error": "Invalid credentials"}

        user = self.users[username]

        # Check lockout
        if user.locked_until and time.time() < user.locked_until:
            remaining = int(user.locked_until - time.time())
            return {"success": False, "error": f"Account locked. Try in {remaining}s"}

        try:
            self.hasher.verify(user.password_hash, password)

            # Check if rehash needed (parameters upgraded)
            if self.hasher.check_needs_rehash(user.password_hash):
                user.password_hash = self.hasher.hash(password)

            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = time.time()
            return {"success": True, "message": f"Welcome, {username}!"}

        except exceptions.VerifyMismatchError:
            user.failed_attempts += 1
            if user.failed_attempts >= self.MAX_ATTEMPTS:
                user.locked_until = time.time() + self.LOCKOUT_SECONDS
            return {
                "success": False,
                "error": "Invalid credentials",
                "attempts_remaining": max(0, self.MAX_ATTEMPTS - user.failed_attempts)
            }

# Usage
store = PasswordStore()

# Register
print(store.register("alice", "short"))  # Fails - too short
print(store.register("alice", "MySecureP@ss123"))  # Succeeds
print(store.register("alice", "AnotherPassword1"))  # Fails - exists

# Authenticate
print(store.authenticate("alice", "MySecureP@ss123"))  # Success
print(store.authenticate("alice", "wrong-password"))    # Failure
print(store.authenticate("bob", "anything"))            # User not found
```

---

## 6. HMAC: Message Authentication

HMAC (Hash-based Message Authentication Code) provides both integrity and authenticity. It combines a secret key with a hash function to produce a tag that can only be created and verified by someone who knows the key.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HMAC Construction                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  HMAC(K, m) = H((K' ⊕ opad) || H((K' ⊕ ipad) || m))               │
│                                                                      │
│  Where:                                                              │
│  K  = secret key                                                    │
│  K' = key padded/hashed to block size                               │
│  opad = 0x5c repeated to block size                                 │
│  ipad = 0x36 repeated to block size                                 │
│  H  = hash function (SHA-256, etc.)                                 │
│  || = concatenation                                                 │
│  ⊕  = XOR                                                           │
│                                                                      │
│  Step by step:                                                      │
│  1. If key > block size: K' = H(K)                                  │
│  2. Else: K' = K padded with zeros                                  │
│  3. Inner hash: H((K' ⊕ ipad) || message)                          │
│  4. Outer hash: H((K' ⊕ opad) || inner_hash)                       │
│                                                                      │
│  Why not just H(key || message)?                                    │
│  → Vulnerable to length-extension attacks with Merkle-Damgard       │
│    hashes (SHA-256). HMAC's nested structure prevents this.         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.1 HMAC in Python

```python
import hmac
import hashlib
import os
import time

# Generate a secret key
key = os.urandom(32)  # 256-bit key

# Create HMAC
message = b"Transfer $1000 to account 12345"
mac = hmac.new(key, message, hashlib.sha256).hexdigest()
print(f"Message: {message.decode()}")
print(f"HMAC:    {mac}")

# Verify HMAC (constant-time comparison)
received_mac = mac
is_valid = hmac.compare_digest(
    mac,
    hmac.new(key, message, hashlib.sha256).hexdigest()
)
print(f"Valid:   {is_valid}")

# Tampered message
tampered = b"Transfer $9999 to account 12345"
is_valid = hmac.compare_digest(
    mac,
    hmac.new(key, tampered, hashlib.sha256).hexdigest()
)
print(f"Tampered valid: {is_valid}")  # False
```

### 6.2 Practical: API Request Signing

```python
import hmac
import hashlib
import json
import time
import os
from urllib.parse import urlencode

class APIRequestSigner:
    """
    Sign API requests using HMAC-SHA256.
    Similar to AWS Signature V4, Stripe webhook signatures, etc.
    """

    def __init__(self, api_key: str, api_secret: bytes):
        self.api_key = api_key
        self.api_secret = api_secret

    def sign_request(self, method: str, path: str,
                      body: dict = None, timestamp: float = None) -> dict:
        """Sign an API request and return headers."""
        timestamp = timestamp or time.time()
        ts_str = str(int(timestamp))

        # Build canonical request string
        body_str = json.dumps(body, sort_keys=True, separators=(',', ':')) if body else ""
        canonical = f"{method}\n{path}\n{ts_str}\n{body_str}"

        # Compute HMAC
        signature = hmac.new(
            self.api_secret,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": ts_str,
            "X-Signature": signature,
        }

    def verify_request(self, method: str, path: str,
                        headers: dict, body: dict = None,
                        max_age_seconds: int = 300) -> dict:
        """Verify a signed API request."""
        api_key = headers.get("X-API-Key", "")
        timestamp = headers.get("X-Timestamp", "")
        signature = headers.get("X-Signature", "")

        # Check API key
        if api_key != self.api_key:
            return {"valid": False, "error": "Invalid API key"}

        # Check timestamp (prevent replay attacks)
        try:
            ts = int(timestamp)
            age = abs(time.time() - ts)
            if age > max_age_seconds:
                return {"valid": False, "error": f"Request too old ({age:.0f}s)"}
        except ValueError:
            return {"valid": False, "error": "Invalid timestamp"}

        # Recompute and verify signature
        body_str = json.dumps(body, sort_keys=True, separators=(',', ':')) if body else ""
        canonical = f"{method}\n{path}\n{timestamp}\n{body_str}"

        expected = hmac.new(
            self.api_secret,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return {"valid": False, "error": "Invalid signature"}

        return {"valid": True, "api_key": api_key}

# Usage
api_key = "ak_live_abc123"
api_secret = os.urandom(32)

signer = APIRequestSigner(api_key, api_secret)

# Sign a request
headers = signer.sign_request(
    method="POST",
    path="/api/v1/transfers",
    body={"amount": 1000, "currency": "USD", "to": "acct_xyz"}
)
print("Signed Request Headers:")
for k, v in headers.items():
    print(f"  {k}: {v}")

# Verify the request
result = signer.verify_request(
    method="POST",
    path="/api/v1/transfers",
    headers=headers,
    body={"amount": 1000, "currency": "USD", "to": "acct_xyz"}
)
print(f"\nVerification: {result}")

# Tampered body
result = signer.verify_request(
    method="POST",
    path="/api/v1/transfers",
    headers=headers,
    body={"amount": 9999, "currency": "USD", "to": "acct_xyz"}
)
print(f"Tampered body: {result}")
```

### 6.3 Webhook Signature Verification

```python
import hmac
import hashlib

def verify_webhook_signature(
    payload: bytes,
    signature_header: str,
    secret: bytes,
    tolerance_seconds: int = 300,
) -> bool:
    """
    Verify a webhook signature (Stripe-style).
    Header format: t=<timestamp>,v1=<signature>
    """
    # Parse the header
    elements = {}
    for part in signature_header.split(","):
        key, _, value = part.partition("=")
        elements[key.strip()] = value.strip()

    timestamp = elements.get("t", "")
    received_sig = elements.get("v1", "")

    if not timestamp or not received_sig:
        return False

    # Check timestamp freshness
    import time
    try:
        ts = int(timestamp)
        if abs(time.time() - ts) > tolerance_seconds:
            return False
    except ValueError:
        return False

    # Compute expected signature
    signed_payload = f"{timestamp}.".encode() + payload
    expected_sig = hmac.new(
        secret, signed_payload, hashlib.sha256
    ).hexdigest()

    # Constant-time comparison
    return hmac.compare_digest(expected_sig, received_sig)

# Simulate webhook
import time
webhook_secret = b"whsec_test_secret_key_123"
payload = b'{"event":"payment.success","amount":5000}'
ts = str(int(time.time()))
sig = hmac.new(
    webhook_secret,
    f"{ts}.".encode() + payload,
    hashlib.sha256
).hexdigest()

header = f"t={ts},v1={sig}"

is_valid = verify_webhook_signature(payload, header, webhook_secret)
print(f"Webhook signature valid: {is_valid}")

# Tampered payload
is_valid = verify_webhook_signature(
    b'{"event":"payment.success","amount":50000}',  # Changed amount
    header,
    webhook_secret
)
print(f"Tampered webhook valid: {is_valid}")
```

---

## 7. Merkle Trees

A Merkle tree is a binary tree of hashes where each leaf node is a hash of a data block, and each internal node is a hash of its children. The root hash (Merkle root) summarizes all the data.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Merkle Tree                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        Root Hash                                     │
│                      H(H12 || H34)                                  │
│                       /        \                                     │
│                      /          \                                    │
│                   H12            H34                                 │
│                H(H1||H2)      H(H3||H4)                             │
│                /      \        /      \                              │
│              H1       H2     H3       H4                            │
│            H(D1)    H(D2)  H(D3)   H(D4)                           │
│              |        |      |        |                              │
│            Data1   Data2   Data3   Data4                            │
│                                                                      │
│  Properties:                                                        │
│  • Root hash changes if ANY data block changes                      │
│  • Can verify a single block with O(log n) hashes (Merkle proof)   │
│  • Used in: Git, Bitcoin, Certificate Transparency, IPFS            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.1 Merkle Tree Implementation

```python
import hashlib
from typing import List, Optional
from dataclasses import dataclass

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

@dataclass
class MerkleNode:
    hash: bytes
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None

class MerkleTree:
    """A complete Merkle tree implementation with proof generation/verification."""

    def __init__(self, data_blocks: List[bytes]):
        if not data_blocks:
            raise ValueError("Cannot create Merkle tree from empty data")

        self.leaves = [MerkleNode(hash=sha256(block)) for block in data_blocks]
        self.root = self._build_tree(self.leaves)

    def _build_tree(self, nodes: List[MerkleNode]) -> MerkleNode:
        """Build the tree bottom-up."""
        if len(nodes) == 1:
            return nodes[0]

        # If odd number of nodes, duplicate the last one
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])

        parent_nodes = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i].hash + nodes[i + 1].hash
            parent = MerkleNode(
                hash=sha256(combined),
                left=nodes[i],
                right=nodes[i + 1],
            )
            parent_nodes.append(parent)

        return self._build_tree(parent_nodes)

    @property
    def root_hash(self) -> str:
        return self.root.hash.hex()

    def get_proof(self, index: int) -> List[tuple]:
        """
        Generate a Merkle proof for the leaf at the given index.
        Returns list of (hash, position) tuples where position is 'left' or 'right'.
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Index {index} out of range")

        proof = []
        nodes = self.leaves[:]

        # If odd, duplicate last
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])

        current_index = index

        while len(nodes) > 1:
            next_level = []

            for i in range(0, len(nodes), 2):
                if i == current_index or i + 1 == current_index:
                    # This is the pair containing our node
                    if current_index % 2 == 0:
                        # Our node is on the left, sibling is on the right
                        proof.append((nodes[i + 1].hash, "right"))
                    else:
                        # Our node is on the right, sibling is on the left
                        proof.append((nodes[i].hash, "left"))

                combined = nodes[i].hash + nodes[i + 1].hash
                parent = MerkleNode(hash=sha256(combined))
                next_level.append(parent)

            current_index = current_index // 2
            nodes = next_level

            if len(nodes) > 1 and len(nodes) % 2 == 1:
                nodes.append(nodes[-1])

        return proof

    @staticmethod
    def verify_proof(data: bytes, proof: List[tuple], root_hash: str) -> bool:
        """Verify a Merkle proof."""
        current_hash = sha256(data)

        for sibling_hash, position in proof:
            if position == "left":
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash
            current_hash = sha256(combined)

        return current_hash.hex() == root_hash

# Build a Merkle tree
data_blocks = [
    b"Transaction: Alice -> Bob $100",
    b"Transaction: Bob -> Charlie $50",
    b"Transaction: Charlie -> Dave $25",
    b"Transaction: Dave -> Alice $75",
]

tree = MerkleTree(data_blocks)
print(f"Merkle root: {tree.root_hash}")
print(f"Leaves: {len(tree.leaves)}")

# Generate proof for transaction at index 1
proof = tree.get_proof(1)
print(f"\nProof for block 1 ({len(proof)} nodes):")
for h, pos in proof:
    print(f"  {pos}: {h.hex()[:32]}...")

# Verify the proof
is_valid = MerkleTree.verify_proof(
    data_blocks[1], proof, tree.root_hash
)
print(f"\nProof valid: {is_valid}")

# Tampered data fails
is_valid = MerkleTree.verify_proof(
    b"Transaction: Bob -> Charlie $5000",  # Tampered!
    proof, tree.root_hash
)
print(f"Tampered proof valid: {is_valid}")

# Show efficiency: verifying 1 block out of 1M requires only ~20 hashes
import math
n_blocks = 1_000_000
proof_size = math.ceil(math.log2(n_blocks))
print(f"\nFor {n_blocks:,} blocks:")
print(f"  Proof size: {proof_size} hashes ({proof_size * 32} bytes)")
print(f"  vs checking all: {n_blocks * 32:,} bytes")
print(f"  Efficiency: {n_blocks * 32 / (proof_size * 32):,.0f}x smaller")
```

---

## 8. Content-Addressable Storage

Content-addressable storage (CAS) uses the hash of data as its address/key. This is the foundation of Git, IPFS, Docker layers, and many deduplication systems.

```
┌──────────────────────────────────────────────────────────────────────┐
│              Content-Addressable Storage (CAS)                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Traditional Storage:                                                │
│    filename → data                                                  │
│    /docs/report.pdf → [file contents]                               │
│                                                                      │
│  Content-Addressable Storage:                                        │
│    hash(data) → data                                                │
│    sha256:a3f2b8... → [file contents]                               │
│                                                                      │
│  Benefits:                                                          │
│  • Automatic deduplication (same content = same hash = stored once) │
│  • Built-in integrity verification (hash IS the address)            │
│  • Immutable by design (changing content changes the address)       │
│  • Cache-friendly (content never changes for a given address)       │
│                                                                      │
│  Used in:                                                            │
│  • Git (blob objects addressed by SHA-1/SHA-256)                    │
│  • Docker (image layers addressed by SHA-256)                       │
│  • IPFS (blocks addressed by multihash)                             │
│  • Nix package manager                                              │
│  • Content delivery networks                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.1 CAS Implementation

```python
import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class CASStats:
    total_objects: int = 0
    total_bytes: int = 0
    deduplicated_bytes: int = 0

class ContentAddressableStore:
    """
    A simple content-addressable store using SHA-256.
    Similar to how Git stores objects.
    """

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _hash_content(self, data: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(data).hexdigest()

    def _object_path(self, content_hash: str) -> Path:
        """
        Get the filesystem path for an object.
        Uses first 2 chars as directory (like Git) to avoid
        too many files in a single directory.
        """
        return self.store_dir / content_hash[:2] / content_hash[2:]

    def put(self, data: bytes) -> str:
        """
        Store data and return its content hash.
        If data already exists, this is a no-op (deduplication).
        """
        content_hash = self._hash_content(data)
        obj_path = self._object_path(content_hash)

        if not obj_path.exists():
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(data)

        return content_hash

    def get(self, content_hash: str) -> Optional[bytes]:
        """Retrieve data by its content hash."""
        obj_path = self._object_path(content_hash)
        if not obj_path.exists():
            return None

        data = obj_path.read_bytes()

        # Verify integrity on read
        actual_hash = self._hash_content(data)
        if actual_hash != content_hash:
            raise RuntimeError(
                f"Integrity error! Expected {content_hash}, got {actual_hash}"
            )

        return data

    def exists(self, content_hash: str) -> bool:
        """Check if an object exists."""
        return self._object_path(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        """Delete an object (use with care in production)."""
        obj_path = self._object_path(content_hash)
        if obj_path.exists():
            obj_path.unlink()
            # Clean up empty directory
            try:
                obj_path.parent.rmdir()
            except OSError:
                pass
            return True
        return False

    def stats(self) -> CASStats:
        """Compute storage statistics."""
        stats = CASStats()
        for subdir in self.store_dir.iterdir():
            if subdir.is_dir():
                for obj_file in subdir.iterdir():
                    stats.total_objects += 1
                    stats.total_bytes += obj_file.stat().st_size
        return stats

# Usage
store = ContentAddressableStore("/tmp/cas_demo")

# Store some data
data1 = b"Hello, content-addressable world!"
data2 = b"Another piece of data"
data3 = b"Hello, content-addressable world!"  # Duplicate of data1!

hash1 = store.put(data1)
hash2 = store.put(data2)
hash3 = store.put(data3)

print(f"Data 1 hash: {hash1}")
print(f"Data 2 hash: {hash2}")
print(f"Data 3 hash: {hash3}")
print(f"Data 1 == Data 3 hash? {hash1 == hash3}")  # True - deduplication!

# Retrieve by hash
retrieved = store.get(hash1)
print(f"\nRetrieved: {retrieved.decode()}")
print(f"Integrity: {retrieved == data1}")

# Stats
stats = store.stats()
print(f"\nStore stats:")
print(f"  Objects: {stats.total_objects}")  # 2, not 3 (dedup!)
print(f"  Total bytes: {stats.total_bytes}")

# Clean up
import shutil
shutil.rmtree("/tmp/cas_demo")
```

### 8.2 Git-Style Object Store

```python
import hashlib
import zlib
import os
from pathlib import Path

class GitObjectStore:
    """
    Simplified Git object store.
    Git stores objects as: header + content, compressed with zlib.
    Header format: "<type> <size>\0"
    """

    def __init__(self, git_dir: str):
        self.objects_dir = Path(git_dir) / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def hash_object(self, data: bytes, obj_type: str = "blob") -> str:
        """Hash an object (like 'git hash-object')."""
        header = f"{obj_type} {len(data)}\0".encode()
        full_content = header + data
        return hashlib.sha1(full_content).hexdigest()  # Git uses SHA-1 (transitioning to SHA-256)

    def write_object(self, data: bytes, obj_type: str = "blob") -> str:
        """Write an object to the store (like 'git hash-object -w')."""
        header = f"{obj_type} {len(data)}\0".encode()
        full_content = header + data
        obj_hash = hashlib.sha1(full_content).hexdigest()

        # Store compressed
        obj_path = self.objects_dir / obj_hash[:2] / obj_hash[2:]
        if not obj_path.exists():
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            compressed = zlib.compress(full_content)
            obj_path.write_bytes(compressed)

        return obj_hash

    def read_object(self, obj_hash: str) -> tuple:
        """Read an object (like 'git cat-file')."""
        obj_path = self.objects_dir / obj_hash[:2] / obj_hash[2:]
        if not obj_path.exists():
            raise FileNotFoundError(f"Object {obj_hash} not found")

        compressed = obj_path.read_bytes()
        full_content = zlib.decompress(compressed)

        # Parse header
        null_pos = full_content.index(b'\0')
        header = full_content[:null_pos].decode()
        obj_type, size = header.split(' ')
        data = full_content[null_pos + 1:]

        assert len(data) == int(size), "Size mismatch"
        return obj_type, data

# Usage
store = GitObjectStore("/tmp/git_demo/.git")

# Store a blob (file content)
content = b"print('Hello, World!')\n"
blob_hash = store.write_object(content, "blob")
print(f"Blob hash: {blob_hash}")

# Read it back
obj_type, data = store.read_object(blob_hash)
print(f"Type: {obj_type}")
print(f"Content: {data.decode()}", end="")

# Store a tree (directory listing) - simplified
tree_content = f"100644 hello.py\0".encode() + bytes.fromhex(blob_hash)
tree_hash = store.write_object(tree_content, "tree")
print(f"\nTree hash: {tree_hash}")

# Clean up
import shutil
shutil.rmtree("/tmp/git_demo")
```

---

## 9. Timing Attacks and Constant-Time Comparison

### 9.1 The Problem

When comparing two strings character by character, the time taken depends on where the first difference occurs. An attacker can measure response times to determine how many characters of a secret value they have guessed correctly.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Timing Attack on String Comparison                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Secret MAC: "a3f2b8c9d1e4"                                        │
│                                                                      │
│  Attempt 1: "x3f2b8c9d1e4"  → Fails at position 0  → ~100 ns     │
│  Attempt 2: "a4f2b8c9d1e4"  → Fails at position 1  → ~110 ns     │
│  Attempt 3: "a3g2b8c9d1e4"  → Fails at position 2  → ~120 ns     │
│                                                                      │
│  The attacker notices each correct character adds ~10 ns.           │
│  By trying all values for each position, they can recover           │
│  the entire MAC one character at a time: O(n × 16) instead of      │
│  O(16^n).                                                           │
│                                                                      │
│  For a 32-character hex MAC:                                        │
│  Brute force: 16^32 ≈ 3.4 × 10^38 attempts                       │
│  Timing attack: 32 × 16 = 512 attempts (!)                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 Vulnerable vs Secure Comparison

```python
import hmac
import hashlib
import time
import os

# VULNERABLE: Early-exit string comparison
def insecure_compare(a: str, b: str) -> bool:
    """
    DO NOT USE! Vulnerable to timing attacks.
    Returns False as soon as a mismatch is found.
    """
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False  # Early exit leaks information!
    return True

# SECURE: Constant-time comparison
def secure_compare(a: str, b: str) -> bool:
    """
    Constant-time comparison. Takes the same time regardless
    of where (or if) the strings differ.
    """
    return hmac.compare_digest(a, b)

# Demonstrate the timing difference
secret_mac = hashlib.sha256(b"secret-key" + b"message").hexdigest()

# Generate test MACs with increasing correct prefix length
test_macs = []
for i in range(0, len(secret_mac), 4):
    correct_prefix = secret_mac[:i]
    wrong_suffix = "0" * (len(secret_mac) - i)
    test_macs.append(correct_prefix + wrong_suffix)

print("Timing Attack Demonstration")
print("=" * 60)
print(f"Secret MAC: {secret_mac[:32]}...")
print()

# Measure insecure comparison times
print("INSECURE comparison (early exit):")
for mac in test_macs[:8]:
    correct_chars = sum(a == b for a, b in zip(mac, secret_mac))
    times = []
    for _ in range(10000):
        start = time.perf_counter_ns()
        insecure_compare(mac, secret_mac)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  {correct_chars:2d} correct chars → avg {avg:6.0f} ns")

print()

# Measure secure comparison times
print("SECURE comparison (constant-time):")
for mac in test_macs[:8]:
    correct_chars = sum(a == b for a, b in zip(mac, secret_mac))
    times = []
    for _ in range(10000):
        start = time.perf_counter_ns()
        secure_compare(mac, secret_mac)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  {correct_chars:2d} correct chars → avg {avg:6.0f} ns")

print()
print("Note: Secure comparison time should be roughly constant")
print("regardless of how many characters match.")
```

### 9.3 Rules for Timing-Safe Code

```
┌─────────────────────────────────────────────────────────────────────┐
│               Rules for Constant-Time Operations                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. NEVER use == to compare secrets (MACs, tokens, passwords)       │
│     USE: hmac.compare_digest() or secrets.compare_digest()          │
│                                                                      │
│  2. NEVER return early based on secret data                          │
│     BAD:  if mac[i] != expected[i]: return False                    │
│     GOOD: result |= (mac[i] ^ expected[i])   # accumulate diffs   │
│                                                                      │
│  3. NEVER branch on secret data                                      │
│     BAD:  if secret_key[0] == 'a': ...                              │
│     GOOD: Use constant-time selection / masking                     │
│                                                                      │
│  4. NEVER index arrays with secret values                            │
│     BAD:  table[secret_byte]   # cache timing leak                 │
│     GOOD: Use constant-time lookup tables                           │
│                                                                      │
│  5. ALWAYS verify MACs before decrypting                             │
│     This prevents padding oracle attacks (Encrypt-then-MAC)        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Hash Function Attacks and Mitigations

### 10.1 Known Attacks

```
┌─────────────────────────────────────────────────────────────────────┐
│                Hash Function Attack Summary                          │
├────────────────────┬────────────────────────────────────────────────┤
│ Attack             │ Description and Status                         │
├────────────────────┼────────────────────────────────────────────────┤
│ Birthday attack    │ Find ANY collision in O(2^(n/2))              │
│                    │ SHA-256: 2^128 (safe), MD5: 2^64 (broken)     │
├────────────────────┼────────────────────────────────────────────────┤
│ Length-extension   │ Given H(m), compute H(m||pad||suffix) without │
│                    │ knowing m. Affects SHA-256, SHA-512.           │
│                    │ NOT: SHA-3, BLAKE2, HMAC, SHA-512/256         │
├────────────────────┼────────────────────────────────────────────────┤
│ Rainbow tables     │ Precomputed hash-to-password lookup tables.   │
│                    │ Defeated by salting passwords.                 │
├────────────────────┼────────────────────────────────────────────────┤
│ Dictionary attack  │ Try common passwords against a hash.          │
│                    │ Mitigated by slow password hashes (Argon2).   │
├────────────────────┼────────────────────────────────────────────────┤
│ GPU brute force    │ GPUs can compute billions of hashes/second.   │
│                    │ Mitigated by memory-hard hashes (Argon2id).   │
├────────────────────┼────────────────────────────────────────────────┤
│ SHAttered (2017)   │ Practical SHA-1 collision found by Google.    │
│                    │ Two PDFs with same SHA-1 but different content │
│                    │ SHA-1 is now considered broken for security.   │
├────────────────────┼────────────────────────────────────────────────┤
│ Multi-collision    │ Find many messages with the same hash.        │
│                    │ Easier than expected for iterative hashes.     │
│                    │ SHA-3's sponge construction is more resistant. │
└────────────────────┴────────────────────────────────────────────────┘
```

### 10.2 Length-Extension Attack

```python
# Length-extension attack demonstration
# This is why you should use HMAC instead of H(key || message)
import hashlib
import struct

def sha256_pad(message_len: int) -> bytes:
    """Compute SHA-256 padding for a message of given length."""
    bit_len = message_len * 8
    # Padding: 1 bit, then zeros, then 64-bit length
    padding = b'\x80'
    padding += b'\x00' * ((56 - (message_len + 1) % 64) % 64)
    padding += struct.pack('>Q', bit_len)
    return padding

# The vulnerability: H(secret || message) is NOT a secure MAC
#
# If an attacker knows:
#   - H(secret || message)  (the MAC)
#   - len(secret)           (or can guess it)
#   - message               (the original message)
#
# They can compute H(secret || message || padding || extension)
# WITHOUT knowing the secret!
#
# This is because SHA-256 processes data in blocks, and the hash
# state after processing (secret || message || padding) is exactly
# the public hash value. The attacker can resume hashing from there.

print("Length-Extension Attack Concept:")
print("=" * 60)
print()
print("VULNERABLE construction: MAC = SHA-256(secret || message)")
print("An attacker who knows MAC and len(secret) can extend the message.")
print()
print("SAFE alternatives:")
print("  1. HMAC-SHA256(key, message)     — HMAC construction")
print("  2. SHA-3-256(key || message)      — SHA-3 is not vulnerable")
print("  3. BLAKE2b(message, key=key)      — Built-in keying")
print("  4. SHA-512/256(key || message)    — Truncated hash")
```

---

## 11. Exercises

### Exercise 1: Hash Explorer (Beginner)

Write a Python program that:
1. Takes a filename as input
2. Computes SHA-256, SHA-3-256, BLAKE2b-256, and BLAKE3 hashes
3. Displays all hashes and the time taken for each
4. Compares the speed of each algorithm on a 100 MB file
5. Verifies the file has not been modified by comparing hashes

### Exercise 2: Password Cracker Defense (Intermediate)

Implement a password cracking simulation:
1. Create a list of 1000 "user accounts" with passwords hashed using:
   a. Plain SHA-256 (no salt)
   b. SHA-256 with unique salt
   c. bcrypt (cost=10)
   d. Argon2id (default parameters)
2. Attempt to crack all passwords using a dictionary of 10,000 common passwords
3. Measure and compare the time needed for each hashing method
4. Generate a report showing cracked vs uncracked percentages

### Exercise 3: Merkle Tree File Verifier (Intermediate)

Build a file integrity verification system using Merkle trees:
1. Given a directory, compute a Merkle tree over all files (sorted by path)
2. Store the Merkle root and tree structure
3. Later, verify any individual file by recomputing only O(log n) hashes
4. Generate a Merkle proof that can be independently verified
5. Handle file additions and deletions efficiently

### Exercise 4: HMAC-based API Authentication (Intermediate)

Implement a complete API authentication system:
1. Server issues API key + secret to each client
2. Client signs each request with HMAC-SHA256:
   - Include: method, path, timestamp, body hash, nonce
3. Server verifies: valid signature, fresh timestamp (within 5 min), unused nonce
4. Implement replay attack protection using a nonce cache
5. Handle clock skew between client and server

### Exercise 5: Content-Addressable File Sync (Advanced)

Build a simplified file synchronization system (like rsync with content addressing):
1. Both sides maintain a CAS of their files
2. To sync, exchange only the list of content hashes
3. Transfer only the blocks that the other side is missing
4. Verify integrity of all received blocks
5. Use a Merkle tree to efficiently detect which parts of large files differ

### Exercise 6: Timing Attack Lab (Advanced)

Build a controlled timing attack:
1. Create a server that verifies API tokens using insecure comparison (`==`)
2. Write a client that measures response times to guess the token one character at a time
3. Demonstrate that the attack recovers the full token
4. Fix the server to use `hmac.compare_digest()`
5. Show that the attack no longer works
6. Discuss other timing side channels (cache timing, power analysis)

---

## References

- Aumasson, J.P. (2017). *Serious Cryptography*. No Starch Press.
- NIST FIPS 180-4: Secure Hash Standard (SHA-2)
- NIST FIPS 202: SHA-3 Standard
- RFC 7693: The BLAKE2 Cryptographic Hash
- BLAKE3 Specification - https://github.com/BLAKE3-team/BLAKE3-specs
- RFC 2104: HMAC: Keyed-Hashing for Message Authentication
- OWASP Password Storage Cheat Sheet - https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
- Argon2 Reference - https://github.com/P-H-C/phc-winner-argon2
- Merkle, R.C. (1979). "A Certified Digital Signature"

---

**Previous**: [02. Cryptography Basics](./02_Cryptography_Basics.md) | **Next**: [04. TLS/SSL and Public Key Infrastructure](./04_TLS_and_PKI.md)

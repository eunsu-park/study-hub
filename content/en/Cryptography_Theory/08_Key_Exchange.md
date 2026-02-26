# Lesson 8: Key Exchange

**Previous**: [Digital Signatures](./07_Digital_Signatures.md) | **Next**: [PKI and Certificates](./09_PKI_and_Certificates.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the key distribution problem and why it is fundamental to secure communication
2. Derive the Diffie-Hellman key exchange protocol from discrete logarithm assumptions
3. Implement both classical DH and Elliptic Curve DH (ECDH) in Python
4. Compare authenticated vs. unauthenticated key exchange and identify man-in-the-middle vulnerabilities
5. Describe forward secrecy and explain why ephemeral keys are critical for modern protocols
6. Outline the Signal protocol's X3DH handshake and its security properties
7. Apply key derivation functions (HKDF) to transform shared secrets into usable keys

---

Every cryptographic system we have studied so far — symmetric ciphers (Lesson 3), public-key encryption (Lesson 5), digital signatures (Lesson 7) — assumes that the communicating parties already share some secret or know each other's public keys. But how do two strangers, communicating over an insecure channel with an eavesdropper listening to every bit, agree on a shared secret in the first place? This is the **key distribution problem**, and its elegant solution — the Diffie-Hellman protocol — is arguably the most important invention in the history of cryptography. It made the entire edifice of modern secure communication possible.

## Table of Contents

1. [The Key Distribution Problem](#1-the-key-distribution-problem)
2. [Diffie-Hellman Key Exchange](#2-diffie-hellman-key-exchange)
3. [Elliptic Curve Diffie-Hellman (ECDH)](#3-elliptic-curve-diffie-hellman-ecdh)
4. [Man-in-the-Middle and Authenticated Key Exchange](#4-man-in-the-middle-and-authenticated-key-exchange)
5. [Forward Secrecy and Ephemeral Keys](#5-forward-secrecy-and-ephemeral-keys)
6. [The Signal Protocol and X3DH](#6-the-signal-protocol-and-x3dh)
7. [Key Derivation Functions](#7-key-derivation-functions)
8. [Summary](#8-summary)
9. [Exercises](#9-exercises)

---

## 1. The Key Distribution Problem

### 1.1 The Fundamental Dilemma

Suppose Alice wants to send Bob an encrypted message using AES-256 (Lesson 3). She needs a 256-bit key that only she and Bob know. But how does she get this key to Bob?

- **Physical courier**: Secure but impractical at scale. How do you exchange keys with a website you've never visited?
- **Trusted third party (Kerberos model)**: Requires a central server that everyone trusts — a single point of failure.
- **Pre-shared keys**: Works for small groups but scales as $O(n^2)$. A network of 1,000 users needs $\frac{1000 \times 999}{2} = 499,500$ unique keys.

### 1.2 The Breakthrough Idea

In 1976, Whitfield Diffie and Martin Hellman published "New Directions in Cryptography," proposing that two parties could agree on a shared secret over a public channel without ever transmitting the secret itself. This was revolutionary — it seemed to violate intuition about information exchange.

> **Analogy:** Diffie-Hellman is like mixing paint colors. Alice and Bob agree on a common base color (public). Alice adds her secret color and sends the mixture to Bob; Bob adds his secret color and sends his mixture to Alice. Now each adds their own secret color to the other's mixture. Both arrive at the same final color — but an eavesdropper who saw only the two intermediate mixtures cannot unmix them to recover the individual secret colors.

---

## 2. Diffie-Hellman Key Exchange

### 2.1 Mathematical Foundation

The security of DH rests on the **Discrete Logarithm Problem (DLP)**:

Given a prime $p$, a generator $g$ of $\mathbb{Z}_p^*$, and a value $A = g^a \bmod p$, finding $a$ is computationally infeasible for large $p$.

The related **Decisional Diffie-Hellman (DDH) assumption** states:

$$
(g^a, g^b, g^{ab}) \approx_c (g^a, g^b, g^r)
$$

where $r$ is random. That is, $g^{ab}$ looks random to anyone who only knows $g^a$ and $g^b$.

### 2.2 The Protocol

**Public parameters**: A large prime $p$ and a generator $g$ of $\mathbb{Z}_p^*$.

| Step | Alice | Channel | Bob |
|------|-------|---------|-----|
| 1 | Choose random $a \in \{2, \ldots, p-2\}$ | | Choose random $b \in \{2, \ldots, p-2\}$ |
| 2 | Compute $A = g^a \bmod p$ | $A \longrightarrow$ | |
| 3 | | $\longleftarrow B$ | Compute $B = g^b \bmod p$ |
| 4 | Compute $s = B^a \bmod p$ | | Compute $s = A^b \bmod p$ |

**Correctness**: Both compute the same value because:

$$
B^a = (g^b)^a = g^{ab} = (g^a)^b = A^b \pmod{p}
$$

### 2.3 Implementation

```python
"""
Diffie-Hellman Key Exchange — Educational Implementation

Why we use Python's built-in pow(base, exp, mod):
  Python's three-argument pow() uses fast modular exponentiation
  (square-and-multiply), making it efficient even for large numbers.
"""

import secrets
import hashlib


def generate_dh_parameters():
    """
    Generate DH parameters (p, g).

    In production, use well-known groups like RFC 3526 Group 14 (2048-bit).
    Here we use a small safe prime for demonstration.

    Why a safe prime? If p = 2q + 1 where q is also prime, the subgroup
    of order q has no small subgroups to leak information through.
    """
    # RFC 3526 Group 14 (2048-bit MODP) — truncated for readability
    # In practice, you'd use the full value
    p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
    g = 2
    return p, g


def dh_key_exchange():
    """Simulate a complete Diffie-Hellman key exchange."""
    p, g = generate_dh_parameters()

    # Alice's side
    a = secrets.randbelow(p - 2) + 2  # Why +2: avoid 0 and 1
    A = pow(g, a, p)                   # Alice's public value

    # Bob's side
    b = secrets.randbelow(p - 2) + 2
    B = pow(g, b, p)                   # Bob's public value

    # Key agreement
    # Why both computations yield the same result:
    # Alice computes B^a mod p = g^(ba) mod p
    # Bob computes A^b mod p = g^(ab) mod p
    # Since ab = ba, the shared secret is identical.
    shared_secret_alice = pow(B, a, p)
    shared_secret_bob = pow(A, b, p)

    assert shared_secret_alice == shared_secret_bob, "Key exchange failed!"

    # Why we hash the shared secret:
    # The raw DH output has mathematical structure (it's an element of Z_p*).
    # Hashing extracts a uniformly random key from this structured value.
    key = hashlib.sha256(
        shared_secret_alice.to_bytes(256, 'big')
    ).digest()

    print(f"DH parameters: p has {p.bit_length()} bits, g = {g}")
    print(f"Alice's public value A: {hex(A)[:20]}...")
    print(f"Bob's public value B:   {hex(B)[:20]}...")
    print(f"Shared secret matches:  {shared_secret_alice == shared_secret_bob}")
    print(f"Derived 256-bit key:    {key.hex()[:32]}...")

    return key


if __name__ == "__main__":
    dh_key_exchange()
```

### 2.4 Parameter Selection

Choosing DH parameters carelessly can be catastrophic:

| Parameter | Minimum | Recommended | Why |
|-----------|---------|-------------|-----|
| $p$ bit length | 1024 | 2048+ | 1024-bit DLP is within reach of nation-states (Logjam attack) |
| $g$ | 2 or 5 | 2 | Small generators are fine; security depends on $p$ |
| Private exponent $a$ | 160 bits | 256 bits | Must be at least twice the target security level |

> **Warning**: Many implementations reuse the same DH parameters across all connections. The 2015 Logjam attack showed that precomputing a single discrete log table for a commonly-used 1024-bit prime could break connections to 80% of TLS servers.

---

## 3. Elliptic Curve Diffie-Hellman (ECDH)

### 3.1 Why Elliptic Curves?

Classical DH requires 2048-bit or larger primes for adequate security. Elliptic Curve DH achieves equivalent security with much smaller keys:

| Security Level | DH Key Size | ECDH Key Size | Ratio |
|---------------|-------------|---------------|-------|
| 128-bit | 3072 bits | 256 bits | 12:1 |
| 192-bit | 7680 bits | 384 bits | 20:1 |
| 256-bit | 15360 bits | 512 bits | 30:1 |

### 3.2 ECDH Protocol

Instead of working in $\mathbb{Z}_p^*$, we work on an elliptic curve $E$ over $\mathbb{F}_p$ with a base point $G$ of order $n$ (see Lesson 6 for ECC fundamentals).

| Step | Alice | Bob |
|------|-------|-----|
| 1 | Choose random $a \in \{1, \ldots, n-1\}$ | Choose random $b \in \{1, \ldots, n-1\}$ |
| 2 | Compute $A = aG$ (point multiplication) | Compute $B = bG$ |
| 3 | Send $A$ to Bob | Send $B$ to Alice |
| 4 | Compute $S = aB = a(bG)$ | Compute $S = bA = b(aG)$ |

Both arrive at the same point $S = abG$ because point multiplication is commutative. The shared secret is typically the $x$-coordinate of $S$.

### 3.3 ECDH with the `cryptography` Library

```python
"""
ECDH Key Exchange using the cryptography library.

Why SECP256R1 (P-256)? It's the NIST-recommended curve for 128-bit
security, widely supported in TLS, and hardware-accelerated on many CPUs.
Curve25519 is the modern alternative preferred by many cryptographers
for its simpler, constant-time implementation.
"""

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def ecdh_key_exchange():
    """Demonstrate ECDH using NIST P-256 curve."""

    # Alice generates her key pair
    # Why ec.SECP256R1(): 256-bit prime-order curve, equivalent to 3072-bit RSA
    alice_private = ec.generate_private_key(ec.SECP256R1())
    alice_public = alice_private.public_key()

    # Bob generates his key pair
    bob_private = ec.generate_private_key(ec.SECP256R1())
    bob_public = bob_private.public_key()

    # Key agreement
    # Why ECDH().exchange() instead of manual math?
    # The library handles point validation (rejecting invalid curve points),
    # constant-time scalar multiplication, and proper encoding.
    alice_shared = alice_private.exchange(ec.ECDH(), bob_public)
    bob_shared = bob_private.exchange(ec.ECDH(), alice_public)

    assert alice_shared == bob_shared

    # Why HKDF? The raw ECDH output (x-coordinate of the shared point)
    # is not uniformly random — it's biased by the curve equation.
    # HKDF extracts and expands it into a proper symmetric key.
    alice_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,            # 256-bit key
        salt=None,            # Optional but recommended in practice
        info=b"ecdh-demo",    # Context separation (binds key to purpose)
    ).derive(alice_shared)

    print(f"ECDH shared secret (hex): {alice_shared.hex()[:32]}...")
    print(f"Derived key (hex):        {alice_key.hex()}")
    return alice_key


def ecdh_curve25519():
    """ECDH using Curve25519 (modern, preferred curve)."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    # Why Curve25519? Designed by Daniel Bernstein with:
    # - Built-in resistance to timing attacks (constant-time by design)
    # - No need for point validation (every 32-byte string is valid)
    # - Faster than P-256 in software implementations
    alice_private = X25519PrivateKey.generate()
    alice_public = alice_private.public_key()

    bob_private = X25519PrivateKey.generate()
    bob_public = bob_private.public_key()

    alice_shared = alice_private.exchange(bob_public)
    bob_shared = bob_private.exchange(alice_public)

    assert alice_shared == bob_shared
    print(f"X25519 shared secret: {alice_shared.hex()[:32]}...")
    return alice_shared


if __name__ == "__main__":
    print("=== ECDH with P-256 ===")
    ecdh_key_exchange()
    print("\n=== ECDH with Curve25519 ===")
    ecdh_curve25519()
```

---

## 4. Man-in-the-Middle and Authenticated Key Exchange

### 4.1 The MitM Attack on Bare DH

Plain Diffie-Hellman is vulnerable to a **man-in-the-middle (MitM)** attack because neither party authenticates the other:

```
Alice                    Eve (attacker)                Bob
  |                          |                          |
  |--- A = g^a mod p ------->|                          |
  |                          |--- E1 = g^e1 mod p ----->|
  |                          |<--- B = g^b mod p -------|
  |<--- E2 = g^e2 mod p ----|                          |
  |                          |                          |
  s1 = E2^a (Alice-Eve)     s1 = A^e2, s2 = B^e1      s2 = E1^b (Eve-Bob)
```

Eve establishes separate shared secrets with Alice and Bob. She decrypts, reads, re-encrypts, and forwards all messages. Neither Alice nor Bob detects the interception.

### 4.2 Station-to-Station (STS) Protocol

The STS protocol adds authentication by signing the DH exchange values:

1. Alice sends $A = g^a \bmod p$
2. Bob sends $B = g^b \bmod p$ along with $\text{Sign}_{B}(B \| A)$ encrypted under $K = g^{ab}$
3. Alice verifies Bob's signature, then sends $\text{Sign}_{A}(A \| B)$ encrypted under $K$

This binds the identities (via signatures) to the exchange values, preventing Eve from substituting her own values.

### 4.3 Other Authenticated Key Exchange Protocols

| Protocol | Authentication Method | Used In |
|----------|----------------------|---------|
| STS | Digital signatures | IPsec (IKE) |
| SIGMA | Signatures + MAC | IKEv2 |
| HMQV | Implicit (no signatures needed) | Academic, some smart cards |
| TLS 1.3 | Server certificate + optional client cert | HTTPS, every web browser |
| Noise Framework | Multiple patterns (XX, IK, NK, etc.) | WireGuard, Signal |

---

## 5. Forward Secrecy and Ephemeral Keys

### 5.1 Why Long-Term Keys Are Not Enough

Consider a scenario: a server uses a static RSA key pair for years. An adversary records all encrypted traffic. Years later, the server is compromised and the private key is stolen. Now the adversary can decrypt **all previously recorded traffic**.

> **Analogy:** Using a static key for key exchange is like using the same master key to lock every letter you've ever sent. If someone steals the master key, they can open every letter — past, present, and future.

### 5.2 Forward Secrecy (FS)

**Forward secrecy** (also called **perfect forward secrecy**, PFS) guarantees that compromise of long-term keys does not compromise past session keys.

The mechanism is simple: use **ephemeral (one-time) DH keys** for each session.

```
Session 1: Alice generates a_1, Bob generates b_1 → K_1 = g^(a_1 * b_1)
Session 2: Alice generates a_2, Bob generates b_2 → K_2 = g^(a_2 * b_2)
...
```

After each session, $a_i$ and $b_i$ are securely deleted. Even if an attacker later obtains Alice's long-term signing key, they cannot recover $a_1$ or $b_1$ and thus cannot compute $K_1$.

### 5.3 Ephemeral DH (DHE / ECDHE) in TLS

In TLS 1.3, **all** key exchanges use ephemeral keys (the static RSA key exchange mode was removed entirely). The naming convention:

- **DH**: Static Diffie-Hellman (deprecated)
- **DHE**: Ephemeral Diffie-Hellman (forward secret)
- **ECDHE**: Ephemeral Elliptic Curve DH (forward secret, smaller keys)

```python
"""
Demonstrating ephemeral key exchange sessions.

Why delete private keys after use? Each session's privacy depends
on the secrecy of its ephemeral key. Once the shared secret is
derived, the private key serves no further purpose and becomes a
liability if retained.
"""

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def ephemeral_session(session_id: int) -> bytes:
    """Simulate one ephemeral key exchange session."""
    # Generate fresh keys — never reused
    alice_eph = X25519PrivateKey.generate()
    bob_eph = X25519PrivateKey.generate()

    alice_pub = alice_eph.public_key()
    bob_pub = bob_eph.public_key()

    # Key agreement
    shared = alice_eph.exchange(bob_pub)

    # Derive session key
    session_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"session-{session_id}".encode(),
    ).derive(shared)

    # In a real system, alice_eph and bob_eph would be securely
    # zeroed from memory here. Python doesn't guarantee this,
    # but C/Rust implementations use explicit zeroization.
    del alice_eph, bob_eph

    return session_key


# Each session produces an independent key
for i in range(3):
    key = ephemeral_session(i)
    print(f"Session {i} key: {key.hex()[:16]}...")
    # Compromising one session key reveals nothing about others
```

### 5.4 Post-Snowden Impact

The 2013 Snowden revelations showed that intelligence agencies were recording encrypted traffic at scale, hoping to decrypt it later if keys were compromised. This made forward secrecy an urgent priority:

- **Before 2013**: ~30% of HTTPS sites supported forward secrecy
- **After 2013**: >90% adoption, with TLS 1.3 making it mandatory
- **"Harvest now, decrypt later"**: This threat remains relevant for post-quantum migration (Lesson 11)

---

## 6. The Signal Protocol and X3DH

### 6.1 The Asynchronous Key Exchange Problem

Standard DH requires both parties to be online simultaneously. But messaging apps need to work asynchronously — Alice sends a message while Bob's phone is off. The Signal protocol solves this with **X3DH (Extended Triple Diffie-Hellman)**.

### 6.2 X3DH Key Agreement

Bob pre-publishes several key types to a server:

- **IK_B**: Identity key (long-term, for authentication)
- **SPK_B**: Signed pre-key (medium-term, rotated periodically)
- **OPK_B**: One-time pre-keys (each used exactly once)

When Alice wants to message Bob (who is offline):

$$
\text{SK} = \text{KDF}(\text{DH}(IK_A, SPK_B) \| \text{DH}(EK_A, IK_B) \| \text{DH}(EK_A, SPK_B) \| \text{DH}(EK_A, OPK_B))
$$

Where $EK_A$ is Alice's ephemeral key generated for this session.

### 6.3 Why Four DH Operations?

Each DH computation provides a different security property:

| DH Pair | Property |
|---------|----------|
| $\text{DH}(IK_A, SPK_B)$ | Mutual authentication (both long-term keys involved) |
| $\text{DH}(EK_A, IK_B)$ | If Alice's IK is compromised, past sessions remain secure |
| $\text{DH}(EK_A, SPK_B)$ | Forward secrecy (ephemeral key is deleted after use) |
| $\text{DH}(EK_A, OPK_B)$ | Prevents replay attacks (one-time key consumed) |

### 6.4 Double Ratchet (Overview)

After X3DH establishes the initial shared secret, the **Double Ratchet algorithm** ensures that each message uses a unique key. It combines:

1. **DH ratchet**: New DH exchange with every message round-trip (provides forward secrecy for each message)
2. **Symmetric ratchet**: Hash-based key chain for messages within a single DH round (provides ordering and efficiency)

We explore the Double Ratchet in detail in Lesson 14 (Applied Cryptographic Protocols).

---

## 7. Key Derivation Functions

### 7.1 Why Not Use the Shared Secret Directly?

The raw output of DH/ECDH has mathematical structure and is not uniformly random. A **Key Derivation Function (KDF)** transforms it into one or more cryptographically strong keys.

### 7.2 HKDF (HMAC-Based KDF)

HKDF (RFC 5869) operates in two stages:

1. **Extract**: Condense the input keying material into a pseudorandom key
   $$\text{PRK} = \text{HMAC-Hash}(\text{salt}, \text{IKM})$$

2. **Expand**: Produce output keying material of desired length
   $$\text{OKM} = T_1 \| T_2 \| \ldots$$
   where $T_i = \text{HMAC-Hash}(\text{PRK}, T_{i-1} \| \text{info} \| i)$

```python
"""
HKDF: Extracting multiple keys from a single shared secret.

Why separate extract and expand? The extract step handles potentially
biased or non-uniform input (like a DH shared secret). The expand step
then generates as many independent-looking keys as needed from the
extracted PRK.
"""

import hmac
import hashlib
import math


def hkdf_extract(salt: bytes, ikm: bytes, hash_algo=hashlib.sha256) -> bytes:
    """
    Extract: condense input keying material into a PRK.

    Why salt? It ensures that even if the IKM has some structure,
    the PRK is uniformly distributed. If no salt is available,
    a string of zeros (hash length) is used.
    """
    if not salt:
        salt = b'\x00' * hash_algo().digest_size
    return hmac.new(salt, ikm, hash_algo).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int,
                hash_algo=hashlib.sha256) -> bytes:
    """
    Expand: derive output keying material from PRK.

    Why info? It binds the derived key to a specific context.
    Using different info values from the same PRK produces
    independent keys (e.g., one for encryption, one for MAC).
    """
    hash_len = hash_algo().digest_size
    n = math.ceil(length / hash_len)

    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hash_algo).digest()
        okm += t

    return okm[:length]


def hkdf(ikm: bytes, length: int, salt: bytes = b"",
          info: bytes = b"") -> bytes:
    """Full HKDF: extract then expand."""
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)


# Example: derive encryption key and MAC key from one DH secret
dh_shared_secret = bytes.fromhex(
    "a3f2b8c1d4e5f607182930a1b2c3d4e5f60718293041526374859607a8b9c0d1"
)

enc_key = hkdf(dh_shared_secret, 32, info=b"encryption")
mac_key = hkdf(dh_shared_secret, 32, info=b"authentication")

print(f"Encryption key: {enc_key.hex()}")
print(f"MAC key:        {mac_key.hex()}")
# These keys are cryptographically independent despite sharing a source
assert enc_key != mac_key
```

### 7.3 Other KDFs

| KDF | Based On | Used In |
|-----|----------|---------|
| HKDF | HMAC | TLS 1.3, Signal, Noise |
| NIST SP 800-108 | HMAC or CMAC | Government systems |
| PBKDF2 | HMAC (iterated) | Password hashing (but Argon2 preferred) |
| X963KDF | Hash function | ECIES (Lesson 6) |

> **Note**: HKDF is designed for high-entropy inputs (like DH shared secrets). For passwords (low entropy), use Argon2 or scrypt — they intentionally waste time and memory to resist brute force.

---

## 8. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Key distribution problem | Cannot securely transmit keys over insecure channels without DH |
| Diffie-Hellman | $g^{ab} \bmod p$ — elegant but requires authentication |
| ECDH | Same idea on elliptic curves; 256-bit keys match 3072-bit DH |
| MitM attack | Bare DH is vulnerable; authentication (STS, certificates) is mandatory |
| Forward secrecy | Ephemeral keys ensure past sessions are safe even if long-term keys leak |
| X3DH | Asynchronous authenticated key exchange (Signal protocol) |
| HKDF | Extract-then-expand paradigm for deriving keys from shared secrets |

---

## 9. Exercises

### Exercise 1: Small DH by Hand (Conceptual)

Given $p = 23$, $g = 5$, $a = 6$, $b = 15$:

1. Compute Alice's public value $A = g^a \bmod p$
2. Compute Bob's public value $B = g^b \bmod p$
3. Compute the shared secret from both sides and verify they match
4. As the eavesdropper Eve, you see $A$ and $B$ but not $a$ or $b$. Try to find the shared secret. How many operations does brute force require? How does this scale with the size of $p$?

### Exercise 2: DH Parameter Validation (Coding)

Write a Python function `validate_dh_params(p, g, A)` that checks:
- $p$ is prime
- $g$ is a generator of a large subgroup of $\mathbb{Z}_p^*$
- $A$ is in the range $[2, p-2]$ (rejects trivial values 0, 1, $p-1$)
- $A^q \equiv 1 \pmod{p}$ where $q = (p-1)/2$ (for safe primes)

### Exercise 3: Forward Secrecy Simulation (Coding)

Implement a chat simulation with two modes:
1. **Static key mode**: Reuse the same DH key pair for all messages
2. **Ephemeral key mode**: Generate fresh keys for each message

Simulate an attacker who compromises one session key. Show that in static mode, all messages are compromised, while in ephemeral mode, only one message is exposed.

### Exercise 4: HKDF Test Vectors (Coding)

Implement HKDF from scratch (without using a library) and verify your implementation against the test vectors in RFC 5869, Appendix A. Your implementation should pass all three test cases.

### Exercise 5: X3DH Walkthrough (Challenging)

Implement a simplified X3DH protocol:
1. Bob publishes $IK_B$, $SPK_B$ (signed by $IK_B$), and a set of $OPK_B$
2. Alice performs the four DH operations and derives a shared key
3. Alice encrypts an initial message with this key
4. Bob receives and decrypts

Focus on correctness rather than security (you may omit some checks a production implementation would need). Verify that Alice and Bob derive the same key.

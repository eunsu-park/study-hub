# Lesson 7: Digital Signatures

**Previous**: [Elliptic Curve Cryptography](./06_Elliptic_Curve_Cryptography.md) | **Next**: [Key Exchange](./08_Key_Exchange.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the security goals of digital signatures (authentication, integrity, non-repudiation) and contrast them with MACs
2. Describe and implement RSA signatures with PSS padding
3. Implement ECDSA and trace through the signing and verification algorithms
4. Explain why nonce reuse in ECDSA is catastrophic, using the PlayStation 3 hack as a case study
5. Describe EdDSA/Ed25519 and explain why deterministic nonces eliminate an entire class of vulnerabilities
6. Compare Schnorr signatures, blind signatures, and aggregate signatures

---

A digital signature is the cryptographic equivalent of a handwritten signature — but far more powerful. While a handwritten signature can be forged, copied, and detached from its document, a digital signature mathematically binds the signer's identity to the exact contents of the message. Any modification, no matter how small, invalidates the signature. Digital signatures are the foundation of code signing, certificate authorities, blockchain transactions, and secure software updates. This lesson covers the major signature schemes from RSA-PSS through ECDSA to the modern Ed25519.

> **Analogy:** A digital signature is like a wax seal — it proves the sender's identity and that the message has not been tampered with. But unlike a wax seal, anyone can verify it (using the public key), and it is mathematically bound to the specific message (changing even one bit breaks the seal).

## Table of Contents

1. [Digital Signature Concepts](#1-digital-signature-concepts)
2. [RSA Signatures](#2-rsa-signatures)
3. [RSA-PSS: Probabilistic Signature Scheme](#3-rsa-pss-probabilistic-signature-scheme)
4. [ECDSA: Elliptic Curve Digital Signature Algorithm](#4-ecdsa-elliptic-curve-digital-signature-algorithm)
5. [Nonce Reuse: The PlayStation 3 Disaster](#5-nonce-reuse-the-playstation-3-disaster)
6. [EdDSA and Ed25519](#6-eddsa-and-ed25519)
7. [Schnorr Signatures](#7-schnorr-signatures)
8. [Advanced Signature Schemes](#8-advanced-signature-schemes)
9. [Security Models](#9-security-models)
10. [Exercises](#10-exercises)

---

## 1. Digital Signature Concepts

### 1.1 What Signatures Provide

| Property | Description | MAC provides? | Signature provides? |
|----------|-------------|---------------|-------------------|
| **Authentication** | Verify who sent the message | Yes | Yes |
| **Integrity** | Detect if message was modified | Yes | Yes |
| **Non-repudiation** | Signer cannot deny signing | **No** | **Yes** |
| **Public verifiability** | Anyone can verify | **No** (shared secret) | **Yes** (public key) |

Non-repudiation is the critical difference. With HMAC, both parties share the key, so either party could have created the MAC. With a digital signature, only the holder of the private key could have created the signature.

### 1.2 Signature Scheme Components

A digital signature scheme consists of three algorithms:

1. **KeyGen()** $\rightarrow$ (public key $pk$, private key $sk$)
2. **Sign($sk$, $m$)** $\rightarrow$ signature $\sigma$
3. **Verify($pk$, $m$, $\sigma$)** $\rightarrow$ accept/reject

**Correctness:** For any $(pk, sk) \leftarrow \text{KeyGen}()$ and any message $m$:

$$\text{Verify}(pk, m, \text{Sign}(sk, m)) = \text{accept}$$

### 1.3 Hash-Then-Sign Paradigm

All practical signature schemes sign a **hash** of the message, not the message itself:

$$\sigma = \text{Sign}(sk, H(m))$$

**Why hash first:**
- Efficiency: sign a fixed-size digest instead of an arbitrary-length message
- Security: prevents algebraic attacks that exploit the structure of the signing function (as we saw with textbook RSA in Lesson 5, Section 4)

---

## 2. RSA Signatures

### 2.1 Textbook RSA Signatures

**Sign:** $\sigma = H(m)^d \bmod n$

**Verify:** Check that $\sigma^e \bmod n = H(m)$

As discussed in Lesson 5, textbook RSA signatures are vulnerable to existential forgery. The signer must use a padding scheme.

### 2.2 PKCS#1 v1.5 Signatures

The most widely deployed RSA signature padding (though deprecated for new designs):

```
0x00 || 0x01 || [0xFF bytes] || 0x00 || [DigestInfo ASN.1 encoding of hash]
```

```python
import hashlib

def pkcs1_v15_sign(message, n, d):
    """PKCS#1 v1.5 signature with SHA-256.

    Why the DigestInfo encoding: it binds the hash algorithm identifier
    to the signature, preventing cross-algorithm attacks. Without it,
    an attacker could claim the signature was made with a different
    (weaker) hash function.
    """
    # SHA-256 DigestInfo header (DER-encoded ASN.1)
    DIGEST_INFO_SHA256 = bytes.fromhex(
        "3031300d060960864801650304020105000420"
    )

    h = hashlib.sha256(message).digest()
    k = (n.bit_length() + 7) // 8  # Modulus size in bytes

    # Construct padded message
    T = DIGEST_INFO_SHA256 + h
    ps_len = k - len(T) - 3
    if ps_len < 8:
        raise ValueError("Modulus too small for this message")

    em = b'\x00\x01' + b'\xff' * ps_len + b'\x00' + T

    # Sign: m^d mod n
    m_int = int.from_bytes(em, 'big')
    sig = pow(m_int, d, n)

    return sig.to_bytes(k, 'big')

def pkcs1_v15_verify(message, signature, n, e):
    """PKCS#1 v1.5 signature verification."""
    DIGEST_INFO_SHA256 = bytes.fromhex(
        "3031300d060960864801650304020105000420"
    )

    k = (n.bit_length() + 7) // 8
    sig_int = int.from_bytes(signature, 'big')
    em_int = pow(sig_int, e, n)
    em = em_int.to_bytes(k, 'big')

    # Verify padding structure
    if em[0:2] != b'\x00\x01':
        return False

    # Find 0x00 separator
    sep_idx = em.index(b'\x00', 2)
    if em[2:sep_idx] != b'\xff' * (sep_idx - 2):
        return False

    # Extract and verify digest info + hash
    T = em[sep_idx + 1:]
    expected_hash = hashlib.sha256(message).digest()
    expected_T = DIGEST_INFO_SHA256 + expected_hash

    return T == expected_T
```

### 2.3 Bleichenbacher's Signature Forgery (2006)

Some implementations verified PKCS#1 v1.5 signatures incorrectly by only checking the hash at the beginning of the padding, ignoring garbage bytes at the end. This allowed forgery when $e = 3$ (small public exponent).

**Lesson:** Always perform **strict** parsing of the padded message — verify the entire structure, not just the beginning.

---

## 3. RSA-PSS: Probabilistic Signature Scheme

### 3.1 Why PSS Over PKCS#1 v1.5?

| Property | PKCS#1 v1.5 | RSA-PSS |
|----------|-------------|---------|
| Randomized | No (deterministic) | Yes (random salt) |
| Security proof | Heuristic | Provably secure (ROM) |
| Standard status | Legacy | Recommended |

PSS adds a random salt, making the same message produce different signatures each time. This provides provable security in the random oracle model.

### 3.2 PSS Structure (Simplified)

```
mHash = Hash(message)
M' = (8 zero bytes) || mHash || salt
H = Hash(M')
DB = padding || 0x01 || salt
maskedDB = DB XOR MGF1(H)
EM = maskedDB || H || 0xBC
```

```python
def rsa_pss_concept():
    """Explain RSA-PSS structure and its security advantages.

    Why randomized signatures:
    1. Provable security reduction to RSA problem
    2. Each signature is unique (even for the same message)
    3. Prevents deterministic-signature attacks
    4. Recommended by NIST SP 800-131A and PKCS#1 v2.2
    """
    import os

    message = b"Sign this document"
    salt = os.urandom(32)  # Random salt for each signature

    # Step 1: Hash the message
    m_hash = hashlib.sha256(message).digest()

    # Step 2: Create M' = (8 zero bytes) || mHash || salt
    m_prime = b'\x00' * 8 + m_hash + salt

    # Step 3: H = Hash(M')
    H = hashlib.sha256(m_prime).digest()

    print(f"Message hash: {m_hash.hex()[:32]}...")
    print(f"Salt:         {salt.hex()[:32]}...")
    print(f"H (from M'):  {H.hex()[:32]}...")
    print()
    print("Two signatures of the same message produce different H values")
    print("because the salt is random each time.")
    print()

    # Sign again with different salt
    salt2 = os.urandom(32)
    m_prime2 = b'\x00' * 8 + m_hash + salt2
    H2 = hashlib.sha256(m_prime2).digest()
    print(f"H (salt 2):   {H2.hex()[:32]}...")
    print(f"H values differ: {H != H2}")

rsa_pss_concept()
```

---

## 4. ECDSA: Elliptic Curve Digital Signature Algorithm

### 4.1 Parameters

- Elliptic curve $E$ over $\mathbb{F}_p$ with base point $G$ of order $n$
- Private key: $d \in \{1, \ldots, n-1\}$
- Public key: $Q = dG$

### 4.2 Signing Algorithm

To sign message $m$:

1. Compute $e = H(m)$ (truncated to the bit length of $n$)
2. Choose a **random nonce** $k \in \{1, \ldots, n-1\}$
3. Compute $R = kG = (x_R, y_R)$
4. Compute $r = x_R \bmod n$ (if $r = 0$, choose a new $k$)
5. Compute $s = k^{-1}(e + rd) \bmod n$ (if $s = 0$, choose a new $k$)
6. Signature is $(r, s)$

### 4.3 Verification Algorithm

To verify signature $(r, s)$ on message $m$ with public key $Q$:

1. Compute $e = H(m)$
2. Compute $w = s^{-1} \bmod n$
3. Compute $u_1 = ew \bmod n$ and $u_2 = rw \bmod n$
4. Compute $R' = u_1 G + u_2 Q$
5. Accept if $R'_x \equiv r \pmod{n}$

### 4.4 Why Verification Works

$$R' = u_1 G + u_2 Q = ewG + rwQ = ewG + rw(dG) = (ew + rwd)G$$

Since $w = s^{-1}$ and $s = k^{-1}(e + rd)$:

$$ew + rwd = es^{-1} + rds^{-1} = (e + rd)s^{-1} = (e + rd) \cdot \frac{k}{e + rd} = k$$

So $R' = kG = R$, and $R'_x = r$. $\checkmark$

```python
class ECDSA:
    """ECDSA implementation using the EllipticCurveFiniteField class from Lesson 6.

    WARNING: Educational implementation. Production ECDSA must use
    constant-time operations, validated curves, and RFC 6979 nonces.
    """

    def __init__(self, curve, G, n):
        """
        Args:
            curve: EllipticCurveFiniteField instance
            G: base point (generator)
            n: order of G
        """
        self.curve = curve
        self.G = G
        self.n = n

    def keygen(self):
        """Generate ECDSA key pair."""
        import random
        d = random.randrange(1, self.n)
        Q = self.curve.scalar_multiply(d, self.G)
        return d, Q

    def sign(self, message, private_key):
        """ECDSA signing.

        Why the nonce k MUST be:
        1. Random (or deterministic via RFC 6979)
        2. Unique per signature
        3. Secret (never revealed)
        Violating ANY of these leads to private key recovery (Section 5).
        """
        import random

        d = private_key
        e = int(hashlib.sha256(message).hexdigest(), 16) % self.n

        while True:
            k = random.randrange(1, self.n)
            R = self.curve.scalar_multiply(k, self.G)
            r = R[0] % self.n

            if r == 0:
                continue

            k_inv = pow(k, -1, self.n)
            s = (k_inv * (e + r * d)) % self.n

            if s == 0:
                continue

            return (r, s)

    def verify(self, message, signature, public_key):
        """ECDSA verification.

        Why we compute u1*G + u2*Q:
        - This is a linear combination of the base point and public key
        - If the signature is valid, this recovers the point R = kG
        - The x-coordinate of R must equal r from the signature
        """
        r, s = signature
        Q = public_key

        if not (1 <= r < self.n and 1 <= s < self.n):
            return False

        e = int(hashlib.sha256(message).hexdigest(), 16) % self.n

        w = pow(s, -1, self.n)
        u1 = (e * w) % self.n
        u2 = (r * w) % self.n

        # R' = u1*G + u2*Q
        P1 = self.curve.scalar_multiply(u1, self.G)
        P2 = self.curve.scalar_multiply(u2, Q)
        R_prime = self.curve.add(P1, P2)

        if R_prime is None:
            return False

        return R_prime[0] % self.n == r


# Need the EllipticCurveFiniteField class from Lesson 6
class EllipticCurveFiniteField:
    def __init__(self, a, b, p):
        self.a, self.b, self.p = a, b, p

    def is_on_curve(self, P):
        if P is None: return True
        x, y = P
        return (y*y - (x*x*x + self.a*x + self.b)) % self.p == 0

    def add(self, P, Q):
        if P is None: return Q
        if Q is None: return P
        x1, y1 = P; x2, y2 = Q; p = self.p
        if x1 == x2 and y1 == (p - y2) % p: return None
        if x1 == x2 and y1 == y2:
            if y1 == 0: return None
            lam = (3*x1*x1 + self.a) * pow(2*y1, -1, p) % p
        else:
            lam = (y2-y1) * pow(x2-x1, -1, p) % p
        x3 = (lam*lam - x1 - x2) % p
        y3 = (lam*(x1-x3) - y1) % p
        return (x3, y3)

    def scalar_multiply(self, k, P):
        result = None; addend = P
        while k:
            if k & 1: result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def count_points(self):
        count = 1
        for x in range(self.p):
            rhs = (x**3 + self.a*x + self.b) % self.p
            if rhs == 0: count += 1
            elif pow(rhs, (self.p-1)//2, self.p) == 1: count += 2
        return count


# Demo
curve = EllipticCurveFiniteField(a=2, b=3, p=97)
G = (3, 6)
n = curve.count_points()  # Group order

ecdsa = ECDSA(curve, G, n)
d, Q = ecdsa.keygen()

message = b"Hello, ECDSA!"
signature = ecdsa.sign(message, d)

print(f"Private key d = {d}")
print(f"Public key Q = {Q}")
print(f"Signature (r, s) = {signature}")
print(f"Verification: {ecdsa.verify(message, signature, Q)}")
print(f"Tampered: {ecdsa.verify(b'Hello, ECDSA?', signature, Q)}")
```

---

## 5. Nonce Reuse: The PlayStation 3 Disaster

### 5.1 The Attack

In 2010, the fail0verflow team revealed that Sony used the **same nonce $k$** for every ECDSA signature on PlayStation 3 firmware. This allowed complete recovery of Sony's private signing key.

### 5.2 The Mathematics

If two signatures $(r_1, s_1)$ and $(r_2, s_2)$ use the same nonce $k$:

Since $R = kG$ is the same, $r_1 = r_2 = r$.

$$s_1 = k^{-1}(e_1 + rd) \bmod n$$
$$s_2 = k^{-1}(e_2 + rd) \bmod n$$

Subtracting:

$$s_1 - s_2 = k^{-1}(e_1 - e_2) \bmod n$$

Solving for $k$:

$$k = (e_1 - e_2)(s_1 - s_2)^{-1} \bmod n$$

Once $k$ is known, recover $d$:

$$d = r^{-1}(s_1 k - e_1) \bmod n$$

```python
def nonce_reuse_attack_demo():
    """Demonstrate ECDSA private key recovery from nonce reuse.

    Why this is catastrophic: a SINGLE reused nonce across ANY two
    signatures reveals the private key permanently. The attacker
    then has complete control — they can sign arbitrary messages,
    forge certificates, or steal cryptocurrency.

    Real-world victims:
    - PlayStation 3 firmware signing key (2010)
    - Android Bitcoin wallet app (2013): reused k → stolen coins
    - Numerous smart contract vulnerabilities
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    # Generate target's key pair
    import random
    d_secret = random.randrange(1, n)
    Q = curve.scalar_multiply(d_secret, G)

    # Victim signs two messages with the SAME nonce (the vulnerability)
    k = random.randrange(1, n)  # Reused nonce!
    R = curve.scalar_multiply(k, G)
    r = R[0] % n

    msg1 = b"Firmware v4.0"
    msg2 = b"Firmware v4.1"
    e1 = int(hashlib.sha256(msg1).hexdigest(), 16) % n
    e2 = int(hashlib.sha256(msg2).hexdigest(), 16) % n

    k_inv = pow(k, -1, n)
    s1 = (k_inv * (e1 + r * d_secret)) % n
    s2 = (k_inv * (e2 + r * d_secret)) % n

    print(f"Two signatures with the same r (same nonce reused):")
    print(f"  Signature 1: (r={r}, s1={s1})")
    print(f"  Signature 2: (r={r}, s2={s2})")
    print(f"  (r values are identical → nonce reuse detected!)")
    print()

    # ATTACK: recover k from the two signatures
    k_recovered = ((e1 - e2) * pow(s1 - s2, -1, n)) % n
    print(f"  Recovered k = {k_recovered} (actual k = {k}, match: {k_recovered == k})")

    # ATTACK: recover private key d from k
    d_recovered = (pow(r, -1, n) * (s1 * k_recovered - e1)) % n
    print(f"  Recovered d = {d_recovered} (actual d = {d_secret}, match: {d_recovered == d_secret})")
    print()
    print("  GAME OVER: attacker now has the private signing key!")

nonce_reuse_attack_demo()
```

### 5.3 Lessons Learned

1. **Never reuse nonces.** Each signature must use a fresh, unpredictable $k$.
2. **Biased nonces are also dangerous.** Even partial knowledge of $k$ bits enables lattice attacks.
3. **Deterministic nonces** (RFC 6979) eliminate this class of vulnerability entirely — they derive $k$ deterministically from the private key and message, so the same $(d, m)$ always produces the same signature, and different messages produce different nonces.

---

## 6. EdDSA and Ed25519

### 6.1 Motivation

ECDSA's biggest weakness is the nonce: it must be random, unique, and secret. A single failure is catastrophic. **EdDSA** solves this by making the nonce **deterministic**.

### 6.2 Ed25519 Parameters

| Parameter | Value |
|-----------|-------|
| Curve | Twisted Edwards: $-x^2 + y^2 = 1 + dx^2y^2$ over $\mathbb{F}_{2^{255}-19}$ |
| Base point | $G$ (standard, fully specified) |
| Order | $n = 2^{252} + 27742317777372353535851937790883648493$ |
| Hash | SHA-512 |
| Key size | 256-bit private, 256-bit public |
| Signature size | 512 bits (64 bytes) |

### 6.3 EdDSA Signing Algorithm

1. **Key expansion:** $h = \text{SHA-512}(sk)$. Use the lower 256 bits as the scalar $a$ (with bit clamping), and the upper 256 bits as the nonce prefix $\text{prefix}$.
2. **Compute nonce:** $r = \text{SHA-512}(\text{prefix} \| m) \bmod n$
3. **Compute $R = rG$**
4. **Compute $S = (r + \text{SHA-512}(R \| pk \| m) \cdot a) \bmod n$**
5. **Signature:** $(R, S)$

```python
def ed25519_concept():
    """Explain Ed25519's deterministic nonce generation.

    Why deterministic nonces are revolutionary:
    1. No random number generator needed during signing
    2. Nonce reuse is IMPOSSIBLE (same message → same nonce → same signature)
    3. Different messages → different nonces (SHA-512 is collision-resistant)
    4. The nonce depends on the private key, so an attacker without the
       private key cannot predict or influence the nonce
    5. Eliminates the ENTIRE class of nonce-related vulnerabilities

    Why Ed25519 over ECDSA:
    - Deterministic: no nonce disaster possible
    - Faster: ~3x faster signing, ~2x faster verification than ECDSA P-256
    - Smaller: 64-byte signatures (vs 70+ for ECDSA DER-encoded)
    - Simpler: fewer parameters, fewer implementation pitfalls
    - Constant-time: Edwards curve addition has no branches
    """
    print("Ed25519 Signing Process:")
    print("=" * 50)
    print()
    print("1. Key expansion: h = SHA-512(private_key)")
    print("   a = lower_256_bits(h)  (scalar, with bit clamping)")
    print("   prefix = upper_256_bits(h)  (nonce seed)")
    print()
    print("2. Nonce: r = SHA-512(prefix || message) mod n")
    print("   → Deterministic! Same (key, message) → same nonce")
    print("   → Different messages → different nonces")
    print("   → No RNG needed during signing")
    print()
    print("3. R = r * G  (nonce point)")
    print()
    print("4. S = (r + SHA-512(R || pk || message) * a) mod n")
    print()
    print("5. Signature = (R, S)  [64 bytes total]")
    print()
    print("Verification: check that S*G = R + SHA-512(R || pk || m) * Q")
    print()
    print("Adoption:")
    print("  - SSH (default key type since OpenSSH 6.5)")
    print("  - Signal Protocol")
    print("  - WireGuard")
    print("  - Tor")
    print("  - Solana, Cardano, Polkadot blockchains")

ed25519_concept()
```

### 6.4 Ed25519 Verification

To verify signature $(R, S)$ on message $m$ with public key $Q$:

$$S \cdot G \stackrel{?}{=} R + H(R \| Q \| m) \cdot Q$$

This is equivalent to checking:

$$(r + H(R \| Q \| m) \cdot a) \cdot G = r \cdot G + H(R \| Q \| m) \cdot a \cdot G = R + H(R \| Q \| m) \cdot Q$$

### 6.5 Bit Clamping

Ed25519 "clamps" the private scalar $a$:
- Clear the lowest 3 bits (ensures $a$ is a multiple of 8, the cofactor)
- Clear the highest bit
- Set the second-highest bit

```python
def bit_clamping_explanation():
    """Explain why Ed25519 clamps the private scalar.

    Why clamp:
    1. Clearing low 3 bits (cofactor 8): prevents small-subgroup attacks.
       The curve has cofactor h=8, meaning the full group has small
       subgroups of order 2, 4, 8. Clamping ensures the scalar is a
       multiple of 8, so multiplication always lands in the prime-order
       subgroup.

    2. Setting bit 255: ensures constant-time scalar multiplication.
       The Montgomery ladder needs a fixed number of iterations, which
       is determined by the position of the highest set bit.

    3. Clearing bit 256: ensures the scalar is less than 2^255,
       preventing modular reduction issues.
    """
    example_key = bytearray(range(32))  # Toy example

    # Clamp
    example_key[0] &= 248   # Clear lowest 3 bits (248 = 11111000)
    example_key[31] &= 127  # Clear highest bit    (127 = 01111111)
    example_key[31] |= 64   # Set second-highest   (64  = 01000000)

    print(f"Before clamping: byte[0]  = xxxx_xxxx")
    print(f"After clamping:  byte[0]  = xxxx_x000  (multiple of 8)")
    print(f"Before clamping: byte[31] = xxxx_xxxx")
    print(f"After clamping:  byte[31] = 01xx_xxxx  (bit 254 set, bit 255 clear)")

bit_clamping_explanation()
```

---

## 7. Schnorr Signatures

### 7.1 Overview

Schnorr signatures (1989) are the simplest discrete-log-based signature scheme. They were historically underused due to patent restrictions (expired 2008) but have recently gained prominence in Bitcoin (BIP 340, Taproot upgrade, 2021).

### 7.2 Algorithm

**Parameters:** Group $\mathbb{Z}_p^*$ with generator $g$ and prime order $q$, or an elliptic curve group.

**Sign:**
1. Choose random nonce $k$
2. $R = g^k$ (or $R = kG$ for ECC)
3. $e = H(R \| m)$
4. $s = k - ed \bmod q$
5. Signature: $(e, s)$ or $(R, s)$

**Verify:**
1. $R' = g^s \cdot y^e$ where $y = g^d$ is the public key
2. Check $e \stackrel{?}{=} H(R' \| m)$

```python
def schnorr_signature_demo():
    """Demonstrate Schnorr signatures on an elliptic curve.

    Why Schnorr signatures matter:
    1. Simplest DL-based signature — minimal code, minimal attack surface
    2. Provably secure under the DL assumption (in the ROM)
    3. Linear structure enables advanced features:
       - Multi-signatures (MuSig): n signers, 1 combined signature
       - Threshold signatures: t-of-n signing
       - Adaptor signatures: conditional signatures for atomic swaps
    4. Adopted by Bitcoin (BIP 340) for Taproot in 2021
    """
    curve = EllipticCurveFiniteField(a=2, b=3, p=97)
    G = (3, 6)
    n = curve.count_points()

    import random

    # Key generation
    d = random.randrange(1, n)
    Q = curve.scalar_multiply(d, G)

    # Signing
    message = b"Hello, Schnorr!"
    k = random.randrange(1, n)
    R = curve.scalar_multiply(k, G)

    # e = H(R || message) mod n
    e_input = f"{R[0]},{R[1]},{message.hex()}".encode()
    e = int(hashlib.sha256(e_input).hexdigest(), 16) % n

    s = (k - e * d) % n

    print(f"Schnorr Signature:")
    print(f"  Private key d = {d}")
    print(f"  Public key Q = {Q}")
    print(f"  Nonce point R = {R}")
    print(f"  e = {e}, s = {s}")

    # Verification: R' = s*G + e*Q should equal R
    sG = curve.scalar_multiply(s, G)
    eQ = curve.scalar_multiply(e, Q)
    R_prime = curve.add(sG, eQ)

    # Recompute e' = H(R' || message)
    e_prime_input = f"{R_prime[0]},{R_prime[1]},{message.hex()}".encode()
    e_prime = int(hashlib.sha256(e_prime_input).hexdigest(), 16) % n

    print(f"  Verification: R' = {R_prime}")
    print(f"  R' == R: {R_prime == R}")
    print(f"  e' == e: {e_prime == e}")
    print(f"  Signature valid: {e_prime == e}")

schnorr_signature_demo()
```

### 7.3 Schnorr vs ECDSA

| Property | ECDSA | Schnorr |
|----------|-------|---------|
| Security proof | Heuristic | Provable (in ROM) |
| Linearity | No | Yes (enables multi-sigs) |
| Signature size | ~72 bytes (DER) | 64 bytes |
| Batch verification | Possible but complex | Natural and efficient |
| Multi-signatures | Complex (MuSig2) | Elegant |
| Patent status | Always free | Patented until 2008 |
| Adoption | Widespread | Growing (Bitcoin Taproot) |

---

## 8. Advanced Signature Schemes

### 8.1 Blind Signatures

A **blind signature** allows a user to have a message signed without the signer seeing the message content. Introduced by David Chaum (1982) for anonymous digital cash.

```python
def blind_signature_concept():
    """Explain blind signatures and their applications.

    How it works (RSA-based):
    1. Alice has message m
    2. Alice blinds: m' = m * r^e mod n  (r is random blinding factor)
    3. Signer signs: s' = (m')^d = m^d * r mod n
    4. Alice unblinds: s = s' * r^(-1) = m^d mod n
    5. Result: valid signature on m, but signer never saw m!

    Applications:
    - Anonymous digital cash (eCash, David Chaum)
    - Voting systems (vote without revealing to authority)
    - Certificate issuance (prove attribute without revealing identity)
    """
    print("Blind Signature Protocol (RSA-based):")
    print("=" * 50)
    print()
    print("Alice                         Signer")
    print("  |                             |")
    print("  | m' = m * r^e mod n          |")
    print("  |------------ m' ------------>|")
    print("  |                             | s' = (m')^d mod n")
    print("  |<----------- s' -------------|")
    print("  | s = s' * r^(-1) mod n       |")
    print("  |                             |")
    print("  | s is valid signature on m   |")
    print("  | Signer never saw m!         |")

blind_signature_concept()
```

### 8.2 Aggregate Signatures

**Aggregate signatures** combine $n$ signatures from $n$ signers on $n$ messages into a single signature that can be verified against all $n$ public keys.

| Scheme | Aggregate Size | Verification |
|--------|---------------|--------------|
| Individual | $n \times 64$ bytes | $n$ verifications |
| BLS aggregate | 48 bytes (constant!) | 1 pairing check |
| Schnorr multi-sig | 64 bytes (for same message) | 1 verification |

**Applications:**
- Blockchain: compress thousands of transaction signatures into one
- Certificate Transparency: aggregate validator signatures
- IoT: resource-constrained devices send compact proofs

### 8.3 Threshold Signatures

A $(t, n)$ **threshold signature** scheme allows any $t$ out of $n$ parties to collaboratively produce a valid signature, but $t - 1$ or fewer parties cannot.

**Key property:** The resulting signature is indistinguishable from a single-party signature — the verifier does not know (or need to know) that multiple parties participated.

**Applications:**
- Multi-party custody of cryptocurrency
- Distributed certificate authorities
- Corporate signing policies (e.g., 3-of-5 executives must approve)

---

## 9. Security Models

### 9.1 EUF-CMA: Existential Unforgeability under Chosen Message Attack

The standard security definition for digital signatures:

**Game:**
1. Challenger generates $(pk, sk)$ and gives $pk$ to the adversary
2. Adversary can request signatures on any messages of their choice (oracle access)
3. Adversary wins if they produce a valid signature $(m^*, \sigma^*)$ where $m^*$ was never queried

A scheme is **EUF-CMA secure** if no efficient adversary can win this game with non-negligible probability.

### 9.2 SUF-CMA: Strong Unforgeability

A stronger notion: the adversary cannot produce a **new** valid signature even on a previously signed message. This means for any message $m$ that was queried, the adversary cannot produce a different valid signature $\sigma' \neq \sigma$.

### 9.3 Comparison

| Scheme | EUF-CMA | SUF-CMA | Model |
|--------|---------|---------|-------|
| RSA-PKCS#1 v1.5 | Yes (assumed) | No | Heuristic |
| RSA-PSS | Yes | Yes | Random Oracle |
| ECDSA | Yes (assumed) | No | Generic Group |
| EdDSA | Yes | Yes (with checks) | Random Oracle |
| Schnorr | Yes | Yes | Random Oracle |

```python
def security_model_demo():
    """Demonstrate the EUF-CMA game conceptually.

    Why EUF-CMA is the minimum standard:
    - The attacker gets to see many valid signatures (realistic —
      public servers sign millions of messages)
    - Despite all these examples, the attacker still cannot forge
      a signature on any NEW message
    - This models a very powerful attacker; if a scheme is EUF-CMA
      secure, it resists all weaker attacks too
    """
    print("EUF-CMA Security Game:")
    print("=" * 50)
    print()
    print("Setup: Challenger generates (pk, sk), gives pk to Adversary")
    print()
    print("Phase 1 (Learning):")
    print("  Adversary → Challenger: 'Sign m_1'")
    print("  Challenger → Adversary: σ_1 = Sign(sk, m_1)")
    print("  Adversary → Challenger: 'Sign m_2'")
    print("  Challenger → Adversary: σ_2 = Sign(sk, m_2)")
    print("  ... (polynomially many queries)")
    print()
    print("Phase 2 (Forgery):")
    print("  Adversary outputs (m*, σ*)")
    print("  Adversary WINS if:")
    print("    1. Verify(pk, m*, σ*) = accept")
    print("    2. m* was NOT queried in Phase 1")
    print()
    print("A scheme is EUF-CMA secure if no efficient")
    print("adversary can win with non-negligible probability.")

security_model_demo()
```

---

## 10. Exercises

### Exercise 1: RSA Signatures (Basic)

Using RSA with $p = 61$, $q = 53$, $e = 17$:
1. Compute the private key $d$
2. Sign the message hash $h = 42$ (as a number)
3. Verify the signature using the public key
4. Show that modifying the hash to $h = 43$ causes verification to fail

### Exercise 2: ECDSA Implementation (Intermediate)

1. Using the `ECDSA` class from Section 4, sign 10 different messages and verify each
2. Measure the average number of point operations per signing and verification
3. What is the probability that the first nonce $k$ produces $r = 0$? Why is the retry loop necessary?

### Exercise 3: Nonce Reuse Attack (Intermediate)

1. Implement the nonce reuse attack from Section 5
2. Use a curve with order > 100
3. Create two signatures with the same (secretly chosen) nonce
4. Recover the private key from the two signatures
5. Verify that the recovered key can sign new messages correctly

### Exercise 4: Schnorr Multi-Signature (Challenging)

Implement a simple 2-of-2 Schnorr multi-signature protocol:
1. Alice and Bob each have private keys $d_A$ and $d_B$
2. Joint public key: $Q = Q_A + Q_B$
3. To sign: each party contributes a partial signature
4. The combined signature verifies against $Q$
5. Discuss: why is naive key aggregation ($Q = Q_A + Q_B$) vulnerable to a rogue-key attack? How does MuSig2 prevent this?

### Exercise 5: Deterministic vs Random Nonces (Challenging)

1. Implement RFC 6979 deterministic nonce generation for ECDSA
2. Verify that the same (key, message) always produces the same signature
3. Verify that different messages produce different nonces
4. Compare with Ed25519's approach: how are they similar? How do they differ?
5. Discuss: can a deterministic signature scheme ever be SUF-CMA secure? Why or why not?

---

**Previous**: [Elliptic Curve Cryptography](./06_Elliptic_Curve_Cryptography.md) | **Next**: [Key Exchange](./08_Key_Exchange.md)

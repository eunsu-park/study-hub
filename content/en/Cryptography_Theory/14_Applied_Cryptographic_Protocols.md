# Lesson 14: Applied Cryptographic Protocols

**Previous**: [Homomorphic Encryption](./13_Homomorphic_Encryption.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Trace a complete TLS 1.3 handshake and explain each cryptographic operation
2. Describe the Signal protocol's Double Ratchet algorithm and its security properties
3. Explain secure multi-party computation (MPC) and implement Yao's garbled circuits conceptually
4. Implement Shamir's Secret Sharing and understand its threshold properties
5. Design commitment schemes and explain their role in cryptographic protocols
6. Describe oblivious transfer and its foundational role in MPC
7. Analyze how real-world protocols compose the primitives from earlier lessons

---

Throughout this course, we have studied individual cryptographic primitives — symmetric ciphers (Lesson 3), hash functions (Lesson 4), public-key encryption (Lesson 5), digital signatures (Lesson 7), key exchange (Lesson 8), and advanced constructions like ZKPs (Lesson 12) and homomorphic encryption (Lesson 13). In practice, none of these primitives operates in isolation. Real-world security is achieved by **composing** primitives into protocols — carefully orchestrated sequences of operations where the output of one primitive feeds into the input of another. A single design flaw in this composition can render the entire system insecure, regardless of how strong the individual primitives are.

This lesson examines the most important cryptographic protocols in use today and shows how they weave together the primitives we have studied.

> **Analogy:** The Signal protocol is like passing notes in class through relays. Each note is encrypted differently, and even if someone intercepts one, they cannot read past or future notes. This is forward secrecy and future secrecy (also called post-compromise security) combined — the cryptographic equivalent of a self-destructing message chain.

## Table of Contents

1. [TLS 1.3: The Internet's Security Layer](#1-tls-13-the-internets-security-layer)
2. [The Signal Protocol](#2-the-signal-protocol)
3. [Secure Multi-Party Computation (MPC)](#3-secure-multi-party-computation-mpc)
4. [Shamir's Secret Sharing](#4-shamirs-secret-sharing)
5. [Commitment Schemes](#5-commitment-schemes)
6. [Oblivious Transfer](#6-oblivious-transfer)
7. [Threshold Signatures](#7-threshold-signatures)
8. [Private Information Retrieval](#8-private-information-retrieval)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)

---

## 1. TLS 1.3: The Internet's Security Layer

### 1.1 What Is TLS?

Transport Layer Security (TLS) protects virtually all internet communication: HTTPS, email (SMTP/IMAP with STARTTLS), DNS-over-TLS, and more. TLS 1.3 (RFC 8446, published 2018) is a major redesign that removed legacy baggage and improved both security and performance.

### 1.2 TLS 1.3 vs. TLS 1.2

| Property | TLS 1.2 | TLS 1.3 |
|----------|---------|---------|
| Round trips to first data | 2 RTT | **1 RTT** (0-RTT possible) |
| Key exchange | RSA or DHE/ECDHE | **ECDHE only** (forward secrecy mandatory) |
| Cipher suites | 37+ (many insecure) | **5** (all authenticated encryption) |
| Compression | Supported | **Removed** (CRIME attack) |
| Renegotiation | Supported | **Removed** (security issues) |
| 0-RTT | Not available | **Optional** (with replay risk) |

### 1.3 Full Handshake (1-RTT)

```
Client                                              Server

ClientHello
  + key_share (ECDHE public key)
  + supported_versions (TLS 1.3)
  + cipher_suites
  + signature_algorithms
  --------------------------------------------->

                                              ServerHello
                                  + key_share (ECDHE public key)
                           {EncryptedExtensions}
                           {CertificateRequest*}
                           {Certificate}
                           {CertificateVerify}
                           {Finished}
  <---------------------------------------------

{Certificate*}
{CertificateVerify*}
{Finished}
  --------------------------------------------->

[Application Data]            <---->         [Application Data]
```

Items in `{...}` are encrypted with handshake keys derived from the ECDHE shared secret.

### 1.4 Key Derivation in TLS 1.3

TLS 1.3 uses a sophisticated key schedule based on HKDF (Lesson 8):

```
                  0
                  |
                  v
    PSK ->  HKDF-Extract = Early Secret
                  |
                  +-> Derive-Secret(., "ext binder" | "res binder", "")
                  |                     = binder_key
                  +-> Derive-Secret(., "c e traffic", ClientHello)
                  |                     = client_early_traffic_secret
                  +-> Derive-Secret(., "e exp master", ClientHello)
                  |                     = early_exporter_master_secret
                  v
            Derive-Secret(., "derived", "")
                  |
                  v
  (EC)DHE -> HKDF-Extract = Handshake Secret
                  |
                  +-> Derive-Secret(., "c hs traffic", ClientHello..ServerHello)
                  |                     = client_handshake_traffic_secret
                  +-> Derive-Secret(., "s hs traffic", ClientHello..ServerHello)
                  |                     = server_handshake_traffic_secret
                  v
            Derive-Secret(., "derived", "")
                  |
                  v
       0 -> HKDF-Extract = Master Secret
                  |
                  +-> Derive-Secret(., "c ap traffic", ClientHello..Server Finished)
                  |                     = client_application_traffic_secret
                  +-> Derive-Secret(., "s ap traffic", ClientHello..Server Finished)
                                        = server_application_traffic_secret
```

### 1.5 Why This Design?

- **Separate keys for each direction**: Client-to-server and server-to-client use different keys, preventing reflection attacks
- **Handshake vs. application keys**: Separating these limits damage from key exposure
- **Transcript binding**: Each derived key includes a hash of all previous messages, preventing tampering or message reordering
- **PSK support**: Pre-shared keys allow 0-RTT resumption (at the cost of replay vulnerability)

### 1.6 0-RTT Resumption

In 0-RTT mode, the client sends application data in the first message (before the server responds):

```
Client                                              Server
ClientHello + early_data (encrypted with PSK)
  --------------------------------------------->
```

**Risk**: 0-RTT data can be **replayed** by an attacker who captures the ClientHello. TLS 1.3 specifies that servers must handle 0-RTT data as potentially replayed — it should only be used for idempotent operations (e.g., GET requests, not POST).

---

## 2. The Signal Protocol

### 2.1 Overview

The Signal protocol (used by Signal, WhatsApp, Facebook Messenger, Google Messages) provides:
- **End-to-end encryption**: Only sender and receiver can read messages
- **Forward secrecy**: Compromising current keys doesn't reveal past messages
- **Post-compromise security** (future secrecy): After a compromise, security is restored once new keys are exchanged
- **Asynchronous key exchange**: Works when the recipient is offline (via X3DH, Lesson 8)

### 2.2 The Double Ratchet Algorithm

After X3DH (Lesson 8) establishes the initial shared secret, the Double Ratchet maintains ongoing message security through two interlocking "ratchets":

**1. DH Ratchet** (asymmetric):
- Each party generates a new ephemeral DH key pair with every message round-trip
- A new DH shared secret is computed, providing forward secrecy at the message level

**2. Symmetric Ratchet** (KDF chain):
- Between DH ratchet steps, a hash chain derives per-message keys
- Each step: `(chain_key, message_key) = KDF(chain_key_prev)`
- Message keys are used once and deleted

```
Alice sends:                    Bob receives:
  DH_A1 →  msg1 (chain_key_A1) ← DH_B0
  DH_A1 →  msg2 (chain_key_A1)
                                Bob sends:
                                  DH_B1 → msg3 (chain_key_B1) ← DH_A1
  Alice receives:
  DH_A1 ← msg3 (chain_key_B1) ← DH_B1

  Alice sends (new DH ratchet step):
  DH_A2 →  msg4 (chain_key_A2) ← DH_B1
```

### 2.3 Why Two Ratchets?

| Ratchet | Purpose | Security Property |
|---------|---------|-------------------|
| DH ratchet | Generate new shared secrets | Forward secrecy + post-compromise security |
| Symmetric ratchet | Derive per-message keys efficiently | Key separation between messages |

The DH ratchet is expensive (requires a key exchange) but provides the strongest security. The symmetric ratchet is cheap (just a hash) and fills the gaps between DH steps.

### 2.4 Out-of-Order Messages

The symmetric ratchet allows handling out-of-order messages: if message 5 arrives before message 4, the receiver can store the chain key at position 4 and compute the key for message 5. When message 4 eventually arrives, the stored chain key is used to derive its key.

---

## 3. Secure Multi-Party Computation (MPC)

### 3.1 The Millionaires' Problem

Yao's Millionaires' Problem (1982): Two millionaires want to know who is richer without revealing their actual wealth to each other. More generally: $n$ parties, each holding a private input $x_i$, want to compute $f(x_1, \ldots, x_n)$ without any party learning anything beyond the output.

### 3.2 Security Models

| Model | Corruption | Guarantee |
|-------|-----------|-----------|
| **Semi-honest** (honest-but-curious) | Follows protocol but tries to learn from transcripts | Privacy against passive adversary |
| **Malicious** | Can deviate arbitrarily from the protocol | Privacy + correctness against active adversary |
| **Covert** | Deviates if not caught | Cheating is detected with probability $\epsilon$ |

### 3.3 Yao's Garbled Circuits

Yao's protocol (1986) for two-party computation:

1. **Circuit representation**: Express $f$ as a Boolean circuit
2. **Garbling**: One party ("garbler") replaces each wire value (0 or 1) with random labels, and creates an encrypted truth table for each gate
3. **Evaluation**: The other party ("evaluator") obtains labels for their input (via oblivious transfer, Section 6) and evaluates the garbled circuit gate by gate
4. **Result**: The evaluator obtains the output label, which the garbler maps back to 0 or 1

```python
"""
Simplified Garbled Circuit demonstration.

This is a conceptual implementation showing the core idea.
A real garbled circuit system uses point-and-permute optimization,
free-XOR technique, and half-gates for efficiency.

Why garbled circuits? They allow two parties to compute ANY function
on their joint inputs while revealing only the output. This is the
most general form of two-party computation.
"""

import secrets
import hashlib
from typing import Dict, Tuple


def garble_and_gate(
    wire_a_labels: Tuple[bytes, bytes],  # (label_0, label_1) for wire A
    wire_b_labels: Tuple[bytes, bytes],  # (label_0, label_1) for wire B
    wire_c_labels: Tuple[bytes, bytes],  # (label_0, label_1) for output wire C
) -> list[bytes]:
    """
    Create a garbled AND gate.

    The truth table for AND: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1

    Each row encrypts the output label under the two input labels.
    Why double encryption? The evaluator needs both input labels to
    decrypt the correct output label. Having only one label reveals nothing.
    """
    garbled_table = []

    and_truth = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]

    for a_val, b_val, c_val in and_truth:
        # Double-encrypt the output label
        key_a = wire_a_labels[a_val]
        key_b = wire_b_labels[b_val]
        output_label = wire_c_labels[c_val]

        # Encrypt: H(key_a || key_b) XOR output_label
        # Why hash-based encryption? It's a simple PRF construction.
        # Real implementations use AES for efficiency.
        h = hashlib.sha256(key_a + key_b).digest()[:len(output_label)]
        encrypted = bytes(a ^ b for a, b in zip(h, output_label))
        garbled_table.append(encrypted)

    # Randomly permute the table rows
    # Why shuffle? Without shuffling, the position in the table
    # leaks information about the input values.
    indices = list(range(4))
    for i in range(3, 0, -1):
        j = secrets.randbelow(i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    return [garbled_table[i] for i in indices]


def evaluate_garbled_gate(
    garbled_table: list[bytes],
    label_a: bytes,
    label_b: bytes,
    label_length: int = 16,
) -> bytes:
    """
    Evaluate a garbled gate given input labels.

    The evaluator tries all rows (only one will decrypt correctly).
    Why try all rows? The table is shuffled, so the evaluator
    doesn't know which row corresponds to their inputs.
    """
    for encrypted_row in garbled_table:
        h = hashlib.sha256(label_a + label_b).digest()[:label_length]
        decrypted = bytes(a ^ b for a, b in zip(h, encrypted_row))
        # In a real implementation, we'd use point-and-permute to
        # avoid trying all rows (only try the one indicated by
        # permutation bits appended to labels)
        if _is_valid_label(decrypted):
            return decrypted

    raise ValueError("No valid decryption found")


def _is_valid_label(label: bytes) -> bool:
    """
    Check if a decrypted label is valid.

    In this simplified version, we use a simple heuristic.
    Real implementations use point-and-permute with explicit
    validity markers.
    """
    # For demonstration: labels have a known prefix
    return label[:2] == b'\xAB\xCD'


def demo_garbled_circuit():
    """Demonstrate a garbled AND gate."""
    label_length = 16

    def make_label():
        return b'\xAB\xCD' + secrets.token_bytes(label_length - 2)

    # Generate random labels for each wire value
    wire_a = (make_label(), make_label())  # (label_0, label_1)
    wire_b = (make_label(), make_label())
    wire_c = (make_label(), make_label())

    # Garbler creates the garbled table
    garbled_table = garble_and_gate(wire_a, wire_b, wire_c)

    # Test all input combinations
    print("=== Garbled AND Gate ===")
    for a_val in [0, 1]:
        for b_val in [0, 1]:
            result_label = evaluate_garbled_gate(
                garbled_table, wire_a[a_val], wire_b[b_val], label_length
            )
            # Map output label back to bit
            result_bit = 0 if result_label == wire_c[0] else 1
            expected = a_val & b_val
            print(f"  AND({a_val}, {b_val}) = {result_bit} "
                  f"(expected: {expected}, "
                  f"{'CORRECT' if result_bit == expected else 'WRONG'})")


if __name__ == "__main__":
    demo_garbled_circuit()
```

---

## 4. Shamir's Secret Sharing

### 4.1 The Problem

A secret $S$ needs to be split among $n$ parties such that:
- Any $t$ (threshold) or more parties can reconstruct $S$
- Fewer than $t$ parties learn **absolutely nothing** about $S$ (information-theoretic security)

### 4.2 The Scheme

Shamir's Secret Sharing (1979) is based on polynomial interpolation:

1. **Sharing**: Choose a random polynomial $f(x)$ of degree $t-1$ with $f(0) = S$:
   $$f(x) = S + a_1 x + a_2 x^2 + \cdots + a_{t-1} x^{t-1}$$
2. **Distribute**: Give party $i$ the share $(i, f(i))$
3. **Reconstruction**: Any $t$ shares determine the polynomial uniquely (via Lagrange interpolation), recovering $S = f(0)$

### 4.3 Why $t-1$ Points Are Insufficient

A polynomial of degree $t-1$ passes through $t-1$ points in infinitely many ways. Each possible polynomial corresponds to a different secret value, all equally likely. This is **perfect** secrecy — even with $t-1$ shares, the secret could be any value with equal probability.

### 4.4 Implementation

```python
"""
Shamir's Secret Sharing Scheme.

Why work in a finite field (mod p)? In the integers, interpolation
produces fractions. In a prime field Z_p, division is always possible
(every non-zero element has a multiplicative inverse), and the
information-theoretic security proof holds exactly.
"""

import secrets
from typing import List, Tuple


class ShamirSecretSharing:
    """Shamir's (t, n) threshold secret sharing over Z_p."""

    def __init__(self, prime: int = None):
        """
        Initialize with a prime modulus.

        Why a large prime? The prime must be larger than the secret
        and the number of shares. A 256-bit prime works for most
        applications (can share any 256-bit secret).
        """
        if prime is None:
            # Mersenne prime 2^127 - 1 (good for demonstration)
            self.p = (1 << 127) - 1
        else:
            self.p = prime

    def split(self, secret: int, threshold: int,
              num_shares: int) -> List[Tuple[int, int]]:
        """
        Split a secret into shares.

        Parameters:
            secret: The secret value (must be < self.p)
            threshold: Minimum shares needed to reconstruct (t)
            num_shares: Total number of shares to generate (n)

        Returns:
            List of (x, y) shares
        """
        assert 0 <= secret < self.p, f"Secret must be in [0, {self.p})"
        assert threshold <= num_shares, "Threshold cannot exceed num_shares"
        assert threshold >= 2, "Threshold must be at least 2"

        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1}
        coefficients = [secret]  # a0 = secret
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(self.p))

        # Evaluate polynomial at x = 1, 2, ..., n
        shares = []
        for i in range(1, num_shares + 1):
            y = self._evaluate_polynomial(coefficients, i)
            shares.append((i, y))

        return shares

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct the secret from shares using Lagrange interpolation.

        Why Lagrange interpolation? Given t points on a degree-(t-1)
        polynomial, Lagrange interpolation recovers the unique
        polynomial passing through all points. The secret is f(0).
        """
        # Lagrange interpolation at x = 0
        secret = 0

        for i, (xi, yi) in enumerate(shares):
            # Compute Lagrange basis polynomial L_i(0)
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(shares):
                if i != j:
                    # L_i(0) = product of (0 - xj) / (xi - xj)
                    numerator = (numerator * (-xj)) % self.p
                    denominator = (denominator * (xi - xj)) % self.p

            # Why modular inverse? In Z_p, division is multiplication
            # by the modular inverse. This exists for all non-zero
            # elements because p is prime (Fermat's little theorem).
            lagrange_coeff = (numerator * pow(denominator, -1, self.p)) % self.p
            secret = (secret + yi * lagrange_coeff) % self.p

        return secret

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at x using Horner's method."""
        # Horner's method: f(x) = (...((a_{t-1}*x + a_{t-2})*x + a_{t-3})*x + ... + a_0)
        # Why Horner's? It evaluates a degree-d polynomial with only d
        # multiplications instead of d*(d+1)/2 with naive evaluation.
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.p
        return result


def demo_secret_sharing():
    """Demonstrate Shamir's Secret Sharing."""
    sss = ShamirSecretSharing()

    secret = 42
    threshold = 3  # Need at least 3 shares to reconstruct
    num_shares = 5  # Distribute 5 shares total

    print(f"Secret: {secret}")
    print(f"Threshold: {threshold}-of-{num_shares}\n")

    # Split
    shares = sss.split(secret, threshold, num_shares)
    for i, (x, y) in enumerate(shares):
        print(f"  Share {i+1}: ({x}, {y % 1000}...)  [truncated]")

    # Reconstruct with exactly threshold shares
    print(f"\n--- Reconstruct with {threshold} shares ---")
    recovered = sss.reconstruct(shares[:threshold])
    print(f"  Recovered secret: {recovered}")
    print(f"  Correct: {recovered == secret}")

    # Reconstruct with more than threshold shares
    print(f"\n--- Reconstruct with {threshold + 1} shares ---")
    recovered = sss.reconstruct(shares[:threshold + 1])
    print(f"  Recovered secret: {recovered}")
    print(f"  Correct: {recovered == secret}")

    # Attempt with fewer than threshold shares
    print(f"\n--- Reconstruct with {threshold - 1} shares ---")
    wrong = sss.reconstruct(shares[:threshold - 1])
    print(f"  Result: {wrong}")
    print(f"  Correct: {wrong == secret}")
    print(f"  (With fewer than t shares, the result is random — no information leaked)")

    # Different subsets of t shares all work
    print(f"\n--- Different subsets of {threshold} shares ---")
    import itertools
    for combo in list(itertools.combinations(shares, threshold))[:3]:
        recovered = sss.reconstruct(list(combo))
        indices = [s[0] for s in combo]
        print(f"  Shares {indices}: recovered = {recovered}, correct = {recovered == secret}")


if __name__ == "__main__":
    demo_secret_sharing()
```

### 4.5 Applications

| Application | Threshold | Use Case |
|------------|-----------|----------|
| Corporate key escrow | 3-of-5 board members | Recover keys if CEO incapacitated |
| Cryptocurrency wallets | 2-of-3 devices | Prevent single-device compromise |
| Nuclear launch codes | 2-of-2 officers | Dual-person integrity |
| Distributed signing | t-of-n validators | Threshold signatures (Section 7) |

---

## 5. Commitment Schemes

### 5.1 Definition

A **commitment scheme** allows a party to commit to a value while keeping it hidden, and later reveal (open) the commitment.

Two properties:
- **Hiding**: The commitment reveals nothing about the value
- **Binding**: The committer cannot change the value after committing

### 5.2 Hash-Based Commitment

The simplest construction:

$$
\text{Commit}(m, r) = H(m \| r) \quad \text{where } r \text{ is random}
$$

- **Hiding**: $H$ hides $m$ (random oracle model)
- **Binding**: Collision resistance prevents finding $m' \neq m$ with the same commitment

### 5.3 Pedersen Commitment

A commitment with additional algebraic properties:

$$
C = g^m h^r \pmod{p}
$$

where $g, h$ are generators with unknown discrete log relation.

**Key property**: Pedersen commitments are **homomorphic**:

$$
C_1 \cdot C_2 = g^{m_1 + m_2} h^{r_1 + r_2} = \text{Commit}(m_1 + m_2, r_1 + r_2)
$$

This enables committed addition — proving statements about sums of committed values without revealing them.

```python
"""
Commitment scheme implementations.

Why commitment schemes? They are building blocks for:
- Zero-knowledge proofs (Lesson 12: graph coloring ZKP uses commitments)
- Auction protocols (bid without revealing, then reveal after deadline)
- Coin-flipping over the phone (fair randomness without trust)
- MPC protocols (commit to inputs before seeing others' inputs)
"""

import secrets
import hashlib


class HashCommitment:
    """Hash-based commitment scheme."""

    @staticmethod
    def commit(message: bytes) -> tuple[bytes, bytes]:
        """
        Commit to a message.

        Returns (commitment, opening_info).
        The opening_info includes the randomness needed to verify.
        """
        randomness = secrets.token_bytes(32)
        commitment = hashlib.sha256(message + randomness).digest()
        return commitment, randomness

    @staticmethod
    def verify(commitment: bytes, message: bytes,
               randomness: bytes) -> bool:
        """Verify that a commitment opens to the claimed message."""
        expected = hashlib.sha256(message + randomness).digest()
        # Why constant-time comparison? Prevents timing attacks that
        # could leak information about the commitment value.
        return secrets.compare_digest(commitment, expected)


class PedersenCommitment:
    """
    Pedersen commitment over Z_p*.

    Properties:
    - Perfectly hiding (information-theoretic)
    - Computationally binding (under DLP assumption)
    - Homomorphic: Commit(a) * Commit(b) = Commit(a+b)
    """

    def __init__(self, p: int, g: int, h: int):
        """
        Initialize with group parameters.

        CRITICAL: The discrete log of h with respect to g must be
        unknown. If someone knows log_g(h), they can break binding
        (open a commitment to any value).
        """
        self.p = p
        self.g = g
        self.h = h

    def commit(self, value: int) -> tuple[int, int]:
        """Commit to a value. Returns (commitment, randomness)."""
        r = secrets.randbelow(self.p - 1) + 1
        c = (pow(self.g, value, self.p) * pow(self.h, r, self.p)) % self.p
        return c, r

    def verify(self, commitment: int, value: int, randomness: int) -> bool:
        """Verify a Pedersen commitment opening."""
        expected = (pow(self.g, value, self.p) *
                    pow(self.h, randomness, self.p)) % self.p
        return commitment == expected

    def add_commitments(self, c1: int, c2: int) -> int:
        """
        Homomorphically add two commitments.

        If c1 = Commit(v1, r1) and c2 = Commit(v2, r2),
        then c1 * c2 = Commit(v1 + v2, r1 + r2).
        """
        return (c1 * c2) % self.p


def demo_commitments():
    """Demonstrate commitment schemes."""

    # Hash commitment
    print("=== Hash Commitment ===")
    msg = b"My secret bid: $1000"
    comm, randomness = HashCommitment.commit(msg)
    print(f"Commitment: {comm.hex()[:16]}...")
    print(f"Verify (correct):  {HashCommitment.verify(comm, msg, randomness)}")
    print(f"Verify (tampered): {HashCommitment.verify(comm, b'$2000', randomness)}")

    # Pedersen commitment (small prime for demo)
    print("\n=== Pedersen Commitment ===")
    p = 23
    g = 4   # Generator
    h = 9   # Another generator (unknown DLP relation to g)

    pc = PedersenCommitment(p, g, h)

    v1, v2 = 5, 7
    c1, r1 = pc.commit(v1)
    c2, r2 = pc.commit(v2)

    print(f"Commit({v1}) = {c1}")
    print(f"Commit({v2}) = {c2}")
    print(f"Verify c1: {pc.verify(c1, v1, r1)}")
    print(f"Verify c2: {pc.verify(c2, v2, r2)}")

    # Homomorphic property
    c_sum = pc.add_commitments(c1, c2)
    r_sum = (r1 + r2) % (p - 1)
    print(f"\nHomomorphic addition:")
    print(f"Commit({v1}) * Commit({v2}) = {c_sum}")
    print(f"Verify as Commit({v1 + v2}): {pc.verify(c_sum, v1 + v2, r_sum)}")


if __name__ == "__main__":
    demo_commitments()
```

---

## 6. Oblivious Transfer

### 6.1 Definition

**1-out-of-2 Oblivious Transfer (OT)**: The sender has two messages $(m_0, m_1)$. The receiver has a choice bit $b$. After the protocol:
- The receiver learns $m_b$ (the message they chose)
- The sender learns nothing about $b$ (which message was chosen)
- The receiver learns nothing about $m_{1-b}$ (the other message)

### 6.2 Why OT Matters

OT is a fundamental building block in cryptography. Kilian (1988) proved that OT alone is sufficient to implement **any** secure two-party computation — making it a "universal" primitive.

In Yao's garbled circuits (Section 3), OT is used for the evaluator to obtain labels for their input bits without the garbler learning which bits they have.

### 6.3 Simple OT Protocol (RSA-based)

| Step | Sender (has $m_0, m_1$) | Receiver (wants $m_b$) |
|------|------------------------|----------------------|
| 1 | Generate RSA key pair $(e, d, N)$; send $(e, N)$ and two random values $x_0, x_1$ | |
| 2 | | Choose random $k$; compute $v = x_b + k^e \bmod N$; send $v$ |
| 3 | Compute $k_0 = (v - x_0)^d \bmod N$ and $k_1 = (v - x_1)^d \bmod N$ | |
| 4 | Send $m_0' = m_0 + k_0$ and $m_1' = m_1 + k_1$ | |
| 5 | | Compute $m_b = m_b' - k$ (receiver knows $k$ only for the chosen message) |

The receiver can recover $m_b$ because $k_b = ((x_b + k^e) - x_b)^d = k$. For the other message, $(v - x_{1-b})^d$ produces a random-looking value — the receiver cannot recover $k_{1-b}$.

### 6.4 OT Extensions

Performing many OTs is expensive because each requires public-key operations. **OT extension** (Ishai et al., 2003) uses a small number of "base OTs" (e.g., 128) to generate an arbitrary number of OTs using only symmetric-key operations (hashes). This makes large-scale MPC practical.

---

## 7. Threshold Signatures

### 7.1 Motivation

Standard digital signatures have a single point of failure: if the private key is compromised, the attacker can sign anything. **Threshold signatures** distribute the signing key among $n$ parties, requiring $t$-of-$n$ to cooperate for a valid signature.

### 7.2 Properties

- **Threshold**: Any $t$ parties can sign; fewer than $t$ cannot
- **Non-interactive**: Signers produce partial signatures independently (in some schemes)
- **Indistinguishable**: The final signature is identical to a standard single-signer signature

### 7.3 Applications

| System | Threshold | Purpose |
|--------|-----------|---------|
| Cryptocurrency custody | 3-of-5 key holders | Protect large balances |
| Certificate authorities | 2-of-3 HSMs | Prevent single-HSM compromise |
| Distributed validators | 2/3-of-n nodes | Byzantine fault tolerance |
| Government systems | m-of-n officials | Prevent insider threats |

### 7.4 Threshold ECDSA and Schnorr

Threshold Schnorr signatures are simpler than threshold ECDSA because Schnorr has a linear structure:

$$
s = r + c \cdot x = (r_1 + \ldots + r_t) + c \cdot (x_1 + \ldots + x_t)
$$

Each party $i$ computes their partial signature $s_i = r_i + c \cdot x_i$, and the partial signatures are simply added. ECDSA requires more complex MPC techniques because of its multiplicative structure.

---

## 8. Private Information Retrieval

### 8.1 The Problem

A user wants to query a database without the server learning which item was queried. Naive solution: download the entire database — correct but impractical for large databases.

### 8.2 PIR Approaches

| Approach | Communication | Server Computation | Trust |
|----------|--------------|-------------------|-------|
| Trivial (download all) | $O(N)$ | $O(1)$ | None |
| Computational PIR (cPIR) | $O(\text{polylog}(N))$ | $O(N)$ | Computational hardness |
| IT-PIR (multi-server) | $O(\text{polylog}(N))$ | $O(N)$ | Non-colluding servers |
| HE-based PIR | $O(\sqrt{N})$ | $O(N)$ | LWE hardness |

### 8.3 How HE-Based PIR Works

1. Client creates a query vector: $\mathbf{q} = (0, 0, \ldots, 1, \ldots, 0)$ where the 1 is at position $i$ (the desired item)
2. Client encrypts $\mathbf{q}$ with FHE: $\overline{\mathbf{q}} = \text{Enc}(\mathbf{q})$
3. Server computes $\overline{\text{result}} = \overline{\mathbf{q}} \cdot \mathbf{DB}$ (inner product of encrypted query with database)
4. Client decrypts: $\text{result} = \text{Dec}(\overline{\text{result}}) = \text{DB}[i]$

The server learns nothing because the query is encrypted — it cannot distinguish a query for item 1 from a query for item 1000.

---

## 9. Summary

| Protocol | Primitives Used | Key Property |
|----------|----------------|-------------|
| TLS 1.3 | ECDHE, HKDF, AEAD, certificates | Forward secrecy, 1-RTT handshake |
| Signal (Double Ratchet) | X3DH, DH ratchet, KDF chains | Forward + post-compromise security |
| MPC (Garbled Circuits) | OT, symmetric encryption, hash | Compute on joint inputs without revealing them |
| Shamir's Secret Sharing | Polynomial interpolation over $\mathbb{F}_p$ | $t$-of-$n$ reconstruction, perfect secrecy |
| Commitment schemes | Hash functions, DLP | Hide then reveal; homomorphic (Pedersen) |
| Oblivious Transfer | Public-key crypto | Universal primitive for MPC |
| Threshold Signatures | Secret sharing + signatures | Distributed trust, no single point of failure |
| PIR | HE or multi-server | Query databases without revealing the query |

The recurring theme: no primitive works alone. Secure systems require careful composition — and the composition itself must be proven secure, not just the individual pieces.

---

## 10. Exercises

### Exercise 1: TLS 1.3 Key Schedule (Coding)

Implement the TLS 1.3 key derivation schedule:
1. Given a pre-shared key (PSK) and an ECDHE shared secret, compute the early secret, handshake secret, and master secret using HKDF
2. Derive the client and server handshake traffic secrets
3. Verify your implementation against the test vectors in RFC 8448

### Exercise 2: Double Ratchet Simulation (Coding)

Implement a simplified Double Ratchet:
1. Initialize with a shared root key (simulating post-X3DH)
2. Implement the symmetric ratchet (KDF chain for per-message keys)
3. Implement the DH ratchet (new DH exchange when the sender changes)
4. Show that compromising one message key does not reveal past or future message keys
5. Simulate an out-of-order message delivery and show correct decryption

### Exercise 3: Secret Sharing Applications (Coding)

Extend the Shamir implementation:
1. Implement **proactive secret sharing**: periodically refresh shares without changing the secret (generate new random polynomial with the same constant term)
2. Implement **verifiable secret sharing** (Feldman VSS): each share comes with a commitment, and shareholders can verify their shares are consistent without revealing them
3. Demonstrate share refresh: create shares, refresh them, and verify that old and new shares reconstruct the same secret

### Exercise 4: Secure Coin Flip (Coding)

Implement a secure coin-flipping protocol between Alice and Bob:
1. Alice commits to a random bit $a$
2. Bob sends his random bit $b$ (in the clear)
3. Alice opens her commitment
4. The coin flip result is $a \oplus b$

Show that:
- Neither party can bias the result (binding prevents Alice from changing $a$; $a \oplus b$ is uniformly random if either $a$ or $b$ is random)
- Implement using both hash-based and Pedersen commitments
- What happens if we reverse the order (Bob commits first)?

### Exercise 5: Protocol Composition Analysis (Challenging)

Consider a simplified "secure auction" protocol:
1. Each bidder encrypts their bid with Paillier (Lesson 13)
2. The auctioneer homomorphically computes the sum of all bids
3. The auctioneer uses a ZKP (Lesson 12) to prove they computed the sum correctly
4. The winning bid is determined by a garbled circuit that compares all bids

Analyze:
- What cryptographic assumptions does this protocol rely on?
- What happens if the auctioneer is malicious? What additional mechanisms are needed?
- Can you replace any component with a different primitive to improve efficiency?
- Design a modified protocol that handles tied bids
- What is the communication complexity in terms of the number of bidders $n$?

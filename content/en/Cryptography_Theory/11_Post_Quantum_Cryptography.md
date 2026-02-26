# Lesson 11: Post-Quantum Cryptography

**Previous**: [Lattice-Based Cryptography](./10_Lattice_Based_Cryptography.md) | **Next**: [Zero-Knowledge Proofs](./12_Zero_Knowledge_Proofs.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how Shor's algorithm breaks RSA, Diffie-Hellman, and ECC
2. Describe Grover's algorithm and its impact on symmetric cryptography
3. Trace the NIST Post-Quantum Cryptography standardization process and identify the winners
4. Compare the four families of post-quantum cryptography (lattice, code, hash, isogeny)
5. Explain why SIKE was broken and what lessons it teaches about cryptographic confidence
6. Design a hybrid cryptographic migration strategy for an organization
7. Apply the concept of crypto agility to future-proof systems

---

Every public-key cryptosystem deployed on the internet today — RSA, Diffie-Hellman, ECDSA, EdDSA — relies on the hardness of integer factoring or discrete logarithms. In 1994, Peter Shor proved that a sufficiently large quantum computer can solve both problems in polynomial time. While current quantum computers are far too small and error-prone, the threat follows a "harvest now, decrypt later" model: adversaries recording today's encrypted traffic could decrypt it once quantum computers mature. This lesson surveys the quantum threat landscape and the new cryptographic systems designed to withstand it.

> **Analogy:** Post-quantum crypto is like building flood barriers before climate change hits. We cannot wait until quantum computers break RSA to start migrating — the transition takes years, and recorded data may need to stay secret for decades.

## Table of Contents

1. [The Quantum Computing Threat](#1-the-quantum-computing-threat)
2. [Shor's Algorithm](#2-shors-algorithm)
3. [Grover's Algorithm](#3-grovers-algorithm)
4. [NIST PQC Standardization](#4-nist-pqc-standardization)
5. [Lattice-Based Cryptography (Winners)](#5-lattice-based-cryptography-winners)
6. [Code-Based Cryptography](#6-code-based-cryptography)
7. [Hash-Based Signatures](#7-hash-based-signatures)
8. [Isogeny-Based Cryptography (The Cautionary Tale)](#8-isogeny-based-cryptography-the-cautionary-tale)
9. [Hybrid Schemes and Crypto Agility](#9-hybrid-schemes-and-crypto-agility)
10. [Migration Roadmap](#10-migration-roadmap)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)

---

## 1. The Quantum Computing Threat

### 1.1 Quantum Computing Basics (For Cryptographers)

A classical computer manipulates bits (0 or 1). A quantum computer manipulates **qubits** that can exist in superposition:

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1
$$

Two quantum phenomena enable computational advantages:

- **Superposition**: Process $2^n$ states simultaneously with $n$ qubits
- **Entanglement**: Qubits can be correlated in ways with no classical analogue

### 1.2 Impact on Cryptography

| Cryptographic Primitive | Classical Security | Quantum Impact |
|------------------------|-------------------|----------------|
| AES-128 | 128-bit | 64-bit (Grover) — double key length |
| AES-256 | 256-bit | 128-bit (Grover) — still secure |
| SHA-256 | 256-bit preimage | 128-bit (Grover) — still secure |
| RSA-2048 | ~112-bit | **Broken** (Shor) |
| ECDSA P-256 | 128-bit | **Broken** (Shor) |
| DH-2048 | ~112-bit | **Broken** (Shor) |
| Kyber-768 | 192-bit (PQ) | 192-bit — designed to resist |

### 1.3 Timeline Estimates

When will quantum computers be large enough to break RSA-2048?

- **Optimistic** (quantum computing companies): 2030-2035
- **Mainstream** (academic consensus): 2035-2050
- **Conservative**: 2050+
- **NSA/NIST position**: Start migrating now regardless of timeline

The key insight: even if the threat is 15-20 years away, data encrypted today may need to remain confidential for 30+ years (medical records, classified information, trade secrets). The migration itself takes 5-10 years for large organizations.

---

## 2. Shor's Algorithm

### 2.1 The Core Idea

Shor's algorithm efficiently finds the **period** of a function using the **Quantum Fourier Transform (QFT)**. Factoring and discrete logarithms can both be reduced to period-finding.

**Factoring via period-finding:**
1. Given $N$ to factor, choose random $a < N$
2. Find the period $r$ of $f(x) = a^x \bmod N$ (i.e., the smallest $r$ such that $a^r \equiv 1 \pmod{N}$)
3. If $r$ is even, compute $\gcd(a^{r/2} \pm 1, N)$ — this yields a factor of $N$ with high probability

The quantum part is step 2: classically, finding the period requires exponential time; the QFT does it in polynomial time.

### 2.2 Conceptual Demonstration

```python
"""
Shor's algorithm — classical simulation of the period-finding step.

Why a classical simulation? A real quantum implementation requires
a quantum computer. This code demonstrates the mathematical reduction
from factoring to period-finding, which is the classical part of Shor's.

WARNING: This is exponentially slow for large N (it brute-forces the
period classically). The quantum speedup replaces this brute force
with polynomial-time QFT.
"""

import math
import random


def classical_period_finding(a: int, N: int) -> int:
    """
    Find the period r of f(x) = a^x mod N classically.

    This is the step that Shor's quantum algorithm accelerates
    from O(exp(n)) to O(poly(n)).
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            return -1  # No period found (shouldn't happen if gcd(a,N)=1)
    return r


def shors_factoring(N: int, max_attempts: int = 20) -> tuple[int, int]:
    """
    Simulate Shor's algorithm to factor N.

    Steps:
    1. Choose random a
    2. Check gcd(a, N) — if > 1, we got lucky
    3. Find period r of a^x mod N
    4. If r is even, try gcd(a^(r/2) ± 1, N)
    """
    if N % 2 == 0:
        return 2, N // 2

    for attempt in range(max_attempts):
        a = random.randint(2, N - 1)

        # Lucky case: a shares a factor with N
        g = math.gcd(a, N)
        if g > 1:
            print(f"  Attempt {attempt + 1}: Lucky! gcd({a}, {N}) = {g}")
            return g, N // g

        # Find the period (quantum computer does this efficiently)
        r = classical_period_finding(a, N)

        if r == -1 or r % 2 != 0:
            print(f"  Attempt {attempt + 1}: a={a}, r={r} (odd or not found, retry)")
            continue

        # Try to extract a factor
        # Why a^(r/2)? Because a^r ≡ 1 (mod N) means
        # (a^(r/2))^2 ≡ 1 (mod N), so (a^(r/2) - 1)(a^(r/2) + 1) ≡ 0 (mod N)
        x = pow(a, r // 2, N)

        if x == N - 1:  # Trivial: x ≡ -1 (mod N)
            print(f"  Attempt {attempt + 1}: a={a}, r={r} (trivial, retry)")
            continue

        factor1 = math.gcd(x - 1, N)
        factor2 = math.gcd(x + 1, N)

        if factor1 > 1 and factor1 < N:
            print(f"  Attempt {attempt + 1}: a={a}, r={r} → "
                  f"factors: {factor1} × {N // factor1}")
            return factor1, N // factor1

        if factor2 > 1 and factor2 < N:
            print(f"  Attempt {attempt + 1}: a={a}, r={r} → "
                  f"factors: {factor2} × {N // factor2}")
            return factor2, N // factor2

    return None, None


def demo_shors():
    """Demonstrate Shor's factoring on small numbers."""
    test_numbers = [15, 21, 35, 77, 91, 143, 221, 323]

    for N in test_numbers:
        print(f"\nFactoring N = {N}:")
        p, q = shors_factoring(N)
        if p and q:
            assert p * q == N, f"Factoring failed: {p} × {q} ≠ {N}"
            print(f"  Result: {N} = {p} × {q}")
        else:
            print(f"  Failed to factor {N}")


if __name__ == "__main__":
    demo_shors()
```

### 2.3 Resource Estimates for Breaking RSA

To factor a 2048-bit RSA modulus using Shor's algorithm:

| Resource | Estimate |
|----------|----------|
| Logical qubits | ~4,000 |
| Physical qubits (with error correction) | ~4,000,000 (at current error rates) |
| Gate operations | ~$10^{12}$ |
| Wall-clock time | Hours to days |

Current state-of-the-art: IBM's largest processor has ~1,000 physical qubits. The gap is still large but closing.

### 2.4 Breaking ECC and DH

Shor's algorithm also solves the discrete logarithm problem efficiently:
- **DH/DSA**: Find $a$ from $g^a \bmod p$ in $O((\log p)^3)$
- **ECDLP**: Find $k$ from $kG$ on an elliptic curve in $O((\log n)^3)$

ECC actually requires **fewer** qubits to break than RSA because the numbers involved are smaller (256-bit vs. 2048-bit).

---

## 3. Grover's Algorithm

### 3.1 Quadratic Speedup for Search

Grover's algorithm finds a target item in an unstructured database of $N$ items using only $O(\sqrt{N})$ queries, compared to $O(N)$ classically.

### 3.2 Impact on Symmetric Cryptography

| Cipher | Classical Security | Post-Quantum Security | Action |
|--------|-------------------|----------------------|--------|
| AES-128 | $2^{128}$ | $2^{64}$ | Upgrade to AES-256 |
| AES-256 | $2^{256}$ | $2^{128}$ | Already sufficient |
| ChaCha20 | $2^{256}$ | $2^{128}$ | Already sufficient |

### 3.3 Impact on Hash Functions

| Hash | Preimage (Classical) | Preimage (Quantum) | Collision (Quantum) |
|------|---------------------|-------------------|---------------------|
| SHA-256 | $2^{256}$ | $2^{128}$ | $2^{85}$ (BHT algorithm) |
| SHA-384 | $2^{384}$ | $2^{192}$ | $2^{128}$ |
| SHA-512 | $2^{512}$ | $2^{256}$ | $2^{170}$ |

SHA-256 remains secure for preimage resistance but offers reduced collision resistance. For long-term security, SHA-384 or SHA-512 is recommended.

### 3.4 Why Grover Is Less Concerning Than Shor

- **Shor**: Exponential speedup → completely breaks RSA/ECC → requires **replacement**
- **Grover**: Quadratic speedup → halves security bits → requires only **parameter doubling**

---

## 4. NIST PQC Standardization

### 4.1 Timeline

| Date | Milestone |
|------|-----------|
| 2016 | NIST calls for PQC proposals |
| 2017 | 82 submissions received |
| 2019 | Round 2: 26 candidates |
| 2020 | Round 3: 15 candidates (7 finalists + 8 alternates) |
| 2022 | Winners announced |
| 2024 | FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA) published |
| 2024+ | Round 4 continues for additional signatures |

### 4.2 Winners and Standards

| Standard | Algorithm | Family | Purpose |
|----------|-----------|--------|---------|
| **FIPS 203** (ML-KEM) | CRYSTALS-Kyber | Lattice | Key encapsulation |
| **FIPS 204** (ML-DSA) | CRYSTALS-Dilithium | Lattice | Digital signatures |
| **FIPS 205** (SLH-DSA) | SPHINCS+ | Hash-based | Digital signatures (stateless) |
| Round 4 | FALCON | Lattice | Signatures (compact) |
| Round 4 | BIKE, HQC, Classic McEliece | Code-based | KEMs (diversity) |

### 4.3 Why Lattices Dominated

- **Performance**: Lattice schemes are fast (often faster than RSA)
- **Key sizes**: Manageable (1-2 KB, compared to 100+ KB for code-based)
- **Versatility**: Same framework supports KEMs, signatures, and advanced protocols (FHE, ZKPs)
- **Theoretical foundation**: Strong worst-case to average-case reductions

---

## 5. Lattice-Based Cryptography (Winners)

Covered in detail in Lesson 10. Key points for the PQC context:

### 5.1 ML-KEM (Kyber) Summary

- Module-LWE based key encapsulation
- Public key: 800-1568 bytes depending on security level
- Ciphertext: 768-1568 bytes
- Already deployed in Chrome, Signal, iMessage

### 5.2 ML-DSA (Dilithium) Summary

- Module-LWE based digital signatures
- Signature: 2420-4595 bytes
- Fastest PQC signature scheme in most benchmarks

### 5.3 FALCON

- NTRU-lattice based signatures using Gaussian sampling
- **Much smaller signatures** than Dilithium (~666 bytes at NIST Level 1 vs 2420 bytes)
- More complex implementation (requires double-precision floating point for Gaussian sampling)
- Being standardized in a follow-up round

---

## 6. Code-Based Cryptography

### 6.1 The Idea

Error-correcting codes can detect and correct errors in transmitted data. Code-based cryptography reverses this: the code structure is a trapdoor, and without knowing it, decoding is an NP-hard problem.

### 6.2 McEliece Cryptosystem (1978)

The oldest unbroken post-quantum cryptosystem:

1. **Key generation**: Choose a binary Goppa code with generator matrix $G$. Scramble it: $G' = SGP$ where $S$ is invertible and $P$ is a permutation.
2. **Encryption**: $c = mG' + e$ where $e$ is a random error vector of weight $t$
3. **Decryption**: Apply $P^{-1}$, decode using the Goppa code, apply $S^{-1}$

**The catch**: Public keys are enormous.

| Scheme | Public Key | Ciphertext | Security Level |
|--------|-----------|------------|----------------|
| Classic McEliece-348864 | **261 KB** | 128 B | NIST Level 1 |
| Classic McEliece-6960119 | **1,044 KB** | 226 B | NIST Level 5 |

> The 261 KB public key is ~200x larger than Kyber-768's 1.2 KB key. This makes McEliece impractical for most internet protocols but suitable for specialized applications where keys are pre-distributed.

### 6.3 BIKE and HQC

BIKE (Bit Flipping Key Encapsulation) and HQC (Hamming Quasi-Cyclic) are code-based KEMs with much smaller keys (~2-5 KB), but they have non-negligible decryption failure rates that must be carefully analyzed.

---

## 7. Hash-Based Signatures

### 7.1 The Appeal

Hash-based signatures derive their security **solely** from the collision resistance of hash functions — the most conservative and well-understood assumption in cryptography. If SHA-256 is secure, these signatures are secure.

### 7.2 One-Time Signatures (Lamport/Winternitz)

**Lamport's one-time signature** (1979):
1. Generate $2n$ random values: $(\text{sk}_0^1, \text{sk}_1^1, \ldots, \text{sk}_0^n, \text{sk}_1^n)$
2. Public key: hash each value $(\text{pk}_0^1 = H(\text{sk}_0^1), \ldots)$
3. To sign bit $b_i$ of the message hash: reveal $\text{sk}_{b_i}^i$
4. Verification: hash each revealed value and compare to public key

**Problem**: Each key pair can sign exactly **one** message. Reuse leaks the private key.

### 7.3 Merkle Trees: Many Signatures from One-Time Keys

Ralph Merkle's insight: organize $2^h$ one-time key pairs as leaves of a binary hash tree. The root hash is the public key.

```
         Root (public key)
        /                \
      H01                H23
     /    \            /    \
   H0      H1       H2      H3
   |       |        |       |
  OTS_0   OTS_1   OTS_2   OTS_3
```

To sign with key $i$: provide the OTS signature plus the authentication path (sibling hashes from leaf to root).

**Limitation**: The tree has finite capacity ($2^h$ signatures). The signer must track which keys have been used.

### 7.4 XMSS (eXtended Merkle Signature Scheme)

XMSS (RFC 8391) is a **stateful** hash-based signature scheme:
- Uses a Merkle tree of Winternitz one-time signatures
- Signer must maintain state (index of next unused key)
- Very small signatures (~2.5 KB) and fast verification

**Risk**: If the state is lost or duplicated (e.g., VM snapshot restore), the same OTS key may be reused, completely breaking security.

### 7.5 SPHINCS+ (SLH-DSA)

SPHINCS+ (standardized as SLH-DSA in FIPS 205) is a **stateless** hash-based signature:
- Uses a hyper-tree of Merkle trees with FORS (Forest of Random Subsets) at the leaves
- No state to manage — same key can safely sign unlimited messages
- **Trade-off**: Larger signatures (~8-50 KB depending on parameters)

```python
"""
Compare PQC signature sizes.

Why SPHINCS+ despite large signatures? It provides the most conservative
security assumption (only hash function security) and is stateless.
For environments where state management is risky (distributed systems,
HSMs with backup), SPHINCS+ is the safest choice.
"""


def compare_pqc_signatures():
    """Compare signature scheme sizes and properties."""

    schemes = [
        {
            "name": "ECDSA P-256",
            "family": "Elliptic Curve",
            "pubkey_bytes": 64,
            "sig_bytes": 64,
            "quantum_safe": False,
            "stateful": False,
            "assumption": "ECDLP",
        },
        {
            "name": "Ed25519",
            "family": "Elliptic Curve",
            "pubkey_bytes": 32,
            "sig_bytes": 64,
            "quantum_safe": False,
            "stateful": False,
            "assumption": "ECDLP",
        },
        {
            "name": "ML-DSA-65 (Dilithium3)",
            "family": "Lattice",
            "pubkey_bytes": 1952,
            "sig_bytes": 3293,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "Module-LWE",
        },
        {
            "name": "FALCON-512",
            "family": "Lattice",
            "pubkey_bytes": 897,
            "sig_bytes": 666,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "NTRU lattice",
        },
        {
            "name": "SLH-DSA-SHA2-128s (SPHINCS+)",
            "family": "Hash-based",
            "pubkey_bytes": 32,
            "sig_bytes": 7856,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "Hash function",
        },
        {
            "name": "SLH-DSA-SHA2-128f (SPHINCS+)",
            "family": "Hash-based",
            "pubkey_bytes": 32,
            "sig_bytes": 17088,
            "quantum_safe": True,
            "stateful": False,
            "assumption": "Hash function (fast variant)",
        },
        {
            "name": "XMSS (h=10)",
            "family": "Hash-based",
            "pubkey_bytes": 64,
            "sig_bytes": 2500,
            "quantum_safe": True,
            "stateful": True,
            "assumption": "Hash function",
        },
    ]

    print(f"{'Scheme':<35} {'Family':<15} {'PubKey':>8} {'Sig':>8} "
          f"{'PQ':>4} {'State':>6}")
    print("-" * 85)
    for s in schemes:
        pq = "Yes" if s["quantum_safe"] else "No"
        st = "Yes" if s["stateful"] else "No"
        print(f"{s['name']:<35} {s['family']:<15} "
              f"{s['pubkey_bytes']:>6} B {s['sig_bytes']:>6} B "
              f"{pq:>4} {st:>6}")

    # Highlight the trade-off
    print("\n--- Key Trade-offs ---")
    print("FALCON: Smallest PQ signatures but complex implementation")
    print("Dilithium: Best general-purpose PQ signature (balanced)")
    print("SPHINCS+: Most conservative assumption but largest signatures")
    print("XMSS: Small signatures but requires careful state management")


if __name__ == "__main__":
    compare_pqc_signatures()
```

---

## 8. Isogeny-Based Cryptography (The Cautionary Tale)

### 8.1 What Are Isogenies?

An **isogeny** is a structure-preserving map between elliptic curves. The security of isogeny-based crypto relied on the hardness of finding an isogeny between two given supersingular elliptic curves.

### 8.2 SIKE/SIDH

**SIDH** (Supersingular Isogeny Diffie-Hellman) and its KEM variant **SIKE** offered remarkably small key sizes:

| Scheme | Public Key | Ciphertext |
|--------|-----------|------------|
| SIKE-p434 | 330 B | 346 B |
| Kyber-512 | 800 B | 768 B |

SIKE reached the fourth round of the NIST competition.

### 8.3 The Spectacular Break (2022)

In July 2022, Wouter Castryck and Thomas Decru published a devastating attack that broke SIKE in **minutes** on a single laptop. The attack exploited the auxiliary torsion point information that SIDH published alongside the isogeny.

**Key lessons:**

1. **Diversity matters**: Relying on a single assumption is dangerous. NIST wisely selected schemes from multiple families.
2. **Cryptanalytic maturity**: Lattice problems have been studied for 40+ years; isogeny problems for ~10 years. More scrutiny reveals more attacks.
3. **Auxiliary information is dangerous**: SIDH's vulnerability came not from the core isogeny problem but from the extra information (torsion points) needed to make the protocol work.

> **Note**: The break of SIKE does not invalidate all isogeny-based crypto. New proposals (like CSIDH and SQISign) avoid the vulnerability by not publishing torsion points, but they are slower and have larger keys.

---

## 9. Hybrid Schemes and Crypto Agility

### 9.1 Why Hybrid?

During the transition period, we cannot be fully confident in either:
- **Classical crypto alone**: Vulnerable to future quantum computers
- **PQC alone**: Relatively new, may contain undiscovered weaknesses (as SIKE demonstrated)

**Hybrid schemes** combine both:

$$
\text{SharedSecret} = \text{KDF}(\text{ECDH\_secret} \| \text{Kyber\_secret})
$$

If either scheme is broken, the other still provides security. If both are secure, the combined scheme is at least as secure as the stronger one.

### 9.2 Hybrid Deployments in Practice

| System | Classical | Post-Quantum | Since |
|--------|-----------|-------------|-------|
| Chrome/TLS | X25519 | Kyber-768 | August 2023 |
| Signal | X25519 | Kyber-1024 | September 2023 |
| Apple iMessage | ECDH P-256 | Kyber-768 | February 2024 |
| Cloudflare | X25519 | Kyber-768 | 2023 |
| AWS KMS | ECDH | Kyber-768 | 2024 |

### 9.3 Crypto Agility

**Crypto agility** is the ability to swap cryptographic algorithms without redesigning the entire system. This requires:

1. **Algorithm negotiation**: Protocols should negotiate algorithms at runtime (TLS cipher suites already do this)
2. **Abstraction layers**: Application code should not hard-code specific algorithms
3. **Configuration-driven**: Algorithms should be configurable, not compiled-in
4. **Graceful upgrade**: Support both old and new algorithms during transition

```python
"""
Crypto agility pattern — algorithm-agnostic key exchange.

Why crypto agility? When a cryptographic algorithm is broken (like SIKE)
or deprecated, the system should be able to switch algorithms with
configuration changes, not code rewrites.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class KeyPair:
    public_key: bytes
    private_key: bytes


@dataclass
class KEMResult:
    shared_secret: bytes
    ciphertext: bytes


class KEMScheme(ABC):
    """Abstract KEM interface — any algorithm can implement this."""

    @abstractmethod
    def keygen(self) -> KeyPair:
        """Generate a key pair."""
        ...

    @abstractmethod
    def encapsulate(self, public_key: bytes) -> KEMResult:
        """Encapsulate: generate shared secret and ciphertext."""
        ...

    @abstractmethod
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate: recover shared secret from ciphertext."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class HybridKEM(KEMScheme):
    """
    Combine two KEMs for hybrid security.

    Why combine with KDF? Simple concatenation might have subtle issues
    if one scheme's output is correlated with the other's. The KDF
    ensures the combined secret is uniformly random.
    """

    def __init__(self, classical: KEMScheme, post_quantum: KEMScheme):
        self.classical = classical
        self.post_quantum = post_quantum

    @property
    def name(self) -> str:
        return f"Hybrid({self.classical.name}+{self.post_quantum.name})"

    def keygen(self) -> KeyPair:
        ck = self.classical.keygen()
        pk = self.post_quantum.keygen()
        # In practice, keys would be serialized with length prefixes
        return KeyPair(
            public_key=ck.public_key + pk.public_key,
            private_key=ck.private_key + pk.private_key,
        )

    def encapsulate(self, public_key: bytes) -> KEMResult:
        # Split public key (in practice, use proper serialization)
        mid = len(public_key) // 2
        c_result = self.classical.encapsulate(public_key[:mid])
        p_result = self.post_quantum.encapsulate(public_key[mid:])

        import hashlib
        combined_secret = hashlib.sha256(
            c_result.shared_secret + p_result.shared_secret
        ).digest()

        return KEMResult(
            shared_secret=combined_secret,
            ciphertext=c_result.ciphertext + p_result.ciphertext,
        )

    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        mid_key = len(private_key) // 2
        mid_ct = len(ciphertext) // 2

        c_secret = self.classical.decapsulate(
            private_key[:mid_key], ciphertext[:mid_ct]
        )
        p_secret = self.post_quantum.decapsulate(
            private_key[mid_key:], ciphertext[mid_ct:]
        )

        import hashlib
        return hashlib.sha256(c_secret + p_secret).digest()


# Usage:
# kem = HybridKEM(X25519KEM(), KyberKEM())
# keys = kem.keygen()
# result = kem.encapsulate(keys.public_key)
# recovered = kem.decapsulate(keys.private_key, result.ciphertext)
# assert result.shared_secret == recovered
```

---

## 10. Migration Roadmap

### 10.1 Phases of PQC Migration

| Phase | Timeline | Actions |
|-------|----------|---------|
| **1. Inventory** | Now | Catalog all cryptographic usage (libraries, protocols, certificates, hardware) |
| **2. Assess** | Now | Identify highest-risk systems (long-lived secrets, compliance requirements) |
| **3. Test** | Now - 2026 | Deploy hybrid schemes in testing; benchmark performance impact |
| **4. Hybrid Deploy** | 2025-2028 | Deploy hybrid (classical + PQC) in production for key exchange |
| **5. PQC Signatures** | 2026-2030 | Migrate certificate chains, code signing to PQC signatures |
| **6. Full PQC** | 2030+ | Remove classical-only cipher suites |

### 10.2 Challenges

- **Certificate chains**: Each certificate in a TLS chain grows by ~2-3 KB (Dilithium). A 3-certificate chain adds ~7-9 KB to every TLS handshake.
- **Embedded/IoT**: Constrained devices may not have resources for PQC
- **HSMs**: Hardware Security Modules need firmware updates for new algorithms
- **Compliance**: Regulations (FIPS, Common Criteria) must be updated
- **Interoperability**: Both sides of every connection must support PQC

### 10.3 Government Mandates

- **NSA CNSA 2.0 (2022)**: All national security systems must migrate to PQC by 2035
- **White House Memo NSM-10 (2022)**: Federal agencies must inventory quantum-vulnerable cryptography
- **NIST IR 8547 (2024)**: Transition to PQC standards guidance
- **EU**: European Cybersecurity Certification Framework includes PQC requirements

---

## 11. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Shor's algorithm | Breaks RSA, DH, and ECC in polynomial time on quantum computers |
| Grover's algorithm | Halves security of symmetric crypto; double key length suffices |
| NIST standards | ML-KEM (Kyber), ML-DSA (Dilithium), SLH-DSA (SPHINCS+) |
| Lattice-based | Best balance of performance, key size, and versatility |
| Code-based | Conservative but large keys (McEliece: 261 KB) |
| Hash-based | Most conservative assumption; SPHINCS+ is stateless but large signatures |
| SIKE | Broken in 2022; demonstrates need for diversity and cryptanalytic maturity |
| Hybrid schemes | Combine classical + PQC during transition period |
| Crypto agility | Design systems to swap algorithms without rewrites |

---

## 12. Exercises

### Exercise 1: Quantum Threat Assessment (Conceptual)

For each of the following systems, assess the quantum threat level (high/medium/low) and recommend a migration priority:
1. A hospital's electronic medical records system (30-year retention)
2. A real-time gaming platform (session keys live for minutes)
3. A government intelligence agency's classified communications
4. A public website serving blog posts (all content is already public)
5. A cryptocurrency wallet storing private keys long-term

### Exercise 2: Shor's Algorithm Exploration (Coding)

Using the classical period-finding simulation from this lesson:
1. Factor all semiprimes up to 1000 (products of two primes)
2. Record the number of attempts needed for each
3. Plot the success rate vs. the size of $N$
4. Why does the algorithm sometimes need multiple attempts? (Explain mathematically.)

### Exercise 3: PQC Key Size Impact (Coding)

Write a simulation that measures the impact of PQC on TLS handshakes:
1. Model a TLS 1.3 handshake with classical algorithms (X25519 + Ed25519)
2. Replace with PQC (Kyber-768 + Dilithium3)
3. Calculate the total bytes transmitted in each case
4. For a certificate chain of depth 3, how much does each handshake grow?
5. At what network latency does the increased size add noticeable delay?

### Exercise 4: Hybrid KEM Implementation (Coding)

Implement a hybrid KEM that combines X25519 and a simplified LWE scheme:
1. Use the `cryptography` library for X25519
2. Use the LWE implementation from Lesson 10
3. Combine shared secrets with HKDF
4. Write test cases showing that if you "break" one scheme (by leaking its secret), the hybrid shared secret is still secure

### Exercise 5: Crypto Agility Audit (Challenging)

Choose an open-source project (e.g., a web framework, an SSH library, or a VPN client) and audit it for crypto agility:
1. Where are cryptographic algorithms specified? (Configuration? Source code? Constants?)
2. How difficult would it be to add Kyber support?
3. What abstractions exist (or are missing) for algorithm-agnostic crypto?
4. Write a brief migration plan for adding PQC support to the project.

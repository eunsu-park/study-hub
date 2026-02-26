# Cryptography Theory

## Overview

This topic provides a rigorous treatment of modern cryptography, from number-theoretic foundations through classical and post-quantum algorithms to advanced protocols like zero-knowledge proofs and homomorphic encryption. While the Security topic covers *applied* cryptography (using libraries for TLS, JWT, hashing), this topic explains *why* those tools work — the mathematical structures, security proofs, and design principles behind them.

## Prerequisites

- Modular arithmetic basics (gcd, modular inverse)
- Python intermediate level (functions, classes)
- Comfort with mathematical notation (proofs, group theory helpful but not required)
- Recommended: Security L02 (Cryptography Basics) for practical context

## Lesson Plan

### Foundations

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [01_Number_Theory_Foundations.md](./01_Number_Theory_Foundations.md) | ⭐⭐ | Modular arithmetic, GCD, Euler's theorem, CRT, primality testing | Math prerequisites for all later lessons |
| [02_Symmetric_Ciphers.md](./02_Symmetric_Ciphers.md) | ⭐⭐ | AES internals (SubBytes, ShiftRows, MixColumns, AddRoundKey), DES history, Feistel networks | Block cipher construction |
| [03_Block_Cipher_Modes.md](./03_Block_Cipher_Modes.md) | ⭐⭐⭐ | ECB, CBC, CTR, GCM, CCM, padding oracles | Authenticated encryption |
| [04_Hash_Functions.md](./04_Hash_Functions.md) | ⭐⭐⭐ | Merkle-Damgård, SHA-256 internals, birthday attack, HMAC, SHA-3/Keccak | Collision resistance |

### Public-Key Cryptography

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [05_RSA_Cryptosystem.md](./05_RSA_Cryptosystem.md) | ⭐⭐⭐ | RSA key generation, encryption/decryption, OAEP, CCA security, factoring attacks | Mathematical proof of correctness |
| [06_Elliptic_Curve_Cryptography.md](./06_Elliptic_Curve_Cryptography.md) | ⭐⭐⭐⭐ | Elliptic curves over finite fields, point addition, scalar multiplication, ECDLP, Curve25519 | Geometric intuition + algebra |
| [07_Digital_Signatures.md](./07_Digital_Signatures.md) | ⭐⭐⭐ | RSA-PSS, ECDSA, EdDSA, Schnorr signatures, security models | Non-repudiation proofs |
| [08_Key_Exchange.md](./08_Key_Exchange.md) | ⭐⭐⭐ | Diffie-Hellman, ECDH, authenticated key exchange, forward secrecy, Signal protocol | Man-in-the-middle prevention |

### Infrastructure

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [09_PKI_and_Certificates.md](./09_PKI_and_Certificates.md) | ⭐⭐⭐ | X.509, certificate chains, CRL/OCSP, Certificate Transparency, Web of Trust | Trust model comparison |

### Post-Quantum Cryptography

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [10_Lattice_Based_Cryptography.md](./10_Lattice_Based_Cryptography.md) | ⭐⭐⭐⭐ | Lattices, SVP/CVP, LWE, NTRU, CRYSTALS-Kyber, CRYSTALS-Dilithium | NIST PQC standards |
| [11_Post_Quantum_Cryptography.md](./11_Post_Quantum_Cryptography.md) | ⭐⭐⭐⭐ | Code-based (McEliece), hash-based (SPHINCS+), isogeny-based, NIST PQC timeline | Migration strategies |

### Advanced Topics

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [12_Zero_Knowledge_Proofs.md](./12_Zero_Knowledge_Proofs.md) | ⭐⭐⭐⭐ | Interactive proofs, Schnorr ZKP, zk-SNARKs, zk-STARKs, applications | Blockchain and privacy |
| [13_Homomorphic_Encryption.md](./13_Homomorphic_Encryption.md) | ⭐⭐⭐⭐ | Partially/Somewhat/Fully HE, BFV/CKKS schemes, noise growth, bootstrapping | Privacy-preserving computation |
| [14_Applied_Cryptographic_Protocols.md](./14_Applied_Cryptographic_Protocols.md) | ⭐⭐⭐⭐ | TLS 1.3 handshake, Signal/Double Ratchet, MPC, secret sharing, threshold signatures | Real-world protocol analysis |

## Recommended Learning Path

```
Foundations (L01-L04)         Public-Key (L05-L08)         Infrastructure (L09)
       │                            │                            │
       ▼                            ▼                            ▼
  Number Theory              RSA / ECC / Sigs            PKI / Certificates
  Symmetric Ciphers          Key Exchange                Trust Models
  Modes / Hashing            Forward Secrecy
       │                            │                            │
       └────────────────────────────┴────────────────────────────┘
                                    │
                                    ▼
                      Post-Quantum (L10-L11)
                      Lattice / Code / Hash-Based
                                    │
                                    ▼
                      Advanced (L12-L14)
                      ZKP, HE, Protocols
```

## Relationship to Other Topics

| Topic | Relationship |
|-------|-------------|
| Security | Security covers *applied* crypto (library usage, TLS config); this topic covers *theory* (why algorithms work) |
| Math_for_AI | Shares linear algebra, probability foundations; this topic adds number theory, finite fields |
| Formal_Languages | Computational complexity (P, NP) underpins cryptographic hardness assumptions |
| Networking | TLS handshake (L14) connects to Networking L10/L12 |

## Example Code

Example code for this topic is available in `examples/Cryptography_Theory/`.

## Total

- **14 lessons** (4 foundations + 4 public-key + 1 infrastructure + 2 post-quantum + 3 advanced)
- **Difficulty range**: ⭐⭐ to ⭐⭐⭐⭐
- **Languages**: Python (primary)
- **Key libraries**: Built-in (for theory), cryptography (for verification), pycryptodome (for low-level)

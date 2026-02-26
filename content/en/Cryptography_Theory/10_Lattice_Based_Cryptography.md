# Lesson 10: Lattice-Based Cryptography

**Previous**: [PKI and Certificates](./09_PKI_and_Certificates.md) | **Next**: [Post-Quantum Cryptography](./11_Post_Quantum_Cryptography.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define lattices mathematically and visualize them in low dimensions
2. Explain the core lattice problems (SVP, CVP, GapSVP) and why they are believed to be hard
3. Describe the Learning With Errors (LWE) problem and its ring variant (Ring-LWE)
4. Implement a simple LWE-based encryption scheme in Python
5. Outline how CRYSTALS-Kyber (ML-KEM) constructs a key encapsulation mechanism from LWE
6. Compare lattice-based cryptography with RSA and ECC in terms of security, key sizes, and performance
7. Explain why lattice problems are believed to resist quantum attacks

---

The cryptographic systems we have studied so far — RSA (Lesson 5), Diffie-Hellman (Lesson 8), and ECC (Lesson 6) — all derive their security from the hardness of number-theoretic problems: integer factoring and discrete logarithms. In 1994, Peter Shor showed that a sufficiently powerful quantum computer could solve both problems efficiently, rendering these systems obsolete. Lattice-based cryptography offers an alternative foundation built on geometric problems in high-dimensional spaces — problems that appear resistant to both classical and quantum algorithms. This is not a speculative future: NIST selected lattice-based schemes as the primary post-quantum standards in 2024, and their deployment is already underway.

## Table of Contents

1. [What Is a Lattice?](#1-what-is-a-lattice)
2. [Hard Lattice Problems](#2-hard-lattice-problems)
3. [Learning With Errors (LWE)](#3-learning-with-errors-lwe)
4. [Ring-LWE and Module-LWE](#4-ring-lwe-and-module-lwe)
5. [NTRU: A Historical Perspective](#5-ntru-a-historical-perspective)
6. [CRYSTALS-Kyber (ML-KEM)](#6-crystals-kyber-ml-kem)
7. [CRYSTALS-Dilithium (ML-DSA)](#7-crystals-dilithium-ml-dsa)
8. [Comparison with Classical Cryptography](#8-comparison-with-classical-cryptography)
9. [Why Lattice Problems Resist Quantum Attacks](#9-why-lattice-problems-resist-quantum-attacks)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)

---

## 1. What Is a Lattice?

### 1.1 Formal Definition

A **lattice** $\mathcal{L}$ in $\mathbb{R}^n$ is the set of all integer linear combinations of a set of linearly independent basis vectors $\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n \in \mathbb{R}^n$:

$$
\mathcal{L}(\mathbf{B}) = \left\{ \sum_{i=1}^{n} z_i \mathbf{b}_i \mid z_i \in \mathbb{Z} \right\}
$$

where $\mathbf{B} = [\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n]$ is the **basis matrix**.

### 1.2 Intuition

> **Analogy:** Finding the shortest vector in a high-dimensional lattice is like finding the closest subway station in a city with millions of stations and no map. In a 2D grid, you just look around and walk to the nearest one. But in a 1000-dimensional "city," there are so many directions to search that even the fastest computers (and quantum computers) cannot find the shortest path efficiently.

### 1.3 Visualization in 2D

```python
"""
Visualize a 2D lattice and its basis vectors.

Why start with 2D? Lattice problems are easy in low dimensions
(solvable in polynomial time up to ~4-5D). The difficulty explosion
in high dimensions is what makes lattice crypto possible.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_lattice_2d(basis: np.ndarray, range_val: int = 5,
                     title: str = "2D Lattice"):
    """
    Plot a 2D lattice with its basis vectors.

    Parameters:
        basis: 2x2 matrix where columns are basis vectors
        range_val: range of integer coefficients to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Generate lattice points as integer combinations of basis vectors
    # Why nested loops? Each lattice point is z1*b1 + z2*b2
    points = []
    for z1 in range(-range_val, range_val + 1):
        for z2 in range(-range_val, range_val + 1):
            point = z1 * basis[:, 0] + z2 * basis[:, 1]
            points.append(point)

    points = np.array(points)

    # Plot lattice points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=20, zorder=3)

    # Plot basis vectors as arrows from origin
    origin = np.array([0, 0])
    ax.annotate('', xy=basis[:, 0], xytext=origin,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=basis[:, 1], xytext=origin,
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.annotate(f'b1={basis[:, 0]}', xy=basis[:, 0], fontsize=10, color='red')
    ax.annotate(f'b2={basis[:, 1]}', xy=basis[:, 1], fontsize=10, color='green')

    # Highlight shortest vector
    distances = np.linalg.norm(points, axis=1)
    distances[distances == 0] = np.inf  # Exclude origin
    shortest_idx = np.argmin(distances)
    ax.scatter(*points[shortest_idx], c='orange', s=100, zorder=4,
               label=f'Shortest: {points[shortest_idx]}')

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig("lattice_2d.png", dpi=100)
    plt.close()
    print(f"Lattice plotted with {len(points)} points")
    print(f"Shortest non-zero vector: {points[shortest_idx]}, "
          f"length: {distances[shortest_idx]:.4f}")


# "Good" basis — nearly orthogonal, easy to find shortest vector
good_basis = np.array([[1, 0],
                        [0, 1]])

# "Bad" basis — same lattice, but skewed, harder to work with
bad_basis = np.array([[1, 47],
                       [0, 53]])

print("=== Good Basis (Z^2 standard) ===")
plot_lattice_2d(good_basis, title="Good Basis")

print("\n=== Bad Basis (same lattice, skewed) ===")
# Why does basis quality matter? An adversary given the bad basis
# cannot easily find short vectors, while the good basis makes it trivial.
# Lattice crypto uses bad bases as public keys and good bases as private keys.
plot_lattice_2d(bad_basis, range_val=3, title="Bad Basis (same lattice)")
```

### 1.4 Key Property: Same Lattice, Different Bases

A lattice has infinitely many bases. Two bases $\mathbf{B}_1$ and $\mathbf{B}_2$ generate the same lattice if and only if $\mathbf{B}_2 = \mathbf{B}_1 \mathbf{U}$ where $\mathbf{U}$ is a unimodular matrix ($\det(\mathbf{U}) = \pm 1$, integer entries).

This is analogous to how the same number can be expressed in different bases — and the quality of the basis matters enormously for solving lattice problems.

---

## 2. Hard Lattice Problems

### 2.1 Shortest Vector Problem (SVP)

**Problem**: Given a lattice basis $\mathbf{B}$, find the shortest non-zero lattice vector.

$$
\text{Find } \mathbf{v} \in \mathcal{L}(\mathbf{B}) \setminus \{\mathbf{0}\} \text{ that minimizes } \|\mathbf{v}\|
$$

The SVP is NP-hard under randomized reductions. No known algorithm (classical or quantum) solves it in polynomial time for arbitrary lattices.

### 2.2 Closest Vector Problem (CVP)

**Problem**: Given a lattice basis $\mathbf{B}$ and a target point $\mathbf{t}$, find the lattice point closest to $\mathbf{t}$.

$$
\text{Find } \mathbf{v} \in \mathcal{L}(\mathbf{B}) \text{ that minimizes } \|\mathbf{v} - \mathbf{t}\|
$$

CVP is at least as hard as SVP (there is a polynomial-time reduction from SVP to CVP).

### 2.3 Approximate Versions (GapSVP, ApproxCVP)

In cryptography, we usually work with **approximate** versions where we need to find a vector within a factor $\gamma$ of the shortest:

$$
\gamma\text{-GapSVP}: \text{Find } \mathbf{v} \text{ with } \|\mathbf{v}\| \leq \gamma \cdot \lambda_1(\mathcal{L})
$$

where $\lambda_1(\mathcal{L})$ is the length of the shortest non-zero vector.

For polynomial $\gamma$, the approximate problems are still believed to be hard.

### 2.4 Best Known Algorithms

| Algorithm | Time Complexity | Type |
|-----------|----------------|------|
| LLL (1982) | $\text{poly}(n)$ | Finds $2^{O(n)}$ approximation |
| BKZ (Block Korkine-Zolotarev) | $2^{O(n)}$ | Better approximation, tunable block size |
| Sieving algorithms | $2^{O(n)}$ | Best asymptotic for exact SVP |
| Quantum sieving | $2^{O(n)}$ | Only constant factor improvement |

The crucial observation: even quantum algorithms achieve only a **constant factor** speedup for lattice problems, unlike the **exponential** speedup Shor's algorithm provides for factoring and discrete log.

---

## 3. Learning With Errors (LWE)

### 3.1 The LWE Problem

Introduced by Oded Regev in 2005, LWE has become the foundation of modern lattice cryptography.

**Setup**: Choose dimension $n$, modulus $q$, and error distribution $\chi$ (typically a discrete Gaussian with small standard deviation $\sigma$).

**Problem**: Given many samples of the form

$$
(\mathbf{a}_i, b_i = \langle \mathbf{a}_i, \mathbf{s} \rangle + e_i \pmod{q})
$$

where $\mathbf{a}_i \in \mathbb{Z}_q^n$ is random, $\mathbf{s} \in \mathbb{Z}_q^n$ is a secret vector, and $e_i \leftarrow \chi$ is a small error, find $\mathbf{s}$.

### 3.2 Why Is LWE Hard?

Without the error $e_i$, this is just a system of linear equations — solvable in $O(n^3)$ by Gaussian elimination. The small error terms make the system **noisy**, and recovering $\mathbf{s}$ from noisy linear equations is computationally equivalent to hard lattice problems.

**Regev's theorem**: Solving LWE is at least as hard as solving worst-case approximate lattice problems (GapSVP) in polynomial time.

> **Analogy:** Imagine you're trying to determine the slope of a line, but every measurement has a small random error. With exact measurements, two points suffice. With noise, you need many measurements and statistical techniques. LWE is like this, but in $n$ dimensions where $n$ is hundreds or thousands — the noise makes the system exponentially harder, not just statistically harder.

### 3.3 LWE Encryption Scheme

```python
"""
A simple LWE-based public-key encryption scheme.

This is the Regev encryption scheme — the conceptual ancestor of
all modern lattice-based crypto. It encrypts one bit at a time.

Why such small parameters here? Real implementations use n=512 or
n=1024 with q around 2^12 to 2^32. We use small values to make
the computation transparent.
"""

import numpy as np
import secrets


def lwe_keygen(n: int = 8, q: int = 97, num_samples: int = 20):
    """
    Generate LWE key pair.

    Parameters:
        n: dimension of the secret vector (security parameter)
        q: modulus (must be prime, much larger than the error)
        num_samples: number of LWE samples (rows of A)

    Returns:
        public_key: (A, b)  where b = A*s + e mod q
        secret_key: s
    """
    # Secret key: random vector in Z_q^n
    s = np.array([secrets.randbelow(q) for _ in range(n)])

    # Public matrix: random m x n matrix
    A = np.array([[secrets.randbelow(q) for _ in range(n)]
                   for _ in range(num_samples)])

    # Error vector: small Gaussian noise
    # Why small error? The error must be small enough that decryption
    # works (roughly |e| < q/4), but large enough that the LWE problem
    # is hard. This tension determines the parameters.
    sigma = 2.0
    e = np.round(np.random.normal(0, sigma, num_samples)).astype(int) % q

    # Public key: b = A*s + e mod q
    b = (A @ s + e) % q

    return (A, b), s


def lwe_encrypt(public_key, bit: int, q: int = 97):
    """
    Encrypt a single bit (0 or 1).

    Why a random subset sum? The encryption creates a new LWE sample
    by combining random rows of the public key. This "rerandomizes"
    the public key, ensuring each ciphertext looks fresh.
    """
    A, b = public_key
    m = len(b)

    # Choose a random binary vector (subset selector)
    r = np.array([secrets.randbelow(2) for _ in range(m)])

    # Ciphertext: (u, v) where
    #   u = r^T * A mod q  (n-dimensional vector)
    #   v = r^T * b + bit * floor(q/2) mod q  (scalar)
    u = (r @ A) % q
    v = (r @ b + bit * (q // 2)) % q

    return u, v


def lwe_decrypt(secret_key, ciphertext, q: int = 97) -> int:
    """
    Decrypt a ciphertext.

    Why does this work? v - u*s = r^T*(A*s + e) + bit*q/2 - (r^T*A)*s
                                = r^T*e + bit*q/2
    Since r^T*e is small (bounded by m*max_error), and q/2 is large,
    we can determine the bit by checking if the result is closer to
    0 (bit=0) or q/2 (bit=1).
    """
    u, v = ciphertext

    # Compute v - <u, s> mod q
    decrypted = (v - u @ secret_key) % q

    # Decision: closer to 0 → bit 0, closer to q/2 → bit 1
    if decrypted > q // 4 and decrypted < 3 * q // 4:
        return 1
    else:
        return 0


def demo_lwe():
    """Demonstrate LWE encryption and decryption."""
    n, q = 16, 97
    num_samples = 40

    print(f"LWE parameters: n={n}, q={q}, samples={num_samples}")

    # Key generation
    public_key, secret_key = lwe_keygen(n, q, num_samples)
    print(f"Secret key (first 5): {secret_key[:5]}")

    # Encrypt and decrypt multiple bits
    test_bits = [0, 1, 1, 0, 1, 0, 0, 1]
    decrypted_bits = []

    for bit in test_bits:
        ct = lwe_encrypt(public_key, bit, q)
        dec = lwe_decrypt(secret_key, ct, q)
        decrypted_bits.append(dec)

    print(f"Original:  {test_bits}")
    print(f"Decrypted: {decrypted_bits}")
    print(f"All correct: {test_bits == decrypted_bits}")


if __name__ == "__main__":
    demo_lwe()
```

### 3.4 Decryption Correctness Analysis

The decryption noise is $r^T \mathbf{e}$, where $r \in \{0,1\}^m$ and $\mathbf{e}$ has entries bounded by $B$. The noise magnitude is bounded by:

$$
|r^T \mathbf{e}| \leq m \cdot B
$$

For correct decryption, we need $m \cdot B < q/4$. This constrains the relationship between the security parameter (which requires large $m$ and $B$) and correctness (which requires small $m \cdot B$ relative to $q$).

---

## 4. Ring-LWE and Module-LWE

### 4.1 The Efficiency Problem with LWE

Standard LWE has large key sizes because the public matrix $\mathbf{A}$ is $m \times n$ random elements in $\mathbb{Z}_q$. For 128-bit security:
- $n \approx 512$, $m \approx 1024$
- Public key $\approx$ 500 KB (impractically large)

### 4.2 Ring-LWE

**Ring-LWE** replaces vectors and matrices with polynomials in the ring:

$$
R_q = \mathbb{Z}_q[x] / (x^n + 1)
$$

where $n$ is a power of 2. Multiplication in this ring can be computed in $O(n \log n)$ using the Number Theoretic Transform (NTT), which is the modular analogue of the FFT.

**Key size reduction**: A single ring element (polynomial of degree $n-1$) replaces an entire $n \times n$ matrix, reducing key sizes by a factor of $n$.

### 4.3 Module-LWE

Module-LWE is a middle ground between LWE and Ring-LWE, working with small matrices of ring elements. For a $k \times k$ matrix of ring elements:

- $k = 1$: Ring-LWE
- $k = n$: Standard LWE (with polynomial structure)
- $k = 2, 3, 4$: Module-LWE (used in Kyber/Dilithium)

**Why Module-LWE?** It provides a flexible security/performance trade-off without relying on the full algebraic structure of Ring-LWE.

### 4.4 Parameter Comparison

| Scheme | Key Size | Ciphertext Size | Security Assumption |
|--------|----------|-----------------|---------------------|
| LWE ($n=512$) | ~500 KB | ~500 KB | LWE (strongest) |
| Ring-LWE ($n=1024$) | ~1 KB | ~1 KB | Ring-LWE (more structured) |
| Module-LWE ($k=3, n=256$) | ~1.5 KB | ~1.5 KB | Module-LWE (balanced) |
| RSA-2048 | 256 B | 256 B | Factoring |
| ECC P-256 | 64 B | 64 B | ECDLP |

---

## 5. NTRU: A Historical Perspective

### 5.1 The First Practical Lattice Scheme

NTRU, introduced in 1996 by Hoffstein, Pipher, and Silverman, was the first practical public-key cryptosystem based on lattice problems — predating LWE by nearly a decade.

### 5.2 How NTRU Works (Simplified)

NTRU operates in the polynomial ring $\mathbb{Z}[x]/(x^N - 1)$ with two moduli $p$ and $q$:

1. **Key generation**: Choose random polynomials $f$ and $g$ with small coefficients. The public key is $h = p \cdot g \cdot f^{-1} \pmod{q}$.
2. **Encryption**: For message polynomial $m$ and random blinding polynomial $r$: $c = r \cdot h + m \pmod{q}$.
3. **Decryption**: Multiply by $f$ and reduce modulo $p$.

### 5.3 NTRU's Legacy

NTRU influenced all subsequent lattice schemes but was not selected for NIST standardization due to:
- Patent history (now expired)
- Slightly less efficient than Kyber for equivalent security
- Less understood security reduction compared to (Module-)LWE

NTRU was submitted as NTRUEncrypt and NTRU-HRSS to the NIST PQC competition and reached the final round.

---

## 6. CRYSTALS-Kyber (ML-KEM)

### 6.1 NIST's Choice

In 2024, NIST standardized CRYSTALS-Kyber as **ML-KEM** (Module-Lattice-based Key Encapsulation Mechanism) in FIPS 203. It is the primary post-quantum key exchange mechanism and will be used in TLS, SSH, VPNs, and virtually all internet protocols.

### 6.2 KEM vs. Key Exchange

A KEM (Key Encapsulation Mechanism) is slightly different from key exchange:

| | Key Exchange (DH) | KEM (Kyber) |
|---|---|---|
| **Interaction** | Both parties contribute randomness | One party encapsulates a random key |
| **Output** | Shared secret derived from both inputs | Shared secret generated by encapsulator |
| **Round trips** | 1 round trip | 1 message (encapsulate) + 1 reply (accept) |

### 6.3 Kyber Overview

**Parameters** (Kyber-768, NIST Security Level 3):
- Module dimension: $k = 3$
- Polynomial degree: $n = 256$
- Modulus: $q = 3329$
- Error distribution: Centered binomial with $\eta = 2$

**Key Generation**:
1. Sample random matrix $\mathbf{A} \in R_q^{k \times k}$
2. Sample secret $\mathbf{s} \in R_q^k$ and error $\mathbf{e} \in R_q^k$ (small)
3. Public key: $(\mathbf{A}, \mathbf{t} = \mathbf{A}\mathbf{s} + \mathbf{e})$
4. Secret key: $\mathbf{s}$

**Encapsulation** (Alice to Bob):
1. Sample random $\mathbf{r}, \mathbf{e}_1, e_2$ (small)
2. Compute $\mathbf{u} = \mathbf{A}^T\mathbf{r} + \mathbf{e}_1$
3. Compute $v = \mathbf{t}^T\mathbf{r} + e_2 + \lceil q/2 \rceil \cdot m$
4. Ciphertext: $(\mathbf{u}, v)$

**Decapsulation** (Bob):
1. Compute $v - \mathbf{s}^T\mathbf{u}$
2. Round to recover $m$

### 6.4 Kyber Parameter Sets

| Parameter Set | NIST Level | $k$ | Public Key | Ciphertext | Shared Secret |
|--------------|------------|-----|------------|------------|---------------|
| Kyber-512 (ML-KEM-512) | 1 (AES-128) | 2 | 800 B | 768 B | 32 B |
| Kyber-768 (ML-KEM-768) | 3 (AES-192) | 3 | 1,184 B | 1,088 B | 32 B |
| Kyber-1024 (ML-KEM-1024) | 5 (AES-256) | 4 | 1,568 B | 1,568 B | 32 B |

### 6.5 Kyber in Practice

Kyber-768 is already being deployed:
- **Google Chrome** (August 2023): Hybrid X25519+Kyber-768 in TLS
- **Signal** (September 2023): PQXDH (post-quantum X3DH with Kyber)
- **Apple iMessage** (February 2024): PQ3 protocol with Kyber-768
- **AWS KMS** (2024): Hybrid TLS with Kyber

---

## 7. CRYSTALS-Dilithium (ML-DSA)

### 7.1 Digital Signatures from Lattices

CRYSTALS-Dilithium was standardized as **ML-DSA** (Module-Lattice-based Digital Signature Algorithm) in FIPS 204. It replaces (or supplements) ECDSA and EdDSA for digital signatures.

### 7.2 Fiat-Shamir with Aborts

Dilithium uses a technique called **Fiat-Shamir with Aborts**:

1. The signer generates a random masking vector $\mathbf{y}$
2. Computes a commitment $\mathbf{w} = \mathbf{A}\mathbf{y}$
3. Derives a challenge $c = H(\mathbf{w}, \text{message})$
4. Computes the response $\mathbf{z} = \mathbf{y} + c\mathbf{s}$
5. **Abort check**: If $\mathbf{z}$ is too large (would leak information about $\mathbf{s}$), discard and restart

The abort step is essential — without it, an attacker could statistically recover the secret key from many signatures.

### 7.3 Dilithium Sizes

| Parameter Set | NIST Level | Public Key | Signature | Signing Time |
|--------------|------------|------------|-----------|-------------|
| Dilithium2 (ML-DSA-44) | 2 | 1,312 B | 2,420 B | ~0.5 ms |
| Dilithium3 (ML-DSA-65) | 3 | 1,952 B | 3,293 B | ~0.8 ms |
| Dilithium5 (ML-DSA-87) | 5 | 2,592 B | 4,595 B | ~1.0 ms |

Compared to ECDSA P-256 (64 B signature), Dilithium signatures are ~40-70x larger. This is a significant consideration for constrained environments.

---

## 8. Comparison with Classical Cryptography

### 8.1 Size Comparison

```python
"""
Compare key and ciphertext/signature sizes across crypto families.

Why size matters: Larger keys and ciphertexts increase bandwidth,
storage, and latency. This is especially critical for:
- TLS handshakes (adding ~1-2 KB)
- IoT devices (limited bandwidth)
- Certificate chains (each cert grows)
- Blockchain (every node stores every signature)
"""


def compare_sizes():
    """Compare sizes of different cryptographic schemes."""

    schemes = {
        "RSA-2048": {
            "type": "KEM + Signature",
            "public_key": 256,
            "private_key": 256,
            "ciphertext_or_sig": 256,
            "security_level": "112-bit",
            "quantum_safe": False,
        },
        "RSA-3072": {
            "type": "KEM + Signature",
            "public_key": 384,
            "private_key": 384,
            "ciphertext_or_sig": 384,
            "security_level": "128-bit",
            "quantum_safe": False,
        },
        "ECDSA P-256": {
            "type": "Signature",
            "public_key": 64,
            "private_key": 32,
            "ciphertext_or_sig": 64,
            "security_level": "128-bit",
            "quantum_safe": False,
        },
        "X25519 + Ed25519": {
            "type": "KEM + Signature",
            "public_key": 32,
            "private_key": 32,
            "ciphertext_or_sig": 64,
            "security_level": "128-bit",
            "quantum_safe": False,
        },
        "Kyber-768 (ML-KEM)": {
            "type": "KEM",
            "public_key": 1184,
            "private_key": 2400,
            "ciphertext_or_sig": 1088,
            "security_level": "192-bit (PQ)",
            "quantum_safe": True,
        },
        "Dilithium3 (ML-DSA)": {
            "type": "Signature",
            "public_key": 1952,
            "private_key": 4000,
            "ciphertext_or_sig": 3293,
            "security_level": "192-bit (PQ)",
            "quantum_safe": True,
        },
    }

    print(f"{'Scheme':<25} {'Type':<18} {'PubKey':>8} {'CT/Sig':>8} "
          f"{'Security':<14} {'PQ?'}")
    print("-" * 90)

    for name, data in schemes.items():
        pq = "Yes" if data["quantum_safe"] else "No"
        print(f"{name:<25} {data['type']:<18} "
              f"{data['public_key']:>6} B {data['ciphertext_or_sig']:>6} B "
              f"{data['security_level']:<14} {pq}")


if __name__ == "__main__":
    compare_sizes()
```

### 8.2 Performance Comparison

| Operation | RSA-2048 | ECDSA P-256 | Kyber-768 | Dilithium3 |
|-----------|----------|-------------|-----------|------------|
| KeyGen | ~1 ms | ~0.1 ms | ~0.05 ms | ~0.3 ms |
| Encrypt/Encapsulate | ~0.1 ms | ~0.2 ms | ~0.07 ms | N/A |
| Decrypt/Decapsulate | ~1 ms | ~0.2 ms | ~0.08 ms | N/A |
| Sign | N/A | ~0.2 ms | N/A | ~0.8 ms |
| Verify | N/A | ~0.3 ms | N/A | ~0.3 ms |

Lattice schemes are surprisingly **fast** — often faster than RSA. The trade-off is in **size**, not speed.

---

## 9. Why Lattice Problems Resist Quantum Attacks

### 9.1 Shor's Algorithm and Its Limits

Shor's algorithm exploits the **periodic structure** in modular exponentiation to find factors and discrete logarithms efficiently. Specifically, it uses the Quantum Fourier Transform to find the period of $f(x) = a^x \bmod N$.

Lattice problems lack this periodic structure. There is no known way to formulate SVP or LWE as a hidden subgroup problem that Shor's algorithm can exploit.

### 9.2 Grover's Algorithm

Grover's algorithm provides a generic $O(\sqrt{N})$ speedup for unstructured search. Applied to lattice problems:
- Brute-force SVP in dimension $n$: classical $2^{O(n)}$, quantum $2^{O(n/2)}$
- This is a constant factor in the exponent, not a qualitative improvement

To maintain security against Grover, we simply increase dimension by ~2x (equivalent to doubling the key size for symmetric ciphers).

### 9.3 Current State of Quantum Lattice Attacks

The best known quantum algorithms for lattice problems (quantum sieving, quantum BKZ) achieve at most a small polynomial speedup over classical algorithms. No exponential quantum speedup is known, and there are theoretical arguments (based on the structure of the lattice problem) why one may not exist.

---

## 10. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Lattice | Regular grid of points in $\mathbb{R}^n$; hard problems arise in high dimensions |
| SVP/CVP | Finding shortest/closest vectors — believed hard classically and quantumly |
| LWE | Noisy linear equations; reduces to worst-case lattice problems |
| Ring/Module-LWE | Structured variants enabling practical key sizes |
| Kyber (ML-KEM) | NIST's standard PQ key encapsulation; ~1.2 KB keys |
| Dilithium (ML-DSA) | NIST's standard PQ signatures; ~2-3 KB signatures |
| Quantum resistance | No known exponential quantum speedup for lattice problems |

---

## 11. Exercises

### Exercise 1: Lattice Basics (Conceptual)

Given the basis $\mathbf{B} = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$:

1. List all lattice points $(x, y)$ where $|x| \leq 10$ and $|y| \leq 10$
2. Find the shortest non-zero vector by inspection
3. Find an alternative basis for the same lattice that is "more orthogonal" (apply the LLL algorithm by hand for the 2D case)

### Exercise 2: LWE Encryption (Coding)

Extend the LWE encryption demo to encrypt multi-bit messages:
1. Implement a byte-by-byte encryption scheme that encrypts 8 bits at a time
2. Add error correction: what happens if you increase the error standard deviation? At what point does decryption start failing?
3. Plot the decryption failure rate as a function of error standard deviation for fixed $n$ and $q$.

### Exercise 3: Ring Operations (Coding)

Implement polynomial arithmetic in $\mathbb{Z}_q[x]/(x^n + 1)$:
1. Addition, subtraction, and multiplication of polynomials modulo $x^n + 1$ and $q$
2. Implement the schoolbook multiplication ($O(n^2)$) and compare with NTT-based multiplication ($O(n \log n)$)
3. Benchmark both for $n = 256, 512, 1024$

### Exercise 4: Parameter Selection (Conceptual + Coding)

For an LWE-based encryption scheme:
1. Fix the security level at 128 bits (meaning $n \cdot \log_2 q \cdot \sigma \geq 128$ by the concrete security estimate)
2. Find the minimum $n$ and $q$ that allow correct decryption with error probability $< 2^{-40}$
3. Compare the resulting key size with RSA-3072 and Kyber-768

### Exercise 5: Hybrid Key Exchange (Challenging)

Implement a hybrid key exchange that combines X25519 and a simplified Kyber-like scheme:
1. Perform both X25519 ECDH and your LWE-based key exchange
2. Combine both shared secrets using HKDF (Lesson 8)
3. Explain why hybrid schemes are recommended during the post-quantum transition: what happens if one of the two schemes is broken?

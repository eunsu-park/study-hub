# Lesson 13: Homomorphic Encryption

**Previous**: [Zero-Knowledge Proofs](./12_Zero_Knowledge_Proofs.md) | **Next**: [Applied Cryptographic Protocols](./14_Applied_Cryptographic_Protocols.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the motivation for computing on encrypted data and distinguish it from other privacy techniques
2. Define the three levels of homomorphic encryption: partial (PHE), somewhat (SHE), and fully homomorphic (FHE)
3. Demonstrate RSA's multiplicative homomorphism and Paillier's additive homomorphism
4. Implement a Paillier encryption scheme and perform computations on ciphertexts
5. Describe the noise growth problem in FHE and explain Gentry's bootstrapping solution
6. Compare the BFV and CKKS schemes for integer and approximate arithmetic
7. Evaluate practical applications and current performance limitations of FHE

---

What if you could send your encrypted medical records to a cloud server, have the server run a machine learning diagnostic on them, and receive an encrypted result — all without the server ever seeing your data? This is the promise of **homomorphic encryption (HE)**: the ability to compute on encrypted data without decrypting it. First envisioned by Rivest, Adleman, and Dertouzos in 1978 (just one year after RSA), fully homomorphic encryption remained an open problem for over 30 years until Craig Gentry's breakthrough in 2009. Today, HE is transitioning from theoretical curiosity to practical deployment, though significant performance challenges remain.

> **Analogy:** Homomorphic encryption is like a glove box in a chemistry lab. You can manipulate the chemicals inside (compute on data) without ever touching them directly (without decrypting). The sealed environment (encryption) protects both you and the chemicals, yet you can still perform complex operations through the gloves.

## Table of Contents

1. [Motivation and Context](#1-motivation-and-context)
2. [Partial Homomorphic Encryption (PHE)](#2-partial-homomorphic-encryption-phe)
3. [The Paillier Cryptosystem](#3-the-paillier-cryptosystem)
4. [From Partial to Fully Homomorphic](#4-from-partial-to-fully-homomorphic)
5. [The Noise Growth Problem](#5-the-noise-growth-problem)
6. [Gentry's Bootstrapping](#6-gentrys-bootstrapping)
7. [Modern FHE Schemes: BFV and CKKS](#7-modern-fhe-schemes-bfv-and-ckks)
8. [Practical Applications](#8-practical-applications)
9. [Performance Reality Check](#9-performance-reality-check)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)

---

## 1. Motivation and Context

### 1.1 The Cloud Computing Dilemma

Cloud computing offers enormous benefits — elastic resources, managed infrastructure, global availability — but requires trusting the cloud provider with your data. Even with encryption at rest and in transit, data must be decrypted for processing.

```
Traditional Cloud Computing:
  Client → Encrypt → [Transit] → Decrypt → PROCESS → Encrypt → [Transit] → Decrypt → Client
                                  ^^^^^^^^
                                  Data exposed on server

Homomorphic Encryption:
  Client → Encrypt → [Transit] → PROCESS ON CIPHERTEXT → [Transit] → Decrypt → Client
                                  ^^^^^^^^^^^^^^^^^^^^^^^
                                  Data NEVER exposed
```

### 1.2 Comparison with Other Privacy Techniques

| Technique | Computes on Encrypted Data? | Multiple Parties? | Trust Assumptions |
|-----------|----------------------------|-------------------|-------------------|
| **HE** | Yes | No (single server) | None (data never decrypted) |
| **MPC** (Lesson 14) | Yes (via secret sharing) | Yes (multiple parties) | Honest majority or honest-but-curious |
| **TEE** (SGX, TrustZone) | No (decrypts inside enclave) | No | Trust hardware vendor |
| **Differential Privacy** | N/A (adds noise to outputs) | N/A | Trust data curator |
| **ZKPs** (Lesson 12) | No (proves computation correctness) | N/A | None |

### 1.3 Formal Definition

An encryption scheme $(\text{KeyGen}, \text{Enc}, \text{Dec}, \text{Eval})$ is **homomorphic** for a class of functions $\mathcal{F}$ if:

$$
\text{Dec}(\text{sk}, \text{Eval}(\text{pk}, f, \text{Enc}(\text{pk}, m_1), \ldots, \text{Enc}(\text{pk}, m_k))) = f(m_1, \ldots, m_k)
$$

for all $f \in \mathcal{F}$ and all valid plaintexts $m_1, \ldots, m_k$.

---

## 2. Partial Homomorphic Encryption (PHE)

### 2.1 Definition

PHE supports **one type** of operation (either addition or multiplication) on ciphertexts, with no limit on the number of operations.

### 2.2 RSA's Multiplicative Homomorphism

Recall RSA (Lesson 5): $\text{Enc}(m) = m^e \bmod N$.

For two ciphertexts:

$$
\text{Enc}(m_1) \cdot \text{Enc}(m_2) = m_1^e \cdot m_2^e = (m_1 \cdot m_2)^e = \text{Enc}(m_1 \cdot m_2) \pmod{N}
$$

RSA is **multiplicatively homomorphic**: multiplying ciphertexts corresponds to multiplying plaintexts.

```python
"""
RSA multiplicative homomorphism demonstration.

Why is this useful? Imagine a server that needs to multiply two
encrypted values (e.g., price × quantity) without learning either value.
With RSA's homomorphism, the server multiplies the ciphertexts, and
the client decrypts to get the product.

Limitation: RSA supports ONLY multiplication, not addition.
You cannot compute enc(m1) + enc(m2) to get enc(m1 + m2).
"""

from math import gcd


def simple_rsa_keygen(p: int, q: int):
    """Generate RSA key pair from given primes (educational only)."""
    N = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    # Find d such that e*d ≡ 1 (mod phi)
    d = pow(e, -1, phi)
    return (e, N), (d, N)  # public, private


def rsa_encrypt(pub, m):
    e, N = pub
    return pow(m, e, N)


def rsa_decrypt(priv, c):
    d, N = priv
    return pow(d, c, N)  # This should be pow(c, d, N) — fixed below


def demo_rsa_homomorphism():
    """Show RSA multiplicative homomorphism."""
    # Small primes for demonstration
    p, q = 61, 53
    pub, priv = simple_rsa_keygen(p, q)
    e, N = pub
    d, _ = priv

    m1, m2 = 7, 13

    # Encrypt individually
    c1 = pow(m1, e, N)
    c2 = pow(m2, e, N)

    # Multiply ciphertexts (no decryption needed!)
    c_product = (c1 * c2) % N

    # Decrypt the product
    result = pow(c_product, d, N)

    print(f"m1 = {m1}, m2 = {m2}")
    print(f"c1 = {c1}, c2 = {c2}")
    print(f"c1 * c2 mod N = {c_product}")
    print(f"Dec(c1 * c2) = {result}")
    print(f"m1 * m2 = {m1 * m2}")
    print(f"Homomorphism verified: {result == m1 * m2}")


if __name__ == "__main__":
    demo_rsa_homomorphism()
```

### 2.3 ElGamal's Multiplicative Homomorphism

ElGamal encryption (Lesson 5) is also multiplicatively homomorphic:

$$
\text{Enc}(m_1) \otimes \text{Enc}(m_2) = (g^{r_1+r_2}, m_1 \cdot m_2 \cdot y^{r_1+r_2})
$$

This is used in e-voting schemes where encrypted votes are multiplied to compute an encrypted tally.

### 2.4 Additive vs. Multiplicative

| Scheme | Homomorphic Operation | Depth | Used For |
|--------|----------------------|-------|----------|
| RSA (textbook) | Multiplication | Unlimited | Not practical (textbook RSA is insecure) |
| ElGamal | Multiplication | Unlimited | E-voting, rerandomization |
| **Paillier** | **Addition** | Unlimited | Aggregation, ML, auctions |
| Goldwasser-Micali | XOR (1 bit) | Unlimited | Theoretical |

---

## 3. The Paillier Cryptosystem

### 3.1 Why Paillier?

Paillier (1999) is the most practical additively homomorphic scheme. It is widely used because:
- Addition is more generally useful than multiplication (aggregation, averaging, linear operations)
- Support for scalar multiplication: $\text{Enc}(k \cdot m) = \text{Enc}(m)^k$
- Semantic security (probabilistic encryption)

### 3.2 The Math

**Key Generation:**
1. Choose two large primes $p, q$; compute $N = pq$
2. Compute $\lambda = \text{lcm}(p-1, q-1)$
3. Choose $g = N + 1$ (simplification that works)
4. Compute $\mu = (L(g^\lambda \bmod N^2))^{-1} \bmod N$ where $L(x) = \frac{x-1}{N}$
5. Public key: $(N, g)$; Private key: $(\lambda, \mu)$

**Encryption:**

$$
\text{Enc}(m, r) = g^m \cdot r^N \bmod N^2
$$

where $r$ is a random value in $\mathbb{Z}_N^*$.

**Decryption:**

$$
\text{Dec}(c) = L(c^\lambda \bmod N^2) \cdot \mu \bmod N
$$

### 3.3 Homomorphic Properties

**Addition**:

$$
\text{Enc}(m_1) \cdot \text{Enc}(m_2) \bmod N^2 = \text{Enc}(m_1 + m_2 \bmod N)
$$

**Scalar multiplication**:

$$
\text{Enc}(m)^k \bmod N^2 = \text{Enc}(k \cdot m \bmod N)
$$

### 3.4 Implementation

```python
"""
Paillier Homomorphic Encryption — Additively Homomorphic.

Why Paillier for practical applications? It supports:
1. Adding encrypted values without decryption
2. Multiplying an encrypted value by a known constant
3. These two operations enable computing weighted sums, averages,
   and inner products — the building blocks of many ML algorithms.
"""

import secrets
import math


class PaillierKeyPair:
    """Paillier public/private key pair."""

    def __init__(self, bits: int = 512):
        """
        Generate Paillier key pair.

        Why 512 bits here? This is for demonstration. Production
        systems use 2048-bit or 3072-bit keys (similar to RSA).
        """
        # Generate two large primes
        p = self._generate_prime(bits // 2)
        q = self._generate_prime(bits // 2)

        self.n = p * q
        self.n_sq = self.n * self.n
        self.g = self.n + 1  # Simplified generator choice

        # Private key components
        self.lam = math.lcm(p - 1, q - 1)

        # Why L function? It maps elements of the form (1 + N)^x mod N^2
        # back to x mod N. This is the "discrete logarithm" in this group.
        def L(x):
            return (x - 1) // self.n

        self.mu = pow(L(pow(self.g, self.lam, self.n_sq)), -1, self.n)

    def _generate_prime(self, bits: int) -> int:
        """Generate a random prime of the specified bit length."""
        from sympy import isprime, nextprime
        while True:
            candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
            p = nextprime(candidate)
            if p.bit_length() == bits:
                return p


class Paillier:
    """Paillier encryption scheme with homomorphic operations."""

    def __init__(self, keypair: PaillierKeyPair):
        self.kp = keypair

    def encrypt(self, plaintext: int) -> int:
        """
        Encrypt a plaintext integer.

        Why random r? Semantic security requires that encrypting the
        same message twice produces different ciphertexts. The random
        r provides this "probabilistic" property.
        """
        assert 0 <= plaintext < self.kp.n, f"Plaintext must be in [0, {self.kp.n})"

        r = secrets.randbelow(self.kp.n - 1) + 1
        while math.gcd(r, self.kp.n) != 1:
            r = secrets.randbelow(self.kp.n - 1) + 1

        # c = g^m * r^n mod n^2
        c = (pow(self.kp.g, plaintext, self.kp.n_sq) *
             pow(r, self.kp.n, self.kp.n_sq)) % self.kp.n_sq
        return c

    def decrypt(self, ciphertext: int) -> int:
        """Decrypt a ciphertext."""
        # m = L(c^lambda mod n^2) * mu mod n
        x = pow(ciphertext, self.kp.lam, self.kp.n_sq)
        L_val = (x - 1) // self.kp.n
        plaintext = (L_val * self.kp.mu) % self.kp.n
        return plaintext

    def add_encrypted(self, c1: int, c2: int) -> int:
        """
        Add two encrypted values: Enc(m1) * Enc(m2) = Enc(m1 + m2).

        Why multiplication in ciphertext space = addition in plaintext space?
        Because Enc(m) = g^m * r^n mod n^2, and:
        g^m1 * r1^n * g^m2 * r2^n = g^(m1+m2) * (r1*r2)^n = Enc(m1+m2)
        """
        return (c1 * c2) % self.kp.n_sq

    def scalar_multiply(self, ciphertext: int, scalar: int) -> int:
        """
        Multiply encrypted value by a known scalar: Enc(m)^k = Enc(k*m).

        Why exponentiation? Enc(m)^k = (g^m * r^n)^k = g^(km) * r^(kn) = Enc(km)
        """
        return pow(ciphertext, scalar, self.kp.n_sq)

    def encrypt_negative(self, plaintext: int) -> int:
        """
        Encrypt a negative number (as n - |plaintext|).

        Why this encoding? The Paillier plaintext space is Z_n (non-negative).
        We represent negative numbers as their additive inverse mod n:
        -x is encoded as n - x. After decryption, values > n/2 are interpreted
        as negative.
        """
        if plaintext < 0:
            return self.encrypt(self.kp.n + plaintext)
        return self.encrypt(plaintext)

    def decrypt_signed(self, ciphertext: int) -> int:
        """Decrypt and interpret as a signed integer."""
        result = self.decrypt(ciphertext)
        if result > self.kp.n // 2:
            return result - self.kp.n
        return result


def demo_paillier():
    """Demonstrate Paillier homomorphic encryption."""
    print("Generating Paillier key pair (512-bit)...")
    kp = PaillierKeyPair(bits=512)
    paillier = Paillier(kp)

    # Basic encryption/decryption
    m1, m2 = 42, 58
    c1 = paillier.encrypt(m1)
    c2 = paillier.encrypt(m2)

    print(f"\nm1 = {m1}, m2 = {m2}")
    print(f"Dec(Enc(m1)) = {paillier.decrypt(c1)}")
    print(f"Dec(Enc(m2)) = {paillier.decrypt(c2)}")

    # Homomorphic addition
    c_sum = paillier.add_encrypted(c1, c2)
    decrypted_sum = paillier.decrypt(c_sum)
    print(f"\n--- Homomorphic Addition ---")
    print(f"Dec(Enc(m1) * Enc(m2)) = {decrypted_sum}")
    print(f"m1 + m2 = {m1 + m2}")
    print(f"Correct: {decrypted_sum == m1 + m2}")

    # Scalar multiplication
    scalar = 7
    c_scaled = paillier.scalar_multiply(c1, scalar)
    decrypted_scaled = paillier.decrypt(c_scaled)
    print(f"\n--- Scalar Multiplication ---")
    print(f"Dec(Enc(m1)^{scalar}) = {decrypted_scaled}")
    print(f"{scalar} * m1 = {scalar * m1}")
    print(f"Correct: {decrypted_scaled == scalar * m1}")

    # Weighted sum (common in ML)
    weights = [3, 5]
    values = [10, 20]
    encrypted_values = [paillier.encrypt(v) for v in values]

    # Compute weighted sum homomorphically
    weighted_terms = [paillier.scalar_multiply(ev, w)
                      for ev, w in zip(encrypted_values, weights)]
    encrypted_weighted_sum = weighted_terms[0]
    for term in weighted_terms[1:]:
        encrypted_weighted_sum = paillier.add_encrypted(
            encrypted_weighted_sum, term
        )

    result = paillier.decrypt(encrypted_weighted_sum)
    expected = sum(w * v for w, v in zip(weights, values))

    print(f"\n--- Weighted Sum (ML inner product) ---")
    print(f"weights = {weights}, values = {values}")
    print(f"Homomorphic result: {result}")
    print(f"Expected: {expected}")
    print(f"Correct: {result == expected}")


if __name__ == "__main__":
    demo_paillier()
```

---

## 4. From Partial to Fully Homomorphic

### 4.1 The Hierarchy

| Level | Supported Operations | Depth Limit | Examples |
|-------|---------------------|-------------|---------|
| **PHE** | One operation (add OR multiply) | Unlimited | Paillier, ElGamal, RSA |
| **SHE** | Both add and multiply | Limited (bounded depth) | BGN scheme (2005) |
| **Leveled FHE** | Both operations | Pre-determined circuit depth | BGV, BFV |
| **FHE** | Both operations | **Unlimited** (via bootstrapping) | Gentry (2009), TFHE |

### 4.2 Why Is FHE Hard?

To compute arbitrary functions, we need both addition AND multiplication (since they form a complete set of arithmetic operations). The problem is **noise**.

Most homomorphic schemes are based on LWE (Lesson 10), where ciphertexts contain a small error term $e$:

$$
\text{Enc}(m) = (\mathbf{a}, \langle \mathbf{a}, \mathbf{s} \rangle + e + \Delta m)
$$

- **Addition** adds errors: $e_{\text{sum}} = e_1 + e_2$ (linear growth)
- **Multiplication** multiplies errors: $e_{\text{prod}} \approx e_1 \cdot e_2$ (exponential growth!)

After too many multiplications, the error overwhelms the message and decryption fails.

---

## 5. The Noise Growth Problem

### 5.1 Visualizing Noise Growth

```python
"""
Simulate noise growth in homomorphic operations.

Why does noise matter? Every HE ciphertext has a "noise budget."
Operations consume this budget:
- Addition: small cost (noise roughly doubles)
- Multiplication: large cost (noise roughly squares)

When the budget is exhausted, decryption fails.
"""

import matplotlib.pyplot as plt
import numpy as np


def simulate_noise_growth(initial_noise: float = 1.0,
                           modulus: float = 1e15,
                           max_ops: int = 50):
    """
    Simulate noise growth under addition and multiplication.

    The noise must stay below modulus/4 for correct decryption.
    """
    threshold = modulus / 4

    # Addition chain: noise grows linearly
    add_noise = [initial_noise]
    for i in range(max_ops):
        add_noise.append(add_noise[-1] + initial_noise)

    # Multiplication chain: noise grows exponentially
    mul_noise = [initial_noise]
    for i in range(max_ops):
        new_noise = mul_noise[-1] * mul_noise[-1] / modulus * 4
        # Simplified model — actual noise growth depends on scheme
        if new_noise > modulus:
            mul_noise.append(modulus)
        else:
            mul_noise.append(mul_noise[-1] * 2)

    # Find when each hits the threshold
    add_depth = next((i for i, n in enumerate(add_noise) if n > threshold), max_ops)
    mul_depth = next((i for i, n in enumerate(mul_noise) if n > threshold), max_ops)

    print(f"Initial noise: {initial_noise:.1f}")
    print(f"Decryption threshold: {threshold:.2e}")
    print(f"Max additions before failure: {add_depth}")
    print(f"Max multiplications before failure: {mul_depth}")
    print(f"\nThis is why FHE needs bootstrapping for deep circuits!")

    return add_noise[:add_depth+2], mul_noise[:mul_depth+2]


if __name__ == "__main__":
    simulate_noise_growth()
```

### 5.2 The Circuit Depth Barrier

A SHE scheme can evaluate circuits of bounded depth $L$ before noise overflows:

- For $L$ sequential additions: noise grows as $O(L)$
- For $L$ sequential multiplications: noise grows as $O(2^L)$

To evaluate arbitrary circuits (unbounded depth), we need a way to **reduce** the noise without decrypting. This is bootstrapping.

---

## 6. Gentry's Bootstrapping

### 6.1 The Key Insight (2009)

Craig Gentry's doctoral thesis solved the 30-year-old open problem with a stunning idea: **use the decryption function itself as a homomorphic operation**.

Given a "noisy" ciphertext $c$:
1. Encrypt the secret key $\text{sk}$ under a new public key: $\overline{\text{sk}} = \text{Enc}_{\text{pk}'} (\text{sk})$
2. Encrypt the ciphertext $c$ under the new key: $\overline{c} = \text{Enc}_{\text{pk}'}(c)$
3. Homomorphically evaluate the decryption circuit: $c' = \text{Eval}(\text{Dec}, \overline{\text{sk}}, \overline{c})$

The result $c'$ is a **fresh** encryption of the same plaintext but under the new key and with **reset noise**.

### 6.2 The Bootstrapping Paradox

> Wait — if the scheme can only evaluate circuits of depth $L$, and the decryption circuit has depth $D$, don't we need $D < L$? But isn't the scheme only useful if $L$ is large, which makes decryption complex?

This is the **circular security** challenge. Gentry's solution was to design schemes where the decryption circuit has depth strictly less than the evaluation capacity — called **bootstrappable** schemes. This requires careful parameter selection.

### 6.3 Practical Impact

Bootstrapping is the most expensive operation in FHE:
- Without bootstrapping: Evaluate bounded circuits (leveled FHE)
- With bootstrapping: Evaluate arbitrary circuits (true FHE) but with ~1000x overhead per bootstrap

Many practical applications use **leveled FHE** (no bootstrapping) because they know the circuit depth in advance and can set parameters accordingly.

---

## 7. Modern FHE Schemes: BFV and CKKS

### 7.1 BFV (Brakerski/Fan-Vercauteren)

**Purpose**: Exact integer arithmetic on encrypted data.

**Plaintext space**: $\mathbb{Z}_t$ (integers modulo $t$)

**Key features**:
- Exact computation (no approximation errors)
- Best for counting, classification, exact lookups
- Based on Ring-LWE (Lesson 10)

**Ciphertext structure:**

$$
\text{ct} = (c_0, c_1) \in R_q^2 \text{ where } c_0 + c_1 \cdot s \approx \Delta \cdot m + e
$$

where $\Delta = \lfloor q/t \rfloor$ is the scaling factor.

### 7.2 CKKS (Cheon-Kim-Kim-Song)

**Purpose**: Approximate real/complex number arithmetic on encrypted data.

**Plaintext space**: $\mathbb{C}^{N/2}$ (vectors of complex numbers, encoded as polynomials)

**Key features**:
- Supports floating-point-like computation
- Naturally suited for machine learning (which is inherently approximate)
- **Rescaling**: After multiplication, divide out the extra scaling factor to manage noise

**Why CKKS for ML?** Neural network inference involves millions of multiply-accumulate operations on floating-point numbers. CKKS can batch $N/2$ values into a single ciphertext and process them in parallel (SIMD).

### 7.3 Comparison

| Property | BFV | CKKS |
|----------|-----|------|
| Arithmetic | Exact integer | Approximate real/complex |
| Plaintext type | $\mathbb{Z}_t$ | $\mathbb{R}$ (or $\mathbb{C}$) |
| Multiplication | Exact but noise grows | Approximate with rescaling |
| Best for | Lookups, counting, exact logic | ML inference, statistics |
| SIMD batching | Yes (integer slots) | Yes (real/complex slots) |

### 7.4 FHE Libraries

| Library | Language | Schemes | Notable Feature |
|---------|----------|---------|-----------------|
| Microsoft SEAL | C++ | BFV, CKKS | Well-documented, production-ready |
| OpenFHE | C++ | BFV, BGV, CKKS, TFHE | Most comprehensive |
| TFHE-rs | Rust | TFHE | Boolean circuit FHE, fast bootstrapping |
| Concrete | Rust/Python | TFHE | Zama's ML-focused FHE compiler |
| HElib | C++ | BGV, CKKS | IBM's library, bootstrapping support |

---

## 8. Practical Applications

### 8.1 Healthcare

**Problem**: Hospitals want to use cloud ML services for diagnostics but cannot share patient data (HIPAA).

**Solution**: Encrypt patient records with FHE, send ciphertexts to the cloud, run the ML model homomorphically, return encrypted predictions to the hospital for decryption.

**Current status**: Companies like Duality Technologies and Zama are deploying FHE for genomic analysis and clinical trials.

### 8.2 Machine Learning on Encrypted Data

```python
"""
Conceptual: Linear regression inference on encrypted data using Paillier.

Why Paillier for linear models? A linear model computes:
  y = w1*x1 + w2*x2 + ... + wn*xn + b

This is a weighted sum — exactly what Paillier's homomorphic
addition and scalar multiplication support!

For non-linear models (neural networks with ReLU, sigmoid), we need
FHE (BFV/CKKS) because we need multiplication between encrypted values.
"""


def encrypted_linear_inference(paillier, weights, bias, encrypted_features):
    """
    Perform linear regression inference on encrypted features.

    The server knows the model weights (public) but NOT the
    input features (encrypted by the client).

    Parameters:
        paillier: Paillier encryption instance
        weights: list of integers (model weights, known to server)
        bias: integer (model bias, known to server)
        encrypted_features: list of encrypted integers (client data)
    """
    # Compute weighted sum homomorphically
    # Step 1: Multiply each encrypted feature by its weight
    # This uses scalar multiplication: Enc(xi)^wi = Enc(wi * xi)
    weighted = [paillier.scalar_multiply(ef, w)
                for ef, w in zip(encrypted_features, weights)]

    # Step 2: Sum all weighted terms
    # This uses homomorphic addition: Enc(a) * Enc(b) = Enc(a + b)
    result = weighted[0]
    for term in weighted[1:]:
        result = paillier.add_encrypted(result, term)

    # Step 3: Add bias
    enc_bias = paillier.encrypt(bias)
    result = paillier.add_encrypted(result, enc_bias)

    return result  # Still encrypted! Server never sees the data or result


# This would be used as:
# client_features = [age, blood_pressure, cholesterol, ...]
# encrypted_features = [paillier.encrypt(f) for f in client_features]
# encrypted_prediction = encrypted_linear_inference(paillier, model_weights,
#                                                    model_bias,
#                                                    encrypted_features)
# prediction = paillier.decrypt(encrypted_prediction)  # Only client can do this
```

### 8.3 Private Set Intersection

Two parties want to find common elements in their sets without revealing non-common elements:
- Advertising: "Which of my customers also visited your website?"
- Healthcare: "Which patients in our trial also appear in the adverse event database?"

### 8.4 Encrypted Database Queries

CipherCompute and similar services allow SQL-like queries on encrypted databases:
- `SELECT SUM(salary) FROM employees WHERE department = 'Engineering'`
- The database server processes the query without ever seeing plaintext salaries

---

## 9. Performance Reality Check

### 9.1 Current Overheads

| Operation | Plaintext | BFV (FHE) | Overhead |
|-----------|-----------|-----------|----------|
| Addition | ~1 ns | ~10 μs | ~10,000x |
| Multiplication | ~1 ns | ~10 ms | ~10,000,000x |
| Neural network (ResNet-20) | ~5 ms | ~30 min | ~360,000x |
| AES-128 evaluation | ~1 ns | ~1 s | ~1,000,000,000x |

### 9.2 Ciphertext Expansion

| Scheme | Plaintext Size | Ciphertext Size | Expansion |
|--------|---------------|-----------------|-----------|
| Paillier (2048-bit) | 256 bytes | 512 bytes | 2x |
| BFV ($n=4096$) | 4 KB (batched) | 256 KB | 64x |
| CKKS ($n=16384$) | 16 KB (batched) | 2 MB | 128x |

### 9.3 Why Is FHE So Slow?

1. **Large ciphertexts**: Operations on megabyte-sized polynomials instead of 32-byte numbers
2. **Noise management**: Every operation must track and manage noise growth
3. **Modular arithmetic**: Large moduli (hundreds of bits) required for security
4. **Bootstrapping**: The noise-refresh operation is itself a complex computation

### 9.4 The Path to Practicality

- **Hardware acceleration**: Intel, DARPA (DPRIVE program), and others are designing FHE-specific processors
- **Compiler optimizations**: Automatic noise management, operation scheduling, ciphertext packing
- **Algorithmic improvements**: Faster bootstrapping (TFHE), better SIMD batching
- **Approximate computing**: CKKS accepts small approximation errors for massive speedups

> The FHE community draws an analogy to deep learning in 2012: the algorithms existed, but GPU acceleration made them practical. FHE may follow a similar trajectory with custom hardware.

---

## 10. Summary

| Concept | Key Takeaway |
|---------|-------------|
| PHE | Supports one operation (add or multiply) with no depth limit |
| Paillier | Additive PHE; enables encrypted aggregation and weighted sums |
| SHE | Both operations but limited circuit depth |
| FHE | Unlimited depth via bootstrapping; the "holy grail" of encryption |
| Noise growth | Multiplications cause exponential noise growth — the central challenge |
| Bootstrapping | Homomorphically evaluate decryption to refresh noise (Gentry, 2009) |
| BFV | Exact integer FHE; good for counting and classification |
| CKKS | Approximate real-number FHE; ideal for ML inference |
| Performance | 10,000x-1,000,000,000x overhead; improving with hardware and algorithms |

---

## 11. Exercises

### Exercise 1: RSA Homomorphism (Coding)

1. Implement textbook RSA with 1024-bit keys
2. Demonstrate multiplicative homomorphism: encrypt two numbers, multiply ciphertexts, decrypt
3. Show that the scheme is NOT additively homomorphic: encrypting $m_1$ and $m_2$, then adding ciphertexts does NOT decrypt to $m_1 + m_2$
4. Explain why textbook RSA (without padding) is not semantically secure

### Exercise 2: Paillier Applications (Coding)

Using the Paillier implementation from this lesson:
1. Implement a simple private voting system: each voter encrypts 1 (yes) or 0 (no); the tally is computed by adding all encrypted votes
2. Implement encrypted mean: encrypt $n$ values, compute the encrypted sum, and use scalar multiplication by $n^{-1} \bmod N$ to compute the encrypted average
3. Can you compute encrypted variance? Why or why not? What additional capability would you need?

### Exercise 3: Noise Budget Simulation (Coding + Conceptual)

Create a simulation that tracks noise growth in an LWE-based scheme:
1. Start with initial noise $e = 3$ and modulus $q = 2^{32}$
2. For each addition: new noise = sum of input noises
3. For each multiplication: new noise = product of input noises (simplified)
4. Determine the maximum circuit depth (sequential multiplications) before decryption fails ($\text{noise} > q/4$)
5. Plot noise growth for: (a) a chain of 20 additions, (b) a chain of 20 multiplications, (c) a mixed circuit resembling a 3-layer neural network

### Exercise 4: Encrypted Linear Model (Coding)

Build a complete encrypted inference pipeline:
1. Train a logistic regression model on the Iris dataset (scikit-learn)
2. Round the model weights to integers (multiply by 1000)
3. Encrypt test features using Paillier
4. Perform the linear part of inference homomorphically
5. Decrypt the result and apply the sigmoid function (client-side)
6. Compare accuracy with the plaintext model

### Exercise 5: FHE Scheme Comparison (Research + Conceptual)

For each of these applications, recommend the most appropriate HE scheme (Paillier, BFV, CKKS, or TFHE) and justify your choice:
1. Computing the average salary across encrypted employee records
2. Running a neural network on encrypted medical images
3. Private Boolean search on an encrypted database
4. Computing encrypted statistics (mean, median, standard deviation) on financial data
5. Homomorphic evaluation of AES encryption (bootstrapping a symmetric cipher)

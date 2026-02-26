# Lesson 5: The RSA Cryptosystem

**Previous**: [Hash Functions](./04_Hash_Functions.md) | **Next**: [Elliptic Curve Cryptography](./06_Elliptic_Curve_Cryptography.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Generate RSA key pairs and explain the role of each parameter ($p$, $q$, $n$, $e$, $d$)
2. Prove mathematically why RSA encryption and decryption are inverse operations
3. Implement RSA encryption, decryption, signing, and verification from scratch
4. Describe textbook RSA attacks (small exponent, common modulus, Wiener's attack) and explain why padding is essential
5. Compare PKCS#1 v1.5 padding with OAEP and explain the Bleichenbacher attack
6. Evaluate RSA key size recommendations and the impact of quantum computing

---

RSA, published in 1977 by Rivest, Shamir, and Adleman, was the first practical public-key cryptosystem and remains one of the most widely deployed algorithms in the world. Every TLS connection, code signing certificate, and SSH key exchange can involve RSA. Its elegance lies in a profound mathematical insight: multiplying two large primes is easy, but factoring their product is extraordinarily hard. This lesson builds RSA from the number theory foundations of Lesson 1 and reveals both its mathematical beauty and its practical pitfalls.

> **Analogy:** RSA is like a mailbox — anyone can drop a letter in (encrypt with the public key), but only the owner with the physical key can open it (decrypt with the private key). The "hardness" of RSA is that building the mailbox (multiplying primes) is trivial, but breaking into it (factoring the product) is nearly impossible.

## Table of Contents

1. [Key Generation](#1-key-generation)
2. [Encryption and Decryption](#2-encryption-and-decryption)
3. [Proof of Correctness](#3-proof-of-correctness)
4. [RSA Signatures](#4-rsa-signatures)
5. [Textbook RSA Attacks](#5-textbook-rsa-attacks)
6. [Padding: PKCS#1 and OAEP](#6-padding-pkcs1-and-oaep)
7. [The Bleichenbacher Attack](#7-the-bleichenbacher-attack)
8. [Key Size and Performance](#8-key-size-and-performance)
9. [Complete RSA Implementation](#9-complete-rsa-implementation)
10. [Exercises](#10-exercises)

---

## 1. Key Generation

### 1.1 Algorithm

1. Choose two large, distinct primes $p$ and $q$ (typically 1024 bits each for RSA-2048)
2. Compute $n = p \cdot q$ (the **modulus**)
3. Compute $\phi(n) = (p-1)(q-1)$ (Euler's totient, or more precisely, $\lambda(n) = \text{lcm}(p-1, q-1)$)
4. Choose $e$ such that $1 < e < \phi(n)$ and $\gcd(e, \phi(n)) = 1$ (the **public exponent**)
5. Compute $d = e^{-1} \bmod \phi(n)$ (the **private exponent**)

**Public key:** $(n, e)$
**Private key:** $(n, d)$ (or equivalently, $p$, $q$, $d$, and CRT parameters)

### 1.2 Parameter Choices

| Parameter | Typical Value | Rationale |
|-----------|---------------|-----------|
| $e$ | 65537 ($2^{16} + 1$) | Small Hamming weight → fast encryption; large enough to avoid small-$e$ attacks |
| $p, q$ | 1024+ bits each | Factoring $n$ must be infeasible |
| $\|p - q\|$ | Large | Prevents Fermat factorization (which exploits $p \approx q$) |

```python
import random

def generate_prime(bits, k=20):
    """Generate a random prime of the specified bit length.

    Uses Miller-Rabin primality test from Lesson 1.
    """
    def miller_rabin(n, k):
        if n < 2: return False
        if n == 2 or n == 3: return True
        if n % 2 == 0: return False
        s, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else:
                return False
        return True

    while True:
        n = random.getrandbits(bits)
        n |= (1 << (bits - 1)) | 1
        if miller_rabin(n, k):
            return n

def rsa_keygen(bits=2048):
    """Generate an RSA key pair.

    Why e = 65537:
    - Binary: 10000000000000001 (only 2 bits set)
    - Fast modular exponentiation: only 17 squarings + 1 multiply
    - Large enough to prevent small-exponent attacks
    - Standard choice across all major implementations

    Why phi(n) vs lambda(n):
    - Textbooks use phi(n) = (p-1)(q-1)
    - PKCS#1 recommends lambda(n) = lcm(p-1, q-1)
    - Both work; lambda(n) gives a smaller d (slightly faster)
    """
    from math import gcd

    half_bits = bits // 2

    while True:
        p = generate_prime(half_bits)
        q = generate_prime(half_bits)

        # Ensure p != q and |p - q| is large (prevents Fermat factoring)
        if p == q:
            continue
        if abs(p - q).bit_length() < half_bits - 10:
            continue  # p and q too close together

        n = p * q
        phi_n = (p - 1) * (q - 1)

        e = 65537
        if gcd(e, phi_n) != 1:
            continue  # Extremely rare, but check anyway

        d = pow(e, -1, phi_n)  # Modular inverse (Python 3.8+)

        public_key = (n, e)
        private_key = (n, d, p, q)
        return public_key, private_key

# Generate a small key pair for demonstration
pub, priv = rsa_keygen(512)  # 512-bit for speed; use 2048+ in production
n, e = pub
_, d, p, q = priv

print(f"p = {p}")
print(f"q = {q}")
print(f"n = {n}")
print(f"e = {e}")
print(f"d = {d}")
print(f"n bit length: {n.bit_length()}")
```

### 1.3 Why Keep $p$ and $q$ Secret?

Knowing $p$ and $q$ allows computing $\phi(n) = (p-1)(q-1)$, which immediately reveals $d = e^{-1} \bmod \phi(n)$. Equivalently, knowing $\phi(n)$ and $n$ allows factoring $n$ (since $p + q = n - \phi(n) + 1$ and $p \cdot q = n$, giving a quadratic equation).

---

## 2. Encryption and Decryption

### 2.1 Textbook RSA

**Encryption:** Given public key $(n, e)$ and plaintext $m \in \{0, 1, \ldots, n-1\}$:

$$c = m^e \bmod n$$

**Decryption:** Given private key $(n, d)$ and ciphertext $c$:

$$m = c^d \bmod n$$

```python
def rsa_encrypt(m, public_key):
    """Textbook RSA encryption.

    Why modular exponentiation: Python's pow(m, e, n) uses the
    square-and-multiply algorithm (Lesson 1, Section 7), making
    this efficient even for 2048-bit numbers.
    """
    n, e = public_key
    if m >= n:
        raise ValueError("Message must be less than modulus n")
    return pow(m, e, n)

def rsa_decrypt(c, private_key):
    """Textbook RSA decryption."""
    n, d = private_key[0], private_key[1]
    return pow(c, d, n)

# Encrypt and decrypt a number
message = 42
ciphertext = rsa_encrypt(message, pub)
decrypted = rsa_decrypt(ciphertext, priv)

print(f"Message:    {message}")
print(f"Ciphertext: {ciphertext}")
print(f"Decrypted:  {decrypted}")
print(f"Correct:    {message == decrypted}")
```

### 2.2 CRT Optimization

RSA decryption can be accelerated ~4x using the Chinese Remainder Theorem (Lesson 1, Section 6):

$$m_p = c^{d \bmod (p-1)} \bmod p$$
$$m_q = c^{d \bmod (q-1)} \bmod q$$
$$m = \text{CRT}(m_p, m_q, p, q)$$

```python
def rsa_decrypt_crt(c, private_key):
    """RSA decryption with CRT optimization.

    Why CRT is ~4x faster:
    - Two exponentiations with half-size exponents and moduli
    - Exponentiation cost is O(k^3) where k is bit length
    - Half the bits → (1/2)^3 = 1/8 the cost per exponentiation
    - Two exponentiations: 2 * (1/8) = 1/4 the total cost
    """
    n, d, p, q = private_key

    # Precompute (these can be stored with the private key)
    dp = d % (p - 1)
    dq = d % (q - 1)
    q_inv = pow(q, -1, p)

    # Decrypt modulo p and q separately
    m_p = pow(c, dp, p)
    m_q = pow(c, dq, q)

    # Combine using Garner's formula (efficient CRT for 2 moduli)
    h = (q_inv * (m_p - m_q)) % p
    m = m_q + h * q

    return m

# Verify CRT decryption matches standard decryption
ciphertext = rsa_encrypt(42, pub)
m1 = rsa_decrypt(ciphertext, priv)
m2 = rsa_decrypt_crt(ciphertext, priv)
print(f"Standard: {m1}, CRT: {m2}, Match: {m1 == m2}")
```

---

## 3. Proof of Correctness

### 3.1 Why Does $m^{ed} \equiv m \pmod{n}$?

We need to show that decrypting an encrypted message recovers the original.

**Given:** $ed \equiv 1 \pmod{\phi(n)}$, so $ed = 1 + k\phi(n)$ for some integer $k$.

**Case 1: $\gcd(m, n) = 1$**

By Euler's theorem (Lesson 1, Section 5):

$$m^{\phi(n)} \equiv 1 \pmod{n}$$

Therefore:

$$m^{ed} = m^{1 + k\phi(n)} = m \cdot (m^{\phi(n)})^k \equiv m \cdot 1^k = m \pmod{n}$$

**Case 2: $\gcd(m, n) \neq 1$**

Since $n = pq$, if $\gcd(m, n) \neq 1$, then $p \mid m$ or $q \mid m$ (but not both, since $m < n$).

Suppose $p \mid m$ (so $m \equiv 0 \pmod{p}$):
- Then $m^{ed} \equiv 0 \equiv m \pmod{p}$
- For $\pmod{q}$: since $\gcd(m, q) = 1$, by Fermat: $m^{q-1} \equiv 1 \pmod{q}$, so $m^{ed} \equiv m \pmod{q}$

By CRT, $m^{ed} \equiv m \pmod{n}$. $\blacksquare$

### 3.2 Using Carmichael's Function

The proof also works with $\lambda(n) = \text{lcm}(p-1, q-1)$ instead of $\phi(n)$. Since $\phi(n)$ is always a multiple of $\lambda(n)$, if $ed \equiv 1 \pmod{\lambda(n)}$, correctness follows by the same argument.

---

## 4. RSA Signatures

### 4.1 Hash-Then-Sign

RSA signatures use the private key to "encrypt" a message hash:

$$\text{Sign: } \sigma = H(m)^d \bmod n$$
$$\text{Verify: } H(m) \stackrel{?}{=} \sigma^e \bmod n$$

```python
import hashlib

def rsa_sign(message, private_key):
    """RSA signature: hash the message, then 'decrypt' the hash.

    Why hash first: signing the raw message would be slow (message
    could be gigabytes) and insecure (textbook RSA has algebraic
    structure that enables forgery — see Section 5).
    """
    n, d = private_key[0], private_key[1]
    h = int(hashlib.sha256(message).hexdigest(), 16) % n
    signature = pow(h, d, n)
    return signature

def rsa_verify(message, signature, public_key):
    """RSA signature verification: 'encrypt' the signature and compare."""
    n, e = public_key
    h = int(hashlib.sha256(message).hexdigest(), 16) % n
    recovered = pow(signature, e, n)
    return h == recovered

# Sign and verify
message = b"This is an important document."
sig = rsa_sign(message, priv)
valid = rsa_verify(message, sig, pub)
print(f"Signature valid: {valid}")

# Tampered message fails verification
tampered = b"This is an important document!"
valid_tampered = rsa_verify(tampered, sig, pub)
print(f"Tampered signature valid: {valid_tampered}")
```

### 4.2 Existential Forgery of Textbook RSA Signatures

Without hashing, an attacker can forge signatures:

**Attack 1: Choosing the signature first.** Pick any $\sigma$, compute $m = \sigma^e \bmod n$. Then $(m, \sigma)$ is a valid message-signature pair (though $m$ is random garbage).

**Attack 2: Multiplicative property.** Given signatures $\sigma_1$ on $m_1$ and $\sigma_2$ on $m_2$:

$$\sigma_1 \cdot \sigma_2 = (m_1^d)(m_2^d) = (m_1 m_2)^d \bmod n$$

This is a valid signature on $m_1 \cdot m_2 \bmod n$. Hashing prevents this because $H(m_1 \cdot m_2) \neq H(m_1) \cdot H(m_2)$ (hash functions are not multiplicative).

---

## 5. Textbook RSA Attacks

Textbook RSA ($c = m^e \bmod n$ with no padding) is vulnerable to several attacks:

### 5.1 Small Public Exponent Attack

If $e = 3$ and $m^3 < n$ (message is small), then $c = m^3$ over the integers (no modular reduction). The attacker simply computes $m = \sqrt[3]{c}$.

```python
def small_exponent_attack():
    """Demonstrate the small-e attack on textbook RSA.

    Why e=3 is dangerous without padding: if the message is small
    enough that m^3 < n, the modular reduction never happens, and
    the attacker can compute the cube root over the integers.
    """
    # Generate RSA with e=3 (for demonstration)
    from math import gcd, isqrt

    # Small example
    p, q = 61, 53
    n = p * q  # 3233
    e = 3
    phi_n = (p-1) * (q-1)  # 3120
    # Verify gcd(3, 3120) = 3 — actually, this doesn't work for e=3 with these primes
    # Let's use primes where it works
    p, q = 59, 71
    n = p * q  # 4189
    phi_n = (p-1) * (q-1)  # 4060
    assert gcd(3, phi_n) == 1
    d = pow(3, -1, phi_n)

    # Small message: m^3 < n
    m = 15
    c = pow(m, 3, n)
    print(f"n = {n}, e = 3, m = {m}, c = {c}")
    print(f"m^3 = {m**3}, n = {n}, m^3 < n: {m**3 < n}")

    if m**3 < n:
        # Attacker computes integer cube root
        recovered = round(c ** (1/3))
        # More precise method for large numbers:
        def integer_cube_root(x):
            if x < 0: return -integer_cube_root(-x)
            if x == 0: return 0
            guess = int(x ** (1/3)) + 1
            while guess ** 3 > x: guess -= 1
            return guess

        recovered = integer_cube_root(c)
        print(f"Recovered m = {recovered} (correct: {recovered == m})")

small_exponent_attack()
```

### 5.2 Common Modulus Attack

If the same message is encrypted with the same modulus but different public exponents $e_1, e_2$ where $\gcd(e_1, e_2) = 1$:

$$c_1 = m^{e_1} \bmod n, \quad c_2 = m^{e_2} \bmod n$$

Find $a, b$ such that $ae_1 + be_2 = 1$ (Extended GCD), then:

$$c_1^a \cdot c_2^b = m^{ae_1 + be_2} = m^1 = m \pmod{n}$$

### 5.3 Hastad's Broadcast Attack

If the same message $m$ is encrypted with $e$ different public keys (all with $e = 3$):

$$c_1 = m^3 \bmod n_1, \quad c_2 = m^3 \bmod n_2, \quad c_3 = m^3 \bmod n_3$$

Use CRT to find $c' = m^3 \bmod (n_1 n_2 n_3)$. Since $m < \min(n_i)$, we have $m^3 < n_1 n_2 n_3$, so $c' = m^3$ over the integers, and $m = \sqrt[3]{c'}$.

### 5.4 Wiener's Attack

If $d < \frac{1}{3} n^{1/4}$, the continued fraction expansion of $e/n$ reveals $d$.

```python
def wieners_attack_demo():
    """Demonstrate Wiener's attack on RSA with small d.

    Why small d is dangerous: the key equation ed ≡ 1 (mod phi(n))
    can be written as ed = 1 + k*phi(n), giving e/n ≈ k/d.
    The continued fraction expansion of e/n reveals k/d when
    d is small (d < n^(1/4) / 3).
    """
    def continued_fraction(numerator, denominator):
        """Compute the continued fraction expansion of num/den."""
        cf = []
        while denominator:
            q = numerator // denominator
            cf.append(q)
            numerator, denominator = denominator, numerator - q * denominator
        return cf

    def convergents(cf):
        """Compute convergents from a continued fraction."""
        convs = []
        h_prev, h_curr = 0, 1
        k_prev, k_curr = 1, 0
        for a in cf:
            h_prev, h_curr = h_curr, a * h_curr + h_prev
            k_prev, k_curr = k_curr, a * k_curr + k_prev
            convs.append((h_curr, k_curr))
        return convs

    # Example with small d (vulnerable to Wiener's attack)
    p, q = 1009, 3643
    n = p * q  # 3675787
    phi_n = (p - 1) * (q - 1)
    d = 29  # Deliberately small
    e = pow(d, -1, phi_n)

    print(f"n = {n}, e = {e}, d = {d}")
    print(f"d < n^(1/4)/3 = {n**(1/4)/3:.1f}: {d < n**(1/4)/3}")

    # Attack: compute continued fraction of e/n
    cf = continued_fraction(e, n)
    convs = convergents(cf)

    for k, d_guess in convs:
        if k == 0:
            continue
        # Check if d_guess could be the private exponent
        if (e * d_guess - 1) % k == 0:
            phi_guess = (e * d_guess - 1) // k
            # Verify: n - phi + 1 should be p + q, and p*q should be n
            s = n - phi_guess + 1  # p + q
            # p and q are roots of x^2 - s*x + n = 0
            discriminant = s * s - 4 * n
            if discriminant >= 0:
                from math import isqrt
                sqrt_disc = isqrt(discriminant)
                if sqrt_disc * sqrt_disc == discriminant:
                    print(f"Recovered d = {d_guess} (correct: {d_guess == d})")
                    p_recovered = (s + sqrt_disc) // 2
                    q_recovered = (s - sqrt_disc) // 2
                    print(f"Recovered p = {p_recovered}, q = {q_recovered}")
                    break

wieners_attack_demo()
```

---

## 6. Padding: PKCS#1 and OAEP

### 6.1 Why Padding Is Essential

Textbook RSA is deterministic ($E(m) = m^e \bmod n$), which means:
- Identical messages produce identical ciphertexts (no semantic security)
- The multiplicative property enables chosen-ciphertext attacks
- Small messages are vulnerable to root extraction

Padding adds randomness and structure to prevent these attacks.

### 6.2 PKCS#1 v1.5

The oldest widely-deployed RSA padding scheme (still used in many systems):

```
0x00 || 0x02 || [random non-zero bytes] || 0x00 || [message]
```

- The `0x02` byte indicates encryption (signing uses `0x01`)
- At least 8 random non-zero padding bytes
- The `0x00` separator marks the start of the actual message

**Problem:** Vulnerable to the Bleichenbacher attack (Section 7).

### 6.3 OAEP (Optimal Asymmetric Encryption Padding)

OAEP (Bellare-Rogaway, 1994; standardized in PKCS#1 v2.2) provides **CCA security** (security against chosen-ciphertext attacks):

```
         ┌──── message ────┐
         │ m || 00...01    │
         │   (padded)      │
seed ──→ [G]  ──XOR──→ maskedDB
  │                         │
  └────────── [H] ←────────┘
         maskedSeed

OAEP output = 0x00 || maskedSeed || maskedDB
```

Where $G$ and $H$ are mask generation functions (typically based on SHA-256).

```python
def oaep_concept():
    """Explain OAEP's structure and security properties.

    Why OAEP over PKCS#1 v1.5:
    1. Provably CCA-secure (in the random oracle model)
    2. Randomized: same message → different ciphertexts each time
    3. All-or-nothing transform: partial information about the
       plaintext cannot be extracted from the ciphertext
    4. Not vulnerable to Bleichenbacher's attack
    """
    import hashlib
    import os

    def mgf1(seed, length, hash_func=hashlib.sha256):
        """Mask Generation Function 1 (MGF1) per PKCS#1.

        Why MGF1: it stretches a short seed into an arbitrarily long
        mask, similar to a key derivation function. Using a hash
        ensures the mask appears random.
        """
        mask = b''
        counter = 0
        while len(mask) < length:
            c = counter.to_bytes(4, 'big')
            mask += hash_func(seed + c).digest()
            counter += 1
        return mask[:length]

    # OAEP encoding (simplified)
    k = 64  # RSA modulus size in bytes (512-bit for demo)
    h_len = 32  # SHA-256 output length
    message = b"Hello, OAEP!"
    max_msg_len = k - 2 * h_len - 2

    if len(message) > max_msg_len:
        raise ValueError(f"Message too long (max {max_msg_len} bytes)")

    # Create data block: lHash || PS || 0x01 || message
    l_hash = hashlib.sha256(b'').digest()  # Hash of empty label
    ps = b'\x00' * (max_msg_len - len(message))  # Padding
    db = l_hash + ps + b'\x01' + message

    # Generate random seed
    seed = os.urandom(h_len)

    # Mask db with seed
    db_mask = mgf1(seed, len(db))
    masked_db = bytes(a ^ b for a, b in zip(db, db_mask))

    # Mask seed with masked_db
    seed_mask = mgf1(masked_db, h_len)
    masked_seed = bytes(a ^ b for a, b in zip(seed, seed_mask))

    # OAEP encoded message
    em = b'\x00' + masked_seed + masked_db

    print(f"Message: {message}")
    print(f"OAEP encoded ({len(em)} bytes): {em[:20].hex()}...")
    print(f"Encoded same message again (different randomness):")

    seed2 = os.urandom(h_len)
    db_mask2 = mgf1(seed2, len(db))
    masked_db2 = bytes(a ^ b for a, b in zip(db, db_mask2))
    seed_mask2 = mgf1(masked_db2, h_len)
    masked_seed2 = bytes(a ^ b for a, b in zip(seed2, seed_mask2))
    em2 = b'\x00' + masked_seed2 + masked_db2

    print(f"  {em2[:20].hex()}...")
    print(f"Different encodings: {em != em2}")

oaep_concept()
```

---

## 7. The Bleichenbacher Attack

### 7.1 Overview

In 1998, Daniel Bleichenbacher showed that PKCS#1 v1.5 is vulnerable to an **adaptive chosen-ciphertext attack**. If a server reveals whether the decrypted ciphertext has valid PKCS#1 v1.5 formatting (even through side channels like different error messages or timing), an attacker can decrypt any ciphertext.

### 7.2 How It Works

1. The attacker has a ciphertext $c$ they want to decrypt
2. They choose multipliers $s$ and send $c' = c \cdot s^e \bmod n$ to the server
3. Because RSA is multiplicative: $\text{Dec}(c') = m \cdot s \bmod n$
4. The server checks if $m \cdot s$ has valid PKCS#1 format (starts with `0x00 0x02`)
5. Valid/invalid responses progressively narrow down the possible values of $m$
6. After ~1 million queries, the attacker recovers $m$ completely

### 7.3 Impact and Mitigation

- **DROWN attack (2016):** Exploited SSLv2 servers to break TLS connections
- **ROBOT attack (2017):** Found many TLS implementations still vulnerable
- **Mitigation:** Use OAEP, or implement constant-time PKCS#1 v1.5 processing (extremely difficult)

```python
def bleichenbacher_concept():
    """Explain the Bleichenbacher attack strategy.

    Why this attack is devastating:
    - Does NOT require finding factors of n
    - Does NOT require any weakness in AES or the hash function
    - Only requires a padding oracle (valid/invalid distinction)
    - ~1M queries to decrypt a message (trivially fast)

    This attack is why PKCS#1 v1.5 encryption is deprecated and
    OAEP is mandatory for new applications.
    """
    print("Bleichenbacher's Attack (1998):")
    print("=" * 50)
    print()
    print("1. Attacker captures ciphertext c = m^e mod n")
    print("2. Attacker sends c' = c * s^e mod n (for various s)")
    print("3. Server decrypts: m' = m * s mod n")
    print("4. Server checks PKCS#1 format: does m' start with 0x00 0x02?")
    print("5. Server response (YES/NO) narrows the range of m")
    print()
    print("After ~1 million queries:")
    print("  - Range of m narrows to a single value")
    print("  - Attacker has fully decrypted c without the private key")
    print()
    print("Defenses:")
    print("  - Use OAEP (not PKCS#1 v1.5) for encryption")
    print("  - Constant-time processing (no timing side channels)")
    print("  - TLS 1.3 removed RSA key exchange entirely")

bleichenbacher_concept()
```

---

## 8. Key Size and Performance

### 8.1 RSA Key Size Recommendations

| Security level (bits) | RSA key size | Equivalent AES | Status |
|-----------------------|-------------|----------------|--------|
| 80 | 1024 | - | **Deprecated** |
| 112 | 2048 | AES-128 | Minimum acceptable |
| 128 | 3072 | AES-128 | Recommended |
| 192 | 7680 | AES-192 | Conservative |
| 256 | 15360 | AES-256 | Post-2030+ |

### 8.2 RSA vs ECC Performance

RSA keys grow much faster than ECC keys for equivalent security:

| Security | RSA key | ECC key | Ratio |
|----------|---------|---------|-------|
| 128-bit | 3072 bits | 256 bits | 12:1 |
| 192-bit | 7680 bits | 384 bits | 20:1 |
| 256-bit | 15360 bits | 521 bits | 30:1 |

RSA decryption/signing is much slower than ECC at equivalent security levels. This is a major reason why modern systems (TLS 1.3, SSH) increasingly prefer ECDSA/Ed25519 over RSA (Lesson 6 and 7).

### 8.3 Quantum Threat

Shor's algorithm on a quantum computer can factor $n$ in polynomial time, breaking RSA completely. A cryptographically-relevant quantum computer does not exist today, but the threat motivates:

1. **Harvest now, decrypt later:** Adversaries may record RSA-encrypted data today and decrypt it when quantum computers become available
2. **Post-quantum migration:** NIST has standardized lattice-based algorithms (Lesson 10) to replace RSA

---

## 9. Complete RSA Implementation

```python
import hashlib
import os
import random

class RSA:
    """Complete RSA implementation for educational purposes.

    WARNING: This is for learning only. In production, use a
    well-audited library like `cryptography` (Python) or OpenSSL.

    Why not roll your own crypto:
    - Side-channel attacks (timing, power analysis)
    - Subtle mathematical attacks (Bleichenbacher, Manger)
    - Random number generation vulnerabilities
    - Padding implementation bugs
    """

    def __init__(self, bits=2048):
        self.bits = bits
        self.public_key, self.private_key = self._keygen()

    def _keygen(self):
        from math import gcd
        half = self.bits // 2

        while True:
            p = self._generate_prime(half)
            q = self._generate_prime(half)
            if p == q or abs(p - q).bit_length() < half - 10:
                continue

            n = p * q
            phi = (p - 1) * (q - 1)
            e = 65537

            if gcd(e, phi) != 1:
                continue

            d = pow(e, -1, phi)
            return (n, e), (n, d, p, q)

    def _generate_prime(self, bits):
        while True:
            n = random.getrandbits(bits)
            n |= (1 << (bits - 1)) | 1
            if self._miller_rabin(n):
                return n

    def _miller_rabin(self, n, k=20):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0: return False
        s, d = 0, n - 1
        while d % 2 == 0:
            d //= 2; s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else: return False
        return True

    def encrypt(self, plaintext_bytes):
        """Encrypt bytes using RSA with simple PKCS#1-like padding."""
        n, e = self.public_key
        k = (n.bit_length() + 7) // 8  # Modulus size in bytes

        # Simple padding: 0x00 || 0x02 || random || 0x00 || message
        max_msg = k - 11  # 8 bytes minimum random padding
        if len(plaintext_bytes) > max_msg:
            raise ValueError(f"Message too long (max {max_msg} bytes)")

        ps = bytes(random.randint(1, 255) for _ in range(k - len(plaintext_bytes) - 3))
        em = b'\x00\x02' + ps + b'\x00' + plaintext_bytes

        m = int.from_bytes(em, 'big')
        c = pow(m, e, n)
        return c.to_bytes(k, 'big')

    def decrypt(self, ciphertext_bytes):
        """Decrypt bytes using RSA-CRT with padding removal."""
        n, d, p, q = self.private_key
        c = int.from_bytes(ciphertext_bytes, 'big')

        # CRT decryption
        dp, dq = d % (p - 1), d % (q - 1)
        q_inv = pow(q, -1, p)
        m_p, m_q = pow(c, dp, p), pow(c, dq, q)
        h = (q_inv * (m_p - m_q)) % p
        m = m_q + h * q

        k = (n.bit_length() + 7) // 8
        em = m.to_bytes(k, 'big')

        # Remove padding
        if em[0:2] != b'\x00\x02':
            raise ValueError("Decryption error")
        sep = em.index(b'\x00', 2)
        return em[sep + 1:]

    def sign(self, message):
        """Sign a message using RSA (hash-then-sign)."""
        n, d = self.private_key[0], self.private_key[1]
        h = int(hashlib.sha256(message).hexdigest(), 16)
        # Ensure hash < n
        h = h % n
        sig = pow(h, d, n)
        k = (n.bit_length() + 7) // 8
        return sig.to_bytes(k, 'big')

    def verify(self, message, signature):
        """Verify an RSA signature."""
        n, e = self.public_key
        sig_int = int.from_bytes(signature, 'big')
        recovered = pow(sig_int, e, n)
        h = int(hashlib.sha256(message).hexdigest(), 16) % n
        return recovered == h

# Full demonstration
rsa = RSA(bits=1024)  # 1024-bit for speed in demo

# Encryption / Decryption
message = b"RSA encryption works!"
ct = rsa.encrypt(message)
pt = rsa.decrypt(ct)
print(f"Encrypt/Decrypt: {pt.decode()}")

# Signing / Verification
sig = rsa.sign(b"Sign this document")
valid = rsa.verify(b"Sign this document", sig)
print(f"Signature valid: {valid}")

tampered = rsa.verify(b"Sign this ducument", sig)  # typo in 'document'
print(f"Tampered valid:  {tampered}")
```

---

## 10. Exercises

### Exercise 1: RSA by Hand (Basic)

Using $p = 61$, $q = 53$, $e = 17$:
1. Compute $n$, $\phi(n)$, and $d$
2. Encrypt $m = 65$. What is the ciphertext $c$?
3. Decrypt $c$. Verify you recover $m = 65$.
4. Sign $m = 42$. Verify the signature.

### Exercise 2: RSA Key Generation Timing (Intermediate)

1. Time RSA key generation for 512, 1024, 2048, and 4096 bits.
2. For each key size, time encryption and decryption of a 32-byte message.
3. Plot key generation time and decryption time vs. key size.
4. Extrapolate: how long would 8192-bit RSA decryption take?

### Exercise 3: Common Modulus Attack (Intermediate)

Implement the common modulus attack:
1. Generate one RSA modulus $n = pq$
2. Create two public keys: $(n, e_1 = 17)$ and $(n, e_2 = 65537)$ where $\gcd(e_1, e_2) = 1$
3. Encrypt the same message $m$ with both keys
4. Using only $c_1$, $c_2$, $e_1$, $e_2$, and $n$ (but NOT $d_1$ or $d_2$), recover $m$

### Exercise 4: Wiener's Attack (Challenging)

1. Generate an RSA key pair where $d < n^{1/4} / 3$ (choose small $d$ first, then compute $e$)
2. Implement Wiener's attack using continued fractions
3. Verify that your attack recovers $d$ from only $(n, e)$
4. Find the threshold: what is the largest $d$ your attack can recover?

### Exercise 5: RSA-CRT Fault Attack (Challenging)

Research and explain the Bellcore attack on RSA-CRT:
1. If a single fault occurs during CRT computation (e.g., $m_p$ is wrong but $m_q$ is correct), show how an attacker can factor $n$
2. Implement a simulation: inject a fault into CRT decryption and recover $p$ and $q$
3. Explain the countermeasure: verify $m^e \equiv c \pmod{n}$ after CRT decryption

---

**Previous**: [Hash Functions](./04_Hash_Functions.md) | **Next**: [Elliptic Curve Cryptography](./06_Elliptic_Curve_Cryptography.md)

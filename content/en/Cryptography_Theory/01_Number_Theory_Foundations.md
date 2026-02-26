# Lesson 1: Number Theory Foundations

**Next**: [Symmetric Ciphers](./02_Symmetric_Ciphers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Perform modular arithmetic operations and reason about congruence classes
2. Implement the Extended Euclidean Algorithm and compute modular inverses
3. Apply Euler's theorem and Fermat's little theorem to simplify modular exponentiation
4. State and apply the Chinese Remainder Theorem
5. Implement the Miller-Rabin primality test and explain its probabilistic guarantees
6. Explain why the discrete logarithm problem is computationally hard

---

Modern cryptography rests on a surprisingly small set of number-theoretic results. Every time you connect to a website over HTTPS, your browser performs modular exponentiation, checks prime numbers, and computes modular inverses — all within milliseconds. This lesson builds the mathematical toolkit that powers RSA (Lesson 5), Diffie-Hellman key exchange, and elliptic curve cryptography (Lesson 6). Master these foundations, and the rest of cryptography becomes a series of elegant applications.

## Table of Contents

1. [Modular Arithmetic](#1-modular-arithmetic)
2. [GCD and the Extended Euclidean Algorithm](#2-gcd-and-the-extended-euclidean-algorithm)
3. [Modular Inverse](#3-modular-inverse)
4. [Euler's Totient Function](#4-eulers-totient-function)
5. [Euler's Theorem and Fermat's Little Theorem](#5-eulers-theorem-and-fermats-little-theorem)
6. [Chinese Remainder Theorem](#6-chinese-remainder-theorem)
7. [Fast Modular Exponentiation](#7-fast-modular-exponentiation)
8. [Primality Testing](#8-primality-testing)
9. [The Discrete Logarithm Problem](#9-the-discrete-logarithm-problem)
10. [Exercises](#10-exercises)

---

## 1. Modular Arithmetic

> **Analogy:** Modular arithmetic is like a clock — after 12 comes 1 again, not 13. If it is 10 o'clock and you wait 5 hours, the clock shows 3, not 15. We say $15 \equiv 3 \pmod{12}$.

### 1.1 Definition

For integers $a$, $b$, and a positive integer $n$, we say $a$ is **congruent** to $b$ modulo $n$, written:

$$a \equiv b \pmod{n}$$

if $n$ divides $(a - b)$, i.e., $n \mid (a - b)$.

Equivalently, $a$ and $b$ leave the same remainder when divided by $n$.

### 1.2 Properties

Modular arithmetic preserves the basic operations:

| Property | Statement |
|----------|-----------|
| Addition | If $a \equiv b$ and $c \equiv d \pmod{n}$, then $a + c \equiv b + d \pmod{n}$ |
| Subtraction | If $a \equiv b$ and $c \equiv d \pmod{n}$, then $a - c \equiv b - d \pmod{n}$ |
| Multiplication | If $a \equiv b$ and $c \equiv d \pmod{n}$, then $ac \equiv bd \pmod{n}$ |
| Exponentiation | If $a \equiv b \pmod{n}$, then $a^k \equiv b^k \pmod{n}$ for $k \geq 0$ |

**Division does NOT always work.** For example, $6 \equiv 0 \pmod{6}$ and $3 \equiv 3 \pmod{6}$, but $6/3 = 2 \not\equiv 0/3 = 0 \pmod{6}$. Division requires a **modular inverse**, which we address in Section 3.

### 1.3 The Set $\mathbb{Z}_n$

The **residue classes** modulo $n$ form the set:

$$\mathbb{Z}_n = \{0, 1, 2, \ldots, n-1\}$$

This is a **ring** under addition and multiplication modulo $n$. When we restrict to elements with multiplicative inverses, we get:

$$\mathbb{Z}_n^* = \{a \in \mathbb{Z}_n \mid \gcd(a, n) = 1\}$$

This is a **multiplicative group**. Its order is $|\mathbb{Z}_n^*| = \phi(n)$ (Euler's totient function, Section 4).

```python
def residue_classes(n):
    """Demonstrate Z_n and Z_n^* for a given modulus."""
    from math import gcd

    z_n = list(range(n))
    # Why filter by gcd == 1: only elements coprime to n have multiplicative inverses
    z_n_star = [a for a in z_n if gcd(a, n) == 1]

    print(f"Z_{n}  = {z_n}")
    print(f"Z_{n}* = {z_n_star}  (|Z_{n}*| = {len(z_n_star)})")
    return z_n_star

# Example: Z_12 has phi(12) = 4 invertible elements
residue_classes(12)
# Z_12  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Z_12* = [1, 5, 7, 11]  (|Z_12*| = 4)
```

---

## 2. GCD and the Extended Euclidean Algorithm

### 2.1 Greatest Common Divisor

The **greatest common divisor** $\gcd(a, b)$ is the largest positive integer that divides both $a$ and $b$.

**Euclidean Algorithm** — Based on the identity $\gcd(a, b) = \gcd(b, a \bmod b)$:

```python
def gcd(a, b):
    """Euclidean algorithm for GCD.

    Why this works: if d divides both a and b, then d also divides
    a - qb = a mod b. So gcd(a, b) = gcd(b, a mod b).
    Terminates because the remainder strictly decreases toward 0.
    """
    while b != 0:
        a, b = b, a % b
    return a

print(gcd(48, 18))   # 6
print(gcd(1071, 462)) # 21
```

**Time complexity:** $O(\log(\min(a, b)))$ — the number of steps is at most $5 \times$ the number of digits in the smaller number (Lamé's theorem).

### 2.2 Extended Euclidean Algorithm

The **Extended Euclidean Algorithm** finds integers $x$ and $y$ such that:

$$ax + by = \gcd(a, b)$$

This equation is known as **Bezout's identity**. Finding $x$ and $y$ is essential for computing modular inverses.

```python
def extended_gcd(a, b):
    """Extended Euclidean Algorithm.

    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).

    Why we track x, y: in RSA key generation, we need the modular
    inverse of e modulo phi(n), which requires exactly this algorithm.
    """
    if a == 0:
        return b, 0, 1

    # Recursive step: gcd(b % a, a) returns (g, x1, y1)
    # such that (b % a)*x1 + a*y1 = g
    g, x1, y1 = extended_gcd(b % a, a)

    # Since b % a = b - (b // a) * a, substitute back:
    # (b - (b//a)*a)*x1 + a*y1 = g
    # b*x1 + a*(y1 - (b//a)*x1) = g
    x = y1 - (b // a) * x1
    y = x1

    return g, x, y

# Verify: 48*(-1) + 18*(3) = -48 + 54 = 6 = gcd(48, 18)
g, x, y = extended_gcd(48, 18)
print(f"gcd = {g}, x = {x}, y = {y}")
print(f"Verification: 48*{x} + 18*{y} = {48*x + 18*y}")
```

### 2.3 Iterative Version

For production code, the iterative version avoids recursion depth limits:

```python
def extended_gcd_iterative(a, b):
    """Iterative Extended GCD — avoids stack overflow for large inputs.

    Why iterative: Python's default recursion limit is 1000. For
    cryptographic-size numbers (2048+ bits), the recursive version
    would exceed this limit.
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    # old_r = gcd, old_s = x, old_t = y
    return old_r, old_s, old_t
```

---

## 3. Modular Inverse

### 3.1 Definition

The **modular inverse** of $a$ modulo $n$ is an integer $a^{-1}$ such that:

$$a \cdot a^{-1} \equiv 1 \pmod{n}$$

A modular inverse exists **if and only if** $\gcd(a, n) = 1$ (i.e., $a$ and $n$ are coprime).

### 3.2 Computing the Inverse

From Bezout's identity, if $\gcd(a, n) = 1$, then there exist $x, y$ such that $ax + ny = 1$. Taking both sides modulo $n$:

$$ax \equiv 1 \pmod{n}$$

So $x \bmod n$ is the modular inverse of $a$.

```python
def mod_inverse(a, n):
    """Compute the modular inverse of a modulo n.

    Why this matters for RSA: the private key d is the modular
    inverse of the public exponent e modulo phi(n). Without this
    function, RSA key generation would be impossible.

    Raises ValueError if gcd(a, n) != 1 (no inverse exists).
    """
    g, x, _ = extended_gcd(a, n)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist: gcd({a}, {n}) = {g}")
    return x % n  # Why % n: ensure the result is in [0, n-1]

# In RSA: if e = 65537 and phi(n) = 3120, find d such that e*d ≡ 1 (mod phi(n))
e = 65537
phi_n = 3233 * 3216  # example value
d = mod_inverse(e, phi_n)
print(f"d = {d}")
print(f"Verification: e*d mod phi(n) = {(e * d) % phi_n}")  # Should be 1
```

### 3.3 Python's Built-in

Python 3.8+ provides `pow(a, -1, n)` for modular inverse:

```python
# Built-in modular inverse (uses the same algorithm internally)
print(pow(7, -1, 11))  # 8, because 7*8 = 56 ≡ 1 (mod 11)
```

---

## 4. Euler's Totient Function

### 4.1 Definition

**Euler's totient function** $\phi(n)$ counts the number of integers in $\{1, 2, \ldots, n\}$ that are coprime to $n$:

$$\phi(n) = |\{k : 1 \leq k \leq n, \gcd(k, n) = 1\}|$$

### 4.2 Key Formulas

| Case | Formula | Example |
|------|---------|---------|
| $p$ is prime | $\phi(p) = p - 1$ | $\phi(7) = 6$ |
| $p^k$ (prime power) | $\phi(p^k) = p^k - p^{k-1} = p^{k-1}(p-1)$ | $\phi(8) = 4$ |
| $\gcd(m,n) = 1$ | $\phi(mn) = \phi(m)\phi(n)$ (multiplicative) | $\phi(12) = \phi(4)\phi(3) = 2 \cdot 2 = 4$ |
| General | $\phi(n) = n \prod_{p \mid n}\left(1 - \frac{1}{p}\right)$ | $\phi(12) = 12(1-\frac{1}{2})(1-\frac{1}{3}) = 4$ |

The formula for $n = pq$ where $p, q$ are distinct primes is especially important for RSA:

$$\phi(pq) = (p-1)(q-1)$$

```python
def euler_totient(n):
    """Compute Euler's totient function using prime factorization.

    Why we need this: In RSA, phi(n) = (p-1)(q-1) determines the
    relationship between public and private keys. Knowing phi(n)
    is equivalent to being able to break RSA.
    """
    result = n
    p = 2
    temp = n

    while p * p <= temp:
        if temp % p == 0:
            # Remove all factors of p
            while temp % p == 0:
                temp //= p
            # Why multiply by (1 - 1/p): exclude multiples of p
            result -= result // p
        p += 1

    if temp > 1:
        # temp is a remaining prime factor
        result -= result // temp

    return result

# Verify: phi(12) = 4, phi(7) = 6, phi(100) = 40
for n in [12, 7, 100, 3233]:
    print(f"phi({n}) = {euler_totient(n)}")
```

---

## 5. Euler's Theorem and Fermat's Little Theorem

### 5.1 Euler's Theorem

If $\gcd(a, n) = 1$, then:

$$a^{\phi(n)} \equiv 1 \pmod{n}$$

> **Analogy:** Imagine repeatedly multiplying $a$ by itself modulo $n$. Euler's theorem guarantees that this process is **cyclic** — after exactly $\phi(n)$ multiplications (or a divisor thereof), you return to 1. It is like walking around a circular track of length $\phi(n)$.

**Why this matters for RSA:** If $ed \equiv 1 \pmod{\phi(n)}$, then for any message $m$ coprime to $n$:

$$m^{ed} = m^{1 + k\phi(n)} = m \cdot (m^{\phi(n)})^k \equiv m \cdot 1^k = m \pmod{n}$$

This is the mathematical foundation of RSA encryption and decryption.

### 5.2 Fermat's Little Theorem

A special case of Euler's theorem when $n = p$ is prime (so $\phi(p) = p - 1$):

$$a^{p-1} \equiv 1 \pmod{p} \quad \text{for } \gcd(a, p) = 1$$

Equivalently:

$$a^p \equiv a \pmod{p} \quad \text{for all } a$$

```python
def demonstrate_euler_fermat():
    """Demonstrate Euler's theorem and Fermat's little theorem."""

    # Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
    p = 17
    for a in [2, 5, 10, 16]:
        result = pow(a, p - 1, p)
        print(f"{a}^{p-1} mod {p} = {result}")  # Always 1

    print()

    # Euler's theorem: a^phi(n) ≡ 1 (mod n) for gcd(a, n) = 1
    n = 15  # phi(15) = phi(3)*phi(5) = 2*4 = 8
    phi_n = 8
    for a in [2, 4, 7, 11]:
        result = pow(a, phi_n, n)
        print(f"{a}^{phi_n} mod {n} = {result}")  # Always 1

demonstrate_euler_fermat()
```

### 5.3 Computing Modular Inverse via Euler's Theorem

Since $a^{\phi(n)} \equiv 1 \pmod{n}$, we have:

$$a^{-1} \equiv a^{\phi(n)-1} \pmod{n}$$

When $n$ is prime, this simplifies to $a^{-1} \equiv a^{n-2} \pmod{n}$.

```python
# Alternative modular inverse using Fermat's little theorem (only for prime modulus)
p = 17
a = 5
inverse = pow(a, p - 2, p)
print(f"{a}^(-1) mod {p} = {inverse}")
print(f"Verification: {a} * {inverse} mod {p} = {(a * inverse) % p}")
```

---

## 6. Chinese Remainder Theorem

### 6.1 Statement

If $n_1, n_2, \ldots, n_k$ are pairwise coprime (i.e., $\gcd(n_i, n_j) = 1$ for $i \neq j$), then the system of congruences:

$$x \equiv a_1 \pmod{n_1}$$
$$x \equiv a_2 \pmod{n_2}$$
$$\vdots$$
$$x \equiv a_k \pmod{n_k}$$

has a **unique** solution modulo $N = n_1 n_2 \cdots n_k$.

> **Analogy:** Imagine you have a number and you only know its remainders when divided by 3, 5, and 7. The CRT says these three remainders uniquely determine the number modulo 105 ($= 3 \times 5 \times 7$). It is like identifying a person by three independent characteristics — if the characteristics are "independent" (coprime moduli), the combination is unique.

### 6.2 Construction

Let $N = \prod_{i=1}^k n_i$ and $N_i = N / n_i$. Then:

$$x \equiv \sum_{i=1}^k a_i N_i (N_i^{-1} \bmod n_i) \pmod{N}$$

```python
def chinese_remainder_theorem(remainders, moduli):
    """Solve a system of congruences using the Chinese Remainder Theorem.

    Why CRT matters in cryptography:
    1. RSA decryption speedup: decrypt mod p and mod q separately,
       then combine using CRT (4x faster than direct computation)
    2. Secret sharing: split a secret into shares using CRT
    3. Batch verification: check multiple conditions simultaneously

    Args:
        remainders: list of a_i values
        moduli: list of n_i values (must be pairwise coprime)

    Returns:
        x such that x ≡ a_i (mod n_i) for all i
    """
    from functools import reduce

    N = reduce(lambda a, b: a * b, moduli)  # Product of all moduli

    x = 0
    for a_i, n_i in zip(remainders, moduli):
        N_i = N // n_i
        # Why mod_inverse: we need N_i^(-1) mod n_i for the CRT formula
        N_i_inv = pow(N_i, -1, n_i)  # Python 3.8+ built-in
        x += a_i * N_i * N_i_inv

    return x % N

# Example: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
x = chinese_remainder_theorem([2, 3, 2], [3, 5, 7])
print(f"x = {x}")  # x = 23
print(f"Verification: {x % 3}, {x % 5}, {x % 7}")  # 2, 3, 2
```

### 6.3 RSA-CRT Optimization

In RSA, instead of computing $m = c^d \bmod n$ directly, we compute:

$$m_p = c^{d \bmod (p-1)} \bmod p$$
$$m_q = c^{d \bmod (q-1)} \bmod q$$

Then combine using CRT. Since $p$ and $q$ are roughly half the size of $n$, and modular exponentiation is $O(k^3)$ in the bit length $k$, this provides approximately a **4x speedup**.

---

## 7. Fast Modular Exponentiation

### 7.1 The Problem

Computing $a^b \bmod n$ naively by multiplying $a$ by itself $b$ times is impossibly slow when $b$ has hundreds of digits. We need a method that runs in $O(\log b)$ multiplications.

### 7.2 Square-and-Multiply Algorithm

The key insight: express $b$ in binary and use **repeated squaring**.

If $b = b_k b_{k-1} \cdots b_1 b_0$ in binary, then:

$$a^b = a^{b_k \cdot 2^k + b_{k-1} \cdot 2^{k-1} + \cdots + b_0} = \prod_{i: b_i = 1} a^{2^i}$$

```python
def mod_pow(base, exp, mod):
    """Square-and-multiply modular exponentiation.

    Why this algorithm: computing 2^(2048) mod n directly would
    require 2^2048 multiplications. Square-and-multiply does it
    in at most 2*2048 = 4096 multiplications — the difference
    between impossible and instantaneous.

    Time complexity: O(log(exp)) multiplications, each O(log(mod)^2)
    Total: O(log(exp) * log(mod)^2)
    """
    result = 1
    base = base % mod  # Why: reduce base first to keep numbers small

    while exp > 0:
        if exp & 1:  # If current bit is 1
            result = (result * base) % mod
        exp >>= 1  # Shift to next bit
        base = (base * base) % mod  # Square the base

    return result

# Example: compute 7^256 mod 13
print(mod_pow(7, 256, 13))  # 9
# Verify with Python's built-in (which uses the same algorithm)
print(pow(7, 256, 13))      # 9
```

### 7.3 Why Python's `pow(base, exp, mod)` is Preferred

Python's built-in three-argument `pow()` uses optimized C code for modular exponentiation. Always prefer it over manual implementation in production code. The manual version above is for understanding the algorithm.

---

## 8. Primality Testing

### 8.1 Why Primes Matter

RSA security depends on the difficulty of factoring $n = pq$ where $p$ and $q$ are large primes. Generating these primes requires an efficient primality test.

### 8.2 Trial Division

Check divisibility by all integers up to $\sqrt{n}$:

```python
def is_prime_trial(n):
    """Trial division — simple but too slow for cryptographic sizes.

    Time complexity: O(sqrt(n)), which for a 1024-bit number
    means checking ~2^512 divisors. At 10^9 checks/second,
    this would take longer than the age of the universe.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

### 8.3 Fermat Primality Test

By Fermat's little theorem, if $p$ is prime then $a^{p-1} \equiv 1 \pmod{p}$. The **contrapositive**: if $a^{n-1} \not\equiv 1 \pmod{n}$, then $n$ is **definitely composite**.

**Problem:** Carmichael numbers (e.g., 561 = 3 * 11 * 17) satisfy $a^{n-1} \equiv 1 \pmod{n}$ for all $a$ coprime to $n$, despite being composite. We need a stronger test.

### 8.4 Miller-Rabin Primality Test

Write $n - 1 = 2^s \cdot d$ where $d$ is odd. For a randomly chosen witness $a$, compute:

$$a^d \bmod n$$

Then repeatedly square the result. If $n$ is prime, the sequence $a^d, a^{2d}, a^{4d}, \ldots, a^{2^s d}$ must satisfy one of:
- $a^d \equiv 1 \pmod{n}$, **or**
- $a^{2^r d} \equiv -1 \pmod{n}$ for some $0 \leq r < s$

If neither holds, $n$ is **definitely composite** (the witness $a$ "catches" the compositeness).

```python
import random

def miller_rabin(n, k=20):
    """Miller-Rabin probabilistic primality test.

    Why Miller-Rabin over Fermat: Miller-Rabin has no "Carmichael number"
    problem. Each witness catches a composite with probability >= 3/4,
    so k rounds give error probability <= (1/4)^k.

    With k=20, the false positive probability is < 2^(-40), which is
    negligible for most applications. For key generation, k=64 is common.

    Args:
        n: integer to test
        k: number of rounds (more rounds = higher confidence)

    Returns:
        True if n is probably prime, False if definitely composite
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^s * d with d odd
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    # k rounds of testing
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)  # a^d mod n

        if x == 1 or x == n - 1:
            continue  # This witness is inconclusive

        for _ in range(s - 1):
            x = pow(x, 2, n)  # Square x
            if x == n - 1:
                break  # Found -1, inconclusive
        else:
            # Went through all squarings without finding -1
            return False  # Definitely composite

    return True  # Probably prime

# Test some numbers
for n in [2, 17, 561, 1009, 1000000007, 2**61 - 1]:
    result = miller_rabin(n)
    print(f"miller_rabin({n}) = {result}")
```

### 8.5 Generating Large Primes

To generate a random $k$-bit prime:

```python
def generate_prime(bits, k=20):
    """Generate a random prime of the specified bit length.

    Why random odd numbers: by the Prime Number Theorem, roughly
    1 in every ln(2^bits) ≈ bits * ln(2) numbers near 2^bits is prime.
    For 1024-bit numbers, about 1 in 710 odd numbers is prime,
    so we expect ~355 trials on average.
    """
    while True:
        # Generate random odd number of the right size
        n = random.getrandbits(bits)
        n |= (1 << (bits - 1)) | 1  # Set MSB and LSB (ensure correct bit length and odd)

        if miller_rabin(n, k):
            return n

# Generate a 256-bit prime (for demonstration; RSA uses 1024+ bits)
p = generate_prime(256)
print(f"Generated prime ({p.bit_length()} bits): {p}")
```

---

## 9. The Discrete Logarithm Problem

### 9.1 Definition

Given a prime $p$, a generator $g$ of $\mathbb{Z}_p^*$, and an element $h \in \mathbb{Z}_p^*$, the **discrete logarithm problem (DLP)** asks:

$$\text{Find } x \text{ such that } g^x \equiv h \pmod{p}$$

> **Analogy:** Computing $g^x \bmod p$ is like mixing paints — easy to do forward (mix red and blue to get purple), but extremely hard to reverse (given purple, determine the exact amounts of red and blue). This **one-way** property is the foundation of Diffie-Hellman key exchange and DSA/ECDSA signatures.

### 9.2 Why It's Hard

- **Forward direction** (exponentiation): $O(\log x)$ multiplications using square-and-multiply
- **Reverse direction** (discrete log): best known classical algorithms are sub-exponential:
  - Baby-step giant-step: $O(\sqrt{p})$ time and space
  - Index calculus: $O(\exp(c \cdot (\log p)^{1/3} (\log \log p)^{2/3}))$

For a 2048-bit prime, even index calculus is infeasible with current technology.

### 9.3 Baby-Step Giant-Step Algorithm

A meet-in-the-middle approach to solve DLP in $O(\sqrt{p})$:

```python
import math

def baby_step_giant_step(g, h, p):
    """Solve g^x ≡ h (mod p) for x using baby-step giant-step.

    Why this algorithm: it demonstrates that DLP is not as hard as
    brute force (O(p)) but still exponential in the bit length.
    For small primes (educational use), this is practical.

    Time/space complexity: O(sqrt(p))
    """
    m = math.isqrt(p) + 1

    # Baby step: compute g^j mod p for j = 0, 1, ..., m-1
    table = {}
    power = 1
    for j in range(m):
        table[power] = j
        power = (power * g) % p

    # Giant step: compute h * (g^(-m))^i for i = 0, 1, ..., m-1
    # g^(-m) = modular inverse of g^m
    g_inv_m = pow(g, -m, p)  # g^(-m) mod p

    gamma = h
    for i in range(m):
        if gamma in table:
            x = i * m + table[gamma]
            return x
        gamma = (gamma * g_inv_m) % p

    return None  # No solution found (shouldn't happen if g is a generator)

# Example: solve 2^x ≡ 9 (mod 23)
x = baby_step_giant_step(2, 9, 23)
print(f"2^x ≡ 9 (mod 23) → x = {x}")
print(f"Verification: 2^{x} mod 23 = {pow(2, x, 23)}")
```

### 9.4 Groups and Generators

An element $g \in \mathbb{Z}_p^*$ is a **generator** (or **primitive root**) if the set $\{g^0, g^1, g^2, \ldots, g^{p-2}\}$ equals $\mathbb{Z}_p^*$. Not all elements are generators — the number of generators of $\mathbb{Z}_p^*$ is $\phi(p-1)$.

```python
def find_generator(p):
    """Find a generator (primitive root) of Z_p^*.

    Why generators matter: Diffie-Hellman requires a generator g
    so that g^x covers all of Z_p^*, ensuring no structural weakness.
    """
    # Check each candidate
    for g in range(2, p):
        # g is a generator iff g^((p-1)/q) ≢ 1 (mod p) for each
        # prime factor q of p-1
        is_generator = True
        temp = p - 1

        # Find prime factors of p-1
        factors = set()
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.add(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)

        for q in factors:
            if pow(g, (p - 1) // q, p) == 1:
                is_generator = False
                break

        if is_generator:
            return g

    return None

p = 23
g = find_generator(p)
print(f"Generator of Z_{p}*: {g}")
# Verify: list all powers
powers = [pow(g, i, p) for i in range(p - 1)]
print(f"Powers: {sorted(powers)}")  # Should be [1, 2, ..., p-1]
```

---

## 10. Exercises

### Exercise 1: Modular Arithmetic Warm-up (Basic)

Compute by hand (then verify with Python):
1. $17^3 \bmod 13$
2. $7^{-1} \bmod 31$
3. $\phi(360)$

### Exercise 2: Extended GCD Application (Intermediate)

Use the Extended Euclidean Algorithm to solve $1234x + 567y = \gcd(1234, 567)$. Then find $1234^{-1} \bmod 567$ (if it exists).

### Exercise 3: CRT Problem (Intermediate)

A number leaves remainder 2 when divided by 3, remainder 3 when divided by 5, and remainder 4 when divided by 7. Find the smallest positive number satisfying all three conditions using:
1. The CRT algorithm from Section 6
2. Brute force (for verification)

### Exercise 4: Miller-Rabin Analysis (Challenging)

1. Show that 561 is a Carmichael number: verify that $a^{560} \equiv 1 \pmod{561}$ for $a \in \{2, 5, 10\}$.
2. Show that Miller-Rabin correctly identifies 561 as composite. Find a witness $a$ that exposes 561.
3. How many Miller-Rabin rounds are needed to achieve a false positive probability below $2^{-128}$?

### Exercise 5: Discrete Logarithm (Challenging)

1. Find all generators of $\mathbb{Z}_{23}^*$.
2. Solve $5^x \equiv 8 \pmod{23}$ using baby-step giant-step.
3. Explain why the DLP in $\mathbb{Z}_p^*$ with a 256-bit prime $p$ is considered insecure today, while a 2048-bit prime is considered secure.

---

**Next**: [Symmetric Ciphers](./02_Symmetric_Ciphers.md)

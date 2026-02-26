# Lesson 10: Shor's Factoring Algorithm

[← Previous: Quantum Fourier Transform](09_Quantum_Fourier_Transform.md) | [Next: Quantum Error Correction →](11_Quantum_Error_Correction.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why integer factoring is computationally hard classically and why it matters for cryptography
2. Describe the reduction from factoring to period finding
3. Walk through the quantum period-finding subroutine using QPE and modular exponentiation
4. Trace Shor's algorithm step by step for small numbers (N = 15, 21)
5. Analyze the complexity of Shor's algorithm: $O((\log N)^3)$ quantum gates
6. Discuss the implications for RSA and post-quantum cryptography
7. Implement a classical simulation of Shor's algorithm for small inputs

---

In 1994, Peter Shor published an algorithm that shook the foundations of modern cryptography. He showed that a quantum computer could factor large integers in polynomial time — a task that the best classical algorithms solve only in sub-exponential time. Since the security of RSA encryption (the backbone of internet security) rests on the assumption that factoring is intractable, Shor's algorithm implies that a sufficiently powerful quantum computer would break RSA, along with Diffie-Hellman key exchange and elliptic curve cryptography.

Shor's algorithm is not just a theoretical curiosity. It is the primary reason that governments and corporations are investing billions in quantum computing research, and it is the driving force behind the development of post-quantum cryptography standards (see Security L05 for TLS/PKI context). Understanding Shor's algorithm requires the QFT machinery from Lesson 9, and it provides one of the most compelling demonstrations of quantum computational advantage.

> **Analogy:** Shor's algorithm exploits quantum interference like finding the rhythm in a song. The QFT reveals the period (rhythm) of modular exponentiation, which classically would require listening to an astronomically long piece. Imagine you need to find the beat frequency of a song that lasts a billion years — classical approaches must listen for a very long time, but quantum interference lets you identify the rhythm from a brief quantum "sample."

## Table of Contents

1. [The Factoring Problem](#1-the-factoring-problem)
2. [Reduction: Factoring to Period Finding](#2-reduction-factoring-to-period-finding)
3. [Classical Period Finding Is Hard](#3-classical-period-finding-is-hard)
4. [Quantum Period Finding](#4-quantum-period-finding)
5. [The Complete Shor's Algorithm](#5-the-complete-shors-algorithm)
6. [Worked Example: Factoring 15](#6-worked-example-factoring-15)
7. [Worked Example: Factoring 21](#7-worked-example-factoring-21)
8. [Complexity Analysis](#8-complexity-analysis)
9. [Implications for Cryptography](#9-implications-for-cryptography)
10. [Python Implementation](#10-python-implementation)
11. [Exercises](#11-exercises)

---

## 1. The Factoring Problem

### 1.1 Problem Statement

**Integer Factoring**: Given a composite integer $N$, find a non-trivial factor $p$ such that $1 < p < N$ and $p | N$.

For example: $N = 15 \to p = 3$ (or $p = 5$), since $15 = 3 \times 5$.

### 1.2 Why Factoring Is Believed to Be Hard

The best known classical algorithms for factoring:

| Algorithm | Complexity | Type |
|-----------|-----------|------|
| Trial division | $O(\sqrt{N})$ | Exponential in $\log N$ |
| Pollard's rho | $O(N^{1/4})$ | Sub-exponential (heuristic) |
| Quadratic sieve | $O(e^{c\sqrt{\ln N \ln\ln N}})$ | Sub-exponential |
| General number field sieve (GNFS) | $O(e^{c(\ln N)^{1/3}(\ln\ln N)^{2/3}})$ | Sub-exponential |

None of these is polynomial in the input size $n = \log_2 N$. The GNFS is the fastest known classical algorithm, but its sub-exponential runtime means that doubling the number of digits roughly cubes the computation time.

### 1.3 Connection to Cryptography

RSA encryption relies on the difficulty of factoring the product of two large primes:

1. **Key generation**: Choose two large primes $p, q$ (each ~1024 bits). Compute $N = pq$.
2. **Public key**: $(N, e)$ where $e$ is coprime to $\phi(N) = (p-1)(q-1)$
3. **Private key**: $d = e^{-1} \mod \phi(N)$

Security depends on: given $N$, finding $p$ and $q$ is computationally infeasible. If factoring were easy, one could compute $\phi(N)$ and then $d$, breaking RSA completely.

Current RSA keys use $N$ of 2048-4096 bits. The GNFS would take longer than the age of the universe to factor these. Shor's algorithm would need roughly $4000 \cdot n$ logical qubits and $O(n^3)$ gates, where $n = \log_2 N$.

---

## 2. Reduction: Factoring to Period Finding

The key insight of Shor's algorithm is that **factoring reduces to finding the period of a function**. This reduction is entirely classical — the quantum speedup comes only in the period-finding step.

### 2.1 Order Finding

For an integer $a$ coprime to $N$ (i.e., $\gcd(a, N) = 1$), the **order** of $a$ modulo $N$ is the smallest positive integer $r$ such that:

$$a^r \equiv 1 \pmod{N}$$

The order always exists (by Euler's theorem, $r | \phi(N)$) and $r \leq N$.

### 2.2 From Order to Factor

**Theorem**: If $r$ is the order of $a$ modulo $N$ and $r$ is even, then:

$$a^r - 1 = (a^{r/2} - 1)(a^{r/2} + 1) \equiv 0 \pmod{N}$$

This means $N | (a^{r/2} - 1)(a^{r/2} + 1)$. If neither factor is divisible by $N$ (i.e., $a^{r/2} \not\equiv \pm 1 \pmod{N}$), then:

$$\gcd(a^{r/2} - 1, N) \quad \text{and} \quad \gcd(a^{r/2} + 1, N)$$

are both non-trivial factors of $N$.

### 2.3 The Reduction Algorithm

Given $N$ to factor:

1. Choose a random $a \in \{2, 3, \ldots, N-1\}$
2. Compute $\gcd(a, N)$. If $\gcd(a, N) > 1$, we found a factor (lucky!)
3. Find the order $r$ of $a$ modulo $N$ **(this is the hard step)**
4. If $r$ is odd, go back to step 1
5. Compute $\gcd(a^{r/2} - 1, N)$ and $\gcd(a^{r/2} + 1, N)$
6. If either gives a non-trivial factor, output it; otherwise, go back to step 1

**Success probability**: For a random $a$, the probability that $r$ is even and $a^{r/2} \not\equiv -1 \pmod{N}$ is at least $1/2$ (when $N$ has two or more distinct odd prime factors). So we expect $O(1)$ repetitions on average.

### 2.4 Python: Classical Reduction

```python
import numpy as np
from math import gcd

def factoring_via_order(N, a, order_r):
    """Given the order r of a mod N, attempt to find a factor.

    Why use gcd? The key algebraic identity a^r - 1 = (a^{r/2} - 1)(a^{r/2} + 1)
    means N divides the product but (hopefully) not either factor individually.
    The gcd extracts the shared factor between N and each term.
    """
    if order_r % 2 != 0:
        return None, "Order is odd — retry with different a"

    x = pow(a, order_r // 2, N)  # a^{r/2} mod N

    if x == N - 1:  # x ≡ -1 (mod N)
        return None, "a^{r/2} ≡ -1 (mod N) — retry with different a"

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    if factor1 not in (1, N):
        return factor1, f"gcd({x}-1, {N}) = {factor1}"
    if factor2 not in (1, N):
        return factor2, f"gcd({x}+1, {N}) = {factor2}"

    return None, "Both gcds are trivial — retry"

# Example: Factor 15 using a = 7
N = 15
a = 7
# Order of 7 mod 15: 7^1=7, 7^2=49≡4, 7^3=343≡13, 7^4=2401≡1 → r=4
r = 4
factor, msg = factoring_via_order(N, a, r)
print(f"N={N}, a={a}, r={r}: {msg}")
if factor:
    print(f"  {N} = {factor} × {N // factor}")
```

---

## 3. Classical Period Finding Is Hard

### 3.1 The Period-Finding Problem

Given a function $f(x) = a^x \bmod N$, find the period $r$ — the smallest positive integer such that $f(x + r) = f(x)$ for all $x$.

### 3.2 Why Classical Approaches Fail

**Brute force**: Compute $a^0, a^1, a^2, \ldots$ until we see $a^r \equiv 1$. Since $r$ can be as large as $N$, and $N$ can be a 2048-bit number ($N \sim 2^{2048}$), this requires an astronomical number of steps.

**Birthday-style attacks**: Pollard's rho algorithm uses $O(\sqrt{r})$ space and time, but $\sqrt{r}$ can still be $\sim 2^{1024}$, which is far beyond reach.

The fundamental issue is that classically, there is no known way to detect periodicity in $f(x) = a^x \bmod N$ without evaluating $f$ at many points. Quantum mechanics provides a shortcut: **evaluate $f$ on a superposition of all inputs simultaneously, then use the QFT to extract the period from interference patterns**.

---

## 4. Quantum Period Finding

### 4.1 Overview

The quantum subroutine uses two registers:

- **Counting register**: $t$ qubits, where $t = 2n + 1$ and $n = \lceil \log_2 N \rceil$ (sufficient precision)
- **Target register**: $n$ qubits to hold $f(x) = a^x \bmod N$

### 4.2 Step-by-Step

**Step 1: Initialize**

$$|0\rangle^{\otimes t} |0\rangle^{\otimes n}$$

**Step 2: Create superposition** (apply $H^{\otimes t}$ to the counting register)

$$\frac{1}{\sqrt{2^t}} \sum_{x=0}^{2^t - 1} |x\rangle |0\rangle$$

**Step 3: Apply modular exponentiation** (the quantum oracle)

$$\frac{1}{\sqrt{2^t}} \sum_{x=0}^{2^t - 1} |x\rangle |a^x \bmod N\rangle$$

This is the most expensive step: the controlled modular exponentiation circuit requires $O(n^3)$ gates.

**Step 4: Apply inverse QFT to the counting register**

After the QFT (technically the inverse QFT), the counting register contains a superposition of states that encode information about the period $r$.

**Step 5: Measure the counting register**

The measurement outcome $m$ satisfies approximately:

$$\frac{m}{2^t} \approx \frac{s}{r}$$

for some integer $s \in \{0, 1, \ldots, r-1\}$. Using the continued fraction algorithm on $m/2^t$, we can extract $r$.

### 4.3 Why the QFT Reveals the Period

After step 3, suppose the target register is measured (conceptually) and yields some value $f_0 = a^{x_0} \bmod N$. The counting register collapses to a superposition over all $x$ values that map to $f_0$:

$$|\psi\rangle = \frac{1}{\sqrt{A}} \sum_{j=0}^{A-1} |x_0 + jr\rangle$$

where $A \approx 2^t / r$ is the number of valid $x$ values. This state is periodic with period $r$.

Applying the QFT to a periodic state concentrates the probability at multiples of $2^t / r$, giving measurement outcomes near $s \cdot 2^t / r$ for integer $s$. This is exactly the periodicity-detection property we studied in Lesson 9.

### 4.4 Continued Fraction Algorithm

From the measurement outcome $m$, we form the fraction $m / 2^t$ and compute its continued fraction expansion. The convergent whose denominator is $\leq N$ gives a candidate for $r$.

For example, if $2^t = 256$ and $m = 85$:

$$\frac{85}{256} = 0.33203125 \approx \frac{1}{3}$$

The continued fraction expansion of $85/256$ is $[0; 3, 85/256 \text{ details}]$, which gives the convergent $1/3$. So $r = 3$ is the candidate period.

```python
def continued_fraction(numerator, denominator, max_denom):
    """Extract the period using continued fractions.

    Why continued fractions? The measurement outcome m/2^t is a noisy estimate
    of s/r. Continued fractions find the "simplest" fraction (smallest
    denominator) close to a given real number — exactly what we need to
    recover r from an approximate rational.
    """
    # Compute continued fraction coefficients
    cf = []
    n, d = numerator, denominator
    while d > 0 and len(cf) < 50:
        cf.append(n // d)
        n, d = d, n % d

    # Compute convergents and return the one with denominator ≤ max_denom
    convergents = []
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    for a in cf:
        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
        if k_curr > max_denom:
            break
        convergents.append((h_curr, k_curr))

    return convergents

# Example: recover period from measurement outcome
m = 85
T = 256
N = 15

convergents = continued_fraction(m, T, N)
print(f"m/2^t = {m}/{T} = {m/T:.6f}")
print("Convergents:")
for h, k in convergents:
    print(f"  {h}/{k} = {h/k:.6f}")
```

---

## 5. The Complete Shor's Algorithm

### 5.1 Algorithm Summary

**Input**: Composite integer $N$ to factor.

**Output**: A non-trivial factor of $N$.

1. **Check trivial cases**: If $N$ is even, return 2. If $N = p^k$ for some prime $p$ and $k \geq 2$, use classical methods.

2. **Choose random $a$**: Pick $a$ uniformly at random from $\{2, \ldots, N-1\}$.

3. **Check GCD**: Compute $g = \gcd(a, N)$. If $g > 1$, return $g$.

4. **Quantum period finding**:
   - Prepare counting register ($t = 2\lceil\log_2 N\rceil + 1$ qubits) and work register ($\lceil\log_2 N\rceil$ qubits)
   - Apply $H^{\otimes t}$ to counting register
   - Apply controlled modular exponentiation: $|x\rangle|y\rangle \to |x\rangle|y \cdot a^x \bmod N\rangle$
   - Apply $\text{QFT}^{-1}$ to counting register
   - Measure counting register to get outcome $m$

5. **Classical post-processing**:
   - Use continued fractions on $m / 2^t$ to find candidate period $r$
   - If $r$ is odd or $a^{r/2} \equiv -1 \pmod{N}$, go to step 2
   - Compute $\gcd(a^{r/2} \pm 1, N)$ and return any non-trivial factor

6. **Repeat** if necessary (expected $O(1)$ repetitions).

### 5.2 Circuit Diagram

```
Counting register (t qubits):

|0⟩ ─── H ─── ───────── ───── ───── ┌───────┐
|0⟩ ─── H ─── ───────── ───── U²   │       │
|0⟩ ─── H ─── ───────── U⁴   ───── │ QFT⁻¹ │ ── Measure → m
|0⟩ ─── H ─── U⁸       ───── ───── │       │
        ...                          └───────┘

Work register (n qubits):

|1⟩ ─── ───── target ── target target ────────────
```

where $U|y\rangle = |ay \bmod N\rangle$ is the modular multiplication operator. The controlled-$U^{2^j}$ gates implement $|y\rangle \to |a^{2^j} y \bmod N\rangle$ when the control qubit is $|1\rangle$.

---

## 6. Worked Example: Factoring 15

### 6.1 Setup

$N = 15$. We need $n = \lceil\log_2 15\rceil = 4$ bits to represent $N$.

Choose $a = 7$ (coprime to 15, verified: $\gcd(7, 15) = 1$).

### 6.2 Computing the Order Classically (for Verification)

$$7^1 = 7, \quad 7^2 = 49 \equiv 4, \quad 7^3 = 343 \equiv 13, \quad 7^4 = 2401 \equiv 1 \pmod{15}$$

So $r = 4$.

### 6.3 Quantum Period Finding

Use $t = 2 \times 4 + 1 = 9$ counting qubits, so $2^t = 512$.

After the modular exponentiation and inverse QFT, the counting register contains a superposition peaked near multiples of $2^t / r = 512 / 4 = 128$:

$$m \in \{0, 128, 256, 384\}$$

Each is measured with approximately equal probability $\approx 1/4$.

### 6.4 Post-Processing

| Measurement $m$ | $m/2^t$ | Continued fraction | Candidate $r$ | Valid? |
|-----|---------|-------------------|---------------|--------|
| 0 | 0/512 = 0 | $0/1$ | 1 | No (trivial) |
| 128 | 128/512 = 1/4 | $1/4$ | 4 | **Yes** |
| 256 | 256/512 = 1/2 | $1/2$ | 2 | Maybe (try) |
| 384 | 384/512 = 3/4 | $3/4$ | 4 | **Yes** |

For $r = 4$ (even, good):

$$a^{r/2} = 7^2 = 49 \equiv 4 \pmod{15}$$

Since $4 \not\equiv -1 \pmod{15}$:

$$\gcd(4 - 1, 15) = \gcd(3, 15) = 3$$
$$\gcd(4 + 1, 15) = \gcd(5, 15) = 5$$

**Result**: $15 = 3 \times 5$. Factoring successful!

---

## 7. Worked Example: Factoring 21

### 7.1 Setup

$N = 21 = 3 \times 7$. Choose $a = 2$.

### 7.2 Order Computation

$$2^1 = 2, \quad 2^2 = 4, \quad 2^3 = 8, \quad 2^4 = 16, \quad 2^5 = 32 \equiv 11, \quad 2^6 = 64 \equiv 1 \pmod{21}$$

So $r = 6$.

### 7.3 Post-Processing

$r = 6$ is even. Compute $a^{r/2} = 2^3 = 8$.

Check: $8 \not\equiv -1 \pmod{21}$ (since $-1 \equiv 20$).

$$\gcd(8 - 1, 21) = \gcd(7, 21) = 7$$
$$\gcd(8 + 1, 21) = \gcd(9, 21) = 3$$

**Result**: $21 = 3 \times 7$. Success!

### 7.4 What If We Chose $a = 4$?

$$4^1 = 4, \quad 4^2 = 16, \quad 4^3 = 64 \equiv 1 \pmod{21}$$

$r = 3$ is odd. The algorithm says: retry with a different $a$. This demonstrates why we sometimes need multiple attempts.

---

## 8. Complexity Analysis

### 8.1 Gate Count

The quantum circuit for Shor's algorithm has three main components:

| Component | Gate count |
|-----------|-----------|
| Hadamard layer ($H^{\otimes t}$) | $O(n)$ |
| Modular exponentiation ($a^x \bmod N$) | $O(n^3)$ |
| Inverse QFT | $O(n^2)$ |

The modular exponentiation dominates: it requires $O(n)$ sequential controlled modular multiplications, each costing $O(n^2)$ gates. Total: $O(n^3)$ where $n = \lceil\log_2 N\rceil$.

### 8.2 Qubit Count

- Counting register: $t = 2n + 1$ qubits
- Work register: $n$ qubits
- Ancilla qubits for modular arithmetic: $O(n)$

**Total**: $O(n)$ qubits $= O(\log N)$ qubits.

For RSA-2048 ($n = 2048$ bits), this means roughly $4n + O(n) \approx 10{,}000$ logical qubits. With quantum error correction overhead (roughly 1000-10000 physical qubits per logical qubit), this translates to millions of physical qubits.

### 8.3 Total Complexity

$$\text{Shor's algorithm}: O(n^3) = O((\log N)^3) \text{ quantum gates}$$

Compare with the best classical algorithm (GNFS):

$$\text{GNFS}: O\left(\exp\left(c \cdot (\log N)^{1/3} (\log\log N)^{2/3}\right)\right)$$

For $N$ with 2048 bits:
- **Shor's**: $\sim 2048^3 \approx 10^{10}$ operations (feasible)
- **GNFS**: $\sim 10^{30}$ operations (intractable)

This is an *exponential* quantum speedup.

---

## 9. Implications for Cryptography

### 9.1 What Shor's Algorithm Breaks

| Cryptosystem | Based on | Broken by Shor? |
|-------------|----------|-----------------|
| RSA | Integer factoring | **Yes** |
| Diffie-Hellman | Discrete logarithm | **Yes** (same algorithm, minor modification) |
| Elliptic Curve (ECDSA, ECDH) | Elliptic curve discrete log | **Yes** |
| AES (symmetric) | — | **No** (Grover gives $\sqrt{}$ speedup, mitigated by doubling key size) |
| SHA-256 (hashing) | — | **No** (Grover gives $\sqrt{}$ speedup for preimage) |

### 9.2 Timeline and Threat Assessment

As of 2025:

- **Largest number factored by Shor's**: 21 (on actual quantum hardware)
- **Largest number factored by quantum methods**: small numbers (<100) using variational approaches
- **Estimated year for RSA-2048 threat**: 2030s-2040s (depends on hardware progress)
- **Harvest now, decrypt later**: adversaries may already be collecting encrypted data to decrypt once quantum computers are available

### 9.3 Post-Quantum Cryptography

NIST standardized post-quantum algorithms in 2024:

- **CRYSTALS-Kyber** (ML-KEM): Lattice-based key encapsulation
- **CRYSTALS-Dilithium** (ML-DSA): Lattice-based digital signatures
- **SPHINCS+** (SLH-DSA): Hash-based digital signatures

These are believed to be secure against both classical and quantum attacks (see Security L05 for TLS/PKI context).

---

## 10. Python Implementation

### 10.1 Complete Shor's Algorithm Simulation

```python
import numpy as np
from math import gcd, isqrt

def is_prime_power(N):
    """Check if N is a prime power.

    Why check this first? Shor's algorithm assumes N is composite and not
    a prime power. Prime powers can be factored classically in polynomial
    time using Newton's method for k-th roots.
    """
    for k in range(2, int(np.log2(N)) + 1):
        root = round(N**(1/k))
        for r in [root - 1, root, root + 1]:
            if r > 1 and r**k == N:
                return True, r, k
    return False, None, None

def classical_order_finding(a, N):
    """Find the order of a modulo N by brute force.

    Why brute force? On a real quantum computer, this step is replaced by
    quantum period finding. We use the classical version for simulation
    and verification of small cases.
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            return None  # Should not happen if gcd(a,N) = 1
    return r

def quantum_period_finding_simulation(a, N, n_counting=None):
    """Simulate the quantum period-finding subroutine.

    This simulates what the quantum circuit does: it creates the probability
    distribution that the inverse QFT produces, then samples from it.
    A real quantum computer would use actual quantum gates.
    """
    if n_counting is None:
        n = int(np.ceil(np.log2(N)))
        n_counting = 2 * n + 1

    Q = 2**n_counting  # Size of counting register

    # Build the state after modular exponentiation + inverse QFT
    # For each measurement outcome m, the probability is:
    # P(m) = |1/Q * sum_{x: a^x mod N = f0} exp(-2πi·m·x/Q)|²
    # summed over all possible f0 values

    # For simulation, we compute the order classically, then simulate
    # the measurement distribution
    r = classical_order_finding(a, N)
    if r is None:
        return None, None

    # The probability peaks near multiples of Q/r
    # P(m) is largest when m ≈ s·Q/r for integer s
    probs = np.zeros(Q)
    for m in range(Q):
        # Sum over s from 0 to r-1
        amplitude = 0
        for x0 in range(r):
            # Number of terms in the sum with this residue
            count = (Q - x0 + r - 1) // r
            for j in range(count):
                x = x0 + j * r
                amplitude += np.exp(-2j * np.pi * m * x / Q)
        probs[m] = abs(amplitude / Q)**2

    # Normalize
    probs /= probs.sum()

    # Sample from the distribution
    measurement = np.random.choice(Q, p=probs)

    return measurement, Q

def continued_fractions_period(m, Q, N):
    """Extract candidate period from measurement using continued fractions.

    Why continued fractions? The measurement gives m/Q ≈ s/r for unknown s and r.
    Continued fractions find the best rational approximation with small denominator,
    which recovers r even from an imperfect measurement.
    """
    if m == 0:
        return None

    # Compute continued fraction expansion of m/Q
    candidates = []
    n, d = m, Q
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    while d > 0:
        a_i = n // d
        n, d = d, n % d

        h_prev, h_curr = h_curr, a_i * h_curr + h_prev
        k_prev, k_curr = k_curr, a_i * k_curr + k_prev

        if k_curr > N:
            break
        if k_curr > 0:
            candidates.append(k_curr)

    return candidates

def shors_algorithm(N, max_attempts=20):
    """Complete Shor's algorithm for factoring N.

    Returns a non-trivial factor of N, or None if all attempts fail.
    """
    print(f"Factoring N = {N}")
    print("=" * 50)

    # Step 1: Trivial checks
    if N % 2 == 0:
        print(f"  N is even: factor = 2")
        return 2

    is_pp, base, exp = is_prime_power(N)
    if is_pp:
        print(f"  N = {base}^{exp}: factor = {base}")
        return base

    # Step 2-6: Main loop
    for attempt in range(1, max_attempts + 1):
        a = np.random.randint(2, N)
        print(f"\n  Attempt {attempt}: a = {a}")

        # Step 3: Check GCD
        g = gcd(a, N)
        if g > 1:
            print(f"    Lucky! gcd({a}, {N}) = {g}")
            return g

        # Step 4: Quantum period finding (simulated)
        r = classical_order_finding(a, N)
        print(f"    Order of {a} mod {N}: r = {r}")

        if r is None:
            continue

        # Step 5: Check conditions
        if r % 2 != 0:
            print(f"    r = {r} is odd — retrying")
            continue

        x = pow(a, r // 2, N)
        print(f"    a^(r/2) mod N = {a}^{r//2} mod {N} = {x}")

        if x == N - 1:
            print(f"    a^(r/2) ≡ -1 (mod N) — retrying")
            continue

        # Step 6: Compute factors
        f1 = gcd(x - 1, N)
        f2 = gcd(x + 1, N)
        print(f"    gcd({x}-1, {N}) = {f1}")
        print(f"    gcd({x}+1, {N}) = {f2}")

        if f1 not in (1, N):
            print(f"    SUCCESS: {N} = {f1} × {N // f1}")
            return f1
        if f2 not in (1, N):
            print(f"    SUCCESS: {N} = {f2} × {N // f2}")
            return f2

        print(f"    Both factors trivial — retrying")

    print(f"\n  Failed after {max_attempts} attempts")
    return None

# Run Shor's algorithm on several numbers
for N in [15, 21, 35, 77, 91]:
    factor = shors_algorithm(N)
    print()
```

### 10.2 Quantum Measurement Simulation

```python
import numpy as np
from math import gcd

def simulate_shor_measurement(a, N, n_counting_extra=3):
    """Simulate the quantum measurement step of Shor's algorithm.

    This shows the probability distribution over measurement outcomes
    and demonstrates how the peaks encode the period.
    """
    n = int(np.ceil(np.log2(N)))
    t = 2 * n + n_counting_extra  # Extra bits for precision
    Q = 2**t

    # Find the true order (for simulation purposes)
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1

    print(f"Simulating Shor's measurement for a={a}, N={N}")
    print(f"  True period: r = {r}")
    print(f"  Counting register: {t} qubits (Q = {Q})")
    print(f"  Expected peaks near multiples of Q/r = {Q/r:.2f}")

    # Compute probability distribution
    # The state after modular exponentiation has the form:
    # (1/√Q) Σ_x |x⟩|a^x mod N⟩
    # After measuring the work register (conceptually), we get:
    # (1/√A) Σ_j |x_0 + jr⟩ for some x_0
    # The inverse QFT maps this to peaks near multiples of Q/r

    probs = np.zeros(Q)
    A = Q // r  # Approximate number of terms

    for m in range(Q):
        # Average over all possible x_0 values (0 to r-1)
        total_prob = 0
        for x0 in range(r):
            amplitude = 0
            j = 0
            x = x0
            while x < Q:
                amplitude += np.exp(-2j * np.pi * m * x / Q)
                x += r
                j += 1
            total_prob += abs(amplitude)**2
        probs[m] = total_prob / (r * Q)

    probs /= probs.sum()

    # Show the peaks
    print(f"\n  Top measurement outcomes:")
    top_indices = np.argsort(probs)[::-1][:min(2*r, 10)]
    for idx in sorted(top_indices):
        if probs[idx] > 0.005:
            ratio = idx / Q
            # Find nearest s/r
            best_s = round(ratio * r)
            print(f"    m = {idx:4d}: P = {probs[idx]:.4f}, "
                  f"m/Q = {ratio:.6f}, nearest s/r = {best_s}/{r} = {best_s/r:.6f}")

    # Simulate sampling
    print(f"\n  Simulating 10 measurements:")
    for trial in range(10):
        m = np.random.choice(Q, p=probs)
        candidates = continued_fractions_period(m, Q, N)
        print(f"    m = {m}, m/Q = {m/Q:.6f}, "
              f"candidate periods: {candidates}")

# Simulate for N=15, a=7
simulate_shor_measurement(7, 15)

print("\n" + "=" * 60)

# Simulate for N=21, a=2
simulate_shor_measurement(2, 21)
```

### 10.3 Modular Exponentiation (Classical)

```python
def modular_exponentiation_trace(a, x, N):
    """Compute a^x mod N using repeated squaring, showing the trace.

    Why repeated squaring? This is the classical algorithm that makes
    modular exponentiation efficient: O(log x) multiplications instead of
    O(x). The quantum circuit uses a similar decomposition, with each
    squaring step becoming a controlled-U^{2^j} operation.
    """
    # Write x in binary: x = x_{t-1} * 2^{t-1} + ... + x_1 * 2 + x_0
    binary = bin(x)[2:]
    t = len(binary)

    print(f"Computing {a}^{x} mod {N}")
    print(f"  x = {x} = {binary}₂ ({t} bits)")

    # Precompute a^{2^j} mod N for j = 0, ..., t-1
    powers = [a % N]
    for j in range(1, t):
        powers.append(powers[-1]**2 % N)
    print(f"  Powers of a: a^(2^j) mod {N} = {powers}")

    # Multiply the powers corresponding to 1-bits
    result = 1
    for j in range(t):
        bit = int(binary[t - 1 - j])
        if bit == 1:
            result = (result * powers[j]) % N
            print(f"  bit {j} = 1: multiply by {powers[j]}, result = {result}")
        else:
            print(f"  bit {j} = 0: skip")

    print(f"  Final: {a}^{x} mod {N} = {result}")
    return result

# Trace modular exponentiation for Shor's algorithm
modular_exponentiation_trace(7, 11, 15)
```

---

## 11. Exercises

### Exercise 1: Manual Shor's Algorithm

Apply Shor's algorithm by hand to factor $N = 35$:
(a) Choose $a = 2$. Compute the order of 2 modulo 35.
(b) Is $r$ even? If so, compute $a^{r/2} \bmod 35$.
(c) Compute $\gcd(a^{r/2} \pm 1, 35)$ to find the factors.
(d) Repeat with $a = 3$. Does it work? Why or why not?

### Exercise 2: Measurement Distribution

Using the simulation code, analyze the measurement distribution for Shor's algorithm with $N = 21$ and $a = 2$ (period $r = 6$).
(a) How many peaks are there in the probability distribution?
(b) For each peak, verify that $m/Q \approx s/r$ for some integer $s$.
(c) What fraction of measurement outcomes successfully yield the correct period via continued fractions?

### Exercise 3: Success Probability Analysis

For $N = pq$ where $p$ and $q$ are distinct odd primes:
(a) Show that there are exactly $\phi(N) = (p-1)(q-1)$ values of $a$ coprime to $N$.
(b) Among these, what fraction gives an even period $r$? (Hint: consider the Chinese Remainder Theorem.)
(c) Simulate Shor's algorithm for $N = 77 = 7 \times 11$ with 100 random choices of $a$. What fraction succeeds on the first try?

### Exercise 4: Post-Quantum Security

Suppose a quantum computer can execute $10^{10}$ gates per second.
(a) Estimate the time to factor RSA-2048 using Shor's algorithm ($O(n^3)$ gates with $n = 2048$).
(b) Estimate the time for RSA-4096.
(c) How many logical qubits are needed for RSA-2048? (Use the estimate $2n + O(n)$.)
(d) If each logical qubit requires 1000 physical qubits for error correction, how many physical qubits are needed?

### Exercise 5: Implement Order Finding for Larger Numbers

Extend the simulation to handle $N = 143 = 11 \times 13$ and $N = 221 = 13 \times 17$:
(a) For each, find all values of $a$ that yield a successful factorization.
(b) Compute the success probability (fraction of valid $a$ values).
(c) How does the measurement distribution change as $N$ grows? (Plot the distributions.)

---

[← Previous: Quantum Fourier Transform](09_Quantum_Fourier_Transform.md) | [Next: Quantum Error Correction →](11_Quantum_Error_Correction.md)

# Mathematics and Number Theory

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply modular arithmetic properties to compute sums, products, and modular inverses under a given modulus without overflow
2. Implement the Euclidean algorithm and extended Euclidean algorithm to compute GCD, LCM, and modular inverses
3. Identify prime numbers using the Sieve of Eratosthenes and implement efficient primality tests
4. Calculate combinations and permutations using Pascal's triangle or factorials with modular inverses
5. Use fast matrix exponentiation to evaluate linear recurrences in O(k³ log n) time
6. Solve competitive programming problems that require number theory techniques such as Euler's totient and the Chinese Remainder Theorem

---

## Overview

This covers mathematical concepts frequently encountered in algorithm problems. Includes modular arithmetic, GCD, prime numbers, combinatorics, and matrix exponentiation.

---

## Table of Contents

1. [Modular Arithmetic](#1-modular-arithmetic)
2. [GCD and LCM](#2-gcd-and-lcm)
3. [Prime Numbers](#3-prime-numbers)
4. [Combinatorics](#4-combinatorics)
5. [Matrix Exponentiation](#5-matrix-exponentiation)
6. [Other Mathematics](#6-other-mathematics)
7. [Practice Problems](#7-practice-problems)

---

## 1. Modular Arithmetic

### 1.1 Basic Properties

```
Basic properties of modular arithmetic (mod m):

(a + b) mod m = ((a mod m) + (b mod m)) mod m
(a - b) mod m = ((a mod m) - (b mod m) + m) mod m
(a * b) mod m = ((a mod m) * (b mod m)) mod m

Note: Division cannot be directly applied! → Modular inverse needed

Common moduli:
- 10^9 + 7 (prime)
- 10^9 + 9 (prime)
- 998244353 (prime, suitable for NTT)
```

### 1.2 Modular Addition/Subtraction/Multiplication

```python
MOD = 10**9 + 7

def mod_add(a, b):
    return (a + b) % MOD

def mod_sub(a, b):
    return (a - b + MOD) % MOD

def mod_mul(a, b):
    return (a * b) % MOD

# Example
a, b = 10**18, 10**18
print(mod_mul(a, b))  # Compute without overflow
```

```cpp
// C++ (watch for overflow)
const long long MOD = 1e9 + 7;

long long mod_add(long long a, long long b) {
    return (a + b) % MOD;
}

long long mod_sub(long long a, long long b) {
    return (a - b % MOD + MOD) % MOD;
}

long long mod_mul(long long a, long long b) {
    return (a % MOD) * (b % MOD) % MOD;
}
```

### 1.3 Fast Exponentiation (Modular Exponentiation)

```
Compute a^n mod m - O(log n)

Idea:
a^8 = (a^4)^2 = ((a^2)^2)^2

a^13 = a^8 * a^4 * a^1  (13 = 1101₂)
```

```python
def mod_pow(base, exp, mod):
    """
    Fast exponentiation (divide and conquer)
    Time: O(log exp)

    Why this works: We decompose exp in binary.  For each bit position k,
    base^(2^k) = (base^(2^(k-1)))^2, so we can build up all needed powers
    with only log(exp) squarings.  We accumulate the product only when the
    corresponding bit of exp is set — exactly the standard binary method.
    """
    result = 1
    base %= mod  # Reduce base upfront to keep all intermediate values < mod

    while exp > 0:
        if exp & 1:  # Current bit is set: multiply the running result by base^(2^k)
            result = (result * base) % mod
        exp >>= 1        # Move to the next bit
        base = (base * base) % mod  # Square: base^(2^k) → base^(2^(k+1))

    return result

# Example
print(mod_pow(2, 10, 1000))  # 1024 % 1000 = 24
print(mod_pow(2, 100, 10**9 + 7))  # Large numbers computed in O(log n)
```

```cpp
// C++
long long mod_pow(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;

    while (exp > 0) {
        if (exp & 1) {
            result = result * base % mod;
        }
        exp >>= 1;
        base = base * base % mod;
    }

    return result;
}
```

### 1.4 Modular Inverse

```
Modular inverse a^(-1) of a:
a * a^(-1) ≡ 1 (mod m)

Condition: gcd(a, m) = 1

Method 1: Fermat's Little Theorem (when m is prime)
a^(-1) ≡ a^(m-2) (mod m)

Method 2: Extended Euclidean Algorithm
```

```python
def mod_inverse(a, mod):
    """
    Modular inverse using Fermat's Little Theorem
    mod must be prime
    """
    return mod_pow(a, mod - 2, mod)

# Division: a / b mod m = a * b^(-1) mod m
def mod_div(a, b, mod):
    return (a * mod_inverse(b, mod)) % mod

# Example
MOD = 10**9 + 7
a, b = 10, 3
print(mod_div(a, b, MOD))  # (10 * 3^(-1)) mod MOD
```

### 1.5 Extended Euclidean Algorithm

```python
def extended_gcd(a, b):
    """
    Return x, y, gcd satisfying ax + by = gcd(a, b)
    """
    if b == 0:
        return a, 1, 0

    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1

    return gcd, x, y

def mod_inverse_ext(a, mod):
    """Modular inverse using extended Euclidean"""
    gcd, x, _ = extended_gcd(a, mod)
    if gcd != 1:
        return -1  # No inverse
    return (x % mod + mod) % mod

# Example
gcd, x, y = extended_gcd(35, 15)
print(f"gcd={gcd}, x={x}, y={y}")  # 35*x + 15*y = 5
print(35 * x + 15 * y)  # 5
```

---

## 2. GCD and LCM

### 2.1 Euclidean Algorithm

```
gcd(a, b) = gcd(b, a mod b)
gcd(a, 0) = a

Example:
gcd(48, 18) = gcd(18, 12) = gcd(12, 6) = gcd(6, 0) = 6
```

```python
def gcd(a, b):
    """Euclidean algorithm - O(log(min(a, b)))"""
    while b:
        a, b = b, a % b
    return a

def gcd_recursive(a, b):
    """Recursive version"""
    return a if b == 0 else gcd_recursive(b, a % b)

# Python built-in
from math import gcd
print(gcd(48, 18))  # 6

# GCD of multiple numbers
from functools import reduce
numbers = [48, 36, 24, 12]
print(reduce(gcd, numbers))  # 12
```

```cpp
// C++
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

// C++17 and above
#include <numeric>
int g = std::gcd(48, 18);  // 6
```

### 2.2 Least Common Multiple (LCM)

```
lcm(a, b) = a * b / gcd(a, b)

Overflow prevention: lcm(a, b) = a / gcd(a, b) * b
```

```python
def lcm(a, b):
    return a // gcd(a, b) * b

# Python 3.9+
from math import lcm
print(lcm(4, 6))  # 12

# LCM of multiple numbers
numbers = [4, 6, 8]
print(reduce(lcm, numbers))  # 24
```

### 2.3 Coprimality Check

```python
def is_coprime(a, b):
    """Check if a and b are coprime"""
    return gcd(a, b) == 1

# Example
print(is_coprime(8, 15))  # True
print(is_coprime(8, 12))  # False (gcd=4)
```

---

## 3. Prime Numbers

### 3.1 Primality Test

```python
def is_prime(n):
    """
    Primality test - O(sqrt(n))
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2

    return True

# Example
print(is_prime(17))  # True
print(is_prime(18))  # False
```

### 3.2 Sieve of Eratosthenes

```
Find all primes up to n - O(n log log n)

Process:
2 3 4 5 6 7 8 9 10 11 12 13 14 15
2 3 ✗ 5 ✗ 7 ✗ 9 ✗  11 ✗  13 ✗  15  (remove multiples of 2)
2 3   5   7   ✗    11    13    ✗   (remove multiples of 3)
2 3   5   7        11    13        (remove multiples of 5 - done)
```

```python
def sieve_of_eratosthenes(n):
    """
    Sieve of Eratosthenes
    Time: O(n log log n)
    Space: O(n)
    """
    if n < 2:
        return []

    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Start marking at i*i, not 2*i: every smaller multiple of i
            # already has a factor < i and was crossed out in an earlier pass.
            # This optimization roughly halves the inner-loop iterations.
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]

# Example
primes = sieve_of_eratosthenes(100)
print(primes)  # [2, 3, 5, 7, 11, 13, ...]
print(len(primes))  # 25
```

```cpp
// C++
vector<int> sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }

    vector<int> primes;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) primes.push_back(i);
    }
    return primes;
}
```

### 3.3 Prime Factorization

```python
def factorize(n):
    """
    Prime factorization
    Time: O(sqrt(n))
    """
    factors = []
    d = 2

    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1

    if n > 1:
        factors.append(n)

    return factors

def factorize_with_count(n):
    """Prime factors with exponents"""
    factors = {}
    d = 2

    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1

    if n > 1:
        factors[n] = 1

    return factors

# Example
print(factorize(60))  # [2, 2, 3, 5]
print(factorize_with_count(60))  # {2: 2, 3: 1, 5: 1}
```

### 3.4 Fast Factorization (Preprocessing)

```python
def precompute_spf(n):
    """
    Precompute Smallest Prime Factor
    """
    spf = list(range(n + 1))

    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i

    return spf

def fast_factorize(n, spf):
    """O(log n) factorization using precomputed SPF"""
    factors = []
    while n > 1:
        factors.append(spf[n])
        n //= spf[n]
    return factors

# Example
spf = precompute_spf(100)
print(fast_factorize(60, spf))  # [2, 2, 3, 5]
```

### 3.5 Finding Divisors

```python
def get_divisors(n):
    """
    All divisors of n
    Time: O(sqrt(n))
    """
    divisors = []

    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1

    divisors.sort()
    return divisors

# Example
print(get_divisors(36))  # [1, 2, 3, 4, 6, 9, 12, 18, 36]
```

---

## 4. Combinatorics

### 4.1 Factorial and Inverse Factorial

```python
MOD = 10**9 + 7
MAX_N = 10**6

# Preprocessing: build factorial and inverse-factorial tables once in O(N).
# This lets every nCr / nPr query run in O(1) — critical when thousands of
# queries arrive in a single test case.
factorial = [1] * (MAX_N + 1)
inv_factorial = [1] * (MAX_N + 1)

for i in range(1, MAX_N + 1):
    factorial[i] = factorial[i - 1] * i % MOD

# Compute inv_factorial[MAX_N] with one modular exponentiation (Fermat's theorem),
# then fill downward using inv_factorial[i] = inv_factorial[i+1] * (i+1) % MOD.
# This avoids MAX_N separate modular exponentiations (each O(log MOD)).
inv_factorial[MAX_N] = mod_pow(factorial[MAX_N], MOD - 2, MOD)
for i in range(MAX_N - 1, -1, -1):
    inv_factorial[i] = inv_factorial[i + 1] * (i + 1) % MOD

def nCr(n, r):
    """Binomial coefficient nCr - O(1) (after preprocessing)"""
    if r < 0 or r > n:
        return 0
    # C(n,r) = n! / (r! * (n-r)!) ; division becomes multiplication by
    # modular inverse so the result stays in [0, MOD).
    return factorial[n] * inv_factorial[r] % MOD * inv_factorial[n - r] % MOD

def nPr(n, r):
    """Permutation nPr - O(1) (after preprocessing)"""
    if r < 0 or r > n:
        return 0
    return factorial[n] * inv_factorial[n - r] % MOD

# Example
print(nCr(10, 3))  # 120
print(nPr(10, 3))  # 720
```

### 4.2 Pascal's Triangle

```
          1
        1   1
      1   2   1
    1   3   3   1
  1   4   6   4   1

C(n, r) = C(n-1, r-1) + C(n-1, r)
```

```python
def pascal_triangle(n):
    """
    Generate Pascal's triangle
    Time: O(n²)
    """
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = dp[i][i] = 1
        for j in range(1, i):
            dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % MOD

    return dp

# Example
C = pascal_triangle(100)
print(C[10][3])  # 120
```

### 4.3 Catalan Numbers

```
Catalan number C_n:
- Number of valid parentheses pairs
- Number of binary trees with n+1 leaves
- Number of triangulations of convex n+2-gon

C_0 = 1
C_n = C(2n, n) / (n + 1) = C(2n, n) - C(2n, n+1)

C_n: 1, 1, 2, 5, 14, 42, 132, ...
```

```python
def catalan(n):
    """
    Catalan number C_n
    """
    return nCr(2 * n, n) * mod_inverse(n + 1, MOD) % MOD

def catalan_dp(n):
    """DP approach"""
    if n <= 1:
        return 1

    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1

    for i in range(2, n + 1):
        for j in range(i):
            dp[i] = (dp[i] + dp[j] * dp[i - 1 - j]) % MOD

    return dp[n]

# Example
for i in range(10):
    print(catalan(i), end=" ")  # 1 1 2 5 14 42 132 429 1430 4862
```

### 4.4 Combinations and Permutations with Repetition

```python
def nHr(n, r):
    """
    Combinations with repetition: choose r from n (repetition allowed)
    nHr = C(n+r-1, r)
    """
    return nCr(n + r - 1, r)

def repeated_permutation(n, r):
    """
    Permutations with repetition: arrange r from n (repetition allowed)
    n^r
    """
    return mod_pow(n, r, MOD)

# Example
print(nHr(3, 2))  # 6: (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
print(repeated_permutation(3, 2))  # 9: 3^2
```

---

## 5. Matrix Exponentiation

### 5.1 Matrix Multiplication

```python
def matrix_mult(A, B, mod):
    """
    Matrix multiplication A * B
    Time: O(n³)
    """
    n = len(A)
    m = len(B[0])
    k = len(B)

    C = [[0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i][j] = (C[i][j] + A[i][l] * B[l][j]) % mod

    return C
```

### 5.2 Matrix Exponentiation

```python
def matrix_pow(M, n, mod):
    """
    Matrix M to the power n
    Time: O(k³ log n) (k = matrix size)

    Why the identity matrix as the starting value: multiplying any matrix
    by the identity leaves it unchanged, so `result` starts as M^0 = I.
    This mirrors scalar fast exponentiation where `result` starts at 1.
    The same repeated-squaring trick applies: O(log n) matrix multiplications
    each costing O(k³), giving O(k³ log n) overall.
    """
    size = len(M)
    # Identity matrix: I[i][j] = 1 iff i == j
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    while n > 0:
        if n & 1:
            # Current bit set: accumulate M^(2^k) into the result
            result = matrix_mult(result, M, mod)
        M = matrix_mult(M, M, mod)  # Square: M^(2^k) → M^(2^(k+1))
        n >>= 1

    return result
```

### 5.3 Fibonacci in O(log n)

```
Matrix representation of Fibonacci recurrence:

[F(n+1)]   [1 1]^n   [F(1)]
[F(n)  ] = [1 0]   * [F(0)]

→ Compute F(n) in O(log n)
```

```python
def fibonacci_matrix(n, mod=10**9 + 7):
    """
    Fibonacci number F(n) - O(log n)
    """
    if n <= 1:
        return n

    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n, mod)

    return result[1][0]

# Example
for i in range(15):
    print(fibonacci_matrix(i), end=" ")
# 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377

# Large number computation
print(fibonacci_matrix(10**18))  # Computed in O(log n)
```

### 5.4 General Linear Recurrence

```
Recurrence: f(n) = a*f(n-1) + b*f(n-2) + c*f(n-3)

Matrix representation:
[f(n)  ]   [a b c]^(n-2)   [f(2)]
[f(n-1)] = [1 0 0]       * [f(1)]
[f(n-2)]   [0 1 0]         [f(0)]
```

```python
def solve_linear_recurrence(coeffs, initial, n, mod):
    """
    Solve linear recurrence
    coeffs: [a, b, c, ...] (f(n) = a*f(n-1) + b*f(n-2) + ...)
    initial: [f(0), f(1), f(2), ...] (k values)
    """
    k = len(coeffs)

    if n < k:
        return initial[n]

    # Build transition matrix
    M = [[0] * k for _ in range(k)]
    for j in range(k):
        M[0][j] = coeffs[j]
    for i in range(1, k):
        M[i][i - 1] = 1

    result = matrix_pow(M, n - k + 1, mod)

    ans = 0
    for j in range(k):
        ans = (ans + result[0][j] * initial[k - 1 - j]) % mod

    return ans

# Example: f(n) = f(n-1) + f(n-2), f(0)=0, f(1)=1
print(solve_linear_recurrence([1, 1], [0, 1], 10, 10**9 + 7))  # 55
```

---

## 6. Other Mathematics

### 6.1 Euler's Totient Function

```
φ(n) = count of numbers <= n that are coprime with n

φ(p) = p - 1  (p is prime)
φ(p^k) = p^k - p^(k-1)
φ(mn) = φ(m) * φ(n)  (m, n are coprime)
```

```python
def euler_phi(n):
    """
    Euler's totient function φ(n)
    Time: O(sqrt(n))
    """
    result = n
    p = 2

    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1

    if n > 1:
        result -= result // n

    return result

# Example
print(euler_phi(12))  # 4 (1, 5, 7, 11)
print(euler_phi(7))   # 6 (1, 2, 3, 4, 5, 6)
```

### 6.2 Lucas' Theorem

```
When p is prime:
C(m, n) mod p = Π C(m_i, n_i) mod p

Product of binomial coefficients of each digit when m and n are in base p
```

```python
def lucas(m, n, p):
    """
    Lucas' Theorem: C(m, n) mod p
    Use when p is prime
    """
    def nCr_small(a, b, p):
        if b > a:
            return 0
        if b == 0 or a == b:
            return 1

        num = den = 1
        for i in range(b):
            num = num * (a - i) % p
            den = den * (i + 1) % p

        return num * mod_pow(den, p - 2, p) % p

    result = 1
    while m > 0 or n > 0:
        mi, ni = m % p, n % p
        result = result * nCr_small(mi, ni, p) % p
        m //= p
        n //= p

    return result

# Example
print(lucas(1000, 500, 13))  # C(1000, 500) mod 13
```

### 6.3 Chinese Remainder Theorem (CRT)

```
System of congruences:
x ≡ a1 (mod m1)
x ≡ a2 (mod m2)
...

When m1, m2, ... are pairwise coprime, unique solution exists (mod M, M = m1*m2*...)
```

```python
def chinese_remainder_theorem(remainders, moduli):
    """
    Chinese Remainder Theorem
    remainders: [a1, a2, ...]
    moduli: [m1, m2, ...]  (pairwise coprime)
    """
    M = 1
    for m in moduli:
        M *= m

    result = 0
    for a, m in zip(remainders, moduli):
        Mi = M // m
        _, y, _ = extended_gcd(Mi, m)
        result = (result + a * Mi * y) % M

    return (result + M) % M

# Example
# x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
print(chinese_remainder_theorem([2, 3, 2], [3, 5, 7]))  # 23
```

---

## 7. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐ | [Binomial Coefficient 1](https://www.acmicpc.net/problem/11050) | BOJ | Basic combo |
| ⭐⭐ | [GCD and LCM](https://www.acmicpc.net/problem/2609) | BOJ | GCD/LCM |
| ⭐⭐ | [Find Primes](https://www.acmicpc.net/problem/1978) | BOJ | Primality |
| ⭐⭐⭐ | [Binomial Coefficient 3](https://www.acmicpc.net/problem/11401) | BOJ | Mod inverse |
| ⭐⭐⭐ | [Fibonacci 6](https://www.acmicpc.net/problem/11444) | BOJ | Matrix exp |
| ⭐⭐⭐ | [Goldbach's Conjecture](https://www.acmicpc.net/problem/9020) | BOJ | Sieve |
| ⭐⭐⭐⭐ | [Binomial Coefficient 4](https://www.acmicpc.net/problem/11402) | BOJ | Lucas |

---

## Time Complexity Summary

```
┌─────────────────────┬─────────────────┐
│ Algorithm           │ Time Complexity │
├─────────────────────┼─────────────────┤
│ Fast exponentiation │ O(log n)        │
│ GCD (Euclidean)     │ O(log min(a,b)) │
│ Primality test      │ O(√n)           │
│ Sieve of Erat.      │ O(n log log n)  │
│ Prime factorization │ O(√n)           │
│ Binomial (preproc.) │ O(1)            │
│ Matrix mult (k×k)   │ O(k³)           │
│ Matrix exp          │ O(k³ log n)     │
└─────────────────────┴─────────────────┘
```

---

## Next Steps

- [22_String_Algorithms.md](./22_String_Algorithms.md) - String Algorithms

---

## References

- [Modular Arithmetic](https://cp-algorithms.com/algebra/module-inverse.html)
- Introduction to Algorithms (CLRS) - Chapter 31

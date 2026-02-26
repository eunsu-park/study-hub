# 레슨 1: 수론 기초

**다음**: [대칭 암호](./02_Symmetric_Ciphers.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 모듈러 산술(Modular Arithmetic) 연산을 수행하고 합동류(Congruence Class)에 대해 추론할 수 있습니다
2. 확장 유클리드 알고리즘(Extended Euclidean Algorithm)을 구현하고 모듈러 역원(Modular Inverse)을 계산할 수 있습니다
3. 오일러 정리(Euler's Theorem)와 페르마 소정리(Fermat's Little Theorem)를 적용하여 모듈러 거듭제곱을 단순화할 수 있습니다
4. 중국인의 나머지 정리(Chinese Remainder Theorem)를 설명하고 적용할 수 있습니다
5. 밀러-라빈 소수판별법(Miller-Rabin Primality Test)을 구현하고 그 확률적 보장을 설명할 수 있습니다
6. 이산 로그 문제(Discrete Logarithm Problem)가 계산적으로 어려운 이유를 설명할 수 있습니다

---

현대 암호학은 놀랍도록 작은 수론적 결과들의 집합 위에 세워져 있습니다. HTTPS로 웹사이트에 접속할 때마다, 브라우저는 밀리초 이내에 모듈러 거듭제곱을 계산하고, 소수를 확인하며, 모듈러 역원을 구합니다. 이 레슨은 RSA(레슨 5), 디피-헬만 키 교환(Diffie-Hellman Key Exchange), 타원 곡선 암호(Elliptic Curve Cryptography, 레슨 6)를 구동하는 수학적 도구 세트를 구축합니다. 이 기초를 완전히 익히면, 암호학의 나머지 부분은 우아한 응용들의 연속이 됩니다.

## 목차

1. [모듈러 산술](#1-모듈러-산술)
2. [최대공약수와 확장 유클리드 알고리즘](#2-최대공약수와-확장-유클리드-알고리즘)
3. [모듈러 역원](#3-모듈러-역원)
4. [오일러 피 함수](#4-오일러-피-함수)
5. [오일러 정리와 페르마 소정리](#5-오일러-정리와-페르마-소정리)
6. [중국인의 나머지 정리](#6-중국인의-나머지-정리)
7. [고속 모듈러 거듭제곱](#7-고속-모듈러-거듭제곱)
8. [소수 판별](#8-소수-판별)
9. [이산 로그 문제](#9-이산-로그-문제)
10. [연습 문제](#10-연습-문제)

---

## 1. 모듈러 산술

> **비유:** 모듈러 산술은 시계와 같습니다 — 12 다음에는 13이 아니라 1이 옵니다. 지금 10시이고 5시간을 기다리면 시계는 15가 아닌 3을 가리킵니다. 이를 $15 \equiv 3 \pmod{12}$라고 씁니다.

### 1.1 정의

정수 $a$, $b$, 그리고 양의 정수 $n$에 대해, $n$이 $(a - b)$를 나눌 때, 즉 $n \mid (a - b)$일 때, $a$는 $n$을 법으로 $b$와 **합동(Congruent)**이라 하며 다음과 같이 씁니다:

$$a \equiv b \pmod{n}$$

동치적으로, $a$와 $b$를 $n$으로 나눌 때 나머지가 같습니다.

### 1.2 성질

모듈러 산술은 기본 연산들을 보존합니다:

| 성질 | 내용 |
|------|------|
| 덧셈 | $a \equiv b$이고 $c \equiv d \pmod{n}$이면, $a + c \equiv b + d \pmod{n}$ |
| 뺄셈 | $a \equiv b$이고 $c \equiv d \pmod{n}$이면, $a - c \equiv b - d \pmod{n}$ |
| 곱셈 | $a \equiv b$이고 $c \equiv d \pmod{n}$이면, $ac \equiv bd \pmod{n}$ |
| 거듭제곱 | $a \equiv b \pmod{n}$이면, $k \geq 0$에 대해 $a^k \equiv b^k \pmod{n}$ |

**나눗셈은 항상 성립하지 않습니다.** 예를 들어, $6 \equiv 0 \pmod{6}$이고 $3 \equiv 3 \pmod{6}$이지만, $6/3 = 2 \not\equiv 0/3 = 0 \pmod{6}$입니다. 나눗셈을 하려면 **모듈러 역원(Modular Inverse)**이 필요하며, 이는 3절에서 다룹니다.

### 1.3 집합 $\mathbb{Z}_n$

$n$을 법으로 하는 **잉여류(Residue Class)**들은 다음 집합을 이룹니다:

$$\mathbb{Z}_n = \{0, 1, 2, \ldots, n-1\}$$

이 집합은 $n$을 법으로 하는 덧셈과 곱셈 아래 **환(Ring)**을 이룹니다. 곱셈 역원이 존재하는 원소들로 제한하면:

$$\mathbb{Z}_n^* = \{a \in \mathbb{Z}_n \mid \gcd(a, n) = 1\}$$

이것은 **곱셈군(Multiplicative Group)**입니다. 그 위수(Order)는 $|\mathbb{Z}_n^*| = \phi(n)$이며, 여기서 $\phi$는 오일러 피 함수(4절)입니다.

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

## 2. 최대공약수와 확장 유클리드 알고리즘

### 2.1 최대공약수(Greatest Common Divisor)

**최대공약수** $\gcd(a, b)$는 $a$와 $b$ 모두를 나누는 가장 큰 양의 정수입니다.

**유클리드 알고리즘(Euclidean Algorithm)** — $\gcd(a, b) = \gcd(b, a \bmod b)$ 항등식에 기반합니다:

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

**시간 복잡도:** $O(\log(\min(a, b)))$ — 단계 수는 작은 수의 자릿수의 최대 5배입니다(라메 정리, Lamé's Theorem).

### 2.2 확장 유클리드 알고리즘(Extended Euclidean Algorithm)

**확장 유클리드 알고리즘**은 다음을 만족하는 정수 $x$와 $y$를 구합니다:

$$ax + by = \gcd(a, b)$$

이 등식은 **베주 항등식(Bezout's Identity)**으로 알려져 있습니다. $x$와 $y$를 구하는 것은 모듈러 역원 계산에 필수적입니다.

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

### 2.3 반복 버전

프로덕션 코드에서는 재귀 깊이 한계를 피하기 위해 반복 버전을 사용합니다:

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

## 3. 모듈러 역원

### 3.1 정의

$n$을 법으로 하는 $a$의 **모듈러 역원(Modular Inverse)**은 다음을 만족하는 정수 $a^{-1}$입니다:

$$a \cdot a^{-1} \equiv 1 \pmod{n}$$

모듈러 역원은 $\gcd(a, n) = 1$인 경우, 즉 $a$와 $n$이 서로소(Coprime)인 경우에만 존재합니다.

### 3.2 역원 계산

베주 항등식으로부터, $\gcd(a, n) = 1$이면 $ax + ny = 1$을 만족하는 $x, y$가 존재합니다. 양변을 $n$으로 나누면:

$$ax \equiv 1 \pmod{n}$$

따라서 $x \bmod n$이 $a$의 모듈러 역원입니다.

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

### 3.3 Python 내장 함수

Python 3.8+에서는 모듈러 역원을 위해 `pow(a, -1, n)`을 제공합니다:

```python
# Built-in modular inverse (uses the same algorithm internally)
print(pow(7, -1, 11))  # 8, because 7*8 = 56 ≡ 1 (mod 11)
```

---

## 4. 오일러 피 함수

### 4.1 정의

**오일러 피 함수(Euler's Totient Function)** $\phi(n)$은 $\{1, 2, \ldots, n\}$ 중 $n$과 서로소인 정수의 개수를 셉니다:

$$\phi(n) = |\{k : 1 \leq k \leq n, \gcd(k, n) = 1\}|$$

### 4.2 주요 공식

| 경우 | 공식 | 예시 |
|------|------|------|
| $p$가 소수 | $\phi(p) = p - 1$ | $\phi(7) = 6$ |
| $p^k$ (소수의 거듭제곱) | $\phi(p^k) = p^k - p^{k-1} = p^{k-1}(p-1)$ | $\phi(8) = 4$ |
| $\gcd(m,n) = 1$ | $\phi(mn) = \phi(m)\phi(n)$ (곱셈적) | $\phi(12) = \phi(4)\phi(3) = 2 \cdot 2 = 4$ |
| 일반 | $\phi(n) = n \prod_{p \mid n}\left(1 - \frac{1}{p}\right)$ | $\phi(12) = 12(1-\frac{1}{2})(1-\frac{1}{3}) = 4$ |

$p, q$가 서로 다른 소수일 때 $n = pq$에 대한 공식은 RSA에서 특히 중요합니다:

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

## 5. 오일러 정리와 페르마 소정리

### 5.1 오일러 정리

$\gcd(a, n) = 1$이면:

$$a^{\phi(n)} \equiv 1 \pmod{n}$$

> **비유:** $a$를 $n$을 법으로 거듭제곱한다고 상상해 보세요. 오일러 정리는 이 과정이 **순환적(Cyclic)**임을 보장합니다 — 정확히 $\phi(n)$번의 곱셈 이후(또는 그 약수만큼), 1로 돌아옵니다. 마치 길이 $\phi(n)$의 원형 트랙을 걷는 것과 같습니다.

**RSA에서 왜 중요한가:** $ed \equiv 1 \pmod{\phi(n)}$이면, $n$과 서로소인 임의의 메시지 $m$에 대해:

$$m^{ed} = m^{1 + k\phi(n)} = m \cdot (m^{\phi(n)})^k \equiv m \cdot 1^k = m \pmod{n}$$

이것이 RSA 암호화와 복호화의 수학적 토대입니다.

### 5.2 페르마 소정리

$n = p$가 소수일 때($\phi(p) = p - 1$) 오일러 정리의 특수한 경우:

$$a^{p-1} \equiv 1 \pmod{p} \quad \text{단, } \gcd(a, p) = 1$$

동치적으로:

$$a^p \equiv a \pmod{p} \quad \text{모든 } a \text{에 대해}$$

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

### 5.3 오일러 정리를 이용한 모듈러 역원 계산

$a^{\phi(n)} \equiv 1 \pmod{n}$이므로:

$$a^{-1} \equiv a^{\phi(n)-1} \pmod{n}$$

$n$이 소수이면 이는 $a^{-1} \equiv a^{n-2} \pmod{n}$으로 단순화됩니다.

```python
# Alternative modular inverse using Fermat's little theorem (only for prime modulus)
p = 17
a = 5
inverse = pow(a, p - 2, p)
print(f"{a}^(-1) mod {p} = {inverse}")
print(f"Verification: {a} * {inverse} mod {p} = {(a * inverse) % p}")
```

---

## 6. 중국인의 나머지 정리

### 6.1 내용

$n_1, n_2, \ldots, n_k$가 쌍마다 서로소(Pairwise Coprime), 즉 $i \neq j$에 대해 $\gcd(n_i, n_j) = 1$이면, 다음 연립 합동식은:

$$x \equiv a_1 \pmod{n_1}$$
$$x \equiv a_2 \pmod{n_2}$$
$$\vdots$$
$$x \equiv a_k \pmod{n_k}$$

$N = n_1 n_2 \cdots n_k$를 법으로 하는 **유일한** 해를 가집니다.

> **비유:** 어떤 수의 3, 5, 7로 나눈 나머지만 알고 있다고 상상해 보세요. 중국인의 나머지 정리(CRT)는 이 세 나머지가 105($= 3 \times 5 \times 7$)를 법으로 그 수를 유일하게 결정한다고 말합니다. 세 가지 독립적인 특성으로 사람을 식별하는 것과 같습니다 — 특성들이 "독립적"(서로소인 법)이면, 그 조합은 유일합니다.

### 6.2 구성

$N = \prod_{i=1}^k n_i$이고 $N_i = N / n_i$라 할 때:

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

### 6.3 RSA-CRT 최적화

RSA에서는 $m = c^d \bmod n$을 직접 계산하는 대신 다음을 계산합니다:

$$m_p = c^{d \bmod (p-1)} \bmod p$$
$$m_q = c^{d \bmod (q-1)} \bmod q$$

그런 다음 CRT를 사용하여 결합합니다. $p$와 $q$는 $n$의 절반 크기이고, 모듈러 거듭제곱의 비트 길이 $k$에 대한 복잡도는 $O(k^3)$이므로, 이 방법은 약 **4배의 속도 향상**을 제공합니다.

---

## 7. 고속 모듈러 거듭제곱

### 7.1 문제

$b$가 수백 자리일 때 $a$를 $b$번 곱하는 방식으로 $a^b \bmod n$을 나이브하게 계산하는 것은 불가능할 정도로 느립니다. $O(\log b)$ 번의 곱셈으로 실행되는 방법이 필요합니다.

### 7.2 제곱-곱셈 알고리즘(Square-and-Multiply Algorithm)

핵심 아이디어: $b$를 이진수로 표현하고 **반복 제곱(Repeated Squaring)**을 사용합니다.

$b = b_k b_{k-1} \cdots b_1 b_0$을 이진수로 표현하면:

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

### 7.3 Python의 `pow(base, exp, mod)`를 권장하는 이유

Python의 내장 세 인수 `pow()`는 모듈러 거듭제곱을 위해 최적화된 C 코드를 사용합니다. 프로덕션 코드에서는 항상 수동 구현보다 이를 선호하세요. 위의 수동 버전은 알고리즘 이해를 위한 것입니다.

---

## 8. 소수 판별

### 8.1 소수가 중요한 이유

RSA의 보안은 $n = pq$ ($p$, $q$는 큰 소수)의 인수분해 어려움에 의존합니다. 이 소수들을 생성하려면 효율적인 소수 판별법이 필요합니다.

### 8.2 시험 나눗셈(Trial Division)

$\sqrt{n}$까지의 모든 정수로 나누어 떨어지는지 확인합니다:

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

### 8.3 페르마 소수 판별법(Fermat Primality Test)

페르마 소정리에 의하면, $p$가 소수이면 $a^{p-1} \equiv 1 \pmod{p}$입니다. **대우**: $a^{n-1} \not\equiv 1 \pmod{n}$이면, $n$은 **확실히 합성수(Composite)**입니다.

**문제:** 카마이클 수(Carmichael Number, 예: 561 = 3 * 11 * 17)는 합성수임에도 불구하고 $n$과 서로소인 모든 $a$에 대해 $a^{n-1} \equiv 1 \pmod{n}$을 만족합니다. 더 강력한 판별법이 필요합니다.

### 8.4 밀러-라빈 소수 판별법(Miller-Rabin Primality Test)

$n - 1 = 2^s \cdot d$ ($d$는 홀수)로 씁니다. 무작위로 선택된 증인(Witness) $a$에 대해 다음을 계산합니다:

$$a^d \bmod n$$

그런 다음 결과를 반복적으로 제곱합니다. $n$이 소수이면, 수열 $a^d, a^{2d}, a^{4d}, \ldots, a^{2^s d}$는 다음 중 하나를 만족해야 합니다:
- $a^d \equiv 1 \pmod{n}$, **또는**
- $0 \leq r < s$인 어떤 $r$에 대해 $a^{2^r d} \equiv -1 \pmod{n}$

둘 다 만족하지 않으면, $n$은 **확실히 합성수**입니다(증인 $a$가 합성수임을 "포착").

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

### 8.5 큰 소수 생성

$k$비트 임의 소수를 생성하는 방법:

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

## 9. 이산 로그 문제

### 9.1 정의

소수 $p$, $\mathbb{Z}_p^*$의 생성원(Generator) $g$, 그리고 원소 $h \in \mathbb{Z}_p^*$가 주어졌을 때, **이산 로그 문제(Discrete Logarithm Problem, DLP)**는 다음을 묻습니다:

$$g^x \equiv h \pmod{p} \text{를 만족하는 } x \text{를 구하시오}$$

> **비유:** $g^x \bmod p$를 계산하는 것은 물감을 섞는 것과 같습니다 — 정방향은 쉽습니다(빨간색과 파란색을 섞어 보라색 만들기), 하지만 역방향은 극도로 어렵습니다(보라색이 주어졌을 때 빨간색과 파란색의 정확한 양 결정하기). 이 **일방향(One-Way)** 성질이 디피-헬만 키 교환(Diffie-Hellman Key Exchange)과 DSA/ECDSA 서명의 토대입니다.

### 9.2 어려운 이유

- **정방향** (거듭제곱): 제곱-곱셈을 사용하면 $O(\log x)$번의 곱셈
- **역방향** (이산 로그): 알려진 최선의 고전 알고리즘들은 준지수적(Sub-Exponential):
  - 아기걸음-거인걸음(Baby-Step Giant-Step): $O(\sqrt{p})$ 시간 및 공간
  - 지수 계산법(Index Calculus): $O(\exp(c \cdot (\log p)^{1/3} (\log \log p)^{2/3}))$

2048비트 소수에 대해서는 현재 기술로 지수 계산법도 실행 불가능합니다.

### 9.3 아기걸음-거인걸음 알고리즘(Baby-Step Giant-Step Algorithm)

중간 만남(Meet-in-the-Middle) 방식으로 $O(\sqrt{p})$에 DLP를 풉니다:

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

### 9.4 군과 생성원(Groups and Generators)

$g \in \mathbb{Z}_p^*$가 집합 $\{g^0, g^1, g^2, \ldots, g^{p-2}\}$가 $\mathbb{Z}_p^*$와 같으면 **생성원(Generator)** 또는 **원시근(Primitive Root)**이라 합니다. 모든 원소가 생성원인 것은 아니며 — $\mathbb{Z}_p^*$의 생성원 수는 $\phi(p-1)$개입니다.

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

## 10. 연습 문제

### 연습 문제 1: 모듈러 산술 워밍업 (기초)

손으로 계산한 후 Python으로 검증하세요:
1. $17^3 \bmod 13$
2. $7^{-1} \bmod 31$
3. $\phi(360)$

### 연습 문제 2: 확장 유클리드 알고리즘 적용 (중급)

확장 유클리드 알고리즘을 사용하여 $1234x + 567y = \gcd(1234, 567)$을 푸세요. 그런 다음 (존재한다면) $1234^{-1} \bmod 567$을 구하세요.

### 연습 문제 3: 중국인의 나머지 정리 문제 (중급)

어떤 수를 3으로 나누면 나머지가 2, 5로 나누면 나머지가 3, 7로 나누면 나머지가 4입니다. 다음 두 방법으로 세 조건을 모두 만족하는 가장 작은 양의 정수를 구하세요:
1. 6절의 중국인의 나머지 정리 알고리즘 사용
2. 브루트 포스(검증용)

### 연습 문제 4: 밀러-라빈 분석 (심화)

1. 561이 카마이클 수임을 보여주세요: $a \in \{2, 5, 10\}$에 대해 $a^{560} \equiv 1 \pmod{561}$임을 확인하세요.
2. 밀러-라빈이 561을 합성수로 올바르게 판별하는 것을 보여주세요. 561을 노출하는 증인 $a$를 찾으세요.
3. 거짓 양성(False Positive) 확률을 $2^{-128}$ 미만으로 달성하려면 몇 번의 밀러-라빈 라운드가 필요한가요?

### 연습 문제 5: 이산 로그 (심화)

1. $\mathbb{Z}_{23}^*$의 모든 생성원을 구하세요.
2. 아기걸음-거인걸음 알고리즘을 사용하여 $5^x \equiv 8 \pmod{23}$을 푸세요.
3. 256비트 소수 $p$를 사용한 $\mathbb{Z}_p^*$에서의 DLP는 오늘날 왜 안전하지 않다고 여겨지는 반면, 2048비트 소수는 안전하다고 여겨지는지 설명하세요.

---

**다음**: [대칭 암호](./02_Symmetric_Ciphers.md)

# 수학과 정수론 (Mathematics and Number Theory)

## 개요

알고리즘 문제에서 자주 등장하는 수학적 개념들을 다룹니다. 모듈러 연산, 최대공약수, 소수, 조합론, 행렬 거듭제곱 등이 포함됩니다.

---

## 목차

1. [모듈러 연산](#1-모듈러-연산)
2. [최대공약수와 최소공배수](#2-최대공약수와-최소공배수)
3. [소수](#3-소수)
4. [조합론](#4-조합론)
5. [행렬 거듭제곱](#5-행렬-거듭제곱)
6. [기타 수학](#6-기타-수학)
7. [연습 문제](#7-연습-문제)

---

## 1. 모듈러 연산

### 1.1 기본 성질

```
모듈러 연산 기본 성질 (mod m):

(a + b) mod m = ((a mod m) + (b mod m)) mod m
(a - b) mod m = ((a mod m) - (b mod m) + m) mod m
(a * b) mod m = ((a mod m) * (b mod m)) mod m

주의: 나눗셈은 직접 적용 불가! → 모듈러 역원 필요

자주 쓰는 모듈러:
- 10^9 + 7 (소수)
- 10^9 + 9 (소수)
- 998244353 (소수, NTT에 적합)
```

### 1.2 모듈러 덧셈/뺄셈/곱셈

```python
MOD = 10**9 + 7

def mod_add(a, b):
    return (a + b) % MOD

def mod_sub(a, b):
    return (a - b + MOD) % MOD

def mod_mul(a, b):
    return (a * b) % MOD

# 예시
a, b = 10**18, 10**18
print(mod_mul(a, b))  # 오버플로우 없이 계산
```

```cpp
// C++ (오버플로우 주의)
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

### 1.3 빠른 거듭제곱 (Modular Exponentiation)

```
a^n mod m 계산 - O(log n)

아이디어:
a^8 = (a^4)^2 = ((a^2)^2)^2

a^13 = a^8 * a^4 * a^1  (13 = 1101₂)
```

```python
def mod_pow(base, exp, mod):
    """
    빠른 거듭제곱 (분할정복)
    시간: O(log exp)
    """
    result = 1
    base %= mod

    while exp > 0:
        if exp & 1:  # exp가 홀수면
            result = (result * base) % mod
        exp >>= 1  # exp //= 2
        base = (base * base) % mod

    return result

# 예시
print(mod_pow(2, 10, 1000))  # 1024 % 1000 = 24
print(mod_pow(2, 100, 10**9 + 7))  # 큰 수도 O(log n)에 계산
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

### 1.4 모듈러 역원 (Modular Inverse)

```
a의 모듈러 역원 a^(-1):
a * a^(-1) ≡ 1 (mod m)

조건: gcd(a, m) = 1

방법 1: 페르마 소정리 (m이 소수일 때)
a^(-1) ≡ a^(m-2) (mod m)

방법 2: 확장 유클리드 알고리즘
```

```python
def mod_inverse(a, mod):
    """
    페르마 소정리를 이용한 모듈러 역원
    mod가 소수여야 함
    """
    return mod_pow(a, mod - 2, mod)

# 나눗셈: a / b mod m = a * b^(-1) mod m
def mod_div(a, b, mod):
    return (a * mod_inverse(b, mod)) % mod

# 예시
MOD = 10**9 + 7
a, b = 10, 3
print(mod_div(a, b, MOD))  # (10 * 3^(-1)) mod MOD
```

### 1.5 확장 유클리드 알고리즘

```python
def extended_gcd(a, b):
    """
    ax + by = gcd(a, b)를 만족하는 x, y와 gcd 반환
    """
    if b == 0:
        return a, 1, 0

    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1

    return gcd, x, y

def mod_inverse_ext(a, mod):
    """확장 유클리드를 이용한 모듈러 역원"""
    gcd, x, _ = extended_gcd(a, mod)
    if gcd != 1:
        return -1  # 역원 없음
    return (x % mod + mod) % mod

# 예시
gcd, x, y = extended_gcd(35, 15)
print(f"gcd={gcd}, x={x}, y={y}")  # 35*x + 15*y = 5
print(35 * x + 15 * y)  # 5
```

---

## 2. 최대공약수와 최소공배수

### 2.1 유클리드 호제법

```
gcd(a, b) = gcd(b, a mod b)
gcd(a, 0) = a

예시:
gcd(48, 18) = gcd(18, 12) = gcd(12, 6) = gcd(6, 0) = 6
```

```python
def gcd(a, b):
    """유클리드 호제법 - O(log(min(a, b)))"""
    while b:
        a, b = b, a % b
    return a

def gcd_recursive(a, b):
    """재귀 버전"""
    return a if b == 0 else gcd_recursive(b, a % b)

# Python 내장 함수
from math import gcd
print(gcd(48, 18))  # 6

# 여러 수의 GCD
from functools import reduce
numbers = [48, 36, 24, 12]
print(reduce(gcd, numbers))  # 12
```

```cpp
// C++
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

// C++17 이상
#include <numeric>
int g = std::gcd(48, 18);  // 6
```

### 2.2 최소공배수 (LCM)

```
lcm(a, b) = a * b / gcd(a, b)

오버플로우 방지: lcm(a, b) = a / gcd(a, b) * b
```

```python
def lcm(a, b):
    return a // gcd(a, b) * b

# Python 3.9+
from math import lcm
print(lcm(4, 6))  # 12

# 여러 수의 LCM
numbers = [4, 6, 8]
print(reduce(lcm, numbers))  # 24
```

### 2.3 서로소 판별

```python
def is_coprime(a, b):
    """a와 b가 서로소인지 확인"""
    return gcd(a, b) == 1

# 예시
print(is_coprime(8, 15))  # True
print(is_coprime(8, 12))  # False (gcd=4)
```

---

## 3. 소수

### 3.1 소수 판별

```python
def is_prime(n):
    """
    소수 판별 - O(√n)
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

# 예시
print(is_prime(17))  # True
print(is_prime(18))  # False
```

### 3.2 에라토스테네스의 체

```
n 이하의 모든 소수 찾기 - O(n log log n)

과정:
2 3 4 5 6 7 8 9 10 11 12 13 14 15
2 3 ✗ 5 ✗ 7 ✗ 9 ✗  11 ✗  13 ✗  15  (2의 배수 제거)
2 3   5   7   ✗    11    13    ✗   (3의 배수 제거)
2 3   5   7        11    13        (5의 배수 제거 - 완료)
```

```python
def sieve_of_eratosthenes(n):
    """
    에라토스테네스의 체
    시간: O(n log log n)
    공간: O(n)
    """
    if n < 2:
        return []

    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]

# 예시
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

### 3.3 소인수분해

```python
def factorize(n):
    """
    소인수분해
    시간: O(√n)
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
    """소인수와 지수"""
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

# 예시
print(factorize(60))  # [2, 2, 3, 5]
print(factorize_with_count(60))  # {2: 2, 3: 1, 5: 1}
```

### 3.4 빠른 소인수분해 (전처리)

```python
def precompute_spf(n):
    """
    최소 소인수 (Smallest Prime Factor) 전처리
    """
    spf = list(range(n + 1))

    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i가 소수
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i

    return spf

def fast_factorize(n, spf):
    """전처리된 SPF를 이용한 O(log n) 소인수분해"""
    factors = []
    while n > 1:
        factors.append(spf[n])
        n //= spf[n]
    return factors

# 예시
spf = precompute_spf(100)
print(fast_factorize(60, spf))  # [2, 2, 3, 5]
```

### 3.5 약수 구하기

```python
def get_divisors(n):
    """
    n의 모든 약수
    시간: O(√n)
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

# 예시
print(get_divisors(36))  # [1, 2, 3, 4, 6, 9, 12, 18, 36]
```

---

## 4. 조합론

### 4.1 팩토리얼과 역팩토리얼

```python
MOD = 10**9 + 7
MAX_N = 10**6

# 전처리
factorial = [1] * (MAX_N + 1)
inv_factorial = [1] * (MAX_N + 1)

for i in range(1, MAX_N + 1):
    factorial[i] = factorial[i - 1] * i % MOD

inv_factorial[MAX_N] = mod_pow(factorial[MAX_N], MOD - 2, MOD)
for i in range(MAX_N - 1, -1, -1):
    inv_factorial[i] = inv_factorial[i + 1] * (i + 1) % MOD

def nCr(n, r):
    """이항계수 nCr - O(1) (전처리 후)"""
    if r < 0 or r > n:
        return 0
    return factorial[n] * inv_factorial[r] % MOD * inv_factorial[n - r] % MOD

def nPr(n, r):
    """순열 nPr - O(1) (전처리 후)"""
    if r < 0 or r > n:
        return 0
    return factorial[n] * inv_factorial[n - r] % MOD

# 예시
print(nCr(10, 3))  # 120
print(nPr(10, 3))  # 720
```

### 4.2 파스칼 삼각형

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
    파스칼 삼각형 생성
    시간: O(n²)
    """
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = dp[i][i] = 1
        for j in range(1, i):
            dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % MOD

    return dp

# 예시
C = pascal_triangle(100)
print(C[10][3])  # 120
```

### 4.3 카탈란 수

```
카탈란 수 C_n:
- 올바른 괄호 쌍의 수
- n+1개 리프 이진트리의 수
- 볼록 n+2각형의 삼각분할 수

C_0 = 1
C_n = C(2n, n) / (n + 1) = C(2n, n) - C(2n, n+1)

C_n: 1, 1, 2, 5, 14, 42, 132, ...
```

```python
def catalan(n):
    """
    카탈란 수 C_n
    """
    return nCr(2 * n, n) * mod_inverse(n + 1, MOD) % MOD

def catalan_dp(n):
    """DP 방식"""
    if n <= 1:
        return 1

    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1

    for i in range(2, n + 1):
        for j in range(i):
            dp[i] = (dp[i] + dp[j] * dp[i - 1 - j]) % MOD

    return dp[n]

# 예시
for i in range(10):
    print(catalan(i), end=" ")  # 1 1 2 5 14 42 132 429 1430 4862
```

### 4.4 중복 조합과 중복 순열

```python
def nHr(n, r):
    """
    중복 조합: n개에서 r개 선택 (중복 허용)
    nHr = C(n+r-1, r)
    """
    return nCr(n + r - 1, r)

def repeated_permutation(n, r):
    """
    중복 순열: n개에서 r개 배열 (중복 허용)
    n^r
    """
    return mod_pow(n, r, MOD)

# 예시
print(nHr(3, 2))  # 6: (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
print(repeated_permutation(3, 2))  # 9: 3^2
```

---

## 5. 행렬 거듭제곱

### 5.1 행렬 곱셈

```python
def matrix_mult(A, B, mod):
    """
    행렬 곱셈 A * B
    시간: O(n³)
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

### 5.2 행렬 거듭제곱

```python
def matrix_pow(M, n, mod):
    """
    행렬 M의 n제곱
    시간: O(k³ log n) (k = 행렬 크기)
    """
    size = len(M)
    # 단위 행렬
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    while n > 0:
        if n & 1:
            result = matrix_mult(result, M, mod)
        M = matrix_mult(M, M, mod)
        n >>= 1

    return result
```

### 5.3 피보나치 O(log n)

```
피보나치 점화식의 행렬 표현:

[F(n+1)]   [1 1]^n   [F(1)]
[F(n)  ] = [1 0]   * [F(0)]

→ F(n)을 O(log n)에 계산
```

```python
def fibonacci_matrix(n, mod=10**9 + 7):
    """
    피보나치 수 F(n) - O(log n)
    """
    if n <= 1:
        return n

    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n, mod)

    return result[1][0]

# 예시
for i in range(15):
    print(fibonacci_matrix(i), end=" ")
# 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377

# 큰 수 계산
print(fibonacci_matrix(10**18))  # O(log n)에 계산
```

### 5.4 일반 선형 점화식

```
점화식: f(n) = a*f(n-1) + b*f(n-2) + c*f(n-3)

행렬 표현:
[f(n)  ]   [a b c]^(n-2)   [f(2)]
[f(n-1)] = [1 0 0]       * [f(1)]
[f(n-2)]   [0 1 0]         [f(0)]
```

```python
def solve_linear_recurrence(coeffs, initial, n, mod):
    """
    선형 점화식 해결
    coeffs: [a, b, c, ...] (f(n) = a*f(n-1) + b*f(n-2) + ...)
    initial: [f(0), f(1), f(2), ...] (k개)
    """
    k = len(coeffs)

    if n < k:
        return initial[n]

    # 전이 행렬 구성
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

# 예시: f(n) = f(n-1) + f(n-2), f(0)=0, f(1)=1
print(solve_linear_recurrence([1, 1], [0, 1], 10, 10**9 + 7))  # 55
```

---

## 6. 기타 수학

### 6.1 오일러 피 함수

```
φ(n) = n 이하의 n과 서로소인 수의 개수

φ(p) = p - 1  (p가 소수)
φ(p^k) = p^k - p^(k-1)
φ(mn) = φ(m) * φ(n)  (m, n이 서로소)
```

```python
def euler_phi(n):
    """
    오일러 피 함수 φ(n)
    시간: O(√n)
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

# 예시
print(euler_phi(12))  # 4 (1, 5, 7, 11)
print(euler_phi(7))   # 6 (1, 2, 3, 4, 5, 6)
```

### 6.2 뤼카 정리 (Lucas' Theorem)

```
p가 소수일 때:
C(m, n) mod p = Π C(m_i, n_i) mod p

m과 n을 p진법으로 표현했을 때 각 자릿수의 이항계수 곱
```

```python
def lucas(m, n, p):
    """
    뤼카 정리: C(m, n) mod p
    p가 소수일 때 사용
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

# 예시
print(lucas(1000, 500, 13))  # C(1000, 500) mod 13
```

### 6.3 중국인의 나머지 정리 (CRT)

```
연립 합동식:
x ≡ a1 (mod m1)
x ≡ a2 (mod m2)
...

m1, m2, ... 가 서로소일 때 유일한 해 존재 (mod M, M = m1*m2*...)
```

```python
def chinese_remainder_theorem(remainders, moduli):
    """
    중국인의 나머지 정리
    remainders: [a1, a2, ...]
    moduli: [m1, m2, ...]  (쌍마다 서로소)
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

# 예시
# x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
print(chinese_remainder_theorem([2, 3, 2], [3, 5, 7]))  # 23
```

---

## 7. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐ | [이항 계수 1](https://www.acmicpc.net/problem/11050) | 백준 | 조합 기초 |
| ⭐⭐ | [최대공약수와 최소공배수](https://www.acmicpc.net/problem/2609) | 백준 | GCD/LCM |
| ⭐⭐ | [소수 찾기](https://www.acmicpc.net/problem/1978) | 백준 | 소수 판별 |
| ⭐⭐⭐ | [이항 계수 3](https://www.acmicpc.net/problem/11401) | 백준 | 모듈러 역원 |
| ⭐⭐⭐ | [피보나치 수 6](https://www.acmicpc.net/problem/11444) | 백준 | 행렬 거듭제곱 |
| ⭐⭐⭐ | [골드바흐의 추측](https://www.acmicpc.net/problem/9020) | 백준 | 에라토스테네스 |
| ⭐⭐⭐⭐ | [이항 계수 4](https://www.acmicpc.net/problem/11402) | 백준 | 뤼카 정리 |

---

## 시간 복잡도 정리

```
┌─────────────────────┬─────────────────┐
│ 알고리즘             │ 시간 복잡도      │
├─────────────────────┼─────────────────┤
│ 빠른 거듭제곱        │ O(log n)        │
│ GCD (유클리드)       │ O(log min(a,b)) │
│ 소수 판별            │ O(√n)           │
│ 에라토스테네스 체     │ O(n log log n)  │
│ 소인수분해           │ O(√n)           │
│ 이항계수 (전처리 후)  │ O(1)            │
│ 행렬 곱셈 (k×k)      │ O(k³)           │
│ 행렬 거듭제곱        │ O(k³ log n)     │
└─────────────────────┴─────────────────┘
```

---

## 다음 단계

- [22_String_Algorithms.md](./22_String_Algorithms.md) - 문자열 알고리즘

---

## 참고 자료

- [Modular Arithmetic](https://cp-algorithms.com/algebra/module-inverse.html)
- Introduction to Algorithms (CLRS) - Chapter 31

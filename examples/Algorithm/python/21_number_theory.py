"""
Number Theory
Number Theory Algorithms

Implements fundamental algorithms in number theory.
"""

from typing import List, Tuple
from functools import reduce
import math


# =============================================================================
# 1. GCD / LCM
# =============================================================================

def gcd(a: int, b: int) -> int:
    """
    Greatest Common Divisor (Euclidean algorithm)
    Time Complexity: O(log(min(a, b)))
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Least Common Multiple"""
    return a * b // gcd(a, b)


def gcd_multiple(numbers: List[int]) -> int:
    """GCD of multiple numbers"""
    return reduce(gcd, numbers)


def lcm_multiple(numbers: List[int]) -> int:
    """LCM of multiple numbers"""
    return reduce(lcm, numbers)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm
    Returns (gcd, x, y) such that gcd(a, b) = a*x + b*y
    """
    if b == 0:
        return a, 1, 0

    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1

    return g, x, y


# =============================================================================
# 2. Prime Numbers
# =============================================================================

def is_prime(n: int) -> bool:
    """
    Primality test
    Time Complexity: O(sqrt(n))
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    Sieve of Eratosthenes
    Returns all primes up to n
    Time Complexity: O(n log log n)
    """
    if n < 2:
        return []

    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False

    for i in range(2, int(n ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, n + 1, i):
                is_prime_arr[j] = False

    return [i for i in range(n + 1) if is_prime_arr[i]]


def prime_factorization(n: int) -> List[Tuple[int, int]]:
    """
    Prime factorization
    Returns: [(prime, exponent), ...]
    Time Complexity: O(sqrt(n))
    """
    factors = []
    d = 2

    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                n //= d
                count += 1
            factors.append((d, count))
        d += 1

    if n > 1:
        factors.append((n, 1))

    return factors


# =============================================================================
# 3. Modular Arithmetic
# =============================================================================

def mod_pow(base: int, exp: int, mod: int) -> int:
    """
    Fast exponentiation (modular)
    Time Complexity: O(log exp)
    """
    result = 1
    base %= mod

    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp //= 2
        base = (base * base) % mod

    return result


def mod_inverse(a: int, mod: int) -> int:
    """
    Modular inverse (when mod is prime)
    a^(-1) mod p = a^(p-2) mod p (Fermat's Little Theorem)
    """
    return mod_pow(a, mod - 2, mod)


def mod_inverse_extended(a: int, mod: int) -> int:
    """
    Modular inverse (Extended Euclidean)
    Exists only when gcd(a, mod) = 1
    """
    g, x, _ = extended_gcd(a, mod)
    if g != 1:
        return -1  # No inverse
    return x % mod


# =============================================================================
# 4. Combinatorics
# =============================================================================

def factorial_mod(n: int, mod: int) -> int:
    """n! mod p"""
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % mod
    return result


class Combination:
    """
    Modular combination calculation
    Preprocessing: O(n)
    Query: O(1)
    """

    def __init__(self, n: int, mod: int):
        self.mod = mod
        self.fact = [1] * (n + 1)
        self.inv_fact = [1] * (n + 1)

        # Compute factorials
        for i in range(1, n + 1):
            self.fact[i] = self.fact[i - 1] * i % mod

        # Compute inverse factorials
        self.inv_fact[n] = mod_pow(self.fact[n], mod - 2, mod)
        for i in range(n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % mod

    def nCr(self, n: int, r: int) -> int:
        """nCr mod p"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod

    def nPr(self, n: int, r: int) -> int:
        """nPr mod p"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[n - r] % self.mod


# =============================================================================
# 5. Euler's Totient Function
# =============================================================================

def euler_phi(n: int) -> int:
    """
    phi(n) = count of positive integers <= n that are coprime with n
    Time Complexity: O(sqrt(n))
    """
    result = n

    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            result -= result // d
        d += 1

    if n > 1:
        result -= result // n

    return result


def euler_phi_sieve(n: int) -> List[int]:
    """Euler's totient function values for 1~n (sieve method)"""
    phi = list(range(n + 1))

    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i

    return phi


# =============================================================================
# 6. Chinese Remainder Theorem (CRT)
# =============================================================================

def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """
    Find the smallest positive integer x satisfying x = r_i (mod m_i)
    All m_i must be pairwise coprime
    """
    M = 1
    for m in moduli:
        M *= m

    result = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        yi = mod_inverse_extended(Mi, m)
        result += r * Mi * yi

    return result % M


# =============================================================================
# 7. Lucas' Theorem
# =============================================================================

def lucas(n: int, r: int, p: int) -> int:
    """
    Compute nCr mod p using Lucas' Theorem
    Useful when p is prime and n, r are very large
    """
    if r == 0:
        return 1
    return lucas(n // p, r // p, p) * nCr_small(n % p, r % p, p) % p


def nCr_small(n: int, r: int, p: int) -> int:
    """Small nCr mod p"""
    if r > n:
        return 0
    if r == 0 or r == n:
        return 1

    num = den = 1
    for i in range(r):
        num = num * (n - i) % p
        den = den * (i + 1) % p

    return num * mod_pow(den, p - 2, p) % p


# =============================================================================
# 8. Divisors
# =============================================================================

def divisors(n: int) -> List[int]:
    """
    All divisors of n
    Time Complexity: O(sqrt(n))
    """
    result = []

    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)

    return sorted(result)


def divisor_count(n: int) -> int:
    """Number of divisors"""
    factors = prime_factorization(n)
    count = 1
    for _, exp in factors:
        count *= (exp + 1)
    return count


def divisor_sum(n: int) -> int:
    """Sum of divisors"""
    factors = prime_factorization(n)
    total = 1
    for p, e in factors:
        total *= (pow(p, e + 1) - 1) // (p - 1)
    return total


# =============================================================================
# 9. Linear Diophantine Equation
# =============================================================================

def solve_linear_diophantine(a: int, b: int, c: int) -> Tuple[bool, int, int]:
    """
    Find integer solution (x, y) for ax + by = c
    Returns: (solution exists, x, y)
    """
    g, x0, y0 = extended_gcd(abs(a), abs(b))

    if c % g != 0:
        return False, 0, 0

    x0 *= c // g
    y0 *= c // g

    if a < 0:
        x0 = -x0
    if b < 0:
        y0 = -y0

    return True, x0, y0


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Number Theory Examples")
    print("=" * 60)

    # 1. GCD / LCM
    print("\n[1] GCD / LCM")
    a, b = 48, 18
    print(f"    gcd({a}, {b}) = {gcd(a, b)}")
    print(f"    lcm({a}, {b}) = {lcm(a, b)}")
    g, x, y = extended_gcd(a, b)
    print(f"    Extended Euclidean: {a}*{x} + {b}*{y} = {g}")

    # 2. Primes
    print("\n[2] Prime Numbers")
    print(f"    17 is prime: {is_prime(17)}")
    print(f"    18 is prime: {is_prime(18)}")
    primes = sieve_of_eratosthenes(50)
    print(f"    Primes up to 50: {primes}")
    print(f"    Prime factorization of 60: {prime_factorization(60)}")

    # 3. Modular Arithmetic
    print("\n[3] Modular Arithmetic")
    mod = 1000000007
    print(f"    2^10 mod {mod} = {mod_pow(2, 10, mod)}")
    print(f"    Inverse of 3 mod 7 = {mod_inverse(3, 7)}")
    print(f"    Verification: 3 * {mod_inverse(3, 7)} mod 7 = {3 * mod_inverse(3, 7) % 7}")

    # 4. Combinatorics
    print("\n[4] Combinatorics")
    comb = Combination(1000, mod)
    print(f"    10C3 = {comb.nCr(10, 3)}")
    print(f"    10P3 = {comb.nPr(10, 3)}")
    print(f"    100C50 mod {mod} = {comb.nCr(100, 50)}")

    # 5. Euler's Totient Function
    print("\n[5] Euler's Totient Function")
    for n in [10, 12, 7, 1]:
        print(f"    phi({n}) = {euler_phi(n)}")

    # 6. Chinese Remainder Theorem
    print("\n[6] Chinese Remainder Theorem")
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    x = chinese_remainder_theorem(remainders, moduli)
    print(f"    x = 2 (mod 3), x = 3 (mod 5), x = 2 (mod 7)")
    print(f"    x = {x}")
    print(f"    Verification: {x % 3}, {x % 5}, {x % 7}")

    # 7. Lucas' Theorem
    print("\n[7] Lucas' Theorem")
    n, r, p = 1000, 500, 13
    print(f"    C({n}, {r}) mod {p} = {lucas(n, r, p)}")

    # 8. Divisors
    print("\n[8] Divisors")
    n = 36
    print(f"    Divisors of {n}: {divisors(n)}")
    print(f"    Divisor count: {divisor_count(n)}")
    print(f"    Divisor sum: {divisor_sum(n)}")

    # 9. Diophantine Equation
    print("\n[9] Linear Diophantine Equation")
    a, b, c = 3, 5, 7
    exists, x, y = solve_linear_diophantine(a, b, c)
    print(f"    {a}x + {b}y = {c}")
    print(f"    Solution exists: {exists}, x={x}, y={y}")
    print(f"    Verification: {a}*{x} + {b}*{y} = {a * x + b * y}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

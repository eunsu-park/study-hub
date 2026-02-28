"""
Exercises for Lesson 21: Math and Number Theory
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: GCD/LCM, Sieve of Eratosthenes, Modular Inverse, Matrix Exponentiation.
"""


# === Exercise 1: GCD and LCM ===
# Problem: Compute GCD and LCM of two numbers.

def exercise_1():
    """Solution: Euclidean algorithm for GCD, derive LCM."""
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a // gcd(a, b) * b

    tests = [
        (12, 8, 4, 24),
        (100, 75, 25, 300),
        (7, 13, 1, 91),
        (1, 1, 1, 1),
        (0, 5, 5, 0),
    ]

    for a, b, exp_gcd, exp_lcm in tests:
        g = gcd(a, b)
        l = lcm(a, b) if a > 0 and b > 0 else 0
        print(f"gcd({a}, {b}) = {g}, lcm({a}, {b}) = {l}")
        assert g == exp_gcd
        if a > 0 and b > 0:
            assert l == exp_lcm

    print("All GCD/LCM tests passed!")


# === Exercise 2: Sieve of Eratosthenes ===
# Problem: Find all primes up to n.

def exercise_2():
    """Solution: Classic sieve in O(n log log n)."""
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False

        p = 2
        while p * p <= n:
            if is_prime[p]:
                # Mark all multiples of p as composite
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
            p += 1

        return [i for i in range(n + 1) if is_prime[i]]

    primes_100 = sieve(100)
    print(f"Primes up to 100: {primes_100}")
    assert primes_100 == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                          47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    primes_30 = sieve(30)
    print(f"Primes up to 30: {primes_30}")
    assert primes_30 == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    assert sieve(1) == []
    assert sieve(2) == [2]

    print("All Sieve tests passed!")


# === Exercise 3: Fast Exponentiation (Modular) ===
# Problem: Compute a^n mod m efficiently.

def exercise_3():
    """Solution: Binary exponentiation in O(log n)."""
    def power_mod(base, exp, mod):
        result = 1
        base = base % mod

        while exp > 0:
            if exp & 1:  # exp is odd
                result = (result * base) % mod
            exp >>= 1
            base = (base * base) % mod

        return result

    tests = [
        (2, 10, 1000, 24),        # 2^10 = 1024 mod 1000 = 24
        (3, 13, 1000000007, 1594323),
        (2, 0, 100, 1),           # anything^0 = 1
        (7, 1, 100, 7),
    ]

    for base, exp, mod, expected in tests:
        result = power_mod(base, exp, mod)
        print(f"{base}^{exp} mod {mod} = {result}")
        assert result == expected

    # Verify against Python's built-in pow
    assert power_mod(123456, 789, 1000000007) == pow(123456, 789, 1000000007)

    print("All Fast Exponentiation tests passed!")


# === Exercise 4: Modular Inverse ===
# Problem: Find the modular inverse of a modulo m (a^(-1) mod m).
#   Exists only when gcd(a, m) = 1.
# Approach: Using Fermat's little theorem (when m is prime): a^(-1) = a^(m-2) mod m.

def exercise_4():
    """Solution: Modular inverse using Fermat's little theorem."""
    def mod_inverse(a, m):
        """Find a^(-1) mod m, where m is prime."""
        def power_mod(base, exp, mod):
            result = 1
            base = base % mod
            while exp > 0:
                if exp & 1:
                    result = (result * base) % mod
                exp >>= 1
                base = (base * base) % mod
            return result

        return power_mod(a, m - 2, m)

    MOD = 1000000007

    tests = [
        (2, MOD),
        (3, MOD),
        (100, MOD),
    ]

    for a, m in tests:
        inv = mod_inverse(a, m)
        check = (a * inv) % m
        print(f"inverse({a}) mod {m} = {inv}, check: {a}*{inv} mod {m} = {check}")
        assert check == 1

    print("All Modular Inverse tests passed!")


# === Exercise 5: Matrix Exponentiation for Fibonacci ===
# Problem: Compute the n-th Fibonacci number in O(log n) using matrix exponentiation.
#   [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n

def exercise_5():
    """Solution: Matrix exponentiation for Fibonacci in O(log n)."""
    def mat_mult(A, B, mod):
        """Multiply two 2x2 matrices under modular arithmetic."""
        return [
            [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod,
             (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod],
            [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod,
             (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod],
        ]

    def mat_pow(M, n, mod):
        """Compute M^n mod 'mod' using binary exponentiation."""
        result = [[1, 0], [0, 1]]  # Identity matrix
        while n > 0:
            if n & 1:
                result = mat_mult(result, M, mod)
            M = mat_mult(M, M, mod)
            n >>= 1
        return result

    def fibonacci(n, mod=1000000007):
        if n <= 0:
            return 0
        if n <= 2:
            return 1
        base = [[1, 1], [1, 0]]
        result = mat_pow(base, n - 1, mod)
        return result[0][0]

    # Test against iterative computation
    def fib_iterative(n):
        if n <= 0:
            return 0
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    for n in [1, 2, 5, 10, 20, 50]:
        result = fibonacci(n)
        expected = fib_iterative(n)
        print(f"fib({n}) = {result}")
        assert result == expected % 1000000007

    # Large n
    result = fibonacci(1000000)
    print(f"fib(1000000) mod 10^9+7 = {result}")
    assert isinstance(result, int) and 0 <= result < 1000000007

    print("All Matrix Exponentiation tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: GCD and LCM ===")
    exercise_1()
    print("\n=== Exercise 2: Sieve of Eratosthenes ===")
    exercise_2()
    print("\n=== Exercise 3: Fast Exponentiation ===")
    exercise_3()
    print("\n=== Exercise 4: Modular Inverse ===")
    exercise_4()
    print("\n=== Exercise 5: Matrix Exponentiation for Fibonacci ===")
    exercise_5()
    print("\nAll exercises completed!")

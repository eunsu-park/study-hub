"""
Exercises for Lesson 01: Complexity Analysis
Topic: Algorithm

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: Calculate Complexity ===
# Problem: Find the time complexity of the following code.
#   void mystery(int n) {
#       for (int i = n; i >= 1; i /= 2) {
#           for (int j = 1; j <= n; j *= 2) {
#               // O(1) work
#           }
#       }
#   }

def exercise_1():
    """
    Solution: O(log^2 n)

    - Outer loop: i starts at n and halves each time -> log(n) iterations
    - Inner loop: j starts at 1 and doubles each time -> log(n) iterations
    - Total: log(n) * log(n) = O(log^2 n)

    We demonstrate by counting iterations for various n.
    """
    def mystery_count(n):
        count = 0
        i = n
        while i >= 1:
            j = 1
            while j <= n:
                count += 1
                j *= 2
            i //= 2
            if i == 0:
                break
        return count

    for n in [16, 64, 256, 1024, 4096]:
        actual = mystery_count(n)
        log_n = math.log2(n)
        predicted = log_n * log_n  # O(log^2 n)
        print(f"n={n:5d}: iterations={actual:4d}, log^2(n)={predicted:.1f}, "
              f"ratio={actual/predicted:.2f}")

    print("\nAnswer: O(log^2 n)")
    print("The ratio stays roughly constant, confirming log^2(n) behavior.")


# === Exercise 2: Complexity Comparison ===
# Problem: Compare operation counts when n = 1000.
#   A. O(n)
#   B. O(n log n)
#   C. O(n^2)
#   D. O(2^(log n))

def exercise_2():
    """
    Solution:
    A. O(n) = 1,000
    B. O(n log n) ~ 10,000
    C. O(n^2) = 1,000,000
    D. O(2^(log n)) = O(n) = 1,000

    Since 2^(log2(n)) = n, A and D are equivalent.
    """
    n = 1000

    a = n
    b = n * math.log2(n)
    c = n * n
    d = 2 ** math.log2(n)

    print(f"n = {n}")
    print(f"A. O(n)          = {a:,.0f}")
    print(f"B. O(n log n)    = {b:,.0f}")
    print(f"C. O(n^2)        = {c:,.0f}")
    print(f"D. O(2^(log n))  = {d:,.0f}")
    print(f"\nSince 2^(log2(n)) = n, A and D are equal: {a} == {d:.0f}")


# === Exercise 3: Recursion Analysis ===
# Problem: Analyze the time complexity of:
#   int f(int n) {
#       if (n <= 1) return 1;
#       return f(n/2) + f(n/2);
#   }

def exercise_3():
    """
    Solution: O(n)

    Recurrence: T(n) = 2T(n/2) + O(1)
    By Master Theorem: a=2, b=2, f(n)=O(1)
    n^(log_b(a)) = n^(log_2(2)) = n^1 = n
    f(n) = O(1) < O(n^(1-epsilon)) => Case 1
    T(n) = Theta(n)

    We verify by counting recursive calls.
    """
    call_count = 0

    def f(n):
        nonlocal call_count
        call_count += 1
        if n <= 1:
            return 1
        return f(n // 2) + f(n // 2)

    for n in [16, 64, 256, 1024]:
        call_count = 0
        result = f(n)
        print(f"f({n:4d}): calls={call_count:5d}, ratio calls/n = {call_count/n:.2f}")

    print("\nAnswer: O(n)")
    print("The ratio of calls to n approaches ~2, confirming O(n) behavior.")


if __name__ == "__main__":
    print("=== Exercise 1: Calculate Complexity ===")
    exercise_1()
    print("\n=== Exercise 2: Complexity Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Recursion Analysis ===")
    exercise_3()
    print("\nAll exercises completed!")

"""
Exercises for Lesson 04: Control Flow Patterns
Topic: Programming

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Refactor with Guard Clauses ===
# Problem: Refactor nested code using early returns.

def exercise_1():
    """Solution: Replace deep nesting with guard clauses."""

    # Simulated Order and User for testing
    class User:
        def __init__(self, is_verified=True):
            self.is_verified = is_verified

    class Order:
        def __init__(self, valid=True, total=100, user_verified=True):
            self._valid = valid
            self.total = total
            self.user = User(user_verified)

        def is_valid(self):
            return self._valid

    # Original deeply nested version (for reference):
    # def process_order(order):
    #     if order is not None:
    #         if order.is_valid():
    #             if order.total > 0:
    #                 if order.user.is_verified:
    #                     print("Processing order")
    #                 else: print("User not verified")
    #             else: print("Order total must be positive")
    #         else: print("Invalid order")
    #     else: print("No order")

    # Refactored with guard clauses: each check returns early on failure.
    # The "happy path" logic is at the end, not buried in nesting.
    def process_order(order):
        """Process an order using guard clauses for clarity."""
        if order is None:
            return "No order"

        if not order.is_valid():
            return "Invalid order"

        if order.total <= 0:
            return "Order total must be positive"

        if not order.user.is_verified:
            return "User not verified"

        # Happy path: all guards passed
        return "Processing order"

    # Test all branches
    test_cases = [
        (None, "No order"),
        (Order(valid=False), "Invalid order"),
        (Order(total=0), "Order total must be positive"),
        (Order(user_verified=False), "User not verified"),
        (Order(), "Processing order"),
    ]

    for order, expected in test_cases:
        result = process_order(order)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: {result}")


# === Exercise 2: Recursion vs Iteration ===
# Problem: Implement array sum both recursively and iteratively.

def exercise_2():
    """Solution: Sum array elements using recursion and iteration."""

    def sum_recursive(numbers):
        """
        Recursive sum: base case is empty list (sum = 0).
        Recursive case: first element + sum of the rest.
        This naturally follows the mathematical definition of summation.
        """
        if not numbers:
            return 0  # Base case: empty list sums to 0
        return numbers[0] + sum_recursive(numbers[1:])

    def sum_iterative(numbers):
        """
        Iterative sum: accumulate total in a loop.
        More efficient because no function call overhead or stack usage.
        """
        total = 0
        for num in numbers:
            total += num
        return total

    test_data = [1, 2, 3, 4, 5]
    expected = 15

    print(f"  Input: {test_data}")
    print(f"  Recursive: {sum_recursive(test_data)}")
    print(f"  Iterative: {sum_iterative(test_data)}")
    print(f"  Expected: {expected}")

    # Edge cases
    print(f"  Empty list - recursive: {sum_recursive([])}, iterative: {sum_iterative([])}")
    print(f"  Single element - recursive: {sum_recursive([42])}, iterative: {sum_iterative([42])}")

    print("\n  Which is clearer?")
    print("    Recursive: clearer for mathematical definitions, mirrors the formula")
    print("    Iterative: clearer for simple accumulation patterns")
    print("  Which is more efficient?")
    print("    Iterative: O(n) time, O(1) space")
    print("    Recursive: O(n) time, O(n) space (call stack + list slicing)")


# === Exercise 3: Generator ===
# Problem: Write a generator that produces the Fibonacci sequence indefinitely.

def exercise_3():
    """Solution: Infinite Fibonacci generator using yield."""

    def fibonacci_gen():
        """
        Generate Fibonacci numbers indefinitely.

        Uses yield to produce values lazily (on-demand), so this can
        represent the infinite Fibonacci sequence without using infinite memory.
        Only two variables (a, b) are needed regardless of how many numbers
        are generated.
        """
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b  # Simultaneous assignment avoids temp variable

    # Get first 15 Fibonacci numbers
    gen = fibonacci_gen()
    fibs = [next(gen) for _ in range(15)]
    print(f"  First 15 Fibonacci numbers: {fibs}")

    # Can also use in a for loop with a limit
    print("  First 10 via for loop:")
    for i, fib in enumerate(fibonacci_gen()):
        if i >= 10:
            break
        print(f"    F({i}) = {fib}")


# === Exercise 4: Pattern Matching ===
# Problem: Classify a numeric value using pattern matching.

def exercise_4():
    """Solution: Value classifier using Python 3.10+ match/case."""

    def classify_value(value):
        """
        Classify a numeric value into categories.
        Uses structural pattern matching (Python 3.10+).
        Guard clauses (if conditions) handle ranges.
        """
        match value:
            case 0:
                return "zero"
            case int(x) if x < 0:
                return "negative"
            case int(x) if 1 <= x <= 10:
                return "small"
            case int(x) if 11 <= x <= 100:
                return "medium"
            case int(x) if x > 100:
                return "large"
            case _:
                return "unknown"

    # Also provide a traditional if/elif version for compatibility
    def classify_value_compat(value):
        """Same logic using traditional if/elif (works on all Python versions)."""
        if not isinstance(value, (int, float)):
            return "unknown"
        if value == 0:
            return "zero"
        if value < 0:
            return "negative"
        if value <= 10:
            return "small"
        if value <= 100:
            return "medium"
        return "large"

    test_values = [0, -5, 1, 5, 10, 11, 50, 100, 101, 1000, "text", 3.14]
    print("  Value classifications:")
    for val in test_values:
        # Use the compatible version to ensure it runs on all Python versions
        result = classify_value_compat(val)
        print(f"    {val!r:>8} -> {result}")


# === Exercise 5: Error Handling ===
# Problem: Implement safe_divide with exceptions and Result type.

def exercise_5():
    """Solution: Safe division using exceptions and Result type."""

    # Version 1: Exception-based
    # Pythonic approach using try/except. Good when errors are truly exceptional.
    def safe_divide_exception(a, b):
        """Divide a by b, returning None on division by zero."""
        try:
            return a / b
        except ZeroDivisionError:
            return None

    # Version 2: Result-type approach
    # Makes error handling explicit in the return type.
    # Caller must check success before using the value.
    def safe_divide_result(a, b):
        """
        Divide a by b, returning a (success, value_or_error) tuple.

        This is a lightweight Result type: the caller is forced to handle
        both success and failure cases, unlike exceptions which can be
        silently ignored.
        """
        if b == 0:
            return (False, "Cannot divide by zero")
        return (True, a / b)

    # Test both approaches
    print("  Exception-based approach:")
    print(f"    10 / 2 = {safe_divide_exception(10, 2)}")
    print(f"    10 / 0 = {safe_divide_exception(10, 0)}")

    print("\n  Result-type approach:")
    for a, b in [(10, 2), (10, 0), (7, 3)]:
        success, value = safe_divide_result(a, b)
        if success:
            print(f"    {a} / {b} = {value:.4f}")
        else:
            print(f"    {a} / {b} -> Error: {value}")

    print("\n  Trade-offs:")
    print("    Exceptions: simpler call site, but errors can be silently missed")
    print("    Result type: forces explicit handling, but more verbose")


# === Exercise 6: Tail Recursion ===
# Problem: Rewrite Fibonacci to be tail-recursive.

def exercise_6():
    """Solution: Tail-recursive Fibonacci with accumulators."""

    def fibonacci_tail(n, a=0, b=1):
        """
        Tail-recursive Fibonacci.

        Uses two accumulators (a, b) that carry the state forward,
        so the recursive call is the LAST operation (tail position).
        In languages with tail-call optimization (Scheme, Scala),
        this runs in O(1) stack space.

        Note: Python does NOT optimize tail calls, so this will still
        hit the recursion limit for large n. But it demonstrates the pattern.
        """
        if n == 0:
            return a  # Base case: return accumulated result
        return fibonacci_tail(n - 1, b, a + b)  # Tail call with updated accumulators

    # Compare with naive recursive Fibonacci
    def fibonacci_naive(n):
        """Standard recursive Fibonacci (exponential time, for comparison)."""
        if n <= 1:
            return n
        return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

    # Test correctness
    print("  Comparing tail-recursive vs naive Fibonacci:")
    for i in range(11):
        tail = fibonacci_tail(i)
        naive = fibonacci_naive(i)
        match = "OK" if tail == naive else "MISMATCH"
        print(f"    F({i:2d}) = {tail:4d}  [{match}]")

    # Tail-recursive version can handle much larger inputs
    print(f"\n  F(100) via tail recursion: {fibonacci_tail(100)}")
    print("  (Naive recursion would take impossibly long for F(100))")

    # Iterative equivalent (what the compiler would optimize tail recursion to)
    def fibonacci_iterative(n):
        """Iterative version - equivalent to optimized tail recursion."""
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    print(f"  F(100) via iteration: {fibonacci_iterative(100)}")
    print(f"  Results match: {fibonacci_tail(100) == fibonacci_iterative(100)}")


if __name__ == "__main__":
    print("=== Exercise 1: Refactor with Guard Clauses ===")
    exercise_1()
    print("\n=== Exercise 2: Recursion vs Iteration ===")
    exercise_2()
    print("\n=== Exercise 3: Generator ===")
    exercise_3()
    print("\n=== Exercise 4: Pattern Matching ===")
    exercise_4()
    print("\n=== Exercise 5: Error Handling ===")
    exercise_5()
    print("\n=== Exercise 6: Tail Recursion ===")
    exercise_6()
    print("\nAll exercises completed!")

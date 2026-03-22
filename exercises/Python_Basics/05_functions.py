"""
Exercise 05: Functions

Practice function definitions, recursion, closures, and *args/**kwargs.
"""


def factorial(n):
    """Return the factorial of n using recursion.

    Args:
        n: Non-negative integer.

    Returns:
        n! (n factorial).

    Raises:
        ValueError: If n is negative.
    """
    # TODO: Implement this recursively
    pass


def is_palindrome(s):
    """Check if a string is a palindrome (case-insensitive, ignoring spaces).

    Args:
        s: Input string.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    # TODO: Implement this
    pass


def flatten_list(nested):
    """Flatten an arbitrarily nested list into a single flat list.

    Example: [1, [2, [3, 4]], 5] -> [1, 2, 3, 4, 5]

    Args:
        nested: A list that may contain nested lists.

    Returns:
        A flat list with all elements.
    """
    # TODO: Implement this recursively
    pass


def make_memoized_fibonacci():
    """Return a memoized Fibonacci function using a closure.

    The returned function fib(n) should compute the nth Fibonacci number
    (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...) and cache results.

    Returns:
        A function that computes Fibonacci numbers with memoization.
    """
    # TODO: Implement a closure with a cache dict
    pass


def sum_all(*args):
    """Return the sum of all positional arguments.

    Should handle any number of numeric arguments.

    Args:
        *args: Variable number of numeric values.

    Returns:
        Sum of all arguments. Returns 0 if no arguments.
    """
    # TODO: Implement this
    pass


def apply_operations(value, *functions):
    """Apply a sequence of functions to a value (pipeline).

    Each function takes one argument and returns one value.
    Apply them left-to-right.

    Example: apply_operations(5, str, len) -> 1

    Args:
        value: Initial value.
        *functions: Functions to apply in sequence.

    Returns:
        Final result after all functions are applied.
    """
    # TODO: Implement this
    pass


# === Tests ===

assert factorial(0) == 1, "0!"
assert factorial(1) == 1, "1!"
assert factorial(5) == 120, "5!"
try:
    factorial(-1)
    assert False, "Should raise ValueError"
except ValueError:
    pass

assert is_palindrome("racecar") is True, "racecar"
assert is_palindrome("A man a plan a canal Panama") is True, "Panama"
assert is_palindrome("hello") is False, "hello"
assert is_palindrome("Was it a car or a cat I saw") is True, "Was it a car"

assert flatten_list([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5], "Nested"
assert flatten_list([1, 2, 3]) == [1, 2, 3], "Already flat"
assert flatten_list([]) == [], "Empty"
assert flatten_list([[[[1]]]]) == [1], "Deeply nested"

fib = make_memoized_fibonacci()
assert fib(0) == 0, "fib(0)"
assert fib(1) == 1, "fib(1)"
assert fib(10) == 55, "fib(10)"
assert fib(30) == 832040, "fib(30) - tests memoization speed"

assert sum_all() == 0, "No args"
assert sum_all(1, 2, 3) == 6, "Three args"
assert sum_all(10) == 10, "One arg"

assert apply_operations(5, str, len) == 1, "5 -> '5' -> 1"
assert apply_operations(-3, abs, str) == "3", "-3 -> 3 -> '3'"

print("All tests passed!")

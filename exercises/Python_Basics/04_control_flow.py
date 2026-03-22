"""
Exercise 04: Control Flow

Practice conditionals, loops, and pattern printing.
"""


def fizzbuzz(n):
    """Return a list of FizzBuzz results from 1 to n (inclusive).

    Rules:
    - Divisible by 3 and 5: "FizzBuzz"
    - Divisible by 3 only: "Fizz"
    - Divisible by 5 only: "Buzz"
    - Otherwise: the number itself (as int)

    Args:
        n: Upper bound (positive integer).

    Returns:
        List of FizzBuzz results.
    """
    # TODO: Implement this
    pass


def grade_calculator(score):
    """Convert a numeric score to a letter grade.

    Grading scale:
        90-100 -> "A"
        80-89  -> "B"
        70-79  -> "C"
        60-69  -> "D"
        0-59   -> "F"

    Args:
        score: Integer score (0-100).

    Returns:
        Letter grade string.

    Raises:
        ValueError: If score is not in range 0-100.
    """
    # TODO: Implement this
    pass


def collatz_length(n):
    """Return the number of steps for n to reach 1 (Collatz conjecture).

    Rules:
    - If n is even: n = n // 2
    - If n is odd: n = 3*n + 1
    - Count steps until n == 1

    Args:
        n: Starting positive integer (>= 1).

    Returns:
        Number of steps to reach 1.
    """
    # TODO: Implement this
    pass


def print_triangle(n):
    """Return a right triangle pattern as a string.

    For n=4, the output should be:
    "*\n**\n***\n****"

    Args:
        n: Height of the triangle.

    Returns:
        Triangle pattern as a single string with newlines.
    """
    # TODO: Implement this
    pass


def find_primes(limit):
    """Return a list of all prime numbers up to limit (inclusive).

    Use the Sieve of Eratosthenes or trial division.

    Args:
        limit: Upper bound (positive integer >= 2).

    Returns:
        List of prime numbers.
    """
    # TODO: Implement this
    pass


def sum_digits(n):
    """Return the sum of digits of a non-negative integer.

    Args:
        n: Non-negative integer.

    Returns:
        Sum of all digits.
    """
    # TODO: Implement this
    pass


# === Tests ===

fb = fizzbuzz(15)
assert fb[0] == 1, "FizzBuzz: 1"
assert fb[2] == "Fizz", "FizzBuzz: 3"
assert fb[4] == "Buzz", "FizzBuzz: 5"
assert fb[14] == "FizzBuzz", "FizzBuzz: 15"
assert len(fb) == 15, "FizzBuzz length"

assert grade_calculator(95) == "A", "Grade A"
assert grade_calculator(85) == "B", "Grade B"
assert grade_calculator(75) == "C", "Grade C"
assert grade_calculator(65) == "D", "Grade D"
assert grade_calculator(55) == "F", "Grade F"
try:
    grade_calculator(101)
    assert False, "Should raise ValueError"
except ValueError:
    pass

assert collatz_length(1) == 0, "Collatz 1"
assert collatz_length(2) == 1, "Collatz 2"
assert collatz_length(6) == 8, "Collatz 6"

assert print_triangle(4) == "*\n**\n***\n****", "Triangle 4"
assert print_triangle(1) == "*", "Triangle 1"

assert find_primes(10) == [2, 3, 5, 7], "Primes up to 10"
assert find_primes(20) == [2, 3, 5, 7, 11, 13, 17, 19], "Primes up to 20"

assert sum_digits(123) == 6, "Sum digits 123"
assert sum_digits(0) == 0, "Sum digits 0"
assert sum_digits(9999) == 36, "Sum digits 9999"

print("All tests passed!")

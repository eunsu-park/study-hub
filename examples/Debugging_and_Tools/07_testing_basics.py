"""
07 Testing Basics
=================
Demonstrates testing with assert, pytest patterns, test-driven
debugging, and parametrized tests.
"""


# --- Functions to test ---

def add(a, b):
    """Add two numbers."""
    return a + b


def divide(a, b):
    """Divide a by b. Raises ZeroDivisionError if b is 0."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def clamp(value, minimum, maximum):
    """Restrict value to [minimum, maximum] range."""
    if minimum > maximum:
        raise ValueError("minimum must be <= maximum")
    return max(minimum, min(value, maximum))


def find_max(numbers):
    """Return the maximum value in a list. Raises ValueError if empty."""
    if not numbers:
        raise ValueError("Cannot find max of empty list")
    result = numbers[0]
    for n in numbers[1:]:
        if n > result:
            result = n
    return result


# --- Assert-based tests ---

def test_with_assert():
    """Demonstrate assert-based testing."""
    print("=== Assert-Based Testing ===")

    assert add(2, 3) == 5, f"Expected 5, got {add(2, 3)}"
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    print("  add() tests: PASSED")

    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
    print("  clamp() tests: PASSED")

    assert find_max([1, 3, 2]) == 3
    assert find_max([5]) == 5
    assert find_max([-1, -5, -2]) == -1
    print("  find_max() tests: PASSED")

    # Testing exceptions
    try:
        divide(10, 0)
        assert False, "Should have raised ZeroDivisionError"
    except ZeroDivisionError:
        pass  # Expected
    print("  divide(10, 0) raises ZeroDivisionError: PASSED")

    print()


# --- pytest-style tests (run with: pytest 07_testing_basics.py) ---

def test_add_positive():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-1, -2) == -3


def test_add_zero():
    assert add(5, 0) == 5
    assert add(0, 5) == 5


def test_clamp_within_range():
    assert clamp(5, 0, 10) == 5


def test_clamp_below_minimum():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above_maximum():
    assert clamp(15, 0, 10) == 10


def test_clamp_at_boundaries():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10


def test_clamp_equal_min_max():
    assert clamp(5, 7, 7) == 7


def test_clamp_negative_range():
    assert clamp(-5, -10, -1) == -5


def test_find_max_normal():
    assert find_max([1, 3, 2]) == 3


def test_find_max_single():
    assert find_max([42]) == 42


def test_find_max_negative():
    assert find_max([-1, -5, -2]) == -1


# --- Test-driven debugging demo ---

def test_driven_debugging_demo():
    """Show the test-driven debugging workflow."""
    print("=== Test-Driven Debugging Demo ===")

    # Buggy function
    def average_buggy(numbers):
        return sum(numbers) / (len(numbers) + 1)  # BUG: +1

    # Step 1: Write failing test
    result = average_buggy([42])
    expected = 42.0
    print(f"  test_average([42]) = {result} (expected {expected})")
    print(f"  Test: {'PASSED' if result == expected else 'FAILED'}")

    # Step 2: Fix
    def average_fixed(numbers):
        return sum(numbers) / len(numbers)

    result = average_fixed([42])
    print(f"  After fix: average([42]) = {result}")
    print(f"  Test: {'PASSED' if result == expected else 'FAILED'}")

    # Step 3: Run all tests
    test_cases = [
        ([42], 42.0),
        ([10, 20, 30], 20.0),
        ([-10, 10], 0.0),
        ([1, 2, 3, 4, 5], 3.0),
    ]
    all_pass = True
    for inputs, expected in test_cases:
        result = average_fixed(inputs)
        passed = abs(result - expected) < 1e-9
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  average({inputs}) = {result} [{status}]")

    print(f"  All tests: {'PASSED' if all_pass else 'FAILED'}")
    print()


# --- Parametrized test concept ---

def parametrized_test_demo():
    """Demonstrate the concept of parametrized tests."""
    print("=== Parametrized Tests (Concept) ===")

    test_cases = [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (100, 200, 300),
        (-5, -3, -8),
    ]
    for a, b, expected in test_cases:
        result = add(a, b)
        status = "PASS" if result == expected else "FAIL"
        print(f"  add({a}, {b}) = {result} (expected {expected}) [{status}]")

    print("\n  In pytest, use @pytest.mark.parametrize for this pattern")
    print()


if __name__ == "__main__":
    test_with_assert()
    test_driven_debugging_demo()
    parametrized_test_demo()
    print("All demonstrations complete.")

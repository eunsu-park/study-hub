"""
Exercise 07: Testing Basics

Practice writing tests with assert and pytest patterns.
"""


# --- Functions to test ---

def safe_divide(a, b):
    """Divide a by b. Returns None if b is zero."""
    if b == 0:
        return None
    return a / b


def flatten(nested_list):
    """Flatten a list of lists into a single list.

    Example: [[1,2],[3,4],[5]] → [1,2,3,4,5]
    """
    result = []
    for sublist in nested_list:
        result.extend(sublist)
    return result


def count_vowels(text):
    """Count the number of vowels (a,e,i,o,u) in text.

    Case-insensitive.
    """
    return sum(1 for c in text.lower() if c in "aeiou")


# --- Exercises ---

def test_safe_divide():
    """Write comprehensive tests for safe_divide.

    Test cases to cover:
    - Normal division
    - Division by zero (should return None)
    - Negative numbers
    - Float results
    - Zero as numerator
    """
    # TODO: Write at least 5 test assertions
    pass


def test_flatten():
    """Write comprehensive tests for flatten.

    Test cases to cover:
    - Normal nested list
    - Empty outer list
    - Empty inner lists
    - Single-element lists
    - Mixed sizes
    """
    # TODO: Write at least 5 test assertions
    pass


def test_count_vowels():
    """Write comprehensive tests for count_vowels.

    Test cases to cover:
    - Normal string
    - Empty string
    - No vowels
    - All vowels
    - Case insensitivity
    """
    # TODO: Write at least 5 test assertions
    pass


def test_driven_debug():
    """Use test-driven debugging to find and fix the bug.

    The function below has a bug. Write a failing test first,
    then fix the function.

    Returns:
        tuple: (buggy_result, fixed_result, test_passed)
    """

    def remove_duplicates(items):
        """Remove duplicates while preserving order. BUGGY."""
        seen = set()
        result = []
        for item in items:
            if item in seen:  # BUG: should be "not in seen"
                seen.add(item)
                result.append(item)
        return result

    # TODO: Write a failing test, then fix the function
    # 1. Test that remove_duplicates([1,2,2,3,1]) == [1,2,3]
    # 2. Fix the bug
    # 3. Verify the test passes

    buggy_result = remove_duplicates([1, 2, 2, 3, 1])
    # TODO: Fix the function and get correct result
    fixed_result = None
    test_passed = False

    return buggy_result, fixed_result, test_passed


def write_parametrized_tests():
    """Write parametrized-style tests for a function.

    Test the built-in abs() function with multiple inputs.
    Return a list of (input, expected, passed) tuples.

    Returns:
        list: List of (input, expected, passed) tuples.
    """
    # TODO: Create test cases and run them
    test_cases = [
        # (input, expected_output),
        # Add at least 6 test cases covering:
        # positive, negative, zero, float, large numbers
    ]
    results = []
    # TODO: Run each test case and record results
    return results


if __name__ == "__main__":
    # Run tests
    try:
        test_safe_divide()
        print("test_safe_divide: PASSED")
    except (AssertionError, TypeError) as e:
        print(f"test_safe_divide: FAILED - {e}")

    try:
        test_flatten()
        print("test_flatten: PASSED")
    except (AssertionError, TypeError) as e:
        print(f"test_flatten: FAILED - {e}")

    try:
        test_count_vowels()
        print("test_count_vowels: PASSED")
    except (AssertionError, TypeError) as e:
        print(f"test_count_vowels: FAILED - {e}")

    buggy, fixed, passed = test_driven_debug()
    print(f"test_driven_debug: buggy={buggy}, fixed={fixed}, passed={passed}")

    results = write_parametrized_tests()
    if results:
        all_passed = all(r[2] for r in results)
        print(f"write_parametrized_tests: {len(results)} tests, "
              f"all passed: {all_passed}")
    else:
        print("write_parametrized_tests: No tests written yet")

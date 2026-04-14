"""
Exercise 02: Print Debugging

Practice strategic print debugging techniques including
labeled output, repr(), and data flow tracing.
"""


def find_bug_with_prints(numbers):
    """Find and fix the bug using strategic print statements.

    This function should return the sum of all even numbers
    in the list, but it has a bug. Add print statements to
    find the bug, then fix it.

    Args:
        numbers: A list of integers.

    Returns:
        int: Sum of all even numbers.
    """
    # TODO: Add strategic print statements to find the bug, then fix it
    total = 0
    for n in numbers:
        if n % 2 == 0:
            total = n  # BUG: should be total += n
    return total


def find_whitespace_bug(items):
    """Find the whitespace bug using repr().

    This function should count how many items match "hello",
    but some items have hidden whitespace. Use !r formatting
    to find and fix the issue.

    Args:
        items: A list of strings.

    Returns:
        int: Count of items that are exactly "hello".
    """
    # TODO: Use repr() / !r to find the whitespace bug, then fix it
    count = 0
    for item in items:
        if item == "hello":
            count += 1
    return count


def create_debug_print(enabled=True):
    """Create a debug_print function with enable/disable flag.

    Returns a function that prints debug messages when enabled
    and does nothing when disabled. Each message should be
    prefixed with "[DEBUG]".

    Args:
        enabled: Whether debug output is enabled.

    Returns:
        A callable that prints debug messages when enabled.
    """
    # TODO: Implement this
    pass


def trace_pipeline(data):
    """Trace data through a transformation pipeline.

    Apply these transformations in order and add labeled prints
    at each step to trace the data flow:
    1. Strip whitespace from each string
    2. Convert to lowercase
    3. Remove duplicates (preserve order)
    4. Sort alphabetically

    Args:
        data: A list of strings.

    Returns:
        list: Transformed list of strings.
    """
    # TODO: Implement with labeled print statements at each step
    pass


if __name__ == "__main__":
    # Test find_bug_with_prints
    result = find_bug_with_prints([1, 2, 3, 4, 5, 6])
    assert result == 12, f"Expected 12, got {result}"
    print("find_bug_with_prints: PASSED")

    # Test find_whitespace_bug
    items = ["hello", "hello ", "hello", " hello", "hello\t", "world"]
    result = find_whitespace_bug(items)
    # After fix, should strip whitespace and count all "hello" variants
    assert result >= 2, f"Expected at least 2, got {result}"
    print("find_whitespace_bug: PASSED")

    # Test create_debug_print
    debug = create_debug_print(enabled=True)
    assert callable(debug), "Should return a callable"
    debug("test message")  # Should print: [DEBUG] test message

    quiet = create_debug_print(enabled=False)
    quiet("silent message")  # Should print nothing
    print("create_debug_print: PASSED")

    # Test trace_pipeline
    data = ["  Banana  ", "apple", " CHERRY ", "banana", "Apple"]
    result = trace_pipeline(data)
    assert result == ["apple", "banana", "cherry"], f"Got {result}"
    print("trace_pipeline: PASSED")

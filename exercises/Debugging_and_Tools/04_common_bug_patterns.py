"""
Exercise 04: Common Bug Patterns

Practice identifying and fixing common Python bug patterns.
"""
import math


def fix_off_by_one(n):
    """Fix the off-by-one error.

    Return the sum of integers from 1 to n (inclusive).
    Example: fix_off_by_one(5) → 1+2+3+4+5 = 15

    Args:
        n: Upper bound (inclusive).

    Returns:
        int: Sum from 1 to n.
    """
    # TODO: Fix the off-by-one error
    total = 0
    for i in range(1, n):  # BUG: should be range(1, n+1)
        total += i
    return total


def fix_mutable_default(item, items=None):
    """Fix the mutable default argument bug.

    Append item to items and return the list.
    Each call with items=None should start with a fresh list.

    Args:
        item: Item to append.
        items: Optional existing list.

    Returns:
        list: The list with item appended.
    """
    # TODO: Fix the mutable default argument bug
    if items is None:
        items = []  # Students need to add this pattern
    items.append(item)
    return items


def fix_aliasing(original):
    """Fix the aliasing bug.

    Return a modified copy of the list with each element doubled,
    WITHOUT modifying the original list.

    Args:
        original: A list of numbers.

    Returns:
        list: New list with doubled values.
    """
    # TODO: Fix the aliasing bug
    copy = original  # BUG: creates alias, not copy
    for i in range(len(copy)):
        copy[i] = copy[i] * 2
    return copy


def fix_none_handling(data):
    """Fix the None handling bug.

    Process a list of records, extracting the "value" field.
    Some records may be None or missing the "value" key.
    Return the sum of all valid values.

    Args:
        data: A list where each element is a dict or None.

    Returns:
        int: Sum of all valid "value" fields.
    """
    # TODO: Fix to handle None records and missing keys
    total = 0
    for record in data:
        total += record["value"]  # BUG: crashes on None or missing key
    return total


def fix_float_comparison(a, b):
    """Fix the floating-point comparison bug.

    Return True if a and b are approximately equal.

    Args:
        a: First float.
        b: Second float.

    Returns:
        bool: True if approximately equal.
    """
    # TODO: Fix the float comparison
    return a == b  # BUG: direct float comparison


def fix_nested_list(rows, cols):
    """Fix the nested list creation bug.

    Create a rows x cols grid initialized with zeros.
    Modifying one cell should NOT affect other rows.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        list: 2D grid of zeros.
    """
    # TODO: Fix the nested list creation
    return [[0] * cols] * rows  # BUG: all rows are the same object


if __name__ == "__main__":
    # Test fix_off_by_one
    assert fix_off_by_one(5) == 15, f"Got {fix_off_by_one(5)}"
    assert fix_off_by_one(1) == 1
    assert fix_off_by_one(10) == 55
    print("fix_off_by_one: PASSED")

    # Test fix_mutable_default
    r1 = fix_mutable_default("a")
    r2 = fix_mutable_default("b")
    assert r1 == ["a"], f"Got {r1}"
    assert r2 == ["b"], f"Got {r2}"
    print("fix_mutable_default: PASSED")

    # Test fix_aliasing
    original = [1, 2, 3]
    result = fix_aliasing(original)
    assert result == [2, 4, 6], f"Got {result}"
    assert original == [1, 2, 3], f"Original was modified: {original}"
    print("fix_aliasing: PASSED")

    # Test fix_none_handling
    data = [
        {"value": 10},
        None,
        {"value": 20},
        {"name": "no value"},
        {"value": 30},
    ]
    result = fix_none_handling(data)
    assert result == 60, f"Got {result}"
    print("fix_none_handling: PASSED")

    # Test fix_float_comparison
    assert fix_float_comparison(0.1 + 0.2, 0.3) is True
    assert fix_float_comparison(1.0, 1.0) is True
    assert fix_float_comparison(1.0, 2.0) is False
    print("fix_float_comparison: PASSED")

    # Test fix_nested_list
    grid = fix_nested_list(3, 3)
    grid[0][0] = 1
    assert grid[1][0] == 0, "Modifying row 0 should not affect row 1"
    assert grid[0][0] == 1
    print("fix_nested_list: PASSED")

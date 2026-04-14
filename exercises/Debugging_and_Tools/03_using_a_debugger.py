"""
Exercise 03: Using a Debugger

Practice using breakpoint() and pdb to debug functions.
Run with: python -m pdb 03_using_a_debugger.py
"""


def find_bug_with_debugger(numbers):
    """Find and fix the bug using a debugger.

    This function should return the product of all non-zero numbers.
    Use breakpoint() to pause and inspect variables.

    Args:
        numbers: A list of numbers.

    Returns:
        int/float: Product of all non-zero numbers, or 1 if all are zero.
    """
    # TODO: Use breakpoint() to find and fix the bug
    result = 0  # BUG: should start at 1 for multiplication
    for n in numbers:
        if n != 0:
            result *= n
    return result


def find_accumulation_bug(records):
    """Find and fix the accumulation bug using a debugger.

    This function should return a dict mapping each category
    to its total value. Use breakpoint() to track the accumulator.

    Args:
        records: A list of dicts with "category" and "value" keys.

    Returns:
        dict: Category to total value mapping.
    """
    # TODO: Use breakpoint() to find and fix the bug
    totals = {}
    for record in records:
        cat = record["category"]
        val = record["value"]
        totals[cat] = val  # BUG: should accumulate, not overwrite
    return totals


def find_recursion_bug(n):
    """Find and fix the recursion bug using a debugger.

    This function should return the nth Fibonacci number.
    fib(0) = 0, fib(1) = 1, fib(n) = fib(n-1) + fib(n-2).

    Use conditional breakpoints or print to debug.

    Args:
        n: The index (non-negative integer).

    Returns:
        int: The nth Fibonacci number.
    """
    # TODO: Find and fix the bug
    if n == 0:
        return 0
    if n == 1:
        return 1
    return find_recursion_bug(n - 1) + find_recursion_bug(n - 3)  # BUG: n-3


def find_nested_bug(matrix):
    """Find and fix the bug in this matrix operation using a debugger.

    This function should return the sum of each row as a list.
    Example: [[1,2,3],[4,5,6]] -> [6, 15]

    Use up/down in pdb to navigate the call stack.

    Args:
        matrix: A 2D list of numbers.

    Returns:
        list: Sum of each row.
    """
    # TODO: Find and fix the bug
    def sum_row(row):
        total = 0
        for val in row:
            total += val
        return total

    results = []
    for row in matrix:
        results.append(sum_row(row))
        results.append(0)  # BUG: extra zero appended
    return results


if __name__ == "__main__":
    # Test find_bug_with_debugger
    assert find_bug_with_debugger([2, 3, 4]) == 24, "2*3*4=24"
    assert find_bug_with_debugger([5, 0, 3]) == 15, "5*3=15 (skip 0)"
    assert find_bug_with_debugger([0, 0, 0]) == 1, "All zeros → 1"
    print("find_bug_with_debugger: PASSED")

    # Test find_accumulation_bug
    records = [
        {"category": "A", "value": 10},
        {"category": "B", "value": 20},
        {"category": "A", "value": 30},
        {"category": "B", "value": 5},
    ]
    result = find_accumulation_bug(records)
    assert result == {"A": 40, "B": 25}, f"Got {result}"
    print("find_accumulation_bug: PASSED")

    # Test find_recursion_bug
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    for i, exp in enumerate(expected):
        result = find_recursion_bug(i)
        assert result == exp, f"fib({i}) = {result}, expected {exp}"
    print("find_recursion_bug: PASSED")

    # Test find_nested_bug
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = find_nested_bug(matrix)
    assert result == [6, 15, 24], f"Got {result}"
    print("find_nested_bug: PASSED")

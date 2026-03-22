"""
Exercise 06: Data Structures

Practice list manipulation, dict operations, set problems, and matrices.
"""


def rotate_list(lst, k):
    """Rotate a list to the right by k positions.

    Example: rotate_list([1, 2, 3, 4, 5], 2) -> [4, 5, 1, 2, 3]

    Args:
        lst: Input list.
        k: Number of positions to rotate (non-negative).

    Returns:
        A new rotated list.
    """
    # TODO: Implement this
    pass


def merge_dicts(*dicts):
    """Merge multiple dictionaries, summing values for duplicate keys.

    Example: merge_dicts({"a": 1}, {"a": 2, "b": 3}) -> {"a": 3, "b": 3}

    Args:
        *dicts: Variable number of dictionaries.

    Returns:
        A single merged dictionary.
    """
    # TODO: Implement this
    pass


def invert_dict(d):
    """Invert a dictionary, swapping keys and values.

    If multiple keys map to the same value, collect them in a list.
    Example: {"a": 1, "b": 1, "c": 2} -> {1: ["a", "b"], 2: ["c"]}

    Args:
        d: Input dictionary.

    Returns:
        Inverted dictionary with values as lists of original keys.
    """
    # TODO: Implement this
    pass


def set_operations(set_a, set_b):
    """Return a dict with common set operations.

    Keys: "union", "intersection", "a_minus_b", "symmetric_diff"

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Dict with set operation results.
    """
    # TODO: Implement this
    pass


def transpose_matrix(matrix):
    """Transpose a 2D matrix (list of lists).

    Example: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]

    Args:
        matrix: A 2D list (list of lists), assumed rectangular.

    Returns:
        Transposed matrix.
    """
    # TODO: Implement this
    pass


def group_by(items, key_func):
    """Group items by the result of key_func.

    Example: group_by([1,2,3,4,5,6], lambda x: x % 2)
             -> {0: [2, 4, 6], 1: [1, 3, 5]}

    Args:
        items: Iterable of items.
        key_func: Function that returns a grouping key.

    Returns:
        Dict mapping keys to lists of items.
    """
    # TODO: Implement this
    pass


# === Tests ===

assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3], "Rotate right 2"
assert rotate_list([1, 2, 3], 0) == [1, 2, 3], "Rotate 0"
assert rotate_list([1, 2, 3], 3) == [1, 2, 3], "Rotate full cycle"

assert merge_dicts({"a": 1}, {"a": 2, "b": 3}) == {"a": 3, "b": 3}, "Merge 2"
assert merge_dicts({"x": 10}, {"y": 20}, {"x": 5}) == {"x": 15, "y": 20}, "Merge 3"

inv = invert_dict({"a": 1, "b": 1, "c": 2})
assert inv[1] == ["a", "b"] or set(inv[1]) == {"a", "b"}, "Invert dup values"
assert inv[2] == ["c"], "Invert unique value"

ops = set_operations({1, 2, 3}, {2, 3, 4})
assert ops["union"] == {1, 2, 3, 4}, "Union"
assert ops["intersection"] == {2, 3}, "Intersection"
assert ops["a_minus_b"] == {1}, "Difference"
assert ops["symmetric_diff"] == {1, 4}, "Symmetric diff"

assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]], "Transpose"
assert transpose_matrix([[1]]) == [[1]], "Transpose 1x1"

grouped = group_by([1, 2, 3, 4, 5, 6], lambda x: x % 2)
assert grouped[0] == [2, 4, 6], "Group even"
assert grouped[1] == [1, 3, 5], "Group odd"

print("All tests passed!")

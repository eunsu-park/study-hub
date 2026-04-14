"""
Exercise 01: Arrays and Lists

Practice dynamic arrays, slicing, two-pointer,
and sliding window techniques.
"""


def rotate_left(arr, k):
    """Rotate array left by k positions.

    Args:
        arr: List of elements.
        k: Number of positions to rotate.

    Returns:
        A new list rotated left by k positions.

    >>> rotate_left([1, 2, 3, 4, 5], 2)
    [3, 4, 5, 1, 2]
    >>> rotate_left([1, 2, 3], 0)
    [1, 2, 3]
    """
    # TODO: Implement this
    pass


def max_sum_subarray(nums, k):
    """Find the maximum sum of any contiguous subarray of length k.

    Use the sliding window technique for O(n) time.

    Args:
        nums: List of integers.
        k: Window size.

    Returns:
        Maximum sum, or None if len(nums) < k.

    >>> max_sum_subarray([2, 1, 5, 1, 3, 2], 3)
    9
    >>> max_sum_subarray([1, 2], 3)
    """
    # TODO: Implement this
    pass


def two_sum_sorted(nums, target):
    """Find indices of two numbers in a sorted list that sum to target.

    Use the two-pointer technique for O(n) time, O(1) space.

    Args:
        nums: Sorted list of integers.
        target: Target sum.

    Returns:
        Tuple (i, j) of indices, or None if not found.

    >>> two_sum_sorted([1, 2, 4, 7, 11, 15], 9)
    (1, 3)
    """
    # TODO: Implement this
    pass


def flatten_matrix(matrix):
    """Flatten a 2D matrix into a 1D list.

    Args:
        matrix: List of lists.

    Returns:
        Flattened list.

    >>> flatten_matrix([[1, 2], [3, 4], [5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    # TODO: Implement this
    pass


def remove_duplicates_sorted(nums):
    """Remove duplicates from a sorted list in-place.

    Returns the number of unique elements.
    The first k elements of nums should contain the unique values.

    Args:
        nums: Sorted list of integers (modified in-place).

    Returns:
        Number of unique elements.

    >>> nums = [1, 1, 2, 2, 3, 4, 4]
    >>> k = remove_duplicates_sorted(nums)
    >>> k
    4
    >>> nums[:k]
    [1, 2, 3, 4]
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    # Test rotate_left
    assert rotate_left([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]
    assert rotate_left([1, 2, 3], 0) == [1, 2, 3]
    assert rotate_left([1], 5) == [1]
    print("rotate_left: PASSED")

    # Test max_sum_subarray
    assert max_sum_subarray([2, 1, 5, 1, 3, 2], 3) == 9
    assert max_sum_subarray([1, 2], 3) is None
    assert max_sum_subarray([4, 2, 1, 7, 8, 1, 2, 8, 1, 0], 3) == 16
    print("max_sum_subarray: PASSED")

    # Test two_sum_sorted
    assert two_sum_sorted([1, 2, 4, 7, 11, 15], 9) == (1, 3)
    assert two_sum_sorted([1, 2, 3], 10) is None
    print("two_sum_sorted: PASSED")

    # Test flatten_matrix
    assert flatten_matrix([[1, 2], [3, 4], [5, 6]]) == [1, 2, 3, 4, 5, 6]
    assert flatten_matrix([]) == []
    print("flatten_matrix: PASSED")

    # Test remove_duplicates_sorted
    nums = [1, 1, 2, 2, 3, 4, 4]
    k = remove_duplicates_sorted(nums)
    assert k == 4
    assert nums[:k] == [1, 2, 3, 4]
    print("remove_duplicates_sorted: PASSED")

    print("\nAll tests passed!")

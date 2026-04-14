"""
Exercise 13: Searching Fundamentals

Practice binary search and its variants.
"""


def binary_search(arr, target):
    """Implement binary search on a sorted array.

    Returns index of target, or -1 if not found.

    >>> binary_search([1, 3, 5, 7, 9], 5)
    2
    >>> binary_search([1, 3, 5, 7, 9], 4)
    -1
    """
    # TODO: Implement this
    pass


def first_occurrence(arr, target):
    """Find the index of the first occurrence of target.

    >>> first_occurrence([1, 2, 4, 4, 4, 6, 8], 4)
    2
    >>> first_occurrence([1, 2, 3], 5)
    -1
    """
    # TODO: Implement this
    pass


def last_occurrence(arr, target):
    """Find the index of the last occurrence of target.

    >>> last_occurrence([1, 2, 4, 4, 4, 6, 8], 4)
    4
    >>> last_occurrence([1, 2, 3], 5)
    -1
    """
    # TODO: Implement this
    pass


def count_occurrences(arr, target):
    """Count occurrences of target in sorted array.

    Must run in O(log n) time.

    >>> count_occurrences([1, 2, 4, 4, 4, 6, 8], 4)
    3
    >>> count_occurrences([1, 2, 3], 5)
    0
    """
    # TODO: Implement this
    pass


def search_rotated(nums, target):
    """Search in a rotated sorted array.

    The array was sorted then rotated at some pivot.

    >>> search_rotated([4, 5, 6, 7, 0, 1, 2], 0)
    4
    >>> search_rotated([4, 5, 6, 7, 0, 1, 2], 3)
    -1
    """
    # TODO: Implement this
    pass


def find_peak(arr):
    """Find a peak element index using binary search.

    A peak is an element greater than its neighbors.

    >>> find_peak([1, 3, 20, 4, 1, 0]) in [2]
    True
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    assert binary_search([1, 3, 5, 7, 9], 5) == 2
    assert binary_search([1, 3, 5, 7, 9], 4) == -1
    assert binary_search([], 1) == -1
    print("binary_search: PASSED")

    assert first_occurrence([1, 2, 4, 4, 4, 6, 8], 4) == 2
    assert first_occurrence([1, 2, 3], 5) == -1
    print("first_occurrence: PASSED")

    assert last_occurrence([1, 2, 4, 4, 4, 6, 8], 4) == 4
    assert last_occurrence([1, 2, 3], 5) == -1
    print("last_occurrence: PASSED")

    assert count_occurrences([1, 2, 4, 4, 4, 6, 8], 4) == 3
    assert count_occurrences([1, 2, 3], 5) == 0
    print("count_occurrences: PASSED")

    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
    print("search_rotated: PASSED")

    peak = find_peak([1, 3, 20, 4, 1, 0])
    assert peak == 2
    print("find_peak: PASSED")

    print("\nAll tests passed!")

"""
Exercise 12: Sorting Fundamentals

Practice sorting algorithm implementations.
"""


def insertion_sort(arr):
    """Implement insertion sort.

    Must be in-place and stable. Return the sorted list.

    >>> insertion_sort([5, 3, 8, 1, 2])
    [1, 2, 3, 5, 8]
    >>> insertion_sort([])
    []
    """
    # TODO: Implement this
    pass


def merge_sort(arr):
    """Implement merge sort.

    Return a new sorted list (do not modify the original).

    >>> merge_sort([5, 3, 8, 1, 2])
    [1, 2, 3, 5, 8]
    """
    # TODO: Implement this
    pass


def quick_sort(arr):
    """Implement quick sort.

    Return a new sorted list.

    >>> quick_sort([5, 3, 8, 1, 2])
    [1, 2, 3, 5, 8]
    """
    # TODO: Implement this
    pass


def sort_colors(nums):
    """Sort an array of 0s, 1s, and 2s in-place (Dutch National Flag).

    Must be single-pass, O(1) space.

    >>> nums = [2, 0, 2, 1, 1, 0]
    >>> sort_colors(nums)
    >>> nums
    [0, 0, 1, 1, 2, 2]
    """
    # TODO: Implement this
    pass


def merge_sorted_arrays(arr1, arr2):
    """Merge two sorted arrays into one sorted array.

    >>> merge_sorted_arrays([1, 3, 5], [2, 4, 6])
    [1, 2, 3, 4, 5, 6]
    >>> merge_sorted_arrays([], [1, 2])
    [1, 2]
    """
    # TODO: Implement this
    pass


def find_kth_smallest(arr, k):
    """Find the kth smallest element (1-based).

    You may modify the array. Use quickselect or sorting.

    >>> find_kth_smallest([7, 10, 4, 3, 20, 15], 3)
    7
    >>> find_kth_smallest([7, 10, 4, 3, 20, 15], 1)
    3
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    assert insertion_sort([5, 3, 8, 1, 2]) == [1, 2, 3, 5, 8]
    assert insertion_sort([]) == []
    assert insertion_sort([1]) == [1]
    print("insertion_sort: PASSED")

    assert merge_sort([5, 3, 8, 1, 2]) == [1, 2, 3, 5, 8]
    assert merge_sort([]) == []
    print("merge_sort: PASSED")

    assert quick_sort([5, 3, 8, 1, 2]) == [1, 2, 3, 5, 8]
    assert quick_sort([3, 3, 3]) == [3, 3, 3]
    print("quick_sort: PASSED")

    nums = [2, 0, 2, 1, 1, 0]
    sort_colors(nums)
    assert nums == [0, 0, 1, 1, 2, 2]
    print("sort_colors: PASSED")

    assert merge_sorted_arrays([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted_arrays([], [1, 2]) == [1, 2]
    print("merge_sorted_arrays: PASSED")

    assert find_kth_smallest([7, 10, 4, 3, 20, 15], 3) == 7
    assert find_kth_smallest([7, 10, 4, 3, 20, 15], 1) == 3
    print("find_kth_smallest: PASSED")

    print("\nAll tests passed!")

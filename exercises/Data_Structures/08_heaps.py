"""
Exercise 08: Heaps

Practice heap operations, priority queue, and heap-based problems.
"""

import heapq


def kth_largest(nums, k):
    """Find the kth largest element using a heap.

    >>> kth_largest([3, 2, 1, 5, 6, 4], 2)
    5
    >>> kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4)
    4
    """
    # TODO: Implement this
    pass


def merge_k_sorted(lists):
    """Merge k sorted lists into one sorted list.

    Use a heap for O(n log k) time.

    >>> merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6]])
    [1, 1, 2, 3, 4, 4, 5, 6]
    >>> merge_k_sorted([])
    []
    """
    # TODO: Implement this
    pass


def top_k_frequent(nums, k):
    """Find the k most frequent elements.

    >>> sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
    [1, 2]
    """
    # TODO: Implement this
    pass


class MedianFinder:
    """Find the median of a number stream.

    Use two heaps: a max-heap for the lower half
    and a min-heap for the upper half.

    >>> mf = MedianFinder()
    >>> mf.add(1); mf.add(2); mf.find_median()
    1.5
    >>> mf.add(3); mf.find_median()
    2
    """

    def __init__(self):
        # TODO: Initialize two heaps
        pass

    def add(self, num):
        # TODO: Implement this
        pass

    def find_median(self):
        # TODO: Implement this
        pass


if __name__ == "__main__":
    assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    assert kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    print("kth_largest: PASSED")

    assert merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6]]) == [1, 1, 2, 3, 4, 4, 5, 6]
    assert merge_k_sorted([]) == []
    print("merge_k_sorted: PASSED")

    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    print("top_k_frequent: PASSED")

    mf = MedianFinder()
    mf.add(1); mf.add(2)
    assert mf.find_median() == 1.5
    mf.add(3)
    assert mf.find_median() == 2
    print("MedianFinder: PASSED")

    print("\nAll tests passed!")

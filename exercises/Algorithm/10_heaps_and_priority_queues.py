"""
Exercises for Lesson 10: Heaps and Priority Queues
Topic: Algorithm

Solutions to practice problems from the lesson.
The lesson provides recommended problems: Max Heap, Kth Largest, Running Median,
Merge K Sorted Lists, Top K Frequent. We implement key ones here.
"""

import heapq
from collections import Counter


# === Exercise 1: Kth Largest Element Using Heap ===
# Problem: Find the kth largest element in an unsorted array.
#   Input: [3, 2, 1, 5, 6, 4], k = 2
#   Output: 5
# Approach: Maintain a min-heap of size k. After processing all elements,
#   the heap root is the kth largest.

def exercise_1():
    """Solution: Min-heap of size k for O(n log k) time."""
    def find_kth_largest(nums, k):
        # Use a min-heap of size k. After processing all elements,
        # the smallest element in the heap is the kth largest overall.
        min_heap = []
        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)  # remove smallest
        return min_heap[0]

    tests = [
        ([3, 2, 1, 5, 6, 4], 2, 5),
        ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4),
        ([1], 1, 1),
        ([7, 6, 5, 4, 3, 2, 1], 5, 3),
    ]

    for nums, k, expected in tests:
        result = find_kth_largest(nums, k)
        print(f"kth_largest({nums}, k={k}) = {result}")
        assert result == expected

    print("All Kth Largest tests passed!")


# === Exercise 2: Running Median (Two Heaps) ===
# Problem: Maintain a running median as numbers are added one by one.
#   After adding each number, report the current median.
# Approach: Use a max-heap for the lower half and a min-heap for the upper half.
#   Python only has min-heap, so negate values for the max-heap.

def exercise_2():
    """Solution: Two heaps for O(log n) insert and O(1) median."""
    class MedianFinder:
        def __init__(self):
            self.max_heap = []  # lower half (negated values for max-heap behavior)
            self.min_heap = []  # upper half

        def add_num(self, num):
            # Always add to max_heap first (negated for max-heap)
            heapq.heappush(self.max_heap, -num)

            # Ensure max_heap's largest <= min_heap's smallest
            if self.min_heap and (-self.max_heap[0] > self.min_heap[0]):
                val = -heapq.heappop(self.max_heap)
                heapq.heappush(self.min_heap, val)

            # Balance sizes: max_heap can have at most 1 more element
            if len(self.max_heap) > len(self.min_heap) + 1:
                val = -heapq.heappop(self.max_heap)
                heapq.heappush(self.min_heap, val)
            elif len(self.min_heap) > len(self.max_heap):
                val = heapq.heappop(self.min_heap)
                heapq.heappush(self.max_heap, -val)

        def find_median(self):
            if len(self.max_heap) > len(self.min_heap):
                return -self.max_heap[0]
            else:
                return (-self.max_heap[0] + self.min_heap[0]) / 2.0

    mf = MedianFinder()
    sequence = [5, 15, 1, 3, 8, 7, 9, 10, 6, 2]
    medians = []

    for num in sequence:
        mf.add_num(num)
        median = mf.find_median()
        medians.append(median)
        print(f"Add {num:2d} -> median = {median}")

    # Verify final medians against sorted check
    for i in range(len(sequence)):
        sorted_prefix = sorted(sequence[:i + 1])
        n = len(sorted_prefix)
        if n % 2 == 1:
            expected = sorted_prefix[n // 2]
        else:
            expected = (sorted_prefix[n // 2 - 1] + sorted_prefix[n // 2]) / 2.0
        assert medians[i] == expected, f"At step {i+1}: expected {expected}, got {medians[i]}"

    print("All Running Median tests passed!")


# === Exercise 3: Top K Frequent Elements ===
# Problem: Given an array, return the k most frequent elements.
#   Input: [1,1,1,2,2,3], k = 2
#   Output: [1, 2]
# Approach: Count frequencies, then use a min-heap of size k.

def exercise_3():
    """Solution: Frequency count + heap for O(n log k)."""
    def top_k_frequent(nums, k):
        freq = Counter(nums)

        # Use a min-heap of size k based on frequency.
        # heapq.nlargest is a clean way to get the top-k.
        return heapq.nlargest(k, freq.keys(), key=freq.get)

    # Alternative: manual min-heap approach
    def top_k_frequent_manual(nums, k):
        freq = Counter(nums)
        min_heap = []

        for num, count in freq.items():
            heapq.heappush(min_heap, (count, num))
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        return [num for count, num in min_heap]

    tests = [
        ([1, 1, 1, 2, 2, 3], 2, {1, 2}),
        ([1], 1, {1}),
        ([4, 4, 4, 6, 6, 2, 2, 2, 2], 2, {2, 4}),
    ]

    for nums, k, expected_set in tests:
        result = top_k_frequent(nums, k)
        print(f"top_k_frequent({nums}, k={k}) = {result}")
        assert set(result) == expected_set

        result2 = top_k_frequent_manual(nums, k)
        assert set(result2) == expected_set

    print("All Top K Frequent tests passed!")


# === Exercise 4: Merge K Sorted Lists ===
# Problem: Merge k sorted lists into one sorted list.
#   Input: [[1,4,5], [1,3,4], [2,6]]
#   Output: [1,1,2,3,4,4,5,6]
# Approach: Use a min-heap to always extract the smallest element across all lists.

def exercise_4():
    """Solution: Min-heap merging in O(N log k) where N is total elements."""
    def merge_k_sorted(lists):
        min_heap = []
        result = []

        # Initialize heap with first element from each list.
        # Tuple: (value, list_index, element_index)
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(min_heap, (lst[0], i, 0))

        while min_heap:
            val, list_idx, elem_idx = heapq.heappop(min_heap)
            result.append(val)

            # Push next element from the same list
            next_idx = elem_idx + 1
            if next_idx < len(lists[list_idx]):
                heapq.heappush(min_heap, (lists[list_idx][next_idx], list_idx, next_idx))

        return result

    tests = [
        ([[1, 4, 5], [1, 3, 4], [2, 6]], [1, 1, 2, 3, 4, 4, 5, 6]),
        ([], []),
        ([[]], []),
        ([[1], [2], [3]], [1, 2, 3]),
    ]

    for lists, expected in tests:
        result = merge_k_sorted(lists)
        print(f"merge_k_sorted({lists}) = {result}")
        assert result == expected

    print("All Merge K Sorted Lists tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Kth Largest Element Using Heap ===")
    exercise_1()
    print("\n=== Exercise 2: Running Median ===")
    exercise_2()
    print("\n=== Exercise 3: Top K Frequent Elements ===")
    exercise_3()
    print("\n=== Exercise 4: Merge K Sorted Lists ===")
    exercise_4()
    print("\nAll exercises completed!")

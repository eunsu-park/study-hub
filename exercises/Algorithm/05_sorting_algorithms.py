"""
Exercises for Lesson 05: Sorting Algorithms
Topic: Algorithm

Solutions to practice problems from the lesson.
"""

import random


# === Exercise 1: Kth Largest Element ===
# Problem: Find the Kth largest element in O(n) average time without full sorting.
# Approach: Quick Select algorithm using partitioning from Quick Sort.

def exercise_1():
    """Solution: Quick Select for O(n) average-case kth element."""
    def find_kth_largest(arr, k):
        # Convert kth largest to kth smallest index:
        # kth largest = (n-k)th smallest (0-indexed)
        k_idx = len(arr) - k

        def quick_select(left, right):
            # Lomuto partition: choose rightmost element as pivot
            pivot = arr[right]
            i = left

            for j in range(left, right):
                if arr[j] < pivot:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1

            arr[i], arr[right] = arr[right], arr[i]

            if i == k_idx:
                return arr[i]
            elif i < k_idx:
                return quick_select(i + 1, right)
            else:
                return quick_select(left, i - 1)

        return quick_select(0, len(arr) - 1)

    # Test cases
    arr1 = [3, 2, 1, 5, 6, 4]
    k1 = 2
    result = find_kth_largest(arr1[:], k1)  # copy to preserve original
    print(f"Array: [3,2,1,5,6,4], k=2 -> {result}")
    assert result == 5

    arr2 = [3, 2, 3, 1, 2, 4, 5, 5, 6]
    k2 = 4
    result = find_kth_largest(arr2[:], k2)
    print(f"Array: [3,2,3,1,2,4,5,5,6], k=4 -> {result}")
    assert result == 4

    arr3 = [1]
    k3 = 1
    result = find_kth_largest(arr3[:], k3)
    print(f"Array: [1], k=1 -> {result}")
    assert result == 1

    # Large random test
    arr4 = list(range(1, 101))
    random.shuffle(arr4)
    result = find_kth_largest(arr4[:], 10)
    print(f"Array: shuffled 1..100, k=10 -> {result}")
    assert result == 91

    print("All Kth Largest Element tests passed!")


# === Exercise 2: Color Sort (Dutch National Flag) ===
# Problem: Sort an array of 0s, 1s, and 2s in one pass.
#   Input:  [2, 0, 2, 1, 1, 0]
#   Output: [0, 0, 1, 1, 2, 2]
# Approach: Three-way partition with low, mid, high pointers.

def exercise_2():
    """Solution: Dutch National Flag - O(n) time, O(1) space, single pass."""
    def sort_colors(arr):
        low, mid, high = 0, 0, len(arr) - 1

        while mid <= high:
            if arr[mid] == 0:
                # Swap to the low region and advance both pointers.
                # low <= mid always, so arr[low] is either 0 or 1 (already processed).
                arr[low], arr[mid] = arr[mid], arr[low]
                low += 1
                mid += 1
            elif arr[mid] == 1:
                # 1 is in the correct middle region, just advance.
                mid += 1
            else:  # arr[mid] == 2
                # Swap to the high region. Do NOT advance mid because the
                # swapped value from high hasn't been examined yet.
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1

        return arr

    # Test cases
    tests = [
        ([2, 0, 2, 1, 1, 0], [0, 0, 1, 1, 2, 2]),
        ([2, 0, 1], [0, 1, 2]),
        ([0], [0]),
        ([1, 0], [0, 1]),
        ([0, 0, 0], [0, 0, 0]),
        ([2, 2, 2, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 2]),
    ]

    for arr, expected in tests:
        original = arr[:]
        result = sort_colors(arr)
        print(f"{original} -> {result}")
        assert result == expected

    print("All Color Sort tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Kth Largest Element ===")
    exercise_1()
    print("\n=== Exercise 2: Color Sort (Dutch National Flag) ===")
    exercise_2()
    print("\nAll exercises completed!")

"""
Exercises for Lesson 02: Arrays and Strings
Topic: Algorithm

Solutions to practice problems from the lesson.
"""

from collections import defaultdict


# === Exercise 1: Array Rotation ===
# Problem: Rotate array to the right by k positions.
#   Input: [1, 2, 3, 4, 5], k = 2
#   Output: [4, 5, 1, 2, 3]
# Constraint: O(1) space using three reversals.

def exercise_1():
    """Solution using three reversals: O(n) time, O(1) space."""
    def rotate(arr, k):
        n = len(arr)
        k = k % n  # k might be larger than n

        def reverse(start, end):
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1

        # Three-step reversal:
        # 1. Reverse entire array:   [5, 4, 3, 2, 1]
        # 2. Reverse first k elems:  [4, 5, 3, 2, 1]
        # 3. Reverse remaining:      [4, 5, 1, 2, 3]
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)

    # Test case 1
    arr1 = [1, 2, 3, 4, 5]
    print(f"Original: {arr1}, k=2")
    rotate(arr1, 2)
    print(f"Rotated:  {arr1}")
    assert arr1 == [4, 5, 1, 2, 3], f"Expected [4, 5, 1, 2, 3], got {arr1}"

    # Test case 2
    arr2 = [1, 2, 3, 4, 5, 6, 7]
    print(f"\nOriginal: {arr2}, k=3")
    rotate(arr2, 3)
    print(f"Rotated:  {arr2}")
    assert arr2 == [5, 6, 7, 1, 2, 3, 4]

    # Test case 3: k larger than n
    arr3 = [1, 2, 3]
    print(f"\nOriginal: {arr3}, k=5")
    rotate(arr3, 5)
    print(f"Rotated:  {arr3}")
    assert arr3 == [2, 3, 1]

    print("\nAll test cases passed!")


# === Exercise 2: Count Subarrays with Sum k ===
# Problem: Count the number of contiguous subarrays with sum equal to k.
#   Input: [1, 1, 1], k = 2
#   Output: 2 (two subarrays [1,1])
# Approach: Use prefix sum + hash map.
#   prefix[j] - prefix[i] = k  =>  prefix[i] = prefix[j] - k

def exercise_2():
    """Solution using prefix sum + hash map: O(n) time, O(n) space."""
    def subarray_sum(arr, k):
        count = 0
        prefix_sum = 0
        # Track how many times each prefix sum has appeared.
        # Initialize with {0: 1} to handle the case where the subarray
        # starts from index 0 (prefix_sum itself equals k).
        prefix_count = defaultdict(int)
        prefix_count[0] = 1

        for num in arr:
            prefix_sum += num

            # If (prefix_sum - k) appeared before, then the subarray
            # between that point and current index sums to k.
            if prefix_sum - k in prefix_count:
                count += prefix_count[prefix_sum - k]

            prefix_count[prefix_sum] += 1

        return count

    # Test case 1
    arr1 = [1, 1, 1]
    k1 = 2
    result1 = subarray_sum(arr1, k1)
    print(f"Array: {arr1}, k={k1}")
    print(f"Count: {result1}")
    assert result1 == 2

    # Test case 2
    arr2 = [1, 2, 3]
    k2 = 3
    result2 = subarray_sum(arr2, k2)
    print(f"\nArray: {arr2}, k={k2}")
    print(f"Count: {result2}")
    assert result2 == 2  # [1,2] and [3]

    # Test case 3: negative numbers
    arr3 = [1, -1, 0]
    k3 = 0
    result3 = subarray_sum(arr3, k3)
    print(f"\nArray: {arr3}, k={k3}")
    print(f"Count: {result3}")
    assert result3 == 3  # [1,-1], [-1,0], [1,-1,0]

    # Test case 4
    arr4 = [3, 4, 7, 2, -3, 1, 4, 2]
    k4 = 7
    result4 = subarray_sum(arr4, k4)
    print(f"\nArray: {arr4}, k={k4}")
    print(f"Count: {result4}")
    assert result4 == 4  # [3,4], [7], [7,2,-3,1], [1,4,2]

    print("\nAll test cases passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Array Rotation ===")
    exercise_1()
    print("\n=== Exercise 2: Count Subarrays with Sum k ===")
    exercise_2()
    print("\nAll exercises completed!")

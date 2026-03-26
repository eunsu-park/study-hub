"""
Exercises for Lesson 07: Divide and Conquer
Topic: Algorithm

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Count Inversions ===
# Problem: Count pairs where i < j and arr[i] > arr[j].
#   Input: [2, 4, 1, 3, 5]
#   Output: 3 (inversions: (2,1), (4,1), (4,3))
# Approach: Count during the merge sort process. When an element from the
#   right half is placed before remaining elements in the left half, all
#   remaining left-half elements form inversions with it.

def exercise_1():
    """Solution: Merge sort-based inversion count in O(n log n)."""
    def count_inversions(arr):
        def merge_count(arr, temp, left, mid, right):
            i = left       # pointer for left subarray
            j = mid + 1    # pointer for right subarray
            k = left       # pointer for merged result
            inv_count = 0

            while i <= mid and j <= right:
                if arr[i] <= arr[j]:
                    temp[k] = arr[i]
                    i += 1
                else:
                    temp[k] = arr[j]
                    # Key insight: if arr[j] < arr[i], then arr[j] is also
                    # less than all elements arr[i..mid] (because left half
                    # is sorted). So we get (mid - i + 1) inversions.
                    inv_count += (mid - i + 1)
                    j += 1
                k += 1

            # Copy remaining elements from left half
            while i <= mid:
                temp[k] = arr[i]
                i += 1
                k += 1

            # Copy remaining elements from right half
            while j <= right:
                temp[k] = arr[j]
                j += 1
                k += 1

            # Write merged result back to arr
            for idx in range(left, right + 1):
                arr[idx] = temp[idx]

            return inv_count

        def merge_sort_count(arr, temp, left, right):
            inv_count = 0
            if left < right:
                mid = (left + right) // 2
                inv_count += merge_sort_count(arr, temp, left, mid)
                inv_count += merge_sort_count(arr, temp, mid + 1, right)
                inv_count += merge_count(arr, temp, left, mid, right)
            return inv_count

        n = len(arr)
        temp = [0] * n
        return merge_sort_count(arr, temp, 0, n - 1)

    # Brute force verifier for small inputs
    def brute_force_inversions(arr):
        count = 0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    count += 1
        return count

    # Test cases
    tests = [
        ([2, 4, 1, 3, 5], 3),
        ([1, 2, 3, 4, 5], 0),       # sorted -> 0 inversions
        ([5, 4, 3, 2, 1], 10),      # reverse sorted -> n*(n-1)/2
        ([1, 3, 5, 2, 4, 6], 3),
        ([1], 0),
        ([2, 1], 1),
    ]

    for arr, expected in tests:
        arr_copy = arr[:]
        result = count_inversions(arr_copy)
        bf_result = brute_force_inversions(arr[:])
        print(f"Array: {arr} -> inversions: {result} (brute force: {bf_result})")
        assert result == expected, f"Expected {expected}, got {result}"
        assert result == bf_result, f"Mismatch with brute force"

    print("All Count Inversions tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Count Inversions ===")
    exercise_1()
    print("\nAll exercises completed!")

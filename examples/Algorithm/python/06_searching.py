"""
Binary Search
Binary Search Algorithms

Algorithms that search in sorted data in O(log n) time.
"""

from typing import List, Optional
import bisect


# =============================================================================
# 1. Basic Binary Search
# =============================================================================
def binary_search(arr: List[int], target: int) -> int:
    """
    Find the index of target in a sorted array
    Returns -1 if not found
    Time Complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Prevent overflow

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def binary_search_recursive(arr: List[int], target: int, left: int, right: int) -> int:
    """Binary Search (Recursive version)"""
    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


# =============================================================================
# 2. Lower Bound / Upper Bound
# =============================================================================
def lower_bound(arr: List[int], target: int) -> int:
    """
    Index of the first element greater than or equal to target
    Returns len(arr) if all elements are less than target
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


def upper_bound(arr: List[int], target: int) -> int:
    """
    Index of the first element strictly greater than target
    Returns len(arr) if all elements are less than or equal to target
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left


def count_occurrences(arr: List[int], target: int) -> int:
    """Count occurrences of a specific value"""
    return upper_bound(arr, target) - lower_bound(arr, target)


# =============================================================================
# 3. Find First/Last Position
# =============================================================================
def find_first_position(arr: List[int], target: int) -> int:
    """Index where target first appears (returns -1 if not found)"""
    idx = lower_bound(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1


def find_last_position(arr: List[int], target: int) -> int:
    """Index where target last appears (returns -1 if not found)"""
    idx = upper_bound(arr, target) - 1
    if idx >= 0 and arr[idx] == target:
        return idx
    return -1


# =============================================================================
# 4. Search in Rotated Sorted Array
# =============================================================================
def search_rotated(arr: List[int], target: int) -> int:
    """
    Search in a rotated sorted array
    Example: [4, 5, 6, 7, 0, 1, 2] - rotated from [0,1,2,4,5,6,7]
    Time Complexity: O(log n)
    """
    if not arr:
        return -1

    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        # If the left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # If the right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


# =============================================================================
# 5. Find Minimum in Rotated Array
# =============================================================================
def find_minimum_rotated(arr: List[int]) -> int:
    """Find the minimum value in a rotated sorted array"""
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid

    return arr[left]


# =============================================================================
# 6. Integer Square Root
# =============================================================================
def integer_sqrt(n: int) -> int:
    """
    Integer square root of n (floor)
    Example: sqrt(8) = 2
    """
    if n < 0:
        raise ValueError("Square root of a negative number is undefined")
    if n == 0:
        return 0

    left, right = 1, n

    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid

        if square == n:
            return mid
        elif square < n:
            left = mid + 1
        else:
            right = mid - 1

    return right  # floor(sqrt(n))


# =============================================================================
# 7. Parametric Search
# =============================================================================
def can_split(arr: List[int], m: int, max_sum: int) -> bool:
    """
    Check if the array can be split into m or fewer groups
    where each group sum is at most max_sum
    """
    count = 1
    current_sum = 0

    for num in arr:
        if current_sum + num > max_sum:
            count += 1
            current_sum = num
            if count > m:
                return False
        else:
            current_sum += num

    return True


def split_array_min_largest_sum(arr: List[int], m: int) -> int:
    """
    Split array into m contiguous subarrays
    minimizing the maximum subarray sum
    Time Complexity: O(n log(sum))
    """
    left = max(arr)      # Minimum possible value: largest element
    right = sum(arr)     # Maximum possible value: total sum

    while left < right:
        mid = left + (right - left) // 2
        if can_split(arr, m, mid):
            right = mid
        else:
            left = mid + 1

    return left


# =============================================================================
# 8. Find Peak Element
# =============================================================================
def find_peak_element(arr: List[int]) -> int:
    """
    Find the index of a peak element in the array
    peak: arr[i] > arr[i-1] and arr[i] > arr[i+1]
    Time Complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Binary Search Examples")
    print("=" * 60)

    # 1. Basic Binary Search
    print("\n[1] Basic Binary Search")
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search(arr, target)
    print(f"    Array: {arr}")
    print(f"    Find {target} -> index: {result}")

    # 2. Lower/Upper Bound
    print("\n[2] Lower Bound / Upper Bound")
    arr = [1, 2, 2, 2, 3, 4, 5]
    target = 2
    lb = lower_bound(arr, target)
    ub = upper_bound(arr, target)
    count = count_occurrences(arr, target)
    print(f"    Array: {arr}")
    print(f"    target={target}")
    print(f"    lower_bound: {lb}, upper_bound: {ub}")
    print(f"    Occurrence count: {count}")

    # Compare with bisect module
    print(f"    (bisect_left: {bisect.bisect_left(arr, target)}, "
          f"bisect_right: {bisect.bisect_right(arr, target)})")

    # 3. First/Last Position
    print("\n[3] First/Last Position")
    arr = [5, 7, 7, 8, 8, 8, 10]
    target = 8
    first = find_first_position(arr, target)
    last = find_last_position(arr, target)
    print(f"    Array: {arr}")
    print(f"    First position of {target}: {first}, Last position: {last}")

    # 4. Rotated Array Search
    print("\n[4] Search in Rotated Sorted Array")
    arr = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    result = search_rotated(arr, target)
    print(f"    Rotated array: {arr}")
    print(f"    Find {target} -> index: {result}")

    # 5. Rotated Array Minimum
    print("\n[5] Minimum in Rotated Array")
    arr = [4, 5, 6, 7, 0, 1, 2]
    result = find_minimum_rotated(arr)
    print(f"    Rotated array: {arr}")
    print(f"    Minimum: {result}")

    # 6. Square Root
    print("\n[6] Integer Square Root")
    for n in [4, 8, 16, 17, 100]:
        result = integer_sqrt(n)
        print(f"    sqrt({n}) = {result}")

    # 7. Parametric Search
    print("\n[7] Parametric Search (Array Split)")
    arr = [7, 2, 5, 10, 8]
    m = 2
    result = split_array_min_largest_sum(arr, m)
    print(f"    Array: {arr}, Splits: {m}")
    print(f"    Minimum largest sum: {result}")

    # 8. Peak Element
    print("\n[8] Find Peak Element")
    arr = [1, 2, 1, 3, 5, 6, 4]
    result = find_peak_element(arr)
    print(f"    Array: {arr}")
    print(f"    Peak index: {result} (value: {arr[result]})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

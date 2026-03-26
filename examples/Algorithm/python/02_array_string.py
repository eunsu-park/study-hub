"""
Two Pointer Technique
Two Pointer Technique

Efficiently solves array/list problems using two pointers.
"""

from typing import List, Tuple, Optional


# =============================================================================
# 1. Two Sum (Sorted Array)
# =============================================================================
def two_sum_sorted(arr: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Find indices of two numbers in a sorted array that sum to target
    Time Complexity: O(n), Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return (left, right)
        elif current_sum < target:
            left += 1  # If sum is too small, move left pointer right
        else:
            right -= 1  # If sum is too large, move right pointer left

    return None


# =============================================================================
# 2. Three Sum (3Sum)
# =============================================================================
def three_sum(arr: List[int]) -> List[List[int]]:
    """
    Find all triplets that sum to 0 (remove duplicates)
    Time Complexity: O(n^2), Space Complexity: O(1)
    """
    arr.sort()
    result = []
    n = len(arr)

    for i in range(n - 2):
        # Skip duplicates
        if i > 0 and arr[i] == arr[i - 1]:
            continue

        left, right = i + 1, n - 1

        while left < right:
            total = arr[i] + arr[left] + arr[right]

            if total == 0:
                result.append([arr[i], arr[left], arr[right]])
                # Skip duplicates
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result


# =============================================================================
# 3. Container With Most Water
# =============================================================================
def max_water(heights: List[int]) -> int:
    """
    Maximum amount of water that can be held between two walls
    Time Complexity: O(n), Space Complexity: O(1)
    """
    left, right = 0, len(heights) - 1
    max_area = 0

    while left < right:
        # Calculate current area
        width = right - left
        height = min(heights[left], heights[right])
        area = width * height
        max_area = max(max_area, area)

        # Move the pointer on the shorter side
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1

    return max_area


# =============================================================================
# 4. Merge Two Sorted Arrays
# =============================================================================
def merge_sorted_arrays(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Merge two sorted arrays into one sorted array
    Time Complexity: O(n + m), Space Complexity: O(n + m)
    """
    result = []
    i, j = 0, 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Append remaining elements
    result.extend(arr1[i:])
    result.extend(arr2[j:])

    return result


# =============================================================================
# 5. Remove Duplicates (Sorted Array)
# =============================================================================
def remove_duplicates(arr: List[int]) -> int:
    """
    Remove duplicates from a sorted array (in-place)
    Returns: number of unique elements
    Time Complexity: O(n), Space Complexity: O(1)
    """
    if not arr:
        return 0

    write_idx = 1  # Position to write the next unique value

    for read_idx in range(1, len(arr)):
        if arr[read_idx] != arr[write_idx - 1]:
            arr[write_idx] = arr[read_idx]
            write_idx += 1

    return write_idx


# =============================================================================
# 6. Palindrome Check
# =============================================================================
def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome (comparing only alphanumeric characters)
    Time Complexity: O(n), Space Complexity: O(1)
    """
    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True


# =============================================================================
# 7. Sliding Window (Maximum Sum)
# =============================================================================
def max_sum_subarray(arr: List[int], k: int) -> int:
    """
    Maximum sum of a contiguous subarray of size k
    Time Complexity: O(n), Space Complexity: O(1)
    """
    if len(arr) < k:
        return 0

    # Initial window sum
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Two Pointer Technique Examples")
    print("=" * 60)

    # 1. Two Sum
    print("\n[1] Two Sum (Sorted Array)")
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 10
    result = two_sum_sorted(arr, target)
    print(f"    Array: {arr}")
    print(f"    Target: {target}")
    print(f"    Result: indices {result} -> {arr[result[0]]} + {arr[result[1]]} = {target}")

    # 2. Three Sum
    print("\n[2] Three Sum (3Sum)")
    arr = [-1, 0, 1, 2, -1, -4]
    result = three_sum(arr)
    print(f"    Array: {arr}")
    print(f"    Triplets summing to 0: {result}")

    # 3. Container With Most Water
    print("\n[3] Container With Most Water")
    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    result = max_water(heights)
    print(f"    Heights: {heights}")
    print(f"    Maximum water volume: {result}")

    # 4. Merge Sorted Arrays
    print("\n[4] Merge Two Sorted Arrays")
    arr1 = [1, 3, 5, 7]
    arr2 = [2, 4, 6, 8, 10]
    result = merge_sorted_arrays(arr1, arr2)
    print(f"    Array1: {arr1}")
    print(f"    Array2: {arr2}")
    print(f"    Merged result: {result}")

    # 5. Remove Duplicates
    print("\n[5] Remove Duplicates (in-place)")
    arr = [1, 1, 2, 2, 2, 3, 4, 4, 5]
    count = remove_duplicates(arr)
    print(f"    Original: [1, 1, 2, 2, 2, 3, 4, 4, 5]")
    print(f"    Unique element count: {count}")
    print(f"    Result array (first {count}): {arr[:count]}")

    # 6. Palindrome Check
    print("\n[6] Palindrome Check")
    test_strings = ["A man, a plan, a canal: Panama", "race a car", "Was it a car or a cat I saw?"]
    for s in test_strings:
        result = is_palindrome(s)
        print(f"    '{s}' -> {result}")

    # 7. Sliding Window
    print("\n[7] Sliding Window (Max Sum of Size k)")
    arr = [2, 1, 5, 1, 3, 2]
    k = 3
    result = max_sum_subarray(arr, k)
    print(f"    Array: {arr}, k={k}")
    print(f"    Maximum sum: {result}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

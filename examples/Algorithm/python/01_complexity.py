"""
Time Complexity Comparison Experiment
Time Complexity Comparison

Compares the execution times of algorithms with various time complexities.
"""

import time
import random


def measure_time(func, *args):
    """Measure function execution time"""
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return end - start, result


# =============================================================================
# O(1) - Constant Time
# =============================================================================
def constant_time(arr):
    """Return the first element of an array - O(1)"""
    if arr:
        return arr[0]
    return None


# =============================================================================
# O(log n) - Logarithmic Time
# =============================================================================
def binary_search(arr, target):
    """Binary Search - O(log n)"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# =============================================================================
# O(n) - Linear Time
# =============================================================================
def linear_search(arr, target):
    """Linear Search - O(n)"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


def find_max(arr):
    """Find Maximum Value - O(n)"""
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val


# =============================================================================
# O(n log n) - Linearithmic Time
# =============================================================================
def merge_sort(arr):
    """Merge Sort - O(n log n)"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# =============================================================================
# O(n^2) - Quadratic Time
# =============================================================================
def bubble_sort(arr):
    """Bubble Sort - O(n^2)"""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def has_duplicate_naive(arr):
    """Duplicate Check (naive) - O(n^2)"""
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] == arr[j]:
                return True
    return False


# =============================================================================
# O(2^n) - Exponential Time
# =============================================================================
def fibonacci_recursive(n):
    """Fibonacci (Recursive) - O(2^n)"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_dp(n):
    """Fibonacci (DP) - O(n)"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# =============================================================================
# Experiment and Results Output
# =============================================================================
def run_experiments():
    """Run time complexity experiments"""
    print("=" * 60)
    print("Time Complexity Comparison Experiment")
    print("=" * 60)

    # Prepare data
    sizes = [100, 1000, 10000]

    for size in sizes:
        print(f"\n[ Array Size: {size} ]")
        print("-" * 40)

        arr = list(range(size))
        random_arr = random.sample(range(size * 10), size)
        target = arr[size // 2]  # Middle value

        # O(1) test
        t, _ = measure_time(constant_time, arr)
        print(f"O(1)     Constant Time:   {t * 1000:.6f} ms")

        # O(log n) test
        t, _ = measure_time(binary_search, arr, target)
        print(f"O(log n) Binary Search:   {t * 1000:.6f} ms")

        # O(n) test
        t, _ = measure_time(linear_search, arr, target)
        print(f"O(n)     Linear Search:   {t * 1000:.6f} ms")

        t, _ = measure_time(find_max, random_arr)
        print(f"O(n)     Find Max:        {t * 1000:.6f} ms")

        # O(n log n) test
        if size <= 10000:
            t, _ = measure_time(merge_sort, random_arr)
            print(f"O(n log n) Merge Sort:  {t * 1000:.6f} ms")

        # O(n^2) test (small sizes only)
        if size <= 1000:
            t, _ = measure_time(bubble_sort, random_arr)
            print(f"O(n^2)   Bubble Sort:     {t * 1000:.6f} ms")

    # O(2^n) vs O(n) Fibonacci comparison
    print("\n[ Fibonacci Comparison: O(2^n) vs O(n) ]")
    print("-" * 40)

    for n in [10, 20, 30]:
        t_recursive, _ = measure_time(fibonacci_recursive, n)
        t_dp, _ = measure_time(fibonacci_dp, n)
        print(f"n={n}: Recursive O(2^n) = {t_recursive * 1000:.4f} ms, "
              f"DP O(n) = {t_dp * 1000:.6f} ms")

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiments()

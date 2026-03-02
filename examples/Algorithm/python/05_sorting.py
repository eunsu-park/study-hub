"""
Sorting Algorithms
Sorting Algorithms Comparison

Implementation and comparison of various sorting algorithms.
"""

import random
import time
from typing import List


# =============================================================================
# 1. Bubble Sort
# =============================================================================
def bubble_sort(arr: List[int]) -> List[int]:
    """
    Bubble Sort
    Time: O(n^2), Space: O(1), Stable
    """
    arr = arr.copy()
    n = len(arr)

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break

    return arr


# =============================================================================
# 2. Selection Sort
# =============================================================================
def selection_sort(arr: List[int]) -> List[int]:
    """
    Selection Sort
    Time: O(n^2), Space: O(1), Unstable
    """
    arr = arr.copy()
    n = len(arr)

    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


# =============================================================================
# 3. Insertion Sort
# =============================================================================
def insertion_sort(arr: List[int]) -> List[int]:
    """
    Insertion Sort
    Time: O(n^2), Best O(n), Space: O(1), Stable
    Efficient for nearly sorted data
    """
    arr = arr.copy()
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr


# =============================================================================
# 4. Merge Sort
# =============================================================================
def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge Sort
    Time: O(n log n), Space: O(n), Stable
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Merge two sorted arrays"""
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
# 5. Quick Sort
# =============================================================================
def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick Sort
    Time: Average O(n log n), Worst O(n^2), Space: O(log n), Unstable
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def quick_sort_inplace(arr: List[int], low: int = 0, high: int = None) -> List[int]:
    """Quick Sort (in-place)"""
    if high is None:
        arr = arr.copy()
        high = len(arr) - 1

    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort_inplace(arr, low, pivot_idx - 1)
        quick_sort_inplace(arr, pivot_idx + 1, high)

    return arr


def partition(arr: List[int], low: int, high: int) -> int:
    """Partition (Lomuto scheme)"""
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# =============================================================================
# 6. Heap Sort
# =============================================================================
def heap_sort(arr: List[int]) -> List[int]:
    """
    Heap Sort
    Time: O(n log n), Space: O(1), Unstable
    """
    arr = arr.copy()
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr


def heapify(arr: List[int], n: int, i: int):
    """Maintain max heap property"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


# =============================================================================
# 7. Counting Sort
# =============================================================================
def counting_sort(arr: List[int]) -> List[int]:
    """
    Counting Sort
    Time: O(n + k), Space: O(k), Stable
    k = max_value - min_value + 1
    Efficient when the integer range is small
    """
    if not arr:
        return []

    min_val, max_val = min(arr), max(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    # Count
    for num in arr:
        count[num - min_val] += 1

    # Accumulate
    for i in range(1, range_val):
        count[i] += count[i - 1]

    # Place in reverse order (to maintain stability)
    for i in range(len(arr) - 1, -1, -1):
        num = arr[i]
        count[num - min_val] -= 1
        output[count[num - min_val]] = num

    return output


# =============================================================================
# 8. Radix Sort
# =============================================================================
def radix_sort(arr: List[int]) -> List[int]:
    """
    Radix Sort (LSD)
    Time: O(d * (n + k)), Space: O(n + k)
    d = number of digits, k = radix (usually 10)
    Version without negative number support
    """
    if not arr or min(arr) < 0:
        return sorted(arr)  # Fall back to default sort for negative numbers

    max_val = max(arr)

    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr


def counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
    """Counting sort by a specific digit"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        count[digit] -= 1
        output[count[digit]] = arr[i]

    return output


# =============================================================================
# Performance Comparison
# =============================================================================
def benchmark_sorts(size: int = 1000):
    """Sorting algorithm performance comparison"""
    arr = [random.randint(0, 10000) for _ in range(size)]

    algorithms = [
        ("Bubble Sort", bubble_sort, size <= 1000),
        ("Selection Sort", selection_sort, size <= 1000),
        ("Insertion Sort", insertion_sort, size <= 1000),
        ("Merge Sort", merge_sort, True),
        ("Quick Sort", quick_sort, True),
        ("Heap Sort", heap_sort, True),
        ("Counting Sort", counting_sort, True),
        ("Radix Sort", radix_sort, True),
        ("Python sorted()", sorted, True),
    ]

    print(f"\nArray Size: {size}")
    print("-" * 50)

    for name, func, should_run in algorithms:
        if should_run:
            test_arr = arr.copy()
            start = time.perf_counter()
            result = func(test_arr)
            elapsed = time.perf_counter() - start
            print(f"{name:20s}: {elapsed * 1000:8.3f} ms")
        else:
            print(f"{name:20s}: (skipped due to size)")


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Sorting Algorithms")
    print("=" * 60)

    # Test array
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"\nOriginal array: {arr}")
    print("-" * 40)

    # Test each algorithm
    algorithms = [
        ("Bubble Sort", bubble_sort),
        ("Selection Sort", selection_sort),
        ("Insertion Sort", insertion_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort", quick_sort),
        ("Heap Sort", heap_sort),
        ("Counting Sort", counting_sort),
        ("Radix Sort", radix_sort),
    ]

    for name, func in algorithms:
        result = func(arr.copy())
        print(f"{name}: {result}")

    # Performance Comparison
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    for size in [100, 1000, 5000]:
        benchmark_sorts(size)

    # Sorting Algorithm Comparison Table
    print("\n" + "=" * 60)
    print("Sorting Algorithm Comparison")
    print("=" * 60)
    print("""
    | Algorithm       | Average  | Worst    | Space  | Stable | Notes                        |
    |----------------|----------|----------|--------|--------|------------------------------|
    | Bubble Sort    | O(n^2)   | O(n^2)   | O(1)   | Yes    | Simple, educational          |
    | Selection Sort | O(n^2)   | O(n^2)   | O(1)   | No     | Simple, fewer swaps          |
    | Insertion Sort | O(n^2)   | O(n^2)   | O(1)   | Yes    | Good for nearly sorted data  |
    | Merge Sort     | O(nlogn) | O(nlogn) | O(n)   | Yes    | Consistent performance       |
    | Quick Sort     | O(nlogn) | O(n^2)   | O(logn)| No     | Fastest on average           |
    | Heap Sort      | O(nlogn) | O(nlogn) | O(1)   | No     | In-place, consistent perf    |
    | Counting Sort  | O(n+k)   | O(n+k)   | O(k)   | Yes    | Integers, small range        |
    | Radix Sort     | O(d(n+k))| O(d(n+k))| O(n+k) | Yes    | Digit-based                  |

    Practical Selection Guide:
    - General case: Quick Sort or built-in language sort
    - Stability needed: Merge Sort
    - Memory constrained: Heap Sort
    - Nearly sorted: Insertion Sort
    - Integers + small range: Counting/Radix Sort
    """)


if __name__ == "__main__":
    main()

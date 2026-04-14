"""
12 Sorting Fundamentals
=======================
Demonstrates bubble, selection, insertion, merge,
and quick sort with correctness and timing.
"""

import random
import time


def bubble_sort(arr):
    a = arr[:]; n = len(a)
    for i in range(n):
        swapped = False
        for j in range(n-1-i):
            if a[j] > a[j+1]: a[j], a[j+1] = a[j+1], a[j]; swapped = True
        if not swapped: break
    return a


def selection_sort(arr):
    a = arr[:]; n = len(a)
    for i in range(n):
        mi = i
        for j in range(i+1, n):
            if a[j] < a[mi]: mi = j
        a[i], a[mi] = a[mi], a[i]
    return a


def insertion_sort(arr):
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]; j = i - 1
        while j >= 0 and a[j] > key: a[j+1] = a[j]; j -= 1
        a[j+1] = key
    return a


def merge_sort(arr):
    if len(arr) <= 1: return arr[:]
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    result = []; i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]: result.append(left[i]); i += 1
        else: result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result


def quick_sort(arr):
    if len(arr) <= 1: return arr[:]
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)


if __name__ == "__main__":
    test = [5, 3, 8, 1, 2, 7, 4, 6]
    print(f"Input: {test}")
    for name, fn in [("Bubble", bubble_sort), ("Selection", selection_sort),
                     ("Insertion", insertion_sort), ("Merge", merge_sort),
                     ("Quick", quick_sort)]:
        print(f"  {name:>10}: {fn(test)}")

    print(f"\nPerformance (n=2000):")
    data = random.sample(range(10000), 2000)
    for name, fn in [("Bubble", bubble_sort), ("Selection", selection_sort),
                     ("Insertion", insertion_sort), ("Merge", merge_sort),
                     ("Quick", quick_sort), ("Timsort", sorted)]:
        t = time.perf_counter()
        result = fn(data)
        elapsed = time.perf_counter() - t
        assert result == sorted(data)
        print(f"  {name:>10}: {elapsed:.4f}s")

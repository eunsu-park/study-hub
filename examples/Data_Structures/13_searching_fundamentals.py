"""
13 Searching Fundamentals
=========================
Demonstrates linear search, binary search variants,
bisect module, and binary search on answer.
"""

import bisect


def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target: return i
    return -1


def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: left = mid + 1
        else: right = mid - 1
    return -1


def lower_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target: left = mid + 1
        else: right = mid
    return left


def upper_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target: left = mid + 1
        else: right = mid
    return left


def sqrt_integer(n):
    if n < 2: return n
    lo, hi = 1, n // 2
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        sq = mid * mid
        if sq == n: return mid
        elif sq < n: lo = mid + 1
        else: hi = mid - 1
    return hi


if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Array: {arr}")
    print(f"linear_search(7):  {linear_search(arr, 7)}")
    print(f"binary_search(7):  {binary_search(arr, 7)}")
    print(f"binary_search(8):  {binary_search(arr, 8)}")

    arr2 = [1, 2, 4, 4, 4, 6, 8]
    print(f"\nArray: {arr2}")
    print(f"lower_bound(4): {lower_bound(arr2, 4)}")
    print(f"upper_bound(4): {upper_bound(arr2, 4)}")
    print(f"count(4): {upper_bound(arr2, 4) - lower_bound(arr2, 4)}")

    print(f"\nbisect_left(arr, 7):  {bisect.bisect_left(arr, 7)}")
    print(f"bisect_right(arr, 7): {bisect.bisect_right(arr, 7)}")

    print("\nInteger square roots:")
    for n in [0, 1, 4, 8, 16, 25, 100]:
        print(f"  sqrt({n}) = {sqrt_integer(n)}")

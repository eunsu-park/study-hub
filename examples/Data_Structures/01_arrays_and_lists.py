"""
01 Arrays and Lists
===================
Demonstrates dynamic arrays, list operations, slicing,
comprehensions, and common array patterns.
"""

import array
import sys


def array_module_demo():
    """Typed arrays using the array module."""
    nums = array.array('i', [10, 20, 30, 40, 50])
    print(f"array('i', ...): {nums}")
    print(f"Element at index 2: {nums[2]}")
    print(f"Memory size: {nums.buffer_info()[1] * nums.itemsize} bytes")

    # Compare with list memory
    lst = [10, 20, 30, 40, 50]
    print(f"list sys.getsizeof: {sys.getsizeof(lst)} bytes")
    print(f"array sys.getsizeof: {sys.getsizeof(nums)} bytes")


def dynamic_array_growth():
    """Show how Python list capacity grows."""
    lst = []
    prev_size = 0
    print(f"{'Length':>6} {'Size (bytes)':>12} {'Growth':>8}")
    for i in range(25):
        lst.append(i)
        curr_size = sys.getsizeof(lst)
        marker = " <-- resize" if curr_size != prev_size else ""
        print(f"{len(lst):>6} {curr_size:>12}{marker}")
        prev_size = curr_size


def indexing_and_slicing():
    """Demonstrate list indexing and slicing."""
    fruits = ["apple", "banana", "cherry", "date", "elderberry"]
    print(f"List: {fruits}")
    print(f"fruits[0] = {fruits[0]}")
    print(f"fruits[-1] = {fruits[-1]}")
    print(f"fruits[1:3] = {fruits[1:3]}")
    print(f"fruits[::2] = {fruits[::2]}")
    print(f"fruits[::-1] = {fruits[::-1]}")

    # Slice assignment
    copy = fruits[:]
    copy[1:3] = ["blueberry"]
    print(f"After slice assign: {copy}")


def list_comprehensions():
    """List comprehension patterns."""
    squares = [x ** 2 for x in range(10)]
    print(f"Squares: {squares}")

    evens = [x for x in range(20) if x % 2 == 0]
    print(f"Evens: {evens}")

    matrix = [[i * 3 + j for j in range(3)] for i in range(3)]
    print(f"Matrix: {matrix}")

    flat = [x for row in matrix for x in row]
    print(f"Flat: {flat}")


def two_pointer_demo():
    """Two-pointer technique on sorted array."""
    def two_sum_sorted(nums, target):
        left, right = 0, len(nums) - 1
        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                return (left, right)
            elif s < target:
                left += 1
            else:
                right -= 1
        return None

    nums = [1, 2, 4, 7, 11, 15]
    target = 9
    result = two_sum_sorted(nums, target)
    print(f"two_sum_sorted({nums}, {target}) = {result}")
    if result:
        i, j = result
        print(f"  nums[{i}] + nums[{j}] = {nums[i]} + {nums[j]} = {target}")


def sliding_window_demo():
    """Sliding window maximum sum."""
    def max_sum_subarray(nums, k):
        if len(nums) < k:
            return None
        window_sum = sum(nums[:k])
        max_sum = window_sum
        for i in range(k, len(nums)):
            window_sum += nums[i] - nums[i - k]
            max_sum = max(max_sum, window_sum)
        return max_sum

    nums = [2, 1, 5, 1, 3, 2]
    k = 3
    result = max_sum_subarray(nums, k)
    print(f"max_sum_subarray({nums}, k={k}) = {result}")


def matrix_operations():
    """2D list (matrix) operations."""
    rows, cols = 3, 4
    matrix = [[0] * cols for _ in range(rows)]
    matrix[1][2] = 42
    matrix[0][0] = 1

    print("Matrix:")
    for row in matrix:
        print(f"  {row}")

    # Transpose
    transposed = [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    print("Transposed:")
    for row in transposed:
        print(f"  {row}")


if __name__ == "__main__":
    sections = [
        ("Array Module", array_module_demo),
        ("Dynamic Array Growth", dynamic_array_growth),
        ("Indexing and Slicing", indexing_and_slicing),
        ("List Comprehensions", list_comprehensions),
        ("Two-Pointer Technique", two_pointer_demo),
        ("Sliding Window", sliding_window_demo),
        ("Matrix Operations", matrix_operations),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

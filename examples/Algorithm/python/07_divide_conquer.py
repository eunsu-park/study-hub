"""
Divide and Conquer
Divide and Conquer Algorithms

Algorithms that solve problems by breaking them into smaller subproblems.
"""

from typing import List, Tuple, Optional
import random


# =============================================================================
# 1. Merge Sort
# =============================================================================

def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge Sort
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable Sort
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
# 2. Quick Sort
# =============================================================================

def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick Sort (Lomuto Partition)
    Time Complexity: Average O(n log n), Worst O(n^2)
    Space Complexity: O(log n) - recursion stack
    Unstable Sort
    """
    if len(arr) <= 1:
        return arr

    arr = arr.copy()
    _quick_sort(arr, 0, len(arr) - 1)
    return arr


def _quick_sort(arr: List[int], low: int, high: int) -> None:
    if low < high:
        pivot_idx = partition(arr, low, high)
        _quick_sort(arr, low, pivot_idx - 1)
        _quick_sort(arr, pivot_idx + 1, high)


def partition(arr: List[int], low: int, high: int) -> int:
    """Lomuto Partition"""
    # Random pivot to prevent worst case
    pivot_idx = random.randint(low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]

    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# =============================================================================
# 3. Fast Exponentiation
# =============================================================================

def power(base: int, exp: int, mod: int = None) -> int:
    """
    Fast Exponentiation
    Time Complexity: O(log n)
    """
    if exp == 0:
        return 1

    if exp % 2 == 0:
        half = power(base, exp // 2, mod)
        result = half * half
    else:
        result = base * power(base, exp - 1, mod)

    return result % mod if mod else result


def power_iterative(base: int, exp: int, mod: int = None) -> int:
    """Fast Exponentiation (Iterative)"""
    result = 1

    while exp > 0:
        if exp % 2 == 1:
            result = result * base
            if mod:
                result %= mod
        base = base * base
        if mod:
            base %= mod
        exp //= 2

    return result


# =============================================================================
# 4. Matrix Exponentiation
# =============================================================================

def matrix_multiply(A: List[List[int]], B: List[List[int]], mod: int = None) -> List[List[int]]:
    """2x2 Matrix Multiplication"""
    n = len(A)
    C = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                if mod:
                    C[i][j] %= mod

    return C


def matrix_power(M: List[List[int]], exp: int, mod: int = None) -> List[List[int]]:
    """
    Matrix Exponentiation
    Time Complexity: O(k^3 log n), k = matrix size
    """
    n = len(M)
    # Identity matrix
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    while exp > 0:
        if exp % 2 == 1:
            result = matrix_multiply(result, M, mod)
        M = matrix_multiply(M, M, mod)
        exp //= 2

    return result


def fibonacci_matrix(n: int, mod: int = None) -> int:
    """
    Fibonacci Sequence (Matrix Exponentiation)
    Time Complexity: O(log n)
    """
    if n <= 1:
        return n

    # [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n
    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n, mod)
    return result[0][1]


# =============================================================================
# 5. Inversion Count
# =============================================================================

def count_inversions(arr: List[int]) -> int:
    """
    Count inversion pairs (i < j but arr[i] > arr[j])
    Merge sort variant
    Time Complexity: O(n log n)
    """

    def merge_count(arr: List[int]) -> Tuple[List[int], int]:
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, left_inv = merge_count(arr[:mid])
        right, right_inv = merge_count(arr[mid:])

        merged = []
        inversions = left_inv + right_inv
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inversions += len(left) - i  # Remaining left elements count as inversions
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])

        return merged, inversions

    _, count = merge_count(arr)
    return count


# =============================================================================
# 6. Closest Pair of Points
# =============================================================================

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def closest_pair(points: List[Tuple[float, float]]) -> float:
    """
    Distance between the closest pair of points
    Time Complexity: O(n log n)
    """

    def closest_recursive(px: List, py: List) -> float:
        n = len(px)

        # Base case: brute force
        if n <= 3:
            min_dist = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_dist = min(min_dist, distance(px[i], px[j]))
            return min_dist

        mid = n // 2
        mid_point = px[mid]

        # Split by x-coordinate
        pyl = [p for p in py if p[0] <= mid_point[0]]
        pyr = [p for p in py if p[0] > mid_point[0]]

        dl = closest_recursive(px[:mid], pyl)
        dr = closest_recursive(px[mid:], pyr)

        d = min(dl, dr)

        # Check the middle strip
        strip = [p for p in py if abs(p[0] - mid_point[0]) < d]

        # Compare points within the strip (check at most 7)
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):
                if strip[j][1] - strip[i][1] >= d:
                    break
                d = min(d, distance(strip[i], strip[j]))

        return d

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])

    return closest_recursive(px, py)


# =============================================================================
# 7. Maximum Subarray Sum (D&C)
# =============================================================================

def max_subarray_dc(arr: List[int]) -> int:
    """
    Maximum Subarray Sum (Divide and Conquer)
    Time Complexity: O(n log n)
    """

    def max_crossing_sum(arr: List[int], low: int, mid: int, high: int) -> int:
        # Left maximum
        left_sum = float('-inf')
        total = 0
        for i in range(mid, low - 1, -1):
            total += arr[i]
            left_sum = max(left_sum, total)

        # Right maximum
        right_sum = float('-inf')
        total = 0
        for i in range(mid + 1, high + 1):
            total += arr[i]
            right_sum = max(right_sum, total)

        return left_sum + right_sum

    def max_subarray(arr: List[int], low: int, high: int) -> int:
        if low == high:
            return arr[low]

        mid = (low + high) // 2

        left_max = max_subarray(arr, low, mid)
        right_max = max_subarray(arr, mid + 1, high)
        cross_max = max_crossing_sum(arr, low, mid, high)

        return max(left_max, right_max, cross_max)

    if not arr:
        return 0
    return max_subarray(arr, 0, len(arr) - 1)


# =============================================================================
# 8. Karatsuba Multiplication
# =============================================================================

def karatsuba(x: int, y: int) -> int:
    """
    Karatsuba Large Number Multiplication
    Time Complexity: O(n^1.585)
    """
    # Base case
    if x < 10 or y < 10:
        return x * y

    # Calculate number of digits
    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # x = a * 10^m + b, y = c * 10^m + d
    divisor = 10 ** m

    a, b = divmod(x, divisor)
    c, d = divmod(y, divisor)

    # Three multiplications
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    ad_bc = karatsuba(a + b, c + d) - ac - bd

    return ac * (10 ** (2 * m)) + ad_bc * (10 ** m) + bd


# =============================================================================
# 9. Strassen's Matrix Multiplication
# =============================================================================

def strassen(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """
    Strassen's Matrix Multiplication
    Time Complexity: O(n^2.807)
    (For small matrices, overhead is large, so naive multiplication is used below a threshold)
    """
    n = len(A)

    # Base case
    if n <= 64:  # Threshold
        return naive_matrix_multiply(A, B)

    # Split matrices
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # 7 multiplications (Strassen's formulas)
    M1 = strassen(matrix_add(A11, A22), matrix_add(B11, B22))
    M2 = strassen(matrix_add(A21, A22), B11)
    M3 = strassen(A11, matrix_sub(B12, B22))
    M4 = strassen(A22, matrix_sub(B21, B11))
    M5 = strassen(matrix_add(A11, A12), B22)
    M6 = strassen(matrix_sub(A21, A11), matrix_add(B11, B12))
    M7 = strassen(matrix_sub(A12, A22), matrix_add(B21, B22))

    # Combine results
    C11 = matrix_add(matrix_sub(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_add(matrix_sub(matrix_add(M1, M3), M2), M6)

    return combine_matrices(C11, C12, C21, C22)


def naive_matrix_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Naive matrix multiplication"""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matrix_add(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Matrix addition"""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def matrix_sub(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Matrix subtraction"""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def combine_matrices(C11, C12, C21, C22) -> List[List[int]]:
    """Combine 4 submatrices"""
    n = len(C11)
    result = [[0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = C11[i][j]
            result[i][j + n] = C12[i][j]
            result[i + n][j] = C21[i][j]
            result[i + n][j + n] = C22[i][j]
    return result


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Divide and Conquer Examples")
    print("=" * 60)

    # 1. Merge Sort
    print("\n[1] Merge Sort")
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = merge_sort(arr)
    print(f"    Original: {arr}")
    print(f"    Sorted: {sorted_arr}")

    # 2. Quick Sort
    print("\n[2] Quick Sort")
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = quick_sort(arr)
    print(f"    Original: {arr}")
    print(f"    Sorted: {sorted_arr}")

    # 3. Fast Exponentiation
    print("\n[3] Fast Exponentiation")
    print(f"    2^10 = {power(2, 10)}")
    print(f"    2^10 (iterative) = {power_iterative(2, 10)}")
    print(f"    3^7 mod 1000 = {power(3, 7, 1000)}")

    # 4. Fibonacci (Matrix Exponentiation)
    print("\n[4] Fibonacci (Matrix Exponentiation)")
    for n in [10, 20, 50]:
        fib = fibonacci_matrix(n)
        print(f"    F({n}) = {fib}")

    # 5. Inversion Count
    print("\n[5] Inversion Count")
    arr = [2, 4, 1, 3, 5]
    inv = count_inversions(arr)
    print(f"    Array: {arr}")
    print(f"    Inversion pairs: {inv}")

    # 6. Closest Pair of Points
    print("\n[6] Closest Pair of Points")
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    dist = closest_pair(points)
    print(f"    Points: {points}")
    print(f"    Minimum distance: {dist:.4f}")

    # 7. Maximum Subarray Sum (D&C)
    print("\n[7] Maximum Subarray Sum (Divide and Conquer)")
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = max_subarray_dc(arr)
    print(f"    Array: {arr}")
    print(f"    Maximum sum: {max_sum}")

    # 8. Karatsuba Multiplication
    print("\n[8] Karatsuba Multiplication")
    x, y = 1234, 5678
    result = karatsuba(x, y)
    print(f"    {x} x {y} = {result}")
    print(f"    Verification: {x * y}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

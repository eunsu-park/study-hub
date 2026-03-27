"""
Exercises for Lesson 01: NumPy Basics
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Array Creation ===
# Problem: Create an array containing only multiples of 3 from 1 to 100.
def exercise_1():
    """Solution using np.arange with step parameter."""
    # Approach 1: Direct creation with step=3, starting at 3
    arr1 = np.arange(3, 101, 3)
    print("Approach 1 (arange with step):")
    print(f"  Array: {arr1}")
    print(f"  Length: {len(arr1)}")

    # Approach 2: Create full range then filter with boolean indexing
    # This demonstrates NumPy's boolean masking capability
    arr_full = np.arange(1, 101)
    arr2 = arr_full[arr_full % 3 == 0]
    print("\nApproach 2 (boolean indexing):")
    print(f"  Array: {arr2}")
    print(f"  Length: {len(arr2)}")

    # Verify both approaches produce identical results
    assert np.array_equal(arr1, arr2), "Both approaches should yield the same result"
    print("\nBoth approaches produce identical arrays.")


# === Exercise 2: Matrix Operations ===
# Problem: Find the sum of diagonal elements of a 3x3 identity matrix.
def exercise_2():
    """Solution using np.trace and np.diag."""
    eye = np.eye(3)
    print("3x3 Identity matrix:")
    print(eye)

    # Approach 1: np.trace computes the sum of diagonal elements directly
    # Trace is a fundamental linear algebra operation: tr(I_n) = n
    diagonal_sum_trace = np.trace(eye)
    print(f"\nDiagonal sum (np.trace): {diagonal_sum_trace}")

    # Approach 2: Extract diagonal first, then sum
    # np.diag extracts the main diagonal as a 1D array
    diagonal = np.diag(eye)
    diagonal_sum_diag = np.sum(diagonal)
    print(f"Diagonal elements: {diagonal}")
    print(f"Diagonal sum (np.diag + np.sum): {diagonal_sum_diag}")

    # For an n x n identity matrix, the trace is always n
    for n in [2, 4, 5]:
        assert np.trace(np.eye(n)) == n, f"Trace of I_{n} should be {n}"
    print("\nVerified: tr(I_n) = n for n = 2, 3, 4, 5")


# === Exercise 3: Broadcasting ===
# Problem: Normalize a 4x4 matrix by dividing each element by the maximum value
#          of its column.
def exercise_3():
    """Solution demonstrating NumPy broadcasting for column-wise normalization."""
    arr = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9,  10, 11, 12],
                    [13, 14, 15, 16]])
    print("Original matrix:")
    print(arr)

    # axis=0 computes max along rows => one max per column
    # Result shape: (4,), which broadcasts against (4, 4) along axis=1
    col_max = np.max(arr, axis=0)
    print(f"\nColumn maxima: {col_max}")

    # Broadcasting: (4,4) / (4,) => each row is divided element-wise by col_max
    # This scales every column so its maximum becomes 1.0
    normalized = arr / col_max
    print("\nNormalized matrix (each column's max = 1.0):")
    print(normalized)

    # Verify: the last row should be all 1.0 since it contains the column maxima
    assert np.allclose(normalized[-1], 1.0), "Last row should be all 1.0"
    # Verify: all values should be in [0, 1]
    assert np.all(normalized >= 0) and np.all(normalized <= 1), \
        "All normalized values should be in [0, 1]"
    print("\nVerified: last row is all 1.0, all values in [0, 1].")


if __name__ == "__main__":
    print("=== Exercise 1: Array Creation ===")
    exercise_1()
    print("\n=== Exercise 2: Matrix Operations ===")
    exercise_2()
    print("\n=== Exercise 3: Broadcasting ===")
    exercise_3()
    print("\nAll exercises completed!")

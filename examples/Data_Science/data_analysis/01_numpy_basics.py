"""
NumPy Basics — Fundamental Array Operations

Demonstrates:
- Array creation, indexing, slicing, and reshaping
- Element-wise operations and broadcasting
- Aggregation functions with axis semantics
- Linear algebra routines (solve, eig, SVD)

Theory:
- NumPy's ndarray stores homogeneous data in contiguous memory, enabling
  vectorized operations that bypass Python's per-element interpreter
  overhead and run in compiled C/Fortran instead.
- Broadcasting lets arrays of different shapes combine without copies
  by virtually stretching dimensions of size 1.
- np.linalg wraps LAPACK routines — the same battle-tested linear
  algebra library used by MATLAB and R.

Adapted from Data_Science Lesson 01.
"""

import numpy as np


# =============================================================================
# 1. Array Creation
# =============================================================================
def array_creation():
    """Various array creation methods."""
    print("\n[1] Array Creation")
    print("=" * 50)

    # From Python list
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"From list: {arr1}")

    # 2D array
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"2D array:\n{arr2d}")

    # Why: Specialized constructors (zeros, ones, eye) allocate contiguous
    # memory in one call, avoiding the overhead of building a Python list first.
    zeros = np.zeros((3, 4))  # filled with 0
    ones = np.ones((2, 3))    # filled with 1
    empty = np.empty((2, 2))  # uninitialized (faster, but contains garbage values)
    full = np.full((2, 3), 7) # filled with specific value
    eye = np.eye(3)           # identity matrix

    print(f"\nnp.zeros((3,4)):\n{zeros}")
    print(f"\nnp.eye(3):\n{eye}")

    # Why: linspace is preferred over arange for floating-point ranges because
    # arange can produce inconsistent element counts due to floating-point
    # rounding, while linspace guarantees the exact number of points.
    arange = np.arange(0, 10, 2)  # start, stop, step
    linspace = np.linspace(0, 1, 5)  # start, stop, num
    logspace = np.logspace(0, 3, 4)  # 10^0 to 10^3

    print(f"\nnp.arange(0, 10, 2): {arange}")
    print(f"np.linspace(0, 1, 5): {linspace}")
    print(f"np.logspace(0, 3, 4): {logspace}")

    # Random arrays
    rand = np.random.rand(3, 3)        # uniform [0, 1)
    randn = np.random.randn(3, 3)      # standard normal distribution
    randint = np.random.randint(0, 10, (3, 3))  # integers

    print(f"\nnp.random.rand(3,3):\n{rand}")


# =============================================================================
# 2. Array Attributes and Reshaping
# =============================================================================
def array_attributes():
    """Array attributes and shape manipulation."""
    print("\n[2] Array Attributes and Reshaping")
    print("=" * 50)

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    print(f"Array:\n{arr}")
    print(f"shape: {arr.shape}")      # (2, 3)
    print(f"ndim: {arr.ndim}")        # number of dimensions
    print(f"size: {arr.size}")        # total element count
    print(f"dtype: {arr.dtype}")      # data type

    # Shape transformation
    reshaped = arr.reshape(3, 2)
    print(f"\nreshape(3, 2):\n{reshaped}")

    # Why: flatten() always returns a copy; ravel() returns a view when
    # possible (same underlying memory). Use ravel() for read-only access
    # to avoid unnecessary memory allocation.
    flattened = arr.flatten()
    print(f"flatten(): {flattened}")

    raveled = arr.ravel()
    print(f"ravel(): {raveled}")

    # Transpose
    transposed = arr.T
    print(f"\nTranspose (T):\n{transposed}")

    # Add/remove dimensions
    arr1d = np.array([1, 2, 3])
    expanded = np.expand_dims(arr1d, axis=0)  # (3,) -> (1, 3)
    print(f"\nexpand_dims: {arr1d.shape} -> {expanded.shape}")

    squeezed = np.squeeze(expanded)
    print(f"squeeze: {expanded.shape} -> {squeezed.shape}")


# =============================================================================
# 3. Indexing and Slicing
# =============================================================================
def indexing_slicing():
    """Indexing and slicing."""
    print("\n[3] Indexing and Slicing")
    print("=" * 50)

    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

    print(f"Original array:\n{arr}")

    # Basic indexing
    print(f"\narr[0, 0] = {arr[0, 0]}")  # first element
    print(f"arr[1, 2] = {arr[1, 2]}")    # 7
    print(f"arr[-1, -1] = {arr[-1, -1]}")  # last element

    # Slicing (returns a view, not a copy)
    print(f"\narr[0, :] = {arr[0, :]}")    # first row
    print(f"arr[:, 0] = {arr[:, 0]}")      # first column
    print(f"arr[0:2, 1:3] =\n{arr[0:2, 1:3]}")  # sub-array

    # Fancy indexing (returns a copy, unlike basic slicing)
    indices = [0, 2]
    print(f"\narr[indices] =\n{arr[indices]}")  # rows 0 and 2

    # Why: Boolean indexing is the idiomatic NumPy way to filter data.
    # It avoids Python loops and works on any shape, making it central
    # to data cleaning (e.g., removing outliers, selecting valid rows).
    mask = arr > 5
    print(f"\nmask (arr > 5):\n{mask}")
    print(f"arr[arr > 5] = {arr[arr > 5]}")


# =============================================================================
# 4. Array Operations
# =============================================================================
def array_operations():
    """Element-wise operations and broadcasting."""
    print("\n[4] Array Operations")
    print("=" * 50)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    print(f"a =\n{a}")
    print(f"b =\n{b}")

    # Element-wise operations
    print(f"\na + b =\n{a + b}")
    print(f"a * b =\n{a * b}")    # element-wise (Hadamard) product
    print(f"a / b =\n{a / b}")
    print(f"a ** 2 =\n{a ** 2}")

    # Why: The @ operator (PEP 465) is preferred over np.dot() for matrix
    # multiplication because it is unambiguous — np.dot() behaves differently
    # for 1D vs 2D arrays, while @ always means matrix multiplication.
    print(f"\na @ b (matrix multiply) =\n{a @ b}")
    print(f"np.dot(a, b) =\n{np.dot(a, b)}")

    # Why: Broadcasting avoids explicit loops by letting NumPy align shapes
    # automatically. This is 10-100x faster than Python loops for large
    # arrays because operations run in compiled C on contiguous memory.
    print("\n[Broadcasting]")
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    scalar = 10
    row_vec = np.array([1, 0, 1])       # (3,) broadcasts to (2,3)
    col_vec = np.array([[10], [20]])     # (2,1) broadcasts to (2,3)

    print(f"arr + scalar =\n{arr + scalar}")
    print(f"arr * row_vec =\n{arr * row_vec}")
    print(f"arr + col_vec =\n{arr + col_vec}")


# =============================================================================
# 5. Math Functions
# =============================================================================
def math_functions():
    """Universal functions (ufuncs)."""
    print("\n[5] Math Functions")
    print("=" * 50)

    arr = np.array([1, 4, 9, 16, 25])
    print(f"arr = {arr}")

    # Why: NumPy ufuncs (sqrt, exp, log, etc.) operate element-wise in C,
    # so they are vastly faster than calling math.sqrt() in a Python loop.
    print(f"\nnp.sqrt(arr) = {np.sqrt(arr)}")
    print(f"np.exp(arr[:3]) = {np.exp(arr[:3])}")
    print(f"np.log(arr) = {np.log(arr)}")

    # Trigonometric functions
    angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    print(f"\nangles = {np.degrees(angles)}")
    print(f"np.sin(angles) = {np.sin(angles)}")

    # Rounding
    float_arr = np.array([1.2, 2.5, 3.7, -1.2])
    print(f"\nfloat_arr = {float_arr}")
    print(f"np.round(float_arr) = {np.round(float_arr)}")
    print(f"np.floor(float_arr) = {np.floor(float_arr)}")
    print(f"np.ceil(float_arr) = {np.ceil(float_arr)}")


# =============================================================================
# 6. Aggregation Functions
# =============================================================================
def aggregation_functions():
    """Aggregation functions with axis semantics."""
    print("\n[6] Aggregation Functions")
    print("=" * 50)

    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    print(f"Array:\n{arr}")

    # Global aggregation
    print(f"\nnp.sum(arr) = {np.sum(arr)}")
    print(f"np.mean(arr) = {np.mean(arr)}")
    print(f"np.std(arr) = {np.std(arr):.4f}")
    print(f"np.min(arr) = {np.min(arr)}")
    print(f"np.max(arr) = {np.max(arr)}")

    # Why: axis=0 collapses rows (aggregates "down"), axis=1 collapses
    # columns (aggregates "across"). This is the most common source of
    # confusion — think of the axis as the dimension that disappears.
    print(f"\nnp.sum(arr, axis=0) = {np.sum(arr, axis=0)}")  # column sums
    print(f"np.sum(arr, axis=1) = {np.sum(arr, axis=1)}")    # row sums
    print(f"np.mean(arr, axis=0) = {np.mean(arr, axis=0)}")

    # Cumulative
    print(f"\nnp.cumsum(arr.flatten()) = {np.cumsum(arr.flatten())}")

    # Argmax/argmin return index into the flattened array
    print(f"\nnp.argmax(arr) = {np.argmax(arr)}")  # flattened index
    print(f"np.argmin(arr) = {np.argmin(arr)}")


# =============================================================================
# 7. Array Manipulation
# =============================================================================
def array_manipulation():
    """Array stacking, splitting, and sorting."""
    print("\n[7] Array Manipulation")
    print("=" * 50)

    # Concatenation
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    print(f"a =\n{a}")
    print(f"b =\n{b}")

    vstack = np.vstack([a, b])  # vertical stack
    hstack = np.hstack([a, b])  # horizontal stack
    concat = np.concatenate([a, b], axis=0)

    print(f"\nvstack:\n{vstack}")
    print(f"\nhstack:\n{hstack}")

    # Splitting
    arr = np.arange(16).reshape(4, 4)
    print(f"\nArray to split:\n{arr}")

    vsplit = np.vsplit(arr, 2)
    hsplit = np.hsplit(arr, 2)

    print(f"\nvsplit(2):\n{vsplit[0]}\n{vsplit[1]}")

    # Sorting
    unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"\nBefore sort: {unsorted}")
    print(f"np.sort(): {np.sort(unsorted)}")
    # Why: argsort returns indices that would sort the array. This is
    # essential when you need to reorder a parallel array by the same
    # permutation (e.g., sort labels by their corresponding scores).
    print(f"np.argsort(): {np.argsort(unsorted)}")


# =============================================================================
# 8. Linear Algebra
# =============================================================================
def linear_algebra():
    """Linear algebra operations (LAPACK wrappers)."""
    print("\n[8] Linear Algebra")
    print("=" * 50)

    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])

    print(f"A =\n{A}")
    print(f"b = {b}")

    # Determinant
    det = np.linalg.det(A)
    print(f"\ndet(A) = {det:.4f}")

    # Inverse
    A_inv = np.linalg.inv(A)
    print(f"\nA^(-1) =\n{A_inv}")

    # Why: linalg.solve(A, b) is preferred over inv(A) @ b because it
    # uses LU decomposition directly, which is both faster and more
    # numerically stable (avoids explicitly forming the inverse).
    x = np.linalg.solve(A, b)
    print(f"\nSolution of Ax = b: x = {x}")
    print(f"Verification A @ x = {A @ x}")

    # Eigenvalues / eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    print(f"\nSVD:")
    print(f"U =\n{U}")
    print(f"S = {S}")
    print(f"Vt =\n{Vt}")


# =============================================================================
# 9. Practical Examples
# =============================================================================
def practical_examples():
    """Practical examples combining multiple NumPy features."""
    print("\n[9] Practical Examples")
    print("=" * 50)

    # Example 1: Euclidean distance
    print("\nExample 1: Euclidean Distance")
    point1 = np.array([1, 2, 3])
    point2 = np.array([4, 5, 6])
    distance = np.linalg.norm(point1 - point2)
    print(f"Point 1: {point1}, Point 2: {point2}")
    print(f"Distance: {distance:.4f}")

    # Example 2: Moving average
    # Why: np.convolve with a uniform kernel computes a moving average in one
    # vectorized call — much cleaner than a manual sliding-window loop.
    print("\nExample 2: Moving Average")
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window = 3
    moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
    print(f"Data: {data}")
    print(f"Moving average (window={window}): {moving_avg}")

    # Example 3: Normalization
    print("\nExample 3: Normalization")
    data = np.array([10, 20, 30, 40, 50])
    normalized = (data - data.mean()) / data.std()
    min_max = (data - data.min()) / (data.max() - data.min())
    print(f"Original: {data}")
    print(f"Z-score: {normalized}")
    print(f"Min-Max: {min_max}")

    # Example 4: Solving a system of linear equations
    print("\nExample 4: Solving Linear Equations")
    # 2x + 3y = 8
    # 3x + 4y = 11
    A = np.array([[2, 3], [3, 4]])
    b = np.array([8, 11])
    solution = np.linalg.solve(A, b)
    print(f"2x + 3y = 8")
    print(f"3x + 4y = 11")
    print(f"Solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("NumPy Basics Examples")
    print("=" * 60)

    array_creation()
    array_attributes()
    indexing_slicing()
    array_operations()
    math_functions()
    aggregation_functions()
    array_manipulation()
    linear_algebra()
    practical_examples()

    print("\n" + "=" * 60)
    print("NumPy Summary")
    print("=" * 60)
    print("""
    Core Concepts:
    - ndarray: N-dimensional array (contiguous memory, homogeneous dtype)
    - Broadcasting: operations between arrays of different shapes
    - Vectorization: compiled C loops instead of Python loops

    Commonly Used Functions:
    - Creation: array, zeros, ones, arange, linspace
    - Shape:    reshape, flatten, T
    - Aggregation: sum, mean, std, min, max
    - Operations: +, -, *, /, @, dot
    - Linear algebra: linalg.inv, linalg.solve, linalg.eig

    Tips:
    - Use vectorized operations instead of loops (10-100x faster)
    - Understand axis parameter (0 = collapse rows, 1 = collapse columns)
    - Know copy vs view (copy() vs slicing)
    """)


if __name__ == "__main__":
    main()

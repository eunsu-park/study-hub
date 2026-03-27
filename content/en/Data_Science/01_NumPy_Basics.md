# 1. NumPy Basics

[Next: NumPy Advanced](./02_NumPy_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create NumPy arrays from lists, special constructors, and sequential generators
2. Describe array attributes including shape, dtype, ndim, and memory layout
3. Apply basic indexing, slicing, boolean indexing, and fancy indexing to select array elements
4. Implement array reshaping, flattening, and transposition operations
5. Apply element-wise arithmetic, universal functions (ufuncs), and aggregation functions
6. Explain broadcasting rules and apply them to operations between arrays of different shapes
7. Combine and split arrays using stacking, concatenation, and splitting functions
8. Distinguish between views and copies and identify when each is created

---

NumPy (Numerical Python) is the foundation upon which nearly every scientific and data-oriented Python library is built. Whether you are preprocessing data for machine learning, running statistical analyses, or performing large-scale simulations, efficient numerical computation starts here. Mastering NumPy arrays and their operations will dramatically accelerate your data workflows compared to plain Python lists.

---

## 1. Creating NumPy Arrays

### 1.1 Basic Array Creation

```python
import numpy as np

# Creating array from list
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]
print(type(arr1))  # <class 'numpy.ndarray'>

# 2D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# 3D array
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr3.shape)  # (2, 2, 2)
```

### 1.2 Creating Special Arrays

```python
# Array filled with zeros
# Use zeros (not np.empty) when you need a known initial state — np.empty allocates
# memory without initializing it, so values are whatever was previously in that memory,
# making bugs very hard to reproduce and diagnose.
zeros = np.zeros((3, 4))
print(zeros)

# Array filled with ones
ones = np.ones((2, 3))
print(ones)

# Array filled with specific value
full = np.full((2, 2), 7)
print(full)  # [[7 7], [7 7]]

# Identity matrix
eye = np.eye(3)
print(eye)

# Empty array (uninitialized values)
# Only use np.empty when you will immediately overwrite every element and raw
# performance is critical — it skips initialization and is therefore faster, but unsafe
# if any element is ever read before being explicitly written.
empty = np.empty((2, 3))
```

### 1.3 Creating Sequential Arrays

```python
# arange: range specification
# Use arange when the step size is what matters (e.g., every 2 units).
# Caution: with floating-point steps, rounding can cause the endpoint count to vary.
arr = np.arange(0, 10, 2)  # From 0 to less than 10, step 2
print(arr)  # [0 2 4 6 8]

# linspace: evenly spaced division
# Use linspace when the number of points is what matters (e.g., you need exactly 100
# samples between two bounds). linspace always includes the endpoint and guarantees
# the exact count — it computes step size from the count, not the other way around.
arr = np.linspace(0, 1, 5)  # 0 to 1 divided into 5 equal parts
print(arr)  # [0.   0.25 0.5  0.75 1.  ]

# logspace: logarithmic scale spacing
arr = np.logspace(0, 2, 5)  # 10^0 to 10^2
print(arr)  # [  1.    3.16  10.   31.62 100. ]
```

---

## 2. Array Attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Number of dimensions
print(arr.ndim)  # 2

# Shape
print(arr.shape)  # (2, 3)

# Total number of elements
print(arr.size)  # 6

# Data type
print(arr.dtype)  # int64

# Bytes per element
print(arr.itemsize)  # 8

# Total bytes
print(arr.nbytes)  # 48
```

### Data Type Specification

```python
# Integer
arr_int = np.array([1, 2, 3], dtype=np.int32)

# Float
arr_float = np.array([1, 2, 3], dtype=np.float64)

# Complex
arr_complex = np.array([1, 2, 3], dtype=np.complex128)

# Boolean
arr_bool = np.array([0, 1, 0, 1], dtype=np.bool_)

# Type conversion
arr = np.array([1.5, 2.7, 3.9])
arr_int = arr.astype(np.int32)  # [1, 2, 3]
```

---

## 3. Indexing and Slicing

### 3.1 Basic Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single element access
print(arr[0])   # 10
print(arr[-1])  # 50

# 2D array
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr2d[0, 0])  # 1
print(arr2d[1, 2])  # 6
print(arr2d[-1, -1])  # 9
```

### 3.2 Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Basic slicing [start:stop:step]
print(arr[2:7])     # [2 3 4 5 6]
print(arr[::2])     # [0 2 4 6 8]
print(arr[::-1])    # [9 8 7 6 5 4 3 2 1 0]

# 2D slicing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr2d[0:2, 1:3])
# [[2 3]
#  [5 6]]

print(arr2d[:, 0])  # First column: [1 4 7]
print(arr2d[1, :])  # Second row: [4 5 6]
```

### 3.3 Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Select elements satisfying condition
mask = arr > 5
print(mask)  # [False False False False False  True  True  True  True  True]
print(arr[mask])  # [ 6  7  8  9 10]

# Direct condition use
print(arr[arr > 5])  # [ 6  7  8  9 10]
print(arr[arr % 2 == 0])  # [ 2  4  6  8 10]

# Compound conditions
print(arr[(arr > 3) & (arr < 8)])  # [4 5 6 7]
print(arr[(arr < 3) | (arr > 8)])  # [ 1  2  9 10]
```

### 3.4 Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Access by index array
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# 2D array
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Select specific rows
print(arr2d[[0, 2]])
# [[1 2 3]
#  [7 8 9]]

# Select specific elements
rows = [0, 1, 2]
cols = [0, 1, 2]
print(arr2d[rows, cols])  # [1 5 9] (diagonal elements)
```

---

## 4. Reshaping Arrays

### 4.1 reshape

```python
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Convert to 2D
arr2d = arr.reshape(3, 4)
print(arr2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Using -1: automatic calculation
arr2d = arr.reshape(4, -1)  # 4 rows, columns auto-calculated
print(arr2d.shape)  # (4, 3)

arr2d = arr.reshape(-1, 6)  # Rows auto-calculated, 6 columns
print(arr2d.shape)  # (2, 6)
```

### 4.2 flatten and ravel

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# flatten: creates copy
flat = arr2d.flatten()
print(flat)  # [1 2 3 4 5 6]

# ravel: creates view (shares original)
rav = arr2d.ravel()
print(rav)  # [1 2 3 4 5 6]
```

### 4.3 Transpose

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)

# Transpose
transposed = arr.T
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]
print(transposed.shape)  # (3, 2)

# Multidimensional transpose
arr3d = np.arange(24).reshape(2, 3, 4)
print(arr3d.transpose(1, 0, 2).shape)  # (3, 2, 4)
```

### 4.4 Adding/Removing Dimensions

```python
arr = np.array([1, 2, 3])
print(arr.shape)  # (3,)

# Add dimension
arr_2d = arr[np.newaxis, :]  # or arr.reshape(1, -1)
print(arr_2d.shape)  # (1, 3)

arr_col = arr[:, np.newaxis]  # or arr.reshape(-1, 1)
print(arr_col.shape)  # (3, 1)

# Using expand_dims
arr_exp = np.expand_dims(arr, axis=0)
print(arr_exp.shape)  # (1, 3)

# squeeze: remove dimensions of size 1
arr = np.array([[[1, 2, 3]]])
print(arr.shape)  # (1, 1, 3)
print(np.squeeze(arr).shape)  # (3,)
```

---

## 5. Array Operations

### 5.1 Basic Arithmetic Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise operations
print(a + b)   # [11 22 33 44]
print(a - b)   # [ -9 -18 -27 -36]
print(a * b)   # [ 10  40  90 160]
print(a / b)   # [0.1 0.1 0.1 0.1]
print(a ** 2)  # [ 1  4  9 16]
print(a % 2)   # [1 0 1 0]
print(a // 2)  # [0 1 1 2]

# Scalar operations
print(a + 10)  # [11 12 13 14]
print(a * 2)   # [2 4 6 8]
```

### 5.2 Universal Functions (ufuncs)

```python
arr = np.array([1, 4, 9, 16, 25])

# Mathematical functions
print(np.sqrt(arr))   # [1. 2. 3. 4. 5.]
print(np.exp(arr))    # exponential
print(np.log(arr))    # natural logarithm
print(np.log10(arr))  # common logarithm

# Trigonometric functions
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(np.sin(angles))
print(np.cos(angles))
print(np.tan(angles))

# Rounding
arr = np.array([1.2, 2.5, 3.7, 4.4])
print(np.round(arr))   # [1. 2. 4. 4.]
print(np.floor(arr))   # [1. 2. 3. 4.]
print(np.ceil(arr))    # [2. 3. 4. 5.]
print(np.trunc(arr))   # [1. 2. 3. 4.]

# Absolute value
arr = np.array([-1, -2, 3, -4])
print(np.abs(arr))  # [1 2 3 4]
```

### 5.3 Aggregation Functions

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Overall aggregation
print(np.sum(arr))    # 21
print(np.mean(arr))   # 3.5
print(np.std(arr))    # 1.707...
print(np.var(arr))    # 2.916...
print(np.min(arr))    # 1
print(np.max(arr))    # 6
print(np.prod(arr))   # 720 (product of all elements)

# Axis-based aggregation
print(np.sum(arr, axis=0))  # column sum: [5 7 9]
print(np.sum(arr, axis=1))  # row sum: [6 15]

print(np.mean(arr, axis=0))  # column mean: [2.5 3.5 4.5]
print(np.mean(arr, axis=1))  # row mean: [2. 5.]

# Cumulative sum/product
print(np.cumsum(arr))  # [ 1  3  6 10 15 21]
print(np.cumprod(arr)) # [  1   2   6  24 120 720]

# Return indices
print(np.argmin(arr))  # 0 (index of minimum)
print(np.argmax(arr))  # 5 (index of maximum)
```

---

## 6. Broadcasting

Broadcasting is a core NumPy feature that enables operations between arrays of different sizes.

The key mental model: NumPy **virtually copies** the smaller array along any dimension of size 1 to match the larger array's shape — without actually allocating the extra memory. This means you get the concise syntax of writing `arr + row` instead of manually tiling `row` into a 2-D array, while paying no extra memory cost.

### 6.1 Broadcasting Rules

1. If arrays have different number of dimensions, prepend 1s to the shape of the smaller array
2. Arrays with size 1 in any dimension are stretched to match the size of the other array
3. If sizes are not equal and neither is 1, an error occurs

```python
# Scalar and array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + 10)
# [[11 12 13]
#  [14 15 16]]

# 1D and 2D
arr = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
print(arr + row)
# [[11 22 33]
#  [14 25 36]]

# Column vector and 2D
col = np.array([[100], [200]])
print(arr + col)
# [[101 102 103]
#  [204 205 206]]
```

### 6.2 Broadcasting Examples

```python
# Standardization
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean = np.mean(data, axis=0)  # [4. 5. 6.]
std = np.std(data, axis=0)    # [2.449 2.449 2.449]
standardized = (data - mean) / std

# Distance calculation
point = np.array([1, 2])
points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
print(distances)  # [2.236 1.    1.414 2.236]
```

---

## 7. Combining and Splitting Arrays

### 7.1 Combining Arrays

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stacking (row direction)
v_stack = np.vstack([a, b])
print(v_stack)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Horizontal stacking (column direction)
h_stack = np.hstack([a, b])
print(h_stack)
# [[1 2 5 6]
#  [3 4 7 8]]

# Using concatenate
concat_v = np.concatenate([a, b], axis=0)  # same as vstack
concat_h = np.concatenate([a, b], axis=1)  # same as hstack

# Depth stacking
d_stack = np.dstack([a, b])
print(d_stack.shape)  # (2, 2, 2)
```

### 7.2 Splitting Arrays

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Vertical split
v_split = np.vsplit(arr, 3)  # split into 3
print(len(v_split))  # 3

# Horizontal split
h_split = np.hsplit(arr, 2)  # split into 2
print(h_split[0])
# [[ 1  2]
#  [ 5  6]
#  [ 9 10]]

# Using split
split_arr = np.split(arr, [1, 2], axis=0)  # split at indices 1, 2
print(len(split_arr))  # 3
```

---

## 8. Copying and Views

### 8.1 View - Shallow Copy

```python
arr = np.array([1, 2, 3, 4, 5])

# Slicing creates a view
view = arr[1:4]
view[0] = 100

print(arr)   # [  1 100   3   4   5]  # original also changed
print(view)  # [100   3   4]
```

### 8.2 Copy - Deep Copy

```python
arr = np.array([1, 2, 3, 4, 5])

# Explicit copy
copy = arr.copy()
copy[0] = 100

print(arr)   # [1 2 3 4 5]  # original preserved
print(copy)  # [100 2 3 4 5]
```

---

## Practice Problems

### Problem 1: Array Creation
Create an array containing only multiples of 3 from 1 to 100.

```python
# Solution
arr = np.arange(3, 101, 3)
# or
arr = np.arange(1, 101)
arr = arr[arr % 3 == 0]
```

### Problem 2: Matrix Operations
Find the sum of diagonal elements of a 3x3 identity matrix.

```python
# Solution
eye = np.eye(3)
diagonal_sum = np.trace(eye)  # 3.0
# or
diagonal_sum = np.sum(np.diag(eye))
```

### Problem 3: Broadcasting
Normalize a 4x4 matrix by dividing each element by the maximum value of its column.

```python
# Solution
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

col_max = np.max(arr, axis=0)  # [13, 14, 15, 16]
normalized = arr / col_max
```

---

## Summary

| Feature | Functions/Methods |
|------|------------|
| Array Creation | `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()` |
| Array Attributes | `shape`, `dtype`, `ndim`, `size` |
| Indexing | `arr[i]`, `arr[i, j]`, `arr[condition]`, `arr[indices]` |
| Reshaping | `reshape()`, `flatten()`, `ravel()`, `T` |
| Operations | `+`, `-`, `*`, `/`, `np.sum()`, `np.mean()`, `np.std()` |
| Combining/Splitting | `np.vstack()`, `np.hstack()`, `np.split()` |

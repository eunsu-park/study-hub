# 2. NumPy Advanced

[Previous: NumPy Basics](./01_NumPy_Basics.md) | [Next: Pandas Basics](./03_Pandas_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement matrix multiplication, decomposition, and linear system solving using `np.linalg`
2. Calculate descriptive statistics, correlation coefficients, and covariance matrices with NumPy
3. Apply the new-style random number generator (`default_rng`) to sample from various probability distributions
4. Describe structured arrays and record arrays for heterogeneous tabular data
5. Compare C-order and Fortran-order memory layouts and explain their performance implications
6. Demonstrate vectorized operations and explain why they outperform Python loops
7. Apply advanced indexing techniques including `np.where`, masked arrays, and `np.take`/`np.put`
8. Implement array persistence using binary (`.npy`/`.npz`) and text (CSV) formats, and use memory mapping for large datasets

---

Once you are comfortable creating and manipulating arrays, the next step is leveraging NumPy's powerful computational toolkit. From solving systems of linear equations to generating reproducible random samples, these advanced features turn NumPy into a complete numerical computing environment. Understanding memory layout and vectorization will also help you write code that runs orders of magnitude faster than naive Python loops.

---

## 1. Linear Algebra

### 1.1 Matrix Multiplication

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication (dot product)
C = np.dot(A, B)
print(C)
# [[19 22]
#  [43 50]]

# @ operator (Python 3.5+)
C = A @ B

# matmul function
C = np.matmul(A, B)

# Vector dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 32
```

### 1.2 Matrix Decomposition

```python
A = np.array([[1, 2], [3, 4]])

# Determinant
det = np.linalg.det(A)
print(det)  # -2.0

# Inverse
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verification: A @ A_inv = I
print(A @ A_inv)
# [[1. 0.]
#  [0. 1.]]

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)
print("U:\n", U)
print("S:", S)
print("Vt:\n", Vt)

# QR decomposition
Q, R = np.linalg.qr(A)

# Cholesky decomposition (symmetric positive definite matrix)
B = np.array([[4, 2], [2, 5]])
L = np.linalg.cholesky(B)
```

### 1.3 Solving Linear Equations

```python
# Linear system Ax = b
# 2x + y = 5
# x + 3y = 6

A = np.array([[2, 1], [1, 3]])
b = np.array([5, 6])

# Solve
x = np.linalg.solve(A, b)
print(x)  # [1.8 1.4]

# Verification
print(A @ x)  # [5. 6.]

# Least squares
A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
b = np.array([2, 3, 4.5, 5])
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print("Coefficients:", x)  # [0.75 1.1]
```

### 1.4 Matrix Norms and Condition Number

```python
A = np.array([[1, 2], [3, 4]])

# Frobenius norm
fro_norm = np.linalg.norm(A, 'fro')

# L2 norm (spectral norm)
l2_norm = np.linalg.norm(A, 2)

# L1 norm
l1_norm = np.linalg.norm(A, 1)

# Infinity norm
inf_norm = np.linalg.norm(A, np.inf)

# Vector norm
v = np.array([3, 4])
print(np.linalg.norm(v))  # 5.0 (Euclidean distance)

# Condition number
cond = np.linalg.cond(A)
print("Condition number:", cond)
```

### 1.5 Matrix Rank and Trace

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Rank
rank = np.linalg.matrix_rank(A)
print("Rank:", rank)  # 2

# Trace (sum of diagonal)
trace = np.trace(A)
print("Trace:", trace)  # 15
```

---

## 2. Statistical Functions

### 2.1 Descriptive Statistics

```python
data = np.array([23, 45, 67, 89, 12, 34, 56, 78, 90, 11])

# Basic statistics
print("Mean:", np.mean(data))      # 50.5
print("Median:", np.median(data))  # 50.5
print("Std dev:", np.std(data))    # 28.07
print("Variance:", np.var(data))   # 788.25
print("Min:", np.min(data))        # 11
print("Max:", np.max(data))        # 90
print("Range:", np.ptp(data))      # 79 (peak to peak)

# Percentiles
print("25%:", np.percentile(data, 25))
print("50%:", np.percentile(data, 50))
print("75%:", np.percentile(data, 75))

# Quantiles
print("1st quartile:", np.quantile(data, 0.25))
print("3rd quartile:", np.quantile(data, 0.75))
```

### 2.2 Correlation and Covariance

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Correlation coefficient matrix
corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
# [[1.   0.77]
#  [0.77 1.  ]]

# Covariance matrix
cov_matrix = np.cov(x, y)
print(cov_matrix)
# [[2.5 1.5]
#  [1.5 1.3]]

# Multivariate data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.corrcoef(data))  # correlation between variables
```

### 2.3 Histograms and Frequency

```python
data = np.random.randn(1000)

# Calculate histogram
counts, bin_edges = np.histogram(data, bins=10)
print("Counts:", counts)
print("Bins:", bin_edges)

# Custom bins
counts, bin_edges = np.histogram(data, bins=[-3, -2, -1, 0, 1, 2, 3])

# Unique values and counts
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
unique, counts = np.unique(arr, return_counts=True)
print("Unique:", unique)   # [1 2 3 4]
print("Counts:", counts)   # [1 2 3 4]
```

---

## 3. Random Number Generation

### 3.1 Basic Random Numbers

```python
# New style (NumPy 1.17+)
rng = np.random.default_rng(seed=42)

# Uniform distribution [0, 1)
print(rng.random(5))

# Random integers
print(rng.integers(1, 100, size=10))

# Uniform distribution [low, high)
print(rng.uniform(0, 10, size=5))

# Legacy style
np.random.seed(42)
print(np.random.rand(5))        # [0, 1) uniform
print(np.random.randint(1, 10, 5))  # random integers
```

### 3.2 Probability Distributions

```python
rng = np.random.default_rng(42)

# Normal distribution (Gaussian)
normal = rng.normal(loc=0, scale=1, size=1000)  # mean 0, std 1
print(f"Mean: {normal.mean():.3f}, Std: {normal.std():.3f}")

# Standard normal distribution
standard_normal = rng.standard_normal(1000)

# Binomial distribution
binomial = rng.binomial(n=10, p=0.5, size=1000)  # n trials, probability p

# Poisson distribution
poisson = rng.poisson(lam=5, size=1000)  # mean 5

# Exponential distribution
exponential = rng.exponential(scale=2, size=1000)

# Beta distribution
beta = rng.beta(a=2, b=5, size=1000)

# Gamma distribution
gamma = rng.gamma(shape=2, scale=1, size=1000)

# Chi-square distribution
chisquare = rng.chisquare(df=5, size=1000)

# t distribution
t = rng.standard_t(df=10, size=1000)
```

### 3.3 Random Sampling

```python
rng = np.random.default_rng(42)

arr = np.array([10, 20, 30, 40, 50])

# Random choice
sample = rng.choice(arr, size=3, replace=False)  # without replacement
print(sample)

# Weighted probability
weights = [0.1, 0.1, 0.3, 0.3, 0.2]
sample = rng.choice(arr, size=10, p=weights)

# Shuffle array
arr_copy = arr.copy()
rng.shuffle(arr_copy)
print(arr_copy)

# Permutation (returns new array)
permuted = rng.permutation(arr)
print(permuted)
```

---

## 4. Structured Arrays

### 4.1 Structured dtype

```python
# Define structured array
dt = np.dtype([
    ('name', 'U20'),      # Unicode string (max 20 chars)
    ('age', 'i4'),        # 32-bit integer
    ('height', 'f8'),     # 64-bit float
    ('is_student', '?')   # boolean
])

# Create data
data = np.array([
    ('Alice', 25, 165.5, True),
    ('Bob', 30, 178.2, False),
    ('Charlie', 22, 172.0, True)
], dtype=dt)

# Field access
print(data['name'])    # ['Alice' 'Bob' 'Charlie']
print(data['age'])     # [25 30 22]
print(data[0])         # ('Alice', 25, 165.5, True)
print(data[0]['name']) # Alice

# Conditional filtering
students = data[data['is_student']]
print(students['name'])
```

### 4.2 Record Arrays

```python
# Convert to recarray (attribute access)
rec = data.view(np.recarray)
print(rec.name)    # ['Alice' 'Bob' 'Charlie']
print(rec.age)     # [25 30 22]
print(rec[0].name) # Alice
```

---

## 5. Memory Layout and Performance

### 5.1 C-order vs Fortran-order

```python
# C-order (row-major): default
c_arr = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(c_arr.flags['C_CONTIGUOUS'])  # True

# Fortran-order (column-major)
f_arr = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f_arr.flags['F_CONTIGUOUS'])  # True

# Check memory layout
print(c_arr.strides)  # (24, 8) - 24 bytes per row, 8 bytes per column
print(f_arr.strides)  # (8, 16)
```

### 5.2 View vs Copy Performance

```python
import time

arr = np.arange(10000000)

# Slicing (view) - fast
start = time.time()
view = arr[::2]
print(f"View creation: {time.time() - start:.6f}s")

# Copy - slow
start = time.time()
copy = arr[::2].copy()
print(f"Copy: {time.time() - start:.6f}s")
```

### 5.3 Vectorization vs Loops

```python
import time

n = 1000000
arr = np.random.rand(n)

# Python loop (slow)
start = time.time()
result = []
for x in arr:
    result.append(x ** 2)
print(f"Python loop: {time.time() - start:.4f}s")

# NumPy vectorization (fast)
start = time.time()
result = arr ** 2
print(f"NumPy vectorization: {time.time() - start:.4f}s")
```

### 5.4 Universal Functions Optimization

```python
# Using where
arr = np.array([1, -2, 3, -4, 5])
result = np.where(arr > 0, arr, 0)  # keep positive, set negative to 0
print(result)  # [1 0 3 0 5]

# Using select (multiple conditions)
conditions = [arr < 0, arr == 0, arr > 0]
choices = [-1, 0, 1]
result = np.select(conditions, choices)
print(result)  # [ 1 -1  1 -1  1]

# Using clip
arr = np.array([-5, -2, 0, 3, 7, 10])
result = np.clip(arr, 0, 5)  # limit between 0 and 5
print(result)  # [0 0 0 3 5 5]
```

---

## 6. Advanced Indexing and Masking

### 6.1 Using np.where

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indices satisfying condition
indices = np.where(arr > 5)
print(indices)  # (array([1, 2, 2, 2]), array([2, 0, 1, 2]))

# Conditional value assignment
result = np.where(arr % 2 == 0, 'even', 'odd')
print(result)
```

### 6.2 np.take and np.put

```python
arr = np.array([10, 20, 30, 40, 50])

# take: get elements by indices
indices = [0, 2, 4]
print(np.take(arr, indices))  # [10 30 50]

# put: place values at indices
np.put(arr, [0, 2, 4], [100, 300, 500])
print(arr)  # [100  20 300  40 500]
```

### 6.3 Masked Arrays

```python
# Create masked array
data = np.array([1, 2, -999, 4, -999, 6])
mask = (data == -999)

masked_arr = np.ma.masked_array(data, mask)
print(masked_arr)  # [1 2 -- 4 -- 6]
print(masked_arr.mean())  # 3.25 (excludes masked values)

# Fill masked values
filled = masked_arr.filled(0)
print(filled)  # [1 2 0 4 0 6]
```

---

## 7. Saving and Loading Arrays

### 7.1 Binary Format

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save/load single array
np.save('array.npy', arr)
loaded = np.load('array.npy')

# Save/load multiple arrays
np.savez('arrays.npz', arr1=arr, arr2=arr*2)
data = np.load('arrays.npz')
print(data['arr1'])
print(data['arr2'])

# Compressed save
np.savez_compressed('arrays_compressed.npz', arr1=arr)
```

### 7.2 Text Format

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save CSV
np.savetxt('array.csv', arr, delimiter=',', fmt='%d')

# Load CSV
loaded = np.loadtxt('array.csv', delimiter=',')

# Save with header
np.savetxt('array_header.csv', arr, delimiter=',',
           header='col1,col2,col3', comments='')

# genfromtxt (can handle missing values)
data = np.genfromtxt('array.csv', delimiter=',',
                     missing_values='NA', filling_values=0)
```

---

## 8. Memory Mapping

Useful for processing large files without loading everything into memory.

```python
# Create memory-mapped array
shape = (10000, 10000)
dtype = np.float64

# File-based memory mapping
mmap = np.memmap('large_array.dat', dtype=dtype, mode='w+', shape=shape)
mmap[:100, :100] = np.random.rand(100, 100)
mmap.flush()  # write to disk

# Read-only loading
mmap_read = np.memmap('large_array.dat', dtype=dtype, mode='r', shape=shape)
print(mmap_read[:10, :10])
```

---

## Practice Problems

### Problem 1: Linear Regression
Use least squares to find linear regression coefficients for the following data.

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])

# Solution
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"Slope: {m:.3f}, Intercept: {c:.3f}")
```

### Problem 2: Eigenvalue Decomposition of Covariance Matrix
Calculate the covariance matrix of 3-variable data and perform eigenvalue decomposition.

```python
data = np.random.randn(100, 3)
data[:, 1] = data[:, 0] * 2 + np.random.randn(100) * 0.1  # add correlation

# Solution
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### Problem 3: Monte Carlo Simulation
Estimate the area of a circle (π) using random numbers.

```python
# Solution
n = 1000000
rng = np.random.default_rng(42)
x = rng.uniform(-1, 1, n)
y = rng.uniform(-1, 1, n)
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * inside.sum() / n
print(f"π estimate: {pi_estimate:.6f}")
```

---

## Summary

| Feature | Functions/Methods |
|------|------------|
| Matrix Multiplication | `np.dot()`, `@`, `np.matmul()` |
| Linear Algebra | `np.linalg.inv()`, `solve()`, `eig()`, `svd()` |
| Statistics | `np.mean()`, `np.std()`, `np.corrcoef()`, `np.cov()` |
| Random | `np.random.default_rng()`, `random()`, `normal()`, `choice()` |
| Save/Load | `np.save()`, `np.load()`, `np.savetxt()`, `np.loadtxt()` |
| Performance | Vectorized operations, `np.where()`, memory mapping |

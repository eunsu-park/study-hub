# Tensor Operations

**Previous**: [Tensors](./02_Tensors.md) | **Next**: [Autograd](./04_Autograd.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Index and slice tensors using integer, boolean, and fancy indexing
2. Apply broadcasting rules to combine tensors of different shapes
3. Perform matrix multiplication using `@`, `torch.matmul`, and `torch.mm`
4. Use element-wise operations (arithmetic, comparison, logical)
5. Apply reduction operations (sum, mean, max, argmax) along specific dimensions
6. Concatenate and stack tensors along existing or new dimensions
7. Use `torch.where`, `torch.clamp`, and other conditional operations
8. Understand in-place operations and their naming convention

---

Tensor operations are the building blocks of all neural network computations. This lesson covers the operations you will use daily when working with PyTorch, from basic indexing to matrix algebra.

---

## 1. Indexing and Slicing

### 1.1 Basic Indexing

```python
import torch

x = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Single element
print(x[0, 0])     # tensor(1)
print(x[1, 2])     # tensor(6)
print(x[-1, -1])   # tensor(9)

# Row and column
print(x[0])         # tensor([1, 2, 3])  -- first row
print(x[:, 0])      # tensor([1, 4, 7])  -- first column
print(x[:, -1])     # tensor([3, 6, 9])  -- last column
```

### 1.2 Slicing

```python
x = torch.arange(20).view(4, 5)
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])

# Slice rows
print(x[1:3])       # rows 1-2
# tensor([[ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14]])

# Slice rows and columns
print(x[1:3, 2:4])  # rows 1-2, columns 2-3
# tensor([[ 7,  8],
#         [12, 13]])

# Step slicing
print(x[::2])       # every other row
# tensor([[ 0,  1,  2,  3,  4],
#         [10, 11, 12, 13, 14]])

# Reverse
print(x[:, ::-1])   # reverse columns (PyTorch 2.x)
```

> **Note**: Slices return **views** -- they share memory with the original tensor.

### 1.3 Boolean (Mask) Indexing

```python
x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])

# Boolean mask
mask = x > 0
print(mask)        # tensor([ True, False,  True, False,  True])
print(x[mask])     # tensor([1., 3., 5.])

# In-place modification with mask
x[x < 0] = 0
print(x)           # tensor([1., 0., 3., 0., 5.])

# Multi-dimensional
m = torch.randn(3, 4)
print(m[m > 0])    # 1D tensor of all positive elements
```

### 1.4 Fancy (Advanced) Indexing

```python
x = torch.tensor([10, 20, 30, 40, 50])

# Index with a list of indices
idx = torch.tensor([0, 2, 4])
print(x[idx])      # tensor([10, 30, 50])

# 2D fancy indexing
m = torch.arange(12).view(3, 4)
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([1, 2, 3])
print(m[rows, cols])  # tensor([ 1,  6, 11])  -- diagonal-like

# gather: for batch-wise element selection
src = torch.tensor([[1, 2], [3, 4], [5, 6]])
idx = torch.tensor([[0], [1], [0]])
print(torch.gather(src, 1, idx))  # tensor([[1], [4], [5]])
```

---

## 2. Element-wise Operations

### 2.1 Arithmetic

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Operators
print(a + b)    # tensor([5., 7., 9.])
print(a - b)    # tensor([-3., -3., -3.])
print(a * b)    # tensor([ 4., 10., 18.])
print(a / b)    # tensor([0.2500, 0.4000, 0.5000])
print(a ** 2)   # tensor([1., 4., 9.])

# Functions
print(torch.add(a, b))       # same as a + b
print(torch.mul(a, b))       # same as a * b
print(torch.div(a, b))       # same as a / b
print(torch.pow(a, 2))       # same as a ** 2

# Scalar operations
print(a + 10)   # tensor([11., 12., 13.])
print(a * 0.5)  # tensor([0.5000, 1.0000, 1.5000])
```

### 2.2 Math Functions

```python
x = torch.tensor([0.0, 1.0, 2.0, 3.0])

print(torch.exp(x))     # tensor([ 1.0000,  2.7183,  7.3891, 20.0855])
print(torch.log(x + 1)) # tensor([0.0000, 0.6931, 1.0986, 1.3863])
print(torch.sqrt(x))    # tensor([0.0000, 1.0000, 1.4142, 1.7321])
print(torch.abs(torch.tensor([-1.0, 2.0, -3.0])))  # tensor([1., 2., 3.])
print(torch.sin(x))     # tensor([0.0000, 0.8415, 0.9093, 0.1411])

# Clamp values to a range
y = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.clamp(y, min=-1.0, max=1.0))  # tensor([-1., -1.,  0.,  1.,  1.])
```

### 2.3 Comparison and Logical Operations

```python
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([2, 2, 2, 2])

# Comparison (returns bool tensor)
print(a > b)    # tensor([False, False,  True,  True])
print(a == b)   # tensor([False,  True, False, False])
print(a >= b)   # tensor([False,  True,  True,  True])
print(a != b)   # tensor([ True, False,  True,  True])

# Logical operations
x = torch.tensor([True, True, False, False])
y = torch.tensor([True, False, True, False])
print(x & y)    # tensor([ True, False, False, False])
print(x | y)    # tensor([ True,  True,  True, False])
print(~x)       # tensor([False, False,  True,  True])

# torch.where: conditional selection
cond = torch.tensor([True, False, True, False])
a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([10.0, 20.0, 30.0, 40.0])
print(torch.where(cond, a, b))  # tensor([ 1., 20.,  3., 40.])
```

---

## 3. Reduction Operations

### 3.1 Basic Reductions

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

# Global reductions
print(x.sum())     # tensor(21.)
print(x.mean())    # tensor(3.5000)
print(x.max())     # tensor(6.)
print(x.min())     # tensor(1.)
print(x.prod())    # tensor(720.)  (1*2*3*4*5*6)
print(x.std())     # tensor(1.8708)
```

### 3.2 Reductions Along Dimensions

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

# Along dim=0 (reduce rows, result has shape [3])
print(x.sum(dim=0))   # tensor([5., 7., 9.])
print(x.mean(dim=0))  # tensor([2.5000, 3.5000, 4.5000])

# Along dim=1 (reduce columns, result has shape [2])
print(x.sum(dim=1))   # tensor([ 6., 15.])
print(x.mean(dim=1))  # tensor([2., 5.])

# Keep dimensions (useful for broadcasting)
print(x.sum(dim=1, keepdim=True))
# tensor([[ 6.],
#         [15.]])
# shape: [2, 1] instead of [2]
```

### 3.3 argmax and argmin

```python
x = torch.tensor([[3, 1, 4],
                   [1, 5, 9],
                   [2, 6, 5]])

# Global argmax (flattened index)
print(x.argmax())      # tensor(5)  (index of 9 in flattened tensor)

# Along a dimension
print(x.argmax(dim=0)) # tensor([0, 2, 1])  -- column-wise max indices
print(x.argmax(dim=1)) # tensor([2, 2, 1])  -- row-wise max indices

# max returns both values and indices
values, indices = x.max(dim=1)
print(values)   # tensor([4, 9, 6])
print(indices)  # tensor([2, 2, 1])
```

### 3.4 Sorting

```python
x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])

# Sort
sorted_vals, sorted_idx = torch.sort(x)
print(sorted_vals)  # tensor([1, 1, 2, 3, 4, 5, 6, 9])
print(sorted_idx)   # tensor([1, 3, 6, 0, 2, 4, 7, 5])

# Top-k
vals, idx = torch.topk(x, k=3)
print(vals)  # tensor([9, 6, 5])
print(idx)   # tensor([5, 7, 4])
```

---

## 4. Broadcasting

Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions.

### 4.1 Broadcasting Rules

Two tensors are broadcastable if, for each dimension (from the trailing dimension):
1. The dimensions are equal, OR
2. One of the dimensions is 1, OR
3. One tensor has fewer dimensions (it is prepended with 1s)

```python
# Example 1: scalar broadcast
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])   # shape [2, 3]
y = torch.tensor(10)             # shape []
print((x + y).shape)             # [2, 3]

# Example 2: vector broadcast
x = torch.randn(2, 3)            # shape [2, 3]
y = torch.randn(3)               # shape [3] -> broadcast to [2, 3]
print((x + y).shape)             # [2, 3]

# Example 3: column vector broadcast
x = torch.randn(2, 3)            # shape [2, 3]
y = torch.randn(2, 1)            # shape [2, 1] -> broadcast to [2, 3]
print((x + y).shape)             # [2, 3]

# Example 4: outer product via broadcasting
a = torch.tensor([1, 2, 3]).unsqueeze(1)  # shape [3, 1]
b = torch.tensor([10, 20])                # shape [2]
print(a * b)
# tensor([[10, 20],
#         [20, 40],
#         [30, 60]])
```

### 4.2 Common Broadcasting Pattern: Normalization

```python
# Normalize each row to have zero mean and unit variance
x = torch.randn(4, 5)
mean = x.mean(dim=1, keepdim=True)  # shape [4, 1]
std = x.std(dim=1, keepdim=True)    # shape [4, 1]
x_normalized = (x - mean) / std     # broadcasting: [4, 5] - [4, 1]
```

### 4.3 Broadcasting Gotcha

```python
# These shapes are NOT broadcastable:
# [2, 3] and [2]  -- trailing dimensions 3 and 2 don't match

a = torch.randn(2, 3)
b = torch.randn(2)
# a + b  # ERROR: sizes not broadcastable

# Fix: add a dimension
result = a + b.unsqueeze(1)  # [2, 3] + [2, 1] -> [2, 3]
```

---

## 5. Matrix Operations

### 5.1 Matrix Multiplication

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)

# Three equivalent ways
C = A @ B                    # operator
C = torch.matmul(A, B)      # function (most general)
C = torch.mm(A, B)          # 2D only
print(C.shape)               # [2, 4]

# Matrix-vector multiplication
x = torch.randn(3)
y = A @ x                   # [2, 3] @ [3] -> [2]

# Batch matrix multiplication
batch_A = torch.randn(8, 2, 3)
batch_B = torch.randn(8, 3, 4)
batch_C = torch.bmm(batch_A, batch_B)   # [8, 2, 4]
# or
batch_C = batch_A @ batch_B             # works with broadcasting too
```

### 5.2 Dot Product and Outer Product

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Dot product
print(torch.dot(a, b))         # tensor(32.)  (1*4 + 2*5 + 3*6)

# Outer product
print(torch.outer(a, b))
# tensor([[ 4.,  5.,  6.],
#         [ 8., 10., 12.],
#         [12., 15., 18.]])
```

### 5.3 Linear Algebra

```python
A = torch.randn(3, 3)

# Transpose
print(A.T.shape)                    # [3, 3]

# Determinant
print(torch.linalg.det(A))         # scalar

# Inverse
A_inv = torch.linalg.inv(A)
print(A @ A_inv)                    # ~identity

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)

# Singular Value Decomposition
U, S, Vh = torch.linalg.svd(A)

# Solve linear system: Ax = b
b = torch.randn(3)
x = torch.linalg.solve(A, b)

# Norm
print(torch.linalg.norm(A))        # Frobenius norm
print(torch.linalg.norm(A, ord=2)) # spectral norm
```

---

## 6. Concatenation and Stacking

### 6.1 torch.cat (Concatenate)

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)

# Concatenate along dim=0 (vertically)
cat0 = torch.cat([a, b, c], dim=0)
print(cat0.shape)  # [6, 3]

# Concatenate along dim=1 (horizontally)
cat1 = torch.cat([a, b, c], dim=1)
print(cat1.shape)  # [2, 9]
```

### 6.2 torch.stack (New Dimension)

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = torch.randn(3, 4)

# Stack creates a NEW dimension
stacked = torch.stack([a, b, c], dim=0)
print(stacked.shape)  # [3, 3, 4]  -- 3 tensors of shape [3, 4]

stacked = torch.stack([a, b, c], dim=1)
print(stacked.shape)  # [3, 3, 4]

# Common use: create a batch from individual samples
samples = [torch.randn(28, 28) for _ in range(16)]
batch = torch.stack(samples)  # [16, 28, 28]
```

### 6.3 split and chunk

```python
x = torch.randn(6, 4)

# split by size
parts = torch.split(x, 2, dim=0)  # 3 tensors of shape [2, 4]

# split into unequal sizes
parts = torch.split(x, [1, 2, 3], dim=0)  # shapes: [1,4], [2,4], [3,4]

# chunk into n equal pieces
chunks = torch.chunk(x, 3, dim=0)  # 3 tensors of shape [2, 4]
```

---

## 7. In-Place Operations

In-place operations modify a tensor directly without creating a new one. They are marked with an underscore suffix:

```python
x = torch.tensor([1.0, 2.0, 3.0])

# In-place operations
x.add_(10)       # x = tensor([11., 12., 13.])
x.mul_(2)        # x = tensor([22., 24., 26.])
x.clamp_(0, 25)  # x = tensor([22., 24., 25.])
x.zero_()        # x = tensor([0., 0., 0.])
x.fill_(7)       # x = tensor([7., 7., 7.])

# In-place assignment
x = torch.randn(3, 4)
x[:, 0] = 0      # set first column to zero
```

> **Warning**: In-place operations on tensors with `requires_grad=True` are generally not allowed because they can corrupt the computational graph needed for backpropagation.

---

## 8. Practical Patterns

### 8.1 One-Hot Encoding

```python
labels = torch.tensor([0, 2, 1, 3])
num_classes = 4
one_hot = torch.zeros(len(labels), num_classes)
one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
print(one_hot)
# tensor([[1., 0., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 0., 1.]])

# Simpler with F.one_hot
import torch.nn.functional as F
one_hot = F.one_hot(labels, num_classes=4).float()
```

### 8.2 Masking and Padding

```python
# Create a padding mask for variable-length sequences
lengths = torch.tensor([3, 5, 2])
max_len = 5
mask = torch.arange(max_len).expand(3, -1) < lengths.unsqueeze(1)
print(mask)
# tensor([[ True,  True,  True, False, False],
#         [ True,  True,  True,  True,  True],
#         [ True,  True, False, False, False]])

# Apply mask: set padded positions to -inf (for attention)
scores = torch.randn(3, 5)
scores = scores.masked_fill(~mask, float('-inf'))
```

### 8.3 Einsum

```python
# Einstein summation -- compact notation for many tensor operations

# Matrix multiplication: C_ij = sum_k A_ik * B_kj
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.einsum('ik,kj->ij', A, B)  # same as A @ B

# Batch matrix multiplication
A = torch.randn(8, 2, 3)
B = torch.randn(8, 3, 4)
C = torch.einsum('bij,bjk->bik', A, B)

# Trace of a matrix
A = torch.randn(3, 3)
trace = torch.einsum('ii->', A)

# Diagonal
diag = torch.einsum('ii->i', A)
```

---

## Summary

| Category | Key Operations |
|----------|---------------|
| Indexing | `x[0]`, `x[:, 1]`, `x[mask]`, `x[indices]` |
| Element-wise | `+`, `-`, `*`, `/`, `**`, `torch.exp`, `torch.clamp` |
| Reduction | `.sum()`, `.mean()`, `.max()`, `.argmax()`, with `dim=` and `keepdim=` |
| Broadcasting | Trailing dimensions must match or be 1; use `unsqueeze` to align |
| Matrix ops | `@`, `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.linalg.*` |
| Concatenation | `torch.cat` (along existing dim), `torch.stack` (new dim) |
| In-place | Underscore suffix (`add_`, `mul_`, `zero_`); avoid with `requires_grad` |

---

**Next**: [Autograd](./04_Autograd.md) -- Automatic differentiation, computational graphs, and gradient computation.

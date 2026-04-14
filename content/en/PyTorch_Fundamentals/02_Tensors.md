# Tensors

**Previous**: [Introduction to PyTorch](./01_Introduction_to_PyTorch.md) | **Next**: [Tensor Operations](./03_Tensor_Operations.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create tensors using multiple methods (from data, from NumPy, factory functions)
2. Explain tensor attributes: shape, dtype, device, and layout
3. Choose the appropriate dtype for different use cases (float32 vs float16 vs int64)
4. Move tensors between CPU and GPU with `.to()` and understand the performance implications
5. Distinguish between views and copies, and predict when PyTorch shares memory
6. Reshape tensors using `view()`, `reshape()`, `unsqueeze()`, `squeeze()`, and `permute()`
7. Understand tensor memory layout (contiguous vs non-contiguous) and its impact on performance
8. Convert between PyTorch tensors and other formats (NumPy, Python lists, PIL images)

---

Tensors are the fundamental data structure of PyTorch. Every piece of data -- inputs, weights, gradients, outputs -- flows through PyTorch as a tensor. Mastering tensor creation, manipulation, and memory behavior is essential before working with any higher-level PyTorch API.

---

## 1. Tensor Creation

### 1.1 From Data

```python
import torch

# From a Python list
t1 = torch.tensor([1, 2, 3])
print(t1)        # tensor([1, 2, 3])
print(t1.dtype)  # torch.int64  (inferred from int data)

# From a nested list (2D)
t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t2.shape)  # torch.Size([2, 2])
print(t2.dtype)  # torch.float32  (inferred from float data)

# Explicit dtype
t3 = torch.tensor([1, 2, 3], dtype=torch.float32)
print(t3.dtype)  # torch.float32
```

### 1.2 From NumPy

```python
import numpy as np

np_array = np.array([1.0, 2.0, 3.0])

# Shared memory (zero-copy)
t_shared = torch.from_numpy(np_array)
np_array[0] = 99.0
print(t_shared[0])  # tensor(99.)  -- memory is shared!

# Independent copy
t_copy = torch.tensor(np_array)  # always copies data
np_array[0] = -1.0
print(t_copy[0])    # tensor(99.)  -- not affected
```

### 1.3 Factory Functions

```python
# Zeros and ones
z = torch.zeros(3, 4)        # 3x4 matrix of zeros
o = torch.ones(2, 3, 5)      # 2x3x5 tensor of ones

# Random tensors
u = torch.rand(3, 4)         # uniform [0, 1)
n = torch.randn(3, 4)        # standard normal N(0, 1)
ri = torch.randint(0, 10, (3, 4))  # random integers in [0, 10)

# Sequences
a = torch.arange(0, 10, 2)   # tensor([0, 2, 4, 6, 8])
l = torch.linspace(0, 1, 5)  # tensor([0.0, 0.25, 0.5, 0.75, 1.0])

# Special matrices
e = torch.eye(3)              # 3x3 identity matrix
d = torch.diag(torch.tensor([1, 2, 3]))  # diagonal matrix

# Like another tensor (same shape, dtype, device)
x = torch.randn(3, 4, device='cpu', dtype=torch.float32)
y = torch.zeros_like(x)      # same shape, dtype, device as x
z = torch.ones_like(x)
r = torch.rand_like(x)
```

### 1.4 Reproducibility with Seeds

```python
# Set seed for reproducibility
torch.manual_seed(42)
a = torch.randn(3)

torch.manual_seed(42)
b = torch.randn(3)

print(torch.equal(a, b))  # True -- same seed produces same values

# For GPU reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # for multi-GPU
```

---

## 2. Tensor Attributes

Every tensor has four key attributes:

### 2.1 Shape (size)

```python
t = torch.randn(2, 3, 4)

print(t.shape)      # torch.Size([2, 3, 4])
print(t.size())     # torch.Size([2, 3, 4])  -- same thing
print(t.ndim)       # 3  (number of dimensions)
print(t.numel())    # 24 (total number of elements: 2*3*4)

# Access individual dimensions
print(t.shape[0])   # 2
print(t.size(1))    # 3
```

### 2.2 Data Type (dtype)

```python
# Common dtypes
t_f32 = torch.tensor([1.0])         # torch.float32 (default for floats)
t_f64 = torch.tensor([1.0]).double() # torch.float64
t_f16 = torch.tensor([1.0]).half()   # torch.float16
t_bf16 = torch.tensor([1.0]).bfloat16()  # torch.bfloat16
t_i64 = torch.tensor([1])           # torch.int64 (default for ints)
t_i32 = torch.tensor([1], dtype=torch.int32)
t_bool = torch.tensor([True, False]) # torch.bool

# Type casting
x = torch.tensor([1, 2, 3])       # int64
x_float = x.float()               # float32
x_half = x.half()                 # float16
x_double = x.to(torch.float64)    # float64

print(x_float.dtype)  # torch.float32
```

**When to use which dtype:**

| dtype | Bits | Use case |
|-------|------|----------|
| `float32` | 32 | Default for model parameters and training |
| `float16` | 16 | Mixed precision training (AMP), inference |
| `bfloat16` | 16 | Mixed precision on newer GPUs (better range than fp16) |
| `float64` | 64 | Scientific computing, loss computation (rarely needed) |
| `int64` | 64 | Indices, labels, token IDs |
| `int32` | 32 | Indices when memory matters |
| `bool` | 8 | Masks, conditions |

### 2.3 Device

```python
# CPU tensor (default)
cpu_tensor = torch.tensor([1.0, 2.0])
print(cpu_tensor.device)  # cpu

# GPU tensor
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1.0, 2.0], device='cuda')
    print(gpu_tensor.device)  # cuda:0

    # Move CPU tensor to GPU
    moved = cpu_tensor.to('cuda')
    # or
    moved = cpu_tensor.cuda()

    # Move GPU tensor to CPU
    back = gpu_tensor.to('cpu')
    # or
    back = gpu_tensor.cpu()

# Device-agnostic code pattern
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 4, device=device)
```

> **Important**: Operations between tensors on different devices will raise an error. Always ensure tensors are on the same device before combining them.

### 2.4 requires_grad

```python
# For parameters that need gradient computation
w = torch.randn(3, requires_grad=True)
print(w.requires_grad)  # True

# Detach from computation graph
w_detached = w.detach()
print(w_detached.requires_grad)  # False

# Data tensors typically don't need gradients
x = torch.randn(3)  # requires_grad=False by default
```

---

## 3. View vs Copy

Understanding when PyTorch shares memory is critical for both correctness and performance.

### 3.1 Views (Shared Memory)

A **view** is a different way of looking at the same data -- no data is copied:

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# view() creates a view
y = x.view(2, 3)
print(y)
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

# Modifying y modifies x (shared memory!)
y[0, 0] = 99.0
print(x[0])  # tensor(99.)

# These operations also return views:
# .view(), .reshape() (when possible), .T, .transpose(),
# .expand(), .narrow(), slicing (x[1:3])
```

### 3.2 Copies (Independent Memory)

```python
x = torch.tensor([1.0, 2.0, 3.0])

# .clone() creates a copy
y = x.clone()
y[0] = 99.0
print(x[0])  # tensor(1.)  -- x is not affected

# torch.tensor() always copies
z = torch.tensor(x)  # independent copy

# .contiguous() copies only if non-contiguous
t = torch.randn(3, 4).T         # transpose makes it non-contiguous
t_contig = t.contiguous()       # copies data to contiguous layout
```

### 3.3 Checking if Tensors Share Memory

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = x.view(2, 2)
z = x.clone()

print(x.data_ptr() == y.data_ptr())  # True  -- same memory
print(x.data_ptr() == z.data_ptr())  # False -- different memory

# Also useful: x.storage().data_ptr()
print(x.storage().data_ptr() == y.storage().data_ptr())  # True
```

---

## 4. Reshaping Operations

### 4.1 view() vs reshape()

```python
x = torch.arange(12)  # tensor([0, 1, 2, ..., 11])

# view() requires contiguous memory
y = x.view(3, 4)     # OK: x is contiguous

# reshape() works even for non-contiguous tensors
t = torch.randn(3, 4).T  # non-contiguous
# t.view(12)             # ERROR: not contiguous
r = t.reshape(12)         # OK: copies if needed

# Use -1 to infer one dimension
z = x.view(3, -1)   # shape: [3, 4]  (-1 inferred as 12/3=4)
z = x.view(-1, 6)   # shape: [2, 6]
```

### 4.2 squeeze() and unsqueeze()

```python
x = torch.randn(1, 3, 1, 4)

# squeeze: remove dimensions of size 1
y = x.squeeze()       # shape: [3, 4]
y = x.squeeze(0)      # shape: [3, 1, 4]  (only dim 0)
y = x.squeeze(2)      # shape: [1, 3, 4]  (only dim 2)

# unsqueeze: add a dimension of size 1
z = torch.randn(3, 4)
z = z.unsqueeze(0)    # shape: [1, 3, 4]  (add batch dim)
z = z.unsqueeze(-1)   # shape: [1, 3, 4, 1]  (add at end)
```

### 4.3 permute() and transpose()

```python
x = torch.randn(2, 3, 4)  # [batch, height, width]

# transpose: swap two dimensions
y = x.transpose(1, 2)     # [2, 4, 3] -- swap height and width

# permute: reorder all dimensions
z = x.permute(2, 0, 1)    # [4, 2, 3] -- width, batch, height

# .T for 2D tensors only
m = torch.randn(3, 4)
print(m.T.shape)           # [4, 3]

# .mT for batch matrix transpose (last two dims)
batch = torch.randn(5, 3, 4)
print(batch.mT.shape)      # [5, 4, 3]
```

### 4.4 flatten() and unflatten()

```python
x = torch.randn(2, 3, 4)

# Flatten all dimensions
flat = x.flatten()          # shape: [24]

# Flatten specific range
flat_partial = x.flatten(1)  # shape: [2, 12] -- flatten dims 1 and 2
flat_range = x.flatten(1, 2) # shape: [2, 12] -- same

# Unflatten a dimension
y = torch.randn(2, 12)
z = y.unflatten(1, (3, 4))  # shape: [2, 3, 4]
```

---

## 5. Memory Layout and Contiguity

### 5.1 Strides

PyTorch uses **strides** to map multi-dimensional indices to flat memory:

```python
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

print(x.stride())  # (3, 1)
# stride[0]=3: moving one row means skipping 3 elements
# stride[1]=1: moving one column means skipping 1 element

# After transpose, strides change but data doesn't move
y = x.T
print(y.stride())  # (1, 3)
print(y.is_contiguous())  # False
```

### 5.2 Why Contiguity Matters

```python
x = torch.randn(3, 4)
y = x.T  # non-contiguous

# Some operations require contiguous tensors
# y.view(12)  # ERROR!

# Solution 1: use reshape (copies if needed)
flat = y.reshape(12)

# Solution 2: explicitly make contiguous
y_c = y.contiguous()
flat = y_c.view(12)

# .contiguous() is a no-op if already contiguous
x_c = x.contiguous()  # no copy, x is already contiguous
print(x.data_ptr() == x_c.data_ptr())  # True
```

---

## 6. Type Casting and Conversion

### 6.1 Between dtypes

```python
x = torch.tensor([1, 2, 3])  # int64

# Method 1: .to()
x_f32 = x.to(torch.float32)

# Method 2: convenience methods
x_f32 = x.float()
x_f64 = x.double()
x_f16 = x.half()
x_i32 = x.int()
x_bool = x.bool()

# Method 3: .type()
x_f32 = x.type(torch.FloatTensor)
```

### 6.2 Between Formats

```python
import numpy as np

# Tensor -> NumPy
t = torch.tensor([1.0, 2.0, 3.0])
n = t.numpy()        # shared memory (CPU only)
n = t.detach().cpu().numpy()  # safe for any tensor

# Tensor -> Python scalar
s = torch.tensor(3.14)
print(s.item())      # 3.14 (Python float)

# Tensor -> Python list
t = torch.tensor([[1, 2], [3, 4]])
print(t.tolist())    # [[1, 2], [3, 4]]
```

---

## 7. Common Pitfalls

### 7.1 Integer Division

```python
# Python 3 division returns float, but integer tensor division truncates
a = torch.tensor(7)
b = torch.tensor(2)
print(a / b)          # tensor(3.5000) -- float division (PyTorch 2.x)
print(a // b)         # tensor(3)      -- integer division
```

### 7.2 In-Place Operations

```python
x = torch.randn(3, requires_grad=True)

# In-place operations end with underscore
# x.add_(1)  # ERROR when requires_grad=True (modifies leaf variable)

# Safe: out-of-place
y = x + 1  # creates new tensor

# In-place is fine for non-grad tensors
z = torch.randn(3)
z.add_(1)    # OK
z.mul_(2)    # OK
z.zero_()    # OK
```

### 7.3 Device Mismatch

```python
if torch.cuda.is_available():
    cpu_t = torch.tensor([1.0])
    gpu_t = torch.tensor([2.0], device='cuda')

    # cpu_t + gpu_t  # ERROR: expected same device

    # Fix: move to same device
    result = cpu_t.to('cuda') + gpu_t
    # or
    result = cpu_t + gpu_t.to('cpu')
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Creation | `torch.tensor()` copies; `torch.from_numpy()` shares memory |
| Shape | `.shape`, `.ndim`, `.numel()` for dimension inspection |
| dtype | `float32` default for floats; use `float16`/`bfloat16` for efficiency |
| Device | Always keep interacting tensors on the same device |
| Views | `view()`, slicing, `.T` share memory -- mutations propagate |
| Copies | `.clone()`, `torch.tensor()` create independent copies |
| Reshaping | `view()` needs contiguous; `reshape()` handles both cases |
| Contiguity | Transposed tensors are non-contiguous; use `.contiguous()` when needed |

---

**Next**: [Tensor Operations](./03_Tensor_Operations.md) -- Indexing, slicing, broadcasting, and matrix operations.

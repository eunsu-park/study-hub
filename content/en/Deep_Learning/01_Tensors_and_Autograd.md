# 01. Tensors and Autograd

[Next: Neural Network Basics](./02_Neural_Network_Basics.md)

---

> **PyTorch 2.x Note**: This lesson is based on PyTorch 2.0+ (2023~).
>
> Key PyTorch 2.0 features:
> - `torch.compile()`: Graph compilation for significant training/inference speedup
> - `torch.func`: Function transforms (vmap, grad, jacrev, etc.)
> - Enhanced CUDA graph support
>
> Installation: `pip install torch>=2.0`

## Learning Objectives

- Understand the concept of tensors and their differences from NumPy arrays
- Understand PyTorch's automatic differentiation (Autograd) system
- Learn the basics of GPU operations
- (PyTorch 2.x) torch.compile basics

---

## 1. What is a Tensor?

A tensor is a generalized concept of multi-dimensional arrays.

| Dimension | Name | Example |
|-----------|------|---------|
| 0D | Scalar | Single number (5) |
| 1D | Vector | [1, 2, 3] |
| 2D | Matrix | [[1,2], [3,4]] |
| 3D | 3D Tensor | Image (H, W, C) |
| 4D | 4D Tensor | Batch of images (N, C, H, W) |

---

## 2. NumPy vs PyTorch Tensor Comparison

Why do we need a new data structure when NumPy already provides n-dimensional arrays? NumPy arrays live on the CPU and have no concept of gradient tracking. PyTorch tensors carry additional metadata — `device` (CPU or GPU), `requires_grad` (whether to record operations), and a reference to the computational graph — that together enable automatic differentiation, the backbone of all neural network training. In short, a PyTorch tensor is a NumPy array *plus* the bookkeeping needed to train models.

### Creation

```python
import numpy as np
import torch

# NumPy
np_arr = np.array([1, 2, 3])
np_zeros = np.zeros((3, 4))
np_rand = np.random.randn(3, 4)

# PyTorch
pt_tensor = torch.tensor([1, 2, 3])
pt_zeros = torch.zeros(3, 4)
pt_rand = torch.randn(3, 4)
```

### Conversion

```python
# NumPy → PyTorch
tensor = torch.from_numpy(np_arr)

# PyTorch → NumPy
array = tensor.numpy()  # Only works for CPU tensors
```

### Key Differences

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| GPU Support | ❌ | ✅ (`tensor.to('cuda')`) |
| Automatic Differentiation | ❌ | ✅ (`requires_grad=True`) |
| Default Type | float64 | float32 |
| Memory Sharing | - | `from_numpy` shares memory |

---

## 3. Automatic Differentiation (Autograd)

A core feature of PyTorch that automatically computes backpropagation.

Training a neural network requires computing the gradient of the loss with respect to every parameter — potentially millions of partial derivatives. Doing this by hand is impractical. Autograd solves this by recording every operation in a computational graph during the forward pass, then walking the graph in reverse to compute all gradients automatically via the chain rule. This is what makes the leap from "defining a model" to "training a model" almost effortless.

### Basic Usage

```python
# Why: requires_grad=True tells PyTorch to record every operation on this tensor
# into the computational graph, so that gradients can be computed later via .backward().
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# Why: .backward() traverses the computational graph in reverse (topological order)
# to compute all partial derivatives via the chain rule.
y.backward()

# Check gradient
print(x.grad)  # tensor([7.])  # dy/dx = 2x + 3 = 2*2 + 3 = 7
```

### Computational Graph

```
    x ─────┐
           │
    x² ────┼──▶ + ──▶ y
           │
    3x ────┘
```

- **Forward pass**: Computation from input → output. Each operation (`**`, `*`, `+`) is recorded as a node in a directed acyclic graph (DAG). PyTorch builds this graph dynamically — every time you run a computation, a fresh graph is created.
- **Backward pass**: Starting at the output, PyTorch walks the graph in reverse (topological order) and applies the chain rule at each node to compute ∂y/∂x. After `.backward()` completes, the graph is **destroyed** by default (`retain_graph=False`), freeing memory.

**Chain Rule in Action — a concrete example.** Consider a composite function `y = f(g(x))` where `g(x) = x²` and `f(u) = 3u + 1`. For `x = 2`:

```
Forward:  g = x² = 4,   y = 3g + 1 = 13
Backward: dy/dg = 3,    dg/dx = 2x = 4
          dy/dx = (dy/dg) × (dg/dx) = 3 × 4 = 12
```

Each node only needs its *local* derivative (how its output changes w.r.t. its input), and the chain rule multiplies them together. This is exactly what autograd does at every node in the computational graph — no matter how deep the network.

### Gradient Accumulation and Initialization

```python
# PyTorch accumulates gradients by default — calling backward() adds to
# existing .grad values rather than replacing them.  This is intentional:
# it allows gradient accumulation across multiple mini-batches (useful when
# the desired batch size exceeds GPU memory).  However, in a standard
# training loop you must zero gradients before each step, otherwise the
# optimizer uses the *sum* of all past gradients.
x.grad.zero_()  # Reset to 0; without this, gradients from previous steps pile up
```

---

## 4. Operations and Broadcasting

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Basic operations
c = a + b           # Element-wise addition
c = a * b           # Element-wise multiplication (Hadamard product)
c = a @ b           # Matrix multiplication
c = torch.matmul(a, b)  # Matrix multiplication

# Broadcasting
a = torch.tensor([[1], [2], [3]])  # (3, 1)
b = torch.tensor([10, 20, 30])     # (3,)
c = a + b  # (3, 3) automatic expansion
```

---

## 5. GPU Operations

```python
# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Move tensor to GPU
x = torch.randn(1000, 1000)
x_gpu = x.to(device)
# Or
x_gpu = x.cuda()

# Operations (performed on the same device)
y_gpu = x_gpu @ x_gpu

# Bring result back to CPU
y_cpu = y_gpu.cpu()
```

---

## 6. Exercise: NumPy vs PyTorch Automatic Differentiation Comparison

### Problem: Find the derivative of f(x) = x³ + 2x² - 5x + 3 at x=2

Mathematical solution:
- f'(x) = 3x² + 4x - 5
- f'(2) = 3(4) + 4(2) - 5 = 12 + 8 - 5 = 15

### NumPy (Manual Differentiation)

```python
import numpy as np

def f(x):
    return x**3 + 2*x**2 - 5*x + 3

def df(x):
    """Manually compute derivative"""
    return 3*x**2 + 4*x - 5

x = 2.0
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {df(x)}")  # 15.0
```

### PyTorch (Automatic Differentiation)

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**3 + 2*x**2 - 5*x + 3

y.backward()
print(f"f({x.item()}) = {y.item()}")
print(f"f'({x.item()}) = {x.grad.item()}")  # 15.0
```

---

## 7. Important Notes

### In-place Operations

```python
# In-place operations can conflict with autograd
x = torch.tensor([1.0], requires_grad=True)
# x += 1  # May cause error
x = x + 1  # Create new tensor (safe)
```

### Disabling Gradient Tracking

```python
# Why: During inference we don't need gradients, so wrapping in torch.no_grad()
# skips building the computational graph — saving memory and improving speed
# (typically 20-30% faster for forward-only passes).
with torch.no_grad():
    y = model(x)  # No gradient computation

# Or
x.requires_grad = False
```

### detach()

```python
# Detach from computational graph — creates a new tensor that shares the
# same data but is not part of the autograd graph.  Common uses:
#   1. Prevent gradients flowing into a frozen sub-network (e.g., target
#      network in DQN, discriminator update in GANs)
#   2. Convert a tracked tensor to a plain value for logging/plotting
y = x.detach()  # y has the same values as x but no gradient history
```

---

## 8. PyTorch 2.x New Features

### torch.compile()

The flagship feature of PyTorch 2.0, compiling models for improved performance.

```python
import torch

# Define model
model = MyModel()

# Compile the model (PyTorch 2.0+)
compiled_model = torch.compile(model)

# Usage is the same
output = compiled_model(input_data)
```

### Compilation Modes

```python
# Default mode (balanced)
model = torch.compile(model)

# Maximum performance mode
model = torch.compile(model, mode="max-autotune")

# Memory-saving mode
model = torch.compile(model, mode="reduce-overhead")
```

### torch.func (Function Transforms)

```python
from torch.func import vmap, grad, jacrev

# vmap: Automatic batch operations
def single_fn(x):
    return x ** 2

batched_fn = vmap(single_fn)
result = batched_fn(torch.randn(10, 3))  # Batch processing

# grad: Functional gradients
def f(x):
    return (x ** 2).sum()

grad_f = grad(f)
x = torch.randn(3)
print(grad_f(x))  # 2 * x
```

### Notes

```python
# torch.compile has compilation overhead on first run
# Warm-up recommended for production

# Dynamic shapes may cause recompilation
# Mitigate with dynamic=True option
model = torch.compile(model, dynamic=True)
```

---

## Summary

### What to Understand from NumPy
- Tensors are multi-dimensional arrays
- Matrix operations (multiplication, transpose, broadcasting)

### What PyTorch Adds
- `requires_grad`: Enable automatic differentiation
- `backward()`: Perform backpropagation
- `grad`: Computed gradients
- GPU acceleration

### PyTorch 2.x Additions
- `torch.compile()`: Performance optimization
- `torch.func`: Function transforms (vmap, grad)

---

## Next Steps

In [02_Neural_Network_Basics.md](./02_Neural_Network_Basics.md), we'll use these tensors and automatic differentiation to build neural networks.

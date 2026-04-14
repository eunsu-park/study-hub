# Introduction to PyTorch

**Next**: [Tensors](./02_Tensors.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe PyTorch's history and its position in the deep learning ecosystem
2. Explain why PyTorch uses dynamic computational graphs (define-by-run)
3. Install PyTorch and verify the installation on CPU and GPU
4. Create your first tensor and perform basic arithmetic operations
5. Compare PyTorch with other frameworks (TensorFlow, JAX) at a high level
6. Navigate the PyTorch documentation and community resources
7. Understand the relationship between PyTorch and its predecessor Torch (Lua)

---

## 1. What is PyTorch?

PyTorch is an open-source machine learning framework developed by Meta AI (formerly Facebook AI Research, FAIR). It provides two core capabilities:

1. **N-dimensional tensor computation** with strong GPU acceleration (similar to NumPy but on GPU)
2. **Automatic differentiation** for building and training neural networks

### 1.1 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Dynamic graphs** | Computational graph is built on-the-fly during execution (eager mode) |
| **Pythonic API** | Feels like native Python -- uses standard control flow (`if`, `for`, `while`) |
| **Strong GPU support** | Seamless CPU/GPU tensor movement with `.to(device)` |
| **Research-first** | Dominant framework in academic ML research (80%+ of papers at NeurIPS, ICML) |
| **Production-ready** | TorchScript, ONNX export, TorchServe for deployment |

### 1.2 Brief History

```
2002: Torch (Lua) created at NYU by Ronan Collobert et al.
2016: PyTorch 0.1 released by FAIR -- Python frontend for Torch's C backend
2018: PyTorch 1.0 merges Caffe2 (production) with PyTorch (research)
2019: PyTorch overtakes TensorFlow in research paper adoption
2022: PyTorch 2.0 introduces torch.compile() for graph-mode optimization
2023: PyTorch moves to Linux Foundation (PyTorch Foundation)
2024: PyTorch 2.x continues with FlexAttention, torch.export, etc.
```

---

## 2. PyTorch vs Other Frameworks

### 2.1 Comparison Table

| Aspect | PyTorch | TensorFlow | JAX |
|--------|---------|------------|-----|
| **Graph mode** | Eager (default), compile optional | Eager (TF2), graph (tf.function) | Functional transforms (jit, vmap) |
| **API style** | Object-oriented (nn.Module) | Keras layers + tf.function | Pure functions + transforms |
| **Debugging** | Standard Python debugger works | Harder in graph mode | Requires functional style |
| **Research adoption** | Dominant (~80%) | Declining in research | Growing, especially at Google |
| **Deployment** | TorchScript, ONNX, ExecuTorch | TF Lite, TF Serving, TF.js | JAX2TF, Orbax |
| **Learning curve** | Gentle for Python developers | Moderate (Keras easy, raw TF hard) | Steep (functional paradigm) |

### 2.2 Why PyTorch Wins in Research

```python
# PyTorch: natural Python control flow in models
class DynamicNet(nn.Module):
    def forward(self, x):
        # Standard Python if/else -- works perfectly
        if x.sum() > 0:
            return self.positive_branch(x)
        else:
            return self.negative_branch(x)
```

This is possible because PyTorch uses **eager execution** -- operations execute immediately, just like regular Python. The computational graph is built implicitly as operations run, which is why it's called "define-by-run."

---

## 3. Installation

### 3.1 CPU-Only Installation

```bash
# Using pip
pip install torch torchvision torchaudio

# Using conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3.2 GPU Installation (CUDA)

```bash
# CUDA 12.1 (check your NVIDIA driver version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 Verify Installation

```python
import torch

# Version info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:    {torch.version.cuda}")
    print(f"GPU device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU count:       {torch.cuda.device_count()}")

# Quick functionality test
x = torch.tensor([1.0, 2.0, 3.0])
print(f"\nTensor: {x}")
print(f"Sum:    {x.sum()}")
print(f"Device: {x.device}")
```

Expected output (CPU):
```
PyTorch version: 2.2.0
CUDA available:  False
Tensor: tensor([1., 2., 3.])
Sum:    6.0
Device: cpu
```

---

## 4. Your First Tensor

A **tensor** is the fundamental data structure in PyTorch -- a multi-dimensional array with support for automatic differentiation and GPU acceleration.

### 4.1 Creating Tensors

```python
import torch

# From Python lists
a = torch.tensor([1, 2, 3])
print(a)         # tensor([1, 2, 3])
print(a.dtype)   # torch.int64

# From a 2D list (matrix)
b = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])
print(b.shape)   # torch.Size([2, 2])
print(b.dtype)   # torch.float32

# Common creation functions
zeros = torch.zeros(3, 4)          # 3x4 matrix of zeros
ones = torch.ones(2, 3)            # 2x3 matrix of ones
rand = torch.rand(2, 3)            # uniform random [0, 1)
randn = torch.randn(2, 3)          # standard normal distribution
arange = torch.arange(0, 10, 2)    # tensor([0, 2, 4, 6, 8])
eye = torch.eye(3)                 # 3x3 identity matrix
```

### 4.2 Basic Arithmetic

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
print(x + y)       # tensor([5., 7., 9.])
print(x * y)       # tensor([ 4., 10., 18.])
print(x ** 2)      # tensor([1., 4., 9.])

# Equivalent functional forms
print(torch.add(x, y))   # tensor([5., 7., 9.])
print(torch.mul(x, y))   # tensor([ 4., 10., 18.])

# Reduction operations
print(x.sum())     # tensor(6.)
print(x.mean())    # tensor(2.)
print(x.max())     # tensor(3.)
print(x.min())     # tensor(1.)

# Dot product
print(torch.dot(x, y))  # tensor(32.)  (1*4 + 2*5 + 3*6)
```

### 4.3 NumPy Interoperability

PyTorch tensors and NumPy arrays can share memory for zero-copy conversion:

```python
import numpy as np

# NumPy to PyTorch (shared memory on CPU)
np_array = np.array([1.0, 2.0, 3.0])
tensor_from_np = torch.from_numpy(np_array)

# Modifying one affects the other (shared memory!)
np_array[0] = 99.0
print(tensor_from_np)  # tensor([99.,  2.,  3.])

# PyTorch to NumPy (shared memory on CPU)
tensor = torch.tensor([1.0, 2.0, 3.0])
np_from_tensor = tensor.numpy()

# Independent copy (no shared memory)
tensor_copy = torch.tensor(np_array)  # always copies
```

> **Warning**: Shared memory between NumPy and PyTorch only works for CPU tensors. GPU tensors must be moved to CPU first with `.cpu()`.

---

## 5. PyTorch's Core Components

PyTorch is organized into several submodules, each serving a specific purpose:

```
torch
├── torch.Tensor          # Multi-dimensional array (the core data structure)
├── torch.autograd        # Automatic differentiation engine
├── torch.nn              # Neural network layers, loss functions
├── torch.optim           # Optimization algorithms (SGD, Adam, etc.)
├── torch.utils.data      # Dataset, DataLoader for data pipelines
├── torch.cuda            # GPU operations and memory management
├── torch.jit             # TorchScript for model compilation
├── torch.onnx            # ONNX model export
├── torch.distributed     # Distributed training utilities
└── torch.compile         # Graph-mode compiler (PyTorch 2.0+)
```

### 5.1 Relationship Between Components

```
         Data                      Model                    Training
    ┌─────────────┐          ┌──────────────┐         ┌──────────────┐
    │  Dataset    │          │  nn.Module   │         │  Loss fn     │
    │  DataLoader │  ──▶     │  Parameters  │  ──▶    │  Optimizer   │
    │  Transforms │          │  forward()   │         │  backward()  │
    └─────────────┘          └──────────────┘         └──────────────┘
```

You will learn each of these in dedicated lessons.

---

## 6. Eager Mode vs Graph Mode

### 6.1 Eager Mode (Default)

In eager mode, operations execute immediately:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2        # executes NOW, result is available immediately
z = y + 1        # executes NOW
print(z)          # tensor([3., 5., 7.])
```

This makes debugging straightforward -- you can use `print()`, `breakpoint()`, or any Python debugger.

### 6.2 Graph Mode (torch.compile)

PyTorch 2.0 introduced `torch.compile()` for performance optimization:

```python
@torch.compile
def optimized_fn(x):
    y = x * 2
    z = y + 1
    return z

# First call compiles; subsequent calls are faster
result = optimized_fn(torch.randn(1000))
```

`torch.compile` traces the function, applies optimizations (operator fusion, memory planning), and generates optimized kernels -- all while keeping the code in eager-looking Python.

---

## 7. Hello World: Linear Regression in PyTorch

Let's put it all together with a minimal example:

```python
import torch

# 1. Generate synthetic data: y = 2x + 1 + noise
torch.manual_seed(42)
X = torch.rand(100, 1) * 10                # 100 samples, 1 feature
y = 2 * X + 1 + torch.randn(100, 1) * 0.5  # true: slope=2, intercept=1

# 2. Initialize parameters
w = torch.randn(1, requires_grad=True)   # weight (slope)
b = torch.zeros(1, requires_grad=True)   # bias (intercept)

# 3. Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    y_pred = X * w + b

    # Compute loss (MSE)
    loss = ((y_pred - y) ** 2).mean()

    # Backward pass (compute gradients)
    loss.backward()

    # Update parameters (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
              f"w: {w.item():.4f} | b: {b.item():.4f}")

print(f"\nLearned: y = {w.item():.2f}x + {b.item():.2f}")
print(f"True:    y = 2.00x + 1.00")
```

This example demonstrates the four key PyTorch operations you'll master:
1. **Tensor creation** -- `torch.rand`, `torch.randn`
2. **Automatic differentiation** -- `requires_grad=True`, `loss.backward()`
3. **Gradient-based optimization** -- manual SGD with `w -= lr * w.grad`
4. **Gradient management** -- `torch.no_grad()`, `grad.zero_()`

---

## 8. PyTorch Community and Resources

### 8.1 Official Resources

| Resource | URL |
|----------|-----|
| Documentation | https://pytorch.org/docs/stable/ |
| Tutorials | https://pytorch.org/tutorials/ |
| GitHub | https://github.com/pytorch/pytorch |
| Discussion Forum | https://discuss.pytorch.org/ |
| Blog | https://pytorch.org/blog/ |

### 8.2 Learning Strategy

1. **This course**: Complete all 14 lessons for comprehensive PyTorch fluency
2. **Official tutorials**: Cross-reference with PyTorch's own tutorial series
3. **Source code**: Read PyTorch source when you want to understand internals
4. **Practice**: Build small projects after each lesson to solidify understanding

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| PyTorch | Open-source ML framework by Meta AI, dominant in research |
| Tensor | Multi-dimensional array, the fundamental data structure |
| Eager mode | Operations execute immediately (default, great for debugging) |
| Dynamic graphs | Computational graph built on-the-fly during forward pass |
| NumPy bridge | Zero-copy conversion between NumPy arrays and CPU tensors |
| torch.compile | Optional graph-mode compilation for performance (PyTorch 2.0+) |

---

**Next**: [Tensors](./02_Tensors.md) -- Deep dive into tensor creation, dtypes, devices, and memory layout.

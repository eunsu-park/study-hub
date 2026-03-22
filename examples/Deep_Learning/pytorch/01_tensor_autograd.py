"""
01. Tensors and Autograd - PyTorch Version

Learn PyTorch's core features: tensor operations and automatic differentiation.
Compare with the NumPy version (examples/numpy/01_tensor_basics.py).
"""

import torch
import numpy as np

print("=" * 60)
print("PyTorch Tensors and Autograd")
print("=" * 60)


# ============================================
# 1. Creating Tensors
# ============================================
print("\n[1] Creating Tensors")
print("-" * 40)

# From a list
tensor1 = torch.tensor([1, 2, 3, 4])
print(f"List -> Tensor: {tensor1}")
print(f"  shape: {tensor1.shape}, dtype: {tensor1.dtype}")

# Special tensors
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand = torch.randn(2, 3)  # Standard normal distribution
arange = torch.arange(0, 10, 2)

print(f"zeros(3,4): shape {zeros.shape}")
print(f"randn(2,3):\n{rand}")

# Specifying dtype
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"float32 tensor: {float_tensor}")


# ============================================
# 2. NumPy Conversion
# ============================================
print("\n[2] NumPy Conversion")
print("-" * 40)

# NumPy -> PyTorch
np_arr = np.array([1.0, 2.0, 3.0])
torch_from_np = torch.from_numpy(np_arr)
print(f"NumPy -> PyTorch: {torch_from_np}")

# Note: memory is shared
np_arr[0] = 100
print(f"After modifying NumPy, PyTorch: {torch_from_np}")  # Changes together

# PyTorch -> NumPy
pt_tensor = torch.tensor([4.0, 5.0, 6.0])
np_from_torch = pt_tensor.numpy()
print(f"PyTorch -> NumPy: {np_from_torch}")


# ============================================
# 3. Tensor Operations
# ============================================
print("\n[3] Tensor Operations")
print("-" * 40)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Element-wise operations
print(f"a + b:\n{a + b}")
print(f"a * b (element-wise):\n{a * b}")

# Matrix multiplication
print(f"a @ b (matrix multiply):\n{a @ b}")
print(f"torch.matmul(a, b):\n{torch.matmul(a, b)}")

# Statistics
print(f"a.sum(): {a.sum()}")
print(f"a.mean(): {a.mean()}")
print(f"a.max(): {a.max()}")


# ============================================
# 4. Broadcasting
# ============================================
print("\n[4] Broadcasting")
print("-" * 40)

x = torch.tensor([[1], [2], [3]])  # (3, 1)
y = torch.tensor([10, 20, 30])     # (3,)

result = x + y  # Automatically expanded to (3, 3)
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"x + y shape: {result.shape}")
print(f"x + y:\n{result}")


# ============================================
# 5. Automatic Differentiation (Autograd) Basics
# ============================================
print("\n[5] Automatic Differentiation (Autograd)")
print("-" * 40)

# Enable gradient tracking with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)
print(f"x: {x}, requires_grad: {x.requires_grad}")

# Forward pass
y = x ** 2 + 3 * x + 1  # y = x^2 + 3x + 1
print(f"y = x^2 + 3x + 1 = {y.item()}")

# Backward pass
y.backward()

# Check gradient (dy/dx = 2x + 3 = 2*2 + 3 = 7)
print(f"dy/dx at x=2: {x.grad.item()}")
print("Verification: dy/dx = 2x + 3 = 2*2 + 3 = 7 ✓")


# ============================================
# 6. Automatic Differentiation of Complex Functions
# ============================================
print("\n[6] Differentiating Complex Functions")
print("-" * 40)

# f(x) = x^3 + 2x^2 - 5x + 3
# f'(x) = 3x^2 + 4x - 5
# f'(2) = 12 + 8 - 5 = 15

x = torch.tensor([2.0], requires_grad=True)
f = x**3 + 2*x**2 - 5*x + 3

f.backward()
print(f"f(x) = x^3 + 2x^2 - 5x + 3")
print(f"f(2) = {f.item()}")
print(f"f'(2) = {x.grad.item()}")
print("Verification: f'(x) = 3x^2 + 4x - 5 = 12 + 8 - 5 = 15 ✓")


# ============================================
# 7. Multivariable Function Differentiation (Gradient)
# ============================================
print("\n[7] Multivariable Function Differentiation")
print("-" * 40)

# f(x, y) = x^2 + y^2 + xy
# df/dx = 2x + y
# df/dy = 2y + x

x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)

f = x**2 + y**2 + x*y
f.backward()

print(f"f(x, y) = x^2 + y^2 + xy")
print(f"f(3, 4) = {f.item()}")
print(f"df/dx at (3,4) = {x.grad.item()}")  # 2*3 + 4 = 10
print(f"df/dy at (3,4) = {y.grad.item()}")  # 2*4 + 3 = 11


# ============================================
# 8. Gradient Reset
# ============================================
print("\n[8] Gradient Reset")
print("-" * 40)

x = torch.tensor([1.0], requires_grad=True)

# First backward pass
y1 = x * 2
y1.backward()
print(f"First grad: {x.grad}")

# Gradients accumulate!
y2 = x * 3
y2.backward()
print(f"Accumulated grad: {x.grad}")  # 2 + 3 = 5

# After reset
x.grad.zero_()  # Important!
y3 = x * 4
y3.backward()
print(f"Grad after reset: {x.grad}")


# ============================================
# 9. GPU Operations
# ============================================
print("\n[9] GPU Operations")
print("-" * 40)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Move tensor to GPU
    x_cpu = torch.randn(1000, 1000)
    x_gpu = x_cpu.to(device)

    # Compute on GPU
    y_gpu = x_gpu @ x_gpu

    # Move result back to CPU
    y_cpu = y_gpu.cpu()
    print(f"GPU matrix multiplication done: {y_cpu.shape}")
else:
    print("GPU unavailable, running in CPU mode")


# ============================================
# 10. no_grad Context
# ============================================
print("\n[10] no_grad Context")
print("-" * 40)

x = torch.tensor([1.0], requires_grad=True)

# Normal operation (gradient tracking)
y = x * 2
print(f"Normal operation: requires_grad = {y.requires_grad}")

# Inside no_grad (no gradient tracking)
with torch.no_grad():
    z = x * 2
    print(f"Inside no_grad: requires_grad = {z.requires_grad}")

# Detach
w = x.detach() * 2
print(f"After detach: requires_grad = {w.requires_grad}")


print("\n" + "=" * 60)
print("PyTorch Tensors and Autograd complete!")
print("Compare with NumPy version: examples/numpy/01_tensor_basics.py")
print("=" * 60)

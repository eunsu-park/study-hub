"""
01. Tensor Basics - NumPy Version

Implements tensor operations and manual differentiation with NumPy.
Compare with the PyTorch version (examples/pytorch/01_tensor_autograd.py).

Key Differences:
- NumPy: No automatic differentiation, must compute derivatives manually
- PyTorch: Automatic differentiation with autograd
"""

import numpy as np

print("=" * 60)
print("NumPy Tensor Basics and Manual Differentiation")
print("=" * 60)


# ============================================
# 1. Array Creation (Tensors)
# ============================================
print("\n[1] Array Creation")
print("-" * 40)

# Create from list
arr1 = np.array([1, 2, 3, 4])
print(f"List -> Array: {arr1}")
print(f"  shape: {arr1.shape}, dtype: {arr1.dtype}")

# Special arrays
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
rand = np.random.randn(2, 3)  # Standard normal distribution
arange = np.arange(0, 10, 2)

print(f"zeros(3,4): shape {zeros.shape}")
print(f"randn(2,3):\n{rand}")

# Specify dtype
float_arr = np.array([1, 2, 3], dtype=np.float32)
print(f"float32 array: {float_arr}")


# ============================================
# 2. Array Operations
# ============================================
print("\n[2] Array Operations")
print("-" * 40)

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5, 6], [7, 8]], dtype=np.float32)

# Element-wise operations
print(f"a + b:\n{a + b}")
print(f"a * b (element-wise):\n{a * b}")

# Matrix multiplication
print(f"a @ b (matrix multiplication):\n{a @ b}")
print(f"np.dot(a, b):\n{np.dot(a, b)}")

# Statistics
print(f"a.sum(): {a.sum()}")
print(f"a.mean(): {a.mean()}")
print(f"a.max(): {a.max()}")


# ============================================
# 3. Broadcasting
# ============================================
print("\n[3] Broadcasting")
print("-" * 40)

x = np.array([[1], [2], [3]])  # (3, 1)
y = np.array([10, 20, 30])     # (3,)

result = x + y  # Automatically broadcast to (3, 3)
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"x + y shape: {result.shape}")
print(f"x + y:\n{result}")


# ============================================
# 4. Manual Differentiation - Basics
# ============================================
print("\n[4] Manual Differentiation - Basics")
print("-" * 40)

# y = x² + 3x + 1
# dy/dx = 2x + 3

def f1(x):
    """Forward pass: y = x² + 3x + 1"""
    return x**2 + 3*x + 1

def df1(x):
    """Manual derivative: dy/dx = 2x + 3"""
    return 2*x + 3

x = 2.0
print(f"f(x) = x² + 3x + 1")
print(f"f({x}) = {f1(x)}")
print(f"f'({x}) = {df1(x)}")  # 2*2 + 3 = 7
print("Verification: dy/dx = 2x + 3 = 2*2 + 3 = 7 ✓")


# ============================================
# 5. Manual Differentiation - Complex Function
# ============================================
print("\n[5] Manual Differentiation - Complex Function")
print("-" * 40)

# f(x) = x³ + 2x² - 5x + 3
# f'(x) = 3x² + 4x - 5

def f2(x):
    """Forward pass"""
    return x**3 + 2*x**2 - 5*x + 3

def df2(x):
    """Manual derivative"""
    return 3*x**2 + 4*x - 5

x = 2.0
print(f"f(x) = x³ + 2x² - 5x + 3")
print(f"f({x}) = {f2(x)}")
print(f"f'({x}) = {df2(x)}")  # 3*4 + 4*2 - 5 = 15
print("Verification: f'(x) = 3x² + 4x - 5 = 12 + 8 - 5 = 15 ✓")


# ============================================
# 6. Manual Differentiation - Multivariable Function
# ============================================
print("\n[6] Manual Differentiation - Multivariable Function")
print("-" * 40)

# f(x, y) = x² + y² + xy
# ∂f/∂x = 2x + y
# ∂f/∂y = 2y + x

def f3(x, y):
    """Forward pass"""
    return x**2 + y**2 + x*y

def df3_dx(x, y):
    """Partial derivative ∂f/∂x"""
    return 2*x + y

def df3_dy(x, y):
    """Partial derivative ∂f/∂y"""
    return 2*y + x

x, y = 3.0, 4.0
print(f"f(x, y) = x² + y² + xy")
print(f"f({x}, {y}) = {f3(x, y)}")
print(f"∂f/∂x at ({x},{y}) = {df3_dx(x, y)}")  # 2*3 + 4 = 10
print(f"∂f/∂y at ({x},{y}) = {df3_dy(x, y)}")  # 2*4 + 3 = 11


# ============================================
# 7. Numerical Differentiation
# ============================================
print("\n[7] Numerical Differentiation")
print("-" * 40)

def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical derivative using central difference method
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# Test with f(x) = x³ + 2x² - 5x + 3
x = 2.0
numerical_grad = numerical_gradient(f2, x)
analytical_grad = df2(x)

print(f"Analytical derivative: {analytical_grad}")
print(f"Numerical derivative:  {numerical_grad:.10f}")
print(f"Error:                 {abs(numerical_grad - analytical_grad):.2e}")


# ============================================
# 8. Differentiation for Vector Inputs
# ============================================
print("\n[8] Vector Input Differentiation")
print("-" * 40)

def f_vec(x):
    """f(x) = sum(x²) = x₁² + x₂² + x₃²"""
    return np.sum(x**2)

def df_vec(x):
    """∇f = [2x₁, 2x₂, 2x₃]"""
    return 2 * x

x = np.array([1.0, 2.0, 3.0])
print(f"f(x) = sum(x²)")
print(f"x = {x}")
print(f"f(x) = {f_vec(x)}")
print(f"∇f(x) = {df_vec(x)}")


# ============================================
# 9. Chain Rule Example
# ============================================
print("\n[9] Chain Rule")
print("-" * 40)

# h(x) = f(g(x))
# g(x) = x²
# f(u) = sin(u)
# h(x) = sin(x²)
# dh/dx = df/du * dg/dx = cos(x²) * 2x

def g(x):
    return x**2

def f(u):
    return np.sin(u)

def h(x):
    return f(g(x))  # h(x) = sin(x²)

def dh_dx(x):
    """Chain rule: dh/dx = cos(x²) * 2x"""
    return np.cos(x**2) * (2*x)

x = 1.0
print(f"g(x) = x², f(u) = sin(u)")
print(f"h(x) = f(g(x)) = sin(x²)")
print(f"h({x}) = {h(x):.6f}")
print(f"dh/dx at x={x}: {dh_dx(x):.6f}")
print("Chain rule: dh/dx = cos(x²) * 2x")


# ============================================
# 10. Loss Function and Derivative Example
# ============================================
print("\n[10] Loss Function and Derivative")
print("-" * 40)

def mse_loss(y_pred, y_true):
    """MSE: L = (1/n) * Σ(y_pred - y_true)²"""
    return np.mean((y_pred - y_true)**2)

def mse_gradient(y_pred, y_true):
    """∂L/∂y_pred = (2/n) * (y_pred - y_true)"""
    n = len(y_pred)
    return (2/n) * (y_pred - y_true)

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.2, 2.8])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MSE Loss: {loss:.4f}")
print(f"Gradient: {grad}")


# ============================================
# NumPy vs PyTorch Summary
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch Comparison")
print("=" * 60)

comparison = """
| Feature        | NumPy                | PyTorch                    |
|----------------|----------------------|----------------------------|
| Array creation | np.array()           | torch.tensor()             |
| Derivatives    | Must implement manually | .backward() auto-computes |
| GPU            | Not supported        | .to('cuda') supported      |
| Strengths      | Understand algorithm principles | Fast development, auto-diff |
"""
print(comparison)

print("NumPy Tensor Basics and Manual Differentiation complete!")
print("Compare with PyTorch version: examples/pytorch/01_tensor_autograd.py")
print("=" * 60)

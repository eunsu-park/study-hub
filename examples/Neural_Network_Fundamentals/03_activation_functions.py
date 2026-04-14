"""
03. Activation Functions
=========================
Implements and compares all major activation functions,
their derivatives, and demonstrates the vanishing gradient problem.

Key Concepts:
  - Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, GELU, Softmax
  - Derivative computation
  - Vanishing gradient demonstration
"""

import numpy as np


# ============================================================
# 1. Activation Function Implementations
# ============================================================
print("=" * 60)
print("1. Activation Functions")
print("=" * 60)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_fn(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(z))

def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)


# Test each function
z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\nInput z: {z}")
print(f"Sigmoid:    {sigmoid(z)}")
print(f"Tanh:       {tanh_fn(z)}")
print(f"ReLU:       {relu(z)}")
print(f"Leaky ReLU: {leaky_relu(z)}")
print(f"ELU:        {elu(z)}")
print(f"GELU:       {gelu(z)}")


# ============================================================
# 2. Derivatives
# ============================================================
print("\n" + "=" * 60)
print("2. Activation Function Derivatives")
print("=" * 60)

print(f"\nAt z = {z}:")
print(f"Sigmoid':    {sigmoid_derivative(z)}")
print(f"Tanh':       {tanh_derivative(z)}")
print(f"ReLU':       {relu_derivative(z)}")
print(f"Leaky ReLU': {leaky_relu_derivative(z)}")
print(f"ELU':        {elu_derivative(z)}")

# Verify sigmoid max derivative = 0.25 at z=0
print(f"\nSigmoid'(0) = {sigmoid_derivative(np.array([0.0]))[0]:.4f} (should be 0.25)")


# ============================================================
# 3. Numerical Derivative Verification
# ============================================================
print("\n" + "=" * 60)
print("3. Numerical Derivative Verification")
print("=" * 60)

eps = 1e-7
z_test = np.array([0.5])

funcs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh_fn, tanh_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
]

for name, fn, deriv_fn in funcs:
    numerical = (fn(z_test + eps) - fn(z_test - eps)) / (2 * eps)
    analytical = deriv_fn(z_test)
    error = abs(numerical - analytical)[0]
    print(f"  {name:12s}: analytical={analytical[0]:.6f}, "
          f"numerical={numerical[0]:.6f}, error={error:.2e}")


# ============================================================
# 4. Softmax
# ============================================================
print("\n" + "=" * 60)
print("4. Softmax Function")
print("=" * 60)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"\nLogits:  {logits}")
print(f"Softmax: {probs}")
print(f"Sum:     {probs.sum():.6f} (should be 1.0)")


# ============================================================
# 5. Vanishing Gradient Demonstration
# ============================================================
print("\n" + "=" * 60)
print("5. Vanishing Gradient Problem")
print("=" * 60)

print("\nMultiplying sigmoid derivatives through L layers:")
for n_layers in [1, 4, 10, 20]:
    grad_product = sigmoid_derivative(np.array([0.0]))[0] ** n_layers
    print(f"  {n_layers:2d} layers: gradient magnitude = {grad_product:.2e}")

print("\nReLU gradient through layers (z > 0):")
for n_layers in [1, 4, 10, 20]:
    grad_product = 1.0 ** n_layers
    print(f"  {n_layers:2d} layers: gradient magnitude = {grad_product:.2e}")
print("  ReLU maintains gradient = 1 regardless of depth!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

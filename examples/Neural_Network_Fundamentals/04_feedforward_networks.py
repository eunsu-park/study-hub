"""
04. Feedforward Networks
=========================
Implements the forward pass of a Multi-Layer Perceptron (MLP)
using NumPy matrix operations.

Key Concepts:
  - MLP architecture and matrix dimensions
  - Forward pass with ReLU and Softmax
  - Batch processing
  - Parameter counting
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. Network Initialization
# ============================================================
print("=" * 60)
print("1. MLP Initialization")
print("=" * 60)


def initialize_network(layer_sizes):
    """Initialize weights (He) and biases for an MLP."""
    params = []
    for i in range(1, len(layer_sizes)):
        fan_in = layer_sizes[i - 1]
        fan_out = layer_sizes[i]
        W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        b = np.zeros((fan_out, 1))
        params.append((W, b))
    return params


def count_parameters(params):
    """Count total trainable parameters."""
    total = 0
    for i, (W, b) in enumerate(params):
        n = W.size + b.size
        print(f"  Layer {i+1}: W{W.shape} + b{b.shape} = {n} params")
        total += n
    print(f"  Total: {total}")
    return total


layer_sizes = [4, 8, 6, 3]
params = initialize_network(layer_sizes)
print(f"\nNetwork architecture: {layer_sizes}")
count_parameters(params)


# ============================================================
# 2. Forward Pass
# ============================================================
print("\n" + "=" * 60)
print("2. Forward Pass")
print("=" * 60)


def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward(X, params):
    """Forward pass through the network."""
    cache = []
    a = X
    for i, (W, b) in enumerate(params):
        z = W @ a + b
        if i < len(params) - 1:
            a = relu(z)
        else:
            a = softmax(z)
        cache.append((z, a))
    return a, cache


# Single sample
x = np.random.randn(4, 1)
output, cache = forward(x, params)
print(f"\nSingle sample:")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Output: {output.ravel()}")
print(f"  Sum:    {output.sum():.6f}")

# Batch of 32 samples
X_batch = np.random.randn(4, 32)
output_batch, _ = forward(X_batch, params)
print(f"\nBatch processing:")
print(f"  Input shape:  {X_batch.shape}")
print(f"  Output shape: {output_batch.shape}")
print(f"  All rows sum to 1: {np.allclose(output_batch.sum(axis=0), 1.0)}")


# ============================================================
# 3. Different Architectures
# ============================================================
print("\n" + "=" * 60)
print("3. Architecture Comparison")
print("=" * 60)

architectures = {
    "Shallow wide": [4, 256, 3],
    "Deep narrow":  [4, 32, 32, 32, 3],
    "Balanced":     [4, 64, 32, 3],
    "MNIST-like":   [784, 256, 128, 10],
}

for name, layers in architectures.items():
    p = initialize_network(layers)
    total = sum(W.size + b.size for W, b in p)
    depth = len(layers) - 1
    print(f"  {name:15s}: {str(layers):25s} -> {total:6d} params, depth={depth}")


# ============================================================
# 4. Common MLP Architectures
# ============================================================
print("\n" + "=" * 60)
print("4. Task-Specific Output Layers")
print("=" * 60)

# Binary classification
x = np.random.randn(10, 1)
W_out = np.random.randn(1, 10) * 0.1
b_out = np.zeros((1, 1))
z_out = W_out @ x + b_out
sigmoid_out = 1 / (1 + np.exp(-z_out))
print(f"\nBinary classification: P(y=1) = {sigmoid_out[0,0]:.4f}")

# Regression (no activation)
print(f"Regression output: y = {z_out[0,0]:.4f} (identity activation)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

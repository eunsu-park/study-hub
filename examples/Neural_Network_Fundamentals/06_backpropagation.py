"""
06. Backpropagation
====================
Implements forward pass, backward pass, and gradient checking
for a 2-layer MLP from scratch.

Key Concepts:
  - Chain rule through layers
  - Weight and bias gradient computation
  - Numerical gradient verification
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. Helper Functions
# ============================================================
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[1]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / m


# ============================================================
# 2. Forward and Backward Pass
# ============================================================
print("=" * 60)
print("Backpropagation: 2-Layer MLP")
print("=" * 60)


def forward_pass(X, params):
    """Forward pass, returns output and cache."""
    caches = [{'a': X}]
    a = X
    for i, (W, b) in enumerate(params):
        z = W @ a + b
        if i < len(params) - 1:
            a = relu(z)
        else:
            a = softmax(z)
        caches.append({'z': z, 'a': a})
    return a, caches


def backward_pass(Y, params, caches):
    """Backward pass, returns gradients for each layer."""
    m = Y.shape[1]
    L = len(params)
    grads = []

    dz = caches[L]['a'] - Y  # softmax + CCE combined

    for l in range(L, 0, -1):
        a_prev = caches[l - 1]['a']
        dW = (1 / m) * dz @ a_prev.T
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        grads.insert(0, (dW, db))

        if l > 1:
            da = params[l - 1][0].T @ dz
            dz = da * relu_derivative(caches[l - 1]['z'])

    return grads


# Setup network [3, 4, 2]
W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros((4, 1))
W2 = np.random.randn(2, 4) * 0.1
b2 = np.zeros((2, 1))
params = [(W1, b1), (W2, b2)]

# Data: 5 samples, 3 features, 2 classes
X = np.random.randn(3, 5)
Y = np.array([[1, 0, 1, 0, 1],
              [0, 1, 0, 1, 0]], dtype=float)

# Forward
y_pred, caches = forward_pass(X, params)
loss = cross_entropy_loss(y_pred, Y)
print(f"\nForward pass:")
print(f"  Input shape:  {X.shape}")
print(f"  Output shape: {y_pred.shape}")
print(f"  Loss: {loss:.6f}")

# Backward
grads = backward_pass(Y, params, caches)
print(f"\nBackward pass:")
for i, (dW, db) in enumerate(grads):
    print(f"  Layer {i+1}: dW shape {dW.shape}, db shape {db.shape}")


# ============================================================
# 3. Gradient Checking
# ============================================================
print("\n" + "=" * 60)
print("Gradient Checking")
print("=" * 60)

epsilon = 1e-7
print(f"\nUsing epsilon = {epsilon}")

for layer_idx in range(len(params)):
    W, b = params[layer_idx]
    dW_analytical = grads[layer_idx][0]

    # Check 3 random elements
    max_error = 0
    for _ in range(5):
        i = np.random.randint(W.shape[0])
        j = np.random.randint(W.shape[1])

        original = W[i, j]

        W[i, j] = original + epsilon
        y_plus, _ = forward_pass(X, params)
        loss_plus = cross_entropy_loss(y_plus, Y)

        W[i, j] = original - epsilon
        y_minus, _ = forward_pass(X, params)
        loss_minus = cross_entropy_loss(y_minus, Y)

        W[i, j] = original

        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        analytical = dW_analytical[i, j]
        rel_error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-15)
        max_error = max(max_error, rel_error)

    status = "PASS" if max_error < 1e-5 else "FAIL"
    print(f"  Layer {layer_idx+1}: max relative error = {max_error:.2e} [{status}]")


# ============================================================
# 4. Training Step
# ============================================================
print("\n" + "=" * 60)
print("Single Training Step")
print("=" * 60)

lr = 0.1
print(f"\nBefore update: Loss = {loss:.6f}")

# Update weights
for i in range(len(params)):
    W, b = params[i]
    dW, db = grads[i]
    params[i] = (W - lr * dW, b - lr * db)

y_pred_new, _ = forward_pass(X, params)
loss_new = cross_entropy_loss(y_pred_new, Y)
print(f"After update:  Loss = {loss_new:.6f}")
print(f"Loss decreased: {loss_new < loss}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

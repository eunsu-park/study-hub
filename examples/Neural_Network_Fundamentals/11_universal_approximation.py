"""
11. Universal Approximation Theorem
=====================================
Demonstrates the UAT by approximating sin(x) using bump functions
and training a 1-hidden-layer network.

Key Concepts:
  - Bump function construction
  - Width vs approximation quality
  - Practical training of a shallow network
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. Bump Function Construction
# ============================================================
print("=" * 60)
print("1. Approximating sin(x) with Bump Functions")
print("=" * 60)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def bump(x, center, width, height, sharpness=50):
    left = sigmoid(sharpness * (x - (center - width / 2)))
    right = sigmoid(sharpness * (x - (center + width / 2)))
    return height * (left - right)


x = np.linspace(0, 2 * np.pi, 1000)
target = np.sin(x)

for N in [5, 10, 20, 50]:
    approx = np.zeros_like(x)
    for i in range(N):
        center = (i + 0.5) * 2 * np.pi / N
        width = 2 * np.pi / N
        height = np.sin(center)
        approx += bump(x, center, width, height)
    error = np.max(np.abs(target - approx))
    print(f"  {N:3d} bumps: max error = {error:.6f}")


# ============================================================
# 2. Training a 1-Hidden-Layer Network
# ============================================================
print("\n" + "=" * 60)
print("2. Training a 1-Hidden-Layer Network on sin(x)")
print("=" * 60)


def relu(z):
    return np.maximum(0, z)

x_train = np.linspace(-np.pi, np.pi, 200).reshape(1, -1)
y_train = np.sin(x_train)

for N_neurons in [10, 25, 50, 100]:
    np.random.seed(42)
    W1 = np.random.randn(N_neurons, 1) * 0.5
    b1 = np.random.randn(N_neurons, 1) * 0.5
    W2 = np.random.randn(1, N_neurons) * 0.1
    b2 = np.zeros((1, 1))

    lr = 0.001
    for epoch in range(3000):
        z1 = W1 @ x_train + b1
        a1 = relu(z1)
        y_pred = W2 @ a1 + b2

        loss = np.mean((y_pred - y_train) ** 2)

        dy = 2 * (y_pred - y_train) / x_train.shape[1]
        dW2 = dy @ a1.T
        db2 = np.sum(dy, axis=1, keepdims=True)
        da1 = W2.T @ dy
        dz1 = da1 * (z1 > 0)
        dW1 = dz1 @ x_train.T
        db1 = np.sum(dz1, axis=1, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    print(f"  {N_neurons:3d} neurons: final MSE = {loss:.6f}")


# ============================================================
# 3. Approximating x^2
# ============================================================
print("\n" + "=" * 60)
print("3. Approximating x^2 on [0, 1]")
print("=" * 60)

x_sq = np.linspace(0, 1, 200).reshape(1, -1)
y_sq = x_sq ** 2

np.random.seed(42)
N = 50
W1 = np.random.randn(N, 1) * 1.0
b1 = np.random.randn(N, 1) * 0.5
W2 = np.random.randn(1, N) * 0.1
b2 = np.zeros((1, 1))

for epoch in range(5000):
    z1 = W1 @ x_sq + b1
    a1 = relu(z1)
    y_pred = W2 @ a1 + b2
    loss = np.mean((y_pred - y_sq) ** 2)

    dy = 2 * (y_pred - y_sq) / x_sq.shape[1]
    dW2 = dy @ a1.T
    db2 = np.sum(dy, axis=1, keepdims=True)
    da1 = W2.T @ dy
    dz1 = da1 * (z1 > 0)
    dW1 = dz1 @ x_sq.T
    db1 = np.sum(dz1, axis=1, keepdims=True)

    W2 -= 0.001 * dW2
    b2 -= 0.001 * db2
    W1 -= 0.001 * dW1
    b1 -= 0.001 * db1

print(f"  50 neurons, 5000 epochs: MSE = {loss:.8f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

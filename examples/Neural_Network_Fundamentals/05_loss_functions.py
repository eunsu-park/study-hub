"""
05. Loss Functions
===================
Implements and compares MSE, Binary Cross-Entropy, Categorical
Cross-Entropy, and Hinge loss with gradient computation.

Key Concepts:
  - MSE for regression
  - Cross-entropy for classification
  - Why CE beats MSE for classification
  - Numerical stability tricks
"""

import numpy as np


# ============================================================
# 1. Mean Squared Error
# ============================================================
print("=" * 60)
print("1. Mean Squared Error (MSE)")
print("=" * 60)


def mse_loss(y_pred, y_true):
    return 0.5 * np.mean((y_pred - y_true) ** 2)

def mse_gradient(y_pred, y_true):
    return (y_pred - y_true) / y_true.size


y_true = np.array([1.0, 0.0, 1.0, 0.5])
y_pred = np.array([0.8, 0.2, 0.9, 0.6])
print(f"\ny_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MSE Loss: {mse_loss(y_pred, y_true):.6f}")
print(f"Gradient: {mse_gradient(y_pred, y_true)}")


# ============================================================
# 2. Binary Cross-Entropy
# ============================================================
print("\n" + "=" * 60)
print("2. Binary Cross-Entropy (BCE)")
print("=" * 60)


def binary_cross_entropy(y_pred, y_true, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_gradient(y_pred, y_true, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / y_true.size


y_true_b = np.array([1, 0, 1, 1], dtype=float)
y_pred_b = np.array([0.9, 0.1, 0.8, 0.7])
print(f"\ny_true: {y_true_b}")
print(f"y_pred: {y_pred_b}")
print(f"BCE Loss: {binary_cross_entropy(y_pred_b, y_true_b):.6f}")


# ============================================================
# 3. Categorical Cross-Entropy
# ============================================================
print("\n" + "=" * 60)
print("3. Categorical Cross-Entropy (CCE)")
print("=" * 60)


def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def categorical_cross_entropy(y_pred, y_true, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1.0)
    return -np.sum(y_true * np.log(y_pred))


logits = np.array([2.1, 0.5, -1.2])
y_true_c = np.array([1, 0, 0])  # one-hot: class 0
y_pred_c = softmax(logits)

print(f"\nLogits:  {logits}")
print(f"Softmax: {y_pred_c}")
print(f"y_true:  {y_true_c}")
print(f"CCE Loss: {categorical_cross_entropy(y_pred_c, y_true_c):.6f}")
print(f"Gradient (softmax+CCE): {y_pred_c - y_true_c}")


# ============================================================
# 4. Why Cross-Entropy Beats MSE for Classification
# ============================================================
print("\n" + "=" * 60)
print("4. CE vs MSE Gradient Comparison")
print("=" * 60)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Wrong prediction: y=1 but z=-5 (sigmoid ≈ 0.007)
z_wrong = -5.0
y_target = 1.0
s = sigmoid(z_wrong)

# MSE gradient through sigmoid
mse_grad = (s - y_target) * s * (1 - s)
# BCE gradient through sigmoid
bce_grad = s - y_target

print(f"\nWrong prediction: y=1, z={z_wrong}, sigmoid(z)={s:.6f}")
print(f"  MSE gradient at z: {mse_grad:.6f}")
print(f"  BCE gradient at z: {bce_grad:.6f}")
print(f"  BCE/MSE ratio:     {abs(bce_grad/mse_grad):.0f}x larger")
print("  BCE learns much faster from large errors!")


# ============================================================
# 5. Hinge Loss
# ============================================================
print("\n" + "=" * 60)
print("5. Hinge Loss")
print("=" * 60)


def hinge_loss(y_pred, y_true):
    """Hinge loss for y ∈ {-1, +1}."""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


y_true_h = np.array([1, -1, 1, -1], dtype=float)
y_pred_h = np.array([0.5, -0.8, 1.5, 0.3])
print(f"\ny_true: {y_true_h}")
print(f"y_pred: {y_pred_h}")
print(f"Hinge Loss: {hinge_loss(y_pred_h, y_true_h):.6f}")


# ============================================================
# 6. Numerical Stability
# ============================================================
print("\n" + "=" * 60)
print("6. Log-Sum-Exp Trick")
print("=" * 60)


def log_sum_exp(z):
    z_max = np.max(z)
    return z_max + np.log(np.sum(np.exp(z - z_max)))


z_large = np.array([1000, 1001, 999])
print(f"\nz = {z_large}")
print(f"Stable log-sum-exp: {log_sum_exp(z_large):.4f}")
print("(Naive np.log(np.sum(np.exp(z))) would overflow!)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

"""
08. Weight Initialization
==========================
Demonstrates Xavier and He initialization and their effect
on activation statistics through deep networks.

Key Concepts:
  - Symmetry breaking problem
  - Xavier/Glorot initialization
  - He/Kaiming initialization
  - Variance propagation analysis
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. Initialization Schemes
# ============================================================
print("=" * 60)
print("Weight Initialization Schemes")
print("=" * 60)


def random_init(fan_in, fan_out):
    return np.random.randn(fan_out, fan_in) * 1.0

def small_init(fan_in, fan_out):
    return np.random.randn(fan_out, fan_in) * 0.01

def xavier_normal(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_out, fan_in) * std

def he_normal(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_out, fan_in) * std


# ============================================================
# 2. Activation Statistics Through Layers
# ============================================================
print("\n" + "=" * 60)
print("Activation Statistics Through 10 Layers")
print("=" * 60)


def test_initialization(init_fn, name, n_layers=10, n=256, activation='relu'):
    """Track activation statistics through layers."""
    np.random.seed(42)
    x = np.random.randn(n, 1)
    a = x

    print(f"\n--- {name} ({activation}) ---")
    for i in range(n_layers):
        W = init_fn(n, n)
        z = W @ a
        if activation == 'relu':
            a = np.maximum(0, z)
        elif activation == 'tanh':
            a = np.tanh(z)
        bar_len = min(int(a.std() * 20), 50)
        bar = '█' * bar_len if bar_len > 0 else '·'
        print(f"  Layer {i+1:2d}: mean={a.mean():+.4f}, std={a.std():.4f} |{bar}")


# Random init → explodes
test_initialization(random_init, "Random N(0,1)", activation='relu')

# Small init → vanishes
test_initialization(small_init, "Small N(0,0.01)", activation='relu')

# He init → stable for ReLU
test_initialization(he_normal, "He Normal", activation='relu')

# Xavier init → stable for tanh
test_initialization(xavier_normal, "Xavier Normal", activation='tanh')


# ============================================================
# 3. Symmetry Breaking Problem
# ============================================================
print("\n" + "=" * 60)
print("Symmetry Breaking Problem")
print("=" * 60)

# Zero initialization — all neurons compute the same thing
W_zero = np.zeros((4, 3))
x = np.array([[1.0], [0.5], [-1.0]])
z = W_zero @ x
print(f"\nZero init output (all neurons identical):")
print(f"  z = {z.ravel()}")
print("  All neurons produce the same output = 0!")

# Random init — each neuron is different
W_rand = np.random.randn(4, 3) * 0.1
z_rand = W_rand @ x
print(f"\nRandom init output (all neurons different):")
print(f"  z = {z_rand.ravel()}")
print("  Each neuron computes a different value → symmetry broken!")


# ============================================================
# 4. Variance Verification
# ============================================================
print("\n" + "=" * 60)
print("Variance Verification")
print("=" * 60)

fan_in, fan_out = 256, 128

W_he = he_normal(fan_in, fan_out)
W_xavier = xavier_normal(fan_in, fan_out)

print(f"\nHe Normal ({fan_in} → {fan_out}):")
print(f"  Expected std: {np.sqrt(2.0 / fan_in):.6f}")
print(f"  Actual std:   {W_he.std():.6f}")

print(f"\nXavier Normal ({fan_in} → {fan_out}):")
print(f"  Expected std: {np.sqrt(2.0 / (fan_in + fan_out)):.6f}")
print(f"  Actual std:   {W_xavier.std():.6f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

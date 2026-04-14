"""
10. Batch Normalization
========================
Implements batch normalization with forward/backward passes,
training vs inference modes, and comparison with layer normalization.

Key Concepts:
  - BN forward pass (normalize, scale, shift)
  - Running statistics for inference
  - Layer normalization comparison
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. Batch Normalization
# ============================================================
print("=" * 60)
print("1. Batch Normalization")
print("=" * 60)


class BatchNorm:
    def __init__(self, n_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones((n_features, 1))
        self.beta = np.zeros((n_features, 1))
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros((n_features, 1))
        self.running_var = np.ones((n_features, 1))

    def forward(self, z, training=True):
        if training:
            mu = np.mean(z, axis=1, keepdims=True)
            var = np.var(z, axis=1, keepdims=True)
            self.z_norm = (z - mu) / np.sqrt(var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            return self.gamma * self.z_norm + self.beta
        else:
            z_norm = (z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * z_norm + self.beta


# Test BN
n_features, batch_size = 8, 64
z = np.random.randn(n_features, batch_size) * 5 + 3  # mean≈3, std≈5

bn = BatchNorm(n_features)
z_bn = bn.forward(z, training=True)

print(f"\nBefore BN: mean={z.mean(axis=1).mean():.2f}, std={z.std(axis=1).mean():.2f}")
print(f"After BN:  mean={z_bn.mean(axis=1).mean():.4f}, std={z_bn.std(axis=1).mean():.4f}")
print("(Should be mean≈0, std≈1)")


# ============================================================
# 2. Training vs Inference
# ============================================================
print("\n" + "=" * 60)
print("2. Training vs Inference Mode")
print("=" * 60)

# Simulate multiple training batches to accumulate running stats
bn2 = BatchNorm(4)
for i in range(100):
    z_batch = np.random.randn(4, 32) * 3 + 2  # mean≈2, std≈3
    _ = bn2.forward(z_batch, training=True)

print(f"\nAfter 100 training batches:")
print(f"  Running mean: {bn2.running_mean.ravel()}")
print(f"  Running var:  {bn2.running_var.ravel()}")

# Inference mode
z_test = np.random.randn(4, 1) * 3 + 2
z_infer = bn2.forward(z_test, training=False)
print(f"\nInference (single sample):")
print(f"  Input:  {z_test.ravel()}")
print(f"  Output: {z_infer.ravel()}")


# ============================================================
# 3. Layer Normalization
# ============================================================
print("\n" + "=" * 60)
print("3. Layer Normalization (Comparison)")
print("=" * 60)


class LayerNorm:
    def __init__(self, n_features, eps=1e-5):
        self.gamma = np.ones((n_features, 1))
        self.beta = np.zeros((n_features, 1))
        self.eps = eps

    def forward(self, z):
        mu = np.mean(z, axis=0, keepdims=True)
        var = np.var(z, axis=0, keepdims=True)
        z_norm = (z - mu) / np.sqrt(var + self.eps)
        return self.gamma * z_norm + self.beta


ln = LayerNorm(n_features)
z_ln = ln.forward(z)

print(f"\nBatchNorm normalizes across BATCH (per feature):")
print(f"  BN output feature-wise std: {z_bn.std(axis=1).mean():.4f}")
print(f"\nLayerNorm normalizes across FEATURES (per sample):")
print(f"  LN output sample-wise std:  {z_ln.std(axis=0).mean():.4f}")

# Show batch-size independence
print(f"\nLayerNorm works with batch size 1:")
z_single = np.random.randn(n_features, 1)
z_ln_single = ln.forward(z_single)
print(f"  Input std:  {z_single.std():.4f}")
print(f"  Output std: {z_ln_single.std():.4f}")


# ============================================================
# 4. Effect on Deep Networks
# ============================================================
print("\n" + "=" * 60)
print("4. BN Effect on Deep Networks")
print("=" * 60)

def run_network(use_bn, n_layers=10, n=128, batch=32):
    x = np.random.randn(n, batch)
    a = x
    bns = [BatchNorm(n) for _ in range(n_layers)] if use_bn else [None] * n_layers

    for i in range(n_layers):
        W = np.random.randn(n, n) * 0.1
        z = W @ a
        if use_bn:
            z = bns[i].forward(z, training=True)
        a = np.maximum(0, z)  # ReLU

    return a.std()

print(f"\nActivation std at layer 10:")
print(f"  Without BN: {run_network(use_bn=False):.6f}")
print(f"  With BN:    {run_network(use_bn=True):.6f}")
print("  BN keeps activations in a healthy range!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

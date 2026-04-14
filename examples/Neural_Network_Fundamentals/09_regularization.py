"""
09. Regularization
===================
Implements L1/L2 regularization, dropout, and early stopping.

Key Concepts:
  - L2 (weight decay) and L1 (lasso) regularization
  - Inverted dropout
  - Early stopping with patience
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. L2 Regularization
# ============================================================
print("=" * 60)
print("1. L2 Regularization (Weight Decay)")
print("=" * 60)


def l2_penalty(params, lambd):
    reg = sum(np.sum(W**2) for W, b in params)
    return (lambd / 2) * reg

def l2_gradient(W, lambd):
    return lambd * W


W = np.random.randn(4, 3)
lambd = 0.01
print(f"\nL2 penalty (λ={lambd}): {l2_penalty([(W, np.zeros((4,1)))], lambd):.6f}")
print(f"L2 gradient shape: {l2_gradient(W, lambd).shape}")


# ============================================================
# 2. L1 Regularization
# ============================================================
print("\n" + "=" * 60)
print("2. L1 Regularization (Lasso)")
print("=" * 60)


def l1_penalty(params, lambd):
    reg = sum(np.sum(np.abs(W)) for W, b in params)
    return lambd * reg

def l1_gradient(W, lambd):
    return lambd * np.sign(W)


print(f"\nL1 penalty (λ={lambd}): {l1_penalty([(W, np.zeros((4,1)))], lambd):.6f}")

# Show sparsity effect
print("\nL1 pushes small weights to zero:")
w_small = np.array([0.5, 0.01, -0.02, 1.0, -0.005])
for step in range(50):
    w_small -= 0.01 * np.sign(w_small) * lambd
print(f"  After 50 L1 steps: {w_small}")


# ============================================================
# 3. Dropout
# ============================================================
print("\n" + "=" * 60)
print("3. Inverted Dropout")
print("=" * 60)


class Dropout:
    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, a, training=True):
        if not training:
            return a
        self.mask = (np.random.rand(*a.shape) < self.keep_prob).astype(float)
        return a * self.mask / self.keep_prob

    def backward(self, da):
        return da * self.mask / self.keep_prob


# Demonstrate
a = np.ones((10, 1))
dropout = Dropout(keep_prob=0.8)

print(f"\nInput (all 1s): {a.ravel()}")
a_dropped = dropout.forward(a, training=True)
print(f"After dropout:  {a_dropped.ravel()}")
print(f"Fraction kept:  {(a_dropped > 0).mean():.1f}")
print(f"Expected mean:  {a.mean():.1f}")
print(f"Dropout mean:   {a_dropped.mean():.1f} (≈ 1.0 due to scaling)")

# Verify expected value matches
np.random.seed(42)
n_trials = 10000
means = []
for _ in range(n_trials):
    d = Dropout(keep_prob=0.5)
    out = d.forward(np.ones((100, 1)), training=True)
    means.append(out.mean())
print(f"\nExpected value over {n_trials} trials: {np.mean(means):.4f} (should ≈ 1.0)")


# ============================================================
# 4. Early Stopping
# ============================================================
print("\n" + "=" * 60)
print("4. Early Stopping")
print("=" * 60)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# Simulate training with overfitting
es = EarlyStopping(patience=3)
val_losses = [1.0, 0.8, 0.6, 0.5, 0.48, 0.49, 0.51, 0.55, 0.6]

print(f"\nSimulated validation losses:")
for epoch, val_loss in enumerate(val_losses):
    should_stop = es.check(val_loss)
    status = " ← STOP" if should_stop else ""
    print(f"  Epoch {epoch}: val_loss = {val_loss:.2f}{status}")
    if should_stop:
        print(f"  Early stopping triggered! Best loss: {es.best_loss:.2f}")
        break


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

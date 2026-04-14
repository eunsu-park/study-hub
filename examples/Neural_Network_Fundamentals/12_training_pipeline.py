"""
12. Training Pipeline
======================
Demonstrates a complete training pipeline including data prep,
mini-batch training, validation, and hyperparameter search.

Key Concepts:
  - Feature scaling (StandardScaler)
  - Train/val/test split
  - Mini-batch training loop
  - Random hyperparameter search
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. Generate Synthetic Data
# ============================================================
print("=" * 60)
print("1. Data Preparation")
print("=" * 60)


def make_spiral(n_samples=200, n_classes=3):
    X = np.zeros((2, n_samples * n_classes))
    Y = np.zeros((n_classes, n_samples * n_classes))
    for c in range(n_classes):
        idx = range(n_samples * c, n_samples * (c + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(c * 4, (c + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        X[0, idx] = r * np.sin(t)
        X[1, idx] = r * np.cos(t)
        Y[c, idx] = 1
    return X, Y


X, Y = make_spiral()
n = X.shape[1]
perm = np.random.permutation(n)
X, Y = X[:, perm], Y[:, perm]

n_train = int(0.7 * n)
n_val = int(0.15 * n)
X_train, Y_train = X[:, :n_train], Y[:, :n_train]
X_val, Y_val = X[:, n_train:n_train + n_val], Y[:, n_train:n_train + n_val]
X_test, Y_test = X[:, n_train + n_val:], Y[:, n_train + n_val:]

print(f"Train: {X_train.shape[1]}, Val: {X_val.shape[1]}, Test: {X_test.shape[1]}")

# Feature scaling
mean = X_train.mean(axis=1, keepdims=True)
std = X_train.std(axis=1, keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# ============================================================
# 2. Simple MLP Training
# ============================================================
print("\n" + "=" * 60)
print("2. MLP Training Loop")
print("=" * 60)


def relu(z): return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


def train_mlp(X_tr, Y_tr, X_v, Y_v, hidden=64, lr=0.01, epochs=200, batch_size=32):
    n_in, n_out = X_tr.shape[0], Y_tr.shape[0]
    W1 = np.random.randn(hidden, n_in) * np.sqrt(2.0 / n_in)
    b1 = np.zeros((hidden, 1))
    W2 = np.random.randn(n_out, hidden) * np.sqrt(2.0 / hidden)
    b2 = np.zeros((n_out, 1))

    best_val_acc = 0
    for epoch in range(epochs):
        perm = np.random.permutation(X_tr.shape[1])
        X_s, Y_s = X_tr[:, perm], Y_tr[:, perm]

        for i in range(0, X_tr.shape[1], batch_size):
            xb = X_s[:, i:i + batch_size]
            yb = Y_s[:, i:i + batch_size]
            m = xb.shape[1]

            z1 = W1 @ xb + b1
            a1 = relu(z1)
            z2 = W2 @ a1 + b2
            a2 = softmax(z2)

            dz2 = a2 - yb
            dW2 = (1 / m) * dz2 @ a1.T
            db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
            da1 = W2.T @ dz2
            dz1 = da1 * (z1 > 0)
            dW1 = (1 / m) * dz1 @ xb.T
            db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

            W2 -= lr * dW2; b2 -= lr * db2
            W1 -= lr * dW1; b1 -= lr * db1

        # Validation
        a1v = relu(W1 @ X_v + b1)
        a2v = softmax(W2 @ a1v + b2)
        val_acc = np.mean(np.argmax(a2v, axis=0) == np.argmax(Y_v, axis=0))
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc


acc = train_mlp(X_train, Y_train, X_val, Y_val)
print(f"Best validation accuracy: {acc:.4f}")


# ============================================================
# 3. Random Hyperparameter Search
# ============================================================
print("\n" + "=" * 60)
print("3. Random Hyperparameter Search (5 trials)")
print("=" * 60)

best_acc = 0
best_cfg = None
for trial in range(5):
    cfg = {
        'hidden': np.random.choice([32, 64, 128]),
        'lr': 10 ** np.random.uniform(-3, -1),
    }
    acc = train_mlp(X_train, Y_train, X_val, Y_val,
                    hidden=cfg['hidden'], lr=cfg['lr'], epochs=100)
    print(f"  Trial {trial+1}: hidden={cfg['hidden']:3d}, lr={cfg['lr']:.4f} -> val_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_cfg = cfg

print(f"\nBest config: {best_cfg}")
print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

"""
Putting It All Together

Complete worked example: train a classifier on the spiral dataset using
all mathematical concepts from the course:
- He initialization, batched linear layers (Lesson 01)
- ReLU gradients (Lesson 02)
- Backpropagation (Lesson 03)
- Adam optimizer with bias correction (Lesson 05)
- Softmax cross-entropy loss (Lessons 07, 08)
- L2 regularization as MAP (Lesson 07)
- Numerically stable softmax (Lesson 10)

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_spiral(N=200, K=3):
    """Generate K-class spiral dataset."""
    X = np.zeros((N * K, 2))
    y = np.zeros(N * K, dtype=int)
    for k in range(K):
        ix = range(N * k, N * (k + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(k * 4, (k + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k
    Y = np.zeros((N * K, K))
    Y[np.arange(N * K), y] = 1
    return X, y, Y


def stable_softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def train_classifier():
    """Train a 2-layer classifier on the spiral dataset."""
    print("=" * 60)
    print("SPIRAL DATASET CLASSIFIER")
    print("=" * 60)

    np.random.seed(42)
    X, y, Y = generate_spiral()
    N, K = X.shape[0], Y.shape[1]
    n_in, n_h, n_out = 2, 100, K

    # He initialization (Lesson 01)
    W1 = np.random.randn(n_h, n_in) * np.sqrt(2.0 / n_in)
    b1 = np.zeros(n_h)
    W2 = np.random.randn(n_out, n_h) * np.sqrt(2.0 / n_h)
    b2 = np.zeros(n_out)

    # Adam state
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr, beta1, beta2, eps, reg = 0.001, 0.9, 0.999, 1e-8, 1e-3

    losses = []
    for epoch in range(1, 501):
        # Forward (Lessons 01, 02, 10)
        z1 = X @ W1.T + b1
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ W2.T + b2
        probs = stable_softmax(z2)

        # Loss (Lessons 07, 08): CE + L2
        ce = -np.sum(Y * np.log(probs + 1e-10)) / N
        loss = ce + 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        losses.append(loss)

        # Backward (Lesson 03)
        dz2 = (probs - Y) / N
        dW2 = dz2.T @ a1 + reg * W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0)
        dW1 = dz1.T @ X + reg * W1
        db1 = dz1.sum(axis=0)
        grads = [dW1, db1, dW2, db2]

        # Adam update (Lesson 05)
        for i in range(4):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i]**2
            mh = m[i] / (1 - beta1**epoch)
            vh = v[i] / (1 - beta2**epoch)
            params[i] = params[i] - lr * mh / (np.sqrt(vh) + eps)
        W1, b1, W2, b2 = params

        if epoch % 100 == 0:
            acc = np.mean(np.argmax(probs, axis=1) == y)
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.3f}")

    # Final accuracy
    z1 = X @ W1.T + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ W2.T + b2
    probs = stable_softmax(z2)
    acc = np.mean(np.argmax(probs, axis=1) == y)
    print(f"\nFinal accuracy: {acc:.3f}")

    # Save loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (CE + L2 regularization)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spiral_training.png', dpi=100)
    plt.close()
    print("Saved spiral_training.png")


if __name__ == "__main__":
    train_classifier()

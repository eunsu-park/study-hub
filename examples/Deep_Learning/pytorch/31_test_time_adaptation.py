"""
Test-Time Adaptation Demo

Demonstrates TTA concepts:
  1. Batch Normalization adaptation to shifted data
  2. TENT (entropy minimization) for adapting model predictions
  3. Comparison of methods under distribution shift

Uses a simple MLP with BatchNorm on synthetic data to
illustrate how BN statistics and affine parameters can
be adapted at test time without labels.
"""

import numpy as np


# ============================================================
# 1. Synthetic BatchNorm Layer
# ============================================================
class BatchNorm1D:
    """Batch Normalization with running statistics.

    During training: uses batch mean/var, updates running stats.
    During eval: uses running mean/var from training.
    During TTA (BN adapt): uses batch mean/var from test data.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable affine parameters (updated by TENT)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics (updated during training)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True
        # Cache for backward pass
        self.x_norm = None
        self.std = None

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            # Update running statistics
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                 + self.momentum * mean)
            self.running_var = ((1 - self.momentum) * self.running_var
                                + self.momentum * var)
        else:
            mean = self.running_mean
            var = self.running_var

        self.std = np.sqrt(var + self.eps)
        self.x_norm = (x - mean) / self.std
        return self.gamma * self.x_norm + self.beta

    def adapt_statistics(self, x):
        """BN Adapt: use current batch statistics instead of running stats."""
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        return self.gamma * x_norm + self.beta


# ============================================================
# 2. Simple MLP with BatchNorm
# ============================================================
class MLPWithBN:
    """2-layer MLP with batch normalization.

    Architecture: input → Linear → BN → ReLU → Linear → output
    """

    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = rng.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.bn = BatchNorm1D(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def forward(self, x, use_bn_adapt=False):
        """Forward pass through MLP.

        use_bn_adapt: if True, use current batch statistics
        instead of stored running statistics.
        """
        self.h_pre = x @ self.W1 + self.b1

        if use_bn_adapt:
            self.h_bn = self.bn.adapt_statistics(self.h_pre)
        else:
            self.h_bn = self.bn.forward(self.h_pre)

        self.h_relu = np.maximum(self.h_bn, 0)
        logits = self.h_relu @ self.W2 + self.b2
        return logits

    def train(self):
        self.bn.training = True

    def eval(self):
        self.bn.training = False


# ============================================================
# 3. Training Utilities
# ============================================================
def softmax(logits):
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits, labels):
    """Cross-entropy loss."""
    probs = softmax(logits)
    n = len(labels)
    return -np.log(probs[np.arange(n), labels] + 1e-10).mean()


def entropy(logits):
    """Shannon entropy of predictions (measures uncertainty)."""
    probs = softmax(logits)
    log_probs = np.log(probs + 1e-10)
    return -(probs * log_probs).sum(axis=1).mean()


def accuracy(logits, labels):
    return (logits.argmax(axis=1) == labels).mean()


def train_model(model, X_train, y_train, epochs=100, lr=0.01,
                batch_size=64):
    """Train the MLP with SGD."""
    model.train()
    n = len(X_train)

    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        for i in range(0, n, batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]

            # Forward
            logits = model.forward(xb)
            probs = softmax(logits)

            # Backward (simplified for 2-layer MLP)
            n_batch = len(yb)
            d_logits = probs.copy()
            d_logits[np.arange(n_batch), yb] -= 1.0
            d_logits /= n_batch

            d_W2 = model.h_relu.T @ d_logits
            d_b2 = d_logits.sum(axis=0)
            d_h_relu = d_logits @ model.W2.T
            d_h_bn = d_h_relu * (model.h_bn > 0)

            # Through BN (simplified: update gamma/beta)
            d_gamma = (d_h_bn * model.bn.x_norm).sum(axis=0)
            d_beta = d_h_bn.sum(axis=0)
            d_x_norm = d_h_bn * model.bn.gamma
            d_h_pre = d_x_norm / model.bn.std

            d_W1 = xb.T @ d_h_pre
            d_b1 = d_h_pre.sum(axis=0)

            # SGD update
            model.W2 -= lr * d_W2
            model.b2 -= lr * d_b2
            model.W1 -= lr * d_W1
            model.b1 -= lr * d_b1
            model.bn.gamma -= lr * d_gamma
            model.bn.beta -= lr * d_beta


# ============================================================
# 4. Generate Data with Distribution Shift
# ============================================================
def generate_data(n_samples, n_classes=5, dim=20, shift=0.0,
                  noise_scale=1.0, seed=42):
    """Generate classification data with optional distribution shift.

    shift: translates all features (simulates domain shift)
    noise_scale: scales the noise (simulates corruption)
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, dim) * 3.0

    X = []
    y = []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        samples = (centers[c] + rng.randn(per_class, dim)
                   * noise_scale + shift)
        X.append(samples)
        y.extend([c] * per_class)

    return np.vstack(X), np.array(y)


# ============================================================
# 5. TENT: Test-Time Entropy Minimization
# ============================================================
def tent_adapt(model, X_test, lr=0.01, steps=3):
    """TENT: adapt BN affine parameters by minimizing prediction entropy.

    Only updates gamma and beta of BatchNorm layers.
    The key insight: minimizing entropy pushes the model toward
    more confident (and usually more correct) predictions.
    """
    # Save original parameters
    gamma_orig = model.bn.gamma.copy()
    beta_orig = model.bn.beta.copy()

    model.eval()  # use running stats for everything except affine

    for step in range(steps):
        # Forward with current gamma/beta
        logits = model.forward(X_test, use_bn_adapt=True)
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-10)

        # Entropy gradient w.r.t. logits
        # H = -sum(p * log(p))
        # d_H/d_logits = p * (1 + log(p)) - p * sum(p * (1 + log(p)))
        ent_grad = probs * (1.0 + log_probs)
        ent_grad -= probs * ent_grad.sum(axis=1, keepdims=True)

        # Backprop through final linear layer
        d_h_relu = ent_grad @ model.W2.T
        d_h_bn = d_h_relu * (model.h_bn > 0)

        # Gradients for gamma and beta only
        d_gamma = (d_h_bn * model.bn.x_norm).sum(axis=0)
        d_beta = d_h_bn.sum(axis=0)

        # Update only BN affine parameters
        model.bn.gamma -= lr * d_gamma
        model.bn.beta -= lr * d_beta

    return model


# ============================================================
# 6. Demo: Compare TTA Methods
# ============================================================
def demo_tta():
    """Compare standard inference, BN adapt, and TENT under shift."""
    print("=" * 60)
    print("Test-Time Adaptation Demo")
    print("=" * 60)

    dim = 20
    n_classes = 5

    # Generate training data (source domain)
    X_train, y_train = generate_data(1000, n_classes, dim,
                                      shift=0.0, noise_scale=0.5,
                                      seed=42)

    # Train model on source data
    model = MLPWithBN(dim, 64, n_classes, seed=42)
    train_model(model, X_train, y_train, epochs=80, lr=0.02)

    # Evaluate on source domain
    model.eval()
    logits = model.forward(X_train[:200])
    print(f"\nSource domain accuracy: {accuracy(logits, y_train[:200]):.1%}")
    print(f"Source domain entropy:  {entropy(logits):.3f}")

    # Test under various distribution shifts
    shifts = [0.0, 1.0, 2.0, 3.0, 5.0]
    print(f"\n{'Shift':>6} | {'Standard':>10} | {'BN Adapt':>10} | "
          f"{'TENT':>10} | {'Entropy (std)':>14} | {'Entropy (TENT)':>14}")
    print("-" * 80)

    for shift in shifts:
        X_test, y_test = generate_data(200, n_classes, dim,
                                        shift=shift, noise_scale=0.5,
                                        seed=99)

        # Method 1: Standard inference (uses training BN stats)
        model_std = MLPWithBN(dim, 64, n_classes, seed=42)
        # Copy trained weights
        model_std.W1 = model.W1.copy()
        model_std.b1 = model.b1.copy()
        model_std.W2 = model.W2.copy()
        model_std.b2 = model.b2.copy()
        model_std.bn.gamma = model.bn.gamma.copy()
        model_std.bn.beta = model.bn.beta.copy()
        model_std.bn.running_mean = model.bn.running_mean.copy()
        model_std.bn.running_var = model.bn.running_var.copy()
        model_std.eval()

        logits_std = model_std.forward(X_test)
        acc_std = accuracy(logits_std, y_test)
        ent_std = entropy(logits_std)

        # Method 2: BN Adapt (use test batch statistics)
        model_bn = MLPWithBN(dim, 64, n_classes, seed=42)
        model_bn.W1 = model.W1.copy()
        model_bn.b1 = model.b1.copy()
        model_bn.W2 = model.W2.copy()
        model_bn.b2 = model.b2.copy()
        model_bn.bn.gamma = model.bn.gamma.copy()
        model_bn.bn.beta = model.bn.beta.copy()
        model_bn.eval()

        logits_bn = model_bn.forward(X_test, use_bn_adapt=True)
        acc_bn = accuracy(logits_bn, y_test)

        # Method 3: TENT (BN adapt + entropy minimization)
        model_tent = MLPWithBN(dim, 64, n_classes, seed=42)
        model_tent.W1 = model.W1.copy()
        model_tent.b1 = model.b1.copy()
        model_tent.W2 = model.W2.copy()
        model_tent.b2 = model.b2.copy()
        model_tent.bn.gamma = model.bn.gamma.copy()
        model_tent.bn.beta = model.bn.beta.copy()
        model_tent.bn.running_mean = model.bn.running_mean.copy()
        model_tent.bn.running_var = model.bn.running_var.copy()

        tent_adapt(model_tent, X_test, lr=0.01, steps=5)
        logits_tent = model_tent.forward(X_test, use_bn_adapt=True)
        acc_tent = accuracy(logits_tent, y_test)
        ent_tent = entropy(logits_tent)

        print(f"{shift:6.1f} | {acc_std:10.1%} | {acc_bn:10.1%} | "
              f"{acc_tent:10.1%} | {ent_std:14.3f} | {ent_tent:14.3f}")

    print()
    print("Key observations:")
    print("  - Standard inference degrades rapidly with shift")
    print("  - BN Adapt recovers performance by using test batch stats")
    print("  - TENT further improves by optimizing BN affine parameters")
    print("  - TENT reduces entropy (model becomes more confident)")


# ============================================================
# 7. Batch Size Sensitivity Study
# ============================================================
def demo_batch_size():
    """Show how batch size affects BN Adapt quality."""
    print("\n" + "=" * 60)
    print("Batch Size Sensitivity for BN Adaptation")
    print("=" * 60)

    dim = 20
    n_classes = 5

    # Train on source
    X_train, y_train = generate_data(1000, n_classes, dim,
                                      shift=0.0, seed=42)
    model = MLPWithBN(dim, 64, n_classes, seed=42)
    train_model(model, X_train, y_train, epochs=80, lr=0.02)

    # Test with shift
    X_test, y_test = generate_data(500, n_classes, dim,
                                    shift=3.0, seed=99)
    model.eval()

    batch_sizes = [4, 8, 16, 32, 64, 128, 256, 500]
    print(f"\n{'Batch Size':>10} | {'BN Adapt Acc':>12} | {'Std Dev':>10}")
    print("-" * 40)

    for bs in batch_sizes:
        accs = []
        for start in range(0, len(X_test) - bs + 1, bs):
            batch = X_test[start:start+bs]
            labels = y_test[start:start+bs]

            model_copy = MLPWithBN(dim, 64, n_classes, seed=42)
            model_copy.W1 = model.W1.copy()
            model_copy.b1 = model.b1.copy()
            model_copy.W2 = model.W2.copy()
            model_copy.b2 = model.b2.copy()
            model_copy.bn.gamma = model.bn.gamma.copy()
            model_copy.bn.beta = model.bn.beta.copy()
            model_copy.eval()

            logits = model_copy.forward(batch, use_bn_adapt=True)
            accs.append(accuracy(logits, labels))

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{bs:10d} | {mean_acc:12.1%} | {std_acc:10.3f}")

    print("\nSmaller batches → noisier statistics → less reliable adaptation")


if __name__ == "__main__":
    demo_tta()
    demo_batch_size()

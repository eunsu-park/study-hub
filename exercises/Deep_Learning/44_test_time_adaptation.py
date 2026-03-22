"""
Exercises for Lesson 44: Test-Time Adaptation
Topic: Deep_Learning

Solutions to practice problems covering distribution shift analysis,
BN adaptation, TENT implementation, batch size sensitivity,
and continual adaptation.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Shared: Simple MLP with BatchNorm for classification
# ============================================================
class BatchNorm:
    """Batch normalization layer with running statistics."""

    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.momentum = momentum
        self.eps = eps
        self.training = True
        self.x_norm = None
        self.std = None

    def forward(self, x, use_batch_stats=False):
        if self.training or use_batch_stats:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            if self.training:
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


class SimpleMLP:
    """2-layer MLP with BatchNorm for TTA experiments."""

    def __init__(self, in_dim, hidden, out_dim, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, hidden) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden)
        self.bn = BatchNorm(hidden)
        self.W2 = rng.randn(hidden, out_dim) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(out_dim)

    def forward(self, x, use_batch_stats=False):
        self.h_pre = x @ self.W1 + self.b1
        self.h_bn = self.bn.forward(self.h_pre, use_batch_stats)
        self.h_relu = np.maximum(self.h_bn, 0)
        return self.h_relu @ self.W2 + self.b2

    def train_mode(self):
        self.bn.training = True

    def eval_mode(self):
        self.bn.training = False

    def copy(self):
        """Create an independent copy of this model."""
        import copy
        return copy.deepcopy(self)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def accuracy(logits, labels):
    return (logits.argmax(axis=1) == labels).mean()


def entropy(logits):
    p = softmax(logits)
    return -(p * np.log(p + 1e-10)).sum(axis=1).mean()


def generate_data(n, n_classes=5, dim=20, shift=0.0, noise=0.5, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, dim) * 3.0
    X, y = [], []
    per_class = n // n_classes
    for c in range(n_classes):
        X.append(centers[c] + rng.randn(per_class, dim) * noise + shift)
        y.extend([c] * per_class)
    return np.vstack(X), np.array(y)


def train_model(model, X, y, epochs=60, lr=0.02, bs=64):
    model.train_mode()
    n = len(X)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, bs):
            xb = X[idx[i:i+bs]]
            yb = y[idx[i:i+bs]]
            logits = model.forward(xb)
            probs = softmax(logits)
            nb = len(yb)
            d_logits = probs.copy()
            d_logits[np.arange(nb), yb] -= 1.0
            d_logits /= nb
            d_W2 = model.h_relu.T @ d_logits
            d_b2 = d_logits.sum(axis=0)
            d_h = d_logits @ model.W2.T * (model.h_bn > 0)
            d_gamma = (d_h * model.bn.x_norm).sum(axis=0)
            d_beta = d_h.sum(axis=0)
            d_pre = d_h * model.bn.gamma / model.bn.std
            d_W1 = xb.T @ d_pre
            d_b1 = d_pre.sum(axis=0)
            model.W2 -= lr * d_W2
            model.b2 -= lr * d_b2
            model.W1 -= lr * d_W1
            model.b1 -= lr * d_b1
            model.bn.gamma -= lr * d_gamma
            model.bn.beta -= lr * d_beta


def exercise_1():
    """
    Distribution shift analysis: measure accuracy degradation
    under various shift magnitudes.
    """
    print("=== Exercise 1: Distribution Shift Analysis ===\n")

    dim, n_classes = 20, 5
    X_train, y_train = generate_data(1000, n_classes, dim, seed=42)
    model = SimpleMLP(dim, 64, n_classes, seed=42)
    train_model(model, X_train, y_train, epochs=80)
    model.eval_mode()

    print(f"  {'Shift':>8} | {'Accuracy':>10} | {'Entropy':>10} | {'Drop':>8}")
    print("  " + "-" * 45)

    base_acc = None
    for shift in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        X_test, y_test = generate_data(200, n_classes, dim,
                                        shift=shift, seed=99)
        logits = model.forward(X_test)
        acc = accuracy(logits, y_test)
        ent = entropy(logits)
        if base_acc is None:
            base_acc = acc
        drop = base_acc - acc
        print(f"  {shift:8.1f} | {acc:10.1%} | {ent:10.3f} | {drop:8.1%}")

    print(f"\n  Key observation: accuracy drops monotonically with shift")
    print(f"  Entropy increases: model becomes less confident on shifted data")
    print()


def exercise_2():
    """
    BN Adaptation: replace running statistics with test batch statistics.
    """
    print("=== Exercise 2: Batch Normalization Adaptation ===\n")

    dim, n_classes = 20, 5
    X_train, y_train = generate_data(1000, n_classes, dim, seed=42)
    model = SimpleMLP(dim, 64, n_classes, seed=42)
    train_model(model, X_train, y_train, epochs=80)

    shifts = [1.0, 2.0, 3.0, 5.0]
    num_batches_list = [1, 5, 10, 50]

    print(f"  {'Shift':>6} | {'Standard':>10} |", end="")
    for nb in num_batches_list:
        print(f" {'BN-'+str(nb):>8} |", end="")
    print()
    print("  " + "-" * (30 + 11 * len(num_batches_list)))

    for shift in shifts:
        X_test, y_test = generate_data(500, n_classes, dim,
                                        shift=shift, seed=99)

        # Standard (no adaptation)
        m_std = model.copy()
        m_std.eval_mode()
        acc_std = accuracy(m_std.forward(X_test), y_test)

        print(f"  {shift:6.1f} | {acc_std:10.1%} |", end="")

        # BN Adapt with different numbers of batches
        for n_batches in num_batches_list:
            m_bn = model.copy()
            m_bn.eval_mode()
            # Reset running stats
            m_bn.bn.running_mean = np.zeros_like(m_bn.bn.running_mean)
            m_bn.bn.running_var = np.ones_like(m_bn.bn.running_var)
            m_bn.bn.momentum = None  # cumulative average

            # Collect statistics
            bs = 64
            count = 0
            for i in range(0, len(X_test), bs):
                if count >= n_batches:
                    break
                batch = X_test[i:i+bs]
                h = batch @ m_bn.W1 + m_bn.b1
                mean = h.mean(axis=0)
                var = h.var(axis=0)
                if count == 0:
                    m_bn.bn.running_mean = mean
                    m_bn.bn.running_var = var
                else:
                    m_bn.bn.running_mean = (
                        count * m_bn.bn.running_mean + mean) / (count + 1)
                    m_bn.bn.running_var = (
                        count * m_bn.bn.running_var + var) / (count + 1)
                count += 1

            acc_bn = accuracy(m_bn.forward(X_test), y_test)
            print(f" {acc_bn:8.1%} |", end="")
        print()

    print(f"\n  More batches → more reliable statistics → better adaptation")
    print(f"  Even 5 batches significantly improves over standard inference")
    print()


def exercise_3():
    """
    TENT implementation: entropy minimization on BN affine parameters.
    """
    print("=== Exercise 3: TENT Implementation ===\n")

    dim, n_classes = 20, 5
    X_train, y_train = generate_data(1000, n_classes, dim, seed=42)
    model = SimpleMLP(dim, 64, n_classes, seed=42)
    train_model(model, X_train, y_train, epochs=80)

    shift = 3.0
    X_test, y_test = generate_data(200, n_classes, dim,
                                    shift=shift, seed=99)

    steps_list = [0, 1, 3, 5, 10, 20]
    print(f"  Distribution shift = {shift}")
    print(f"\n  {'TENT Steps':>11} | {'Accuracy':>10} | {'Entropy':>10}")
    print("  " + "-" * 40)

    for n_steps in steps_list:
        m = model.copy()
        m.eval_mode()

        # TENT: optimize gamma/beta to minimize entropy
        lr = 0.01
        for step in range(n_steps):
            logits = m.forward(X_test, use_batch_stats=True)
            probs = softmax(logits)
            log_probs = np.log(probs + 1e-10)

            # Entropy gradient w.r.t. logits
            ent_grad = probs * (1.0 + log_probs)
            ent_grad -= probs * ent_grad.sum(axis=1, keepdims=True)

            # Backprop to BN parameters
            d_h = ent_grad @ m.W2.T * (m.h_bn > 0)
            d_gamma = (d_h * m.bn.x_norm).sum(axis=0)
            d_beta = d_h.sum(axis=0)
            m.bn.gamma -= lr * d_gamma
            m.bn.beta -= lr * d_beta

        logits = m.forward(X_test, use_batch_stats=True)
        acc = accuracy(logits, y_test)
        ent = entropy(logits)
        print(f"  {n_steps:11d} | {acc:10.1%} | {ent:10.3f}")

    print(f"\n  TENT reduces entropy → more confident predictions")
    print(f"  But too many steps can overfit → watch for accuracy drop")
    print()


def exercise_4():
    """
    Batch size sensitivity: BN adapt performance vs batch size.
    """
    print("=== Exercise 4: Batch Size Sensitivity ===\n")

    dim, n_classes = 20, 5
    X_train, y_train = generate_data(1000, n_classes, dim, seed=42)
    model = SimpleMLP(dim, 64, n_classes, seed=42)
    train_model(model, X_train, y_train, epochs=80)

    X_test, y_test = generate_data(500, n_classes, dim,
                                    shift=3.0, seed=99)
    model.eval_mode()

    batch_sizes = [4, 8, 16, 32, 64, 128, 256]
    print(f"  {'Batch Size':>10} | {'Mean Acc':>10} | {'Std':>8} | "
          f"{'Min':>8} | {'Max':>8}")
    print("  " + "-" * 55)

    for bs in batch_sizes:
        accs = []
        for start in range(0, len(X_test) - bs + 1, bs):
            batch_x = X_test[start:start+bs]
            batch_y = y_test[start:start+bs]
            m = model.copy()
            m.eval_mode()
            logits = m.forward(batch_x, use_batch_stats=True)
            accs.append(accuracy(logits, batch_y))

        print(f"  {bs:10d} | {np.mean(accs):10.1%} | {np.std(accs):8.3f} | "
              f"{np.min(accs):8.1%} | {np.max(accs):8.1%}")

    print(f"\n  Small batches have high variance in BN statistics")
    print(f"  For batch_size=4, some batches may even hurt performance")
    print(f"  Recommendation: use batch_size ≥ 32 for reliable BN adaptation")
    print()


def exercise_5():
    """
    Continual adaptation: simulate shifting distributions over time.
    Compare naive TENT vs. stochastic restore (CoTTA-style).
    """
    print("=== Exercise 5: Continual Test-Time Adaptation ===\n")

    dim, n_classes = 20, 5
    X_train, y_train = generate_data(1000, n_classes, dim, seed=42)
    model = SimpleMLP(dim, 64, n_classes, seed=42)
    train_model(model, X_train, y_train, epochs=80)

    # Simulate continual shift: shift changes every 5 batches
    shift_sequence = [1.0, 2.0, 3.0, 5.0, 2.0, 0.5, 4.0, 1.0]
    batches_per_shift = 5
    batch_size = 64

    print(f"  Shift sequence: {shift_sequence}")
    print(f"  {batches_per_shift} batches per shift, batch_size={batch_size}\n")

    # Method 1: Naive TENT (accumulates errors)
    m_naive = model.copy()
    m_naive.eval_mode()
    naive_accs = []

    # Method 2: CoTTA-style with stochastic restore
    m_cotta = model.copy()
    m_cotta.eval_mode()
    source_gamma = model.bn.gamma.copy()
    source_beta = model.bn.beta.copy()
    restore_prob = 0.05
    cotta_accs = []

    for shift in shift_sequence:
        for batch_idx in range(batches_per_shift):
            X_batch, y_batch = generate_data(
                batch_size, n_classes, dim, shift=shift,
                seed=int(shift * 100 + batch_idx))

            # Naive TENT
            for _ in range(3):
                logits = m_naive.forward(X_batch, use_batch_stats=True)
                probs = softmax(logits)
                log_probs = np.log(probs + 1e-10)
                ent_grad = probs * (1.0 + log_probs)
                ent_grad -= probs * ent_grad.sum(axis=1, keepdims=True)
                d_h = ent_grad @ m_naive.W2.T * (m_naive.h_bn > 0)
                m_naive.bn.gamma -= 0.01 * (d_h * m_naive.bn.x_norm).sum(0)
                m_naive.bn.beta -= 0.01 * d_h.sum(0)

            logits_naive = m_naive.forward(X_batch, use_batch_stats=True)
            naive_accs.append(accuracy(logits_naive, y_batch))

            # CoTTA-style: TENT + stochastic restore
            for _ in range(3):
                logits = m_cotta.forward(X_batch, use_batch_stats=True)
                probs = softmax(logits)
                log_probs = np.log(probs + 1e-10)
                ent_grad = probs * (1.0 + log_probs)
                ent_grad -= probs * ent_grad.sum(axis=1, keepdims=True)
                d_h = ent_grad @ m_cotta.W2.T * (m_cotta.h_bn > 0)
                m_cotta.bn.gamma -= 0.01 * (d_h * m_cotta.bn.x_norm).sum(0)
                m_cotta.bn.beta -= 0.01 * d_h.sum(0)

            # Stochastic restore: randomly reset some parameters to source
            mask_g = np.random.random(len(m_cotta.bn.gamma)) < restore_prob
            mask_b = np.random.random(len(m_cotta.bn.beta)) < restore_prob
            m_cotta.bn.gamma[mask_g] = source_gamma[mask_g]
            m_cotta.bn.beta[mask_b] = source_beta[mask_b]

            logits_cotta = m_cotta.forward(X_batch, use_batch_stats=True)
            cotta_accs.append(accuracy(logits_cotta, y_batch))

    # Report results
    print(f"  {'Phase':>6} | {'Shift':>6} | {'Naive TENT':>11} | {'CoTTA':>10}")
    print("  " + "-" * 45)

    for i, shift in enumerate(shift_sequence):
        start = i * batches_per_shift
        end = start + batches_per_shift
        n_acc = np.mean(naive_accs[start:end])
        c_acc = np.mean(cotta_accs[start:end])
        print(f"  {i+1:6d} | {shift:6.1f} | {n_acc:11.1%} | {c_acc:10.1%}")

    print(f"\n  Overall: Naive TENT = {np.mean(naive_accs):.1%}, "
          f"CoTTA = {np.mean(cotta_accs):.1%}")
    print(f"  Stochastic restore prevents drift from source knowledge")
    print(f"  Naive TENT accumulates errors as shift changes")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

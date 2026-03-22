"""
Few-Shot Learning — Prototypical Networks Demo

Implements a Prototypical Network for few-shot classification
using a simple embedding network. Demonstrates N-way K-shot
episodic training and evaluation on synthetic data.

Key concepts:
  - Episodic training with support and query sets
  - Embedding network → prototype computation → distance-based classification
  - N-way K-shot task construction
"""

import numpy as np


# ============================================================
# 1. Synthetic Dataset: Clustered Gaussians
# ============================================================
class FewShotDataset:
    """Generate synthetic data for few-shot learning.

    Creates C classes, each a Gaussian cluster in D dimensions.
    Each class has many samples available, but only K are shown
    during a few-shot episode.
    """

    def __init__(self, num_classes=20, dim=64, samples_per_class=50,
                 seed=42):
        rng = np.random.RandomState(seed)
        self.num_classes = num_classes
        self.dim = dim

        # Generate class centers spread apart
        self.centers = rng.randn(num_classes, dim) * 3.0
        # Generate samples around each center
        self.data = {}
        for c in range(num_classes):
            self.data[c] = self.centers[c] + rng.randn(
                samples_per_class, dim) * 0.5

    def sample_episode(self, n_way, k_shot, n_query, rng=None):
        """Sample a few-shot episode.

        Returns:
            support_x: (n_way * k_shot, dim) support examples
            support_y: (n_way * k_shot,) support labels (0..n_way-1)
            query_x:   (n_way * n_query, dim) query examples
            query_y:   (n_way * n_query,) query labels
        """
        if rng is None:
            rng = np.random.RandomState()

        # Sample n_way classes
        classes = rng.choice(self.num_classes, n_way, replace=False)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for label, cls in enumerate(classes):
            samples = self.data[cls]
            indices = rng.choice(len(samples), k_shot + n_query,
                                 replace=False)
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:]

            support_x.append(samples[support_idx])
            support_y.extend([label] * k_shot)
            query_x.append(samples[query_idx])
            query_y.extend([label] * n_query)

        support_x = np.vstack(support_x)
        query_x = np.vstack(query_x)
        return (support_x, np.array(support_y),
                query_x, np.array(query_y))


# ============================================================
# 2. Embedding Network (simple 2-layer MLP)
# ============================================================
class EmbeddingNetwork:
    """Simple MLP embedding for few-shot learning.

    Maps input features to a lower-dimensional embedding space
    where distance-based classification works well.
    """

    def __init__(self, input_dim, hidden_dim=128, embed_dim=32,
                 seed=42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = rng.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, embed_dim) * scale2
        self.b2 = np.zeros(embed_dim)

        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        """Forward pass: x → ReLU → embedding."""
        self.x = x
        self.h = x @ self.W1 + self.b1
        self.h_relu = np.maximum(self.h, 0)
        self.out = self.h_relu @ self.W2 + self.b2
        return self.out

    def backward(self, d_out):
        """Backward pass: compute gradients for all parameters."""
        d_h_relu = d_out @ self.W2.T
        d_W2 = self.h_relu.T @ d_out
        d_b2 = d_out.sum(axis=0)

        d_h = d_h_relu * (self.h > 0).astype(float)
        d_W1 = self.x.T @ d_h
        d_b1 = d_h.sum(axis=0)

        self.grads = [d_W1, d_b1, d_W2, d_b2]
        return self.grads


# ============================================================
# 3. Prototypical Network
# ============================================================
class PrototypicalNetwork:
    """Prototypical Network for few-shot classification.

    Algorithm:
      1. Embed all support and query examples
      2. Compute class prototypes = mean of support embeddings per class
      3. Classify queries by nearest prototype (Euclidean distance)
    """

    def __init__(self, embedding_net):
        self.embedding = embedding_net

    def compute_prototypes(self, support_embeddings, support_labels,
                           n_way):
        """Compute class prototypes from support set.

        prototype_c = mean of embeddings for class c
        """
        prototypes = np.zeros((n_way, support_embeddings.shape[1]))
        for c in range(n_way):
            mask = support_labels == c
            prototypes[c] = support_embeddings[mask].mean(axis=0)
        return prototypes

    def predict(self, support_x, support_y, query_x, n_way):
        """Predict query labels using prototypical classification."""
        # Embed support and query
        support_emb = self.embedding.forward(support_x)
        query_emb = self.embedding.forward(query_x)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_emb, support_y,
                                             n_way)

        # Compute distances: each query to each prototype
        # dists[i, c] = ||query_i - prototype_c||^2
        dists = np.zeros((len(query_x), n_way))
        for c in range(n_way):
            diff = query_emb - prototypes[c]
            dists[:, c] = (diff ** 2).sum(axis=1)

        # Predict: nearest prototype
        predictions = dists.argmin(axis=1)
        return predictions, dists

    def train_step(self, support_x, support_y, query_x, query_y,
                   n_way, lr=0.01):
        """One training step on an episode.

        Loss: cross-entropy on negative distances (softmax over -d).
        """
        # Forward pass
        all_x = np.vstack([support_x, query_x])
        all_emb = self.embedding.forward(all_x)
        n_support = len(support_x)
        support_emb = all_emb[:n_support]
        query_emb = all_emb[n_support:]

        # Compute prototypes
        prototypes = self.compute_prototypes(support_emb, support_y,
                                             n_way)

        # Negative squared distances → logits
        dists = np.zeros((len(query_x), n_way))
        for c in range(n_way):
            diff = query_emb - prototypes[c]
            dists[:, c] = (diff ** 2).sum(axis=1)
        logits = -dists  # negative distance as similarity

        # Softmax cross-entropy loss
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        n_query = len(query_y)
        loss = -np.log(softmax[np.arange(n_query), query_y] + 1e-10).mean()

        # Backward pass (gradient of loss w.r.t. query embeddings)
        d_softmax = softmax.copy()
        d_softmax[np.arange(n_query), query_y] -= 1.0
        d_softmax /= n_query

        # Gradient through -dists → d_query_emb
        d_query_emb = np.zeros_like(query_emb)
        for c in range(n_way):
            diff = query_emb - prototypes[c]
            d_query_emb += 2 * diff * d_softmax[:, c:c+1] * (-1)

        # Simplified: backprop through embedding
        d_all_emb = np.zeros_like(all_emb)
        d_all_emb[n_support:] = d_query_emb
        self.embedding.x = all_x  # reset for backward
        self.embedding.forward(all_x)  # recompute activations
        grads = self.embedding.backward(d_all_emb)

        # SGD update
        for param, grad in zip(self.embedding.params, grads):
            param -= lr * grad

        accuracy = (logits.argmax(axis=1) == query_y).mean()
        return loss, accuracy


# ============================================================
# 4. Demo: Train and Evaluate
# ============================================================
def demo_prototypical_network():
    """Train a Prototypical Network on synthetic few-shot tasks."""
    print("=" * 60)
    print("Prototypical Network — Few-Shot Learning Demo")
    print("=" * 60)

    # Setup
    n_way = 5     # 5-way classification
    k_shot = 5    # 5 examples per class (support)
    n_query = 10  # 10 queries per class

    # Create train and test datasets (disjoint classes)
    train_dataset = FewShotDataset(num_classes=15, dim=64, seed=42)
    test_dataset = FewShotDataset(num_classes=5, dim=64, seed=99)

    # Build model
    embedding = EmbeddingNetwork(input_dim=64, hidden_dim=128,
                                  embed_dim=32)
    model = PrototypicalNetwork(embedding)

    # Training
    print(f"\nTraining: {n_way}-way {k_shot}-shot")
    print(f"  Support: {n_way * k_shot} examples per episode")
    print(f"  Query:   {n_way * n_query} examples per episode")
    print()

    rng = np.random.RandomState(0)
    num_episodes = 200
    running_loss = 0.0
    running_acc = 0.0

    for ep in range(1, num_episodes + 1):
        sx, sy, qx, qy = train_dataset.sample_episode(
            n_way, k_shot, n_query, rng)
        loss, acc = model.train_step(sx, sy, qx, qy, n_way, lr=0.005)
        running_loss += loss
        running_acc += acc

        if ep % 50 == 0:
            print(f"  Episode {ep:4d}: "
                  f"loss={running_loss/50:.3f}, "
                  f"acc={running_acc/50:.1%}")
            running_loss = 0.0
            running_acc = 0.0

    # Evaluation
    print("\nEvaluation on 100 test episodes:")
    test_accs = []
    for _ in range(100):
        sx, sy, qx, qy = test_dataset.sample_episode(
            n_way, k_shot, n_query, rng)
        preds, _ = model.predict(sx, sy, qx, n_way)
        acc = (preds == qy).mean()
        test_accs.append(acc)

    print(f"  Mean accuracy: {np.mean(test_accs):.1%} "
          f"± {np.std(test_accs):.1%}")
    print(f"  Random baseline: {1/n_way:.1%}")

    # Show effect of K-shot
    print("\nEffect of K (number of shots):")
    for k in [1, 3, 5, 10, 20]:
        accs = []
        for _ in range(50):
            sx, sy, qx, qy = test_dataset.sample_episode(
                n_way, k, n_query, rng)
            preds, _ = model.predict(sx, sy, qx, n_way)
            accs.append((preds == qy).mean())
        print(f"  {k:2d}-shot: {np.mean(accs):.1%}")


# ============================================================
# 5. MAML-Style Inner Loop (simplified)
# ============================================================
def demo_maml_concept():
    """Demonstrate the MAML inner-loop concept."""
    print("\n" + "=" * 60)
    print("MAML Concept — Adapt Parameters at Test Time")
    print("=" * 60)

    # Simple 1D regression: learn to fit a sine with random phase
    def generate_task(rng):
        """Each task: y = sin(x + phase)."""
        phase = rng.uniform(0, 2 * np.pi)
        return phase

    def task_data(phase, n_samples, rng):
        x = rng.uniform(-5, 5, (n_samples, 1))
        y = np.sin(x + phase)
        return x, y

    rng = np.random.RandomState(42)

    # Simple model: y = w2 * relu(w1 * x + b1) + b2
    w1 = rng.randn(1, 40) * 0.5
    b1 = np.zeros(40)
    w2 = rng.randn(40, 1) * 0.1
    b2 = np.zeros(1)

    def predict(x, w1, b1, w2, b2):
        h = np.maximum(x @ w1 + b1, 0)
        return h @ w2 + b2

    def loss(x, y, w1, b1, w2, b2):
        pred = predict(x, w1, b1, w2, b2)
        return ((pred - y) ** 2).mean()

    print("\nTest: adapt to new sine task with 5 examples")
    test_phase = generate_task(rng)
    # Support: 5 examples
    sx, sy = task_data(test_phase, 5, rng)
    # Query: 50 examples
    qx, qy = task_data(test_phase, 50, rng)

    print(f"  Before adaptation: MSE = "
          f"{loss(qx, qy, w1, b1, w2, b2):.4f}")

    # MAML inner loop: a few gradient steps on support set
    inner_lr = 0.01
    w1_adapted, b1_adapted = w1.copy(), b1.copy()
    w2_adapted, b2_adapted = w2.copy(), b2.copy()

    for step in range(10):
        # Forward
        h = np.maximum(sx @ w1_adapted + b1_adapted, 0)
        pred = h @ w2_adapted + b2_adapted
        err = pred - sy
        # Backward (simplified gradients)
        d_w2 = h.T @ (2 * err / len(sx))
        d_b2 = (2 * err / len(sx)).sum(axis=0)
        d_h = (2 * err / len(sx)) @ w2_adapted.T
        d_h = d_h * (h > 0)
        d_w1 = sx.T @ d_h
        d_b1 = d_h.sum(axis=0)
        # Update
        w1_adapted -= inner_lr * d_w1
        b1_adapted -= inner_lr * d_b1
        w2_adapted -= inner_lr * d_w2
        b2_adapted -= inner_lr * d_b2

    adapted_loss = loss(qx, qy, w1_adapted, b1_adapted,
                        w2_adapted, b2_adapted)
    print(f"  After 10 inner steps: MSE = {adapted_loss:.4f}")
    print("  MAML's meta-training would optimize the INITIAL weights")
    print("  so that a few gradient steps yield good adaptation.")


if __name__ == "__main__":
    demo_prototypical_network()
    demo_maml_concept()

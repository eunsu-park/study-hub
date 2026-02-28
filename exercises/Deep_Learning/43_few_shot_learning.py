"""
Exercises for Lesson 43: Few-Shot Learning
Topic: Deep_Learning

Solutions to practice problems covering prototypical networks,
matching networks, MAML inner-loop, shot count analysis,
and cross-domain few-shot evaluation.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Shared: Synthetic Few-Shot Data Generator
# ============================================================
class FewShotData:
    """Generates synthetic classification data for few-shot tasks.

    Each class is a Gaussian cluster in D-dimensional space.
    """

    def __init__(self, num_classes=20, dim=32, samples_per_class=100,
                 seed=42):
        rng = np.random.RandomState(seed)
        self.num_classes = num_classes
        self.dim = dim
        self.centers = rng.randn(num_classes, dim) * 3.0
        self.data = {}
        for c in range(num_classes):
            self.data[c] = (self.centers[c]
                            + rng.randn(samples_per_class, dim) * 0.5)

    def sample_episode(self, n_way, k_shot, n_query, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        classes = rng.choice(self.num_classes, n_way, replace=False)
        sx, sy, qx, qy = [], [], [], []
        for label, cls in enumerate(classes):
            idx = rng.choice(len(self.data[cls]), k_shot + n_query,
                             replace=False)
            sx.append(self.data[cls][idx[:k_shot]])
            sy.extend([label] * k_shot)
            qx.append(self.data[cls][idx[k_shot:]])
            qy.extend([label] * n_query)
        return (np.vstack(sx), np.array(sy),
                np.vstack(qx), np.array(qy))


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def exercise_1():
    """
    Prototypical Network implementation and evaluation.
    Compute prototypes, classify by nearest centroid, measure accuracy.
    """
    print("=== Exercise 1: Prototypical Network ===\n")

    dataset = FewShotData(num_classes=15, dim=32, seed=42)
    rng = np.random.RandomState(0)
    n_way, k_shot, n_query = 5, 5, 15

    # Simple embedding: linear projection
    W = rng.randn(32, 16) * 0.3

    def embed(x):
        return np.tanh(x @ W)

    accuracies = []
    for ep in range(200):
        sx, sy, qx, qy = dataset.sample_episode(n_way, k_shot, n_query, rng)
        s_emb = embed(sx)
        q_emb = embed(qx)

        # Compute prototypes: mean embedding per class
        prototypes = np.zeros((n_way, 16))
        for c in range(n_way):
            prototypes[c] = s_emb[sy == c].mean(axis=0)

        # Classify queries by nearest prototype
        dists = np.zeros((len(qx), n_way))
        for c in range(n_way):
            diff = q_emb - prototypes[c]
            dists[:, c] = (diff ** 2).sum(axis=1)

        preds = dists.argmin(axis=1)
        accuracies.append((preds == qy).mean())

    print(f"  5-way 5-shot accuracy: {np.mean(accuracies):.1%} "
          f"± {np.std(accuracies):.1%}")
    print(f"  Random baseline: {1/n_way:.1%}")
    print(f"  Key insight: prototypes = class centroids in embedding space")
    print()


def exercise_2():
    """
    K-shot analysis: measure how accuracy scales with K.
    """
    print("=== Exercise 2: Effect of K (Number of Shots) ===\n")

    dataset = FewShotData(num_classes=20, dim=32, seed=42)
    rng = np.random.RandomState(0)
    n_way = 5
    n_query = 15

    W = rng.randn(32, 16) * 0.3

    def embed(x):
        return np.tanh(x @ W)

    print(f"  {'K-shot':>8} | {'Accuracy':>10} | {'Std':>8}")
    print("  " + "-" * 35)

    for k_shot in [1, 2, 3, 5, 10, 20, 50]:
        accs = []
        for _ in range(100):
            sx, sy, qx, qy = dataset.sample_episode(
                n_way, k_shot, n_query, rng)
            s_emb = embed(sx)
            q_emb = embed(qx)
            protos = np.zeros((n_way, 16))
            for c in range(n_way):
                protos[c] = s_emb[sy == c].mean(axis=0)
            dists = np.zeros((len(qx), n_way))
            for c in range(n_way):
                dists[:, c] = ((q_emb - protos[c]) ** 2).sum(axis=1)
            preds = dists.argmin(axis=1)
            accs.append((preds == qy).mean())

        print(f"  {k_shot:8d} | {np.mean(accs):10.1%} | {np.std(accs):8.3f}")

    print("\n  More shots → better prototypes → higher accuracy")
    print("  Diminishing returns: 20-shot vs 50-shot is smaller gain than 1 vs 5")
    print()


def exercise_3():
    """
    Matching Networks: attention-based classification using cosine similarity.
    """
    print("=== Exercise 3: Matching Networks ===\n")

    dataset = FewShotData(num_classes=15, dim=32, seed=42)
    rng = np.random.RandomState(0)
    n_way, k_shot, n_query = 5, 5, 15

    W = rng.randn(32, 16) * 0.3

    def embed(x):
        h = np.tanh(x @ W)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
        return h / norms

    accuracies = []
    for _ in range(200):
        sx, sy, qx, qy = dataset.sample_episode(n_way, k_shot, n_query, rng)
        s_emb = embed(sx)
        q_emb = embed(qx)

        # Attention: cosine similarity between each query and all support
        # sim[i, j] = cosine(query_i, support_j)
        sim = q_emb @ s_emb.T  # (n_query*n_way, n_support*n_way)

        # Softmax attention over support set
        attn = np.exp(sim - sim.max(axis=1, keepdims=True))
        attn = attn / attn.sum(axis=1, keepdims=True)

        # Weighted vote: sum attention weights per class
        class_scores = np.zeros((len(qx), n_way))
        for c in range(n_way):
            mask = (sy == c).astype(float)
            class_scores[:, c] = (attn * mask).sum(axis=1)

        preds = class_scores.argmax(axis=1)
        accuracies.append((preds == qy).mean())

    print(f"  Matching Network accuracy: {np.mean(accuracies):.1%}")
    print(f"  (vs Prototypical Network from Ex1 for comparison)")
    print(f"\n  Key difference from ProtoNet:")
    print(f"    ProtoNet: compares query to CLASS prototypes (averaged)")
    print(f"    MatchNet: compares query to EACH support example (weighted)")
    print()


def exercise_4():
    """
    MAML inner-loop: demonstrate rapid adaptation with few gradient steps.
    """
    print("=== Exercise 4: MAML Inner-Loop Adaptation ===\n")

    # Task family: classify 2D points, different rotation per task
    rng = np.random.RandomState(42)

    def generate_task(rng):
        """Create a 3-class classification task with random rotation."""
        angle = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        # 3 class centers at 120° apart, then rotated
        centers = np.array([[2, 0], [-1, 1.73], [-1, -1.73]])
        rotated_centers = centers @ rot.T
        return rotated_centers

    def sample_data(centers, k_per_class, rng):
        X, y = [], []
        for c, center in enumerate(centers):
            X.append(center + rng.randn(k_per_class, 2) * 0.3)
            y.extend([c] * k_per_class)
        return np.vstack(X), np.array(y)

    # Initialize meta-parameters
    W = rng.randn(2, 8) * 0.5
    b = np.zeros(8)
    V = rng.randn(8, 3) * 0.3
    c = np.zeros(3)

    def predict(x, W, b, V, c):
        h = np.maximum(x @ W + b, 0)
        return h @ V + c

    def loss_and_acc(x, y, W, b, V, c):
        logits = predict(x, W, b, V, c)
        probs = softmax(logits)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-10).mean()
        acc = (logits.argmax(axis=1) == y).mean()
        return loss, acc

    # Test adaptation on a new task
    test_centers = generate_task(rng)
    support_x, support_y = sample_data(test_centers, 5, rng)
    query_x, query_y = sample_data(test_centers, 30, rng)

    print("  Before adaptation:")
    l0, a0 = loss_and_acc(query_x, query_y, W, b, V, c)
    print(f"    Loss = {l0:.3f}, Accuracy = {a0:.1%}")

    # MAML inner loop: gradient descent on support set
    Wa, ba, Va, ca = W.copy(), b.copy(), V.copy(), c.copy()
    inner_lr = 0.05

    for step in range(20):
        h = np.maximum(support_x @ Wa + ba, 0)
        logits = h @ Va + ca
        probs = softmax(logits)
        n = len(support_y)

        # Gradient computation
        d_logits = probs.copy()
        d_logits[np.arange(n), support_y] -= 1.0
        d_logits /= n

        d_Va = h.T @ d_logits
        d_ca = d_logits.sum(axis=0)
        d_h = d_logits @ Va.T
        d_h *= (h > 0)
        d_Wa = support_x.T @ d_h
        d_ba = d_h.sum(axis=0)

        Wa -= inner_lr * d_Wa
        ba -= inner_lr * d_ba
        Va -= inner_lr * d_Va
        ca -= inner_lr * d_ca

    print(f"\n  After {20} inner-loop steps on 5-shot support:")
    l1, a1 = loss_and_acc(query_x, query_y, Wa, ba, Va, ca)
    print(f"    Loss = {l1:.3f}, Accuracy = {a1:.1%}")
    print(f"\n  MAML meta-training would optimize initial (W,b,V,c)")
    print(f"  so that these few steps yield even better adaptation.")
    print()


def exercise_5():
    """
    Cross-domain evaluation: train on one distribution, test on another.
    """
    print("=== Exercise 5: Cross-Domain Few-Shot ===\n")

    # Source: classes with small variance
    source = FewShotData(num_classes=15, dim=32,
                         samples_per_class=100, seed=42)
    # Target: classes with different statistics (shifted, higher variance)
    rng_target = np.random.RandomState(99)
    target = FewShotData(num_classes=10, dim=32,
                         samples_per_class=100, seed=99)
    # Shift target data
    for c in target.data:
        target.data[c] += rng_target.randn(1, 32) * 2.0

    rng = np.random.RandomState(0)
    n_way, k_shot, n_query = 5, 5, 15

    # Learn embedding on source domain
    W = rng.randn(32, 16) * 0.3

    def embed(x):
        return np.tanh(x @ W)

    # Evaluate on source (in-domain)
    source_accs = []
    for _ in range(100):
        sx, sy, qx, qy = source.sample_episode(
            n_way, k_shot, n_query, rng)
        s_emb, q_emb = embed(sx), embed(qx)
        protos = np.zeros((n_way, 16))
        for c in range(n_way):
            protos[c] = s_emb[sy == c].mean(axis=0)
        dists = np.zeros((len(qx), n_way))
        for c in range(n_way):
            dists[:, c] = ((q_emb - protos[c]) ** 2).sum(axis=1)
        source_accs.append((dists.argmin(axis=1) == qy).mean())

    # Evaluate on target (cross-domain)
    target_accs = []
    for _ in range(100):
        sx, sy, qx, qy = target.sample_episode(
            n_way, k_shot, n_query, rng)
        s_emb, q_emb = embed(sx), embed(qx)
        protos = np.zeros((n_way, 16))
        for c in range(n_way):
            protos[c] = s_emb[sy == c].mean(axis=0)
        dists = np.zeros((len(qx), n_way))
        for c in range(n_way):
            dists[:, c] = ((q_emb - protos[c]) ** 2).sum(axis=1)
        target_accs.append((dists.argmin(axis=1) == qy).mean())

    print(f"  Source domain (in-distribution): {np.mean(source_accs):.1%}")
    print(f"  Target domain (cross-domain):    {np.mean(target_accs):.1%}")
    print(f"  Drop: {np.mean(source_accs) - np.mean(target_accs):.1%}")
    print(f"\n  Cross-domain gap shows that embeddings are domain-specific.")
    print(f"  Solutions: meta-learning across domains, feature alignment,")
    print(f"  or domain-agnostic training (e.g., self-supervised pre-training).")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

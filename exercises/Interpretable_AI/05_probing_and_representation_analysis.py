"""
Exercises for Lesson 05: Probing and Representation Analysis
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Train a Linear Probe ===
# Problem: Train a linear probe (logistic regression) on frozen
# representations to detect if a specific feature is encoded.

def exercise_1():
    """Train a linear probe to detect linguistic features in representations."""
    print("=" * 60)
    print("Exercise 1: Train a Linear Probe")
    print("=" * 60)

    np.random.seed(42)

    # Simulate hidden representations from a language model
    # Task: probe whether "number" (singular vs plural) is encoded
    n_samples = 200
    d_repr = 8  # representation dimension

    # Generate synthetic representations where number info is partially encoded
    # Singular nouns: representations tend to have positive dim 2 and dim 5
    # Plural nouns: representations tend to have negative dim 2 and dim 5
    representations = np.random.randn(n_samples, d_repr) * 0.5
    labels = np.random.randint(0, 2, n_samples)  # 0=singular, 1=plural

    # Inject signal into dimensions 2 and 5
    signal_strength = 1.0
    for i in range(n_samples):
        if labels[i] == 0:  # singular
            representations[i, 2] += signal_strength
            representations[i, 5] += signal_strength * 0.5
        else:  # plural
            representations[i, 2] -= signal_strength
            representations[i, 5] -= signal_strength * 0.5

    # Split into train and test
    n_train = 150
    X_train, X_test = representations[:n_train], representations[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    # Logistic regression (gradient descent implementation)
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def train_logistic(X, y, lr=0.1, epochs=200):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        losses = []
        for epoch in range(epochs):
            z = X @ w + b
            pred = sigmoid(z)
            loss = -np.mean(y * np.log(pred + 1e-10) +
                            (1 - y) * np.log(1 - pred + 1e-10))
            dw = X.T @ (pred - y) / n
            db = np.mean(pred - y)
            w -= lr * dw
            b -= lr * db
            if epoch % 50 == 0:
                losses.append(loss)
        return w, b, losses

    w, b, losses = train_logistic(X_train, y_train)

    # Evaluate
    train_pred = (sigmoid(X_train @ w + b) > 0.5).astype(int)
    test_pred = (sigmoid(X_test @ w + b) > 0.5).astype(int)
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    print(f"\n  Probing task: Singular vs Plural number detection")
    print(f"  Representation dim: {d_repr}")
    print(f"  Train samples: {n_train}, Test samples: {n_samples - n_train}")
    print(f"\n  Training loss progression: {[f'{l:.4f}' for l in losses]}")
    print(f"\n  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy:  {test_acc:.2%}")

    print(f"\n  Learned probe weights:")
    for dim in range(d_repr):
        bar = "#" * int(abs(w[dim]) * 10)
        sign = "+" if w[dim] > 0 else "-"
        print(f"    dim {dim}: {w[dim]:+.4f}  {sign}{bar}")

    print(f"\n  Interpretation: Dimensions with large |weight| encode number info.")
    print(f"  Dims 2 and 5 have the largest weights, confirming the signal.")
    print(f"  High probe accuracy ({test_acc:.0%}) means the representation")
    print(f"  does encode grammatical number information.")


# === Exercise 2: Compute Selectivity Score ===
# Problem: Compute the selectivity score (Hewitt & Liang, 2019) to
# ensure the probe is not just memorizing. Compare probe accuracy
# on real labels vs random (control) labels.

def exercise_2():
    """Compute selectivity score to validate probe results."""
    print("\n" + "=" * 60)
    print("Exercise 2: Selectivity Score")
    print("=" * 60)

    np.random.seed(42)

    n_samples = 300
    d_repr = 10

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def train_and_eval(X_train, y_train, X_test, y_test, lr=0.1, epochs=300):
        n, d = X_train.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(epochs):
            z = X_train @ w + b
            pred = sigmoid(z)
            dw = X_train.T @ (pred - y_train) / n
            db = np.mean(pred - y_train)
            w -= lr * dw
            b -= lr * db
        test_pred = (sigmoid(X_test @ w + b) > 0.5).astype(int)
        return np.mean(test_pred == y_test)

    # Generate representations with encoded linguistic feature
    representations = np.random.randn(n_samples, d_repr) * 0.5
    real_labels = np.random.randint(0, 2, n_samples)

    # Encode feature in dims 3 and 7
    for i in range(n_samples):
        if real_labels[i] == 1:
            representations[i, 3] += 0.8
            representations[i, 7] -= 0.6
        else:
            representations[i, 3] -= 0.8
            representations[i, 7] += 0.6

    n_train = 220
    X_train = representations[:n_train]
    X_test = representations[n_train:]

    # Probe with real labels
    y_train_real = real_labels[:n_train]
    y_test_real = real_labels[n_train:]
    acc_real = train_and_eval(X_train, y_train_real, X_test, y_test_real)

    # Probe with random (control) labels
    n_control_runs = 5
    acc_controls = []
    for run in range(n_control_runs):
        random_labels = np.random.randint(0, 2, n_samples)
        y_train_rand = random_labels[:n_train]
        y_test_rand = random_labels[n_train:]
        acc_ctrl = train_and_eval(X_train, y_train_rand, X_test, y_test_rand)
        acc_controls.append(acc_ctrl)

    avg_control_acc = np.mean(acc_controls)

    # Selectivity = real_accuracy - control_accuracy
    selectivity = acc_real - avg_control_acc

    print(f"\n  Probe accuracy on real labels:    {acc_real:.2%}")
    print(f"  Probe accuracy on random labels:  "
          f"{avg_control_acc:.2%} (avg over {n_control_runs} runs)")
    print(f"  Individual control runs: {[f'{a:.2%}' for a in acc_controls]}")
    print(f"\n  Selectivity score: {selectivity:.2%}")

    if selectivity > 0.1:
        verdict = "HIGH selectivity: probe captures genuine linguistic info"
    elif selectivity > 0.02:
        verdict = "MODERATE selectivity: some genuine info, but caution needed"
    else:
        verdict = "LOW selectivity: probe may be memorizing, not detecting features"

    print(f"  Verdict: {verdict}")
    print(f"\n  Hewitt & Liang (2019) recommend selectivity > 0 to ensure")
    print(f"  the probe detects a real feature, not just exploiting the")
    print(f"  expressiveness of the probe architecture.")


# === Exercise 3: CKA Between Two Representation Matrices ===
# Problem: Compute Centered Kernel Alignment (CKA) to compare
# similarity between representations from two layers/models.

def exercise_3():
    """Implement CKA between two representation matrices."""
    print("\n" + "=" * 60)
    print("Exercise 3: Centered Kernel Alignment (CKA)")
    print("=" * 60)

    np.random.seed(42)

    n_samples = 50
    d1 = 6  # dimension of representation space 1
    d2 = 8  # dimension of representation space 2

    # Generate representations
    # X: original representation
    X = np.random.randn(n_samples, d1)

    # Y1: linear transformation of X (should have high CKA)
    transform = np.random.randn(d1, d2) * 0.5
    Y1 = X @ transform + np.random.randn(n_samples, d2) * 0.1

    # Y2: random, unrelated representation (should have low CKA)
    Y2 = np.random.randn(n_samples, d2)

    # Y3: nonlinear transformation of X (moderate CKA)
    Y3 = np.tanh(X @ np.random.randn(d1, d2) * 0.5)

    def linear_cka(X, Y):
        """Compute linear CKA between two representation matrices.

        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        """
        # Center the representations
        X_c = X - X.mean(axis=0)
        Y_c = Y - Y.mean(axis=0)

        # Compute Gram matrices (linear kernel)
        # Using the dual form for CKA with linear kernels
        XtX = X_c.T @ X_c
        YtY = Y_c.T @ Y_c
        XtY = X_c.T @ Y_c

        # HSIC estimates
        hsic_xy = np.sum(XtY ** 2)
        hsic_xx = np.sum(XtX ** 2)
        hsic_yy = np.sum(YtY ** 2)

        cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
        return cka

    pairs = [
        ("X vs Y1 (linear transform)", X, Y1),
        ("X vs Y2 (random/unrelated)", X, Y2),
        ("X vs Y3 (nonlinear transform)", X, Y3),
        ("X vs X (identity)", X, X),
    ]

    print(f"\n  CKA measures representation similarity invariant to")
    print(f"  linear transformations and isotropic scaling.\n")
    print(f"  {'Comparison':<35} {'CKA Score':<12} {'Interpretation':<25}")
    print("  " + "-" * 72)

    for name, A, B in pairs:
        cka = linear_cka(A, B)
        if cka > 0.9:
            interp = "Very similar"
        elif cka > 0.5:
            interp = "Moderately similar"
        elif cka > 0.2:
            interp = "Weakly similar"
        else:
            interp = "Dissimilar"
        print(f"  {name:<35} {cka:<12.4f} {interp:<25}")

    print(f"\n  Key properties of CKA:")
    print(f"  1. Invariant to orthogonal transforms and isotropic scaling")
    print(f"  2. CKA(X,X) = 1.0 (identity)")
    print(f"  3. CKA ~ 0 for unrelated representations")
    print(f"  4. Can compare layers with different dimensions")
    print(f"  5. Linear CKA captures shared linear structure;")
    print(f"     kernel CKA (with RBF) captures nonlinear similarity too")


# === Exercise 4: Logit Lens Applied to a Simple Model ===
# Problem: Implement the logit lens technique — project intermediate
# representations through the output layer to see what the model
# would predict at each layer.

def exercise_4():
    """Apply logit lens to a simple feedforward model."""
    print("\n" + "=" * 60)
    print("Exercise 4: Logit Lens")
    print("=" * 60)

    np.random.seed(42)

    # Simulate a 4-layer feedforward network for classification
    # Input -> Layer1 -> Layer2 -> Layer3 -> Layer4 -> Output
    d_hidden = 8
    n_classes = 4
    class_names = ["cat", "dog", "bird", "fish"]

    # Create layer weights (simulate a trained network)
    W1 = np.random.randn(d_hidden, d_hidden) * 0.4
    W2 = np.random.randn(d_hidden, d_hidden) * 0.4
    W3 = np.random.randn(d_hidden, d_hidden) * 0.4
    W4 = np.random.randn(d_hidden, d_hidden) * 0.4

    # Output projection (unembedding)
    W_out = np.random.randn(n_classes, d_hidden) * 0.5

    # Bias the output layer so "cat" is the correct answer for our input
    W_out[0] = np.array([0.8, 0.3, -0.2, 0.5, -0.1, 0.4, 0.2, 0.1])

    def softmax(z):
        z_shifted = z - z.max()
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum()

    def relu(x):
        return np.maximum(0, x)

    # Input representation (e.g., from embedding layer)
    x = np.random.randn(d_hidden) * 0.3
    x[0] = 1.0   # seed some signal

    # Forward pass, collecting intermediate representations
    h0 = x                            # input
    h1 = relu(W1 @ h0)               # after layer 1
    h2 = relu(W2 @ h1)               # after layer 2
    h3 = relu(W3 @ h2)               # after layer 3
    h4 = relu(W4 @ h3)               # after layer 4
    logits = W_out @ h4               # final logits
    final_probs = softmax(logits)

    layers = [
        ("Input (h0)", h0),
        ("After Layer 1 (h1)", h1),
        ("After Layer 2 (h2)", h2),
        ("After Layer 3 (h3)", h3),
        ("After Layer 4 (h4)", h4),
    ]

    print(f"\n  Logit lens: project each layer's hidden state through")
    print(f"  the output (unembedding) layer to see early predictions.\n")
    print(f"  {'Layer':<25} ", end="")
    for name in class_names:
        print(f"{name:<10} ", end="")
    print(f"{'Predicted':<10}")
    print("  " + "-" * 75)

    for layer_name, h in layers:
        # Logit lens: apply output projection to intermediate representation
        layer_logits = W_out @ h
        layer_probs = softmax(layer_logits)
        predicted_class = class_names[np.argmax(layer_probs)]

        print(f"  {layer_name:<25} ", end="")
        for p in layer_probs:
            print(f"{p:<10.3f} ", end="")
        print(f"{predicted_class:<10}")

    print(f"\n  Final output logits:  {logits}")
    print(f"  Final probabilities:  {final_probs}")
    print(f"  Final prediction:     {class_names[np.argmax(final_probs)]}")

    print(f"\n  Interpretation:")
    print(f"  The logit lens reveals how the model's 'belief' evolves")
    print(f"  through layers. Early layers may be uncertain or predict")
    print(f"  incorrectly, while later layers converge to the final answer.")
    print(f"  This technique was popularized by nostalgebraist (2020) for")
    print(f"  analyzing GPT-2 and shows that transformers often arrive at")
    print(f"  the correct token prediction gradually across layers.")
    print(f"\n  Limitations:")
    print(f"  - Assumes intermediate representations are in the same 'space'")
    print(f"    as the final layer (may not hold if layers learn different bases)")
    print(f"  - The 'tuned lens' (Belrose et al., 2023) adds a learned affine")
    print(f"    transform per layer to account for representational drift.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

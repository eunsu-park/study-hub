#!/bin/bash
# Exercises for Lesson 15: Linear Algebra in Machine Learning
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Feature Matrix and Linear Regression ===
# Problem: Construct the design matrix for polynomial features (degree 3)
# from 1D data and solve the normal equations.
exercise_1() {
    echo "=== Exercise 1: Feature Matrix and Linear Regression ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
x = np.linspace(0, 2, 20)
y = 1 + 2*x - 0.5*x**2 + 0.3*x**3 + np.random.randn(20) * 0.2

# Design matrix: [1, x, x^2, x^3]
X = np.column_stack([np.ones_like(x), x, x**2, x**3])
print(f"Design matrix X: {X.shape}")
print(f"First 5 rows:\n{np.round(X[:5], 4)}")

# Normal equations: w = (X^T X)^{-1} X^T y
w = np.linalg.solve(X.T @ X, X.T @ y)
print(f"\nEstimated coefficients: {np.round(w, 4)}")
print(f"True coefficients:      [1.0, 2.0, -0.5, 0.3]")

# Residual
y_pred = X @ w
residual = np.linalg.norm(y - y_pred)
r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
print(f"\nResidual norm: {residual:.4f}")
print(f"R^2: {r_squared:.4f}")
SOLUTION
}

# === Exercise 2: Kernel Matrix ===
# Problem: Construct the RBF kernel matrix for 5 data points and verify
# it is positive semidefinite.
exercise_2() {
    echo "=== Exercise 2: Kernel Matrix ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
X = np.random.randn(5, 2)
gamma = 0.5

# RBF (Gaussian) kernel: K(x,y) = exp(-gamma * ||x-y||^2)
n = X.shape[0]
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)

print(f"Data points:\n{np.round(X, 4)}")
print(f"\nKernel matrix K:\n{np.round(K, 4)}")

# Verify PSD: all eigenvalues >= 0
evals = np.linalg.eigvalsh(K)
print(f"\nEigenvalues: {np.round(evals, 6)}")
print(f"Positive semidefinite: {np.all(evals >= -1e-10)}")
print(f"Symmetric: {np.allclose(K, K.T)}")
print(f"K[i,i] = 1 for all i: {np.allclose(np.diag(K), 1)}")
SOLUTION
}

# === Exercise 3: Word Embedding Similarity ===
# Problem: Compute cosine similarities between word vectors and find
# the closest pair.
exercise_3() {
    echo "=== Exercise 3: Word Embedding Similarity ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Simulated word embeddings (dimension 5)
np.random.seed(42)
words = ["king", "queen", "man", "woman", "child"]
# Make king-queen similar, man-woman similar
embeddings = {
    "king":  np.array([0.8, 0.2, 0.9, 0.1, 0.5]),
    "queen": np.array([0.7, 0.3, 0.1, 0.9, 0.5]),
    "man":   np.array([0.6, 0.1, 0.8, 0.1, 0.3]),
    "woman": np.array([0.5, 0.2, 0.1, 0.8, 0.3]),
    "child": np.array([0.3, 0.9, 0.4, 0.4, 0.7]),
}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Cosine similarity matrix:")
print(f"{'':>8}", end="")
for w in words:
    print(f"{w:>8}", end="")
print()

for w1 in words:
    print(f"{w1:>8}", end="")
    for w2 in words:
        sim = cosine_sim(embeddings[w1], embeddings[w2])
        print(f"{sim:8.3f}", end="")
    print()

# Analogy: king - man + woman ~= queen
result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
print(f"\nking - man + woman:")
best_word = max(words, key=lambda w: cosine_sim(result, embeddings[w]))
print(f"Closest word: {best_word} (sim={cosine_sim(result, embeddings[best_word]):.4f})")
SOLUTION
}

# === Exercise 4: Neural Network Forward Pass ===
# Problem: Implement a 2-layer neural network forward pass as matrix
# operations: h = relu(W1 x + b1), y = W2 h + b2.
exercise_4() {
    echo "=== Exercise 4: Neural Network Forward Pass ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)

# Network: input(3) -> hidden(4) -> output(2)
d_in, d_hidden, d_out = 3, 4, 2
batch_size = 5

# Initialize weights
W1 = np.random.randn(d_hidden, d_in) * 0.5
b1 = np.zeros(d_hidden)
W2 = np.random.randn(d_out, d_hidden) * 0.5
b2 = np.zeros(d_out)

# Input batch
X = np.random.randn(batch_size, d_in)

# Forward pass
def relu(x):
    return np.maximum(0, x)

# Layer 1: h = relu(X @ W1^T + b1)
z1 = X @ W1.T + b1  # (batch, d_hidden)
h = relu(z1)

# Layer 2: y = h @ W2^T + b2
y = h @ W2.T + b2  # (batch, d_out)

print(f"Input X: {X.shape}")
print(f"W1: {W1.shape}, b1: {b1.shape}")
print(f"Hidden h: {h.shape}")
print(f"W2: {W2.shape}, b2: {b2.shape}")
print(f"Output y: {y.shape}")
print(f"\nOutput:\n{np.round(y, 4)}")

# Count parameters
n_params = W1.size + b1.size + W2.size + b2.size
print(f"\nTotal parameters: {n_params}")
print(f"  W1: {W1.size}, b1: {b1.size}")
print(f"  W2: {W2.size}, b2: {b2.size}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 15: Linear Algebra in Machine Learning"
echo "====================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4

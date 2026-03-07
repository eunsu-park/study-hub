#!/bin/bash
# Exercises for Lesson 08: Principal Component Analysis
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: PCA from Scratch ===
# Problem: Perform PCA on a 2D dataset and find the principal components.
exercise_1() {
    echo "=== Exercise 1: PCA from Scratch ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
# Generate correlated 2D data
n = 100
X = np.column_stack([
    np.random.randn(n) * 3,
    np.random.randn(n) * 1
])
# Rotate by 30 degrees
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
X = X @ R.T

# Step 1: Center
mean = np.mean(X, axis=0)
X_c = X - mean

# Step 2: Covariance matrix
C = np.cov(X_c.T)
print(f"Covariance matrix:\n{np.round(C, 4)}")

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(C)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
print(f"PC1 direction: {np.round(eigenvectors[:, 0], 4)}")
print(f"PC2 direction: {np.round(eigenvectors[:, 1], 4)}")

# Explained variance
explained = eigenvalues / np.sum(eigenvalues)
print(f"\nExplained variance: {np.round(explained, 4)}")
print(f"PC1 captures {explained[0]*100:.1f}% of variance")
SOLUTION
}

# === Exercise 2: PCA via SVD ===
# Problem: Show that PCA via eigendecomposition of covariance matrix
# gives the same result as PCA via SVD of the centered data matrix.
exercise_2() {
    echo "=== Exercise 2: PCA via SVD ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
X = np.random.randn(50, 5) @ np.diag([5, 3, 2, 1, 0.5])

# Center
X_c = X - np.mean(X, axis=0)

# Method 1: Eigendecomposition of covariance
C = np.cov(X_c.T)
eig_vals, eig_vecs = np.linalg.eigh(C)
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# Method 2: SVD of centered data
U, sigma, Vt = np.linalg.svd(X_c, full_matrices=False)
svd_eig_vals = sigma ** 2 / (X.shape[0] - 1)

print("Eigenvalues comparison:")
print(f"  Covariance eig: {np.round(eig_vals, 6)}")
print(f"  SVD (s^2/(n-1)): {np.round(svd_eig_vals, 6)}")
print(f"  Match: {np.allclose(eig_vals, svd_eig_vals)}")

# Principal components match (up to sign)
for i in range(3):
    dot = abs(np.dot(eig_vecs[:, i], Vt[i, :]))
    print(f"  PC{i+1} alignment: {dot:.6f} (1.0 = perfect)")
SOLUTION
}

# === Exercise 3: Choosing Number of Components ===
# Problem: Given a dataset with 10 features, determine how many PCs
# are needed to capture 95% of variance.
exercise_3() {
    echo "=== Exercise 3: Choosing Number of Components ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
n, d = 200, 10
# Data with 3 strong components and noise
X = np.random.randn(n, 3) @ np.random.randn(3, d) * 5 + np.random.randn(n, d) * 0.5

X_c = X - np.mean(X, axis=0)
sigma = np.linalg.svd(X_c, compute_uv=False)
eigenvalues = sigma ** 2 / (n - 1)
explained = eigenvalues / np.sum(eigenvalues)
cumulative = np.cumsum(explained)

print(f"Explained variance ratios:")
for i, (ev, cum) in enumerate(zip(explained, cumulative)):
    marker = " <-- 95%" if i == np.searchsorted(cumulative, 0.95) else ""
    print(f"  PC{i+1:2d}: {ev:.4f} (cumulative: {cum:.4f}){marker}")

k_95 = np.searchsorted(cumulative, 0.95) + 1
print(f"\nComponents for 95% variance: {k_95} out of {d}")
print(f"Dimensionality reduction: {d} -> {k_95} ({(1-k_95/d)*100:.0f}% reduction)")
SOLUTION
}

# === Exercise 4: Reconstruction Error ===
# Problem: Project 5D data onto 2 PCs and compute reconstruction error.
exercise_4() {
    echo "=== Exercise 4: Reconstruction Error ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
n, d = 100, 5
X = np.random.randn(n, d) @ np.diag([10, 5, 1, 0.5, 0.1])

X_c = X - np.mean(X, axis=0)
U, sigma, Vt = np.linalg.svd(X_c, full_matrices=False)

k = 2
# Project to k dimensions
X_projected = X_c @ Vt[:k].T  # (n, k)

# Reconstruct
X_reconstructed = X_projected @ Vt[:k]  # (n, d)

# Error
error = np.linalg.norm(X_c - X_reconstructed, 'fro')
total = np.linalg.norm(X_c, 'fro')
rel_error = error / total

print(f"Original dimensions: {d}")
print(f"Reduced dimensions: {k}")
print(f"Reconstruction error: {error:.4f}")
print(f"Relative error: {rel_error:.4f}")

# This should equal sqrt(sum of discarded eigenvalues)
theoretical = np.sqrt(np.sum(sigma[k:]**2))
print(f"\nTheoretical error: {theoretical:.4f}")
print(f"Match: {np.isclose(error, theoretical)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 08: Principal Component Analysis"
echo "==============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4

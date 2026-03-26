# Lesson 7: Singular Value Decomposition

## Learning Objectives

- Derive the SVD from the eigendecomposition of $A^T A$ and $A A^T$
- Interpret the SVD geometrically as rotation-scale-rotation
- State and apply the Eckart-Young theorem for optimal low-rank approximation
- Implement image compression using truncated SVD
- Connect SVD to the pseudoinverse, rank, norms, and condition number

---

## 1. Definition and Derivation

### 1.1 The SVD Theorem

Every matrix $A \in \mathbb{R}^{m \times n}$ (any shape, any rank) has a **singular value decomposition**:

$$A = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal (columns are **left singular vectors**)
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix of **singular values** $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{\min(m,n)} \ge 0$
- $V \in \mathbb{R}^{n \times n}$ is orthogonal (columns are **right singular vectors**)

### 1.2 Derivation from Eigendecomposition

Consider the symmetric positive semi-definite matrices $A^T A$ and $A A^T$:

- $A^T A = V \Sigma^T \Sigma V^T$ -- eigendecomposition gives $V$ and $\sigma_i^2$
- $A A^T = U \Sigma \Sigma^T U^T$ -- eigendecomposition gives $U$

The singular values are the square roots of the eigenvalues of $A^T A$ (or $A A^T$):

$$\sigma_i = \sqrt{\lambda_i(A^T A)}$$

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3, 2, 2],
              [2, 3, -2]], dtype=float)

# Full SVD
U, s, Vt = np.linalg.svd(A)
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, s)

print(f"U ({U.shape}):\n{np.round(U, 4)}")
print(f"Singular values: {s}")
print(f"V^T ({Vt.shape}):\n{np.round(Vt, 4)}")

# Verify A = U Sigma V^T
A_reconstructed = U @ Sigma @ Vt
print(f"\nU @ Sigma @ V^T:\n{np.round(A_reconstructed, 10)}")
print(f"Matches A? {np.allclose(A, A_reconstructed)}")

# Verify: singular values = sqrt(eigenvalues of A^T A)
eigvals_ATA = np.linalg.eigvalsh(A.T @ A)
print(f"\neigenvalues of A^T A: {np.sort(eigvals_ATA)[::-1]}")
print(f"sigma^2:              {s**2}")
```

### 1.3 Compact (Economy) SVD

When $m > n$, we can use the **compact SVD** $A = U_r \Sigma_r V_r^T$ where $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{n \times r}$, and $r = \mathrm{rank}(A)$.

```python
# Economy SVD
U_econ, s_econ, Vt_econ = np.linalg.svd(A, full_matrices=False)
print(f"Economy U shape: {U_econ.shape}")    # (2, 2) instead of (2, 2)
print(f"Economy V^T shape: {Vt_econ.shape}")  # (2, 3) instead of (3, 3)
```

---

## 2. Geometric Interpretation

### 2.1 SVD as Three Transformations

The SVD reveals that any linear transformation can be decomposed into three steps:

1. **Rotate** the input space (by $V^T$)
2. **Scale** along the axes (by $\Sigma$)
3. **Rotate** into the output space (by $U$)

The unit sphere in $\mathbb{R}^n$ is mapped to an ellipsoid in $\mathbb{R}^m$. The singular values give the semi-axis lengths of the ellipsoid.

```python
# Visualize SVD of a 2x2 matrix
A = np.array([[2, 1],
              [1, 3]], dtype=float)

U, s, Vt = np.linalg.svd(A)

# Unit circle
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Step 0: Original unit circle
axes[0].plot(circle[0], circle[1], 'b-')
axes[0].set_title('Unit circle')

# Step 1: After V^T rotation
step1 = Vt @ circle
axes[1].plot(step1[0], step1[1], 'g-')
axes[1].set_title(r'After $V^T$ (rotation)')

# Step 2: After Sigma scaling
step2 = np.diag(s) @ step1
axes[2].plot(step2[0], step2[1], 'orange')
axes[2].set_title(r'After $\Sigma$ (scaling)')

# Step 3: After U rotation = A @ circle
step3 = U @ step2
axes[3].plot(step3[0], step3[1], 'r-')
axes[3].set_title(r'After $U$ (rotation) = $A \cdot$ circle')

for ax in axes:
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print(f"Semi-axis lengths: {s}")
```

### 2.2 Singular Vectors as Principal Axes

- The **right singular vectors** (columns of $V$) are the principal directions in the input space
- The **left singular vectors** (columns of $U$) are the principal directions in the output space
- The singular values $\sigma_i$ give the stretching factor along each principal direction

---

## 3. Low-Rank Approximation

### 3.1 Truncated SVD

The **rank-$k$ truncated SVD** keeps only the $k$ largest singular values:

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where $U_k$ is the first $k$ columns of $U$, etc.

### 3.2 Eckart-Young Theorem

The truncated SVD gives the **best rank-$k$ approximation** to $A$ in both the Frobenius norm and the operator 2-norm:

$$A_k = \arg\min_{\mathrm{rank}(B) \le k} \|A - B\|_F$$

The approximation error is:

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

$$\|A - A_k\|_2 = \sigma_{k+1}$$

```python
# Create a matrix and approximate it
np.random.seed(42)
A = np.random.randn(10, 8)

U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"Singular values: {np.round(s, 3)}")

# Rank-k approximations
for k in [1, 2, 3, 5]:
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    error_fro = np.linalg.norm(A - A_k, 'fro')
    error_theory = np.sqrt(np.sum(s[k:]**2))
    print(f"Rank-{k}: ||A - A_k||_F = {error_fro:.4f} (theory: {error_theory:.4f})")

# Cumulative energy retained
energy = np.cumsum(s**2) / np.sum(s**2)
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(s)+1), s**2 / np.sum(s**2), alpha=0.5, label='Individual')
plt.plot(range(1, len(s)+1), energy, 'ro-', label='Cumulative')
plt.xlabel('Singular value index')
plt.ylabel('Fraction of total energy')
plt.title('Singular Value Energy Distribution')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

---

## 4. Image Compression with SVD

One of the most intuitive applications of SVD: approximate an image using fewer singular values.

```python
# Create a simple test image (or use a real one)
from scipy.datasets import face

# Use a grayscale image
try:
    img = face(gray=True).astype(float)
except:
    # Fallback: create a synthetic image
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    img = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X + Y)

print(f"Image shape: {img.shape}")

# SVD of the image
U, s, Vt = np.linalg.svd(img, full_matrices=False)
print(f"Number of singular values: {len(s)}")

# Reconstruct with different ranks
ranks = [1, 5, 20, 50, 100]
fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(20, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original\n({img.shape[0]}x{img.shape[1]})')
axes[0].axis('off')

for ax, k in zip(axes[1:], ranks):
    img_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    compression = (k * (img.shape[0] + img.shape[1] + 1)) / (img.shape[0] * img.shape[1])
    ax.imshow(img_k, cmap='gray')
    ax.set_title(f'Rank {k}\n({compression:.1%} storage)')
    ax.axis('off')

plt.suptitle('Image Compression via Truncated SVD', fontsize=14)
plt.tight_layout()
plt.show()

# Storage comparison
m, n = img.shape
for k in ranks:
    original_entries = m * n
    compressed_entries = k * (m + n + 1)
    ratio = compressed_entries / original_entries
    print(f"Rank {k}: {compressed_entries} entries ({ratio:.1%} of original)")
```

---

## 5. SVD and the Pseudoinverse

### 5.1 Moore-Penrose Pseudoinverse

The **pseudoinverse** (or Moore-Penrose inverse) $A^+$ is defined via SVD:

$$A^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ is formed by taking the reciprocal of each non-zero singular value:

$$\Sigma^+ = \mathrm{diag}(1/\sigma_1, \ldots, 1/\sigma_r, 0, \ldots, 0)^T$$

### 5.2 Properties

- If $A$ is invertible: $A^+ = A^{-1}$
- $A A^+ A = A$
- $A^+ A A^+ = A^+$
- $A^+ \mathbf{b}$ gives the minimum-norm least-squares solution to $A\mathbf{x} = \mathbf{b}$

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

# Pseudoinverse via NumPy
A_pinv = np.linalg.pinv(A)
print(f"A^+ =\n{np.round(A_pinv, 4)}")

# Verify via SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1 / s)
A_pinv_svd = Vt.T @ S_inv @ U.T
print(f"A^+ (via SVD) =\n{np.round(A_pinv_svd, 4)}")
print(f"Match? {np.allclose(A_pinv, A_pinv_svd)}")

# Verify properties
print(f"\nA @ A^+ @ A == A? {np.allclose(A @ A_pinv @ A, A)}")
print(f"A^+ @ A @ A^+ == A^+? {np.allclose(A_pinv @ A @ A_pinv, A_pinv)}")

# Least squares solution via pseudoinverse
b = np.array([1, 2, 3], dtype=float)
x_pinv = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"\nx (pseudoinverse): {x_pinv}")
print(f"x (lstsq):         {x_lstsq}")
print(f"Match? {np.allclose(x_pinv, x_lstsq)}")
```

---

## 6. SVD and Matrix Properties

### 6.1 Rank

The rank of $A$ equals the number of non-zero singular values:

$$\mathrm{rank}(A) = \#\{i : \sigma_i > 0\}$$

### 6.2 Norms

- **Operator 2-norm**: $\|A\|_2 = \sigma_1$ (largest singular value)
- **Frobenius norm**: $\|A\|_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2}$

### 6.3 Condition Number

The **condition number** measures how sensitive $A\mathbf{x} = \mathbf{b}$ is to perturbations:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_r}$$

A large condition number means the matrix is **ill-conditioned**.

```python
# Well-conditioned matrix
A_good = np.array([[1, 0], [0, 1]], dtype=float)

# Ill-conditioned matrix
A_bad = np.array([[1, 1], [1, 1.0001]], dtype=float)

for name, M in [("Well-conditioned", A_good), ("Ill-conditioned", A_bad)]:
    s = np.linalg.svd(M, compute_uv=False)
    print(f"\n{name}:")
    print(f"  Singular values: {s}")
    print(f"  Rank: {np.sum(s > 1e-10)}")
    print(f"  ||A||_2 = {s[0]:.6f}")
    print(f"  ||A||_F = {np.sqrt(np.sum(s**2)):.6f}")
    print(f"  Condition number: {s[0]/s[-1]:.2f}")

# Demonstration: ill-conditioning amplifies errors
b = np.array([2, 2.0001])
x = np.linalg.solve(A_bad, b)
print(f"\nSolution to ill-conditioned system: {x}")

b_perturbed = b + np.array([0.0001, 0])
x_perturbed = np.linalg.solve(A_bad, b_perturbed)
print(f"Perturbed solution: {x_perturbed}")
print(f"Relative change in b: {np.linalg.norm(b - b_perturbed) / np.linalg.norm(b):.6f}")
print(f"Relative change in x: {np.linalg.norm(x - x_perturbed) / np.linalg.norm(x):.6f}")
```

---

## 7. SVD as a Sum of Rank-1 Matrices

The SVD expresses $A$ as a weighted sum of rank-1 outer products:

$$A = \sum_{i=1}^r \sigma_i \, \mathbf{u}_i \mathbf{v}_i^T$$

Each term $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ is a rank-1 matrix, and the singular values determine the "importance" of each component.

```python
A = np.array([[3, 2, 2],
              [2, 3, -2]], dtype=float)

U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Build A as sum of rank-1 terms
A_sum = np.zeros_like(A)
r = len(s)

fig, axes = plt.subplots(1, r + 1, figsize=(4 * (r + 1), 3))

for i in range(r):
    rank1 = s[i] * np.outer(U[:, i], Vt[i, :])
    A_sum += rank1
    axes[i].imshow(rank1, cmap='RdBu', vmin=-4, vmax=4)
    axes[i].set_title(f'sigma_{i+1}={s[i]:.2f}\nRank-1 term')
    axes[i].axis('off')

axes[r].imshow(A_sum, cmap='RdBu', vmin=-4, vmax=4)
axes[r].set_title('Sum = A')
axes[r].axis('off')

plt.suptitle('SVD: A as sum of rank-1 matrices')
plt.tight_layout()
plt.show()

print(f"Reconstruction matches A? {np.allclose(A, A_sum)}")
```

---

## 8. Practical Considerations

### 8.1 Numerical Rank

In practice, singular values are never exactly zero due to floating-point arithmetic. The **numerical rank** is the number of singular values above a threshold (often $\epsilon \cdot \sigma_1$ where $\epsilon$ is machine epsilon).

```python
# Matrix that is "nearly" rank-deficient
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9.0001]])  # almost singular

s = np.linalg.svd(A, compute_uv=False)
print(f"Singular values: {s}")
print(f"Rank (tol=1e-10): {np.sum(s > 1e-10)}")
print(f"Rank (tol=1e-3):  {np.sum(s > 1e-3)}")
print(f"NumPy rank:       {np.linalg.matrix_rank(A)}")
```

### 8.2 Randomized SVD for Large Matrices

For very large matrices, computing the full SVD is expensive ($O(mn \min(m,n))$). Randomized SVD provides an approximate rank-$k$ decomposition much faster.

```python
from scipy.linalg import svd as scipy_svd

def randomized_svd(A, k, n_oversamples=10, n_iter=2):
    """Compute approximate rank-k SVD using randomized algorithm."""
    m, n = A.shape

    # Step 1: Random projection
    Omega = np.random.randn(n, k + n_oversamples)
    Y = A @ Omega

    # Step 2: Power iteration for better approximation
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Step 3: Orthonormalize
    Q, _ = np.linalg.qr(Y)

    # Step 4: Project and compute SVD of small matrix
    B = Q.T @ A
    U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :k], s[:k], Vt[:k, :]

# Test on a large-ish matrix
np.random.seed(42)
A_large = np.random.randn(500, 300)

# Full SVD
U_full, s_full, Vt_full = np.linalg.svd(A_large, full_matrices=False)

# Randomized SVD (rank 20)
k = 20
U_rand, s_rand, Vt_rand = randomized_svd(A_large, k)

print(f"Top {k} singular values (full):       {np.round(s_full[:k], 3)}")
print(f"Top {k} singular values (randomized): {np.round(s_rand, 3)}")
print(f"Relative error: {np.linalg.norm(s_full[:k] - s_rand) / np.linalg.norm(s_full[:k]):.6f}")
```

---

## 9. Summary

| Concept | Description |
|---------|-------------|
| SVD | $A = U\Sigma V^T$ (any matrix) |
| Singular values | $\sigma_i = \sqrt{\lambda_i(A^TA)}$, ordered $\sigma_1 \ge \sigma_2 \ge \cdots$ |
| Geometric meaning | Rotate ($V^T$) -- Scale ($\Sigma$) -- Rotate ($U$) |
| Truncated SVD | Best rank-$k$ approximation (Eckart-Young) |
| Pseudoinverse | $A^+ = V\Sigma^+ U^T$ |
| Condition number | $\kappa = \sigma_1 / \sigma_r$ |
| NumPy | `np.linalg.svd(A)` or `np.linalg.svd(A, full_matrices=False)` |

---

## Exercises

### Exercise 1: SVD by Hand

Compute the SVD of $A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$ by hand. Verify with NumPy.

### Exercise 2: Low-Rank Approximation

Create a $100 \times 100$ matrix of rank 5 (as a product of a $100 \times 5$ and a $5 \times 100$ matrix plus small noise). Use the SVD to recover the rank-5 structure. Plot the singular values and the approximation error as a function of rank.

### Exercise 3: Image Compression

Load (or create) a $256 \times 256$ grayscale image. Compress it using truncated SVD at ranks 1, 5, 10, 25, 50, and 100. For each rank, compute the compression ratio and the relative Frobenius error. Plot both.

### Exercise 4: Condition Number

Generate random $n \times n$ matrices for $n = 10, 50, 100, 200$ and compute their condition numbers. Also generate Hilbert matrices (`scipy.linalg.hilbert(n)`) for the same sizes. Compare the condition numbers and discuss the implications for solving linear systems.

### Exercise 5: Pseudoinverse

For the matrix $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$, compute $A^+$ using the SVD. Use it to find the least-squares solution to $A\mathbf{x} = [1, 2, 2]^T$. Verify that the solution minimizes $\|A\mathbf{x} - \mathbf{b}\|_2$.

---

[<< Previous: Lesson 6 - Eigenvalues and Eigenvectors](06_Eigenvalues_and_Eigenvectors.md) | [Overview](00_Overview.md) | [Next: Lesson 8 - Principal Component Analysis >>](08_Principal_Component_Analysis.md)

**License**: CC BY-NC 4.0

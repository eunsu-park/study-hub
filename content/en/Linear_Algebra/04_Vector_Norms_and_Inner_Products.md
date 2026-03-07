# Lesson 4: Vector Norms and Inner Products

## Learning Objectives

- Define and compute L1, L2, Lp, and infinity norms for vectors, and the Frobenius norm for matrices
- Understand the axioms of inner product spaces and verify the Cauchy-Schwarz inequality
- Determine orthogonality between vectors and compute the angle between them
- Apply norms and inner products to practical tasks such as distance computation and regularization
- Distinguish between different norms and know when each is appropriate

---

## 1. Vector Norms

### 1.1 What Is a Norm?

A **norm** is a function $\|\cdot\| : \mathbb{R}^n \to \mathbb{R}$ that assigns a non-negative "length" to each vector. A valid norm must satisfy three axioms:

1. **Non-negativity**: $\|\mathbf{v}\| \ge 0$, with equality if and only if $\mathbf{v} = \mathbf{0}$
2. **Homogeneity**: $\|\alpha \mathbf{v}\| = |\alpha| \, \|\mathbf{v}\|$
3. **Triangle inequality**: $\|\mathbf{u} + \mathbf{v}\| \le \|\mathbf{u}\| + \|\mathbf{v}\|$

### 1.2 The $L^p$ Norms

The general $L^p$ norm ($p \ge 1$) is defined as:

$$\|\mathbf{v}\|_p = \left( \sum_{i=1}^n |v_i|^p \right)^{1/p}$$

```python
import numpy as np
import matplotlib.pyplot as plt

v = np.array([3, -4])

# L1 norm (Manhattan / taxicab distance)
l1 = np.linalg.norm(v, ord=1)
print(f"L1 norm: {l1}")   # |3| + |-4| = 7

# L2 norm (Euclidean distance) -- the default
l2 = np.linalg.norm(v)    # or np.linalg.norm(v, ord=2)
print(f"L2 norm: {l2}")   # sqrt(9 + 16) = 5

# L3 norm
l3 = np.linalg.norm(v, ord=3)
print(f"L3 norm: {l3:.4f}")

# L-infinity norm (max absolute value)
linf = np.linalg.norm(v, ord=np.inf)
print(f"L-inf norm: {linf}")  # max(|3|, |-4|) = 4
```

### 1.3 Specific Norms in Detail

#### L1 Norm (Manhattan Norm)

$$\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$$

The L1 norm measures the "taxicab distance" -- the total distance traveled along axes. In machine learning, L1 regularization (LASSO) promotes **sparsity**: it drives some components to exactly zero.

#### L2 Norm (Euclidean Norm)

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$$

The L2 norm is the familiar straight-line distance. L2 regularization (ridge regression) penalizes large weights but does not produce sparse solutions.

#### L-infinity Norm (Max Norm)

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

Measures the largest component in absolute value. Useful in worst-case analysis and adversarial robustness.

### 1.4 Visualizing Unit Balls

The **unit ball** for a norm is the set $\{\mathbf{v} : \|\mathbf{v}\|_p \le 1\}$. Its boundary is the set of all vectors with norm exactly 1.

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

theta = np.linspace(0, 2 * np.pi, 1000)
norms = [0.5, 1, 2, np.inf]
titles = ['p = 0.5 (not a norm)', 'p = 1 (L1)', 'p = 2 (L2)', 'p = inf (L-inf)']

for ax, p, title in zip(axes, norms, titles):
    if p == np.inf:
        # Square: max(|x|, |y|) = 1
        x = np.array([-1, 1, 1, -1, -1])
        y = np.array([-1, -1, 1, 1, -1])
        ax.plot(x, y, 'b-', linewidth=2)
        ax.fill(x, y, alpha=0.2)
    else:
        # Parametric: (cos(t), sin(t)) scaled
        pts = []
        for t in theta:
            x = np.cos(t)
            y = np.sin(t)
            vec = np.array([x, y])
            norm_val = (np.abs(x)**p + np.abs(y)**p)**(1/p)
            pts.append(vec / norm_val)
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=2)
        ax.fill(pts[:, 0], pts[:, 1], alpha=0.2)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

---

## 2. Matrix Norms

### 2.1 Frobenius Norm

The **Frobenius norm** treats the matrix as a long vector and computes the L2 norm:

$$\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\mathrm{tr}(A^T A)}$$

```python
A = np.array([[1, 2],
              [3, 4]])

fro = np.linalg.norm(A, 'fro')
print(f"Frobenius norm: {fro:.4f}")
print(f"Manual: {np.sqrt(np.sum(A**2)):.4f}")
print(f"Via trace: {np.sqrt(np.trace(A.T @ A)):.4f}")
```

### 2.2 Operator Norms

The **operator norm** (or induced norm) of a matrix measures the maximum factor by which it can stretch a vector:

$$\|A\|_p = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_p}{\|\mathbf{x}\|_p} = \max_{\|\mathbf{x}\|_p = 1} \|A\mathbf{x}\|_p$$

Common cases:
- $\|A\|_1$ = maximum absolute column sum
- $\|A\|_2$ = largest singular value ($\sigma_{\max}$)
- $\|A\|_\infty$ = maximum absolute row sum

```python
A = np.array([[1, 2],
              [3, 4]])

print(f"Operator 1-norm:   {np.linalg.norm(A, 1):.4f}")
print(f"Operator 2-norm:   {np.linalg.norm(A, 2):.4f}")
print(f"Operator inf-norm: {np.linalg.norm(A, np.inf):.4f}")
print(f"Frobenius norm:    {np.linalg.norm(A, 'fro'):.4f}")

# Verify 2-norm = largest singular value
sv = np.linalg.svd(A, compute_uv=False)
print(f"Largest singular value: {sv[0]:.4f}")
```

### 2.3 Relationship Between Norms

For any $A \in \mathbb{R}^{m \times n}$:

$$\|A\|_2 \le \|A\|_F \le \sqrt{r} \, \|A\|_2$$

where $r = \mathrm{rank}(A)$.

---

## 3. Inner Products

### 3.1 Definition

An **inner product** on a real vector space $V$ is a function $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfying:

1. **Symmetry**: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. **Linearity in first argument**: $\langle \alpha\mathbf{u} + \beta\mathbf{w}, \mathbf{v} \rangle = \alpha\langle \mathbf{u}, \mathbf{v} \rangle + \beta\langle \mathbf{w}, \mathbf{v} \rangle$
3. **Positive definiteness**: $\langle \mathbf{v}, \mathbf{v} \rangle \ge 0$ with equality iff $\mathbf{v} = \mathbf{0}$

The standard inner product on $\mathbb{R}^n$ is the dot product:

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i$$

### 3.2 Weighted Inner Product

A **weighted inner product** uses a symmetric positive definite matrix $W$:

$$\langle \mathbf{u}, \mathbf{v} \rangle_W = \mathbf{u}^T W \mathbf{v}$$

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Standard inner product
standard = np.dot(u, v)
print(f"Standard: <u, v> = {standard}")  # 1*4 + 2*5 + 3*6 = 32

# Weighted inner product
W = np.array([[2, 0, 0],
              [0, 1, 0],
              [0, 0, 3]])
weighted = u @ W @ v
print(f"Weighted: <u, v>_W = {weighted}")  # 1*2*4 + 2*1*5 + 3*3*6 = 72

# Verify W is positive definite (all eigenvalues > 0)
eigvals = np.linalg.eigvals(W)
print(f"Eigenvalues of W: {eigvals} (all positive: {np.all(eigvals > 0)})")
```

### 3.3 Inner Product Induces a Norm

Every inner product induces a norm:

$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

The standard inner product induces the L2 norm.

---

## 4. Cauchy-Schwarz Inequality

### 4.1 Statement

For any inner product:

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\| \, \|\mathbf{v}\|$$

Equality holds if and only if $\mathbf{u}$ and $\mathbf{v}$ are linearly dependent (one is a scalar multiple of the other).

### 4.2 Consequences

1. **Angle is well-defined**: Since $|\cos\theta| \le 1$, the formula $\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|}$ always yields a valid angle.
2. **Triangle inequality**: The Cauchy-Schwarz inequality is used to prove the triangle inequality for the induced norm.
3. **Correlation bound**: In statistics, the correlation coefficient $\rho \in [-1, 1]$ is a consequence of Cauchy-Schwarz.

```python
# Verify Cauchy-Schwarz for random vectors
np.random.seed(42)
for _ in range(5):
    u = np.random.randn(10)
    v = np.random.randn(10)
    lhs = abs(np.dot(u, v))
    rhs = np.linalg.norm(u) * np.linalg.norm(v)
    print(f"|<u,v>| = {lhs:.4f} <= ||u||*||v|| = {rhs:.4f}: {lhs <= rhs + 1e-10}")

# Equality case: v = c * u
u = np.array([1, 2, 3])
v = 2.5 * u
lhs = abs(np.dot(u, v))
rhs = np.linalg.norm(u) * np.linalg.norm(v)
print(f"\nCollinear case: |<u,v>| = {lhs:.4f}, ||u||*||v|| = {rhs:.4f}")
print(f"Equal? {np.isclose(lhs, rhs)}")
```

---

## 5. Orthogonality

### 5.1 Definition

Two vectors $\mathbf{u}$ and $\mathbf{v}$ are **orthogonal** if their inner product is zero:

$$\langle \mathbf{u}, \mathbf{v} \rangle = 0 \quad (\mathbf{u} \perp \mathbf{v})$$

A set of vectors is **orthogonal** if every pair is orthogonal. If, in addition, every vector has unit norm, the set is **orthonormal**.

### 5.2 Angle Between Vectors

The angle $\theta$ between two non-zero vectors is:

$$\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \, \|\mathbf{v}\|}$$

```python
u = np.array([1, 0, 0])
v = np.array([0, 1, 0])
w = np.array([1, 1, 0])

def angle_between(a, b):
    """Return the angle between vectors a and b in degrees."""
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

print(f"Angle(u, v) = {angle_between(u, v):.1f} degrees")  # 90
print(f"Angle(u, w) = {angle_between(u, w):.1f} degrees")  # 45
print(f"Angle(v, w) = {angle_between(v, w):.1f} degrees")  # 45
```

### 5.3 Orthonormal Sets

An **orthonormal** set $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ satisfies:

$$\langle \mathbf{q}_i, \mathbf{q}_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

```python
# Check if a set is orthonormal
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=float)

# Gram matrix should equal identity
gram = Q.T @ Q
print(f"Gram matrix:\n{gram}")
print(f"Orthonormal? {np.allclose(gram, np.eye(3))}")

# Create an orthonormal set from arbitrary vectors (preview of Gram-Schmidt)
v1 = np.array([1, 1, 0], dtype=float)
v2 = np.array([1, 0, 1], dtype=float)

# Normalize v1
q1 = v1 / np.linalg.norm(v1)

# Orthogonalize v2 against q1
v2_orth = v2 - np.dot(v2, q1) * q1
q2 = v2_orth / np.linalg.norm(v2_orth)

print(f"\nq1 = {q1}")
print(f"q2 = {q2}")
print(f"<q1, q2> = {np.dot(q1, q2):.10f}")  # ~0
print(f"||q1|| = {np.linalg.norm(q1):.10f}")  # 1
print(f"||q2|| = {np.linalg.norm(q2):.10f}")  # 1
```

---

## 6. Cosine Similarity

**Cosine similarity** measures the cosine of the angle between two vectors, ignoring magnitude:

$$\mathrm{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|} \in [-1, 1]$$

This is widely used in NLP (document similarity), recommendation systems, and information retrieval.

```python
# Document vectors (bag-of-words representation)
doc1 = np.array([3, 1, 0, 2, 1])  # word counts
doc2 = np.array([2, 0, 1, 3, 0])
doc3 = np.array([0, 0, 0, 0, 0])
doc4 = np.array([6, 2, 0, 4, 2])  # proportional to doc1

def cosine_similarity(u, v):
    """Compute cosine similarity between u and v."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.dot(u, v) / (norm_u * norm_v)

print(f"sim(doc1, doc2) = {cosine_similarity(doc1, doc2):.4f}")
print(f"sim(doc1, doc4) = {cosine_similarity(doc1, doc4):.4f}")  # 1.0 (same direction)
print(f"sim(doc1, doc3) = {cosine_similarity(doc1, doc3):.4f}")  # 0.0 (zero vector)

# Pairwise similarity matrix
docs = np.array([doc1, doc2, doc4])
n = len(docs)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity(docs[i], docs[j])
print(f"\nPairwise cosine similarity:\n{np.round(sim_matrix, 4)}")
```

---

## 7. Distance Metrics

Norms induce **distance functions** (metrics):

$$d_p(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_p$$

### 7.1 Common Distances

| Distance | Formula | Use case |
|----------|---------|----------|
| Euclidean (L2) | $\sqrt{\sum(u_i - v_i)^2}$ | General-purpose distance |
| Manhattan (L1) | $\sum \|u_i - v_i\|$ | Grid-based distance, sparse features |
| Chebyshev (L-inf) | $\max_i \|u_i - v_i\|$ | Worst-case deviation |
| Cosine distance | $1 - \mathrm{sim}(\mathbf{u}, \mathbf{v})$ | Text/document similarity |

```python
from scipy.spatial.distance import cdist

points = np.array([[0, 0], [3, 4], [1, 1], [6, 8]])

# Compute all pairwise distances
for metric in ['euclidean', 'cityblock', 'chebyshev', 'cosine']:
    D = cdist(points, points, metric=metric)
    print(f"\n{metric} distance matrix:")
    print(np.round(D, 4))
```

---

## 8. Applications in Machine Learning

### 8.1 L1 and L2 Regularization

In machine learning, norms are used to **regularize** model parameters:

- **L1 (LASSO)**: $\mathcal{L} + \lambda \|\mathbf{w}\|_1$ -- promotes sparsity
- **L2 (Ridge)**: $\mathcal{L} + \lambda \|\mathbf{w}\|_2^2$ -- prevents large weights

```python
# Demonstrate sparsity effect of L1 vs L2
np.random.seed(42)
n_features = 20
w_true = np.zeros(n_features)
w_true[:5] = np.array([3, -2, 1.5, 0.5, -1])  # only 5 non-zero

# Simulate noisy weights
w_noisy = w_true + np.random.randn(n_features) * 0.3

# L1 proximal operator (soft thresholding)
def soft_threshold(w, lam):
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)

# L2 shrinkage
def l2_shrink(w, lam):
    return w / (1 + lam)

lam = 0.5
w_l1 = soft_threshold(w_noisy, lam)
w_l2 = l2_shrink(w_noisy, lam)

print(f"True sparsity (zeros):  {np.sum(np.abs(w_true) < 1e-10)}")
print(f"L1 sparsity (zeros):    {np.sum(np.abs(w_l1) < 1e-10)}")
print(f"L2 sparsity (zeros):    {np.sum(np.abs(w_l2) < 1e-10)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(range(n_features), w_true); axes[0].set_title('True weights')
axes[1].bar(range(n_features), w_l1); axes[1].set_title('After L1 (sparse)')
axes[2].bar(range(n_features), w_l2); axes[2].set_title('After L2 (shrunk)')
for ax in axes:
    ax.set_xlabel('Feature index'); ax.set_ylabel('Weight')
plt.tight_layout()
plt.show()
```

### 8.2 Nearest Neighbors

The $k$-nearest neighbors algorithm relies on distance computation:

```python
from collections import Counter

# Simple KNN implementation
def knn_predict(X_train, y_train, x_query, k=3, metric='euclidean'):
    """Predict the label of x_query using k-nearest neighbors."""
    if metric == 'euclidean':
        distances = np.linalg.norm(X_train - x_query, axis=1)
    elif metric == 'manhattan':
        distances = np.sum(np.abs(X_train - x_query), axis=1)

    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_idx]
    return Counter(nearest_labels).most_common(1)[0][0]

# Example
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])
x_query = np.array([5, 5])

label = knn_predict(X_train, y_train, x_query, k=3)
print(f"Predicted label for {x_query}: {label}")
```

---

## 9. Summary

| Concept | Formula | NumPy |
|---------|---------|-------|
| L1 norm | $\sum \|v_i\|$ | `np.linalg.norm(v, 1)` |
| L2 norm | $\sqrt{\sum v_i^2}$ | `np.linalg.norm(v)` |
| Lp norm | $(\sum \|v_i\|^p)^{1/p}$ | `np.linalg.norm(v, p)` |
| L-inf norm | $\max \|v_i\|$ | `np.linalg.norm(v, np.inf)` |
| Frobenius norm | $\sqrt{\sum a_{ij}^2}$ | `np.linalg.norm(A, 'fro')` |
| Inner product | $\mathbf{u}^T \mathbf{v}$ | `np.dot(u, v)` |
| Cosine similarity | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | manual |
| Orthogonality | $\mathbf{u} \cdot \mathbf{v} = 0$ | `np.dot(u, v) == 0` |

---

## Exercises

### Exercise 1: Norm Computation

For $\mathbf{v} = [1, -2, 3, -4, 5]^T$, compute $\|\mathbf{v}\|_1$, $\|\mathbf{v}\|_2$, $\|\mathbf{v}\|_3$, and $\|\mathbf{v}\|_\infty$. Verify that $\|\mathbf{v}\|_\infty \le \|\mathbf{v}\|_2 \le \|\mathbf{v}\|_1$.

### Exercise 2: Cauchy-Schwarz

For $\mathbf{u} = [1, 2, 3]^T$ and $\mathbf{v} = [4, 5, 6]^T$:
1. Verify the Cauchy-Schwarz inequality numerically.
2. Find a scalar $c$ such that equality holds for $\mathbf{u}$ and $c\mathbf{u}$.

### Exercise 3: Orthogonal Decomposition

Given $\mathbf{v} = [3, 4]^T$ and $\mathbf{u} = [1, 0]^T$, decompose $\mathbf{v}$ into a component parallel to $\mathbf{u}$ and a component orthogonal to $\mathbf{u}$. Verify that the two components are orthogonal and sum to $\mathbf{v}$.

### Exercise 4: Distance Comparison

Consider points $A = (1, 2)$, $B = (4, 6)$, $C = (7, 2)$. Compute the L1, L2, and L-infinity distances between every pair. Which metric makes $A$ closest to $B$? Which makes $A$ closest to $C$?

### Exercise 5: Unit Ball Boundaries

Write a Python program that generates and plots the unit ball boundaries for $p = 0.5, 1, 2, 4, \infty$ in $\mathbb{R}^2$. Describe how the shape changes as $p$ increases.

---

[<< Previous: Lesson 3 - Systems of Linear Equations](03_Systems_of_Linear_Equations.md) | [Overview](00_Overview.md) | [Next: Lesson 5 - Linear Transformations >>](05_Linear_Transformations.md)

**License**: CC BY-NC 4.0

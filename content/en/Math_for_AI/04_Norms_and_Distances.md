# 04. Norms and Distance Metrics

## Learning Objectives

- Understand and compute different types of vector norms (L1, L2, Lp, L∞)
- Learn the meaning and applications of matrix norms (Frobenius, spectral, nuclear)
- Understand the characteristics of various distance metrics (Euclidean, Mahalanobis, cosine)
- Learn the relationship between regularization and norms, and the geometric interpretation of L1/L2 regularization
- Understand through practice how norms and distances are utilized in machine learning
- Be able to compute and visualize norms and distances using NumPy and scikit-learn

---

## 1. Vector Norms

### 1.1 Definition of Norm

A norm on a vector space is a function $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}_+$ satisfying three properties:

1. **Positive definiteness**: $\|\mathbf{x}\| \geq 0$ and $\|\mathbf{x}\| = 0 \Leftrightarrow \mathbf{x} = \mathbf{0}$
2. **Homogeneity**: $\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$ for all $\alpha \in \mathbb{R}$
3. **Triangle inequality**: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

### 1.2 Lp Norm

The general $L_p$ norm is defined as:

$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}$$

### 1.3 L1 Norm (Manhattan Distance)

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$$

- Sum of absolute values
- Also called Manhattan distance or taxicab distance
- Used in regularization to induce sparsity

### 1.4 L2 Norm (Euclidean Distance)

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

- Most common norm
- Euclidean distance
- Differentiable and easy to work with

### 1.5 L∞ Norm (Maximum Norm)

$$\|\mathbf{x}\|_\infty = \max_i |x_i|$$

- Largest absolute value
- Obtained as the limit $p \to \infty$

### 1.6 Norm Computation Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Define vector
x = np.array([3, 4])

# Compute various norms
l1_norm = np.linalg.norm(x, ord=1)
l2_norm = np.linalg.norm(x, ord=2)
l_inf_norm = np.linalg.norm(x, ord=np.inf)

# Verify by manual computation
l1_manual = np.sum(np.abs(x))
l2_manual = np.sqrt(np.sum(x**2))
l_inf_manual = np.max(np.abs(x))

print("Vector:", x)
print(f"\nL1 norm: {l1_norm:.4f} (manual: {l1_manual:.4f})")
print(f"L2 norm: {l2_norm:.4f} (manual: {l2_manual:.4f})")
print(f"L∞ norm: {l_inf_norm:.4f} (manual: {l_inf_manual:.4f})")

# Lp norm for p = 0.5, 1, 2, 3, 10
p_values = [0.5, 1, 2, 3, 10]
norms = [np.sum(np.abs(x)**p)**(1/p) for p in p_values]

print(f"\nLp norm (various p):")
for p, norm in zip(p_values, norms):
    print(f"  L{p} = {norm:.4f}")
```

### 1.7 Unit Sphere Visualization

```python
# Visualize unit sphere ||x||_p = 1 in 2D
theta = np.linspace(0, 2*np.pi, 1000)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
p_values = [1, 2, 3, np.inf]

for ax, p in zip(axes, p_values):
    if p == np.inf:
        # L∞: square
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
        ax.plot(square[:, 0], square[:, 1], 'b-', linewidth=2)
        title = r'$L_\infty$ norm'
    else:
        # Lp: parametric curve
        # x(t) = sign(cos(t)) * |cos(t)|^(2/p)
        # y(t) = sign(sin(t)) * |sin(t)|^(2/p)
        x_coords = np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/p)
        y_coords = np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/p)
        ax.plot(x_coords, y_coords, 'b-', linewidth=2)
        title = f'$L_{p}$ norm'

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

plt.suptitle('Unit sphere: $\|x\|_p = 1$', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('unit_spheres.png', dpi=150, bbox_inches='tight')
plt.close()

print("Unit sphere visualization saved: unit_spheres.png")
```

### 1.8 Verification of Norm Properties

```python
# Verify triangle inequality
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

for p in [1, 2, np.inf]:
    norm_x = np.linalg.norm(x, ord=p)
    norm_y = np.linalg.norm(y, ord=p)
    norm_sum = np.linalg.norm(x + y, ord=p)

    print(f"\nL{p} norm triangle inequality:")
    print(f"  ||x|| = {norm_x:.4f}")
    print(f"  ||y|| = {norm_y:.4f}")
    print(f"  ||x+y|| = {norm_sum:.4f}")
    print(f"  ||x|| + ||y|| = {norm_x + norm_y:.4f}")
    print(f"  Inequality holds: {norm_sum <= norm_x + norm_y + 1e-10}")
```

## 2. Matrix Norms

### 2.1 Frobenius Norm

$$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

- Square root of sum of squared elements
- Extension of L2 norm for vectors to matrices

### 2.2 Spectral Norm

$$\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$$

- Maximum singular value
- Induced norm: $\|A\|_2 = \max_{\|\mathbf{x}\|_2=1} \|A\mathbf{x}\|_2$
- Measure of how much a matrix can stretch a vector

### 2.3 Nuclear Norm

$$\|A\|_* = \sum_i \sigma_i(A)$$

- Sum of all singular values
- Used in low-rank matrix approximation (matrix completion problems)
- Matrix version of L1 norm (induces sparsity)

### 2.4 Matrix Norm Computation

```python
import numpy as np

# Generate random matrix
A = np.random.randn(4, 3)

# Frobenius norm
frobenius = np.linalg.norm(A, ord='fro')
frobenius_manual = np.sqrt(np.sum(A**2))

# Spectral norm (maximum singular value)
spectral = np.linalg.norm(A, ord=2)
U, s, Vt = np.linalg.svd(A)
spectral_manual = s[0]

# Nuclear norm (sum of singular values)
nuclear = np.sum(s)

print("Matrix shape:", A.shape)
print(f"\nFrobenius norm: {frobenius:.4f}")
print(f"  manual:         {frobenius_manual:.4f}")
print(f"\nSpectral norm:  {spectral:.4f}")
print(f"  max singular:   {spectral_manual:.4f}")
print(f"\nNuclear norm:   {nuclear:.4f}")
print(f"  singular values: {s}")
```

### 2.5 Properties of Matrix Norms

```python
# Matrix norm inequalities
A = np.random.randn(5, 5)

frobenius = np.linalg.norm(A, ord='fro')
spectral = np.linalg.norm(A, ord=2)

# Inequality: ||A||_2 ≤ ||A||_F ≤ sqrt(rank(A)) * ||A||_2
rank_A = np.linalg.matrix_rank(A)

print("Matrix norm inequalities:")
print(f"Spectral norm:           {spectral:.4f}")
print(f"Frobenius norm:          {frobenius:.4f}")
print(f"sqrt(rank) * spectral:   {np.sqrt(rank_A) * spectral:.4f}")
print(f"\n||A||_2 ≤ ||A||_F: {spectral <= frobenius + 1e-10}")
print(f"||A||_F ≤ sqrt(r)*||A||_2: {frobenius <= np.sqrt(rank_A) * spectral + 1e-10}")
```

## 3. Distance Metrics

### 3.1 Euclidean Distance

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

The most common distance metric.

### 3.2 Mahalanobis Distance

$$d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^T \Sigma^{-1} (\mathbf{x} - \mathbf{y})}$$

- Considers covariance matrix $\Sigma$
- Accounts for correlations between variables and scale
- Useful for outlier detection

### 3.3 Cosine Similarity

$$\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$$

$$d_{\text{cosine}}(\mathbf{x}, \mathbf{y}) = 1 - \text{sim}(\mathbf{x}, \mathbf{y})$$

- Compares only direction of vectors (ignores magnitude)
- Used for text similarity and embedding comparison

### 3.4 Hamming Distance

$$d_H(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^n \mathbb{1}[x_i \neq y_i]$$

- Number of different elements
- Used for binary data and categorical data

### 3.5 Distance Metric Comparison

```python
from scipy.spatial import distance

# Sample data
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 4, 5])

# Compute various distances
euclidean = distance.euclidean(x, y)
manhattan = distance.cityblock(x, y)
chebyshev = distance.chebyshev(x, y)
cosine_dist = distance.cosine(x, y)

# Manual computation
euclidean_manual = np.linalg.norm(x - y)
manhattan_manual = np.sum(np.abs(x - y))
chebyshev_manual = np.max(np.abs(x - y))
cosine_manual = 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print("Distance metric comparison:")
print(f"Euclidean distance:  {euclidean:.4f} (manual: {euclidean_manual:.4f})")
print(f"Manhattan distance:  {manhattan:.4f} (manual: {manhattan_manual:.4f})")
print(f"Chebyshev distance:  {chebyshev:.4f} (manual: {chebyshev_manual:.4f})")
print(f"Cosine distance:     {cosine_dist:.4f} (manual: {cosine_manual:.4f})")
```

### 3.6 Mahalanobis Distance Example

```python
from scipy.spatial.distance import mahalanobis

# Sample from multivariate normal distribution
mean = np.array([0, 0])
cov = np.array([[2, 1], [1, 2]])
np.random.seed(42)
samples = np.random.multivariate_normal(mean, cov, 500)

# Euclidean distance from origin
euclidean_dists = np.linalg.norm(samples, axis=1)

# Mahalanobis distance from origin
cov_inv = np.linalg.inv(cov)
mahal_dists = np.array([mahalanobis(s, mean, cov_inv) for s in samples])

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Euclidean distance
scatter1 = axes[0].scatter(samples[:, 0], samples[:, 1], c=euclidean_dists,
                           cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0].set_title('Euclidean Distance', fontsize=12)
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Distance')

# Mahalanobis distance
scatter2 = axes[1].scatter(samples[:, 0], samples[:, 1], c=mahal_dists,
                           cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
axes[1].set_title('Mahalanobis Distance (with covariance)', fontsize=12)
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].axis('equal')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Distance')

plt.tight_layout()
plt.savefig('mahalanobis_distance.png', dpi=150)
plt.close()

print("Mahalanobis distance visualization saved: mahalanobis_distance.png")
print(f"Mean Euclidean distance: {euclidean_dists.mean():.4f}")
print(f"Mean Mahalanobis distance: {mahal_dists.mean():.4f}")
```

## 4. Regularization and Norms

### 4.1 L1 Regularization (Lasso)

Adding L1 norm penalty to loss function:

$$L(\mathbf{w}) = L_{\text{data}}(\mathbf{w}) + \lambda \|\mathbf{w}\|_1$$

- Induces sparsity (many weights become exactly 0)
- Feature selection effect

### 4.2 L2 Regularization (Ridge)

Adding L2 norm penalty to loss function:

$$L(\mathbf{w}) = L_{\text{data}}(\mathbf{w}) + \lambda \|\mathbf{w}\|_2^2$$

- Weight shrinkage
- Prevents overfitting
- Differentiable

### 4.3 Elastic Net

Combination of L1 and L2:

$$L(\mathbf{w}) = L_{\text{data}}(\mathbf{w}) + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

### 4.4 Regularization Comparison (Regression)

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data (sparse true weights)
X, y, true_coef = make_regression(
    n_samples=200, n_features=50, n_informative=10,
    noise=10, coef=True, random_state=42
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'No regularization': LinearRegression(),
    'L1 (Lasso)': Lasso(alpha=1.0),
    'L2 (Ridge)': Ridge(alpha=1.0),
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)

    results[name] = {
        'train_r2': train_score,
        'test_r2': test_score,
        'n_nonzero': n_nonzero,
        'coef': model.coef_
    }

    print(f"\n{name}:")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/50")
    print(f"  Coefficient L1 norm: {np.linalg.norm(model.coef_, ord=1):.4f}")
    print(f"  Coefficient L2 norm: {np.linalg.norm(model.coef_, ord=2):.4f}")

# Coefficient visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    axes[idx].stem(range(50), result['coef'], basefmt=' ')
    axes[idx].axhline(y=0, color='k', linewidth=0.5)
    axes[idx].set_title(f'{name}\nNon-zero coefficients: {result["n_nonzero"]}/50', fontsize=11)
    axes[idx].set_xlabel('Feature index')
    axes[idx].set_ylabel('Coefficient value')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150)
plt.close()

print("\nRegularization comparison visualization saved: regularization_comparison.png")
```

## 5. Geometry of Norms

### 5.1 Why Does L1 Produce Sparse Solutions?

This can be understood through the geometric relationship between contours and constraints.

```python
# Geometry of L1 vs L2 regularization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create grid
w1 = np.linspace(-2, 2, 300)
w2 = np.linspace(-2, 2, 300)
W1, W2 = np.meshgrid(w1, w2)

# Loss function (quadratic form): L = (w1-1)^2 + (w2-0.5)^2
w_optimal = np.array([1.0, 0.5])
L = (W1 - w_optimal[0])**2 + (W2 - w_optimal[1])**2

# L1 constraint
axes[0].contour(W1, W2, L, levels=20, cmap='viridis', alpha=0.6)
theta = np.linspace(0, 2*np.pi, 1000)
constraint_size = 1.0
# L1: diamond
l1_x = constraint_size * np.sign(np.cos(theta)) * np.abs(np.cos(theta))
l1_y = constraint_size * np.sign(np.sin(theta)) * np.abs(np.sin(theta))
axes[0].plot(l1_x, l1_y, 'r-', linewidth=3, label=r'$\|w\|_1 = 1$')
axes[0].plot([1, 0], [0, 0], 'ro', markersize=10, label='Optimal (sparse)')
axes[0].set_title('L1 Regularization: Sparse Solution', fontsize=12)
axes[0].set_xlabel('$w_1$')
axes[0].set_ylabel('$w_2$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# L2 constraint
axes[1].contour(W1, W2, L, levels=20, cmap='viridis', alpha=0.6)
# L2: circle
l2_x = constraint_size * np.cos(theta)
l2_y = constraint_size * np.sin(theta)
axes[1].plot(l2_x, l2_y, 'b-', linewidth=3, label=r'$\|w\|_2 = 1$')
# Find optimal solution (tangent point between circle and contour)
opt_w2 = w_optimal / np.linalg.norm(w_optimal) * constraint_size
axes[1].plot([opt_w2[0]], [opt_w2[1]], 'bo', markersize=10, label='Optimal (dense)')
axes[1].set_title('L2 Regularization: Dense Solution', fontsize=12)
axes[1].set_xlabel('$w_1$')
axes[1].set_ylabel('$w_2$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('l1_l2_geometry.png', dpi=150)
plt.close()

print("L1/L2 regularization geometry visualization saved: l1_l2_geometry.png")
```

### 5.2 Constraints and Optimization

Understanding through Lagrangian formulation:

**Constraint form**:
$$\min_\mathbf{w} L_{\text{data}}(\mathbf{w}) \quad \text{s.t.} \quad \|\mathbf{w}\|_p \leq t$$

**Penalty form** (equivalent):
$$\min_\mathbf{w} L_{\text{data}}(\mathbf{w}) + \lambda \|\mathbf{w}\|_p$$

Sparsity arises when the corners of L1 meet the axes.

### 5.3 3D Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

# L1/L2 constraints in 3D
fig = plt.figure(figsize=(14, 6))

# L1 ball
ax1 = fig.add_subplot(121, projection='3d')
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
U, V = np.meshgrid(u, v)

# L1 ball is a regular octahedron
r = 1.0
vertices = np.array([
    [r, 0, 0], [-r, 0, 0], [0, r, 0],
    [0, -r, 0], [0, 0, r], [0, 0, -r]
])
# Show vertices only for simplicity
ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
            c='red', s=100, alpha=0.8)
ax1.set_title('$L_1$ unit ball (octahedron)', fontsize=12)

# L2 ball (standard sphere)
ax2 = fig.add_subplot(122, projection='3d')
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))
ax2.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
ax2.set_title('$L_2$ unit ball (sphere)', fontsize=12)

for ax in [ax1, ax2]:
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_zlabel('$w_3$')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('l1_l2_3d.png', dpi=150)
plt.close()

print("3D unit ball visualization saved: l1_l2_3d.png")
```

## 6. ML Applications

### 6.1 Distance Selection in k-NN

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generate data
X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Train k-NN with various distance metrics
distances = {
    'Euclidean (L2)': 'euclidean',
    'Manhattan (L1)': 'manhattan',
    'Chebyshev (L∞)': 'chebyshev',
    'Minkowski (p=3)': 'minkowski'
}

print("k-NN distance metric comparison (5-fold CV):")
for name, metric in distances.items():
    if metric == 'minkowski':
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3)
    else:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)

    scores = cross_val_score(knn, X, y, cv=5)
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 6.2 Embedding Similarity (Cosine vs Euclidean)

```python
# Simulate text embeddings
np.random.seed(42)
n_docs = 100
embed_dim = 50

# Document embeddings (normalized to unit vectors)
embeddings = np.random.randn(n_docs, embed_dim)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Query document
query = embeddings[0]

# Euclidean distance
euclidean_dists = np.linalg.norm(embeddings - query, axis=1)

# Cosine similarity
cosine_sims = embeddings @ query
cosine_dists = 1 - cosine_sims

# Top-10 most similar documents
top_k = 10
euclidean_top = np.argsort(euclidean_dists)[1:top_k+1]
cosine_top = np.argsort(cosine_dists)[1:top_k+1]

print("Embedding similarity comparison:")
print(f"Top {top_k} by Euclidean distance: {euclidean_top}")
print(f"Top {top_k} by cosine distance:    {cosine_top}")
print(f"Overlapping documents: {len(set(euclidean_top) & set(cosine_top))}/{top_k}")

# Correlation
from scipy.stats import spearmanr
corr, p_value = spearmanr(euclidean_dists, cosine_dists)
print(f"\nEuclidean-cosine distance correlation: {corr:.4f} (p={p_value:.4e})")
```

### 6.3 Batch Normalization and Gradient Norm

```python
import torch
import torch.nn as nn

# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self, use_batchnorm=False):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.bn1 = nn.BatchNorm1d(50) if use_batchnorm else nn.Identity()
        self.fc2 = nn.Linear(50, 20)
        self.bn2 = nn.BatchNorm1d(20) if use_batchnorm else nn.Identity()
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# Generate data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Compare models
models = {
    'Without BatchNorm': SimpleNet(use_batchnorm=False),
    'With BatchNorm': SimpleNet(use_batchnorm=True)
}

print("Effect of batch normalization on gradient norm:\n")
for name, model in models.items():
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    output = model(X)
    loss = nn.MSELoss()(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Compute gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print(f"{name}:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient L2 norm: {total_norm:.4f}\n")
```

### 6.4 Gradient Clipping

```python
# Prevent gradient explosion
def train_step_with_clipping(model, X, y, optimizer, max_norm=1.0):
    """Training step with gradient clipping"""
    optimizer.zero_grad()
    output = model(X)
    loss = nn.MSELoss()(output, y)
    loss.backward()

    # Compute gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # Clipping
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
        clipped_norm = max_norm
    else:
        clipped_norm = total_norm

    optimizer.step()
    return loss.item(), total_norm, clipped_norm

# Test
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss, orig_norm, clipped_norm = train_step_with_clipping(
    model, X, y, optimizer, max_norm=1.0
)

print("Gradient clipping:")
print(f"Original gradient norm: {orig_norm:.4f}")
print(f"Norm after clipping:    {clipped_norm:.4f}")
print(f"Clipping occurred:      {orig_norm > 1.0}")
```

## Practice Problems

### Problem 1: Proof of Norm Properties
For $p \geq 1$, prove that the $L_p$ norm satisfies the triangle inequality:

$$\|\mathbf{x} + \mathbf{y}\|_p \leq \|\mathbf{x}\|_p + \|\mathbf{y}\|_p$$

Hint: Use Minkowski's inequality or verify numerically.

### Problem 2: Mahalanobis Distance Implementation
Write a function that computes the Mahalanobis distance for a given dataset and detects outliers.
Set the threshold using the chi-squared distribution.

### Problem 3: Regularization Path
Track the order in which coefficients become zero as $\lambda$ increases from 0 in Lasso regression.
Use scikit-learn's `lasso_path` or implement it yourself.

### Problem 4: Norm-Preserving Transformations
For an orthogonal matrix $Q$ (i.e., $Q^T Q = I$), prove:

$$\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$$

Verify numerically and also check for the Frobenius norm.

### Problem 5: Distance-Based Clustering
Perform k-means clustering on given 2D data using Euclidean, Manhattan, and cosine distances separately, and compare the results.
Evaluate cluster quality using the silhouette score.

## References

### Online Resources
- [Norms and Distance Metrics](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) - Bishop's PRML textbook
- [Regularization in Machine Learning](https://scikit-learn.org/stable/modules/linear_model.html#regularization) - scikit-learn documentation
- [Understanding L1 vs L2 Regularization](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)

### Textbooks
- Boyd & Vandenberghe, *Convex Optimization*, Chapter 3 (Norms)
- Hastie et al., *The Elements of Statistical Learning*, Chapter 3.4 (Shrinkage Methods)
- Murphy, *Machine Learning: A Probabilistic Perspective*, Chapter 13.3

### Papers
- Tibshirani, *Regression Shrinkage and Selection via the Lasso* (JRSS 1996)
- Zou & Hastie, *Regularization and Variable Selection via the Elastic Net* (JRSS 2005)
- Mahalanobis, *On the Generalized Distance in Statistics* (1936)

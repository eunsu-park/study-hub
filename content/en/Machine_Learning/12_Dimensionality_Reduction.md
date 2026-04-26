# Dimensionality Reduction

**Previous**: [Clustering](./11_Clustering.md) | **Next**: [Pipelines and Practice](./13_Pipelines_and_Practice.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the curse of dimensionality and why high-dimensional data harms model performance
2. Distinguish between feature selection and feature extraction approaches
3. Implement PCA and interpret principal components, explained variance ratios, and scree plots
4. Apply t-SNE and UMAP for non-linear visualization and compare their behavior with PCA
5. Choose the number of PCA components using cumulative explained variance thresholds
6. Implement filter, wrapper, and embedded feature selection methods using scikit-learn
7. Evaluate the impact of dimensionality reduction on downstream classification performance

---

Real-world datasets routinely contain dozens, hundreds, or even thousands of features -- yet most of the useful information often lies along a much smaller number of directions. Dimensionality reduction lets you find those directions, stripping away noise and redundancy so that models train faster, visualizations become meaningful, and the curse of dimensionality no longer cripples your algorithms. Mastering these techniques is a prerequisite for working with any high-dimensional data, from genomics to computer vision to NLP embeddings.

> **Shadows on the wall.** Imagine shining a flashlight on a complex 3D sculpture -- the shadow on the wall is a 2D representation that captures the sculpture's essential shape, though some detail is lost. PCA works the same way: it finds the "viewing angle" (principal components) that casts the most informative shadow when projecting high-dimensional data down to fewer dimensions. The first principal component is the direction that preserves the most variance -- the angle that makes the shadow most recognizable.

---

## Theory & Principles

The three families of dimensionality reduction in this lesson — PCA, t-SNE, UMAP — answer three different questions about high-dimensional data. PCA preserves global linear structure (variance). t-SNE preserves local neighborhoods at the cost of global geometry. UMAP balances the two with a fuzzy-topology framework. Knowing which question each one answers is what tells you which to use.

### A. PCA: Eigendecomposition of the Covariance Matrix

PCA finds the orthogonal directions along which the data has maximum variance. Center the data so each column has mean zero, stack into `X ∈ ℝ^{N×p}`. The empirical covariance matrix is

```
S = (1/(N-1)) · XᵀX        ∈ ℝ^{p×p}
```

PCA solves: find unit vector `v` that maximizes the variance of the projection `Xv`. Variance of `Xv` is `vᵀSv`, and maximizing `vᵀSv` subject to `‖v‖ = 1` gives (via Lagrange multipliers) the eigenvalue problem:

```
S v = λ v
```

The principal components `v_1, v_2, ..., v_p` are the eigenvectors of `S`, ordered by eigenvalue magnitude. The eigenvalue `λ_k` *is* the variance of the data projected onto `v_k`. Picking the top `r` eigenvectors gives the projection that preserves the most variance for any chosen rank `r`.

### A.1 PCA via SVD: a Numerically Better Path

Forming `S = XᵀX` squares the condition number — bad for numerical stability when features are nearly collinear. The **singular value decomposition** of `X` gives the same answer without forming `S`:

```
X = U Σ Vᵀ          (U: N×p, Σ: diagonal, V: p×p)
```

Then `XᵀX = V Σ² Vᵀ`, so the eigenvectors of `S` are the columns of `V` (the right singular vectors), and the eigenvalues are `σ_k² / (N-1)`. Modern PCA implementations (including scikit-learn's) use SVD directly. Cost is `O(min(N²p, Np²))`.

### A.2 What PCA Preserves and What It Loses

The fraction of variance preserved by the top `r` components is:

```
explained_variance_ratio(r) = Σ_{k=1..r} λ_k  /  Σ_{k=1..p} λ_k
```

This is the standard graph for picking `r` — plot it cumulatively and find the elbow.

PCA assumes the **interesting structure is linear and aligned with maximum variance**. Both assumptions can fail:
- A nonlinear manifold (e.g., a Swiss roll) is unrolled by PCA into a flat blob — the manifold structure is destroyed.
- Variance and "interesting" can disagree — a class label can vary along a low-variance direction that PCA discards.

Standardize features before PCA when they are on different scales (otherwise the largest-magnitude feature dominates the variance).

### B. t-SNE: Probabilistic Local Neighborhood Preservation

t-SNE (van der Maaten & Hinton, 2008) gives up on global geometry and instead preserves local neighborhoods. The recipe:

**1. Build a probability distribution over neighbors in high-D.** For each pair `(i, j)`, define a conditional probability that `j` is `i`'s neighbor:

```
p_{j|i} = exp(-‖x_i - x_j‖² / 2σ_i²)  /  Σ_{k ≠ i} exp(-‖x_i - x_k‖² / 2σ_i²)
p_{ij}  = (p_{j|i} + p_{i|j}) / (2N)        ← symmetrize
```

The bandwidth `σ_i` is set per-point to make the entropy of `p_{·|i}` equal `log(perplexity)` — `perplexity` is roughly the effective number of neighbors (5–50 typical). Each point thus has a similarity distribution adapted to local density.

**2. Build a probability distribution over neighbors in low-D using a heavy-tailed (Student-t) kernel.** For low-D embedding `y_i`:

```
q_{ij} = (1 + ‖y_i - y_j‖²)⁻¹  /  Σ_{k ≠ l} (1 + ‖y_k - y_l‖²)⁻¹
```

The Student-t (with 1 degree of freedom) has heavier tails than a Gaussian, which lets distant points in low-D be embedded farther apart without penalty — solving the "crowding problem" that PCA-style methods struggle with.

**3. Minimize KL divergence between the two distributions.**

```
L = KL(P ∥ Q) = Σ_{ij}  p_{ij} · log(p_{ij} / q_{ij})
```

Gradient descent on `L` produces the embedding. The asymmetry of KL — penalizing `q_{ij}` being small when `p_{ij}` is large — means t-SNE strongly preserves *near* neighbors but tolerates moving *far* neighbors arbitrarily. That is why t-SNE plots show clean local clusters but inter-cluster distances are not interpretable.

### B.1 t-SNE Caveats

- Distances *between* clusters in a t-SNE plot are meaningless; only *within*-cluster structure is faithful.
- Cluster *sizes* depend on local density, not on actual cluster volume — small dense clusters look big.
- The `perplexity` hyperparameter materially changes the result. Always inspect with multiple values (e.g., 5, 30, 50).
- Slow: `O(N²)` naively, `O(N log N)` with Barnes-Hut approximation. Above `N ≈ 50K` it gets painful.

### C. UMAP: Fuzzy Simplicial Sets

UMAP (McInnes et al., 2018) shares t-SNE's "preserve local neighborhoods" philosophy but builds it on a different mathematical foundation: **fuzzy simplicial complexes** from algebraic topology. The mechanics:

**1. Construct a fuzzy graph in high-D.** For each point, connect to its `k` nearest neighbors with edge weights given by an exponentially decaying function of distance, normalized so each point has the same "amount" of connectivity.

**2. Construct a fuzzy graph in low-D using a different kernel** with adjustable parameters `a, b` (the smoothness of the embedding distance function).

**3. Minimize the cross-entropy** between the two fuzzy graphs (an objective that resembles KL divergence but is two-sided — penalizing both attractive forces between similar points and repulsive forces between dissimilar ones).

The practical differences from t-SNE:
- **Faster**: `O(N · log N)` per epoch with stochastic gradient descent on the graph.
- **Better global structure**: the symmetric cross-entropy preserves more long-range distance information than t-SNE's asymmetric KL.
- **Reproducible**: less sensitive to initialization than t-SNE.
- **Embeddable**: a transformed test point can be added to an existing UMAP embedding (t-SNE has no such operation).

The hyperparameter `n_neighbors` plays a similar role to `perplexity` — small values emphasize local structure, large values emphasize global.

### D. PCA vs t-SNE vs UMAP: When to Use Which

| Need | Use | Why |
|------|-----|-----|
| Linear preprocessing for downstream model | PCA | Fast, exact, preserves variance |
| Visualize cluster structure in 2D/3D | t-SNE or UMAP | Local-neighborhood preservation |
| Visualize cluster structure with global distances meaningful | UMAP | Better global structure than t-SNE |
| Want to embed new test points | PCA or UMAP | Both have a transform-method analog; t-SNE does not |
| Need very fast embedding for large `N` | UMAP | Best scaling |
| Want explained-variance ratio | PCA | Eigenvalues directly give it |

A common pipeline: PCA down to ~50 dimensions first (removes noise, speeds up subsequent algorithms), then t-SNE or UMAP for 2D visualization.

### From Theory to the Code Below

- Section 2's `PCA(n_components=r)` runs the SVD-based PCA from (A) and (A.1); `pca.explained_variance_ratio_` is the formula in (A.2).
- Section 3's `TSNE(perplexity=...)` is the algorithm in (B); `perplexity` is the bandwidth-control parameter from (B).
- Section 4's `UMAP(n_neighbors=..., min_dist=...)` is the fuzzy-simplicial-set algorithm from (C).
- The standardization (`StandardScaler`) before PCA in any preprocessing step is the scale-equalization warning from (A.2).

---

## 1. Why Dimensionality Reduction?

### 1.1 Curse of Dimensionality

**Problem**: As dimensions increase:
- Data becomes sparse (most points are far apart)
- Distance metrics become meaningless
- Computational cost increases exponentially
- Overfitting risk increases

### 1.2 Benefits of Dimensionality Reduction

1. **Visualization**: Project high-D data to 2D/3D for plotting
2. **Noise Reduction**: Remove irrelevant features
3. **Faster Training**: Fewer features → faster algorithms
4. **Storage Efficiency**: Smaller datasets
5. **Multicollinearity Reduction**: Remove redundant features

### 1.3 Types of Dimensionality Reduction

| Type | Description | Examples |
|------|-------------|----------|
| **Feature Selection** | Select subset of original features | Filter, Wrapper, Embedded methods |
| **Feature Extraction** | Create new features from original ones | PCA, t-SNE, LDA |

---

## 2. Principal Component Analysis (PCA)

### 2.1 Concepts

**Goal**: Find orthogonal axes (principal components) that capture maximum variance

**Key Ideas**:
- **PC1** (1st component): Direction of maximum variance
- **PC2** (2nd component): Orthogonal to PC1, captures next most variance
- **PC3, PC4, ...**: Continue orthogonally

**Mathematical Foundation**:
- Eigenvalue decomposition of covariance matrix
- Eigenvectors → Principal components
- Eigenvalues → Variance explained

### 2.2 Basic Implementation

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Standardize (IMPORTANT for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Iris Dataset')
plt.colorbar(scatter, label='Species')
plt.show()

# Print explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

### 2.3 Choosing Number of Components

**Method 1: Explained Variance Ratio**

```python
# PCA with all components
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot cumulative explained variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.legend()
plt.grid(True)
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

**Method 2: Scree Plot** (Elbow method)

```python
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_) + 1),
         pca_full.explained_variance_, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (Variance)')
plt.title('Scree Plot')
plt.grid(True)
plt.show()
```

### 2.4 PCA with Automatic Variance Selection

```python
# Keep components that explain at least 95% variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"Original dimensions: {X_scaled.shape[1]}")
print(f"Reduced dimensions: {X_pca_95.shape[1]}")
print(f"Variance explained: {pca_95.explained_variance_ratio_.sum():.2%}")
```

### 2.5 Inverse Transform (Reconstruction)

```python
# Reduce to 2 components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Reconstruct original data
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate reconstruction error
mse = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {mse:.4f}")

# Visualize first sample
print("Original (scaled):", X_scaled[0])
print("Reconstructed:    ", X_reconstructed[0])
```

### 2.6 PCA Components (Loadings)

```python
# Get component loadings
components = pca_full.components_

# Visualize first two components
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i, ax in enumerate(axes):
    ax.bar(range(len(components[i])), components[i])
    ax.set_xlabel('Original Feature')
    ax.set_ylabel('Contribution')
    ax.set_title(f'PC{i+1} Loadings')
    ax.set_xticks(range(len(iris.feature_names)))
    ax.set_xticklabels(iris.feature_names, rotation=45)

plt.tight_layout()
plt.show()
```

---

## 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

### 3.1 Concepts

**Goal**: Preserve local structure (nearby points stay nearby in low dimensions)

**Key Differences from PCA**:
- **Non-linear** (PCA is linear)
- **Primarily for visualization** (2D/3D), not feature reduction
- **Stochastic** (different runs give different results)
- **Slower** than PCA

**Perplexity**: Balances local vs global structure (typically 5-50)

### 3.2 Basic Implementation

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load digits dataset (64 dimensions)
digits = load_digits()
X = digits.data
y = digits.target

# Apply t-SNE (WARNING: slow on large datasets)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=5)
plt.colorbar(scatter, label='Digit')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Digits Dataset')
plt.show()
```

### 3.3 Effect of Perplexity

```python
# Compare different perplexity values
perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, perplexity in zip(axes, perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X[:500])  # Use subset for speed

    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y[:500], cmap='tab10', alpha=0.6, s=10)
    ax.set_title(f'Perplexity = {perplexity}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

### 3.4 PCA → t-SNE Pipeline (Recommended)

```python
# Speed up t-SNE by first reducing dimensions with PCA
from sklearn.pipeline import Pipeline

# Pipeline: PCA (reduce to 50 components) → t-SNE (reduce to 2)
tsne_pipeline = Pipeline([
    ('pca', PCA(n_components=50)),
    ('tsne', TSNE(n_components=2, perplexity=30, random_state=42))
])

X_tsne_fast = tsne_pipeline.fit_transform(X)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne_fast[:, 0], X_tsne_fast[:, 1], c=y, cmap='tab10', alpha=0.6, s=5)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE with PCA Preprocessing')
plt.show()
```

### 3.5 t-SNE vs PCA Comparison

```python
from sklearn.datasets import load_wine

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0].set_title('PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[1].set_title('t-SNE')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

---

## 4. Other Dimensionality Reduction Techniques

### 4.1 Truncated SVD (Similar to PCA, works with sparse matrices)

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_scaled)

print(f"Explained variance ratio: {svd.explained_variance_ratio_}")
```

### 4.2 UMAP (Uniform Manifold Approximation and Projection)

**Advantages over t-SNE**:
- Faster
- Better preserves global structure
- Can be used for feature extraction (not just visualization)

```python
# pip install umap-learn
import umap

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Visualization')
plt.colorbar(label='Class')
plt.show()
```

---

## 5. Feature Selection Methods

### 5.1 Filter Methods (Statistical tests)

```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Chi-squared test (for non-negative features)
X_positive = X - X.min()  # Make all values non-negative
selector = SelectKBest(chi2, k=2)
X_selected = selector.fit_transform(X_positive, y)

print(f"Original shape: {X.shape}")
print(f"Selected shape: {X_selected.shape}")
print(f"Selected features: {selector.get_support()}")

# F-statistic (ANOVA)
selector_f = SelectKBest(f_classif, k=2)
X_selected_f = selector_f.fit_transform(X, y)
print(f"F-scores: {selector_f.scores_}")
```

### 5.2 Wrapper Methods (Recursive Feature Elimination)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rfc, n_features_to_select=2)
rfe.fit(X, y)

print(f"Selected features: {rfe.support_}")
print(f"Feature ranking: {rfe.ranking_}")
```

### 5.3 Embedded Methods (Feature Importance)

```python
# Train Random Forest
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X, y)

# Get feature importances
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [iris.feature_names[i] for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances from Random Forest')
plt.tight_layout()
plt.show()

# Select top k features
k = 2
top_k_indices = indices[:k]
X_selected = X[:, top_k_indices]
print(f"Top {k} features: {[iris.feature_names[i] for i in top_k_indices]}")
```

### 5.4 L1 Regularization (Lasso)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# L1 regularization
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso.fit(X_scaled, y)

# Select features based on L1 coefficients
selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X_scaled)

print(f"Original features: {X_scaled.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Selected feature mask: {selector.get_support()}")
```

---

## 6. Practical Application: High-Dimensional Data

### 6.1 MNIST Digits Visualization

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST (WARNING: large dataset)
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data[:5000], mnist.target[:5000]  # Use subset

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA (fast)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE with PCA preprocessing
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca_50)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5, s=1)
axes[0].set_title('PCA on MNIST')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5, s=1)
axes[1].set_title('t-SNE on MNIST')

plt.tight_layout()
plt.show()
```

### 6.2 Impact on Classification Performance

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Original data
start = time.time()
rfc = RandomForestClassifier(n_estimators=50, random_state=42)
rfc.fit(X_scaled, y)
y_pred = rfc.predict(X_scaled)
time_original = time.time() - start
acc_original = accuracy_score(y, y_pred)

# PCA-reduced data (50 components)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

start = time.time()
rfc_pca = RandomForestClassifier(n_estimators=50, random_state=42)
rfc_pca.fit(X_pca, y)
y_pred_pca = rfc_pca.predict(X_pca)
time_pca = time.time() - start
acc_pca = accuracy_score(y, y_pred_pca)

print(f"Original (784 features):")
print(f"  Accuracy: {acc_original:.4f}, Time: {time_original:.2f}s")
print(f"\nPCA (50 components):")
print(f"  Accuracy: {acc_pca:.4f}, Time: {time_pca:.2f}s")
print(f"  Speedup: {time_original/time_pca:.2f}x")
```

---

## 7. Comparison of Methods

| Method | Type | Linear | Speed | Use Case |
|--------|------|--------|-------|----------|
| **PCA** | Extraction | Yes | Fast | Feature reduction, noise reduction |
| **t-SNE** | Extraction | No | Slow | Visualization only |
| **UMAP** | Extraction | No | Medium | Visualization + feature reduction |
| **SelectKBest** | Selection | - | Fast | Quick feature selection |
| **RFE** | Selection | - | Slow | Thorough feature selection |
| **Feature Importance** | Selection | - | Medium | Interpretable feature selection |

---

## 8. Practical Tips

### 8.1 When to Use PCA

- High-dimensional data (>50 features)
- Features are correlated
- Need linear transformation
- Want to remove multicollinearity
- Need reproducible results

### 8.2 When to Use t-SNE/UMAP

- Visualization of high-D data in 2D/3D
- Need to preserve local structure
- Exploring clusters in data
- Not for feature extraction in pipelines (except UMAP)

### 8.3 Feature Selection vs Extraction

| Aspect | Feature Selection | Feature Extraction |
|--------|-------------------|-------------------|
| **Interpretability** | High (keeps original features) | Low (new features) |
| **Information Loss** | Possible (discards features) | Minimal (transforms all) |
| **Use Case** | When features have meaning | When interpretability isn't critical |

---

## 9. Exercises

### Exercise 1: PCA on Wine Dataset
Load the wine dataset, apply PCA, and determine how many components capture 90% variance.

```python
from sklearn.datasets import load_wine

# Your code here
```

### Exercise 2: t-SNE Visualization
Load the digits dataset and compare t-SNE with different perplexity values (5, 30, 50).

```python
from sklearn.datasets import load_digits

# Your code here
```

### Exercise 3: Feature Selection
Use RFE to select the best 3 features from the breast cancer dataset and evaluate performance.

```python
from sklearn.datasets import load_breast_cancer

# Your code here
```

### Exercise 4: PCA + Classification
Compare classification performance on original vs PCA-reduced data on a dataset of your choice.

```python
# Your code here
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **PCA** | Linear, maximize variance, eigenvectors, choose components by explained variance |
| **Scree Plot** | Elbow method for PCA, plot eigenvalues |
| **Inverse Transform** | Reconstruct original data from reduced dimensions |
| **t-SNE** | Non-linear, visualization, perplexity parameter, slow |
| **UMAP** | Faster than t-SNE, preserves global structure |
| **Feature Selection** | Filter (statistical), Wrapper (RFE), Embedded (importance) |
| **Standardization** | Always scale features before PCA/t-SNE |
| **Applications** | Visualization, noise reduction, faster training, multicollinearity |

**Key Takeaway**: Use PCA for linear dimensionality reduction and feature extraction. Use t-SNE/UMAP for non-linear visualization. Use feature selection when interpretability is important.

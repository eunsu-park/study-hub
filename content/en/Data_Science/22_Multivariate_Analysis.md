# 22. Multivariate Analysis

[Previous: Time Series Models](./21_Time_Series_Models.md) | [Next: Nonparametric Statistics](./23_Nonparametric_Statistics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the mathematical foundations of PCA, including eigenvalue decomposition and variance maximization
2. Implement PCA from scratch and verify results against scikit-learn
3. Determine the appropriate number of principal components using the Kaiser rule, variance criteria, and scree plots
4. Distinguish between PCA (variance-based) and Factor Analysis (latent-variable-based) and identify when each is appropriate
5. Apply Varimax rotation to improve factor interpretability
6. Compare LDA and QDA for supervised classification, including their assumptions about class covariance
7. Evaluate clustering quality using internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) and external metrics (ARI, NMI)

---

Real-world datasets rarely have just one or two variables. When dozens of features compete for your attention, it becomes impossible to visualize patterns or build stable models without first understanding how those variables relate to each other. Multivariate analysis provides the tools to reduce complexity, uncover latent structure, classify observations, and validate groupings -- turning high-dimensional noise into actionable insight.

---

> **Shadow projection.** Imagine holding a 3D wireframe object in front of a lamp. The shadow on the wall is a 2D projection that captures the object's shape -- but some angles cast more informative shadows than others. Dimensionality reduction (PCA, Factor Analysis) works the same way: it finds the "viewing angle" that preserves the most structure when projecting high-dimensional data onto fewer dimensions. The principal components are the directions that cast the most informative shadow.

## 1. Principal Component Analysis (PCA)

### 1.1 PCA Concept

**Goal**: Project high-dimensional data onto lower dimensions while preserving variance

**Principal Component**: Orthogonal direction that maximizes data variance

**Mathematical Definition**:
- First principal component: Unit vector w₁ that maximizes Var(w₁ᵀX)
- k-th principal component: Maximizes variance while orthogonal to previous components

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

np.random.seed(42)

# Intuitive understanding of PCA with 2D example
n = 200
theta = np.pi / 4
cov = [[3, 2], [2, 2]]
X_2d = np.random.multivariate_normal([0, 0], cov, n)

# Perform PCA
pca_2d = PCA()
X_pca = pca_2d.fit_transform(X_2d)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original data
ax = axes[0]
ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)
# Show principal component directions
mean = X_2d.mean(axis=0)
for i, (comp, var) in enumerate(zip(pca_2d.components_, pca_2d.explained_variance_)):
    ax.annotate('', xy=mean + 2 * np.sqrt(var) * comp, xytext=mean,
                arrowprops=dict(arrowstyle='->', color=['red', 'blue'][i], lw=2))
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Original Data and Principal Component Directions')
ax.axis('equal')
ax.grid(True, alpha=0.3)

# Principal component space
ax = axes[1]
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Principal Component Space')
ax.axis('equal')
ax.grid(True, alpha=0.3)

# Explained variance
ax = axes[2]
explained_var_ratio = pca_2d.explained_variance_ratio_
ax.bar([1, 2], explained_var_ratio, alpha=0.7)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title(f'Explained Variance: PC1={explained_var_ratio[0]:.1%}, PC2={explained_var_ratio[1]:.1%}')
ax.set_xticks([1, 2])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 1.2 PCA Theory

```python
def pca_from_scratch(X, n_components=None):
    """
    PCA implementation from scratch

    1. Center data (mean 0)
    2. Compute covariance matrix
    3. Eigenvalue decomposition
    4. Sort eigenvectors (descending eigenvalues)
    """
    # Center data
    X_centered = X - X.mean(axis=0)

    # Covariance matrix
    n = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance ratio
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]

    # Project
    X_pca = X_centered @ eigenvectors

    return {
        'components': eigenvectors.T,
        'explained_variance': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'transformed': X_pca
    }

# Verify: compare with sklearn
X_test = np.random.randn(100, 5)
result_scratch = pca_from_scratch(X_test, n_components=3)
pca_sklearn = PCA(n_components=3).fit(X_test)

print("=== PCA Implementation Verification ===")
print(f"Explained variance ratio (scratch): {result_scratch['explained_variance_ratio']}")
print(f"Explained variance ratio (sklearn): {pca_sklearn.explained_variance_ratio_}")
print("(Signs may differ but absolute values should match)")
```

### 1.3 PCA with sklearn

```python
# Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Results analysis
print("=== Iris PCA Results ===")
print(f"Original features: {X_iris.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scree plot
ax = axes[0]
ax.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7, label='Individual')
ax.plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_), 'ro-', label='Cumulative')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Scree Plot')
ax.legend()
ax.set_xticks(range(1, 5))
ax.grid(True, alpha=0.3)

# PC1 vs PC2
ax = axes[1]
for target in np.unique(y_iris):
    mask = y_iris == target
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.7,
               label=iris.target_names[target])
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA: PC1 vs PC2')
ax.legend()
ax.grid(True, alpha=0.3)

# Component loadings
ax = axes[2]
loadings = pd.DataFrame(
    pca.components_[:2].T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
loadings.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_ylabel('Loading')
ax.set_title('Principal Component Loadings')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 1.4 Determining Number of Components

```python
def determine_n_components(X, methods=['kaiser', 'variance', 'elbow']):
    """
    Methods for determining number of principal components

    1. Kaiser rule: eigenvalue > 1 (for standardized data)
    2. Variance criterion: cumulative variance >= threshold (typically 85-95%)
    3. Scree plot: elbow point
    """
    pca = PCA()
    pca.fit(X)

    results = {}

    # Kaiser rule
    if 'kaiser' in methods:
        n_kaiser = np.sum(pca.explained_variance_ > 1)
        results['kaiser'] = n_kaiser
        print(f"Kaiser rule (eigenvalue > 1): {n_kaiser} components")

    # Variance criterion
    if 'variance' in methods:
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_80 = np.argmax(cumsum >= 0.80) + 1
        n_90 = np.argmax(cumsum >= 0.90) + 1
        n_95 = np.argmax(cumsum >= 0.95) + 1
        results['variance_80'] = n_80
        results['variance_90'] = n_90
        results['variance_95'] = n_95
        print(f"Variance criterion 80%: {n_80} components")
        print(f"Variance criterion 90%: {n_90} components")
        print(f"Variance criterion 95%: {n_95} components")

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(range(1, len(pca.explained_variance_) + 1),
            pca.explained_variance_, 'bo-')
    ax.axhline(1, color='r', linestyle='--', label='Kaiser (eigenvalue=1)')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree Plot (Eigenvalues)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
            np.cumsum(pca.explained_variance_ratio_), 'go-')
    ax.axhline(0.80, color='orange', linestyle='--', label='80%')
    ax.axhline(0.90, color='r', linestyle='--', label='90%')
    ax.axhline(0.95, color='purple', linestyle='--', label='95%')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

# Test with Wine dataset
wine = load_wine()
X_wine = StandardScaler().fit_transform(wine.data)

print("=== Wine Dataset Component Selection ===")
results = determine_n_components(X_wine)
```

### 1.5 Biplot

```python
def biplot(X, y, pca, feature_names, target_names, ax=None):
    """
    PCA biplot: display observations and variables simultaneously
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    X_pca = pca.transform(X)

    # Scaling (same scale for observations and loadings)
    scale = 1 / np.max(np.abs(X_pca[:, :2]))

    # Plot observations
    for target in np.unique(y):
        mask = y == target
        ax.scatter(X_pca[mask, 0] * scale, X_pca[mask, 1] * scale,
                   alpha=0.5, label=target_names[target], s=30)

    # Loading vectors
    loadings = pca.components_[:2].T
    for i, (loading, name) in enumerate(zip(loadings, feature_names)):
        ax.arrow(0, 0, loading[0], loading[1],
                 head_width=0.05, head_length=0.03, fc='red', ec='red')
        ax.text(loading[0] * 1.1, loading[1] * 1.1, name, fontsize=9,
                ha='center', va='center')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Biplot')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    return ax

# Iris biplot
pca_iris = PCA(n_components=4).fit(X_scaled)
biplot(X_scaled, y_iris, pca_iris, feature_names, iris.target_names)
plt.show()
```

---

## 2. Factor Analysis

### 2.1 Factor Analysis vs PCA

| Aspect | PCA | Factor Analysis |
|------|-----|----------|
| **Goal** | Maximize variance | Discover latent factors |
| **Model** | Data = principal components | Observed variable = factors + error |
| **Unique variance** | None | Unique variance per variable |
| **Rotation** | Unnecessary (orthogonal) | Rotation for interpretation |
| **Use** | Dimensionality reduction | Structure discovery, survey analysis |

### 2.2 Factor Analysis Model

**Model**:
$$X_i = \mu_i + \lambda_{i1}F_1 + \lambda_{i2}F_2 + ... + \lambda_{im}F_m + \epsilon_i$$

- Fⱼ: Common factor (latent factor)
- λᵢⱼ: Factor loading
- εᵢ: Unique factor (error)

```python
from sklearn.decomposition import FactorAnalysis
from scipy.stats import zscore

# Factor analysis example
np.random.seed(42)

# Generate data with 2 latent factors
n = 300
F1 = np.random.normal(0, 1, n)  # Factor 1
F2 = np.random.normal(0, 1, n)  # Factor 2

# 6 observed variables (3 loading on each factor)
X1 = 0.8 * F1 + 0.1 * F2 + np.random.normal(0, 0.3, n)
X2 = 0.7 * F1 + 0.2 * F2 + np.random.normal(0, 0.3, n)
X3 = 0.9 * F1 + 0.0 * F2 + np.random.normal(0, 0.3, n)
X4 = 0.1 * F1 + 0.8 * F2 + np.random.normal(0, 0.3, n)
X5 = 0.2 * F1 + 0.7 * F2 + np.random.normal(0, 0.3, n)
X6 = 0.0 * F1 + 0.9 * F2 + np.random.normal(0, 0.3, n)

X_fa = np.column_stack([X1, X2, X3, X4, X5, X6])
X_fa = zscore(X_fa)  # Standardize

# Factor analysis
fa = FactorAnalysis(n_components=2, random_state=42)
F_scores = fa.fit_transform(X_fa)

print("=== Factor Analysis Results ===")
print("\nFactor Loadings:")
loadings_df = pd.DataFrame(
    fa.components_.T,
    columns=['Factor 1', 'Factor 2'],
    index=[f'X{i+1}' for i in range(6)]
)
print(loadings_df.round(3))

print(f"\nUniqueness:")
print(pd.Series(fa.noise_variance_, index=[f'X{i+1}' for i in range(6)]).round(3))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Factor loadings plot
ax = axes[0]
loadings_df.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_ylabel('Loading')
ax.set_title('Factor Loadings')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(0, color='k', linewidth=0.5)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Factor scores
ax = axes[1]
ax.scatter(F_scores[:, 0], F_scores[:, 1], alpha=0.5)
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_title('Factor Scores')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.3 Factor Rotation

```python
def varimax_rotation(loadings, n_iter=100, tol=1e-6):
    """
    Varimax rotation (orthogonal rotation)
    Maximize loading variance for easier interpretation
    """
    p, k = loadings.shape
    rotated = loadings.copy()

    for _ in range(n_iter):
        old_rotated = rotated.copy()

        for i in range(k - 1):
            for j in range(i + 1, k):
                # 2x2 rotation
                x = rotated[:, i]
                y = rotated[:, j]

                u = x**2 - y**2
                v = 2 * x * y

                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)

                phi = 0.25 * np.arctan2(D - 2 * A * B / p,
                                         C - (A**2 - B**2) / p)

                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                rotated[:, i] = x * cos_phi + y * sin_phi
                rotated[:, j] = -x * sin_phi + y * cos_phi

        if np.max(np.abs(rotated - old_rotated)) < tol:
            break

    return rotated

# Compare before and after rotation
loadings_original = fa.components_.T
loadings_rotated = varimax_rotation(loadings_original)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before rotation
ax = axes[0]
pd.DataFrame(loadings_original, columns=['F1', 'F2'],
             index=[f'X{i+1}' for i in range(6)]).plot(kind='bar', ax=ax, alpha=0.7)
ax.set_title('Before Rotation')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# After rotation
ax = axes[1]
pd.DataFrame(loadings_rotated, columns=['F1', 'F2'],
             index=[f'X{i+1}' for i in range(6)]).plot(kind='bar', ax=ax, alpha=0.7)
ax.set_title('After Varimax Rotation')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("=== Loadings After Varimax Rotation ===")
print(pd.DataFrame(loadings_rotated, columns=['Factor 1', 'Factor 2'],
                   index=[f'X{i+1}' for i in range(6)]).round(3))
```

---

## 3. Discriminant Analysis

### 3.1 LDA (Linear Discriminant Analysis)

**Goal**: Find linear combinations that maximize class separation

**Criterion**: Maximize between-class variance / within-class variance

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Iris data LDA
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_scaled, y_iris)

print("=== LDA Results ===")
print(f"Number of discriminant functions: {X_lda.shape[1]}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LDA projection
ax = axes[0]
for target in np.unique(y_iris):
    mask = y_iris == target
    ax.scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.7,
               label=iris.target_names[target])
ax.set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.1%})')
ax.set_title('LDA Projection')
ax.legend()
ax.grid(True, alpha=0.3)

# PCA vs LDA comparison
ax = axes[1]
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
for target in np.unique(y_iris):
    mask = y_iris == target
    ax.scatter(X_pca_2[mask, 0], X_pca_2[mask, 1], alpha=0.7,
               label=iris.target_names[target])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Projection (comparison)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 LDA Classifier

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_iris, test_size=0.3, random_state=42
)

# LDA classifier
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)
y_pred_lda = lda_clf.predict(X_test)

print("=== LDA Classification Performance ===")
print(f"Train accuracy: {lda_clf.score(X_train, y_train):.4f}")
print(f"Test accuracy: {lda_clf.score(X_test, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lda, target_names=iris.target_names))

# Cross-validation
cv_scores = cross_val_score(lda_clf, X_scaled, y_iris, cv=5)
print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
```

### 3.3 QDA (Quadratic Discriminant Analysis)

```python
# QDA: allows different covariances per class
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

print("=== QDA Classification Performance ===")
print(f"Train accuracy: {qda.score(X_train, y_train):.4f}")
print(f"Test accuracy: {qda.score(X_test, y_test):.4f}")

# LDA vs QDA comparison
print("\n=== LDA vs QDA Comparison ===")
comparison = pd.DataFrame({
    'Model': ['LDA', 'QDA'],
    'Train Accuracy': [lda_clf.score(X_train, y_train),
                       qda.score(X_train, y_train)],
    'Test Accuracy': [lda_clf.score(X_test, y_test),
                      qda.score(X_test, y_test)]
})
print(comparison)

print("\nLDA vs QDA Selection Criteria:")
print("- LDA: Assumes equal covariances across classes, simpler, suitable for small data")
print("- QDA: Allows different covariances per class, more flexible, suitable for large data")
```

### 3.4 Decision Boundary Visualization

```python
def plot_decision_boundary_2d(model, X, y, title='', ax=None):
    """Visualize 2D decision boundary"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)

    # Data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         edgecolors='k', s=50, alpha=0.7)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax

# Reduce to 2D for visualization
X_2d = X_scaled[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_iris, test_size=0.3, random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LDA decision boundary
lda_2d = LinearDiscriminantAnalysis()
lda_2d.fit(X_train_2d, y_train_2d)
plot_decision_boundary_2d(lda_2d, X_2d, y_iris, 'LDA Decision Boundary', axes[0])

# QDA decision boundary
qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X_train_2d, y_train_2d)
plot_decision_boundary_2d(qda_2d, X_2d, y_iris, 'QDA Decision Boundary', axes[1])

plt.tight_layout()
plt.show()
```

---

## 4. Cluster Validation

### 4.1 Internal Metrics

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

# Clustering analysis example
np.random.seed(42)

# Generate data (3 clusters)
n_samples = 300
X_cluster = np.vstack([
    np.random.normal([0, 0], 0.5, (n_samples//3, 2)),
    np.random.normal([3, 3], 0.5, (n_samples//3, 2)),
    np.random.normal([0, 3], 0.5, (n_samples//3, 2))
])

def evaluate_clustering(X, k_range=range(2, 8)):
    """
    Evaluate cluster validity for various K
    """
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        inertia = kmeans.inertia_

        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'inertia': inertia
        })

    return pd.DataFrame(results)

# Evaluation
eval_results = evaluate_clustering(X_cluster)
print("=== Cluster Validity Metrics ===")
print(eval_results)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Elbow plot (Inertia)
ax = axes[0, 0]
ax.plot(eval_results['k'], eval_results['inertia'], 'bo-')
ax.set_xlabel('K')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Plot')
ax.grid(True, alpha=0.3)

# Silhouette score
ax = axes[0, 1]
ax.plot(eval_results['k'], eval_results['silhouette'], 'go-')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score (higher is better)')
ax.grid(True, alpha=0.3)

# Calinski-Harabasz
ax = axes[1, 0]
ax.plot(eval_results['k'], eval_results['calinski_harabasz'], 'ro-')
ax.set_xlabel('K')
ax.set_ylabel('Calinski-Harabasz Index')
ax.set_title('Calinski-Harabasz (higher is better)')
ax.grid(True, alpha=0.3)

# Davies-Bouldin
ax = axes[1, 1]
ax.plot(eval_results['k'], eval_results['davies_bouldin'], 'mo-')
ax.set_xlabel('K')
ax.set_ylabel('Davies-Bouldin Index')
ax.set_title('Davies-Bouldin (lower is better)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.2 Silhouette Analysis

```python
def silhouette_analysis(X, n_clusters):
    """
    Silhouette analysis visualization
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette plot
    ax = axes[0]
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         alpha=0.7, label=f'Cluster {i}')

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f'Average: {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.set_title(f'Silhouette Analysis (K={n_clusters})')
    ax.legend()

    # Cluster visualization
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    for i, c in enumerate(colors):
        ax.scatter(X[labels == i, 0], X[labels == i, 1],
                   color=c, alpha=0.7, label=f'Cluster {i}')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'K-Means Clustering (K={n_clusters})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return silhouette_avg

# Silhouette analysis with K=3
silhouette_analysis(X_cluster, n_clusters=3)

# Compare with K=4
silhouette_analysis(X_cluster, n_clusters=4)
```

### 4.3 External Metrics (when labels are available)

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

# True labels
true_labels = np.repeat([0, 1, 2], n_samples//3)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
pred_labels = kmeans.fit_predict(X_cluster)

# External metrics
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, pred_labels)

print("=== External Cluster Validity Metrics ===")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"  - Range: [-1, 1], 1 is perfect match")
print(f"\nNormalized Mutual Information (NMI): {nmi:.4f}")
print(f"  - Range: [0, 1], 1 is perfect match")
print(f"\nHomogeneity: {homogeneity:.4f}")
print(f"  - Extent to which each cluster contains only members of a single class")
print(f"\nCompleteness: {completeness:.4f}")
print(f"  - Extent to which all members of a class are assigned to the same cluster")
print(f"\nV-measure: {v_measure:.4f}")
print(f"  - Harmonic mean of homogeneity and completeness")
```

---

## 5. Practice Example

### 5.1 Comprehensive Multivariate Analysis

```python
def comprehensive_multivariate_analysis(X, y, feature_names, target_names):
    """
    Perform comprehensive multivariate analysis
    """
    print("="*60)
    print("Comprehensive Multivariate Analysis")
    print("="*60)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PCA
    print("\n[1] Principal Component Analysis (PCA)")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance: {pca.explained_variance_ratio_.round(3)}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_).round(3)}")

    # 2. LDA
    print("\n[2] Linear Discriminant Analysis (LDA)")
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_scaled, y)
    print(f"Number of LDA axes: {X_lda.shape[1]}")
    print(f"Explained variance: {lda.explained_variance_ratio_.round(3)}")

    # 3. Classification performance
    print("\n[3] Classification Performance")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    models = {
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PCA
    ax = axes[0, 0]
    for target in np.unique(y):
        mask = y == target
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.7,
                   label=target_names[target])
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LDA
    ax = axes[0, 1]
    for target in np.unique(y):
        mask = y == target
        if X_lda.shape[1] >= 2:
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.7,
                       label=target_names[target])
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
        else:
            ax.scatter(X_lda[mask, 0], np.random.randn(mask.sum())*0.1, alpha=0.7,
                       label=target_names[target])
            ax.set_xlabel('LD1')
            ax.set_ylabel('(jitter)')
    ax.set_title('LDA')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scree plot
    ax = axes[1, 0]
    ax.bar(range(1, len(pca.explained_variance_ratio_)+1),
           pca.explained_variance_ratio_, alpha=0.7, label='Individual')
    ax.plot(range(1, len(pca.explained_variance_ratio_)+1),
            np.cumsum(pca.explained_variance_ratio_), 'ro-', label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Scree Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loadings
    ax = axes[1, 1]
    loadings_df = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    loadings_df.plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_ylabel('Loading')
    ax.set_title('PC1, PC2 Loadings')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# Test with Wine data
comprehensive_multivariate_analysis(wine.data, wine.target,
                                     wine.feature_names, wine.target_names)
```

---

## 6. Practice Problems

### Problem 1: PCA
Apply PCA to the breast cancer dataset (load_breast_cancer):
1. Determine number of components needed to explain 95% variance
2. Visualize with first 2 principal components
3. Identify top 3 features contributing to PC1

### Problem 2: Factor Analysis
Generate a 6-variable dataset (with 2 latent factors):
1. Fit a 2-factor model
2. Apply Varimax rotation
3. Interpret each factor

### Problem 3: LDA vs QDA
Using Wine dataset:
1. Compare LDA and QDA classification performance
2. Perform 5-fold cross-validation
3. Visualize decision boundaries (after 2D reduction)

### Problem 4: Cluster Validation
K-means clustering on synthetic data:
1. Determine optimal K using elbow method
2. Perform silhouette analysis
3. Compare cluster quality at K=2, 3, 4

---

## 7. Key Summary

### Method Selection Guide

| Purpose | Method | Characteristics |
|------|------|------|
| Dimensionality reduction (unsupervised) | PCA | Maximizes variance, fast |
| Structure discovery | Factor Analysis | Interprets latent variables |
| Dimensionality reduction (supervised) | LDA | Maximizes class separation |
| Classification (linear) | LDA | Assumes equal covariances |
| Classification (nonlinear) | QDA | Allows different covariances |

### PCA Essentials

- Apply after standardization (uniform variable scale)
- Number of components: Kaiser rule, variance criterion, scree plot
- Loading interpretation: contribution of each variable to components

### Cluster Validation

- Internal metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin
- External metrics (when labels available): ARI, NMI

### Next Chapter Preview

Chapter 13 **Nonparametric Statistics** will cover:
- When nonparametric tests are needed
- Mann-Whitney U, Wilcoxon, Kruskal-Wallis
- Spearman/Kendall correlation

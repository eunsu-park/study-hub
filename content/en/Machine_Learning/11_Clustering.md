# Clustering

**Previous**: [kNN and Naive Bayes](./10_kNN_and_Naive_Bayes.md) | **Next**: [Dimensionality Reduction](./12_Dimensionality_Reduction.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the goal of clustering and distinguish it from supervised classification
2. Implement K-Means clustering and interpret centroid assignments
3. Apply the elbow method and silhouette analysis to select the optimal number of clusters
4. Compare K-Means, DBSCAN, hierarchical clustering, and Gaussian Mixture Models
5. Identify when density-based methods (DBSCAN) outperform centroid-based methods (K-Means)
6. Evaluate clustering quality using both internal metrics (silhouette, Davies-Bouldin) and external metrics (ARI, NMI)
7. Apply clustering to a real-world customer segmentation problem

---

All the supervised algorithms we have studied so far require labeled data -- someone has to tell the model what the "right answer" is. But what if you have no labels at all? Clustering is the gateway to **unsupervised learning**: it discovers hidden groupings in data by measuring similarity, without any labeled examples to guide it. Understanding clustering is essential because real-world data is overwhelmingly unlabeled, and the ability to reveal its latent structure powers applications from customer segmentation to anomaly detection to gene expression analysis.

---

## 1. Clustering Concepts

### 1.1 What is Clustering?

**Goal**: Partition data into groups (clusters) where:
- Points within a cluster are similar (high intra-cluster similarity)
- Points in different clusters are dissimilar (low inter-cluster similarity)

### 1.2 Types of Clustering

| Type | Description | Example Algorithm |
|------|-------------|-------------------|
| **Partitioning** | Divides data into K non-overlapping clusters | K-Means |
| **Hierarchical** | Builds a tree of clusters (dendrogram) | Agglomerative, Divisive |
| **Density-based** | Groups points in dense regions | DBSCAN |
| **Distribution-based** | Assumes data follows probability distributions | Gaussian Mixture Models |

### 1.3 Similarity Measures

- **Euclidean distance**: √(Σ(xi - yi)²)
- **Manhattan distance**: Σ|xi - yi|
- **Cosine similarity**: x·y / (||x|| ||y||)
- **Correlation**: Measures linear relationship

---

## 2. K-Means Clustering

### 2.1 Algorithm

```
1. Randomly initialize K cluster centroids
2. Repeat until convergence:
   a. Assign each point to the nearest centroid
   b. Update centroids as the mean of assigned points
3. Stop when centroids no longer change
```

### 2.2 Basic Implementation

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fixes centroid initialization seed — K-Means is sensitive to initial
# centroids; different seeds can produce different local optima (clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', edgecolors='black', linewidths=2,
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
```

### 2.3 Choosing Optimal K: Elbow Method

```python
# Try different K values
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)  # Sum of squared distances to nearest centroid; decreases monotonically with K

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal K')
plt.xticks(K_range)
plt.grid(True)
plt.show()
```

**How to interpret**:
- Look for the "elbow" where inertia starts decreasing slowly
- In this example, K=4 is the optimal choice

### 2.4 Silhouette Analysis

**Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters
- Range: [-1, 1]
- **Close to 1**: Well-clustered
- **Close to 0**: On the boundary between clusters
- **Negative**: May be assigned to wrong cluster

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Try different K values
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal K')
plt.xticks(K_range)
plt.grid(True)
plt.show()
```

### 2.5 Silhouette Plot

```python
from matplotlib import cm

def plot_silhouette(X, n_clusters):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f'Silhouette Plot (K={n_clusters})')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Average: {silhouette_avg:.3f}')
    ax.legend()
    plt.show()

# Visualize for different K
for k in [3, 4, 5]:
    plot_silhouette(X, k)
```

### 2.6 K-Means Limitations

1. **Must specify K in advance**
2. **Sensitive to initialization** (use `n_init` parameter for multiple runs)
3. **Assumes spherical clusters** (equal variance)
4. **Sensitive to outliers**
5. **Doesn't handle non-convex shapes well**

---

## 3. DBSCAN (Density-Based Spatial Clustering)

### 3.1 Algorithm

**Core Idea**: Clusters are dense regions separated by sparse regions

**Key Parameters**:
- **eps (ε)**: Maximum distance between two points to be neighbors
- **min_samples**: Minimum number of points to form a dense region

**Point Types**:
- **Core point**: Has at least `min_samples` neighbors within `eps`
- **Border point**: Within `eps` of a core point, but not a core point itself
- **Noise point**: Neither core nor border (outlier)

### 3.2 Basic Implementation

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate non-linearly separable data
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=0)

# eps defines the neighborhood radius — points within eps distance are considered
# reachable neighbors. Too small → most points become noise; too large → everything
# merges into one cluster. Use a k-NN distance plot elbow method to tune eps.
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='viridis', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()

# Print statistics
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

### 3.3 Choosing eps and min_samples

**Method 1: K-distance plot** (find the "elbow")

```python
from sklearn.neighbors import NearestNeighbors

# Compute k-nearest neighbor distances
k = 5  # min_samples
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Sort distances
distances = np.sort(distances[:, k-1], axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('K-distance Plot for eps Selection')
plt.grid(True)
plt.show()

# Look for the "elbow" to determine eps
```

**Rule of Thumb**:
- **min_samples**: Start with `2 * dimensions` or at least 5
- **eps**: Use k-distance plot or try multiple values

### 3.4 Comparison: K-Means vs DBSCAN

```python
from sklearn.datasets import make_circles

# Generate circular data (K-Means will fail)
X, y_true = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=0)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.6)
axes[0].set_title('K-Means (Fails on circular data)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

axes[1].scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='viridis', alpha=0.6)
axes[1].set_title('DBSCAN (Works well)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### 3.5 DBSCAN Advantages and Disadvantages

**Advantages**:
- Doesn't require specifying number of clusters
- Can find clusters of arbitrary shape
- Robust to outliers (identifies noise points)

**Disadvantages**:
- Sensitive to eps and min_samples
- Struggles with varying densities
- Doesn't work well in high dimensions

---

## 4. Hierarchical Clustering

### 4.1 Agglomerative (Bottom-up)

```
1. Start with each point as its own cluster
2. Repeatedly merge the two closest clusters
3. Stop when all points are in one cluster
```

### 4.2 Linkage Methods

| Linkage | Description | Use Case |
|---------|-------------|----------|
| **Single** | Minimum distance between clusters | Elongated clusters |
| **Complete** | Maximum distance between clusters | Compact clusters |
| **Average** | Average distance between all pairs | Balanced |
| **Ward** | Minimizes within-cluster variance | Most common, balanced clusters |

### 4.3 Dendrogram Visualization

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)

# Compute linkage
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
```

### 4.4 Agglomerative Clustering with sklearn

```python
from sklearn.cluster import AgglomerativeClustering

# Train
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_agg = agg_cluster.fit_predict(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_agg, s=50, cmap='viridis', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Agglomerative Clustering')
plt.show()
```

---

## 5. Gaussian Mixture Models (GMM)

### 5.1 Concept

**Probabilistic model**: Assumes data is generated from a mixture of Gaussian distributions

- Each cluster is modeled as a Gaussian distribution
- Points have **soft assignments** (probabilities for each cluster)
- Uses **Expectation-Maximization (EM) algorithm**

### 5.2 Implementation

```python
from sklearn.mixture import GaussianMixture

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Train GMM
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict
y_gmm = gmm.predict(X)

# Get probabilities (soft clustering)
probs = gmm.predict_proba(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, s=50, cmap='viridis', alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
            s=200, c='red', marker='X', edgecolors='black', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model')
plt.show()

# Print probability for first few samples
print("Sample probabilities (first 5):")
print(probs[:5])
```

### 5.3 Choosing Number of Components (AIC/BIC)

```python
# Try different number of components
n_components_range = range(1, 11)
aics = []
bics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X)
    aics.append(gmm.aic(X))
    bics.append(gmm.bic(X))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, aics, marker='o', label='AIC')
plt.plot(n_components_range, bics, marker='s', label='BIC')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.title('Model Selection (lower is better)')
plt.legend()
plt.grid(True)
plt.show()
```

**AIC (Akaike Information Criterion)** and **BIC (Bayesian Information Criterion)**: Lower is better

---

## 6. Cluster Evaluation Metrics

### 6.1 Internal Metrics (no ground truth needed)

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Train K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Silhouette Score (higher is better, range [-1, 1])
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette:.4f}")

# Davies-Bouldin Index (lower is better, range [0, ∞))
davies_bouldin = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# Calinski-Harabasz Index (higher is better, range [0, ∞))
calinski_harabasz = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
```

### 6.2 External Metrics (with ground truth labels)

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

# Adjusted Rand Index (range [-1, 1], 1 is perfect)
ari = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index: {ari:.4f}")

# Normalized Mutual Information (range [0, 1], 1 is perfect)
nmi = normalized_mutual_info_score(y_true, labels)
print(f"Normalized Mutual Information: {nmi:.4f}")

# Fowlkes-Mallows Score (range [0, 1], 1 is perfect)
fmi = fowlkes_mallows_score(y_true, labels)
print(f"Fowlkes-Mallows Index: {fmi:.4f}")
```

---

## 7. Practical Application: Customer Segmentation

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer data
np.random.seed(42)
n_customers = 500

data = {
    'annual_income': np.random.normal(50000, 20000, n_customers),
    'spending_score': np.random.randint(1, 100, n_customers),
    'age': np.random.randint(18, 70, n_customers),
    'purchase_frequency': np.random.randint(1, 50, n_customers)
}
df = pd.DataFrame(data)

# Standardize features — K-Means uses Euclidean distance, so features on
# different scales (e.g., income in $50k vs spending_score in 1-100) would
# make income dominate the distance calculation and distort clusters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal K using elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Customer Segmentation')
plt.xticks(K_range)
plt.grid(True)
plt.show()

# Choose K=4 and train
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = df.groupby('cluster').mean()
print("Cluster Summary:")
print(cluster_summary)

# Visualize (2D projection using first two features)
plt.figure(figsize=(10, 6))
plt.scatter(df['annual_income'], df['spending_score'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.colorbar(label='Cluster')
plt.show()
```

---

## 8. Comparison of Clustering Algorithms

| Algorithm | Clusters Shape | Outliers | Scalability | Parameters |
|-----------|----------------|----------|-------------|------------|
| **K-Means** | Spherical | Sensitive | High | K |
| **DBSCAN** | Arbitrary | Robust | Medium | eps, min_samples |
| **Hierarchical** | Any | Sensitive | Low (O(N²)) | K, linkage |
| **GMM** | Elliptical | Sensitive | Medium | K, covariance_type |

---

## 9. Exercises

### Exercise 1: K-Means on Iris
Load the iris dataset and perform K-Means clustering. Compare results with true labels.

```python
from sklearn.datasets import load_iris

# Your code here
```

### Exercise 2: DBSCAN Parameter Tuning
Generate data with varying densities and find optimal eps and min_samples for DBSCAN.

```python
# Your code here
```

### Exercise 3: Hierarchical Clustering
Perform hierarchical clustering on the wine dataset and visualize the dendrogram.

```python
from sklearn.datasets import load_wine

# Your code here
```

### Exercise 4: Customer Segmentation
Create a customer segmentation project using GMM and compare with K-Means.

```python
# Your code here
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **K-Means** | Partitioning, spherical clusters, requires K, use elbow/silhouette |
| **Elbow Method** | Plot inertia vs K, look for elbow |
| **Silhouette Score** | Measures clustering quality, range [-1, 1] |
| **DBSCAN** | Density-based, arbitrary shapes, handles outliers, eps + min_samples |
| **Hierarchical** | Dendrogram, linkage methods, bottom-up or top-down |
| **GMM** | Probabilistic, soft clustering, EM algorithm, AIC/BIC for selection |
| **Evaluation** | Internal (silhouette, Davies-Bouldin) vs External (ARI, NMI) |
| **Applications** | Customer segmentation, image compression, anomaly detection |

**Key Takeaway**: Choose clustering algorithm based on data characteristics. K-Means for spherical clusters, DBSCAN for arbitrary shapes with outliers, GMM for probabilistic interpretation.

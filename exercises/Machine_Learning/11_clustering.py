"""
Clustering - Exercise Solutions
================================
Lesson 11: Clustering

Exercises cover:
  1. K-Means on Iris dataset, compare with true labels
  2. DBSCAN parameter tuning on varying-density data
  3. Hierarchical clustering on wine dataset with dendrogram
  4. Customer segmentation with GMM vs K-Means
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score
)
from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# Exercise 1: K-Means on Iris
# Cluster iris data and compare with true labels.
# ============================================================
def exercise_1_kmeans_iris():
    """K-Means clustering on Iris and comparison with ground truth.

    K-Means partitions data into K spherical clusters by minimizing
    within-cluster sum of squared distances (inertia). Since Iris has
    3 species, we set K=3 and see how well unsupervised clustering
    recovers the known labels.

    ARI (Adjusted Rand Index) measures agreement between true labels
    and cluster assignments, adjusted for chance: 1.0 = perfect, 0.0 = random.
    """
    print("=" * 60)
    print("Exercise 1: K-Means on Iris")
    print("=" * 60)

    iris = load_iris()
    X = iris.data
    y_true = iris.target

    # Scale features for better clustering -- K-Means uses Euclidean
    # distance, so features on different scales would bias clusters
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_scaled)

    # Evaluate cluster quality
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    silhouette = silhouette_score(X_scaled, y_pred)

    print(f"Adjusted Rand Index:           {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Silhouette Score:              {silhouette:.4f}")

    # Cross-tabulation: how clusters map to true species
    print(f"\nCluster vs True Label Cross-tabulation:")
    print(f"{'Cluster':<10} {'setosa':<10} {'versicolor':<12} {'virginica':<10}")
    print("-" * 42)
    for c in range(3):
        counts = [np.sum((y_pred == c) & (y_true == t)) for t in range(3)]
        print(f"{c:<10} {counts[0]:<10} {counts[1]:<12} {counts[2]:<10}")

    # Elbow method to verify K=3 is optimal
    inertias = []
    K_range = range(1, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, "o-")
    plt.axvline(x=3, color="red", linestyle="--", label="K=3")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K-Means")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("11_ex1_elbow.png", dpi=100)
    plt.close()
    print("\nPlot saved: 11_ex1_elbow.png")


# ============================================================
# Exercise 2: DBSCAN Parameter Tuning
# Find optimal eps and min_samples for varying-density data.
# ============================================================
def exercise_2_dbscan_tuning():
    """DBSCAN parameter tuning on data with varying cluster densities.

    DBSCAN's two parameters:
    - eps: maximum distance between two samples to be neighbors
    - min_samples: minimum neighbors to form a core point

    Unlike K-Means, DBSCAN discovers the number of clusters automatically
    and identifies noise points (label=-1). It handles arbitrary cluster
    shapes but struggles with clusters of very different densities.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: DBSCAN Parameter Tuning")
    print("=" * 60)

    # Generate data with 3 clusters of different densities
    np.random.seed(42)
    # Tight cluster
    c1 = np.random.randn(100, 2) * 0.3 + [0, 0]
    # Medium cluster
    c2 = np.random.randn(80, 2) * 0.6 + [4, 4]
    # Spread cluster
    c3 = np.random.randn(60, 2) * 1.0 + [8, 1]
    # Noise
    noise = np.random.uniform(-2, 12, (20, 2))

    X = np.vstack([c1, c2, c3, noise])
    y_true = np.array([0]*100 + [1]*80 + [2]*60 + [-1]*20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Grid search over eps and min_samples
    eps_range = [0.2, 0.3, 0.4, 0.5, 0.7]
    min_samples_range = [3, 5, 7, 10]

    best_score = -1
    best_params = {}

    print(f"{'eps':<8} {'min_samples':<14} {'n_clusters':<12} {'noise':<8} {'silhouette':<12}")
    print("-" * 56)

    for eps in eps_range:
        for ms in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)

            if n_clusters >= 2:
                # Silhouette requires at least 2 clusters and non-noise points
                non_noise = labels != -1
                if np.sum(non_noise) > n_clusters:
                    score = silhouette_score(X_scaled[non_noise], labels[non_noise])
                    if score > best_score:
                        best_score = score
                        best_params = {"eps": eps, "min_samples": ms}
                    print(f"{eps:<8.2f} {ms:<14} {n_clusters:<12} {n_noise:<8} {score:<12.4f}")
                else:
                    print(f"{eps:<8.2f} {ms:<14} {n_clusters:<12} {n_noise:<8} {'N/A':<12}")
            else:
                print(f"{eps:<8.2f} {ms:<14} {n_clusters:<12} {n_noise:<8} {'N/A':<12}")

    print(f"\nBest parameters: eps={best_params['eps']}, "
          f"min_samples={best_params['min_samples']}")
    print(f"Best silhouette score: {best_score:.4f}")


# ============================================================
# Exercise 3: Hierarchical Clustering
# Cluster wine dataset and visualize dendrogram.
# ============================================================
def exercise_3_hierarchical():
    """Hierarchical clustering with dendrogram on the wine dataset.

    Agglomerative (bottom-up) clustering starts with each sample as its
    own cluster and iteratively merges the closest pair. The dendrogram
    visualizes this merge history, allowing you to choose the number of
    clusters by cutting at different heights.

    Linkage methods determine what "closest" means:
    - ward: minimizes within-cluster variance (produces compact clusters)
    - average: mean distance between all pairs
    - complete: maximum distance between any pair in each cluster
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Hierarchical Clustering (Wine)")
    print("=" * 60)

    wine = load_wine()
    X = wine.data
    y_true = wine.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dendrogram (using scipy linkage for visualization)
    Z = linkage(X_scaled, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="level", p=5, leaf_font_size=8)
    plt.axhline(y=20, color="red", linestyle="--", label="Cut at 3 clusters")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram (Wine Dataset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("11_ex3_dendrogram.png", dpi=100)
    plt.close()
    print("Dendrogram saved: 11_ex3_dendrogram.png")

    # Sklearn AgglomerativeClustering for evaluation
    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    y_pred = agg.fit_predict(X_scaled)

    ari = adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(X_scaled, y_pred)
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Silhouette Score:    {silhouette:.4f}")

    # Compare linkage methods
    print(f"\n{'Linkage':<12} {'ARI':>8} {'Silhouette':>12}")
    print("-" * 34)
    for method in ["ward", "average", "complete"]:
        agg = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = agg.fit_predict(X_scaled)
        ari = adjusted_rand_score(y_true, labels)
        sil = silhouette_score(X_scaled, labels)
        print(f"{method:<12} {ari:>8.4f} {sil:>12.4f}")


# ============================================================
# Exercise 4: Customer Segmentation
# GMM vs K-Means on synthetic customer data.
# ============================================================
def exercise_4_customer_segmentation():
    """Customer segmentation: GMM vs K-Means on synthetic data.

    GMM (Gaussian Mixture Model) provides soft clustering -- each point
    gets a probability of belonging to each cluster, rather than a hard
    assignment. This is useful when cluster boundaries are fuzzy.

    GMM also handles elliptical clusters naturally (different variances
    per cluster), while K-Means assumes spherical, equally-sized clusters.
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Customer Segmentation (GMM vs K-Means)")
    print("=" * 60)

    # Simulate customer data: age, annual spending, visit frequency
    np.random.seed(42)
    # Budget shoppers: young, low spending, moderate visits
    c1 = np.random.randn(100, 3) * [5, 50, 2] + [25, 200, 10]
    # Premium shoppers: middle-aged, high spending, frequent
    c2 = np.random.randn(80, 3) * [8, 100, 3] + [45, 800, 20]
    # Occasional shoppers: varied age, low spending, rare visits
    c3 = np.random.randn(60, 3) * [15, 30, 1] + [40, 100, 3]

    X = np.vstack([c1, c2, c3])
    y_true = np.array([0]*100 + [1]*80 + [2]*60)
    feature_names = ["Age", "Annual Spending ($)", "Monthly Visits"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)

    # GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_proba = gmm.predict_proba(X_scaled)

    # Evaluation
    print(f"{'Method':<12} {'ARI':>8} {'Silhouette':>12}")
    print("-" * 34)
    km_ari = adjusted_rand_score(y_true, km_labels)
    km_sil = silhouette_score(X_scaled, km_labels)
    gmm_ari = adjusted_rand_score(y_true, gmm_labels)
    gmm_sil = silhouette_score(X_scaled, gmm_labels)
    print(f"{'K-Means':<12} {km_ari:>8.4f} {km_sil:>12.4f}")
    print(f"{'GMM':<12} {gmm_ari:>8.4f} {gmm_sil:>12.4f}")

    # GMM advantage: soft assignments show uncertainty at cluster boundaries
    uncertain = np.max(gmm_proba, axis=1) < 0.8
    print(f"\nGMM uncertain assignments (max prob < 0.8): {np.sum(uncertain)}/{len(X)}")

    # Segment profiles -- shows the business value of clustering
    print(f"\nSegment profiles (K-Means):")
    for c in range(3):
        mask = km_labels == c
        means = X[mask].mean(axis=0)
        print(f"  Cluster {c} (n={np.sum(mask)}):")
        for name, val in zip(feature_names, means):
            print(f"    {name}: {val:.1f}")


if __name__ == "__main__":
    exercise_1_kmeans_iris()
    exercise_2_dbscan_tuning()
    exercise_3_hierarchical()
    exercise_4_customer_segmentation()

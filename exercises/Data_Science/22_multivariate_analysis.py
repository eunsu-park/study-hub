"""
Exercises for Lesson 22: Multivariate Analysis
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: PCA from Scratch ===
# Problem: Implement PCA on a synthetic 4-dimensional dataset using
#   eigenvalue decomposition of the covariance matrix. Determine the
#   number of components to retain using the Kaiser rule and the
#   cumulative variance threshold.
def exercise_1():
    """Solution implementing PCA via eigenvalue decomposition.

    PCA steps:
    1. Standardize the data (zero mean, unit variance)
    2. Compute the covariance matrix
    3. Compute eigenvalues and eigenvectors
    4. Sort by eigenvalue (descending)
    5. Project data onto the top k eigenvectors

    Kaiser rule: retain components with eigenvalue > 1 (on standardized data).
    Variance threshold: retain enough components to explain >= 80-90% of variance.
    """
    np.random.seed(42)
    n = 200
    p = 4

    # Generate correlated data: X1 and X2 are correlated, X3 weakly, X4 noise
    z1 = np.random.normal(0, 3, n)
    z2 = np.random.normal(0, 1, n)
    z3 = np.random.normal(0, 0.5, n)

    X = np.column_stack([
        z1 + np.random.normal(0, 0.5, n),       # strongly related to z1
        0.8 * z1 + z2 + np.random.normal(0, 0.5, n),  # correlated with X1
        z2 + z3 + np.random.normal(0, 0.5, n),  # weakly related
        np.random.normal(0, 1, n)                 # pure noise
    ])

    feature_names = ["X1", "X2", "X3", "X4"]

    print(f"Data: {n} samples, {p} features")
    print(f"Feature standard deviations: {np.std(X, axis=0, ddof=1).round(3).tolist()}")

    # Step 1: Standardize
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    X_std = (X - means) / stds

    # Step 2: Covariance matrix of standardized data (= correlation matrix)
    cov_matrix = np.cov(X_std, rowvar=False)
    print(f"\nCorrelation matrix:")
    for i in range(p):
        row = "  ["
        for j in range(p):
            row += f" {cov_matrix[i, j]:7.4f}"
        row += " ]"
        print(row)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance ratio
    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var
    cumulative_ratio = np.cumsum(explained_ratio)

    print(f"\nPrincipal Components:")
    print(f"  {'PC':>4s}  {'Eigenvalue':>11s}  {'Var %':>8s}  {'Cumul %':>9s}  {'Kaiser':>8s}")
    print(f"  {'-'*46}")
    for i in range(p):
        kaiser = "Retain" if eigenvalues[i] > 1.0 else "Drop"
        print(f"  {i+1:4d}  {eigenvalues[i]:11.4f}  {explained_ratio[i]*100:7.2f}%  "
              f"{cumulative_ratio[i]*100:8.2f}%  {kaiser:>8s}")

    # Kaiser rule: retain eigenvalues > 1
    n_kaiser = np.sum(eigenvalues > 1.0)
    # 80% threshold
    n_80 = np.argmax(cumulative_ratio >= 0.80) + 1
    # 90% threshold
    n_90 = np.argmax(cumulative_ratio >= 0.90) + 1

    print(f"\nNumber of components to retain:")
    print(f"  Kaiser rule (eigenvalue > 1):    {n_kaiser}")
    print(f"  80% variance threshold:          {n_80}")
    print(f"  90% variance threshold:          {n_90}")

    # Project onto top k components
    k = n_kaiser
    X_projected = X_std @ eigenvectors[:, :k]
    print(f"\nProjected data: {n} samples x {k} components")
    proj_var = np.var(X_projected, axis=0, ddof=1)
    print(f"Projected variances: {proj_var.round(4).tolist()}")
    print(f"Eigenvalues:         {eigenvalues[:k].round(4).tolist()}")


# === Exercise 2: Factor Analysis Comparison ===
# Problem: Implement a simple factor model X = L*F + noise where L is
#   the loading matrix. Compare the factor analysis approach (latent
#   variables) with PCA (variance maximization).
def exercise_2():
    """Solution comparing Factor Analysis with PCA.

    Factor Analysis assumes: X = L*F + epsilon
    - L: loading matrix (p x k)
    - F: latent factors (k x 1), F ~ N(0, I)
    - epsilon: specific noise, epsilon ~ N(0, Psi) where Psi is diagonal

    Key difference from PCA:
    - PCA finds directions of maximum variance
    - FA finds latent factors that explain correlations between variables
    - FA separates common variance from unique variance
    """
    np.random.seed(42)
    n = 500
    p = 5
    k = 2  # true number of factors

    # Define true loading matrix (5 variables, 2 latent factors)
    L_true = np.array([
        [0.9, 0.1],   # X1 loads strongly on Factor 1
        [0.8, 0.2],   # X2 loads strongly on Factor 1
        [0.1, 0.9],   # X3 loads strongly on Factor 2
        [0.2, 0.8],   # X4 loads strongly on Factor 2
        [0.5, 0.5],   # X5 loads on both
    ])
    var_names = ["X1", "X2", "X3", "X4", "X5"]

    # Specific variances (unique to each variable)
    psi = np.array([0.1, 0.15, 0.1, 0.12, 0.2])

    # Generate data
    F = np.random.normal(0, 1, (n, k))       # latent factors
    E = np.random.normal(0, 1, (n, p)) * np.sqrt(psi)  # specific noise

    X = F @ L_true.T + E

    print(f"Factor Analysis Model: X = L*F + epsilon")
    print(f"  n={n}, p={p} variables, k={k} true factors")
    print(f"\nTrue loading matrix L:")
    for i in range(p):
        print(f"  {var_names[i]}: [{L_true[i, 0]:.1f}, {L_true[i, 1]:.1f}]")

    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    # PCA for comparison
    cov_mat = np.cov(X_std, rowvar=False)
    evals, evecs = np.linalg.eigh(cov_mat)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    print(f"\nPCA eigenvalues: {evals.round(4).tolist()}")
    print(f"  Gap after {k} components: {evals[k-1]:.4f} vs {evals[k]:.4f}")

    # PCA loadings (eigenvector * sqrt(eigenvalue))
    pca_loadings = evecs[:, :k] * np.sqrt(evals[:k])

    print(f"\nPCA loadings (scaled):")
    print(f"  {'Var':>4s}  {'PC1':>8s}  {'PC2':>8s}")
    print(f"  {'-'*24}")
    for i in range(p):
        print(f"  {var_names[i]:>4s}  {pca_loadings[i, 0]:8.4f}  {pca_loadings[i, 1]:8.4f}")

    # Simple Factor Analysis: estimate via principal factor method
    # Communality estimation: use squared multiple correlation as initial
    # Here we use the PCA-based estimate for simplicity
    communalities = np.sum(pca_loadings**2, axis=1)
    uniquenesses = 1 - communalities

    print(f"\nCommunalities and uniquenesses:")
    print(f"  {'Var':>4s}  {'Communality':>12s}  {'Uniqueness':>12s}  {'True Psi':>10s}")
    print(f"  {'-'*42}")
    for i in range(p):
        print(f"  {var_names[i]:>4s}  {communalities[i]:12.4f}  {uniquenesses[i]:12.4f}  {psi[i]:10.4f}")

    # Key conceptual comparison
    print(f"\nPCA vs Factor Analysis:")
    print(f"  PCA: maximizes total variance, no noise model")
    print(f"  FA:  separates common from unique variance, generative model")
    explained_pca = evals[:k].sum() / evals.sum()
    print(f"  PCA: {k} components explain {explained_pca*100:.1f}% of total variance")


# === Exercise 3: K-Means Clustering from Scratch ===
# Problem: Implement Lloyd's K-means algorithm and evaluate clustering
#   quality using the silhouette score.
def exercise_3():
    """Solution implementing K-means clustering and silhouette evaluation.

    K-means algorithm (Lloyd's):
    1. Initialize k centroids randomly
    2. Assign each point to the nearest centroid
    3. Recompute centroids as the mean of assigned points
    4. Repeat until convergence

    Silhouette score for point i:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) = mean distance to same-cluster points,
          b(i) = mean distance to nearest other cluster's points.
    """
    np.random.seed(42)

    # Generate 3 well-separated clusters in 2D
    n_per_cluster = 50
    centers = np.array([[0, 0], [5, 5], [10, 0]])
    k_true = len(centers)

    X = np.vstack([
        np.random.normal(c, 1.0, (n_per_cluster, 2)) for c in centers
    ])
    true_labels = np.repeat(np.arange(k_true), n_per_cluster)

    n = len(X)
    print(f"Data: {n} points, {k_true} true clusters, 2D")
    print(f"True centers: {centers.tolist()}")

    # K-means implementation
    k = 3
    max_iter = 100

    # Initialize centroids by randomly selecting k data points
    init_idx = np.random.choice(n, k, replace=False)
    centroids = X[init_idx].copy()

    print(f"\nK-means (k={k}):")
    print(f"  Initial centroids: {centroids.round(2).tolist()}")

    for iteration in range(max_iter):
        # Assignment step: compute distances to all centroids
        distances = np.zeros((n, k))
        for j in range(k):
            diff = X - centroids[j]
            distances[:, j] = np.sum(diff**2, axis=1)
        labels = np.argmin(distances, axis=1)

        # Update step: recompute centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        # Check convergence
        shift = np.sqrt(np.sum((new_centroids - centroids)**2))
        centroids = new_centroids

        if shift < 1e-6:
            print(f"  Converged at iteration {iteration + 1}")
            break

    print(f"  Final centroids: {centroids.round(4).tolist()}")

    # Cluster sizes
    for j in range(k):
        count = np.sum(labels == j)
        print(f"  Cluster {j}: {count} points, centroid = {centroids[j].round(3).tolist()}")

    # Inertia (within-cluster sum of squares)
    inertia = 0
    for j in range(k):
        cluster_points = X[labels == j]
        inertia += np.sum((cluster_points - centroids[j])**2)
    print(f"  Inertia (WCSS): {inertia:.2f}")

    # Silhouette score (simplified: compute for a random subset)
    def silhouette_for_point(X, labels, i, k):
        """Compute silhouette score for a single point."""
        ci = labels[i]
        same_mask = labels == ci
        same_mask[i] = False
        a_i = np.mean(np.sqrt(np.sum((X[same_mask] - X[i])**2, axis=1)))
        b_i = np.inf
        for j in range(k):
            if j == ci:
                continue
            other = X[labels == j]
            if len(other) > 0:
                b_i = min(b_i, np.mean(np.sqrt(np.sum((other - X[i])**2, axis=1))))
        return (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    sil_scores = np.array([silhouette_for_point(X, labels, i, k) for i in range(n)])
    mean_sil = sil_scores.mean()

    print(f"\nSilhouette Analysis:")
    print(f"  Mean silhouette score: {mean_sil:.4f}")
    strength = ("strong" if mean_sil > 0.7 else "reasonable" if mean_sil > 0.5
                else "weak" if mean_sil > 0.25 else "none")
    print(f"  Cluster structure: {strength}")


# === Exercise 4: LDA (Linear Discriminant Analysis) ===
# Problem: Implement Fisher's LDA for two-class classification in 2D.
#   Project onto the direction that maximizes between-class separation
#   relative to within-class spread.
def exercise_4():
    """Solution implementing Fisher's Linear Discriminant Analysis.

    Fisher's LDA finds the projection direction w that maximizes:
        J(w) = (w^T S_B w) / (w^T S_W w)

    where S_B is the between-class scatter matrix and S_W is the
    within-class scatter matrix.

    The optimal w is proportional to S_W^{-1} (mu_1 - mu_2).
    """
    np.random.seed(42)

    # Generate two-class data in 2D
    n1, n2 = 60, 60
    mu1 = np.array([2, 3])
    mu2 = np.array([5, 4])

    # Both classes have the same covariance (homoscedastic)
    cov_shared = np.array([[2.0, 0.8], [0.8, 1.5]])
    L = np.linalg.cholesky(cov_shared)

    X1 = np.random.normal(size=(n1, 2)) @ L.T + mu1
    X2 = np.random.normal(size=(n2, 2)) @ L.T + mu2

    print("Fisher's Linear Discriminant Analysis (2-class)")
    print(f"  Class 0: n={n1}, mean={mu1.tolist()}")
    print(f"  Class 1: n={n2}, mean={mu2.tolist()}")

    # Compute class means
    m1 = X1.mean(axis=0)
    m2 = X2.mean(axis=0)
    print(f"\n  Sample means: C0=[{m1[0]:.3f}, {m1[1]:.3f}], C1=[{m2[0]:.3f}, {m2[1]:.3f}]")

    # Within-class scatter matrix: S_W = S_1 + S_2
    S1 = (X1 - m1).T @ (X1 - m1)
    S2 = (X2 - m2).T @ (X2 - m2)
    S_W = S1 + S2

    # Optimal projection direction: w = S_W^{-1} (m1 - m2)
    w = np.linalg.solve(S_W, m1 - m2)
    w = w / np.linalg.norm(w)
    print(f"  LDA direction (unit): [{w[0]:.4f}, {w[1]:.4f}]")

    # Project data onto w
    proj1 = X1 @ w
    proj2 = X2 @ w

    # Fisher's criterion
    between = (proj1.mean() - proj2.mean())**2
    within = proj1.var() + proj2.var()
    print(f"  Fisher's J(w) = {between/within:.4f}")

    # Classification
    threshold = (proj1.mean() + proj2.mean()) / 2
    if proj1.mean() > proj2.mean():
        correct = np.sum(proj1 > threshold) + np.sum(proj2 <= threshold)
    else:
        correct = np.sum(proj1 <= threshold) + np.sum(proj2 > threshold)
    accuracy = correct / (n1 + n2)
    print(f"  Classification accuracy: {accuracy:.3f}")
    print(f"\n  LDA maximizes class separation; PCA maximizes total variance.")


if __name__ == "__main__":
    print("=== Exercise 1: PCA from Scratch ===")
    exercise_1()
    print("\n=== Exercise 2: Factor Analysis Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: K-Means Clustering from Scratch ===")
    exercise_3()
    print("\n=== Exercise 4: LDA (Linear Discriminant Analysis) ===")
    exercise_4()
    print("\nAll exercises completed!")

"""
Exercises for Lesson 04: Norms and Distance Metrics
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


# === Exercise 1: Proof of Norm Properties ===
# Problem: For p >= 1, verify that the Lp norm satisfies the triangle inequality
# numerically for many random vectors.

def exercise_1():
    """Numerical verification of Lp norm triangle inequality."""
    np.random.seed(42)
    n_tests = 1000
    dim = 10
    p_values = [1, 1.5, 2, 3, 5, 10]

    print("Verifying triangle inequality: ||x+y||_p <= ||x||_p + ||y||_p")
    print(f"Testing with {n_tests} random vector pairs in R^{dim}\n")

    for p in p_values:
        violations = 0
        max_ratio = 0  # max of ||x+y||_p / (||x||_p + ||y||_p)

        for _ in range(n_tests):
            x = np.random.randn(dim)
            y = np.random.randn(dim)

            norm_x = np.linalg.norm(x, ord=p)
            norm_y = np.linalg.norm(y, ord=p)
            norm_sum = np.linalg.norm(x + y, ord=p)

            ratio = norm_sum / (norm_x + norm_y)
            max_ratio = max(max_ratio, ratio)

            if norm_sum > norm_x + norm_y + 1e-10:
                violations += 1

        print(f"L{p} norm: violations = {violations}/{n_tests}, "
              f"max ratio = {max_ratio:.6f} (must be <= 1.0)")

    # Also verify that p < 1 does NOT satisfy triangle inequality
    print("\n--- p < 1 (not a valid norm) ---")
    p = 0.5
    violations = 0
    for _ in range(n_tests):
        x = np.random.randn(dim)
        y = np.random.randn(dim)
        lp_x = np.sum(np.abs(x) ** p) ** (1 / p)
        lp_y = np.sum(np.abs(y) ** p) ** (1 / p)
        lp_sum = np.sum(np.abs(x + y) ** p) ** (1 / p)
        if lp_sum > lp_x + lp_y + 1e-10:
            violations += 1
    print(f"L{p}: violations = {violations}/{n_tests} (expected: many violations)")


# === Exercise 2: Mahalanobis Distance Implementation ===
# Problem: Compute Mahalanobis distance and detect outliers using chi-squared threshold.

def exercise_2():
    """Mahalanobis distance-based outlier detection."""
    np.random.seed(42)

    # Generate normal data
    n_samples = 200
    mean = np.array([0, 0])
    cov = np.array([[2, 1],
                    [1, 3]])
    data = np.random.multivariate_normal(mean, cov, n_samples)

    # Add some outliers
    n_outliers = 10
    outliers = np.random.uniform(low=-8, high=8, size=(n_outliers, 2))
    data_with_outliers = np.vstack([data, outliers])
    true_labels = np.array([0] * n_samples + [1] * n_outliers)  # 0=normal, 1=outlier

    # Compute Mahalanobis distance for each point
    sample_mean = np.mean(data_with_outliers, axis=0)
    sample_cov = np.cov(data_with_outliers.T)
    cov_inv = np.linalg.inv(sample_cov)

    mahal_distances = np.array([
        mahalanobis(point, sample_mean, cov_inv)
        for point in data_with_outliers
    ])

    # Chi-squared threshold (d=2 dimensions, alpha=0.01)
    d = 2
    alpha = 0.01
    threshold = np.sqrt(chi2.ppf(1 - alpha, df=d))
    print(f"Chi-squared threshold (d={d}, alpha={alpha}): {threshold:.4f}")

    # Detect outliers
    predicted_outliers = mahal_distances > threshold
    n_detected = np.sum(predicted_outliers)
    n_true_detected = np.sum(predicted_outliers & (true_labels == 1))
    n_false_positives = np.sum(predicted_outliers & (true_labels == 0))

    print(f"\nOutlier detection results:")
    print(f"  Total points: {len(data_with_outliers)}")
    print(f"  True outliers: {n_outliers}")
    print(f"  Detected outliers: {n_detected}")
    print(f"  True positives: {n_true_detected}/{n_outliers}")
    print(f"  False positives: {n_false_positives}")

    # Show distances
    print(f"\nMahalanobis distance statistics:")
    print(f"  Normal data mean distance: {mahal_distances[:n_samples].mean():.4f}")
    print(f"  Outlier mean distance: {mahal_distances[n_samples:].mean():.4f}")


# === Exercise 3: Regularization Path ===
# Problem: Track how coefficients become zero as lambda increases in Lasso.

def exercise_3():
    """Lasso regularization path showing sparsification order."""
    from sklearn.linear_model import lasso_path
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler

    # Generate data with known sparse structure
    np.random.seed(42)
    X, y, true_coef = make_regression(
        n_samples=200, n_features=10, n_informative=5,
        noise=5, coef=True, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute Lasso path
    alphas, coefs, _ = lasso_path(X_scaled, y, n_alphas=100)

    print(f"True coefficients: {np.round(true_coef, 2)}")
    print(f"Non-zero true coefficients: {np.sum(np.abs(true_coef) > 0.1)}")
    print(f"\nAlpha range: [{alphas[-1]:.4f}, {alphas[0]:.4f}]")

    # Track order of coefficient zeroing
    print("\nCoefficient elimination order (as alpha increases):")
    for feature_idx in range(coefs.shape[0]):
        coef_path = coefs[feature_idx, :]
        # Find alpha where coefficient first becomes zero
        nonzero_mask = np.abs(coef_path) > 1e-10
        if np.any(~nonzero_mask):
            first_zero_idx = np.argmax(~nonzero_mask)
            alpha_zero = alphas[first_zero_idx]
        else:
            alpha_zero = float('inf')

        true_val = true_coef[feature_idx]
        status = "informative" if abs(true_val) > 0.1 else "noise"
        print(f"  Feature {feature_idx:2d} ({status:12s}, true={true_val:7.2f}): "
              f"zeroed at alpha = {alpha_zero:.4f}")

    # Count non-zero at various alphas
    print("\nNumber of non-zero coefficients at selected alphas:")
    for alpha_idx in [0, 25, 50, 75, -1]:
        n_nonzero = np.sum(np.abs(coefs[:, alpha_idx]) > 1e-10)
        print(f"  alpha={alphas[alpha_idx]:.4f}: {n_nonzero} non-zero coefficients")


# === Exercise 4: Norm-Preserving Transformations ===
# Problem: Prove and verify that orthogonal matrices preserve L2 norm.

def exercise_4():
    """Verify that orthogonal transformations preserve L2 and Frobenius norms."""
    np.random.seed(42)

    # Create random orthogonal matrix via QR decomposition
    A = np.random.randn(5, 5)
    Q, _ = np.linalg.qr(A)

    print(f"Q is orthogonal (Q^T Q = I): {np.allclose(Q.T @ Q, np.eye(5))}")

    # Proof sketch for L2 norm preservation:
    print("\nProof: ||Qx||_2^2 = (Qx)^T(Qx) = x^T Q^T Q x = x^T I x = x^T x = ||x||_2^2")

    # Numerical verification with random vectors
    print("\nNumerical verification (L2 norm):")
    for i in range(5):
        x = np.random.randn(5)
        norm_x = np.linalg.norm(x, 2)
        norm_qx = np.linalg.norm(Q @ x, 2)
        print(f"  ||x||_2 = {norm_x:.6f}, ||Qx||_2 = {norm_qx:.6f}, "
              f"preserved: {np.isclose(norm_x, norm_qx)}")

    # Frobenius norm: ||QA||_F = ||A||_F
    print("\nFrobenius norm preservation:")
    M = np.random.randn(5, 5)
    norm_M = np.linalg.norm(M, 'fro')
    norm_QM = np.linalg.norm(Q @ M, 'fro')
    norm_MQ = np.linalg.norm(M @ Q, 'fro')
    print(f"  ||M||_F  = {norm_M:.6f}")
    print(f"  ||QM||_F = {norm_QM:.6f} (left multiply)")
    print(f"  ||MQ||_F = {norm_MQ:.6f} (right multiply)")
    print(f"  Preserved: {np.isclose(norm_M, norm_QM) and np.isclose(norm_M, norm_MQ)}")

    # Proof for Frobenius: ||QA||_F^2 = tr((QA)^T QA) = tr(A^T Q^T Q A) = tr(A^T A) = ||A||_F^2
    print("\nProof: ||QA||_F^2 = tr((QA)^T QA) = tr(A^T Q^T Q A) = tr(A^T A) = ||A||_F^2")


# === Exercise 5: Distance-Based Clustering ===
# Problem: k-means with Euclidean, Manhattan, and cosine distances.

def exercise_5():
    """Clustering comparison with different distance metrics."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    np.random.seed(42)

    # Generate 2D data with 3 clusters
    n_per_cluster = 100
    centers = np.array([[0, 0], [5, 5], [0, 8]])
    X = np.vstack([
        np.random.randn(n_per_cluster, 2) + centers[i]
        for i in range(3)
    ])
    true_labels = np.repeat([0, 1, 2], n_per_cluster)

    print(f"Data shape: {X.shape}")
    print(f"True clusters: 3 x {n_per_cluster} points\n")

    # 1. Euclidean distance (standard k-means)
    kmeans_eucl = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_eucl = kmeans_eucl.fit_predict(X)
    sil_eucl = silhouette_score(X, labels_eucl, metric='euclidean')

    # 2. Manhattan distance (custom implementation using scipy)
    # sklearn KMeans only supports Euclidean, so we use a simple
    # k-medoids-like approach with Manhattan distance
    from scipy.spatial.distance import cdist

    def kmeans_custom(X, k, metric='cityblock', max_iter=100):
        """Simple k-means with custom distance metric."""
        n = len(X)
        # Initialize with random points
        rng = np.random.RandomState(42)
        centers = X[rng.choice(n, k, replace=False)]
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign points to nearest center
            dists = cdist(X, centers, metric=metric)
            new_labels = np.argmin(dists, axis=1)

            if np.all(new_labels == labels):
                break
            labels = new_labels

            # Update centers (use median for Manhattan, mean for others)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    if metric == 'cityblock':
                        centers[j] = np.median(X[mask], axis=0)
                    else:
                        centers[j] = np.mean(X[mask], axis=0)
        return labels

    labels_manh = kmeans_custom(X, 3, metric='cityblock')
    sil_manh = silhouette_score(X, labels_manh, metric='cityblock')

    # 3. Cosine distance
    X_norm = normalize(X)  # normalize for cosine
    labels_cos = kmeans_custom(X_norm, 3, metric='cosine')
    sil_cos = silhouette_score(X_norm, labels_cos, metric='cosine')

    print("Clustering results (silhouette score, higher = better):")
    print(f"  Euclidean (k-means): {sil_eucl:.4f}")
    print(f"  Manhattan (k-medians): {sil_manh:.4f}")
    print(f"  Cosine: {sil_cos:.4f}")

    # Compare with true labels (using adjusted Rand index)
    from sklearn.metrics import adjusted_rand_score
    ari_eucl = adjusted_rand_score(true_labels, labels_eucl)
    ari_manh = adjusted_rand_score(true_labels, labels_manh)
    ari_cos = adjusted_rand_score(true_labels, labels_cos)

    print(f"\nAdjusted Rand Index (agreement with true labels):")
    print(f"  Euclidean: {ari_eucl:.4f}")
    print(f"  Manhattan: {ari_manh:.4f}")
    print(f"  Cosine: {ari_cos:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Norm Triangle Inequality ===")
    exercise_1()
    print("\n=== Exercise 2: Mahalanobis Distance Outlier Detection ===")
    exercise_2()
    print("\n=== Exercise 3: Regularization Path ===")
    exercise_3()
    print("\n=== Exercise 4: Norm-Preserving Transformations ===")
    exercise_4()
    print("\n=== Exercise 5: Distance-Based Clustering ===")
    exercise_5()
    print("\nAll exercises completed!")

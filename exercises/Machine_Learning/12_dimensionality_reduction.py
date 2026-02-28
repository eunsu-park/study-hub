"""
Dimensionality Reduction - Exercise Solutions
===============================================
Lesson 12: Dimensionality Reduction

Exercises cover:
  1. PCA on wine dataset: determine components for 90% variance
  2. t-SNE visualization with different perplexity values
  3. Feature selection using RFE on breast cancer
  4. PCA + classification: compare original vs reduced data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# Exercise 1: PCA on Wine Dataset
# Determine how many components capture 90% variance.
# ============================================================
def exercise_1_pca_wine():
    """PCA on wine data to find minimum components for 90% variance.

    PCA finds orthogonal directions (principal components) that capture
    the most variance in the data. The first PC captures the most variance,
    the second PC captures the most remaining variance orthogonal to the
    first, and so on.

    Choosing 90% explained variance is a common heuristic -- it retains
    most of the information while potentially reducing dimensions significantly.
    """
    print("=" * 60)
    print("Exercise 1: PCA on Wine Dataset")
    print("=" * 60)

    wine = load_wine()
    X = wine.data

    # PCA requires centered (and ideally scaled) data because it maximizes
    # variance along PCs; unscaled features with large ranges would dominate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with all components to see full variance spectrum
    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Find minimum components for 90% variance
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Components for 90% variance: {n_90}")
    print(f"Exact cumulative variance at {n_90} components: {cumulative_var[n_90-1]:.4f}")

    print(f"\nVariance explained per component:")
    for i in range(min(6, X.shape[1])):
        print(f"  PC{i+1}: {pca_full.explained_variance_ratio_[i]:.4f} "
              f"(cumulative: {cumulative_var[i]:.4f})")

    # Scree plot
    plt.figure(figsize=(9, 5))
    plt.bar(range(1, len(cumulative_var) + 1),
            pca_full.explained_variance_ratio_, alpha=0.6, label="Individual")
    plt.step(range(1, len(cumulative_var) + 1),
             cumulative_var, where="mid", label="Cumulative", color="red")
    plt.axhline(y=0.90, color="green", linestyle="--", label="90% threshold")
    plt.axvline(x=n_90, color="orange", linestyle="--", label=f"n={n_90}")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot (Wine Dataset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("12_ex1_pca_scree.png", dpi=100)
    plt.close()
    print("\nPlot saved: 12_ex1_pca_scree.png")


# ============================================================
# Exercise 2: t-SNE Visualization
# Compare perplexity values on digits dataset.
# ============================================================
def exercise_2_tsne():
    """t-SNE visualization of digits dataset with different perplexity.

    Perplexity is roughly the number of effective nearest neighbors.
    - Low perplexity (5): focuses on very local structure, may fragment clusters
    - Medium perplexity (30): default, good balance of local and global
    - High perplexity (50): more global structure, may merge nearby clusters

    Note: t-SNE is for visualization only -- distances between clusters in
    the 2D embedding don't necessarily reflect true distances in the original space.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: t-SNE Visualization (Digits)")
    print("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target

    # Subsample for speed -- t-SNE is O(n^2) so full digits dataset is slow
    np.random.seed(42)
    idx = np.random.choice(len(X), 500, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    perplexities = [5, 30, 50]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, perp in zip(axes, perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                     n_iter=1000)
        X_2d = tsne.fit_transform(X_scaled)

        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_sub, cmap="tab10",
                             s=15, alpha=0.7)
        ax.set_title(f"Perplexity = {perp}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("t-SNE: Effect of Perplexity on Digit Visualization", fontsize=13)
    plt.colorbar(scatter, ax=axes, label="Digit", shrink=0.8)
    plt.tight_layout()
    plt.savefig("12_ex2_tsne_perplexity.png", dpi=100)
    plt.close()
    print("Plot saved: 12_ex2_tsne_perplexity.png")

    # Quantitative check: how well-separated are clusters?
    from sklearn.metrics import silhouette_score
    for perp in perplexities:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_2d = tsne.fit_transform(X_scaled)
        sil = silhouette_score(X_2d, y_sub)
        print(f"  Perplexity {perp:>3}: silhouette = {sil:.4f}")


# ============================================================
# Exercise 3: Feature Selection with RFE
# Select best 3 features from breast cancer dataset.
# ============================================================
def exercise_3_feature_selection():
    """Recursive Feature Elimination (RFE) on breast cancer data.

    RFE works by repeatedly training a model, ranking features by their
    importance (coefficients for linear models, importances for trees),
    removing the least important feature, and repeating. This wrapper
    approach considers feature interactions, unlike filter methods.

    Selecting 3 features from 30 is aggressive but tests whether a
    small subset retains classification power.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Feature Selection with RFE")
    print("=" * 60)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # RFE with logistic regression as the estimator
    estimator = LogisticRegression(max_iter=5000, random_state=42)
    rfe = RFE(estimator, n_features_to_select=3)
    rfe.fit(X_scaled, y)

    selected = cancer.feature_names[rfe.support_]
    print(f"Selected features ({len(selected)}):")
    for name in selected:
        print(f"  - {name}")

    # Evaluate with selected vs all features
    # All features
    cv_all = cross_val_score(
        Pipeline([("scaler", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=5000))]),
        X, y, cv=5
    )

    # Selected features only
    X_selected = X[:, rfe.support_]
    cv_selected = cross_val_score(
        Pipeline([("scaler", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=5000))]),
        X_selected, y, cv=5
    )

    print(f"\nAll {X.shape[1]} features:    CV accuracy = {cv_all.mean():.4f} "
          f"+/- {cv_all.std():.4f}")
    print(f"Selected 3 features: CV accuracy = {cv_selected.mean():.4f} "
          f"+/- {cv_selected.std():.4f}")

    # Show feature ranking
    print(f"\nFull feature ranking (1=best):")
    ranked = sorted(zip(rfe.ranking_, cancer.feature_names))
    for rank, name in ranked[:10]:
        print(f"  Rank {rank:2d}: {name}")


# ============================================================
# Exercise 4: PCA + Classification
# Compare original vs PCA-reduced data classification.
# ============================================================
def exercise_4_pca_classification():
    """Compare classification on original vs PCA-reduced data.

    PCA as a preprocessing step can:
    1. Speed up training (fewer features)
    2. Reduce noise (small-variance components are often noise)
    3. Remove multicollinearity (PCs are orthogonal by construction)

    But it may also discard discriminative information if important
    class differences lie along low-variance directions.
    """
    print("\n" + "=" * 60)
    print("Exercise 4: PCA + Classification")
    print("=" * 60)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Compare different numbers of PCA components
    n_components_list = [2, 5, 10, 15, 20, 30]

    print(f"{'Components':<12} {'CV Accuracy':>12} {'Std':>8} {'Var Explained':>15}")
    print("-" * 50)

    for n_comp in n_components_list:
        if n_comp <= X.shape[1]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_comp)),
                ("lr", LogisticRegression(max_iter=5000, random_state=42))
            ])
            scores = cross_val_score(pipe, X, y, cv=5)

            # Calculate explained variance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_comp)
            pca.fit(X_scaled)
            var_explained = sum(pca.explained_variance_ratio_)

            marker = " <-- all" if n_comp == 30 else ""
            print(f"{n_comp:<12} {scores.mean():>12.4f} {scores.std():>8.4f} "
                  f"{var_explained:>14.4f}{marker}")

    # No PCA baseline
    pipe_no_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, random_state=42))
    ])
    scores_no_pca = cross_val_score(pipe_no_pca, X, y, cv=5)
    print(f"{'No PCA':<12} {scores_no_pca.mean():>12.4f} {scores_no_pca.std():>8.4f} "
          f"{'1.0000':>15}")

    print("\nKey insight: A small number of PCA components often achieves")
    print("comparable accuracy, suggesting high redundancy in the original features.")


if __name__ == "__main__":
    exercise_1_pca_wine()
    exercise_2_tsne()
    exercise_3_feature_selection()
    exercise_4_pca_classification()

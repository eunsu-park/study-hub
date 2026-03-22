"""
Support Vector Machines - Exercise Solutions
==============================================
Lesson 09: SVM

Exercises cover:
  1. Linear vs RBF kernel comparison on wine dataset
  2. Hyperparameter tuning (C, gamma) with GridSearchCV
  3. SVM Regression (SVR) with different kernels
  4. Effect of feature scaling on SVM performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# ============================================================
# Exercise 1: Linear vs RBF Kernel
# Compare on the wine dataset.
# ============================================================
def exercise_1_linear_vs_rbf():
    """Compare linear and RBF kernels on the wine dataset.

    Linear kernel: finds a single hyperplane separating classes.
      Best when data is linearly separable or high-dimensional.
    RBF kernel: maps data to infinite-dimensional space using
      K(x,x') = exp(-gamma * ||x-x'||^2), allowing complex
      nonlinear boundaries. Best for most general-purpose tasks.

    Scaling is mandatory for SVM because the margin calculation depends
    on Euclidean distances -- unscaled features would dominate the kernel.
    """
    print("=" * 60)
    print("Exercise 1: Linear vs RBF Kernel (Wine Dataset)")
    print("=" * 60)

    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features -- critical for SVM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Compare kernels with cross-validation for robustness
    kernels = ["linear", "rbf", "poly"]
    print(f"{'Kernel':<12} {'CV Mean':>10} {'CV Std':>10} {'Test Acc':>10}")
    print("-" * 45)

    for kernel in kernels:
        svm = SVC(kernel=kernel, random_state=42)
        cv_scores = cross_val_score(svm, X_train_s, y_train, cv=5)
        svm.fit(X_train_s, y_train)
        test_acc = svm.score(X_test_s, y_test)
        print(f"{kernel:<12} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f} "
              f"{test_acc:>10.4f}")


# ============================================================
# Exercise 2: Hyperparameter Tuning
# GridSearchCV for C and gamma on breast cancer data.
# ============================================================
def exercise_2_hyperparameter_tuning():
    """Tune C and gamma for RBF SVM via GridSearchCV.

    C controls the trade-off between a smooth decision boundary and
    classifying training points correctly:
    - High C: narrow margin, fits training data closely (may overfit)
    - Low C: wide margin, more misclassifications allowed (may underfit)

    gamma controls the RBF kernel's reach:
    - High gamma: each sample influences only nearby points (complex boundary)
    - Low gamma: each sample has wide influence (smoother boundary)

    The C-gamma interaction creates a 2D landscape; Grid Search explores it.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: SVM Hyperparameter Tuning")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Log-spaced grid -- regularization and kernel parameters have
    # multiplicative effects, so linear spacing wastes budget on
    # redundant nearby values
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
    }

    grid = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train_s, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV score:   {grid.best_score_:.4f}")
    print(f"Test score:      {grid.score(X_test_s, y_test):.4f}")

    # Heatmap of C vs gamma scores
    scores = grid.cv_results_["mean_test_score"].reshape(
        len(param_grid["C"]), len(param_grid["gamma"])
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(scores, interpolation="nearest", cmap="viridis")
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.xticks(range(len(param_grid["gamma"])), param_grid["gamma"])
    plt.yticks(range(len(param_grid["C"])), param_grid["C"])
    plt.colorbar(label="CV Accuracy")
    plt.title("SVM Grid Search: C vs gamma")
    for i in range(len(param_grid["C"])):
        for j in range(len(param_grid["gamma"])):
            plt.text(j, i, f"{scores[i,j]:.3f}", ha="center", va="center",
                     color="white" if scores[i, j] < 0.97 else "black",
                     fontsize=9)
    plt.tight_layout()
    plt.savefig("09_ex2_svm_heatmap.png", dpi=100)
    plt.close()
    print("Heatmap saved: 09_ex2_svm_heatmap.png")


# ============================================================
# Exercise 3: SVM Regression
# Compare SVR with different kernels on synthetic polynomial data.
# ============================================================
def exercise_3_svr():
    """SVR (Support Vector Regression) with different kernels.

    SVR fits a tube of width epsilon around the predictions; only
    samples outside this tube (support vectors) contribute to the loss.
    This epsilon-insensitive loss makes SVR robust to small noise.

    Different kernels capture different relationship types:
    - linear: straight line fit
    - rbf: flexible nonlinear fit
    - poly: polynomial fit (degree controls complexity)
    """
    print("\n" + "=" * 60)
    print("Exercise 3: SVM Regression")
    print("=" * 60)

    # Generate polynomial data: y = 0.5*x^2 + x + noise
    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, 200)).reshape(-1, 1)
    y = 0.5 * X.ravel() ** 2 + X.ravel() + np.random.randn(200) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    kernels = {
        "linear": SVR(kernel="linear", C=10),
        "rbf": SVR(kernel="rbf", C=10, gamma=0.5),
        "poly (degree=2)": SVR(kernel="poly", degree=2, C=10),
    }

    plt.figure(figsize=(12, 4))
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    X_plot_s = scaler.transform(X_plot)

    for i, (name, svr) in enumerate(kernels.items(), 1):
        svr.fit(X_train_s, y_train)
        y_pred = svr.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"{name:<20}: R²={r2:.4f}, RMSE={rmse:.4f}")

        plt.subplot(1, 3, i)
        plt.scatter(X_train, y_train, alpha=0.3, s=15, label="Train")
        plt.scatter(X_test, y_test, alpha=0.5, s=15, color="orange", label="Test")
        plt.plot(X_plot, svr.predict(X_plot_s), "r-", linewidth=2, label="SVR")
        plt.title(f"{name}\nR²={r2:.3f}")
        plt.xlabel("X")
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("09_ex3_svr.png", dpi=100)
    plt.close()
    print("Plot saved: 09_ex3_svr.png")


# ============================================================
# Exercise 4: Effect of Scaling
# Compare SVM on iris with and without scaling.
# ============================================================
def exercise_4_scaling_effect():
    """Demonstrate the critical effect of feature scaling on SVM.

    SVM computes distances between data points (for margin and kernel).
    Without scaling, features with larger numeric ranges dominate the
    distance calculation, effectively ignoring features with small ranges.

    Example: if feature A ranges [0, 1000] and feature B ranges [0, 1],
    the SVM kernel will be almost entirely determined by feature A.
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Effect of Scaling on SVM")
    print("=" * 60)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    # Without scaling
    svm_unscaled = SVC(kernel="rbf", random_state=42)
    svm_unscaled.fit(X_train, y_train)
    acc_unscaled = svm_unscaled.score(X_test, y_test)

    # With scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm_scaled = SVC(kernel="rbf", random_state=42)
    svm_scaled.fit(X_train_s, y_train)
    acc_scaled = svm_scaled.score(X_test_s, y_test)

    print(f"Without scaling: {acc_unscaled:.4f}")
    print(f"With scaling:    {acc_scaled:.4f}")
    print(f"Improvement:     {acc_scaled - acc_unscaled:+.4f}")

    # Cross-validated comparison for robustness
    cv_unscaled = cross_val_score(SVC(kernel="rbf", random_state=42),
                                  iris.data, iris.target, cv=5)
    # Scale within each fold using a pipeline to avoid leakage
    from sklearn.pipeline import Pipeline
    pipe_scaled = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=42))
    ])
    cv_scaled = cross_val_score(pipe_scaled, iris.data, iris.target, cv=5)

    print(f"\nCross-validated comparison:")
    print(f"  Unscaled CV: {cv_unscaled.mean():.4f} +/- {cv_unscaled.std():.4f}")
    print(f"  Scaled CV:   {cv_scaled.mean():.4f} +/- {cv_scaled.std():.4f}")

    # Show feature ranges to explain why scaling matters
    print(f"\nFeature ranges (training set):")
    for i, name in enumerate(iris.feature_names):
        print(f"  {name}: [{X_train[:, i].min():.2f}, {X_train[:, i].max():.2f}]"
              f"  range={X_train[:, i].max() - X_train[:, i].min():.2f}")


if __name__ == "__main__":
    exercise_1_linear_vs_rbf()
    exercise_2_hyperparameter_tuning()
    exercise_3_svr()
    exercise_4_scaling_effect()

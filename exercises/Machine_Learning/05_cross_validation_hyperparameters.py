"""
Cross-Validation and Hyperparameter Tuning - Exercise Solutions
================================================================
Lesson 05: Cross-Validation and Hyperparameters

Exercises cover:
  1. 10-Fold cross-validation on Iris dataset
  2. Grid Search to tune C parameter of logistic regression
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt


# ============================================================
# Exercise 1: K-Fold Cross-Validation
# Perform 10-Fold cross-validation on the Iris dataset.
# ============================================================
def exercise_1_kfold_cv():
    """10-Fold cross-validation provides a robust performance estimate.

    Why K-Fold instead of a single train/test split? A single split is
    noisy -- you might get lucky or unlucky depending on which samples
    end up in the test set. K-Fold trains K models on K different
    train/test partitions and averages the results, giving a more
    reliable estimate with a confidence interval.

    K=10 is a common choice: it uses 90% of data for training each fold
    (large enough for good models) while evaluating on the remaining 10%.
    """
    print("=" * 60)
    print("Exercise 1: 10-Fold Cross-Validation")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    model = LogisticRegression(max_iter=1000, random_state=42)

    # StratifiedKFold preserves class proportions in each fold -- critical
    # for multi-class datasets where random splitting might leave a fold
    # without samples from one class
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    print(f"Number of folds: 10")
    print(f"Individual fold scores:")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i:2d}: {score:.4f}")

    print(f"\nMean accuracy: {scores.mean():.4f}")
    print(f"Std deviation: {scores.std():.4f}")
    print(f"95% CI: {scores.mean():.4f} +/- {scores.std() * 2:.4f}")
    print(f"         [{scores.mean() - 2*scores.std():.4f}, "
          f"{scores.mean() + 2*scores.std():.4f}]")

    # Compare with different K values
    print("\n--- K comparison ---")
    for k in [3, 5, 10, 15]:
        skf_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        sc = cross_val_score(model, X, y, cv=skf_k, scoring="accuracy")
        print(f"  K={k:2d}: mean={sc.mean():.4f}, std={sc.std():.4f}")


# ============================================================
# Exercise 2: Grid Search
# Tune the C parameter of logistic regression.
# ============================================================
def exercise_2_grid_search():
    """Grid Search over the regularization parameter C.

    C is the inverse of regularization strength (C = 1/lambda):
    - Small C -> strong regularization -> simpler model (may underfit)
    - Large C -> weak regularization -> complex model (may overfit)

    Grid Search exhaustively evaluates every candidate value using
    cross-validation, guaranteeing we find the best C in our grid.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Grid Search for Logistic Regression")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    # Define the parameter grid -- log-spaced values cover a wide range
    # because regularization strength effects are multiplicative
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        return_train_score=True
    )
    grid.fit(X, y)

    print(f"Best C: {grid.best_params_['C']}")
    print(f"Best CV score: {grid.best_score_:.4f}")

    # Show all results to understand the regularization landscape
    print(f"\n{'C':<10} {'Train score':<14} {'CV score':<14} {'Std':<10}")
    print("-" * 50)
    for i, c_val in enumerate(param_grid["C"]):
        train_sc = grid.cv_results_["mean_train_score"][i]
        test_sc = grid.cv_results_["mean_test_score"][i]
        std_sc = grid.cv_results_["std_test_score"][i]
        marker = " <-- best" if c_val == grid.best_params_["C"] else ""
        print(f"{c_val:<10.3f} {train_sc:<14.4f} {test_sc:<14.4f} {std_sc:<10.4f}{marker}")

    # Visualization of the C tuning curve
    c_values = param_grid["C"]
    train_scores = grid.cv_results_["mean_train_score"]
    test_scores = grid.cv_results_["mean_test_score"]
    test_stds = grid.cv_results_["std_test_score"]

    plt.figure(figsize=(9, 5))
    plt.semilogx(c_values, train_scores, "o-", label="Train score", color="blue")
    plt.semilogx(c_values, test_scores, "s-", label="CV score", color="orange")
    plt.fill_between(c_values,
                     np.array(test_scores) - np.array(test_stds),
                     np.array(test_scores) + np.array(test_stds),
                     alpha=0.2, color="orange")
    plt.axvline(x=grid.best_params_["C"], color="red", linestyle="--",
                label=f"Best C={grid.best_params_['C']}")
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("Accuracy")
    plt.title("Grid Search: C Parameter Tuning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_ex2_grid_search.png", dpi=100)
    plt.close()
    print("\nPlot saved: 05_ex2_grid_search.png")


if __name__ == "__main__":
    exercise_1_kfold_cv()
    exercise_2_grid_search()

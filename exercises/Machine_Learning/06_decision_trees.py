"""
Decision Trees - Exercise Solutions
=====================================
Lesson 06: Decision Trees

Exercises cover:
  1. Basic classification on breast cancer data
  2. Cost-complexity pruning (CCP) with cross-validation
  3. Regression tree on diabetes data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score
)


# ============================================================
# Exercise 1: Basic Classification
# Train a decision tree on breast cancer data and evaluate.
# ============================================================
def exercise_1_basic_classification():
    """Decision tree classification with controlled max_depth.

    Decision trees are prone to overfitting when grown without limits --
    they can memorize every training sample by creating one leaf per sample.
    Setting max_depth=5 forces the tree to generalize by learning broader
    patterns rather than noise-specific splits.
    """
    print("=" * 60)
    print("Exercise 1: Decision Tree Classification")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    # max_depth=5 is a reasonable starting point for pre-pruning:
    # deep enough to capture feature interactions, shallow enough
    # to prevent overfitting on 30 features
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Tree depth: {clf.get_depth()}")
    print(f"Number of leaves: {clf.get_n_leaves()}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))

    # Feature importance -- decision trees provide a natural ranking
    # based on how much each feature reduces impurity across all splits
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]

    print("Top 10 features by importance:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {cancer.feature_names[idx]}: {importances[idx]:.4f}")


# ============================================================
# Exercise 2: Pruning
# Find optimal alpha using CCP and prune the tree.
# ============================================================
def exercise_2_pruning():
    """Cost-Complexity Pruning (CCP) with cross-validation.

    CCP adds a penalty alpha * |T| to the tree's cost, where |T| is the
    number of leaves. Larger alpha -> simpler tree (more pruning).

    Process:
    1. Grow a full tree to get the CCP alpha path
    2. For each alpha, evaluate with cross-validation
    3. Select the alpha that maximizes CV score
    4. Retrain with that alpha for the final model
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Cost-Complexity Pruning")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    # Step 1: Grow full tree and extract CCP path
    clf_full = DecisionTreeClassifier(random_state=42)
    clf_full.fit(X_train, y_train)
    print(f"Full tree: depth={clf_full.get_depth()}, "
          f"leaves={clf_full.get_n_leaves()}, "
          f"train_acc={clf_full.score(X_train, y_train):.4f}, "
          f"test_acc={clf_full.score(X_test, y_test):.4f}")

    path = clf_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    # Step 2: Cross-validate each alpha (sample for efficiency)
    # We sample every 5th alpha to reduce computation while still
    # covering the full range of tree complexities
    sampled_alphas = ccp_alphas[::5]
    cv_scores = []
    for alpha in sampled_alphas:
        clf_ccp = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        scores = cross_val_score(clf_ccp, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())

    # Step 3: Select best alpha
    best_idx = np.argmax(cv_scores)
    best_alpha = sampled_alphas[best_idx]
    print(f"\nOptimal alpha: {best_alpha:.6f}")
    print(f"Best CV score: {cv_scores[best_idx]:.4f}")

    # Step 4: Train pruned tree
    clf_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
    clf_pruned.fit(X_train, y_train)

    print(f"\nPruned tree: depth={clf_pruned.get_depth()}, "
          f"leaves={clf_pruned.get_n_leaves()}, "
          f"train_acc={clf_pruned.score(X_train, y_train):.4f}, "
          f"test_acc={clf_pruned.score(X_test, y_test):.4f}")

    # Visualization: alpha vs CV score
    plt.figure(figsize=(9, 5))
    plt.plot(sampled_alphas, cv_scores, "o-", markersize=4)
    plt.axvline(x=best_alpha, color="red", linestyle="--",
                label=f"Best alpha={best_alpha:.4f}")
    plt.xlabel("CCP Alpha")
    plt.ylabel("CV Accuracy")
    plt.title("Cost-Complexity Pruning: Alpha Selection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("06_ex2_ccp_pruning.png", dpi=100)
    plt.close()
    print("Plot saved: 06_ex2_ccp_pruning.png")


# ============================================================
# Exercise 3: Regression Tree
# Train a regression tree on diabetes data.
# ============================================================
def exercise_3_regression_tree():
    """Decision tree for regression on the diabetes dataset.

    Regression trees predict continuous values by averaging the target
    values of training samples in each leaf node. The tree splits to
    minimize mean squared error (MSE) at each node.

    Key hyperparameters for controlling overfitting:
    - max_depth: limits tree depth
    - min_samples_leaf: ensures each leaf has enough samples for a
      reliable average
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Regression Tree (Diabetes)")
    print("=" * 60)

    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )

    # min_samples_leaf=10 prevents leaves with very few samples,
    # which would have unreliable mean predictions (high variance)
    reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10,
                                random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Tree depth: {reg.get_depth()}")
    print(f"Number of leaves: {reg.get_n_leaves()}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"  -> Model explains {r2*100:.1f}% of variance in diabetes progression")

    # Feature importance for the regression tree
    importances = reg.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    print(f"\nTop 5 features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {diabetes.feature_names[idx]}: {importances[idx]:.4f}")

    # Actual vs predicted plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="black", linewidths=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Regression Tree (R² = {r2:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("06_ex3_regression_tree.png", dpi=100)
    plt.close()
    print("Plot saved: 06_ex3_regression_tree.png")


if __name__ == "__main__":
    exercise_1_basic_classification()
    exercise_2_pruning()
    exercise_3_regression_tree()

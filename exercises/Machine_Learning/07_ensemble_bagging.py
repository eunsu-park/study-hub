"""
Ensemble Learning - Bagging - Exercise Solutions
==================================================
Lesson 07: Ensemble Learning - Bagging

Exercises cover:
  1. Random Forest classification with feature importance analysis
  2. Hyperparameter tuning with Grid Search
  3. Voting Classifier combining multiple models
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


# ============================================================
# Exercise 1: Random Forest Classification
# Train RF on breast cancer data and analyze feature importance.
# ============================================================
def exercise_1_random_forest():
    """Random Forest with OOB score and feature importance analysis.

    Random Forest reduces overfitting compared to a single tree by:
    1. Bootstrap sampling: each tree sees a different 63% of data
    2. Random feature subsets: each split considers sqrt(n) features,
       decorrelating trees so their errors cancel out when averaged.

    OOB (Out-of-Bag) score uses the ~37% of data each tree never saw
    during training as a built-in validation set -- no need for a
    separate holdout.
    """
    print("=" * 60)
    print("Exercise 1: Random Forest Classification")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {rf.score(X_train, y_train):.4f}")
    print(f"Test accuracy:     {rf.score(X_test, y_test):.4f}")
    print(f"OOB score:         {rf.oob_score_:.4f}")

    # Feature importance analysis
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 features by importance:")
    for i in range(10):
        idx = indices[i]
        print(f"  {i+1:2d}. {cancer.feature_names[idx]:<30s} {importances[idx]:.4f}")

    # Plot top 10 features
    top_n = 10
    top_idx = indices[:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[top_idx][::-1])
    plt.yticks(range(top_n),
               [cancer.feature_names[i] for i in top_idx][::-1])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest - Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("07_ex1_rf_importance.png", dpi=100)
    plt.close()
    print("\nPlot saved: 07_ex1_rf_importance.png")


# ============================================================
# Exercise 2: Hyperparameter Tuning
# Find optimal RF parameters using Grid Search.
# ============================================================
def exercise_2_hyperparameter_tuning():
    """Grid Search for Random Forest hyperparameters.

    Key hyperparameters and their effects:
    - n_estimators: more trees = more stable predictions, diminishing returns
    - max_depth: limits tree complexity; None means grow until pure leaves
    - min_samples_leaf: higher values prevent small leaves that capture noise

    We use a modest grid to keep runtime reasonable; in practice, start with
    RandomizedSearchCV for wider exploration, then refine with GridSearchCV.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: RF Hyperparameter Tuning")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score:   {grid_search.best_score_:.4f}")
    print(f"Test score:      {grid_search.score(X_test, y_test):.4f}")

    # Show top 5 configurations
    import pandas as pd
    results = pd.DataFrame(grid_search.cv_results_)
    top5 = results.nsmallest(5, "rank_test_score")[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ]
    print(f"\nTop 5 configurations:")
    for _, row in top5.iterrows():
        print(f"  Rank {int(row['rank_test_score'])}: "
              f"score={row['mean_test_score']:.4f} +/- {row['std_test_score']:.4f} "
              f"| {row['params']}")


# ============================================================
# Exercise 3: Voting Ensemble
# Create a Voting Classifier combining multiple models.
# ============================================================
def exercise_3_voting_ensemble():
    """Soft Voting Classifier combines diverse models.

    Soft voting averages the predicted probabilities from all base models,
    then picks the class with the highest average probability. This works
    better than hard voting (majority vote) when models output well-calibrated
    probabilities, because it accounts for each model's confidence level.

    Diversity is key: combining models that make similar errors doesn't help.
    LR (linear boundary), RF (ensemble of trees), DT (single tree) provide
    different inductive biases.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Voting Classifier")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    # Define individual models with diverse architectures
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Soft voting: average predicted probabilities across models
    voting_clf = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("dt", dt)
        ],
        voting="soft"
    )

    # Compare individual models with the ensemble
    print(f"{'Model':<20} {'Accuracy':>10}")
    print("-" * 32)
    for name, model in [("LogisticRegression", lr),
                        ("RandomForest", rf),
                        ("DecisionTree", dt),
                        ("SoftVoting", voting_clf)]:
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{name:<20} {acc:>10.4f}")

    print("\nThe ensemble typically matches or exceeds the best individual model")
    print("because averaging cancels out individual model errors.")


if __name__ == "__main__":
    exercise_1_random_forest()
    exercise_2_hyperparameter_tuning()
    exercise_3_voting_ensemble()

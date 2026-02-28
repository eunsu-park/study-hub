"""
Ensemble Learning - Boosting - Exercise Solutions
==================================================
Lesson 08: Ensemble Learning - Boosting

Exercises cover:
  1. AdaBoost vs Gradient Boosting comparison
  2. Gradient Boosting hyperparameter tuning (sklearn-only, no XGBoost required)
  3. Gradient Boosting with early stopping
  4. Feature importance comparison: RF vs GradientBoosting
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import accuracy_score


# ============================================================
# Exercise 1: AdaBoost vs Gradient Boosting
# Compare on the Iris dataset.
# ============================================================
def exercise_1_adaboost_vs_gb():
    """Compare AdaBoost and Gradient Boosting on Iris.

    Both are boosting methods but differ fundamentally:
    - AdaBoost: adjusts sample weights to focus on misclassified examples,
      combines weak learners with weighted majority vote.
    - Gradient Boosting: fits each new tree to the negative gradient of the
      loss (i.e., residual errors), performing gradient descent in function space.

    GradientBoosting is generally more flexible because it can optimize
    arbitrary differentiable loss functions.
    """
    print("=" * 60)
    print("Exercise 1: AdaBoost vs Gradient Boosting")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    models = {
        "AdaBoost (50 estimators)": AdaBoostClassifier(
            n_estimators=50, random_state=42, algorithm="SAMME"
        ),
        "AdaBoost (100 estimators)": AdaBoostClassifier(
            n_estimators=100, random_state=42, algorithm="SAMME"
        ),
        "GradientBoosting (50 estimators)": GradientBoostingClassifier(
            n_estimators=50, random_state=42
        ),
        "GradientBoosting (100 estimators)": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    print(f"{'Model':<40} {'CV Mean':>10} {'CV Std':>10}")
    print("-" * 62)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"{name:<40} {scores.mean():>10.4f} {scores.std():>10.4f}")

    print("\nGradient Boosting typically outperforms AdaBoost because")
    print("it optimizes a smooth loss function via gradient descent,")
    print("while AdaBoost relies on discrete sample reweighting.")


# ============================================================
# Exercise 2: Hyperparameter Tuning (sklearn GradientBoosting)
# GridSearchCV for GradientBoostingClassifier on the wine dataset.
# ============================================================
def exercise_2_gb_hyperparameter_tuning():
    """Tune GradientBoosting hyperparameters with GridSearchCV.

    Key parameters and tuning strategy:
    1. learning_rate + n_estimators: lower rate needs more trees but
       generalizes better (the two are inversely coupled)
    2. max_depth: controls tree complexity; 3-5 is typical for boosting
       (shallow trees are better weak learners than deep ones)
    3. subsample < 1.0 adds stochastic gradient boosting -- like mini-batch
       SGD, it introduces regularization through noise
    """
    print("\n" + "=" * 60)
    print("Exercise 2: GradientBoosting Hyperparameter Tuning")
    print("=" * 60)

    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
    }

    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV score:   {grid.best_score_:.4f}")
    print(f"Test score:      {grid.score(X_test, y_test):.4f}")


# ============================================================
# Exercise 3: GradientBoosting with Early Stopping
# Train with staged_predict and identify optimal n_estimators.
# ============================================================
def exercise_3_early_stopping():
    """GradientBoosting with early stopping via staged_predict.

    Early stopping prevents overfitting by monitoring validation loss
    during training. Once validation loss stops improving (or worsens),
    we stop adding trees. This is crucial for boosting because, unlike
    bagging, adding more boosting iterations can increase overfitting.

    sklearn's GradientBoosting provides staged_predict() to evaluate
    the model at each boosting stage without retraining.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: GradientBoosting Early Stopping")
    print("=" * 60)

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    # Train a model with many estimators; we'll find the optimal count
    n_estimators = 300
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)

    # Track accuracy at each boosting stage using staged_predict
    train_scores = []
    test_scores = []
    for y_pred_train in gb.staged_predict(X_train):
        train_scores.append(accuracy_score(y_train, y_pred_train))
    for y_pred_test in gb.staged_predict(X_test):
        test_scores.append(accuracy_score(y_test, y_pred_test))

    # Find the optimal number of estimators (peak test accuracy)
    best_n = np.argmax(test_scores) + 1
    print(f"Total estimators trained: {n_estimators}")
    print(f"Optimal n_estimators:     {best_n}")
    print(f"Best test accuracy:       {max(test_scores):.4f}")
    print(f"Final test accuracy:      {test_scores[-1]:.4f}")

    if test_scores[-1] < max(test_scores):
        print("  -> Model overfits after the optimal point; early stopping helps!")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    stages = range(1, n_estimators + 1)
    plt.plot(stages, train_scores, label="Train", alpha=0.7)
    plt.plot(stages, test_scores, label="Test", alpha=0.7)
    plt.axvline(x=best_n, color="red", linestyle="--",
                label=f"Optimal n={best_n}")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("GradientBoosting: Training vs Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("08_ex3_early_stopping.png", dpi=100)
    plt.close()
    print("Plot saved: 08_ex3_early_stopping.png")


# ============================================================
# Exercise 4: Feature Importance Comparison
# Compare feature importances from RF and GradientBoosting.
# ============================================================
def exercise_4_feature_importance_comparison():
    """Compare feature importances: Random Forest vs GradientBoosting.

    Different algorithms can rank features differently because:
    - RF importance is based on mean decrease in impurity across all trees,
      and trees see random feature subsets at each split.
    - GB importance reflects how much each feature contributes to reducing
      loss in the sequential boosting process, where later trees focus on
      harder examples.

    Comparing both gives a more robust picture of which features truly matter.
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Feature Importance Comparison")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    # Train both models
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    print(f"Random Forest test accuracy:      {rf.score(X_test, y_test):.4f}")
    print(f"GradientBoosting test accuracy:    {gb.score(X_test, y_test):.4f}")

    # Compare top features
    rf_imp = rf.feature_importances_
    gb_imp = gb.feature_importances_

    rf_top10 = np.argsort(rf_imp)[::-1][:10]
    gb_top10 = np.argsort(gb_imp)[::-1][:10]

    print(f"\n{'Rank':<6} {'Random Forest':<30} {'GradientBoosting':<30}")
    print("-" * 66)
    for i in range(10):
        rf_name = cancer.feature_names[rf_top10[i]]
        gb_name = cancer.feature_names[gb_top10[i]]
        print(f"{i+1:<6} {rf_name:<30} {gb_name:<30}")

    # Check overlap in top features
    rf_set = set(rf_top10)
    gb_set = set(gb_top10)
    overlap = rf_set & gb_set
    print(f"\nFeatures in both top-10 lists: {len(overlap)}/10")
    for idx in overlap:
        print(f"  - {cancer.feature_names[idx]}")

    # Side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(range(10), rf_imp[rf_top10][::-1])
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels([cancer.feature_names[i] for i in rf_top10][::-1])
    axes[0].set_title("Random Forest")
    axes[0].set_xlabel("Importance")

    axes[1].barh(range(10), gb_imp[gb_top10][::-1])
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([cancer.feature_names[i] for i in gb_top10][::-1])
    axes[1].set_title("GradientBoosting")
    axes[1].set_xlabel("Importance")

    plt.suptitle("Feature Importance Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("08_ex4_importance_comparison.png", dpi=100)
    plt.close()
    print("\nPlot saved: 08_ex4_importance_comparison.png")


if __name__ == "__main__":
    exercise_1_adaboost_vs_gb()
    exercise_2_gb_hyperparameter_tuning()
    exercise_3_early_stopping()
    exercise_4_feature_importance_comparison()

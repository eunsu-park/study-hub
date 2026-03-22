"""
Model Explainability - Exercise Solutions
==========================================
Lesson 16: Model Explainability

Exercises cover:
  1. Explain a classification model with permutation importance
  2. Compare feature importance methods (permutation vs impurity)
  3. Fairness audit on synthetic demographic data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# Exercise 1: Explain a Classification Model
# Permutation importance + partial dependence on breast cancer.
# ============================================================
def exercise_1_explain_classification():
    """Explain GradientBoosting predictions using permutation importance.

    Permutation importance measures how much a model's performance drops
    when a single feature's values are randomly shuffled, breaking its
    relationship with the target. Unlike impurity-based importance, it:
    - Works for any model (model-agnostic)
    - Accounts for feature interactions (shuffling breaks them)
    - Is not biased toward high-cardinality features

    We also show partial dependence plots (PDP) for the top features,
    which reveal how changing a feature's value affects the prediction
    while averaging over all other features.
    """
    print("=" * 60)
    print("Exercise 1: Explain a Classification Model")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    # Train model
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    print(f"Test accuracy: {gb.score(X_test, y_test):.4f}")

    # Permutation importance on the test set -- measured on held-out data
    # to reflect generalization importance, not training importance
    perm_imp = permutation_importance(
        gb, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1
    )

    sorted_idx = perm_imp.importances_mean.argsort()[::-1]

    print(f"\nTop 10 features (permutation importance):")
    for i in range(10):
        idx = sorted_idx[i]
        print(f"  {i+1:2d}. {cancer.feature_names[idx]:<30s} "
              f"{perm_imp.importances_mean[idx]:.4f} "
              f"+/- {perm_imp.importances_std[idx]:.4f}")

    # Partial dependence plots for top 2 features
    top_features = [sorted_idx[0], sorted_idx[1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat_idx in zip(axes, top_features):
        display = PartialDependenceDisplay.from_estimator(
            gb, X_train, [feat_idx], ax=ax,
            feature_names=cancer.feature_names
        )
        ax.set_title(f"PDP: {cancer.feature_names[feat_idx]}")

    plt.suptitle("Partial Dependence Plots", fontsize=13)
    plt.tight_layout()
    plt.savefig("16_ex1_pdp.png", dpi=100)
    plt.close()
    print("\nPlot saved: 16_ex1_pdp.png")

    # Compare permutation importance with impurity-based
    print(f"\n--- Impurity vs Permutation comparison ---")
    print(f"{'Feature':<30s} {'Impurity':>10} {'Permutation':>12}")
    print("-" * 55)
    for i in range(10):
        idx = sorted_idx[i]
        print(f"{cancer.feature_names[idx]:<30s} "
              f"{gb.feature_importances_[idx]:>10.4f} "
              f"{perm_imp.importances_mean[idx]:>12.4f}")


# ============================================================
# Exercise 2: LIME vs SHAP Comparison (simulated)
# Compare impurity-based vs permutation-based importance methods.
# ============================================================
def exercise_2_importance_comparison():
    """Compare two feature importance methods on correlated data.

    When features are correlated, importance methods can disagree:
    - Impurity-based importance may split the credit among correlated
      features, underestimating each one individually.
    - Permutation importance may show low importance for a correlated
      feature because permuting one doesn't help if its twin remains.

    This exercise demonstrates why you should always use multiple
    explanation methods and understand their limitations.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Importance Method Comparison")
    print("=" * 60)

    # Create dataset with correlated features
    np.random.seed(42)
    n = 1000

    # 3 informative features, but feature 0 and 1 are correlated
    X_base = np.random.randn(n, 3)
    X_corr = np.column_stack([
        X_base[:, 0],                        # feature 0
        X_base[:, 0] + np.random.randn(n) * 0.1,  # feature 1 (correlated with 0)
        X_base[:, 1],                        # feature 2 (independent)
        X_base[:, 2],                        # feature 3 (independent)
        np.random.randn(n),                  # feature 4 (noise)
    ])
    y = (X_base[:, 0] + X_base[:, 1] + X_base[:, 2] > 0).astype(int)
    feature_names = [f"feat_{i}" for i in range(5)]

    X_train, X_test, y_train, y_test = train_test_split(
        X_corr, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Test accuracy: {rf.score(X_test, y_test):.4f}")

    # Method 1: Impurity-based (MDI)
    mdi = rf.feature_importances_

    # Method 2: Permutation importance
    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1
    )

    print(f"\n{'Feature':<12} {'Impurity (MDI)':>15} {'Permutation':>15} {'Note':<30}")
    print("-" * 75)
    for i in range(5):
        note = ""
        if i == 1:
            note = "correlated with feat_0"
        elif i == 4:
            note = "noise feature"
        print(f"{feature_names[i]:<12} {mdi[i]:>15.4f} "
              f"{perm.importances_mean[i]:>15.4f} {note:<30}")

    print(f"\nCorrelation between feat_0 and feat_1: "
          f"{np.corrcoef(X_corr[:, 0], X_corr[:, 1])[0, 1]:.4f}")
    print("Note: Correlated features split importance in MDI but")
    print("permutation may also underestimate each correlated feature.")


# ============================================================
# Exercise 3: Fairness Audit
# Audit a classifier for demographic bias on synthetic data.
# ============================================================
def exercise_3_fairness_audit():
    """Fairness audit on a classifier with synthetic demographic data.

    Fairness metrics:
    - Demographic Parity: P(y_hat=1 | group=A) == P(y_hat=1 | group=B)
      Ensures equal positive prediction rates across groups.
    - Equalized Odds: P(y_hat=1 | y=1, group=A) == P(y_hat=1 | y=1, group=B)
      Ensures equal true positive rates (and false positive rates).

    Real-world impact: If a hiring model has different recall for men vs women,
    it systematically misses qualified candidates from one group.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Fairness Audit")
    print("=" * 60)

    # Generate synthetic data with demographic features
    np.random.seed(42)
    n = 2000

    # Demographics
    gender = np.random.choice([0, 1], n)  # 0: group_A, 1: group_B
    age = np.random.randint(18, 65, n)

    # Skill features (should drive the prediction)
    experience = np.random.poisson(5, n) + gender * 0.5  # slight bias in data
    education = np.random.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    test_score = np.random.randn(n) * 10 + 60 + education * 5

    # True outcome (should not depend on gender)
    true_skill = 0.3 * experience + 0.2 * education + 0.01 * test_score
    y = (true_skill + np.random.randn(n) * 0.5 > 2.5).astype(int)

    X = np.column_stack([gender, age, experience, education, test_score])
    feature_names = ["gender", "age", "experience", "education", "test_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model (including gender as a feature)
    clf_with = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_with.fit(X_train, y_train)
    y_pred_with = clf_with.predict(X_test)

    # Train model without gender
    X_train_no_gender = X_train[:, 1:]  # remove gender column
    X_test_no_gender = X_test[:, 1:]
    clf_without = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_without.fit(X_train_no_gender, y_train)
    y_pred_without = clf_without.predict(X_test_no_gender)

    # --- Fairness Metrics ---
    gender_test = X_test[:, 0]

    print("=== Model WITH gender feature ===")
    _compute_fairness(y_test, y_pred_with, gender_test)

    print("\n=== Model WITHOUT gender feature ===")
    _compute_fairness(y_test, y_pred_without, gender_test)

    print(f"\nAccuracy WITH gender:    {accuracy_score(y_test, y_pred_with):.4f}")
    print(f"Accuracy WITHOUT gender: {accuracy_score(y_test, y_pred_without):.4f}")

    print("\nConclusion: Removing the sensitive feature reduces demographic")
    print("parity gap while maintaining comparable accuracy.")


def _compute_fairness(y_true, y_pred, group):
    """Helper to compute fairness metrics for two groups."""
    for g, name in [(0, "Group A"), (1, "Group B")]:
        mask = group == g
        positive_rate = y_pred[mask].mean()
        tp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 1))
        actual_pos = np.sum(y_true[mask] == 1)
        tpr = tp / actual_pos if actual_pos > 0 else 0
        fp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 0))
        actual_neg = np.sum(y_true[mask] == 0)
        fpr = fp / actual_neg if actual_neg > 0 else 0
        print(f"  {name}: positive_rate={positive_rate:.3f}, "
              f"TPR={tpr:.3f}, FPR={fpr:.3f}")

    # Demographic Parity
    rate_a = y_pred[group == 0].mean()
    rate_b = y_pred[group == 1].mean()
    dp_gap = abs(rate_a - rate_b)
    print(f"  Demographic Parity Gap: {dp_gap:.4f} (0 = perfect parity)")

    # Equalized Odds (TPR difference)
    tpr_a = (np.sum((y_pred[group == 0] == 1) & (y_true[group == 0] == 1))
             / max(np.sum(y_true[group == 0] == 1), 1))
    tpr_b = (np.sum((y_pred[group == 1] == 1) & (y_true[group == 1] == 1))
             / max(np.sum(y_true[group == 1] == 1), 1))
    eo_gap = abs(tpr_a - tpr_b)
    print(f"  Equalized Odds Gap (TPR): {eo_gap:.4f} (0 = perfect equality)")


if __name__ == "__main__":
    exercise_1_explain_classification()
    exercise_2_importance_comparison()
    exercise_3_fairness_audit()

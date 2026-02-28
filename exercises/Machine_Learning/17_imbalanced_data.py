"""
Imbalanced Data - Exercise Solutions
======================================
Lesson 17: Imbalanced Data

Exercises cover:
  1. Credit card fraud detection (synthetic) with multiple strategies
  2. Multi-class imbalance handling
  3. Pipeline design for rare medical diagnosis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, f1_score, average_precision_score,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    matthews_corrcoef, precision_score, recall_score
)


# ============================================================
# Exercise 1: Credit Card Fraud Detection
# Compare 4 strategies on highly imbalanced data.
# ============================================================
def exercise_1_fraud_detection():
    """Fraud detection with 4 strategies on imbalanced data.

    Strategies compared:
    1. Baseline: no handling (the model will predict "not fraud" for almost everything)
    2. Class weights: penalize minority misclassification in the loss function
    3. Threshold tuning: adjust decision threshold after training
    4. Oversampling: duplicate minority samples to balance the training set

    Key metric choices:
    - NOT accuracy (99%+ by always predicting majority)
    - F1: harmonic mean of precision and recall
    - PR-AUC: area under precision-recall curve
    - MCC: Matthews Correlation Coefficient, robust to imbalance
    """
    print("=" * 60)
    print("Exercise 1: Credit Card Fraud Detection")
    print("=" * 60)

    # Generate highly imbalanced data (1% fraud)
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.99, 0.01],
        random_state=42,
        flip_y=0.01,
    )
    print(f"Class distribution: {np.bincount(y)} (0=normal, 1=fraud)")
    print(f"Fraud rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # Strategy 1: Baseline (no handling)
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)
    y_proba_base = rf_base.predict_proba(X_test)[:, 1]
    results["Baseline"] = _evaluate(y_test, y_pred_base, y_proba_base)

    # Strategy 2: Class weights
    rf_weighted = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_weighted.fit(X_train, y_train)
    y_pred_w = rf_weighted.predict(X_test)
    y_proba_w = rf_weighted.predict_proba(X_test)[:, 1]
    results["Class Weights"] = _evaluate(y_test, y_pred_w, y_proba_w)

    # Strategy 3: Threshold tuning
    # Use baseline model's probabilities but optimize the threshold
    # to maximize F1 on the training set
    precisions, recalls, thresholds = precision_recall_curve(
        y_train, rf_base.predict_proba(X_train)[:, 1]
    )
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_thresh = (y_proba_base >= optimal_threshold).astype(int)
    results[f"Threshold={optimal_threshold:.2f}"] = _evaluate(y_test, y_pred_thresh, y_proba_base)

    # Strategy 4: Random oversampling (manual, no imblearn needed)
    minority_idx = np.where(y_train == 1)[0]
    majority_idx = np.where(y_train == 0)[0]
    # Duplicate minority samples to match majority count
    oversampled_idx = np.concatenate([
        majority_idx,
        np.random.choice(minority_idx, len(majority_idx), replace=True)
    ])
    np.random.shuffle(oversampled_idx)
    X_over, y_over = X_train[oversampled_idx], y_train[oversampled_idx]

    rf_over = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_over.fit(X_over, y_over)
    y_pred_over = rf_over.predict(X_test)
    y_proba_over = rf_over.predict_proba(X_test)[:, 1]
    results["Oversampling"] = _evaluate(y_test, y_pred_over, y_proba_over)

    # Print comparison
    print(f"\n{'Strategy':<25} {'F1':>8} {'PR-AUC':>8} {'MCC':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['f1']:>8.4f} {metrics['pr_auc']:>8.4f} "
              f"{metrics['mcc']:>8.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>8.4f}")

    # Cost-based analysis: assume FN costs 100x FP
    print(f"\n--- Cost Analysis (FN cost = 100x FP cost) ---")
    for name, metrics in results.items():
        cm = metrics["cm"]
        cost = cm[1, 0] * 100 + cm[0, 1] * 1  # FN * 100 + FP * 1
        print(f"{name:<25}: total cost = {cost:>6d} "
              f"(FN={cm[1,0]}, FP={cm[0,1]})")


def _evaluate(y_true, y_pred, y_proba):
    """Compute evaluation metrics for imbalanced classification."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "f1": f1_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "cm": cm,
    }


# ============================================================
# Exercise 2: Multi-Class Imbalance
# Handle 4-class imbalanced dataset.
# ============================================================
def exercise_2_multiclass_imbalance():
    """Multi-class imbalance with comparison of strategies.

    Multi-class imbalance is harder because:
    - SMOTE creates synthetic samples per class independently
    - class_weight='balanced' adjusts each class weight inversely
      proportional to its frequency
    - Macro F1 treats all classes equally, revealing if the model
      neglects minority classes
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Multi-Class Imbalance")
    print("=" * 60)

    X, y = make_classification(
        n_samples=5000,
        n_classes=4,
        n_informative=10,
        n_redundant=3,
        n_features=15,
        weights=[0.60, 0.25, 0.10, 0.05],
        random_state=42,
    )
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    strategies = {
        "No handling": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "class_weight=balanced": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "class_weight=balanced_subsample": RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
    }

    print(f"\n{'Strategy':<35} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 60)

    for name, clf in strategies.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"{name:<35} {macro_f1:>10.4f} {weighted_f1:>12.4f}")

    # Detailed report for the best strategy
    best_clf = strategies["class_weight=balanced"]
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    print(f"\nDetailed report (class_weight=balanced):")
    print(classification_report(y_test, y_pred))


# ============================================================
# Exercise 3: Pipeline Design for Medical Diagnosis
# Achieve Recall >= 0.95 with Precision >= 0.10.
# ============================================================
def exercise_3_medical_pipeline():
    """Design pipeline for rare disease detection with recall constraint.

    Medical screening requirements:
    - Recall >= 0.95: must catch 95%+ of sick patients
    - Precision >= 0.10: acceptable false alarm rate (10% of flagged are true)

    Strategy:
    1. Train with class_weight to improve baseline recall
    2. Use probability calibration for reliable threshold tuning
    3. Optimize threshold to satisfy the recall constraint
    4. Report real-world impact (patients correctly identified vs false alarms)
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Medical Diagnosis Pipeline")
    print("=" * 60)

    # Extremely imbalanced: 0.5% positive
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=8,
        n_classes=2,
        weights=[0.995, 0.005],
        random_state=42,
        flip_y=0.005,
    )
    print(f"Class distribution: {np.bincount(y)} (0=healthy, 1=disease)")
    print(f"Disease prevalence: {y.mean():.3%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model with balanced weights to boost recall
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )

    # Manual sample weight to handle imbalance
    sample_weight = np.ones(len(y_train))
    sample_weight[y_train == 1] = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    # Get probabilities
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Find optimal threshold for recall >= 0.95
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Search for the highest threshold that achieves recall >= 0.95
    # Higher threshold = higher precision, lower recall
    valid_mask = recalls[:-1] >= 0.95
    if valid_mask.any():
        # Among thresholds achieving recall >= 0.95, pick the one with
        # highest precision (highest threshold)
        valid_thresholds = thresholds[valid_mask]
        valid_precisions = precisions[:-1][valid_mask]
        best_idx = np.argmax(valid_precisions)
        optimal_threshold = valid_thresholds[best_idx]
    else:
        # Fallback: lowest threshold to maximize recall
        optimal_threshold = thresholds[0]

    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Evaluate
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Recall:    {rec:.4f} (target >= 0.95)")
    print(f"Precision: {prec:.4f} (target >= 0.10)")

    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")

    # Real-world impact
    total_patients = len(y_test)
    actual_sick = np.sum(y_test == 1)
    caught = tp
    missed = fn
    false_alarms = fp

    print(f"\n--- Real-World Impact (on {total_patients} patients) ---")
    print(f"  Actual sick patients: {actual_sick}")
    print(f"  Correctly identified: {caught} ({caught/max(actual_sick,1)*100:.1f}%)")
    print(f"  Missed (FN):          {missed}")
    print(f"  False alarms (FP):    {false_alarms}")
    print(f"  Patients needing follow-up: {caught + false_alarms} "
          f"({(caught+false_alarms)/total_patients*100:.1f}% of all)")

    # Check constraints
    rec_ok = "PASS" if rec >= 0.95 else "FAIL"
    prec_ok = "PASS" if prec >= 0.10 else "FAIL"
    print(f"\n  Recall constraint:    {rec_ok} ({rec:.4f} >= 0.95)")
    print(f"  Precision constraint: {prec_ok} ({prec:.4f} >= 0.10)")


if __name__ == "__main__":
    exercise_1_fraud_detection()
    exercise_2_multiclass_imbalance()
    exercise_3_medical_pipeline()

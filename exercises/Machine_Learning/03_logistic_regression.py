"""
Logistic Regression - Exercise Solutions
==========================================
Lesson 03: Logistic Regression

Exercises cover:
  1. Binary classification with F1-score on breast cancer data
  2. Multi-class classification on Iris data with Softmax
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# ============================================================
# Exercise 1: Binary Classification
# Train logistic regression on breast cancer data, compute F1-score.
# ============================================================
def exercise_1_binary_classification():
    """Binary classification with logistic regression and F1-score.

    Why F1-score instead of accuracy? Breast cancer has mild class imbalance
    (~63% benign, ~37% malignant). F1-score balances precision (avoid false
    alarms) and recall (catch real cancers). In medical contexts, recall is
    often more important -- missing a malignant tumor is worse than a false alarm.
    """
    print("=" * 60)
    print("Exercise 1: Binary Classification (Breast Cancer)")
    print("=" * 60)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    print(f"Classes: {cancer.target_names}")
    print(f"Class distribution: {np.bincount(y)} (malignant=0, benign=1)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling is critical for logistic regression -- the solver computes
    # gradients of the log-loss, and unscaled features distort the loss
    # landscape causing slow or unstable convergence
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=cancer.target_names)
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Logistic Regression - Breast Cancer")
    plt.tight_layout()
    plt.savefig("03_ex1_binary_cm.png", dpi=100)
    plt.close()
    print("Confusion matrix saved: 03_ex1_binary_cm.png")


# ============================================================
# Exercise 2: Multi-class Classification
# Perform 3-class classification on Iris data.
# ============================================================
def exercise_2_multiclass_classification():
    """Multi-class logistic regression using Softmax (multinomial).

    Softmax generalizes the sigmoid to K classes by computing
    P(y=k|X) = exp(theta_k . X) / sum_j(exp(theta_j . X)).
    Unlike One-vs-Rest, Softmax jointly optimizes all class boundaries,
    which typically gives better-calibrated probabilities.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Multi-class Classification (Iris)")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Multinomial logistic regression (Softmax)
    model = LogisticRegression(multi_class="multinomial", max_iter=1000,
                               random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"Accuracy: {model.score(X_test, y_test):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Show prediction probabilities for first 5 samples to demonstrate
    # how Softmax distributes probability mass across classes
    print("Prediction probabilities (first 5 samples):")
    for i in range(5):
        probs = ", ".join(f"{p:.3f}" for p in y_proba[i])
        predicted = iris.target_names[y_pred[i]]
        actual = iris.target_names[y_test[i]]
        status = "OK" if y_pred[i] == y_test[i] else "MISS"
        print(f"  Sample {i}: [{probs}] -> {predicted} (actual: {actual}) {status}")

    # Compare OvR vs Multinomial approaches
    print("\n--- OvR vs Multinomial comparison ---")
    from sklearn.model_selection import cross_val_score

    model_ovr = LogisticRegression(multi_class="ovr", max_iter=1000, random_state=42)
    model_multi = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)

    scores_ovr = cross_val_score(model_ovr, X, y, cv=5, scoring="accuracy")
    scores_multi = cross_val_score(model_multi, X, y, cv=5, scoring="accuracy")

    print(f"OvR CV accuracy:         {scores_ovr.mean():.4f} (+/- {scores_ovr.std():.4f})")
    print(f"Multinomial CV accuracy: {scores_multi.mean():.4f} (+/- {scores_multi.std():.4f})")


if __name__ == "__main__":
    exercise_1_binary_classification()
    exercise_2_multiclass_classification()

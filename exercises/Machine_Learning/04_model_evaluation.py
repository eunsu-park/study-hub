"""
Model Evaluation - Exercise Solutions
=======================================
Lesson 04: Model Evaluation

Exercises cover:
  1. Classification evaluation: compute Precision, Recall, F1 from confusion matrix
  2. Regression evaluation: calculate R-squared from predictions
"""

import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt


# ============================================================
# Exercise 1: Classification Evaluation
# Calculate Precision, Recall, and F1-score from confusion matrix values.
# ============================================================
def exercise_1_classification_evaluation():
    """Compute classification metrics manually from confusion matrix components.

    Confusion matrix layout:
                    Predicted
                    Neg    Pos
    Actual  Neg  [  TN=50  FP=10 ]
            Pos  [  FN=5   TP=35 ]

    - Precision = TP/(TP+FP): "Of those we predicted positive, how many truly are?"
      High precision means few false alarms.
    - Recall = TP/(TP+FN): "Of those that are truly positive, how many did we catch?"
      High recall means few missed positives.
    - F1 = harmonic mean of precision and recall -- useful when you need a single
      number that balances both, especially with imbalanced classes.
    """
    print("=" * 60)
    print("Exercise 1: Classification Evaluation")
    print("=" * 60)

    # Given confusion matrix components
    tn, fp, fn, tp = 50, 10, 5, 35

    # Manual calculation
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"  -> {precision*100:.1f}% of predicted positives are truly positive")
    print(f"Recall:    {recall:.4f}")
    print(f"  -> {recall*100:.1f}% of actual positives are correctly identified")
    print(f"F1-Score:  {f1:.4f}")
    print(f"  -> Harmonic mean balances precision and recall")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"  -> {accuracy*100:.1f}% of all predictions are correct")

    # Verification with sklearn
    # Reconstruct y_true and y_pred from the confusion matrix counts
    y_true = [0] * (tn + fp) + [1] * (fn + tp)
    y_pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp

    print(f"\nVerification with sklearn:")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_true, y_pred):.4f}")

    # Scenario analysis: when to prioritize which metric
    print("\n--- Metric Selection Guide ---")
    print("  Medical diagnosis: Prioritize Recall (don't miss sick patients)")
    print("  Spam filter: Balance Precision and Recall (F1)")
    print("  Fraud detection: Prioritize Precision (minimize false alarms)")


# ============================================================
# Exercise 2: Regression Evaluation
# Calculate R-squared from predictions and actual values.
# ============================================================
def exercise_2_regression_evaluation():
    """Calculate R-squared and other regression metrics.

    R-squared (coefficient of determination) measures the proportion of
    variance in y explained by the model:
      R^2 = 1 - SS_res / SS_tot
      where SS_res = sum((y - y_hat)^2) and SS_tot = sum((y - y_mean)^2)

    - R^2 = 1.0: perfect prediction
    - R^2 = 0.0: model is no better than predicting the mean
    - R^2 < 0: model is worse than the mean (possible with bad models)
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Regression Evaluation")
    print("=" * 60)

    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 240, 290])

    # Manual R-squared calculation
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)  # residual sum of squares
    ss_tot = np.sum((y_true - y_mean) ** 2)  # total sum of squares
    r2_manual = 1 - ss_res / ss_tot

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"\nManual R-squared:")
    print(f"  y_mean = {y_mean:.1f}")
    print(f"  SS_res = {ss_res:.1f}")
    print(f"  SS_tot = {ss_tot:.1f}")
    print(f"  R^2 = 1 - {ss_res:.1f}/{ss_tot:.1f} = {r2_manual:.4f}")

    # Sklearn calculation for verification
    r2_sklearn = r2_score(y_true, y_pred)
    print(f"\nsklearn R^2: {r2_sklearn:.4f}")

    # Additional metrics for context
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nAdditional metrics:")
    print(f"  MAE:  {mae:.2f}  (average absolute error)")
    print(f"  MSE:  {mse:.2f}  (penalizes large errors more)")
    print(f"  RMSE: {rmse:.2f}  (same unit as y, interpretable)")

    # Visualization: actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=80, color="steelblue", edgecolors="black")
    plt.plot([90, 310], [90, 310], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted (R² = {r2_sklearn:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_ex2_regression_eval.png", dpi=100)
    plt.close()
    print("\nPlot saved: 04_ex2_regression_eval.png")


if __name__ == "__main__":
    exercise_1_classification_evaluation()
    exercise_2_regression_evaluation()

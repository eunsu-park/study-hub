"""
Machine Learning Overview - Exercise Solutions
================================================
Lesson 01: ML Overview

Exercises cover:
  1. Data Splitting with stratification
  2. Basic Model Training with logistic regression
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# Exercise 1: Data Splitting
# Split iris data into 80:20 while maintaining class proportions.
# ============================================================
def exercise_1_data_splitting():
    """Split iris data with stratification to preserve class balance.

    Why stratify? Without stratification, small test sets may end up with
    an unrepresentative class distribution (e.g., one class entirely missing),
    leading to misleading evaluation results.
    """
    print("=" * 60)
    print("Exercise 1: Data Splitting")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    # stratify=y ensures each class appears in the same proportion in
    # both train and test sets -- critical for small, multi-class datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"\nTraining class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution:     {np.bincount(y_test)}")

    # Verify proportions are preserved
    train_ratio = np.bincount(y_train) / len(y_train)
    test_ratio = np.bincount(y_test) / len(y_test)
    print(f"\nTraining class ratios: {train_ratio}")
    print(f"Test class ratios:     {test_ratio}")
    print("Ratios match closely -- stratification is working correctly.")


# ============================================================
# Exercise 2: Basic Model Training
# Train a logistic regression model and compute accuracy.
# ============================================================
def exercise_2_basic_model_training():
    """Train logistic regression on iris data with proper preprocessing.

    Key ML workflow steps demonstrated:
    1. Split data BEFORE any preprocessing (prevent data leakage)
    2. Fit scaler on train only, transform both train and test
    3. Train model, predict, and evaluate
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Basic Model Training")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    # Step 1: Split data first to prevent information leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 2: Scale features -- logistic regression uses gradient-based
    # optimization, so features on different scales would cause the loss
    # landscape to be elongated, leading to slow convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # transform only on test -- no re-fitting to avoid test data leakage
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Train logistic regression
    # max_iter=200 ensures convergence for this dataset
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Step 4: Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Detailed per-class metrics reveal how the model performs on each species
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Inspect model coefficients for interpretability
    print("Model coefficients (one row per class):")
    for i, name in enumerate(iris.target_names):
        coefs = model.coef_[i]
        print(f"  {name}: {coefs.round(3)}")
    print(f"Intercepts: {model.intercept_.round(3)}")


if __name__ == "__main__":
    exercise_1_data_splitting()
    exercise_2_basic_model_training()

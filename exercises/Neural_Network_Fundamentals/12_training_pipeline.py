"""
12. Training Pipeline - Exercises
===================================
Lesson 12: Training Pipeline

Exercises cover:
  1. Complete pipeline for Iris classification (NumPy only)
  2. K-fold cross-validation
"""

import numpy as np


# ============================================================
# Exercise 1: Iris Classification Pipeline
# Build a complete training pipeline for Iris using only NumPy.
# ============================================================
def exercise_1_iris_pipeline():
    """Complete Iris classification pipeline."""
    print("=" * 60)
    print("Exercise 1: Iris Classification Pipeline")
    print("=" * 60)

    # TODO: Generate synthetic Iris-like data (3 classes, 4 features)
    # Or load from sklearn if available
    # Steps:
    # 1. Generate/load data
    # 2. Standardize features (fit on train only)
    # 3. Split into train/val/test (70/15/15)
    # 4. Build MLP [4, 32, 16, 3]
    # 5. Train with Adam, batch_size=16, 100 epochs
    # 6. Report train/val/test accuracy
    raise NotImplementedError("Build Iris classification pipeline")


# ============================================================
# Exercise 2: 5-Fold Cross-Validation
# Implement 5-fold CV and report mean ± std accuracy.
# ============================================================
def exercise_2_kfold_cv():
    """Implement 5-fold cross-validation."""
    print("\n" + "=" * 60)
    print("Exercise 2: 5-Fold Cross-Validation")
    print("=" * 60)

    # TODO: Implement k_fold_cv function:
    # 1. Split data into k equal folds
    # 2. For each fold: use it as validation, rest as training
    # 3. Train model, evaluate on fold
    # 4. Collect scores
    # 5. Report mean ± std

    # def k_fold_cv(X, Y, k=5, build_and_train_fn):
    #     ...
    raise NotImplementedError("Implement k-fold CV")


if __name__ == "__main__":
    exercise_1_iris_pipeline()
    exercise_2_kfold_cv()

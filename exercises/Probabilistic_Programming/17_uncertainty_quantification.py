"""Exercises for Lesson 17: Uncertainty Quantification — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Compute ECE and plot calibration curve for a classifier.
def exercise_calibration():
    # TODO: Train a classifier (e.g., RandomForest) on make_classification data
    # TODO: Compute ECE on test set
    # TODO: Apply Platt scaling and recompute ECE
    pass

# Exercise 2: Implement split conformal prediction for regression.
def exercise_conformal_prediction():
    # TODO: Split data into train/calibration/test
    # TODO: Fit GradientBoosting on train
    # TODO: Compute conformal intervals on test
    # TODO: Verify coverage >= 1 - alpha
    pass

# Exercise 3: CQR (Conformalized Quantile Regression) for adaptive intervals.
def exercise_cqr():
    # TODO: Train quantile regressors for alpha/2 and 1-alpha/2
    # TODO: Compute conformity scores on calibration set
    # TODO: Adjust intervals for test set
    # TODO: Compare width with split conformal
    pass

# Exercise 4: Decision under uncertainty with expected utility.
def exercise_decision():
    # TODO: Define 3 actions with different cost structures
    # TODO: Given posterior samples of uncertain parameter
    # TODO: Compute expected loss for each action
    # TODO: Find optimal action and EVPI
    pass

if __name__ == "__main__":
    for ex in [exercise_calibration, exercise_conformal_prediction,
               exercise_cqr, exercise_decision]:
        print(f"\n{ex.__name__}")
        ex()

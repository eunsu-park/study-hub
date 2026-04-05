"""Exercises for Lesson 06: Bayesian Regression — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Implement Bayesian linear regression with PyMC. Generate data with
# true_beta=[3, -2, 1], true_sigma=1.5. Compare posterior with OLS estimates.
def exercise_linear_regression():
    # TODO: Generate data, fit PyMC model, compare with np.linalg.lstsq
    pass

# Exercise 2: Bayesian logistic regression on synthetic binary classification data.
# Compute posterior predictive decision boundary with uncertainty.
def exercise_logistic_regression():
    # TODO: Generate binary data, fit logistic model, plot decision boundary ± 2σ
    pass

# Exercise 3: Robust regression with Student-t likelihood.
# Add 5% outliers to linear data and compare Normal vs Student-t models.
def exercise_robust_regression():
    # TODO: Generate data with outliers
    # TODO: Fit Normal and Student-t models
    # TODO: Compare coefficient estimates
    pass

# Exercise 4: Horseshoe prior for sparse regression.
# 20 predictors, only 3 are relevant. Compare Horseshoe vs Normal priors.
def exercise_horseshoe():
    # TODO: Generate sparse data
    # TODO: Fit models with Normal and Horseshoe priors
    # TODO: Compare variable selection performance
    pass

if __name__ == "__main__":
    for ex in [exercise_linear_regression, exercise_logistic_regression,
               exercise_robust_regression, exercise_horseshoe]:
        print(f"\n{ex.__name__}")
        ex()

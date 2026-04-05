"""Exercises for Lesson 16: Model Comparison — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Compare polynomial models using LOO-CV in PyMC/ArviZ.
def exercise_loo_comparison():
    # TODO: Fit degree 1, 2, 3, 5 polynomial models
    # TODO: Compute LOO for each using az.compare()
    # TODO: Print ranking and interpret
    pass

# Exercise 2: Implement posterior predictive checks.
def exercise_ppc():
    # TODO: Fit a Normal model to skewed data
    # TODO: Generate posterior predictive samples
    # TODO: Check mean, std, skewness, min, max statistics
    # TODO: Identify which statistics reveal model misfit
    pass

# Exercise 3: Compute Bayes factor using Savage-Dickey ratio.
def exercise_bayes_factor():
    # TODO: Fit model with a coefficient of interest
    # TODO: Compute Savage-Dickey BF for H0: beta=0
    # TODO: Try different priors and show sensitivity
    pass

# Exercise 4: Model averaging with stacking weights.
def exercise_stacking():
    # TODO: Fit 3 models of varying complexity
    # TODO: Compute stacking weights from LOO
    # TODO: Generate model-averaged predictions
    pass

if __name__ == "__main__":
    for ex in [exercise_loo_comparison, exercise_ppc, exercise_bayes_factor, exercise_stacking]:
        print(f"\n{ex.__name__}")
        ex()

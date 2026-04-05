"""Exercises for Lesson 14: Bayesian Deep Learning — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Implement MC Dropout for regression with uncertainty.
def exercise_mc_dropout():
    # TODO: Train a 2-layer NN with dropout on sin(x) data
    # TODO: At test time, run 100 forward passes with dropout
    # TODO: Plot mean ± 2σ, show uncertainty grows outside training range
    pass

# Exercise 2: Implement Bayes by Backprop (BayesLinear layer).
def exercise_bayes_by_backprop():
    # TODO: Implement BayesLinear with mu and rho parameters
    # TODO: Train with ELBO loss = NLL + KL
    # TODO: Compare with standard NN
    pass

# Exercise 3: Uncertainty decomposition (epistemic vs aleatoric).
def exercise_uncertainty_decomposition():
    # TODO: Build heteroscedastic BNN outputting (mean, log_var)
    # TODO: Train with heteroscedastic Gaussian NLL
    # TODO: Decompose and plot epistemic and aleatoric uncertainty
    pass

# Exercise 4: Deep ensemble for calibrated uncertainty.
def exercise_deep_ensemble():
    # TODO: Train 5 NNs with different random seeds
    # TODO: Use bootstrap sampling for each
    # TODO: Compute ensemble mean and std
    # TODO: Compare calibration with single model
    pass

if __name__ == "__main__":
    for ex in [exercise_mc_dropout, exercise_bayes_by_backprop,
               exercise_uncertainty_decomposition, exercise_deep_ensemble]:
        print(f"\n{ex.__name__}")
        ex()

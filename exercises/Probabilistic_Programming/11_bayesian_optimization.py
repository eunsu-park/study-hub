"""Exercises for Lesson 11: Bayesian Optimization — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Implement Expected Improvement from scratch.
def exercise_ei():
    # TODO: Given GP mean and std at candidate points, compute EI
    # TODO: Verify against a known test case
    pass

# Exercise 2: Full BO loop on Branin function (2D).
def exercise_bo_2d():
    # TODO: Define Branin function
    # TODO: Run 30 iterations of BO
    # TODO: Plot convergence and final surrogate
    pass

# Exercise 3: Compare EI vs UCB vs Thompson Sampling.
def exercise_compare_acquisitions():
    # TODO: Run BO with each acquisition function
    # TODO: Compare convergence speed over 10 random seeds
    pass

# Exercise 4: Hyperparameter tuning of a sklearn model with BO.
def exercise_hyperparameter_tuning():
    # TODO: Use skopt.gp_minimize to tune RandomForest on a dataset
    # TODO: Compare with random search
    pass

if __name__ == "__main__":
    for ex in [exercise_ei, exercise_bo_2d, exercise_compare_acquisitions, exercise_hyperparameter_tuning]:
        print(f"\n{ex.__name__}")
        ex()

"""Exercises for Lesson 09: Gaussian Processes — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Implement GP regression from scratch. Fit to y=sin(x)+noise.
def exercise_gp_regression():
    # TODO: Implement RBF kernel
    # TODO: Implement GP posterior (mean and variance)
    # TODO: Plot predictions with ±2σ bands
    pass

# Exercise 2: Implement Matern-3/2 and periodic kernels. Compare GP samples.
def exercise_kernels():
    # TODO: Implement matern_32 and periodic kernels
    # TODO: Draw 3 prior samples from each
    # TODO: Fit GP regression with each kernel on sin(x) data
    pass

# Exercise 3: Hyperparameter optimization via marginal likelihood.
def exercise_hyperparameter_optimization():
    # TODO: Implement log marginal likelihood
    # TODO: Optimize l and sigma_f using scipy.optimize.minimize
    # TODO: Compare fit before and after optimization
    pass

# Exercise 4: Sparse GP with 10 inducing points on 500 data points.
def exercise_sparse_gp():
    # TODO: Generate 500 data points from sin(x)
    # TODO: Implement FITC or VFE approximation with M=10
    # TODO: Compare predictions and timing with full GP
    pass

if __name__ == "__main__":
    for ex in [exercise_gp_regression, exercise_kernels,
               exercise_hyperparameter_optimization, exercise_sparse_gp]:
        print(f"\n{ex.__name__}")
        ex()

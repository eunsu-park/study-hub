"""
Exercises for Lesson 03: MCMC Fundamentals
Topic: Probabilistic_Programming
"""
import numpy as np
from scipy import stats

# === Exercise 1: Implement Metropolis-Hastings ===
# Implement MH to sample from P(x) ∝ exp(-x^4 + 2x^2)
def exercise_mh_sampler(n_samples=20000, proposal_std=1.0):
    # TODO: Define log_target(x) = -x**4 + 2*x**2
    # TODO: Implement MH loop with Gaussian proposal
    # TODO: Return samples and acceptance rate
    pass

# === Exercise 2: Gibbs Sampler for Bivariate Normal ===
# Sample from N([0,0], [[1, 0.8],[0.8, 1]]) using Gibbs sampling
def exercise_gibbs_bivariate(n_samples=10000, rho=0.8):
    # TODO: Use conditional distributions: X|Y ~ N(rho*y, 1-rho^2)
    # TODO: Alternately sample X|Y and Y|X
    # TODO: Return samples and verify covariance
    pass

# === Exercise 3: Convergence Diagnostics ===
# Run 4 chains and compute R-hat and ESS
def exercise_convergence(chains=None):
    # TODO: If chains is None, generate 4 chains from exercise 1
    # TODO: Compute R-hat
    # TODO: Compute effective sample size
    # TODO: Print diagnostics
    pass

# === Exercise 4: Tune Proposal Distribution ===
# Find the proposal_std that gives ~23% acceptance for a 10D Gaussian
def exercise_tune_proposal(n_dims=10):
    # TODO: Target = N(0, I_10)
    # TODO: Start with proposal_std=1.0
    # TODO: Adaptively adjust to hit 23% acceptance
    pass

# === Exercise 5: Compare MH vs Gibbs ===
# Sample from a 2D correlated Gaussian using both MH and Gibbs
# Compare ESS and runtime
def exercise_compare_samplers():
    # TODO: Run MH and Gibbs for the same target
    # TODO: Compare ESS per second
    pass

if __name__ == "__main__":
    print("Exercise 1: MH Sampler")
    exercise_mh_sampler()
    print("\nExercise 2: Gibbs Bivariate")
    exercise_gibbs_bivariate()
    print("\nExercise 3: Convergence")
    exercise_convergence()

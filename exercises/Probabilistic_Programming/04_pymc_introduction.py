"""
Exercises for Lesson 04: PyMC Introduction
Topic: Probabilistic_Programming
"""
import numpy as np

# === Exercise 1: Poisson Model ===
# Build a PyMC model: lambda ~ Gamma(2, 1), data ~ Poisson(lambda)
# Data: [3, 5, 2, 4, 6, 3, 5, 4, 7, 2]
def exercise_poisson_model():
    # TODO: Build PyMC model with Gamma prior on lambda
    # TODO: Sample and print summary
    # TODO: Compare with analytical posterior Gamma(2+sum(data), 1+n)
    pass

# === Exercise 2: Normal with Unknown Mean and Variance ===
# Estimate mu and sigma from data = np.random.normal(10, 3, 50)
def exercise_normal_model():
    # TODO: Priors: mu ~ Normal(0, 20), sigma ~ HalfNormal(10)
    # TODO: Sample, check convergence, plot posterior
    pass

# === Exercise 3: Posterior Predictive Check ===
# Fit a Normal model to skewed data and show PPC failure
def exercise_ppc_failure():
    # TODO: Generate skewed data (e.g., Gamma or LogNormal)
    # TODO: Fit Normal model
    # TODO: Run PPC and check skewness statistic
    pass

# === Exercise 4: Prior Predictive Simulation ===
# Check if Normal(0, 100) prior for heights is reasonable
def exercise_prior_predictive():
    # TODO: Build model for human heights
    # TODO: Sample from prior predictive
    # TODO: Check if simulated heights are plausible
    pass

# === Exercise 5: Two-Sample Comparison ===
# Bayesian t-test: compare two groups
def exercise_bayesian_ttest():
    # TODO: Generate two groups with different means
    # TODO: Build hierarchical model for difference
    # TODO: Compute P(mu_1 > mu_2) and effect size
    pass

if __name__ == "__main__":
    print("Exercise 1: Poisson Model")
    exercise_poisson_model()
    print("\nExercise 2: Normal Model")
    exercise_normal_model()

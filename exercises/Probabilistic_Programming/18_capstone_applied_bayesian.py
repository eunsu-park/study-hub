"""Exercises for Lesson 18: Capstone Applied Bayesian — Topic: Probabilistic_Programming"""
import numpy as np

# === Capstone Project: End-to-End Bayesian A/B Testing Platform ===
# Implement a complete Bayesian A/B testing pipeline.

# Exercise 1: Data simulation and model specification.
def exercise_data_and_model():
    # TODO: Simulate 3-variant A/B test data
    # TODO: Build hierarchical Beta-Binomial model in PyMC
    # TODO: Include revenue-weighted comparison
    pass

# Exercise 2: Full Bayesian workflow.
def exercise_full_workflow():
    # TODO: Prior predictive check
    # TODO: Fit model with MCMC (4 chains)
    # TODO: Check convergence (R-hat, ESS, divergences)
    # TODO: Posterior predictive check
    pass

# Exercise 3: Decision metrics and reporting.
def exercise_decision_metrics():
    # TODO: Compute P(each variant is best)
    # TODO: Compute expected loss for each variant
    # TODO: Compute expected revenue per variant
    # TODO: Generate executive summary
    pass

# Exercise 4: Sequential monitoring with early stopping.
def exercise_sequential_monitoring():
    # TODO: Implement day-by-day Bayesian updating
    # TODO: Stop when P(best variant) > 0.95
    # TODO: Compare with fixed-horizon test
    pass

# Exercise 5: Sensitivity analysis.
def exercise_sensitivity():
    # TODO: Run the analysis with 3 different priors
    # TODO: Compare posterior conclusions
    # TODO: Report which findings are robust
    pass

if __name__ == "__main__":
    for ex in [exercise_data_and_model, exercise_full_workflow,
               exercise_decision_metrics, exercise_sequential_monitoring,
               exercise_sensitivity]:
        print(f"\n{ex.__name__}")
        ex()

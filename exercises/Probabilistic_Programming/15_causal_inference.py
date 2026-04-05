"""Exercises for Lesson 15: Causal Inference — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Backdoor adjustment. Generate confounded data and estimate the
# causal effect using both naive and adjusted regression.
def exercise_backdoor():
    # TODO: Generate Z→X→Y with confounder C→X, C→Y
    # TODO: Compute naive vs adjusted causal effect
    # TODO: Show adjusted estimate is closer to truth
    pass

# Exercise 2: Instrumental variable estimation with 2SLS.
def exercise_iv():
    # TODO: Generate data with instrument Z, confounder U, treatment X, outcome Y
    # TODO: Implement 2SLS
    # TODO: Compare with naive regression
    pass

# Exercise 3: Bayesian causal effect with PyMC.
def exercise_bayesian_causal():
    # TODO: Build PyMC model controlling for confounders
    # TODO: Compute posterior of causal effect
    # TODO: Report 95% HDI and P(effect > 0)
    pass

# Exercise 4: Propensity score matching for observational study.
def exercise_propensity_matching():
    # TODO: Generate observational data with treatment selection bias
    # TODO: Estimate propensity scores with logistic regression
    # TODO: Match treated/control and estimate ATE
    pass

if __name__ == "__main__":
    for ex in [exercise_backdoor, exercise_iv, exercise_bayesian_causal, exercise_propensity_matching]:
        print(f"\n{ex.__name__}")
        ex()

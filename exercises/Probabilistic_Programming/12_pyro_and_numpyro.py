"""Exercises for Lesson 12: Pyro and NumPyro — Topic: Probabilistic_Programming"""

# Exercise 1: Build a Pyro model for Bayesian linear regression with SVI.
def exercise_pyro_regression():
    # TODO: Define model with Normal priors on weights
    # TODO: Define mean-field guide
    # TODO: Train with SVI for 3000 steps
    # TODO: Compare learned params with true values
    pass

# Exercise 2: Use AutoGuide (AutoDiagonalNormal) instead of manual guide.
def exercise_autoguide():
    # TODO: Use Pyro's AutoDiagonalNormal
    # TODO: Compare convergence with manual guide
    pass

# Exercise 3: NumPyro NUTS sampling for a hierarchical model.
def exercise_numpyro_hierarchical():
    # TODO: Define a hierarchical model in NumPyro
    # TODO: Run NUTS with 4 chains
    # TODO: Print summary and check diagnostics
    pass

# Exercise 4: Compare Pyro SVI vs NumPyro NUTS on the same model.
def exercise_svi_vs_nuts():
    # TODO: Run both on a regression model
    # TODO: Compare wall-clock time and posterior estimates
    pass

if __name__ == "__main__":
    for ex in [exercise_pyro_regression, exercise_autoguide,
               exercise_numpyro_hierarchical, exercise_svi_vs_nuts]:
        print(f"\n{ex.__name__}")
        ex()

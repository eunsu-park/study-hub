"""Exercises for Lesson 07: Stan and CmdStanPy — Topic: Probabilistic_Programming"""

# Exercise 1: Write a Stan model for Bayesian linear regression.
# Include generated quantities for posterior predictive and log_lik.
def exercise_stan_regression():
    # TODO: Write Stan code string for linear regression
    # TODO: Save to temp file, compile, and run with CmdStanPy
    # TODO: Convert to ArviZ and print summary
    pass

# Exercise 2: Hierarchical Normal model in Stan (non-centered).
# Implement the eight schools model.
def exercise_stan_hierarchical():
    # TODO: Write non-centered eight schools Stan code
    # TODO: Run with adapt_delta=0.95
    # TODO: Check for divergences
    pass

# Exercise 3: Zero-Inflated Poisson in Stan.
# Write a custom log-likelihood using the functions block.
def exercise_stan_zip():
    # TODO: Write ZIP model with custom zip_lpmf function
    # TODO: Generate synthetic ZIP data
    # TODO: Fit and recover parameters
    pass

# Exercise 4: Compare Stan optimize() vs sample().
# Run both MAP and MCMC on the same model. Compare speed and estimates.
def exercise_map_vs_mcmc():
    # TODO: Run model.optimize() for MAP
    # TODO: Run model.sample() for MCMC
    # TODO: Compare estimates and timing
    pass

if __name__ == "__main__":
    for ex in [exercise_stan_regression, exercise_stan_hierarchical,
               exercise_stan_zip, exercise_map_vs_mcmc]:
        print(f"\n{ex.__name__}")
        ex()

"""
Exercises for Lesson 01: Bayesian Thinking
Topic: Probabilistic_Programming
"""
import numpy as np
from scipy import stats


# === Exercise 1: Implement Bayesian Updating ===
# Problem: Implement a BetaBinomialModel class that performs
# sequential Bayesian updating.

class BetaBinomialModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        # TODO: Store alpha and beta parameters
        pass

    def update(self, n_successes, n_trials):
        # TODO: Update alpha and beta with observed data
        pass

    @property
    def mean(self):
        # TODO: Return the posterior mean
        pass

    def credible_interval(self, level=0.95):
        # TODO: Return (lower, upper) credible interval
        pass


# === Exercise 2: Grid Approximation ===
# Problem: Implement grid approximation for a Poisson-Gamma model.
# Data: counts = [3, 5, 2, 4, 6, 3, 5, 4]
# Prior: Gamma(alpha=2, beta=1)
# Compute the posterior for lambda (the Poisson rate).

def poisson_grid_approximation(counts, alpha_prior=2, beta_prior=1, n_grid=1000):
    # TODO: Create a grid for lambda from 0.01 to 15
    # TODO: Compute log prior (Gamma)
    # TODO: Compute log likelihood (Poisson)
    # TODO: Compute and normalize posterior
    # TODO: Return lambda_grid, posterior
    pass


# === Exercise 3: Prior Sensitivity ===
# Problem: For the coin flip data (7 heads out of 10),
# compute the posterior mean and 95% CI for three different priors:
# 1. Flat: Beta(1, 1)
# 2. Skeptical: Beta(10, 10)
# 3. Biased: Beta(2, 8)

def prior_sensitivity(n_heads=7, n_total=10):
    priors = {
        "Flat": (1, 1),
        "Skeptical": (10, 10),
        "Biased": (2, 8),
    }
    # TODO: For each prior, compute posterior Beta parameters
    # TODO: Compute and print posterior mean and 95% CI
    pass


# === Exercise 4: Predictive Distribution ===
# Problem: Given posterior Beta(9, 5) from a coin flip experiment,
# compute P(next 10 flips have exactly 7 heads) using the
# posterior predictive distribution.

def posterior_predictive(alpha_post=9, beta_post=5, n_future=10, k=7, n_samples=100000):
    # TODO: Sample theta from Beta(alpha_post, beta_post)
    # TODO: For each theta, compute P(k heads | theta, n_future)
    # TODO: Return the average (this is the posterior predictive probability)
    pass


# === Exercise 5: MAP vs Posterior Mean ===
# Problem: For data = [1, 1, 0, 1, 1, 1, 0, 1] with Beta(3, 3) prior,
# compute and compare: MLE, MAP, posterior mean, posterior median.

def compare_estimates(data=None, alpha=3, beta=3):
    if data is None:
        data = np.array([1, 1, 0, 1, 1, 1, 0, 1])
    # TODO: Compute MLE = k/n
    # TODO: Compute MAP = (alpha + k - 1) / (alpha + beta + n - 2)
    # TODO: Compute posterior mean = (alpha + k) / (alpha + beta + n)
    # TODO: Compute posterior median = stats.beta.ppf(0.5, alpha+k, beta+n-k)
    # TODO: Print all four estimates
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Bayesian Updating ===")
    model = BetaBinomialModel(2, 2)
    for heads, total in [(7, 10), (8, 10), (6, 10)]:
        model.update(heads, total)
    print(f"  Final mean: {model.mean}")

    print("\n=== Exercise 2: Grid Approximation ===")
    result = poisson_grid_approximation([3, 5, 2, 4, 6, 3, 5, 4])
    print(f"  Result: {result}")

    print("\n=== Exercise 3: Prior Sensitivity ===")
    prior_sensitivity()

    print("\n=== Exercise 4: Predictive Distribution ===")
    prob = posterior_predictive()
    print(f"  P(7 heads in 10 flips): {prob}")

    print("\n=== Exercise 5: MAP vs Posterior Mean ===")
    compare_estimates()

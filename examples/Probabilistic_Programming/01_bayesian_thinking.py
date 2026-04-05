"""
Bayesian Thinking Examples
- Bayes theorem, conjugate priors, sequential updating, prior sensitivity
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# === Example 1: Beta-Binomial Conjugate Updating ===
def beta_binomial_updating():
    """Sequential Bayesian updating with Beta-Binomial conjugacy."""
    alpha, beta_param = 2.0, 2.0
    data_batches = [(7, 10), (8, 10), (6, 10), (9, 10)]
    theta = np.linspace(0, 1, 500)

    print("Sequential Bayesian Updating:")
    print(f"  Prior: Beta({alpha:.0f}, {beta_param:.0f}), mean={alpha/(alpha+beta_param):.3f}")
    for i, (heads, total) in enumerate(data_batches):
        alpha += heads
        beta_param += total - heads
        mean = alpha / (alpha + beta_param)
        lo, hi = stats.beta.ppf([0.025, 0.975], alpha, beta_param)
        print(f"  Batch {i+1} ({heads}/{total}): Beta({alpha:.0f},{beta_param:.0f}), "
              f"mean={mean:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")


# === Example 2: Prior Sensitivity Analysis ===
def prior_sensitivity():
    """Compare posteriors under different priors."""
    n_heads, n_tails = 7, 3
    priors = {
        "Flat Beta(1,1)": (1, 1),
        "Jeffreys Beta(0.5,0.5)": (0.5, 0.5),
        "Weak Beta(2,2)": (2, 2),
        "Strong Beta(50,50)": (50, 50),
    }
    print("\nPrior Sensitivity Analysis (data: 7H, 3T):")
    for name, (a, b) in priors.items():
        a_post, b_post = a + n_heads, b + n_tails
        mean = a_post / (a_post + b_post)
        print(f"  {name:25s} -> posterior mean = {mean:.4f}")


# === Example 3: Grid Approximation ===
def grid_approximation():
    """Compute posterior via grid approximation."""
    data = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    theta_grid = np.linspace(0, 1, 1000)
    log_prior = stats.beta.logpdf(theta_grid, 2, 2)
    k, n = data.sum(), len(data)
    log_lik = k * np.log(theta_grid + 1e-10) + (n-k) * np.log(1 - theta_grid + 1e-10)
    log_post = log_prior + log_lik
    log_post -= log_post.max()
    posterior = np.exp(log_post)
    posterior /= np.trapz(posterior, theta_grid)
    post_mean = np.trapz(theta_grid * posterior, theta_grid)
    exact_mean = (2 + k) / (4 + n)
    print(f"\nGrid Approximation: mean={post_mean:.4f}, exact={exact_mean:.4f}")


# === Example 4: A/B Test Decision ===
def ab_test_decision():
    """Bayesian A/B test with decision metrics."""
    np.random.seed(42)
    post_a = np.random.beta(1 + 120, 1 + 880, 50000)
    post_b = np.random.beta(1 + 145, 1 + 855, 50000)
    print(f"\nA/B Test Decision:")
    print(f"  P(B > A):      {(post_b > post_a).mean():.4f}")
    print(f"  Expected lift:  {(post_b - post_a).mean():.5f}")
    print(f"  Risk of B:      {np.maximum(post_a - post_b, 0).mean():.5f}")


if __name__ == "__main__":
    beta_binomial_updating()
    prior_sensitivity()
    grid_approximation()
    ab_test_decision()

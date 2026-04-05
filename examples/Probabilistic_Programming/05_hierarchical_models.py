"""
Hierarchical Models Examples
- Partial pooling, shrinkage, eight schools, non-centered parameterization
"""
import numpy as np


def shrinkage_demo():
    """Demonstrate shrinkage in a hierarchical model."""
    np.random.seed(42)
    n_groups = 10
    true_mu = 0.3
    true_sigma = 0.05
    true_theta = np.random.normal(true_mu, true_sigma, n_groups)
    n_obs = np.random.randint(10, 200, n_groups)
    observed = np.array([np.random.binomial(n, t) / n for n, t in zip(n_obs, true_theta)])

    # Empirical Bayes shrinkage estimate
    grand_mean = np.average(observed, weights=n_obs)
    shrinkage_est = []
    for i in range(n_groups):
        B = true_sigma**2 / (true_sigma**2 + 1/(4*n_obs[i]))
        shrinkage_est.append(grand_mean + B * (observed[i] - grand_mean))

    print("Shrinkage Demo:")
    print(f"{'Group':>6} {'n':>5} {'Observed':>10} {'Shrunk':>10} {'True':>10}")
    for i in range(n_groups):
        print(f"{i:6d} {n_obs[i]:5d} {observed[i]:10.4f} {shrinkage_est[i]:10.4f} {true_theta[i]:10.4f}")


def eight_schools_data():
    """Print the eight schools dataset."""
    schools = ["A", "B", "C", "D", "E", "F", "G", "H"]
    y = [28, 8, -3, 7, -1, 1, 18, 12]
    sigma = [15, 10, 16, 11, 9, 11, 10, 18]
    print("\nEight Schools Data:")
    for s, yi, si in zip(schools, y, sigma):
        print(f"  School {s}: effect = {yi:3d} ± {si}")


if __name__ == "__main__":
    shrinkage_demo()
    eight_schools_data()

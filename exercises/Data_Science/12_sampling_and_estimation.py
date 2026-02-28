"""
Exercises for Lesson 12: Sampling and Estimation
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


# === Exercise 1: Sampling Distribution ===
# Problem: Normal population with mu=100, sigma=15, sample size n=36.
#   (a) Expected value and standard error of the sample mean
#   (b) P(95 <= X_bar <= 105)
def exercise_1():
    """Solution computing the sampling distribution of the sample mean.

    When sampling from N(mu, sigma^2), the sample mean follows:
        X_bar ~ N(mu, sigma^2/n)
    This is exact for normal populations (no CLT approximation needed).
    """
    mu, sigma, n = 100, 15, 36

    # (a) The sampling distribution of X_bar
    E_xbar = mu                     # E[X_bar] = mu (unbiased)
    SE = sigma / np.sqrt(n)         # Standard Error = sigma / sqrt(n)

    print(f"Population: N(mu={mu}, sigma={sigma})")
    print(f"Sample size: n={n}")
    print(f"\n(a) Sampling distribution of X_bar:")
    print(f"    E[X_bar] = {E_xbar}")
    print(f"    SE(X_bar) = sigma/sqrt(n) = {sigma}/sqrt({n}) = {SE:.4f}")

    # (b) P(95 <= X_bar <= 105) using the exact normal distribution
    z_lower = (95 - mu) / SE
    z_upper = (105 - mu) / SE
    prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)

    print(f"\n(b) P(95 <= X_bar <= 105):")
    print(f"    Z_lower = (95 - {mu}) / {SE:.4f} = {z_lower:.4f}")
    print(f"    Z_upper = (105 - {mu}) / {SE:.4f} = {z_upper:.4f}")
    print(f"    P(95 <= X_bar <= 105) = {prob:.6f}")
    print(f"    This means {prob*100:.2f}% of all possible sample means fall within 95-105.")


# === Exercise 2: MLE ===
# Problem: Given Poisson data, find MLE for lambda and 95% CI.
def exercise_2():
    """Solution for Maximum Likelihood Estimation of a Poisson parameter.

    For Poisson(lambda), the MLE is simply the sample mean:
        lambda_hat = X_bar
    The 95% CI can be constructed using the asymptotic normality of the MLE:
        lambda_hat +/- z_{0.025} * sqrt(lambda_hat / n)
    """
    data = np.array([3, 5, 4, 2, 6, 3, 4, 5, 2, 4])
    n = len(data)

    # MLE for Poisson lambda is the sample mean
    # Proof: L(lambda) = prod(e^{-lambda} * lambda^{x_i} / x_i!)
    # log L = -n*lambda + sum(x_i)*log(lambda) - sum(log(x_i!))
    # d/d_lambda log L = -n + sum(x_i)/lambda = 0
    # => lambda_hat = sum(x_i)/n = X_bar
    lambda_hat = data.mean()

    print(f"Data: {data.tolist()}")
    print(f"n = {n}")
    print(f"\nMLE for lambda: lambda_hat = X_bar = {lambda_hat:.4f}")

    # 95% CI using asymptotic normality
    # The Fisher information for Poisson is I(lambda) = n/lambda
    # Asymptotic variance of MLE: 1/I(lambda) = lambda/n
    # SE = sqrt(lambda_hat / n)
    se = np.sqrt(lambda_hat / n)
    z_crit = stats.norm.ppf(0.975)  # 1.96 for 95% CI

    ci_lower = lambda_hat - z_crit * se
    ci_upper = lambda_hat + z_crit * se

    print(f"\n95% Confidence Interval (Wald):")
    print(f"  SE = sqrt(lambda_hat / n) = sqrt({lambda_hat:.4f}/{n}) = {se:.4f}")
    print(f"  CI = {lambda_hat:.4f} +/- {z_crit:.4f} * {se:.4f}")
    print(f"  CI = ({ci_lower:.4f}, {ci_upper:.4f})")

    # Verification: scipy MLE
    # For Poisson, scipy.stats.poisson.fit isn't directly available,
    # but we can use the negative log-likelihood approach
    neg_log_lik = lambda lam: -np.sum(stats.poisson.logpmf(data, mu=lam))
    result = minimize_scalar(neg_log_lik, bounds=(0.1, 20), method='bounded')
    print(f"\n  Scipy MLE verification: lambda_hat = {result.x:.4f}")

    # Exact CI using chi-squared distribution (more appropriate for small n)
    # 95% CI: [chi2(0.025, 2*sum(x)) / (2n), chi2(0.975, 2*sum(x)+2) / (2n)]
    total = data.sum()
    ci_exact_lower = stats.chi2.ppf(0.025, 2 * total) / (2 * n)
    ci_exact_upper = stats.chi2.ppf(0.975, 2 * total + 2) / (2 * n)
    print(f"\n  Exact CI (chi-squared): ({ci_exact_lower:.4f}, {ci_exact_upper:.4f})")


# === Exercise 3: Method of Moments ===
# Problem: Derive MoM estimators for U(a, b).
#   Hint: E[X] = (a+b)/2, Var(X) = (b-a)^2/12
def exercise_3():
    """Solution for Method of Moments estimation of Uniform(a, b) parameters.

    Method of moments: set sample moments equal to population moments.
    1st moment: E[X] = (a+b)/2 = X_bar
    2nd moment: Var(X) = (b-a)^2/12 = S^2

    Solving:
        a + b = 2 * X_bar         ... (1)
        b - a = sqrt(12 * S^2)     ... (2)

    From (1) and (2):
        a = X_bar - sqrt(3) * S
        b = X_bar + sqrt(3) * S
    """
    # Generate sample data from U(2, 8) to verify
    np.random.seed(42)
    true_a, true_b = 2, 8
    data = np.random.uniform(true_a, true_b, 200)

    print(f"True parameters: a={true_a}, b={true_b}")
    print(f"Sample size: n={len(data)}")

    # Sample statistics
    x_bar = data.mean()
    s_squared = data.var(ddof=1)  # unbiased sample variance

    print(f"\nSample statistics:")
    print(f"  X_bar = {x_bar:.4f}")
    print(f"  S^2 = {s_squared:.4f}")
    print(f"  S = {np.sqrt(s_squared):.4f}")

    # Method of moments estimators
    # From E[X] = (a+b)/2 and Var(X) = (b-a)^2/12
    # We get: a_hat = X_bar - sqrt(3*S^2), b_hat = X_bar + sqrt(3*S^2)
    a_hat = x_bar - np.sqrt(3 * s_squared)
    b_hat = x_bar + np.sqrt(3 * s_squared)

    print(f"\nMethod of Moments estimators:")
    print(f"  a_hat = X_bar - sqrt(3*S^2) = {x_bar:.4f} - {np.sqrt(3 * s_squared):.4f} = {a_hat:.4f}")
    print(f"  b_hat = X_bar + sqrt(3*S^2) = {x_bar:.4f} + {np.sqrt(3 * s_squared):.4f} = {b_hat:.4f}")
    print(f"\n  True: a={true_a}, b={true_b}")
    print(f"  MoM:  a_hat={a_hat:.4f}, b_hat={b_hat:.4f}")
    print(f"  Error: |a_hat - a| = {abs(a_hat - true_a):.4f}, |b_hat - b| = {abs(b_hat - true_b):.4f}")

    # Compare with MLE (which uses min and max of the sample)
    a_mle = data.min()
    b_mle = data.max()
    print(f"\n  MLE comparison: a_mle={a_mle:.4f}, b_mle={b_mle:.4f}")
    print(f"  Note: MLE for Uniform uses sample min/max, while MoM uses mean/variance.")
    print(f"  MLE is biased inward but has smaller MSE for large samples.")


if __name__ == "__main__":
    print("=== Exercise 1: Sampling Distribution ===")
    exercise_1()
    print("\n=== Exercise 2: MLE ===")
    exercise_2()
    print("\n=== Exercise 3: Method of Moments ===")
    exercise_3()
    print("\nAll exercises completed!")

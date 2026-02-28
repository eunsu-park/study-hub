"""
Exercises for Lesson 09: Maximum Likelihood and MAP
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, minimize


# === Exercise 1: MLE for Exponential Distribution ===
# Problem: Derive and compute MLE for Exp(lambda).
# p(x|lambda) = lambda * exp(-lambda * x)

def exercise_1():
    """MLE for exponential distribution."""
    print("Exponential distribution: p(x|lambda) = lambda * exp(-lambda*x)")
    print()
    print("(a) Log-likelihood:")
    print("  L(lambda) = sum log(lambda * e^{-lambda*x_i})")
    print("            = n*log(lambda) - lambda * sum(x_i)")
    print()
    print("(b) MLE derivation:")
    print("  dL/d(lambda) = n/lambda - sum(x_i) = 0")
    print("  lambda_MLE = n / sum(x_i) = 1 / x_bar")
    print()

    # (c) Numerical verification
    np.random.seed(42)
    true_lambda = 2.5
    n = 500
    data = np.random.exponential(1.0 / true_lambda, n)

    # MLE
    lambda_mle = 1.0 / np.mean(data)

    # Scipy verification
    loc, scale = stats.expon.fit(data, floc=0)
    lambda_scipy = 1.0 / scale

    print(f"(c) Simulation (n={n}, true lambda={true_lambda}):")
    print(f"  Sample mean: {np.mean(data):.4f} (expected: {1/true_lambda:.4f})")
    print(f"  MLE lambda:  {lambda_mle:.4f}")
    print(f"  Scipy fit:   {lambda_scipy:.4f}")
    print(f"  True lambda: {true_lambda:.4f}")

    # Log-likelihood at various lambda values
    print(f"\n  Log-likelihood comparison:")
    for lam in [1.0, 2.0, lambda_mle, 3.0, 4.0]:
        ll = n * np.log(lam) - lam * np.sum(data)
        print(f"    lambda={lam:.3f}: log-lik = {ll:.2f}")


# === Exercise 2: MAP with Different Priors ===
# Problem: Compare Gaussian prior (L2 reg) vs Laplace prior (L1 reg).

def exercise_2():
    """MAP estimation with Gaussian and Laplace priors."""
    np.random.seed(42)

    # Generate data
    n = 50
    X = np.random.randn(n, 1) * 2
    true_w = 3.0
    true_b = 1.0
    sigma = 1.0
    y = true_w * X.flatten() + true_b + np.random.randn(n) * sigma

    # Add intercept
    X_design = np.column_stack([X, np.ones(n)])

    print(f"True model: y = {true_w}x + {true_b} + noise")
    print()

    # MLE (no regularization)
    w_mle = np.linalg.lstsq(X_design, y, rcond=None)[0]
    print(f"MLE solution: w={w_mle[0]:.4f}, b={w_mle[1]:.4f}")

    # (a) MAP with Gaussian prior (L2 regularization)
    # -log posterior = (1/2sigma^2)||y-Xw||^2 + (1/2*sigma_w^2)||w||^2
    sigma_w = 1.0  # prior std
    lambda_l2 = sigma**2 / sigma_w**2

    def neg_log_posterior_l2(theta):
        residual = y - X_design @ theta
        return 0.5 * np.sum(residual**2) / sigma**2 + 0.5 * np.sum(theta**2) / sigma_w**2

    # Analytical: w_MAP = (X^T X + lambda*I)^{-1} X^T y
    w_map_l2 = np.linalg.solve(
        X_design.T @ X_design + lambda_l2 * np.eye(2),
        X_design.T @ y
    )
    print(f"\n(a) MAP with Gaussian prior (L2, sigma_w={sigma_w}):")
    print(f"    w={w_map_l2[0]:.4f}, b={w_map_l2[1]:.4f}")
    print(f"    Equivalent to Ridge regression with lambda={lambda_l2:.2f}")

    # (b) MAP with Laplace prior (L1 regularization)
    # -log posterior = (1/2sigma^2)||y-Xw||^2 + (1/b)||w||_1
    b_laplace = 1.0
    lambda_l1 = sigma**2 / b_laplace

    def neg_log_posterior_l1(theta):
        residual = y - X_design @ theta
        return 0.5 * np.sum(residual**2) / sigma**2 + np.sum(np.abs(theta)) / b_laplace

    result_l1 = minimize(neg_log_posterior_l1, x0=np.zeros(2), method='Nelder-Mead')
    w_map_l1 = result_l1.x

    print(f"\n(b) MAP with Laplace prior (L1, b={b_laplace}):")
    print(f"    w={w_map_l1[0]:.4f}, b={w_map_l1[1]:.4f}")
    print(f"    Equivalent to Lasso regression with lambda={lambda_l1:.2f}")

    # Compare
    print(f"\nComparison:")
    print(f"  MLE:            w={w_mle[0]:.4f}, b={w_mle[1]:.4f}")
    print(f"  MAP (Gaussian): w={w_map_l2[0]:.4f}, b={w_map_l2[1]:.4f}")
    print(f"  MAP (Laplace):  w={w_map_l1[0]:.4f}, b={w_map_l1[1]:.4f}")
    print(f"  True:           w={true_w:.4f}, b={true_b:.4f}")


# === Exercise 3: EM for Coin Toss ===
# Problem: Two coins with unknown head probabilities, estimate via EM.

def exercise_3():
    """EM algorithm for mixture of two coins."""
    # Observed: 5 trials of 10 flips each
    # Each trial uses one of two coins (unknown which)
    observations = np.array([
        [5, 5],   # 5 heads, 5 tails
        [9, 1],   # 9 heads, 1 tail
        [8, 2],   # 8 heads, 2 tails
        [4, 6],   # 4 heads, 6 tails
        [7, 3],   # 7 heads, 3 tails
    ])

    n_trials = len(observations)
    n_flips = observations.sum(axis=1)
    heads = observations[:, 0]

    # Initialize
    theta_A = 0.6
    theta_B = 0.5
    n_iterations = 20

    print("EM Algorithm for Two-Coin Problem")
    print(f"Observations (heads/total): {list(zip(heads, n_flips))}")
    print(f"Initial: theta_A={theta_A}, theta_B={theta_B}")
    print()

    print(f"{'Iter':>4} {'theta_A':>10} {'theta_B':>10}")
    print("-" * 28)

    for iteration in range(n_iterations):
        # E-step: compute responsibilities
        # P(coin=A | data_i) proportional to theta_A^heads_i * (1-theta_A)^tails_i
        log_lik_A = heads * np.log(theta_A + 1e-10) + (n_flips - heads) * np.log(1 - theta_A + 1e-10)
        log_lik_B = heads * np.log(theta_B + 1e-10) + (n_flips - heads) * np.log(1 - theta_B + 1e-10)

        # Normalize responsibilities
        log_max = np.maximum(log_lik_A, log_lik_B)
        resp_A = np.exp(log_lik_A - log_max) / (np.exp(log_lik_A - log_max) + np.exp(log_lik_B - log_max))
        resp_B = 1 - resp_A

        # M-step: update parameters
        theta_A_new = np.sum(resp_A * heads) / np.sum(resp_A * n_flips)
        theta_B_new = np.sum(resp_B * heads) / np.sum(resp_B * n_flips)

        theta_A = theta_A_new
        theta_B = theta_B_new

        if iteration < 5 or iteration == n_iterations - 1:
            print(f"{iteration+1:>4} {theta_A:>10.6f} {theta_B:>10.6f}")

    print()
    print(f"Final estimates: theta_A={theta_A:.6f}, theta_B={theta_B:.6f}")
    print(f"Final responsibilities (P(coin=A)):")
    for i, (h, n) in enumerate(zip(heads, n_flips)):
        print(f"  Trial {i+1} ({h}/{n} heads): P(A)={resp_A[i]:.4f}, P(B)={resp_B[i]:.4f}")


# === Exercise 4: Fisher Information ===
# Problem: Fisher information for Bernoulli distribution.

def exercise_4():
    """Fisher information and Cramer-Rao bound for Bernoulli."""
    print("Bernoulli distribution: p(x|theta) = theta^x * (1-theta)^(1-x)")
    print()

    # (a) Fisher information
    print("(a) Fisher information:")
    print("  log p(x|theta) = x*log(theta) + (1-x)*log(1-theta)")
    print("  d/d(theta) log p = x/theta - (1-x)/(1-theta)")
    print("  d^2/d(theta)^2 log p = -x/theta^2 - (1-x)/(1-theta)^2")
    print("  I(theta) = -E[d^2/d(theta)^2 log p]")
    print("           = E[x]/theta^2 + E[1-x]/(1-theta)^2")
    print("           = theta/theta^2 + (1-theta)/(1-theta)^2")
    print("           = 1/theta + 1/(1-theta)")
    print("           = 1/(theta*(1-theta))")
    print()

    # (b) Cramer-Rao lower bound
    print("(b) Cramer-Rao lower bound:")
    print("  Var(theta_hat) >= 1/(n*I(theta)) = theta*(1-theta)/n")
    print()

    # (c) Verify MLE variance reaches the bound
    print("(c) Verify MLE theta_hat = k/n achieves the bound:")
    print("  Var(k/n) = Var(k)/n^2 = n*theta*(1-theta)/n^2 = theta*(1-theta)/n")
    print("  This equals the CRLB => MLE is efficient!")
    print()

    # Numerical verification
    theta_true = 0.3
    n_trials = 100
    n_simulations = 10000

    np.random.seed(42)
    estimates = []
    for _ in range(n_simulations):
        samples = np.random.binomial(1, theta_true, n_trials)
        theta_hat = np.mean(samples)
        estimates.append(theta_hat)

    empirical_var = np.var(estimates)
    theoretical_var = theta_true * (1 - theta_true) / n_trials
    fisher_info = 1.0 / (theta_true * (1 - theta_true))

    print(f"Simulation (theta={theta_true}, n={n_trials}, {n_simulations} simulations):")
    print(f"  Fisher information I(theta): {fisher_info:.4f}")
    print(f"  CRLB = 1/(n*I) = {1/(n_trials*fisher_info):.6f}")
    print(f"  Theoretical Var(k/n) = {theoretical_var:.6f}")
    print(f"  Empirical Var(theta_hat) = {empirical_var:.6f}")
    print(f"  MLE is efficient: {np.isclose(theoretical_var, 1/(n_trials*fisher_info))}")


# === Exercise 5: EM for Mixture of Exponentials ===
# Problem: Implement EM for mixture of two exponential distributions.

def exercise_5():
    """EM algorithm for mixture of exponential distributions."""
    np.random.seed(42)

    # True parameters
    pi_true = 0.4
    lambda1_true = 1.0
    lambda2_true = 5.0

    # Generate data
    n = 500
    z = np.random.binomial(1, pi_true, n)
    data = np.where(z == 1,
                    np.random.exponential(1.0 / lambda1_true, n),
                    np.random.exponential(1.0 / lambda2_true, n))

    print(f"True parameters: pi={pi_true}, lambda1={lambda1_true}, lambda2={lambda2_true}")
    print(f"Data: {n} samples, mean={np.mean(data):.4f}")
    print()

    # Initialize EM
    pi = 0.5
    lam1 = 0.5
    lam2 = 2.0
    n_iterations = 50

    print(f"{'Iter':>4} {'pi':>8} {'lambda1':>10} {'lambda2':>10} {'log-lik':>12}")
    print("-" * 48)

    for iteration in range(n_iterations):
        # E-step: compute responsibilities
        # r_ik = pi_k * p(x_i | lambda_k) / sum_j(pi_j * p(x_i | lambda_j))
        p1 = pi * lam1 * np.exp(-lam1 * data)
        p2 = (1 - pi) * lam2 * np.exp(-lam2 * data)
        total = p1 + p2 + 1e-300

        r1 = p1 / total
        r2 = p2 / total

        # M-step: update parameters
        n1 = np.sum(r1)
        n2 = np.sum(r2)

        pi = n1 / n
        lam1 = n1 / (np.sum(r1 * data) + 1e-10)
        lam2 = n2 / (np.sum(r2 * data) + 1e-10)

        # Log-likelihood
        log_lik = np.sum(np.log(p1 + p2 + 1e-300))

        if iteration < 5 or iteration == n_iterations - 1 or (iteration + 1) % 10 == 0:
            print(f"{iteration+1:>4} {pi:>8.4f} {lam1:>10.4f} {lam2:>10.4f} {log_lik:>12.2f}")

    print()
    print(f"Estimated: pi={pi:.4f}, lambda1={lam1:.4f}, lambda2={lam2:.4f}")
    print(f"True:      pi={pi_true:.4f}, lambda1={lambda1_true:.4f}, lambda2={lambda2_true:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: MLE for Exponential Distribution ===")
    exercise_1()
    print("\n=== Exercise 2: MAP with Different Priors ===")
    exercise_2()
    print("\n=== Exercise 3: EM for Coin Toss ===")
    exercise_3()
    print("\n=== Exercise 4: Fisher Information ===")
    exercise_4()
    print("\n=== Exercise 5: EM for Mixture of Exponentials ===")
    exercise_5()
    print("\nAll exercises completed!")

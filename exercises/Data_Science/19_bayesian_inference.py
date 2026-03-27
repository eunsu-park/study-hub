"""
Exercises for Lesson 19: Bayesian Inference
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Metropolis-Hastings Algorithm ===
# Problem: Implement the Metropolis-Hastings algorithm to sample from a
#   target distribution that is a mixture of two Gaussians. Use a normal
#   proposal distribution and tune the proposal width.
def exercise_1():
    """Solution implementing Metropolis-Hastings MCMC from scratch.

    The Metropolis-Hastings algorithm generates samples from a target
    distribution by proposing random moves and accepting or rejecting
    them based on the ratio of target densities. The proposal width
    controls the trade-off between exploration and acceptance rate.
    An ideal acceptance rate is roughly 20-50%.
    """
    np.random.seed(42)

    def target_unnormalized(x):
        """Unnormalized target: mixture of two Gaussians."""
        comp1 = 0.3 * np.exp(-0.5 * ((x - 2.0) / 0.8)**2)
        comp2 = 0.7 * np.exp(-0.5 * ((x + 1.0) / 1.2)**2)
        return comp1 + comp2

    def metropolis_hastings(target_fn, n_samples, proposal_width, x_init=0.0):
        """Run Metropolis-Hastings with a symmetric Normal proposal."""
        samples = np.zeros(n_samples)
        x_current = x_init
        n_accepted = 0

        for i in range(n_samples):
            # Propose a new state from N(x_current, proposal_width^2)
            x_proposed = x_current + np.random.normal(0, proposal_width)

            # Acceptance ratio (symmetric proposal, so ratio simplifies)
            target_current = target_fn(x_current)
            target_proposed = target_fn(x_proposed)

            if target_current > 0:
                acceptance_ratio = target_proposed / target_current
            else:
                acceptance_ratio = 1.0

            # Accept or reject
            if np.random.random() < min(1.0, acceptance_ratio):
                x_current = x_proposed
                n_accepted += 1

            samples[i] = x_current

        acceptance_rate = n_accepted / n_samples
        return samples, acceptance_rate

    n_samples = 10000
    burn_in = 1000

    # Try different proposal widths
    print("Metropolis-Hastings: Effect of Proposal Width\n")
    print(f"{'Width':>8s}  {'Accept Rate':>12s}  {'Mean':>8s}  {'Std':>8s}")
    print("-" * 42)

    for width in [0.1, 0.5, 1.0, 2.0, 5.0]:
        samples, acc_rate = metropolis_hastings(
            target_unnormalized, n_samples, width
        )
        post_burn = samples[burn_in:]
        print(f"{width:8.1f}  {acc_rate:12.3f}  "
              f"{post_burn.mean():8.3f}  {post_burn.std():8.3f}")

    # Detailed run with optimal width
    print("\n--- Detailed run with proposal width = 1.0 ---")
    samples, acc_rate = metropolis_hastings(
        target_unnormalized, n_samples, proposal_width=1.0
    )
    post_burn = samples[burn_in:]

    print(f"Total samples:     {n_samples}")
    print(f"Burn-in discarded: {burn_in}")
    print(f"Effective samples: {len(post_burn)}")
    print(f"Acceptance rate:   {acc_rate:.3f}")
    print(f"Posterior mean:    {post_burn.mean():.4f}")
    print(f"Posterior std:     {post_burn.std():.4f}")
    print(f"Posterior median:  {np.median(post_burn):.4f}")

    # Check convergence: split chain in half and compare
    first_half = post_burn[:len(post_burn)//2]
    second_half = post_burn[len(post_burn)//2:]
    print(f"\nConvergence check (split-chain means):")
    print(f"  First half mean:  {first_half.mean():.4f}")
    print(f"  Second half mean: {second_half.mean():.4f}")
    print(f"  Difference:       {abs(first_half.mean() - second_half.mean()):.4f}")


# === Exercise 2: Bayesian Linear Regression ===
# Problem: Given data y = 2x + 1 + noise, compute the Bayesian posterior
#   for the slope and intercept using conjugate Normal prior, and compare
#   with ordinary least squares estimates.
def exercise_2():
    """Solution for Bayesian linear regression with conjugate priors.

    For y = X*beta + epsilon with epsilon ~ N(0, sigma^2) and
    prior beta ~ N(mu_0, Sigma_0), the posterior is:
        Sigma_post = inv(inv(Sigma_0) + X^T X / sigma^2)
        mu_post = Sigma_post @ (inv(Sigma_0) @ mu_0 + X^T y / sigma^2)

    We compare the Bayesian posterior mean with the OLS estimate.
    """
    np.random.seed(42)

    # Generate data: y = 2x + 1 + noise
    n = 30
    true_intercept = 1.0
    true_slope = 2.0
    sigma_noise = 1.5

    x = np.random.uniform(0, 5, n)
    y = true_intercept + true_slope * x + np.random.normal(0, sigma_noise, n)

    # Design matrix [1, x]
    X = np.column_stack([np.ones(n), x])

    # --- OLS estimate ---
    # beta_ols = (X^T X)^{-1} X^T y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_ols = XtX_inv @ (X.T @ y)

    print("Ordinary Least Squares:")
    print(f"  Intercept: {beta_ols[0]:.4f} (true: {true_intercept})")
    print(f"  Slope:     {beta_ols[1]:.4f} (true: {true_slope})")

    # Residual standard deviation
    residuals = y - X @ beta_ols
    sigma_hat = np.sqrt(np.sum(residuals**2) / (n - 2))
    print(f"  Residual std: {sigma_hat:.4f} (true: {sigma_noise})")

    # --- Bayesian estimate ---
    # Prior: beta ~ N(mu_0, Sigma_0)
    mu_0 = np.array([0.0, 0.0])       # vague prior centered at zero
    Sigma_0 = np.diag([100.0, 100.0])  # wide prior (low precision)
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    # Assume sigma is known (= sigma_noise) for conjugacy
    sigma2 = sigma_noise**2

    # Posterior
    Sigma_post_inv = Sigma_0_inv + XtX / sigma2
    Sigma_post = np.linalg.inv(Sigma_post_inv)
    mu_post = Sigma_post @ (Sigma_0_inv @ mu_0 + X.T @ y / sigma2)

    post_std = np.sqrt(np.diag(Sigma_post))

    print(f"\nBayesian Posterior (known sigma={sigma_noise}):")
    print(f"  Intercept: {mu_post[0]:.4f} +/- {post_std[0]:.4f}")
    print(f"  Slope:     {mu_post[1]:.4f} +/- {post_std[1]:.4f}")

    # 95% credible intervals
    for i, name in enumerate(["Intercept", "Slope"]):
        ci_lo = mu_post[i] - 1.96 * post_std[i]
        ci_hi = mu_post[i] + 1.96 * post_std[i]
        print(f"  {name} 95% CI: ({ci_lo:.4f}, {ci_hi:.4f})")

    # Comparison
    print(f"\nComparison:")
    print(f"  {'':12s} {'OLS':>10s} {'Bayesian':>10s} {'True':>10s}")
    print(f"  {'Intercept':12s} {beta_ols[0]:10.4f} {mu_post[0]:10.4f} {true_intercept:10.4f}")
    print(f"  {'Slope':12s} {beta_ols[1]:10.4f} {mu_post[1]:10.4f} {true_slope:10.4f}")
    print()
    print("  With a wide prior (Sigma_0 = 100*I), the Bayesian posterior")
    print("  is very close to OLS. The prior has minimal influence.")


# === Exercise 3: Bayesian Model Comparison ===
# Problem: Compare two models for count data:
#   Model 1: Poisson(lambda=3)
#   Model 2: Poisson(lambda=5)
#   Given observed data, compute Bayes factors and posterior model probabilities.
def exercise_3():
    """Solution for Bayesian model comparison using Bayes factors.

    The Bayes factor BF_{12} = P(data | M1) / P(data | M2).
    - BF > 1 favors Model 1
    - BF < 1 favors Model 2

    With equal prior model probabilities:
    P(M1 | data) = BF / (BF + 1)

    We compute the marginal likelihood for each Poisson model by
    evaluating the product of Poisson PMFs.
    """
    np.random.seed(42)

    # Observed data (count data)
    data = np.array([2, 4, 3, 1, 5, 3, 2, 4, 3, 2])
    n = len(data)

    print(f"Observed data: {data.tolist()}")
    print(f"Sample mean: {data.mean():.2f}")
    print(f"Sample var:  {data.var(ddof=1):.2f}")

    def poisson_log_likelihood(data, lam):
        """Compute log P(data | Poisson(lambda))."""
        # log P(x | lambda) = x*log(lambda) - lambda - log(x!)
        from math import lgamma
        log_lik = 0.0
        for x in data:
            log_lik += x * np.log(lam) - lam - lgamma(x + 1)
        return log_lik

    # Model 1: lambda = 3
    lam1 = 3.0
    log_lik1 = poisson_log_likelihood(data, lam1)

    # Model 2: lambda = 5
    lam2 = 5.0
    log_lik2 = poisson_log_likelihood(data, lam2)

    # Log Bayes factor
    log_bf_12 = log_lik1 - log_lik2
    bf_12 = np.exp(log_bf_12)

    print(f"\nModel 1: Poisson(lambda={lam1})")
    print(f"  Log-likelihood: {log_lik1:.4f}")
    print(f"\nModel 2: Poisson(lambda={lam2})")
    print(f"  Log-likelihood: {log_lik2:.4f}")

    print(f"\nBayes Factor BF_12: {bf_12:.4f}")
    print(f"Log Bayes Factor:   {log_bf_12:.4f}")

    # Interpretation (Kass & Raftery 1995 scale)
    log10_bf = log_bf_12 / np.log(10)
    if abs(log10_bf) < 0.5:
        strength = "Not worth more than a bare mention"
    elif abs(log10_bf) < 1.0:
        strength = "Substantial"
    elif abs(log10_bf) < 2.0:
        strength = "Strong"
    else:
        strength = "Decisive"
    favored = "Model 1" if bf_12 > 1 else "Model 2"
    print(f"  2*log10(BF): {2*log10_bf:.2f}")
    print(f"  Evidence: {strength} in favor of {favored}")

    # Posterior model probabilities (equal priors)
    prior_m1 = 0.5
    prior_m2 = 0.5
    post_m1 = (bf_12 * prior_m1) / (bf_12 * prior_m1 + prior_m2)
    post_m2 = 1 - post_m1

    print(f"\nPosterior model probabilities (equal priors):")
    print(f"  P(Model 1 | data) = {post_m1:.4f}")
    print(f"  P(Model 2 | data) = {post_m2:.4f}")

    # Compare with more lambda values
    print("\n--- Model comparison across lambda values ---")
    print(f"{'Lambda':>8s}  {'Log-lik':>10s}  {'BF vs best':>12s}")
    print("-" * 34)

    lambdas = [1.0, 2.0, 2.5, 2.9, 3.0, 3.5, 4.0, 5.0]
    log_liks = [poisson_log_likelihood(data, l) for l in lambdas]
    best_ll = max(log_liks)

    for lam, ll in zip(lambdas, log_liks):
        bf_vs_best = np.exp(ll - best_ll)
        print(f"{lam:8.1f}  {ll:10.4f}  {bf_vs_best:12.4f}")

    best_idx = np.argmax(log_liks)
    print(f"\nBest model: Poisson(lambda={lambdas[best_idx]})")
    print(f"Note: sample mean = {data.mean():.2f}, which is the MLE for lambda.")


# === Exercise 4: Posterior Predictive Distribution ===
# Problem: After observing coin flip data (Beta-Binomial model), compute
#   the posterior predictive distribution for the next m flips and
#   assess model fit via a posterior predictive check.
def exercise_4():
    """Solution for posterior predictive distribution and checking.

    The posterior predictive distribution integrates over parameter
    uncertainty:  P(y_new | data) = integral P(y_new | theta) P(theta | data) d_theta

    For Beta-Binomial, the predictive distribution for the number of
    heads in m new flips is Beta-Binomial(m, alpha_post, beta_post).
    We approximate it by sampling.
    """
    np.random.seed(42)

    # Observed data: 7 heads in 10 flips
    n_obs = 10
    k_obs = 7

    # Prior: Beta(1, 1) = Uniform
    alpha_prior = 1
    beta_prior = 1

    # Posterior: Beta(alpha_post, beta_post)
    alpha_post = alpha_prior + k_obs
    beta_post = beta_prior + (n_obs - k_obs)
    post_mean = alpha_post / (alpha_post + beta_post)

    print(f"Observed: {k_obs} heads in {n_obs} flips")
    print(f"Prior: Beta({alpha_prior}, {beta_prior})")
    print(f"Posterior: Beta({alpha_post}, {beta_post})")
    print(f"Posterior mean: {post_mean:.4f}")

    # Posterior predictive: predict number of heads in next m flips
    m = 10  # next m flips
    n_mc = 50000

    # Step 1: Sample theta from posterior Beta(alpha_post, beta_post)
    # Using numpy's beta distribution
    theta_samples = np.random.beta(alpha_post, beta_post, size=n_mc)

    # Step 2: For each theta, sample number of heads in m flips
    y_pred = np.array([np.random.binomial(m, theta) for theta in theta_samples])

    print(f"\nPosterior Predictive: heads in next {m} flips")
    print(f"  (Monte Carlo with {n_mc} samples)")
    print(f"\n  {'k heads':>8s}  {'P(k)':>8s}  {'Histogram':>20s}")
    print("  " + "-" * 40)

    for k in range(m + 1):
        prob = np.mean(y_pred == k)
        bar = "#" * int(prob * 100)
        print(f"  {k:8d}  {prob:8.4f}  {bar}")

    print(f"\n  Predictive mean:   {y_pred.mean():.2f}")
    print(f"  Predictive std:    {y_pred.std():.2f}")
    print(f"  Predictive median: {np.median(y_pred):.0f}")

    # Posterior predictive check: is the observed statistic consistent?
    # Use the proportion of heads as the test statistic
    obs_prop = k_obs / n_obs
    pred_props = y_pred / m
    p_value_ppc = np.mean(pred_props >= obs_prop)

    print(f"\nPosterior Predictive Check:")
    print(f"  Observed proportion: {obs_prop:.2f}")
    print(f"  P(pred_prop >= obs_prop): {p_value_ppc:.4f}")
    print(f"  Model is {'consistent' if 0.05 < p_value_ppc < 0.95 else 'suspect'} "
          f"with observed data.")


if __name__ == "__main__":
    print("=== Exercise 1: Metropolis-Hastings Algorithm ===")
    exercise_1()
    print("\n=== Exercise 2: Bayesian Linear Regression ===")
    exercise_2()
    print("\n=== Exercise 3: Bayesian Model Comparison ===")
    exercise_3()
    print("\n=== Exercise 4: Posterior Predictive Distribution ===")
    exercise_4()
    print("\nAll exercises completed!")

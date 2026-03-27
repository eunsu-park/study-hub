"""
Exercises for Lesson 26: Bayesian Advanced Methods
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Metropolis-Hastings Sampler ===
# Problem: Implement a Metropolis-Hastings sampler for a normal model
#          and demonstrate its limitations in higher dimensions.
def exercise_1():
    """Solution for Metropolis-Hastings MCMC sampler.

    Metropolis-Hastings algorithm:
    1. Initialize theta_0
    2. Propose theta* ~ q(theta*|theta_t)
    3. Accept with probability min(1, (p(theta*|data) * q(theta_t|theta*))
                                    / (p(theta_t|data) * q(theta*|theta_t)))
    4. For symmetric proposal (normal), simplifies to min(1, p(theta*|data)/p(theta_t|data))

    This exercise shows why MH struggles in high dimensions
    and motivates HMC/NUTS.
    """
    np.random.seed(42)

    # Observed data
    true_mu = 5.0
    true_sigma = 2.0
    n_obs = 50
    data = np.random.normal(true_mu, true_sigma, n_obs)

    def log_posterior(mu, sigma, obs):
        """Unnormalized log posterior for normal model with flat priors."""
        if sigma <= 0:
            return -np.inf
        n = len(obs)
        # Log-likelihood
        ll = -n * np.log(sigma) - 0.5 * np.sum((obs - mu) ** 2) / sigma ** 2
        # Weak prior: mu ~ Normal(0, 100), sigma ~ HalfNormal(0, 10)
        lp_mu = -0.5 * (mu / 100) ** 2
        lp_sigma = -0.5 * (sigma / 10) ** 2
        return ll + lp_mu + lp_sigma

    def metropolis_hastings(data, n_samples=5000, step_size=0.5):
        """Run Metropolis-Hastings for normal model."""
        samples = np.zeros((n_samples, 2))  # [mu, sigma]
        current = np.array([0.0, 1.0])      # initial values
        current_lp = log_posterior(current[0], current[1], data)
        accepted = 0

        for i in range(n_samples):
            # Symmetric normal proposal
            proposal = current + np.random.normal(0, step_size, 2)
            proposal_lp = log_posterior(proposal[0], proposal[1], data)

            # Acceptance ratio (log scale)
            log_alpha = proposal_lp - current_lp
            if np.log(np.random.uniform()) < log_alpha:
                current = proposal
                current_lp = proposal_lp
                accepted += 1

            samples[i] = current

        return samples, accepted / n_samples

    # Run with different step sizes
    print("Metropolis-Hastings Sampler for Normal Model:")
    print(f"  True parameters: mu={true_mu}, sigma={true_sigma}")
    print(f"  Data: n={n_obs}, sample mean={np.mean(data):.3f}, "
          f"sample sd={np.std(data, ddof=1):.3f}")

    for step in [0.1, 0.5, 2.0, 5.0]:
        samples, acc_rate = metropolis_hastings(data, n_samples=10000,
                                                step_size=step)
        burn_in = 2000
        post_burn = samples[burn_in:]

        mu_est = np.mean(post_burn[:, 0])
        sigma_est = np.mean(post_burn[:, 1])

        # Effective sample size (simple autocorrelation-based estimate)
        autocorr = np.corrcoef(post_burn[:-1, 0], post_burn[1:, 0])[0, 1]
        ess_approx = len(post_burn) * (1 - autocorr) / (1 + autocorr)

        print(f"\n  Step size = {step}:")
        print(f"    Acceptance rate: {acc_rate:.3f} (ideal: 0.20-0.50)")
        print(f"    mu estimate: {mu_est:.3f} (true: {true_mu})")
        print(f"    sigma estimate: {sigma_est:.3f} (true: {true_sigma})")
        print(f"    Lag-1 autocorrelation: {autocorr:.3f}")
        print(f"    Approximate ESS: {ess_approx:.0f} / {len(post_burn)}")

    print("\n  Conclusion: MH acceptance rate and ESS degrade with dimension.")
    print("  HMC/NUTS use gradients to maintain efficiency in high dimensions.")


# === Exercise 2: Hierarchical Model (Partial Pooling) ===
# Problem: Compare no-pooling vs complete pooling vs partial pooling
#          estimates for group means.
def exercise_2():
    """Solution for hierarchical model comparison.

    Three estimation strategies:
    1. No pooling: Separate estimate per group (high variance for small groups)
    2. Complete pooling: Single estimate for all groups (ignores group structure)
    3. Partial pooling: Hierarchical model that shrinks small groups toward
       the population mean (best bias-variance trade-off)

    Shrinkage: Small groups get pulled more toward the grand mean.
    """
    np.random.seed(42)

    # Simulate school test scores
    n_schools = 8
    true_pop_mean = 70.0
    true_pop_sd = 8.0
    true_school_means = np.random.normal(true_pop_mean, true_pop_sd, n_schools)
    within_sd = 12.0

    # Varying sample sizes (some schools have few students)
    n_students = np.array([5, 10, 15, 50, 8, 100, 12, 200])

    # Generate data
    all_scores = []
    school_ids = []
    for j in range(n_schools):
        scores = np.random.normal(true_school_means[j], within_sd, n_students[j])
        all_scores.append(scores)
        school_ids.extend([j] * n_students[j])

    scores_flat = np.concatenate(all_scores)

    print("Hierarchical Model: No Pooling vs Complete Pooling vs Partial Pooling")
    print(f"  True population mean: {true_pop_mean}")
    print(f"  True between-school SD: {true_pop_sd}")
    print(f"  Within-school SD: {within_sd}")

    # 1. Complete pooling: grand mean for all schools
    grand_mean = np.mean(scores_flat)

    # 2. No pooling: school-specific means
    school_means = np.array([np.mean(s) for s in all_scores])

    # 3. Partial pooling (empirical Bayes / James-Stein shrinkage)
    # Shrinkage factor for school j:
    #   B_j = within_var/n_j / (within_var/n_j + between_var)
    # Partial pooling estimate:
    #   theta_j = (1 - B_j) * school_mean_j + B_j * grand_mean

    # Estimate between-school variance (method of moments)
    between_var_raw = np.var(school_means, ddof=1)
    avg_within_var = within_sd ** 2
    mean_n = np.mean(n_students)
    between_var_est = max(0, between_var_raw - avg_within_var / mean_n)

    partial_means = np.zeros(n_schools)
    shrinkage = np.zeros(n_schools)
    for j in range(n_schools):
        within_var_j = avg_within_var / n_students[j]
        b_j = within_var_j / (within_var_j + between_var_est)
        shrinkage[j] = b_j
        partial_means[j] = (1 - b_j) * school_means[j] + b_j * grand_mean

    # Comparison table
    print(f"\n  {'School':>8} {'N':>5} {'True':>8} {'No Pool':>9} "
          f"{'Full Pool':>10} {'Partial':>9} {'Shrink':>8}")
    print(f"  {'-' * 62}")
    for j in range(n_schools):
        print(f"  {j:>8} {n_students[j]:>5} {true_school_means[j]:>8.2f} "
              f"{school_means[j]:>9.2f} {grand_mean:>10.2f} "
              f"{partial_means[j]:>9.2f} {shrinkage[j]:>8.3f}")

    # MSE comparison
    mse_no_pool = np.mean((school_means - true_school_means) ** 2)
    mse_full_pool = np.mean((grand_mean - true_school_means) ** 2)
    mse_partial = np.mean((partial_means - true_school_means) ** 2)

    print(f"\n  Mean Squared Error vs True Means:")
    print(f"    No pooling:       {mse_no_pool:.4f}")
    print(f"    Complete pooling: {mse_full_pool:.4f}")
    print(f"    Partial pooling:  {mse_partial:.4f}")

    print(f"\n  Small schools shrink more; partial pooling typically has lowest MSE.")


# === Exercise 3: Bayesian Model Comparison ===
# Problem: Compare linear vs quadratic models using information
#          criteria (AIC/BIC as stand-ins for WAIC/LOO).
def exercise_3():
    """Solution for Bayesian-style model comparison.

    In full Bayesian analysis, WAIC and LOO-CV are preferred.
    Here we demonstrate the concept using AIC/BIC:

    AIC = -2*logL + 2*k
    BIC = -2*logL + k*log(n)

    Lower values indicate better trade-off between fit and complexity.
    BIC penalizes complexity more strongly for large n.
    """
    np.random.seed(42)

    # Generate data with a quadratic relationship
    n = 100
    x = np.random.uniform(-3, 3, n)
    y_true = 2 + 0.5 * x - 0.3 * x ** 2
    y = y_true + np.random.normal(0, 1, n)

    def fit_polynomial(x_data, y_data, degree):
        """Fit polynomial regression and compute AIC/BIC."""
        X = np.column_stack([x_data ** d for d in range(degree + 1)])

        # OLS: beta = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y_data
        beta = np.linalg.solve(XtX, Xty)

        y_hat = X @ beta
        n_obs = len(y_data)
        k = degree + 1
        rss = np.sum((y_data - y_hat) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - rss / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1)
        log_lik = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(rss / n_obs) + 1)
        aic = -2 * log_lik + 2 * k
        bic = -2 * log_lik + k * np.log(n_obs)
        return {
            'degree': degree, 'k': k, 'r_squared': r_squared,
            'adj_r_squared': adj_r_squared, 'aic': aic, 'bic': bic,
        }

    print("Bayesian Model Comparison (AIC/BIC):")
    print(f"  True model: y = 2 + 0.5*x - 0.3*x^2 + noise")
    print(f"  Data: n={n}")

    results = []
    for degree in range(1, 6):
        res = fit_polynomial(x, y, degree)
        results.append(res)

    # Display comparison
    print(f"\n  {'Model':<15} {'k':>4} {'R2':>8} {'Adj R2':>8} "
          f"{'AIC':>10} {'BIC':>10}")
    print(f"  {'-' * 58}")

    best_aic_model = min(results, key=lambda r: r['aic'])
    best_bic_model = min(results, key=lambda r: r['bic'])

    for res in results:
        model_name = f"Degree {res['degree']}"
        aic_flag = " *" if res['aic'] == best_aic_model['aic'] else ""
        bic_flag = " *" if res['bic'] == best_bic_model['bic'] else ""
        print(f"  {model_name:<15} {res['k']:>4} {res['r_squared']:>8.4f} "
              f"{res['adj_r_squared']:>8.4f} "
              f"{res['aic']:>10.2f}{aic_flag} {res['bic']:>10.2f}{bic_flag}")

    print(f"\n  Best by AIC: Degree {best_aic_model['degree']} "
          f"(AIC={best_aic_model['aic']:.2f})")
    print(f"  Best by BIC: Degree {best_bic_model['degree']} "
          f"(BIC={best_bic_model['bic']:.2f})")

    print(f"\n  True coefficients: intercept=2.0, x^1=0.5, x^2=-0.3")
    print(f"  Note: In full Bayesian analysis, use LOO-CV (via PSIS) or WAIC")
    print(f"  instead of AIC/BIC. Check Pareto k diagnostics for reliability.")


# === Exercise 4: Convergence Diagnostics ===
# Problem: Implement R-hat and effective sample size diagnostics
#          for MCMC chains.
def exercise_4():
    """Solution for MCMC convergence diagnostics.

    R-hat (Gelman-Rubin):
    Compares within-chain and between-chain variance.
    R-hat = sqrt(var_hat / W), where var_hat combines B and W.
    R-hat < 1.01 indicates convergence.

    Effective Sample Size (ESS):
    Accounts for autocorrelation in the chain.
    ESS = n * m / (1 + 2 * sum(autocorrelations))
    ESS > 400 is recommended for reliable inference.
    """
    np.random.seed(42)

    def simulate_chains(mu_true, sigma_true, n_chains=4, n_samples=2000,
                        step_size=1.0, converged=True):
        """Simulate MCMC chains (MH for a normal model)."""
        chains = np.zeros((n_chains, n_samples))
        data = np.random.normal(mu_true, sigma_true, 50)

        for c in range(n_chains):
            if converged:
                current = np.random.normal(mu_true, 2)
            else:
                current = np.random.normal(mu_true + c * 10, 1)

            for i in range(n_samples):
                proposal = current + np.random.normal(0, step_size)
                # Simplified log-posterior for mu (sigma known)
                lp_curr = -0.5 * np.sum((data - current) ** 2) / sigma_true ** 2
                lp_prop = -0.5 * np.sum((data - proposal) ** 2) / sigma_true ** 2
                if np.log(np.random.uniform()) < (lp_prop - lp_curr):
                    current = proposal
                chains[c, i] = current

        return chains

    def compute_rhat(chains):
        """Compute split R-hat statistic."""
        n_chains, n_samples = chains.shape
        # Split each chain in half
        half = n_samples // 2
        split_chains = []
        for c in range(n_chains):
            split_chains.append(chains[c, :half])
            split_chains.append(chains[c, half:])

        m = len(split_chains)
        n = len(split_chains[0])

        chain_means = np.array([np.mean(c) for c in split_chains])
        chain_vars = np.array([np.var(c, ddof=1) for c in split_chains])

        grand_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)  # between-chain variance
        W = np.mean(chain_vars)               # within-chain variance

        var_hat = (1 - 1 / n) * W + B / n
        rhat = np.sqrt(var_hat / W) if W > 0 else float('inf')

        return rhat

    def compute_ess(chain):
        """Compute effective sample size for a single chain."""
        n = len(chain)
        mean_val = np.mean(chain)
        var_val = np.var(chain, ddof=1)
        if var_val == 0:
            return n

        # Compute autocorrelations
        max_lag = min(n // 2, 100)
        autocorrs = np.zeros(max_lag)
        for lag in range(max_lag):
            c = np.mean((chain[:n - lag] - mean_val) * (chain[lag:] - mean_val))
            autocorrs[lag] = c / var_val

        # Sum autocorrelations (stop when sum of consecutive pairs goes negative)
        sum_corr = 0.0
        for lag in range(1, max_lag - 1, 2):
            pair_sum = autocorrs[lag] + autocorrs[lag + 1]
            if pair_sum < 0:
                break
            sum_corr += pair_sum

        ess = n / (1 + 2 * sum_corr)
        return max(1, ess)

    # Case 1: Converged chains
    print("MCMC Convergence Diagnostics:")
    print("\n  Case 1: Well-converged chains")
    chains_good = simulate_chains(5.0, 2.0, n_chains=4, n_samples=2000,
                                  converged=True)
    rhat_good = compute_rhat(chains_good)
    ess_good = np.mean([compute_ess(chains_good[c]) for c in range(4)])
    print(f"    R-hat: {rhat_good:.4f} ({'OK' if rhat_good < 1.01 else 'NOT converged'})")
    print(f"    Mean ESS: {ess_good:.0f} ({'OK' if ess_good > 400 else 'LOW'})")
    print(f"    Chain means: {[f'{np.mean(chains_good[c]):.3f}' for c in range(4)]}")

    # Case 2: Non-converged chains (different starting points, no mixing)
    print("\n  Case 2: Non-converged chains (dispersed starts)")
    chains_bad = simulate_chains(5.0, 2.0, n_chains=4, n_samples=500,
                                 step_size=0.1, converged=False)
    rhat_bad = compute_rhat(chains_bad)
    ess_bad = np.mean([compute_ess(chains_bad[c]) for c in range(4)])
    print(f"    R-hat: {rhat_bad:.4f} ({'OK' if rhat_bad < 1.01 else 'NOT converged'})")
    print(f"    Mean ESS: {ess_bad:.0f} ({'OK' if ess_bad > 400 else 'LOW'})")
    print(f"    Chain means: {[f'{np.mean(chains_bad[c]):.3f}' for c in range(4)]}")

    print(f"\n  Checklist: R-hat < 1.01, ESS > 400, no divergences, "
          f"good trace mixing.")


if __name__ == "__main__":
    print("=== Exercise 1: Metropolis-Hastings Sampler ===")
    exercise_1()
    print("\n=== Exercise 2: Hierarchical Model (Partial Pooling) ===")
    exercise_2()
    print("\n=== Exercise 3: Bayesian Model Comparison ===")
    exercise_3()
    print("\n=== Exercise 4: Convergence Diagnostics ===")
    exercise_4()
    print("\nAll exercises completed!")

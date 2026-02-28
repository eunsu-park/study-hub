"""
Exercises for Lesson 12: Sampling and Monte Carlo
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import integrate, stats


# === Exercise 1: Monte Carlo Integration ===
# Problem: Estimate integral_0^1 exp(-x^2) dx using Monte Carlo.

def exercise_1():
    """Monte Carlo integration of exp(-x^2) from 0 to 1."""
    # Exact value using scipy
    exact, _ = integrate.quad(lambda x: np.exp(-x**2), 0, 1)
    print(f"Exact integral (scipy.quad): {exact:.10f}")
    print()

    np.random.seed(42)
    sample_sizes = [10, 100, 1000, 10000]
    errors_list = []

    print(f"{'N':>8} {'Estimate':>12} {'Error':>12} {'Theoretical O(1/sqrt(N))':>25}")
    print("-" * 60)

    for n in sample_sizes:
        # Monte Carlo: E[f(X)] where X ~ Uniform(0,1)
        # Integral = (b-a) * mean(f(X_i))
        samples = np.random.uniform(0, 1, n)
        values = np.exp(-samples**2)
        estimate = np.mean(values)
        error = abs(estimate - exact)
        errors_list.append(error)

        # Theoretical error bound: sigma / sqrt(N)
        sigma = np.std(values)
        theoretical = sigma / np.sqrt(n)

        print(f"{n:>8} {estimate:>12.8f} {error:>12.8f} {theoretical:>25.8f}")

    # (b) Verify O(1/sqrt(N)) convergence
    print("\nConvergence rate verification (log-log slope should be ~ -0.5):")
    for i in range(1, len(sample_sizes)):
        log_n_ratio = np.log(sample_sizes[i] / sample_sizes[i-1])
        # Average over many trials for stable error
        errors_i_prev = []
        errors_i = []
        for _ in range(200):
            s1 = np.random.uniform(0, 1, sample_sizes[i-1])
            s2 = np.random.uniform(0, 1, sample_sizes[i])
            errors_i_prev.append(abs(np.mean(np.exp(-s1**2)) - exact))
            errors_i.append(abs(np.mean(np.exp(-s2**2)) - exact))

        avg_e_prev = np.mean(errors_i_prev)
        avg_e = np.mean(errors_i)
        if avg_e > 0 and avg_e_prev > 0:
            log_err_ratio = np.log(avg_e / avg_e_prev)
            slope = log_err_ratio / log_n_ratio
            print(f"  N={sample_sizes[i-1]}->{sample_sizes[i]}: slope = {slope:.3f} (expected: -0.5)")


# === Exercise 2: Importance Sampling ===
# Problem: Compute E[x^2] where target p(x) proportional to exp(-|x|^3).

def exercise_2():
    """Importance sampling with different proposal distributions."""
    np.random.seed(42)
    n_samples = 50000

    # Target: p(x) proportional to exp(-|x|^3)
    # Normalization constant via numerical integration
    Z, _ = integrate.quad(lambda x: np.exp(-abs(x)**3), -10, 10)

    def target_unnorm(x):
        return np.exp(-np.abs(x)**3)

    # True E[x^2] via numerical integration
    true_val, _ = integrate.quad(lambda x: x**2 * np.exp(-abs(x)**3) / Z, -10, 10)
    print(f"True E[x^2] = {true_val:.6f}")
    print()

    # (a) Proposal: q(x) = N(0, 1)
    samples_normal = np.random.randn(n_samples)
    w_normal = target_unnorm(samples_normal) / stats.norm.pdf(samples_normal)
    w_normal_normalized = w_normal / np.sum(w_normal)
    estimate_normal = np.sum(w_normal_normalized * samples_normal**2)
    ess_normal = 1.0 / np.sum(w_normal_normalized**2)

    print(f"(a) Proposal: N(0, 1)")
    print(f"  Estimate: {estimate_normal:.6f}")
    print(f"  Error: {abs(estimate_normal - true_val):.6f}")
    print(f"  ESS: {ess_normal:.0f} / {n_samples}")
    print()

    # (b) Proposal: Laplace(0, 1)
    samples_laplace = np.random.laplace(0, 1, n_samples)
    w_laplace = target_unnorm(samples_laplace) / stats.laplace.pdf(samples_laplace)
    w_laplace_normalized = w_laplace / np.sum(w_laplace)
    estimate_laplace = np.sum(w_laplace_normalized * samples_laplace**2)
    ess_laplace = 1.0 / np.sum(w_laplace_normalized**2)

    print(f"(b) Proposal: Laplace(0, 1)")
    print(f"  Estimate: {estimate_laplace:.6f}")
    print(f"  Error: {abs(estimate_laplace - true_val):.6f}")
    print(f"  ESS: {ess_laplace:.0f} / {n_samples}")
    print()

    # (c) Analysis
    print("(c) Which is better?")
    if ess_laplace > ess_normal:
        print(f"  Laplace has higher ESS ({ess_laplace:.0f} vs {ess_normal:.0f})")
        print("  Laplace tails decay as exp(-|x|), closer to target exp(-|x|^3)")
    else:
        print(f"  Normal has higher ESS ({ess_normal:.0f} vs {ess_laplace:.0f})")
    print("  The proposal closer to target shape gives lower variance weights")


# === Exercise 3: MCMC Diagnostics ===
# Problem: Metropolis-Hastings on bimodal distribution.

def exercise_3():
    """Metropolis-Hastings with diagnostics for bimodal distribution."""
    np.random.seed(42)

    # Target: p(x) proportional to exp(-(x-3)^2/2) + exp(-(x+3)^2/2)
    def log_target(x):
        return np.logaddexp(-0.5*(x-3)**2, -0.5*(x+3)**2)

    def metropolis_hastings(n_samples, proposal_std, x0=0.0):
        x = x0
        samples = [x]
        n_accepted = 0

        for _ in range(n_samples - 1):
            x_proposal = x + np.random.randn() * proposal_std
            log_alpha = log_target(x_proposal) - log_target(x)

            if np.log(np.random.rand()) < log_alpha:
                x = x_proposal
                n_accepted += 1

            samples.append(x)

        return np.array(samples), n_accepted / (n_samples - 1)

    n_samples = 10000
    proposal_stds = [0.1, 1.0, 10.0]

    print("Metropolis-Hastings on bimodal: p(x) ~ exp(-(x-3)^2/2) + exp(-(x+3)^2/2)")
    print()

    for sigma in proposal_stds:
        samples, accept_rate = metropolis_hastings(n_samples, sigma)

        # Autocorrelation
        def autocorrelation(x, max_lag=50):
            n = len(x)
            x_centered = x - np.mean(x)
            var = np.var(x)
            acf = []
            for lag in range(max_lag + 1):
                if var > 0:
                    acf.append(np.mean(x_centered[:n-lag] * x_centered[lag:]) / var)
                else:
                    acf.append(0)
            return np.array(acf)

        acf = autocorrelation(samples, max_lag=50)

        # Effective sample size
        # ESS = N / (1 + 2 * sum(autocorrelations))
        ess = n_samples / (1 + 2 * np.sum(np.abs(acf[1:])))

        # Check if both modes are visited
        n_positive = np.sum(samples > 0)
        n_negative = np.sum(samples <= 0)

        print(f"Proposal std = {sigma}:")
        print(f"  Acceptance rate: {accept_rate:.4f}")
        print(f"  ESS: {ess:.0f} / {n_samples}")
        print(f"  ACF at lag 1: {acf[1]:.4f}")
        print(f"  ACF at lag 10: {acf[10]:.4f}")
        print(f"  Samples near mode +3: {n_positive}, near mode -3: {n_negative}")
        print(f"  Both modes visited: {n_positive > 100 and n_negative > 100}")
        print()

    print("Optimal proposal std: ~1.0 (acceptance ~23-44%, good ESS, visits both modes)")


# === Exercise 4: Reparameterization Trick ===
# Problem: Demonstrate reparameterization trick for gradient estimation.

def exercise_4():
    """Reparameterization trick for gradient estimation."""
    np.random.seed(42)

    # Simple example: minimize E_{z~N(mu,sigma^2)}[z^2]
    # Analytical: E[z^2] = mu^2 + sigma^2
    # d/d_mu E[z^2] = 2*mu
    # d/d_sigma E[z^2] = 2*sigma (via reparameterization)

    mu = 2.0
    sigma = 1.5
    n_samples = 10000

    # Method 1: Reparameterization trick
    # z = mu + sigma * eps, eps ~ N(0,1)
    eps = np.random.randn(n_samples)
    z_reparam = mu + sigma * eps

    # d/d_mu f(z) = d/d_mu f(mu + sigma*eps) = f'(z) * 1
    # d/d_sigma f(z) = d/d_sigma f(mu + sigma*eps) = f'(z) * eps
    f_z = z_reparam**2
    df_dz = 2 * z_reparam

    grad_mu_reparam = np.mean(df_dz * 1)
    grad_sigma_reparam = np.mean(df_dz * eps)

    # Method 2: REINFORCE (score function estimator)
    # grad E[f(z)] = E[f(z) * grad log p(z)]
    # For N(mu, sigma^2): grad_mu log p(z) = (z - mu)/sigma^2
    #                      grad_sigma log p(z) = (z-mu)^2/sigma^3 - 1/sigma
    z_reinforce = np.random.normal(mu, sigma, n_samples)
    f_reinforce = z_reinforce**2

    score_mu = (z_reinforce - mu) / sigma**2
    score_sigma = ((z_reinforce - mu)**2 / sigma**3) - 1.0 / sigma

    grad_mu_reinforce = np.mean(f_reinforce * score_mu)
    grad_sigma_reinforce = np.mean(f_reinforce * score_sigma)

    # Analytical gradients
    grad_mu_analytical = 2 * mu
    grad_sigma_analytical = 2 * sigma

    print("Gradient estimation: d/d_params E_{z~N(mu,sigma^2)}[z^2]")
    print(f"mu={mu}, sigma={sigma}")
    print()
    print(f"{'Method':<20} {'d/d_mu':>10} {'d/d_sigma':>12}")
    print("-" * 44)
    print(f"{'Analytical':<20} {grad_mu_analytical:>10.4f} {grad_sigma_analytical:>12.4f}")
    print(f"{'Reparameterization':<20} {grad_mu_reparam:>10.4f} {grad_sigma_reparam:>12.4f}")
    print(f"{'REINFORCE':<20} {grad_mu_reinforce:>10.4f} {grad_sigma_reinforce:>12.4f}")

    # Variance comparison
    var_reparam_mu = np.var(df_dz * 1)
    var_reinforce_mu = np.var(f_reinforce * score_mu)

    print(f"\nVariance of gradient estimator (d/d_mu):")
    print(f"  Reparameterization: {var_reparam_mu:.4f}")
    print(f"  REINFORCE:          {var_reinforce_mu:.4f}")
    print(f"  Ratio (REINFORCE/reparam): {var_reinforce_mu/var_reparam_mu:.1f}x")


# === Exercise 5: Gibbs Sampling for 3-variate Gaussian ===
# Problem: Gibbs sampling from trivariate Gaussian.

def exercise_5():
    """Gibbs sampling for trivariate Gaussian."""
    np.random.seed(42)

    mu = np.zeros(3)
    Sigma = np.array([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.7],
        [0.5, 0.7, 1.0]
    ])

    Sigma_inv = np.linalg.inv(Sigma)
    n_iterations = 10000
    burn_in = 1000

    # Conditional distributions for Gibbs sampling
    # p(x_i | x_{-i}) = N(mu_cond, sigma_cond^2)
    def conditional_params(i, x, mu, Sigma):
        """Compute conditional mean and variance for x_i given x_{-i}."""
        d = len(mu)
        idx_not_i = [j for j in range(d) if j != i]

        Sigma_ii = Sigma[i, i]
        Sigma_i_noti = Sigma[i, idx_not_i]
        Sigma_noti_noti = Sigma[np.ix_(idx_not_i, idx_not_i)]

        Sigma_noti_inv = np.linalg.inv(Sigma_noti_noti)

        mu_cond = mu[i] + Sigma_i_noti @ Sigma_noti_inv @ (x[idx_not_i] - mu[idx_not_i])
        sigma_cond = Sigma_ii - Sigma_i_noti @ Sigma_noti_inv @ Sigma_i_noti
        return mu_cond, np.sqrt(max(sigma_cond, 1e-10))

    # Gibbs sampling
    x = np.zeros(3)  # initial state
    samples = []

    for t in range(n_iterations + burn_in):
        for i in range(3):
            mu_cond, sigma_cond = conditional_params(i, x, mu, Sigma)
            x[i] = np.random.normal(mu_cond, sigma_cond)

        if t >= burn_in:
            samples.append(x.copy())

    samples = np.array(samples)

    # (b) Compute sample covariance
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples.T)

    print("Gibbs Sampling for Trivariate Gaussian")
    print(f"True mean: {mu}")
    print(f"True covariance:\n{Sigma}")
    print()
    print(f"Sample mean ({n_iterations} samples after {burn_in} burn-in):")
    print(f"  {np.round(sample_mean, 4)}")
    print(f"\nSample covariance:")
    print(f"{np.round(sample_cov, 4)}")
    print()

    # (c) Compare
    print("Element-wise comparison (True vs Sample):")
    max_diff = np.max(np.abs(Sigma - sample_cov))
    print(f"  Max absolute difference: {max_diff:.4f}")
    print(f"  All within 0.05: {max_diff < 0.05}")


if __name__ == "__main__":
    print("=== Exercise 1: Monte Carlo Integration ===")
    exercise_1()
    print("\n=== Exercise 2: Importance Sampling ===")
    exercise_2()
    print("\n=== Exercise 3: MCMC Diagnostics ===")
    exercise_3()
    print("\n=== Exercise 4: Reparameterization Trick ===")
    exercise_4()
    print("\n=== Exercise 5: Gibbs Sampling ===")
    exercise_5()
    print("\nAll exercises completed!")

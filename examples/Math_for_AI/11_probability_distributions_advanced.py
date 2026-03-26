"""
Advanced Probability Distributions for AI/ML

Demonstrates:
- Exponential family of distributions and sufficient statistics
- Multivariate Gaussian: PDF, sampling, conditioning
- Conjugate priors: Bayesian update (Beta-Binomial, Gaussian-Gaussian)
- Visualization of multivariate Gaussian contours and conjugate updates

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# 1. Exponential Family
# ---------------------------------------------------------------------------

def exponential_family():
    """
    Exponential family: p(x|η) = h(x) exp(η · T(x) - A(η))

    Shows that Gaussian, Bernoulli, Poisson, and Gamma all belong to this family.
    """
    print("=" * 60)
    print("EXPONENTIAL FAMILY OF DISTRIBUTIONS")
    print("=" * 60)

    print("\nCanonical form: p(x|η) = h(x) · exp(η·T(x) - A(η))")
    print("  η  — natural parameters")
    print("  T(x) — sufficient statistics")
    print("  A(η) — log-partition function (normalizer)")

    distributions = [
        ("Bernoulli(p)",
         "η = log(p/(1-p))",
         "T(x) = x",
         "A(η) = log(1 + e^η)"),
        ("Gaussian(μ, σ²)",
         "η = [μ/σ², -1/(2σ²)]",
         "T(x) = [x, x²]",
         "A(η) = -η₁²/(4η₂) - ½log(-2η₂)"),
        ("Poisson(λ)",
         "η = log(λ)",
         "T(x) = x",
         "A(η) = e^η"),
        ("Gamma(α, β)",
         "η = [α-1, -β]",
         "T(x) = [log(x), x]",
         "A(η) = log Γ(η₁+1) - (η₁+1)log(-η₂)"),
    ]

    print("\n{:<20s} {:<28s} {:<20s} {}".format(
        "Distribution", "Natural params η", "Sufficient T(x)", "Log-partition A(η)"))
    print("-" * 90)
    for name, eta, Tx, A in distributions:
        print(f"{name:<20s} {eta:<28s} {Tx:<20s} {A}")

    # Demonstrate MLE = method of moments for exponential family
    print("\n--- Key Property: MLE via Sufficient Statistics ---")
    print("For exponential family, MLE satisfies: E_θ[T(X)] = (1/n) Σ T(x_i)")

    # Gaussian example: sufficient stats are mean and second moment
    np.random.seed(42)
    true_mu, true_sigma = 3.0, 1.5
    data = np.random.normal(true_mu, true_sigma, size=500)

    # MLE via sufficient statistics: T(x) = (x, x^2)
    T1 = np.mean(data)          # sufficient stat for μ
    T2 = np.mean(data**2)       # sufficient stat for σ²

    # Recover parameters
    mu_mle = T1
    sigma2_mle = T2 - T1**2

    print(f"\nGaussian example (n=500 samples):")
    print(f"  True μ = {true_mu}, σ² = {true_sigma**2}")
    print(f"  Sufficient stats: E[x] = {T1:.4f},  E[x²] = {T2:.4f}")
    print(f"  MLE estimates: μ = {mu_mle:.4f},  σ² = {sigma2_mle:.4f}")


# ---------------------------------------------------------------------------
# 2. Multivariate Gaussian
# ---------------------------------------------------------------------------

def multivariate_gaussian():
    """Multivariate Gaussian: PDF, sampling, marginals, conditionals"""
    print("\n" + "=" * 60)
    print("MULTIVARIATE GAUSSIAN DISTRIBUTION")
    print("=" * 60)

    print("\nN(x | μ, Σ) = (2π)^(-k/2) |Σ|^(-1/2) exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))")

    # 2D Gaussian with correlation
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[2.0, 0.8],
                      [0.8, 1.0]])

    print(f"\nμ = {mu}")
    print(f"Σ =\n{Sigma}")
    print(f"det(Σ) = {np.linalg.det(Sigma):.4f}")
    print(f"Correlation ρ = Σ₁₂/√(Σ₁₁·Σ₂₂) = {Sigma[0,1]/np.sqrt(Sigma[0,0]*Sigma[1,1]):.4f}")

    # Sample from the distribution
    np.random.seed(42)
    samples = np.random.multivariate_normal(mu, Sigma, size=1000)
    print(f"\nSampled 1000 points:")
    print(f"  Sample mean: {samples.mean(axis=0)}")
    print(f"  Sample covariance:\n{np.cov(samples.T)}")

    # Marginal distributions
    print("\n--- Marginals ---")
    print("Marginal of x₁: N(μ₁, Σ₁₁) = N({:.1f}, {:.1f})".format(mu[0], Sigma[0,0]))
    print("Marginal of x₂: N(μ₂, Σ₂₂) = N({:.1f}, {:.1f})".format(mu[1], Sigma[1,1]))

    # Conditional distribution x₁ | x₂ = v
    print("\n--- Conditional Distribution ---")
    print("x₁ | x₂ = v  ~  N(μ₁|₂, σ²₁|₂)")
    print("  μ₁|₂ = μ₁ + Σ₁₂/Σ₂₂ · (v - μ₂)")
    print("  σ²₁|₂ = Σ₁₁ - Σ₁₂²/Σ₂₂")

    sigma2_cond = Sigma[0,0] - Sigma[0,1]**2 / Sigma[1,1]
    print(f"\nConditional variance σ²₁|₂ = {sigma2_cond:.4f}")

    for v in [1.0, 2.0, 3.0]:
        mu_cond = mu[0] + Sigma[0,1]/Sigma[1,1] * (v - mu[1])
        print(f"  x₁ | x₂={v}: N({mu_cond:.4f}, {sigma2_cond:.4f})")

    # Mahalanobis distance
    print("\n--- Mahalanobis Distance ---")
    print("d_M(x, μ) = √((x-μ)ᵀ Σ⁻¹ (x-μ))")
    Sigma_inv = np.linalg.inv(Sigma)
    test_points = [np.array([1.0, 2.0]), np.array([3.0, 3.0]), np.array([0.0, 0.0])]
    for p in test_points:
        diff = p - mu
        d_mah = np.sqrt(diff @ Sigma_inv @ diff)
        d_euc = np.linalg.norm(diff)
        print(f"  Point {p}: Mahalanobis = {d_mah:.4f},  Euclidean = {d_euc:.4f}")

    return samples, mu, Sigma


# ---------------------------------------------------------------------------
# 3. Conjugate Priors
# ---------------------------------------------------------------------------

def conjugate_prior_beta_binomial():
    """
    Beta-Binomial conjugacy: observe coin flips, update belief about p.
    Prior: p ~ Beta(α, β)
    Likelihood: k successes in n trials
    Posterior: p | data ~ Beta(α + k, β + n - k)
    """
    print("\n" + "=" * 60)
    print("CONJUGATE PRIOR: BETA-BINOMIAL")
    print("=" * 60)

    print("\nModel:")
    print("  Prior:      p ~ Beta(α, β)")
    print("  Likelihood: k | p ~ Binomial(n, p)")
    print("  Posterior:  p | k ~ Beta(α + k, β + n - k)")

    # True coin bias
    true_p = 0.7

    # Prior (weakly informative)
    alpha_prior = 2.0
    beta_prior = 2.0

    print(f"\nTrue p = {true_p}  (unknown)")
    print(f"Prior: Beta({alpha_prior}, {beta_prior})")
    print(f"Prior mean = {alpha_prior/(alpha_prior + beta_prior):.4f}")

    # Sequential updates
    np.random.seed(0)
    observations = [5, 15, 30, 50]   # cumulative sample sizes to show
    flips = np.random.binomial(1, true_p, size=observations[-1])

    alpha = alpha_prior
    beta_param = beta_prior
    prev_n = 0

    print("\n{:<12s} {:<8s} {:<8s} {:<10s} {:<10s}".format(
        "Cum. flips", "Heads", "Tails", "Post. mean", "Post. std"))
    print("-" * 55)

    for n_total in observations:
        batch = flips[prev_n:n_total]
        k = batch.sum()
        alpha += k
        beta_param += len(batch) - k
        post_mean = alpha / (alpha + beta_param)
        post_std = np.sqrt(alpha * beta_param / ((alpha + beta_param)**2 * (alpha + beta_param + 1)))
        heads = int(alpha - alpha_prior)
        tails = int(beta_param - beta_prior)
        print(f"{n_total:<12d} {heads:<8d} {tails:<8d} {post_mean:10.4f} {post_std:10.4f}")
        prev_n = n_total

    print(f"\nFinal posterior: Beta({alpha:.0f}, {beta_param:.0f})")
    print(f"Posterior mean converges toward true p = {true_p}")


def conjugate_prior_gaussian():
    """
    Gaussian-Gaussian conjugacy: known variance, unknown mean.
    Prior:     μ ~ N(μ₀, τ²)
    Likelihood: xᵢ | μ ~ N(μ, σ²)
    Posterior:  μ | data ~ N(μₙ, τₙ²)
    """
    print("\n" + "=" * 60)
    print("CONJUGATE PRIOR: GAUSSIAN-GAUSSIAN (KNOWN VARIANCE)")
    print("=" * 60)

    print("\nModel (known σ² = likelihood variance):")
    print("  Prior:      μ ~ N(μ₀, τ₀²)")
    print("  Likelihood: xᵢ | μ ~ N(μ, σ²)")
    print("  Posterior:  μ | data ~ N(μₙ, τₙ²)")
    print("  where:")
    print("    1/τₙ² = 1/τ₀² + n/σ²")
    print("    μₙ = τₙ² · (μ₀/τ₀² + n·x̄/σ²)")

    true_mu = 5.0
    sigma2 = 4.0    # known likelihood variance

    # Prior
    mu_0 = 0.0
    tau2_0 = 10.0   # weakly informative prior

    print(f"\nTrue μ = {true_mu}  (unknown)")
    print(f"Known σ² = {sigma2}")
    print(f"Prior: N(μ₀={mu_0}, τ₀²={tau2_0})")

    np.random.seed(1)
    data = np.random.normal(true_mu, np.sqrt(sigma2), size=50)

    print("\n{:<10s} {:<12s} {:<12s}".format("n", "Post. mean", "Post. std"))
    print("-" * 38)

    for n in [1, 5, 10, 20, 50]:
        x_bar = data[:n].mean()
        tau2_n = 1.0 / (1.0/tau2_0 + n/sigma2)
        mu_n = tau2_n * (mu_0/tau2_0 + n*x_bar/sigma2)
        print(f"{n:<10d} {mu_n:12.4f} {np.sqrt(tau2_n):12.4f}")

    print(f"\nNote: posterior std shrinks as n increases (more data → more certainty)")
    print(f"Posterior mean converges toward true μ = {true_mu}")


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def visualize_distributions(samples, mu, Sigma):
    """Visualize multivariate Gaussian and conjugate prior updates"""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Plot 1: 2D Gaussian samples and contours ---
    ax1 = axes[0]
    ax1.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=8, color='steelblue', label='Samples')

    # Contour of true PDF
    x1 = np.linspace(mu[0] - 4, mu[0] + 4, 200)
    x2 = np.linspace(mu[1] - 4, mu[1] + 4, 200)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.stack([X1, X2], axis=-1)
    rv = stats.multivariate_normal(mean=mu, cov=Sigma)
    Z = rv.pdf(pos)
    ax1.contour(X1, X2, Z, levels=6, colors='red', linewidths=1.5)
    ax1.plot(*mu, 'r*', markersize=14, label='Mean μ')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('2D Gaussian\nContours + Samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # --- Plot 2: Beta-Binomial conjugate update ---
    ax2 = axes[1]
    p_vals = np.linspace(0, 1, 300)
    true_p = 0.7

    np.random.seed(0)
    data = np.random.binomial(1, true_p, size=100)

    configs = [
        (2, 2, 0, 'Prior Beta(2,2)'),
        (2, 2, 10, '10 flips'),
        (2, 2, 30, '30 flips'),
        (2, 2, 100, '100 flips'),
    ]
    colors = ['gray', 'orange', 'blue', 'green']

    for (a, b, n, label), color in zip(configs, colors):
        if n > 0:
            k = data[:n].sum()
            a += k
            b += (n - k)
        pdf = stats.beta.pdf(p_vals, a, b)
        ax2.plot(p_vals, pdf, color=color, linewidth=2, label=label)

    ax2.axvline(true_p, color='red', linestyle='--', label=f'True p={true_p}')
    ax2.set_xlabel('p')
    ax2.set_ylabel('Density')
    ax2.set_title('Beta-Binomial\nConjugate Update')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Gaussian-Gaussian conjugate update ---
    ax3 = axes[2]
    mu_range = np.linspace(-2, 10, 300)
    true_mu = 5.0
    sigma2 = 4.0
    mu_0, tau2_0 = 0.0, 10.0

    np.random.seed(1)
    data_g = np.random.normal(true_mu, np.sqrt(sigma2), size=50)

    configs_g = [(0, 'Prior N(0, 10)'), (5, '5 obs'), (20, '20 obs'), (50, '50 obs')]
    colors_g = ['gray', 'orange', 'blue', 'green']

    for (n, label), color in zip(configs_g, colors_g):
        if n == 0:
            mu_n, tau2_n = mu_0, tau2_0
        else:
            x_bar = data_g[:n].mean()
            tau2_n = 1.0 / (1.0/tau2_0 + n/sigma2)
            mu_n = tau2_n * (mu_0/tau2_0 + n*x_bar/sigma2)
        pdf = stats.norm.pdf(mu_range, mu_n, np.sqrt(tau2_n))
        ax3.plot(mu_range, pdf, color=color, linewidth=2, label=label)

    ax3.axvline(true_mu, color='red', linestyle='--', label=f'True μ={true_mu}')
    ax3.set_xlabel('μ')
    ax3.set_ylabel('Density')
    ax3.set_title('Gaussian-Gaussian\nConjugate Update')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('probability_distributions_advanced.png', dpi=150)
    print("Visualization saved to probability_distributions_advanced.png")
    plt.close()


if __name__ == "__main__":
    exponential_family()
    samples, mu, Sigma = multivariate_gaussian()
    conjugate_prior_beta_binomial()
    conjugate_prior_gaussian()
    visualize_distributions(samples, mu, Sigma)

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)

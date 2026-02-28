"""
Exercises for Lesson 11: Probability Distributions Advanced
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import stats


# === Exercise 1: Exponential Family Transformation ===
# Problem: Transform Geometric and Laplace distributions to exponential family form.

def exercise_1():
    """Identify exponential family components for various distributions."""
    print("Exponential Family Form: p(x|eta) = h(x) * exp(eta^T T(x) - A(eta))")
    print()

    # (a) Geometric distribution: p(x|p) = (1-p)^{x-1} * p, x=1,2,...
    print("(a) Geometric distribution: p(x|p) = (1-p)^{x-1} * p")
    print()
    print("  Rewrite: p(x|p) = p * (1-p)^{x-1}")
    print("         = exp(log(p) + (x-1)*log(1-p))")
    print("         = exp(log(p) - log(1-p) + x*log(1-p))")
    print()
    print("  Natural parameter: eta = log(1-p)")
    print("  Sufficient statistic: T(x) = x")
    print("  Log-partition: A(eta) = -log(1 - e^eta) = -log(p)")
    print("  Base measure: h(x) = 1")
    print()

    # Verify numerically
    p = 0.3
    eta = np.log(1 - p)
    x_vals = np.arange(1, 11)
    prob_direct = (1 - p)**(x_vals - 1) * p
    prob_expfam = np.exp(eta * x_vals + np.log(p) - np.log(1 - p))

    # Fix: A(eta) = log(p) - log(1-p) when moving constant out
    # Actually: p(x) = exp(x*log(1-p) + log(p/(1-p)))
    # So: A(eta) = -log(1 - e^eta), with eta = log(1-p)
    # And h(x) = 1, with an extra constant
    A_eta = -np.log(1 - np.exp(eta))
    prob_expfam2 = np.exp(eta * x_vals - A_eta)

    print(f"  Verification (p={p}):")
    print(f"  x:        {x_vals[:5]}")
    print(f"  Direct:   {np.round(prob_direct[:5], 6)}")
    print(f"  Exp-fam:  {np.round(prob_expfam2[:5], 6)}")
    print(f"  Match: {np.allclose(prob_direct[:5], prob_expfam2[:5], atol=1e-6)}")
    print()

    # (b) Laplace distribution with fixed b: p(x|mu, b) = (1/2b)*exp(-|x-mu|/b)
    print("(b) Laplace distribution (b fixed): p(x|mu,b) = (1/(2b)) * exp(-|x-mu|/b)")
    print()
    print("  Note: This is NOT naturally in exponential family form")
    print("  because |x-mu| is not linear in mu.")
    print("  However, for fixed mu (known location), with b as parameter:")
    print("  p(x|b) = (1/(2b)) * exp(-|x-mu|/b)")
    print("         = exp(-log(2b) - |x-mu|/b)")
    print()
    print("  Natural parameter: eta = -1/b")
    print("  Sufficient statistic: T(x) = |x - mu|")
    print("  Log-partition: A(eta) = -log(-2*eta) = log(2b)")
    print("  Base measure: h(x) = 1")


# === Exercise 2: Properties of Log Partition Function ===
# Problem: Verify E[T(X)] = dA/d_eta and Var(T(X)) = d^2A/d_eta^2.

def exercise_2():
    """Verify log-partition function derivatives give moments."""
    print("Exponential family: p(x|eta) = h(x) exp(eta*T(x) - A(eta))")
    print()
    print("Proof sketch:")
    print("  Normalization: integral h(x) exp(eta*T(x) - A(eta)) dx = 1")
    print("  => integral h(x) exp(eta*T(x)) dx = exp(A(eta))")
    print()
    print("  Differentiate w.r.t. eta:")
    print("  integral T(x) h(x) exp(eta*T(x)) dx = A'(eta) exp(A(eta))")
    print("  => E[T(X)] = A'(eta)")
    print()
    print("  Differentiate again:")
    print("  => Var(T(X)) = A''(eta)")
    print()

    # Verify for Poisson distribution
    # Poisson: p(x|lambda) = lambda^x * e^{-lambda} / x!
    # Exponential family: eta = log(lambda), T(x) = x, A(eta) = e^eta = lambda
    print("Verification with Poisson distribution:")
    print("  eta = log(lambda), A(eta) = exp(eta) = lambda")
    print("  A'(eta) = exp(eta) = lambda = E[X]  (correct!)")
    print("  A''(eta) = exp(eta) = lambda = Var(X)  (correct!)")
    print()

    for lam in [1, 5, 10]:
        eta = np.log(lam)
        A_prime = np.exp(eta)  # should be lambda
        A_double_prime = np.exp(eta)  # should also be lambda

        # Empirical verification
        np.random.seed(42)
        samples = np.random.poisson(lam, 100000)
        emp_mean = np.mean(samples)
        emp_var = np.var(samples)

        print(f"  lambda={lam}: A'={A_prime:.4f} (E[X]={emp_mean:.4f}), "
              f"A''={A_double_prime:.4f} (Var(X)={emp_var:.4f})")


# === Exercise 3: Gamma-Poisson Conjugacy ===
# Problem: Show Gamma is conjugate prior for Poisson, derive posterior.

def exercise_3():
    """Gamma-Poisson conjugacy with Bayesian update."""
    print("Poisson likelihood: p(x|lambda) = lambda^x * e^{-lambda} / x!")
    print("Gamma prior: p(lambda) = Gamma(alpha, beta)")
    print("           = beta^alpha / Gamma(alpha) * lambda^{alpha-1} * e^{-beta*lambda}")
    print()
    print("Posterior after n observations with sum = s:")
    print("  p(lambda|D) propto lambda^{alpha+s-1} * e^{-(beta+n)*lambda}")
    print("  => lambda|D ~ Gamma(alpha + s, beta + n)")
    print()

    # Prior parameters
    alpha_prior = 2.0
    beta_prior = 1.0

    # Generate data from Poisson(lambda_true)
    np.random.seed(42)
    lambda_true = 5.0
    n_obs = 20
    data = np.random.poisson(lambda_true, n_obs)
    s = np.sum(data)

    # Posterior
    alpha_post = alpha_prior + s
    beta_post = beta_prior + n_obs

    # Posterior mean and variance
    post_mean = alpha_post / beta_post
    post_var = alpha_post / beta_post**2

    print(f"True lambda: {lambda_true}")
    print(f"Data: n={n_obs}, sum={s}, sample mean={np.mean(data):.2f}")
    print()
    print(f"Prior:     Gamma({alpha_prior}, {beta_prior})")
    print(f"  Mean = {alpha_prior/beta_prior:.4f}, Var = {alpha_prior/beta_prior**2:.4f}")
    print()
    print(f"Posterior: Gamma({alpha_post}, {beta_post})")
    print(f"  Mean = {post_mean:.4f}, Var = {post_var:.4f}")
    print(f"  95% CI: [{stats.gamma.ppf(0.025, alpha_post, scale=1/beta_post):.4f}, "
          f"{stats.gamma.ppf(0.975, alpha_post, scale=1/beta_post):.4f}]")
    print()

    # Sequential updates
    print("Sequential Bayesian update:")
    alpha_n = alpha_prior
    beta_n = beta_prior
    for i in range(min(10, n_obs)):
        alpha_n = alpha_n + data[i]
        beta_n = beta_n + 1
        print(f"  After obs {i+1} (x={data[i]}): "
              f"Gamma({alpha_n:.0f}, {beta_n:.0f}), "
              f"mean={alpha_n/beta_n:.4f}")


# === Exercise 4: Conditional of Multivariate Gaussian ===
# Problem: Compute conditional distribution (x1, x2) | x3 = 1.5.

def exercise_4():
    """Conditional distribution of multivariate Gaussian."""
    mu = np.array([0, 1, 2])
    Sigma = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 2.0, 0.4],
        [0.3, 0.4, 1.0]
    ])

    # Partition: X_1 = (x1, x2), X_2 = (x3)
    # mu_1 = [0, 1], mu_2 = [2]
    mu_1 = mu[:2]
    mu_2 = mu[2:]

    Sigma_11 = Sigma[:2, :2]
    Sigma_12 = Sigma[:2, 2:]
    Sigma_21 = Sigma[2:, :2]
    Sigma_22 = Sigma[2:, 2:]

    x3_obs = 1.5

    # Conditional: (x1, x2) | x3 ~ N(mu_cond, Sigma_cond)
    # mu_cond = mu_1 + Sigma_12 @ Sigma_22^{-1} @ (x3 - mu_2)
    # Sigma_cond = Sigma_11 - Sigma_12 @ Sigma_22^{-1} @ Sigma_21
    Sigma_22_inv = 1.0 / Sigma_22[0, 0]

    mu_cond = mu_1 + Sigma_12.flatten() * Sigma_22_inv * (x3_obs - mu_2[0])
    Sigma_cond = Sigma_11 - Sigma_22_inv * np.outer(Sigma_12.flatten(), Sigma_21.flatten())

    print("Original distribution:")
    print(f"  mu = {mu}")
    print(f"  Sigma =\n{Sigma}")
    print()
    print(f"Conditioning on x3 = {x3_obs}")
    print()
    print("Conditional distribution: (x1, x2) | x3 = 1.5")
    print(f"  mu_cond = {np.round(mu_cond, 4)}")
    print(f"  Sigma_cond =\n{np.round(Sigma_cond, 4)}")
    print()

    # Verification by sampling
    np.random.seed(42)
    n_samples = 100000
    samples = np.random.multivariate_normal(mu, Sigma, n_samples)

    # Filter samples where x3 is close to 1.5
    tol = 0.1
    mask = np.abs(samples[:, 2] - x3_obs) < tol
    filtered = samples[mask, :2]

    print(f"Empirical verification ({np.sum(mask)} samples with |x3 - 1.5| < {tol}):")
    print(f"  Empirical mean:     {np.round(np.mean(filtered, axis=0), 4)}")
    print(f"  Conditional mean:   {np.round(mu_cond, 4)}")
    print(f"  Empirical cov:\n{np.round(np.cov(filtered.T), 4)}")
    print(f"  Conditional cov:\n{np.round(Sigma_cond, 4)}")


# === Exercise 5: GMM vs K-means ===
# Problem: Compare GMM and K-means on the same data.

def exercise_5():
    """Compare GMM and K-means clustering."""
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    np.random.seed(42)

    # Generate data with non-spherical clusters
    n = 300
    # Cluster 1: elongated
    C1 = np.random.randn(n, 2) @ np.array([[2, 0.5], [0.5, 0.3]]) + np.array([0, 0])
    # Cluster 2: compact
    C2 = np.random.randn(n, 2) * 0.5 + np.array([5, 3])
    # Cluster 3: circular but larger
    C3 = np.random.randn(n, 2) * 1.5 + np.array([2, 6])

    X = np.vstack([C1, C2, C3])
    true_labels = np.array([0]*n + [1]*n + [2]*n)

    print("Data: 3 clusters with different shapes and sizes")
    print(f"Total samples: {len(X)}")
    print()

    # (a) Differences between GMM and K-means
    print("(a) Key differences:")
    print("  K-means: Hard assignment, spherical clusters, equal-sized clusters assumed")
    print("  GMM: Soft assignment (probabilities), elliptical clusters, different sizes OK")
    print()

    # (b) K-means as special case of GMM
    print("(b) K-means as GMM special case:")
    print("  When Sigma_k = sigma^2 * I for all k and sigma -> 0:")
    print("  - Soft assignments become hard (responsibility -> 0 or 1)")
    print("  - Log-likelihood reduces to within-cluster sum of squares")
    print("  - EM becomes equivalent to Lloyd's algorithm")
    print()

    # (c) Comparison on data
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)

    labels_km = kmeans.fit_predict(X)
    labels_gmm = gmm.fit_predict(X)

    ari_km = adjusted_rand_score(true_labels, labels_km)
    ari_gmm = adjusted_rand_score(true_labels, labels_gmm)
    sil_km = silhouette_score(X, labels_km)
    sil_gmm = silhouette_score(X, labels_gmm)

    print("(c) Comparison results:")
    print(f"  {'Metric':<25} {'K-means':>10} {'GMM':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Adjusted Rand Index':<25} {ari_km:>10.4f} {ari_gmm:>10.4f}")
    print(f"  {'Silhouette Score':<25} {sil_km:>10.4f} {sil_gmm:>10.4f}")

    # GMM gives cluster probabilities
    probs = gmm.predict_proba(X[:5])
    print(f"\n  GMM soft assignments (first 5 samples):")
    for i in range(5):
        print(f"    Sample {i}: {np.round(probs[i], 3)} (true: {true_labels[i]})")

    # GMM covariance info
    print(f"\n  GMM learned covariances (diagonal elements):")
    for k in range(3):
        diag = np.diag(gmm.covariances_[k])
        print(f"    Cluster {k}: {np.round(diag, 3)}")


if __name__ == "__main__":
    print("=== Exercise 1: Exponential Family Transformation ===")
    exercise_1()
    print("\n=== Exercise 2: Log Partition Function Properties ===")
    exercise_2()
    print("\n=== Exercise 3: Gamma-Poisson Conjugacy ===")
    exercise_3()
    print("\n=== Exercise 4: Conditional of Multivariate Gaussian ===")
    exercise_4()
    print("\n=== Exercise 5: GMM vs K-means ===")
    exercise_5()
    print("\nAll exercises completed!")

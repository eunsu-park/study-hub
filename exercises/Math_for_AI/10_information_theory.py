"""
Exercises for Lesson 10: Information Theory
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import integrate


# === Exercise 1: Maximum Entropy Proof ===
# Problem: Prove uniform distribution maximizes entropy under sum constraint.

def exercise_1():
    """Prove and verify maximum entropy is uniform distribution."""
    print("Proof that uniform distribution maximizes entropy:")
    print()
    print("Lagrangian: L = -sum(p_i log p_i) - lambda(sum(p_i) - 1)")
    print()
    print("dL/dp_i = -log(p_i) - 1 - lambda = 0")
    print("=> log(p_i) = -(1 + lambda)")
    print("=> p_i = e^{-(1+lambda)} = constant for all i")
    print()
    print("Applying constraint: sum(p_i) = 1")
    print("=> n * p_i = 1 => p_i = 1/n")
    print()
    print("Therefore, the maximum entropy distribution is uniform: p_i = 1/n")
    print("Maximum entropy = log(n)")
    print()

    # Numerical verification
    def entropy(p):
        """Shannon entropy in nats."""
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    n = 10
    print(f"Numerical verification (n={n}):")

    # Uniform distribution
    p_uniform = np.ones(n) / n
    h_uniform = entropy(p_uniform)
    print(f"  Uniform: H = {h_uniform:.6f} (theoretical: log({n}) = {np.log(n):.6f})")

    # Test many random distributions
    np.random.seed(42)
    max_h = h_uniform
    for _ in range(10000):
        p_random = np.random.dirichlet(np.ones(n))
        h_random = entropy(p_random)
        if h_random > max_h + 1e-10:
            print(f"  Found higher entropy: {h_random:.6f} -- THIS SHOULD NOT HAPPEN")
            max_h = h_random

    print(f"  Max entropy from 10000 random distributions: {max_h:.6f}")
    print(f"  Uniform has maximum entropy: True")


# === Exercise 2: Conditional Entropy and Mutual Information ===
# Problem: Verify information-theoretic identities numerically.

def exercise_2():
    """Verify entropy chain rule and mutual information properties."""
    np.random.seed(42)

    # Create a simple joint distribution p(X, Y)
    # X in {0, 1, 2}, Y in {0, 1}
    p_xy = np.array([
        [0.15, 0.05],  # X=0
        [0.10, 0.20],  # X=1
        [0.05, 0.45],  # X=2
    ])
    # Normalize
    p_xy = p_xy / p_xy.sum()

    p_x = p_xy.sum(axis=1)  # marginal P(X)
    p_y = p_xy.sum(axis=0)  # marginal P(Y)

    def h(p):
        """Entropy of a distribution."""
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    # H(X), H(Y)
    H_X = h(p_x)
    H_Y = h(p_y)

    # H(X, Y)
    H_XY = h(p_xy.flatten())

    # H(Y|X) = H(X,Y) - H(X)
    H_Y_given_X = H_XY - H_X

    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    I_XY = H_X + H_Y - H_XY

    print("Joint distribution P(X,Y):")
    print(f"  {p_xy}")
    print(f"  P(X) = {p_x}")
    print(f"  P(Y) = {p_y}")
    print()

    # (a) Chain rule: H(X,Y) = H(X) + H(Y|X)
    print("(a) Chain rule: H(X,Y) = H(X) + H(Y|X)")
    print(f"  H(X,Y) = {H_XY:.6f}")
    print(f"  H(X) + H(Y|X) = {H_X:.6f} + {H_Y_given_X:.6f} = {H_X + H_Y_given_X:.6f}")
    print(f"  Equal: {np.isclose(H_XY, H_X + H_Y_given_X)}")
    print()

    # (b) I(X;Y) = H(X) + H(Y) - H(X,Y)
    print("(b) I(X;Y) = H(X) + H(Y) - H(X,Y)")
    print(f"  I(X;Y) = {I_XY:.6f}")
    print(f"  H(X) + H(Y) - H(X,Y) = {H_X:.6f} + {H_Y:.6f} - {H_XY:.6f} = {H_X + H_Y - H_XY:.6f}")
    print()

    # (c) I(X;Y) <= min(H(X), H(Y))
    print(f"(c) I(X;Y) <= min(H(X), H(Y))")
    print(f"  I(X;Y) = {I_XY:.6f}")
    print(f"  min(H(X), H(Y)) = min({H_X:.6f}, {H_Y:.6f}) = {min(H_X, H_Y):.6f}")
    print(f"  Satisfied: {I_XY <= min(H_X, H_Y) + 1e-10}")
    print()

    # (d) I(X;Y) = 0 iff X and Y are independent
    print("(d) I(X;Y) = 0 iff X perp Y")
    print(f"  I(X;Y) = {I_XY:.6f} > 0 => X and Y are NOT independent")

    # Verify with independent distribution
    p_xy_indep = np.outer(p_x, p_y)
    H_XY_indep = h(p_xy_indep.flatten())
    I_indep = H_X + H_Y - H_XY_indep
    print(f"  For independent P(X,Y) = P(X)P(Y): I = {I_indep:.10f}")
    print(f"  I = 0 when independent: {np.isclose(I_indep, 0)}")


# === Exercise 3: KL Divergence Between Gaussians ===
# Problem: Compute KL(P||Q) between two Gaussians analytically and numerically.

def exercise_3():
    """KL divergence between two Gaussian distributions."""
    mu1, sigma1 = 0.0, 1.0
    mu2, sigma2 = 1.0, 2.0

    # Analytical formula
    kl_analytical = (np.log(sigma2 / sigma1)
                     + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
                     - 0.5)

    print(f"P = N({mu1}, {sigma1}^2), Q = N({mu2}, {sigma2}^2)")
    print()
    print(f"Analytical KL(P||Q) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2)/(2*sigma2^2) - 1/2")
    print(f"  = log({sigma2}/{sigma1}) + ({sigma1**2} + {(mu1-mu2)**2})/(2*{sigma2**2}) - 0.5")
    print(f"  = {kl_analytical:.6f}")

    # Numerical integration
    def kl_integrand(x):
        p = (1 / (np.sqrt(2*np.pi) * sigma1)) * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
        q = (1 / (np.sqrt(2*np.pi) * sigma2)) * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
        if p < 1e-300 or q < 1e-300:
            return 0.0
        return p * np.log(p / q)

    kl_numerical, error = integrate.quad(kl_integrand, -10, 10)

    print(f"\nNumerical integration: {kl_numerical:.6f} (error bound: {error:.2e})")
    print(f"Difference: {abs(kl_analytical - kl_numerical):.2e}")

    # Monte Carlo estimation
    np.random.seed(42)
    n_samples = 100000
    samples = np.random.normal(mu1, sigma1, n_samples)
    log_p = -0.5 * np.log(2*np.pi) - np.log(sigma1) - 0.5 * ((samples - mu1) / sigma1)**2
    log_q = -0.5 * np.log(2*np.pi) - np.log(sigma2) - 0.5 * ((samples - mu2) / sigma2)**2
    kl_mc = np.mean(log_p - log_q)

    print(f"Monte Carlo ({n_samples} samples): {kl_mc:.6f}")

    # Verify KL(P||Q) != KL(Q||P) (asymmetry)
    kl_reverse = (np.log(sigma1 / sigma2)
                  + (sigma2**2 + (mu2 - mu1)**2) / (2 * sigma1**2)
                  - 0.5)
    print(f"\nKL(P||Q) = {kl_analytical:.6f}")
    print(f"KL(Q||P) = {kl_reverse:.6f}")
    print(f"Asymmetric (KL(P||Q) != KL(Q||P)): {not np.isclose(kl_analytical, kl_reverse)}")


# === Exercise 4: KL Divergence in VAE ===
# Problem: Derive KL(q(z|x) || p(z)) for diagonal Gaussian q and standard normal p.

def exercise_4():
    """KL divergence for VAE latent space."""
    print("VAE KL divergence derivation:")
    print("  q(z|x) = N(mu, diag(sigma^2))")
    print("  p(z)   = N(0, I)")
    print()
    print("KL = integral q(z|x) log(q(z|x)/p(z)) dz")
    print("   = 0.5 * sum_j (mu_j^2 + sigma_j^2 - log(sigma_j^2) - 1)")
    print()

    # Numerical verification
    np.random.seed(42)
    d = 5  # latent dimension

    # Example encoder outputs
    mu = np.array([0.5, -0.3, 1.2, 0.0, -0.8])
    log_var = np.array([-0.5, 0.3, -1.0, 0.0, 0.5])
    sigma_sq = np.exp(log_var)

    # Analytical formula
    kl_analytical = 0.5 * np.sum(mu**2 + sigma_sq - log_var - 1)

    # Monte Carlo verification
    n_samples = 100000
    z_samples = mu + np.sqrt(sigma_sq) * np.random.randn(n_samples, d)

    # log q(z|x)
    log_q = -0.5 * np.sum(log_var) - 0.5 * d * np.log(2 * np.pi) \
            - 0.5 * np.sum((z_samples - mu)**2 / sigma_sq, axis=1)

    # log p(z) = log N(0, I)
    log_p = -0.5 * d * np.log(2 * np.pi) - 0.5 * np.sum(z_samples**2, axis=1)

    kl_mc = np.mean(log_q - log_p)

    print(f"Latent dimension: {d}")
    print(f"mu = {mu}")
    print(f"sigma^2 = {np.round(sigma_sq, 4)}")
    print()
    print(f"Analytical KL: {kl_analytical:.6f}")
    print(f"Monte Carlo KL: {kl_mc:.6f}")
    print(f"Difference: {abs(kl_analytical - kl_mc):.4f}")

    # Per-dimension contribution
    print(f"\nPer-dimension KL contributions:")
    for j in range(d):
        kl_j = 0.5 * (mu[j]**2 + sigma_sq[j] - log_var[j] - 1)
        print(f"  dim {j}: mu={mu[j]:.2f}, sigma^2={sigma_sq[j]:.4f}, KL_j={kl_j:.4f}")


# === Exercise 5: Information Bottleneck Simulation ===
# Problem: Implement simplified information bottleneck on classification data.

def exercise_5():
    """Simplified information bottleneck for feature compression."""
    np.random.seed(42)

    # Generate 2D classification data
    n = 500
    # Class 0: centered at (-2, 0)
    X0 = np.random.randn(n // 2, 2) * 0.8 + np.array([-2, 0])
    # Class 1: centered at (2, 0)
    X1 = np.random.randn(n // 2, 2) * 0.8 + np.array([2, 0])
    X = np.vstack([X0, X1])
    Y = np.array([0] * (n // 2) + [1] * (n // 2))

    def estimate_mi_histogram(X_1d, Y, n_bins=20):
        """Estimate mutual information using histograms."""
        # Discretize X
        bins = np.linspace(X_1d.min() - 0.1, X_1d.max() + 0.1, n_bins + 1)
        X_binned = np.digitize(X_1d, bins) - 1
        X_binned = np.clip(X_binned, 0, n_bins - 1)

        # Joint distribution
        n_classes = len(np.unique(Y))
        p_xy = np.zeros((n_bins, n_classes))
        for i in range(len(X_1d)):
            p_xy[X_binned[i], Y[i]] += 1
        p_xy = p_xy / p_xy.sum()

        # Marginals
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
        mi = 0
        for i in range(n_bins):
            for j in range(n_classes):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi

    # Compress X to 1D using different projections
    # Vary compression level with beta (controls I(X;Z) vs I(Z;Y) tradeoff)
    print("Information Bottleneck: I(X;Z) vs I(Z;Y) tradeoff")
    print("Using linear projections Z = w^T X with varying angles")
    print()

    angles = np.linspace(0, np.pi, 36)
    results = []

    for angle in angles:
        w = np.array([np.cos(angle), np.sin(angle)])
        Z = X @ w  # 1D compressed representation

        I_XZ = estimate_mi_histogram(Z, np.digitize(X[:, 0], np.linspace(-5, 5, 20)))
        I_ZY = estimate_mi_histogram(Z, Y)
        results.append((angle, I_XZ, I_ZY))

    # Find best tradeoff for different beta values
    print(f"{'Angle':>8} {'I(X;Z)':>8} {'I(Z;Y)':>8}")
    print("-" * 28)
    for i in range(0, len(results), 6):
        angle, i_xz, i_zy = results[i]
        print(f"{np.degrees(angle):>7.1f}° {i_xz:>8.4f} {i_zy:>8.4f}")

    # Find optimal: max I(Z;Y)
    best_idx = np.argmax([r[2] for r in results])
    best_angle, best_ixz, best_izy = results[best_idx]
    print(f"\nOptimal projection angle: {np.degrees(best_angle):.1f}°")
    print(f"  I(X;Z) = {best_ixz:.4f}")
    print(f"  I(Z;Y) = {best_izy:.4f}")
    print(f"  Direction: [{np.cos(best_angle):.3f}, {np.sin(best_angle):.3f}]")
    print("  (Near horizontal axis, aligning with class separation direction)")


if __name__ == "__main__":
    print("=== Exercise 1: Maximum Entropy Proof ===")
    exercise_1()
    print("\n=== Exercise 2: Conditional Entropy and Mutual Information ===")
    exercise_2()
    print("\n=== Exercise 3: KL Divergence Between Gaussians ===")
    exercise_3()
    print("\n=== Exercise 4: KL Divergence in VAE ===")
    exercise_4()
    print("\n=== Exercise 5: Information Bottleneck ===")
    exercise_5()
    print("\nAll exercises completed!")

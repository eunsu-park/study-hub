"""
Exercises for Lesson 06: Probability Distributions for DL
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_bce_logit_gradient():
    """Derive gradient of BCE w.r.t. logit z (before sigmoid).

    Show that d(BCE)/dz = sigmoid(z) - y.
    Verify numerically for z=2.0, y=1.
    """
    z, y = 2.0, 1.0

    # TODO: Compute analytical gradient
    # sigmoid(z) = 1 / (1 + exp(-z))
    # d(BCE)/dz = sigmoid(z) - y
    grad_analytical = None  # Replace

    # TODO: Compute numerical gradient
    eps = 1e-5
    def bce_from_logit(z, y):
        p = 1 / (1 + np.exp(-z))
        return -(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    grad_numerical = None  # Replace: (bce(z+eps) - bce(z-eps)) / (2*eps)

    return grad_analytical, grad_numerical


def exercise_2_kl_two_gaussians():
    """Compute KL(N(mu1,s1^2) || N(mu2,s2^2)) and verify with Monte Carlo.

    KL = log(s2/s1) + (s1^2 + (mu1-mu2)^2) / (2*s2^2) - 0.5
    """
    mu1, sigma1 = 1.0, 0.5
    mu2, sigma2 = 0.0, 1.0

    # TODO: Compute KL analytically
    kl_analytical = None  # Replace

    # TODO: Monte Carlo verification
    n = 100000
    np.random.seed(42)
    kl_mc = None  # Replace: sample from q, compute mean of log(q/p)

    return kl_analytical, kl_mc


def exercise_3_vae_kl_gradient():
    """Compute gradient of VAE KL term w.r.t. mu and log_sigma.

    KL = -0.5 * sum(1 + 2*log_sigma - mu^2 - exp(2*log_sigma))
    """
    mu = np.array([1.0, -0.5])
    log_sigma = np.array([0.5, -0.3])

    # TODO: Compute dKL/dmu and dKL/dlog_sigma
    dkl_dmu = None  # Replace
    dkl_dlog_sigma = None  # Replace

    return dkl_dmu, dkl_dlog_sigma


if __name__ == "__main__":
    print("Exercise 1: BCE logit gradient")
    ga, gn = exercise_1_bce_logit_gradient()
    if ga is not None and gn is not None:
        print(f"  Analytical: {ga:.6f}, Numerical: {gn:.6f}")
        print(f"  Pass: {abs(ga - gn) < 1e-4}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: KL between Gaussians")
    ka, km = exercise_2_kl_two_gaussians()
    if ka is not None:
        print(f"  Analytical: {ka:.4f}")
        if km is not None:
            print(f"  Monte Carlo: {km:.4f}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: VAE KL gradient")
    dm, ds = exercise_3_vae_kl_gradient()
    if dm is not None:
        print(f"  dKL/dmu = {dm}")
        print(f"  dKL/dlog_sigma = {ds}")
    else:
        print("  Not implemented yet")

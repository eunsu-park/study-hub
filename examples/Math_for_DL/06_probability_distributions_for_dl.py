"""
Probability Distributions for Deep Learning

Demonstrates distributions used in DL and their loss connections:
- Bernoulli -> BCE, Gaussian -> MSE, Categorical -> CE
- Reparameterization trick for VAEs
- KL divergence between Gaussians

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def distribution_loss_connection():
    """Show the distribution-to-loss mapping."""
    print("=" * 60)
    print("DISTRIBUTION-LOSS CONNECTIONS")
    print("=" * 60)

    # Bernoulli -> BCE
    y, p = 1.0, 0.8
    bce = -(y*np.log(p) + (1-y)*np.log(1-p))
    print(f"Bernoulli(p={p}), y={y}: BCE = {bce:.4f}")

    # Gaussian -> MSE
    y_true, y_pred, sigma = 3.0, 2.5, 1.0
    nll_gauss = (y_true - y_pred)**2 / (2*sigma**2) + 0.5*np.log(2*np.pi*sigma**2)
    mse = (y_true - y_pred)**2
    print(f"Gaussian(mu={y_pred}), y={y_true}: NLL = {nll_gauss:.4f}, MSE = {mse:.4f}")

    # Categorical -> CE
    logits = np.array([2.0, 1.0, 0.1, -1.0, 3.0])
    true_class = 4
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    ce = -np.log(probs[true_class])
    print(f"Categorical, true_class={true_class}: CE = {ce:.4f}, P(true) = {probs[true_class]:.4f}")


def reparameterization_trick():
    """Demonstrate the reparameterization trick."""
    print("\n" + "=" * 60)
    print("REPARAMETERIZATION TRICK")
    print("=" * 60)

    np.random.seed(42)
    mu = np.array([1.0, -0.5])
    log_sigma = np.array([0.5, -0.3])
    sigma = np.exp(log_sigma)

    n_samples = 10000
    eps = np.random.randn(n_samples, 2)
    z = mu + sigma * eps

    print(f"mu = {mu}, sigma = {sigma}")
    print(f"Sample mean: {z.mean(axis=0).round(3)}")
    print(f"Sample std:  {z.std(axis=0).round(3)}")

    # Gradients: dz/dmu = I, dz/dsigma = diag(eps)
    dz_dmu = np.eye(2)
    print(f"dz/dmu = I (shape {dz_dmu.shape})")


def kl_gaussian():
    """KL divergence between Gaussian and standard normal."""
    print("\n" + "=" * 60)
    print("KL DIVERGENCE: GAUSSIAN -> STANDARD NORMAL")
    print("=" * 60)

    mu = np.array([1.0, -0.5, 0.3])
    log_sigma = np.array([0.5, -0.3, 0.1])
    sigma = np.exp(log_sigma)

    # Analytical
    kl = -0.5 * np.sum(1 + 2*log_sigma - mu**2 - sigma**2)

    # Monte Carlo verification
    n = 100000
    z = mu + sigma * np.random.randn(n, 3)
    log_q = -0.5*np.sum((z-mu)**2/sigma**2, axis=1) - np.sum(log_sigma) - 1.5*np.log(2*np.pi)
    log_p = -0.5*np.sum(z**2, axis=1) - 1.5*np.log(2*np.pi)
    kl_mc = np.mean(log_q - log_p)

    print(f"KL analytical:   {kl:.4f}")
    print(f"KL Monte Carlo:  {kl_mc:.4f}")


if __name__ == "__main__":
    distribution_loss_connection()
    reparameterization_trick()
    kl_gaussian()

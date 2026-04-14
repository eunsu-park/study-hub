"""
Maximum Likelihood Estimation

Demonstrates MLE and its connection to DL training:
- MLE for Gaussian parameters
- Softmax cross-entropy gradient derivation
- Logistic regression from scratch with gradient descent
- L2 regularization as MAP estimation

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian_mle():
    """MLE for Gaussian parameters."""
    print("=" * 60)
    print("GAUSSIAN MLE")
    print("=" * 60)
    np.random.seed(42)
    true_mu, true_sigma = 3.0, 1.5
    data = np.random.normal(true_mu, true_sigma, 200)
    mu_mle = np.mean(data)
    sigma_mle = np.sqrt(np.mean((data - mu_mle)**2))
    print(f"True:  mu={true_mu}, sigma={true_sigma}")
    print(f"MLE:   mu={mu_mle:.3f}, sigma={sigma_mle:.3f}")


def softmax_ce_gradient():
    """Verify softmax cross-entropy gradient = probs - labels."""
    print("\n" + "=" * 60)
    print("SOFTMAX CROSS-ENTROPY GRADIENT")
    print("=" * 60)

    K = 5
    z = np.array([2.0, 1.0, 0.1, -1.0, 3.0])
    y = np.zeros(K); y[2] = 1.0

    # Forward
    e = np.exp(z - np.max(z))
    s = e / e.sum()
    loss = -np.sum(y * np.log(s + 1e-10))

    # Analytical gradient
    grad_ana = s - y

    # Numerical gradient
    eps = 1e-5
    grad_num = np.zeros(K)
    for j in range(K):
        zp = z.copy(); zp[j] += eps
        zm = z.copy(); zm[j] -= eps
        ep = np.exp(zp - np.max(zp)); sp = ep / ep.sum()
        em = np.exp(zm - np.max(zm)); sm = em / em.sum()
        grad_num[j] = (-np.sum(y*np.log(sp+1e-10)) + np.sum(y*np.log(sm+1e-10))) / (2*eps)

    print(f"Probs: {s.round(4)}")
    print(f"Grad analytical: {grad_ana.round(6)}")
    print(f"Grad numerical:  {grad_num.round(6)}")
    print(f"Sum of gradient: {grad_ana.sum():.2e} (should be ~0)")


def logistic_regression():
    """Train logistic regression via MLE (gradient descent)."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION (MLE)")
    print("=" * 60)

    np.random.seed(42)
    N = 200
    X_pos = np.random.randn(N//2, 2) + np.array([1.5, 1.5])
    X_neg = np.random.randn(N//2, 2) + np.array([-1.5, -1.5])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(N//2), np.zeros(N//2)])

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    w, b, lr = np.zeros(2), 0.0, 0.1
    for epoch in range(200):
        p_hat = sigmoid(X @ w + b)
        loss = -np.mean(y*np.log(p_hat+1e-7) + (1-y)*np.log(1-p_hat+1e-7))
        err = p_hat - y
        w -= lr * (X.T @ err / N)
        b -= lr * np.mean(err)

    acc = np.mean((sigmoid(X @ w + b) > 0.5) == y)
    print(f"Weights: {w.round(3)}, bias: {b:.3f}")
    print(f"Final loss: {loss:.4f}, accuracy: {acc:.3f}")


if __name__ == "__main__":
    gaussian_mle()
    softmax_ce_gradient()
    logistic_regression()

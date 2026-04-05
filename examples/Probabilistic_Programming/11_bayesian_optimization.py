"""
Bayesian Optimization Examples
- GP surrogate, acquisition functions, optimization loop
"""
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist


def rbf_kernel(X1, X2, l=1.0, sf=1.0):
    return sf**2 * np.exp(-0.5 * cdist(X1.reshape(-1,1), X2.reshape(-1,1), 'sqeuclidean') / l**2)


def expected_improvement(X, X_train, y_train, y_best, l=0.5, sf=2.0, sn=0.01):
    K = rbf_kernel(X_train, X_train, l, sf) + sn * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)
    Ks = rbf_kernel(X, X_train, l, sf)
    mu = Ks @ K_inv @ y_train
    var = sf**2 - np.sum(Ks @ K_inv * Ks, axis=1)
    sigma = np.sqrt(np.clip(var, 1e-10, None))
    Z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)


def bayesian_optimization(objective, bounds, n_init=5, n_iter=15):
    np.random.seed(42)
    X = np.random.uniform(bounds[0], bounds[1], n_init)
    y = np.array([objective(x) for x in X])

    for i in range(n_iter):
        x_cand = np.linspace(bounds[0], bounds[1], 500)
        ei = expected_improvement(x_cand, X, y, y.min())
        x_next = x_cand[np.argmax(ei)]
        y_next = objective(x_next)
        X = np.append(X, x_next)
        y = np.append(y, y_next)
        if (i+1) % 5 == 0:
            print(f"  Iter {i+1}: best={y.min():.4f} at x={X[np.argmin(y)]:.3f}")

    return X[np.argmin(y)], y.min()


if __name__ == "__main__":
    objective = lambda x: (x-2)**2 * np.sin(3*x) + 0.5*x
    print("Bayesian Optimization:")
    x_opt, y_opt = bayesian_optimization(objective, bounds=(-2, 5))
    print(f"  Optimum: x={x_opt:.4f}, f(x)={y_opt:.4f}")

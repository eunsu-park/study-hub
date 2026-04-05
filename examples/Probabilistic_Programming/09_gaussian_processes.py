"""
Gaussian Processes Examples
- GP regression from scratch, kernels, hyperparameter optimization
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def rbf_kernel(X1, X2, l=1.0, var=1.0):
    return var * np.exp(-0.5 * cdist(X1, X2, 'sqeuclidean') / l**2)


class GPRegression:
    def __init__(self, l=1.0, var=1.0, noise=0.1):
        self.l, self.var, self.noise = l, var, noise

    def fit(self, X, y):
        self.X, self.y = X, y
        K = rbf_kernel(X, X, self.l, self.var) + self.noise**2 * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))

    def predict(self, Xs):
        Ks = rbf_kernel(Xs, self.X, self.l, self.var)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = self.var - np.sum(v**2, axis=0)
        return mu, np.sqrt(np.clip(var, 0, None))

    def log_marginal_likelihood(self):
        n = len(self.y)
        return (-0.5 * self.y @ self.alpha
                - np.sum(np.log(np.diag(self.L)))
                - 0.5 * n * np.log(2 * np.pi))


if __name__ == "__main__":
    np.random.seed(42)
    X = np.sort(np.random.uniform(-5, 5, 20)).reshape(-1, 1)
    y = np.sin(X.flatten()) + np.random.normal(0, 0.2, 20)

    gp = GPRegression(l=1.0, var=1.0, noise=0.2)
    gp.fit(X, y)
    Xs = np.linspace(-6, 6, 100).reshape(-1, 1)
    mu, std = gp.predict(Xs)

    print(f"GP Regression: log marginal likelihood = {gp.log_marginal_likelihood():.3f}")
    print(f"Predictions: mu range [{mu.min():.3f}, {mu.max():.3f}]")

    # Optimize hyperparameters
    def neg_lml(log_params):
        l, v, n = np.exp(log_params)
        gp2 = GPRegression(l, v, n)
        gp2.fit(X, y)
        return -gp2.log_marginal_likelihood()

    res = minimize(neg_lml, np.log([1, 1, 0.1]), method='L-BFGS-B')
    opt = np.exp(res.x)
    print(f"Optimal: l={opt[0]:.3f}, var={opt[1]:.3f}, noise={opt[2]:.3f}")

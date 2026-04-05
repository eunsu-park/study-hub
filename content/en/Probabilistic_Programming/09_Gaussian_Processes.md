# 09. Gaussian Processes

[Previous: Variational Inference](./08_Variational_Inference.md) | [Next: Bayesian Time Series](./10_Bayesian_Time_Series.md)

---

> **Framework Note**: This lesson uses NumPy for GP fundamentals and PyMC for inference.
>
> Installation: `pip install pymc arviz numpy scipy matplotlib`

## Learning Objectives

- Understand Gaussian processes as distributions over functions
- Implement GP regression from scratch
- Learn common kernel functions and their properties
- Perform hyperparameter optimization (marginal likelihood)
- Use sparse GP approximations for scalability

---

## 1. What is a Gaussian Process?

A GP defines a distribution over functions. Any finite set of function values is jointly Gaussian.

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x'}))$$

### 1.1 Prior Samples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """Radial Basis Function (Squared Exponential) kernel."""
    dists = cdist(X1, X2, 'sqeuclidean')
    return variance * np.exp(-0.5 * dists / length_scale**2)

# Sample functions from a GP prior
np.random.seed(42)
x_star = np.linspace(-5, 5, 200).reshape(-1, 1)
K = rbf_kernel(x_star, x_star, length_scale=1.0, variance=1.0)
K += 1e-6 * np.eye(len(x_star))  # numerical stability

# Draw samples
L = np.linalg.cholesky(K)
f_samples = L @ np.random.randn(len(x_star), 5)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(5):
    ax.plot(x_star, f_samples[:, i], alpha=0.7)
ax.fill_between(x_star.flatten(),
                -2 * np.sqrt(np.diag(K)),
                2 * np.sqrt(np.diag(K)),
                alpha=0.15, color='gray')
ax.set_title("GP Prior Samples (RBF kernel, l=1.0, σ²=1.0)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plt.tight_layout()
plt.savefig("gp_prior.png", dpi=100)
plt.show()
```

---

## 2. GP Regression

### 2.1 Posterior Derivation

Given training data $(X, y)$ with noise $\sigma_n^2$ and test points $X_*$:

$$f_* | X, y, X_* \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))$$

$$\bar{f}_* = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} y$$

$$\text{cov}(f_*) = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$$

### 2.2 Implementation

```python
class GaussianProcessRegression:
    """GP regression with RBF kernel."""

    def __init__(self, length_scale=1.0, variance=1.0, noise=0.1):
        self.l = length_scale
        self.var = variance
        self.noise = noise

    def fit(self, X, y):
        """Compute posterior given training data."""
        self.X_train = X
        self.y_train = y
        self.K = rbf_kernel(X, X, self.l, self.var) + self.noise**2 * np.eye(len(X))
        self.L = cho_factor(self.K)
        self.alpha = cho_solve(self.L, y)
        return self

    def predict(self, X_star, return_std=True):
        """Predict at test points."""
        K_star = rbf_kernel(X_star, self.X_train, self.l, self.var)
        mu = K_star @ self.alpha

        if return_std:
            K_ss = rbf_kernel(X_star, X_star, self.l, self.var)
            v = cho_solve(self.L, K_star.T)
            cov = K_ss - K_star @ v
            std = np.sqrt(np.diag(cov).clip(0))
            return mu, std
        return mu

    def log_marginal_likelihood(self):
        """Compute log marginal likelihood for hyperparameter optimization."""
        n = len(self.y_train)
        log_det = 2 * np.sum(np.log(np.diag(self.L[0])))
        data_fit = -0.5 * self.y_train @ self.alpha
        complexity = -0.5 * log_det
        const = -0.5 * n * np.log(2 * np.pi)
        return data_fit + complexity + const


# Example
np.random.seed(42)
X_train = np.sort(np.random.uniform(-5, 5, 20)).reshape(-1, 1)
y_train = np.sin(X_train.flatten()) + np.random.normal(0, 0.2, 20)

gp = GaussianProcessRegression(length_scale=1.0, variance=1.0, noise=0.2)
gp.fit(X_train, y_train)

X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
mu, std = gp.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X_train, y_train, c='red', zorder=5, label='Training data')
ax.plot(X_test, mu, 'b-', linewidth=2, label='GP mean')
ax.fill_between(X_test.flatten(), mu - 2*std, mu + 2*std, alpha=0.2, label='±2σ')
ax.plot(X_test, np.sin(X_test), 'k--', alpha=0.5, label='True function')
ax.legend()
ax.set_title("GP Regression")
plt.tight_layout()
plt.savefig("gp_regression.png", dpi=100)
plt.show()
```

---

## 3. Kernel Functions

The kernel (covariance function) encodes our assumptions about the function we're modeling.

### 3.1 Common Kernels

```python
def matern_32(X1, X2, length_scale=1.0, variance=1.0):
    """Matern 3/2 kernel: once differentiable."""
    dists = cdist(X1, X2, 'euclidean') / length_scale
    return variance * (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)

def matern_52(X1, X2, length_scale=1.0, variance=1.0):
    """Matern 5/2 kernel: twice differentiable."""
    dists = cdist(X1, X2, 'euclidean') / length_scale
    return variance * (1 + np.sqrt(5) * dists + 5/3 * dists**2) * np.exp(-np.sqrt(5) * dists)

def periodic_kernel(X1, X2, length_scale=1.0, variance=1.0, period=1.0):
    """Periodic kernel for recurring patterns."""
    dists = cdist(X1, X2, 'euclidean')
    return variance * np.exp(-2 * np.sin(np.pi * dists / period)**2 / length_scale**2)

def rational_quadratic(X1, X2, length_scale=1.0, variance=1.0, alpha=1.0):
    """Rational Quadratic: infinite mixture of RBFs."""
    dists = cdist(X1, X2, 'sqeuclidean')
    return variance * (1 + dists / (2 * alpha * length_scale**2))**(-alpha)

# Visualize kernel properties
x = np.linspace(-5, 5, 200).reshape(-1, 1)
kernels = {
    "RBF": rbf_kernel(x, x),
    "Matern 3/2": matern_32(x, x),
    "Matern 5/2": matern_52(x, x),
    "Periodic": periodic_kernel(x, x, period=2.0),
    "Rational Quadratic": rational_quadratic(x, x, alpha=0.5),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, K) in zip(axes.flat, kernels.items()):
    K += 1e-6 * np.eye(len(x))
    L = np.linalg.cholesky(K)
    for _ in range(3):
        ax.plot(x, L @ np.random.randn(len(x)), alpha=0.7)
    ax.set_title(f"{name}")
    ax.set_xlabel("x")
axes.flat[-1].axis('off')
plt.suptitle("GP Samples from Different Kernels", fontsize=14)
plt.tight_layout()
plt.savefig("gp_kernels.png", dpi=100)
plt.show()
```

### 3.2 Kernel Composition

```python
# Kernels can be combined by addition and multiplication
# k1 + k2: sum of independent processes (OR patterns)
# k1 * k2: interaction (AND patterns)

def composite_kernel(X1, X2):
    """Long-term trend + periodic pattern + noise."""
    k_trend = rbf_kernel(X1, X2, length_scale=10.0, variance=2.0)
    k_periodic = periodic_kernel(X1, X2, length_scale=0.5, period=1.0, variance=1.0)
    k_noise = rbf_kernel(X1, X2, length_scale=0.1, variance=0.5)
    return k_trend + k_periodic + k_noise
```

---

## 4. Hyperparameter Optimization

### 4.1 Maximizing Marginal Likelihood

```python
from scipy.optimize import minimize

def optimize_hyperparameters(X, y, kernel_fn):
    """Optimize kernel hyperparameters by maximizing log marginal likelihood."""

    def neg_log_marginal_likelihood(log_params):
        l, var, noise = np.exp(log_params)
        K = rbf_kernel(X, X, l, var) + noise**2 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * len(y) * np.log(2 * np.pi)
            return nll
        except np.linalg.LinAlgError:
            return 1e10

    result = minimize(neg_log_marginal_likelihood,
                      x0=np.log([1.0, 1.0, 0.1]),
                      method='L-BFGS-B')

    optimal_params = np.exp(result.x)
    print(f"Optimal: l={optimal_params[0]:.3f}, σ²={optimal_params[1]:.3f}, "
          f"σ_n={optimal_params[2]:.3f}")
    return optimal_params

opt_params = optimize_hyperparameters(X_train, y_train, rbf_kernel)
```

---

## 5. GP in PyMC

```python
import pymc as pm

with pm.Model() as gp_model:
    # Hyperpriors
    l = pm.InverseGamma("l", alpha=5, beta=5)
    eta = pm.HalfNormal("eta", sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Kernel
    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=l)

    # GP with noise
    gp = pm.gp.Marginal(cov_func=cov)

    # Likelihood
    y_ = gp.marginal_likelihood("y", X=X_train, y=y_train, sigma=sigma)

    # Sample
    trace_gp = pm.sample(2000, tune=1000, chains=4,
                         target_accept=0.9, random_seed=42)

# Posterior predictions
with gp_model:
    f_pred = gp.conditional("f_pred", X_test)
    pred_samples = pm.sample_posterior_predictive(
        trace_gp, var_names=["f_pred"], random_seed=42
    )

# Plot
f_pred_vals = pred_samples.posterior_predictive["f_pred"].values
mu_pred = f_pred_vals.mean(axis=(0, 1))
std_pred = f_pred_vals.std(axis=(0, 1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X_train, y_train, c='red', zorder=5)
ax.plot(X_test, mu_pred, 'b-', linewidth=2)
ax.fill_between(X_test.flatten(), mu_pred - 2*std_pred, mu_pred + 2*std_pred, alpha=0.2)
ax.set_title("GP Regression with PyMC (Bayesian Hyperparameters)")
plt.tight_layout()
plt.savefig("gp_pymc.png", dpi=100)
plt.show()
```

---

## 6. Sparse Gaussian Processes

Standard GPs have $O(N^3)$ complexity. Sparse methods reduce this to $O(NM^2)$ using $M \ll N$ inducing points.

### 6.1 Sparse Variational GP (SVGP)

```python
class SparseGP:
    """Sparse GP using FITC approximation."""

    def __init__(self, X_inducing, length_scale=1.0, variance=1.0, noise=0.1):
        self.Z = X_inducing  # inducing points
        self.l = length_scale
        self.var = variance
        self.noise = noise

    def fit(self, X, y):
        """Fit sparse GP."""
        self.X_train = X
        self.y_train = y
        M = len(self.Z)
        N = len(X)

        Kuu = rbf_kernel(self.Z, self.Z, self.l, self.var) + 1e-6 * np.eye(M)
        Kuf = rbf_kernel(self.Z, X, self.l, self.var)
        Kff_diag = self.var * np.ones(N)

        # FITC approximation
        Luu = np.linalg.cholesky(Kuu)
        V = np.linalg.solve(Luu, Kuf)
        Qff_diag = np.sum(V**2, axis=0)
        Lambda = np.diag(Kff_diag - Qff_diag + self.noise**2)

        # Effective kernel
        B = np.eye(M) + V @ np.linalg.solve(Lambda, V.T)
        LB = np.linalg.cholesky(B)
        self.Luu = Luu
        self.LB = LB
        self.V = V
        self.Lambda = Lambda
        return self

    def predict(self, X_star):
        """Predict at test points."""
        Kus = rbf_kernel(self.Z, X_star, self.l, self.var)
        Vs = np.linalg.solve(self.Luu, Kus)

        alpha = np.linalg.solve(self.LB, self.V @ np.linalg.solve(self.Lambda, self.y_train))
        mu = Vs.T @ np.linalg.solve(self.LB, alpha)
        return mu

# Example with many data points
np.random.seed(42)
N_large = 1000
X_large = np.sort(np.random.uniform(-5, 5, N_large)).reshape(-1, 1)
y_large = np.sin(X_large.flatten()) + 0.2 * np.random.randn(N_large)

# Choose M=20 inducing points
M = 20
Z = np.linspace(-5, 5, M).reshape(-1, 1)

sgp = SparseGP(Z, length_scale=1.0, variance=1.0, noise=0.2)
sgp.fit(X_large, y_large)
mu_sparse = sgp.predict(X_test)

print(f"Full GP: O({N_large}³) = O({N_large**3:,}) operations")
print(f"Sparse GP: O({N_large}×{M}²) = O({N_large * M**2:,}) operations")
```

---

## 7. GP Classification

```python
# GP classification uses a Bernoulli likelihood with a GP latent function
# P(y=1|x) = σ(f(x)) where f ~ GP and σ is the logistic sigmoid

with pm.Model() as gp_classification:
    # Kernel hyperparameters
    l = pm.InverseGamma("l", alpha=5, beta=5)
    eta = pm.HalfNormal("eta", sigma=2)

    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=l)
    gp = pm.gp.Latent(cov_func=cov)

    # Latent function
    f = gp.prior("f", X=X_train)

    # Bernoulli likelihood with logit link
    y = pm.Bernoulli("y", logit_p=f, observed=y_binary_train)

    trace_gpc = pm.sample(2000, tune=1000, chains=4, random_seed=42)
```

---

## 8. Multi-Output GPs

```python
# When we have multiple correlated output functions
# Use the Intrinsic Coregionalization Model (ICM)

# k((x,i), (x',j)) = k_x(x, x') * B[i,j]
# where B is a positive semidefinite matrix capturing output correlations

def icm_kernel(X1, task1, X2, task2, B, length_scale=1.0, variance=1.0):
    """Intrinsic Coregionalization Model kernel."""
    k_x = rbf_kernel(X1, X2, length_scale, variance)
    k_task = B[np.ix_(task1, task2)]
    return k_x * k_task
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| GP prior | Distribution over functions; specified by mean and kernel |
| GP posterior | Conditioning on data gives exact posterior (with Gaussian likelihood) |
| Kernels | Encode assumptions: smoothness, periodicity, stationarity |
| Hyperparameters | Optimize via marginal likelihood (type-II ML) or full Bayesian |
| Sparse GPs | Inducing points reduce O(N³) to O(NM²) |
| Kernel composition | Sum (OR), product (AND), applied to build expressive kernels |

---

## References

1. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
2. Titsias, M. (2009). "Variational Learning of Inducing Variables in Sparse Gaussian Processes." *AISTATS*.
3. Wilson, A. & Adams, R. (2013). "Gaussian Process Kernels for Pattern Discovery and Extrapolation." *ICML*.
4. Hensman, J., et al. (2015). "Scalable Variational Gaussian Process Classification." *AISTATS*.

---

[Previous: Variational Inference](./08_Variational_Inference.md) | [Next: Bayesian Time Series →](./10_Bayesian_Time_Series.md)

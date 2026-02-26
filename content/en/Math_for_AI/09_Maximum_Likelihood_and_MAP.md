# 09. Maximum Likelihood and MAP

## Learning Objectives

- Understand the difference between likelihood and probability, and grasp the role of log-likelihood
- Define Maximum Likelihood Estimation (MLE) and derive MLE for various probability distributions
- Understand the principles of MAP estimation and explain the differences from MLE
- Mathematically derive how regularization terms connect to prior distributions
- Understand the principles of the EM algorithm and apply it to models with latent variables
- Learn how MLE and MAP are utilized in machine learning through practical examples

---

## 1. Likelihood Function

### 1.1 Likelihood vs Probability

**Probability** and **Likelihood** are computed using the same formula, but have different meanings.

- **Probability**: When parameter $\theta$ is fixed, the probability of data $D$ occurring
  $$P(D|\theta)$$

- **Likelihood**: After data $D$ is observed, a function of parameter $\theta$
  $$\mathcal{L}(\theta|D) = P(D|\theta)$$

**Key difference**:
- Probability: $\theta$ fixed, $D$ variable → $\sum_D P(D|\theta) = 1$
- Likelihood: $D$ fixed, $\theta$ variable → $\int \mathcal{L}(\theta|D) d\theta \neq 1$ (in general)

**Example**: Coin toss
- The **probability** of getting 7 heads in 10 tosses when the coin's head probability is $\theta = 0.7$
- The **likelihood** that $\theta$ is 0.7 when 7 heads are observed in 10 tosses

### 1.2 Log-Likelihood

In practice, we use **log-likelihood**:

$$\ell(\theta|D) = \log \mathcal{L}(\theta|D) = \log P(D|\theta)$$

**Reasons for using log-likelihood**:

1. **Numerical stability**: Products of probabilities become very small, causing underflow
2. **Convert products to sums**: $\log(ab) = \log a + \log b$
3. **Easier differentiation**: Derivatives of exponential functions become simpler
4. **Same optimization**: $\log$ is monotonically increasing, so argmax is identical

### 1.3 i.i.d. Assumption and Product→Sum Conversion

If data $D = \{x_1, x_2, \ldots, x_n\}$ follows **independent and identically distributed (i.i.d.)**:

$$P(D|\theta) = \prod_{i=1}^n P(x_i|\theta)$$

Taking the log:

$$\ell(\theta|D) = \log \prod_{i=1}^n P(x_i|\theta) = \sum_{i=1}^n \log P(x_i|\theta)$$

This transformation is the core of MLE.

## 2. Maximum Likelihood Estimation (MLE)

### 2.1 MLE Definition

**Maximum Likelihood Estimation (MLE)** is the method of finding the parameter that maximizes the likelihood of observed data:

$$\theta_{\text{MLE}} = \arg\max_{\theta} \mathcal{L}(\theta|D) = \arg\max_{\theta} \log P(D|\theta)$$

**Computation method**:
1. Write the likelihood function $\mathcal{L}(\theta|D)$
2. Compute log-likelihood $\ell(\theta|D)$
3. Solve $\frac{\partial \ell}{\partial \theta} = 0$
4. Verify maximum with second derivative

### 2.2 MLE for Normal Distribution

**Problem**: Given $n$ data points $\{x_1, \ldots, x_n\}$ from $\mathcal{N}(\mu, \sigma^2)$, find the MLE of $\mu$ and $\sigma^2$.

**Solution**:

Log-likelihood:
$$\ell(\mu, \sigma^2) = \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

$$= -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2$$

**Optimization w.r.t. $\mu$**:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$

$$\Rightarrow \mu_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i$$

**Optimization w.r.t. $\sigma^2$**:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i-\mu)^2 = 0$$

$$\Rightarrow \sigma^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \mu_{\text{MLE}})^2$$

**Result**: Sample mean and sample variance!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data
np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
n_samples = 100
data = np.random.normal(true_mu, true_sigma, n_samples)

# Compute MLE
mu_mle = np.mean(data)
sigma2_mle = np.var(data, ddof=0)  # ddof=0: MLE (biased)
sigma_mle = np.sqrt(sigma2_mle)

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")

# Visualize log-likelihood function
def log_likelihood(mu, sigma, data):
    n = len(data)
    return -n/2 * np.log(2*np.pi) - n * np.log(sigma) - \
           np.sum((data - mu)**2) / (2 * sigma**2)

# Log-likelihood with respect to μ
mu_range = np.linspace(3, 7, 100)
ll_mu = [log_likelihood(m, true_sigma, data) for m in mu_range]

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(mu_range, ll_mu)
plt.axvline(mu_mle, color='r', linestyle='--', label=f'MLE={mu_mle:.2f}')
plt.xlabel('μ')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs μ')
plt.legend()
plt.grid(True)

# Log-likelihood with respect to σ
sigma_range = np.linspace(0.5, 4, 100)
ll_sigma = [log_likelihood(true_mu, s, data) for s in sigma_range]

plt.subplot(132)
plt.plot(sigma_range, ll_sigma)
plt.axvline(sigma_mle, color='r', linestyle='--', label=f'MLE={sigma_mle:.2f}')
plt.xlabel('σ')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs σ')
plt.legend()
plt.grid(True)

# Histogram and MLE distribution
plt.subplot(133)
plt.hist(data, bins=30, density=True, alpha=0.7, label='Data')
x = np.linspace(data.min(), data.max(), 100)
plt.plot(x, stats.norm.pdf(x, mu_mle, sigma_mle), 'r-',
         linewidth=2, label='MLE fit')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Data vs MLE Distribution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mle_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 MLE for Bernoulli Distribution

**Problem**: In $n$ coin tosses resulting in $k$ heads, what is the MLE of head probability $\theta$?

**Solution**:

$$\mathcal{L}(\theta) = \theta^k (1-\theta)^{n-k}$$

$$\ell(\theta) = k \log \theta + (n-k) \log(1-\theta)$$

$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

$$\Rightarrow \theta_{\text{MLE}} = \frac{k}{n}$$

**Intuition**: The observed relative frequency!

### 2.4 Properties of MLE

1. **Consistency**: As $n \to \infty$, $\theta_{\text{MLE}} \to \theta_{\text{true}}$ (in probability)

2. **Asymptotic Normality**:
   $$\sqrt{n}(\theta_{\text{MLE}} - \theta_{\text{true}}) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})$$
   where $I(\theta)$ is the Fisher information matrix

3. **Unbiasedness not guaranteed**: For example, $\sigma^2_{\text{MLE}}$ is a biased estimator (ddof=0)

4. **Invariance**: If $\theta_{\text{MLE}}$ is the MLE of $\theta$, then $g(\theta_{\text{MLE}})$ is the MLE of $g(\theta)$

## 3. MAP Estimation (Maximum A Posteriori)

### 3.1 Bayes' Theorem and Posterior Distribution

Bayes' theorem:

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

- $P(\theta|D)$: Posterior distribution
- $P(D|\theta)$: Likelihood
- $P(\theta)$: Prior distribution
- $P(D)$: Evidence, normalization constant

### 3.2 MAP Definition

**MAP estimation (Maximum A Posteriori)** finds the parameter that maximizes the posterior distribution:

$$\theta_{\text{MAP}} = \arg\max_{\theta} P(\theta|D)$$

$$= \arg\max_{\theta} P(D|\theta)P(\theta)$$

$$= \arg\max_{\theta} \left[\log P(D|\theta) + \log P(\theta)\right]$$

**Relationship with MLE**:
- MLE: $\arg\max_{\theta} \log P(D|\theta)$
- MAP: $\arg\max_{\theta} \left[\log P(D|\theta) + \log P(\theta)\right]$

MAP = MLE + prior term

### 3.3 MLE vs MAP Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data: 8 heads out of 10 flips
n, k = 10, 8

# Likelihood function
theta_range = np.linspace(0.01, 0.99, 100)
likelihood = stats.binom.pmf(k, n, theta_range)

# MLE
theta_mle = k / n

# MAP with Beta prior (α, β)
alpha, beta = 2, 2  # Weak prior
prior = stats.beta.pdf(theta_range, alpha, beta)
posterior = likelihood * prior
posterior = posterior / np.trapz(posterior, theta_range)  # Normalize
theta_map = theta_range[np.argmax(posterior)]

plt.figure(figsize=(14, 4))

# Likelihood
plt.subplot(131)
plt.plot(theta_range, likelihood / np.max(likelihood), label='Likelihood')
plt.axvline(theta_mle, color='r', linestyle='--', label=f'MLE={theta_mle:.2f}')
plt.xlabel('θ')
plt.ylabel('Normalized Likelihood')
plt.title('Likelihood Function')
plt.legend()
plt.grid(True)

# Prior distribution
plt.subplot(132)
plt.plot(theta_range, prior, label=f'Beta({alpha},{beta})')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Prior Distribution')
plt.legend()
plt.grid(True)

# Posterior distribution
plt.subplot(133)
plt.plot(theta_range, posterior, label='Posterior', linewidth=2)
plt.axvline(theta_mle, color='r', linestyle='--', alpha=0.7, label=f'MLE={theta_mle:.2f}')
plt.axvline(theta_map, color='g', linestyle='--', alpha=0.7, label=f'MAP={theta_map:.2f}')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Posterior Distribution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mle_vs_map.png', dpi=150, bbox_inches='tight')
plt.show()

# Difference with small data
print("With small data (n=10, k=8):")
print(f"MLE: {theta_mle:.3f}")
print(f"MAP: {theta_map:.3f}")

# With large data
n2, k2 = 1000, 800
theta_mle2 = k2 / n2
likelihood2 = stats.binom.pmf(k2, n2, theta_range)
posterior2 = likelihood2 * prior
theta_map2 = theta_range[np.argmax(posterior2)]

print("\nWith large data (n=1000, k=800):")
print(f"MLE: {theta_mle2:.3f}")
print(f"MAP: {theta_map2:.3f}")
print("→ With more data, MLE and MAP converge")
```

### 3.4 Role of Prior Distribution

- **With little data**: Prior has significant influence (regularization effect)
- **With much data**: Likelihood dominates, MAP ≈ MLE
- **Appropriate prior**: Reflects domain knowledge, prevents overfitting
- **Uninformative prior**: $P(\theta) \propto 1$ → MAP = MLE

## 4. Regularization = Prior Distribution

### 4.1 L2 Regularization and Gaussian Prior

**L2 regularization in linear regression** (Ridge):

$$\min_w \left[\frac{1}{2}\sum_{i=1}^n (y_i - w^T x_i)^2 + \frac{\lambda}{2}\|w\|^2\right]$$

This is equivalent to the following MAP:

**Prior**: $w \sim \mathcal{N}(0, \sigma_w^2 I)$

$$\log P(w) = -\frac{1}{2\sigma_w^2}\|w\|^2 + \text{const}$$

**Likelihood** (Gaussian noise): $y|x,w \sim \mathcal{N}(w^T x, \sigma^2)$

$$\log P(D|w) = -\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - w^T x_i)^2 + \text{const}$$

**Log posterior**:

$$\log P(w|D) = \log P(D|w) + \log P(w)$$

$$= -\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - w^T x_i)^2 - \frac{1}{2\sigma_w^2}\|w\|^2 + \text{const}$$

**Maximization = Minimization**:

$$\arg\max_w \log P(w|D) = \arg\min_w \left[\sum_{i=1}^n (y_i - w^T x_i)^2 + \frac{\sigma^2}{\sigma_w^2}\|w\|^2\right]$$

Therefore $\lambda = \frac{\sigma^2}{\sigma_w^2}$

**Conclusion**: L2 regularization = MAP with Gaussian prior

### 4.2 L1 Regularization and Laplace Prior

**Lasso regression**:

$$\min_w \left[\frac{1}{2}\sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda\|w\|_1\right]$$

**Laplace prior**: $P(w_j) = \frac{1}{2b}\exp\left(-\frac{|w_j|}{b}\right)$

$$\log P(w) = -\frac{1}{b}\|w\|_1 + \text{const}$$

Therefore $\lambda = \frac{\sigma^2}{b}$

**Conclusion**: L1 regularization = MAP with Laplace prior

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Visualize prior distributions
w_range = np.linspace(-3, 3, 1000)

# Gaussian prior (L2)
sigma_w = 1.0
gaussian_prior = stats.norm.pdf(w_range, 0, sigma_w)
log_gaussian = stats.norm.logpdf(w_range, 0, sigma_w)

# Laplace prior (L1)
b = 1.0
laplace_prior = stats.laplace.pdf(w_range, 0, b)
log_laplace = stats.laplace.logpdf(w_range, 0, b)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(w_range, gaussian_prior, label='Gaussian (L2)', linewidth=2)
plt.plot(w_range, laplace_prior, label='Laplace (L1)', linewidth=2)
plt.xlabel('w')
plt.ylabel('Density')
plt.title('Prior Distributions')
plt.legend()
plt.grid(True)

plt.subplot(132)
plt.plot(w_range, log_gaussian, label='log Gaussian', linewidth=2)
plt.plot(w_range, log_laplace, label='log Laplace', linewidth=2)
plt.xlabel('w')
plt.ylabel('Log Density')
plt.title('Log Prior (Regularization Term)')
plt.legend()
plt.grid(True)

plt.subplot(133)
plt.plot(w_range, -w_range**2 / (2*sigma_w**2),
         label=r'$-\frac{w^2}{2\sigma_w^2}$ (L2)', linewidth=2)
plt.plot(w_range, -np.abs(w_range) / b,
         label=r'$-\frac{|w|}{b}$ (L1)', linewidth=2)
plt.xlabel('w')
plt.ylabel('Regularization Penalty')
plt.title('Penalty Terms')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('regularization_prior.png', dpi=150, bbox_inches='tight')
plt.show()

print("L2 (Gaussian) applies a quadratic penalty to large weights")
print("L1 (Laplace) applies a linear penalty to all weights → sparsity")
```

## 5. EM Algorithm (Expectation-Maximization)

### 5.1 Latent Variable Problem

With **observed variable** $X$ and **latent variable** $Z$:

$$P(X|\theta) = \sum_Z P(X, Z|\theta)$$

Direct maximization is difficult (log of sum).

**EM algorithm** iteratively maximizes a lower bound.

### 5.2 ELBO and EM Derivation

Using Jensen's inequality:

$$\log P(X|\theta) = \log \sum_Z P(X, Z|\theta)$$

$$= \log \sum_Z Q(Z) \frac{P(X, Z|\theta)}{Q(Z)}$$

$$\geq \sum_Z Q(Z) \log \frac{P(X, Z|\theta)}{Q(Z)}$$

$$= \mathbb{E}_{Q(Z)} [\log P(X, Z|\theta)] + H(Q)$$

This is called the **ELBO** (Evidence Lower BOund).

**E-step**: Optimize $Q(Z)$ for current $\theta^{(t)}$
$$Q^{(t+1)}(Z) = P(Z|X, \theta^{(t)})$$

**M-step**: Optimize $\theta$ for current $Q^{(t+1)}$
$$\theta^{(t+1)} = \arg\max_{\theta} \mathbb{E}_{Q^{(t+1)}(Z)} [\log P(X, Z|\theta)]$$

### 5.3 Gaussian Mixture Model (GMM)

**Model**:
- $K$ Gaussian components
- Latent variable $z_i \in \{1, \ldots, K\}$: component of data $x_i$
- Parameters: $\pi_k$ (mixing proportions), $\mu_k, \Sigma_k$ (mean, covariance of each Gaussian)

**Generative process**:
1. $z_i \sim \text{Categorical}(\pi)$
2. $x_i | z_i=k \sim \mathcal{N}(\mu_k, \Sigma_k)$

**E-step**: Compute responsibility

$$\gamma_{ik} = P(z_i=k | x_i, \theta^{(t)}) = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}$$

**M-step**: Update parameters

$$\pi_k^{\text{new}} = \frac{1}{n}\sum_{i=1}^n \gamma_{ik}$$

$$\mu_k^{\text{new}} = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}$$

$$\Sigma_k^{\text{new}} = \frac{\sum_{i=1}^n \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T}{\sum_{i=1}^n \gamma_{ik}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class GMM:
    def __init__(self, K, max_iter=100, tol=1e-4):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, d = X.shape

        # Initialize: k-means++ style
        self.pi = np.ones(self.K) / self.K
        # Random initialization
        idx = np.random.choice(n, self.K, replace=False)
        self.mu = X[idx]
        self.Sigma = np.array([np.eye(d) for _ in range(self.K)])

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            gamma = self._e_step(X)

            # M-step
            self._m_step(X, gamma)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.2f}")

            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            log_likelihood_old = log_likelihood

        return self

    def _e_step(self, X):
        n = X.shape[0]
        gamma = np.zeros((n, self.K))

        for k in range(self.K):
            gamma[:, k] = self.pi[k] * stats.multivariate_normal.pdf(
                X, self.mu[k], self.Sigma[k])

        # Normalize
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, X, gamma):
        n = X.shape[0]

        # Effective number of points assigned to each cluster
        Nk = gamma.sum(axis=0)

        # Update mixing proportions
        self.pi = Nk / n

        # Update means
        self.mu = (gamma.T @ X) / Nk[:, np.newaxis]

        # Update covariances
        for k in range(self.K):
            diff = X - self.mu[k]
            self.Sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / Nk[k]
            # Numerical stability
            self.Sigma[k] += 1e-6 * np.eye(X.shape[1])

    def _compute_log_likelihood(self, X):
        n = X.shape[0]
        log_likelihood = 0

        for i in range(n):
            prob = 0
            for k in range(self.K):
                prob += self.pi[k] * stats.multivariate_normal.pdf(
                    X[i], self.mu[k], self.Sigma[k])
            log_likelihood += np.log(prob + 1e-10)

        return log_likelihood

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

# Test: 3 Gaussian mixture
np.random.seed(42)
n_samples = 300

# Generate 3 clusters
X1 = np.random.randn(n_samples, 2) + np.array([0, 0])
X2 = np.random.randn(n_samples, 2) + np.array([5, 5])
X3 = np.random.randn(n_samples, 2) + np.array([0, 5])
X = np.vstack([X1, X2, X3])

# Train GMM
gmm = GMM(K=3, max_iter=100)
gmm.fit(X)

# Predict
labels = gmm.predict(X)

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Original Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
for k in range(gmm.K):
    plt.scatter(gmm.mu[k, 0], gmm.mu[k, 1],
                marker='x', s=200, linewidths=3, color='red')
plt.title('GMM Clustering')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

# Decision boundaries
plt.subplot(133)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
            alpha=0.7, edgecolors='k')
plt.title('Decision Boundaries')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

plt.tight_layout()
plt.savefig('gmm_em.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 6. Machine Learning Applications

### 6.1 Logistic Regression = Bernoulli MLE

**Model**: $P(y=1|x,w) = \sigma(w^T x)$, where $\sigma(z) = \frac{1}{1+e^{-z}}$

**Log-likelihood**:

$$\ell(w) = \sum_{i=1}^n \left[y_i \log \sigma(w^T x_i) + (1-y_i) \log(1-\sigma(w^T x_i))\right]$$

This is the log-likelihood of the Bernoulli distribution!

**Optimization**: Gradient descent

$$\nabla_w \ell = \sum_{i=1}^n (y_i - \sigma(w^T x_i)) x_i$$

### 6.2 Neural Network Training = MLE

The neural network output for **classification problems** is interpreted as softmax:

$$P(y=k|x,\theta) = \frac{\exp(f_k(x;\theta))}{\sum_j \exp(f_j(x;\theta))}$$

**Cross-entropy loss** = **Negative log-likelihood**:

$$\mathcal{L}(\theta) = -\sum_{i=1}^n \log P(y_i|x_i, \theta)$$

Therefore, neural network training = MLE!

### 6.3 Introduction to Bayesian Neural Networks

**Problem**: Neural networks only provide point estimates, not uncertainty

**Bayesian neural networks**: Infer posterior distribution over weights

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

**Prediction**: Integration over posterior

$$P(y|x, D) = \int P(y|x, \theta) P(\theta|D) d\theta$$

**Challenge**: High-dimensional integration → requires variational inference, MCMC

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Simple logistic regression example
np.random.seed(42)
torch.manual_seed(42)

# Generate data
n = 200
X = np.random.randn(n, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Logistic regression model
model = nn.Linear(2, 1)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy = Bernoulli MLE
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_tensor).squeeze()
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (Negative Log-Likelihood)')
plt.title('Training Loss = -Log-Likelihood')
plt.grid(True)

plt.subplot(132)
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', alpha=0.6)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data')
plt.legend()
plt.grid(True)

# Decision boundary
plt.subplot(133)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
with torch.no_grad():
    Z = torch.sigmoid(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))).numpy()
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, levels=20)
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', alpha=0.7, edgecolors='k')
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', alpha=0.7, edgecolors='k')
plt.colorbar(label='P(y=1|x)')
plt.title('MLE Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('logistic_mle.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Learned weights: {model.weight.data.numpy()}")
print(f"Learned bias: {model.bias.data.numpy()}")
```

## Practice Problems

### Problem 1: MLE for Exponential Distribution

The PDF of the exponential distribution $\text{Exp}(\lambda)$ is $p(x|\lambda) = \lambda e^{-\lambda x}$ (for $x \geq 0$).

Given $n$ i.i.d. samples $\{x_1, \ldots, x_n\}$:

(a) Derive the log-likelihood function.
(b) Find the MLE of $\lambda$.
(c) Verify with Python (generate simulation data and compute MLE).

### Problem 2: MAP with Different Priors

In a linear regression problem:
- Data: $(x_1, y_1), \ldots, (x_n, y_n)$
- Model: $y = wx + b + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$

Compare MAP for the following two cases:

(a) Gaussian prior: $w \sim \mathcal{N}(0, \sigma_w^2)$ → L2 regularization
(b) Laplace prior: $p(w) \propto \exp(-|w|/b)$ → L1 regularization

Write the optimization problems for each and implement in Python to visualize the differences.

### Problem 3: EM for Coin Toss

Two coins A and B have head probabilities $\theta_A, \theta_B$ respectively.

Experiment:
- In 5 trials, each trial randomly selects a coin (unobservable)
- Results: {H, T, T, H, H}

Use EM algorithm to estimate $\theta_A, \theta_B$:

(a) E-step: Compute probability of coin A for each trial
(b) M-step: Update $\theta_A, \theta_B$
(c) Implement in Python and visualize convergence process

### Problem 4: Fisher Information

Fisher information is defined as:

$$I(\theta) = -\mathbb{E}\left[\frac{\partial^2 \log p(X|\theta)}{\partial \theta^2}\right]$$

For Bernoulli distribution $p(x|\theta) = \theta^x (1-\theta)^{1-x}$:

(a) Compute Fisher information $I(\theta)$.
(b) Verify the Cramér-Rao lower bound $\text{Var}(\hat{\theta}) \geq \frac{1}{nI(\theta)}$.
(c) Show that the variance of MLE $\hat{\theta} = \frac{k}{n}$ reaches this lower bound.

### Problem 5: EM for Mixture of Exponentials

Mixture model of exponential distributions:

$$p(x) = \pi \lambda_1 e^{-\lambda_1 x} + (1-\pi) \lambda_2 e^{-\lambda_2 x}$$

Derive EM algorithm and implement in Python:

(a) Derive E-step responsibility formula
(b) Derive M-step update formulas for $\pi, \lambda_1, \lambda_2$
(c) Verify with simulation data (recover true parameters from known ground truth)

## References

1. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Chapter 9 (EM Algorithm).
2. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: An Introduction*. Chapter 8 (MLE), Chapter 10 (MAP).
3. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Chapter 8.
4. **MacKay, D. J. C. (2003).** *Information Theory, Inference, and Learning Algorithms*. Chapter 22 (EM).
5. **Paper**: Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm". *Journal of the Royal Statistical Society*.
6. **scikit-learn documentation**: Gaussian Mixture Models - https://scikit-learn.org/stable/modules/mixture.html
7. **PyTorch documentation**: Loss Functions - https://pytorch.org/docs/stable/nn.html#loss-functions

# Lesson 6: Probability Distributions for Deep Learning

## Learning Objectives

- Review the fundamentals of probability: random variables, PMFs, PDFs, expectation, and variance
- Describe the Bernoulli, categorical, and Gaussian distributions and their parameterizations
- Connect probability distributions to neural network output layers and loss functions
- Derive the negative log-likelihood loss from a probabilistic perspective
- Understand the reparameterization trick for backpropagation through stochastic nodes
- Recognize mixture models and their role in generative modeling
- Compute KL divergence between simple distributions analytically

---

## 1. Probability Review

### 1.1 Random Variables and Distributions

A **random variable** $X$ maps outcomes of a random experiment to real numbers. Its distribution describes the probability of each possible value.

**Discrete**: Probability mass function (PMF) $P(X = x) = p(x)$, with $\sum_x p(x) = 1$.

**Continuous**: Probability density function (PDF) $p(x)$, with $\int_{-\infty}^{\infty} p(x) \, dx = 1$.

### 1.2 Expectation and Variance

$$\mathbb{E}[X] = \sum_x x \, p(x) \quad \text{or} \quad \int x \, p(x) \, dx$$

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

### 1.3 Joint and Conditional Distributions

**Joint**: $p(x, y) = p(x | y) p(y)$ (chain rule of probability)

**Bayes' theorem**: $p(y | x) = \frac{p(x | y) p(y)}{p(x)}$

**Marginalization**: $p(x) = \sum_y p(x, y)$ or $\int p(x, y) \, dy$

---

## 2. The Bernoulli Distribution

### 2.1 Definition

A Bernoulli random variable $X \in \{0, 1\}$ with parameter $p \in [0, 1]$:

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

Compactly: $P(X = x) = p^x (1 - p)^{1-x}$

**Properties**: $\mathbb{E}[X] = p$, $\text{Var}(X) = p(1 - p)$.

### 2.2 Bernoulli in Deep Learning

Binary classification: the model outputs $\hat{p} = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ using sigmoid, modeling $P(y = 1 | \mathbf{x})$.

The **negative log-likelihood** of a Bernoulli observation is:

$$-\log P(y | \hat{p}) = -[y \log \hat{p} + (1 - y) \log(1 - \hat{p})]$$

This is exactly the **binary cross-entropy** loss.

```python
import numpy as np
import matplotlib.pyplot as plt

def binary_cross_entropy(y, p_hat):
    """BCE loss for a single sample."""
    eps = 1e-7  # For numerical stability
    p_hat = np.clip(p_hat, eps, 1 - eps)
    return -(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

# Visualize BCE loss for y=1 and y=0
p = np.linspace(0.01, 0.99, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(p, binary_cross_entropy(1, p), 'b-', linewidth=2)
axes[0].set_xlabel('$\hat{p}$')
axes[0].set_ylabel('Loss')
axes[0].set_title('BCE when $y = 1$: $-\log(\hat{p})$')
axes[0].grid(True, alpha=0.3)

axes[1].plot(p, binary_cross_entropy(0, p), 'r-', linewidth=2)
axes[1].set_xlabel('$\hat{p}$')
axes[1].set_ylabel('Loss')
axes[1].set_title('BCE when $y = 0$: $-\log(1 - \hat{p})$')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 3. The Categorical Distribution

### 3.1 Definition

A categorical random variable $X \in \{1, 2, \ldots, K\}$ with parameters $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$ where $\pi_k \geq 0$ and $\sum_k \pi_k = 1$:

$$P(X = k) = \pi_k$$

Using a one-hot vector $\mathbf{y}$ (where $y_k = 1$ and $y_j = 0$ for $j \neq k$):

$$P(\mathbf{y} | \boldsymbol{\pi}) = \prod_{k=1}^{K} \pi_k^{y_k}$$

### 3.2 Categorical in Deep Learning

Multiclass classification: the model outputs $\hat{\boldsymbol{\pi}} = \text{softmax}(\mathbf{z})$ where $\mathbf{z}$ are the logits.

The negative log-likelihood is:

$$-\log P(\mathbf{y} | \hat{\boldsymbol{\pi}}) = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k$$

Since only one $y_k = 1$, this simplifies to $-\log \hat{\pi}_c$ where $c$ is the true class. This is the **categorical cross-entropy** loss.

```python
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

def categorical_cross_entropy(y_onehot, logits):
    """Cross-entropy loss from logits."""
    probs = softmax(logits)
    return -np.sum(y_onehot * np.log(probs + 1e-7))

# Example: 5-class classification
K = 5
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.0])
true_class = 4  # 0-indexed
y = np.zeros(K)
y[true_class] = 1.0

probs = softmax(logits)
loss = categorical_cross_entropy(y, logits)

print(f"Logits: {logits}")
print(f"Probabilities: {probs.round(4)}")
print(f"True class: {true_class}, P(true) = {probs[true_class]:.4f}")
print(f"Cross-entropy loss: {loss:.4f}")
print(f"Equivalent: -log(P(true)) = {-np.log(probs[true_class]):.4f}")
```

---

## 4. The Gaussian (Normal) Distribution

### 4.1 Univariate Gaussian

$$p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Properties**: $\mathbb{E}[X] = \mu$, $\text{Var}(X) = \sigma^2$.

### 4.2 Multivariate Gaussian

$$p(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

The exponent $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ is the **Mahalanobis distance** from $\mathbf{x}$ to $\boldsymbol{\mu}$.

### 4.3 Gaussian in Deep Learning

**Regression**: If we model $y | \mathbf{x} \sim \mathcal{N}(f_\theta(\mathbf{x}), \sigma^2)$, the negative log-likelihood is:

$$-\log p(y | \mathbf{x}) = \frac{(y - f_\theta(\mathbf{x}))^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)$$

Dropping constants and setting $\sigma = 1$: this is the **MSE loss** (up to a factor of 2).

> **Key insight**: MSE loss implicitly assumes Gaussian noise on the targets.

**VAE latent space**: The prior is $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and the encoder outputs $q(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$.

```python
# Visualize different Gaussians
x = np.linspace(-5, 8, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Univariate
for mu, sigma in [(0, 1), (2, 0.5), (-1, 2)]:
    pdf = 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))
    axes[0].plot(x, pdf, linewidth=2, label=f'μ={mu}, σ={sigma}')
axes[0].set_title('Univariate Gaussian')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bivariate contours
from matplotlib.patches import Ellipse

ax = axes[1]
mu = np.array([0, 0])
covariances = [
    (np.eye(2), 'Isotropic'),
    (np.array([[2, 0.8], [0.8, 0.5]]), 'Correlated'),
    (np.array([[0.5, -0.3], [-0.3, 2]]), 'Anti-corr'),
]

colors = ['blue', 'red', 'green']
for (cov, name), color in zip(covariances, colors):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    for n_std in [1, 2]:
        ell = Ellipse(mu, 2*n_std*np.sqrt(eigvals[0]), 2*n_std*np.sqrt(eigvals[1]),
                      angle=angle, fill=False, color=color, linewidth=2 if n_std==1 else 1)
        ax.add_patch(ell)
    ax.plot([], [], color=color, label=name)

ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_title('Bivariate Gaussian contours')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5. Distribution-Loss Connection

### 5.1 The Unifying Principle

Every standard loss function in deep learning corresponds to the negative log-likelihood of a specific probability distribution:

| Output Type | Distribution | Loss Function | Output Activation |
|-------------|-------------|--------------|-------------------|
| Continuous | Gaussian | MSE | None (linear) |
| Binary | Bernoulli | Binary cross-entropy | Sigmoid |
| Multiclass | Categorical | Cross-entropy | Softmax |
| Positive continuous | Laplacian | MAE (L1) | None |
| Count | Poisson | Poisson NLL | Exp |

### 5.2 Deriving BCE from Bernoulli MLE

Given data $\{(x_i, y_i)\}_{i=1}^N$ where $y_i \in \{0, 1\}$:

$$\text{Likelihood} = \prod_{i=1}^N p_i^{y_i} (1 - p_i)^{1-y_i}$$

$$\text{Log-likelihood} = \sum_{i=1}^N [y_i \log p_i + (1 - y_i) \log(1 - p_i)]$$

$$\text{NLL (loss)} = -\frac{1}{N}\sum_{i=1}^N [y_i \log p_i + (1 - y_i) \log(1 - p_i)]$$

This is exactly the average binary cross-entropy.

### 5.3 Deriving MSE from Gaussian MLE

Assuming $y_i \sim \mathcal{N}(\hat{y}_i, \sigma^2)$:

$$\text{Log-likelihood} = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

Maximizing over $\hat{y}_i$ (treating $\sigma$ as fixed) is equivalent to minimizing:

$$\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2 = \text{MSE}$$

---

## 6. The Reparameterization Trick

### 6.1 The Problem

In variational autoencoders (VAEs), we need to backpropagate through a stochastic sampling step:

$$\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$$

But sampling is not differentiable. We cannot compute $\frac{\partial L}{\partial \boldsymbol{\mu}}$ through the sampling operation.

### 6.2 The Solution

Reparameterize: express $\mathbf{z}$ as a deterministic function of $\boldsymbol{\mu}$, $\boldsymbol{\sigma}$, and noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$

Now $\mathbf{z}$ is differentiable w.r.t. $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$:

$$\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} = \mathbf{I}, \quad \frac{\partial \mathbf{z}}{\partial \boldsymbol{\sigma}} = \text{diag}(\boldsymbol{\epsilon})$$

```python
# Reparameterization trick
np.random.seed(42)

# Encoder outputs
mu = np.array([1.0, -0.5])
log_sigma = np.array([0.5, -0.3])  # log(sigma) for numerical stability
sigma = np.exp(log_sigma)

# Sample using reparameterization
n_samples = 1000
epsilon = np.random.randn(n_samples, 2)
z = mu + sigma * epsilon  # (n_samples, 2)

print(f"mu = {mu}")
print(f"sigma = {sigma}")
print(f"Sample mean: {z.mean(axis=0).round(3)}")
print(f"Sample std:  {z.std(axis=0).round(3)}")

# Visualize
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(z[:, 0], z[:, 1], alpha=0.3, s=5)
ax.plot(mu[0], mu[1], 'r*', markersize=15, label='μ')
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_title('Reparameterized samples')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 7. KL Divergence Between Gaussians

### 7.1 Definition

The Kullback-Leibler divergence from $q$ to $p$:

$$D_\text{KL}(q \| p) = \mathbb{E}_q\left[\log \frac{q(x)}{p(x)}\right]$$

### 7.2 KL Between Two Univariate Gaussians

For $q = \mathcal{N}(\mu_1, \sigma_1^2)$ and $p = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$D_\text{KL}(q \| p) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### 7.3 KL from Gaussian to Standard Normal

For $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$:

$$D_\text{KL}(q \| p) = -\frac{1}{2}\left(1 + \log \sigma^2 - \mu^2 - \sigma^2\right)$$

For multivariate diagonal Gaussian $q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$D_\text{KL}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

This is the KL term in the VAE loss (ELBO).

```python
def kl_gaussian_standard(mu, log_sigma):
    """KL divergence from N(mu, sigma^2) to N(0, 1)."""
    sigma_sq = np.exp(2 * log_sigma)
    return -0.5 * np.sum(1 + 2*log_sigma - mu**2 - sigma_sq)

# Verify by Monte Carlo estimation
mu = np.array([1.0, -0.5])
log_sigma = np.array([0.5, -0.3])
sigma = np.exp(log_sigma)

# Analytical
kl_analytical = kl_gaussian_standard(mu, log_sigma)

# Monte Carlo
n_mc = 100000
samples = mu + sigma * np.random.randn(n_mc, 2)
log_q = -0.5 * np.sum((samples - mu)**2 / sigma**2, axis=1) - np.sum(log_sigma) - len(mu)/2 * np.log(2*np.pi)
log_p = -0.5 * np.sum(samples**2, axis=1) - len(mu)/2 * np.log(2*np.pi)
kl_mc = np.mean(log_q - log_p)

print(f"KL (analytical): {kl_analytical:.4f}")
print(f"KL (Monte Carlo): {kl_mc:.4f}")
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Bernoulli | Models binary outcomes; NLL = binary cross-entropy |
| Categorical | Models multiclass; NLL = categorical cross-entropy |
| Gaussian | Models continuous targets; NLL $\propto$ MSE |
| Distribution-loss link | Every standard loss = NLL of some distribution |
| Reparameterization | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$; enables backprop through sampling |
| KL divergence | Measures distribution mismatch; closed-form for Gaussians |

---

## Exercises

1. Derive the gradient of binary cross-entropy w.r.t. the logit $z$ (before sigmoid), showing it simplifies to $\hat{p} - y$.
2. Show that the Bernoulli MLE for $p$ given i.i.d. samples is the sample mean $\bar{y}$.
3. Implement a function that computes the KL divergence between two arbitrary univariate Gaussians and verify it with Monte Carlo.
4. Derive the gradient of the VAE KL term $D_\text{KL}(q \| p)$ w.r.t. $\mu$ and $\log \sigma$.
5. Compare MSE and MAE losses by deriving which probability distributions they correspond to.

---

**Next**: [07. Maximum Likelihood Estimation](07_Maximum_Likelihood_Estimation.md)

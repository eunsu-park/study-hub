# Lesson 7: Maximum Likelihood Estimation

## Learning Objectives

- Define the likelihood and log-likelihood functions for a parametric model
- Derive MLE estimators for Bernoulli, Gaussian, and categorical distributions
- Connect MLE to minimizing loss functions used in deep learning
- Understand the relationship between MLE and cross-entropy loss
- Derive the gradient of the softmax cross-entropy loss w.r.t. logits
- Explain regularization as maximum a posteriori (MAP) estimation
- Understand the bias-variance tradeoff from a probabilistic perspective
- Implement MLE for a simple logistic regression model from scratch

---

## 1. The Likelihood Function

### 1.1 Setup

Given:
- A parametric model $p(\mathbf{x} | \boldsymbol{\theta})$ with parameters $\boldsymbol{\theta}$
- Observed data $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$, assumed i.i.d.

The **likelihood function** is the probability of observing the data as a function of the parameters:

$$\mathcal{L}(\boldsymbol{\theta}) = p(\mathcal{D} | \boldsymbol{\theta}) = \prod_{i=1}^{N} p(\mathbf{x}_i | \boldsymbol{\theta})$$

### 1.2 Log-Likelihood

Products of many small probabilities underflow to zero. The **log-likelihood** converts products to sums:

$$\ell(\boldsymbol{\theta}) = \log \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \log p(\mathbf{x}_i | \boldsymbol{\theta})$$

Since $\log$ is monotonically increasing, maximizing $\ell$ is equivalent to maximizing $\mathcal{L}$.

### 1.3 MLE Objective

The **maximum likelihood estimator** (MLE) is:

$$\hat{\boldsymbol{\theta}}_\text{MLE} = \arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}) = \arg\min_{\boldsymbol{\theta}} \left[-\frac{1}{N}\sum_{i=1}^{N} \log p(\mathbf{x}_i | \boldsymbol{\theta})\right]$$

The right-hand side is the **negative log-likelihood (NLL)** -- this is the loss function we minimize.

---

## 2. MLE for Common Distributions

### 2.1 Bernoulli MLE

Data: $y_1, \ldots, y_N \in \{0, 1\}$, model: $P(y = 1) = p$.

$$\ell(p) = \sum_{i=1}^{N} [y_i \log p + (1 - y_i) \log(1 - p)]$$

Setting $\frac{d\ell}{dp} = 0$:

$$\frac{d\ell}{dp} = \sum_{i=1}^{N} \left[\frac{y_i}{p} - \frac{1 - y_i}{1 - p}\right] = 0$$

$$\frac{\sum y_i}{p} = \frac{N - \sum y_i}{1 - p}$$

$$\hat{p}_\text{MLE} = \frac{1}{N}\sum_{i=1}^{N} y_i = \bar{y}$$

The MLE for a Bernoulli is simply the sample mean.

### 2.2 Gaussian MLE

Data: $x_1, \ldots, x_N \in \mathbb{R}$, model: $p(x | \mu, \sigma^2) = \mathcal{N}(\mu, \sigma^2)$.

$$\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi) - \frac{N}{2}\log \sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x_i - \mu)^2$$

Setting partial derivatives to zero:

$$\hat{\mu}_\text{MLE} = \frac{1}{N}\sum_{i=1}^{N} x_i = \bar{x}$$

$$\hat{\sigma}^2_\text{MLE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2$$

Note: the MLE for variance uses $N$ (not $N-1$), so it is **biased**. The unbiased estimator uses $N - 1$.

### 2.3 Categorical MLE

Data: $N$ observations with counts $n_1, \ldots, n_K$ (where $\sum n_k = N$), model: $P(X = k) = \pi_k$.

Using Lagrange multipliers with the constraint $\sum \pi_k = 1$:

$$\hat{\pi}_k = \frac{n_k}{N}$$

The MLE is the relative frequency.

```python
import numpy as np
import matplotlib.pyplot as plt

# MLE for Gaussian: demonstrate on synthetic data
np.random.seed(42)
true_mu, true_sigma = 3.0, 1.5
N = 50
data = np.random.normal(true_mu, true_sigma, N)

# MLE estimates
mu_mle = np.mean(data)
sigma_mle = np.sqrt(np.mean((data - mu_mle)**2))

# Visualize
x = np.linspace(-2, 8, 500)
pdf_true = 1/(true_sigma*np.sqrt(2*np.pi)) * np.exp(-(x-true_mu)**2/(2*true_sigma**2))
pdf_mle = 1/(sigma_mle*np.sqrt(2*np.pi)) * np.exp(-(x-mu_mle)**2/(2*sigma_mle**2))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(data, bins=15, density=True, alpha=0.5, label='Data')
ax.plot(x, pdf_true, 'g-', linewidth=2, label=f'True: μ={true_mu}, σ={true_sigma}')
ax.plot(x, pdf_mle, 'r--', linewidth=2, label=f'MLE: μ={mu_mle:.2f}, σ={sigma_mle:.2f}')
ax.legend()
ax.set_title(f'Gaussian MLE (N={N})')
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 3. MLE and Deep Learning Loss Functions

### 3.1 The Central Connection

Training a neural network is performing MLE. The model defines a conditional distribution $p(y | \mathbf{x}; \boldsymbol{\theta})$, and we minimize the NLL:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \left[-\frac{1}{N}\sum_{i=1}^{N} \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})\right]$$

### 3.2 Mapping Table

| Task | Model $p(y|\mathbf{x};\theta)$ | NLL Loss |
|------|-------------------------------|----------|
| Regression | $\mathcal{N}(f_\theta(\mathbf{x}), \sigma^2)$ | $\frac{1}{2\sigma^2}\|y - f_\theta(\mathbf{x})\|^2 + \text{const}$ = MSE |
| Binary classification | $\text{Bernoulli}(\sigma(f_\theta(\mathbf{x})))$ | $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ = BCE |
| Multiclass | $\text{Cat}(\text{softmax}(f_\theta(\mathbf{x})))$ | $-\sum_k y_k \log \hat{\pi}_k$ = CE |

### 3.3 Why Log-Likelihood, Not Likelihood?

1. **Numerical**: Products of probabilities underflow; sums of log-probs don't
2. **Mathematical**: Sums are easier to differentiate than products
3. **Statistical**: Log-likelihood has nicer asymptotic properties (asymptotic normality)
4. **Information-theoretic**: NLL connects to cross-entropy and KL divergence

---

## 4. Softmax Cross-Entropy Gradient

### 4.1 The Combined Operation

In practice, we never compute softmax and cross-entropy separately. The combined gradient has a remarkably simple form.

Let $\mathbf{z} \in \mathbb{R}^K$ be the logits, $\hat{\boldsymbol{\pi}} = \text{softmax}(\mathbf{z})$, and $c$ be the true class index.

$$L = -\log \hat{\pi}_c = -\log \frac{e^{z_c}}{\sum_k e^{z_k}} = \log\sum_k e^{z_k} - z_c$$

### 4.2 Gradient Derivation

$$\frac{\partial L}{\partial z_j} = \frac{\partial}{\partial z_j}\left(\log\sum_k e^{z_k}\right) - \frac{\partial z_c}{\partial z_j}$$

$$= \frac{e^{z_j}}{\sum_k e^{z_k}} - \delta_{jc} = \hat{\pi}_j - \delta_{jc}$$

In vector form:

$$\boxed{\frac{\partial L}{\partial \mathbf{z}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

where $\mathbf{y}$ is the one-hot vector. This is the **softmax cross-entropy gradient**: simply the predicted probabilities minus the true one-hot label.

**Properties**:
- When the model is correct ($\hat{\pi}_c = 1$): gradient is zero
- When the model is wrong ($\hat{\pi}_c = 0$): gradient pushes $z_c$ up and others down
- The gradient sums to zero: $\sum_j \frac{\partial L}{\partial z_j} = 1 - 1 = 0$

```python
# Verify softmax cross-entropy gradient
def softmax_ce_forward(z, y_onehot):
    """Forward pass: softmax + cross-entropy."""
    s = np.exp(z - np.max(z))
    s = s / s.sum()
    loss = -np.sum(y_onehot * np.log(s + 1e-10))
    return loss, s

def softmax_ce_grad(z, y_onehot):
    """Analytical gradient."""
    _, s = softmax_ce_forward(z, y_onehot)
    return s - y_onehot

# Test
K = 5
z = np.array([2.0, 1.0, 0.1, -1.0, 3.0])
y = np.zeros(K)
y[2] = 1.0  # True class = 2

grad_analytical = softmax_ce_grad(z, y)

# Numerical gradient
eps = 1e-5
grad_numerical = np.zeros(K)
for j in range(K):
    z_plus = z.copy(); z_plus[j] += eps
    z_minus = z.copy(); z_minus[j] -= eps
    L_plus, _ = softmax_ce_forward(z_plus, y)
    L_minus, _ = softmax_ce_forward(z_minus, y)
    grad_numerical[j] = (L_plus - L_minus) / (2 * eps)

print(f"Analytical gradient: {grad_analytical.round(6)}")
print(f"Numerical gradient:  {grad_numerical.round(6)}")
print(f"Max error: {np.max(np.abs(grad_analytical - grad_numerical)):.2e}")
print(f"Sum of gradient: {grad_analytical.sum():.2e} (should be ~0)")
```

---

## 5. Regularization as MAP Estimation

### 5.1 Maximum A Posteriori (MAP)

Instead of maximizing $\ell(\boldsymbol{\theta})$, we can maximize the **posterior**:

$$\boldsymbol{\theta}_\text{MAP} = \arg\max_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta} | \mathcal{D}) = \arg\max_{\boldsymbol{\theta}} [\ell(\boldsymbol{\theta}) + \log p(\boldsymbol{\theta})]$$

The prior $p(\boldsymbol{\theta})$ encodes our belief about plausible parameter values.

### 5.2 Gaussian Prior = L2 Regularization (Weight Decay)

If $\theta_j \sim \mathcal{N}(0, \tau^2)$ independently:

$$\log p(\boldsymbol{\theta}) = -\frac{1}{2\tau^2}\sum_j \theta_j^2 + \text{const} = -\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2 + \text{const}$$

where $\lambda = 1/\tau^2$. The MAP objective becomes:

$$\hat{\boldsymbol{\theta}}_\text{MAP} = \arg\min_{\boldsymbol{\theta}} \left[\text{NLL} + \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right]$$

This is exactly the training loss with **L2 regularization** (weight decay).

### 5.3 Laplace Prior = L1 Regularization

If $\theta_j \sim \text{Laplace}(0, b)$:

$$\log p(\boldsymbol{\theta}) \propto -\frac{1}{b}\sum_j |\theta_j|$$

This gives **L1 regularization** (Lasso), which encourages sparse weights.

```python
# Demonstrate L2 regularization effect
np.random.seed(42)

# Generate noisy polynomial data
N = 20
x_data = np.linspace(-1, 1, N)
y_data = 0.5*x_data**2 + 0.3*x_data + np.random.randn(N) * 0.1

# Fit polynomials with different regularization
degrees = 15
X_poly = np.vander(x_data, degrees + 1)

lambdas = [0, 0.001, 0.1, 10.0]
x_plot = np.linspace(-1.1, 1.1, 200)
X_plot = np.vander(x_plot, degrees + 1)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, lam in zip(axes, lambdas):
    # MAP/regularized least squares: (X^T X + lambda I) theta = X^T y
    theta = np.linalg.solve(X_poly.T @ X_poly + lam * np.eye(degrees + 1), X_poly.T @ y_data)
    y_plot = X_plot @ theta

    ax.scatter(x_data, y_data, c='blue', s=20)
    ax.plot(x_plot, y_plot, 'r-', linewidth=2)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(f'λ = {lam}\n||θ|| = {np.linalg.norm(theta):.1f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of L2 regularization on polynomial fitting')
plt.tight_layout()
plt.show()
```

---

## 6. MLE for Logistic Regression

### 6.1 The Model

Logistic regression models $P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$.

The NLL (binary cross-entropy) loss:

$$L(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log \sigma(z_i) + (1-y_i)\log(1 - \sigma(z_i))]$$

where $z_i = \mathbf{w}^\top \mathbf{x}_i + b$.

### 6.2 Gradient

Using the identity $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and the chain rule:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{N}\sum_{i=1}^{N}(\sigma(z_i) - y_i)\mathbf{x}_i$$

$$\frac{\partial L}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}(\sigma(z_i) - y_i)$$

The gradient has the same elegant form: $(\hat{y} - y) \cdot \mathbf{x}$.

### 6.3 Implementation

```python
# MLE for logistic regression from scratch
np.random.seed(42)

# Generate 2D binary classification data
N = 200
X_pos = np.random.randn(N//2, 2) + np.array([1, 1])
X_neg = np.random.randn(N//2, 2) + np.array([-1, -1])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(N//2), np.zeros(N//2)])

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Train logistic regression with gradient descent
w = np.zeros(2)
b = 0.0
lr = 0.1
losses = []

for epoch in range(200):
    # Forward
    z = X @ w + b
    p_hat = sigmoid(z)

    # Loss (NLL)
    loss = -np.mean(y * np.log(p_hat + 1e-7) + (1-y) * np.log(1-p_hat + 1e-7))
    losses.append(loss)

    # Gradients
    error = p_hat - y  # (N,)
    dw = X.T @ error / N
    db = np.mean(error)

    # Update
    w -= lr * dw
    b -= lr * db

print(f"Learned weights: w = {w.round(3)}, b = {b:.3f}")
print(f"Final loss: {losses[-1]:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision boundary
xx = np.linspace(-4, 4, 200)
yy = np.linspace(-4, 4, 200)
XX, YY = np.meshgrid(xx, yy)
ZZ = sigmoid(XX * w[0] + YY * w[1] + b)

axes[0].contourf(XX, YY, ZZ, levels=20, cmap='RdBu', alpha=0.5)
axes[0].scatter(X_pos[:, 0], X_pos[:, 1], c='blue', s=10, label='Class 1')
axes[0].scatter(X_neg[:, 0], X_neg[:, 1], c='red', s=10, label='Class 0')
axes[0].contour(XX, YY, ZZ, levels=[0.5], colors='black', linewidths=2)
axes[0].legend()
axes[0].set_title('Decision boundary')

# Loss curve
axes[1].plot(losses)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('NLL Loss')
axes[1].set_title('Training loss (NLL)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 7. Properties of the MLE

### 7.1 Consistency

As $N \to \infty$, $\hat{\boldsymbol{\theta}}_\text{MLE} \to \boldsymbol{\theta}_\text{true}$ (in probability).

### 7.2 Asymptotic Normality

For large $N$:

$$\hat{\boldsymbol{\theta}}_\text{MLE} \sim \mathcal{N}\left(\boldsymbol{\theta}_\text{true}, \frac{1}{N}\mathbf{F}^{-1}\right)$$

where $\mathbf{F}$ is the Fisher information matrix. The MLE achieves the **Cramer-Rao lower bound** asymptotically -- it is efficient.

### 7.3 Invariance

If $\hat{\theta}_\text{MLE}$ is the MLE of $\theta$, then $g(\hat{\theta}_\text{MLE})$ is the MLE of $g(\theta)$ for any function $g$.

### 7.4 Limitations

- MLE can **overfit** with finite data (no regularization)
- MLE gives **point estimates**, not uncertainty (unlike Bayesian methods)
- MLE assumes the model is **correctly specified** (the true distribution is in the model family)

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Likelihood | $\mathcal{L}(\theta) = \prod p(x_i|\theta)$; work with log-likelihood to avoid underflow |
| MLE | $\hat{\theta} = \arg\max \ell(\theta) = \arg\min \text{NLL}$ |
| MLE = DL training | Minimizing NLL $\Leftrightarrow$ minimizing standard loss functions |
| Softmax CE gradient | $\nabla_\mathbf{z} L = \hat{\boldsymbol{\pi}} - \mathbf{y}$ (predicted minus true) |
| MAP = regularized MLE | Gaussian prior $\to$ L2, Laplace prior $\to$ L1 |
| MLE properties | Consistent, asymptotically normal, efficient |

---

## Exercises

1. Derive the MLE for a Poisson distribution $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$ and show it is the sample mean.
2. Implement softmax cross-entropy loss and its gradient, verify with finite differences.
3. Add L2 regularization to the logistic regression implementation and observe the effect on the decision boundary.
4. Derive the MLE for a multivariate Gaussian (both $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$).
5. Implement a function that computes the Fisher information matrix for logistic regression and verify the asymptotic variance of the MLE.

---

**Next**: [08. Information Theory](08_Information_Theory.md)

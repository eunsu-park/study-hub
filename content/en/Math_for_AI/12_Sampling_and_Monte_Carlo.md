# 12. Sampling and Monte Carlo Methods

## Learning Objectives

- Understand the fundamental principles of Monte Carlo methods and their convergence rate
- Implement basic sampling methods (inverse transform, rejection sampling) and recognize their limitations
- Master variance reduction techniques through importance sampling
- Understand the principles and implementation of MCMC (Metropolis-Hastings, Gibbs sampling)
- Grasp the mathematical background of the reparameterization trick and its role in VAEs
- Learn various use cases of sampling in machine learning

---

## 1. Why Sampling?

### 1.1 Integrals that Cannot Be Computed Analytically

Many machine learning problems require computing integrals of the form:

$$
\mathbb{E}_{p(x)}[f(x)] = \int f(x) p(x) dx
$$

For example:
- **Bayesian inference**: Normalizing constant of the posterior $\int p(x|\theta) p(\theta) d\theta$
- **Reinforcement learning**: Expected reward of a policy $\mathbb{E}_{\pi}[R]$
- **VAE**: Expected value in ELBO $\mathbb{E}_{q(z|x)}[\log p(x|z)]$

These integrals are intractable analytically in high dimensions.

### 1.2 Monte Carlo Estimation

**Monte Carlo principle**: If we draw $N$ samples $x_1, \ldots, x_N$ from distribution $p(x)$:

$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)
$$

This is guaranteed by the **Law of Large Numbers**.

### 1.3 Convergence Rate

The standard error of Monte Carlo estimation is:

$$
\text{SE} = \frac{\sigma}{\sqrt{N}}
$$

where $\sigma^2 = \text{Var}_{p(x)}[f(x)]$.

**Key observation**:
- Error decreases as $O(1/\sqrt{N})$ → to increase accuracy by 10x, need 100x more samples
- **Dimension-independent**: applicable to high-dimensional integrals (grid methods suffer from the curse of dimensionality)

```python
import numpy as np
import matplotlib.pyplot as plt

# Estimate pi using Monte Carlo
def estimate_pi(n_samples):
    """Generate random points in a unit square and estimate π from the fraction inside the unit circle"""
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.mean(inside_circle)
    return pi_estimate

# Convergence with increasing sample size
sample_sizes = [10, 100, 1000, 10000, 100000]
estimates = [estimate_pi(n) for n in sample_sizes]

print("Monte Carlo π estimation:")
for n, est in zip(sample_sizes, estimates):
    error = abs(est - np.pi)
    print(f"N={n:6d}: π ≈ {est:.6f}, error = {error:.6f}")

# Visualize convergence
n_trials = 100
sample_range = np.logspace(1, 5, 50, dtype=int)
errors = []

for n in sample_range:
    trial_estimates = [estimate_pi(n) for _ in range(n_trials)]
    errors.append(np.std(trial_estimates))

plt.figure(figsize=(10, 6))
plt.loglog(sample_range, errors, 'b-', label='Actual standard error')
plt.loglog(sample_range, 1/np.sqrt(sample_range), 'r--', label=r'$O(1/\sqrt{N})$')
plt.xlabel('Number of samples (N)')
plt.ylabel('Standard error')
plt.title('Convergence rate of Monte Carlo estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 2. Basic Sampling Methods

### 2.1 Inverse Transform Sampling

**Theorem**: If $U \sim \text{Uniform}(0, 1)$ and $F$ is a cumulative distribution function (CDF), then $X = F^{-1}(U)$ follows CDF $F$.

**Algorithm**:
1. Generate $u \sim \text{Uniform}(0, 1)$
2. Compute $x = F^{-1}(u)$

```python
# Inverse transform sampling example: exponential distribution
def inverse_transform_exponential(n_samples, lambda_param=1.0):
    """Sample from exponential distribution Exp(λ)"""
    u = np.random.uniform(0, 1, n_samples)
    # F^(-1)(u) = -ln(1-u)/λ
    x = -np.log(1 - u) / lambda_param
    return x

# Verification
samples = inverse_transform_exponential(10000, lambda_param=2.0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Sampling result')
x_range = np.linspace(0, 5, 100)
plt.plot(x_range, 2 * np.exp(-2 * x_range), 'r-', linewidth=2, label='Theoretical PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Inverse transform sampling: Exponential distribution')
plt.legend()

plt.subplot(1, 2, 2)
# Q-Q plot
from scipy import stats
stats.probplot(samples, dist=stats.expon(scale=0.5), plot=plt)
plt.title('Q-Q Plot')
plt.tight_layout()
plt.show()
```

**Limitation**: Computing $F^{-1}$ is often difficult.

### 2.2 Rejection Sampling

When direct sampling from the target distribution $p(x)$ is difficult, use a proposal distribution $q(x)$.

**Algorithm**:
1. Choose constant $M$ such that $M q(x) \geq p(x)$ (for all $x$)
2. Generate $x \sim q(x)$
3. Generate $u \sim \text{Uniform}(0, 1)$
4. Accept $x$ if $u \leq \frac{p(x)}{M q(x)}$, otherwise reject

**Acceptance rate**: $\frac{1}{M}$

```python
# Rejection sampling example: Beta distribution Beta(2, 5)
def target_pdf(x):
    """Target distribution: Beta(2, 5)"""
    if 0 <= x <= 1:
        return 30 * x * (1-x)**4  # Normalized Beta(2,5)
    return 0

def proposal_pdf(x):
    """Proposal distribution: Uniform(0, 1)"""
    return 1.0 if 0 <= x <= 1 else 0

# Find M (maximum value)
x_grid = np.linspace(0, 1, 1000)
M = max(target_pdf(x) / proposal_pdf(x) for x in x_grid)
print(f"M = {M:.2f}")

def rejection_sampling(n_samples):
    """Rejection sampling implementation"""
    samples = []
    n_rejected = 0

    while len(samples) < n_samples:
        x = np.random.uniform(0, 1)  # Sample from proposal distribution
        u = np.random.uniform(0, 1)

        if u <= target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)
        else:
            n_rejected += 1

    acceptance_rate = n_samples / (n_samples + n_rejected)
    print(f"Acceptance rate: {acceptance_rate:.2%} (theoretical: {1/M:.2%})")
    return np.array(samples)

samples = rejection_sampling(10000)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Sampling result')
x_range = np.linspace(0, 1, 100)
plt.plot(x_range, [target_pdf(x) for x in x_range], 'r-', linewidth=2, label='Target PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Rejection sampling: Beta(2, 5)')
plt.legend()
plt.show()
```

### 2.3 Limitations in High Dimensions

The acceptance rate of rejection sampling is $1/M$. In high dimensions, $M$ grows exponentially, making it inefficient.

**Example**: For a 10-dimensional Gaussian, $M \sim e^{10}$ → acceptance rate $< 0.01\%$

---

## 3. Importance Sampling

### 3.1 Basic Principle

Goal: Compute $\mathbb{E}_{p(x)}[f(x)]$

**Importance sampling identity**:

$$
\mathbb{E}_{p(x)}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{q(x)}\left[f(x) \frac{p(x)}{q(x)}\right]
$$

$q(x)$ is called the **proposal distribution**, and $w(x) = \frac{p(x)}{q(x)}$ is the **importance weight**.

**Monte Carlo estimate**:
$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i) w(x_i), \quad x_i \sim q(x)
$$

### 3.2 Choosing the Proposal Distribution

**Good proposal distribution**:
- If $q(x)$ is proportional to $|f(x)| p(x)$, variance is minimized
- In practice, $q(x)$ should cover heavy tails of $p(x)$

**Bad choice**: If $q(x)$ has lighter tails than $p(x)$ → weights explode, leading to high variance

### 3.3 Self-Normalized Importance Sampling

When $p(x)$ is known only up to a normalizing constant ($p(x) = \tilde{p}(x)/Z$):

$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{\sum_{i=1}^{N} f(x_i) \tilde{w}(x_i)}{\sum_{i=1}^{N} \tilde{w}(x_i)}, \quad \tilde{w}(x_i) = \frac{\tilde{p}(x_i)}{q(x_i)}
$$

```python
# Importance sampling example
def target_unnormalized(x):
    """Unnormalized target distribution: mixture of Gaussians"""
    return 0.3 * np.exp(-0.5 * ((x-2)/0.5)**2) + \
           0.7 * np.exp(-0.5 * ((x+2)/1.0)**2)

def proposal(x):
    """Proposal distribution: N(0, 3)"""
    return np.exp(-0.5 * (x/3)**2) / (3 * np.sqrt(2*np.pi))

def f(x):
    """Function to compute: x^2"""
    return x**2

# Naive Monte Carlo (assuming direct sampling from target is possible)
def naive_monte_carlo(n_samples):
    # Sample from target distribution via rejection sampling
    samples = []
    while len(samples) < n_samples:
        x = np.random.normal(0, 3)
        u = np.random.uniform(0, target_unnormalized(x))
        if u <= target_unnormalized(x):
            samples.append(x)
    samples = np.array(samples)
    return np.mean(f(samples))

# Importance sampling
def importance_sampling(n_samples):
    # Sample from proposal distribution
    samples = np.random.normal(0, 3, n_samples)
    weights = target_unnormalized(samples) / proposal(samples)
    weights = weights / np.sum(weights)  # Self-normalization
    return np.sum(f(samples) * weights * n_samples)

# Comparison
n_trials = 100
n_samples = 1000

naive_results = [naive_monte_carlo(n_samples) for _ in range(n_trials)]
is_results = [importance_sampling(n_samples) for _ in range(n_trials)]

print(f"Naive MC: mean = {np.mean(naive_results):.4f}, std = {np.std(naive_results):.4f}")
print(f"Importance sampling: mean = {np.mean(is_results):.4f}, std = {np.std(is_results):.4f}")
print(f"Variance reduction ratio: {np.var(naive_results) / np.var(is_results):.2f}x")
```

### 3.4 Connection to Reinforcement Learning

Importance sampling is central to policy gradient methods:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) Q(s,a) \right]
$$

This enables **off-policy learning** (PPO, TRPO).

---

## 4. Markov Chain Monte Carlo (MCMC)

### 4.1 Markov Chain Basics

**Markov chain**: Stochastic state transitions $P(x_{t+1} | x_t)$

**Stationary distribution**: $\pi(x) = \int \pi(x') P(x|x') dx'$

**Detailed balance condition**:
$$
\pi(x) P(x'|x) = \pi(x') P(x|x')
$$

If detailed balance is satisfied, $\pi$ is the stationary distribution.

### 4.2 Metropolis-Hastings Algorithm

**Goal**: Sample from target distribution $\pi(x)$ (normalizing constant not needed)

**Algorithm**:
1. Start from current state $x_t$
2. Generate candidate $x'$ from proposal distribution $q(x'|x_t)$
3. Compute acceptance probability:
$$
\alpha(x', x_t) = \min\left(1, \frac{\pi(x') q(x_t|x')}{\pi(x_t) q(x'|x_t)}\right)
$$
4. Accept $x_{t+1} = x'$ with probability $\alpha$, otherwise $x_{t+1} = x_t$

```python
# Metropolis-Hastings implementation
def target_distribution(x):
    """Target distribution: bimodal distribution"""
    return np.exp(-0.5 * ((x-3)/0.8)**2) + np.exp(-0.5 * ((x+3)/0.8)**2)

def metropolis_hastings(n_samples, proposal_std=1.0):
    """MH algorithm (symmetric proposal distribution)"""
    samples = np.zeros(n_samples)
    x = 0.0  # Initial state
    n_accepted = 0

    for i in range(n_samples):
        # Generate proposal (symmetric Gaussian)
        x_proposed = x + np.random.normal(0, proposal_std)

        # Compute acceptance probability (q terms cancel due to symmetry)
        acceptance_prob = min(1, target_distribution(x_proposed) / target_distribution(x))

        # Accept/reject decision
        if np.random.uniform() < acceptance_prob:
            x = x_proposed
            n_accepted += 1

        samples[i] = x

    acceptance_rate = n_accepted / n_samples
    print(f"Acceptance rate: {acceptance_rate:.2%}")
    return samples

# Sampling
samples = metropolis_hastings(50000, proposal_std=2.0)

# Remove burn-in
burn_in = 5000
samples_burned = samples[burn_in:]

plt.figure(figsize=(15, 5))

# Trace plot
plt.subplot(1, 3, 1)
plt.plot(samples[:1000])
plt.xlabel('Iteration')
plt.ylabel('x')
plt.title('MCMC trace (first 1000)')
plt.axvline(burn_in, color='r', linestyle='--', label='End of burn-in')

# Histogram
plt.subplot(1, 3, 2)
plt.hist(samples_burned, bins=50, density=True, alpha=0.7, label='MCMC samples')
x_range = np.linspace(-6, 6, 200)
plt.plot(x_range, target_distribution(x_range) /
         (np.sqrt(2*np.pi*0.8**2) * 2), 'r-', linewidth=2, label='Target distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Sample distribution')
plt.legend()

# Autocorrelation
plt.subplot(1, 3, 3)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(samples_burned[::10], lags=50, ax=plt.gca())
plt.title('Autocorrelation function (ACF)')
plt.tight_layout()
plt.show()
```

### 4.3 Gibbs Sampling

Used when conditional distributions $p(x_i | x_{-i})$ are known for multivariate distribution $p(x_1, \ldots, x_d)$.

**Algorithm**:
1. Iterate over each variable
2. Sample $x_i^{(t+1)} \sim p(x_i | x_1^{(t+1)}, \ldots, x_{i-1}^{(t+1)}, x_{i+1}^{(t)}, \ldots, x_d^{(t)})$

```python
# Gibbs sampling: bivariate Gaussian
def gibbs_sampling_bivariate_gaussian(n_samples, rho=0.9):
    """Sample from bivariate Gaussian N([0,0], [[1,ρ],[ρ,1]])"""
    samples = np.zeros((n_samples, 2))
    x, y = 0.0, 0.0  # Initial values

    for i in range(n_samples):
        # x | y ~ N(ρy, 1-ρ²)
        x = np.random.normal(rho * y, np.sqrt(1 - rho**2))
        # y | x ~ N(ρx, 1-ρ²)
        y = np.random.normal(rho * x, np.sqrt(1 - rho**2))
        samples[i] = [x, y]

    return samples

samples = gibbs_sampling_bivariate_gaussian(10000, rho=0.9)
samples = samples[1000:]  # Remove burn-in

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gibbs sampling: bivariate Gaussian (ρ=0.9)')
plt.axis('equal')

plt.subplot(1, 2, 2)
# Theoretical contours
from scipy.stats import multivariate_normal
x_grid = np.linspace(-4, 4, 100)
y_grid = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X, Y))
rv = multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]])
plt.contour(X, Y, rv.pdf(pos), levels=10)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison with theoretical distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

### 4.4 Burn-in and Autocorrelation

- **Burn-in**: Initial samples haven't reached the stationary distribution, so discard them
- **Autocorrelation**: Consecutive samples are correlated → thin by taking every k-th sample
- **Effective sample size**: $n_{\text{eff}} = \frac{n}{1 + 2\sum_{k=1}^{\infty} \rho_k}$

---

## 5. Reparameterization Trick

### 5.1 Problem: Backpropagating Through Stochastic Nodes

In VAE, the encoder outputs $q_\phi(z|x)$ and samples $z \sim q_\phi(z|x)$. The loss function:

$$
\mathcal{L}(\phi) = \mathbb{E}_{z \sim q_\phi(z|x)}[f(z)]
$$

**Problem**: How to compute $\frac{\partial}{\partial \phi} \mathbb{E}_{z \sim q_\phi}[f(z)]$?

Sampling operations are **not differentiable**.

### 5.2 Reparameterization Solution

**Idea**: Separate randomness into noise independent of $\phi$.

**Gaussian distribution**: $z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$

Reparameterization: $z = \mu_\phi + \sigma_\phi \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$

Now:
$$
\frac{\partial}{\partial \phi} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[f(\mu_\phi + \sigma_\phi \epsilon)] = \mathbb{E}_{\epsilon}\left[\frac{\partial f(z)}{\partial z} \frac{\partial z}{\partial \phi}\right]
$$

**Gradients can flow through the sampling operation!**

```python
import torch
import torch.nn as nn

# Reparameterization trick implementation
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # ε ~ N(0, 1)
        z = mu + std * eps           # z = μ + σε
        return z

# Gradient flow test
encoder = VAEEncoder(input_dim=10, latent_dim=2)
x = torch.randn(32, 10)

mu, logvar = encoder(x)
z = encoder.reparameterize(mu, logvar)

# Arbitrary loss function
loss = (z ** 2).sum()
loss.backward()

print("Gradients computed successfully:")
print(f"  Gradient of mu: {encoder.fc_mu.weight.grad is not None}")
print(f"  Gradient of logvar: {encoder.fc_logvar.weight.grad is not None}")
print(f"  Gradient norm: {encoder.fc_mu.weight.grad.norm():.4f}")
```

### 5.3 Reparameterization for Other Distributions

**Bernoulli**: Gumbel-Softmax trick
$$
z = \text{one-hot}\left(\arg\max_i \left[\log \pi_i + G_i\right]\right)
$$
where $G_i \sim \text{Gumbel}(0, 1)$

**Gamma distribution**: Shape augmentation

**General principle**: Reparameterizable if expressible as $z = g(\epsilon; \theta)$

### 5.4 VAE ELBO

VAE loss function (ELBO):

$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

**Applying reparameterization**:
- First term: Compute gradients using reparameterization trick
- Second term: Analytically computable under Gaussian prior assumption

```python
def vae_loss(x, x_recon, mu, logvar):
    """VAE loss function (ELBO)"""
    # Reconstruction loss (log p(x|z))
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence (analytically computed under Gaussian assumption)
    # KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

# Full VAE training loop (simplified)
def train_vae(encoder, decoder, data_loader, optimizer, epochs):
    for epoch in range(epochs):
        for x in data_loader:
            # Encoder
            mu, logvar = encoder(x)
            z = encoder.reparameterize(mu, logvar)

            # Decoder
            x_recon = decoder(z)

            # Compute loss and backpropagate
            loss = vae_loss(x, x_recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()  # Gradients flow thanks to reparameterization!
            optimizer.step()
```

---

## 6. Machine Learning Applications

### 6.1 VAE: ELBO Optimization

Without the reparameterization trick, VAEs would be impossible. REINFORCE (score function estimation) has too high variance.

### 6.2 Reinforcement Learning: REINFORCE = Monte Carlo

Policy gradient theorem:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)\right]
$$

This is estimated using Monte Carlo sampling.

**Variance reduction**: Use baselines, GAE (Generalized Advantage Estimation)

### 6.3 Bayesian Inference: MCMC vs Variational Inference

**MCMC**:
- Pros: Theoretically exact samples
- Cons: Slow, difficult to diagnose convergence

**Variational Inference**:
- Pros: Fast, scalable
- Cons: Approximation quality limited by expressiveness of $q$

### 6.4 Dropout = Bernoulli Sampling

Dropout is Bernoulli sampling during training:
$$
h_{\text{drop}} = h \odot \text{Bernoulli}(p)
$$

**Bayesian interpretation**: Dropout can be viewed as approximate Bayesian inference (Gal & Ghahramani, 2016).

```python
# Uncertainty estimation with MC Dropout
class MCDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout even at inference time
        x = self.fc2(x)
        return x

def predict_with_uncertainty(model, x, n_samples=100):
    """Estimate prediction uncertainty with MC Dropout"""
    model.train()  # Enable dropout
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.cat(predictions, dim=1)
    mean = predictions.mean(dim=1)
    std = predictions.std(dim=1)
    return mean, std

# Test
model = MCDropoutModel()
x_test = torch.randn(10, 10)
mean, std = predict_with_uncertainty(model, x_test, n_samples=100)
print(f"Prediction mean: {mean[:5]}")
print(f"Prediction std (uncertainty): {std[:5]}")
```

---

## Practice Problems

### Problem 1: Monte Carlo Integration
Estimate the following integral using Monte Carlo method:
$$
I = \int_0^1 e^{-x^2} dx
$$

(a) Estimate using 10, 100, 1000, 10000 samples and calculate errors.
(b) Verify on a log-log plot that error decreases as $O(1/\sqrt{N})$.
(c) Compare with results from `scipy.integrate.quad`.

### Problem 2: Importance Sampling
You want to compute $\mathbb{E}[x^2]$ from target distribution $p(x) \propto e^{-|x|^3}$.

(a) Implement importance sampling using proposal distribution $q(x) = \mathcal{N}(0, 1)$.
(b) Compare variance with the case using $q(x) = \text{Laplace}(0, 1)$ as proposal.
(c) Which proposal distribution is more efficient? Explain why.

### Problem 3: MCMC Diagnostics
Perform Metropolis-Hastings sampling from bimodal distribution $p(x) \propto e^{-(x-3)^2/2} + e^{-(x+3)^2/2}$.

(a) Observe acceptance rates as you vary proposal distribution standard deviation: 0.1, 1.0, 10.0.
(b) Plot the autocorrelation function (ACF) for each case and estimate effective sample size.
(c) What is the optimal proposal distribution standard deviation?

### Problem 4: Reparameterization Trick
Implement a simple VAE and verify the effect of the reparameterization trick.

(a) Train a small VAE on MNIST dataset (latent_dim=2).
(b) Compare with training using REINFORCE without reparameterization trick.
(c) Visualize the latent space with 2D plots.

### Problem 5: Gibbs Sampling Extension
Implement Gibbs sampling for a 3-variate Gaussian distribution.

$$
\mathbf{x} \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} 1 & 0.8 & 0.5 \\ 0.8 & 1 & 0.7 \\ 0.5 & 0.7 & 1 \end{bmatrix}\right)
$$

(a) Derive conditional distributions $p(x_i | x_{-i})$ (use Gaussian conditional formula).
(b) Perform 10,000 iterations of Gibbs sampling and compute sample covariance.
(c) Compare with theoretical covariance.

---

## References

### Books
- **Pattern Recognition and Machine Learning** (Bishop, 2006) - Chapter 11: Sampling Methods
- **Monte Carlo Statistical Methods** (Robert & Casella, 2004)
- **Deep Learning** (Goodfellow et al., 2016) - Chapter 17: Monte Carlo Methods

### Papers
- Kingma & Welling (2013), "Auto-Encoding Variational Bayes" - VAE and reparameterization trick
- Metropolis et al. (1953), "Equation of State Calculations by Fast Computing Machines"
- Hastings (1970), "Monte Carlo Sampling Methods Using Markov Chains"
- Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"

### Online Resources
- [MCMC Interactive Gallery](https://chi-feng.github.io/mcmc-demo/)
- [Distill.pub: Visualizing Sampling Methods](https://distill.pub)
- [PyMC Documentation](https://www.pymc.io) - Practical Bayesian inference library

### Libraries
- **PyMC3/PyMC4**: Bayesian inference and MCMC
- **Stan**: Powerful MCMC inference engine
- **PyTorch**: VAE implementation
- **NumPyro**: JAX-based probabilistic programming

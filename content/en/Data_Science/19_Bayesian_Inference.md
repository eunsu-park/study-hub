# 19. Bayesian Inference

[Previous: Introduction to Bayesian Statistics](./18_Bayesian_Statistics_Basics.md) | [Next: Time Series Basics](./20_Time_Series_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why MCMC is necessary when conjugate priors are unavailable and the posterior has no closed form
2. Implement the Metropolis-Hastings algorithm from scratch and tune the proposal distribution for an appropriate acceptance rate
3. Build and sample Bayesian models using PyMC, including specifying priors, likelihoods, and sampling parameters
4. Implement Bayesian linear regression and compare posterior estimates with frequentist OLS results
5. Compare competing models using WAIC, LOO-CV, and interpret Bayes factors
6. Perform posterior predictive checks to assess model fit and detect model misspecification
7. Evaluate MCMC convergence using R-hat, effective sample size, trace plots, and autocorrelation diagnostics

---

The previous lesson showed how conjugate priors yield elegant closed-form posteriors, but most real-world models are too complex for that luxury. When the posterior cannot be computed analytically, MCMC algorithms let you draw samples from it instead -- turning an intractable integral into a simulation problem. This lesson teaches you both the theory behind MCMC and the practical workflow of modern Bayesian modeling with PyMC.

---

## 1. Introduction to MCMC

### 1.1 Why Do We Need MCMC?

**Problem**: Difficulty calculating the normalizing constant of complex posterior distributions

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{\int P(D|\theta)P(\theta)d\theta}$$

The integral in the denominator may be intractable in high dimensions.

**Solution**: Approximate the distribution by sampling directly from the posterior

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Visualize core idea of MCMC
np.random.seed(42)

# Target distribution: mixture of normals (ratio known even without normalizing constant)
def target_unnormalized(x):
    """Unnormalized target distribution"""
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) + \
           0.7 * np.exp(-0.5 * ((x + 1) / 1)**2)

x_range = np.linspace(-5, 5, 200)
y = target_unnormalized(x_range)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Unnormalized target distribution
axes[0].plot(x_range, y, 'b-', lw=2, label='Unnormalized target')
axes[0].fill_between(x_range, y, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Unnormalized density')
axes[0].set_title('Unnormalized Target Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MCMC sample histogram (to be filled in next section)
axes[1].text(0.5, 0.5, 'MCMC Samples\n(filled in next section)',
             transform=axes[1].transAxes, ha='center', va='center', fontsize=14)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Density')
axes[1].set_title('MCMC Sample Histogram')

plt.tight_layout()
plt.show()
```

### 1.2 Markov Chain Basics

**Definition**: A stochastic process where the next state depends only on the current state

$$P(X_{t+1}|X_1, X_2, ..., X_t) = P(X_{t+1}|X_t)$$

```python
def simple_markov_chain_demo():
    """Simple Markov chain simulation"""

    # Weather transition probabilities: Sunny(0), Cloudy(1), Rainy(2)
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],  # Sunny → Sunny/Cloudy/Rainy
        [0.3, 0.4, 0.3],  # Cloudy → Sunny/Cloudy/Rainy
        [0.2, 0.3, 0.5],  # Rainy → Sunny/Cloudy/Rainy
    ])

    states = ['Sunny', 'Cloudy', 'Rainy']

    # Simulation
    n_steps = 1000
    current_state = 0  # Start sunny
    chain = [current_state]

    for _ in range(n_steps):
        probs = transition_matrix[current_state]
        current_state = np.random.choice([0, 1, 2], p=probs)
        chain.append(current_state)

    # Estimate stationary distribution
    unique, counts = np.unique(chain, return_counts=True)
    empirical_dist = counts / len(chain)

    # Theoretical stationary distribution (left eigenvector of transition matrix)
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1))
    stationary_dist = eigenvectors[:, stationary_idx].real
    stationary_dist = stationary_dist / stationary_dist.sum()

    print("=== Markov Chain Weather Simulation ===")
    print(f"Simulation steps: {n_steps}")
    print("\nFrequency by state:")
    for i, state in enumerate(states):
        print(f"  {state}: Empirical={empirical_dist[i]:.3f}, Theoretical={stationary_dist[i]:.3f}")

    return chain

chain = simple_markov_chain_demo()
```

---

## 2. Metropolis-Hastings Algorithm

### 2.1 Algorithm Overview

**Goal**: Sample from target distribution π(x)

**Algorithm**:
1. Choose initial value x₀
2. Generate candidate x' from proposal distribution q(x'|x)
3. Calculate acceptance probability: α = min(1, [π(x')q(x|x')] / [π(x)q(x'|x)])
4. If U ~ Uniform(0,1):
   - If U < α: Accept x' (x_{t+1} = x')
   - Otherwise: Keep x (x_{t+1} = x)
5. Repeat 2-4

### 2.2 Implementation

```python
def metropolis_hastings(target, proposal_std, n_samples, initial_value=0, burn_in=1000):
    """
    Metropolis-Hastings algorithm

    Parameters:
    -----------
    target : callable
        Unnormalized target density function
    proposal_std : float
        Standard deviation of proposal distribution (normal)
    n_samples : int
        Number of samples to generate
    initial_value : float
        Initial value
    burn_in : int
        Burn-in period (number of initial samples to discard)

    Returns:
    --------
    samples : ndarray
        MCMC samples
    acceptance_rate : float
        Acceptance rate
    """
    samples = np.zeros(n_samples + burn_in)
    samples[0] = initial_value
    n_accepted = 0

    for i in range(1, n_samples + burn_in):
        current = samples[i-1]

        # Proposal (symmetric normal)
        proposal = np.random.normal(current, proposal_std)

        # Acceptance probability (q cancels for symmetric proposal)
        acceptance_ratio = target(proposal) / target(current)
        acceptance_prob = min(1, acceptance_ratio)

        # Accept/reject
        if np.random.uniform() < acceptance_prob:
            samples[i] = proposal
            n_accepted += 1
        else:
            samples[i] = current

    # Remove burn-in
    samples = samples[burn_in:]
    acceptance_rate = n_accepted / (n_samples + burn_in)

    return samples, acceptance_rate


# Example: Sample from mixture of normals
def mixture_target(x):
    """Mixture of normals (unnormalized)"""
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) + \
           0.7 * np.exp(-0.5 * ((x + 1) / 1)**2)

# Experiment with different proposal distributions
proposal_stds = [0.1, 1.0, 5.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

np.random.seed(42)

for i, proposal_std in enumerate(proposal_stds):
    samples, acc_rate = metropolis_hastings(
        mixture_target,
        proposal_std=proposal_std,
        n_samples=10000,
        burn_in=1000
    )

    # Trace plot
    axes[0, i].plot(samples[:500], 'b-', alpha=0.7, lw=0.5)
    axes[0, i].set_xlabel('Iteration')
    axes[0, i].set_ylabel('Value')
    axes[0, i].set_title(f'Trace (σ={proposal_std}, acc={acc_rate:.2%})')
    axes[0, i].grid(True, alpha=0.3)

    # Histogram
    x_range = np.linspace(-5, 5, 200)
    true_density = mixture_target(x_range)
    true_density = true_density / np.trapz(true_density, x_range)

    axes[1, i].hist(samples, bins=50, density=True, alpha=0.7, label='MCMC samples')
    axes[1, i].plot(x_range, true_density, 'r-', lw=2, label='True density')
    axes[1, i].set_xlabel('Value')
    axes[1, i].set_ylabel('Density')
    axes[1, i].set_title(f'Histogram (σ={proposal_std})')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle('Metropolis-Hastings: Effect of Proposal Distribution Std Dev')
plt.tight_layout()
plt.show()
```

### 2.3 Multivariate Extension

```python
def metropolis_hastings_2d(target, proposal_cov, n_samples, initial_value=None, burn_in=1000):
    """
    2D Metropolis-Hastings

    Parameters:
    -----------
    target : callable
        Unnormalized 2D target density function
    proposal_cov : ndarray
        Covariance matrix of proposal distribution (2x2)
    n_samples : int
        Number of samples to generate
    """
    if initial_value is None:
        initial_value = np.array([0.0, 0.0])

    samples = np.zeros((n_samples + burn_in, 2))
    samples[0] = initial_value
    n_accepted = 0

    for i in range(1, n_samples + burn_in):
        current = samples[i-1]

        # Propose from multivariate normal
        proposal = np.random.multivariate_normal(current, proposal_cov)

        # Acceptance probability
        acceptance_ratio = target(proposal) / (target(current) + 1e-10)
        acceptance_prob = min(1, acceptance_ratio)

        if np.random.uniform() < acceptance_prob:
            samples[i] = proposal
            n_accepted += 1
        else:
            samples[i] = current

    samples = samples[burn_in:]
    acceptance_rate = n_accepted / (n_samples + burn_in)

    return samples, acceptance_rate


# 2D mixture of normals
def target_2d(x):
    """2D mixture of normals"""
    # First component
    mu1 = np.array([2, 2])
    cov1 = np.array([[0.5, 0.3], [0.3, 0.5]])
    term1 = stats.multivariate_normal(mu1, cov1).pdf(x)

    # Second component
    mu2 = np.array([-1, -1])
    cov2 = np.array([[0.8, -0.2], [-0.2, 0.8]])
    term2 = stats.multivariate_normal(mu2, cov2).pdf(x)

    return 0.4 * term1 + 0.6 * term2

# Sampling
np.random.seed(42)
proposal_cov = np.array([[0.5, 0], [0, 0.5]])
samples_2d, acc_rate = metropolis_hastings_2d(
    target_2d,
    proposal_cov=proposal_cov,
    n_samples=20000,
    burn_in=5000
)

print(f"Acceptance rate: {acc_rate:.2%}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Trace plot
axes[0].plot(samples_2d[:1000, 0], 'b-', alpha=0.7, lw=0.5, label='x1')
axes[0].plot(samples_2d[:1000, 1], 'r-', alpha=0.7, lw=0.5, label='x2')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Value')
axes[0].set_title('Trace Plot (first 1000)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D scatter
axes[1].scatter(samples_2d[::10, 0], samples_2d[::10, 1], alpha=0.3, s=5)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_title('MCMC Samples (2D)')
axes[1].grid(True, alpha=0.3)

# Contour comparison
x1_range = np.linspace(-4, 5, 100)
x2_range = np.linspace(-4, 5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
positions = np.dstack((X1, X2))
Z = np.array([[target_2d(p) for p in row] for row in positions])

axes[2].contour(X1, X2, Z, levels=10, cmap='viridis')
axes[2].scatter(samples_2d[::20, 0], samples_2d[::20, 1], alpha=0.2, s=3, c='red')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].set_title('Samples vs True Density')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3. Introduction to PyMC

### 3.1 Installation and Basic Setup

```bash
# Install PyMC (Python 3.9+)
pip install pymc arviz

# Or with conda
conda install -c conda-forge pymc arviz
```

```python
# Basic imports
import pymc as pm
import arviz as az

# Visualization settings
az.style.use("arviz-darkgrid")

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")
```

### 3.2 First PyMC Model: Coin Flipping

```python
import pymc as pm
import arviz as az

# Data
n_trials = 100
n_heads = 65

# Define PyMC model
with pm.Model() as coin_model:
    # Prior: Beta(1, 1) = Uniform(0, 1)
    p = pm.Beta('p', alpha=1, beta=1)

    # Likelihood: Binomial
    y = pm.Binomial('y', n=n_trials, p=p, observed=n_heads)

    # Sample posterior
    trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)

# Check results
print(az.summary(trace, var_names=['p']))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Trace plot
az.plot_trace(trace, var_names=['p'], axes=axes.reshape(1, 2))

plt.tight_layout()
plt.show()

# Posterior plot
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_posterior(trace, var_names=['p'], ax=ax)
ax.axvline(0.65, color='r', linestyle='--', label=f'MLE = 0.65')
ax.legend()
plt.show()
```

### 3.3 Normal Distribution Parameter Estimation

```python
# Generate data
np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
data = np.random.normal(true_mu, true_sigma, 50)

print(f"True μ: {true_mu}, σ: {true_sigma}")
print(f"Sample mean: {data.mean():.3f}, std: {data.std():.3f}")

# PyMC model
with pm.Model() as normal_model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=10)  # Weakly informative prior
    sigma = pm.HalfNormal('sigma', sigma=5)  # Positive constraint

    # Likelihood
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)

    # Sampling
    trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)

# Summary
print("\n=== Posterior Summary ===")
print(az.summary(trace, var_names=['mu', 'sigma']))

# Visualization
fig = plt.figure(figsize=(14, 8))

# Trace plot
axes = fig.subplots(2, 2)
az.plot_trace(trace, var_names=['mu', 'sigma'], axes=axes)
plt.tight_layout()
plt.show()

# Joint posterior
fig, ax = plt.subplots(figsize=(8, 8))
az.plot_pair(trace, var_names=['mu', 'sigma'], kind='kde', ax=ax)
ax.axvline(true_mu, color='r', linestyle='--', alpha=0.5)
ax.axhline(true_sigma, color='r', linestyle='--', alpha=0.5)
ax.scatter([true_mu], [true_sigma], color='r', s=100, marker='x', label='True values')
ax.legend()
plt.show()
```

---

## 4. Bayesian Linear Regression

### 4.1 Model Definition

$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$

$$\epsilon_i \sim N(0, \sigma^2)$$

**Bayesian setup**:
- β₀ ~ N(0, 10²)
- β₁ ~ N(0, 10²)
- σ ~ HalfNormal(5)

### 4.2 PyMC Implementation

```python
# Generate data
np.random.seed(42)
n = 100
x = np.random.uniform(0, 10, n)
true_beta0 = 2.5
true_beta1 = 1.5
true_sigma = 1.0
y = true_beta0 + true_beta1 * x + np.random.normal(0, true_sigma, n)

print(f"True parameters: β0={true_beta0}, β1={true_beta1}, σ={true_sigma}")

# Frequentist OLS comparison
from scipy import stats as scipy_stats
slope_ols, intercept_ols, _, _, _ = scipy_stats.linregress(x, y)
print(f"OLS estimates: β0={intercept_ols:.3f}, β1={slope_ols:.3f}")

# Bayesian linear regression
with pm.Model() as linear_model:
    # Priors
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Linear prediction
    mu = beta0 + beta1 * x

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Sampling
    trace = pm.sample(3000, tune=1000, cores=1, random_seed=42)

# Results
print("\n=== Bayesian Estimation Results ===")
summary = az.summary(trace, var_names=['beta0', 'beta1', 'sigma'])
print(summary)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Posteriors
az.plot_posterior(trace, var_names=['beta0'], ax=axes[0, 0])
axes[0, 0].axvline(true_beta0, color='r', linestyle='--', label='True')
axes[0, 0].axvline(intercept_ols, color='g', linestyle=':', label='OLS')
axes[0, 0].legend()

az.plot_posterior(trace, var_names=['beta1'], ax=axes[0, 1])
axes[0, 1].axvline(true_beta1, color='r', linestyle='--', label='True')
axes[0, 1].axvline(slope_ols, color='g', linestyle=':', label='OLS')
axes[0, 1].legend()

az.plot_posterior(trace, var_names=['sigma'], ax=axes[0, 2])
axes[0, 2].axvline(true_sigma, color='r', linestyle='--', label='True')
axes[0, 2].legend()

# Regression line uncertainty
ax = axes[1, 0]
ax.scatter(x, y, alpha=0.5, s=20)

# Regression lines from posterior samples
x_plot = np.linspace(0, 10, 100)
posterior_samples = trace.posterior

# Draw 100 posterior samples
for i in range(100):
    b0 = posterior_samples['beta0'].values.flatten()[i * 30]  # thin
    b1 = posterior_samples['beta1'].values.flatten()[i * 30]
    ax.plot(x_plot, b0 + b1 * x_plot, 'b-', alpha=0.05)

# Posterior mean regression line
mean_b0 = posterior_samples['beta0'].mean().values
mean_b1 = posterior_samples['beta1'].mean().values
ax.plot(x_plot, mean_b0 + mean_b1 * x_plot, 'r-', lw=2, label='Posterior mean')
ax.plot(x_plot, true_beta0 + true_beta1 * x_plot, 'g--', lw=2, label='True')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Regression Line Uncertainty')
ax.legend()
ax.grid(True, alpha=0.3)

# Joint posterior: beta0 vs beta1
ax = axes[1, 1]
az.plot_pair(trace, var_names=['beta0', 'beta1'], kind='kde', ax=ax)
ax.scatter([true_beta0], [true_beta1], color='r', s=100, marker='x', zorder=5)
ax.set_title('β0 vs β1 joint distribution')

# Trace plot
ax = axes[1, 2]
az.plot_trace(trace, var_names=['beta1'], axes=np.array([[ax, ax]]))
ax.set_title('β1 trace')

plt.tight_layout()
plt.show()
```

### 4.3 Posterior Prediction

```python
# Posterior predictive distribution
with linear_model:
    # Prediction for new x values
    x_new = np.array([3, 5, 7])

    # Posterior predictive samples
    pm.set_data({'x': x_new})  # If using shared variables

    # Manual posterior prediction
    posterior = trace.posterior

    beta0_samples = posterior['beta0'].values.flatten()
    beta1_samples = posterior['beta1'].values.flatten()
    sigma_samples = posterior['sigma'].values.flatten()

# Predictions
predictions = {}
for x_val in x_new:
    mu_pred = beta0_samples + beta1_samples * x_val
    y_pred = np.random.normal(mu_pred, sigma_samples)
    predictions[x_val] = {
        'mean': mu_pred.mean(),
        'std': mu_pred.std(),
        'pred_mean': y_pred.mean(),
        'pred_std': y_pred.std(),
        'ci_95': (np.percentile(mu_pred, 2.5), np.percentile(mu_pred, 97.5)),
        'pred_95': (np.percentile(y_pred, 2.5), np.percentile(y_pred, 97.5))
    }

print("=== Posterior Prediction ===")
for x_val, pred in predictions.items():
    print(f"\nx = {x_val}:")
    print(f"  E[y|x] = {pred['mean']:.3f} ± {pred['std']:.3f}")
    print(f"  95% CI for E[y|x]: ({pred['ci_95'][0]:.3f}, {pred['ci_95'][1]:.3f})")
    print(f"  95% PI for y: ({pred['pred_95'][0]:.3f}, {pred['pred_95'][1]:.3f})")
```

---

## 5. Model Comparison

### 5.1 WAIC (Widely Applicable Information Criterion)

```python
# Compare two models: simple vs quadratic regression

# Generate quadratic data
np.random.seed(42)
n = 100
x = np.random.uniform(0, 10, n)
y_quad = 2 + 0.5 * x + 0.1 * x**2 + np.random.normal(0, 1.5, n)

# Model 1: Linear
with pm.Model() as model_linear:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = beta0 + beta1 * x
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_quad)

    trace_linear = pm.sample(2000, tune=1000, cores=1, random_seed=42)
    # Compute log_likelihood for WAIC
    pm.compute_log_likelihood(trace_linear)

# Model 2: Quadratic
with pm.Model() as model_quadratic:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = beta0 + beta1 * x + beta2 * x**2
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_quad)

    trace_quadratic = pm.sample(2000, tune=1000, cores=1, random_seed=42)
    pm.compute_log_likelihood(trace_quadratic)

# WAIC comparison
print("=== WAIC Comparison ===")
waic_linear = az.waic(trace_linear)
waic_quadratic = az.waic(trace_quadratic)

print(f"Linear model WAIC: {waic_linear.waic:.2f}")
print(f"Quadratic model WAIC: {waic_quadratic.waic:.2f}")
print(f"\nLower WAIC is better")

# Model comparison
comparison = az.compare({
    'Linear': trace_linear,
    'Quadratic': trace_quadratic
}, ic='waic')
print("\n=== Model Comparison ===")
print(comparison)

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_compare(comparison, ax=ax)
plt.title('WAIC Model Comparison')
plt.tight_layout()
plt.show()
```

### 5.2 LOO-CV (Leave-One-Out Cross-Validation)

```python
# LOO-CV (Pareto Smoothed Importance Sampling LOO)
print("=== LOO-CV Comparison ===")

loo_linear = az.loo(trace_linear)
loo_quadratic = az.loo(trace_quadratic)

print(f"Linear model LOO: {loo_linear.loo:.2f}")
print(f"Quadratic model LOO: {loo_quadratic.loo:.2f}")

# LOO diagnostics
print("\n=== LOO Diagnostics: Pareto k ===")
print(f"Linear - observations with k > 0.7: {(loo_linear.pareto_k > 0.7).sum()}")
print(f"Quadratic - observations with k > 0.7: {(loo_quadratic.pareto_k > 0.7).sum()}")

# Comparison
comparison_loo = az.compare({
    'Linear': trace_linear,
    'Quadratic': trace_quadratic
}, ic='loo')
print("\n=== LOO Model Comparison ===")
print(comparison_loo)
```

### 5.3 Bayes Factor (Concept)

```python
def bayes_factor_example():
    """Explain Bayes factor concept"""

    print("=== Bayes Factor ===")
    print()
    print("Definition: BF₁₀ = P(D|M₁) / P(D|M₀)")
    print()
    print("Interpretation:")
    print("  BF > 100   : Decisive evidence (supports M₁)")
    print("  30 < BF < 100: Very strong evidence")
    print("  10 < BF < 30 : Strong evidence")
    print("  3 < BF < 10  : Moderate evidence")
    print("  1 < BF < 3   : Weak evidence")
    print("  BF = 1       : No evidence")
    print("  BF < 1       : Supports M₀")
    print()
    print("Note: Calculating Bayes factor is generally difficult")
    print("      In PyMC, methods like Savage-Dickey ratio can be used")

bayes_factor_example()
```

---

## 6. Posterior Predictive Checks

### 6.1 Concept

Posterior predictive distribution: Data expected to be generated by the model after learning from observed data

$$p(\tilde{y}|y) = \int p(\tilde{y}|\theta) p(\theta|y) d\theta$$

### 6.2 Implementation

```python
# Poisson regression example
np.random.seed(42)
n = 100
x = np.random.uniform(0, 5, n)
true_rate = np.exp(0.5 + 0.3 * x)
y_count = np.random.poisson(true_rate)

# Wrong model: assume normal distribution
with pm.Model() as wrong_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=5)
    beta1 = pm.Normal('beta1', mu=0, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = beta0 + beta1 * x
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_count)

    trace_wrong = pm.sample(2000, tune=1000, cores=1, random_seed=42)

    # Posterior predictive
    ppc_wrong = pm.sample_posterior_predictive(trace_wrong, random_seed=42)

# Correct model: assume Poisson
with pm.Model() as correct_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=5)
    beta1 = pm.Normal('beta1', mu=0, sigma=5)

    log_rate = beta0 + beta1 * x
    rate = pm.math.exp(log_rate)

    y_obs = pm.Poisson('y_obs', mu=rate, observed=y_count)

    trace_correct = pm.sample(2000, tune=1000, cores=1, random_seed=42)

    ppc_correct = pm.sample_posterior_predictive(trace_correct, random_seed=42)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Wrong model PPC
ax = axes[0, 0]
ppc_data_wrong = ppc_wrong.posterior_predictive['y_obs'].values.flatten()
ax.hist(y_count, bins=20, density=True, alpha=0.7, label='Observed')
ax.hist(ppc_data_wrong[:len(y_count)*10], bins=20, density=True, alpha=0.5, label='PPC')
ax.set_xlabel('y')
ax.set_ylabel('Density')
ax.set_title('Wrong Model (Normal): PPC')
ax.legend()

# Correct model PPC
ax = axes[0, 1]
ppc_data_correct = ppc_correct.posterior_predictive['y_obs'].values.flatten()
ax.hist(y_count, bins=20, density=True, alpha=0.7, label='Observed')
ax.hist(ppc_data_correct[:len(y_count)*10], bins=20, density=True, alpha=0.5, label='PPC')
ax.set_xlabel('y')
ax.set_ylabel('Density')
ax.set_title('Correct Model (Poisson): PPC')
ax.legend()

# Residual analysis - wrong model
ax = axes[1, 0]
mean_pred_wrong = ppc_wrong.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
residuals_wrong = y_count - mean_pred_wrong
ax.scatter(mean_pred_wrong, residuals_wrong, alpha=0.5)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Residual')
ax.set_title('Wrong Model: Residuals')

# Residual analysis - correct model
ax = axes[1, 1]
mean_pred_correct = ppc_correct.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
residuals_correct = y_count - mean_pred_correct
ax.scatter(mean_pred_correct, residuals_correct, alpha=0.5)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Residual')
ax.set_title('Correct Model: Residuals')

plt.tight_layout()
plt.show()
```

### 6.3 PPC Using ArviZ

```python
# ArviZ PPC plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Add observed data to InferenceData
idata_wrong = az.from_pymc3(trace_wrong, posterior_predictive=ppc_wrong)
idata_wrong.add_groups({'observed_data': {'y_obs': y_count}})

idata_correct = az.from_pymc3(trace_correct, posterior_predictive=ppc_correct)
idata_correct.add_groups({'observed_data': {'y_obs': y_count}})

# PPC plots
az.plot_ppc(idata_wrong, ax=axes[0], num_pp_samples=100)
axes[0].set_title('Wrong Model (Normal)')

az.plot_ppc(idata_correct, ax=axes[1], num_pp_samples=100)
axes[1].set_title('Correct Model (Poisson)')

plt.tight_layout()
plt.show()
```

---

## 7. Convergence Diagnostics

### 7.1 R-hat (Gelman-Rubin Statistic)

```python
def demonstrate_convergence_diagnostics():
    """Demonstrate convergence diagnostics"""

    # Sample with multiple chains
    np.random.seed(42)
    data = np.random.normal(5, 2, 50)

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=5)
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)

        # Multiple chains
        trace = pm.sample(2000, tune=1000, chains=4, cores=1, random_seed=42)

    # Calculate R-hat
    summary = az.summary(trace)
    print("=== Convergence Diagnostics ===")
    print(summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])

    print("\nInterpretation:")
    print("  R-hat < 1.01: Good convergence")
    print("  ESS_bulk > 400: Sufficient bulk ESS")
    print("  ESS_tail > 400: Sufficient tail ESS")

    return trace

trace_diagnostic = demonstrate_convergence_diagnostics()

# Trace plot by chain
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
az.plot_trace(trace_diagnostic, var_names=['mu', 'sigma'], axes=axes)
plt.suptitle('Multi-chain Trace Plot')
plt.tight_layout()
plt.show()
```

### 7.2 Autocorrelation Analysis

```python
# Autocorrelation plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

az.plot_autocorr(trace_diagnostic, var_names=['mu'], ax=axes[0])
axes[0].set_title('μ autocorrelation')

az.plot_autocorr(trace_diagnostic, var_names=['sigma'], ax=axes[1])
axes[1].set_title('σ autocorrelation')

plt.tight_layout()
plt.show()

print("Autocorrelation quickly decaying to 0 indicates good mixing")
```

### 7.3 Effective Sample Size (ESS)

```python
# ESS calculation
ess = az.ess(trace_diagnostic)
print("=== Effective Sample Size (ESS) ===")
for var in ['mu', 'sigma']:
    print(f"{var}: bulk={ess[var].values:.0f}, tail={az.ess(trace_diagnostic, method='tail')[var].values:.0f}")

print("\nRecommendation: ESS > 400 (minimum 100+)")
print("If ESS is low, need more samples or improve model/sampler")
```

---

## 8. Practice Example

### 8.1 Hierarchical Model

```python
# Student scores from multiple schools
np.random.seed(42)

n_schools = 8
n_students_per_school = np.random.randint(15, 30, n_schools)

# True school effects
true_school_mean = 70
true_school_std = 5
true_school_effects = np.random.normal(0, true_school_std, n_schools)
true_noise_std = 10

# Generate data
schools = []
scores = []
for i in range(n_schools):
    school_mean = true_school_mean + true_school_effects[i]
    student_scores = np.random.normal(school_mean, true_noise_std, n_students_per_school[i])
    schools.extend([i] * n_students_per_school[i])
    scores.extend(student_scores)

schools = np.array(schools)
scores = np.array(scores)

print(f"Number of schools: {n_schools}")
print(f"Total students: {len(scores)}")
print(f"Average score by school: {[scores[schools == i].mean():.1f for i in range(n_schools)]}")

# Hierarchical model
with pm.Model() as hierarchical_model:
    # Hyperparameters
    mu_school = pm.Normal('mu_school', mu=70, sigma=20)
    sigma_school = pm.HalfNormal('sigma_school', sigma=10)

    # School effects
    school_effect = pm.Normal('school_effect', mu=0, sigma=sigma_school, shape=n_schools)

    # Residual standard deviation
    sigma_residual = pm.HalfNormal('sigma_residual', sigma=10)

    # Expected values
    mu = mu_school + school_effect[schools]

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_residual, observed=scores)

    # Sampling
    trace_hier = pm.sample(3000, tune=1500, cores=1, random_seed=42)

# Results
print("\n=== Hierarchical Model Results ===")
print(az.summary(trace_hier, var_names=['mu_school', 'sigma_school', 'sigma_residual']))

# Compare school effects
school_effects_post = trace_hier.posterior['school_effect'].mean(dim=['chain', 'draw']).values
school_effects_ci = az.hdi(trace_hier, var_names=['school_effect'])['school_effect'].values

fig, ax = plt.subplots(figsize=(10, 6))
school_ids = np.arange(n_schools)

# Posterior mean and credible intervals
ax.errorbar(school_ids, school_effects_post,
            yerr=[school_effects_post - school_effects_ci[:, 0],
                  school_effects_ci[:, 1] - school_effects_post],
            fmt='o', capsize=5, label='Posterior (hierarchical)')

# True effects
ax.scatter(school_ids + 0.1, true_school_effects, marker='x', s=100,
           color='r', label='True effects', zorder=5)

# Individual school means (no pooling)
school_means = [scores[schools == i].mean() - true_school_mean for i in range(n_schools)]
ax.scatter(school_ids - 0.1, school_means, marker='s', s=60,
           color='g', alpha=0.5, label='No pooling (raw)')

ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('School ID')
ax.set_ylabel('School Effect')
ax.set_title('School Effects: Hierarchical Model vs Individual Estimation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

print("\nShrinkage effect:")
print("Hierarchical model shrinks extreme estimates toward the overall mean")
```

---

## 9. Practice Problems

### Problem 1: Implement MH Algorithm
Sample from the following target distribution using Metropolis-Hastings:
- Target: Gamma(5, 1)
- Proposal: Uniform ±0.5 from current value

### Problem 2: PyMC Modeling
Perform Bayesian inference with PyMC on the following data:
- 3 defects out of 20 products
- Beta(1,1) prior
- Calculate posterior mean and 95% credible interval for defect rate

### Problem 3: Model Comparison
Compare linear regression with polynomial regression (2nd, 3rd order):
- Generate data: y = 2 + 3x + 0.5x² + noise
- Select model using WAIC
- Perform posterior predictive checks

### Problem 4: Convergence Diagnostics
Sample with multiple chains and verify:
- R-hat < 1.01
- Sufficient ESS
- Trace plots mix well

---

## 10. Key Summary

### MCMC Essentials

1. **Metropolis-Hastings**: Repeated proposal-accept/reject
2. **Acceptance rate**: 20-50% is appropriate
3. **Burn-in**: Remove initial unstable samples

### PyMC Workflow

```python
with pm.Model() as model:
    # 1. Define priors
    theta = pm.Distribution('theta', ...)

    # 2. Define likelihood
    y = pm.Distribution('y', ..., observed=data)

    # 3. Sampling
    trace = pm.sample(...)

    # 4. Posterior predictive
    ppc = pm.sample_posterior_predictive(trace)

# 5. Diagnostics and summary
az.summary(trace)
az.plot_trace(trace)
az.plot_ppc(...)
```

### Convergence Diagnostics Checklist

- [ ] R-hat < 1.01
- [ ] ESS > 400
- [ ] Trace plots are stable
- [ ] Autocorrelation decays quickly
- [ ] Chains mix well

### Preview of Next Chapter

Chapter 10 **Time Series Analysis Basics** covers:
- Time series components (trend, seasonality, noise)
- Stationarity and unit root tests
- ACF/PACF analysis
- Time series decomposition

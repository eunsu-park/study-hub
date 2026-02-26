# 18. Introduction to Bayesian Statistics

[Previous: Generalized Linear Models](./17_Generalized_Linear_Models.md) | [Next: Bayesian Inference](./19_Bayesian_Inference.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare the frequentist and Bayesian interpretations of probability, parameters, and intervals
2. Apply Bayes' theorem to compute posterior probabilities for discrete and continuous problems
3. Demonstrate sequential Bayesian updating and explain how the posterior evolves with new data
4. Identify and implement conjugate prior pairs (Beta-Binomial, Normal-Normal, Gamma-Poisson)
5. Distinguish between MLE, MAP, and posterior mean estimation and explain when each is appropriate
6. Describe the connection between MAP estimation and regularization (Ridge, Lasso)
7. Design and interpret a Bayesian A/B test, including probability of improvement and expected lift

---

> **Updating beliefs with evidence.** Imagine a doctor who initially estimates a 1% chance you have a rare disease (prior). A positive test result arrives -- but the test has a 5% false-positive rate (likelihood). Bayes' theorem combines the prior belief with the new evidence to give an updated probability (posterior) of roughly 17%. The key insight: your final belief depends on *both* what you knew before *and* what the data tells you. This is Bayesian thinking -- a formal framework for learning from evidence.

Frequentist methods treat parameters as fixed unknowns and answer questions about data; Bayesian methods treat parameters as uncertain quantities and answer questions directly about those parameters. This shift in perspective lets you incorporate prior knowledge, produce intuitive probability statements about hypotheses, and update beliefs seamlessly as new data arrives. This lesson lays the conceptual and computational foundations of Bayesian thinking.

---

## 1. Frequentist vs Bayesian Paradigm

### 1.1 Differences in Probability Interpretation

| Perspective | Frequentist | Bayesian |
|-------------|-------------|----------|
| **Meaning of probability** | Long-run frequency (infinite repetition) | Degree of uncertainty (belief) |
| **Parameters** | Fixed unknown constants | Random variables (have distributions) |
| **Inference goal** | Point estimates, confidence intervals | Entire posterior distribution |
| **Prior information** | Not used | Reflected in prior distribution |
| **Interpretation** | "95% confidence interval" (95% contain true value in repeated sampling) | "95% probability that true value is in the interval" |

### 1.2 Comparison Example: Coin Flipping

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Data: 10 flips, 7 heads
n_trials = 10
n_heads = 7

# === Frequentist Approach ===
# Point estimate: MLE
p_mle = n_heads / n_trials
print(f"[Frequentist] MLE estimate: {p_mle:.3f}")

# 95% confidence interval (Wald interval)
se = np.sqrt(p_mle * (1 - p_mle) / n_trials)
ci_freq = (p_mle - 1.96 * se, p_mle + 1.96 * se)
print(f"[Frequentist] 95% confidence interval: ({ci_freq[0]:.3f}, {ci_freq[1]:.3f})")

# === Bayesian Approach ===
# Prior: Beta(1, 1) = Uniform(0, 1)
alpha_prior, beta_prior = 1, 1

# Posterior: Beta(alpha + heads, beta + tails)
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + (n_trials - n_heads)

posterior = stats.beta(alpha_post, beta_post)

# Posterior mean
p_bayes = posterior.mean()
print(f"\n[Bayesian] Posterior mean: {p_bayes:.3f}")

# 95% credible interval
ci_bayes = posterior.interval(0.95)
print(f"[Bayesian] 95% credible interval: ({ci_bayes[0]:.3f}, {ci_bayes[1]:.3f})")
```

### 1.3 Interpretation Differences

```python
# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

p_range = np.linspace(0, 1, 200)

# Frequentist: Likelihood function
likelihood = stats.binom.pmf(n_heads, n_trials, p_range)
likelihood = likelihood / likelihood.max()  # Normalize

axes[0].plot(p_range, likelihood, 'b-', lw=2, label='Likelihood')
axes[0].axvline(p_mle, color='r', linestyle='--', label=f'MLE = {p_mle:.2f}')
axes[0].axvspan(ci_freq[0], ci_freq[1], alpha=0.3, color='r', label='95% CI')
axes[0].set_xlabel('p (probability of heads)')
axes[0].set_ylabel('Normalized likelihood')
axes[0].set_title('Frequentist: Likelihood Function and Confidence Interval')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bayesian: Posterior distribution
prior_pdf = stats.beta(alpha_prior, beta_prior).pdf(p_range)
posterior_pdf = posterior.pdf(p_range)

axes[1].plot(p_range, prior_pdf, 'g--', lw=2, label='Prior: Beta(1,1)')
axes[1].plot(p_range, posterior_pdf, 'b-', lw=2, label=f'Posterior: Beta({alpha_post},{beta_post})')
axes[1].axvline(p_bayes, color='r', linestyle='--', label=f'Posterior mean = {p_bayes:.2f}')
axes[1].axvspan(ci_bayes[0], ci_bayes[1], alpha=0.3, color='b', label='95% Credible Interval')
axes[1].set_xlabel('p (probability of heads)')
axes[1].set_ylabel('Density')
axes[1].set_title('Bayesian: Prior and Posterior Distributions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. Bayes' Theorem

### 2.1 Derivation of Bayes' Theorem

Starting from the definition of conditional probability:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

$$P(B|A) = \frac{P(A \cap B)}{P(A)}$$

Eliminating P(A ∩ B) from these two equations:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Expressed in terms of parameter θ and data D:**

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

### 2.2 Terminology

| Term | Formula | Meaning |
|------|---------|---------|
| **Posterior** | P(θ\|D) | Belief about parameter after observing data |
| **Likelihood** | P(D\|θ) | Probability of observing data given parameter |
| **Prior** | P(θ) | Prior knowledge about parameter before data |
| **Evidence (Marginal likelihood)** | P(D) | Normalizing constant (integral over all θ) |

### 2.3 Implementing Bayes' Theorem

```python
def bayes_theorem_discrete(prior: dict, likelihood: dict) -> dict:
    """
    Calculate discrete Bayes' theorem

    Parameters:
    -----------
    prior : dict
        Prior probabilities {hypothesis: probability}
    likelihood : dict
        Likelihood {hypothesis: P(data|hypothesis)}

    Returns:
    --------
    posterior : dict
        Posterior probabilities {hypothesis: P(hypothesis|data)}
    """
    # Unnormalized posterior
    unnormalized = {h: prior[h] * likelihood[h] for h in prior}

    # Marginal likelihood (normalizing constant)
    evidence = sum(unnormalized.values())

    # Normalize
    posterior = {h: p / evidence for h, p in unnormalized.items()}

    return posterior

# Example: Disease diagnosis
# Hypotheses: Disease present (D), Disease absent (~D)
prior = {
    'D': 0.001,   # Prevalence 0.1%
    '~D': 0.999
}

# Likelihood when test is positive (+)
likelihood = {
    'D': 0.99,    # Sensitivity: P(+|D) = 99%
    '~D': 0.05    # False positive rate: P(+|~D) = 5%
}

posterior = bayes_theorem_discrete(prior, likelihood)

print("=== Disease Diagnosis Bayes Update ===")
print(f"Prior P(disease) = {prior['D']:.4f}")
print(f"Likelihood P(positive|disease) = {likelihood['D']:.2f}")
print(f"Likelihood P(positive|healthy) = {likelihood['~D']:.2f}")
print(f"\nPosterior after positive test:")
print(f"P(disease|positive) = {posterior['D']:.4f} ({posterior['D']*100:.2f}%)")
print(f"P(healthy|positive) = {posterior['~D']:.4f}")
```

### 2.4 Sequential Updates

```python
def sequential_bayes_update(prior_dist, data_points, likelihood_func):
    """
    Sequential Bayes update (when data arrives one at a time)

    Parameters:
    -----------
    prior_dist : scipy.stats distribution
        Initial prior distribution
    data_points : array-like
        Observed data
    likelihood_func : callable
        likelihood_func(x, theta) -> P(x|theta)
    """
    theta_range = np.linspace(0, 1, 1000)
    current_prior = prior_dist.pdf(theta_range)

    history = [current_prior.copy()]

    for x in data_points:
        # Calculate likelihood
        likelihood = likelihood_func(x, theta_range)

        # Unnormalized posterior
        unnormalized_posterior = current_prior * likelihood

        # Normalize (numerical integration)
        posterior = unnormalized_posterior / np.trapz(unnormalized_posterior, theta_range)

        history.append(posterior.copy())
        current_prior = posterior

    return theta_range, history

# Coin flip simulation
np.random.seed(123)
true_p = 0.7
data = np.random.binomial(1, true_p, size=20)  # 20 flips

# Bernoulli likelihood
def bernoulli_likelihood(x, theta):
    return theta**x * (1 - theta)**(1 - x)

# Sequential update
theta_range, history = sequential_bayes_update(
    stats.beta(1, 1),  # Uniform prior
    data,
    bernoulli_likelihood
)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

update_points = [0, 1, 5, 10, 15, 20]
for i, n in enumerate(update_points):
    axes[i].plot(theta_range, history[n], 'b-', lw=2)
    axes[i].axvline(true_p, color='r', linestyle='--', alpha=0.7, label=f'True p = {true_p}')
    axes[i].fill_between(theta_range, history[n], alpha=0.3)
    axes[i].set_xlabel('θ')
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'After n = {n} observations')
    axes[i].legend()
    axes[i].set_xlim(0, 1)

plt.tight_layout()
plt.suptitle('Sequential Bayes Update: Changes in Posterior Distribution', y=1.02)
plt.show()

print(f"Data: {data}")
print(f"Total heads: {data.sum()}, Total: {len(data)}")
```

---

## 3. Prior, Likelihood, and Posterior

### 3.1 Types of Prior Distributions

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
x = np.linspace(0, 1, 200)

# 1. Non-informative prior
ax = axes[0, 0]
uniform_prior = stats.beta(1, 1).pdf(x)
ax.plot(x, uniform_prior, 'b-', lw=2)
ax.set_title('Non-informative Prior\nBeta(1, 1) = Uniform(0, 1)')
ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.fill_between(x, uniform_prior, alpha=0.3)

# 2. Jeffreys prior
ax = axes[0, 1]
jeffreys_prior = stats.beta(0.5, 0.5).pdf(x)
ax.plot(x, jeffreys_prior, 'b-', lw=2)
ax.set_title('Jeffreys Prior\nBeta(0.5, 0.5)')
ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.fill_between(x, jeffreys_prior, alpha=0.3)

# 3. Weakly informative prior
ax = axes[0, 2]
weak_prior = stats.beta(2, 2).pdf(x)
ax.plot(x, weak_prior, 'b-', lw=2)
ax.set_title('Weakly Informative Prior\nBeta(2, 2)')
ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.fill_between(x, weak_prior, alpha=0.3)

# 4. Informative prior - centered at 0.3
ax = axes[1, 0]
informative_prior1 = stats.beta(3, 7).pdf(x)
ax.plot(x, informative_prior1, 'b-', lw=2)
ax.set_title('Informative Prior (low p)\nBeta(3, 7), mean=0.3')
ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.fill_between(x, informative_prior1, alpha=0.3)

# 5. Informative prior - centered at 0.7
ax = axes[1, 1]
informative_prior2 = stats.beta(7, 3).pdf(x)
ax.plot(x, informative_prior2, 'b-', lw=2)
ax.set_title('Informative Prior (high p)\nBeta(7, 3), mean=0.7')
ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.fill_between(x, informative_prior2, alpha=0.3)

# 6. Strong informative prior
ax = axes[1, 2]
strong_prior = stats.beta(30, 30).pdf(x)
ax.plot(x, strong_prior, 'b-', lw=2)
ax.set_title('Strong Informative Prior\nBeta(30, 30), mean=0.5')
ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.fill_between(x, strong_prior, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 Influence of Prior Distributions

```python
# Same data, different priors
n_trials, n_heads = 10, 7

priors = [
    ('Non-informative Beta(1,1)', 1, 1),
    ('Jeffreys Beta(0.5,0.5)', 0.5, 0.5),
    ('Weak Beta(2,2)', 2, 2),
    ('Informative Beta(5,5)', 5, 5),
    ('Strong Beta(20,20)', 20, 20),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
theta = np.linspace(0, 1, 200)

for i, (name, a, b) in enumerate(priors):
    # Prior
    prior = stats.beta(a, b)

    # Posterior
    a_post = a + n_heads
    b_post = b + (n_trials - n_heads)
    posterior = stats.beta(a_post, b_post)

    # Visualization
    axes[i].plot(theta, prior.pdf(theta), 'g--', lw=2, label='Prior')
    axes[i].plot(theta, posterior.pdf(theta), 'b-', lw=2, label='Posterior')
    axes[i].axvline(0.7, color='r', linestyle=':', alpha=0.7, label='MLE')
    axes[i].axvline(posterior.mean(), color='purple', linestyle='--',
                    alpha=0.7, label=f'Post Mean={posterior.mean():.2f}')
    axes[i].set_xlabel('θ')
    axes[i].set_title(f'{name}')
    axes[i].legend(fontsize=8)
    axes[i].set_xlim(0, 1)

plt.suptitle(f'Influence of Prior (Data: {n_heads}/{n_trials} successes)', y=1.02)
plt.tight_layout()
plt.show()
```

### 3.3 Likelihood Function

```python
def plot_likelihood_function(data, distribution='binomial'):
    """
    Visualize likelihood function
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if distribution == 'binomial':
        n = len(data)
        k = sum(data)
        theta_range = np.linspace(0.001, 0.999, 200)

        # Likelihood function: L(θ) = θ^k * (1-θ)^(n-k)
        likelihood = theta_range**k * (1 - theta_range)**(n - k)
        log_likelihood = k * np.log(theta_range) + (n - k) * np.log(1 - theta_range)

        # MLE
        mle = k / n

        # Likelihood function
        axes[0].plot(theta_range, likelihood, 'b-', lw=2)
        axes[0].axvline(mle, color='r', linestyle='--', label=f'MLE = {mle:.3f}')
        axes[0].set_xlabel('θ')
        axes[0].set_ylabel('L(θ)')
        axes[0].set_title('Likelihood Function')
        axes[0].legend()
        axes[0].fill_between(theta_range, likelihood, alpha=0.3)

        # Log-likelihood function
        axes[1].plot(theta_range, log_likelihood, 'b-', lw=2)
        axes[1].axvline(mle, color='r', linestyle='--', label=f'MLE = {mle:.3f}')
        axes[1].set_xlabel('θ')
        axes[1].set_ylabel('log L(θ)')
        axes[1].set_title('Log-Likelihood Function')
        axes[1].legend()

    plt.tight_layout()
    plt.show()

    return mle

# Example
data = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]  # 7/10 successes
mle = plot_likelihood_function(data)
print(f"Data: {data}")
print(f"Successes: {sum(data)}, Trials: {len(data)}")
print(f"MLE: {mle}")
```

---

## 4. Conjugate Priors

### 4.1 What are Conjugate Priors?

**Definition**: If the prior and posterior belong to the same distribution family, they are "conjugate".

**Advantages**:
- Closed-form posterior calculation possible
- No numerical methods needed
- Sequential updates are simple

### 4.2 Beta-Binomial Conjugate

**Model**:
- Likelihood: X ~ Binomial(n, θ)
- Prior: θ ~ Beta(α, β)
- Posterior: θ|X ~ Beta(α + x, β + n - x)

```python
class BetaBinomialModel:
    """Beta-Binomial conjugate model"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        """
        Parameters:
        -----------
        alpha_prior : float
            Alpha parameter of Beta prior
        beta_prior : float
            Beta parameter of Beta prior
        """
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.n_observations = 0
        self.n_successes = 0

    @property
    def prior(self):
        return stats.beta(self.alpha, self.beta)

    @property
    def posterior(self):
        return stats.beta(
            self.alpha + self.n_successes,
            self.beta + self.n_observations - self.n_successes
        )

    def update(self, n_successes, n_trials):
        """Update posterior with data"""
        self.n_successes += n_successes
        self.n_observations += n_trials
        return self

    def posterior_mean(self):
        return self.posterior.mean()

    def posterior_std(self):
        return self.posterior.std()

    def credible_interval(self, alpha=0.05):
        """Calculate credible interval"""
        return self.posterior.interval(1 - alpha)

    def posterior_predictive(self, n_trials):
        """
        Posterior predictive distribution: number of successes in new n_trials
        Beta-Binomial distribution
        """
        a = self.alpha + self.n_successes
        b = self.beta + self.n_observations - self.n_successes
        return stats.betabinom(n_trials, a, b)

    def summary(self):
        """Model summary"""
        print("=== Beta-Binomial Model Summary ===")
        print(f"Prior: Beta({self.alpha}, {self.beta})")
        print(f"Data: {self.n_successes} successes / {self.n_observations} trials")
        print(f"Posterior: Beta({self.alpha + self.n_successes}, "
              f"{self.beta + self.n_observations - self.n_successes})")
        print(f"Posterior Mean: {self.posterior_mean():.4f}")
        print(f"Posterior Std: {self.posterior_std():.4f}")
        ci = self.credible_interval()
        print(f"95% Credible Interval: ({ci[0]:.4f}, {ci[1]:.4f})")


# Example: Estimating click-through rate
model = BetaBinomialModel(alpha_prior=2, beta_prior=8)  # Prior: 20% CTR

# 1st data: 35 clicks out of 100
model.update(35, 100)
print("After 1st data:")
model.summary()

# 2nd data: 22 clicks out of 50
model.update(22, 50)
print("\nAfter 2nd data:")
model.summary()

# Visualization
theta = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(10, 6))

# Prior
prior = stats.beta(2, 8)
ax.plot(theta, prior.pdf(theta), 'g--', lw=2, label='Prior: Beta(2,8)')

# Intermediate posterior (after 1st data)
post1 = stats.beta(2 + 35, 8 + 100 - 35)
ax.plot(theta, post1.pdf(theta), 'orange', lw=2, linestyle='-.',
        label='After 1st data: Beta(37, 73)')

# Final posterior
ax.plot(theta, model.posterior.pdf(theta), 'b-', lw=2,
        label=f'Final Posterior: Beta(59, 95)')

ax.fill_between(theta, model.posterior.pdf(theta), alpha=0.3)
ax.axvline(model.posterior_mean(), color='r', linestyle=':',
           label=f'Posterior Mean = {model.posterior_mean():.3f}')

ax.set_xlabel('θ (click-through rate)')
ax.set_ylabel('Density')
ax.set_title('Beta-Binomial Conjugate: Click-through Rate Estimation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 4.3 Normal-Normal Conjugate (Known Variance)

**Model**:
- Likelihood: X ~ N(μ, σ²)  (σ² known)
- Prior: μ ~ N(μ₀, σ₀²)
- Posterior: μ|X ~ N(μₙ, σₙ²)

$$\mu_n = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

$$\sigma_n^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

```python
class NormalNormalModel:
    """Normal-Normal conjugate model (known variance)"""

    def __init__(self, mu_prior=0, sigma_prior=10, sigma_known=1):
        """
        Parameters:
        -----------
        mu_prior : float
            Prior mean
        sigma_prior : float
            Prior standard deviation
        sigma_known : float
            Known data standard deviation
        """
        self.mu_0 = mu_prior
        self.sigma_0 = sigma_prior
        self.sigma = sigma_known
        self.data = []

    @property
    def n(self):
        return len(self.data)

    @property
    def x_bar(self):
        return np.mean(self.data) if self.data else 0

    @property
    def prior(self):
        return stats.norm(self.mu_0, self.sigma_0)

    @property
    def posterior_precision(self):
        """Posterior precision (inverse of variance)"""
        return 1/self.sigma_0**2 + self.n/self.sigma**2

    @property
    def posterior_variance(self):
        return 1 / self.posterior_precision

    @property
    def posterior_mean(self):
        if self.n == 0:
            return self.mu_0
        return (self.mu_0/self.sigma_0**2 + self.n*self.x_bar/self.sigma**2) / \
               self.posterior_precision

    @property
    def posterior_std(self):
        return np.sqrt(self.posterior_variance)

    @property
    def posterior(self):
        return stats.norm(self.posterior_mean, self.posterior_std)

    def update(self, data):
        """Add data"""
        if isinstance(data, (list, np.ndarray)):
            self.data.extend(data)
        else:
            self.data.append(data)
        return self

    def credible_interval(self, alpha=0.05):
        return self.posterior.interval(1 - alpha)

    def summary(self):
        print("=== Normal-Normal Model Summary ===")
        print(f"Prior: N({self.mu_0}, {self.sigma_0}²)")
        print(f"Known σ: {self.sigma}")
        print(f"Data: n={self.n}, x̄={self.x_bar:.4f}" if self.n > 0 else "Data: none")
        print(f"Posterior: N({self.posterior_mean:.4f}, {self.posterior_std:.4f}²)")
        ci = self.credible_interval()
        print(f"95% Credible Interval: ({ci[0]:.4f}, {ci[1]:.4f})")


# Example: Body temperature measurement
# Prior knowledge: Average temperature about 36.5°C, uncertainty σ=0.5
# Measurement error σ=0.2 (known)

model = NormalNormalModel(mu_prior=36.5, sigma_prior=0.5, sigma_known=0.2)
print("Prior:")
model.summary()

# Data collection: Patient temperature measurements
measurements = [37.1, 37.3, 36.9, 37.2, 37.0]
model.update(measurements)
print("\nAfter data update:")
model.summary()

# Visualization
mu_range = np.linspace(35.5, 38, 200)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mu_range, model.prior.pdf(mu_range), 'g--', lw=2,
        label=f'Prior: N({model.mu_0}, {model.sigma_0}²)')
ax.plot(mu_range, model.posterior.pdf(mu_range), 'b-', lw=2,
        label=f'Posterior: N({model.posterior_mean:.2f}, {model.posterior_std:.3f}²)')
ax.fill_between(mu_range, model.posterior.pdf(mu_range), alpha=0.3)

# Show individual measurements
for i, x in enumerate(measurements):
    ax.axvline(x, color='orange', linestyle=':', alpha=0.5,
               label='Measurements' if i == 0 else '')

ax.axvline(np.mean(measurements), color='r', linestyle='--',
           label=f'Sample Mean = {np.mean(measurements):.2f}')

ax.set_xlabel('μ (body temperature)')
ax.set_ylabel('Density')
ax.set_title('Normal-Normal Conjugate: Body Temperature Estimation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 4.4 Gamma-Poisson Conjugate

**Model**:
- Likelihood: X ~ Poisson(λ)
- Prior: λ ~ Gamma(α, β)
- Posterior: λ|X ~ Gamma(α + Σxᵢ, β + n)

```python
class GammaPoissonModel:
    """Gamma-Poisson conjugate model"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        """
        Parameters:
        -----------
        alpha_prior : float (shape)
        beta_prior : float (rate)
        """
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.data = []

    @property
    def n(self):
        return len(self.data)

    @property
    def sum_x(self):
        return sum(self.data)

    @property
    def prior(self):
        return stats.gamma(self.alpha, scale=1/self.beta)

    @property
    def posterior(self):
        alpha_post = self.alpha + self.sum_x
        beta_post = self.beta + self.n
        return stats.gamma(alpha_post, scale=1/beta_post)

    def update(self, data):
        if isinstance(data, (list, np.ndarray)):
            self.data.extend(data)
        else:
            self.data.append(data)
        return self

    def posterior_mean(self):
        return (self.alpha + self.sum_x) / (self.beta + self.n)

    def posterior_std(self):
        return self.posterior.std()

    def credible_interval(self, alpha=0.05):
        return self.posterior.interval(1 - alpha)

    def posterior_predictive(self):
        """
        Posterior predictive distribution: Negative Binomial
        """
        alpha_post = self.alpha + self.sum_x
        beta_post = self.beta + self.n
        p = beta_post / (beta_post + 1)
        return stats.nbinom(alpha_post, p)

    def summary(self):
        print("=== Gamma-Poisson Model Summary ===")
        print(f"Prior: Gamma({self.alpha}, {self.beta})")
        print(f"Data: n={self.n}, Σx={self.sum_x}")
        alpha_post = self.alpha + self.sum_x
        beta_post = self.beta + self.n
        print(f"Posterior: Gamma({alpha_post}, {beta_post})")
        print(f"Posterior Mean (λ): {self.posterior_mean():.4f}")
        print(f"Posterior Std: {self.posterior_std():.4f}")
        ci = self.credible_interval()
        print(f"95% Credible Interval: ({ci[0]:.4f}, {ci[1]:.4f})")


# Example: Call center call count estimation
# Prior: Expect about 10 calls per hour, with uncertainty
model = GammaPoissonModel(alpha_prior=10, beta_prior=1)  # Mean 10, variance 10
print("Prior:")
model.summary()

# Call counts over 5 hours
calls = [8, 12, 15, 9, 11]
model.update(calls)
print("\nAfter data:")
model.summary()

# Visualization
lambda_range = np.linspace(0, 25, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Posterior
ax = axes[0]
ax.plot(lambda_range, model.prior.pdf(lambda_range), 'g--', lw=2, label='Prior')
ax.plot(lambda_range, model.posterior.pdf(lambda_range), 'b-', lw=2, label='Posterior')
ax.fill_between(lambda_range, model.posterior.pdf(lambda_range), alpha=0.3)
ax.axvline(model.posterior_mean(), color='r', linestyle='--',
           label=f'Post Mean = {model.posterior_mean():.2f}')
ax.axvline(np.mean(calls), color='orange', linestyle=':',
           label=f'Sample Mean = {np.mean(calls):.1f}')
ax.set_xlabel('λ (calls per hour)')
ax.set_ylabel('Density')
ax.set_title('Gamma-Poisson Conjugate: Call Rate Estimation')
ax.legend()
ax.grid(True, alpha=0.3)

# Posterior predictive distribution
ax = axes[1]
x_pred = np.arange(0, 30)
pred_dist = model.posterior_predictive()
ax.bar(x_pred, pred_dist.pmf(x_pred), alpha=0.6, label='Posterior Predictive')
ax.axvline(pred_dist.mean(), color='r', linestyle='--',
           label=f'Expected = {pred_dist.mean():.1f}')
ax.set_xlabel('Next hour call count')
ax.set_ylabel('Probability')
ax.set_title('Posterior Predictive Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.5 Conjugate Prior Summary Table

| Likelihood | Prior | Posterior | Update Rule |
|------------|-------|-----------|-------------|
| Binomial(n, p) | Beta(α, β) | Beta(α+x, β+n-x) | α←α+x, β←β+(n-x) |
| Poisson(λ) | Gamma(α, β) | Gamma(α+Σx, β+n) | α←α+Σx, β←β+n |
| Normal(μ, σ²) | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) | Precision-weighted average |
| Exponential(λ) | Gamma(α, β) | Gamma(α+n, β+Σx) | α←α+n, β←β+Σx |
| Multinomial | Dirichlet | Dirichlet | α←α+counts |

---

## 5. MAP Estimation (Maximum A Posteriori)

### 5.1 MAP vs MLE

**MLE**: Maximize likelihood
$$\hat{\theta}_{MLE} = \arg\max_\theta P(D|\theta)$$

**MAP**: Maximize posterior
$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta)P(\theta)$$

```python
def compare_mle_map(n_trials, n_successes, alpha_prior, beta_prior):
    """Compare MLE and MAP"""

    # MLE
    mle = n_successes / n_trials

    # MAP for Beta-Binomial
    # posterior: Beta(alpha + k, beta + n - k)
    # mode of Beta(a, b) = (a - 1) / (a + b - 2) when a, b > 1
    a = alpha_prior + n_successes
    b = beta_prior + (n_trials - n_successes)

    if a > 1 and b > 1:
        map_est = (a - 1) / (a + b - 2)
    else:
        # Mode at boundary
        map_est = a / (a + b)  # Use posterior mean

    # Posterior mean
    posterior_mean = a / (a + b)

    return mle, map_est, posterior_mean

# Compare across various priors and data sizes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scenarios = [
    (10, 7, 'n=10, k=7'),
    (100, 70, 'n=100, k=70'),
    (10, 3, 'n=10, k=3'),
    (100, 30, 'n=100, k=30'),
]

priors = [
    ('Uniform', 1, 1),
    ('Weak', 2, 2),
    ('Informative 0.5', 10, 10),
    ('Strong 0.5', 50, 50),
]

for i, (n, k, title) in enumerate(scenarios):
    ax = axes.flatten()[i]

    x = np.arange(len(priors))
    width = 0.25

    mles, maps, means = [], [], []
    for name, a, b in priors:
        mle, map_est, post_mean = compare_mle_map(n, k, a, b)
        mles.append(mle)
        maps.append(map_est)
        means.append(post_mean)

    ax.bar(x - width, mles, width, label='MLE', alpha=0.8)
    ax.bar(x, maps, width, label='MAP', alpha=0.8)
    ax.bar(x + width, means, width, label='Posterior Mean', alpha=0.8)

    ax.axhline(k/n, color='r', linestyle='--', alpha=0.5, label=f'True ratio = {k/n:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in priors], rotation=15)
    ax.set_ylabel('Estimate')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('MLE vs MAP vs Posterior Mean: Influence of Prior', y=1.02)
plt.tight_layout()
plt.show()
```

### 5.2 Numerical MAP Estimation

```python
from scipy.optimize import minimize_scalar, minimize

def map_estimation_normal(data, prior_mean, prior_std, known_std):
    """
    MAP estimation for normal data (numerical method)
    """
    n = len(data)
    x_bar = np.mean(data)

    # Negative log posterior (to minimize)
    def neg_log_posterior(mu):
        # Log likelihood
        log_likelihood = -n * np.log(known_std) - \
                         0.5 * np.sum((data - mu)**2) / known_std**2
        # Log prior
        log_prior = -0.5 * ((mu - prior_mean)**2) / prior_std**2

        return -(log_likelihood + log_prior)

    # Optimization
    result = minimize_scalar(neg_log_posterior, bounds=(x_bar - 3*known_std, x_bar + 3*known_std))

    return result.x

# Example
np.random.seed(42)
true_mu = 5
data = np.random.normal(true_mu, 1, 10)

# MAP with various priors
priors = [
    (5, 100),   # Very weak prior
    (5, 2),     # Weak prior
    (0, 1),     # Wrong strong prior
    (5, 0.5),   # Correct strong prior
]

print(f"True μ: {true_mu}")
print(f"Sample mean: {np.mean(data):.4f}")
print(f"Sample size: {len(data)}")
print("\nMAP estimates with different priors:")

for prior_mean, prior_std in priors:
    map_est = map_estimation_normal(data, prior_mean, prior_std, known_std=1)
    print(f"Prior N({prior_mean}, {prior_std}²): MAP = {map_est:.4f}")
```

### 5.3 MAP and Regularization Connection

```python
# MAP estimation is equivalent to regularization
# Beta(α, β) prior for Binomial → Similar effect to L2 regularization

def show_map_regularization_connection():
    """Connection between MAP and regularization"""

    print("=== Relationship between MAP and Regularization ===")
    print()
    print("1. In logistic regression:")
    print("   - Normal prior N(0, σ²) → L2 regularization (Ridge)")
    print("   - Laplace prior → L1 regularization (Lasso)")
    print()
    print("2. In linear regression:")
    print("   - MAP with N(0, σ²) prior ≡ Ridge regression")
    print("   - λ = σ²_noise / σ²_prior")
    print()
    print("3. Bayesian perspective:")
    print("   - Regularization strength = Confidence in prior")
    print("   - Strong prior → Strong regularization")
    print("   - Non-informative prior → No regularization (MLE)")

show_map_regularization_connection()

# Visual example
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ridge vs MAP
theta = np.linspace(-3, 3, 200)

# Ridge perspective: L2 penalty
ridge_penalty = theta**2
axes[0].plot(theta, ridge_penalty, 'b-', lw=2, label='L2 penalty: λθ²')
axes[0].set_xlabel('θ')
axes[0].set_ylabel('Penalty')
axes[0].set_title('Ridge Regression: L2 Penalty')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAP perspective: Normal prior
prior_pdf = stats.norm(0, 1).pdf(theta)
neg_log_prior = -np.log(prior_pdf + 1e-10)
axes[1].plot(theta, neg_log_prior, 'r-', lw=2, label='-log P(θ) for N(0, 1)')
axes[1].set_xlabel('θ')
axes[1].set_ylabel('-log Prior')
axes[1].set_title('MAP: Negative Log Prior')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6. Practice Examples

### 6.1 Bayesian A/B Test

```python
class BayesianABTest:
    """Bayesian A/B Test"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def fit(self, successes_A, trials_A, successes_B, trials_B):
        """Fit data"""
        self.successes_A = successes_A
        self.trials_A = trials_A
        self.successes_B = successes_B
        self.trials_B = trials_B

        # Posteriors
        self.posterior_A = stats.beta(
            self.alpha_prior + successes_A,
            self.beta_prior + trials_A - successes_A
        )
        self.posterior_B = stats.beta(
            self.alpha_prior + successes_B,
            self.beta_prior + trials_B - successes_B
        )

    def prob_B_better(self, n_samples=100000):
        """Calculate P(B > A) by simulation"""
        samples_A = self.posterior_A.rvs(n_samples)
        samples_B = self.posterior_B.rvs(n_samples)
        return (samples_B > samples_A).mean()

    def expected_lift(self, n_samples=100000):
        """Expected lift"""
        samples_A = self.posterior_A.rvs(n_samples)
        samples_B = self.posterior_B.rvs(n_samples)
        lift = (samples_B - samples_A) / samples_A
        return lift.mean(), np.percentile(lift, [2.5, 97.5])

    def summary(self):
        print("=== Bayesian A/B Test Results ===")
        print(f"\nGroup A: {self.successes_A}/{self.trials_A} "
              f"({self.successes_A/self.trials_A*100:.1f}%)")
        print(f"Group B: {self.successes_B}/{self.trials_B} "
              f"({self.successes_B/self.trials_B*100:.1f}%)")

        print(f"\nPosterior Mean A: {self.posterior_A.mean():.4f}")
        print(f"Posterior Mean B: {self.posterior_B.mean():.4f}")

        prob_b_better = self.prob_B_better()
        print(f"\nP(B > A): {prob_b_better:.4f} ({prob_b_better*100:.1f}%)")

        lift_mean, lift_ci = self.expected_lift()
        print(f"Expected Lift: {lift_mean*100:.2f}%")
        print(f"95% CI for Lift: ({lift_ci[0]*100:.2f}%, {lift_ci[1]*100:.2f}%)")

    def plot(self):
        """Visualize results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        p_range = np.linspace(0, 1, 200)

        # Compare posteriors
        ax = axes[0]
        ax.plot(p_range, self.posterior_A.pdf(p_range), 'b-', lw=2, label='A')
        ax.plot(p_range, self.posterior_B.pdf(p_range), 'r-', lw=2, label='B')
        ax.fill_between(p_range, self.posterior_A.pdf(p_range), alpha=0.3)
        ax.fill_between(p_range, self.posterior_B.pdf(p_range), alpha=0.3)
        ax.set_xlabel('Conversion rate')
        ax.set_ylabel('Density')
        ax.set_title('Posterior distributions of conversion rates')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Difference distribution
        ax = axes[1]
        samples_A = self.posterior_A.rvs(50000)
        samples_B = self.posterior_B.rvs(50000)
        diff = samples_B - samples_A
        ax.hist(diff, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='No difference')
        ax.axvline(diff.mean(), color='g', linestyle='-',
                   label=f'Mean diff = {diff.mean():.4f}')
        ax.set_xlabel('B - A')
        ax.set_ylabel('Density')
        ax.set_title('Conversion rate difference distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Lift distribution
        ax = axes[2]
        lift = (samples_B - samples_A) / samples_A
        lift = lift[(lift > -1) & (lift < 2)]  # Remove extremes
        ax.hist(lift, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='No lift')
        ax.axvline(np.median(lift), color='g', linestyle='-',
                   label=f'Median = {np.median(lift)*100:.1f}%')
        ax.set_xlabel('Lift (B-A)/A')
        ax.set_ylabel('Density')
        ax.set_title('Lift distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example: Website button color A/B test
ab_test = BayesianABTest(alpha_prior=1, beta_prior=1)
ab_test.fit(
    successes_A=48, trials_A=500,   # Original button: 48/500 clicks
    successes_B=63, trials_B=500    # New button: 63/500 clicks
)
ab_test.summary()
ab_test.plot()
```

### 6.2 Bayesian Approach to Quality Control

```python
def bayesian_quality_control():
    """Bayesian analysis for manufacturing quality control"""

    # Scenario: Defect rate estimation
    # Prior knowledge: Defect rate about 5% from past experience, moderate confidence

    # Prior: Beta(2, 38) → Mean 5%, equivalent information from sample size 40
    alpha_prior = 2
    beta_prior = 38
    prior = stats.beta(alpha_prior, beta_prior)

    print("=== Bayesian Analysis for Manufacturing Quality Control ===")
    print(f"Prior: Beta({alpha_prior}, {beta_prior})")
    print(f"Prior mean (expected defect rate): {prior.mean()*100:.1f}%")
    print(f"Prior 95% interval: ({prior.ppf(0.025)*100:.2f}%, {prior.ppf(0.975)*100:.2f}%)")

    # New data: 8 defects out of 100 samples
    n_samples = 100
    n_defects = 8

    # Posterior
    alpha_post = alpha_prior + n_defects
    beta_post = beta_prior + (n_samples - n_defects)
    posterior = stats.beta(alpha_post, beta_post)

    print(f"\nData: {n_defects}/{n_samples} defects")
    print(f"Posterior: Beta({alpha_post}, {beta_post})")
    print(f"Posterior mean (estimated defect rate): {posterior.mean()*100:.2f}%")
    ci = posterior.interval(0.95)
    print(f"Posterior 95% credible interval: ({ci[0]*100:.2f}%, {ci[1]*100:.2f}%)")

    # Decision: Probability of defect rate exceeding 10%
    prob_exceed_10 = 1 - posterior.cdf(0.10)
    print(f"\nP(defect rate > 10%): {prob_exceed_10*100:.2f}%")

    if prob_exceed_10 > 0.05:
        print("⚠️ Recommendation: Quality inspection needed (>10% probability exceeds 5%)")
    else:
        print("✓ Normal: Quality standard met")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    p_range = np.linspace(0, 0.25, 200)

    # Prior/posterior comparison
    ax = axes[0]
    ax.plot(p_range, prior.pdf(p_range), 'g--', lw=2, label='Prior')
    ax.plot(p_range, posterior.pdf(p_range), 'b-', lw=2, label='Posterior')
    ax.fill_between(p_range, posterior.pdf(p_range), alpha=0.3)
    ax.axvline(0.10, color='r', linestyle=':', lw=2, label='Threshold (10%)')
    ax.axvline(posterior.mean(), color='purple', linestyle='--',
               label=f'Post Mean = {posterior.mean()*100:.1f}%')
    ax.set_xlabel('Defect rate')
    ax.set_ylabel('Density')
    ax.set_title('Prior/Posterior distributions of defect rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Probability of exceeding threshold
    ax = axes[1]
    thresholds = np.linspace(0.01, 0.20, 50)
    exceed_probs = [1 - posterior.cdf(t) for t in thresholds]
    ax.plot(thresholds * 100, exceed_probs, 'b-', lw=2)
    ax.axvline(10, color='r', linestyle='--', label='Current threshold (10%)')
    ax.axhline(0.05, color='orange', linestyle=':', label='Significance level (5%)')
    ax.set_xlabel('Defect rate threshold (%)')
    ax.set_ylabel('Probability of exceedance')
    ax.set_title('Probability of exceeding threshold by defect rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

bayesian_quality_control()
```

---

## 7. Practice Problems

### Problem 1: Apply Bayes' Theorem
You want to implement a spam filter.
- P(spam) = 0.3 (30% are spam)
- P("free"|spam) = 0.8 (80% of spam contains "free")
- P("free"|legitimate) = 0.1 (10% of legitimate emails contain "free")

Calculate the probability that an email containing "free" is spam.

### Problem 2: Choose Conjugate Prior
Select appropriate conjugate priors for the following situations and explain:
1. Estimating website click-through rate
2. Estimating call center calls per hour
3. Estimating average product weight (variance is known)

### Problem 3: Prior Sensitivity Analysis
Perform MAP estimation with the following priors for the same data (100 trials, 30 successes):
1. Beta(1, 1)
2. Beta(5, 5)
3. Beta(1, 9)
4. Beta(9, 1)

Compare MAP estimates with MLE for each case and analyze the influence of priors on results.

### Problem 4: Sequential Update
10 customers visit daily, and purchase data for 3 days is [3, 5, 4].
Starting with Beta(1, 1) prior, update posterior with each day's data.
Find the mean and 95% credible interval of the final posterior.

---

## 8. Key Summary

### Core Concepts of Bayesian Statistics

1. **Meaning of probability**: Degree of uncertainty (belief)
2. **Bayes' theorem**: Prior knowledge + Data → Posterior knowledge
3. **Conjugate priors**: Closed-form posterior calculation possible

### Major Conjugate Pairs

```
Binomial + Beta → Beta
Poisson + Gamma → Gamma
Normal(μ, σ²_known) + Normal → Normal
```

### Comparison of Estimation Methods

| Method | Objective | Feature |
|--------|-----------|---------|
| MLE | max P(D\|θ) | Uses data only |
| MAP | max P(θ\|D) | Reflects prior |
| Posterior mean | E[θ\|D] | Reflects full uncertainty |

### Preview of Next Chapter

Chapter 09 **Bayesian Inference** covers:
- MCMC methods (Metropolis-Hastings, Gibbs Sampling)
- Bayesian modeling with PyMC
- Bayesian regression analysis
- Model comparison and selection

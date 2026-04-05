# 01. Bayesian Thinking

[Next: Probabilistic Graphical Models](./02_Probabilistic_Graphical_Models.md)

---

> **Framework Note**: This lesson uses NumPy and SciPy for hands-on Bayesian computation.
> Later lessons introduce PyMC, Stan, and Pyro for scalable inference.
>
> Installation: `pip install numpy scipy matplotlib`

## Learning Objectives

- Understand Bayes' theorem and its components (prior, likelihood, posterior)
- Distinguish between frequentist and Bayesian interpretations of probability
- Implement Bayesian updating with conjugate priors
- Visualize the effect of prior strength on posterior inference

---

## 1. Two Schools of Probability

Probability has two major interpretations, and understanding the distinction is essential before diving into probabilistic programming.

### 1.1 Frequentist Interpretation

In the frequentist view, probability is the **long-run frequency** of an event. Parameters are fixed but unknown constants; only data are random. Inference relies on sampling distributions and p-values.

```python
import numpy as np

# Frequentist: estimate p(heads) by repeating the experiment
np.random.seed(42)
n_flips = 10000
flips = np.random.binomial(1, 0.7, size=n_flips)
freq_estimate = flips.mean()
print(f"Frequentist estimate of p(heads): {freq_estimate:.4f}")
# Close to 0.7 as n_flips → ∞
```

### 1.2 Bayesian Interpretation

In the Bayesian view, probability represents a **degree of belief**. Parameters themselves have probability distributions. We start with a prior belief, observe data, and update to a posterior belief.

```python
# Bayesian: we express uncertainty about p(heads) as a distribution
# Before seeing any data, we might believe p ~ Uniform(0, 1)
# After seeing data, we update our belief using Bayes' theorem
```

### 1.3 Key Philosophical Differences

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Probability | Long-run frequency | Degree of belief |
| Parameters | Fixed, unknown | Random variables |
| Inference | MLE, confidence intervals | Posterior distributions |
| Prior information | Not used | Explicitly incorporated |
| Small samples | Can be unreliable | Naturally regularized by prior |

---

## 2. Bayes' Theorem

The cornerstone of Bayesian inference is Bayes' theorem, which tells us how to update our beliefs in light of new evidence.

### 2.1 The Formula

$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

Where:
- $P(\theta | D)$: **Posterior** — our updated belief about $\theta$ after seeing data $D$
- $P(D | \theta)$: **Likelihood** — probability of the data given $\theta$
- $P(\theta)$: **Prior** — our belief about $\theta$ before seeing data
- $P(D)$: **Evidence** (marginal likelihood) — normalizing constant

### 2.2 The Normalizing Constant

The evidence $P(D)$ ensures the posterior integrates to 1:

$$P(D) = \int P(D | \theta) P(\theta) \, d\theta$$

In practice, this integral is often intractable — which is why we need MCMC and variational inference (covered in later lessons).

### 2.3 Proportionality Form

Since $P(D)$ is a constant with respect to $\theta$, we often write:

$$P(\theta | D) \propto P(D | \theta) \cdot P(\theta)$$

**Posterior ∝ Likelihood × Prior**

This is the most important equation in Bayesian statistics.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Example: Coin flip inference
# Prior: Beta(2, 2) — slight belief that coin is fair
# Likelihood: Binomial
# Posterior: Beta(2 + heads, 2 + tails)  [conjugacy!]

alpha_prior, beta_prior = 2, 2
n_heads, n_tails = 7, 3

alpha_post = alpha_prior + n_heads   # 9
beta_post = beta_prior + n_tails     # 5

theta = np.linspace(0, 1, 1000)
prior = stats.beta.pdf(theta, alpha_prior, beta_prior)
likelihood = stats.binom.pmf(n_heads, n_heads + n_tails, theta)
posterior = stats.beta.pdf(theta, alpha_post, beta_post)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, dist, title, color in zip(
    axes,
    [prior, likelihood, posterior],
    ["Prior Beta(2,2)", "Likelihood (7H, 3T)", "Posterior Beta(9,5)"],
    ["blue", "green", "red"]
):
    ax.plot(theta, dist, color=color, linewidth=2)
    ax.fill_between(theta, dist, alpha=0.3, color=color)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("θ (probability of heads)")
    ax.set_ylabel("Density")
plt.tight_layout()
plt.savefig("bayesian_updating.png", dpi=100)
plt.show()
```

---

## 3. Prior Distributions

The prior encodes what we know (or assume) before observing data. Choosing an appropriate prior is a core skill in Bayesian modeling.

### 3.1 Informative Priors

Informative priors encode specific domain knowledge. They can significantly influence the posterior, especially with small datasets.

```python
# Informative prior: We know this coin is roughly fair
# Beta(50, 50) is concentrated around 0.5
alpha_info, beta_info = 50, 50
prior_info = stats.beta.pdf(theta, alpha_info, beta_info)

# Even with 7/10 heads, posterior stays closer to 0.5
alpha_post_info = alpha_info + n_heads   # 57
beta_post_info = beta_info + n_tails     # 53
posterior_info = stats.beta.pdf(theta, alpha_post_info, beta_post_info)
print(f"Posterior mean (informative prior): {alpha_post_info / (alpha_post_info + beta_post_info):.4f}")
# ~0.518, pulled toward 0.5
```

### 3.2 Weakly Informative Priors

Weakly informative priors constrain parameters to reasonable ranges without committing to specific values. This is the recommended default in modern Bayesian practice.

```python
# Weakly informative: Normal(0, 10) for regression coefficients
# Rules out extreme values but doesn't favor any particular value strongly
x = np.linspace(-30, 30, 1000)
weak_prior = stats.norm.pdf(x, 0, 10)
```

### 3.3 Non-informative (Flat) Priors

Non-informative priors attempt to "let the data speak." Common choices include uniform distributions and Jeffreys priors.

```python
# Flat prior: Beta(1, 1) = Uniform(0, 1)
# Jeffreys prior for Bernoulli: Beta(0.5, 0.5)
flat_prior = stats.beta.pdf(theta, 1, 1)
jeffreys_prior = stats.beta.pdf(theta, 0.5, 0.5)
```

### 3.4 Prior Sensitivity Analysis

A responsible Bayesian practitioner always checks how sensitive conclusions are to the prior choice.

```python
def prior_sensitivity(n_heads, n_tails, priors):
    """Compare posteriors under different priors."""
    theta = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (a, b) in priors.items():
        a_post = a + n_heads
        b_post = b + n_tails
        posterior = stats.beta.pdf(theta, a_post, b_post)
        mean = a_post / (a_post + b_post)
        ax.plot(theta, posterior, label=f"{name}: post mean={mean:.3f}", linewidth=2)

    ax.set_xlabel("θ")
    ax.set_ylabel("Posterior density")
    ax.set_title(f"Prior Sensitivity (data: {n_heads}H, {n_tails}T)")
    ax.legend()
    plt.tight_layout()
    return fig

priors = {
    "Flat Beta(1,1)": (1, 1),
    "Jeffreys Beta(0.5,0.5)": (0.5, 0.5),
    "Weak Beta(2,2)": (2, 2),
    "Informative Beta(50,50)": (50, 50),
}
fig = prior_sensitivity(7, 3, priors)
plt.savefig("prior_sensitivity.png", dpi=100)
plt.show()
```

---

## 4. Likelihood Functions

The likelihood function measures how well the data are explained by different parameter values.

### 4.1 Common Likelihood Functions

```python
# Bernoulli / Binomial likelihood
def binomial_likelihood(theta, n_heads, n_total):
    """P(data | theta) for coin flips."""
    from scipy.special import comb
    return comb(n_total, n_heads) * theta**n_heads * (1-theta)**(n_total - n_heads)

# Normal likelihood
def normal_likelihood(data, mu, sigma):
    """P(data | mu, sigma) for Gaussian observations."""
    return np.prod(stats.norm.pdf(data, mu, sigma))

# Poisson likelihood
def poisson_likelihood(data, lam):
    """P(data | lambda) for count data."""
    return np.prod(stats.poisson.pmf(data, lam))
```

### 4.2 Log-Likelihood for Numerical Stability

In practice, always work with log-likelihoods to avoid numerical underflow.

```python
def log_likelihood_normal(data, mu, sigma):
    """Log P(data | mu, sigma)."""
    n = len(data)
    return -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)

# Example
data = np.random.normal(5.0, 2.0, size=100)
mu_grid = np.linspace(3, 7, 200)
ll_values = [log_likelihood_normal(data, mu, 2.0) for mu in mu_grid]
mle_mu = mu_grid[np.argmax(ll_values)]
print(f"MLE estimate of mu: {mle_mu:.3f} (true: 5.0)")
```

---

## 5. Conjugate Priors

When the prior and posterior belong to the same distribution family, we have **conjugacy**. This allows closed-form Bayesian updating without numerical integration.

### 5.1 Common Conjugate Pairs

| Likelihood | Prior | Posterior | Parameters |
|------------|-------|-----------|------------|
| Bernoulli/Binomial | Beta(α, β) | Beta(α+k, β+n-k) | k successes in n trials |
| Poisson | Gamma(α, β) | Gamma(α+Σx, β+n) | n observations |
| Normal (known σ) | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) | Precision-weighted mean |
| Normal (known μ) | Inverse-Gamma(α, β) | Inverse-Gamma(α+n/2, β+SS/2) | SS = sum of squares |
| Multinomial | Dirichlet(α) | Dirichlet(α+counts) | Category counts |
| Exponential | Gamma(α, β) | Gamma(α+n, β+Σx) | n observations |

### 5.2 Beta-Binomial Conjugacy in Detail

The most commonly used conjugate pair. Let's implement sequential updating.

```python
class BetaBinomialModel:
    """Sequential Bayesian updating with Beta-Binomial conjugacy."""

    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.history = [(alpha_prior, beta_prior)]

    def update(self, n_successes, n_trials):
        """Update posterior after observing data."""
        self.alpha += n_successes
        self.beta += (n_trials - n_successes)
        self.history.append((self.alpha, self.beta))
        return self

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))

    def credible_interval(self, level=0.95):
        """Compute credible interval."""
        tail = (1 - level) / 2
        lo = stats.beta.ppf(tail, self.alpha, self.beta)
        hi = stats.beta.ppf(1 - tail, self.alpha, self.beta)
        return lo, hi

    def plot_history(self):
        """Visualize sequential updating."""
        theta = np.linspace(0, 1, 500)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (a, b) in enumerate(self.history):
            pdf = stats.beta.pdf(theta, a, b)
            ax.plot(theta, pdf, label=f"Step {i}: Beta({a:.0f},{b:.0f})")
        ax.set_xlabel("θ")
        ax.set_ylabel("Density")
        ax.set_title("Sequential Bayesian Updating")
        ax.legend()
        plt.tight_layout()
        return fig


# Sequential updating example
model = BetaBinomialModel(alpha_prior=2, beta_prior=2)

# Observe batches of coin flips
batches = [(6, 10), (8, 10), (5, 10), (7, 10)]
for heads, total in batches:
    model.update(heads, total)
    lo, hi = model.credible_interval()
    print(f"After {total} flips ({heads}H): "
          f"mean={model.mean:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")

model.plot_history()
plt.savefig("sequential_updating.png", dpi=100)
plt.show()
```

### 5.3 Normal-Normal Conjugacy

For Gaussian data with known variance, the posterior of the mean is also Gaussian.

```python
class NormalNormalModel:
    """Bayesian updating for Normal likelihood with known variance."""

    def __init__(self, mu_prior, sigma_prior, sigma_likelihood):
        self.mu = mu_prior
        self.sigma = sigma_prior
        self.sigma_lik = sigma_likelihood

    def update(self, data):
        """Update posterior after observing data points."""
        n = len(data)
        data_mean = np.mean(data)

        # Precision = 1/variance
        prior_precision = 1 / self.sigma**2
        lik_precision = n / self.sigma_lik**2

        post_precision = prior_precision + lik_precision
        post_sigma = np.sqrt(1 / post_precision)
        post_mu = (prior_precision * self.mu + lik_precision * data_mean) / post_precision

        self.mu = post_mu
        self.sigma = post_sigma
        return self

    def credible_interval(self, level=0.95):
        z = stats.norm.ppf(1 - (1 - level) / 2)
        return self.mu - z * self.sigma, self.mu + z * self.sigma


# Example: Estimate the mean temperature
model = NormalNormalModel(mu_prior=20.0, sigma_prior=5.0, sigma_likelihood=2.0)
temperature_data = np.random.normal(22.5, 2.0, size=30)
model.update(temperature_data)
lo, hi = model.credible_interval()
print(f"Posterior mean: {model.mu:.2f}, 95% CI: [{lo:.2f}, {hi:.2f}]")
```

---

## 6. Bayesian vs Frequentist: Practical Comparison

### 6.1 Confidence Interval vs Credible Interval

```python
from scipy.stats import t as t_dist

# Frequentist 95% confidence interval
data = np.array([23.1, 22.5, 24.0, 21.8, 23.5, 22.9, 24.2, 23.0])
n = len(data)
mean = data.mean()
se = data.std(ddof=1) / np.sqrt(n)
t_crit = t_dist.ppf(0.975, df=n-1)
ci_freq = (mean - t_crit * se, mean + t_crit * se)
print(f"Frequentist 95% CI: [{ci_freq[0]:.3f}, {ci_freq[1]:.3f}]")
# Interpretation: 95% of such intervals would contain the true mean

# Bayesian 95% credible interval
bayes_model = NormalNormalModel(mu_prior=22.0, sigma_prior=5.0, sigma_likelihood=1.0)
bayes_model.update(data)
ci_bayes = bayes_model.credible_interval()
print(f"Bayesian 95% CI:    [{ci_bayes[0]:.3f}, {ci_bayes[1]:.3f}]")
# Interpretation: There is a 95% probability that the true mean lies in this interval
```

### 6.2 When Bayesian Wins

1. **Small samples**: Priors regularize estimates
2. **Sequential updating**: Natural batch-by-batch learning
3. **Uncertainty quantification**: Full posterior, not just point estimates
4. **Decision making**: Posterior enables expected utility calculations

---

## 7. Maximum A Posteriori (MAP) Estimation

MAP estimation finds the mode of the posterior — a bridge between MLE and full Bayesian inference.

```python
def map_vs_mle_demo():
    """Compare MAP and MLE for a biased coin."""
    # True bias = 0.3, but we only have 5 observations
    np.random.seed(42)
    data = np.random.binomial(1, 0.3, size=5)
    k = data.sum()  # number of heads
    n = len(data)

    # MLE: k/n
    mle = k / n
    print(f"Data: {data}, k={k}, n={n}")
    print(f"MLE:  {mle:.3f}")

    # MAP with Beta(2, 5) prior (we believe coin is biased toward tails)
    alpha, beta = 2, 5
    map_estimate = (alpha + k - 1) / (alpha + beta + n - 2)
    print(f"MAP (Beta(2,5)):  {map_estimate:.3f}")

    # Posterior mean (different from MAP for skewed distributions)
    post_mean = (alpha + k) / (alpha + beta + n)
    print(f"Posterior mean:   {post_mean:.3f}")

    # Visualize
    theta = np.linspace(0, 1, 1000)
    a_post, b_post = alpha + k, beta + n - k
    posterior = stats.beta.pdf(theta, a_post, b_post)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta, posterior, 'b-', linewidth=2, label="Posterior")
    ax.axvline(mle, color='r', linestyle='--', label=f"MLE = {mle:.3f}")
    ax.axvline(map_estimate, color='g', linestyle='--', label=f"MAP = {map_estimate:.3f}")
    ax.axvline(post_mean, color='purple', linestyle='--', label=f"Post. mean = {post_mean:.3f}")
    ax.axvline(0.3, color='k', linestyle=':', label="True θ = 0.3")
    ax.legend()
    ax.set_xlabel("θ")
    ax.set_ylabel("Density")
    ax.set_title("MLE vs MAP vs Posterior Mean")
    plt.tight_layout()
    plt.savefig("map_vs_mle.png", dpi=100)
    plt.show()

map_vs_mle_demo()
```

---

## 8. Predictive Distributions

A key advantage of the Bayesian approach: instead of predicting with a single parameter estimate, we average predictions over the entire posterior.

### 8.1 Prior Predictive Distribution

```python
def prior_predictive(alpha, beta, n_trials, n_samples=10000):
    """Sample from the prior predictive distribution."""
    # Step 1: Sample theta from the prior
    thetas = np.random.beta(alpha, beta, size=n_samples)
    # Step 2: For each theta, sample the number of successes
    predictions = np.random.binomial(n_trials, thetas)
    return predictions

prior_pred = prior_predictive(2, 2, n_trials=10)
print(f"Prior predictive mean: {prior_pred.mean():.2f}")
print(f"Prior predictive std:  {prior_pred.std():.2f}")
```

### 8.2 Posterior Predictive Distribution

```python
def posterior_predictive(alpha_post, beta_post, n_trials, n_samples=10000):
    """Sample from the posterior predictive distribution."""
    thetas = np.random.beta(alpha_post, beta_post, size=n_samples)
    predictions = np.random.binomial(n_trials, thetas)
    return predictions

# After observing 7/10 heads with Beta(2,2) prior
# Posterior: Beta(9, 5)
post_pred = posterior_predictive(9, 5, n_trials=10)
print(f"Posterior predictive: {post_pred.mean():.2f} ± {post_pred.std():.2f}")

# Compare with plug-in prediction (using point estimate)
plugin_pred = np.random.binomial(10, 9/14, size=10000)
print(f"Plug-in prediction:  {plugin_pred.mean():.2f} ± {plugin_pred.std():.2f}")
# Posterior predictive has wider uncertainty (accounts for parameter uncertainty)
```

---

## 9. Grid Approximation

Before we reach MCMC in Lesson 03, grid approximation is a simple method for computing posteriors numerically.

```python
def grid_approximation(data, n_grid=1000):
    """Compute posterior via grid approximation for a Bernoulli model."""
    theta_grid = np.linspace(0, 1, n_grid)

    # Prior: Beta(2, 2)
    log_prior = stats.beta.logpdf(theta_grid, 2, 2)

    # Likelihood: product of Bernoulli
    k = data.sum()
    n = len(data)
    log_likelihood = k * np.log(theta_grid + 1e-10) + (n - k) * np.log(1 - theta_grid + 1e-10)

    # Unnormalized log-posterior
    log_posterior = log_prior + log_likelihood

    # Normalize (in log space for stability)
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= np.trapz(posterior, theta_grid)

    return theta_grid, posterior


# Example
data = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
theta_grid, posterior = grid_approximation(data)
post_mean = np.trapz(theta_grid * posterior, theta_grid)
print(f"Grid approximation posterior mean: {post_mean:.4f}")

# Compare with analytical result
alpha_post = 2 + data.sum()
beta_post = 2 + len(data) - data.sum()
print(f"Analytical posterior mean:         {alpha_post / (alpha_post + beta_post):.4f}")
```

---

## 10. The Bayesian Workflow

A systematic approach to Bayesian modeling, as advocated by Gelman et al.:

### 10.1 The Workflow Steps

```
┌─────────────────┐
│ 1. Define Model │  Choose likelihood, priors, and structure
└────────┬────────┘
         │
┌────────▼────────┐
│ 2. Prior Check  │  Prior predictive simulation
└────────┬────────┘
         │
┌────────▼────────┐
│ 3. Fit Model    │  MCMC, VI, or conjugate updating
└────────┬────────┘
         │
┌────────▼────────┐
│ 4. Diagnose     │  Convergence checks, R-hat, ESS
└────────┬────────┘
         │
┌────────▼────────────┐
│ 5. Posterior Check   │  Posterior predictive checks
└────────┬────────────┘
         │
┌────────▼────────┐
│ 6. Model Compare│  WAIC, LOO-CV, Bayes factor
└────────┬────────┘
         │
┌────────▼────────┐
│ 7. Communicate  │  Report posterior summaries, decisions
└─────────────────┘
```

### 10.2 Prior Predictive Checking

Before fitting, simulate data from the prior to verify the model generates plausible data.

```python
def prior_predictive_check():
    """Check if our model specification is reasonable."""
    n_simulations = 1000
    n_observations = 50

    # Model: y ~ Normal(mu, sigma)
    # Priors: mu ~ Normal(0, 10), sigma ~ HalfNormal(5)
    simulated_means = []
    for _ in range(n_simulations):
        mu = np.random.normal(0, 10)
        sigma = abs(np.random.normal(0, 5))
        y_sim = np.random.normal(mu, sigma, size=n_observations)
        simulated_means.append(y_sim.mean())

    simulated_means = np.array(simulated_means)
    print(f"Prior predictive mean range: [{simulated_means.min():.1f}, {simulated_means.max():.1f}]")
    print(f"Prior predictive mean of means: {simulated_means.mean():.2f}")

    # If these ranges are unreasonable for your domain, adjust priors!

prior_predictive_check()
```

---

## 11. Bayesian Decision Theory

The posterior distribution enables principled decision-making under uncertainty.

### 11.1 Loss Functions

```python
def bayesian_decision(posterior_samples, loss_fn="squared"):
    """Find the optimal point estimate under a given loss function."""
    if loss_fn == "squared":
        # Optimal: posterior mean (minimizes expected squared error)
        return np.mean(posterior_samples)
    elif loss_fn == "absolute":
        # Optimal: posterior median (minimizes expected absolute error)
        return np.median(posterior_samples)
    elif loss_fn == "zero_one":
        # Optimal: posterior mode (MAP)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(posterior_samples)
        grid = np.linspace(posterior_samples.min(), posterior_samples.max(), 1000)
        return grid[np.argmax(kde(grid))]


# Generate posterior samples from Beta(9, 5)
posterior_samples = np.random.beta(9, 5, size=50000)

for loss in ["squared", "absolute", "zero_one"]:
    estimate = bayesian_decision(posterior_samples, loss)
    print(f"Optimal estimate ({loss} loss): {estimate:.4f}")
```

### 11.2 Expected Utility

```python
def ab_test_decision(posterior_a, posterior_b):
    """A/B test: which variant is better?"""
    prob_b_better = np.mean(posterior_b > posterior_a)
    expected_lift = np.mean(posterior_b - posterior_a)
    risk_b = np.mean(np.maximum(posterior_a - posterior_b, 0))

    print(f"P(B > A):       {prob_b_better:.4f}")
    print(f"Expected lift:  {expected_lift:.4f}")
    print(f"Risk of B:      {risk_b:.4f}")
    return prob_b_better, expected_lift, risk_b


# Variant A: 120/1000 conversions, Variant B: 145/1000
posterior_a = np.random.beta(1 + 120, 1 + 880, size=50000)
posterior_b = np.random.beta(1 + 145, 1 + 855, size=50000)
ab_test_decision(posterior_a, posterior_b)
```

---

## 12. Common Pitfalls in Bayesian Thinking

### 12.1 Base Rate Neglect

```python
def medical_test_example():
    """The classic medical testing problem."""
    prevalence = 0.001       # P(disease) = 0.1%
    sensitivity = 0.99       # P(positive | disease) = 99%
    false_positive = 0.05    # P(positive | no disease) = 5%

    # P(disease | positive) via Bayes' theorem
    p_positive = sensitivity * prevalence + false_positive * (1 - prevalence)
    p_disease_given_positive = (sensitivity * prevalence) / p_positive

    print(f"P(disease | positive test) = {p_disease_given_positive:.4f}")
    print(f"Despite a 99% sensitive test, only {p_disease_given_positive*100:.1f}% "
          f"of positive tests indicate disease!")
    print(f"This is because the base rate (prevalence) is so low.")

medical_test_example()
```

### 12.2 Ignoring the Prior's Influence

```python
def prior_dominance_demo():
    """Show when the prior dominates vs when data dominates."""
    n_data_points = [1, 5, 10, 50, 200, 1000]
    true_theta = 0.7
    alpha_prior, beta_prior = 50, 50  # Strong prior centered at 0.5

    print(f"Strong prior: Beta({alpha_prior},{beta_prior}), true θ = {true_theta}")
    print("-" * 60)

    for n in n_data_points:
        k = np.random.binomial(n, true_theta)
        post_mean = (alpha_prior + k) / (alpha_prior + beta_prior + n)
        mle = k / n if n > 0 else 0
        print(f"n={n:4d}: k={k:4d}, MLE={mle:.3f}, Posterior mean={post_mean:.3f}")

prior_dominance_demo()
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Bayes' theorem | Posterior ∝ Likelihood × Prior |
| Prior | Encodes pre-data knowledge; always do sensitivity analysis |
| Conjugate priors | Enable closed-form updating (Beta-Binomial, Normal-Normal, etc.) |
| Posterior predictive | Averages over parameter uncertainty for honest predictions |
| Grid approximation | Simple but scales poorly; MCMC is needed for complex models |
| Bayesian workflow | Prior check → Fit → Diagnose → Posterior check → Compare |
| Decision theory | Use the full posterior for optimal decisions under loss functions |

---

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Edition. CRC Press.
2. McElreath, R. (2020). *Statistical Rethinking*, 2nd Edition. CRC Press.
3. Kruschke, J. K. (2014). *Doing Bayesian Data Analysis*, 2nd Edition. Academic Press.
4. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.

---

[Next: Probabilistic Graphical Models →](./02_Probabilistic_Graphical_Models.md)

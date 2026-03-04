# Bayesian Inference

**Previous**: [Hypothesis Testing](./14_Hypothesis_Testing.md) | **Next**: [Nonparametric Methods](./16_Nonparametric_Methods.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Bayesian paradigm and contrast it with the frequentist approach
2. Apply Bayes' theorem to compute posterior distributions from prior and likelihood
3. Identify and use conjugate priors for common models
4. Describe non-informative priors including Jeffreys prior
5. Compute MAP and posterior mean point estimates
6. Construct credible intervals (equal-tailed and HPD)
7. Derive and interpret the posterior predictive distribution
8. Implement Beta-Binomial updating in Python

---

Bayesian inference treats unknown parameters as random variables with probability distributions, updating beliefs in light of observed data. This provides a coherent framework for combining prior knowledge with evidence.

---

## 1. Bayesian Paradigm vs. Frequentist

### 1.1 Fundamental Difference

| Aspect | Frequentist | Bayesian |
|---|---|---|
| Parameters | Fixed but unknown constants | Random variables with distributions |
| Probability | Long-run frequency of events | Degree of belief |
| Inference | Based on sampling distribution of estimators | Based on posterior distribution of parameters |
| Prior information | Not formally incorporated | Encoded in the prior distribution |
| Interval | Confidence interval (coverage property) | Credible interval (probability statement about $\theta$) |

### 1.2 When to Prefer Bayesian Methods

- When prior information is available and should be incorporated
- When the sample size is small and priors can stabilize estimates
- When you want direct probability statements about parameters
- When hierarchical or complex models are needed
- When sequential updating of beliefs is natural

---

## 2. Bayes' Theorem for Inference

### 2.1 The Core Formula

Given data $x$ and parameter $\theta$:

$$\pi(\theta \mid x) = \frac{f(x \mid \theta) \, \pi(\theta)}{f(x)} = \frac{f(x \mid \theta) \, \pi(\theta)}{\int f(x \mid \theta) \, \pi(\theta) \, d\theta}$$

In shorthand:

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

$$\pi(\theta \mid x) \propto f(x \mid \theta) \, \pi(\theta)$$

### 2.2 Components

- **Prior** $\pi(\theta)$: Encodes beliefs about $\theta$ before seeing data.
- **Likelihood** $f(x \mid \theta) = L(\theta \mid x)$: The probability of the observed data given $\theta$.
- **Posterior** $\pi(\theta \mid x)$: Updated beliefs after observing data.
- **Marginal likelihood** $f(x) = \int f(x \mid \theta)\pi(\theta) \, d\theta$: Normalizing constant (important for model comparison).

### 2.3 Sequential Updating

A powerful feature of Bayesian inference is that today's posterior becomes tomorrow's prior. If we observe data $x_1$, then later $x_2$:

$$\pi(\theta \mid x_1, x_2) \propto f(x_2 \mid \theta) \cdot \underbrace{f(x_1 \mid \theta) \pi(\theta)}_{\propto \, \pi(\theta \mid x_1)}$$

This makes Bayesian methods naturally suited for online or streaming data.

---

## 3. Conjugate Priors

A prior $\pi(\theta)$ is **conjugate** for a likelihood $f(x \mid \theta)$ if the posterior $\pi(\theta \mid x)$ belongs to the same family as the prior. This yields closed-form posterior updates.

### 3.1 Beta-Binomial

**Model**: $X \mid p \sim \text{Binomial}(n, p)$

**Prior**: $p \sim \text{Beta}(\alpha, \beta)$

**Posterior**: $p \mid x \sim \text{Beta}(\alpha + x, \, \beta + n - x)$

The Beta prior is flexible: $\alpha = \beta = 1$ gives a uniform prior; $\alpha, \beta > 1$ concentrates mass away from 0 and 1.

**Prior mean**: $E[p] = \frac{\alpha}{\alpha + \beta}$

**Posterior mean**: $E[p \mid x] = \frac{\alpha + x}{\alpha + \beta + n}$

This is a weighted average of the prior mean and the sample proportion $\hat{p} = x/n$.

### 3.2 Normal-Normal

**Model**: $X_1, \ldots, X_n \mid \mu \sim N(\mu, \sigma^2)$ with $\sigma^2$ known.

**Prior**: $\mu \sim N(\mu_0, \tau^2)$

**Posterior**: $\mu \mid x \sim N(\mu_n, \tau_n^2)$ where:

$$\mu_n = \frac{\frac{\mu_0}{\tau^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau^2} + \frac{n}{\sigma^2}}, \quad \tau_n^2 = \frac{1}{\frac{1}{\tau^2} + \frac{n}{\sigma^2}}$$

The posterior mean is a **precision-weighted average** of the prior mean and the sample mean, where precision = $1/\text{variance}$.

### 3.3 Gamma-Poisson

**Model**: $X_1, \ldots, X_n \mid \lambda \sim \text{Poisson}(\lambda)$

**Prior**: $\lambda \sim \text{Gamma}(\alpha, \beta)$

**Posterior**: $\lambda \mid x \sim \text{Gamma}(\alpha + \sum x_i, \, \beta + n)$

### 3.4 Summary Table

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Binomial$(n, p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + x, \beta + n - x)$ |
| Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum x_i, \beta + n)$ |
| Normal$(\mu, \sigma^2)$ ($\sigma^2$ known) | Normal$(\mu_0, \tau^2)$ | Normal$(\mu_n, \tau_n^2)$ |
| Normal$(\mu, \sigma^2)$ ($\mu$ known) | Inverse-Gamma$(a, b)$ | Inverse-Gamma$(a + n/2, b + \sum(x_i-\mu)^2/2)$ |
| Exponential$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + n, \beta + \sum x_i)$ |

---

## 4. Non-Informative Priors

When we lack prior information, we seek priors that "let the data speak."

### 4.1 Flat (Uniform) Prior

$\pi(\theta) \propto 1$ over the parameter space. Simple but not invariant under reparameterization: if $\pi(\theta) \propto 1$, then $\pi(\phi) \propto |d\theta/d\phi|$ for $\phi = g(\theta)$, which is generally not flat.

### 4.2 Jeffreys Prior

Jeffreys prior is invariant under reparameterization:

$$\pi_J(\theta) \propto \sqrt{I(\theta)}$$

where $I(\theta) = -E\left[\frac{\partial^2 \ln f(X \mid \theta)}{\partial \theta^2}\right]$ is the Fisher information.

**Examples**:
- Binomial: $\pi_J(p) \propto p^{-1/2}(1-p)^{-1/2} = \text{Beta}(1/2, 1/2)$
- Normal mean ($\sigma$ known): $\pi_J(\mu) \propto 1$
- Normal variance ($\mu$ known): $\pi_J(\sigma^2) \propto 1/\sigma^2$

### 4.3 Weakly Informative Priors

In practice, fully non-informative priors can lead to improper posteriors or poor behavior. **Weakly informative priors** provide mild regularization without dominating the data. For example, a $N(0, 100)$ prior on a regression coefficient constrains it to a reasonable range without strongly influencing the posterior.

---

## 5. Point Estimation

### 5.1 Maximum a Posteriori (MAP)

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \pi(\theta \mid x) = \arg\max_\theta [f(x \mid \theta) \pi(\theta)]$$

MAP generalizes MLE by incorporating the prior. With a flat prior, MAP = MLE.

### 5.2 Posterior Mean

$$\hat{\theta}_{\text{PM}} = E[\theta \mid x] = \int \theta \, \pi(\theta \mid x) \, d\theta$$

The posterior mean minimizes the expected squared error loss $E[(\theta - a)^2 \mid x]$.

### 5.3 Posterior Median

The posterior median minimizes the expected absolute error loss $E[|\theta - a| \mid x]$.

### 5.4 Comparison

| Estimator | Optimal under | Properties |
|---|---|---|
| MAP | 0-1 loss | Mode of posterior; may not exist for multimodal posteriors |
| Posterior mean | Squared error loss | Always exists for proper posteriors; sensitive to tails |
| Posterior median | Absolute error loss | Robust to skewness |

For symmetric unimodal posteriors, all three coincide.

---

## 6. Credible Intervals

### 6.1 Definition

A $(1-\alpha)$ **credible interval** $[a, b]$ satisfies:

$$P(a \leq \theta \leq b \mid x) = 1 - \alpha$$

Unlike confidence intervals, this is a direct probability statement about $\theta$ (given the data and the model).

### 6.2 Equal-Tailed Interval

Choose $a$ and $b$ such that:

$$P(\theta < a \mid x) = \alpha/2, \quad P(\theta > b \mid x) = \alpha/2$$

This places equal probability in each tail.

### 6.3 Highest Posterior Density (HPD) Interval

The HPD interval is the shortest interval containing $1-\alpha$ probability:

$$C = \{\theta : \pi(\theta \mid x) \geq c\}$$

where $c$ is chosen so that $P(\theta \in C \mid x) = 1-\alpha$.

For symmetric unimodal posteriors, the HPD and equal-tailed intervals coincide. For skewed posteriors, the HPD interval is shorter.

---

## 7. Posterior Predictive Distribution

To predict a new observation $\tilde{x}$ given observed data $x$:

$$f(\tilde{x} \mid x) = \int f(\tilde{x} \mid \theta) \, \pi(\theta \mid x) \, d\theta$$

This averages the likelihood over the posterior, incorporating parameter uncertainty into predictions.

**Example (Beta-Binomial)**: If $p \mid x \sim \text{Beta}(\alpha', \beta')$, then the predictive probability of $k$ successes in $m$ new trials is:

$$P(\tilde{X} = k \mid x) = \binom{m}{k} \frac{B(\alpha' + k, \beta' + m - k)}{B(\alpha', \beta')}$$

where $B(\cdot, \cdot)$ is the Beta function. This is the **Beta-Binomial** distribution.

---

## 8. Bayesian vs. Frequentist Comparison

| Feature | Frequentist | Bayesian |
|---|---|---|
| Probability of parameter | Not defined | $\pi(\theta \mid x)$ |
| Confidence/Credible interval | Coverage guarantee over repetitions | Direct probability statement |
| Prior information | Not used (or used informally) | Formally incorporated |
| Small samples | Can be unreliable | Prior helps regularize |
| Computational burden | Usually lower | Can be high (MCMC for complex models) |
| Consistency | Consistent estimators under regularity | Posterior concentrates on true $\theta$ |
| Subjectivity | Choice of estimator, test | Choice of prior |
| Hypothesis testing | p-values, rejection regions | Bayes factors, posterior probabilities |

In practice, for large samples and vague priors, Bayesian and frequentist results often agree closely.

---

## 9. Python Examples: Beta-Binomial Updating

```python
import math
import random

class BetaBinomial:
    """Beta-Binomial conjugate model for Bayesian updating."""

    def __init__(self, alpha=1.0, beta=1.0):
        """Initialize with Beta(alpha, beta) prior.

        Default: uniform prior (alpha=1, beta=1).
        """
        self.alpha = alpha
        self.beta = beta

    def update(self, successes, trials):
        """Update posterior with observed data."""
        self.alpha += successes
        self.beta += trials - successes
        return self

    def posterior_mean(self):
        return self.alpha / (self.alpha + self.beta)

    def posterior_mode(self):
        """MAP estimate (mode of Beta distribution)."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None  # Mode at boundary if alpha or beta <= 1

    def posterior_variance(self):
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))

    def _beta_function(self, a, b):
        """Compute Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
        return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

    def pdf(self, p):
        """Evaluate posterior Beta PDF at p."""
        a, b = self.alpha, self.beta
        if p <= 0 or p >= 1:
            return 0.0
        return p**(a-1) * (1-p)**(b-1) / self._beta_function(a, b)

    def equal_tailed_interval(self, level=0.95, n_grid=10000):
        """Approximate equal-tailed credible interval via grid."""
        alpha_tail = (1 - level) / 2
        grid = [i / n_grid for i in range(1, n_grid)]
        pdf_values = [self.pdf(p) for p in grid]
        total = sum(pdf_values)
        cdf = 0
        lower, upper = 0.0, 1.0
        for p, pdf_val in zip(grid, pdf_values):
            cdf += pdf_val / total
            if cdf >= alpha_tail and lower == 0.0:
                lower = p
            if cdf >= 1 - alpha_tail and upper == 1.0:
                upper = p
                break
        return (lower, upper)

    def __repr__(self):
        return f"Beta({self.alpha:.2f}, {self.beta:.2f})"


# ----- Example: Sequential coin-flip updating -----
print("=== Beta-Binomial Sequential Updating ===\n")
model = BetaBinomial(alpha=2, beta=2)  # Mild prior: expect p near 0.5
print(f"Prior:          {model}")
print(f"Prior mean:     {model.posterior_mean():.4f}\n")

# Observe batches of coin flips
batches = [(7, 10), (6, 10), (14, 20)]  # (successes, trials)
total_s, total_t = 0, 0
for successes, trials in batches:
    total_s += successes
    total_t += trials
    model.update(successes, trials)
    print(f"After {total_t} trials ({total_s} successes):")
    print(f"  Posterior:     {model}")
    print(f"  Post. mean:    {model.posterior_mean():.4f}")
    print(f"  Post. mode:    {model.posterior_mode():.4f}")
    print(f"  Post. var:     {model.posterior_variance():.6f}")
    ci = model.equal_tailed_interval()
    print(f"  95% CI:        ({ci[0]:.4f}, {ci[1]:.4f})\n")


# ----- Example: Prior sensitivity analysis -----
print("=== Prior Sensitivity Analysis ===\n")
print("Observed: 3 successes in 10 trials\n")

priors = [
    ("Uniform", 1, 1),
    ("Jeffreys", 0.5, 0.5),
    ("Informative (p~0.5)", 10, 10),
    ("Informative (p~0.8)", 16, 4),
]

for name, a, b in priors:
    m = BetaBinomial(a, b)
    m.update(3, 10)
    print(f"Prior: {name:30s} -> Posterior mean = {m.posterior_mean():.4f}")


# ----- Example: Posterior predictive -----
print("\n=== Posterior Predictive ===\n")
model = BetaBinomial(2, 2)
model.update(7, 10)
print(f"Posterior: {model}")

# P(next flip = heads) = posterior mean of p
print(f"P(next flip = heads) = {model.posterior_mean():.4f}")

# P(k heads in next 5 flips) using Beta-Binomial predictive
def predictive_pmf(k, m, alpha, beta):
    """P(X=k) for Beta-Binomial(m, alpha, beta)."""
    log_binom = math.lgamma(m+1) - math.lgamma(k+1) - math.lgamma(m-k+1)
    log_beta_num = math.lgamma(alpha+k) + math.lgamma(beta+m-k) - math.lgamma(alpha+beta+m)
    log_beta_den = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha+beta)
    return math.exp(log_binom + log_beta_num - log_beta_den)

m_new = 5
print(f"\nPredictive distribution for {m_new} new trials:")
for k in range(m_new + 1):
    prob = predictive_pmf(k, m_new, model.alpha, model.beta)
    bar = "#" * int(prob * 50)
    print(f"  P(X={k}) = {prob:.4f}  {bar}")
```

---

## 10. Key Takeaways

| Concept | Key Point |
|---|---|
| Bayes' theorem | Posterior $\propto$ Likelihood $\times$ Prior |
| Conjugate priors | Closed-form posterior; easy sequential updating |
| Jeffreys prior | Non-informative; invariant under reparameterization |
| MAP estimation | Mode of posterior; equals MLE with flat prior |
| Posterior mean | Minimizes squared error loss; precision-weighted average |
| Credible interval | Direct probability statement: $P(\theta \in C \mid x) = 1 - \alpha$ |
| HPD interval | Shortest interval with given credibility level |
| Posterior predictive | Integrates out parameter uncertainty for prediction |
| Prior sensitivity | Always check how results change with different priors |

**Key insight**: As the sample size grows, the likelihood dominates the prior, and Bayesian and frequentist estimates converge. The prior matters most when data are scarce.

---

**Previous**: [Hypothesis Testing](./14_Hypothesis_Testing.md) | **Next**: [Nonparametric Methods](./16_Nonparametric_Methods.md)

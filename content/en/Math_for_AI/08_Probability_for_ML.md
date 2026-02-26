# 08. Probability for Machine Learning

## Learning Objectives

- Understand and apply basic probability axioms, conditional probability, and Bayes' theorem
- Learn the concept of random variables and the differences between discrete and continuous distributions
- Calculate and interpret key statistics of random variables such as expectation, variance, and covariance
- Learn the characteristics and applications of probability distributions commonly used in machine learning
- Implement probabilistic inference and Bayesian updates using Bayes' theorem
- Understand the difference between generative and discriminative models from a probabilistic perspective

---

## 1. Foundations of Probability

### 1.1 Axioms of Probability

**Sample Space** $\Omega$: set of all possible outcomes

**Event** $A$: subset of the sample space

**Probability Measure** $P$ satisfies the following axioms:

1. **Non-negativity**: $P(A) \geq 0$ for all $A$
2. **Normalization**: $P(\Omega) = 1$
3. **Countable Additivity**: For mutually exclusive events $A_1, A_2, \ldots$
   $$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$$

### 1.2 Conditional Probability

Probability of event $A$ occurring given that event $B$ has occurred:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{if } P(B) > 0
$$

**Intuition**: When we know $B$ has occurred, the sample space shrinks from $\Omega$ to $B$.

### 1.3 Independence

Events $A$ and $B$ are independent if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

or equivalently:
$$
P(A|B) = P(A)
$$

### 1.4 Law of Total Probability

If $B_1, \ldots, B_n$ form a partition of the sample space:

$$
P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)
$$

### 1.5 Bayes' Theorem

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

or using the law of total probability:

$$
P(A|B) = \frac{P(B|A)P(A)}{\sum_{i} P(B|A_i)P(A_i)}
$$

**Terminology:**
- $P(A)$: **prior probability**
- $P(B|A)$: **likelihood**
- $P(A|B)$: **posterior probability**
- $P(B)$: **marginal probability** or **evidence**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bayes' theorem example: medical diagnosis
# Disease prevalence: 1%
P_disease = 0.01
P_no_disease = 1 - P_disease

# Test accuracy
# Sensitivity: probability of positive given disease
P_positive_given_disease = 0.95
# Specificity: probability of negative given no disease
P_negative_given_no_disease = 0.95
P_positive_given_no_disease = 1 - P_negative_given_no_disease

# Total probability: probability of positive test
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_no_disease * P_no_disease)

# Bayes' theorem: probability of disease given positive test
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print("Medical Diagnosis Example (Bayes' Theorem)")
print(f"Disease prevalence (prior probability): {P_disease:.1%}")
print(f"Test sensitivity: {P_positive_given_disease:.1%}")
print(f"Test specificity: {P_negative_given_no_disease:.1%}")
print(f"\nProbability of positive test (total probability): {P_positive:.4f}")
print(f"Probability of disease given positive test (posterior): {P_disease_given_positive:.1%}")
print(f"\nInterpretation: Even with a positive test, the probability of actually having the disease is only {P_disease_given_positive:.1%}")
print("       (Many false positives due to low disease prevalence)")

# Visualization: Bayes' theorem
fig, ax = plt.subplots(figsize=(12, 6))

categories = ['Prior Probability\n(Disease)', 'Likelihood\n(Positive|Disease)', 'Posterior Probability\n(Disease|Positive)']
probabilities = [P_disease, P_positive_given_disease, P_disease_given_positive]
colors = ['skyblue', 'lightgreen', 'salmon']

bars = ax.bar(categories, probabilities, color=colors, edgecolor='black', linewidth=2)

# Display values
for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{prob:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Probability', fontsize=13)
ax.set_title("Bayes' Theorem: Medical Diagnosis Example", fontsize=15)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bayes_theorem_medical.png', dpi=150)
plt.show()
```

## 2. Random Variables

### 2.1 Definition of Random Variables

**Random Variable**: a function from the sample space to real numbers
$$X: \Omega \to \mathbb{R}$$

**Discrete random variable**: takes countable values (e.g., dice, coins)
**Continuous random variable**: takes continuous values (e.g., height, temperature)

### 2.2 Probability Mass Function (PMF)

For a discrete random variable $X$:

$$
p_X(x) = P(X = x)
$$

**Properties:**
- $p_X(x) \geq 0$ for all $x$
- $\sum_{x} p_X(x) = 1$

### 2.3 Probability Density Function (PDF)

For a continuous random variable $X$:

$$
P(a \leq X \leq b) = \int_a^b f_X(x) dx
$$

**Properties:**
- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$
- $P(X = x) = 0$ (probability at a single point is 0)

### 2.4 Cumulative Distribution Function (CDF)

$$
F_X(x) = P(X \leq x)
$$

**Properties:**
- Non-decreasing function
- $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$
- For continuous random variables: $f_X(x) = \frac{d}{dx}F_X(x)$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Discrete: Binomial distribution
n, p = 10, 0.5
x_binom = np.arange(0, n+1)
pmf_binom = stats.binom.pmf(x_binom, n, p)
cdf_binom = stats.binom.cdf(x_binom, n, p)

axes[0, 0].bar(x_binom, pmf_binom, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Binomial PMF\n$n=10, p=0.5$', fontsize=12)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('P(X=x)')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].step(x_binom, cdf_binom, where='post', linewidth=2, color='blue')
axes[1, 0].set_title('Binomial CDF', fontsize=12)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('P(X≤x)')
axes[1, 0].grid(True, alpha=0.3)

# 2. Continuous: Normal distribution
mu, sigma = 0, 1
x_norm = np.linspace(-4, 4, 1000)
pdf_norm = stats.norm.pdf(x_norm, mu, sigma)
cdf_norm = stats.norm.cdf(x_norm, mu, sigma)

axes[0, 1].plot(x_norm, pdf_norm, linewidth=2, color='red')
axes[0, 1].fill_between(x_norm, pdf_norm, alpha=0.3, color='red')
axes[0, 1].set_title('Normal PDF\n$\mu=0, \sigma=1$', fontsize=12)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('f(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(x_norm, cdf_norm, linewidth=2, color='darkred')
axes[1, 1].set_title('Normal CDF', fontsize=12)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('F(x)')
axes[1, 1].grid(True, alpha=0.3)

# 3. Continuous: Exponential distribution
lam = 1.0
x_exp = np.linspace(0, 5, 1000)
pdf_exp = stats.expon.pdf(x_exp, scale=1/lam)
cdf_exp = stats.expon.cdf(x_exp, scale=1/lam)

axes[0, 2].plot(x_exp, pdf_exp, linewidth=2, color='green')
axes[0, 2].fill_between(x_exp, pdf_exp, alpha=0.3, color='green')
axes[0, 2].set_title('Exponential PDF\n$\lambda=1$', fontsize=12)
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('f(x)')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 2].plot(x_exp, cdf_exp, linewidth=2, color='darkgreen')
axes[1, 2].set_title('Exponential CDF', fontsize=12)
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('F(x)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pmf_pdf_cdf.png', dpi=150)
plt.show()

print("PMF vs PDF:")
print("  PMF (discrete): probability at a specific value P(X=x)")
print("  PDF (continuous): probability density, interval probability computed by integration")
print("  CDF: cumulative probability P(X≤x), defined for both discrete and continuous")
```

### 2.5 Joint, Marginal, and Conditional Distributions

**Joint Distribution:**
$$P(X = x, Y = y)$$ or $$f_{X,Y}(x, y)$$

**Marginal Distribution:**
$$p_X(x) = \sum_y p_{X,Y}(x, y)$$ or $$f_X(x) = \int f_{X,Y}(x, y) dy$$

**Conditional Distribution:**
$$p_{X|Y}(x|y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}$$

```python
# Joint distribution example: bivariate normal distribution
from scipy.stats import multivariate_normal

# Parameters
mu = np.array([0, 0])
cov = np.array([[1, 0.7],
                [0.7, 1]])

# Grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Joint PDF
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Contour plot
ax = axes[0]
contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
plt.colorbar(contour, ax=ax)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Joint Distribution $f_{X,Y}(x,y)$ (Bivariate Normal)', fontsize=14)
ax.grid(True, alpha=0.3)

# Marginal distributions
ax = axes[1]
marginal_X = stats.norm.pdf(x, mu[0], np.sqrt(cov[0, 0]))
marginal_Y = stats.norm.pdf(y, mu[1], np.sqrt(cov[1, 1]))
ax.plot(x, marginal_X, linewidth=3, label='Marginal $f_X(x)$', color='blue')
ax.plot(y, marginal_Y, linewidth=3, label='Marginal $f_Y(y)$', color='red')
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Marginal Distributions', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('joint_marginal_distributions.png', dpi=150)
plt.show()

print(f"Covariance matrix:\n{cov}")
print(f"Correlation coefficient: {cov[0,1] / np.sqrt(cov[0,0] * cov[1,1]):.2f}")
```

## 3. Expectation and Variance

### 3.1 Expectation

**Discrete:**
$$\mathbb{E}[X] = \sum_x x \cdot p_X(x)$$

**Continuous:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

**Expectation of a function (LOTUS - Law of the Unconscious Statistician):**
$$\mathbb{E}[g(X)] = \sum_x g(x) \cdot p_X(x) \quad \text{or} \quad \int g(x) \cdot f_X(x) dx$$

### 3.2 Properties of Expectation

1. **Linearity:**
   $$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

2. **Product of independent variables:**
   If $X, Y$ independent then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

### 3.3 Variance

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**Standard Deviation:**
$$\sigma_X = \sqrt{\text{Var}(X)}$$

**Properties of variance:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- If $X, Y$ independent then $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 3.4 Covariance

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

**Correlation Coefficient:**
$$
\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo estimation of expectation and variance
np.random.seed(42)

# Sampling from normal distribution
mu, sigma = 2, 1.5
samples = np.random.normal(mu, sigma, 10000)

# Estimate expectation and variance
estimated_mean = np.mean(samples)
estimated_var = np.var(samples, ddof=0)
estimated_std = np.std(samples, ddof=0)

print("Monte Carlo Estimation")
print(f"Theoretical mean: {mu}, Estimated mean: {estimated_mean:.4f}")
print(f"Theoretical variance: {sigma**2}, Estimated variance: {estimated_var:.4f}")
print(f"Theoretical std dev: {sigma}, Estimated std dev: {estimated_std:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram + theoretical PDF
ax = axes[0]
ax.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue',
        edgecolor='black', label='Sample histogram')
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(x, mu, sigma)
ax.plot(x, pdf, linewidth=3, color='red', label='Theoretical PDF')
ax.axvline(estimated_mean, color='green', linestyle='--', linewidth=2,
           label=f'Estimated mean = {estimated_mean:.2f}')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Normal Distribution Sampling (μ={mu}, σ={sigma})', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Convergence with increasing sample size
ax = axes[1]
sample_sizes = np.arange(10, 10001, 10)
running_means = [np.mean(samples[:n]) for n in sample_sizes]

ax.plot(sample_sizes, running_means, linewidth=2, color='blue',
        label='Running mean')
ax.axhline(mu, color='red', linestyle='--', linewidth=2, label=f'Theoretical mean = {mu}')
ax.set_xlabel('Sample size', fontsize=12)
ax.set_ylabel('Running mean', fontsize=12)
ax.set_title('Law of Large Numbers', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('expectation_variance_estimation.png', dpi=150)
plt.show()

# Covariance example
np.random.seed(42)
n = 1000

# Positive correlation
X1 = np.random.randn(n)
Y1 = 0.8 * X1 + 0.3 * np.random.randn(n)

# Negative correlation
X2 = np.random.randn(n)
Y2 = -0.8 * X2 + 0.3 * np.random.randn(n)

# Independent
X3 = np.random.randn(n)
Y3 = np.random.randn(n)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [(X1, Y1, 'Positive Correlation'), (X2, Y2, 'Negative Correlation'), (X3, Y3, 'Independent (No Correlation)')]
for idx, (X, Y, title) in enumerate(datasets):
    ax = axes[idx]
    ax.scatter(X, Y, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    # Compute statistics
    cov = np.cov(X, Y)[0, 1]
    corr = np.corrcoef(X, Y)[0, 1]

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'{title}\nCov={cov:.3f}, ρ={corr:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('covariance_correlation.png', dpi=150)
plt.show()

print("\nCovariance and Correlation:")
print("  Cov > 0: Positive relationship (X increases → Y increases)")
print("  Cov < 0: Negative relationship (X increases → Y decreases)")
print("  Cov = 0: No linear relationship (independence implies Cov=0, but not vice versa)")
print("  ρ ∈ [-1, 1]: Normalized covariance (unit-independent)")
```

## 4. Common Probability Distributions

### 4.1 Discrete Distributions

**Bernoulli Distribution:**
$$X \sim \text{Ber}(p), \quad P(X=1) = p, \; P(X=0) = 1-p$$
- Mean: $p$, Variance: $p(1-p)$

**Binomial Distribution:**
$$X \sim \text{Bin}(n, p), \quad P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$$
- Number of successes in $n$ independent Bernoulli trials
- Mean: $np$, Variance: $np(1-p)$

**Poisson Distribution:**
$$X \sim \text{Pois}(\lambda), \quad P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
- Number of events in unit time/space
- Mean: $\lambda$, Variance: $\lambda$

**Categorical Distribution:**
$$X \sim \text{Cat}(p_1, \ldots, p_K), \quad P(X=k) = p_k, \; \sum p_k = 1$$
- Basic distribution for multiclass classification

### 4.2 Continuous Distributions

**Uniform Distribution:**
$$X \sim \text{Unif}(a, b), \quad f(x) = \frac{1}{b-a} \text{ for } x \in [a, b]$$
- Mean: $\frac{a+b}{2}$, Variance: $\frac{(b-a)^2}{12}$

**Normal/Gaussian Distribution:**
$$X \sim \mathcal{N}(\mu, \sigma^2), \quad f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- Mean: $\mu$, Variance: $\sigma^2$
- Arises naturally via Central Limit Theorem

**Exponential Distribution:**
$$X \sim \text{Exp}(\lambda), \quad f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0$$
- Waiting time in Poisson process
- Mean: $1/\lambda$, Variance: $1/\lambda^2$

**Beta Distribution:**
$$X \sim \text{Beta}(\alpha, \beta), \quad f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \text{ for } x \in [0, 1]$$
- Distribution of probabilities (prior in Bayesian inference)

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1. Bernoulli
ax = axes[0, 0]
p = 0.7
x = [0, 1]
pmf = [1-p, p]
ax.bar(x, pmf, color='skyblue', edgecolor='black', width=0.4)
ax.set_title(f'Bernoulli (p={p})', fontsize=12)
ax.set_xticks([0, 1])
ax.set_ylabel('P(X=x)')

# 2. Binomial
ax = axes[0, 1]
n, p = 20, 0.5
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)
ax.bar(x, pmf, color='lightgreen', edgecolor='black')
ax.set_title(f'Binomial (n={n}, p={p})', fontsize=12)
ax.set_xlabel('x')

# 3. Poisson
ax = axes[0, 2]
lam = 5
x = np.arange(0, 20)
pmf = stats.poisson.pmf(x, lam)
ax.bar(x, pmf, color='salmon', edgecolor='black')
ax.set_title(f'Poisson (λ={lam})', fontsize=12)
ax.set_xlabel('x')

# 4. Uniform
ax = axes[1, 0]
a, b = 0, 1
x = np.linspace(-0.5, 1.5, 1000)
pdf = stats.uniform.pdf(x, a, b-a)
ax.plot(x, pdf, linewidth=3, color='blue')
ax.fill_between(x, pdf, alpha=0.3, color='blue')
ax.set_title(f'Uniform (a={a}, b={b})', fontsize=12)
ax.set_ylabel('f(x)')

# 5. Normal (various parameters)
ax = axes[1, 1]
x = np.linspace(-5, 5, 1000)
params = [(0, 1), (0, 0.5), (1, 1)]
for mu, sigma in params:
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, linewidth=2, label=f'μ={mu}, σ={sigma}')
ax.set_title('Normal Distribution', fontsize=12)
ax.legend(fontsize=9)

# 6. Exponential
ax = axes[1, 2]
x = np.linspace(0, 5, 1000)
lambdas = [0.5, 1, 2]
for lam in lambdas:
    pdf = stats.expon.pdf(x, scale=1/lam)
    ax.plot(x, pdf, linewidth=2, label=f'λ={lam}')
ax.set_title('Exponential Distribution', fontsize=12)
ax.legend(fontsize=9)

# 7. Gamma
ax = axes[2, 0]
x = np.linspace(0, 20, 1000)
params = [(1, 1), (2, 2), (5, 1)]
for k, theta in params:
    pdf = stats.gamma.pdf(x, k, scale=theta)
    ax.plot(x, pdf, linewidth=2, label=f'k={k}, θ={theta}')
ax.set_title('Gamma Distribution', fontsize=12)
ax.set_ylabel('f(x)')
ax.legend(fontsize=9)

# 8. Beta
ax = axes[2, 1]
x = np.linspace(0, 1, 1000)
params = [(0.5, 0.5), (2, 2), (5, 2)]
for alpha, beta in params:
    pdf = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, pdf, linewidth=2, label=f'α={alpha}, β={beta}')
ax.set_title('Beta Distribution', fontsize=12)
ax.legend(fontsize=9)

# 9. Chi-squared
ax = axes[2, 2]
x = np.linspace(0, 15, 1000)
dfs = [2, 4, 6]
for df in dfs:
    pdf = stats.chi2.pdf(x, df)
    ax.plot(x, pdf, linewidth=2, label=f'df={df}')
ax.set_title('Chi-squared Distribution', fontsize=12)
ax.legend(fontsize=9)

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('common_distributions.png', dpi=150)
plt.show()

print("Applications in machine learning:")
print("  Bernoulli/Binomial: Binary classification")
print("  Categorical: Multi-class classification")
print("  Normal: Continuous data, error models, VAE latent space")
print("  Poisson: Count data (recommendation systems, web traffic)")
print("  Beta: Prior distribution in Bayesian inference")
print("  Exponential/Gamma: Waiting times, survival analysis")
```

### 4.3 Multivariate Normal Distribution

$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\boldsymbol{\Sigma}|}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

- $\boldsymbol{\mu} \in \mathbb{R}^d$: mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$: covariance matrix (positive definite)

```python
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

# Multivariate normal distribution visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

mu = np.array([0, 0])
covs = [
    np.array([[1, 0], [0, 1]]),      # Independent
    np.array([[1, 0.8], [0.8, 1]]),  # Positive correlation
    np.array([[1, -0.8], [-0.8, 1]]) # Negative correlation
]
titles = ['Independent (ρ=0)', 'Positive Correlation (ρ=0.8)', 'Negative Correlation (ρ=-0.8)']

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

for ax, cov, title in zip(axes, covs, titles):
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)

    contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
    ax.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('$X_1$', fontsize=12)
    ax.set_ylabel('$X_2$', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multivariate_normal.png', dpi=150)
plt.show()

print("Multivariate Normal Distribution:")
print("  - Fundamental building block for high-dimensional data modeling")
print("  - Core component in Gaussian processes, GMM, VAE, etc.")
print("  - Covariance matrix captures dependencies between variables")
```

## 5. Advanced Bayes' Theorem

### 5.1 Bayesian Update

**Prior** → **Data** → **Posterior**

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)} \propto P(D | \theta) P(\theta)
$$

- $\theta$: parameter (treated as random variable)
- $D$: observed data
- $P(\theta)$: prior probability (belief before data)
- $P(D | \theta)$: likelihood (plausibility of data given parameter)
- $P(\theta | D)$: posterior probability (updated belief after data)

### 5.2 Example: Coin Flip (Beta-Binomial Model)

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Beta-Binomial model: estimating the probability of heads for a coin
# Prior: Beta(α, β)
# Likelihood: Binomial
# Posterior: Beta(α + n_heads, β + n_tails)

np.random.seed(42)

# True coin probability (assumed unknown)
true_p = 0.7

# Prior distribution (uniform prior: Beta(1, 1))
alpha_prior, beta_prior = 1, 1

# Coin flip simulation
n_flips_list = [0, 1, 5, 20, 100]
data = np.random.binomial(1, true_p, 100)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

p_vals = np.linspace(0, 1, 1000)

for idx, n_flips in enumerate(n_flips_list):
    ax = axes[idx]

    if n_flips == 0:
        # Prior distribution only
        prior_pdf = stats.beta.pdf(p_vals, alpha_prior, beta_prior)
        ax.plot(p_vals, prior_pdf, linewidth=3, color='blue', label='Prior')
    else:
        # Data
        observed_data = data[:n_flips]
        n_heads = np.sum(observed_data)
        n_tails = n_flips - n_heads

        # Posterior distribution
        alpha_post = alpha_prior + n_heads
        beta_post = beta_prior + n_tails
        posterior_pdf = stats.beta.pdf(p_vals, alpha_post, beta_post)

        # Prior distribution
        prior_pdf = stats.beta.pdf(p_vals, alpha_prior, beta_prior)

        ax.plot(p_vals, prior_pdf, linewidth=2, color='blue', linestyle='--',
                label='Prior', alpha=0.7)
        ax.plot(p_vals, posterior_pdf, linewidth=3, color='red', label='Posterior')

        # MAP estimate (maximum a posteriori)
        map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
        ax.axvline(map_estimate, color='red', linestyle=':', linewidth=2,
                   label=f'MAP = {map_estimate:.3f}')

    # True probability
    ax.axvline(true_p, color='green', linestyle='--', linewidth=2,
               label=f'True p = {true_p}')

    ax.set_xlabel('p (head probability)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'After {n_flips} coin flips' if n_flips > 0 else 'Prior Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

# Convergence curve
ax = axes[-1]
n_range = np.arange(1, 101)
map_estimates = []
for n in n_range:
    n_heads = np.sum(data[:n])
    n_tails = n - n_heads
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + n_tails
    map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
    map_estimates.append(map_est)

ax.plot(n_range, map_estimates, linewidth=2, color='red', label='MAP estimate')
ax.axhline(true_p, color='green', linestyle='--', linewidth=2, label=f'True p = {true_p}')
ax.set_xlabel('Number of coin flips', fontsize=11)
ax.set_ylabel('Estimated p', fontsize=11)
ax.set_title('Bayesian Learning Convergence', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_update_coin.png', dpi=150)
plt.show()

print("Bayesian Update:")
print("  - As data increases, the posterior concentrates around the true value")
print("  - The influence of the prior decreases as data grows")
print("  - Uncertainty is expressed as a distribution (not a point estimate)")
```

## 6. Probability in Machine Learning

### 6.1 Generative vs Discriminative Models

**Generative Model:**
- Models $P(X, Y) = P(Y)P(X|Y)$
- Learns data distribution per class
- Prediction: $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$ via Bayes' theorem
- Examples: Naive Bayes, GMM, VAE, GAN

**Discriminative Model:**
- Directly models $P(Y|X)$
- Learns only decision boundary
- Examples: Logistic regression, SVM, neural networks

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Generate data
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Generative model: Naive Bayes
generative_model = GaussianNB()
generative_model.fit(X_train, y_train)

# Discriminative model: Logistic Regression
discriminative_model = LogisticRegression()
discriminative_model.fit(X_train, y_train)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

models = [
    (generative_model, 'Generative Model (Naive Bayes)', axes[0]),
    (discriminative_model, 'Discriminative Model (Logistic Regression)', axes[1])
]

for model, title, ax in models:
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
               c='blue', marker='o', s=50, edgecolors='k', label='Class 0')
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
               c='red', marker='s', s=50, edgecolors='k', label='Class 1')

    score = model.score(X_test, y_test)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'{title}\nTest Accuracy: {score:.3f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('generative_vs_discriminative.png', dpi=150)
plt.show()

print("Generative vs Discriminative:")
print("  Generative: Models full distribution P(X,Y) → can generate samples")
print("  Discriminative: Models only P(Y|X) conditional → prediction only, usually higher accuracy")
```

### 6.2 Naive Bayes Classifier

**Assumption**: features are conditionally independent given the class

$$
P(X_1, \ldots, X_d | Y) = \prod_{i=1}^d P(X_i | Y)
$$

**Prediction:**
$$
\hat{y} = \arg\max_y P(Y=y) \prod_{i=1}^d P(X_i | Y=y)
$$

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Text classification example
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Feature extraction (Bag-of-Words)
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, train.target)

# Predict
y_pred = nb_model.predict(X_test)

print("Naive Bayes Text Classification:")
print(classification_report(test.target, y_pred, target_names=test.target_names))

print("\nCharacteristics of Naive Bayes:")
print("  - Conditional independence assumption (naive) → computationally efficient")
print("  - Works well in high-dimensional data (text classification)")
print("  - Probabilistic interpretation available")
print("  - Reasonable performance even with small datasets")
```

### 6.3 Probabilistic Graphical Models

- **Bayesian Network**: represents conditional independence with directed acyclic graph (DAG)
- **Markov Random Field**: undirected graph
- **Hidden Markov Model (HMM)**: inference of hidden states in time series data
- **Applications**: speech recognition, natural language processing, computer vision

```python
# Simple Bayesian network example (conceptual)
import networkx as nx
import matplotlib.pyplot as plt

# Bayesian network structure
# Rain → Sprinkler, Rain → Grass Wet, Sprinkler → Grass Wet
G = nx.DiGraph()
G.add_edges_from([('Rain', 'Sprinkler'), ('Rain', 'Grass Wet'),
                  ('Sprinkler', 'Grass Wet')])

plt.figure(figsize=(10, 6))
pos = {'Rain': (0.5, 1), 'Sprinkler': (0, 0), 'Grass Wet': (1, 0)}
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
        font_size=12, font_weight='bold', arrowsize=20, arrows=True)
plt.title('Bayesian Network: Rain → Sprinkler, Grass Wet', fontsize=14)
plt.tight_layout()
plt.savefig('bayesian_network_example.png', dpi=150)
plt.show()

print("Probabilistic Graphical Models:")
print("  - Represent dependencies between variables as a graph")
print("  - Computational efficiency via conditional independence")
print("  - Inference: estimate hidden variables from observed variables")
print("  - Learning: learn graph structure and probability parameters from data")
```

## Practice Problems

1. **Bayes' Theorem Application**: Design a spam filter using Bayes' theorem. Derive the formula for calculating spam probability given the presence of specific words, and implement with simple example data.

2. **Distribution Fitting**: Use `scipy.stats` to fit a normal distribution to real data (e.g., height, test scores) and verify goodness-of-fit with Q-Q plot. If normal distribution is inadequate, try other distributions.

3. **Monte Carlo Integration**: For $X \sim \mathcal{N}(0, 1)$, compute $\mathbb{E}[e^X]$ (1) analytically and (2) estimate via Monte Carlo sampling. Verify convergence as sample size increases.

4. **Bayesian Linear Regression**: Implement linear regression from a Bayesian perspective. Assign normal prior to weights and update posterior with each data observation. Visualize posterior mean and uncertainty.

5. **Naive Bayes vs Logistic Regression**: Compare performance of Naive Bayes (generative) and logistic regression (discriminative) on Iris dataset. Plot learning curves as training data size varies. Analyze which model is advantageous in which situations.

## References

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press
  - The bible of ML from a probabilistic viewpoint
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer
  - Chapter 1-2: Probability foundations and distributions
- Wasserman, L. (2004). *All of Statistics*. Springer
  - Concise summary of statistics and probability
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press
  - Comprehensive textbook on probabilistic graphical models
- SciPy Stats Documentation: https://docs.scipy.org/doc/scipy/reference/stats.html
- Seeing Theory (probability/statistics visualization): https://seeing-theory.brown.edu/
- Bayesian Methods for Hackers (online book): https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers

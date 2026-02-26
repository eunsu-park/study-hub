# 12. Sampling and Estimation

[Previous: Probability Review](./11_Probability_Review.md) | [Next: Confidence Intervals](./13_Confidence_Intervals.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between population parameters and sample statistics, and explain why sampling is necessary
2. Describe simple random, stratified, and systematic sampling methods
3. Derive the sampling distribution of the sample mean and calculate standard errors
4. Evaluate estimator quality using the properties of unbiasedness, consistency, and efficiency
5. Implement Maximum Likelihood Estimation (MLE) for common distributions (Bernoulli, normal, Poisson, exponential)
6. Apply the Method of Moments (MoM) to estimate parameters of gamma and beta distributions
7. Compare MLE and MoM in terms of bias, variance, and MSE through simulation

---

In practice, you almost never have access to an entire population -- you work with samples. The central challenge of statistics is bridging the gap between what a finite sample tells you and what is true about the whole population. This lesson develops the theoretical machinery for that bridge: sampling distributions that quantify how sample statistics behave, and estimation methods that extract the best possible parameter guesses from limited data.

---

## 1. Population and Sample

### 1.1 Basic Concepts

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Population: All objects of interest
# Sample: A portion extracted from the population

# Example: Height of adult males in South Korea (population)
population_mean = 173.5  # Population mean μ
population_std = 6.0     # Population standard deviation σ

# In practice, we cannot know the entire population
# We estimate parameters through samples

# Sample extraction (simple random sampling)
np.random.seed(42)
sample_size = 100
sample = np.random.normal(population_mean, population_std, sample_size)

print("Population parameters (unknown in practice):")
print(f"  Population mean (μ): {population_mean}")
print(f"  Population std (σ): {population_std}")

print(f"\nSample statistics (n={sample_size}):")
print(f"  Sample mean (x̄): {sample.mean():.2f}")
print(f"  Sample std (s): {sample.std(ddof=1):.2f}")  # ddof=1 for sample std
```

### 1.2 Sampling Methods

```python
# Various sampling methods

# 1. Simple Random Sampling
np.random.seed(42)
population = np.arange(1, 1001)  # Population from 1 to 1000
simple_random_sample = np.random.choice(population, size=50, replace=False)
print(f"Simple random sampling: {simple_random_sample[:10]}...")

# 2. Stratified Sampling
# Divide population into strata and sample from each
strata_A = np.random.normal(50, 10, 500)  # Stratum A
strata_B = np.random.normal(80, 15, 500)  # Stratum B

# Proportional sampling from each stratum
sample_A = np.random.choice(strata_A, size=25, replace=False)
sample_B = np.random.choice(strata_B, size=25, replace=False)
stratified_sample = np.concatenate([sample_A, sample_B])

print(f"\nStratified sampling:")
print(f"  Stratum A sample mean: {sample_A.mean():.2f}")
print(f"  Stratum B sample mean: {sample_B.mean():.2f}")
print(f"  Overall sample mean: {stratified_sample.mean():.2f}")

# 3. Systematic Sampling
# Sample every k-th element
k = 20  # Sampling interval
start = np.random.randint(0, k)
systematic_sample = population[start::k]
print(f"\nSystematic sampling (k={k}): {systematic_sample[:5]}...")
```

---

## 2. Sampling Distribution

### 2.1 Distribution of Sample Mean

```python
# Simulation of sampling distribution of sample mean
np.random.seed(42)

# Population parameters
mu = 100
sigma = 20

# Various sample sizes
sample_sizes = [5, 30, 100]
n_simulations = 10000

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, n in enumerate(sample_sizes):
    # Draw many samples and calculate their means
    sample_means = []
    for _ in range(n_simulations):
        sample = np.random.normal(mu, sigma, n)
        sample_means.append(sample.mean())

    sample_means = np.array(sample_means)

    # Visualization
    axes[i].hist(sample_means, bins=50, density=True, alpha=0.7, color='steelblue')

    # Theoretical distribution: X̄ ~ N(μ, σ²/n)
    theoretical_std = sigma / np.sqrt(n)
    x = np.linspace(mu - 4*theoretical_std, mu + 4*theoretical_std, 100)
    axes[i].plot(x, stats.norm.pdf(x, mu, theoretical_std), 'r-', lw=2)

    axes[i].axvline(mu, color='k', linestyle='--', alpha=0.5)
    axes[i].set_xlabel('Sample mean')
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'n = {n}\nSD(X̄) = σ/√n = {theoretical_std:.2f}')

plt.suptitle('Sampling Distribution of Sample Mean', fontsize=14)
plt.tight_layout()
plt.show()

# Numerical verification
print("\nCharacteristics of sample mean distribution:")
for n in sample_sizes:
    sample_means = [np.random.normal(mu, sigma, n).mean() for _ in range(n_simulations)]
    print(f"n={n:3d}: E[X̄]={np.mean(sample_means):.2f}, SD(X̄)={np.std(sample_means):.2f}, "
          f"Theoretical={sigma/np.sqrt(n):.2f}")
```

### 2.2 Standard Error

```python
# Standard error = standard deviation of sample statistic
# Standard error of sample mean: SE(X̄) = σ/√n (or s/√n)

def standard_error(sample):
    """Calculate standard error of sample mean"""
    n = len(sample)
    s = np.std(sample, ddof=1)  # Sample standard deviation
    se = s / np.sqrt(n)
    return se

# Example
np.random.seed(42)
sample_sizes = [10, 30, 100, 500]

print("Standard error by sample size:")
print("-" * 50)

for n in sample_sizes:
    sample = np.random.normal(100, 20, n)
    se = standard_error(sample)
    theoretical_se = 20 / np.sqrt(n)
    print(f"n = {n:4d}: SE = {se:.3f}, Theoretical = {theoretical_se:.3f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))
n_range = np.arange(10, 501)
se_values = 20 / np.sqrt(n_range)

ax.plot(n_range, se_values, 'b-', linewidth=2)
ax.fill_between(n_range, 0, se_values, alpha=0.2)
ax.set_xlabel('Sample size (n)')
ax.set_ylabel('Standard error (SE)')
ax.set_title('Relationship between sample size and standard error: SE = σ/√n')
ax.grid(alpha=0.3)

# Mark specific points
for n in [30, 100, 400]:
    se = 20 / np.sqrt(n)
    ax.scatter([n], [se], color='red', s=100, zorder=5)
    ax.annotate(f'n={n}, SE={se:.2f}', (n, se), xytext=(n+20, se+0.5))

plt.show()
```

### 2.3 Distribution of Other Sample Statistics

```python
# Distribution of sample variance
np.random.seed(42)

# For normal population (n-1)S²/σ² ~ χ²(n-1)
n = 20
sigma_squared = 100  # Population variance
n_simulations = 10000

chi_squared_values = []
for _ in range(n_simulations):
    sample = np.random.normal(0, np.sqrt(sigma_squared), n)
    s_squared = np.var(sample, ddof=1)  # Sample variance
    chi_squared = (n - 1) * s_squared / sigma_squared
    chi_squared_values.append(chi_squared)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sample variance distribution
sample_variances = [np.var(np.random.normal(0, 10, n), ddof=1) for _ in range(n_simulations)]
axes[0].hist(sample_variances, bins=50, density=True, alpha=0.7)
axes[0].axvline(sigma_squared, color='r', linestyle='--', label=f'σ² = {sigma_squared}')
axes[0].set_xlabel('Sample variance (S²)')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of Sample Variance')
axes[0].legend()

# Chi-squared transformation
axes[1].hist(chi_squared_values, bins=50, density=True, alpha=0.7, label='Simulation')
x = np.linspace(0, 50, 100)
axes[1].plot(x, stats.chi2.pdf(x, df=n-1), 'r-', lw=2, label=f'χ²({n-1})')
axes[1].set_xlabel('(n-1)S²/σ²')
axes[1].set_ylabel('Density')
axes[1].set_title('Chi-squared Transformation of Sample Variance')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Simulation mean: {np.mean(chi_squared_values):.2f}, Theoretical (df=n-1): {n-1}")
print(f"Simulation variance: {np.var(chi_squared_values):.2f}, Theoretical (2*df): {2*(n-1)}")
```

---

## 3. Point Estimation

### 3.1 Properties of Estimators

```python
# Properties of good estimators: Unbiasedness, Consistency, Efficiency

# 1. Unbiasedness: E[θ̂] = θ
# Sample mean is an unbiased estimator of population mean
# Sample variance (divided by n-1) is an unbiased estimator of population variance

np.random.seed(42)
mu, sigma = 50, 10
n = 30
n_simulations = 10000

# Unbiasedness of sample mean
sample_means = [np.random.normal(mu, sigma, n).mean() for _ in range(n_simulations)]
print(f"Expected value of sample mean: {np.mean(sample_means):.4f}, Population mean: {mu}")
print(f"→ Sample mean is an unbiased estimator")

# Comparison of sample variance (divide by n vs n-1)
var_n = []    # Divide by n (biased)
var_n_1 = []  # Divide by n-1 (unbiased)

for _ in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    var_n.append(np.var(sample, ddof=0))
    var_n_1.append(np.var(sample, ddof=1))

print(f"\nddof=0 (divide by n) expected value: {np.mean(var_n):.2f}")
print(f"ddof=1 (divide by n-1) expected value: {np.mean(var_n_1):.2f}")
print(f"Population variance: {sigma**2}")
print(f"→ Sample variance divided by n-1 is unbiased")
```

```python
# 2. Consistency: θ̂ → θ as n → ∞
np.random.seed(42)
mu = 100

sample_sizes = [10, 50, 100, 500, 1000, 5000]
mean_estimates = []
var_estimates = []

for n in sample_sizes:
    sample = np.random.normal(mu, 20, n)
    mean_estimates.append(sample.mean())
    var_estimates.append(np.var(sample, ddof=1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(sample_sizes, mean_estimates, 'bo-')
axes[0].axhline(mu, color='r', linestyle='--', label=f'μ = {mu}')
axes[0].set_xlabel('Sample size (n)')
axes[0].set_ylabel('Sample mean')
axes[0].set_title('Consistency: Sample mean → Population mean')
axes[0].legend()
axes[0].set_xscale('log')

axes[1].plot(sample_sizes, var_estimates, 'go-')
axes[1].axhline(400, color='r', linestyle='--', label='σ² = 400')
axes[1].set_xlabel('Sample size (n)')
axes[1].set_ylabel('Sample variance')
axes[1].set_title('Consistency: Sample variance → Population variance')
axes[1].legend()
axes[1].set_xscale('log')

plt.tight_layout()
plt.show()
```

```python
# 3. Efficiency: More efficient when variance is smaller
# Estimating mean in normal distribution: sample mean vs sample median

np.random.seed(42)
n = 30
n_simulations = 10000
mu = 50

means = []
medians = []

for _ in range(n_simulations):
    sample = np.random.normal(mu, 10, n)
    means.append(sample.mean())
    medians.append(np.median(sample))

print("Efficiency comparison (estimating population mean in normal distribution):")
print(f"Sample mean: Var = {np.var(means):.4f}")
print(f"Sample median: Var = {np.var(medians):.4f}")
print(f"Relative efficiency: {np.var(means) / np.var(medians):.4f}")
print("→ Sample mean is more efficient in normal distribution (ARE ≈ π/2 ≈ 1.57)")
```

### 3.2 Bias-Variance Tradeoff

```python
# MSE = Bias² + Variance

def mse_analysis(estimates, true_value):
    """MSE decomposition"""
    estimates = np.array(estimates)
    bias = np.mean(estimates) - true_value
    variance = np.var(estimates)
    mse = np.mean((estimates - true_value)**2)
    return bias, variance, mse

# Example: Variance estimation (n vs n-1)
np.random.seed(42)
n = 10
sigma_squared = 100
n_simulations = 10000

var_n = [np.var(np.random.normal(0, 10, n), ddof=0) for _ in range(n_simulations)]
var_n_1 = [np.var(np.random.normal(0, 10, n), ddof=1) for _ in range(n_simulations)]

print("MSE analysis of variance estimators:")
print("-" * 60)
print(f"{'Estimator':<15} {'Bias':<12} {'Variance':<12} {'MSE':<12}")
print("-" * 60)

for name, estimates in [("Divide by n", var_n), ("Divide by n-1", var_n_1)]:
    bias, var, mse = mse_analysis(estimates, sigma_squared)
    print(f"{name:<15} {bias:<12.4f} {var:<12.4f} {mse:<12.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(var_n, bins=50, alpha=0.5, label=f'Divide by n (biased)', density=True)
ax.hist(var_n_1, bins=50, alpha=0.5, label=f'Divide by n-1 (unbiased)', density=True)
ax.axvline(sigma_squared, color='k', linestyle='--', linewidth=2, label=f'σ² = {sigma_squared}')
ax.set_xlabel('Estimate')
ax.set_ylabel('Density')
ax.set_title('Bias-Variance Tradeoff')
ax.legend()
plt.show()
```

---

## 4. Maximum Likelihood Estimation (MLE)

### 4.1 MLE Concept

```python
# Likelihood function: L(θ|x) = P(X=x|θ)
# MLE: Find θ that maximizes L(θ)

# Example: MLE for Bernoulli distribution
def bernoulli_likelihood(p, data):
    """Likelihood function of Bernoulli distribution"""
    n = len(data)
    k = sum(data)  # Number of successes
    return (p ** k) * ((1 - p) ** (n - k))

def bernoulli_log_likelihood(p, data):
    """Log-likelihood function of Bernoulli distribution"""
    n = len(data)
    k = sum(data)
    if p <= 0 or p >= 1:
        return -np.inf
    return k * np.log(p) + (n - k) * np.log(1 - p)

# Generate data
np.random.seed(42)
true_p = 0.3
data = np.random.binomial(1, true_p, size=50)
print(f"True p = {true_p}, Observed success rate = {data.mean():.2f}")

# Visualize likelihood function
p_values = np.linspace(0.01, 0.99, 100)
likelihoods = [bernoulli_likelihood(p, data) for p in p_values]
log_likelihoods = [bernoulli_log_likelihood(p, data) for p in p_values]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(p_values, likelihoods, 'b-', linewidth=2)
axes[0].axvline(data.mean(), color='r', linestyle='--', label=f'MLE p̂ = {data.mean():.2f}')
axes[0].axvline(true_p, color='g', linestyle=':', label=f'True p = {true_p}')
axes[0].set_xlabel('p')
axes[0].set_ylabel('L(p)')
axes[0].set_title('Likelihood Function')
axes[0].legend()

axes[1].plot(p_values, log_likelihoods, 'b-', linewidth=2)
axes[1].axvline(data.mean(), color='r', linestyle='--', label=f'MLE p̂ = {data.mean():.2f}')
axes[1].axvline(true_p, color='g', linestyle=':', label=f'True p = {true_p}')
axes[1].set_xlabel('p')
axes[1].set_ylabel('log L(p)')
axes[1].set_title('Log-Likelihood Function')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nBernoulli MLE: p̂ = x̄ = {data.mean():.4f}")
```

### 4.2 MLE for Normal Distribution

```python
from scipy.optimize import minimize

# MLE for normal distribution N(μ, σ²)
# MLE: μ̂ = x̄, σ̂² = (1/n)Σ(x_i - x̄)²

def normal_neg_log_likelihood(params, data):
    """Negative log-likelihood function of normal distribution"""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    n = len(data)
    ll = -n/2 * np.log(2*np.pi) - n*np.log(sigma) - np.sum((data - mu)**2) / (2*sigma**2)
    return -ll

# Generate data
np.random.seed(42)
true_mu, true_sigma = 50, 10
data = np.random.normal(true_mu, true_sigma, 100)

# Numerical optimization
initial_guess = [0, 1]
result = minimize(normal_neg_log_likelihood, initial_guess, args=(data,),
                  method='L-BFGS-B', bounds=[(-np.inf, np.inf), (0.001, np.inf)])

mu_mle, sigma_mle = result.x

# Compare with analytical solution
mu_analytical = data.mean()
sigma_analytical = np.std(data, ddof=0)  # MLE divides by n

print("Normal distribution MLE:")
print("-" * 50)
print(f"True parameters: μ = {true_mu}, σ = {true_sigma}")
print(f"Analytical MLE:  μ̂ = {mu_analytical:.4f}, σ̂ = {sigma_analytical:.4f}")
print(f"Numerical MLE:   μ̂ = {mu_mle:.4f}, σ̂ = {sigma_mle:.4f}")

# Using scipy.stats
mu_fit, sigma_fit = stats.norm.fit(data)
print(f"scipy.stats:     μ̂ = {mu_fit:.4f}, σ̂ = {sigma_fit:.4f}")
```

### 4.3 MLE for Poisson Distribution

```python
# MLE for Poisson distribution Poisson(λ)
# MLE: λ̂ = x̄

def poisson_neg_log_likelihood(lam, data):
    """Negative log-likelihood function of Poisson distribution"""
    if lam <= 0:
        return np.inf
    return -np.sum(stats.poisson.logpmf(data, lam))

# Generate data
np.random.seed(42)
true_lambda = 5
data = np.random.poisson(true_lambda, 100)

# Numerical optimization
result = minimize(poisson_neg_log_likelihood, x0=[1], args=(data,),
                  method='L-BFGS-B', bounds=[(0.001, np.inf)])

lambda_mle = result.x[0]
lambda_analytical = data.mean()

print("Poisson distribution MLE:")
print(f"True λ = {true_lambda}")
print(f"Analytical MLE: λ̂ = {lambda_analytical:.4f}")
print(f"Numerical MLE: λ̂ = {lambda_mle:.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

x_vals = np.arange(0, 15)
ax.bar(x_vals - 0.2, np.bincount(data, minlength=15)[:15] / len(data),
       width=0.4, alpha=0.7, label='Observed frequency')
ax.bar(x_vals + 0.2, stats.poisson.pmf(x_vals, lambda_mle),
       width=0.4, alpha=0.7, label=f'MLE Poisson(λ̂={lambda_mle:.2f})')

ax.set_xlabel('x')
ax.set_ylabel('Probability/Frequency')
ax.set_title('Poisson Distribution MLE Fitting')
ax.legend()
ax.set_xticks(x_vals)
plt.show()
```

### 4.4 MLE for Exponential Distribution

```python
# MLE for exponential distribution Exp(λ)
# MLE: λ̂ = 1/x̄

def exponential_neg_log_likelihood(lam, data):
    """Negative log-likelihood function of exponential distribution"""
    if lam <= 0:
        return np.inf
    n = len(data)
    return -n * np.log(lam) + lam * np.sum(data)

# Generate data
np.random.seed(42)
true_lambda = 0.5
data = np.random.exponential(scale=1/true_lambda, size=100)

# MLE
lambda_mle = 1 / data.mean()

print("Exponential distribution MLE:")
print(f"True λ = {true_lambda}")
print(f"MLE: λ̂ = 1/x̄ = {lambda_mle:.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(data, bins=30, density=True, alpha=0.7, label='Data')
x = np.linspace(0, data.max(), 100)
ax.plot(x, stats.expon.pdf(x, scale=1/lambda_mle), 'r-', lw=2,
        label=f'MLE Exp(λ̂={lambda_mle:.3f})')
ax.plot(x, stats.expon.pdf(x, scale=1/true_lambda), 'g--', lw=2,
        label=f'True Exp(λ={true_lambda})')

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Exponential Distribution MLE Fitting')
ax.legend()
plt.show()
```

### 4.5 Asymptotic Properties of MLE

```python
# Asymptotic normality of MLE
# √n(θ̂_MLE - θ) → N(0, I(θ)^{-1}) (Fisher information)

# Simulation: Asymptotic distribution of MLE for normal mean
np.random.seed(42)
true_mu = 50
true_sigma = 10
sample_sizes = [10, 30, 100, 500]
n_simulations = 5000

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, n in enumerate(sample_sizes):
    mle_estimates = []
    for _ in range(n_simulations):
        sample = np.random.normal(true_mu, true_sigma, n)
        mle_estimates.append(sample.mean())

    # Standardize
    standardized = np.sqrt(n) * (np.array(mle_estimates) - true_mu) / true_sigma

    axes[i].hist(standardized, bins=50, density=True, alpha=0.7)
    x = np.linspace(-4, 4, 100)
    axes[i].plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
    axes[i].set_xlabel('√n(μ̂ - μ)/σ')
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'n = {n}')
    axes[i].legend()
    axes[i].set_xlim(-4, 4)

plt.suptitle('Asymptotic Normality of MLE', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 5. Method of Moments (MoM)

### 5.1 Method of Moments Concept

```python
# Method of Moments: Estimate parameters by setting sample moments = population moments
# k-th sample moment: m_k = (1/n)Σx_i^k
# k-th central moment: m'_k = (1/n)Σ(x_i - x̄)^k

def sample_moments(data, k):
    """Calculate k-th sample moment"""
    return np.mean(data ** k)

def sample_central_moments(data, k):
    """Calculate k-th sample central moment"""
    return np.mean((data - data.mean()) ** k)

# Example data
np.random.seed(42)
data = np.random.gamma(shape=5, scale=2, size=1000)

print("Sample moments:")
for k in range(1, 5):
    print(f"  {k}-th moment: {sample_moments(data, k):.4f}")

print("\nSample central moments:")
for k in range(2, 5):
    print(f"  {k}-th central moment: {sample_central_moments(data, k):.4f}")
```

### 5.2 Method of Moments for Gamma Distribution

```python
# Gamma distribution Gamma(α, β): E[X] = α/β, Var(X) = α/β²
# MoM: α̂ = x̄² / s², β̂ = x̄ / s²

def gamma_mom_estimator(data):
    """Method of moments estimator for gamma distribution"""
    x_bar = data.mean()
    s_squared = data.var(ddof=1)

    alpha_hat = x_bar ** 2 / s_squared
    beta_hat = x_bar / s_squared

    return alpha_hat, beta_hat

# Generate data
np.random.seed(42)
true_alpha, true_beta = 5, 2
data = np.random.gamma(shape=true_alpha, scale=1/true_beta, size=500)

# Method of moments
alpha_mom, beta_mom = gamma_mom_estimator(data)

# MLE (scipy)
alpha_mle, loc_mle, scale_mle = stats.gamma.fit(data, floc=0)
beta_mle = 1 / scale_mle

print("Gamma distribution parameter estimation:")
print("-" * 50)
print(f"{'Method':<15} {'α':<12} {'β':<12}")
print("-" * 50)
print(f"{'True':<15} {true_alpha:<12} {true_beta:<12}")
print(f"{'MoM':<15} {alpha_mom:<12.4f} {beta_mom:<12.4f}")
print(f"{'MLE':<15} {alpha_mle:<12.4f} {beta_mle:<12.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(data, bins=50, density=True, alpha=0.7, label='Data')
x = np.linspace(0, data.max(), 100)
ax.plot(x, stats.gamma.pdf(x, a=true_alpha, scale=1/true_beta), 'g--', lw=2,
        label=f'True: α={true_alpha}, β={true_beta}')
ax.plot(x, stats.gamma.pdf(x, a=alpha_mom, scale=1/beta_mom), 'r-', lw=2,
        label=f'MoM: α={alpha_mom:.2f}, β={beta_mom:.2f}')
ax.plot(x, stats.gamma.pdf(x, a=alpha_mle, scale=1/beta_mle), 'b:', lw=2,
        label=f'MLE: α={alpha_mle:.2f}, β={beta_mle:.2f}')

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Gamma Distribution Estimation: MoM vs MLE')
ax.legend()
plt.show()
```

### 5.3 Method of Moments for Beta Distribution

```python
# Beta distribution Beta(α, β): E[X] = α/(α+β), Var(X) = αβ/((α+β)²(α+β+1))
# Derive MoM

def beta_mom_estimator(data):
    """Method of moments estimator for beta distribution"""
    x_bar = data.mean()
    s_squared = data.var(ddof=1)

    # Solution to moment equations
    temp = (x_bar * (1 - x_bar) / s_squared) - 1
    alpha_hat = x_bar * temp
    beta_hat = (1 - x_bar) * temp

    return alpha_hat, beta_hat

# Generate data
np.random.seed(42)
true_alpha, true_beta = 2, 5
data = np.random.beta(true_alpha, true_beta, size=500)

# Method of moments
alpha_mom, beta_mom = beta_mom_estimator(data)

# MLE (scipy)
alpha_mle, beta_mle, loc_mle, scale_mle = stats.beta.fit(data, floc=0, fscale=1)

print("Beta distribution parameter estimation:")
print("-" * 50)
print(f"{'Method':<15} {'α':<12} {'β':<12}")
print("-" * 50)
print(f"{'True':<15} {true_alpha:<12} {true_beta:<12}")
print(f"{'MoM':<15} {alpha_mom:<12.4f} {beta_mom:<12.4f}")
print(f"{'MLE':<15} {alpha_mle:<12.4f} {beta_mle:<12.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(data, bins=30, density=True, alpha=0.7, label='Data')
x = np.linspace(0.001, 0.999, 100)
ax.plot(x, stats.beta.pdf(x, true_alpha, true_beta), 'g--', lw=2,
        label=f'True: α={true_alpha}, β={true_beta}')
ax.plot(x, stats.beta.pdf(x, alpha_mom, beta_mom), 'r-', lw=2,
        label=f'MoM: α={alpha_mom:.2f}, β={beta_mom:.2f}')
ax.plot(x, stats.beta.pdf(x, alpha_mle, beta_mle), 'b:', lw=2,
        label=f'MLE: α={alpha_mle:.2f}, β={beta_mle:.2f}')

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Beta Distribution Estimation: MoM vs MLE')
ax.legend()
plt.show()
```

### 5.4 MLE vs MoM Comparison

```python
# Simulation comparing MLE vs MoM

np.random.seed(42)
true_alpha, true_beta = 3, 2
sample_sizes = [20, 50, 100, 500]
n_simulations = 1000

results = []

for n in sample_sizes:
    mle_alpha, mle_beta = [], []
    mom_alpha, mom_beta = [], []

    for _ in range(n_simulations):
        data = np.random.gamma(shape=true_alpha, scale=1/true_beta, size=n)

        # MoM
        a_mom, b_mom = gamma_mom_estimator(data)
        mom_alpha.append(a_mom)
        mom_beta.append(b_mom)

        # MLE
        try:
            a_mle, _, scale_mle = stats.gamma.fit(data, floc=0)
            b_mle = 1 / scale_mle
            mle_alpha.append(a_mle)
            mle_beta.append(b_mle)
        except:
            pass

    results.append({
        'n': n,
        'MoM_alpha_bias': np.mean(mom_alpha) - true_alpha,
        'MoM_alpha_var': np.var(mom_alpha),
        'MLE_alpha_bias': np.mean(mle_alpha) - true_alpha,
        'MLE_alpha_var': np.var(mle_alpha),
    })

# Display results
import pandas as pd
df_results = pd.DataFrame(results)
print("MLE vs MoM comparison (Gamma distribution α estimation):")
print(df_results.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bias
axes[0].plot(df_results['n'], np.abs(df_results['MoM_alpha_bias']), 'r-o', label='MoM')
axes[0].plot(df_results['n'], np.abs(df_results['MLE_alpha_bias']), 'b-s', label='MLE')
axes[0].set_xlabel('Sample size (n)')
axes[0].set_ylabel('|Bias|')
axes[0].set_title('Bias Comparison')
axes[0].legend()
axes[0].set_xscale('log')

# Variance
axes[1].plot(df_results['n'], df_results['MoM_alpha_var'], 'r-o', label='MoM')
axes[1].plot(df_results['n'], df_results['MLE_alpha_var'], 'b-s', label='MLE')
axes[1].set_xlabel('Sample size (n)')
axes[1].set_ylabel('Variance')
axes[1].set_title('Variance Comparison')
axes[1].legend()
axes[1].set_xscale('log')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()
```

---

## 6. Practical Example: Distribution Fitting

### 6.1 Comparing Multiple Distribution Fits

```python
# Fit multiple distributions to data and compare

np.random.seed(42)

# Real data (generated from gamma distribution)
true_data = np.random.gamma(shape=3, scale=2, size=500)

# Fit multiple distributions
distributions = {
    'Normal': stats.norm,
    'Gamma': stats.gamma,
    'Lognormal': stats.lognorm,
    'Exponential': stats.expon,
    'Weibull': stats.weibull_min
}

fit_results = {}

for name, dist in distributions.items():
    try:
        params = dist.fit(true_data)
        # Calculate log-likelihood
        log_likelihood = np.sum(dist.logpdf(true_data, *params))
        # AIC = -2*logL + 2*k
        k = len(params)
        aic = -2 * log_likelihood + 2 * k
        fit_results[name] = {'params': params, 'logL': log_likelihood, 'AIC': aic}
    except Exception as e:
        fit_results[name] = {'error': str(e)}

# Display results
print("Distribution fitting results (sorted by AIC):")
print("-" * 60)
print(f"{'Distribution':<15} {'Log-Likelihood':<18} {'AIC':<12}")
print("-" * 60)

sorted_results = sorted(fit_results.items(), key=lambda x: x[1].get('AIC', np.inf))
for name, result in sorted_results:
    if 'AIC' in result:
        print(f"{name:<15} {result['logL']:<18.2f} {result['AIC']:<12.2f}")

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(true_data, bins=50, density=True, alpha=0.5, label='Data', color='gray')
x = np.linspace(0.01, true_data.max(), 200)

colors = plt.cm.Set1(np.linspace(0, 1, len(distributions)))
for (name, dist), color in zip(distributions.items(), colors):
    if name in fit_results and 'params' in fit_results[name]:
        params = fit_results[name]['params']
        ax.plot(x, dist.pdf(x, *params), linewidth=2, color=color,
                label=f'{name} (AIC={fit_results[name]["AIC"]:.1f})')

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Comparison of Multiple Distribution Fits')
ax.legend()
ax.set_xlim(0, true_data.max())
plt.show()

print(f"\nBest fitting distribution: {sorted_results[0][0]}")
```

### 6.2 Goodness-of-Fit Tests

```python
# Kolmogorov-Smirnov test and Anderson-Darling test

# Goodness-of-fit test for normal distribution
np.random.seed(42)
data_normal = np.random.normal(50, 10, 200)
data_skewed = np.random.gamma(2, 5, 200)

def goodness_of_fit_tests(data, dist_name='norm'):
    """Perform goodness-of-fit tests"""
    # Fit distribution
    if dist_name == 'norm':
        params = stats.norm.fit(data)
        dist = stats.norm(*params)
    elif dist_name == 'expon':
        params = stats.expon.fit(data)
        dist = stats.expon(*params)

    # KS test
    ks_stat, ks_pvalue = stats.kstest(data, dist.cdf)

    # Shapiro-Wilk test (for normality)
    if dist_name == 'norm':
        sw_stat, sw_pvalue = stats.shapiro(data)
    else:
        sw_stat, sw_pvalue = np.nan, np.nan

    return {
        'KS_statistic': ks_stat,
        'KS_pvalue': ks_pvalue,
        'SW_statistic': sw_stat,
        'SW_pvalue': sw_pvalue
    }

print("Goodness-of-fit test results:")
print("=" * 60)

print("\nNormal data:")
results_normal = goodness_of_fit_tests(data_normal, 'norm')
print(f"  KS test: statistic={results_normal['KS_statistic']:.4f}, p={results_normal['KS_pvalue']:.4f}")
print(f"  SW test: statistic={results_normal['SW_statistic']:.4f}, p={results_normal['SW_pvalue']:.4f}")
print(f"  → Fits normal distribution (p > 0.05)")

print("\nSkewed data:")
results_skewed = goodness_of_fit_tests(data_skewed, 'norm')
print(f"  KS test: statistic={results_skewed['KS_statistic']:.4f}, p={results_skewed['KS_pvalue']:.4f}")
print(f"  SW test: statistic={results_skewed['SW_statistic']:.4f}, p={results_skewed['SW_pvalue']:.4f}")
print(f"  → Does not fit normal distribution (p < 0.05)")

# Q-Q plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(data_normal, dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot for Normal Data')
axes[0].grid(alpha=0.3)

stats.probplot(data_skewed, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot for Skewed Data')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Practice Problems

### Problem 1: Sampling Distribution
For a normal population with mean 100 and standard deviation 15, when drawing a sample of size 36:
- (a) What are the expected value and standard error of the sample mean?
- (b) What is the probability that the sample mean is between 95 and 105?

### Problem 2: MLE
Given the following data follows a Poisson distribution, find the MLE for λ and estimate a 95% confidence interval.
```python
data = [3, 5, 4, 2, 6, 3, 4, 5, 2, 4]
```

### Problem 3: Method of Moments
Derive the method of moments estimators for a and b in the uniform distribution U(a, b).
(Hint: E[X] = (a+b)/2, Var(X) = (b-a)²/12)

---

## Summary

| Concept | Key Content | Python Function |
|---------|-------------|-----------------|
| Sample mean distribution | X̄ ~ N(μ, σ²/n) | Simulation |
| Standard error | SE = σ/√n | `s / np.sqrt(n)` |
| Unbiasedness | E[θ̂] = θ | Use `ddof=1` |
| Consistency | θ̂ → θ as n → ∞ | Simulation |
| MLE | Maximize L(θ) | `scipy.stats.fit()` |
| Method of moments | Sample moments = Population moments | Derive formula |
| Goodness-of-fit | KS, Anderson-Darling | `stats.kstest()` |

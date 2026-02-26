# 11. Probability Review

[Previous: From EDA to Inference](./10_From_EDA_to_Inference.md) | [Next: Sampling and Estimation](./12_Sampling_and_Estimation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the three Kolmogorov axioms and verify them programmatically
2. Distinguish between discrete and continuous random variables and their distribution functions (PMF, PDF, CDF)
3. Describe the major probability distributions (binomial, Poisson, normal, exponential, gamma, beta) and their parameters
4. Calculate joint, marginal, and conditional probabilities and apply Bayes' theorem
5. Compute expected value, variance, covariance, and correlation for random variables
6. Apply the Law of Large Numbers and Central Limit Theorem through simulation
7. Implement distribution fitting and use `scipy.stats` distribution objects effectively

---

Probability theory underpins every statistical method you will encounter in the remaining lessons. Without a firm grasp of distributions, expectations, and limit theorems, techniques like hypothesis testing, regression, and Bayesian inference remain opaque recipes rather than reasoned tools. This review ensures that your probabilistic foundations are solid before we build on them.

---

## 1. Axioms of Probability

### 1.1 Sample Space and Events

**Sample Space**: The set of all possible outcomes Ω

**Event**: A subset of the sample space

```python
# Example: Dice roll
sample_space = {1, 2, 3, 4, 5, 6}  # Ω

# Define events
event_even = {2, 4, 6}  # Event of even numbers
event_greater_than_4 = {5, 6}  # Event of numbers greater than 4

# Union and intersection
event_union = event_even | event_greater_than_4  # {2, 4, 5, 6}
event_intersection = event_even & event_greater_than_4  # {6}
```

### 1.2 Kolmogorov Axioms

Three axioms:

1. **Non-negativity**: P(A) ≥ 0 (for all events A)
2. **Normalization**: P(Ω) = 1 (probability of entire sample space is 1)
3. **Countable additivity**: Probability of union of disjoint events = sum of individual probabilities

```python
import numpy as np

def verify_probability_axioms(probabilities: dict) -> bool:
    """Verify probability axioms"""
    probs = list(probabilities.values())

    # Axiom 1: Non-negativity
    axiom1 = all(p >= 0 for p in probs)

    # Axiom 2: Normalization (sum = 1)
    axiom2 = np.isclose(sum(probs), 1.0)

    print(f"Axiom 1 (Non-negativity): {axiom1}")
    print(f"Axiom 2 (Normalization, sum={sum(probs):.4f}): {axiom2}")

    return axiom1 and axiom2

# Dice probabilities
dice_probs = {i: 1/6 for i in range(1, 7)}
verify_probability_axioms(dice_probs)
```

### 1.3 Basic Properties of Probability

```python
# P(A^c) = 1 - P(A) (complement)
P_A = 0.3
P_A_complement = 1 - P_A
print(f"P(A) = {P_A}, P(A^c) = {P_A_complement}")

# P(A ∪ B) = P(A) + P(B) - P(A ∩ B) (inclusion-exclusion principle)
P_A, P_B, P_A_and_B = 0.5, 0.4, 0.2
P_A_or_B = P_A + P_B - P_A_and_B
print(f"P(A ∪ B) = {P_A_or_B}")
```

---

## 2. Random Variables

### 2.1 Discrete Random Variable

When a random variable X takes only countable values

**Probability Mass Function (PMF)**: P(X = x) = f(x)

```python
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# Discrete random variable example: Binomial distribution
n, p = 10, 0.3
X = stats.binom(n=n, p=p)

# Calculate PMF
x_values = np.arange(0, n+1)
pmf_values = X.pmf(x_values)

# Expected value and variance
print(f"Binomial distribution B({n}, {p})")
print(f"E[X] = np = {n*p}")
print(f"Var(X) = np(1-p) = {n*p*(1-p)}")
print(f"scipy calculation: mean={X.mean():.2f}, variance={X.var():.2f}")

# Visualization
plt.figure(figsize=(10, 4))
plt.bar(x_values, pmf_values, color='steelblue', alpha=0.7)
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.title(f'Binomial Distribution PMF: B({n}, {p})')
plt.xticks(x_values)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### 2.2 Continuous Random Variable

When a random variable X takes continuous values

**Probability Density Function (PDF)**: f(x), P(a ≤ X ≤ b) = ∫[a,b] f(x)dx

```python
# Continuous random variable example: Normal distribution
mu, sigma = 0, 1
X = stats.norm(loc=mu, scale=sigma)

# Calculate PDF
x_values = np.linspace(-4, 4, 1000)
pdf_values = X.pdf(x_values)

# Probability of a specific interval
prob_between = X.cdf(1) - X.cdf(-1)  # P(-1 ≤ X ≤ 1)
print(f"P(-1 ≤ X ≤ 1) = {prob_between:.4f}")  # approximately 68.27%

# Visualization
plt.figure(figsize=(10, 4))
plt.plot(x_values, pdf_values, 'b-', linewidth=2)
plt.fill_between(x_values, pdf_values, where=(x_values >= -1) & (x_values <= 1),
                  alpha=0.3, color='steelblue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Standard Normal Distribution PDF')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.grid(alpha=0.3)
plt.show()
```

### 2.3 Cumulative Distribution Function (CDF)

```python
# CDF: F(x) = P(X ≤ x)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Discrete distribution CDF (Binomial)
n, p = 10, 0.5
X_binom = stats.binom(n=n, p=p)
x_discrete = np.arange(0, n+2)
cdf_discrete = X_binom.cdf(x_discrete)

axes[0].step(x_discrete, cdf_discrete, where='post', color='steelblue', linewidth=2)
axes[0].scatter(x_discrete[:-1], cdf_discrete[:-1], color='steelblue', s=50, zorder=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('F(x)')
axes[0].set_title(f'Binomial Distribution CDF: B({n}, {p})')
axes[0].grid(alpha=0.3)

# Continuous distribution CDF (Normal)
X_norm = stats.norm(0, 1)
x_cont = np.linspace(-4, 4, 1000)
cdf_cont = X_norm.cdf(x_cont)

axes[1].plot(x_cont, cdf_cont, 'b-', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('F(x)')
axes[1].set_title('Standard Normal Distribution CDF')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3. Major Probability Distributions

### 3.1 Discrete Distributions

#### Binomial Distribution

Number of successes in n independent Bernoulli trials

```python
# Binomial distribution: X ~ B(n, p)
n, p = 20, 0.4

X = stats.binom(n=n, p=p)

print(f"Binomial distribution B({n}, {p})")
print(f"Mean: E[X] = {X.mean():.2f}")
print(f"Variance: Var(X) = {X.var():.2f}")
print(f"P(X = 8) = {X.pmf(8):.4f}")
print(f"P(X ≤ 8) = {X.cdf(8):.4f}")
print(f"P(X > 8) = {1 - X.cdf(8):.4f}")

# Generate random samples
samples = X.rvs(size=1000, random_state=42)
print(f"\nMean of 1000 samples: {np.mean(samples):.2f}")
```

#### Poisson Distribution

Number of events occurring at an average rate of λ per unit time/space

```python
# Poisson distribution: X ~ Poisson(λ)
lambda_param = 5

X = stats.poisson(mu=lambda_param)

print(f"Poisson distribution Poisson({lambda_param})")
print(f"Mean: E[X] = {X.mean():.2f}")
print(f"Variance: Var(X) = {X.var():.2f}")
print(f"P(X = 3) = {X.pmf(3):.4f}")
print(f"P(X ≤ 3) = {X.cdf(3):.4f}")

# Comparison of Poisson and Binomial distributions (when n is large and p is small)
n, p = 100, 0.05  # np = 5 = λ
X_binom = stats.binom(n=n, p=p)
X_poisson = stats.poisson(mu=n*p)

x_vals = np.arange(0, 15)
plt.figure(figsize=(10, 4))
plt.bar(x_vals - 0.2, X_binom.pmf(x_vals), width=0.4, label=f'Binom({n},{p})', alpha=0.7)
plt.bar(x_vals + 0.2, X_poisson.pmf(x_vals), width=0.4, label=f'Poisson({n*p})', alpha=0.7)
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.title('Comparison of Binomial and Poisson Distributions (large n, small p)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

#### Geometric Distribution

Number of trials until the first success

```python
# Geometric distribution: X ~ Geom(p)
p = 0.3

X = stats.geom(p=p)

print(f"Geometric distribution Geom({p})")
print(f"Mean: E[X] = 1/p = {1/p:.2f}")
print(f"Variance: Var(X) = (1-p)/p² = {(1-p)/p**2:.2f}")
print(f"P(X = 3) = {X.pmf(3):.4f}")  # First success on 3rd trial

# Memoryless property
# P(X > s+t | X > s) = P(X > t)
s, t = 2, 3
P_Xgt_s_plus_t = 1 - X.cdf(s+t)
P_Xgt_s = 1 - X.cdf(s)
P_Xgt_t = 1 - X.cdf(t)

print(f"\nMemoryless property verification:")
print(f"P(X > {s+t}) / P(X > {s}) = {P_Xgt_s_plus_t / P_Xgt_s:.4f}")
print(f"P(X > {t}) = {P_Xgt_t:.4f}")
```

### 3.2 Continuous Distributions

#### Normal Distribution

The most important continuous distribution, key to the Central Limit Theorem

```python
# Normal distribution: X ~ N(μ, σ²)
mu, sigma = 100, 15  # IQ example

X = stats.norm(loc=mu, scale=sigma)

print(f"Normal distribution N({mu}, {sigma}²)")
print(f"Mean: E[X] = {X.mean():.2f}")
print(f"Standard deviation: SD(X) = {X.std():.2f}")

# Standardization and percentiles
x_val = 130
z_score = (x_val - mu) / sigma
print(f"\nZ-score of X = {x_val}: {z_score:.2f}")
print(f"P(X ≤ {x_val}) = {X.cdf(x_val):.4f}")

# Finding percentiles
percentiles = [0.25, 0.50, 0.75, 0.95]
for p in percentiles:
    print(f"{int(p*100)}th percentile: {X.ppf(p):.2f}")

# Empirical rule (68-95-99.7 rule)
print("\nEmpirical rule:")
print(f"P(μ-σ < X < μ+σ) = {X.cdf(mu+sigma) - X.cdf(mu-sigma):.4f} (≈ 68%)")
print(f"P(μ-2σ < X < μ+2σ) = {X.cdf(mu+2*sigma) - X.cdf(mu-2*sigma):.4f} (≈ 95%)")
print(f"P(μ-3σ < X < μ+3σ) = {X.cdf(mu+3*sigma) - X.cdf(mu-3*sigma):.4f} (≈ 99.7%)")
```

#### Exponential Distribution

Waiting time until an event occurs

```python
# Exponential distribution: X ~ Exp(λ) - scipy uses scale = 1/λ
lambda_param = 0.5  # Mean waiting time = 2
scale = 1 / lambda_param

X = stats.expon(scale=scale)

print(f"Exponential distribution Exp({lambda_param})")
print(f"Mean: E[X] = 1/λ = {X.mean():.2f}")
print(f"Variance: Var(X) = 1/λ² = {X.var():.2f}")

# Memoryless property (connection with Poisson process)
print("\nMemoryless property: Even after waiting t time, additional waiting time distribution is the same")

# Relationship between exponential and Poisson distributions
# Poisson: number of events per unit time | Exponential: waiting time until event
plt.figure(figsize=(10, 4))
x_vals = np.linspace(0, 10, 1000)
for lam in [0.5, 1.0, 2.0]:
    pdf_vals = stats.expon(scale=1/lam).pdf(x_vals)
    plt.plot(x_vals, pdf_vals, label=f'λ = {lam}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Exponential Distribution PDF (various λ)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

#### Gamma and Beta Distributions

```python
# Gamma distribution: X ~ Gamma(α, β)
# Mean = α/β, Variance = α/β²
alpha, beta_param = 5, 2

X_gamma = stats.gamma(a=alpha, scale=1/beta_param)
print(f"Gamma distribution Gamma({alpha}, {beta_param})")
print(f"Mean: {X_gamma.mean():.2f}, Variance: {X_gamma.var():.2f}")

# Beta distribution: X ~ Beta(α, β) - modeling probabilities between 0 and 1
# Mean = α/(α+β)
alpha, beta_param = 2, 5

X_beta = stats.beta(a=alpha, b=beta_param)
print(f"\nBeta distribution Beta({alpha}, {beta_param})")
print(f"Mean: {X_beta.mean():.2f}, Variance: {X_beta.var():.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gamma distribution
x_gamma = np.linspace(0, 10, 1000)
for a in [1, 2, 5]:
    axes[0].plot(x_gamma, stats.gamma(a=a, scale=1).pdf(x_gamma), label=f'α={a}, β=1')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Gamma Distribution PDF')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Beta distribution
x_beta = np.linspace(0, 1, 1000)
beta_params = [(0.5, 0.5), (1, 1), (2, 5), (5, 2)]
for a, b in beta_params:
    axes[1].plot(x_beta, stats.beta(a=a, b=b).pdf(x_beta), label=f'α={a}, β={b}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title('Beta Distribution PDF')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. Joint and Conditional Probability

### 4.1 Joint Distribution

```python
import pandas as pd

# Discrete joint probability distribution example
# X: Education level (1=high school, 2=bachelor's, 3=graduate)
# Y: Income level (1=low, 2=medium, 3=high)

# Joint probability mass function (joint PMF)
joint_pmf = np.array([
    [0.10, 0.15, 0.05],  # X=1
    [0.05, 0.20, 0.15],  # X=2
    [0.02, 0.08, 0.20]   # X=3
])

df_joint = pd.DataFrame(joint_pmf,
                        index=['X=1(HS)', 'X=2(BS)', 'X=3(Grad)'],
                        columns=['Y=1(Low)', 'Y=2(Med)', 'Y=3(High)'])

print("Joint probability distribution:")
print(df_joint)
print(f"\nTotal: {joint_pmf.sum():.2f}")

# Marginal probability
marginal_X = joint_pmf.sum(axis=1)
marginal_Y = joint_pmf.sum(axis=0)

print(f"\nMarginal probability of X: {marginal_X}")
print(f"Marginal probability of Y: {marginal_Y}")
```

### 4.2 Conditional Probability

P(A|B) = P(A ∩ B) / P(B)

```python
# Conditional probability mass function
# P(Y|X=2): Income distribution given bachelor's degree
P_Y_given_X2 = joint_pmf[1, :] / marginal_X[1]
print(f"P(Y|X=BS) = {P_Y_given_X2}")
print(f"Sum: {P_Y_given_X2.sum():.2f}")

# P(X|Y=3): Education distribution given high income
P_X_given_Y3 = joint_pmf[:, 2] / marginal_Y[2]
print(f"P(X|Y=High) = {P_X_given_Y3}")

# Bayes' theorem
# P(X=Grad|Y=High) = P(Y=High|X=Grad) * P(X=Grad) / P(Y=High)
P_Y3_given_X3 = joint_pmf[2, 2] / marginal_X[2]
P_X3 = marginal_X[2]
P_Y3 = marginal_Y[2]

P_X3_given_Y3_bayes = (P_Y3_given_X3 * P_X3) / P_Y3
print(f"\nP(X=Grad|Y=High) using Bayes' theorem = {P_X3_given_Y3_bayes:.4f}")
print(f"Direct calculation: {P_X_given_Y3[2]:.4f}")
```

### 4.3 Independence

Two random variables X, Y are independent if: P(X, Y) = P(X)P(Y)

```python
def check_independence(joint_pmf):
    """Test independence of joint distribution"""
    marginal_X = joint_pmf.sum(axis=1)
    marginal_Y = joint_pmf.sum(axis=0)

    # If independent, P(X,Y) = P(X)P(Y)
    expected_if_independent = np.outer(marginal_X, marginal_Y)

    print("Expected joint probability under independence:")
    print(np.round(expected_if_independent, 4))
    print("\nActual joint probability:")
    print(np.round(joint_pmf, 4))

    # Calculate difference
    diff = np.abs(joint_pmf - expected_if_independent)
    print(f"\nMaximum difference: {diff.max():.4f}")
    print(f"Independent: {'Yes' if np.allclose(joint_pmf, expected_if_independent) else 'No'}")

check_independence(joint_pmf)
```

---

## 5. Expected Value, Variance, and Covariance

### 5.1 Expected Value

E[X] = Σ x·P(X=x) (discrete) or ∫ x·f(x)dx (continuous)

```python
# Expected value of discrete random variable
def expected_value_discrete(values, probabilities):
    """Calculate expected value of discrete random variable"""
    return np.sum(values * probabilities)

# Example: Lottery expected value
# Prizes: 0 (prob 0.9), 1000 (0.08), 10000 (0.019), 100000 (0.001)
prizes = np.array([0, 1000, 10000, 100000])
probs = np.array([0.9, 0.08, 0.019, 0.001])

E_X = expected_value_discrete(prizes, probs)
print(f"Expected lottery prize: {E_X:.0f} won")
print(f"If lottery costs 500 won, expected profit: {E_X - 500:.0f} won")

# Linearity of expected value
# E[aX + b] = aE[X] + b
a, b = 2, 100
print(f"\nE[2X + 100] = 2*E[X] + 100 = {2*E_X + 100:.0f} won")
```

### 5.2 Variance and Standard Deviation

Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

```python
def variance_discrete(values, probabilities):
    """Calculate variance of discrete random variable"""
    E_X = np.sum(values * probabilities)
    E_X2 = np.sum(values**2 * probabilities)
    return E_X2 - E_X**2

Var_X = variance_discrete(prizes, probs)
SD_X = np.sqrt(Var_X)

print(f"Variance: Var(X) = {Var_X:.0f}")
print(f"Standard deviation: SD(X) = {SD_X:.0f} won")

# Properties of variance
# Var(aX + b) = a²Var(X)
print(f"\nVar(2X + 100) = 4*Var(X) = {4*Var_X:.0f}")
```

### 5.3 Covariance and Correlation

```python
# Covariance and correlation for continuous variables
np.random.seed(42)

# Generate data with positive correlation
n = 500
X = np.random.normal(50, 10, n)
Y = 0.8 * X + np.random.normal(0, 5, n)

# Covariance: Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)]
cov_XY = np.cov(X, Y, ddof=1)[0, 1]
print(f"Covariance: Cov(X,Y) = {cov_XY:.2f}")

# Correlation coefficient: ρ = Cov(X,Y) / (SD(X) * SD(Y))
corr_XY = np.corrcoef(X, Y)[0, 1]
print(f"Correlation coefficient: ρ = {corr_XY:.4f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Positive correlation
axes[0].scatter(X, Y, alpha=0.5, s=10)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title(f'Positive Correlation (ρ = {corr_XY:.2f})')

# Negative correlation
Y_neg = -0.8 * X + 100 + np.random.normal(0, 5, n)
corr_neg = np.corrcoef(X, Y_neg)[0, 1]
axes[1].scatter(X, Y_neg, alpha=0.5, s=10, color='red')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title(f'Negative Correlation (ρ = {corr_neg:.2f})')

# No correlation
Y_none = np.random.normal(50, 10, n)
corr_none = np.corrcoef(X, Y_none)[0, 1]
axes[2].scatter(X, Y_none, alpha=0.5, s=10, color='green')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].set_title(f'No Correlation (ρ = {corr_none:.2f})')

plt.tight_layout()
plt.show()
```

### 5.4 Properties of Expected Value and Variance

```python
# Sum of two random variables
# E[X + Y] = E[X] + E[Y] (always holds)
# Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
# If independent: Var(X + Y) = Var(X) + Var(Y)

np.random.seed(42)
X = np.random.normal(100, 10, 10000)
Y = np.random.normal(50, 5, 10000)  # Independent

print("Two independent random variables X, Y:")
print(f"E[X] = {X.mean():.2f}, E[Y] = {Y.mean():.2f}")
print(f"E[X+Y] = {(X+Y).mean():.2f} ≈ E[X] + E[Y] = {X.mean() + Y.mean():.2f}")

print(f"\nVar(X) = {X.var():.2f}, Var(Y) = {Y.var():.2f}")
print(f"Var(X+Y) = {(X+Y).var():.2f} ≈ Var(X) + Var(Y) = {X.var() + Y.var():.2f}")
```

---

## 6. Law of Large Numbers and Central Limit Theorem

### 6.1 Law of Large Numbers

As sample size increases, the sample mean converges to the population mean

```python
# Simulation of Law of Large Numbers
np.random.seed(42)

# Population: Dice (expected value = 3.5)
population_mean = 3.5

# Calculate mean for various sample sizes
sample_sizes = np.logspace(1, 5, 50, dtype=int)
sample_means = []

for n in sample_sizes:
    sample = np.random.randint(1, 7, size=n)
    sample_means.append(sample.mean())

# Visualization
plt.figure(figsize=(10, 5))
plt.semilogx(sample_sizes, sample_means, 'b-', alpha=0.7, linewidth=1)
plt.axhline(y=population_mean, color='r', linestyle='--', label=f'Population mean = {population_mean}')
plt.xlabel('Sample size (n)')
plt.ylabel('Sample mean')
plt.title('Law of Large Numbers: Convergence of Sample Mean to Population Mean')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"n=10: Sample mean = {np.random.randint(1, 7, 10).mean():.2f}")
print(f"n=100: Sample mean = {np.random.randint(1, 7, 100).mean():.2f}")
print(f"n=10000: Sample mean = {np.random.randint(1, 7, 10000).mean():.2f}")
```

### 6.2 Central Limit Theorem

For sufficiently large sample size, the distribution of sample means approximates a normal distribution

X̄ ~ N(μ, σ²/n) approximately, for large n

```python
def demonstrate_clt(distribution, params, dist_name, n_samples=1000):
    """Visualize Central Limit Theorem"""
    sample_sizes = [1, 5, 30, 100]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, n in enumerate(sample_sizes):
        # Generate sample means
        sample_means = []
        for _ in range(n_samples):
            sample = distribution.rvs(**params, size=n)
            sample_means.append(sample.mean())

        # Original distribution (first row)
        if i == 0:
            x = np.linspace(distribution.ppf(0.001, **params),
                           distribution.ppf(0.999, **params), 100)
            axes[0, 0].plot(x, distribution.pdf(x, **params), 'b-', lw=2)
            axes[0, 0].set_title(f'Original distribution: {dist_name}')
            axes[0, 0].set_ylabel('Density')

        axes[0, i].hist(distribution.rvs(**params, size=1000), bins=30,
                        density=True, alpha=0.7, color='steelblue')
        axes[0, i].set_title(f'n=1 (Original)')

        # Sample mean distribution (second row)
        axes[1, i].hist(sample_means, bins=30, density=True, alpha=0.7, color='coral')

        # Theoretical normal distribution (by CLT)
        mu = distribution.mean(**params)
        sigma = distribution.std(**params)
        x = np.linspace(mu - 4*sigma/np.sqrt(n), mu + 4*sigma/np.sqrt(n), 100)
        axes[1, i].plot(x, stats.norm.pdf(x, mu, sigma/np.sqrt(n)),
                        'k--', lw=2, label='Theoretical normal')

        axes[1, i].set_title(f'Sample mean distribution for n={n}')
        axes[1, i].set_xlabel('Sample mean')
        if i == 0:
            axes[1, i].set_ylabel('Density')
        axes[1, i].legend()

    plt.suptitle(f'Central Limit Theorem: {dist_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

# Verify CLT with exponential distribution (asymmetric)
demonstrate_clt(stats.expon, {'scale': 2}, 'Exponential distribution (λ=0.5)')
```

```python
# CLT with uniform distribution
demonstrate_clt(stats.uniform, {'loc': 0, 'scale': 1}, 'Uniform distribution U(0,1)')
```

### 6.3 Mathematical Form of CLT

```python
# Standardized form
# Z = (X̄ - μ) / (σ/√n) ~ N(0, 1)

np.random.seed(42)
n = 30
mu, sigma = 50, 10  # Population parameters

# Generate 10000 sample means
sample_means = []
for _ in range(10000):
    sample = np.random.normal(mu, sigma, n)
    sample_means.append(sample.mean())

sample_means = np.array(sample_means)

# Standardization
Z = (sample_means - mu) / (sigma / np.sqrt(n))

# Compare with standard normal distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sample mean distribution
axes[0].hist(sample_means, bins=50, density=True, alpha=0.7, color='steelblue')
x = np.linspace(mu - 4*sigma/np.sqrt(n), mu + 4*sigma/np.sqrt(n), 100)
axes[0].plot(x, stats.norm.pdf(x, mu, sigma/np.sqrt(n)), 'r-', lw=2)
axes[0].set_xlabel('Sample mean')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Sample mean distribution (n={n})')
axes[0].axvline(mu, color='k', linestyle='--', label=f'μ={mu}')
axes[0].legend()

# Standardized distribution
axes[1].hist(Z, bins=50, density=True, alpha=0.7, color='coral')
x_std = np.linspace(-4, 4, 100)
axes[1].plot(x_std, stats.norm.pdf(x_std, 0, 1), 'r-', lw=2, label='N(0,1)')
axes[1].set_xlabel('Z = (X̄ - μ) / (σ/√n)')
axes[1].set_ylabel('Density')
axes[1].set_title('Standardized sample mean distribution')
axes[1].legend()

plt.tight_layout()
plt.show()

# Normality test
stat, p_value = stats.shapiro(Z[:5000])  # Shapiro-Wilk max 5000
print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4f}")
```

---

## 7. Comprehensive scipy.stats Distribution Examples

### 7.1 Using Distribution Objects

```python
# Common methods for scipy.stats distribution objects

# Normal distribution example
X = stats.norm(loc=0, scale=1)

print("scipy.stats distribution object methods:")
print(f"pdf(0): {X.pdf(0):.4f}  # Probability density function")
print(f"cdf(1.96): {X.cdf(1.96):.4f}  # Cumulative distribution function")
print(f"ppf(0.975): {X.ppf(0.975):.4f}  # Quantile function (inverse of CDF)")
print(f"sf(1.96): {X.sf(1.96):.4f}  # Survival function = 1 - CDF")
print(f"mean(): {X.mean():.4f}  # Mean")
print(f"var(): {X.var():.4f}  # Variance")
print(f"std(): {X.std():.4f}  # Standard deviation")
print(f"rvs(5): {X.rvs(5, random_state=42)}  # Random number generation")

# Interval probability
print(f"\nP(-1 < X < 1): {X.cdf(1) - X.cdf(-1):.4f}")
print(f"interval(0.95): {X.interval(0.95)}  # Central 95% interval")
```

### 7.2 Distribution Fitting

```python
# Fitting distribution to data
np.random.seed(42)

# Real data (generated from exponential distribution)
true_lambda = 0.5
data = stats.expon(scale=1/true_lambda).rvs(size=500)

# Fit exponential distribution
loc_fit, scale_fit = stats.expon.fit(data)
lambda_fit = 1 / scale_fit

print(f"True λ: {true_lambda}")
print(f"Estimated λ: {lambda_fit:.4f}")

# Visualize fitting results
plt.figure(figsize=(10, 5))
plt.hist(data, bins=30, density=True, alpha=0.7, label='Data')

x = np.linspace(0, data.max(), 100)
plt.plot(x, stats.expon(loc=loc_fit, scale=scale_fit).pdf(x),
         'r-', lw=2, label=f'Fitted exponential (λ={lambda_fit:.3f})')
plt.plot(x, stats.expon(scale=1/true_lambda).pdf(x),
         'g--', lw=2, label=f'True distribution (λ={true_lambda})')

plt.xlabel('x')
plt.ylabel('Density')
plt.title('Distribution Fitting Example')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 7.3 Summary Table of Major Distributions

```python
# Summary of major distributions
distributions = [
    ('Normal', 'norm', {'loc': 0, 'scale': 1}),
    ('Exponential', 'expon', {'scale': 2}),
    ('Gamma', 'gamma', {'a': 2, 'scale': 2}),
    ('Beta', 'beta', {'a': 2, 'b': 5}),
    ('Chi-square', 'chi2', {'df': 5}),
    ('t-distribution', 't', {'df': 10}),
    ('F-distribution', 'f', {'dfn': 5, 'dfd': 20}),
]

print("Summary of major continuous distributions:")
print("-" * 70)
print(f"{'Distribution':<15} {'scipy.stats':<12} {'Mean':<12} {'Variance':<12}")
print("-" * 70)

for name, dist_name, params in distributions:
    dist = getattr(stats, dist_name)(**params)
    print(f"{name:<15} {dist_name:<12} {dist.mean():<12.4f} {dist.var():<12.4f}")

print("\nMajor discrete distributions:")
print("-" * 70)
discrete_distributions = [
    ('Binomial', 'binom', {'n': 10, 'p': 0.3}),
    ('Poisson', 'poisson', {'mu': 5}),
    ('Geometric', 'geom', {'p': 0.3}),
    ('Negative binomial', 'nbinom', {'n': 5, 'p': 0.3}),
]

for name, dist_name, params in discrete_distributions:
    dist = getattr(stats, dist_name)(**params)
    print(f"{name:<15} {dist_name:<12} {dist.mean():<12.4f} {dist.var():<12.4f}")
```

---

## Practice Problems

### Problem 1: Probability Calculation
The lifespan of parts produced in a factory follows a normal distribution with mean 1000 hours and standard deviation 100 hours.
- (a) What is the probability that a part lasts at least 900 hours?
- (b) What is the minimum time for the top 5% lifespan?

### Problem 2: Central Limit Theorem
When drawing a sample of size n=50 from Poisson(λ=4), use CLT to find the probability that the sample mean is between 3.5 and 4.5.

### Problem 3: Joint Distribution
Given the joint probability mass function of X and Y:
- P(X=0, Y=0) = 0.1, P(X=0, Y=1) = 0.2
- P(X=1, Y=0) = 0.3, P(X=1, Y=1) = 0.4

(a) Find the marginal distributions.
(b) Are X and Y independent?
(c) Find Cov(X, Y).

---

## Summary

| Concept | Key Content | Python Function |
|---------|-------------|-----------------|
| Probability axioms | Non-negativity, normalization, countable additivity | - |
| PMF/PDF | Discrete/continuous probability distribution functions | `dist.pmf()`, `dist.pdf()` |
| CDF | Cumulative distribution function P(X ≤ x) | `dist.cdf()` |
| Expected value | E[X] = Σ x·P(x) | `dist.mean()`, `np.mean()` |
| Variance | Var(X) = E[(X-μ)²] | `dist.var()`, `np.var()` |
| Covariance | Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)] | `np.cov()` |
| Correlation | ρ = Cov(X,Y) / (σ_X·σ_Y) | `np.corrcoef()` |
| LLN | X̄ → μ as n → ∞ | Simulation |
| CLT | X̄ ~ N(μ, σ²/n) for large n | Simulation |

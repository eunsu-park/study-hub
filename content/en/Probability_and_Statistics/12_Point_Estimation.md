# Point Estimation

**Previous**: [Law of Large Numbers and CLT](./11_Law_of_Large_Numbers_and_CLT.md) | **Next**: [Interval Estimation](./13_Interval_Estimation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Formalise the estimation problem: distinguish parameters, estimators, and estimates
2. Define and evaluate the properties of estimators: unbiasedness, consistency, and efficiency
3. Derive estimators using the Method of Moments
4. Construct Maximum Likelihood Estimators via the likelihood and log-likelihood functions
5. State the asymptotic properties of MLEs: consistency, normality, and invariance
6. Compute Fisher information and apply the Cramer-Rao lower bound
7. Apply the Neyman factorization theorem to identify sufficient statistics
8. Explain completeness, the Rao-Blackwell theorem, and UMVUE

---

Point estimation is the problem of using observed data to produce a single "best guess" for an unknown population parameter. This lesson develops the theoretical framework for evaluating and constructing estimators, culminating in the powerful Maximum Likelihood method and the optimality theory that identifies the best possible estimator.

---

## 1. The Estimation Problem

### 1.1 Setup

- **Population model**: Data $X_1, X_2, \ldots, X_n$ are i.i.d. from a distribution $f(x; \theta)$, where $\theta \in \Theta$ is an unknown **parameter** (scalar or vector).
- **Estimator**: A function $\hat{\theta} = T(X_1, \ldots, X_n)$ of the data (not depending on $\theta$).
- **Estimate**: The numerical value of $\hat{\theta}$ computed from a specific sample.

### 1.2 Example

If $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with both parameters unknown, then $\theta = (\mu, \sigma^2)$. The sample mean $\bar{X}$ is an estimator of $\mu$; the sample variance $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ is an estimator of $\sigma^2$.

### 1.3 Sampling Distribution

Since $\hat{\theta}$ is a function of random data, it is itself a random variable with a **sampling distribution**. The properties of this distribution determine how good the estimator is.

---

## 2. Properties of Estimators

### 2.1 Bias and Unbiasedness

The **bias** of an estimator $\hat{\theta}$ for $\theta$ is:

$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

An estimator is **unbiased** if $E[\hat{\theta}] = \theta$ for all $\theta \in \Theta$.

**Examples**:

- $\bar{X}$ is unbiased for $\mu$: $E[\bar{X}] = \mu$.
- $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ is unbiased for $\sigma^2$. The $n-1$ denominator (Bessel's correction) removes the bias present in $\frac{1}{n}\sum(X_i - \bar{X})^2$.

### 2.2 Mean Squared Error

The **mean squared error** (MSE) combines bias and variance:

$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$$

A biased estimator can have lower MSE than an unbiased one if its variance is sufficiently reduced. This is the **bias-variance tradeoff**.

### 2.3 Consistency

An estimator $\hat{\theta}_n$ is **consistent** if $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$.

A sufficient condition: if $\text{Bias}(\hat{\theta}_n) \to 0$ and $\text{Var}(\hat{\theta}_n) \to 0$, then $\hat{\theta}_n$ is consistent (since $\text{MSE} \to 0$ implies convergence in probability).

### 2.4 Efficiency

Among all unbiased estimators, one is **efficient** if it achieves the smallest possible variance. The Cramer-Rao lower bound (Section 7) provides this minimum variance. The **relative efficiency** of two estimators is:

$$e(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{Var}(\hat{\theta}_2)}{\text{Var}(\hat{\theta}_1)}$$

If $e > 1$, then $\hat{\theta}_1$ is more efficient.

---

## 3. Method of Moments (MoM)

### 3.1 Procedure

1. Compute the first $k$ **population moments**: $\mu_j' = E[X^j]$ for $j = 1, \ldots, k$, expressed as functions of the parameters $\theta_1, \ldots, \theta_k$.
2. Set them equal to the corresponding **sample moments**: $m_j' = \frac{1}{n}\sum_{i=1}^n X_i^j$.
3. Solve the system of equations $m_j' = \mu_j'(\theta_1, \ldots, \theta_k)$ for $\hat{\theta}_1, \ldots, \hat{\theta}_k$.

### 3.2 Example: Normal Distribution

For $X \sim N(\mu, \sigma^2)$:

- First moment: $E[X] = \mu$, so $\hat{\mu}_{MoM} = \bar{X}$.
- Second moment: $E[X^2] = \mu^2 + \sigma^2$, so $\hat{\sigma}^2_{MoM} = \frac{1}{n}\sum X_i^2 - \bar{X}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$.

Note: $\hat{\sigma}^2_{MoM}$ uses divisor $n$, making it slightly biased.

### 3.3 Example: Gamma Distribution

For $X \sim \text{Gamma}(\alpha, \beta)$ (rate parameterisation): $E[X] = \alpha/\beta$ and $E[X^2] = \alpha(\alpha+1)/\beta^2$.

Solving:

$$\hat{\beta}_{MoM} = \frac{\bar{X}}{m_2' - \bar{X}^2}, \qquad \hat{\alpha}_{MoM} = \bar{X}\, \hat{\beta}_{MoM}$$

### 3.4 Properties

- MoM estimators are generally **consistent** (by the LLN, sample moments converge to population moments).
- They are typically **not efficient** (they may have larger variance than MLE).
- They are easy to compute and provide good starting values for iterative MLE algorithms.

---

## 4. Maximum Likelihood Estimation (MLE)

### 4.1 The Likelihood Function

Given observed data $x_1, \ldots, x_n$, the **likelihood function** is:

$$L(\theta) = L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta)$$

It is the joint density evaluated at the observed data, viewed as a function of $\theta$.

### 4.2 The Log-Likelihood

Since products are difficult to maximise, we work with the **log-likelihood**:

$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)$$

Maximising $\ell(\theta)$ is equivalent to maximising $L(\theta)$.

### 4.3 The Score Function

The **score function** is the gradient of the log-likelihood:

$$S(\theta) = \frac{\partial \ell(\theta)}{\partial \theta}$$

The MLE $\hat{\theta}_{MLE}$ satisfies the **score equation** (likelihood equation):

$$S(\hat{\theta}) = \sum_{i=1}^n \frac{\partial}{\partial\theta} \ln f(x_i; \hat{\theta}) = 0$$

### 4.4 Example: Normal Distribution

For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2$$

Setting partial derivatives to zero:

$$\hat{\mu}_{MLE} = \bar{X}, \qquad \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2$$

Note: $\hat{\sigma}^2_{MLE}$ is biased (divides by $n$ instead of $n-1$), but it is consistent.

### 4.5 Example: Poisson Distribution

For $X_1, \ldots, X_n \sim \text{Poisson}(\lambda)$:

$$\ell(\lambda) = \sum_{i=1}^n \left[x_i \ln\lambda - \lambda - \ln(x_i!)\right]$$

$$\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda}_{MLE} = \bar{X}$$

---

## 5. Properties of MLE

### 5.1 Consistency

Under regularity conditions, $\hat{\theta}_{MLE} \xrightarrow{P} \theta_0$ (the true parameter value).

### 5.2 Asymptotic Normality

$$\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} N\!\left(0, \frac{1}{I(\theta_0)}\right)$$

where $I(\theta)$ is the Fisher information (see Section 6). The MLE is **asymptotically efficient**: it achieves the Cramer-Rao lower bound in the limit.

### 5.3 Invariance Property

If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$. This is called the **invariance principle**.

For example, if $\hat{\sigma}^2$ is the MLE of $\sigma^2$, then $\hat{\sigma} = \sqrt{\hat{\sigma}^2}$ is the MLE of $\sigma$.

### 5.4 Asymptotic Efficiency

Among all consistent and asymptotically normal estimators, the MLE has the smallest asymptotic variance. No regular estimator can do better in the large-sample limit.

---

## 6. Fisher Information

### 6.1 Definition

The **Fisher information** about $\theta$ contained in a single observation is:

$$I(\theta) = E\!\left[\left(\frac{\partial}{\partial\theta} \ln f(X; \theta)\right)^2\right] = E[S(\theta)^2]$$

Under regularity conditions, this equals:

$$I(\theta) = -E\!\left[\frac{\partial^2}{\partial\theta^2} \ln f(X; \theta)\right]$$

### 6.2 Interpretation

- Fisher information measures the **curvature** of the log-likelihood at the true parameter value.
- High curvature means the log-likelihood is sharply peaked, making it easier to pinpoint $\theta$.
- For $n$ i.i.d. observations, the total Fisher information is $I_n(\theta) = n \cdot I(\theta)$.

### 6.3 Example: Bernoulli

For $X \sim \text{Bernoulli}(p)$: $\ln f(x; p) = x\ln p + (1-x)\ln(1-p)$.

$$\frac{\partial^2}{\partial p^2} \ln f = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}$$

$$I(p) = E\!\left[\frac{X}{p^2} + \frac{1-X}{(1-p)^2}\right] = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}$$

Information is highest when $p = 0.5$ (most uncertainty) and lowest near $p = 0$ or $p = 1$.

### 6.4 Multivariate Fisher Information

For a parameter vector $\boldsymbol{\theta} \in \mathbb{R}^k$, the Fisher information is a $k \times k$ **matrix**:

$$[\mathbf{I}(\boldsymbol{\theta})]_{jk} = -E\!\left[\frac{\partial^2 \ell}{\partial\theta_j \partial\theta_k}\right]$$

---

## 7. Cramer-Rao Lower Bound

### 7.1 Statement

For any **unbiased** estimator $\hat{\theta}$ of $\theta$:

$$\text{Var}(\hat{\theta}) \ge \frac{1}{I_n(\theta)} = \frac{1}{n \cdot I(\theta)}$$

### 7.2 Interpretation

- The CRLB gives the **smallest possible variance** for any unbiased estimator.
- If $\text{Var}(\hat{\theta}) = 1/(nI(\theta))$, the estimator is said to be **efficient** or to achieve the CRLB.
- The MLE achieves the CRLB asymptotically (for large $n$), and sometimes exactly for finite $n$.

### 7.3 Example: Bernoulli

For $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$, the MLE is $\hat{p} = \bar{X}$. Its variance is $p(1-p)/n$.

The CRLB is $1/(nI(p)) = p(1-p)/n$.

Since $\text{Var}(\hat{p}) = \text{CRLB}$, the sample proportion is efficient for estimating $p$.

### 7.4 Extension for Biased Estimators

If $E[\hat{\theta}] = g(\theta)$ (possibly biased), the bound generalises to:

$$\text{Var}(\hat{\theta}) \ge \frac{[g'(\theta)]^2}{nI(\theta)}$$

---

## 8. Sufficient Statistics

### 8.1 Definition

A statistic $T = T(X_1, \ldots, X_n)$ is **sufficient** for $\theta$ if the conditional distribution of the data given $T$ does not depend on $\theta$:

$$f(x_1, \ldots, x_n \mid T = t; \theta) \text{ is free of } \theta$$

Intuitively, $T$ captures **all the information** in the data about $\theta$. Once $T$ is known, the remaining variation in the data is pure noise.

### 8.2 Neyman Factorization Theorem

$T(X_1, \ldots, X_n)$ is sufficient for $\theta$ if and only if the joint density can be factored as:

$$f(x_1, \ldots, x_n; \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})$$

where $g$ depends on the data only through $T$, and $h$ does not depend on $\theta$.

### 8.3 Examples

**Normal** ($\mu$ unknown, $\sigma^2$ known): $T = \sum X_i$ (or equivalently $\bar{X}$) is sufficient.

$$\prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma}e^{-(x_i-\mu)^2/(2\sigma^2)} = \underbrace{\exp\!\left(\frac{\mu \sum x_i}{\sigma^2} - \frac{n\mu^2}{2\sigma^2}\right)}_{g(T, \mu)} \cdot \underbrace{\frac{1}{(2\pi\sigma^2)^{n/2}} \exp\!\left(-\frac{\sum x_i^2}{2\sigma^2}\right)}_{h(\mathbf{x})}$$

**Poisson** ($\lambda$ unknown): $T = \sum X_i$ is sufficient.

**Normal** (both $\mu$ and $\sigma^2$ unknown): $T = (\sum X_i, \sum X_i^2)$ is jointly sufficient.

---

## 9. Completeness, UMVUE, and Rao-Blackwell

### 9.1 Completeness

A sufficient statistic $T$ is **complete** if for every function $g$:

$$E_\theta[g(T)] = 0 \text{ for all } \theta \implies P(g(T) = 0) = 1 \text{ for all } \theta$$

In other words, the only unbiased estimator of zero based on $T$ is the zero function. Completeness rules out "redundant" sufficient statistics.

**Exponential family** distributions have complete sufficient statistics (under mild conditions).

### 9.2 Rao-Blackwell Theorem

If $\hat{\theta}$ is an unbiased estimator of $\theta$ and $T$ is a sufficient statistic, define:

$$\hat{\theta}^* = E[\hat{\theta} \mid T]$$

Then:

1. $\hat{\theta}^*$ is unbiased: $E[\hat{\theta}^*] = \theta$.
2. $\text{Var}(\hat{\theta}^*) \le \text{Var}(\hat{\theta})$, with equality only if $\hat{\theta}$ is already a function of $T$.

The Rao-Blackwell theorem says: **condition any unbiased estimator on a sufficient statistic to get a (weakly) better estimator**.

### 9.3 UMVUE (Uniformly Minimum Variance Unbiased Estimator)

An unbiased estimator is **UMVUE** if it has the smallest variance among all unbiased estimators, for every value of $\theta$.

**Lehmann-Scheffe Theorem**: If $T$ is a complete sufficient statistic and $\hat{\theta}^* = g(T)$ is unbiased for $\theta$, then $\hat{\theta}^*$ is the UMVUE.

### 9.4 Recipe for Finding UMVUE

1. Find a sufficient statistic $T$ (via Neyman factorization).
2. Check completeness (often guaranteed by exponential family).
3. Find a function $g(T)$ that is unbiased for $\theta$.
4. By Lehmann-Scheffe, $g(T)$ is the UMVUE.

### 9.5 Example: UMVUE for Normal Mean

For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with $\sigma^2$ known:

- Sufficient statistic: $T = \bar{X}$
- Complete: Yes (exponential family)
- Unbiased: $E[\bar{X}] = \mu$
- Conclusion: $\bar{X}$ is the UMVUE for $\mu$

---

## 10. Python Examples

### 10.1 Method of Moments for Gamma Distribution

```python
import random
import math

random.seed(42)

# True parameters
alpha_true = 3.0
beta_true = 2.0  # rate parameterisation
# Gamma(alpha, beta): mean = alpha/beta, var = alpha/beta^2

# Generate Gamma samples as sum of Exponentials (alpha must be integer for this)
n = 500
samples = []
for _ in range(n):
    # Gamma(3, 2) = sum of 3 independent Exp(2)
    val = sum(random.expovariate(beta_true) for _ in range(int(alpha_true)))
    samples.append(val)

# Method of Moments
m1 = sum(samples) / n                              # first sample moment
m2 = sum(x ** 2 for x in samples) / n              # second sample moment
sample_var = m2 - m1 ** 2                           # = E[X^2] - (E[X])^2

alpha_mom = m1 ** 2 / sample_var
beta_mom = m1 / sample_var

print("Method of Moments for Gamma(alpha, beta):")
print(f"  alpha_hat = {alpha_mom:.4f}  (true: {alpha_true})")
print(f"  beta_hat  = {beta_mom:.4f}  (true: {beta_true})")
print(f"  sample mean     = {m1:.4f}  (theoretical: {alpha_true/beta_true:.4f})")
print(f"  sample variance = {sample_var:.4f}  "
      f"(theoretical: {alpha_true/beta_true**2:.4f})")
```

### 10.2 MLE for Normal Distribution

```python
import random
import math

random.seed(123)

mu_true = 5.0
sigma_true = 2.0
n = 200
samples = [random.gauss(mu_true, sigma_true) for _ in range(n)]

# MLE
mu_mle = sum(samples) / n
sigma2_mle = sum((x - mu_mle) ** 2 for x in samples) / n       # biased
sigma2_unbiased = sum((x - mu_mle) ** 2 for x in samples) / (n - 1)  # unbiased

print("MLE for Normal(mu, sigma^2):")
print(f"  mu_MLE     = {mu_mle:.4f}  (true: {mu_true})")
print(f"  sigma2_MLE = {sigma2_mle:.4f}  (true: {sigma_true**2})")
print(f"  sigma2_S2  = {sigma2_unbiased:.4f}  (unbiased, true: {sigma_true**2})")
print(f"  sigma_MLE  = {math.sqrt(sigma2_mle):.4f}  "
      f"(true: {sigma_true}, by invariance)")
```

### 10.3 MLE for Poisson Distribution

```python
import random
import math

random.seed(77)

lambda_true = 4.5
n = 300

# Generate Poisson samples via inverse transform
def poisson_sample(lam):
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= random.random()
        if p < L:
            return k
        k += 1

samples = [poisson_sample(lambda_true) for _ in range(n)]

# MLE for Poisson is the sample mean
lambda_mle = sum(samples) / n

# Fisher information: I(lambda) = 1/lambda
# CRLB: Var(lambda_hat) >= lambda / n
fisher_info = 1.0 / lambda_true
crlb = lambda_true / n

# Simulate variance of MLE
n_sim = 10_000
mle_values = []
for _ in range(n_sim):
    sim_samples = [poisson_sample(lambda_true) for _ in range(n)]
    mle_values.append(sum(sim_samples) / n)

var_mle = sum((v - lambda_true) ** 2 for v in mle_values) / (n_sim - 1)

print(f"\nMLE for Poisson(lambda = {lambda_true}):")
print(f"  lambda_MLE = {lambda_mle:.4f}")
print(f"  Fisher info I(lambda) = {fisher_info:.4f}")
print(f"  CRLB = {crlb:.6f}")
print(f"  Simulated Var(MLE) = {var_mle:.6f}")
print(f"  MLE achieves CRLB: {abs(var_mle - crlb) < 0.005}")
```

### 10.4 Visualising Log-Likelihood for Normal Mean

```python
import random
import math

random.seed(2024)

mu_true = 3.0
sigma = 1.5
n = 50
samples = [random.gauss(mu_true, sigma) for _ in range(n)]

def log_likelihood_normal_mean(mu, data, sigma):
    """Log-likelihood for Normal mean with known sigma."""
    n = len(data)
    ss = sum((x - mu) ** 2 for x in data)
    return -n / 2 * math.log(2 * math.pi * sigma ** 2) - ss / (2 * sigma ** 2)

# Evaluate over a grid
mu_grid = [2.0 + 0.05 * i for i in range(41)]  # 2.0 to 4.0
ll_values = [log_likelihood_normal_mean(mu, samples, sigma) for mu in mu_grid]

# Find MLE
mu_mle = sum(samples) / n
ll_max = log_likelihood_normal_mean(mu_mle, samples, sigma)

print(f"\nLog-likelihood for Normal mean (sigma = {sigma} known):")
print(f"  MLE: mu_hat = {mu_mle:.4f}")
print(f"  Max log-likelihood: {ll_max:.2f}")

# ASCII plot of log-likelihood
ll_min = min(ll_values)
ll_range = ll_max - ll_min
print("\n  mu    | log-likelihood")
print("  " + "-" * 55)
for mu, ll in zip(mu_grid, ll_values):
    bar_len = int((ll - ll_min) / ll_range * 40) if ll_range > 0 else 0
    marker = " <-- MLE" if abs(mu - mu_mle) < 0.03 else ""
    print(f"  {mu:5.2f} | {'#' * bar_len}{marker}")
```

### 10.5 Cramer-Rao Bound: Bernoulli

```python
import random
import math

random.seed(555)

p_true = 0.3
n_obs = 100
n_sim = 50_000

# CRLB for Bernoulli: Var(p_hat) >= p(1-p)/n
crlb = p_true * (1 - p_true) / n_obs

# Simulate sampling distribution of p_hat = X_bar
p_hat_values = []
for _ in range(n_sim):
    successes = sum(1 for _ in range(n_obs) if random.random() < p_true)
    p_hat_values.append(successes / n_obs)

mean_phat = sum(p_hat_values) / n_sim
var_phat = sum((p - mean_phat) ** 2 for p in p_hat_values) / (n_sim - 1)

print(f"\nCramer-Rao Bound for Bernoulli(p = {p_true}), n = {n_obs}:")
print(f"  CRLB = {crlb:.6f}")
print(f"  Simulated Var(p_hat) = {var_phat:.6f}")
print(f"  Ratio Var/CRLB = {var_phat / crlb:.4f}  (should be ~1.0)")
print(f"  E[p_hat] = {mean_phat:.4f}  (unbiased: true p = {p_true})")
```

### 10.6 Rao-Blackwell Improvement

```python
import random

random.seed(888)

# Estimating P(X=0) for Poisson(lambda)
# True value: e^(-lambda)
# Naive unbiased estimator: T(X1) = 1 if X1=0, else 0
# Sufficient statistic: S = sum(Xi)
# Rao-Blackwell: E[T | S] = P(X1=0 | sum=s) = ((n-1)/n)^s * C(n-1,s-0)/C(n,s)
# For Poisson, E[1(X1=0) | S=s] = ((n-1)/n)^s

import math

lambda_true = 2.0
n = 20
n_sim = 50_000
true_value = math.exp(-lambda_true)

def poisson_sample(lam):
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p < L:
            return k
        k += 1

naive_estimates = []
rb_estimates = []

for _ in range(n_sim):
    data = [poisson_sample(lambda_true) for _ in range(n)]

    # Naive: use only first observation
    naive = 1.0 if data[0] == 0 else 0.0
    naive_estimates.append(naive)

    # Rao-Blackwell: E[1(X1=0) | S=s] = ((n-1)/n)^s
    s = sum(data)
    rb = ((n - 1) / n) ** s
    rb_estimates.append(rb)

mean_naive = sum(naive_estimates) / n_sim
var_naive = sum((x - mean_naive) ** 2 for x in naive_estimates) / (n_sim - 1)
mean_rb = sum(rb_estimates) / n_sim
var_rb = sum((x - mean_rb) ** 2 for x in rb_estimates) / (n_sim - 1)

print(f"\nRao-Blackwell improvement for estimating e^(-lambda):")
print(f"  True value: {true_value:.6f}")
print(f"  Naive estimator (uses X1 only):")
print(f"    Mean = {mean_naive:.6f},  Var = {var_naive:.6f}")
print(f"  Rao-Blackwell estimator (conditions on sum):")
print(f"    Mean = {mean_rb:.6f},  Var = {var_rb:.6f}")
print(f"  Variance reduction: {(1 - var_rb / var_naive) * 100:.1f}%")
```

---

## Key Takeaways

1. **Unbiasedness, consistency, and efficiency** are the three pillars of estimator evaluation. Consistency is often considered the most important: a good estimator should at least converge to the truth.
2. The **Method of Moments** equates sample and population moments. It is simple and consistent but generally not the most efficient.
3. **Maximum Likelihood Estimation** maximises the probability of the observed data. MLEs are consistent, asymptotically normal, asymptotically efficient, and invariant under reparameterisation.
4. **Fisher information** measures how much a single observation tells us about $\theta$. It determines the precision limit via the Cramer-Rao lower bound.
5. The **Cramer-Rao Lower Bound** gives a floor on the variance of any unbiased estimator: $\text{Var}(\hat{\theta}) \ge 1/(nI(\theta))$.
6. **Sufficient statistics** compress the data without losing information about $\theta$. The Neyman factorization theorem provides a practical test.
7. The **Rao-Blackwell theorem** shows that conditioning any unbiased estimator on a sufficient statistic cannot increase its variance. Combined with completeness (Lehmann-Scheffe), this identifies the UMVUE.

---

*Next lesson: [Interval Estimation](./13_Interval_Estimation.md)*

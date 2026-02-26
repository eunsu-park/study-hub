# 11. Advanced Probability Distributions

## Learning Objectives

- Understand the general form and properties of the exponential family and represent various distributions in exponential family form
- Grasp the concepts of sufficient statistics and natural parameters and understand their connection to GLMs
- Understand the principles of conjugate priors and learn their application in Bayesian inference
- Master the properties of multivariate Gaussian distributions and perform conditional/marginalization operations
- Understand the structure of Gaussian Mixture Models (GMM) and apply the EM algorithm
- Learn how advanced probability distributions are utilized in machine learning through practical examples

---

## 1. Exponential Family

### 1.1 General Form

A probability distribution belongs to the **exponential family** if it can be expressed in the following form:

$$p(x|\eta) = h(x) \exp\left(\eta^T T(x) - A(\eta)\right)$$

where:
- $\eta$: **Natural parameter** or canonical parameter
- $T(x)$: **Sufficient statistic**
- $A(\eta)$: **Log partition function** or cumulant generating function
- $h(x)$: Base measure

**Normalization condition**:

$$\int h(x) \exp\left(\eta^T T(x) - A(\eta)\right) dx = 1$$

Therefore:

$$A(\eta) = \log \int h(x) \exp\left(\eta^T T(x)\right) dx$$

### 1.2 Why Exponential Family?

The exponential family has excellent properties:

1. **Sufficient statistics**: Data can be summarized by $T(x)$
2. **Analytical MLE**: Closed-form solution
3. **Conjugate priors exist**: Bayesian inference is easy
4. **Moment computation**: Obtained from derivatives of $A(\eta)$
5. **Foundation of GLM**: Generalized Linear Models

**Properties of log partition function**:

$$\frac{\partial A}{\partial \eta} = \mathbb{E}[T(X)]$$

$$\frac{\partial^2 A}{\partial \eta^2} = \text{Var}(T(X))$$

### 1.3 Sufficient Statistic

**Definition**: $T(x)$ is a **sufficient statistic** for $\eta$.

**Meaning**: $T(x)$ contains all information needed for parameter estimation.

$$P(\theta | x) = P(\theta | T(x))$$

**Example**: For $n$ normal distribution samples, $\bar{x} = \frac{1}{n}\sum x_i$ and $s^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$ are sufficient statistics.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Log partition function of exponential family distributions

# Bernoulli: η = log(p/(1-p)), A(η) = log(1 + e^η)
eta_bernoulli = np.linspace(-5, 5, 100)
A_bernoulli = np.log(1 + np.exp(eta_bernoulli))
p_bernoulli = 1 / (1 + np.exp(-eta_bernoulli))  # p = sigmoid(η)

# Gaussian (mean only): η = μ, A(η) = η²/2 (variance=1 fixed)
eta_gaussian = np.linspace(-3, 3, 100)
A_gaussian = eta_gaussian**2 / 2

plt.figure(figsize=(14, 4))

# Bernoulli
plt.subplot(131)
plt.plot(eta_bernoulli, A_bernoulli, linewidth=2)
plt.xlabel('η (natural parameter)')
plt.ylabel('A(η)')
plt.title('Bernoulli: A(η) = log(1 + exp(η))')
plt.grid(True)

plt.subplot(132)
plt.plot(eta_bernoulli, p_bernoulli, linewidth=2, color='green')
plt.xlabel('η')
plt.ylabel('p = dA/dη')
plt.title('Bernoulli: E[X] = sigmoid(η)')
plt.grid(True)

# Gaussian
plt.subplot(133)
plt.plot(eta_gaussian, A_gaussian, linewidth=2, color='red')
plt.xlabel('η (= μ)')
plt.ylabel('A(η)')
plt.title('Gaussian (σ=1): A(η) = η²/2')
plt.grid(True)

plt.tight_layout()
plt.savefig('exponential_family_log_partition.png', dpi=150, bbox_inches='tight')
plt.show()

print("Derivative of A(η) = Expectation E[T(X)]")
print("Second derivative of A(η) = Variance Var(T(X))")
```

## 2. Examples of Exponential Family

### 2.1 Bernoulli Distribution → Exponential Family

**General form**:
$$p(x|p) = p^x (1-p)^{1-x}$$

**Exponential family transformation**:

$$= \exp\left[x \log p + (1-x) \log(1-p)\right]$$

$$= \exp\left[x \log \frac{p}{1-p} + \log(1-p)\right]$$

**Identification**:
- $\eta = \log \frac{p}{1-p}$ (logit)
- $T(x) = x$
- $A(\eta) = -\log(1-p) = \log(1 + e^\eta)$
- $h(x) = 1$

**Inverse transformation**: $p = \sigma(\eta) = \frac{1}{1+e^{-\eta}}$ (sigmoid!)

### 2.2 Gaussian Distribution → Exponential Family

**General form** (with known $\sigma^2$):

$$p(x|\mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right) \exp\left(\frac{\mu x}{\sigma^2} - \frac{\mu^2}{2\sigma^2}\right)$$

**Identification**:
- $\eta = \frac{\mu}{\sigma^2}$
- $T(x) = x$
- $A(\eta) = \frac{\mu^2}{2\sigma^2} = \frac{\sigma^2 \eta^2}{2}$
- $h(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)$

**When both $\mu$ and $\sigma^2$ are unknown**:

$$\eta = \begin{bmatrix} \mu/\sigma^2 \\ -1/(2\sigma^2) \end{bmatrix}, \quad T(x) = \begin{bmatrix} x \\ x^2 \end{bmatrix}$$

### 2.3 Poisson Distribution

$$p(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$$

$$= \frac{1}{x!} \exp(x \log \lambda - \lambda)$$

- $\eta = \log \lambda$
- $T(x) = x$
- $A(\eta) = \lambda = e^\eta$
- $h(x) = 1/x!$

### 2.4 Gamma Distribution

$$p(x|\alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$$

(with $\alpha$ fixed, $\beta$ variable):
- $\eta = -\beta$
- $T(x) = x$
- $A(\eta) = -\alpha \log(-\eta)$

### 2.5 Beta Distribution

$$p(x|\alpha, \beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}$$

- $\eta = \begin{bmatrix} \alpha-1 \\ \beta-1 \end{bmatrix}$
- $T(x) = \begin{bmatrix} \log x \\ \log(1-x) \end{bmatrix}$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Various exponential family distributions
x_range = np.linspace(0, 10, 1000)

# Poisson distributions (varying λ)
lambdas = [1, 2, 4]
x_poisson = np.arange(0, 15)

plt.figure(figsize=(14, 9))

# Poisson
plt.subplot(331)
for lam in lambdas:
    pmf = stats.poisson.pmf(x_poisson, lam)
    plt.plot(x_poisson, pmf, 'o-', label=f'λ={lam}')
plt.xlabel('x')
plt.ylabel('P(X=x)')
plt.title('Poisson Distribution')
plt.legend()
plt.grid(True)

# Exponential distribution (special case of Gamma, α=1)
plt.subplot(332)
for lam in lambdas:
    pdf = stats.expon.pdf(x_range, scale=1/lam)
    plt.plot(x_range, pdf, label=f'λ={lam}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Exponential Distribution')
plt.legend()
plt.grid(True)

# Gamma distribution
plt.subplot(333)
alphas = [1, 2, 5]
beta = 1
for alpha in alphas:
    pdf = stats.gamma.pdf(x_range, alpha, scale=1/beta)
    plt.plot(x_range, pdf, label=f'α={alpha}, β={beta}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Gamma Distribution')
plt.legend()
plt.grid(True)

# Beta distribution
x_beta = np.linspace(0, 1, 1000)
plt.subplot(334)
params = [(0.5, 0.5), (2, 2), (5, 2)]
for alpha, beta in params:
    pdf = stats.beta.pdf(x_beta, alpha, beta)
    plt.plot(x_beta, pdf, label=f'α={alpha}, β={beta}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Beta Distribution')
plt.legend()
plt.grid(True)

# Gaussian (various μ)
plt.subplot(335)
mus = [-1, 0, 1]
sigma = 1
x_gauss = np.linspace(-4, 4, 1000)
for mu in mus:
    pdf = stats.norm.pdf(x_gauss, mu, sigma)
    plt.plot(x_gauss, pdf, label=f'μ={mu}, σ={sigma}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Gaussian Distribution')
plt.legend()
plt.grid(True)

# Bernoulli (various p)
plt.subplot(336)
ps = [0.2, 0.5, 0.8]
x_bern = [0, 1]
for p in ps:
    pmf = [1-p, p]
    plt.plot(x_bern, pmf, 'o-', markersize=10, label=f'p={p}')
plt.xlabel('x')
plt.ylabel('P(X=x)')
plt.title('Bernoulli Distribution')
plt.xticks([0, 1])
plt.legend()
plt.grid(True)

# Categorical (multinomial)
plt.subplot(337)
categories = ['A', 'B', 'C', 'D']
probs1 = [0.25, 0.25, 0.25, 0.25]
probs2 = [0.5, 0.3, 0.15, 0.05]
x_pos = np.arange(len(categories))
width = 0.35
plt.bar(x_pos - width/2, probs1, width, label='Uniform', alpha=0.7)
plt.bar(x_pos + width/2, probs2, width, label='Skewed', alpha=0.7)
plt.xlabel('Category')
plt.ylabel('Probability')
plt.title('Categorical Distribution')
plt.xticks(x_pos, categories)
plt.legend()
plt.grid(True, axis='y')

# Multivariate normal (2D contour)
plt.subplot(338)
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))
rv = stats.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
plt.contourf(X1, X2, rv.pdf(pos), levels=15, cmap='viridis')
plt.colorbar()
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Bivariate Gaussian')
plt.grid(True)

# Dirichlet (simplified: project 3D to 2D)
plt.subplot(339)
from matplotlib.patches import Polygon
# Visualize 2-simplex (sum=1 constraint)
alpha_params = [(1, 1, 1), (2, 5, 2), (10, 5, 3)]
for alpha in alpha_params:
    # Simple representation: weights at each vertex
    label = f'α=({alpha[0]},{alpha[1]},{alpha[2]})'
    plt.text(0.5, sum(alpha)/30, label, ha='center')
plt.text(0.5, 0.5, 'Dirichlet\n(simplex)', ha='center', va='center',
         fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Dirichlet Distribution (conceptual)')
plt.axis('off')

plt.tight_layout()
plt.savefig('exponential_family_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print("All exponential family distributions share the same mathematical structure!")
```

## 3. Conjugate Priors

### 3.1 Definition

For prior $p(\theta)$ and likelihood $p(x|\theta)$, if posterior $p(\theta|x)$ belongs to the **same distribution family** as the prior, the prior is called a **conjugate prior**.

$$p(\theta), p(\theta|x) \in \text{same family}$$

**Advantages**:
1. Closed-form posterior
2. Sequential updates possible
3. Bayesian inference is analytical

### 3.2 Beta-Bernoulli Conjugacy

**Likelihood**: Bernoulli $p(x|\theta) = \theta^x (1-\theta)^{1-x}$

**Prior**: Beta $p(\theta) = \text{Beta}(\alpha, \beta)$

$$p(\theta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}$$

**Posterior** ($n$ trials, $k$ successes):

$$p(\theta|D) \propto p(D|\theta) p(\theta)$$

$$\propto \theta^k (1-\theta)^{n-k} \cdot \theta^{\alpha-1} (1-\theta)^{\beta-1}$$

$$= \theta^{k+\alpha-1} (1-\theta)^{n-k+\beta-1}$$

$$= \text{Beta}(\alpha + k, \beta + n - k)$$

**Interpretation**:
- $\alpha$: Prior "success count" (pseudo-observations)
- $\beta$: Prior "failure count"
- Posterior: Prior pseudo-observations + actual observations

### 3.3 Normal-Normal Conjugacy

**Likelihood**: $x \sim \mathcal{N}(\mu, \sigma^2)$ (with known $\sigma^2$)

**Prior**: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**Posterior**: $\mu | D \sim \mathcal{N}(\mu_n, \sigma_n^2)$

$$\mu_n = \frac{\sigma^2 \mu_0 + n \sigma_0^2 \bar{x}}{\sigma^2 + n \sigma_0^2}$$

$$\sigma_n^2 = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n \sigma_0^2}$$

**Interpretation**:
- $\mu_n$: Weighted average of prior mean and sample mean
- As data increases ($n \to \infty$) $\mu_n \to \bar{x}$

### 3.4 Dirichlet-Categorical Conjugacy

**Likelihood**: Categorical $p(x|\theta) = \prod_{k=1}^K \theta_k^{x_k}$ (one-hot $x$)

**Prior**: Dirichlet $p(\theta) = \text{Dir}(\alpha)$

$$p(\theta|\alpha) = \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \prod_{k=1}^K \theta_k^{\alpha_k - 1}$$

**Posterior** (category $k$ observed $n_k$ times):

$$p(\theta|D) = \text{Dir}(\alpha_1 + n_1, \ldots, \alpha_K + n_K)$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Beta-Bernoulli conjugacy demonstration
np.random.seed(42)

# Prior: Beta(2, 2) (slight preference for center)
alpha_prior, beta_prior = 2, 2

# True coin: p = 0.7
true_p = 0.7
n_flips = [0, 1, 5, 20, 100]

theta_range = np.linspace(0, 1, 1000)

plt.figure(figsize=(14, 8))

for i, n in enumerate(n_flips):
    # Generate data
    if n > 0:
        data = np.random.binomial(1, true_p, n)
        k = data.sum()  # Number of successes
    else:
        k = 0

    # Posterior distribution
    alpha_post = alpha_prior + k
    beta_post = beta_prior + (n - k)

    # Visualization
    plt.subplot(2, 3, i+1)

    # Prior (only in first plot)
    if i == 0:
        prior = stats.beta.pdf(theta_range, alpha_prior, beta_prior)
        plt.plot(theta_range, prior, 'b--', linewidth=2, label='Prior')

    # Posterior distribution
    posterior = stats.beta.pdf(theta_range, alpha_post, beta_post)
    plt.plot(theta_range, posterior, 'r-', linewidth=2, label='Posterior')

    # True value
    plt.axvline(true_p, color='g', linestyle='--', alpha=0.7,
                label=f'True p={true_p}')

    # Posterior mean
    post_mean = alpha_post / (alpha_post + beta_post)
    plt.axvline(post_mean, color='orange', linestyle='--', alpha=0.7,
                label=f'Post mean={post_mean:.3f}')

    plt.xlabel('θ')
    plt.ylabel('Density')
    plt.title(f'n={n}, k={k if n > 0 else 0}\nBeta({alpha_post}, {beta_post})')
    plt.legend(fontsize=8)
    plt.grid(True)

# 6th plot: sequential update
plt.subplot(2, 3, 6)
alphas = [alpha_prior]
betas = [beta_prior]

n_total = 50
for flip in range(n_total):
    result = np.random.binomial(1, true_p)
    alphas.append(alphas[-1] + result)
    betas.append(betas[-1] + (1 - result))

# Change in mean
means = [a / (a + b) for a, b in zip(alphas, betas)]
plt.plot(means, linewidth=2, label='Posterior mean')
plt.axhline(true_p, color='g', linestyle='--', label=f'True p={true_p}')
plt.xlabel('Number of flips')
plt.ylabel('Posterior mean of θ')
plt.title('Sequential Bayesian Update')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('conjugate_beta_bernoulli.png', dpi=150, bbox_inches='tight')
plt.show()

print("Advantages of conjugate priors:")
print("1. Posterior has a closed form (remains Beta)")
print("2. Sequential updates are possible")
print("3. Converges to the true value as data increases")
```

### 3.5 Main Conjugate Pairs

| Likelihood | Prior | Posterior |
|------|---------|---------|
| Bernoulli | Beta | Beta |
| Binomial | Beta | Beta |
| Categorical | Dirichlet | Dirichlet |
| Multinomial | Dirichlet | Dirichlet |
| Gaussian (μ, σ² known) | Gaussian | Gaussian |
| Gaussian (σ² known) | Gaussian (on μ) | Gaussian |
| Gaussian (μ known) | Inverse-Gamma (on σ²) | Inverse-Gamma |
| Poisson | Gamma | Gamma |
| Exponential | Gamma | Gamma |

## 4. Multivariate Gaussian Distribution

### 4.1 Definition

$d$-dimensional **multivariate Gaussian distribution** $\mathcal{N}(\mu, \Sigma)$:

$$p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$

- $\mu \in \mathbb{R}^d$: Mean vector
- $\Sigma \in \mathbb{R}^{d \times d}$: Covariance matrix (positive definite)
- $|\Sigma|$: Determinant
- $\Sigma^{-1}$: Precision matrix

### 4.2 Role of Covariance Matrix

Covariance matrix $\Sigma$:

$$\Sigma_{ij} = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$$

**Properties**:
1. Symmetric: $\Sigma = \Sigma^T$
2. Positive definite: $x^T \Sigma x > 0$ for all $x \neq 0$
3. Eigenvalue decomposition: $\Sigma = Q \Lambda Q^T$

**Contours**: $(x-\mu)^T \Sigma^{-1} (x-\mu) = c$ is an ellipse

### 4.3 Marginalization

Partition $x = \begin{bmatrix} x_a \\ x_b \end{bmatrix}$, and

$$\mu = \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix}$$

**Marginal distribution**:

$$p(x_a) = \int p(x_a, x_b) dx_b = \mathcal{N}(x_a | \mu_a, \Sigma_{aa})$$

→ Marginal of Gaussian is also Gaussian!

### 4.4 Conditional Distribution

**Conditional distribution**:

$$p(x_a | x_b) = \mathcal{N}(x_a | \mu_{a|b}, \Sigma_{a|b})$$

$$\mu_{a|b} = \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b)$$

$$\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}$$

→ Conditional of Gaussian is also Gaussian!

**Important property**: $x_a$ and $x_b$ uncorrelated ($\Sigma_{ab} = 0$) $\Leftrightarrow$ independent

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 2D Gaussian visualization
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8],
                  [0.8, 1]])

# Sampling
np.random.seed(42)
samples = np.random.multivariate_normal(mu, Sigma, 500)

# Grid
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))

# PDF
rv = stats.multivariate_normal(mu, Sigma)
pdf = rv.pdf(pos)

plt.figure(figsize=(14, 4))

# Contour and samples
plt.subplot(131)
plt.contour(X1, X2, pdf, levels=10, cmap='viridis')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Bivariate Gaussian\n(ρ=0.8)')
plt.axis('equal')
plt.grid(True)

# Marginal distribution
plt.subplot(132)
# Marginal distribution of x1
mu_x1 = mu[0]
sigma_x1 = np.sqrt(Sigma[0, 0])
x1_range = np.linspace(-3, 3, 100)
pdf_x1 = stats.norm.pdf(x1_range, mu_x1, sigma_x1)

plt.plot(x1_range, pdf_x1, linewidth=2, label='Marginal p(x₁)')
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.5,
         label='Samples')
plt.xlabel('x₁')
plt.ylabel('Density')
plt.title('Marginal Distribution')
plt.legend()
plt.grid(True)

# Conditional distribution
plt.subplot(133)
# Condition on x2 = 1
x2_cond = 1.0

# Conditional mean and variance
mu_cond = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (x2_cond - mu[1])
sigma_cond = np.sqrt(Sigma[0, 0] - Sigma[0, 1]**2 / Sigma[1, 1])

pdf_cond = stats.norm.pdf(x1_range, mu_cond, sigma_cond)

plt.plot(x1_range, pdf_cond, linewidth=2,
         label=f'p(x₁|x₂={x2_cond})')

# Extract conditional samples
cond_samples = samples[np.abs(samples[:, 1] - x2_cond) < 0.1, 0]
plt.hist(cond_samples, bins=20, density=True, alpha=0.5,
         label='Conditional samples')

plt.xlabel('x₁')
plt.ylabel('Density')
plt.title(f'Conditional Distribution\np(x₁|x₂={x2_cond})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('multivariate_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()

# Formula verification
print("Conditional distribution parameters:")
print(f"  μ_(x₁|x₂={x2_cond}) = {mu_cond:.3f}")
print(f"  σ_(x₁|x₂={x2_cond}) = {sigma_cond:.3f}")

print("\nCovariance matrix:")
print(Sigma)
print(f"\nCorrelation coefficient: ρ = {Sigma[0,1] / np.sqrt(Sigma[0,0] * Sigma[1,1]):.3f}")
```

### 4.5 Geometry of Covariance Matrix

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gaussian_ellipse(mu, Sigma, ax, color='blue', label=''):
    """Draw the contour ellipse of a Gaussian"""
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    # Angle of ellipse
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle = np.degrees(angle)

    # 2-sigma ellipse
    width, height = 2 * 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(mu, width, height, angle=angle,
                      facecolor='none', edgecolor=color,
                      linewidth=2, label=label)
    ax.add_patch(ellipse)

    # Draw principal axes
    for i in range(2):
        v = eigenvectors[:, i] * 2 * np.sqrt(eigenvalues[i])
        ax.arrow(mu[0], mu[1], v[0], v[1],
                head_width=0.1, head_length=0.1,
                fc=color, ec=color, alpha=0.5)

mu = np.array([0, 0])

# Various covariance matrices
Sigmas = [
    (np.array([[1, 0], [0, 1]]), 'Isotropic'),
    (np.array([[2, 0], [0, 0.5]]), 'Diagonal'),
    (np.array([[1, 0.8], [0.8, 1]]), 'Correlated (+)'),
    (np.array([[1, -0.8], [-0.8, 1]]), 'Correlated (-)'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, (Sigma, title) in zip(axes, Sigmas):
    # Contour ellipse
    plot_gaussian_ellipse(mu, Sigma, ax, color='blue', label='2σ ellipse')

    # Samples
    samples = np.random.multivariate_normal(mu, Sigma, 200)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    # Display covariance matrix
    ax.text(0.05, 0.95, f'Σ =\n{Sigma}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('covariance_geometry.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 5. Gaussian Mixture Model (GMM)

### 5.1 Model Definition

**Gaussian Mixture Model (GMM)**:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

- $K$: Number of components
- $\pi_k$: Mixing proportions ($\sum_k \pi_k = 1, \pi_k \geq 0$)
- $\mu_k, \Sigma_k$: Mean, covariance of $k$-th Gaussian

**Latent variable interpretation**:
- $z_i \in \{1, \ldots, K\}$: Which component data $x_i$ came from
- $p(z_i = k) = \pi_k$
- $p(x_i | z_i = k) = \mathcal{N}(x_i | \mu_k, \Sigma_k)$

### 5.2 EM Algorithm (Connection to Lesson 09)

**E-step**: Compute responsibility

$$\gamma_{ik} = p(z_i = k | x_i) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**M-step**: Update parameters

$$N_k = \sum_{i=1}^n \gamma_{ik}$$

$$\pi_k = \frac{N_k}{n}$$

$$\mu_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i$$

$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$$

### 5.3 Implementation

Extend GMM code from Lesson 09 to multivariate:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class GaussianMixture:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, d = X.shape

        # Initialize
        self.pi = np.ones(self.K) / self.K
        # k-means++ style initialization
        idx = np.random.choice(n, self.K, replace=False)
        self.mu = X[idx]
        self.Sigma = np.array([np.eye(d) for _ in range(self.K)])

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            gamma = self._e_step(X)

            # M-step
            self._m_step(X, gamma)

            # Log-likelihood
            log_likelihood = self._compute_log_likelihood(X)

            if iteration % 10 == 0:
                print(f"Iter {iteration}: log-likelihood = {log_likelihood:.2f}")

            if abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            log_likelihood_old = log_likelihood

        return self

    def _e_step(self, X):
        n = X.shape[0]
        gamma = np.zeros((n, self.K))

        for k in range(self.K):
            gamma[:, k] = self.pi[k] * stats.multivariate_normal.pdf(
                X, self.mu[k], self.Sigma[k])

        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, X, gamma):
        n = X.shape[0]
        Nk = gamma.sum(axis=0)

        self.pi = Nk / n
        self.mu = (gamma.T @ X) / Nk[:, np.newaxis]

        for k in range(self.K):
            diff = X - self.mu[k]
            self.Sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / Nk[k]
            self.Sigma[k] += 1e-6 * np.eye(X.shape[1])

    def _compute_log_likelihood(self, X):
        n = X.shape[0]
        log_likelihood = 0

        for i in range(n):
            prob = sum(self.pi[k] * stats.multivariate_normal.pdf(
                X[i], self.mu[k], self.Sigma[k]) for k in range(self.K))
            log_likelihood += np.log(prob + 1e-10)

        return log_likelihood

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(self, X):
        return self._e_step(X)

# Generate data from 3 Gaussians
np.random.seed(42)
n_samples = 300

X1 = np.random.randn(n_samples, 2) @ np.array([[1, 0.5], [0.5, 1]]) + np.array([0, 0])
X2 = np.random.randn(n_samples, 2) @ np.array([[1, -0.3], [-0.3, 0.5]]) + np.array([5, 5])
X3 = np.random.randn(n_samples, 2) @ np.array([[0.5, 0], [0, 1.5]]) + np.array([0, 5])
X = np.vstack([X1, X2, X3])

# Train GMM
gmm = GaussianMixture(n_components=3, max_iter=100)
gmm.fit(X)

# Predict
labels = gmm.predict(X)
proba = gmm.predict_proba(X)

# Visualization
fig = plt.figure(figsize=(16, 5))

# Data
ax1 = plt.subplot(131)
ax1.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10)
ax1.set_title('Original Data')
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.grid(True)

# GMM results
ax2 = plt.subplot(132)
ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5, s=10)
for k in range(gmm.K):
    ax2.scatter(gmm.mu[k, 0], gmm.mu[k, 1],
               marker='x', s=300, linewidths=4, color='red')

    # 2-sigma ellipse
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eig(gmm.Sigma[k])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(gmm.mu[k], width, height, angle=angle,
                     facecolor='none', edgecolor='red', linewidth=2)
    ax2.add_patch(ellipse)

ax2.set_title('GMM Clustering (with 2σ ellipses)')
ax2.set_xlabel('x₁')
ax2.set_ylabel('x₂')
ax2.grid(True)

# Probability density
ax3 = plt.subplot(133)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# GMM probability density
Z = np.zeros_like(xx)
for k in range(gmm.K):
    Z += gmm.pi[k] * stats.multivariate_normal.pdf(
        np.c_[xx.ravel(), yy.ravel()], gmm.mu[k], gmm.Sigma[k]
    ).reshape(xx.shape)

contour = ax3.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.6)
ax3.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
           alpha=0.3, s=10, edgecolors='k', linewidth=0.3)
plt.colorbar(contour, ax=ax3)
ax3.set_title('GMM Probability Density')
ax3.set_xlabel('x₁')
ax3.set_ylabel('x₂')
ax3.grid(True)

plt.tight_layout()
plt.savefig('gmm_2d.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLearned parameters:")
for k in range(gmm.K):
    print(f"\nComponent {k}:")
    print(f"  π_{k} = {gmm.pi[k]:.3f}")
    print(f"  μ_{k} = {gmm.mu[k]}")
    print(f"  Σ_{k} =\n{gmm.Sigma[k]}")
```

## 6. Machine Learning Applications

### 6.1 GLM (Generalized Linear Models)

**Generalized Linear Models** = exponential family + link function:

$$\mathbb{E}[Y|X] = g^{-1}(w^T x)$$

- $g$: Link function
- Response variable $Y$ follows exponential family distribution

**Examples**:
- Linear regression: Gaussian, $g$ = identity
- Logistic regression: Bernoulli, $g$ = logit
- Poisson regression: Poisson, $g$ = log

### 6.2 Bayesian Neural Networks

Prior on weights:

$$w \sim \mathcal{N}(0, \sigma_w^2 I)$$

Posterior approximation (variational inference):

$$q(w) \approx p(w|D)$$

Enables uncertainty quantification during prediction.

### 6.3 Gaussian Process

**Prior in function space**:

$$f \sim \mathcal{GP}(m, k)$$

- $m(x)$: Mean function
- $k(x, x')$: Kernel function (covariance)

**Property**: Function values at any finite set of points follow multivariate Gaussian.

**Advantages**: Uncertainty quantification, kernel trick

### 6.4 Normalizing Flows

Transform from Gaussian to complex distribution:

$$x = f_\theta(z), \quad z \sim \mathcal{N}(0, I)$$

$$p(x) = p(z) \left|\det \frac{\partial f_\theta}{\partial z}\right|^{-1}$$

Learn invertible transformation $f_\theta$.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simple GLM: Poisson regression
from scipy import stats

# Generate data
np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
true_w = np.array([1.5, -0.8])
true_b = 0.5

# Log link: log(λ) = w^T x + b
log_lambda = X @ true_w + true_b
lambda_true = np.exp(log_lambda)

# Poisson sampling
y = np.random.poisson(lambda_true)

# Poisson regression with PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

class PoissonRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        log_lambda = self.linear(x)
        return torch.exp(log_lambda)

model = PoissonRegression(2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Poisson NLL
def poisson_nll(pred, target):
    return -torch.mean(target * torch.log(pred + 1e-8) - pred)

# Training
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(X_tensor).squeeze()
    loss = poisson_nll(pred, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Results
learned_w = model.linear.weight.data.numpy().squeeze()
learned_b = model.linear.bias.data.numpy()[0]

plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Negative Log-Likelihood')
plt.title('Training Loss (Poisson Regression)')
plt.grid(True)

plt.subplot(132)
with torch.no_grad():
    pred_lambda = model(X_tensor).squeeze().numpy()

plt.scatter(lambda_true, y, alpha=0.5, label='True λ vs Observed y')
plt.scatter(pred_lambda, y, alpha=0.5, label='Predicted λ vs Observed y')
plt.plot([0, lambda_true.max()], [0, lambda_true.max()], 'r--', label='y=λ')
plt.xlabel('λ (rate parameter)')
plt.ylabel('y (count)')
plt.title('Poisson Regression Predictions')
plt.legend()
plt.grid(True)

plt.subplot(133)
plt.scatter(range(2), true_w, s=100, label='True weights', zorder=5)
plt.scatter(range(2), learned_w, s=100, marker='x', label='Learned weights', zorder=5)
plt.scatter([2], [true_b], s=100)
plt.scatter([2], [learned_b], s=100, marker='x')
plt.xticks([0, 1, 2], ['w₁', 'w₂', 'b'])
plt.ylabel('Value')
plt.title('Parameter Recovery')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('glm_poisson.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"True weights: {true_w}, bias: {true_b:.3f}")
print(f"Learned weights: {learned_w}, bias: {learned_b:.3f}")
```

## Practice Problems

### Problem 1: Exponential Family Transformation

Transform the following distributions to exponential family form and identify $\eta, T(x), A(\eta), h(x)$:

(a) Geometric distribution: $p(x|p) = (1-p)^{x-1} p$ for $x = 1, 2, \ldots$

(b) Laplace distribution: $p(x|\mu, b) = \frac{1}{2b} \exp\left(-\frac{|x-\mu|}{b}\right)$ (with $b$ fixed)

### Problem 2: Properties of Log Partition Function

In exponential family distributions, prove that the first and second derivatives of $A(\eta)$ are the expectation and variance:

(a) $\frac{\partial A}{\partial \eta} = \mathbb{E}[T(X)]$

(b) $\frac{\partial^2 A}{\partial \eta^2} = \text{Var}(T(X))$

**Hint**: Differentiate the normalization condition $\int h(x) \exp(\eta^T T(x) - A(\eta)) dx = 1$ with respect to $\eta$.

### Problem 3: Gamma-Poisson Conjugacy

Show that the conjugate prior for Poisson distribution $p(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$ is the Gamma distribution $\text{Gamma}(\alpha, \beta)$, and derive the posterior.

**Data**: $n$ observations, total $\sum x_i = s$

**Posterior**: $\lambda | D \sim \text{Gamma}(\alpha + s, \beta + n)$

Verify with Python.

### Problem 4: Conditional of Multivariate Gaussian

For 3-dimensional Gaussian $\mathcal{N}(\mu, \Sigma)$:

$$\mu = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} 1 & 0.5 & 0.3 \\ 0.5 & 2 & 0.4 \\ 0.3 & 0.4 & 1 \end{bmatrix}$$

Given $x_3 = 1.5$, compute the conditional distribution parameters for $(x_1, x_2)$.

### Problem 5: GMM vs K-means

(a) Explain the differences between GMM and K-means.

(b) Show that K-means can be interpreted as a special case of GMM.
    **Hint**: Limit where $\Sigma_k = \sigma^2 I$ and $\sigma \to 0$.

(c) Compare K-means and GMM on the same data in Python.

## References

1. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Chapter 2 (Probability Distributions), Chapter 9 (EM and GMM).
2. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: An Introduction*. Chapter 3 (Probability Distributions), Chapter 21 (GLM).
3. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Chapter 6 (Probability and Distributions).
4. **Rasmussen, C. E., & Williams, C. K. I. (2006).** *Gaussian Processes for Machine Learning*. MIT Press.
5. **Paper**: McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall.
6. **scikit-learn documentation**: Gaussian Mixture - https://scikit-learn.org/stable/modules/mixture.html
7. **Tutorial**: Rezende, D. J., & Mohamed, S. (2015). "Variational Inference with Normalizing Flows". *ICML*.
8. **Blog**: Distill.pub - Exponential Family - https://distill.pub/

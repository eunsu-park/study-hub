# Multivariate Normal Distribution

**Previous**: [Transformations of Random Variables](./08_Transformations_of_Random_Variables.md) | **Next**: [Convergence Concepts](./10_Convergence_Concepts.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write the density of the multivariate normal $\mathbf{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
2. Explain the role of the covariance matrix $\boldsymbol{\Sigma}$ and its properties
3. Derive marginal and conditional distributions from a joint multivariate normal
4. Apply linear transformations to multivariate normal vectors
5. Compute and interpret the Mahalanobis distance
6. Connect the multivariate normal to the chi-squared distribution
7. Describe the bivariate normal case and its contour geometry
8. Generate multivariate normal samples via Cholesky decomposition

---

The multivariate normal (MVN) distribution is the foundation of multivariate statistics, underpinning linear regression, discriminant analysis, principal components, and countless other methods. It extends the familiar bell curve to $p$ dimensions, where the covariance matrix captures the full dependence structure among variables.

---

## 1. Definition and Density

### 1.1 The Density Formula

A random vector $\mathbf{X} = (X_1, X_2, \ldots, X_p)^T$ follows a **$p$-variate normal distribution** $\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ if its joint PDF is:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where:

- $\boldsymbol{\mu} \in \mathbb{R}^p$ is the **mean vector**
- $\boldsymbol{\Sigma} \in \mathbb{R}^{p \times p}$ is the **covariance matrix** (symmetric, positive definite)
- $|\boldsymbol{\Sigma}|$ denotes the determinant of $\boldsymbol{\Sigma}$

### 1.2 Equivalent Characterisation via MGF

$$M_{\mathbf{X}}(\mathbf{t}) = E[e^{\mathbf{t}^T \mathbf{X}}] = \exp\!\left(\mathbf{t}^T \boldsymbol{\mu} + \frac{1}{2}\mathbf{t}^T \boldsymbol{\Sigma}\, \mathbf{t}\right)$$

This MGF uniquely determines the MVN distribution.

### 1.3 Alternative Definition

$\mathbf{X}$ is multivariate normal if and only if every linear combination $\mathbf{a}^T \mathbf{X} = a_1 X_1 + \cdots + a_p X_p$ is (univariate) normal for all $\mathbf{a} \in \mathbb{R}^p$.

---

## 2. The Covariance Matrix $\boldsymbol{\Sigma}$

### 2.1 Structure

$$\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1p} \\ \sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_p^2 \end{pmatrix}$$

where $\sigma_{ij} = \text{Cov}(X_i, X_j)$ and $\sigma_{ii} = \text{Var}(X_i)$.

### 2.2 Positive Semi-Definite Property

$\boldsymbol{\Sigma}$ must be **positive semi-definite** (PSD): for all $\mathbf{a} \in \mathbb{R}^p$,

$$\mathbf{a}^T \boldsymbol{\Sigma}\, \mathbf{a} = \text{Var}(\mathbf{a}^T \mathbf{X}) \ge 0$$

When $\boldsymbol{\Sigma}$ is **positive definite** (all eigenvalues strictly positive), the density exists and the distribution is non-degenerate.

### 2.3 Eigendecomposition

Since $\boldsymbol{\Sigma}$ is symmetric PSD, it has the spectral decomposition:

$$\boldsymbol{\Sigma} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$$

where $\mathbf{Q}$ is orthogonal (columns are eigenvectors) and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_p)$ with $\lambda_i \ge 0$.

The eigenvectors define the **principal axes** of the distribution, and the eigenvalues give the variance along each axis.

### 2.4 Correlation Matrix

The correlation matrix $\mathbf{R}$ is the standardised covariance:

$$R_{ij} = \frac{\sigma_{ij}}{\sigma_i \sigma_j}, \quad \mathbf{R} = \mathbf{D}^{-1} \boldsymbol{\Sigma}\, \mathbf{D}^{-1}$$

where $\mathbf{D} = \text{diag}(\sigma_1, \ldots, \sigma_p)$.

---

## 3. Marginal Distributions

### 3.1 Marginals Are Normal

If $\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then any sub-vector of $\mathbf{X}$ is also multivariate normal.

Partition $\mathbf{X}$, $\boldsymbol{\mu}$, and $\boldsymbol{\Sigma}$ into two groups:

$$\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$$

Then:

$$\mathbf{X}_1 \sim N(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11}), \qquad \mathbf{X}_2 \sim N(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_{22})$$

### 3.2 Important Caveat

The converse is **not** true: having individually normal marginals does not guarantee a joint multivariate normal. The MVN requires all linear combinations to be normal.

---

## 4. Conditional Distributions

### 4.1 Conditional Normal Formula

The conditional distribution of $\mathbf{X}_1$ given $\mathbf{X}_2 = \mathbf{x}_2$ is:

$$\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim N\!\left(\boldsymbol{\mu}_{1|2},\, \boldsymbol{\Sigma}_{1|2}\right)$$

where:

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

### 4.2 Key Observations

- The conditional mean is a **linear function** of the conditioning variable $\mathbf{x}_2$.
- The conditional covariance $\boldsymbol{\Sigma}_{1|2}$ does **not depend on** $\mathbf{x}_2$; it is always the same regardless of what value is observed.
- The matrix $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$ plays the role of a **regression coefficient** matrix.

### 4.3 Scalar Example (Bivariate Case)

For $p = 2$ with correlation $\rho$:

$$X_1 \mid X_2 = x_2 \sim N\!\left(\mu_1 + \rho\frac{\sigma_1}{\sigma_2}(x_2 - \mu_2),\; \sigma_1^2(1 - \rho^2)\right)$$

The conditional variance is reduced by the factor $(1 - \rho^2)$: the stronger the correlation, the more information $X_2$ provides about $X_1$.

---

## 5. Linear Transformations

### 5.1 Main Result

If $\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{A}$ is a $q \times p$ matrix with $\mathbf{b} \in \mathbb{R}^q$, then:

$$\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b} \sim N_q(\mathbf{A}\boldsymbol{\mu} + \mathbf{b},\; \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

### 5.2 Consequences

- **Standardisation**: $\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu}) \sim N_p(\mathbf{0}, \mathbf{I}_p)$
- **Decorrelation**: Using the eigen-decomposition $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$, define $\mathbf{Y} = \mathbf{Q}^T(\mathbf{X} - \boldsymbol{\mu})$. Then $\mathbf{Y} \sim N_p(\mathbf{0}, \boldsymbol{\Lambda})$, meaning the components of $\mathbf{Y}$ are independent.
- **Projection**: Any $\mathbf{a}^T \mathbf{X}$ is univariate normal with mean $\mathbf{a}^T\boldsymbol{\mu}$ and variance $\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}$.

---

## 6. Mahalanobis Distance

### 6.1 Definition

The **Mahalanobis distance** of a point $\mathbf{x}$ from the distribution $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ is:

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

### 6.2 Interpretation

- It generalises the univariate $z$-score $|x - \mu|/\sigma$ to multiple dimensions.
- It accounts for correlations among variables and different variances along each axis.
- Points on the same contour of the MVN density have the same Mahalanobis distance.
- When $\boldsymbol{\Sigma} = \mathbf{I}$, Mahalanobis distance reduces to Euclidean distance from $\boldsymbol{\mu}$.

### 6.3 Use Cases

- Outlier detection in multivariate data
- Classification (e.g., quadratic discriminant analysis)
- Hypothesis testing for multivariate means

---

## 7. Connection to Chi-Squared

### 7.1 Squared Mahalanobis Distance

If $\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then the squared Mahalanobis distance:

$$D_M^2 = (\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2(p)$$

**Proof sketch**: Let $\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu}) \sim N_p(\mathbf{0}, \mathbf{I})$. Then $D_M^2 = \mathbf{Z}^T\mathbf{Z} = \sum_{i=1}^p Z_i^2$, which is a sum of $p$ independent squared standard normals, hence $\chi^2(p)$.

### 7.2 Application: Confidence Ellipsoids

The set $\{\mathbf{x} : (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) \le c^2\}$ is an ellipsoid that contains probability mass equal to $P(\chi^2(p) \le c^2)$. For $p = 2$, $c^2 = 5.991$ gives a 95% confidence ellipse.

---

## 8. Bivariate Normal: Geometry and Contours

### 8.1 The Bivariate Case

For $p = 2$ with parameters $\boldsymbol{\mu} = (\mu_1, \mu_2)^T$, variances $\sigma_1^2, \sigma_2^2$, and correlation $\rho$:

$$f(x_1, x_2) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}} \exp\!\left(-\frac{Q}{2(1-\rho^2)}\right)$$

where:

$$Q = \frac{(x_1-\mu_1)^2}{\sigma_1^2} - \frac{2\rho(x_1-\mu_1)(x_2-\mu_2)}{\sigma_1\sigma_2} + \frac{(x_2-\mu_2)^2}{\sigma_2^2}$$

### 8.2 Contour Geometry

The constant-density contours are **ellipses** centred at $\boldsymbol{\mu}$:

- When $\rho = 0$: ellipses aligned with the coordinate axes
- When $\rho > 0$: ellipses tilted toward the positive diagonal
- When $\rho < 0$: ellipses tilted toward the negative diagonal
- When $\sigma_1 = \sigma_2$ and $\rho = 0$: the contours are circles

The principal axes of the ellipse are determined by the eigenvectors of $\boldsymbol{\Sigma}$, and their lengths are proportional to the square roots of the eigenvalues.

### 8.3 Independence vs. Uncorrelatedness

For the MVN (and **only** for the MVN), uncorrelatedness implies independence:

$$\rho = 0 \iff X_1 \perp X_2 \quad \text{(for jointly normal variables)}$$

This is a special property that does not hold for arbitrary joint distributions.

---

## 9. Principal Components Connection

### 9.1 Brief Overview

**Principal Component Analysis (PCA)** is directly linked to the eigendecomposition of $\boldsymbol{\Sigma}$.

- The $k$-th principal component is $Y_k = \mathbf{q}_k^T(\mathbf{X} - \boldsymbol{\mu})$, where $\mathbf{q}_k$ is the $k$-th eigenvector of $\boldsymbol{\Sigma}$.
- $\text{Var}(Y_k) = \lambda_k$ (the $k$-th eigenvalue).
- Under the MVN, the principal components $Y_1, Y_2, \ldots, Y_p$ are mutually independent.
- PCA finds the directions of maximum variance in the data, which correspond to the principal axes of the MVN density contours.

This connection between the MVN and PCA is why normality assumptions appear so frequently in multivariate analysis.

---

## 10. Python Examples

### 10.1 Generating Bivariate Normal Samples via Cholesky Decomposition

The Cholesky decomposition factorises a positive definite matrix as $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$ where $\mathbf{L}$ is lower triangular. To sample $\mathbf{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

1. Generate $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$
2. Compute $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$

```python
import random
import math

def cholesky_2x2(sigma):
    """Cholesky decomposition for a 2x2 positive definite matrix.
    sigma = [[a, b], [b, d]] -> L such that L @ L^T = sigma.
    """
    a, b = sigma[0][0], sigma[0][1]
    d = sigma[1][1]
    l11 = math.sqrt(a)
    l21 = b / l11
    l22 = math.sqrt(d - l21 ** 2)
    return [[l11, 0.0], [l21, l22]]

def sample_bivariate_normal(mu, sigma, n, seed=42):
    """Generate n samples from N(mu, sigma) using Cholesky."""
    random.seed(seed)
    L = cholesky_2x2(sigma)
    samples = []
    for _ in range(n):
        z1 = random.gauss(0, 1)
        z2 = random.gauss(0, 1)
        x1 = mu[0] + L[0][0] * z1 + L[0][1] * z2
        x2 = mu[1] + L[1][0] * z1 + L[1][1] * z2
        samples.append((x1, x2))
    return samples

# Parameters
mu = [2.0, 5.0]
sigma = [[4.0, 3.0],    # Var(X1)=4, Cov=3
         [3.0, 9.0]]    # Var(X2)=9
rho_true = 3.0 / (2.0 * 3.0)  # 0.5

samples = sample_bivariate_normal(mu, sigma, n=50_000)

# Verify sample statistics
x1 = [s[0] for s in samples]
x2 = [s[1] for s in samples]

mean1 = sum(x1) / len(x1)
mean2 = sum(x2) / len(x2)
var1 = sum((v - mean1) ** 2 for v in x1) / (len(x1) - 1)
var2 = sum((v - mean2) ** 2 for v in x2) / (len(x2) - 1)
cov12 = sum((x1[i] - mean1) * (x2[i] - mean2)
            for i in range(len(x1))) / (len(x1) - 1)

print("Bivariate Normal via Cholesky:")
print(f"  E[X1] = {mean1:.4f}  (true: {mu[0]})")
print(f"  E[X2] = {mean2:.4f}  (true: {mu[1]})")
print(f"  Var(X1) = {var1:.4f}  (true: {sigma[0][0]})")
print(f"  Var(X2) = {var2:.4f}  (true: {sigma[1][1]})")
print(f"  Cov(X1,X2) = {cov12:.4f}  (true: {sigma[0][1]})")
print(f"  Corr = {cov12 / math.sqrt(var1 * var2):.4f}  (true: {rho_true:.4f})")
```

### 10.2 Conditional Distribution Verification

```python
import random
import math

random.seed(100)
mu = [0.0, 0.0]
sigma = [[1.0, 0.6], [0.6, 1.0]]  # rho = 0.6
n = 200_000

samples = sample_bivariate_normal(mu, sigma, n, seed=100)

# Condition on X2 being near 1.5
x2_target = 1.5
tol = 0.05
conditional_x1 = [s[0] for s in samples
                   if abs(s[1] - x2_target) < tol]

# Theoretical: X1 | X2=1.5 ~ N(rho * 1.5, 1 - rho^2)
rho = 0.6
cond_mean_theory = rho * x2_target
cond_var_theory = 1 - rho ** 2

cond_mean_sample = sum(conditional_x1) / len(conditional_x1)
cond_var_sample = (sum((x - cond_mean_sample) ** 2 for x in conditional_x1)
                   / (len(conditional_x1) - 1))

print(f"\nConditional X1 | X2 ~ {x2_target} (n = {len(conditional_x1)}):")
print(f"  Sample mean:     {cond_mean_sample:.4f}  (theory: {cond_mean_theory:.4f})")
print(f"  Sample variance: {cond_var_sample:.4f}  (theory: {cond_var_theory:.4f})")
```

### 10.3 Mahalanobis Distance and Chi-Squared Check

```python
import random
import math

random.seed(200)
mu = [1.0, 3.0]
sigma = [[2.0, 1.0], [1.0, 4.0]]
n = 100_000

samples = sample_bivariate_normal(mu, sigma, n, seed=200)

# Compute inverse of 2x2 sigma
det_s = sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0]
inv_s = [[sigma[1][1] / det_s, -sigma[0][1] / det_s],
         [-sigma[1][0] / det_s, sigma[0][0] / det_s]]

# Compute squared Mahalanobis distances
d2_samples = []
for x1, x2 in samples:
    dx = [x1 - mu[0], x2 - mu[1]]
    d2 = (dx[0] * (inv_s[0][0] * dx[0] + inv_s[0][1] * dx[1])
          + dx[1] * (inv_s[1][0] * dx[0] + inv_s[1][1] * dx[1]))
    d2_samples.append(d2)

# Should follow chi-squared(2): mean=2, variance=4
d2_mean = sum(d2_samples) / n
d2_var = sum((d - d2_mean) ** 2 for d in d2_samples) / (n - 1)

print(f"\nSquared Mahalanobis distance (should be Chi-sq(2)):")
print(f"  Mean: {d2_mean:.4f}  (theoretical: 2)")
print(f"  Var:  {d2_var:.4f}  (theoretical: 4)")

# Fraction within 95% confidence ellipse
# chi2(2) at 95% is 5.991
within_95 = sum(1 for d in d2_samples if d <= 5.991) / n
print(f"  Within 95% ellipse: {within_95:.4f}  (theoretical: 0.95)")
```

### 10.4 Linear Transformation Verification

```python
import random
import math

random.seed(300)
mu = [1.0, 2.0]
sigma = [[4.0, 1.0], [1.0, 2.0]]
n = 100_000

samples = sample_bivariate_normal(mu, sigma, n, seed=300)

# Apply A = [[2, 1], [0, 3]], b = [1, -1]
# Y = AX + b ~ N(A*mu + b, A*Sigma*A^T)
A = [[2, 1], [0, 3]]
b = [1.0, -1.0]

y_samples = []
for x1, x2 in samples:
    y1 = A[0][0] * x1 + A[0][1] * x2 + b[0]
    y2 = A[1][0] * x1 + A[1][1] * x2 + b[1]
    y_samples.append((y1, y2))

# Theoretical: E[Y] = A*mu + b
ey1 = A[0][0] * mu[0] + A[0][1] * mu[1] + b[0]
ey2 = A[1][0] * mu[0] + A[1][1] * mu[1] + b[1]

# A*Sigma*A^T
# First: A*Sigma
AS = [[A[0][0]*sigma[0][0]+A[0][1]*sigma[1][0],
       A[0][0]*sigma[0][1]+A[0][1]*sigma[1][1]],
      [A[1][0]*sigma[0][0]+A[1][1]*sigma[1][0],
       A[1][0]*sigma[0][1]+A[1][1]*sigma[1][1]]]
# Then: (A*Sigma)*A^T
ASAT = [[AS[0][0]*A[0][0]+AS[0][1]*A[0][1],
         AS[0][0]*A[1][0]+AS[0][1]*A[1][1]],
        [AS[1][0]*A[0][0]+AS[1][1]*A[0][1],
         AS[1][0]*A[1][0]+AS[1][1]*A[1][1]]]

y1_vals = [y[0] for y in y_samples]
y2_vals = [y[1] for y in y_samples]
my1 = sum(y1_vals) / n
my2 = sum(y2_vals) / n

print(f"\nLinear transformation Y = AX + b:")
print(f"  E[Y1] = {my1:.4f}  (theory: {ey1:.4f})")
print(f"  E[Y2] = {my2:.4f}  (theory: {ey2:.4f})")
print(f"  Var(Y1) = {sum((v-my1)**2 for v in y1_vals)/(n-1):.4f}  "
      f"(theory: {ASAT[0][0]:.4f})")
print(f"  Var(Y2) = {sum((v-my2)**2 for v in y2_vals)/(n-1):.4f}  "
      f"(theory: {ASAT[1][1]:.4f})")
```

---

## Key Takeaways

1. The **multivariate normal** is fully specified by its mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$; the quadratic form in the exponent is the squared Mahalanobis distance.
2. **Marginal distributions** of any sub-vector are also multivariate normal; **conditional distributions** are normal with a mean that is linear in the conditioning variable.
3. **Linear transformations** preserve normality: $\mathbf{A}\mathbf{X} + \mathbf{b}$ is MVN when $\mathbf{X}$ is MVN.
4. The **Mahalanobis distance** generalises the $z$-score to multiple dimensions and connects to $\chi^2(p)$.
5. For jointly normal variables, **uncorrelated implies independent** — a property unique to the normal family.
6. The **Cholesky decomposition** provides an efficient algorithm for generating MVN samples from independent standard normals.
7. The eigendecomposition of $\boldsymbol{\Sigma}$ reveals the principal axes and connects to PCA.

---

*Next lesson: [Convergence Concepts](./10_Convergence_Concepts.md)*

# Transformations of Random Variables

**Previous**: [Continuous Distribution Families](./07_Continuous_Distribution_Families.md) | **Next**: [Multivariate Normal Distribution](./09_Multivariate_Normal_Distribution.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive the distribution of $Y = g(X)$ using the CDF technique
2. Apply the change-of-variable formula (Jacobian method) for monotonic transformations
3. Transform discrete random variables through function mappings
4. Find the distribution of sums, products, and ratios of two random variables
5. Use the convolution formula for sums of independent random variables
6. Apply the MGF technique to identify distributions of sums
7. Derive the PDFs of order statistics
8. Explain and implement the Box-Muller transform

---

Given a random variable $X$ with a known distribution, we frequently need the distribution of a transformed variable $Y = g(X)$. This lesson develops the core techniques for finding such distributions, from single-variable transformations to functions of multiple random variables and order statistics.

---

## 1. Functions of a Single Random Variable

### 1.1 The Problem

Suppose $X$ has a known PDF $f_X(x)$ and we define $Y = g(X)$. What is $f_Y(y)$?

There is no single formula that works for every function $g$; the approach depends on whether $g$ is monotonic, piecewise monotonic, or more complex. We introduce two general techniques: the CDF method and the PDF (Jacobian) method.

---

## 2. CDF Technique

### 2.1 General Procedure

The CDF technique works for **any** measurable function $g$. The idea is straightforward:

1. Write $F_Y(y) = P(Y \le y) = P(g(X) \le y)$.
2. Express the event $\{g(X) \le y\}$ in terms of $X$ (i.e., find the set $A_y = \{x : g(x) \le y\}$).
3. Compute $F_Y(y) = P(X \in A_y) = \int_{A_y} f_X(x)\,dx$.
4. Differentiate: $f_Y(y) = F_Y'(y)$.

### 2.2 Example: $Y = X^2$ where $X \sim N(0, 1)$

For $y > 0$:

$$F_Y(y) = P(X^2 \le y) = P(-\sqrt{y} \le X \le \sqrt{y}) = \Phi(\sqrt{y}) - \Phi(-\sqrt{y}) = 2\Phi(\sqrt{y}) - 1$$

Differentiating:

$$f_Y(y) = 2\phi(\sqrt{y}) \cdot \frac{1}{2\sqrt{y}} = \frac{1}{\sqrt{y}} \phi(\sqrt{y}) = \frac{1}{\sqrt{2\pi y}}\, e^{-y/2}$$

This is the PDF of $\chi^2(1)$, confirming that the square of a standard normal is chi-squared with one degree of freedom.

### 2.3 Example: $Y = -\ln(X)$ where $X \sim \text{Uniform}(0, 1)$

For $y \ge 0$:

$$F_Y(y) = P(-\ln X \le y) = P(X \ge e^{-y}) = 1 - e^{-y}$$

This is the CDF of $\text{Exp}(1)$. This is the inverse-transform method for generating exponential random variables.

---

## 3. PDF Technique with Jacobian (Monotonic Transformations)

### 3.1 One-to-One (Monotonic) Case

If $g$ is a strictly monotonic and differentiable function with inverse $x = g^{-1}(y)$, then:

$$f_Y(y) = f_X\!\left(g^{-1}(y)\right) \cdot \left|\frac{dx}{dy}\right|$$

The factor $\left|\frac{dx}{dy}\right| = \left|\frac{d}{dy}g^{-1}(y)\right|$ is the **Jacobian** of the inverse transformation. The absolute value ensures the density remains non-negative regardless of whether $g$ is increasing or decreasing.

### 3.2 Derivation

Starting from the CDF technique for a strictly increasing $g$:

$$F_Y(y) = P(g(X) \le y) = P(X \le g^{-1}(y)) = F_X(g^{-1}(y))$$

Differentiating by the chain rule:

$$f_Y(y) = f_X(g^{-1}(y)) \cdot \frac{d}{dy}g^{-1}(y)$$

For strictly decreasing $g$, the inequality reverses, introducing a minus sign that the absolute value absorbs.

### 3.3 Example: Log-Normal

If $X \sim N(\mu, \sigma^2)$ and $Y = e^X$, then $X = \ln Y$ and $dx/dy = 1/y$. For $y > 0$:

$$f_Y(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right)$$

This is the **log-normal** distribution.

### 3.4 Piecewise Monotonic Functions

If $g$ is not globally monotonic but can be split into intervals where it is monotonic, sum over each branch:

$$f_Y(y) = \sum_{i} f_X(x_i) \cdot \left|\frac{dx_i}{dy}\right|$$

where $x_1, x_2, \ldots$ are all solutions of $g(x) = y$.

---

## 4. Transformations of Discrete Variables

For a discrete random variable $X$ with PMF $p_X(x)$ and a function $Y = g(X)$:

$$p_Y(y) = P(Y = y) = \sum_{x:\, g(x) = y} p_X(x)$$

Simply collect all values of $X$ that map to the same $y$ and sum their probabilities.

**Example**: Let $X \sim \text{Binomial}(n, p)$ and $Y = n - X$ (number of failures). Then $Y \sim \text{Binomial}(n, 1-p)$.

---

## 5. Functions of Two Random Variables

### 5.1 Sum: $Z = X + Y$

Given the joint PDF $f_{X,Y}(x, y)$, define the transformation $(X, Y) \to (Z, W)$ where $Z = X + Y$ and $W = Y$ (auxiliary). The Jacobian is:

$$\frac{\partial(x, y)}{\partial(z, w)} = \begin{vmatrix} 1 & -1 \\ 0 & 1 \end{vmatrix} = 1$$

So $f_{Z,W}(z, w) = f_{X,Y}(z - w,\, w)$, and marginalising over $w$:

$$f_Z(z) = \int_{-\infty}^{\infty} f_{X,Y}(z - w,\, w)\, dw$$

### 5.2 Product: $Z = XY$

Use the substitution $Z = XY$, $W = Y$. Then $X = Z/W$, and $|J| = 1/|w|$:

$$f_Z(z) = \int_{-\infty}^{\infty} f_{X,Y}\!\left(\frac{z}{w},\, w\right) \frac{1}{|w|}\, dw$$

### 5.3 Ratio: $Z = X/Y$

Use $Z = X/Y$, $W = Y$. Then $X = ZW$, and $|J| = |w|$:

$$f_Z(z) = \int_{-\infty}^{\infty} f_{X,Y}(zw,\, w)\, |w|\, dw$$

This technique is used to derive the $t$-distribution and $F$-distribution from normals and chi-squareds.

---

## 6. Convolution for Sums of Independent Random Variables

### 6.1 The Convolution Formula

When $X$ and $Y$ are **independent**, $f_{X,Y}(x, y) = f_X(x) f_Y(y)$, and the PDF of $Z = X + Y$ simplifies to the **convolution**:

$$f_Z(z) = (f_X * f_Y)(z) = \int_{-\infty}^{\infty} f_X(z - y)\, f_Y(y)\, dy$$

### 6.2 Example: Sum of Two Independent Exponentials

Let $X \sim \text{Exp}(\lambda)$ and $Y \sim \text{Exp}(\lambda)$ independently. For $z > 0$:

$$f_Z(z) = \int_0^z \lambda e^{-\lambda(z-y)} \cdot \lambda e^{-\lambda y}\, dy = \lambda^2 e^{-\lambda z} \int_0^z dy = \lambda^2 z\, e^{-\lambda z}$$

This is $\text{Gamma}(2, \lambda)$ (i.e., $\text{Erlang}(2, \lambda)$), confirming the additive property of the Gamma family.

### 6.3 Discrete Convolution

For independent discrete random variables:

$$p_Z(z) = \sum_k p_X(k)\, p_Y(z - k)$$

---

## 7. MGF Technique for Sums

### 7.1 The Key Property

If $X$ and $Y$ are independent, the MGF of $Z = X + Y$ is:

$$M_Z(t) = E[e^{tZ}] = E[e^{t(X+Y)}] = E[e^{tX}] \cdot E[e^{tY}] = M_X(t) \cdot M_Y(t)$$

Since the MGF uniquely determines the distribution (when it exists in a neighbourhood of zero), we can identify $f_Z$ by recognising the product of MGFs.

### 7.2 Example: Sum of Independent Normals

If $X \sim N(\mu_1, \sigma_1^2)$ and $Y \sim N(\mu_2, \sigma_2^2)$ are independent:

$$M_Z(t) = \exp\!\left(\mu_1 t + \frac{\sigma_1^2 t^2}{2}\right) \cdot \exp\!\left(\mu_2 t + \frac{\sigma_2^2 t^2}{2}\right) = \exp\!\left((\mu_1+\mu_2)t + \frac{(\sigma_1^2+\sigma_2^2)t^2}{2}\right)$$

This is the MGF of $N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$, confirming closure of the normal family under summation.

### 7.3 When to Prefer MGF over Convolution

- **MGF technique**: Best when the resulting MGF has a recognisable form. Fast and elegant.
- **Convolution**: Necessary when the MGF does not exist or the resulting MGF is not easily recognisable.

---

## 8. Order Statistics

### 8.1 Setup

Given $X_1, X_2, \ldots, X_n$ i.i.d. with CDF $F(x)$ and PDF $f(x)$, the **order statistics** are:

$$X_{(1)} \le X_{(2)} \le \cdots \le X_{(n)}$$

where $X_{(k)}$ is the $k$-th smallest value.

### 8.2 Distribution of the Minimum $X_{(1)}$

$$P(X_{(1)} > x) = P(\text{all } X_i > x) = [1 - F(x)]^n$$

$$f_{X_{(1)}}(x) = n[1 - F(x)]^{n-1} f(x)$$

### 8.3 Distribution of the Maximum $X_{(n)}$

$$F_{X_{(n)}}(x) = P(\text{all } X_i \le x) = [F(x)]^n$$

$$f_{X_{(n)}}(x) = n[F(x)]^{n-1} f(x)$$

### 8.4 General $k$-th Order Statistic

$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} [F(x)]^{k-1} [1-F(x)]^{n-k} f(x)$$

**Interpretation**: Of the $n$ values, exactly $k-1$ are below $x$, one is at $x$, and $n-k$ are above $x$. The combinatorial factor counts the arrangements.

### 8.5 Special Case: Uniform Order Statistics

If $X_i \sim \text{Uniform}(0,1)$, then $X_{(k)} \sim \text{Beta}(k, n - k + 1)$. This elegant connection is fundamental to nonparametric statistics.

---

## 9. The Box-Muller Transform

### 9.1 Motivation

How can we generate normal random variables from uniform ones? The Box-Muller transform provides an exact method.

### 9.2 The Transform

Let $U_1, U_2 \sim \text{Uniform}(0,1)$ be independent. Define:

$$Z_1 = \sqrt{-2\ln U_1}\, \cos(2\pi U_2)$$
$$Z_2 = \sqrt{-2\ln U_1}\, \sin(2\pi U_2)$$

Then $Z_1$ and $Z_2$ are **independent** $N(0,1)$ random variables.

### 9.3 Why It Works

The transform can be verified by the change-of-variable technique in two dimensions. In polar coordinates $(R, \Theta)$ where $R = \sqrt{-2\ln U_1}$ and $\Theta = 2\pi U_2$:

- $R^2 = -2\ln U_1 \sim \text{Exp}(1/2)$, which equals $\chi^2(2)$
- $\Theta \sim \text{Uniform}(0, 2\pi)$
- $R$ and $\Theta$ are independent

The joint density of $(Z_1, Z_2)$ factors into two independent standard normal densities.

---

## 10. Python Examples

### 10.1 CDF Technique: Verifying $Y = X^2$ for Standard Normal

```python
import random
import math

random.seed(42)
n = 200_000
z_samples = [random.gauss(0, 1) for _ in range(n)]
y_samples = [z ** 2 for z in z_samples]

# Chi-squared(1) has mean=1 and variance=2
mean_y = sum(y_samples) / n
var_y = sum((y - mean_y) ** 2 for y in y_samples) / (n - 1)

print("Y = Z^2 where Z ~ N(0,1):")
print(f"  Sample mean:     {mean_y:.4f}  (theoretical: 1.0)")
print(f"  Sample variance: {var_y:.4f}  (theoretical: 2.0)")
```

### 10.2 Inverse-Transform: Generating Exponential from Uniform

```python
import random
import math

random.seed(7)
lam = 2.5
n = 100_000

# Inverse-transform: X = -ln(U) / lambda
uniform_samples = [random.random() for _ in range(n)]
exp_samples = [-math.log(u) / lam for u in uniform_samples]

mean_est = sum(exp_samples) / n
var_est = sum((x - mean_est) ** 2 for x in exp_samples) / (n - 1)

print(f"Exp({lam}) via inverse transform:")
print(f"  Sample mean:     {mean_est:.4f}  (theoretical: {1/lam:.4f})")
print(f"  Sample variance: {var_est:.4f}  (theoretical: {1/lam**2:.4f})")
```

### 10.3 Convolution: Sum of Two Independent Uniforms

```python
import random

random.seed(314)
n = 100_000

# Z = U1 + U2, each Uniform(0,1)
# The result is a triangular distribution on [0, 2]
z_samples = [random.random() + random.random() for _ in range(n)]

mean_z = sum(z_samples) / n
var_z = sum((z - mean_z) ** 2 for z in z_samples) / (n - 1)

print("Z = U1 + U2 (triangular distribution):")
print(f"  Sample mean:     {mean_z:.4f}  (theoretical: 1.0)")
print(f"  Sample variance: {var_z:.4f}  (theoretical: {1/6:.4f})")

# Histogram approximation using text
bins = [0] * 20
for z in z_samples:
    idx = min(int(z * 10), 19)
    bins[idx] += 1

print("\nApproximate shape (triangular):")
for i, count in enumerate(bins):
    bar = '#' * (count * 80 // max(bins))
    print(f"  [{i*0.1:4.1f}-{(i+1)*0.1:4.1f}): {bar}")
```

### 10.4 Box-Muller Transform Implementation

```python
import random
import math

random.seed(2024)
n = 100_000

z1_samples = []
z2_samples = []

for _ in range(n // 2):
    u1 = random.random()
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    z1_samples.append(r * math.cos(theta))
    z2_samples.append(r * math.sin(theta))

all_z = z1_samples + z2_samples

mean_z = sum(all_z) / len(all_z)
var_z = sum((z - mean_z) ** 2 for z in all_z) / (len(all_z) - 1)

print("Box-Muller generated N(0,1) samples:")
print(f"  Sample mean:     {mean_z:.4f}  (theoretical: 0.0)")
print(f"  Sample variance: {var_z:.4f}  (theoretical: 1.0)")

# Verify independence: correlation between z1 and z2
n_pairs = len(z1_samples)
mean1 = sum(z1_samples) / n_pairs
mean2 = sum(z2_samples) / n_pairs
cov = sum((z1_samples[i] - mean1) * (z2_samples[i] - mean2)
          for i in range(n_pairs)) / (n_pairs - 1)
std1 = math.sqrt(sum((z - mean1) ** 2 for z in z1_samples) / (n_pairs - 1))
std2 = math.sqrt(sum((z - mean2) ** 2 for z in z2_samples) / (n_pairs - 1))
corr = cov / (std1 * std2)

print(f"  Correlation(Z1, Z2): {corr:.4f}  (theoretical: 0.0)")
```

### 10.5 Order Statistics: Min and Max of Uniform Samples

```python
import random

random.seed(99)
n_sim = 50_000
sample_size = 10

min_samples = []
max_samples = []

for _ in range(n_sim):
    data = [random.random() for _ in range(sample_size)]
    min_samples.append(min(data))
    max_samples.append(max(data))

# X_(1) ~ Beta(1, n) => mean = 1/(n+1)
# X_(n) ~ Beta(n, 1) => mean = n/(n+1)
min_mean = sum(min_samples) / n_sim
max_mean = sum(max_samples) / n_sim

k = sample_size
print(f"Order statistics for Uniform(0,1), sample size = {k}:")
print(f"  E[X_(1)]:  {min_mean:.4f}  (theoretical: {1/(k+1):.4f})")
print(f"  E[X_({k})]: {max_mean:.4f}  (theoretical: {k/(k+1):.4f})")
```

---

## Key Takeaways

1. The **CDF technique** is the most general approach: compute $F_Y(y) = P(g(X) \le y)$ and differentiate. It works for any transformation.
2. For **monotonic** transformations, the Jacobian formula $f_Y(y) = f_X(g^{-1}(y)) \cdot |dx/dy|$ provides a direct shortcut.
3. The **convolution** formula gives the PDF of a sum of independent random variables; the **MGF technique** often provides a more elegant route when the MGF is recognisable.
4. **Order statistics** have explicit density formulas; for uniforms, order statistics follow Beta distributions.
5. The **Box-Muller transform** converts two independent uniforms into two independent normals, combining the inverse-CDF idea with a polar coordinate decomposition.
6. Choosing the right technique depends on the problem: use CDF for general functions, Jacobian for monotonic maps, MGF for sums with known generating functions.

---

*Next lesson: [Multivariate Normal Distribution](./09_Multivariate_Normal_Distribution.md)*

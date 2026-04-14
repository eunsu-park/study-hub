# Lesson 8: Information Theory

## Learning Objectives

- Define Shannon entropy and interpret it as the expected surprise of a distribution
- Compute cross-entropy and understand why it is the standard classification loss
- Define KL divergence and prove its non-negativity (Gibbs' inequality)
- Relate cross-entropy, entropy, and KL divergence through the decomposition $H(p, q) = H(p) + D_\text{KL}(p \| q)$
- Apply mutual information to measure dependency between variables
- Use information-theoretic quantities to analyze DL models (information bottleneck, VAE ELBO)
- Compute entropy, cross-entropy, and KL divergence numerically for discrete and continuous distributions

---

## 1. Shannon Entropy

### 1.1 Intuition: Surprise and Uncertainty

Consider a random variable $X$ with possible outcomes. How "surprised" are we when we observe outcome $x$?

- If $P(X = x) = 1$: no surprise (it was certain)
- If $P(X = x) \approx 0$: huge surprise (very unlikely event)

A natural measure of surprise is $-\log P(x)$ (using base 2 for bits, base $e$ for nats).

### 1.2 Definition

The **Shannon entropy** of a discrete distribution $p$ is the expected surprise:

$$H(p) = -\sum_{x} p(x) \log p(x) = \mathbb{E}_{x \sim p}[-\log p(x)]$$

For continuous distributions, we use **differential entropy**:

$$h(p) = -\int p(x) \log p(x) \, dx$$

### 1.3 Properties

1. $H(p) \geq 0$ (non-negative for discrete distributions)
2. $H(p) = 0$ iff $p$ is deterministic (one outcome has probability 1)
3. $H(p) \leq \log K$ for $K$ outcomes, with equality iff $p$ is uniform
4. Adding more possible outcomes can only increase entropy

### 1.4 Examples

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    """Shannon entropy in nats."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # Ignore zero probabilities
    return -np.sum(p * np.log(p))

# Entropy of a binary distribution
ps = np.linspace(0.001, 0.999, 500)
H_binary = -ps * np.log(ps) - (1 - ps) * np.log(1 - ps)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ps, H_binary, 'b-', linewidth=2)
axes[0].set_xlabel('$p$')
axes[0].set_ylabel('$H(p)$')
axes[0].set_title('Entropy of Bernoulli($p$)')
axes[0].axhline(y=np.log(2), color='r', linestyle='--', label='$\ln 2$ (max)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Entropy of various distributions
distributions = {
    'Deterministic': np.array([1, 0, 0, 0]),
    'Peaked': np.array([0.9, 0.05, 0.03, 0.02]),
    'Moderate': np.array([0.5, 0.25, 0.15, 0.10]),
    'Uniform': np.array([0.25, 0.25, 0.25, 0.25]),
}

names = []
entropies = []
for name, p in distributions.items():
    H = entropy(p)
    names.append(name)
    entropies.append(H)
    print(f"{name:15s}: H = {H:.4f} nats, p = {p}")

axes[1].bar(names, entropies, color=['red', 'orange', 'blue', 'green'])
axes[1].set_ylabel('Entropy (nats)')
axes[1].set_title('Entropy of different distributions')
axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### 1.5 Entropy of a Gaussian

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$$

The Gaussian has the **maximum entropy** among all distributions with a given mean and variance. This is one reason why the Gaussian assumption is so common -- it is the "least informative" distribution given first and second moments.

---

## 2. Cross-Entropy

### 2.1 Definition

The **cross-entropy** between a true distribution $p$ and a model distribution $q$:

$$H(p, q) = -\sum_{x} p(x) \log q(x) = \mathbb{E}_{x \sim p}[-\log q(x)]$$

**Interpretation**: The expected number of nats needed to encode samples from $p$ using an encoding optimized for $q$.

### 2.2 Cross-Entropy as a Loss Function

In classification, $p$ is the true label distribution (one-hot) and $q = \hat{\boldsymbol{\pi}}$ is the model's predicted distribution:

$$H(p, q) = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k$$

Since $\mathbf{y}$ is one-hot with $y_c = 1$:

$$H(p, q) = -\log \hat{\pi}_c$$

This is exactly the categorical cross-entropy loss.

### 2.3 The Decomposition

$$\boxed{H(p, q) = H(p) + D_\text{KL}(p \| q)}$$

Since $H(p)$ is constant w.r.t. the model parameters, **minimizing cross-entropy is equivalent to minimizing KL divergence**. The model learns to match $q$ to $p$.

```python
def cross_entropy(p, q):
    """Cross-entropy H(p, q) in nats."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask] + 1e-10))

def kl_divergence(p, q):
    """KL divergence D_KL(p || q) in nats."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10)))

# Verify the decomposition: H(p, q) = H(p) + D_KL(p || q)
p = np.array([0.6, 0.3, 0.1])
q = np.array([0.4, 0.4, 0.2])

H_p = entropy(p)
H_pq = cross_entropy(p, q)
D_KL = kl_divergence(p, q)

print(f"H(p) = {H_p:.4f}")
print(f"H(p, q) = {H_pq:.4f}")
print(f"D_KL(p || q) = {D_KL:.4f}")
print(f"H(p) + D_KL(p||q) = {H_p + D_KL:.4f}")
print(f"Match: {np.isclose(H_pq, H_p + D_KL)}")
```

---

## 3. KL Divergence

### 3.1 Definition

$$D_\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

### 3.2 Properties

1. **Non-negativity** (Gibbs' inequality): $D_\text{KL}(p \| q) \geq 0$, with equality iff $p = q$
2. **Asymmetry**: $D_\text{KL}(p \| q) \neq D_\text{KL}(q \| p)$ in general
3. **Not a metric**: violates symmetry and triangle inequality

### 3.3 Proof of Non-Negativity

Using Jensen's inequality (since $-\log$ is convex):

$$D_\text{KL}(p \| q) = -\mathbb{E}_{x \sim p}\left[\log \frac{q(x)}{p(x)}\right] \geq -\log \mathbb{E}_{x \sim p}\left[\frac{q(x)}{p(x)}\right] = -\log \sum_x q(x) = -\log 1 = 0$$

Equality holds iff $q(x)/p(x)$ is constant a.s. under $p$, i.e., $p = q$.

### 3.4 Forward KL vs. Reverse KL

The choice of direction matters enormously in practice:

**Forward KL** $D_\text{KL}(p \| q)$ (mean-seeking):
- Penalizes $q$ heavily when $q(x) \approx 0$ but $p(x) > 0$
- The approximation $q$ tends to cover all modes of $p$ ("mean-seeking")
- Used in: supervised learning, variational inference (expectation over true posterior)

**Reverse KL** $D_\text{KL}(q \| p)$ (mode-seeking):
- Penalizes $q$ when $q(x) > 0$ but $p(x) \approx 0$
- The approximation $q$ tends to concentrate on one mode of $p$ ("mode-seeking")
- Used in: VAEs, policy gradient RL

```python
# Visualize forward vs reverse KL
x = np.linspace(-5, 8, 1000)

# True distribution: mixture of Gaussians
def p_true(x):
    return 0.5 * np.exp(-0.5*(x - 0)**2) / np.sqrt(2*np.pi) + \
           0.5 * np.exp(-0.5*(x - 4)**2) / np.sqrt(2*np.pi)

# Approximation: single Gaussian
def q_approx(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Forward KL minimizer (covers both modes)
axes[0].plot(x, p_true(x), 'b-', linewidth=2, label='$p$ (true)')
axes[0].plot(x, q_approx(x, 2.0, 2.5), 'r--', linewidth=2, label='$q$ (forward KL)')
axes[0].set_title('Forward KL: mean-seeking\n$q$ covers both modes')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Reverse KL minimizer (locks onto one mode)
axes[1].plot(x, p_true(x), 'b-', linewidth=2, label='$p$ (true)')
axes[1].plot(x, q_approx(x, 0.0, 1.0), 'r--', linewidth=2, label='$q$ (reverse KL)')
axes[1].set_title('Reverse KL: mode-seeking\n$q$ locks onto one mode')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. Mutual Information

### 4.1 Definition

The **mutual information** between random variables $X$ and $Y$:

$$I(X; Y) = D_\text{KL}(p(x, y) \| p(x)p(y)) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$

### 4.2 Properties

- $I(X; Y) \geq 0$, with equality iff $X$ and $Y$ are independent
- $I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$
- $I(X; Y) = H(X) + H(Y) - H(X, Y)$
- Symmetric: $I(X; Y) = I(Y; X)$

### 4.3 Mutual Information in DL

**Information Bottleneck** (Tishby et al.): A neural network layer $T$ should maximize $I(T; Y)$ (preserving label information) while minimizing $I(T; X)$ (compressing input information).

**Representation learning**: Good representations have high mutual information with the task label.

```python
# Compute mutual information for a simple joint distribution
# Joint distribution p(x, y)
joint = np.array([
    [0.1, 0.05, 0.01],
    [0.05, 0.2, 0.05],
    [0.01, 0.05, 0.48],
])

# Marginals
p_x = joint.sum(axis=1)
p_y = joint.sum(axis=0)

# Mutual information
MI = 0
for i in range(joint.shape[0]):
    for j in range(joint.shape[1]):
        if joint[i, j] > 0:
            MI += joint[i, j] * np.log(joint[i, j] / (p_x[i] * p_y[j]))

H_X = entropy(p_x)
H_Y = entropy(p_y)
H_XY = -np.sum(joint[joint > 0] * np.log(joint[joint > 0]))

print(f"H(X) = {H_X:.4f}")
print(f"H(Y) = {H_Y:.4f}")
print(f"H(X,Y) = {H_XY:.4f}")
print(f"I(X;Y) = {MI:.4f}")
print(f"H(X) + H(Y) - H(X,Y) = {H_X + H_Y - H_XY:.4f}")
print(f"Match: {np.isclose(MI, H_X + H_Y - H_XY)}")
```

---

## 5. Information Theory in VAEs

### 5.1 The Evidence Lower Bound (ELBO)

For a generative model with latent variables $\mathbf{z}$:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_\text{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

The left side is the log-evidence (intractable). The right side is the **ELBO**:

- **Reconstruction term**: $\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})]$ -- how well the decoder reconstructs $\mathbf{x}$
- **KL term**: $D_\text{KL}(q \| p)$ -- how close the encoder posterior is to the prior

### 5.2 ELBO Derivation

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} = \log \int \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} q(\mathbf{z}|\mathbf{x}) \, d\mathbf{z}$$

By Jensen's inequality:

$$\geq \int q(\mathbf{z}|\mathbf{x}) \log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \, d\mathbf{z} = \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - D_\text{KL}(q \| p)$$

The gap between $\log p(\mathbf{x})$ and the ELBO is exactly $D_\text{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$, which is non-negative.

### 5.3 Beta-VAE and Rate-Distortion

The $\beta$-VAE modifies the objective:

$$\mathcal{L}_\beta = \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - \beta \cdot D_\text{KL}(q \| p)$$

- $\beta > 1$: stronger compression, more disentangled latent space
- $\beta < 1$: better reconstruction, less regularized latent space
- $\beta = 1$: standard VAE (ELBO)

This connects to the **rate-distortion theory** in information theory.

---

## 6. Label Smoothing: An Information-Theoretic View

### 6.1 Standard Hard Labels

One-hot labels $\mathbf{y}$ have zero entropy: $H(\mathbf{y}) = 0$. The model is pushed to output $\hat{\pi}_c = 1$ (zero cross-entropy), which leads to overconfident predictions and large logit magnitudes.

### 6.2 Smoothed Labels

Label smoothing replaces the one-hot with:

$$y_k^{\text{smooth}} = (1 - \alpha) y_k + \frac{\alpha}{K}$$

For the true class: $y_c = 1 - \alpha + \alpha/K$. For others: $y_k = \alpha/K$.

Now $H(\mathbf{y}^{\text{smooth}}) > 0$, so perfect cross-entropy requires less extreme logits. The model cannot drive its predictions to infinity.

```python
# Compare cross-entropy loss with hard vs smooth labels
K = 10
alpha = 0.1

# Hard label: class 3
y_hard = np.zeros(K)
y_hard[3] = 1.0

# Smooth label
y_smooth = np.full(K, alpha / K)
y_smooth[3] = 1 - alpha + alpha / K

print(f"Hard label:   {y_hard}")
print(f"Smooth label: {y_smooth.round(4)}")
print(f"H(hard) = {entropy(y_hard):.4f}")
print(f"H(smooth) = {entropy(y_smooth):.4f}")

# Compare losses at different confidence levels
confidences = np.linspace(0.1, 0.99, 100)
losses_hard = []
losses_smooth = []
for conf in confidences:
    q = np.full(K, (1 - conf) / (K - 1))
    q[3] = conf
    losses_hard.append(cross_entropy(y_hard, q))
    losses_smooth.append(cross_entropy(y_smooth, q))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(confidences, losses_hard, 'b-', linewidth=2, label='Hard labels')
ax.plot(confidences, losses_smooth, 'r--', linewidth=2, label=f'Smooth labels (α={alpha})')
ax.set_xlabel('Model confidence on true class')
ax.set_ylabel('Cross-entropy loss')
ax.set_title('Label smoothing effect on loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Connections and Summary Table

| Quantity | Formula | DL Usage |
|---------|---------|----------|
| Entropy $H(p)$ | $-\sum p \log p$ | Measures uncertainty; max-entropy regularization |
| Cross-entropy $H(p, q)$ | $-\sum p \log q$ | Standard classification loss |
| KL divergence $D_\text{KL}$ | $\sum p \log(p/q)$ | VAE loss, knowledge distillation, policy gradient |
| Mutual information $I$ | $D_\text{KL}(p_{xy} \| p_x p_y)$ | Information bottleneck, representation learning |
| ELBO | $\mathbb{E}_q[\log p(x|z)] - D_\text{KL}(q\|p)$ | VAE training objective |

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Entropy | Expected surprise; maximized by uniform distribution |
| Cross-entropy | Expected code length under wrong model; = classification loss |
| Decomposition | $H(p, q) = H(p) + D_\text{KL}(p\|q)$; minimizing CE = minimizing KL |
| KL non-negativity | Proved via Jensen's inequality |
| Forward vs. reverse KL | Mean-seeking vs. mode-seeking behavior |
| Mutual information | Symmetric measure of dependency; $I = H(X) + H(Y) - H(X,Y)$ |
| ELBO | Lower bound on log-evidence; reconstruction - KL |

---

## Exercises

1. Prove that the uniform distribution maximizes entropy over $K$ outcomes using Lagrange multipliers.
2. Compute $D_\text{KL}(\text{Bernoulli}(p) \| \text{Bernoulli}(q))$ analytically and plot it as a function of $q$ for $p = 0.7$.
3. Implement the ELBO loss for a VAE with Gaussian encoder and decoder, and show that it upper-bounds the negative log-likelihood.
4. Compute the mutual information between two jointly Gaussian variables with correlation $\rho$ and plot $I$ vs. $\rho$.
5. Implement label smoothing and compare training dynamics (loss, accuracy, calibration) with hard labels on a simple classification problem.

---

**Next**: [09. Matrix Decompositions](09_Matrix_Decompositions.md)

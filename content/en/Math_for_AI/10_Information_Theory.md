# 10. Information Theory

## Learning Objectives

- Understand the concepts of information content and entropy and grasp their role as measures of uncertainty
- Learn the definitions and properties of cross-entropy and KL divergence and understand their applications in machine learning
- Understand mutual information and learn methods to measure dependencies between variables
- Use Jensen's inequality to prove key inequalities in information theory
- Learn how information theory is utilized in generative models such as VAE and GAN
- Compute and visualize entropy, KL divergence, and mutual information using Python

---

## 1. Information and Entropy

### 1.1 Self-Information

The **information content** when event $x$ occurs:

$$I(x) = -\log P(x) = \log \frac{1}{P(x)}$$

**Intuition**:
- Low probability event → high information (surprise)
- High probability event → low information

**Units**:
- $\log_2$: bits
- $\log_e$: nats

**Examples**:
- Coin toss (head probability 0.5): $I(\text{H}) = -\log_2(0.5) = 1$ bit
- Dice (each face probability 1/6): $I(1) = -\log_2(1/6) \approx 2.58$ bits

### 1.2 Shannon Entropy

The **entropy** of random variable $X$ is the average information content:

$$H(X) = -\sum_{x} P(x) \log P(x) = \mathbb{E}_{P(x)}[-\log P(x)]$$

For continuous random variables (differential entropy):

$$h(X) = -\int p(x) \log p(x) dx$$

**Properties**:
1. $H(X) \geq 0$ (non-negativity)
2. $H(X) = 0 \Leftrightarrow X$ is deterministic (only one event with probability 1)
3. Maximum entropy: Uniform distribution

### 1.3 Meaning of Entropy

Entropy can be interpreted as:

1. **Measure of uncertainty**: How uncertain is the distribution?
2. **Average information content**: Expected information when sampling
3. **Minimum encoding length**: Minimum number of bits needed to compress data

**Example**:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def entropy(p):
    """Compute entropy (treat 0*log(0) = 0)"""
    p = np.array(p)
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

# Entropy of binary distribution
p_range = np.linspace(0.01, 0.99, 100)
H_binary = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in p_range]

plt.figure(figsize=(14, 4))

# Binary entropy
plt.subplot(131)
plt.plot(p_range, H_binary, linewidth=2)
plt.axhline(1, color='r', linestyle='--', alpha=0.5, label='Maximum H=1')
plt.axvline(0.5, color='g', linestyle='--', alpha=0.5, label='p=0.5')
plt.xlabel('p (probability)')
plt.ylabel('H(X) (bits)')
plt.title('Binary Entropy Function')
plt.legend()
plt.grid(True)

# Fair die vs biased die
fair_die = [1/6] * 6
biased_die = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]

H_fair = entropy(fair_die)
H_biased = entropy(biased_die)

plt.subplot(132)
x = np.arange(1, 7)
width = 0.35
plt.bar(x - width/2, fair_die, width, label=f'Fair (H={H_fair:.2f})', alpha=0.7)
plt.bar(x + width/2, biased_die, width, label=f'Biased (H={H_biased:.2f})', alpha=0.7)
plt.xlabel('Face')
plt.ylabel('Probability')
plt.title('Dice Distributions')
plt.legend()
plt.grid(True, axis='y')

# Entropy of various distributions
n_outcomes = 10
distributions = {
    'Uniform': np.ones(n_outcomes) / n_outcomes,
    'Peaked': stats.norm.pdf(np.arange(n_outcomes), 5, 1),
    'Very Peaked': stats.norm.pdf(np.arange(n_outcomes), 5, 0.5),
}

# Normalize
for key in distributions:
    distributions[key] /= distributions[key].sum()

plt.subplot(133)
x_pos = np.arange(len(distributions))
entropies = [entropy(distributions[key]) for key in distributions]
bars = plt.bar(x_pos, entropies, alpha=0.7)
plt.xticks(x_pos, distributions.keys())
plt.ylabel('Entropy (bits)')
plt.title('Entropy of Different Distributions')
plt.grid(True, axis='y')

# Display values on each bar
for i, (bar, h) in enumerate(zip(bars, entropies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{h:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('entropy_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print("Uniform distribution has maximum entropy!")
print(f"Fair die: {H_fair:.3f} bits")
print(f"Biased die: {H_biased:.3f} bits")
```

### 1.4 Maximum Entropy Distribution

**Theorem**: Without constraints, the distribution with maximum entropy among discrete distributions with $n$ outcomes is the **uniform distribution**.

$$H(X) \leq \log n$$

Equality holds when $P(x) = 1/n$ (for all $x$).

**Proof**: Use Lagrange multipliers.

**For continuous distributions**:
- No constraints → undefined (infinity)
- Fixed mean → Exponential distribution
- Fixed mean and variance → **Gaussian distribution** (maximum entropy!)

This is one reason Gaussians are common in nature.

## 2. Cross-Entropy

### 2.1 Definition

**Cross-entropy** is the average number of bits needed to encode data sampled from distribution $P$ using distribution $Q$:

$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = \mathbb{E}_{P(x)}[-\log Q(x)]$$

**Interpretation**:
- $P$: True distribution (data)
- $Q$: Model distribution (prediction)
- $H(P, Q)$: "Cost" of approximating $P$ with $Q$

**Properties**:
1. $H(P, Q) \geq H(P)$ (equality when $P = Q$)
2. Asymmetric: $H(P, Q) \neq H(Q, P)$ (in general)

### 2.2 Binary Cross-Entropy

Used in logistic regression:

$$H(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

- $y \in \{0, 1\}$: True label
- $\hat{y} \in [0, 1]$: Predicted probability

**Average over n samples**:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

### 2.3 Categorical Cross-Entropy

Used in multi-class classification:

$$H(P, Q) = -\sum_{k=1}^K P_k \log Q_k$$

**For one-hot encoded** case ($P_k = \delta_{k,c}$):

$$\mathcal{L}_{\text{CCE}} = -\log Q_c$$

where $c$ is the correct class.

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Binary cross-entropy visualization
y_true = np.array([0, 0, 1, 1])  # True labels
y_pred_range = np.linspace(0.01, 0.99, 100)

# Loss for each sample
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, y in enumerate(y_true):
    if y == 1:
        loss = -np.log(y_pred_range)
    else:
        loss = -np.log(1 - y_pred_range)

    axes[i].plot(y_pred_range, loss, linewidth=2)
    axes[i].set_xlabel('Predicted Probability')
    axes[i].set_ylabel('Loss')
    axes[i].set_title(f'True Label: {y}')
    axes[i].grid(True)
    axes[i].axvline(y, color='r', linestyle='--', alpha=0.5, label=f'Optimal: {y}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('binary_cross_entropy.png', dpi=150, bbox_inches='tight')
plt.show()

# PyTorch implementation verification
y_true_tensor = torch.tensor([0., 0., 1., 1.])
y_pred_tensor = torch.tensor([0.1, 0.2, 0.8, 0.9])

bce_loss = nn.BCELoss()
loss = bce_loss(y_pred_tensor, y_true_tensor)

# Manual computation
manual_loss = -(y_true_tensor * torch.log(y_pred_tensor) +
                (1 - y_true_tensor) * torch.log(1 - y_pred_tensor)).mean()

print(f"PyTorch BCE Loss: {loss.item():.4f}")
print(f"Manual BCE Loss: {manual_loss.item():.4f}")
print(f"Difference: {abs(loss - manual_loss).item():.10f}")

# Categorical cross-entropy
print("\nCategorical Cross-Entropy:")
y_true_cat = torch.tensor([2])  # Class 2 is correct
y_pred_logits = torch.tensor([[1.0, 2.0, 3.0, 1.5]])  # Logits

ce_loss = nn.CrossEntropyLoss()
loss_cat = ce_loss(y_pred_logits, y_true_cat)

# Manual computation
y_pred_softmax = torch.softmax(y_pred_logits, dim=1)
manual_loss_cat = -torch.log(y_pred_softmax[0, 2])

print(f"PyTorch CE Loss: {loss_cat.item():.4f}")
print(f"Manual CE Loss: {manual_loss_cat.item():.4f}")
print(f"Softmax probabilities: {y_pred_softmax.numpy()}")
```

## 3. KL Divergence (Kullback-Leibler Divergence)

### 3.1 Definition

**KL divergence** measures the "distance" between two distributions $P$ and $Q$:

$$D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{P(x)}\left[\log \frac{P(x)}{Q(x)}\right]$$

Continuous distributions:

$$D_{\text{KL}}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

### 3.2 Properties of KL Divergence

1. **Non-negativity**: $D_{\text{KL}}(P \| Q) \geq 0$
2. **Equality condition**: $D_{\text{KL}}(P \| Q) = 0 \Leftrightarrow P = Q$ (almost everywhere)
3. **Asymmetric**: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$ → not a distance metric!
4. **Relationship with cross-entropy**:
   $$D_{\text{KL}}(P \| Q) = H(P, Q) - H(P)$$

### 3.3 Proof of Non-negativity (Gibbs' Inequality)

Using **Jensen's inequality**: $-\log$ is a convex function, so

$$-\mathbb{E}[\log X] \geq -\log \mathbb{E}[X]$$

Application:

$$D_{\text{KL}}(P \| Q) = \mathbb{E}_{P(x)}\left[\log \frac{P(x)}{Q(x)}\right]$$

$$= -\mathbb{E}_{P(x)}\left[\log \frac{Q(x)}{P(x)}\right]$$

$$\geq -\log \mathbb{E}_{P(x)}\left[\frac{Q(x)}{P(x)}\right]$$

$$= -\log \sum_{x} P(x) \frac{Q(x)}{P(x)}$$

$$= -\log \sum_{x} Q(x) = -\log 1 = 0$$

### 3.4 Forward KL vs Reverse KL

**Forward KL**: $D_{\text{KL}}(P \| Q)$
- Forces $Q$ to cover $P$ (mode-covering)
- If $P(x) > 0$ then $Q(x) > 0$ must hold

**Reverse KL**: $D_{\text{KL}}(Q \| P)$
- $Q$ selects modes of $P$ (mode-seeking)
- $Q$ can focus on only one mode of $P$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Target: mixture of two Gaussians
np.random.seed(42)
x = np.linspace(-5, 10, 1000)

# Target: mixture of two Gaussians
p = 0.5 * stats.norm.pdf(x, 0, 1) + 0.5 * stats.norm.pdf(x, 5, 1)
p = p / np.trapz(p, x)  # Normalize

# Approximate with Forward KL (mode-covering)
def kl_divergence_forward(mu, sigma, x, p_true):
    q = stats.norm.pdf(x, mu, sigma)
    q = q / np.trapz(q, x)
    # Forward KL: E_p[log(p/q)]
    kl = np.trapz(p_true * np.log((p_true + 1e-10) / (q + 1e-10)), x)
    return kl

# Approximate with Reverse KL (mode-seeking)
def kl_divergence_reverse(mu, sigma, x, p_true):
    q = stats.norm.pdf(x, mu, sigma)
    q = q / np.trapz(q, x)
    # Reverse KL: E_q[log(q/p)]
    kl = np.trapz(q * np.log((q + 1e-10) / (p_true + 1e-10)), x)
    return kl

# Grid search optimization (in practice, use gradient descent)
mu_range = np.linspace(-2, 7, 50)
sigma_range = np.linspace(0.5, 3, 30)

best_forward_kl = float('inf')
best_reverse_kl = float('inf')
best_forward_params = None
best_reverse_params = None

for mu in mu_range:
    for sigma in sigma_range:
        fkl = kl_divergence_forward(mu, sigma, x, p)
        rkl = kl_divergence_reverse(mu, sigma, x, p)

        if fkl < best_forward_kl:
            best_forward_kl = fkl
            best_forward_params = (mu, sigma)

        if rkl < best_reverse_kl:
            best_reverse_kl = rkl
            best_reverse_params = (mu, sigma)

# Optimal distributions
q_forward = stats.norm.pdf(x, *best_forward_params)
q_forward = q_forward / np.trapz(q_forward, x)

q_reverse = stats.norm.pdf(x, *best_reverse_params)
q_reverse = q_reverse / np.trapz(q_reverse, x)

# Visualization
plt.figure(figsize=(14, 5))

plt.subplot(121)
plt.plot(x, p, 'k-', linewidth=2, label='True P (bimodal)')
plt.plot(x, q_forward, 'b--', linewidth=2,
         label=f'Forward KL Q (μ={best_forward_params[0]:.2f}, σ={best_forward_params[1]:.2f})')
plt.fill_between(x, 0, p, alpha=0.2, color='k')
plt.fill_between(x, 0, q_forward, alpha=0.2, color='b')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Forward KL: D(P||Q) (mode-covering)\nKL = {best_forward_kl:.4f}')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.plot(x, p, 'k-', linewidth=2, label='True P (bimodal)')
plt.plot(x, q_reverse, 'r--', linewidth=2,
         label=f'Reverse KL Q (μ={best_reverse_params[0]:.2f}, σ={best_reverse_params[1]:.2f})')
plt.fill_between(x, 0, p, alpha=0.2, color='k')
plt.fill_between(x, 0, q_reverse, alpha=0.2, color='r')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Reverse KL: D(Q||P) (mode-seeking)\nKL = {best_reverse_kl:.4f}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('forward_vs_reverse_kl.png', dpi=150, bbox_inches='tight')
plt.show()

print("Forward KL (mode-covering): Spreads broadly between both modes")
print(f"  Best params: μ={best_forward_params[0]:.2f}, σ={best_forward_params[1]:.2f}")
print("\nReverse KL (mode-seeking): Selects one mode")
print(f"  Best params: μ={best_reverse_params[0]:.2f}, σ={best_reverse_params[1]:.2f}")
```

## 4. Mutual Information

### 4.1 Definition

**Mutual information** between two random variables $X$ and $Y$:

$$I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x)P(y)}$$

$$= D_{\text{KL}}(P(X, Y) \| P(X)P(Y))$$

**Alternative expressions**:

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$= H(X) + H(Y) - H(X, Y)$$

### 4.2 Conditional Entropy

**Conditional entropy** of $X$ given $Y$:

$$H(X|Y) = \sum_{y} P(y) H(X|Y=y)$$

$$= -\sum_{x, y} P(x, y) \log P(x|y)$$

**Chain rule**:

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

### 4.3 Properties of Mutual Information

1. **Non-negativity**: $I(X; Y) \geq 0$
2. **Symmetry**: $I(X; Y) = I(Y; X)$
3. **Independence**: $I(X; Y) = 0 \Leftrightarrow X \perp Y$
4. **Upper bound**: $I(X; Y) \leq \min(H(X), H(Y))$

**Intuition**:
- $I(X; Y)$: How much does knowing $X$ reduce uncertainty about $Y$?
- $= H(Y) - H(Y|X)$: Entropy of $Y$ minus conditional entropy of $Y$ given $X$

### 4.4 Application: Feature Selection

In machine learning, if mutual information between feature $X$ and target $Y$ is high, $X$ is a useful feature.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, n_repeated=0, random_state=42)

# Compute mutual information
mi = mutual_info_classif(X, y, random_state=42)

# Visualization
plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.bar(range(len(mi)), mi)
plt.xlabel('Feature Index')
plt.ylabel('Mutual Information')
plt.title('Mutual Information with Target')
plt.grid(True, axis='y')

# Features with high vs low mutual information
high_mi_idx = np.argmax(mi)
low_mi_idx = np.argmin(mi)

plt.subplot(132)
plt.scatter(X[:, high_mi_idx], y, alpha=0.5)
plt.xlabel(f'Feature {high_mi_idx}')
plt.ylabel('Target')
plt.title(f'High MI Feature (MI={mi[high_mi_idx]:.3f})')
plt.grid(True)

plt.subplot(133)
plt.scatter(X[:, low_mi_idx], y, alpha=0.5)
plt.xlabel(f'Feature {low_mi_idx}')
plt.ylabel('Target')
plt.title(f'Low MI Feature (MI={mi[low_mi_idx]:.3f})')
plt.grid(True)

plt.tight_layout()
plt.savefig('mutual_information.png', dpi=150, bbox_inches='tight')
plt.show()

print("Features with high mutual information:")
top_features = np.argsort(mi)[-5:][::-1]
for idx in top_features:
    print(f"  Feature {idx}: MI = {mi[idx]:.4f}")

print("\nFeatures with low mutual information:")
bottom_features = np.argsort(mi)[:5]
for idx in bottom_features:
    print(f"  Feature {idx}: MI = {mi[idx]:.4f}")
```

## 5. Jensen's Inequality

### 5.1 Definition

If function $f$ is **convex**:

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

If function $f$ is **concave**:

$$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$$

**Discrete form**:

$$f\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i)$$

where $\lambda_i \geq 0, \sum_i \lambda_i = 1$.

### 5.2 Examples of Convex Functions

- $f(x) = x^2$
- $f(x) = e^x$
- $f(x) = -\log x$ (for $x > 0$)

### 5.3 Application: Non-negativity of KL Divergence

$$D_{\text{KL}}(P \| Q) = \mathbb{E}_{P}\left[\log \frac{P(x)}{Q(x)}\right]$$

$$= -\mathbb{E}_{P}\left[\log \frac{Q(x)}{P(x)}\right]$$

$-\log$ is convex, so Jensen's inequality:

$$-\mathbb{E}_{P}\left[\log \frac{Q(x)}{P(x)}\right] \geq -\log \mathbb{E}_{P}\left[\frac{Q(x)}{P(x)}\right]$$

$$= -\log \sum_{x} P(x) \frac{Q(x)}{P(x)} = -\log \sum_{x} Q(x) = 0$$

### 5.4 Application: ELBO Derivation (VAE)

**Evidence Lower BOund (ELBO)** in VAE:

$$\log P(x) = \log \int P(x, z) dz = \log \int Q(z) \frac{P(x, z)}{Q(z)} dz$$

Jensen ($\log$ is concave):

$$\geq \int Q(z) \log \frac{P(x, z)}{Q(z)} dz$$

$$= \mathbb{E}_{Q(z)}[\log P(x, z)] - \mathbb{E}_{Q(z)}[\log Q(z)]$$

$$= \mathbb{E}_{Q(z)}[\log P(x|z)] - D_{\text{KL}}(Q(z) \| P(z))$$

This is the loss function for VAE!

```python
import numpy as np
import matplotlib.pyplot as plt

# Jensen's inequality visualization
np.random.seed(42)

# Convex function: f(x) = x^2
x_values = np.linspace(-2, 2, 100)
f_convex = x_values**2

# Samples
samples = np.array([-1.5, -0.5, 0.5, 1.5])
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Expectation
E_x = np.sum(weights * samples)
f_E_x = E_x**2

# Expectation of f(x)
E_f_x = np.sum(weights * samples**2)

plt.figure(figsize=(14, 5))

# Convex function
plt.subplot(121)
plt.plot(x_values, f_convex, 'b-', linewidth=2, label='f(x) = x²')
plt.scatter(samples, samples**2, color='r', s=100, zorder=5,
            label='Sample points')

# E[X]
plt.axvline(E_x, color='g', linestyle='--', alpha=0.7,
            label=f'E[X] = {E_x:.2f}')
plt.scatter([E_x], [f_E_x], color='g', s=200, marker='s',
            zorder=5, label=f'f(E[X]) = {f_E_x:.2f}')

# E[f(X)]
plt.axhline(E_f_x, color='orange', linestyle='--', alpha=0.7,
            label=f'E[f(X)] = {E_f_x:.2f}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Jensen\'s Inequality (Convex)\nf(E[X]) ≤ E[f(X)]')
plt.legend()
plt.grid(True)

# Concave function: g(x) = log(x)
x_values_pos = np.linspace(0.1, 3, 100)
g_concave = np.log(x_values_pos)

samples_pos = np.array([0.5, 1.0, 1.5, 2.0])
E_x_pos = np.sum(weights * samples_pos)
g_E_x = np.log(E_x_pos)
E_g_x = np.sum(weights * np.log(samples_pos))

plt.subplot(122)
plt.plot(x_values_pos, g_concave, 'b-', linewidth=2, label='g(x) = log(x)')
plt.scatter(samples_pos, np.log(samples_pos), color='r', s=100,
            zorder=5, label='Sample points')

plt.axvline(E_x_pos, color='g', linestyle='--', alpha=0.7,
            label=f'E[X] = {E_x_pos:.2f}')
plt.scatter([E_x_pos], [g_E_x], color='g', s=200, marker='s',
            zorder=5, label=f'g(E[X]) = {g_E_x:.2f}')

plt.axhline(E_g_x, color='orange', linestyle='--', alpha=0.7,
            label=f'E[g(X)] = {E_g_x:.2f}')

plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Jensen\'s Inequality (Concave)\ng(E[X]) ≥ E[g(X)]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('jensen_inequality.png', dpi=150, bbox_inches='tight')
plt.show()

print("Convex (f(x) = x²):")
print(f"  f(E[X]) = {f_E_x:.4f}")
print(f"  E[f(X)] = {E_f_x:.4f}")
print(f"  f(E[X]) ≤ E[f(X)]: {f_E_x <= E_f_x}")

print("\nConcave (g(x) = log(x)):")
print(f"  g(E[X]) = {g_E_x:.4f}")
print(f"  E[g(X)] = {E_g_x:.4f}")
print(f"  g(E[X]) ≥ E[g(X)]: {g_E_x >= E_g_x}")
```

## 6. Information Theory in Machine Learning

### 6.1 Cross-Entropy Loss = MLE

Minimizing cross-entropy in classification = Minimizing negative log-likelihood:

$$\min_{\theta} H(P_{\text{data}}, P_{\theta}) = \min_{\theta} -\mathbb{E}_{(x,y) \sim P_{\text{data}}}[\log P_{\theta}(y|x)]$$

This is equivalent to MLE!

### 6.2 VAE's ELBO

**Variational Autoencoder** objective function:

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

- First term: Reconstruction loss
- Second term: KL regularization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross-Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence: D_KL(q(z|x) || p(z))
    # Since p(z) = N(0, I), can be computed analytically
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

# Simple test
model = VAE(input_dim=784, latent_dim=20)
x = torch.randn(32, 784)  # Batch size 32

recon_x, mu, logvar = model(x)
loss, bce, kld = vae_loss(recon_x, x, mu, logvar)

print(f"Total Loss: {loss.item():.2f}")
print(f"Reconstruction Loss (BCE): {bce.item():.2f}")
print(f"KL Divergence: {kld.item():.2f}")
print(f"\nELBO = -Loss = {-loss.item():.2f}")
```

### 6.3 GAN's JS Divergence

The original objective of **Generative Adversarial Network** is related to **Jensen-Shannon (JS) divergence**:

$$D_{\text{JS}}(P \| Q) = \frac{1}{2}D_{\text{KL}}(P \| M) + \frac{1}{2}D_{\text{KL}}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$.

**GAN's optimal discriminator** estimates the JS divergence.

### 6.4 Information Bottleneck

**Information bottleneck theory** finds a compressed representation $Z$ between input $X$ and target $Y$:

$$\min_{p(z|x)} I(X; Z) - \beta I(Z; Y)$$

- $I(X; Z)$: Minimize → compression
- $I(Z; Y)$: Maximize → preserve information
- $\beta$: trade-off parameter

This is used for theoretical understanding of deep learning.

```python
import numpy as np
import matplotlib.pyplot as plt

# JS divergence visualization
def js_divergence(p, q):
    """Jensen-Shannon divergence"""
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
    return 0.5 * kl_pm + 0.5 * kl_qm

# Two distributions
n_bins = 10
p = np.random.dirichlet(np.ones(n_bins))
q = np.random.dirichlet(np.ones(n_bins))

# KL vs JS
kl_pq = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
kl_qp = np.sum(q * np.log((q + 1e-10) / (p + 1e-10)))
js_pq = js_divergence(p, q)

plt.figure(figsize=(14, 4))

plt.subplot(131)
x = np.arange(n_bins)
width = 0.35
plt.bar(x - width/2, p, width, label='P', alpha=0.7)
plt.bar(x + width/2, q, width, label='Q', alpha=0.7)
plt.xlabel('Bin')
plt.ylabel('Probability')
plt.title('Two Distributions')
plt.legend()
plt.grid(True, axis='y')

plt.subplot(132)
divergences = [kl_pq, kl_qp, js_pq]
labels = ['KL(P||Q)', 'KL(Q||P)', 'JS(P||Q)']
colors = ['blue', 'red', 'green']
bars = plt.bar(labels, divergences, color=colors, alpha=0.7)
plt.ylabel('Divergence')
plt.title('Divergence Comparison')
plt.grid(True, axis='y')

for bar, val in zip(bars, divergences):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom')

# Symmetry of JS divergence
plt.subplot(133)
# For various distribution pairs
n_tests = 20
kl_diffs = []
js_vals = []

for _ in range(n_tests):
    p_test = np.random.dirichlet(np.ones(n_bins))
    q_test = np.random.dirichlet(np.ones(n_bins))

    kl_pq_test = np.sum(p_test * np.log((p_test + 1e-10) / (q_test + 1e-10)))
    kl_qp_test = np.sum(q_test * np.log((q_test + 1e-10) / (p_test + 1e-10)))
    js_test = js_divergence(p_test, q_test)

    kl_diffs.append(abs(kl_pq_test - kl_qp_test))
    js_vals.append(js_test)

plt.scatter(kl_diffs, js_vals, alpha=0.7)
plt.xlabel('|KL(P||Q) - KL(Q||P)|')
plt.ylabel('JS(P||Q)')
plt.title('JS is Symmetric, KL is Not')
plt.grid(True)

plt.tight_layout()
plt.savefig('js_divergence.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"KL(P||Q) = {kl_pq:.4f}")
print(f"KL(Q||P) = {kl_qp:.4f}")
print(f"JS(P||Q) = {js_pq:.4f}")
print(f"\nKL is asymmetric, JS is symmetric!")
print(f"JS(P||Q) = JS(Q||P) always holds")
```

## Practice Problems

### Problem 1: Maximum Entropy Proof

Using Lagrange multipliers, prove that under constraint $\sum_{i=1}^n p_i = 1$, the distribution that maximizes entropy $H = -\sum_{i=1}^n p_i \log p_i$ is the uniform distribution $p_i = 1/n$.

**Hint**:
- Lagrangian: $L = -\sum_i p_i \log p_i - \lambda(\sum_i p_i - 1)$
- Solve $\frac{\partial L}{\partial p_i} = 0$

### Problem 2: Conditional Entropy and Mutual Information

Prove the following:

(a) $H(X, Y) = H(X) + H(Y|X)$ (chain rule)

(b) $I(X; Y) = H(X) + H(Y) - H(X, Y)$

(c) $I(X; Y) \leq \min(H(X), H(Y))$

(d) $I(X; Y) = 0 \Leftrightarrow X \perp Y$

### Problem 3: KL Divergence Calculation

Analytically compute the KL divergence between two Gaussian distributions $P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$.

**Result**:
$$D_{\text{KL}}(P \| Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

Verify with Python (numerical integration vs analytical solution).

### Problem 4: KL Divergence in VAE

In VAE, when $q(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$, derive that the KL divergence is:

$$D_{\text{KL}}(q(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

where $d$ is the dimension of the latent space.

### Problem 5: Information Bottleneck Simulation

Implement the information bottleneck principle for a simple dataset (e.g., 2D classification problem):

(a) Define input $X$, compressed representation $Z$, target $Y$

(b) Estimate $I(X; Z)$ and $I(Z; Y)$ (histogram-based)

(c) Plot trade-off curve for various $\beta$ values

(d) Find optimal compression level

## References

1. **Cover, T. M., & Thomas, J. A. (2006).** *Elements of Information Theory* (2nd ed.). Wiley. [The bible of information theory]
2. **MacKay, D. J. C. (2003).** *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
3. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: An Introduction*. Chapter 6 (Information Theory).
4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. Chapter 3 (Information Theory).
5. **Paper**: Tishby, N., & Zaslavsky, N. (2015). "Deep Learning and the Information Bottleneck Principle". *IEEE Information Theory Workshop*.
6. **Paper**: Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes". *ICLR*.
7. **Tutorial**: Shwartz-Ziv, R., & Tishby, N. (2017). "Opening the Black Box of Deep Neural Networks via Information". *arXiv:1703.00810*.
8. **Blog**: Colah's Blog - Visual Information Theory - https://colah.github.io/posts/2015-09-Visual-Information/

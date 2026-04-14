# Lesson 11: Attention and Softmax Math

## Learning Objectives

- Derive the scaled dot-product attention formula from first principles
- Explain why attention scores are scaled by $1/\sqrt{d_k}$
- Analyze the gradient flow through attention mechanisms
- Understand softmax temperature and its effect on attention distributions
- Derive the Jacobian of softmax and its implications for gradient computation
- Understand multi-head attention as parallel subspace projections
- Analyze the computational complexity of attention and motivations for efficient variants
- Connect attention to kernel methods and probabilistic interpretation

---

## 1. Dot-Product Attention

### 1.1 Queries, Keys, and Values

Given a sequence of $T$ input vectors $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \mathbb{R}^{d_\text{model}}$, attention computes:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

where $\mathbf{X} \in \mathbb{R}^{T \times d_\text{model}}$ and $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_\text{model} \times d_k}$, $\mathbf{W}_V \in \mathbb{R}^{d_\text{model} \times d_v}$.

**Intuition**:
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "What information do I provide?"

### 1.2 Attention Weights

The attention weight from position $i$ to position $j$ measures how relevant position $j$ is to position $i$:

$$\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_{l=1}^{T} \exp(\mathbf{q}_i^\top \mathbf{k}_l / \sqrt{d_k})}$$

In matrix form:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Q: (T_q, d_k)
    K: (T_k, d_k)
    V: (T_k, d_v)
    Returns: (T_q, d_v), attention weights (T_q, T_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (T_q, T_k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Stable softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    output = attn_weights @ V  # (T_q, d_v)
    return output, attn_weights

# Example
T, d_k, d_v = 6, 8, 8
np.random.seed(42)
Q = np.random.randn(T, d_k)
K = np.random.randn(T, d_k)
V = np.random.randn(T, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Q shape: {Q.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights sum per row: {weights.sum(axis=-1).round(4)}")

# Visualize attention weights
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(weights, cmap='Blues')
ax.set_xlabel('Key position')
ax.set_ylabel('Query position')
ax.set_title('Attention weights')
plt.colorbar(im)
plt.show()
```

---

## 2. Why Scale by $1/\sqrt{d_k}$?

### 2.1 The Variance Argument

Assume $q_i$ and $k_j$ are independent random variables with zero mean and unit variance. Their dot product:

$$\mathbf{q}^\top \mathbf{k} = \sum_{l=1}^{d_k} q_l k_l$$

Each term $q_l k_l$ has:
- $\mathbb{E}[q_l k_l] = 0$
- $\text{Var}(q_l k_l) = \text{Var}(q_l)\text{Var}(k_l) = 1$

By independence:

$$\text{Var}(\mathbf{q}^\top \mathbf{k}) = d_k$$

So the dot product has standard deviation $\sqrt{d_k}$. For large $d_k$ (e.g., 64), some dot products will be very large, pushing softmax into its saturated regime (near one-hot).

### 2.2 The Softmax Saturation Problem

When softmax inputs have large magnitude, the output is nearly one-hot:

$$\text{softmax}(c \cdot [1, 0]) = \left[\frac{e^c}{e^c + 1}, \frac{1}{e^c + 1}\right] \xrightarrow{c \to \infty} [1, 0]$$

In the saturated regime, gradients through softmax are nearly zero (vanishing gradient problem).

### 2.3 The Fix

Dividing by $\sqrt{d_k}$ normalizes the variance of dot products to 1:

$$\text{Var}\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

This keeps softmax in its sensitive (non-saturated) regime regardless of $d_k$.

```python
# Demonstrate the effect of scaling
d_k_values = [4, 16, 64, 256]

fig, axes = plt.subplots(1, len(d_k_values), figsize=(20, 4))
for ax, d_k in zip(axes, d_k_values):
    q = np.random.randn(d_k)
    K = np.random.randn(100, d_k)

    # Unscaled
    scores_unscaled = K @ q
    # Scaled
    scores_scaled = K @ q / np.sqrt(d_k)

    ax.hist(scores_unscaled, bins=30, alpha=0.5, label='Unscaled', density=True)
    ax.hist(scores_scaled, bins=30, alpha=0.5, label='Scaled', density=True)
    ax.set_title(f'$d_k$ = {d_k}\nstd unscaled = {scores_unscaled.std():.1f}')
    ax.legend()
    ax.set_xlim(-10, 10)

plt.suptitle('Dot product distributions: scaled vs unscaled')
plt.tight_layout()
plt.show()

# Effect on softmax entropy
print("\nSoftmax entropy at different scales:")
for scale in [1, 2, 4, 8, 16]:
    z = np.array([1.0, 0.5, 0.2, -0.3, -1.0]) * scale
    s = np.exp(z - np.max(z))
    s = s / s.sum()
    H = -np.sum(s * np.log(s + 1e-10))
    print(f"  scale={scale:2d}: entropy={H:.4f}, max_prob={s.max():.4f}")
```

---

## 3. Softmax Properties

### 3.1 Softmax Jacobian (Review)

$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$

$$\mathbf{J} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$$

### 3.2 Temperature Scaling

The **temperature** parameter $\tau$ controls the sharpness of the softmax distribution:

$$s_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

| $\tau$ | Effect |
|--------|--------|
| $\tau \to 0^+$ | Hard argmax (one-hot) |
| $\tau = 1$ | Standard softmax |
| $\tau \to \infty$ | Uniform distribution |

**DL usage**:
- Knowledge distillation: $\tau > 1$ to soften teacher outputs
- Sampling: $\tau < 1$ for sharper (more "confident") generation
- Contrastive learning: temperature controls the hardness of negative mining

```python
# Temperature scaling
z = np.array([2.0, 1.0, 0.5, -1.0, -2.0])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for tau in temperatures:
    s = np.exp(z / tau - np.max(z / tau))
    s = s / s.sum()
    axes[0].plot(range(len(z)), s, 'o-', label=f'τ={tau}')

axes[0].set_xlabel('Class')
axes[0].set_ylabel('Probability')
axes[0].set_title('Softmax with temperature')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Entropy vs temperature
taus = np.linspace(0.1, 10, 100)
entropies = []
for tau in taus:
    s = np.exp(z / tau - np.max(z / tau))
    s = s / s.sum()
    H = -np.sum(s * np.log(s + 1e-10))
    entropies.append(H)

axes[1].plot(taus, entropies, 'b-', linewidth=2)
axes[1].axhline(y=np.log(len(z)), color='r', linestyle='--', label=f'Max entropy = ln({len(z)})')
axes[1].set_xlabel('Temperature τ')
axes[1].set_ylabel('Entropy')
axes[1].set_title('Entropy vs. temperature')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. Gradient Flow Through Attention

### 4.1 Backward Through Attention

Let $\mathbf{O} = \mathbf{A}\mathbf{V}$ where $\mathbf{A} = \text{softmax}(\mathbf{S}/\sqrt{d_k})$ and $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$.

Given $\frac{\partial L}{\partial \mathbf{O}}$:

**Step 1**: Gradient w.r.t. $\mathbf{V}$ and $\mathbf{A}$:
$$\frac{\partial L}{\partial \mathbf{V}} = \mathbf{A}^\top \frac{\partial L}{\partial \mathbf{O}}$$
$$\frac{\partial L}{\partial \mathbf{A}} = \frac{\partial L}{\partial \mathbf{O}} \mathbf{V}^\top$$

**Step 2**: Gradient through softmax (per row $i$):
$$\frac{\partial L}{\partial S_{ij}} = \sum_l \frac{\partial L}{\partial A_{il}} \frac{\partial A_{il}}{\partial S_{ij}} = \sum_l \frac{\partial L}{\partial A_{il}} A_{il}(\delta_{lj} - A_{ij})$$

$$= A_{ij}\left(\frac{\partial L}{\partial A_{ij}} - \sum_l \frac{\partial L}{\partial A_{il}} A_{il}\right)$$

Let $\bar{A}_{ij} = \frac{\partial L}{\partial A_{ij}}$:

$$\frac{\partial L}{\partial S_{ij}} = A_{ij}\left(\bar{A}_{ij} - \sum_l \bar{A}_{il} A_{il}\right) = A_{ij}(\bar{A}_{ij} - \mathbf{a}_i \cdot \bar{\mathbf{a}}_i)$$

**Step 3**: Gradient w.r.t. $\mathbf{Q}$ and $\mathbf{K}$:
$$\frac{\partial L}{\partial \mathbf{Q}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial \mathbf{S}} \mathbf{K}$$
$$\frac{\partial L}{\partial \mathbf{K}} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial L}{\partial \mathbf{S}}\right)^\top \mathbf{Q}$$

```python
# Full attention forward + backward
def attention_forward_backward(Q, K, V, dO):
    """Complete attention forward and backward pass."""
    d_k = Q.shape[-1]
    T = Q.shape[0]

    # Forward
    S = Q @ K.T / np.sqrt(d_k)
    S_max = S.max(axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    A = exp_S / exp_S.sum(axis=-1, keepdims=True)
    O = A @ V

    # Backward
    dV = A.T @ dO                          # (T, d_v)
    dA = dO @ V.T                          # (T, T)

    # Through softmax
    # dS[i,j] = A[i,j] * (dA[i,j] - sum_l dA[i,l]*A[i,l])
    dot_dA_A = np.sum(dA * A, axis=-1, keepdims=True)  # (T, 1)
    dS = A * (dA - dot_dA_A) / np.sqrt(d_k)

    dQ = dS @ K
    dK = dS.T @ Q

    return O, A, dQ, dK, dV

# Verify with numerical gradients
T, d_k, d_v = 4, 3, 3
np.random.seed(42)
Q = np.random.randn(T, d_k)
K = np.random.randn(T, d_k)
V = np.random.randn(T, d_v)
dO = np.random.randn(T, d_v)

O, A, dQ, dK, dV = attention_forward_backward(Q, K, V, dO)

# Numerical check for dQ
eps = 1e-5
dQ_num = np.zeros_like(Q)
for i in range(T):
    for j in range(d_k):
        Q_plus = Q.copy(); Q_plus[i, j] += eps
        Q_minus = Q.copy(); Q_minus[i, j] -= eps
        O_plus, _, _, _, _ = attention_forward_backward(Q_plus, K, V, dO)
        O_minus, _, _, _, _ = attention_forward_backward(Q_minus, K, V, dO)
        dQ_num[i, j] = np.sum(dO * (O_plus - O_minus)) / (2 * eps)

print(f"dQ max error: {np.max(np.abs(dQ - dQ_num)):.2e}")
```

---

## 5. Multi-Head Attention

### 5.1 Motivation

A single attention head can only focus on one type of relationship per position. Multi-head attention runs $h$ attention heads in parallel, each with different learned projections:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = [\text{head}_1; \ldots; \text{head}_h]\mathbf{W}_O$$

where $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^{(i)}, \mathbf{K}\mathbf{W}_K^{(i)}, \mathbf{V}\mathbf{W}_V^{(i)})$

### 5.2 Dimensionality

With $h$ heads and $d_\text{model}$ model dimension:
- Each head operates in $d_k = d_v = d_\text{model} / h$
- Total computation is the same as single-head attention with full dimensionality
- But each head can learn to attend to different aspects

### 5.3 Implementation Trick

Instead of separate weight matrices per head, use a single large projection and reshape:

```python
def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads):
    """Multi-head attention.

    X: (T, d_model)
    W_Q, W_K, W_V: (d_model, d_model)
    W_O: (d_model, d_model)
    """
    T, d_model = X.shape
    d_head = d_model // n_heads

    # Project
    Q = X @ W_Q  # (T, d_model)
    K = X @ W_K
    V = X @ W_V

    # Reshape to (n_heads, T, d_head)
    Q = Q.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(T, n_heads, d_head).transpose(1, 0, 2)

    # Attention per head
    d_k = d_head
    scores = np.einsum('htd,hsd->hts', Q, K) / np.sqrt(d_k)
    scores_max = scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores - scores_max)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    # Apply attention to values
    out = np.einsum('hts,hsd->htd', attn, V)  # (n_heads, T, d_head)

    # Concatenate heads
    out = out.transpose(1, 0, 2).reshape(T, d_model)  # (T, d_model)

    # Final projection
    return out @ W_O, attn

# Test
T, d_model, n_heads = 8, 64, 8
np.random.seed(42)
X = np.random.randn(T, d_model) * 0.1
W_Q = np.random.randn(d_model, d_model) * 0.02
W_K = np.random.randn(d_model, d_model) * 0.02
W_V = np.random.randn(d_model, d_model) * 0.02
W_O = np.random.randn(d_model, d_model) * 0.02

output, attn_weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads)
print(f"Input: {X.shape}, Output: {output.shape}")
print(f"Attention weights per head: {attn_weights.shape}")
```

---

## 6. Computational Complexity

### 6.1 Standard Attention

| Operation | Complexity |
|-----------|-----------|
| $\mathbf{Q}\mathbf{K}^\top$ | $O(T^2 d_k)$ |
| Softmax | $O(T^2)$ |
| $\mathbf{A}\mathbf{V}$ | $O(T^2 d_v)$ |
| **Total** | **$O(T^2 d)$** |
| Memory for $\mathbf{A}$ | $O(T^2)$ per head |

For $T = 4096$ tokens, $h = 32$ heads: storing attention weights requires $32 \times 4096^2 \times 4$ bytes $\approx 2$ GB.

### 6.2 The Quadratic Bottleneck

The $O(T^2)$ scaling limits context length. This has motivated:
- **Flash Attention**: Same computation, but tile-based to avoid materializing the full $T \times T$ matrix
- **Linear Attention**: Replace softmax with a kernel to achieve $O(T \cdot d)$
- **Sparse Attention**: Only attend to a subset of positions

### 6.3 Causal (Autoregressive) Masking

For language models, position $i$ should only attend to positions $\leq i$:

$$A_{ij} = \begin{cases} \frac{\exp(S_{ij})}{\sum_{l \leq i} \exp(S_{il})} & j \leq i \\ 0 & j > i \end{cases}$$

Implemented by setting $S_{ij} = -\infty$ for $j > i$ before softmax.

```python
# Causal masking
T = 6
causal_mask = np.tril(np.ones((T, T), dtype=bool))

Q = np.random.randn(T, 8)
K = np.random.randn(T, 8)
V = np.random.randn(T, 8)

output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(causal_mask, cmap='Greys')
axes[0].set_title('Causal mask')
axes[1].imshow(weights_causal, cmap='Blues')
axes[1].set_title('Causal attention weights')
for ax in axes:
    ax.set_xlabel('Key position')
    ax.set_ylabel('Query position')
plt.tight_layout()
plt.show()
```

---

## 7. Attention as Soft Dictionary Lookup

### 7.1 Probabilistic Interpretation

Attention can be viewed as a **soft lookup** in a key-value store:

$$\text{output}_i = \sum_j P(j | i) \cdot \mathbf{v}_j = \mathbb{E}_{j \sim P(\cdot|i)}[\mathbf{v}_j]$$

where $P(j | i) = \alpha_{ij}$ is the attention weight. The output is a weighted average of values, with weights determined by query-key similarity.

### 7.2 Hard vs. Soft Attention

- **Hard attention** ($\tau \to 0$): Select the single most relevant value (argmax). Non-differentiable.
- **Soft attention** ($\tau = 1$): Weighted average of all values. Differentiable.
- **Sparse attention**: Most weights are zero, but the non-zero ones are soft. Differentiable.

### 7.3 Connection to Kernel Methods

Linear attention replaces $\exp(\mathbf{q}^\top \mathbf{k})$ with a kernel $\phi(\mathbf{q})^\top \phi(\mathbf{k})$:

$$\text{Attention}_i = \frac{\sum_j \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j) \mathbf{v}_j}{\sum_j \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)} = \frac{\phi(\mathbf{q}_i)^\top \sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top}{\phi(\mathbf{q}_i)^\top \sum_j \phi(\mathbf{k}_j)}$$

The key insight: $\sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top$ can be precomputed once, making the complexity $O(T \cdot d^2)$ instead of $O(T^2 \cdot d)$.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Scaled dot-product | $\text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\mathbf{V}$; scaling prevents softmax saturation |
| Scaling factor | $1/\sqrt{d_k}$ normalizes dot-product variance to 1 |
| Temperature | $\tau < 1$: sharper; $\tau > 1$: smoother; $\tau \to 0$: argmax |
| Attention gradient | Through softmax: $dS_{ij} = A_{ij}(\bar{A}_{ij} - \sum_l \bar{A}_{il} A_{il})$ |
| Multi-head | Parallel attention in $h$ subspaces; total cost unchanged |
| Complexity | $O(T^2 d)$ time, $O(T^2)$ memory per head |
| Causal mask | Set future scores to $-\infty$ for autoregressive models |

---

## Exercises

1. Implement scaled dot-product attention with causal masking and verify that position $i$ only attends to positions $\leq i$.
2. Plot the entropy of attention weights as a function of $d_k$ (without scaling) to show why scaling is necessary.
3. Implement multi-head attention from scratch and verify shapes at each step.
4. Compare the attention weight entropy at different temperatures and discuss the implications for knowledge distillation.
5. Implement linear attention using the $\text{elu}(x) + 1$ feature map and compare its output with standard attention.

---

**Next**: [12. Putting It All Together](12_Putting_It_All_Together.md)

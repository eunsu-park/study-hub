# Batch Normalization

**Previous**: [Regularization](./09_Regularization.md) | **Next**: [Universal Approximation](./11_Universal_Approximation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain internal covariate shift and why it hinders training
2. Derive the batch normalization forward pass step by step
3. Implement batch normalization for training and inference modes
4. Compute the backward pass through batch normalization
5. Explain the roles of learnable parameters gamma and beta
6. Describe how running statistics are maintained for inference
7. Discuss where to place batch normalization in a network
8. Compare batch normalization with layer normalization

---

Batch normalization (BatchNorm, BN), introduced by Ioffe and Szegedy in 2015, was one of the most impactful innovations in deep learning. By normalizing activations within each mini-batch, it stabilizes training, allows higher learning rates, reduces sensitivity to initialization, and acts as a mild regularizer. Understanding BN deeply is essential for both using and debugging modern neural networks.

---

## 1. The Problem: Internal Covariate Shift

### 1.1 What Is It?

As training progresses, the weights in each layer change. This means the distribution of inputs to each layer shifts constantly:

```
Epoch 1:   Layer 1 outputs have mean=0.5, std=1.2
Epoch 10:  Layer 1 outputs have mean=-0.3, std=0.8
Epoch 100: Layer 1 outputs have mean=1.1, std=2.5

Layer 2 must constantly readjust to a moving input distribution.
This is "internal covariate shift."
```

### 1.2 Consequences

- **Slower training**: Each layer must chase a moving target
- **Requires careful initialization**: Bad init → extreme activation shifts
- **Lower learning rates**: Must use small steps to avoid instability
- **Saturated activations**: Shifting distributions may push values into saturation zones

---

## 2. The Batch Normalization Algorithm

### 2.1 Core Idea

Normalize each feature to have zero mean and unit variance within each mini-batch, then apply a learnable affine transformation:

```
For each feature k in a mini-batch of m samples:

Step 1: Mini-batch mean
  μ_B = (1/m) Σ_{i=1}^{m} z_i^(k)

Step 2: Mini-batch variance
  σ_B^2 = (1/m) Σ_{i=1}^{m} (z_i^(k) - μ_B)^2

Step 3: Normalize
  ẑ_i^(k) = (z_i^(k) - μ_B) / √(σ_B^2 + ε)

Step 4: Scale and shift (learnable)
  y_i^(k) = γ · ẑ_i^(k) + β

Where γ (scale) and β (shift) are learnable parameters.
```

### 2.2 Why Learnable γ and β?

If we only normalized (Step 3), we would restrict the network's expressiveness. The learnable parameters allow the network to undo the normalization if needed:

```
If γ = √(σ_B^2 + ε) and β = μ_B:
  y = γ · ẑ + β = γ · (z - μ) / √(σ^2 + ε) + β = z

The network can learn to make BN an identity operation if that's optimal.
```

### 2.3 Where to Apply BN

```
Option A (original paper):     z → BN → activation → a
Option B (modern practice):    z → activation → BN → a

                Linear    BN    Activation
  a^(l-1) ──► W·a + b ──► BN ──► ReLU ──► a^(l)

When using BN, the bias b in the linear layer is redundant
(β in BN serves the same purpose). Often set b = 0.
```

---

## 3. Implementation

### 3.1 Training Mode

```python
import numpy as np

class BatchNorm:
    """Batch Normalization layer."""

    def __init__(self, n_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones((n_features, 1))
        self.beta = np.zeros((n_features, 1))
        self.eps = eps
        self.momentum = momentum

        # Running statistics for inference
        self.running_mean = np.zeros((n_features, 1))
        self.running_var = np.ones((n_features, 1))

        # Cache for backward pass
        self.cache = None

    def forward(self, z, training=True):
        """
        Args:
            z: Pre-activations, shape (n_features, batch_size)
            training: Boolean flag
        """
        if training:
            # Compute batch statistics
            mu = np.mean(z, axis=1, keepdims=True)      # (n, 1)
            var = np.var(z, axis=1, keepdims=True)       # (n, 1)

            # Normalize
            z_norm = (z - mu) / np.sqrt(var + self.eps)  # (n, m)

            # Scale and shift
            out = self.gamma * z_norm + self.beta         # (n, m)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            # Cache for backward
            self.cache = (z, z_norm, mu, var)

            return out
        else:
            # Use running statistics
            z_norm = (z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * z_norm + self.beta
```

### 3.2 Backward Pass

The backward pass through BN requires careful derivation:

```
Given: dout = ∂L/∂y    (upstream gradient)

∂L/∂γ = Σ_i dout_i · ẑ_i         (sum over batch)
∂L/∂β = Σ_i dout_i               (sum over batch)

∂L/∂ẑ = dout · γ
∂L/∂σ^2 = Σ_i ∂L/∂ẑ_i · (z_i - μ) · (-1/2)(σ^2 + ε)^(-3/2)
∂L/∂μ = Σ_i ∂L/∂ẑ_i · (-1/√(σ^2 + ε))
∂L/∂z_i = ∂L/∂ẑ_i / √(σ^2 + ε) + ∂L/∂σ^2 · 2(z_i - μ)/m + ∂L/∂μ / m
```

```python
    def backward(self, dout):
        """Backward pass through batch normalization."""
        z, z_norm, mu, var = self.cache
        m = z.shape[1]  # batch size

        # Gradients for learnable parameters
        dgamma = np.sum(dout * z_norm, axis=1, keepdims=True)
        dbeta = np.sum(dout, axis=1, keepdims=True)

        # Gradient for input (efficient formula)
        dz_norm = dout * self.gamma
        dvar = np.sum(dz_norm * (z - mu) * (-0.5) * (var + self.eps)**(-1.5),
                      axis=1, keepdims=True)
        dmu = np.sum(dz_norm * (-1.0 / np.sqrt(var + self.eps)),
                     axis=1, keepdims=True)

        dz = (dz_norm / np.sqrt(var + self.eps)
              + dvar * 2 * (z - mu) / m
              + dmu / m)

        return dz, dgamma, dbeta
```

---

## 4. Training vs. Inference

### 4.1 The Two Modes

```
Training Mode:
  - Use batch statistics (μ_B, σ_B^2) from current mini-batch
  - Update running_mean and running_var with exponential moving average
  - Apply dropout (if any)

Inference Mode:
  - Use running statistics (accumulated during training)
  - No dropout
  - Deterministic output

IMPORTANT: Forgetting to switch to eval mode is a common bug!
```

### 4.2 Running Statistics Update

```
running_mean ← (1 - α) · running_mean + α · μ_B
running_var  ← (1 - α) · running_var  + α · σ_B^2

Where α = momentum (typically 0.1)

Over many batches, running statistics converge to population statistics.
```

---

## 5. Benefits of Batch Normalization

### 5.1 Allows Higher Learning Rates

```
Without BN:  lr = 0.001 (higher → diverges)
With BN:     lr = 0.01 or 0.1 (stable at higher LR)

→ Faster convergence (fewer epochs needed)
```

### 5.2 Reduces Sensitivity to Initialization

```
Without BN:  Bad init → activations explode/vanish → training fails
With BN:     BN normalizes activations → training succeeds regardless

This doesn't mean init doesn't matter at all, but BN makes it far less critical.
```

### 5.3 Regularization Effect

BN introduces noise via batch statistics (μ_B and σ_B^2 are noisy estimates of population statistics). This noise acts as a regularizer, reducing the need for dropout.

---

## 6. Batch Normalization Limitations

### 6.1 Batch Size Dependency

```
Large batch (B=256):  μ_B ≈ μ_population    (good estimate)
Small batch (B=4):    μ_B ≈ noisy           (unreliable)
Batch size 1:         μ_B = z               (BN is useless)
```

### 6.2 Alternatives for Small Batches

| Method | Normalization Axis | Batch Size Dependent? |
|--------|-------------------|----------------------|
| Batch Norm | Across batch, per feature | Yes |
| Layer Norm | Across features, per sample | No |
| Instance Norm | Per sample, per channel | No |
| Group Norm | Per sample, groups of channels | No |

### 6.3 Layer Normalization

```
Batch Norm:  normalize across the BATCH dimension (each feature independently)
Layer Norm:  normalize across the FEATURE dimension (each sample independently)

BN: for each feature k, compute mean/var over all samples in batch
LN: for each sample i, compute mean/var over all features

Layer Norm is preferred for:
  - Transformers / NLP (variable sequence lengths)
  - Small batch sizes
  - Recurrent networks
```

```python
class LayerNorm:
    """Layer normalization (batch-size independent)."""

    def __init__(self, n_features, eps=1e-5):
        self.gamma = np.ones((n_features, 1))
        self.beta = np.zeros((n_features, 1))
        self.eps = eps

    def forward(self, z):
        """z: shape (n_features, batch_size)"""
        mu = np.mean(z, axis=0, keepdims=True)      # mean over features
        var = np.var(z, axis=0, keepdims=True)       # var over features
        z_norm = (z - mu) / np.sqrt(var + self.eps)
        return self.gamma * z_norm + self.beta
```

---

## 7. Practical Tips

### 7.1 BN Placement

```
Recommended: Conv/Linear → BN → ReLU (before activation)
Also works:  Conv/Linear → ReLU → BN (after activation)

Remove bias from linear layer when using BN (β replaces it).
```

### 7.2 BN with Dropout

```
Using both can cause issues:
  - BN assumes consistent activation distributions
  - Dropout changes the distribution between training and inference

Modern practice:
  - For CNNs: BN only (no dropout)
  - For MLPs: BN + light dropout (p=0.1-0.2)
  - For Transformers: Layer Norm + dropout
```

---

## 8. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Internal covariate shift: layer inputs shift as earlier layers change
2. BN normalizes to zero mean, unit variance per feature, per batch
3. Learnable γ (scale) and β (shift) preserve expressiveness
4. Training: batch statistics; Inference: running statistics
5. Benefits: faster training, higher LR, less init sensitivity
6. Limitation: requires large enough batch size
7. Layer Norm: alternative for small batches and Transformers
8. Remove bias in linear layers when using BN
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Implement BN and verify that layer activations have mean ≈ 0, std ≈ 1
2. Train a network with and without BN; compare convergence speed
3. Implement Layer Normalization and compare with Batch Normalization
4. Experiment with batch sizes (4, 16, 64, 256) and observe BN stability

---

**Previous**: [Regularization](./09_Regularization.md) | **Next**: [Universal Approximation](./11_Universal_Approximation.md)

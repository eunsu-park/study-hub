[Previous: Generative Models - VAE](./30_Generative_Models_VAE.md) | [Next: Diffusion Models](./32_Diffusion_Models.md)

---

# 31. Variational Autoencoder (VAE)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the generative model goal of VAE and describe why direct maximization of the marginal likelihood p(x) is intractable.
2. Derive the Evidence Lower BOund (ELBO) from the variational inference framework and interpret its two components: reconstruction loss and KL divergence regularization.
3. Describe the reparameterization trick and explain why it is necessary for backpropagating gradients through the stochastic sampling step.
4. Implement a VAE from scratch in PyTorch, including the encoder (posterior network), decoder (likelihood network), and ELBO loss computation.
5. Perform latent space interpolation and arithmetic to demonstrate that the VAE learns a smooth, structured latent manifold.
6. Compare VAE with GAN in terms of training stability, output sharpness, and latent space interpretability, and identify scenarios where each excels.

---

## Overview

Variational Autoencoder (VAE) is a foundational generative model architecture that learns latent representations of data and can generate new samples. "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)

---

## Mathematical Background

### 1. Generative Model Goal

```
Goal: model p(x)
- x: observed data (images, etc.)
- z: latent variable

Generation process:
z ~ p(z)         # Prior (usually N(0, I))
x ~ p(x|z)       # Decoder/Generator

Problem: p(x) = ∫ p(x|z)p(z)dz is intractable
```

### 2. Variational Inference

```
Posterior p(z|x) is also intractable
→ Learn approximate distribution q(z|x) (Encoder)

ELBO (Evidence Lower BOund):
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
         ────────────────   ─────────────────────
         Reconstruction     Regularization
         Loss               (Prior matching)

Objective to maximize:
L(θ, φ; x) = E_q_φ(z|x)[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

### 3. Reparameterization Trick

```
Problem: sampling z ~ q(z|x) = N(μ, σ²) is not differentiable

Solution: Reparameterization
ε ~ N(0, I)
z = μ + σ ⊙ ε

Now gradient can backpropagate through μ, σ!

┌─────────────────────────────────────────┐
│  Encoder                                │
│  x → [μ, log σ²]                        │
│                                         │
│  Reparameterization                     │
│  ε ~ N(0, I)                           │
│  z = μ + σ ⊙ ε                         │
│                                         │
│  Decoder                                │
│  z → x̂                                  │
└─────────────────────────────────────────┘
```

### 4. Loss Function

```
L = L_recon + β * L_KL

Reconstruction Loss (images):
- Binary: BCE(x, x̂) = -Σ[x·log(x̂) + (1-x)·log(1-x̂)]
- Continuous: MSE(x, x̂) = ||x - x̂||²

KL Divergence (Gaussian prior):
KL(N(μ, σ²) || N(0, 1)) = -½ Σ(1 + log σ² - μ² - σ²)

β-VAE:
β > 1: stronger disentanglement
β < 1: better reconstruction
```

---

## VAE Architecture

### Standard VAE (MNIST)

```
Encoder:
Input (28×28×1)
    ↓
Conv2d(1→32, k=3, s=2, p=1)  → (14×14×32)
    ↓ ReLU
Conv2d(32→64, k=3, s=2, p=1) → (7×7×64)
    ↓ ReLU
Flatten → (7×7×64 = 3136)
    ↓
Linear(3136→256)
    ↓ ReLU
┌────────────────┬────────────────┐
│ Linear(256→z)  │ Linear(256→z)  │
│     μ          │    log σ²      │
└────────────────┴────────────────┘

Reparameterization:
z = μ + σ ⊙ ε,  ε ~ N(0, I)

Decoder:
z (latent_dim)
    ↓
Linear(z→256)
    ↓ ReLU
Linear(256→3136)
    ↓ ReLU
Reshape → (7×7×64)
    ↓
ConvT2d(64→32, k=3, s=2, p=1, op=1) → (14×14×32)
    ↓ ReLU
ConvT2d(32→1, k=3, s=2, p=1, op=1)  → (28×28×1)
    ↓ Sigmoid
Output (28×28×1)
```

---

## File Structure

```
11_VAE/
├── README.md
├── numpy/
│   └── vae_numpy.py          # NumPy VAE (forward only)
├── pytorch_lowlevel/
│   └── vae_lowlevel.py       # PyTorch Low-Level VAE
├── paper/
│   └── vae_paper.py          # Paper reproduction
└── exercises/
    ├── 01_latent_space.md    # Latent space visualization
    └── 02_interpolation.md   # Latent space interpolation
```

---

## Core Concepts

### 1. Latent Space

```
Good latent space characteristics:
1. Continuity: nearby points produce similar outputs
2. Completeness: all points generate meaningful outputs
3. (Disentanglement): each dimension controls independent features

VAE vs AE:
- AE: point embeddings → discontinuous, has empty spaces
- VAE: distribution embeddings → continuous, can sample
```

### 2. VAE Variants

```
β-VAE (β > 1):
- Stronger KL regularization
- Better disentanglement
- Worse reconstruction

Conditional VAE (CVAE):
- Add condition c: q(z|x, c), p(x|z, c)
- Enables conditional generation

VQ-VAE:
- Discrete codebook instead of continuous latent space
- Used in DALL-E, AudioLM, etc.
```

### 3. Training Stability

```
KL Annealing:
- Initial: β=0 (focus on reconstruction)
- Gradually β→1 (add regularization)

Free Bits:
- Ensure minimum KL (prevent posterior collapse)
- L_KL = max(KL, λ)
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Directly use F.conv2d, F.linear
- Implement reparameterization trick
- Implement ELBO loss function

### Level 3: Paper Implementation (paper/)
- Implement β-VAE
- Implement CVAE (Conditional)
- Latent space visualization

---

## Learning Checklist

- [ ] Understand ELBO derivation process
- [ ] Understand reparameterization trick
- [ ] Calculate KL divergence
- [ ] Understand role of β
- [ ] Visualize latent space
- [ ] Implement Conditional VAE

---

## References

- Kingma & Welling (2013). "Auto-Encoding Variational Bayes"
- Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- [../Deep_Learning/16_VAE.md](../Deep_Learning/16_VAE.md)

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

## Theory & Principles

This implementation lesson grounds the VAE math from the previous one in concrete tensor operations. Three details consistently catch first-time implementers: the `log_var` parameterization (and why we predict log-variance, not variance), the closed-form KL term and its sign, and the relative weighting of reconstruction vs KL when both are summed into one scalar loss.

This section covers:

- **A.** Why predict `log \sigma^2` instead of `\sigma`
- **B.** The KL closed form, term by term
- **C.** Reconstruction loss choices: BCE vs MSE
- **D.** Latent space arithmetic and what makes it work

### A. Why `log_var` Not `var`

The encoder must output a positive variance. Two ways to ensure this:

1. **Output `\sigma` directly and apply `softplus` or `exp`** to ensure positivity.
2. **Output `log \sigma^2` and exponentiate when needed** for sampling.

Option 2 is universally preferred because:

- `log \sigma^2` is unconstrained — the network can produce any real value, including negative (for small variances) or large positive (for high variances), without any activation function.
- `exp(0.5 * log_var)` gives `\sigma` cleanly when needed.
- The KL formula has `log \sigma^2` directly, so no extra logarithm is computed.

Numerical stability also favors `log_var`: small variances (`\sigma \approx 10^{-3}`) are easier to represent as `log_var \approx -7` than as direct floats with risk of underflow.

### B. The KL Closed Form, Term by Term

For a `d`-dimensional Gaussian `q = N(\mu, diag(\sigma^2))` and prior `p = N(0, I)`:

```
KL(q || p) = 0.5 * sum_{j=1}^{d} [\mu_j^2 + \sigma_j^2 - log \sigma_j^2 - 1]
```

Each term has meaning:

- `\mu_j^2`: penalizes posterior means far from 0 (the prior's mean).
- `\sigma_j^2`: penalizes posterior variances bigger than 1 (the prior's variance).
- `-log \sigma_j^2`: penalizes variances *smaller* than 1 (the log goes to `+inf` as `\sigma \to 0`).
- `-1`: a constant that makes the value 0 when `q = p` exactly.

The term `\sigma^2 - log \sigma^2 - 1` is non-negative for `\sigma > 0` with minimum 0 at `\sigma = 1`. It pushes variances toward 1.

In code: `kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()`. Sign convention: KL is *added* to reconstruction loss for total loss = `-ELBO`.

### C. Reconstruction Loss: BCE vs MSE

Two common choices:

- **BCE per pixel** (treats pixel intensities as Bernoulli probabilities, summing across spatial dims): the original VAE paper used this for binarized MNIST. Sharp boundaries; works for `[0, 1]` intensity images.
- **MSE per pixel** (treats outputs as Gaussian-noise observations of the true pixel): standard for color images. Slightly blurrier; corresponds to maximum likelihood under Gaussian observation noise.

Crucially: the reconstruction term in ELBO is `-log p(x | z)`. For BCE this is `BCELoss(x_hat, x)`; for Gaussian observations with fixed variance `\sigma^2 = 0.5` it is `0.5 * MSE`. The factor matters when balancing against KL.

### D. Latent Space Arithmetic

VAE latent spaces support meaningful arithmetic — `z(king) - z(man) + z(woman) ≈ z(queen)` style operations — and smooth interpolation. Why does this work?

Two structural properties:

1. **Continuity**: small changes in `z` produce small changes in `decoder(z)`. The KL term enforces this by preventing the encoder from putting different examples at wildly different `z` values.
2. **Topological coverage**: the prior `N(0, I)` puts probability mass densely throughout the latent space, and the encoder is trained to produce posteriors that respect this. Linear interpolations between two points stay within the support of the prior.

GAN latent spaces also support arithmetic but less reliably — there is no KL term enforcing continuity, so interpolated latents can pass through "void" regions where the generator has not been trained to produce reasonable output.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| Encoder output | `mu, log_var = self.encoder(x).split(latent_dim, dim=-1)` |
| Reparameterization | `z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)` |
| KL closed form | `-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())` |
| Reconstruction | `F.binary_cross_entropy(x_hat, x, reduction='sum')` |
| Latent arithmetic | `z_new = z_a - z_b + z_c; out = decoder(z_new)` |

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

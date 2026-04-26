[Previous: Generative Models - GAN](./28_Generative_Models_GAN.md) | [Next: Generative Models - VAE](./30_Generative_Models_VAE.md)

---

# 29. Generative Adversarial Networks (GAN)

## Learning Objectives

After completing this lesson, you will be able to:

1. Formulate the GAN minimax objective and explain the roles of the generator and discriminator in the adversarial training game.
2. Derive the optimal discriminator D*(x) for a fixed generator and use it to show that the GAN objective minimizes Jensen-Shannon divergence.
3. Identify and explain common GAN training instabilities — mode collapse, vanishing gradients, and non-convergence — and describe techniques to mitigate them.
4. Implement a DCGAN (Deep Convolutional GAN) from scratch in PyTorch using transposed convolutions in the generator and strided convolutions in the discriminator.
5. Train a GAN on an image dataset and evaluate generated sample quality using visual inspection and the Fréchet Inception Distance (FID) metric.
6. Compare GAN variants (DCGAN, WGAN, Conditional GAN) and explain the architectural or loss modifications each introduces to improve training stability.

---

## Theory & Principles

The DCGAN implementation builds on the GAN theory from the previous lesson. This section focuses on the *implementation* truths: the architectural rules that make GAN training work in practice, the FID metric's mathematical content, and what each WGAN/WGAN-GP code change is actually computing.

This section covers:

- **A.** DCGAN architectural guidelines and why they exist
- **B.** Strided / transposed convolutions for downsampling and upsampling
- **C.** FID: comparing distributions via Inception features
- **D.** WGAN-GP: where the gradient penalty term comes from

### A. DCGAN Architectural Guidelines

Radford et al. (2016) cataloged a handful of architectural rules that empirically make convolutional GANs train stably:

1. **No fully-connected layers**, except input projection in G and output classification in D.
2. **Use BatchNorm in both G and D**, except at G's output layer and D's input layer.
3. **In G, replace pooling with strided transposed convolution.** In D, replace pooling with strided convolution. Lets the network learn its own up/down sampling.
4. **ReLU in G (except output Tanh), Leaky ReLU in D.** Tanh output keeps generator output in `[-1, 1]` matching the normalized image range; Leaky ReLU avoids dying-ReLU in the discriminator under early-training conditions.
5. **No fully connected hidden layers** anywhere.

These are not derivations; they are consolidated empirical wisdom. Understanding *why* each helps tells you when you can break the rules — for example, BatchNorm in D conflicts with WGAN-GP (because gradient penalty is computed on individual examples), so WGAN-GP uses LayerNorm or no norm there.

### B. Strided and Transposed Convolutions

For downsampling in D: `Conv2d(in, out, kernel=4, stride=2, padding=1)` halves spatial size. For upsampling in G: `ConvTranspose2d(in, out, kernel=4, stride=2, padding=1)` doubles it. Output size formulas:

```
Strided conv:        H_out = floor((H_in + 2P - K) / S) + 1
Transposed conv:     H_out = (H_in - 1) * S - 2P + K
```

A common DCGAN pattern: G upsamples from a `1x1` latent code through `4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64` via a chain of transposed convs; D mirrors this with strided convs going `64 -> 32 -> 16 -> 8 -> 4 -> 1`.

The kernel-4 / stride-2 / padding-1 choice avoids "checkerboard artifacts" that plague kernel-3 / stride-2 transposed convs (because stride doesn't divide kernel size cleanly).

### C. Fréchet Inception Distance (FID)

FID measures GAN output quality without human evaluation:

1. Run a pretrained Inception-v3 on real images and on generated images, extracting the 2048-d feature vector from the final pooling layer.
2. Fit a multivariate Gaussian to each set's features: `N(\mu_real, \Sigma_real)` and `N(\mu_gen, \Sigma_gen)`.
3. Compute the **Fréchet distance** between the two Gaussians:

```
FID = ||\mu_real - \mu_gen||^2 + Tr(\Sigma_real + \Sigma_gen - 2 (\Sigma_real \Sigma_gen)^{1/2})
```

Lower FID = generated features distribute more like real ones. Reasonable scale: FID < 10 is excellent, > 50 is poor for typical image datasets.

The choice of Inception features is somewhat arbitrary but standardized across the literature. A common criticism: Inception was trained on ImageNet, so FID is biased toward ImageNet-like features. For non-natural images (medical, satellite), domain-specific feature extractors give more meaningful comparisons.

### D. Gradient Penalty Derivation

WGAN-GP enforces the 1-Lipschitz constraint on the critic by penalizing gradient magnitudes:

```
L_GP = E_{\hat{x}} [(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]
```

`\hat{x}` is sampled by interpolating between real and generated samples: `\hat{x} = \alpha x + (1 - \alpha) G(z)` with `\alpha ~ U[0, 1]`. The penalty pushes the gradient norm toward 1 (not just below 1) because the optimal Lipschitz function for the Wasserstein dual has gradient norm exactly 1 almost everywhere.

Implementation note: `torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)` gives the per-input gradient; `create_graph=True` is essential because we need to backpropagate through this gradient when updating D.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| DCGAN G | `ConvTranspose2d` chain with BN + ReLU, Tanh output |
| DCGAN D | `Conv2d` chain with stride 2 + LeakyReLU, sigmoid (or none for WGAN) |
| FID | `pytorch_fid` package or manual `mu`/`sigma` computation |
| Gradient penalty | `(grads.norm(2, dim=1) - 1).pow(2).mean()` |

---


## Overview

Generative Adversarial Networks (GAN) learn to generate realistic data through an adversarial game between a Generator and a Discriminator. "Generative Adversarial Networks" (Goodfellow et al., 2014)

---

## Mathematical Background

### 1. Minimax Game

```
Goal: Generator G fools Discriminator D

Minimax objective:
min_G max_D V(D, G) = Eₓ~pdata[log D(x)] + Ez~pz[log(1 - D(G(z)))]
                      ──────────────────   ──────────────────────
                      Real data            Fake data
                      (maximize D(x)→1)    (maximize D(G(z))→0)

D's goal: Maximize V (distinguish real from fake)
G's goal: Minimize V (fool D into thinking fake is real)

Optimal discriminator:
D*(x) = pdata(x) / (pdata(x) + pg(x))

At Nash equilibrium: pg = pdata, D*(x) = 1/2
```

### 2. Training Dynamics

```
Alternating optimization:

Step 1: Update D (fix G)
  Maximize log D(x) + log(1 - D(G(z)))
  → D learns to classify real vs fake

Step 2: Update G (fix D)
  Minimize log(1 - D(G(z)))
  or Maximize log D(G(z))  ← Non-saturating variant (better gradient)

Why maximize log D(G(z))?
- Early training: G is bad → D(G(z)) ≈ 0
- log(1 - D(G(z))) ≈ 0 → vanishing gradient
- log D(G(z)) provides stronger gradient

┌─────────────────────────────────────────┐
│  GAN Training Loop:                     │
│                                         │
│  for epoch in epochs:                   │
│    for real_batch in dataloader:        │
│      # 1. Update Discriminator          │
│      z ~ N(0, I)                        │
│      fake = G(z)                        │
│      loss_D = -log D(real) - log(1-D(fake))
│      D.step()                           │
│                                         │
│      # 2. Update Generator              │
│      z ~ N(0, I)                        │
│      fake = G(z)                        │
│      loss_G = -log D(fake)              │
│      G.step()                           │
└─────────────────────────────────────────┘
```

### 3. Loss Functions

```
Original GAN (Minimax):
L_D = -[log D(x) + log(1 - D(G(z)))]
L_G = -log D(G(z))  (non-saturating)

WGAN (Wasserstein GAN):
L_D = -[D(x) - D(G(z))]  (no sigmoid, critic instead)
L_G = -D(G(z))
+ Weight clipping or Gradient Penalty

LSGAN (Least Squares GAN):
L_D = (D(x) - 1)² + D(G(z))²
L_G = (D(G(z)) - 1)²

Hinge Loss (Spectral Norm GAN):
L_D = -min(0, -1 + D(x)) - min(0, -1 - D(G(z)))
L_G = -D(G(z))
```

---

## DCGAN Architecture

Deep Convolutional GAN (Radford et al., 2015) - Stable training guidelines

### Generator (64×64 RGB images)

```
Latent Code z (100-dim)
    ↓
Linear(100→4×4×1024) + BatchNorm + ReLU
    ↓
Reshape → (4×4×1024)
    ↓
ConvTranspose2d(1024→512, k=4, s=2, p=1) → (8×8×512)
    ↓ BatchNorm + ReLU
ConvTranspose2d(512→256, k=4, s=2, p=1)  → (16×16×256)
    ↓ BatchNorm + ReLU
ConvTranspose2d(256→128, k=4, s=2, p=1)  → (32×32×128)
    ↓ BatchNorm + ReLU
ConvTranspose2d(128→3, k=4, s=2, p=1)    → (64×64×3)
    ↓ Tanh
Output (64×64×3, range [-1, 1])

Key design choices:
- No fully connected layers (except first projection)
- Use transposed convolutions for upsampling
- BatchNorm in all layers except output
- ReLU activation in G
- Tanh output (images normalized to [-1, 1])
```

### Discriminator (64×64 RGB images)

```
Input (64×64×3)
    ↓
Conv2d(3→128, k=4, s=2, p=1)  → (32×32×128)
    ↓ LeakyReLU(0.2)
Conv2d(128→256, k=4, s=2, p=1) → (16×16×256)
    ↓ BatchNorm + LeakyReLU(0.2)
Conv2d(256→512, k=4, s=2, p=1) → (8×8×512)
    ↓ BatchNorm + LeakyReLU(0.2)
Conv2d(512→1024, k=4, s=2, p=1) → (4×4×1024)
    ↓ BatchNorm + LeakyReLU(0.2)
Conv2d(1024→1, k=4, s=1, p=0) → (1×1×1)
    ↓ Sigmoid (or remove for WGAN)
Output (scalar probability)

Key design choices:
- No fully connected layers (except implicit in final conv)
- Strided convolutions for downsampling (no pooling)
- BatchNorm in all layers except input/output
- LeakyReLU activation (α=0.2)
- Sigmoid output for binary classification
```

### DCGAN Guidelines

```
1. Replace pooling with strided convolutions (D) / transposed convs (G)
2. Use BatchNorm in both G and D
3. Remove fully connected hidden layers
4. Use ReLU in G (except output: Tanh)
5. Use LeakyReLU in D (α=0.2)
```

---

## Training Techniques

### 1. Label Smoothing

```
Problem: D becomes too confident (D(real)→1, D(fake)→0)
→ Vanishing gradients for G

Solution: Smooth labels
Real labels: 1.0 → 0.9 (one-sided label smoothing)
Fake labels: 0.0 (keep as is)

loss_D_real = BCE(D(real), 0.9)  # Instead of 1.0
loss_D_fake = BCE(D(fake), 0.0)
```

### 2. Feature Matching

```
Problem: G optimizes for fooling D, not generating realistic samples

Solution: Match intermediate features
loss_G = ||E[f(x)] - E[f(G(z))]||²

Where f(·) is intermediate layer of D

Stabilizes training, reduces mode collapse
```

### 3. Minibatch Discrimination

```
Problem: G produces limited variety (mode collapse)

Solution: Let D look at entire batch
1. Extract features from each sample
2. Compute similarity within batch
3. Append batch statistics to each sample

D can detect if G produces identical samples
```

### 4. Spectral Normalization

```
Problem: Discriminator gradients explode

Solution: Normalize weight matrices
W_SN = W / σ(W)

Where σ(W) is largest singular value

Stabilizes training (Miyato et al., 2018)
Used in BigGAN, StyleGAN
```

### 5. Progressive Growing

```
Start training at low resolution (4×4)
Gradually add layers to reach high resolution (1024×1024)

4×4 → 8×8 → 16×16 → ... → 1024×1024

Smooth transition between resolutions
Used in ProGAN, StyleGAN
```

---

## Mode Collapse

### 1. What is Mode Collapse?

```
Problem: G produces limited variety

Modes of data distribution:
                 Mode 1  Mode 2  Mode 3
Real data:         ●●●     ●●●     ●●●
Healthy GAN:       ○○○     ○○○     ○○○
Mode collapse:     ○○○
Full collapse:              ○○○○○○○

Generator ignores parts of data distribution
```

### 2. Detecting Mode Collapse

```
Symptoms:
1. Generated samples look similar
2. Low diversity despite different z
3. Training loss oscillates
4. NLL (Negative Log-Likelihood) high despite low FID

Metrics:
- Inception Score (IS): measures diversity + quality
- Fréchet Inception Distance (FID): distribution distance
- Precision/Recall: mode coverage
```

### 3. Mitigation Strategies

```
1. Unrolled GAN:
   - G optimizes against future D (k steps ahead)
   - Prevents G from exploiting current D

2. Minibatch Discrimination:
   - D detects lack of diversity

3. WGAN / WGAN-GP:
   - Smoother gradient flow
   - Better training stability

4. Multiple Discriminators:
   - Each D captures different modes

5. Regularization:
   - Add noise to D inputs
   - Dropout in D
```

---

## File Structure

```
14_GAN/
├── README.md
├── pytorch_lowlevel/
│   ├── dcgan_mnist.py        # DCGAN on MNIST (28×28)
│   └── dcgan_cifar.py        # DCGAN on CIFAR-10 (32×32)
├── paper/
│   ├── dcgan_paper.py        # Full DCGAN (64×64)
│   ├── wgan_gp.py            # WGAN with Gradient Penalty
│   ├── stylegan_simple.py    # Simplified StyleGAN
│   └── conditional_gan.py    # Conditional GAN (cGAN)
└── exercises/
    ├── 01_mode_collapse.md   # Diagnose mode collapse
    └── 02_spectral_norm.md   # Implement spectral normalization
```

---

## Core Concepts

### 1. GAN Variants

```
Conditional GAN (cGAN):
- Add class label c: G(z, c), D(x, c)
- Controlled generation (e.g., generate digit 7)

WGAN (Wasserstein GAN):
- Replace JS divergence with Wasserstein distance
- No sigmoid in D (becomes critic)
- Weight clipping or gradient penalty
- More stable training

StyleGAN:
- Progressive architecture
- Style modulation (AdaIN)
- Disentangled latent space W
- State-of-the-art image quality

CycleGAN:
- Unpaired image-to-image translation
- Cycle consistency loss: G(F(x)) ≈ x
- Use cases: horse↔zebra, summer↔winter
```

### 2. Evaluation Metrics

```
Inception Score (IS):
IS = exp(E[KL(p(y|x) || p(y))])

Higher is better (quality + diversity)
Range: 1 to C (num classes)

Fréchet Inception Distance (FID):
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real·Σ_fake))

Lower is better (closer to real distribution)
Gold standard for GANs

Precision/Recall:
Precision = quality (fake samples are realistic)
Recall = coverage (all modes captured)
```

### 3. Training Tips

```
1. Learning rates:
   - D: 2e-4 (Adam, β₁=0.5, β₂=0.999)
   - G: 2e-4 (same optimizer settings)

2. Update frequency:
   - 1 G update per 1-5 D updates
   - D should be slightly ahead

3. Initialization:
   - Xavier/He initialization
   - BatchNorm parameters: γ=1, β=0

4. Data:
   - Normalize images to [-1, 1] (Tanh output)
   - Random flip augmentation

5. Latent code:
   - z ~ N(0, I), dim = 100-512
   - Can use uniform U(-1, 1)

6. Monitoring:
   - Log D(real), D(fake) (should hover around 0.5)
   - Generate fixed z samples every epoch
   - Watch for mode collapse (repetitive samples)
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Build DCGAN architecture from scratch
- Implement alternating training loop
- Train on MNIST and CIFAR-10
- Visualize generated samples

### Level 3: Paper Implementation (paper/)
- Full DCGAN with training tricks
- WGAN with Gradient Penalty
- Conditional GAN (class-conditional)
- Simplified StyleGAN with style modulation
- FID/IS evaluation

---

## Training Loop

```python
# Pseudocode
for epoch in epochs:
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)

        # ========== Train Discriminator ==========
        # Real images
        real_labels = torch.ones(batch_size, 1) * 0.9  # Label smoothing
        output_real = D(real_images)
        loss_D_real = BCE(output_real, real_labels)

        # Fake images
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        fake_labels = torch.zeros(batch_size, 1)
        output_fake = D(fake_images.detach())  # Detach to avoid G gradients
        loss_D_fake = BCE(output_fake, fake_labels)

        # Total D loss
        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ========== Train Generator ==========
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        output = D(fake_images)
        real_labels = torch.ones(batch_size, 1)
        loss_G = BCE(output, real_labels)  # G wants D(fake)→1

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
```

---

## Sampling

```python
# Generate new images
z = torch.randn(64, latent_dim)  # Batch of 64
with torch.no_grad():
    fake_images = G(z)

# Denormalize from [-1, 1] to [0, 1]
fake_images = (fake_images + 1) / 2

# For conditional GAN
labels = torch.randint(0, 10, (64,))  # Generate 64 samples from 10 classes
fake_images = G(z, labels)
```

---

## Learning Checklist

- [ ] Understand minimax game formulation
- [ ] Implement DCGAN architecture guidelines
- [ ] Master alternating training loop
- [ ] Recognize mode collapse symptoms
- [ ] Implement label smoothing and spectral normalization
- [ ] Understand WGAN and gradient penalty
- [ ] Compute FID and Inception Score
- [ ] Implement conditional GAN

---

## References

- Goodfellow et al. (2014). "Generative Adversarial Networks"
- Radford et al. (2015). "Unsupervised Representation Learning with Deep Convolutional GANs"
- Arjovsky et al. (2017). "Wasserstein GAN"
- Gulrajani et al. (2017). "Improved Training of Wasserstein GANs"
- Miyato et al. (2018). "Spectral Normalization for GANs"
- Karras et al. (2019). "A Style-Based Generator Architecture for GANs"
- [../Deep_Learning/15_GAN.md](../Deep_Learning/15_GAN.md)

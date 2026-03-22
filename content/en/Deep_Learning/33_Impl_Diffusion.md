[Previous: Diffusion Models](./32_Diffusion_Models.md) | [Next: CLIP and Multimodal Learning](./34_CLIP_Multimodal.md)

---

# 33. Diffusion Models (DDPM)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the forward diffusion process — the Markov chain that gradually adds Gaussian noise — and derive the closed-form expression q(x_t | x_0) using the cumulative noise schedule.
2. Formulate the reverse process and explain how a neural network is trained to predict and remove noise at each timestep.
3. Derive the simplified DDPM training objective (noise prediction loss) from the ELBO and explain why this parameterization works in practice.
4. Implement the DDPM training loop and inference (reverse diffusion sampling) from scratch in PyTorch using a U-Net noise predictor.
5. Implement and compare noise schedules (linear and cosine) and explain how the choice of schedule affects generation quality.
6. Compare DDPM with GAN and VAE in terms of sample quality, training stability, inference speed, and likelihood estimation.

---

## Overview

Denoising Diffusion Probabilistic Models (DDPM) are powerful generative models that learn to generate data by reversing a gradual noising process. "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

---

## Mathematical Background

### 1. Forward Diffusion Process

```
Goal: Gradually add Gaussian noise to data x₀

q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

Where:
- x₀: original data
- xₜ: noisy data at timestep t
- βₜ: noise schedule (β₁, ..., βₜ)
- T: total timesteps (typically 1000)

Closed form (using αₜ = 1 - βₜ, ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ):
q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)

xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε,  ε ~ N(0, I)

As t → T: xₜ → N(0, I) (pure noise)
```

### 2. Reverse Diffusion Process

```
Goal: Learn to denoise p(xₜ₋₁|xₜ)

True posterior (intractable):
q(xₜ₋₁|xₜ, x₀) = N(xₜ₋₁; μ̃ₜ(xₜ, x₀), β̃ₜI)

Where:
μ̃ₜ(xₜ, x₀) = (√ᾱₜ₋₁ βₜ)/(1-ᾱₜ) x₀ + (√αₜ(1-ᾱₜ₋₁))/(1-ᾱₜ) xₜ
β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ) · βₜ

Learned reverse process:
pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))

Simplified: predict noise ε instead of mean
εθ(xₜ, t) ≈ ε
```

### 3. Training Objective

```
Variational Lower Bound (ELBO):
L = Eₜ,x₀,ε[||ε - εθ(xₜ, t)||²]

Where:
- t ~ Uniform(1, T)
- x₀ ~ q(x₀)
- ε ~ N(0, I)
- xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε

Simple MSE loss on predicted noise!

┌─────────────────────────────────────────┐
│  Training:                              │
│  1. Sample x₀, t, ε                     │
│  2. Create xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε     │
│  3. Predict ε̂ = εθ(xₜ, t)              │
│  4. Loss = ||ε - ε̂||²                  │
└─────────────────────────────────────────┘
```

### 4. Sampling (Generation)

```
Start from xₜ ~ N(0, I)

For t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0

    ε̂ = εθ(xₜ, t)

    xₜ₋₁ = 1/√αₜ (xₜ - (1-αₜ)/√(1-ᾱₜ) ε̂) + σₜz

Where:
σₜ = √β̃ₜ or √βₜ (variance schedule)

Final: x₀ is the generated sample
```

---

## DDPM Architecture

### UNet with Time Embedding

```
Time Embedding (Sinusoidal Positional Encoding):
t (scalar)
    ↓
PE(t, dim) = [sin(t/10000^(0/d)), cos(t/10000^(0/d)),
              sin(t/10000^(2/d)), cos(t/10000^(2/d)), ...]
    ↓
Linear(dim→4*dim) + SiLU + Linear(4*dim→4*dim)
    ↓
time_emb (broadcast to spatial dimensions)


UNet Structure (e.g., 32×32×3 images):

Input xₜ (32×32×3) + time_emb
    ↓
┌─────────────────────────────────────────┐
│  Encoder (Downsampling)                 │
├─────────────────────────────────────────┤
│ Conv(3→64) + TimeEmb + ResBlock         │ → skip1
│     ↓ Downsample                        │
│ Conv(64→128) + TimeEmb + ResBlock       │ → skip2
│     ↓ Downsample                        │
│ Conv(128→256) + TimeEmb + ResBlock      │ → skip3
│     ↓ Downsample                        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Bottleneck                             │
│  Conv(256→512) + Attention + ResBlock   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Decoder (Upsampling)                   │
├─────────────────────────────────────────┤
│     ↑ Upsample + Concat(skip3)          │
│ Conv(512+256→256) + TimeEmb + ResBlock  │
│     ↑ Upsample + Concat(skip2)          │
│ Conv(256+128→128) + TimeEmb + ResBlock  │
│     ↑ Upsample + Concat(skip1)          │
│ Conv(128+64→64) + TimeEmb + ResBlock    │
└─────────────────────────────────────────┘
    ↓
Conv(64→3) + GroupNorm
    ↓
Output εθ(xₜ, t) (32×32×3)
```

### ResBlock with Time Embedding

```
x, time_emb → ResBlock → out

┌─────────────────────────────────────────┐
│  GroupNorm → SiLU → Conv                │
│       ↓                                 │
│  + time_emb (broadcast)                 │
│       ↓                                 │
│  GroupNorm → SiLU → Conv                │
│       ↓                                 │
│  + skip connection (with projection)    │
└─────────────────────────────────────────┘
```

---

## Noise Schedule

### Linear Schedule

```python
# Linear schedule (Ho et al., 2020)
β₁ = 1e-4
βₜ = 0.02
βₜ = linear_interpolate(β₁, βₜ, t/T)

# Precompute for efficiency
αₜ = 1 - βₜ
ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ
√ᾱₜ, √(1-ᾱₜ)  # Used in forward process
```

### Cosine Schedule (Improved)

```python
# Cosine schedule (Nichol & Dhariwal, 2021)
s = 0.008
f(t) = cos²((t/T + s)/(1 + s) · π/2)
ᾱₜ = f(t) / f(0)
βₜ = 1 - αₜ/αₜ₋₁

# Smoother noise schedule, better for high resolution
```

---

## File Structure

```
13_Diffusion/
├── README.md
├── pytorch_lowlevel/
│   ├── ddpm_mnist.py         # DDPM on MNIST (28×28)
│   └── ddpm_cifar.py         # DDPM on CIFAR-10 (32×32)
├── paper/
│   ├── ddpm_paper.py         # Full DDPM implementation
│   ├── ddim_sampling.py      # DDIM faster sampling
│   └── cosine_schedule.py    # Improved noise schedule
└── exercises/
    ├── 01_noise_schedule.md  # Visualize noise schedules
    └── 02_sampling_steps.md  # Compare DDPM vs DDIM
```

---

## Core Concepts

### 1. DDPM vs DDIM Sampling

```
DDPM (Ho et al., 2020):
- Stochastic sampling (adds noise z at each step)
- Requires T steps (e.g., 1000 steps)
- High quality but slow

DDIM (Song et al., 2020):
- Deterministic sampling (z = 0)
- Skip timesteps: use subset [τ₁, τ₂, ..., τₛ]
- 10-50x faster (e.g., 50 steps)
- Slight quality drop

DDIM update:
xₜ₋₁ = √ᾱₜ₋₁ x̂₀ + √(1-ᾱₜ₋₁) εθ(xₜ, t)

Where x̂₀ = (xₜ - √(1-ᾱₜ)εθ(xₜ, t))/√ᾱₜ
```

### 2. Classifier Guidance

```
Goal: Generate samples conditioned on class y

Conditional score:
∇ₓ log p(xₜ|y) ≈ ∇ₓ log p(xₜ) + s·∇ₓ log p(y|xₜ)
                  ─────────────   ─────────────────
                  Unconditional   Classifier gradient

Guided noise prediction:
ε̂ = εθ(xₜ, t) - s·√(1-ᾱₜ)·∇ₓ log pφ(y|xₜ)

s: guidance scale (s > 1 → stronger conditioning)
```

### 3. Classifier-Free Guidance

```
No separate classifier needed!

Train model to handle both conditional and unconditional:
εθ(xₜ, t, c) with probability p
εθ(xₜ, t, ∅) with probability 1-p (∅ = null class)

Guided prediction:
ε̂ = εθ(xₜ, t, ∅) + w·(εθ(xₜ, t, c) - εθ(xₜ, t, ∅))

w: guidance weight (w=0 → unconditional, w>1 → stronger)

Used in: Stable Diffusion, DALL-E 2, Imagen
```

### 4. Training Tips

```
1. EMA (Exponential Moving Average):
   - Maintain θ_ema = 0.9999·θ_ema + 0.0001·θ
   - Use θ_ema for sampling

2. Progressive Training:
   - Start with smaller resolution
   - Gradually increase (8×8 → 16×16 → 32×32)

3. Data Augmentation:
   - Random horizontal flip
   - Normalize to [-1, 1]

4. Learning Rate:
   - 2e-4 for MNIST/CIFAR
   - 1e-4 for high resolution

5. Batch Size:
   - 128-256 for small images
   - 32-64 for large images
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Implement forward/reverse diffusion
- Implement noise schedule (linear)
- Build UNet with time embedding
- Train on MNIST (28×28) and CIFAR-10 (32×32)

### Level 3: Paper Implementation (paper/)
- Full DDPM with cosine schedule
- DDIM sampling (faster inference)
- Classifier-free guidance
- FID/IS evaluation metrics

---

## Training Loop

```python
# Pseudocode
for epoch in epochs:
    for x0, _ in dataloader:
        # Sample random timestep
        t = torch.randint(1, T+1, (batch_size,))

        # Sample noise
        noise = torch.randn_like(x0)

        # Forward diffusion: create noisy image
        xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise

        # Predict noise
        noise_pred = model(xt, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Sampling Loop

```python
# DDPM sampling
x = torch.randn(batch_size, 3, 32, 32)  # Start from noise

for t in reversed(range(1, T+1)):
    # Predict noise
    t_batch = torch.full((batch_size,), t)
    noise_pred = model(x, t_batch)

    # Compute mean
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    mean = (x - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_t)

    # Add noise (except last step)
    if t > 1:
        noise = torch.randn_like(x)
        sigma_t = sqrt(beta[t])
        x = mean + sigma_t * noise
    else:
        x = mean

# x is the generated image
```

---

## Learning Checklist

- [ ] Understand forward diffusion closed-form
- [ ] Derive reverse diffusion from ELBO
- [ ] Implement noise schedules (linear, cosine)
- [ ] Build UNet with time embedding
- [ ] Understand DDPM vs DDIM sampling
- [ ] Implement classifier-free guidance
- [ ] Calculate FID score for evaluation

---

## References

- Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
- Song et al. (2020). "Denoising Diffusion Implicit Models"
- Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models"
- Ho & Salimans (2022). "Classifier-Free Diffusion Guidance"
- [32_Diffusion_Models.md](./32_Diffusion_Models.md)

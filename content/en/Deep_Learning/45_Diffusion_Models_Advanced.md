[Previous: Test-Time Adaptation](./44_Test_Time_Adaptation.md) | [Next: State Space Models](./46_State_Space_Models.md)

---

# 45. Diffusion Models — Advanced Topics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain classifier-free guidance and its effect on sample quality vs diversity
2. Describe DDIM and other accelerated sampling methods that reduce generation steps
3. Understand the Latent Diffusion / Stable Diffusion architecture end-to-end
4. Apply conditional generation techniques including ControlNet and IP-Adapter
5. Connect score-based generative models with the SDE framework
6. Implement efficient samplers (DPM-Solver) and fine-tuning methods (LoRA, DreamBooth)
7. Explain consistency models and flow matching as next-generation approaches

---

## Table of Contents

1. [Classifier-Free Guidance](#1-classifier-free-guidance)
2. [DDIM and Accelerated Sampling](#2-ddim-and-accelerated-sampling)
3. [Latent Diffusion Models](#3-latent-diffusion-models)
4. [ControlNet and Conditional Generation](#4-controlnet-and-conditional-generation)
5. [Score-Based Generative Models (SDE Perspective)](#5-score-based-generative-models-sde-perspective)
6. [DPM-Solver and Efficient Samplers](#6-dpm-solver-and-efficient-samplers)
7. [Fine-Tuning: LoRA, DreamBooth, Textual Inversion](#7-fine-tuning-lora-dreambooth-textual-inversion)
8. [Consistency Models](#8-consistency-models)
9. [Flow Matching](#9-flow-matching)
10. [Exercises](#10-exercises)

---

## 1. Classifier-Free Guidance

### 1.1 Background: Classifier Guidance

In the original classifier guidance (Dhariwal & Nichol, 2021), a pretrained classifier p(y|x_t) steers the reverse process toward a target class y:

```
Guided score = unconditional score + s * ∇_{x_t} log p(y | x_t)

where s is the guidance scale
```

**Problem**: Requires a separate classifier trained on noisy inputs at every timestep.

### 1.2 Classifier-Free Guidance (CFG)

Ho & Salimans (2022) eliminated the external classifier by training one model in both conditional and unconditional modes:

```
During training:
  - With probability p_uncond (e.g., 10%), drop the conditioning signal c
    (replace with a null token ∅)
  - Otherwise, train normally with condition c

During sampling:
  ε_guided = ε_uncond + w * (ε_cond - ε_uncond)

  where w is the guidance scale (typically 3-15)
```

```
Guidance scale effect:

w = 1.0: No guidance (pure conditional model)
         Low quality, high diversity

w = 7.5: Moderate guidance (common default)
         Good balance of quality and diversity

w = 20:  Strong guidance
         High quality, low diversity (can become oversaturated)
```

### 1.3 PyTorch Implementation

```python
import torch
import torch.nn as nn


class CFGDiffusionModel(nn.Module):
    """Diffusion model with classifier-free guidance support."""

    def __init__(self, base_model, p_uncond=0.1):
        super().__init__()
        self.base_model = base_model  # UNet that takes (x_t, t, cond)
        self.p_uncond = p_uncond

    def forward(self, x_t, t, cond):
        """Training forward: randomly drop conditioning."""
        if self.training:
            # Randomly replace conditioning with null for unconditional training
            batch_size = x_t.shape[0]
            mask = torch.rand(batch_size, device=x_t.device) < self.p_uncond
            # Replace conditioned embeddings with zeros (null token)
            cond = cond.clone()
            cond[mask] = 0.0
        return self.base_model(x_t, t, cond)

    @torch.no_grad()
    def guided_sample(self, x_t, t, cond, guidance_scale=7.5):
        """Sampling with classifier-free guidance."""
        # Unconditional prediction
        null_cond = torch.zeros_like(cond)
        eps_uncond = self.base_model(x_t, t, null_cond)

        # Conditional prediction
        eps_cond = self.base_model(x_t, t, cond)

        # Guided prediction
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return eps_guided
```

### 1.4 Dynamic and Rescaled Guidance

Modern systems use **dynamic guidance** to avoid artifacts:

```python
def dynamic_cfg(eps_uncond, eps_cond, guidance_scale, rescale=0.7):
    """CFG with rescaling to prevent oversaturation (Imagen-style)."""
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # Rescale to prevent color saturation
    std_guided = eps_guided.std(dim=list(range(1, eps_guided.ndim)), keepdim=True)
    std_cond = eps_cond.std(dim=list(range(1, eps_cond.ndim)), keepdim=True)
    factor = std_cond / (std_guided + 1e-8)
    factor = rescale * factor + (1 - rescale)

    return eps_guided * factor
```

---

## 2. DDIM and Accelerated Sampling

### 2.1 The Problem with DDPM Sampling

DDPM requires T steps (typically 1000) for generation — extremely slow:

```
DDPM: x_T → x_{T-1} → x_{T-2} → ... → x_1 → x_0   (1000 NFE)

NFE = Number of Function Evaluations (neural network forward passes)
Each step requires one UNet forward pass (~0.1s on A100 for 512x512)
Total: ~100 seconds per image
```

### 2.2 DDIM: Denoising Diffusion Implicit Models

Song et al. (2021) showed that the DDPM forward process can be generalized to a **non-Markovian** process, enabling deterministic sampling with fewer steps:

```
DDIM update rule:

x_{t-1} = √(ᾱ_{t-1}) * predicted_x0
         + √(1 - ᾱ_{t-1} - σ²_t) * predicted_direction
         + σ_t * noise

where:
  predicted_x0 = (x_t - √(1 - ᾱ_t) * ε_θ(x_t, t)) / √(ᾱ_t)
  predicted_direction = ε_θ(x_t, t)
  σ_t = 0 gives deterministic sampling (DDIM)
  σ_t = √(β̃_t) gives stochastic sampling (DDPM)
```

### 2.3 DDIM Implementation

```python
import torch
import numpy as np


class DDIMSampler:
    """DDIM sampler with configurable step count."""

    def __init__(self, model, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.num_train_timesteps = num_train_timesteps

        # Precompute noise schedule
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_timestep_subsequence(self, num_inference_steps):
        """Select evenly spaced timesteps from the training schedule."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        return torch.from_numpy(timesteps).flip(0)  # Reverse: T -> 0

    @torch.no_grad()
    def sample(self, shape, num_inference_steps=50, eta=0.0, cond=None,
               guidance_scale=7.5):
        """
        Generate samples using DDIM.

        Args:
            shape: (B, C, H, W) output shape
            num_inference_steps: number of denoising steps (e.g., 20-50)
            eta: 0.0 = deterministic DDIM, 1.0 = DDPM-equivalent
            cond: conditioning signal (text embeddings, class labels, etc.)
            guidance_scale: CFG scale
        """
        device = next(self.model.parameters()).device
        timesteps = self.get_timestep_subsequence(num_inference_steps).to(device)

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(shape[0])

            # Get model prediction (with optional CFG)
            if cond is not None and guidance_scale > 1.0:
                eps_pred = self.model.guided_sample(
                    x_t, t_batch, cond, guidance_scale
                )
            else:
                eps_pred = self.model(x_t, t_batch, cond)

            # DDIM update
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0)
            )

            # Predict x_0
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)  # Clip for stability

            # Compute sigma for stochasticity
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred

            # DDIM step
            x_t = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

            if sigma_t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + sigma_t * noise

        return x_t
```

### 2.4 Step Count vs Quality

```
Steps:  Quality (FID↓):  Time (512x512, A100):
1000    ~3.2              ~100s    (DDPM baseline)
 200    ~3.5              ~20s
  50    ~4.0              ~5s      (DDIM sweet spot)
  20    ~5.5              ~2s
  10    ~12.0             ~1s      (noticeable degradation)
   1    ~50+              ~0.1s    (requires consistency distillation)
```

---

## 3. Latent Diffusion Models

### 3.1 Motivation: Pixel Space Is Expensive

Running diffusion in pixel space on high-resolution images is computationally prohibitive:

```
Pixel space (256×256×3):  196,608 dimensions
Latent space (32×32×4):    4,096 dimensions   (48× compression)

Pixel space (512×512×3):  786,432 dimensions
Latent space (64×64×4):   16,384 dimensions   (48× compression)
```

### 3.2 Stable Diffusion Architecture

```
Stable Diffusion Pipeline:

Text Prompt ──► CLIP Text Encoder ──► Text Embeddings (77×768)
                                            │
                                            ▼
Random Noise ──► ┌──────────────────────────────────┐
  (64×64×4)      │  U-Net (in latent space)         │
                 │  - Cross-attention for text cond  │
                 │  - Self-attention for spatial     │
                 │  - ResNet blocks for features     │ × N denoising steps
                 └──────────────────────────────────┘
                                            │
                                            ▼
                 Denoised Latent (64×64×4)
                                            │
                                            ▼
                 VAE Decoder ──► Image (512×512×3)
```

### 3.3 Key Components

```python
class LatentDiffusionModel(nn.Module):
    """Simplified Latent Diffusion Model (LDM) architecture."""

    def __init__(self, vae, unet, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.vae = vae              # Pretrained VAE (encoder + decoder)
        self.unet = unet            # Conditional U-Net in latent space
        self.text_encoder = text_encoder  # CLIP text encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler  # DDIM, DPM-Solver, etc.
        self.vae_scale_factor = 0.18215  # Scaling factor for VAE latents

    def encode_prompt(self, prompt):
        """Encode text prompt to embeddings."""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(tokens.input_ids)[0]
        return text_embeddings

    def encode_image(self, image):
        """Encode image to latent space via VAE."""
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae_scale_factor

    def decode_latents(self, latents):
        """Decode latents to pixel space via VAE."""
        latents = latents / self.vae_scale_factor
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5,
                 height=512, width=512):
        """Full text-to-image generation pipeline."""
        device = self.unet.device

        # 1. Encode text
        text_emb = self.encode_prompt(prompt).to(device)
        uncond_emb = self.encode_prompt("").to(device)
        text_emb = torch.cat([uncond_emb, text_emb])  # For CFG

        # 2. Initialize latent noise
        latents = torch.randn(
            (1, 4, height // 8, width // 8), device=device
        )

        # 3. Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 4. Denoising loop
        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)  # For CFG
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=text_emb).sample

            # Classifier-free guidance
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Decode latents to image
        image = self.decode_latents(latents)
        return image
```

### 3.4 Cross-Attention Mechanism

The key to text conditioning in the UNet:

```python
class CrossAttention(nn.Module):
    """Cross-attention between spatial features and text embeddings."""

    def __init__(self, query_dim, context_dim=768, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """
        Args:
            x: spatial features (B, H*W, D)
            context: text embeddings (B, 77, 768)
        """
        B, N, _ = x.shape
        h = self.heads

        q = self.to_q(x).view(B, N, h, -1).transpose(1, 2)
        k = self.to_k(context).view(B, -1, h, k.shape[-1] // h).transpose(1, 2)
        v = self.to_v(context).view(B, -1, h, v.shape[-1] // h).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)
```

---

## 4. ControlNet and Conditional Generation

### 4.1 ControlNet Architecture

ControlNet (Zhang et al., 2023) adds spatial conditioning (edges, depth, pose) to a pretrained diffusion model:

```
ControlNet Architecture:

                    ┌──────────────────┐
Input Condition ──► │ Trainable Copy   │
(e.g., Canny edge)  │ of UNet Encoder  │
                    │ (locked original  │
                    │  + zero convs)    │
                    └────────┬─────────┘
                             │ residuals
                             ▼
                    ┌──────────────────┐
Noisy Latent ─────► │ Frozen Original  │ ──► Denoised Output
+ Text Condition    │ UNet             │
                    └──────────────────┘
```

### 4.2 Zero Convolution

The key innovation: initialize new connections with zero weights so training starts from the pretrained model exactly:

```python
class ZeroConv(nn.Module):
    """Zero-initialized convolution for stable ControlNet training."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlNetBlock(nn.Module):
    """Simplified ControlNet block."""

    def __init__(self, frozen_unet_encoder_block):
        super().__init__()
        # Trainable copy of the frozen encoder block
        self.trainable_copy = copy.deepcopy(frozen_unet_encoder_block)
        # Zero convolution for output
        self.zero_conv = ZeroConv(self.trainable_copy.out_channels)

    def forward(self, control_input):
        h = self.trainable_copy(control_input)
        return self.zero_conv(h)
```

### 4.3 Types of Conditioning

```
Condition Type     Input Example         Use Case
─────────────────────────────────────────────────────
Canny Edges        Edge map              Preserve structure
Depth Map          MiDaS depth           Scene layout
OpenPose           Skeleton keypoints    Human pose control
Segmentation       Semantic masks        Region-based control
Scribble           User drawings         Sketch-to-image
Normal Map         Surface normals       3D-aware generation
IP-Adapter         Reference image       Style/identity transfer
T2I-Adapter        Lightweight conds     Efficient conditioning
```

---

## 5. Score-Based Generative Models (SDE Perspective)

### 5.1 Unifying Framework

Song et al. (2021) showed that both DDPM and score matching can be unified as Stochastic Differential Equations:

```
Forward SDE (adding noise):
  dx = f(x, t) dt + g(t) dw

  where:
    f(x, t) = drift coefficient
    g(t)    = diffusion coefficient
    dw      = Wiener process (Brownian motion)

Reverse SDE (removing noise):
  dx = [f(x, t) - g(t)² ∇_x log p_t(x)] dt + g(t) dw̄

  where:
    ∇_x log p_t(x) = score function (what the network learns)
    dw̄ = reverse Wiener process
```

### 5.2 Two Canonical SDEs

```
Variance Exploding (VE-SDE):     corresponds to SMLD / NCSN
  f(x, t) = 0
  g(t) = σ(t) * √(2 log(σ_max/σ_min))

Variance Preserving (VP-SDE):    corresponds to DDPM
  f(x, t) = -½ β(t) x
  g(t) = √β(t)
```

### 5.3 Probability Flow ODE

A key insight: the reverse SDE has a **deterministic** ODE counterpart (no noise term):

```
Probability Flow ODE:
  dx/dt = f(x, t) - ½ g(t)² ∇_x log p_t(x)

Advantages:
  - Deterministic: same noise → same image (useful for editing)
  - Enables exact likelihood computation
  - Can use fast ODE solvers (not just Euler-Maruyama)
```

```python
from scipy.integrate import solve_ivp


def probability_flow_ode(score_model, x_T, t_start=1.0, t_end=0.0,
                          beta_min=0.1, beta_max=20.0):
    """Solve the probability flow ODE for deterministic sampling."""

    def drift_fn(t, x_flat):
        x = torch.tensor(x_flat, dtype=torch.float32).reshape(1, *shape)
        t_tensor = torch.tensor([t], dtype=torch.float32)

        beta_t = beta_min + t * (beta_max - beta_min)
        with torch.no_grad():
            score = score_model(x, t_tensor)

        drift = -0.5 * beta_t * x - 0.5 * beta_t * score
        return drift.flatten().numpy()

    shape = x_T.shape[1:]
    solution = solve_ivp(
        drift_fn,
        t_span=(t_start, t_end),
        y0=x_T.flatten().numpy(),
        method='RK45',
        rtol=1e-5, atol=1e-5
    )
    return torch.tensor(solution.y[:, -1]).reshape(1, *shape)
```

---

## 6. DPM-Solver and Efficient Samplers

### 6.1 Overview of Fast Samplers

```
Sampler          Steps for Good Quality   Type            Key Idea
──────────────────────────────────────────────────────────────────
DDPM             1000                     Stochastic      Original SDE discretization
DDIM             50                       Deterministic   Non-Markovian skip steps
PNDM             50                       Deterministic   Pseudo numerical methods
DPM-Solver       20                       Deterministic   Exact solution of ODE
DPM-Solver++     15-20                    Both            Multi-step + thresholding
UniPC            10-15                    Deterministic   Unified predictor-corrector
Euler Ancestral  25-30                    Stochastic      Euler method + noise injection
```

### 6.2 DPM-Solver: Exact Diffusion ODE Solver

Lu et al. (2022) derived an **exact** solution to the diffusion ODE using the change-of-variable formula:

```python
class DPMSolverSecondOrder:
    """Simplified DPM-Solver-2 (second-order solver)."""

    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def lambda_t(self, t):
        """Log signal-to-noise ratio."""
        alpha_bar = self.alphas_cumprod[t]
        return 0.5 * torch.log(alpha_bar / (1 - alpha_bar))

    def predict_x0(self, x_t, t, eps_pred):
        """Predict x_0 from noise prediction."""
        alpha_bar = self.alphas_cumprod[t]
        return (x_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

    @torch.no_grad()
    def step(self, x_t, t, t_prev, t_mid=None):
        """One DPM-Solver-2 step (second-order)."""
        eps_t = self.model(x_t, t)
        x0_pred = self.predict_x0(x_t, t, eps_t)

        if t_mid is not None:
            # Second-order: use midpoint
            lambda_t = self.lambda_t(t)
            lambda_mid = self.lambda_t(t_mid)
            lambda_prev = self.lambda_t(t_prev)

            h = lambda_prev - lambda_t
            h_mid = lambda_mid - lambda_t
            r = h_mid / h

            # First-order estimate at midpoint
            alpha_mid = self.alphas_cumprod[t_mid]
            sigma_mid = torch.sqrt(1 - alpha_mid)
            x_mid = (
                torch.sqrt(alpha_mid / self.alphas_cumprod[t]) * x_t
                - sigma_mid * (torch.exp(-h_mid) - 1) * eps_t
            )

            # Second-order correction
            eps_mid = self.model(x_mid, t_mid)
            alpha_prev = self.alphas_cumprod[t_prev]
            sigma_prev = torch.sqrt(1 - alpha_prev)

            x_prev = (
                torch.sqrt(alpha_prev / self.alphas_cumprod[t]) * x_t
                - sigma_prev * (torch.exp(-h) - 1) * eps_t
                - sigma_prev * (0.5 / r) * (torch.exp(-h) - 1) * (eps_mid - eps_t)
            )
            return x_prev
        else:
            # First-order fallback
            alpha_prev = self.alphas_cumprod[t_prev]
            sigma_prev = torch.sqrt(1 - alpha_prev)
            h = self.lambda_t(t_prev) - self.lambda_t(t)
            x_prev = (
                torch.sqrt(alpha_prev / self.alphas_cumprod[t]) * x_t
                - sigma_prev * (torch.exp(-h) - 1) * eps_t
            )
            return x_prev
```

---

## 7. Fine-Tuning: LoRA, DreamBooth, Textual Inversion

### 7.1 LoRA (Low-Rank Adaptation)

LoRA injects trainable low-rank matrices into frozen attention layers:

```
Original weight:  W ∈ R^{d×d}     (frozen)
LoRA update:      ΔW = BA          where B ∈ R^{d×r}, A ∈ R^{r×d}
Effective weight: W' = W + α * BA  (α = scaling factor, r << d)

Typical rank r = 4-64 (vs d = 320-1280 in Stable Diffusion UNet)
Parameters: 2 * d * r vs d * d  →  ~100× fewer trainable params
```

```python
class LoRALinear(nn.Module):
    """LoRA-adapted linear layer."""

    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)

        d_out, d_in = original_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale = alpha / rank

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return original_out + self.scale * lora_out


def inject_lora(unet, rank=4, alpha=1.0, target_modules=("to_q", "to_v")):
    """Inject LoRA into attention layers of a UNet."""
    for name, module in unet.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0]
                attr_name = name.rsplit(".", 1)[1]
                parent = dict(unet.named_modules())[parent_name]
                setattr(parent, attr_name, LoRALinear(module, rank, alpha))
    return unet
```

### 7.2 DreamBooth

Fine-tune the entire model on 3-5 images of a subject with a unique identifier:

```
Training:
  1. Choose a rare token: "a photo of [V] dog"
  2. Fine-tune UNet + text encoder on 3-5 subject images
  3. Apply prior preservation loss to prevent language drift

Prior Preservation Loss:
  L = L_diffusion(subject images, "[V] dog")
    + λ * L_diffusion(class images, "dog")

  where class images are generated by the frozen model
```

```python
def dreambooth_training_step(model, vae, noise_scheduler, text_encoder,
                              subject_batch, class_batch,
                              subject_prompt, class_prompt,
                              prior_weight=1.0):
    """One DreamBooth training step with prior preservation."""
    # Subject loss
    subject_latents = vae.encode(subject_batch).latent_dist.sample() * 0.18215
    noise = torch.randn_like(subject_latents)
    timesteps = torch.randint(0, 1000, (subject_latents.shape[0],), device=subject_latents.device)
    noisy_latents = noise_scheduler.add_noise(subject_latents, noise, timesteps)

    subject_emb = text_encoder(subject_prompt)
    subject_pred = model(noisy_latents, timesteps, subject_emb).sample
    subject_loss = nn.functional.mse_loss(subject_pred, noise)

    # Prior preservation loss
    class_latents = vae.encode(class_batch).latent_dist.sample() * 0.18215
    class_noise = torch.randn_like(class_latents)
    class_timesteps = torch.randint(0, 1000, (class_latents.shape[0],), device=class_latents.device)
    class_noisy = noise_scheduler.add_noise(class_latents, class_noise, class_timesteps)

    class_emb = text_encoder(class_prompt)
    class_pred = model(class_noisy, class_timesteps, class_emb).sample
    class_loss = nn.functional.mse_loss(class_pred, class_noise)

    return subject_loss + prior_weight * class_loss
```

### 7.3 Textual Inversion

Learn a new "word" (embedding vector) to represent a concept:

```
Approach:
  1. Freeze entire model (UNet + text encoder)
  2. Only optimize a single embedding vector v* for token [V]
  3. v* ∈ R^{768}  (CLIP embedding dimension)
  4. Much smaller than LoRA: just one vector

Training: minimize L_diffusion over only v*
Advantage: tiny model size (~3KB), combinable with any prompt
Limitation: less expressive than DreamBooth/LoRA
```

### 7.4 Comparison

```
Method              Trainable Params   Training Data   Quality   Model Size
───────────────────────────────────────────────────────────────────────────
Textual Inversion   ~768 (1 vector)    3-5 images      ★★★       ~3KB
LoRA                ~1-10M             varies          ★★★★      ~10-100MB
DreamBooth          ~860M (full UNet)  3-5 images      ★★★★★     ~2-4GB
DreamBooth+LoRA     ~1-10M             3-5 images      ★★★★½     ~10-100MB
```

---

## 8. Consistency Models

### 8.1 Motivation

Song et al. (2023) proposed **consistency models** that learn to map any point on the ODE trajectory directly to the origin (x_0):

```
Standard diffusion (multi-step):
  x_T → x_{T-1} → ... → x_1 → x_0     (many steps)

Consistency model (single-step):
  x_T ─────────────────────────► x_0    (one step!)
  x_{T/2} ─────────────────────► x_0    (same x_0!)

Key property (self-consistency):
  f(x_t, t) = f(x_{t'}, t')  for any t, t' on the same ODE trajectory
```

### 8.2 Training Approaches

```
1. Consistency Distillation (CD):
   - Start with a pretrained diffusion model
   - Train consistency model to satisfy the self-consistency property
   - Use the ODE solver to find (x_t, x_{t-1}) pairs on the same trajectory
   - Loss: ||f(x_{t+1}, t+1) - f̂(x_t, t)||²
     where f̂ is an EMA of f (stop-gradient target)

2. Consistency Training (CT):
   - Train from scratch (no pretrained model needed)
   - Use the forward process to create noisy pairs
   - Gradually reduce the step size during training
```

```python
class ConsistencyModel(nn.Module):
    """Simplified consistency model."""

    def __init__(self, backbone, sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.backbone = backbone
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def skip_scaling(self, sigma):
        """Output parameterization: c_skip(σ) x + c_out(σ) F(x, σ)."""
        c_skip = self.sigma_min**2 / (sigma**2 + self.sigma_min**2)
        c_out = sigma * self.sigma_min / torch.sqrt(sigma**2 + self.sigma_min**2)
        return c_skip, c_out

    def forward(self, x, sigma):
        """Map noisy input directly to clean output."""
        c_skip, c_out = self.skip_scaling(sigma)
        F = self.backbone(x, sigma)
        return c_skip * x + c_out * F

    @torch.no_grad()
    def single_step_generate(self, z, sigma=80.0):
        """Generate in a single step from noise z."""
        sigma_t = torch.full((z.shape[0],), sigma, device=z.device)
        return self.forward(z, sigma_t)

    @torch.no_grad()
    def multi_step_generate(self, z, sigmas):
        """Multi-step generation for improved quality."""
        x = z
        for i, sigma in enumerate(sigmas):
            sigma_t = torch.full((z.shape[0],), sigma, device=z.device)
            x = self.forward(x, sigma_t)
            if i < len(sigmas) - 1:
                # Re-add noise at next sigma level
                noise = torch.randn_like(x)
                x = x + sigmas[i + 1] * noise
        return x
```

---

## 9. Flow Matching

### 9.1 Core Idea

Flow matching (Lipman et al., 2023) provides a simpler and more stable alternative to score matching. Instead of learning the score function, it learns a **velocity field** that transforms noise into data along straight paths:

```
Score matching (diffusion):
  Learn: ∇_x log p_t(x)        (score function)
  SDE:   dx = [f - g² ∇log p] dt + g dw

Flow matching:
  Learn: v_t(x)                 (velocity field)
  ODE:   dx/dt = v_t(x)        (no stochastic term!)
  Path:  x_t = (1-t) * x_0 + t * x_1    (straight line from data to noise)
```

### 9.2 Conditional Flow Matching (CFM)

```python
class FlowMatchingModel(nn.Module):
    """Flow matching with optimal transport paths."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # Predicts velocity v(x_t, t)

    def forward(self, x_t, t):
        return self.backbone(x_t, t)

    def compute_loss(self, x_0, x_1=None):
        """
        Conditional flow matching loss.
        x_0: data samples
        x_1: noise samples (if None, sample from N(0,I))
        """
        if x_1 is None:
            x_1 = torch.randn_like(x_0)

        # Sample random time
        t = torch.rand(x_0.shape[0], 1, 1, 1, device=x_0.device)

        # Interpolate along straight path (optimal transport)
        x_t = (1 - t) * x_0 + t * x_1

        # Target velocity: direction from data to noise
        target_v = x_1 - x_0

        # Predicted velocity
        pred_v = self.forward(x_t, t.squeeze())

        # Simple MSE loss
        return nn.functional.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def generate(self, z, num_steps=50):
        """Generate samples by integrating the velocity field from t=1 to t=0."""
        dt = -1.0 / num_steps
        x = z

        for i in range(num_steps):
            t = 1.0 - i / num_steps
            t_batch = torch.full((z.shape[0],), t, device=z.device)
            v = self.forward(x, t_batch)
            x = x + v * dt  # Euler integration

        return x
```

### 9.3 Advantages of Flow Matching

```
Property              Score Matching           Flow Matching
───────────────────────────────────────────────────────────────
Training target       Score (∇ log p)          Velocity (v)
Training stability    Can be unstable          More stable
Path geometry         Curved (SDE)             Straight (ODE)
Likelihood            Via ODE conversion       Direct ODE
Sampling              SDE or ODE               ODE only
Step efficiency       Needs 20-50 steps        10-20 steps often sufficient
Used in               DDPM, Stable Diffusion   Stable Diffusion 3, Flux
```

### 9.4 Rectified Flow

Rectified flow (Liu et al., 2023) iteratively straightens the flow paths for even faster sampling:

```
Round 1: Train flow matching model → generate (x_0, x_1) pairs
Round 2: "Reflow" — retrain on the pairs → straighter paths
Round 3: Reflow again → even straighter

After k rounds of reflow, paths are nearly straight
→ can sample with very few Euler steps (1-5)
```

---

## 10. Exercises

### Exercise 1: Classifier-Free Guidance Exploration

Implement a simple CFG experiment with a toy 2D distribution:

```python
"""
Exercise 1: Implement classifier-free guidance on a 2D Gaussian mixture.

Tasks:
1. Create a conditional model that learns to generate 2D points
   for 4 different classes (4 Gaussian clusters)
2. Implement CFG training (random conditioning dropout)
3. Generate samples with different guidance scales (w=1, 3, 7, 15)
4. Plot the generated distributions and observe how guidance
   affects quality vs diversity

Starter code:
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleDiffusion2D(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=256, p_uncond=0.1):
        super().__init__()
        self.p_uncond = p_uncond
        self.class_embed = nn.Embedding(num_classes + 1, 64)  # +1 for null class
        self.null_class = num_classes

        self.net = nn.Sequential(
            nn.Linear(2 + 64 + 1, hidden_dim),  # x, class_emb, t
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_t, t, class_label):
        # TODO: Implement with random unconditional dropout during training
        pass

    def guided_predict(self, x_t, t, class_label, guidance_scale):
        # TODO: Implement CFG sampling
        pass

# TODO: Train the model and visualize results at different guidance scales
```

### Exercise 2: DDIM Sampler

Implement DDIM sampling and compare quality at different step counts:

```python
"""
Exercise 2: DDIM sampling with varying step counts.

Tasks:
1. Given a pretrained DDPM model, implement DDIM sampling
2. Generate images with 10, 20, 50, 100, and 1000 steps
3. Compute and compare generation quality (visual inspection)
4. Measure wall-clock time for each step count
5. Implement both deterministic (η=0) and stochastic (η=1) modes

Expected observations:
- 50 steps should be nearly as good as 1000
- η=0 gives deterministic outputs (same noise → same image)
- η=1 adds stochasticity (same noise → different images)
"""

def ddim_sample(model, alphas_cumprod, shape, num_steps, eta=0.0):
    """
    Implement DDIM sampling.

    Args:
        model: trained noise prediction model ε_θ
        alphas_cumprod: cumulative product of (1-β)
        shape: output shape (B, C, H, W)
        num_steps: number of denoising steps
        eta: stochasticity parameter (0=deterministic, 1=DDPM)

    Returns:
        Generated samples
    """
    # TODO: Implement DDIM sampling
    # 1. Compute timestep subsequence
    # 2. Start from random noise
    # 3. For each timestep:
    #    a. Predict noise
    #    b. Predict x_0
    #    c. Compute sigma based on eta
    #    d. DDIM update step
    pass
```

### Exercise 3: LoRA Fine-Tuning

Implement LoRA from scratch and apply it to a small UNet:

```python
"""
Exercise 3: Implement LoRA and fine-tune a diffusion model.

Tasks:
1. Implement the LoRALinear module from scratch
2. Write a function to inject LoRA into all attention layers
3. Count and compare parameter counts (original vs LoRA)
4. Train LoRA weights on a small dataset (e.g., 10 images)
5. Verify that only LoRA parameters have gradients

Questions to answer:
- How does rank r affect quality and training speed?
- What is the effect of the scaling factor α?
- Which layers benefit most from LoRA (Q, K, V, or output)?
"""

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        # TODO: Implement LoRA wrapper
        pass

    def forward(self, x):
        # TODO: Implement forward with LoRA
        pass

def count_trainable_params(model):
    """Count parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# TODO: Inject LoRA and verify parameter counts
```

### Exercise 4: Flow Matching

Implement flow matching on a 2D dataset:

```python
"""
Exercise 4: Flow matching on Swiss Roll dataset.

Tasks:
1. Generate Swiss Roll data as target distribution
2. Implement conditional flow matching training
3. Train velocity field network
4. Generate samples by integrating ODE from t=1 to t=0
5. Compare Euler vs RK4 integration at different step counts

Bonus:
- Implement rectified flow (1 round of reflow)
- Compare path straightness before and after reflow
"""
import torch
from sklearn.datasets import make_swiss_roll

class VelocityField(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 2D point + time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def flow_matching_loss(model, x_data):
    """
    Compute conditional flow matching loss.
    # TODO: Implement
    """
    pass

def generate_euler(model, num_samples, num_steps=100):
    """
    Generate by Euler integration of the learned velocity field.
    # TODO: Implement
    """
    pass

# TODO: Train and generate
```

---

**Previous**: [Test-Time Adaptation](./44_Test_Time_Adaptation.md) | **Next**: [State Space Models](./46_State_Space_Models.md)

---

*End of Lesson 45*

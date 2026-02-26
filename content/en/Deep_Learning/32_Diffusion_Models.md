[Previous: Variational Autoencoder (VAE)](./31_Impl_VAE.md) | [Next: Diffusion Models (DDPM)](./33_Impl_Diffusion.md)

---

# 32. Diffusion Models

## Learning Objectives

- Understand Diffusion Process theory (Forward/Reverse)
- DDPM (Denoising Diffusion Probabilistic Models) principles
- Score-based Generative Models concepts
- U-Net architecture for Diffusion
- Stable Diffusion core principles
- Classifier-free Guidance
- Simple DDPM PyTorch implementation

---

## 1. Diffusion Process Overview

### Core Idea

```
Gradually add noise to data (Forward Process)
    x_0 → x_1 → x_2 → ... → x_T (pure noise)

Gradually remove noise to generate data (Reverse Process)
    x_T → x_{T-1} → ... → x_0 (clean image)

Key: Learn the Reverse Process with neural network
```

### Visual Understanding

```
Forward (adding noise):
[Clean image] ──t=0──▶ [Slightly noisy] ──t=500──▶ [More noise] ──t=1000──▶ [Complete noise]

Reverse (removing noise):
[Complete noise] ──t=1000──▶ [Slightly clear] ──t=500──▶ [Clearer] ──t=0──▶ [Clean image]
```

---

## 2. DDPM (Denoising Diffusion Probabilistic Models)

### Forward Process (q)

```python
# Forward process: q(x_t | x_{t-1})
# x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon

# Closed form (directly from x_0 to x_t):
# x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

# alpha_t = 1 - beta_t
# alpha_bar_t = prod(alpha_1 * alpha_2 * ... * alpha_t)
```

### Mathematical Definition

```python
import torch

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule (⭐⭐)"""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (better performance) (⭐⭐⭐)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_index_from_list(vals, t, x_shape):
    """Extract t-timestep values for each sample in the batch"""
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

### Forward Diffusion Implementation

```python
class DiffusionSchedule:
    """Diffusion schedule manager (⭐⭐⭐)"""
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Values needed for computation
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # For posterior computation
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward process: sample x_t from x_0 (⭐⭐⭐)

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
```

---

## 3. Noise Prediction Network

### Objective

```
Model predicts noise epsilon added to x_t
epsilon_theta(x_t, t) ≈ epsilon

Loss function:
L = E[||epsilon - epsilon_theta(x_t, t)||^2]
```

### Simple U-Net Structure

```python
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """Time embedding (⭐⭐⭐)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic Conv Block"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SimpleUNet(nn.Module):
    """Simple U-Net for Diffusion (⭐⭐⭐)"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([
            Block(64, 128, time_dim),
            Block(128, 256, time_dim),
            Block(256, 256, time_dim),
        ])

        # Upsampling
        self.ups = nn.ModuleList([
            Block(256, 128, time_dim, up=True),
            Block(128, 64, time_dim, up=True),
            Block(64, 64, time_dim, up=True),
        ])

        # Output
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Downsampling
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        # Upsampling with skip connections
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.output(x)
```

---

## 4. Training Process

### Training Algorithm (DDPM)

```
1. x_0 ~ q(x_0): Sample from data
2. t ~ Uniform(1, T): Random timestep
3. epsilon ~ N(0, I): Random noise
4. x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
5. Loss = ||epsilon - epsilon_theta(x_t, t)||^2
6. Backpropagate and update
```

### Training Code

```python
def train_diffusion(model, schedule, dataloader, epochs=100, lr=1e-4):
    """Train Diffusion model (⭐⭐⭐)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.size(0)

            # Random timestep
            t = torch.randint(0, schedule.timesteps, (batch_size,), device=device).long()

            # Add noise
            noise = torch.randn_like(images)
            x_t = schedule.q_sample(images, t, noise)

            # Predict noise
            noise_pred = model(x_t, t)

            # Compute loss
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

---

## 5. Sampling (Reverse Process)

### DDPM Sampling

```python
@torch.no_grad()
def sample_ddpm(model, schedule, shape, device):
    """DDPM sampling (⭐⭐⭐)

    Start from x_T ~ N(0, I) and generate x_0
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(shape, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        betas_t = get_index_from_list(schedule.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            schedule.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(
            schedule.sqrt_recip_alphas, t, x.shape
        )

        # Predict noise
        noise_pred = model(x, t)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        if i > 0:
            posterior_variance_t = get_index_from_list(
                schedule.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x = model_mean

    return x
```

### DDIM Sampling (Faster)

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, device, num_inference_steps=50, eta=0.0):
    """DDIM sampling (⭐⭐⭐⭐)

    Fast sampling with fewer steps
    eta=0: deterministic, eta=1: same as DDPM
    """
    model.eval()

    # Step interval
    step_size = schedule.timesteps // num_inference_steps
    timesteps = list(range(0, schedule.timesteps, step_size))
    timesteps = list(reversed(timesteps))

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        alpha_cumprod_t = schedule.alphas_cumprod[t]
        alpha_cumprod_prev = schedule.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else 1.0

        # Predict noise
        noise_pred = model(x, t_tensor)

        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Compute direction
        sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)) * \
                     torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev)

        pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise_pred

        # Add noise (when eta > 0)
        noise = torch.randn_like(x) if eta > 0 else 0

        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir + sigma * noise

    return x
```

---

## 6. Score-based Models

### Score Function

```
Score = gradient of log probability
s(x) = ∇_x log p(x)

Score of noisy data:
s_theta(x_t, t) ≈ ∇_{x_t} log p(x_t)
```

### Relationship with DDPM

```python
# Relationship between noise prediction and score in DDPM:
# epsilon_theta(x_t, t) = -sqrt(1 - alpha_bar_t) * s_theta(x_t, t)

# Score prediction → can be converted to noise prediction
```

---

## 7. Stable Diffusion Principles

### Latent Diffusion

```
Diffusion in latent space instead of image space

1. Encoder: Image → Latent representation z
2. Diffusion: Add/remove noise in z
3. Decoder: Latent representation → Image

Advantages:
- Computational efficiency (diffusion at smaller resolution)
- Can generate high-resolution images
```

### Architecture

```
┌──────────────┐
│  Text Prompt │
└──────┬───────┘
       │ CLIP Text Encoder
       ▼
┌──────────────────────────────────────────┐
│              Cross-Attention              │
├──────────────────────────────────────────┤
│                                          │
│  z_T ──▶ U-Net ──▶ z_{T-1} ──▶ ... ──▶ z_0  │
│         (time embedding)                 │
│                                          │
└──────────────────────────────────────────┘
       │ VAE Decoder
       ▼
┌──────────────┐
│    Image     │
└──────────────┘
```

### Conditional Generation (Cross-Attention)

```python
class CrossAttention(nn.Module):
    """Text-Image Cross Attention (⭐⭐⭐⭐)"""
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        # x: image features (batch, hw, dim)
        # context: text embeddings (batch, seq_len, context_dim)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
```

---

## 8. Classifier-free Guidance

### Idea

```
Mix conditional and unconditional generation

epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)

w > 1: Stronger condition (sharper but less diversity)
w = 1: Normal conditional generation
w < 1: Weaker condition
```

### Implementation

```python
def classifier_free_guidance_sample(model, schedule, shape, condition, w=7.5, device='cuda'):
    """Classifier-free Guidance sampling (⭐⭐⭐⭐)"""
    model.eval()

    x = torch.randn(shape, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        # Conditional prediction
        noise_cond = model(x, t, condition)

        # Unconditional prediction (condition = None or empty embedding)
        noise_uncond = model(x, t, None)

        # Guidance
        noise_pred = noise_uncond + w * (noise_cond - noise_uncond)

        # Sampling step (DDPM or DDIM)
        x = sampling_step(x, noise_pred, t, schedule)

    return x
```

### Condition Dropout During Training

```python
def train_with_cfg(model, dataloader, drop_prob=0.1):
    """Training for CFG (condition dropout) (⭐⭐⭐)"""
    for images, conditions in dataloader:
        # With some probability, set condition to None
        mask = torch.rand(images.size(0)) < drop_prob
        conditions[mask] = None  # or empty embedding

        # Normal training...
```

---

## 9. Complete Simple DDPM Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Configuration
image_size = 28
channels = 1
timesteps = 1000
batch_size = 64
epochs = 50
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # [0, 1] → [-1, 1]
])

train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Schedule
schedule = DiffusionSchedule(timesteps=timesteps, beta_schedule='linear')

# Model (simple version)
model = SimpleUNet(in_channels=channels, out_channels=channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(epochs):
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        batch_size = images.size(0)

        # Random timestep
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Add noise (forward process)
        noise = torch.randn_like(images)
        x_t = schedule.q_sample(images, t, noise)

        # Predict noise
        noise_pred = model(x_t, t)

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# Sampling
model.eval()
with torch.no_grad():
    samples = sample_ddpm(model, schedule, (16, channels, image_size, image_size), device)
    samples = (samples + 1) / 2  # [-1, 1] → [0, 1]

# Visualization
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0].cpu(), cmap='gray')
    ax.axis('off')
plt.savefig('diffusion_samples.png')
print("Saved samples: diffusion_samples.png")
```

---

## Summary

### Key Concepts

1. **Forward Process**: Gradual noise addition q(x_t|x_0)
2. **Reverse Process**: Gradual noise removal p(x_{t-1}|x_t)
3. **DDPM**: Learn reverse process by noise prediction
4. **DDIM**: Fast generation with deterministic sampling
5. **Latent Diffusion**: Efficient generation in latent space
6. **CFG**: Control condition strength

### Key Formulas

```
Forward: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

Loss: L = E[||epsilon - epsilon_theta(x_t, t)||^2]

Reverse: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta) + sigma_t * z
```

### Diffusion vs GAN vs VAE

| Characteristic | Diffusion | GAN | VAE |
|---------------|-----------|-----|-----|
| Training stability | Very high | Low | High |
| Image quality | Best | Good | Blurry |
| Sampling speed | Slow | Fast | Fast |
| Mode Coverage | Good | Mode Collapse | Good |
| Density estimation | Possible | Not possible | Possible |

---

## Exercises

### Exercise 1: Visualize the Forward Diffusion Process

Using the `DiffusionSchedule` and `q_sample` method:
1. Load a single MNIST image and apply forward diffusion at timesteps `t = 0, 100, 250, 500, 750, 999`.
2. Plot the six noisy images side by side in a row.
3. For each timestep, compute and print the signal-to-noise ratio: `SNR(t) = alpha_bar_t / (1 - alpha_bar_t)`.
4. Describe the relationship between SNR and image perceptibility. At what timestep does the original image become unrecognizable?

### Exercise 2: Compare Linear vs Cosine Noise Schedules

Using `linear_beta_schedule` and `cosine_beta_schedule`:
1. Plot `beta_t` vs. `t` for both schedules on the same graph.
2. Plot `alpha_bar_t` (cumulative product) vs. `t` for both schedules.
3. Apply both schedules to the same image and compare the progression of noise at `t = 200, 500, 800`.
4. Explain why the cosine schedule is generally preferred: where does the linear schedule "use up" its noise budget too quickly?

### Exercise 3: Train a Simple DDPM on MNIST

Using the provided `SimpleUNet`, `DiffusionSchedule`, and training code:
1. Train for 50 epochs with `timesteps=1000`, `batch_size=64`, `lr=1e-3`.
2. After training, generate 16 samples using `sample_ddpm` and save them as a grid.
3. Monitor and plot the training loss curve.
4. Experiment: what happens if you use only `timesteps=200` during training? Does the model still generate reasonable samples?

### Exercise 4: Implement DDIM Sampling and Compare Speed

After training a DDPM model:
1. Generate 16 samples using `sample_ddpm` (1000 steps) and measure wall-clock time.
2. Generate 16 samples using `sample_ddim` with `num_inference_steps=50` and `eta=0.0`, and measure time.
3. Visually compare the quality of samples from both methods.
4. Experiment with `eta=0.5` and `eta=1.0` in DDIM. Describe how eta controls the trade-off between determinism and sample diversity.

### Exercise 5: Implement Classifier-Free Guidance for a Conditional Model

Extend the DDPM to support class-conditional generation on MNIST:
1. Modify `SimpleUNet` to accept an optional class label `c` (embed it with `nn.Embedding(10, time_dim)` and add it to the time embedding).
2. During training, randomly drop the condition with probability `p=0.1` (replace label with a null embedding index, e.g., index 10).
3. Implement `classifier_free_guidance_sample` that combines conditional and unconditional noise predictions: `noise_guided = noise_uncond + w * (noise_cond - noise_uncond)`.
4. Generate samples for each digit class `0-9` with guidance scales `w=1.0, 3.0, 7.5`. Observe how higher guidance strength affects sample sharpness and class fidelity.

---

## Next Steps

Learn about Attention mechanism in depth in [17_Attention_Deep_Dive.md](./17_Attention_Deep_Dive.md).

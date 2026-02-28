"""
Exercises for Lesson 32: Diffusion Models
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Visualize the Forward Diffusion Process ===
# Problem: Apply forward diffusion at various timesteps and compute SNR.

def exercise_1():
    """Forward diffusion process visualization and SNR analysis."""
    torch.manual_seed(42)

    T = 1000

    # Linear beta schedule
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    def q_sample(x0, t, noise=None):
        """Add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
        sqrt_one_minus = torch.sqrt(1 - alpha_bar[t])
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    # Simulate a simple "image" (1x28x28)
    x0 = torch.randn(1, 1, 28, 28) * 0.5

    timesteps = [0, 100, 250, 500, 750, 999]
    print(f"  {'Timestep':>8} {'alpha_bar':>10} {'SNR':>10} {'Pixel Std':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for t in timesteps:
        ab = alpha_bar[t].item()
        snr = ab / (1 - ab) if ab < 1 else float('inf')
        noisy = q_sample(x0, t)
        pixel_std = noisy.std().item()
        print(f"  {t:8d} {ab:10.4f} {snr:10.4f} {pixel_std:10.4f}")

    print("\n  As t increases: alpha_bar -> 0, SNR -> 0, image becomes pure noise.")
    print("  Image becomes unrecognizable around t=300-500 (SNR < 1).")


# === Exercise 2: Compare Linear vs Cosine Noise Schedules ===
# Problem: Compare how noise budgets are distributed.

def exercise_2():
    """Linear vs cosine noise schedule comparison."""
    T = 1000

    # Linear schedule
    beta_linear = torch.linspace(1e-4, 0.02, T)
    alpha_linear = 1 - beta_linear
    alpha_bar_linear = torch.cumprod(alpha_linear, dim=0)

    # Cosine schedule
    def cosine_beta_schedule(T, s=0.008):
        t = torch.arange(T + 1) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(beta, 0, 0.999)

    beta_cosine = cosine_beta_schedule(T)
    alpha_cosine = 1 - beta_cosine
    alpha_bar_cosine = torch.cumprod(alpha_cosine, dim=0)

    print(f"  {'Timestep':>8} {'Linear alpha_bar':>16} {'Cosine alpha_bar':>16}")
    print(f"  {'-'*8} {'-'*16} {'-'*16}")

    for t in [0, 100, 200, 500, 800, 999]:
        print(f"  {t:8d} {alpha_bar_linear[t].item():16.4f} "
              f"{alpha_bar_cosine[t].item():16.4f}")

    # Where does each schedule hit 50% noise?
    linear_half = (alpha_bar_linear < 0.5).nonzero(as_tuple=True)[0][0].item()
    cosine_half = (alpha_bar_cosine < 0.5).nonzero(as_tuple=True)[0][0].item()

    print(f"\n  50% signal retained at: linear t={linear_half}, cosine t={cosine_half}")
    print("  Linear schedule uses up noise budget too quickly in early steps,")
    print("  wasting training capacity. Cosine distributes noise more evenly.")


# === Exercise 3: Train Simple DDPM ===
# Problem: Train a noise predictor on simple synthetic data.

def exercise_3():
    """Train a simple denoiser (simplified DDPM) on 1D data."""
    torch.manual_seed(42)

    T = 200  # Fewer steps for speed
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # Simple noise predictor (MLP instead of UNet for simplicity)
    class SimpleDenoiser(nn.Module):
        def __init__(self, data_dim=784, time_dim=64):
            super().__init__()
            self.time_embed = nn.Sequential(
                nn.Linear(1, time_dim), nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            self.net = nn.Sequential(
                nn.Linear(data_dim + time_dim, 512), nn.SiLU(),
                nn.Linear(512, 512), nn.SiLU(),
                nn.Linear(512, data_dim),
            )

        def forward(self, x, t):
            t_emb = self.time_embed(t.float().unsqueeze(-1) / T)
            return self.net(torch.cat([x, t_emb], dim=-1))

    # Synthetic data: structured patterns
    n_samples = 2000
    data_dim = 100  # Reduced for speed
    X = torch.randn(n_samples, data_dim) * 0.5

    model = SimpleDenoiser(data_dim=data_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

    losses = []
    for epoch in range(50):
        epoch_loss = 0
        for (x0,) in loader:
            t = torch.randint(0, T, (x0.size(0),))
            noise = torch.randn_like(x0)
            sqrt_ab = torch.sqrt(alpha_bar[t]).unsqueeze(-1)
            sqrt_1_ab = torch.sqrt(1 - alpha_bar[t]).unsqueeze(-1)
            x_noisy = sqrt_ab * x0 + sqrt_1_ab * noise

            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))

    print(f"  Training loss: epoch 1={losses[0]:.4f}, epoch 50={losses[-1]:.4f}")

    # Sample using DDPM
    model.eval()
    with torch.no_grad():
        x = torch.randn(16, data_dim)
        for t_step in reversed(range(T)):
            t = torch.full((16,), t_step, dtype=torch.long)
            noise_pred = model(x, t)
            beta_t = beta[t_step]
            alpha_t = alpha[t_step]
            alpha_bar_t = alpha_bar[t_step]

            x = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_pred)
            if t_step > 0:
                x += torch.sqrt(beta_t) * torch.randn_like(x)

    print(f"  Generated sample stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    print(f"  Original data stats:    mean={X.mean().item():.4f}, std={X.std().item():.4f}")


# === Exercise 4: DDIM Sampling Speed Comparison ===
# Problem: Compare DDPM (T steps) vs DDIM (fewer steps).

def exercise_4():
    """DDIM vs DDPM sampling speed comparison."""
    import time
    torch.manual_seed(42)

    T = 1000
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    data_dim = 50
    model = nn.Sequential(
        nn.Linear(data_dim + 1, 128), nn.SiLU(),
        nn.Linear(128, data_dim),
    )

    def predict_noise(model, x, t_idx):
        t_emb = torch.full((x.size(0), 1), t_idx / T)
        return model(torch.cat([x, t_emb], dim=-1))

    # DDPM: full T steps
    def sample_ddpm(model, n, steps=T):
        x = torch.randn(n, data_dim)
        with torch.no_grad():
            for t in reversed(range(steps)):
                noise_pred = predict_noise(model, x, t)
                bt = beta[t]
                at = alpha[t]
                ab_t = alpha_bar[t]
                x = (1 / at.sqrt()) * (x - bt / (1 - ab_t).sqrt() * noise_pred)
                if t > 0:
                    x += bt.sqrt() * torch.randn_like(x)
        return x

    # DDIM: stride over timesteps
    def sample_ddim(model, n, num_steps=50, eta=0.0):
        step_size = T // num_steps
        timesteps = list(range(0, T, step_size))[::-1]
        x = torch.randn(n, data_dim)
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                noise_pred = predict_noise(model, x, t)
                ab_t = alpha_bar[t]
                ab_prev = alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

                pred_x0 = (x - (1 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()
                sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
                dir_xt = (1 - ab_prev - sigma ** 2).sqrt() * noise_pred
                x = ab_prev.sqrt() * pred_x0 + dir_xt
                if i + 1 < len(timesteps):
                    x += sigma * torch.randn_like(x)
        return x

    n = 16

    t0 = time.time()
    samples_ddpm = sample_ddpm(model, n, steps=1000)
    time_ddpm = time.time() - t0

    t0 = time.time()
    samples_ddim = sample_ddim(model, n, num_steps=50, eta=0.0)
    time_ddim = time.time() - t0

    print(f"  DDPM (1000 steps): {time_ddpm:.3f}s")
    print(f"  DDIM (50 steps):   {time_ddim:.3f}s")
    print(f"  Speedup:           {time_ddpm / time_ddim:.1f}x")

    # Different eta values
    for eta in [0.0, 0.5, 1.0]:
        samples = sample_ddim(model, n, num_steps=50, eta=eta)
        print(f"  DDIM eta={eta}: std={samples.std().item():.4f}")

    print("\n  eta=0: deterministic (same noise -> same output)")
    print("  eta=1: equivalent to DDPM (full stochasticity)")
    print("  eta between: controls diversity-quality trade-off")


# === Exercise 5: Classifier-Free Guidance ===
# Problem: Implement conditional diffusion with CFG.

def exercise_5():
    """Classifier-free guidance for conditional generation."""
    torch.manual_seed(42)

    T = 200
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    data_dim = 50
    n_classes = 10

    class ConditionalDenoiser(nn.Module):
        def __init__(self, data_dim, n_classes, time_dim=32):
            super().__init__()
            self.time_embed = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU())
            self.class_embed = nn.Embedding(n_classes + 1, time_dim)  # +1 for null class
            self.net = nn.Sequential(
                nn.Linear(data_dim + time_dim * 2, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, data_dim),
            )

        def forward(self, x, t, c):
            t_emb = self.time_embed(t.float().unsqueeze(-1) / T)
            c_emb = self.class_embed(c)
            return self.net(torch.cat([x, t_emb, c_emb], dim=-1))

    model = ConditionalDenoiser(data_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic conditional data
    X = torch.randn(2000, data_dim)
    y = torch.randint(0, n_classes, (2000,))
    # Make data class-dependent
    for c in range(n_classes):
        mask = y == c
        X[mask, c * 5:(c + 1) * 5] += 2.0  # Class signature

    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    null_class = n_classes  # Index 10 is "no class"

    for epoch in range(30):
        for x0, c in loader:
            t = torch.randint(0, T, (x0.size(0),))
            noise = torch.randn_like(x0)
            ab = alpha_bar[t].unsqueeze(-1)
            x_noisy = ab.sqrt() * x0 + (1 - ab).sqrt() * noise

            # Random condition dropout (p=0.1)
            drop_mask = torch.rand(c.size(0)) < 0.1
            c_input = c.clone()
            c_input[drop_mask] = null_class

            noise_pred = model(x_noisy, t, c_input)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # CFG sampling
    model.eval()

    def cfg_sample(model, target_class, n=16, guidance_scale=3.0):
        x = torch.randn(n, data_dim)
        with torch.no_grad():
            for t_step in reversed(range(T)):
                t = torch.full((n,), t_step, dtype=torch.long)
                c_cond = torch.full((n,), target_class, dtype=torch.long)
                c_uncond = torch.full((n,), null_class, dtype=torch.long)

                noise_cond = model(x, t, c_cond)
                noise_uncond = model(x, t, c_uncond)
                noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                bt = beta[t_step]
                at = alpha[t_step]
                ab_t = alpha_bar[t_step]
                x = (1 / at.sqrt()) * (x - bt / (1 - ab_t).sqrt() * noise_guided)
                if t_step > 0:
                    x += bt.sqrt() * torch.randn_like(x)
        return x

    print(f"  CFG sampling with different guidance scales:")
    for w in [1.0, 3.0, 7.5]:
        samples = cfg_sample(model, target_class=3, guidance_scale=w)
        # Check class signature at positions 15-20 (class 3)
        sig_strength = samples[:, 15:20].mean().item()
        print(f"    w={w:.1f}: class-3 signature strength={sig_strength:.4f}")

    print("\n  Higher guidance -> stronger class signal -> sharper but less diverse.")


if __name__ == "__main__":
    print("=== Exercise 1: Forward Diffusion ===")
    exercise_1()
    print("\n=== Exercise 2: Linear vs Cosine Schedule ===")
    exercise_2()
    print("\n=== Exercise 3: Train Simple DDPM ===")
    exercise_3()
    print("\n=== Exercise 4: DDIM vs DDPM ===")
    exercise_4()
    print("\n=== Exercise 5: Classifier-Free Guidance ===")
    exercise_5()
    print("\nAll exercises completed!")

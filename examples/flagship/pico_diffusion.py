"""
Pico Diffusion — Minimal Denoising Diffusion Probabilistic Model

A minimal DDPM implementation that generates 2D point distributions
using a simple MLP denoiser. Demonstrates the forward/reverse diffusion
process without requiring image data or convolutional networks.

Paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

Learning Objectives:
1. Understand forward diffusion: gradually adding noise q(x_t | x_{t-1})
2. Implement the noise schedule (linear beta schedule)
3. Train a noise predictor: epsilon_theta(x_t, t)
4. Sample via reverse diffusion: iterative denoising from pure noise
5. Visualize the denoising trajectory
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DiffusionConfig:
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    data_dim: int = 2
    hidden_dim: int = 256
    time_dim: int = 32
    n_epochs: int = 800
    batch_size: int = 256
    lr: float = 1e-3


# ── Noise schedule ───────────────────────────────────────────────────
class NoiseSchedule:
    """Precomputes all diffusion constants from a linear beta schedule."""

    def __init__(self, cfg: DiffusionConfig, device: torch.device) -> None:
        self.T = cfg.timesteps
        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - torch.cat([torch.tensor([1.0], device=device),
                                            self.alpha_cumprod[:-1]]))
            / (1.0 - self.alpha_cumprod)
        )

    def forward_diffuse(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps."""
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_cumprod[t].unsqueeze(-1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise

    def sample_timesteps(self, n: int) -> Tensor:
        return torch.randint(0, self.T, (n,), device=self.betas.device)


# ── Sinusoidal time embedding ────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """Maps integer timestep to a fixed-dim vector (sin/cos encoding)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── Noise predictor (MLP) ────────────────────────────────────────────
class NoisePredictor(nn.Module):
    """epsilon_theta(x_t, t): predicts the noise added at timestep t."""

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(cfg.time_dim)
        input_dim = cfg.data_dim + cfg.time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.data_dim),
        )
        self.residual_proj = nn.Linear(cfg.data_dim, cfg.data_dim)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_emb(t)
        h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h) + self.residual_proj(x_t)


# ── Orchestrator ──────────────────────────────────────────────────────
class PicoDiffusion:
    """End-to-end DDPM: train a noise predictor, then sample via reverse diffusion."""

    def __init__(self, cfg: DiffusionConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.schedule = NoiseSchedule(cfg, device)
        self.model = NoisePredictor(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.loss_history: List[float] = []

    # ── Training ──────────────────────────────────────────────────────
    def train(self, data: Tensor) -> None:
        self.model.train()
        n = data.shape[0]
        for epoch in range(1, self.cfg.n_epochs + 1):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            steps = 0
            for i in range(0, n, self.cfg.batch_size):
                batch = data[perm[i: i + self.cfg.batch_size]]
                t = self.schedule.sample_timesteps(batch.shape[0])
                x_t, noise = self.schedule.forward_diffuse(batch, t)
                pred_noise = self.model(x_t, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            avg = epoch_loss / max(steps, 1)
            self.loss_history.append(avg)
            if epoch % 200 == 0 or epoch == 1:
                print(f"Epoch {epoch:>4d}/{self.cfg.n_epochs}  loss={avg:.6f}")

    # ── Sampling (reverse diffusion) ──────────────────────────────────
    @torch.no_grad()
    def sample(self, n: int) -> Tuple[Tensor, dict[int, Tensor]]:
        self.model.eval()
        s = self.schedule
        x = torch.randn(n, self.cfg.data_dim, device=self.device)
        snapshots: dict[int, Tensor] = {self.cfg.timesteps: x.clone()}
        for t_val in reversed(range(self.cfg.timesteps)):
            t_batch = torch.full((n,), t_val, device=self.device, dtype=torch.long)
            pred = self.model(x, t_batch)
            alpha_t = s.alphas[t_val]
            alpha_bar_t = s.alpha_cumprod[t_val]
            beta_t = s.betas[t_val]
            coeff = beta_t / s.sqrt_one_minus_alpha_cumprod[t_val]
            x = (1.0 / torch.sqrt(alpha_t)) * (x - coeff * pred)
            if t_val > 0:
                sigma = torch.sqrt(s.posterior_variance[t_val])
                x = x + sigma * torch.randn_like(x)
            if t_val in (150, 100, 50, 0):
                snapshots[t_val] = x.clone()
        return x, snapshots


# ── Data generation ───────────────────────────────────────────────────
def make_spiral(n: int, device: torch.device) -> Tensor:
    """2D Archimedean spiral: r = theta / (4*pi)."""
    theta = torch.linspace(0, 4 * math.pi, n, device=device)
    r = theta / (4 * math.pi)
    x = r * torch.cos(theta) + 0.01 * torch.randn(n, device=device)
    y = r * torch.sin(theta) + 0.01 * torch.randn(n, device=device)
    return torch.stack([x, y], dim=-1)


# ── Visualization ─────────────────────────────────────────────────────
def visualize(
    real: Tensor,
    generated: Tensor,
    snapshots: dict[int, Tensor],
    loss_history: List[float],
    cfg: DiffusionConfig,
    save_path: str,
) -> None:
    real_np = real.cpu().numpy()
    gen_np = generated.cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pico Diffusion — DDPM on 2D Spiral", fontsize=14, fontweight="bold")

    # Panel 1: Denoising trajectory (top-left → split into 5 mini plots)
    ax0 = axes[0, 0]
    ax0.set_visible(False)
    sub_left = fig.add_axes([0.06, 0.54, 0.38, 0.36])
    sub_left.set_visible(False)
    steps = [cfg.timesteps, 150, 100, 50, 0]
    labels = [f"t={s}" for s in steps]
    for idx, s in enumerate(steps):
        inset = fig.add_axes([0.06 + idx * 0.076, 0.56, 0.072, 0.32])
        pts = snapshots[s].cpu().numpy()
        inset.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.5, c="purple")
        inset.set_title(labels[idx], fontsize=8)
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_xlim(-1.5, 1.5); inset.set_ylim(-1.5, 1.5)

    # Panel 2: Generated vs Real (top-right)
    ax1 = axes[0, 1]
    ax1.scatter(real_np[:, 0], real_np[:, 1], s=4, alpha=0.4, c="royalblue", label="Real")
    ax1.scatter(gen_np[:, 0], gen_np[:, 1], s=4, alpha=0.4, c="crimson", label="Generated")
    ax1.legend(fontsize=8); ax1.set_title("Generated vs Real")
    ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")

    # Panel 3: Training loss (bottom-left)
    ax2 = axes[1, 0]
    ax2.plot(loss_history, linewidth=0.8, color="teal")
    ax2.set_yscale("log"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss (log)")
    ax2.set_title("Training Loss"); ax2.grid(True, alpha=0.3)

    # Panel 4: Forward diffusion (bottom-right → 5 mini plots)
    ax3 = axes[1, 1]
    ax3.set_visible(False)
    schedule = NoiseSchedule(cfg, real.device)
    fwd_steps = [0, 50, 100, 150, 200]
    for idx, fs in enumerate(fwd_steps):
        inset = fig.add_axes([0.56 + idx * 0.076, 0.08, 0.072, 0.32])
        if fs == 0:
            pts = real_np
        else:
            t_batch = torch.full((real.shape[0],), min(fs, cfg.timesteps - 1),
                                 device=real.device, dtype=torch.long)
            x_t, _ = schedule.forward_diffuse(real, t_batch)
            pts = x_t.cpu().numpy()
        inset.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.5, c="darkorange")
        inset.set_title(f"t={fs}", fontsize=8)
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_xlim(-1.5, 1.5); inset.set_ylim(-1.5, 1.5)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DiffusionConfig()

    print("Generating spiral data …")
    data = make_spiral(2048, device)

    print(f"Training DDPM ({cfg.n_epochs} epochs) …")
    diffusion = PicoDiffusion(cfg, device)
    diffusion.train(data)

    print("Sampling via reverse diffusion …")
    generated, snapshots = diffusion.sample(1024)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "pico_diffusion.png")
    visualize(data, generated, snapshots, diffusion.loss_history, cfg, save_path)

    final_loss = diffusion.loss_history[-1]
    print(f"Final loss: {final_loss:.6f}")
    # Expected: noise prediction MSE loss < 0.40 after 800 epochs

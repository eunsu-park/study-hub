"""
Tiny GAN — Generative Adversarial Network on 2D Distributions

A minimal GAN implementation that learns to generate 2D point
distributions (ring, grid, Swiss roll) using only fully-connected
networks. Demonstrates the adversarial training paradigm.

Learning Objectives:
1. Understand the Generator vs Discriminator adversarial game
2. Implement minimax loss: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]
3. Observe mode collapse and training dynamics
4. Visualize how generated distributions evolve during training
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

# ── Config ────────────────────────────────────────────────────────────


@dataclass
class GANConfig:
    latent_dim: int = 2
    hidden_dim: int = 128
    data_dim: int = 2
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    n_steps: int = 3000
    batch_size: int = 256


# ── Networks ──────────────────────────────────────────────────────────


class Generator(nn.Module):
    """z(2) -> FC(128) -> ReLU -> FC(128) -> ReLU -> FC(2)"""

    def __init__(self, cfg: GANConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.data_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """x(2) -> FC(128) -> LeakyReLU(0.2) -> FC(128) -> LeakyReLU(0.2) -> FC(1)"""

    def __init__(self, cfg: GANConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.data_dim, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data generators ──────────────────────────────────────────────────


def make_ring(n: int, noise: float = 0.1) -> torch.Tensor:
    """Points on unit circle + Gaussian noise."""
    theta = torch.linspace(0, 2 * math.pi, n + 1)[:n]
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return x + torch.randn_like(x) * noise


def make_grid(n: int, noise: float = 0.05) -> torch.Tensor:
    """Points on a 5x5 grid + Gaussian noise."""
    lin = torch.linspace(-1, 1, 5)
    xx, yy = torch.meshgrid(lin, lin, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (25, 2)
    # Repeat grid to reach n points, then truncate
    repeats = (n // grid.size(0)) + 1
    pts = grid.repeat(repeats, 1)[:n]
    return pts + torch.randn_like(pts) * noise


def make_swissroll(n: int, noise: float = 0.1) -> torch.Tensor:
    """Parametric spiral: (t*cos(t), t*sin(t))."""
    t = torch.linspace(1.5 * math.pi, 4.5 * math.pi, n)
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    pts = torch.stack([x, y], dim=1)
    # Normalise to roughly [-1, 1]
    pts = pts / pts.abs().max()
    return pts + torch.randn_like(pts) * noise


# ── Training ──────────────────────────────────────────────────────────


def train_gan(
    cfg: GANConfig,
    data_fn: callable,
) -> tuple[Generator, Discriminator, dict[str, list[float]], dict[int, torch.Tensor]]:
    """Train GAN and return models, loss history, and snapshot tensors."""
    G = Generator(cfg)
    D = Discriminator(cfg)
    opt_g = Adam(G.parameters(), lr=cfg.lr_g)
    opt_d = Adam(D.parameters(), lr=cfg.lr_d)
    criterion = nn.BCEWithLogitsLoss()

    history: dict[str, list[float]] = {"d_loss": [], "g_loss": []}
    snapshots: dict[int, torch.Tensor] = {}
    snapshot_steps = {1000, cfg.n_steps}

    for step in range(1, cfg.n_steps + 1):
        # ── Discriminator update ──
        real = data_fn(cfg.batch_size)
        z = torch.randn(cfg.batch_size, cfg.latent_dim)
        fake = G(z).detach()

        real_labels = torch.ones(cfg.batch_size, 1)
        fake_labels = torch.zeros(cfg.batch_size, 1)

        d_loss_real = criterion(D(real), real_labels)
        d_loss_fake = criterion(D(fake), fake_labels)
        d_loss = d_loss_real + d_loss_fake

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # ── Generator update ──
        z = torch.randn(cfg.batch_size, cfg.latent_dim)
        fake = G(z)
        g_loss = criterion(D(fake), real_labels)  # G wants D to output 1

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        history["d_loss"].append(d_loss.item())
        history["g_loss"].append(g_loss.item())

        # Snapshots for visualisation
        if step in snapshot_steps:
            with torch.no_grad():
                z_vis = torch.randn(512, cfg.latent_dim)
                snapshots[step] = G(z_vis).cpu()

        if step % 500 == 0:
            print(f"Step {step:>5d}  D_loss={d_loss.item():.4f}  G_loss={g_loss.item():.4f}")

    return G, D, history, snapshots


# ── Visualisation ─────────────────────────────────────────────────────


def visualise(
    cfg: GANConfig,
    G: Generator,
    D: Discriminator,
    history: dict[str, list[float]],
    snapshots: dict[int, torch.Tensor],
    save_path: str,
) -> None:
    """Create 6-panel figure (2 rows x 3 cols) and save as PNG."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Tiny GAN — 2D Distribution Learning", fontsize=14, fontweight="bold")

    # ── Row 1 ─────────────────────────────────────────────────────────
    # Panel (0,0): Real data
    real = make_ring(1024)
    ax = axes[0, 0]
    ax.scatter(real[:, 0], real[:, 1], s=4, alpha=0.5, c="steelblue")
    ax.set_title("Real Data (Ring)")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

    # Panel (0,1): Generated at step 1000
    gen_1k = snapshots[1000]
    ax = axes[0, 1]
    ax.scatter(gen_1k[:, 0], gen_1k[:, 1], s=4, alpha=0.5, c="coral")
    ax.set_title("Generated @ Step 1000")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

    # Panel (0,2): Generated at step 3000
    gen_3k = snapshots[cfg.n_steps]
    ax = axes[0, 2]
    ax.scatter(gen_3k[:, 0], gen_3k[:, 1], s=4, alpha=0.5, c="coral")
    ax.set_title(f"Generated @ Step {cfg.n_steps}")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

    # ── Row 2 ─────────────────────────────────────────────────────────
    # Panel (1,0): Loss curves
    ax = axes[1, 0]
    ax.plot(history["d_loss"], label="D_loss", alpha=0.7, linewidth=0.5)
    ax.plot(history["g_loss"], label="G_loss", alpha=0.7, linewidth=0.5)
    ax.set_title("Training Losses")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()

    # Panel (1,1): Discriminator heatmap
    ax = axes[1, 1]
    grid_pts = torch.linspace(-2, 2, 100)
    xx, yy = torch.meshgrid(grid_pts, grid_pts, indexing="xy")
    grid_input = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        d_out = torch.sigmoid(D(grid_input)).reshape(100, 100).numpy()
    cf = ax.contourf(
        xx.numpy(), yy.numpy(), d_out, levels=20, cmap="RdYlGn"
    )
    fig.colorbar(cf, ax=ax, fraction=0.046)
    ax.set_title("Discriminator Output")
    ax.set_aspect("equal")

    # Panel (1,2): All 3 distributions generated
    ax = axes[1, 2]
    # Train quick generators for grid and swissroll
    colors = ["coral", "mediumseagreen", "mediumpurple"]
    labels = ["Ring", "Grid", "Swiss Roll"]
    data_fns = [make_ring, make_grid, make_swissroll]
    for i, (fn, color, label) in enumerate(zip(data_fns, colors, labels)):
        cfg_quick = GANConfig(n_steps=2000, lr_g=2e-4, lr_d=2e-4)
        G_q, _, _, _ = train_gan(cfg_quick, fn)
        with torch.no_grad():
            z = torch.randn(512, cfg_quick.latent_dim)
            samples = G_q(z).cpu()
        ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.4, c=color, label=label)
    ax.set_title("Generated (All Distributions)")
    ax.legend(markerscale=4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────
# Expected: D_loss oscillates around 0.5-1.5, generated ring matches real by step 2000

if __name__ == "__main__":
    torch.manual_seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = GANConfig()

    print("Training GAN on ring distribution...")
    G, D, history, snapshots = train_gan(cfg, make_ring)

    print("\nGenerating visualisation (includes training auxiliary GANs)...")
    save_path = os.path.join(save_dir, "tiny_gan.png")
    visualise(cfg, G, D, history, snapshots, save_path)
    print("Done.")

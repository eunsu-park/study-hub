"""
Micro VAE — Variational Autoencoder with Latent Space Visualization

A minimal VAE that learns latent representations of synthetic 2D shapes.
Uses a 2D latent space for direct visualization of the learned manifold.
Demonstrates the ELBO, reparameterization trick, and latent space structure.

Learning Objectives:
1. Implement the VAE encoder-decoder architecture
2. Derive and compute the ELBO: E_q[log p(x|z)] - KL(q(z|x) || p(z))
3. Apply the reparameterization trick: z = mu + sigma * epsilon
4. Visualize latent space clustering and generation quality
5. Understand the reconstruction-regularization tradeoff (beta-VAE)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ── Config ────────────────────────────────────────────────────────────


@dataclass
class VAEConfig:
    input_dim: int = 64       # 8x8 flattened
    hidden_dim: int = 128
    latent_dim: int = 2
    beta: float = 1.0         # KL weight (beta-VAE)
    lr: float = 1e-3
    n_epochs: int = 500
    batch_size: int = 64


# ── Networks ──────────────────────────────────────────────────────────


class Encoder(nn.Module):
    """x(64) -> FC(128) -> ReLU -> FC(128) -> ReLU -> mu(2), log_var(2)"""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_log_var = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """z(2) -> FC(128) -> ReLU -> FC(128) -> ReLU -> FC(64) -> Sigmoid"""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MicroVAE(nn.Module):
    """Variational Autoencoder with 2D latent space."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss(
        self, x: torch.Tensor, x_recon: torch.Tensor,
        mu: torch.Tensor, log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / batch_size
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        total = recon_loss + self.cfg.beta * kl_loss
        return total, recon_loss, kl_loss


# ── Data Generation ──────────────────────────────────────────────────


def make_shapes(n_per_class: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 8x8 binary images of 4 shapes with noise and translation."""
    templates: list[np.ndarray] = []

    # Circle: pixels within radius 2.5 of center (3.5, 3.5)
    circle = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            if (r - 3.5) ** 2 + (c - 3.5) ** 2 <= 2.5 ** 2:
                circle[r, c] = 1.0
    templates.append(circle)

    # Square: 3x3 filled block at rows 3-5, cols 3-5
    square = np.zeros((8, 8), dtype=np.float32)
    square[3:6, 3:6] = 1.0
    templates.append(square)

    # Triangle: lower-left triangle (rows 2-6, row >= col shifted)
    triangle = np.zeros((8, 8), dtype=np.float32)
    for r in range(2, 7):
        for c in range(1, r):
            triangle[r, c] = 1.0
    templates.append(triangle)

    # Cross: horizontal bar at rows 3-4, vertical bar at cols 3-4
    cross = np.zeros((8, 8), dtype=np.float32)
    cross[3:5, 1:7] = 1.0
    cross[1:7, 3:5] = 1.0
    templates.append(cross)

    all_data: list[np.ndarray] = []
    all_labels: list[int] = []

    for label, tmpl in enumerate(templates):
        for _ in range(n_per_class):
            img = tmpl.copy()
            # Random small translation: shift 0-1 pixel in each direction
            shift_r = np.random.randint(0, 2)
            shift_c = np.random.randint(0, 2)
            img = np.roll(img, shift_r, axis=0)
            img = np.roll(img, shift_c, axis=1)
            # Random noise: flip probability 0.05
            flip_mask = np.random.random((8, 8)) < 0.05
            img = np.where(flip_mask, 1.0 - img, img)
            all_data.append(img.flatten())
            all_labels.append(label)

    data = torch.tensor(np.array(all_data), dtype=torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return data, labels


# ── Training ──────────────────────────────────────────────────────────


def train(
    cfg: VAEConfig,
    data: torch.Tensor,
) -> tuple[MicroVAE, dict[str, list[float]]]:
    """Train VAE and return model + loss history."""
    model = MicroVAE(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    n = data.size(0)

    history: dict[str, list[float]] = {"total": [], "recon": [], "kl": []}

    for epoch in range(1, cfg.n_epochs + 1):
        perm = torch.randperm(n)
        epoch_total, epoch_recon, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0

        for i in range(0, n, cfg.batch_size):
            batch = data[perm[i : i + cfg.batch_size]]
            x_recon, mu, log_var = model(batch)
            total, recon_loss, kl_loss = model.loss(batch, x_recon, mu, log_var)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_total += total.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        history["total"].append(epoch_total / n_batches)
        history["recon"].append(epoch_recon / n_batches)
        history["kl"].append(epoch_kl / n_batches)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:>4d}  ELBO={history['total'][-1]:.2f}  "
                f"Recon={history['recon'][-1]:.2f}  KL={history['kl'][-1]:.2f}"
            )

    return model, history


# ── Visualization ─────────────────────────────────────────────────────


def visualise(
    model: MicroVAE,
    data: torch.Tensor,
    labels: torch.Tensor,
    history: dict[str, list[float]],
    save_path: str,
) -> None:
    """Create 2x2 figure: latent space, reconstructions, generation, loss curves."""
    model.eval()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Micro VAE — Latent Space Visualization", fontsize=14, fontweight="bold")
    shape_names = ["Circle", "Square", "Triangle", "Cross"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    # ── Panel 1: Latent space scatter ─────────────────────────────────
    ax = axes[0, 0]
    with torch.no_grad():
        mu, _ = model.encode(data)
        mu_np = mu.numpy()
    for i, (name, color) in enumerate(zip(shape_names, colors)):
        mask = (labels == i).numpy()
        ax.scatter(mu_np[mask, 0], mu_np[mask, 1], s=12, alpha=0.6, c=color, label=name)
    ax.set_title("Latent Space (Encoded mu)")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(fontsize=8)

    # ── Panel 2: Reconstructions ──────────────────────────────────────
    ax = axes[0, 1]
    idx = torch.randperm(data.size(0))[:8]
    originals = data[idx]
    with torch.no_grad():
        recons, _, _ = model(originals)
    grid = np.zeros((2 * 8, 8 * 8))
    for j in range(8):
        grid[0:8, j * 8 : (j + 1) * 8] = originals[j].numpy().reshape(8, 8)
        grid[8:16, j * 8 : (j + 1) * 8] = recons[j].numpy().reshape(8, 8)
    ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("Reconstructions (top=orig, bottom=recon)")
    ax.set_xticks([])
    ax.set_yticks([])

    # ── Panel 3: Random generation ────────────────────────────────────
    ax = axes[1, 0]
    with torch.no_grad():
        z_rand = torch.randn(16, model.cfg.latent_dim)
        generated = model.decode(z_rand)
    gen_grid = np.zeros((4 * 8, 4 * 8))
    for j in range(16):
        r, c = divmod(j, 4)
        gen_grid[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8] = (
            generated[j].numpy().reshape(8, 8)
        )
    ax.imshow(gen_grid, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("Random Generations (z ~ N(0, I))")
    ax.set_xticks([])
    ax.set_yticks([])

    # ── Panel 4: Loss curves ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(history["total"], label="ELBO (total)", linewidth=1.2)
    ax.plot(history["recon"], label="Recon loss", linewidth=1.2)
    ax.plot(history["kl"], label="KL divergence", linewidth=1.2)
    ax.set_title("Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────
# Expected: ELBO loss < 20.0 after 500 epochs, latent space shows 4 clusters

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = VAEConfig()

    print("Generating shape data (4 classes x 200 samples)...")
    data, labels = make_shapes(n_per_class=200)
    print(f"Data shape: {data.shape}, Labels: {labels.unique().tolist()}")

    print("\nTraining Micro VAE...")
    model, history = train(cfg, data)

    print(f"\nFinal ELBO: {history['total'][-1]:.2f}")
    save_path = os.path.join(save_dir, "micro_vae.png")
    visualise(model, data, labels, history, save_path)
    print("Done.")

"""
PyTorch Low-Level Variational Autoencoder (VAE) Implementation

Directly implements ELBO and the Reparameterization Trick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VAEConfig:
    """VAE Configuration"""
    image_size: int = 28
    in_channels: int = 1
    latent_dim: int = 20
    hidden_dims: Tuple[int, ...] = (32, 64)
    beta: float = 1.0  # beta-VAE


class Encoder(nn.Module):
    """VAE Encoder: x -> (mu, log sigma^2)"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Convolutional layers
        modules = []
        in_channels = config.in_channels

        for h_dim in config.hidden_dims:
            modules.append(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1)
            )
            modules.append(nn.ReLU())
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Compute final feature map size
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flatten_dim = config.hidden_dims[-1] * self.final_size * self.final_size

        # FC layers for mu and log sigma^2
        self.fc_mu = nn.Linear(self.flatten_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        # Encode
        h = self.encoder(x)
        h = h.flatten(start_dim=1)

        # mu and log sigma^2
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder: z -> x_hat"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flatten_dim = config.hidden_dims[-1] * self.final_size * self.final_size

        # FC layer
        self.fc = nn.Linear(config.latent_dim, self.flatten_dim)

        # Transposed convolutions
        modules = []
        hidden_dims = list(config.hidden_dims)[::-1]  # Reverse order

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.ConvTranspose2d(
                    hidden_dims[i], hidden_dims[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            modules.append(nn.ReLU())

        # Final layer
        modules.append(
            nn.ConvTranspose2d(
                hidden_dims[-1], config.in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )
        modules.append(nn.Sigmoid())  # [0, 1] range

        self.decoder = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, C, H, W)
        """
        h = self.fc(z)
        h = h.view(-1, self.config.hidden_dims[-1], self.final_size, self.final_size)
        x_recon = self.decoder(h)
        return x_recon


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick

        z = mu + sigma * epsilon, where epsilon ~ N(0, I)

        This allows gradients of z to backpropagate through mu and sigma.
        """
        # sigma = exp(log sigma^2 / 2) = exp(logvar / 2)
        std = torch.exp(0.5 * logvar)

        # epsilon ~ N(0, I)
        eps = torch.randn_like(std)

        # z = mu + sigma * epsilon
        z = mu + std * eps

        return z

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: (B, C, H, W)
            return_latent: Whether to return latent z

        Returns:
            x_recon: (B, C, H, W) reconstructed image
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            z: (optional) (B, latent_dim)
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        if return_latent:
            return x_recon, mu, logvar, z

        return x_recon, mu, logvar

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode only"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode only"""
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from latent space to generate images"""
        # Sample from prior: z ~ N(0, I)
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        samples = self.decode(z)
        return samples


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE Loss: Negative ELBO

    L = L_recon + beta * L_KL

    Args:
        x: Original image (B, C, H, W)
        x_recon: Reconstructed image (B, C, H, W)
        mu: Mean (B, latent_dim)
        logvar: Log variance (B, latent_dim)
        beta: KL weight (beta-VAE)
        reduction: "mean" or "sum"

    Returns:
        total_loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    batch_size = x.size(0)

    # Reconstruction loss (Binary Cross-Entropy)
    # BCE models each pixel as an independent Bernoulli
    recon_loss = F.binary_cross_entropy(
        x_recon, x, reduction='sum'
    )

    # KL Divergence: KL(N(mu, sigma^2) || N(0, 1))
    # = -0.5 * sum(1 + log sigma^2 - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    if reduction == "mean":
        total_loss = total_loss / batch_size
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size

    return total_loss, recon_loss, kl_loss


class BetaVAE(VAE):
    """beta-VAE: Variant for disentanglement"""

    def __init__(self, config: VAEConfig):
        super().__init__(config)

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """beta-VAE loss computation"""
        total_loss, recon_loss, kl_loss = vae_loss(
            x, x_recon, mu, logvar,
            beta=self.config.beta
        )

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item()
        }


class ConditionalVAE(nn.Module):
    """Conditional VAE: Conditional generation"""

    def __init__(self, config: VAEConfig, num_classes: int = 10):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Class embedding
        self.class_embed = nn.Embedding(num_classes, config.latent_dim)

        # Encoder and Decoder are the same
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Projection for adding condition to Encoder
        self.cond_proj = nn.Linear(config.latent_dim, config.in_channels * config.image_size * config.image_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
            labels: (B,) class labels

        Returns:
            x_recon, mu, logvar
        """
        # Class embedding
        c = self.class_embed(labels)  # (B, latent_dim)

        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode with condition
        z_cond = z + c  # Add condition
        x_recon = self.decoder(z_cond)

        return x_recon, mu, logvar

    def sample(
        self,
        num_samples: int,
        labels: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Conditional sampling"""
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        c = self.class_embed(labels)
        z_cond = z + c
        samples = self.decoder(z_cond)
        return samples


# Latent space visualization
def visualize_latent_space(
    model: VAE,
    data_loader,
    device: torch.device,
    num_samples: int = 1000
):
    """Visualize latent space in 2D (for latent_dim=2)"""
    import matplotlib.pyplot as plt

    model.eval()
    latents = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            if len(latents) * data.size(0) >= num_samples:
                break

            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu())
            labels_list.append(labels)

    latents = torch.cat(latents, dim=0)[:num_samples]
    labels = torch.cat(labels_list, dim=0)[:num_samples]

    # 2D visualization (using first 2 dimensions only)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        latents[:, 0].numpy(),
        latents[:, 1].numpy(),
        c=labels.numpy(),
        cmap='tab10',
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('VAE Latent Space')
    plt.savefig('vae_latent_space.png')
    print("Saved vae_latent_space.png")


def interpolate_latent(
    model: VAE,
    x1: torch.Tensor,
    x2: torch.Tensor,
    num_steps: int = 10
) -> torch.Tensor:
    """Interpolate between two images in latent space"""
    model.eval()

    with torch.no_grad():
        # Encode both images
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        # Linear interpolation
        alphas = torch.linspace(0, 1, num_steps).to(mu1.device)
        interpolated = []

        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            x_recon = model.decode(z)
            interpolated.append(x_recon)

        return torch.cat(interpolated, dim=0)


def demo_training():
    """Train a VAE on synthetic toy data and visualize results."""
    # Expected: total loss decreasing, recon loss < 600 after 30 epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Toy data: random rectangles on 28x28 grayscale images ---
    num_images = 200
    images = torch.zeros(num_images, 1, 28, 28)
    for i in range(num_images):
        # Random rectangle position and size
        x1 = torch.randint(0, 14, (1,)).item()
        y1 = torch.randint(0, 14, (1,)).item()
        w = torch.randint(5, 15, (1,)).item()
        h = torch.randint(5, 15, (1,)).item()
        x2 = min(x1 + w, 28)
        y2 = min(y1 + h, 28)
        intensity = 0.5 + 0.5 * torch.rand(1).item()
        images[i, 0, y1:y2, x1:x2] = intensity

    images = images.to(device)

    # --- Model and optimizer ---
    config = VAEConfig(
        image_size=28,
        in_channels=1,
        latent_dim=20,
        hidden_dims=(32, 64),
        beta=1.0,
    )
    model = VAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Training loop (30 epochs) ---
    num_epochs = 30
    history_total = []
    history_recon = []
    history_kl = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        x_recon, mu, logvar = model(images)
        total_loss, recon_loss, kl_loss = vae_loss(
            images, x_recon, mu, logvar, beta=config.beta
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history_total.append(total_loss.item())
        history_recon.append(recon_loss.item())
        history_kl.append(kl_loss.item())

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Total: {total_loss.item():.2f} | "
                f"Recon: {recon_loss.item():.2f} | "
                f"KL: {kl_loss.item():.2f}"
            )

    # --- Visualization (3-panel figure) ---
    model.eval()
    with torch.no_grad():
        x_recon_vis, _, _ = model(images)
        sampled = model.sample(16, device)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Loss curves
    epochs_range = range(1, num_epochs + 1)
    axes[0].plot(epochs_range, history_total, label="Total Loss")
    axes[0].plot(epochs_range, history_recon, label="Recon Loss")
    axes[0].plot(epochs_range, history_kl, label="KL Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()

    # Panel 2: Reconstruction comparison (top: originals, bottom: reconstructions)
    n_show = 8
    originals = images[:n_show].cpu()
    reconstructions = x_recon_vis[:n_show].cpu()

    comparison = torch.zeros(2, n_show, 1, 28, 28)
    comparison[0] = originals
    comparison[1] = reconstructions

    grid_recon = torch.cat(
        [torch.cat([comparison[row, col] for col in range(n_show)], dim=2)
         for row in range(2)],
        dim=1,
    )  # (1, 56, 224)
    axes[1].imshow(grid_recon.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction (top: original, bottom: reconstructed)")
    axes[1].axis("off")

    # Panel 3: Sampled images from prior (4x4 grid)
    sampled_cpu = sampled.cpu()
    grid_sample = torch.cat(
        [torch.cat([sampled_cpu[r * 4 + c] for c in range(4)], dim=2)
         for r in range(4)],
        dim=1,
    )  # (1, 112, 112)
    axes[2].imshow(grid_sample.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Samples from Prior (4x4)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("vae_training_results.png", dpi=150)
    plt.close()
    print("\nSaved vae_training_results.png")


# Test
if __name__ == "__main__":
    print("=== VAE Low-Level Implementation ===\n")

    # Configuration
    config = VAEConfig(
        image_size=28,
        in_channels=1,
        latent_dim=20,
        hidden_dims=(32, 64),
        beta=1.0
    )
    print(f"Config: {config}\n")

    # Create model
    model = VAE(config)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Test input
    batch_size = 8
    x = torch.rand(batch_size, 1, 28, 28)

    # Forward
    x_recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Loss computation
    total_loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar)
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print(f"Recon Loss: {recon_loss.item():.4f}")
    print(f"KL Loss: {kl_loss.item():.4f}")

    # Sampling test
    samples = model.sample(16, x.device)
    print(f"\nSampled images shape: {samples.shape}")

    # beta-VAE test
    print("\n=== beta-VAE Test ===")
    config_beta = VAEConfig(beta=4.0)  # beta > 1 for disentanglement
    beta_vae = BetaVAE(config_beta)

    x_recon, mu, logvar = beta_vae(x)
    loss, metrics = beta_vae.compute_loss(x, x_recon, mu, logvar)
    print(f"beta-VAE Loss: {metrics}")

    # Conditional VAE test
    print("\n=== Conditional VAE Test ===")
    cvae = ConditionalVAE(config, num_classes=10)
    labels = torch.randint(0, 10, (batch_size,))

    x_recon, mu, logvar = cvae(x, labels)
    print(f"CVAE Reconstruction shape: {x_recon.shape}")

    # Conditional sampling
    cond_samples = cvae.sample(16, torch.arange(10).repeat(2)[:16], x.device)
    print(f"Conditional samples shape: {cond_samples.shape}")

    # Latent space interpolation
    print("\n=== Latent Interpolation ===")
    interp = interpolate_latent(model, x[:1], x[1:2], num_steps=5)
    print(f"Interpolated images shape: {interp.shape}")

    print("\nAll tests passed!")

    # Demo training with toy data and visualization
    print("\n=== Demo Training ===")
    demo_training()

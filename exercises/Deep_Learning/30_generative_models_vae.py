"""
Exercises for Lesson 30: Generative Models -- VAE
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Derive the KL Divergence Term ===
# Problem: Verify the closed-form KL divergence for Gaussian VAE.

def exercise_1():
    """Verify KL divergence formula numerically."""
    torch.manual_seed(42)

    # KL(q(z|x) || p(z)) where q = N(mu, sigma^2), p = N(0, 1)
    # Closed form: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    mu = torch.tensor([0.5, -0.3, 1.2])
    log_var = torch.tensor([-0.5, 0.3, -1.0])  # log(sigma^2)

    # Closed-form KL
    kl_closed = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Monte Carlo estimate for verification
    n_samples = 100000
    sigma = (0.5 * log_var).exp()
    eps = torch.randn(n_samples, 3)
    z = mu + sigma * eps  # Reparameterized samples

    # log q(z|x) - log p(z)
    log_q = -0.5 * ((z - mu) / sigma).pow(2).sum(-1) - 0.5 * log_var.sum() - 0.5 * 3 * np.log(2 * np.pi)
    log_p = -0.5 * z.pow(2).sum(-1) - 0.5 * 3 * np.log(2 * np.pi)
    kl_mc = (log_q - log_p).mean()

    print(f"  mu = {mu.tolist()}")
    print(f"  log_var = {log_var.tolist()}")
    print(f"  Closed-form KL: {kl_closed.item():.4f}")
    print(f"  Monte Carlo KL: {kl_mc.item():.4f}")
    print(f"  Difference:     {abs(kl_closed.item() - kl_mc.item()):.4f}")
    print("  Closed form eliminates sampling noise and is exact for Gaussians.")


# === Exercise 2: Visualize the Reparameterization Trick ===
# Problem: Show that reparameterization enables gradient flow.

def exercise_2():
    """Demonstrate gradient flow through reparameterization trick."""
    # Small encoder network
    encoder = nn.Linear(10, 4)  # Outputs 2 mu + 2 log_var

    x = torch.randn(1, 10)
    out = encoder(x)
    mu = out[:, :2]
    log_var = out[:, 2:]

    # Reparameterized sampling: z = mu + std * eps
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + std * eps

    # Compute loss and backprop
    loss = z.sum()
    loss.backward()

    print(f"  mu = {mu.detach().tolist()}")
    print(f"  log_var = {log_var.detach().tolist()}")
    print(f"  z (reparameterized) = {z.detach().tolist()}")
    print(f"  encoder.weight.grad is not None: {encoder.weight.grad is not None}")
    print(f"  Gradient norm: {encoder.weight.grad.norm().item():.4f}")
    print("\n  Direct sampling z ~ N(mu, sigma) would break the computational graph")
    print("  because sampling is not differentiable. The reparameterization trick")
    print("  moves the stochasticity to eps ~ N(0,1), which is constant w.r.t. parameters.")


# === Exercise 3: Train VAE and Visualize Latent Space ===
# Problem: Train VAE with latent_dim=2 and visualize 2D latent space.

def exercise_3():
    """Train VAE on synthetic digit-like data, visualize latent space."""
    torch.manual_seed(42)

    class VAE(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=256, latent_dim=2):
            super().__init__()
            # Encoder
            self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_var = nn.Linear(hidden_dim, latent_dim)
            # Decoder
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_var(h)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            return self.dec(z), mu, log_var

    def vae_loss(recon, x, mu, log_var):
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss

    # Synthetic data: 10 cluster centers in 784-dim
    n_samples = 1000
    n_classes = 10
    X = torch.zeros(n_samples, 784)
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_classes):
        center = torch.randn(784) * 0.1
        X[i * 100:(i + 1) * 100] = torch.sigmoid(center + torch.randn(100, 784) * 0.05)
        y[i * 100:(i + 1) * 100] = i

    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    model = VAE(latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for (xb, _) in loader:
            recon, mu, log_var = model(xb)
            loss = vae_loss(recon, xb, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f"  Final training loss: {total_loss / len(X):.2f}")

    # Visualize latent space
    model.eval()
    with torch.no_grad():
        mu_all, _ = model.encode(X)
        z = mu_all.numpy()

    print(f"\n  Latent space statistics:")
    for cls in range(n_classes):
        mask = (y == cls).numpy()
        cls_z = z[mask]
        print(f"    Class {cls}: center=({cls_z[:, 0].mean():.2f}, {cls_z[:, 1].mean():.2f})")

    # Generate from grid
    with torch.no_grad():
        z_grid = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        generated = model.dec(z_grid)
    print(f"\n  Generated images at z=(-2,-2), (0,0), (2,2):")
    for i, z_pt in enumerate(z_grid):
        print(f"    z={z_pt.tolist()}: pixel_mean={generated[i].mean().item():.4f}")


# === Exercise 4: Beta-VAE Experiment ===
# Problem: Compare different beta values on latent space structure.

def exercise_4():
    """Beta-VAE: trade-off between reconstruction and disentanglement."""
    torch.manual_seed(42)

    class VAE(nn.Module):
        def __init__(self, input_dim=100, latent_dim=2):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
            self.fc_mu = nn.Linear(64, latent_dim)
            self.fc_var = nn.Linear(64, latent_dim)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, input_dim), nn.Sigmoid(),
            )

        def forward(self, x):
            h = self.enc(x)
            mu, log_var = self.fc_mu(h), self.fc_var(h)
            z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
            return self.dec(z), mu, log_var

    X = torch.sigmoid(torch.randn(500, 100))
    loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

    print(f"  {'Beta':>6} {'Recon Loss':>12} {'KL Loss':>10} {'Latent Std':>11}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*11}")

    for beta in [0.5, 1.0, 4.0]:
        torch.manual_seed(42)
        model = VAE(input_dim=100, latent_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(30):
            for (xb,) in loader:
                recon, mu, log_var = model(xb)
                recon_loss = F.binary_cross_entropy(recon, xb, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + beta * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recon, mu_all, log_var_all = model(X)
            final_recon = F.binary_cross_entropy(recon, X, reduction='sum').item() / len(X)
            final_kl = (-0.5 * torch.sum(1 + log_var_all - mu_all.pow(2) - log_var_all.exp())).item() / len(X)
            latent_std = mu_all.std(dim=0).mean().item()

        print(f"  {beta:6.1f} {final_recon:12.4f} {final_kl:10.4f} {latent_std:11.4f}")

    print("\n  Higher beta pushes KL down -> tighter latent space -> better disentanglement")
    print("  but worse reconstruction quality (trade-off).")


# === Exercise 5: Conditional VAE (CVAE) ===
# Problem: Extend VAE to conditional generation with class labels.

def exercise_5():
    """Conditional VAE: generate samples of a specific class."""
    torch.manual_seed(42)

    class CVAE(nn.Module):
        def __init__(self, input_dim=784, latent_dim=8, n_classes=10):
            super().__init__()
            self.n_classes = n_classes
            # Encoder: input + one-hot label
            self.enc = nn.Sequential(
                nn.Linear(input_dim + n_classes, 256), nn.ReLU()
            )
            self.fc_mu = nn.Linear(256, latent_dim)
            self.fc_var = nn.Linear(256, latent_dim)
            # Decoder: z + one-hot label
            self.dec = nn.Sequential(
                nn.Linear(latent_dim + n_classes, 256), nn.ReLU(),
                nn.Linear(256, input_dim), nn.Sigmoid(),
            )

        def encode(self, x, y_onehot):
            h = self.enc(torch.cat([x, y_onehot], dim=1))
            return self.fc_mu(h), self.fc_var(h)

        def decode(self, z, y_onehot):
            return self.dec(torch.cat([z, y_onehot], dim=1))

        def forward(self, x, y):
            y_onehot = F.one_hot(y, self.n_classes).float()
            mu, log_var = self.encode(x, y_onehot)
            z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
            return self.decode(z, y_onehot), mu, log_var

        def generate(self, y, num_samples=10):
            y_onehot = F.one_hot(torch.full((num_samples,), y, dtype=torch.long),
                                 self.n_classes).float()
            z = torch.randn(num_samples, self.fc_mu.out_features)
            return self.decode(z, y_onehot)

    # Synthetic data with class structure
    n_classes = 10
    n_per_class = 100
    X = torch.zeros(n_classes * n_per_class, 784)
    y = torch.zeros(n_classes * n_per_class, dtype=torch.long)

    for c in range(n_classes):
        # Each class has a distinct pattern
        pattern = torch.zeros(784)
        pattern[c * 78:(c + 1) * 78] = 1.0
        X[c * n_per_class:(c + 1) * n_per_class] = \
            torch.sigmoid(pattern + torch.randn(n_per_class, 784) * 0.1)
        y[c * n_per_class:(c + 1) * n_per_class] = c

    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    model = CVAE(latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        for xb, yb in loader:
            recon, mu, log_var = model(xb, yb)
            recon_loss = F.binary_cross_entropy(recon, xb, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Generate samples for each class
    model.eval()
    print(f"  Generated sample statistics per class:")
    for c in range(n_classes):
        with torch.no_grad():
            samples = model.generate(c, num_samples=10)
            # Check if the class pattern is present
            region_mean = samples[:, c * 78:(c + 1) * 78].mean().item()
            other_mean = samples[:, :c * 78].mean().item() if c > 0 else 0
        print(f"    Class {c}: target_region_mean={region_mean:.4f}, "
              f"other_region_mean={other_mean:.4f}")

    print("\n  CVAE successfully generates class-specific samples by conditioning")
    print("  both encoder and decoder on the class label.")


if __name__ == "__main__":
    print("=== Exercise 1: KL Divergence Derivation ===")
    exercise_1()
    print("\n=== Exercise 2: Reparameterization Trick ===")
    exercise_2()
    print("\n=== Exercise 3: Train VAE ===")
    exercise_3()
    print("\n=== Exercise 4: Beta-VAE ===")
    exercise_4()
    print("\n=== Exercise 5: Conditional VAE ===")
    exercise_5()
    print("\nAll exercises completed!")

"""
Exercises for Lesson 28: Generative Models -- GAN
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Explain the Min-Max Game ===
# Problem: Conceptual + code demonstration of GAN min-max game.

def exercise_1():
    """Demonstrate GAN min-max game with simple 1D distribution."""
    torch.manual_seed(42)

    # 1. Discriminator maximizes: E[log D(x)] + E[log(1 - D(G(z)))]
    #    -> Correctly classify real as real, fake as fake
    # 2. Generator minimizes: E[log(1 - D(G(z)))]
    #    -> Make D(G(z)) close to 1 (fool the discriminator)
    # 3. Non-saturating loss: -E[log D(G(z))] gives stronger gradients
    #    when D is confident, avoiding the flat region of log(1-x) near x=0

    # Demonstrate gradient difference
    D_Gz = torch.tensor(0.01, requires_grad=True)  # D is very confident fake

    # Original formulation: log(1 - D(G(z)))
    loss_original = torch.log(1 - D_Gz)
    loss_original.backward()
    grad_original = D_Gz.grad.item()

    D_Gz2 = torch.tensor(0.01, requires_grad=True)
    # Non-saturating: -log(D(G(z)))
    loss_ns = -torch.log(D_Gz2)
    loss_ns.backward()
    grad_ns = D_Gz2.grad.item()

    print(f"  When D(G(z)) = 0.01 (D confident it's fake):")
    print(f"    Original loss gradient:       {grad_original:.4f}")
    print(f"    Non-saturating loss gradient: {grad_ns:.4f}")
    print(f"    Non-saturating provides {abs(grad_ns/grad_original):.0f}x stronger gradient")
    print(f"\n  This is why -log(D(G(z))) is used: stronger signal early in training")
    print(f"  when the generator is poor and D is confident.")


# === Exercise 2: Weight Initialization for DCGAN ===
# Problem: Implement DCGAN-style weight initialization.

def exercise_2():
    """DCGAN weight initialization: Normal(0, 0.02) for Conv and BN."""

    class DCGANGenerator(nn.Module):
        def __init__(self, latent_dim=100, channels=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)  # Added: initialize BN bias to 0

        def forward(self, z):
            return self.net(z.view(z.size(0), -1, 1, 1))

    model = DCGANGenerator()

    # Verify initialization
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        elif 'bias' in name:
            print(f"  {name}: mean={param.data.mean():.4f}")

    print("\n  Custom init (std=0.02) is important because:")
    print("  - PyTorch default Kaiming init can cause mode collapse")
    print("  - Tighter initialization keeps activations in stable range")
    print("  - BN bias=0 ensures normalization starts centered")


# === Exercise 3: Train a Vanilla GAN on Synthetic Data ===
# Problem: Train GAN on a simple distribution.

def exercise_3():
    """Train vanilla GAN on synthetic MNIST-like data."""
    torch.manual_seed(42)

    latent_dim = 64
    img_dim = 28 * 28

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, img_dim), nn.Tanh(),
            )

        def forward(self, z):
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(img_dim, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    G = Generator()
    D = Discriminator()
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Synthetic data: normally distributed "images"
    real_data = torch.randn(1000, img_dim) * 0.5
    loader = DataLoader(TensorDataset(real_data), batch_size=64, shuffle=True)

    g_losses, d_losses = [], []

    for epoch in range(50):
        epoch_g_loss = epoch_d_loss = 0
        for (real,) in loader:
            batch_size = real.size(0)
            real_label = torch.ones(batch_size, 1)
            fake_label = torch.zeros(batch_size, 1)

            # Train Discriminator
            z = torch.randn(batch_size, latent_dim)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            loss_D = criterion(d_real, real_label) + criterion(d_fake, fake_label)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim)
            fake = G(z)
            d_fake = D(fake)
            loss_G = criterion(d_fake, real_label)  # Non-saturating
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

        g_losses.append(epoch_g_loss / len(loader))
        d_losses.append(epoch_d_loss / len(loader))

    print(f"  Epoch  1: D_loss={d_losses[0]:.4f}, G_loss={g_losses[0]:.4f}")
    print(f"  Epoch 25: D_loss={d_losses[24]:.4f}, G_loss={g_losses[24]:.4f}")
    print(f"  Epoch 50: D_loss={d_losses[49]:.4f}, G_loss={g_losses[49]:.4f}")

    # Generate samples
    with torch.no_grad():
        samples = G(torch.randn(16, latent_dim))
    print(f"  Generated sample stats: mean={samples.mean():.4f}, std={samples.std():.4f}")
    print("  GAN losses don't converge to 0; they oscillate as D and G compete.")


# === Exercise 4: Compare WGAN vs Vanilla GAN ===
# Problem: Replace BCE loss with Wasserstein loss.

def exercise_4():
    """WGAN with weight clipping vs vanilla GAN."""
    torch.manual_seed(42)

    latent_dim = 64
    data_dim = 100

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, data_dim), nn.Tanh(),
            )

        def forward(self, z):
            return self.net(z)

    class Critic(nn.Module):
        """WGAN uses a Critic (no Sigmoid) instead of Discriminator."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(data_dim, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 1),  # No Sigmoid
            )

        def forward(self, x):
            return self.net(x)

    real_data = torch.randn(500, data_dim) * 0.3

    for mode in ["Vanilla", "WGAN"]:
        torch.manual_seed(42)
        G = Generator()
        C = Critic()
        opt_G = torch.optim.RMSprop(G.parameters(), lr=0.00005)
        opt_C = torch.optim.RMSprop(C.parameters(), lr=0.00005)

        n_critic = 5 if mode == "WGAN" else 1
        losses = []

        for epoch in range(20):
            for _ in range(n_critic):
                idx = torch.randint(0, 500, (64,))
                real = real_data[idx]
                z = torch.randn(64, latent_dim)
                fake = G(z).detach()

                if mode == "WGAN":
                    loss_C = -(C(real).mean() - C(fake).mean())
                else:
                    loss_C = -torch.log(torch.sigmoid(C(real))).mean() \
                             - torch.log(1 - torch.sigmoid(C(fake))).mean()

                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

                if mode == "WGAN":
                    for p in C.parameters():
                        p.data.clamp_(-0.01, 0.01)

            z = torch.randn(64, latent_dim)
            fake = G(z)
            if mode == "WGAN":
                loss_G = -C(fake).mean()
            else:
                loss_G = -torch.log(torch.sigmoid(C(fake))).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            losses.append(loss_G.item())

        print(f"  {mode}: final_G_loss={losses[-1]:.4f}, "
              f"loss_std={np.std(losses[-10:]):.4f}")

    print("  WGAN provides more stable gradients via Wasserstein distance.")
    print("  Weight clipping enforces Lipschitz constraint on the critic.")


# === Exercise 5: Latent Space Interpolation ===
# Problem: Linear vs spherical interpolation in GAN latent space.

def exercise_5():
    """Linear vs spherical interpolation in GAN latent space."""
    torch.manual_seed(42)

    latent_dim = 64

    # Simple trained generator
    G = nn.Sequential(
        nn.Linear(latent_dim, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 784), nn.Tanh(),
    )

    z1 = torch.randn(latent_dim)
    z2 = torch.randn(latent_dim)

    # Linear interpolation
    def lerp(z1, z2, steps=10):
        alphas = torch.linspace(0, 1, steps)
        return torch.stack([(1 - a) * z1 + a * z2 for a in alphas])

    # Spherical interpolation (slerp)
    def slerp(z1, z2, steps=10):
        z1_n = z1 / z1.norm()
        z2_n = z2 / z2.norm()
        omega = torch.acos(torch.clamp(torch.dot(z1_n, z2_n), -1, 1))
        alphas = torch.linspace(0, 1, steps)
        if omega.abs() < 1e-6:
            return lerp(z1, z2, steps)
        results = []
        for a in alphas:
            z = (torch.sin((1 - a) * omega) * z1 + torch.sin(a * omega) * z2) / torch.sin(omega)
            results.append(z)
        return torch.stack(results)

    z_linear = lerp(z1, z2, 10)
    z_spherical = slerp(z1, z2, 10)

    # Compare norms along interpolation
    print(f"  Norms along interpolation path:")
    print(f"  {'Step':>4} {'Linear Norm':>12} {'Slerp Norm':>12}")
    for i in range(10):
        print(f"  {i:4d} {z_linear[i].norm().item():12.4f} "
              f"{z_spherical[i].norm().item():12.4f}")

    # Generate images
    with torch.no_grad():
        imgs_linear = G(z_linear)
        imgs_slerp = G(z_spherical)

    print(f"\n  Linear interp midpoint norm: {z_linear[5].norm().item():.4f}")
    print(f"  Slerp interp midpoint norm:  {z_spherical[5].norm().item():.4f}")
    print(f"  Expected norm (sqrt(d)):      {np.sqrt(latent_dim):.4f}")
    print("\n  Slerp maintains constant norm along the interpolation path.")
    print("  Linear interp dips in norm at the midpoint, passing through low-density regions.")
    print("  This is why slerp produces smoother transitions for Gaussian latent spaces.")


if __name__ == "__main__":
    print("=== Exercise 1: Min-Max Game ===")
    exercise_1()
    print("\n=== Exercise 2: DCGAN Weight Init ===")
    exercise_2()
    print("\n=== Exercise 3: Train Vanilla GAN ===")
    exercise_3()
    print("\n=== Exercise 4: WGAN vs Vanilla GAN ===")
    exercise_4()
    print("\n=== Exercise 5: Latent Space Interpolation ===")
    exercise_5()
    print("\nAll exercises completed!")

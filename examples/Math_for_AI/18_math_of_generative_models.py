"""
Mathematics of Generative Models

Demonstrates:
- VAE: ELBO decomposition (reconstruction + KL divergence), reparameterization trick
- GAN: minimax objective, generator/discriminator loss dynamics
- Diffusion model: forward noising process and score function intuition

Dependencies: numpy, torch, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. VAE: Evidence Lower Bound (ELBO)
# ---------------------------------------------------------------------------

def vae_elbo():
    """
    ELBO derivation and computation.

    log p(x) ≥ E_{q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z))
             = Reconstruction term - Regularization term

    For Gaussian q and Gaussian p(z) = N(0, I):
      KL = ½ Σ (μ² + σ² - log σ² - 1)
    """
    print("=" * 60)
    print("VAE: EVIDENCE LOWER BOUND (ELBO)")
    print("=" * 60)

    print("\nlog p(x) ≥ ELBO(x) = E_q[log p(x|z)] - KL(q(z|x) || p(z))")
    print()
    print("Components:")
    print("  Reconstruction: how well decoder p(x|z) reconstructs x")
    print("  KL penalty:     how close encoder q(z|x) is to prior p(z)")

    # Analytical KL for diagonal Gaussians
    def kl_gaussian(mu, log_var):
        """KL(N(μ, σ²) || N(0, 1)) = ½(μ² + σ² - log σ² - 1)"""
        return 0.5 * torch.sum(mu**2 + log_var.exp() - log_var - 1, dim=-1)

    print("\n--- KL Divergence: q(z|x) || p(z) = N(0,I) ---")
    print("KL = ½ Σᵢ (μᵢ² + σᵢ² - log σᵢ² - 1)")
    print()

    # Demonstrate KL behavior
    configs = [
        ("q = N(0, 1)  [matches prior]",   torch.zeros(1, 2), torch.zeros(1, 2)),
        ("q = N(2, 1)  [shifted mean]",     torch.tensor([[2.0, 2.0]]), torch.zeros(1, 2)),
        ("q = N(0, 4)  [larger variance]",  torch.zeros(1, 2), torch.tensor([[np.log(4), np.log(4)]])),
        ("q = N(2, 4)  [both differ]",      torch.tensor([[2.0, 2.0]]), torch.tensor([[np.log(4), np.log(4)]])),
    ]

    print(f"{'Configuration':35s} {'KL value':>10s}")
    print("-" * 50)
    for label, mu, log_var in configs:
        kl = kl_gaussian(mu, log_var).item()
        print(f"{label:35s} {kl:10.4f}")

    # Reparameterization trick
    print("\n--- Reparameterization Trick ---")
    print("To backprop through sampling z ~ q(z|x) = N(μ, σ²):")
    print("  z = μ + σ * ε,   ε ~ N(0, I)")
    print("Gradient flows through μ and σ, not through the stochastic node ε")

    torch.manual_seed(42)
    mu = torch.tensor([1.0, -0.5], requires_grad=True)
    log_var = torch.tensor([0.5, -0.3], requires_grad=True)

    # Sample using reparameterization
    eps = torch.randn_like(mu)
    z = mu + (0.5 * log_var).exp() * eps

    # Downstream loss
    loss = (z**2).sum()
    loss.backward()

    print(f"\nμ = {mu.detach().numpy()}")
    print(f"log_var = {log_var.detach().numpy()}")
    print(f"ε ~ N(0,I) = {eps.numpy()}")
    print(f"z = μ + σε = {z.detach().numpy()}")
    print(f"∂loss/∂μ = {mu.grad.numpy()}  (non-zero → trainable)")
    print(f"∂loss/∂log_var = {log_var.grad.numpy()}  (non-zero → trainable)")

    # Full ELBO example with tiny VAE-like setup
    print("\n--- Mini VAE: ELBO over Training Steps ---")

    class MiniVAE(nn.Module):
        """Tiny VAE for 1D data"""
        def __init__(self, latent_dim=2):
            super().__init__()
            self.encoder_mu = nn.Linear(4, latent_dim)
            self.encoder_logvar = nn.Linear(4, latent_dim)
            self.decoder = nn.Linear(latent_dim, 4)

        def encode(self, x):
            return self.encoder_mu(x), self.encoder_logvar(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, log_var = self.encode(x)
            eps = torch.randn_like(mu)
            z = mu + (0.5 * log_var).exp() * eps
            x_recon = self.decode(z)
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
            kl_loss = kl_gaussian(mu, log_var).sum()
            elbo = -(recon_loss + kl_loss)
            return elbo, recon_loss.item(), kl_loss.item()

    torch.manual_seed(0)
    vae = MiniVAE(latent_dim=2)
    optimizer = optim.Adam(vae.parameters(), lr=1e-2)

    # Synthetic data: 4-dim Gaussian blobs
    data = torch.randn(200, 4)

    elbos, recon_ls, kl_ls = [], [], []
    for step in range(200):
        optimizer.zero_grad()
        elbo, recon_l, kl_l = vae(data)
        loss = -elbo  # maximize ELBO = minimize negative ELBO
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            elbos.append(elbo.item())
            recon_ls.append(recon_l)
            kl_ls.append(kl_l)

    print(f"  {'Step':>5s} {'ELBO':>10s} {'Recon':>10s} {'KL':>10s}")
    for i, (step, el, rl, kl) in enumerate(zip(range(0, 200, 20), elbos, recon_ls, kl_ls)):
        print(f"  {step:>5d} {el:>10.2f} {rl:>10.2f} {kl:>10.2f}")

    return elbos, recon_ls, kl_ls


# ---------------------------------------------------------------------------
# 2. GAN: Minimax Objective
# ---------------------------------------------------------------------------

def gan_minimax():
    """
    GAN objective: min_G max_D V(D, G)
      = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

    At optimum: D*(x) = p_data(x) / (p_data(x) + p_g(x))
    Generator minimizes: E_z[log(1 - D(G(z)))]
    """
    print("\n" + "=" * 60)
    print("GAN: MINIMAX OBJECTIVE")
    print("=" * 60)

    print("\nValue function V(D, G):")
    print("  E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]")
    print()
    print("Discriminator D: real→1, fake→0  (max V)")
    print("Generator G:     fool D            (min V)")

    # Optimal discriminator for fixed G
    print("\n--- Optimal Discriminator D* ---")
    print("For fixed G: D*(x) = p_data(x) / (p_data(x) + p_g(x))")
    print("At Nash equilibrium: p_g = p_data, D*(x) = 1/2 everywhere")
    print("JS divergence interpretation:")
    print("  max_D V(D, G) = 2·JSD(p_data || p_g) - log 4")
    print("  JSD = 0 iff p_data = p_g  → generator objective minimizes JSD")

    # Demonstrate discriminator loss dynamics
    print("\n--- Discriminator Loss Landscape ---")
    d_output_real = np.linspace(0.01, 0.99, 100)
    # Loss for real samples: -log D(x)
    loss_real = -np.log(d_output_real)
    # Loss for fake samples: -log(1 - D(G(z)))
    loss_fake = -np.log(1 - d_output_real)

    print(f"At D(real)=0.9: loss_real={-np.log(0.9):.4f},  "
          f"loss_fake={-np.log(0.1):.4f}")
    print(f"At D(real)=0.5: loss_real={-np.log(0.5):.4f},  "
          f"loss_fake={-np.log(0.5):.4f} [optimal]")

    # Training dynamics illustration — 1D toy GAN
    print("\n--- Toy 1D GAN Training (Gaussian target) ---")

    class TinyG(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        def forward(self, z):
            return self.net(z)

    class TinyD(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        def forward(self, x):
            return torch.sigmoid(self.net(x))

    torch.manual_seed(5)
    G = TinyG()
    D = TinyD()
    opt_G = optim.Adam(G.parameters(), lr=1e-3)
    opt_D = optim.Adam(D.parameters(), lr=1e-3)

    real_mu, real_sigma = 3.0, 0.5
    d_losses, g_losses = [], []

    for step in range(500):
        # Train D
        real = torch.tensor(np.random.normal(real_mu, real_sigma, (64, 1)), dtype=torch.float32)
        z = torch.randn(64, 1)
        fake = G(z).detach()

        d_loss = -(torch.log(D(real) + 1e-8) + torch.log(1 - D(fake) + 1e-8)).mean()
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # Train G
        z = torch.randn(64, 1)
        g_loss = -torch.log(D(G(z)) + 1e-8).mean()  # non-saturating variant
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        if step % 50 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

    print(f"  {'Step':>5s} {'D_loss':>10s} {'G_loss':>10s}")
    for i, (dl, gl) in enumerate(zip(d_losses, g_losses)):
        print(f"  {i*50:>5d} {dl:>10.4f} {gl:>10.4f}")

    # Check generator output distribution
    with torch.no_grad():
        z = torch.randn(1000, 1)
        fake_samples = G(z).numpy().flatten()
    print(f"\nGenerator output (should approach N({real_mu}, {real_sigma}²)):")
    print(f"  Generated mean: {fake_samples.mean():.4f}  (target: {real_mu})")
    print(f"  Generated std:  {fake_samples.std():.4f}  (target: {real_sigma})")

    return G, D, d_losses, g_losses, fake_samples, real_mu, real_sigma


# ---------------------------------------------------------------------------
# 3. Diffusion: Forward Process
# ---------------------------------------------------------------------------

def diffusion_forward_process():
    """
    Diffusion model forward process: gradually add Gaussian noise.

    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

    Closed form: q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
    where ᾱ_t = ∏_{s=1}^t (1-β_s)
    """
    print("\n" + "=" * 60)
    print("DIFFUSION MODEL: FORWARD PROCESS")
    print("=" * 60)

    print("\nForward diffusion: gradually corrupt x_0 → x_T ~ N(0, I)")
    print()
    print("Step-wise: q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1},  β_t I)")
    print("Closed form: q(x_t | x_0) = N(√ᾱ_t x_0,  (1-ᾱ_t) I)")
    print("             where ᾱ_t = ∏_{s=1}^t α_s,  α_s = 1 - β_s")

    T = 1000
    # Linear noise schedule
    beta_start, beta_end = 1e-4, 0.02
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)  # ᾱ_t

    print(f"\nNoise schedule: linear β from {beta_start} to {beta_end}")
    print(f"T = {T} steps")

    print("\nSignal and noise levels at key timesteps:")
    print(f"{'t':>6s} {'√ᾱ_t (signal)':>16s} {'√(1-ᾱ_t) (noise)':>18s}")
    print("-" * 45)
    for t in [0, 100, 250, 500, 750, 999]:
        signal = np.sqrt(alpha_bar[t])
        noise = np.sqrt(1 - alpha_bar[t])
        print(f"{t:>6d} {signal:>16.4f} {noise:>18.4f}")

    print("\nAt t=999: signal ≈ 0, noise ≈ 1 → x_T ~ N(0, I) [pure noise]")

    # Score function intuition
    print("\n--- Score Function ---")
    print("Score: s_θ(x_t, t) ≈ ∇_{x_t} log q(x_t)")
    print("For Gaussian q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I):")
    print("  ∇_{x_t} log q(x_t|x_0) = -(x_t - √ᾱ_t x_0) / (1-ᾱ_t)")
    print("                          = -ε / √(1-ᾱ_t)")
    print("where ε is the Gaussian noise added at step t")
    print("\nDenoising score matching trains a network to predict ε from x_t:")
    print("  L = E_{t, x_0, ε}[‖ε_θ(x_t, t) - ε‖²]")

    # Demonstrate forward process on a 1D signal
    np.random.seed(42)
    x0 = np.array([2.0])  # clean data point

    print("\n1D example: x_0 = 2.0")
    print(f"{'t':>6s} {'E[x_t|x_0]':>14s} {'Std[x_t|x_0]':>14s} {'Sample x_t':>12s}")
    print("-" * 52)
    for t in [0, 100, 250, 500, 999]:
        ab = alpha_bar[t]
        mean_xt = np.sqrt(ab) * x0
        std_xt = np.sqrt(1 - ab)
        sample_xt = mean_xt + std_xt * np.random.randn(*x0.shape)
        print(f"{t:>6d} {mean_xt[0]:>14.4f} {std_xt:>14.4f} {sample_xt[0]:>12.4f}")

    return betas, alpha_bar


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def visualize_generative(elbos, recon_ls, kl_ls,
                         fake_samples, real_mu, real_sigma,
                         betas, alpha_bar):
    """Visualize VAE ELBO, GAN samples, and diffusion schedule"""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Plot 1: VAE ELBO decomposition ---
    ax1 = axes[0]
    steps = list(range(0, 200, 20))
    ax1.plot(steps, [-e for e in elbos], 'b-o', label='-ELBO (total)', linewidth=2)
    ax1.plot(steps, recon_ls, 'r--s', label='Recon loss', linewidth=1.5)
    ax1.plot(steps, kl_ls, 'g--^', label='KL loss', linewidth=1.5)
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Loss')
    ax1.set_title('VAE ELBO Decomposition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: GAN generator distribution ---
    ax2 = axes[1]
    x_range = np.linspace(0, 6, 200)
    real_pdf = (1 / (real_sigma * np.sqrt(2*np.pi)) *
                np.exp(-0.5 * ((x_range - real_mu)/real_sigma)**2))
    ax2.hist(fake_samples, bins=40, density=True, alpha=0.5, color='steelblue', label='Generated')
    ax2.plot(x_range, real_pdf, 'r-', linewidth=2, label=f'Target N({real_mu},{real_sigma}²)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title('GAN: Generated vs Target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Diffusion signal/noise schedule ---
    ax3 = axes[2]
    t_vals = np.arange(len(alpha_bar))
    signal = np.sqrt(alpha_bar)
    noise = np.sqrt(1 - alpha_bar)
    ax3.plot(t_vals, signal, 'b-', linewidth=2, label='Signal √ᾱ_t')
    ax3.plot(t_vals, noise, 'r-', linewidth=2, label='Noise √(1-ᾱ_t)')
    ax3.fill_between(t_vals, 0, signal, alpha=0.1, color='blue')
    ax3.fill_between(t_vals, signal, 1, alpha=0.1, color='red')
    ax3.set_xlabel('Diffusion step t')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Diffusion Forward Process\nSignal vs Noise Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('math_of_generative_models.png', dpi=150)
    print("Visualization saved to math_of_generative_models.png")
    plt.close()


if __name__ == "__main__":
    elbos, recon_ls, kl_ls = vae_elbo()
    G, D, d_losses, g_losses, fake_samples, real_mu, real_sigma = gan_minimax()
    betas, alpha_bar = diffusion_forward_process()
    visualize_generative(elbos, recon_ls, kl_ls, fake_samples, real_mu, real_sigma, betas, alpha_bar)

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)

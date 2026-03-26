"""
Exercises for Lesson 18: Math of Generative Models
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Two Methods of ELBO Derivation ===
# Problem: Derive ELBO using (1) Jensen's inequality and (2) KL decomposition.

def exercise_1():
    """ELBO derivation via Jensen's inequality and KL decomposition."""
    print("ELBO Derivation: Two Equivalent Methods\n")

    print("=" * 50)
    print("Method 1: Jensen's Inequality")
    print("=" * 50)
    print()
    print("Start with log-likelihood:")
    print("  log p(x) = log integral p(x,z) dz")
    print("           = log integral p(x,z) * q(z|x)/q(z|x) dz")
    print("           = log E_{q(z|x)} [p(x,z) / q(z|x)]")
    print()
    print("By Jensen's inequality (log is concave):")
    print("  log E[f(Z)] >= E[log f(Z)]")
    print()
    print("Therefore:")
    print("  log p(x) >= E_{q(z|x)} [log p(x,z) / q(z|x)]")
    print("           = E_{q(z|x)} [log p(x,z)] - E_{q(z|x)} [log q(z|x)]")
    print("           = E_{q(z|x)} [log p(x|z) + log p(z)] + H[q(z|x)]")
    print("           = E_{q(z|x)} [log p(x|z)] - KL(q(z|x) || p(z))")
    print()
    print("  This is the ELBO (Evidence Lower Bound).")

    print()
    print("=" * 50)
    print("Method 2: KL Decomposition")
    print("=" * 50)
    print()
    print("Start with KL divergence (always >= 0):")
    print("  KL(q(z|x) || p(z|x)) = E_{q} [log q(z|x) - log p(z|x)]")
    print("                        = E_{q} [log q(z|x) - log p(x,z) + log p(x)]")
    print("                        = E_{q} [log q(z|x) - log p(x,z)] + log p(x)")
    print()
    print("Rearranging:")
    print("  log p(x) = KL(q(z|x) || p(z|x)) + E_{q} [log p(x,z) - log q(z|x)]")
    print("  log p(x) = KL(q(z|x) || p(z|x)) + ELBO")
    print()
    print("Since KL >= 0:")
    print("  log p(x) >= ELBO")
    print("  ELBO = E_{q(z|x)} [log p(x|z)] - KL(q(z|x) || p(z|x))")
    print()
    print("  Equality holds iff q(z|x) = p(z|x) (true posterior).")

    print()
    print("=" * 50)
    print("Why Both Methods Give the Same Result")
    print("=" * 50)
    print()
    print("Both methods express the same mathematical identity:")
    print("  log p(x) = ELBO + KL(q || p(z|x))")
    print()
    print("Method 1 bounds log p(x) from below using Jensen's inequality,")
    print("which is equivalent to dropping the non-negative KL term.")
    print("Method 2 explicitly decomposes log p(x) into ELBO + KL gap.")
    print("The ELBO expression is identical in both cases:")
    print("  ELBO = E_{q(z|x)} [log p(x|z)] - KL(q(z|x) || p(z))")

    # Numerical verification with a simple example
    print("\n\nNumerical Verification (1D Gaussian):")
    np.random.seed(42)

    # True model: p(z) = N(0,1), p(x|z) = N(z, 0.5^2)
    # So p(x) = N(0, 1 + 0.25) = N(0, 1.25)
    # True posterior: p(z|x) = N(x*4/5, 0.5^2*4/5) = N(0.8x, 0.4)
    sigma_x_given_z = 0.5
    x_obs = 1.5

    # Variational approximation q(z|x) = N(mu_q, sigma_q^2)
    mu_q = 1.0
    sigma_q = 0.7

    # Monte Carlo ELBO
    n_samples = 100000
    z_samples = mu_q + sigma_q * np.random.randn(n_samples)

    # log p(x|z) = log N(x; z, sigma^2)
    log_p_x_given_z = -0.5 * ((x_obs - z_samples) / sigma_x_given_z) ** 2 - np.log(
        sigma_x_given_z * np.sqrt(2 * np.pi))

    # KL(q || p(z)) where p(z) = N(0,1), q = N(mu_q, sigma_q^2)
    kl_q_pz = 0.5 * (sigma_q ** 2 + mu_q ** 2 - 1 - 2 * np.log(sigma_q))

    elbo = np.mean(log_p_x_given_z) - kl_q_pz

    # True log p(x)
    sigma_px = np.sqrt(1 + sigma_x_given_z ** 2)
    log_px = -0.5 * (x_obs / sigma_px) ** 2 - np.log(sigma_px * np.sqrt(2 * np.pi))

    print(f"  x = {x_obs}")
    print(f"  q(z|x) = N({mu_q}, {sigma_q}^2)")
    print(f"  ELBO = {elbo:.4f}")
    print(f"  log p(x) = {log_px:.4f}")
    print(f"  Gap (KL): {log_px - elbo:.4f} >= 0: {log_px >= elbo - 0.01}")


# === Exercise 2: VAE Implementation and Experiments ===
# Problem: VAE on 2D Gaussian mixture, training curves, beta-VAE.

def exercise_2():
    """VAE implementation on 2D Gaussian mixture data."""
    np.random.seed(42)

    print("VAE Implementation on 2D Gaussian Mixture\n")

    # Generate 2D Gaussian mixture
    n_per_mode = 200
    centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    X_data = np.vstack([
        np.random.randn(n_per_mode, 2) * 0.5 + center
        for center in centers
    ])
    np.random.shuffle(X_data)

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(x):
        return np.maximum(0, x)

    class VAE:
        """Simple VAE with 2D latent space."""
        def __init__(self, d_input=2, d_latent=2, d_hidden=32):
            scale = 0.1
            self.d_latent = d_latent
            # Encoder: input -> hidden -> (mu, log_var)
            self.W_enc1 = np.random.randn(d_input, d_hidden) * scale
            self.b_enc1 = np.zeros(d_hidden)
            self.W_mu = np.random.randn(d_hidden, d_latent) * scale
            self.b_mu = np.zeros(d_latent)
            self.W_logvar = np.random.randn(d_hidden, d_latent) * scale
            self.b_logvar = np.zeros(d_latent)

            # Decoder: latent -> hidden -> output
            self.W_dec1 = np.random.randn(d_latent, d_hidden) * scale
            self.b_dec1 = np.zeros(d_hidden)
            self.W_dec2 = np.random.randn(d_hidden, d_input) * scale
            self.b_dec2 = np.zeros(d_input)

        def encode(self, x):
            h = relu(x @ self.W_enc1 + self.b_enc1)
            mu = h @ self.W_mu + self.b_mu
            log_var = h @ self.W_logvar + self.b_logvar
            return mu, log_var, h

        def reparameterize(self, mu, log_var):
            std = np.exp(0.5 * log_var)
            eps = np.random.randn(*mu.shape)
            return mu + std * eps

        def decode(self, z):
            h = relu(z @ self.W_dec1 + self.b_dec1)
            x_recon = h @ self.W_dec2 + self.b_dec2
            return x_recon, h

        def train_step(self, x_batch, lr=0.001, beta=1.0):
            batch_size = x_batch.shape[0]

            # Forward
            mu, log_var, h_enc = self.encode(x_batch)
            z = self.reparameterize(mu, log_var)
            x_recon, h_dec = self.decode(z)

            # Losses
            recon_loss = np.mean(np.sum((x_batch - x_recon) ** 2, axis=1))
            kl_loss = -0.5 * np.mean(np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=1))
            total_loss = recon_loss + beta * kl_loss

            # Backward (simplified gradient computation)
            # d_recon / d_x_recon
            d_recon = 2 * (x_recon - x_batch) / batch_size

            # Decoder gradients
            d_W_dec2 = h_dec.T @ d_recon
            d_b_dec2 = d_recon.sum(axis=0)
            d_h_dec = d_recon @ self.W_dec2.T
            d_h_dec *= (h_dec > 0)  # relu gradient
            d_W_dec1 = z.T @ d_h_dec
            d_b_dec1 = d_h_dec.sum(axis=0)
            d_z = d_h_dec @ self.W_dec1.T

            # KL gradients
            d_mu_kl = mu / batch_size * beta
            d_logvar_kl = 0.5 * (np.exp(log_var) - 1) / batch_size * beta

            # Encoder gradients (through reparameterization)
            d_mu_total = d_z + d_mu_kl
            d_logvar_total = d_z * (z - mu) * 0.5 + d_logvar_kl

            d_W_mu = h_enc.T @ d_mu_total
            d_b_mu = d_mu_total.sum(axis=0)
            d_W_logvar = h_enc.T @ d_logvar_total
            d_b_logvar = d_logvar_total.sum(axis=0)

            d_h_enc = d_mu_total @ self.W_mu.T + d_logvar_total @ self.W_logvar.T
            d_h_enc *= (h_enc > 0)
            d_W_enc1 = x_batch.T @ d_h_enc
            d_b_enc1 = d_h_enc.sum(axis=0)

            # Update weights
            clip = 5.0
            for param, grad in [
                (self.W_dec2, d_W_dec2), (self.b_dec2, d_b_dec2),
                (self.W_dec1, d_W_dec1), (self.b_dec1, d_b_dec1),
                (self.W_mu, d_W_mu), (self.b_mu, d_b_mu),
                (self.W_logvar, d_W_logvar), (self.b_logvar, d_b_logvar),
                (self.W_enc1, d_W_enc1), (self.b_enc1, d_b_enc1),
            ]:
                param -= lr * np.clip(grad, -clip, clip)

            return recon_loss, kl_loss, total_loss

    # (1) Train VAE (beta=1)
    print("(1) Training VAE (beta=1.0):")
    vae = VAE(d_input=2, d_latent=2, d_hidden=64)
    recon_losses = []
    kl_losses = []
    batch_size = 64

    for epoch in range(200):
        idx = np.random.permutation(len(X_data))
        epoch_recon = 0
        epoch_kl = 0
        n_batches = 0
        for i in range(0, len(X_data) - batch_size, batch_size):
            batch = X_data[idx[i:i + batch_size]]
            rl, kl, _ = vae.train_step(batch, lr=0.002, beta=1.0)
            epoch_recon += rl
            epoch_kl += kl
            n_batches += 1
        recon_losses.append(epoch_recon / n_batches)
        kl_losses.append(epoch_kl / n_batches)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: recon={recon_losses[-1]:.4f}, KL={kl_losses[-1]:.4f}")

    # (2) Visualize latent space
    print("\n(2) Latent space visualization:")
    mu_all, _, _ = vae.encode(X_data)
    print(f"  Latent space range: x=[{mu_all[:, 0].min():.2f}, {mu_all[:, 0].max():.2f}], "
          f"y=[{mu_all[:, 1].min():.2f}, {mu_all[:, 1].max():.2f}]")

    # (3) Grid sampling from decoder
    print("\n(3) Decoder grid sampling:")
    grid_size = 15
    z1 = np.linspace(-3, 3, grid_size)
    z2 = np.linspace(-3, 3, grid_size)
    z_grid = np.array([[a, b] for a in z1 for b in z2])
    x_decoded, _ = vae.decode(z_grid)
    print(f"  Generated {grid_size}x{grid_size}={len(z_grid)} points from grid")

    # (4) Beta-VAE comparison
    print("\n(4) Beta-VAE analysis:")
    betas = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_results = {}

    for beta in betas:
        vae_beta = VAE(d_input=2, d_latent=2, d_hidden=64)
        final_recon = 0
        final_kl = 0
        for epoch in range(150):
            idx = np.random.permutation(len(X_data))
            r_sum, k_sum, n_b = 0, 0, 0
            for i in range(0, len(X_data) - batch_size, batch_size):
                batch = X_data[idx[i:i + batch_size]]
                rl, kl, _ = vae_beta.train_step(batch, lr=0.002, beta=beta)
                r_sum += rl
                k_sum += kl
                n_b += 1
            final_recon = r_sum / n_b
            final_kl = k_sum / n_b

        mu_beta, logvar_beta, _ = vae_beta.encode(X_data)
        avg_var = np.mean(np.exp(logvar_beta))
        beta_results[beta] = (vae_beta, final_recon, final_kl, avg_var)
        print(f"  beta={beta:4.1f}: recon={final_recon:.4f}, KL={final_kl:.4f}, "
              f"avg_var={avg_var:.4f}")

    print("\n  Higher beta -> stronger KL penalty -> more Gaussian latent")
    print("  Lower beta -> better reconstruction, less structured latent")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Training curves
    axes[0, 0].plot(recon_losses, label='Reconstruction')
    axes[0, 0].plot(kl_losses, label='KL')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('VAE Training Curves (beta=1)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Latent space
    axes[0, 1].scatter(mu_all[:, 0], mu_all[:, 1], s=5, alpha=0.3)
    axes[0, 1].set_xlabel('z1')
    axes[0, 1].set_ylabel('z2')
    axes[0, 1].set_title('Latent Space (beta=1)')
    axes[0, 1].set_aspect('equal')

    # Generated points
    axes[0, 2].scatter(X_data[:, 0], X_data[:, 1], s=3, alpha=0.2, label='Data')
    axes[0, 2].scatter(x_decoded[:, 0], x_decoded[:, 1], s=5, c='red', alpha=0.5, label='Generated')
    axes[0, 2].set_title('Data vs Generated (grid)')
    axes[0, 2].legend()
    axes[0, 2].set_aspect('equal')

    # Beta comparison
    beta_plot = [0.1, 1.0, 5.0]
    for i, beta in enumerate(beta_plot):
        vae_b = beta_results[beta][0]
        mu_b, _, _ = vae_b.encode(X_data)
        axes[1, i].scatter(mu_b[:, 0], mu_b[:, 1], s=5, alpha=0.3)
        axes[1, i].set_title(f'Latent Space (beta={beta})')
        axes[1, i].set_xlabel('z1')
        axes[1, i].set_ylabel('z2')
        axes[1, i].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('ex18_2_vae.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_2_vae.png")


# === Exercise 3: Proof of Optimal Discriminator for GAN ===
# Problem: Prove D*(x) = p_data(x) / (p_data(x) + p_g(x)).

def exercise_3():
    """Proof and numerical verification of GAN optimal discriminator."""
    print("Proof of Optimal Discriminator for GAN\n")

    print("Discriminator objective (G fixed):")
    print("  max_D V(D, G) = E_{x~p_data} [log D(x)] + E_{x~p_g} [log(1-D(x))]")
    print()
    print("For any x, the integrand is:")
    print("  f(D(x)) = p_data(x) * log(D(x)) + p_g(x) * log(1 - D(x))")
    print()
    print("Taking derivative w.r.t. D(x) and setting to zero:")
    print("  df/dD = p_data(x)/D(x) - p_g(x)/(1-D(x)) = 0")
    print()
    print("Solving:")
    print("  p_data(x) * (1-D(x)) = p_g(x) * D(x)")
    print("  p_data(x) - p_data(x)*D(x) = p_g(x)*D(x)")
    print("  p_data(x) = D(x) * (p_data(x) + p_g(x))")
    print()
    print("  D*(x) = p_data(x) / (p_data(x) + p_g(x))  QED")
    print()
    print("Second derivative check:")
    print("  d^2f/dD^2 = -p_data/D^2 - p_g/(1-D)^2 < 0")
    print("  So D* is indeed a maximum.")

    # Numerical verification
    print("\nNumerical Verification (1D Gaussians):")
    np.random.seed(42)

    # p_data = N(0, 1), p_g = N(2, 1.5^2)
    x = np.linspace(-5, 8, 1000)
    p_data = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    p_g = np.exp(-(x - 2) ** 2 / (2 * 1.5 ** 2)) / (1.5 * np.sqrt(2 * np.pi))

    # Optimal discriminator
    D_star = p_data / (p_data + p_g + 1e-10)

    print(f"  p_data = N(0, 1), p_g = N(2, 1.5^2)")
    print(f"  D*(x=0) = {D_star[np.argmin(np.abs(x))]:. 4f} "
          f"(p_data high, p_g low -> close to 1)")
    print(f"  D*(x=2) = {D_star[np.argmin(np.abs(x - 2))]:. 4f} "
          f"(both present -> close to 0.5)")
    print(f"  D*(x=5) = {D_star[np.argmin(np.abs(x - 5))]:. 4f} "
          f"(p_data low, p_g low -> depends on ratio)")

    # When p_data = p_g, D* = 0.5 everywhere (Nash equilibrium)
    p_same = p_data.copy()
    D_star_equal = p_same / (p_same + p_same + 1e-10)
    print(f"\n  When p_g = p_data: D*(x) = {D_star_equal[500]:.4f} everywhere (= 0.5)")
    print("  This is the Nash equilibrium of the GAN minimax game.")

    # Value of V(D*, G) at optimality
    V_optimal = np.trapz(p_data * np.log(D_star + 1e-10) + p_g * np.log(1 - D_star + 1e-10), x)
    V_equilibrium = np.trapz(
        p_data * np.log(0.5) + p_data * np.log(0.5), x)
    print(f"\n  V(D*, G) = {V_optimal:.4f}")
    print(f"  V at equilibrium = {V_equilibrium:.4f} (= -2*log(2) = {-2 * np.log(2):.4f})")

    # Simple GAN training simulation (1D)
    print("\nSimple 1D GAN training:")

    def train_simple_gan(n_iter=1000):
        # Generator: shift and scale of standard normal
        g_mu = 5.0  # initial guess far from data
        g_sigma = 2.0
        # Discriminator: logistic regression with polynomial features
        d_weights = np.random.randn(5) * 0.1

        def disc_features(x_in):
            return np.column_stack([np.ones_like(x_in), x_in, x_in ** 2, x_in ** 3, x_in ** 4])

        def disc_output(x_in, w):
            features = disc_features(x_in)
            return 1 / (1 + np.exp(-features @ w))

        losses_d = []
        losses_g = []

        for it in range(n_iter):
            # Sample
            x_real = np.random.randn(200)
            x_fake = g_mu + g_sigma * np.random.randn(200)

            # Train discriminator
            for _ in range(3):
                D_real = disc_output(x_real, d_weights)
                D_fake = disc_output(x_fake, d_weights)

                # Gradient ascent on V
                feat_real = disc_features(x_real)
                feat_fake = disc_features(x_fake)
                grad_d = np.mean(feat_real * (1 - D_real)[:, np.newaxis], axis=0) - \
                          np.mean(feat_fake * D_fake[:, np.newaxis], axis=0)
                d_weights += 0.01 * grad_d

            # Train generator (minimize log(1-D(G(z))))
            z = np.random.randn(200)
            x_gen = g_mu + g_sigma * z
            D_gen = disc_output(x_gen, d_weights)

            # Gradient of -E[log(D(G(z)))] w.r.t. g_mu, g_sigma
            d_feat = disc_features(x_gen)
            d_D_dx = D_gen * (1 - D_gen)  # sigmoid derivative
            d_D_dx_feat = np.sum(d_feat * d_weights, axis=1) * d_D_dx

            grad_mu = -np.mean(d_D_dx_feat / (D_gen + 1e-10))
            grad_sigma = -np.mean(d_D_dx_feat * z / (D_gen + 1e-10))

            g_mu -= 0.01 * grad_mu
            g_sigma -= 0.005 * np.clip(grad_sigma, -1, 1)
            g_sigma = max(g_sigma, 0.1)

            if (it + 1) % 200 == 0:
                loss_d = np.mean(np.log(D_real + 1e-10)) + np.mean(np.log(1 - D_fake + 1e-10))
                print(f"    Iter {it + 1}: g_mu={g_mu:.2f}, g_sigma={g_sigma:.2f}, D_loss={loss_d:.3f}")

        return g_mu, g_sigma

    g_mu_final, g_sigma_final = train_simple_gan(1000)
    print(f"  Final generator: N({g_mu_final:.2f}, {g_sigma_final:.2f}^2)")
    print(f"  Target: N(0, 1)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(x, p_data, 'b-', label='p_data')
    axes[0].plot(x, p_g, 'r-', label='p_g')
    axes[0].set_title('Data and Generator Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, D_star, 'g-', linewidth=2)
    axes[1].axhline(0.5, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title('Optimal Discriminator D*(x)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('D*(x)')
    axes[1].grid(True, alpha=0.3)

    # f(D) for a specific x
    x_test = 1.0
    p_d_x = np.exp(-x_test ** 2 / 2) / np.sqrt(2 * np.pi)
    p_g_x = np.exp(-(x_test - 2) ** 2 / (2 * 1.5 ** 2)) / (1.5 * np.sqrt(2 * np.pi))
    D_range = np.linspace(0.01, 0.99, 100)
    f_D = p_d_x * np.log(D_range) + p_g_x * np.log(1 - D_range)
    D_star_x = p_d_x / (p_d_x + p_g_x)
    axes[2].plot(D_range, f_D, 'b-')
    axes[2].axvline(D_star_x, color='r', linestyle='--', label=f'D*={D_star_x:.3f}')
    axes[2].set_xlabel('D(x)')
    axes[2].set_ylabel('f(D)')
    axes[2].set_title(f'Objective at x={x_test}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex18_3_gan_discriminator.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_3_gan_discriminator.png")


# === Exercise 4: Forward Process of Diffusion Models ===
# Problem: Derive q(x_t|x_0), visualize forward diffusion, show convergence to N(0,I).

def exercise_4():
    """Diffusion model forward process: derivation, visualization, convergence."""
    print("Forward Process of Diffusion Models\n")

    print("Derivation of q(x_t | x_0):")
    print()
    print("Forward process: q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)")
    print()
    print("Define alpha_t = 1 - beta_t, alpha_bar_t = prod_{s=1}^{t} alpha_s")
    print()
    print("Recursive derivation:")
    print("  x_1 = sqrt(alpha_1)*x_0 + sqrt(1-alpha_1)*eps_1")
    print("  x_2 = sqrt(alpha_2)*x_1 + sqrt(1-alpha_2)*eps_2")
    print("       = sqrt(alpha_2)*[sqrt(alpha_1)*x_0 + sqrt(1-alpha_1)*eps_1] + sqrt(1-alpha_2)*eps_2")
    print("       = sqrt(alpha_1*alpha_2)*x_0 + sqrt(alpha_2(1-alpha_1))*eps_1 + sqrt(1-alpha_2)*eps_2")
    print()
    print("  The sum of two independent Gaussians:")
    print("  N(0, alpha_2(1-alpha_1)) + N(0, 1-alpha_2)")
    print("  = N(0, alpha_2 - alpha_1*alpha_2 + 1 - alpha_2)")
    print("  = N(0, 1 - alpha_1*alpha_2)")
    print("  = N(0, 1 - alpha_bar_2)")
    print()
    print("  By induction:")
    print("  x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps")
    print("  q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)  QED")

    # Schedule
    T = 1000
    beta = np.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)

    print(f"\n  Schedule: linear beta from {beta[0]:.4f} to {beta[-1]:.4f}, T={T}")
    print(f"  alpha_bar[0] = {alpha_bar[0]:.6f}")
    print(f"  alpha_bar[T//2] = {alpha_bar[T // 2]:.6f}")
    print(f"  alpha_bar[T-1] = {alpha_bar[-1]:.6f}")

    # (2) Visualize x_t for 1D data
    print("\nVisualization of q(x_t|x_0) for 1D data:")
    np.random.seed(42)

    # Bimodal data
    n_data = 1000
    x0 = np.concatenate([np.random.randn(n_data // 2) - 3,
                          np.random.randn(n_data // 2) + 3])

    timesteps = [0, 50, 100, 250, 500, 999]
    print(f"  Data: bimodal Gaussian (modes at -3 and +3)")

    for t in timesteps:
        if t == 0:
            xt = x0
        else:
            sqrt_ab = np.sqrt(alpha_bar[t])
            sqrt_1_ab = np.sqrt(1 - alpha_bar[t])
            xt = sqrt_ab * x0 + sqrt_1_ab * np.random.randn(len(x0))
        print(f"  t={t:4d}: mean={xt.mean():.3f}, std={xt.std():.3f}, "
              f"alpha_bar={alpha_bar[min(t, T - 1)]:.4f}")

    # (3) Show convergence to N(0, I)
    print(f"\n  Convergence: x_T ~ N(0, I)")
    print(f"  At t={T}: mean coeff = sqrt(alpha_bar_T) = {np.sqrt(alpha_bar[-1]):.6f} ~= 0")
    print(f"  At t={T}: var coeff = 1 - alpha_bar_T = {1 - alpha_bar[-1]:.6f} ~= 1")
    xT = np.sqrt(alpha_bar[-1]) * x0 + np.sqrt(1 - alpha_bar[-1]) * np.random.randn(len(x0))
    # KS test against normal
    xT_sorted = np.sort(xT)
    n_pts = len(xT_sorted)
    cdf_empirical = np.arange(1, n_pts + 1) / n_pts
    from scipy.special import erfc
    cdf_normal = 0.5 * erfc(-xT_sorted / np.sqrt(2))
    ks_stat = np.max(np.abs(cdf_empirical - cdf_normal))
    print(f"  KS statistic (x_T vs N(0,1)): {ks_stat:.4f} (should be small)")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, t in enumerate(timesteps):
        ax = axes[i // 3, i % 3]
        if t == 0:
            xt = x0
        else:
            xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * np.random.randn(len(x0))
        ax.hist(xt, bins=50, density=True, alpha=0.7, color='blue')
        # Overlay N(0,1)
        x_plot = np.linspace(-6, 6, 200)
        ax.plot(x_plot, np.exp(-x_plot ** 2 / 2) / np.sqrt(2 * np.pi),
                'r--', label='N(0,1)')
        ab = alpha_bar[min(t, T - 1)]
        ax.set_title(f't={t}, alpha_bar={ab:.4f}')
        ax.set_xlim([-6, 6])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('ex18_4_diffusion_forward.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_4_diffusion_forward.png")


# === Exercise 5: Flow Matching vs Diffusion Models ===
# Problem: Implement simple Flow Matching and DDPM, compare on 2D data.

def exercise_5():
    """Flow Matching vs DDPM comparison on 2D data."""
    np.random.seed(42)

    print("Flow Matching vs Diffusion Models\n")

    # 2D target data: circle
    n_data = 2000
    theta = 2 * np.pi * np.random.rand(n_data)
    r = 2 + 0.2 * np.random.randn(n_data)
    X_data = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    def relu(x):
        return np.maximum(0, x)

    class SimpleNN:
        """Tiny MLP for score/velocity prediction: input (x,t) -> output (x)."""
        def __init__(self, d_in, d_out, d_hidden=64):
            scale = 0.1
            self.W1 = np.random.randn(d_in, d_hidden) * scale
            self.b1 = np.zeros(d_hidden)
            self.W2 = np.random.randn(d_hidden, d_hidden) * scale
            self.b2 = np.zeros(d_hidden)
            self.W3 = np.random.randn(d_hidden, d_out) * scale
            self.b3 = np.zeros(d_out)

        def forward(self, x):
            h1 = relu(x @ self.W1 + self.b1)
            h2 = relu(h1 @ self.W2 + self.b2)
            return h2 @ self.W3 + self.b3

        def train_step(self, x, target, lr=0.001):
            # Forward
            h1_pre = x @ self.W1 + self.b1
            h1 = relu(h1_pre)
            h2_pre = h1 @ self.W2 + self.b2
            h2 = relu(h2_pre)
            pred = h2 @ self.W3 + self.b3

            # Loss
            diff = pred - target
            loss = np.mean(np.sum(diff ** 2, axis=1))

            batch = x.shape[0]
            # Backward
            d_pred = 2 * diff / batch
            d_W3 = h2.T @ d_pred
            d_b3 = d_pred.sum(axis=0)
            d_h2 = d_pred @ self.W3.T * (h2_pre > 0)
            d_W2 = h1.T @ d_h2
            d_b2 = d_h2.sum(axis=0)
            d_h1 = d_h2 @ self.W2.T * (h1_pre > 0)
            d_W1 = x.T @ d_h1
            d_b1 = d_h1.sum(axis=0)

            # Update
            clip = 5.0
            for param, grad in [(self.W1, d_W1), (self.b1, d_b1),
                                (self.W2, d_W2), (self.b2, d_b2),
                                (self.W3, d_W3), (self.b3, d_b3)]:
                param -= lr * np.clip(grad, -clip, clip)

            return loss

    # (1) Simple DDPM
    print("(1) Training DDPM (simplified):")
    T_ddpm = 100
    beta = np.linspace(1e-4, 0.02, T_ddpm)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)

    # Noise prediction network
    ddpm_net = SimpleNN(d_in=3, d_out=2, d_hidden=64)  # input: (x, t)

    batch_size = 128
    for epoch in range(300):
        idx = np.random.choice(n_data, batch_size)
        x0 = X_data[idx]

        # Random timestep
        t = np.random.randint(0, T_ddpm, batch_size)
        eps = np.random.randn(batch_size, 2)

        # Forward diffusion
        ab = alpha_bar[t][:, np.newaxis]
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps

        # Input: (xt, t/T)
        net_input = np.column_stack([xt, t / T_ddpm])

        # Train to predict noise
        loss = ddpm_net.train_step(net_input, eps, lr=0.003)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}: loss={loss:.4f}")

    # DDPM sampling
    print("  Sampling from DDPM...")
    n_gen = 500
    x = np.random.randn(n_gen, 2)

    for t in range(T_ddpm - 1, -1, -1):
        net_input = np.column_stack([x, np.full(n_gen, t / T_ddpm)])
        eps_pred = ddpm_net.forward(net_input)

        # DDPM update: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps)
        coeff1 = 1 / np.sqrt(alpha[t])
        coeff2 = beta[t] / np.sqrt(1 - alpha_bar[t])
        x = coeff1 * (x - coeff2 * eps_pred)

        if t > 0:
            noise = np.sqrt(beta[t]) * np.random.randn(n_gen, 2)
            x += noise

    samples_ddpm = x

    # (2) Simple Flow Matching
    print("\n(2) Training Flow Matching:")
    fm_net = SimpleNN(d_in=3, d_out=2, d_hidden=64)

    for epoch in range(300):
        idx = np.random.choice(n_data, batch_size)
        x1 = X_data[idx]  # target (data)
        x0 = np.random.randn(batch_size, 2)  # source (noise)

        # Random time
        t = np.random.rand(batch_size)[:, np.newaxis]

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        xt = (1 - t) * x0 + t * x1

        # Target velocity: v = x1 - x0 (conditional optimal transport)
        v_target = x1 - x0

        # Input: (xt, t)
        net_input = np.column_stack([xt, t])
        loss = fm_net.train_step(net_input, v_target, lr=0.003)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}: loss={loss:.4f}")

    # Flow Matching sampling (Euler integration)
    print("  Sampling from Flow Matching...")
    n_steps_list = [5, 10, 20, 50]
    fm_samples = {}

    for n_steps in n_steps_list:
        x = np.random.randn(n_gen, 2)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            net_input = np.column_stack([x, np.full(n_gen, t_val)])
            v = fm_net.forward(net_input)
            x = x + v * dt
        fm_samples[n_steps] = x

    # (3) Compare sampling speed
    print("\n(3) Sampling speed comparison:")
    print(f"  DDPM: {T_ddpm} steps (fixed)")
    for n_steps in n_steps_list:
        print(f"  Flow Matching: {n_steps} steps")

    # (4) Compare quality (simple metric: distance to nearest data point)
    print("\n(4) Quality comparison (mean distance to nearest data point):")

    def mean_nn_dist(samples, data):
        """Mean distance from generated samples to nearest real data point."""
        dists = np.sqrt(np.sum((samples[:, np.newaxis] - data[np.newaxis, :]) ** 2, axis=2))
        return np.mean(np.min(dists, axis=1))

    dist_ddpm = mean_nn_dist(samples_ddpm, X_data)
    print(f"  DDPM ({T_ddpm} steps): {dist_ddpm:.4f}")
    for n_steps in n_steps_list:
        dist_fm = mean_nn_dist(fm_samples[n_steps], X_data)
        print(f"  Flow Matching ({n_steps} steps): {dist_fm:.4f}")

    print("\n  Flow Matching achieves comparable quality with fewer steps")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].scatter(X_data[:, 0], X_data[:, 1], s=3, alpha=0.3)
    axes[0, 0].set_title('Target Data')
    axes[0, 0].set_aspect('equal')

    axes[0, 1].scatter(samples_ddpm[:, 0], samples_ddpm[:, 1], s=5, alpha=0.5, c='red')
    axes[0, 1].set_title(f'DDPM ({T_ddpm} steps)')
    axes[0, 1].set_aspect('equal')

    axes[0, 2].scatter(fm_samples[20][:, 0], fm_samples[20][:, 1], s=5, alpha=0.5, c='green')
    axes[0, 2].set_title('Flow Matching (20 steps)')
    axes[0, 2].set_aspect('equal')

    # FM with different steps
    for i, n_steps in enumerate([5, 10, 50]):
        axes[1, i].scatter(fm_samples[n_steps][:, 0], fm_samples[n_steps][:, 1],
                           s=5, alpha=0.5, c='green')
        axes[1, i].scatter(X_data[:500, 0], X_data[:500, 1], s=1, alpha=0.1, c='blue')
        axes[1, i].set_title(f'Flow Matching ({n_steps} steps)')
        axes[1, i].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('ex18_5_flow_matching.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_5_flow_matching.png")


# === Main ===

def main():
    exercises = [
        ("Exercise 1: Two Methods of ELBO Derivation", exercise_1),
        ("Exercise 2: VAE Implementation and Experiments", exercise_2),
        ("Exercise 3: Proof of Optimal Discriminator for GAN", exercise_3),
        ("Exercise 4: Forward Process of Diffusion Models", exercise_4),
        ("Exercise 5: Flow Matching vs Diffusion Models", exercise_5),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")
        func()


if __name__ == "__main__":
    main()

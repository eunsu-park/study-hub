[Previous: Generative Adversarial Networks (GAN)](./29_Impl_GAN.md) | [Next: Variational Autoencoder (VAE)](./31_Impl_VAE.md)

---

# 30. Generative Models - VAE (Variational Autoencoder)

## Learning Objectives

- Theoretical foundation of VAE (Variational Inference)
- Understanding Latent Space and probabilistic generation
- Deriving ELBO loss function
- Reparameterization Trick
- Beta-VAE and Disentanglement
- PyTorch implementation and visualization

---

## 1. VAE Theory

### Autoencoder vs VAE

```
Autoencoder:
    Input → Encoder → Latent vector z (deterministic) → Decoder → Reconstruction

VAE:
    Input → Encoder → Mean mu, Variance sigma → Sample z ~ N(mu, sigma) → Decoder → Reconstruction
                         ↓
                    Latent space is continuous and follows normal distribution
                    → Can generate new images
```

### Why Probabilistic?

```
Problems with regular Autoencoder:
- Latent space is discontinuous
- Strange output when inputting z not in training data
- Difficult to use as generative model

VAE solution:
- Regularize latent space to normal distribution
- Continuous latent space
- Can sample from arbitrary z ~ N(0, I) for generation
```

### Graphical Model

```
Generative Process:
    z ~ p(z) = N(0, I)           # Prior distribution
    x ~ p_theta(x|z)             # Decoder

Inference:
    q_phi(z|x) ≈ p(z|x)          # Encoder approximates posterior

Goal:
    Maximize log p(x) (data likelihood)
    → Maximize ELBO (Evidence Lower Bound)
```

---

## 2. ELBO Loss Function

### Derivation

```
log p(x) = log ∫ p(x, z) dz

         = log ∫ p(x|z) p(z) dz

         = log ∫ q(z|x) * [p(x|z) p(z) / q(z|x)] dz

         ≥ ∫ q(z|x) log[p(x|z) p(z) / q(z|x)] dz    (Jensen's inequality)

         = E_q[log p(x|z)] - KL(q(z|x) || p(z))

         = ELBO (Evidence Lower Bound)
```

### Meaning of Two Terms

```python
# ELBO = Reconstruction - KL Divergence

# 1. Reconstruction Term: E_q[log p(x|z)]
#    - How well decoder reconstructs x from z
#    - Reconstruction loss (MSE or BCE)

# 2. KL Divergence: KL(q(z|x) || p(z))
#    - How close encoded distribution is to prior N(0,I)
#    - Latent space regularization
```

### Loss Function Implementation

```python
def vae_loss(x, x_recon, mu, log_var):
    """VAE loss function (⭐⭐⭐)

    Args:
        x: original image (batch, ...)
        x_recon: reconstructed image (batch, ...)
        mu: mean (batch, latent_dim)
        log_var: log variance (batch, latent_dim)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (binary image: BCE, continuous image: MSE)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # Or MSE
    # recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss
```

---

## 3. Reparameterization Trick

### Problem

```
Sampling from z ~ q(z|x) = N(mu, sigma)

Problem: Sampling is not differentiable → Cannot backpropagate
```

### Solution: Reparameterization

```python
def reparameterize(mu, log_var):
    """Reparameterization Trick (⭐⭐⭐)

    z = mu + sigma * epsilon
    epsilon ~ N(0, I)

    This way, randomness is in epsilon,
    and we can differentiate with respect to mu, sigma
    """
    std = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log(sigma^2))
    eps = torch.randn_like(std)     # epsilon ~ N(0, I)
    z = mu + std * eps
    return z
```

### Visual Understanding

```
[Not differentiable]
mu, sigma → Sampling → z → Decoder

[Differentiable - Reparameterization]
mu ──────────────┐
                 │
                 ▼
sigma ────────▶ (mu + sigma * eps) ──▶ z ──▶ Decoder
                       ▲
eps ~ N(0, I) ─────────┘ (treated as constant)
```

---

## 4. VAE Model Implementation

### Encoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """VAE Encoder (⭐⭐⭐)

    image → mu, log_var
    """
    def __init__(self, in_channels=1, latent_dim=20):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 14 → 7
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
```

### Decoder

```python
class VAEDecoder(nn.Module):
    """VAE Decoder (⭐⭐⭐)

    z → image
    """
    def __init__(self, latent_dim=20, out_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 14 → 28
            nn.Sigmoid()  # [0, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        x_recon = self.deconv_layers(h)
        return x_recon
```

### Complete VAE

```python
class VAE(nn.Module):
    """Variational Autoencoder (⭐⭐⭐)"""
    def __init__(self, in_channels=1, latent_dim=20):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def generate(self, num_samples, device):
        """Generate new samples"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x):
        """Reconstruct images"""
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
        return x_recon
```

---

## 5. Training Loop

```python
def train_vae(model, dataloader, epochs=50, lr=1e-3):
    """VAE training (⭐⭐⭐)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)

            optimizer.zero_grad()

            # Forward
            x_recon, mu, log_var = model(data)

            # Loss
            loss, recon_loss, kl_loss = vae_loss(data, x_recon, mu, log_var)

            # Normalize by batch size
            loss = loss / data.size(0)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item() / data.size(0)
            total_kl += kl_loss.item() / data.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

    return model
```

---

## 6. Beta-VAE

### Idea

```
ELBO = Reconstruction - beta * KL

beta > 1: Greater weight on KL term
    → Latent space more regularized
    → Disentangled representations
    → Each latent dimension captures independent feature

beta = 1: Regular VAE
beta < 1: Focus on reconstruction
```

### Implementation

```python
def beta_vae_loss(x, x_recon, mu, log_var, beta=4.0):
    """Beta-VAE loss function (⭐⭐⭐)"""
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Beta weighting
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
```

### Disentanglement Example

```
Beta-VAE trained on MNIST (beta=4):
    z[0]: Digit tilt
    z[1]: Line thickness
    z[2]: Digit type
    ...

Independently controlling each dimension changes only that feature
```

---

## 7. Latent Space Visualization

### 2D Latent Space

```python
def visualize_latent_space(model, dataloader, device):
    """Latent space visualization (⭐⭐)"""
    model.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu())
            labels.append(label)

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('VAE Latent Space')
    plt.savefig('latent_space.png')
    plt.close()
```

### Latent Space Exploration

```python
def explore_latent_dimension(model, dim_idx, range_vals, fixed_z, device):
    """Explore a specific latent dimension (⭐⭐)"""
    model.eval()
    images = []

    with torch.no_grad():
        for val in range_vals:
            z = fixed_z.clone()
            z[0, dim_idx] = val
            img = model.decoder(z.to(device))
            images.append(img.cpu())

    return torch.cat(images, dim=0)
```

### Manifold Generation

```python
def generate_manifold(model, n=20, latent_dim=2, device='cpu'):
    """Generate manifold over 2D latent space (⭐⭐⭐)"""
    model.eval()

    # Generate grid in range (-3, 3)
    grid_x = torch.linspace(-3, 3, n)
    grid_y = torch.linspace(-3, 3, n)

    figure = np.zeros((28 * n, 28 * n))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.zeros(1, latent_dim)
                z[0, 0] = xi
                z[0, 1] = yi

                x_decoded = model.decoder(z.to(device))
                digit = x_decoded[0, 0].cpu().numpy()

                figure[i * 28:(i + 1) * 28,
                       j * 28:(j + 1) * 28] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.savefig('vae_manifold.png')
    plt.close()
```

---

## 8. VAE vs GAN Comparison

| Characteristic | VAE | GAN |
|---------------|-----|-----|
| Training method | Likelihood maximization | Adversarial training |
| Loss function | ELBO (explicit) | Min-max (implicit) |
| Training stability | Stable | Unstable |
| Image quality | Tends to be blurry | Sharp |
| Latent space | Structured | Hard to interpret |
| Mode Coverage | Good | Mode Collapse possible |
| Density estimation | Possible | Not possible |

### Advantages and Disadvantages

```
VAE Advantages:
- Explicit density model
- Stable training
- Meaningful latent space
- Can both reconstruct and generate

VAE Disadvantages:
- Blurry images due to reconstruction loss
- KL term limits latent space expressiveness

GAN Advantages:
- Sharp high-quality images
- Implicit density → More flexible

GAN Disadvantages:
- Unstable training
- Mode Collapse
- Difficult to evaluate
```

---

## 9. Advanced VAE Variants

### Conditional VAE (CVAE)

```python
class CVAE(nn.Module):
    """Conditional VAE (⭐⭐⭐)

    Given a condition (e.g. class label), generate a specific type
    """
    def __init__(self, in_channels=1, latent_dim=20, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Concatenate condition as one-hot
        self.encoder = CVAEEncoder(in_channels, latent_dim, num_classes)
        self.decoder = CVAEDecoder(latent_dim, in_channels, num_classes)
        self.latent_dim = latent_dim

    def forward(self, x, label):
        # One-hot encoding
        y = F.one_hot(label, self.num_classes).float()

        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z, y)

        return x_recon, mu, log_var

    def generate(self, label, num_samples, device):
        """Generate samples for a specific class"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = F.one_hot(label, self.num_classes).float().to(device)
        y = y.expand(num_samples, -1)
        return self.decoder(z, y)
```

### VQ-VAE (Vector Quantized VAE)

```python
# VQ-VAE uses a discrete codebook instead of continuous latent space
# Effective for high-quality image/audio generation

class VectorQuantizer(nn.Module):
    """Vector quantization for VQ-VAE (⭐⭐⭐⭐)"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z: (batch, channels, H, W)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Find nearest codebook vector
        distances = torch.cdist(z_flat, self.embeddings.weight)
        indices = torch.argmin(distances, dim=1)
        z_q = self.embeddings(indices).view(z.shape[0], z.shape[2], z.shape[3], -1)
        z_q = z_q.permute(0, 3, 1, 2)

        # Loss: codebook learning + commitment loss
        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, indices
```

---

## 10. Complete MNIST VAE Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configuration
latent_dim = 20
batch_size = 128
epochs = 30
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
transform = transforms.ToTensor()
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Model
model = VAE(in_channels=1, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        x_recon, mu, log_var = model(data)

        # Loss
        recon_loss = F.binary_cross_entropy(x_recon, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = (recon_loss + kl_loss) / data.size(0)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {train_loss / len(train_loader):.4f}")

# Generation
model.eval()
with torch.no_grad():
    samples = model.generate(16, device)
    # save or visualize...

print("VAE training complete!")
```

---

## Summary

### Key Concepts

1. **VAE**: Generative model with probabilistic latent space
2. **ELBO**: Reconstruction + KL Divergence
3. **Reparameterization**: z = mu + sigma * epsilon
4. **Beta-VAE**: Control KL weight for disentanglement
5. **Latent space**: Continuous, structured

### Core Code

```python
# Encoder output
mu, log_var = encoder(x)

# Reparameterization
std = torch.exp(0.5 * log_var)
z = mu + std * torch.randn_like(std)

# Decoder
x_recon = decoder(z)

# Loss
recon = F.binary_cross_entropy(x_recon, x)
kl = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
loss = recon + kl
```

### Use Case Scenarios

| Purpose | Recommended Method |
|---------|-------------------|
| Data generation | VAE or GAN |
| Latent space analysis | VAE (especially Beta-VAE) |
| High-quality images | GAN or VQ-VAE |
| Conditional generation | CVAE |
| Compression/reconstruction | VAE |

---

## Exercises

### Exercise 1: Derive the KL Divergence Term

For a Gaussian encoder `q(z|x) = N(mu, sigma^2 I)` and a standard normal prior `p(z) = N(0, I)`, derive the closed-form KL divergence:

```
KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

1. Start from the definition: `KL(q || p) = E_q[log q(z|x) - log p(z)]`.
2. Substitute the Gaussian probability density functions.
3. Simplify term by term to arrive at the formula used in `vae_loss`.
4. Why does this closed-form expression eliminate the need to estimate the KL term via sampling?

### Exercise 2: Visualize the Reparameterization Trick

Implement a small experiment to demonstrate that the reparameterization trick enables gradient flow:
1. Create a simple 2D Gaussian encoder: `mu, log_var = some_network(x)`.
2. Sample `z` using `z = mu + std * eps` (reparameterized).
3. Compute a simple loss `loss = z.sum()` and call `loss.backward()`.
4. Verify that `mu.grad` and `log_var.grad` are non-None after backpropagation.
5. Explain in your own words why direct sampling `z ~ N(mu, sigma)` would fail this test.

### Exercise 3: Train a VAE on MNIST and Visualize the Latent Space

Using the provided `VAE` implementation:
1. Train for 30 epochs with `latent_dim=2` on MNIST.
2. After training, use `visualize_latent_space` to plot the 2D latent space colored by digit class.
3. Use `generate_manifold` to generate a 20×20 grid of images spanning `(-3, 3)` in both latent dimensions.
4. Describe what you observe: are the digit classes well-separated? Is the latent space smooth and continuous?

### Exercise 4: Experiment with Beta-VAE

Modify the training loop to use `beta_vae_loss` with different beta values:
1. Train three models with `beta=0.5`, `beta=1.0` (standard VAE), and `beta=4.0`.
2. For each model, visualize the 2D latent space.
3. Compare reconstruction quality vs. latent space structure: how does increasing beta affect the trade-off?
4. Qualitatively assess disentanglement: does `beta=4.0` produce latent dimensions that correspond more clearly to independent visual features (e.g., rotation, thickness)?

### Exercise 5: Implement a Conditional VAE (CVAE)

Extend the VAE to generate specific digit classes:
1. Modify the encoder to accept a one-hot class label concatenated to the flattened image features.
2. Modify the decoder to accept a one-hot label concatenated to the latent vector `z`.
3. Train the CVAE on MNIST and implement the `generate` method to produce `num_samples` images of a specified digit.
4. Generate 10 samples of each digit class (0-9) and display them in a 10×10 grid. Verify that the conditional generation produces the correct digit class.

---

## Next Steps

Learn about the latest generative model, Diffusion models, in [32_Diffusion_Models.md](./32_Diffusion_Models.md).

[Previous: TensorBoard Visualization](./27_TensorBoard.md) | [Next: Generative Adversarial Networks (GAN)](./29_Impl_GAN.md)

---

# 28. Generative Models - GAN (Generative Adversarial Networks)

## Learning Objectives

- Understand GAN fundamentals and adversarial training
- Design Generator and Discriminator architectures
- Various loss functions (Adversarial, Wasserstein, WGAN-GP)
- Implement DCGAN architecture
- Apply training stabilization techniques
- Understand StyleGAN concepts

---

## 1. GAN Fundamentals

### Concept

```
GAN = Competitive learning between two neural networks

Generator (G): Creates fake data
    Random noise z → Fake image

Discriminator (D): Classifies real/fake
    Image → Real(1) / Fake(0)

Goal: G generates fakes good enough to fool D
```

### Adversarial Training

```
┌────────────┐     ┌────────────┐
│  Noise z   │────▶│ Generator  │───┬──▶ Fake image
└────────────┘     └────────────┘   │
                                    │
┌────────────┐                      ▼
│ Real image │────────────────────▶ Discriminator ──▶ Real/Fake
└────────────┘

D training: Maximize accuracy of Real=1, Fake=0 classification
G training: Make D classify fakes as real
```

### Min-Max Game

```python
# GAN objective function (min-max game):
#
#   min_G max_D  V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
#
# Symbol definitions:
#   D(·)  — Discriminator: maps an image to probability of being real
#   G(·)  — Generator: maps latent noise z to a synthetic image
#   x     — sample from the real data distribution p_data
#   z     — sample from a prior noise distribution (typically N(0, I))
#   E_x   — expectation over real data;  E_z — expectation over noise

# D's goal: Maximize V(D, G)
#   - D(x) → 1 (classify real as real)
#   - D(G(z)) → 0 (classify fake as fake)

# G's goal: Minimize V(D, G)
#   - D(G(z)) → 1 (make D classify fake as real)
```

---

## 2. Basic GAN Implementation

### Generator

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """Simple Generator (⭐⭐)"""
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()  # output: [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)
```

### Discriminator

```python
class Discriminator(nn.Module):
    """Simple Discriminator (⭐⭐)"""
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # output: [0, 1] probability
        )

    def forward(self, img):
        return self.model(img)
```

### Training Loop

```python
def train_gan(generator, discriminator, dataloader, epochs=100, latent_dim=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ==================
            # Train Discriminator
            # ==================
            optimizer_D.zero_grad()

            # Real images
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)

            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            # .detach() is critical: D's loss must NOT backpropagate into G.
            # Without detach, optimizer_D.step() would update D to also
            # minimize G's generation quality — breaking the adversarial game.
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ==================
            # Train Generator
            # ==================
            optimizer_G.zero_grad()

            # Pass fakes through D again (without detach this time, so
            # gradients flow from D's output back through G's parameters).
            # We use *real* labels because G wants D to believe fakes are real.
            # This is the "non-saturating" loss: instead of minimizing
            # log(1 - D(G(z))) (which saturates when D is confident), we
            # maximize log(D(G(z))) — same optimum, but stronger gradients
            # early in training when G is still poor.
            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)

            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
```

---

## 3. Loss Functions

### Vanilla GAN Loss (BCE)

```python
# Binary Cross Entropy
criterion = nn.BCELoss()

# D loss
d_loss = criterion(D(real), 1) + criterion(D(G(z)), 0)

# G loss (non-saturating)
g_loss = criterion(D(G(z)), 1)  # -log(D(G(z)))

# G loss (original, saturating)
# g_loss = -criterion(D(G(z)), 0)  # log(1 - D(G(z))) - rarely used
```

### Wasserstein Loss (WGAN)

```python
def wasserstein_loss(y_pred, y_true):
    """Wasserstein distance (Earth Mover's Distance)"""
    return torch.mean(y_pred * y_true)

# D (Critic) loss - maximize
d_loss = torch.mean(D(real)) - torch.mean(D(G(z)))
# → To minimize, flip sign: -D(real) + D(G(z))

# G loss - minimize
g_loss = -torch.mean(D(G(z)))

# Weight Clipping (WGAN)
for p in discriminator.parameters():
    p.data.clamp_(-0.01, 0.01)
```

### WGAN-GP (Gradient Penalty)

```python
def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    """Gradient Penalty for WGAN-GP (⭐⭐⭐)"""
    batch_size = real_imgs.size(0)

    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolated.requires_grad_(True)

    # Discriminator output
    d_interpolated = discriminator(interpolated)

    # Gradient computation
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    # Gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty: (||grad|| - 1)^2
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty

# WGAN-GP Loss
lambda_gp = 10
gp = gradient_penalty(discriminator, real_imgs, fake_imgs, device)
d_loss = -torch.mean(D(real)) + torch.mean(D(G(z))) + lambda_gp * gp
```

### Hinge Loss

```python
# Discriminator loss
d_loss_real = torch.mean(torch.relu(1.0 - D(real)))
d_loss_fake = torch.mean(torch.relu(1.0 + D(G(z))))
d_loss = d_loss_real + d_loss_fake

# Generator loss
g_loss = -torch.mean(D(G(z)))
```

---

## 4. DCGAN Architecture

### Core Principles

```
1. Remove pooling → Strided Conv (D), Transposed Conv (G)
2. Use BatchNorm (except G output layer, D input layer)
3. ReLU in G (Tanh for output layer)
4. LeakyReLU in D
```

### DCGAN Generator

```python
class DCGANGenerator(nn.Module):
    """DCGAN Generator (⭐⭐⭐)

    z (100,) → (1024, 4, 4) → (512, 8, 8) → (256, 16, 16) → (128, 32, 32) → (3, 64, 64)
    """
    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()

        self.main = nn.Sequential(
            # Input: z (latent_dim,)
            # Output: (ngf*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8, 4, 4) → (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4, 8, 8) → (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2, 16, 16) → (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf, 32, 32) → (nc, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # z: (batch, latent_dim) → (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)
```

### DCGAN Discriminator

```python
class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator (⭐⭐⭐)

    (3, 64, 64) → (64, 32, 32) → (128, 16, 16) → (256, 8, 8) → (512, 4, 4) → (1,)
    """
    def __init__(self, nc=3, ndf=64):
        super().__init__()

        self.main = nn.Sequential(
            # (nc, 64, 64) → (ndf, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf, 32, 32) → (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2, 16, 16) → (ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4, 8, 8) → (ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8, 4, 4) → (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        return self.main(img).view(-1, 1)
```

---

## 5. Training Stabilization Techniques

### Spectral Normalization

```python
from torch.nn.utils import spectral_norm

class SNDiscriminator(nn.Module):
    """Discriminator with Spectral Normalization (⭐⭐⭐)"""
    def __init__(self, nc=3, ndf=64):
        super().__init__()

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        )

    def forward(self, img):
        return self.main(img).view(-1, 1)
```

### Label Smoothing

```python
# One-sided label smoothing
real_labels = torch.ones(batch_size, 1, device=device) * 0.9  # 1.0 → 0.9
fake_labels = torch.zeros(batch_size, 1, device=device)

# Or noisy labels
real_labels = 0.7 + 0.3 * torch.rand(batch_size, 1, device=device)  # [0.7, 1.0]
```

### Two Time-Scale Update Rule (TTUR)

```python
# D uses higher learning rate
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.9))
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
```

### Progressive Growing

```python
# Start at small resolution and progressively grow
# ProGAN (Progressive GAN)

resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Train at each resolution for some epochs, then move to next
# Fade-in: progressively add new layers
```

---

## 6. StyleGAN Overview

### Core Ideas

```
Traditional GAN: z → G → Image
StyleGAN: z → Mapping Network → w → Synthesis Network → Image

Mapping Network: 8-layer MLP, transforms z to "disentangled" w
AdaIN: Uses w to inject style into each layer
```

### Mapping Network

```python
class MappingNetwork(nn.Module):
    """StyleGAN Mapping Network (⭐⭐⭐⭐)"""
    def __init__(self, latent_dim=512, w_dim=512, num_layers=8):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(latent_dim if i == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)
```

### AdaIN (Adaptive Instance Normalization)

```python
class AdaIN(nn.Module):
    """Adaptive Instance Normalization (⭐⭐⭐⭐)"""
    def __init__(self, num_features, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.style = nn.Linear(w_dim, num_features * 2)  # scale + bias

    def forward(self, x, w):
        # x: (batch, channels, H, W)
        # w: (batch, w_dim)

        normalized = self.norm(x)

        style = self.style(w)  # (batch, channels*2)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * normalized + beta
```

### Style Mixing

```python
# Generate w from different z values
z1, z2 = torch.randn(2, latent_dim)
w1, w2 = mapping(z1), mapping(z2)

# Use w1 up to certain layer, w2 after
# → Style mixing (coarse: w1, fine: w2)
```

---

## 7. Image Generation and Evaluation

### Generating Sample Images

```python
def generate_samples(generator, latent_dim, num_samples=64):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_imgs = generator(z)
    return fake_imgs

# Visualization
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def show_generated_images(images, nrow=8):
    """Display grid of generated images"""
    # [-1, 1] → [0, 1]
    images = (images + 1) / 2
    grid = vutils.make_grid(images.cpu(), nrow=nrow, normalize=False)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig('generated_samples.png')
    plt.close()
```

### Interpolation

```python
def interpolate_latent(generator, z1, z2, steps=10):
    """Latent space interpolation (⭐⭐)"""
    generator.eval()
    images = []

    with torch.no_grad():
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z.unsqueeze(0))
            images.append(img)

    return torch.cat(images, dim=0)

# Spherical interpolation (slerp) - better results
def slerp(z1, z2, alpha):
    """Spherical linear interpolation"""
    omega = torch.acos((z1 * z2).sum() / (z1.norm() * z2.norm()))
    return (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * z1 + \
           (torch.sin(alpha * omega) / torch.sin(omega)) * z2
```

### FID (Frechet Inception Distance)

```python
# FID computation (using pytorch-fid library)
# pip install pytorch-fid

# from pytorch_fid import fid_score
# fid = fid_score.calculate_fid_given_paths(
#     [real_images_path, fake_images_path],
#     batch_size=50,
#     device=device,
#     dims=2048
# )
# Lower is better (0 is perfect)
```

---

## 8. Complete MNIST GAN Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 100
lr = 0.0002
batch_size = 64
epochs = 50

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

mnist = datasets.MNIST('data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Simple models
G = Generator(latent_dim=latent_dim, img_shape=(1, 28, 28)).to(device)
D = Discriminator(img_shape=(1, 28, 28)).to(device)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Training
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # Labels
        real = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train D
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)

        d_loss = criterion(D(real_imgs), real) + criterion(D(fake_imgs.detach()), fake)
        d_loss.backward()
        optimizer_D.step()

        # Train G
        optimizer_G.zero_grad()
        g_loss = criterion(D(fake_imgs), real)
        g_loss.backward()
        optimizer_G.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: D={d_loss.item():.4f}, G={g_loss.item():.4f}")

        # Save samples
        with torch.no_grad():
            sample = G(torch.randn(16, latent_dim, device=device))
            # save code...
```

---

## Summary

### Key Concepts

1. **GAN**: Adversarial training between Generator and Discriminator
2. **Loss Functions**: BCE, Wasserstein, Gradient Penalty
3. **DCGAN**: Transposed Conv + BatchNorm + LeakyReLU
4. **Stabilization**: Spectral Norm, TTUR, Progressive Growing
5. **StyleGAN**: Style control via Mapping Network + AdaIN

### GAN Training Tips

```
1. Maintain D and G balance (if D is too strong, G cannot learn)
2. BatchNorm applies to entire minibatch (don't separate real/fake)
3. Use Adam beta1=0.5 (reduce momentum)
4. Learning rate: typically 0.0001 ~ 0.0002
5. Periodically check generated images
```

### Mode Collapse

Mode collapse is the most common GAN failure mode: **G learns to produce only a few types of output** instead of covering the full data distribution.

**Why it happens**: G finds a single output (or a few) that reliably fools D, then has no incentive to diversify. D eventually catches on and shifts its decision boundary, but G simply jumps to a different narrow mode — creating an oscillation rather than convergence.

**Signs of mode collapse**:
- Generated images look nearly identical across different noise inputs
- FID score improves but then plateaus or oscillates

**Mitigations**:
- **WGAN/WGAN-GP**: The Wasserstein distance provides meaningful gradients even when D is confident, encouraging G to cover more modes
- **Spectral Normalization**: Constrains D's Lipschitz constant, preventing it from becoming too sharp and forcing G into a corner
- **Diversity regularization**: Penalize G when different z inputs produce similar outputs

### Loss Function Comparison

| Loss Function | Advantages | Disadvantages |
|--------------|-----------|---------------|
| BCE | Simple | Mode collapse |
| WGAN | Stable training | Weight clipping |
| WGAN-GP | Very stable | Computational cost |
| Hinge | Simple and effective | - |

---

## Exercises

### Exercise 1: Explain the Min-Max Game

In your own words, explain why GAN training is formulated as a min-max game. Specifically:
1. What objective does the Discriminator maximize and why?
2. What objective does the Generator minimize and why?
3. Why does the "non-saturating" Generator loss (`-log D(G(z))`) work better than the original formulation (`log(1 - D(G(z)))`) early in training? Hint: consider what happens to the gradient when D is confident.

### Exercise 2: Implement Weight Initialization for DCGAN

The DCGAN paper specifies that all Conv and BatchNorm weights should be initialized from a Normal distribution with `mean=0.0, std=0.02`. Review the `_init_weights` method in `DCGANGenerator` and answer:
1. Why is custom weight initialization important for GAN stability compared to PyTorch's default initialization?
2. Modify the `DCGANGenerator` to also initialize the `BatchNorm` bias to `0` using `nn.init.constant_`. Verify your change matches what `DCGANDiscriminator._init_weights` already does.

### Exercise 3: Train a Vanilla GAN on MNIST

Using the provided `Generator`, `Discriminator`, and training loop code, train a GAN on the MNIST dataset:
1. Set `latent_dim=100`, `batch_size=64`, `epochs=50`, `lr=0.0002`.
2. Save a grid of 16 generated images every 10 epochs using `show_generated_images`.
3. Plot the discriminator loss and generator loss curves over training.
4. Observe and describe: does either loss converge? What does the training behavior tell you about the adversarial game?

### Exercise 4: Compare WGAN vs Vanilla GAN

Replace the BCE loss in the training loop with the Wasserstein loss:
1. Implement the WGAN training loop with weight clipping (`clamp_(-0.01, 0.01)` after each discriminator step).
2. Remove the `Sigmoid` activation from the Discriminator (WGAN uses an unconstrained "critic").
3. Train both Vanilla GAN and WGAN on MNIST for 20 epochs.
4. Compare the generated image quality and training stability. Document your observations: which version shows more stable loss curves?

### Exercise 5: Implement and Evaluate Latent Space Interpolation

Using a trained MNIST GAN:
1. Sample two random latent vectors `z1` and `z2`.
2. Implement both linear interpolation and spherical interpolation (`slerp`) between `z1` and `z2` with 10 steps.
3. Generate and visualize the image sequence for both interpolation methods.
4. Explain why spherical interpolation tends to produce smoother transitions than linear interpolation for Gaussian latent spaces. Hint: think about the expected norm of a random Gaussian vector and what happens in the middle of a linear interpolation.

---

## Next Steps

Learn about VAE (Variational Autoencoder) in [30_Generative_Models_VAE.md](./30_Generative_Models_VAE.md).

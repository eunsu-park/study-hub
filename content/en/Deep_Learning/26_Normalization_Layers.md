[Previous: Optimizers](./25_Optimizers.md) | [Next: TensorBoard Visualization](./27_TensorBoard.md)

---

# 26. Normalization Layers

## Learning Objectives

- Understand the motivation for normalization in deep learning and how it smooths the loss landscape
- Master Batch Normalization, Layer Normalization, Group Normalization, and their use cases
- Learn RMSNorm and why it's preferred in modern large language models
- Implement normalization layers from scratch and understand their computational implications
- Apply the right normalization technique based on architecture and batch size constraints

---

## 1. Why Normalization?

### 1.1 The Problem: Internal Covariate Shift

**Internal Covariate Shift** refers to the change in the distribution of network activations during training. As parameters in earlier layers change, the inputs to later layers shift, forcing them to continuously adapt.

**Original motivation** (Ioffe & Szegedy, 2015):
- Stabilize activation distributions across layers
- Allow each layer to learn on a more stable input distribution
- Enable higher learning rates without divergence

### 1.2 Modern Understanding: Loss Landscape Smoothing

Recent research (Santurkar et al., 2018) shows normalization's primary benefit is **smoothing the loss landscape**:

```
Without Normalization:          With Normalization:

    |\                              /\
    | \        /\                  /  \
    |  \  /\  /  \                /    \
    |___\/  \/____\__           /______\_____

    Rough, irregular             Smoother, more predictable
    gradients                    gradients
```

**Benefits**:
1. **Faster convergence** — smoother gradients allow larger steps
2. **Higher learning rates** — reduced risk of divergence
3. **Regularization effect** — noise from batch statistics acts as implicit regularization
4. **Reduced sensitivity to initialization** — less dependence on careful weight initialization

### 1.3 Normalization Axes

Different normalization methods normalize across different dimensions:

```
Input tensor shape: (N, C, H, W)
N = batch size
C = channels
H, W = spatial dimensions

┌─────────────────────────────────────────────────────────────┐
│  Batch Norm:     normalize across N     for each (C, H, W)  │
│  Layer Norm:     normalize across C,H,W for each N          │
│  Instance Norm:  normalize across H,W   for each (N, C)     │
│  Group Norm:     normalize across C/G,H,W for each (N, G)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Batch Normalization

### 2.1 Core Concept

**Batch Normalization** (BatchNorm) normalizes activations across the batch dimension for each feature independently.

**Algorithm**:

```
Input: mini-batch B = {x₁, ..., xₘ}
Parameters: γ (scale), β (shift) — learnable

1. Calculate batch statistics:
   μ_B = (1/m) Σ xᵢ                    # mean
   σ²_B = (1/m) Σ (xᵢ - μ_B)²          # variance

2. Normalize:
   x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)       # ε for numerical stability

3. Scale and shift:
   yᵢ = γ x̂ᵢ + β                        # learnable transformation
```

**Why scale and shift?** The network can learn to undo normalization if needed (e.g., `γ = √σ²`, `β = μ` recovers the original distribution).

### 2.2 Training vs Inference Mode

**Training**:
- Use batch statistics (μ_B, σ²_B)
- Update running estimates for inference:
  ```
  running_mean = momentum × running_mean + (1 - momentum) × μ_B
  running_var = momentum × running_var + (1 - momentum) × σ²_B
  ```

**Inference**:
- Use running statistics (fixed)
- No dependence on current batch

```python
import torch
import torch.nn as nn

# Training mode
bn = nn.BatchNorm2d(64)
bn.train()
out = bn(x)  # Uses batch statistics

# Inference mode
bn.eval()
out = bn(x)  # Uses running statistics
```

### 2.3 Where to Place BatchNorm?

**Option 1: After activation** (original paper)
```
Linear/Conv → Activation → BatchNorm
```

**Option 2: Before activation** (common practice)
```
Linear/Conv → BatchNorm → Activation
```

**Modern consensus**: Before activation works better in practice, especially with ReLU.

### 2.4 PyTorch BatchNorm

```python
import torch.nn as nn

# For fully connected layers (1D)
bn1d = nn.BatchNorm1d(num_features=128)

# For convolutional layers (2D)
bn2d = nn.BatchNorm2d(num_features=64)

# For 3D convolutions (video, volumetric data)
bn3d = nn.BatchNorm3d(num_features=32)

# Example CNN block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Usage
model = ConvBlock(3, 64)
x = torch.randn(32, 3, 224, 224)  # (N, C, H, W)
out = model(x)
print(out.shape)  # torch.Size([32, 64, 224, 224])
```

### 2.5 Manual Implementation

```python
import torch
import torch.nn as nn

class BatchNorm2dManual(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running statistics (not updated by gradient descent)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x shape: (N, C, H, W)

        if self.training:
            # Calculate batch statistics
            # Mean and var across (N, H, W) for each channel C
            mean = x.mean(dim=[0, 2, 3], keepdim=False)  # Shape: (C,)
            var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=False)  # Shape: (C,)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        # Reshape for broadcasting: (1, C, 1, 1)
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = gamma * x_norm + beta

        return out

# Test
manual_bn = BatchNorm2dManual(64)
pytorch_bn = nn.BatchNorm2d(64)

x = torch.randn(32, 64, 16, 16)

# Training mode
manual_bn.train()
pytorch_bn.train()
out_manual = manual_bn(x)
out_pytorch = pytorch_bn(x)

print(f"Output shape: {out_manual.shape}")
print(f"Mean close to 0: {out_manual.mean().item():.6f}")
print(f"Std close to 1: {out_manual.std().item():.6f}")
```

### 2.6 Limitations of BatchNorm

**1. Batch size dependency**
- Small batches → noisy statistics → poor performance
- Batch size < 8 is problematic

**2. Sequence models (RNNs)**
- Different sequence lengths in a batch
- Hard to apply across time dimension

**3. Distributed training**
- Each GPU has a different batch
- Sync BatchNorm needed (expensive)

**4. Online learning**
- Single sample at a time
- No batch statistics available

---

## 3. Layer Normalization

### 3.1 Core Concept

**Layer Normalization** (LayerNorm) normalizes across all features for each sample independently, making it batch-independent.

```
BatchNorm:  normalize across samples    for each feature
LayerNorm:  normalize across features   for each sample
```

**Formula**:

```
For each sample x in batch:
  μ = (1/D) Σ xᵢ                        # mean across features
  σ² = (1/D) Σ (xᵢ - μ)²                # variance across features
  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)             # normalize
  yᵢ = γ x̂ᵢ + β                         # scale and shift
```

### 3.2 Why LayerNorm for Transformers?

**Advantages**:
1. **Batch-independent** — works with batch size = 1
2. **Sequence-length independent** — each position normalized the same way
3. **Deterministic at inference** — no running statistics needed

**Transformer architecture**:

```
┌─────────────────────────────────────────┐
│  Pre-Norm (modern):                     │
│    x → LayerNorm → Attention → Add(x)   │
│    x → LayerNorm → FFN → Add(x)         │
│                                          │
│  Post-Norm (original):                  │
│    x → Attention → Add(x) → LayerNorm   │
│    x → FFN → Add(x) → LayerNorm         │
└─────────────────────────────────────────┘
```

**Pre-Norm vs Post-Norm**:
- **Pre-Norm**: Better gradient flow, easier to train, used in GPT, LLaMA
- **Post-Norm**: Original Transformer design, slightly better performance with careful tuning

### 3.3 PyTorch LayerNorm

```python
import torch
import torch.nn as nn

# LayerNorm for Transformers
ln = nn.LayerNorm(512)  # d_model = 512

# Example: Self-Attention with Pre-Norm
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Pre-Norm
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

# Usage
model = TransformerBlock(d_model=512, num_heads=8)
x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
out = model(x)
print(out.shape)  # torch.Size([32, 100, 512])
```

### 3.4 Manual Implementation

```python
import torch
import torch.nn as nn

class LayerNormManual(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # x shape: (N, ..., normalized_shape)
        # E.g., (N, seq_len, d_model) for Transformers

        # Calculate mean and variance across the last dimensions
        # Keep dims for broadcasting
        dims = list(range(-len(self.gamma.shape) if isinstance(self.normalized_shape, tuple)
                          else -1, 0))

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        return out

# Test
manual_ln = LayerNormManual(512)
pytorch_ln = nn.LayerNorm(512)

x = torch.randn(32, 100, 512)  # (batch, seq, features)

out_manual = manual_ln(x)
out_pytorch = pytorch_ln(x)

print(f"Output shape: {out_manual.shape}")
print(f"Mean per sample close to 0: {out_manual[0].mean().item():.6f}")
print(f"Std per sample close to 1: {out_manual[0].std().item():.6f}")
```

### 3.5 Use Cases

**Best for**:
- Transformers (BERT, GPT, ViT)
- RNNs, LSTMs
- Small batch sizes
- Variable sequence lengths

---

## 4. Group Normalization

### 4.1 Core Concept

**Group Normalization** (GroupNorm) divides channels into groups and normalizes within each group.

```
Input: (N, C, H, W)
Groups: G

Split C channels into G groups of (C/G) channels each
Normalize each group separately

Special cases:
  G = 1     → Layer Normalization (one group = all channels)
  G = C     → Instance Normalization (each channel is a group)
  G = 32    → Common choice (Wu & He, 2018)
```

**Visualization**:

```
Channels: [c0, c1, c2, c3, c4, c5, c6, c7]
Groups (G=4): [c0,c1] [c2,c3] [c4,c5] [c6,c7]

For each sample in batch:
  For each group:
    Calculate mean/var across (C/G, H, W)
    Normalize
```

### 4.2 Formula

```
For each sample n, group g:
  μₙ,ₘ = (1/(C/G · H · W)) Σ x_n,g,h,w
  σ²ₙ,ₘ = (1/(C/G · H · W)) Σ (x_n,g,h,w - μₙ,ₘ)²
  x̂_n,g,h,w = (x_n,g,h,w - μₙ,ₘ) / √(σ²ₙ,ₘ + ε)
  y_n,c,h,w = γ_c · x̂_n,c,h,w + β_c
```

### 4.3 PyTorch GroupNorm

```python
import torch
import torch.nn as nn

# GroupNorm with 32 groups (common choice)
gn = nn.GroupNorm(num_groups=32, num_channels=64)

# Example: ResNet block with GroupNorm
class ResNetBlock(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out

# Usage
model = ResNetBlock(channels=64, groups=32)
x = torch.randn(4, 64, 56, 56)  # Small batch size!
out = model(x)
print(out.shape)  # torch.Size([4, 64, 56, 56])
```

### 4.4 Choosing Number of Groups

```python
import torch
import torch.nn as nn

# Rule: num_channels must be divisible by num_groups

# Common configurations
configs = [
    (64, 32),   # 64 channels, 32 groups → 2 channels/group
    (128, 32),  # 128 channels, 32 groups → 4 channels/group
    (256, 32),  # 256 channels, 32 groups → 8 channels/group
]

for channels, groups in configs:
    gn = nn.GroupNorm(groups, channels)
    x = torch.randn(2, channels, 16, 16)  # Small batch!
    out = gn(x)
    print(f"{channels} channels, {groups} groups → "
          f"{channels // groups} channels/group, shape: {out.shape}")

# Special cases
gn_layer = nn.GroupNorm(1, 64)      # G=1 → LayerNorm behavior
gn_instance = nn.GroupNorm(64, 64)  # G=C → InstanceNorm behavior
```

### 4.5 Use Cases

**Best for**:
- Object detection (Mask R-CNN, Faster R-CNN)
- Image segmentation
- Small batch sizes (batch size = 1, 2, 4)
- Transfer learning with frozen BatchNorm
- Scenarios where BatchNorm statistics are unreliable

**Performance**:
- COCO object detection: GroupNorm matches BatchNorm with large batches
- With small batches (1-4): GroupNorm significantly outperforms BatchNorm

---

## 5. Instance Normalization

### 5.1 Core Concept

**Instance Normalization** (InstanceNorm) normalizes each channel of each sample independently.

```
For each sample, for each channel:
  Calculate mean and variance across spatial dimensions (H, W)
  Normalize
```

**Equivalent to GroupNorm with G = C** (each channel is its own group).

### 5.2 Formula

```
For each sample n, channel c:
  μₙ,c = (1/(H · W)) Σ x_n,c,h,w
  σ²ₙ,c = (1/(H · W)) Σ (x_n,c,h,w - μₙ,c)²
  x̂_n,c,h,w = (x_n,c,h,w - μₙ,c) / √(σ²ₙ,c + ε)
  y_n,c,h,w = γ_c · x̂_n,c,h,w + β_c
```

### 5.3 PyTorch InstanceNorm

```python
import torch
import torch.nn as nn

# For 2D images
in2d = nn.InstanceNorm2d(64)

# For 1D sequences
in1d = nn.InstanceNorm1d(128)

# Example: Style Transfer Network
class StyleTransferBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

# Usage
model = StyleTransferBlock(64)
x = torch.randn(1, 64, 256, 256)  # Batch size = 1 is fine!
out = model(x)
print(out.shape)  # torch.Size([1, 64, 256, 256])
```

### 5.4 Why Instance Normalization?

**Key insight**: For style transfer, we want to normalize out instance-specific contrast information.

**Example**:
```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Load content and style images
content = Image.open('content.jpg')
style = Image.open('style.jpg')

# Style transfer with InstanceNorm
class FastStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with InstanceNorm
        self.residual = nn.Sequential(
            *[self._residual_block(64) for _ in range(5)]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 9, padding=4),
            nn.Tanh()
        )

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x
```

### 5.5 Use Cases

**Best for**:
- Style transfer (neural style, fast style transfer)
- Image-to-image translation (pix2pix, CycleGAN)
- Generative models (GANs for image synthesis)
- Texture synthesis

---

## 6. RMSNorm (Root Mean Square Normalization)

### 6.1 Core Concept

**RMSNorm** simplifies LayerNorm by removing mean centering, normalizing only by the root mean square.

**Key difference**:
```
LayerNorm:  x̂ = (x - μ) / σ           # center then scale
RMSNorm:    x̂ = x / RMS(x)            # scale only
```

**Formula**:
```
RMS(x) = √((1/n) Σ xᵢ²)
x̂ᵢ = xᵢ / RMS(x)
yᵢ = γ · x̂ᵢ                            # scale (no bias β)
```

### 6.2 Why RMSNorm?

**Advantages**:
1. **Simpler computation** — no mean calculation or subtraction
2. **Faster** — ~10-15% speedup in large models
3. **Similar performance** — empirically matches LayerNorm
4. **Widely adopted** — LLaMA, LLaMA 2, LLaMA 3, Gemma, Mistral

**Computational savings**:
```
LayerNorm:  2 passes (mean, then variance) + 2 ops (subtract, divide)
RMSNorm:    1 pass (RMS) + 1 op (divide)
```

### 6.3 Manual Implementation

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (..., dim)

        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_norm = x / rms
        out = self.weight * x_norm

        return out

# Test
rms_norm = RMSNorm(512)
x = torch.randn(32, 100, 512)  # (batch, seq, features)
out = rms_norm(x)

print(f"Output shape: {out.shape}")
print(f"RMS: {torch.sqrt(torch.mean(out[0] ** 2)).item():.6f}")  # Should be ~1.0
```

### 6.4 Comparison with LayerNorm

```python
import torch
import torch.nn as nn
import time

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

# Benchmark
dim = 4096
seq_len = 2048
batch_size = 8

x = torch.randn(batch_size, seq_len, dim, device='cuda')

layer_norm = nn.LayerNorm(dim).cuda()
rms_norm = RMSNorm(dim).cuda()

# Warmup
for _ in range(10):
    _ = layer_norm(x)
    _ = rms_norm(x)

# LayerNorm timing
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = layer_norm(x)
torch.cuda.synchronize()
ln_time = time.time() - start

# RMSNorm timing
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = rms_norm(x)
torch.cuda.synchronize()
rms_time = time.time() - start

print(f"LayerNorm: {ln_time:.4f}s")
print(f"RMSNorm:   {rms_time:.4f}s")
print(f"Speedup:   {ln_time / rms_time:.2f}x")
```

### 6.5 RMSNorm in LLaMA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMATransformerBlock(nn.Module):
    """LLaMA-style transformer block with RMSNorm."""

    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()

        # Pre-normalization with RMSNorm
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=False),
            nn.SiLU(),  # LLaMA uses SiLU (Swish) activation
            nn.Linear(mlp_ratio * dim, dim, bias=False)
        )

    def forward(self, x):
        # Attention with RMSNorm
        h = self.attn_norm(x)
        h = self.attn(h, h, h)[0]
        x = x + h

        # FFN with RMSNorm
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x

# Example usage
model = LLaMATransformerBlock(dim=4096, num_heads=32)
x = torch.randn(1, 2048, 4096)  # (batch, seq_len, dim)
out = model(x)
print(out.shape)  # torch.Size([1, 2048, 4096])
```

### 6.6 Use Cases

**Best for**:
- Large language models (LLaMA, Mistral, Gemma)
- Any Transformer-based model where speed matters
- Models trained from scratch (not fine-tuning LayerNorm models)

---

## 7. Other Normalization Techniques

### 7.1 Weight Normalization

**Weight Normalization** reparameterizes weight vectors to decouple magnitude and direction.

```
Original:     w
Reparameterized:   w = g · (v / ||v||)

g = scalar magnitude (learnable)
v = direction vector (learnable)
```

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# Apply weight normalization to a layer
linear = nn.Linear(128, 64)
linear = weight_norm(linear, name='weight')

# The weight is reparameterized as: weight = g * v / ||v||
print(linear.weight_g)  # magnitude parameter
print(linear.weight_v)  # direction parameter

# Forward pass
x = torch.randn(32, 128)
out = linear(x)

# Remove weight normalization (merge g and v back into weight)
linear = nn.utils.remove_weight_norm(linear)
```

**Use cases**: RNNs, GANs, reinforcement learning (A3C)

### 7.2 Spectral Normalization

**Spectral Normalization** constrains the spectral norm (largest singular value) of weight matrices to 1, stabilizing GAN training.

```
Spectral norm: σ(W) = max singular value of W
Normalized weight: W_SN = W / σ(W)
```

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Apply spectral normalization
conv = nn.Conv2d(3, 64, 3, padding=1)
conv = spectral_norm(conv)

# Discriminator with Spectral Normalization (for GANs)
class SNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, 4, stride=1, padding=0)),
        )

    def forward(self, x):
        return self.model(x)

# Usage
disc = SNDiscriminator()
x = torch.randn(16, 3, 64, 64)
out = disc(x)
print(out.shape)  # torch.Size([16, 1, 5, 5])
```

**Use cases**: GAN discriminators (SNGAN, BigGAN, StyleGAN2)

### 7.3 Adaptive Instance Normalization (AdaIN)

**AdaIN** adaptively adjusts InstanceNorm statistics based on style input, enabling real-time style transfer.

```
AdaIN(content, style) = σ(style) · ((content - μ(content)) / σ(content)) + μ(style)

Transfer style statistics (mean, std) to content features
```

```python
import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        # content, style: (N, C, H, W)

        # Calculate statistics
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True)

        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True)

        # Normalize content, then apply style statistics
        normalized = (content - content_mean) / (content_std + 1e-5)
        stylized = normalized * style_std + style_mean

        return stylized

# Style transfer with AdaIN
class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.adain = AdaIN()
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, content, style):
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)

        # AdaIN layer
        t = self.adain(content_feat, style_feat)

        out = self.decoder(t)
        return out

# Usage
model = StyleTransferNet()
content = torch.randn(1, 3, 256, 256)
style = torch.randn(1, 3, 256, 256)
stylized = model(content, style)
print(stylized.shape)  # torch.Size([1, 3, 256, 256])
```

**Use cases**: Real-time style transfer, image-to-image translation

### 7.4 Comparison Table

| Method | Normalizes Across | Learnable Params | Batch-Dependent | Use Case |
|--------|-------------------|------------------|-----------------|----------|
| **Batch Norm** | (N) for each (C,H,W) | γ, β, running stats | Yes | CNNs, large batches |
| **Layer Norm** | (C,H,W) for each N | γ, β | No | Transformers, RNNs |
| **Instance Norm** | (H,W) for each (N,C) | γ, β (optional) | No | Style transfer, GANs |
| **Group Norm** | (C/G,H,W) for each (N,G) | γ, β | No | Detection, small batches |
| **RMSNorm** | (C,H,W) for each N | γ | No | LLMs, fast Transformers |
| **Weight Norm** | Weight vectors | g, v | No | RNNs, GANs |
| **Spectral Norm** | Weight matrices | — | No | GAN discriminators |
| **AdaIN** | (H,W) conditioned on style | — | No | Style transfer |

---

## 8. Comprehensive Comparison

### 8.1 Visual Comparison

```
Input tensor: (N, C, H, W)
N = batch (4 samples)
C = channels (3)
H, W = height, width (32 × 32)

┌─────────────────────────────────────────────────────────────────────┐
│  Batch Normalization                                                │
│  ┌──────┬──────┬──────┬──────┐                                      │
│  │ N=0  │ N=1  │ N=2  │ N=3  │                                      │
│  ├──────┼──────┼──────┼──────┤  For each (C, H, W) position:       │
│  │ c=0  │ c=0  │ c=0  │ c=0  │  Calculate mean/var across N        │
│  │ h,w  │ h,w  │ h,w  │ h,w  │  (4 values)                          │
│  └──────┴──────┴──────┴──────┘                                      │
│  Normalize: (x - μ_batch) / σ_batch                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Layer Normalization                                                │
│  ┌───────────────────────────┐                                      │
│  │       N=0                 │  For each sample:                    │
│  ├───────┬───────┬───────────┤  Calculate mean/var across           │
│  │ c=0   │ c=1   │ c=2       │  all (C, H, W)                       │
│  │ (all  │ (all  │ (all      │  (3 × 32 × 32 = 3072 values)         │
│  │ h,w)  │ h,w)  │ h,w)      │                                      │
│  └───────┴───────┴───────────┘                                      │
│  Normalize: (x - μ_layer) / σ_layer                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Instance Normalization                                             │
│  ┌────────────────┐                                                 │
│  │  N=0, C=0      │  For each (sample, channel):                    │
│  ├────────────────┤  Calculate mean/var across                      │
│  │   (all h,w)    │  (H, W)                                         │
│  │                │  (32 × 32 = 1024 values)                        │
│  └────────────────┘                                                 │
│  Normalize: (x - μ_instance) / σ_instance                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Group Normalization (G=3, so 1 channel per group)                 │
│  ┌────────────────┐                                                 │
│  │ N=0, G=0       │  For each (sample, group):                      │
│  │ (c=0 only)     │  Calculate mean/var across                      │
│  ├────────────────┤  (C/G, H, W)                                    │
│  │   (all h,w)    │  (1 × 32 × 32 = 1024 values)                    │
│  └────────────────┘                                                 │
│  Normalize: (x - μ_group) / σ_group                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 When to Use Which?

#### Decision Tree

```
Are you using a Transformer?
  ├─ Yes → LayerNorm or RMSNorm
  │         ├─ Speed critical? → RMSNorm (LLaMA, Mistral)
  │         └─ Otherwise → LayerNorm (BERT, ViT)
  │
  └─ No → Are you using a CNN?
           ├─ Yes → Is batch size large (≥16)?
           │         ├─ Yes → BatchNorm
           │         └─ No → GroupNorm
           │
           └─ No → Is it a GAN or style transfer?
                     ├─ Yes → InstanceNorm or AdaIN
                     └─ No → LayerNorm (safe default)
```

#### Architecture-Specific Recommendations

**Convolutional Neural Networks (CNNs)**:
```python
# Standard classification (ImageNet)
# Batch size: 32-256
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),  # ← BatchNorm for large batches
            nn.ReLU(inplace=True)
        )
```

**Object Detection / Segmentation**:
```python
# Mask R-CNN, Faster R-CNN
# Batch size: 1-4 (limited by GPU memory for large images)
class DetectionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(32, 64),  # ← GroupNorm for small batches
            nn.ReLU(inplace=True)
        )
```

**Transformers (Vision or Language)**:
```python
# BERT, GPT, ViT
class TransformerEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)  # ← LayerNorm standard
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
```

**Large Language Models**:
```python
# LLaMA, Mistral, Gemma
class LLMTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)  # ← RMSNorm for speed
        self.ffn_norm = RMSNorm(dim)
```

**Generative Adversarial Networks (GANs)**:
```python
# Generator: InstanceNorm for style
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),  # ← InstanceNorm
            nn.ReLU(inplace=True)
        )

# Discriminator: Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, stride=2, padding=1)),  # ← SpectralNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
```

### 8.3 Performance Benchmarks

```python
import torch
import torch.nn as nn
import time

def benchmark_normalization(norm_layer, input_shape, num_iterations=1000):
    """Benchmark a normalization layer."""
    x = torch.randn(*input_shape, device='cuda')

    # Warmup
    for _ in range(10):
        _ = norm_layer(x)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = norm_layer(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed

# Test configuration
batch_size = 32
channels = 256
height, width = 56, 56
input_shape = (batch_size, channels, height, width)

# Normalization layers
norms = {
    'BatchNorm2d': nn.BatchNorm2d(channels).cuda(),
    'GroupNorm (G=32)': nn.GroupNorm(32, channels).cuda(),
    'InstanceNorm2d': nn.InstanceNorm2d(channels).cuda(),
}

print(f"Input shape: {input_shape}")
print(f"Iterations: 1000\n")

results = {}
for name, norm in norms.items():
    norm.eval()
    elapsed = benchmark_normalization(norm, input_shape)
    results[name] = elapsed
    print(f"{name:20s}: {elapsed:.4f}s")

# Relative speeds
baseline = results['BatchNorm2d']
print("\nRelative to BatchNorm2d:")
for name, elapsed in results.items():
    print(f"{name:20s}: {elapsed / baseline:.2f}x")
```

**Typical results** (RTX 3090):
```
BatchNorm2d         : 0.1234s  (1.00x)
GroupNorm (G=32)    : 0.1456s  (1.18x)
InstanceNorm2d      : 0.1389s  (1.13x)
```

**For Transformers** (seq_len=2048, d_model=4096):
```
LayerNorm           : 0.2145s  (1.00x)
RMSNorm             : 0.1876s  (0.87x)  ← ~13% faster
```

---

## 9. Practical Tips

### 9.1 Initialization of γ and β

**Default initialization** (PyTorch):
```python
# γ (scale) initialized to 1
# β (shift) initialized to 0
# This preserves the original distribution initially
```

**Special cases**:

```python
import torch.nn as nn

# Initialize γ to 0 for residual blocks (He et al., 2019)
# "Fixup Initialization" — helps very deep networks
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Zero-initialize the last BatchNorm in each block
        nn.init.constant_(self.bn2.weight, 0)  # γ = 0
        nn.init.constant_(self.bn2.bias, 0)    # β = 0

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)  # Initially outputs 0, so out = identity
        out += identity
        out = F.relu(out)
        return out
```

### 9.2 Interaction with Weight Initialization

**BatchNorm makes networks less sensitive to weight initialization**, but you should still use proper initialization:

```python
import torch.nn as nn
import torch.nn.init as init

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # He initialization for ReLU
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
```

### 9.3 Normalization and Learning Rate

**Key insight**: Normalization allows higher learning rates.

```python
import torch.optim as optim

# Without normalization
model_no_norm = MyModel(use_norm=False)
optimizer = optim.SGD(model_no_norm.parameters(), lr=0.01)  # Conservative LR

# With BatchNorm/LayerNorm
model_with_norm = MyModel(use_norm=True)
optimizer = optim.SGD(model_with_norm.parameters(), lr=0.1)  # 10x higher LR!

# Modern Transformers with LayerNorm
# Can use even higher learning rates with warmup
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

**Learning rate scaling rule** (Goyal et al., 2017):
```
When increasing batch size by k, increase learning rate by k
(only works with BatchNorm!)

Batch 256, LR 0.1  →  Batch 1024, LR 0.4
```

### 9.4 Common Bugs and Pitfalls

#### Bug 1: Forgetting model.eval()

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

# Training mode
model.train()
x = torch.randn(1, 3, 32, 32)
out_train = model(x)

# Inference — WRONG! Still using batch statistics
# out_test = model(x)  # Bug: still in training mode!

# Inference — CORRECT
model.eval()  # Switch to eval mode!
out_test = model(x)

print(f"Outputs same? {torch.allclose(out_train, out_test)}")  # False
```

#### Bug 2: Wrong dimension ordering

```python
# WRONG: (seq_len, batch, features) for LayerNorm
x = torch.randn(100, 32, 512)  # (seq, batch, features)
ln = nn.LayerNorm(512)
out = ln(x)  # Works, but normalizes last dim only

# CORRECT: Normalize across features (last dim)
# Make sure your tensor layout matches the normalized_shape!

# For (batch, seq, features) — standard now
x = torch.randn(32, 100, 512)  # (batch, seq, features)
ln = nn.LayerNorm(512)
out = ln(x)  # Correct: normalizes across features dim
```

#### Bug 3: GroupNorm with incompatible channels

```python
# WRONG: num_channels not divisible by num_groups
try:
    gn = nn.GroupNorm(num_groups=32, num_channels=50)  # 50 % 32 != 0
except ValueError as e:
    print(f"Error: {e}")

# CORRECT: ensure divisibility
gn = nn.GroupNorm(num_groups=32, num_channels=64)  # 64 % 32 == 0 ✓
```

#### Bug 4: BatchNorm with batch_size = 1

```python
# PROBLEM: BatchNorm with single sample
bn = nn.BatchNorm2d(64)
x = torch.randn(1, 64, 32, 32)  # Batch size = 1

bn.train()
out = bn(x)  # Variance = 0! (single sample)
# Results in NaN or unstable training

# SOLUTION 1: Use GroupNorm or LayerNorm
gn = nn.GroupNorm(32, 64)
out = gn(x)  # Works fine with batch_size=1

# SOLUTION 2: Set BatchNorm to eval mode
bn.eval()
out = bn(x)  # Uses running statistics
```

#### Bug 5: Mixing frozen and trainable BatchNorm

```python
# PROBLEM: Fine-tuning with frozen BatchNorm statistics
pretrained_model = torchvision.models.resnet50(pretrained=True)

# Freeze all parameters
for param in pretrained_model.parameters():
    param.requires_grad = False

# This is NOT enough! BatchNorm still uses training mode statistics
pretrained_model.train()  # BUG: BatchNorm in training mode!

# SOLUTION: Set to eval mode OR set BatchNorm modules to eval
pretrained_model.eval()  # Safe for inference

# For fine-tuning:
def set_bn_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

pretrained_model.train()
pretrained_model.apply(set_bn_eval)  # Keep BatchNorm in eval mode during training
```

### 9.5 Best Practices Checklist

✅ **Always call `model.eval()` before inference**

✅ **Match normalization to your architecture**:
   - CNNs (large batch) → BatchNorm
   - CNNs (small batch) → GroupNorm
   - Transformers → LayerNorm or RMSNorm
   - GANs → InstanceNorm (G), SpectralNorm (D)

✅ **Use appropriate initialization** (Kaiming for ReLU, Xavier for Tanh)

✅ **Increase learning rate** when using normalization

✅ **For transfer learning**, consider freezing BatchNorm statistics or replacing with GroupNorm

✅ **Monitor running statistics** — ensure they stabilize during training

✅ **For distributed training**, use SyncBatchNorm if needed:
```python
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

---

## Exercises

### Exercise 1: Implement and Compare Normalization Methods

Implement a CNN with different normalization methods and compare their performance on CIFAR-10.

**Tasks**:
1. Create four identical CNNs, each using a different normalization:
   - BatchNorm2d
   - GroupNorm (32 groups)
   - LayerNorm
   - No normalization (baseline)
2. Train each for 20 epochs on CIFAR-10
3. Plot training curves (loss and accuracy)
4. Report final test accuracy for each
5. Experiment with batch sizes [4, 16, 64] and observe which normalization is most robust

**Starter code**:
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class CIFAR10Net(nn.Module):
    def __init__(self, norm_type='batch'):
        super().__init__()
        self.norm_type = norm_type

        def conv_block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]

            if norm_type == 'batch':
                layers.append(nn.BatchNorm2d(out_c))
            elif norm_type == 'group':
                layers.append(nn.GroupNorm(32, out_c))
            elif norm_type == 'layer':
                # LayerNorm for 2D: normalize over (C, H, W)
                layers.append(nn.GroupNorm(1, out_c))  # G=1 is LayerNorm
            # 'none': no normalization

            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),
            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# TODO: Implement training loop and comparison
```

### Exercise 2: RMSNorm vs LayerNorm in Transformers

Implement a small Transformer and compare RMSNorm vs LayerNorm in terms of speed and performance.

**Tasks**:
1. Implement a character-level language model (predict next character)
2. Train two versions: one with LayerNorm, one with RMSNorm
3. Use a text dataset (e.g., Shakespeare, WikiText-2)
4. Measure:
   - Training time per epoch
   - Final perplexity
   - Inference speed
5. Analyze: Does RMSNorm match LayerNorm performance while being faster?

**Starter code**:
```python
import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, norm_type='layer'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        # Choose normalization
        if norm_type == 'layer':
            norm_cls = lambda: nn.LayerNorm(d_model)
        elif norm_type == 'rms':
            norm_cls = lambda: RMSNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, norm_cls)
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]

        for layer in self.layers:
            x = layer(x)

        logits = self.output(x)
        return logits

# TODO: Implement training and benchmarking
```

### Exercise 3: Adaptive Instance Normalization for Style Transfer

Implement a simple style transfer network using AdaIN.

**Tasks**:
1. Implement an encoder-decoder architecture with AdaIN in the middle
2. Use a pre-trained VGG network as the encoder (freeze weights)
3. Train a decoder to reconstruct images
4. Implement the AdaIN layer that transfers style statistics
5. Test on content and style images (use torchvision datasets or your own)
6. Visualize the stylized output
7. **Bonus**: Implement controllable style transfer (α parameter to blend content/style)

**Starter code**:
```python
import torch
import torch.nn as nn
from torchvision import models

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        # TODO: Implement AdaIN
        # 1. Calculate mean and std of content
        # 2. Calculate mean and std of style
        # 3. Normalize content, apply style statistics
        pass

class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: VGG19 (frozen)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])  # Up to relu4_1
        for param in self.encoder.parameters():
            param.requires_grad = False

        # AdaIN layer
        self.adain = AdaIN()

        # Decoder: mirror of encoder
        self.decoder = nn.Sequential(
            # TODO: Implement decoder (reverse of encoder)
            # Use ConvTranspose2d or Upsample + Conv2d
        )

    def forward(self, content, style):
        # TODO:
        # 1. Encode content and style
        # 2. Apply AdaIN
        # 3. Decode
        pass

# TODO: Implement training loop with perceptual loss
```

---

## References

1. **Batch Normalization**:
   - Ioffe & Szegedy (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML.
   - Santurkar et al. (2018). "How Does Batch Normalization Help Optimization?" NeurIPS.

2. **Layer Normalization**:
   - Ba et al. (2016). "Layer Normalization." arXiv:1607.06450.

3. **Group Normalization**:
   - Wu & He (2018). "Group Normalization." ECCV.

4. **Instance Normalization**:
   - Ulyanov et al. (2016). "Instance Normalization: The Missing Ingredient for Fast Stylization." arXiv:1607.08022.

5. **RMSNorm**:
   - Zhang & Sennrich (2019). "Root Mean Square Layer Normalization." NeurIPS.
   - Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971.

6. **Spectral Normalization**:
   - Miyato et al. (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR.

7. **AdaIN**:
   - Huang & Belongie (2017). "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization." ICCV.

8. **Comprehensive Analysis**:
   - Bjorck et al. (2018). "Understanding Batch Normalization." NeurIPS.
   - Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv:1706.02677.

9. **PyTorch Documentation**:
   - https://pytorch.org/docs/stable/nn.html#normalization-layers
   - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
   - https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
   - https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

10. **Practical Guides**:
    - He et al. (2019). "Bag of Tricks for Image Classification with Convolutional Neural Networks." CVPR.
    - Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.

# 21. Vision Transformer (ViT)

[Previous: GPT](./20_Impl_GPT.md) | [Next: Vision Transformer (ViT) Implementation](./22_Impl_ViT.md)

---

## Learning Objectives

- Understanding Vision Transformer architecture
- Patch Embedding principles
- CLS token and Position Embedding
- ViT variants (DeiT, Swin Transformer)
- PyTorch implementation and applications

---

## 1. Vision Transformer Overview

### Core Idea

```
Traditional CNN: local features → global features (hierarchical)
ViT: Convert image to patch sequence → process with Transformer

Image (224×224) → 196 patches of 16×16 → Transformer Encoder
```

### Why Transformer for Vision?

```
1. Self-Attention Advantages
   - Captures long-range dependencies
   - Considers global context

2. Scalability
   - Surpasses CNN on large datasets
   - Easy to scale

3. Architecture Unification
   - Can unify Vision + Language
   - Favorable for multimodal learning
```

---

## 2. ViT Architecture

### Overall Structure

```
Input Image (224×224×3)
        ↓
[Patch Embedding] → 196 patch vectors (each 768-dim)
        ↓
[Add CLS Token] → 197 tokens
        ↓
[Add Position Embedding]
        ↓
[Transformer Encoder × L layers]
        ↓
[Extract CLS Token output]
        ↓
[MLP Head] → Classification result
```

### Formula Summary

```
# Input
x ∈ R^(H×W×C)  # e.g., 224×224×3

# Patch splitting
P = patch_size  # e.g., 16
N = (H/P) × (W/P)  # number of patches: 196

# Patch Embedding
x_p ∈ R^(N×(P²·C))  # 196×768 (16×16×3 = 768)
z_0 = [x_class; x_p·E] + E_pos  # E: projection matrix

# Transformer
z_l = MSA(LN(z_{l-1})) + z_{l-1}  # Multi-Head Self-Attention
z_l = MLP(LN(z_l)) + z_l         # Feed Forward

# Output
y = LN(z_L^0)  # Final representation of CLS token
```

---

## 3. Patch Embedding

### Concept

```python
# Split image into patches
# (B, 3, 224, 224) → (B, 196, 768)

# Method 1: reshape
patches = image.reshape(B, N, P*P*C)  # Direct reconstruction

# Method 2: Conv2d (more efficient)
# stride=kernel_size for non-overlapping patches
conv = nn.Conv2d(3, 768, kernel_size=16, stride=16)
patches = conv(image)  # (B, 768, 14, 14)
patches = patches.flatten(2).transpose(1, 2)  # (B, 196, 768)
```

### PyTorch Implementation

```python
class PatchEmbedding(nn.Module):
    """Patch Embedding Layer (⭐⭐)"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Extract patches + embed with Conv2d
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        return x
```

---

## 4. CLS Token and Position Embedding

### CLS Token

```python
# Concept borrowed from BERT
# Special token that learns representation of entire image

class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
# Broadcast to batch
cls_tokens = class_token.expand(batch_size, -1, -1)  # (B, 1, D)
# Concatenate before patch embeddings
x = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, N+1, D)
```

### Position Embedding

```python
# Provide patch position information (Transformer has no position info)

class PositionEmbedding(nn.Module):
    """Learnable Position Embedding (⭐⭐)"""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # +1 for CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

    def forward(self, x):
        return x + self.pos_embedding
```

### Position Embedding Visualization

```python
def visualize_position_embedding(pos_embed, img_size=224, patch_size=16):
    """Visualize position embedding similarity (⭐⭐)"""
    # pos_embed: (1, N+1, D)
    # Exclude CLS token
    pos_embed = pos_embed[0, 1:]  # (N, D)

    # Similarity matrix
    similarity = torch.mm(pos_embed, pos_embed.T)  # (N, N)

    # Similarity with specific patch
    num_patches = (img_size // patch_size)
    center_idx = num_patches * (num_patches // 2) + (num_patches // 2)
    center_sim = similarity[center_idx].reshape(num_patches, num_patches)

    return center_sim  # Similarity map with center patch
```

---

## 5. Complete Vision Transformer Implementation

### Basic ViT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention (⭐⭐⭐)"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # QKV computation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP Block (⭐⭐)"""
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block (⭐⭐⭐)"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) (⭐⭐⭐⭐)"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position Embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add Position Embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract and classify CLS token
        cls_output = x[:, 0]
        return self.head(cls_output)
```

### ViT Model Variants

```python
# ViT-Base (ViT-B/16)
vit_base = VisionTransformer(
    img_size=224, patch_size=16,
    embed_dim=768, depth=12, num_heads=12
)

# ViT-Large (ViT-L/16)
vit_large = VisionTransformer(
    img_size=224, patch_size=16,
    embed_dim=1024, depth=24, num_heads=16
)

# ViT-Huge (ViT-H/14)
vit_huge = VisionTransformer(
    img_size=224, patch_size=14,
    embed_dim=1280, depth=32, num_heads=16
)
```

---

## 6. DeiT (Data-efficient Image Transformer)

### Key Improvements

```
Problem: ViT requires large-scale data (JFT-300M etc)
Solution: Knowledge distillation + strong data augmentation for ImageNet-only training

1. Distillation Token: Learn knowledge from CNN teacher
2. Strong Data Augmentation
3. Regularization (Stochastic Depth, Dropout)
```

### Distillation Token

```python
class DeiT(nn.Module):
    """Data-efficient Image Transformer (⭐⭐⭐⭐)"""
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        # CLS Token + Distillation Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position Embedding (+2 for CLS and DIST)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 2, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Two heads
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use both CLS and DIST tokens
        cls_output = self.head(x[:, 0])
        dist_output = self.head_dist(x[:, 1])

        if self.training:
            return cls_output, dist_output
        else:
            # Average during inference
            return (cls_output + dist_output) / 2
```

### DeiT Training

```python
def train_deit_with_distillation(student, teacher, dataloader, epochs=100):
    """DeiT Knowledge Distillation Training (⭐⭐⭐)"""
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dist = nn.CrossEntropyLoss()

    teacher.eval()

    for epoch in range(epochs):
        for images, labels in dataloader:
            # Teacher prediction (soft labels)
            with torch.no_grad():
                teacher_output = teacher(images)

            # Student predictions
            cls_output, dist_output = student(images)

            # Losses
            loss_cls = criterion_ce(cls_output, labels)
            loss_dist = criterion_dist(dist_output, teacher_output.argmax(dim=1))

            loss = 0.5 * loss_cls + 0.5 * loss_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 7. Swin Transformer

### Key Idea

```
Problem: ViT's O(n²) complexity → difficult to process high-resolution images
Solution: Hierarchical structure + Shifted Window Attention

Features:
1. Window Attention: attention only within local windows
2. Shifted Windows: information exchange between windows
3. Hierarchical structure: progressive feature map resolution reduction
```

### Window Attention

```python
def window_partition(x, window_size):
    """Partition image into windows (⭐⭐⭐)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Merge windows back to image (⭐⭐⭐)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention (⭐⭐⭐⭐)"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Create relative position index
        self._create_relative_position_index()

    def _create_relative_position_index(self):
        coords = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords, coords], indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x
```

### Shifted Window

```python
class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with (Shifted) Window Attention (⭐⭐⭐⭐)"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Window reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
```

---

## 8. Using Pretrained Models

### Using torchvision

```python
from torchvision.models import vit_b_16, vit_l_16, swin_t, swin_s

# ViT-B/16 (pretrained)
model = vit_b_16(weights='IMAGENET1K_V1')

# Use as feature extractor
model.heads = nn.Identity()
features = model(image)  # (B, 768)

# Fine-tuning
model = vit_b_16(weights='IMAGENET1K_V1')
model.heads = nn.Linear(768, num_classes)

# Differential learning rates
params = [
    {'params': model.encoder.parameters(), 'lr': 1e-5},  # backbone
    {'params': model.heads.parameters(), 'lr': 1e-3}     # head
]
optimizer = torch.optim.AdamW(params)
```

### Using timm Library

```python
import timm

# List available ViT models
vit_models = timm.list_models('vit*', pretrained=True)
print(f"Available ViT models: {len(vit_models)}")

# Load model
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Custom classification head
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=10  # Automatically replace head
)

# DeiT model
deit_model = timm.create_model('deit_base_patch16_224', pretrained=True)

# Swin Transformer
swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
```

---

## 9. Practical Fine-tuning

### CIFAR-10 Fine-tuning

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def finetune_vit_cifar10(epochs=10):
    """ViT CIFAR-10 Fine-tuning (⭐⭐⭐)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preprocessing (resize to ViT input size)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('data', train=False, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    # Model
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=10)
    model = model.to(device)

    # Optimizer (differential learning rates)
    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    head_params = [p for n, p in model.named_parameters() if 'head' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=0.01)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%')

        scheduler.step()

    return model
```

---

## 10. ViT vs CNN Comparison

### Characteristics Comparison

| Feature | CNN | ViT |
|-----|-----|-----|
| Inductive bias | Locality, equivariance | None |
| Data requirement | Less | More |
| Computational complexity | O(n) | O(n²) |
| Long-range dependencies | Difficult | Easy |
| Interpretability | Filter visualization | Attention visualization |

### Usage Guidelines

```
Prefer CNN:
- Small datasets
- Limited computational resources
- Real-time inference required

Prefer ViT:
- Large datasets or pretrained models available
- Tasks requiring global context
- Planning multimodal learning
```

---

## Summary

### Key Concepts

1. **Patch Embedding**: Convert image to patch sequence
2. **CLS Token**: Learn global image representation
3. **Position Embedding**: Provide patch position information
4. **DeiT**: Data-efficient training (knowledge distillation)
5. **Swin**: Window-based efficient attention

### Model Selection Guide

```
General classification: ViT-B/16 or DeiT
High resolution: Swin Transformer
Limited resources: ViT-Small, DeiT-Tiny
Best performance: ViT-Large, Swin-Large
```

### PyTorch Practical Tips

```python
# 1. Recommend using timm
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 2. Differential learning rates essential
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-3}
])

# 3. Pay attention to input size (224, 384, etc)

# 4. Use strong data augmentation
```

---

## Exercises

### Exercise 1: Patch Embedding Shape Verification

Verify the patch embedding produces the expected tensor shapes.

1. Instantiate `PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)`.
2. Pass a random input `torch.randn(2, 3, 224, 224)` and print the output shape.
3. Confirm the output is `(2, 196, 768)`. Calculate 196 manually: `(224 // 16) ** 2`.
4. Repeat with `patch_size=32` and `patch_size=8`. What is `num_patches` in each case?
5. Explain the trade-off: smaller patches give more tokens (better resolution) but increase the O(n²) attention cost.

### Exercise 2: CLS Token Role Verification

Verify that the CLS token aggregates global information from all patches.

1. Build a small ViT (depth=2, `d_model=64`, `nhead=4`) and pass a random image batch.
2. Extract the CLS token output `x[:, 0]` and each patch token output `x[:, 1:]` after the encoder.
3. Compute the cosine similarity between the CLS token and each patch token.
4. Visualize the similarity map by reshaping patch similarities into a `(H/P, W/P)` grid.
5. Explain why the CLS token serves as the classification representation rather than mean-pooling all patch tokens.

### Exercise 3: ViT Fine-tuning on CIFAR-10

Fine-tune a pretrained ViT on CIFAR-10 using differential learning rates.

1. Load `timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=10)`.
2. Set backbone learning rate to `1e-5` and head learning rate to `1e-3`.
3. Apply the correct ImageNet normalization and resize CIFAR-10 images to 224×224.
4. Train for 10 epochs with `CosineAnnealingLR` and report test accuracy.
5. Compare with training from scratch (same architecture, random initialization) for 10 epochs.

### Exercise 4: Window Attention Complexity Analysis

Quantify the computational advantage of Swin's window attention over full attention.

1. For an input feature map of size H=56, W=56 with `d_model=96`:
   - Compute the number of attention operations for full self-attention: O(N²) where N = H × W.
   - Compute the number for window attention with `window_size=7`: each window has 49 tokens; there are `(56/7)² = 64` windows.
2. Express the ratio as `full_ops / window_ops`.
3. Generalize the formula: for image size H×W and window size W_s, derive the complexity ratio.
4. At what image size does window attention become more than 100× cheaper than full attention?

### Exercise 5: DeiT Knowledge Distillation

Train a DeiT model with knowledge distillation from a CNN teacher.

1. Load a pretrained ResNet-18 as the teacher model.
2. Build a small DeiT-style model (depth=4, `d_model=192`) as the student.
3. Implement the combined loss: `0.5 * CrossEntropy(cls_output, labels) + 0.5 * CrossEntropy(dist_output, teacher_argmax)`.
4. Train on CIFAR-10 for 20 epochs and record validation accuracy.
5. Compare against the same student model trained without the distillation token and hard labels only. Explain the observed accuracy difference.

---

## References

- ViT Original: https://arxiv.org/abs/2010.11929
- DeiT: https://arxiv.org/abs/2012.12877
- Swin Transformer: https://arxiv.org/abs/2103.14030
- timm Library: https://github.com/huggingface/pytorch-image-models

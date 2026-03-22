[Previous: CLIP (Contrastive Language-Image Pre-training)](./35_Impl_CLIP.md) | [Next: Modern Deep Learning Architectures](./37_Modern_Architectures.md)

---

# 36. Self-Supervised Learning

## Learning Objectives

- Understand Self-Supervised Learning concepts and necessity
- Contrastive Learning (SimCLR, MoCo, BYOL)
- Masked Image Modeling (MAE)
- Transfer learning of pre-trained representations
- PyTorch implementation and practice

---

## 1. Self-Supervised Learning Overview

### Definition and Necessity

```
Definition: Generate learning signals from data itself without labels

Why is it needed?
1. Labeling cost: ImageNet (14 million images) labeling cost is enormous
2. Abundant unlabeled data: Most data on the internet
3. Generalization ability: Transfer to various downstream tasks

Paradigm shift:
Supervised learning: Data + Labels → Model
Self-supervised learning: Data → Pretext Task → Representation → Downstream Task
```

### Types of SSL

```
┌─────────────────────────────────────────────────────────────┐
│              Self-Supervised Learning Methods               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Contrastive Learning          Masked Modeling               │
│  ├── SimCLR                   ├── MAE (Vision)               │
│  ├── MoCo                     ├── BERT (NLP)                 │
│  ├── BYOL                     └── BEiT                       │
│  └── SimSiam                                                 │
│                                                              │
│  Clustering                    Generative                    │
│  ├── DeepCluster              ├── VAE                        │
│  └── SwAV                     └── GAN                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Contrastive Learning Basics

### Core Idea

```
Different augmentations of the same image → Close (Positive)
Different images → Far (Negative)

x ──┬── Aug1 → view1 ──┐
    │                   │ → maximize similarity (positive pair)
    └── Aug2 → view2 ──┘

x1 ── Aug → view1 ─┐
                    │ → minimize similarity (negative pair)
x2 ── Aug → view2 ─┘
```

### InfoNCE Loss

```python
import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.5):
    """InfoNCE (NT-Xent) Loss (⭐⭐⭐)

    features: (2N, D) - two augmentations of N images

    Returns:
        loss: contrastive loss
    """
    batch_size = features.shape[0] // 2

    # Normalize
    features = F.normalize(features, dim=1)

    # Similarity matrix
    similarity_matrix = features @ features.T / temperature

    # Mask diagonal and same-image pairs
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)])
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # Mask self-similarity (diagonal)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Positives: same image, different augmentation
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # Negatives: different images
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # Logits
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    return F.cross_entropy(logits, labels.to(logits.device))
```

---

## 3. SimCLR (Simple Framework for Contrastive Learning)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SimCLR Framework                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│     Image x                                                  │
│        │                                                     │
│        ▼                                                     │
│  ┌─────────────┐                                            │
│  │ Data Aug    │ → generate two views (t, t')               │
│  └─────────────┘                                            │
│      │     │                                                 │
│     t(x)  t'(x)                                             │
│      │     │                                                 │
│      ▼     ▼                                                 │
│  ┌─────────────┐                                            │
│  │  Encoder f  │ → feature extraction (ResNet, etc.)        │
│  └─────────────┘                                            │
│      │     │                                                 │
│     h_i   h_j                                                │
│      │     │                                                 │
│      ▼     ▼                                                 │
│  ┌─────────────┐                                            │
│  │Projection g │ → MLP to low-dim embedding                 │
│  └─────────────┘                                            │
│      │     │                                                 │
│     z_i   z_j                                                │
│      │     │                                                 │
│      └──┬──┘                                                │
│         ▼                                                    │
│    NT-Xent Loss                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50

class SimCLR(nn.Module):
    """SimCLR Model (⭐⭐⭐⭐)"""
    def __init__(self, base_encoder=resnet50, projection_dim=128):
        super().__init__()

        # Encoder (pretrained ResNet without FC)
        self.encoder = base_encoder(weights=None)
        self.encoder_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # Projection Head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)  # feature vector
        z = self.projection_head(h)  # projection vector
        return h, z

    def get_features(self, x):
        """Feature extraction for downstream tasks"""
        return self.encoder(x)


class SimCLRAugmentation:
    """SimCLR Data Augmentation (⭐⭐⭐)"""
    def __init__(self, size=224):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size) // 2 * 2 + 1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
```

### SimCLR Training

```python
def train_simclr(model, train_loader, epochs=100, lr=0.3, temperature=0.5):
    """SimCLR Training Loop (⭐⭐⭐)"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0

        for (x_i, x_j), _ in train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            # Forward
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            # Concatenate
            z = torch.cat([z_i, z_j], dim=0)

            # Loss
            loss = info_nce_loss(z, temperature)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss/len(train_loader):.4f}")

    return model
```

---

## 4. MoCo (Momentum Contrast)

### Core Idea

```
SimCLR problem: Requires large batch size (4096+)
MoCo solution: Momentum Encoder + Queue for many negatives

Features:
1. Queue: Store embeddings from previous batches (65536)
2. Momentum Encoder: Slowly updated encoder
```

### MoCo Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MoCo Framework                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Query                          Key                         │
│     │                              │                         │
│     ▼                              ▼                         │
│ ┌─────────┐                  ┌─────────┐                    │
│ │Encoder f│                  │Encoder  │                    │
│ │  (q)    │                  │f_k (mom)│ ← momentum update  │
│ └────┬────┘                  └────┬────┘                    │
│      │                            │                          │
│      ▼                            ▼                          │
│     q_i                         k_i ────────────────┐       │
│      │                            │                 │        │
│      │                            ▼                 ▼        │
│      │                       ┌─────────┐     ┌──────────┐   │
│      │                       │  Queue  │ ←── │ enqueue  │   │
│      │                       │ (k-)    │     └──────────┘   │
│      │                       └────┬────┘                    │
│      │                            │                          │
│      └────────────┬───────────────┘                         │
│                   ▼                                          │
│            InfoNCE Loss                                      │
│        (q·k+ / q·k-)                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch Implementation

```python
import copy

class MoCo(nn.Module):
    """Momentum Contrast (MoCo v2) (⭐⭐⭐⭐)"""
    def __init__(self, base_encoder=resnet50, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K  # Queue size
        self.m = m  # Momentum
        self.T = T  # Temperature

        # Query encoder
        self.encoder_q = base_encoder(weights=None)
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(self.encoder_q.fc.in_features, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

        # Key encoder (momentum)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # no backprop

        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace oldest
        if ptr + batch_size > self.K:
            # Wrap around
            self.queue[:, ptr:] = keys[:self.K - ptr].T
            self.queue[:, :batch_size - (self.K - ptr)] = keys[self.K - ptr:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Args:
            im_q: query image
            im_k: key image (different augmentation)
        """
        # Query
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # Key (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


def train_moco(model, train_loader, epochs=200):
    """MoCo Training (⭐⭐⭐)"""
    optimizer = torch.optim.SGD(model.encoder_q.parameters(),
                                lr=0.03, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for (im_q, im_k), _ in train_loader:
            logits, labels = model(im_q.cuda(), im_k.cuda())
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 5. BYOL (Bootstrap Your Own Latent)

### Core Idea

```
Question: Are negative samples really necessary?
BYOL: Learn without negatives! (Online + Target network)

Key: Predictor network + EMA to prevent collapse
```

### BYOL Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BYOL Framework                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Online Network                 Target Network              │
│   (trained)                      (updated with EMA)          │
│                                                              │
│  ┌─────────┐                    ┌─────────┐                 │
│  │Encoder  │                    │Encoder  │                 │
│  │(θ)      │                    │(ξ)      │ ← EMA           │
│  └────┬────┘                    └────┬────┘                 │
│       │                              │                       │
│       ▼                              ▼                       │
│  ┌─────────┐                    ┌─────────┐                 │
│  │Projector│                    │Projector│                 │
│  └────┬────┘                    └────┬────┘                 │
│       │                              │                       │
│       ▼                              │                       │
│  ┌─────────┐                         │                       │
│  │Predictor│ (Online only)            │                       │
│  └────┬────┘                         │                       │
│       │                              │                       │
│      q_θ                           z_ξ                       │
│       │                              │                       │
│       └──────────┬───────────────────┘                      │
│                  ▼                                           │
│         MSE Loss (q_θ, sg(z_ξ))                             │
│         sg = stop gradient                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch Implementation

```python
class BYOL(nn.Module):
    """Bootstrap Your Own Latent (⭐⭐⭐⭐)"""
    def __init__(self, base_encoder=resnet50, hidden_dim=4096, proj_dim=256, pred_dim=256):
        super().__init__()

        # Online network
        encoder = base_encoder(weights=None)
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.online_encoder = encoder
        self.online_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, proj_dim)
        )

        # Target network (EMA)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau=0.99):
        """Target network EMA update"""
        for online, target in zip(self.online_encoder.parameters(),
                                   self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

        for online, target in zip(self.online_projector.parameters(),
                                   self.target_projector.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    def forward(self, x1, x2):
        # Online predictions
        online_proj_1 = self.online_projector(self.online_encoder(x1))
        online_proj_2 = self.online_projector(self.online_encoder(x2))

        online_pred_1 = self.predictor(online_proj_1)
        online_pred_2 = self.predictor(online_proj_2)

        # Target projections (no gradient)
        with torch.no_grad():
            target_proj_1 = self.target_projector(self.target_encoder(x1))
            target_proj_2 = self.target_projector(self.target_encoder(x2))

        return online_pred_1, online_pred_2, target_proj_1, target_proj_2


def byol_loss(pred, target):
    """BYOL Loss: Negative Cosine Similarity (⭐⭐⭐)"""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 2 - 2 * (pred * target).sum(dim=-1).mean()


def train_byol(model, train_loader, epochs=100):
    """BYOL Training (⭐⭐⭐)"""
    optimizer = torch.optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.predictor.parameters()),
        lr=3e-4
    )

    for epoch in range(epochs):
        for (x1, x2), _ in train_loader:
            x1, x2 = x1.cuda(), x2.cuda()

            pred1, pred2, target1, target2 = model(x1, x2)

            # Symmetric loss
            loss = byol_loss(pred1, target2) + byol_loss(pred2, target1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target
            model.update_target(tau=0.99)
```

---

## 6. MAE (Masked Autoencoder)

### Core Idea

```
Apply BERT's idea from NLP to Vision:
- Mask portion of image (75%)
- Train to reconstruct masked portions

Advantages:
1. High masking ratio → strong learning signal
2. Efficient: encode only visible patches
3. Naturally combines with ViT
```

### MAE Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MAE Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Image → Patches                                            │
│         ↓                                                    │
│   ┌─────────────────────────────────────────────┐           │
│   │ [P1] [P2] [P3] ... [P196]                   │           │
│   └─────────────────────────────────────────────┘           │
│         ↓ Random Masking (75%)                               │
│   ┌─────────────────────────────────────────────┐           │
│   │ [P1] [M] [P3] [M] [M] ... [P196]            │           │
│   └─────────────────────────────────────────────┘           │
│         ↓ (visible patches only)                             │
│   ┌─────────────────────────────────────────────┐           │
│   │ Encoder (ViT) - only on visible patches     │           │
│   │ [P1] [P3] ... [P196]                        │           │
│   └─────────────────────────────────────────────┘           │
│         ↓ + Mask tokens + Position Embedding                 │
│   ┌─────────────────────────────────────────────┐           │
│   │ Decoder (small ViT)                         │           │
│   │ [P1] [M] [P3] [M] [M] ... [P196]            │           │
│   └─────────────────────────────────────────────┘           │
│         ↓                                                    │
│   Reconstruct masked patches (MSE Loss)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch Implementation

```python
import random

class MAE(nn.Module):
    """Masked Autoencoder (⭐⭐⭐⭐)"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_heads=16,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        num_patches = (img_size // patch_size) ** 2

        # Encoder
        self.patch_embed = nn.Conv2d(in_channels, encoder_embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, encoder_embed_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(encoder_embed_dim, encoder_heads, batch_first=True),
            num_layers=encoder_depth
        )
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, decoder_embed_dim))

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(decoder_embed_dim, decoder_heads, batch_first=True),
            num_layers=decoder_depth
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels)

    def random_masking(self, x, mask_ratio):
        """Random masking (⭐⭐⭐)"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add position embedding (without cls)
        x = x + self.pos_embed[:, 1:, :]

        # Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)

        # Encoder
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # Decoder embed
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add position embedding
        x = x + self.decoder_pos_embed

        # Decoder
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # Predictor
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token

        return x, mask

    def loss(self, pred, target, mask):
        """MAE Loss: MSE on masked patches only (⭐⭐⭐)"""
        target = self.patchify(target)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # mean per patch
        loss = (loss * mask).sum() / mask.sum()  # mean on masked
        return loss

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x
```

### MAE Training

```python
def train_mae(model, train_loader, epochs=400, lr=1.5e-4):
    """MAE Training (⭐⭐⭐)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0

        for images, _ in train_loader:
            images = images.cuda()

            pred, mask = model(images)
            loss = model.loss(pred, images, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss/len(train_loader):.4f}")
```

---

## 7. Linear Evaluation

### Evaluating Representation Quality

```python
def linear_evaluation(encoder, train_loader, test_loader, num_classes=10):
    """SSL representation Linear Evaluation (⭐⭐⭐)"""
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Linear classifier
    feature_dim = 2048  # ResNet-50 based
    classifier = nn.Linear(feature_dim, num_classes).cuda()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    # Extract features
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                feat = encoder(images.cuda())
                features.append(feat.cpu())
                labels.append(targets)
        return torch.cat(features), torch.cat(labels)

    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)

    # Train linear classifier
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    for epoch in range(100):
        for features, labels in train_loader:
            features, labels = features.cuda(), labels.cuda()
            output = classifier(features)
            loss = F.cross_entropy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        output = classifier(test_features.cuda())
        pred = output.argmax(dim=1).cpu()
        accuracy = (pred == test_labels).float().mean().item()

    return accuracy * 100
```

---

## 8. Method Comparison

### Characteristics Comparison

| Method | Negatives | Batch Size | Key Features |
|--------|-----------|------------|--------------|
| SimCLR | Required | 4096+ | Simple, strong augmentation |
| MoCo | Required (Queue) | 256 | Memory efficient |
| BYOL | Not required | 256 | Predictor + EMA |
| SimSiam | Not required | 256 | BYOL simplified (no EMA) |
| MAE | Not required | 256 | Reconstruction |

### Performance Comparison (ImageNet Linear Probe)

```
SimCLR (ResNet-50, 8192 batch): 69.3%
MoCo v2 (ResNet-50): 71.1%
BYOL (ResNet-50): 74.3%
MAE (ViT-Base): 67.8% → Fine-tune: 83.6%
```

---

## Summary

### Key Concepts

1. **Contrastive Learning**: Learn with positive/negative pairs
2. **InfoNCE Loss**: Contrastive loss function
3. **Momentum Encoder**: Slowly updated target
4. **Masked Modeling**: Train to reconstruct partial input
5. **Linear Evaluation**: Linear classifier performance on frozen representations

### Selection Guide

```
Large batch possible: SimCLR (simple and effective)
Limited resources: MoCo (efficient with queue)
Without negatives: BYOL, SimSiam
ViT-based: MAE (reconstruction-based)
```

### Practical Tips

```python
# 1. Data Augmentation is key
# - Strong augmentation = better representation

# 2. Be careful with temperature parameter
# - Too low: unstable training
# - Too high: weak learning signal

# 3. Long training required
# - Minimum 200-800 epochs recommended

# 4. Linear eval vs Fine-tune
# - Linear: evaluate representation quality
# - Fine-tune: actual performance (higher)
```

---

## Exercises

### Exercise 1: Understand Contrastive Learning Augmentations

Strong data augmentation is the most critical factor in contrastive learning quality.
1. Using `SimCLRAugmentation`, generate 8 augmented views of the same CIFAR-10 image and visualize them.
2. Identify which augmentations (crop, flip, color jitter, grayscale, blur) are most visually drastic.
3. Explain why color jitter is particularly important for contrastive learning: what visual shortcut would the model exploit if color jitter were removed?
4. What is the purpose of using `GaussianBlur` as an augmentation? What invariance does it encourage the encoder to learn?

### Exercise 2: Implement and Run SimCLR Training

Using the provided `SimCLR` and `SimCLRAugmentation` code:
1. Wrap a CIFAR-10 dataset so that `__getitem__` returns `(augmented_view_1, augmented_view_2)` using `SimCLRAugmentation`.
2. Train a `SimCLR` model with a `ResNet-18` backbone for 20 epochs with `batch_size=256` and `temperature=0.5`.
3. After training, freeze the encoder and train a linear classifier on top (linear evaluation) for 30 epochs.
4. Report the linear probe accuracy on the CIFAR-10 test set. How does it compare to fully supervised training with the same architecture?

### Exercise 3: Analyze MoCo's Momentum Encoder

The momentum encoder is the key innovation in MoCo that allows using a smaller batch size than SimCLR.
1. In `MoCo._momentum_update_key_encoder`, the update rule is: `param_k = m * param_k + (1-m) * param_q`. Explain in your own words why the key encoder parameters are NOT updated through backpropagation.
2. Experiment: initialize two copies of a small network. Update one with SGD and the other with momentum update (m=0.999) for 100 steps. Plot the norm of parameter differences over time.
3. What would happen if `m=0.0` (no momentum)? What if `m=1.0`? Why is a value close to 1 (e.g., 0.999) used in practice?

### Exercise 4: Implement MAE Masking Visualization

Using the `MAE.random_masking` function:
1. Take a 224×224 image and divide it into 14×14 = 196 patches of size 16×16.
2. Apply random masking with ratios `mask_ratio=0.25, 0.50, 0.75, 0.90` and visualize the resulting visible patches (gray out masked patches).
3. The original MAE paper uses 75% masking. Explain why such a high masking ratio is beneficial: how does it prevent the model from using simple local texture patterns to reconstruct patches?
4. Compare MAE's reconstruction-based approach to SimCLR's contrastive approach. In what scenarios would MAE-style pre-training be preferred? When would contrastive learning be preferred?

### Exercise 5: Evaluate and Compare SSL Methods

Using pre-trained model weights (e.g., from torchvision or timm):
1. Load a SimCLR-pre-trained ResNet-50 and a supervised ResNet-50, both frozen.
2. Train a linear classifier on top of each for 100 epochs using the STL-10 dataset (which has few labeled samples).
3. Compare the linear probe accuracy of both approaches.
4. Now fine-tune both models end-to-end with 10% labeled data. Which approach benefits more from fine-tuning? Why does self-supervised pre-training shine particularly in low-label regimes?

---

## References

- SimCLR: https://arxiv.org/abs/2002.05709
- MoCo: https://arxiv.org/abs/1911.05722
- BYOL: https://arxiv.org/abs/2006.07733
- MAE: https://arxiv.org/abs/2111.06377
- SimSiam: https://arxiv.org/abs/2011.10566

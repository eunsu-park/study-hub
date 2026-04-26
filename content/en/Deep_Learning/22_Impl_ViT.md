# 22. Vision Transformer (ViT)

[Previous: Vision Transformer](./21_Vision_Transformer.md) | [Next: Training Optimization](./23_Training_Optimization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the core idea of ViT — treating an image as a sequence of flattened patches — and describe how this reformulates image classification as a sequence modeling problem.
2. Calculate the number of patches produced from an input image given the patch size, and determine the embedding dimension for each patch token.
3. Describe the role of the [CLS] token and 1D learnable positional embeddings in the ViT architecture.
4. Implement the full ViT pipeline from scratch in PyTorch: patch embedding, Transformer encoder blocks, and classification head.
5. Fine-tune a pretrained ViT model on a downstream image classification dataset and compare its performance against CNN baselines.
6. Identify the data requirements and computational trade-offs of ViT compared to CNNs, and explain why ViT benefits significantly from large-scale pretraining.

---

## Theory & Principles

Implementing a ViT from scratch crystallizes the conceptual content of the previous lesson into actual tensor shapes. This section emphasizes three implementation realities: how patch embedding lines up with `Conv2d`, how to handle CLS-token concatenation and positional embedding addition without shape errors, and the standard hyperparameter choices (ViT-Base vs Large vs Huge).

This section covers:

- **A.** The patch-embedding-via-Conv2d trick
- **B.** CLS prepending and positional embedding broadcasting
- **C.** ViT model sizes and width/depth/head counts
- **D.** Hybrid models and the path to Swin and beyond

### A. Patch Embedding via Conv2d

A `Conv2d(in_channels=3, out_channels=d_model, kernel_size=P, stride=P)` produces an output of shape `(B, d_model, H/P, W/P)`. Each spatial location of this output corresponds to one patch of the input image, and its `d_model` channels are the patch embedding. Reshape to a sequence:

```
x = patch_conv(image)                              # (B, d, H/P, W/P)
x = x.flatten(2).transpose(1, 2)                   # (B, N, d) where N = (H/P)*(W/P)
```

This is mathematically identical to "extract `P x P` patches, flatten each to a `3 P^2` vector, multiply by a learned `3 P^2 x d` matrix" — but is implemented in one fused convolution kernel. The reason: convolution is highly optimized; the equivalent `unfold + reshape + matmul` would be slower and more verbose.

### B. CLS Prepending and Positional Embedding

After patch embedding:

```
B, N, d = x.shape
cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d)
x = torch.cat([cls, x], dim=1)                      # (B, N+1, d)
x = x + self.pos_embed                              # broadcast: pos_embed is (1, N+1, d)
```

Two shape gotchas:

1. **CLS token is a learnable parameter of shape `(1, 1, d)`** registered with `nn.Parameter`, and `.expand` (not `.repeat`) is used so it does not allocate batch-sized memory.
2. **Positional embedding is `(1, N+1, d)`**, broadcasting over the batch. Forgetting the `+1` for CLS is the most common bug.

After this, `x` has shape `(B, N+1, d)` and is fed straight into a standard Transformer encoder.

### C. ViT Sizes

Dosovitskiy et al. defined a family of ViTs scaling depth, width, and head count proportionally:

| Model    | Layers | Hidden d | MLP d | Heads | Params |
|----------|--------|----------|-------|-------|--------|
| ViT-Base  | 12     | 768      | 3072  | 12    | 86M    |
| ViT-Large | 24     | 1024     | 4096  | 16    | 307M   |
| ViT-Huge  | 32     | 1280     | 5120  | 16    | 632M   |

Patch size `P = 16` is the most common; smaller patches (`P = 14` or `P = 8`) give more tokens for higher accuracy at proportionally more compute (cost is `O(N^2)` in attention, so `P/2` patches make it 16x more expensive).

The "Base/Large/Huge" naming and the rough 4x MLP expansion are direct inheritances from BERT — ViT is, architecturally, BERT applied to image patches.

### D. Hybrid Models and Swin

Pure ViT has limitations: quadratic attention cost makes high-resolution inputs expensive, and the lack of locality bias hurts on small datasets. Two important successors:

- **Hybrid ViT**: use a CNN (e.g., ResNet-50's first stages) to produce feature maps, *then* apply ViT to those maps. The CNN provides locality bias for the bottom layers; the Transformer handles long-range mixing on top.
- **Swin Transformer (Liu et al. 2021)**: compute attention only within local windows, with a clever shifted-window scheme to allow cross-window information flow. Reduces complexity from `O(N^2)` to `O(N * window_size)`. The current default for vision Transformers in dense prediction tasks (segmentation, detection).

The lesson — start with pure ViT to internalize the architecture, then move to hybrid or windowed variants for production.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| Patch embed via conv | `self.patch_embed = nn.Conv2d(3, d, P, stride=P)` |
| CLS as `nn.Parameter` | `self.cls_token = nn.Parameter(torch.zeros(1, 1, d))` |
| Positional embed | `self.pos_embed = nn.Parameter(torch.zeros(1, N+1, d))` |
| Transformer stack | `nn.TransformerEncoder(layer, num_layers)` |
| Classification head | `nn.Linear(d, num_classes)` on `x[:, 0]` (CLS) |

---


## Overview

Vision Transformer (ViT) applies the Transformer architecture to image classification. It divides images into patches and treats each patch like a token. "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

---

## Mathematical Background

### 1. Image Patchification

```
Input image: x ∈ R^(H × W × C)
Patch size: P × P

Patch sequence:
x_p ∈ R^(N × P² × C)  where N = (H × W) / P²

Example:
- Image: 224 × 224 × 3
- Patch: 16 × 16
- N = (224 × 224) / (16 × 16) = 196 patches
- Each patch: 16 × 16 × 3 = 768 dimensions
```

### 2. Patch Embedding

```
Linear Projection:
z_0 = [x_class; x_p¹E; x_p²E; ...; x_pⁿE] + E_pos

Where:
- x_class: learnable [CLS] token
- E ∈ R^(P²C × D): patch embedding matrix
- E_pos ∈ R^((N+1) × D): position embedding

z_0 ∈ R^((N+1) × D): initial embedding sequence
```

### 3. Transformer Encoder

```
Encoder block (L layers):

z'_l = MSA(LN(z_{l-1})) + z_{l-1}
z_l = MLP(LN(z'_l)) + z'_l

Final output:
y = LN(z_L⁰)  # use only [CLS] token

Where z_L⁰ is the [CLS] token at layer L
```

---

## ViT Architecture Variants

```
ViT-Base (B/16):
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- MLP size: 3072
- Patch size: 16
- Parameters: 86M

ViT-Large (L/16):
- Hidden size: 1024
- Layers: 24
- Attention heads: 16
- MLP size: 4096
- Patch size: 16
- Parameters: 307M

ViT-Huge (H/14):
- Hidden size: 1280
- Layers: 32
- Attention heads: 16
- MLP size: 5120
- Patch size: 14
- Parameters: 632M
```

---

## File Structure

```
10_ViT/
├── README.md
├── pytorch_lowlevel/
│   └── vit_lowlevel.py         # Direct ViT implementation
├── paper/
│   └── vit_paper.py            # Paper reproduction
└── exercises/
    ├── 01_patch_embedding.md   # Patch embedding visualization
    └── 02_attention_maps.md    # Attention visualization
```

---

## Core Concepts

### 1. CNN vs ViT

```
CNN:
- Local receptive field
- Inductive bias: locality, translation equivariance
- Favorable for small datasets

ViT:
- Global receptive field (global from start)
- Minimal inductive bias
- Favorable for large-scale datasets (JFT-300M)
- Small data: needs pre-training
```

### 2. Position Embedding

```
1D Learnable (ViT default):
- N+1 learnable vectors
- Learn order information

2D Positional (variant):
- Separate embedding for (row, col)
- Reflects image structure

Sinusoidal:
- Fixed trigonometric functions
- Extrapolation capability
```

### 3. [CLS] Token vs Global Average Pooling

```
[CLS] Token:
- Added at first position
- Aggregates entire image representation
- BERT style

Global Average Pooling:
- Average all patches
- CNN style
- Similar performance
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Use F.linear, F.layer_norm
- Don't use nn.TransformerEncoder
- Direct patchification implementation

### Level 3: Paper Implementation (paper/)
- Exact paper specifications
- JFT/ImageNet pre-training
- Fine-tuning code

### Level 4: Code Analysis (separate)
- Analyze timm library
- Analyze HuggingFace ViT

---

## Learning Checklist

- [ ] Understand patch embedding formula
- [ ] Role of position embedding
- [ ] Role of [CLS] token
- [ ] Pros/cons compared to CNN
- [ ] Visualize attention maps
- [ ] Fine-tuning strategy

---

## References

- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- [timm ViT](https://github.com/rwightman/pytorch-image-models)
- [21_Vision_Transformer.md](./21_Vision_Transformer.md)

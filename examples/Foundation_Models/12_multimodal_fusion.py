#!/usr/bin/env python3
"""
Foundation Models - Multimodal Fusion (Vision-Language)
=======================================================

Implements core multimodal techniques from scratch:
1. Image patch embeddings (ViT-style)
2. Cross-attention fusion (Flamingo-style)
3. Contrastive learning (CLIP-style)

Key Idea:
    Multimodal models must align representations from different modalities
    (text, images, audio) into a shared embedding space. The two dominant
    approaches are:
    - Contrastive: Learn to match image-text pairs (CLIP, SigLIP)
    - Generative: Fuse visual tokens into language model (LLaVA, Flamingo)

Requires: PyTorch, numpy, matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class PatchEmbedding(nn.Module):
    """
    Convert an image into a sequence of patch embeddings (ViT-style).

    An image of size (C, H, W) is split into (H/P * W/P) non-overlapping
    patches of size (C, P, P), then each patch is linearly projected.

    Why patches instead of CNN features?
        - Patches are simpler and scale better with resolution
        - No inductive bias (the model learns spatial relations from data)
        - Compatible with Transformer sequence processing
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Why: A convolution with kernel_size=stride=patch_size is equivalent
        # to splitting into patches and applying a linear projection, but
        # is faster because it avoids explicit reshaping.
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Why: The [CLS] token aggregates information from all patches
        # via self-attention, giving a single vector for the whole image.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Why: Learned position embeddings (not RoPE) are standard in ViT.
        # +1 for the CLS token.
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

    def forward(self, images):
        """
        Args:
            images: (batch, channels, height, width)

        Returns:
            (batch, num_patches + 1, embed_dim)
        """
        batch_size = images.size(0)

        # Project patches: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.projection(images)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add position embeddings
        x = x + self.position_embedding

        return x


class CrossAttention(nn.Module):
    """
    Cross-attention layer for fusing visual features into text.

    Used in Flamingo, LLaVA-style architectures. Text tokens attend to
    image patch embeddings, learning which visual regions are relevant
    for each text token.

    Why cross-attention over concatenation?
        - Concatenation is O(n_text + n_image) but doesn't model
          fine-grained alignment between specific words and image regions
        - Cross-attention explicitly learns word-to-region alignment
        - More parameter-efficient for long image sequences
    """

    def __init__(self, text_dim: int, image_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        assert text_dim % num_heads == 0

        # Why: Q comes from text (the "reader"), K and V come from image
        # (the "content being read"). This is what makes it "cross" attention.
        self.q_proj = nn.Linear(text_dim, text_dim)
        self.k_proj = nn.Linear(image_dim, text_dim)
        self.v_proj = nn.Linear(image_dim, text_dim)
        self.out_proj = nn.Linear(text_dim, text_dim)

        self.layer_norm_text = nn.LayerNorm(text_dim)
        self.layer_norm_image = nn.LayerNorm(image_dim)

    def forward(self, text_features, image_features):
        """
        Args:
            text_features: (batch, text_len, text_dim) — queries
            image_features: (batch, num_patches, image_dim) — keys/values

        Returns:
            (batch, text_len, text_dim) — text features enriched with visual info
        """
        batch, text_len, _ = text_features.shape

        # Layer norm before attention (Pre-LN Transformer style)
        text_normed = self.layer_norm_text(text_features)
        image_normed = self.layer_norm_image(image_features)

        Q = self.q_proj(text_normed)
        K = self.k_proj(image_normed)
        V = self.v_proj(image_normed)

        # Reshape for multi-head attention
        Q = Q.view(batch, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)

        # Recombine heads
        attended = attended.transpose(1, 2).contiguous().view(batch, text_len, -1)
        output = self.out_proj(attended)

        # Why: Residual connection so the layer can be "skipped" if
        # visual information is not useful for this particular token.
        return text_features + output, weights


class CLIPModel(nn.Module):
    """
    Simplified CLIP (Contrastive Language-Image Pre-training).

    Trains image and text encoders to produce similar embeddings for
    matching pairs and dissimilar embeddings for non-matching pairs.

    Why contrastive learning?
        - No need for labeled data — just image-text pairs (abundant online)
        - Learns a shared embedding space for zero-shot classification
        - The trained encoders transfer well to downstream tasks
    """

    def __init__(self, image_embed_dim: int = 128, text_embed_dim: int = 128,
                 projection_dim: int = 64):
        super().__init__()

        # Why: Separate projection heads map different-dimensional features
        # into a shared space. The projection head is typically small to
        # encourage the encoder to do the heavy lifting.
        self.image_projection = nn.Sequential(
            nn.Linear(image_embed_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Why: Learned temperature controls the sharpness of the softmax.
        # Too low: all pairs look similar. Too high: only exact matches work.
        self.temperature = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, image_features, text_features):
        """
        Compute CLIP contrastive loss.

        Args:
            image_features: (batch, image_embed_dim) — pooled image embeddings
            text_features: (batch, text_embed_dim) — pooled text embeddings

        Returns:
            loss: Scalar contrastive loss
            similarity: (batch, batch) similarity matrix
        """
        # Project to shared space and normalize
        # Why: L2 normalization ensures dot product = cosine similarity.
        # This makes the scale of embeddings irrelevant, focusing on direction.
        image_embeds = F.normalize(self.image_projection(image_features), dim=-1)
        text_embeds = F.normalize(self.text_projection(text_features), dim=-1)

        # Compute similarity matrix
        temperature = self.temperature.exp()
        similarity = (image_embeds @ text_embeds.T) * temperature

        # Why: The diagonal of the similarity matrix contains matching pairs.
        # The loss pushes diagonal values up and off-diagonal values down.
        # This is symmetric: image→text and text→image cross-entropy.
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device)

        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.T, labels)

        loss = (loss_i2t + loss_t2i) / 2
        return loss, similarity.detach()


def visualize_multimodal(clip_model, cross_attn, patch_embed):
    """Visualize multimodal representations and attention patterns."""
    batch_size = 8
    image_size = 64
    patch_size = 16
    embed_dim = 128
    text_len = 12

    # Generate synthetic data
    images = torch.randn(batch_size, 3, image_size, image_size)
    text_features = torch.randn(batch_size, text_len, embed_dim)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Patch embedding visualization
    patches = patch_embed(images)
    # Show how a single image is split into patches
    img = images[0].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[0, 0].imshow(img)
    num_patches_side = image_size // patch_size
    for i in range(1, num_patches_side):
        axes[0, 0].axhline(y=i * patch_size, color='red', linewidth=1)
        axes[0, 0].axvline(x=i * patch_size, color='red', linewidth=1)
    axes[0, 0].set_title(f"Image Patches ({num_patches_side}x{num_patches_side} = "
                         f"{num_patches_side ** 2} patches)")

    # 2. Cross-attention weights
    cross_attn.eval()
    with torch.no_grad():
        _, attn_weights = cross_attn(text_features, patches)
    # Average over heads, show first example
    avg_weights = attn_weights[0].mean(0).numpy()  # (text_len, num_patches+1)
    im2 = axes[0, 1].imshow(avg_weights, cmap='viridis', aspect='auto')
    axes[0, 1].set_xlabel("Image patch index")
    axes[0, 1].set_ylabel("Text token index")
    axes[0, 1].set_title("Cross-Attention: Text -> Image")
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. CLIP similarity matrix
    clip_model.eval()
    with torch.no_grad():
        img_pooled = patches[:, 0, :]  # CLS token
        txt_pooled = text_features.mean(dim=1)  # Mean pool text
        _, sim_matrix = clip_model(img_pooled, txt_pooled)
    im3 = axes[1, 0].imshow(sim_matrix.numpy(), cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_xlabel("Text sample")
    axes[1, 0].set_ylabel("Image sample")
    axes[1, 0].set_title("CLIP Similarity Matrix\n(diagonal = matching pairs)")
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. Embedding space visualization (2D PCA)
    img_embeds = F.normalize(
        clip_model.image_projection(img_pooled), dim=-1
    ).detach().numpy()
    txt_embeds = F.normalize(
        clip_model.text_projection(txt_pooled), dim=-1
    ).detach().numpy()

    # Simple 2D projection via PCA
    all_embeds = np.vstack([img_embeds, txt_embeds])
    centered = all_embeds - all_embeds.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T

    n = batch_size
    axes[1, 1].scatter(proj[:n, 0], proj[:n, 1], c='blue', marker='o',
                       s=100, label='Image', alpha=0.7)
    axes[1, 1].scatter(proj[n:, 0], proj[n:, 1], c='red', marker='^',
                       s=100, label='Text', alpha=0.7)
    # Draw lines between matching pairs
    for i in range(n):
        axes[1, 1].plot([proj[i, 0], proj[n + i, 0]],
                        [proj[i, 1], proj[n + i, 1]],
                        'k--', alpha=0.3, linewidth=0.8)
    axes[1, 1].set_title("Embedding Space (PCA)\n(lines connect matching pairs)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Multimodal Vision-Language Fusion", fontsize=14)
    plt.tight_layout()
    plt.savefig("multimodal_fusion.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: multimodal_fusion.png")


if __name__ == "__main__":
    torch.manual_seed(42)

    # Configuration
    image_size = 64
    patch_size = 16
    embed_dim = 128
    text_len = 12
    batch_size = 16
    projection_dim = 64

    print("=" * 60)
    print("Multimodal Vision-Language Fusion Demonstration")
    print("=" * 60)

    # 1. Patch Embedding
    print("\n--- 1. Image Patch Embeddings ---")
    patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
    images = torch.randn(batch_size, 3, image_size, image_size)
    patches = patch_embed(images)
    num_patches = (image_size // patch_size) ** 2
    print(f"Image: {image_size}x{image_size} -> {num_patches} patches of {patch_size}x{patch_size}")
    print(f"Patch embeddings shape: {patches.shape}  (includes CLS token)")

    # 2. Cross-Attention
    print("\n--- 2. Cross-Attention Fusion ---")
    cross_attn = CrossAttention(embed_dim, embed_dim, num_heads=4)
    text_features = torch.randn(batch_size, text_len, embed_dim)
    fused_text, attn_weights = cross_attn(text_features, patches)
    print(f"Text input: {text_features.shape}")
    print(f"Fused output: {fused_text.shape}")
    print(f"Attention weights: {attn_weights.shape}  (batch, heads, text_len, num_patches+1)")

    # 3. CLIP Contrastive Learning
    print("\n--- 3. CLIP Contrastive Training ---")
    clip = CLIPModel(embed_dim, embed_dim, projection_dim)
    optimizer = torch.optim.Adam(clip.parameters(), lr=3e-4)

    clip.train()
    for step in range(100):
        # Simulate batch where index i is a matching pair
        img_features = torch.randn(batch_size, embed_dim)
        txt_features = torch.randn(batch_size, embed_dim)
        # Why: Add shared signal to matching pairs so the model can learn
        shared_signal = torch.randn(batch_size, embed_dim) * 0.5
        img_features = img_features + shared_signal
        txt_features = txt_features + shared_signal

        loss, sim = clip(img_features, txt_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            # Check accuracy: does the diagonal have the highest values?
            preds = sim.argmax(dim=-1)
            labels = torch.arange(batch_size)
            acc = (preds == labels).float().mean()
            print(f"  Step {step + 1:3d}: loss={loss.item():.4f}, "
                  f"acc={acc.item():.2%}")

    # 4. Visualize
    print("\nGenerating visualization...")
    visualize_multimodal(clip, cross_attn, patch_embed)

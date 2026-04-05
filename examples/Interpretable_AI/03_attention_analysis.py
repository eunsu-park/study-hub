"""
03. Transformer Attention Analysis and Rollout

Extracts and visualizes attention patterns from a pretrained BERT model
(HuggingFace Transformers). Compares raw attention weights from individual
heads with the Attention Rollout algorithm, which accounts for residual
connections to produce a more faithful representation of information flow.

Covered topics:
    - Extracting multi-head attention weights from BERT
    - Raw attention heatmaps per head and per layer
    - Attention Rollout: product across layers with residual mixing
    - Head-level analysis: identifying specialized vs. redundant heads
    - Comparison of raw attention vs. rollout for sentiment analysis

Related to: L04 - Attention Mechanisms and Interpretability

Requirements:
    pip install torch transformers matplotlib numpy
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ====== Model and Tokenizer Setup ======

def load_bert_model(model_name: str = "bert-base-uncased") -> tuple:
    """Load a pretrained BERT model configured to output attention weights.

    We set output_attentions=True so that every forward pass returns a
    tuple of attention tensors -- one per layer -- without needing hooks.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        (model, tokenizer) tuple, both on CPU.
    """
    from transformers import BertTokenizer, BertModel

    print(f"  Loading {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    print(f"  Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads per layer.")
    return model, tokenizer


def tokenize_input(text: str, tokenizer) -> tuple:
    """Tokenize text and return input IDs plus human-readable tokens.

    BERT adds [CLS] at the start and [SEP] at the end. We return the
    token strings so that attention heatmaps have meaningful axis labels.

    Returns:
        (input_ids tensor of shape (1, seq_len), list of token strings)
    """
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return inputs, tokens


# ====== Attention Extraction ======

def extract_attention_weights(
    model,
    inputs: dict,
) -> torch.Tensor:
    """Run a forward pass and collect attention weights from all layers.

    Returns:
        Tensor of shape (num_layers, num_heads, seq_len, seq_len).
        Each entry [l, h, i, j] is the attention weight that token i
        places on token j in layer l, head h.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of (num_layers) tensors,
    # each of shape (batch, num_heads, seq_len, seq_len)
    attention_layers = outputs.attentions

    # Stack into a single tensor and remove the batch dimension
    attention = torch.stack(attention_layers).squeeze(1)
    # Shape: (num_layers, num_heads, seq_len, seq_len)
    return attention


# ====== Attention Rollout ======

def attention_rollout(
    attention: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Compute Attention Rollout across all transformer layers.

    Raw attention from a single layer does not account for the residual
    connections that allow information to bypass the attention mechanism.
    Attention Rollout (Abnar & Zuidema, 2020) addresses this by:

    1. For each layer, fuse multi-head attention into a single matrix.
    2. Add the identity matrix (simulating the residual connection).
    3. Re-normalize each row to sum to 1.
    4. Multiply the resulting matrices across layers.

    The final matrix R gives R[i,j] = "how much of token j's information
    ends up in token i's representation after all layers."

    Args:
        attention: Shape (num_layers, num_heads, seq_len, seq_len).
        head_fusion: How to combine heads. 'mean' averages all heads,
                     'max' takes the element-wise maximum, 'min' takes
                     the minimum.
        discard_ratio: Fraction of lowest-attention entries to zero out
                       before re-normalization. Helps remove noise from
                       near-uniform attention patterns.

    Returns:
        Rollout matrix of shape (seq_len, seq_len) as a numpy array.
    """
    num_layers, num_heads, seq_len, _ = attention.shape

    # Start with identity: before any layer, each token is 100% itself
    rollout = torch.eye(seq_len)

    for layer_idx in range(num_layers):
        # Fuse heads into a single attention matrix per layer
        if head_fusion == "mean":
            attn = attention[layer_idx].mean(dim=0)  # (seq_len, seq_len)
        elif head_fusion == "max":
            attn = attention[layer_idx].max(dim=0).values
        elif head_fusion == "min":
            attn = attention[layer_idx].min(dim=0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Optional: zero out the lowest-attention entries to reduce noise
        if discard_ratio > 0:
            flat = attn.flatten()
            threshold = flat.quantile(discard_ratio)
            attn = attn * (attn > threshold).float()

        # Add identity to simulate the residual connection:
        # the output of each layer is attention(x) + x, so the
        # effective mixing matrix is (I + A) / 2
        attn = attn + torch.eye(seq_len)

        # Re-normalize rows so they sum to 1 (maintain probability interpretation)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Accumulate: multiply with the rollout from previous layers
        rollout = torch.matmul(attn, rollout)

    return rollout.numpy()


# ====== Head-Level Analysis ======

def analyze_head_patterns(
    attention: torch.Tensor,
    tokens: list[str],
) -> dict:
    """Characterize individual attention heads by their behavior patterns.

    Different heads learn different linguistic functions:
    - Some attend to the previous token (positional)
    - Some attend to [CLS] or [SEP] (special tokens)
    - Some attend to semantically related tokens (content-based)
    - Some spread attention uniformly (high entropy = unfocused)

    We measure three diagnostics per head:
    1. Entropy: how spread out the attention distribution is.
    2. [CLS] attention: fraction of total attention directed at [CLS].
    3. Diagonal dominance: fraction attending to self or neighbors.
    """
    num_layers, num_heads, seq_len, _ = attention.shape
    results = {}

    for layer in range(num_layers):
        for head in range(num_heads):
            attn_matrix = attention[layer, head].numpy()  # (seq_len, seq_len)

            # Entropy: -sum(p * log(p)) averaged across query positions
            # Higher entropy = more uniform (less informative) attention
            eps = 1e-10
            entropy = -np.sum(
                attn_matrix * np.log(attn_matrix + eps), axis=-1
            ).mean()

            # CLS attention: mean fraction of attention going to [CLS]
            # High values suggest the head routes information to the
            # classification token
            cls_attention = attn_matrix[:, 0].mean()

            # Diagonal dominance: mean attention to self + immediate neighbors
            # High values indicate a positional (local) attention pattern
            diag_sum = 0.0
            for i in range(seq_len):
                for offset in [-1, 0, 1]:
                    j = i + offset
                    if 0 <= j < seq_len:
                        diag_sum += attn_matrix[i, j]
            diagonal_dominance = diag_sum / seq_len

            results[(layer, head)] = {
                "entropy": float(entropy),
                "cls_attention": float(cls_attention),
                "diagonal_dominance": float(diagonal_dominance),
            }

    return results


def find_interesting_heads(head_stats: dict, top_k: int = 5) -> dict:
    """Identify the most specialized heads by different criteria.

    Returns the top-k heads for each category: lowest entropy (most focused),
    highest CLS attention, and highest diagonal dominance (most positional).
    """
    items = list(head_stats.items())

    # Most focused (lowest entropy = sharpest attention distribution)
    by_entropy = sorted(items, key=lambda x: x[1]["entropy"])
    most_focused = by_entropy[:top_k]

    # Most CLS-oriented
    by_cls = sorted(items, key=lambda x: x[1]["cls_attention"], reverse=True)
    most_cls = by_cls[:top_k]

    # Most positional (highest diagonal dominance)
    by_diag = sorted(items, key=lambda x: x[1]["diagonal_dominance"], reverse=True)
    most_positional = by_diag[:top_k]

    return {
        "most_focused": most_focused,
        "most_cls_oriented": most_cls,
        "most_positional": most_positional,
    }


# ====== Visualization Functions ======

def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    tokens: list[str],
    title: str = "Attention Weights",
    save_path: str = None,
) -> None:
    """Plot a single attention matrix as a heatmap with token labels.

    Rows = query tokens (who is looking), columns = key tokens (who is
    being looked at). Darker colors indicate stronger attention.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)

    ax.set_xlabel("Key (attended to)", fontsize=11)
    ax.set_ylabel("Query (attending from)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to: {save_path}")
    plt.close()


def plot_multi_head_attention(
    attention: torch.Tensor,
    tokens: list[str],
    layer: int = 0,
    save_path: str = "attention_heads.png",
) -> None:
    """Visualize all attention heads from a single layer in a grid.

    This reveals the diversity of attention patterns: some heads attend
    locally, some globally, some to specific token types.
    """
    num_heads = attention.shape[1]
    cols = 4
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for head in range(num_heads):
        attn_matrix = attention[layer, head].numpy()
        im = axes[head].imshow(attn_matrix, cmap="Blues", aspect="auto")
        axes[head].set_title(f"Head {head}", fontsize=10)
        axes[head].set_xticks(range(len(tokens)))
        axes[head].set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
        axes[head].set_yticks(range(len(tokens)))
        axes[head].set_yticklabels(tokens, fontsize=6)

    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"All Attention Heads — Layer {layer}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


def plot_rollout_comparison(
    raw_attn: np.ndarray,
    rollout_attn: np.ndarray,
    tokens: list[str],
    save_path: str = "attention_rollout_comparison.png",
) -> None:
    """Side-by-side comparison of raw last-layer attention vs. rollout.

    Raw attention from the final layer misses the cumulative effect of
    residual connections. Rollout gives a more faithful picture of
    which input tokens actually influence each output position.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    im0 = axes[0].imshow(raw_attn, cmap="Blues", aspect="auto")
    axes[0].set_title("Raw Attention (Last Layer, Mean over Heads)", fontsize=12)
    axes[0].set_xticks(range(len(tokens)))
    axes[0].set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticks(range(len(tokens)))
    axes[0].set_yticklabels(tokens, fontsize=9)
    axes[0].set_xlabel("Key")
    axes[0].set_ylabel("Query")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rollout_attn, cmap="Reds", aspect="auto")
    axes[1].set_title("Attention Rollout (All Layers + Residual)", fontsize=12)
    axes[1].set_xticks(range(len(tokens)))
    axes[1].set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticks(range(len(tokens)))
    axes[1].set_yticklabels(tokens, fontsize=9)
    axes[1].set_xlabel("Key")
    axes[1].set_ylabel("Query")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Raw Attention vs Attention Rollout", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


def plot_cls_attribution(
    rollout_matrix: np.ndarray,
    tokens: list[str],
    save_path: str = "cls_token_attribution.png",
) -> None:
    """Bar chart showing how much each input token contributes to [CLS].

    In BERT, the [CLS] token's final representation is used for classification.
    The rollout matrix row for [CLS] tells us which input tokens most influenced
    the classification decision.
    """
    # Row 0 = [CLS] query: how much of each token flows into [CLS]
    cls_weights = rollout_matrix[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Reds(cls_weights / cls_weights.max())
    bars = ax.bar(range(len(tokens)), cls_weights, color=colors, edgecolor="gray")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Rollout Weight", fontsize=11)
    ax.set_title("[CLS] Token Attribution via Attention Rollout", fontsize=13,
                 fontweight="bold")

    # Annotate the top-3 most influential tokens
    top_indices = np.argsort(cls_weights)[::-1][:3]
    for idx in top_indices:
        ax.annotate(
            f"{cls_weights[idx]:.3f}",
            xy=(idx, cls_weights[idx]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


# ====== Main Pipeline ======

def main() -> None:
    """Analyze BERT attention patterns and compare raw vs. rollout."""
    print("=" * 60)
    print("  Transformer Attention Analysis")
    print("  Raw Attention | Attention Rollout | Head Analysis")
    print("=" * 60)

    # --- Step 1: Load model ---
    print("\n[1] Loading BERT model...")
    model, tokenizer = load_bert_model("bert-base-uncased")

    # --- Step 2: Tokenize a sentiment example ---
    # Why this sentence? It has clear positive sentiment carried by specific
    # words ("outstanding", "breathtaking"), letting us check whether
    # attention highlights the right tokens.
    text = "The movie was absolutely outstanding with breathtaking visual effects."
    print(f"\n[2] Input text: \"{text}\"")
    inputs, tokens = tokenize_input(text, tokenizer)
    print(f"  Tokens ({len(tokens)}): {tokens}")

    # --- Step 3: Extract attention ---
    print("\n[3] Extracting attention weights...")
    t0 = time.time()
    attention = extract_attention_weights(model, inputs)
    t_extract = time.time() - t0
    print(f"  Attention shape: {attention.shape}")
    print(f"  (layers={attention.shape[0]}, heads={attention.shape[1]}, "
          f"seq_len={attention.shape[2]})")
    print(f"  Time: {t_extract:.3f}s")

    # --- Step 4: Raw attention from the last layer ---
    print("\n[4] Visualizing raw attention (last layer)...")
    last_layer = attention.shape[0] - 1
    raw_last_layer = attention[last_layer].mean(dim=0).numpy()

    plot_attention_heatmap(
        raw_last_layer, tokens,
        title=f"Raw Attention — Layer {last_layer} (Mean over Heads)",
        save_path="raw_attention_last_layer.png",
    )

    # --- Step 5: Multi-head visualization ---
    # Layer 0 heads tend to be more interpretable because they operate
    # directly on the input embeddings (no cumulative residual effects).
    print("\n[5] Visualizing all heads from Layer 0...")
    plot_multi_head_attention(attention, tokens, layer=0,
                              save_path="attention_heads_layer0.png")

    print("  Visualizing all heads from last layer...")
    plot_multi_head_attention(attention, tokens, layer=last_layer,
                              save_path="attention_heads_last_layer.png")

    # --- Step 6: Attention Rollout ---
    print("\n[6] Computing Attention Rollout...")
    t0 = time.time()
    rollout_matrix = attention_rollout(attention, head_fusion="mean",
                                       discard_ratio=0.0)
    t_rollout = time.time() - t0
    print(f"  Rollout shape: {rollout_matrix.shape}")
    print(f"  Time: {t_rollout:.3f}s")

    # --- Step 7: Compare raw vs. rollout ---
    print("\n[7] Comparing raw attention vs. rollout...")
    plot_rollout_comparison(raw_last_layer, rollout_matrix, tokens)

    # --- Step 8: CLS token attribution ---
    print("\n[8] Visualizing [CLS] token attribution via rollout...")
    plot_cls_attribution(rollout_matrix, tokens)

    # Print the top tokens contributing to [CLS]
    cls_weights = rollout_matrix[0]
    sorted_indices = np.argsort(cls_weights)[::-1]
    print("\n  Top-5 tokens contributing to [CLS]:")
    for rank, idx in enumerate(sorted_indices[:5]):
        print(f"    {rank+1}. {tokens[idx]:>15s} -> {cls_weights[idx]:.4f}")

    # --- Step 9: Head-level analysis ---
    print("\n[9] Analyzing head specialization patterns...")
    head_stats = analyze_head_patterns(attention, tokens)
    interesting = find_interesting_heads(head_stats, top_k=5)

    print("\n  Most focused heads (lowest entropy):")
    for (layer, head), stats in interesting["most_focused"]:
        print(f"    Layer {layer:2d}, Head {head:2d}: "
              f"entropy={stats['entropy']:.3f}")

    print("\n  Most [CLS]-oriented heads:")
    for (layer, head), stats in interesting["most_cls_oriented"]:
        print(f"    Layer {layer:2d}, Head {head:2d}: "
              f"cls_attn={stats['cls_attention']:.3f}")

    print("\n  Most positional heads (high diagonal dominance):")
    for (layer, head), stats in interesting["most_positional"]:
        print(f"    Layer {layer:2d}, Head {head:2d}: "
              f"diag_dom={stats['diagonal_dominance']:.3f}")

    # --- Step 10: Rollout with discard for comparison ---
    print("\n[10] Comparing rollout with different discard ratios...")
    for discard in [0.0, 0.3, 0.6, 0.9]:
        rollout_d = attention_rollout(attention, head_fusion="mean",
                                      discard_ratio=discard)
        cls_entropy = -np.sum(
            rollout_d[0] * np.log(rollout_d[0] + 1e-10)
        )
        print(f"  Discard ratio={discard:.1f}: "
              f"[CLS] entropy={cls_entropy:.3f}, "
              f"max_weight={rollout_d[0].max():.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Observations:")
    print("  - Raw attention from a single layer ignores residual connections")
    print("    and can be misleading about actual information flow.")
    print("  - Attention Rollout propagates attention through all layers,")
    print("    giving a more faithful picture of token influence.")
    print("  - Different heads specialize: some attend locally (positional),")
    print("    some route to [CLS], some focus on specific content.")
    print("  - Higher discard ratios sharpen the rollout but may discard")
    print("    important low-magnitude attention patterns.")
    print("=" * 60)


if __name__ == "__main__":
    main()

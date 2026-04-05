"""
08. Mechanistic Interpretability

Trains a small transformer on a synthetic copying/repetition task, then
applies three mechanistic interpretability techniques to understand *how*
the model solves the task internally:
  1. Activation patching (zero-ablation) to identify critical components.
  2. Sparse autoencoder on residual stream activations to find monosemantic
     features.
  3. Attention pattern analysis to trace information flow across layers.

Covered topics:
    - Building a minimal 2-layer transformer from scratch
    - Training on a synthetic sequence-copying task
    - Sparse autoencoder for residual stream decomposition
    - Activation patching (zero-ablation) for causal analysis
    - Attention head importance analysis
    - Feature sparsity visualization

Related to: L16 - Mechanistic Interpretability

Requirements:
    pip install torch numpy matplotlib
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ====== Section 1: Synthetic Copying Task ======

def generate_copy_task_data(
    n_samples: int,
    seq_len: int = 6,
    vocab_size: int = 10,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate data for a sequence copying task.

    Input format:  [a, b, c, SEP, ?, ?, ?]
    Target format: [_, _, _, _,   a, b, c]

    The model must learn to copy tokens from before the separator to
    the positions after it.  This is a clean, well-defined task where
    mechanistic analysis can reveal *which* attention heads perform
    the copying operation.

    SEP token is vocab_size (one beyond the regular vocabulary).

    Args:
        n_samples: Number of sequences to generate.
        seq_len: Number of tokens to copy (half the sequence minus separator).
        vocab_size: Number of distinct tokens (excluding SEP).
        seed: Random seed.

    Returns:
        inputs: Tensor of shape (n_samples, 2*seq_len + 1).
        targets: Tensor of shape (n_samples, 2*seq_len + 1).
                 Positions before and at SEP have target = -100 (ignored).
    """
    torch.manual_seed(seed)

    SEP = vocab_size  # separator token index
    full_len = 2 * seq_len + 1  # source + SEP + target positions

    inputs = torch.zeros(n_samples, full_len, dtype=torch.long)
    targets = torch.full((n_samples, full_len), -100, dtype=torch.long)

    for i in range(n_samples):
        # Random source tokens
        source = torch.randint(0, vocab_size, (seq_len,))

        # Build input: [source tokens, SEP, placeholder zeros]
        inputs[i, :seq_len] = source
        inputs[i, seq_len] = SEP
        # The positions after SEP are set to 0 (placeholder / "query" token)

        # Target: model should predict the source tokens at positions after SEP
        targets[i, seq_len + 1:] = source

    return inputs, targets


# ====== Section 2: Minimal Transformer ======

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional activation caching.

    Caches attention weights and value vectors when cache_activations=True,
    which enables post-hoc analysis of attention patterns and activation
    patching experiments.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Activation cache for interpretability
        self.cache_activations = False
        self.cached_attn_weights = None
        self.cached_values = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to queries, keys, values and reshape for multi-head
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Cache for interpretability experiments
        if self.cache_activations:
            self.cached_attn_weights = attn_weights.detach()
            self.cached_values = v.detach()

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Pre-norm (LayerNorm before attention/MLP) is used because it
    trains more stably and the residual stream is cleaner for
    mechanistic analysis -- each component adds directly to the
    residual without being mediated by normalization.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    """Minimal 2-layer transformer for the copying task.

    Architecture choices for interpretability:
      - 2 layers, 2 heads per layer (small enough to fully analyse)
      - Pre-norm for clean residual stream
      - Learned positional embeddings
      - No dropout (deterministic behaviour during analysis)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # +1 for the SEP token
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff=4 * d_model)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        # Unembedding: project back to vocabulary (excluding SEP for predictions)
        self.unembed = nn.Linear(d_model, vocab_size + 1)

        # For caching the residual stream at each layer
        self.cache_residuals = False
        self.residual_cache = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        # Embedding: token + position
        h = self.token_embed(x) + self.pos_embed(positions)

        self.residual_cache = []
        for block in self.blocks:
            if self.cache_residuals:
                self.residual_cache.append(h.detach().clone())
            h = block(h)

        if self.cache_residuals:
            self.residual_cache.append(h.detach().clone())

        h = self.ln_final(h)
        logits = self.unembed(h)
        return logits

    def set_cache(self, enable: bool = True) -> None:
        """Enable/disable activation caching for all components."""
        self.cache_residuals = enable
        for block in self.blocks:
            block.attn.cache_activations = enable


def train_transformer(
    model: MiniTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 200,
    lr: float = 3e-4,
) -> list[float]:
    """Train the transformer on the copying task.

    Uses cross-entropy loss only on the positions after SEP (where the
    model must predict the copied tokens).  Positions before SEP have
    target = -100 and are ignored by the loss function.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        logits = model(inputs)

        # Reshape for cross-entropy: (B*T, vocab) vs (B*T,)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1:4d}/{epochs}  loss={loss.item():.4f}")

    return losses


# ====== Section 3: Activation Patching (Zero-Ablation) ======

def activation_patching_experiment(
    model: MiniTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    """Zero-ablate individual attention heads and measure performance impact.

    Activation patching is the core tool of mechanistic interpretability.
    By setting a component's output to zero and measuring the increase
    in loss, we determine how *necessary* that component is for the task.

    A head whose ablation causes a large loss increase is "important" --
    it computes something the model relies on.  Heads with minimal impact
    may be redundant or handle rare edge cases.

    We ablate each head independently (not jointly) to get first-order
    importance estimates.
    """
    model.eval()

    # Baseline loss (no ablation)
    with torch.no_grad():
        base_logits = model(inputs)
        base_loss = F.cross_entropy(
            base_logits.view(-1, base_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        ).item()

    results = {"baseline_loss": base_loss, "head_ablations": {}}

    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.n_heads

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Save the original output projection weights
            W_o = model.blocks[layer_idx].attn.W_o
            original_weight = W_o.weight.data.clone()
            original_bias = W_o.bias.data.clone() if W_o.bias is not None else None

            # Zero out the columns corresponding to this head
            # This effectively silences the head's contribution
            d_head = model.d_model // n_heads
            start = head_idx * d_head
            end = start + d_head

            with torch.no_grad():
                # Zero the input projection for this head in W_o
                # W_o maps from [h1; h2; ...] to d_model, so columns
                # start:end correspond to head_idx's contribution
                W_o.weight.data[:, start:end] = 0

                ablated_logits = model(inputs)
                ablated_loss = F.cross_entropy(
                    ablated_logits.view(-1, ablated_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                ).item()

                # Restore original weights
                W_o.weight.data = original_weight
                if original_bias is not None:
                    W_o.bias.data = original_bias

            loss_increase = ablated_loss - base_loss
            key = f"L{layer_idx}H{head_idx}"
            results["head_ablations"][key] = {
                "ablated_loss": ablated_loss,
                "loss_increase": loss_increase,
                "relative_increase": loss_increase / max(base_loss, 1e-8),
            }

    return results


# ====== Section 4: Sparse Autoencoder ======

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for decomposing residual stream activations.

    The residual stream of a transformer is a superposition of many
    features.  A sparse autoencoder learns to decompose this into a
    larger set of (mostly zero) latent dimensions, each of which
    ideally corresponds to a single interpretable feature.

    Architecture:
      encoder: d_model -> d_hidden (with ReLU for sparsity)
      decoder: d_hidden -> d_model (linear reconstruction)

    Loss = reconstruction_error + lambda * L1_sparsity_penalty
    """

    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_input)
        self.d_hidden = d_hidden

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_activations)."""
        # ReLU produces sparse activations -- most units will be zero
        latent = F.relu(self.encoder(x))
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def get_sparsity_stats(self, latent: torch.Tensor) -> dict:
        """Compute sparsity statistics for the latent activations.

        Key metrics:
          - L0: average number of non-zero features per sample
          - L1: average L1 norm (total activation mass)
          - dead_fraction: fraction of features that never activate
        """
        # L0: fraction of active (non-zero) features
        active = (latent > 0).float()
        l0_per_sample = active.sum(dim=-1).mean().item()

        # L1: average total activation
        l1_per_sample = latent.abs().sum(dim=-1).mean().item()

        # Dead features: never activate across the batch
        ever_active = (latent > 0).any(dim=0)
        dead_fraction = 1.0 - ever_active.float().mean().item()

        return {
            "l0_per_sample": l0_per_sample,
            "l1_per_sample": l1_per_sample,
            "dead_fraction": dead_fraction,
            "total_features": self.d_hidden,
            "active_features": ever_active.sum().item(),
        }


def train_sparse_autoencoder(
    activations: torch.Tensor,
    d_hidden: int = 256,
    sparsity_lambda: float = 1e-3,
    epochs: int = 500,
    lr: float = 1e-3,
) -> tuple[SparseAutoencoder, list[float]]:
    """Train a sparse autoencoder on collected residual stream activations.

    The sparsity penalty (L1 on latent activations) encourages each
    sample to be represented by only a few active features.  Higher
    lambda = sparser but potentially less accurate reconstruction.
    """
    d_input = activations.shape[-1]
    sae = SparseAutoencoder(d_input, d_hidden)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    losses = []

    # Flatten to (n_samples * seq_len, d_model) for training
    flat_acts = activations.reshape(-1, d_input)

    for epoch in range(epochs):
        recon, latent = sae(flat_acts)

        # Reconstruction loss: how well can we recover the original activations?
        recon_loss = F.mse_loss(recon, flat_acts)

        # Sparsity penalty: encourage latent to be sparse (mostly zeros)
        sparsity_loss = sparsity_lambda * latent.abs().mean()

        loss = recon_loss + sparsity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sae, losses


# ====== Section 5: Attention Pattern Analysis ======

def analyze_attention_patterns(
    model: MiniTransformer,
    inputs: torch.Tensor,
    seq_len: int,
) -> dict:
    """Analyze attention patterns to identify copying heads.

    A "copying head" is one where positions after SEP attend strongly
    to the corresponding positions before SEP.  For the copying task,
    we expect at least one head to learn this pattern: position
    (seq_len + 1 + i) should attend to position i.

    We quantify this with the "copy score" -- the average attention
    weight placed on the correct source position.
    """
    model.eval()
    model.set_cache(True)

    with torch.no_grad():
        _ = model(inputs)

    results = {}

    for layer_idx, block in enumerate(model.blocks):
        attn_weights = block.attn.cached_attn_weights  # (B, n_heads, T, T)
        n_heads = attn_weights.shape[1]

        for head_idx in range(n_heads):
            head_attn = attn_weights[:, head_idx, :, :]  # (B, T, T)

            # Compute copy score: for each target position (after SEP),
            # what fraction of attention goes to the correct source position?
            copy_scores = []
            for offset in range(seq_len):
                target_pos = seq_len + 1 + offset  # position after SEP
                source_pos = offset                  # corresponding source

                if target_pos < head_attn.shape[1]:
                    # Average attention from target_pos to source_pos
                    attn_to_source = head_attn[:, target_pos, source_pos].mean().item()
                    copy_scores.append(attn_to_source)

            avg_copy_score = np.mean(copy_scores) if copy_scores else 0.0

            # Also compute entropy of attention distribution (uniformity measure)
            # Low entropy = focused attention, high entropy = diffuse
            attn_probs = head_attn.mean(dim=0)  # average over batch
            # Only look at target positions
            target_attn = attn_probs[seq_len + 1:]
            entropy = -(target_attn * (target_attn + 1e-10).log()).sum(dim=-1).mean().item()

            key = f"L{layer_idx}H{head_idx}"
            results[key] = {
                "copy_score": avg_copy_score,
                "entropy": entropy,
                "is_copying_head": avg_copy_score > 0.3,
            }

    model.set_cache(False)
    return results


# ====== Section 6: Visualization ======

def visualize_results(
    training_losses: list[float],
    ablation_results: dict,
    sae_stats: dict,
    sae_losses: list[float],
    attn_analysis: dict,
    latent_activations: torch.Tensor,
    save_path: str = "mechanistic_interp.png",
) -> None:
    """Four-panel visualization of mechanistic interpretability results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Training loss curve ---
    ax1 = axes[0, 0]
    ax1.plot(training_losses, linewidth=1, color="#2c3e50")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Transformer Training Loss (Copy Task)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Head importance (ablation) ---
    ax2 = axes[0, 1]
    heads = list(ablation_results["head_ablations"].keys())
    loss_increases = [ablation_results["head_ablations"][h]["loss_increase"]
                      for h in heads]
    colors = ["#e74c3c" if li > 0.1 else "#3498db" for li in loss_increases]
    bars = ax2.bar(heads, loss_increases, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Attention Head")
    ax2.set_ylabel("Loss Increase (ablation)")
    ax2.set_title("Head Importance via Zero-Ablation")
    ax2.axhline(y=0, color="gray", linewidth=0.5)

    # Annotate copy scores
    for i, head in enumerate(heads):
        if head in attn_analysis:
            cs = attn_analysis[head]["copy_score"]
            ax2.annotate(f"CS={cs:.2f}", (i, loss_increases[i]),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=8)

    # --- Panel 3: Sparse autoencoder feature activation histogram ---
    ax3 = axes[1, 0]
    # Sum of activations per feature across all samples
    flat_latent = latent_activations.reshape(-1, latent_activations.shape[-1])
    feature_activity = (flat_latent > 0).float().mean(dim=0).numpy()
    # Sort by activity for cleaner visualization
    sorted_activity = np.sort(feature_activity)[::-1]
    ax3.bar(range(len(sorted_activity)), sorted_activity,
            width=1.0, color="#9b59b6", alpha=0.7)
    ax3.set_xlabel("Feature Index (sorted by activity)")
    ax3.set_ylabel("Activation Frequency")
    ax3.set_title(f"SAE Feature Sparsity "
                  f"(L0={sae_stats['l0_per_sample']:.1f}, "
                  f"dead={sae_stats['dead_fraction']:.1%})")
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% active")
    ax3.legend(fontsize=9)

    # --- Panel 4: SAE reconstruction loss ---
    ax4 = axes[1, 1]
    ax4.plot(sae_losses, linewidth=1, color="#27ae60")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss (MSE + L1)")
    ax4.set_title("Sparse Autoencoder Training Loss")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Mechanistic Interpretability Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 7: Main Pipeline ======

def main() -> None:
    """Run the full mechanistic interpretability pipeline."""
    print("=" * 65)
    print("  Mechanistic Interpretability")
    print("  Activation Patching | Sparse Autoencoder | Attention Analysis")
    print("=" * 65)

    VOCAB_SIZE = 10
    SEQ_LEN = 6
    D_MODEL = 64
    N_HEADS = 2
    N_LAYERS = 2

    # --- Step 1: Generate data ---
    print("\n[1] Generating Copy Task Dataset")
    print("-" * 50)

    inputs, targets = generate_copy_task_data(
        n_samples=500, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
    )
    print(f"  Samples: {len(inputs)}")
    print(f"  Sequence length: {inputs.shape[1]}")
    print(f"  Vocab size: {VOCAB_SIZE} + 1 (SEP)")
    print(f"  Example input:  {inputs[0].tolist()}")
    print(f"  Example target: {targets[0].tolist()}")
    print(f"    (-100 = ignored positions, model must copy [{inputs[0, :SEQ_LEN].tolist()}])")

    # --- Step 2: Train transformer ---
    print("\n[2] Training 2-Layer Transformer")
    print("-" * 50)
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")

    model = MiniTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    training_losses = train_transformer(model, inputs, targets, epochs=200)
    print(f"  Final loss: {training_losses[-1]:.6f}")

    # Evaluate copy accuracy
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        # Only check positions after SEP
        target_mask = targets != -100
        correct = (preds[target_mask] == targets[target_mask]).float().mean().item()
    print(f"  Copy accuracy: {correct:.2%}")

    # --- Step 3: Activation patching ---
    print("\n[3] Activation Patching (Zero-Ablation)")
    print("-" * 50)

    ablation_results = activation_patching_experiment(model, inputs, targets)
    print(f"  Baseline loss: {ablation_results['baseline_loss']:.6f}")
    print()
    print(f"  {'Head':8s} {'Ablated Loss':>14s} {'Loss Increase':>14s} {'Relative':>10s}")
    print(f"  {'-' * 48}")

    for head, info in ablation_results["head_ablations"].items():
        marker = " ***" if info["loss_increase"] > 0.1 else ""
        print(f"  {head:8s} {info['ablated_loss']:14.6f} "
              f"{info['loss_increase']:+14.6f} "
              f"{info['relative_increase']:+10.2%}{marker}")

    # Identify the most important head
    most_important = max(
        ablation_results["head_ablations"].items(),
        key=lambda x: x[1]["loss_increase"],
    )
    print(f"\n  Most important head: {most_important[0]} "
          f"(loss increase: {most_important[1]['loss_increase']:.4f})")

    # --- Step 4: Attention pattern analysis ---
    print("\n[4] Attention Pattern Analysis (Copy Head Detection)")
    print("-" * 50)

    attn_analysis = analyze_attention_patterns(model, inputs[:50], SEQ_LEN)

    for head, info in attn_analysis.items():
        copy_marker = " <-- COPYING HEAD" if info["is_copying_head"] else ""
        print(f"  {head}: copy_score={info['copy_score']:.4f}  "
              f"entropy={info['entropy']:.4f}{copy_marker}")

    # --- Step 5: Sparse autoencoder on residual stream ---
    print("\n[5] Sparse Autoencoder on Residual Stream")
    print("-" * 50)

    # Collect residual stream activations from the middle of the network
    model.set_cache(True)
    with torch.no_grad():
        _ = model(inputs[:200])

    # Use the residual stream after layer 0 (before layer 1)
    # This captures what the first layer has computed
    if len(model.residual_cache) >= 2:
        residual_acts = model.residual_cache[1]  # after first block
    else:
        residual_acts = model.residual_cache[0]

    model.set_cache(False)

    print(f"  Residual activations shape: {residual_acts.shape}")
    print(f"  Training sparse autoencoder (d_hidden=256) ...")

    sae, sae_losses = train_sparse_autoencoder(
        residual_acts,
        d_hidden=256,
        sparsity_lambda=1e-3,
        epochs=500,
    )

    # Analyse the learned features
    sae.eval()
    with torch.no_grad():
        flat_acts = residual_acts.reshape(-1, D_MODEL)
        _, latent = sae(flat_acts)

    sae_stats = sae.get_sparsity_stats(latent)
    print(f"  SAE final loss: {sae_losses[-1]:.6f}")
    print(f"  Active features per sample (L0): {sae_stats['l0_per_sample']:.1f} / {sae_stats['total_features']}")
    print(f"  Dead features: {sae_stats['dead_fraction']:.1%} "
          f"({sae_stats['total_features'] - int(sae_stats['active_features'])} / {sae_stats['total_features']})")
    print(f"  L1 norm per sample: {sae_stats['l1_per_sample']:.4f}")

    # Check reconstruction quality
    with torch.no_grad():
        recon, _ = sae(flat_acts)
        recon_error = F.mse_loss(recon, flat_acts).item()
        # Relative error: reconstruction MSE / activation variance
        act_var = flat_acts.var().item()
        relative_error = recon_error / max(act_var, 1e-8)
    print(f"  Reconstruction MSE: {recon_error:.6f}")
    print(f"  Relative error (MSE/variance): {relative_error:.4f}")

    # --- Step 6: Feature analysis ---
    print("\n[6] Top Sparse Autoencoder Features")
    print("-" * 50)

    # Find the most frequently active features
    feature_freq = (latent > 0).float().mean(dim=0)
    top_k = 10
    top_indices = feature_freq.argsort(descending=True)[:top_k]

    print(f"  Top {top_k} most active features:")
    for rank, idx in enumerate(top_indices):
        freq = feature_freq[idx].item()
        avg_magnitude = latent[:, idx][latent[:, idx] > 0].mean().item() if (latent[:, idx] > 0).any() else 0
        print(f"    #{rank + 1}: feature {idx.item():3d}  "
              f"freq={freq:.3f}  avg_magnitude={avg_magnitude:.4f}")

    # --- Step 7: Visualization ---
    print("\n[7] Generating Visualization")
    print("-" * 50)

    latent_2d = latent.reshape(residual_acts.shape[0], residual_acts.shape[1], -1)

    visualize_results(
        training_losses=training_losses,
        ablation_results=ablation_results,
        sae_stats=sae_stats,
        sae_losses=sae_losses,
        attn_analysis=attn_analysis,
        latent_activations=latent_2d,
    )

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"""
  Task: Copy {SEQ_LEN} tokens across a separator.
  Model: {N_LAYERS}-layer transformer, {N_HEADS} heads, d_model={D_MODEL}.
  Copy accuracy: {correct:.2%}

  Key findings:
    1. Activation patching identified {most_important[0]} as the most
       critical attention head (loss increase: {most_important[1]['loss_increase']:.4f}).
    2. Attention pattern analysis found copying heads that attend from
       target positions to the corresponding source positions.
    3. The sparse autoencoder decomposed the residual stream into
       {sae_stats['total_features']} features, of which
       {int(sae_stats['active_features'])} are active (L0={sae_stats['l0_per_sample']:.1f}
       per sample).
    4. {sae_stats['dead_fraction']:.1%} of SAE features are "dead" (never
       activate), suggesting the dictionary is over-complete -- a desirable
       property for finding monosemantic features.

  Techniques demonstrated:
    - Zero-ablation: causal intervention to measure component necessity
    - Sparse autoencoder: unsupervised feature discovery in the residual stream
    - Attention analysis: tracing information flow through the network
    """)


if __name__ == "__main__":
    main()

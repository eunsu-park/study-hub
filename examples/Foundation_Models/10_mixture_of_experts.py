#!/usr/bin/env python3
"""
Foundation Models - Mixture of Experts (MoE)
=============================================

Implements a Mixture of Experts layer from scratch using PyTorch.
Demonstrates top-k gating, expert networks, load balancing loss,
and sparse activation — the architecture behind Mixtral, Switch
Transformer, and GPT-4 (rumored).

Key Idea:
    Instead of one large FFN, use N smaller "expert" FFNs and a gating
    network that routes each token to only the top-k experts. This
    achieves large model capacity with sub-linear compute cost.

Why MoE matters:
    - A 47B-parameter MoE model can match a 13B dense model's quality
      while using only 2/8 experts per token (much less compute)
    - Scales model capacity without proportional compute increase
    - But introduces load balancing challenges (experts must be used evenly)

Requires: PyTorch, numpy, matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Expert(nn.Module):
    """
    A single expert network — a standard FFN (Feed-Forward Network).

    In Mixtral, each expert is an FFN with SwiGLU activation.
    Here we use a simplified 2-layer FFN for clarity.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Why: SiLU (Swish) is used in modern LLMs over ReLU because it
        # is smooth (no dead neurons) and slightly outperforms GELU.
        return self.fc2(F.silu(self.fc1(x)))


class TopKGating(nn.Module):
    """
    Top-k gating network that routes tokens to experts.

    The gating network is a simple linear layer that produces logits
    for each expert, then selects the top-k with softmax weights.

    Why top-k (not top-1)?
        Top-1 (Switch Transformer) is most efficient but less stable.
        Top-2 (Mixtral) balances efficiency with training stability.
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2,
                 noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Why: A single linear layer is sufficient because the gating
        # decision is based on the input representation, which is already
        # rich from previous layers.
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, dim) or (batch_size, dim)

        Returns:
            gates: (batch, top_k) — softmax weights for selected experts
            indices: (batch, top_k) — which experts are selected
            load: (num_experts,) — fraction of tokens routed to each expert
        """
        # Flatten to 2D for gating
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))

        logits = self.gate(x)  # (N, num_experts)

        # Why: Adding noise during training encourages exploration of
        # different experts, preventing "expert collapse" where only
        # 1-2 experts are ever used.
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Select top-k experts per token
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Why: Softmax over only the selected experts (not all N).
        # This makes the weights sum to 1 across the active experts.
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Compute load for balancing loss
        # Why: We want each expert to handle roughly 1/N of all tokens.
        # Without balancing, the rich get richer (popular experts get more
        # training signal, become better, attract more tokens).
        with torch.no_grad():
            # Count how many tokens each expert receives
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                expert_counts.scatter_add_(
                    0, top_k_indices[:, k],
                    torch.ones(x.size(0), device=x.device)
                )
            load = expert_counts / (x.size(0) * self.top_k)

        return top_k_gates, top_k_indices, load


class MoELayer(nn.Module):
    """
    Complete Mixture of Experts layer with load balancing.

    Architecture:
        Input → Gating → Route to top-k experts → Weighted sum → Output

    This replaces the standard FFN in a Transformer block.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 8,
                 top_k: int = 2, balance_coeff: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_coeff = balance_coeff

        self.gating = TopKGating(input_dim, num_experts, top_k)
        # Why: nn.ModuleList ensures all experts are registered as
        # submodules (so .parameters() includes them for optimization).
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim) for _ in range(num_experts)
        ])

    def load_balancing_loss(self, load: torch.Tensor):
        """
        Auxiliary loss to encourage even expert utilization.

        L_balance = N * sum(f_i * P_i)
        where f_i = fraction of tokens to expert i
              P_i = mean gating probability for expert i

        Why multiply by N?
            Without N, the loss is O(1/N) and vanishes for large N.
            Multiplying by N makes it O(1) regardless of expert count.
        """
        # Ideal load: each expert gets 1/N of tokens
        target = torch.ones_like(load) / self.num_experts
        # Why: We use the variance of the load distribution.
        # Minimizing variance pushes all experts toward equal usage.
        loss = self.num_experts * torch.sum(load * load)
        return loss

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, dim) or (batch_size, dim)

        Returns:
            output: Same shape as input
            aux_loss: Load balancing loss (scalar)
        """
        orig_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*S, D)
        else:
            x_flat = x
            dim = x.size(-1)

        gates, indices, load = self.gating(x_flat)
        # gates: (N, top_k), indices: (N, top_k)

        # Compute output by dispatching to selected experts
        # Why: In production, this uses a scatter/gather pattern for
        # GPU efficiency. Here we use a simple loop for clarity.
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = indices[:, k]  # (N,) — which expert for each token
            gate_weight = gates[:, k]   # (N,) — weight for this expert

            # Group tokens by expert for batch processing
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    # Why: Multiply by gate weight so the final output is
                    # a weighted combination of the top-k expert outputs.
                    output[mask] += gate_weight[mask].unsqueeze(-1) * expert_output

        if len(orig_shape) == 3:
            output = output.reshape(orig_shape)

        aux_loss = self.balance_coeff * self.load_balancing_loss(load)
        return output, aux_loss


def visualize_expert_routing(moe_layer, x, title="Expert Routing"):
    """Visualize which experts each token is routed to."""
    moe_layer.eval()
    with torch.no_grad():
        gates, indices, load = moe_layer.gating(x.reshape(-1, x.size(-1)))

    num_tokens = indices.size(0)
    num_experts = moe_layer.num_experts

    # Build routing matrix
    routing = np.zeros((num_tokens, num_experts))
    for k in range(moe_layer.top_k):
        for i in range(num_tokens):
            routing[i, indices[i, k].item()] = gates[i, k].item()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Routing heatmap
    im = axes[0].imshow(routing[:min(32, num_tokens)].T, aspect='auto',
                        cmap='Blues', vmin=0, vmax=1)
    axes[0].set_xlabel("Token index")
    axes[0].set_ylabel("Expert index")
    axes[0].set_title("Token-Expert Routing Weights")
    plt.colorbar(im, ax=axes[0])

    # Plot 2: Expert load distribution
    expert_load = load.numpy()
    colors = ['green' if abs(l - 1.0 / num_experts) < 0.05 else 'red'
              for l in expert_load]
    axes[1].bar(range(num_experts), expert_load, color=colors, alpha=0.7)
    axes[1].axhline(y=1.0 / num_experts, color='k', linestyle='--',
                    label=f'Ideal = 1/{num_experts}')
    axes[1].set_xlabel("Expert index")
    axes[1].set_ylabel("Fraction of tokens")
    axes[1].set_title("Expert Load Distribution")
    axes[1].legend()

    # Plot 3: Gate weight distribution
    axes[2].hist(gates.numpy().flatten(), bins=30, alpha=0.7, color='steelblue')
    axes[2].set_xlabel("Gate weight")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Gate Weight Distribution")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("moe_routing.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: moe_routing.png")


if __name__ == "__main__":
    torch.manual_seed(42)

    # Configuration
    input_dim = 64
    hidden_dim = 128
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 16

    print("=" * 60)
    print("Mixture of Experts (MoE) Layer Demonstration")
    print("=" * 60)
    print(f"\nConfig: dim={input_dim}, hidden={hidden_dim}, "
          f"experts={num_experts}, top_k={top_k}")
    print(f"Input: batch={batch_size}, seq_len={seq_len}")

    # Create MoE layer
    moe = MoELayer(input_dim, hidden_dim, num_experts, top_k)

    # Count parameters
    total_params = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for p in moe.experts.parameters())
    dense_equiv = input_dim * hidden_dim * 2 + input_dim + hidden_dim
    print(f"\nMoE total parameters: {total_params:,}")
    print(f"Dense FFN equivalent: {dense_equiv:,}")
    print(f"Active params per token: ~{expert_params // num_experts * top_k:,} "
          f"({top_k}/{num_experts} experts)")

    # Forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    output, aux_loss = moe(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Aux loss (load balancing): {aux_loss.item():.4f}")

    # Training loop demonstrating load balancing
    print("\n--- Training with Load Balancing ---")
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)
    moe.train()

    for step in range(50):
        x = torch.randn(batch_size, seq_len, input_dim)
        target = torch.randn_like(x)

        output, aux_loss = moe(x)
        task_loss = F.mse_loss(output, target)
        # Why: Total loss = task loss + auxiliary balancing loss.
        # The aux_loss nudges the gating network to distribute evenly.
        total_loss = task_loss + aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            _, _, load = moe.gating(x.reshape(-1, input_dim))
            load_std = load.std().item()
            print(f"  Step {step + 1:3d}: task={task_loss.item():.4f}, "
                  f"aux={aux_loss.item():.4f}, load_std={load_std:.4f}")

    # Visualize routing
    print("\nGenerating visualization...")
    x_viz = torch.randn(1, 64, input_dim)
    visualize_expert_routing(moe, x_viz, "MoE Expert Routing After Training")

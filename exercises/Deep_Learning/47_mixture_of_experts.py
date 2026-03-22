"""
Exercises for Lesson 47: Mixture of Experts
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Exercise 1: MoE Parameter Efficiency ===
# Problem: Compare parameter counts between a dense FFN and a sparse MoE layer
# with the same active compute (same FLOPs per token).
# Dense FFN: d_model -> 4*d_model -> d_model
# MoE: E experts, top-k=2, each expert = d_model -> d_ff -> d_model

def exercise_1():
    """Compare parameter counts: dense FFN vs MoE at similar FLOPs."""
    d_model = 512
    dense_d_ff = 4 * d_model   # Standard transformer FFN expansion ratio

    # Dense FFN parameters (2 weight matrices)
    dense_params = d_model * dense_d_ff + dense_d_ff * d_model
    print("  Dense FFN params: {:,} (d_model={}, d_ff={})".format(
        dense_params, d_model, dense_d_ff))

    # MoE: with top-2 routing, we want each token to touch ~same FLOPs as dense
    # If top_k=2, each expert can be half the width: d_ff_expert = 2*d_model
    top_k = 2
    d_ff_expert = dense_d_ff // top_k   # Each expert is smaller to match FLOPs per token

    for num_experts in [4, 8, 16, 64]:
        # Expert params
        expert_params_each = d_model * d_ff_expert + d_ff_expert * d_model
        total_expert_params = num_experts * expert_params_each
        # Router params
        router_params = d_model * num_experts
        total_moe_params = total_expert_params + router_params

        # FLOPs per token: only top_k experts activated
        active_params = top_k * expert_params_each
        param_ratio = total_moe_params / dense_params
        flop_ratio = active_params / dense_params

        print("  MoE E={:2d}: total params={:,} ({:.1f}x dense), "
              "active per token={:,} ({:.2f}x dense FLOPs)".format(
            num_experts, total_moe_params, param_ratio, active_params, flop_ratio))

    print("  Key insight: MoE scales total params without proportional FLOP increase.")


# === Exercise 2: Top-k Router ===
# Problem: Implement a top-k router that returns gating weights and expert indices.
# Test that routing is sparse (only top_k experts receive non-zero weights per token).

def exercise_2():
    """Implement and test a top-k router."""

    class TopKRouter(nn.Module):
        """Token-choice router: each token independently selects top-k experts."""

        def __init__(self, d_model, num_experts, top_k=2):
            super().__init__()
            self.top_k = top_k
            self.num_experts = num_experts
            self.gate = nn.Linear(d_model, num_experts, bias=False)

        def forward(self, x):
            """
            Args:
                x: (num_tokens, d_model)
            Returns:
                indices: (num_tokens, top_k) — selected expert indices
                weights: (num_tokens, top_k) — normalized gating weights (softmax)
                logits:  (num_tokens, num_experts) — raw router logits
            """
            logits = self.gate(x)                              # (T, E)
            topk_logits, indices = torch.topk(logits, self.top_k, dim=-1)
            weights = F.softmax(topk_logits, dim=-1)           # Normalized over selected experts
            return indices, weights, logits

    torch.manual_seed(42)
    d_model, num_experts, top_k = 64, 8, 2
    router = TopKRouter(d_model, num_experts, top_k)

    # Batch of tokens
    tokens = torch.randn(16, d_model)
    indices, weights, logits = router(tokens)

    print("  Tokens: {}".format(tokens.shape))
    print("  Selected expert indices shape: {}".format(indices.shape))
    print("  Gating weights shape: {}".format(weights.shape))
    print("  Weights sum to 1 per token: {}".format(
        torch.allclose(weights.sum(dim=-1), torch.ones(16))))
    print("  Unique experts used (first batch): {}".format(
        indices.unique().tolist()))

    # Token distribution per expert (load balance check)
    expert_load = torch.zeros(num_experts, dtype=torch.long)
    for e in range(num_experts):
        expert_load[e] = (indices == e).sum()
    print("  Tokens per expert: {}".format(expert_load.tolist()))
    print("  Ideal balanced load: {} tokens per expert (total={})".format(
        16 * top_k // num_experts, 16 * top_k))


# === Exercise 3: MoE Layer Forward Pass ===
# Problem: Build a complete MoE layer that:
# 1. Routes each token to top-k experts
# 2. Computes each expert's output
# 3. Combines outputs with gating weights
# 4. Returns the combined output and a load balancing auxiliary loss.

def exercise_3():
    """Implement a full MoE layer and verify forward pass."""

    class Expert(nn.Module):
        """Standard FFN expert."""

        def __init__(self, d_model, d_ff):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x):
            return self.w2(F.relu(self.w1(x)))

    class MoELayer(nn.Module):
        """Sparse MoE layer with top-k routing and load-balancing auxiliary loss."""

        def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.gate = nn.Linear(d_model, num_experts, bias=False)
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

        def load_balancing_loss(self, router_logits):
            """
            Auxiliary load balancing loss from Switch Transformer.
            Encourages uniform token distribution across experts.

            Loss = num_experts * sum_over_experts(f_i * P_i)
            where f_i = fraction of tokens routed to expert i,
                  P_i = average routing probability for expert i.
            """
            probs = F.softmax(router_logits, dim=-1)          # (T, E)
            indices = router_logits.argmax(dim=-1)             # top-1 for simplicity
            num_tokens = router_logits.shape[0]

            # Fraction of tokens dispatched to each expert
            f = torch.zeros(self.num_experts, device=router_logits.device)
            for e in range(self.num_experts):
                f[e] = (indices == e).float().sum() / num_tokens

            # Mean routing probability per expert
            P = probs.mean(dim=0)  # (E,)

            # Auxiliary loss
            aux_loss = self.num_experts * (f * P).sum()
            return aux_loss

        def forward(self, x):
            """
            Args:
                x: (B, L, D)
            Returns:
                out: (B, L, D)
                aux_loss: scalar load-balancing loss
            """
            B, L, D = x.shape
            x_flat = x.reshape(-1, D)  # (T, D) where T = B*L

            logits = self.gate(x_flat)  # (T, E)
            topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1)  # (T, top_k)

            output = torch.zeros_like(x_flat)
            for k in range(self.top_k):
                expert_idx = topk_indices[:, k]    # (T,) — which expert for slot k
                weights_k = topk_weights[:, k]     # (T,) — weight for slot k

                for e in range(self.num_experts):
                    mask = (expert_idx == e)       # tokens routed to expert e
                    if mask.any():
                        expert_out = self.experts[e](x_flat[mask])  # (m, D)
                        output[mask] += weights_k[mask].unsqueeze(-1) * expert_out

            aux_loss = self.load_balancing_loss(logits)
            return output.reshape(B, L, D), aux_loss

    torch.manual_seed(7)
    d_model, d_ff, num_experts, top_k = 64, 256, 8, 2
    moe = MoELayer(d_model, d_ff, num_experts, top_k)

    x = torch.randn(2, 16, d_model)
    out, aux_loss = moe(x)

    print("  Input shape:  {}".format(x.shape))
    print("  Output shape: {}".format(out.shape))
    print("  Shapes match: {}".format(x.shape == out.shape))
    print("  Aux loss (load balancing): {:.4f}".format(aux_loss.item()))
    print("  Ideal aux loss = 1.0 (perfectly balanced)")

    total_params = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for p in moe.experts.parameters())
    print("  Total MoE params: {:,}".format(total_params))
    print("  Expert params:    {:,} ({:.1f}% of total)".format(
        expert_params, 100 * expert_params / total_params))


# === Exercise 4: Expert Collapse Detection ===
# Problem: Train a small MoE for a few steps with and without the auxiliary
# load-balancing loss. Observe expert collapse (all tokens route to the same expert)
# when the aux loss is absent.

def exercise_4():
    """Demonstrate expert collapse with and without load-balancing loss."""

    class TinyMoE(nn.Module):
        def __init__(self, d_model=16, num_experts=4, top_k=1):
            super().__init__()
            self.gate = nn.Linear(d_model, num_experts, bias=False)
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
                for _ in range(num_experts)
            ])
            self.num_experts = num_experts
            self.top_k = top_k

        def forward(self, x, use_aux_loss=True):
            T, D = x.shape
            logits = self.gate(x)
            topk_logits, indices = torch.topk(logits, self.top_k, dim=-1)
            weights = F.softmax(topk_logits, dim=-1)

            out = torch.zeros_like(x)
            for e in range(self.num_experts):
                mask = (indices[:, 0] == e)
                if mask.any():
                    out[mask] += weights[mask, 0:1] * self.experts[e](x[mask])

            # Load balancing aux loss (Switch Transformer style)
            if use_aux_loss:
                probs = F.softmax(logits, dim=-1)
                f = torch.zeros(self.num_experts)
                for e in range(self.num_experts):
                    f[e] = (indices[:, 0] == e).float().mean()
                P = probs.mean(dim=0)
                aux_loss = self.num_experts * (f * P).sum()
            else:
                aux_loss = torch.tensor(0.0)

            return out, aux_loss

        def expert_usage(self, x):
            """Return how many tokens each expert receives."""
            logits = self.gate(x)
            _, indices = torch.topk(logits, self.top_k, dim=-1)
            counts = [(indices[:, 0] == e).sum().item() for e in range(self.num_experts)]
            return counts

    torch.manual_seed(0)

    for use_aux in [False, True]:
        model = TinyMoE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(300):
            x = torch.randn(64, 16)
            y_target = torch.randn(64, 16)
            out, aux_loss = model(x, use_aux_loss=use_aux)
            loss = F.mse_loss(out, y_target) + (0.01 * aux_loss if use_aux else 0.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check expert usage after training
        x_eval = torch.randn(256, 16)
        with torch.no_grad():
            usage = model.expert_usage(x_eval)

        label = "WITH" if use_aux else "WITHOUT"
        print("  {} aux loss — expert token counts: {}".format(label, usage))
        max_expert = max(usage)
        min_expert = min(usage)
        collapsed = max_expert > 0.8 * sum(usage)
        print("    Max={}, Min={}, Expert collapse: {}".format(
            max_expert, min_expert, collapsed))


if __name__ == "__main__":
    print("=== Exercise 1: MoE Parameter Efficiency ===")
    exercise_1()
    print("\n=== Exercise 2: Top-k Router Implementation ===")
    exercise_2()
    print("\n=== Exercise 3: Full MoE Layer Forward Pass ===")
    exercise_3()
    print("\n=== Exercise 4: Expert Collapse Detection ===")
    exercise_4()
    print("\nAll exercises completed!")

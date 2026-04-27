[Previous: State Space Models](./46_State_Space_Models.md)

---

# 47. Mixture of Experts

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between sparse and dense models and why sparsity matters
2. Describe the MoE architecture components: router, experts, and gating functions
3. Implement top-k routing with load balancing
4. Understand the Switch Transformer and GShard designs
5. Analyze the Mixtral architecture in detail
6. Identify and address MoE training challenges (expert collapse, load imbalance)
7. Apply auxiliary losses for balanced routing
8. Optimize MoE inference for practical deployment
9. Understand scaling laws specific to MoE models

---

## Table of Contents

1. [Sparse vs Dense Models](#1-sparse-vs-dense-models)
2. [MoE Architecture](#2-moe-architecture)
3. [Top-k Routing and Load Balancing](#3-top-k-routing-and-load-balancing)
4. [Switch Transformer](#4-switch-transformer)
5. [GShard and Expert Parallelism](#5-gshard-and-expert-parallelism)
6. [Mixtral Architecture Deep Dive](#6-mixtral-architecture-deep-dive)
7. [Training Challenges](#7-training-challenges)
8. [Auxiliary Losses for Balanced Routing](#8-auxiliary-losses-for-balanced-routing)
9. [MoE Inference Optimization](#9-moe-inference-optimization)
10. [Scaling Laws for MoE Models](#10-scaling-laws-for-moe-models)
11. [Exercises](#11-exercises)

## 1. Sparse vs Dense Models

### Theory: Sparse vs Dense

A dense feed-forward layer has `~ 8 d_model^2` parameters (the standard `d -> 4d -> d` block) and uses all of them on every token. Cost per token: `O(d_model^2)` FLOPs and `O(d_model^2)` memory.

An MoE replaces the FFN with `E` expert FFNs (each `d -> 4d -> d`, parameter count `8 d^2 each`) plus a router. Per token:

- Router picks top-`k` experts (typically `k = 1` or `2`).
- Token is processed by *only* the chosen experts.
- Outputs are weighted by the router's gate values and summed.

Total parameters: `E * 8 d^2` (huge). Per-token FLOPs: `k * 8 d^2` (small). With `E = 8, k = 2`: 8x more total parameters, 2x more per-token compute. The model has more *capacity* per token at almost the same compute cost.

This is the trick behind giant LLMs at "modest" inference cost: train a model with billions of parameters but activate only a small fraction per token.


### 1.1 The Scaling Dilemma

Larger models perform better, but compute costs grow linearly with parameters:

```
Dense model scaling:

Parameters    FLOPs per token    Quality (PPL)
──────────────────────────────────────────────
  125M          250M               ~30
  350M          700M               ~24
  1.3B          2.6B               ~18.5
  7B            14B                ~12
  70B           140B               ~7
  175B          350B               ~5.5

Problem: FLOPs ∝ Parameters for dense models
Want: more parameters WITHOUT proportional FLOPs increase
```

### 1.2 Sparse Activation

MoE decouples total parameters from per-token compute:

```
Dense model:
  7B params → 7B params activated per token → 14B FLOPs per token

MoE model (8 experts, top-2 routing):
  47B total params → ~12B params activated per token → ~24B FLOPs per token
  but with knowledge of 47B parameters!

Efficiency ratio:
  MoE uses 47B params with ~24B FLOPs
  Dense equivalent quality would need ~13B params with ~26B FLOPs
  → MoE achieves similar quality at similar FLOPs but with more knowledge
```

```
Visualization:

Dense model (all params active):
┌─────────────────────────────────────┐
│█████████████████████████████████████│  ← all params used
└─────────────────────────────────────┘

MoE model (sparse activation):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│█████│     │     │█████│     │     │     │     │  ← only 2/8 experts active
│ E1  │ E2  │ E3  │ E4  │ E5  │ E6  │ E7  │ E8  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ▲                  ▲
  └──── active ──────┘  (router selects per token)
```

---

## 2. MoE Architecture

### 2.1 Core Components

```
MoE Layer (replaces the FFN in a Transformer block):

Input token x ∈ R^d
    │
    ▼
┌─────────┐
│ Router  │──► gating weights g = softmax(W_r · x)
│ (linear)│    g ∈ R^E  (one weight per expert)
└─────────┘
    │
    ▼ select top-k experts
    │
    ├──► Expert 1 (FFN): e_1 = FFN_1(x)  ──► g_1 * e_1
    ├──► Expert 4 (FFN): e_4 = FFN_4(x)  ──► g_4 * e_4
    │
    ▼ weighted sum
    │
Output y = Σ_{i ∈ top-k} g_i * FFN_i(x)
```

### 2.2 Basic Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert (standard FFN)."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # For SwiGLU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU activation (same as Llama FFN)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Router(nn.Module):
    """Token-choice router: each token selects its top-k experts."""

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch * seq_len, d_model)
        Returns:
            top_k_indices: (batch * seq_len, top_k)
            top_k_weights: (batch * seq_len, top_k) — normalized
        """
        logits = self.gate(x)  # (tokens, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_indices, top_k_weights, logits


class MoELayer(nn.Module):
    """Mixture of Experts layer."""

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
            aux_loss: load balancing loss
        """
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*L, D)

        # Route tokens to experts
        top_k_indices, top_k_weights, router_logits = self.router(x_flat)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*L,)
            expert_weights = top_k_weights[:, k]   # (B*L,)

            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        # Compute auxiliary loss for load balancing
        aux_loss = self.load_balancing_loss(router_logits, top_k_indices)

        return output.reshape(B, L, D), aux_loss

    def load_balancing_loss(self, router_logits, top_k_indices):
        """Compute load balancing auxiliary loss."""
        num_tokens = router_logits.shape[0]

        # Fraction of tokens routed to each expert
        # Count how many times each expert is selected
        counts = torch.zeros(self.num_experts, device=router_logits.device)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                counts[e] += (top_k_indices[:, k] == e).float().sum()
        f = counts / (num_tokens * self.top_k)  # (num_experts,)

        # Average routing probability for each expert
        probs = F.softmax(router_logits, dim=-1)
        p = probs.mean(dim=0)  # (num_experts,)

        # Auxiliary loss: dot product of f and p
        # Minimized when both are uniform (1/num_experts)
        aux_loss = self.num_experts * (f * p).sum()
        return aux_loss
```

---

## 3. Top-k Routing and Load Balancing

### Theory: Top-k Routing

The router is a small linear layer mapping each token's representation to a logit per expert:

```
logits = u @ W_gate         # (N, E)
weights = softmax(logits)   # (N, E), but most values are wastefully small
```

Then for top-k routing:

1. For each token, find the `k` experts with the largest logits.
2. Compute softmax over only those k logits to get gate weights `g_1, ..., g_k`.
3. Process the token through each chosen expert.
4. Output = `sum_i g_i * expert_i(token)`.

Implementation challenge: routing creates *imbalanced* loads on experts (some get many tokens, some get few), which destroys GPU efficiency (you cannot fill an expert's batch). This is what load balancing addresses.


### 3.1 Routing Strategies

```
Strategy          k    Description                  Used In
──────────────────────────────────────────────────────────────
Top-1 (Switch)    1    Each token → 1 expert        Switch Transformer
Top-2             2    Each token → 2 experts        Mixtral, GShard
Expert Choice     -    Each expert picks top-T       Expert Choice (Zhou 2022)
                       tokens from the batch
Soft MoE          -    Soft assignment via           Soft MoE (Puigcerver 2024)
                       weighted average of all
```

### 3.2 The Load Balancing Problem

Without balancing, routing degenerates:

```
Ideal routing (uniform):
  Expert 1: 12.5% of tokens  ✓
  Expert 2: 12.5% of tokens  ✓
  ...
  Expert 8: 12.5% of tokens  ✓

Degenerate routing (expert collapse):
  Expert 1: 90% of tokens   ← overloaded, bottleneck
  Expert 2: 5% of tokens
  Expert 3: 3% of tokens
  Expert 4-8: ~0.4% each    ← undertrained, wasted capacity

Why this happens:
  1. Expert 1 gets slightly more tokens early in training
  2. More tokens → more gradient updates → Expert 1 improves
  3. Router sends even more tokens to Expert 1
  4. Positive feedback loop → collapse
```

### 3.3 Capacity Factor

```
Capacity factor C controls the maximum tokens per expert:

  Expert capacity = C * (total_tokens / num_experts)

  C = 1.0: each expert can handle exactly its fair share
  C = 1.25: 25% buffer (recommended)
  C = 2.0: generous buffer (more memory)

If an expert exceeds capacity:
  - Overflow tokens are passed through a residual connection (skip expert)
  - Or overflow tokens are dropped (Switch Transformer training)
```

```python
class CapacityRouter(nn.Module):
    """Router with capacity factor to prevent overload."""

    def __init__(self, d_model, num_experts, top_k=1, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (B*L, D)
        Returns: dispatch_mask, combine_weights for token-expert assignment
        """
        num_tokens = x.shape[0]
        logits = self.gate(x)  # (num_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Capacity limit
        expert_capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)

        # Count tokens per expert
        dispatch_mask = torch.zeros(num_tokens, self.num_experts, dtype=torch.bool,
                                     device=x.device)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        for k in range(self.top_k):
            for token_idx in range(num_tokens):
                expert_idx = top_k_indices[token_idx, k].item()
                if expert_counts[expert_idx] < expert_capacity:
                    dispatch_mask[token_idx, expert_idx] = True
                    expert_counts[expert_idx] += 1
                # else: token overflows, handled by residual

        return dispatch_mask, top_k_probs, logits
```

---

## 4. Switch Transformer

### Theory: Switch Transformer, GShard, Mixtral

**Switch Transformer** (Fedus et al. 2021): top-1 routing — each token goes to *one* expert only. Simpler, often works as well as top-2 with the right load balancing. Scaled to T5-like models with 1.6T parameters.

**GShard** (Lepikhin et al. 2020): top-2 routing with extensive expert-parallelism (experts on different devices). Made MoE practical at scale.

**Mixtral 8x7B** (Mistral 2023): an open-weight MoE LLM with 8 experts of 7B each, top-2 routing. Total ~47B parameters but only ~13B activated per token. Quality matches or exceeds 70B dense models at similar inference cost. Demonstrated MoE is now a real alternative to dense scaling for production LLMs.

The current consensus: dense models are simpler and easier to fine-tune; MoE models are more parameter-efficient but harder to train and serve. The trade-off depends on whether you optimize for training compute (favor MoE) or deployment simplicity (favor dense).


### 4.1 Design Philosophy

Fedus et al. (2022) simplified MoE to top-1 routing:

```
Key decisions:
  1. Route each token to exactly 1 expert (k=1)
     → simpler, less communication, lower FLOPs per token
  2. Use capacity factor of 1.0-1.5
  3. Scale to 1.6T parameters with 2048 experts
  4. Use bfloat16 for expert computation (float32 for router)

Result: 4-7× speedup over dense T5 at equivalent quality
```

### 4.2 Switch Transformer Architecture

```
Standard Transformer Block:          Switch Transformer Block:
┌───────────────────────┐            ┌───────────────────────┐
│   Self-Attention      │            │   Self-Attention      │
│   + Residual + Norm   │            │   + Residual + Norm   │
├───────────────────────┤            ├───────────────────────┤
│   FFN (dense)         │            │   Switch Layer (MoE)  │
│   + Residual + Norm   │            │   + Residual + Norm   │
└───────────────────────┘            └───────────────────────┘

Only the FFN is replaced → attention is shared (not expert-ified)
```

### 4.3 Scaling Results

```
Model             Total Params   Active Params   Speedup vs Dense T5
──────────────────────────────────────────────────────────────────────
Switch-Base       7.4B           0.2B            7× faster to quality
Switch-Large      26.3B          0.7B            7× faster to quality
Switch-XXL        395B           11B             4× faster to quality
Switch-C          1.6T           ~13B            4× faster to quality
```

---

## 5. GShard and Expert Parallelism

### 5.1 Expert Parallelism

When the model is too large for one device, MoE enables a natural parallelism strategy:

```
Data Parallelism:           Model (Tensor) Parallelism:     Expert Parallelism:
┌────────┐  ┌────────┐     ┌────────┬────────┐             ┌────────┐  ┌────────┐
│ GPU 0  │  │ GPU 1  │     │ GPU 0  │ GPU 1  │             │ GPU 0  │  │ GPU 1  │
│ Full   │  │ Full   │     │ Half   │ Half   │             │ Attn   │  │ Attn   │
│ Model  │  │ Model  │     │ Model  │ Model  │             │ E1-E4  │  │ E5-E8  │
│ Data/2 │  │ Data/2 │     │ Full   │ Full   │             │ (same  │  │ (same  │
│        │  │        │     │ Data   │ Data   │             │  data) │  │  data) │
└────────┘  └────────┘     └────────┴────────┘             └────────┘  └────────┘

Expert Parallelism:
  - Each GPU holds a subset of experts
  - All-to-all communication to route tokens to correct GPU
  - Attention layers are replicated on all GPUs
  - Naturally balanced if routing is balanced
```

### 5.2 All-to-All Communication

```
Routing with 4 GPUs, 8 experts (2 per GPU):

Before all-to-all:           After all-to-all:
GPU 0: tokens for E0-E7      GPU 0: all tokens for E0, E1
GPU 1: tokens for E0-E7      GPU 1: all tokens for E2, E3
GPU 2: tokens for E0-E7      GPU 2: all tokens for E4, E5
GPU 3: tokens for E0-E7      GPU 3: all tokens for E6, E7

Each GPU computes its local experts, then all-to-all back
```

### 5.3 GShard Design

```
GShard (Lepikhin et al., 2021):
  - 600B parameter MoE for machine translation
  - 2048 experts across 2048 TPU cores
  - Top-2 routing with auxiliary loss
  - Each expert is a standard FFN
  - Random routing for second expert (during training)
  - Achieved SOTA on 100+ language pairs
```

---

## 6. Mixtral Architecture Deep Dive

### 6.1 Architecture

Mixtral (Mistral AI, 2024) — one of the most successful open MoE models:

```
Mixtral 8x7B:
  Total parameters:    46.7B
  Active parameters:   12.9B per token
  Experts:             8 per layer
  Top-k:               2
  Hidden dim:          4096
  FFN dim:             14336
  Layers:              32
  Attention heads:     32
  KV heads:            8 (GQA)
  Context length:      32K
  Sliding window:      4096 (for some layers)

Comparison:
  Mixtral 8x7B (~13B active)  ≈  Llama 2 70B quality
  At 5× fewer active parameters!
```

### 6.2 Mixtral Block

```python
class MixtralBlock(nn.Module):
    """One Mixtral transformer block."""

    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8,
                 d_ff=14336, num_experts=8, top_k=2):
        super().__init__()
        # Attention (shared, not MoE)
        self.attn_norm = nn.RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)

        # MoE FFN
        self.ffn_norm = nn.RMSNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)

    def forward(self, x, mask=None):
        # Attention with residual
        h = self.attn_norm(x)
        h = self.attention(h, mask=mask)
        x = x + h

        # MoE FFN with residual
        h = self.ffn_norm(x)
        h, aux_loss = self.moe(h)
        x = x + h

        return x, aux_loss


class GroupedQueryAttention(nn.Module):
    """GQA: multiple query heads share KV heads."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # How many Q heads per KV head
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads for each query group
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)
```

### 6.3 Expert Specialization in Mixtral

Analysis of trained Mixtral models reveals expert specialization:

```
Expert routing patterns (observed in Mixtral 8x7B):

Layer 0-5 (early layers):
  - Experts specialize by token type (punctuation, numbers, words)
  - Relatively uniform routing

Layer 6-20 (middle layers):
  - Experts specialize by domain/topic
  - Expert 2: math and code tokens
  - Expert 5: natural language, narrative
  - Expert 7: multilingual tokens

Layer 21-31 (late layers):
  - More uniform routing again
  - Experts specialize by output pattern rather than input
```

---

## 7. Training Challenges

### 7.1 Expert Collapse

```
Expert collapse: router learns to always select the same expert(s)

Symptoms:
  - One or two experts receive >80% of tokens
  - Other experts are rarely activated
  - Effectively reduces to a dense model
  - Wasted parameters and compute

Causes:
  1. Positive feedback loop (more tokens → better expert → more tokens)
  2. Router overfitting early in training
  3. Insufficient exploration of expert assignments
```

### 7.2 Training Instability

```
MoE models are more prone to training instabilities:

Issue                  Cause                          Mitigation
────────────────────────────────────────────────────────────────────
Loss spikes            Discrete routing decisions      Router z-loss
Expert collapse        Positive feedback loop          Auxiliary loss
Gradient noise         Different experts per batch     Larger batch size
Overflow               Router logits too large         Float32 router
Dead experts           Never selected after init       Expert reset
```

### 7.3 Expert Dropout and Reset

```python
class MoEWithExpertReset(nn.Module):
    """MoE layer with expert monitoring and reset."""

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2,
                 reset_threshold=0.01):
        super().__init__()
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.reset_threshold = reset_threshold
        self.expert_usage = torch.zeros(num_experts)
        self.step_count = 0

    def check_and_reset_dead_experts(self):
        """Reset experts that receive too few tokens."""
        if self.step_count < 1000:
            return  # Wait for initial training

        avg_usage = self.expert_usage / self.step_count
        dead_mask = avg_usage < self.reset_threshold

        if dead_mask.any():
            # Find the most-used expert
            best_expert = avg_usage.argmax().item()

            for dead_idx in dead_mask.nonzero().squeeze(-1):
                dead_idx = dead_idx.item()
                print(f"Resetting dead expert {dead_idx} from expert {best_expert}")

                # Copy weights from best expert + noise
                with torch.no_grad():
                    for dead_p, best_p in zip(
                        self.moe.experts[dead_idx].parameters(),
                        self.moe.experts[best_expert].parameters()
                    ):
                        dead_p.copy_(best_p + 0.01 * torch.randn_like(best_p))

            # Reset counters
            self.expert_usage.zero_()
            self.step_count = 0
```

---

## 8. Auxiliary Losses for Balanced Routing

### Theory: Load Balancing and Auxiliary Loss

If the router consistently sends most tokens to a few "popular" experts, the others are wasted parameters and the popular ones become a bottleneck. Two mechanisms force balance:

1. **Capacity factor**: each expert can process at most `C = capacity_factor * (N / E)` tokens per batch. Tokens routed to a full expert are dropped (or routed to a fallback).
2. **Auxiliary load-balancing loss**:
   ```
   L_aux = E * sum_i (f_i * P_i)
   ```
   where `f_i` is the fraction of tokens routed to expert `i` (after dispatch) and `P_i` is the average router probability for expert `i` (before dispatch). This penalizes consistently lopsided routing. Coefficient is small (~0.01) but consistently applied during training.

A well-trained MoE has near-uniform expert utilization and only a small fraction of dropped tokens.


### 8.1 Load Balancing Loss

The standard auxiliary loss from Switch Transformer:

```
Load Balancing Loss:

L_balance = α * N * Σ_i (f_i * p_i)

where:
  N = number of experts
  f_i = fraction of tokens dispatched to expert i
  p_i = mean routing probability for expert i
  α = coefficient (typically 0.01 - 0.1)

When routing is perfectly balanced:
  f_i = 1/N for all i
  p_i = 1/N for all i
  L_balance = α * N * N * (1/N)² = α

When routing is collapsed (all to expert 0):
  f_0 = 1, p_0 ≈ 1
  L_balance ≈ α * N  (much larger)
```

### 8.2 Router Z-Loss

Prevents router logits from becoming too large (stabilizes training):

```
Router z-loss:

L_z = β * (1/T) * Σ_t (log Σ_i exp(r_t,i))²

where:
  r_t,i = router logit for token t, expert i
  β = coefficient (typically 0.001)

This penalizes large logits, keeping the routing "soft"
→ prevents the router from becoming too confident
→ reduces training instability
```

```python
def router_z_loss(router_logits, coefficient=0.001):
    """
    Router z-loss for training stability.
    router_logits: (num_tokens, num_experts)
    """
    # Log-sum-exp of router logits
    log_z = torch.logsumexp(router_logits, dim=-1)  # (num_tokens,)
    z_loss = coefficient * (log_z ** 2).mean()
    return z_loss


def combined_moe_loss(main_loss, router_logits, top_k_indices, num_experts,
                       balance_coef=0.01, z_coef=0.001):
    """Combine main loss with MoE auxiliary losses."""
    # Load balancing loss
    num_tokens = router_logits.shape[0]
    probs = F.softmax(router_logits, dim=-1)
    p = probs.mean(dim=0)  # Average probability per expert

    counts = torch.zeros(num_experts, device=router_logits.device)
    for k in range(top_k_indices.shape[1]):
        one_hot = F.one_hot(top_k_indices[:, k], num_experts).float()
        counts += one_hot.sum(dim=0)
    f = counts / counts.sum()

    balance_loss = balance_coef * num_experts * (f * p).sum()

    # Z-loss
    z_loss = router_z_loss(router_logits, z_coef)

    return main_loss + balance_loss + z_loss
```

### 8.3 Expert-Choice Routing

An alternative approach where experts choose tokens (instead of tokens choosing experts):

```
Token-choice routing:        Expert-choice routing:
  Each token picks top-k     Each expert picks top-T tokens
  experts from E options     from the batch

Advantage: perfectly balanced by construction!
  Every expert processes exactly T tokens.

Disadvantage:
  - Tokens may not be selected by any expert (dropped)
  - Tokens may be selected by many experts (overrepresented)
  - Harder to parallelize across sequence dimension
```

```python
class ExpertChoiceRouter(nn.Module):
    """Expert-choice routing: each expert selects its tokens."""

    def __init__(self, d_model, num_experts, capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (B*L, D)
        Each expert selects top-T tokens based on its routing scores.
        """
        num_tokens = x.shape[0]
        T = int(self.capacity_factor * num_tokens / self.num_experts)

        logits = self.gate(x)  # (num_tokens, num_experts)
        scores = F.softmax(logits, dim=0)  # Softmax over tokens (not experts!)

        # Each expert selects its top-T tokens
        top_T_scores, top_T_indices = torch.topk(scores.T, T, dim=-1)
        # top_T_indices: (num_experts, T)
        # top_T_scores: (num_experts, T)

        return top_T_indices, top_T_scores, logits
```

---

## 9. MoE Inference Optimization

### 9.1 Inference Challenges

```
MoE inference challenges:

1. Memory: All experts must be in memory
   8 experts × 7B each = 56B parameters = ~112GB in float16
   vs dense 13B model = ~26GB

2. Bandwidth: Only 2/8 experts active, but all must be loadable
   If experts are on different GPUs: all-to-all communication

3. Batch efficiency: Different tokens route to different experts
   → irregular computation patterns
   → poor GPU utilization at small batch sizes
```

### 9.2 Expert Offloading

```python
class OffloadedMoE(nn.Module):
    """MoE with expert offloading for memory efficiency."""

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2,
                 max_gpu_experts=2):
        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.max_gpu_experts = max_gpu_experts
        self.num_experts = num_experts

        # Move non-active experts to CPU
        for i in range(max_gpu_experts, num_experts):
            self.experts[i] = self.experts[i].cpu()

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        device = x.device

        # Route
        top_k_indices, top_k_weights, logits = self.router(x_flat)
        needed_experts = top_k_indices.unique().tolist()

        # Prefetch needed experts to GPU
        for e_idx in needed_experts:
            if next(self.experts[e_idx].parameters()).device != device:
                self.experts[e_idx] = self.experts[e_idx].to(device)

        # Compute (same as standard MoE)
        output = torch.zeros_like(x_flat)
        for k in range(self.router.top_k):
            for e in needed_experts:
                mask = (top_k_indices[:, k] == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    output[mask] += top_k_weights[mask, k:k+1] * expert_out

        # Optionally offload unused experts back to CPU
        for e_idx in range(self.num_experts):
            if e_idx not in needed_experts:
                self.experts[e_idx] = self.experts[e_idx].cpu()

        return output.reshape(B, L, D)
```

### 9.3 Optimization Techniques

```
Technique                  Speedup    Memory Savings    Description
──────────────────────────────────────────────────────────────────────
Expert offloading          1×         50-75%            Keep inactive on CPU
Expert quantization        1.5-2×     50-75%            INT4/INT8 experts
Expert pruning             1.5×       25-50%            Remove least-used experts
Speculative routing        1.2×       0%                Predict routing ahead
Expert merging             1×         50%               Merge similar experts post-train
Kernel fusion              1.5×       0%                Fuse router + dispatch + expert
Dynamic batching           2-3×       0%                Group same-expert tokens
```

---

## 10. Scaling Laws for MoE Models

### 10.1 MoE Scaling Behavior

```
MoE scaling differs from dense model scaling:

Dense scaling law (Chinchilla):
  L(N, D) = A/N^α + B/D^β + E
  where N = parameters, D = training tokens

MoE scaling:
  L(N_total, N_active, E, D) = f(N_total, N_active, E, D)
  where E = number of experts, N_active = active params per token

Key findings:
  1. Doubling experts (with fixed active params) improves loss
     but with diminishing returns
  2. 8-16 experts is a sweet spot for most scales
  3. Very large expert counts (>256) show minimal gains
  4. Active parameter count matters more than total for final quality
```

### 10.2 Granularity

```
Expert granularity:

"Fine-grained" MoE: many small experts (e.g., 64 × small FFN)
"Coarse-grained" MoE: few large experts (e.g., 8 × large FFN)

Research finding: at fixed total params and FLOPs,
more fine-grained experts tend to perform better
(up to a point — communication overhead increases)

Example (same total/active params):
  8 experts, top-2:   loss = 2.85
  16 experts, top-4:  loss = 2.80
  32 experts, top-8:  loss = 2.77
  64 experts, top-16: loss = 2.76   ← diminishing returns
  128 experts, top-32: loss = 2.76  ← no further gain
```

### 10.3 Practical Recommendations

```
MoE Design Guidelines:

Parameter                  Recommendation
────────────────────────────────────────────────────────
Number of experts          8-16 for most use cases
Top-k                      2 (good quality/efficiency balance)
Auxiliary loss weight       0.01-0.1 (tune carefully)
Router z-loss weight       0.001
Capacity factor            1.0-1.5
Expert size                Same as dense FFN
Which layers               Every layer or every other layer
Training batch size        2-4× larger than dense equivalent
Learning rate              Same as dense equivalent
```

---

## 11. Exercises

### Exercise 1: Basic MoE Layer

Implement a Mixture of Experts layer from scratch:

```python
"""
Exercise 1: Implement MoE and verify expert utilization.

Tasks:
1. Implement the MoELayer class with top-2 routing
2. Create a simple sequence classification task (e.g., MNIST sequences)
3. Train the MoE model and log:
   - Expert utilization per layer (what fraction of tokens each expert gets)
   - Load balancing loss over time
   - Whether any experts collapse
4. Visualize routing patterns: which experts handle which types of inputs?

Starter code:
"""

class SimpleMoEClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, num_classes=10,
                 num_experts=8, top_k=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.moe1 = MoELayer(d_model, d_model * 4, num_experts, top_k)
        self.moe2 = MoELayer(d_model, d_model * 4, num_experts, top_k)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x, loss1 = self.moe1(x)
        x = F.gelu(x)
        x, loss2 = self.moe2(x)
        x = x.mean(dim=1)  # Pool over sequence
        return self.classifier(x), loss1 + loss2

# TODO: Train and analyze expert utilization
```

### Exercise 2: Load Balancing

Experiment with different auxiliary losses and their effect on expert balance:

```python
"""
Exercise 2: Auxiliary loss comparison.

Tasks:
1. Train MoE models with different auxiliary losses:
   a. No auxiliary loss (baseline — expect collapse)
   b. Load balancing loss only (α = 0.01)
   c. Load balancing + z-loss (α = 0.01, β = 0.001)
   d. Load balancing with high coefficient (α = 0.1)

2. For each, track:
   - Expert utilization histogram over training
   - Gini coefficient of expert usage (0 = perfect balance, 1 = collapse)
   - Final model quality (loss/accuracy)

3. Find the best auxiliary loss configuration

Expected: (b) or (c) should work best; (a) collapses; (d) hurts quality
"""

def gini_coefficient(usage_counts):
    """Compute Gini coefficient of expert utilization."""
    sorted_counts = torch.sort(usage_counts)[0]
    n = len(sorted_counts)
    cumsum = torch.cumsum(sorted_counts, dim=0)
    gini = (2 * torch.arange(1, n+1, dtype=torch.float32).dot(sorted_counts)
            - (n + 1) * sorted_counts.sum()) / (n * sorted_counts.sum())
    return gini.item()

# TODO: Implement comparison
```

### Exercise 3: Expert Specialization Analysis

Analyze what each expert has learned:

```python
"""
Exercise 3: Expert specialization on a text dataset.

Tasks:
1. Train a small MoE language model on a diverse text corpus
2. After training, run inference and record:
   - Which expert is selected for each token
   - Token type (word, number, punctuation, code, etc.)
3. Create a heatmap: experts × token types
4. Analyze: do experts specialize? How?

Bonus:
- Visualize expert specialization across layers
- Compare early vs late layer routing patterns
"""

def analyze_expert_routing(model, tokenizer, texts):
    """Analyze which experts handle which types of tokens."""
    expert_token_counts = {}  # expert_idx -> {token_type: count}

    for text in texts:
        tokens = tokenizer.encode(text)
        # TODO: Run through model, record routing decisions
        # TODO: Classify each token type and count expert assignments
        pass

    return expert_token_counts

# TODO: Implement and visualize
```

### Exercise 4: MoE vs Dense Comparison

Compare MoE and dense models at equivalent compute:

```python
"""
Exercise 4: MoE vs Dense model comparison.

Tasks:
1. Create three models with approximately equal FLOPs per token:
   a. Dense model: 12M parameters
   b. MoE-8E model: 48M total, ~12M active (8 experts, top-2)
   c. MoE-16E model: 96M total, ~12M active (16 experts, top-4)

2. Train all three on the same dataset for the same number of tokens
3. Compare:
   - Final training loss
   - Validation loss
   - Training wall-clock time
   - Memory usage

4. Plot learning curves for all three models

Expected:
  - MoE models converge faster (more total knowledge)
  - MoE-8E slightly better than dense at same FLOPs
  - MoE-16E marginally better than MoE-8E
  - MoE uses more memory
"""

def create_matched_models(d_model=256, d_ff=512, n_layers=6, vocab_size=10000):
    """Create dense and MoE models matched on FLOPs."""

    # Dense model
    dense = TransformerLM(vocab_size, d_model, d_ff, n_layers)

    # MoE-8E: replace FFN with MoE, same FFN size per expert
    moe_8 = MoETransformerLM(vocab_size, d_model, d_ff, n_layers,
                              num_experts=8, top_k=2)

    # MoE-16E: more experts, same total active params
    moe_16 = MoETransformerLM(vocab_size, d_model, d_ff // 2, n_layers,
                               num_experts=16, top_k=4)

    return dense, moe_8, moe_16

# TODO: Train and compare
```

---

**Previous**: [State Space Models](./46_State_Space_Models.md)

---

*End of Lesson 47*

# Mistral & Mixture of Experts

## Learning Objectives
- Understand the architectural features of Mistral 7B
- Grasp Mixture of Experts (MoE) concepts and operation principles
- Learn the Mixtral 8x7B structure
- Master the pros/cons and practical applications of Sparse MoE

---

## 1. Mistral 7B Overview

### 1.1 Mistral's Innovation

**Mistral 7B** is a model released by Mistral AI in 2023, achieving 13B-level performance with only 7B parameters.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mistral 7B Features                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Performance Comparison (as of 2023.10):                        │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Model          │ Params │ MMLU  │ HellaSwag │ GSM8K │      │
│  │  ───────────────│────────│───────│───────────│───────│      │
│  │  LLaMA 2 7B     │ 7B     │ 45.3  │ 77.2      │ 14.6  │      │
│  │  LLaMA 2 13B    │ 13B    │ 54.8  │ 80.7      │ 28.7  │      │
│  │  Mistral 7B     │ 7B     │ 60.1  │ 81.3      │ 52.2  │ ←!   │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  Key Technologies:                                               │
│  • Sliding Window Attention (SWA)                               │
│  • Grouped Query Attention (GQA)                                │
│  • Over-training with more data                                 │
│  • Flash Attention 2 optimization                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Mistral Architecture Specifications

```python
MISTRAL_CONFIGS = {
    "mistral-7b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,           # GQA
        "head_dim": 128,
        "hidden_dim": 14336,
        "vocab_size": 32000,
        "context_length": 32768,   # Technical limit
        "sliding_window": 4096,    # Sliding Window Attention
        "rope_theta": 10000.0,
    },
}

# Comparison with LLaMA 2 7B
LLAMA2_7B = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 32,              # MHA (no GQA)
    "hidden_dim": 11008,
    "context_length": 4096,
    "sliding_window": None,        # Full attention
}
```

---

## 2. Sliding Window Attention (SWA)

### 2.1 Concept

**Sliding Window Attention** restricts each token to attend only to tokens within a fixed window.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sliding Window Attention                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full Attention (Traditional):                                  │
│  ────────────────────────                                       │
│  Every token attends to all previous tokens                     │
│  Complexity: O(n²)                                              │
│                                                                 │
│  Position:  1  2  3  4  5  6  7  8  9  10                       │
│  Token 10:  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓                       │
│                                                                 │
│  Sliding Window (W=4):                                          │
│  ────────────────────────                                       │
│  Only attend to tokens within window size W                     │
│  Complexity: O(n × W)                                           │
│                                                                 │
│  Position:  1  2  3  4  5  6  7  8  9  10                       │
│  Token 10:  ✗  ✗  ✗  ✗  ✗  ✗  ✓  ✓  ✓  ✓                       │
│                         ↑     └───────┬───────┘                 │
│                    Window start       Window (W=4)              │
│                                                                 │
│  Layer Stacking Effect:                                         │
│  ────────────────────────                                       │
│  L layers → Effective receptive field = L × W                   │
│  32 layers × 4096 window = 131,072 token range!                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation

```python
import torch
import torch.nn.functional as F
import math

def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int = 4096,
    causal: bool = True,
):
    """
    Sliding Window Attention Implementation

    Args:
        query: (batch, n_heads, seq_len, head_dim)
        key: (batch, n_heads, seq_len, head_dim)
        value: (batch, n_heads, seq_len, head_dim)
        window_size: Window size
        causal: Whether to apply causal masking
    """
    batch, n_heads, seq_len, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Create sliding window mask
    # Each position i attends from max(0, i-W+1) to i
    row_idx = torch.arange(seq_len).unsqueeze(1)  # (seq, 1)
    col_idx = torch.arange(seq_len).unsqueeze(0)  # (1, seq)

    # Causal: col <= row
    # Window: col >= row - window_size + 1
    if causal:
        mask = (col_idx <= row_idx) & (col_idx >= row_idx - window_size + 1)
    else:
        mask = torch.abs(row_idx - col_idx) < window_size

    # Apply mask
    mask = mask.to(scores.device)
    scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax & output
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output

# Memory comparison
def compare_attention_memory(seq_len, window_size=4096):
    """Full vs Sliding Window memory comparison"""
    full_attention_mem = seq_len * seq_len  # O(n²)
    sliding_window_mem = seq_len * window_size  # O(n × W)

    print(f"Sequence length: {seq_len:,}")
    print(f"Full Attention: {full_attention_mem:,} elements")
    print(f"Sliding Window: {sliding_window_mem:,} elements")
    print(f"Memory savings: {(1 - sliding_window_mem/full_attention_mem)*100:.1f}%")

compare_attention_memory(32768, 4096)
# Sequence length: 32,768
# Full Attention: 1,073,741,824 elements
# Sliding Window: 134,217,728 elements
# Memory savings: 87.5%
```

### 2.3 Rolling Buffer KV Cache

```python
"""
Rolling Buffer: Process long sequences with fixed-size KV cache

Normal KV Cache:
- Store KV for all tokens
- Memory: O(seq_len)

Rolling Buffer:
- Store only window_size tokens
- Overwrite old KV
- Memory: O(window_size) = constant!

Example (window=4):
Step 1: [K1, K2, K3, K4]
Step 2: [K5, K2, K3, K4]  ← K5 stored at K1 position
Step 3: [K5, K6, K3, K4]  ← K6 stored at K2 position
...

Advantages:
- Can process infinite sequences (fixed memory)
- Constant inference speed

Disadvantages:
- Loses old information
- Compensated by layer stacking
"""

class RollingKVCache:
    def __init__(self, window_size: int, n_layers: int, n_kv_heads: int, head_dim: int):
        self.window_size = window_size
        self.cache_k = torch.zeros(n_layers, 1, window_size, n_kv_heads, head_dim)
        self.cache_v = torch.zeros(n_layers, 1, window_size, n_kv_heads, head_dim)
        self.pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Add new KV to cache (circular buffer)"""
        seq_len = k.shape[1]
        for i in range(seq_len):
            idx = (self.pos + i) % self.window_size
            self.cache_k[layer_idx, :, idx] = k[:, i]
            self.cache_v[layer_idx, :, idx] = v[:, i]
        self.pos = (self.pos + seq_len) % self.window_size

    def get(self, layer_idx: int):
        return self.cache_k[layer_idx], self.cache_v[layer_idx]
```

---

## 3. Mixture of Experts (MoE) Basics

### 3.1 MoE Concept

**Mixture of Experts** is an architecture that improves efficiency by activating only some "expert" networks among many.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mixture of Experts Concept                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dense Model:                                                   │
│  ─────────────────                                              │
│  Input ──► [FFN (entire)] ──► Output                            │
│  • All parameters activated every time                          │
│  • Computation = proportional to parameter count                │
│                                                                 │
│  Sparse MoE Model:                                              │
│  ─────────────────                                              │
│                        ┌──► Expert 1 ──┐                        │
│                        │               │                        │
│  Input ──► Router ─────┼──► Expert 2 ──┼──► Combine ──► Output  │
│              ↓         │               │                        │
│         (Top-K select) └──► Expert 3 ──┘                        │
│                        └──► Expert N (inactive)                 │
│                                                                 │
│  • Router selects only K experts                                │
│  • Many parameters, little computation                          │
│  • Example: 8 experts, 2 activated → 1/4 computation            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Router (Gating Network)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """
    Top-K Router: Select K experts for each input

    Formula:
    G(x) = softmax(TopK(x · W_g))

    Where TopK keeps only top K values, rest set to -inf
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            router_probs: (batch, seq_len, top_k) - Selected expert weights
            expert_indices: (batch, seq_len, top_k) - Selected expert indices
        """
        # Compute router logits
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax (among selected experts)
        router_probs = F.softmax(top_k_logits, dim=-1)

        return router_probs, top_k_indices

# Example
router = TopKRouter(dim=4096, num_experts=8, top_k=2)
x = torch.randn(2, 10, 4096)  # batch=2, seq=10
probs, indices = router(x)
print(f"Router probs shape: {probs.shape}")    # (2, 10, 2)
print(f"Expert indices shape: {indices.shape}")  # (2, 10, 2)
print(f"Selected experts for token 0: {indices[0, 0]}")  # e.g., tensor([3, 7])
```

### 3.3 Expert Layer

```python
class MoELayer(nn.Module):
    """
    Mixture of Experts Layer

    Each token is routed to Top-K experts for processing
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(dim, num_experts, top_k)

        # Experts (each is an independent FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            output: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # Routing
        router_probs, expert_indices = self.router(x)
        # router_probs: (batch, seq_len, top_k)
        # expert_indices: (batch, seq_len, top_k)

        # Initialize output
        output = torch.zeros_like(x)

        # Process by each expert (simple implementation, actually more optimized)
        for k in range(self.top_k):
            expert_idx = expert_indices[:, :, k]  # (batch, seq_len)
            expert_prob = router_probs[:, :, k:k+1]  # (batch, seq_len, 1)

            for e in range(self.num_experts):
                # Find positions where this expert is selected
                mask = (expert_idx == e)
                if mask.any():
                    # Extract relevant tokens
                    selected = x[mask]  # (num_selected, dim)
                    # Apply expert
                    expert_output = self.experts[e](selected)
                    # Add weighted result to output
                    output[mask] += expert_prob[mask].squeeze(-1).unsqueeze(-1) * expert_output

        return output

# Usage example
moe = MoELayer(dim=4096, hidden_dim=14336, num_experts=8, top_k=2)
x = torch.randn(2, 10, 4096)
output = moe(x)
print(f"Output shape: {output.shape}")  # (2, 10, 4096)
```

---

## 4. Mixtral 8x7B

### 4.1 Architecture

**Mixtral 8x7B** is an MoE model with 8 experts, activating only 2 experts per layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mixtral 8x7B Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Transformer Block                      │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              Attention (GQA)                     │    │    │
│  │  │  • 32 query heads, 8 KV heads                   │    │    │
│  │  │  • Sliding Window (4096)                        │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                        │                                │    │
│  │                        ▼                                │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │         Sparse MoE FFN Layer                    │    │    │
│  │  │  ┌─────────────────────────────────────────┐    │    │    │
│  │  │  │              Router                      │    │    │    │
│  │  │  │         (Select Top-2)                   │    │    │    │
│  │  │  └────┬────┬────┬────┬────┬────┬────┬────┬─┘    │    │    │
│  │  │       │    │    │    │    │    │    │    │      │    │    │
│  │  │       ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼      │    │    │
│  │  │     [E1] [E2] [E3] [E4] [E5] [E6] [E7] [E8]     │    │    │
│  │  │      ✓         ✓                               │    │    │
│  │  │   Selected  Selected  Inactive  Inactive ...   │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Total Parameters: ~46.7B (8 experts × 7B FFN params)           │
│  Active Parameters: ~12.9B (2/8 experts)                        │
│  Inference Speed: Similar to 12.9B dense model                  │
│  Performance: 70B dense model level                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Mixtral Specifications

```python
MIXTRAL_CONFIG = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "hidden_dim": 14336,
    "vocab_size": 32000,

    # MoE settings
    "num_experts": 8,
    "num_experts_per_tok": 2,  # Top-K

    # Attention
    "sliding_window": 4096,
    "context_length": 32768,

    # Parameter calculation
    # Attention: 4 × dim² × n_layers = 4 × 4096² × 32 ≈ 2.1B
    # MoE FFN: 8 × 3 × dim × hidden × n_layers = 8 × 3 × 4096 × 14336 × 32 ≈ 44.6B
    # Total: ~46.7B
    # Active: ~12.9B (attention + 2/8 FFN)
}
```

### 4.3 Load Balancing Loss

One key challenge of MoE is the **expert imbalance** problem.

```python
def load_balancing_loss(router_probs, expert_indices, num_experts):
    """
    Load Balancing Loss: Encourage balanced expert usage

    Problem: Some experts overused (winner-take-all)
    Solution: Auxiliary loss to encourage balanced routing

    Formula:
    L_balance = α × Σ_e (f_e × P_e)

    f_e = Fraction of tokens assigned to expert e
    P_e = Average routing probability assigned to expert e
    α = Scaling coefficient (e.g., 0.01)
    """
    batch, seq_len, top_k = router_probs.shape
    num_tokens = batch * seq_len

    # f_e: Fraction selected for each expert
    expert_counts = torch.zeros(num_experts, device=router_probs.device)
    for e in range(num_experts):
        expert_counts[e] = (expert_indices == e).float().sum() / (num_tokens * top_k)

    # P_e: Average probability assigned to each expert
    expert_probs = torch.zeros(num_experts, device=router_probs.device)
    # (Simplified calculation - actually computed from gate logits)

    # Balance loss
    loss = (expert_counts * expert_probs).sum() * num_experts

    return loss

# During training
"""
total_loss = language_model_loss + alpha * load_balancing_loss
"""
```

---

## 5. Pros and Cons of MoE

### 5.1 Advantages

```
┌─────────────────────────────────────────────────────────────────┐
│                    Advantages of MoE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Parameter Efficiency                                        │
│     • Many parameters, little computation                       │
│     • Mixtral 8x7B: 46.7B params, 12.9B active                  │
│     • Dense 70B performance, 13B speed                          │
│                                                                 │
│  2. Specialization                                              │
│     • Each expert learns different patterns/domains             │
│     • Example: Expert 1=math, Expert 2=code, Expert 3=language  │
│     • Can encode deeper specialized knowledge                   │
│                                                                 │
│  3. Scaling                                                     │
│     • Easy to expand model by adding experts                    │
│     • Increase capacity with minimal computation increase       │
│     • Google Switch Transformer: 1.6T params!                   │
│                                                                 │
│  4. Training Efficiency                                         │
│     • Can train larger models with same computation             │
│     • Advantageous from Scaling Law perspective                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Disadvantages

```
┌─────────────────────────────────────────────────────────────────┐
│                    Disadvantages of MoE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Memory Requirements                                         │
│     • Must load all experts into memory                         │
│     • Mixtral 8x7B: 46.7B params ≈ 93GB (FP16)                  │
│     • Requires large GPU memory for inference                   │
│                                                                 │
│  2. Training Instability                                        │
│     • Difficult to train router                                 │
│     • Expert imbalance (only some used)                         │
│     • Auxiliary loss tuning needed                              │
│                                                                 │
│  3. Distributed Training Complexity                             │
│     • Expert parallelism required                               │
│     • Communication overhead                                    │
│     • Load balancing difficult                                  │
│                                                                 │
│  4. Fine-tuning Challenges                                      │
│     • Need to adapt while maintaining expert specialization     │
│     • Fine-tune only some experts?                              │
│     • Research ongoing                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Mistral/Mixtral Practice

### 6.1 Using Mistral 7B

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Mistral 7B
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Text generation
prompt = "[INST] Explain the concept of machine learning in simple terms. [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 6.2 Using Mixtral 8x7B

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mixtral 8x7B (requires lots of memory!)
model_name = "mistralai/Mixtral-8x7B-v0.1"

# 4-bit quantization for memory savings
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Usage
prompt = "[INST] Write a Python function to calculate fibonacci numbers. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 6.3 Efficient Serving with vLLM

```python
from vllm import LLM, SamplingParams

# vLLM efficiently serves MoE models
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    tensor_parallel_size=2,  # 2 GPUs
    dtype="float16",
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200,
)

prompts = [
    "[INST] What is machine learning? [/INST]",
    "[INST] Explain quantum computing. [/INST]",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")
    print("-" * 50)
```

---

## 7. MoE Variants

### 7.1 Major MoE Models

| Model | Organization | Experts | Top-K | Total Params | Active Params |
|-------|--------------|---------|-------|--------------|---------------|
| Switch Transformer | Google | 2048 | 1 | 1.6T | <1B |
| GLaM | Google | 64 | 2 | 1.2T | ~100B |
| Mixtral 8x7B | Mistral | 8 | 2 | 46.7B | 12.9B |
| Mixtral 8x22B | Mistral | 8 | 2 | 141B | 39B |
| DeepSeek MoE | DeepSeek | 64 | 6 | 145B | 22B |

### 7.2 Fine-grained MoE

```python
"""
Fine-grained MoE: More small experts

Traditional (Coarse-grained):
- 8 large experts, Top-2 selection
- Each expert covers broad range

Fine-grained (DeepSeek style):
- 64 small experts, Top-6 selection
- More granular specialization possible
- Increased routing flexibility

Advantages:
- More fine-grained specialization
- Better load balancing
- Scalability

Disadvantages:
- Routing overhead
- Training complexity
"""
```

---

## 8. Expert Routing Mechanisms

The router is the brain of an MoE model — it decides which experts process each token. The choice of routing strategy profoundly affects model quality, training stability, and inference efficiency. This section dives deep into routing strategies, load balancing, and capacity management.

### 8.1 Token Routing Strategies

#### Top-K Routing (Standard)

The most common approach: a learned linear layer scores each expert, and the top K experts are selected.

```
Input token x → Linear(dim, n_experts) → logits
                                          ↓
                               Top-K selection → softmax over K
                                          ↓
                               Weighted sum of K expert outputs

Used by: Mixtral (K=2), GLaM (K=2), GShard (K=2)
```

#### Expert Choice Routing (Zhou et al., 2022)

Instead of tokens choosing experts, **experts choose tokens**. Each expert selects the top-C tokens it is most confident about.

```
Standard (token chooses expert):
  Each token picks its best K experts → load can be uneven

Expert Choice (expert chooses tokens):
  Each expert picks its best C tokens → perfect load balance

  For E experts, N tokens, capacity factor C:
  Each expert processes exactly (N × C / E) tokens

Advantages:
  • Guaranteed load balance (no auxiliary loss needed)
  • Experts see tokens they are most suited for

Disadvantages:
  • Some tokens may not be selected by any expert (dropped)
  • Some tokens may be selected by too many experts (redundant)
  • Harder to implement efficiently for autoregressive decoding

Used by: Zhou et al. (2022), some Google internal models
```

#### Hash Routing (Roller et al., 2021)

A non-learned, deterministic routing strategy based on hashing:

```
expert_id = hash(token_id) % n_experts

Advantages:
  • No learnable parameters in the router
  • Perfectly balanced by design
  • No routing collapse or instability

Disadvantages:
  • No input-dependent specialization
  • Cannot adapt routing based on context
  • Primarily used as a baseline or for analysis

Surprisingly, hash routing performs reasonably well, suggesting
that much of MoE's benefit comes from increased capacity
rather than intelligent routing.
```

### 8.2 Router Architecture Details

#### Softmax Gate (Standard)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRouter(nn.Module):
    """
    Standard softmax router.

    Why softmax: We want a probability distribution over experts
    so that the weighted combination of expert outputs is well-
    scaled. Softmax ensures non-negative weights that sum to 1
    (among selected experts after top-k filtering).
    """
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        # x: (batch, seq_len, dim)
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Select top-K experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over selected experts only
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        return top_k_probs, top_k_indices, logits
```

#### Noisy Top-K (Shazeer et al., 2017)

Adding noise to router logits before top-K selection improves exploration during training:

```python
class NoisyTopKRouter(nn.Module):
    """
    Noisy Top-K Gating from the original MoE paper (Shazeer 2017).

    Why add noise: Without noise, the router tends to converge
    early to always selecting the same experts (rich-get-richer).
    Gaussian noise encourages exploration, helping all experts
    receive training signal.
    """
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.noise_linear = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        if self.training:
            # Learnable noise scale
            noise_stddev = F.softplus(self.noise_linear(x))
            noise = torch.randn_like(logits) * noise_stddev
            logits = logits + noise

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        return top_k_probs, top_k_indices, logits
```

### 8.3 Load Balancing Loss and Auxiliary Loss

Without explicit encouragement, routers tend to route most tokens to a few "favorite" experts (winner-take-all collapse). Auxiliary losses prevent this.

#### Standard Load Balancing Loss (Switch Transformer)

```python
def switch_load_balancing_loss(router_logits, expert_indices, num_experts):
    """
    Load balancing loss from Switch Transformer (Fedus et al., 2022).

    Intuition: We want each expert to receive roughly 1/E of all
    tokens AND roughly 1/E of the total routing probability mass.
    The loss penalizes deviation from uniform distribution.

    L_balance = E × Σ_e (f_e × p_e)

    f_e = fraction of tokens routed to expert e
    p_e = mean routing probability for expert e (before top-k)
    E   = number of experts

    The ideal value of (f_e × p_e) for each expert is 1/E²,
    so the ideal total loss is E × E × (1/E²) = 1.0.
    Values > 1.0 indicate imbalance.
    """
    batch, seq_len = expert_indices.shape[:2]
    num_tokens = batch * seq_len

    # f_e: fraction of tokens dispatched to each expert
    f = torch.zeros(num_experts, device=router_logits.device)
    for e in range(num_experts):
        f[e] = (expert_indices == e).float().sum() / num_tokens

    # p_e: mean routing probability for each expert (from full softmax)
    full_probs = F.softmax(router_logits, dim=-1)  # (batch, seq, n_experts)
    p = full_probs.mean(dim=[0, 1])  # (n_experts,)

    # Balance loss
    loss = num_experts * (f * p).sum()

    return loss
```

#### Z-Loss (ST-MoE, Zoph et al., 2022)

An additional auxiliary loss that penalizes large router logits to improve stability:

```python
def z_loss(router_logits):
    """
    Z-loss penalizes large router logit magnitudes.

    Why: Large logits lead to very peaked softmax distributions,
    which cause training instability. Z-loss acts as a soft
    regularizer on the router's output scale.

    L_z = (1/N) × Σ (log Σ exp(logits))²
    """
    log_z = torch.logsumexp(router_logits, dim=-1)  # (batch, seq)
    return (log_z ** 2).mean()
```

### 8.4 Capacity Factor and Overflow Handling

Each expert has a maximum **capacity** — the number of tokens it can process per forward pass. This is critical for efficient batched computation.

```
Capacity = (total_tokens / num_experts) × capacity_factor

capacity_factor (CF):
  CF = 1.0 → each expert processes exactly 1/E of tokens (tight)
  CF = 1.5 → 50% buffer for uneven routing (typical)
  CF = 2.0 → generous buffer, wastes some computation

Overflow: When more tokens are routed to an expert than its
capacity allows, excess tokens are DROPPED (their expert
output is zero, only the residual connection passes through).

Underflow: When fewer tokens are routed, the expert's buffer
has padding (wasted computation but no correctness issue).
```

```python
def apply_capacity_factor(expert_indices, num_experts, capacity_factor=1.5):
    """
    Apply capacity constraints to expert assignments.

    Why capacity factor > 1.0: Even with load balancing loss,
    routing is never perfectly uniform. A buffer of 1.25-1.5
    prevents frequent token dropping while keeping memory bounded.
    """
    batch, seq_len, top_k = expert_indices.shape
    total_tokens = batch * seq_len
    capacity = int((total_tokens / num_experts) * capacity_factor)

    # Track how many tokens each expert has accepted
    expert_load = torch.zeros(num_experts, dtype=torch.long)
    overflow_mask = torch.zeros(batch, seq_len, top_k, dtype=torch.bool)

    for b in range(batch):
        for s in range(seq_len):
            for k in range(top_k):
                e = expert_indices[b, s, k].item()
                if expert_load[e] < capacity:
                    expert_load[e] += 1
                else:
                    overflow_mask[b, s, k] = True  # This assignment is dropped

    return overflow_mask, expert_load
```

### 8.5 Switch Transformer vs GShard Routing Comparison

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│ Feature          │ Switch Transformer   │ GShard               │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Top-K            │ K=1 (single expert)  │ K=2 (two experts)    │
│ Capacity factor  │ 1.0-1.5              │ 2.0                  │
│ Overflow         │ Drop token           │ Drop token           │
│ Aux loss         │ L_balance            │ L_balance + L_load   │
│ Noise            │ Not used             │ Noisy top-k          │
│ Expert count     │ 128-2048             │ 64-512               │
│ Quality at scale │ Good with many exp.  │ Better per-expert    │
│ Communication    │ Lower (K=1)          │ Higher (K=2)         │
│ Typical use      │ Extreme scale (>1T)  │ Moderate scale       │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Key Insight      │ Simplicity: 1 expert │ Robustness: 2 experts│
│                  │ per token is enough  │ provide redundancy   │
│                  │ at large scale       │ and smoother training│
└──────────────────┴──────────────────────┴──────────────────────┘

Modern practice (Mixtral, DeepSeek):
- K=2 (GShard-style) has become the default
- Combined with improved load balancing (auxiliary loss + z-loss)
- Capacity factor of 1.25-1.5 (empirically tuned)
```

### 8.6 Complete Router Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    """
    Complete MoE Router with load balancing and capacity management.

    This implementation combines the key ideas from Switch Transformer,
    GShard, and ST-MoE into a single practical router.
    """
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.5,
        balance_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.balance_loss_weight = balance_loss_weight
        self.z_loss_weight = z_loss_weight

        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            expert_probs: (batch, seq_len, top_k)
            expert_indices: (batch, seq_len, top_k)
            aux_loss: scalar auxiliary loss for training
        """
        batch, seq_len, dim = x.shape
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Compute auxiliary losses during training
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            # Load balancing loss
            full_probs = F.softmax(logits, dim=-1)
            p = full_probs.mean(dim=[0, 1])  # Mean prob per expert

            num_tokens = batch * seq_len
            f = torch.zeros(self.num_experts, device=x.device)
            for e in range(self.num_experts):
                f[e] = (top_k_indices == e).float().sum() / (num_tokens * self.top_k)

            balance_loss = self.num_experts * (f * p).sum()

            # Z-loss for stability
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss_val = (log_z ** 2).mean()

            aux_loss = (self.balance_loss_weight * balance_loss +
                        self.z_loss_weight * z_loss_val)

        return top_k_probs, top_k_indices, aux_loss


# Usage example
router = MoERouter(dim=4096, num_experts=8, top_k=2)
x = torch.randn(2, 10, 4096)
probs, indices, aux_loss = router(x)

print(f"Expert probs: {probs.shape}")      # (2, 10, 2)
print(f"Expert indices: {indices.shape}")    # (2, 10, 2)
print(f"Auxiliary loss: {aux_loss.item():.4f}")
```

---

## Summary

### Mistral Core
- **Sliding Window Attention**: O(W) memory for long sequences
- **GQA**: KV cache efficiency
- **Over-training**: Small model, lots of data

### MoE Core
- **Sparse Activation**: Many parameters, little computation
- **Router**: Top-K expert selection
- **Load Balancing**: Maintain expert balance

### Practical Selection Guide
| Situation | Recommended Model |
|-----------|-------------------|
| Single GPU (16GB) | Mistral 7B (4-bit) |
| 2× GPU (48GB) | Mixtral 8x7B (4-bit) |
| Server-grade (8× A100) | Mixtral 8x22B |
| Speed priority | Mistral 7B |
| Performance priority | Mixtral 8x7B+ |

### Next Steps
- [10_Long_Context_Models.md](10_Long_Context_Models.md): Long context processing
- [22_Inference_Optimization.md](22_Inference_Optimization.md): Efficient inference

---

## References

### Core Papers
- Jiang et al. (2023). "Mistral 7B"
- Jiang et al. (2024). "Mixtral of Experts"
- Fedus et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models"
- Du et al. (2022). "GLaM: Efficient Scaling of Language Models"

### Code & Resources
- [Mistral GitHub](https://github.com/mistralai/mistral-src)
- [HuggingFace Mistral](https://huggingface.co/mistralai)
- [vLLM MoE Support](https://docs.vllm.ai/)

---

## Exercises

### Exercise 1: MoE Effective Parameter Analysis

Mixtral 8x7B has 8 experts per MoE layer, activates 2 per token (Top-2 routing), and has 46.7B total parameters with 12.9B active parameters per forward pass.

1. Why does Mixtral activate only 12.9B parameters for each token despite having 46.7B total?
2. What is the "computation efficiency ratio" (active / total) and what does it mean for inference cost vs a dense 46.7B model?
3. If Mixtral had K=4 (instead of K=2), how would active parameters and inference cost change?

<details>
<summary>Show Answer</summary>

**1. Why only 12.9B active parameters:**

In each MoE layer, the router selects only 2 of 8 experts to process each token. The non-selected 6 experts simply don't execute — their computations are skipped entirely. The transformer attention layers (shared across all tokens) plus only 2/8 of the FFN parameters are active. Approximately:
- Shared attention params + 2/8 of MoE FFN params ≈ 12.9B per token

**2. Computation efficiency ratio:**

```
Efficiency ratio = 12.9B / 46.7B ≈ 27.6%
```

This means each forward pass only costs ~27.6% of what a dense 46.7B model would cost. Yet the model has 46.7B parameters worth of knowledge capacity. This is the core MoE trade-off: pay for inference like a ~13B dense model but benefit from 46.7B parameter capacity.

**3. Effect of K=4:**
- Active parameters would roughly double: ~25.8B per forward pass
- Inference cost increases proportionally (2× more expert computations per token)
- Computation efficiency ratio: 25.8 / 46.7 ≈ 55.2%
- The model becomes a 25.8B-equivalent dense model in terms of compute, but still has 46.7B knowledge capacity
- Quality typically improves with K=4 but at the cost of losing the efficiency advantage

</details>

---

### Exercise 2: Sliding Window Attention Memory Analysis

Mistral 7B uses Sliding Window Attention (SWA) with window size W=4096 and rolling buffer KV cache.

1. Standard self-attention has O(n²) memory complexity for n tokens. What is the memory complexity of SWA for n tokens?
2. For a sequence of 32,768 tokens (32K), compare the KV cache memory for standard MHA vs SWA (assume 32 heads, head_dim=128, fp16, 32 layers).
3. What information is lost when a token "falls out" of the sliding window, and how does the multi-layer architecture partially compensate?

<details>
<summary>Show Answer</summary>

**1. Memory complexity of SWA:**
SWA stores only the last W tokens in the KV cache. As n grows, memory stays bounded:
- **O(W × d)** = O(d) — effectively constant with respect to sequence length, since W is fixed.
- More precisely: O(n) time (each token is processed once) but O(W) KV cache space.

**2. KV cache memory comparison for n=32768 tokens:**

Standard MHA (stores all n tokens):
```
32768 × 2 (K+V) × 32 heads × 128 head_dim × 2 bytes × 32 layers
= 32768 × 2 × 32 × 128 × 2 × 32 = 17,179,869,184 bytes ≈ 16 GB
```

SWA (stores only last W=4096 tokens):
```
4096 × 2 × 32 × 128 × 2 × 32 = 2,147,483,648 bytes ≈ 2 GB
```

**SWA saves ~14 GB** of KV cache memory for 32K tokens — an 8× reduction.

**3. Information loss and compensation:**

When a token falls outside the window (older than W positions back), direct attention to it is lost. However, the multi-layer architecture partially compensates through the **receptive field expansion** property:
- Layer 1 can attend to the last W tokens
- Layer 2 attends to tokens that Layer 1 already "summarized" — effectively seeing W² tokens of history
- After k layers: receptive field ≈ W × k

For Mistral 7B with 32 layers and W=4096: effective receptive field ≈ 4096 × 32 = 131,072 tokens — far beyond the window size. Early context is implicitly "compressed" into intermediate representations rather than directly accessible.

</details>

---

### Exercise 3: Load Balancing Loss Implementation

The lesson shows the Switch Transformer load balancing loss. Explain what would happen to expert utilization during training if this loss were **not** included, and trace through why router collapse occurs.

<details>
<summary>Show Answer</summary>

**Without load balancing loss, router collapse proceeds as follows:**

**Step 1: Initial asymmetry**
During initialization, the router's linear layer has random weights. By chance, some experts receive slightly higher routing probabilities for common input patterns. This asymmetry is tiny but present.

**Step 2: Rich-get-richer dynamics**
The "preferred" experts receive more training tokens → their parameters improve more → their outputs are better → the reward signal (lower loss) for routing to them is higher → the router learns to prefer them more strongly.

**Step 3: Expert collapse**
After sufficient training, 1-2 experts receive nearly all tokens. The other 6-7 experts receive almost no gradient signal and stagnate. Their parameters don't improve because they're never trained.

**Step 4: Complete collapse**
The model effectively becomes equivalent to a dense model with only 1-2 FFN matrices — it loses all the capacity benefits of having 8 experts. The total parameter count is still 8× larger (all expert weights still exist) but only 1-2 are useful.

**What the load balancing loss does:**
By penalizing when `f_e` (fraction of tokens to expert e) deviates from uniform (1/E), the loss creates a gradient signal that pushes the router to distribute tokens more evenly. Combined with noisy top-k (adding exploration noise to router logits), this prevents early convergence to a degenerate solution.

</details>

---

### Exercise 4: MoE vs Dense Trade-off Decision

You are choosing between two architectures for a production language model:
- **Model A**: Dense 13B parameter model
- **Model B**: Sparse MoE with 8 experts, top-2 routing, 46.7B total parameters, 13B active

Both have approximately the same inference compute cost. Evaluate which to choose for each scenario:

1. Deployment on a memory-constrained edge server (32GB total RAM).
2. A research lab that regularly fine-tunes on small domain-specific datasets.
3. A production API that needs maximum knowledge breadth across diverse topics.
4. A system requiring exactly reproducible outputs for the same input.

<details>
<summary>Show Answer</summary>

**1. Memory-constrained edge server (32GB RAM):**
- **Choose Dense 13B** — Model B requires loading all 46.7B parameters (~93GB in fp16), far exceeding the 32GB limit even with quantization. Dense 13B needs ~26GB in fp16. For edge deployment with tight memory budgets, dense models are strongly preferred.

**2. Research lab, fine-tuning on small datasets:**
- **Choose Dense 13B** — MoE models are notoriously harder to fine-tune:
  - The router must learn appropriate expert assignments simultaneously
  - Small datasets don't provide enough signal to properly update all experts
  - Fine-tuning instability is more common (some experts collapse, some over-specialize)
  - Dense models fine-tune more predictably and efficiently on small datasets.

**3. Maximum knowledge breadth across diverse topics:**
- **Choose MoE 46.7B** — The entire motivation for MoE is that different experts can specialize in different domains (coding, reasoning, language patterns, factual knowledge). With 46.7B total parameters, Model B has far more capacity for diverse knowledge than the 13B dense model. For a general-purpose API, this breadth advantage is significant.

**4. Exactly reproducible outputs:**
- **Prefer Dense 13B** (but caveats apply to both) — MoE introduces an additional source of non-determinism: the discrete top-K routing decision. Minor floating-point differences between hardware runs can flip which expert is selected, changing outputs even with temperature=0. Dense models only have floating-point precision as a source of non-determinism. Note: with sufficient precision control, both can be made deterministic.

</details>

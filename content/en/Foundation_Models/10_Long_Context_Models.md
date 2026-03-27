# 10. Long Context Models

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why standard Transformer self-attention has O(n²) time and memory complexity and identify the practical context length limits this imposes.
2. Compare efficient attention mechanisms (sparse attention, linear attention, FlashAttention) and explain how each reduces computational cost while preserving model quality.
3. Describe position encoding extension techniques (RoPE, ALiBi, YaRN, LongRoPE) and explain how they enable models trained on short contexts to generalize to longer sequences.
4. Analyze the "lost in the middle" phenomenon in long-context models and explain its implications for retrieval, agent design, and document understanding tasks.
5. Implement sliding window and chunked attention patterns, and compare them to full attention in terms of accuracy and memory usage.
6. Evaluate long-context benchmarks (SCROLLS, L-Eval, RULER) and discuss the trade-offs between context length, inference latency, and memory footprint in production deployments.

---

## Overview

Standard Transformer Self-Attention has limitations in processing long sequences due to O(n²) complexity. This lesson covers various techniques for extending context length.

---

## 1. Importance of Context Length

### 1.1 Why Do We Need Long Context?

```
┌──────────────────────────────────────────────────────────────────┐
│                   Long Context Use Cases                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Document Analysis                                               │
│  - Entire papers (10K-50K tokens)                               │
│  - Legal documents (100K+ tokens)                               │
│  - Full book summarization                                       │
│                                                                  │
│  Code Understanding                                              │
│  - Entire codebase analysis                                     │
│  - Long function/class refactoring                              │
│  - Multi-file debugging                                         │
│                                                                  │
│  Agent Systems                                                   │
│  - Maintain long conversation history                           │
│  - Complex multi-step tasks                                     │
│  - Accumulated tool usage records                               │
│                                                                  │
│  RAG Improvement                                                 │
│  - Include more relevant documents                              │
│  - Provide full documents instead of fragments                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Context Length Comparison by Model

| Model | Context Length | Release Date |
|-------|----------------|--------------|
| GPT-3 | 2,048 | 2020 |
| GPT-3.5 | 4,096 / 16,384 | 2022-2023 |
| GPT-4 | 8,192 / 32,768 / 128K | 2023-2024 |
| Claude 2 | 100,000 | 2023 |
| Claude 3 | 200,000 | 2024 |
| Gemini 1.5 | 1,000,000 / 2,000,000 | 2024 |
| LLaMA 2 | 4,096 | 2023 |
| LLaMA 3 | 8,192 / 128K | 2024 |

---

## 2. Efficient Attention Mechanisms

### 2.1 Sparse Attention

```
┌─────────────────────────────────────────────────────────────┐
│                    Sparse Attention Patterns                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Local Attention        Global Attention                   │
│  ■ ■ ■ □ □ □ □         ■ □ □ □ □ □ □                      │
│  ■ ■ ■ ■ □ □ □         ■ ■ □ □ □ □ □                      │
│  □ ■ ■ ■ ■ □ □         ■ □ ■ □ □ □ □                      │
│  □ □ ■ ■ ■ ■ □         ■ □ □ ■ □ □ □                      │
│  □ □ □ ■ ■ ■ ■         ■ □ □ □ ■ □ □                      │
│  □ □ □ □ ■ ■ ■         ■ □ □ □ □ ■ □                      │
│  □ □ □ □ □ ■ ■         ■ □ □ □ □ □ ■                      │
│                                                             │
│  Longformer: Local + Global token combination               │
│  BigBird: Local + Global + Random                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Longformer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LongformerAttention(nn.Module):
    """
    Longformer: Sliding Window + Global Attention

    Complexity: O(n × w) where w = window size
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 256,
        global_tokens: int = 2  # [CLS], [SEP]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.global_tokens = global_tokens

        # Q, K, V projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Separate projection for global attention
        self.global_query = nn.Linear(hidden_size, hidden_size)
        self.global_key = nn.Linear(hidden_size, hidden_size)
        self.global_value = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 1. Sliding Window Attention (local)
        local_output = self._sliding_window_attention(Q, K, V)

        # 2. Global Attention (first global_tokens)
        global_output = self._global_attention(
            hidden_states, Q, K, V
        )

        # Combine (use global result for global token positions)
        output = local_output.clone()
        output[:, :self.global_tokens] = global_output[:, :self.global_tokens]

        # Output projection
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.output(output)

        return output

    def _sliding_window_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Sliding Window Attention

        Each token only attends to tokens within window_size range
        """
        batch_size, seq_len, num_heads, head_dim = Q.shape
        w = self.window_size // 2

        # Add padding
        Q_padded = F.pad(Q, (0, 0, 0, 0, w, w), value=0)
        K_padded = F.pad(K, (0, 0, 0, 0, w, w), value=0)
        V_padded = F.pad(V, (0, 0, 0, 0, w, w), value=0)

        # Extract windows (unfold)
        # Actual implementation is more complex, this is simplified for understanding
        output = torch.zeros_like(Q)

        for i in range(seq_len):
            # Window for i-th token: [i, i + window_size]
            start = i
            end = i + self.window_size

            q_i = Q[:, i:i+1]  # (batch, 1, heads, dim)
            k_window = K_padded[:, start:end]  # (batch, window, heads, dim)
            v_window = V_padded[:, start:end]

            # Attention
            scores = torch.einsum('bihd,bjhd->bijh', q_i, k_window)
            scores = scores / math.sqrt(head_dim)
            weights = F.softmax(scores, dim=2)
            out_i = torch.einsum('bijh,bjhd->bihd', weights, v_window)

            output[:, i] = out_i[:, 0]

        return output

    def _global_attention(
        self,
        hidden_states: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """Global Attention: global tokens attend to entire sequence"""
        batch_size, seq_len, _ = hidden_states.shape

        # Extract only global tokens
        global_hidden = hidden_states[:, :self.global_tokens]

        # Global Q, K, V
        global_Q = self.global_query(global_hidden)
        global_K = self.global_key(hidden_states)
        global_V = self.global_value(hidden_states)

        # Attention over entire sequence
        global_Q = global_Q.view(batch_size, self.global_tokens,
                                  self.num_heads, self.head_dim)
        global_K = global_K.view(batch_size, seq_len,
                                  self.num_heads, self.head_dim)
        global_V = global_V.view(batch_size, seq_len,
                                  self.num_heads, self.head_dim)

        # (batch, global, heads, seq) attention
        scores = torch.einsum('bghd,bshd->bghs', global_Q, global_K)
        scores = scores / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)

        # Output: (batch, global, heads, dim)
        output = torch.einsum('bghs,bshd->bghd', weights, global_V)

        return output
```

### 2.3 Flash Attention

```python
# Flash Attention is implemented as a CUDA kernel
# Here we only explain the concept

"""
Flash Attention Key Ideas:

1. Tiling:
   - Divide Q, K, V into blocks that fit in SRAM
   - Minimize HBM ↔ SRAM data transfer

2. Recomputation:
   - Don't store attention weights in forward pass
   - Recompute when needed in backward pass
   - Memory savings (O(n) vs O(n²))

3. Results:
   - Memory: O(n) vs O(n²)
   - Speed: 2-4x faster
   - Accuracy: Numerically identical
"""

# Using with PyTorch 2.0+
def use_flash_attention():
    import torch.nn.functional as F

    # Scaled Dot-Product Attention (automatically uses Flash Attention)
    Q = torch.randn(2, 8, 1024, 64, device='cuda')
    K = torch.randn(2, 8, 1024, 64, device='cuda')
    V = torch.randn(2, 8, 1024, 64, device='cuda')

    # PyTorch 2.0+ SDPA
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        output = F.scaled_dot_product_attention(Q, K, V)

    return output


# Using xFormers
def use_xformers():
    from xformers.ops import memory_efficient_attention

    Q = torch.randn(2, 1024, 8, 64, device='cuda')
    K = torch.randn(2, 1024, 8, 64, device='cuda')
    V = torch.randn(2, 1024, 8, 64, device='cuda')

    output = memory_efficient_attention(Q, K, V)
    return output
```

---

## 3. Position Encoding Extension

### 3.1 Problem: Extrapolation Beyond Training Length

```
Training: 4096 tokens
Inference: 8192+ tokens

Problem:
- Absolute position encoding: Positions after 4096 not learned
- RoPE: Requires interpolation/extrapolation
```

### 3.2 Position Interpolation (PI)

```python
def linear_position_interpolation(
    position_ids: torch.Tensor,
    original_max_length: int,
    extended_max_length: int
) -> torch.Tensor:
    """
    Linear Position Interpolation

    Idea: Scale new positions to original range

    Compress position_ids to [0, original_max_length)
    """
    scale = original_max_length / extended_max_length
    return position_ids.float() * scale


class RoPEWithInterpolation(nn.Module):
    """RoPE with Position Interpolation applied"""

    def __init__(
        self,
        dim: int,
        original_max_length: int = 4096,
        extended_max_length: int = 16384,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.original_max_length = original_max_length
        self.extended_max_length = extended_max_length
        self.base = base

        # Frequency calculation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Scale factor
        self.scale = original_max_length / extended_max_length

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, heads, dim)
            position_ids: (batch, seq_len)
        """
        # Position interpolation
        scaled_positions = position_ids.float() * self.scale

        # Frequency calculation
        freqs = torch.einsum('bi,d->bid', scaled_positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(2)  # (batch, seq, 1, dim)
        sin = emb.sin().unsqueeze(2)

        # Apply RoPE
        x_rope = self._apply_rope(x, cos, sin)

        return x_rope

    def _apply_rope(self, x, cos, sin):
        """Apply RoPE"""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]

        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin
```

### 3.3 YaRN (Yet another RoPE extension method)

```python
class YaRNRoPE(nn.Module):
    """
    YaRN: NTK-aware Interpolation

    Problem with Position Interpolation:
    - High frequency information loss (higher dimensions)

    YaRN Solution:
    - Low frequency: Interpolation
    - High frequency: Extrapolation
    - Adjust frequencies with NTK scaling
    """

    def __init__(
        self,
        dim: int,
        original_max_length: int = 4096,
        extended_max_length: int = 32768,
        base: float = 10000.0,
        beta_fast: float = 32,
        beta_slow: float = 1,
    ):
        super().__init__()
        self.dim = dim
        self.original_max_length = original_max_length
        self.extended_max_length = extended_max_length

        scale = extended_max_length / original_max_length

        # Calculate interpolation ratio per dimension
        # Low frequency (lower dimensions): More interpolation
        # High frequency (higher dimensions): Less interpolation (closer to extrapolation)
        dims = torch.arange(0, dim, 2)
        low = max(0, math.floor(dim * math.log(scale) / (2 * math.log(original_max_length))))
        high = min(dim // 2 - 1, math.ceil(dim * math.log(scale) / (2 * math.log(original_max_length))))

        # Ramp function to determine interpolation/extrapolation ratio
        ramp = torch.zeros(dim // 2)
        ramp[:low] = 0.0  # Extrapolation
        ramp[high:] = 1.0  # Interpolation

        if high > low:
            ramp[low:high] = (dims[low:high] - low) / (high - low)

        # NTK-aware base adjustment
        inv_freq = 1.0 / (base ** (dims.float() / dim))

        # Mix of interpolation and extrapolation
        inv_freq_inter = inv_freq / scale
        self.register_buffer(
            'inv_freq',
            (1 - ramp) * inv_freq + ramp * inv_freq_inter
        )

        # Attention scaling
        self.mscale = 0.1 * math.log(scale) + 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # Frequency calculation (using already adjusted inv_freq)
        freqs = torch.einsum('bi,d->bid', position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(2) * self.mscale
        sin = emb.sin().unsqueeze(2) * self.mscale

        return self._apply_rope(x, cos, sin)
```

---

## 4. ALiBi (Attention with Linear Biases)

### 4.1 Concept

```
ALiBi: Position encoding without training

Idea:
- Don't use position encoding
- Instead, add distance-based penalty to attention scores
- Tokens further away get lower attention scores

Attention score modification:
score(q_i, k_j) = q_i · k_j - m × |i - j|

m: Slope per head (fixed, not learned)
m_h = 2^(-8/H) for head h (H = total number of heads)
```

### 4.2 Implementation

```python
class ALiBiAttention(nn.Module):
    """ALiBi: Attention with Linear Biases"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        # ALiBi slopes: Decrease exponentially
        # 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Precompute distance matrix
        positions = torch.arange(max_seq_len)
        distance_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
        distance_matrix = distance_matrix.abs()
        self.register_buffer('distance_matrix', distance_matrix)

    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Calculate ALiBi slope per head"""

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Interpolate to nearest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)

            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
            extra_slopes = extra_slopes[0::2][:num_heads - closest_power_of_2]
            slopes = slopes + extra_slopes

        return torch.tensor(slopes).view(1, num_heads, 1, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, heads, seq, dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ALiBi bias: -m × |i - j|
        alibi_bias = -self.slopes * self.distance_matrix[:seq_len, :seq_len]
        scores = scores + alibi_bias

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device) * float('-inf'),
            diagonal=1
        )
        scores = scores + causal_mask

        # Attention weights
        weights = F.softmax(scores, dim=-1)

        # Output
        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.output(output)

        return output
```

---

## 5. Ring Attention

### 5.1 Concept

```
Ring Attention: Distributed Long Context

Idea:
- Distribute sequence across multiple GPUs
- Each GPU processes its chunk + circulating KV
- Overlap communication and computation

┌────────────────────────────────────────────────┐
│                Ring Attention                   │
├────────────────────────────────────────────────┤
│                                                │
│  GPU 0: Q[0:n/4]     GPU 1: Q[n/4:n/2]        │
│          ↓ KV circulates   ↓ KV circulates    │
│  Step 1: K[0:n/4]    Step 1: K[n/4:n/2]       │
│  Step 2: K[n/4:n/2]  Step 2: K[n/2:3n/4]      │
│  Step 3: K[n/2:3n/4] Step 3: K[3n/4:n]        │
│  Step 4: K[3n/4:n]   Step 4: K[0:n/4]         │
│                                                │
│  KV circulates like a ring, combining with    │
│  each GPU's Q                                  │
│                                                │
└────────────────────────────────────────────────┘
```

### 5.2 Implementation Overview

```python
import torch.distributed as dist

def ring_attention_forward(
    Q: torch.Tensor,  # Local Q chunk
    K: torch.Tensor,  # Local K chunk
    V: torch.Tensor,  # Local V chunk
    world_size: int,
    rank: int
):
    """
    Ring Attention Forward Pass (conceptual implementation)

    Actual implementation requires CUDA kernels and complex synchronization
    """
    local_seq_len = Q.shape[1]

    # Accumulated attention output
    output = torch.zeros_like(Q)
    max_scores = torch.full((Q.shape[0], Q.shape[2], local_seq_len), float('-inf'))
    sum_exp = torch.zeros_like(max_scores)

    # Current KV
    current_K = K.clone()
    current_V = V.clone()

    for step in range(world_size):
        # Compute attention for this chunk's KV
        scores = torch.matmul(Q, current_K.transpose(-2, -1))
        scores = scores / math.sqrt(Q.shape[-1])

        # Online softmax (numerically stable)
        new_max = torch.max(scores.max(dim=-1).values, max_scores)
        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))

        # Scale previous results
        scale = torch.exp(max_scores - new_max)
        output = output * scale.unsqueeze(-1) + torch.matmul(exp_scores, current_V)

        sum_exp = sum_exp * scale + exp_scores.sum(dim=-1)
        max_scores = new_max

        # Send KV to next GPU (ring)
        if step < world_size - 1:
            # Async send/recv
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1) % world_size

            # Receive KV from next GPU
            current_K = ring_pass(current_K, send_rank, recv_rank)
            current_V = ring_pass(current_V, send_rank, recv_rank)

    # Final normalization
    output = output / sum_exp.unsqueeze(-1)

    return output


def ring_pass(tensor, send_rank, recv_rank):
    """Pass tensor in ring topology"""
    recv_tensor = torch.empty_like(tensor)

    send_op = dist.isend(tensor, send_rank)
    recv_op = dist.irecv(recv_tensor, recv_rank)

    send_op.wait()
    recv_op.wait()

    return recv_tensor
```

---

## 6. Practical Guide

### 6.1 Choosing Context Extension Method

```
┌──────────────────────────────────────────────────────────────┐
│              When to Use Which Method?                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  4K → 8K:                                                    │
│  - Position Interpolation (simple, good performance)         │
│  - Some fine-tuning recommended                              │
│                                                              │
│  4K → 32K:                                                   │
│  - YaRN (better than PI)                                    │
│  - Or ALiBi (if training from scratch)                      │
│                                                              │
│  32K → 100K+:                                                │
│  - Flash Attention essential                                 │
│  - Ring Attention (multi-GPU)                               │
│  - Consider Sparse Attention                                 │
│                                                              │
│  1M+:                                                        │
│  - Special architectures needed                              │
│  - Mamba/State Space Models                                  │
│  - Or extremely sparse attention                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Practical Tips

```python
# 1. Gradient Checkpointing is essential
model.gradient_checkpointing_enable()

# 2. Use Mixed Precision
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(**inputs)

# 3. KV Cache optimization (during inference)
# - Sliding Window Cache
# - Paged Attention (vLLM)

# 4. Chunk-based processing (long documents)
def process_long_document(model, document, chunk_size=4096, overlap=512):
    """Process long document in chunks"""
    tokens = tokenizer.encode(document)
    results = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        output = model.generate(chunk)
        results.append(output)

    return merge_results(results)
```

---

## References

### Papers
- Beltagy et al. (2020). "Longformer: The Long-Document Transformer"
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases"
- Peng et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models"

### Related Lessons
- [08_LLaMA_Family.md](08_LLaMA_Family.md) - RoPE Basics
- [09_Mistral_MoE.md](09_Mistral_MoE.md) - Sliding Window Attention

---

## Exercises

### Exercise 1: Attention Complexity Analysis

For a sequence of n=8192 tokens with d_model=4096 and 32 attention heads (head_dim=128):

1. Calculate the memory required to store the full attention weight matrix (n × n) in fp16.
2. What is the theoretical peak memory for standard full attention vs FlashAttention? (FlashAttention reduces attention memory from O(n²) to O(n) by operating in tiles.)
3. At what sequence length does the attention weight matrix alone exceed 40GB (a single A100 GPU)?

<details>
<summary>Show Answer</summary>

**1. Full attention weight matrix memory (n=8192):**

For each of 32 heads, attention weights are shape (n × n):
```
32 heads × 8192 × 8192 × 2 bytes (fp16)
= 32 × 67,108,864 × 2
= 4,294,967,296 bytes
≈ 4 GB
```

**2. Peak memory comparison:**

Standard attention materializes the full n×n matrix:
- Attention weights: **~4 GB** (as calculated above)
- Plus softmax, dropout, etc.: roughly 2-3× → ~8-12 GB just for attention

FlashAttention computes attention in tiles (SRAM blocks) without storing the full matrix:
- O(n × d) = 8192 × 128 × 32 × 2 bytes ≈ **64 MB**
- This is ~60-180× less memory for n=8192

**3. When does attention matrix exceed 40GB?**

Memory for attention matrix = 32 × n² × 2 bytes = 64n²

```
64 × n² = 40 × 10⁹
n² = 40 × 10⁹ / 64 = 625,000,000
n = √625,000,000 ≈ 25,000 tokens
```

At about **25K tokens**, the attention weight matrix alone would fill a single A100 GPU — not counting model parameters, activations, or gradients.

</details>

---

### Exercise 2: "Lost in the Middle" Phenomenon

Research has shown that LLMs perform best when relevant information is at the beginning or end of a long context, and worst when it's in the middle.

1. Propose a hypothesis explaining why this phenomenon occurs based on the training process.
2. For a RAG system that retrieves 10 documents to answer a question, suggest two practical mitigation strategies.
3. How does this phenomenon affect agent system design where accumulated tool outputs are placed in context?

<details>
<summary>Show Answer</summary>

**1. Hypothesis for "lost in the middle":**

During training on natural documents, relevant information for predictions tends to appear:
- At the beginning (introductions, premises, context-setting)
- At the end (conclusions, answers, resolutions)

The middle of long documents often contains supporting detail, transitions, and elaboration that is less frequently the direct "answer" target during next-token prediction training. The model's attention mechanisms may implicitly learn to weight early (strong causal position signal in absolute PE) and late tokens more heavily. Additionally, gradient flow during backpropagation may be stronger for positions at the boundaries of long sequences.

**2. RAG mitigation strategies:**

- **Reranking and position-aware ordering**: Place the most relevant retrieved documents at the beginning and end of the context, not in the middle. If you have 10 documents and can estimate relevance, put #1 and #2 at positions 1 and 10, and fill the middle with less relevant documents.

- **Lost-in-the-middle-aware chunking**: Instead of concatenating all documents, use a "sandwich" prompt structure: the question first, then context, then the question again. Or use multiple shorter context windows with aggregation rather than one very long context.

**3. Impact on agent system design:**

Tool outputs accumulated during multi-step agent execution tend to grow the context as the task progresses — with early tool results pushed toward the middle. This means:
- Early important results (e.g., initial search establishing the user's goal) may be "forgotten"
- The agent may over-rely on the most recent tool outputs (at the end) even when earlier results are more relevant
- **Mitigation**: Use a running "scratchpad" or structured state summary that is refreshed at the top of the context each turn, rather than pure context accumulation. Prioritize recency for raw tool outputs but maintain an explicit summary of critical findings from earlier steps.

</details>

---

### Exercise 3: Position Encoding Extension Strategy

A 7B model was trained with RoPE at 4K context. You want to extend it to 32K context without full retraining. Compare these three approaches:

| Approach | Description | Key parameters | Trade-off |
|----------|-------------|---------------|-----------|
| Direct inference at 32K | Just run the model at longer lengths | None | ? |
| Position Interpolation | Scale positions down | Scale factor s = 8 | ? |
| YaRN | NTK-aware interpolation with attention temperature | Scale + attention factor | ? |

<details>
<summary>Show Answer</summary>

| Approach | Description | Key parameters | Trade-off |
|----------|-------------|---------------|-----------|
| **Direct inference at 32K** | Use positions 4097-32768 which were never seen in training | None | Catastrophic — RoPE values at positions >4K are completely out-of-distribution. The attention patterns collapse because the model has no learned behavior for these rotation angles. Performance degrades severely beyond training length. |
| **Position Interpolation** | Scale all positions by s=32K/4K=8 so max position is 4K | s=8 (compression factor) | Works but degrades quality for short sequences: after scaling, positions 1-512 are compressed into 0-64, making it harder to distinguish nearby tokens. Requires some fine-tuning (even 100-1000 steps) to restore quality. Better than direct inference, simpler than YaRN. |
| **YaRN** | Applies different scaling to different frequency components of RoPE; adds attention temperature scaling (1/√s multiplier) | Scale + attention scaling factor | Best quality among the three; specifically addresses the frequency-specific distortion introduced by uniform scaling. The attention temperature prevents the attention distribution from becoming too peaked at long ranges. Minor fine-tuning recommended but often works zero-shot. Preferred method in practice. |

**Practical recommendation:** Use YaRN for the best quality extension; use Position Interpolation if simplicity is preferred. Never use direct inference beyond ~1.5-2× training length.

</details>

---

### Exercise 4: FlashAttention Algorithm Understanding

FlashAttention's key innovation is computing attention in "tiles" that fit in GPU SRAM, avoiding materializing the full O(n²) attention matrix. Explain:

1. Why is the naive computation `Attention(Q,K,V) = softmax(QK^T/√d)V` problematic for memory when n is large?
2. What is the "online softmax" trick that enables FlashAttention to compute exact (not approximate) attention without storing the full matrix?
3. Why does FlashAttention typically run faster than standard attention even though it computes more arithmetic operations?

<details>
<summary>Show Answer</summary>

**1. Memory problem with naive attention:**

The naive implementation materializes these intermediate tensors:
- `S = QK^T`: shape (n × n) — full attention score matrix
- `P = softmax(S)`: shape (n × n) — normalized attention weights
- `O = PV`: shape (n × d) — output

For n=32K and fp16: QK^T alone is 32768 × 32768 × 2 bytes ≈ 2 GB per head, times 32 heads = 64 GB — far exceeding GPU memory.

**2. Online softmax trick:**

Standard softmax requires two passes: first find the maximum (for numerical stability), then compute exp values. FlashAttention uses an incremental (online) update rule:

For each new tile of K,V processed:
```
# New block S_new = Q · K_new^T
# Maintain running max m and sum of exps l
m_new = max(m_old, max(S_new))
l_new = exp(m_old - m_new) * l_old + sum(exp(S_new - m_new))
O_new = (exp(m_old - m_new) * l_old * O_old + exp(S_new - m_new) * V_new) / l_new
```

This allows the exact final softmax to be computed without ever storing the full n×n matrix — only the running statistics (m, l) and the current output (n × d) are needed in memory.

**3. Why FlashAttention is faster despite more arithmetic:**

Standard attention is **memory-bandwidth bound**, not compute bound. The bottleneck is not the floating-point multiplications — it's reading and writing the large intermediate matrices (S, P) to and from HBM (high-bandwidth memory). Even though HBM is fast, reading 2 GB of data per forward pass per head is extremely slow compared to SRAM operations.

FlashAttention keeps all intermediate data in SRAM (fast on-chip cache):
- SRAM bandwidth: ~19 TB/s (vs HBM: ~2 TB/s)
- By computing each tile entirely in SRAM without HBM round-trips, the total data movement to HBM is reduced by O(n) factor
- This wall-clock speedup (2-4×) occurs despite 10-15% more arithmetic operations

</details>

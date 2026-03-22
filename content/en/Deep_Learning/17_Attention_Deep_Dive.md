# 17. Attention Mechanism Deep Dive

[Previous: Attention & Transformer](./16_Attention_Transformer.md) | [Next: Transformer Implementation](./18_Impl_Transformer.md)

---

## Learning Objectives

- In-depth understanding of Multi-Head Attention mathematics
- Attention complexity analysis (O(n^2))
- Flash Attention principles
- Sparse Attention techniques
- Advanced positional encoding (RoPE, ALiBi)
- Attention visualization techniques
- Efficient PyTorch implementation

---

## 1. Multi-Head Attention Mathematics

### Formula Review

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Q = XW_Q  (query)
K = XW_K  (key)
V = XW_V  (value)

X: input (batch, seq_len, d_model)
W_Q, W_K, W_V: learnable weights
```

### Detailed Dimension Analysis

```python
# Input
# X: (batch, seq_len, d_model)
# Example: (32, 512, 768)

# Weight matrices
# W_Q, W_K, W_V: (d_model, d_model)
# Example: (768, 768)

# Q, K, V computation
# Q = X @ W_Q: (32, 512, 768)
# K = X @ W_K: (32, 512, 768)
# V = X @ W_V: (32, 512, 768)

# Multi-Head split
# num_heads = 12, head_dim = 768 / 12 = 64
# Q: (32, 512, 768) → (32, 12, 512, 64)
# K: (32, 512, 768) → (32, 12, 512, 64)
# V: (32, 512, 768) → (32, 12, 512, 64)

# Attention Score
# scores = Q @ K^T: (32, 12, 512, 64) @ (32, 12, 64, 512) = (32, 12, 512, 512)

# Scaled
# scores = scores / sqrt(64) = scores / 8

# Softmax
# weights = softmax(scores): (32, 12, 512, 512)

# Attention Output
# output = weights @ V: (32, 12, 512, 512) @ (32, 12, 512, 64) = (32, 12, 512, 64)

# Concat
# output: (32, 12, 512, 64) → (32, 512, 768)

# Output Projection
# output = output @ W_O: (32, 512, 768)
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Detailed Multi-Head Attention Implementation (⭐⭐⭐)"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Weight matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size, seq_len, _ = query.size()

        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads
        # (batch, seq, d_model) -> (batch, num_heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        # (batch, heads, seq_q, head_dim) @ (batch, heads, head_dim, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 4. Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 6. Apply attention to values
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, head_dim)
        attention_output = torch.matmul(attention_weights, V)

        # 7. Concatenate heads
        # (batch, heads, seq, head_dim) -> (batch, seq, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        # 8. Output projection
        output = self.W_o(attention_output)

        if return_attention:
            return output, attention_weights
        return output
```

---

## 2. Attention Complexity Analysis

### Time Complexity

```
Q @ K^T: O(n * n * d) = O(n^2 * d)
    Q: (n, d), K^T: (d, n) → result: (n, n)

softmax: O(n^2)

weights @ V: O(n^2 * d)
    weights: (n, n), V: (n, d)

Total: O(n^2 * d)
```

### Space Complexity

```
Attention matrix: O(n^2)
    scores: (n, n) storage required

Long sequence problem:
    n = 10,000, heads = 12
    memory = 12 * 10000 * 10000 * 4 bytes = 4.8 GB (float32)
```

### Bottleneck

```python
# Memory bottleneck visualization

n = [1000, 2000, 4000, 8000, 16000]
memory_gb = [h * n_i * n_i * 4 / (1024**3) for n_i in n for h in [12]]

# n=1000:  ~0.048 GB
# n=4000:  ~0.77 GB
# n=8000:  ~3.1 GB
# n=16000: ~12.3 GB

# Cannot process long sequences due to GPU memory limits
```

---

## 3. Flash Attention

### Key Idea

```
Problem: loading entire attention matrix (n x n) into memory → memory explosion

Solution: compute in blocks, utilize SRAM (fast on-chip memory)

1. Split Q, K, V into blocks
2. Compute attention per block (in SRAM)
3. Merge block results with online softmax
```

### Online Softmax

```python
def online_softmax(scores_blocks):
    """Online Softmax Algorithm (⭐⭐⭐⭐)

    Compute and merge softmax per block
    Avoid loading all scores into memory
    """
    # Block 1: [s1, s2, s3]
    # Block 2: [s4, s5, s6]

    # Standard softmax:
    # exp(s1 - max_all) / sum_all_exp

    # Online:
    # Maintain local_max, local_sum per block
    # Update global_max, global_sum when seeing new blocks

    max_so_far = float('-inf')
    sum_so_far = 0

    for block in scores_blocks:
        block_max = block.max()
        new_max = max(max_so_far, block_max)

        # Adjust existing sum
        sum_so_far = sum_so_far * math.exp(max_so_far - new_max)
        # Add new block
        sum_so_far += (block - new_max).exp().sum()

        max_so_far = new_max

    return max_so_far, sum_so_far
```

### PyTorch 2.0+ Flash Attention

```python
# Built-in support from PyTorch 2.0

def flash_attention_example():
    """Flash Attention Usage Example (⭐⭐⭐)"""
    batch_size = 32
    seq_len = 4096
    num_heads = 12
    head_dim = 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

    # PyTorch 2.0+ scaled_dot_product_attention
    # Flash Attention automatically applied (when conditions met)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        output = F.scaled_dot_product_attention(Q, K, V)

    return output

# Memory comparison
# Standard: O(n^2)
# Flash: O(n) - doesn't store attention matrix
```

### Performance Comparison

```python
def benchmark_attention(seq_lens, batch_size=8, num_heads=12, head_dim=64):
    """Attention Performance Benchmark (⭐⭐)"""
    import time

    results = []

    for seq_len in seq_lens:
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

        # Warmup
        _ = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()

        # Timing
        start = time.time()
        for _ in range(100):
            _ = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100

        # Memory
        torch.cuda.reset_peak_memory_stats()
        _ = F.scaled_dot_product_attention(Q, K, V)
        memory = torch.cuda.max_memory_allocated() / 1e9

        results.append({
            'seq_len': seq_len,
            'time_ms': elapsed * 1000,
            'memory_gb': memory
        })

    return results
```

---

## 4. Sparse Attention

### Local Attention

```python
class LocalAttention(nn.Module):
    """Local (Sliding Window) Attention (⭐⭐⭐)

    Each token attends only to nearby window_size tokens
    Complexity: O(n * window_size)
    """
    def __init__(self, d_model, num_heads, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Create mask: outside window is -inf
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1

        return self.attention(x, x, x, mask=mask)
```

### Strided Attention

```python
class StridedAttention(nn.Module):
    """Strided Attention (⭐⭐⭐)

    Attend to tokens at fixed intervals
    Example: stride=4 attends to 0, 4, 8, 12, ...
    """
    def __init__(self, d_model, num_heads, stride=4):
        super().__init__()
        self.stride = stride
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Mask: only attend to stride-spaced tokens
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            indices = list(range(0, seq_len, self.stride))
            mask[i, indices] = 1

        return self.attention(x, x, x, mask=mask)
```

### BigBird Attention

```python
class BigBirdAttention(nn.Module):
    """BigBird Sparse Attention (⭐⭐⭐⭐)

    Combines 3 patterns:
    1. Local: nearby tokens
    2. Global: special tokens (CLS etc) attend to all
    3. Random: random token connections
    """
    def __init__(self, d_model, num_heads, window_size=64, num_global=2, num_random=3):
        super().__init__()
        self.window_size = window_size
        self.num_global = num_global
        self.num_random = num_random
        self.attention = MultiHeadAttention(d_model, num_heads)

    def create_bigbird_mask(self, seq_len, device):
        mask = torch.zeros(seq_len, seq_len, device=device)

        # 1. Local attention
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1

        # 2. Global tokens (first num_global tokens)
        mask[:self.num_global, :] = 1  # global → all
        mask[:, :self.num_global] = 1  # all → global

        # 3. Random connections
        for i in range(seq_len):
            random_indices = torch.randperm(seq_len)[:self.num_random]
            mask[i, random_indices] = 1

        return mask

    def forward(self, x):
        batch, seq_len, _ = x.shape
        mask = self.create_bigbird_mask(seq_len, x.device)
        return self.attention(x, x, x, mask=mask.unsqueeze(0).unsqueeze(0))
```

### Longformer Attention

```python
class LongformerAttention(nn.Module):
    """Longformer Attention (⭐⭐⭐⭐)

    Local + Global tokens
    Global: only specific tokens (CLS, SEP etc) attend globally
    """
    def __init__(self, d_model, num_heads, window_size=256, global_indices=None):
        super().__init__()
        self.window_size = window_size
        self.global_indices = global_indices or [0]  # default: CLS only
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        mask = torch.zeros(seq_len, seq_len, device=x.device)

        # Local
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1

        # Global
        for idx in self.global_indices:
            mask[idx, :] = 1
            mask[:, idx] = 1

        return self.attention(x, x, x, mask=mask)
```

---

## 5. Advanced Positional Encoding

### Sinusoidal (Original Transformer)

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (⭐⭐)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### Learned Position Embeddings

```python
class LearnedPositionalEncoding(nn.Module):
    """Learnable Positional Encoding (⭐⭐)"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
```

### RoPE (Rotary Position Embedding)

```python
class RotaryPositionalEmbedding(nn.Module):
    """RoPE - Used in LLaMA, GPT-NeoX etc (⭐⭐⭐⭐)

    Encode position information via rotation matrices
    Naturally represents relative position information
    """
    def __init__(self, dim, max_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

        # Caching
        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, q, k, seq_len):
        # q, k: (batch, heads, seq, head_dim)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(self, x, cos, sin):
        # x: (batch, heads, seq, head_dim)
        # Split x in half and rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Apply rotation
        rotated = torch.stack(
            [-x2, x1], dim=-1
        ).flatten(-2)

        return x * cos + rotated * sin


def apply_rope(q, k, rope_module, seq_len):
    """RoPE Application Example"""
    return rope_module(q, k, seq_len)
```

### ALiBi (Attention with Linear Biases)

```python
class ALiBiPositionalBias(nn.Module):
    """ALiBi - Used in MPT, BLOOM etc (⭐⭐⭐⭐)

    Add linear bias to attention scores instead of position embeddings
    No learnable parameters
    Excellent length extrapolation capability
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Different slope per head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

    def _get_slopes(self, n):
        """Compute slopes: 2^(-8/n), 2^(-16/n), ..."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            closest_power = 2 ** math.floor(math.log2(n))
            return torch.tensor(
                get_slopes_power_of_2(closest_power) +
                get_slopes_power_of_2(2 * closest_power)[0::2][:n - closest_power]
            )

    def forward(self, seq_len, device):
        # Relative distance matrix
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.abs().unsqueeze(0)

        # bias = -slope * distance
        alibi = -self.slopes.unsqueeze(1).unsqueeze(1) * relative_positions

        return alibi  # (num_heads, seq_len, seq_len)


def attention_with_alibi(Q, K, V, alibi_bias):
    """Attention with ALiBi (⭐⭐⭐)"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Add ALiBi bias
    scores = scores + alibi_bias

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

---

## 6. Attention Visualization

### Visualizing Attention Weights

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens=None, layer=0, head=0):
    """Visualize single head attention (⭐⭐)"""
    # attention_weights: (batch, heads, seq, seq)
    weights = attention_weights[0, head].cpu().detach().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap='Blues', square=True)

    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens, rotation=0)

    plt.title(f'Attention Weights (Layer {layer}, Head {head})')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    plt.savefig(f'attention_layer{layer}_head{head}.png')
    plt.close()


def visualize_all_heads(attention_weights, tokens=None, ncols=4):
    """Visualize all heads (⭐⭐)"""
    num_heads = attention_weights.size(1)
    nrows = (num_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()

    for head in range(num_heads):
        weights = attention_weights[0, head].cpu().detach().numpy()
        ax = axes[head]
        im = ax.imshow(weights, cmap='Blues')
        ax.set_title(f'Head {head}')
        ax.axis('off')

    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('attention_all_heads.png')
    plt.close()
```

### Attention Flow (BertViz Style)

```python
def plot_attention_flow(attention_weights, tokens, top_k=3):
    """Visualize top K attention connections (⭐⭐⭐)"""
    weights = attention_weights[0, 0].cpu().detach().numpy()
    seq_len = len(tokens)

    plt.figure(figsize=(12, 8))

    # Token positions
    left_positions = [(0, i) for i in range(seq_len)]
    right_positions = [(1, i) for i in range(seq_len)]

    # Display tokens
    for i, token in enumerate(tokens):
        plt.text(-0.1, i, token, ha='right', va='center', fontsize=10)
        plt.text(1.1, i, token, ha='left', va='center', fontsize=10)

    # Top K connections
    for i in range(seq_len):
        top_indices = weights[i].argsort()[-top_k:]
        for j in top_indices:
            alpha = weights[i, j] / weights[i].max()
            plt.plot([0, 1], [i, j], 'b-', alpha=alpha, linewidth=2)

    plt.xlim(-0.5, 1.5)
    plt.ylim(-1, seq_len)
    plt.axis('off')
    plt.title('Attention Flow')
    plt.savefig('attention_flow.png')
    plt.close()
```

---

## 7. Efficient Implementation

### PyTorch scaled_dot_product_attention

```python
def efficient_attention_comparison():
    """Efficient attention implementation comparison (⭐⭐⭐)"""
    import time

    batch, heads, seq, dim = 8, 12, 2048, 64

    Q = torch.randn(batch, heads, seq, dim, device='cuda')
    K = torch.randn(batch, heads, seq, dim, device='cuda')
    V = torch.randn(batch, heads, seq, dim, device='cuda')

    # 1. Naive implementation
    def naive_attention(Q, K, V):
        scale = math.sqrt(Q.size(-1))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    # 2. PyTorch built-in (may use Flash Attention)
    def pytorch_attention(Q, K, V):
        return F.scaled_dot_product_attention(Q, K, V)

    # Warmup
    _ = naive_attention(Q, K, V)
    _ = pytorch_attention(Q, K, V)
    torch.cuda.synchronize()

    # Benchmark naive
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(10):
        _ = naive_attention(Q, K, V)
    torch.cuda.synchronize()
    naive_time = (time.time() - start) / 10
    naive_mem = torch.cuda.max_memory_allocated() / 1e9

    # Benchmark pytorch
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(10):
        _ = pytorch_attention(Q, K, V)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 10
    pytorch_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"Naive:   {naive_time*1000:.2f}ms, {naive_mem:.2f}GB")
    print(f"PyTorch: {pytorch_time*1000:.2f}ms, {pytorch_mem:.2f}GB")

    return naive_time, pytorch_time
```

### Memory-Efficient Attention

```python
class MemoryEfficientAttention(nn.Module):
    """Memory Efficient Attention (chunk-based) (⭐⭐⭐⭐)"""
    def __init__(self, d_model, num_heads, chunk_size=256):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape

        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Process in chunks to save memory
        outputs = []
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            Q_chunk = Q[:, :, i:end]

            # Attention for this chunk against all K, V
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self.scale
            weights = F.softmax(scores, dim=-1)
            chunk_out = torch.matmul(weights, V)
            outputs.append(chunk_out)

        output = torch.cat(outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(B, N, self.d_model)

        return self.W_o(output)
```

---

## 8. Attention Pattern Analysis

### Common Observed Patterns

```python
def analyze_attention_patterns(attention_weights):
    """Attention Pattern Analysis (⭐⭐⭐)"""
    # attention_weights: (batch, heads, seq, seq)

    patterns = {}

    for head in range(attention_weights.size(1)):
        weights = attention_weights[0, head].cpu().detach()

        # 1. Diagonal (self-attention)
        diagonal = torch.diag(weights).mean()

        # 2. Previous token
        prev_token = torch.diag(weights, -1).mean() if weights.size(0) > 1 else 0

        # 3. First token (CLS etc)
        first_token = weights[:, 0].mean()

        # 4. Uniform (entropy)
        uniform_entropy = -(weights * weights.log()).sum(dim=-1).mean()

        patterns[f'head_{head}'] = {
            'diagonal': diagonal.item(),
            'prev_token': prev_token.item() if isinstance(prev_token, torch.Tensor) else prev_token,
            'first_token': first_token.item(),
            'entropy': uniform_entropy.item()
        }

    return patterns
```

---

## Summary

### Key Concepts

1. **MHA Mathematics**: Q, K, V projection → scaling → softmax → weighted sum
2. **Complexity**: O(n^2) time/space - bottleneck for long sequences
3. **Flash Attention**: Block processing + online softmax → O(n) memory
4. **Sparse Attention**: Local, Strided, BigBird etc
5. **Positional Encoding**: Sinusoidal, Learned, RoPE, ALiBi

### Efficiency Summary

| Technique | Time | Space | Features |
|-----|------|------|------|
| Standard | O(n^2) | O(n^2) | Baseline |
| Flash | O(n^2) | O(n) | HW optimized |
| Local | O(nw) | O(nw) | window w |
| Sparse | O(n sqrt(n)) | O(n) | Pattern combination |

### PyTorch Practical Tips

```python
# 1. Use PyTorch 2.0+
output = F.scaled_dot_product_attention(Q, K, V)

# 2. Force Flash Attention activation
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V)

# 3. Consider chunking for long sequences

# 4. RoPE/ALiBi favorable for length extrapolation
```

---

## Exercises

### Exercise 1: Attention Memory Scaling

Empirically measure how attention memory grows with sequence length.

1. Write a function that allocates Q, K, V tensors of shape `(1, 1, seq_len, 64)` in float32 on CPU.
2. Run standard attention (`Q @ K.T / sqrt(d_k)` followed by softmax and `weights @ V`) for `seq_len` values of 512, 1024, 2048, 4096.
3. Use Python's `tracemalloc` or `torch.cuda.max_memory_allocated()` (if CUDA available) to record peak memory at each length.
4. Plot memory vs. sequence length on a log-log scale and confirm the O(n²) trend.
5. Explain at what sequence length a single A100 GPU (80 GB) would run out of memory for 12-head MHA in float16.

### Exercise 2: Online Softmax Verification

Verify that the online softmax algorithm from the lesson produces identical results to standard softmax.

1. Create a vector of scores with 100 elements.
2. Compute standard softmax: `exp(scores - max) / sum(exp(scores - max))`.
3. Implement the online softmax algorithm that processes the scores in 4 blocks of 25.
4. Compare the outputs element-by-element and confirm they match to within `1e-6`.
5. Explain why numerical stability (subtracting the max) is required and what happens without it.

### Exercise 3: Attention Pattern Analysis

Analyze what patterns different attention heads learn in a pretrained model.

1. Load a small pretrained Transformer from `transformers` (e.g., `distilbert-base-uncased`).
2. Pass 3 different sentences (a declarative, a question, and a list) through the model.
3. Extract and visualize the 6 attention heads from layer 2 for each sentence.
4. For each head, identify whether it shows diagonal (local), first-token, or uniform pattern using the `analyze_attention_patterns` function from the lesson.
5. Report which heads behave differently across sentence types and hypothesize what each head specializes in.

### Exercise 4: RoPE vs Sinusoidal Encoding

Compare the positional encoding approaches on a length-extrapolation task.

1. Build two small Transformers (2 layers, `d_model=128`): one with sinusoidal encoding and one with RoPE.
2. Train both on sequences of length 64 using a copy task (predict input tokens at the output).
3. Evaluate both models on sequences of length 128 (out-of-distribution length).
4. Report accuracy at both lengths and explain why RoPE generalizes better to unseen lengths.

### Exercise 5: Local Attention vs Full Attention Trade-off

Implement and compare `LocalAttention` and full attention on a long sequence task.

1. Generate a long sequence task with `seq_len=2048`: the label is the token at position 0 (global information).
2. Train a 2-layer Transformer using full attention and another using `LocalAttention(window_size=128)`.
3. Measure training time per epoch and peak memory usage for both.
4. Compare accuracy after 20 epochs.
5. Explain why the local attention model struggles with this specific task and how BigBird's global tokens would solve the problem.

---

## References

- Flash Attention: https://arxiv.org/abs/2205.14135
- RoPE: https://arxiv.org/abs/2104.09864
- ALiBi: https://arxiv.org/abs/2108.12409
- BigBird: https://arxiv.org/abs/2007.14062

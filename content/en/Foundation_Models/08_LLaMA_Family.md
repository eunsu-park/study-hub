# LLaMA Family

## Learning Objectives
- Understand the architectural evolution of LLaMA 1/2/3
- Master core technologies: RoPE, RMSNorm, SwiGLU
- Grasp the Grouped Query Attention (GQA) mechanism
- Learn practical LLaMA usage methods

---

## 1. LLaMA Overview

### 1.1 Significance of LLaMA

**LLaMA** (Large Language Model Meta AI) is an open-source LLM released by Meta in 2023, leading the democratization of Foundation Model research.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Historical Significance of LLaMA              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Before LLaMA (2022):                                           │
│  • Best-performing models only available via API (GPT-3.5, PaLM)│
│  • Academic research models lacked performance                  │
│  • Limited LLM access for open-source community                 │
│                                                                 │
│  After LLaMA (2023):                                            │
│  • Researchers can directly experiment with state-of-the-art    │
│  • Explosive growth of derivative models (Alpaca, Vicuna, etc.) │
│  • Rapid acceleration of LLM research                           │
│                                                                 │
│  Key Contributions:                                              │
│  • Applied Chinchilla rules (D=20N or more)                     │
│  • Validated efficient architecture choices                      │
│  • Published training data composition                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Version Comparison

| Feature | LLaMA 1 | LLaMA 2 | LLaMA 3 | LLaMA 3.1 | LLaMA 3.2 |
|---------|---------|---------|---------|-----------|-----------|
| Release | 2023.02 | 2023.07 | 2024.04 | 2024.07 | 2024.09 |
| Sizes | 7/13/33/65B | 7/13/70B | 8/70B | 8/70/405B | 1/3/11/90B |
| Tokens | 1.4T | 2T | 15T+ | 15T+ | 15T+ |
| Context | 2K | 4K | 8K | 128K | 128K |
| License | Research | Commercial (conditional) | Commercial (relaxed) | Commercial (relaxed) | Commercial (relaxed) |
| GQA | ❌ | ✅ (70B) | ✅ (all) | ✅ (all) | ✅ (all) |
| Features | Base architecture | RLHF, Safety | Improved reasoning | 128K native, Tool Use | Vision models added |

> **LLaMA 3.1/3.2 Major Updates** (2024):
> - **LLaMA 3.1**: 128K native context, 405B flagship model, Tool Use capability
> - **LLaMA 3.2**: Lightweight models (1B/3B) and Vision models (11B/90B) added

---

## 2. LLaMA Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLaMA Architecture                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Tokens                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │         Token Embedding                 │                    │
│  └─────────────────────────────────────────┘                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │     RoPE (Rotary Position Embedding)    │  ← Position info   │
│  └─────────────────────────────────────────┘                    │
│       │                                                         │
│  ┌────┴────────────────────────────────────┐                    │
│  │         Transformer Block × N           │                    │
│  │  ┌───────────────────────────────────┐  │                    │
│  │  │  RMSNorm (Pre-normalization)      │  │  ← Replaces LN     │
│  │  │            ↓                      │  │                    │
│  │  │  Grouped Query Attention (GQA)    │  │  ← KV Cache eff.   │
│  │  │            ↓                      │  │                    │
│  │  │  Residual Connection              │  │                    │
│  │  │            ↓                      │  │                    │
│  │  │  RMSNorm                          │  │                    │
│  │  │            ↓                      │  │                    │
│  │  │  SwiGLU FFN                       │  │  ← Replaces GELU   │
│  │  │            ↓                      │  │                    │
│  │  │  Residual Connection              │  │                    │
│  │  └───────────────────────────────────┘  │                    │
│  └─────────────────────────────────────────┘                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │         RMSNorm → Linear → Vocab        │                    │
│  └─────────────────────────────────────────┘                    │
│       │                                                         │
│       ▼                                                         │
│  Output Logits                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Hyperparameters

```python
"""
LLaMA Model Specification Comparison
"""

LLAMA_CONFIGS = {
    "llama-7b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,  # MHA (no GQA)
        "vocab_size": 32000,
        "ffn_dim": 11008,  # approx 2.67 × dim
        "context_length": 2048,
    },
    "llama-13b": {
        "dim": 5120,
        "n_layers": 40,
        "n_heads": 40,
        "n_kv_heads": 40,
        "vocab_size": 32000,
        "ffn_dim": 13824,
        "context_length": 2048,
    },
    "llama-70b": {
        "dim": 8192,
        "n_layers": 80,
        "n_heads": 64,
        "n_kv_heads": 8,  # GQA! 8 KV heads
        "vocab_size": 32000,
        "ffn_dim": 28672,
        "context_length": 4096,
    },
    "llama3-8b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,  # GQA
        "vocab_size": 128256,  # Extended vocab
        "ffn_dim": 14336,
        "context_length": 8192,
    },
    "llama3-70b": {
        "dim": 8192,
        "n_layers": 80,
        "n_heads": 64,
        "n_kv_heads": 8,  # GQA
        "vocab_size": 128256,
        "ffn_dim": 28672,
        "context_length": 8192,
    },
    # LLaMA 3.1 (2024.07)
    "llama3.1-8b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim": 14336,
        "context_length": 131072,  # 128K native
    },
    "llama3.1-405b": {
        "dim": 16384,
        "n_layers": 126,
        "n_heads": 128,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim": 53248,
        "context_length": 131072,  # 128K native
    },
    # LLaMA 3.2 (2024.09) - Lightweight text models
    "llama3.2-1b": {
        "dim": 2048,
        "n_layers": 16,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim": 8192,
        "context_length": 131072,
    },
    "llama3.2-3b": {
        "dim": 3072,
        "n_layers": 28,
        "n_heads": 24,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim": 8192,
        "context_length": 131072,
    },
}
```

---

## 3. RoPE (Rotary Position Embedding)

### 3.1 Concept

**RoPE** is a method that encodes positional information using rotation matrices.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Position Encoding Comparison                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Sinusoidal (Original Transformer)                           │
│     PE(pos, 2i) = sin(pos / 10000^(2i/d))                       │
│     PE(pos, 2i+1) = cos(pos / 10000^(2i/d))                     │
│     → Added to input (additive)                                 │
│     → Weak relative position information                        │
│                                                                 │
│  2. Learned (BERT, GPT)                                         │
│     PE = Embedding(position)                                    │
│     → Learned vectors                                           │
│     → Difficult to generalize beyond trained lengths            │
│                                                                 │
│  3. RoPE (LLaMA)                                                │
│     R(θ) = rotation matrix, θ = f(position)                     │
│     q' = R(θ_m) × q, k' = R(θ_n) × k                           │
│     q' · k' = q · k × cos(θ_m - θ_n)                           │
│     → Naturally encodes relative positions                      │
│     → Length extrapolation possible (with modifications)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Mathematical Understanding

```python
import torch
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute complex frequencies for RoPE

    Args:
        dim: Embedding dimension (head_dim)
        seq_len: Maximum sequence length
        theta: Base frequency (10000)

    Returns:
        freqs_cis: (seq_len, dim//2) complex tensor
    """
    # Frequency calculation: θ_i = 1 / (theta^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Angle per position: m * θ_i
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    # Complex form: e^(i*θ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply RoPE to Query and Key

    Args:
        xq: Query (batch, seq_len, n_heads, head_dim)
        xk: Key (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: Precomputed complex frequencies

    Returns:
        Rotated Query and Key
    """
    # Convert real to complex (pair adjacent elements)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Apply rotation (complex multiplication)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Example
batch, seq_len, n_heads, head_dim = 2, 128, 32, 128
xq = torch.randn(batch, seq_len, n_heads, head_dim)
xk = torch.randn(batch, seq_len, n_heads, head_dim)
freqs_cis = precompute_freqs_cis(head_dim, seq_len)

xq_rope, xk_rope = apply_rotary_emb(xq, xk, freqs_cis)
print(f"Output shape: {xq_rope.shape}")  # (2, 128, 32, 128)
```

### 3.3 Advantages of RoPE

```python
"""
Advantages of RoPE:

1. Natural relative position encoding
   - q_m · k_n ∝ cos(θ_m - θ_n)
   - Depends on relative distance, not absolute position

2. Extrapolation capability
   - Can extend beyond trained lengths
   - (Performance degrades → improved with NTK, YaRN)

3. Efficiency
   - No additional parameters
   - Fast element-wise operations

4. Compatible with linear Self-attention
   - Can combine with some efficient attention methods
"""
```

---

## 4. RMSNorm

### 4.1 Concept

**RMSNorm** is a simplified version of LayerNorm that removes mean computation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LayerNorm vs RMSNorm                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LayerNorm:                                                     │
│  ──────────────────────────────────                             │
│  μ = mean(x)                                                    │
│  σ = std(x)                                                     │
│  y = γ × (x - μ) / σ + β                                        │
│                                                                 │
│  • Subtract mean + divide by variance                           │
│  • Learnable scale(γ) and shift(β)                              │
│                                                                 │
│  RMSNorm:                                                       │
│  ──────────────────────────────────                             │
│  RMS(x) = sqrt(mean(x^2))                                       │
│  y = γ × x / RMS(x)                                             │
│                                                                 │
│  • No mean subtraction → Re-centering removed                   │
│  • Scale by RMS only                                            │
│  • No shift(β)                                                  │
│  • Reduced computation, similar performance                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Paper: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # scale parameter γ

    def _norm(self, x):
        # RMS = sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # output = γ × (x / RMS(x))
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Compare with LayerNorm
x = torch.randn(2, 10, 512)

layer_norm = nn.LayerNorm(512)
rms_norm = RMSNorm(512)

# Computation time comparison (RMSNorm is slightly faster)
import time

start = time.time()
for _ in range(1000):
    _ = layer_norm(x)
print(f"LayerNorm: {time.time() - start:.4f}s")

start = time.time()
for _ in range(1000):
    _ = rms_norm(x)
print(f"RMSNorm: {time.time() - start:.4f}s")
```

---

## 5. SwiGLU

### 5.1 Concept

**SwiGLU** is a variant of GLU (Gated Linear Unit) that uses the Swish activation function.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FFN Activation Function Comparison            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ReLU FFN (Original Transformer):                            │
│     FFN(x) = max(0, xW₁ + b₁)W₂ + b₂                            │
│     • Simple but loses information in negative region           │
│                                                                 │
│  2. GELU FFN (BERT, GPT):                                       │
│     FFN(x) = GELU(xW₁)W₂                                        │
│     GELU(x) = x × Φ(x)  (Φ = CDF of normal)                     │
│     • Smooth activation, improved performance                   │
│                                                                 │
│  3. SwiGLU FFN (LLaMA):                                         │
│     FFN(x) = (Swish(xW₁) ⊙ xV)W₂                                │
│     Swish(x) = x × σ(x)  (σ = sigmoid)                          │
│     ⊙ = element-wise multiplication                             │
│                                                                 │
│     • Gating mechanism controls information flow                │
│     • More parameters, better performance                       │
│     • 2/3 × 4d hidden dim (maintains parameter count)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit

    FFN(x) = (Swish(xW₁) ⊙ xV) W₂

    Paper: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()

        # hidden_dim calculation: 2/3 × 4d, rounded to multiple of 256
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up projection

    def forward(self, x):
        # SwiGLU: Swish(xW₁) ⊙ (xW₃) → W₂
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Compare with standard FFN
class StandardFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

# Parameter count comparison
dim = 4096
swiglu = SwiGLU(dim)  # 3 linear layers: dim→hidden, dim→hidden, hidden→dim
standard = StandardFFN(dim)  # 2 linear layers: dim→4*dim, 4*dim→dim

print(f"SwiGLU params: {sum(p.numel() for p in swiglu.parameters()):,}")
print(f"Standard FFN params: {sum(p.numel() for p in standard.parameters()):,}")
# SwiGLU has slightly more but adjusted with hidden_dim to be similar
```

---

## 6. Grouped Query Attention (GQA)

### 6.1 Concept

**GQA** is an intermediate form between Multi-Head Attention and Multi-Query Attention.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Attention Method Comparison                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Multi-Head Attention (MHA):                                 │
│     Q heads: 32  │  K heads: 32  │  V heads: 32                 │
│     • Each Q head uses independent KV head                      │
│     • Memory: High (32 × KV cache)                              │
│                                                                 │
│  2. Multi-Query Attention (MQA):                                │
│     Q heads: 32  │  K heads: 1   │  V heads: 1                  │
│     • All Q heads share same KV                                 │
│     • Memory: Minimum (1 × KV cache)                            │
│     • Quality: Slightly lower than MHA                          │
│                                                                 │
│  3. Grouped Query Attention (GQA):                              │
│     Q heads: 32  │  K heads: 8   │  V heads: 8                  │
│     • Q heads divided into groups sharing KV                    │
│     • Example: 4 Q heads share 1 KV head                        │
│     • Memory: Medium (8 × KV cache)                             │
│     • Quality: Nearly identical to MHA                          │
│                                                                 │
│  ┌──────────────────────────────────────────────┐               │
│  │ MHA          │ MQA           │ GQA           │               │
│  │ Q Q Q Q Q Q  │ Q Q Q Q Q Q   │ Q Q│Q Q│Q Q   │               │
│  │ ↓ ↓ ↓ ↓ ↓ ↓  │ ↓ ↓ ↓ ↓ ↓ ↓   │ ↓ ↓│↓ ↓│↓ ↓   │               │
│  │ K K K K K K  │     K         │  K │ K │ K    │               │
│  │ V V V V V V  │     V         │  V │ V │ V    │               │
│  └──────────────────────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    Paper: https://arxiv.org/abs/2305.13245
    """
    def __init__(
        self,
        dim: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,  # Number of KV heads (< n_heads)
        head_dim: int = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or dim // n_heads

        # Verify Q heads > KV heads
        assert n_heads % n_kv_heads == 0
        self.n_rep = n_heads // n_kv_heads  # Number of Q heads per KV head

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, mask=None, kv_cache=None):
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE (if available)
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV Cache handling (during inference)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)

        # Expand KV heads: n_kv_heads → n_heads
        # (batch, seq, n_kv_heads, head_dim) → (batch, seq, n_heads, head_dim)
        xk = self._repeat_kv(xk)
        xv = self._repeat_kv(xv)

        # Compute Attention
        xq = xq.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, xv)

        # Combine results
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output), (xk, xv)

    def _repeat_kv(self, x):
        """Repeat KV heads to match Q heads count"""
        if self.n_rep == 1:
            return x
        batch, seq_len, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, self.n_rep, head_dim)
        return x.reshape(batch, seq_len, n_kv_heads * self.n_rep, head_dim)

# Memory usage comparison
def compare_kv_cache_memory(seq_len, batch_size=1, dtype_bytes=2):
    """Compare KV cache memory (FP16 basis)"""
    configs = {
        "LLaMA-70B (MHA)": {"n_layers": 80, "n_kv_heads": 64, "head_dim": 128},
        "LLaMA-70B (GQA)": {"n_layers": 80, "n_kv_heads": 8, "head_dim": 128},
    }

    for name, cfg in configs.items():
        kv_mem = (2 *  # K and V
                  batch_size *
                  cfg["n_layers"] *
                  seq_len *
                  cfg["n_kv_heads"] *
                  cfg["head_dim"] *
                  dtype_bytes)
        print(f"{name}: {kv_mem / 1e9:.2f} GB for {seq_len} tokens")

compare_kv_cache_memory(4096)
# LLaMA-70B (MHA): 5.24 GB for 4096 tokens
# LLaMA-70B (GQA): 0.66 GB for 4096 tokens  ← 8x savings!
```

---

## 7. LLaMA Practice

### 7.1 Using with HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLaMA 2 7B
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Text generation
def generate_text(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage example
prompt = "Explain the concept of machine learning in simple terms:"
response = generate_text(prompt)
print(response)
```

### 7.2 Efficient Usage with Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,     # Double quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Check memory usage
print(f"Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")
# Approximately 4GB (75% savings compared to FP16)
```

### 7.3 Using LLaMA 3

```python
# LLaMA 3 8B (128K tokenizer)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # bfloat16 recommended for LLaMA 3
    device_map="auto"
)

# LLaMA 3 features:
# - 128K tokenizer (more efficient)
# - 8K base context (extendable to 128K)
# - Improved reasoning capability

prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 8. LLaMA 3.1/3.2 Details

### 8.1 LLaMA 3.1 (July 2024)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLaMA 3.1 Key Features                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 128K Native Context                                         │
│     • Supports 128K tokens from training                        │
│     • Handles long context without RoPE scaling                 │
│                                                                 │
│  2. 405B Flagship Model                                         │
│     • GPT-4 level performance                                   │
│     • 126 layers, 16K embedding dimension                       │
│                                                                 │
│  3. Tool Use Capability                                         │
│     • Function Calling                                          │
│     • Code Interpreter                                          │
│     • Search tool integration                                   │
│                                                                 │
│  4. Enhanced Multilingual Support                               │
│     • English, German, French, Italian                          │
│     • Portuguese, Hindi, Spanish, Thai                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
# LLaMA 3.1 Tool Use Example
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Tool Use format (LLaMA 3.1 specialized)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What's the weather in Seoul?"}
]

# Generate tool call
inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### 8.2 LLaMA 3.2 (September 2024)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLaMA 3.2 Model Lineup                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Lightweight Text Models (on-device optimized):                 │
│  ┌─────────────────────────────────────────────┐                │
│  │  LLaMA 3.2 1B: For mobile/edge devices      │                │
│  │  LLaMA 3.2 3B: For lightweight applications │                │
│  └─────────────────────────────────────────────┘                │
│                                                                 │
│  Vision Models (multimodal):                                    │
│  ┌─────────────────────────────────────────────┐                │
│  │  LLaMA 3.2 11B-Vision: Image understanding  │                │
│  │  LLaMA 3.2 90B-Vision: High-performance     │                │
│  └─────────────────────────────────────────────┘                │
│                                                                 │
│  Features:                                                       │
│  • 1B/3B: 128K context, on-device inference capable             │
│  • 11B/90B: Vision encoder integrated, image+text processing    │
│  • Optimized for Qualcomm, MediaTek hardware                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
# LLaMA 3.2 Vision Example
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Load image
url = "https://example.com/image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Vision conversation
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do you see in this image?"}
        ]
    }
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0]))
```

---

## Summary

### LLaMA Core Technologies
| Technology | Effect |
|------------|--------|
| **RoPE** | Relative position encoding, length extension possible |
| **RMSNorm** | Faster and more effective than LayerNorm |
| **SwiGLU** | Better performance than GELU |
| **GQA** | 8x KV cache memory savings |

### Practical Recommendations
1. **7B/8B**: Single GPU (16GB+), for quick experiments
2. **13B**: 24GB GPU, balanced choice
3. **70B**: Multiple GPUs, when top performance needed
4. **Quantization**: 75% memory savings with 4-bit

### Next Steps
- [09_Mistral_MoE.md](09_Mistral_MoE.md): Mixture of Experts architecture
- [19_PEFT_Unified.md](19_PEFT_Unified.md): LLaMA Fine-tuning (LoRA)

---

## References

### Core Papers
- Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
- Touvron et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"
- Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models"

### Code & Resources
- [LLaMA GitHub](https://github.com/facebookresearch/llama)
- [HuggingFace LLaMA](https://huggingface.co/meta-llama)
- [LLaMA 3 Recipes](https://github.com/meta-llama/llama-recipes)

---

## Exercises

### Exercise 1: RoPE vs Absolute Position Encoding

Compare Rotary Position Encoding (RoPE) with learned absolute position encoding on the following dimensions:

| Dimension | Absolute PE | RoPE |
|-----------|------------|------|
| Extrapolation beyond training length | ? | ? |
| How position is encoded | ? | ? |
| Relative distance representation | ? | ? |
| Memory cost | ? | ? |

<details>
<summary>Show Answer</summary>

| Dimension | Absolute PE | RoPE |
|-----------|------------|------|
| **Extrapolation beyond training length** | Poor — positions beyond the trained maximum have no learned embedding, causing degraded performance | Better — RoPE's mathematical structure (rotation in complex plane) generalizes more gracefully to longer sequences; can be further extended with YaRN or LongRoPE scaling |
| **How position is encoded** | Adds a learned or fixed vector to each token embedding (position information is in the embedding space) | Multiplies query and key vectors by a rotation matrix based on position (position information is encoded in the attention dot-product relationship) |
| **Relative distance representation** | Indirect — relative distance is computed from absolute positions, requiring the model to learn this relationship | Direct — RoPE encodes relative distance explicitly: the dot product between RoPE-rotated Q and K depends only on their relative position, not absolute positions |
| **Memory cost** | Requires storing learned embedding matrix of size (max_len × dim) | No additional parameters — computed on-the-fly via rotation matrices |

The key practical advantage of RoPE: `Q_m · K_n ∝ f(m-n)` — relative distance is "baked in" to the attention computation, making it easier for the model to learn position-independent patterns.

</details>

---

### Exercise 2: GQA Memory Savings Calculation

LLaMA 2 70B uses Grouped Query Attention (GQA) with 64 query heads and 8 KV heads.

1. How much memory does the KV cache require per token per layer for this model (assuming fp16, head_dim=128)?
2. If Multi-Head Attention (MHA) were used instead (64 KV heads), how much more memory would the KV cache require?
3. For a batch of 32 sequences, each 4096 tokens long, with 80 transformer layers, what is the total KV cache memory saving from GQA vs MHA?

<details>
<summary>Show Answer</summary>

**1. KV cache with GQA (8 KV heads):**
```
Per token per layer = 2 (K+V) × 8 KV heads × 128 head_dim × 2 bytes (fp16)
= 2 × 8 × 128 × 2 = 4,096 bytes = 4 KB
```

**2. KV cache with MHA (64 KV heads):**
```
Per token per layer = 2 × 64 × 128 × 2 = 32,768 bytes = 32 KB
```
MHA requires 8× more KV cache than GQA.

**3. Total savings for batch=32, seq_len=4096, 80 layers:**

GQA total: 4 KB × 32 × 4096 × 80 = 4,096 × 10,485,760 bytes ≈ **40.96 GB**
MHA total: 32 KB × 32 × 4096 × 80 = 32,768 × 10,485,760 bytes ≈ **327.68 GB**

**Savings: ~286.72 GB** — GQA enables serving the 70B model at this batch/sequence size on far fewer GPUs than MHA would require.

</details>

---

### Exercise 3: SwiGLU vs GELU Analysis

LLaMA uses SwiGLU instead of GELU as the activation function. SwiGLU computes:

```python
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
```

1. What does the element-wise multiplication `⊙` (gating) accomplish that a standard activation function cannot?
2. Why might a gating mechanism be beneficial for a language model's feed-forward layers, which act as "key-value memories"?
3. SwiGLU requires two weight matrices (W and V) instead of one. If the original FFN had hidden_dim=4×d_model, how does LLaMA adjust the hidden dimension to keep parameter count similar?

<details>
<summary>Show Answer</summary>

**1. What gating accomplishes:**
The element-wise multiplication creates a **conditional computation** mechanism: the gate `Swish(xW)` acts as a learnable filter that can "turn off" or scale down specific dimensions of the value projection `(xV)`. This allows the network to be selective about which information flows through — effectively implementing input-dependent feature selection. A standard activation function applies the same nonlinearity regardless of context.

**2. Why gating benefits FFN "key-value memories":**
In the key-value memory interpretation, each FFN hidden unit corresponds to a "key" pattern and a "value" to write. The gate allows the network to dynamically suppress irrelevant memories based on the current input. For language models processing diverse contexts (code, reasoning, factual recall), being able to selectively activate relevant knowledge stores is more expressive than always applying the full hidden layer uniformly.

**3. Dimension adjustment:**
Standard FFN: 2 matrices of shape (d_model × 4·d_model) → total params = 2 × d × 4d = 8d²

SwiGLU FFN: 3 matrices (W, V, and output projection) each of shape (d_model × h) → total = 3 × d × h

To match: 3dh ≈ 8d² → h ≈ 8d/3 ≈ 2.67×d_model

LLaMA sets the FFN hidden dimension to ~(8/3)×d_model (rounded to a multiple of 256 for hardware efficiency), so instead of 4× expansion it uses ~2.67× expansion with 3 matrices — approximately the same total parameter count.

</details>

---

### Exercise 4: LLaMA Version Selection

A researcher has the following use cases. For each, recommend the most appropriate LLaMA version and size and explain your reasoning.

1. Running inference on a Raspberry Pi 5 with 8GB RAM for a smart home assistant.
2. Fine-tuning on a medical QA dataset with a single A100 80GB GPU for a hospital chatbot.
3. Deploying a general-purpose chat assistant on a server with 4× H100 80GB GPUs.
4. Processing 50K-token legal documents for contract analysis.

<details>
<summary>Show Answer</summary>

**1. Raspberry Pi / edge device (8GB RAM):**
- **LLaMA 3.2 1B** — The 1B model is specifically designed for on-device inference on mobile/edge hardware (optimized for Qualcomm, MediaTek chips). With 4-bit quantization, ~0.5GB memory footprint. LLaMA 3.2 was explicitly designed for edge deployment.

**2. Single A100 80GB, medical fine-tuning:**
- **LLaMA 3 8B** with QLoRA — The 8B model fits comfortably in fp16 (16GB), leaving ample room for fine-tuning memory overhead. QLoRA (4-bit base + LoRA adapters) reduces memory to ~6GB, enabling larger batch sizes. LLaMA 3 8B has better base performance than LLaMA 2 for domain adaptation.

**3. 4× H100 server, general chat:**
- **LLaMA 3.1 70B** — The 70B model in bf16 requires ~140GB, which fits across 4×H100 with tensor parallelism. This size provides GPT-4-competitive performance for general chat. LLaMA 3.1 specifically added strong instruction-following and tool use capabilities suitable for a production assistant.

**4. 50K-token document processing:**
- **LLaMA 3.1 8B or 70B** — LLaMA 3.1 natively supports 128K context window, which is essential for 50K-token documents. LLaMA 3 (without .1) only supports 8K context natively and would require expensive context extension techniques for such long documents. Choose 8B for cost, 70B for accuracy.

</details>

[Previous: Diffusion Models — Advanced Topics](./45_Diffusion_Models_Advanced.md) | [Next: Mixture of Experts](./47_Mixture_of_Experts.md)

---

# 46. State Space Models

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the limitations of Transformer attention that motivate alternative architectures
2. Derive the continuous and discrete State Space Model (SSM) formulation
3. Describe S4 and its efficient computation via structured matrices
4. Explain the Mamba selective state space mechanism and hardware-aware algorithm
5. Compare SSMs and Transformers in terms of speed, quality, and scaling
6. Understand hybrid architectures that combine attention and SSMs
7. Implement basic SSM layers in PyTorch

---

## Table of Contents

1. [Limitations of Transformers](#1-limitations-of-transformers)
2. [State Space Models: Mathematical Foundation](#2-state-space-models-mathematical-foundation)
3. [S4: Structured State Spaces for Sequences](#3-s4-structured-state-spaces-for-sequences)
4. [Mamba: Selective State Spaces](#4-mamba-selective-state-spaces)
5. [Mamba-2 and Improvements](#5-mamba-2-and-improvements)
6. [SSM vs Transformer Comparison](#6-ssm-vs-transformer-comparison)
7. [Hybrid Architectures](#7-hybrid-architectures)
8. [Implementation Details and Training](#8-implementation-details-and-training)
9. [Exercises](#9-exercises)

## 1. Limitations of Transformers

### Theory: Why Transformers Hit a Wall

A standard attention layer computes `softmax(QK^T / sqrt(d)) V`. The `QK^T` matrix is `N x N` for sequence length `N`, requiring `O(N^2)` time and memory. For `N = 100k` (a long document), this is 10^10 operations and 40 GB of memory at fp32 — completely impractical.

Many partial fixes exist (sparse attention, linear attention approximations, FlashAttention's memory tricks), but they trade off either quality or fundamental complexity. SSMs take a different route: drop attention entirely and use a different sequence-modeling primitive that is O(N) by construction.


### 1.1 The Quadratic Attention Problem

Self-attention computes pairwise interactions between all tokens:

```
Self-attention complexity:

Input: sequence of N tokens, each with dimension D

Q, K, V = X @ W_Q, X @ W_K, X @ W_V      O(N * D²)
Attn = softmax(Q @ K^T / √d) @ V          O(N² * D)

For long sequences:
  N = 1K    → 1M operations       ✓ Fast
  N = 8K    → 64M operations      ✓ Manageable
  N = 32K   → 1B operations       △ Slow
  N = 128K  → 16B operations      ✗ Very expensive
  N = 1M    → 1T operations       ✗ Impractical

Memory for attention matrix:
  N = 8K, float16:  8K × 8K × 2 bytes = 128 MB per head per layer
  N = 128K:         128K × 128K × 2 bytes = 32 GB per head per layer
```

### 1.2 Attention Approximations and Their Limits

```
Approach             Complexity    Quality    Adoption
──────────────────────────────────────────────────────
Full attention       O(N²)         Best       Universal
FlashAttention       O(N²)*        Best       Universal (*IO-optimized)
Linear attention     O(N)          Degraded   Limited
Sparse attention     O(N√N)        Good       Some models
Sliding window       O(N*W)        Good       Mistral, etc.
SSMs                 O(N)          Good       Mamba, etc.
```

### 1.3 The Inference Bottleneck

```
Autoregressive Transformer inference:

Step 1: Process prompt (prefill)     → O(N²) but parallelizable
Step 2: Generate tokens one by one   → each new token attends to ALL previous

Token 1:    attend to [1]                          → 1 attention op
Token 2:    attend to [1, 2]                       → 2 attention ops
Token 3:    attend to [1, 2, 3]                    → 3 attention ops
...
Token N:    attend to [1, 2, ..., N]               → N attention ops

Total for generation: 1 + 2 + ... + N = O(N²) total
Plus: KV cache grows linearly: O(N * D * L) memory

SSM inference:
Each token: update fixed-size state → O(1) per token, O(N) total
No KV cache needed → constant memory
```

---

## 2. State Space Models: Mathematical Foundation

### Theory: Continuous SSM and Discretization

A continuous-time SSM is a linear ODE:

```
\dot{h}(t) = A h(t) + B u(t)
y(t) = C h(t)
```

`u(t)` is the input, `h(t)` is the hidden state, `y(t)` is the output. `A, B, C` are learnable matrices. This is exactly the standard linear time-invariant system from control theory.

To use it on discrete sequences, discretize with step size `\Delta`:

```
h_t = \bar A h_{t-1} + \bar B u_t
y_t = C h_t
\bar A = exp(\Delta A),  \bar B = (\Delta A)^{-1} (exp(\Delta A) - I) \cdot \Delta B
```

Now you have a linear recurrence — exactly like an RNN but without the nonlinearity. Two facts make this useful for deep learning:

1. **Linear recurrence is parallelizable** via the *parallel scan* algorithm — O(N) work but O(log N) depth, exploiting GPU parallelism unlike a standard RNN.
2. **The recurrence can be expressed as a long convolution** with a kernel `K = (CB, CAB, CA^2 B, ...)`, computable via FFT in O(N log N).

So SSMs combine RNN-like sequential modeling with CNN-like parallelism. The trick is making them *expressive enough* to compete with attention.


### 2.1 Continuous State Space Model

SSMs originate from control theory. A continuous-time linear SSM:

```
State equation:     h'(t) = A h(t) + B x(t)
Output equation:    y(t)  = C h(t) + D x(t)

where:
  x(t) ∈ R^1        input signal (scalar per channel)
  h(t) ∈ R^N        hidden state (N-dimensional)
  y(t) ∈ R^1        output signal
  A ∈ R^{N×N}       state matrix (dynamics)
  B ∈ R^{N×1}       input matrix
  C ∈ R^{1×N}       output matrix
  D ∈ R^{1×1}       skip connection (often set to 0)
```

### 2.2 Discretization

To apply SSMs to sequences, we discretize with step size Δ:

```
Zero-Order Hold (ZOH) discretization:

Ā = exp(ΔA)                    ← matrix exponential
B̄ = (ΔA)^{-1} (exp(ΔA) - I) ΔB
  ≈ ΔB                         ← first-order approximation

Discrete recurrence:
  h_k = Ā h_{k-1} + B̄ x_k
  y_k = C h_k

This is a linear recurrence — can be computed in two modes:
  1. Recurrent mode: O(N) per step — good for inference
  2. Convolutional mode: O(L log L) for full sequence — good for training
```

### 2.3 The Convolution View

The discrete SSM can be unrolled into a convolution:

```
h_0 = B̄ x_0
h_1 = Ā B̄ x_0 + B̄ x_1
h_2 = Ā² B̄ x_0 + Ā B̄ x_1 + B̄ x_2
...

y_k = C h_k = C Ā^k B̄ x_0 + C Ā^{k-1} B̄ x_1 + ... + C B̄ x_k

This is a convolution with kernel:
  K̄ = (C B̄, C Ā B̄, C Ā² B̄, ..., C Ā^{L-1} B̄)

y = K̄ * x    (convolution, computed via FFT in O(L log L))
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicSSM(nn.Module):
    """Basic State Space Model layer."""

    def __init__(self, d_model, state_dim=64):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # SSM parameters (per channel)
        self.A = nn.Parameter(torch.randn(d_model, state_dim))
        self.B = nn.Parameter(torch.randn(d_model, state_dim))
        self.C = nn.Parameter(torch.randn(d_model, state_dim))
        self.D = nn.Parameter(torch.ones(d_model))

        # Discretization step size (learnable, per channel)
        self.log_delta = nn.Parameter(torch.randn(d_model) - 4.0)

    def discretize(self):
        """Compute discrete A_bar, B_bar using ZOH."""
        delta = torch.exp(self.log_delta)  # (D,)
        # Simplified: diagonal A for efficiency
        A_bar = torch.exp(delta.unsqueeze(-1) * self.A)  # (D, N)
        B_bar = delta.unsqueeze(-1) * self.B               # (D, N)
        return A_bar, B_bar

    def forward_recurrent(self, x):
        """
        Recurrent mode: O(L) per step, O(L*N) total.
        x: (B, L, D)
        """
        B_size, L, D = x.shape
        A_bar, B_bar = self.discretize()

        h = torch.zeros(B_size, D, self.state_dim, device=x.device)
        outputs = []

        for t in range(L):
            x_t = x[:, t, :]  # (B, D)
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x_t.unsqueeze(-1)
            y_t = (self.C.unsqueeze(0) * h).sum(dim=-1)  # (B, D)
            y_t = y_t + self.D * x_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, D)

    def compute_kernel(self, L):
        """Compute convolution kernel of length L."""
        A_bar, B_bar = self.discretize()

        # K[i] = C * A_bar^i * B_bar for each channel
        kernel = []
        power = torch.ones_like(A_bar)  # A_bar^0 = I (diagonal)
        for i in range(L):
            k_i = (self.C * power * B_bar).sum(dim=-1)  # (D,)
            kernel.append(k_i)
            power = power * A_bar
        return torch.stack(kernel, dim=-1)  # (D, L)

    def forward_conv(self, x):
        """
        Convolutional mode: O(L log L) via FFT.
        x: (B, L, D)
        """
        B_size, L, D = x.shape
        K = self.compute_kernel(L)  # (D, L)

        # FFT convolution
        x_perm = x.transpose(1, 2)  # (B, D, L)
        K_f = torch.fft.rfft(K, n=2*L)
        x_f = torch.fft.rfft(x_perm, n=2*L)
        y = torch.fft.irfft(K_f * x_f, n=2*L)[..., :L]  # (B, D, L)

        y = y.transpose(1, 2)  # (B, L, D)
        y = y + self.D * x  # Skip connection
        return y

    def forward(self, x, mode="conv"):
        if mode == "conv":
            return self.forward_conv(x)
        else:
            return self.forward_recurrent(x)
```

---

## 3. S4: Structured State Spaces for Sequences

### Theory: S4: Structured Matrices

A naive SSM with full `A in R^{d x d}` is too expensive. S4 (Gu, Goel, Re 2021) restricts `A` to a structured form: HiPPO-LegS, derived from approximating a continuous function by polynomials. This particular structure has two crucial properties:

1. **Long-range memory**: the state captures information from arbitrarily far in the past with controllable forgetting rate.
2. **Efficient computation**: convolution kernels can be computed in O(N log N) via Cauchy kernels.

S4 was the first SSM variant to match Transformer performance on long-range benchmarks (Path-X, Long Range Arena), proving that O(N^2) attention is not necessary for long-context modeling.


### 3.1 The Challenge of Long-Range Dependencies

The naive SSM above has a critical problem: the matrix A_bar^k decays or explodes as k grows, making it hard to capture long-range dependencies.

```
Problem:
  If eigenvalues of A_bar have |λ| < 1: signal decays exponentially
  If eigenvalues of A_bar have |λ| > 1: signal explodes
  If eigenvalues of A_bar have |λ| = 1: gradient vanishing

This is exactly the vanishing/exploding gradient problem!
```

### 3.2 HiPPO Initialization

S4 (Gu et al., 2022) solved this with the **HiPPO** (High-order Polynomial Projection Operator) initialization:

```
HiPPO-LegS matrix:

A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
A_{nk} = -(n+1)                        if n = k
A_{nk} = 0                              if n < k

This matrix has the special property that the state h(t) optimally
compresses the history of the input signal into a polynomial basis.
```

```python
def hippo_legs_matrix(N):
    """Construct the HiPPO-LegS matrix of size N×N."""
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))
    A = torch.zeros(N, N)
    for n in range(N):
        for k in range(n + 1):
            if n > k:
                A[n, k] = -P[n] * P[k]
            elif n == k:
                A[n, k] = -(n + 1)
    return A
```

### 3.3 S4 Architecture

```
S4 Block:

Input (B, L, D)
    │
    ▼
Linear projection → (B, L, H)     H independent SSM channels
    │
    ▼
┌──────────────────┐
│  H parallel SSMs │    Each SSM: state_dim = 64
│  (diagonal or    │    Using HiPPO-initialized A
│   DPLR form)     │    Convolution mode for training
└──────────────────┘
    │
    ▼
Activation (GELU)
    │
    ▼
Linear projection → (B, L, D)
    │
    ▼
Residual + LayerNorm → Output (B, L, D)
```

### 3.4 DPLR Parameterization

S4 represents A as Diagonal Plus Low-Rank (DPLR) for efficient kernel computation:

```
A = Λ - P P*    (diagonal + rank-1 correction)

Kernel computation via Cauchy kernel:
  K̂(ω) = C * (iω - A)^{-1} * B

For DPLR A:
  (iω - Λ + PP*)^{-1} can be computed in O(N) using Woodbury identity

Total kernel computation: O(N * L) instead of O(N² * L)
With FFT for convolution: O(L log L) total
```

---

## 4. Mamba: Selective State Spaces

### Theory: Mamba: Selective State Space

S4's `(A, B, C)` are *fixed* (input-independent), which means it cannot selectively focus on certain inputs. **Mamba** (Gu & Dao 2023) makes them *input-dependent*:

```
B_t = Linear(u_t),  C_t = Linear(u_t),  \Delta_t = softplus(Linear(u_t))
h_t = \bar A_t h_{t-1} + \bar B_t u_t              (where \bar A_t depends on \Delta_t)
y_t = C_t h_t
```

Now the SSM can "zoom in" on important tokens (small `\Delta` keeps state around) and "skip past" unimportant ones (large `\Delta` accelerates decay). This is a content-based selection mechanism analogous to attention's `softmax(QK^T)`, but with fundamentally different scaling.

Mamba also introduces a **hardware-aware kernel** that performs the parallel scan in SRAM rather than HBM, achieving 5x faster training than equivalent Transformers at long context. Empirically, Mamba matches or exceeds Transformer LMs of comparable size on standard NLP benchmarks.

The current consensus: SSMs are now a real alternative to Transformers for very long sequences, and hybrid architectures (Mamba + attention layers) are increasingly popular.


### 4.1 The Selectivity Problem

Linear SSMs like S4 have a fundamental limitation: they are **Linear Time-Invariant (LTI)**, meaning A, B, C are fixed regardless of input content:

```
LTI SSM:
  h_k = Ā h_{k-1} + B̄ x_k     ← same Ā, B̄ for every token
  y_k = C h_k                    ← same C for every token

Problem: Cannot selectively focus on or ignore tokens
  "The cat sat on the mat" → "cat" and "mat" treated equally
  Cannot implement content-based filtering
```

### 4.2 Mamba's Selection Mechanism

Gu & Dao (2023) made SSM parameters **input-dependent**:

```
Standard SSM:  B, C, Δ are fixed parameters
Mamba SSM:     B, C, Δ are functions of the input x

  B_k = Linear_B(x_k)      ← input-dependent
  C_k = Linear_C(x_k)      ← input-dependent
  Δ_k = softplus(Linear_Δ(x_k))  ← input-dependent step size

  Ā_k = exp(Δ_k * A)       ← now varies per token!
  B̄_k = Δ_k * B_k

Key insight: Δ controls "how much to update the state"
  Large Δ → update state significantly (attend to this token)
  Small Δ → ignore this token (state barely changes)
```

### 4.3 Hardware-Aware Algorithm

The selectivity means Mamba cannot use the convolution mode (parameters change per step). Gu & Dao designed a custom CUDA kernel:

```
Problem:
  Selective SSM: h_k = A_k h_{k-1} + B_k x_k   (time-varying!)
  Cannot precompute kernel → must use recurrence
  Naive recurrence: O(L * N * D) with many HBM reads

Solution: Hardware-aware selective scan

  1. Load chunks of (x, Δ, B, C) from HBM to SRAM
  2. Compute discretized (Ā, B̄) in SRAM
  3. Run recurrence in SRAM (fast!)
  4. Store only outputs back to HBM
  5. For backward pass: recompute intermediate states
     (recomputation is faster than HBM I/O)

Memory:  O(B * L * D * N) → O(B * L * D) with recomputation
Speed:   ~3-5× faster than naive implementation
```

### 4.4 Mamba Block Architecture

```
Mamba Block (replaces Transformer block):

Input (B, L, D)
    │
    ├────────────────────────┐
    ▼                        ▼
Linear (D → E)           Linear (D → E)
    │                        │
    ▼                        │
Conv1D (kernel=4)            │
    │                        │
    ▼                        │
SiLU activation              │
    │                        │
    ▼                        │
┌────────────────┐           │
│ Selective SSM  │           │
│ (B,C,Δ from x) │          │
└────────────────┘           │
    │                        │
    ▼                        ▼
    × ◄──── SiLU(·) ────────┘    (gating)
    │
    ▼
Linear (E → D)
    │
    ▼
Output (B, L, D)

E = expand_factor * D (typically 2×)
No attention, no MLP — just this block repeated
```

```python
class MambaBlock(nn.Module):
    """Simplified Mamba block (without custom CUDA kernel)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        # Input projections
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Conv1D
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                             .unsqueeze(0).expand(d_inner, -1)))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Selection projections (input-dependent B, C, delta)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def selective_scan(self, x, delta, B, C):
        """
        Run selective SSM.
        x: (B, L, D_inner)
        delta: (B, L, D_inner)
        B: (B, L, N)
        C: (B, L, N)
        """
        batch, L, d_inner = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log)  # (D_inner, N)

        # Discretize per-token
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D_inner, N)
        delta_B_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D_inner, N)

        # Sequential scan
        h = torch.zeros(batch, d_inner, N, device=x.device)
        outputs = []

        for t in range(L):
            h = delta_A[:, t] * h + delta_B_x[:, t]  # (B, D_inner, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D_inner)
        y = y + self.D * x  # Skip connection
        return y

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B_size, L, D = x.shape

        # Dual projection
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv1D
        x_branch = x_branch.transpose(1, 2)  # (B, D_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :L]  # causal padding
        x_branch = x_branch.transpose(1, 2)  # (B, L, D_inner)
        x_branch = F.silu(x_branch)

        # Input-dependent SSM parameters
        x_ssm = self.x_proj(x_branch)  # (B, L, 2N+1)
        B_param = x_ssm[:, :, :self.d_state]
        C_param = x_ssm[:, :, self.d_state:2*self.d_state]
        delta = F.softplus(self.dt_proj(x_ssm[:, :, -1:]))  # (B, L, D_inner)

        # Selective scan
        y = self.selective_scan(x_branch, delta, B_param, C_param)

        # Gating
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)
```

### 4.5 Mamba Model

```python
class MambaModel(nn.Module):
    """Full Mamba language model."""

    def __init__(self, vocab_size, d_model=768, n_layers=24,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': MambaBlock(d_model, d_state, d_conv, expand),
            })
            for _ in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = x + layer['mamba'](layer['norm'](x))

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
        """Autoregressive generation with constant memory per step."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
```

---

## 5. Mamba-2 and Improvements

### 5.1 Mamba-2: State Space Duality (SSD)

Dao & Gu (2024) showed a deep connection between SSMs and attention:

```
Key insight: Structured SSMs are equivalent to a form of
             semi-separable matrix multiplication

SSM recurrence:
  h_k = A_k h_{k-1} + B_k x_k
  y_k = C_k h_k

Can be written as matrix multiplication:
  y = M x

where M is a semi-separable matrix:
  M_{ij} = C_i (∏_{k=j+1}^{i} A_k) B_j    for i ≥ j
  M_{ij} = 0                                  for i < j

This is structurally similar to causal attention!
```

### 5.2 SSD Algorithm

```
Mamba-2 SSD Algorithm:

1. Chunk the sequence into blocks of size Q (e.g., 256)
2. Within each chunk: compute "mini-attention" using semi-separable structure
3. Between chunks: propagate states via recurrence

Complexity: O(L * Q * N)  where Q is chunk size
  If Q = √L: total = O(L * √L * N) — subquadratic

Benefits:
  - Can leverage tensor cores (matrix multiply hardware)
  - 2-8× faster than Mamba-1 on GPU
  - Explicit multi-head structure (like attention)
```

### 5.3 Mamba-2 Improvements

```
Mamba-1 vs Mamba-2:

Feature                    Mamba-1              Mamba-2
──────────────────────────────────────────────────────────
Core algorithm             Selective scan       SSD (chunk-wise)
Multi-head support         No (single head)     Yes (like attention)
GPU utilization            Custom kernel        Tensor cores
Speed (training)           ~1.0×                ~2-8× faster
State dimension (N)        16                   64-256
Parallelism                Sequence-level       Block-level
```

---

## 6. SSM vs Transformer Comparison

### 6.1 Computational Complexity

```
Operation          Transformer       SSM (Mamba)     Notes
────────────────────────────────────────────────────────────
Training (per step)  O(N²D)          O(NND_s)        D_s = state dim
Inference (per tok)  O(ND + KV)      O(D * D_s)      KV = KV cache access
Memory (inference)   O(N * D * L)    O(D * D_s)      L = num layers
Prefill              O(N²)           O(N)            Initial prompt
Throughput at 1M     Very low        High            Long context advantage
```

### 6.2 Quality Comparison

```
Task Type              Transformer    Mamba     Explanation
──────────────────────────────────────────────────────────────
Language modeling       ★★★★★         ★★★★      Close but Transformer edges ahead
In-context learning     ★★★★★         ★★★       Attention excels at retrieval
Long-range deps         ★★★           ★★★★★     SSM state compresses history
Copying/retrieval       ★★★★★         ★★        SSM struggles with exact recall
Audio/signal            ★★★           ★★★★★     SSMs are natural for signals
DNA/genomics            ★★★           ★★★★★     Very long sequences favor SSMs
Code generation         ★★★★★         ★★★★      Attention helps for structure
```

### 6.3 Scaling Behavior

```
Model Size       Transformer PPL    Mamba PPL    Notes
────────────────────────────────────────────────────────
125M             ~30.0              ~29.5        Mamba slightly better
350M             ~24.0              ~23.5        Comparable
1.3B             ~18.5              ~18.2        Comparable
2.8B             ~15.8              ~15.5        Comparable

At equal parameter count, Mamba matches Transformer quality
on standard language benchmarks (Pile, LAMBADA, etc.)
while being faster at inference.
```

---

## 7. Hybrid Architectures

### 7.1 Why Hybrid?

SSMs and Transformers have complementary strengths:

```
Combine:
  SSM strengths:  O(1) per-token inference, long-range, signal processing
  Attn strengths: precise retrieval, in-context learning, copying

Hybrid approach:
  Use SSM layers for most of the model (cheap, good at compression)
  Sprinkle attention layers for retrieval tasks (expensive but precise)
```

### 7.2 Jamba (AI21)

```
Jamba Architecture (AI21, 2024):

Total: 52B parameters (12B active due to MoE)

Layer composition:
  ┌──────────────────────────────────────┐
  │  Mamba layer    ← most layers        │
  │  Mamba layer                         │
  │  Mamba layer                         │
  │  Attention layer  ← every 4th layer  │
  │  Mamba layer                         │
  │  Mamba layer                         │
  │  Mamba layer                         │
  │  Attention + MoE layer               │
  │  ...                                 │
  └──────────────────────────────────────┘

Ratio: ~7:1 Mamba to Attention layers
Result: 256K context, fits in single 80GB GPU
```

### 7.3 Other Hybrid Designs

```
Model          Architecture                    Context    Release
──────────────────────────────────────────────────────────────────
Jamba          Mamba + Attention + MoE          256K       2024
Zamba          Shared attention + Mamba         Variable   2024
Griffin        Gated linear recurrence + Attn   Variable   2024 (Google)
RecurrentGemma LRRL (Linear Recurrence) + Attn  Variable   2024 (Google)
StripedHyena   Hyena (conv) + Attention         Variable   2023 (Together)
```

### 7.4 Simple Hybrid Implementation

```python
class HybridBlock(nn.Module):
    """A block that can be either Mamba or Attention."""

    def __init__(self, d_model, block_type="mamba", n_heads=8, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block_type = block_type

        if block_type == "mamba":
            self.layer = MambaBlock(d_model, d_state=d_state)
        elif block_type == "attention":
            self.layer = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)

        if self.block_type == "mamba":
            x = self.layer(x)
        else:
            x, _ = self.layer(x, x, x, attn_mask=mask)

        return residual + x


class HybridModel(nn.Module):
    """Hybrid Mamba-Attention model (Jamba-like)."""

    def __init__(self, vocab_size, d_model=768, n_layers=24, n_heads=8,
                 d_state=16, attn_every_n=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if (i + 1) % attn_every_n == 0:
                block_type = "attention"
            else:
                block_type = "mamba"
            self.layers.append(
                HybridBlock(d_model, block_type, n_heads, d_state)
            )

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)
```

---

## 8. Implementation Details and Training

### 8.1 Training Tips for SSMs

```
Hyperparameter          Recommended Value          Notes
──────────────────────────────────────────────────────────────
Learning rate           3e-4 to 8e-4               Similar to Transformers
Weight decay            0.1                         Standard
Optimizer               AdamW                       Same as Transformers
Warmup steps            1-2% of total              Standard warmup
State dimension (N)     16 (Mamba-1), 64+ (Mamba-2)  Larger N = more memory
Expand factor           2                           Inner dim = 2 * d_model
Conv kernel size        4                           Local context window
Initialization          A: HiPPO or log-spaced     Critical for performance
                        B, C: small random
                        Δ: inverse softplus of U(0.001, 0.1)
```

### 8.2 Long Sequence Training

```python
def train_ssm_long_context(model, dataloader, optimizer, max_length=65536):
    """Training loop for long-context SSM models."""
    model.train()

    for batch in dataloader:
        input_ids = batch['input_ids']  # (B, L), L can be very long
        labels = batch['labels']

        # SSMs can handle long sequences without memory issues
        # No need for gradient checkpointing in SSM layers
        # (unlike Transformers where attention memory grows quadratically)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()
```

### 8.3 Efficient Inference

```python
class MambaInferenceCache:
    """Cache for efficient Mamba autoregressive inference."""

    def __init__(self, n_layers, d_inner, d_state, d_conv, device):
        self.n_layers = n_layers
        # SSM state: constant size regardless of sequence length!
        self.ssm_states = [
            torch.zeros(1, d_inner, d_state, device=device)
            for _ in range(n_layers)
        ]
        # Conv state: small sliding window
        self.conv_states = [
            torch.zeros(1, d_inner, d_conv, device=device)
            for _ in range(n_layers)
        ]

    def memory_usage(self):
        """Compare with Transformer KV cache."""
        ssm_mem = sum(s.numel() * 2 for s in self.ssm_states)  # float16
        conv_mem = sum(s.numel() * 2 for s in self.conv_states)
        total = ssm_mem + conv_mem
        return total  # Constant! Does not grow with sequence length
```

---

## 9. Exercises

### Exercise 1: Basic SSM Layer

Implement a basic SSM and verify the equivalence of recurrent and convolutional modes:

```python
"""
Exercise 1: SSM mode equivalence.

Tasks:
1. Implement the BasicSSM class with both recurrent and convolutional forward
2. Generate a random input sequence
3. Verify that both modes produce identical outputs (within floating point tolerance)
4. Benchmark both modes for different sequence lengths (100, 1000, 10000, 100000)
5. Plot the runtime comparison

Expected: conv mode faster for long sequences (O(L log L) vs O(L*N))
"""

def verify_ssm_equivalence():
    d_model = 64
    state_dim = 16
    ssm = BasicSSM(d_model, state_dim)

    for L in [100, 1000, 10000]:
        x = torch.randn(1, L, d_model)
        y_rec = ssm.forward_recurrent(x)
        y_conv = ssm.forward_conv(x)

        diff = (y_rec - y_conv).abs().max().item()
        print(f"L={L}: max difference = {diff:.2e}")
        assert diff < 1e-4, f"Modes diverge at L={L}!"

# TODO: Run verification and benchmark
```

### Exercise 2: Selective SSM

Implement the Mamba selection mechanism and show its advantage over LTI SSMs:

```python
"""
Exercise 2: Selective vs non-selective SSM on a filtering task.

Task: Given a sequence of random tokens with special "marker" tokens,
the model must output only the tokens that follow markers.

Example:
  Input:  [a, b, MARKER, c, d, MARKER, e, f]
  Target: [0, 0, 0,      c, 0, 0,      e, 0]

This requires content-dependent filtering — impossible for LTI SSMs!

1. Implement a non-selective (LTI) SSM baseline
2. Implement a selective SSM (Mamba-style)
3. Train both on the filtering task
4. Show that only the selective SSM can learn it

Expected: LTI SSM ~50% accuracy, Selective SSM ~99%+ accuracy
"""

def generate_filtering_data(batch_size, seq_len, vocab_size=32, marker_id=0):
    """Generate data for the selective filtering task."""
    # TODO: Generate input sequences with random marker positions
    # TODO: Create target: copy token after marker, 0 otherwise
    pass

# TODO: Train and compare both models
```

### Exercise 3: Hybrid Model

Build and evaluate a simple hybrid Mamba-Attention model:

```python
"""
Exercise 3: Hybrid model comparison.

Tasks:
1. Create three models with the same parameter count:
   a. Pure Transformer (all attention layers)
   b. Pure Mamba (all Mamba layers)
   c. Hybrid (attention every 4th layer, rest Mamba)

2. Train all three on a simple language modeling task (e.g., WikiText-2)
3. Compare:
   - Training loss curves
   - Inference speed at different sequence lengths
   - Memory usage during inference

4. Test on a retrieval task: "The key to locker 42 is BLUE. ...
   [long context] ... What color is the key to locker 42?"

Expected:
  - Pure Transformer: best retrieval, slowest inference
  - Pure Mamba: fastest inference, worst retrieval
  - Hybrid: good balance of both
"""

# TODO: Build models, train, and evaluate
```

### Exercise 4: SSM for Time Series

Apply SSMs to a time series forecasting task:

```python
"""
Exercise 4: SSM for time series forecasting.

Tasks:
1. Generate or load a multivariate time series dataset
   (e.g., synthetic sine waves with different frequencies)
2. Build an SSM-based forecasting model
3. Compare with a Transformer baseline
4. Evaluate on sequences of increasing length (256, 1024, 4096, 16384)
5. Measure both accuracy and inference time

Starter code:
"""

class SSMForecaster(nn.Module):
    def __init__(self, input_dim, d_model=128, n_layers=4,
                 state_dim=16, forecast_horizon=96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.ssm_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                BasicSSM(d_model, state_dim)
            )
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, input_dim * forecast_horizon)
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim

    def forward(self, x):
        # x: (B, L, input_dim)
        h = self.input_proj(x)
        for layer in self.ssm_layers:
            h = h + layer(h)
        # Use last hidden state for forecasting
        out = self.output_proj(h[:, -1, :])
        return out.view(-1, self.forecast_horizon, self.input_dim)

# TODO: Generate data, train, and compare with Transformer
```

---

**Previous**: [Diffusion Models — Advanced Topics](./45_Diffusion_Models_Advanced.md) | **Next**: [Mixture of Experts](./47_Mixture_of_Experts.md)

---

*End of Lesson 46*

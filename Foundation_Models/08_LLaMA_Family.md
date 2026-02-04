# LLaMA Family

## 학습 목표
- LLaMA 1/2/3의 아키텍처 진화 이해
- RoPE, RMSNorm, SwiGLU 등 핵심 기술 습득
- Grouped Query Attention (GQA) 메커니즘 파악
- 실무에서의 LLaMA 활용법 학습

---

## 1. LLaMA 개요

### 1.1 LLaMA의 의의

**LLaMA**(Large Language Model Meta AI)는 2023년 Meta가 공개한 오픈소스 LLM으로, Foundation Model 연구의 민주화를 이끌었습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLaMA의 역사적 의의                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Before LLaMA (2022):                                           │
│  • 최고 성능 모델은 API만 제공 (GPT-3.5, PaLM)                    │
│  • 학술 연구용 모델은 성능 부족                                    │
│  • 오픈소스 커뮤니티의 LLM 접근 제한적                             │
│                                                                 │
│  After LLaMA (2023):                                            │
│  • 연구자들이 최첨단 모델 직접 실험 가능                           │
│  • Alpaca, Vicuna 등 파생 모델 폭발적 증가                        │
│  • LLM 연구 속도 급격히 가속화                                    │
│                                                                 │
│  핵심 기여:                                                       │
│  • Chinchilla 규칙 적용 (D=20N 이상)                             │
│  • 효율적 아키텍처 선택 검증                                      │
│  • 학습 데이터 구성 공개                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 버전 비교

| 특성 | LLaMA 1 | LLaMA 2 | LLaMA 3 |
|------|---------|---------|---------|
| 출시 | 2023.02 | 2023.07 | 2024.04 |
| 크기 | 7/13/33/65B | 7/13/70B | 8/70/405B |
| 토큰 | 1.4T | 2T | 15T+ |
| Context | 2K | 4K | 8K (128K 확장) |
| License | 연구용 | 상업적 (조건부) | 상업적 (완화) |
| GQA | ❌ | ✅ (70B) | ✅ (전체) |
| 특징 | 기본 아키텍처 | RLHF, Safety | 멀티모달, 코드 |

---

## 2. LLaMA 아키텍처

### 2.1 핵심 구성 요소

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
│  │     RoPE (Rotary Position Embedding)    │  ← 위치 정보        │
│  └─────────────────────────────────────────┘                    │
│       │                                                         │
│  ┌────┴────────────────────────────────────┐                    │
│  │         Transformer Block × N           │                    │
│  │  ┌───────────────────────────────────┐  │                    │
│  │  │  RMSNorm (Pre-normalization)      │  │  ← LayerNorm 대체   │
│  │  │            ↓                      │  │                    │
│  │  │  Grouped Query Attention (GQA)    │  │  ← KV Cache 효율   │
│  │  │            ↓                      │  │                    │
│  │  │  Residual Connection              │  │                    │
│  │  │            ↓                      │  │                    │
│  │  │  RMSNorm                          │  │                    │
│  │  │            ↓                      │  │                    │
│  │  │  SwiGLU FFN                       │  │  ← GELU 대체        │
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

### 2.2 하이퍼파라미터

```python
"""
LLaMA 모델 사양 비교
"""

LLAMA_CONFIGS = {
    "llama-7b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,  # MHA (GQA 미사용)
        "vocab_size": 32000,
        "ffn_dim": 11008,  # 약 2.67 × dim
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
        "n_kv_heads": 8,  # GQA! 8개 KV heads
        "vocab_size": 32000,
        "ffn_dim": 28672,
        "context_length": 4096,
    },
    "llama3-8b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,  # GQA
        "vocab_size": 128256,  # 확장된 vocab
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
}
```

---

## 3. RoPE (Rotary Position Embedding)

### 3.1 개념

**RoPE**는 위치 정보를 회전 행렬로 인코딩하는 방식입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Position Encoding 비교                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Sinusoidal (Transformer 원본)                               │
│     PE(pos, 2i) = sin(pos / 10000^(2i/d))                       │
│     PE(pos, 2i+1) = cos(pos / 10000^(2i/d))                     │
│     → 입력에 더함 (additive)                                     │
│     → 상대 위치 정보 약함                                         │
│                                                                 │
│  2. Learned (BERT, GPT)                                         │
│     PE = Embedding(position)                                    │
│     → 학습된 벡터                                                 │
│     → 학습 중 본 길이 이상 일반화 어려움                           │
│                                                                 │
│  3. RoPE (LLaMA)                                                │
│     R(θ) = 회전 행렬, θ = f(position)                           │
│     q' = R(θ_m) × q, k' = R(θ_n) × k                           │
│     q' · k' = q · k × cos(θ_m - θ_n)                           │
│     → 상대 위치 자연스럽게 인코딩                                  │
│     → 길이 외삽 가능 (with modifications)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 수학적 이해

```python
import torch
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    RoPE를 위한 복소수 주파수 사전 계산

    Args:
        dim: 임베딩 차원 (head_dim)
        seq_len: 최대 시퀀스 길이
        theta: 기본 주파수 (10000)

    Returns:
        freqs_cis: (seq_len, dim//2) 복소수 텐서
    """
    # 주파수 계산: θ_i = 1 / (theta^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # 위치별 각도: m * θ_i
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    # 복소수 형태: e^(i*θ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Query와 Key에 RoPE 적용

    Args:
        xq: Query (batch, seq_len, n_heads, head_dim)
        xk: Key (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: 사전 계산된 복소수 주파수

    Returns:
        회전된 Query와 Key
    """
    # 실수를 복소수로 변환 (인접한 2개씩 묶음)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 회전 적용 (복소수 곱)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# 예시
batch, seq_len, n_heads, head_dim = 2, 128, 32, 128
xq = torch.randn(batch, seq_len, n_heads, head_dim)
xk = torch.randn(batch, seq_len, n_heads, head_dim)
freqs_cis = precompute_freqs_cis(head_dim, seq_len)

xq_rope, xk_rope = apply_rotary_emb(xq, xk, freqs_cis)
print(f"Output shape: {xq_rope.shape}")  # (2, 128, 32, 128)
```

### 3.3 RoPE의 장점

```python
"""
RoPE의 장점:

1. 상대 위치 자연스럽게 인코딩
   - q_m · k_n ∝ cos(θ_m - θ_n)
   - 절대 위치가 아닌 상대 거리 의존

2. 외삽 가능성
   - 학습 시 본 길이 이상으로 확장 가능
   - (단, 성능 저하 있음 → NTK, YaRN으로 개선)

3. 효율성
   - 추가 파라미터 없음
   - Element-wise 연산으로 빠름

4. 선형 Self-attention과 호환
   - 일부 효율적 attention 방식과 결합 가능
"""
```

---

## 4. RMSNorm

### 4.1 개념

**RMSNorm**은 LayerNorm의 단순화 버전으로, 평균 계산을 제거합니다.

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
│  • 평균 빼기 + 분산으로 나누기                                     │
│  • 학습 가능한 scale(γ)와 shift(β)                               │
│                                                                 │
│  RMSNorm:                                                       │
│  ──────────────────────────────────                             │
│  RMS(x) = sqrt(mean(x^2))                                       │
│  y = γ × x / RMS(x)                                             │
│                                                                 │
│  • 평균 빼기 없음 → Re-centering 제거                             │
│  • RMS로만 스케일링                                               │
│  • shift(β) 없음                                                 │
│  • 연산량 감소, 성능 유사                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 구현

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    논문: https://arxiv.org/abs/1910.07467
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

# LayerNorm과 비교
x = torch.randn(2, 10, 512)

layer_norm = nn.LayerNorm(512)
rms_norm = RMSNorm(512)

# 연산 시간 비교 (RMSNorm이 약간 빠름)
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

### 5.1 개념

**SwiGLU**는 GLU(Gated Linear Unit)의 변형으로, Swish 활성화 함수를 사용합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FFN 활성화 함수 비교                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ReLU FFN (Transformer 원본):                                │
│     FFN(x) = max(0, xW₁ + b₁)W₂ + b₂                            │
│     • 단순하지만 음수 영역 정보 손실                               │
│                                                                 │
│  2. GELU FFN (BERT, GPT):                                       │
│     FFN(x) = GELU(xW₁)W₂                                        │
│     GELU(x) = x × Φ(x)  (Φ = CDF of normal)                     │
│     • 부드러운 활성화, 성능 향상                                   │
│                                                                 │
│  3. SwiGLU FFN (LLaMA):                                         │
│     FFN(x) = (Swish(xW₁) ⊙ xV)W₂                                │
│     Swish(x) = x × σ(x)  (σ = sigmoid)                          │
│     ⊙ = element-wise multiplication                             │
│                                                                 │
│     • Gating mechanism으로 정보 흐름 제어                         │
│     • 더 많은 파라미터, 더 좋은 성능                               │
│     • 2/3 × 4d hidden dim (파라미터 수 유지)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit

    FFN(x) = (Swish(xW₁) ⊙ xV) W₂

    논문: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()

        # hidden_dim 계산: 2/3 × 4d, 256의 배수로 반올림
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up projection

    def forward(self, x):
        # SwiGLU: Swish(xW₁) ⊙ (xW₃) → W₂
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# 기존 FFN과 비교
class StandardFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

# 파라미터 수 비교
dim = 4096
swiglu = SwiGLU(dim)  # 3개의 linear: dim→hidden, dim→hidden, hidden→dim
standard = StandardFFN(dim)  # 2개의 linear: dim→4*dim, 4*dim→dim

print(f"SwiGLU params: {sum(p.numel() for p in swiglu.parameters()):,}")
print(f"Standard FFN params: {sum(p.numel() for p in standard.parameters()):,}")
# SwiGLU가 약간 더 많지만 hidden_dim 조정으로 비슷하게 맞춤
```

---

## 6. Grouped Query Attention (GQA)

### 6.1 개념

**GQA**는 Multi-Head Attention과 Multi-Query Attention의 중간 형태입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Attention 방식 비교                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Multi-Head Attention (MHA):                                 │
│     Q heads: 32  │  K heads: 32  │  V heads: 32                 │
│     • 각 Q head가 독립적인 KV head 사용                           │
│     • 메모리: 많음 (32 × KV cache)                                │
│                                                                 │
│  2. Multi-Query Attention (MQA):                                │
│     Q heads: 32  │  K heads: 1   │  V heads: 1                  │
│     • 모든 Q head가 같은 KV 공유                                  │
│     • 메모리: 최소 (1 × KV cache)                                 │
│     • 품질: MHA보다 약간 낮음                                     │
│                                                                 │
│  3. Grouped Query Attention (GQA):                              │
│     Q heads: 32  │  K heads: 8   │  V heads: 8                  │
│     • Q heads를 그룹으로 나눠 KV 공유                              │
│     • 예: 4개의 Q head가 1개의 KV head 공유                       │
│     • 메모리: 중간 (8 × KV cache)                                 │
│     • 품질: MHA와 거의 동일                                       │
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

### 6.2 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    논문: https://arxiv.org/abs/2305.13245
    """
    def __init__(
        self,
        dim: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,  # KV heads 수 (< n_heads)
        head_dim: int = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or dim // n_heads

        # Q heads > KV heads 검증
        assert n_heads % n_kv_heads == 0
        self.n_rep = n_heads // n_kv_heads  # 각 KV head가 담당하는 Q head 수

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, mask=None, kv_cache=None):
        batch, seq_len, _ = x.shape

        # Q, K, V 계산
        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # RoPE 적용 (있는 경우)
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV Cache 처리 (추론 시)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)

        # KV heads 확장: n_kv_heads → n_heads
        # (batch, seq, n_kv_heads, head_dim) → (batch, seq, n_heads, head_dim)
        xk = self._repeat_kv(xk)
        xv = self._repeat_kv(xv)

        # Attention 계산
        xq = xq.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, xv)

        # 결과 합치기
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output), (xk, xv)

    def _repeat_kv(self, x):
        """KV heads를 Q heads 수만큼 반복"""
        if self.n_rep == 1:
            return x
        batch, seq_len, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, self.n_rep, head_dim)
        return x.reshape(batch, seq_len, n_kv_heads * self.n_rep, head_dim)

# 메모리 사용량 비교
def compare_kv_cache_memory(seq_len, batch_size=1, dtype_bytes=2):
    """KV cache 메모리 비교 (FP16 기준)"""
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
# LLaMA-70B (GQA): 0.66 GB for 4096 tokens  ← 8배 절약!
```

---

## 7. LLaMA 실습

### 7.1 HuggingFace로 사용하기

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# LLaMA 2 7B 로드
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 텍스트 생성
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

# 사용 예시
prompt = "Explain the concept of machine learning in simple terms:"
response = generate_text(prompt)
print(response)
```

### 7.2 양자화로 효율적 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,     # 이중 양자화
)

# 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 메모리 사용량 확인
print(f"Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")
# 약 4GB (FP16 대비 75% 절약)
```

### 7.3 LLaMA 3 사용

```python
# LLaMA 3 8B (128K 토크나이저)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # LLaMA 3는 bfloat16 권장
    device_map="auto"
)

# LLaMA 3 특징:
# - 128K 토크나이저 (더 효율적)
# - 8K 기본 컨텍스트 (128K까지 확장 가능)
# - 개선된 추론 능력

prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 정리

### LLaMA 핵심 기술
| 기술 | 효과 |
|------|------|
| **RoPE** | 상대 위치 인코딩, 길이 확장 가능 |
| **RMSNorm** | LayerNorm보다 빠르고 효과적 |
| **SwiGLU** | GELU보다 좋은 성능 |
| **GQA** | KV cache 메모리 8배 절약 |

### 실무 권장 사항
1. **7B/8B**: 단일 GPU (16GB+), 빠른 실험용
2. **13B**: 24GB GPU, 균형 잡힌 선택
3. **70B**: 여러 GPU, 최고 성능 필요 시
4. **양자화**: 4-bit으로 메모리 75% 절약

### 다음 단계
- [09_Mistral_MoE.md](09_Mistral_MoE.md): Mixture of Experts 아키텍처
- [19_PEFT_Unified.md](19_PEFT_Unified.md): LLaMA Fine-tuning (LoRA)

---

## 참고 자료

### 핵심 논문
- Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
- Touvron et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"
- Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models"

### 코드 & 자료
- [LLaMA GitHub](https://github.com/facebookresearch/llama)
- [HuggingFace LLaMA](https://huggingface.co/meta-llama)
- [LLaMA 3 Recipes](https://github.com/meta-llama/llama-recipes)

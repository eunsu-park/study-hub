# Mistral & Mixture of Experts

## 학습 목표
- Mistral 7B의 아키텍처 특징 이해
- Mixture of Experts (MoE) 개념과 동작 원리 파악
- Mixtral 8x7B 구조 학습
- Sparse MoE의 장단점과 실무 활용법 습득

---

## 1. Mistral 7B 개요

### 1.1 Mistral의 혁신

**Mistral 7B**는 2023년 Mistral AI가 공개한 모델로, 7B 파라미터로 13B 급 성능을 달성했습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mistral 7B 특징                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  성능 비교 (2023.10 기준):                                        │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Model          │ Params │ MMLU  │ HellaSwag │ GSM8K │      │
│  │  ───────────────│────────│───────│───────────│───────│      │
│  │  LLaMA 2 7B     │ 7B     │ 45.3  │ 77.2      │ 14.6  │      │
│  │  LLaMA 2 13B    │ 13B    │ 54.8  │ 80.7      │ 28.7  │      │
│  │  Mistral 7B     │ 7B     │ 60.1  │ 81.3      │ 52.2  │ ←!   │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  핵심 기술:                                                       │
│  • Sliding Window Attention (SWA)                               │
│  • Grouped Query Attention (GQA)                                │
│  • 더 많은 데이터로 Over-training                                 │
│  • Flash Attention 2 최적화                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Mistral 아키텍처 사양

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
        "context_length": 32768,   # 기술적 한계
        "sliding_window": 4096,    # Sliding Window Attention
        "rope_theta": 10000.0,
    },
}

# LLaMA 2 7B와 비교
LLAMA2_7B = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 32,              # MHA (GQA 미사용)
    "hidden_dim": 11008,
    "context_length": 4096,
    "sliding_window": None,        # 전체 attention
}
```

---

## 2. Sliding Window Attention (SWA)

### 2.1 개념

**Sliding Window Attention**은 각 토큰이 고정된 윈도우 내의 토큰만 attend하도록 제한합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sliding Window Attention                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full Attention (기존):                                          │
│  ────────────────────────                                       │
│  모든 토큰이 모든 이전 토큰에 attend                               │
│  복잡도: O(n²)                                                   │
│                                                                 │
│  Position:  1  2  3  4  5  6  7  8  9  10                       │
│  Token 10:  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓                       │
│                                                                 │
│  Sliding Window (W=4):                                          │
│  ────────────────────────                                       │
│  윈도우 크기 W 내의 토큰만 attend                                  │
│  복잡도: O(n × W)                                                │
│                                                                 │
│  Position:  1  2  3  4  5  6  7  8  9  10                       │
│  Token 10:  ✗  ✗  ✗  ✗  ✗  ✗  ✓  ✓  ✓  ✓                       │
│                         ↑     └───────┬───────┘                 │
│                    Window start       Window (W=4)              │
│                                                                 │
│  레이어 쌓기 효과:                                                │
│  ────────────────────────                                       │
│  L개 레이어 → 실제 receptive field = L × W                       │
│  32 layers × 4096 window = 131,072 토큰 범위!                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 구현

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
    Sliding Window Attention 구현

    Args:
        query: (batch, n_heads, seq_len, head_dim)
        key: (batch, n_heads, seq_len, head_dim)
        value: (batch, n_heads, seq_len, head_dim)
        window_size: 윈도우 크기
        causal: Causal masking 적용 여부
    """
    batch, n_heads, seq_len, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Sliding window mask 생성
    # 각 위치 i는 max(0, i-W+1)부터 i까지만 attend
    row_idx = torch.arange(seq_len).unsqueeze(1)  # (seq, 1)
    col_idx = torch.arange(seq_len).unsqueeze(0)  # (1, seq)

    # Causal: col <= row
    # Window: col >= row - window_size + 1
    if causal:
        mask = (col_idx <= row_idx) & (col_idx >= row_idx - window_size + 1)
    else:
        mask = torch.abs(row_idx - col_idx) < window_size

    # Mask 적용
    mask = mask.to(scores.device)
    scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax & output
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output

# 메모리 비교
def compare_attention_memory(seq_len, window_size=4096):
    """Full vs Sliding Window 메모리 비교"""
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
Rolling Buffer: 고정 크기 KV cache로 긴 시퀀스 처리

일반 KV Cache:
- 모든 토큰의 KV 저장
- 메모리: O(seq_len)

Rolling Buffer:
- window_size만큼만 저장
- 오래된 KV는 덮어씀
- 메모리: O(window_size) = 상수!

예시 (window=4):
Step 1: [K1, K2, K3, K4]
Step 2: [K5, K2, K3, K4]  ← K1 위치에 K5 저장
Step 3: [K5, K6, K3, K4]  ← K2 위치에 K6 저장
...

장점:
- 무한 시퀀스 처리 가능 (메모리 고정)
- 추론 속도 일정

단점:
- 오래된 정보 손실
- 레이어 쌓기로 보완
"""

class RollingKVCache:
    def __init__(self, window_size: int, n_layers: int, n_kv_heads: int, head_dim: int):
        self.window_size = window_size
        self.cache_k = torch.zeros(n_layers, 1, window_size, n_kv_heads, head_dim)
        self.cache_v = torch.zeros(n_layers, 1, window_size, n_kv_heads, head_dim)
        self.pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """새로운 KV를 cache에 추가 (circular buffer)"""
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

## 3. Mixture of Experts (MoE) 기초

### 3.1 MoE 개념

**Mixture of Experts**는 여러 "전문가" 네트워크 중 일부만 활성화하여 효율성을 높이는 아키텍처입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mixture of Experts 개념                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dense Model:                                                   │
│  ─────────────────                                              │
│  Input ──► [FFN (전체)] ──► Output                              │
│  • 모든 파라미터가 매번 활성화                                     │
│  • 계산량 = 파라미터 수에 비례                                    │
│                                                                 │
│  Sparse MoE Model:                                              │
│  ─────────────────                                              │
│                        ┌──► Expert 1 ──┐                        │
│                        │               │                        │
│  Input ──► Router ─────┼──► Expert 2 ──┼──► Combine ──► Output  │
│              ↓         │               │                        │
│         (Top-K 선택)   └──► Expert 3 ──┘                        │
│                        └──► Expert N (비활성화)                   │
│                                                                 │
│  • 라우터가 K개 전문가만 선택                                      │
│  • 파라미터 多, 계산량 少                                         │
│  • 예: 8개 전문가, 2개만 활성화 → 계산량 1/4                       │
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
    Top-K Router: 입력마다 K개의 전문가 선택

    수식:
    G(x) = softmax(TopK(x · W_g))

    여기서 TopK는 상위 K개만 유지, 나머지는 -inf
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
            router_probs: (batch, seq_len, top_k) - 선택된 전문가 가중치
            expert_indices: (batch, seq_len, top_k) - 선택된 전문가 인덱스
        """
        # 라우터 로짓 계산
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Top-K 선택
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax (선택된 전문가들 사이에서)
        router_probs = F.softmax(top_k_logits, dim=-1)

        return router_probs, top_k_indices

# 예시
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

    각 토큰이 Top-K 전문가에게 라우팅되어 처리됨
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

        # Experts (각각 독립적인 FFN)
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

        # 라우팅
        router_probs, expert_indices = self.router(x)
        # router_probs: (batch, seq_len, top_k)
        # expert_indices: (batch, seq_len, top_k)

        # 출력 초기화
        output = torch.zeros_like(x)

        # 각 전문가별로 처리 (간단한 구현, 실제로는 더 최적화됨)
        for k in range(self.top_k):
            expert_idx = expert_indices[:, :, k]  # (batch, seq_len)
            expert_prob = router_probs[:, :, k:k+1]  # (batch, seq_len, 1)

            for e in range(self.num_experts):
                # 이 전문가가 선택된 위치 찾기
                mask = (expert_idx == e)
                if mask.any():
                    # 해당 토큰들 추출
                    selected = x[mask]  # (num_selected, dim)
                    # 전문가 적용
                    expert_output = self.experts[e](selected)
                    # 가중치 적용하여 결과에 추가
                    output[mask] += expert_prob[mask].squeeze(-1).unsqueeze(-1) * expert_output

        return output

# 사용 예시
moe = MoELayer(dim=4096, hidden_dim=14336, num_experts=8, top_k=2)
x = torch.randn(2, 10, 4096)
output = moe(x)
print(f"Output shape: {output.shape}")  # (2, 10, 4096)
```

---

## 4. Mixtral 8x7B

### 4.1 아키텍처

**Mixtral 8x7B**는 8개의 전문가를 가진 MoE 모델로, 각 레이어에서 2개의 전문가만 활성화됩니다.

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
│  │  │     선택      선택    비활성   비활성   ...        │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  총 파라미터: ~46.7B (8 experts × 7B FFN params)                 │
│  활성 파라미터: ~12.9B (2/8 experts)                              │
│  추론 속도: 12.9B dense 모델과 유사                               │
│  성능: 70B dense 모델 수준                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Mixtral 사양

```python
MIXTRAL_CONFIG = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "hidden_dim": 14336,
    "vocab_size": 32000,

    # MoE 설정
    "num_experts": 8,
    "num_experts_per_tok": 2,  # Top-K

    # Attention
    "sliding_window": 4096,
    "context_length": 32768,

    # 파라미터 계산
    # Attention: 4 × dim² × n_layers = 4 × 4096² × 32 ≈ 2.1B
    # MoE FFN: 8 × 3 × dim × hidden × n_layers = 8 × 3 × 4096 × 14336 × 32 ≈ 44.6B
    # Total: ~46.7B
    # Active: ~12.9B (attention + 2/8 FFN)
}
```

### 4.3 Load Balancing Loss

MoE의 핵심 과제 중 하나는 **전문가 불균형** 문제입니다.

```python
def load_balancing_loss(router_probs, expert_indices, num_experts):
    """
    Load Balancing Loss: 전문가들이 균등하게 사용되도록 유도

    문제: 일부 전문가만 과도하게 사용되는 현상 (winner-take-all)
    해결: 균형 잡힌 라우팅을 유도하는 auxiliary loss

    수식:
    L_balance = α × Σ_e (f_e × P_e)

    f_e = 전문가 e가 선택된 토큰 비율
    P_e = 전문가 e에 할당된 라우팅 확률 평균
    α = 스케일링 계수 (예: 0.01)
    """
    batch, seq_len, top_k = router_probs.shape
    num_tokens = batch * seq_len

    # f_e: 각 전문가가 선택된 비율
    expert_counts = torch.zeros(num_experts, device=router_probs.device)
    for e in range(num_experts):
        expert_counts[e] = (expert_indices == e).float().sum() / (num_tokens * top_k)

    # P_e: 각 전문가에 할당된 평균 확률
    expert_probs = torch.zeros(num_experts, device=router_probs.device)
    # (간소화된 계산 - 실제로는 gate logits에서 계산)

    # Balance loss
    loss = (expert_counts * expert_probs).sum() * num_experts

    return loss

# 학습 시
"""
total_loss = language_model_loss + alpha * load_balancing_loss
"""
```

---

## 5. MoE의 장단점

### 5.1 장점

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoE의 장점                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 파라미터 효율성                                               │
│     • 많은 파라미터, 적은 계산량                                  │
│     • Mixtral 8x7B: 46.7B params, 12.9B active                   │
│     • Dense 70B 급 성능, 13B 급 속도                             │
│                                                                 │
│  2. 전문화 (Specialization)                                      │
│     • 각 전문가가 다른 패턴/도메인 학습                            │
│     • 예: Expert 1=수학, Expert 2=코드, Expert 3=언어             │
│     • 더 깊은 전문 지식 인코딩 가능                                │
│                                                                 │
│  3. 스케일링                                                      │
│     • 전문가 수 늘려 모델 확장 용이                                │
│     • 계산량 증가 최소화하며 용량 증가                             │
│     • Google Switch Transformer: 1.6T params!                    │
│                                                                 │
│  4. 학습 효율                                                     │
│     • 같은 계산량으로 더 큰 모델 학습 가능                          │
│     • Scaling Law 관점에서 유리                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 단점

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoE의 단점                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 메모리 요구량                                                 │
│     • 모든 전문가를 메모리에 로드해야 함                           │
│     • Mixtral 8x7B: 46.7B params ≈ 93GB (FP16)                   │
│     • 추론 시 많은 GPU 메모리 필요                                 │
│                                                                 │
│  2. 학습 불안정성                                                 │
│     • 라우터 학습이 어려움                                        │
│     • 전문가 불균형 (일부만 사용)                                  │
│     • Auxiliary loss 튜닝 필요                                   │
│                                                                 │
│  3. 분산 학습 복잡성                                              │
│     • Expert parallelism 필요                                    │
│     • 통신 오버헤드                                               │
│     • 로드 밸런싱 어려움                                          │
│                                                                 │
│  4. Fine-tuning 어려움                                           │
│     • 전문가 specialization 유지하며 적응 필요                    │
│     • 일부 전문가만 fine-tune?                                    │
│     • 연구 진행 중                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Mistral/Mixtral 실습

### 6.1 Mistral 7B 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mistral 7B 로드
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 텍스트 생성
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

### 6.2 Mixtral 8x7B 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mixtral 8x7B (많은 메모리 필요!)
model_name = "mistralai/Mixtral-8x7B-v0.1"

# 4-bit 양자화로 메모리 절약
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

# 사용
prompt = "[INST] Write a Python function to calculate fibonacci numbers. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 6.3 vLLM으로 효율적 서빙

```python
from vllm import LLM, SamplingParams

# vLLM은 MoE 모델을 효율적으로 서빙
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    tensor_parallel_size=2,  # 2 GPU
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

## 7. MoE 변형들

### 7.1 주요 MoE 모델들

| 모델 | 조직 | 전문가 수 | Top-K | 총 파라미터 | 활성 파라미터 |
|------|------|----------|-------|------------|-------------|
| Switch Transformer | Google | 2048 | 1 | 1.6T | <1B |
| GLaM | Google | 64 | 2 | 1.2T | ~100B |
| Mixtral 8x7B | Mistral | 8 | 2 | 46.7B | 12.9B |
| Mixtral 8x22B | Mistral | 8 | 2 | 141B | 39B |
| DeepSeek MoE | DeepSeek | 64 | 6 | 145B | 22B |

### 7.2 Fine-grained MoE

```python
"""
Fine-grained MoE: 더 많은 작은 전문가

기존 (Coarse-grained):
- 8개 큰 전문가, Top-2 선택
- 각 전문가가 넓은 범위 담당

Fine-grained (DeepSeek 스타일):
- 64개 작은 전문가, Top-6 선택
- 더 세밀한 전문화 가능
- 라우팅 유연성 증가

장점:
- 더 세밀한 전문화
- 더 나은 로드 밸런싱
- 확장성

단점:
- 라우팅 오버헤드
- 학습 복잡성
"""
```

---

## 정리

### Mistral 핵심
- **Sliding Window Attention**: 메모리 O(W)로 긴 시퀀스 처리
- **GQA**: KV cache 효율성
- **Over-training**: 작은 모델, 많은 데이터

### MoE 핵심
- **Sparse Activation**: 파라미터 多, 계산 少
- **Router**: Top-K 전문가 선택
- **Load Balancing**: 전문가 균형 유지

### 실무 선택 가이드
| 상황 | 권장 모델 |
|------|----------|
| 단일 GPU (16GB) | Mistral 7B (4-bit) |
| 2× GPU (48GB) | Mixtral 8x7B (4-bit) |
| 서버급 (8× A100) | Mixtral 8x22B |
| 속도 우선 | Mistral 7B |
| 성능 우선 | Mixtral 8x7B+ |

### 다음 단계
- [10_Long_Context_Models.md](10_Long_Context_Models.md): 긴 컨텍스트 처리
- [22_Inference_Optimization.md](22_Inference_Optimization.md): 효율적 추론

---

## 참고 자료

### 핵심 논문
- Jiang et al. (2023). "Mistral 7B"
- Jiang et al. (2024). "Mixtral of Experts"
- Fedus et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models"
- Du et al. (2022). "GLaM: Efficient Scaling of Language Models"

### 코드 & 자료
- [Mistral GitHub](https://github.com/mistralai/mistral-src)
- [HuggingFace Mistral](https://huggingface.co/mistralai)
- [vLLM MoE Support](https://docs.vllm.ai/)

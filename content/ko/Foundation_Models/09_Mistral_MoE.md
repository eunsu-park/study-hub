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

## 8. 전문가 라우팅 메커니즘 (Expert Routing Mechanisms)

라우터(Router)는 MoE 모델의 두뇌입니다 — 어떤 전문가(Expert)가 각 토큰을 처리할지 결정합니다. 라우팅 전략의 선택은 모델 품질, 학습 안정성, 추론 효율성에 지대한 영향을 미칩니다. 이 섹션에서는 라우팅 전략, 로드 밸런싱(Load Balancing), 용량(Capacity) 관리를 심층적으로 다룹니다.

### 8.1 토큰 라우팅 전략

#### Top-K 라우팅 (표준)

가장 일반적인 접근법: 학습된 선형 레이어가 각 전문가에 점수를 매기고, 상위 K개 전문가를 선택합니다.

```
Input token x → Linear(dim, n_experts) → logits
                                          ↓
                               Top-K selection → softmax over K
                                          ↓
                               Weighted sum of K expert outputs

Used by: Mixtral (K=2), GLaM (K=2), GShard (K=2)
```

#### Expert Choice 라우팅 (Zhou et al., 2022)

토큰이 전문가를 선택하는 대신, **전문가가 토큰을 선택**합니다. 각 전문가가 자신이 가장 자신 있는 상위 C개 토큰을 선택합니다.

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

#### 해시 라우팅 (Hash Routing, Roller et al., 2021)

해싱 기반의 비학습(Non-learned), 결정적(Deterministic) 라우팅 전략:

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

### 8.2 라우터 아키텍처 상세

#### Softmax 게이트 (표준)

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

Top-K 선택 전에 라우터 로짓(Logit)에 노이즈(Noise)를 추가하면 학습 중 탐색(Exploration)이 개선됩니다:

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

### 8.3 로드 밸런싱 손실(Load Balancing Loss)과 보조 손실(Auxiliary Loss)

명시적인 장려 없이는 라우터가 대부분의 토큰을 소수의 "선호" 전문가에게 라우팅하는 경향이 있습니다 (Winner-take-all 붕괴). 보조 손실(Auxiliary Loss)이 이를 방지합니다.

#### 표준 로드 밸런싱 손실 (Switch Transformer)

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

큰 라우터 로짓(Logit) 값을 패널티하여 안정성을 향상시키는 추가 보조 손실:

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

### 8.4 용량 인수(Capacity Factor)와 오버플로우(Overflow) 처리

각 전문가는 최대 **용량(Capacity)** — 순전파(Forward Pass)당 처리할 수 있는 토큰 수 — 이 있습니다. 이는 효율적인 배치 연산에 핵심적입니다.

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

### 8.5 Switch Transformer vs GShard 라우팅 비교

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│ 특성             │ Switch Transformer   │ GShard               │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Top-K            │ K=1 (단일 전문가)    │ K=2 (두 전문가)      │
│ 용량 인수        │ 1.0-1.5              │ 2.0                  │
│ 오버플로우       │ 토큰 드롭            │ 토큰 드롭            │
│ 보조 손실        │ L_balance            │ L_balance + L_load   │
│ 노이즈           │ 미사용               │ Noisy top-k          │
│ 전문가 수        │ 128-2048             │ 64-512               │
│ 대규모 품질      │ 많은 전문가로 우수   │ 전문가당 품질 우수   │
│ 통신량           │ 적음 (K=1)           │ 많음 (K=2)           │
├──────────────────┼──────────────────────┼──────────────────────┤
│ 핵심 통찰        │ 단순함: 토큰당 1개   │ 견고함: 2개 전문가가 │
│                  │ 전문가면 대규모에서  │ 중복성과 안정적      │
│                  │ 충분함               │ 학습 제공            │
└──────────────────┴──────────────────────┴──────────────────────┘

현대적 관행 (Mixtral, DeepSeek):
- K=2 (GShard 방식)가 기본이 됨
- 개선된 로드 밸런싱(Auxiliary Loss + Z-Loss) 병행
- 용량 인수(Capacity Factor) 1.25-1.5 (경험적 튜닝)
```

### 8.6 완전한 라우터 예제

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

---

## 연습 문제

### 연습 문제 1: MoE 유효 파라미터 분석

Mixtral 8x7B는 MoE 레이어당 8개의 전문가를 가지고, 토큰당 2개를 활성화(Top-2 라우팅)하며, 총 46.7B 파라미터에 포워드 패스당 12.9B의 활성 파라미터를 가집니다.

1. Mixtral이 총 46.7B 파라미터를 가짐에도 각 토큰에 대해 12.9B 파라미터만 활성화하는 이유는 무엇인가요?
2. "계산 효율 비율" (활성 / 전체)은 무엇이며, 밀집(dense) 46.7B 모델과 비교한 추론 비용에 어떤 의미를 가지나요?
3. K=4 (K=2 대신)였다면 활성 파라미터와 추론 비용이 어떻게 변할까요?

<details>
<summary>정답 보기</summary>

**1. 12.9B 활성 파라미터만 사용하는 이유:**

각 MoE 레이어에서 라우터는 8개의 전문가 중 2개만 선택하여 각 토큰을 처리합니다. 선택되지 않은 6개의 전문가는 단순히 실행하지 않습니다 — 계산이 완전히 건너뜁니다. 모든 토큰에 공유된 트랜스포머 어텐션 레이어와 MoE FFN 파라미터의 2/8만 활성화됩니다. 대략:
- 공유 어텐션 파라미터 + MoE FFN 파라미터의 2/8 ≈ 토큰당 12.9B

**2. 계산 효율 비율:**

```
효율 비율 = 12.9B / 46.7B ≈ 27.6%
```

각 포워드 패스는 밀집 46.7B 모델 비용의 약 27.6%만 소요됩니다. 하지만 모델은 46.7B 파라미터 가치의 지식 용량을 가집니다. 이것이 핵심 MoE 트레이드오프입니다: ~13B 밀집 모델처럼 추론 비용을 지불하면서 46.7B 파라미터 용량의 혜택을 받습니다.

**3. K=4의 효과:**
- 활성 파라미터가 대략 두 배로 증가: 포워드 패스당 ~25.8B
- 추론 비용이 비례적으로 증가 (토큰당 2배 더 많은 전문가 계산)
- 계산 효율 비율: 25.8 / 46.7 ≈ 55.2%
- 모델은 계산 측면에서 25.8B 등가 밀집 모델이 되지만 여전히 46.7B 지식 용량을 가짐
- 품질은 K=4로 향상되지만 효율성 이점을 잃는 비용이 발생

</details>

---

### 연습 문제 2: 슬라이딩 윈도우 어텐션(SWA) 메모리 분석

Mistral 7B는 윈도우 크기 W=4096과 롤링 버퍼(rolling buffer) KV 캐시로 슬라이딩 윈도우 어텐션(SWA, Sliding Window Attention)을 사용합니다.

1. 표준 셀프 어텐션은 n 토큰에 대해 O(n²) 메모리 복잡도를 가집니다. n 토큰에 대한 SWA의 메모리 복잡도는 무엇인가요?
2. 32,768 토큰(32K) 시퀀스에서 표준 MHA vs SWA의 KV 캐시 메모리를 비교하세요 (32 헤드, head_dim=128, fp16, 32 레이어 가정).
3. 토큰이 슬라이딩 윈도우에서 "벗어날" 때 어떤 정보가 손실되며, 다층 아키텍처는 어떻게 부분적으로 보상하나요?

<details>
<summary>정답 보기</summary>

**1. SWA의 메모리 복잡도:**
SWA는 KV 캐시에 최근 W 토큰만 저장합니다. n이 증가해도 메모리는 제한됩니다:
- **O(W × d)** = O(d) — W가 고정되어 있으므로 시퀀스 길이에 대해 사실상 상수
- 더 정확히: O(n) 시간 (각 토큰이 한 번 처리됨)이지만 O(W) KV 캐시 공간

**2. n=32768 토큰에서의 KV 캐시 메모리 비교:**

표준 MHA (모든 n 토큰 저장):
```
32768 × 2(K+V) × 32헤드 × 128헤드차원 × 2바이트 × 32레이어 ≈ 16 GB
```

SWA (마지막 W=4096 토큰만 저장):
```
4096 × 2 × 32 × 128 × 2 × 32 ≈ 2 GB
```

**SWA는 32K 토큰에서 ~14 GB KV 캐시 메모리를 절약** — 8배 감소.

**3. 정보 손실과 보상:**

토큰이 윈도우 밖으로 나가면 (W 위치보다 이전), 해당 토큰에 대한 직접 어텐션이 손실됩니다. 하지만 다층 아키텍처는 **수용 영역 확장(receptive field expansion)** 속성을 통해 부분적으로 보상합니다:
- 레이어 1: 마지막 W 토큰에 어텐션 가능
- 레이어 2: 레이어 1이 이미 "요약한" 토큰에 어텐션 — 사실상 W² 토큰의 이력을 봄
- k 레이어 후: 수용 영역 ≈ W × k

Mistral 7B의 32 레이어와 W=4096의 경우: 유효 수용 영역 ≈ 4096 × 32 = 131,072 토큰 — 윈도우 크기를 훨씬 초과합니다. 초기 컨텍스트는 직접 접근 가능하지 않고 중간 표현에 암묵적으로 "압축"됩니다.

</details>

---

### 연습 문제 3: 로드 밸런싱(Load Balancing) 손실 구현

수업에서 Switch Transformer 로드 밸런싱 손실을 보여줍니다. 이 손실이 **포함되지 않으면** 학습 중 전문가 활용에 어떤 일이 발생하는지 설명하고, 왜 라우터 붕괴(router collapse)가 발생하는지 추적하세요.

<details>
<summary>정답 보기</summary>

**로드 밸런싱 손실 없이 라우터 붕괴가 발생하는 과정:**

**1단계: 초기 비대칭성**
초기화 중에 라우터의 선형 레이어는 무작위 가중치를 가집니다. 우연히 일부 전문가가 일반적인 입력 패턴에 대해 약간 더 높은 라우팅 확률을 받습니다. 이 비대칭성은 작지만 존재합니다.

**2단계: 부익부(Rich-get-richer) 역학**
"선호된" 전문가들이 더 많은 학습 토큰을 받음 → 파라미터가 더 많이 향상됨 → 출력이 더 좋아짐 → 라우팅에 대한 보상 신호(더 낮은 손실)가 높아짐 → 라우터가 더 강하게 그들을 선호하도록 학습됨.

**3단계: 전문가 붕괴**
충분한 학습 후, 1-2개의 전문가가 거의 모든 토큰을 받습니다. 나머지 6-7개의 전문가는 거의 그래디언트 신호를 받지 못하고 정체됩니다. 학습을 받지 않기 때문에 파라미터가 개선되지 않습니다.

**4단계: 완전한 붕괴**
모델은 사실상 1-2개의 FFN 행렬만 있는 밀집 모델과 동등해집니다 — 8개의 전문가를 갖는 용량 이점을 모두 잃습니다. 총 파라미터 수는 여전히 8배 더 크지만 (모든 전문가 가중치가 여전히 존재) 1-2개만 유용합니다.

**로드 밸런싱 손실이 하는 일:**
`f_e` (전문가 e에 대한 토큰 비율)가 균일(1/E)에서 벗어날 때 페널티를 줌으로써, 손실은 라우터가 토큰을 더 균등하게 분배하도록 밀어주는 그래디언트 신호를 생성합니다. 잡음이 있는 top-k (라우터 로짓에 탐색 잡음 추가)와 결합하여 퇴화된 해로의 조기 수렴을 방지합니다.

</details>

---

### 연습 문제 4: MoE vs 밀집(Dense) 트레이드오프 결정

프로덕션 언어 모델을 위해 두 아키텍처 중 하나를 선택하고 있습니다:
- **모델 A**: 밀집 13B 파라미터 모델
- **모델 B**: 8개 전문가, top-2 라우팅, 46.7B 총 파라미터, 13B 활성 파라미터의 희소 MoE

두 모델의 추론 계산 비용은 대략 동일합니다. 각 시나리오에서 어느 것을 선택할지 평가하세요:

1. 메모리 제한 엣지 서버 배포 (32GB 총 RAM).
2. 소규모 도메인별 데이터셋으로 정기적으로 파인튜닝하는 연구 실험실.
3. 다양한 주제에 걸쳐 최대한 광범위한 지식이 필요한 프로덕션 API.
4. 동일한 입력에 대해 완전히 재현 가능한 출력이 필요한 시스템.

<details>
<summary>정답 보기</summary>

**1. 메모리 제한 엣지 서버 (32GB RAM):**
- **밀집 13B 선택** — 모델 B는 모든 46.7B 파라미터를 로드해야 합니다 (~93GB in fp16). 양자화해도 32GB 한도를 초과합니다. 밀집 13B는 fp16에서 ~26GB가 필요합니다. 타이트한 메모리 예산의 엣지 배포에는 밀집 모델이 강하게 선호됩니다.

**2. 연구 실험실, 소규모 데이터셋 파인튜닝:**
- **밀집 13B 선택** — MoE 모델은 파인튜닝이 훨씬 어렵습니다:
  - 라우터가 적절한 전문가 할당을 동시에 학습해야 함
  - 소규모 데이터셋은 모든 전문가를 적절히 업데이트할 충분한 신호를 제공하지 않음
  - 파인튜닝 불안정성이 더 흔함 (일부 전문가가 붕괴하거나 과도하게 특화됨)
  - 밀집 모델은 소규모 데이터셋에서 더 예측 가능하고 효율적으로 파인튜닝됨.

**3. 다양한 주제의 최대 지식 범위:**
- **MoE 46.7B 선택** — MoE의 전체 동기는 서로 다른 전문가가 다른 도메인(코딩, 추론, 언어 패턴, 사실 지식)에 특화할 수 있다는 것입니다. 46.7B 총 파라미터로 모델 B는 13B 밀집 모델보다 훨씬 더 많은 다양한 지식 용량을 가집니다. 범용 API에서 이 범위 이점은 중요합니다.

**4. 완전히 재현 가능한 출력:**
- **밀집 13B 선호** (하지만 두 모델 모두 주의사항이 있음) — MoE는 추가적인 비결정성 원인을 도입합니다: 이산적인 top-K 라우팅 결정. 하드웨어 실행 간의 미묘한 부동소수점 차이가 어떤 전문가가 선택되는지를 바꿀 수 있어, temperature=0에서도 출력이 변경됩니다. 밀집 모델은 비결정성의 원인으로 부동소수점 정밀도만 있습니다. 참고: 충분한 정밀도 제어로 두 모델 모두 결정론적으로 만들 수 있습니다.

</details>

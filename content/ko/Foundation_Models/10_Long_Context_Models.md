# 10. Long Context Models

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 표준 Transformer의 셀프 어텐션(Self-Attention)이 O(n²) 시간 및 메모리 복잡도를 가지는 이유와 이로 인해 발생하는 실질적인 컨텍스트 길이 한계를 설명할 수 있습니다.
2. 효율적인 어텐션 메커니즘(스파스 어텐션, 선형 어텐션, FlashAttention)을 비교하고, 각각이 모델 품질을 유지하면서 연산 비용을 줄이는 방식을 설명할 수 있습니다.
3. 포지션 인코딩 확장 기법(RoPE, ALiBi, YaRN, LongRoPE)을 설명하고, 짧은 컨텍스트로 학습된 모델이 더 긴 시퀀스로 일반화될 수 있는 방식을 설명할 수 있습니다.
4. 롱 컨텍스트 모델에서 "중간에서 잃어버린(Lost in the Middle)" 현상을 분석하고, 검색(Retrieval), 에이전트(Agent) 설계, 문서 이해 과제에 대한 함의를 설명할 수 있습니다.
5. 슬라이딩 윈도우(Sliding Window) 및 청크 어텐션(Chunked Attention) 패턴을 구현하고, 정확도와 메모리 사용량 측면에서 전체 어텐션(Full Attention)과 비교할 수 있습니다.
6. 롱 컨텍스트 벤치마크(SCROLLS, L-Eval, RULER)를 평가하고, 프로덕션 배포 환경에서 컨텍스트 길이, 추론 지연, 메모리 사용량 간의 트레이드오프를 논의할 수 있습니다.

---

## 개요

표준 Transformer의 Self-Attention은 O(n²) 복잡도로 인해 긴 시퀀스 처리에 한계가 있습니다. 이 레슨에서는 컨텍스트 길이를 확장하는 다양한 기법을 다룹니다.

---

## 1. Context Length의 중요성

### 1.1 왜 긴 컨텍스트가 필요한가?

```
┌──────────────────────────────────────────────────────────────────┐
│                   Long Context 사용 사례                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📚 문서 분석                                                    │
│  - 논문 전체 (10K-50K 토큰)                                     │
│  - 법률 문서 (100K+ 토큰)                                       │
│  - 책 전체 요약                                                  │
│                                                                  │
│  💻 코드 이해                                                    │
│  - 전체 코드베이스 분석                                          │
│  - 긴 함수/클래스 리팩토링                                       │
│  - 멀티파일 디버깅                                               │
│                                                                  │
│  🤖 Agent 시스템                                                 │
│  - 긴 대화 히스토리 유지                                         │
│  - 복잡한 멀티스텝 태스크                                        │
│  - Tool 사용 기록 누적                                           │
│                                                                  │
│  🔍 RAG 개선                                                     │
│  - 더 많은 관련 문서 포함                                        │
│  - 문서 조각 대신 전체 문서 제공                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 모델별 컨텍스트 길이 비교

| 모델 | 컨텍스트 길이 | 출시 시기 |
|------|---------------|-----------|
| GPT-3 | 2,048 | 2020 |
| GPT-3.5 | 4,096 / 16,384 | 2022-2023 |
| GPT-4 | 8,192 / 32,768 / 128K | 2023-2024 |
| Claude 2 | 100,000 | 2023 |
| Claude 3 | 200,000 | 2024 |
| Gemini 1.5 | 1,000,000 / 2,000,000 | 2024 |
| LLaMA 2 | 4,096 | 2023 |
| LLaMA 3 | 8,192 / 128K | 2024 |

---

## 2. 효율적인 Attention 메커니즘

### 2.1 Sparse Attention

```
┌─────────────────────────────────────────────────────────────┐
│                    Sparse Attention 패턴                    │
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
│  Longformer: Local + Global 토큰 조합                       │
│  BigBird: Local + Global + Random                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Longformer 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LongformerAttention(nn.Module):
    """
    Longformer: Sliding Window + Global Attention

    복잡도: O(n × w) where w = window size
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

        # Global attention용 별도 projection
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

        # Q, K, V 계산
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 1. Sliding Window Attention (local)
        local_output = self._sliding_window_attention(Q, K, V)

        # 2. Global Attention (처음 global_tokens개)
        global_output = self._global_attention(
            hidden_states, Q, K, V
        )

        # 결합 (global 토큰 위치는 global 결과 사용)
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

        각 토큰은 window_size 범위 내의 토큰만 참조
        """
        batch_size, seq_len, num_heads, head_dim = Q.shape
        w = self.window_size // 2

        # 패딩 추가
        Q_padded = F.pad(Q, (0, 0, 0, 0, w, w), value=0)
        K_padded = F.pad(K, (0, 0, 0, 0, w, w), value=0)
        V_padded = F.pad(V, (0, 0, 0, 0, w, w), value=0)

        # 윈도우 추출 (unfold)
        # 실제 구현은 더 복잡하지만 개념 이해용 간소화 버전
        output = torch.zeros_like(Q)

        for i in range(seq_len):
            # i번째 토큰의 윈도우: [i, i + window_size]
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
        """Global Attention: global 토큰은 전체 시퀀스 참조"""
        batch_size, seq_len, _ = hidden_states.shape

        # Global 토큰만 추출
        global_hidden = hidden_states[:, :self.global_tokens]

        # Global Q, K, V
        global_Q = self.global_query(global_hidden)
        global_K = self.global_key(hidden_states)
        global_V = self.global_value(hidden_states)

        # 전체 시퀀스에 대해 attention
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
# Flash Attention은 CUDA 커널로 구현되어 있음
# 여기서는 개념만 설명

"""
Flash Attention 핵심 아이디어:

1. 타일링 (Tiling):
   - Q, K, V를 SRAM에 맞는 블록으로 분할
   - HBM ↔ SRAM 데이터 전송 최소화

2. 재계산 (Recomputation):
   - Forward에서 attention weights 저장 안 함
   - Backward에서 필요할 때 재계산
   - 메모리 절약 (O(n) vs O(n²))

3. 결과:
   - 메모리: O(n) vs O(n²)
   - 속도: 2-4x 빠름
   - 정확도: 수치적으로 동일
"""

# PyTorch 2.0+에서 사용
def use_flash_attention():
    import torch.nn.functional as F

    # Scaled Dot-Product Attention (Flash Attention 자동 사용)
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


# xFormers 사용
def use_xformers():
    from xformers.ops import memory_efficient_attention

    Q = torch.randn(2, 1024, 8, 64, device='cuda')
    K = torch.randn(2, 1024, 8, 64, device='cuda')
    V = torch.randn(2, 1024, 8, 64, device='cuda')

    output = memory_efficient_attention(Q, K, V)
    return output
```

---

## 3. 위치 인코딩 확장

### 3.1 문제: 학습 길이를 넘어서 외삽

```
학습: 4096 토큰
추론: 8192+ 토큰

문제:
- 절대 위치 인코딩: 4096 이후 위치 학습 안 됨
- RoPE: 보간/외삽 필요
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

    아이디어: 새 위치를 원본 범위로 스케일링

    position_ids를 [0, original_max_length)로 압축
    """
    scale = original_max_length / extended_max_length
    return position_ids.float() * scale


class RoPEWithInterpolation(nn.Module):
    """Position Interpolation이 적용된 RoPE"""

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

        # 주파수 계산
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 스케일 팩터
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
        # 위치 보간
        scaled_positions = position_ids.float() * self.scale

        # 주파수 계산
        freqs = torch.einsum('bi,d->bid', scaled_positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(2)  # (batch, seq, 1, dim)
        sin = emb.sin().unsqueeze(2)

        # RoPE 적용
        x_rope = self._apply_rope(x, cos, sin)

        return x_rope

    def _apply_rope(self, x, cos, sin):
        """RoPE 적용"""
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

    Position Interpolation의 문제:
    - 고주파 정보 손실 (높은 차원)

    YaRN 해결책:
    - 저주파: 보간 (interpolation)
    - 고주파: 외삽 (extrapolation)
    - NTK 스케일링으로 주파수 조정
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

        # 차원별 보간 비율 계산
        # 저주파 (낮은 차원): 더 많이 보간
        # 고주파 (높은 차원): 덜 보간 (외삽에 가까움)
        dims = torch.arange(0, dim, 2)
        low = max(0, math.floor(dim * math.log(scale) / (2 * math.log(original_max_length))))
        high = min(dim // 2 - 1, math.ceil(dim * math.log(scale) / (2 * math.log(original_max_length))))

        # 램프 함수로 보간/외삽 비율 결정
        ramp = torch.zeros(dim // 2)
        ramp[:low] = 0.0  # 외삽
        ramp[high:] = 1.0  # 보간

        if high > low:
            ramp[low:high] = (dims[low:high] - low) / (high - low)

        # NTK-aware base 조정
        inv_freq = 1.0 / (base ** (dims.float() / dim))

        # 보간과 외삽의 혼합
        inv_freq_inter = inv_freq / scale
        self.register_buffer(
            'inv_freq',
            (1 - ramp) * inv_freq + ramp * inv_freq_inter
        )

        # Attention scaling
        self.mscale = 0.1 * math.log(scale) + 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # 주파수 계산 (이미 조정된 inv_freq 사용)
        freqs = torch.einsum('bi,d->bid', position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(2) * self.mscale
        sin = emb.sin().unsqueeze(2) * self.mscale

        return self._apply_rope(x, cos, sin)
```

---

## 4. ALiBi (Attention with Linear Biases)

### 4.1 개념

```
ALiBi: 학습 없는 위치 인코딩

아이디어:
- 위치 인코딩을 사용하지 않음
- 대신, attention 점수에 거리 기반 패널티 추가
- 멀리 있는 토큰일수록 attention 점수 감소

Attention score modification:
score(q_i, k_j) = q_i · k_j - m × |i - j|

m: head별 기울기 (고정, 학습 안 함)
m_h = 2^(-8/H) for head h (H = 총 head 수)
```

### 4.2 구현

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

        # ALiBi slopes: 기하급수적으로 감소
        # 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # 거리 행렬 사전 계산
        positions = torch.arange(max_seq_len)
        distance_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
        distance_matrix = distance_matrix.abs()
        self.register_buffer('distance_matrix', distance_matrix)

    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Head별 ALiBi slope 계산"""

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # 가장 가까운 2의 거듭제곱으로 보간
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

        # Q, K, V 계산
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

### 5.1 개념

```
Ring Attention: 분산 Long Context

아이디어:
- 시퀀스를 여러 GPU에 분산
- 각 GPU가 자신의 청크 + 순환하는 KV 처리
- 통신과 계산 오버랩

┌────────────────────────────────────────────────┐
│                Ring Attention                   │
├────────────────────────────────────────────────┤
│                                                │
│  GPU 0: Q[0:n/4]     GPU 1: Q[n/4:n/2]        │
│          ↓ KV 순환        ↓ KV 순환            │
│  Step 1: K[0:n/4]    Step 1: K[n/4:n/2]       │
│  Step 2: K[n/4:n/2]  Step 2: K[n/2:3n/4]      │
│  Step 3: K[n/2:3n/4] Step 3: K[3n/4:n]        │
│  Step 4: K[3n/4:n]   Step 4: K[0:n/4]         │
│                                                │
│  KV가 링처럼 순환하며 각 GPU의 Q와 결합         │
│                                                │
└────────────────────────────────────────────────┘
```

### 5.2 구현 개요

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
    Ring Attention Forward Pass (개념적 구현)

    실제 구현은 CUDA 커널과 복잡한 동기화 필요
    """
    local_seq_len = Q.shape[1]

    # 누적 attention 출력
    output = torch.zeros_like(Q)
    max_scores = torch.full((Q.shape[0], Q.shape[2], local_seq_len), float('-inf'))
    sum_exp = torch.zeros_like(max_scores)

    # 현재 KV
    current_K = K.clone()
    current_V = V.clone()

    for step in range(world_size):
        # 이 청크의 KV에 대해 attention 계산
        scores = torch.matmul(Q, current_K.transpose(-2, -1))
        scores = scores / math.sqrt(Q.shape[-1])

        # Online softmax (numerically stable)
        new_max = torch.max(scores.max(dim=-1).values, max_scores)
        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))

        # 이전 결과 스케일 조정
        scale = torch.exp(max_scores - new_max)
        output = output * scale.unsqueeze(-1) + torch.matmul(exp_scores, current_V)

        sum_exp = sum_exp * scale + exp_scores.sum(dim=-1)
        max_scores = new_max

        # KV를 다음 GPU로 전송 (ring)
        if step < world_size - 1:
            # 비동기 send/recv
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1) % world_size

            # 다음 GPU에서 KV 수신
            current_K = ring_pass(current_K, send_rank, recv_rank)
            current_V = ring_pass(current_V, send_rank, recv_rank)

    # 최종 정규화
    output = output / sum_exp.unsqueeze(-1)

    return output


def ring_pass(tensor, send_rank, recv_rank):
    """Ring topology에서 텐서 전달"""
    recv_tensor = torch.empty_like(tensor)

    send_op = dist.isend(tensor, send_rank)
    recv_op = dist.irecv(recv_tensor, recv_rank)

    send_op.wait()
    recv_op.wait()

    return recv_tensor
```

---

## 6. 실용적 가이드

### 6.1 컨텍스트 확장 방법 선택

```
┌──────────────────────────────────────────────────────────────┐
│              언제 어떤 방법을 사용할까?                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  4K → 8K:                                                    │
│  - Position Interpolation (간단, 성능 좋음)                  │
│  - 약간의 fine-tuning 권장                                   │
│                                                              │
│  4K → 32K:                                                   │
│  - YaRN (PI보다 성능 좋음)                                   │
│  - 또는 ALiBi (처음부터 학습 시)                             │
│                                                              │
│  32K → 100K+:                                                │
│  - Flash Attention 필수                                      │
│  - Ring Attention (다중 GPU)                                 │
│  - Sparse Attention 고려                                     │
│                                                              │
│  1M+:                                                        │
│  - 특수 아키텍처 필요                                        │
│  - Mamba/State Space Models                                  │
│  - 또는 극도로 희소한 attention                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 실전 팁

```python
# 1. Gradient Checkpointing은 필수
model.gradient_checkpointing_enable()

# 2. Mixed Precision 사용
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(**inputs)

# 3. KV Cache 최적화 (추론 시)
# - Sliding Window Cache
# - Paged Attention (vLLM)

# 4. 청크 단위 처리 (긴 문서)
def process_long_document(model, document, chunk_size=4096, overlap=512):
    """긴 문서를 청크로 나눠 처리"""
    tokens = tokenizer.encode(document)
    results = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        output = model.generate(chunk)
        results.append(output)

    return merge_results(results)
```

---

## 참고 자료

### 논문
- Beltagy et al. (2020). "Longformer: The Long-Document Transformer"
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases"
- Peng et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models"

### 관련 레슨
- [08_LLaMA_Family.md](08_LLaMA_Family.md) - RoPE 기본
- [09_Mistral_MoE.md](09_Mistral_MoE.md) - Sliding Window Attention

---

## 연습 문제

### 연습 문제 1: 어텐션 복잡도 분석

n=8192 토큰, d_model=4096, 32개의 어텐션 헤드 (head_dim=128) 시퀀스에서:

1. 전체 어텐션 가중치 행렬(n × n)을 fp16으로 저장하는 데 필요한 메모리를 계산하세요.
2. 표준 전체 어텐션 vs FlashAttention의 이론적 피크 메모리는 얼마인가요? (FlashAttention은 타일 단위로 연산하여 어텐션 메모리를 O(n²)에서 O(n)으로 줄입니다.)
3. 어텐션 가중치 행렬만으로 40GB(단일 A100 GPU)를 초과하는 시퀀스 길이는 얼마인가요?

<details>
<summary>정답 보기</summary>

**1. 전체 어텐션 가중치 행렬 메모리 (n=8192):**

32개 헤드 각각의 어텐션 가중치 형태 (n × n):
```
32헤드 × 8192 × 8192 × 2바이트 (fp16)
= 32 × 67,108,864 × 2
= 4,294,967,296 바이트
≈ 4 GB
```

**2. 피크 메모리 비교:**

표준 어텐션은 전체 n×n 행렬을 구체화합니다:
- 어텐션 가중치: **~4 GB** (위 계산)
- 소프트맥스, 드롭아웃 등 포함: 약 2-3배 → 어텐션만으로 ~8-12 GB

FlashAttention은 전체 행렬을 저장하지 않고 타일(SRAM 블록)로 계산:
- O(n × d) = 8192 × 128 × 32 × 2바이트 ≈ **64 MB**
- n=8192에서 약 60-180배 적은 메모리

**3. 어텐션 행렬이 40GB를 초과하는 시점:**

어텐션 행렬 메모리 = 32 × n² × 2 바이트 = 64n²

```
64 × n² = 40 × 10⁹
n² = 40 × 10⁹ / 64 = 625,000,000
n = √625,000,000 ≈ 25,000 토큰
```

약 **25K 토큰**에서 어텐션 가중치 행렬만으로 단일 A100 GPU를 채웁니다 — 모델 파라미터, 활성화, 그래디언트를 제외하고.

</details>

---

### 연습 문제 2: "중간에서 길을 잃음(Lost in the Middle)" 현상

연구에서 LLM은 관련 정보가 긴 컨텍스트의 시작 또는 끝에 있을 때 가장 좋은 성능을 보이고, 중간에 있을 때 가장 나쁜 성능을 보임을 보여줍니다.

1. 학습 과정을 기반으로 이 현상이 발생하는 이유에 대한 가설을 제안하세요.
2. 질문에 답하기 위해 10개의 문서를 검색하는 RAG 시스템에서 두 가지 실용적인 완화 전략을 제안하세요.
3. 이 현상이 누적된 도구 출력이 컨텍스트에 배치되는 에이전트 시스템 설계에 어떤 영향을 미치나요?

<details>
<summary>정답 보기</summary>

**1. "중간에서 길을 잃음" 가설:**

자연 문서에 대한 학습 중, 예측을 위한 관련 정보는 주로:
- 시작 부분 (소개, 전제, 컨텍스트 설정)
- 끝 부분 (결론, 답변, 해결)

긴 문서의 중간 부분은 종종 다음 토큰 예측 학습 중 직접적인 "답변" 목표가 되는 빈도가 낮은 보조 세부사항, 전환, 상술이 포함됩니다. 모델의 어텐션 메커니즘은 암묵적으로 초기(절대 PE에서 강한 인과 위치 신호)와 후기 토큰에 더 높은 가중치를 부여하도록 학습될 수 있습니다.

**2. RAG 완화 전략:**

- **재순위화 및 위치 인식 정렬**: 가장 관련성 높은 검색 문서를 컨텍스트의 시작과 끝에 배치합니다. 10개의 문서와 관련성 추정이 가능하다면, #1과 #2를 위치 1과 10에 놓고 중간을 덜 관련성 있는 문서로 채웁니다.

- **중간 손실 인식 청킹**: 모든 문서를 연결하는 대신 "샌드위치" 프롬프트 구조를 사용합니다: 먼저 질문, 그 다음 컨텍스트, 그 다음 다시 질문. 또는 하나의 매우 긴 컨텍스트 대신 집계를 통한 여러 짧은 컨텍스트 창을 사용합니다.

**3. 에이전트 시스템 설계에 미치는 영향:**

다단계 에이전트 실행 중 축적된 도구 출력은 작업이 진행됨에 따라 컨텍스트를 성장시킵니다 — 초기 도구 결과가 중간으로 밀립니다. 이는 다음을 의미합니다:
- 초기의 중요한 결과(예: 사용자의 목표를 확립하는 초기 검색)가 "잊혀질" 수 있음
- 에이전트가 더 관련성이 높더라도 이전 결과보다 가장 최근의 도구 출력(끝에 있는)에 과도하게 의존할 수 있음
- **완화 방법**: 순수 컨텍스트 누적 대신 각 턴마다 컨텍스트 상단에서 갱신되는 실행 "스크래치패드(scratchpad)" 또는 구조화된 상태 요약을 사용합니다. 원시 도구 출력에는 최근성을 우선시하되 이전 단계의 중요한 발견에 대한 명시적 요약을 유지합니다.

</details>

---

### 연습 문제 3: 위치 인코딩(Position Encoding) 확장 전략

7B 모델이 4K 컨텍스트에서 RoPE로 학습되었습니다. 전체 재학습 없이 32K 컨텍스트로 확장하고 싶습니다. 세 가지 접근 방식을 비교하세요:

| 접근 방식 | 설명 | 주요 파라미터 | 트레이드오프 |
|---------|-----|------------|-----------|
| 32K에서 직접 추론 | 더 긴 길이에서 모델 실행 | 없음 | ? |
| Position Interpolation | 위치를 스케일 다운 | 스케일 인수 s = 8 | ? |
| YaRN | NTK 인식 보간 + 어텐션 온도 | 스케일 + 어텐션 인수 | ? |

<details>
<summary>정답 보기</summary>

| 접근 방식 | 설명 | 주요 파라미터 | 트레이드오프 |
|---------|-----|------------|-----------|
| **32K에서 직접 추론** | 학습에서 본 적 없는 위치 4097-32768 사용 | 없음 | 재앙적 — 4K 이상의 RoPE 값은 완전히 분포 외입니다. 모델이 이러한 회전 각도에 대한 학습된 동작이 없어 어텐션 패턴이 붕괴됩니다. 학습 길이를 초과하면 성능이 심각하게 저하됩니다. |
| **Position Interpolation** | 모든 위치를 s=32K/4K=8로 스케일링하여 최대 위치가 4K가 되도록 함 | s=8 (압축 인수) | 작동하지만 짧은 시퀀스의 품질이 저하됩니다: 스케일링 후 위치 1-512가 0-64로 압축되어 인접 토큰을 구별하기 어려워집니다. 품질 복원을 위해 일부 파인튜닝(100-1000 스텝)이 필요합니다. 직접 추론보다 낫고 YaRN보다 단순합니다. |
| **YaRN** | RoPE의 다른 주파수 구성 요소에 다른 스케일링 적용; 어텐션 온도 스케일링 추가 (1/√s 배수) | 스케일 + 어텐션 스케일링 인수 | 세 가지 중 최고 품질; 균일 스케일링으로 인한 주파수별 왜곡을 구체적으로 해결합니다. 어텐션 온도는 긴 범위에서 어텐션 분포가 너무 뾰족해지는 것을 방지합니다. 약간의 파인튜닝 권장 사항이 있지만 종종 제로샷(zero-shot)으로 작동합니다. 실제로 선호되는 방법입니다. |

**실용적 권장 사항:** YaRN을 사용하여 최고 품질의 확장을 위해; 단순함이 선호된다면 Position Interpolation. 학습 길이의 ~1.5-2배를 초과하는 직접 추론은 절대 사용하지 마세요.

</details>

---

### 연습 문제 4: FlashAttention 알고리즘 이해

FlashAttention의 핵심 혁신은 GPU SRAM에 맞는 "타일(tile)"로 어텐션을 계산하여 전체 O(n²) 어텐션 행렬을 구체화하지 않는 것입니다. 다음을 설명하세요:

1. n이 클 때 순진한 계산 `Attention(Q,K,V) = softmax(QK^T/√d)V`가 메모리 측면에서 왜 문제가 되나요?
2. FlashAttention이 전체 행렬 저장 없이 정확한(근사가 아닌) 어텐션을 계산하는 "온라인 소프트맥스(online softmax)" 트릭은 무엇인가요?
3. FlashAttention은 더 많은 산술 연산을 수행함에도 불구하고 왜 표준 어텐션보다 일반적으로 더 빠른가요?

<details>
<summary>정답 보기</summary>

**1. 순진한 어텐션의 메모리 문제:**

순진한 구현은 이러한 중간 텐서를 구체화합니다:
- `S = QK^T`: 형태 (n × n) — 전체 어텐션 스코어 행렬
- `P = softmax(S)`: 형태 (n × n) — 정규화된 어텐션 가중치
- `O = PV`: 형태 (n × d) — 출력

n=32K와 fp16의 경우: QK^T만으로 헤드당 32768 × 32768 × 2 바이트 ≈ 2 GB, 32헤드 × = 64 GB — GPU 메모리를 훨씬 초과합니다.

**2. 온라인 소프트맥스(online softmax) 트릭:**

표준 소프트맥스는 두 번의 패스가 필요합니다: 먼저 최대값 찾기(수치 안정성을 위해), 그 다음 exp 값 계산. FlashAttention은 점진적(온라인) 업데이트 규칙을 사용합니다:

처리되는 K,V의 각 새 타일에 대해:
```
# 새 블록 S_new = Q · K_new^T
# 실행 최대값 m과 exp 합계 l 유지
m_new = max(m_old, max(S_new))
l_new = exp(m_old - m_new) * l_old + sum(exp(S_new - m_new))
O_new = (exp(m_old - m_new) * l_old * O_old + exp(S_new - m_new) * V_new) / l_new
```

이를 통해 전체 n×n 행렬을 저장하지 않고 정확한 최종 소프트맥스를 계산할 수 있습니다 — 메모리에는 실행 통계(m, l)와 현재 출력(n × d)만 필요합니다.

**3. FlashAttention이 더 많은 산술 연산에도 불구하고 빠른 이유:**

표준 어텐션은 **메모리 대역폭 제한(memory-bandwidth bound)**이며 계산 제한이 아닙니다. 병목은 부동소수점 곱셈이 아니라 HBM(고대역폭 메모리)에서 대형 중간 행렬(S, P)을 읽고 쓰는 것입니다. HBM이 빠르더라도 헤드당 포워드 패스당 2 GB의 데이터를 읽는 것은 SRAM 연산과 비교해 매우 느립니다.

FlashAttention은 모든 중간 데이터를 SRAM(빠른 온칩 캐시)에 유지합니다:
- SRAM 대역폭: ~19 TB/s (vs HBM: ~2 TB/s)
- HBM 왕복 없이 SRAM에서 각 타일을 완전히 계산함으로써 HBM으로의 총 데이터 이동이 O(n) 배 줄어듦
- 이 벽시계(wall-clock) 속도 향상(2-4배)은 10-15% 더 많은 산술 연산에도 불구하고 발생

</details>

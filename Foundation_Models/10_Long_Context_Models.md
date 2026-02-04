# 10. Long Context Models

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

# 18. Attention 메커니즘 심화

## 학습 목표

- Multi-Head Attention 수학적 세부 이해
- Attention 복잡도 분석 (O(n^2))
- Flash Attention 원리
- Sparse Attention 기법들
- 위치 인코딩 심화 (RoPE, ALiBi)
- Attention 시각화 기법
- PyTorch 효율적 구현

---

## 1. Multi-Head Attention 수학

### 수식 복습

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Q = XW_Q  (query)
K = XW_K  (key)
V = XW_V  (value)

X: 입력 (batch, seq_len, d_model)
W_Q, W_K, W_V: 학습 가능한 가중치
```

### 상세 차원 분석

```python
# 입력
# X: (batch, seq_len, d_model)
# 예: (32, 512, 768)

# 가중치 행렬
# W_Q, W_K, W_V: (d_model, d_model)
# 예: (768, 768)

# Q, K, V 계산
# Q = X @ W_Q: (32, 512, 768)
# K = X @ W_K: (32, 512, 768)
# V = X @ W_V: (32, 512, 768)

# Multi-Head 분할
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

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention 상세 구현 (⭐⭐⭐)"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # 가중치 행렬
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

## 2. Attention 복잡도 분석

### 시간 복잡도

```
Q @ K^T: O(n * n * d) = O(n^2 * d)
    Q: (n, d), K^T: (d, n) → 결과: (n, n)

softmax: O(n^2)

weights @ V: O(n^2 * d)
    weights: (n, n), V: (n, d)

Total: O(n^2 * d)
```

### 공간 복잡도

```
Attention matrix: O(n^2)
    scores: (n, n) 저장 필요

긴 시퀀스 문제:
    n = 10,000, heads = 12
    메모리 = 12 * 10000 * 10000 * 4 bytes = 4.8 GB (float32)
```

### 병목점

```python
# 메모리 병목 시각화

n = [1000, 2000, 4000, 8000, 16000]
memory_gb = [h * n_i * n_i * 4 / (1024**3) for n_i in n for h in [12]]

# n=1000:  ~0.048 GB
# n=4000:  ~0.77 GB
# n=8000:  ~3.1 GB
# n=16000: ~12.3 GB

# GPU 메모리 한계로 긴 시퀀스 처리 불가
```

---

## 3. Flash Attention

### 핵심 아이디어

```
문제: attention matrix (n x n)을 한 번에 메모리에 올림 → 메모리 폭발

해결: 블록 단위로 계산, SRAM(빠른 on-chip 메모리) 활용

1. Q, K, V를 블록으로 분할
2. 각 블록에서 attention 계산 (SRAM에서)
3. Online softmax로 블록 결과 병합
```

### Online Softmax

```python
def online_softmax(scores_blocks):
    """Online Softmax 알고리즘 (⭐⭐⭐⭐)

    블록별로 softmax를 계산하고 병합
    전체 점수를 메모리에 올리지 않음
    """
    # Block 1: [s1, s2, s3]
    # Block 2: [s4, s5, s6]

    # 일반 softmax:
    # exp(s1 - max_all) / sum_all_exp

    # Online:
    # 블록마다 local_max, local_sum 유지
    # 새 블록 볼 때 global_max, global_sum 업데이트

    max_so_far = float('-inf')
    sum_so_far = 0

    for block in scores_blocks:
        block_max = block.max()
        new_max = max(max_so_far, block_max)

        # 기존 sum 조정
        sum_so_far = sum_so_far * math.exp(max_so_far - new_max)
        # 새 블록 추가
        sum_so_far += (block - new_max).exp().sum()

        max_so_far = new_max

    return max_so_far, sum_so_far
```

### PyTorch 2.0+ Flash Attention

```python
# PyTorch 2.0부터 내장 지원

def flash_attention_example():
    """Flash Attention 사용 예시 (⭐⭐⭐)"""
    batch_size = 32
    seq_len = 4096
    num_heads = 12
    head_dim = 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

    # PyTorch 2.0+ scaled_dot_product_attention
    # Flash Attention 자동 적용 (조건 충족 시)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        output = F.scaled_dot_product_attention(Q, K, V)

    return output

# 메모리 비교
# 일반: O(n^2)
# Flash: O(n) - attention matrix를 저장하지 않음
```

### 성능 비교

```python
def benchmark_attention(seq_lens, batch_size=8, num_heads=12, head_dim=64):
    """Attention 성능 벤치마크 (⭐⭐)"""
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

    각 토큰이 주변 window_size 토큰만 참조
    복잡도: O(n * window_size)
    """
    def __init__(self, d_model, num_heads, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # 마스크 생성: window 밖은 -inf
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

    일정 간격으로 떨어진 토큰만 참조
    예: stride=4면 0, 4, 8, 12, ... 만 참조
    """
    def __init__(self, d_model, num_heads, stride=4):
        super().__init__()
        self.stride = stride
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # 마스크: stride 간격 토큰만 attend
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

    3가지 패턴 조합:
    1. Local: 주변 토큰
    2. Global: 특정 토큰(CLS 등)이 모든 토큰 참조
    3. Random: 랜덤 토큰 참조
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

        # 2. Global tokens (처음 num_global 토큰)
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
    Global: 특정 토큰 (CLS, SEP 등)만 전체 참조
    """
    def __init__(self, d_model, num_heads, window_size=256, global_indices=None):
        super().__init__()
        self.window_size = window_size
        self.global_indices = global_indices or [0]  # 기본: CLS만
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

## 5. 위치 인코딩 심화

### Sinusoidal (원본 Transformer)

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal 위치 인코딩 (⭐⭐)"""
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
    """학습 가능한 위치 인코딩 (⭐⭐)"""
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
    """RoPE - LLaMA, GPT-NeoX 등에서 사용 (⭐⭐⭐⭐)

    위치 정보를 회전 행렬로 인코딩
    상대 위치 정보를 자연스럽게 표현
    """
    def __init__(self, dim, max_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

        # 캐싱
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
        # x를 반으로 나눠서 회전
        x1, x2 = x[..., ::2], x[..., 1::2]
        # 회전 적용
        rotated = torch.stack(
            [-x2, x1], dim=-1
        ).flatten(-2)

        return x * cos + rotated * sin


def apply_rope(q, k, rope_module, seq_len):
    """RoPE 적용 예시"""
    return rope_module(q, k, seq_len)
```

### ALiBi (Attention with Linear Biases)

```python
class ALiBiPositionalBias(nn.Module):
    """ALiBi - MPT, BLOOM 등에서 사용 (⭐⭐⭐⭐)

    위치 임베딩 대신 attention score에 선형 bias 추가
    학습 파라미터 없음
    길이 외삽 능력 우수
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # 각 head마다 다른 기울기
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

    def _get_slopes(self, n):
        """기울기 계산: 2^(-8/n), 2^(-16/n), ..."""
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
        # 상대 거리 행렬
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.abs().unsqueeze(0)

        # bias = -slope * distance
        alibi = -self.slopes.unsqueeze(1).unsqueeze(1) * relative_positions

        return alibi  # (num_heads, seq_len, seq_len)


def attention_with_alibi(Q, K, V, alibi_bias):
    """ALiBi 적용 attention (⭐⭐⭐)"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # ALiBi bias 추가
    scores = scores + alibi_bias

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

---

## 6. Attention 시각화

### Attention Weights 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens=None, layer=0, head=0):
    """단일 head attention 시각화 (⭐⭐)"""
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
    """모든 head 시각화 (⭐⭐)"""
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

### Attention Flow (BertViz 스타일)

```python
def plot_attention_flow(attention_weights, tokens, top_k=3):
    """상위 K개 attention 연결 시각화 (⭐⭐⭐)"""
    weights = attention_weights[0, 0].cpu().detach().numpy()
    seq_len = len(tokens)

    plt.figure(figsize=(12, 8))

    # 토큰 위치
    left_positions = [(0, i) for i in range(seq_len)]
    right_positions = [(1, i) for i in range(seq_len)]

    # 토큰 표시
    for i, token in enumerate(tokens):
        plt.text(-0.1, i, token, ha='right', va='center', fontsize=10)
        plt.text(1.1, i, token, ha='left', va='center', fontsize=10)

    # 상위 K개 연결
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

## 7. 효율적 구현

### PyTorch scaled_dot_product_attention

```python
def efficient_attention_comparison():
    """효율적 attention 구현 비교 (⭐⭐⭐)"""
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
    """메모리 효율적 Attention (청크 기반) (⭐⭐⭐⭐)"""
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

        # 청크별 처리로 메모리 절약
        outputs = []
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            Q_chunk = Q[:, :, i:end]

            # 이 청크에 대해 전체 K, V와 attention
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self.scale
            weights = F.softmax(scores, dim=-1)
            chunk_out = torch.matmul(weights, V)
            outputs.append(chunk_out)

        output = torch.cat(outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(B, N, self.d_model)

        return self.W_o(output)
```

---

## 8. Attention 패턴 분석

### 자주 관찰되는 패턴

```python
def analyze_attention_patterns(attention_weights):
    """Attention 패턴 분석 (⭐⭐⭐)"""
    # attention_weights: (batch, heads, seq, seq)

    patterns = {}

    for head in range(attention_weights.size(1)):
        weights = attention_weights[0, head].cpu().detach()

        # 1. Diagonal (자기 자신 주목)
        diagonal = torch.diag(weights).mean()

        # 2. Previous token (이전 토큰)
        prev_token = torch.diag(weights, -1).mean() if weights.size(0) > 1 else 0

        # 3. First token (CLS 등)
        first_token = weights[:, 0].mean()

        # 4. Uniform (균일 분포)
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

## 정리

### 핵심 개념

1. **MHA 수학**: Q, K, V 투영 → 스케일링 → softmax → 가중합
2. **복잡도**: O(n^2) 시간/공간 - 긴 시퀀스 병목
3. **Flash Attention**: 블록 처리 + Online softmax → O(n) 메모리
4. **Sparse Attention**: Local, Strided, BigBird 등
5. **위치 인코딩**: Sinusoidal, Learned, RoPE, ALiBi

### 효율화 요약

| 기법 | 시간 | 공간 | 특징 |
|-----|------|------|------|
| Standard | O(n^2) | O(n^2) | 기본 |
| Flash | O(n^2) | O(n) | HW 최적화 |
| Local | O(nw) | O(nw) | window w |
| Sparse | O(n sqrt(n)) | O(n) | 패턴 조합 |

### PyTorch 실전 팁

```python
# 1. PyTorch 2.0+ 사용
output = F.scaled_dot_product_attention(Q, K, V)

# 2. Flash Attention 강제 활성화
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V)

# 3. 긴 시퀀스는 청킹 고려

# 4. RoPE/ALiBi는 길이 외삽에 유리
```

---

## 연습 문제

### 연습 1: 어텐션 메모리 스케일링 측정

어텐션 메모리가 시퀀스 길이에 따라 어떻게 증가하는지 경험적으로 측정하세요.

1. CPU에서 float32로 `(1, 1, seq_len, 64)` 형태의 Q, K, V 텐서를 할당하는 함수를 작성하세요.
2. `seq_len` 값 512, 1024, 2048, 4096에 대해 표준 어텐션(`Q @ K.T / sqrt(d_k)` 후 소프트맥스, `weights @ V`)을 실행하세요.
3. Python의 `tracemalloc` 또는 (CUDA 사용 가능 시) `torch.cuda.max_memory_allocated()`로 각 길이에서의 최대 메모리를 기록하세요.
4. 로그-로그 스케일로 메모리 vs 시퀀스 길이를 그래프로 표시하고 O(n²) 추세를 확인하세요.
5. A100 GPU(80 GB)에서 float16, 12-헤드 MHA로 메모리가 부족해지는 시퀀스 길이를 계산하세요.

### 연습 2: 온라인 소프트맥스(Online Softmax) 검증

수업의 온라인 소프트맥스 알고리즘이 표준 소프트맥스와 동일한 결과를 내는지 검증하세요.

1. 100개 원소로 이루어진 점수 벡터를 만드세요.
2. 표준 소프트맥스 `exp(scores - max) / sum(exp(scores - max))`를 계산하세요.
3. 점수를 25개씩 4개 블록으로 처리하는 온라인 소프트맥스 알고리즘을 구현하세요.
4. 출력을 원소별로 비교하여 `1e-6` 이내로 일치하는지 확인하세요.
5. 수치 안정성을 위해 최댓값을 빼는 이유와, 생략했을 때 어떤 문제가 발생하는지 설명하세요.

### 연습 3: 사전 학습 모델의 어텐션 패턴 분석

사전 학습된 Transformer에서 다양한 어텐션 헤드가 어떤 패턴을 학습하는지 분석하세요.

1. `transformers`에서 소형 사전 학습 Transformer를 불러오세요 (예: `distilbert-base-uncased`).
2. 평서문, 의문문, 목록 등 3가지 다른 문장을 모델에 전달하세요.
3. 레이어 2의 6개 어텐션 헤드를 각 문장에 대해 추출하고 시각화하세요.
4. 수업의 `analyze_attention_patterns` 함수로 각 헤드가 대각선(국소), 첫 번째 토큰, 또는 균일 패턴을 보이는지 식별하세요.
5. 문장 유형에 따라 다르게 동작하는 헤드를 보고하고, 각 헤드가 어떤 언어 현상을 전문화하는지 가설을 제시하세요.

### 연습 4: RoPE vs 정현파(Sinusoidal) 인코딩

길이 외삽(Length Extrapolation) 태스크에서 두 위치 인코딩 방식을 비교하세요.

1. 정현파 인코딩을 사용하는 소형 Transformer와 RoPE를 사용하는 Transformer를 각각 2개 구성하세요 (2 레이어, `d_model=128`).
2. 두 모델 모두 길이 64의 시퀀스에서 복사 태스크(입력 토큰을 출력에서 예측)로 학습하세요.
3. 두 모델을 길이 128의 시퀀스(분포 외 길이)에서 평가하세요.
4. 두 길이에서의 정확도를 보고하고, RoPE가 미경험 길이에 더 잘 일반화되는 이유를 설명하세요.

### 연습 5: 로컬 어텐션(Local Attention) vs 전체 어텐션 트레이드오프

긴 시퀀스 태스크에서 `LocalAttention`과 전체 어텐션을 구현하고 비교하세요.

1. `seq_len=2048`의 긴 시퀀스 태스크를 생성하세요: 레이블은 위치 0의 토큰(전역 정보).
2. 전체 어텐션을 사용하는 2층 Transformer와 `LocalAttention(window_size=128)`을 사용하는 Transformer를 각각 학습하세요.
3. 두 모델의 에포크당 학습 시간과 최대 메모리 사용량을 측정하세요.
4. 20 에포크 후 정확도를 비교하세요.
5. 로컬 어텐션 모델이 이 특정 태스크에서 어려움을 겪는 이유와, BigBird의 전역 토큰이 이 문제를 어떻게 해결하는지 설명하세요.

---

## 참고 자료

- Flash Attention: https://arxiv.org/abs/2205.14135
- RoPE: https://arxiv.org/abs/2104.09864
- ALiBi: https://arxiv.org/abs/2108.12409
- BigBird: https://arxiv.org/abs/2007.14062

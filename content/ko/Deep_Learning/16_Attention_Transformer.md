# 16. Attention과 Transformer

[이전: LSTM / GRU 구현](./15_Impl_LSTM_GRU.md) | [다음: Attention 메커니즘 심화](./17_Attention_Deep_Dive.md)

---

## 학습 목표

- Attention 메커니즘의 원리
- Self-Attention 이해
- Transformer 아키텍처
- PyTorch 구현

---

## 이론과 원리

Transformer (Vaswani et al. 2017)는 약 2년 안에 NLP에서 전체 RNN 패밀리를 대체했습니다. 그 이유는 attention이 점화식보다 "더 똑똑한" 메커니즘이라서가 아닙니다 — attention이 시퀀스 축에 걸친 완전한 병렬성을 가능하게 하면서 장거리 의존성 모델링을 보존(그리고 개선)하기 때문입니다. 이 섹션은 scaled dot-product attention의 수학, `sqrt(d_k)` 정규화가 선택사항이 아닌 이유, 그리고 multi-head가 단일 attention head를 넘어 무엇을 사는지 설명합니다.

이 섹션에서 다루는 내용:

- **A.** 부드러운, 내용 주소화된 룩업으로서의 attention
- **B.** Query/key/value에서 유도된 scaled dot-product attention
- **C.** `sqrt(d_k)`가 중요한 이유: 분산 보존
- **D.** 부분공간 분해로서의 multi-head attention

### A. 부드러운 룩업으로서의 Attention

해시 테이블은 하드 룩업입니다: `lookup(query) -> value`이며 `query`가 키와 정확히 일치해야 합니다. Attention은 이를 실수값 벡터에 대한 *부드러운* 룩업으로 일반화합니다:

```
attention(query, keys, values) = sum_i softmax(score(query, k_i)) * v_i
```

출력은 모든 값의 볼록 결합이며, 각 키가 query와 얼마나 잘 일치하는지로 가중됩니다. `score`가 한 키에 더 날카롭게 정점일수록 attention은 하드 룩업에 가까워지고; 평탄해지면 평균에 가까워집니다. 메커니즘이 완전히 미분 가능하기 때문에, 신경망 안에서 종단간(end-to-end) 학습될 수 있습니다.

이 관점은 Transformer 층이 무엇을 하는지도 명확하게 합니다: 각 토큰이 query를 발행하고, 모든 다른 토큰이 key+value 쌍을 제공하며, 층은 각 토큰을 모든 값의 가중 혼합으로 다시 씁니다.

### B. Scaled Dot-Product Attention

Vaswani et al.은 가능한 가장 단순한 점수 함수를 선택했습니다: scaled dot product. Query, key, value를 행렬로 묶음 `Q in R^{n x d_k}`, `K in R^{n x d_k}`, `V in R^{n x d_v}`. 그러면:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

차원:

- `Q K^T`는 `n x n`, **attention 점수 행렬**: `(QK^T)_{ij}`는 query `i`가 key `j`와 얼마나 강하게 일치하는지 측정.
- 행 방향 `softmax(... / sqrt(d_k))`이 **attention 가중치**를 만들며, 각 행이 1로 합쳐짐.
- `V`를 곱하면 query별 출력: 값 벡터의 가중 합.

전체 계산은 두 번의 행렬 곱과 softmax — GPU에 완벽히 적합하며, (결정적으로) 모든 `n` query 위치에 걸쳐 병렬화 가능. RNN은 본질적으로 순차적; attention은 본질적으로 병렬.

### C. `sqrt(d_k)`인 이유: 분산 보존

`q`와 `k`가 평균 0, 분산 1의 i.i.d. 성분을 가진 벡터라고 가정합니다. 그들의 내적은:

```
q . k = sum_{i=1}^{d_k} q_i k_i
```

각 항이 평균 0, 분산 1이므로 `Var(q . k) = d_k` (독립성)이고 `std(q . k) = sqrt(d_k)`. 재스케일링 없이는 attention 점수가 `d_k`에 따라 자라는 표준편차를 가집니다. 일반적인 `d_k = 64`의 경우, 점수가 쉽게 8 이상의 크기에 도달할 수 있습니다.

이제 그것들을 softmax에 넣어보세요: `softmax([8, 0, 0]) ≈ [0.9997, 0.00015, 0.00015]`. Softmax가 가장 큰 점수에서 거의 완전히 포화됩니다. 더 나쁜 것은, 그 그래디언트가 거의 0에 가까워집니다 — softmax 도함수는 `p_i (1 - p_i)`를 포함하며, 이는 `p_i \approx 1`일 때 사라집니다. **포화된 softmax는 죽은 그래디언트를 의미**하며, 이는 attention 층이 학습할 수 없음을 의미합니다.

`sqrt(d_k)`로 나누면 점수가 `d_k`와 무관하게 `O(1)` 표준편차로 다시 스케일되어, softmax를 비포화 영역에 유지합니다. 이는 휴리스틱이 아닙니다; 이는 원본 논문이 정확히 이 분석을 통해 발견한 *필수* 설계 선택입니다.

### D. Multi-Head: 부분공간 분해

단일 head attention은 시퀀스의 모든 관계를 하나의 attention 패턴에 강제합니다. Multi-head attention은 `h`개 병렬 attention 층을 실행하며, 각각이 *투영된 부분공간*에서 작동:

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)        W_i^Q, W_i^K in R^{d_model x d_k} 등
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

각 head는 입력의 다른 `d_k`-차원 투영을 봅니다. `h` head와 `d_k = d_model / h`로, 총 파라미터 수는 전체 차원의 한 head와 같습니다 — multi-head 구조는 더 많은 파라미터를 추가하는 게 아니라 하나의 큰 attention을 `h`개의 작은 전문화된 것으로 *인수분해*하는 방법입니다.

실험적으로, 다른 head들은 다른 관계를 학습합니다: 일부는 구문 의존성에, 일부는 공참조(coreference)에, 일부는 위치 상대 패턴에 attention합니다. 이 모두를 단일 attention 패턴에 강제하는 것은 별도 head로 공존하게 두는 것보다 엄격히 덜 표현력 있습니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| 부드러운 룩업 | `softmax(QK^T) V` |
| Scaled dot product | Attention 공식의 `/ sqrt(d_k)` 제수 |
| 포화 softmax 회피 | `/ sqrt(d_k)`를 제거하면 학습이 깨지는 이유 |
| Multi-head 부분공간 | `Q.view(B, n, h, d_k).transpose(1, 2)` 후 head별 attention |

---


## 1. Attention의 필요성

### Seq2Seq의 한계

```
인코더: "나는 학교에 간다" → 고정 크기 벡터
                              ↓
디코더: 고정 벡터 → "I go to school"

문제: 긴 문장의 정보가 압축되어 손실
```

### Attention의 해결

```
디코더가 각 출력 단어를 생성할 때
인코더의 모든 단어를 "주목"할 수 있음

"I" 생성 시 → "나는"에 높은 attention
"school" 생성 시 → "학교"에 높은 attention
```

---

## 2. Attention 메커니즘

### 수식

```python
# Query, Key, Value
Q = 현재 디코더 상태
K = 인코더 모든 상태
V = 인코더 모든 상태 (보통 K와 동일)

# Attention Score
score = Q @ K.T  # (query_len, key_len)

# Attention Weight (softmax)
weight = softmax(score / sqrt(d_k))  # 스케일링

# Context
context = weight @ V  # 가중 합
```

### Scaled Dot-Product Attention

```python
def attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

---

## 3. Self-Attention

### 개념

```
같은 시퀀스 내에서 각 단어가 다른 모든 단어에 attention

"The cat sat on the mat because it was tired"
"it"이 "cat"에 높은 attention → 대명사 해석
```

### 수식

```python
# 입력 X에서 Q, K, V 생성
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Self-Attention
output = attention(Q, K, V)
```

---

## 4. Multi-Head Attention

### 아이디어

```
여러 개의 attention head가 서로 다른 관계 학습

Head 1: 문법적 관계
Head 2: 의미적 관계
Head 3: 위치 관계
...
```

### 수식

```python
def multi_head_attention(Q, K, V, num_heads):
    d_model = Q.size(-1)
    d_k = d_model // num_heads

    # 헤드 분할
    Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
    K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)

    # 각 헤드에서 attention
    attn_output, _ = attention(Q, K, V)

    # 헤드 결합
    output = attn_output.transpose(1, 2).contiguous().view(batch, seq, d_model)
    return output
```

---

## 5. Transformer 아키텍처

### 구조

```
입력 → Embedding → Positional Encoding
                      ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│           ↓                         │
│  Add & LayerNorm                    │
│           ↓                         │
│  Feed Forward Network               │
│           ↓                         │
│  Add & LayerNorm                    │
└─────────────────────────────────────┘
            × N layers
                ↓
             출력
```

### 핵심 컴포넌트

1. **Multi-Head Attention**
2. **Position-wise Feed Forward**
3. **Residual Connection**
4. **Layer Normalization**
5. **Positional Encoding**

---

## 6. Positional Encoding

### 필요성

```
Attention은 순서 정보가 없음
→ 위치 정보를 명시적으로 추가
```

### Sinusoidal Encoding

```python
def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE
```

---

## 7. PyTorch Transformer

### 기본 사용

```python
import torch.nn as nn

# Transformer 인코더
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# 순전파
x = torch.randn(10, 32, 512)  # (seq, batch, d_model)
output = encoder(x)
```

### 분류 모델

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 평균 풀링
        return self.fc(x)
```

---

## 8. Vision Transformer (ViT)

### 아이디어

```
이미지를 패치로 분할 → 시퀀스로 처리

이미지 (224×224) → 16×16 패치 196개 → Transformer
```

### 구조

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model, nhead, num_layers):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 패치 추출 및 임베딩
        patches = extract_patches(x)
        x = self.patch_embed(patches)

        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 위치 임베딩
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x.transpose(0, 1))

        # 분류 (CLS 토큰 사용)
        return self.fc(x[0])
```

---

## 9. Attention vs RNN 비교

| 항목 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 병렬화 | 어려움 | 용이 |
| 장거리 의존성 | 어려움 | 용이 |
| 학습 속도 | 느림 | 빠름 |
| 메모리 | O(n) | O(n²) |
| 위치 정보 | 암시적 | 명시적 |

---

## 10. 실전 활용

### NLP

- BERT: 양방향 인코더
- GPT: 디코더 기반 생성
- T5: 인코더-디코더

### Vision

- ViT: 이미지 분류
- DETR: 객체 검출
- Swin Transformer: 계층적 구조

---

## 11. Multi-Query Attention (MQA)과 Grouped-Query Attention (GQA)

표준 Multi-Head Attention(MHA)은 각 헤드마다 고유한 Key와 Value 프로젝션을 가집니다. 이는 표현력이 뛰어나지만, 자기회귀(Autoregressive) 추론 시 모든 헤드의 K와 V를 모든 과거 위치에 대해 캐시해야 하므로 메모리 병목이 됩니다. MQA와 GQA는 이 오버헤드를 극적으로 줄이는 두 가지 중요한 최적화 기법입니다.

### 11.1 KV 캐시(KV Cache) 문제

자기회귀 생성(예: GPT가 한 번에 하나의 토큰 생성) 시, 재계산을 피하기 위해 이전 모든 위치의 K, V 텐서를 캐시합니다. 이 **KV 캐시**의 크기는 다음과 같이 증가합니다:

```
KV cache memory = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes_per_param

For a standard MHA model (e.g., LLaMA 1 65B):
  n_layers=80, n_heads=64, head_dim=128, seq_len=2048, FP16
  = 2 × 80 × 64 × 2048 × 128 × 2 bytes ≈ 5.2 GB per sequence!

This limits batch size and maximum sequence length at inference time.
```

### 11.2 Multi-Query Attention (MQA)

**MQA** (Shazeer, 2019)는 모든 쿼리 헤드에서 하나의 공유 K와 V를 사용합니다:

```
Standard MHA:
  Q: (batch, n_heads, seq, head_dim)   ← separate per head
  K: (batch, n_heads, seq, head_dim)   ← separate per head
  V: (batch, n_heads, seq, head_dim)   ← separate per head

MQA:
  Q: (batch, n_heads, seq, head_dim)   ← separate per head
  K: (batch, 1, seq, head_dim)         ← ONE shared K
  V: (batch, 1, seq, head_dim)         ← ONE shared V

  K and V are broadcast across all query heads.

KV cache reduced by factor of n_heads (e.g., 64× less memory).
```

### 11.3 Grouped-Query Attention (GQA)

**GQA** (Ainslie et al., 2023)는 중간 지점입니다: 1개의 공유 KV(MQA)도 아니고 n_heads개의 개별 KV(MHA)도 아닌, **G개 그룹**을 사용하며 각 쿼리 헤드 그룹이 하나의 K와 V를 공유합니다.

```
MHA (n_kv_heads = n_heads):      Each head has its own K, V
GQA (1 < n_kv_heads < n_heads):  Groups of heads share K, V
MQA (n_kv_heads = 1):            All heads share one K, V

Example with 8 query heads, 2 KV groups:
  Q heads: [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
  KV groups: [KV_0, KV_1]

  Q0, Q1, Q2, Q3 → attend using KV_0
  Q4, Q5, Q6, Q7 → attend using KV_1

LLaMA 2 70B uses GQA with 64 query heads and 8 KV heads.
Mistral 7B uses GQA with 32 query heads and 8 KV heads.
```

### 11.4 메모리 및 지연 시간 비교

```
┌──────────────────┬───────────┬───────────┬───────────┐
│ 지표             │ MHA       │ GQA       │ MQA       │
├──────────────────┼───────────┼───────────┼───────────┤
│ KV 헤드 수       │ n_heads   │ n_groups  │ 1         │
│ KV 캐시 크기     │ 1×        │ G/H ×     │ 1/H ×     │
│ KV 파라미터      │ 1×        │ G/H ×     │ 1/H ×     │
│ 품질             │ 최고      │ MHA에 근접 │ 약간 ↓    │
│ 추론 속도        │ 가장 느림 │ 빠름      │ 가장 빠름 │
│ 디코딩 지연      │ 최고      │ 낮음      │ 최저      │
├──────────────────┼───────────┼───────────┼───────────┤
│ 예시             │ GPT-3     │ LLaMA 2   │ PaLM      │
│ (H=heads, G=grp) │ H=96,G=96│ H=64,G=8  │ H=16,G=1  │
│ KV 캐시 절감     │ 0%        │ 87.5%     │ 93.75%    │
└──────────────────┴───────────┴───────────┴───────────┘

H = 총 쿼리 헤드 수, G = KV 그룹 수
KV 캐시 절감률 = (1 - G/H) × 100%
```

### 11.5 GQA의 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Why GQA over MHA: At inference time, the KV cache is the primary
    memory bottleneck. By sharing K/V across groups of query heads,
    GQA reduces cache size by (n_heads / n_kv_heads)× while retaining
    most of MHA's quality. This is the approach used by LLaMA 2 70B,
    Mistral 7B, and many modern LLMs.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # How many Q heads per KV group
        self.head_dim = d_model // n_heads

        # Q has n_heads projections; K, V have only n_kv_heads projections
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        V = self.W_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        K = K.transpose(1, 2)  # (batch, n_kv_heads, seq, head_dim)
        V = V.transpose(1, 2)  # (batch, n_kv_heads, seq, head_dim)

        # Expand K, V to match Q's head count by repeating within groups
        # Each KV head is shared by (n_heads // n_kv_heads) query heads
        K = K.repeat_interleave(self.n_groups, dim=1)  # (batch, n_heads, seq, head_dim)
        V = V.repeat_interleave(self.n_groups, dim=1)  # (batch, n_heads, seq, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.W_o(attn_output)


# Example: LLaMA 2 70B style GQA
gqa = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=2)
x = torch.randn(2, 10, 512)  # batch=2, seq=10
output = gqa(x)
print(f"Output shape: {output.shape}")  # (2, 10, 512)

# Compare parameter counts
mha_kv_params = 2 * 512 * 512  # MHA: W_k + W_v, both (512, 512)
gqa_kv_params = 2 * 512 * 128  # GQA: W_k + W_v, both (512, 128) for 2 KV heads
print(f"MHA KV params: {mha_kv_params:,}")    # 524,288
print(f"GQA KV params: {gqa_kv_params:,}")    # 131,072
print(f"KV param reduction: {(1 - gqa_kv_params/mha_kv_params)*100:.0f}%")  # 75%
```

### 11.6 현대 LLM에서의 GQA 사용 현황

```
┌─────────────────────┬──────────┬───────────┬───────────────────────┐
│ 모델                │ Q 헤드   │ KV 헤드   │ Attention 유형        │
├─────────────────────┼──────────┼───────────┼───────────────────────┤
│ GPT-3 175B          │ 96       │ 96        │ MHA                   │
│ PaLM 540B           │ 16       │ 1         │ MQA                   │
│ LLaMA 1 65B         │ 64       │ 64        │ MHA                   │
│ LLaMA 2 70B         │ 64       │ 8         │ GQA (8 groups)        │
│ Mistral 7B          │ 32       │ 8         │ GQA (4 groups)        │
│ Gemma 7B            │ 16       │ 16        │ MHA                   │
│ Falcon 40B          │ 64       │ 1         │ MQA                   │
│ Qwen2 72B           │ 64       │ 8         │ GQA                   │
└─────────────────────┴──────────┴───────────┴───────────────────────┘

추세: GQA는 7B 이상 모델에서 기본이 되었습니다.
품질-효율 트레이드오프(Trade-off)가 가장 우수하기 때문입니다.

핵심 통찰: MQA는 대형 모델에서 품질이 눈에 띄게 저하될 수 있습니다.
8개 KV 헤드의 GQA는 MHA의 품질을 거의 유지하면서
MQA의 속도 이점 대부분을 제공합니다.
```

---

## 정리

### 핵심 개념

1. **Attention**: Query-Key-Value로 관련성 계산
2. **Self-Attention**: 시퀀스 내 모든 위치 참조
3. **Multi-Head**: 다양한 관계 동시 학습
4. **Positional Encoding**: 순서 정보 추가

### 핵심 코드

```python
# Scaled Dot-Product Attention
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V

# PyTorch Transformer
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
)
```

---

## 연습 문제

### 연습 1: 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention) 직접 구현

NumPy만 사용하여 스케일드 닷 프로덕트 어텐션을 구현하세요 (PyTorch 불사용).

1. 형태 `(4, 8)` (seq_len=4, d_k=8)의 난수 Q, K, V 행렬을 만드세요.
2. 어텐션 점수 `Q @ K.T`를 계산하고, `1/sqrt(d_k)` 스케일링 인수를 적용한 뒤 소프트맥스를 적용하세요.
3. 결과 가중치를 V에 곱하여 컨텍스트(Context) 행렬을 구하세요.
4. 동일한 값으로 PyTorch의 `F.scaled_dot_product_attention`과 결과가 일치하는지 확인하세요.
5. `sqrt(d_k)`로 나누는 이유를 설명하세요: 스케일링을 생략하면 어떤 문제가 발생하나요?

### 연습 2: 자기 어텐션(Self-Attention) 시각화

짧은 문장에 어텐션 가중치를 시각화하여 어떤 단어가 어떤 단어에 어텐션하는지 확인하세요.

1. 수업의 `MultiHeadAttention` 모듈(또는 `nn.MultiheadAttention`)을 구성하세요.
2. 짧은 문장(예: "The cat sat on the mat")을 단어 수준에서 토크나이즈하고 `nn.Embedding`으로 각 단어를 임베딩하세요.
3. `return_attention=True`로 순전파를 실행하고 어텐션 가중치를 추출하세요.
4. 두 축 모두 단어 레이블을 붙인 히트맵으로 어텐션 가중치 행렬을 시각화하세요.
5. 흥미로운 어텐션 패턴을 하나 이상 식별하세요 (예: "sat"이 가장 강하게 어텐션하는 단어는?).

### 연습 3: 분류를 위한 Transformer 인코더 구성

PyTorch의 `nn.TransformerEncoder`로 시퀀스를 분류하세요.

1. 합성 데이터를 만드세요: 길이 20인 정수 시퀀스, 평균이 50보다 높으면 1, 낮으면 0으로 레이블.
2. `d_model=64`, `nhead=4`, `num_layers=2`로 `TransformerClassifier`를 구성하세요.
3. 인코더 뒤에 시퀀스를 단일 벡터로 집계하는 평균 풀링(Mean Pooling) 단계를 추가하세요.
4. 30 에포크 학습하고 정확도를 보고하세요.
5. 비슷한 파라미터 수를 가진 단순 LSTM 분류기와 비교하세요 — 어느 것이 더 빨리 수렴하나요?

### 연습 4: GQA vs MHA 파라미터 및 메모리 분석

그룹 쿼리 어텐션(Grouped-Query Attention)의 메모리 절감 효과를 정량적으로 측정하세요.

1. 다음 설정으로 `GroupedQueryAttention`을 인스턴스화하세요: `(d_model=512, n_heads=8, n_kv_heads=8)` (MHA), `(d_model=512, n_heads=8, n_kv_heads=4)` (GQA-4), `(d_model=512, n_heads=8, n_kv_heads=1)` (MQA).
2. 각 설정의 KV 프로젝션 파라미터 수를 계산하세요.
3. 1개 레이어 기준으로 `seq_len=2048`, float16에서의 이론적 KV 캐시 크기를 MB 단위로 계산하세요.
4. 형태 `(2, 32, 512)`의 배치로 순전파를 실행하고 모든 출력의 형태가 동일한지 확인하세요.
5. 트레이드오프를 요약하세요: MQA의 메모리 절감 이점이 품질 저하를 능가하는 시점은 언제인가요?

### 연습 5: 인과적(Causal, 자기회귀) 어텐션 마스크

인과 마스크를 구현하고 미래 토큰에 어텐션하지 않음을 확인하세요.

1. 위치 `i`가 `0..i` 위치에만 어텐션할 수 있는 형태 `(seq_len, seq_len)`의 인과 마스크를 만드세요 (상삼각 부분은 `-inf`).
2. `scaled_dot_product_attention` 내부에 이 마스크를 적용하세요.
3. 미래 위치에 대한 소프트맥스 가중치가 정확히 0임을 확인하여 마스크가 올바르게 작동하는지 검증하세요.
4. 소형 인과 Transformer(2 레이어, `d_model=64`)를 구성하고, 50개 빈으로 이산화된 사인파에서 다음 토큰을 예측하도록 학습하세요.
5. 데이터의 홀드아웃 10%에 대한 다음 토큰 예측 정확도를 보고하세요.

---

## 다음 단계

[학습 최적화](./23_Training_Optimization.md)에서 고급 학습 기법을 배웁니다.

# 03. Transformer 복습

## 학습 목표

- NLP 관점에서 Transformer 이해
- Encoder와 Decoder 구조
- 언어 모델링 관점의 Attention
- BERT/GPT 기반 구조 이해

---

## 1. Transformer 개요

### 구조 요약

```
인코더 (BERT 스타일):
    입력 → [Embedding + Positional] → [Self-Attention + FFN] × N → 출력

디코더 (GPT 스타일):
    입력 → [Embedding + Positional] → [Masked Self-Attention + FFN] × N → 출력

인코더-디코더 (T5 스타일):
    입력 → 인코더 → [Cross-Attention] → 디코더 → 출력
```

### NLP에서의 역할

| 모델 | 구조 | 용도 |
|------|------|------|
| BERT | 인코더 only | 분류, QA, NER |
| GPT | 디코더 only | 텍스트 생성 |
| T5, BART | 인코더-디코더 | 번역, 요약 |

---

## 2. Self-Attention (NLP 관점)

### 문장 내 관계 학습

```
"The cat sat on the mat because it was tired"

"it" → Attention → "cat" (높은 가중치)
                → "mat" (낮은 가중치)

모델이 대명사 "it"이 "cat"을 지칭함을 학습
```

### Query, Key, Value

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V 계산
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Multi-head 분할
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq, d_k)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # 헤드 결합
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attention_weights
```

---

## 3. Causal Masking (GPT 스타일)

### 자기회귀 언어 모델

```
"I love NLP" 학습:
    입력: [I]         → 예측: love
    입력: [I, love]   → 예측: NLP
    입력: [I, love, NLP] → 예측: <eos>

미래 토큰을 보면 안 됨 → Causal Mask 필요
```

### Causal Mask 구현

```python
def create_causal_mask(seq_len):
    """하삼각 마스크 생성 (미래 토큰 차단)"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 = 참조 가능, 0 = 마스킹

# 예시 (seq_len=4)
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=512):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        # 미리 계산된 마스크 등록
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask)

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.mask[:seq_len, :seq_len]
        return self.attention(x, mask)
```

---

## 4. Encoder vs Decoder

### 인코더 (양방향)

```python
class TransformerEncoderBlock(nn.Module):
    """BERT 스타일 인코더 블록"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # Self-Attention (양방향)
        attn_out, _ = self.self_attn(x, padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

### 디코더 (단방향)

```python
class TransformerDecoderBlock(nn.Module):
    """GPT 스타일 디코더 블록"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Masked Self-Attention (단방향)
        attn_out, _ = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

---

## 5. Positional Encoding

### Sinusoidal (원본 Transformer)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Learnable (BERT, GPT)

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
```

---

## 6. Complete Transformer Model

### GPT-스타일 언어 모델

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 토큰 + 위치 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Decoder 블록
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (선택)
        self.head.weight = self.token_embedding.weight

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape

        # 임베딩
        tok_emb = self.token_embedding(x)
        pos = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        # Transformer 블록
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab_size)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """자기회귀 텍스트 생성"""
        for _ in range(max_new_tokens):
            # 마지막 위치의 logits
            logits = self(idx)[:, -1, :]  # (batch, vocab)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
```

### BERT-스타일 인코더

```python
class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # 문장 구분

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # 임베딩 결합
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        seg_emb = self.segment_embedding(segment_ids)

        x = tok_emb + pos_emb + seg_emb

        # Transformer 블록
        for block in self.blocks:
            x = block(x, attention_mask)

        return self.ln_f(x)
```

---

## 7. 학습 목표별 비교

### Masked Language Modeling (BERT)

```
입력: "The [MASK] sat on the mat"
예측: [MASK] → "cat"

15% 토큰을 마스킹하여 예측
양방향 문맥 활용
```

### Causal Language Modeling (GPT)

```
입력: "The cat sat on"
예측: "the" "cat" "sat" "on" "the" "mat"

다음 토큰 예측
단방향 (왼쪽→오른쪽)
```

### Seq2Seq (T5, BART)

```
입력: "translate English to French: Hello"
출력: "Bonjour"

인코더: 입력 이해
디코더: 출력 생성
```

---

## 8. PyTorch 내장 Transformer

```python
import torch.nn as nn

# 인코더
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# 디코더
decoder_layer = nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

# 사용
x = torch.randn(32, 100, 512)  # (batch, seq, d_model)
encoded = encoder(x)
decoded = decoder(x, encoded)
```

---

## 정리

### 모델 비교

| 항목 | BERT (인코더) | GPT (디코더) | T5 (Enc-Dec) |
|------|--------------|-------------|--------------|
| Attention | 양방향 | 단방향 (Causal) | 양방향 + 단방향 |
| 학습 | MLM + NSP | 다음 토큰 예측 | Denoising |
| 출력 | 문맥 벡터 | 생성 | 생성 |
| 용도 | 분류, QA | 생성, 대화 | 번역, 요약 |

### 핵심 코드

```python
# Causal Mask
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, -1e9)

# Multi-Head Attention 분할
Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)

# Scaled Dot-Product
scores = Q @ K.T / sqrt(d_k)
attn = softmax(scores) @ V
```

---

## 연습 문제

### 연습 문제 1: 인과 마스크(Causal Mask) 동작

길이 5인 시퀀스에 대한 전체 인과 마스크 행렬을 손으로 쓰거나 코드로 작성하세요. 그런 다음 시퀀스의 3번째 토큰(인덱스 2)이 어떤 토큰에 어텐션할 수 있는지, 그리고 그 이유를 설명하세요. 자기회귀(autoregressive) 생성 중에 이 마스크가 없다면 어떤 일이 발생할까요?

<details>
<summary>정답 보기</summary>

```python
import torch

def create_causal_mask(seq_len):
    """하삼각 행렬: 1 = 어텐션 가능, 0 = 차단"""
    return torch.tril(torch.ones(seq_len, seq_len))

mask = create_causal_mask(5)
print(mask)
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

**인덱스 2(3번째 토큰)에 대한 마스크 해석**:
- 2행은 `[1, 1, 1, 0, 0]`입니다.
- 위치 0, 1, 2의 토큰(자기 자신과 모든 이전 토큰)에 어텐션할 수 있습니다.
- 위치 3, 4의 토큰(미래 토큰)에는 어텐션할 수 **없습니다**.

**어텐션 계산에서의 효과**:
```python
# mask=0인 위치는 softmax 전에 -1e9 점수를 받음
scores[mask == 0] = -1e9
# softmax 이후 이 위치들의 어텐션 가중치 ≈ 0
# 따라서 출력 벡터는 과거+현재 토큰의 가중합만으로 이루어짐
```

**인과 마스크가 없다면**:
- 학습 중: 모델이 다음 단어를 예측하기 위해 미래 토큰을 "커닝"할 수 있어 태스크가 지나치게 쉬워지고 실제 생성에는 쓸모없는 모델이 됩니다.
- 추론 중: 미래 토큰은 아직 존재하지 않으므로 (하나씩 생성 중) 모델이 일관성 없는 출력을 생성하거나 모든 토큰을 미리 알아야 합니다.

인과 마스크는 **자기회귀 특성**을 강제합니다: 위치 `t`의 예측은 위치 `0`부터 `t-1`까지만 의존합니다.

</details>

### 연습 문제 2: 인코더 vs 디코더 아키텍처 차이점

다음 비교 표를 채우고, BERT를 개방형 텍스트 생성에 직접 사용할 수 없는 이유와 GPT를 완전한 양방향 이해가 필요한 태스크에 직접 사용할 수 없는 이유를 각각 한 문장으로 설명하세요.

| 특징 | BERT (인코더) | GPT (디코더) |
|------|--------------|-------------|
| 어텐션 유형 | ? | ? |
| 학습 목표 | ? | ? |
| 일반적인 용도 | ? | ? |
| 미래 토큰 참조 가능? | ? | ? |

<details>
<summary>정답 보기</summary>

| 특징 | BERT (인코더) | GPT (디코더) |
|------|--------------|-------------|
| 어텐션 유형 | 양방향 셀프 어텐션(self-attention) | 인과적(causal) 단방향 셀프 어텐션 |
| 학습 목표 | MLM(Masked Language Model) + NSP(Next Sentence Prediction) | 다음 토큰 예측 (CLM) |
| 일반적인 용도 | 분류, NER, QA, 유사도 | 텍스트 생성, 대화, 자동 완성 |
| 미래 토큰 참조 가능? | 예 (전체 시퀀스 가시) | 아니오 (과거 토큰만 가시) |

**BERT가 텍스트를 생성할 수 없는 이유**:
BERT는 양쪽 문맥이 주어진 상황에서 마스킹된 토큰을 채우도록 학습되었습니다. 추론 시 생성은 토큰 `t+1`이 존재하기 전에 토큰 `t`를 예측해야 하지만, BERT는 자기회귀 생성 메커니즘이 없어 완전한(마스킹된) 입력을 기대하지 확장할 부분 입력을 기대하지 않습니다.

**GPT가 양방향 태스크를 잘 수행하지 못하는 이유**:
GPT의 인과 마스킹은 각 토큰의 표현이 과거 토큰으로만 계산됨을 의미합니다. NER이나 QA처럼 토큰의 레이블이 미래 문맥에 의존할 수 있는 태스크(예: "Washington"이 사람인지 장소인지 판단하려면 이후 단어를 확인해야 함)에서 GPT의 단방향 어텐션은 근본적인 한계를 가집니다.

</details>

### 연습 문제 3: 위치 인코딩(Positional Encoding) 특성

사인파 위치 인코딩은 공식 `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`에 기반한 주파수를 사용합니다. 이 인코딩을 계산하는 함수를 구현하고 두 가지 핵심 특성을 확인하세요: (1) 서로 다른 위치는 서로 다른 인코딩을 생성한다, (2) 위치 간의 상대적 오프셋은 절대적 위치에 관계없이 일정하다.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn.functional as F
import math

def sinusoidal_encoding(max_len, d_model):
    """사인파 위치 인코딩 계산"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원
    return pe

pe = sinusoidal_encoding(max_len=100, d_model=64)

# 특성 1: 서로 다른 위치는 서로 다른 인코딩을 생성
pos_5 = pe[5]
pos_10 = pe[10]
pos_50 = pe[50]

sim_5_10 = F.cosine_similarity(pos_5.unsqueeze(0), pos_10.unsqueeze(0)).item()
sim_5_50 = F.cosine_similarity(pos_5.unsqueeze(0), pos_50.unsqueeze(0)).item()
print(f"위치 5 vs 10 유사도: {sim_5_10:.4f}")   # < 1.0 (구별됨)
print(f"위치 5 vs 50 유사도: {sim_5_50:.4f}")   # 더 덜 유사 (더 멀리 떨어짐)

# 특성 2: 일정한 상대적 오프셋
# pe[pos] · pe[pos+k]의 내적은 같은 k에 대해 어떤 pos에서도 유사해야 함
offset = 5
dots = []
for pos in [0, 10, 20, 50]:
    dot = (pe[pos] * pe[pos + offset]).sum().item()
    dots.append(dot)
    print(f"내적 pe[{pos}] · pe[{pos+offset}] = {dot:.4f}")

# 모든 값이 대략 동일해야 함
print(f"내적의 표준편차: {torch.tensor(dots).std():.4f}")  # 작아야 함
```

**핵심 특성**:
1. 각 위치는 고유한 인코딩 벡터를 가집니다 (코사인 유사도 < 1.0으로 확인).
2. 고정된 오프셋 `k`에 대해 내적 `pe[pos] · pe[pos+k]`는 절대적 `pos`에 관계없이 대략 일정합니다. 이를 통해 모델은 위치에 걸쳐 일반화되는 상대적 위치 패턴을 학습할 수 있습니다.

이러한 특성 덕분에 사인파 인코딩은 학습 중에 보지 못한 길이의 시퀀스에도 적용 가능합니다 — 학습된 위치 임베딩에 비해 핵심적인 장점입니다.

</details>

### 연습 문제 4: 언어 모델의 가중치 공유(Weight Tying)

`GPTModel` 구현에서 `self.head.weight = self.token_embedding.weight`라는 줄이 있습니다. 이 문맥에서 "가중치 공유(weight tying)"가 무엇을 의미하는지, 왜 이 방법을 사용하는지, 그리고 실질적인 이점은 무엇인지 설명하세요.

<details>
<summary>정답 보기</summary>

**가중치 공유의 의미**:

출력 프로젝션 레이어(`self.head`)는 은닉 차원에서 어휘 크기로 매핑하여 다음 토큰 예측을 위한 로짓(logit)을 생성합니다. 토큰 임베딩 행렬은 어휘 인덱스에서 은닉 차원으로 매핑합니다. 가중치 공유는 이 두 행렬을 메모리에서 동일한 객체로 설정합니다:

```python
# 가중치 공유 없이: 두 개의 별도 행렬
# embedding: (vocab_size, d_model)  →  token_id를 벡터로 매핑
# head:      (d_model, vocab_size)  →  벡터를 토큰당 로짓으로 매핑

# 가중치 공유: 동일한 데이터를 공유
self.head.weight = self.token_embedding.weight
# head.weight는 사실상 embedding.weight.T
# (PyTorch 선형 레이어는 W @ x를 사용하므로 가중치 형태는 (out, in) = (vocab, d_model))
# 즉: 입력 임베딩과 출력 임베딩이 동일한 행렬
```

**사용 이유**:

단어의 임베딩 벡터가 은닉 상태 `h`에 가깝다면, 모델은 그 단어를 다음 토큰으로 높은 확률을 할당해야 한다는 우아한 대칭성이 있습니다. 입력 임베딩과 출력 프로젝션 행렬은 동일한 의미 공간에서 역방향 연산을 수행합니다.

**실질적인 이점**:

1. **파라미터 감소**: `vocab_size × d_model` 행렬 하나를 제거합니다. GPT-2의 경우 vocab_size=50,257, d_model=768이면 약 3,860만 개의 파라미터를 절약합니다.

2. **정규화**: 입력 및 출력 임베딩을 동일하게 강제하면 언어 모델링 목표에서 과적합을 방지하는 암묵적 제약이 추가됩니다.

3. **더 좋은 임베딩 품질**: 공유 행렬은 임베딩(입력) 방향과 예측(출력) 방향 모두에서 그레이디언트 업데이트를 받아 일반적으로 더 높은 품질의 단어 표현을 만들어냅니다.

```python
# 실제로 가중치 공유 검증
import torch.nn as nn

class TiedModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # 가중치 공유

# 파라미터 수 계산
model = TiedModel()
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터: {total_params}")  # 64,000 (행렬의 단일 복사본만)
# 공유 없이: 128,000 파라미터 (두 개의 별도 행렬)
```

</details>

## 다음 단계

[BERT 이해](./04_BERT_Understanding.md)에서 BERT의 구조와 학습 방법을 상세히 학습합니다.

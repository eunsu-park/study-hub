# 05. GPT 이해

## 학습 목표

- GPT 아키텍처 이해
- 자기회귀 언어 모델링
- 텍스트 생성 기법
- GPT 시리즈 발전

---

## 1. GPT 개요

### Generative Pre-trained Transformer

```
GPT = Transformer 디코더 스택

특징:
- 단방향 (왼쪽→오른쪽)
- 자기회귀 생성
- 다음 토큰 예측으로 학습
```

### BERT vs GPT

| 항목 | BERT | GPT |
|------|------|-----|
| 구조 | 인코더 | 디코더 |
| 방향 | 양방향 | 단방향 |
| 학습 | MLM | 다음 토큰 예측 |
| 용도 | 이해 (분류, QA) | 생성 (대화, 작문) |

---

## 2. 자기회귀 언어 모델링

### 학습 목표

```
P(x) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ...

문장: "I love NLP"
P("I") × P("love"|"I") × P("NLP"|"I love") × P("<eos>"|"I love NLP")

손실: -log P(다음 토큰 | 이전 토큰들)
```

### Causal Language Modeling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_lm_loss(logits, targets):
    """
    logits: (batch, seq, vocab_size)
    targets: (batch, seq) - 다음 토큰

    입력: [BOS, I, love, NLP]
    타겟: [I, love, NLP, EOS]
    """
    batch_size, seq_len, vocab_size = logits.shape

    # (batch*seq, vocab) vs (batch*seq,)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-100  # 패딩 무시
    )
    return loss
```

---

## 3. GPT 아키텍처

### 구조

```
입력 토큰
    ↓
Token Embedding + Position Embedding
    ↓
┌─────────────────────────────────┐
│  Masked Multi-Head Attention    │
│           ↓                     │
│  Add & LayerNorm                │
│           ↓                     │
│  Feed Forward                   │
│           ↓                     │
│  Add & LayerNorm                │
└─────────────────────────────────┘
            × N layers
    ↓
LayerNorm
    ↓
Linear (vocab_size)
    ↓
Softmax → 다음 토큰 확률
```

### 구현

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-LayerNorm (GPT-2 스타일)
        ln_x = self.ln1(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        ln_x = self.ln2(x)
        x = x + self.ffn(ln_x)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        # Causal mask 등록
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_len

        # 임베딩
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Causal mask
        mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer 블록
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab)

        return logits
```

---

## 4. 텍스트 생성

### Greedy Decoding

```python
def generate_greedy(model, input_ids, max_new_tokens):
    """항상 가장 확률 높은 토큰 선택"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Temperature Sampling

```python
def generate_with_temperature(model, input_ids, max_new_tokens, temperature=1.0):
    """Temperature로 분포 조절"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

# temperature < 1: 더 결정적 (높은 확률 토큰 선호)
# temperature > 1: 더 무작위 (다양성 증가)
```

### Top-k Sampling

```python
def generate_top_k(model, input_ids, max_new_tokens, k=50, temperature=1.0):
    """상위 k개 토큰에서만 샘플링"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature

            # Top-k 필터링
            top_k_logits, top_k_indices = logits.topk(k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            # 샘플링
            idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Top-p (Nucleus) Sampling

```python
def generate_top_p(model, input_ids, max_new_tokens, p=0.9, temperature=1.0):
    """누적 확률 p까지의 토큰에서 샘플링"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # 확률 내림차순 정렬
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            # p 이후 토큰 마스킹
            mask = cumsum - sorted_probs > p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # 샘플링
            idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

---

## 5. GPT 시리즈

### GPT-1 (2018)

```
- 12 레이어, 768 차원, 117M 파라미터
- BooksCorpus로 학습
- 파인튜닝 패러다임 도입
```

### GPT-2 (2019)

```
- 최대 48 레이어, 1.5B 파라미터
- WebText (40GB) 학습
- Zero-shot 능력 발견
- "Too dangerous to release"

크기 변형:
- Small: 117M (GPT-1과 동일)
- Medium: 345M
- Large: 762M
- XL: 1.5B
```

### GPT-3 (2020)

```
- 96 레이어, 175B 파라미터
- Few-shot / In-context Learning
- API로만 제공

주요 발견:
- 프롬프트만으로 다양한 태스크 수행
- 스케일링 법칙: 모델 크기 ↑ = 성능 ↑
```

### GPT-4 (2023)

```
- 멀티모달 (텍스트 + 이미지)
- 더 긴 컨텍스트 (8K, 32K, 128K)
- 향상된 추론 능력
- RLHF로 정렬
```

---

## 6. HuggingFace GPT-2

### 기본 사용

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 텍스트 생성
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 생성
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### 생성 파라미터

```python
output = model.generate(
    input_ids,
    max_length=100,           # 최대 길이
    min_length=10,            # 최소 길이
    do_sample=True,           # 샘플링 사용
    temperature=0.8,          # 온도
    top_k=50,                 # Top-k
    top_p=0.95,               # Top-p
    num_return_sequences=3,   # 생성 개수
    no_repeat_ngram_size=2,   # n-gram 반복 방지
    repetition_penalty=1.2,   # 반복 페널티
    pad_token_id=tokenizer.eos_token_id
)
```

### 조건부 생성

```python
# 프롬프트 기반 생성
prompt = """
Q: What is the capital of France?
A:"""

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=False  # Greedy
)
print(tokenizer.decode(output[0]))
```

---

## 7. In-Context Learning

### Zero-shot

```
프롬프트만으로 태스크 수행:

"Translate English to French:
Hello, how are you? →"
```

### Few-shot

```
예제를 프롬프트에 포함:

"Translate English to French:
Hello → Bonjour
Thank you → Merci
Good morning → Bonjour
How are you? →"
```

### Chain-of-Thought (CoT)

```
단계별 추론 유도:

"Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
How many balls does he have now?
A: Let's think step by step.
Roger started with 5 balls.
2 cans of 3 balls each = 6 balls.
5 + 6 = 11 balls.
The answer is 11."
```

---

## 8. KV Cache

### 효율적인 생성

```python
class GPTWithKVCache(nn.Module):
    def forward(self, input_ids, past_key_values=None):
        """
        past_key_values: 이전 토큰의 K, V 캐시
        새 토큰에 대해서만 계산 후 캐시 업데이트
        """
        if past_key_values is None:
            # 전체 시퀀스 계산
            ...
        else:
            # 마지막 토큰만 계산
            ...

        return logits, new_past_key_values

# 생성 시
past = None
for _ in range(max_new_tokens):
    logits, past = model(new_token, past_key_values=past)
    # O(n) 대신 O(1) 복잡도
```

### HuggingFace KV Cache

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True  # KV Cache 활성화 (기본값)
)
```

---

## 정리

### 생성 전략 비교

| 방법 | 장점 | 단점 | 용도 |
|------|------|------|------|
| Greedy | 빠름, 일관성 | 반복, 지루함 | 번역, QA |
| Temperature | 다양성 조절 | 튜닝 필요 | 일반 생성 |
| Top-k | 안정적 | 고정 k | 일반 생성 |
| Top-p | 적응적 | 약간 느림 | 창작, 대화 |

### 핵심 코드

```python
# HuggingFace GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 생성
output = model.generate(input_ids, max_length=50, do_sample=True,
                        temperature=0.7, top_p=0.9)
```

---

## 연습 문제

### 연습 문제 1: 생성 전략 비교

HuggingFace의 GPT-2를 사용하여 동일한 프롬프트에서 네 가지 다른 전략으로 텍스트를 생성하세요: 탐욕적 디코딩(greedy decoding), 온도 샘플링(temperature sampling, T=0.5), top-k 샘플링(k=50), top-p 샘플링(p=0.9). 출력을 비교하고 실제 애플리케이션에서 각 전략을 언제 선택할지 설명하세요.

<details>
<summary>정답 보기</summary>

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

def decode(output):
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 1. 탐욕적 디코딩 - 항상 가장 가능성 높은 토큰 선택
greedy = model.generate(input_ids, max_new_tokens=30, do_sample=False)
print("탐욕적:", decode(greedy))
# 결정적, 종종 반복적

# 2. 온도 샘플링 (T=0.5) - 더 날카로운 분포, 덜 무작위
low_temp = model.generate(
    input_ids, max_new_tokens=30, do_sample=True, temperature=0.5
)
print("\nTEMP=0.5:", decode(low_temp))
# 더 집중적이고, 다양성은 낮지만 여전히 변화 있음

# 3. Top-k 샘플링 (k=50)
top_k = model.generate(
    input_ids, max_new_tokens=30, do_sample=True, top_k=50
)
print("\nTOP-K=50:", decode(top_k))
# 매우 낮은 확률의 토큰 제외, 안정적인 품질

# 4. Top-p (nucleus) 샘플링 (p=0.9)
top_p = model.generate(
    input_ids, max_new_tokens=30, do_sample=True, top_p=0.9, temperature=1.0
)
print("\nTOP-P=0.9:", decode(top_p))
# 누적 확률에 기반한 적응적 어휘 크기
```

**각 전략을 사용해야 하는 경우**:

| 전략 | 가장 좋은 경우 | 이유 |
|------|--------------|------|
| 탐욕적 | 번역, 사실 기반 QA | 가능도 최대화, 일관적이고 재현 가능 |
| 낮은 온도 | 코드 생성, 공식 텍스트 | 제어된 창의성, 거의 결정적 |
| 높은 온도 | 브레인스토밍, 시 | 높은 다양성, 일관성 희생 가능 |
| Top-k | 대화, 챗봇 | 희귀한 아티팩트(artifact)를 방지하면서 다양성 허용 |
| Top-p | 창작 글쓰기, 스토리텔링 | 문맥 복잡성에 따라 어휘 크기 적응 |

실제로 **top-p와 temperature를 결합** (예: `p=0.9, temperature=0.8`)하는 것이 두 가지 제어 방식을 모두 결합하여 범용 생성에 가장 일반적으로 사용되는 전략입니다.

</details>

### 연습 문제 2: KV 캐시(KV Cache) 메모리 절약

자기회귀 생성 중 KV 캐시(Key-Value Cache)의 계산 이점을 설명하세요. 12개의 어텐션 레이어를 가정하고, 50토큰 프롬프트에서 100개의 새 토큰을 생성할 때 캐시 없이(키와 값 행렬의 재계산 횟수)와 캐시 있을 때의 계산량을 비교하세요.

<details>
<summary>정답 보기</summary>

**KV 캐시 없이**:

각 생성 단계 `t`에서 모델은 지금까지 본 전체 시퀀스(프롬프트 + 생성된 토큰)에 대해 K와 V를 계산합니다. 따라서 단계 `t`에서 `50 + t` 토큰을 12개 레이어 모두에서 처리합니다.

```python
# KV 캐시 없이: 총 KV 계산량
prompt_len = 50
new_tokens = 100
num_layers = 12

# 각 새 토큰마다 모든 이전 토큰의 K와 V를 재계산
total_kv_without_cache = 0
for t in range(new_tokens):
    seq_len = prompt_len + t + 1  # 현재 시퀀스 길이
    total_kv_without_cache += seq_len * num_layers

print(f"캐시 없이 총 KV 계산: {total_kv_without_cache}")
# = sum(51부터 150까지) * 12 = 10050 * 12 = 120,600

# KV 캐시 있을 때: 새 토큰에 대해서만 K와 V 계산
total_kv_with_cache = new_tokens * num_layers
print(f"캐시 있을 때 총 KV 계산: {total_kv_with_cache}")
# = 100 * 12 = 1,200

speedup = total_kv_without_cache / total_kv_with_cache
print(f"속도 향상: {speedup:.1f}x")
# ≈ KV 계산에서 100.5배 속도 향상
```

**KV 캐시 작동 방식**:

```python
# KV 캐시 메커니즘 개념
class AttentionWithCache:
    def forward(self, x, past_kv=None):
        # 현재 토큰에 대해서만 Q, K, V 계산
        q = self.W_q(x)  # 새 토큰에 대해서만: (batch, 1, d_k)
        k = self.W_k(x)  # 새 토큰에 대해서만: (batch, 1, d_k)
        v = self.W_v(x)  # 새 토큰에 대해서만: (batch, 1, d_k)

        if past_kv is not None:
            past_k, past_v = past_kv
            # 이전 단계의 캐시된 K, V와 연결
            k = torch.cat([past_k, k], dim=1)  # (batch, seq+1, d_k)
            v = torch.cat([past_v, v], dim=1)

        # 전체 K, V를 사용하되 새 Q만으로 어텐션 수행
        attn = softmax(q @ k.T / sqrt(d_k)) @ v  # (batch, 1, d_k)

        return attn, (k, v)  # 업데이트된 캐시 반환
```

**메모리 트레이드오프**: KV 캐시는 계산을 메모리와 교환합니다 — 모든 이전 토큰의 K와 V를 저장해야 합니다. GPT-3의 경우 96개 레이어, 1750억 파라미터, 컨텍스트 길이 4096에서: 각 K와 V 행렬은 `(batch, seq, 128, d_k)` 형태이며, 캐시만으로도 약 10GB의 GPU 메모리가 필요합니다. 이것이 LLM 추론에서 신중한 메모리 관리가 필요한 이유입니다.

</details>

### 연습 문제 3: 인-컨텍스트 러닝(In-Context Learning) 프롬프트 설계

영화 리뷰를 긍정/부정으로 분류하는 텍스트 분류 태스크를 위해 세 가지 버전의 프롬프트를 설계하세요: 제로샷(zero-shot), 퓨샷(few-shot, 3개 예제), 그리고 연쇄적 사고(chain-of-thought). 각각이 점진적으로 모델 성능을 향상시키는 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 버전 1: 제로샷 (Zero-shot)
zero_shot_prompt = """Classify the following movie review as Positive or Negative.

Review: "The acting was superb and the story kept me engaged throughout."
Sentiment:"""

# 버전 2: 퓨샷 (Few-shot, 3개 예제)
few_shot_prompt = """Classify the following movie review as Positive or Negative.

Review: "Absolutely terrible. I walked out after 30 minutes."
Sentiment: Negative

Review: "One of the best films I've seen this decade. Masterpiece!"
Sentiment: Positive

Review: "Mediocre plot but the cinematography saved it somewhat."
Sentiment: Negative

Review: "The acting was superb and the story kept me engaged throughout."
Sentiment:"""

# 버전 3: 연쇄적 사고 (Chain-of-Thought)
cot_prompt = """Classify the following movie review as Positive or Negative.
Think step by step before giving your final answer.

Review: "Absolutely terrible. I walked out after 30 minutes."
Reasoning: The reviewer says "absolutely terrible" which is very negative, and they
left early (walked out after 30 minutes), showing they couldn't finish watching.
Sentiment: Negative

Review: "The acting was superb and the story kept me engaged throughout."
Reasoning:"""
```

**각 접근법이 점진적으로 성능을 향상시키는 이유**:

**제로샷**: 사전학습 중 학습된 패턴에 전적으로 의존합니다. 모델은 형식만으로 태스크를 추론해야 합니다. 모델이 학습 중에 유사한 형식을 본 적이 있는 단순한 태스크에서 작동합니다.

**퓨샷**: 구체적인 입력-출력 예제를 제공하여:
- 태스크 형식 명확화 ("Sentiment:"가 어떻게 보여야 하는지)
- 출력 어휘 시연 ("Positive", "Negative" — "pos", "neg", 또는 "good"이 아님)
- 실제 예제로 모델의 결정 경계 조정

GPT-3 논문은 퓨샷 성능이 표준 벤치마크에서 파인튜닝된 모델과 종종 일치한다는 것을 보여주었습니다.

**연쇄적 사고(CoT)**: 모델이 다음을 수행하도록 강제합니다:
- 텍스트에서 관련 증거 식별
- 답을 확정하기 전에 명시적으로 추론
- "성급한 결론" 오류 감소

CoT는 감성이 즉시 명확하지 않은 미묘한 리뷰(예: 혼합 리뷰, 비꼬기)에서 특히 유용합니다. 중간 추론 단계는 모델의 결정을 더 해석 가능하게 만들기도 합니다.

</details>

### 연습 문제 4: 자기회귀 학습 설정

소형 문자 수준 GPT 모델에 대한 완전한 학습 루프를 작성하세요. 모델은 문자 단위로 시퀀스를 생성하는 방법을 학습해야 합니다. 입력과 타깃 시퀀스가 어떻게 구성되는지, 인과적 언어 모델링 손실이 어떻게 계산되는지, 학습 진행 상황을 어떻게 모니터링하는지 보여주세요.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_model*4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # 가중치 공유

    def forward(self, input_ids, causal_mask=None):
        seq_len = input_ids.size(1)
        if causal_mask is None:
            # 인과 마스크 생성: True = 마스킹됨 (어텐션 불가)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
            ).bool()

        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)

        return self.head(self.ln_f(x))

# 문자 수준 데이터셋 준비
text = "Hello, World! This is a training example for our tiny GPT model."
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

# 텍스트 인코딩
data = torch.tensor([stoi[c] for c in text])

def get_batch(data, block_size=32, batch_size=4):
    """CLM 학습을 위한 입력/타깃 쌍 생성"""
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[s:s+block_size] for s in starts])
    # 타깃은 1칸 이동된 입력: 다음 문자 예측
    y = torch.stack([data[s+1:s+block_size+1] for s in starts])
    return x, y

# 학습 루프
model = TinyGPT(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(200):
    x, y = get_batch(data)
    logits = model(x)  # (batch, seq, vocab_size)

    # 인과적 LM 손실: 각 다음 토큰 예측
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),  # (batch*seq, vocab)
        y.view(-1)                    # (batch*seq,)
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그레이디언트 클리핑
    optimizer.step()

    if step % 50 == 0:
        print(f"스텝 {step}: 손실 = {loss.item():.4f}, "
              f"퍼플렉시티(perplexity) = {torch.exp(loss).item():.2f}")

# 생성
model.eval()
with torch.no_grad():
    start = torch.tensor([[stoi['H']]])  # 'H'로 시작
    for _ in range(30):
        logits = model(start)
        next_char = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        start = torch.cat([start, next_char], dim=1)
    print("생성:", ''.join([itos[i.item()] for i in start[0]]))
```

**핵심 설계 결정 설명**:
- **입력 vs 타깃 오프셋**: `x = data[t:t+L]`, `y = data[t+1:t+L+1]` — 즉, `x`의 위치 `i`에 대해 모델은 `y[i] = x[i+1]`을 예측합니다. 모든 위치가 하나의 포워드 패스에서 동시에 학습됩니다.
- **그레이디언트 클리핑**: `clip_grad_norm_(..., 1.0)`은 폭발하는 그레이디언트를 방지하며, 트랜스포머 학습에 매우 중요합니다.
- **퍼플렉시티(Perplexity)**: `exp(loss)`는 더 해석하기 쉬운 지표입니다 — 퍼플렉시티가 2라면 모델이 평균적으로 2개의 토큰 사이에서 공평한 동전 던지기만큼 불확실하다는 것을 의미합니다.

</details>

## 다음 단계

[HuggingFace 기초](./06_HuggingFace_Basics.md)에서 HuggingFace Transformers 라이브러리를 학습합니다.

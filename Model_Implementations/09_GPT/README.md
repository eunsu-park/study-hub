# 09. GPT

## 개요

GPT (Generative Pre-trained Transformer)는 OpenAI가 개발한 자기회귀(autoregressive) 언어 모델입니다. **왼쪽에서 오른쪽으로** 텍스트를 생성하며, 현대 LLM의 기반이 되었습니다.

---

## 수학적 배경

### 1. Causal Language Modeling

```
목적함수:
L = -Σ log P(x_t | x_<t)

자기회귀 모델:
P(x_1, x_2, ..., x_n) = Π P(x_t | x_1, ..., x_{t-1})

특징:
- 미래 토큰 참조 불가 (causal mask)
- 모든 토큰이 학습 신호
- 텍스트 생성에 자연스러움
```

### 2. Causal Self-Attention

```
표준 Attention:
Attention(Q, K, V) = softmax(QK^T / √d) V

Causal Attention (미래 마스킹):
mask = upper_triangular(-∞)
Attention(Q, K, V) = softmax((QK^T + mask) / √d) V

행렬 시각화:
Q\K  | t1  t2  t3  t4
---------------------
t1   |  ✓   ×   ×   ×
t2   |  ✓   ✓   ×   ×
t3   |  ✓   ✓   ✓   ×
t4   |  ✓   ✓   ✓   ✓
```

### 3. GPT vs BERT

```
BERT (Bidirectional):
- Masked LM: 15% 마스킹
- 양방향 컨텍스트
- 분류/이해 태스크에 강함

GPT (Autoregressive):
- Causal LM: 다음 토큰 예측
- 왼쪽 컨텍스트만
- 생성 태스크에 강함
```

---

## GPT-2 아키텍처

```
GPT-2 Small (117M):
- Hidden size: 768
- Layers: 12
- Attention heads: 12

GPT-2 Medium (345M):
- Hidden size: 1024
- Layers: 24
- Attention heads: 16

GPT-2 Large (774M):
- Hidden size: 1280
- Layers: 36
- Attention heads: 20

GPT-2 XL (1.5B):
- Hidden size: 1600
- Layers: 48
- Attention heads: 25

구조:
Token Embedding + Position Embedding
  ↓
Transformer Decoder × L layers (Pre-LN)
  ↓
Layer Norm
  ↓
LM Head (shared with embedding)
```

---

## 파일 구조

```
09_GPT/
├── README.md
├── pytorch_lowlevel/
│   └── gpt_lowlevel.py         # GPT Decoder 직접 구현
├── paper/
│   └── gpt2_paper.py           # GPT-2 논문 재현
└── exercises/
    ├── 01_text_generation.md   # 텍스트 생성 실습
    └── 02_kv_cache.md          # KV Cache 구현
```

---

## 핵심 개념

### 1. Pre-LN vs Post-LN

```
Post-LN (원본 Transformer):
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm

Pre-LN (GPT-2):
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add

Pre-LN 장점:
- 학습 안정성 향상
- 더 깊은 네트워크 가능
```

### 2. Weight Tying

```
Embedding과 LM Head 가중치 공유:

E = Embedding matrix (vocab_size × hidden_size)
LM_head = E.T (또는 공유)

장점:
- 파라미터 절약
- 일관된 표현 학습
```

### 3. 생성 전략

```
Greedy: argmax(P(x_t | x_<t))
- 결정적, 반복 문제

Sampling: x_t ~ P(x_t | x_<t)
- 다양성, 품질 저하 가능

Top-K: 상위 K개에서 샘플링
- 품질과 다양성 균형

Top-P (Nucleus): 누적 확률 P까지만
- 동적 후보 크기

Temperature: softmax(logits / T)
- T < 1: 더 결정적
- T > 1: 더 다양
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Causal Attention 직접 구현
- Pre-LN 구조
- 텍스트 생성 함수

### Level 3: Paper Implementation (paper/)
- GPT-2 정확한 사양
- WebText 스타일 학습
- 다양한 생성 전략

### Level 4: Code Analysis (별도 문서)
- HuggingFace GPT2 분석
- nanoGPT 코드 분석

---

## 학습 체크리스트

- [ ] Causal mask 구현
- [ ] Pre-LN 구조 이해
- [ ] Weight tying 이해
- [ ] 다양한 생성 전략 구현
- [ ] KV Cache 최적화
- [ ] GPT vs BERT 차이점

---

## 참고 자료

- Radford et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [../LLM_and_NLP/03_BERT_GPT_Architecture.md](../LLM_and_NLP/03_BERT_GPT_Architecture.md)

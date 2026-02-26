# 09. GPT

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 자기회귀(Autoregressive) 인과적 언어 모델링 목적함수를 설명하고, 아키텍처, 사전 훈련, 활용 사례 측면에서 GPT가 BERT와 어떻게 다른지 서술합니다.
2. 인과적 자기 어텐션(Causal Self-Attention) 메커니즘을 설명하고, 인과적 마스크(Causal Mask)가 학습 중 자기회귀 특성을 어떻게 강제하는지 설명합니다.
3. 인과적 마스크 다중 헤드 어텐션(Causal Masked Multi-Head Attention), 층 정규화(Layer Normalization), 위치 임베딩(Positional Embedding)을 포함하여 PyTorch에서 GPT 디코더 아키텍처를 처음부터 구현합니다.
4. 자기회귀 샘플링 전략(그리디(Greedy), Top-k, 뉴클리어스 샘플링(Nucleus Sampling))을 사용하여 텍스트를 생성하고, 생성 품질과 다양성 간의 트레이드오프를 설명합니다.
5. GPT 모델을 다운스트림 생성 작업(예: 텍스트 요약, 대화)에 파인튜닝하고 출력 품질을 평가합니다.
6. GPT-1에서 GPT-2, GPT-3으로의 진화를 추적하고, 성능 향상을 이끈 스케일링 결정(모델 크기, 데이터, 연산)을 식별합니다.

---

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

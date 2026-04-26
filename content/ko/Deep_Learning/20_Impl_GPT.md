# 20. GPT

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 자기회귀(Autoregressive) 인과적 언어 모델링 목적함수를 설명하고, 아키텍처, 사전 훈련, 활용 사례 측면에서 GPT가 BERT와 어떻게 다른지 서술합니다.
2. 인과적 자기 어텐션(Causal Self-Attention) 메커니즘을 설명하고, 인과적 마스크(Causal Mask)가 학습 중 자기회귀 특성을 어떻게 강제하는지 설명합니다.
3. 인과적 마스크 다중 헤드 어텐션(Causal Masked Multi-Head Attention), 층 정규화(Layer Normalization), 위치 임베딩(Positional Embedding)을 포함하여 PyTorch에서 GPT 디코더 아키텍처를 처음부터 구현합니다.
4. 자기회귀 샘플링 전략(그리디(Greedy), Top-k, 뉴클리어스 샘플링(Nucleus Sampling))을 사용하여 텍스트를 생성하고, 생성 품질과 다양성 간의 트레이드오프를 설명합니다.
5. GPT 모델을 다운스트림 생성 작업(예: 텍스트 요약, 대화)에 파인튜닝하고 출력 품질을 평가합니다.
6. GPT-1에서 GPT-2, GPT-3으로의 진화를 추적하고, 성능 향상을 이끈 스케일링 결정(모델 크기, 데이터, 연산)을 식별합니다.

---

## 이론과 원리

GPT (Radford et al. 2018-2020)는 좌-우 언어 모델로 학습된 decoder-only Transformer입니다: 모든 이전 토큰이 주어지면 다음 토큰 예측. 아키텍처적으로 원래 Transformer의 decoder와 비교해 새로운 것이 거의 없습니다; 돌파구는 *규모*와 작업 특이적 아키텍처 없이 규모만으로 일반 능력을 만든다는 발견입니다. 이 섹션은 자기 회귀 LM 학습의 수학, 생성을 위한 샘플링 전략, 그리고 단일 모델에 수십억 달러를 쓰는 것을 정당화하는 스케일링 법칙을 제공합니다.

이 섹션에서 다루는 내용:

- **A.** 자기 회귀 언어 모델 인수분해
- **B.** 자연스러운 손실로서의 cross-entropy; 메트릭으로서의 perplexity
- **C.** 샘플링 전략: greedy, top-k, top-p, temperature
- **D.** 스케일링 법칙과 compute-optimal 학습 (Chinchilla)

### A. 자기 회귀 인수분해

언어 모델은 시퀀스 `x_1, ..., x_T`에 확률을 할당합니다. 확률의 체인 룰에 의해:

```
p(x_1, ..., x_T) = prod_{t=1}^{T} p(x_t | x_1, ..., x_{t-1})
```

이 인수분해는 *정확*합니다 — 근사 없음. 신경 언어 모델은 각 조건부를 `p(x_t | x_{<t}) = softmax(f_\theta(x_{<t}))`로 매개변수화하며, 여기서 `f_\theta`는 어휘에 대한 로짓(logit)을 만드는 Transformer입니다. 학습은 로그 우도를 최대화:

```
log p(x_1, ..., x_T) = sum_t log p(x_t | x_{<t})
```

음수화하면 손실; 시퀀스에 대해 평균하면 토큰별 손실. Teacher forcing + causal mask 덕분에 모든 `T`개 조건부 확률이 한 번의 순전파로 계산됩니다.

### B. Cross-Entropy와 Perplexity

토큰별 손실은 정확히 예측 분포와 원-핫 진실 사이의 cross-entropy입니다:

```
L_t = -log p(x_t | x_{<t}) = -log softmax(logits)[x_t]
```

이것이 `nn.CrossEntropyLoss`가 계산하는 것입니다. 두 편리한 해석:

- **정보 이론적**: `L_t`는 모델 분포가 주어졌을 때 `x_t`를 인코딩하는 데 필요한 비트(`log_2` 사용) 또는 nat(`log_e` 사용) 수. 낮을수록 = 더 나은 압축 = 더 나은 모델.
- **Perplexity**: `PPL = exp(mean L)`. 모델이 토큰당 고려하는 "동등 가능 선택의 유효 수"에 대략. 5만 어휘면 무작위는 `PPL = 50000`; 잘 학습된 GPT-2는 Wikipedia에서 ~20.

Cross-entropy와 perplexity는 LM 자체의 유일한 의미 있는 평가 메트릭입니다; 다운스트림 작업 성능은 별도(그리고 종종 더 흥미로운) 질문입니다.

### C. 샘플링 전략

학습 후, 모델은 `p(x_t | x_{<t})`를 만듭니다 — 하지만 실제 다음 토큰을 어떻게 선택할까요? 네 가지 전략:

- **Greedy / argmax**: `x_t = argmax_v p(v)`. 결정적, 종종 지루하거나 반복적(모델이 루프에 갇힘).
- **전체 분포에서 샘플링**: `x_t ~ p`. 다양하지만 때로 일관성 없음(저확률 토큰이 뽑힘).
- **Top-k**: `k`개 최고 확률 토큰만 유지, 재정규화, 샘플링. 최악 사례 나쁜 샘플 제한.
- **Top-p (nucleus)**: 누적 확률 `>= p`(예: 0.9)인 가장 작은 토큰 집합 유지, 재정규화, 샘플링. 분포 형태에 적응 — 날카로운 분포는 적은 토큰, 평평한 것은 많이 유지.
- **Temperature**: softmax 전 로짓을 `1/T`로 재스케일. `T < 1`은 날카롭게(argmax에 가까움), `T > 1`은 평탄하게(더 다양), `T = 1`은 자연 분포.

현대 LLM은 일반적으로 생성에 top-p + temperature를 결합하며, `T = 0.7-0.9`가 흔한 실용 기본값.

### D. 스케일링 법칙과 Chinchilla

Kaplan et al. (2020)과 Hoffmann et al. (2022, "Chinchilla")는 LM 손실이 **계산(C), 파라미터(N), 학습 토큰(D)**에 대해 예측 가능한 거듭제곱 법칙을 따른다는 것을 보였습니다:

```
L(N, D) ≈ L_inf + A / N^\alpha + B / D^\beta
```

`\alpha, \beta ≈ 0.3-0.4`. 고정 계산 예산 `C ≈ 6 N D`에 대해, 손실 최소화 할당은 **N에 비례하는 D** — 파라미터와 토큰의 대략 동등한 스케일링. GPT-3 (175B 파라미터, 3000억 토큰)은 *과소 학습*되었습니다; Chinchilla (70B 파라미터, 1.4T 토큰)는 2.5배 적은 파라미터로 그것을 능가했습니다.

시사점은 LM 품질이 "더 크면 더 좋다"가 아니라 "더 크고 더 많은 데이터, 균형 있게, 더 좋다"라는 것입니다. 현대 LLM (LLaMA-2, GPT-4)는 Chinchilla 최적 레시피를 훨씬 더 가깝게 따릅니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| AR 인수분해 | Causal mask + 토큰별 cross-entropy 손실 |
| LM 손실 | `F.cross_entropy(logits.view(-1, V), targets.view(-1))` |
| Perplexity | 홀드 아웃 셋에서 `torch.exp(loss)` |
| Top-p 샘플링 | 확률 정렬, 임계값까지 누적합, 재정규화 |

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

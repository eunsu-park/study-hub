# 19. BERT

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. BERT의 핵심 혁신(양방향 컨텍스트, 마스크 언어 모델링(Masked Language Modeling, MLM), 다음 문장 예측(Next Sentence Prediction, NSP))을 설명하고, GPT와 같은 단방향 언어 모델과 대비합니다.
2. MLM 목적함수의 수학적 공식을 서술하고 토큰 마스킹 전략(80/10/10 분할)을 설명합니다.
3. 다중 헤드 자기 어텐션(Multi-Head Self-Attention), 피드포워드 층(Feed-Forward Layer), 위치 임베딩(Positional Embedding)을 포함하여 PyTorch에서 BERT 인코더 아키텍처를 처음부터 구현합니다.
4. 텍스트 분류, 개체명 인식(Named Entity Recognition), 질의응답(Question Answering)과 같은 다운스트림 작업을 위해 사전 훈련된 BERT 모델을 파인튜닝(Fine-tuning)합니다.
5. BERT의 특수 토큰([CLS], [SEP], [MASK])을 해석하고 사전 훈련과 파인튜닝 과정에서 이들이 어떻게 사용되는지 설명합니다.
6. BERT-base와 BERT-large 구성을 비교하고 모델 크기와 작업 성능 간의 트레이드오프를 평가합니다.

---

## 이론과 원리

BERT (Devlin et al. 2018)는 *encoder-only* Transformer가 올바른 자기 지도(self-supervised) 목적으로 학습되면 파인튜닝을 통해 다운스트림 NLP 작업을 지배할 수 있다는 증명이었습니다. 아키텍처는 단지 Transformer encoder입니다; 새로운 아이디어는 사전학습 목적(MLM과 NSP), 특수 토큰(CLS, SEP), 그리고 하나의 사전학습된 모델을 많은 작업에 적응시키는 레시피입니다.

이 섹션에서 다루는 내용:

- **A.** 양방향 self-attention vs 좌-우
- **B.** 노이즈 제거 오토인코딩으로서의 Masked Language Modeling (MLM)
- **C.** Next Sentence Prediction (NSP)과 후속 모델이 그것을 떨어뜨린 이유
- **D.** 사전학습-후-파인튜닝 패러다임

### A. 양방향 Self-Attention

표준 좌-우 LM (GPT)에서, 토큰 `t`는 `<= t`에만 attention합니다. BERT의 encoder에서, 각 토큰은 입력의 *모든* 토큰에 attention합니다 — 과거와 미래. 이 양방향성은 많은 작업에 필수적입니다: 문장 감성 분류는 전체 문장 보기를 요구; 명명 개체 인식은 종종 나중 컨텍스트를 요구합니다.

도전: language-modeling 같은 목적으로 양방향 모델을 어떻게 학습할까요? 다음 토큰을 그냥 예측할 수 없습니다, 모델이 이미 그것을 볼 수 있으니까요. 답은 마스킹입니다.

### B. Masked Language Modeling

BERT는 입력 토큰의 15%를 무작위로 대체하고 모델에게 원본 재구성을 요청:

- 선택된 토큰의 80%: `[MASK]`로 대체
- 10%: 무작위 토큰으로 대체
- 10%: 변경되지 않음

모델은 손상된 시퀀스를 보고 모든 위치에서 예측을 생성; 손실은 마스킹된 위치에서만 계산. 이는 변장한 **노이즈 제거 오토인코더(denoising autoencoder)**입니다: 네트워크가 양방향 컨텍스트를 사용해 누락된 단어를 채우도록 학습.

왜 무작위 토큰 / 변경 없음 분할일까요? 파인튜닝 시 어떤 토큰도 `[MASK]`가 아닐 것입니다(그 토큰은 사전학습에 고유). 손상된 토큰의 100%가 `[MASK]`였다면 모델은 비-`[MASK]` 위치를 무시하도록 학습할 것입니다. 10/10 분할은 모델이 마스킹된 것뿐 아니라 *모든* 토큰 표현을 사용하도록 강제합니다.

MLM 목적은 다음 토큰 예측보다 훨씬 어렵습니다(예제당 위치의 15%에서만 그래디언트를 얻음), 하지만 매우 잘 전이되는 표현을 만듭니다.

### C. Next Sentence Prediction과 그 쇠퇴

BERT는 또한 **NSP**로 사전학습되었습니다: 두 문장 A와 B가 주어지면, B가 소스 코퍼스에서 실제로 A를 따르는지, 또는 무작위 문장인지 예측. CLS 토큰의 최종 은닉 상태가 이진 분류기의 입력으로 사용되었습니다.

후속 연구(RoBERTa, ALBERT)는 NSP가 가치를 거의 더하지 않고 때로는 해친다는 것을 발견했습니다: 무작위 문장 부정이 너무 쉬운데, 무작위 문장이 보통 다른 주제에서 오기 때문에 모델이 문장 관계 분류기가 아닌 주제 분류기를 학습합니다. 현대 BERT 파생은 NSP를 떨어뜨리고 MLM에만 의존하며, 종종 더 긴 학습과 더 큰 배치로.

### D. 사전학습-후-파인튜닝

BERT가 확립한 레시피:

1. **사전학습** 거대 비레이블 코퍼스(BooksCorpus + Wikipedia, 33억 단어)에서 MLM (+ NSP)으로. 비용: ~64 TPU로 4일.
2. **파인튜닝** 작업 특이적 헤드를 추가하고 작은 학습률(~5e-5)로 전체 모델을 종단간 학습하여 작은 레이블 작업 데이터셋에. 비용: 분에서 시간.

이는 비싼 부분(표현 학습)을 저렴한 부분(작업 적응)에서 분리했고, 개별 연구실이 많은 작업에서 최첨단을 밀어내는 것을 경제적으로 만들었습니다. 모든 현대 Foundation Model — BERT, GPT, T5, LLaMA — 이 레시피에서 유래하며, 크기와 사전학습 목적만 다릅니다.

수학은 변하지 않았습니다; 레버리지는 전적으로 사전학습 규모에서 옵니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| 양방향 attention | Encoder에 causal mask 없음 |
| MLM 손상 | 80/10/10 토큰 손상 로직 |
| CLS 토큰 | 입력 앞에 추가된 학습된 `[CLS]` 임베딩 |
| 파인튜닝 헤드 | CLS 은닉 위 `nn.Linear(d_model, num_classes)` |

---


## 개요

BERT (Bidirectional Encoder Representations from Transformers)는 Google이 2018년에 발표한 모델로, NLP 분야에 혁명을 일으켰습니다. **양방향 컨텍스트**를 사용하여 단어의 의미를 이해합니다.

---

## 수학적 배경

### 1. Masked Language Modeling (MLM)

```
목적함수:
L_MLM = -Σ log P(x_mask | x_context)

마스킹 전략 (15% 토큰):
- 80%: [MASK] 토큰으로 대체
- 10%: 랜덤 토큰으로 대체
- 10%: 원본 유지

예시:
입력: "The [MASK] sat on the mat"
목표: "cat" 예측
```

### 2. Next Sentence Prediction (NSP)

```
50% IsNext:    Sentence A → Sentence B (실제 연속)
50% NotNext:   Sentence A → Random B

입력: [CLS] Sentence A [SEP] Sentence B [SEP]
출력: IsNext / NotNext 분류
```

### 3. BERT Embedding

```
Token Embedding:     단어의 의미
Segment Embedding:   문장 A/B 구분
Position Embedding:  위치 정보

Input = Token_Emb + Segment_Emb + Position_Emb
```

---

## BERT 아키텍처

```
BERT-Base:
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Parameters: 110M

BERT-Large:
- Hidden size: 1024
- Layers: 24
- Attention heads: 16
- Parameters: 340M

구조:
[CLS] Token1 Token2 ... [SEP] Token1 ... [SEP]
  ↓
Embedding Layer (Token + Segment + Position)
  ↓
Transformer Encoder × L layers
  ↓
[CLS]: 분류 / Token: 토큰 예측
```

---

## 파일 구조

```
08_BERT/
├── README.md
├── pytorch_lowlevel/
│   └── bert_lowlevel.py        # BERT Encoder 직접 구현
├── paper/
│   └── bert_paper.py           # 논문 재현
└── exercises/
    ├── 01_mlm_training.md      # MLM 학습 실습
    └── 02_finetuning.md        # 분류 fine-tuning
```

---

## 핵심 개념

### 1. Bidirectional Context

```
GPT (Left-to-Right):
"The cat sat" → 왼쪽만 참조하여 다음 예측

BERT (Bidirectional):
"The [MASK] sat on the mat" → 양쪽 모두 참조하여 [MASK] 예측

장점: 더 풍부한 문맥 이해
단점: 텍스트 생성에 부적합
```

### 2. Pre-training & Fine-tuning

```
Phase 1: Pre-training (대규모 corpus)
- MLM + NSP 태스크
- Wikipedia + BookCorpus (3.3B 토큰)

Phase 2: Fine-tuning (downstream task)
- [CLS] 토큰으로 분류
- 또는 모든 토큰 출력으로 시퀀스 라벨링
```

### 3. 입력 형식

```
단일 문장: [CLS] tokens [SEP]
문장 쌍:   [CLS] tokens_A [SEP] tokens_B [SEP]

Segment IDs:
[CLS] A A A [SEP] B B B [SEP]
  0   0 0 0   0   1 1 1   1
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- F.linear, F.layer_norm 사용
- nn.TransformerEncoder 미사용
- Embedding 수동 구현

### Level 3: Paper Implementation (paper/)
- 논문의 정확한 사양 재현
- MLM + NSP pre-training
- 분류 fine-tuning

### Level 4: Code Analysis (별도 문서)
- HuggingFace transformers 코드 분석
- BertModel, BertForSequenceClassification

---

## 학습 체크리스트

- [ ] MLM 마스킹 전략 이해
- [ ] NSP 태스크 이해
- [ ] Token/Segment/Position Embedding 이해
- [ ] [CLS] 토큰의 역할
- [ ] Fine-tuning 방법 (분류, NER, QA)
- [ ] BERT vs GPT 차이점

---

## 참고 자료

- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- [HuggingFace BERT](https://huggingface.co/docs/transformers/model_doc/bert)
- [../LLM_and_NLP/03_BERT_GPT_Architecture.md](../LLM_and_NLP/03_BERT_GPT_Architecture.md)

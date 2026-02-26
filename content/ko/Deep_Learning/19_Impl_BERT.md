# 08. BERT

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. BERT의 핵심 혁신(양방향 컨텍스트, 마스크 언어 모델링(Masked Language Modeling, MLM), 다음 문장 예측(Next Sentence Prediction, NSP))을 설명하고, GPT와 같은 단방향 언어 모델과 대비합니다.
2. MLM 목적함수의 수학적 공식을 서술하고 토큰 마스킹 전략(80/10/10 분할)을 설명합니다.
3. 다중 헤드 자기 어텐션(Multi-Head Self-Attention), 피드포워드 층(Feed-Forward Layer), 위치 임베딩(Positional Embedding)을 포함하여 PyTorch에서 BERT 인코더 아키텍처를 처음부터 구현합니다.
4. 텍스트 분류, 개체명 인식(Named Entity Recognition), 질의응답(Question Answering)과 같은 다운스트림 작업을 위해 사전 훈련된 BERT 모델을 파인튜닝(Fine-tuning)합니다.
5. BERT의 특수 토큰([CLS], [SEP], [MASK])을 해석하고 사전 훈련과 파인튜닝 과정에서 이들이 어떻게 사용되는지 설명합니다.
6. BERT-base와 BERT-large 구성을 비교하고 모델 크기와 작업 성능 간의 트레이드오프를 평가합니다.

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

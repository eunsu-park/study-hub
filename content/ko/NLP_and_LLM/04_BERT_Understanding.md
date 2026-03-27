# 04. BERT 이해

## 학습 목표

- BERT 아키텍처 이해
- 사전학습 목표 (MLM, NSP)
- 입력 표현
- 다양한 BERT 변형

---

## 1. BERT 개요

### Bidirectional Encoder Representations from Transformers

```
BERT = Transformer 인코더 스택

특징:
- 양방향 문맥 이해
- 사전학습 + 파인튜닝 패러다임
- 다양한 NLP 태스크에 범용 적용
```

### 모델 크기

| 모델 | 레이어 | d_model | 헤드 | 파라미터 |
|------|-------|---------|------|---------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

---

## 2. 입력 표현

### 세 가지 임베딩의 합

```
입력: [CLS] I love NLP [SEP] It is fun [SEP]

Token Embedding:    [E_CLS, E_I, E_love, E_NLP, E_SEP, E_It, E_is, E_fun, E_SEP]
Segment Embedding:  [E_A,   E_A, E_A,    E_A,   E_A,   E_B,  E_B,  E_B,   E_B  ]
Position Embedding: [E_0,   E_1, E_2,    E_3,   E_4,   E_5,  E_6,  E_7,   E_8  ]
                    ─────────────────────────────────────────────────────────────
                    = 최종 입력 임베딩 (합)
```

### 특수 토큰

| 토큰 | 역할 |
|------|------|
| [CLS] | 분류 태스크용 집계 토큰 |
| [SEP] | 문장 구분자 |
| [PAD] | 패딩 |
| [MASK] | MLM에서 마스킹된 토큰 |
| [UNK] | 미등록 단어 |

### 입력 구현

```python
import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=768, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)

        # 위치 인덱스
        position_ids = torch.arange(seq_len, device=input_ids.device)

        # 임베딩 합
        embeddings = (
            self.token_embedding(input_ids) +
            self.position_embedding(position_ids) +
            self.segment_embedding(segment_ids)
        )

        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
```

---

## 3. 사전학습 목표

### Masked Language Model (MLM)

```
15%의 토큰을 선택:
- 80%: [MASK]로 교체
- 10%: 랜덤 토큰으로 교체
- 10%: 그대로 유지

예시:
입력: "The cat sat on the mat"
     → "The [MASK] sat on the mat"
목표: [MASK] → "cat" 예측
```

```python
import random

def create_mlm_data(tokens, vocab, mask_prob=0.15):
    """MLM 학습 데이터 생성"""
    labels = [-100] * len(tokens)  # -100은 손실 계산에서 무시

    for i, token in enumerate(tokens):
        if random.random() < mask_prob:
            labels[i] = vocab[token]  # 원래 토큰 ID

            rand = random.random()
            if rand < 0.8:
                tokens[i] = '[MASK]'
            elif rand < 0.9:
                tokens[i] = random.choice(list(vocab.keys()))
            # else: 그대로 유지

    return tokens, labels
```

### Next Sentence Prediction (NSP)

```
입력: [CLS] 문장A [SEP] 문장B [SEP]
목표: 문장B가 문장A의 실제 다음 문장인지 이진 분류

예시:
긍정 (IsNext):
    A: "The man went to the store"
    B: "He bought a gallon of milk"

부정 (NotNext):
    A: "The man went to the store"
    B: "Penguins are flightless birds"
```

```python
class BERTPreTrainingHeads(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # MLM 헤드
        self.mlm = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        # NSP 헤드
        self.nsp = nn.Linear(d_model, 2)

    def forward(self, sequence_output, cls_output):
        mlm_scores = self.mlm(sequence_output)  # (batch, seq, vocab)
        nsp_scores = self.nsp(cls_output)       # (batch, 2)
        return mlm_scores, nsp_scores
```

---

## 4. BERT 전체 구조

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()

        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        # 임베딩
        x = self.embedding(input_ids, segment_ids)

        # 패딩 마스크 변환
        if attention_mask is not None:
            # (batch, seq) → (batch, seq) with True for padding
            attention_mask = (attention_mask == 0)

        # 인코더
        output = self.encoder(x, src_key_padding_mask=attention_mask)

        return output  # (batch, seq, d_model)


class BERTForPreTraining(nn.Module):
    def __init__(self, vocab_size, d_model=768, **kwargs):
        super().__init__()
        self.bert = BERT(vocab_size, d_model, **kwargs)
        self.heads = BERTPreTrainingHeads(d_model, vocab_size)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        sequence_output = self.bert(input_ids, segment_ids, attention_mask)
        cls_output = sequence_output[:, 0]  # [CLS] 토큰

        mlm_scores, nsp_scores = self.heads(sequence_output, cls_output)
        return mlm_scores, nsp_scores
```

---

## 5. 파인튜닝 패턴

### 문장 분류 (Single Sentence)

```python
class BERTForSequenceClassification(nn.Module):
    def __init__(self, bert, num_classes, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.embedding.token_embedding.embedding_dim,
                                    num_classes)

    def forward(self, input_ids, segment_ids, attention_mask):
        output = self.bert(input_ids, segment_ids, attention_mask)
        cls_output = output[:, 0]  # [CLS]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)
```

### 토큰 분류 (NER)

```python
class BERTForTokenClassification(nn.Module):
    def __init__(self, bert, num_labels, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.embedding.token_embedding.embedding_dim,
                                    num_labels)

    def forward(self, input_ids, segment_ids, attention_mask):
        output = self.bert(input_ids, segment_ids, attention_mask)
        output = self.dropout(output)
        return self.classifier(output)  # (batch, seq, num_labels)
```

### 질의응답 (QA)

```python
class BERTForQuestionAnswering(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        hidden_size = bert.embedding.token_embedding.embedding_dim
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start, end

    def forward(self, input_ids, segment_ids, attention_mask):
        output = self.bert(input_ids, segment_ids, attention_mask)
        logits = self.qa_outputs(output)  # (batch, seq, 2)

        start_logits = logits[:, :, 0]  # (batch, seq)
        end_logits = logits[:, :, 1]

        return start_logits, end_logits
```

---

## 6. BERT 변형 모델

### RoBERTa

```
변경점:
- NSP 제거 (MLM만 사용)
- 동적 마스킹 (매 에포크 다른 마스킹)
- 더 큰 배치, 더 긴 학습
- Byte-Level BPE 토크나이저

결과: BERT보다 성능 향상
```

### ALBERT

```
변경점:
- 임베딩 분해 (V×E, E×H → V×E, E<<H)
- 레이어 파라미터 공유
- NSP → SOP (Sentence Order Prediction)

결과: 파라미터 대폭 감소, 유사 성능
```

### DistilBERT

```
변경점:
- 지식 증류 (Teacher: BERT → Student: 작은 모델)
- 6 레이어 (BERT의 절반)

결과: 40% 작음, 60% 빠름, 97% 성능 유지
```

### Comparison

| 모델 | 레이어 | 파라미터 | 속도 | 특징 |
|------|-------|---------|------|------|
| BERT-base | 12 | 110M | 1x | 기준 |
| RoBERTa | 12 | 125M | 1x | 최적화된 학습 |
| ALBERT-base | 12 | 12M | 1x | 파라미터 공유 |
| DistilBERT | 6 | 66M | 2x | 지식 증류 |

---

## 7. HuggingFace BERT 사용

### 기본 사용

```python
from transformers import BertTokenizer, BertModel

# 토크나이저와 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 인코딩
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# 순전파
outputs = model(**inputs)

# 출력
last_hidden_state = outputs.last_hidden_state  # (1, seq, 768)
pooler_output = outputs.pooler_output          # (1, 768) - [CLS] 변환
```

### 분류 모델

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

inputs = tokenizer("I love this movie!", return_tensors='pt')
outputs = model(**inputs)
logits = outputs.logits  # (1, 2)
```

### Attention 시각화

```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

inputs = tokenizer("The cat sat on the mat", return_tensors='pt')
outputs = model(**inputs)

# Attention weights: (num_layers, batch, heads, seq, seq)
attentions = outputs.attentions

# 첫 번째 레이어, 첫 번째 헤드
attn = attentions[0][0, 0].detach().numpy()
```

---

## 8. BERT 입력 포맷

### Single Sentence

```
[CLS] sentence [SEP]
segment_ids: [0, 0, 0, ..., 0]
```

### Sentence Pair

```
[CLS] sentence A [SEP] sentence B [SEP]
segment_ids: [0, 0, ..., 0, 1, 1, ..., 1]
```

### HuggingFace에서 Pair 처리

```python
# 두 문장 입력
text_a = "How old are you?"
text_b = "I am 25 years old."

inputs = tokenizer(
    text_a, text_b,
    padding='max_length',
    max_length=32,
    truncation=True,
    return_tensors='pt'
)

print(inputs['token_type_ids'])  # segment_ids
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...]
```

---

## 정리

### 핵심 개념

1. **양방향 인코더**: 전체 문맥을 양방향으로 이해
2. **MLM**: 마스킹된 토큰 예측으로 문맥 학습
3. **NSP**: 문장 관계 이해 (RoBERTa에서 제거)
4. **[CLS] 토큰**: 문장 수준 표현
5. **Segment Embedding**: 문장 구분

### 핵심 코드

```python
# HuggingFace BERT
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 인코딩
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]
```

---

## 연습 문제

### 연습 문제 1: MLM 데이터 준비

`create_mlm_data` 함수를 보다 견고하게 구현하고, 마스킹 확률 15%로 `"The cat sat on the mat"` 문장이 처리될 때 어떤 일이 발생하는지 추적하세요. 선택된 각 토큰에 대한 세 가지 가능한 대체 방법을 보여주고, 선택된 토큰의 10%를 변경하지 않고 유지하는 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

```python
import random

def create_mlm_data(tokens, vocab, mask_prob=0.15, mask_token='[MASK]'):
    """
    BERT의 마스킹 전략에 따른 MLM(Masked Language Model) 학습 데이터 생성.

    선택된 15%의 토큰에 대해:
    - 80%: [MASK]로 교체
    - 10%: 어휘에서 랜덤 토큰으로 교체
    - 10%: 변경하지 않고 유지 (하지만 여전히 예측)
    """
    tokens = tokens.copy()
    labels = [-100] * len(tokens)  # -100 = 교차 엔트로피 손실에서 무시

    for i, token in enumerate(tokens):
        # 특수 토큰 건너뜀
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue

        if random.random() < mask_prob:
            labels[i] = vocab.get(token, vocab.get('[UNK]', 0))

            rand = random.random()
            if rand < 0.8:
                tokens[i] = mask_token                        # 80%: [MASK]
            elif rand < 0.9:
                tokens[i] = random.choice(list(vocab.keys())) # 10%: 랜덤 단어
            # else: 원래 토큰 유지                            # 10%: 변경 없음

    return tokens, labels

# "The cat sat on the mat"에 대한 예시 추적
vocab = {'[CLS]': 101, '[SEP]': 102, '[PAD]': 0, '[MASK]': 103,
         'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5}

tokens = ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '[SEP]']

# mask_prob=0.15가 "cat"(인덱스 2)과 "mat"(인덱스 6)을 선택한다고 가정
# "cat"에 대해: 80% 확률 → ['[CLS]', 'the', '[MASK]', 'sat', 'on', 'the', 'mat', '[SEP]']
#              labels[2] = 2 ("cat"의 원래 ID)
# "mat"에 대해: 10% 확률 → ['[CLS]', 'the', '[MASK]', 'sat', 'on', 'the', 'dog', '[SEP]']
#              labels[6] = 5 ("mat"의 원래 ID)

print("입력 토큰:", ['[CLS]', 'the', '[MASK]', 'sat', 'on', 'the', 'dog', '[SEP]'])
print("레이블:  ", [-100, -100, 2, -100, -100, -100, 5, -100])
```

**선택된 토큰의 10%를 변경하지 않고 유지하는 이유**:

선택된 모든 토큰이 항상 `[MASK]`로 교체된다면, 모델은 `[MASK]` 위치에서만 예측하도록 학습될 것입니다 — 파인튜닝 중에는 `[MASK]` 토큰이 없기 때문에 실제 위치에서의 실제 토큰을 표현할 필요가 없어집니다. 10%를 그대로 유지하면서도 예측하도록 함으로써, 모델은 마스크 위치뿐만 아니라 모든 토큰에 대해 유용한 표현을 유지해야 합니다. 이는 다운스트림(downstream) 태스크로 더 잘 전이되는 표현을 생성합니다.

</details>

### 연습 문제 2: BERT 입력 포맷팅

전제(premise) `"The cat is on the mat"`와 가설(hypothesis) `"The cat is sleeping"`을 사용하는 자연어 추론(NLI, Natural Language Inference) 태스크에 대해 완전한 BERT 입력 포맷을 구성하세요. `input_ids`, `segment_ids`(token_type_ids), `attention_mask` 배열을 보여주세요. HuggingFace를 사용하여 수동 구성을 검증하세요.

<details>
<summary>정답 보기</summary>

```python
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

premise = "The cat is on the mat"
hypothesis = "The cat is sleeping"

# HuggingFace는 자동으로 [CLS] 전제 [SEP] 가설 [SEP] 형식을 처리함
inputs = tokenizer(
    premise, hypothesis,
    padding='max_length',
    max_length=32,
    truncation=True,
    return_tensors='pt'
)

# 구조를 시각화하기 위해 디코딩
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("토큰:", tokens)
# ['[CLS]', 'the', 'cat', 'is', 'on', 'the', 'mat', '[SEP]',
#  'the', 'cat', 'is', 'sleeping', '[SEP]',
#  '[PAD]', '[PAD]', ...]

print("\nSegment IDs (token_type_ids):")
print(inputs['token_type_ids'][0])
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, ...])
#         CLS    전제          SEP   가설        SEP  PAD...

print("\n어텐션 마스크:")
print(inputs['attention_mask'][0])
# tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...])
#         ^--- 실제 토큰 = 1 --------^  ^- 패딩 = 0 -^

# 수동 구성 (HuggingFace가 내부적으로 수행하는 것):
premise_tokens = ['[CLS]'] + tokenizer.tokenize(premise) + ['[SEP]']
hypothesis_tokens = tokenizer.tokenize(hypothesis) + ['[SEP]']
all_tokens = premise_tokens + hypothesis_tokens

segment_ids = [0] * len(premise_tokens) + [1] * len(hypothesis_tokens)
attention_mask = [1] * len(all_tokens)

print(f"\n수동 구성: {len(all_tokens)} 토큰, {len(segment_ids)} 세그먼트 ID")
# 세그먼트 0 = 전제, 세그먼트 1 = 가설
```

**핵심 인사이트**: 위치 0의 `[CLS]` 토큰은 집계된 시퀀스 표현 역할을 합니다. NLI의 경우, `[CLS]` 출력 위에 선형 분류기가 `entailment`(함의), `contradiction`(모순), 또는 `neutral`(중립)을 예측하도록 학습됩니다.

</details>

### 연습 문제 3: BERT vs RoBERTa 차이점

RoBERTa는 BERT의 학습 절차에 여러 핵심적인 변경을 가했습니다. 아래 나열된 각 변경 사항에 대해 왜 그렇게 했는지, 그리고 어떤 개선을 제공하는지 설명하세요:
1. NSP(Next Sentence Prediction) 제거
2. 정적 마스킹 대신 동적 마스킹
3. 더 많은 데이터로 더 큰 배치 크기 사용

<details>
<summary>정답 보기</summary>

**1. NSP 제거**

원래 BERT 논문은 NSP가 NLI(Natural Language Inference)나 QA 같은 문장 쌍 태스크에 도움이 된다고 주장했습니다. 그러나 RoBERTa 논문의 절제 연구(ablation study)는 NSP가 실제로 여러 벤치마크에서 성능을 저하시킨다는 것을 보여주었습니다.

NSP가 문제였던 이유:
- NSP 학습의 "NotNext" 예제는 서로 다른 문서에서 가져오므로, 모델은 일관된 문장 추론이 아닌 주제 기반으로 구별하는 방법을 학습할 수 있습니다.
- NSP는 짧은 시퀀스(두 반길이 문장)를 강제하여 장거리 문맥 모델링의 이점을 줄입니다.
- 전체 길이 단일 문서에서 MLM만으로도 더 강력한 양방향 표현을 제공합니다.

**2. 동적 마스킹 vs 정적 마스킹**

원래 BERT에서는 마스킹이 데이터 전처리 중에 한 번 적용되었습니다(정적) — 모든 에포크에서 동일한 토큰이 항상 마스킹되었습니다.

```python
# 정적 마스킹 (원래 BERT)
# 전처리 중:
masked_tokens = apply_masking(tokens, seed=42)  # 고정된 마스크
# 에포크 1, 2, 3, ...에서 동일한 마스크 사용

# 동적 마스킹 (RoBERTa)
# 각 포워드 패스 / 에포크 중:
masked_tokens = apply_masking(tokens, seed=epoch_seed)  # 매번 새로운 마스크
```

동적 마스킹은 각 학습 예제가 에포크마다 다른 마스킹 위치를 보게 하여, 더 다양한 학습 신호를 제공하고 추가적인 데이터 증강(data augmentation) 역할을 합니다. 모델은 특정 위치뿐만 아니라 어떤 문맥에서도 어떤 토큰이든 예측하는 방법을 학습합니다.

**3. 더 큰 배치와 더 많은 데이터**

```
원래 BERT: batch_size=256, 100만 스텝, 16GB 데이터
RoBERTa:   batch_size=8192, 50만 스텝, 160GB 데이터
```

- **더 큰 배치**: 업데이트당 더 나은 그레이디언트(gradient) 추정, 많은 GPU를 병렬로 사용할 때 실제 시간 기준으로 더 빠른 수렴.
- **더 많은 데이터**: 더 다양한 언어 패턴이 일반화를 개선합니다. BERT는 BooksCorpus + Wikipedia에서 학습했지만; RoBERTa는 CommonCrawl News, OpenWebText, Stories를 추가했습니다.
- **더 긴 학습**: 더 적은 스텝에도 불구하고, 더 큰 배치 크기는 스텝당 훨씬 더 많은 토큰이 처리됨을 의미합니다.

통합 결과: RoBERTa는 아키텍처 변경 없이 GLUE 벤치마크에서 BERT보다 지속적으로 2-4% 더 좋은 성능을 보입니다 — 학습 절차가 아키텍처만큼 중요함을 증명합니다.

</details>

### 연습 문제 4: 감성 분석을 위한 BERT 파인튜닝

HuggingFace를 사용하여 이진 감성 분류를 위해 `bert-base-uncased`를 파인튜닝(fine-tuning)하는 완전한 코드를 작성하세요. 모델 설정, 학습 단계를 포함하고, 각 레이어의 학습률(learning rate)이 어떻게 되는지(그리고 BERT 레이어에 분류기 헤드보다 낮은 학습률을 사용하는 것이 좋은 관행인 이유)를 설명하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# 설정
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # 긍정 / 부정
)

# 샘플 데이터
texts = [
    "I love this movie, it's fantastic!",
    "This film was a complete waste of time.",
    "Amazing performances and great story.",
    "Boring and predictable plot.",
]
labels = [1, 0, 1, 0]  # 1=긍정, 0=부정

# 토큰화
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
labels_tensor = torch.tensor(labels)

# 차별적 학습률:
# - 사전학습된 BERT 레이어에 낮은 LR (기존 지식 보존)
# - 새로운 분류기 헤드에 높은 LR (처음부터 학습)
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},        # BERT 레이어: 낮은 LR
    {'params': model.classifier.parameters(), 'lr': 1e-3},  # 새 헤드: 높은 LR
])

# 학습 단계
model.train()
optimizer.zero_grad()

outputs = model(**inputs, labels=labels_tensor)
loss = outputs.loss
logits = outputs.logits

loss.backward()
optimizer.step()

print(f"손실: {loss.item():.4f}")
print(f"예측: {torch.argmax(logits, dim=1).tolist()}")

# 추론
model.eval()
with torch.no_grad():
    test_input = tokenizer(
        "This movie exceeded all my expectations!",
        return_tensors='pt'
    )
    output = model(**test_input)
    pred = torch.argmax(output.logits, dim=1).item()
    print(f"감성: {'긍정' if pred == 1 else '부정'}")
```

**차별적 학습률(discriminative learning rate)이 작동하는 이유**:

BERT의 하위 레이어는 일반적인 언어 지식(구문, 형태론)을 인코딩하고, 상위 레이어는 태스크 관련 의미론을 인코딩합니다. 이러한 표현은 이미 수십억 개의 토큰에서 잘 학습되어 있습니다. 작은 학습률(`2e-5`)을 사용하면 이 지식을 덮어쓰는 대신 특정 태스크에 맞게 정제하는 작은 조정을 할 수 있습니다.

분류기 헤드는 랜덤으로 초기화되어 처음부터 학습해야 하므로, 더 큰 학습률(`1e-3`)이 빠른 수렴을 돕습니다.

균일한 큰 LR을 사용했을 경우 **치명적 망각(catastrophic forgetting)**이 발생할 수 있습니다 — 사전학습된 가중치가 크게 변하여 BERT를 유용하게 만드는 일반적인 언어 이해를 잃을 수 있습니다.

</details>

## 다음 단계

[GPT 이해](./05_GPT_Understanding.md)에서 GPT 모델과 자기회귀 언어 모델을 학습합니다.

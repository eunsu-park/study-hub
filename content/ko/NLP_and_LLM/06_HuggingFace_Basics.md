# 06. HuggingFace 기초

## 학습 목표

- Transformers 라이브러리 이해
- Pipeline API 사용
- 토크나이저와 모델 로드
- 다양한 태스크 수행

---

## 이론과 원리

HuggingFace Transformers 라이브러리는 본질적으로 수백 가지의 서로 다른 모델 아키텍처에 대한 **표준화된 추상 계층**입니다. 편리한 `AutoModel.from_pretrained("bert-base")` 호출 뒤에는 세 가지 독립적 관심사를 분리한 신중한 설계가 있습니다 — **토크나이즈**(텍스트 → 텐서), **아키텍처**(모델을 구현하는 Python 클래스), **가중치**(Hub에 저장된 학습된 파라미터). 이 조각들이 어떻게 맞물리는지, 그리고 상호 운용을 가능하게 하는 관습들을 이해하는 것이 — 연구 논문에서 작동하는 모델까지 세 줄로 가는 비결입니다.

이 섹션은 다음을 다룹니다:

- **(A) Auto-클래스 추상화** — `AutoModel`, `AutoTokenizer`, `AutoConfig`가 모델의 `config.json`을 보고 올바른 구체 클래스로 라우팅하는 방식.
- **(B) 토크나이저 내부** — fast(Rust) vs slow(Python), 인코딩 파이프라인, `BatchEncoding`이 실제로 담는 것.
- **(C) 모델 로딩과 가중치 초기화** — `state_dict`, weight tying, 부분 로드, `device_map`과 양자화 옵션.
- **(D) Pipeline API** — `pipeline("sentiment-analysis")`이 전처리·모델 추론·후처리를 어떻게 구성하는가.
- **(E) Datasets / Trainer 파이프라인** — 스트리밍과 캐싱을 위한 `datasets`, 표준 학습 루프를 위한 `Trainer`.
- **(F) Hub** — 모델 카드, 버전 관리, `from_pretrained` 해석 알고리즘.

### A. Auto-클래스 추상화

`AutoTokenizer.from_pretrained("bert-base-uncased")`는 대략 다음을 합니다:

```
1. 모델 식별자 → HF Hub URL 해석.
2. config.json 다운로드 — "model_type": "bert"와 아키텍처 하이퍼파라미터 포함.
3. model_type을 레지스트리에서 조회 (bert → BertTokenizer 또는 BertTokenizerFast).
4. 다운로드된 config와 vocab 파일로 그 구체 클래스 인스턴스화.
```

`AutoModel`, `AutoConfig`, `AutoFeatureExtractor`, `AutoProcessor`도 같은 알고리즘. 이점은 **사용자 코드를 모델 아키텍처와 분리**하는 것 — `bert-base-uncased`에서 `roberta-base` 또는 `microsoft/deberta-v3-base`로 바꾸는 데 문자열 하나만 수정하면 됩니다. Auto-클래스는 모달리티당 ~20개의 서로 다른 구체 클래스를 숨깁니다.

작업 특화 Auto-클래스(`AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`)는 기본 인코더에 적절한 작업 헤드를 둘러쌉니다. 작업 특화 체크포인트가 없으면 헤드의 무작위 초기화 가중치를 처리합니다.

### B. 토크나이저 내부

**B.1 Fast vs slow.** Python 토크나이저(slow)는 BPE/WordPiece를 순수 Python으로 구현; fast 토크나이저(`tokenizers` 라이브러리)는 같은 알고리즘을 Rust로 병렬 구현. 100만 문장 코퍼스의 BERT-base에서 fast는 ~50-100배 빠릅니다 — 토크나이즈가 종종 학습 병목이라 중요합니다.

**B.2 인코딩 파이프라인.** `tokenizer(text)` 호출은 다음을 실행:

```
text → 정규화 → 사전 토큰화 → 토큰화 (BPE/WordPiece/Unigram) → 후처리 ([CLS], [SEP] 추가) → BatchEncoding 반환
```

각 단계는 설정 가능. 정규화는 Unicode(NFC/NFD), 악센트 제거, 소문자화 처리. 사전 토큰화는 공백·구두점에서 분리하지만 원본 문자 오프셋 기록을 유지. 후처리는 특수 토큰을 추가하고 어텐션 마스크를 만듭니다.

**B.3 BatchEncoding 내용.**

```python
out = tokenizer(["Hello world", "Hi"], padding=True, return_tensors="pt")
# out["input_ids"]:        토큰 ID 텐서, shape (B, L)
# out["attention_mask"]:   실제 토큰은 1, 패딩은 0, shape (B, L)
# out["token_type_ids"]:   문장 쌍 모델용 세그먼트 ID (0/1)
# out.offset_mapping:      원본 문자열로의 문자 오프셋 (fast 토크나이저만)
```

`offset_mapping`은 NER 같은 토큰 수준 작업에 결정적 — 예측된 토큰 라벨을 원본 텍스트의 문자 범위로 매핑할 수 있게 합니다. 오프셋 추적 없이는 정확한 매핑이 불가능합니다.

### C. 모델 로딩과 가중치 초기화

**C.1 `from_pretrained`가 하는 일.**

```
1. config.json + 토크나이저 파일 + safetensors/pytorch_model.bin 다운로드.
2. config로부터 아키텍처 인스턴스화 (무작위 가중치).
3. 다운로드한 state_dict를 이름으로 매핑하여 로드.
4. missing keys (아키텍처가 기대했지만 체크포인트에 없는 가중치)와
   unexpected keys (체크포인트에 있지만 이 아키텍처가 사용하지 않는 가중치) 보고.
```

Missing keys는 자동 초기화됩니다 — `BertForSequenceClassification`의 분류 헤드 가중치는 사전학습된 `bert-base-uncased` 체크포인트에 그런 헤드가 없으므로 무작위입니다. 어떤 가중치가 사전학습되지 않았는지 알 수 있도록 라이브러리가 경고를 로그합니다.

**C.2 Weight tying.** LM에서 입력 임베딩 `E ∈ ℝ^{V × d}`와 출력 사영 `W ∈ ℝ^{d × V}`는 보통 묶여 있습니다(`W = E^T`). 임베딩/출력의 파라미터 수를 절반으로 줄이고, 두 사영이 동시에 두 역할을 만족해야 한다는 정규화 효과도 줍니다. HF 라이브러리가 로딩 중 자동으로 처리합니다.

**C.3 `device_map`과 양자화.** 현대 LLM은 단일 GPU에 풀 정밀도로 들어가지 않습니다. `from_pretrained(..., device_map="auto", load_in_4bit=True)`는 다음을 트리거:

- meta 디바이스에 아키텍처 인스턴스화(할당 없음).
- 가중치별 로딩: 각 텐서를 디스크에서 로드, 선택적으로 NF4/INT8로 양자화, 메모리 예산에 따라 GPU/CPU/디스크에 배치.
- 결과: 각 파라미터가 자동 결정된 디바이스에 있는, 끝까지 실행 가능한 모델.

이는 상당한 추상화입니다 — 없으면 `accelerate` 글루 코드를 수백 줄 작성해야 합니다.

### D. Pipeline API

```python
nlp = pipeline("sentiment-analysis")
nlp("I love this!")  # → [{"label": "POSITIVE", "score": 0.99}]
```

파이프라인은 다음을 구성하는 얇은 래퍼:

```
1. 토크나이저 (모델의 기본값에서).
2. 모델 (no_grad로 forward).
3. 후처리: 분류는 argmax, 생성은 디코딩, QA는 span 추출 등.
```

파이프라인은 편의 래퍼입니다 — 기본 토크나이즈, 기본 디코딩 전략, Python 수준 배칭에 묶입니다. 프로덕션에는 직접 토크나이저 + 모델 호출로 내려가고, 프로토타이핑이나 일회성 추론에는 파이프라인이 완벽합니다.

### E. Datasets와 Trainer

**E.1 `datasets`.** 로컬 파일, Hub, 커스텀 로더 위의 통합 API로, NLP에 중요한 세 속성:

- **메모리 매핑 저장**(Apache Arrow): RAM에 안 들어가도 100 GB 데이터셋을 질의 가능.
- **스트리밍**: `load_dataset(..., streaming=True)`이 요청 시 다운로드하는 이터레이터 반환 — C4(700 GB+) 같은 데이터셋에 필수.
- **캐싱이 있는 `map()`**: `dataset.map(tokenize_fn)`이 한 번 실행되고 토크나이즈 결과를 디스크에 캐시. 재실행 시 캐시를 사용.

**E.2 `Trainer`.** 표준 학습 루프 구현:

```
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % eval_steps == 0:
            evaluate(...)
```

내장: 그래디언트 누적, 혼합 정밀도 학습, 분산 학습, 학습률 스케줄링, 체크포인팅, W&B / TensorBoard 통합. 연구에서는 `Trainer`를 직접 작성한 PyTorch로 대체해도 좋지만, 어떤 표준 파인튜닝 작업이든 `Trainer`가 수백 줄을 절약합니다.

### F. Hub

`huggingface.co/<org>/<model>`의 각 모델은 `config.json`, 토크나이저 파일, 가중치, `README.md`(모델 카드)를 담은 Git 저장소입니다. 모델 버전 관리는 Git 리비전을 사용하므로 `from_pretrained(..., revision="v1.0")`이 특정 커밋에 고정합니다.

Hub는 현대 NLP의 의존성 메커니즘입니다 — 논문이 코드를 공개할 때 표준 패턴은 학습된 체크포인트도 Hub에 공개하여 모든 독자가 `from_pretrained`로 결과를 재현할 수 있게 하는 것입니다. 이것이 NLP 분야의 재현성이 ML 전반보다 훨씬 나은 구조적 이유입니다.

### 이론에서 아래 함수들로

- §1 (생태계) — §A Auto-클래스 계층과 §F Hub 모델 개관.
- §2 (Pipeline API) — 일반 작업을 위한 §D 래퍼.
- §3 (토크나이저) — §B `AutoTokenizer`의 직접 사용, offset_mapping 예시.
- §4 (모델 로딩) — §C `from_pretrained` 메커니즘, `AutoModel`과 작업 특화 헤드.
- §5 (Datasets 라이브러리) — §E.1의 load/map/cache 패턴.
- §6 (Trainer API) — §E.2의 표준 파인튜닝 루프.
- §7 (저장/로드) — `save_pretrained` / `push_to_hub` (§F).
- §8 (감정 분류) — §B, §C, §E를 묶는 엔드 투 엔드 프로젝트.

---

## 1. HuggingFace 생태계

### 주요 구성요소

```
HuggingFace
├── Transformers   # 모델 라이브러리
├── Datasets       # 데이터셋
├── Tokenizers     # 토크나이저
├── Hub            # 모델/데이터 저장소
├── Accelerate     # 분산 학습
└── Evaluate       # 평가 메트릭
```

### 설치

```bash
pip install transformers datasets tokenizers accelerate evaluate
```

---

## 2. Pipeline API

### 가장 간단한 사용법

```python
from transformers import pipeline

# 감성 분석
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 배치 처리
results = classifier([
    "I love this movie!",
    "This is terrible."
])
```

### 지원 태스크

| 태스크 | Pipeline 이름 | 설명 |
|--------|--------------|------|
| 감성 분석 | sentiment-analysis | 긍정/부정 분류 |
| 텍스트 분류 | text-classification | 일반 분류 |
| NER | ner | 개체명 인식 |
| QA | question-answering | 질의응답 |
| 요약 | summarization | 텍스트 요약 |
| 번역 | translation | 언어 번역 |
| 텍스트 생성 | text-generation | 문장 생성 |
| Fill-Mask | fill-mask | 마스크 예측 |
| Zero-shot | zero-shot-classification | 레이블 없는 분류 |

### 다양한 Pipeline 예제

```python
# 질의응답
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France."
)
# {'answer': 'Paris', 'score': 0.99, 'start': 0, 'end': 5}

# 요약
summarizer = pipeline("summarization")
text = "Very long article text here..."
summary = summarizer(text, max_length=50, min_length=10)

# 번역
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]

# 텍스트 생성
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# NER
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is John and I work at Google in New York")
# [{'entity_group': 'PER', 'word': 'John', ...},
#  {'entity_group': 'ORG', 'word': 'Google', ...},
#  {'entity_group': 'LOC', 'word': 'New York', ...}]

# Zero-shot 분류
classifier = pipeline("zero-shot-classification")
result = classifier(
    "I want to go to the beach",
    candidate_labels=["travel", "cooking", "technology"]
)
# {'labels': ['travel', 'cooking', 'technology'], 'scores': [0.95, 0.03, 0.02]}
```

### 특정 모델 지정

```python
# 한국어 모델
classifier = pipeline(
    "sentiment-analysis",
    model="beomi/kcbert-base"
)

# 다국어 모델
qa = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2"
)
```

---

## 3. 토크나이저

### AutoTokenizer

```python
from transformers import AutoTokenizer

# 자동으로 적합한 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 인코딩
text = "Hello, how are you?"
encoded = tokenizer(text)
print(encoded)
# {'input_ids': [101, 7592, ...], 'attention_mask': [1, 1, ...], ...}

# 텐서로 반환
encoded = tokenizer(text, return_tensors='pt')
```

### 주요 파라미터

```python
encoded = tokenizer(
    text,
    padding=True,              # 패딩 추가
    truncation=True,           # 최대 길이 자르기
    max_length=128,            # 최대 길이
    return_tensors='pt',       # PyTorch 텐서
    return_attention_mask=True,
    return_token_type_ids=True
)
```

### 배치 인코딩

```python
texts = ["Hello world", "How are you?", "I'm fine"]

# 동적 패딩
encoded = tokenizer(
    texts,
    padding=True,     # 가장 긴 시퀀스에 맞춤
    truncation=True,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)  # (3, max_len)
```

### 디코딩

```python
# 디코딩
decoded = tokenizer.decode(encoded['input_ids'][0])
print(decoded)  # "[CLS] hello world [SEP]"

# 특수 토큰 제거
decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print(decoded)  # "hello world"
```

### 토큰 확인

```python
# 토큰 목록
tokens = tokenizer.tokenize("Hello, how are you?")
print(tokens)  # ['hello', ',', 'how', 'are', 'you', '?']

# 토큰 → ID
ids = tokenizer.convert_tokens_to_ids(tokens)

# ID → 토큰
tokens = tokenizer.convert_ids_to_tokens(ids)
```

---

## 4. 모델 로드

### AutoModel

```python
from transformers import AutoModel, AutoModelForSequenceClassification

# 기본 모델 (출력: 은닉 상태)
model = AutoModel.from_pretrained("bert-base-uncased")

# 분류 모델 (출력: 로짓)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

### 태스크별 AutoModel

```python
from transformers import (
    AutoModelForSequenceClassification,  # 문장 분류
    AutoModelForTokenClassification,      # 토큰 분류 (NER)
    AutoModelForQuestionAnswering,        # QA
    AutoModelForCausalLM,                 # GPT 스타일 생성
    AutoModelForSeq2SeqLM,                # 인코더-디코더 (번역, 요약)
    AutoModelForMaskedLM                  # BERT 스타일 MLM
)
```

### 추론

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 인코딩
inputs = tokenizer("I love this movie!", return_tensors="pt")

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 예측
predictions = torch.softmax(logits, dim=-1)
predicted_class = predictions.argmax().item()
print(f"Class: {predicted_class}, Confidence: {predictions[0][predicted_class]:.4f}")
```

---

## 5. Datasets 라이브러리

### 데이터셋 로드

```python
from datasets import load_dataset

# HuggingFace Hub에서 로드
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# 분할 지정
train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test[:1000]")  # 처음 1000개

# 샘플 확인
print(train_data[0])
# {'text': '...', 'label': 1}
```

### 데이터 전처리

```python
def preprocess(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

# map 적용
tokenized_dataset = dataset.map(preprocess, batched=True)

# 불필요한 컬럼 제거
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# PyTorch 포맷 설정
tokenized_dataset.set_format('torch')
```

### DataLoader 생성

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    tokenized_dataset['train'],
    batch_size=16,
    shuffle=True
)

for batch in train_loader:
    print(batch['input_ids'].shape)  # (16, 256)
    break
```

---

## 6. Trainer API

### 기본 학습

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 데이터
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
)

# 학습
trainer.train()

# 평가
results = trainer.evaluate()
print(results)
```

### 커스텀 메트릭

```python
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics
)
```

---

## 7. 모델 저장/로드

### 로컬 저장

```python
# 저장
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 로드
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

### Hub에 업로드

```python
# 로그인
from huggingface_hub import login
login(token="your_token")

# 업로드
model.push_to_hub("my-username/my-model")
tokenizer.push_to_hub("my-username/my-model")

# 또는 Trainer로
trainer.push_to_hub("my-model")
```

---

## 8. 실전 예제: 감성 분류

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# 1. 데이터 로드
dataset = load_dataset("imdb")

# 2. 토크나이저
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 4. 메트릭
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. 학습 설정
args = TrainingArguments(
    output_dir="./imdb_classifier",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Mixed Precision
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
)

# 7. 학습
trainer.train()

# 8. 추론
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return "Positive" if probs[0][1] > 0.5 else "Negative", probs[0][1].item()

print(predict("This movie was amazing!"))
# ('Positive', 0.9876)
```

---

## 정리

### 핵심 클래스

| 클래스 | 용도 |
|--------|------|
| pipeline | 빠른 추론 |
| AutoTokenizer | 토크나이저 자동 로드 |
| AutoModel* | 모델 자동 로드 |
| Trainer | 학습 루프 자동화 |
| TrainingArguments | 학습 설정 |

### 핵심 코드

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 빠른 추론
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# 커스텀 추론
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs)
```

---

## 연습 문제

### 연습 문제 1: Pipeline 태스크 탐색

HuggingFace `pipeline` API를 사용하여 세 가지 다른 NLP 태스크를 실행하세요: 사람 이름과 조직이 포함된 문장에서 NER(개체명 인식), 사용자 정의 레이블을 사용한 뉴스 헤드라인에서 제로샷(zero-shot) 분류, 질의응답. 각 결과에 대해 모델이 출력하는 것과 점수(score)가 무엇을 나타내는지 설명하세요.

<details>
<summary>정답 보기</summary>

```python
from transformers import pipeline

# 태스크 1: 개체명 인식 (NER, Named Entity Recognition)
ner = pipeline("ner", grouped_entities=True)
sentence = "Elon Musk founded SpaceX in California and Tesla in Texas."
ner_result = ner(sentence)
print("NER 결과:")
for entity in ner_result:
    print(f"  '{entity['word']}' → {entity['entity_group']} (score: {entity['score']:.3f})")
# 'Elon Musk' → PER (score: 0.998)  — 사람(Person)
# 'SpaceX'    → ORG (score: 0.995)  — 조직(Organization)
# 'California'→ LOC (score: 0.992)  — 위치(Location)
# 'Tesla'     → ORG (score: 0.989)  — 조직(Organization)
# 'Texas'     → LOC (score: 0.991)  — 위치(Location)
# 점수: 개체 레이블 할당에 대한 신뢰도

# 태스크 2: 제로샷 분류 (Zero-shot Classification)
zero_shot = pipeline("zero-shot-classification")
headline = "Scientists discover new exoplanet with potential for liquid water"
result = zero_shot(
    headline,
    candidate_labels=["astronomy", "biology", "technology", "sports", "politics"]
)
print("\n제로샷 분류:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.3f}")
# astronomy: 0.812
# biology: 0.124
# technology: 0.047
# 점수: 텍스트가 각 레이블에 속할 확률
# (합계 ~1.0, 파인튜닝 불필요!)

# 태스크 3: 질의응답 (Question Answering)
qa = pipeline("question-answering")
context = """
The HuggingFace Transformers library was created in 2018 by Thomas Wolf and Lysandre Debut.
It provides thousands of pretrained models for natural language processing tasks.
The library supports PyTorch, TensorFlow, and JAX frameworks.
"""
questions = [
    "Who created the HuggingFace Transformers library?",
    "What frameworks does it support?",
]
print("\n질의응답:")
for q in questions:
    result = qa(question=q, context=context)
    print(f"  Q: {q}")
    print(f"  A: '{result['answer']}' (score: {result['score']:.3f})")
    print(f"     문자 범위: [{result['start']}, {result['end']}]")
# A: 'Thomas Wolf and Lysandre Debut' (score: 0.97)
# 점수: 이 텍스트 스팬이 정답임에 대한 신뢰도
# start/end: 컨텍스트 문자열에서의 문자 위치
```

**태스크별 점수 해석**:
- **NER**: 개체별로 레이블(PER, ORG, LOC)이 올바른지에 대한 신뢰도. 1.0에 가까운 값은 높은 신뢰도를 나타냅니다.
- **제로샷**: 후보 레이블에 대한 소프트맥스(softmax) 확률 분포. 모델이 학습 중에 이 특정 레이블을 본 적이 없어도 — 자연어 함의(entailment)를 사용하여 순위를 매깁니다.
- **QA**: 추출된 스팬이 정답임에 대한 확률. 낮은 점수(<0.5)는 컨텍스트에 답변이 없을 수 있음을 시사합니다.

</details>

### 연습 문제 2: Trainer API와 커스텀 데이터셋

`Trainer` API는 특정 형식의 데이터셋을 요구합니다. 단순한 (텍스트, 레이블) 튜플 목록을 `Trainer`와 함께 사용할 수 있는 HuggingFace `Dataset` 객체로 변환하는 함수를 작성하세요. 그런 다음 학습 평가에 커스텀 메트릭(F1 점수)을 추가하는 방법을 보여주세요.

<details>
<summary>정답 보기</summary>

```python
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import evaluate
import numpy as np

# 샘플 데이터
train_data = [
    ("I love this product!", 1),
    ("Terrible quality, don't buy.", 0),
    ("Amazing experience, highly recommend!", 1),
    ("Worst purchase I've ever made.", 0),
    ("Five stars, very satisfied!", 1),
    ("Broke after one week, total waste.", 0),
]
test_data = [
    ("Great value for money.", 1),
    ("Not what I expected, disappointed.", 0),
]

def create_dataset(data, tokenizer, max_length=64):
    """(텍스트, 레이블) 튜플을 HuggingFace Dataset으로 변환."""
    texts, labels = zip(*data)

    # 모든 텍스트를 한 번에 토큰화
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )

    # Dataset에 필요한 딕셔너리 형식 구성
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': list(labels),  # Trainer는 'label'이 아닌 'labels' 키를 기대함
    }

    return Dataset.from_dict(dataset_dict)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

train_dataset = create_dataset(train_data, tokenizer)
test_dataset = create_dataset(test_data, tokenizer)

print("데이터셋 구조:", train_dataset)
print("샘플:", train_dataset[0])

# 커스텀 메트릭: 정확도 + F1 점수
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='binary')

    return {
        'accuracy': acc['accuracy'],
        'f1': f1['f1']
    }

# 모델 및 학습 설정
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)

args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy='epoch',
    save_strategy='no',
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# trainer.train()  # 실제로 학습하려면 실행
```

**`Trainer`에 대한 핵심 형식 요건**:
- 데이터셋은 `label`이 아닌 `labels` 키를 포함해야 합니다.
- 입력 텐서는 PyTorch 호환 가능해야 합니다 (`map` 기반 데이터셋에는 `set_format('torch')` 사용)
- `compute_metrics`는 이름 있는 튜플(named tuple) `EvalPrediction`으로 `(logits, labels)`를 받습니다.

</details>

### 연습 문제 3: 태스크에 맞는 모델 선택

아래 각 NLP 태스크에 대해 올바른 `AutoModel` 클래스를 명시하고, 왜 그 특정 클래스가 적합한지 설명하세요:

1. 의미론적 유사도를 위한 문장 임베딩 추출
2. 개체명 인식 (NER)
3. 기계 번역 (영어 → 프랑스어)
4. 문장의 빈칸 채우기 (마스킹된 토큰 예측)
5. 개방형 텍스트 생성

<details>
<summary>정답 보기</summary>

```python
from transformers import (
    AutoModel,                           # 태스크 1: 기본 모델
    AutoModelForTokenClassification,     # 태스크 2: NER
    AutoModelForSeq2SeqLM,               # 태스크 3: 번역
    AutoModelForMaskedLM,                # 태스크 4: Fill-mask
    AutoModelForCausalLM,                # 태스크 5: 생성
)

# 태스크 1: 의미론적 유사도를 위한 문장 임베딩
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# 이유: 임베딩을 계산하기 위한 원시 은닉 상태가 필요함 (last_hidden_state의 평균 풀링)
# 태스크별 헤드 불필요
tokenizer_1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inputs = tokenizer_1("Hello world", return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1)  # 평균 풀링 → (1, 384)

# 태스크 2: 개체명 인식 (NER)
ner_model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
# 이유: NER은 각 토큰에 대한 예측이 필요함 ([CLS]만이 아닌)
# TokenClassification은 토큰당 선형 헤드를 추가: (batch, seq, num_labels)

# 태스크 3: 기계 번역
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr"
)
# 이유: 번역은 인코더(원본 이해) + 디코더(대상 생성)가 필요함
# Seq2SeqLM은 인코더 출력과 디코더 간의 크로스 어텐션(cross-attention)을 처리함

# 태스크 4: Fill-mask (MLM 예측)
mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# 이유: MaskedLM은 [MASK] 위치에서 전체 어휘에 대한 예측 헤드를 추가함
# 출력: (batch, seq, vocab_size) — 각 위치에서 토큰 예측

from transformers import AutoTokenizer
import torch
tokenizer_4 = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "The capital of France is [MASK]."
inputs = tokenizer_4(text, return_tensors="pt")
with torch.no_grad():
    logits = mlm_model(**inputs).logits
mask_idx = (inputs.input_ids == tokenizer_4.mask_token_id).nonzero()[0][1]
top5 = logits[0, mask_idx].topk(5)
predictions = [tokenizer_4.decode([idx]) for idx in top5.indices]
print("상위 5개 예측:", predictions)  # ['paris', 'lyon', ...]

# 태스크 5: 텍스트 생성
gpt = AutoModelForCausalLM.from_pretrained("gpt2")
# 이유: 인과 마스크를 가진 CausalLM(디코더 전용), 다음 토큰 예측으로 학습됨
# generate() 메서드는 다양한 디코딩 전략을 지원함
```

**올바른 모델 클래스가 중요한 이유**:

| 클래스 | 출력 헤드 | 학습 목표 |
|--------|----------|----------|
| `AutoModel` | 없음 (원시 은닉 상태) | — |
| `AutoModelForTokenClassification` | 토큰당 선형 | 토큰당 교차 엔트로피 |
| `AutoModelForSeq2SeqLM` | 크로스 어텐션이 있는 디코더 | Seq-to-seq 교차 엔트로피 |
| `AutoModelForMaskedLM` | 마스크 위치에서 어휘에 대한 선형 | MLM 교차 엔트로피 |
| `AutoModelForCausalLM` | 모든 위치에서 어휘에 대한 선형 | 다음 토큰 교차 엔트로피 |

잘못된 클래스를 사용하면 잘못된 형태를 출력하거나 태스크별 헤드에 대한 올바른 사전학습 가중치를 로드하지 못합니다.

</details>

## 다음 단계

[파인튜닝](./07_Fine_Tuning.md)에서 다양한 태스크에 대한 파인튜닝 기법을 학습합니다.

# 07. 파인튜닝

## 학습 목표

- 파인튜닝 전략 이해
- 다양한 태스크 파인튜닝
- 효율적인 파인튜닝 기법 (LoRA, QLoRA)
- 실전 파인튜닝 파이프라인

---

## 1. 파인튜닝 개요

### 이론: 전이학습이 지배적인 이유

10K 문장 감정 데이터셋에서 Transformer를 처음부터 학습하면 파국적으로 과소적합됩니다 — 모델은 100M+ 파라미터, 데이터셋은 10⁴개 예시. 무작위 초기화의 유효 파라미터 수가 10⁴개 예시가 제약할 수 있는 것을 훨씬 초과합니다.

사전학습은 비율을 뒤집습니다 — 100M 파라미터를 10¹¹+ 토큰(BERT) 또는 10¹²+(LLaMA)으로 제약합니다. 사전학습 가중치는 이미 일반 언어 패턴을 부호화합니다. 파인튜닝은 묻습니다 — "거의 모든 것에 대해 거의 옳은 이 출발점에서, 이 작업을 향해 살짝 밀어 달라". 유효 가설 공간이 극적으로 줄어듭니다 — 사전학습 가중치 근처의 모델만 도달 가능 — 그리고 그 편향이 정확히 데이터가 부족한 파인튜닝에 필요한 것입니다.

경험적으로: 파인튜닝된 BERT는 대부분의 NLP 작업에서 처음부터 학습한 LSTM을 10+ F1 점수 차로 이기며, 100배 적은 라벨 데이터를 사용합니다.

### 이론: 갱신 스펙트럼: 헤드만 → 특징 → 풀 → PEFT

사전학습 가중치를 얼마나 적극적으로 갱신할지에 대한 연속체가 있습니다:

| 전략 | 갱신 파라미터 | 비용 | 사용 시점 |
|------|--------------|------|----------|
| 헤드만 (linear probe) | 헤드만 | 가장 작음 | 매우 적은 데이터, 인코더가 이미 클래스를 분리함을 증명 |
| 마지막 몇 층 | 상위 `k`층 + 헤드 | 작음 | 적은 데이터, 인코더가 거의 작동 |
| 풀 파인튜닝 | 모든 파라미터 | 큼 | 중간 데이터, 타깃 작업이 사전학습과 다름 |
| PEFT (LoRA 등) | 주입된 어댑터만 (~0.1-1%) | 작음 | 큰 모델, 풀 FT 품질을 적은 비용으로 |

경험칙: **작업이 사전학습과 다를수록 더 많은 파라미터를 갱신해야 합니다.** 감정 분류(일반 영어 LM에 가까움)는 헤드만으로 잘 됩니다. 희귀 엔티티의 의료 NER(일반 영어와 멀음)은 풀 파인튜닝이 필요합니다.

### 전이학습 패러다임

```
사전학습 (Pre-training)
    │  대규모 텍스트로 일반적인 언어 이해 학습
    ▼
파인튜닝 (Fine-tuning)
    │  특정 태스크 데이터로 모델 조정
    ▼
태스크 수행
```

### 파인튜닝 전략

| 전략 | 설명 | 사용 시점 |
|------|------|----------|
| Full Fine-tuning | 전체 파라미터 업데이트 | 충분한 데이터, 컴퓨팅 |
| Feature Extraction | 분류기만 학습 | 적은 데이터 |
| LoRA | 저랭크 어댑터 | 효율적인 학습 |
| Prompt Tuning | 프롬프트만 학습 | 매우 적은 데이터 |

---

## 2. 텍스트 분류 파인튜닝

### 이론: 작업별 손실 함수

인코더는 동일하고, 헤드와 손실이 변합니다:

**E.1 시퀀스 분류.** `[CLS]`에 선형 → `C` 클래스에 대한 softmax; cross-entropy `−Σ y log p`.

**E.2 토큰 분류 (NER).** *모든* 토큰 위치에 선형 → 태그 집합에 대한 softmax; 실제(non-pad, non-special) 토큰들에 대한 cross-entropy 합. 같은 단어의 서브워드들은 보통 같은 라벨을 공유하며, 첫 서브워드에서만 손실을 계산하는 관행이 있습니다.

**E.3 추출형 QA (SQuAD 형식).** 모든 토큰에 두 선형 헤드: 하나는 이 토큰이 답 span의 *시작*일 확률, 다른 하나는 *끝*. 손실은 두 cross-entropy의 합. 추론 시 예측 span은 길이 제약 하에서 `argmax_{i ≤ j}  p_start(i) · p_end(j)`.

**E.4 Seq2seq (T5, BART).** 인코더-디코더; 손실은 디코더의 예측 토큰을 타깃 시퀀스에 대해 cross-entropy(teacher forcing). 자기회귀 LM 손실과 같지만 인코딩된 소스에 조건부.

**E.5 Causal LM 파인튜닝 (instruction tuning).** 모델은 GPT 형식; 손실은 응답 토큰에 대해서만 cross-entropy(프롬프트 토큰은 마스킹하여 사용자 입력 예측에 용량 낭비를 피함).

### 기본 파이프라인

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# 데이터 로드
dataset = load_dataset("imdb")

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

tokenized = dataset.map(tokenize, batched=True)

# 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 학습 설정
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
)

trainer.train()
```

### 다중 레이블 분류

```python
from transformers import AutoModelForSequenceClassification
import torch

# 다중 레이블용 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)

# 손실 함수 자동으로 BCEWithLogitsLoss 사용

# 레이블 형식: [1, 0, 1, 0, 1] (다중 레이블)
```

---

## 3. 토큰 분류 (NER) 파인튜닝

### NER 데이터 형식

```python
from datasets import load_dataset

# CoNLL-2003 NER 데이터셋
dataset = load_dataset("conll2003")

# 샘플
print(dataset['train'][0])
# {'tokens': ['EU', 'rejects', 'German', 'call', ...],
#  'ner_tags': [3, 0, 7, 0, ...]}

# 레이블
label_names = dataset['train'].features['ner_tags'].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### 토큰 정렬

```python
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True  # 이미 토큰화된 입력
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 특수 토큰
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # 첫 토큰
            else:
                label_ids.append(-100)  # 서브워드 무시
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized['labels'] = labels
    return tokenized
```

### NER 파인튜닝

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_names)
)

# seqeval 메트릭
import evaluate
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # 실제 레이블만 추출
    true_predictions = []
    true_labels = []

    for pred, label in zip(predictions, labels):
        true_preds = []
        true_labs = []
        for p, l in zip(pred, label):
            if l != -100:
                true_preds.append(label_names[p])
                true_labs.append(label_names[l])
        true_predictions.append(true_preds)
        true_labels.append(true_labs)

    return seqeval.compute(predictions=true_predictions, references=true_labels)
```

---

## 4. 질의응답 (QA) 파인튜닝

### SQuAD 데이터

```python
dataset = load_dataset("squad")

print(dataset['train'][0])
# {'id': '...', 'title': 'University_of_Notre_Dame',
#  'context': 'Architecturally, the school has...',
#  'question': 'To whom did the Virgin Mary appear in 1858?',
#  'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}
```

### QA 전처리

```python
def prepare_train_features(examples):
    tokenized = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 토큰 위치 찾기
            token_start = 0
            token_end = 0
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    token_start = idx
                if start < end_char <= end:
                    token_end = idx
                    break

            tokenized["start_positions"].append(token_start)
            tokenized["end_positions"].append(token_end)

    return tokenized
```

### QA 모델

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# 출력: start_logits, end_logits
```

---

## 5. 효율적인 파인튜닝 (PEFT)

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정
lora_config = LoraConfig(
    r=8,                      # 랭크
    lora_alpha=32,            # 스케일링
    target_modules=["query", "value"],  # 적용 모듈
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# 모델에 LoRA 적용
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
```

### Prompt Tuning

```python
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the sentiment: "
)

model = get_peft_model(model, config)
```

---

## 6. 대화형 모델 파인튜닝

### Instruction Tuning 데이터 형식

```python
# Alpaca 형식
{
    "instruction": "Summarize the following text.",
    "input": "Long article text here...",
    "output": "Summary of the article."
}

# ChatML 형식
"""
<|system|>
You are a helpful assistant.
<|user|>
What is the capital of France?
<|assistant|>
The capital of France is Paris.
"""
```

### SFT (Supervised Fine-Tuning)

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
    ),
)

trainer.train()
```

### DPO (Direct Preference Optimization)

```python
from trl import DPOTrainer

# 선호도 데이터
# {'prompt': '...', 'chosen': '...', 'rejected': '...'}

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 기준 모델
    train_dataset=dataset,
    beta=0.1,
    args=TrainingArguments(...),
)

trainer.train()
```

---

## 7. 학습 최적화

### 이론: 파국적 망각

원래 학습률의 평범한 SGD로 사전학습 모델을 새 작업에 계속 학습시키면, 모델은 새 작업에 빠르게 맞춰지면서 — 알고 있던 모든 것을 잊어버립니다. 이것이 **파국적 망각**입니다 — 새 작업의 그래디언트 갱신이 일반 지식을 부호화한 가중치 패턴을 덮어씁니다.

증상: 새 작업의 학습 정확도가 99%에 도달하지만, 보존 LM perplexity나 관련 작업에서 평가하면 성능이 폭락합니다.

발생 원리: 사전학습 모델은 LM 손실의 국소 최솟값에 위치합니다. 파인튜닝 손실은 *다른* 최솟값을 가집니다. 큰 학습률에서 모델은 새 최솟값을 향해 빠르게 걸어가며 LM 최솟값을 뒤에 남깁니다. "일반 지식"은 가중치 공간 내 위치에 암묵적이었으므로, 그 이웃을 떠나면 사라집니다.

완화책:

- **작은 학습률** (§D) — 갱신을 이웃 안에 머물 만큼 작게.
- **적은 에폭** — 멀리 떠다닐 시간이 없게.
- **점진적 unfreezing** (ULMFiT) — 상위 레이어 먼저 파인튜닝, 그 후 하위 레이어를 점진적으로 해제.
- **PEFT 방법** (LoRA) — 기본 가중치를 완전히 동결하고 작은 추가만 학습.
- **Replay** / **다중 작업** — 파인튜닝 작업과 원래 LM 목적함수를 교차.

### 이론: 학습률 스케줄과 워밍업

전형적 BERT 파인튜닝 레시피:

```
optimizer: AdamW(lr=2e-5, weight_decay=0.01)
schedule: 단계의 10% 동안 선형 워밍업, 그 후 0까지 선형 감쇠
epochs: 2-4
batch size: 16-32
```

**왜 `1e-3`이 아닌 `2e-5`인가?** C에서: 큰 갱신은 망각을 일으킵니다. 사전학습 가중치는 원소별 크기가 작으므로(LayerNorm으로 안정화) 작은 상대 갱신만으로도 작업을 학습하기 충분합니다.

**왜 워밍업인가?** AdamW의 적응적 학습률은 그래디언트 2차 모멘트의 이동 추정치를 사용합니다. 0번째 단계에서 이 추정치는 편향 보정된 초기값(0에 가까운 값을 0에 가까운 값으로 나눔)이라 매우 큰 유효 학습률을 만들 수 있습니다. 워밍업은 처음 ~10% 단계 동안 LR을 0에서 정점까지 선형으로 올려, Adam이 풀 갱신을 적용하기 전에 좋은 2차 모멘트 추정치를 만들 시간을 줍니다.

**왜 선형 감쇠인가?** 경험적으로 가장 좋은 단순 스케줄. 모델은 초반에는 적극적 갱신, 후반에는 미세 다듬기가 필요합니다. 코사인 감쇠도 작동합니다.

### 이론: 정규화

10K 예시에 파인튜닝되는 100M 파라미터 모델은 정규화 없이는 과적합합니다.

- **Early stopping**: 보존 집합을 모니터링, 정체하거나 악화되면 중단.
- **Dropout**: 보통 어텐션과 FFN에 `0.1`. 사전학습 체크포인트에 이미 내장; 작은 데이터셋에서 올리면 도움이 될 수 있습니다.
- **Weight decay**: AdamW에서 `0.01`. 0으로부터의 이탈을 패널티 — 가중치를 사전학습 값(자체가 크기 0 근처) 근처에 간접적으로 유지.
- **Label smoothing**: 원-핫 타깃을 `(1−ε)·onehot + ε/C·uniform`로 대체. 과신을 방지, 보정(calibration)을 향상. 일반적으로 `ε = 0.1`.
- **Layer-wise LR decay (ULMFiT)**: 하위 레이어가 상위 레이어보다 작은 LR — 하위 레이어는 더 일반적인 특징을 부호화하고 갱신이 덜 필요합니다.
- **Mixout**: dropout과 비슷하지만 마스킹된 위치가 0이 아닌 사전학습 값으로 되돌아갑니다. 파국적 망각 대처용으로 특별히 설계.

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```

### Mixed Precision

```python
args = TrainingArguments(
    ...,
    fp16=True,  # 또는 bf16=True
)
```

### Gradient Accumulation

```python
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 실효 배치 = 4 * 8 = 32
)
```

### DeepSpeed

```python
args = TrainingArguments(
    ...,
    deepspeed="ds_config.json"
)

# ds_config.json
{
    "fp16": {"enabled": true},
    "zero_optimization": {"stage": 2}
}
```

---

## 8. 전체 파인튜닝 예제

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import evaluate

# 1. 데이터
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 2. 모델 + LoRA
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 학습 설정
args = TrainingArguments(
    output_dir="./lora_imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

# 4. 메트릭
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. 학습
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. 저장
model.save_pretrained("./lora_imdb_final")
```

---

## 정리

### 파인튜닝 선택 가이드

| 상황 | 추천 방법 |
|------|----------|
| 충분한 데이터 + GPU | Full Fine-tuning |
| 제한된 GPU 메모리 | LoRA / QLoRA |
| 매우 적은 데이터 | Prompt Tuning |
| LLM 정렬 | SFT + DPO/RLHF |

### 핵심 코드

```python
# LoRA
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, target_modules=["query", "value"])
model = get_peft_model(model, lora_config)

# Trainer
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
```

---

## 연습 문제

### 연습 문제 1: LoRA 파라미터 수 분석

LoRA가 쿼리 및 값 프로젝션 행렬에 적용된 BERT-base 모델(12 레이어, d_model=768, num_heads=12)에 대해 다음을 계산하세요: (a) 랭크 r=8에서 학습 가능한 파라미터 수, (b) 학습되는 총 파라미터의 비율, (c) 전체 파인튜닝과의 비교.

<details>
<summary>정답 보기</summary>

```python
# BERT-base 아키텍처 파라미터
num_layers = 12
d_model = 768
d_k = d_model // 12  # = 64, 헤드당 차원
num_heads = 12

# 쿼리 및 값 프로젝션 차원
# W_q: (d_model, d_model) = (768, 768)
# W_v: (d_model, d_model) = (768, 768)

# BERT-base의 총 파라미터 (근사값)
d_ff = 3072
vocab_size = 30522

embeddings = vocab_size * d_model + 512 * d_model + 2 * d_model  # 토큰 + 위치 + 세그먼트
per_layer = (4 * d_model**2) + (2 * d_model * d_ff) + (4 * d_model)  # 어텐션 + FFN + 정규화
pooler = d_model * d_model + d_model

total_bert = embeddings + (num_layers * per_layer) + pooler
print(f"BERT-base 총 파라미터: {total_bert:,}")
# ≈ 109,482,240 (110M)

# 랭크 r=8에서 LoRA 파라미터
r = 8
lora_r = r

# W_q와 W_v에 적용된 각 LoRA 레이어에 대해:
# A 행렬: (d_model, r) — d_model → r 매핑
# B 행렬: (r, d_model) — r → d_model 매핑
lora_params_per_matrix = d_model * r + r * d_model  # A + B
lora_targets = 2  # query와 value 모두

# 모든 12개 레이어에 적용
total_lora_params = num_layers * lora_targets * lora_params_per_matrix
print(f"LoRA 파라미터 (r={r}): {total_lora_params:,}")
# = 12 * 2 * (768*8 + 8*768) = 12 * 2 * 12288 = 294,912

percentage = total_lora_params / total_bert * 100
print(f"학습 가능 비율: {percentage:.3f}%")
# ≈ 0.27%

# 비교
print(f"\n전체 파인튜닝: {total_bert:,} 파라미터 (100%)")
print(f"LoRA 파인튜닝: {total_lora_params:,} 파라미터 ({percentage:.2f}%)")
print(f"감소: 학습 가능한 파라미터가 {total_bert / total_lora_params:.0f}배 적음")
# ~371배 적은 파라미터

# PEFT로 검증
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS
)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%
```

**LoRA가 작동하는 이유**:

LoRA는 가중치 업데이트 `ΔW = BA`(`B ∈ R^{d×r}`, `A ∈ R^{r×d}`)가 되도록 저랭크 행렬 `A`와 `B`를 추가합니다. 포워드 패스 중: `h = W₀x + BAx = (W₀ + BA)x`. `A`와 `B`만 업데이트되고 `W₀`는 동결됩니다.

가설은 태스크 적응의 내재적 차원이 `d`보다 훨씬 낮다는 것이므로, 랭크-8 업데이트가 필요한 적응의 대부분을 포착합니다. 이는 많은 벤치마크에서 경험적으로 검증되었습니다.

</details>

### 연습 문제 2: NER을 위한 토큰 정렬

NER(Named Entity Recognition) 파인튜닝의 까다로운 측면 중 하나는 WordPiece 토큰화가 단어를 여러 서브워드 토큰으로 분리할 수 있지만, NER 레이블은 단어 수준에서 할당된다는 것입니다. NER 레이블을 서브워드 토큰에 적절히 정렬하는 함수를 작성하고, 구체적인 예시를 통해 추적하세요.

<details>
<summary>정답 보기</summary>

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 예시: 단어 수준의 NER
words = ['Barack', 'Obama', 'was', 'born', 'in', 'Hawaii']
ner_labels = [1, 2, 0, 0, 0, 5]  # B-PER, I-PER, O, O, O, B-LOC
# 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC

def align_labels_with_tokens(words, labels, tokenizer):
    """
    단어 수준의 NER 레이블을 서브워드 토큰에 정렬.
    규칙:
    - 특수 토큰([CLS], [SEP])은 -100 레이블 (손실에서 무시)
    - 단어의 첫 번째 서브워드는 단어의 레이블을 받음
    - 같은 단어의 이후 서브워드는 -100 (무시)
    """
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=False,
        truncation=True,
    )

    word_ids = tokenized.word_ids()  # 각 토큰 위치를 단어 인덱스에 매핑

    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # 특수 토큰 ([CLS], [SEP], [PAD])
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # 새 단어의 첫 번째 서브워드: 단어의 레이블 사용
            aligned_labels.append(labels[word_id])
        else:
            # 연속 서브워드: 손실에서 무시
            aligned_labels.append(-100)

        previous_word_id = word_id

    return tokenized, aligned_labels, word_ids

tokenized, aligned_labels, word_ids = align_labels_with_tokens(
    words, ner_labels, tokenizer
)

# 정렬 표시
tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

print(f"{'토큰':<15} {'단어 ID':<10} {'레이블':<15} {'레이블명'}")
print("-" * 50)
for token, word_id, label in zip(tokens, word_ids, aligned_labels):
    label_str = label_names[label] if label != -100 else "무시"
    word_id_str = str(word_id) if word_id is not None else "특수"
    print(f"{token:<15} {word_id_str:<10} {str(label):<15} {label_str}")

# 출력:
# 토큰            단어 ID    레이블          레이블명
# --------------------------------------------------
# [CLS]           특수       -100            무시
# barack          0          1               B-PER
# ##ob            0          -100            무시  ← 서브워드
# ##ama           0          -100            무시  ← 서브워드
# was             2          0               O
# born            3          0               O
# in              4          0               O
# hawaii          5          5               B-LOC
# [SEP]           특수       -100            무시
```

**이것이 중요한 이유**: "Barack"의 세 서브워드 ("barack", "##ob", "##ama") 모두에 `B-PER`을 단순히 할당하면, 모델은 `##ama`에 대해 `B-PER`을 예측하려고 할 것입니다. 그러나 실제로 개체는 단어 중간에서 시작하지 않습니다. 연속 서브워드에 `-100`을 사용하면 첫 번째 서브워드 예측에만 학습을 올바르게 집중시킵니다.

</details>

### 연습 문제 3: 파인튜닝 전략 선택

아래 각 시나리오에 대해 가장 적합한 파인튜닝 전략을 선택하고 근거를 제시하세요:

1. 레이블이 지정된 영화 리뷰 10만 개와 A100 GPU 4개가 있는 경우
2. 16GB RAM이 있는 노트북에서 고객 지원을 위해 70억 파라미터 LLM을 적응시켜야 하는 경우
3. 전문화된 의료 분류 태스크에 레이블이 지정된 예제가 50개만 있는 경우
4. 선호도 데이터(선택/거부 쌍)로 명령 따르기를 파인튜닝해야 하는 경우

<details>
<summary>정답 보기</summary>

**시나리오 1: 전체 파인튜닝(Full Fine-tuning)**

- 10만 개의 샘플은 과적합 없이 모든 파라미터를 업데이트하기에 충분합니다.
- A100 GPU 4개(각 40GB VRAM)로 전체 모델을 메모리에 맞출 수 있습니다.
- 데이터가 풍부할 때 전체 파인튜닝이 최대 유연성과 일반적으로 최고의 성능을 제공합니다.

```python
# 전체 파인튜닝 설정
args = TrainingArguments(
    per_device_train_batch_size=32,  # 4개 GPU를 활용하는 큰 배치
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,  # 속도를 위한 혼합 정밀도
)
```

**시나리오 2: QLoRA (Quantized LoRA)**

- fp16의 70억 파라미터 모델은 가중치만으로도 ~14GB가 필요해 노트북에 겨우 맞습니다.
- 4비트 양자화(quantization)는 ~3.5GB로 줄여 활성화와 LoRA 어댑터를 위한 공간을 남깁니다.
- LoRA는 추가적인 학습 가능 파라미터의 ~0.3%만 추가합니다.

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

**시나리오 3: Prompt Tuning 또는 동결 모델로 퓨샷 학습**

- 50개의 예제는 신뢰할 수 있는 전체 파인튜닝이나 LoRA 파인튜닝에 너무 적습니다 (과적합 위험 높음).
- 프롬프트 튜닝은 소프트 프롬프트 토큰만 학습하여 (~파라미터의 1%) 과적합 가능성을 크게 줄입니다.
- 대안: 그레이디언트 업데이트 없이 퓨샷 인-컨텍스트 러닝(in-context learning)으로 사전학습된 모델 사용.

```python
from peft import PromptTuningConfig, TaskType, get_peft_model

config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,  # 적은 수의 학습 가능한 프롬프트 토큰
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the following medical text: "
)
model = get_peft_model(frozen_model, config)
```

**시나리오 4: SFT + DPO (Direct Preference Optimization)**

- 명령 따르기는 모델에게 어떤 출력이 선호되는지 가르쳐야 합니다.
- 1단계: SFT(Supervised Fine-Tuning)로 선택된 응답에서 대상 행동 학습
- 2단계: DPO는 보상 모델 없이 (선택, 거부) 쌍을 사용하여 선호도 정렬을 직접 최적화합니다.

```python
from trl import SFTTrainer, DPOTrainer

# 1단계: 지도 파인튜닝 (SFT)
sft_trainer = SFTTrainer(model=model, train_dataset=instruction_dataset)
sft_trainer.train()

# 2단계: 선호도 정렬을 위한 DPO
dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model_copy,  # 기준 모델 (동결)
    train_dataset=preference_dataset,  # {prompt, chosen, rejected}
    beta=0.1,
)
dpo_trainer.train()
```

</details>

## 다음 단계

[PEFT (Parameter-Efficient Fine-Tuning)](./08_PEFT_and_QLoRA.md)에서 파라미터 효율적 파인튜닝 기법을 학습합니다.

# 06. HuggingFace Basics

## Learning Objectives

- Understanding the Transformers library
- Using Pipeline API
- Loading tokenizers and models
- Performing various tasks

---

## Theory & Principles

The HuggingFace Transformers library is, at its core, a **standardized abstraction layer** over hundreds of different model architectures. Behind the convenient `AutoModel.from_pretrained("bert-base")` call sits a careful design that decouples three orthogonal concerns: **tokenization** (text → tensors), **architecture** (Python class implementing the model), and **weights** (trained parameters stored on the Hub). Understanding how these pieces fit together — and the conventions that make them interoperable — is what lets you go from a research paper to a working model in three lines.

This section covers:

- **(A) The Auto-class abstraction** — how `AutoModel`, `AutoTokenizer`, `AutoConfig` route to the right concrete class based on a model's `config.json`.
- **(B) Tokenizer internals** — fast (Rust) vs slow (Python), the encoding pipeline, and what `BatchEncoding` actually contains.
- **(C) Model loading and weight initialization** — `state_dict`, weight tying, partial loads, the `device_map` and quantization options.
- **(D) The Pipeline API** — how `pipeline("sentiment-analysis")` composes preprocessing, model inference, and postprocessing.
- **(E) The Datasets / Trainer pipeline** — `datasets` for streaming and caching, `Trainer` for the standard training loop.
- **(F) The Hub** — model cards, versioning, and the `from_pretrained` resolution algorithm.

### A. The Auto-Class Abstraction

`AutoTokenizer.from_pretrained("bert-base-uncased")` does roughly the following:

```
1. Resolve the model identifier → URL on the HF Hub.
2. Download config.json — contains "model_type": "bert" and architecture hyperparameters.
3. Look up the model_type in a registry mapping bert → BertTokenizer (or BertTokenizerFast).
4. Instantiate that concrete class with the downloaded config and vocab files.
```

Same algorithm for `AutoModel`, `AutoConfig`, `AutoFeatureExtractor`, `AutoProcessor`. The benefit is **decoupling user code from model architecture** — switching from `bert-base-uncased` to `roberta-base` or `microsoft/deberta-v3-base` requires only changing one string. The Auto-classes hide ~20 different concrete classes per modality.

Task-specific Auto-classes (`AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`) wrap the base encoder with the appropriate task head. They handle the head's randomly-initialized weights when no task-specific checkpoint exists.

### B. Tokenizer Internals

**B.1 Fast vs slow.** Python tokenizers (slow) implement BPE/WordPiece in pure Python; fast tokenizers (`tokenizers` library) implement the same algorithms in Rust with parallelism. For BERT-base on a 1M-sentence corpus, fast tokenizers are ~50-100× faster — important because tokenization is often a training bottleneck.

**B.2 The encoding pipeline.** Calling `tokenizer(text)` runs:

```
text → normalize → pre-tokenize → tokenize (BPE/WordPiece/Unigram) → post-process (add [CLS], [SEP]) → return BatchEncoding
```

Each stage is configurable. Normalization handles Unicode (NFC/NFD), accent stripping, lowercasing. Pre-tokenization splits on whitespace and punctuation but keeps a record of original character offsets. Post-processing adds special tokens and builds the attention mask.

**B.3 BatchEncoding contents.**

```python
out = tokenizer(["Hello world", "Hi"], padding=True, return_tensors="pt")
# out["input_ids"]:        tensor of token IDs, shape (B, L)
# out["attention_mask"]:   1 for real tokens, 0 for padding, shape (B, L)
# out["token_type_ids"]:   segment IDs (0/1) for sentence-pair models
# out.offset_mapping:      char offsets back to the original string (only with fast tokenizers)
```

The `offset_mapping` is crucial for token-level tasks like NER: it lets you map predicted token labels back to character spans in the original text — a step that is impossible to do correctly without offset tracking.

### C. Model Loading and Weight Initialization

**C.1 What `from_pretrained` does.**

```
1. Download config.json + tokenizer files + safetensors/pytorch_model.bin.
2. Instantiate the architecture from config (random weights).
3. Load the downloaded state_dict, mapping each weight by name.
4. Report missing keys (architecture expected weights not in the checkpoint)
   and unexpected keys (checkpoint weights not used by this architecture).
```

Missing keys are auto-initialized — for `BertForSequenceClassification`, the classification head weights are random because the pre-trained `bert-base-uncased` checkpoint has no such head. The library logs a warning so you know which weights are not pre-trained.

**C.2 Weight tying.** In LMs, the input embedding `E ∈ ℝ^{V × d}` and the output projection `W ∈ ℝ^{d × V}` are usually tied (`W = E^T`). This halves the parameter count on the embedding/output and provides a regularization benefit — both projections must satisfy each role simultaneously. The HF library handles tying automatically during loading.

**C.3 `device_map` and quantization.** Modern LLMs do not fit on a single GPU at full precision. `from_pretrained(..., device_map="auto", load_in_4bit=True)` triggers:

- Architecture instantiation on the meta device (no allocation).
- Weight-by-weight loading: each tensor is loaded from disk, optionally quantized to NF4/INT8, and placed on the GPU/CPU/disk according to a memory budget.
- The result: a model that runs end-to-end with each parameter on the device decided automatically.

This is a significant abstraction — without it, you would write hundreds of lines of `accelerate` glue.

### D. The Pipeline API

```python
nlp = pipeline("sentiment-analysis")
nlp("I love this!")  # → [{"label": "POSITIVE", "score": 0.99}]
```

A pipeline is a thin wrapper that composes:

```
1. Tokenizer (from the model's default).
2. Model (forward pass with no_grad).
3. Postprocessor: argmax for classification, decoding for generation, span extraction for QA, etc.
```

Pipelines are convenience wrappers — they lock you into default tokenization, default decoding strategies, and Python-level batching. For production, drop down to direct tokenizer + model calls; for prototyping or one-off inference, pipelines are perfect.

### E. Datasets and Trainer

**E.1 `datasets`.** A unified API over local files, the Hub, and custom loaders, with three properties that matter for NLP:

- **Memory-mapped storage** (Apache Arrow): a 100 GB dataset is queryable without fitting in RAM.
- **Streaming**: `load_dataset(..., streaming=True)` returns an iterator that downloads on demand — necessary for datasets like C4 (700 GB+).
- **`map()` with caching**: `dataset.map(tokenize_fn)` runs once and caches the tokenized output to disk. Re-running picks up the cache.

**E.2 `Trainer`.** Implements the standard training loop:

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

Built-in: gradient accumulation, mixed-precision training, distributed training, learning-rate scheduling, checkpointing, integration with W&B / TensorBoard. Replacing `Trainer` with hand-written PyTorch is fine for research; for any standard fine-tuning task, `Trainer` saves hundreds of lines.

### F. The Hub

Each model on `huggingface.co/<org>/<model>` is a Git repository containing `config.json`, tokenizer files, weights, and a `README.md` (model card). Model versioning uses Git revisions, so `from_pretrained(..., revision="v1.0")` pins to a specific commit.

The Hub is the dependency mechanism for modern NLP — when a paper releases code, the standard pattern is to also release the trained checkpoint to the Hub, so any reader can reproduce the result with `from_pretrained`. This is structurally why the field's reproducibility is much better than ML overall.

### From Theory to the Functions Below

- §1 (ecosystem) — overview of the §A Auto-class layer and the §F Hub model.
- §2 (Pipeline API) — the §D wrapper for common tasks.
- §3 (tokenizers) — direct use of §B's `AutoTokenizer`, with offset_mapping examples.
- §4 (model loading) — §C's `from_pretrained` mechanics, with `AutoModel` and task-specific heads.
- §5 (Datasets library) — §E.1's load/map/cache patterns.
- §6 (Trainer API) — §E.2's standard fine-tuning loop.
- §7 (saving/loading) — `save_pretrained` / `push_to_hub` (§F).
- §8 (sentiment classification) — end-to-end project tying §B, §C, §E together.

---

## 1. HuggingFace Ecosystem

### Main Components

```
HuggingFace
├── Transformers   # Model library
├── Datasets       # Datasets
├── Tokenizers     # Tokenizers
├── Hub            # Model/data repository
├── Accelerate     # Distributed training
└── Evaluate       # Evaluation metrics
```

### Installation

```bash
pip install transformers datasets tokenizers accelerate evaluate
```

---

## 2. Pipeline API

### Simplest Usage

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch processing
results = classifier([
    "I love this movie!",
    "This is terrible."
])
```

### Supported Tasks

| Task | Pipeline Name | Description |
|------|--------------|-------------|
| Sentiment Analysis | sentiment-analysis | Positive/negative classification |
| Text Classification | text-classification | General classification |
| NER | ner | Named entity recognition |
| QA | question-answering | Question answering |
| Summarization | summarization | Text summarization |
| Translation | translation | Language translation |
| Text Generation | text-generation | Sentence generation |
| Fill-Mask | fill-mask | Mask prediction |
| Zero-shot | zero-shot-classification | Classification without labels |

### Various Pipeline Examples

```python
# Question answering
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France."
)
# {'answer': 'Paris', 'score': 0.99, 'start': 0, 'end': 5}

# Summarization
summarizer = pipeline("summarization")
text = "Very long article text here..."
summary = summarizer(text, max_length=50, min_length=10)

# Translation
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# NER
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is John and I work at Google in New York")
# [{'entity_group': 'PER', 'word': 'John', ...},
#  {'entity_group': 'ORG', 'word': 'Google', ...},
#  {'entity_group': 'LOC', 'word': 'New York', ...}]

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "I want to go to the beach",
    candidate_labels=["travel", "cooking", "technology"]
)
# {'labels': ['travel', 'cooking', 'technology'], 'scores': [0.95, 0.03, 0.02]}
```

### Specifying Models

```python
# Korean model
classifier = pipeline(
    "sentiment-analysis",
    model="beomi/kcbert-base"
)

# Multilingual model
qa = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2"
)
```

---

## 3. Tokenizers

### AutoTokenizer

```python
from transformers import AutoTokenizer

# Automatically load appropriate tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encoding
text = "Hello, how are you?"
encoded = tokenizer(text)
print(encoded)
# {'input_ids': [101, 7592, ...], 'attention_mask': [1, 1, ...], ...}

# Return as tensors
encoded = tokenizer(text, return_tensors='pt')
```

### Key Parameters

```python
encoded = tokenizer(
    text,
    padding=True,              # Add padding
    truncation=True,           # Truncate to max length
    max_length=128,            # Maximum length
    return_tensors='pt',       # PyTorch tensors
    return_attention_mask=True,
    return_token_type_ids=True
)
```

### Batch Encoding

```python
texts = ["Hello world", "How are you?", "I'm fine"]

# Dynamic padding
encoded = tokenizer(
    texts,
    padding=True,     # Pad to longest sequence
    truncation=True,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)  # (3, max_len)
```

### Decoding

```python
# Decoding
decoded = tokenizer.decode(encoded['input_ids'][0])
print(decoded)  # "[CLS] hello world [SEP]"

# Remove special tokens
decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print(decoded)  # "hello world"
```

### Token Inspection

```python
# Token list
tokens = tokenizer.tokenize("Hello, how are you?")
print(tokens)  # ['hello', ',', 'how', 'are', 'you', '?']

# Tokens → IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

# IDs → Tokens
tokens = tokenizer.convert_ids_to_tokens(ids)
```

---

## 4. Model Loading

### AutoModel

```python
from transformers import AutoModel, AutoModelForSequenceClassification

# Base model (output: hidden states)
model = AutoModel.from_pretrained("bert-base-uncased")

# Classification model (output: logits)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

### Task-specific AutoModels

```python
from transformers import (
    AutoModelForSequenceClassification,  # Sequence classification
    AutoModelForTokenClassification,      # Token classification (NER)
    AutoModelForQuestionAnswering,        # QA
    AutoModelForCausalLM,                 # GPT-style generation
    AutoModelForSeq2SeqLM,                # Encoder-decoder (translation, summarization)
    AutoModelForMaskedLM                  # BERT-style MLM
)
```

### Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Encoding
inputs = tokenizer("I love this movie!", return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Prediction
predictions = torch.softmax(logits, dim=-1)
predicted_class = predictions.argmax().item()
print(f"Class: {predicted_class}, Confidence: {predictions[0][predicted_class]:.4f}")
```

---

## 5. Datasets Library

### Loading Datasets

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# Specify split
train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test[:1000]")  # First 1000

# Check sample
print(train_data[0])
# {'text': '...', 'label': 1}
```

### Data Preprocessing

```python
def preprocess(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

# Apply map
tokenized_dataset = dataset.map(preprocess, batched=True)

# Remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# Set PyTorch format
tokenized_dataset.set_format('torch')
```

### Creating DataLoader

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

### Basic Training

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Data
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training configuration
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

# Training
trainer.train()

# Evaluation
results = trainer.evaluate()
print(results)
```

### Custom Metrics

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

## 7. Model Saving/Loading

### Local Save

```python
# Save
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

### Upload to Hub

```python
# Login
from huggingface_hub import login
login(token="your_token")

# Upload
model.push_to_hub("my-username/my-model")
tokenizer.push_to_hub("my-username/my-model")

# Or with Trainer
trainer.push_to_hub("my-model")
```

---

## 8. Practical Example: Sentiment Classification

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

# 1. Load data
dataset = load_dataset("imdb")

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 4. Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. Training configuration
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

# 7. Training
trainer.train()

# 8. Inference
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

## Summary

### Key Classes

| Class | Purpose |
|-------|---------|
| pipeline | Quick inference |
| AutoTokenizer | Automatic tokenizer loading |
| AutoModel* | Automatic model loading |
| Trainer | Training loop automation |
| TrainingArguments | Training configuration |

### Key Code

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Quick inference
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# Custom inference
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs)
```

---

## Exercises

### Exercise 1: Pipeline Task Exploration

Using the HuggingFace `pipeline` API, run three different NLP tasks: NER on a sentence containing person names and organizations, zero-shot classification on a news headline with custom labels, and question answering. For each result, explain what the model outputs and what the scores represent.

<details>
<summary>Show Answer</summary>

```python
from transformers import pipeline

# Task 1: Named Entity Recognition (NER)
ner = pipeline("ner", grouped_entities=True)
sentence = "Elon Musk founded SpaceX in California and Tesla in Texas."
ner_result = ner(sentence)
print("NER Results:")
for entity in ner_result:
    print(f"  '{entity['word']}' → {entity['entity_group']} (score: {entity['score']:.3f})")
# 'Elon Musk' → PER (score: 0.998)  — Person
# 'SpaceX'    → ORG (score: 0.995)  — Organization
# 'California'→ LOC (score: 0.992)  — Location
# 'Tesla'     → ORG (score: 0.989)  — Organization
# 'Texas'     → LOC (score: 0.991)  — Location
# Scores: confidence of the entity label assignment

# Task 2: Zero-shot Classification
zero_shot = pipeline("zero-shot-classification")
headline = "Scientists discover new exoplanet with potential for liquid water"
result = zero_shot(
    headline,
    candidate_labels=["astronomy", "biology", "technology", "sports", "politics"]
)
print("\nZero-shot Classification:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.3f}")
# astronomy: 0.812
# biology: 0.124
# technology: 0.047
# Scores: probability that the text belongs to each label
# (sums to ~1.0, no fine-tuning needed!)

# Task 3: Question Answering
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
print("\nQuestion Answering:")
for q in questions:
    result = qa(question=q, context=context)
    print(f"  Q: {q}")
    print(f"  A: '{result['answer']}' (score: {result['score']:.3f})")
    print(f"     Character span: [{result['start']}, {result['end']}]")
# A: 'Thomas Wolf and Lysandre Debut' (score: 0.97)
# Score: confidence that this text span is the correct answer
# start/end: character positions in the context string
```

**Score interpretations by task**:
- **NER**: Per-entity confidence that the label (PER, ORG, LOC) is correct. Values near 1.0 indicate high confidence.
- **Zero-shot**: Softmax probability distribution over candidate labels. The model never seen these specific labels during training — it uses natural language entailment to rank them.
- **QA**: Probability that the extracted span is the correct answer. Low scores (<0.5) suggest the answer may not be in the context.

</details>

### Exercise 2: Custom Dataset with Trainer API

The `Trainer` API requires datasets in a specific format. Write a function to convert a simple list of (text, label) tuples into a HuggingFace `Dataset` object that can be used with the `Trainer`. Then show how to add a custom metric (F1 score) to the training evaluation.

<details>
<summary>Show Answer</summary>

```python
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import evaluate
import numpy as np

# Sample data
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
    """Convert (text, label) tuples to HuggingFace Dataset."""
    texts, labels = zip(*data)

    # Tokenize all texts at once
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )

    # Build dictionary format required by Dataset
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': list(labels),  # Trainer expects 'labels' key (not 'label')
    }

    return Dataset.from_dict(dataset_dict)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

train_dataset = create_dataset(train_data, tokenizer)
test_dataset = create_dataset(test_data, tokenizer)

print("Dataset structure:", train_dataset)
print("Sample:", train_dataset[0])

# Custom metrics: accuracy + F1 score
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

# Model and training setup
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

# trainer.train()  # Run to actually train
```

**Key format requirements for `Trainer`**:
- The dataset must contain a `labels` key (not `label`)
- Input tensors must be PyTorch-compatible (use `set_format('torch')` for `map`-based datasets)
- `compute_metrics` receives `(logits, labels)` as a named tuple `EvalPrediction`

</details>

### Exercise 3: Model Selection for Tasks

For each NLP task below, specify the correct `AutoModel` class to use and explain why that specific class is appropriate:

1. Extracting sentence embeddings for semantic similarity
2. Named Entity Recognition
3. Machine translation (English to French)
4. Filling in blanks in a sentence (masked token prediction)
5. Open-ended text generation

<details>
<summary>Show Answer</summary>

```python
from transformers import (
    AutoModel,                           # Task 1: Base model
    AutoModelForTokenClassification,     # Task 2: NER
    AutoModelForSeq2SeqLM,               # Task 3: Translation
    AutoModelForMaskedLM,                # Task 4: Fill-mask
    AutoModelForCausalLM,                # Task 5: Generation
)

# Task 1: Sentence embeddings for semantic similarity
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Why: We need raw hidden states to compute embeddings (mean pooling of last_hidden_state)
# No task-specific head needed
tokenizer_1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inputs = tokenizer_1("Hello world", return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling → (1, 384)

# Task 2: Named Entity Recognition
ner_model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
# Why: NER requires a prediction FOR EACH TOKEN (not just [CLS])
# TokenClassification adds a linear head per token: (batch, seq, num_labels)

# Task 3: Machine Translation
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr"
)
# Why: Translation requires an encoder (understand source) + decoder (generate target)
# Seq2SeqLM handles cross-attention between encoder output and decoder

# Task 4: Fill-mask (MLM prediction)
mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# Why: MaskedLM adds a prediction head over the entire vocabulary at [MASK] positions
# Output: (batch, seq, vocab_size) — predict token at each position

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
print("Top 5 predictions:", predictions)  # ['paris', 'lyon', ...]

# Task 5: Text Generation
gpt = AutoModelForCausalLM.from_pretrained("gpt2")
# Why: CausalLM (decoder-only) with causal mask, trained on next-token prediction
# generate() method supports various decoding strategies
```

**Why the right model class matters**:

| Class | Output head | Training objective |
|-------|-------------|-------------------|
| `AutoModel` | None (raw hidden states) | — |
| `AutoModelForTokenClassification` | Linear per token | Cross-entropy per token |
| `AutoModelForSeq2SeqLM` | Decoder with cross-attention | Seq-to-seq cross-entropy |
| `AutoModelForMaskedLM` | Linear over vocab at mask positions | MLM cross-entropy |
| `AutoModelForCausalLM` | Linear over vocab at all positions | Next-token cross-entropy |

Using the wrong class would either output the wrong shape or fail to load the correct pre-trained weights for the task-specific head.

</details>

## Next Steps

Learn fine-tuning techniques for various tasks in [07_Fine_Tuning.md](./07_Fine_Tuning.md).

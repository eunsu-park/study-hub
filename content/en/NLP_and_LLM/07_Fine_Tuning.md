# 07. Fine-Tuning

## Learning Objectives

- Understanding fine-tuning strategies
- Fine-tuning for various tasks
- Efficient fine-tuning techniques (LoRA, QLoRA)
- Practical fine-tuning pipelines

---

## Theory & Principles

Fine-tuning takes a model that already knows *general* language and bends it toward a *specific* task or domain. The pre-training did the expensive work — learning syntax, basic semantics, world knowledge — using terabytes of unlabeled text. Fine-tuning leverages that representation with a small labeled dataset, often a few thousand examples instead of billions. The whole game is **how to update the weights enough to learn the task without destroying the general knowledge that made the model useful in the first place.**

This section covers:

- **(A) Transfer learning** — why pre-train then fine-tune dominates train-from-scratch.
- **(B) Full fine-tuning vs feature extraction vs head-only** — the spectrum of how many parameters you update.
- **(C) Catastrophic forgetting** — the failure mode that drives every "be gentle" recipe.
- **(D) Learning rate schedules and warm-up** — why fine-tuning uses `2e-5` and a linear schedule, not `1e-3` cosine.
- **(E) Loss functions per task** — cross-entropy variants for classification, NER (per-token), QA (start/end span), seq2seq.
- **(F) Regularization tricks** — early stopping, dropout, weight decay, label smoothing, layer-wise learning rate decay.

### A. Transfer Learning: Why It Dominates

Training a Transformer from scratch on, say, a 10K-sentence sentiment dataset would underfit catastrophically — the model has 100M+ parameters and the dataset has 10⁴ examples. The number of effective parameters in random initialization vastly exceeds what 10⁴ examples can constrain.

Pre-training inverts the ratio: 100M parameters constrained by 10¹¹+ tokens (BERT) or 10¹²+ (LLaMA). The pre-trained weights already encode general language patterns. The fine-tuning step asks: "given this almost-correct-for-everything starting point, nudge it toward this task." The effective hypothesis space shrinks dramatically — only models close to the pre-trained weights are reachable — and that bias is exactly what data-poor fine-tuning needs.

Empirically: a fine-tuned BERT beats a from-scratch LSTM by 10+ F1 points on most NLP tasks, with 100× less labeled data.

### B. The Update-Spectrum: Head-Only → Feature → Full → PEFT

There is a continuum of how aggressively to update the pre-trained weights:

| Strategy | Updated parameters | Cost | When to use |
|----------|-------------------|------|-------------|
| Head-only (linear probe) | head only | smallest | tiny data, prove the encoder already separates classes |
| Last few layers | top `k` layers + head | small | small data, encoder mostly already works |
| Full fine-tuning | all parameters | large | medium data, target task differs from pre-training |
| PEFT (LoRA, etc.) | injected adapters only (~0.1-1%) | small | large model, want ~full FT quality at fraction of cost |

The rule of thumb: **the more your task differs from pre-training, the more parameters you should update.** Sentiment classification (close to general English LM) does well head-only; medical NER on rare entities (far from general English) needs full fine-tuning.

### C. Catastrophic Forgetting

When you continue training a pre-trained model on a new task with vanilla SGD at the original learning rate, the model rapidly fits the new task — and forgets everything it knew. This is **catastrophic forgetting**: the gradient updates from the new task overwrite weight patterns that encoded general knowledge.

Symptoms: training accuracy on the new task climbs to 99%, but if you evaluate on a held-out language modeling perplexity or on a related task, performance crashes.

Why it happens: the pre-trained model sits at a local minimum of the LM loss. The fine-tuning loss has a *different* minimum. With a large learning rate, the model rapidly walks toward the new minimum, leaving the LM minimum behind. The "general knowledge" was implicit in the position in weight space; once you leave that neighborhood, it is gone.

Mitigations:

- **Small learning rate** (§D) — keep updates small enough to stay in the neighborhood.
- **Few epochs** — don't have time to drift far.
- **Gradual unfreezing** (ULMFiT) — fine-tune the top layer first, then progressively unfreeze lower layers.
- **PEFT methods** (LoRA) — leave the base weights frozen entirely, learn small additions.
- **Replay** / **multi-task** — interleave the fine-tuning task with the original LM objective.

### D. Learning Rate Schedules and Warm-Up

A typical BERT fine-tuning recipe:

```
optimizer: AdamW(lr=2e-5, weight_decay=0.01)
schedule: linear warmup over 10% of steps, then linear decay to 0
epochs: 2-4
batch size: 16-32
```

**Why `2e-5` instead of `1e-3`?** From C: large updates cause forgetting. The pre-trained weights have small per-element magnitudes (LayerNorm-stabilized), so even small relative updates are enough to learn a task.

**Why warm-up?** AdamW's adaptive learning rate uses a running estimate of the gradient's second moment. At step 0, this estimate is the bias-corrected initial value (close to zero divided by close to zero), which can produce wildly large effective learning rates. Warm-up linearly raises the LR from 0 to peak over the first ~10% of steps, giving Adam time to build a good second-moment estimate before applying full updates.

**Why linear decay?** Empirically the best simple schedule. The model needs aggressive updates early, fine-grained refinement late. Cosine decay also works.

### E. Loss Functions per Task

The encoder is the same; the head and loss change:

**E.1 Sequence classification.** Linear `[CLS]` → softmax over `C` classes; cross-entropy `−Σ y log p`.

**E.2 Token classification (NER).** Linear at *every* token position → softmax over tag set; cross-entropy summed over real (non-pad, non-special) tokens. Subword tokens of the same word usually share a label, with the convention of computing loss only at the first subword.

**E.3 Extractive QA (SQuAD-style).** Two linear heads at every token: one predicts probability that this token is the *start* of the answer span, the other the *end*. Loss is the sum of the two cross-entropies. At inference, the predicted span is `argmax_{i ≤ j}  p_start(i) · p_end(j)` subject to length constraints.

**E.4 Seq2seq (T5, BART).** Encoder-decoder; loss is cross-entropy on the decoder's predicted tokens against the target sequence (teacher forcing). Same as autoregressive LM loss but conditioned on the encoded source.

**E.5 Causal LM fine-tuning (instruction tuning).** Model is GPT-like; loss is cross-entropy on the response tokens only (mask out the prompt tokens to avoid wasting capacity learning to predict the user's input).

### F. Regularization

A model with 100M parameters fine-tuned on 10K examples will overfit unless regularized.

- **Early stopping**: monitor a held-out set, stop when it plateaus or degrades.
- **Dropout**: usually `0.1` in attention and FFN. Already baked into pre-trained checkpoints; raising it can help small datasets.
- **Weight decay**: `0.01` in AdamW. Penalizes drift from zero, indirectly keeping weights close to the pre-trained values (which are themselves near zero in magnitude).
- **Label smoothing**: replace one-hot targets with `(1−ε)·onehot + ε/C·uniform`. Prevents over-confidence, improves calibration. Common `ε = 0.1`.
- **Layer-wise LR decay (ULMFiT)**: lower layers get smaller LR than upper layers, since lower layers encode more general features that need less updating.
- **Mixout**: dropout-like, but masked positions revert to the pre-trained value rather than zero. Specifically designed to combat catastrophic forgetting.

### From Theory to the Functions Below

- §1 (overview) — frames the §A transfer learning thesis.
- §2 (text classification FT) — implements §E.1 with `BertForSequenceClassification`, using §D's learning-rate recipe.
- §3 (NER FT) — implements §E.2 with token-level alignment for subwords.
- §4 (QA FT) — implements §E.3's start/end span loss on SQuAD.
- §5 (PEFT) — points to the §B right column, full lesson in 08.
- §6 (conversational FT) — implements §E.5 instruction tuning.
- §7 (training optimization) — applies §D schedules, mixed precision, gradient accumulation.
- §8 (complete example) — wires §C-§F into a runnable end-to-end pipeline.

---

## 1. Fine-Tuning Overview

### Transfer Learning Paradigm

```
Pre-training
    │  Learn general language understanding from large-scale text
    ▼
Fine-tuning
    │  Adapt model to specific task data
    ▼
Task Performance
```

### Fine-Tuning Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Full Fine-tuning | Update all parameters | Sufficient data, compute |
| Feature Extraction | Train classifier only | Limited data |
| LoRA | Low-rank adapters | Efficient training |
| Prompt Tuning | Train prompts only | Very limited data |

---

## 2. Text Classification Fine-Tuning

### Basic Pipeline

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# Load data
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

tokenized = dataset.map(tokenize, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training configuration
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

### Multi-Label Classification

```python
from transformers import AutoModelForSequenceClassification
import torch

# Model for multi-label
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)

# Automatically uses BCEWithLogitsLoss

# Label format: [1, 0, 1, 0, 1] (multi-label)
```

---

## 3. Token Classification (NER) Fine-Tuning

### NER Data Format

```python
from datasets import load_dataset

# CoNLL-2003 NER dataset
dataset = load_dataset("conll2003")

# Sample
print(dataset['train'][0])
# {'tokens': ['EU', 'rejects', 'German', 'call', ...],
#  'ner_tags': [3, 0, 7, 0, ...]}

# Labels
label_names = dataset['train'].features['ner_tags'].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### Token Alignment

```python
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True  # Already tokenized input
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # First token
            else:
                label_ids.append(-100)  # Ignore subwords
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized['labels'] = labels
    return tokenized
```

### NER Fine-Tuning

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_names)
)

# seqeval metric
import evaluate
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # Extract only actual labels
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

## 4. Question Answering (QA) Fine-Tuning

### SQuAD Data

```python
dataset = load_dataset("squad")

print(dataset['train'][0])
# {'id': '...', 'title': 'University_of_Notre_Dame',
#  'context': 'Architecturally, the school has...',
#  'question': 'To whom did the Virgin Mary appear in 1858?',
#  'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}
```

### QA Preprocessing

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

            # Find token positions
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

### QA Model

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Output: start_logits, end_logits
```

---

## 5. Efficient Fine-Tuning (PEFT)

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    r=8,                      # Rank
    lora_alpha=32,            # Scaling
    target_modules=["query", "value"],  # Target modules
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# Apply LoRA to model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
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

## 6. Conversational Model Fine-Tuning

### Instruction Tuning Data Format

```python
# Alpaca format
{
    "instruction": "Summarize the following text.",
    "input": "Long article text here...",
    "output": "Summary of the article."
}

# ChatML format
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

# Preference data
# {'prompt': '...', 'chosen': '...', 'rejected': '...'}

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # Reference model
    train_dataset=dataset,
    beta=0.1,
    args=TrainingArguments(...),
)

trainer.train()
```

---

## 7. Training Optimization

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```

### Mixed Precision

```python
args = TrainingArguments(
    ...,
    fp16=True,  # or bf16=True
)
```

### Gradient Accumulation

```python
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 4 * 8 = 32
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

## 8. Complete Fine-Tuning Example

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

# 1. Data
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 2. Model + LoRA
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

# 3. Training configuration
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

# 4. Metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. Training
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. Save
model.save_pretrained("./lora_imdb_final")
```

---

## Summary

### Fine-Tuning Selection Guide

| Situation | Recommended Method |
|-----------|-------------------|
| Sufficient data + GPU | Full Fine-tuning |
| Limited GPU memory | LoRA / QLoRA |
| Very limited data | Prompt Tuning |
| LLM alignment | SFT + DPO/RLHF |

### Key Code

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

## Exercises

### Exercise 1: LoRA Parameter Count Analysis

Given a BERT-base model (12 layers, d_model=768, num_heads=12) with LoRA applied to the query and value projection matrices, calculate: (a) the number of trainable parameters with rank r=8, (b) the percentage of total parameters being trained, and (c) how this compares to full fine-tuning.

<details>
<summary>Show Answer</summary>

```python
# BERT-base architecture parameters
num_layers = 12
d_model = 768
d_k = d_model // 12  # = 64, dimension per head
num_heads = 12

# Query and Value projection dimensions
# W_q: (d_model, d_model) = (768, 768)
# W_v: (d_model, d_model) = (768, 768)

# Total BERT-base parameters (approximate)
# Embeddings: vocab_size * d_model ≈ 30522 * 768 = 23,440,896
# Per layer: attention (4 * d_model^2) + FFN (2 * d_model * d_ff) + norms
# d_ff = 3072 for BERT-base
d_ff = 3072
vocab_size = 30522

embeddings = vocab_size * d_model + 512 * d_model + 2 * d_model  # token + pos + segment
per_layer = (4 * d_model**2) + (2 * d_model * d_ff) + (4 * d_model)  # attn + FFN + norms
pooler = d_model * d_model + d_model

total_bert = embeddings + (num_layers * per_layer) + pooler
print(f"Total BERT-base parameters: {total_bert:,}")
# ≈ 109,482,240 (110M)

# LoRA parameters for rank r=8
r = 8
lora_r = r

# For each LoRA layer applied to W_q and W_v:
# A matrix: (d_model, r) — maps d_model → r
# B matrix: (r, d_model) — maps r → d_model
lora_params_per_matrix = d_model * r + r * d_model  # A + B
lora_targets = 2  # query AND value

# Applied to all 12 layers
total_lora_params = num_layers * lora_targets * lora_params_per_matrix
print(f"LoRA parameters (r={r}): {total_lora_params:,}")
# = 12 * 2 * (768*8 + 8*768) = 12 * 2 * 12288 = 294,912

percentage = total_lora_params / total_bert * 100
print(f"Trainable percentage: {percentage:.3f}%")
# ≈ 0.27%

# Compare
print(f"\nFull fine-tuning: {total_bert:,} parameters (100%)")
print(f"LoRA fine-tuning: {total_lora_params:,} parameters ({percentage:.2f}%)")
print(f"Reduction: {total_bert / total_lora_params:.0f}x fewer trainable parameters")
# ~371x fewer parameters

# Verify with PEFT
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

**Why LoRA works**:

LoRA adds low-rank matrices `A` and `B` such that the weight update `ΔW = BA` (where `B ∈ R^{d×r}` and `A ∈ R^{r×d}`). During forward pass: `h = W₀x + BAx = (W₀ + BA)x`. Only `A` and `B` are updated — `W₀` is frozen.

The hypothesis is that the intrinsic dimensionality of task adaptation is much lower than `d`, so a rank-8 update captures most of the necessary adaptation. This has been validated empirically across many benchmarks.

</details>

### Exercise 2: Token Alignment for NER

One of the trickier aspects of NER fine-tuning is that WordPiece tokenization can split a word into multiple subword tokens, but NER labels are assigned at the word level. Write a function that properly aligns NER labels to subword tokens, and trace through a concrete example.

<details>
<summary>Show Answer</summary>

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example: NER at word level
words = ['Barack', 'Obama', 'was', 'born', 'in', 'Hawaii']
ner_labels = [1, 2, 0, 0, 0, 5]  # B-PER, I-PER, O, O, O, B-LOC
# 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC

def align_labels_with_tokens(words, labels, tokenizer):
    """
    Align word-level NER labels to subword tokens.
    Rules:
    - Special tokens ([CLS], [SEP]) get label -100 (ignored in loss)
    - First subword of a word gets the word's label
    - Subsequent subwords of the same word get label -100 (ignored)
    """
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=False,
        truncation=True,
    )

    word_ids = tokenized.word_ids()  # Maps each token position to word index

    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], [PAD])
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First subword of a new word: use the word's label
            aligned_labels.append(labels[word_id])
        else:
            # Continuation subword: ignore in loss
            aligned_labels.append(-100)

        previous_word_id = word_id

    return tokenized, aligned_labels, word_ids

tokenized, aligned_labels, word_ids = align_labels_with_tokens(
    words, ner_labels, tokenizer
)

# Show the alignment
tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

print(f"{'Token':<15} {'Word ID':<10} {'Label':<15} {'Label Name'}")
print("-" * 50)
for token, word_id, label in zip(tokens, word_ids, aligned_labels):
    label_str = label_names[label] if label != -100 else "IGNORE"
    word_id_str = str(word_id) if word_id is not None else "special"
    print(f"{token:<15} {word_id_str:<10} {str(label):<15} {label_str}")

# Output:
# Token           Word ID    Label           Label Name
# --------------------------------------------------
# [CLS]           special    -100            IGNORE
# barack          0          1               B-PER
# ##ob            0          -100            IGNORE  ← subword
# ##ama           0          -100            IGNORE  ← subword
# was             2          0               O
# born            3          0               O
# in              4          0               O
# hawaii          5          5               B-LOC
# [SEP]           special    -100            IGNORE
```

**Why this matters**: If we naively assigned `B-PER` to all three subwords of "Barack" ("barack", "##ob", "##ama"), the model would try to predict `B-PER` for `##ama` even though in practice, an entity never starts mid-word. Using `-100` for continuation subwords correctly focuses learning on first-subword predictions only.

</details>

### Exercise 3: Fine-Tuning Strategy Selection

For each scenario below, choose the most appropriate fine-tuning strategy and justify your choice:

1. You have 100,000 labeled movie reviews and 4 A100 GPUs
2. You need to adapt a 7B parameter LLM for customer support on a laptop with 16GB RAM
3. You have only 50 labeled examples for a specialized medical classification task
4. You need to fine-tune for instruction following on preference data (chosen/rejected pairs)

<details>
<summary>Show Answer</summary>

**Scenario 1: Full Fine-tuning**

- 100k samples is sufficient to update all parameters without overfitting
- With 4 A100 GPUs (40GB VRAM each), you can fit the full model in memory
- Full fine-tuning gives maximum flexibility and typically best performance when data is abundant

```python
# Full fine-tuning setup
args = TrainingArguments(
    per_device_train_batch_size=32,  # Large batch exploits 4 GPUs
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,  # Mixed precision for speed
)
```

**Scenario 2: QLoRA (Quantized LoRA)**

- 7B parameter model in fp16 would need ~14GB just for weights — barely fits on laptop
- 4-bit quantization reduces to ~3.5GB, leaving room for activations and LoRA adapters
- LoRA adds only ~0.3% additional trainable parameters

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

**Scenario 3: Prompt Tuning or Few-Shot with Frozen Model**

- 50 examples is too few for reliable full or even LoRA fine-tuning (high risk of overfitting)
- Prompt tuning trains only soft prompt tokens (~1% of parameters), dramatically reducing the chance of overfitting
- Alternative: Use the pre-trained model with few-shot in-context learning (no gradient updates at all)

```python
from peft import PromptTuningConfig, TaskType, get_peft_model

config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,  # Few learnable prompt tokens
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the following medical text: "
)
model = get_peft_model(frozen_model, config)
```

**Scenario 4: SFT + DPO (Direct Preference Optimization)**

- Instruction following requires teaching the model what outputs are preferred
- Step 1: SFT on chosen responses to learn the target behavior
- Step 2: DPO uses (chosen, rejected) pairs to directly optimize preference alignment without a reward model

```python
from trl import SFTTrainer, DPOTrainer

# Step 1: Supervised Fine-Tuning
sft_trainer = SFTTrainer(model=model, train_dataset=instruction_dataset)
sft_trainer.train()

# Step 2: DPO for preference alignment
dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model_copy,  # Reference model (frozen)
    train_dataset=preference_dataset,  # {prompt, chosen, rejected}
    beta=0.1,
)
dpo_trainer.train()
```

</details>

## Next Steps

Learn about parameter-efficient fine-tuning in [08_PEFT_and_QLoRA.md](./08_PEFT_and_QLoRA.md).

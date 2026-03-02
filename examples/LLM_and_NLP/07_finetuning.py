"""
07. Fine-tuning Example

Model fine-tuning using HuggingFace Trainer
"""

print("=" * 60)
print("Fine-tuning")
print("=" * 60)


# ============================================
# 1. Basic Fine-tuning (Code Example)
# ============================================
print("\n[1] Basic Fine-tuning")
print("-" * 40)

basic_finetuning = '''
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Load data
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training configuration
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# Train
trainer.train()
'''
print(basic_finetuning)


# ============================================
# 2. LoRA Fine-tuning
# ============================================
print("\n[2] LoRA Fine-tuning")
print("-" * 40)

lora_code = '''
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    r=8,                           # Rank
    lora_alpha=32,                 # Scaling
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
# trainable: 0.27% (approx. 300K / 110M)

# Train with standard Trainer
trainer = Trainer(model=model, args=args, ...)
trainer.train()
'''
print(lora_code)


# ============================================
# 3. QLoRA (Quantization + LoRA)
# ============================================
print("\n[3] QLoRA")
print("-" * 40)

qlora_code = '''
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

# Train
trainer = Trainer(model=model, ...)
'''
print(qlora_code)


# ============================================
# 4. Custom Metrics
# ============================================
print("\n[4] Custom Metrics")
print("-" * 40)

try:
    import evaluate
    import numpy as np

    # Load metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        }

    print("Custom metrics function defined")

    # Test
    mock_pred = (np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]))
    result = compute_metrics(mock_pred)
    print(f"Test result: {result}")

except ImportError:
    print("evaluate not installed (pip install evaluate)")


# ============================================
# 5. NER Fine-tuning
# ============================================
print("\n[5] NER Fine-tuning")
print("-" * 40)

ner_code = '''
from transformers import AutoModelForTokenClassification

# Labels
label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

# Model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_names)
)

# Token alignment (subword handling)
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized
'''
print(ner_code)


# ============================================
# 6. QA Fine-tuning
# ============================================
print("\n[6] QA Fine-tuning")
print("-" * 40)

qa_code = '''
from transformers import AutoModelForQuestionAnswering

# Model
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Preprocessing (find start/end positions)
def prepare_train_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Convert answer character positions to token positions
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        # Answer start/end character position -> token position
        ...

    return tokenized
'''
print(qa_code)


# ============================================
# 7. Training Optimization Tips
# ============================================
print("\n[7] Training Optimization Tips")
print("-" * 40)

optimization_tips = '''
# Gradient Checkpointing (memory savings)
model.gradient_checkpointing_enable()

# Mixed Precision (speed improvement)
args = TrainingArguments(
    ...,
    fp16=True,  # or bf16=True
)

# Gradient Accumulation (effective large batch)
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
)

# DeepSpeed (distributed training)
args = TrainingArguments(
    ...,
    deepspeed="ds_config.json"
)

# Learning Rate Scheduler
args = TrainingArguments(
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)
'''
print(optimization_tips)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Fine-tuning Summary")
print("=" * 60)

summary = """
Fine-tuning Selection Guide:
    - Sufficient GPU: Full Fine-tuning
    - Limited memory: LoRA / QLoRA
    - Very little data: Prompt Tuning

Key Code:
    # Trainer
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    # LoRA
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=8, target_modules=["query", "value"])
    model = get_peft_model(model, config)
"""
print(summary)
